"""Resident, patch-pinned League mechanics warehouse.

This module is intentionally small and dependency-free.  It materializes the
validated fastpack into an in-memory SQLite star schema at MCP startup, then
keeps the connection read-only for the lifetime of the process.  The shape is
Snowflake-like (dimensions, facts, a snapshot receipt, bounded SQL), but it is
local: the serving path never needs a cloud warehouse, a network round trip,
or a request-time rebuild.

The warehouse is a source/query layer, not a game emulator.  Static champion
and item facts are executable when their inputs are closed; effect-bearing
items, runes, buffs, and live transitions are returned as explicit
``unsupported`` state rather than being ignored.  This boundary is what lets
the natural-language oracle remain fast without making a plausible but false
calculation.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "scryglass:league-state-warehouse:v1"
ENGINE_VERSION = "league-state-warehouse-v1.0.0"
MAX_SQL_LENGTH = 50_000
DEFAULT_MAX_ROWS = 200
MAX_MAX_ROWS = 1_000


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _norm(value: Any) -> str:
    decomposed = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(char for char in decomposed if char.isalnum())


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return value


def _mentions(question: str, candidate: str) -> bool:
    parts = re.findall(r"[a-z0-9]+", str(candidate).casefold())
    if not parts:
        return False
    pattern = r"(?<![a-z0-9])" + r"\W+".join(re.escape(part) for part in parts) + r"(?![a-z0-9])"
    return re.search(pattern, question.casefold()) is not None


def _source_url(title: str) -> str:
    from urllib.parse import quote

    return "https://wiki.leagueoflegends.com/en-us/" + quote(
        str(title).replace(" ", "_"), safe="_-\'()"
    )


def _client_root_url(client_patch: str | None) -> str | None:
    if not client_patch:
        return None
    return (
        f"https://raw.communitydragon.org/{client_patch}/"
        "plugins/rcp-be-lol-game-data/global/default/v1/"
    )


def _client_file_url(client_patch: str | None, relative_path: Any) -> str | None:
    if not client_patch or not isinstance(relative_path, str) or not relative_path:
        return _client_root_url(client_patch)
    return f"https://raw.communitydragon.org/{client_patch}/{relative_path.lstrip('/')}"


@dataclass(frozen=True)
class WarehouseSnapshot:
    schema_version: str
    engine_version: str
    patch: str
    client_patch: str | None
    source_hash: str | None
    index_sha256: str | None
    snapshot_sha256: str

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "engine_version": self.engine_version,
            "patch": self.patch,
            "client_patch": self.client_patch,
            "source_hash": self.source_hash,
            "index_sha256": self.index_sha256,
            "snapshot_sha256": self.snapshot_sha256,
        }


class WarehouseQueryError(ValueError):
    """A bounded warehouse query is malformed or attempts to mutate state."""


class StateWarehouse:
    """Materialize one exact packet into a resident read-only query warehouse."""

    def __init__(
        self,
        pack: Mapping[str, Any],
        *,
        oracle: Any | None = None,
        index_path: Path | None = None,
    ) -> None:
        if not isinstance(pack, Mapping):
            raise TypeError("pack must be a mapping")
        self.pack = pack
        self.oracle = oracle
        self.index_path = Path(index_path).resolve() if index_path is not None else None
        self.connection = sqlite3.connect(":memory:")
        self.connection.row_factory = sqlite3.Row
        self._cache: dict[tuple[str, int], dict[str, Any]] = {}
        self._queries = 0
        self._cache_hits = 0
        self._closed = False
        self._create_schema()
        self._populate()
        # The build is complete.  SQLite's connection-level query_only flag
        # prevents accidental writes through the serving object, including
        # from a future SQL tool implementation.
        self.connection.execute("PRAGMA query_only = ON")
        self.snapshot = self._snapshot()

    @property
    def ready(self) -> bool:
        return not self._closed

    @staticmethod
    def _safe_identifier(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", value)

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE warehouse_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE dim_patch (
                patch TEXT PRIMARY KEY,
                client_patch TEXT,
                source TEXT,
                source_hash TEXT,
                index_sha256 TEXT,
                schema_version TEXT NOT NULL
            );
            CREATE TABLE dim_champion (
                champion_key TEXT PRIMARY KEY,
                champion_id INTEGER NOT NULL UNIQUE,
                alias TEXT NOT NULL,
                name TEXT NOT NULL,
                resource_type TEXT,
                status TEXT NOT NULL,
                unresolved_json TEXT NOT NULL,
                source_path TEXT,
                source_sha256 TEXT,
                aliases_json TEXT NOT NULL
            );
            CREATE TABLE fact_champion_stat (
                champion_key TEXT NOT NULL REFERENCES dim_champion(champion_key),
                level INTEGER NOT NULL CHECK (level BETWEEN 1 AND 18),
                max_health REAL,
                attack_damage REAL,
                armor REAL,
                magic_resist REAL,
                health_regen_per_5 REAL,
                max_resource REAL,
                resource_regen_per_5 REAL,
                PRIMARY KEY (champion_key, level)
            );
            CREATE INDEX ix_champion_stat_level ON fact_champion_stat(level, champion_key);
            CREATE TABLE dim_ability (
                champion_key TEXT NOT NULL REFERENCES dim_champion(champion_key),
                ability_key TEXT NOT NULL,
                ability_name TEXT,
                script_name TEXT,
                max_rank INTEGER,
                cost_json TEXT,
                cooldown_json TEXT,
                description TEXT,
                execution_status TEXT,
                formula_semantics_status TEXT,
                source_json TEXT NOT NULL,
                PRIMARY KEY (champion_key, ability_key)
            );
            CREATE INDEX ix_ability_name ON dim_ability(ability_name, champion_key);
            CREATE TABLE fact_ability_value (
                champion_key TEXT NOT NULL,
                ability_key TEXT NOT NULL,
                value_name TEXT NOT NULL,
                client_index INTEGER NOT NULL,
                value REAL,
                values_json TEXT NOT NULL,
                PRIMARY KEY (champion_key, ability_key, value_name, client_index),
                FOREIGN KEY (champion_key, ability_key)
                    REFERENCES dim_ability(champion_key, ability_key)
            );
            CREATE TABLE dim_item (
                item_key TEXT PRIMARY KEY,
                item_id INTEGER NOT NULL UNIQUE,
                name TEXT NOT NULL,
                in_store INTEGER NOT NULL,
                has_passive INTEGER NOT NULL,
                categories_json TEXT NOT NULL,
                description TEXT NOT NULL,
                price_total REAL,
                source_sha256 TEXT
            );
            CREATE INDEX ix_item_name ON dim_item(name);
            CREATE TABLE fact_item_stat (
                item_key TEXT NOT NULL REFERENCES dim_item(item_key),
                stat_key TEXT NOT NULL,
                value REAL NOT NULL,
                is_percent INTEGER NOT NULL,
                label TEXT,
                PRIMARY KEY (item_key, stat_key)
            );
            CREATE TABLE dim_rune (
                rune_key TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                effect_status TEXT NOT NULL,
                source_url TEXT,
                notes TEXT
            );
            CREATE TABLE dim_mode (
                mode_key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                rules_status TEXT NOT NULL
            );
            CREATE TABLE dim_structure (
                structure_key TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                health REAL,
                base_attack_damage REAL,
                attack_step REAL,
                attack_first_second REAL,
                attack_cap REAL,
                source_url TEXT,
                source_revision_id INTEGER,
                source_revision_timestamp TEXT,
                source_content_sha256 TEXT
            );
            CREATE TABLE dim_rule_receipt (
                rule_key TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                source_url TEXT,
                revision_id INTEGER,
                revision_timestamp TEXT,
                content_sha256 TEXT,
                notes TEXT
            );
            """
        )

    def _populate(self) -> None:
        patch = str(self.pack.get("patch") or "")
        client_patch = self.pack.get("client_patch")
        source_hash = self.pack.get("source_hash")
        index_sha256 = self.pack.get("index_sha256")
        self.connection.execute(
            "INSERT INTO dim_patch VALUES (?, ?, ?, ?, ?, ?)",
            (
                patch,
                str(client_patch) if client_patch is not None else None,
                str(self.pack.get("source") or ""),
                str(source_hash) if source_hash is not None else None,
                str(index_sha256) if index_sha256 is not None else None,
                SCHEMA_VERSION,
            ),
        )
        self.connection.executemany(
            "INSERT INTO dim_mode VALUES (?, ?, ?)",
            [
                ("summoners_rift", "Summoner's Rift", "supported_static_rules"),
                ("howling_abyss", "Howling Abyss", "source_present_execution_partial"),
                ("arena", "Arena", "source_present_execution_partial"),
                ("urf", "URF", "source_present_execution_partial"),
                ("nexus_blitz", "Nexus Blitz", "source_present_execution_partial"),
            ],
        )
        champions = self.pack.get("champions")
        if isinstance(champions, Mapping):
            for champion_key, champion in champions.items():
                if not isinstance(champion, Mapping):
                    continue
                key = str(champion_key)
                champion_id = champion.get("id")
                if not isinstance(champion_id, int) or isinstance(champion_id, bool):
                    continue
                source = champion.get("source") if isinstance(champion.get("source"), Mapping) else {}
                self.connection.execute(
                    "INSERT INTO dim_champion VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        champion_id,
                        str(champion.get("alias") or champion.get("name") or key),
                        str(champion.get("name") or champion.get("alias") or key),
                        str(champion.get("resource_type") or "none"),
                        str(champion.get("status") or "unknown"),
                        _json(champion.get("unresolved") or []),
                        source.get("bin_json_path"),
                        source.get("bin_sha256"),
                        _json(champion.get("aliases") or []),
                    ),
                )
                levels = champion.get("levels")
                if isinstance(levels, Mapping):
                    for raw_level, stats in levels.items():
                        if not isinstance(stats, Mapping):
                            continue
                        try:
                            level = int(raw_level)
                        except (TypeError, ValueError):
                            continue
                        if not 1 <= level <= 18:
                            continue
                        self.connection.execute(
                            "INSERT INTO fact_champion_stat VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (
                                key,
                                level,
                                _number(stats.get("max_health")),
                                _number(stats.get("attack_damage")),
                                _number(stats.get("armor")),
                                _number(stats.get("magic_resist")),
                                _number(stats.get("health_regen_per_5")),
                                _number(stats.get("max_resource")),
                                _number(stats.get("resource_regen_per_5")),
                            ),
                        )
        self._populate_abilities()
        self._populate_items()
        self._populate_runes()
        self._populate_structures()
        self._populate_receipts()
        self.connection.commit()

    def _populate_abilities(self) -> None:
        if self.oracle is None:
            return
        spells_by_name = getattr(self.oracle, "_spells", {})
        mechanics_by_id = getattr(self.oracle, "_mechanics", {})
        for champion_key, champion in (self.pack.get("champions") or {}).items():
            if not isinstance(champion, Mapping):
                continue
            name_key = _norm(champion.get("name") or champion.get("alias") or "")
            raw_spells = spells_by_name.get(name_key, []) if isinstance(spells_by_name, Mapping) else []
            raw_mechanics = mechanics_by_id.get(champion.get("id"), {}) if isinstance(mechanics_by_id, Mapping) else {}
            mechanics_spells = raw_mechanics.get("spells", []) if isinstance(raw_mechanics, Mapping) else []
            for raw in raw_spells if isinstance(raw_spells, Sequence) else []:
                if not isinstance(raw, Mapping):
                    continue
                key = str(raw.get("spellKey") or "").upper()
                if key not in {"Q", "W", "E", "R"}:
                    continue
                script_name = None
                for candidate in mechanics_spells if isinstance(mechanics_spells, Sequence) else []:
                    if not isinstance(candidate, Mapping):
                        continue
                    raw_name = str(candidate.get("script_name") or "")
                    if _norm(raw_name) == _norm(str(raw.get("name") or "")):
                        script_name = raw_name
                        break
                if script_name is None:
                    # The raw champion packet remains useful even when the
                    # formula graph did not map the display spell name.
                    script_name = str(raw.get("name") or "") or None
                mapped = next(
                    (
                        item
                        for item in mechanics_spells
                        if isinstance(item, Mapping)
                        and (
                            _norm(str(item.get("script_name") or "")) == _norm(str(script_name or ""))
                            or _norm(str(item.get("name") or "")) == _norm(str(raw.get("name") or ""))
                        )
                    ),
                    {},
                )
                max_rank_raw = raw.get("maxLevel")
                max_rank = int(max_rank_raw) if isinstance(max_rank_raw, int) and max_rank_raw > 0 else 5
                self.connection.execute(
                    "INSERT OR REPLACE INTO dim_ability VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        str(champion_key),
                        key,
                        raw.get("name"),
                        script_name,
                        max_rank,
                        _json(raw.get("costCoefficients") or []),
                        _json(raw.get("cooldownCoefficients") or []),
                        str(raw.get("description") or ""),
                        str(mapped.get("execution_status") or raw_mechanics.get("execution_status") or "raw_only"),
                        str(raw_mechanics.get("formula_semantics_status") or "raw_formula_graph_preserved"),
                        _json({
                            "spell_key": key,
                            "ability_icon_path": raw.get("abilityIconPath"),
                            "range": raw.get("range"),
                            "ammo": raw.get("ammo"),
                            "max_level": raw.get("maxLevel"),
                            "source": champion.get("source"),
                        }),
                    ),
                )
                values = mapped.get("data_values") if isinstance(mapped, Mapping) else None
                if isinstance(values, Sequence):
                    for value_record in values:
                        if not isinstance(value_record, Mapping):
                            continue
                        value_name = str(value_record.get("name") or "")
                        raw_values = value_record.get("values")
                        if not value_name or not isinstance(raw_values, Sequence):
                            continue
                        values_json = _json(raw_values)
                        for client_index, raw_value in enumerate(raw_values):
                            numeric = _number(raw_value)
                            if numeric is None:
                                continue
                            self.connection.execute(
                                "INSERT OR REPLACE INTO fact_ability_value VALUES (?, ?, ?, ?, ?, ?)",
                                (str(champion_key), key, value_name, client_index, numeric, values_json),
                            )

    def _populate_items(self) -> None:
        items = getattr(self.oracle, "_items", {}) if self.oracle is not None else {}
        if not isinstance(items, Mapping):
            return
        for item_key, item in items.items():
            if not isinstance(item, Mapping):
                continue
            item_id = item.get("id")
            if not isinstance(item_id, int) or isinstance(item_id, bool):
                continue
            key = str(item_key)
            description = str(item.get("description") or "")
            self.connection.execute(
                "INSERT OR REPLACE INTO dim_item VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    key,
                    item_id,
                    str(item.get("name") or key),
                    int(bool(item.get("in_store", True))),
                    int(bool(item.get("has_passive"))),
                    _json(item.get("categories") or []),
                    description,
                    _number(item.get("price_total")),
                    _sha({"id": item_id, "name": item.get("name"), "description": description}),
                ),
            )
            stats = item.get("stats")
            if isinstance(stats, Mapping):
                for stat_key, stat in stats.items():
                    if not isinstance(stat, Mapping):
                        continue
                    value = _number(stat.get("value"))
                    if value is None:
                        continue
                    self.connection.execute(
                        "INSERT OR REPLACE INTO fact_item_stat VALUES (?, ?, ?, ?, ?)",
                        (key, str(stat_key), value, int(bool(stat.get("percent"))), stat.get("label")),
                    )

    def _populate_runes(self) -> None:
        # The client packet currently has no executable rune table.  Keeping
        # names in the dimension makes joins and unsupported-state reporting
        # queryable without pretending that a prose effect is a formula.
        try:
            from .lol_oracle import _KNOWN_RUNE_NAMES

            names = tuple(str(name) for name in _KNOWN_RUNE_NAMES)
        except (ImportError, AttributeError):
            names = ("Manaflow Band",)
        self.connection.executemany(
            "INSERT INTO dim_rune VALUES (?, ?, ?, ?, ?)",
            [
                (_norm(name), name, "named_state_only" if name != "Manaflow Band" else "static_component_receipted", _source_url(name), "Rune effect execution is outside the current patch packet except the explicit Manaflow component.")
                for name in names
            ],
        )

    def _populate_structures(self) -> None:
        try:
            from .wiki_rules import STRUCTURES, wiki_rule_source
        except ImportError:
            return
        receipt = wiki_rule_source("Turret")
        rows = []
        for key, record in STRUCTURES.items():
            rows.append(
                (
                    key,
                    f"{key.title()} turret",
                    _number(record.get("health")),
                    _number(record.get("base_attack_damage")),
                    _number(record.get("attack_step")),
                    _number(record.get("attack_first_second")),
                    _number(record.get("attack_cap")),
                    receipt.get("url"),
                    receipt.get("revision_id"),
                    receipt.get("revision_timestamp"),
                    receipt.get("content_sha256"),
                )
            )
        self.connection.executemany("INSERT INTO dim_structure VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)

    def _populate_receipts(self) -> None:
        try:
            from .wiki_rules import WIKI_RULE_SOURCES
        except ImportError:
            return
        rows = []
        for key, receipt in WIKI_RULE_SOURCES.items():
            if not isinstance(receipt, Mapping):
                continue
            rows.append(
                (
                    key,
                    "revision_receipted",
                    receipt.get("url"),
                    receipt.get("revision_id"),
                    receipt.get("revision_timestamp"),
                    receipt.get("content_sha256"),
                    receipt.get("label"),
                )
            )
        self.connection.executemany("INSERT OR REPLACE INTO dim_rule_receipt VALUES (?, ?, ?, ?, ?, ?, ?)", rows)

    def _snapshot(self) -> WarehouseSnapshot:
        counts = {}
        for table in (
            "dim_champion", "fact_champion_stat", "dim_ability", "fact_ability_value",
            "dim_item", "fact_item_stat", "dim_rune", "dim_structure", "dim_rule_receipt",
        ):
            counts[table] = int(self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        payload = {
            "schema_version": SCHEMA_VERSION,
            "patch": self.pack.get("patch"),
            "client_patch": self.pack.get("client_patch"),
            "source_hash": self.pack.get("source_hash"),
            "index_sha256": self.pack.get("index_sha256"),
            "counts": counts,
        }
        return WarehouseSnapshot(
            schema_version=SCHEMA_VERSION,
            engine_version=ENGINE_VERSION,
            patch=str(self.pack.get("patch") or ""),
            client_patch=str(self.pack.get("client_patch")) if self.pack.get("client_patch") is not None else None,
            source_hash=str(self.pack.get("source_hash")) if self.pack.get("source_hash") is not None else None,
            index_sha256=str(self.pack.get("index_sha256")) if self.pack.get("index_sha256") is not None else None,
            snapshot_sha256=_sha(payload),
        )

    def _validate_sql(self, sql: str, max_rows: int) -> tuple[str, int]:
        if not isinstance(sql, str) or not sql.strip():
            raise WarehouseQueryError("sql must be a non-empty string")
        sql = sql.strip()
        if len(sql) > MAX_SQL_LENGTH:
            raise WarehouseQueryError(f"sql exceeds the {MAX_SQL_LENGTH} character limit")
        if not isinstance(max_rows, int) or isinstance(max_rows, bool) or not 1 <= max_rows <= MAX_MAX_ROWS:
            raise WarehouseQueryError(f"max_rows must be an integer in [1, {MAX_MAX_ROWS}]")
        if ";" in sql:
            raise WarehouseQueryError("multi-statement SQL is not allowed")
        if re.search(r"--|/\*|\*/", sql):
            raise WarehouseQueryError("SQL comments are not allowed")
        if not re.match(r"^(?:select|with)\b", sql, flags=re.IGNORECASE):
            raise WarehouseQueryError("only SELECT/WITH warehouse queries are allowed")
        forbidden = r"\b(?:insert|update|delete|replace|drop|alter|create|attach|detach|vacuum|pragma|reindex|transaction|commit|rollback|load_extension|readfile|writefile)\b"
        if re.search(forbidden, sql, flags=re.IGNORECASE):
            raise WarehouseQueryError("mutating or connection-control SQL is not allowed")
        return sql, max_rows

    def query_sql(self, sql: str, *, max_rows: int = DEFAULT_MAX_ROWS) -> dict[str, Any]:
        """Execute one bounded SELECT/WITH query against the resident schema."""

        sql, max_rows = self._validate_sql(sql, max_rows)
        key = (re.sub(r"\s+", " ", sql).strip(), max_rows)
        self._queries += 1
        cached = self._cache.get(key)
        if cached is not None:
            self._cache_hits += 1
            result = dict(cached)
            result["cache_hit"] = True
            return result
        started = time.perf_counter()
        try:
            # Wrapping the user query gives us a hard row bound even when the
            # caller forgot LIMIT.  Parameters cannot be used for the inner
            # statement because it is already a validated read-only string.
            cursor = self.connection.execute(
                f"SELECT * FROM ({sql}) AS warehouse_query LIMIT {max_rows + 1}"
            )
            raw_rows = cursor.fetchall()
        except sqlite3.Error as exc:
            raise WarehouseQueryError(f"warehouse SQL failed: {exc}") from exc
        truncated = len(raw_rows) > max_rows
        rows = [dict(row) for row in raw_rows[:max_rows]]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result = {
            "status": "available",
            "route": "warehouse_sql",
            "engine": ENGINE_VERSION,
            "patch": self.snapshot.patch,
            "snapshot": self.snapshot.to_mapping(),
            "query_sha256": _sha({"sql": key[0], "max_rows": max_rows}),
            "rows": rows,
            "row_count": len(rows),
            "truncated": truncated,
            "max_rows": max_rows,
            "cache_hit": False,
            "latency_ms": round(elapsed_ms, 3),
            "calculation": "Bounded read-only SELECT/WITH over the resident patch warehouse.",
            "sources": [
                {
                    "kind": "patch_packet",
                    "label": "patch-pinned mechanics packet",
                    "patch": self.snapshot.patch,
                    "sha256": self.snapshot.source_hash,
                    "url": _client_root_url(self.snapshot.client_patch),
                },
                {
                    "kind": "wiki_formula",
                    "label": "League Wiki champion-stat rules",
                    "url": _source_url("Champion statistic"),
                },
            ],
        }
        self._cache[key] = dict(result)
        return result

    def schema(self) -> dict[str, Any]:
        tables = {}
        rows = self.connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
        for row in rows:
            name = str(row[0])
            columns = self.connection.execute(f"PRAGMA table_info({self._safe_identifier(name)})").fetchall()
            tables[name] = [
                {"name": str(column[1]), "type": str(column[2]), "notnull": bool(column[3]), "primary_key": bool(column[5])}
                for column in columns
            ]
        return {
            "status": "available",
            "route": "warehouse_schema",
            "engine": ENGINE_VERSION,
            "snapshot": self.snapshot.to_mapping(),
            "tables": tables,
        }

    def status(self) -> dict[str, Any]:
        counts = {}
        for table in (
            "dim_champion", "fact_champion_stat", "dim_ability", "fact_ability_value",
            "dim_item", "fact_item_stat", "dim_rune", "dim_structure", "dim_rule_receipt",
        ):
            counts[table] = int(self.connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
        return {
            "status": "ready" if self.ready else "closed",
            "engine": ENGINE_VERSION,
            "schema_version": SCHEMA_VERSION,
            "resident": self.ready,
            "snapshot": self.snapshot.to_mapping(),
            "table_counts": counts,
            "queries": self._queries,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "query_contract": {
                "read_only": True,
                "max_rows": MAX_MAX_ROWS,
                "allowed_statements": ["SELECT", "WITH"],
                "source_bound": True,
            },
        }

    def _resolve_champion(self, value: str) -> dict[str, Any] | None:
        rows = self.connection.execute(
            "SELECT * FROM dim_champion WHERE champion_key = ? OR lower(name) = lower(?) OR lower(alias) = lower(?)",
            (_norm(value), value, value),
        ).fetchall()
        if rows:
            return dict(rows[0])
        # Deterministic token-boundary fallback handles punctuation such as
        # K'Sante without introducing fuzzy nearest-neighbour guesses.
        candidates = self.connection.execute("SELECT * FROM dim_champion ORDER BY length(name) DESC, champion_id").fetchall()
        for row in candidates:
            if _mentions(str(value), str(row["name"])) or _mentions(str(value), str(row["alias"])):
                return dict(row)
        return None

    def _resolve_item(self, value: str) -> dict[str, Any] | None:
        rows = self.connection.execute(
            "SELECT * FROM dim_item WHERE item_key = ? OR lower(name) = lower(?)",
            (_norm(value), value),
        ).fetchall()
        if rows:
            return dict(rows[0])
        candidates = self.connection.execute("SELECT * FROM dim_item ORDER BY length(name) DESC, item_id").fetchall()
        for row in candidates:
            if _mentions(str(value), str(row["name"])):
                return dict(row)
        return None

    @staticmethod
    def _item_names(value: Any) -> list[tuple[str, int]]:
        if value is None:
            return []
        if not isinstance(value, (list, tuple)):
            raise WarehouseQueryError("state.items must be an array")
        result = []
        for item in value:
            if isinstance(item, str):
                result.append((item, 1))
            elif isinstance(item, Mapping) and isinstance(item.get("name"), str):
                count = item.get("count", 1)
                if not isinstance(count, int) or isinstance(count, bool) or count < 1 or count > 7:
                    raise WarehouseQueryError("item count must be an integer in [1, 7]")
                result.append((str(item["name"]), count))
            else:
                raise WarehouseQueryError("each state item must be a name or {name, count}")
        return result

    def state_query(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Resolve one champion state against static facts and item components.

        This is the structured companion to natural-language questions.  It is
        deliberately broad about the shape of state, but strict about what is
        executable: static flat item stats are composed; named effect rules,
        non-empty buffs/debuffs, and unspecified live transitions are surfaced
        as blockers.
        """

        if not isinstance(state, Mapping):
            raise WarehouseQueryError("state must be an object")
        requested_patch = state.get("patch")
        if requested_patch is not None and str(requested_patch).removeprefix("v") != self.snapshot.patch:
            return {
                "status": "invalid_scenario",
                "route": "warehouse_state",
                "reason": f"state requests patch {requested_patch!r}, but the resident snapshot is {self.snapshot.patch}",
                "snapshot": self.snapshot.to_mapping(),
            }
        requested_mode = state.get("mode")
        if requested_mode is not None and str(requested_mode).casefold() not in {"summoners_rift", "summoner's rift", "sr"}:
            return {
                "status": "unsupported",
                "route": "warehouse_state",
                "reason": "static state composition is currently validated only for Summoner's Rift",
                "mode": requested_mode,
                "snapshot": self.snapshot.to_mapping(),
            }
        champion_value = state.get("champion") or state.get("champion_name")
        if not isinstance(champion_value, str) or not champion_value.strip():
            return {
                "status": "needs_input",
                "route": "warehouse_state",
                "required_inputs": [{"path": "champion", "type": "champion", "reason": "state identity is required"}],
                "snapshot": self.snapshot.to_mapping(),
            }
        champion = self._resolve_champion(champion_value)
        if champion is None:
            return {
                "status": "invalid_scenario",
                "route": "warehouse_state",
                "reason": f"{champion_value!r} is not an exact champion identity in patch {self.snapshot.patch}",
                "snapshot": self.snapshot.to_mapping(),
            }
        level = state.get("level")
        if not isinstance(level, int) or isinstance(level, bool):
            return {
                "status": "needs_input",
                "route": "warehouse_state",
                "required_inputs": [{"path": "level", "type": "integer[1-18]", "reason": "level selects the exact champion-stat fact"}],
                "champion": champion["name"],
                "snapshot": self.snapshot.to_mapping(),
            }
        if not 1 <= level <= 18:
            return {
                "status": "invalid_scenario",
                "route": "warehouse_state",
                "reason": "level must be an integer in [1, 18]",
                "snapshot": self.snapshot.to_mapping(),
            }
        row = self.connection.execute(
            "SELECT * FROM fact_champion_stat WHERE champion_key = ? AND level = ?",
            (champion["champion_key"], level),
        ).fetchone()
        if row is None:
            return {
                "status": "unsupported",
                "route": "warehouse_state",
                "reason": "the exact patch packet has no champion-stat row for this state",
                "snapshot": self.snapshot.to_mapping(),
            }
        base = dict(row)
        derived = {
            key: base.get(key)
            for key in (
                "max_health", "attack_damage", "armor", "magic_resist",
                "health_regen_per_5", "max_resource", "resource_regen_per_5",
            )
        }
        # Flat item stats that have no champion base cell still get an explicit
        # accumulator.  Otherwise a harmless item such as an Amplifying Tome
        # could disappear from the result without a blocker.
        for key in (
            "ability_power", "ability_haste", "critical_strike_chance",
            "move_speed", "lethality", "magic_penetration", "armor_penetration",
            "life_steal", "attack_speed_percent",
        ):
            derived.setdefault(key, 0.0)
        components: list[dict[str, Any]] = []
        blockers: list[str] = []
        component_sources: list[dict[str, Any]] = []
        ability_result: dict[str, Any] | None = None
        ability_spec = state.get("ability")
        if ability_spec is not None:
            if isinstance(ability_spec, str):
                ability_key = ability_spec.upper()
                ability_rank = 1
            elif isinstance(ability_spec, Mapping):
                ability_key = str(ability_spec.get("key") or "").upper()
                ability_rank = ability_spec.get("rank", 1)
            else:
                raise WarehouseQueryError("state.ability must be a Q/W/E/R name or {key, rank}")
            if ability_key not in {"Q", "W", "E", "R"} or not isinstance(ability_rank, int) or isinstance(ability_rank, bool):
                return {
                    "status": "invalid_scenario",
                    "route": "warehouse_state",
                    "reason": "ability must identify Q, W, E, or R and an integer rank",
                    "snapshot": self.snapshot.to_mapping(),
                }
            ability_row = self.connection.execute(
                "SELECT * FROM dim_ability WHERE champion_key = ? AND ability_key = ?",
                (champion["champion_key"], ability_key),
            ).fetchone()
            if ability_row is None:
                blockers.append(f"{champion['name']} has no normalized {ability_key} ability row in this packet")
            elif not 1 <= ability_rank <= int(ability_row["max_rank"] or 5):
                return {
                    "status": "invalid_scenario",
                    "route": "warehouse_state",
                    "reason": f"{champion['name']} {ability_key} rank must be in [1, {ability_row['max_rank']}]",
                    "snapshot": self.snapshot.to_mapping(),
                }
            else:
                costs = json.loads(str(ability_row["cost_json"] or "[]"))
                cooldowns = json.loads(str(ability_row["cooldown_json"] or "[]"))
                ability_values: dict[str, list[float]] = {}
                for value_row in self.connection.execute(
                    "SELECT value_name, client_index, value FROM fact_ability_value WHERE champion_key = ? AND ability_key = ? ORDER BY value_name, client_index",
                    (champion["champion_key"], ability_key),
                ).fetchall():
                    ability_values.setdefault(str(value_row["value_name"]), []).append(float(value_row["value"]))
                ability_result = {
                    "key": ability_key,
                    "rank": ability_rank,
                    "name": ability_row["ability_name"],
                    "cost": costs[ability_rank - 1] if isinstance(costs, list) and len(costs) >= ability_rank else None,
                    "cooldown": cooldowns[ability_rank - 1] if isinstance(cooldowns, list) and len(cooldowns) >= ability_rank else None,
                    "execution_status": ability_row["execution_status"],
                    "formula_semantics_status": ability_row["formula_semantics_status"],
                    "values_by_client_index": ability_values,
                }
        item_input = self._item_names(state.get("items"))
        for item_name, count in item_input:
            item = self._resolve_item(item_name)
            if item is None:
                blockers.append(f"unknown item: {item_name}")
                continue
            component_sources.append(
                {
                    "kind": "item_packet",
                    "label": "patch-pinned item data",
                    "item": item["name"],
                    "item_id": item["item_id"],
                    "sha256": item.get("source_sha256"),
                    "url": _source_url(str(item["name"])),
                }
            )
            if int(item["has_passive"]):
                blockers.append(f"{item['name']} has effect text; its trigger state is not executable in the warehouse kernel")
            stats_rows = self.connection.execute(
                "SELECT stat_key, value, is_percent, label FROM fact_item_stat WHERE item_key = ? ORDER BY stat_key",
                (item["item_key"],),
            ).fetchall()
            item_stats: dict[str, float] = {}
            for stat_row in stats_rows:
                stat_key = str(stat_row["stat_key"])
                value = float(stat_row["value"]) * count
                item_stats[stat_key] = item_stats.get(stat_key, 0.0) + value
                if int(stat_row["is_percent"]):
                    blockers.append(f"{item['name']} {stat_key} is percentage-based and needs an attack-speed/penetration state rule")
                elif stat_key in {"health", "mana", "attack_damage", "armor", "magic_resist", "ability_power", "ability_haste", "critical_strike_chance", "move_speed", "lethality", "magic_penetration", "armor_penetration", "life_steal"}:
                    target_key = {
                        "health": "max_health",
                        "mana": "max_resource",
                    }.get(stat_key, stat_key)
                    if target_key in derived and derived[target_key] is not None:
                        derived[target_key] = float(derived[target_key]) + value
                    else:
                        blockers.append(f"{item['name']} {stat_key} cannot be composed with this champion's resource/stat schema")
                else:
                    blockers.append(f"{item['name']} has an unmapped static stat {stat_key}")
            components.append({"item": item["name"], "count": count, "static_stats": item_stats, "has_passive": bool(item["has_passive"])})

        runes = state.get("runes")
        if runes is not None:
            if not isinstance(runes, (list, tuple)):
                raise WarehouseQueryError("state.runes must be an array")
            for rune_name in runes:
                if not isinstance(rune_name, str):
                    raise WarehouseQueryError("state rune names must be strings")
                rune = self.connection.execute(
                    "SELECT * FROM dim_rune WHERE lower(name) = lower(?) OR rune_key = ?",
                    (rune_name, _norm(rune_name)),
                ).fetchone()
                if rune is None:
                    blockers.append(f"unknown rune: {rune_name}")
                elif str(rune["effect_status"]) != "static_component_receipted":
                    blockers.append(f"{rune['name']} effect semantics are not executable in this snapshot")
                elif _norm(rune_name) == _norm("Manaflow Band"):
                    rune_stacks = state.get("rune_stacks", {}).get("Manaflow Band") if isinstance(state.get("rune_stacks"), Mapping) else None
                    if not isinstance(rune_stacks, int) or isinstance(rune_stacks, bool) or not 0 <= rune_stacks <= 10:
                        blockers.append("Manaflow Band requires an explicit stack count in state.rune_stacks")
                    elif derived.get("max_resource") is not None:
                        derived["max_resource"] = float(derived["max_resource"]) + 25.0 * rune_stacks
                        components.append({"rune": "Manaflow Band", "stacks": rune_stacks, "maximum_resource": 25.0 * rune_stacks})

        for field in ("buffs", "debuffs", "shields", "event_state", "transforms"):
            value = state.get(field)
            if value not in (None, {}, [], (), False):
                blockers.append(f"{field} is non-empty and requires a validated transition rule")
        known_state_fields = {
            "champion", "champion_name", "level", "patch", "mode", "items", "runes", "rune_stacks", "ability",
            "buffs", "debuffs", "shields", "event_state", "transforms",
        }
        for key, value in state.items():
            if key not in known_state_fields and value not in (None, {}, [], (), False):
                blockers.append(f"state field {key!r} is not in the validated warehouse schema")
        status = "unsupported" if blockers else "available"
        result = {
            "status": status,
            "route": "warehouse_state",
            "engine": ENGINE_VERSION,
            "patch": self.snapshot.patch,
            "mode": state.get("mode") or "summoners_rift",
            "champion": champion["name"],
            "level": level,
            "ability": ability_result,
            "base_stats": {key: base.get(key) for key in derived},
            "derived_stats": derived,
            "components": components,
            "unavailable": sorted(set(blockers)),
            "snapshot": self.snapshot.to_mapping(),
            "calculation": "Champion level fact plus explicitly named flat item/rune components; no hidden effects.",
            "assumptions": ["patch is the resident patch-pinned warehouse snapshot", "mode defaults to Summoner's Rift for static champion/item facts"],
            "sources": [
                {"kind": "champion_packet", "label": "patch-pinned champion data", "champion": champion["name"], "sha256": champion.get("source_sha256"), "url": _client_file_url(self.snapshot.client_patch, champion.get("source_path"))},
                {"kind": "wiki_formula", "label": "League Wiki champion-stat growth formula", "url": _source_url("Champion statistic")},
                *component_sources,
            ],
        }
        return result

    def natural_query(self, question: str) -> dict[str, Any] | None:
        """Plan a closed static-state question without a second model call.

        The exact oracle remains first.  This small planner is only a bridge
        for questions such as "Malphite's level-6 stats with Sapphire Crystal"
        that name a closed champion/item state but do not match one of the
        narrow legacy arithmetic handlers.  It intentionally returns ``None``
        when level, identity, or a state-shaped intent is missing, allowing the
        semantic layer to issue its normal typed contract.
        """

        if not isinstance(question, str) or not question.strip():
            return None
        if not re.search(r"\b(?:stats?|health|mana|armor|resist|AD|attack damage|build|items?|state)\b", question, re.I):
            return None
        levels = [int(value) for value in re.findall(r"\b(?:level|lvl|lv)\s*(?:=|:|-)?\s*(\d+)\b", question, re.I)]
        if len(levels) != 1:
            return None
        champion = None
        rows = self.connection.execute("SELECT name, alias FROM dim_champion ORDER BY length(name) DESC, champion_id").fetchall()
        for row in rows:
            if _mentions(question, str(row["name"])) or _mentions(question, str(row["alias"])):
                champion = str(row["name"])
                break
        if champion is None:
            return None
        item_names = []
        item_rows = self.connection.execute("SELECT name FROM dim_item ORDER BY length(name) DESC, item_id").fetchall()
        for row in item_rows:
            name = str(row["name"])
            if _mentions(question, name):
                item_names.append(name)
        # Named runes are included only when the warehouse has the exact
        # dimension row.  Their execution status is reported by state_query.
        rune_names = []
        for row in self.connection.execute("SELECT name FROM dim_rune ORDER BY length(name) DESC, rune_key").fetchall():
            name = str(row["name"])
            if _mentions(question, name):
                rune_names.append(name)
        state: dict[str, Any] = {"champion": champion, "level": levels[0], "items": item_names, "runes": rune_names}
        manaflow = next((name for name in rune_names if _norm(name) == _norm("Manaflow Band")), None)
        if manaflow is not None:
            if re.search(r"\b(?:fully|completely|max(?:imum)?)\s*stack(?:ed|s)?\b", question, re.I):
                stacks = 10
            else:
                match = re.search(r"\b(\d+)\s*(?:manaflow\s+band\s+)?stacks?\b", question, re.I)
                stacks = int(match.group(1)) if match else None
            if stacks is not None:
                state["rune_stacks"] = {"Manaflow Band": stacks}
        return self.state_query(state)

    def close(self) -> None:
        if self._closed:
            return
        self.connection.close()
        self._closed = True


__all__ = [
    "DEFAULT_MAX_ROWS",
    "ENGINE_VERSION",
    "MAX_MAX_ROWS",
    "SCHEMA_VERSION",
    "StateWarehouse",
    "WarehouseQueryError",
    "WarehouseSnapshot",
]
