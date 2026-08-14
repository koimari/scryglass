"""Publish descriptive ratings after complete new OE games appear."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pandas as pd
import pyarrow.parquet as pq

from lol_kills.etl.oe_ingest import ingest_oe, validate_accepted_source_receipt
from lol_kills.etl.oe_live_source import (
    _complete_player_game_ids,
    _identity_complete_player_game_ids,
    build_live_source,
)
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.export import pack_spec
from lol_kills.export.pack_records import profile_game_has_complete_stats
from lol_kills.export.public_pack import (
    _complete_player_game_ids as _complete_identity_game_ids,
    export_public_pack,
    source_identity_sha256,
)
from lol_kills.ratings.player_map_grades import CORE_INPUTS
from lol_kills.etl.oe_database import TRANSFORM_VERSION
from lol_kills.research.composition_signal import CompositionSignalError, validate_public_signal
from lol_kills.refresh_ledger import worker_commit


OE_TEAM_CACHE = Path("data/lol/warehouse/parquet/oe_team_games.parquet")
OE_API_TEAM_CACHE = Path("data/lol/warehouse/parquet/oe_api_team_games.parquet")
OE_API_PLAYER_CACHE = Path("data/lol/warehouse/parquet/oe_api_player_games.parquet")
OE_META = Path("data/lol/warehouse/parquet/oe_meta.json")
LIVE_ROOT = Path("data/lol/warehouse/parquet/oe_live")
LIVE_MAPS = LIVE_ROOT / "maps.parquet"
LIVE_TEAMS = LIVE_ROOT / "oe_team_games.parquet"
LIVE_PLAYERS = LIVE_ROOT / "oe_player_games.parquet"
STATIC_APP_PUBLIC_ROOT = (
    Path(__file__).resolve().parents[1] / "apps" / "scryglass" / "public"
).resolve()
REVIEWED_QUARANTINED_GAME_IDS = frozenset(
    {
        "oe:game:40e753290593c20c4cc990b1abfaf674",
        "oe:game:52db934e7e36169f9bf455bb00110045",
        "oe:game:9b9c8c1375765cffab98cacea34a3259",
        "oe:game:b53fb5b013e28c288688f5bb39d518b7",
        "oe:game:b91ad221257bc4357a3e0d9bc6ea75cd",
        # OE re-identified the 2026-08-11 LJL series (L Guide Gaming vs Uwinks)
        # in the 2026-08-12T16:08Z revision; the old IDs left the source.
        "LOLTMNT01_441503",
        "LOLTMNT01_442486",
        "LOLTMNT01_442514",
    }
)


class RefreshValidationError(RuntimeError):
    """A refresh cannot prove complete source-to-pack integration."""


class SyncAlreadyRunning(RuntimeError):
    """Another process owns the refresh lock."""


@dataclass(frozen=True)
class SyncConfig:
    root: Path
    public_root: Path
    output_root: Path
    state_path: Path
    lock_path: Path
    health_path: Path
    years: tuple[int, ...] = (2025, 2026)
    discovery_cache_hours: int = 6
    window_hours: int = 24 * 120
    lookback_days: int = 120
    max_workers: int = 8
    runtime_root: Path | None = None
    accepted_source_receipt: Path | None = None
    accepted_import_receipt: Path | None = None

    @property
    def data_root(self) -> Path:
        return (self.runtime_root or self.root).resolve()


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _canonical_ids(values: Sequence[Any]) -> list[str]:
    return sorted({key for value in values if (key := canonical_source_game_key(value))})


def _source_game_ids(root: Path) -> list[str]:
    def read_ids(path: Path) -> list[str]:
        columns = pq.ParquetFile(path).schema_arrow.names
        identity_column = next(
            (name for name in ("game_uid", "gameid", "oe_gameid") if name in columns),
            None,
        )
        if identity_column is None:
            return []
        values = pq.read_table(path, columns=[identity_column]).column(identity_column).to_pylist()
        return _canonical_ids(values)

    observed: set[str] = set()
    primary_path = root / OE_TEAM_CACHE
    if primary_path.is_file():
        try:
            observed.update(read_ids(primary_path))
        except (OSError, ValueError, RuntimeError):
            pass
    api_player_path = root / OE_API_PLAYER_CACHE
    api_team_path = root / OE_API_TEAM_CACHE
    if api_player_path.is_file() and api_team_path.is_file():
        try:
            players = pd.read_parquet(api_player_path)
            complete_ids = _identity_complete_player_game_ids(players)
            observed.update(set(read_ids(api_team_path)).intersection(complete_ids))
        except (OSError, ValueError, RuntimeError, TypeError, KeyError):
            pass
    return sorted(observed)


def _annual_source_latest(root: Path) -> str | None:
    payload = _load_json(root / OE_META)
    source_files = payload.get("source_files")
    if not isinstance(source_files, list):
        return None
    values = [
        str(item.get("date_max_utc"))
        for item in source_files
        if isinstance(item, dict) and item.get("date_max_utc")
    ]
    return max(values) if values else None


def _source_file_hashes(root: Path) -> dict[str, str]:
    payload = _load_json(root / OE_META)
    values: dict[str, str] = {}
    for item in payload.get("source_files", []):
        if not isinstance(item, dict):
            continue
        year = item.get("year")
        digest = item.get("raw_sha256")
        if isinstance(year, int) and isinstance(digest, str) and len(digest) == 64:
            values[str(year)] = digest
    return values


def ingest_oe_csv(
    root: Path,
    *,
    years: Sequence[int] = (2025, 2026),
    source_receipt: Path | None = None,
    import_receipt: Path | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Refresh the public annual OE cache and return its source receipt."""

    database_refreshed = os.environ.get("SCRYGLASS_OE_DATABASE_REFRESHED") == "1"
    browser_refreshed = os.environ.get("SCRYGLASS_OE_BROWSER_REFRESHED") == "1"
    accepted_source: dict[str, Any] | None = None
    if source_receipt is not None:
        current_year = max(years)
        source_path = (
            root
            / "data/lol/warehouse/raw"
            / f"{current_year}_LoL_esports_match_data_from_OraclesElixir.csv"
        )
        accepted_source = validate_accepted_source_receipt(
            source_receipt,
            source_path,
            current_year,
        )
    accepted_import: dict[str, Any] | None = None
    if import_receipt is not None:
        try:
            loaded = json.loads(import_receipt.expanduser().resolve().read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RefreshValidationError("accepted OE import receipt is invalid") from error
        if not isinstance(loaded, dict):
            raise RefreshValidationError("accepted OE import receipt is not an object")
        if accepted_source is None or loaded.get("source_file_sha256") != accepted_source["raw_sha256"]:
            raise RefreshValidationError("accepted OE import receipt does not match the source")
        accepted_import = loaded
    if not database_refreshed:
        ingest_oe(
            years=[str(year) for year in years],
            download=not browser_refreshed,
            force_download=False,
        )
    game_ids = _source_game_ids(root)
    return {
        "source_mode": "oe_only",
        "source_transport": (
            "supabase_incremental_oe"
            if database_refreshed
            else (
                "brave_origin_browser_download"
                if browser_refreshed
                else "public_google_drive_file"
            )
        ),
        "source_latest": _annual_source_latest(root),
        "source_game_count": len(game_ids),
        "cached": True,
        "accepted_source_receipt": accepted_source,
        "accepted_import_receipt": accepted_import,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_frame(
    path: Path,
    required: Sequence[str] = (),
    optional: Sequence[str] = (),
) -> pd.DataFrame:
    if not path.is_file():
        raise RefreshValidationError(f"missing live source file: {path}")
    columns = pq.ParquetFile(path).schema_arrow.names
    identity_columns = [name for name in ("game_uid", "gameid", "oe_gameid") if name in columns]
    if not identity_columns or any(name not in columns for name in required):
        raise RefreshValidationError(f"live source schema is incomplete: {path}")
    optional_columns = [name for name in optional if name in columns]
    frame = pd.read_parquet(
        path,
        columns=[*identity_columns, *required, *optional_columns],
    )
    frame["_game_id"] = [
        next(
            (
                key
                for column in identity_columns
                if (key := canonical_source_game_key(row[column]))
            ),
            "",
        )
        for _, row in frame.iterrows()
    ]
    if frame["_game_id"].eq("").any():
        raise RefreshValidationError(f"live source contains an empty game identity: {path}")
    return frame


def validate_live_source(root: Path, new_game_ids: Sequence[str]) -> dict[str, Any]:
    """Return the complete canonical source set and verify every new game."""

    requested = set(_canonical_ids(list(new_game_ids)))
    if len(requested) != len(new_game_ids):
        raise RefreshValidationError("new game identities are duplicated or malformed")
    maps = _identity_frame(root / LIVE_MAPS)
    teams = _identity_frame(root / LIVE_TEAMS, ("side", "result", "teamname"))
    players = _identity_frame(
        root / LIVE_PLAYERS,
        ("side", "position", "playername", *CORE_INPUTS),
        ("datacompleteness",),
    )
    map_ids = _canonical_ids(maps["_game_id"].tolist())
    if len(map_ids) != len(maps):
        raise RefreshValidationError("live maps are not one row per canonical game identity")
    valid_team_ids: set[str] = set()
    for game_id, team_rows in teams.groupby("_game_id", sort=False):
        sides = set(team_rows["side"].astype(str).str.title())
        results = set(pd.to_numeric(team_rows["result"], errors="coerce").dropna().astype(int))
        team_names = team_rows["teamname"].astype("string").fillna("").str.strip()
        if (
            len(team_rows) == 2
            and sides == {"Blue", "Red"}
            and results == {0, 1}
            and not team_names.eq("").any()
            and team_names.nunique() == 2
        ):
            valid_team_ids.add(str(game_id))
    legacy_identity_complete_ids = _complete_identity_game_ids(
        players.assign(game_uid=players["_game_id"])
    )
    identity_complete_ids = _identity_complete_player_game_ids(
        players.assign(game_uid=players["_game_id"])
    )
    statistics_complete_ids = _complete_player_game_ids(players)
    accepted_ids = sorted(set(map_ids).intersection(valid_team_ids, identity_complete_ids))
    missing_requested = sorted(requested.difference(statistics_complete_ids))
    if missing_requested:
        game_id = missing_requested[0]
        raise RefreshValidationError(
            f"game {game_id} is absent from the complete OE source set"
        )
    roles = {"top", "jng", "mid", "bot", "sup"}
    for game_id in sorted(requested):
        map_rows = maps[maps["_game_id"].eq(game_id)]
        team_rows = teams[teams["_game_id"].eq(game_id)]
        player_rows = players[players["_game_id"].eq(game_id)]
        if (len(map_rows), len(team_rows), len(player_rows)) != (1, 2, 10):
            raise RefreshValidationError(
                f"game {game_id} has malformed rows: maps={len(map_rows)} teams={len(team_rows)} players={len(player_rows)}"
            )
        sides = set(team_rows["side"].astype(str).str.title())
        results = set(pd.to_numeric(team_rows["result"], errors="coerce").dropna().astype(int))
        team_names = team_rows["teamname"].astype("string").fillna("").str.strip()
        if sides != {"Blue", "Red"} or results != {0, 1} or team_names.eq("").any() or team_names.nunique() != 2:
            raise RefreshValidationError(f"game {game_id} has malformed teams, sides, or results")
        names = player_rows["playername"].astype("string").fillna("").str.strip()
        if names.eq("").any() or names.nunique() != 10:
            raise RefreshValidationError(f"game {game_id} has malformed player identities")
        statistics = player_rows[list(CORE_INPUTS)].apply(pd.to_numeric, errors="coerce")
        if statistics.isna().any().any():
            raise RefreshValidationError(f"game {game_id} has incomplete player statistics")
        nonnegative = (
            "kills", "deaths", "assists", "teamkills", "dpm", "damageshare",
            "totalgold", "cspm", "wpm", "wcpm",
        )
        if statistics[list(nonnegative)].lt(0).any().any() or statistics["gamelength"].le(0).any():
            raise RefreshValidationError(f"game {game_id} has invalid player statistics")
        if statistics["kills"].gt(statistics["teamkills"]).any() or statistics["damageshare"].gt(1).any():
            raise RefreshValidationError(f"game {game_id} has invalid player statistics")
        for side in ("Blue", "Red"):
            side_rows = player_rows[player_rows["side"].astype(str).str.title().eq(side)]
            if len(side_rows) != 5 or set(side_rows["position"].astype(str).str.casefold()) != roles:
                raise RefreshValidationError(f"game {game_id} has malformed {side} roles")
            damage_share = pd.to_numeric(side_rows["damageshare"], errors="coerce").sum()
            if abs(float(damage_share) - 1) > 1e-5:
                raise RefreshValidationError(f"game {game_id} has malformed {side} damage share")
    return {
        "game_ids": accepted_ids,
        "statistics_complete_game_ids": sorted(statistics_complete_ids),
        "legacy_identity_game_ids": sorted(
            set(map_ids).intersection(valid_team_ids, legacy_identity_complete_ids)
        ),
        "game_count": len(accepted_ids),
        "identity_sha256": source_identity_sha256(accepted_ids),
        "candidate_game_count": len(map_ids),
        "rejected_incomplete_game_count": len(set(map_ids).difference(accepted_ids)),
    }


def _published_pack_source(public_root: Path) -> dict[str, Any] | None:
    manifest = _load_json(public_root / "manifest.json")
    if not manifest:
        return None
    ratings = manifest.get("ratings") or {}
    try:
        game_count = int(ratings["source_game_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise RefreshValidationError("published pack has no valid source game count") from error
    identity_sha256 = str(ratings.get("source_identity_sha256") or "")
    if game_count < 1 or len(identity_sha256) != 64:
        raise RefreshValidationError("published pack has no valid source identity binding")
    return {
        "pack_id": str(manifest.get("pack_id") or ""),
        "game_count": game_count,
        "identity_sha256": identity_sha256,
    }


def _continuity_baseline(
    config: SyncConfig,
    state: dict[str, Any],
) -> tuple[list[str], dict[str, Any] | None]:
    """Resolve the exact source set behind the currently published pack."""

    binding = _published_pack_source(config.public_root)
    state_ids = _canonical_ids(state.get("published_game_ids", []))
    if binding is None:
        return state_ids, None
    if (
        len(state_ids) == binding["game_count"]
        and source_identity_sha256(state_ids) == binding["identity_sha256"]
    ):
        return state_ids, binding
    current_source = validate_live_source(config.data_root, [])
    if (
        current_source["game_count"] != binding["game_count"]
        or current_source["identity_sha256"] != binding["identity_sha256"]
    ):
        legacy_ids = _canonical_ids(current_source.get("legacy_identity_game_ids", []))
        if (
            len(legacy_ids) != binding["game_count"]
            or source_identity_sha256(legacy_ids) != binding["identity_sha256"]
        ):
            index_ids = _canonical_ids(state.get("release_index_game_ids", []))
            if (
                index_ids
                and len(state_ids) >= binding["game_count"]
                and set(index_ids) <= set(state_ids)
            ):
                # A bootstrap recovery seeded the worker with a validated
                # source that covers the active release's own published game
                # index. The worker is current or ahead; the next publication
                # supersedes the active release.
                return state_ids, binding
            raise RefreshValidationError(
                "current source cache does not match the published pack; exact continuity cannot be proved"
            )
        return legacy_ids, binding
    return current_source["game_ids"], binding


def validate_source_continuity(
    baseline_game_ids: Sequence[str],
    source: dict[str, Any],
    published: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reject a candidate source when a completed published map disappears."""

    baseline = set(_canonical_ids(list(baseline_game_ids)))
    candidate = set(_canonical_ids(source.get("game_ids", [])))
    missing = baseline.difference(candidate)
    quarantined = sorted(missing.intersection(REVIEWED_QUARANTINED_GAME_IDS))
    unexpected = sorted(missing.difference(REVIEWED_QUARANTINED_GAME_IDS))
    if unexpected:
        raise RefreshValidationError(
            f"source continuity failed; {len(unexpected)} published completed maps disappeared; first={unexpected[0]}"
        )
    if (
        published is not None
        and source["game_count"] + len(quarantined) < published["game_count"]
    ):
        raise RefreshValidationError("source continuity failed; candidate game count decreased")
    return {
        "previous_pack_id": published.get("pack_id") if published else None,
        "previous_game_count": len(baseline),
        "candidate_game_count": source["game_count"],
        "retained_game_count": len(baseline.intersection(candidate)),
        "added_game_count": len(candidate.difference(baseline)),
        "missing_game_count": len(missing),
        "quarantined_game_ids": quarantined,
        "quarantined_identity_sha256": source_identity_sha256(quarantined),
    }


def _read_pack_json(pack_dir: Path, relative: str) -> Any:
    path = pack_dir / relative
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RefreshValidationError(f"public pack JSON is unreadable: {relative}") from error


def validate_public_identity(pack_dir: Path) -> dict[str, Any]:
    """Reject duplicate canonical identities before a pack becomes active."""

    teams = _read_pack_json(pack_dir, "features/ratings_snapshot.json")
    players = _read_pack_json(pack_dir, "features/player_ratings_snapshot.json")
    team_records = _read_pack_json(pack_dir, "features/team_records.json")
    profile_records = _read_pack_json(pack_dir, "features/profile_records.json")

    team_keys: set[str] = set()
    if isinstance(teams, list) and teams:
        keys: list[str] = []
        for row in teams:
            if not isinstance(row, dict):
                raise RefreshValidationError("team ratings contain a non-object row")
            key = str(row.get("team_key") or "").strip()
            name = str(row.get("team") or "").strip()
            if not key or not name:
                raise RefreshValidationError("team ratings contain an empty canonical identity")
            if row.get("evidence_disconnected") is True:
                raise RefreshValidationError(f"disconnected team is present in public ratings: {name}")
            keys.append(key)
        duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
        if duplicate_keys:
            raise RefreshValidationError(
                "team ratings contain duplicate canonical keys: " + ", ".join(duplicate_keys[:3])
            )
        team_keys = set(keys)

    player_name_count = 0
    player_case_collision_count = 0
    if isinstance(players, list) and players:
        names: list[str] = []
        for row in players:
            if not isinstance(row, dict):
                raise RefreshValidationError("player ratings contain a non-object row")
            name = str(row.get("player") or "").strip()
            if not name:
                raise RefreshValidationError("player ratings contain an empty player identity")
            if row.get("evidence_disconnected") is True or str(row.get("evidence_state") or "").casefold() == "disconnected":
                raise RefreshValidationError(f"disconnected player is present in public ratings: {name}")
            names.append(name)
        duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
        if duplicate_names:
            raise RefreshValidationError(
                "player ratings contain duplicate exact names: " + ", ".join(duplicate_names[:3])
            )
        folded = Counter(name.casefold() for name in names)
        player_case_collision_count = sum(1 for count in folded.values() if count > 1)
        player_name_count = len(names)

    record_keys: set[str] = set()
    if isinstance(team_records, dict) and team_records:
        record_values: list[str] = []
        for display_name, record in team_records.items():
            if not isinstance(record, dict):
                raise RefreshValidationError("team records contain a non-object record")
            key = str(record.get("team_key") or "").strip()
            if not key or not str(display_name).strip():
                raise RefreshValidationError("team records contain an empty canonical identity")
            record_values.append(key)
        if len(record_values) != len(set(record_values)):
            raise RefreshValidationError("team records contain duplicate canonical team keys")
        record_keys = set(record_values)
    if team_keys and record_keys and team_keys != record_keys:
        raise RefreshValidationError("team ratings and team records use different canonical teams")

    if isinstance(profile_records, dict) and profile_records.get("games"):
        games = profile_records["games"]
        if not isinstance(games, dict):
            raise RefreshValidationError("profile records have an invalid games index")
        for key, game in games.items():
            if not isinstance(game, dict) or str(game.get("game_id") or "") != str(key):
                raise RefreshValidationError("profile records contain a malformed game identity")
            rows = game.get("players")
            if not isinstance(rows, list) or len(rows) != 10:
                raise RefreshValidationError(f"profile game {key} does not contain ten players")
            names = [str(row.get("player") or "").strip() for row in rows if isinstance(row, dict)]
            if len(names) != 10 or any(not name for name in names) or len(set(names)) != 10:
                raise RefreshValidationError(f"profile game {key} contains malformed player identities")
            expected_roles = {"top", "jungle", "mid", "bot", "support"}
            side_roles: dict[str, set[str]] = {"Blue": set(), "Red": set()}
            champions: list[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    raise RefreshValidationError(f"profile game {key} contains a malformed player")
                side = str(row.get("side") or "").strip().title()
                role = str(row.get("role") or "").strip().casefold()
                champion = str(row.get("champion") or "").strip().casefold()
                if side not in side_roles or role not in expected_roles or not champion:
                    raise RefreshValidationError(f"profile game {key} has an incomplete pick identity")
                side_roles[side].add(role)
                champions.append(champion)
            if any(roles != expected_roles for roles in side_roles.values()) or len(set(champions)) != 10:
                raise RefreshValidationError(f"profile game {key} has an invalid ten-pick identity")
            signal = game.get("draft_contribution")
            if signal is not None:
                try:
                    validate_public_signal(signal, game)
                except CompositionSignalError as error:
                    raise RefreshValidationError(
                        f"profile game {key} has invalid composition evidence: {error}"
                    ) from error

    return {
        "team_key_count": len(team_keys),
        "player_name_count": player_name_count,
        "player_case_collision_count": player_case_collision_count,
    }


def validate_match_archive(pack_dir: Path) -> dict[str, int]:
    """Reject missing, duplicate, or incomplete maps in the public archive."""

    index = _read_pack_json(pack_dir, "features/match_index.json")
    rows = index.get("games") if isinstance(index, dict) else None
    if not isinstance(rows, list):
        raise RefreshValidationError("match archive index is malformed")
    index_ids = [str(row.get("game_id") or "") for row in rows if isinstance(row, dict)]
    if len(index_ids) != len(rows) or any(not game_id for game_id in index_ids):
        raise RefreshValidationError("match archive index contains an empty identity")
    if len(index_ids) != len(set(index_ids)):
        raise RefreshValidationError("match archive index contains duplicate identities")

    detail_ids: set[str] = set()
    year_counts: dict[str, int] = {}
    for year in (2025, 2026):
        year_count = 0
        for quarter in (1, 2, 3, 4):
            payload = _read_pack_json(pack_dir, f"features/match_records_{year}_q{quarter}.json")
            games = payload.get("games") if isinstance(payload, dict) else None
            if not isinstance(games, dict):
                raise RefreshValidationError(f"{year} Q{quarter} match archive is malformed")
            for game_id, game in games.items():
                if game_id in detail_ids:
                    raise RefreshValidationError("match archive details contain duplicate identities")
                if not isinstance(game, dict) or str(game.get("game_id") or "") != str(game_id):
                    raise RefreshValidationError(f"match archive game {game_id} has a malformed identity")
                if not str(game.get("date") or "").startswith(f"{year}-"):
                    raise RefreshValidationError(f"match archive game {game_id} is in the wrong year")
                if not profile_game_has_complete_stats(game):
                    raise RefreshValidationError(f"match archive game {game_id} has incomplete KDA")
                signal = game.get("draft_contribution")
                if signal is not None:
                    try:
                        validate_public_signal(signal, game)
                    except CompositionSignalError as error:
                        raise RefreshValidationError(
                            f"match archive game {game_id} has invalid composition evidence: {error}"
                        ) from error
                detail_ids.add(str(game_id))
                year_count += 1
        year_counts[str(year)] = year_count
    if set(index_ids) != detail_ids:
        raise RefreshValidationError("match archive index and detail files contain different maps")
    return {"maps": len(detail_ids), **year_counts}


def validate_pack(pack_dir: Path, manifest: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Verify the compact ratings pack and its source identity binding."""

    files = manifest.get("files")
    if not isinstance(files, list):
        raise RefreshValidationError("pack manifest has no file inventory")
    paths = {str(item.get("path")) for item in files if isinstance(item, dict)}
    required_paths = set(pack_spec.PUBLIC_RATING_REQUIRED_FILES)
    optional_paths = set(pack_spec.OPTIONAL_PUBLIC_FILES)
    if not required_paths.issubset(paths) or not paths.issubset(required_paths | optional_paths):
        raise RefreshValidationError("pack inventory differs from the public ratings contract")
    total_bytes = 0
    root = pack_dir.resolve()
    for item in files:
        relative = Path(str(item["path"]))
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RefreshValidationError(f"pack path leaves its directory: {relative}") from error
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            raise RefreshValidationError(f"pack file is missing or has the wrong size: {relative}")
        if _sha256(path) != item.get("sha256"):
            raise RefreshValidationError(f"pack checksum mismatch: {relative}")
        if str(relative) in required_paths:
            public_text = path.read_text(encoding="utf-8").casefold()
            if "los ratones" in public_text:
                raise RefreshValidationError(f"excluded team affiliation appears in public pack: {relative}")
            if "oracle_elixir_api" in public_text or "public_datalisk_api" in public_text:
                raise RefreshValidationError(f"transport label appears as a public league: {relative}")
        total_bytes += path.stat().st_size
    ratings = manifest.get("ratings") or {}
    if ratings.get("source_game_count") != source["game_count"]:
        raise RefreshValidationError("pack source game count does not match the live source")
    if ratings.get("source_identity_sha256") != source["identity_sha256"]:
        raise RefreshValidationError("pack source identity digest does not match the live source")
    if manifest.get("total_files") != len(files) or manifest.get("total_bytes") != total_bytes:
        raise RefreshValidationError("pack inventory totals are invalid")
    identity = validate_public_identity(pack_dir)
    match_archive = validate_match_archive(pack_dir)
    return {
        "files": len(files),
        "bytes": total_bytes,
        "identity": identity,
        "match_archive": match_archive,
        **source,
    }


def rollback_public_pack(publication: dict[str, Any], public_root: Path) -> dict[str, Any]:
    """Restore the last known-good local staging pointer."""

    restored = {"runtime": publication.get("runtime"), "pack_id": publication.get("pack_id")}
    old_local = publication.get("previous_local_manifest")
    if isinstance(old_local, dict):
        _atomic_json(public_root / "manifest.json", old_local)
        restored["local"] = True
    return restored


def publish_pack(pack_dir: Path, manifest: dict[str, Any], public_root: Path) -> dict[str, Any]:
    """Stage an immutable local candidate for private Supabase publication."""

    pack_id = str(manifest.get("pack_id") or "")
    if not pack_id or not pack_dir.is_dir():
        raise RefreshValidationError("pack publication has no valid source directory")
    candidate_root = public_root.expanduser().resolve()
    try:
        candidate_root.relative_to(STATIC_APP_PUBLIC_ROOT)
    except ValueError:
        pass
    else:
        raise RefreshValidationError(
            "candidate staging cannot write into the web app public directory"
        )
    public_root = candidate_root
    public_root.mkdir(parents=True, exist_ok=True)
    destination = public_root / pack_id
    if destination.exists():
        raise FileExistsError(f"immutable pack already exists: {destination}")
    staging = public_root / f".{pack_id}.{uuid.uuid4().hex}.incoming"
    old_local_manifest = _load_json(public_root / "manifest.json")
    published = dict(manifest)
    published["base_url"] = f"/packs/{pack_id}"
    try:
        shutil.copytree(pack_dir, staging)
        _atomic_json(staging / "manifest.json", published)
        os.replace(staging, destination)
        _atomic_json(public_root / "manifest.json", published)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if old_local_manifest:
            _atomic_json(public_root / "manifest.json", old_local_manifest)
        else:
            (public_root / "manifest.json").unlink(missing_ok=True)
        raise
    return {
        "pack_id": pack_id,
        "destination": str(destination),
        "runtime": "local_staging",
        "base_url": published["base_url"],
        "previous_local_manifest": old_local_manifest,
    }


def _write_health(config: SyncConfig, status: str, now: datetime, **fields: Any) -> None:
    _atomic_json(config.health_path, {"status": status, "checked_at": _iso(now), **fields})


def sync_once(
    config: SyncConfig,
    *,
    now: datetime | None = None,
    ingest_fn: Callable[..., dict[str, Any]] = ingest_oe_csv,
    build_live_fn: Callable[..., dict[str, Any]] = build_live_source,
    export_pack_fn: Callable[..., dict[str, Any]] = export_public_pack,
    validate_live_fn: Callable[..., dict[str, Any]] = validate_live_source,
    validate_pack_fn: Callable[..., dict[str, Any]] = validate_pack,
    publish_pack_fn: Callable[..., dict[str, Any]] = publish_pack,
    force: bool = False,
) -> dict[str, Any]:
    checked_at = now or datetime.now(timezone.utc)
    state = _load_json(config.state_path)
    baseline_ids, published_source = _continuity_baseline(config, state)
    known = set(baseline_ids)
    _write_health(config, "checking", checked_at)
    try:
        ingest_result = ingest_fn(
            config.data_root,
            years=config.years,
            source_receipt=config.accepted_source_receipt,
            import_receipt=config.accepted_import_receipt,
            start=pd.Timestamp(checked_at - timedelta(hours=config.window_hours)),
            end=pd.Timestamp(checked_at),
            lookback_days=config.lookback_days,
            discovery_cache_hours=config.discovery_cache_hours,
            max_workers=config.max_workers,
        )
        source_observed_through = (
            str(ingest_result.get("source_latest"))
            if isinstance(ingest_result, dict) and ingest_result.get("source_latest")
            else None
        )
        accepted_import = ingest_result.get("accepted_import_receipt")
        corrected_game_count = (
            int(accepted_import.get("corrected_games", 0))
            if isinstance(accepted_import, dict)
            else 0
        )
        accepted_source = ingest_result.get("accepted_source_receipt")
        source_file_sha256 = (
            str(accepted_import.get("source_file_sha256") or "")
            if isinstance(accepted_import, dict)
            else ""
        ) or (
            str(accepted_source.get("raw_sha256") or "")
            if isinstance(accepted_source, dict)
            else ""
        )
        source_changed_since_publish = bool(
            source_file_sha256
            and state.get("source_file_sha256") != source_file_sha256
        )
        observed = set(_source_game_ids(config.data_root))
        validate_source_continuity(
            baseline_ids,
            {
                "game_ids": sorted(observed),
                "game_count": len(observed),
                "identity_sha256": source_identity_sha256(sorted(observed)),
            },
            published_source,
        )
        raw_discovered_ids = sorted(observed - known)
        if (
            not raw_discovered_ids
            and not force
            and corrected_game_count == 0
            and not source_changed_since_publish
        ):
            result = {
                "status": "no_change",
                "new_game_ids": [],
                "pack_id": state.get("pack_id"),
                "source_observed_through": source_observed_through,
            }
            _atomic_json(config.state_path, {**state, **result, "published_game_ids": sorted(known)})
            _write_health(config, "ok", checked_at, pack_id=result["pack_id"])
            return result
        build_live_fn(config.data_root)
        candidate_source = validate_live_fn(config.data_root, [])
        candidate_continuity = validate_source_continuity(
            baseline_ids, candidate_source, published_source
        )
        candidate_ids = set(candidate_source.get("game_ids", []))
        discovered_ids = sorted(candidate_ids - known)
        statistics_complete = set(candidate_source.get("statistics_complete_game_ids", []))
        new_ids = sorted(set(discovered_ids).intersection(statistics_complete))
        pending_ids = sorted(set(discovered_ids).difference(statistics_complete))
        if (
            not new_ids
            and not force
            and corrected_game_count == 0
            and not source_changed_since_publish
        ):
            status = "waiting_for_details" if pending_ids else "no_change"
            result = {
                "status": status,
                "new_game_ids": [],
                "pending_game_ids": pending_ids,
                "pack_id": state.get("pack_id"),
                "source_observed_through": source_observed_through,
            }
            _atomic_json(config.state_path, {**state, **result, "published_game_ids": sorted(known)})
            _write_health(config, status if status != "no_change" else "ok", checked_at, pending_game_ids=pending_ids)
            return result
        source = validate_live_fn(config.data_root, new_ids)
        publish_ids = sorted(
            set(baseline_ids)
            .difference(candidate_continuity["quarantined_game_ids"])
            .union(new_ids)
        )
        source_ids = set(source["game_ids"])
        if not set(publish_ids).issubset(source_ids):
            raise RefreshValidationError("accepted OE source is missing a publishable completed map")
        source = {
            **source,
            "game_ids": publish_ids,
            "game_count": len(publish_ids),
            "identity_sha256": source_identity_sha256(publish_ids),
        }
        continuity = validate_source_continuity(baseline_ids, source, published_source)
        pack_id = f"v{checked_at.strftime('%Y.%m.%d.%H%M%S')}"
        manifest = export_pack_fn(
            years=config.years,
            out_root=config.output_root,
            pack_id=pack_id,
            project_root=config.root,
            runtime_root=config.data_root,
            warehouse_root=config.data_root / "data/lol/warehouse/parquet/oe_live",
            allowed_game_ids=publish_ids,
        )
        files = manifest.get("files") if isinstance(manifest, dict) else None
        artifact_hashes = {
            str(item["path"]): str(item["sha256"])
            for item in files or []
            if isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and isinstance(item.get("sha256"), str)
        }
        source_receipt_sha256 = (
            accepted_source.get("receipt_canonical_sha256")
            if isinstance(accepted_source, dict)
            else None
        )
        accepted_import = ingest_result.get("accepted_import_receipt")
        import_counts = accepted_import if isinstance(accepted_import, dict) else {}
        manifest["release"] = {
            "release_id": pack_id,
            "worker_commit": worker_commit(config.root),
            "source_file_hashes": _source_file_hashes(config.data_root),
            "source_receipt_sha256": source_receipt_sha256,
            "canonical_game_identity_digest": source["identity_sha256"],
            "source_watermark": source_observed_through,
            "transform_version": TRANSFORM_VERSION,
            "ratings_version": "scryglass:dual-elo-public:v1",
            "tier_list_version": None,
            "artifact_hashes": artifact_hashes,
            "accepted_games": int(import_counts.get("accepted_games", source["game_count"])),
            "new_games": int(import_counts.get("new_games", len(new_ids))),
            "corrected_games": int(import_counts.get("corrected_games", 0)),
            "unchanged_games": int(
                import_counts.get("unchanged_games", max(0, len(baseline_ids) - len(new_ids)))
            ),
            "quarantined_games": int(
                import_counts.get("quarantined_games", len(continuity["quarantined_game_ids"]))
            ),
        }
        _atomic_json(config.output_root / pack_id / "manifest.json", manifest)
        validation = validate_pack_fn(config.output_root / pack_id, manifest, source)
        publication = publish_pack_fn(config.output_root / pack_id, manifest, config.public_root)
        validation["continuity"] = continuity
        result = {
            "status": "published",
            "new_game_ids": new_ids,
            "pending_game_ids": pending_ids,
            "pack_id": pack_id,
            "source_observed_through": source_observed_through,
            "source_file_sha256": source_file_sha256 or None,
            "validation": validation,
            "publication": publication,
        }
        _atomic_json(config.state_path, {**result, "published_game_ids": source["game_ids"]})
        _write_health(config, "ok", checked_at, pack_id=pack_id, new_game_ids=new_ids, pending_game_ids=pending_ids)
        return result
    except Exception as error:
        _write_health(config, "error", checked_at, pack_id=state.get("pack_id"), reason=f"{type(error).__name__}: {str(error)[:500]}")
        raise


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SyncAlreadyRunning(f"another ratings sync owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public-root", type=Path, default=Path("/srv/scryglass-data/public-packs"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true", help="publish a validated pack even when no new game ID appears")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = SyncConfig(
        root=root,
        public_root=args.public_root.resolve(),
        output_root=(root / "output/public_pack").resolve(),
        state_path=(root / "data/lol/runtime/postgame-sync.json").resolve(),
        lock_path=(root / "data/lol/runtime/postgame-sync.lock").resolve(),
        health_path=(root / "data/lol/runtime/postgame-sync-health.json").resolve(),
    )
    with exclusive_lock(config.lock_path):
        result = sync_once(config, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
