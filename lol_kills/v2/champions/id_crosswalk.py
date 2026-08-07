"""Strict, source-backed champion vocabulary identity for model-v2.

This module maps the exact Oracle's Elixir champion strings observed in the
current interaction-preflight population to Riot Data Dragon numeric champion
keys.  It does not map competition patches, estimate champion effects, select a
model, or authorize prediction/publication.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import unicodedata
import weakref
from collections.abc import Iterator, Mapping as RuntimeMapping
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCHEMA_ID = "scryglass.champion-id-crosswalk.v1"
GENERATOR_VERSION = "champion-id-crosswalk-generator.v1"
METADATA_VERSION = "16.14.1"
METADATA_URL = (
    "https://ddragon.leagueoflegends.com/cdn/16.14.1/data/en_US/champion.json"
)
RIOT_DATA_DRAGON_DOCS_URL = (
    "https://developer.riotgames.com/docs/lol#data-dragon_champions"
)
EXPECTED_METADATA_ENTRIES = 173
EXPECTED_PREFLIGHT_PAYLOAD_SHA256 = (
    "ba54faed41716cc537268c6e7eecbaaf9330937014bfd2cd5f9a50f930f92eb4"
)
EXPECTED_METADATA_RAW_SHA256 = (
    "19717cae1dd13aa448c6be423723ee57d787b34161e843a82c9ca03100dc9220"
)
EXPECTED_MAPS_RAW_SHA256 = (
    "04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08"
)
EXPECTED_PLAYERS_RAW_SHA256 = (
    "3d2a852daa43dfa402e1e48ef11d1a6858b73f2171f0c2febd82b941b19fceee"
)
EXPECTED_MAPS = 12_708
EXPECTED_ROLE_SLOTS = 127_080
EXPECTED_OE_NAMES = 171
EXPECTED_ACTION_ORDER_COMPLETE = 12_631
EXPECTED_ACTION_ORDER_MISSING = 77

DEFAULT_METADATA_PATH = Path(
    "data/lol/v2/champions/sources/riot-champion-metadata-16.14.1.json"
)
DEFAULT_PREFLIGHT_PATH = Path(
    "data/lol/v2/models/draft-interactions/representation-assay-preflight.json"
)
DEFAULT_MAPS_PATH = Path("data/lol/warehouse/parquet/maps.parquet")
DEFAULT_PLAYERS_PATH = Path("data/lol/warehouse/parquet/oe_player_games.parquet")
DEFAULT_ARTIFACT_PATH = Path(
    "data/lol/v2/champions/champion-id-crosswalk-v1.json"
)
MODULE_LOCATOR = "lol_kills/v2/champions/id_crosswalk.py"
ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
ROLE_ALIASES = {
    "top": "top",
    "jng": "jungle",
    "jungle": "jungle",
    "mid": "mid",
    "bot": "bot",
    "support": "support",
    "sup": "support",
}
SIDE_ALIASES = {"blue": "blue", "red": "red"}
ACTION_ORDER_COLUMNS = tuple(
    f"{side}_pick{number}"
    for side in ("blue", "red")
    for number in range(1, 6)
)
PLAYER_COLUMNS = ("gameid", "side", "position", "champion")
MAP_COLUMNS = ("oe_gameid", *ACTION_ORDER_COLUMNS)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "`": "'"})
_WHITESPACE = re.compile(r"\s+")
_HEX = frozenset("0123456789abcdef")
_TOP_LEVEL_FIELDS = {
    "schema_id",
    "artifact_sha256",
    "publication_decision",
    "development_only",
    "authority",
    "claim_scope",
    "normalization",
    "metadata",
    "competition_patch",
    "preflight",
    "warehouse_sources",
    "generator",
    "explicit_aliases",
    "entries",
    "coverage",
    "action_order_availability",
}


class ChampionIdCrosswalkError(ValueError):
    """Raised when champion identity cannot be established exactly."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: object) -> str:
    """Apply the complete and deliberately narrow vocabulary normalization."""
    if not isinstance(value, str):
        raise ChampionIdCrosswalkError("champion name must be text")
    normalized = unicodedata.normalize("NFKC", value)
    normalized = normalized.strip().casefold().translate(_APOSTROPHES)
    normalized = _WHITESPACE.sub(" ", normalized)
    if not normalized:
        raise ChampionIdCrosswalkError("champion name is blank")
    return normalized


def _require_sha256(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise ChampionIdCrosswalkError(f"{label} must be a lowercase sha256")
    return value


def _require_regular_file(path: Path, label: str) -> None:
    if not path.is_file() or path.is_symlink():
        raise ChampionIdCrosswalkError(
            f"{label} must be a regular, non-symlink source file"
        )


def _generator_identity() -> dict[str, object]:
    module_path = Path(__file__).resolve()
    return {
        "version": GENERATOR_VERSION,
        "executable_dependency_boundary": [
            {"locator": MODULE_LOCATOR, "raw_sha256": raw_sha256(module_path)}
        ],
        "identity_scope": (
            "the exact generator module bytes; metadata, preflight, and parquet "
            "source bytes are pinned separately"
        ),
    }


def load_metadata_bytes(raw: bytes) -> dict[str, dict[str, object]]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChampionIdCrosswalkError("Riot metadata is not valid JSON") from exc
    if not isinstance(payload, dict) or set(payload) < {"version", "data"}:
        raise ChampionIdCrosswalkError("Riot metadata shape is invalid")
    if payload["version"] != METADATA_VERSION:
        raise ChampionIdCrosswalkError("Riot metadata version is not 16.14.1")
    data = payload["data"]
    if not isinstance(data, dict) or len(data) != EXPECTED_METADATA_ENTRIES:
        raise ChampionIdCrosswalkError(
            f"Riot metadata must contain exactly {EXPECTED_METADATA_ENTRIES} champions"
        )

    numeric_ids: set[int] = set()
    internal_ids: set[str] = set()
    display_ids: set[str] = set()
    result: dict[str, dict[str, object]] = {}
    for dictionary_id, record in data.items():
        if not isinstance(dictionary_id, str) or not isinstance(record, dict):
            raise ChampionIdCrosswalkError("Riot champion record is malformed")
        if set(record) < {"id", "key", "name"}:
            raise ChampionIdCrosswalkError("Riot champion identity fields are missing")
        internal_id = record["id"]
        display_name = record["name"]
        numeric_text = record["key"]
        if (
            not isinstance(internal_id, str)
            or not internal_id
            or internal_id != dictionary_id
            or not isinstance(display_name, str)
            or not display_name.strip()
            or isinstance(numeric_text, bool)
            or not isinstance(numeric_text, str)
            or not numeric_text.isdecimal()
        ):
            raise ChampionIdCrosswalkError("Riot champion identity field is invalid")
        numeric_id = int(numeric_text)
        if numeric_id <= 0:
            raise ChampionIdCrosswalkError("Riot champion numeric ID must be positive")
        if numeric_id >= 1000:
            raise ChampionIdCrosswalkError(
                "Riot metadata contains a non-base or mode-variant champion ID"
            )
        if numeric_id in numeric_ids:
            raise ChampionIdCrosswalkError("duplicate Riot champion numeric ID")
        if internal_id in internal_ids:
            raise ChampionIdCrosswalkError("duplicate Riot champion internal ID")
        normalized_display = normalize_name(display_name)
        if normalized_display in display_ids:
            raise ChampionIdCrosswalkError(
                "duplicate normalized Riot champion display name"
            )
        numeric_ids.add(numeric_id)
        internal_ids.add(internal_id)
        display_ids.add(normalized_display)
        result[internal_id] = {
            "internal_id": internal_id,
            "display_name": display_name,
            "numeric_id": numeric_id,
            "stable_champion_id": f"riot:champion:{numeric_id}",
        }
    return result


def _metadata_vocabulary(
    metadata: Mapping[str, Mapping[str, object]],
) -> tuple[dict[str, str], list[dict[str, str]]]:
    vocabulary: dict[str, str] = {}
    aliases: list[dict[str, str]] = []

    def add(value: str, internal_id: str, kind: str) -> None:
        key = normalize_name(value)
        existing = vocabulary.get(key)
        if existing is not None and existing != internal_id:
            raise ChampionIdCrosswalkError(
                f"normalized metadata vocabulary collision for {value!r}"
            )
        vocabulary[key] = internal_id
        if kind == "source_internal_id" and normalize_name(
            str(metadata[internal_id]["display_name"])
        ) != key:
            aliases.append(
                {
                    "input": value,
                    "normalized_input": key,
                    "riot_internal_id": internal_id,
                    "basis": "Riot Data Dragon internal ID differs from display name",
                }
            )

    for internal_id in sorted(metadata):
        record = metadata[internal_id]
        add(str(record["display_name"]), internal_id, "source_display_name")
        add(internal_id, internal_id, "source_internal_id")
    aliases.sort(key=lambda item: (item["normalized_input"], item["riot_internal_id"]))
    return vocabulary, aliases


def _validate_preflight(payload: object) -> dict[str, object]:
    if not isinstance(payload, dict):
        raise ChampionIdCrosswalkError("preflight artifact must be an object")
    submitted = payload.get("artifact_sha256")
    if submitted != EXPECTED_PREFLIGHT_PAYLOAD_SHA256:
        raise ChampionIdCrosswalkError("preflight payload sha256 is not pinned")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    if canonical_sha256(unsigned) != submitted:
        raise ChampionIdCrosswalkError("preflight payload sha256 is invalid")
    if payload.get("eligibility", {}).get("valid_maps") != EXPECTED_MAPS:
        raise ChampionIdCrosswalkError("preflight map population changed")
    source = payload.get("source")
    if not isinstance(source, dict):
        raise ChampionIdCrosswalkError("preflight source manifest is missing")
    if source.get("maps", {}).get("raw_sha256") != EXPECTED_MAPS_RAW_SHA256:
        raise ChampionIdCrosswalkError("preflight maps source hash changed")
    if (
        source.get("player_games", {}).get("raw_sha256")
        != EXPECTED_PLAYERS_RAW_SHA256
    ):
        raise ChampionIdCrosswalkError("preflight player source hash changed")
    return payload


def _role(value: object) -> str:
    if not isinstance(value, str):
        raise ChampionIdCrosswalkError("role is not text")
    role = ROLE_ALIASES.get(value.strip().casefold())
    if role is None:
        raise ChampionIdCrosswalkError("role is outside the five-role contract")
    return role


def _side(value: object) -> str:
    if not isinstance(value, str):
        raise ChampionIdCrosswalkError("side is not text")
    side = SIDE_ALIASES.get(value.strip().casefold())
    if side is None:
        raise ChampionIdCrosswalkError("side is outside the blue/red contract")
    return side


def _is_present(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def build_artifact(
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    preflight_path: Path = DEFAULT_PREFLIGHT_PATH,
    maps_path: Path = DEFAULT_MAPS_PATH,
    players_path: Path = DEFAULT_PLAYERS_PATH,
    metadata_locator: str | None = None,
    preflight_locator: str | None = None,
    maps_locator: str | None = None,
    players_locator: str | None = None,
) -> dict[str, object]:
    for path, label in (
        (metadata_path, "Riot metadata"),
        (preflight_path, "preflight"),
        (maps_path, "maps parquet"),
        (players_path, "player-games parquet"),
    ):
        _require_regular_file(path, label)

    if raw_sha256(metadata_path) != EXPECTED_METADATA_RAW_SHA256:
        raise ChampionIdCrosswalkError("Riot metadata source bytes changed")
    if raw_sha256(maps_path) != EXPECTED_MAPS_RAW_SHA256:
        raise ChampionIdCrosswalkError("maps parquet source bytes changed")
    if raw_sha256(players_path) != EXPECTED_PLAYERS_RAW_SHA256:
        raise ChampionIdCrosswalkError("player-games parquet source bytes changed")

    metadata = load_metadata_bytes(metadata_path.read_bytes())
    vocabulary, explicit_aliases = _metadata_vocabulary(metadata)
    try:
        preflight = json.loads(preflight_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChampionIdCrosswalkError("preflight artifact is not valid JSON") from exc
    _validate_preflight(preflight)

    maps = pd.read_parquet(maps_path, columns=list(MAP_COLUMNS))
    players = pd.read_parquet(players_path, columns=list(PLAYER_COLUMNS))
    if len(maps) != EXPECTED_MAPS or maps["oe_gameid"].duplicated().any():
        raise ChampionIdCrosswalkError("maps parquet no longer matches preflight registry")
    game_ids = set(maps["oe_gameid"].astype(str))
    joined = players.loc[players["gameid"].astype(str).isin(game_ids)].copy()
    if len(joined) != EXPECTED_ROLE_SLOTS:
        raise ChampionIdCrosswalkError("joined role-slot population changed")

    slot_keys: list[tuple[str, str, str]] = []
    mapped_rows: list[tuple[str, str]] = []
    oe_to_internal: dict[str, str] = {}
    normalized_to_oe: dict[str, str] = {}
    for row in joined.itertuples(index=False):
        game_id = str(row.gameid)
        side = _side(row.side)
        role = _role(row.position)
        champion = row.champion
        normalized = normalize_name(champion)
        internal_id = vocabulary.get(normalized)
        if internal_id is None:
            raise ChampionIdCrosswalkError(
                f"unknown or future champion name fails closed: {champion!r}"
            )
        prior_oe = normalized_to_oe.get(normalized)
        if prior_oe is not None and prior_oe != champion:
            raise ChampionIdCrosswalkError(
                "multiple OE spellings collapse to the same normalized champion name"
            )
        prior_internal = oe_to_internal.get(champion)
        if prior_internal is not None and prior_internal != internal_id:
            raise ChampionIdCrosswalkError("OE champion maps to multiple Riot IDs")
        normalized_to_oe[normalized] = champion
        oe_to_internal[champion] = internal_id
        slot_keys.append((game_id, side, role))
        mapped_rows.append((champion, internal_id))

    slot_counts = Counter(game_id for game_id, _, _ in slot_keys)
    if (
        len(set(slot_keys)) != EXPECTED_ROLE_SLOTS
        or set(slot_counts) != game_ids
        or any(count != 10 for count in slot_counts.values())
    ):
        raise ChampionIdCrosswalkError(
            "each preflight map must have ten unique side-role identity slots"
        )
    if len(oe_to_internal) != EXPECTED_OE_NAMES:
        raise ChampionIdCrosswalkError("distinct OE champion vocabulary changed")

    entries = []
    for oe_name in sorted(oe_to_internal, key=lambda value: (normalize_name(value), value)):
        record = metadata[oe_to_internal[oe_name]]
        entries.append(
            {
                "oe_name": oe_name,
                "normalized_oe_name": normalize_name(oe_name),
                "riot_internal_id": record["internal_id"],
                "riot_display_name": record["display_name"],
                "riot_numeric_id": record["numeric_id"],
                "stable_champion_id": record["stable_champion_id"],
            }
        )
    stable_ids = [entry["stable_champion_id"] for entry in entries]
    if len(stable_ids) != len(set(stable_ids)):
        raise ChampionIdCrosswalkError("OE vocabulary has a stable-ID collision")

    action_complete = maps.loc[:, list(ACTION_ORDER_COLUMNS)].apply(
        lambda column: column.map(_is_present)
    ).all(axis=1)
    complete_count = int(action_complete.sum())
    missing_count = int((~action_complete).sum())
    if (
        complete_count != EXPECTED_ACTION_ORDER_COMPLETE
        or missing_count != EXPECTED_ACTION_ORDER_MISSING
    ):
        raise ChampionIdCrosswalkError("action-order availability changed")

    payload: dict[str, object] = {
        "schema_id": SCHEMA_ID,
        "publication_decision": "private_pending_review",
        "development_only": True,
        "authority": {
            "authorizes_prediction": False,
            "authorizes_model_selection": False,
            "authorizes_publication": False,
            "content_addressing_confers_authority": False,
        },
        "claim_scope": {
            "vocabulary_identity_only": True,
            "effect_estimation": False,
            "champion_similarity": False,
            "competition_patch_mapping": False,
        },
        "normalization": {
            "ordered_operations": [
                "Unicode NFKC",
                "strip leading and trailing whitespace",
                "Unicode casefold",
                "map curly and backtick apostrophes to ASCII apostrophe",
                "collapse whitespace runs to one ASCII space",
            ],
            "punctuation_stripping": False,
            "fuzzy_matching": False,
            "partial_matching": False,
        },
        "metadata": {
            "namespace": "Riot Data Dragon metadata version",
            "version": METADATA_VERSION,
            "metadata_version": METADATA_VERSION,
            "source_url": METADATA_URL,
            "documentation_url": RIOT_DATA_DRAGON_DOCS_URL,
            "locator": (
                metadata_locator
                if metadata_locator is not None
                else metadata_path.as_posix()
            ),
            "raw_sha256": EXPECTED_METADATA_RAW_SHA256,
            "base_champion_entries": EXPECTED_METADATA_ENTRIES,
            "numeric_id_constraint": "positive integer less than 1000",
        },
        "competition_patch": {
            "namespace": "Oracle's Elixir source patch token",
            "competition_patch_namespace": "Oracle's Elixir source patch token",
            "official_mapping_status": "none",
            "patch_mapping": "none",
            "exact_patch_authority": False,
            "metadata_version_is_competition_patch": False,
            "note": (
                "Data Dragon 16.14.1 is a metadata vocabulary version. No 16.x to "
                "26.x or float-token to official-patch mapping is inferred."
            ),
        },
        "preflight": {
            "locator": (
                preflight_locator
                if preflight_locator is not None
                else preflight_path.as_posix()
            ),
            "payload_sha256": EXPECTED_PREFLIGHT_PAYLOAD_SHA256,
            "valid_maps": EXPECTED_MAPS,
        },
        "warehouse_sources": {
            "maps": {
                "locator": maps_locator if maps_locator is not None else maps_path.as_posix(),
                "raw_sha256": EXPECTED_MAPS_RAW_SHA256,
                "columns_read": list(MAP_COLUMNS),
            },
            "player_games": {
                "locator": (
                    players_locator
                    if players_locator is not None
                    else players_path.as_posix()
                ),
                "raw_sha256": EXPECTED_PLAYERS_RAW_SHA256,
                "columns_read": list(PLAYER_COLUMNS),
            },
        },
        "generator": _generator_identity(),
        "explicit_aliases": explicit_aliases,
        "entries": entries,
        "coverage": {
            "preflight_maps": EXPECTED_MAPS,
            "role_labeled_slots": EXPECTED_ROLE_SLOTS,
            "maps_with_ten_resolved_role_slots": EXPECTED_MAPS,
            "distinct_oe_names": EXPECTED_OE_NAMES,
            "distinct_oe_names_resolved": EXPECTED_OE_NAMES,
            "unresolved_oe_names": [],
        },
        "action_order_availability": {
            "constraint_stage": "later draft-protocol reconstruction",
            "identity_coverage_affected": False,
            "columns": list(ACTION_ORDER_COLUMNS),
            "complete_maps": complete_count,
            "maps_missing_one_or_more_action_order_fields": missing_count,
            "total_maps": EXPECTED_MAPS,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: Mapping[str, object]) -> None:
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ChampionIdCrosswalkError("crosswalk top-level fields are not exact")
    if payload.get("schema_id") != SCHEMA_ID:
        raise ChampionIdCrosswalkError("crosswalk schema_id mismatch")
    if (
        payload.get("publication_decision") != "private_pending_review"
        or payload.get("development_only") is not True
    ):
        raise ChampionIdCrosswalkError("crosswalk publication ceiling is invalid")
    if payload.get("authority") != {
        "authorizes_prediction": False,
        "authorizes_model_selection": False,
        "authorizes_publication": False,
        "content_addressing_confers_authority": False,
    }:
        raise ChampionIdCrosswalkError("crosswalk authority exceeds vocabulary scope")
    if payload.get("claim_scope") != {
        "vocabulary_identity_only": True,
        "effect_estimation": False,
        "champion_similarity": False,
        "competition_patch_mapping": False,
    }:
        raise ChampionIdCrosswalkError("crosswalk claim scope is invalid")
    normalization = payload.get("normalization")
    if not isinstance(normalization, Mapping) or (
        normalization.get("punctuation_stripping") is not False
        or normalization.get("fuzzy_matching") is not False
        or normalization.get("partial_matching") is not False
    ):
        raise ChampionIdCrosswalkError("crosswalk normalization ceiling is invalid")
    submitted = _require_sha256(payload.get("artifact_sha256"), "artifact_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256")
    if canonical_sha256(unsigned) != submitted:
        raise ChampionIdCrosswalkError("artifact_sha256 does not match canonical payload")
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ChampionIdCrosswalkError("metadata manifest is missing")
    if (
        metadata.get("version") != METADATA_VERSION
        or metadata.get("metadata_version") != METADATA_VERSION
        or metadata.get("raw_sha256") != EXPECTED_METADATA_RAW_SHA256
        or metadata.get("base_champion_entries") != EXPECTED_METADATA_ENTRIES
    ):
        raise ChampionIdCrosswalkError("metadata manifest is not pinned")
    competition_patch = payload.get("competition_patch")
    if not isinstance(competition_patch, Mapping) or (
        competition_patch.get("official_mapping_status") != "none"
        or competition_patch.get("patch_mapping") != "none"
        or competition_patch.get("competition_patch_namespace")
        != "Oracle's Elixir source patch token"
        or competition_patch.get("exact_patch_authority") is not False
        or competition_patch.get("metadata_version_is_competition_patch") is not False
    ):
        raise ChampionIdCrosswalkError("metadata was promoted to competition-patch authority")
    preflight = payload.get("preflight")
    if (
        not isinstance(preflight, Mapping)
        or preflight.get("payload_sha256") != EXPECTED_PREFLIGHT_PAYLOAD_SHA256
        or preflight.get("valid_maps") != EXPECTED_MAPS
    ):
        raise ChampionIdCrosswalkError("preflight manifest is not pinned")
    sources = payload.get("warehouse_sources")
    if not isinstance(sources, Mapping):
        raise ChampionIdCrosswalkError("warehouse source manifest is missing")
    if sources.get("maps", {}).get("raw_sha256") != EXPECTED_MAPS_RAW_SHA256:
        raise ChampionIdCrosswalkError("maps source hash is not pinned")
    if (
        sources.get("player_games", {}).get("raw_sha256")
        != EXPECTED_PLAYERS_RAW_SHA256
    ):
        raise ChampionIdCrosswalkError("player source hash is not pinned")
    generator = payload.get("generator")
    if not isinstance(generator, Mapping) or set(generator) != {
        "version",
        "executable_dependency_boundary",
        "identity_scope",
    }:
        raise ChampionIdCrosswalkError("generator identity shape is invalid")
    if generator.get("version") != GENERATOR_VERSION:
        raise ChampionIdCrosswalkError("generator version mismatch")
    boundary = generator.get("executable_dependency_boundary")
    if (
        not isinstance(boundary, list)
        or len(boundary) != 1
        or not isinstance(boundary[0], Mapping)
        or boundary[0].get("locator") != MODULE_LOCATOR
    ):
        raise ChampionIdCrosswalkError("generator module identity is invalid")
    _require_sha256(boundary[0].get("raw_sha256"), "generator module raw_sha256")

    entries = payload.get("entries")
    if not isinstance(entries, list) or len(entries) != EXPECTED_OE_NAMES:
        raise ChampionIdCrosswalkError("crosswalk must contain exactly 170 OE names")
    normalized_names: set[str] = set()
    oe_names: set[str] = set()
    stable_ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != {
            "oe_name",
            "normalized_oe_name",
            "riot_internal_id",
            "riot_display_name",
            "riot_numeric_id",
            "stable_champion_id",
        }:
            raise ChampionIdCrosswalkError("crosswalk entry shape is invalid")
        oe_name = entry["oe_name"]
        normalized = normalize_name(oe_name)
        if normalized != entry["normalized_oe_name"]:
            raise ChampionIdCrosswalkError("crosswalk normalization is inconsistent")
        numeric_id = entry["riot_numeric_id"]
        if (
            isinstance(numeric_id, bool)
            or not isinstance(numeric_id, int)
            or numeric_id <= 0
            or numeric_id >= 1000
            or entry["stable_champion_id"] != f"riot:champion:{numeric_id}"
        ):
            raise ChampionIdCrosswalkError("crosswalk stable champion ID is invalid")
        if (
            oe_name in oe_names
            or normalized in normalized_names
            or entry["stable_champion_id"] in stable_ids
        ):
            raise ChampionIdCrosswalkError("crosswalk entry collision")
        oe_names.add(oe_name)
        normalized_names.add(normalized)
        stable_ids.add(entry["stable_champion_id"])
    aliases = payload.get("explicit_aliases")
    if not isinstance(aliases, list):
        raise ChampionIdCrosswalkError("explicit alias manifest is missing")
    alias_targets: dict[str, str] = {}
    normalized_entry_targets = {
        entry["normalized_oe_name"]: entry["riot_internal_id"] for entry in entries
    }
    for alias in aliases:
        if not isinstance(alias, Mapping) or set(alias) != {
            "input",
            "normalized_input",
            "riot_internal_id",
            "basis",
        }:
            raise ChampionIdCrosswalkError("explicit alias shape is invalid")
        normalized = normalize_name(alias["input"])
        internal_id = alias["riot_internal_id"]
        if (
            normalized != alias["normalized_input"]
            or alias["basis"]
            != "Riot Data Dragon internal ID differs from display name"
            or not isinstance(internal_id, str)
        ):
            raise ChampionIdCrosswalkError("explicit alias is not source-derived")
        existing = alias_targets.get(normalized)
        if existing is not None and existing != internal_id:
            raise ChampionIdCrosswalkError("explicit alias collision")
        entry_target = normalized_entry_targets.get(normalized)
        if entry_target is not None and entry_target != internal_id:
            raise ChampionIdCrosswalkError("explicit alias collides with OE vocabulary")
        alias_targets[normalized] = internal_id
    if payload.get("coverage") != {
        "preflight_maps": EXPECTED_MAPS,
        "role_labeled_slots": EXPECTED_ROLE_SLOTS,
        "maps_with_ten_resolved_role_slots": EXPECTED_MAPS,
        "distinct_oe_names": EXPECTED_OE_NAMES,
        "distinct_oe_names_resolved": EXPECTED_OE_NAMES,
        "unresolved_oe_names": [],
    }:
        raise ChampionIdCrosswalkError("crosswalk coverage arithmetic mismatch")
    action = payload.get("action_order_availability")
    if not isinstance(action, Mapping) or (
        action.get("complete_maps") != EXPECTED_ACTION_ORDER_COMPLETE
        or action.get("maps_missing_one_or_more_action_order_fields")
        != EXPECTED_ACTION_ORDER_MISSING
        or action.get("total_maps") != EXPECTED_MAPS
        or action.get("identity_coverage_affected") is not False
    ):
        raise ChampionIdCrosswalkError("action-order availability arithmetic mismatch")


def _derive_resolution_table(
    payload: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    """Derive the complete immutable name-to-ID table from a validated payload."""
    validate_artifact(payload)
    entries = payload["entries"]
    aliases = payload["explicit_aliases"]
    assert isinstance(entries, list)
    assert isinstance(aliases, list)

    by_internal_id = {
        str(entry["riot_internal_id"]): str(entry["stable_champion_id"])
        for entry in entries
    }
    table: dict[str, str] = {}

    def add(normalized_name: str, stable_id: str) -> None:
        prior = table.get(normalized_name)
        if prior is not None and prior != stable_id:
            raise ChampionIdCrosswalkError("champion resolution table collision")
        table[normalized_name] = stable_id

    for entry in entries:
        add(
            str(entry["normalized_oe_name"]),
            str(entry["stable_champion_id"]),
        )
    for alias in aliases:
        internal_id = str(alias["riot_internal_id"])
        stable_id = by_internal_id.get(internal_id)
        if stable_id is None:
            # The source manifest covers all Riot display/internal-name
            # differences, while this resolver intentionally covers only the
            # empirically observed OE vocabulary.
            continue
        add(str(alias["normalized_input"]), stable_id)
    return tuple(sorted(table.items()))


def require_exact_competition_patch_authority(
    payload: Mapping[str, object], patch_token: object
) -> None:
    """Fail because this vocabulary artifact never maps OE patch tokens."""
    validate_artifact(payload)
    _ = patch_token
    raise ChampionIdCrosswalkError(
        "champion vocabulary identity does not confer exact competition-patch authority"
    )


def write_artifact(
    path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    metadata_path: Path = DEFAULT_METADATA_PATH,
    preflight_path: Path = DEFAULT_PREFLIGHT_PATH,
    maps_path: Path = DEFAULT_MAPS_PATH,
    players_path: Path = DEFAULT_PLAYERS_PATH,
) -> dict[str, object]:
    payload = build_artifact(
        metadata_path=metadata_path,
        preflight_path=preflight_path,
        maps_path=maps_path,
        players_path=players_path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(payload))
    return payload


def _load_and_replay_payload(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    source_root: Path | None = None,
) -> tuple[
    dict[str, object],
    bytes,
    Path,
    tuple[tuple[str, Path, str], ...],
]:
    _require_regular_file(artifact_path, "persisted crosswalk artifact")
    try:
        persisted_bytes = artifact_path.read_bytes()
        payload = json.loads(persisted_bytes)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChampionIdCrosswalkError("cannot load persisted crosswalk artifact") from exc
    if not isinstance(payload, dict):
        raise ChampionIdCrosswalkError("persisted crosswalk must be an object")
    validate_artifact(payload)
    if persisted_bytes != canonical_bytes(payload):
        raise ChampionIdCrosswalkError("persisted crosswalk bytes are not canonical")
    if payload["generator"] != _generator_identity():
        raise ChampionIdCrosswalkError(
            "persisted generator identity does not match executable module"
        )
    root = source_root if source_root is not None else Path.cwd()

    def resolve(locator: object, label: str) -> Path:
        if not isinstance(locator, str) or not locator:
            raise ChampionIdCrosswalkError(f"{label} locator is invalid")
        candidate = Path(locator)
        path = candidate if candidate.is_absolute() else root / candidate
        _require_regular_file(path, label)
        return path

    metadata_path = resolve(payload["metadata"]["locator"], "Riot metadata")
    preflight_path = resolve(payload["preflight"]["locator"], "preflight")
    maps_path = resolve(payload["warehouse_sources"]["maps"]["locator"], "maps")
    players_path = resolve(
        payload["warehouse_sources"]["player_games"]["locator"], "player games"
    )
    for path, expected, label in (
        (metadata_path, EXPECTED_METADATA_RAW_SHA256, "Riot metadata"),
        (maps_path, EXPECTED_MAPS_RAW_SHA256, "maps"),
        (players_path, EXPECTED_PLAYERS_RAW_SHA256, "player games"),
    ):
        if raw_sha256(path) != expected:
            raise ChampionIdCrosswalkError(f"{label} pinned source bytes changed")
    replayed = build_artifact(
        metadata_path=metadata_path,
        preflight_path=preflight_path,
        maps_path=maps_path,
        players_path=players_path,
        metadata_locator=payload["metadata"]["locator"],
        preflight_locator=payload["preflight"]["locator"],
        maps_locator=payload["warehouse_sources"]["maps"]["locator"],
        players_locator=payload["warehouse_sources"]["player_games"]["locator"],
    )
    if canonical_bytes(replayed) != persisted_bytes:
        raise ChampionIdCrosswalkError(
            "persisted crosswalk does not match source-backed replay"
        )
    source_identities = (
        ("Riot metadata", metadata_path.resolve(), EXPECTED_METADATA_RAW_SHA256),
        ("preflight", preflight_path.resolve(), raw_sha256(preflight_path)),
        ("maps", maps_path.resolve(), EXPECTED_MAPS_RAW_SHA256),
        ("player games", players_path.resolve(), EXPECTED_PLAYERS_RAW_SHA256),
    )
    return (
        replayed,
        persisted_bytes,
        artifact_path.resolve(),
        source_identities,
    )


def _make_runtime_crosswalk_api():
    """Create the only replay-to-resolution path.

    There is deliberately no module-global issuer or live payload accessor.
    Each capability is backed by immutable canonical bytes and an immutable
    derived lookup table.  Resolution also checks that the canonical artifact
    and executable generator have not changed since replay.
    """

    RuntimeState = tuple[
        bytes,
        Path,
        tuple[tuple[str, str], ...],
        tuple[tuple[str, Path, str], ...],
    ]
    store: weakref.WeakKeyDictionary[object, RuntimeState] = (
        weakref.WeakKeyDictionary()
    )

    def require_state(value: object) -> RuntimeState:
        try:
            state = store[value]
        except (KeyError, TypeError) as exc:
            raise ChampionIdCrosswalkError(
                "champion resolution requires a loader-issued verified crosswalk"
            ) from exc
        return state

    class _VerifiedCrosswalk(RuntimeMapping):
        __slots__ = ("__weakref__",)
        __hash__ = object.__hash__
        __eq__ = object.__eq__

        def _payload(self) -> dict[str, object]:
            persisted_bytes, _, _, _ = require_state(self)
            payload = json.loads(persisted_bytes)
            if not isinstance(payload, dict):
                raise ChampionIdCrosswalkError(
                    "verified crosswalk bytes no longer decode to an object"
                )
            return payload

        def __getitem__(self, key: str) -> object:
            return copy.deepcopy(self._payload()[key])

        def __iter__(self) -> Iterator[str]:
            return iter(tuple(self._payload()))

        def __len__(self) -> int:
            return len(self._payload())

        def __copy__(self) -> dict[str, object]:
            return copy.deepcopy(self._payload())

        def __deepcopy__(self, memo: dict[int, object]) -> dict[str, object]:
            return copy.deepcopy(self._payload(), memo)

        def __reduce_ex__(self, protocol: int) -> object:
            _ = protocol
            raise ChampionIdCrosswalkError(
                "verified crosswalk capabilities cannot be serialized"
            )

        def __repr__(self) -> str:
            return "<verified champion crosswalk: source-backed replay complete>"

    def load_and_replay_artifact(
        artifact_path: Path = DEFAULT_ARTIFACT_PATH,
        *,
        source_root: Path | None = None,
    ) -> RuntimeMapping:
        replayed, persisted_bytes, resolved_artifact_path, source_identities = (
            _load_and_replay_payload(
                artifact_path=artifact_path,
                source_root=source_root,
            )
        )
        if resolved_artifact_path != DEFAULT_ARTIFACT_PATH.resolve():
            raise ChampionIdCrosswalkError(
                "resolution authority is restricted to the canonical artifact path"
            )
        verified = _VerifiedCrosswalk()
        store[verified] = (
            bytes(persisted_bytes),
            resolved_artifact_path,
            _derive_resolution_table(replayed),
            source_identities,
        )
        return verified

    def resolve_champion_id(verified_crosswalk: object, value: object) -> str:
        (
            persisted_bytes,
            artifact_path,
            resolution_table,
            source_identities,
        ) = require_state(verified_crosswalk)
        _require_regular_file(artifact_path, "canonical crosswalk artifact")
        try:
            current_bytes = artifact_path.read_bytes()
            payload = json.loads(current_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ChampionIdCrosswalkError(
                "canonical crosswalk artifact is no longer readable"
            ) from exc
        if current_bytes != persisted_bytes:
            raise ChampionIdCrosswalkError(
                "verified crosswalk is stale relative to canonical artifact bytes"
            )
        if not isinstance(payload, dict):
            raise ChampionIdCrosswalkError(
                "canonical crosswalk artifact must remain an object"
            )
        validate_artifact(payload)
        if current_bytes != canonical_bytes(payload):
            raise ChampionIdCrosswalkError(
                "canonical crosswalk artifact bytes changed form"
            )
        if payload["generator"] != _generator_identity():
            raise ChampionIdCrosswalkError(
                "verified crosswalk is stale relative to executable generator"
            )
        for label, source_path, expected_raw_sha256 in source_identities:
            _require_regular_file(source_path, label)
            if raw_sha256(source_path) != expected_raw_sha256:
                raise ChampionIdCrosswalkError(
                    f"verified crosswalk is stale relative to {label} source bytes"
                )
        derived_table = _derive_resolution_table(payload)
        if derived_table != resolution_table:
            raise ChampionIdCrosswalkError(
                "verified crosswalk resolution state failed integrity comparison"
            )
        normalized = normalize_name(value)
        matches = dict(resolution_table)
        try:
            return matches[normalized]
        except KeyError as exc:
            raise ChampionIdCrosswalkError(
                "champion is unknown, future, or only partial"
            ) from exc

    return load_and_replay_artifact, resolve_champion_id


load_and_replay_artifact, resolve_champion_id = _make_runtime_crosswalk_api()
del _make_runtime_crosswalk_api


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.replay:
        load_and_replay_artifact(args.output)
    else:
        write_artifact(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
