"""Build a closed, source-bound future-value research freeze.

The command reads frozen Oracle's Elixir parquet files and already captured
annual and bridge bytes.  It writes a new research directory.  Every input
and generated binding uses an absolute regular-file path, a byte count, and a
SHA-256 digest.  Duplicate resolutions require a separate audit artifact and
receipt.

This command does not fit a model and does not write worker or public files.
Its output keeps all authority flags disabled.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    _canonical_json_bytes,
    _frame_game_ids,
    _utc_text,
    bind_accepted_future_value_source,
    validate_future_value_source_receipt_payload,
    write_source_receipt,
)
from lol_kills.research.future_value_training import (
    DUPLICATE_RESOLUTION_SCHEMA_VERSION,
    FREEZE_SCHEMA_V2_VERSION,
    FutureValueTrainingError,
    KNOWN_DUPLICATE_BRIDGE_GAME_IDS,
    duplicate_resolution_mapping_sha256,
    validate_duplicate_resolution_block,
)
from lol_kills.v2.tierlists.accepted_census import (
    canonical_game_ids,
    census_payload,
    identity_sha256,
)


SCHEMA_VERSION = "scryglass:future-value-source-trust-run:v2"
OE_SOURCE_RECORD_SCHEMA_VERSION = "scryglass:future-value-oe-map-source-record:v1"
MAP_ROWS_FILE = "accepted-oe-map-rows.json"
MAP_RECORD_FILE = "accepted-oe-source-record.json"
FREEZE_FILE = "future-value-source-freeze-v2.json"
CENSUS_FILE = "accepted-census.json"
SOURCE_RECEIPT_FILE = "future-value-source-receipt.json"
RUN_FILE = "future-value-source-trust-run.json"
_HEX64 = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

AUTHORITY: dict[str, bool] = {
    "research_only": True,
    "public": False,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_probability": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
    "promotion": False,
    "merge": False,
    "deployment": False,
}

LEGACY_EXCLUSIONS = tuple(
    f"{prefix}-{prefix}_game_{game}"
    for prefix in ("13420", "13422")
    for game in (1, 2, 3)
)
KNOWN_DUPLICATE_IDS = tuple(sorted(KNOWN_DUPLICATE_BRIDGE_GAME_IDS))
DEFAULT_SEMANTIC_FIELDS = ("date", "league", "patch")
MAP_ROW_FIELDS = (
    "date",
    "gameid",
    "league",
    "patch",
    "team_keys",
    "teams",
    "tournament",
)


class SourceTrustError(ValueError):
    """The frozen source cannot be proved from the supplied evidence."""


def _canonical(value: object) -> bytes:
    try:
        return _canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise SourceTrustError("value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise SourceTrustError(f"cannot read file: {path}") from error
    return digest.hexdigest()


def _json_safe(value: object) -> object:
    """Convert pandas and NumPy scalars into strict JSON values."""

    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime)):
        stamp = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(stamp):
            return None
        return pd.Timestamp(stamp).isoformat().replace("+00:00", "Z")
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool) and missing:
            return None
    except (TypeError, ValueError):
        pass
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (str, bytes, bytearray)):
        try:
            scalar = item()
        except (TypeError, ValueError):
            scalar = value
        if scalar is not value:
            return _json_safe(scalar)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    return value


def _safe_path(value: Path | str, label: str, *, directory: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise SourceTrustError(f"{label} must be an absolute path without '..'")
    path = Path(os.path.abspath(path))
    if path.is_symlink():
        raise SourceTrustError(f"{label} is a symlink")
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise SourceTrustError(f"{label} path contains a symlink")
    except OSError as error:
        raise SourceTrustError(f"{label} cannot be inspected") from error
    if directory:
        if not path.is_dir():
            raise SourceTrustError(f"{label} is not a directory")
    elif not path.is_file():
        raise SourceTrustError(f"{label} is not a regular file")
    return path


def _safe_source_root(value: Path | str) -> Path:
    path = _safe_path(value, "source root", directory=True)
    return path


def _safe_output_root(value: Path | str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise SourceTrustError("output root must be an absolute path without '..'")
    path = Path(os.path.abspath(path))
    if path.exists() and path.is_symlink():
        raise SourceTrustError("output root is a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise SourceTrustError("output root is not a safe directory")
    if any(path.iterdir()):
        raise SourceTrustError("output root must be empty")
    return path


def _file_record(path: Path, label: str, *, year: int | None = None) -> dict[str, Any]:
    safe = _safe_path(path, label)
    record: dict[str, Any] = {
        "path": str(safe),
        "bytes": int(safe.stat().st_size),
        "sha256": _sha256_file(safe),
    }
    if year is not None:
        record["year"] = int(year)
    return record


def _load_json(path: Path, label: str) -> dict[str, Any]:
    safe = _safe_path(path, label)
    try:
        value = json.loads(safe.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SourceTrustError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise SourceTrustError(f"{label} must contain an object")
    return dict(value)


def _write_json(path: Path, value: object) -> bytes:
    if path.exists() or path.is_symlink():
        raise SourceTrustError(f"output already exists: {path}")
    raw = json.dumps(
        _json_safe(value),
        allow_nan=False,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8") + b"\n"
    path.write_bytes(raw)
    return raw


def _source_paths(source_root: Path) -> dict[str, Path]:
    candidates = {
        "maps": ("maps.parquet",),
        "players": ("oe_player_games.parquet", "players.parquet"),
        "teams": ("oe_team_games.parquet", "teams.parquet"),
    }
    result: dict[str, Path] = {}
    for label, names in candidates.items():
        for name in names:
            candidate = source_root / name
            if candidate.exists():
                result[label] = _safe_path(candidate, f"source {label}")
                break
        if label not in result:
            raise SourceTrustError(f"source {label} parquet is missing")
    return result


def _discover_files(root: Path, label: str, *, suffixes: Sequence[str] | None = None) -> list[Path]:
    safe_root = _safe_path(root, label, directory=True)
    allowed = tuple(suffixes or ())
    files: list[Path] = []
    for candidate in sorted(safe_root.iterdir(), key=lambda item: item.name):
        if candidate.is_dir():
            continue
        if candidate.is_symlink() or not candidate.is_file():
            raise SourceTrustError(f"{label} contains an unsafe entry: {candidate.name}")
        if allowed and candidate.suffix.lower() not in allowed:
            continue
        files.append(_safe_path(candidate, f"{label} file"))
    if not files:
        raise SourceTrustError(f"{label} has no input files")
    return files


def _annual_game_count(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                return None
            field = next(
                (
                    name
                    for name in reader.fieldnames
                    if str(name).strip().casefold() in {"gameid", "game_id", "game_uid"}
                ),
                None,
            )
            if field is None:
                return None
            values = {
                canonical_source_game_key(row.get(field))
                for row in reader
                if canonical_source_game_key(row.get(field))
            }
    except (OSError, UnicodeError, csv.Error) as error:
        raise SourceTrustError(f"annual source cannot be read: {path.name}") from error
    return len(values)


def _normalise_spec(value: Mapping[str, Any]) -> dict[str, Any]:
    spec = dict(value)
    exclusions = (
        spec.get("exclude_game_ids")
        or spec.get("excluded_game_ids")
        or (
            spec.get("accepted_census", {}).get("excluded_game_ids")
            if isinstance(spec.get("accepted_census"), Mapping)
            else None
        )
    )
    if not isinstance(exclusions, list):
        raise SourceTrustError("resolution spec must list exclude_game_ids")
    canonical_exclusions = list(canonical_game_ids(exclusions))
    if len(canonical_exclusions) != len(exclusions):
        raise SourceTrustError("resolution spec exclusions are not canonical and unique")
    spec["exclude_game_ids"] = canonical_exclusions
    block = spec.get("duplicate_resolution")
    if block is None:
        block = {}
        spec["duplicate_resolution"] = block
    if not isinstance(block, Mapping):
        raise SourceTrustError("duplicate_resolution must be an object")
    block = dict(block)
    mappings = block.get("mappings") or spec.get("duplicate_mappings") or spec.get("mappings")
    if mappings is None:
        mappings = []
    if not isinstance(mappings, list) or any(not isinstance(row, Mapping) for row in mappings):
        raise SourceTrustError("duplicate resolution mappings are invalid")
    block["mappings"] = [dict(row) for row in mappings]
    rule = block.get("survivor_rule") or spec.get("survivor_rule")
    if rule is None:
        rule = "annual_row_is_survivor_verified_external_identity"
    if not isinstance(rule, str) or not rule.strip():
        raise SourceTrustError("duplicate resolution survivor rule is missing")
    block["survivor_rule"] = rule
    spec["duplicate_resolution"] = block
    return spec


def _map_ids(frame: pd.DataFrame) -> pd.Series:
    try:
        return _frame_game_ids(frame, "maps")
    except FutureValueSourceError as error:
        raise SourceTrustError(str(error)) from error


def _map_row_projection(maps: pd.DataFrame, game_ids: Iterable[str]) -> list[dict[str, Any]]:
    ids = _map_ids(maps).astype(str)
    if ids.duplicated().any():
        raise SourceTrustError("raw maps contain duplicate canonical game IDs")
    by_id = {game_id: row for game_id, row in zip(ids, maps.to_dict(orient="records"))}
    rows: list[dict[str, Any]] = []
    for game_id in canonical_game_ids(game_ids):
        if game_id not in by_id:
            raise SourceTrustError(f"map projection is missing game ID: {game_id}")
        raw = by_id[game_id]
        date = pd.to_datetime(raw.get("date"), errors="coerce", utc=True)
        if pd.isna(date):
            raise SourceTrustError(f"map projection has an invalid date: {game_id}")
        blue_key = raw.get("blue_team_key")
        red_key = raw.get("red_team_key")
        blue_name = raw.get("blue_team")
        red_name = raw.get("red_team")
        if any(value is None or (isinstance(value, str) and not value.strip()) for value in (blue_key, red_key, blue_name, red_name)):
            raise SourceTrustError(f"map projection has incomplete team identity: {game_id}")
        rows.append(
            {
                "date": pd.Timestamp(date).isoformat().replace("+00:00", "Z"),
                "gameid": game_id,
                "league": _json_safe(raw.get("league")),
                "patch": _json_safe(raw.get("patch")),
                "team_keys": [_json_safe(blue_key), _json_safe(red_key)],
                "teams": [_json_safe(blue_name), _json_safe(red_name)],
                "tournament": _json_safe(raw.get("tournament")),
            }
        )
    return rows


def _row_snapshot(row: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in row.items()}


def _mapping_id(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row:
            value = canonical_source_game_key(row.get(key))
            if value:
                return value
    return ""


def _audit_assignments(artifact: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for key in ("assignments", "mappings", "duplicate_mappings", "rows"):
        value = artifact.get(key)
        if isinstance(value, list) and all(isinstance(item, Mapping) for item in value):
            rows.extend(value)
    issues = artifact.get("issues")
    if isinstance(issues, list):
        rows.extend(item for item in issues if isinstance(item, Mapping) and item.get("kind") == "duplicate_source_assignment")
    return rows


def _audit_identity(row: Mapping[str, Any]) -> dict[str, str]:
    scoreboard = str(
        row.get("scoreboard_game_id")
        or row.get("ScoreboardGames.GameId")
        or row.get("scoreboard_id")
        or ""
    ).strip()
    riot = str(
        row.get("scoreboard_riot_platform_game_id")
        or row.get("RiotPlatformGameId")
        or row.get("riot_platform_game_id")
        or ""
    ).strip()
    return {"scoreboard_game_id": scoreboard, "scoreboard_riot_platform_game_id": riot}


def _binding_record(value: Any, label: str) -> dict[str, Any]:
    if isinstance(value, (str, Path)):
        return _file_record(Path(value), label)
    if not isinstance(value, Mapping):
        raise SourceTrustError(f"{label} binding is missing")
    raw_path = value.get("path", value.get("locator"))
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise SourceTrustError(f"{label} path is missing")
    actual = _file_record(Path(raw_path), label)
    expected_bytes = value.get("bytes")
    expected_hash = str(value.get("sha256") or "").lower()
    if expected_bytes is not None and expected_bytes != actual["bytes"]:
        raise SourceTrustError(f"{label} byte count changed")
    if expected_hash and expected_hash != actual["sha256"]:
        raise SourceTrustError(f"{label} hash changed")
    return actual


def _verify_crosswalk_receipt(
    artifact: Mapping[str, Any],
    receipt: Mapping[str, Any],
    artifact_record: Mapping[str, Any],
    receipt_record: Mapping[str, Any],
    *,
    expected_receipt_file_sha256: str,
) -> list[Mapping[str, Any]]:
    """Verify the old direct-identity crosswalk before extracting pairs."""

    if not _HEX64.fullmatch(expected_receipt_file_sha256):
        raise SourceTrustError("expected crosswalk receipt file SHA-256 is invalid")
    if str(receipt_record["sha256"]).lower() != expected_receipt_file_sha256.lower():
        raise SourceTrustError("crosswalk receipt file SHA-256 changed")
    receipt_hash = str(receipt.get("receipt_sha256") or "").lower()
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256", None)
    if not _HEX64.fullmatch(receipt_hash) or _sha256_bytes(_canonical(receipt_body)) != receipt_hash:
        raise SourceTrustError("crosswalk receipt self-hash is invalid")
    authority = receipt.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise SourceTrustError("crosswalk receipt authority is invalid")
    if any(bool(value) for key, value in authority.items() if key != "research_only"):
        raise SourceTrustError("crosswalk receipt grants authority")
    declared_artifact = receipt.get("artifact")
    if isinstance(declared_artifact, Mapping) and (
        declared_artifact.get("bytes") != artifact_record["bytes"]
        or str(declared_artifact.get("sha256") or "").lower() != str(artifact_record["sha256"]).lower()
    ):
        raise SourceTrustError("crosswalk receipt artifact binding changed")
    artifact_hash = str(artifact.get("crosswalk_sha256") or artifact.get("artifact_sha256") or "").lower()
    if not _HEX64.fullmatch(artifact_hash):
        raise SourceTrustError("crosswalk artifact self-hash is missing")
    artifact_body = dict(artifact)
    artifact_body.pop("crosswalk_sha256", None)
    artifact_body.pop("artifact_sha256", None)
    if _sha256_bytes(_canonical(artifact_body)) != artifact_hash:
        raise SourceTrustError("crosswalk artifact self-hash changed")
    receipt_crosswalk_hash = str(receipt.get("crosswalk_sha256") or "").lower()
    if receipt_crosswalk_hash and receipt_crosswalk_hash != artifact_hash:
        raise SourceTrustError("crosswalk receipt artifact digest changed")
    assignments = _audit_assignments(artifact)
    if not assignments:
        raise SourceTrustError("crosswalk artifact has no assignments")
    return assignments


def _crosswalk_binding(value: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("crosswalk", "old_crosswalk", "verified_crosswalk", "direct_series_crosswalk"):
        candidate = value.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    # Accept the explicit source-binding shape used by the direct-series
    # capture receipts.  A bare source binding remains an audit binding.
    source_binding = value.get("source_binding")
    if isinstance(source_binding, Mapping) and str(source_binding.get("kind", "")).lower() in {
        "leaguepedia_crosswalk",
        "verified_leaguepedia_crosswalk",
        "direct_series_crosswalk",
    }:
        return source_binding
    return None


def _generate_duplicate_audit(
    spec: Mapping[str, Any],
    maps: pd.DataFrame,
    *,
    excluded: Sequence[str],
    accepted_identity: str,
    source_receipt_sha256: str,
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Extract explicit pairs from an old crosswalk and seal a new audit."""

    raw_block = spec.get("duplicate_resolution")
    if not isinstance(raw_block, Mapping):
        raise SourceTrustError("duplicate resolution is invalid")
    binding = _crosswalk_binding(raw_block) or _crosswalk_binding(spec)
    if binding is None:
        audit = spec.get("duplicate_audit")
        if isinstance(audit, Mapping):
            binding = _crosswalk_binding(audit)
    if binding is None:
        raise SourceTrustError("verified old crosswalk binding is required")
    artifact_value = binding.get("artifact", binding.get("crosswalk_artifact"))
    receipt_value = binding.get("receipt", binding.get("crosswalk_receipt"))
    artifact_record = _binding_record(artifact_value, "old crosswalk artifact")
    receipt_record = _binding_record(receipt_value, "old crosswalk receipt")
    expected_file_sha = str(
        binding.get("expected_receipt_file_sha256")
        or binding.get("expected_crosswalk_receipt_file_sha256")
        or ""
    )
    artifact = _load_json(Path(artifact_record["path"]), "old crosswalk artifact")
    receipt = _load_json(Path(receipt_record["path"]), "old crosswalk receipt")
    assignments = _verify_crosswalk_receipt(
        artifact,
        receipt,
        artifact_record,
        receipt_record,
        expected_receipt_file_sha256=expected_file_sha,
    )
    by_id: dict[str, list[Mapping[str, Any]]] = {}
    for assignment in assignments:
        game_id = _mapping_id(assignment, "oe_game_id", "game_id", "game_uid")
        if game_id:
            by_id.setdefault(game_id, []).append(assignment)
    pairs = (
        raw_block.get("mappings")
        or raw_block.get("pairs")
        or spec.get("duplicate_mappings")
        or spec.get("duplicate_pairs")
        or spec.get("mappings")
    )
    if not isinstance(pairs, list) or not pairs:
        raise SourceTrustError("duplicate resolution pairs are required")
    map_ids = set(_map_ids(maps).astype(str))
    compact_assignments: list[dict[str, Any]] = []
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise SourceTrustError("duplicate resolution pair is invalid")
        bridge = _mapping_id(pair, "bridge_game_id", "bridge_id", "oe_game_id")
        survivor = _mapping_id(pair, "annual_survivor_game_id", "survivor_game_id", "annual_game_id")
        if not bridge or not survivor or bridge not in map_ids or survivor not in map_ids:
            raise SourceTrustError("duplicate resolution pair is missing from maps")
        if bridge not in set(excluded) or survivor in set(excluded):
            raise SourceTrustError("duplicate resolution pair exclusion is invalid")
        bridge_rows = by_id.get(bridge, [])
        survivor_rows = by_id.get(survivor, [])
        if not bridge_rows or not survivor_rows:
            raise SourceTrustError("old crosswalk does not contain both pair identities")
        bridge_scoreboard = str(bridge_rows[0].get("scoreboard_game_id") or "").strip()
        survivor_scoreboard = str(survivor_rows[0].get("scoreboard_game_id") or "").strip()
        if not bridge_scoreboard or bridge_scoreboard != survivor_scoreboard:
            raise SourceTrustError("old crosswalk pair does not share one scoreboard identity")
        survivor_evidence = survivor_rows[0].get("evidence")
        if not isinstance(survivor_evidence, Mapping):
            raise SourceTrustError("old crosswalk annual identity evidence is missing")
        identity_evidence = survivor_evidence.get("identity")
        if not isinstance(identity_evidence, Mapping) or identity_evidence.get("exact") is not True:
            raise SourceTrustError("old crosswalk annual identity is not exact")
        riot_id = str(
            survivor_rows[0].get("scoreboard_riot_platform_game_id")
            or identity_evidence.get("value")
            or ""
        ).strip()
        if not riot_id:
            raise SourceTrustError("old crosswalk annual Riot identity is missing")
        compact_assignments.append(
            {
                "bridge_game_id": bridge,
                "annual_survivor_game_id": survivor,
                "scoreboard_game_id": bridge_scoreboard,
                "scoreboard_riot_platform_game_id": riot_id,
                "identity_method": "old_verified_direct_riot_platform_game_id_crosswalk",
                "source_crosswalk_artifact_sha256": artifact_record["sha256"],
                "source_crosswalk_receipt_sha256": receipt.get("receipt_sha256"),
            }
        )
    audit: dict[str, Any] = {
        "schema_version": "scryglass:duplicate-audit:v1",
        "status": "verified_research_only",
        "source_identity_sha256": accepted_identity,
        "source_receipt_sha256": source_receipt_sha256,
        "identity_source": {
            "kind": "verified_old_direct_series_crosswalk",
            "artifact_sha256": artifact_record["sha256"],
            "receipt_file_sha256": receipt_record["sha256"],
            "receipt_sha256": receipt.get("receipt_sha256"),
        },
        "assignments": compact_assignments,
        "outcome_free": True,
        "semantic_evidence_policy": {
            "fields": list(DEFAULT_SEMANTIC_FIELDS),
            "excludes_outcome_fields": True,
        },
        "authority": dict(AUTHORITY),
    }
    audit["crosswalk_sha256"] = _sha256_bytes(_canonical(audit))
    audit_path = output_root / "duplicate-audit.json"
    audit_raw = _write_json(audit_path, audit)
    audit_record = _file_record(audit_path, "generated duplicate audit")
    audit_receipt: dict[str, Any] = {
        "schema_version": "scryglass:duplicate-audit-receipt:v1",
        "status": "verified_research_only",
        "artifact": audit_record,
        "crosswalk_sha256": audit["crosswalk_sha256"],
        "source_identity_sha256": accepted_identity,
        "source_receipt_sha256": source_receipt_sha256,
        "authority": dict(AUTHORITY),
    }
    audit_receipt["receipt_sha256"] = _sha256_bytes(_canonical(audit_receipt))
    receipt_path = output_root / "duplicate-audit.receipt.json"
    _write_json(receipt_path, audit_receipt)
    receipt_record_out = _file_record(receipt_path, "generated duplicate audit receipt")
    return audit, {
        "artifact": audit_record,
        "receipt": receipt_record_out,
        "receipt_sha256": audit_receipt["receipt_sha256"],
    }, {
        "path": str(receipt_path),
        "bytes": receipt_record_out["bytes"],
        "sha256": receipt_record_out["sha256"],
    }


def _prepare_duplicate_block(
    spec: Mapping[str, Any],
    maps: pd.DataFrame,
    *,
    excluded: Sequence[str],
    accepted_identity: str,
    source_receipt_sha256: str,
    output_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    raw_block = spec.get("duplicate_resolution")
    if not isinstance(raw_block, Mapping):
        raise SourceTrustError("duplicate resolution is invalid")
    mappings_value = raw_block.get("mappings")
    if not isinstance(mappings_value, list):
        raise SourceTrustError("duplicate resolution mappings are invalid")
    required = set(canonical_game_ids(excluded)) & set(KNOWN_DUPLICATE_BRIDGE_GAME_IDS)
    bridge_ids = {
        _mapping_id(row, "bridge_game_id", "bridge_id", "oe_game_id")
        for row in mappings_value
        if isinstance(row, Mapping)
    }
    bridge_ids.discard("")
    if required and bridge_ids != required:
        raise SourceTrustError("duplicate resolution mappings do not cover excluded bridge IDs")
    if required and not mappings_value:
        raise SourceTrustError("duplicate resolution mappings are required for bridge exclusions")

    audit_spec = spec.get("duplicate_audit")
    generated_audit = (
        _crosswalk_binding(raw_block)
        or _crosswalk_binding(spec)
        or (_crosswalk_binding(audit_spec) if isinstance(audit_spec, Mapping) else None)
    )
    generated_records: tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None = None
    if generated_audit is not None:
        _audit_payload, generated_audit_info, _generated_receipt_record = _generate_duplicate_audit(
            spec,
            maps,
            excluded=excluded,
            accepted_identity=accepted_identity,
            source_receipt_sha256=source_receipt_sha256,
            output_root=output_root,
        )
        generated_records = (
            _audit_payload,
            generated_audit_info,
            _generated_receipt_record,
        )
    audit_value: Any = raw_block.get("source_binding")
    if generated_records is not None:
        audit_value = generated_records[1]
    if audit_value is None:
        audit_value = spec.get("duplicate_audit", spec.get("audit"))
    if not isinstance(audit_value, Mapping):
        raise SourceTrustError("independent duplicate audit binding is required")
    artifact_value = audit_value.get("artifact", audit_value.get("audit_artifact"))
    receipt_value = audit_value.get("receipt", audit_value.get("audit_receipt"))
    if artifact_value is None:
        artifact_value = audit_value.get("artifact_path")
    if receipt_value is None:
        receipt_value = audit_value.get("receipt_path")
    artifact_record = _binding_record(artifact_value, "duplicate audit artifact")
    receipt_record = _binding_record(receipt_value, "duplicate audit receipt")
    artifact = _load_json(Path(artifact_record["path"]), "duplicate audit artifact")
    receipt = _load_json(Path(receipt_record["path"]), "duplicate audit receipt")
    if not _audit_assignments(artifact):
        raise SourceTrustError("duplicate audit artifact has no assignments")
    receipt_hash = str(receipt.get("receipt_sha256") or "").lower()
    receipt_body = dict(receipt)
    receipt_body.pop("receipt_sha256", None)
    if not _HEX64.fullmatch(receipt_hash) or _sha256_bytes(_canonical(receipt_body)) != receipt_hash:
        raise SourceTrustError("duplicate audit receipt self-hash is invalid")
    if receipt.get("source_identity_sha256") != accepted_identity:
        raise SourceTrustError("duplicate audit receipt source identity changed")
    authority = receipt.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise SourceTrustError("duplicate audit receipt authority is invalid")
    if any(bool(value) for key, value in authority.items() if key != "research_only"):
        raise SourceTrustError("duplicate audit receipt grants authority")

    raw_ids = _map_ids(maps).astype(str)
    by_id = {game_id: row for game_id, row in zip(raw_ids, maps.to_dict(orient="records"))}
    audit_by_id: dict[str, dict[str, str]] = {}
    for assignment in _audit_assignments(artifact):
        identity = _audit_identity(assignment)
        if not any(identity.values()):
            continue
        ids = {
            _mapping_id(assignment, key)
            for key in ("oe_game_id", "bridge_game_id", "annual_survivor_game_id", "game_id", "game_uid")
        }
        ids.discard("")
        for game_id in ids:
            previous = audit_by_id.get(game_id)
            if previous is not None and any(
                previous[key] and identity[key] and previous[key] != identity[key]
                for key in identity
            ):
                raise SourceTrustError("duplicate audit artifact maps one ID inconsistently")
            audit_by_id[game_id] = {
                key: previous.get(key, "") if previous else ""
                for key in identity
            }
            for key in identity:
                if identity[key]:
                    audit_by_id[game_id][key] = identity[key]

    normalised: list[dict[str, Any]] = []
    for raw_mapping in mappings_value:
        mapping = dict(raw_mapping)
        bridge = _mapping_id(mapping, "bridge_game_id", "bridge_id", "oe_game_id")
        survivor = _mapping_id(mapping, "annual_survivor_game_id", "survivor_game_id", "annual_game_id")
        if not bridge or not survivor or bridge == survivor:
            raise SourceTrustError("duplicate mapping IDs are invalid")
        if bridge not in by_id or survivor not in by_id:
            raise SourceTrustError("duplicate mapping IDs are missing from raw maps")
        if bridge not in set(excluded) or survivor in set(excluded):
            raise SourceTrustError("duplicate mapping exclusion is invalid")
        bridge_identity = audit_by_id.get(bridge)
        survivor_identity = audit_by_id.get(survivor)
        if bridge_identity is None or survivor_identity is None:
            raise SourceTrustError("duplicate audit artifact misses a mapped game")
        if not any(
            bridge_identity[key] and survivor_identity[key] and bridge_identity[key] == survivor_identity[key]
            for key in bridge_identity
        ):
            raise SourceTrustError("duplicate audit artifact does not prove one external game")
        mapping["bridge_game_id"] = bridge
        mapping["annual_survivor_game_id"] = survivor
        mapping["survivor_rule"] = str(raw_block["survivor_rule"])
        mapping["bridge_source_row"] = _row_snapshot(
            mapping.get("bridge_source_row", mapping.get("bridge_row", by_id[bridge]))
        )
        mapping["annual_survivor_source_row"] = _row_snapshot(
            mapping.get("annual_survivor_source_row", mapping.get("annual_source_row", mapping.get("survivor_row", by_id[survivor])))
        )
        evidence = mapping.get("evidence")
        if not isinstance(evidence, Mapping):
            evidence = {}
        evidence = dict(evidence)
        fields = evidence.get("semantic_fields") or list(DEFAULT_SEMANTIC_FIELDS)
        if not isinstance(fields, list) or any(field not in DEFAULT_SEMANTIC_FIELDS for field in fields):
            raise SourceTrustError("duplicate semantic fields must be exact invariant fields")
        fields = list(dict.fromkeys(fields))
        if not fields:
            raise SourceTrustError("duplicate semantic fields are empty")
        values: dict[str, Any] = {}
        for field in fields:
            source_value = mapping["bridge_source_row"].get(field)
            survivor_value = mapping["annual_survivor_source_row"].get(field)
            if _json_safe(source_value) != _json_safe(survivor_value):
                raise SourceTrustError(f"duplicate semantic field differs: {field}")
            values[field] = _json_safe(source_value)
        evidence["semantic_fields"] = fields
        evidence["field_values"] = values
        evidence["external_identity"] = {
            key: bridge_identity[key] or survivor_identity[key]
            for key in bridge_identity
            if bridge_identity[key] or survivor_identity[key]
        }
        mapping["evidence"] = evidence
        normalised.append(mapping)

    if not normalised:
        if required:
            raise SourceTrustError("duplicate resolution mappings are required")
        return None, None
    block: dict[str, Any] = {
        "schema_version": DUPLICATE_RESOLUTION_SCHEMA_VERSION,
        "survivor_rule": str(raw_block["survivor_rule"]),
        "mappings": normalised,
        "source_binding": {
            "kind": "duplicate_audit",
            "artifact": artifact_record,
            "receipt": receipt_record,
            "expected_receipt_file_sha256": receipt_record["sha256"],
            "source_identity_sha256": accepted_identity,
        },
    }
    supplied_digest = raw_block.get("mapping_sha256")
    block["mapping_sha256"] = duplicate_resolution_mapping_sha256(normalised)
    if supplied_digest is not None and str(supplied_digest).lower() != block["mapping_sha256"]:
        raise SourceTrustError("duplicate mapping digest changed")
    return block, {
        "artifact": artifact_record,
        "receipt": receipt_record,
        "receipt_sha256": receipt_hash,
    }


def _git_fingerprint(repo_root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    relative_files = (
        "benchmarks/build_future_value_source_trust.py",
        "lol_kills/research/future_value_training.py",
        "lol_kills/research/future_value_rating.py",
        "lol_kills/v2/tierlists/accepted_census.py",
    )
    for relative in relative_files:
        path = _safe_path(repo_root / relative, f"code file {relative}")
        files.append({"locator": relative, **_file_record(path, f"code file {relative}")})
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise SourceTrustError("code fingerprint cannot be read") from error
    if not re.fullmatch(r"[0-9a-f]{7,64}", commit, re.IGNORECASE):
        raise SourceTrustError("code commit fingerprint is invalid")
    payload = {"commit": commit, "files": files, "working_tree_dirty": bool(status.strip())}
    payload["fingerprint_sha256"] = _sha256_bytes(_canonical(payload))
    return payload


def build_source_trust(
    *,
    source_root: Path | str,
    output_root: Path | str,
    resolution_spec: Path | str | Mapping[str, Any],
    annual_root: Path | str | None = None,
    bridge_root: Path | str | None = None,
    source_as_of: str | None = None,
    expected_unfiltered_count: int | None = None,
    expected_accepted_count: int | None = None,
) -> dict[str, Any]:
    """Build the source freeze and return the run receipt payload."""

    root = _safe_source_root(source_root)
    out = _safe_output_root(output_root)
    resolution_spec_record: dict[str, Any]
    if isinstance(resolution_spec, Mapping):
        spec = _normalise_spec(resolution_spec)
        resolution_spec_record = {
            "kind": "inline",
            "payload_sha256": _sha256_bytes(_canonical(spec)),
            "payload_bytes": len(_canonical(spec)),
        }
    else:
        spec_path = _safe_path(Path(resolution_spec), "resolution spec")
        spec = _normalise_spec(_load_json(spec_path, "resolution spec"))
        resolution_spec_record = {
            "kind": "file",
            **_file_record(spec_path, "resolution spec"),
            "payload_sha256": _sha256_bytes(_canonical(spec)),
        }
    paths = _source_paths(root)
    try:
        maps = pd.read_parquet(paths["maps"])
        players = pd.read_parquet(paths["players"])
        teams = pd.read_parquet(paths["teams"])
    except Exception as error:
        raise SourceTrustError("normalized source parquet cannot be read") from error
    raw_ids_series = _map_ids(maps)
    raw_ids = tuple(canonical_game_ids(raw_ids_series.astype(str)))
    if len(raw_ids) != len(maps):
        raise SourceTrustError("raw maps contain duplicate canonical game IDs")
    if expected_unfiltered_count is not None and len(raw_ids) != int(expected_unfiltered_count):
        raise SourceTrustError("raw map census count differs from expected count")
    excluded = tuple(canonical_game_ids(spec["exclude_game_ids"]))
    if not excluded or not set(excluded).issubset(set(raw_ids)):
        raise SourceTrustError("source exclusions are missing from the raw census")
    accepted_ids = tuple(game_id for game_id in raw_ids if game_id not in set(excluded))
    if expected_accepted_count is not None and len(accepted_ids) != int(expected_accepted_count):
        raise SourceTrustError("accepted census count differs from expected count")
    accepted = census_payload(accepted_ids)
    maps_dates = pd.to_datetime(maps["date"], errors="coerce", utc=True)
    if maps_dates.isna().any():
        raise SourceTrustError("raw maps contain invalid dates")
    cutoff = _utc_text(source_as_of or maps_dates.max())

    annual_dir = _safe_path(annual_root or (root / "raw"), "annual source root", directory=True)
    bridge_dir = _safe_path(bridge_root or (root / "bridge"), "bridge source root", directory=True)
    annual_paths = _discover_files(annual_dir, "annual source root", suffixes=(".csv",))
    bridge_paths = _discover_files(bridge_dir, "bridge source root")
    annual_records: list[dict[str, Any]] = []
    for path in annual_paths:
        match = re.search(r"(?:^|[^0-9])(20[0-9]{2})(?:[^0-9]|$)", path.name)
        year = int(match.group(1)) if match else None
        if year is None:
            raise SourceTrustError(f"annual source filename has no year: {path.name}")
        record = _file_record(path, f"annual source {path.name}", year=year)
        record.update({"name": path.name, "raw_sha256": record["sha256"], "game_count": _annual_game_count(path)})
        annual_records.append(record)
    bridge_records: list[dict[str, Any]] = []
    for path in bridge_paths:
        record = _file_record(path, f"bridge source {path.name}")
        record.update({"name": path.name, "raw_sha256": record["sha256"]})
        bridge_records.append(record)

    census_path = out / CENSUS_FILE
    census_bytes = _write_json(census_path, accepted)
    census_record = _file_record(census_path, "accepted census")
    source_files: dict[str, dict[str, Any]] = {
        "maps": _file_record(paths["maps"], "source maps"),
        "players": _file_record(paths["players"], "source players"),
        "teams": _file_record(paths["teams"], "source teams"),
        "accepted_census": census_record,
    }
    for record in annual_records:
        source_files[f"annual_{record['year']}"] = {
            key: value for key, value in record.items() if key in {"path", "bytes", "sha256", "year"}
        }
    for record in bridge_records:
        source_files[f"bridge_{record['name']}"] = {
            key: value for key, value in record.items() if key in {"path", "bytes", "sha256"}
        }

    try:
        source = bind_accepted_future_value_source(
            maps,
            players,
            teams,
            census=accepted,
            source_as_of=cutoff,
            source_files=source_files,
        )
        validate_future_value_source_receipt_payload(source.receipt)
    except (FutureValueSourceError, ValueError) as error:
        raise SourceTrustError(f"accepted source validation failed: {error}") from error
    source_receipt_path = out / SOURCE_RECEIPT_FILE
    write_source_receipt(source_receipt_path, source)
    source_receipt_record = _file_record(source_receipt_path, "future-value source receipt")
    if source_receipt_record["bytes"] <= 0:
        raise SourceTrustError("source receipt file is empty")

    accepted_rows = _map_row_projection(maps, accepted_ids)
    stable_team_key_rows = [
        {
            "game_id": row["gameid"],
            "teams": list(row["teams"]),
            "team_keys": list(row["team_keys"]),
        }
        for row in accepted_rows
    ]
    stable_team_key_rows.sort(key=lambda row: str(row["game_id"]))
    stable_team_key_digest = _sha256_bytes(_canonical(stable_team_key_rows))

    duplicate_block, duplicate_audit = _prepare_duplicate_block(
        spec,
        maps,
        excluded=excluded,
        accepted_identity=str(source.receipt["source_identity_sha256"]),
        source_receipt_sha256=str(source.receipt["receipt_sha256"]),
        output_root=out,
    )
    provisional_freeze: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA_V2_VERSION,
        "status": "accepted_source_bound_development_only",
        "source_mode": "oe_only",
        "source_transport": "frozen_oe_parquet_plus_annual_exports_and_bridge_bytes",
        "source_as_of": cutoff,
        "source_root": _file_record(root / "maps.parquet", "source root anchor"),
        "unfiltered_source_game_count": len(raw_ids),
        "unfiltered_source_identity_sha256": identity_sha256(raw_ids),
        "unfiltered_source_game_ids": list(raw_ids),
        "accepted_census": {
            **accepted,
            "excluded_game_ids": list(excluded),
        },
        "model_eligible_census": {
            "game_count": int(source.receipt["model_eligible_game_count"]),
            "source_identity_sha256": str(source.receipt["model_eligible_identity_sha256"]),
            "game_ids": list(source.receipt["model_eligible_game_ids"]),
        },
        "normalized_source_files": {
            label: source_files[label] for label in ("maps", "players", "teams")
        },
        "oe_annual_sources": [
            {
                "name": record["name"],
                "year": record["year"],
                "bytes": record["bytes"],
                "raw_sha256": record["sha256"],
                "game_count": record.get("game_count"),
            }
            for record in annual_records
        ],
        "oe_bridge_sources": [
            {
                "name": record["name"],
                "bytes": record["bytes"],
                "raw_sha256": record["sha256"],
            }
            for record in bridge_records
        ],
        "source_receipt_path": str(source_receipt_path),
        "source_receipt_file_sha256": source_receipt_record["sha256"],
        "reference_source_receipt_sha256": source.receipt["receipt_sha256"],
        "accepted_census_path": str(census_path),
        "accepted_census_file_sha256": census_record["sha256"],
        "stable_team_key_rows_sha256": stable_team_key_digest,
        "duplicate_resolution_required_bridge_game_ids": list(sorted(set(excluded) & set(KNOWN_DUPLICATE_BRIDGE_GAME_IDS))),
        "authority": dict(AUTHORITY),
        "resolution_spec": resolution_spec_record,
    }
    if duplicate_block is not None:
        provisional_freeze["duplicate_resolution"] = duplicate_block
        provisional_freeze["duplicate_audit_receipt_sha256"] = duplicate_audit["receipt_sha256"] if duplicate_audit else None
    try:
        validate_duplicate_resolution_block(maps, provisional_freeze)
    except FutureValueTrainingError as error:
        raise SourceTrustError(f"duplicate resolution validation failed: {error}") from error

    repo_root = Path(__file__).resolve().parents[1]
    code = _git_fingerprint(repo_root)
    provisional_freeze["code_fingerprint"] = code
    freeze_path = out / FREEZE_FILE
    freeze_bytes = _write_json(freeze_path, provisional_freeze)
    freeze_record = _file_record(freeze_path, "future-value source freeze")
    freeze_hash = freeze_record["sha256"]

    map_rows_path = out / MAP_ROWS_FILE
    map_rows_bytes = _write_json(map_rows_path, accepted_rows)
    map_rows_record = _file_record(map_rows_path, "accepted OE map rows")
    payload_bytes = _canonical(accepted_rows)
    map_record = {
        "schema_version": OE_SOURCE_RECORD_SCHEMA_VERSION,
        "locator": str(map_rows_path),
        "path": str(map_rows_path),
        "retrieved_at": cutoff,
        "sha256": map_rows_record["sha256"],
        "bytes": map_rows_record["bytes"],
        "payload_sha256": _sha256_bytes(payload_bytes),
        "payload_bytes": len(payload_bytes),
        "source_payload_sha256": _sha256_bytes(payload_bytes),
        "source_payload_bytes": len(payload_bytes),
        "payload_projection": {
            "scope": "accepted_map_rows",
            "policy": "fixed_outcome_free_projection",
            "fields": list(MAP_ROW_FIELDS),
        },
        "accepted_game_count": len(accepted_ids),
        "accepted_game_identity_sha256": identity_sha256(accepted_ids),
        "accepted_game_ids": list(accepted_ids),
        "integrity_verified": True,
    }
    map_record["stable_team_key_binding"] = {
        "field": "team_keys",
        "rows": stable_team_key_rows,
        "row_count": len(stable_team_key_rows),
        "rows_sha256": stable_team_key_digest,
        "stable_oe_identity_sha256": stable_team_key_digest,
    }
    map_record["stable_team_key_rows_sha256"] = stable_team_key_digest
    map_record_path = out / MAP_RECORD_FILE
    _write_json(map_record_path, map_record)
    map_record_file = _file_record(map_record_path, "accepted OE map source record")

    run: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "source_verified_model_unfitted",
        "source_as_of": cutoff,
        "unfiltered_source_game_count": len(raw_ids),
        "unfiltered_source_identity_sha256": identity_sha256(raw_ids),
        "unfiltered_source_game_ids": list(raw_ids),
        "source_game_count": len(accepted_ids),
        "source_identity_sha256": identity_sha256(accepted_ids),
        "accepted_game_ids": list(accepted_ids),
        "excluded_game_ids": list(excluded),
        "model_eligible_game_count": int(source.receipt["model_eligible_game_count"]),
        "model_eligible_identity_sha256": str(source.receipt["model_eligible_identity_sha256"]),
        "model_eligible_game_ids": list(source.receipt["model_eligible_game_ids"]),
        "source_receipt_sha256": source.receipt["receipt_sha256"],
        "source_receipt": source_receipt_record,
        "resolution_spec": resolution_spec_record,
        "freeze": {**freeze_record, "payload_sha256": freeze_hash},
        "accepted_census": {**census_record, "payload_sha256": census_record["sha256"]},
        "normalized_source_files": source_files,
        "annual_sources": annual_records,
        "bridge_sources": bridge_records,
        "source_records": {"oe": map_record},
        "accepted_oe_map_rows": map_rows_record,
        "accepted_oe_map_source_record": map_record_file,
        "stable_team_key_rows_sha256": stable_team_key_digest,
        "code_fingerprint": code,
        "duplicate_resolution": duplicate_block,
        "duplicate_audit": duplicate_audit,
        "authority": dict(AUTHORITY),
        "blockers": [
            "fitted_metric_weights_missing",
            "complete_chronological_evaluation_missing",
            "current_rating_comparison_missing",
            "downstream_integration_missing",
            "independent_promotion_receipt_missing",
        ],
    }
    run["receipt_sha256"] = _sha256_bytes(_canonical(run))
    _write_json(out / RUN_FILE, run)
    return run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--resolution-spec", "--duplicate-spec", dest="resolution_spec", type=Path, required=True)
    parser.add_argument("--annual-root", type=Path)
    parser.add_argument("--bridge-root", type=Path)
    parser.add_argument("--source-as-of")
    parser.add_argument("--expected-unfiltered-count", type=int)
    parser.add_argument("--expected-accepted-count", type=int)
    args = parser.parse_args(argv)
    try:
        run = build_source_trust(
            source_root=args.source_root,
            output_root=args.output_root,
            resolution_spec=args.resolution_spec,
            annual_root=args.annual_root,
            bridge_root=args.bridge_root,
            source_as_of=args.source_as_of,
            expected_unfiltered_count=args.expected_unfiltered_count,
            expected_accepted_count=args.expected_accepted_count,
        )
    except (SourceTrustError, OSError, FutureValueTrainingError) as error:
        parser.error(str(error))
        return 2
    print(
        json.dumps(
            {
                "output_root": str(Path(args.output_root).resolve()),
                "status": run["status"],
                "source_game_count": run["source_game_count"],
                "model_eligible_game_count": run["model_eligible_game_count"],
                "source_receipt_sha256": run["source_receipt_sha256"],
                "receipt_sha256": run["receipt_sha256"],
                "authority": run["authority"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
