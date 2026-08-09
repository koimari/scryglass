"""Lock the empty outcome-free phase-one collection implementation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.v2.draft.terminal.capture_readiness_registry_v1 import (
    REGISTERED_CAPTURE_ARTIFACT_SHA256 as DRAFT_CAPTURE_ARTIFACT_SHA256,
    REGISTERED_CAPTURE_LOCATOR as DRAFT_CAPTURE_LOCATOR,
    REGISTERED_CAPTURE_LOCKED_AT_UTC as DRAFT_CAPTURE_LOCKED_AT_UTC,
    REGISTERED_CAPTURE_RAW_SHA256 as DRAFT_CAPTURE_RAW_SHA256,
    validate_registered_capture_readiness_v1 as validate_draft_capture,
)
from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v3 import (
    REGISTERED_CAPTURE_ARTIFACT_SHA256 as RATINGS_CAPTURE_ARTIFACT_SHA256,
    REGISTERED_CAPTURE_LOCATOR as RATINGS_CAPTURE_LOCATOR,
    REGISTERED_CAPTURE_LOCKED_AT_UTC as RATINGS_CAPTURE_LOCKED_AT_UTC,
    REGISTERED_CAPTURE_RAW_SHA256 as RATINGS_CAPTURE_RAW_SHA256,
    validate_registered_capture_readiness_v3 as validate_ratings_capture,
)
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)

from . import phase_one_collection_v1 as collection
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as MARKET_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as MARKET_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC as MARKET_PROTOCOL_LOCKED_AT_UTC,
    REGISTERED_PROTOCOL_RAW_SHA256 as MARKET_PROTOCOL_RAW_SHA256,
    validate_registered_match_winner_future_protocol_v1 as validate_market_protocol,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "scryglass:match-winner-phase-one-collection-readiness:v1"
RESULT_STATE = "OUTCOME_FREE_PHASE_ONE_COLLECTION_IMPLEMENTATION_READY_EMPTY"
SOURCE_LOCATOR = (
    "lol_kills/v2/market/phase_one_collection_readiness_v1.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
    "capture-readiness-v1.json"
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    collection.SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
    "lol_kills/v2/draft/terminal/future_prediction_ledger.py",
    "lol_kills/v2/ratings/player/multileague_v3_capture_registry_v3.py",
    "lol_kills/v2/draft/terminal/capture_readiness_registry_v1.py",
    "lol_kills/v2/market/match_winner_future_protocol_registry_v1.py",
    RATINGS_CAPTURE_LOCATOR.as_posix(),
    DRAFT_CAPTURE_LOCATOR.as_posix(),
    MARKET_PROTOCOL_LOCATOR.as_posix(),
)
CLAIM_CEILING = (
    "This receipt proves only that the system-clocked outcome-free phase-one "
    "collection implementation was frozen with zero plans, event bundles, and "
    "joint snapshots. It grants no validation, opening, rating, probability, "
    "odds, expected-value, recommendation, or betting authority."
)


class PhaseOneCollectionReadinessError(RuntimeError):
    """The phase-one collection readiness receipt or lineage drifted."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseOneCollectionReadinessError(
            "collection readiness value is not canonical"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise PhaseOneCollectionReadinessError(
            f"collection readiness source is unavailable: {locator}"
        )
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseOneCollectionReadinessError(
            f"{field} must be RFC-3339"
        ) from exc
    if parsed.tzinfo is None:
        raise PhaseOneCollectionReadinessError(
            f"{field} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _dependency_records(root: Path) -> dict[str, Any]:
    ratings = validate_ratings_capture(root=root)
    draft = validate_draft_capture(root=root)
    market = validate_market_protocol(root=root)
    return {
        "ratings_capture": {
            "locator": RATINGS_CAPTURE_LOCATOR.as_posix(),
            "raw_sha256": RATINGS_CAPTURE_RAW_SHA256,
            "artifact_sha256": RATINGS_CAPTURE_ARTIFACT_SHA256,
            "locked_at_utc": RATINGS_CAPTURE_LOCKED_AT_UTC,
            "result_state": ratings["result_state"],
        },
        "terminal_draft_capture": {
            "locator": DRAFT_CAPTURE_LOCATOR.as_posix(),
            "raw_sha256": DRAFT_CAPTURE_RAW_SHA256,
            "artifact_sha256": DRAFT_CAPTURE_ARTIFACT_SHA256,
            "locked_at_utc": DRAFT_CAPTURE_LOCKED_AT_UTC,
            "result_state": draft["result_state"],
        },
        "match_winner_protocol": {
            "locator": MARKET_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": MARKET_PROTOCOL_RAW_SHA256,
            "artifact_sha256": MARKET_PROTOCOL_ARTIFACT_SHA256,
            "locked_at_utc": MARKET_PROTOCOL_LOCKED_AT_UTC,
            "result_state": market["result_state"],
        },
    }


def _clock_sample(
    clock: Callable[[], datetime], dependencies: Mapping[str, Any]
) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PhaseOneCollectionReadinessError(
            "readiness clock must return a timezone-aware datetime"
        )
    locked = observed.astimezone(timezone.utc)
    latest_dependency = max(
        _timestamp(record["locked_at_utc"], f"{name}.locked_at_utc")
        for name, record in dependencies.items()
    )
    if locked <= latest_dependency:
        raise PhaseOneCollectionReadinessError(
            "collection readiness must follow every frozen dependency"
        )
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise PhaseOneCollectionReadinessError(
            "collection readiness must be locked before the future boundary"
        )
    return locked


def _contract() -> dict[str, Any]:
    signatures = {
        "plan": list(inspect.signature(collection.build_event_plan).parameters),
        "event_bundle": list(
            inspect.signature(collection.build_event_bundle).parameters
        ),
        "joint_snapshot": list(
            inspect.signature(collection.build_joint_ledger_snapshot).parameters
        ),
    }
    if signatures != {
        "plan": ["ratings_prediction_locator", "root", "clock"],
        "event_bundle": ["plan_locator", "root", "clock"],
        "joint_snapshot": [
            "bundle_locators",
            "snapshot_locator",
            "root",
            "clock",
        ],
    }:
        raise PhaseOneCollectionReadinessError(
            "phase-one builder signatures changed"
        )
    return {
        "plan_schema_version": collection.PLAN_SCHEMA_VERSION,
        "event_bundle_schema_version": collection.BUNDLE_SCHEMA_VERSION,
        "joint_ledger_schema_version": collection.SNAPSHOT_SCHEMA_VERSION,
        "plan_prefix": collection.PLAN_PREFIX.as_posix(),
        "event_bundle_prefix": collection.BUNDLE_PREFIX.as_posix(),
        "joint_snapshot_prefix": collection.SNAPSHOT_PREFIX.as_posix(),
        "builder_parameters": signatures,
        "ratings_receipt_must_preexist_plan": True,
        "exact_ratings_bytes_must_match_terminal_draft_embedding": True,
        "exact_event_series_game_league_patch_side_and_team_join_required": True,
        "draft_prediction_must_strictly_precede_actual_map_start": True,
        "map_start_capture_must_be_at_or_after_actual_start": True,
        "event_outcome_fields_rejected_recursively": True,
        "artifact_files_must_be_unaliased_regular_files": True,
        "artifact_symlinks_rejected": True,
        "atomic_fsynced_no_clobber_writes": True,
        "joint_snapshot_rebuilds_registered_ratings_ledger": True,
        "joint_snapshot_rebuilds_registered_draft_ledger": True,
        "all_builder_clocks_sampled_inside_implementation": True,
        "cli_user_capture_or_creation_timestamp_present": False,
        "retrospective_backfill_qualifies": False,
        "metadata_support_itself_authorizes_opening": False,
    }


def _count_json(root: Path, prefix: object) -> int:
    directory = root / Path(str(prefix))
    if not directory.exists():
        return 0
    if not directory.is_dir() or directory.is_symlink():
        raise PhaseOneCollectionReadinessError(
            "phase-one collection directory is not a real directory"
        )
    return sum(1 for path in directory.rglob("*.json") if path.exists())


def _empty_collection_state(root: Path) -> dict[str, Any]:
    state = {
        "plans": _count_json(root, collection.PLAN_PREFIX),
        "event_bundles": _count_json(root, collection.BUNDLE_PREFIX),
        "joint_snapshots": _count_json(root, collection.SNAPSHOT_PREFIX),
        "outcomes_present": False,
        "outcomes_accessed": False,
        "metadata_support_met": False,
        "independently_pinned": False,
        "opening_authority": False,
    }
    if any(state[name] != 0 for name in ("plans", "event_bundles", "joint_snapshots")):
        raise PhaseOneCollectionReadinessError(
            "empty collection readiness cannot be locked after collection starts"
        )
    return state


def _implementation_state() -> dict[str, bool]:
    return {
        "ready_for_outcome_free_phase_one_collection": True,
        "actual_future_evidence_present": False,
        "independent_collection_review_present": False,
        "independent_ledger_pin_present": False,
        "independent_opening_approval_present": False,
    }


def _decision_outputs() -> dict[str, None]:
    return {
        "ratings_validation_authority": None,
        "draft_validation_authority": None,
        "outcome_opening_authority": None,
        "match_probability": None,
        "fair_odds": None,
        "expected_value": None,
        "bet_recommendation": None,
    }


def build_phase_one_collection_readiness_v1(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    dependencies = _dependency_records(root)
    locked = _clock_sample(clock, dependencies)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": locked.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": locked.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "lock_time_not_after_builder_observation": True,
        },
        "dependencies": dependencies,
        "collection_contract": _contract(),
        "locked_empty_collection_state": _empty_collection_state(root),
        "implementation": _implementation_state(),
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": _decision_outputs(),
        "authority": {name: False for name in collection.AUTHORITY_KEYS},
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_one_collection_readiness_v1(payload, root=root)


def validate_phase_one_collection_readiness_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness must be an object"
        )
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "locked_at_utc",
        "clock_attestation",
        "dependencies",
        "collection_contract",
        "locked_empty_collection_state",
        "implementation",
        "source_locks",
        "decision_outputs",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness structure changed"
        )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness identity changed"
        )
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness hash changed"
        )
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    if locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness was not locked pre-boundary"
        )
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness clock changed"
        )
    dependencies = _dependency_records(root)
    if value.get("dependencies") != dependencies:
        raise PhaseOneCollectionReadinessError(
            "phase-one collection dependency binding changed"
        )
    if locked <= max(
        _timestamp(record["locked_at_utc"], f"{name}.locked_at_utc")
        for name, record in dependencies.items()
    ):
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness predates a dependency"
        )
    if value.get("collection_contract") != _contract():
        raise PhaseOneCollectionReadinessError(
            "phase-one collection contract changed"
        )
    if value.get("locked_empty_collection_state") != {
        "plans": 0,
        "event_bundles": 0,
        "joint_snapshots": 0,
        "outcomes_present": False,
        "outcomes_accessed": False,
        "metadata_support_met": False,
        "independently_pinned": False,
        "opening_authority": False,
    }:
        raise PhaseOneCollectionReadinessError(
            "phase-one locked empty collection state changed"
        )
    if value.get("implementation") != _implementation_state():
        raise PhaseOneCollectionReadinessError(
            "phase-one implementation claim changed"
        )
    records = value.get("source_locks")
    expected_records = [_source_record(root, locator) for locator in SOURCE_LOCKS]
    if records != expected_records:
        raise PhaseOneCollectionReadinessError(
            "phase-one collection source inventory drifted"
        )
    if value.get("decision_outputs") != _decision_outputs():
        raise PhaseOneCollectionReadinessError(
            "phase-one collection decision outputs changed"
        )
    authority = value.get("authority") or {}
    if set(authority) != set(collection.AUTHORITY_KEYS) or any(authority.values()):
        raise PhaseOneCollectionReadinessError(
            "phase-one collection readiness exceeds authority"
        )
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseOneCollectionReadinessError(
            "phase-one collection claim ceiling changed"
        )
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseOneCollectionReadinessError(
            f"refusing to replace collection readiness: {path}"
        )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise PhaseOneCollectionReadinessError(
                f"refusing to replace collection readiness: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return hashlib.sha256(raw).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = build_phase_one_collection_readiness_v1(root=args.root)
        raw_sha256 = write_no_clobber(args.out, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "future_evidence_present": False,
                "opening_authority": False,
                "betting_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "PhaseOneCollectionReadinessError",
    "build_phase_one_collection_readiness_v1",
    "validate_phase_one_collection_readiness_v1",
]
