"""Lock the outcome-free terminal Draft Score future-capture implementation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.v2.data.common import sha256_canonical_object
from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v3 import (
    REGISTERED_CAPTURE_ARTIFACT_SHA256 as RATINGS_CAPTURE_ARTIFACT_SHA256,
    REGISTERED_CAPTURE_LOCATOR as RATINGS_CAPTURE_LOCATOR,
    REGISTERED_CAPTURE_LOCKED_AT_UTC as RATINGS_CAPTURE_LOCKED_AT_UTC,
    REGISTERED_CAPTURE_RAW_SHA256 as RATINGS_CAPTURE_RAW_SHA256,
    validate_registered_capture_readiness_v3,
)
from lol_kills.v2.ratings.player.multileague_v3_future_protocol import (
    FUTURE_SEALED_START,
)

from .future_prediction_ledger import (
    AUTHORITY_KEYS,
    DEFAULT_LEDGER,
    LEDGER_SCHEMA_VERSION,
    MAP_START_SCHEMA_VERSION,
    PREDICTION_SCHEMA_VERSION,
    RATING_MODEL_ID,
    build_prediction_ledger,
)
from .future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:draft-terminal-future-capture-readiness:v1"
RESULT_STATE = (
    "SYSTEM_CLOCKED_TERMINAL_DRAFT_CAPTURE_IMPLEMENTATION_READY_EMPTY_LEDGER"
)
SOURCE_LOCATOR = "lol_kills/v2/draft/terminal/capture_readiness_v1.py"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/draft-terminal/future-capture-readiness-v1.json"
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/v2/draft/terminal/future_prediction_ledger.py",
    "lol_kills/v2/draft/terminal/future_protocol_registry_v1.py",
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
    "lol_kills/v2/ratings/player/multileague_v3_capture_registry_v3.py",
    REGISTERED_PROTOCOL_LOCATOR.as_posix(),
    RATINGS_CAPTURE_LOCATOR.as_posix(),
)
CLAIM_CEILING = (
    "This receipt proves only that system-clocked, outcome-free future "
    "Draft Score capture machinery exists. The ledger is empty and "
    "unreviewed, so it grants no probability, odds, expected-value, "
    "recommendation, or betting authority."
)


class DraftCaptureReadinessError(RuntimeError):
    """The future Draft Score capture implementation or lineage drifted."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise DraftCaptureReadinessError(
            f"Draft capture source is unavailable: {locator}"
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
        raise DraftCaptureReadinessError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise DraftCaptureReadinessError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(
    clock: Callable[[], datetime],
    *,
    draft_protocol_time: datetime,
    ratings_capture_time: datetime,
) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise DraftCaptureReadinessError(
            "capture-readiness clock must return a timezone-aware datetime"
        )
    observed = value.astimezone(timezone.utc)
    if observed <= max(draft_protocol_time, ratings_capture_time):
        raise DraftCaptureReadinessError(
            "Draft capture readiness must follow both frozen inputs"
        )
    if observed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise DraftCaptureReadinessError(
            "Draft capture readiness must be locked before the future boundary"
        )
    return observed


def _capture_contract() -> dict[str, Any]:
    return {
        "prediction_receipt_schema": PREDICTION_SCHEMA_VERSION,
        "actual_map_start_receipt_schema": MAP_START_SCHEMA_VERSION,
        "outcome_free_ledger_schema": LEDGER_SCHEMA_VERSION,
        "ledger_locator": DEFAULT_LEDGER.as_posix(),
        "ratings_candidate_id": RATING_MODEL_ID,
        "exact_frozen_ratings_prediction_receipt_required": True,
        "exact_terminal_draft_payload_bytes_and_hash_required": True,
        "terminal_draft_payload_must_be_strict_outcome_free_json": True,
        "terminal_draft_source_rights_review_required": True,
        "terminal_action_order_and_pick_ban_validation_required": True,
        "exact_five_role_assignments_per_side_required": True,
        "fixture_team_side_patch_and_map_identity_bound": True,
        "draft_model_and_source_must_precede_prediction": True,
        "actual_map_start_authority_captured_separately": True,
        "actual_map_start_source_must_not_predate_claimed_start": True,
        "prediction_must_strictly_precede_actual_map_start": True,
        "prediction_system_clock_sampled_inside_builder": True,
        "map_start_system_clock_sampled_inside_builder": True,
        "ledger_system_clock_sampled_inside_builder": True,
        "prediction_cli_user_timestamp_argument_present": False,
        "map_start_cli_user_timestamp_argument_present": False,
        "ledger_cli_user_timestamp_argument_present": False,
        "ledger_validation_reloads_and_revalidates_bound_receipt_bytes": True,
        "event_outcome_fields_rejected_recursively": True,
        "source_payload_outcome_fields_rejected_recursively": True,
        "atomic_no_clobber_writes": True,
        "deterministic_prediction_replay_available": True,
        "neutral_output_directly_outcome_calibrated": False,
        "retrospective_backfill_qualifies": False,
    }


def _ledger_state() -> dict[str, Any]:
    return {
        "status": "EMPTY_NOT_YET_CREATED",
        "entries": 0,
        "metadata_support_met": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
    }


def _implementation_state() -> dict[str, Any]:
    return {
        "ready_for_outcome_free_future_capture": True,
        "actual_future_prediction_evidence_present": False,
        "independent_protocol_review_present": False,
        "independent_ledger_pin_present": False,
        "independent_opening_approval_present": False,
    }


def _decision_outputs() -> dict[str, None]:
    return {
        "incremental_draft_authority": None,
        "neutral_probability": None,
        "contextual_probability": None,
        "fair_odds": None,
        "expected_value": None,
        "bet_recommendation": None,
    }


def build_capture_readiness_v1(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    protocol = validate_registered_future_protocol_v1(root=root)
    ratings_capture = validate_registered_capture_readiness_v3(root=root)
    locked = _clock_sample(
        clock,
        draft_protocol_time=_timestamp(
            protocol["locked_at_utc"], "draft_protocol.locked_at_utc"
        ),
        ratings_capture_time=_timestamp(
            ratings_capture["locked_at_utc"], "ratings_capture.locked_at_utc"
        ),
    )
    if (root / DEFAULT_LEDGER).exists():
        raise DraftCaptureReadinessError(
            "Draft prediction ledger already exists; empty readiness cannot be locked"
        )
    empty_ledger = build_prediction_ledger(
        receipts=[],
        root=root,
        clock=lambda: locked,
    )
    if (
        empty_ledger.get("metadata_support", {}).get("eligible_maps") != 0
        or empty_ledger.get("outcomes_present") is not False
        or any((empty_ledger.get("authority") or {}).values())
    ):
        raise DraftCaptureReadinessError(
            "empty Draft ledger template exceeded its evidence boundary"
        )
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
        "draft_protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "locked_at_utc": REGISTERED_PROTOCOL_LOCKED_AT_UTC,
        },
        "ratings_capture": {
            "locator": RATINGS_CAPTURE_LOCATOR.as_posix(),
            "raw_sha256": RATINGS_CAPTURE_RAW_SHA256,
            "artifact_sha256": RATINGS_CAPTURE_ARTIFACT_SHA256,
            "locked_at_utc": RATINGS_CAPTURE_LOCKED_AT_UTC,
        },
        "capture_contract": _capture_contract(),
        "empty_ledger_template": {
            "schema_version": empty_ledger["schema_version"],
            "status": empty_ledger["status"],
            "artifact_sha256": empty_ledger["artifact_sha256"],
            "created_at_utc": empty_ledger["created_at_utc"],
            "entries": 0,
        },
        "ledger_state": _ledger_state(),
        "implementation": _implementation_state(),
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": _decision_outputs(),
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_capture_readiness_v1(payload, root=root)


def validate_capture_readiness_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise DraftCaptureReadinessError("Draft capture readiness must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "locked_at_utc",
        "clock_attestation",
        "draft_protocol",
        "ratings_capture",
        "capture_contract",
        "empty_ledger_template",
        "ledger_state",
        "implementation",
        "source_locks",
        "decision_outputs",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise DraftCaptureReadinessError(
            "Draft capture readiness structure changed"
        )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise DraftCaptureReadinessError("Draft capture readiness identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != sha256_canonical_object(unsigned):
        raise DraftCaptureReadinessError(
            "Draft capture readiness canonical hash changed"
        )
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    protocol = validate_registered_future_protocol_v1(root=root)
    ratings_capture = validate_registered_capture_readiness_v3(root=root)
    protocol_time = _timestamp(
        protocol["locked_at_utc"], "draft_protocol.locked_at_utc"
    )
    ratings_time = _timestamp(
        ratings_capture["locked_at_utc"], "ratings_capture.locked_at_utc"
    )
    if not max(protocol_time, ratings_time) < locked < FUTURE_SEALED_START.replace(
        tzinfo=timezone.utc
    ):
        raise DraftCaptureReadinessError("Draft capture readiness time order changed")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise DraftCaptureReadinessError(
            "Draft capture readiness clock attestation changed"
        )
    if value.get("draft_protocol") != {
        "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "locked_at_utc": REGISTERED_PROTOCOL_LOCKED_AT_UTC,
    } or protocol.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256:
        raise DraftCaptureReadinessError("Draft protocol binding changed")
    if value.get("ratings_capture") != {
        "locator": RATINGS_CAPTURE_LOCATOR.as_posix(),
        "raw_sha256": RATINGS_CAPTURE_RAW_SHA256,
        "artifact_sha256": RATINGS_CAPTURE_ARTIFACT_SHA256,
        "locked_at_utc": RATINGS_CAPTURE_LOCKED_AT_UTC,
    } or ratings_capture.get("artifact_sha256") != RATINGS_CAPTURE_ARTIFACT_SHA256:
        raise DraftCaptureReadinessError("ratings capture binding changed")
    if value.get("capture_contract") != _capture_contract():
        raise DraftCaptureReadinessError("Draft capture contract changed")
    empty_ledger = build_prediction_ledger(
        receipts=[],
        root=root,
        clock=lambda: locked,
    )
    if value.get("empty_ledger_template") != {
        "schema_version": empty_ledger["schema_version"],
        "status": empty_ledger["status"],
        "artifact_sha256": empty_ledger["artifact_sha256"],
        "created_at_utc": empty_ledger["created_at_utc"],
        "entries": 0,
    }:
        raise DraftCaptureReadinessError("empty Draft ledger template changed")
    if value.get("ledger_state") != _ledger_state():
        raise DraftCaptureReadinessError("empty Draft ledger state changed")
    if value.get("implementation") != _implementation_state():
        raise DraftCaptureReadinessError("Draft capture implementation state changed")
    records = value.get("source_locks")
    if (
        not isinstance(records, list)
        or len(records) != len(SOURCE_LOCKS)
        or [
            item.get("locator") for item in records if isinstance(item, Mapping)
        ]
        != list(SOURCE_LOCKS)
    ):
        raise DraftCaptureReadinessError("Draft capture source inventory changed")
    for record in records:
        if not isinstance(record, Mapping) or set(record) != {
            "locator",
            "bytes",
            "raw_sha256",
        }:
            raise DraftCaptureReadinessError(
                "Draft capture source record is malformed"
            )
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256_path(path) != record.get("raw_sha256")
        ):
            raise DraftCaptureReadinessError(
                f"Draft capture source drifted: {locator}"
            )
    if value.get("decision_outputs") != _decision_outputs():
        raise DraftCaptureReadinessError(
            "Draft capture readiness contains decision outputs"
        )
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(authority.values()):
        raise DraftCaptureReadinessError("Draft capture readiness exceeds authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise DraftCaptureReadinessError("Draft capture claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise DraftCaptureReadinessError(
                f"refusing to overwrite Draft capture readiness: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        payload = build_capture_readiness_v1(root=args.root)
        output = args.out if args.out.is_absolute() else args.root / args.out
        raw_sha256 = write_no_clobber(output, payload)
    except (OSError, ValueError, DraftCaptureReadinessError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(output),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "result_state": payload["result_state"],
                "ledger_entries": 0,
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
    "DraftCaptureReadinessError",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_capture_readiness_v1",
    "validate_capture_readiness_v1",
    "write_no_clobber",
]
