"""Lock the system-clocked v3 prediction and ledger capture path."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .multileague_v3_capture_registry_v2 import (
    REGISTERED_CAPTURE_ARTIFACT_SHA256 as SUPERSEDED_CAPTURE_ARTIFACT_SHA256,
    REGISTERED_CAPTURE_LOCATOR as SUPERSEDED_CAPTURE_LOCATOR,
    REGISTERED_CAPTURE_RAW_SHA256 as SUPERSEDED_CAPTURE_RAW_SHA256,
)
from .multileague_v3_future_protocol import FUTURE_SEALED_START
from .multileague_v3_prediction_ledger import (
    AUTHORITY_KEYS,
    DEFAULT_REGISTRY,
    MODEL_IDS,
    RECEIPT_SCHEMA_VERSION,
    REGISTRY_SCHEMA_VERSION,
)
from .multileague_v3_preflight_v3_registry import (
    REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
    REGISTERED_PREFLIGHT_LOCATOR,
    validate_registered_source_preflight_v3,
)
from .multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    validate_registered_future_protocol_v3,
)
from .multileague_v3_temporal_failure_registry import (
    REGISTERED_FAILURE_ARTIFACT_SHA256,
    REGISTERED_FAILURE_LOCATOR,
    validate_registered_temporal_failure,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-capture-readiness:v3"
RESULT_STATE = "SYSTEM_CLOCKED_PRE_EVENT_CAPTURE_IMPLEMENTATION_READY_EMPTY_LEDGER"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_capture_readiness_v3.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/capture-readiness-v3.json"
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
    "lol_kills/pregame_roster_capture.py",
    "lol_kills/etl/leaguepedia_patch_revisions.py",
    "lol_kills/v2/ratings/player/multileague_v3_capture_readiness_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_capture_registry_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_registry.py",
    SUPERSEDED_CAPTURE_LOCATOR.as_posix(),
    REGISTERED_PREFLIGHT_LOCATOR.as_posix(),
    REGISTERED_PROTOCOL_LOCATOR.as_posix(),
    REGISTERED_FAILURE_LOCATOR.as_posix(),
)


class CaptureReadinessV3Error(RuntimeError):
    """The system-clocked capture implementation or lineage drifted."""


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CaptureReadinessV3Error(
            "capture readiness value is not canonical"
        ) from exc
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise CaptureReadinessV3Error(f"capture source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256(path),
    }


def _time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureReadinessV3Error(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise CaptureReadinessV3Error(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(
    clock: Callable[[], datetime],
    *,
    protocol_time: datetime,
    preflight_time: datetime,
) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CaptureReadinessV3Error(
            "builder clock must return a timezone-aware datetime"
        )
    observed = value.astimezone(timezone.utc)
    if observed <= max(protocol_time, preflight_time):
        raise CaptureReadinessV3Error(
            "capture clock must follow protocol and corrected preflight"
        )
    if observed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise CaptureReadinessV3Error(
            "capture implementation must be locked before the future boundary"
        )
    return observed


def build_capture_readiness_v3(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    protocol = validate_registered_future_protocol_v3(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    failure = validate_registered_temporal_failure(root=root)
    lock_time = _clock_sample(
        clock,
        protocol_time=_time(protocol["locked_at_utc"], "protocol.locked_at"),
        preflight_time=_time(preflight["built_at_utc"], "preflight.built_at"),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": lock_time.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": lock_time.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "lock_time_not_after_builder_observation": True,
        },
        "supersession": {
            "superseded_capture_locator": SUPERSEDED_CAPTURE_LOCATOR.as_posix(),
            "superseded_capture_raw_sha256": SUPERSEDED_CAPTURE_RAW_SHA256,
            "superseded_capture_artifact_sha256": (
                SUPERSEDED_CAPTURE_ARTIFACT_SHA256
            ),
            "supersession_reason": (
                "prediction_and_ledger_user_timestamp_inputs_removed"
            ),
            "superseded_capture_qualifies_as_current_implementation_evidence": False,
            "prediction_receipt_schema_changed": True,
            "prediction_ledger_schema_changed": True,
            "candidate_changed": False,
            "future_boundary_changed": False,
            "future_outcomes_used_for_hardening": False,
            "temporal_failure_artifact_sha256": REGISTERED_FAILURE_ARTIFACT_SHA256,
        },
        "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "source_preflight_artifact_sha256": REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
        "capture_contract": {
            "prediction_receipt_schema": RECEIPT_SCHEMA_VERSION,
            "outcome_free_ledger_schema": REGISTRY_SCHEMA_VERSION,
            "ledger_locator": DEFAULT_REGISTRY.as_posix(),
            "model_ids": list(MODEL_IDS),
            "exact_pre_event_roster_receipt_required": True,
            "pre_event_patch_revision_receipt_required": True,
            "candidate_and_both_comparators_captured": True,
            "fixture_side_and_player_identity_bound": True,
            "source_candidate_patch_roster_and_fixture_hash_bound": True,
            "event_outcome_fields_rejected": True,
            "prediction_system_clock_sampled_inside_builder": True,
            "prediction_cli_user_timestamp_argument_present": False,
            "ledger_system_clock_sampled_inside_builder": True,
            "ledger_builder_user_timestamp_argument_present": False,
            "receipt_validation_rechecks_evidence_protocol_and_event_order": True,
            "ledger_validation_rechecks_receipt_creation_order": True,
            "no_clobber_writes": True,
            "deterministic_replay_available": True,
            "retrospective_backfill_qualifies": False,
        },
        "ledger_state": {
            "status": "EMPTY_NOT_YET_CREATED",
            "entries": 0,
            "metadata_support_met": False,
            "outcomes_present": False,
            "outcomes_accessed": False,
            "independently_pinned": False,
            "opening_authority": False,
        },
        "implementation": {
            "ready_for_pre_event_capture": True,
            "actual_future_prediction_evidence_present": False,
            "independent_protocol_review_present": False,
            "independent_opening_approval_present": False,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "player_rating_authority": None,
            "team_rating_authority": None,
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": (
            "This receipt proves only that system-clocked, outcome-free pre-event "
            "capture machinery exists. The ledger is empty and unreviewed, so it "
            "grants no rating, probability, recommendation, or betting authority."
        ),
    }
    if (failure.get("policy") or {}).get("artifacts_qualify_as_future_evidence") is not False:
        raise CaptureReadinessV3Error("temporal-failure policy changed")
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_capture_readiness_v3(payload, root=root)


def validate_capture_readiness_v3(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CaptureReadinessV3Error("capture readiness must be an object")
    value = dict(payload)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise CaptureReadinessV3Error("capture readiness identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise CaptureReadinessV3Error("capture readiness canonical hash mismatch")
    lock_time = _time(str(value.get("locked_at_utc")), "locked_at_utc")
    protocol = validate_registered_future_protocol_v3(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    failure = validate_registered_temporal_failure(root=root)
    if (
        value.get("protocol_artifact_sha256") != protocol.get("artifact_sha256")
        or value.get("source_preflight_artifact_sha256")
        != preflight.get("artifact_sha256")
        or lock_time <= _time(protocol["locked_at_utc"], "protocol.locked_at")
        or lock_time <= _time(preflight["built_at_utc"], "preflight.built_at")
        or lock_time >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    ):
        raise CaptureReadinessV3Error("capture readiness lineage changed")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": lock_time.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise CaptureReadinessV3Error("capture clock attestation changed")
    supersession = value.get("supersession") or {}
    if (
        supersession.get("superseded_capture_raw_sha256")
        != SUPERSEDED_CAPTURE_RAW_SHA256
        or supersession.get("superseded_capture_artifact_sha256")
        != SUPERSEDED_CAPTURE_ARTIFACT_SHA256
        or supersession.get("temporal_failure_artifact_sha256")
        != failure.get("artifact_sha256")
        or supersession.get("supersession_reason")
        != "prediction_and_ledger_user_timestamp_inputs_removed"
        or supersession.get(
            "superseded_capture_qualifies_as_current_implementation_evidence"
        )
        is not False
        or supersession.get("prediction_receipt_schema_changed") is not True
        or supersession.get("prediction_ledger_schema_changed") is not True
        or supersession.get("candidate_changed") is not False
        or supersession.get("future_boundary_changed") is not False
        or supersession.get("future_outcomes_used_for_hardening") is not False
    ):
        raise CaptureReadinessV3Error("capture supersession lineage changed")
    contract = value.get("capture_contract") or {}
    required_true = {
        "exact_pre_event_roster_receipt_required",
        "pre_event_patch_revision_receipt_required",
        "candidate_and_both_comparators_captured",
        "fixture_side_and_player_identity_bound",
        "source_candidate_patch_roster_and_fixture_hash_bound",
        "event_outcome_fields_rejected",
        "prediction_system_clock_sampled_inside_builder",
        "ledger_system_clock_sampled_inside_builder",
        "receipt_validation_rechecks_evidence_protocol_and_event_order",
        "ledger_validation_rechecks_receipt_creation_order",
        "no_clobber_writes",
        "deterministic_replay_available",
    }
    if (
        any(contract.get(name) is not True for name in required_true)
        or contract.get("prediction_cli_user_timestamp_argument_present") is not False
        or contract.get("ledger_builder_user_timestamp_argument_present") is not False
        or contract.get("retrospective_backfill_qualifies") is not False
        or contract.get("prediction_receipt_schema") != RECEIPT_SCHEMA_VERSION
        or contract.get("outcome_free_ledger_schema") != REGISTRY_SCHEMA_VERSION
        or contract.get("ledger_locator") != DEFAULT_REGISTRY.as_posix()
        or contract.get("model_ids") != list(MODEL_IDS)
    ):
        raise CaptureReadinessV3Error("capture contract changed")
    if value.get("ledger_state") != {
        "status": "EMPTY_NOT_YET_CREATED",
        "entries": 0,
        "metadata_support_met": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
    }:
        raise CaptureReadinessV3Error("empty ledger state changed")
    if value.get("implementation") != {
        "ready_for_pre_event_capture": True,
        "actual_future_prediction_evidence_present": False,
        "independent_protocol_review_present": False,
        "independent_opening_approval_present": False,
    }:
        raise CaptureReadinessV3Error("capture implementation status changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise CaptureReadinessV3Error("capture readiness exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise CaptureReadinessV3Error("capture readiness contains decision outputs")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise CaptureReadinessV3Error("capture source inventory changed")
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != list(SOURCE_LOCKS):
        raise CaptureReadinessV3Error("capture source order changed")
    for record in records:
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("raw_sha256")
        ):
            raise CaptureReadinessV3Error(
                f"capture readiness source drifted: {locator}"
            )
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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
            raise FileExistsError(
                f"refusing to replace capture readiness v3 artifact: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_capture_readiness_v3()
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "result_state": payload["result_state"],
                "ledger_entries": 0,
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CaptureReadinessV3Error",
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_capture_readiness_v3",
    "validate_capture_readiness_v3",
    "write_no_clobber",
]
