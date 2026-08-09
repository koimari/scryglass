"""Clock-check the v3 pre-event capture path while its ledger is empty."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .multileague_v3_capture_registry import (
    REGISTERED_CAPTURE_ARTIFACT_SHA256 as REJECTED_CAPTURE_ARTIFACT_SHA256,
    REGISTERED_CAPTURE_LOCATOR as REJECTED_CAPTURE_LOCATOR,
    REGISTERED_CAPTURE_RAW_SHA256 as REJECTED_CAPTURE_RAW_SHA256,
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
    REGISTERED_FAILURE_RAW_SHA256,
    validate_registered_temporal_failure,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-capture-readiness:v2"
RESULT_STATE = "CLOCK_CORRECTED_PRE_EVENT_CAPTURE_IMPLEMENTATION_READY_EMPTY_LEDGER"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_capture_readiness_v2.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/capture-readiness-v2.json"
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py",
    "lol_kills/pregame_roster_capture.py",
    "lol_kills/etl/leaguepedia_patch_revisions.py",
    "lol_kills/v2/ratings/player/multileague_v3_capture_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_registry.py",
    REJECTED_CAPTURE_LOCATOR.as_posix(),
    REGISTERED_PREFLIGHT_LOCATOR.as_posix(),
    REGISTERED_PROTOCOL_LOCATOR.as_posix(),
    REGISTERED_FAILURE_LOCATOR.as_posix(),
)


class CaptureReadinessV2Error(RuntimeError):
    """The clock-corrected capture path or its empty-ledger boundary drifted."""


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
        raise CaptureReadinessV2Error(
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
        raise CaptureReadinessV2Error(f"capture source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256(path),
    }


def _time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise CaptureReadinessV2Error(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise CaptureReadinessV2Error(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(
    clock: Callable[[], datetime],
    *,
    protocol_time: datetime,
    preflight_time: datetime,
) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CaptureReadinessV2Error(
            "builder clock must return a timezone-aware datetime"
        )
    observed = value.astimezone(timezone.utc)
    if observed <= max(protocol_time, preflight_time):
        raise CaptureReadinessV2Error(
            "capture clock must follow protocol and corrected preflight"
        )
    if observed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise CaptureReadinessV2Error(
            "capture implementation must be locked before the future boundary"
        )
    return observed


def build_capture_readiness_v2(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    protocol = validate_registered_future_protocol_v3(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    failure = validate_registered_temporal_failure(root=root)
    if (
        failure["policy"]["artifacts_qualify_as_future_evidence"] is not False
        or failure["policy"]["capture_implementation_must_be_relocked"] is not True
    ):
        raise CaptureReadinessV2Error("temporal-failure policy changed")
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
            "rejected_capture_locator": REJECTED_CAPTURE_LOCATOR.as_posix(),
            "rejected_capture_raw_sha256": REJECTED_CAPTURE_RAW_SHA256,
            "rejected_capture_artifact_sha256": REJECTED_CAPTURE_ARTIFACT_SHA256,
            "temporal_failure_locator": REGISTERED_FAILURE_LOCATOR.as_posix(),
            "temporal_failure_raw_sha256": REGISTERED_FAILURE_RAW_SHA256,
            "temporal_failure_artifact_sha256": REGISTERED_FAILURE_ARTIFACT_SHA256,
            "rejected_capture_qualifies_as_future_evidence": False,
            "capture_contract_changed": False,
            "future_boundary_changed": False,
            "future_outcomes_used_for_recovery": False,
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
            "This clock-corrected receipt proves only that an outcome-free pre-event "
            "capture path exists. The ledger is empty, so it supplies no future "
            "predictive evidence and no rating, probability, recommendation, or "
            "betting authority."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_capture_readiness_v2(payload, root=root)


def validate_capture_readiness_v2(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CaptureReadinessV2Error("capture readiness must be an object")
    value = dict(payload)
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise CaptureReadinessV2Error("capture readiness identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise CaptureReadinessV2Error("capture readiness canonical hash mismatch")
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
        raise CaptureReadinessV2Error("capture readiness lineage changed")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": lock_time.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise CaptureReadinessV2Error("capture clock attestation changed")
    supersession = value.get("supersession") or {}
    if (
        supersession.get("rejected_capture_raw_sha256")
        != REJECTED_CAPTURE_RAW_SHA256
        or supersession.get("rejected_capture_artifact_sha256")
        != REJECTED_CAPTURE_ARTIFACT_SHA256
        or supersession.get("temporal_failure_artifact_sha256")
        != failure.get("artifact_sha256")
        or supersession.get("rejected_capture_qualifies_as_future_evidence")
        is not False
        or any(
            supersession.get(name) is not False
            for name in (
                "capture_contract_changed",
                "future_boundary_changed",
                "future_outcomes_used_for_recovery",
            )
        )
    ):
        raise CaptureReadinessV2Error("capture supersession lineage changed")
    contract = value.get("capture_contract") or {}
    required_true = {
        "exact_pre_event_roster_receipt_required",
        "pre_event_patch_revision_receipt_required",
        "candidate_and_both_comparators_captured",
        "fixture_side_and_player_identity_bound",
        "source_candidate_patch_roster_and_fixture_hash_bound",
        "event_outcome_fields_rejected",
        "no_clobber_writes",
        "deterministic_replay_available",
    }
    if (
        any(contract.get(name) is not True for name in required_true)
        or contract.get("retrospective_backfill_qualifies") is not False
        or contract.get("prediction_receipt_schema") != RECEIPT_SCHEMA_VERSION
        or contract.get("outcome_free_ledger_schema") != REGISTRY_SCHEMA_VERSION
        or contract.get("ledger_locator") != DEFAULT_REGISTRY.as_posix()
        or contract.get("model_ids") != list(MODEL_IDS)
    ):
        raise CaptureReadinessV2Error("capture contract changed")
    if value.get("ledger_state") != {
        "status": "EMPTY_NOT_YET_CREATED",
        "entries": 0,
        "metadata_support_met": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
    }:
        raise CaptureReadinessV2Error("empty ledger state changed")
    if value.get("implementation") != {
        "ready_for_pre_event_capture": True,
        "actual_future_prediction_evidence_present": False,
        "independent_protocol_review_present": False,
        "independent_opening_approval_present": False,
    }:
        raise CaptureReadinessV2Error("capture implementation status changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise CaptureReadinessV2Error("capture readiness exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise CaptureReadinessV2Error("capture readiness contains decision outputs")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise CaptureReadinessV2Error("capture source inventory changed")
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != list(SOURCE_LOCKS):
        raise CaptureReadinessV2Error("capture source order changed")
    for record in records:
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("raw_sha256")
        ):
            raise CaptureReadinessV2Error(
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
                f"refusing to replace capture readiness v2 artifact: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_capture_readiness_v2()
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
    "CaptureReadinessV2Error",
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_capture_readiness_v2",
    "validate_capture_readiness_v2",
    "write_no_clobber",
]
