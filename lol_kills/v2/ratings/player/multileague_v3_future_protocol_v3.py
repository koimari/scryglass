"""Clock-checked successor to the rejected future-dated ratings protocol."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from .multileague_v3_future_protocol import (
    DOMESTIC_LEAGUES,
    FUTURE_SEALED_START,
    SELECTED_CANDIDATE_ID,
)
from .multileague_v3_preflight_v3_registry import (
    REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
    REGISTERED_PREFLIGHT_BUILT_AT_UTC,
    REGISTERED_PREFLIGHT_LOCATOR,
    REGISTERED_PREFLIGHT_RAW_SHA256,
    validate_registered_source_preflight_v3,
)
from .multileague_v3_registry import validate_registered_future_protocol
from .multileague_v3_registry_v2 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as REJECTED_PROTOCOL_V2_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as REJECTED_PROTOCOL_V2_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as REJECTED_PROTOCOL_V2_RAW_SHA256,
)
from .multileague_v3_source_registry_v2 import (
    MANIFEST_CANONICAL_SHA256,
    MANIFEST_LOCATOR,
    MANIFEST_RAW_SHA256,
    PACKAGE_ID,
    validate_registered_source_snapshot_v2,
)
from .multileague_v3_temporal_failure_registry import (
    REGISTERED_FAILURE_ARTIFACT_SHA256,
    REGISTERED_FAILURE_LOCATOR,
    REGISTERED_FAILURE_RAW_SHA256,
    validate_registered_temporal_failure,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-future-protocol-lock:v3"
RESULT_STATE = "CLOCK_CORRECTED_FUTURE_HOLDOUT_PROTOCOL_LOCKED_EMPTY"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol_v3.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/future-protocol-lock-v3.json"
)
AUTHORITY_KEYS = (
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "probability_authority",
    "recommendation_authority",
    "betting_authority",
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol.py",
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_source_registry_v2.py",
    REJECTED_PROTOCOL_V2_LOCATOR.as_posix(),
    REGISTERED_PREFLIGHT_LOCATOR.as_posix(),
    REGISTERED_FAILURE_LOCATOR.as_posix(),
    MANIFEST_LOCATOR.as_posix(),
)


class FutureProtocolV3Error(RuntimeError):
    """The clock-corrected protocol is malformed, contaminated, or unbound."""


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
        raise FutureProtocolV3Error("protocol value is not canonical") from exc
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
        raise FutureProtocolV3Error(f"bound source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256(path),
    }


def _time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FutureProtocolV3Error(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise FutureProtocolV3Error(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise FutureProtocolV3Error("builder clock must return a timezone-aware datetime")
    observed = value.astimezone(timezone.utc)
    preflight_time = _time(REGISTERED_PREFLIGHT_BUILT_AT_UTC, "preflight built_at")
    boundary = FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    if observed <= preflight_time:
        raise FutureProtocolV3Error("protocol clock must follow corrected preflight")
    if observed >= boundary:
        raise FutureProtocolV3Error("protocol clock must precede the future boundary")
    return observed


def build_future_protocol_lock_v3(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock)
    protocol_v1 = validate_registered_future_protocol(root=root)
    rejected = validate_registered_temporal_failure(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    source = validate_registered_source_snapshot_v2(root=root)
    if rejected["policy"]["artifacts_qualify_as_future_evidence"] is not False:
        raise FutureProtocolV3Error("rejected timing lineage was rehabilitated")
    candidate = protocol_v1["locked_candidate"]
    if (
        candidate.get("candidate_id") != SELECTED_CANDIDATE_ID
        or preflight["locked_candidate"]["definition"]
        != candidate.get("definition")
    ):
        raise FutureProtocolV3Error("frozen candidate changed")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": observed.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": observed.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "lock_time_not_after_builder_observation": True,
        },
        "supersession": {
            "rejected_protocol_v2_locator": REJECTED_PROTOCOL_V2_LOCATOR.as_posix(),
            "rejected_protocol_v2_raw_sha256": REJECTED_PROTOCOL_V2_RAW_SHA256,
            "rejected_protocol_v2_artifact_sha256": (
                REJECTED_PROTOCOL_V2_ARTIFACT_SHA256
            ),
            "temporal_failure_locator": REGISTERED_FAILURE_LOCATOR.as_posix(),
            "temporal_failure_raw_sha256": REGISTERED_FAILURE_RAW_SHA256,
            "temporal_failure_artifact_sha256": (
                REGISTERED_FAILURE_ARTIFACT_SHA256
            ),
            "rejected_artifacts_qualify_as_future_evidence": False,
            "candidate_changed": False,
            "future_boundary_changed": False,
            "evaluation_rule_changed": False,
            "future_outcomes_used_for_recovery": False,
        },
        "adaptation_disclosure": {
            "all_source_snapshot_outcomes_are_adaptive": True,
            "clock_corrected_preflight_is_not_independent_validation": True,
            "candidate_selection_remains_adaptive": True,
            "no_future_target_was_present_or_opened": True,
        },
        "source_snapshot": {
            "package_id": PACKAGE_ID,
            "manifest_locator": MANIFEST_LOCATOR.as_posix(),
            "manifest_raw_sha256": MANIFEST_RAW_SHA256,
            "manifest_canonical_sha256": MANIFEST_CANONICAL_SHA256,
            "maps": source["files"]["maps"],
            "players": source["files"]["players"],
            "latest_observed_source_time": preflight["source_snapshot"][
                "latest_observed_source_time"
            ],
        },
        "clock_corrected_source_preflight": {
            "locator": REGISTERED_PREFLIGHT_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PREFLIGHT_RAW_SHA256,
            "artifact_sha256": REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
            "built_at_utc": REGISTERED_PREFLIGHT_BUILT_AT_UTC,
            "result_state": preflight["result_state"],
            "accepted_maps": preflight["adapter_preflight"]["coverage"][
                "accepted_maps"
            ],
            "development_series": preflight["adapter_preflight"]["coverage"][
                "development_series"
            ],
            "latent_dimension": preflight["numerical_preflight"][
                "latent_dimension"
            ],
            "posterior_state_sha256": preflight["numerical_preflight"][
                "posterior_state_sha256"
            ],
            "preflight_is_authority": False,
        },
        "locked_candidate": {
            "candidate_id": SELECTED_CANDIDATE_ID,
            "definition": candidate["definition"],
            "original_selection_artifact_locator": candidate[
                "selection_artifact_locator"
            ],
            "original_selection_artifact_sha256": candidate[
                "selection_artifact_sha256"
            ],
            "selection_status": "adaptive_choice_unchanged_before_future_holdout",
        },
        "future_holdout": {
            "status": "EMPTY_NOT_YET_ACQUIRED",
            "start_inclusive_source_time": FUTURE_SEALED_START.isoformat(),
            "source_time_semantics": "timezone-naive warehouse timestamp",
            "series_atomic": True,
            "one_time_opening": True,
            "eligibility": {
                "professional_maps_only": True,
                "leagues": ["LCS", "LEC", "LCK", "LPL", "MSI", "EWC"],
                "exact_ten_player_identity_required": True,
                "pre_event_roster_receipt_required": True,
                "pre_event_patch_receipt_required": True,
                "pre_event_prediction_ledger_required": True,
                "candidate_and_both_comparators_must_be_captured": True,
                "prediction_timestamp_strictly_before_event_start": True,
                "prediction_must_bind_candidate_source_roster_patch_and_fixture": True,
                "prediction_must_be_generated_without_event_outcome_access": True,
                "retrospective_prediction_generation_qualifies": False,
            },
            "support_stopping_rule": {
                "overall_series_minimum": 100,
                "each_domestic_league_series_minimum": 20,
                "domestic_leagues": list(DOMESTIC_LEAGUES),
                "one_or_both_rosters_changed_series_minimum": 20,
                "stop_at_first_independently_pinned_snapshot_meeting_metadata_only_thresholds": True,
                "outcomes_must_remain_unopened_while_checking_support": True,
            },
        },
        "prediction_ledger": {
            "status": "NOT_YET_CREATED",
            "entries": 0,
            "registry_locator": None,
            "registry_raw_sha256": None,
            "pre_event_capture_implementation_present": False,
            "retrospective_backfill_permitted": False,
        },
        "evaluation": protocol_v1["evaluation"],
        "opening_authority": {
            "independent_protocol_review_present": False,
            "independent_opening_approval_present": False,
            "approval_must_externally_pin_protocol_and_prediction_ledger_sha256": True,
            "self_authorizing": False,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "sealed_evaluation": None,
            "player_rating_authority": None,
            "team_rating_authority": None,
            "probability": None,
            "odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": (
            "This clock-corrected lock creates an empty future protocol only. "
            "It grants no rating, probability, recommendation, or betting authority."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_future_protocol_lock_v3(payload, root=root)


def validate_future_protocol_lock_v3(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FutureProtocolV3Error("future protocol v3 must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise FutureProtocolV3Error("future protocol v3 identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise FutureProtocolV3Error("future protocol v3 canonical hash mismatch")
    locked_at = _time(str(value.get("locked_at_utc")), "locked_at_utc")
    if locked_at <= _time(REGISTERED_PREFLIGHT_BUILT_AT_UTC, "preflight built_at"):
        raise FutureProtocolV3Error("protocol does not follow corrected preflight")
    if locked_at >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise FutureProtocolV3Error("protocol overlaps the future boundary")
    clock_record = value.get("clock_attestation") or {}
    if clock_record != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "lock_time_not_after_builder_observation": True,
    }:
        raise FutureProtocolV3Error("clock attestation changed")
    protocol_v1 = validate_registered_future_protocol(root=root)
    rejected = validate_registered_temporal_failure(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    source = validate_registered_source_snapshot_v2(root=root)
    supersession = value.get("supersession") or {}
    if (
        supersession.get("rejected_protocol_v2_artifact_sha256")
        != REJECTED_PROTOCOL_V2_ARTIFACT_SHA256
        or supersession.get("temporal_failure_artifact_sha256")
        != rejected.get("artifact_sha256")
        or supersession.get("rejected_artifacts_qualify_as_future_evidence")
        is not False
        or any(
            supersession.get(name) is not False
            for name in (
                "candidate_changed",
                "future_boundary_changed",
                "evaluation_rule_changed",
                "future_outcomes_used_for_recovery",
            )
        )
    ):
        raise FutureProtocolV3Error("clock-correction lineage changed")
    source_record = value.get("source_snapshot") or {}
    if (
        source_record.get("package_id") != source.get("package_id")
        or source_record.get("manifest_raw_sha256") != MANIFEST_RAW_SHA256
        or source_record.get("manifest_canonical_sha256") != MANIFEST_CANONICAL_SHA256
    ):
        raise FutureProtocolV3Error("protocol v3 source binding changed")
    preflight_record = value.get("clock_corrected_source_preflight") or {}
    if (
        preflight_record.get("artifact_sha256") != preflight.get("artifact_sha256")
        or preflight_record.get("built_at_utc") != REGISTERED_PREFLIGHT_BUILT_AT_UTC
        or preflight_record.get("preflight_is_authority") is not False
    ):
        raise FutureProtocolV3Error("clock-corrected preflight binding changed")
    candidate = value.get("locked_candidate") or {}
    if (
        candidate.get("candidate_id") != SELECTED_CANDIDATE_ID
        or candidate.get("definition")
        != protocol_v1["locked_candidate"]["definition"]
        or candidate.get("original_selection_artifact_sha256")
        != protocol_v1["locked_candidate"]["selection_artifact_sha256"]
    ):
        raise FutureProtocolV3Error("protocol v3 candidate binding changed")
    future = value.get("future_holdout") or {}
    if (
        future.get("status") != "EMPTY_NOT_YET_ACQUIRED"
        or future.get("start_inclusive_source_time")
        != FUTURE_SEALED_START.isoformat()
        or future.get("one_time_opening") is not True
    ):
        raise FutureProtocolV3Error("future holdout boundary changed")
    eligibility = future.get("eligibility") or {}
    if (
        eligibility.get("pre_event_prediction_ledger_required") is not True
        or eligibility.get("candidate_and_both_comparators_must_be_captured")
        is not True
        or eligibility.get("retrospective_prediction_generation_qualifies") is not False
    ):
        raise FutureProtocolV3Error("prediction-ledger rule changed")
    if value.get("evaluation") != protocol_v1.get("evaluation"):
        raise FutureProtocolV3Error("evaluation rule changed")
    if value.get("prediction_ledger") != {
        "status": "NOT_YET_CREATED",
        "entries": 0,
        "registry_locator": None,
        "registry_raw_sha256": None,
        "pre_event_capture_implementation_present": False,
        "retrospective_backfill_permitted": False,
    }:
        raise FutureProtocolV3Error("empty prediction ledger state changed")
    opening = value.get("opening_authority") or {}
    if (
        opening.get("independent_protocol_review_present") is not False
        or opening.get("independent_opening_approval_present") is not False
        or opening.get("self_authorizing") is not False
    ):
        raise FutureProtocolV3Error("opening authority was fabricated")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise FutureProtocolV3Error("future protocol v3 exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise FutureProtocolV3Error("empty future protocol contains decision outputs")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise FutureProtocolV3Error("protocol v3 source inventory changed")
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != list(SOURCE_LOCKS):
        raise FutureProtocolV3Error("protocol v3 source order changed")
    for record in records:
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("raw_sha256")
        ):
            raise FutureProtocolV3Error(f"future protocol v3 source drifted: {locator}")
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
            raise FileExistsError(f"refusing to replace future protocol v3: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_future_protocol_lock_v3()
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "result_state": payload["result_state"],
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "FutureProtocolV3Error",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_future_protocol_lock_v3",
    "validate_future_protocol_lock_v3",
    "write_no_clobber",
]
