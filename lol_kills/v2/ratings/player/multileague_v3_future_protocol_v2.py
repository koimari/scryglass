"""Supersede the failed v1 source package before the future ratings boundary.

Protocol v2 changes only the source package and records the successful schema
and numerical preflight. The candidate, future boundary, support rule,
evaluation rule, and requirement for genuinely pre-event predictions remain
frozen. No future target is present or opened by this lock.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from .multileague_v3_future_protocol import (
    DOMESTIC_LEAGUES,
    FUTURE_SEALED_START,
    SELECTED_CANDIDATE_ID,
)
from .multileague_v3_preflight_v1_registry import (
    REGISTERED_PREFLIGHT_ARTIFACT_SHA256 as PREFLIGHT_V1_ARTIFACT_SHA256,
    REGISTERED_PREFLIGHT_LOCATOR as PREFLIGHT_V1_LOCATOR,
    REGISTERED_PREFLIGHT_RAW_SHA256 as PREFLIGHT_V1_RAW_SHA256,
    validate_registered_source_preflight_v1,
)
from .multileague_v3_preflight_v2_registry import (
    REGISTERED_PREFLIGHT_ARTIFACT_SHA256 as PREFLIGHT_V2_ARTIFACT_SHA256,
    REGISTERED_PREFLIGHT_LOCATOR as PREFLIGHT_V2_LOCATOR,
    REGISTERED_PREFLIGHT_RAW_SHA256 as PREFLIGHT_V2_RAW_SHA256,
    validate_registered_source_preflight_v2,
)
from .multileague_v3_registry import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as PROTOCOL_V1_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as PROTOCOL_V1_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256 as PROTOCOL_V1_RAW_SHA256,
    validate_registered_future_protocol,
)
from .multileague_v3_source_registry_v2 import (
    MANIFEST_CANONICAL_SHA256,
    MANIFEST_LOCATOR,
    MANIFEST_RAW_SHA256,
    PACKAGE_ID,
    validate_registered_source_snapshot_v2,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-future-protocol-lock:v2"
RESULT_STATE = "SUPERSEDING_FUTURE_HOLDOUT_PROTOCOL_LOCKED_EMPTY"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol_v2.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/future-protocol-lock-v2.json"
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
    "lol_kills/v2/ratings/player/multileague_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v1.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v1_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v2_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_source_registry_v2.py",
    "lol_kills/v2/ratings/player/multileague_source_snapshot.py",
    "lol_kills/v2/ratings/player/multileague_development.py",
    "lol_kills/v2/ratings/player/multileague_runner.py",
    "lol_kills/v2/ratings/player/multileague_v2_runner.py",
    PROTOCOL_V1_LOCATOR.as_posix(),
    PREFLIGHT_V1_LOCATOR.as_posix(),
    PREFLIGHT_V2_LOCATOR.as_posix(),
    MANIFEST_LOCATOR.as_posix(),
)


class FutureProtocolV2Error(RuntimeError):
    """The superseding future protocol is malformed, contaminated, or unbound."""


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
        raise FutureProtocolV2Error("future protocol value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise FutureProtocolV2Error(f"bound source is unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256(path),
    }


def _parse_utc(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise FutureProtocolV2Error(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        raise FutureProtocolV2Error(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _parse_locked_at(value: str) -> datetime:
    parsed = _parse_utc(value, "locked_at")
    if parsed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise FutureProtocolV2Error(
            "superseding protocol must be locked before the future boundary"
        )
    return parsed


def build_future_protocol_lock_v2(
    *,
    locked_at: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    lock_time = _parse_locked_at(locked_at)
    protocol_v1 = validate_registered_future_protocol(root=root)
    failure_v1 = validate_registered_source_preflight_v1(root=root)
    preflight_v2 = validate_registered_source_preflight_v2(root=root)
    source = validate_registered_source_snapshot_v2(root=root)
    preflight_time = _parse_utc(preflight_v2["built_at_utc"], "preflight built_at")
    if lock_time <= preflight_time:
        raise FutureProtocolV2Error(
            "superseding protocol must be locked after the corrected preflight"
        )
    latest_source = datetime.fromisoformat(
        preflight_v2["source_snapshot"]["latest_observed_source_time"]
    )
    if latest_source >= FUTURE_SEALED_START:
        raise FutureProtocolV2Error("superseding source overlaps the future holdout")
    candidate = protocol_v1["locked_candidate"]
    if (
        candidate.get("candidate_id") != SELECTED_CANDIDATE_ID
        or preflight_v2["locked_candidate"]["definition"]
        != candidate.get("definition")
    ):
        raise FutureProtocolV2Error("frozen candidate changed during remediation")

    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": lock_time.isoformat(),
        "supersession": {
            "protocol_v1_locator": PROTOCOL_V1_LOCATOR.as_posix(),
            "protocol_v1_raw_sha256": PROTOCOL_V1_RAW_SHA256,
            "protocol_v1_artifact_sha256": PROTOCOL_V1_ARTIFACT_SHA256,
            "protocol_v1_operational_status": (
                "SUPERSEDED_AFTER_FAILED_SOURCE_SCHEMA_PREFLIGHT"
            ),
            "source_failure_v1_locator": PREFLIGHT_V1_LOCATOR.as_posix(),
            "source_failure_v1_raw_sha256": PREFLIGHT_V1_RAW_SHA256,
            "source_failure_v1_artifact_sha256": PREFLIGHT_V1_ARTIFACT_SHA256,
            "source_failure_v1_result_state": failure_v1["result_state"],
            "candidate_changed": False,
            "future_boundary_changed": False,
            "evaluation_rule_changed": False,
            "future_outcomes_used_for_remediation": False,
        },
        "adaptation_disclosure": {
            "all_outcomes_in_the_corrected_snapshot_are_adaptive": True,
            "corrected_source_preflight_is_not_independent_validation": True,
            "candidate_selection_remains_adaptive": True,
            "v1_failure_was_source_schema_only": True,
            "remediation_was_locked_before_future_boundary": True,
            "no_future_target_was_present_or_opened": True,
        },
        "source_snapshot": {
            "package_id": PACKAGE_ID,
            "manifest_locator": MANIFEST_LOCATOR.as_posix(),
            "manifest_raw_sha256": MANIFEST_RAW_SHA256,
            "manifest_canonical_sha256": MANIFEST_CANONICAL_SHA256,
            "maps": source["files"]["maps"],
            "players": source["files"]["players"],
            "latest_observed_source_time": preflight_v2["source_snapshot"][
                "latest_observed_source_time"
            ],
            "playoffs_dtype": preflight_v2["source_snapshot"]["playoffs_dtype"],
        },
        "corrected_source_preflight": {
            "locator": PREFLIGHT_V2_LOCATOR.as_posix(),
            "raw_sha256": PREFLIGHT_V2_RAW_SHA256,
            "artifact_sha256": PREFLIGHT_V2_ARTIFACT_SHA256,
            "result_state": preflight_v2["result_state"],
            "accepted_maps": preflight_v2["adapter_preflight"]["coverage"][
                "accepted_maps"
            ],
            "development_series": preflight_v2["adapter_preflight"]["coverage"][
                "development_series"
            ],
            "latent_dimension": preflight_v2["numerical_preflight"][
                "latent_dimension"
            ],
            "posterior_state_sha256": preflight_v2["numerical_preflight"][
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
            "selection_status": (
                "adaptive_development_choice_unchanged_before_future_holdout"
            ),
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
                "pre_event_prediction_ledger_required": True,
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
            "This superseding lock repairs source replay before the future boundary. "
            "It does not validate or authorize player ratings, team ratings, "
            "probabilities, odds, expected value, recommendations, or wagers."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_future_protocol_lock_v2(payload, root=root)


def validate_future_protocol_lock_v2(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise FutureProtocolV2Error("future protocol v2 must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise FutureProtocolV2Error("future protocol v2 identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise FutureProtocolV2Error("future protocol v2 canonical hash mismatch")
    lock_time = _parse_locked_at(str(value.get("locked_at_utc")))
    protocol_v1 = validate_registered_future_protocol(root=root)
    failure_v1 = validate_registered_source_preflight_v1(root=root)
    preflight_v2 = validate_registered_source_preflight_v2(root=root)
    source = validate_registered_source_snapshot_v2(root=root)
    if lock_time <= _parse_utc(preflight_v2["built_at_utc"], "preflight built_at"):
        raise FutureProtocolV2Error("protocol lock does not follow corrected preflight")

    supersession = value.get("supersession") or {}
    required_false = (
        "candidate_changed",
        "future_boundary_changed",
        "evaluation_rule_changed",
        "future_outcomes_used_for_remediation",
    )
    if (
        supersession.get("protocol_v1_artifact_sha256")
        != protocol_v1.get("artifact_sha256")
        or supersession.get("source_failure_v1_artifact_sha256")
        != failure_v1.get("artifact_sha256")
        or supersession.get("protocol_v1_operational_status")
        != "SUPERSEDED_AFTER_FAILED_SOURCE_SCHEMA_PREFLIGHT"
        or any(supersession.get(name) is not False for name in required_false)
    ):
        raise FutureProtocolV2Error("protocol supersession lineage changed")
    adaptation = value.get("adaptation_disclosure") or {}
    if set(adaptation) != {
        "all_outcomes_in_the_corrected_snapshot_are_adaptive",
        "corrected_source_preflight_is_not_independent_validation",
        "candidate_selection_remains_adaptive",
        "v1_failure_was_source_schema_only",
        "remediation_was_locked_before_future_boundary",
        "no_future_target_was_present_or_opened",
    } or any(item is not True for item in adaptation.values()):
        raise FutureProtocolV2Error("protocol adaptation disclosure changed")
    source_record = value.get("source_snapshot") or {}
    if (
        source_record.get("package_id") != source.get("package_id")
        or source_record.get("manifest_raw_sha256") != MANIFEST_RAW_SHA256
        or source_record.get("manifest_canonical_sha256") != MANIFEST_CANONICAL_SHA256
    ):
        raise FutureProtocolV2Error("protocol v2 source binding changed")
    preflight_record = value.get("corrected_source_preflight") or {}
    if (
        preflight_record.get("artifact_sha256")
        != preflight_v2.get("artifact_sha256")
        or preflight_record.get("preflight_is_authority") is not False
        or preflight_record.get("posterior_state_sha256")
        != preflight_v2["numerical_preflight"]["posterior_state_sha256"]
    ):
        raise FutureProtocolV2Error("corrected preflight binding changed")
    candidate = value.get("locked_candidate") or {}
    if (
        candidate.get("candidate_id") != SELECTED_CANDIDATE_ID
        or candidate.get("definition")
        != protocol_v1["locked_candidate"]["definition"]
        or candidate.get("original_selection_artifact_sha256")
        != protocol_v1["locked_candidate"]["selection_artifact_sha256"]
    ):
        raise FutureProtocolV2Error("protocol v2 candidate binding changed")
    future = value.get("future_holdout") or {}
    if (
        future.get("status") != "EMPTY_NOT_YET_ACQUIRED"
        or future.get("start_inclusive_source_time")
        != FUTURE_SEALED_START.isoformat()
        or future.get("one_time_opening") is not True
    ):
        raise FutureProtocolV2Error("future holdout boundary changed")
    eligibility = future.get("eligibility") or {}
    if (
        eligibility.get("pre_event_prediction_ledger_required") is not True
        or eligibility.get("prediction_must_be_generated_without_event_outcome_access")
        is not True
        or eligibility.get("retrospective_prediction_generation_qualifies") is not False
    ):
        raise FutureProtocolV2Error("prediction-ledger boundary changed")
    ledger = value.get("prediction_ledger") or {}
    if ledger != {
        "status": "NOT_YET_CREATED",
        "entries": 0,
        "registry_locator": None,
        "registry_raw_sha256": None,
        "pre_event_capture_implementation_present": False,
        "retrospective_backfill_permitted": False,
    }:
        raise FutureProtocolV2Error("empty prediction ledger state changed")
    if value.get("evaluation") != protocol_v1.get("evaluation"):
        raise FutureProtocolV2Error("evaluation rule changed during supersession")
    opening = value.get("opening_authority") or {}
    if (
        opening.get("independent_protocol_review_present") is not False
        or opening.get("independent_opening_approval_present") is not False
        or opening.get("self_authorizing") is not False
    ):
        raise FutureProtocolV2Error("opening authority was fabricated")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise FutureProtocolV2Error("future protocol v2 exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise FutureProtocolV2Error("empty future protocol contains decision outputs")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise FutureProtocolV2Error("future protocol v2 source inventory changed")
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != list(SOURCE_LOCKS):
        raise FutureProtocolV2Error("future protocol v2 source order changed")
    for record in records:
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != record.get("raw_sha256")
        ):
            raise FutureProtocolV2Error(f"future protocol v2 source drifted: {locator}")
    return value


def write_protocol_lock_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
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
                f"refusing to replace future protocol v2: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_future_protocol_lock_v2(locked_at=args.locked_at)
    raw_sha256 = write_protocol_lock_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "future_holdout_start": FUTURE_SEALED_START.isoformat(),
                "status": payload["result_state"],
                "prediction_ledger_status": payload["prediction_ledger"]["status"],
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_KEYS",
    "DEFAULT_OUTPUT",
    "FutureProtocolV2Error",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "build_future_protocol_lock_v2",
    "validate_future_protocol_lock_v2",
    "write_protocol_lock_no_clobber",
]
