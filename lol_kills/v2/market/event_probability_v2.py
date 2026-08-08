"""File-bound phase-two event probability receipt with correct percentile semantics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping

from . import fast_event_uncertainty_v1 as fast_uncertainty
from . import phase_one_evaluation_v1 as evaluation
from . import phase_two_opening_v1 as opening
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/event_probability_v2.py"
RECEIPT_SCHEMA_VERSION = "scryglass:private-event-probability:v2"
RESULT_STATE = "PHASE_TWO_EVENT_PROBABILITY_CAPTURED_NON_AUTHORIZING"
RECEIPT_PREFIX = PurePosixPath(opening.EVENT_PROBABILITY_PREFIX.as_posix())
MARKET_TYPE = "match_winner"
AUTHORITY = {
    "event_probability_identity_authority": False,
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "File-bound phase-two event probability calculation only. The active "
    "opening permits outcome-free collection, not probability authority. "
    "Independent event registration, quote, settlement, market validation, "
    "recommendation, transaction, and betting authority remain required."
)


class EventProbabilityV2Error(RuntimeError):
    """The opening, target, uncertainty, interval, or receipt failed closed."""


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
        raise EventProbabilityV2Error("event probability is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EventProbabilityV2Error(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise EventProbabilityV2Error(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise EventProbabilityV2Error("event probability clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventProbabilityV2Error(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EventProbabilityV2Error(f"{field} must be finite")
    return result


def _probability(value: Any, field: str) -> float:
    result = _number(value, field)
    if not 0.0 < result < 1.0:
        raise EventProbabilityV2Error(f"{field} must be inside (0,1)")
    return result


def _interval(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise EventProbabilityV2Error(f"{field} must contain two bounds")
    lower = _number(value[0], f"{field}.lower")
    upper = _number(value[1], f"{field}.upper")
    if not 0.0 <= lower <= upper <= 1.0:
        raise EventProbabilityV2Error(f"{field} is invalid")
    return [lower, upper]


def _source_locks(root: Path) -> list[dict[str, Any]]:
    locators = [SOURCE_LOCATOR, opening.SOURCE_LOCATOR]
    for record in fast_uncertainty._source_locks(root):
        locator = str(record["locator"])
        if locator not in locators:
            locators.append(locator)
    return [evaluation._source_record(root, locator) for locator in locators]


def _uncertainty(
    root: Path,
    locator_value: str,
    environment: Mapping[str, str],
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value,
        fast_uncertainty.OUTPUT_PREFIX,
        "fast_uncertainty_locator",
    )
    raw = evaluation._read_regular(root, locator, "fast uncertainty")
    try:
        value = fast_uncertainty.validate_fast_event_uncertainty_v1(
            evaluation._strict_object(raw, "fast uncertainty"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise EventProbabilityV2Error("fast uncertainty is invalid") from exc
    return locator, raw, value


def build_event_probability_v2(
    *,
    fast_uncertainty_locator: str,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    try:
        active = opening.validate_active_phase_two_opening(
            root=root, environment=environment
        )
        protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    except Exception as exc:
        raise EventProbabilityV2Error(
            "active independently registered phase two is unavailable"
        ) from exc
    locator, raw, fast = _uncertainty(
        root, fast_uncertainty_locator, environment
    )
    candidate = fast["frozen_contract_candidate"]
    event = candidate["event"]
    _target_raw, target_prediction, _ratings, _metadata = (
        fast_uncertainty.frozen._target(
            root, event["target_prediction_locator"]
        )
    )
    _rating_locator, _rating_raw, rating = fast_uncertainty._rating_artifact(
        root,
        fast["decomposition"]["rating_bootstrap_locator"],
        environment,
    )
    point = candidate["point_calculation"]
    uncertainty = candidate["uncertainty"]
    captured = _clock_sample(clock)
    built = _timestamp(fast["built_at_utc"], "fast_uncertainty.built_at")
    event_start = _timestamp(rating["event"]["event_start_utc"], "event.start")
    if captured < built or captured >= event_start:
        raise EventProbabilityV2Error(
            "event probability was not captured after uncertainty and before event start"
        )
    probability = _probability(point["probability_blue"], "probability")
    interval = _interval(
        uncertainty["probability_interval_blue"], "probability_interval_blue"
    )
    opening_authority = active["authority"]
    marker = active["marker"]
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": captured.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": {
            "event_id": event["event_id"],
            "series_id": event["series_id"],
            "game_number": event["game_number"],
            "league": event["league"],
            "patch": event["patch"],
            "roster_change_stratum": event["roster_change_stratum"],
            "sparse_or_new_champion_map": target_prediction["draft_index"][
                "sparse_or_new_champion_map"
            ],
            "market_type": MARKET_TYPE,
            "selection": f"winner:{event['blue_organization_id']}",
            "opposing_selection": f"winner:{event['red_organization_id']}",
        },
        "opening_binding": {
            "authority_id": opening_authority["authority_id"],
            "authority_raw_sha256": active["authority_raw_sha256"],
            "marker_locator": opening.MARKER_LOCATOR.as_posix(),
            "marker_raw_sha256": active["marker_raw_sha256"],
            "opened_at_utc": marker["opened_at_utc"],
            "outcome_free_phase_two_collection_active": True,
        },
        "input_binding": {
            "fast_uncertainty_locator": locator,
            "fast_uncertainty_raw_sha256": _sha256_bytes(raw),
            "fast_uncertainty_artifact_sha256": fast["artifact_sha256"],
            "frozen_contract_candidate_artifact_sha256": candidate[
                "artifact_sha256"
            ],
            "target_prediction_locator": event["target_prediction_locator"],
            "target_prediction_artifact_sha256": event[
                "target_prediction_artifact_sha256"
            ],
            "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "generation_source_locator": SOURCE_LOCATOR,
            "generation_source_raw_sha256": evaluation._sha256_path(
                root / SOURCE_LOCATOR
            ),
        },
        "calculation": {
            "method": "bounded_logistic_recalibration",
            "raw_model_probability": point["raw_probability_blue"],
            "calibration_intercept": point["recalibration_intercept"],
            "calibration_slope": point["recalibration_slope"],
            "probability": probability,
            "opposing_probability": 1.0 - probability,
            "rating_only_comparator": dict(fast["evaluation_comparator"]),
        },
        "uncertainty": {
            "method": "series_cluster_bootstrap_full_prediction_pipeline",
            "confidence_level": 0.95,
            "resamples": candidate["bootstrap_contract"]["resamples"],
            "draws_sha256": uncertainty["draws_sha256"],
            "probability_interval": interval,
            "opposing_probability_interval": [
                1.0 - interval[1],
                1.0 - interval[0],
            ],
            "point_inside_percentile_interval": (
                interval[0] <= probability <= interval[1]
            ),
            "point_containment_required": False,
            "interval_is_epistemic": True,
            "interval_is_not_binary_outcome_coverage_guarantee": True,
        },
        "qualification": {
            "phase_two_opening_active": True,
            "phase_one_models_independently_passed": True,
            "recalibration_and_fast_uncertainty_independently_registered": True,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
            "market_price_used_as_model_input": False,
            "independently_registered": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    if protocol.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256:
        raise EventProbabilityV2Error("market protocol binding changed")
    payload["artifact_sha256"] = _canonical_sha256(payload)
    validate_event_probability_v2(
        payload, root=root, environment=environment
    )
    return payload


def validate_event_probability_v2(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise EventProbabilityV2Error("event probability must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "captured_at_utc",
        "clock_attestation",
        "event",
        "opening_binding",
        "input_binding",
        "calculation",
        "uncertainty",
        "qualification",
        "source_locks",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise EventProbabilityV2Error("event probability structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise EventProbabilityV2Error("event probability hash changed")
    if (
        value.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise EventProbabilityV2Error("event probability identity changed")
    captured = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise EventProbabilityV2Error("event probability clock changed")
    try:
        active = opening.validate_active_phase_two_opening(
            root=root, environment=environment
        )
    except Exception as exc:
        raise EventProbabilityV2Error("phase-two opening is inactive") from exc
    marker = active["marker"]
    if value.get("opening_binding") != {
        "authority_id": active["authority"]["authority_id"],
        "authority_raw_sha256": active["authority_raw_sha256"],
        "marker_locator": opening.MARKER_LOCATOR.as_posix(),
        "marker_raw_sha256": active["marker_raw_sha256"],
        "opened_at_utc": marker["opened_at_utc"],
        "outcome_free_phase_two_collection_active": True,
    }:
        raise EventProbabilityV2Error("opening binding changed")
    inputs = value.get("input_binding")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "fast_uncertainty_locator",
        "fast_uncertainty_raw_sha256",
        "fast_uncertainty_artifact_sha256",
        "frozen_contract_candidate_artifact_sha256",
        "target_prediction_locator",
        "target_prediction_artifact_sha256",
        "market_protocol_artifact_sha256",
        "generation_source_locator",
        "generation_source_raw_sha256",
    }:
        raise EventProbabilityV2Error("event probability input binding changed")
    locator, raw, fast = _uncertainty(
        root, str(inputs["fast_uncertainty_locator"]), environment
    )
    candidate = fast["frozen_contract_candidate"]
    candidate_event = candidate["event"]
    _target_raw, target_prediction, _ratings, _metadata = (
        fast_uncertainty.frozen._target(
            root, candidate_event["target_prediction_locator"]
        )
    )
    _rating_locator, _rating_raw, rating = fast_uncertainty._rating_artifact(
        root,
        fast["decomposition"]["rating_bootstrap_locator"],
        environment,
    )
    if (
        locator != inputs["fast_uncertainty_locator"]
        or _sha256_bytes(raw) != inputs["fast_uncertainty_raw_sha256"]
        or fast["artifact_sha256"]
        != inputs["fast_uncertainty_artifact_sha256"]
        or candidate["artifact_sha256"]
        != inputs["frozen_contract_candidate_artifact_sha256"]
        or candidate_event["target_prediction_locator"]
        != inputs["target_prediction_locator"]
        or candidate_event["target_prediction_artifact_sha256"]
        != inputs["target_prediction_artifact_sha256"]
        or inputs["market_protocol_artifact_sha256"]
        != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or inputs["generation_source_locator"] != SOURCE_LOCATOR
        or inputs["generation_source_raw_sha256"]
        != evaluation._sha256_path(root / SOURCE_LOCATOR)
    ):
        raise EventProbabilityV2Error("event probability file binding changed")
    event = value.get("event")
    expected_event = {
        "event_id": candidate_event["event_id"],
        "series_id": candidate_event["series_id"],
        "game_number": candidate_event["game_number"],
        "league": candidate_event["league"],
        "patch": candidate_event["patch"],
        "roster_change_stratum": candidate_event["roster_change_stratum"],
        "sparse_or_new_champion_map": target_prediction["draft_index"][
            "sparse_or_new_champion_map"
        ],
        "market_type": MARKET_TYPE,
        "selection": f"winner:{candidate_event['blue_organization_id']}",
        "opposing_selection": f"winner:{candidate_event['red_organization_id']}",
    }
    if event != expected_event or event["selection"] == event["opposing_selection"]:
        raise EventProbabilityV2Error("event probability event changed")
    point = candidate["point_calculation"]
    calculation = value.get("calculation")
    expected_calculation = {
        "method": "bounded_logistic_recalibration",
        "raw_model_probability": point["raw_probability_blue"],
        "calibration_intercept": point["recalibration_intercept"],
        "calibration_slope": point["recalibration_slope"],
        "probability": point["probability_blue"],
        "opposing_probability": 1.0 - point["probability_blue"],
        "rating_only_comparator": dict(fast["evaluation_comparator"]),
    }
    if calculation != expected_calculation:
        raise EventProbabilityV2Error("event probability calculation changed")
    probability = _probability(calculation["probability"], "probability")
    frozen_result = candidate["uncertainty"]
    interval = _interval(
        frozen_result["probability_interval_blue"], "probability_interval"
    )
    expected_uncertainty = {
        "method": "series_cluster_bootstrap_full_prediction_pipeline",
        "confidence_level": 0.95,
        "resamples": candidate["bootstrap_contract"]["resamples"],
        "draws_sha256": frozen_result["draws_sha256"],
        "probability_interval": interval,
        "opposing_probability_interval": [
            1.0 - interval[1],
            1.0 - interval[0],
        ],
        "point_inside_percentile_interval": interval[0]
        <= probability
        <= interval[1],
        "point_containment_required": False,
        "interval_is_epistemic": True,
        "interval_is_not_binary_outcome_coverage_guarantee": True,
    }
    if value.get("uncertainty") != expected_uncertainty:
        raise EventProbabilityV2Error("event probability uncertainty changed")
    if captured < _timestamp(fast["built_at_utc"], "fast.built_at"):
        raise EventProbabilityV2Error("event probability predates uncertainty")
    if captured >= _timestamp(rating["event"]["event_start_utc"], "event.start"):
        raise EventProbabilityV2Error("event probability was not pre-event")
    if value.get("qualification") != {
        "phase_two_opening_active": True,
        "phase_one_models_independently_passed": True,
        "recalibration_and_fast_uncertainty_independently_registered": True,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
        "market_price_used_as_model_input": False,
        "independently_registered": False,
    }:
        raise EventProbabilityV2Error("event probability qualification changed")
    if value.get("source_locks") != _source_locks(root):
        raise EventProbabilityV2Error("event probability source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise EventProbabilityV2Error("event probability exceeds authority")
    return {
        **value,
        "receipt_sha256": _canonical_sha256(value),
        "probability": probability,
        "probability_interval": interval,
    }


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise EventProbabilityV2Error(f"refusing to replace event probability: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise EventProbabilityV2Error(
                f"refusing to replace event probability: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "RECEIPT_PREFIX",
    "RECEIPT_SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "EventProbabilityV2Error",
    "build_event_probability_v2",
    "validate_event_probability_v2",
    "write_no_clobber",
]
