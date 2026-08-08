"""Live private event probability generated only under semantic market authority."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping

from . import event_probability_v2 as phase_two_probability
from . import fast_event_uncertainty_v1 as fast_uncertainty
from . import phase_one_evaluation_v1 as evaluation
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/production_event_probability_v1.py"
SCHEMA_VERSION = "scryglass:production-event-probability:v1"
RESULT_STATE = "PRIVATE_PRODUCTION_EVENT_PROBABILITY_CAPTURED"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/private_decision_support/match-winner/production-probabilities-v1"
)
MARKET_TYPE = "match_winner"
AUTHORITY = {
    "event_probability_receipt_identity": True,
    "self_authorized_probability_accuracy": False,
    "self_authorized_expected_value": False,
    "self_authorized_recommendation": False,
    "transaction_authority": False,
    "stake_authority": False,
}
CLAIM_CEILING = (
    "Exact private production probability receipt under an active independently "
    "pinned semantic market authority. A current exact quote and semantic decision "
    "validation remain required; no transaction or stake is authorized."
)


class ProductionEventProbabilityError(RuntimeError):
    """The authority, frozen uncertainty, target, timing, or receipt failed."""


def _semantic_authority(
    *, root: Path, environment: Mapping[str, str], as_of: datetime
) -> dict[str, Any]:
    try:
        from .semantic_market_authority_v1 import (
            load_active_semantic_market_authority_v1,
        )

        return load_active_semantic_market_authority_v1(
            root=root, environment=environment, as_of=as_of
        )
    except Exception as exc:
        raise ProductionEventProbabilityError(
            "active semantic market authority is unavailable"
        ) from exc


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionEventProbabilityError("receipt is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _clock(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise ProductionEventProbabilityError("probability clock must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    from . import semantic_market_authority_v1 as market_authority

    locators = [SOURCE_LOCATOR, market_authority.SOURCE_LOCATOR]
    for record in fast_uncertainty._source_locks(root):
        locator = str(record["locator"])
        if locator not in locators:
            locators.append(locator)
    return [evaluation._source_record(root, locator) for locator in locators]


def _components(
    *,
    fast_uncertainty_locator: str,
    root: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    try:
        locator, raw, fast = phase_two_probability._uncertainty(
            root, fast_uncertainty_locator, environment
        )
        candidate = fast["frozen_contract_candidate"]
        event = candidate["event"]
        _target_raw, target, _ratings, _metadata = fast_uncertainty.frozen._target(
            root, event["target_prediction_locator"]
        )
        _rating_locator, _rating_raw, rating = fast_uncertainty._rating_artifact(
            root,
            fast["decomposition"]["rating_bootstrap_locator"],
            environment,
        )
    except Exception as exc:
        raise ProductionEventProbabilityError("frozen uncertainty is invalid") from exc
    point = candidate["point_calculation"]
    probability = phase_two_probability._probability(
        point["probability_blue"], "probability"
    )
    interval = phase_two_probability._interval(
        candidate["uncertainty"]["probability_interval_blue"],
        "probability_interval",
    )
    return {
        "locator": locator,
        "raw": raw,
        "fast": fast,
        "candidate": candidate,
        "event": event,
        "target": target,
        "rating": rating,
        "probability": probability,
        "interval": interval,
    }


def _authority_binding(active: Mapping[str, Any]) -> dict[str, Any]:
    receipt = active["receipt"]
    return {
        "authority_id": receipt["authority_id"],
        "authority_raw_sha256": active["receipt_raw_sha256"],
        "issued_at_utc": receipt["issued_at_utc"],
        "valid_until_utc": receipt["valid_until_utc"],
        "phase_two_evaluation_registry_raw_sha256": active["bindings"]
        ["phase_two_evaluation"]["registry_raw_sha256"],
        "private_probability_generation_authorized": True,
    }


def _event(component: Mapping[str, Any]) -> dict[str, Any]:
    event = component["event"]
    return {
        "event_id": event["event_id"],
        "series_id": event["series_id"],
        "game_number": event["game_number"],
        "league": event["league"],
        "patch": event["patch"],
        "roster_change_stratum": event["roster_change_stratum"],
        "sparse_or_new_champion_map": component["target"]["draft_index"]
        ["sparse_or_new_champion_map"],
        "market_type": MARKET_TYPE,
        "selection": f"winner:{event['blue_organization_id']}",
        "opposing_selection": f"winner:{event['red_organization_id']}",
        "scheduled_event_start_utc": component["rating"]["event"][
            "event_start_utc"
        ],
    }


def _input_binding(component: Mapping[str, Any], root: Path) -> dict[str, Any]:
    return {
        "fast_uncertainty_locator": component["locator"],
        "fast_uncertainty_raw_sha256": _sha256(component["raw"]),
        "fast_uncertainty_artifact_sha256": component["fast"]["artifact_sha256"],
        "frozen_contract_candidate_artifact_sha256": component["candidate"]
        ["artifact_sha256"],
        "target_prediction_locator": component["event"]["target_prediction_locator"],
        "target_prediction_artifact_sha256": component["event"]
        ["target_prediction_artifact_sha256"],
        "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "generation_source_locator": SOURCE_LOCATOR,
        "generation_source_raw_sha256": evaluation._sha256_path(root / SOURCE_LOCATOR),
    }


def _calculation(component: Mapping[str, Any]) -> dict[str, Any]:
    point = component["candidate"]["point_calculation"]
    probability = component["probability"]
    return {
        "method": "bounded_logistic_recalibration",
        "raw_model_probability": point["raw_probability_blue"],
        "calibration_intercept": point["recalibration_intercept"],
        "calibration_slope": point["recalibration_slope"],
        "probability": probability,
        "opposing_probability": 1.0 - probability,
        "rating_only_comparator": dict(component["fast"]["evaluation_comparator"]),
    }


def _uncertainty(component: Mapping[str, Any]) -> dict[str, Any]:
    interval = component["interval"]
    probability = component["probability"]
    candidate = component["candidate"]
    return {
        "method": "series_cluster_bootstrap_full_prediction_pipeline",
        "confidence_level": 0.95,
        "resamples": candidate["bootstrap_contract"]["resamples"],
        "draws_sha256": candidate["uncertainty"]["draws_sha256"],
        "probability_interval": interval,
        "opposing_probability_interval": [1.0 - interval[1], 1.0 - interval[0]],
        "point_inside_percentile_interval": interval[0] <= probability <= interval[1],
        "point_containment_required": False,
        "interval_is_epistemic": True,
        "interval_is_not_binary_outcome_coverage_guarantee": True,
    }


def build_production_event_probability_v1(
    *, fast_uncertainty_locator: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    captured = _clock(clock)
    active = _semantic_authority(root=root, environment=environment, as_of=captured)
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    component = _components(
        fast_uncertainty_locator=fast_uncertainty_locator,
        root=root,
        environment=environment,
    )
    if protocol["artifact_sha256"] != REGISTERED_PROTOCOL_ARTIFACT_SHA256:
        raise ProductionEventProbabilityError("market protocol binding changed")
    if captured < phase_two_probability._timestamp(
        component["fast"]["built_at_utc"], "fast.built_at"
    ) or captured >= phase_two_probability._timestamp(
        component["rating"]["event"]["event_start_utc"], "event.start"
    ):
        raise ProductionEventProbabilityError("probability is not fresh pre-event output")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": captured.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": _event(component),
        "semantic_market_authority_binding": _authority_binding(active),
        "input_binding": _input_binding(component, root),
        "calculation": _calculation(component),
        "uncertainty": _uncertainty(component),
        "qualification": {
            "terminal_phase_two_market_gates_independently_passed": True,
            "semantic_market_authority_active": True,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
            "market_price_used_as_model_input": False,
            "transaction_or_stake_authorized": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    validate_production_event_probability_v1(
        payload, root=root, environment=environment
    )
    return payload


def validate_production_event_probability_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProductionEventProbabilityError("probability receipt must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "captured_at_utc", "clock_attestation",
        "event", "semantic_market_authority_binding", "input_binding",
        "calculation", "uncertainty", "qualification", "source_locks",
        "authority", "claim_ceiling", "artifact_sha256",
    }:
        raise ProductionEventProbabilityError("probability receipt structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise ProductionEventProbabilityError("probability receipt hash changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise ProductionEventProbabilityError("probability receipt identity changed")
    captured = phase_two_probability._timestamp(value.get("captured_at_utc"), "captured_at")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise ProductionEventProbabilityError("probability clock changed")
    active = _semantic_authority(root=root, environment=environment, as_of=captured)
    if value.get("semantic_market_authority_binding") != _authority_binding(active):
        raise ProductionEventProbabilityError("market authority binding changed")
    inputs = value.get("input_binding") or {}
    component = _components(
        fast_uncertainty_locator=inputs.get("fast_uncertainty_locator"),
        root=root,
        environment=environment,
    )
    if (
        inputs != _input_binding(component, root)
        or value.get("event") != _event(component)
        or value.get("calculation") != _calculation(component)
        or value.get("uncertainty") != _uncertainty(component)
    ):
        raise ProductionEventProbabilityError("probability calculation binding changed")
    if captured < phase_two_probability._timestamp(
        component["fast"]["built_at_utc"], "fast.built_at"
    ) or captured >= phase_two_probability._timestamp(
        component["rating"]["event"]["event_start_utc"], "event.start"
    ):
        raise ProductionEventProbabilityError("probability timing changed")
    if value.get("qualification") != {
        "terminal_phase_two_market_gates_independently_passed": True,
        "semantic_market_authority_active": True,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
        "market_price_used_as_model_input": False,
        "transaction_or_stake_authorized": False,
    }:
        raise ProductionEventProbabilityError("probability qualification changed")
    if value.get("source_locks") != _source_locks(root):
        raise ProductionEventProbabilityError("probability source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise ProductionEventProbabilityError("probability receipt exceeds authority")
    return {
        **value,
        "receipt_sha256": _canonical_sha256(value),
        "probability": component["probability"],
        "probability_interval": component["interval"],
    }


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProductionEventProbabilityError(f"refusing to replace probability: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProductionEventProbabilityError(
                f"refusing to replace probability: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return _sha256(raw)


__all__ = [
    "OUTPUT_PREFIX", "SCHEMA_VERSION", "SOURCE_LOCATOR",
    "ProductionEventProbabilityError", "build_production_event_probability_v1",
    "validate_production_event_probability_v1", "write_no_clobber",
]
