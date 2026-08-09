"""Event-specific probability receipts and independently pinned registries.

The receipt is a reproducible calculation record, not authority.  A market
decision may consume it only when its exact digest appears in a separately
pinned registry and every model, protocol, calibration, uncertainty, and
generation-code hash matches the independently approved market authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Callable, Mapping, Sequence


RECEIPT_SCHEMA_VERSION = "scryglass.private-event-probability.v1"
REGISTRY_SCHEMA_VERSION = "scryglass.private-event-probability-registry.v1"
RESULT_STATE = "EVENT_PROBABILITY_CALCULATION_CAPTURED_NON_AUTHORIZING"
REGISTRY_SCOPE = "private_event_probability_identity_only"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CLAIM_CEILING = (
    "Reproducible event-probability calculation only. Independent registry, "
    "model, calibration, uncertainty, quote, settlement, and market authority "
    "are still required; this receipt cannot authorize a wager."
)
AUTHORITY = {
    "probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
QUALIFICATION = {
    "event_outcome_present": False,
    "event_outcome_accessed": False,
    "market_price_used_as_model_input": False,
    "source_prediction_registration_binding_present": True,
    "calibration_binding_present": True,
    "uncertainty_binding_present": True,
    "system_clock_sampled_inside_builder": True,
    "independently_registered": False,
}
ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/event-probabilities"
)
DEFAULT_REGISTRY = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/event-probability-registry.json"
)


class EventProbabilityError(ValueError):
    """An event probability receipt or registry failed closed."""


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EventProbabilityError("probability value is not canonical JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise EventProbabilityError(f"{label} keys do not match the frozen contract")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventProbabilityError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise EventProbabilityError(f"{label} must be a lowercase SHA-256")
    return value


def _time(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise EventProbabilityError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise EventProbabilityError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise EventProbabilityError(
            "probability clock must return a timezone-aware datetime"
        )
    return observed.astimezone(timezone.utc)


def _probability(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise EventProbabilityError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or not 0.0 < number < 1.0:
        raise EventProbabilityError(f"{label} must be finite and inside (0,1)")
    return number


def _interval(value: Any, point: float, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise EventProbabilityError(f"{label} must contain two bounds")
    if any(type(item) not in (int, float) for item in value):
        raise EventProbabilityError(f"{label} bounds must be numeric")
    lower = float(value[0])
    upper = float(value[1])
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or not 0.0 <= lower <= point <= upper <= 1.0
    ):
        raise EventProbabilityError(f"{label} is invalid")
    return [lower, upper]


def _calibrate(raw_probability: float, intercept: float, slope: float) -> tuple[float, float]:
    clipped = min(max(raw_probability, 1e-6), 0.999999)
    raw_logit = math.log(clipped / (1.0 - clipped))
    calibrated_logit = intercept + slope * raw_logit
    if calibrated_logit >= 40.0:
        return clipped, 1.0 - 1e-15
    if calibrated_logit <= -40.0:
        return clipped, 1e-15
    return clipped, 1.0 / (1.0 + math.exp(-calibrated_logit))


def build_event_probability_receipt(
    *,
    event_id: str,
    league: str,
    market_type: str,
    selection: str,
    opposing_selection: str,
    model_artifact_sha256: str,
    market_protocol_artifact_sha256: str,
    calibration_artifact_sha256: str,
    uncertainty_artifact_sha256: str,
    source_prediction_receipt_sha256: str,
    source_prediction_registry_sha256: str,
    generation_code_sha256: str,
    raw_model_probability: float,
    calibration_intercept: float,
    calibration_slope: float,
    probability_interval: Sequence[float],
    uncertainty_draws_sha256: str,
    uncertainty_resamples: int,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    captured_at = _clock_sample(clock)
    event = _nonempty(event_id, "event_id")
    selection_value = _nonempty(selection, "selection")
    opposing_value = _nonempty(opposing_selection, "opposing_selection")
    if selection_value == opposing_value:
        raise EventProbabilityError("event selections must differ")
    raw_probability = _probability(raw_model_probability, "raw_model_probability")
    if type(calibration_intercept) not in (int, float) or type(
        calibration_slope
    ) not in (int, float):
        raise EventProbabilityError("calibration parameters must be numeric")
    intercept = float(calibration_intercept)
    slope = float(calibration_slope)
    if (
        not math.isfinite(intercept)
        or not -2.0 <= intercept <= 2.0
        or not math.isfinite(slope)
        or not 0.25 <= slope <= 4.0
    ):
        raise EventProbabilityError("calibration parameters exceed the frozen bounds")
    clipped, point = _calibrate(raw_probability, intercept, slope)
    interval = _interval(probability_interval, point, "probability_interval")
    if (
        isinstance(uncertainty_resamples, bool)
        or not isinstance(uncertainty_resamples, int)
        or uncertainty_resamples < 2000
    ):
        raise EventProbabilityError("uncertainty_resamples must be at least 2000")
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": captured_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": captured_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": {
            "event_id": event,
            "league": _nonempty(league, "league"),
            "market_type": _nonempty(market_type, "market_type"),
            "selection": selection_value,
            "opposing_selection": opposing_value,
        },
        "bindings": {
            "model_artifact_sha256": _sha(
                model_artifact_sha256, "model_artifact_sha256"
            ),
            "market_protocol_artifact_sha256": _sha(
                market_protocol_artifact_sha256,
                "market_protocol_artifact_sha256",
            ),
            "calibration_artifact_sha256": _sha(
                calibration_artifact_sha256, "calibration_artifact_sha256"
            ),
            "uncertainty_artifact_sha256": _sha(
                uncertainty_artifact_sha256, "uncertainty_artifact_sha256"
            ),
            "source_prediction_receipt_sha256": _sha(
                source_prediction_receipt_sha256,
                "source_prediction_receipt_sha256",
            ),
            "source_prediction_registry_sha256": _sha(
                source_prediction_registry_sha256,
                "source_prediction_registry_sha256",
            ),
            "generation_code_sha256": _sha(
                generation_code_sha256, "generation_code_sha256"
            ),
        },
        "calculation": {
            "method": "bounded_logistic_recalibration",
            "raw_model_probability": raw_probability,
            "clipped_raw_probability": clipped,
            "raw_probability_clip": [1e-6, 0.999999],
            "calibration_intercept": intercept,
            "calibration_slope": slope,
            "formula": "sigmoid(intercept+slope*logit(clipped_raw_probability))",
            "probability": point,
            "opposing_probability": 1.0 - point,
        },
        "uncertainty": {
            "method": "series_cluster_bootstrap_full_prediction_pipeline",
            "confidence_level": 0.95,
            "resamples": uncertainty_resamples,
            "draws_sha256": _sha(
                uncertainty_draws_sha256, "uncertainty_draws_sha256"
            ),
            "probability_interval": interval,
            "opposing_probability_interval": [1.0 - interval[1], 1.0 - interval[0]],
            "interval_is_epistemic": True,
        },
        "qualification": dict(QUALIFICATION),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = sha256_json(payload)
    validate_event_probability_receipt(payload)
    return payload


def validate_event_probability_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_receipt_sha256: str | None = None,
    expected_event_id: str | None = None,
    expected_league: str | None = None,
    expected_market_type: str | None = None,
    expected_selection: str | None = None,
    expected_opposing_selection: str | None = None,
    expected_model_artifact_sha256: str | None = None,
    expected_market_protocol_artifact_sha256: str | None = None,
    expected_calibration_artifact_sha256: str | None = None,
    expected_uncertainty_artifact_sha256: str | None = None,
    expected_generation_code_sha256: str | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise EventProbabilityError("event probability receipt must be an object")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "result_state",
            "captured_at_utc",
            "clock_attestation",
            "event",
            "bindings",
            "calculation",
            "uncertainty",
            "qualification",
            "authority",
            "claim_ceiling",
            "artifact_sha256",
        },
        "event probability receipt",
    )
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("result_state") != RESULT_STATE
    ):
        raise EventProbabilityError("event probability receipt identity changed")
    unsigned = dict(receipt)
    artifact_sha256 = unsigned.pop("artifact_sha256", None)
    if artifact_sha256 != sha256_json(unsigned):
        raise EventProbabilityError("event probability artifact hash changed")
    actual_sha256 = sha256_json(receipt)
    if expected_receipt_sha256 is not None:
        if actual_sha256 != _sha(expected_receipt_sha256, "expected_receipt_sha256"):
            raise EventProbabilityError("event probability receipt digest mismatch")

    captured_at = _time(receipt.get("captured_at_utc"), "captured_at_utc")
    if receipt.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise EventProbabilityError("event probability clock attestation changed")
    if as_of is not None:
        if as_of.tzinfo is None:
            raise EventProbabilityError("as_of must include a timezone")
        if captured_at > as_of.astimezone(timezone.utc):
            raise EventProbabilityError("event probability receipt is from the future")

    event = receipt.get("event")
    if not isinstance(event, Mapping):
        raise EventProbabilityError("event probability event binding is malformed")
    _exact_keys(
        event,
        {"event_id", "league", "market_type", "selection", "opposing_selection"},
        "event probability event",
    )
    for field in ("event_id", "league", "market_type", "selection", "opposing_selection"):
        _nonempty(event.get(field), f"event.{field}")
    if event["selection"] == event["opposing_selection"]:
        raise EventProbabilityError("event probability selections are identical")
    expected_events = {
        "event_id": expected_event_id,
        "league": expected_league,
        "market_type": expected_market_type,
        "selection": expected_selection,
        "opposing_selection": expected_opposing_selection,
    }
    for field, expected in expected_events.items():
        if expected is not None and event[field] != expected:
            raise EventProbabilityError(f"event probability {field} binding mismatch")

    bindings = receipt.get("bindings")
    if not isinstance(bindings, Mapping):
        raise EventProbabilityError("event probability bindings are malformed")
    binding_fields = {
        "model_artifact_sha256",
        "market_protocol_artifact_sha256",
        "calibration_artifact_sha256",
        "uncertainty_artifact_sha256",
        "source_prediction_receipt_sha256",
        "source_prediction_registry_sha256",
        "generation_code_sha256",
    }
    _exact_keys(bindings, binding_fields, "event probability bindings")
    for field in binding_fields:
        _sha(bindings.get(field), f"bindings.{field}")
    expected_bindings = {
        "model_artifact_sha256": expected_model_artifact_sha256,
        "market_protocol_artifact_sha256": expected_market_protocol_artifact_sha256,
        "calibration_artifact_sha256": expected_calibration_artifact_sha256,
        "uncertainty_artifact_sha256": expected_uncertainty_artifact_sha256,
        "generation_code_sha256": expected_generation_code_sha256,
    }
    for field, expected in expected_bindings.items():
        if expected is not None and bindings[field] != _sha(expected, f"expected_{field}"):
            raise EventProbabilityError(f"event probability {field} binding mismatch")

    calculation = receipt.get("calculation")
    if not isinstance(calculation, Mapping):
        raise EventProbabilityError("event probability calculation is malformed")
    _exact_keys(
        calculation,
        {
            "method",
            "raw_model_probability",
            "clipped_raw_probability",
            "raw_probability_clip",
            "calibration_intercept",
            "calibration_slope",
            "formula",
            "probability",
            "opposing_probability",
        },
        "event probability calculation",
    )
    raw = _probability(calculation.get("raw_model_probability"), "raw_model_probability")
    raw_intercept = calculation.get("calibration_intercept")
    raw_slope = calculation.get("calibration_slope")
    if type(raw_intercept) not in (int, float) or type(raw_slope) not in (
        int,
        float,
    ):
        raise EventProbabilityError("event probability calibration is not numeric")
    intercept = float(raw_intercept)
    slope = float(raw_slope)
    if (
        not math.isfinite(intercept)
        or not -2.0 <= intercept <= 2.0
        or not math.isfinite(slope)
        or not 0.25 <= slope <= 4.0
    ):
        raise EventProbabilityError("event probability calibration bounds changed")
    clipped, computed = _calibrate(raw, intercept, slope)
    declared_clipped = _probability(
        calculation.get("clipped_raw_probability"), "clipped_raw_probability"
    )
    declared_probability = _probability(
        calculation.get("probability"), "probability"
    )
    declared_opposing_probability = _probability(
        calculation.get("opposing_probability"), "opposing_probability"
    )
    if (
        calculation.get("method") != "bounded_logistic_recalibration"
        or calculation.get("formula")
        != "sigmoid(intercept+slope*logit(clipped_raw_probability))"
        or calculation.get("raw_probability_clip") != [1e-6, 0.999999]
        or not math.isclose(
            declared_clipped,
            clipped,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            declared_probability,
            computed,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or not math.isclose(
            declared_opposing_probability,
            1.0 - computed,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    ):
        raise EventProbabilityError("event probability calculation does not replay")

    uncertainty = receipt.get("uncertainty")
    if not isinstance(uncertainty, Mapping):
        raise EventProbabilityError("event probability uncertainty is malformed")
    _exact_keys(
        uncertainty,
        {
            "method",
            "confidence_level",
            "resamples",
            "draws_sha256",
            "probability_interval",
            "opposing_probability_interval",
            "interval_is_epistemic",
        },
        "event probability uncertainty",
    )
    resamples = uncertainty.get("resamples")
    if (
        uncertainty.get("method")
        != "series_cluster_bootstrap_full_prediction_pipeline"
        or uncertainty.get("confidence_level") != 0.95
        or isinstance(resamples, bool)
        or not isinstance(resamples, int)
        or resamples < 2000
        or uncertainty.get("interval_is_epistemic") is not True
    ):
        raise EventProbabilityError("event probability uncertainty contract changed")
    _sha(uncertainty.get("draws_sha256"), "uncertainty.draws_sha256")
    interval = _interval(
        uncertainty.get("probability_interval"), computed, "probability_interval"
    )
    opposing_interval = _interval(
        uncertainty.get("opposing_probability_interval"),
        1.0 - computed,
        "opposing_probability_interval",
    )
    expected_opposing = [1.0 - interval[1], 1.0 - interval[0]]
    if any(
        not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-15)
        for actual, expected in zip(opposing_interval, expected_opposing)
    ):
        raise EventProbabilityError("opposing probability interval does not complement")
    if receipt.get("qualification") != QUALIFICATION:
        raise EventProbabilityError("event probability qualification changed")
    if receipt.get("authority") != AUTHORITY:
        raise EventProbabilityError("event probability receipt exceeds authority")
    if receipt.get("claim_ceiling") != CLAIM_CEILING:
        raise EventProbabilityError("event probability claim ceiling changed")
    return {
        **dict(receipt),
        "receipt_sha256": actual_sha256,
        "probability": computed,
        "probability_interval": interval,
    }


def _validate_receipt_locator(locator: Any) -> PurePosixPath:
    path = PurePosixPath(_nonempty(locator, "receipt_locator"))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(RECEIPT_PREFIX.parts)]) != RECEIPT_PREFIX.parts
        or path.suffix != ".json"
    ):
        raise EventProbabilityError(
            "event probability receipt locator is outside the private root"
        )
    return path


def build_event_probability_registry(
    *,
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    registry_id: str,
    independent_reviewer_id: str,
    issued_at: str,
    model_artifact_sha256: str,
    market_protocol_artifact_sha256: str,
    calibration_artifact_sha256: str,
    uncertainty_artifact_sha256: str,
    generation_code_sha256: str,
) -> dict[str, Any]:
    issued = _time(issued_at, "issued_at")
    bindings = {
        "model_artifact_sha256": _sha(
            model_artifact_sha256, "model_artifact_sha256"
        ),
        "market_protocol_artifact_sha256": _sha(
            market_protocol_artifact_sha256, "market_protocol_artifact_sha256"
        ),
        "calibration_artifact_sha256": _sha(
            calibration_artifact_sha256, "calibration_artifact_sha256"
        ),
        "uncertainty_artifact_sha256": _sha(
            uncertainty_artifact_sha256, "uncertainty_artifact_sha256"
        ),
        "generation_code_sha256": _sha(
            generation_code_sha256, "generation_code_sha256"
        ),
    }
    entries: list[dict[str, Any]] = []
    for locator, raw in receipts:
        receipt_locator = _validate_receipt_locator(locator).as_posix()
        checked = validate_event_probability_receipt(
            raw,
            expected_model_artifact_sha256=bindings["model_artifact_sha256"],
            expected_market_protocol_artifact_sha256=bindings[
                "market_protocol_artifact_sha256"
            ],
            expected_calibration_artifact_sha256=bindings[
                "calibration_artifact_sha256"
            ],
            expected_uncertainty_artifact_sha256=bindings[
                "uncertainty_artifact_sha256"
            ],
            expected_generation_code_sha256=bindings["generation_code_sha256"],
        )
        if issued < _time(checked["captured_at_utc"], "captured_at_utc"):
            raise EventProbabilityError("probability registry predates a receipt")
        event = checked["event"]
        entries.append(
            {
                "event_id": event["event_id"],
                "league": event["league"],
                "market_type": event["market_type"],
                "selection": event["selection"],
                "opposing_selection": event["opposing_selection"],
                "captured_at_utc": checked["captured_at_utc"],
                "receipt_locator": receipt_locator,
                "receipt_sha256": checked["receipt_sha256"],
                "source_prediction_receipt_sha256": checked["bindings"][
                    "source_prediction_receipt_sha256"
                ],
                "source_prediction_registry_sha256": checked["bindings"][
                    "source_prediction_registry_sha256"
                ],
            }
        )
    entries.sort(
        key=lambda item: (
            item["event_id"],
            item["market_type"],
            item["selection"],
            item["opposing_selection"],
        )
    )
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "approved",
        "scope": REGISTRY_SCOPE,
        "public_or_transactional_use": False,
        "registry_id": _nonempty(registry_id, "registry_id"),
        "independent_reviewer_id": _nonempty(
            independent_reviewer_id, "independent_reviewer_id"
        ),
        "issued_at": issued.isoformat(),
        "bindings": bindings,
        "entries": entries,
        "authority": dict(AUTHORITY),
    }
    validate_event_probability_registry(
        registry,
        expected_registry_sha256=sha256_json(registry),
        **{f"expected_{key}": value for key, value in bindings.items()},
    )
    return registry


def validate_event_probability_registry(
    registry: Mapping[str, Any],
    *,
    expected_registry_sha256: str | None,
    expected_model_artifact_sha256: str | None = None,
    expected_market_protocol_artifact_sha256: str | None = None,
    expected_calibration_artifact_sha256: str | None = None,
    expected_uncertainty_artifact_sha256: str | None = None,
    expected_generation_code_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_registry_sha256 is None:
        raise EventProbabilityError("event probability registry is not registered")
    if not isinstance(registry, Mapping):
        raise EventProbabilityError("event probability registry must be an object")
    if sha256_json(registry) != _sha(
        expected_registry_sha256, "expected_registry_sha256"
    ):
        raise EventProbabilityError("event probability registry digest mismatch")
    _exact_keys(
        registry,
        {
            "schema_version",
            "status",
            "scope",
            "public_or_transactional_use",
            "registry_id",
            "independent_reviewer_id",
            "issued_at",
            "bindings",
            "entries",
            "authority",
        },
        "event probability registry",
    )
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("status") != "approved"
        or registry.get("scope") != REGISTRY_SCOPE
        or registry.get("public_or_transactional_use") is not False
    ):
        raise EventProbabilityError("event probability registry is not approved")
    _nonempty(registry.get("registry_id"), "registry_id")
    _nonempty(registry.get("independent_reviewer_id"), "independent_reviewer_id")
    _time(registry.get("issued_at"), "issued_at")
    if registry.get("authority") != AUTHORITY:
        raise EventProbabilityError("event probability registry exceeds authority")
    bindings = registry.get("bindings")
    expected_bindings = {
        "model_artifact_sha256": expected_model_artifact_sha256,
        "market_protocol_artifact_sha256": expected_market_protocol_artifact_sha256,
        "calibration_artifact_sha256": expected_calibration_artifact_sha256,
        "uncertainty_artifact_sha256": expected_uncertainty_artifact_sha256,
        "generation_code_sha256": expected_generation_code_sha256,
    }
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_bindings):
        raise EventProbabilityError("event probability registry bindings changed")
    for field, expected in expected_bindings.items():
        actual = _sha(bindings.get(field), f"bindings.{field}")
        if expected is not None and actual != _sha(expected, f"expected_{field}"):
            raise EventProbabilityError(f"event probability registry {field} mismatch")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise EventProbabilityError("event probability registry entries are empty")
    expected_entry_keys = {
        "event_id",
        "league",
        "market_type",
        "selection",
        "opposing_selection",
        "captured_at_utc",
        "receipt_locator",
        "receipt_sha256",
        "source_prediction_receipt_sha256",
        "source_prediction_registry_sha256",
    }
    seen: set[tuple[str, str, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise EventProbabilityError("event probability registry entry is malformed")
        _exact_keys(entry, expected_entry_keys, "event probability registry entry")
        for field in (
            "event_id",
            "league",
            "market_type",
            "selection",
            "opposing_selection",
        ):
            _nonempty(entry.get(field), f"entry.{field}")
        if entry["selection"] == entry["opposing_selection"]:
            raise EventProbabilityError("event probability registry selections match")
        _time(entry.get("captured_at_utc"), "entry.captured_at_utc")
        _validate_receipt_locator(entry.get("receipt_locator"))
        for field in (
            "receipt_sha256",
            "source_prediction_receipt_sha256",
            "source_prediction_registry_sha256",
        ):
            _sha(entry.get(field), f"entry.{field}")
        key = (
            entry["event_id"],
            entry["market_type"],
            entry["selection"],
            entry["opposing_selection"],
        )
        if key in seen:
            raise EventProbabilityError("event probability registry key is ambiguous")
        seen.add(key)
        normalized.append(dict(entry))
    ordered = sorted(
        normalized,
        key=lambda item: (
            item["event_id"],
            item["market_type"],
            item["selection"],
            item["opposing_selection"],
        ),
    )
    if normalized != ordered:
        raise EventProbabilityError("event probability registry is not canonical")
    return {**dict(registry), "entries": normalized, "registry_sha256": sha256_json(registry)}


def validate_registered_event_probability(
    *,
    receipt: Mapping[str, Any],
    expected_receipt_sha256: str | None,
    registry: Mapping[str, Any],
    expected_registry_sha256: str | None,
    event_id: str,
    league: str,
    market_type: str,
    selection: str,
    opposing_selection: str,
    model_artifact_sha256: str,
    market_protocol_artifact_sha256: str,
    calibration_artifact_sha256: str,
    uncertainty_artifact_sha256: str,
    generation_code_sha256: str,
    as_of: datetime,
) -> dict[str, Any]:
    if as_of.tzinfo is None:
        raise EventProbabilityError("as_of must include a timezone")
    checked_registry = validate_event_probability_registry(
        registry,
        expected_registry_sha256=expected_registry_sha256,
        expected_model_artifact_sha256=model_artifact_sha256,
        expected_market_protocol_artifact_sha256=market_protocol_artifact_sha256,
        expected_calibration_artifact_sha256=calibration_artifact_sha256,
        expected_uncertainty_artifact_sha256=uncertainty_artifact_sha256,
        expected_generation_code_sha256=generation_code_sha256,
    )
    if _time(checked_registry["issued_at"], "registry.issued_at") > as_of.astimezone(
        timezone.utc
    ):
        raise EventProbabilityError("event probability registry is from the future")
    wanted = (event_id, market_type, selection, opposing_selection)
    matches = [
        entry
        for entry in checked_registry["entries"]
        if (
            entry["event_id"],
            entry["market_type"],
            entry["selection"],
            entry["opposing_selection"],
        )
        == wanted
    ]
    if len(matches) != 1:
        raise EventProbabilityError("registered event probability is unavailable")
    entry = matches[0]
    if entry["league"] != league:
        raise EventProbabilityError("event probability registry league mismatch")
    receipt_sha = _sha(expected_receipt_sha256, "expected_receipt_sha256")
    if entry["receipt_sha256"] != receipt_sha:
        raise EventProbabilityError("event probability registry receipt mismatch")
    checked = validate_event_probability_receipt(
        receipt,
        expected_receipt_sha256=receipt_sha,
        expected_event_id=event_id,
        expected_league=league,
        expected_market_type=market_type,
        expected_selection=selection,
        expected_opposing_selection=opposing_selection,
        expected_model_artifact_sha256=model_artifact_sha256,
        expected_market_protocol_artifact_sha256=market_protocol_artifact_sha256,
        expected_calibration_artifact_sha256=calibration_artifact_sha256,
        expected_uncertainty_artifact_sha256=uncertainty_artifact_sha256,
        expected_generation_code_sha256=generation_code_sha256,
        as_of=as_of,
    )
    for field in (
        "source_prediction_receipt_sha256",
        "source_prediction_registry_sha256",
    ):
        if checked["bindings"][field] != entry[field]:
            raise EventProbabilityError(f"event probability {field} registry mismatch")
    return {
        **checked,
        "registry_id": checked_registry["registry_id"],
        "registry_sha256": checked_registry["registry_sha256"],
    }


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in items:
            if key in value:
                raise EventProbabilityError(f"{label} contains duplicate key {key!r}")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    except EventProbabilityError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise EventProbabilityError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise EventProbabilityError(f"{label} must be a JSON object")
    return value


def _safe_repo_file(root: Path, locator: str) -> Path:
    relative = PurePosixPath(locator)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise EventProbabilityError("event probability artifact path is invalid")
    root_real = root.resolve(strict=True)
    current = root_real
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise EventProbabilityError(
                "event probability artifact is missing"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise EventProbabilityError(
                "event probability artifact symlink is rejected"
            )
    metadata = os.lstat(current)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise EventProbabilityError(
            "event probability artifact must be an unaliased regular file"
        )
    try:
        current.resolve(strict=True).relative_to(root_real)
    except ValueError as exc:
        raise EventProbabilityError("event probability artifact escaped root") from exc
    return current


def load_registered_event_probability(
    *,
    registry_locator: str,
    expected_registry_sha256: str | None,
    event_id: str,
    league: str,
    market_type: str,
    selection: str,
    opposing_selection: str,
    model_artifact_sha256: str,
    market_protocol_artifact_sha256: str,
    calibration_artifact_sha256: str,
    uncertainty_artifact_sha256: str,
    generation_code_sha256: str,
    as_of: datetime,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load one event probability through an exact independently pinned registry."""
    if expected_registry_sha256 is None:
        raise EventProbabilityError("event probability registry is not registered")
    registry_path = _safe_repo_file(root, registry_locator)
    raw_registry = _strict_object(registry_path.read_bytes(), "probability registry")
    checked_registry = validate_event_probability_registry(
        raw_registry,
        expected_registry_sha256=expected_registry_sha256,
        expected_model_artifact_sha256=model_artifact_sha256,
        expected_market_protocol_artifact_sha256=market_protocol_artifact_sha256,
        expected_calibration_artifact_sha256=calibration_artifact_sha256,
        expected_uncertainty_artifact_sha256=uncertainty_artifact_sha256,
        expected_generation_code_sha256=generation_code_sha256,
    )
    if as_of.tzinfo is None:
        raise EventProbabilityError("as_of must include a timezone")
    if _time(checked_registry["issued_at"], "registry.issued_at") > as_of.astimezone(
        timezone.utc
    ):
        raise EventProbabilityError("event probability registry is from the future")
    wanted = (event_id, market_type, selection, opposing_selection)
    matches = [
        entry
        for entry in checked_registry["entries"]
        if (
            entry["event_id"],
            entry["market_type"],
            entry["selection"],
            entry["opposing_selection"],
        )
        == wanted
    ]
    if len(matches) != 1 or matches[0]["league"] != league:
        raise EventProbabilityError("registered event probability is unavailable")
    entry = matches[0]
    _validate_receipt_locator(entry["receipt_locator"])
    receipt_path = _safe_repo_file(root, entry["receipt_locator"])
    raw_receipt = _strict_object(receipt_path.read_bytes(), "probability receipt")
    checked = validate_registered_event_probability(
        receipt=raw_receipt,
        expected_receipt_sha256=entry["receipt_sha256"],
        registry=raw_registry,
        expected_registry_sha256=expected_registry_sha256,
        event_id=event_id,
        league=league,
        market_type=market_type,
        selection=selection,
        opposing_selection=opposing_selection,
        model_artifact_sha256=model_artifact_sha256,
        market_protocol_artifact_sha256=market_protocol_artifact_sha256,
        calibration_artifact_sha256=calibration_artifact_sha256,
        uncertainty_artifact_sha256=uncertainty_artifact_sha256,
        generation_code_sha256=generation_code_sha256,
        as_of=as_of,
    )
    return {
        "status": "registered",
        "receipt": raw_receipt,
        "receipt_sha256": checked["receipt_sha256"],
        "registry": raw_registry,
        "registry_id": checked["registry_id"],
        "registry_sha256": checked["registry_sha256"],
        "probability": checked["probability"],
        "probability_interval": checked["probability_interval"],
    }


__all__ = [
    "EventProbabilityError",
    "RECEIPT_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "build_event_probability_receipt",
    "build_event_probability_registry",
    "canonical_bytes",
    "load_registered_event_probability",
    "sha256_json",
    "validate_event_probability_receipt",
    "validate_event_probability_registry",
    "validate_registered_event_probability",
]
