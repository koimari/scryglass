"""Exact live Betano quote bound to a production probability and authority."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping, Sequence

from . import betano_br_quote_adapter_v1 as transport
from . import event_probability_v1 as legacy_probability
from . import phase_one_evaluation_v1 as evaluation
from . import production_event_probability_v1 as production_probability


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/production_betano_quote_v1.py"
SCHEMA_VERSION = "scryglass:production-betano-map-winner-quote:v1"
RESULT_STATE = "PRIVATE_PRODUCTION_BETANO_QUOTE_CAPTURED"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/private_decision_support/match-winner/production-quotes-v1"
)
AUTHORITY = {
    "exact_quote_receipt_identity": True,
    "self_authorized_odds_accuracy": False,
    "self_authorized_expected_value": False,
    "self_authorized_recommendation": False,
    "transaction_authority": False,
    "stake_authority": False,
}
CLAIM_CEILING = (
    "Exact public Betano response and transport clocks bound to one production "
    "probability under semantic authority. It does not prove executable limits, "
    "acceptance, transaction success, or authorize a stake."
)


class ProductionBetanoQuoteError(RuntimeError):
    """The authority, probability, bridge, transport, event, or timing failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, allow_nan=False, ensure_ascii=False,
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProductionBetanoQuoteError("quote is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ProductionBetanoQuoteError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise ProductionBetanoQuoteError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


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
        raise ProductionBetanoQuoteError(
            "active semantic market authority is unavailable"
        ) from exc


def _source_locks(root: Path) -> list[dict[str, Any]]:
    from . import semantic_market_authority_v1 as market_authority

    return [
        evaluation._source_record(root, locator)
        for locator in (
            SOURCE_LOCATOR,
            production_probability.SOURCE_LOCATOR,
            transport.SOURCE_LOCATOR,
            market_authority.SOURCE_LOCATOR,
            "lol_kills/v2/market/event_probability_v1.py",
        )
    ]


def _probability(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, production_probability.OUTPUT_PREFIX,
        "production_probability_locator",
    )
    raw = evaluation._read_regular(root, locator, "production probability")
    try:
        checked = production_probability.validate_production_event_probability_v1(
            evaluation._strict_object(raw, "production probability"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise ProductionBetanoQuoteError("production probability is invalid") from exc
    return locator, raw, checked


def _bridge(probability: Mapping[str, Any]) -> dict[str, Any]:
    event = probability["event"]
    calculation = probability["calculation"]
    uncertainty = probability["uncertainty"]
    point = float(calculation["probability"])
    original_interval = list(uncertainty["probability_interval"])
    bridge_interval = [
        min(float(original_interval[0]), point),
        max(float(original_interval[1]), point),
    ]
    inputs = probability["input_binding"]
    authority = probability["semantic_market_authority_binding"]
    receipt = legacy_probability.build_event_probability_receipt(
        event_id=event["event_id"],
        league=event["league"],
        market_type=event["market_type"],
        selection=event["selection"],
        opposing_selection=event["opposing_selection"],
        model_artifact_sha256=inputs["target_prediction_artifact_sha256"],
        market_protocol_artifact_sha256=inputs["market_protocol_artifact_sha256"],
        calibration_artifact_sha256=inputs[
            "frozen_contract_candidate_artifact_sha256"
        ],
        uncertainty_artifact_sha256=inputs["fast_uncertainty_artifact_sha256"],
        source_prediction_receipt_sha256=inputs[
            "target_prediction_artifact_sha256"
        ],
        source_prediction_registry_sha256=authority["authority_raw_sha256"],
        generation_code_sha256=inputs["generation_source_raw_sha256"],
        raw_model_probability=calculation["raw_model_probability"],
        calibration_intercept=calculation["calibration_intercept"],
        calibration_slope=calculation["calibration_slope"],
        probability_interval=bridge_interval,
        uncertainty_draws_sha256=uncertainty["draws_sha256"],
        uncertainty_resamples=uncertainty["resamples"],
        clock=lambda: _timestamp(probability["captured_at_utc"], "captured_at"),
    )
    return {
        "receipt": receipt,
        "receipt_sha256": legacy_probability.sha256_json(receipt),
        "original_probability": point,
        "original_probability_interval": original_interval,
        "bridge_probability_interval": bridge_interval,
        "interval_widened_only_for_legacy_transport_shape": bridge_interval
        != original_interval,
        "bridge_used_as_probability_or_decision_input": False,
        "bridge_persisted_as_probability_evidence": False,
    }


def _authority_binding(active: Mapping[str, Any]) -> dict[str, Any]:
    receipt = active["receipt"]
    return {
        "authority_id": receipt["authority_id"],
        "authority_raw_sha256": active["receipt_raw_sha256"],
        "valid_until_utc": receipt["valid_until_utc"],
        "private_expected_value_authority": True,
        "transaction_authorized": False,
        "stake_authorized": False,
    }


def capture_production_betano_quote_v1(
    *, production_probability_locator: str, request_url: str,
    betano_event_id: str, map_number: int,
    participant_bindings: Sequence[Mapping[str, str]],
    fetcher: transport.PublicDocumentFetcher,
    root: Path = ROOT, environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic_ns: Callable[[], int] = transport.time.monotonic_ns,
) -> dict[str, Any]:
    locator, raw, probability = _probability(
        root=root,
        locator_value=production_probability_locator,
        environment=environment,
    )
    bridge = _bridge(probability)
    try:
        quote = transport.capture_betano_map_winner_quote_v1(
            probability_receipt=bridge["receipt"],
            request_url=request_url,
            betano_event_id=betano_event_id,
            map_number=map_number,
            participant_bindings=participant_bindings,
            fetcher=fetcher,
            root=root,
            clock=clock,
            monotonic_ns=monotonic_ns,
        )
    except Exception as exc:
        raise ProductionBetanoQuoteError("frozen Betano transport failed") from exc
    response = _timestamp(
        quote["transport"]["response_received_at_utc"], "response_received_at"
    )
    active = _semantic_authority(root=root, environment=environment, as_of=response)
    if _authority_binding(active)["authority_raw_sha256"] != probability[
        "semantic_market_authority_binding"
    ]["authority_raw_sha256"]:
        raise ProductionBetanoQuoteError("probability and quote authorities differ")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": quote["captured_at_utc"],
        "production_probability_binding": {
            "locator": locator,
            "raw_sha256": _sha256(raw),
            "artifact_sha256": probability["artifact_sha256"],
            "captured_at_utc": probability["captured_at_utc"],
        },
        "semantic_market_authority_binding": _authority_binding(active),
        "transport_compatibility_bridge": {
            key: value for key, value in bridge.items() if key != "receipt"
        },
        "frozen_public_transport_quote": quote,
        "qualification": {
            "semantic_market_authority_active_at_response": True,
            "production_probability_preceded_request": True,
            "exact_response_body_and_transport_replay_present": True,
            "event_market_and_two_selection_binding_replayed": True,
            "scheduled_event_start_and_market_close_are_future_at_response": True,
            "quote_freshness_for_decision_not_yet_checked": True,
            "cash_acceptance_limit_or_execution_proven": False,
            "transaction_or_stake_authorized": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    validate_production_betano_quote_v1(
        payload, root=root, environment=environment
    )
    return payload


def validate_production_betano_quote_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ProductionBetanoQuoteError("quote must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "captured_at_utc",
        "production_probability_binding", "semantic_market_authority_binding",
        "transport_compatibility_bridge", "frozen_public_transport_quote",
        "qualification", "source_locks", "authority", "claim_ceiling",
        "artifact_sha256",
    }:
        raise ProductionBetanoQuoteError("quote structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise ProductionBetanoQuoteError("quote hash changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise ProductionBetanoQuoteError("quote identity changed")
    binding = value.get("production_probability_binding") or {}
    locator, raw, probability = _probability(
        root=root, locator_value=binding.get("locator"), environment=environment
    )
    if binding != {
        "locator": locator,
        "raw_sha256": _sha256(raw),
        "artifact_sha256": probability["artifact_sha256"],
        "captured_at_utc": probability["captured_at_utc"],
    }:
        raise ProductionBetanoQuoteError("probability file binding changed")
    bridge = _bridge(probability)
    if value.get("transport_compatibility_bridge") != {
        key: item for key, item in bridge.items() if key != "receipt"
    }:
        raise ProductionBetanoQuoteError("transport bridge changed")
    try:
        quote = transport.validate_betano_map_winner_quote_v1(
            value.get("frozen_public_transport_quote"), root=root
        )
    except Exception as exc:
        raise ProductionBetanoQuoteError("public transport quote is invalid") from exc
    response = _timestamp(
        quote["transport"]["response_received_at_utc"], "response_received_at"
    )
    active = _semantic_authority(root=root, environment=environment, as_of=response)
    if value.get("semantic_market_authority_binding") != _authority_binding(active):
        raise ProductionBetanoQuoteError("semantic authority binding changed")
    event = probability["event"]
    prediction = quote["prediction_binding"]
    if (
        prediction["event_probability_receipt_sha256"] != bridge["receipt_sha256"]
        or prediction["prediction_captured_at_utc"] != probability["captured_at_utc"]
        or prediction["scryglass_event_id"] != event["event_id"]
        or prediction["selection"] != event["selection"]
        or prediction["opposing_selection"] != event["opposing_selection"]
        or value.get("captured_at_utc") != quote["captured_at_utc"]
    ):
        raise ProductionBetanoQuoteError("transport did not bind production probability")
    request = _timestamp(
        quote["transport"]["request_started_at_utc"], "request_started_at"
    )
    scheduled = _timestamp(event["scheduled_event_start_utc"], "scheduled_start")
    source_scheduled_ms = quote["source_extraction"]["betano_event"][
        "scheduled_series_start_epoch_ms"
    ]
    if (
        _timestamp(probability["captured_at_utc"], "probability.captured_at") > request
        or response >= scheduled
        or int(scheduled.timestamp() * 1000) != source_scheduled_ms
    ):
        raise ProductionBetanoQuoteError("probability, response, or scheduled start timing changed")
    if value.get("qualification") != {
        "semantic_market_authority_active_at_response": True,
        "production_probability_preceded_request": True,
        "exact_response_body_and_transport_replay_present": True,
        "event_market_and_two_selection_binding_replayed": True,
        "scheduled_event_start_and_market_close_are_future_at_response": True,
        "quote_freshness_for_decision_not_yet_checked": True,
        "cash_acceptance_limit_or_execution_proven": False,
        "transaction_or_stake_authorized": False,
    }:
        raise ProductionBetanoQuoteError("quote qualification changed")
    if value.get("source_locks") != _source_locks(root):
        raise ProductionBetanoQuoteError("quote source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise ProductionBetanoQuoteError("quote exceeds authority")
    return {
        **value,
        "probability": probability,
        "prices": dict(quote["generic_quote_receipt"]["prices"]),
        "response_received_at_utc": quote["transport"]["response_received_at_utc"],
    }


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProductionBetanoQuoteError(f"refusing to replace quote: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise ProductionBetanoQuoteError(f"refusing to replace quote: {path}") from exc
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return _sha256(raw)


__all__ = [
    "OUTPUT_PREFIX", "SCHEMA_VERSION", "SOURCE_LOCATOR",
    "ProductionBetanoQuoteError", "capture_production_betano_quote_v1",
    "validate_production_betano_quote_v1", "write_no_clobber",
]
