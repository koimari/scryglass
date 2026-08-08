"""Bind the frozen Betano transport to event-probability v2 without data drift.

The v1 transport consumes a v1 probability shape even though it only uses the
event identity, selections, and capture time.  This wrapper deterministically
creates a non-persisted, non-authorizing transport bridge.  If the v2
percentile interval excludes the plug-in point, only the bridge interval is
widened to satisfy the legacy type; the outer receipt proves that interval was
never used as a probability, quote feature, or decision input.
"""

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
from . import event_probability_v2 as probability_v2
from . import phase_one_evaluation_v1 as evaluation
from . import phase_two_event_plan_v1 as event_plan


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/betano_br_quote_adapter_v2.py"
SCHEMA_VERSION = "scryglass:betano-br-map-winner-quote-transport:v2"
RESULT_STATE = "V2_PROBABILITY_BOUND_QUOTE_CAPTURED_NON_AUTHORIZING"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/betano-quotes-v2"
)
AUTHORITY = {
    "quote_identity_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Exact Betano public response and frozen transport bound to one v2 event "
    "probability receipt. The legacy bridge is transport-only. Independent "
    "quote registration, actual-map-start qualification, settlement, market "
    "evaluation, recommendation, transaction, and betting authority remain required."
)


class BetanoQuoteAdapterV2Error(RuntimeError):
    """The v2 probability, bridge, transport, or exact file binding failed."""


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
        raise BetanoQuoteAdapterV2Error("v2 quote bundle is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BetanoQuoteAdapterV2Error(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise BetanoQuoteAdapterV2Error(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    locators = [
        SOURCE_LOCATOR,
        transport.SOURCE_LOCATOR,
        probability_v2.SOURCE_LOCATOR,
        event_plan.SOURCE_LOCATOR,
        "lol_kills/v2/market/event_probability_v1.py",
    ]
    return [evaluation._source_record(root, locator) for locator in locators]


def _probability(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, probability_v2.RECEIPT_PREFIX, "event_probability_locator"
    )
    raw = evaluation._read_regular(root, locator, "event probability v2")
    try:
        value = probability_v2.validate_event_probability_v2(
            evaluation._strict_object(raw, "event probability v2"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise BetanoQuoteAdapterV2Error("event probability v2 is invalid") from exc
    return locator, raw, value


def _transport_bridge(probability: Mapping[str, Any]) -> dict[str, Any]:
    event = probability["event"]
    calculation = probability["calculation"]
    uncertainty = probability["uncertainty"]
    point = float(calculation["probability"])
    original_interval = list(uncertainty["probability_interval"])
    bridge_interval = [
        min(float(original_interval[0]), point),
        max(float(original_interval[1]), point),
    ]
    bindings = probability["input_binding"]
    opening_binding = probability["opening_binding"]
    bridge = legacy_probability.build_event_probability_receipt(
        event_id=event["event_id"],
        league=event["league"],
        market_type=event["market_type"],
        selection=event["selection"],
        opposing_selection=event["opposing_selection"],
        model_artifact_sha256=bindings["target_prediction_artifact_sha256"],
        market_protocol_artifact_sha256=bindings[
            "market_protocol_artifact_sha256"
        ],
        calibration_artifact_sha256=bindings[
            "frozen_contract_candidate_artifact_sha256"
        ],
        uncertainty_artifact_sha256=bindings[
            "fast_uncertainty_artifact_sha256"
        ],
        source_prediction_receipt_sha256=bindings[
            "target_prediction_artifact_sha256"
        ],
        source_prediction_registry_sha256=opening_binding["marker_raw_sha256"],
        generation_code_sha256=bindings["generation_source_raw_sha256"],
        raw_model_probability=calculation["raw_model_probability"],
        calibration_intercept=calculation["calibration_intercept"],
        calibration_slope=calculation["calibration_slope"],
        probability_interval=bridge_interval,
        uncertainty_draws_sha256=uncertainty["draws_sha256"],
        uncertainty_resamples=uncertainty["resamples"],
        clock=lambda: _timestamp(
            probability["captured_at_utc"], "probability.captured_at"
        ),
    )
    return {
        "receipt": bridge,
        "receipt_sha256": legacy_probability.sha256_json(bridge),
        "original_v2_probability": point,
        "original_v2_probability_interval": original_interval,
        "bridge_probability_interval": bridge_interval,
        "bridge_interval_widened_only_for_legacy_transport_shape": (
            bridge_interval != original_interval
        ),
        "bridge_interval_used_by_transport_extraction_or_quote": False,
        "bridge_persisted_as_probability_evidence": False,
        "bridge_authority": dict(legacy_probability.AUTHORITY),
    }


def capture_betano_map_winner_quote_v2(
    *,
    event_plan_locator: str,
    event_probability_locator: str,
    request_url: str,
    betano_event_id: str,
    map_number: int,
    participant_bindings: Sequence[Mapping[str, str]],
    fetcher: transport.PublicDocumentFetcher,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic_ns: Callable[[], int] = transport.time.monotonic_ns,
) -> dict[str, Any]:
    locator, raw, probability = _probability(
        root=root,
        locator_value=event_probability_locator,
        environment=environment,
    )
    plan_locator = evaluation._locator(
        event_plan_locator, event_plan.OUTPUT_PREFIX, "event_plan_locator"
    )
    plan_raw = evaluation._read_regular(root, plan_locator, "phase-two event plan")
    try:
        plan = event_plan.validate_phase_two_event_plan_v1(
            evaluation._strict_object(plan_raw, "phase-two event plan"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise BetanoQuoteAdapterV2Error("phase-two event plan is invalid") from exc
    if (
        plan["probability_binding"]["locator"] != locator
        or plan["probability_binding"]["raw_sha256"] != _sha256_bytes(raw)
        or plan["probability_binding"]["artifact_sha256"]
        != probability["artifact_sha256"]
    ):
        raise BetanoQuoteAdapterV2Error("event plan and probability differ")
    bridge = _transport_bridge(probability)
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
        raise BetanoQuoteAdapterV2Error("frozen Betano transport failed") from exc
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": quote["captured_at_utc"],
        "event_probability_v2_binding": {
            "locator": locator,
            "raw_sha256": _sha256_bytes(raw),
            "artifact_sha256": probability["artifact_sha256"],
            "receipt_sha256": probability["receipt_sha256"],
            "captured_at_utc": probability["captured_at_utc"],
        },
        "event_plan_binding": {
            "locator": plan_locator,
            "raw_sha256": _sha256_bytes(plan_raw),
            "artifact_sha256": plan["artifact_sha256"],
            "planned_at_utc": plan["planned_at_utc"],
        },
        "transport_compatibility_bridge": {
            key: value for key, value in bridge.items() if key != "receipt"
        },
        "frozen_v1_transport_quote": quote,
        "qualification": {
            "phase_two_opening_active": True,
            "complete_terms_and_source_adapter_registered": True,
            "v2_probability_preceded_quote_request": True,
            "event_plan_preceded_quote_request": True,
            "legacy_bridge_changed_probability_point": False,
            "legacy_bridge_interval_used_by_quote": False,
            "exact_response_body_and_transport_replay_present": True,
            "actual_map_start_checked": False,
            "quote_independently_registered": False,
            "phase_two_evidence_qualifies": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_betano_map_winner_quote_v2(
        payload, root=root, environment=environment
    )


def validate_betano_map_winner_quote_v2(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BetanoQuoteAdapterV2Error("v2 quote bundle must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "captured_at_utc",
        "event_probability_v2_binding",
        "event_plan_binding",
        "transport_compatibility_bridge",
        "frozen_v1_transport_quote",
        "qualification",
        "source_locks",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise BetanoQuoteAdapterV2Error("v2 quote bundle structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise BetanoQuoteAdapterV2Error("v2 quote bundle hash changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise BetanoQuoteAdapterV2Error("v2 quote bundle identity changed")
    binding = value.get("event_probability_v2_binding")
    if not isinstance(binding, Mapping) or set(binding) != {
        "locator",
        "raw_sha256",
        "artifact_sha256",
        "receipt_sha256",
        "captured_at_utc",
    }:
        raise BetanoQuoteAdapterV2Error("v2 probability binding changed")
    locator, raw, probability = _probability(
        root=root,
        locator_value=str(binding["locator"]),
        environment=environment,
    )
    if binding != {
        "locator": locator,
        "raw_sha256": _sha256_bytes(raw),
        "artifact_sha256": probability["artifact_sha256"],
        "receipt_sha256": probability["receipt_sha256"],
        "captured_at_utc": probability["captured_at_utc"],
    }:
        raise BetanoQuoteAdapterV2Error("v2 probability file binding changed")
    plan_binding = value.get("event_plan_binding")
    if not isinstance(plan_binding, Mapping) or set(plan_binding) != {
        "locator", "raw_sha256", "artifact_sha256", "planned_at_utc"
    }:
        raise BetanoQuoteAdapterV2Error("event plan binding changed")
    plan_locator = evaluation._locator(
        str(plan_binding["locator"]), event_plan.OUTPUT_PREFIX, "event_plan_locator"
    )
    plan_raw = evaluation._read_regular(root, plan_locator, "phase-two event plan")
    try:
        plan = event_plan.validate_phase_two_event_plan_v1(
            evaluation._strict_object(plan_raw, "phase-two event plan"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise BetanoQuoteAdapterV2Error("phase-two event plan is invalid") from exc
    if plan_binding != {
        "locator": plan_locator,
        "raw_sha256": _sha256_bytes(plan_raw),
        "artifact_sha256": plan["artifact_sha256"],
        "planned_at_utc": plan["planned_at_utc"],
    } or plan["probability_binding"]["artifact_sha256"] != probability["artifact_sha256"]:
        raise BetanoQuoteAdapterV2Error("event plan file binding changed")
    bridge = _transport_bridge(probability)
    expected_bridge = {key: item for key, item in bridge.items() if key != "receipt"}
    if value.get("transport_compatibility_bridge") != expected_bridge:
        raise BetanoQuoteAdapterV2Error("transport compatibility bridge changed")
    try:
        quote = transport.validate_betano_map_winner_quote_v1(
            value.get("frozen_v1_transport_quote"), root=root
        )
    except Exception as exc:
        raise BetanoQuoteAdapterV2Error("frozen v1 transport quote is invalid") from exc
    legacy_binding = quote["prediction_binding"]
    probability_event = probability["event"]
    if (
        legacy_binding["event_probability_receipt_sha256"]
        != bridge["receipt_sha256"]
        or legacy_binding["event_probability_artifact_sha256"]
        != bridge["receipt"]["artifact_sha256"]
        or legacy_binding["prediction_captured_at_utc"]
        != probability["captured_at_utc"]
        or legacy_binding["scryglass_event_id"] != probability_event["event_id"]
        or legacy_binding["selection"] != probability_event["selection"]
        or legacy_binding["opposing_selection"]
        != probability_event["opposing_selection"]
        or value.get("captured_at_utc") != quote["captured_at_utc"]
    ):
        raise BetanoQuoteAdapterV2Error("transport did not bind the v2 event")
    if value.get("qualification") != {
        "phase_two_opening_active": True,
        "complete_terms_and_source_adapter_registered": True,
        "v2_probability_preceded_quote_request": True,
        "event_plan_preceded_quote_request": True,
        "legacy_bridge_changed_probability_point": False,
        "legacy_bridge_interval_used_by_quote": False,
        "exact_response_body_and_transport_replay_present": True,
        "actual_map_start_checked": False,
        "quote_independently_registered": False,
        "phase_two_evidence_qualifies": False,
    }:
        raise BetanoQuoteAdapterV2Error("v2 quote qualification changed")
    request_started = _timestamp(
        quote["transport"]["request_started_at_utc"], "quote.request_started"
    )
    if _timestamp(plan["planned_at_utc"], "plan.planned_at") > request_started:
        raise BetanoQuoteAdapterV2Error("event plan did not precede quote request")
    if (
        bridge["original_v2_probability"] != probability["probability"]
        or bridge["bridge_interval_used_by_transport_extraction_or_quote"] is not False
        or bridge["bridge_persisted_as_probability_evidence"] is not False
    ):
        raise BetanoQuoteAdapterV2Error("legacy bridge exceeded transport scope")
    if value.get("source_locks") != _source_locks(root):
        raise BetanoQuoteAdapterV2Error("v2 quote source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise BetanoQuoteAdapterV2Error("v2 quote bundle exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise BetanoQuoteAdapterV2Error(f"refusing to replace v2 quote: {path}")
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
            raise BetanoQuoteAdapterV2Error(
                f"refusing to replace v2 quote: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "OUTPUT_PREFIX",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "BetanoQuoteAdapterV2Error",
    "capture_betano_map_winner_quote_v2",
    "validate_betano_map_winner_quote_v2",
    "write_no_clobber",
]
