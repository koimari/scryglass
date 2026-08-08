"""Qualify one phase-two Betano quote against authoritative actual map start.

The transport receipt is created before the map.  This receipt is necessarily
created afterwards: it joins the frozen response timestamp to outcome-free
actual-map-start authority and proves the protocol's five-second boundary.
It remains a candidate until an independent, externally pinned quote registry
accepts the exact files.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping

from . import betano_br_quote_adapter_v2 as quote_v2
from . import phase_one_evaluation_v1 as evaluation
from ..draft.terminal import future_prediction_ledger as draft_ledger


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/betano_br_quote_qualification_v1.py"
SCHEMA_VERSION = "scryglass:betano-br-map-winner-quote-qualification:v1"
RESULT_STATE = "QUOTE_RESPONSE_QUALIFIED_AGAINST_ACTUAL_MAP_START_NON_AUTHORIZING"
MINIMUM_RESPONSE_TO_START_SECONDS = 5.0
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/qualified-betano-quotes-v1"
)
AUTHORITY = {
    "quote_identity_authority": False,
    "odds_accuracy_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Exact outcome-free proof that one registered-protocol Betano response "
    "preceded authoritative actual map start by at least five seconds. "
    "Independent quote registration, odds accuracy, probability, expected "
    "value, recommendation, transaction, and betting authority remain absent."
)


class BetanoQuoteQualificationError(RuntimeError):
    """The quote, map-start authority, event join, or timing boundary failed."""


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
        raise BetanoQuoteQualificationError("qualification is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BetanoQuoteQualificationError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise BetanoQuoteQualificationError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise BetanoQuoteQualificationError("qualification clock must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    return [
        evaluation._source_record(root, locator)
        for locator in (
            SOURCE_LOCATOR,
            quote_v2.SOURCE_LOCATOR,
            "lol_kills/v2/draft/terminal/future_prediction_ledger.py",
        )
    ]


def _quote(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, quote_v2.OUTPUT_PREFIX, "quote_locator"
    )
    raw = evaluation._read_regular(root, locator, "Betano v2 quote")
    try:
        checked = quote_v2.validate_betano_map_winner_quote_v2(
            evaluation._strict_object(raw, "Betano v2 quote"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise BetanoQuoteQualificationError("Betano v2 quote is invalid") from exc
    return locator, raw, checked


def _map_start(
    *, root: Path, locator_value: str
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, draft_ledger.MAP_START_PREFIX, "map_start_locator"
    )
    raw = evaluation._read_regular(root, locator, "actual map-start receipt")
    try:
        checked = draft_ledger.validate_map_start_receipt(
            evaluation._strict_object(raw, "actual map-start receipt"), root=root
        )
    except Exception as exc:
        raise BetanoQuoteQualificationError("actual map-start receipt is invalid") from exc
    return locator, raw, checked


def _joined(
    *, quote: Mapping[str, Any], map_start: Mapping[str, Any], root: Path,
    environment: Mapping[str, str]
) -> dict[str, Any]:
    probability_locator = quote["event_probability_v2_binding"]["locator"]
    _, _, probability = quote_v2._probability(
        root=root, locator_value=probability_locator, environment=environment
    )
    event = probability["event"]
    start_event = map_start["event"]
    for field in ("event_id", "series_id", "game_number", "league"):
        if event.get(field) != start_event.get(field):
            raise BetanoQuoteQualificationError(f"quote and map start differ: {field}")
    frozen_quote = quote["frozen_v1_transport_quote"]
    response = _timestamp(
        frozen_quote["transport"]["response_received_at_utc"],
        "quote.response_received_at_utc",
    )
    request = _timestamp(
        frozen_quote["transport"]["request_started_at_utc"],
        "quote.request_started_at_utc",
    )
    prediction = _timestamp(probability["captured_at_utc"], "probability.captured_at")
    actual_start = _timestamp(
        start_event["actual_map_start_utc"], "map_start.actual_map_start_utc"
    )
    map_start_capture = _timestamp(map_start["captured_at_utc"], "map_start.captured_at")
    lead = (actual_start - response).total_seconds()
    if not prediction <= request <= response:
        raise BetanoQuoteQualificationError("probability and quote timing is not prospective")
    if lead < MINIMUM_RESPONSE_TO_START_SECONDS:
        raise BetanoQuoteQualificationError(
            "quote response did not precede actual map start by five seconds"
        )
    if map_start_capture < actual_start:
        raise BetanoQuoteQualificationError("map-start authority predates actual start")
    return {
        "event": {
            "event_id": event["event_id"],
            "series_id": event["series_id"],
            "game_number": event["game_number"],
            "league": event["league"],
            "patch": event["patch"],
            "roster_change_stratum": event["roster_change_stratum"],
            "sparse_or_new_champion_map": event[
                "sparse_or_new_champion_map"
            ],
            "market_type": event["market_type"],
            "selection": event["selection"],
            "opposing_selection": event["opposing_selection"],
        },
        "timing": {
            "probability_captured_at_utc": prediction.isoformat(),
            "quote_request_started_at_utc": request.isoformat(),
            "quote_response_received_at_utc": response.isoformat(),
            "actual_map_start_utc": actual_start.isoformat(),
            "map_start_captured_at_utc": map_start_capture.isoformat(),
            "quote_response_to_actual_map_start_seconds": lead,
            "minimum_required_seconds": MINIMUM_RESPONSE_TO_START_SECONDS,
        },
        "probability": probability,
    }


def build_betano_quote_qualification_v1(
    *,
    quote_locator: str,
    map_start_locator: str,
    qualification_output_locator: str,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    quote_locator, quote_raw, quote = _quote(
        root=root, locator_value=quote_locator, environment=environment
    )
    start_locator, start_raw, map_start = _map_start(
        root=root, locator_value=map_start_locator
    )
    planned_qualification_locator = evaluation._locator(
        qualification_output_locator,
        OUTPUT_PREFIX,
        "qualification_output_locator",
    )
    plan_binding = quote["event_plan_binding"]
    plan_raw = evaluation._read_regular(
        root, plan_binding["locator"], "phase-two event plan"
    )
    plan = quote_v2.event_plan.validate_phase_two_event_plan_v1(
        evaluation._strict_object(plan_raw, "phase-two event plan"),
        root=root,
        environment=environment,
    )
    if (
        plan["reserved_outputs"]["quote_locator"] != quote_locator
        or plan["reserved_outputs"]["qualification_locator"]
        != planned_qualification_locator
    ):
        raise BetanoQuoteQualificationError("event-plan reserved outputs differ")
    joined = _joined(
        quote=quote, map_start=map_start, root=root, environment=environment
    )
    built = _clock_sample(clock)
    if built < _timestamp(map_start["captured_at_utc"], "map_start.captured_at"):
        raise BetanoQuoteQualificationError("qualification predates map-start receipt")
    probability = joined.pop("probability")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "qualified_at_utc": built.isoformat(),
        "event": joined["event"],
        "quote_binding": {
            "locator": quote_locator,
            "raw_sha256": _sha256(quote_raw),
            "artifact_sha256": quote["artifact_sha256"],
            "generic_quote_receipt_sha256": quote["frozen_v1_transport_quote"][
                "generic_quote_receipt_sha256"
            ],
            "event_probability_artifact_sha256": probability["artifact_sha256"],
        },
        "event_plan_binding": dict(plan_binding),
        "qualification_output_locator": planned_qualification_locator,
        "map_start_binding": {
            "locator": start_locator,
            "raw_sha256": _sha256(start_raw),
            "artifact_sha256": map_start["artifact_sha256"],
        },
        "opening_binding": dict(probability["opening_binding"]),
        "timing": joined["timing"],
        "qualification": {
            "actual_map_start_authority_present": True,
            "quote_response_preceded_actual_map_start": True,
            "minimum_five_second_boundary_passed": True,
            "probability_preceded_quote_request": True,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
            "retrospective_backfill_qualifies": False,
            "eligible_for_independent_quote_registration": True,
            "quote_independently_registered": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_betano_quote_qualification_v1(
        payload, root=root, environment=environment
    )


def validate_betano_quote_qualification_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BetanoQuoteQualificationError("qualification must be an object")
    value = dict(payload)
    expected_keys = {
        "schema_version", "result_state", "qualified_at_utc", "event",
        "quote_binding", "map_start_binding", "opening_binding", "timing",
        "event_plan_binding", "qualification_output_locator",
        "qualification", "source_locks", "authority", "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected_keys:
        raise BetanoQuoteQualificationError("qualification structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise BetanoQuoteQualificationError("qualification hash changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise BetanoQuoteQualificationError("qualification identity changed")
    qualified = _timestamp(value.get("qualified_at_utc"), "qualified_at_utc")
    quote_binding = value.get("quote_binding") or {}
    start_binding = value.get("map_start_binding") or {}
    quote_locator, quote_raw, quote = _quote(
        root=root, locator_value=quote_binding.get("locator"), environment=environment
    )
    start_locator, start_raw, map_start = _map_start(
        root=root, locator_value=start_binding.get("locator")
    )
    joined = _joined(
        quote=quote, map_start=map_start, root=root, environment=environment
    )
    probability = joined.pop("probability")
    expected_quote_binding = {
        "locator": quote_locator,
        "raw_sha256": _sha256(quote_raw),
        "artifact_sha256": quote["artifact_sha256"],
        "generic_quote_receipt_sha256": quote["frozen_v1_transport_quote"][
            "generic_quote_receipt_sha256"
        ],
        "event_probability_artifact_sha256": probability["artifact_sha256"],
    }
    expected_start_binding = {
        "locator": start_locator,
        "raw_sha256": _sha256(start_raw),
        "artifact_sha256": map_start["artifact_sha256"],
    }
    if value.get("event") != joined["event"]:
        raise BetanoQuoteQualificationError("qualified event changed")
    if quote_binding != expected_quote_binding or start_binding != expected_start_binding:
        raise BetanoQuoteQualificationError("qualified file binding changed")
    plan_binding = value.get("event_plan_binding")
    if plan_binding != quote["event_plan_binding"]:
        raise BetanoQuoteQualificationError("qualification event-plan binding changed")
    plan_raw = evaluation._read_regular(
        root, plan_binding["locator"], "phase-two event plan"
    )
    plan = quote_v2.event_plan.validate_phase_two_event_plan_v1(
        evaluation._strict_object(plan_raw, "phase-two event plan"),
        root=root,
        environment=environment,
    )
    qualification_locator = evaluation._locator(
        value.get("qualification_output_locator"),
        OUTPUT_PREFIX,
        "qualification_output_locator",
    )
    if (
        plan["reserved_outputs"]["quote_locator"] != quote_locator
        or plan["reserved_outputs"]["qualification_locator"]
        != qualification_locator
    ):
        raise BetanoQuoteQualificationError("event-plan reserved outputs changed")
    if value.get("opening_binding") != probability["opening_binding"]:
        raise BetanoQuoteQualificationError("phase-two opening binding changed")
    if value.get("timing") != joined["timing"]:
        raise BetanoQuoteQualificationError("qualified timing changed")
    if qualified < _timestamp(map_start["captured_at_utc"], "map_start.captured_at"):
        raise BetanoQuoteQualificationError("qualification predates map-start receipt")
    if value.get("qualification") != {
        "actual_map_start_authority_present": True,
        "quote_response_preceded_actual_map_start": True,
        "minimum_five_second_boundary_passed": True,
        "probability_preceded_quote_request": True,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
        "retrospective_backfill_qualifies": False,
        "eligible_for_independent_quote_registration": True,
        "quote_independently_registered": False,
    }:
        raise BetanoQuoteQualificationError("qualification claims changed")
    if value.get("source_locks") != _source_locks(root):
        raise BetanoQuoteQualificationError("qualification source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise BetanoQuoteQualificationError("qualification exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise BetanoQuoteQualificationError(f"refusing to replace qualification: {path}")
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
            raise BetanoQuoteQualificationError(
                f"refusing to replace qualification: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


__all__ = [
    "MINIMUM_RESPONSE_TO_START_SECONDS", "OUTPUT_PREFIX", "SCHEMA_VERSION",
    "SOURCE_LOCATOR", "BetanoQuoteQualificationError",
    "build_betano_quote_qualification_v1",
    "validate_betano_quote_qualification_v1", "write_no_clobber",
]
