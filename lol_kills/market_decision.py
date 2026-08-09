"""Private, fail-closed market-decision authority.

This module deliberately separates descriptive model diagnostics from a
betting decision.  A numeric model output and a bookmaker price are not enough
to authorize probability, fair-odds, edge, EV, or wager language.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from lol_kills.bookmaker_quote_capture import (
    QuoteCaptureError,
    RECEIPT_SCHEMA_VERSION as QUOTE_SCHEMA_VERSION,
    validate_quote_receipt,
)
from lol_kills.v2.market.event_probability_v1 import (
    EventProbabilityError,
    validate_registered_event_probability,
)
from lol_kills.v2.market.match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as MATCH_WINNER_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256 as MATCH_WINNER_CAPTURE_PROTOCOL_SHA256,
    REGISTERED_SETTLEMENT_CONTRACT_SHA256 as MATCH_WINNER_SETTLEMENT_RULES_SHA256,
    MatchWinnerFutureProtocolRegistryError,
    validate_registered_match_winner_future_protocol_v1,
)
from lol_kills.v2.market.match_winner_future_protocol_v1 import (
    BOOKMAKER_ID as MATCH_WINNER_BOOKMAKER_ID,
    SETTLEMENT_RULE_ID as MATCH_WINNER_SETTLEMENT_RULE_ID,
)


SCHEMA_VERSION = "scryglass.private-market-authority.v2"
NO_AUTHORIZED_BET = "NO_AUTHORIZED_BET"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DEFAULT_QUOTE_MAX_AGE_SECONDS = 60.0
DEFAULT_PROBABILITY_MAX_AGE_SECONDS = 60.0
AUTHORITY_FIELDS = {
    "schema_version",
    "status",
    "scope",
    "public_or_transactional_use",
    "betting_decision_authorized",
    "authority_record_id",
    "independent_reviewer_id",
    "issued_at",
    "valid_until",
    "model_artifact_sha256",
    "bookmaker_id",
    "settlement_rule_id",
    "market_protocol_artifact_sha256",
    "phase_two_evaluation_artifact_sha256",
    "historical_market_benchmark_sha256",
    "bookmaker_terms_artifact_sha256",
    "settlement_rules_sha256",
    "capture_protocol_sha256",
    "reliability_artifact_sha256",
    "calibration_artifact_sha256",
    "uncertainty_artifact_sha256",
    "event_probability_registry_sha256",
    "event_probability_generation_code_sha256",
    "quote_registry_sha256",
    "leagues",
    "market_types",
    "no_vig_method",
    "out_of_sample_market_comparison",
    "probability_calibration",
    "shadow_policy_evaluation",
    "prospective_latency",
    "dependence_aware_uncertainty",
    "quote_coverage",
    "settlement_review",
    "minimum_edge_pp",
    "minimum_expected_return",
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _parse_time(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be an RFC-3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an RFC-3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def unavailable_authority(*blockers: str) -> dict[str, Any]:
    reasons = sorted(
        set(blockers or ("independent_market_authority_not_registered",))
    )
    return {
        "status": "unavailable",
        "betting_decision_authorized": False,
        "blockers": reasons,
        "authority_record_id": None,
        "minimum_edge_pp": None,
        "minimum_expected_return": None,
    }


def validate_authority_receipt(
    receipt: Mapping[str, Any] | None,
    *,
    expected_sha256: str | None,
    league: str,
    market_type: str,
    model_artifact_sha256: str,
    as_of: datetime,
) -> dict[str, Any]:
    """Validate an independently pinned private betting-authority receipt.

    The expected digest must come from a separate trusted registration step.
    A self-consistent JSON file cannot authorize itself.
    """
    blockers: list[str] = []
    if as_of.tzinfo is None:
        blockers.append("market_authority_as_of_timezone_missing")
    if expected_sha256 is None:
        blockers.append("independent_market_authority_not_registered")
    elif not SHA256_RE.fullmatch(expected_sha256):
        blockers.append("registered_market_authority_digest_invalid")
    if not isinstance(receipt, Mapping):
        blockers.append("market_authority_receipt_missing")
        return unavailable_authority(*blockers)

    if set(receipt) != AUTHORITY_FIELDS:
        blockers.append("market_authority_contract_fields_changed")

    try:
        actual_sha256 = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    except (TypeError, ValueError):
        actual_sha256 = None
        blockers.append("market_authority_not_canonical_json")
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        blockers.append("market_authority_digest_mismatch")
    if receipt.get("schema_version") != SCHEMA_VERSION:
        blockers.append("market_authority_schema_unrecognized")
    if receipt.get("status") != "approved":
        blockers.append("market_authority_not_approved")
    if receipt.get("scope") != "private_personal_decision_support":
        blockers.append("market_authority_scope_invalid")
    if receipt.get("public_or_transactional_use") is not False:
        blockers.append("market_authority_public_scope_forbidden")
    if receipt.get("betting_decision_authorized") is not True:
        blockers.append("betting_decision_not_authorized")

    for field in (
        "model_artifact_sha256",
        "market_protocol_artifact_sha256",
        "phase_two_evaluation_artifact_sha256",
        "historical_market_benchmark_sha256",
        "bookmaker_terms_artifact_sha256",
        "settlement_rules_sha256",
        "capture_protocol_sha256",
        "reliability_artifact_sha256",
        "calibration_artifact_sha256",
        "uncertainty_artifact_sha256",
        "event_probability_registry_sha256",
        "event_probability_generation_code_sha256",
        "quote_registry_sha256",
    ):
        value = receipt.get(field)
        if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
            blockers.append(f"market_authority_{field}_invalid")
    if receipt.get("model_artifact_sha256") != model_artifact_sha256:
        blockers.append("market_authority_model_binding_mismatch")

    if market_type == "match_winner":
        blockers.append(
            "legacy_v1_match_winner_decision_path_non_authorizing_use_semantic_v2"
        )
        try:
            validate_registered_match_winner_future_protocol_v1()
        except (MatchWinnerFutureProtocolRegistryError, OSError, ValueError):
            blockers.append("registered_match_winner_protocol_invalid")
        if (
            receipt.get("market_protocol_artifact_sha256")
            != MATCH_WINNER_PROTOCOL_ARTIFACT_SHA256
        ):
            blockers.append("market_authority_protocol_binding_mismatch")
        if (
            receipt.get("capture_protocol_sha256")
            != MATCH_WINNER_CAPTURE_PROTOCOL_SHA256
        ):
            blockers.append("market_authority_capture_contract_mismatch")
        if (
            receipt.get("settlement_rules_sha256")
            != MATCH_WINNER_SETTLEMENT_RULES_SHA256
        ):
            blockers.append("market_authority_settlement_contract_mismatch")
        if receipt.get("bookmaker_id") != MATCH_WINNER_BOOKMAKER_ID:
            blockers.append("market_authority_bookmaker_binding_mismatch")
        if receipt.get("settlement_rule_id") != MATCH_WINNER_SETTLEMENT_RULE_ID:
            blockers.append("market_authority_settlement_rule_binding_mismatch")
    else:
        blockers.append(f"registered_market_protocol_unavailable:{market_type}")

    leagues = receipt.get("leagues")
    if (
        not isinstance(leagues, list)
        or not leagues
        or any(not isinstance(item, str) or not item.strip() for item in leagues)
        or len(set(leagues)) != len(leagues)
    ):
        blockers.append("market_authority_leagues_invalid")
        leagues = []
    market_types = receipt.get("market_types")
    if (
        not isinstance(market_types, list)
        or not market_types
        or any(
            not isinstance(item, str) or not item.strip() for item in market_types
        )
        or len(set(market_types)) != len(market_types)
    ):
        blockers.append("market_authority_market_types_invalid")
        market_types = []
    if league not in set(leagues):
        blockers.append(f"market_authority_league_unavailable:{league}")
    if market_type not in set(market_types):
        blockers.append(f"market_authority_market_unavailable:{market_type}")
    if receipt.get("no_vig_method") != "two_way_normalized_implied_probability":
        blockers.append("market_authority_no_vig_method_invalid")
    for gate in (
        "out_of_sample_market_comparison",
        "probability_calibration",
        "shadow_policy_evaluation",
        "prospective_latency",
        "dependence_aware_uncertainty",
        "quote_coverage",
        "settlement_review",
    ):
        if receipt.get(gate) != "passed":
            blockers.append(f"market_authority_gate_failed:{gate}")

    try:
        issued_at = _parse_time(receipt.get("issued_at"), "issued_at")
        valid_until = _parse_time(receipt.get("valid_until"), "valid_until")
    except ValueError as exc:
        blockers.append(str(exc))
    else:
        now = as_of.astimezone(timezone.utc)
        if now < issued_at:
            blockers.append("market_authority_not_yet_valid")
        if now > valid_until:
            blockers.append("market_authority_expired")
        if valid_until <= issued_at:
            blockers.append("market_authority_validity_window_invalid")
        elif (valid_until - issued_at).total_seconds() > 30 * 86400:
            blockers.append("market_authority_validity_window_too_long")

    for field in ("authority_record_id", "independent_reviewer_id"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            blockers.append(f"market_authority_{field}_missing")
    for field in ("minimum_edge_pp", "minimum_expected_return"):
        value = receipt.get(field)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0
        ):
            blockers.append(f"market_authority_{field}_invalid")
    minimum_return = receipt.get("minimum_expected_return")
    if isinstance(minimum_return, (int, float)) and float(minimum_return) < 0.02:
        blockers.append("market_authority_expected_return_floor_below_protocol")

    if blockers:
        return unavailable_authority(*blockers)
    return {
        "status": "approved",
        "betting_decision_authorized": True,
        "blockers": [],
        "authority_record_id": receipt["authority_record_id"],
        "authority_sha256": actual_sha256,
        "league": league,
        "market_type": market_type,
        "bookmaker_id": receipt["bookmaker_id"],
        "settlement_rule_id": receipt["settlement_rule_id"],
        "model_artifact_sha256": receipt["model_artifact_sha256"],
        "market_protocol_artifact_sha256": receipt[
            "market_protocol_artifact_sha256"
        ],
        "phase_two_evaluation_artifact_sha256": receipt[
            "phase_two_evaluation_artifact_sha256"
        ],
        "bookmaker_terms_artifact_sha256": receipt[
            "bookmaker_terms_artifact_sha256"
        ],
        "capture_protocol_sha256": receipt["capture_protocol_sha256"],
        "settlement_rules_sha256": receipt["settlement_rules_sha256"],
        "calibration_artifact_sha256": receipt["calibration_artifact_sha256"],
        "uncertainty_artifact_sha256": receipt["uncertainty_artifact_sha256"],
        "event_probability_registry_sha256": receipt[
            "event_probability_registry_sha256"
        ],
        "event_probability_generation_code_sha256": receipt[
            "event_probability_generation_code_sha256"
        ],
        "quote_registry_sha256": receipt["quote_registry_sha256"],
        "minimum_edge_pp": float(receipt["minimum_edge_pp"]),
        "minimum_expected_return": float(receipt["minimum_expected_return"]),
    }


def two_way_no_vig(odds: Sequence[float | int | None]) -> dict[str, Any]:
    """Return raw and normalized implied probabilities for exactly two sides."""
    if len(odds) != 2:
        return {"status": "unavailable", "blockers": ["two_way_quote_required"]}
    values: list[float] = []
    for value in odds:
        if value is None:
            return {"status": "unavailable", "blockers": ["two_way_quote_incomplete"]}
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return {"status": "unavailable", "blockers": ["decimal_odds_invalid"]}
        try:
            number = float(value)
        except (TypeError, ValueError):
            return {"status": "unavailable", "blockers": ["decimal_odds_invalid"]}
        if not math.isfinite(number) or number <= 1.0:
            return {"status": "unavailable", "blockers": ["decimal_odds_invalid"]}
        values.append(number)
    raw = [1.0 / value for value in values]
    total = sum(raw)
    return {
        "status": "available",
        "blockers": [],
        "odds": values,
        "raw_implied_probability": raw,
        "no_vig_probability": [value / total for value in raw],
        "overround": total - 1.0,
    }


def evaluate_two_way_market(
    *,
    model_probability: float | None,
    probability_interval: Sequence[float] | None,
    probability_receipt: Mapping[str, Any] | None = None,
    expected_probability_sha256: str | None = None,
    probability_registry: Mapping[str, Any] | None = None,
    expected_probability_registry_sha256: str | None = None,
    offered_odds: float | None,
    opposing_odds: float | None,
    quote: Mapping[str, Any] | None,
    expected_quote_sha256: str | None,
    quote_registry_sha256: str | None = None,
    authority: Mapping[str, Any] | None,
    expected_authority_sha256: str | None = None,
    as_of: datetime,
    selection: str,
    opposing_selection: str,
    event_id: str,
    market_type: str,
    settlement_rule_id: str,
    quote_max_age_seconds: float = DEFAULT_QUOTE_MAX_AGE_SECONDS,
    probability_max_age_seconds: float = DEFAULT_PROBABILITY_MAX_AGE_SECONDS,
) -> dict[str, Any]:
    """Evaluate one side of a two-way market without crossing authority lanes."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    market = two_way_no_vig((offered_odds, opposing_odds))
    blockers = list(market.get("blockers") or [])
    authority = authority or unavailable_authority()
    if authority.get("status") != "approved" or authority.get(
        "betting_decision_authorized"
    ) is not True:
        blockers.extend(authority.get("blockers") or ["market_authority_unavailable"])
    else:
        if expected_authority_sha256 is None:
            blockers.append("independent_market_authority_not_registered")
        elif not SHA256_RE.fullmatch(expected_authority_sha256):
            blockers.append("registered_market_authority_digest_invalid")
        elif authority.get("authority_sha256") != expected_authority_sha256:
            blockers.append("market_authority_digest_binding_mismatch")
        if authority.get("market_type") != market_type:
            blockers.append("market_authority_market_binding_mismatch")
        if authority.get("settlement_rule_id") != settlement_rule_id:
            blockers.append("market_authority_settlement_rule_binding_mismatch")

    diagnostic_probability = None
    if model_probability is not None:
        try:
            candidate = float(model_probability)
        except (TypeError, ValueError):
            candidate = math.nan
        if math.isfinite(candidate) and 0.0 < candidate < 1.0:
            diagnostic_probability = candidate
    diagnostic_interval = None
    try:
        diagnostic_interval_length = (
            len(probability_interval) if probability_interval is not None else 0
        )
    except TypeError:
        diagnostic_interval_length = 0
    if probability_interval is not None and diagnostic_interval_length == 2:
        try:
            candidate_interval = [
                float(probability_interval[0]),
                float(probability_interval[1]),
            ]
        except (TypeError, ValueError):
            candidate_interval = [math.nan, math.nan]
        if (
            all(math.isfinite(item) for item in candidate_interval)
            and 0.0 <= candidate_interval[0] <= candidate_interval[1] <= 1.0
        ):
            diagnostic_interval = candidate_interval

    probability = lower = upper = None
    probability_captured_at = None
    registered_probability_sha256 = None
    registered_probability_registry_sha256 = None
    if not isinstance(probability_receipt, Mapping):
        blockers.append("event_probability_receipt_missing")
    if not isinstance(probability_registry, Mapping):
        blockers.append("event_probability_registry_missing")
    if expected_probability_sha256 is None:
        blockers.append("independent_event_probability_not_registered")
    elif not SHA256_RE.fullmatch(expected_probability_sha256):
        blockers.append("registered_event_probability_digest_invalid")
    if expected_probability_registry_sha256 is None:
        blockers.append("independent_event_probability_registry_not_registered")
    elif not SHA256_RE.fullmatch(expected_probability_registry_sha256):
        blockers.append("registered_event_probability_registry_digest_invalid")
    if authority.get("status") == "approved":
        if (
            expected_probability_registry_sha256
            != authority.get("event_probability_registry_sha256")
        ):
            blockers.append("event_probability_registry_authority_binding_mismatch")
        if (
            isinstance(probability_receipt, Mapping)
            and isinstance(probability_registry, Mapping)
            and expected_probability_sha256 is not None
            and expected_probability_registry_sha256 is not None
        ):
            try:
                checked_probability = validate_registered_event_probability(
                    receipt=probability_receipt,
                    expected_receipt_sha256=expected_probability_sha256,
                    registry=probability_registry,
                    expected_registry_sha256=expected_probability_registry_sha256,
                    event_id=event_id,
                    league=str(authority.get("league")),
                    market_type=market_type,
                    selection=selection,
                    opposing_selection=opposing_selection,
                    model_artifact_sha256=str(
                        authority.get("model_artifact_sha256")
                    ),
                    market_protocol_artifact_sha256=str(
                        authority.get("market_protocol_artifact_sha256")
                    ),
                    calibration_artifact_sha256=str(
                        authority.get("calibration_artifact_sha256")
                    ),
                    uncertainty_artifact_sha256=str(
                        authority.get("uncertainty_artifact_sha256")
                    ),
                    generation_code_sha256=str(
                        authority.get("event_probability_generation_code_sha256")
                    ),
                    as_of=as_of,
                )
            except (EventProbabilityError, TypeError, ValueError):
                blockers.append("registered_event_probability_invalid")
            else:
                probability = float(checked_probability["probability"])
                lower, upper = checked_probability["probability_interval"]
                probability_captured_at = _parse_time(
                    checked_probability["captured_at_utc"],
                    "probability_captured_at_utc",
                )
                probability_age = (
                    as_of.astimezone(timezone.utc) - probability_captured_at
                ).total_seconds()
                if probability_age < 0:
                    blockers.append("event_probability_from_future")
                elif probability_age > probability_max_age_seconds:
                    blockers.append("event_probability_stale")
                registered_probability_sha256 = expected_probability_sha256
                registered_probability_registry_sha256 = (
                    expected_probability_registry_sha256
                )

    if not isinstance(quote, Mapping):
        blockers.append("market_quote_provenance_missing")
    else:
        if authority.get("status") == "approved":
            if quote_registry_sha256 is None:
                blockers.append("market_quote_registry_missing")
            elif not SHA256_RE.fullmatch(quote_registry_sha256):
                blockers.append("market_quote_registry_digest_invalid")
            elif quote_registry_sha256 != authority.get("quote_registry_sha256"):
                blockers.append("market_quote_registry_authority_binding_mismatch")
        if expected_quote_sha256 is None:
            blockers.append("independent_market_quote_not_registered")
        elif not SHA256_RE.fullmatch(expected_quote_sha256):
            blockers.append("registered_market_quote_digest_invalid")
        else:
            try:
                actual_quote_sha256 = hashlib.sha256(
                    _canonical_bytes(quote)
                ).hexdigest()
            except (TypeError, ValueError):
                blockers.append("market_quote_not_canonical_json")
            else:
                if actual_quote_sha256 != expected_quote_sha256:
                    blockers.append("market_quote_digest_mismatch")
        try:
            checked_quote = validate_quote_receipt(
                quote,
                expected_quote_sha256=expected_quote_sha256,
                expected_capture_protocol_sha256=(
                    authority.get("capture_protocol_sha256")
                    if authority.get("status") == "approved"
                    else None
                ),
                expected_settlement_rules_sha256=(
                    authority.get("settlement_rules_sha256")
                    if authority.get("status") == "approved"
                    else None
                ),
            )
        except QuoteCaptureError:
            checked_quote = None
            blockers.append("market_quote_receipt_invalid")
        required = (
            "schema_version",
            "source",
            "source_url",
            "source_record_id",
            "source_payload_sha256",
            "source_payload_base64",
            "extraction_payload_sha256",
            "extraction_payload_base64",
            "captured_at_utc",
            "event_id",
            "market_type",
            "settlement_rule_id",
            "capture_protocol_sha256",
            "settlement_rules_sha256",
            "extractor_id",
            "extractor_sha256",
        )
        for field in required:
            value = quote.get(field)
            if not isinstance(value, str) or not value.strip():
                blockers.append(f"market_quote_{field}_missing")
        payload_sha = quote.get("source_payload_sha256")
        if isinstance(payload_sha, str) and not SHA256_RE.fullmatch(payload_sha):
            blockers.append("market_quote_source_payload_sha256_invalid")
        payload_base64 = quote.get("source_payload_base64")
        if isinstance(payload_base64, str) and payload_base64:
            try:
                payload_raw = base64.b64decode(payload_base64, validate=True)
            except (TypeError, ValueError):
                blockers.append("market_quote_source_payload_base64_invalid")
            else:
                if (
                    not payload_raw
                    or not isinstance(payload_sha, str)
                    or hashlib.sha256(payload_raw).hexdigest() != payload_sha
                ):
                    blockers.append("market_quote_source_payload_digest_mismatch")
        if quote.get("schema_version") != QUOTE_SCHEMA_VERSION:
            blockers.append("market_quote_schema_unrecognized")
        for field, expected in (
            ("event_id", event_id),
            ("market_type", market_type),
            ("settlement_rule_id", settlement_rule_id),
            (
                "capture_protocol_sha256",
                authority.get("capture_protocol_sha256"),
            ),
            (
                "settlement_rules_sha256",
                authority.get("settlement_rules_sha256"),
            ),
        ):
            if quote.get(field) != expected:
                blockers.append(f"market_quote_{field}_binding_mismatch")
        prices = quote.get("prices")
        if not isinstance(prices, Mapping):
            blockers.append("market_quote_prices_missing")
            prices = {}
        for selection_key, expected in (
            (selection, offered_odds),
            (opposing_selection, opposing_odds),
        ):
            value = prices.get(selection_key)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                blockers.append(f"market_quote_price_invalid:{selection_key}")
            elif expected is None or not math.isclose(
                float(value), float(expected), rel_tol=0.0, abs_tol=1e-12
            ):
                blockers.append(f"market_quote_price_binding_mismatch:{selection_key}")
        try:
            captured_at = _parse_time(
                quote.get("captured_at_utc"), "captured_at_utc"
            )
        except ValueError as exc:
            blockers.append(str(exc))
        else:
            age = (as_of.astimezone(timezone.utc) - captured_at).total_seconds()
            if age < 0:
                blockers.append("market_quote_from_future")
            elif age > quote_max_age_seconds:
                blockers.append("market_quote_stale")
            if (
                probability_captured_at is not None
                and captured_at < probability_captured_at
            ):
                blockers.append("market_quote_predates_event_probability")

    unique_blockers = sorted(set(blockers))
    safe_offered = (
        float(offered_odds)
        if isinstance(offered_odds, (int, float))
        and not isinstance(offered_odds, bool)
        and math.isfinite(float(offered_odds))
        else None
    )
    safe_opposing = (
        float(opposing_odds)
        if isinstance(opposing_odds, (int, float))
        and not isinstance(opposing_odds, bool)
        and math.isfinite(float(opposing_odds))
        else None
    )
    descriptive_market = {
        "status": market.get("status"),
        "offered_odds": safe_offered,
        "opposing_odds": safe_opposing,
        "raw_break_even_probability": (
            market.get("raw_implied_probability", [None])[0]
            if market.get("status") == "available"
            else None
        ),
        "no_vig_break_even_probability": (
            market.get("no_vig_probability", [None])[0]
            if market.get("status") == "available"
            else None
        ),
        "overround": market.get("overround"),
    }
    diagnostic = {
        "model_probability": diagnostic_probability,
        "probability_interval": diagnostic_interval,
        "registered_model_probability": probability,
        "registered_probability_interval": (
            [lower, upper] if lower is not None else None
        ),
        "claim_ceiling": "unregistered_inputs_are_research_diagnostics_only",
    }
    if unique_blockers:
        return {
            "status": "unavailable",
            "decision": NO_AUTHORIZED_BET,
            "blockers": unique_blockers,
            "market": descriptive_market,
            "diagnostic": diagnostic,
            "authorized_probability": None,
            "fair_odds": None,
            "edge_pp": None,
            "expected_return": None,
            "conservative_expected_return": None,
        }

    no_vig = float(market["no_vig_probability"][0])
    edge_pp = 100.0 * (probability - no_vig)
    expected_return = probability * float(offered_odds) - 1.0
    conservative_return = lower * float(offered_odds) - 1.0
    qualifies = (
        edge_pp >= float(authority["minimum_edge_pp"])
        and conservative_return >= float(authority["minimum_expected_return"])
    )
    return {
        "status": "authorized",
        "decision": "BET" if qualifies else "PASS",
        "blockers": [],
        "market": descriptive_market,
        "diagnostic": diagnostic,
        "authorized_probability": probability,
        "fair_odds": 1.0 / probability,
        "edge_pp": edge_pp,
        "expected_return": expected_return,
        "conservative_expected_return": conservative_return,
        "authority_record_id": authority["authority_record_id"],
        "quote_sha256": expected_quote_sha256,
        "quote_registry_sha256": quote_registry_sha256,
        "event_probability_sha256": registered_probability_sha256,
        "event_probability_registry_sha256": (
            registered_probability_registry_sha256
        ),
    }
