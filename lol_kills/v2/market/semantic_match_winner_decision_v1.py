"""Semantic private BET/PASS evaluation from live production receipts."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import os
from pathlib import Path
from typing import Any, Mapping

from . import phase_one_evaluation_v1 as evaluation
from . import production_betano_quote_v1 as production_quote


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/semantic_match_winner_decision_v1.py"
RESULT_SCHEMA_VERSION = "scryglass:semantic-match-winner-decision:v1"
NO_AUTHORIZED_BET = "NO_AUTHORIZED_BET"


class SemanticMatchWinnerDecisionError(RuntimeError):
    """The live authority, probability, quote, freshness, or policy failed."""


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SemanticMatchWinnerDecisionError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SemanticMatchWinnerDecisionError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock(as_of: datetime | None) -> datetime:
    observed = as_of or datetime.now(timezone.utc)
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise SemanticMatchWinnerDecisionError("decision clock must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _unavailable(*blockers: str) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "unavailable",
        "decision": NO_AUTHORIZED_BET,
        "blockers": sorted(set(blockers)),
        "selection": None,
        "evaluated_selection": None,
        "candidate_evaluations": None,
        "authorized_probability": None,
        "probability_interval": None,
        "fair_decimal_odds": None,
        "offered_decimal_odds": None,
        "no_vig_market_probability": None,
        "expected_return": None,
        "lower_bound_expected_return_after_haircut": None,
        "stake": None,
        "transaction_authorized": False,
    }


def _active_authority(
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
        raise SemanticMatchWinnerDecisionError(
            "semantic market authority is unavailable"
        ) from exc


def _quote(
    *, locator_value: str, root: Path, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, production_quote.OUTPUT_PREFIX, "production_quote_locator"
    )
    raw = evaluation._read_regular(root, locator, "production quote")
    try:
        checked = production_quote.validate_production_betano_quote_v1(
            evaluation._strict_object(raw, "production quote"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise SemanticMatchWinnerDecisionError("production quote is invalid") from exc
    return locator, raw, checked


def _price(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SemanticMatchWinnerDecisionError(f"{field} is not decimal odds")
    result = float(value)
    if not math.isfinite(result) or result <= 1.0:
        raise SemanticMatchWinnerDecisionError(f"{field} is not decimal odds")
    return result


def _haircut_return(probability: float, odds: float, haircut: float) -> float:
    return probability * (odds - 1.0) * (1.0 - haircut) - (1.0 - probability)


def evaluate_semantic_match_winner_v1(
    *, production_quote_locator: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    observed = _clock(as_of)
    try:
        active = _active_authority(
            root=root, environment=environment, as_of=observed
        )
        locator, raw, quote = _quote(
            locator_value=production_quote_locator,
            root=root,
            environment=environment,
        )
    except SemanticMatchWinnerDecisionError as exc:
        return _unavailable(str(exc))
    authority = active["receipt"]
    policy = authority["decision_policy"]
    probability = quote["probability"]
    event = probability["event"]
    if (
        quote["semantic_market_authority_binding"]["authority_raw_sha256"]
        != active["receipt_raw_sha256"]
        or probability["semantic_market_authority_binding"]["authority_raw_sha256"]
        != active["receipt_raw_sha256"]
    ):
        return _unavailable("live_receipts_and_authority_differ")
    probability_time = _timestamp(
        probability["captured_at_utc"], "probability.captured_at"
    )
    quote_time = _timestamp(
        quote["response_received_at_utc"], "quote.response_received_at"
    )
    event_start = _timestamp(event["scheduled_event_start_utc"], "event.start")
    probability_age = (observed - probability_time).total_seconds()
    quote_age = (observed - quote_time).total_seconds()
    blockers: list[str] = []
    if probability_age < 0:
        blockers.append("production_probability_from_future")
    elif probability_age > float(policy["maximum_probability_age_seconds"]):
        blockers.append("production_probability_stale")
    if quote_age < 0:
        blockers.append("production_quote_from_future")
    elif quote_age > float(policy["maximum_quote_age_seconds"]):
        blockers.append("production_quote_stale")
    if observed >= event_start:
        blockers.append("scheduled_event_start_reached")
    if quote_time < probability_time:
        blockers.append("production_quote_predates_probability")
    if blockers:
        return _unavailable(*blockers)
    selection = event["selection"]
    opposing = event["opposing_selection"]
    try:
        blue_odds = _price(quote["prices"].get(selection), "selection price")
        red_odds = _price(quote["prices"].get(opposing), "opposing price")
    except SemanticMatchWinnerDecisionError as exc:
        return _unavailable(str(exc))
    blue_probability = float(probability["probability"])
    lower, upper = [float(item) for item in probability["probability_interval"]]
    red_probability = 1.0 - blue_probability
    red_lower = 1.0 - upper
    implied = [1.0 / blue_odds, 1.0 / red_odds]
    implied_total = sum(implied)
    no_vig = [item / implied_total for item in implied]
    haircut = float(policy["positive_expected_return_haircut_fraction"])
    minimum = float(policy["minimum_lower_bound_expected_return"])
    candidates = [
        {
            "selection": selection,
            "probability": blue_probability,
            "lower": lower,
            "odds": blue_odds,
            "no_vig": no_vig[0],
        },
        {
            "selection": opposing,
            "probability": red_probability,
            "lower": red_lower,
            "odds": red_odds,
            "no_vig": no_vig[1],
        },
    ]
    for candidate in candidates:
        candidate["probability_interval"] = (
            [lower, upper]
            if candidate["selection"] == selection
            else [1.0 - upper, 1.0 - lower]
        )
        candidate["fair_odds"] = 1.0 / candidate["probability"]
        candidate["model_minus_no_vig_probability"] = (
            candidate["probability"] - candidate["no_vig"]
        )
        candidate["expected_return"] = (
            candidate["probability"] * candidate["odds"] - 1.0
        )
        candidate["lower_return"] = _haircut_return(
            candidate["lower"], candidate["odds"], haircut
        )
        candidate["qualifies"] = candidate["lower_return"] >= minimum
    qualifying = [item for item in candidates if item["qualifies"]]
    if len(qualifying) > 1:
        return _unavailable("both_sides_qualify_inconsistent")
    selected = qualifying[0] if qualifying else max(
        candidates, key=lambda item: item["lower_return"]
    )
    candidate_evaluations = [
        {
            "selection": candidate["selection"],
            "authorized_probability": candidate["probability"],
            "probability_interval": candidate["probability_interval"],
            "fair_decimal_odds": candidate["fair_odds"],
            "offered_decimal_odds": candidate["odds"],
            "no_vig_market_probability": candidate["no_vig"],
            "model_minus_no_vig_probability": candidate[
                "model_minus_no_vig_probability"
            ],
            "expected_return": candidate["expected_return"],
            "lower_bound_expected_return_after_haircut": candidate[
                "lower_return"
            ],
            "qualifies": candidate["qualifies"],
        }
        for candidate in candidates
    ]
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "authorized",
        "decision": "BET" if qualifying else "PASS",
        "blockers": [],
        "event_id": event["event_id"],
        "series_id": event["series_id"],
        "game_number": event["game_number"],
        "league": event["league"],
        "selection": selected["selection"] if qualifying else None,
        "evaluated_selection": selected["selection"],
        "candidate_evaluations": candidate_evaluations,
        "authorized_probability": selected["probability"],
        "probability_interval": selected["probability_interval"],
        "fair_decimal_odds": selected["fair_odds"],
        "offered_decimal_odds": selected["odds"],
        "no_vig_market_probability": selected["no_vig"],
        "expected_return": selected["expected_return"],
        "lower_bound_expected_return_after_haircut": selected["lower_return"],
        "minimum_required_lower_bound_return": minimum,
        "probability_age_seconds": probability_age,
        "quote_age_seconds": quote_age,
        "production_quote_locator": locator,
        "production_quote_raw_sha256": evaluation._sha256_bytes(raw),
        "semantic_authority_id": authority["authority_id"],
        "semantic_authority_raw_sha256": active["receipt_raw_sha256"],
        "stake": None,
        "transaction_authorized": False,
        "claim_ceiling": authority["claim_ceiling"],
    }


__all__ = [
    "NO_AUTHORIZED_BET", "RESULT_SCHEMA_VERSION", "SOURCE_LOCATOR",
    "SemanticMatchWinnerDecisionError", "evaluate_semantic_match_winner_v1",
]
