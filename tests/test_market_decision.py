from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib

from lol_kills import bookmaker_quote_capture as capture
from lol_kills import market_decision as decision
from lol_kills.v2.market import event_probability_v1 as event_probability
from lol_kills.v2.market.match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
    REGISTERED_SETTLEMENT_CONTRACT_SHA256,
)
from lol_kills.v2.market.match_winner_future_protocol_v1 import BOOKMAKER_ID


NOW = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)
MODEL_SHA = "a" * 64
CALIBRATION_SHA = "b" * 64
UNCERTAINTY_SHA = "c" * 64
GENERATOR_SHA = "d" * 64
QUOTE_REGISTRY_SHA = "e" * 64
SETTLEMENT_RULE_ID = "betano-br-map-winner-shadow-v1"


def probability_receipt(
    *, probability: float = 0.60, interval: tuple[float, float] = (0.57, 0.63)
) -> dict:
    return event_probability.build_event_probability_receipt(
        event_id="series-1-map-1",
        league="LCS",
        market_type="match_winner",
        selection="winner:blue-team",
        opposing_selection="winner:red-team",
        model_artifact_sha256=MODEL_SHA,
        market_protocol_artifact_sha256=REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        calibration_artifact_sha256=CALIBRATION_SHA,
        uncertainty_artifact_sha256=UNCERTAINTY_SHA,
        source_prediction_receipt_sha256="f" * 64,
        source_prediction_registry_sha256="1" * 64,
        generation_code_sha256=GENERATOR_SHA,
        raw_model_probability=probability,
        calibration_intercept=0.0,
        calibration_slope=1.0,
        probability_interval=interval,
        uncertainty_draws_sha256="2" * 64,
        uncertainty_resamples=2000,
        clock=lambda: NOW - timedelta(seconds=2),
    )


def probability_registry(value: dict) -> dict:
    return event_probability.build_event_probability_registry(
        receipts=[
            (
                "data/lol/v2/evaluation/match-winner-market-v1/event-probabilities/series-1-map-1-blue.json",
                value,
            )
        ],
        registry_id="event-probability-registry-1",
        independent_reviewer_id="reviewer-1",
        issued_at=(NOW - timedelta(seconds=1)).isoformat(),
        model_artifact_sha256=MODEL_SHA,
        market_protocol_artifact_sha256=REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        calibration_artifact_sha256=CALIBRATION_SHA,
        uncertainty_artifact_sha256=UNCERTAINTY_SHA,
        generation_code_sha256=GENERATOR_SHA,
    )


def authority_receipt(event_registry_sha256: str) -> dict:
    return {
        "schema_version": decision.SCHEMA_VERSION,
        "status": "approved",
        "scope": "private_personal_decision_support",
        "public_or_transactional_use": False,
        "betting_decision_authorized": True,
        "authority_record_id": "independent-market-review-1",
        "independent_reviewer_id": "reviewer-1",
        "issued_at": (NOW - timedelta(days=1)).isoformat(),
        "valid_until": (NOW + timedelta(days=1)).isoformat(),
        "model_artifact_sha256": MODEL_SHA,
        "bookmaker_id": BOOKMAKER_ID,
        "settlement_rule_id": SETTLEMENT_RULE_ID,
        "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "phase_two_evaluation_artifact_sha256": "3" * 64,
        "historical_market_benchmark_sha256": "4" * 64,
        "bookmaker_terms_artifact_sha256": "5" * 64,
        "settlement_rules_sha256": REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        "capture_protocol_sha256": REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        "reliability_artifact_sha256": "6" * 64,
        "calibration_artifact_sha256": CALIBRATION_SHA,
        "uncertainty_artifact_sha256": UNCERTAINTY_SHA,
        "event_probability_registry_sha256": event_registry_sha256,
        "event_probability_generation_code_sha256": GENERATOR_SHA,
        "quote_registry_sha256": QUOTE_REGISTRY_SHA,
        "leagues": ["LCS"],
        "market_types": ["match_winner"],
        "no_vig_method": "two_way_normalized_implied_probability",
        "out_of_sample_market_comparison": "passed",
        "probability_calibration": "passed",
        "shadow_policy_evaluation": "passed",
        "prospective_latency": "passed",
        "dependence_aware_uncertainty": "passed",
        "quote_coverage": "passed",
        "settlement_review": "passed",
        "minimum_edge_pp": 2.0,
        "minimum_expected_return": 0.02,
    }


def authority(event_registry_sha256: str) -> dict:
    value = authority_receipt(event_registry_sha256)
    expected = hashlib.sha256(decision._canonical_bytes(value)).hexdigest()
    return decision.validate_authority_receipt(
        value,
        expected_sha256=expected,
        league="LCS",
        market_type="match_winner",
        model_artifact_sha256=MODEL_SHA,
        as_of=NOW,
    )


def quote() -> dict:
    source_payload = b'{"bookmaker":"fixture","event":"series-1-map-1"}'
    extraction = capture.build_price_extraction_payload(
        raw_source_payload=source_payload,
        event_id="series-1-map-1",
        market_type="match_winner",
        settlement_rule_id=SETTLEMENT_RULE_ID,
        prices={"winner:blue-team": 1.93, "winner:red-team": 1.80},
        capture_protocol_sha256=REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        settlement_rules_sha256=REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        extractor_id="synthetic-deterministic-extractor-v1",
        extractor_sha256="7" * 64,
    )
    return capture.build_quote_receipt(
        raw_source_payload=source_payload,
        extraction_payload_raw=capture.canonical_bytes(extraction),
        source="bookmaker-capture",
        source_url="https://example.invalid/series-1-map-1",
        source_record_id="quote-1",
        capture_protocol_sha256=REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        settlement_rules_sha256=REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        clock=lambda: NOW,
    )


def evaluate(**overrides: object) -> dict:
    probability_value = overrides.pop("probability_receipt", probability_receipt())
    if "probability_registry" in overrides:
        registry_value = overrides.pop("probability_registry")
    else:
        registry_value = (
            probability_registry(probability_value)
            if probability_value is not None
            else None
        )
    quote_value = overrides.pop("quote", quote())
    event_registry_sha = (
        event_probability.sha256_json(registry_value)
        if registry_value is not None
        else "8" * 64
    )
    if "authority" in overrides:
        authority_value = overrides.pop("authority")
    else:
        authority_value = authority(event_registry_sha)
    arguments = {
        # These remain visible as unregistered diagnostics but are never used for EV.
        "model_probability": 0.01,
        "probability_interval": (0.0, 0.02),
        "probability_receipt": probability_value,
        "expected_probability_sha256": (
            event_probability.sha256_json(probability_value)
            if probability_value is not None
            else None
        ),
        "probability_registry": registry_value,
        "expected_probability_registry_sha256": (
            event_registry_sha if registry_value is not None else None
        ),
        "offered_odds": 1.93,
        "opposing_odds": 1.80,
        "quote": quote_value,
        "expected_quote_sha256": (
            hashlib.sha256(decision._canonical_bytes(quote_value)).hexdigest()
            if quote_value is not None
            else None
        ),
        "quote_registry_sha256": QUOTE_REGISTRY_SHA,
        "authority": authority_value,
        "expected_authority_sha256": authority_value.get("authority_sha256"),
        "as_of": NOW,
        "selection": "winner:blue-team",
        "opposing_selection": "winner:red-team",
        "event_id": "series-1-map-1",
        "market_type": "match_winner",
        "settlement_rule_id": SETTLEMENT_RULE_ID,
    }
    arguments.update(overrides)
    return decision.evaluate_two_way_market(**arguments)


def test_two_way_no_vig_requires_both_prices() -> None:
    missing = decision.two_way_no_vig((1.80, None))
    assert missing["status"] == "unavailable"
    complete = decision.two_way_no_vig((1.80, 1.93))
    assert complete["status"] == "available"
    assert sum(complete["no_vig_probability"]) == 1.0
    assert complete["overround"] > 0
    assert decision.two_way_no_vig(("bad", 1.93))["status"] == "unavailable"


def test_unregistered_authority_cannot_self_authorize() -> None:
    value = probability_receipt()
    registry_value = probability_registry(value)
    receipt_value = authority_receipt(event_probability.sha256_json(registry_value))
    result = decision.validate_authority_receipt(
        receipt_value,
        expected_sha256=None,
        league="LCS",
        market_type="match_winner",
        model_artifact_sha256=MODEL_SHA,
        as_of=NOW,
    )
    assert result["status"] == "unavailable"
    assert "independent_market_authority_not_registered" in result["blockers"]


def test_malformed_authority_lists_fail_closed_without_throwing() -> None:
    value = probability_receipt()
    registry_value = probability_registry(value)
    receipt_value = authority_receipt(event_probability.sha256_json(registry_value))
    receipt_value["leagues"] = 7
    expected = hashlib.sha256(decision._canonical_bytes(receipt_value)).hexdigest()

    result = decision.validate_authority_receipt(
        receipt_value,
        expected_sha256=expected,
        league="LCS",
        market_type="match_winner",
        model_artifact_sha256=MODEL_SHA,
        as_of=NOW,
    )
    assert result["status"] == "unavailable"
    assert "market_authority_leagues_invalid" in result["blockers"]


def test_raw_probability_and_interval_can_never_authorize_ev() -> None:
    value = probability_receipt()
    registry_value = probability_registry(value)
    result = evaluate(
        probability_receipt=None,
        probability_registry=None,
        expected_probability_sha256=None,
        expected_probability_registry_sha256=None,
        authority=authority(event_probability.sha256_json(registry_value)),
    )
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert result["authorized_probability"] is None
    assert result["fair_odds"] is None
    assert result["expected_return"] is None
    assert result["diagnostic"]["model_probability"] == 0.01
    assert "event_probability_receipt_missing" in result["blockers"]
    assert "event_probability_registry_missing" in result["blockers"]


def test_malformed_raw_diagnostic_interval_is_ignored_not_authorized() -> None:
    result = evaluate(
        probability_receipt=None,
        probability_registry=None,
        expected_probability_sha256=None,
        expected_probability_registry_sha256=None,
        probability_interval=7,
    )
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert result["diagnostic"]["probability_interval"] is None


def test_probability_receipt_cannot_self_authorize_without_registry_digest() -> None:
    result = evaluate(expected_probability_registry_sha256=None)
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert "independent_event_probability_registry_not_registered" in result["blockers"]


def test_legacy_v1_authority_cannot_reach_external_digest_binding_lane() -> None:
    result = evaluate(expected_authority_sha256=None)
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert (
        "legacy_v1_match_winner_decision_path_non_authorizing_use_semantic_v2"
        in result["blockers"]
    )


def test_quote_cannot_self_authorize_without_an_independent_digest() -> None:
    result = evaluate(expected_quote_sha256=None)
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert "independent_market_quote_not_registered" in result["blockers"]


def test_stale_quote_fails_closed() -> None:
    stale = quote()
    stale["captured_at_utc"] = (NOW - timedelta(minutes=2)).isoformat()
    stale["clock_attestation"]["observed_wall_clock_utc"] = stale[
        "captured_at_utc"
    ]
    result = evaluate(quote=stale)
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert "market_quote_stale" in result["blockers"]


def test_legacy_v1_match_winner_receipts_cannot_authorize_ev() -> None:
    result = evaluate()
    assert result["status"] == "unavailable"
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert result["authorized_probability"] is None
    assert result["diagnostic"]["model_probability"] == 0.01
    assert result["fair_odds"] is None
    assert result["expected_return"] is None
    assert (
        "legacy_v1_match_winner_decision_path_non_authorizing_use_semantic_v2"
        in result["blockers"]
    )


def test_legacy_v1_lower_bound_cannot_bypass_semantic_v2_requirement() -> None:
    value = probability_receipt(interval=(0.50, 0.66))
    registry_value = probability_registry(value)
    result = evaluate(
        probability_receipt=value,
        probability_registry=registry_value,
        expected_probability_sha256=event_probability.sha256_json(value),
        expected_probability_registry_sha256=event_probability.sha256_json(
            registry_value
        ),
        authority=authority(event_probability.sha256_json(registry_value)),
    )
    assert result["status"] == "unavailable"
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert (
        "legacy_v1_match_winner_decision_path_non_authorizing_use_semantic_v2"
        in result["blockers"]
    )


def test_legacy_v1_probability_registry_cannot_authorize() -> None:
    result = evaluate(
        expected_probability_registry_sha256="9" * 64,
    )
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert (
        "legacy_v1_match_winner_decision_path_non_authorizing_use_semantic_v2"
        in result["blockers"]
    )


def test_quote_must_bind_the_exact_selection_and_prices() -> None:
    mismatched = quote()
    mismatched["prices"] = {
        "winner:blue-team": 2.01,
        "winner:red-team": 1.80,
    }
    result = evaluate(quote=mismatched)
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert "market_quote_price_binding_mismatch:winner:blue-team" in result[
        "blockers"
    ]


def test_quote_digest_detects_post_registration_tampering() -> None:
    original = quote()
    expected = hashlib.sha256(decision._canonical_bytes(original)).hexdigest()
    original["captured_at_utc"] = (NOW - timedelta(seconds=1)).isoformat()
    original["clock_attestation"]["observed_wall_clock_utc"] = original[
        "captured_at_utc"
    ]
    result = evaluate(quote=original, expected_quote_sha256=expected)
    assert result["decision"] == decision.NO_AUTHORIZED_BET
    assert "market_quote_digest_mismatch" in result["blockers"]


def test_total_kills_cannot_borrow_match_winner_authority() -> None:
    value = probability_receipt()
    registry_value = probability_registry(value)
    authority_value = authority_receipt(event_probability.sha256_json(registry_value))
    authority_value["market_types"] = ["total_kills"]
    expected = hashlib.sha256(decision._canonical_bytes(authority_value)).hexdigest()

    result = decision.validate_authority_receipt(
        authority_value,
        expected_sha256=expected,
        league="LCS",
        market_type="total_kills",
        model_artifact_sha256=MODEL_SHA,
        as_of=NOW,
    )
    assert result["status"] == "unavailable"
    assert "registered_market_protocol_unavailable:total_kills" in result["blockers"]


def test_match_winner_authority_cannot_change_settlement_rule() -> None:
    value = probability_receipt()
    registry_value = probability_registry(value)
    authority_value = authority_receipt(event_probability.sha256_json(registry_value))
    authority_value["settlement_rule_id"] = "some-other-rule"
    expected = hashlib.sha256(decision._canonical_bytes(authority_value)).hexdigest()

    result = decision.validate_authority_receipt(
        authority_value,
        expected_sha256=expected,
        league="LCS",
        market_type="match_winner",
        model_artifact_sha256=MODEL_SHA,
        as_of=NOW,
    )
    assert result["status"] == "unavailable"
    assert "market_authority_settlement_rule_binding_mismatch" in result["blockers"]
