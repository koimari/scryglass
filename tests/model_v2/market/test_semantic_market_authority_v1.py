from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import semantic_market_authority_v1 as authority


def _receipt(bindings: dict) -> dict:
    return {
        "schema_version": authority.SCHEMA_VERSION,
        "authority_id": "semantic-authority-1",
        "status": "APPROVED",
        "scope": "PRIVATE_MATCH_WINNER_DECISION_SUPPORT_ONLY",
        "issued_at_utc": "2026-10-04T12:00:00+00:00",
        "valid_until_utc": "2026-10-11T12:00:00+00:00",
        "reviews": [
            {
                "review_scope": scope,
                "reviewer_id": f"deployment-reviewer-{index}",
                "reviewed_at_utc": "2026-10-04T11:00:00+00:00",
                "attestation": attestation,
            }
            for index, (scope, attestation) in enumerate(
                authority.REVIEW_SCOPES.items(), start=1
            )
        ],
        "bindings": bindings,
        "decision_policy": {
            "market_type": "match_winner",
            "no_vig_method": "two_way_normalized_implied_probability",
            "minimum_lower_bound_expected_return": 0.02,
            "maximum_probability_age_seconds": 60.0,
            "maximum_quote_age_seconds": 30.0,
            "positive_expected_return_haircut_fraction": 0.01,
            "flat_stake_or_bankroll_advice_permitted": False,
            "transaction_execution_permitted": False,
        },
        "authority": dict(authority.AUTHORITY),
        "claim_ceiling": authority.CLAIM_CEILING,
    }


def test_semantic_authority_requires_new_independent_deployment_reviewers() -> None:
    bindings = {"reviewer_ids_excluded_from_final_authority": ["prior-reviewer"]}
    receipt = _receipt(bindings)
    checked = authority.validate_semantic_market_authority_v1(
        receipt, expected_bindings=bindings
    )
    assert checked["authority"]["private_expected_value_authority"] is True
    assert checked["authority"]["transaction_authority"] is False
    assert checked["authority"]["stake_authority"] is False

    forged = deepcopy(receipt)
    forged["reviews"][0]["reviewer_id"] = "prior-reviewer"
    with pytest.raises(
        authority.SemanticMarketAuthorityError,
        match="not independent",
    ):
        authority.validate_semantic_market_authority_v1(
            forged, expected_bindings=bindings
        )


def test_semantic_authority_rejects_stake_or_transaction_scope() -> None:
    bindings = {"reviewer_ids_excluded_from_final_authority": []}
    forged = _receipt(bindings)
    forged["authority"]["stake_authority"] = True
    with pytest.raises(
        authority.SemanticMarketAuthorityError,
        match="exceeds scope",
    ):
        authority.validate_semantic_market_authority_v1(
            forged, expected_bindings=bindings
        )
