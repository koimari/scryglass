from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import phase_two_evaluation_result_registry_v1 as registry


def _receipt(binding: dict, *, passed: bool) -> dict:
    return {
        "schema_version": registry.SCHEMA_VERSION,
        "registry_id": "registry-1",
        "status": "REGISTERED_PASS" if passed else "REGISTERED_TERMINAL_FAILURE",
        "registered_at_utc": "2026-10-03T12:00:00+00:00",
        "reviews": [
            {
                "review_scope": scope,
                "reviewer_id": f"reviewer-{index}",
                "reviewed_at_utc": "2026-10-03T11:00:00+00:00",
                "attestation": registry.REVIEW_ATTESTATION,
            }
            for index, scope in enumerate(
                ("MODEL_RESULT", "MARKET_RESULT"), start=1
            )
        ],
        "result_binding": binding,
        "terminal_decision": {
            "phase_two_evaluation_independently_registered": True,
            "phase_two_market_gates_independently_passed": passed,
            "separate_match_winner_market_authority_may_be_considered": passed,
            "probability_or_betting_authorized": False,
            "failure_is_terminal_no_reopening_reselection_or_cohort_substitution": not passed,
        },
        "authority": dict(registry.AUTHORITY),
        "claim_ceiling": registry.CLAIM_CEILING,
    }


def test_result_registry_is_terminal_and_never_self_authorizes_betting() -> None:
    binding = {
        "result_artifact_sha256": "1" * 64,
        "phase_two_market_gates_passed": True,
        "exact_replay_verified": True,
    }
    receipt = _receipt(binding, passed=True)
    checked = registry.validate_phase_two_evaluation_registry_v1(
        receipt, expected_binding=binding
    )
    assert checked["terminal_decision"][
        "phase_two_market_gates_independently_passed"
    ] is True
    assert checked["terminal_decision"]["probability_or_betting_authorized"] is False

    forged = deepcopy(receipt)
    forged["terminal_decision"]["probability_or_betting_authorized"] = True
    with pytest.raises(
        registry.PhaseTwoEvaluationRegistryError,
        match="terminal decision changed",
    ):
        registry.validate_phase_two_evaluation_registry_v1(
            forged, expected_binding=binding
        )
