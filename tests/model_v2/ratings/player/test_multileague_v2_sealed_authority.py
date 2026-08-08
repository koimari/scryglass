from __future__ import annotations

import copy
import hashlib
import json

import pytest

from lol_kills.v2.ratings.player import multileague_v2_sealed_authority as authority


def receipt(bindings: dict) -> dict:
    return {
        "schema_version": authority.SCHEMA_VERSION,
        "authority_id": "independent-review:fixture",
        "status": "APPROVED",
        "scope": "ONE_TIME_SEALED_FINAL_EVALUATION_ONLY",
        "reviewer_id": "independent-reviewer:fixture",
        "reviewed_at": "2026-08-02T00:00:00Z",
        "independence_attestation": {
            "reviewer_not_model_author_or_candidate_selector": True,
            "review_used_only_pinned_adaptive_evidence": True,
            "sealed_final_outcomes_not_accessed_before_approval": True,
            "approval_was_not_generated_by_the_evaluated_system": True,
        },
        "bindings": bindings,
        "one_time_run": {
            "run_id": "sealed-final-fixture-1",
            "authorized_output_locator": authority.OUTPUT_LOCATOR,
            "no_clobber_required": True,
            "second_holdout_opening_prohibited": True,
        },
        "claim_ceiling": {
            "sealed_evaluation_authorized": True,
            "production_rating_authorized": False,
            "match_probability_authorized": False,
            "fair_odds_authorized": False,
            "expected_value_authorized": False,
            "bet_recommendation_authorized": False,
        },
    }


@pytest.fixture(scope="module")
def bindings() -> dict:
    return authority.current_expected_bindings(".")


def test_exact_independent_receipt_authorizes_only_one_holdout_opening(
    bindings: dict,
) -> None:
    value = authority.validate_sealed_opening_authority(
        receipt(bindings),
        expected_bindings=bindings,
    )
    assert value["claim_ceiling"]["sealed_evaluation_authorized"] is True
    assert value["claim_ceiling"]["production_rating_authorized"] is False
    assert value["claim_ceiling"]["match_probability_authorized"] is False
    assert value["one_time_run"]["second_holdout_opening_prohibited"] is True


def test_binding_or_independence_tamper_fails_closed(bindings: dict) -> None:
    binding_tamper = receipt(copy.deepcopy(bindings))
    binding_tamper["bindings"]["selected_candidate_id"] = "other"
    with pytest.raises(authority.SealedOpeningAuthorityError, match="bindings"):
        authority.validate_sealed_opening_authority(
            binding_tamper,
            expected_bindings=bindings,
        )

    independence_tamper = receipt(bindings)
    independence_tamper["independence_attestation"][
        "reviewer_not_model_author_or_candidate_selector"
    ] = False
    with pytest.raises(authority.SealedOpeningAuthorityError, match="independence"):
        authority.validate_sealed_opening_authority(
            independence_tamper,
            expected_bindings=bindings,
        )


def test_external_pin_is_required_and_raw_byte_exact(tmp_path, bindings: dict) -> None:
    path = tmp_path / "receipt.json"
    raw = (json.dumps(receipt(bindings), sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()

    registered = authority.load_pinned_sealed_opening_authority(
        path,
        expected_bindings=bindings,
        external_sha256=digest,
    )
    assert registered["status"] == "registered"
    assert registered["sealed_evaluation_authorized"] is True
    assert registered["match_probability_authorized"] is False
    assert registered["betting_decision_authorized"] is False

    with pytest.raises(authority.SealedOpeningAuthorityError, match="external pin"):
        authority.load_pinned_sealed_opening_authority(
            path,
            expected_bindings=bindings,
            external_sha256="0" * 64,
        )


def test_repository_has_no_self_authorizing_receipt() -> None:
    status = authority.inspect_sealed_opening_authority(".", environment={})
    assert status["status"] == "unavailable"
    assert status["external_digest_pin_present"] is False
    assert status["sealed_evaluation_authorized"] is False
    assert status["betting_decision_authorized"] is False
