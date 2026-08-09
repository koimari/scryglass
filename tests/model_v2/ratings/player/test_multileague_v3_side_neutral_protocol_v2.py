from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import (
    multileague_v3_side_neutral_protocol_v2 as protocol,
)
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_registry_v2 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_side_neutral_protocol_v2,
)


TEST_CLOCK = datetime(2026, 8, 2, 7, 31, tzinfo=timezone.utc)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = protocol._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


@pytest.fixture(scope="module")
def candidate() -> dict:
    return protocol.build_side_neutral_protocol_v2(
        root=Path(".").resolve(), clock=lambda: TEST_CLOCK
    )


def test_complete_candidate_freezes_all_four_capture_stages(candidate: dict) -> None:
    assert candidate["result_state"] == protocol.RESULT_STATE
    assert candidate["locked_empty_state"]["complete_bundles"] == 0
    assert candidate["capture_contract"]["order"] == [
        "pre_side_envelope",
        "public_side_binding",
        "terminal_draft",
        "authoritative_actual_map_start",
        "complete_joint_bundle",
        "independent_ledger_admission",
    ]
    assert candidate["capture_contract"][
        "public_side_source_must_select_one_existing_conditional"
    ] is True
    assert candidate["capture_contract"][
        "rating_refit_after_side_observation_permitted"
    ] is False
    assert candidate["capture_contract"][
        "ambiguous_or_duplicate_side_bindings_invalidate_map"
    ] is True
    assert candidate["registration"]["independent_review_present"] is False
    assert candidate["registration"]["prospective_collection_authorized"] is False
    assert all(value is False for value in candidate["authority"].values())


def test_complete_candidate_preserves_model_and_evaluation(candidate: dict) -> None:
    for field in (
        "candidate_changed",
        "source_snapshot_changed",
        "future_boundary_changed",
        "support_stopping_rule_changed",
        "evaluation_rule_changed",
        "comparators_changed",
        "uncertainty_rule_changed",
        "opening_rule_changed",
        "future_outcomes_used",
        "future_predictions_used",
    ):
        assert candidate["supersession"][field] is False


def test_complete_candidate_requires_external_pre_capture_review(candidate: dict) -> None:
    review = candidate["independent_review_contract"]
    assert review["external_digest_environment_variable"] == (
        "SCRYGLASS_PRIVATE_SIDE_NEUTRAL_PROTOCOL_REVIEW_SHA256"
    )
    assert review["review_must_precede_first_eligible_pre_side_capture"] is True
    assert review["self_review_permitted"] is False
    assert review["review_may_authorize_outcome_opening"] is False
    assert review["review_may_authorize_ratings_probabilities_or_betting"] is False


def test_complete_candidate_rejects_contamination_or_authority(candidate: dict) -> None:
    contaminated = deepcopy(candidate)
    contaminated["supersession"]["future_predictions_used"] = True
    _resign(contaminated)
    with pytest.raises(protocol.SideNeutralProtocolV2Error, match="scope changed"):
        protocol.validate_side_neutral_protocol_v2(
            contaminated, root=Path(".").resolve()
        )

    forged = deepcopy(candidate)
    forged["authority"]["betting_authority"] = True
    _resign(forged)
    with pytest.raises(protocol.SideNeutralProtocolV2Error, match="exceeds authority"):
        protocol.validate_side_neutral_protocol_v2(
            forged, root=Path(".").resolve()
        )


def test_registered_complete_candidate_is_hash_pinned() -> None:
    checked = validate_registered_side_neutral_protocol_v2(root=Path(".").resolve())
    assert checked["artifact_sha256"] == REGISTERED_PROTOCOL_ARTIFACT_SHA256
    assert len(REGISTERED_PROTOCOL_RAW_SHA256) == 64
    assert checked["registration"]["independent_review_present"] is False
