from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import side_neutral_protocol_review_v1 as review
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_registry_v2 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_side_neutral_protocol_v2,
)
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_v2 import (
    INDEPENDENT_REVIEW_ENV,
)
from lol_kills.v2.ratings.player.side_neutral_collection_implementation_registry_v1 import (
    validate_registered_side_neutral_collection_implementation,
)


REVIEWED_AT = "2026-08-02T07:40:00+00:00"
AS_OF = datetime(2026, 8, 2, 8, 0, tzinfo=timezone.utc)


def _payload() -> dict:
    protocol = validate_registered_side_neutral_protocol_v2(root=Path(".").resolve())
    admission = validate_registered_side_neutral_collection_implementation(
        root=Path(".").resolve()
    )
    return {
        "schema_version": review.SCHEMA_VERSION,
        "review_id": "independent-side-neutral-review-test",
        "reviewer": {
            "reviewer_id": "independent-human-test-reviewer",
            "reviewer_role": "independent-human-reviewer",
            "independent_from_implementation": True,
            "not_the_protocol_author": True,
            "conflicts_disclosed": True,
        },
        "reviewed_at_utc": REVIEWED_AT,
        "protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "locked_at_utc": REGISTERED_PROTOCOL_LOCKED_AT_UTC,
        },
        "reviewed_source_locks": protocol["source_locks"],
        "reviewed_admission_implementation": admission["records"],
        "findings": {
            "protocol_and_implementation_reviewed": True,
            "source_provenance_and_exact_roster_binding_reviewed": True,
            "side_selection_without_rating_refit_reviewed": True,
            "terminal_draft_and_actual_start_timing_reviewed": True,
            "duplicate_or_ambiguous_side_binding_policy_reviewed": True,
            "outcome_leakage_controls_reviewed": True,
            "no_clobber_persistence_reviewed": True,
            "model_source_boundary_stopping_evaluation_and_uncertainty_unchanged": True,
            "future_outcomes_accessed": False,
            "future_predictions_accessed": False,
            "unresolved_critical_findings": [],
        },
        "authorization": {
            "prospective_collection_authorized": True,
            "effective_at_utc": REVIEWED_AT,
            "captures_before_effective_time_eligible": False,
            "retrospective_backfill_authorized": False,
            "outcome_opening_authorized": False,
            "rating_or_draft_authority_granted": False,
            "probability_odds_ev_or_recommendation_authorized": False,
            "betting_authorized": False,
        },
        "authority": {
            "prospective_collection_authority": True,
            "outcome_opening_authority": False,
            "model_validation_authority": False,
            "player_rating_authority": False,
            "team_rating_authority": False,
            "draft_validation_authority": False,
            "probability_authority": False,
            "odds_authority": False,
            "expected_value_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "Independent authorization for prospective outcome-free collection only, "
            "effective after this review. No retrospective evidence, outcome opening, "
            "rating, Draft, probability, odds, EV, recommendation, or betting authority."
        ),
    }


def test_independent_review_authorizes_only_future_collection() -> None:
    checked = review.validate_side_neutral_protocol_review(
        _payload(), root=Path(".").resolve(), as_of=AS_OF
    )
    assert checked["authorization"]["prospective_collection_authorized"] is True
    assert checked["authorization"]["captures_before_effective_time_eligible"] is False
    assert checked["authorization"]["outcome_opening_authorized"] is False
    assert checked["authority"]["prospective_collection_authority"] is True
    assert checked["authority"]["betting_authority"] is False


def test_independent_review_rejects_contamination_or_overbroad_authority() -> None:
    contaminated = _payload()
    contaminated["findings"]["future_predictions_accessed"] = True
    with pytest.raises(review.SideNeutralProtocolReviewError, match="findings"):
        review.validate_side_neutral_protocol_review(
            contaminated, root=Path(".").resolve(), as_of=AS_OF
        )

    overbroad = _payload()
    overbroad["authority"]["betting_authority"] = True
    with pytest.raises(review.SideNeutralProtocolReviewError, match="authority"):
        review.validate_side_neutral_protocol_review(
            overbroad, root=Path(".").resolve(), as_of=AS_OF
        )


def test_external_loader_requires_matching_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(review.SideNeutralProtocolReviewError, match="missing external"):
        review.load_active_side_neutral_protocol_review(
            root=Path(".").resolve(), environment={}, as_of=AS_OF
        )

    review_path = tmp_path / "review.json"
    raw = (json.dumps(_payload(), indent=2, sort_keys=True) + "\n").encode()
    review_path.write_bytes(raw)
    monkeypatch.setattr(review, "REVIEW_LOCATOR", review_path)
    checked = review.load_active_side_neutral_protocol_review(
        root=Path(".").resolve(),
        environment={INDEPENDENT_REVIEW_ENV: hashlib.sha256(raw).hexdigest()},
        as_of=AS_OF,
    )
    assert checked["review_id"] == "independent-side-neutral-review-test"

    with pytest.raises(review.SideNeutralProtocolReviewError, match="does not match"):
        review.load_active_side_neutral_protocol_review(
            root=Path(".").resolve(),
            environment={INDEPENDENT_REVIEW_ENV: "f" * 64},
            as_of=AS_OF,
        )
