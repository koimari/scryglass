from __future__ import annotations

import pytest

from lol_kills.export.public_draft_score_result import (
    PublicDraftScoreResultError,
    build_public_draft_score_result,
)
from lol_kills.research.selective_draft_probability import canonical_sha256


def _promotion_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": "scryglass:public-draft-score-promotion-receipt:v1",
        "status": "promoted",
        "authority": "promoted",
        "model_version": "public-draft-score-v1",
        "approved_public_fields": [
            "match_win_probability",
            "controlled_draft_score",
            "side_recommendation",
        ],
        "public_probability": True,
        "public_recommendation": True,
        "betting_odds_ev_stake": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def test_public_draft_score_result_separates_probability_and_draft() -> None:
    result = build_public_draft_score_result(
        release_id="v2026.08.16.220000",
        model_version="public-draft-score-v1",
        promotion_receipt=_promotion_receipt(),
        evidence_start="2025-01-01T00:00:00Z",
        evidence_end="2026-08-16T00:00:00Z",
        blue_win_probability=0.61,
        controlled_model_units=-0.18,
        controlled_edge_percentage_points=-1.9,
        controlled_explanation=(
            "Composition contribution with strength controls held fixed."
        ),
    )

    assert result["side_recommendation"] == "Blue"
    assert result["controlled_draft_score"]["stronger_draft"] == "Red"
    assert result["match_win_probability"]["Red"] == pytest.approx(0.39)
    forbidden = {"betting", "odds", "ev", "expected_value", "stake", "wager"}

    def keys(value: object) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(child) for child in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(child) for child in value))
        return set()

    assert not forbidden.intersection(keys(result))


def test_public_draft_score_result_rejects_direction_conflict() -> None:
    with pytest.raises(PublicDraftScoreResultError, match="direction"):
        build_public_draft_score_result(
            release_id="v2026.08.16.220000",
            model_version="public-draft-score-v1",
            promotion_receipt=_promotion_receipt(),
            evidence_start="2025-01-01T00:00:00Z",
            evidence_end="2026-08-16T00:00:00Z",
            blue_win_probability=0.61,
            controlled_model_units=-0.18,
            controlled_edge_percentage_points=1.9,
            controlled_explanation="Controlled composition contribution.",
        )


def test_public_draft_score_result_rejects_nonzero_edge_for_even_score() -> None:
    with pytest.raises(PublicDraftScoreResultError, match="direction"):
        build_public_draft_score_result(
            release_id="v2026.08.16.220000",
            model_version="public-draft-score-v1",
            promotion_receipt=_promotion_receipt(),
            evidence_start="2025-01-01T00:00:00Z",
            evidence_end="2026-08-16T00:00:00Z",
            blue_win_probability=0.61,
            controlled_model_units=0.0,
            controlled_edge_percentage_points=1.9,
            controlled_explanation="Controlled composition contribution.",
        )


def test_public_draft_score_result_rejects_hash_shaped_placeholder() -> None:
    receipt = _promotion_receipt()
    receipt["receipt_sha256"] = "a" * 64

    with pytest.raises(PublicDraftScoreResultError, match="promotion receipt"):
        build_public_draft_score_result(
            release_id="v2026.08.16.220000",
            model_version="public-draft-score-v1",
            promotion_receipt=receipt,
            evidence_start="2025-01-01T00:00:00Z",
            evidence_end="2026-08-16T00:00:00Z",
            blue_win_probability=0.61,
            controlled_model_units=-0.18,
            controlled_edge_percentage_points=-1.9,
            controlled_explanation="Controlled composition contribution.",
        )
