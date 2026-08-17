from __future__ import annotations

import pytest

from lol_kills.export.paired_public_draft_score import (
    build_paired_public_draft_score_result,
)
from lol_kills.research.controlled_draft_contribution import (
    ControlledDraftContributionError,
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


def _draft_rows() -> list[dict[str, str]]:
    rows = []
    for side in ("Blue", "Red"):
        for index, role in enumerate(("top", "jng", "mid", "bot", "sup")):
            rows.append(
                {
                    "game_uid": "game-1",
                    "date": "2026-08-17T12:00:00Z",
                    "side": side,
                    "position": role,
                    "champion": f"{side}-champion-{index}",
                    "playername": f"{side}-player-{index}",
                    "teamname": f"{side}-team",
                    "league": "LCK",
                }
            )
    return rows


def _swap(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    champions = {
        (row["side"], row["position"]): row["champion"] for row in rows
    }
    return [
        {
            **row,
            "champion": champions[
                ("Red" if row["side"] == "Blue" else "Blue", row["position"])
            ],
        }
        for row in rows
    ]


def test_public_result_uses_paired_intervention_for_draft_value() -> None:
    observed = _draft_rows()
    result = build_paired_public_draft_score_result(
        release_id="v2026.08.17.120000",
        model_version="public-draft-score-v1",
        promotion_receipt=_promotion_receipt(),
        evidence_start="2025-01-01T00:00:00Z",
        evidence_end="2026-08-17T00:00:00Z",
        observed_rows=observed,
        swapped_rows=_swap(observed),
        observed_blue_win_probability=0.61,
        swapped_draft_blue_win_probability=0.47,
    )

    assert result["match_win_probability"]["Blue"] == 0.61
    assert result["side_recommendation"] == "Blue"
    assert result["controlled_draft_score"]["stronger_draft"] == "Blue"
    assert result["controlled_draft_score"]["method"] == (
        "role_matched_champion_swap"
    )
    assert len(
        result["controlled_draft_score"]["intervention_receipt_sha256"]
    ) == 64
    assert "betting" not in result
    assert "odds" not in result
    assert "ev" not in result
    assert "stake" not in result


def test_public_result_rejects_non_draft_change() -> None:
    observed = _draft_rows()
    swapped = _swap(observed)
    swapped[0]["teamname"] = "Different team"

    with pytest.raises(ControlledDraftContributionError):
        build_paired_public_draft_score_result(
            release_id="v2026.08.17.120000",
            model_version="public-draft-score-v1",
            promotion_receipt=_promotion_receipt(),
            evidence_start="2025-01-01T00:00:00Z",
            evidence_end="2026-08-17T00:00:00Z",
            observed_rows=observed,
            swapped_rows=swapped,
            observed_blue_win_probability=0.61,
            swapped_draft_blue_win_probability=0.47,
        )
