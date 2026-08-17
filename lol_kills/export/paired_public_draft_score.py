"""Build a promoted public Draft result from a verified paired intervention."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from lol_kills.export.public_draft_score_result import (
    build_public_draft_score_result,
)
from lol_kills.research.controlled_draft_contribution import (
    isolate_controlled_draft_contribution,
    validate_role_matched_champion_swap,
)


def build_paired_public_draft_score_result(
    *,
    release_id: str,
    model_version: str,
    promotion_receipt: Mapping[str, Any],
    evidence_start: str,
    evidence_end: str,
    observed_rows: Sequence[Mapping[str, Any]],
    swapped_rows: Sequence[Mapping[str, Any]],
    observed_blue_win_probability: float,
    swapped_draft_blue_win_probability: float,
) -> dict[str, Any]:
    """Return a public result whose Draft value comes from one exact swap."""

    intervention = validate_role_matched_champion_swap(
        observed_rows=observed_rows,
        swapped_rows=swapped_rows,
    )
    contribution = isolate_controlled_draft_contribution(
        observed_blue_win_probability=observed_blue_win_probability,
        swapped_draft_blue_win_probability=swapped_draft_blue_win_probability,
    )
    result = build_public_draft_score_result(
        release_id=release_id,
        model_version=model_version,
        promotion_receipt=promotion_receipt,
        evidence_start=evidence_start,
        evidence_end=evidence_end,
        blue_win_probability=observed_blue_win_probability,
        controlled_model_units=contribution["model_units"],
        controlled_edge_percentage_points=contribution["edge_percentage_points"],
        controlled_explanation=(
            "Role-matched champion swap with team strength, player strength, "
            "momentum, league, match context, and side held fixed."
        ),
    )
    result["controlled_draft_score"].update(
        {
            "method": contribution["method"],
            "intervention_receipt_sha256": intervention["receipt_sha256"],
            "isolated_blue_draft_probability": contribution[
                "isolated_blue_draft_probability"
            ],
            "fixed_strength_blue_win_probability": contribution[
                "fixed_strength_blue_win_probability"
            ],
        }
    )
    return result
