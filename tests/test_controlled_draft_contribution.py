from __future__ import annotations

import math

import pytest

from lol_kills.research.controlled_draft_contribution import (
    ControlledDraftContributionError,
    isolate_controlled_draft_contribution,
)


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def test_isolation_reconstructs_both_paired_predictions() -> None:
    result = isolate_controlled_draft_contribution(
        observed_blue_win_probability=0.62,
        swapped_draft_blue_win_probability=0.48,
    )

    assert result["stronger_draft"] == "Blue"
    assert result["model_units"] > 0.0
    assert result["edge_percentage_points"] > 0.0
    assert result["reconstructed_observed_blue_win_probability"] == pytest.approx(
        0.62
    )
    assert result["reconstructed_swapped_blue_win_probability"] == pytest.approx(
        0.48
    )


def test_shared_strength_shift_cannot_change_draft_contribution() -> None:
    base = isolate_controlled_draft_contribution(
        observed_blue_win_probability=_sigmoid(0.30 + 0.16),
        swapped_draft_blue_win_probability=_sigmoid(0.30 - 0.16),
    )
    shifted = isolate_controlled_draft_contribution(
        observed_blue_win_probability=_sigmoid(1.10 + 0.16),
        swapped_draft_blue_win_probability=_sigmoid(1.10 - 0.16),
    )

    assert shifted["model_units"] == pytest.approx(base["model_units"])
    assert shifted["edge_percentage_points"] == pytest.approx(
        base["edge_percentage_points"]
    )
    assert shifted["fixed_strength_blue_win_probability"] > base[
        "fixed_strength_blue_win_probability"
    ]


def test_reversing_the_pair_reverses_only_the_draft_direction() -> None:
    blue = isolate_controlled_draft_contribution(
        observed_blue_win_probability=0.61,
        swapped_draft_blue_win_probability=0.44,
    )
    red = isolate_controlled_draft_contribution(
        observed_blue_win_probability=0.44,
        swapped_draft_blue_win_probability=0.61,
    )

    assert red["model_units"] == pytest.approx(-blue["model_units"])
    assert red["edge_percentage_points"] == pytest.approx(
        -blue["edge_percentage_points"]
    )
    assert blue["stronger_draft"] == "Blue"
    assert red["stronger_draft"] == "Red"
    assert red["fixed_strength_blue_win_probability"] == pytest.approx(
        blue["fixed_strength_blue_win_probability"]
    )


def test_equal_pair_is_an_even_draft() -> None:
    result = isolate_controlled_draft_contribution(
        observed_blue_win_probability=0.57,
        swapped_draft_blue_win_probability=0.57,
    )

    assert result["model_units"] == 0.0
    assert result["edge_percentage_points"] == 0.0
    assert result["stronger_draft"] == "Even"
    assert result["isolated_blue_draft_probability"] == 0.5


@pytest.mark.parametrize("value", [0.0, 1.0, math.nan, math.inf])
def test_invalid_probability_fails_closed(value: float) -> None:
    with pytest.raises(ControlledDraftContributionError):
        isolate_controlled_draft_contribution(
            observed_blue_win_probability=value,
            swapped_draft_blue_win_probability=0.5,
        )
