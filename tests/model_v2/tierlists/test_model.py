"""Estimand-math tests for L9 Tier Value and counterability."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lol_kills.v2.tierlists.model import (
    calibrated_probability,
    reference_mixture_logit,
    response_regret,
    standardized_replacement_probability_points,
    _type7_quantile,
)


def test_iv_is_calibrated_probability_point_difference_not_raw_logit() -> None:
    slope = 0.8
    intercept = 0.0
    champion_logit = 0.05
    reference_logit = 0.0
    iv = standardized_replacement_probability_points(
        champion_logit, reference_logit, calibration_slope=slope, calibration_intercept=intercept
    )
    expected = 100.0 * (
        calibrated_probability(champion_logit, calibration_slope=slope, calibration_intercept=intercept)
        - calibrated_probability(reference_logit, calibration_slope=slope, calibration_intercept=intercept)
    )
    assert iv == pytest.approx(expected, abs=1e-12)
    # it is NOT the raw logit difference times 100
    assert iv != pytest.approx(100.0 * (champion_logit - reference_logit), abs=1e-6)
    # and it is NOT a raw win rate (which would be a single probability, not a delta)
    assert 0.0 < iv < 100.0


def test_iv_monotone_and_zero_at_reference() -> None:
    slope, intercept = 0.8, 0.0
    base = standardized_replacement_probability_points(0.01, 0.01, calibration_slope=slope, calibration_intercept=intercept)
    assert base == pytest.approx(0.0, abs=1e-12)
    higher = standardized_replacement_probability_points(0.2, 0.01, calibration_slope=slope, calibration_intercept=intercept)
    lower = standardized_replacement_probability_points(-0.2, 0.01, calibration_slope=slope, calibration_intercept=intercept)
    assert higher > 0.0 > lower
    # complement symmetry of the calibrated probability implies:
    # IV(x; ref) == -IV(-x; -ref)
    mirrored = standardized_replacement_probability_points(-0.2, -0.01, calibration_slope=slope, calibration_intercept=intercept)
    assert higher == pytest.approx(-mirrored, abs=1e-12)


def test_calibrated_probability_is_complement_symmetric() -> None:
    slope, intercept = 0.8, 0.0
    for logit in (-3.0, -0.4, 0.0, 0.7, 2.2):
        assert calibrated_probability(logit, calibration_slope=slope, calibration_intercept=intercept) == pytest.approx(
            1.0 - calibrated_probability(-logit, calibration_slope=slope, calibration_intercept=intercept), abs=1e-12
        )


def test_reference_mixture_is_equal_weight_mean() -> None:
    assert reference_mixture_logit([0.1, 0.3, -0.1]) == pytest.approx(0.1, abs=1e-12)
    with pytest.raises(ValueError):
        reference_mixture_logit([])


def test_response_regret_nonnegative_and_empty_support_unavailable() -> None:
    member_logits = {"Aatrox": 0.2, "Gnar": 0.1, "Renekton": -0.05}
    result = response_regret(
        role="top",
        champion="Aatrox",
        champion_logit=0.2,
        reference_logit=0.05,
        member_logits=member_logits,
        counter_logit={"top|Aatrox|Gnar": 0.12, "top|Renekton|Aatrox": -0.08},
        calibration_slope=1.0,
        calibration_intercept=0.0,
    )
    assert result is not None
    assert result["nonnegative"] is True
    assert result["regret"] >= 0.0
    assert result["tail_alpha"] == 0.25
    # the champion itself is excluded from its own plausible-response support
    assert "Aatrox" not in result["support"]
    assert result["support_size"] == len(result["support"]) == 2

    empty = response_regret(
        role="top",
        champion="Aatrox",
        champion_logit=0.2,
        reference_logit=0.05,
        member_logits=member_logits,
        counter_logit={},
        calibration_slope=1.0,
        calibration_intercept=0.0,
    )
    assert empty is None  # unavailable, never a fabricated zero


def test_response_regret_matches_type7_quantile_arithmetic() -> None:
    member_logits = {"Aatrox": 0.2, "Gnar": 0.1}
    result = response_regret(
        role="top",
        champion="Aatrox",
        champion_logit=0.2,
        reference_logit=0.1,
        member_logits=member_logits,
        counter_logit={"top|Aatrox|Gnar": 0.3, "top|Aatrox|Renekton": -0.2, "top|Renekton|Gnar": 0.1},
        calibration_slope=1.0,
        calibration_intercept=0.0,
    )
    assert result is not None
    # independent recomputation: the reference side is the frozen cell mixture
    # (members Aatrox, Gnar) averaged against the same opponent
    deltas = []
    for opponent in result["support"]:
        if opponent == "Gnar":
            champion_side = 0.2 + 0.3
            reference_side = 0.1 + (0.3 + 0.0) / 2.0  # Aatrox+Gnar vs Gnar
        else:
            champion_side = 0.2 - 0.2
            # pair keys sort alphabetically: "top|Renekton|Gnar" is only found
            # when the evaluated side is Renekton, so Gnar's contribution is 0
            reference_side = 0.1 + (-0.2 + 0.0) / 2.0  # Aatrox+Gnar vs Renekton
        deltas.append(100.0 * (1.0 / (1.0 + math.exp(-champion_side)) - 1.0 / (1.0 + math.exp(-reference_side))))
    assert result["mean_delta"] == pytest.approx(float(np.mean(deltas)), abs=1e-12)
    assert result["lower_tail_quantile"] == pytest.approx(float(np.quantile(deltas, 0.25)), abs=1e-12)
    assert result["regret"] == pytest.approx(max(0.0, float(np.mean(deltas)) - float(np.quantile(deltas, 0.25))), abs=1e-12)


def test_type7_quantile_matches_numpy() -> None:
    sample = [0.3, 1.2, -0.4, 2.1, 0.0, 1.8, -1.1]
    for alpha in (0.05, 0.25, 0.5, 0.9):
        assert _type7_quantile(sample, alpha) == pytest.approx(float(np.quantile(sample, alpha)), abs=1e-12)


def test_tail_alpha_validation() -> None:
    with pytest.raises(ValueError):
        response_regret(
            role="top", champion="Aatrox", champion_logit=0.2, reference_logit=0.0,
            member_logits={"Aatrox": 0.2}, counter_logit={"top|Aatrox|Gnar": 0.1},
            tail_alpha=1.0, calibration_slope=1.0, calibration_intercept=0.0,
        )
