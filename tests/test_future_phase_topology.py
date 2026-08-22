from __future__ import annotations

import math

from lol_kills.research.future_phase_curve import (
    PHASE_SHAPE_AVAILABILITY_FEATURES,
    PHASE_SHAPE_FEATURES,
    PHASE_SHAPE_INVARIANT_FEATURES,
    PHASE_SHAPE_SIGNED_FEATURES,
    phase_shape_features,
    side_swap_phase_shape_features,
    validate_phase_shape_side_swap,
)


def test_phase_shape_reports_early_late_slopes_and_area() -> None:
    result = phase_shape_features(
        [0.0, 100.0, 200.0, 240.0],
        [0.0, 50.0, 150.0, 250.0],
        available=True,
    )

    assert result["forecast_curve_available"] == 1.0
    assert result["forecast_curve_missing"] == 0.0
    assert result["forecast_gold_slope_10_15"] == 20.0
    assert result["forecast_gold_slope_15_20"] == 20.0
    assert result["forecast_gold_slope_20_25"] == 8.0
    assert result["forecast_gold_early_mean"] == 50.0
    assert result["forecast_gold_late_mean"] == 220.0
    assert result["forecast_gold_late_minus_early"] == 170.0
    assert result["forecast_gold_late_minus_early_slope"] == 17.0
    assert result["forecast_gold_late_minus_early_acceleration"] == -12.0
    assert result["forecast_gold_signed_area"] == 2100.0
    assert result["forecast_gold_first_material_advantage_minute_signed"] == 0.0


def test_phase_shape_counts_crossovers_and_ignores_zero_checkpoints() -> None:
    result = phase_shape_features(
        [-100.0, 0.0, 100.0, -100.0],
        [-100.0, 100.0, -100.0, 100.0],
        available=True,
    )

    assert result["forecast_gold_first_crossover_minute"] == 15.0
    assert result["forecast_gold_crossover_count"] == 2.0
    assert result["forecast_xp_first_crossover_minute"] == 12.5
    assert result["forecast_xp_crossover_count"] == 3.0


def test_phase_shape_material_threshold_is_interpolated_and_signed() -> None:
    result = phase_shape_features(
        [0.0, 100.0, 300.0, 400.0],
        [0.0, -100.0, -300.0, -400.0],
        available=True,
    )

    assert math.isclose(
        result["forecast_gold_first_material_advantage_minute_signed"],
        18.75,
    )
    assert math.isclose(
        result["forecast_xp_first_material_advantage_minute_signed"],
        -18.75,
    )


def test_phase_shape_partial_and_explicitly_missing_curves_fail_closed() -> None:
    partial = phase_shape_features(
        [0.0, None, 100.0, 200.0],
        [0.0, 50.0, 100.0, 150.0],
        available=True,
    )
    missing = phase_shape_features(
        [0.0, 100.0, 200.0, 300.0],
        [0.0, 50.0, 100.0, 150.0],
        available=False,
    )

    for result in (partial, missing):
        assert result["forecast_curve_available"] == 0.0
        assert result["forecast_curve_missing"] == 1.0
        assert all(
            result[name] is None
            for name in (
                *PHASE_SHAPE_SIGNED_FEATURES,
                *(name for name in PHASE_SHAPE_INVARIANT_FEATURES if name not in PHASE_SHAPE_AVAILABILITY_FEATURES),
            )
        )


def test_phase_shape_signed_fields_reverse_and_invariants_survive_side_swap() -> None:
    original = phase_shape_features(
        [-300.0, -100.0, 300.0, 100.0],
        [-100.0, 100.0, -100.0, 100.0],
        available=True,
    )
    swapped = side_swap_phase_shape_features(
        [-300.0, -100.0, 300.0, 100.0],
        [-100.0, 100.0, -100.0, 100.0],
        available=True,
    )

    for name in PHASE_SHAPE_SIGNED_FEATURES:
        assert swapped[name] == -original[name]
    for name in (*PHASE_SHAPE_INVARIANT_FEATURES, *PHASE_SHAPE_AVAILABILITY_FEATURES):
        assert swapped[name] == original[name]
    assert validate_phase_shape_side_swap(original, swapped)["passed"] is True


def test_phase_shape_registries_do_not_include_checkpoint_targets() -> None:
    registered = (*PHASE_SHAPE_SIGNED_FEATURES, *PHASE_SHAPE_INVARIANT_FEATURES)
    assert PHASE_SHAPE_FEATURES == registered
    assert len(registered) == len(set(registered))
    assert "forecast_gold_diff_10" not in registered
    assert "forecast_xp_diff_25" not in registered
    assert all("_diff_" not in name for name in registered)
