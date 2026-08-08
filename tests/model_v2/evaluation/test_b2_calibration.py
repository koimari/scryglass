from __future__ import annotations

import math

import numpy as np
import pytest

from lol_kills.v2.evaluation.calibration import (
    CALIBRATION_FAMILIES,
    apply_registered_transform,
    fit_logistic_calibration,
    select_nested_transform,
)


X = [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]
Y = [0, 1, 0, 0, 1, 0, 1, 1, 1]


def test_scipy_logistic_mle_matches_wolfram_oracle_and_not_ols() -> None:
    result = fit_logistic_calibration(X, Y)
    assert result.status == "ok"
    assert result.intercept == pytest.approx(0.3075762643523149, abs=1e-7)
    assert result.slope == pytest.approx(0.9898041959242025, abs=1e-7)
    assert result.parameters["gradient_inf_norm"] < 1e-6
    assert result.parameters["information_eigenvalues"] == pytest.approx(
        [1.49803350115, 2.17342623665], abs=1e-7
    )
    ols = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), Y, rcond=None)[0]
    assert np.linalg.norm(np.array([result.intercept, result.slope]) - ols) > .1


@pytest.mark.parametrize(
    "x,y,reason",
    [
        (X, [0] * len(X), "constant_outcome"),
        (X, [1] * len(X), "constant_outcome"),
        ([0] * len(X), Y, "constant_logit"),
        ([-4, -3, -2, 2, 3, 4], [0, 0, 0, 1, 1, 1], "separation_or_singular_hessian"),
    ],
)
def test_degenerate_logistic_diagnostics_are_unavailable(x, y, reason) -> None:
    result = fit_logistic_calibration(x, y)
    assert result.status == "unavailable"
    assert result.reason == reason


PARAMETERS = {
    "identity": {},
    "symmetric_temperature": {"slope": .75},
    "symmetrized_platt": {"intercept": .3, "slope": 1.2},
    "symmetrized_beta": {"intercept": -.2, "a": 1.1, "b": .9},
    "symmetrized_bounded_isotonic": {
        "knots": [-3, -1, 0, 1, 3], "levels": [.02, .25, .5, .75, .98]
    },
}


@pytest.mark.parametrize("family", CALIBRATION_FAMILIES)
def test_registered_transforms_are_open_monotone_and_complement_symmetric(family: str) -> None:
    z = np.linspace(-1000, 1000, 4001)
    p = apply_registered_transform(z, family, PARAMETERS[family])
    q = apply_registered_transform(-z, family, PARAMETERS[family])
    assert np.all(np.isfinite(p))
    assert np.all((p > 0) & (p < 1))
    assert np.all(np.diff(p) >= -1e-12)
    assert np.max(np.abs(p + q - 1)) < 1e-12


@pytest.mark.parametrize(
    "family,parameters",
    [
        ("symmetric_temperature", {"slope": -1}),
        ("symmetrized_platt", {"intercept": 0, "slope": -1}),
        ("symmetrized_beta", {"intercept": 0, "a": -1, "b": 1}),
        ("symmetrized_bounded_isotonic", {"knots": [0, 1], "levels": [.8, .2]}),
    ],
)
def test_invalid_transform_parameters_fail_closed(family, parameters) -> None:
    with pytest.raises(ValueError):
        apply_registered_transform([-1, 0, 1], family, parameters)


def test_nested_selection_is_chronological_series_blocked_and_content_addressed() -> None:
    x = [-2.4, -1.7, -1.2, -.8, -.2, .2, .7, 1.1, 1.6, 2.2, 2.5, 2.8]
    y = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    state = select_nested_transform(
        x, y, [f"s{i//2}" for i in range(12)], list(range(12)), [f"r{i}" for i in range(12)]
    )
    assert state.status == "ok"
    assert len(state.selection_sha256) == 64
    assert len(state.calibration_row_sha256) == 64
    changed = select_nested_transform(
        x, y, [f"s{i//2}" for i in range(12)], list(range(12)), [f"x{i}" for i in range(12)]
    )
    assert changed.selection_sha256 != state.selection_sha256


def test_outer_test_label_is_not_an_input_to_nested_selection() -> None:
    import inspect
    assert "test" not in inspect.signature(select_nested_transform).parameters
