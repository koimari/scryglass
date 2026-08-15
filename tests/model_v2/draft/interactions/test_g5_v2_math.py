from __future__ import annotations

import numpy as np
import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import v2_math, v2_runner


def _problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.array([
        [1.0, 0.25], [-1.0, 0.5], [0.5, -1.0], [-0.5, -0.75],
        [1.5, 0.8], [-1.5, -0.2],
    ])
    y = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    return x, y, np.zeros(6), np.array([0.2, 0.35])


def test_extreme_finite_offsets_have_finite_stable_algebra() -> None:
    x = np.ones((4, 1))
    y = np.array([1.0, 1.0, 0.0, 0.0])
    offsets = np.array([1e308, 1e308, -1e308, -1e308])
    objective, gradient, hessian = v2_math.objective_gradient_hessian(
        np.zeros(1), x, y, offsets, np.ones(1)
    )
    assert np.isfinite(objective)
    assert np.all(np.isfinite(gradient))
    assert np.all(np.isfinite(hessian))


@pytest.mark.parametrize(
    "field,value",
    [
        ("precision", np.array([0.0, 1.0])),
        ("precision", np.array([np.nan, 1.0])),
        ("offset", np.array([0, 0, 0, 0, 0, np.inf])),
        ("label", np.array([0, 1, 0, 1, 0, 2])),
        ("design", np.array([[np.nan, 0.0]] * 6)),
    ],
)
def test_nonfinite_nonpositive_inputs_fail_typed(field: str, value: np.ndarray) -> None:
    x, y, offsets, precision = _problem()
    values = {"design": x, "label": y, "offset": offsets, "precision": precision}
    values[field] = value
    with pytest.raises(v2_math.V2NumericalUnavailable, match=v2_math.BLOCKER):
        v2_math.validate_problem(
            values["design"], values["label"], values["offset"], values["precision"]
        )


def test_zero_and_excess_exposure_fail_typed() -> None:
    with pytest.raises(v2_math.V2NumericalUnavailable, match="ZERO_OR_EXCESS"):
        v2_math.train_column_scales(np.array([[1.0, 0.0], [2.0, 0.0]]))
    with pytest.raises(v2_math.V2NumericalUnavailable, match="ZERO_OR_EXCESS"):
        v2_math.train_column_scales(np.array([[1.0, 1e151], [2.0, 1e151]]))
    for value in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(v2_math.V2NumericalUnavailable):
            v2_math.require_positive_finite("DENOMINATOR", value)


def test_train_reparameterization_preserves_predictor_and_penalty_exactly() -> None:
    x, _y, _offsets, precision = _problem()
    scales = v2_math.train_column_scales(x)
    xs, precision_s = v2_math.reparameterize_train(x, precision, scales)
    beta = np.array([0.4, -0.7])
    gamma = scales * beta
    assert np.allclose(x @ beta, xs @ gamma, rtol=0.0, atol=2e-16)
    assert np.isclose(
        0.5 * np.dot(precision, beta * beta),
        0.5 * np.dot(precision_s, gamma * gamma),
        rtol=1e-15,
        atol=0.0,
    )


def test_gradient_and_hessian_match_high_accuracy_finite_differences() -> None:
    x, y, offsets, precision = _problem()
    scales = v2_math.train_column_scales(x)
    xs, ps = v2_math.reparameterize_train(x, precision, scales)
    gamma = np.array([0.2, -0.3])
    objective, gradient, hessian = v2_math.objective_gradient_hessian(
        gamma, xs, y, offsets, ps
    )
    epsilon = 2e-5
    numeric_gradient = np.zeros(2)
    numeric_hessian = np.zeros((2, 2))
    for index in range(2):
        step = np.zeros(2)
        step[index] = epsilon
        plus = v2_math.objective_gradient_hessian(gamma + step, xs, y, offsets, ps)
        minus = v2_math.objective_gradient_hessian(gamma - step, xs, y, offsets, ps)
        numeric_gradient[index] = (plus[0] - minus[0]) / (2 * epsilon)
        numeric_hessian[:, index] = (plus[1] - minus[1]) / (2 * epsilon)
    assert np.isfinite(objective)
    assert np.allclose(gradient, numeric_gradient, rtol=2e-8, atol=2e-9)
    assert np.allclose(hessian, numeric_hessian, rtol=2e-8, atol=2e-9)


def test_damped_newton_is_deterministic_convergent_and_armijo_descends() -> None:
    x, y, offsets, precision = _problem()
    scales = v2_math.train_column_scales(x)
    xs, ps = v2_math.reparameterize_train(x, precision, scales)
    first = v2_math.damped_newton(xs, y, offsets, ps)
    second = v2_math.damped_newton(xs, y, offsets, ps)
    assert first["status"] == "CONVERGED"
    assert first["trace_sha256"] == second["trace_sha256"]
    objectives = [float.fromhex(item["objective_hex"]) for item in first["trace"]]
    candidates = [float.fromhex(item["candidate_objective_hex"]) for item in first["trace"]]
    assert all(candidate <= objective for objective, candidate in zip(objectives, candidates))
    assert np.all(np.linalg.eigvalsh(first["hessian"]) > 0.0)
    assert np.all(np.linalg.eigvalsh(first["covariance_gamma"]) > 0.0)


def test_custom_configuration_is_rejected_before_solver_use() -> None:
    x, y, offsets, precision = _problem()
    scales = v2_math.train_column_scales(x)
    xs, ps = v2_math.reparameterize_train(x, precision, scales)
    config = v2_math.NewtonConfig(initial_alpha=2.0)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="CONFIG_NOT_FROZEN"):
        v2_math.damped_newton(xs, y, offsets, ps, config=config)


@pytest.mark.parametrize(
    "config,match",
    [
        (v2_math.NewtonConfig(max_iterations=True), "CONFIG_INTEGER"),
        (v2_math.NewtonConfig(max_backtracks=0), "CONFIG_INTEGER"),
        (v2_math.NewtonConfig(max_iterations=-1), "CONFIG_INTEGER"),
        (v2_math.NewtonConfig(armijo_c1=np.nan), "CONFIG_NONFINITE"),
        (v2_math.NewtonConfig(armijo_c1=1.0), "CONFIG_ARMIJO"),
        (v2_math.NewtonConfig(shrink=0.0), "CONFIG_NONPOSITIVE"),
        (v2_math.NewtonConfig(shrink=1.0), "CONFIG_SHRINK"),
        (v2_math.NewtonConfig(step_inf_tolerance=False), "CONFIG_NUMERIC_TYPE"),
        (v2_math.NewtonConfig(cholesky_jitter=1e-12), "JITTER_POLICY"),
    ],
)
def test_invalid_configuration_fails_typed(
    config: v2_math.NewtonConfig, match: str
) -> None:
    with pytest.raises(v2_math.V2NumericalUnavailable, match=match):
        v2_math.validate_config(config)


def test_config_hash_binds_actual_and_rejects_custom() -> None:
    assert v2_math.config_hash(v2_math.CONFIG) == v2_math.sha256(
        vars(v2_math.CONFIG)
    )
    with pytest.raises(v2_math.V2NumericalUnavailable, match="CONFIG_NOT_FROZEN"):
        v2_math.config_hash(v2_math.NewtonConfig(initial_alpha=0.75))
    with pytest.raises(v2_math.V2NumericalUnavailable, match="CONFIG_NUMERIC_TYPE"):
        v2_math.validate_config(v2_math.NewtonConfig(initial_alpha=1))


@pytest.mark.parametrize(
    "detail,expected",
    [
        ("ARMIJO_EXHAUSTED", "ARMIJO_EXHAUSTED"),
        ("STAGNATION", "STAGNATION"),
        ("MAX_ITERATIONS", "MAX_ITERATIONS"),
        ("CONFIG_NOT_FROZEN", "CONFIGURATION"),
        ("COVARIANCE_RESIDUAL", "COVARIANCE"),
        ("SOLVE_RESIDUAL", "SOLVE_RESIDUAL"),
        ("HESSIAN_FACTORIZATION", "FACTORIZATION"),
        ("OBJECTIVE_DERIVATIVE_NONFINITE", "NONFINITE"),
    ],
)
def test_internal_failures_map_to_closed_result_blockers(
    detail: str, expected: str
) -> None:
    error = v2_math.V2NumericalUnavailable(f"{v2_math.BLOCKER}:{detail}")
    assert v2_math.closed_result_blocker(error) == f"{v2_math.BLOCKER}:{expected}"


def test_convergence_factorization_and_solve_failures_are_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, offsets, precision = _problem()

    def bad_hessian(*_args):
        return 1.0, np.zeros(2), np.array([[1.0, 0.0], [0.0, -1.0]])

    monkeypatch.setattr(v2_math, "objective_gradient_hessian", bad_hessian)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="HESSIAN_FACTORIZATION"):
        v2_math.damped_newton(x, y, offsets, precision)

    def good_hessian(*_args):
        return 1.0, np.zeros(2), np.eye(2)

    monkeypatch.setattr(v2_math, "objective_gradient_hessian", good_hessian)

    def failed_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("injected")

    monkeypatch.setattr(np.linalg, "solve", failed_solve)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="LINEAR_SOLVE"):
        v2_math.damped_newton(x, y, offsets, precision)


def test_asymmetric_hessian_and_bad_covariance_residual_fail_typed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x, y, offsets, precision = _problem()

    def asymmetric(*_args):
        return 1.0, np.zeros(2), np.array([[2.0, 0.1], [0.0, 2.0]])

    monkeypatch.setattr(v2_math, "objective_gradient_hessian", asymmetric)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="HESSIAN_ASYMMETRIC"):
        v2_math.damped_newton(x, y, offsets, precision)

    monkeypatch.setattr(
        v2_math,
        "objective_gradient_hessian",
        lambda *_args: (1.0, np.zeros(2), np.eye(2)),
    )
    original_solve = np.linalg.solve

    def wrong_solve(a, b):
        result = original_solve(a, b)
        return result * 2.0

    monkeypatch.setattr(np.linalg, "solve", wrong_solve)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="SOLVE_RESIDUAL"):
        v2_math.damped_newton(x, y, offsets, precision)


@pytest.mark.parametrize(
    "covariance,match",
    [
        (np.array([[np.nan, 0.0], [0.0, 1.0]]), "NONFINITE_MATRIX_NORM"),
        (np.array([[1.0, 0.2], [0.0, 1.0]]), "COVARIANCE_ASYMMETRIC"),
        (np.array([[1.0, 0.0], [0.0, -1.0]]), "COVARIANCE_NEGATIVE_DIAGONAL"),
        (np.eye(2) * 2.0, "COVARIANCE_RESIDUAL"),
    ],
)
def test_nonfinite_asymmetric_indefinite_or_bad_residual_covariance_is_typed(
    monkeypatch: pytest.MonkeyPatch, covariance: np.ndarray, match: str
) -> None:
    monkeypatch.setattr(
        v2_math,
        "_checked_cholesky_solve",
        lambda *_args, **_kwargs: covariance.copy(),
    )
    with pytest.raises(v2_math.V2NumericalUnavailable, match=match):
        v2_math._checked_covariance(np.eye(2), np.eye(2), v2_math.CONFIG)


def test_stagnation_and_armijo_exhaustion_are_distinct_typed_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = np.ones((2, 1))
    y = np.array([0.0, 1.0])
    offsets = np.zeros(2)
    precision = np.ones(1)

    def stagnant(gamma, *_args):
        return (1.0 if gamma[0] == 0.0 else 0.5), np.ones(1), np.array([[1e12]])

    monkeypatch.setattr(v2_math, "objective_gradient_hessian", stagnant)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="STAGNATION"):
        v2_math.damped_newton(x, y, offsets, precision)

    def never_decreases(gamma, *_args):
        return (0.0 if gamma[0] == 0.0 else 1.0), np.ones(1), np.eye(1)

    monkeypatch.setattr(v2_math, "objective_gradient_hessian", never_decreases)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="ARMIJO_EXHAUSTED"):
        v2_math.damped_newton(x, y, offsets, precision)


def test_sparse_tiny_and_maximum_admissible_columns_are_stable_or_typed() -> None:
    sparse = np.array([[0.0, 1.0], [2.0, 0.0], [0.0, -1.0], [-2.0, 0.0]])
    scales = v2_math.train_column_scales(sparse)
    xs, ps = v2_math.reparameterize_train(sparse, np.ones(2), scales)
    fit = v2_math.damped_newton(
        xs, np.array([0.0, 1.0, 0.0, 1.0]), np.zeros(4), ps
    )
    assert fit["status"] == "CONVERGED"

    for magnitude, precision in ((1e-150, 1e-300), (1e150, 1.0)):
        extreme = np.array([[magnitude], [-magnitude]])
        scale = v2_math.train_column_scales(extreme)
        transformed = v2_math.reparameterize_train(
            extreme, np.array([precision]), scale
        )
        assert np.all(np.isfinite(transformed[0]))
        assert np.all(np.isfinite(transformed[1]))


def test_aggregate_logit_overflow_and_rank_pathologies_fail_closed_or_fit() -> None:
    for offset, labels in (
        (1e308, np.zeros(4)),
        (-1e308, np.ones(4)),
    ):
        with pytest.raises(
            v2_math.V2NumericalUnavailable,
            match="OBJECTIVE_DERIVATIVE_NONFINITE",
        ):
            v2_math.objective_gradient_hessian(
                np.zeros(1),
                np.ones((4, 1)),
                labels,
                np.full(4, offset),
                np.ones(1),
            )
    duplicate = np.array([[1.0, 1.0], [-1.0, -1.0], [2.0, 2.0], [-2.0, -2.0]])
    near = duplicate.copy()
    near[:, 1] += np.array([1e-12, -1e-12, 2e-12, -2e-12])
    for design in (duplicate, near):
        scale = v2_math.train_column_scales(design)
        xs, ps = v2_math.reparameterize_train(design, np.array([0.1, 0.1]), scale)
        assert v2_math.damped_newton(
            xs, np.array([1.0, 0.0, 1.0, 0.0]), np.zeros(4), ps
        )["status"] == "CONVERGED"


def test_row_column_and_sign_transform_equivalence() -> None:
    x, y, offsets, precision = _problem()

    def fit(design, labels, off, priors):
        scales = v2_math.train_column_scales(design)
        xs, ps = v2_math.reparameterize_train(design, priors, scales)
        result = v2_math.damped_newton(xs, labels, off, ps)
        return result["gamma"] / scales

    base = fit(x, y, offsets, precision)
    order = np.array([5, 2, 0, 4, 1, 3])
    assert np.allclose(fit(x[order], y[order], offsets[order], precision), base)
    swapped = fit(x[:, ::-1], y, offsets, precision[::-1])
    assert np.allclose(swapped[::-1], base)
    signed = x.copy()
    signed[:, 1] *= -1.0
    signed_fit = fit(signed, y, offsets, precision)
    assert np.allclose(signed_fit, base * np.array([1.0, -1.0]))


def test_complete_and_quasi_separation_remain_finite_with_proper_priors() -> None:
    x = np.array([[-4.0], [-2.0], [-1.0], [1.0], [2.0], [4.0]])
    for y in (
        np.array([0, 0, 0, 1, 1, 1], dtype=float),
        np.array([0, 0, 1, 1, 1, 1], dtype=float),
    ):
        scale = v2_math.train_column_scales(x)
        xs, ps = v2_math.reparameterize_train(x, np.array([0.1]), scale)
        fit = v2_math.damped_newton(xs, y, np.zeros(6), ps)
        assert fit["status"] == "CONVERGED"
        assert np.isfinite(fit["objective"])


def test_singular_or_invalid_hessian_inputs_fail_before_selection() -> None:
    x = np.ones((4, 2))
    with pytest.raises(v2_math.V2NumericalUnavailable, match=v2_math.BLOCKER):
        v2_math.damped_newton(
            x, np.array([0, 1, 0, 1], dtype=float), np.zeros(4), np.zeros(2)
        )
    bad = v2_math.NewtonConfig(cholesky_jitter=1e-9)
    with pytest.raises(v2_math.V2NumericalUnavailable, match="JITTER_POLICY"):
        v2_math.damped_newton(
            x, np.array([0, 1, 0, 1], dtype=float), np.zeros(4), np.ones(2),
            config=bad,
        )


def test_unseen_champion_role_is_explicit_prior_only() -> None:
    value = v2_math.prior_only_coordinate(
        blue_count=1, red_count=0, prior_variance=0.01
    )
    assert value == {
        "prior_only": True,
        "net_count": 1,
        "mean_increment": 0.0,
        "variance": 0.01,
    }
    with pytest.raises(v2_math.V2NumericalUnavailable):
        v2_math.prior_only_coordinate(
            blue_count=1, red_count=0, prior_variance=0.0
        )
    with pytest.raises(v2_math.V2NumericalUnavailable, match="PRIOR_ONLY_NONFINITE"):
        v2_math.prior_only_coordinate(
            blue_count=10**2000, red_count=0, prior_variance=0.01
        )


def test_v1_failure_signature_fixture_is_caught_before_v2_selection() -> None:
    x = np.full((2, 1), 1e308)
    beta = np.array([1e308])
    with np.errstate(over="ignore", invalid="ignore"):
        z = x @ beta
        residual = np.array([1.0, -1.0])
        gradient = x.T @ residual
        hessian = x.T @ x
    assert not np.all(np.isfinite(z))
    assert not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian))
    with pytest.raises(v2_math.V2NumericalUnavailable, match="EXCESS_EXPOSURE"):
        v2_math.validate_problem(
            x, np.array([0.0, 1.0]), np.zeros(2), np.ones(1)
        )


def test_train_diagnostic_rejects_non_train_and_emits_no_ids_or_metrics() -> None:
    x, y, offsets, precision = _problem()
    with pytest.raises(v2_runner.V2RunnerError, match="NON_TRAIN"):
        v2_runner.train_only_diagnostic(
            partition="DEVELOPMENT", design=x, labels=y,
            offsets=offsets, precisions=precision,
        )
    payload = v2_runner.train_only_diagnostic(
        partition="TRAIN", design=x, labels=y, offsets=offsets, precisions=precision
    )
    assert payload["emits_labels_or_ids"] is False
    assert payload["selection_metrics"] == "STRUCTURALLY_PROHIBITED"
    assert "labels" not in payload and "ids" not in payload
