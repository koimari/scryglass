"""Stable TRAIN-only numerical primitives for the G5 v2 addendum."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
from scipy.special import expit


BLOCKER = "V2_PREFIT_NUMERICAL_UNAVAILABLE"
MAX_ABS_DESIGN = 1e150


class V2NumericalUnavailable(RuntimeError):
    """A frozen numerical precondition or deterministic solve failed."""


def closed_result_blocker(error: V2NumericalUnavailable) -> str:
    """Collapse internal detail into the frozen, non-favorable result taxonomy."""
    detail = str(error).removeprefix(f"{BLOCKER}:")
    if "ARMIJO" in detail:
        suffix = "ARMIJO_EXHAUSTED"
    elif "STAGNATION" in detail:
        suffix = "STAGNATION"
    elif "MAX_ITERATIONS" in detail:
        suffix = "MAX_ITERATIONS"
    elif "CONFIG" in detail or "JITTER" in detail:
        suffix = "CONFIGURATION"
    elif "COVARIANCE" in detail:
        suffix = "COVARIANCE"
    elif "SOLVE" in detail:
        suffix = "SOLVE_RESIDUAL"
    elif "HESSIAN" in detail or "FACTORIZATION" in detail:
        suffix = "FACTORIZATION"
    else:
        suffix = "NONFINITE"
    return f"{BLOCKER}:{suffix}"


@dataclass(frozen=True)
class NewtonConfig:
    initial_alpha: float = 1.0
    armijo_c1: float = 1e-4
    shrink: float = 0.5
    max_backtracks: int = 40
    max_iterations: int = 100
    gradient_inf_tolerance: float = 1e-9
    step_inf_tolerance: float = 1e-11
    symmetry_tolerance: float = 1e-12
    factorization_residual_tolerance: float = 1e-10
    solve_residual_tolerance: float = 1e-10
    covariance_residual_tolerance: float = 1e-9
    quadratic_form_tolerance: float = 1e-12
    cholesky_jitter: float = 0.0


CONFIG = NewtonConfig()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def config_hash(config: NewtonConfig = CONFIG) -> str:
    validate_config(config)
    return sha256(asdict(config))


def validate_config(config: NewtonConfig) -> NewtonConfig:
    """Require the single frozen solver configuration with exact JSON-safe types."""
    if type(config) is not NewtonConfig:
        raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_TYPE")
    values = asdict(config)
    integer_fields = {"max_backtracks", "max_iterations"}
    for name in integer_fields:
        value = values[name]
        if type(value) is not int or value <= 0:
            raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_INTEGER")
    numeric_fields = set(values) - integer_fields
    for name in numeric_fields:
        value = values[name]
        if type(value) is not float:
            raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_NUMERIC_TYPE")
        if not np.isfinite(float(value)):
            raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_NONFINITE")
    positive = numeric_fields - {"cholesky_jitter"}
    if any(float(values[name]) <= 0.0 for name in positive):
        raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_NONPOSITIVE")
    if not 0.0 < float(config.armijo_c1) < 1.0:
        raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_ARMIJO")
    if not 0.0 < float(config.shrink) < 1.0:
        raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_SHRINK")
    if config.cholesky_jitter != 0.0:
        raise V2NumericalUnavailable(f"{BLOCKER}:JITTER_POLICY")
    if config != CONFIG:
        raise V2NumericalUnavailable(f"{BLOCKER}:CONFIG_NOT_FROZEN")
    return config


def _finite(name: str, value: np.ndarray) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise V2NumericalUnavailable(f"{BLOCKER}:{name}_TYPE") from error
    if not np.all(np.isfinite(array)):
        raise V2NumericalUnavailable(f"{BLOCKER}:{name}_NONFINITE")
    return array


def require_positive_finite(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        raise V2NumericalUnavailable(f"{BLOCKER}:{name}_NONPOSITIVE_OR_NONFINITE")
    return float(value)


def validate_problem(
    design: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    precisions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = _finite("DESIGN", design)
    y = _finite("LABEL", labels)
    o = _finite("OFFSET", offsets)
    precision = _finite("PRECISION", precisions)
    if x.ndim != 2 or y.shape != (x.shape[0],) or o.shape != y.shape:
        raise V2NumericalUnavailable(f"{BLOCKER}:SHAPE")
    if precision.shape != (x.shape[1],):
        raise V2NumericalUnavailable(f"{BLOCKER}:PRECISION_SHAPE")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise V2NumericalUnavailable(f"{BLOCKER}:ZERO_EXPOSURE")
    if np.any((y != 0.0) & (y != 1.0)):
        raise V2NumericalUnavailable(f"{BLOCKER}:LABEL_DOMAIN")
    if np.any(precision <= 0.0):
        raise V2NumericalUnavailable(f"{BLOCKER}:PRECISION_NONPOSITIVE")
    if np.max(np.abs(x)) > MAX_ABS_DESIGN:
        raise V2NumericalUnavailable(f"{BLOCKER}:EXCESS_EXPOSURE")
    return x, y, o, precision


def train_column_scales(design: np.ndarray) -> np.ndarray:
    x = _finite("TRAIN_DESIGN", design)
    if x.ndim != 2 or x.shape[0] == 0:
        raise V2NumericalUnavailable(f"{BLOCKER}:TRAIN_SCALE_SHAPE")
    maximum = np.max(np.abs(x), axis=0)
    if np.any(maximum == 0.0) or np.any(maximum > MAX_ABS_DESIGN):
        raise V2NumericalUnavailable(f"{BLOCKER}:ZERO_OR_EXCESS_EXPOSURE")
    scaled = x / maximum
    rms = maximum * np.sqrt(np.mean(scaled * scaled, axis=0))
    if not np.all(np.isfinite(rms)) or np.any(rms <= 0.0):
        raise V2NumericalUnavailable(f"{BLOCKER}:SCALE_NONPOSITIVE")
    return rms


def reparameterize_train(
    design: np.ndarray, precisions: np.ndarray, scales: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    x = _finite("DESIGN", design)
    precision = _finite("PRECISION", precisions)
    scale = _finite("SCALE", scales)
    if scale.shape != (x.shape[1],) or precision.shape != scale.shape:
        raise V2NumericalUnavailable(f"{BLOCKER}:TRANSFORM_SHAPE")
    if np.any(scale <= 0.0) or np.any(precision <= 0.0):
        raise V2NumericalUnavailable(f"{BLOCKER}:TRANSFORM_NONPOSITIVE")
    transformed_x = x / scale
    # Sequential division preserves lambda/s^2 without prematurely overflowing
    # or underflowing the intermediate square for admissible extreme scales.
    transformed_precision = (precision / scale) / scale
    if not np.all(np.isfinite(transformed_x)) or not np.all(
        np.isfinite(transformed_precision)
    ) or np.any(transformed_precision <= 0.0):
        raise V2NumericalUnavailable(f"{BLOCKER}:TRANSFORM_NONFINITE")
    return transformed_x, transformed_precision


def prior_only_coordinate(
    *, blue_count: int, red_count: int, prior_variance: float
) -> dict[str, float | int | bool]:
    if (
        type(blue_count) is not int
        or type(red_count) is not int
        or blue_count < 0
        or red_count < 0
        or not np.isfinite(prior_variance)
        or prior_variance <= 0.0
    ):
        raise V2NumericalUnavailable(f"{BLOCKER}:PRIOR_ONLY_INPUT")
    net = blue_count - red_count
    try:
        variance = float(net * net * prior_variance)
    except (OverflowError, ValueError, TypeError) as error:
        raise V2NumericalUnavailable(f"{BLOCKER}:PRIOR_ONLY_NONFINITE") from error
    if not np.isfinite(variance):
        raise V2NumericalUnavailable(f"{BLOCKER}:PRIOR_ONLY_NONFINITE")
    return {
        "prior_only": True,
        "net_count": net,
        "mean_increment": 0.0,
        "variance": variance,
    }


def objective_gradient_hessian(
    gamma: np.ndarray,
    design_scaled: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    precision_scaled: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    x, y, o, precision = validate_problem(
        design_scaled, labels, offsets, precision_scaled
    )
    coefficient = _finite("COEFFICIENT", gamma)
    if coefficient.shape != (x.shape[1],):
        raise V2NumericalUnavailable(f"{BLOCKER}:COEFFICIENT_SHAPE")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        z = o + x @ coefficient
    if not np.all(np.isfinite(z)):
        raise V2NumericalUnavailable(f"{BLOCKER}:LINEAR_PREDICTOR_NONFINITE")
    losses = np.where(y == 1.0, np.logaddexp(0.0, -z), np.logaddexp(0.0, z))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        objective = float(
            np.sum(losses) + 0.5 * np.dot(precision, coefficient * coefficient)
        )
    residual = expit(z) - y
    log_weight = -np.logaddexp(0.0, -z) - np.logaddexp(0.0, z)
    weight = np.exp(log_weight)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        gradient = x.T @ residual + precision * coefficient
        hessian = x.T @ (weight[:, None] * x) + np.diag(precision)
    hessian = 0.5 * (hessian + hessian.T)
    if (
        not np.isfinite(objective)
        or not np.all(np.isfinite(gradient))
        or not np.all(np.isfinite(hessian))
    ):
        raise V2NumericalUnavailable(f"{BLOCKER}:OBJECTIVE_DERIVATIVE_NONFINITE")
    return objective, gradient, hessian


def _matrix_inf(value: np.ndarray) -> float:
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = float(np.linalg.norm(value, ord=np.inf))
    except (np.linalg.LinAlgError, TypeError, ValueError, OverflowError) as error:
        raise V2NumericalUnavailable(f"{BLOCKER}:MATRIX_NORM") from error
    if not np.isfinite(result):
        raise V2NumericalUnavailable(f"{BLOCKER}:NONFINITE_MATRIX_NORM")
    return result


def _require_symmetric_positive_definite(
    hessian: np.ndarray, config: NewtonConfig
) -> np.ndarray:
    h = _finite("HESSIAN", hessian)
    if h.ndim != 2 or h.shape[0] == 0 or h.shape[0] != h.shape[1]:
        raise V2NumericalUnavailable(f"{BLOCKER}:HESSIAN_SHAPE")
    scale = max(1.0, _matrix_inf(h))
    asymmetry = _matrix_inf(h - h.T)
    if asymmetry > config.symmetry_tolerance * scale:
        raise V2NumericalUnavailable(f"{BLOCKER}:HESSIAN_ASYMMETRIC")
    h = 0.5 * (h + h.T)
    try:
        chol = np.linalg.cholesky(h)
    except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError, OverflowError) as error:
        raise V2NumericalUnavailable(f"{BLOCKER}:HESSIAN_FACTORIZATION") from error
    if not np.all(np.isfinite(chol)) or np.any(np.diag(chol) <= 0.0):
        raise V2NumericalUnavailable(f"{BLOCKER}:HESSIAN_FACTORIZATION")
    residual = _matrix_inf(chol @ chol.T - h)
    if residual > config.factorization_residual_tolerance * scale:
        raise V2NumericalUnavailable(f"{BLOCKER}:FACTORIZATION_RESIDUAL")
    return chol


def _checked_cholesky_solve(
    hessian: np.ndarray,
    chol: np.ndarray,
    right_hand_side: np.ndarray,
    config: NewtonConfig,
) -> np.ndarray:
    rhs = _finite("SOLVE_RHS", right_hand_side)
    try:
        solution = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError, OverflowError) as error:
        raise V2NumericalUnavailable(f"{BLOCKER}:LINEAR_SOLVE") from error
    solution = _finite("SOLVE_SOLUTION", solution)
    residual = _matrix_inf(hessian @ solution - rhs)
    scale = max(
        1.0,
        _matrix_inf(hessian) * _matrix_inf(solution),
        _matrix_inf(rhs),
    )
    if residual > config.solve_residual_tolerance * scale:
        raise V2NumericalUnavailable(f"{BLOCKER}:SOLVE_RESIDUAL")
    return solution


def _checked_covariance(
    hessian: np.ndarray, chol: np.ndarray, config: NewtonConfig
) -> np.ndarray:
    identity = np.eye(hessian.shape[0], dtype=np.float64)
    covariance = _checked_cholesky_solve(hessian, chol, identity, config)
    scale = max(1.0, _matrix_inf(covariance))
    if _matrix_inf(covariance - covariance.T) > config.symmetry_tolerance * scale:
        raise V2NumericalUnavailable(f"{BLOCKER}:COVARIANCE_ASYMMETRIC")
    covariance = 0.5 * (covariance + covariance.T)
    if np.any(np.diag(covariance) < 0.0):
        raise V2NumericalUnavailable(f"{BLOCKER}:COVARIANCE_NEGATIVE_DIAGONAL")
    try:
        covariance_chol = np.linalg.cholesky(covariance)
        eigenvalues = np.linalg.eigvalsh(covariance)
    except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError, OverflowError) as error:
        raise V2NumericalUnavailable(f"{BLOCKER}:COVARIANCE_NOT_PD") from error
    if (
        not np.all(np.isfinite(covariance_chol))
        or not np.all(np.isfinite(eigenvalues))
        or np.min(eigenvalues) < -config.quadratic_form_tolerance
    ):
        raise V2NumericalUnavailable(f"{BLOCKER}:COVARIANCE_QUADRATIC_FORM")
    inverse_residual = _matrix_inf(hessian @ covariance - identity)
    if inverse_residual > config.covariance_residual_tolerance:
        raise V2NumericalUnavailable(f"{BLOCKER}:COVARIANCE_RESIDUAL")
    return covariance


def damped_newton(
    design_scaled: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    precision_scaled: np.ndarray,
    *,
    initial: np.ndarray | None = None,
    config: NewtonConfig = CONFIG,
) -> dict[str, Any]:
    config = validate_config(config)
    x, y, o, precision = validate_problem(
        design_scaled, labels, offsets, precision_scaled
    )
    gamma = np.zeros(x.shape[1]) if initial is None else _finite("INITIAL", initial).copy()
    if gamma.shape != (x.shape[1],):
        raise V2NumericalUnavailable(f"{BLOCKER}:INITIAL_SHAPE")
    trace: list[dict[str, Any]] = []
    for iteration in range(config.max_iterations + 1):
        objective, gradient, hessian = objective_gradient_hessian(
            gamma, x, y, o, precision
        )
        gradient_inf = float(np.max(np.abs(gradient)))
        if gradient_inf <= config.gradient_inf_tolerance:
            chol = _require_symmetric_positive_definite(hessian, config)
            covariance = _checked_covariance(hessian, chol, config)
            return {
                "status": "CONVERGED",
                "gamma": gamma,
                "objective": objective,
                "gradient_inf": gradient_inf,
                "hessian": hessian,
                "covariance_gamma": covariance,
                "iterations": iteration,
                "trace": tuple(trace),
                "trace_sha256": sha256(trace),
                "config_sha256": config_hash(config),
                "jitter_used": 0.0,
            }
        if iteration == config.max_iterations:
            raise V2NumericalUnavailable(f"{BLOCKER}:MAX_ITERATIONS")
        chol = _require_symmetric_positive_definite(hessian, config)
        direction = -_checked_cholesky_solve(hessian, chol, gradient, config)
        directional = float(np.dot(gradient, direction))
        if not np.all(np.isfinite(direction)) or not np.isfinite(directional) or directional >= 0.0:
            raise V2NumericalUnavailable(f"{BLOCKER}:NON_DESCENT")
        alpha = config.initial_alpha
        accepted = False
        candidate_objective = np.nan
        candidate_gradient: np.ndarray | None = None
        candidate_hessian: np.ndarray | None = None
        for backtrack in range(config.max_backtracks + 1):
            candidate = gamma + alpha * direction
            try:
                candidate_objective, candidate_gradient, candidate_hessian = objective_gradient_hessian(
                    candidate, x, y, o, precision
                )
            except V2NumericalUnavailable:
                candidate_objective = np.nan
            if np.isfinite(candidate_objective) and candidate_objective <= (
                objective + config.armijo_c1 * alpha * directional
            ):
                accepted = True
                break
            alpha *= config.shrink
        if not accepted:
            raise V2NumericalUnavailable(f"{BLOCKER}:ARMIJO_EXHAUSTED")
        step_inf = float(np.max(np.abs(alpha * direction)))
        trace.append({
            "iteration": iteration,
            "objective_hex": objective.hex(),
            "gradient_inf_hex": gradient_inf.hex(),
            "alpha_hex": alpha.hex(),
            "backtracks": backtrack,
            "step_inf_hex": step_inf.hex(),
            "candidate_objective_hex": float(candidate_objective).hex(),
        })
        gamma = candidate
        if candidate_gradient is None or candidate_hessian is None:
            raise V2NumericalUnavailable(f"{BLOCKER}:ACCEPTED_STATE_MISSING")
        if step_inf <= config.step_inf_tolerance:
            # The accepted point was mandatorily and fully reevaluated above.
            candidate_gradient_inf = float(np.max(np.abs(candidate_gradient)))
            if not np.isfinite(candidate_gradient_inf):
                raise V2NumericalUnavailable(f"{BLOCKER}:NONFINITE_GRADIENT")
            if candidate_gradient_inf > config.gradient_inf_tolerance:
                raise V2NumericalUnavailable(f"{BLOCKER}:STAGNATION")
    raise V2NumericalUnavailable(f"{BLOCKER}:UNREACHABLE")
