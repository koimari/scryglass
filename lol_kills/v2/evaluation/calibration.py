"""Proper likelihood calibration and registered symmetric transform selection."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression

from .types import CalibrationState, MatchPrediction, canonical_sha256


CALIBRATION_FAMILIES = (
    "identity",
    "symmetric_temperature",
    "symmetrized_platt",
    "symmetrized_beta",
    "symmetrized_bounded_isotonic",
)
CALIBRATION_CODE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFAULT_CONFIG = {
    "boundary_epsilon": 1e-9,
    "minimum_inner_fit": 4,
    "minimum_inner_validation": 2,
    "paired_uncertainty_z": 1.96,
    "families": list(CALIBRATION_FAMILIES),
}
DEFAULT_CONFIG_SHA256 = canonical_sha256(DEFAULT_CONFIG)


def _unavailable(model_sha256: str, reason: str, support: int, *, kind: str = "affine_logit") -> CalibrationState:
    return CalibrationState(
        kind=kind,
        intercept=0.0,
        slope=1.0,
        model_sha256=model_sha256,
        status="unavailable",
        reason=reason,
        support=support,
        code_sha256=CALIBRATION_CODE_SHA256,
        config_sha256=DEFAULT_CONFIG_SHA256,
    )


def fit_logistic_calibration(
    logits: Sequence[float],
    labels: Sequence[int],
    *,
    offsets: Sequence[float] | None = None,
    model_sha256: str = "",
) -> CalibrationState:
    z = np.asarray(logits, dtype=float)
    y = np.asarray(labels, dtype=float)
    offset = np.zeros_like(z) if offsets is None else np.asarray(offsets, dtype=float)
    n = int(z.size)
    if y.size != n or offset.size != n:
        raise ValueError("logits, labels, and offsets must align")
    if n < 3:
        return _unavailable(model_sha256, "insufficient_support", n)
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(offset)):
        return _unavailable(model_sha256, "nonfinite_input", n)
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        return _unavailable(model_sha256, "nonbinary_outcome", n)
    if np.unique(y).size < 2:
        return _unavailable(model_sha256, "constant_outcome", n)
    if float(np.ptp(z)) <= 1e-14:
        return _unavailable(model_sha256, "constant_logit", n)
    design = np.column_stack((np.ones(n), z))

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offset + design @ theta
        value = float(np.sum(np.logaddexp(0.0, eta) - y * eta))
        gradient = design.T @ (expit(eta) - y)
        return value, gradient

    result = minimize(
        lambda theta: objective(theta)[0],
        np.array([0.0, 1.0]),
        jac=lambda theta: objective(theta)[1],
        method="BFGS",
        options={"gtol": 1e-10, "maxiter": 1000},
    )
    theta = np.asarray(result.x, dtype=float)
    eta = offset + design @ theta
    weights = expit(eta) * expit(-eta)
    information = design.T @ (weights[:, None] * design)
    eigenvalues = np.linalg.eigvalsh(information)
    gradient_norm = float(np.linalg.norm(objective(theta)[1], ord=np.inf))
    if (
        not np.all(np.isfinite(theta))
        or not np.all(np.isfinite(information))
        or float(eigenvalues.min()) <= 1e-10
        or float(np.max(np.abs(theta))) >= 25.0
        or gradient_norm > 1e-6
        or (not result.success and gradient_norm > 1e-7)
    ):
        reason = "separation_or_singular_hessian" if eigenvalues.min() <= 1e-10 or np.max(np.abs(theta)) >= 25 else "nonconvergence"
        return _unavailable(model_sha256, reason, n)
    covariance = np.linalg.inv(information)
    standard_errors = np.sqrt(np.diag(covariance))
    return CalibrationState(
        kind="affine_logit",
        intercept=float(theta[0]),
        slope=float(theta[1]),
        model_sha256=model_sha256,
        status="ok",
        covariance=tuple(tuple(float(v) for v in row) for row in covariance),
        standard_errors=tuple(float(v) for v in standard_errors),
        support=n,
        parameters={
            "offset_sha256": canonical_sha256([float(v) for v in offset]),
            "gradient_inf_norm": gradient_norm,
            "information_eigenvalues": [float(v) for v in eigenvalues],
        },
        code_sha256=CALIBRATION_CODE_SHA256,
        config_sha256=DEFAULT_CONFIG_SHA256,
    )


def fit_calibration(
    predictions: Iterable[MatchPrediction],
    labels: Iterable[int],
    model_sha256: str,
    *,
    kind: str = "affine_logit",
) -> CalibrationState:
    preds = tuple(predictions)
    ys = tuple(int(v) for v in labels)
    if len(preds) != len(ys):
        raise ValueError("predictions and labels must align")
    logits = [
        float(pred.raw_logit)
        if pred.raw_logit is not None
        else float(math.log(_clamp_probability(pred.raw_probability) / (1 - _clamp_probability(pred.raw_probability))))
        for pred in preds
    ]
    state = fit_logistic_calibration(logits, ys, model_sha256=model_sha256)
    return replace(state, kind=kind)


def _symmetrize(base, z: np.ndarray) -> np.ndarray:
    return (base(z) + 1.0 - base(-z)) / 2.0


def apply_registered_transform(
    logits: Sequence[float],
    family: str,
    parameters: Mapping[str, Any],
    *,
    boundary_epsilon: float = 1e-9,
) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    if not np.all(np.isfinite(z)):
        raise ValueError("transform inputs must be finite")
    if family not in CALIBRATION_FAMILIES:
        raise ValueError("unregistered calibration family")
    eps = float(boundary_epsilon)
    if not 0 < eps < .5:
        raise ValueError("invalid finite boundary epsilon")
    if family == "identity":
        values = expit(z)
    elif family == "symmetric_temperature":
        slope = float(parameters["slope"])
        if slope < 0:
            raise ValueError("temperature slope must be nonnegative")
        values = expit(slope * z)
    elif family == "symmetrized_platt":
        slope = float(parameters["slope"])
        intercept = float(parameters.get("intercept", 0.0))
        if slope < 0:
            raise ValueError("Platt slope must be nonnegative")
        values = _symmetrize(lambda x: expit(intercept + slope * x), z)
    elif family == "symmetrized_beta":
        a = float(parameters["a"])
        b = float(parameters["b"])
        intercept = float(parameters.get("intercept", 0.0))
        if a < 0 or b < 0:
            raise ValueError("beta shapes must be nonnegative")
        def base(x):
            p = np.clip(expit(x), eps, 1 - eps)
            return expit(intercept + a * np.log(p) - b * np.log1p(-p))
        values = _symmetrize(base, z)
    else:
        knots = np.asarray(parameters["knots"], dtype=float)
        levels = np.asarray(parameters["levels"], dtype=float)
        if knots.size < 2 or knots.size != levels.size or np.any(np.diff(knots) <= 0) or np.any(np.diff(levels) < 0):
            raise ValueError("isotonic knots/levels are invalid")
        def base(x):
            return np.interp(x, knots, levels, left=levels[0], right=levels[-1])
        values = _symmetrize(base, z)
    return np.clip(values, eps, 1 - eps)


def _fit_family(family: str, z: np.ndarray, y: np.ndarray, eps: float) -> Mapping[str, Any]:
    if family == "identity":
        return {}
    if family == "symmetric_temperature":
        result = minimize(lambda q: np.sum(np.logaddexp(0, np.exp(q[0]) * z) - y * np.exp(q[0]) * z), [0.0])
        return {"slope": float(np.exp(result.x[0]))}
    if family == "symmetrized_bounded_isotonic":
        iso = IsotonicRegression(y_min=eps, y_max=1-eps, out_of_bounds="clip")
        levels = iso.fit_transform(z, y)
        order = np.argsort(z)
        knots, indices = np.unique(z[order], return_index=True)
        return {"knots": [float(v) for v in knots], "levels": [float(v) for v in levels[order][indices]]}
    if family == "symmetrized_platt":
        def loss(q):
            p = apply_registered_transform(z, family, {"intercept": q[0], "slope": np.exp(q[1])}, boundary_epsilon=eps)
            return -float(np.sum(y*np.log(p)+(1-y)*np.log1p(-p)))
        result = minimize(loss, [0.0, 0.0])
        return {"intercept": float(result.x[0]), "slope": float(np.exp(result.x[1]))}
    def loss(q):
        params = {"intercept": q[0], "a": np.exp(q[1]), "b": np.exp(q[2])}
        p = apply_registered_transform(z, family, params, boundary_epsilon=eps)
        return -float(np.sum(y*np.log(p)+(1-y)*np.log1p(-p)))
    result = minimize(loss, [0.0, 0.0, 0.0])
    return {"intercept": float(result.x[0]), "a": float(np.exp(result.x[1])), "b": float(np.exp(result.x[2]))}


def select_nested_transform(
    logits: Sequence[float],
    labels: Sequence[int],
    series_ids: Sequence[str],
    event_order: Sequence[Any],
    row_ids: Sequence[str],
    *,
    boundary_epsilon: float = 1e-9,
) -> CalibrationState:
    n = len(logits)
    if not (n == len(labels) == len(series_ids) == len(event_order) == len(row_ids)):
        raise ValueError("nested calibration inputs must align")
    order = sorted(range(n), key=lambda i: (event_order[i], series_ids[i], row_ids[i]))
    ordered_series: list[str] = []
    for i in order:
        if series_ids[i] not in ordered_series:
            ordered_series.append(series_ids[i])
    if len(ordered_series) < 3:
        return _unavailable("", "insufficient_nested_series_support", n, kind="nested_selection")
    cut = max(1, int(len(ordered_series) * .67))
    fit_series = set(ordered_series[:cut])
    fit_idx = [i for i in order if series_ids[i] in fit_series]
    val_idx = [i for i in order if series_ids[i] not in fit_series]
    if len(fit_idx) < 4 or len(val_idx) < 2:
        return _unavailable("", "insufficient_nested_support", n, kind="nested_selection")
    z, y = np.asarray(logits, float), np.asarray(labels, float)
    losses: list[tuple[float, int, str]] = []
    for rank, family in enumerate(CALIBRATION_FAMILIES):
        params = _fit_family(family, z[fit_idx], y[fit_idx], boundary_epsilon)
        p = apply_registered_transform(z[val_idx], family, params, boundary_epsilon=boundary_epsilon)
        losses.append((-float(np.mean(y[val_idx]*np.log(p)+(1-y[val_idx])*np.log1p(-p))), rank, family))
    best_loss = min(item[0] for item in losses)
    paired_se = float(np.std([item[0] for item in losses], ddof=1) / math.sqrt(len(losses))) if len(losses) > 1 else 0.0
    eligible = [item for item in losses if item[0] <= best_loss + 1.96 * paired_se]
    _, _, selected = min(eligible, key=lambda item: item[1])
    params = _fit_family(selected, z, y, boundary_epsilon)
    row_hash = canonical_sha256([row_ids[i] for i in order])
    unsigned = {
        "family": selected,
        "parameters": params,
        "boundary_epsilon": boundary_epsilon,
        "symmetry": "g(-z)=1-g(z)",
        "calibration_row_sha256": row_hash,
        "inner_fit_series": ordered_series[:cut],
        "inner_validation_series": ordered_series[cut:],
        "candidate_losses": {family: loss for loss, _, family in losses},
        "code_sha256": CALIBRATION_CODE_SHA256,
        "config_sha256": DEFAULT_CONFIG_SHA256,
    }
    return CalibrationState(
        kind=selected,
        intercept=float(params.get("intercept", 0.0)),
        slope=float(params.get("slope", 1.0)),
        model_sha256="",
        status="ok",
        support=n,
        parameters=params,
        boundary_epsilon=boundary_epsilon,
        symmetry="g(-z)=1-g(z)",
        calibration_row_sha256=row_hash,
        selection_sha256=canonical_sha256(unsigned),
        code_sha256=CALIBRATION_CODE_SHA256,
        config_sha256=DEFAULT_CONFIG_SHA256,
    )


def calibrate_logits(predictions: Iterable[MatchPrediction], state: CalibrationState) -> tuple[MatchPrediction, ...]:
    predictions = tuple(predictions)
    if state.status != "ok":
        return predictions
    logits = [
        pred.raw_logit if pred.raw_logit is not None else math.log(_clamp_probability(pred.raw_probability)/(1-_clamp_probability(pred.raw_probability)))
        for pred in predictions
    ]
    if state.kind in CALIBRATION_FAMILIES:
        mapped = apply_registered_transform(logits, state.kind, state.parameters, boundary_epsilon=state.boundary_epsilon)
    else:
        mapped = expit(state.intercept + state.slope * np.asarray(logits))
    return tuple(replace(pred, calibrated_probability=float(prob)) for pred, prob in zip(predictions, mapped))


def _clamp_probability(probability: float) -> float:
    return float(max(1e-12, min(1 - 1e-12, probability)))
