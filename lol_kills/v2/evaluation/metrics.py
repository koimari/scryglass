"""Evaluation metrics used by L2 independent benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Mapping

import numpy as np


def _to_numpy(values: Sequence[float], *, clip01: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if clip01:
        arr = np.clip(arr, 1e-12, 1 - 1e-12)
    return arr


def log_loss(labels: Sequence[int], probs: Sequence[float]) -> float:
    y = _to_numpy(labels)
    p = _to_numpy(probs, clip01=True)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def brier_score(labels: Sequence[int], probs: Sequence[float]) -> float:
    y = _to_numpy(labels)
    p = _to_numpy(probs, clip01=False)
    return float(np.mean((y - p) ** 2))


@dataclass(frozen=True)
class CalibrationDiagnostic:
    """Typed live calibration diagnostic.

    An unavailable diagnostic deliberately carries no numerical estimate.  In
    particular, the ideal values (0, 1) are never used as a failure default.
    """

    status: str
    reason: str
    support: int
    intercept: float | None
    slope: float | None


def calibration_intercept_and_slope(
    labels: Sequence[int], probs: Sequence[float]
) -> CalibrationDiagnostic:
    p = _to_numpy(probs, clip01=True)
    logit = np.log(p / (1 - p))
    from .calibration import fit_logistic_calibration
    result = fit_logistic_calibration(logit, labels, model_sha256="metric-diagnostic")
    if result.status != "ok":
        return CalibrationDiagnostic(
            status="unavailable",
            reason=result.reason or "calibration_fit_unavailable",
            support=result.support,
            intercept=None,
            slope=None,
        )
    return CalibrationDiagnostic(
        status="ok",
        reason="",
        support=result.support,
        intercept=result.intercept,
        slope=result.slope,
    )


def expected_calibration_error(
    labels: Sequence[int],
    probs: Sequence[float],
    *,
    bins: int = 10,
) -> float:
    n = min(len(labels), len(probs))
    if n == 0:
        return 0.0
    y = _to_numpy(labels)[:n]
    p = _to_numpy(probs, clip01=False)[:n]
    bucket_id = np.floor(np.clip(p, 0.0, 1.0 - 1e-12) * bins).astype(int)
    bucket_id = np.clip(bucket_id, 0, bins - 1)
    weighted = 0.0
    for b in range(bins):
        mask = bucket_id == b
        if not bool(np.any(mask)):
            continue
        weighted += mask.mean() * abs(float(p[mask].mean() - y[mask].mean()))
    return float(weighted)


def auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    y = _to_numpy(labels)
    s = _to_numpy(scores, clip01=False)
    pos = s[y == 1]
    neg = s[y == 0]
    if pos.size == 0 or neg.size == 0:
        return 0.5
    # pairwise U-statistic style AUC
    greater = (pos[:, None] > neg[None, :]).mean()
    equal = (pos[:, None] == neg[None, :]).mean() * 0.5
    return float(greater + equal)


def macro_region_log_loss(metrics_by_region: Mapping[str, float]) -> float:
    if not metrics_by_region:
        return 0.0
    return float(np.mean(list(metrics_by_region.values())))


@dataclass(frozen=True)
class MetricSuite:
    log_loss: float
    brier: float
    ece: float
    auc: float
    calibration_status: str
    calibration_reason: str
    calibration_support: int
    calibration_intercept: float | None
    calibration_slope: float | None
