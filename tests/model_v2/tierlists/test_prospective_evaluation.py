"""Tests for the development-only prospective tier-list evaluator."""

from __future__ import annotations

from datetime import datetime, timezone
import warnings

import numpy as np
import pytest

from lol_kills.v2.tierlists.prospective_evaluation import (
    EvaluationRow,
    TierListEvaluationError,
    _bootstrap_deltas,
    _feature_frame,
    _fit_logistic,
    _linear_predictor,
    _tier_value,
)


def _row(
    row_id: str,
    *,
    label: int,
    cluster: str,
    scope: str = "LEC",
    tier_feature: float = 0.1,
) -> EvaluationRow:
    return EvaluationRow(
        row_id=row_id,
        map_id=row_id.split(":")[0],
        event_start=datetime(2026, 1, 1, tzinfo=timezone.utc),
        scope=scope,
        patch="16.1",
        role="mid",
        champion="Ahri",
        side="blue" if label else "red",
        label=label,
        strength_logit=0.2 if label else -0.2,
        draft_other_logit=0.1 if label else -0.1,
        tier_feature=tier_feature,
        tier_value_pp=tier_feature * 100.0,
        played_champion_count=5,
        dependence_cluster_id=cluster,
    )


def test_tier_value_and_feature_frame_are_finite() -> None:
    assert np.isfinite(_tier_value(0.4, -0.1, 1.0))
    rows = [_row("m1:blue:mid", label=1, cluster="a"), _row("m2:red:mid", label=0, cluster="b")]

    frame = _feature_frame(rows, include_tier=True)

    assert list(frame.columns)[-1] == "tier_feature"
    assert frame.shape == (2, 9)
    assert np.isfinite(frame.to_numpy(dtype=float)).all()


def test_logistic_fit_handles_redundant_scope_indicators_without_warnings() -> None:
    features = np.asarray(
        [[1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    labels = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=float)
    weights = np.ones(4, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = _fit_logistic(features, labels, weights)

    probabilities = model.predict_proba(features)
    assert probabilities.shape == (4, 2)
    assert np.isfinite(probabilities).all()
    assert np.all((probabilities > 0.0) & (probabilities < 1.0))


def test_numeric_failure_is_fail_closed() -> None:
    with pytest.raises(TierListEvaluationError, match="non-finite"):
        _linear_predictor(np.asarray([[np.inf]], dtype=float), np.asarray([1.0], dtype=float))


def test_cluster_bootstrap_returns_finite_intervals() -> None:
    rows = [
        _row("m1:blue:mid", label=1, cluster="a"),
        _row("m1:red:mid", label=0, cluster="a"),
        _row("m2:blue:mid", label=1, cluster="b"),
        _row("m2:red:mid", label=0, cluster="b"),
    ]
    baseline = np.asarray([0.5, 0.5, 0.5, 0.5], dtype=float)
    candidate = np.asarray([0.6, 0.4, 0.55, 0.45], dtype=float)

    intervals = _bootstrap_deltas(
        rows,
        baseline,
        candidate,
        replicates=100,
        seed=20260808,
    )

    assert set(intervals) == {"log_loss", "brier"}
    for bounds in intervals.values():
        assert bounds["lower_95"] <= bounds["upper_95"]
        assert np.isfinite(list(bounds.values())).all()
