"""Predeclared, leakage-safe evaluation for Scryglass probability models.

This module does not fit a preferred model.  It defines the evidence contract
that every candidate must satisfy before a final-test score can be compared.
Lower proper scores are better throughout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_PREDICTION_COLUMNS = frozenset(
    {
        "prediction_id",
        "model_id",
        "model_version",
        "event_id",
        "series_id",
        "prediction_time",
        "event_time",
        "data_as_of",
        "outcome",
        "probability",
        "split",
        "league",
        "patch",
        "roster_state_id",
    }
)
ALLOWED_SPLITS = frozenset({"train", "validation", "test"})
SUPPORTED_SCORES = frozenset({"brier", "log_loss"})


class PredictionLedgerError(ValueError):
    """Raised when a prediction ledger cannot support scientific evaluation."""


@dataclass(frozen=True)
class TournamentSpec:
    """Frozen selection rules for one estimand and final-test population."""

    estimand_id: str
    primary_score: str
    minimum_test_events: int
    bootstrap_replicates: int
    moving_block_size: int
    alpha: float = 0.05
    noninferiority_margin: float = 0.0
    random_seed: int = 0

    def validate(self) -> None:
        if not self.estimand_id.strip():
            raise ValueError("estimand_id must be non-empty")
        if self.primary_score not in SUPPORTED_SCORES:
            raise ValueError(
                f"primary_score must be one of {sorted(SUPPORTED_SCORES)}"
            )
        if self.minimum_test_events < 1:
            raise ValueError("minimum_test_events must be positive")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        if self.moving_block_size < 1:
            raise ValueError("moving_block_size must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        if self.noninferiority_margin < 0.0:
            raise ValueError("noninferiority_margin cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


def _as_utc(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_datetime(frame[column], errors="coerce", utc=True)
    if values.isna().any():
        bad = frame.loc[values.isna(), "prediction_id"].astype(str).head(5).tolist()
        raise PredictionLedgerError(
            f"{column} contains invalid timestamps; examples={bad}"
        )
    return values


def validate_prediction_ledger(frame: pd.DataFrame) -> dict[str, Any]:
    """Validate provenance, time ordering, probability bounds, and event grain."""

    if frame is None or frame.empty:
        raise PredictionLedgerError("prediction ledger is empty")
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        raise PredictionLedgerError(f"prediction ledger missing columns: {missing}")

    ledger = frame.copy()
    identity_columns = [
        "prediction_id",
        "model_id",
        "model_version",
        "event_id",
        "series_id",
        "league",
        "patch",
        "roster_state_id",
    ]
    null_identity = ledger[identity_columns].isna().any(axis=1)
    blank_identity = (
        ledger[identity_columns].astype(str).apply(lambda col: col.str.strip().eq(""))
    ).any(axis=1)
    if (null_identity | blank_identity).any():
        examples = (
            ledger.loc[null_identity | blank_identity, "prediction_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise PredictionLedgerError(
            f"prediction identity/provenance fields are blank; examples={examples}"
        )

    if ledger["prediction_id"].duplicated().any():
        duplicates = (
            ledger.loc[ledger["prediction_id"].duplicated(False), "prediction_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise PredictionLedgerError(
            f"prediction_id must be unique; examples={duplicates}"
        )
    model_event_key = ["model_id", "model_version", "event_id"]
    if ledger.duplicated(model_event_key, keep=False).any():
        examples = (
            ledger.loc[ledger.duplicated(model_event_key, False), model_event_key]
            .head(5)
            .to_dict("records")
        )
        raise PredictionLedgerError(
            f"one prediction is allowed per model-version/event; examples={examples}"
        )

    prediction_time = _as_utc(ledger, "prediction_time")
    event_time = _as_utc(ledger, "event_time")
    data_as_of = _as_utc(ledger, "data_as_of")
    future_data = data_as_of > prediction_time
    late_prediction = prediction_time > event_time
    if future_data.any() or late_prediction.any():
        examples = (
            ledger.loc[future_data | late_prediction, "prediction_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise PredictionLedgerError(
            "temporal leakage: require data_as_of <= prediction_time <= event_time; "
            f"examples={examples}"
        )

    probability = pd.to_numeric(ledger["probability"], errors="coerce")
    invalid_probability = (
        probability.isna()
        | ~np.isfinite(probability.to_numpy(dtype=float))
        | probability.lt(0.0)
        | probability.gt(1.0)
    )
    if invalid_probability.any():
        examples = (
            ledger.loc[invalid_probability, "prediction_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise PredictionLedgerError(
            f"probability must be finite and within [0, 1]; examples={examples}"
        )

    outcome = pd.to_numeric(ledger["outcome"], errors="coerce")
    invalid_outcome = outcome.isna() | ~outcome.isin([0, 1])
    if invalid_outcome.any():
        examples = (
            ledger.loc[invalid_outcome, "prediction_id"]
            .astype(str)
            .head(5)
            .tolist()
        )
        raise PredictionLedgerError(
            f"outcome must be binary; examples={examples}"
        )

    invalid_split = ~ledger["split"].isin(ALLOWED_SPLITS)
    if invalid_split.any():
        values = sorted(ledger.loc[invalid_split, "split"].astype(str).unique())
        raise PredictionLedgerError(
            f"split must be one of {sorted(ALLOWED_SPLITS)}; found={values}"
        )

    event_consistency = ledger.assign(
        _outcome=outcome.astype(int),
        _event_time=event_time,
        _series_id=ledger["series_id"].astype(str),
    ).groupby("event_id", sort=False)[["_outcome", "_event_time", "_series_id"]].nunique()
    inconsistent_events = event_consistency.gt(1).any(axis=1)
    if inconsistent_events.any():
        examples = inconsistent_events[inconsistent_events].index.astype(str).tolist()[:5]
        raise PredictionLedgerError(
            "event outcome, time, and series must agree across models; "
            f"examples={examples}"
        )

    return {
        "ok": True,
        "rows": int(len(ledger)),
        "events": int(ledger["event_id"].nunique()),
        "series": int(ledger["series_id"].nunique()),
        "models": int(
            ledger[["model_id", "model_version"]].drop_duplicates().shape[0]
        ),
        "date_min": event_time.min().isoformat(),
        "date_max": event_time.max().isoformat(),
        "splits": {
            str(name): int(count)
            for name, count in ledger["split"].value_counts().sort_index().items()
        },
    }


def proper_score_vector(
    outcome: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
    score: str,
) -> np.ndarray:
    """Return one proper-score contribution per binary event."""

    if score not in SUPPORTED_SCORES:
        raise ValueError(f"unsupported score: {score}")
    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("outcome and probability must be non-empty 1D arrays")
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("outcome must contain only finite zero/one values")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must be finite and within [0, 1]")
    if score == "brier":
        return np.square(p - y)
    clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    return -(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))


def pav_calibrated_probabilities(
    outcome: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> np.ndarray:
    """Pool-adjacent-violators fit with deterministic handling of tied forecasts."""

    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    proper_score_vector(y, p, "brier")

    order = np.argsort(p, kind="mergesort")
    sorted_p = p[order]
    sorted_y = y[order]
    unique_p, first, counts = np.unique(
        sorted_p, return_index=True, return_counts=True
    )
    sums = np.add.reduceat(sorted_y, first)

    blocks: list[dict[str, float | int]] = []
    for group_index, (weight, total) in enumerate(zip(counts, sums)):
        blocks.append(
            {
                "start": group_index,
                "end": group_index,
                "weight": int(weight),
                "mean": float(total / weight),
            }
        )
        while len(blocks) >= 2 and float(blocks[-2]["mean"]) > float(
            blocks[-1]["mean"]
        ):
            right = blocks.pop()
            left = blocks.pop()
            merged_weight = int(left["weight"]) + int(right["weight"])
            merged_mean = (
                float(left["mean"]) * int(left["weight"])
                + float(right["mean"]) * int(right["weight"])
            ) / merged_weight
            blocks.append(
                {
                    "start": int(left["start"]),
                    "end": int(right["end"]),
                    "weight": merged_weight,
                    "mean": merged_mean,
                }
            )

    group_fit = np.empty(len(unique_p), dtype=float)
    for block in blocks:
        group_fit[int(block["start"]) : int(block["end"]) + 1] = float(
            block["mean"]
        )
    sorted_fit = np.repeat(group_fit, counts)
    fitted = np.empty_like(sorted_fit)
    fitted[order] = sorted_fit
    return fitted


def corp_calibration_diagnostics(
    outcome: np.ndarray | pd.Series,
    probability: np.ndarray | pd.Series,
) -> dict[str, Any]:
    """Reproducible PAV calibration and exact Brier score decomposition."""

    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    fitted = pav_calibrated_probabilities(y, p)
    raw_brier = float(proper_score_vector(y, p, "brier").mean())
    recalibrated_brier = float(proper_score_vector(y, fitted, "brier").mean())
    uncertainty = float(y.mean() * (1.0 - y.mean()))
    miscalibration = max(raw_brier - recalibrated_brier, 0.0)
    discrimination = uncertainty - recalibrated_brier
    return {
        "n": int(len(y)),
        "brier": raw_brier,
        "recalibrated_brier": recalibrated_brier,
        "miscalibration": miscalibration,
        "discrimination": discrimination,
        "uncertainty": uncertainty,
        "decomposition_residual": float(
            raw_brier - (miscalibration - discrimination + uncertainty)
        ),
        "fitted_probability": fitted,
    }


def paired_moving_block_comparison(
    frame: pd.DataFrame,
    *,
    candidate_model_id: str,
    baseline_model_id: str,
    spec: TournamentSpec,
) -> dict[str, Any]:
    """Compare aligned test predictions with temporal moving-block uncertainty."""

    spec.validate()
    validate_prediction_ledger(frame)
    test = frame.loc[frame["split"].eq("test")].copy()
    if test.empty:
        raise PredictionLedgerError("prediction ledger has no test rows")

    versions = (
        test.groupby("model_id", sort=False)["model_version"].nunique().to_dict()
    )
    for model_id in (candidate_model_id, baseline_model_id):
        if model_id not in versions:
            raise PredictionLedgerError(f"model is absent from test rows: {model_id}")
        if versions[model_id] != 1:
            raise PredictionLedgerError(
                f"model comparison requires one frozen version: {model_id}"
            )

    selected = test.loc[
        test["model_id"].isin([candidate_model_id, baseline_model_id])
    ].copy()
    event_columns = ["event_id", "outcome", "event_time", "series_id"]
    probabilities = selected.pivot(
        index=event_columns,
        columns="model_id",
        values="probability",
    ).reset_index()
    if probabilities[[candidate_model_id, baseline_model_id]].isna().any().any():
        raise PredictionLedgerError(
            "candidate and baseline must predict the identical test events"
        )
    if len(probabilities) < spec.minimum_test_events:
        raise PredictionLedgerError(
            f"test population has {len(probabilities)} events; "
            f"minimum is {spec.minimum_test_events}"
        )

    candidate_score = proper_score_vector(
        probabilities["outcome"],
        probabilities[candidate_model_id],
        spec.primary_score,
    )
    baseline_score = proper_score_vector(
        probabilities["outcome"],
        probabilities[baseline_model_id],
        spec.primary_score,
    )
    probabilities["_score_delta"] = candidate_score - baseline_score
    probabilities["_event_time"] = pd.to_datetime(
        probabilities["event_time"], errors="raise", utc=True
    )
    series_order = (
        probabilities.groupby("series_id", as_index=False)
        .agg(event_time=("_event_time", "min"))
        .sort_values(["event_time", "series_id"], kind="mergesort")
        ["series_id"]
        .tolist()
    )
    series_deltas = [
        probabilities.loc[
            probabilities["series_id"].eq(series_id), "_score_delta"
        ].to_numpy(dtype=float)
        for series_id in series_order
    ]
    n_clusters = len(series_deltas)
    block_size = min(spec.moving_block_size, n_clusters)
    rng = np.random.default_rng(spec.random_seed)
    bootstrap = np.empty(spec.bootstrap_replicates, dtype=float)
    blocks_needed = int(np.ceil(n_clusters / block_size))
    offsets = np.arange(block_size)
    for replicate in range(spec.bootstrap_replicates):
        starts = rng.integers(0, n_clusters, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n_clusters).ravel()
        sampled = np.concatenate(
            [series_deltas[index] for index in indices[:n_clusters]]
        )
        bootstrap[replicate] = float(sampled.mean())

    point = float(probabilities["_score_delta"].mean())
    low, high = np.quantile(
        bootstrap, [spec.alpha / 2.0, 1.0 - spec.alpha / 2.0]
    )
    if high < 0.0:
        decision = "superior"
    elif high <= spec.noninferiority_margin:
        decision = "noninferior"
    elif low > spec.noninferiority_margin:
        decision = "inferior"
    else:
        decision = "inconclusive"

    return {
        "estimand_id": spec.estimand_id,
        "candidate_model_id": candidate_model_id,
        "baseline_model_id": baseline_model_id,
        "primary_score": spec.primary_score,
        "events": int(len(probabilities)),
        "series_clusters": int(n_clusters),
        "candidate_score": float(candidate_score.mean()),
        "baseline_score": float(baseline_score.mean()),
        "candidate_minus_baseline": point,
        "confidence_level": float(1.0 - spec.alpha),
        "confidence_interval": [float(low), float(high)],
        "noninferiority_margin": float(spec.noninferiority_margin),
        "decision": decision,
        "bootstrap": {
            "method": (
                "paired circular moving-block bootstrap over ordered series, "
                "with event-weighted proper scores"
            ),
            "replicates": int(spec.bootstrap_replicates),
            "block_size_series": int(block_size),
            "seed": int(spec.random_seed),
        },
        "candidate_calibration": {
            key: value
            for key, value in corp_calibration_diagnostics(
                probabilities["outcome"], probabilities[candidate_model_id]
            ).items()
            if key != "fitted_probability"
        },
        "baseline_calibration": {
            key: value
            for key, value in corp_calibration_diagnostics(
                probabilities["outcome"], probabilities[baseline_model_id]
            ).items()
            if key != "fitted_probability"
        },
        "spec": spec.to_dict(),
    }
