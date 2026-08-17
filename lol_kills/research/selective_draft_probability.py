"""Evaluate a side-symmetric confidence gate for Draft win probability.

This module consumes out-of-fold component predictions. It trains the
confidence gate on one earlier selection window. Outcomes from later windows
are used only for the reported evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


SCHEMA_VERSION = "scryglass:selective-draft-probability-evaluation:v1"
CANDIDATE_SCHEMA_VERSION = "scryglass:selective-draft-probability-candidate:v1"
PREDICTORS = ("quantum", "roster", "identity", "development_composite")
CONFIDENCE_COLUMNS = (
    "disagreement",
    "prediction_range",
    "ensemble_margin",
    *(f"margin_{name}" for name in PREDICTORS),
    *(f"distance_{left}_{right}" for index, left in enumerate(PREDICTORS) for right in PREDICTORS[index + 1 :]),
    "team_sigma_pair_scaled",
    "player_sigma_pair_scaled",
    "player_known_fraction_min",
    "history_unique_player_maps_min",
    "availability_player_exact_performance",
    "availability_exact_ally_enemy_pairs",
    "availability_checkpoint_forecasts",
    "availability_parity_conditioned_performance",
    "availability_regional_draft_atoms",
)


class SelectiveDraftProbabilityError(ValueError):
    """Raised when a selective evaluation input fails closed."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _prediction_frame(path: Path, name: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    probability = "probability" if "probability" in frame else "p"
    required = {"game_uid", probability}
    if not required.issubset(frame.columns):
        raise SelectiveDraftProbabilityError(f"{name} prediction columns are incomplete")
    columns = ["game_uid", probability]
    if "y" in frame:
        columns.insert(1, "y")
    output = frame[columns].copy()
    output = output.rename(columns={"y": f"y_{name}", probability: name})
    output["game_uid"] = output["game_uid"].astype(str)
    if output["game_uid"].duplicated().any():
        raise SelectiveDraftProbabilityError(f"{name} predictions contain duplicate games")
    values = output[name].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
        raise SelectiveDraftProbabilityError(f"{name} probabilities are invalid")
    return output


def _validated_predictor_weights(
    weights: Mapping[str, float] | None,
) -> dict[str, float]:
    resolved = (
        {name: 0.25 for name in PREDICTORS}
        if weights is None
        else {str(name): float(value) for name, value in weights.items()}
    )
    if (
        set(resolved) != set(PREDICTORS)
        or any(not math.isfinite(value) or value < 0 for value in resolved.values())
        or not math.isclose(sum(resolved.values()), 1.0, abs_tol=1e-12)
        or sum(value > 0 for value in resolved.values()) < 2
    ):
        raise SelectiveDraftProbabilityError("predictor weights are invalid")
    return {name: resolved[name] for name in PREDICTORS}


def _with_probability_confidence_features(
    frame: pd.DataFrame,
    *,
    predictor_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Derive every predictor-based confidence field from fixed probabilities."""

    output = frame.copy()
    probabilities = output[list(PREDICTORS)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities <= 0) | (probabilities >= 1)
    ):
        raise SelectiveDraftProbabilityError("ensemble probabilities are invalid")
    weights = _validated_predictor_weights(predictor_weights)
    output["ensemble_probability"] = np.sum(
        probabilities
        * np.asarray([weights[name] for name in PREDICTORS], dtype=float),
        axis=1,
    )
    output["disagreement"] = probabilities.std(axis=1)
    output["prediction_range"] = probabilities.max(axis=1) - probabilities.min(axis=1)
    output["ensemble_margin"] = np.abs(output["ensemble_probability"] - 0.5)
    for name in PREDICTORS:
        output[f"margin_{name}"] = np.abs(output[name] - 0.5)
    for index, left in enumerate(PREDICTORS):
        for right in PREDICTORS[index + 1 :]:
            output[f"distance_{left}_{right}"] = np.abs(output[left] - output[right])
    if not np.isfinite(output[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float)).all():
        raise SelectiveDraftProbabilityError("confidence features are not finite")
    return output


def load_evaluation_frame(
    *,
    matrix_path: Path,
    prediction_paths: Mapping[str, Path],
    predictor_weights: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    if set(prediction_paths) != set(PREDICTORS):
        raise SelectiveDraftProbabilityError("the four fixed predictors are required")
    matrix = pd.read_parquet(matrix_path)
    required = {
        "game_uid",
        "series_id",
        "date",
        "league",
        "source_patch",
        "y",
        *CONFIDENCE_COLUMNS[-9:],
    }
    if not required.issubset(matrix.columns):
        raise SelectiveDraftProbabilityError("matrix confidence columns are incomplete")
    frame = matrix[list(required)].copy()
    frame["game_uid"] = frame["game_uid"].astype(str)
    if frame["game_uid"].duplicated().any():
        raise SelectiveDraftProbabilityError("matrix contains duplicate games")
    for name in PREDICTORS:
        frame = frame.merge(
            _prediction_frame(prediction_paths[name], name),
            on="game_uid",
            how="inner",
            validate="one_to_one",
        )
        outcome_column = f"y_{name}"
        if outcome_column in frame:
            if not np.array_equal(
                frame["y"].to_numpy(), frame[outcome_column].to_numpy()
            ):
                raise SelectiveDraftProbabilityError(
                    f"{name} outcomes do not match the matrix"
                )
            frame = frame.drop(columns=outcome_column)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    return _with_probability_confidence_features(
        frame, predictor_weights=predictor_weights
    )


def _metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or frame["y"].nunique() != 2:
        raise SelectiveDraftProbabilityError("evaluation slice needs both outcomes")
    y = frame["y"].to_numpy(dtype=int)
    probability = frame["ensemble_probability"].to_numpy(dtype=float)
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(probability, bins, right=True) - 1, 0, 9)
    ece = 0.0
    for bin_id in range(10):
        mask = bin_ids == bin_id
        if mask.any():
            ece += float(mask.mean()) * abs(float(y[mask].mean()) - float(probability[mask].mean()))
    return {
        "rows": len(frame),
        "auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, probability)),
        "ece_10": ece,
    }


def fit_side_symmetric_calibration(
    outcomes: np.ndarray, probabilities: np.ndarray
) -> float:
    """Fit one positive logit scale with a fixed zero intercept."""

    target = np.asarray(outcomes, dtype=int)
    values = np.asarray(probabilities, dtype=float)
    logits = logit(np.clip(values, 1e-5, 1.0 - 1e-5))
    result = minimize_scalar(
        lambda slope: log_loss(target, expit(float(slope) * logits)),
        bounds=(0.2, 3.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success or not math.isfinite(float(result.x)):
        raise SelectiveDraftProbabilityError("probability calibration failed")
    return float(result.x)


def apply_side_symmetric_calibration(
    probabilities: np.ndarray, slope: float
) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    calibrated = expit(
        float(slope) * logit(np.clip(values, 1e-5, 1.0 - 1e-5))
    )
    if not np.isfinite(calibrated).all():
        raise SelectiveDraftProbabilityError("calibrated probabilities are invalid")
    return calibrated


def _cluster_bootstrap_auc(
    frame: pd.DataFrame, *, repetitions: int = 2000, seed: int = 23071
) -> dict[str, float]:
    clusters = [group.index.to_numpy() for _, group in frame.groupby("series_id", sort=True)]
    if len(clusters) < 2:
        raise SelectiveDraftProbabilityError("bootstrap needs at least two series")
    outcomes = frame["y"].to_numpy(dtype=int)
    probabilities = frame["ensemble_probability"].to_numpy(dtype=float)
    index_lookup = {index: position for position, index in enumerate(frame.index)}
    positions = [np.asarray([index_lookup[index] for index in cluster], dtype=int) for cluster in clusters]
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(int(repetitions)):
        chosen = rng.integers(0, len(positions), size=len(positions))
        sampled = np.concatenate([positions[index] for index in chosen])
        if np.unique(outcomes[sampled]).size == 2:
            values.append(float(roc_auc_score(outcomes[sampled], probabilities[sampled])))
    if len(values) < repetitions * 0.95:
        raise SelectiveDraftProbabilityError("bootstrap produced too few valid draws")
    return {
        "repetitions": len(values),
        "median": float(np.median(values)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
    }


def _group_metrics(frame: pd.DataFrame, column: str, *, minimum_rows: int = 40) -> list[dict[str, Any]]:
    output = []
    for value, group in frame.groupby(column, sort=True):
        if len(group) >= minimum_rows and group["y"].nunique() == 2:
            output.append({column: str(value), **_metrics(group)})
    return output


def _ridge_confidence_predictions(
    training_features: np.ndarray,
    training_target: np.ndarray,
    evaluation_features: np.ndarray,
    *,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a deterministic standardized ridge without version-specific solvers."""

    model = fit_ridge_confidence(
        training_features, training_target, alpha=alpha
    )
    return (
        predict_ridge_confidence(model, training_features),
        predict_ridge_confidence(model, evaluation_features),
    )


def fit_ridge_confidence(
    training_features: np.ndarray,
    training_target: np.ndarray,
    *,
    alpha: float,
) -> dict[str, Any]:
    """Fit the serializable side-symmetric confidence model."""

    mean = training_features.mean(axis=0)
    scale = training_features.std(axis=0)
    scale[scale == 0] = 1.0
    training_scaled = (training_features - mean) / scale
    target_mean = float(training_target.mean())
    centered_target = training_target - target_mean
    penalty = float(alpha) * np.eye(training_scaled.shape[1], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        coefficient = np.linalg.solve(
            training_scaled.T @ training_scaled + penalty,
            training_scaled.T @ centered_target,
        )
    if not np.isfinite(coefficient).all():
        raise SelectiveDraftProbabilityError("confidence ridge produced non-finite values")
    return {
        "schema_version": "scryglass:selective-confidence-ridge:v1",
        "feature_columns": list(CONFIDENCE_COLUMNS),
        "alpha": float(alpha),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "target_mean": target_mean,
        "coefficient": coefficient.tolist(),
    }


def predict_ridge_confidence(
    model: Mapping[str, Any], features: np.ndarray
) -> np.ndarray:
    if model.get("feature_columns") != list(CONFIDENCE_COLUMNS):
        raise SelectiveDraftProbabilityError("confidence feature contract changed")
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    coefficient = np.asarray(model["coefficient"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        prediction = (
            float(model["target_mean"])
            + ((features - mean) / scale) @ coefficient
        )
    if not np.isfinite(prediction).all():
        raise SelectiveDraftProbabilityError("confidence ridge produced non-finite values")
    return prediction


def fit_selective_candidate(
    frame: pd.DataFrame,
    *,
    evidence_end: str,
    target_coverage: float = 0.9,
    ridge_alpha: float = 10.0,
    input_sha256: Mapping[str, str],
    predictor_weights: Mapping[str, float] | None = None,
    predictor_semantics: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze the confidence gate and calibration for later unseen games."""

    end = pd.Timestamp(evidence_end)
    if end.tzinfo is None:
        raise SelectiveDraftProbabilityError("evidence end needs a timezone")
    weights = _validated_predictor_weights(predictor_weights)
    semantics = (
        {name: name for name in PREDICTORS}
        if predictor_semantics is None
        else {str(name): str(value) for name, value in predictor_semantics.items()}
    )
    if set(semantics) != set(PREDICTORS) or any(
        not value.strip() for value in semantics.values()
    ):
        raise SelectiveDraftProbabilityError("predictor semantics are incomplete")
    training = _with_probability_confidence_features(
        frame[frame["date"] < end].copy(), predictor_weights=weights
    )
    if len(training) < 1500:
        raise SelectiveDraftProbabilityError("candidate evidence is too small")
    required_hashes = {"matrix", *PREDICTORS}
    if set(input_sha256) != required_hashes or any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in input_sha256.values()
    ):
        raise SelectiveDraftProbabilityError("candidate input hashes are incomplete")
    features = training[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float)
    squared_error = np.square(
        training["y"] - training["ensemble_probability"]
    ).to_numpy(dtype=float)
    confidence_model = fit_ridge_confidence(
        features, squared_error, alpha=ridge_alpha
    )
    confidence_score = -predict_ridge_confidence(confidence_model, features)
    threshold = float(np.quantile(confidence_score, 1.0 - target_coverage))
    calibration_slope = fit_side_symmetric_calibration(
        training["y"].to_numpy(dtype=int),
        training["ensemble_probability"].to_numpy(dtype=float),
    )
    artifact = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "status": "frozen_candidate_waiting_for_new_holdout",
        "authority": "research_only",
        "evidence": {
            "end_exclusive": end.isoformat(),
            "rows": len(training),
            "input_sha256": dict(sorted(input_sha256.items())),
        },
        "ensemble": {
            "predictors": list(PREDICTORS),
            "weights": weights,
            "semantics": {name: semantics[name] for name in PREDICTORS},
        },
        "confidence": {
            "target_coverage": float(target_coverage),
            "threshold": threshold,
            "model": confidence_model,
        },
        "calibration": {
            "kind": "zero_intercept_logit_scale",
            "slope": calibration_slope,
            "side_symmetric": True,
        },
        "public_probability": False,
        "public_recommendation": False,
        "betting_odds_ev_stake": False,
    }
    artifact["receipt_sha256"] = canonical_sha256(artifact)
    return artifact


def apply_selective_candidate(
    artifact: Mapping[str, Any], frame: pd.DataFrame
) -> pd.DataFrame:
    """Apply one frozen gate without reading outcomes."""

    if artifact.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise SelectiveDraftProbabilityError("candidate schema changed")
    expected_receipt = artifact.get("receipt_sha256")
    unsigned = {key: value for key, value in artifact.items() if key != "receipt_sha256"}
    if expected_receipt != canonical_sha256(unsigned):
        raise SelectiveDraftProbabilityError("candidate receipt does not match")
    if "y" in frame.columns:
        inference = frame.drop(columns="y").copy()
    else:
        inference = frame.copy()
    missing = [
        column
        for column in (*PREDICTORS, *CONFIDENCE_COLUMNS[-9:])
        if column not in inference
    ]
    if missing:
        raise SelectiveDraftProbabilityError("candidate inference columns are incomplete")
    weights = _validated_predictor_weights(artifact.get("ensemble", {}).get("weights"))
    inference = _with_probability_confidence_features(
        inference, predictor_weights=weights
    )
    raw = inference["ensemble_probability"].to_numpy(dtype=float)
    features = inference[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float)
    score = -predict_ridge_confidence(artifact["confidence"]["model"], features)
    output = inference.copy()
    output["ensemble_probability_uncalibrated"] = raw
    output["ensemble_probability"] = apply_side_symmetric_calibration(
        raw, float(artifact["calibration"]["slope"])
    )
    output["confidence_score"] = score
    output["probability_authorized"] = score >= float(
        artifact["confidence"]["threshold"]
    )
    return output


def evaluate_frozen_candidate_holdout(
    artifact: Mapping[str, Any],
    frame: pd.DataFrame,
    *,
    holdout_start: str,
    holdout_end: str,
    minimum_selected_rows: int = 100,
    minimum_coverage: float = 0.75,
    minimum_auc: float = 0.710,
    maximum_ece: float = 0.08,
    minimum_leagues_with_20_rows: int = 3,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate one frozen artifact once on games after its evidence window."""

    if (
        minimum_selected_rows < 100
        or minimum_coverage < 0.75
        or minimum_auc < 0.710
        or maximum_ece > 0.08
        or minimum_leagues_with_20_rows < 3
    ):
        raise SelectiveDraftProbabilityError(
            "frozen holdout gates cannot be weakened"
        )
    start = pd.Timestamp(holdout_start)
    end = pd.Timestamp(holdout_end)
    evidence_end_value = artifact.get("evidence", {}).get("end_exclusive")
    if not isinstance(evidence_end_value, str):
        raise SelectiveDraftProbabilityError("candidate evidence end is missing")
    evidence_end = pd.Timestamp(evidence_end_value)
    if start.tzinfo is None or end.tzinfo is None or evidence_end.tzinfo is None:
        raise SelectiveDraftProbabilityError("holdout boundaries need a timezone")
    if start < evidence_end or start >= end:
        raise SelectiveDraftProbabilityError("holdout overlaps candidate evidence")
    if "y" not in frame:
        raise SelectiveDraftProbabilityError("holdout outcomes are missing")
    holdout = frame[(frame["date"] >= start) & (frame["date"] < end)].copy()
    if holdout.empty:
        raise SelectiveDraftProbabilityError("holdout is empty")
    inference = apply_selective_candidate(artifact, holdout.drop(columns="y"))
    selected = inference[inference["probability_authorized"]].copy()
    coverage = len(selected) / len(holdout)
    selected_league_counts = selected["league"].value_counts()
    leagues_ready = int(selected_league_counts.ge(20).sum())
    if (
        len(selected) < minimum_selected_rows
        or coverage < minimum_coverage
        or leagues_ready < minimum_leagues_with_20_rows
    ):
        raise SelectiveDraftProbabilityError(
            "holdout inventory is incomplete; outcomes remain sealed"
        )
    inference["y"] = holdout["y"].to_numpy(dtype=int)
    selected = inference[inference["probability_authorized"]].copy()
    enough_outcomes = selected["y"].nunique() == 2
    if enough_outcomes:
        metrics = _metrics(selected)
        quantum_baseline = selected.copy()
        quantum_baseline["ensemble_probability"] = quantum_baseline["quantum"]
        baseline_metrics = _metrics(quantum_baseline)
        leagues = _group_metrics(selected, "league", minimum_rows=20)
        bootstrap = _cluster_bootstrap_auc(selected)
    else:
        metrics = None
        baseline_metrics = None
        leagues = []
        bootstrap = None
    gates = {
        "minimum_selected_rows": len(selected) >= minimum_selected_rows,
        "coverage": coverage >= minimum_coverage,
        "both_outcomes": selected["y"].nunique() == 2,
        "auc": bool(metrics and metrics["auc"] > minimum_auc),
        "brier": bool(
            metrics
            and baseline_metrics
            and metrics["brier"] <= baseline_metrics["brier"]
        ),
        "log_loss": bool(
            metrics
            and baseline_metrics
            and metrics["log_loss"] <= baseline_metrics["log_loss"]
        ),
        "ece": bool(metrics and metrics["ece_10"] <= maximum_ece),
        "regional_coverage": len(leagues) >= minimum_leagues_with_20_rows,
        "bootstrap_median_auc": bool(
            bootstrap and bootstrap["median"] > minimum_auc
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "promotion_eligible" if all(gates.values()) else "holdout_pending_or_failed",
        "authority": "independent_receipt_required",
        "candidate_receipt_sha256": artifact["receipt_sha256"],
        "holdout": {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "eligible_rows": len(holdout),
            "selected_rows": len(selected),
            "coverage": coverage,
        },
        "metrics": metrics,
        "same_rows_quantum_baseline": baseline_metrics,
        "leagues": leagues,
        "series_bootstrap_auc": bootstrap,
        "gates": {**gates, "passed": all(gates.values())},
        "public_probability": False,
        "public_recommendation": False,
        "betting_odds_ev_stake": False,
    }
    report["receipt_sha256"] = canonical_sha256(report)
    return report, selected


def evaluate_rolling_selective_probability(
    frame: pd.DataFrame,
    *,
    fold_edges: Sequence[str],
    predictor_weights: Mapping[str, float] | None = None,
    target_coverage: float = 0.9,
    ridge_alpha: float = 10.0,
    minimum_auc: float = 0.710,
    maximum_ece: float = 0.08,
    minimum_rows: int = 1500,
    minimum_leagues_with_100_rows: int = 4,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Replay a confidence gate that learns only from earlier OOF windows."""

    edges = [pd.Timestamp(value) for value in fold_edges]
    if len(edges) < 4 or any(value.tzinfo is None for value in edges):
        raise SelectiveDraftProbabilityError("rolling folds are incomplete")
    weights = _validated_predictor_weights(predictor_weights)
    working = _with_probability_confidence_features(
        frame, predictor_weights=weights
    )
    working["fold"] = -1
    for index, (start, end) in enumerate(zip(edges, edges[1:])):
        working.loc[
            (working["date"] >= start) & (working["date"] < end), "fold"
        ] = index
    outputs = []
    folds = []
    for fold_id in range(1, len(edges) - 1):
        training = working[
            (working["fold"] >= 0) & (working["fold"] < fold_id)
        ].copy()
        evaluation = working[working["fold"] == fold_id].copy()
        if len(training) < 300 or len(evaluation) < 100:
            raise SelectiveDraftProbabilityError("rolling fold is too small")
        target = np.square(training["y"] - training["ensemble_probability"])
        model = fit_ridge_confidence(
            training[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float),
            target.to_numpy(dtype=float),
            alpha=ridge_alpha,
        )
        training_score = -predict_ridge_confidence(
            model, training[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float)
        )
        evaluation["confidence_score"] = -predict_ridge_confidence(
            model, evaluation[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float)
        )
        threshold = float(np.quantile(training_score, 1.0 - target_coverage))
        evaluation["probability_authorized"] = (
            evaluation["confidence_score"] >= threshold
        )
        calibration_slope = fit_side_symmetric_calibration(
            training["y"].to_numpy(dtype=int),
            training["ensemble_probability"].to_numpy(dtype=float),
        )
        evaluation["ensemble_probability_uncalibrated"] = evaluation[
            "ensemble_probability"
        ]
        evaluation["ensemble_probability"] = apply_side_symmetric_calibration(
            evaluation["ensemble_probability"].to_numpy(dtype=float),
            calibration_slope,
        )
        selected = evaluation[evaluation["probability_authorized"]].copy()
        outputs.append(selected)
        folds.append(
            {
                "fold": fold_id,
                "training_rows": len(training),
                "eligible_rows": len(evaluation),
                "selected_rows": len(selected),
                "coverage": len(selected) / len(evaluation),
                "confidence_threshold": threshold,
                "calibration_slope": calibration_slope,
                **_metrics(selected),
            }
        )
    selected = pd.concat(outputs, ignore_index=True)
    metrics = _metrics(selected)
    eligible_rows = int(sum(item["eligible_rows"] for item in folds))
    quantum_baseline = selected.copy()
    quantum_baseline["ensemble_probability"] = quantum_baseline["quantum"]
    baseline_metrics = _metrics(quantum_baseline)
    leagues = _group_metrics(selected, "league", minimum_rows=100)
    patches = _group_metrics(selected, "source_patch")
    gates = {
        "pooled_auc": metrics["auc"] > minimum_auc,
        "pooled_brier": metrics["brier"] <= baseline_metrics["brier"],
        "pooled_log_loss": metrics["log_loss"] <= baseline_metrics["log_loss"],
        "pooled_ece": metrics["ece_10"] <= maximum_ece,
        "minimum_rows": len(selected) >= minimum_rows,
        "regional_coverage": len(leagues) >= minimum_leagues_with_100_rows,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "rolling_development_evaluation",
        "authority": "research_only",
        "target_coverage": float(target_coverage),
        "ridge_alpha": float(ridge_alpha),
        "predictor_weights": weights,
        "eligible_rows": eligible_rows,
        "selected_rows": len(selected),
        "coverage": len(selected) / eligible_rows,
        **metrics,
        "folds": folds,
        "leagues": leagues,
        "patches": patches,
        "same_rows_quantum_baseline": baseline_metrics,
        "delta_vs_quantum": {
            "auc": metrics["auc"] - baseline_metrics["auc"],
            "brier": metrics["brier"] - baseline_metrics["brier"],
            "log_loss": metrics["log_loss"] - baseline_metrics["log_loss"],
        },
        "series_bootstrap_auc": _cluster_bootstrap_auc(selected),
        "gates": {**gates, "passed": all(gates.values())},
    }
    report["receipt_sha256"] = canonical_sha256(report)
    return report, selected


def evaluate_selective_probability(
    frame: pd.DataFrame,
    *,
    selection_start: str,
    selection_end: str,
    evaluation_end: str,
    evaluation_start: str | None = None,
    predictor_weights: Mapping[str, float] | None = None,
    target_coverage: float = 0.8,
    ridge_alpha: float = 10.0,
    minimum_evaluation_coverage: float = 0.75,
    minimum_auc: float = 0.710,
    maximum_ece: float = 0.08,
    minimum_window_rows: int = 300,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if not 0 < target_coverage < 1:
        raise SelectiveDraftProbabilityError("target coverage must be between zero and one")
    selection_start_time = pd.Timestamp(selection_start)
    selection_end_time = pd.Timestamp(selection_end)
    evaluation_start_time = pd.Timestamp(evaluation_start or selection_end)
    evaluation_end_time = pd.Timestamp(evaluation_end)
    if (
        selection_start_time.tzinfo is None
        or selection_end_time.tzinfo is None
        or evaluation_start_time.tzinfo is None
        or evaluation_end_time.tzinfo is None
    ):
        raise SelectiveDraftProbabilityError("window boundaries must include a timezone")
    weights = _validated_predictor_weights(predictor_weights)
    working = _with_probability_confidence_features(
        frame, predictor_weights=weights
    )
    selection = working[
        (working["date"] >= selection_start_time)
        & (working["date"] < selection_end_time)
    ].copy()
    evaluation = working[
        (working["date"] >= evaluation_start_time)
        & (working["date"] < evaluation_end_time)
    ].copy()
    if len(selection) < 300 or len(evaluation) < minimum_window_rows:
        raise SelectiveDraftProbabilityError("selection or evaluation window is too small")

    squared_error = np.square(selection["y"] - selection["ensemble_probability"])
    selection_loss, evaluation_loss = _ridge_confidence_predictions(
        selection[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float),
        squared_error.to_numpy(dtype=float),
        evaluation[list(CONFIDENCE_COLUMNS)].to_numpy(dtype=float),
        alpha=float(ridge_alpha),
    )
    selection["confidence_score"] = -selection_loss
    evaluation["confidence_score"] = -evaluation_loss
    threshold = float(selection["confidence_score"].quantile(1.0 - target_coverage))
    evaluation["probability_authorized"] = evaluation["confidence_score"] >= threshold
    calibration_slope = fit_side_symmetric_calibration(
        selection["y"].to_numpy(dtype=int),
        selection["ensemble_probability"].to_numpy(dtype=float),
    )
    evaluation["ensemble_probability_uncalibrated"] = evaluation[
        "ensemble_probability"
    ]
    evaluation["ensemble_probability"] = apply_side_symmetric_calibration(
        evaluation["ensemble_probability"].to_numpy(dtype=float),
        calibration_slope,
    )
    selected = evaluation[evaluation["probability_authorized"]].copy()
    coverage = len(selected) / len(evaluation)

    folds = []
    fold_edges = (
        ("2026-q1", "2026-01-01T00:00:00Z", "2026-04-01T00:00:00Z"),
        ("2026-spring", "2026-04-01T00:00:00Z", "2026-06-01T00:00:00Z"),
        ("2026-midseason", "2026-06-01T00:00:00Z", evaluation_end),
    )
    for fold_id, start, end in fold_edges:
        part = selected[(selected["date"] >= pd.Timestamp(start)) & (selected["date"] < pd.Timestamp(end))]
        if not part.empty and part["y"].nunique() == 2:
            folds.append({"id": fold_id, **_metrics(part)})

    metrics = _metrics(selected)
    quantum_baseline = selected.copy()
    quantum_baseline["ensemble_probability"] = quantum_baseline["quantum"]
    baseline_metrics = _metrics(quantum_baseline)
    gate = {
        "auc": metrics["auc"] > minimum_auc,
        "coverage": coverage >= minimum_evaluation_coverage,
        "brier": metrics["brier"] <= baseline_metrics["brier"],
        "log_loss": metrics["log_loss"] <= baseline_metrics["log_loss"],
        "ece": metrics["ece_10"] <= maximum_ece,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "exploratory_candidate",
        "authority": "research_only",
        "selection": {
            "rows": len(selection),
            "start": selection_start_time.isoformat(),
            "end": selection_end_time.isoformat(),
            "predictor_weights": weights,
            "confidence_model": "standardized_ridge_squared_error",
            "ridge_alpha": float(ridge_alpha),
            "target_coverage": float(target_coverage),
            "confidence_threshold": threshold,
            "calibration": {
                "kind": "zero_intercept_logit_scale",
                "slope": calibration_slope,
                "side_symmetric": True,
            },
            "side_symmetric_features": list(CONFIDENCE_COLUMNS),
        },
        "evaluation": {
            "start": evaluation_start_time.isoformat(),
            "end": evaluation_end_time.isoformat(),
            "eligible_rows": len(evaluation),
            "selected_rows": len(selected),
            "coverage": coverage,
            **metrics,
            "folds": folds,
            "leagues": _group_metrics(selected, "league"),
            "patches": _group_metrics(selected, "source_patch"),
            "series_bootstrap_auc": _cluster_bootstrap_auc(selected),
            "same_rows_quantum_baseline": baseline_metrics,
            "delta_vs_quantum": {
                "auc": metrics["auc"] - baseline_metrics["auc"],
                "brier": metrics["brier"] - baseline_metrics["brier"],
                "log_loss": metrics["log_loss"] - baseline_metrics["log_loss"],
            },
        },
        "gates": {**gate, "passed": all(gate.values())},
        "public_contract": {
            "authorized_rows": "calibrated win probability and side recommendation",
            "abstained_rows": "descriptive composition score only",
            "betting_fields": False,
        },
    }
    report["receipt_sha256"] = canonical_sha256(report)
    output = evaluation[
        [
            "game_uid",
            "series_id",
            "date",
            "league",
            "source_patch",
            "y",
            "ensemble_probability",
            "confidence_score",
            "probability_authorized",
        ]
    ].copy()
    return report, output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--quantum", type=Path, required=True)
    parser.add_argument("--roster", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--development-composite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--candidate-artifact", type=Path)
    parser.add_argument("--holdout-start")
    parser.add_argument("--holdout-end")
    args = parser.parse_args(argv)
    paths = {
        "quantum": args.quantum,
        "roster": args.roster,
        "identity": args.identity,
        "development_composite": args.development_composite,
    }
    frame = load_evaluation_frame(matrix_path=args.matrix, prediction_paths=paths)
    if args.candidate_artifact:
        if not args.holdout_start or not args.holdout_end:
            parser.error("candidate evaluation needs holdout boundaries")
        try:
            artifact = json.loads(args.candidate_artifact.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SelectiveDraftProbabilityError(
                "candidate artifact is invalid"
            ) from error
        report, predictions = evaluate_frozen_candidate_holdout(
            artifact,
            frame,
            holdout_start=args.holdout_start,
            holdout_end=args.holdout_end,
        )
    else:
        report, predictions = evaluate_selective_probability(
            frame,
            selection_start="2025-07-01T00:00:00Z",
            selection_end="2026-01-01T00:00:00Z",
            evaluation_end="2026-08-09T00:00:00Z",
        )
    report["inputs"] = {
        "matrix": {"path": str(args.matrix), "sha256": file_sha256(args.matrix)},
        "predictions": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in paths.items()
        },
    }
    report["receipt_sha256"] = canonical_sha256({key: value for key, value in report.items() if key != "receipt_sha256"})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    predictions.to_parquet(args.predictions, index=False)
    print(
        json.dumps(
            report.get("evaluation", report.get("holdout")), sort_keys=True
        )
    )
    print(report["receipt_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
