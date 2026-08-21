"""Research-only uncertainty for the corrected future-value point model.

The point model owns its fold-local representation.  This module consumes that
representation after it is frozen.  It resamples whole series from the outer
training population, refits only the zero-intercept logistic coefficients, and
reports epistemic intervals for an outer validation design.

The module does not select a transform, imputation value, feature scale, C, or
calibration.  Those values are inputs.  The pre-event ledger contains no
observed outcome column.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
import warnings
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


SCHEMA_VERSION = "scryglass:future-value-uncertainty:v1"
DEFAULT_REQUESTED_DRAWS = 2000
DEFAULT_SEED = 461
DEFAULT_ALPHA = 0.05
MIN_ACCEPTED_FRACTION = 0.99
MIN_ACCEPTED_DRAWS = 1000
SIDE_SWAP_TOLERANCE = 1e-12
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class FutureValueUncertaintyError(ValueError):
    """The fixed fold cannot support a fail-closed uncertainty artifact."""


@dataclass(frozen=True)
class FixedFutureValueTransform:
    """One already-selected fold-local transform.

    ``imputation_values`` and ``scales`` are never recomputed by this module.
    They are applied in ``feature_names`` order.
    """

    feature_names: tuple[str, ...]
    imputation_values: tuple[float, ...]
    scales: tuple[float, ...]
    sha256: str

    @classmethod
    def build(
        cls,
        feature_names: Sequence[str],
        imputation_values: Sequence[float] | Mapping[str, float] | None,
        scales: Sequence[float] | Mapping[str, float] | None,
        *,
        claimed_sha256: str | None = None,
    ) -> "FixedFutureValueTransform":
        names = tuple(str(value) for value in feature_names)
        if not names or any(not value for value in names) or len(set(names)) != len(names):
            raise FutureValueUncertaintyError("fixed transform feature names are invalid")

        def ordered_values(
            values: Sequence[float] | Mapping[str, float] | None,
            label: str,
        ) -> tuple[float, ...]:
            if values is None:
                raise FutureValueUncertaintyError(
                    f"fixed transform {label} values are required"
                )
            elif isinstance(values, Mapping):
                missing = [name for name in names if name not in values]
                if missing:
                    raise FutureValueUncertaintyError(
                        f"fixed transform {label} is missing: {', '.join(missing)}"
                    )
                result = tuple(float(values[name]) for name in names)
            else:
                if len(values) != len(names):
                    raise FutureValueUncertaintyError(
                        f"fixed transform {label} length does not match feature names"
                    )
                result = tuple(float(value) for value in values)
            if not np.isfinite(np.asarray(result, dtype=float)).all():
                raise FutureValueUncertaintyError(f"fixed transform {label} is non-finite")
            return result

        imputation = ordered_values(imputation_values, "imputation")
        scale = ordered_values(scales, "scales")
        if any(value <= 0.0 for value in scale):
            raise FutureValueUncertaintyError("fixed transform scales must be positive")
        payload = {
            "feature_names": list(names),
            "imputation_values": list(imputation),
            "scales": list(scale),
            "operation": "fixed_fold_local_imputation_then_scale",
        }
        digest = _sha256_json(payload)
        if claimed_sha256 is not None and str(claimed_sha256).lower() != digest:
            raise FutureValueUncertaintyError("fixed transform hash does not match values")
        return cls(names, imputation, scale, digest)

    def payload(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "imputation_values": list(self.imputation_values),
            "scales": list(self.scales),
            "operation": "fixed_fold_local_imputation_then_scale",
            "sha256": self.sha256,
        }


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FutureValueUncertaintyError("uncertainty receipt is not canonical JSON") from error


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=float))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _identity_sha256(values: Iterable[Any]) -> str:
    return _sha256_json(sorted(str(value) for value in values))


def _source_receipt_hash(source_receipt: Mapping[str, Any] | None) -> str:
    payload = dict(source_receipt or {})
    claimed = payload.pop("receipt_sha256", None)
    if claimed is not None:
        if not isinstance(claimed, str) or SHA256_RE.fullmatch(claimed) is None:
            raise FutureValueUncertaintyError("source receipt hash is invalid")
        actual = _sha256_json(payload)
        if actual != claimed.lower():
            raise FutureValueUncertaintyError("source receipt hash does not match payload")
        return claimed.lower()
    return _sha256_json(payload)


def _frame_ids(frame: pd.DataFrame, label: str) -> tuple[str, ...]:
    for column in ("game_id", "game_uid", "gameid", "match_id"):
        if column in frame.columns:
            values = frame[column].astype("string").str.strip()
            if values.isna().any() or values.eq("").any():
                raise FutureValueUncertaintyError(f"{label} has incomplete game identity")
            result = tuple(str(value) for value in values)
            if len(set(result)) != len(result):
                raise FutureValueUncertaintyError(f"{label} has duplicate game identity")
            return result
    result = tuple(str(value) for value in frame.index)
    if len(set(result)) != len(result):
        raise FutureValueUncertaintyError(f"{label} has duplicate index identity")
    return result


def _series_ids(frame: pd.DataFrame, column: str, label: str) -> tuple[str, ...]:
    if column not in frame.columns:
        raise FutureValueUncertaintyError(f"{label} is missing series column '{column}'")
    values = frame[column].astype("string").str.strip()
    if values.isna().any() or values.eq("").any():
        raise FutureValueUncertaintyError(f"{label} has incomplete series identity")
    return tuple(str(value) for value in values)


def _training_target(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise FutureValueUncertaintyError(f"training design is missing target '{column}'")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, (0.0, 1.0)).all():
        raise FutureValueUncertaintyError("training target must contain finite binary values")
    if np.unique(values).size != 2:
        raise FutureValueUncertaintyError("training target must contain both classes")
    return values.astype(int)


def _finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        raise FutureValueUncertaintyError(f"design is missing feature '{column}'")
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values


def _prepare_matrix(
    frame: pd.DataFrame,
    transform: FixedFutureValueTransform,
    *,
    side_feature_columns: Mapping[str, Sequence[str]] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the fixed transform and report missingness per row.

    A side pair is optional.  It lets callers pass the point-model side-level
    columns.  A direct feature column is already a blue-minus-red difference.
    """

    columns: list[np.ndarray] = []
    missing_by_row = np.zeros(len(frame), dtype=bool)
    for feature_index, feature in enumerate(transform.feature_names):
        if side_feature_columns is not None and feature in side_feature_columns:
            pair = tuple(side_feature_columns[feature])
            if len(pair) != 2:
                raise FutureValueUncertaintyError(
                    f"side feature pair for {feature} must have blue and red columns"
                )
            blue = _finite_values(frame, str(pair[0]))
            red = _finite_values(frame, str(pair[1]))
            finite = np.isfinite(blue) & np.isfinite(red)
            missing_by_row |= ~finite
            blue = np.where(np.isfinite(blue), blue, transform.imputation_values[feature_index])
            red = np.where(np.isfinite(red), red, transform.imputation_values[feature_index])
            value = blue - red
        else:
            value = _finite_values(frame, feature)
            finite = np.isfinite(value)
            missing_by_row |= ~finite
            value = np.where(np.isfinite(value), value, transform.imputation_values[feature_index])
        columns.append(value / transform.scales[feature_index])
    matrix = np.column_stack(columns).astype(float, copy=False)
    if not np.isfinite(matrix).all():
        raise FutureValueUncertaintyError("fixed transform produced non-finite design")
    status = np.where(missing_by_row, "imputed", "complete")
    return matrix, missing_by_row, status


def _support_statuses(
    frame: pd.DataFrame,
    missing_by_row: np.ndarray,
    *,
    support_column: str | None,
    imputation_column: str | None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if support_column and support_column in frame.columns:
        support = tuple(
            str(value).strip() if pd.notna(value) and str(value).strip() else "unknown"
            for value in frame[support_column]
        )
    elif "model_features_complete" in frame.columns:
        support = tuple(
            "complete" if bool(value) else "imputed"
            for value in frame["model_features_complete"].fillna(False)
        )
    else:
        support = tuple("imputed" if flag else "complete" for flag in missing_by_row)
    if imputation_column and imputation_column in frame.columns:
        imputation = tuple(
            str(value).strip() if pd.notna(value) and str(value).strip() else "unknown"
            for value in frame[imputation_column]
        )
    else:
        imputation = tuple(
            "fixed_fold_local" if flag else "not_needed" for flag in missing_by_row
        )
    return support, imputation


def cluster_bootstrap_weights(
    series_ids: Sequence[Any],
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return row weights and selected whole-series IDs for one draw."""

    if rng is not None and seed is not None:
        raise FutureValueUncertaintyError("cluster bootstrap accepts rng or seed, not both")
    generator = rng if rng is not None else np.random.default_rng(DEFAULT_SEED if seed is None else seed)
    row_series = np.asarray([str(value) for value in series_ids], dtype=object)
    if row_series.size == 0 or any(not value for value in row_series):
        raise FutureValueUncertaintyError("cluster bootstrap has no series IDs")
    clusters = np.asarray(sorted(set(row_series.tolist())), dtype=object)
    selected_indexes = generator.integers(0, len(clusters), size=len(clusters))
    selected = tuple(str(clusters[index]) for index in selected_indexes)
    counts = np.bincount(selected_indexes, minlength=len(clusters)).astype(float)
    lookup = {str(cluster): counts[index] for index, cluster in enumerate(clusters)}
    weights = np.asarray([lookup[str(value)] for value in row_series], dtype=float)
    if not np.isfinite(weights).all() or not np.any(weights > 0.0):
        raise FutureValueUncertaintyError("cluster bootstrap produced invalid weights")
    return weights, selected


_cluster_bootstrap_weight_vector = cluster_bootstrap_weights


def _fit_zero_intercept_logistic(
    matrix: np.ndarray,
    target: np.ndarray,
    *,
    regularization_c: float,
    sample_weight: np.ndarray | None = None,
    max_iterations: int = 1000,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one finite, converged, zero-intercept logistic model."""

    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise FutureValueUncertaintyError("logistic fit input is non-finite")
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(target),) or not np.isfinite(weights).all():
            raise FutureValueUncertaintyError("logistic fit weights are invalid")
        if not np.any(weights > 0.0):
            raise FutureValueUncertaintyError("logistic fit has no positive cluster weight")
    else:
        weights = None
    if float(regularization_c) <= 0.0 or not math.isfinite(float(regularization_c)):
        raise FutureValueUncertaintyError("regularization C must be finite and positive")
    classifier = LogisticRegression(
        C=float(regularization_c),
        penalty="l2",
        solver="lbfgs",
        fit_intercept=False,
        max_iter=int(max_iterations),
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        try:
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                classifier.fit(matrix, target.astype(int), sample_weight=weights)
        except Exception as error:
            raise FutureValueUncertaintyError("future-value bootstrap fit failed") from error
    convergence_messages = [
        str(item.message)
        for item in caught
        if issubclass(item.category, ConvergenceWarning)
    ]
    iterations = [int(value) for value in np.asarray(classifier.n_iter_).ravel()]
    coefficients = np.asarray(classifier.coef_, dtype=float)
    intercept = np.asarray(classifier.intercept_, dtype=float)
    finite = bool(np.isfinite(coefficients).all() and np.isfinite(intercept).all())
    converged = bool(
        finite
        and not convergence_messages
        and iterations
        and max(iterations) < int(max_iterations)
        and intercept.shape == (1,)
        and abs(float(intercept[0])) <= 1e-15
    )
    evidence = {
        "solver": "lbfgs",
        "fit_intercept": False,
        "success": converged,
        "finite_coefficients": finite,
        "iterations": iterations,
        "max_iterations": int(max_iterations),
        "convergence_warnings": convergence_messages,
        "regularization_c": float(regularization_c),
        "coefficient_sha256": _sha256_array(coefficients.reshape(-1)),
    }
    if not converged:
        raise FutureValueUncertaintyError("future-value bootstrap fit did not converge")
    return coefficients.reshape(-1), evidence


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=float)
    if not np.isfinite(values).all():
        raise FutureValueUncertaintyError("bootstrap logits are non-finite")
    output = np.empty_like(values)
    positive = values >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-np.clip(values[positive], -40.0, 40.0)))
    negative_exp = np.exp(np.clip(values[~positive], -40.0, 40.0))
    output[~positive] = negative_exp / (1.0 + negative_exp)
    return output


def _quantiles(values: np.ndarray, alpha: float) -> dict[str, float]:
    lower, median, upper = np.quantile(
        np.asarray(values, dtype=float),
        [float(alpha) / 2.0, 0.5, 1.0 - float(alpha) / 2.0],
    )
    return {"lower": float(lower), "median": float(median), "upper": float(upper)}


def _required_accepted_draws(requested_draws: int, minimum_accepted_draws: int | None) -> int:
    if minimum_accepted_draws is not None:
        minimum = int(minimum_accepted_draws)
        if minimum < 1:
            raise FutureValueUncertaintyError("minimum accepted draws must be positive")
    elif requested_draws >= MIN_ACCEPTED_DRAWS:
        minimum = MIN_ACCEPTED_DRAWS
    else:
        minimum = 1
    if requested_draws >= MIN_ACCEPTED_DRAWS:
        minimum = max(minimum, MIN_ACCEPTED_DRAWS)
    return max(minimum, int(math.ceil(MIN_ACCEPTED_FRACTION * requested_draws)))


def _status_counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _authority() -> dict[str, bool]:
    return {
        "public_player_value": False,
        "public_team_value": False,
        "public_probability": False,
        "odds": False,
        "expected_value": False,
        "recommendation": False,
        "betting": False,
        "promotion": False,
        "deployment": False,
    }


def bootstrap_future_value_uncertainty(
    train: pd.DataFrame,
    outer_validation: pd.DataFrame,
    *,
    feature_names: Sequence[str],
    selected_c: float,
    imputation_values: Sequence[float] | Mapping[str, float] | None = None,
    scales: Sequence[float] | Mapping[str, float] | None = None,
    source_receipt: Mapping[str, Any] | None = None,
    fold_id: Any = "outer",
    calibration: Mapping[str, Any] | None = None,
    series_column: str = "series_id",
    target_column: str = "target",
    side_feature_columns: Mapping[str, Sequence[str]] | None = None,
    support_column: str | None = "support_status",
    imputation_column: str | None = "imputation_status",
    requested_draws: int = DEFAULT_REQUESTED_DRAWS,
    minimum_accepted_draws: int | None = None,
    seed: int = DEFAULT_SEED,
    alpha: float = DEFAULT_ALPHA,
    transform_sha256: str | None = None,
    point_coefficients: Sequence[float] | None = None,
    max_iterations: int = 1000,
) -> dict[str, Any]:
    """Build a deterministic whole-series uncertainty artifact.

    ``train`` is the outer-training population.  ``outer_validation`` is only
    transformed and scored.  Its rows and targets never enter bootstrap draw
    selection, fitting, or calibration.
    """

    if not isinstance(train, pd.DataFrame) or not isinstance(outer_validation, pd.DataFrame):
        raise FutureValueUncertaintyError("train and outer validation must be data frames")
    requested = int(requested_draws)
    if requested < 1:
        raise FutureValueUncertaintyError("requested draws must be positive")
    if not (0.0 < float(alpha) < 1.0):
        raise FutureValueUncertaintyError("interval alpha must be between zero and one")
    if int(max_iterations) < 2:
        raise FutureValueUncertaintyError("logistic max iterations must be at least two")
    if not isinstance(source_receipt, Mapping):
        raise FutureValueUncertaintyError("source receipt is required")
    if calibration is None:
        calibration = {"method": "fixed_identity", "version": 1}
    if not isinstance(calibration, Mapping):
        raise FutureValueUncertaintyError("fixed calibration binding is required")

    train_game_ids = _frame_ids(train, "training design")
    validation_game_ids = _frame_ids(outer_validation, "outer validation design")
    if set(train_game_ids) & set(validation_game_ids):
        raise FutureValueUncertaintyError("outer validation game IDs enter the training draw population")
    train_series = _series_ids(train, series_column, "training design")
    validation_series = _series_ids(outer_validation, series_column, "outer validation design")
    train_series_set = set(train_series)
    validation_series_set = set(validation_series)
    if train_series_set & validation_series_set:
        raise FutureValueUncertaintyError("outer validation series enter the training draw population")
    target = _training_target(train, target_column)
    transform = FixedFutureValueTransform.build(
        feature_names,
        imputation_values,
        scales,
        claimed_sha256=transform_sha256,
    )
    train_matrix, train_missing, _train_transform_status = _prepare_matrix(
        train,
        transform,
        side_feature_columns=side_feature_columns,
    )
    validation_matrix, validation_missing, _validation_transform_status = _prepare_matrix(
        outer_validation,
        transform,
        side_feature_columns=side_feature_columns,
    )
    train_support, train_imputation = _support_statuses(
        train,
        train_missing,
        support_column=support_column,
        imputation_column=imputation_column,
    )
    validation_support, validation_imputation = _support_statuses(
        outer_validation,
        validation_missing,
        support_column=support_column,
        imputation_column=imputation_column,
    )
    selected_c_value = float(selected_c)
    if not math.isfinite(selected_c_value) or selected_c_value <= 0.0:
        raise FutureValueUncertaintyError("selected C must be finite and positive")
    calibration_payload = json.loads(json.dumps(dict(calibration), allow_nan=False))
    calibration_hash = _sha256_json(calibration_payload)
    source_hash = _source_receipt_hash(source_receipt)

    if point_coefficients is None:
        point, point_optimizer = _fit_zero_intercept_logistic(
            train_matrix,
            target,
            regularization_c=selected_c_value,
            max_iterations=int(max_iterations),
        )
    else:
        point = np.asarray(point_coefficients, dtype=float)
        if point.shape != (len(transform.feature_names),) or not np.isfinite(point).all():
            raise FutureValueUncertaintyError("point coefficients are invalid")
        point_optimizer = {
            "solver": "provided_fixed_point_coefficients",
            "fit_intercept": False,
            "success": True,
            "finite_coefficients": True,
            "coefficient_sha256": _sha256_array(point),
            "regularization_c": selected_c_value,
        }
    point_logit = validation_matrix @ point
    point_probability = _sigmoid(point_logit)

    rng = np.random.default_rng(int(seed))
    accepted_logits: list[np.ndarray] = []
    accepted_probabilities: list[np.ndarray] = []
    draw_records: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    all_train_series = tuple(sorted(train_series_set))
    for draw_index in range(requested):
        weights, selected_series = cluster_bootstrap_weights(train_series, rng=rng)
        selected_payload = {
            "draw": int(draw_index),
            "selected_series": list(selected_series),
            "row_weights": weights.astype(int).tolist(),
        }
        selection_hash = _sha256_json(selected_payload)
        record: dict[str, Any] = {
            "draw": int(draw_index),
            "selection_sha256": selection_hash,
            "selected_series": list(selected_series),
            "series_multiplicities": dict(sorted(Counter(selected_series).items())),
            "selected_series_count": len(selected_series),
            "training_series_count": len(all_train_series),
        }
        try:
            coef, optimizer = _fit_zero_intercept_logistic(
                train_matrix,
                target,
                regularization_c=selected_c_value,
                sample_weight=weights,
                max_iterations=int(max_iterations),
            )
            if (
                not isinstance(optimizer, Mapping)
                or optimizer.get("success") is not True
                or not np.isfinite(np.asarray(coef, dtype=float)).all()
                or np.asarray(coef, dtype=float).shape != (len(transform.feature_names),)
            ):
                raise FutureValueUncertaintyError("bootstrap fit evidence is not converged and finite")
            logits = validation_matrix @ coef
            probabilities = _sigmoid(logits)
            swapped_logits = -logits
            swapped_direct = _sigmoid(swapped_logits)
            complement = 1.0 - probabilities
            complement_error = float(np.max(np.abs(swapped_direct - complement)))
            if complement_error > SIDE_SWAP_TOLERANCE:
                raise FutureValueUncertaintyError("side-swap probability complement failed")
            accepted_logits.append(logits)
            accepted_probabilities.append(probabilities)
            record.update(
                {
                    "status": "accepted",
                    "optimizer": optimizer,
                    "coefficient_sha256": _sha256_array(coef),
                    "validation_logit_sha256": _sha256_array(logits),
                    "validation_probability_sha256": _sha256_array(probabilities),
                    "side_swap_probability_sha256": _sha256_array(complement),
                    "side_swap_max_complement_error": complement_error,
                }
            )
        except Exception as error:
            reason = str(error)
            rejection_counts[reason] += 1
            record.update({"status": "rejected", "reason": reason})
        draw_records.append(record)

    accepted_count = len(accepted_logits)
    required = _required_accepted_draws(requested, minimum_accepted_draws)
    acceptance_fraction = accepted_count / float(requested)
    if accepted_count < required or acceptance_fraction < MIN_ACCEPTED_FRACTION:
        raise FutureValueUncertaintyError(
            "bootstrap blocked: accepted draws "
            f"{accepted_count}/{requested} below required {required} "
            f"and {MIN_ACCEPTED_FRACTION:.2%} acceptance"
        )

    logits_draws = np.vstack(accepted_logits)
    probability_draws = np.vstack(accepted_probabilities)
    side_swap_draws = 1.0 - probability_draws
    logit_intervals = [_quantiles(logits_draws[:, index], alpha) for index in range(len(outer_validation))]
    probability_intervals = [
        _quantiles(probability_draws[:, index], alpha) for index in range(len(outer_validation))
    ]
    side_swap_intervals = [
        {"lower": float(1.0 - interval["upper"]), "median": float(1.0 - interval["median"]), "upper": float(1.0 - interval["lower"])}
        for interval in probability_intervals
    ]

    ledger_rows: list[dict[str, Any]] = []
    for index, game_id in enumerate(validation_game_ids):
        ledger_rows.append(
            {
                "fold": str(fold_id),
                "game_id": str(game_id),
                "series_id": str(validation_series[index]),
                "point_logit": float(point_logit[index]),
                "point_probability": float(point_probability[index]),
                "logit_interval": {
                    **logit_intervals[index],
                    "level": float(1.0 - alpha),
                    "label": "epistemic",
                    "kind": "epistemic",
                },
                "probability_interval": {
                    **probability_intervals[index],
                    "level": float(1.0 - alpha),
                    "label": "epistemic",
                    "kind": "epistemic",
                },
                "side_swap_probability_interval": {
                    **side_swap_intervals[index],
                    "level": float(1.0 - alpha),
                    "label": "epistemic",
                    "kind": "epistemic",
                },
                "support_status": str(validation_support[index]),
                "imputation_status": str(validation_imputation[index]),
                "conditional_on": [
                    "fixed_fold_local_transform",
                    "fixed_fold_local_imputation",
                    "fixed_fold_local_scales",
                    "fixed_selected_regularization",
                    "fixed_calibration",
                ],
            }
        )
    ledger_hash = _sha256_json(ledger_rows)
    accepted_records = [record for record in draw_records if record["status"] == "accepted"]
    rejected_records = [record for record in draw_records if record["status"] == "rejected"]
    draw_hashes = {
        "all_draws_sha256": _sha256_json(draw_records),
        "accepted_draws_sha256": _sha256_json(accepted_records),
        "rejected_draws_sha256": _sha256_json(rejected_records),
        "accepted_logit_matrix_sha256": _sha256_array(logits_draws),
        "accepted_probability_matrix_sha256": _sha256_array(probability_draws),
        "accepted_side_swap_matrix_sha256": _sha256_array(side_swap_draws),
    }
    series_payload = {
        "column": str(series_column),
        "train_series_ids": list(all_train_series),
        "outer_validation_series_ids": sorted(validation_series_set),
        "train_series_sha256": _identity_sha256(all_train_series),
        "outer_validation_series_sha256": _identity_sha256(validation_series_set),
        "whole_series_resampling": True,
    }
    fold_payload = {
        "fold_id": str(fold_id),
        "train_game_count": len(train_game_ids),
        "train_game_identity_sha256": _identity_sha256(train_game_ids),
        "outer_validation_game_count": len(validation_game_ids),
        "outer_validation_game_identity_sha256": _identity_sha256(validation_game_ids),
        "train_series_sha256": series_payload["train_series_sha256"],
        "outer_validation_series_sha256": series_payload["outer_validation_series_sha256"],
        "outer_validation_excluded_from_draws": True,
    }
    transform_payload = transform.payload()
    calibration_binding = {
        "method": calibration_payload.get("method", "fixed") if isinstance(calibration_payload, Mapping) else "fixed",
        "payload_sha256": calibration_hash,
        "conditional": True,
    }
    receipt_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "source_receipt_sha256": source_hash,
        "source_sha256": source_hash,
        "fold_sha256": _sha256_json(fold_payload),
        "series_sha256": _sha256_json(series_payload),
        "transform_sha256": transform.sha256,
        "calibration_sha256": calibration_hash,
        "draws_sha256": draw_hashes["all_draws_sha256"],
        "draw_hashes_sha256": _sha256_json(draw_hashes),
        "ledger_sha256": ledger_hash,
        "source": {
            "receipt_sha256": source_hash,
            "source_as_of": source_receipt.get("source_as_of"),
            "source_game_count": source_receipt.get("source_game_count"),
            "source_identity_sha256": source_receipt.get("source_identity_sha256"),
        },
        "fold": fold_payload,
        "series": series_payload,
        "transform": transform_payload,
        "selected_c": selected_c_value,
        "calibration": calibration_binding,
        "draws": {
            "seed": int(seed),
            "requested": requested,
            "accepted": accepted_count,
            "rejected": len(rejected_records),
            "accepted_fraction": acceptance_fraction,
            "required_accepted": required,
            "hashes": draw_hashes,
            "rejection_counts": dict(sorted(rejection_counts.items())),
        },
        "uncertainty": {
            "interval_level": float(1.0 - alpha),
            "logit_label": "epistemic",
            "probability_label": "epistemic",
            "conditional_on": [
                "fixed_fold_local_transform",
                "fixed_fold_local_imputation",
                "fixed_fold_local_scales",
                "fixed_selected_regularization",
                "fixed_calibration",
            ],
        },
        "authority": _authority(),
        "authority_flags": _authority(),
    }
    receipt_payload["receipt_sha256"] = _sha256_json(receipt_payload)

    ledger = {
        "schema_version": "scryglass:future-value-pre-event-uncertainty-ledger:v1",
        "target_policy": "observed_target_excluded",
        "columns": [key for key in ledger_rows[0] if key != "target"] if ledger_rows else [],
        "row_count": len(ledger_rows),
        "rows": ledger_rows,
        "sha256": ledger_hash,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "receipt": receipt_payload,
        "point": {
            "coefficients": point.tolist(),
            "coefficient_sha256": _sha256_array(point),
            "optimizer": point_optimizer,
        },
        "draws": {
            "requested": requested,
            "accepted": accepted_count,
            "rejected": len(rejected_records),
            "records": draw_records,
        },
        "intervals": {
            "level": float(1.0 - alpha),
            "label": "epistemic",
            "conditional_on": receipt_payload["uncertainty"]["conditional_on"],
            "rows": ledger_rows,
        },
        "support": {
            "train": {
                "status_counts": _status_counts(train_support),
                "imputation_status_counts": _status_counts(train_imputation),
            },
            "outer_validation": {
                "status_counts": _status_counts(validation_support),
                "imputation_status_counts": _status_counts(validation_imputation),
            },
        },
        "pre_event_uncertainty_ledger": ledger,
        "pre_event_ledger": ledger,
        "ledger": ledger,
        "authority": receipt_payload["authority"],
        "authority_flags": receipt_payload["authority_flags"],
    }


def verify_uncertainty_receipt(artifact: Mapping[str, Any]) -> bool:
    """Verify the canonical receipt hash and its bound ledger hash."""

    receipt = artifact.get("receipt")
    if not isinstance(receipt, Mapping):
        raise FutureValueUncertaintyError("uncertainty receipt is missing")
    claimed = receipt.get("receipt_sha256")
    if not isinstance(claimed, str) or SHA256_RE.fullmatch(claimed) is None:
        raise FutureValueUncertaintyError("uncertainty receipt hash is invalid")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    if _sha256_json(payload) != claimed.lower():
        raise FutureValueUncertaintyError("uncertainty receipt hash does not match payload")
    fold = receipt.get("fold")
    series = receipt.get("series")
    transform = receipt.get("transform")
    draws = receipt.get("draws")
    if not isinstance(fold, Mapping) or receipt.get("fold_sha256") != _sha256_json(fold):
        raise FutureValueUncertaintyError("uncertainty receipt does not bind the fold")
    if not isinstance(series, Mapping) or receipt.get("series_sha256") != _sha256_json(series):
        raise FutureValueUncertaintyError("uncertainty receipt does not bind the series")
    if not isinstance(transform, Mapping):
        raise FutureValueUncertaintyError("uncertainty receipt does not bind the transform")
    transform_payload = dict(transform)
    transform_claimed = transform_payload.pop("sha256", None)
    if receipt.get("transform_sha256") != transform_claimed or _sha256_json(transform_payload) != transform_claimed:
        raise FutureValueUncertaintyError("uncertainty receipt does not bind the transform hash")
    if (
        not isinstance(draws, Mapping)
        or not isinstance(draws.get("hashes"), Mapping)
        or receipt.get("draws_sha256") != draws["hashes"].get("all_draws_sha256")
        or receipt.get("draw_hashes_sha256") != _sha256_json(draws["hashes"])
    ):
        raise FutureValueUncertaintyError("uncertainty receipt does not bind draw hashes")
    ledger = artifact.get("pre_event_uncertainty_ledger")
    if not isinstance(ledger, Mapping) or "target" in ledger.get("columns", []):
        raise FutureValueUncertaintyError("pre-event uncertainty ledger exposes observed target")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or _sha256_json(rows) != ledger.get("sha256"):
        raise FutureValueUncertaintyError("pre-event uncertainty ledger hash does not match rows")
    if receipt.get("ledger_sha256") != ledger.get("sha256"):
        raise FutureValueUncertaintyError("uncertainty receipt does not bind the ledger")
    return True


# A short alias makes the research entry point convenient in notebooks while
# keeping the descriptive function name available to callers.
run_future_value_uncertainty = bootstrap_future_value_uncertainty


def bootstrap_future_value_model_uncertainty(
    model: Any,
    train_design: pd.DataFrame,
    outer_validation: pd.DataFrame,
    *,
    source_receipt: Mapping[str, Any] | None = None,
    fold_id: Any = "outer",
    calibration: Mapping[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Adapt a ``FutureValueFoldModel`` to the fixed-design entry point.

    The adapter reads fitted values only.  It does not call the model's fit
    path or select any fold-local parameter.
    """

    feature_names = tuple(str(value) for value in model.feature_names)
    selection = model.regularization_selection
    if not isinstance(selection, Mapping) or "selected_c" not in selection:
        raise FutureValueUncertaintyError("future-value model has no selected C")
    bound_source = source_receipt if source_receipt is not None else model.source_receipt
    if not isinstance(bound_source, Mapping):
        raise FutureValueUncertaintyError("future-value model has no source receipt")
    return bootstrap_future_value_uncertainty(
        train_design,
        outer_validation,
        feature_names=feature_names,
        selected_c=float(selection["selected_c"]),
        imputation_values=model.imputation_values,
        scales=model.scales,
        point_coefficients=model.coefficients,
        source_receipt=bound_source,
        fold_id=fold_id,
        calibration=calibration,
        **kwargs,
    )


__all__ = [
    "DEFAULT_REQUESTED_DRAWS",
    "FixedFutureValueTransform",
    "FutureValueUncertaintyError",
    "bootstrap_future_value_uncertainty",
    "bootstrap_future_value_model_uncertainty",
    "cluster_bootstrap_weights",
    "run_future_value_uncertainty",
    "verify_uncertainty_receipt",
]
