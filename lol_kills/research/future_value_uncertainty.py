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
from sklearn.isotonic import IsotonicRegression


SCHEMA_VERSION = "scryglass:future-value-uncertainty:v1"
DEFAULT_REQUESTED_DRAWS = 2000
DEFAULT_SEED = 461
DEFAULT_ALPHA = 0.05
MIN_ACCEPTED_FRACTION = 0.99
MIN_ACCEPTED_DRAWS = 1000
SIDE_SWAP_TOLERANCE = 1e-12
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)

# This contract is separate from the point-model probability calibration above.
# It maps observed support to an expected out-of-sample residual.  The mapping
# is research-only until a later fold has enough prior rows.
SUPPORT_CALIBRATION_SCHEMA_VERSION = "scryglass:future-value-support-calibration:v1"
SUPPORT_CALIBRATION_RECEIPT_SCHEMA_VERSION = (
    "scryglass:future-value-support-calibration-receipt:v1"
)
SUPPORT_CALIBRATION_MINIMUM_COVERAGE = 1.0
SUPPORT_CALIBRATION_DEFAULT_MINIMUM_ROWS = 20
SUPPORT_CALIBRATION_DEFAULT_MINIMUM_BIN_ROWS = 5
SUPPORT_CALIBRATION_DEFAULT_MINIMUM_BINS = 2
SUPPORT_CALIBRATION_DEFAULT_MAXIMUM_BINS = 10

SUPPORT_CALIBRATION_AUTHORITY = {
    "research_only": True,
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


def _frame_dates(frame: pd.DataFrame, label: str) -> tuple[str, tuple[pd.Timestamp, ...]]:
    """Return canonical UTC dates and require one date for every row."""

    column = "date" if "date" in frame.columns else "timestamp" if "timestamp" in frame.columns else None
    if column is None:
        raise FutureValueUncertaintyError(f"{label} is missing date")
    values = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if len(values) == 0 or values.isna().any():
        raise FutureValueUncertaintyError(f"{label} has an invalid date")
    dates = tuple(pd.Timestamp(value) for value in values)
    return column, dates


def _date_text(value: pd.Timestamp) -> str:
    return value.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _game_series_assignment_rows(
    game_ids: Sequence[str],
    series_ids: Sequence[str],
) -> list[dict[str, str]]:
    if len(game_ids) != len(series_ids):
        raise FutureValueUncertaintyError("game and series assignments have different lengths")
    return sorted(
        (
            {"game_id": str(game_id), "series_id": str(series_id)}
            for game_id, series_id in zip(game_ids, series_ids)
        ),
        key=lambda row: row["game_id"],
    )


def _game_series_assignment_sha256(
    game_ids: Sequence[str],
    series_ids: Sequence[str],
) -> str:
    rows = _game_series_assignment_rows(game_ids, series_ids)
    return _sha256_json(rows)


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


def _positive_weight_rows_have_both_target_classes(
    target: np.ndarray,
    sample_weight: np.ndarray,
) -> bool:
    target_values = np.asarray(target)
    weights = np.asarray(sample_weight, dtype=float)
    if target_values.shape != weights.shape or not np.isfinite(weights).all():
        return False
    positive = weights > 0.0
    classes = np.unique(target_values[positive])
    return bool(classes.size == 2 and np.array_equal(classes, np.asarray([0, 1])))


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


def _required_accepted_draws(requested_draws: int) -> int:
    requested = int(requested_draws)
    if requested < MIN_ACCEPTED_DRAWS:
        raise FutureValueUncertaintyError(
            f"requested draws must be at least {MIN_ACCEPTED_DRAWS}"
        )
    return max(MIN_ACCEPTED_DRAWS, int(math.ceil(MIN_ACCEPTED_FRACTION * requested)))


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


def _fixed_calibration(
    calibration: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], float, str]:
    """Validate and normalize the only supported calibration map.

    The uncertainty layer accepts a fixed scalar slope over the zero-intercept
    logit.  It does not replay isotonic, binning, affine-offset, or arbitrary
    calibration maps during bootstrap draws.
    """

    payload = dict(calibration or {"method": "fixed_identity", "version": 1})
    method_value = payload.get(
        "method",
        payload.get(
            "kind",
            payload.get(
                "type",
                "scalar_zero_intercept"
                if "slope" in payload or "calibration_slope" in payload
                else None,
            ),
        ),
    )
    method = str(method_value or "").strip().casefold()
    identity_methods = {"identity", "fixed_identity", "scalar_zero_intercept", "scalar_logit_slope"}
    if method not in identity_methods:
        raise FutureValueUncertaintyError(
            "unsupported calibration map; only a fixed zero-intercept scalar slope is supported"
        )
    slope_value = payload.get(
        "slope",
        payload.get(
            "calibration_slope",
            1.0 if method in {"identity", "fixed_identity"} else None,
        ),
    )
    if slope_value is None:
        raise FutureValueUncertaintyError("calibration slope is required")
    try:
        slope = float(slope_value)
    except (TypeError, ValueError) as error:
        raise FutureValueUncertaintyError("calibration slope is required") from error
    if not math.isfinite(slope) or slope <= 0.0:
        raise FutureValueUncertaintyError("calibration slope must be finite and positive")
    if method in {"identity", "fixed_identity"} and abs(slope - 1.0) > 1e-15:
        raise FutureValueUncertaintyError("identity calibration must have slope one")
    for key in ("intercept", "offset", "calibration_intercept"):
        if key in payload:
            try:
                offset = float(payload[key])
            except (TypeError, ValueError) as error:
                raise FutureValueUncertaintyError("calibration offset is invalid") from error
            if not math.isfinite(offset) or abs(offset) > 1e-15:
                raise FutureValueUncertaintyError("non-zero calibration offset is unsupported")
    allowed = {
        "method",
        "kind",
        "type",
        "slope",
        "calibration_slope",
        "intercept",
        "offset",
        "calibration_intercept",
        "version",
        "source",
    }
    unsupported = sorted(set(payload) - allowed)
    if unsupported:
        raise FutureValueUncertaintyError(
            "unsupported calibration map field(s): " + ", ".join(unsupported)
        )
    normalized = {
        "method": "scalar_zero_intercept",
        "slope": slope,
        "intercept": 0.0,
        "payload_sha256": _sha256_json(payload),
    }
    if "version" in payload:
        normalized["version"] = payload["version"]
    if "source" in payload:
        normalized["source"] = payload["source"]
    return normalized, slope, _sha256_json(normalized)


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
    if requested < MIN_ACCEPTED_DRAWS:
        raise FutureValueUncertaintyError(
            f"requested draws must be at least {MIN_ACCEPTED_DRAWS}"
        )
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
    train_date_column, train_dates = _frame_dates(train, "training design")
    validation_date_column, validation_dates = _frame_dates(
        outer_validation, "outer validation design"
    )
    if train_date_column != validation_date_column:
        raise FutureValueUncertaintyError("training and validation date columns differ")
    train_date_max = max(train_dates)
    validation_date_min = min(validation_dates)
    if not train_date_max < validation_date_min:
        raise FutureValueUncertaintyError(
            "outer validation must start strictly after the latest training date"
        )
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
    calibration_payload, calibration_slope, calibration_hash = _fixed_calibration(calibration)
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
    point_raw_logit = validation_matrix @ point
    point_logit = calibration_slope * point_raw_logit
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
            if not _positive_weight_rows_have_both_target_classes(target, weights):
                raise FutureValueUncertaintyError(
                    "bootstrap draw positive-weight rows do not contain both target classes"
                )
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
            raw_logits = validation_matrix @ coef
            logits = calibration_slope * raw_logits
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
    required = _required_accepted_draws(requested)
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
                    "fixed_zero_intercept_calibration_slope",
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
        "train_game_series_assignment_sha256": _game_series_assignment_sha256(
            train_game_ids, train_series
        ),
        "outer_validation_game_series_assignment_sha256": _game_series_assignment_sha256(
            validation_game_ids, validation_series
        ),
        "train_game_series_assignments": _game_series_assignment_rows(
            train_game_ids, train_series
        ),
        "outer_validation_game_series_assignments": _game_series_assignment_rows(
            validation_game_ids, validation_series
        ),
        "whole_series_resampling": True,
    }
    fold_payload = {
        "fold_id": str(fold_id),
        "date_column": train_date_column,
        "train_game_count": len(train_game_ids),
        "train_game_identity_sha256": _identity_sha256(train_game_ids),
        "outer_validation_game_count": len(validation_game_ids),
        "outer_validation_game_identity_sha256": _identity_sha256(validation_game_ids),
        "train_date_min": _date_text(min(train_dates)),
        "train_date_max": _date_text(train_date_max),
        "outer_validation_date_min": _date_text(validation_date_min),
        "outer_validation_date_max": _date_text(max(validation_dates)),
        "strict_date_boundary": True,
        "train_series_sha256": series_payload["train_series_sha256"],
        "outer_validation_series_sha256": series_payload["outer_validation_series_sha256"],
        "train_game_series_assignment_sha256": series_payload[
            "train_game_series_assignment_sha256"
        ],
        "outer_validation_game_series_assignment_sha256": series_payload[
            "outer_validation_game_series_assignment_sha256"
        ],
        "outer_validation_excluded_from_draws": True,
    }
    transform_payload = transform.payload()
    calibration_binding = {
        "method": calibration_payload["method"],
        "slope": calibration_slope,
        "intercept": 0.0,
        "payload_sha256": calibration_payload["payload_sha256"],
        "normalized_sha256": calibration_hash,
        "conditional": True,
    }
    receipt_payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "source_receipt_sha256": source_hash,
        "source_sha256": source_hash,
        "fold_sha256": _sha256_json(fold_payload),
        "series_sha256": _sha256_json(series_payload),
        "train_date_max": _date_text(train_date_max),
        "outer_validation_date_min": _date_text(validation_date_min),
        "transform_sha256": transform.sha256,
        "calibration_sha256": calibration_hash,
        "calibration_payload_sha256": calibration_payload["payload_sha256"],
        "calibration_method": "scalar_zero_intercept",
        "calibration_slope": calibration_slope,
        "calibration_intercept": 0.0,
        "calibration_slope_sha256": _sha256_json({"slope": calibration_slope}),
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
                    "fixed_zero_intercept_calibration_slope",
                ],
        },
        "authority": _authority(),
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
            "calibration_slope": calibration_slope,
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
        "authority": receipt_payload["authority"],
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
    expected_authority = _authority()
    for label, value in (
        ("receipt authority", receipt.get("authority")),
        ("artifact authority", artifact.get("authority")),
    ):
        if not isinstance(value, Mapping) or set(value) != set(expected_authority):
            raise FutureValueUncertaintyError(f"{label} flags are invalid")
        if any(value[key] is not False for key in expected_authority):
            raise FutureValueUncertaintyError(f"{label} grants authority")
    if "authority_flags" in artifact or "authority_flags" in receipt:
        raise FutureValueUncertaintyError("duplicate authority flag surface is unsupported")
    if "pre_event_ledger" in artifact or "ledger" in artifact:
        raise FutureValueUncertaintyError("duplicate pre-event ledger surface is unsupported")
    calibration = receipt.get("calibration")
    if not isinstance(calibration, Mapping):
        raise FutureValueUncertaintyError("calibration binding is missing")
    try:
        slope = float(calibration["slope"])
        intercept = float(calibration.get("intercept", 1.0))
    except (KeyError, TypeError, ValueError) as error:
        raise FutureValueUncertaintyError("calibration slope binding is invalid") from error
    if (
        not math.isfinite(slope)
        or slope <= 0.0
        or calibration.get("method") != "scalar_zero_intercept"
        or intercept != 0.0
        or receipt.get("calibration_slope") != slope
        or receipt.get("calibration_intercept") != 0.0
        or receipt.get("calibration_method") != "scalar_zero_intercept"
        or receipt.get("calibration_slope_sha256") != _sha256_json({"slope": slope})
        or receipt.get("calibration_sha256") != calibration.get("normalized_sha256")
        or receipt.get("calibration_payload_sha256") != calibration.get("payload_sha256")
    ):
        raise FutureValueUncertaintyError("calibration slope binding is invalid")
    if not isinstance(fold, Mapping) or fold.get("strict_date_boundary") is not True:
        raise FutureValueUncertaintyError("strict date boundary binding is missing")
    try:
        train_max = pd.Timestamp(fold["train_date_max"])
        validation_min = pd.Timestamp(fold["outer_validation_date_min"])
    except (KeyError, TypeError, ValueError) as error:
        raise FutureValueUncertaintyError("interval timestamp binding is invalid") from error
    if pd.isna(train_max) or pd.isna(validation_min) or not train_max < validation_min:
        raise FutureValueUncertaintyError("interval timestamp binding is invalid")
    if (
        receipt.get("train_date_max") != fold.get("train_date_max")
        or receipt.get("outer_validation_date_min") != fold.get("outer_validation_date_min")
    ):
        raise FutureValueUncertaintyError("interval timestamp binding is invalid")
    if not isinstance(series, Mapping):
        raise FutureValueUncertaintyError("series assignment binding is invalid")
    for rows_field, hash_field in (
        ("train_game_series_assignments", "train_game_series_assignment_sha256"),
        ("outer_validation_game_series_assignments", "outer_validation_game_series_assignment_sha256"),
    ):
        assignments = series.get(rows_field)
        if not isinstance(assignments, list) or series.get(hash_field) != _sha256_json(assignments):
            raise FutureValueUncertaintyError("series assignment binding is invalid")
        if any(
            not isinstance(row, Mapping)
            or set(row) != {"game_id", "series_id"}
            or not str(row["game_id"]).strip()
            or not str(row["series_id"]).strip()
            for row in assignments
        ):
            raise FutureValueUncertaintyError("series assignment binding is invalid")
        if assignments != sorted(assignments, key=lambda row: row["game_id"]):
            raise FutureValueUncertaintyError("series assignment binding is not canonical")
        game_ids = [str(row["game_id"]) for row in assignments]
        if len(set(game_ids)) != len(game_ids):
            raise FutureValueUncertaintyError("series assignment binding has duplicate games")
    train_assignments = series["train_game_series_assignments"]
    validation_assignments = series["outer_validation_game_series_assignments"]
    train_series_ids = series.get("train_series_ids")
    validation_series_ids = series.get("outer_validation_series_ids")
    if not isinstance(train_series_ids, list) or not isinstance(validation_series_ids, list):
        raise FutureValueUncertaintyError("series ID binding is invalid")
    if (
        len(train_assignments) != fold.get("train_game_count")
        or len(validation_assignments) != fold.get("outer_validation_game_count")
        or _identity_sha256(row["game_id"] for row in train_assignments)
        != fold.get("train_game_identity_sha256")
        or _identity_sha256(row["game_id"] for row in validation_assignments)
        != fold.get("outer_validation_game_identity_sha256")
    ):
        raise FutureValueUncertaintyError("series assignment does not bind fold games")
    if (
        sorted({str(row["series_id"]) for row in train_assignments})
        != sorted(str(value) for value in train_series_ids)
        or sorted({str(row["series_id"]) for row in validation_assignments})
        != sorted(str(value) for value in validation_series_ids)
    ):
        raise FutureValueUncertaintyError("series assignment does not bind series IDs")
    ledger = artifact.get("pre_event_uncertainty_ledger")
    if not isinstance(ledger, Mapping) or "target" in ledger.get("columns", []):
        raise FutureValueUncertaintyError("pre-event uncertainty ledger exposes observed target")
    rows = ledger.get("rows")
    columns = ledger.get("columns")
    if not isinstance(rows, list) or not isinstance(columns, list):
        raise FutureValueUncertaintyError("pre-event uncertainty ledger is invalid")
    if any(
        not isinstance(row, Mapping) or "target" in row or set(row) != set(columns)
        for row in rows
    ):
        raise FutureValueUncertaintyError("pre-event uncertainty ledger exposes observed target")
    if _sha256_json(rows) != ledger.get("sha256"):
        raise FutureValueUncertaintyError("pre-event uncertainty ledger hash does not match rows")
    if receipt.get("ledger_sha256") != ledger.get("sha256"):
        raise FutureValueUncertaintyError("uncertainty receipt does not bind the ledger")
    return True


# ---------------------------------------------------------------------------
# Strict-prior support calibration
# ---------------------------------------------------------------------------


def _support_value(row: Mapping[str, Any], support_column: str) -> float:
    candidates = (support_column, "minimum_effective_support", "effective_support", "support")
    for name in candidates:
        if name not in row:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError) as error:
            raise FutureValueUncertaintyError("support value is invalid") from error
        if not math.isfinite(value) or value < 0.0:
            raise FutureValueUncertaintyError("support value must be finite and non-negative")
        return value
    raise FutureValueUncertaintyError("support value is missing")


def _support_prediction(row: Mapping[str, Any]) -> tuple[float, float]:
    """Read one out-of-sample probability or logit and return both forms."""

    logit_names = (
        "prediction_logit",
        "candidate_raw_logit",
        "raw_logit",
        "logit",
    )
    probability_names = (
        "prediction_probability",
        "candidate_raw_probability",
        "raw_probability",
        "probability",
        "prediction",
        "candidate",
    )
    logit_value: float | None = None
    for name in logit_names:
        if name in row:
            try:
                logit_value = float(row[name])
            except (TypeError, ValueError) as error:
                raise FutureValueUncertaintyError("prediction logit is invalid") from error
            break
    probability_value: float | None = None
    for name in probability_names:
        if name in row:
            try:
                probability_value = float(row[name])
            except (TypeError, ValueError) as error:
                raise FutureValueUncertaintyError("prediction probability is invalid") from error
            break
    if logit_value is None and probability_value is None:
        raise FutureValueUncertaintyError("support calibration prediction is missing")
    if logit_value is not None:
        if not math.isfinite(logit_value):
            raise FutureValueUncertaintyError("prediction logit is non-finite")
        derived_probability = float(_sigmoid(np.asarray([logit_value], dtype=float))[0])
        if probability_value is not None:
            if not math.isfinite(probability_value) or not 0.0 < probability_value < 1.0:
                raise FutureValueUncertaintyError("prediction probability must be between zero and one")
            if not math.isclose(probability_value, derived_probability, rel_tol=1e-8, abs_tol=1e-10):
                raise FutureValueUncertaintyError("prediction logit and probability disagree")
        probability_value = derived_probability
    assert probability_value is not None
    if not math.isfinite(probability_value) or not 0.0 < probability_value < 1.0:
        raise FutureValueUncertaintyError("prediction probability must be between zero and one")
    if logit_value is None:
        logit_value = float(math.log(probability_value / (1.0 - probability_value)))
    return float(logit_value), float(probability_value)


def _support_residual_target(
    target: float,
    probability: float,
    logit: float,
    *,
    target_kind: str,
) -> float:
    if target not in (0.0, 1.0):
        raise FutureValueUncertaintyError("support calibration target must be binary")
    if target_kind == "log_loss":
        # Per-row log loss is the proper scoring residual.  It is evaluated
        # on a held-out fold and never used to fit that fold's mapping.
        value = -(
            target * math.log(max(probability, 1e-15))
            + (1.0 - target) * math.log(max(1.0 - probability, 1e-15))
        )
    elif target_kind == "absolute_logit_residual":
        # This is an explicit diagnostic target.  The binary outcome is coded
        # as a signed unit logit target because a binary zero/one target has no
        # finite logit.  It is not presented as a probability calibration.
        value = abs(logit - (2.0 * target - 1.0))
    elif target_kind == "absolute_probability_residual":
        value = abs(probability - target)
    else:
        raise FutureValueUncertaintyError(
            "unsupported support calibration target; use log_loss, "
            "absolute_logit_residual, or absolute_probability_residual"
        )
    if not math.isfinite(value) or value < 0.0:
        raise FutureValueUncertaintyError("support calibration residual is invalid")
    return float(value)


def _support_date(row: Mapping[str, Any], label: str) -> pd.Timestamp:
    value = row.get("date", row.get("timestamp"))
    if value is None:
        raise FutureValueUncertaintyError(f"{label} date is missing")
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(timestamp):
        raise FutureValueUncertaintyError(f"{label} date is invalid")
    return pd.Timestamp(timestamp)


def _support_source_rows(
    source_frame: pd.DataFrame | None,
    source_receipt: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build the accepted source-row lookup used by support verification.

    A source receipt carries the accepted identity, while the map frame carries
    the row facts.  Both are required.  A support row cannot self-declare its
    date, result, or series membership.
    """

    if not isinstance(source_frame, pd.DataFrame):
        raise FutureValueUncertaintyError(
            "support calibration accepted source frame is required"
        )
    required = {"game_id", "date", "target", "series_id"}
    missing = sorted(required - set(source_frame.columns))
    if missing:
        raise FutureValueUncertaintyError(
            "support calibration accepted source frame is missing: "
            + ", ".join(missing)
        )
    frame = source_frame[["game_id", "date", "target", "series_id"]].copy()
    frame["game_id"] = frame["game_id"].astype("string").str.strip()
    frame["series_id"] = frame["series_id"].astype("string").str.strip()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["target"] = pd.to_numeric(frame["target"], errors="coerce")
    if (
        frame["game_id"].isna().any()
        or frame["game_id"].eq("").any()
        or frame["game_id"].duplicated().any()
        or frame["series_id"].isna().any()
        or frame["series_id"].eq("").any()
        or frame["date"].isna().any()
        or frame["target"].isna().any()
        or ~frame["target"].isin({0, 1}).all()
    ):
        raise FutureValueUncertaintyError(
            "support calibration accepted source frame identity is invalid"
        )
    eligible_values = source_receipt.get("model_eligible_game_ids")
    if eligible_values is None:
        eligible_values = source_receipt.get("accepted_game_ids")
    if eligible_values is not None:
        eligible = {str(value) for value in eligible_values}
        if set(frame["game_id"].astype(str)) != eligible:
            raise FutureValueUncertaintyError(
                "support calibration accepted source frame does not match receipt census"
            )
    return {
        str(row.game_id): {
            "game_id": str(row.game_id),
            "date": pd.Timestamp(row.date),
            "target": int(row.target),
            "series_id": str(row.series_id),
        }
        for row in frame.itertuples(index=False)
    }


def _verify_support_source_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_rows: Mapping[str, Mapping[str, Any]],
    validation_start: pd.Timestamp,
    validation_end: pd.Timestamp,
    fold_id: str,
) -> None:
    """Check support input rows against the accepted map facts."""

    seen_games: set[str] = set()
    rows_by_series: dict[str, set[str]] = {}
    source_games_by_series: dict[str, set[str]] = {}
    for game_id, source in source_rows.items():
        source_games_by_series.setdefault(str(source["series_id"]), set()).add(
            str(game_id)
        )
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise FutureValueUncertaintyError("support calibration source row is invalid")
        game_id = str(raw.get("game_id", "")).strip()
        source = source_rows.get(game_id)
        if source is None:
            raise FutureValueUncertaintyError(
                "support calibration source row is outside the accepted census"
            )
        if game_id in seen_games:
            raise FutureValueUncertaintyError("support calibration source rows are not unique")
        series_id = str(raw.get("series_id", "")).strip()
        if series_id != str(source["series_id"]):
            raise FutureValueUncertaintyError(
                "support calibration source series does not match accepted source"
            )
        date = _support_date(raw, f"support calibration source row {game_id}")
        if date != source["date"]:
            raise FutureValueUncertaintyError(
                "support calibration source date does not match accepted source"
            )
        if not validation_start <= date <= validation_end:
            raise FutureValueUncertaintyError(
                "support calibration source row is outside its validation window"
            )
        target_value = raw.get("target")
        try:
            target = float(target_value)
        except (TypeError, ValueError) as error:
            raise FutureValueUncertaintyError(
                "support calibration source target is missing"
            ) from error
        if not math.isfinite(target) or target not in (0.0, 1.0) or int(target) != int(source["target"]):
            raise FutureValueUncertaintyError(
                "support calibration source target does not match accepted source"
            )
        seen_games.add(game_id)
        rows_by_series.setdefault(series_id, set()).add(game_id)
    for series_id, row_series_games in rows_by_series.items():
        if source_games_by_series.get(series_id, set()) != row_series_games:
            raise FutureValueUncertaintyError(
                f"support calibration source rows do not cover complete series: {fold_id}"
            )


def _support_fold_rows(
    fold: Mapping[str, Any],
    *,
    source_hash: str,
    variant: str,
    target_kind: str,
    support_column: str,
    source_rows: Mapping[str, Mapping[str, Any]],
    require_out_of_sample: bool = False,
) -> dict[str, Any]:
    fold_id = fold.get("fold_id", fold.get("fold"))
    if fold_id is None or not str(fold_id).strip():
        raise FutureValueUncertaintyError("support calibration fold ID is missing")
    train_end_value = fold.get("train_end", fold.get("fit_window_end"))
    validation_start_value = fold.get("validation_start", fold.get("validation_interval_start"))
    validation_end_value = fold.get("validation_end", fold.get("validation_interval_end"))
    if train_end_value is None or validation_start_value is None or validation_end_value is None:
        raise FutureValueUncertaintyError("support calibration fold cutoffs are incomplete")
    train_end = pd.to_datetime(train_end_value, utc=True, errors="coerce")
    validation_start = pd.to_datetime(validation_start_value, utc=True, errors="coerce")
    validation_end = pd.to_datetime(validation_end_value, utc=True, errors="coerce")
    if any(pd.isna(value) for value in (train_end, validation_start, validation_end)):
        raise FutureValueUncertaintyError("support calibration fold cutoffs are invalid")
    train_end = pd.Timestamp(train_end)
    validation_start = pd.Timestamp(validation_start)
    validation_end = pd.Timestamp(validation_end)
    if not train_end < validation_start <= validation_end:
        raise FutureValueUncertaintyError("support calibration fold cutoffs are not strictly chronological")
    fold_source = fold.get("source_receipt_sha256")
    if fold_source is None or str(fold_source).lower() != source_hash:
        raise FutureValueUncertaintyError("support calibration source drift across folds")
    fold_variant = fold.get("variant")
    if fold_variant is None or str(fold_variant) != variant:
        raise FutureValueUncertaintyError("support calibration variant drift across folds")
    if require_out_of_sample:
        if fold.get("out_of_sample") is not True:
            raise FutureValueUncertaintyError(
                "support calibration prior fold is not marked out-of-sample"
            )
        if fold.get("whole_series") is not True:
            raise FutureValueUncertaintyError(
                "support calibration prior fold is not whole-series safe"
            )
    raw_rows = fold.get("rows", fold.get("predictions", fold.get("ledger_rows")))
    if isinstance(raw_rows, pd.DataFrame):
        raw_rows = raw_rows.to_dict("records")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise FutureValueUncertaintyError("support calibration fold rows are missing")
    normalized: list[dict[str, Any]] = []
    seen_games: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise FutureValueUncertaintyError("support calibration row is invalid")
        game_id_value = raw.get("game_id", raw.get("game_uid", raw.get("match_id")))
        if game_id_value is None or not str(game_id_value).strip():
            raise FutureValueUncertaintyError("support calibration game ID is missing")
        game_id = str(game_id_value)
        if game_id in seen_games:
            raise FutureValueUncertaintyError("support calibration game IDs are not unique")
        seen_games.add(game_id)
        series_value = raw.get("series_id", raw.get("series"))
        if series_value is None or not str(series_value).strip():
            raise FutureValueUncertaintyError("support calibration series ID is missing")
        series_id = str(series_value)
        date = _support_date(raw, f"support calibration row {game_id}")
        if date < validation_start or date > validation_end:
            raise FutureValueUncertaintyError("support calibration row is outside its validation window")
        target_value = raw.get("target")
        try:
            target = float(target_value)
        except (TypeError, ValueError) as error:
            raise FutureValueUncertaintyError("support calibration target is missing") from error
        logit, probability = _support_prediction(raw)
        support = _support_value(raw, support_column)
        residual = _support_residual_target(
            target,
            probability,
            logit,
            target_kind=target_kind,
        )
        row_source = raw.get("source_receipt_sha256")
        if row_source is not None and str(row_source).lower() != source_hash:
            raise FutureValueUncertaintyError("support calibration row source drift")
        row_variant = raw.get("variant")
        if row_variant is not None and str(row_variant) != variant:
            raise FutureValueUncertaintyError("support calibration row variant drift")
        normalized.append(
            {
                "fold": str(fold_id),
                "game_id": game_id,
                "series_id": series_id,
                "date": _date_text(date),
                "support": support,
                "prediction_logit": logit,
                "prediction_probability": probability,
                "target": target,
                "residual_target": residual,
            }
        )
    normalized.sort(key=lambda row: row["game_id"])
    rows_hash = _sha256_json(normalized)
    claimed_rows_hash = fold.get("rows_sha256", fold.get("prediction_rows_sha256"))
    if claimed_rows_hash is not None and str(claimed_rows_hash).lower() != rows_hash:
        raise FutureValueUncertaintyError("support calibration fold rows hash does not match")
    return {
        "fold_id": str(fold_id),
        "train_end": _date_text(train_end),
        "validation_start": _date_text(validation_start),
        "validation_end": _date_text(validation_end),
        "source_receipt_sha256": source_hash,
        "variant": variant,
        "rows": normalized,
        "rows_sha256": rows_hash,
    }


def _fit_support_bins(
    rows: Sequence[Mapping[str, Any]],
    *,
    minimum_rows: int,
    minimum_bin_rows: int,
    minimum_bins: int,
    maximum_bins: int,
) -> dict[str, Any]:
    if len(rows) < int(minimum_rows):
        raise FutureValueUncertaintyError("support calibration has insufficient prior rows")
    if minimum_bin_rows < 1 or minimum_bins < 2 or maximum_bins < minimum_bins:
        raise FutureValueUncertaintyError("support calibration bin thresholds are invalid")
    support = np.asarray([float(row["support"]) for row in rows], dtype=float)
    residual = np.asarray([float(row["residual_target"]) for row in rows], dtype=float)
    if not np.isfinite(support).all() or not np.isfinite(residual).all():
        raise FutureValueUncertaintyError("support calibration bin inputs are non-finite")
    unique_support = np.unique(support)
    if unique_support.size < int(minimum_bins):
        raise FutureValueUncertaintyError("support calibration has insufficient support values")
    bin_count = min(int(maximum_bins), len(rows) // int(minimum_bin_rows))
    if bin_count < int(minimum_bins):
        raise FutureValueUncertaintyError("support calibration has insufficient rows per bin")
    order = np.argsort(support, kind="stable")
    raw_bins: list[np.ndarray] = []
    for indexes in np.array_split(order, bin_count):
        if len(indexes):
            raw_bins.append(np.asarray(indexes, dtype=int))
    # Merge adjacent bins with the same support center.  Isotonic regression
    # needs a stable non-decreasing support coordinate.
    bins: list[np.ndarray] = []
    for indexes in raw_bins:
        if bins and float(np.mean(support[bins[-1]])) == float(np.mean(support[indexes])):
            bins[-1] = np.concatenate((bins[-1], indexes))
        else:
            bins.append(indexes)
    if len(bins) < int(minimum_bins):
        raise FutureValueUncertaintyError("support calibration has insufficient distinct bins")
    centers = np.asarray([float(np.mean(support[indexes])) for indexes in bins], dtype=float)
    means = np.asarray([float(np.mean(residual[indexes])) for indexes in bins], dtype=float)
    counts = np.asarray([len(indexes) for indexes in bins], dtype=float)
    if (counts < int(minimum_bin_rows)).any():
        raise FutureValueUncertaintyError("support calibration has an undersized support bin")
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = IsotonicRegression(increasing=False, out_of_bounds="clip")
        fitted = np.asarray(model.fit_transform(centers, means, sample_weight=counts), dtype=float)
    if not np.isfinite(fitted).all() or (fitted < 0.0).any():
        raise FutureValueUncertaintyError("support calibration mapping is non-finite")
    if len(fitted) > 1 and (np.diff(fitted) > 1e-12).any():
        raise FutureValueUncertaintyError("support calibration mapping is not monotonic")
    bin_rows = []
    for indexes, center, mean, fitted_value in zip(bins, centers, means, fitted):
        bin_rows.append(
            {
                "lower_support": float(np.min(support[indexes])),
                "upper_support": float(np.max(support[indexes])),
                "center_support": float(center),
                "rows": int(len(indexes)),
                "mean_residual": float(mean),
                "fitted_residual": float(fitted_value),
            }
        )
    return {
        "method": "weighted_decreasing_isotonic_support_bins",
        "monotonic": "non_increasing_with_support",
        "target_scale": "expected_out_of_sample_residual",
        "minimum_support": float(np.min(support)),
        "maximum_support": float(np.max(support)),
        "training_rows": int(len(rows)),
        "training_game_ids": sorted(str(row["game_id"]) for row in rows),
        "training_game_identity_sha256": _identity_sha256(row["game_id"] for row in rows),
        "bins": bin_rows,
        "mapping_sha256": _sha256_json(bin_rows),
    }


def _apply_support_mapping(support: Sequence[float], mapping: Mapping[str, Any]) -> np.ndarray:
    values = np.asarray([float(value) for value in support], dtype=float)
    bins = mapping.get("bins")
    if not isinstance(bins, list) or len(bins) < 2:
        raise FutureValueUncertaintyError("support calibration mapping bins are missing")
    centers = np.asarray([float(row["center_support"]) for row in bins], dtype=float)
    fitted = np.asarray([float(row["fitted_residual"]) for row in bins], dtype=float)
    if not np.isfinite(values).all() or not np.isfinite(centers).all() or not np.isfinite(fitted).all():
        raise FutureValueUncertaintyError("support calibration mapping is non-finite")
    if np.any(np.diff(centers) <= 0.0) or np.any(np.diff(fitted) > 1e-12):
        raise FutureValueUncertaintyError("support calibration mapping is not canonical monotonic")
    output = np.interp(values, centers, fitted, left=fitted[0], right=fitted[-1])
    if not np.isfinite(output).all() or (output < 0.0).any():
        raise FutureValueUncertaintyError("support calibration output is invalid")
    return output


def build_strict_prior_support_calibration(
    folds: Sequence[Mapping[str, Any]],
    *,
    source_receipt: Mapping[str, Any],
    source_frame: pd.DataFrame,
    variant: str,
    calibration_prior_folds: Sequence[Mapping[str, Any]] | None = None,
    target_kind: str = "log_loss",
    support_column: str = "minimum_effective_support",
    minimum_training_rows: int = SUPPORT_CALIBRATION_DEFAULT_MINIMUM_ROWS,
    minimum_bin_rows: int = SUPPORT_CALIBRATION_DEFAULT_MINIMUM_BIN_ROWS,
    minimum_bins: int = SUPPORT_CALIBRATION_DEFAULT_MINIMUM_BINS,
    maximum_bins: int = SUPPORT_CALIBRATION_DEFAULT_MAXIMUM_BINS,
    minimum_coverage: float = SUPPORT_CALIBRATION_MINIMUM_COVERAGE,
) -> dict[str, Any]:
    """Fit a support-to-residual map from earlier validation folds only.

    Each input fold contains out-of-sample predictions.  An explicit prior
    calibration prelude can seed the first evaluation fold.  Without that
    prelude, the first fold is marked blocked.  Later folds use only rows from
    earlier validation windows.  The default residual is per-row log loss,
    which is a proper scoring residual.  No current-fold target enters its
    own mapping.
    """

    if not isinstance(source_receipt, Mapping):
        raise FutureValueUncertaintyError("support calibration source receipt is required")
    if not isinstance(variant, str) or not variant.strip():
        raise FutureValueUncertaintyError("support calibration variant is missing")
    if target_kind not in {"log_loss", "absolute_logit_residual", "absolute_probability_residual"}:
        raise FutureValueUncertaintyError("unsupported support calibration target")
    if not folds:
        raise FutureValueUncertaintyError("support calibration folds are missing")
    claimed_source_hash = source_receipt.get("receipt_sha256")
    if not isinstance(claimed_source_hash, str) or SHA256_RE.fullmatch(claimed_source_hash) is None:
        raise FutureValueUncertaintyError(
            "support calibration source receipt must carry a verified receipt hash"
        )
    source_hash = _source_receipt_hash(source_receipt)
    if source_hash != claimed_source_hash.lower():
        raise FutureValueUncertaintyError("support calibration source receipt hash does not match payload")
    if SHA256_RE.fullmatch(source_hash) is None:
        raise FutureValueUncertaintyError("support calibration source receipt hash is invalid")
    source_rows = _support_source_rows(source_frame, source_receipt)
    coverage_threshold = float(minimum_coverage)
    if not math.isfinite(coverage_threshold) or not 0.0 <= coverage_threshold <= 1.0:
        raise FutureValueUncertaintyError("support calibration coverage threshold is invalid")
    normalized_prior_folds = [
        _support_fold_rows(
            fold,
            source_hash=source_hash,
            variant=variant,
            target_kind=target_kind,
            support_column=support_column,
            source_rows=source_rows,
            require_out_of_sample=True,
        )
        for fold in (calibration_prior_folds or ())
    ]
    normalized_folds = [
        _support_fold_rows(
            fold,
            source_hash=source_hash,
            variant=variant,
            target_kind=target_kind,
            support_column=support_column,
            source_rows=source_rows,
        )
        for fold in folds
    ]
    all_normalized_folds = [*normalized_prior_folds, *normalized_folds]
    if len({fold["fold_id"] for fold in all_normalized_folds}) != len(all_normalized_folds):
        raise FutureValueUncertaintyError("support calibration fold IDs are not unique")
    prior_fold_ids = {fold["fold_id"] for fold in normalized_prior_folds}
    previous_end: pd.Timestamp | None = None
    seen_games: set[str] = set()
    seen_series: set[str] = set()
    first_evaluation_start = pd.Timestamp(normalized_folds[0]["validation_start"])
    for fold in all_normalized_folds:
        start = pd.Timestamp(fold["validation_start"])
        end = pd.Timestamp(fold["validation_end"])
        if previous_end is not None and not previous_end < start:
            raise FutureValueUncertaintyError("support calibration validation windows overlap")
        if fold["fold_id"] in prior_fold_ids and not end < first_evaluation_start:
            raise FutureValueUncertaintyError(
                "support calibration prior fold is not strictly earlier"
            )
        row_games = {str(row["game_id"]) for row in fold["rows"]}
        row_series = {str(row["series_id"]) for row in fold["rows"]}
        if seen_games & row_games:
            raise FutureValueUncertaintyError("support calibration game IDs overlap across folds")
        if seen_series & row_series:
            raise FutureValueUncertaintyError("support calibration series IDs overlap across folds")
        seen_games.update(row_games)
        seen_series.update(row_series)
        previous_end = end

    for fold in all_normalized_folds:
        _verify_support_source_rows(
            fold["rows"],
            source_rows=source_rows,
            validation_start=pd.Timestamp(fold["validation_start"]),
            validation_end=pd.Timestamp(fold["validation_end"]),
            fold_id=str(fold["fold_id"]),
        )

    output_prior_folds: list[dict[str, Any]] = [
        {
            "fold": fold["fold_id"],
            "validation_start": fold["validation_start"],
            "validation_end": fold["validation_end"],
            "train_end": fold["train_end"],
            "source_receipt_sha256": source_hash,
            "variant": variant,
            "out_of_sample": True,
            "whole_series": True,
            "input_rows": list(fold["rows"]),
            "input_rows_sha256": fold["rows_sha256"],
            "status": "available",
        }
        for fold in normalized_prior_folds
    ]
    output_folds: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    available_folds = 0
    eligible_rows = 0
    calibrated_rows = 0
    blockers: list[str] = []
    prior_rows: list[dict[str, Any]] = [
        row for fold in normalized_prior_folds for row in fold["rows"]
    ]
    initial_prior_row_count = len(prior_rows)
    for fold_index, fold in enumerate(normalized_folds):
        rows = list(fold["rows"])
        eligible = bool(prior_rows)
        mapping: dict[str, Any] | None = None
        fold_blockers: list[str] = []
        if not eligible:
            fold_blockers.append("calibration_prior_validation_folds_missing")
        else:
            eligible_rows += len(rows)
            prior_end = pd.Timestamp(
                normalized_prior_folds[-1]["validation_end"]
                if fold_index == 0 and normalized_prior_folds
                else normalized_folds[fold_index - 1]["validation_end"]
            )
            current_start = pd.Timestamp(fold["validation_start"])
            if not prior_end < current_start:
                raise FutureValueUncertaintyError("support calibration prior cutoff is not strict")
            try:
                mapping = _fit_support_bins(
                    prior_rows,
                    minimum_rows=int(minimum_training_rows),
                    minimum_bin_rows=int(minimum_bin_rows),
                    minimum_bins=int(minimum_bins),
                    maximum_bins=int(maximum_bins),
                )
            except FutureValueUncertaintyError as error:
                fold_blockers.append(str(error).replace("support calibration ", "calibration_"))
            if mapping is not None:
                available_folds += 1
        # Bind the prior population even when the fit is blocked by an
        # insufficient-support threshold.  This makes the reason for the
        # blocked fold auditable and prevents an empty list from hiding a
        # chronology or source mismatch.
        calibration_training_ids = [
            str(row["game_id"])
            for prior_fold in [
                *normalized_prior_folds,
                *normalized_folds[:fold_index],
            ]
            for row in prior_fold["rows"]
        ]
        calibration_training_hash = _identity_sha256(calibration_training_ids)
        for row in rows:
            calibrated = None if mapping is None else float(_apply_support_mapping([row["support"]], mapping)[0])
            if calibrated is not None:
                calibrated_rows += 1
            output_rows.append(
                {
                    "fold": fold["fold_id"],
                    "game_id": row["game_id"],
                    "series_id": row["series_id"],
                    "date": row["date"],
                    "support": float(row["support"]),
                    "raw_uncertainty_proxy": float(1.0 / math.sqrt(1.0 + float(row["support"]))),
                    "calibrated_uncertainty": calibrated,
                    "calibration_status": "available" if calibrated is not None else "blocked",
                }
            )
        output_folds.append(
            {
                "fold": fold["fold_id"],
                "validation_start": fold["validation_start"],
                "validation_end": fold["validation_end"],
                "train_end": fold["train_end"],
                "source_receipt_sha256": source_hash,
                "variant": variant,
                "input_rows": len(rows),
                "input_rows_sha256": fold["rows_sha256"],
                "calibration_input_rows": list(fold["rows"]),
                "status": "available" if mapping is not None else "blocked",
                "blockers": sorted(set(fold_blockers)),
                "calibration_training_game_count": len(calibration_training_ids),
                "calibration_training_game_ids": calibration_training_ids,
                "calibration_training_game_identity_sha256": calibration_training_hash,
                "mapping": mapping,
            }
        )
        blockers.extend(fold_blockers)
        prior_rows.extend(rows)

    eligible_fold_count = (
        len(normalized_folds)
        if normalized_prior_folds
        else max(0, len(normalized_folds) - 1)
    )
    row_fraction = calibrated_rows / eligible_rows if eligible_rows else 0.0
    complete_enough = bool(
        eligible_fold_count > 0
        and available_folds == eligible_fold_count
        and row_fraction >= coverage_threshold
    )
    coverage = {
        "eligible_fold_count": eligible_fold_count,
        "available_fold_count": available_folds,
        "eligible_row_count": eligible_rows,
        "calibrated_row_count": calibrated_rows,
        "calibrated_row_fraction": float(row_fraction),
        "minimum_coverage_threshold": coverage_threshold,
        "complete_enough": complete_enough,
        "first_fold_without_history": not bool(normalized_prior_folds),
        "calibration_prior_fold_count": len(normalized_prior_folds),
        "calibration_prior_row_count": initial_prior_row_count,
    }
    if not complete_enough:
        blockers.append("support_calibration_coverage_below_threshold")
    artifact_payload: dict[str, Any] = {
        "schema_version": SUPPORT_CALIBRATION_SCHEMA_VERSION,
        # The first fold is intentionally blocked because no prior outcome
        # history exists.  Keep the complete-enough flag separate from the
        # artifact status so callers cannot mistake later-fold coverage for a
        # fully calibrated chronological evaluation.
        "status": (
            "research_only"
            if complete_enough and "calibration_prior_validation_folds_missing" not in blockers
            else "research_only_partial"
        ),
        "variant": variant,
        "source": {
            "source_receipt_sha256": source_hash,
            "source_as_of": source_receipt.get("source_as_of"),
            "source_game_count": source_receipt.get("source_game_count"),
            "source_identity_sha256": source_receipt.get("source_identity_sha256"),
        },
        "target": {
            "kind": target_kind,
            "description": (
                "per-row held-out log loss, a proper scoring residual"
                if target_kind == "log_loss"
                else "absolute residual from the signed unit outcome logit"
                if target_kind == "absolute_logit_residual"
                else "absolute held-out probability residual"
            ),
            "uses_current_validation_targets_for_fit": False,
        },
        "support": {
            "column": support_column,
            "raw_proxy": "1/sqrt(1+support)",
            "mapping": "decreasing_isotonic_support_bins",
            "minimum_training_rows": int(minimum_training_rows),
            "minimum_bin_rows": int(minimum_bin_rows),
            "minimum_bins": int(minimum_bins),
            "maximum_bins": int(maximum_bins),
        },
        "coverage": coverage,
        "calibration_prior_folds": output_prior_folds,
        "folds": output_folds,
        "rows": sorted(output_rows, key=lambda row: (str(row["fold"]), str(row["game_id"]))),
        "blockers": sorted(set(blockers)),
        "authority": dict(SUPPORT_CALIBRATION_AUTHORITY),
    }
    artifact_hash = _sha256_json(artifact_payload)
    receipt_payload: dict[str, Any] = {
        "schema_version": SUPPORT_CALIBRATION_RECEIPT_SCHEMA_VERSION,
        "status": artifact_payload["status"],
        "variant": variant,
        "source_receipt_sha256": source_hash,
        "artifact_sha256": artifact_hash,
        "folds_sha256": _sha256_json(output_folds),
        "calibration_prior_folds_sha256": _sha256_json(output_prior_folds),
        "calibration_prior_rows_sha256": _sha256_json(
            [row for fold in output_prior_folds for row in fold["input_rows"]]
        ),
        "rows_sha256": _sha256_json(artifact_payload["rows"]),
        "coverage": coverage,
        "calibration_training_game_ids": sorted(
            {
                str(game_id)
                for fold in output_folds
                for game_id in fold["calibration_training_game_ids"]
            }
        ),
        "authority": dict(SUPPORT_CALIBRATION_AUTHORITY),
    }
    receipt_payload["receipt_sha256"] = _sha256_json(receipt_payload)
    artifact = dict(artifact_payload)
    artifact["artifact_sha256"] = artifact_hash
    artifact["receipt"] = receipt_payload
    artifact["receipt_sha256"] = receipt_payload["receipt_sha256"]
    return artifact


def verify_support_calibration_artifact(
    artifact: Mapping[str, Any],
    *,
    source_frame: pd.DataFrame,
    expected_source_receipt_sha256: str | None = None,
    expected_variant: str | None = None,
) -> bool:
    """Verify hashes, source binding, dates, fold isolation, and mappings."""

    if not isinstance(artifact, Mapping):
        raise FutureValueUncertaintyError("support calibration artifact is invalid")
    claimed_artifact = artifact.get("artifact_sha256")
    if not isinstance(claimed_artifact, str) or SHA256_RE.fullmatch(claimed_artifact) is None:
        raise FutureValueUncertaintyError("support calibration artifact hash is invalid")
    artifact_payload = dict(artifact)
    artifact_payload.pop("artifact_sha256", None)
    artifact_payload.pop("receipt", None)
    artifact_payload.pop("receipt_sha256", None)
    if _sha256_json(artifact_payload) != claimed_artifact.lower():
        raise FutureValueUncertaintyError("support calibration artifact hash does not match")
    if artifact.get("schema_version") != SUPPORT_CALIBRATION_SCHEMA_VERSION:
        raise FutureValueUncertaintyError("support calibration artifact schema is invalid")
    receipt = artifact.get("receipt")
    if not isinstance(receipt, Mapping):
        raise FutureValueUncertaintyError("support calibration receipt is missing")
    claimed_receipt = receipt.get("receipt_sha256")
    if not isinstance(claimed_receipt, str) or SHA256_RE.fullmatch(claimed_receipt) is None:
        raise FutureValueUncertaintyError("support calibration receipt hash is invalid")
    receipt_payload = dict(receipt)
    receipt_payload.pop("receipt_sha256", None)
    if _sha256_json(receipt_payload) != claimed_receipt.lower() or artifact.get("receipt_sha256") != claimed_receipt:
        raise FutureValueUncertaintyError("support calibration receipt hash does not match")
    if receipt.get("artifact_sha256") != claimed_artifact:
        raise FutureValueUncertaintyError("support calibration receipt does not bind artifact")
    source = artifact.get("source")
    source_hash = artifact.get("source", {}).get("source_receipt_sha256") if isinstance(source, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(source_hash, str) or SHA256_RE.fullmatch(source_hash) is None:
        raise FutureValueUncertaintyError("support calibration source binding is invalid")
    if expected_source_receipt_sha256 is not None and source_hash != str(expected_source_receipt_sha256).lower():
        raise FutureValueUncertaintyError("support calibration source receipt does not match expected source")
    if expected_variant is not None and artifact.get("variant") != expected_variant:
        raise FutureValueUncertaintyError("support calibration variant does not match expected variant")
    if artifact.get("variant") != receipt.get("variant") or receipt.get("source_receipt_sha256") != source_hash:
        raise FutureValueUncertaintyError("support calibration receipt binding is inconsistent")
    source_rows = _support_source_rows(
        source_frame,
        {
            "receipt_sha256": source_hash,
            **dict(source),
        },
    )
    for value in (artifact.get("authority"), receipt.get("authority")):
        if not isinstance(value, Mapping) or dict(value) != SUPPORT_CALIBRATION_AUTHORITY:
            raise FutureValueUncertaintyError("support calibration authority grants access")
    folds = artifact.get("folds")
    calibration_prior_folds = artifact.get("calibration_prior_folds", [])
    rows = artifact.get("rows")
    if (
        not isinstance(folds, list)
        or not isinstance(calibration_prior_folds, list)
        or not isinstance(rows, list)
        or not folds
        or not rows
    ):
        raise FutureValueUncertaintyError("support calibration fold rows are missing")
    if (
        receipt.get("folds_sha256") != _sha256_json(folds)
        or receipt.get("rows_sha256") != _sha256_json(rows)
        or receipt.get("calibration_prior_folds_sha256", _sha256_json(calibration_prior_folds))
        != _sha256_json(calibration_prior_folds)
    ):
        raise FutureValueUncertaintyError("support calibration receipt does not bind rows")
    fold_ids = [str(fold.get("fold")) for fold in folds if isinstance(fold, Mapping)]
    if len(fold_ids) != len(folds) or len(set(fold_ids)) != len(fold_ids):
        raise FutureValueUncertaintyError("support calibration fold IDs are invalid")
    row_by_fold: dict[str, list[Mapping[str, Any]]] = {fold_id: [] for fold_id in fold_ids}
    for row in rows:
        if not isinstance(row, Mapping) or str(row.get("fold")) not in row_by_fold:
            raise FutureValueUncertaintyError("support calibration output row is invalid")
        row_by_fold[str(row["fold"])].append(row)
    prior_fold_ids = [
        str(fold.get("fold"))
        for fold in calibration_prior_folds
        if isinstance(fold, Mapping)
    ]
    if len(prior_fold_ids) != len(calibration_prior_folds) or len(set(prior_fold_ids)) != len(prior_fold_ids):
        raise FutureValueUncertaintyError("support calibration prior fold IDs are invalid")
    if set(prior_fold_ids) & set(fold_ids):
        raise FutureValueUncertaintyError("support calibration prior fold IDs overlap evaluation folds")
    prior_row_by_fold: dict[str, list[Mapping[str, Any]]] = {}
    for prior_fold in calibration_prior_folds:
        if not isinstance(prior_fold, Mapping):
            raise FutureValueUncertaintyError("support calibration prior fold is invalid")
        input_rows = prior_fold.get("input_rows")
        if not isinstance(input_rows, list) or not input_rows:
            raise FutureValueUncertaintyError("support calibration prior rows are missing")
        claimed_input_hash = prior_fold.get("input_rows_sha256")
        if claimed_input_hash != _sha256_json(input_rows):
            raise FutureValueUncertaintyError("support calibration prior row hash is invalid")
        if prior_fold.get("source_receipt_sha256") != source_hash or prior_fold.get("variant") != artifact.get("variant"):
            raise FutureValueUncertaintyError("support calibration prior binding is inconsistent")
        if prior_fold.get("out_of_sample") is not True or prior_fold.get("whole_series") is not True:
            raise FutureValueUncertaintyError("support calibration prior fold authority is invalid")
        prior_start = pd.to_datetime(
            prior_fold.get("validation_start"), utc=True, errors="coerce"
        )
        prior_end = pd.to_datetime(
            prior_fold.get("validation_end"), utc=True, errors="coerce"
        )
        if pd.isna(prior_start) or pd.isna(prior_end):
            raise FutureValueUncertaintyError("support calibration prior chronology is invalid")
        _verify_support_source_rows(
            input_rows,
            source_rows=source_rows,
            validation_start=pd.Timestamp(prior_start),
            validation_end=pd.Timestamp(prior_end),
            fold_id=str(prior_fold["fold"]),
        )
        prior_row_by_fold[str(prior_fold["fold"])] = input_rows
    if receipt.get("calibration_prior_rows_sha256") is not None:
        expected_prior_rows = [
            row for prior_fold in calibration_prior_folds for row in prior_fold["input_rows"]
        ]
        if receipt.get("calibration_prior_rows_sha256") != _sha256_json(expected_prior_rows):
            raise FutureValueUncertaintyError("support calibration prior rows are not bound")
    previous_end: pd.Timestamp | None = None
    seen_games: set[str] = set()
    series_fold: dict[str, str] = {}
    expected_calibration_ids: set[str] = set()
    evaluation_first_start = pd.to_datetime(
        folds[0].get("validation_start"), utc=True, errors="coerce"
    )
    if pd.isna(evaluation_first_start):
        raise FutureValueUncertaintyError("support calibration evaluation chronology is invalid")
    evaluation_first_start = pd.Timestamp(evaluation_first_start)
    for prior_fold in calibration_prior_folds:
        start = pd.to_datetime(prior_fold.get("validation_start"), utc=True, errors="coerce")
        end = pd.to_datetime(prior_fold.get("validation_end"), utc=True, errors="coerce")
        train_end = pd.to_datetime(prior_fold.get("train_end"), utc=True, errors="coerce")
        if any(pd.isna(value) for value in (start, end, train_end)) or not train_end < start <= end:
            raise FutureValueUncertaintyError("support calibration prior fold chronology is invalid")
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if not end < evaluation_first_start:
            raise FutureValueUncertaintyError("support calibration prior fold is not strictly earlier")
        if previous_end is not None and not previous_end < start:
            raise FutureValueUncertaintyError("support calibration prior fold windows overlap")
        previous_end = end
        for row in prior_row_by_fold[str(prior_fold["fold"])]:
            if not isinstance(row, Mapping):
                raise FutureValueUncertaintyError("support calibration prior row is invalid")
            date = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            game_id = str(row.get("game_id", ""))
            series_id = str(row.get("series_id", ""))
            if (
                pd.isna(date)
                or not start <= pd.Timestamp(date) <= end
                or str(row.get("fold")) != str(prior_fold["fold"])
                or not game_id
                or not series_id
                or game_id in seen_games
                or (
                    series_id in series_fold
                    and series_fold[series_id] != str(prior_fold["fold"])
                )
            ):
                raise FutureValueUncertaintyError("support calibration prior identities are invalid")
            try:
                support = float(row.get("support"))
                probability = float(row.get("prediction_probability"))
                logit = float(row.get("prediction_logit"))
                target = float(row.get("residual_target"))
            except (TypeError, ValueError) as error:
                raise FutureValueUncertaintyError("support calibration prior values are invalid") from error
            if (
                not math.isfinite(support)
                or support < 0.0
                or not math.isfinite(probability)
                or not 0.0 < probability < 1.0
                or not math.isfinite(logit)
                or not math.isfinite(target)
                or target < 0.0
            ):
                raise FutureValueUncertaintyError("support calibration prior values are invalid")
            seen_games.add(game_id)
            series_fold.setdefault(series_id, str(prior_fold["fold"]))
            expected_calibration_ids.add(game_id)
    for index, fold in enumerate(folds):
        if not isinstance(fold, Mapping):
            raise FutureValueUncertaintyError("support calibration fold is invalid")
        start = pd.to_datetime(fold.get("validation_start"), utc=True, errors="coerce")
        end = pd.to_datetime(fold.get("validation_end"), utc=True, errors="coerce")
        train_end = pd.to_datetime(fold.get("train_end"), utc=True, errors="coerce")
        if any(pd.isna(value) for value in (start, end, train_end)) or not train_end < start <= end:
            raise FutureValueUncertaintyError("support calibration fold chronology is invalid")
        start = pd.Timestamp(start)
        end = pd.Timestamp(end)
        if previous_end is not None and not previous_end < start:
            raise FutureValueUncertaintyError("support calibration fold windows overlap")
        previous_end = end
        fold_rows = row_by_fold[str(fold["fold"])]
        if not fold_rows or int(fold.get("input_rows", -1)) != len(fold_rows):
            raise FutureValueUncertaintyError("support calibration fold row count is invalid")
        input_hash = fold.get("input_rows_sha256")
        if not isinstance(input_hash, str) or SHA256_RE.fullmatch(input_hash) is None:
            raise FutureValueUncertaintyError("support calibration input row hash is missing")
        for row in fold_rows:
            date = pd.to_datetime(row.get("date"), utc=True, errors="coerce")
            if pd.isna(date) or not start <= pd.Timestamp(date) <= end:
                raise FutureValueUncertaintyError("support calibration output date is outside its fold")
            game_id = str(row.get("game_id", ""))
            series_id = str(row.get("series_id", ""))
            fold_id = str(fold["fold"])
            if (
                not game_id
                or not series_id
                or game_id in seen_games
                or (series_id in series_fold and series_fold[series_id] != fold_id)
            ):
                raise FutureValueUncertaintyError("support calibration output identities overlap")
            seen_games.add(game_id)
            series_fold.setdefault(series_id, fold_id)
            support = float(row.get("support", float("nan")))
            if not math.isfinite(support) or support < 0.0:
                raise FutureValueUncertaintyError("support calibration output support is invalid")
        calibration_ids = fold.get("calibration_training_game_ids")
        if not isinstance(calibration_ids, list):
            raise FutureValueUncertaintyError("support calibration training IDs are missing")
        calibration_hash = fold.get("calibration_training_game_identity_sha256")
        if calibration_hash != _identity_sha256(calibration_ids):
            raise FutureValueUncertaintyError("support calibration training ID hash is invalid")
        input_rows = fold.get("calibration_input_rows")
        if not isinstance(input_rows, list):
            raise FutureValueUncertaintyError("support calibration input rows are missing")
        if fold.get("input_rows_sha256") != _sha256_json(input_rows):
            raise FutureValueUncertaintyError("support calibration input rows are not bound")
        _verify_support_source_rows(
            input_rows,
            source_rows=source_rows,
            validation_start=start,
            validation_end=end,
            fold_id=str(fold["fold"]),
        )
        if len(input_rows) != len(fold_rows):
            raise FutureValueUncertaintyError("support calibration input row count is invalid")
        input_by_game = {
            str(row.get("game_id")): row
            for row in input_rows
            if isinstance(row, Mapping)
        }
        if len(input_by_game) != len(input_rows):
            raise FutureValueUncertaintyError("support calibration input game IDs are invalid")
        for output_row in fold_rows:
            input_row = input_by_game.get(str(output_row.get("game_id")))
            if input_row is None:
                raise FutureValueUncertaintyError(
                    "support calibration input IDs do not match output rows"
                )
            for field in ("fold", "game_id", "series_id", "date"):
                if str(input_row.get(field)) != str(output_row.get(field)):
                    raise FutureValueUncertaintyError(
                        "support calibration input identity does not match output rows"
                    )
            try:
                input_support = float(input_row.get("support"))
                output_support = float(output_row.get("support"))
            except (TypeError, ValueError) as error:
                raise FutureValueUncertaintyError(
                    "support calibration input support is invalid"
                ) from error
            if not math.isclose(input_support, output_support, rel_tol=0.0, abs_tol=1e-12):
                raise FutureValueUncertaintyError(
                    "support calibration input support does not match output rows"
                )
        expected_prior = sorted(
            [
                str(row["game_id"])
                for prior_fold in calibration_prior_folds
                for row in prior_fold["input_rows"]
            ]
            + [
                str(row["game_id"])
                for prior_fold in folds[:index]
                for row in row_by_fold[str(prior_fold["fold"])]
            ]
        )
        if sorted(str(value) for value in calibration_ids) != expected_prior:
            raise FutureValueUncertaintyError("support calibration training IDs are not strictly prior")
        expected_calibration_ids.update(str(value) for value in calibration_ids)
        mapping = fold.get("mapping")
        if str(fold.get("status")) == "available":
            if not isinstance(mapping, Mapping):
                raise FutureValueUncertaintyError("support calibration mapping is missing")
            bins = mapping.get("bins")
            if not isinstance(bins, list) or len(bins) < 2:
                raise FutureValueUncertaintyError("support calibration mapping bins are invalid")
            centers = [float(bin_row["center_support"]) for bin_row in bins]
            fitted = [float(bin_row["fitted_residual"]) for bin_row in bins]
            if any(not math.isfinite(value) for value in (*centers, *fitted)) or any(
                right <= left for left, right in zip(centers, centers[1:])
            ) or any(right > left + 1e-12 for left, right in zip(fitted, fitted[1:])):
                raise FutureValueUncertaintyError("support calibration mapping is not monotonic")
        else:
            if mapping is not None:
                raise FutureValueUncertaintyError("blocked support calibration fold has a mapping")
            if any(row.get("calibrated_uncertainty") is not None for row in fold_rows):
                raise FutureValueUncertaintyError("blocked support calibration fold has calibrated output")
    if sorted(str(value) for value in receipt.get("calibration_training_game_ids", [])) != sorted(expected_calibration_ids):
        raise FutureValueUncertaintyError("support calibration receipt training IDs are invalid")
    coverage = artifact.get("coverage")
    if not isinstance(coverage, Mapping) or dict(coverage) != dict(receipt.get("coverage", {})):
        raise FutureValueUncertaintyError("support calibration coverage binding is invalid")
    return True


def apply_strict_prior_support_calibration(
    artifact: Mapping[str, Any],
    support: Sequence[float] | pd.Series,
    *,
    fold_id: Any,
    source_frame: pd.DataFrame,
    expected_source_receipt_sha256: str | None = None,
    expected_variant: str | None = None,
) -> pd.Series:
    """Apply one later-fold mapping after complete receipt verification."""

    verify_support_calibration_artifact(
        artifact,
        source_frame=source_frame,
        expected_source_receipt_sha256=expected_source_receipt_sha256,
        expected_variant=expected_variant,
    )
    wanted = str(fold_id)
    fold = next((value for value in artifact["folds"] if str(value.get("fold")) == wanted), None)
    if not isinstance(fold, Mapping) or fold.get("status") != "available" or not isinstance(fold.get("mapping"), Mapping):
        raise FutureValueUncertaintyError("support calibration has no verified prior mapping for this fold")
    values = _apply_support_mapping([float(value) for value in support], fold["mapping"])
    index = support.index if isinstance(support, pd.Series) else None
    return pd.Series(values, index=index, dtype=float, name="calibrated_support_uncertainty")


# Short aliases make the contract easy to find from research notebooks.
fit_strict_prior_support_calibration = build_strict_prior_support_calibration
build_support_uncertainty_calibration = build_strict_prior_support_calibration
verify_strict_prior_support_calibration = verify_support_calibration_artifact
verify_support_uncertainty_calibration = verify_support_calibration_artifact
apply_support_uncertainty_calibration = apply_strict_prior_support_calibration
calibrate_support_uncertainty = apply_strict_prior_support_calibration


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
    "SUPPORT_CALIBRATION_AUTHORITY",
    "SUPPORT_CALIBRATION_SCHEMA_VERSION",
    "SUPPORT_CALIBRATION_RECEIPT_SCHEMA_VERSION",
    "apply_strict_prior_support_calibration",
    "apply_support_uncertainty_calibration",
    "bootstrap_future_value_uncertainty",
    "bootstrap_future_value_model_uncertainty",
    "build_strict_prior_support_calibration",
    "build_support_uncertainty_calibration",
    "calibrate_support_uncertainty",
    "cluster_bootstrap_weights",
    "fit_strict_prior_support_calibration",
    "run_future_value_uncertainty",
    "verify_strict_prior_support_calibration",
    "verify_support_calibration_artifact",
    "verify_support_uncertainty_calibration",
    "verify_uncertainty_receipt",
]
