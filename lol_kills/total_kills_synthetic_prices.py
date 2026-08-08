#!/usr/bin/env python3
"""Private, benchmark-only synthetic prices for total kills.

This module is deliberately separate from the live serving path.  It consumes
the frozen, receipt-backed GRID/Riot cohort and produces reproducible research
artifacts: chronological series-held-out point forecasts, calibrated
over/under probabilities, and no-vig synthetic fair odds.  It never consumes
bookmaker prices and never authorizes a bet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from lol_kills.live_totals_candidate import development_candidate_path
from lol_kills.grid_market_evaluation import (
    CHECKPOINTS,
    LEAGUES,
    EvaluationError,
    chronological_series_split,
    load_cohort,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "data"
    / "lol"
    / "warehouse"
    / "private_grid"
    / "market_cohort"
    / "v1"
    / "manifests"
    / "evaluation-cohort-455c3c75760275f05c5369471b2b61c58e838f26f0990c93d1ca2134456538bc.json"
)
DEFAULT_CURRENT_MODEL = development_candidate_path(ROOT)
DEFAULT_CATALOG = Path.home() / ".codex" / "skills" / "query-grid-research" / "assets" / "grid-capability-catalog.v1.json"
SCHEMA_VERSION = "scryglass.synthetic-total-kills-prices.v1"
SPLIT_FRACTIONS = (0.50, 0.15, 0.15, 0.20)
MIN_CALIBRATION_ROWS = 40
MIN_TEST_ROWS = 40
MIN_SELECTION_RELATIVE_IMPROVEMENT = 0.005
MAX_CDF_ERROR = 0.10
MAX_ECE = 0.10
BOOTSTRAP_REPLICATES = 400
BOOTSTRAP_SEED = 20260730
DEFAULT_LINES = tuple(float(value) + 0.5 for value in range(16, 46, 2))


class SyntheticPriceError(RuntimeError):
    """Raised when a price artifact cannot be built honestly."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    if not actual or len(actual) != len(predicted):
        return {"n": 0, "rmse": None, "mae": None}
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    error = p - y
    return {
        "n": int(len(y)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
    }


def _log_loss(actual: Sequence[int], probability: Sequence[float]) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


def _ece(actual: Sequence[int], probability: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(probability, dtype=float)
    if not len(y):
        return float("nan")
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (
            p <= edges[index + 1]
            if index == bins - 1
            else p < edges[index + 1]
        )
        if np.any(mask):
            result += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return float(result)


def _classification_metrics(actual: Sequence[int], probability: Sequence[float]) -> dict[str, Any]:
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "n": int(len(y)),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": _log_loss(actual, probability),
        "ece": _ece(actual, probability),
        "observed_rate": float(y.mean()) if len(y) else None,
        "predicted_rate": float(p.mean()) if len(p) else None,
    }


def _rows_for_total_kills(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        total = row.get("total_kills")
        current = row.get("current_kills")
        if not _finite(total) or not _finite(current):
            continue
        total_value = float(total)
        current_value = float(current)
        if current_value < 0 or total_value < current_value:
            continue
        result.append({**dict(row), "remaining_kills": total_value - current_value})
    return result


def _feature_names(family: str) -> list[str]:
    names = [
        "intercept",
        *[f"checkpoint:{minute}" for minute in CHECKPOINTS if minute != 10],
        *[f"league:{league}" for league in LEAGUES if league != "CBLOL"],
    ]
    if family in {"kills", "objective_state"}:
        names.append("current_kills")
    if family == "objective_state":
        names.extend(
            [
                "total_dragons_now",
                "total_barons_now",
                "total_inhibitors_now",
            ]
        )
    return names


def _feature_matrix(rows: Sequence[Mapping[str, Any]], family: str) -> np.ndarray:
    names = _feature_names(family)
    matrix = []
    for row in rows:
        values = {name: 0.0 for name in names}
        values["intercept"] = 1.0
        checkpoint_name = f"checkpoint:{int(row['checkpoint'])}"
        league_name = f"league:{row['league']}"
        if checkpoint_name in values:
            values[checkpoint_name] = 1.0
        if league_name in values:
            values[league_name] = 1.0
        if family in {"kills", "objective_state"}:
            values["current_kills"] = float(row["current_kills"])
        if family == "objective_state":
            values["total_dragons_now"] = float(row["total_dragons_now"])
            values["total_barons_now"] = float(row["total_barons_now"])
            values["total_inhibitors_now"] = float(row["total_inhibitors_now"])
        matrix.append([values[name] for name in names])
    return np.asarray(matrix, dtype=float)


def _fit_ridge(rows: Sequence[Mapping[str, Any]], family: str) -> dict[str, Any]:
    if not rows:
        raise SyntheticPriceError("cannot fit a model with no rows")
    names = _feature_names(family)
    raw = _feature_matrix(rows, family)
    centers = raw.mean(axis=0)
    scales = raw.std(axis=0)
    centers[0] = 0.0
    scales[0] = 1.0
    scales[scales < 1e-9] = 1.0
    x = (raw - centers) / scales
    x[:, 0] = 1.0
    y = np.asarray([float(row["remaining_kills"]) for row in rows], dtype=float)
    penalty = np.eye(x.shape[1]) * 10.0
    penalty[0, 0] = 0.0
    # ``einsum`` keeps this deterministic on the bundled macOS NumPy build;
    # the BLAS matmul path can emit spurious floating-point warnings for the
    # same small, finite design matrix.
    gram = np.einsum("ni,nj->ij", x, x) + penalty
    cross = np.einsum("ni,n->i", x, y)
    beta = np.linalg.solve(gram, cross)
    if not np.isfinite(beta).all():
        raise SyntheticPriceError("non-finite synthetic price model")
    return {
        "family": family,
        "feature_names": names,
        "centers": [float(value) for value in centers],
        "scales": [float(value) for value in scales],
        "coefficients": [float(value) for value in beta],
        "ridge_lambda": 10.0,
    }


def _predict_ridge(model: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    if not rows:
        return np.asarray([], dtype=float)
    raw = _feature_matrix(rows, str(model["family"]))
    centers = np.asarray(model["centers"], dtype=float)
    scales = np.asarray(model["scales"], dtype=float)
    x = (raw - centers) / scales
    x[:, 0] = 1.0
    beta = np.asarray(model["coefficients"], dtype=float)
    return np.maximum(0.0, np.einsum("ij,j->i", x, beta))


def _prior_predictions(
    fit_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    global_values: list[float] = []
    for row in fit_rows:
        grouped[(str(row["league"]), int(row["checkpoint"]))].append(float(row["remaining_kills"]))
        global_values.append(float(row["remaining_kills"]))
    fallback = float(np.mean(global_values)) if global_values else 0.0
    return np.asarray(
        [
            float(np.mean(grouped.get((str(row["league"]), int(row["checkpoint"])), [fallback])))
            for row in rows
        ],
        dtype=float,
    )


def _predict_family(
    family: str,
    fit_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, np.ndarray]:
    if family == "prior":
        return None, _prior_predictions(fit_rows, rows)
    model = _fit_ridge(fit_rows, family)
    return model, _predict_ridge(model, rows)


def _nearest_rank(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


def _cdf_calibration(residuals: Sequence[float], actual: Sequence[float], predicted: Sequence[float]) -> dict[str, Any]:
    if not residuals or not actual:
        return {"status": "unavailable", "max_absolute_error": None, "points": []}
    points = []
    for nominal in (0.10, 0.25, 0.50, 0.75, 0.90):
        threshold = _nearest_rank(residuals, nominal)
        observed = sum(
            float(y) - float(p) <= threshold for y, p in zip(actual, predicted)
        ) / len(actual)
        points.append(
            {
                "nominal": nominal,
                "observed": float(observed),
                "absolute_error": abs(float(observed) - nominal),
            }
        )
    maximum = max(float(point["absolute_error"]) for point in points)
    return {
        "status": "passed" if maximum <= MAX_CDF_ERROR else "failed",
        "max_absolute_error": maximum,
        "threshold": MAX_CDF_ERROR,
        "points": points,
    }


def _under_probability(residuals: Sequence[float], predicted_remaining: float, current_kills: float, line: float) -> float:
    # Half-kill settlement makes under(line) equivalent to total_kills < line.
    cutoff = float(line) - float(current_kills) - float(predicted_remaining)
    count = sum(float(residual) <= cutoff for residual in residuals)
    return float((count + 0.5) / (len(residuals) + 1.0))


def _fair_odds(probability: float) -> float:
    return float(1.0 / min(max(float(probability), 1e-6), 1.0 - 1e-6))


def _line_key(line: float) -> str:
    """Use the same stable key convention as the frozen line evaluation."""
    return str(float(line))


def _line_calibration_blockers(
    artifact: Mapping[str, Any], *, checkpoint: int, line: float
) -> list[str]:
    """Return fail-closed blockers for one exact checkpoint/line pair.

    A fitted residual distribution can mechanically price any half-kill line.
    That does not make an unevaluated or failed line calibrated.  Runtime
    pricing therefore binds to both the all-checkpoint line summary and the
    exact checkpoint metric persisted by the held-out evaluation.
    """
    key = _line_key(line)
    heldout = artifact.get("heldout") or {}
    summary = (heldout.get("line_summary") or {}).get(key)
    metrics = ((heldout.get("line_metrics") or {}).get(key) or {}).get(
        str(checkpoint)
    )
    blockers: list[str] = []
    if not isinstance(summary, Mapping):
        blockers.append(f"line_not_evaluated:{key}")
    elif summary.get("status") != "passes_research_calibration":
        blockers.append(f"line_calibration_unavailable:{key}")
    if not isinstance(metrics, Mapping):
        blockers.append(f"checkpoint_line_not_evaluated:{checkpoint}:{key}")
    else:
        try:
            ece = float(metrics["ece"])
            rate_error = abs(
                float(metrics["predicted_rate"])
                - float(metrics["observed_rate"])
            )
        except (KeyError, TypeError, ValueError):
            blockers.append(f"checkpoint_line_metrics_invalid:{checkpoint}:{key}")
        else:
            if ece > MAX_ECE or rate_error > MAX_ECE:
                blockers.append(
                    f"checkpoint_line_calibration_failed:{checkpoint}:{key}"
                )
    return blockers


def price_synthetic_lines(
    artifact: Mapping[str, Any],
    *,
    league: str,
    checkpoint: int,
    current_kills: int | float | None,
    total_dragons_now: int | float | None,
    total_barons_now: int | float | None,
    total_inhibitors_now: int | float | None,
    lines: Sequence[float],
) -> dict[str, Any]:
    """Price a manual state against the frozen research artifact only.

    This is intentionally not wired into a live endpoint.  It accepts exact
    checkpoints and the same at-or-before objective fields used by the
    evaluated family, and returns no edge/EV or bookmaker comparison.
    """
    blockers: list[str] = []
    if artifact.get("schema_version") != SCHEMA_VERSION:
        blockers.append("artifact_schema_unrecognized")
    if league not in LEAGUES:
        blockers.append(f"league_unavailable:{league}")
    if checkpoint not in CHECKPOINTS:
        blockers.append(f"checkpoint_not_validated:{checkpoint}")
    if current_kills is None or not _finite(current_kills) or float(current_kills) < 0:
        blockers.append("current_kills_invalid")
    for name, value in (
        ("total_dragons_now", total_dragons_now),
        ("total_barons_now", total_barons_now),
        ("total_inhibitors_now", total_inhibitors_now),
    ):
        if value is None or not _finite(value) or float(value) < 0:
            blockers.append(f"{name}_missing_or_invalid")
    for line in lines:
        if float(line) <= 0 or float(line).is_integer():
            blockers.append("line_not_half_kill")
    residuals = (artifact.get("calibration_residuals_by_checkpoint") or {}).get(str(checkpoint))
    if not residuals or len(residuals) < MIN_CALIBRATION_ROWS:
        blockers.append("checkpoint_calibration_unavailable")
    model = artifact.get("selected_model")
    if not model or model.get("family") != "objective_state":
        blockers.append("selected_model_unavailable_for_manual_objective_state")
    if blockers:
        return {
            "status": "unavailable",
            "blockers": sorted(set(blockers)),
            "projected_total_kills": None,
            "lines": [
                {
                    "line": float(line),
                    "under_probability": None,
                    "over_probability": None,
                    "under_synthetic_fair_odds": None,
                    "over_synthetic_fair_odds": None,
                    "classification": "WITHHELD",
                }
                for line in lines
            ],
        }
    row = {
        "league": league,
        "checkpoint": int(checkpoint),
        "current_kills": float(current_kills),
        "total_dragons_now": float(total_dragons_now),
        "total_barons_now": float(total_barons_now),
        "total_inhibitors_now": float(total_inhibitors_now),
    }
    predicted_remaining = float(_predict_ridge(model, [row])[0])
    projected_total = float(current_kills) + predicted_remaining
    priced = []
    for line in lines:
        line_blockers = _line_calibration_blockers(
            artifact,
            checkpoint=int(checkpoint),
            line=float(line),
        )
        if line_blockers:
            priced.append(
                {
                    "line": float(line),
                    "under_probability": None,
                    "over_probability": None,
                    "under_synthetic_fair_odds": None,
                    "over_synthetic_fair_odds": None,
                    "classification": "WITHHELD",
                    "blockers": line_blockers,
                }
            )
            continue
        under = _under_probability(
            residuals,
            predicted_remaining,
            float(current_kills),
            float(line),
        )
        priced.append(
            {
                "line": float(line),
                "under_probability": under,
                "over_probability": 1.0 - under,
                "under_synthetic_fair_odds": _fair_odds(under),
                "over_synthetic_fair_odds": _fair_odds(1.0 - under),
                "classification": "SYNTHETIC_RESEARCH_ONLY",
                "blockers": [],
            }
        )
    line_blockers = sorted(
        {
            blocker
            for row in priced
            for blocker in row.get("blockers") or []
        }
    )
    any_priced = any(
        row["classification"] == "SYNTHETIC_RESEARCH_ONLY" for row in priced
    )
    return {
        "status": "synthetic_research_only" if any_priced else "unavailable",
        "blockers": sorted(
            {
                "external_market_benchmark_missing",
                "not_live_serving",
                "synthetic_prices_are_not_market_prices",
                *line_blockers,
            }
        ),
        "league": league,
        "checkpoint": int(checkpoint),
        "projected_total_kills": projected_total,
        "lines": priced,
    }


def _bootstrap_ci(
    rows: Sequence[Mapping[str, Any]],
    values: Sequence[float],
    metric,
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    by_series: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_series[str(row["series_id"])].append(index)
    series = sorted(by_series)
    if len(series) < 2:
        return {"status": "unavailable", "series": len(series), "lower": None, "upper": None}
    rng = np.random.default_rng(seed)
    estimates = []
    array = np.asarray(values, dtype=float)
    for _ in range(replicates):
        chosen = rng.choice(series, size=len(series), replace=True)
        indices = [index for series_id in chosen for index in by_series[str(series_id)]]
        estimates.append(float(metric(array[indices])))
    lower, upper = np.quantile(np.asarray(estimates), [0.025, 0.975])
    return {
        "status": "available",
        "series": len(series),
        "replicates": replicates,
        "seed": seed,
        "lower": float(lower),
        "upper": float(upper),
    }


def _model_summary(
    family: str,
    train: Sequence[Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    model, prediction = _predict_family(family, train, selection)
    return {
        "family": family,
        "selection": _metrics(
            [float(row["remaining_kills"]) for row in selection], prediction
        ),
        "model": model,
    }


def _reference_state(rows: Sequence[Mapping[str, Any]], checkpoint: int) -> dict[str, Any]:
    candidates = [row for row in rows if int(row["checkpoint"]) == checkpoint]
    if not candidates:
        raise SyntheticPriceError(f"no reference state for checkpoint {checkpoint}")
    def median_field(name: str) -> float:
        return float(np.median([float(row[name]) for row in candidates]))
    return {
        "league": "pooled",
        "checkpoint": checkpoint,
        "current_kills": median_field("current_kills"),
        "total_dragons_now": median_field("total_dragons_now"),
        "total_barons_now": median_field("total_barons_now"),
        "total_inhibitors_now": median_field("total_inhibitors_now"),
        "source": "train_plus_selection_medians_only",
        "n": len(candidates),
    }


def _read_current_model(path: Path, *, as_of: datetime | None = None) -> dict[str, Any]:
    if not path.is_file():
        return {
            "status": "unavailable",
            "path": str(path),
            "blocker": "current_model_artifact_missing",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {
            "status": "unavailable",
            "path": str(path),
            "blocker": f"current_model_artifact_unreadable:{type(exc).__name__}",
        }
    meta = payload.get("meta") or {}
    feature_selection = payload.get("feature_selection") or {}
    windows = payload.get("windows") or {}
    supported_windows = [
        f"{minute}:{league}"
        for minute, by_league in windows.items()
        for league, report in by_league.items()
        if report.get("status") == "supported"
    ]
    cutoffs = meta.get("data_cutoff_by_league") or {}
    cutoff_times = []
    for value in cutoffs.values():
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            cutoff_times.append(parsed.astimezone(timezone.utc))
        except ValueError:
            continue
    latest_cutoff = max(cutoff_times) if cutoff_times else None
    reference_time = as_of or datetime.now(timezone.utc)
    age_days = (
        (reference_time - latest_cutoff).total_seconds() / 86400.0
        if latest_cutoff is not None
        else None
    )
    return {
        "status": "observed_development_artifact",
        "path": str(path),
        "sha256": _file_hash(path),
        "schema_version": payload.get("schema_version"),
        "built_at": meta.get("built_at"),
        "games": meta.get("games"),
        "series": meta.get("series"),
        "source": meta.get("source"),
        "data_cutoff_by_league": cutoffs,
        "latest_data_cutoff": latest_cutoff.isoformat() if latest_cutoff else None,
        "freshness_age_days_at_build": round(age_days, 3) if age_days is not None else None,
        "freshness_status_at_build": (
            "fresh" if age_days is not None and age_days <= 14 else "stale"
            if age_days is not None
            else "unavailable"
        ),
        "selected_families": feature_selection.get("selected_families"),
        "feature_objectives": feature_selection.get("objectives"),
        "supported_windows": supported_windows,
        "runtime_freshness_limit_days": 14,
        "authority": "development_evidence_only; independent_grid_scope",
    }


def _read_catalog(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "unavailable", "path": str(path), "blocker": "grid_catalog_missing"}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "unavailable", "path": str(path), "blocker": f"grid_catalog_unreadable:{type(exc).__name__}"}
    return {
        "status": "observed_catalog",
        "path": str(path),
        "sha256": _file_hash(path),
        "catalog_sha256": payload.get("catalog_sha256"),
        "generated_at": payload.get("generated_at"),
        "capabilities": payload.get("capabilities"),
        "odds_capability": next(
            (
                capability
                for capability in payload.get("capabilities") or []
                if capability.get("capability") == "bookmaker_or_market_odds"
            ),
            None,
        ),
    }


def build_artifact(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    current_model_path: Path = DEFAULT_CURRENT_MODEL,
    catalog_path: Path = DEFAULT_CATALOG,
    lines: Sequence[float] = DEFAULT_LINES,
    built_at: str | None = None,
) -> dict[str, Any]:
    manifest, normalized_rows = load_cohort(manifest_path)
    effective_built_at = str(built_at or manifest.get("generated_at") or _now())
    try:
        built_at_time = datetime.fromisoformat(effective_built_at.replace("Z", "+00:00"))
        if built_at_time.tzinfo is None:
            built_at_time = built_at_time.replace(tzinfo=timezone.utc)
        built_at_time = built_at_time.astimezone(timezone.utc)
    except ValueError as exc:
        raise SyntheticPriceError("built_at must be an ISO-8601 timestamp") from exc
    rows = _rows_for_total_kills(normalized_rows)
    if not rows:
        raise SyntheticPriceError("frozen cohort has no verified total-kills checkpoint rows")
    if any(float(line) <= 0 or float(line).is_integer() for line in lines):
        raise SyntheticPriceError("synthetic prices require positive half-kill lines")
    split = chronological_series_split(rows)
    train, selection, calibration, test = [
        split[name] for name in ("train", "selection", "calibration", "test")
    ]
    if len(calibration) < MIN_CALIBRATION_ROWS or len(test) < MIN_TEST_ROWS:
        raise SyntheticPriceError("calibration and test coverage thresholds are not met")
    current_model_summary = _read_current_model(current_model_path, as_of=built_at_time)
    catalog_summary = _read_catalog(catalog_path)

    candidates = [
        _model_summary("prior", train, selection),
        _model_summary("kills", train, selection),
        _model_summary("objective_state", train, selection),
    ]
    prior_rmse = float(candidates[0]["selection"]["rmse"])
    eligible = [
        candidate
        for candidate in candidates[1:]
        if float(candidate["selection"]["rmse"]) < prior_rmse * (1.0 - MIN_SELECTION_RELATIVE_IMPROVEMENT)
    ]
    selected = min(eligible, key=lambda candidate: float(candidate["selection"]["rmse"])) if eligible else candidates[0]
    selected_family = str(selected["family"])

    final_model, calibration_prediction = _predict_family(selected_family, train + selection, calibration)
    _, test_prediction = _predict_family(selected_family, train + selection, test)
    _, prior_test_prediction = _predict_family("prior", train + selection, test)
    calibration_actual = [float(row["remaining_kills"]) for row in calibration]
    test_actual = [float(row["remaining_kills"]) for row in test]
    residuals = [actual - float(prediction) for actual, prediction in zip(calibration_actual, calibration_prediction)]
    selected_test_metrics = _metrics(test_actual, test_prediction)
    prior_test_metrics = _metrics(test_actual, prior_test_prediction)
    cdf = _cdf_calibration(residuals, test_actual, test_prediction)
    rmse_ci = _bootstrap_ci(
        test,
        [float(value) for value in test_prediction - np.asarray(test_actual)],
        lambda errors: float(np.sqrt(np.mean(errors**2))),
    )

    line_metrics: dict[str, dict[str, Any]] = {}
    reference_prices: dict[str, dict[str, Any]] = {}
    for checkpoint in CHECKPOINTS:
        checkpoint_test_indices = [
            index for index, row in enumerate(test) if int(row["checkpoint"]) == checkpoint
        ]
        checkpoint_calibration_residuals = [
            residual
            for row, residual in zip(calibration, residuals)
            if int(row["checkpoint"]) == checkpoint
        ]
        # If a checkpoint cell is sparse, use the pooled calibration residuals
        # for the same exact checkpoint. This is explicit and never a future or
        # post-checkpoint fallback.
        if len(checkpoint_calibration_residuals) < MIN_CALIBRATION_ROWS:
            checkpoint_calibration_residuals = [
                residual
                for row, residual in zip(calibration, residuals)
                if int(row["checkpoint"]) == checkpoint
            ]
        if len(checkpoint_calibration_residuals) < MIN_CALIBRATION_ROWS:
            checkpoint_calibration_residuals = residuals
            calibration_scope = "pooled_all_checkpoints_fallback"
        else:
            calibration_scope = "checkpoint_pooled"
        prices = []
        for line in lines:
            under = [
                _under_probability(
                    checkpoint_calibration_residuals,
                    float(test_prediction[index]),
                    float(test[index]["current_kills"]),
                    float(line),
                )
                for index in checkpoint_test_indices
            ]
            actual_under = [
                int(float(test[index]["total_kills"]) < float(line))
                for index in checkpoint_test_indices
            ]
            metric = _classification_metrics(actual_under, under)
            brier_ci = _bootstrap_ci(
                [test[index] for index in checkpoint_test_indices],
                [
                    (float(probability) - float(actual)) ** 2
                    for probability, actual in zip(under, actual_under)
                ],
                lambda values: float(np.mean(values)),
            )
            key = str(line)
            line_metrics.setdefault(key, {})[str(checkpoint)] = {
                **metric,
                "calibration_scope": calibration_scope,
                "calibration_n": len(checkpoint_calibration_residuals),
                "brier_ci_95": brier_ci,
            }

        reference = _reference_state(train + selection, checkpoint)
        reference_model_prediction = (
            float(_prior_predictions(train + selection, [reference])[0])
            if selected_family == "prior"
            else float(_predict_ridge(final_model, [reference])[0])
        )
        reference_prices[str(checkpoint)] = {
            "state": reference,
            "projected_total_kills": reference["current_kills"] + reference_model_prediction,
            "calibration_scope": calibration_scope,
            "calibration_n": len(checkpoint_calibration_residuals),
            "lines": [
                {
                    "line": float(line),
                    "under_probability": _under_probability(
                        checkpoint_calibration_residuals,
                        reference_model_prediction,
                        reference["current_kills"],
                        float(line),
                    ),
                    "over_probability": 1.0
                    - _under_probability(
                        checkpoint_calibration_residuals,
                        reference_model_prediction,
                        reference["current_kills"],
                        float(line),
                    ),
                    "under_synthetic_fair_odds": _fair_odds(
                        _under_probability(
                            checkpoint_calibration_residuals,
                            reference_model_prediction,
                            reference["current_kills"],
                            float(line),
                        )
                    ),
                    "over_synthetic_fair_odds": _fair_odds(
                        1.0
                        - _under_probability(
                            checkpoint_calibration_residuals,
                            reference_model_prediction,
                            reference["current_kills"],
                            float(line),
                        )
                    ),
                    "price_type": "synthetic_no_vig_research_benchmark",
                }
                for line in lines
            ],
        }

    # The selection/test decision is made on the untouched test split. A
    # synthetic price may be useful for research only when both point accuracy
    # and residual calibration survive that split.
    authority_blockers = [
        "external_market_benchmark_missing",
        "synthetic_prices_are_not_market_prices",
        "live_serving_not_authorized",
        "prospective_live_latency_not_evaluated",
    ]
    if selected_test_metrics["rmse"] > prior_test_metrics["rmse"]:
        authority_blockers.append("heldout_rmse_does_not_beat_prior")
    if cdf.get("status") != "passed":
        authority_blockers.append("heldout_residual_calibration_failed")
    max_line_ece = max(
        float(metric["ece"])
        for by_checkpoint in line_metrics.values()
        for metric in by_checkpoint.values()
    )
    if max_line_ece > MAX_ECE:
        authority_blockers.append("heldout_line_calibration_ece_failed")

    line_summary = {
        line: {
            "max_ece": max(float(by_checkpoint[str(checkpoint)]["ece"]) for checkpoint in CHECKPOINTS),
            "max_rate_error": max(
                abs(
                    float(by_checkpoint[str(checkpoint)]["predicted_rate"])
                    - float(by_checkpoint[str(checkpoint)]["observed_rate"])
                )
                for checkpoint in CHECKPOINTS
            ),
            "status": (
                "passes_research_calibration"
                if max(float(by_checkpoint[str(checkpoint)]["ece"]) for checkpoint in CHECKPOINTS) <= MAX_ECE
                and max(
                    abs(
                        float(by_checkpoint[str(checkpoint)]["predicted_rate"])
                        - float(by_checkpoint[str(checkpoint)]["observed_rate"])
                    )
                    for checkpoint in CHECKPOINTS
                ) <= MAX_ECE
                else "unavailable"
            ),
        }
        for line, by_checkpoint in line_metrics.items()
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": built_at_time.isoformat().replace("+00:00", "Z"),
        "scope": {
            "private_personal_research_only": True,
            "purpose": "synthetic over_under total_kills benchmark",
            "bookmaker_prices_collected": False,
            "models_trained": True,
            "public_or_live_serving": False,
            "historical_replay_authority": True,
            "prospective_live_latency_authority": False,
        },
        "inputs": {
            "cohort_manifest": {
                "path": str(manifest_path),
                "sha256": manifest["manifest_sha256"],
                "verified_maps_total": int((manifest.get("coverage") or {}).get("verified_maps_total") or 0),
                "scope": manifest.get("scope"),
                "derivation": manifest.get("derivation"),
            },
            "grid_catalog": catalog_summary,
            "current_model": current_model_summary,
        },
        "protocol": {
            "split": "chronological_whole_provider_series_train_selection_calibration_test",
            "fractions": list(SPLIT_FRACTIONS),
            "target": "remaining_kills_after_exact_at_or_before_checkpoint",
            "checkpoints": list(CHECKPOINTS),
            "candidate_families": ["prior", "kills", "objective_state"],
            "objective_state_fields": [
                "current_kills",
                "total_dragons_now",
                "total_barons_now",
                "total_inhibitors_now",
            ],
            "gold_difference": "unavailable_in_frozen_GRID_cohort; not imputed",
            "price_rule": "half_kill_under_is_total_kills_less_than_line; no-vig fair odds are reciprocal probabilities",
            "calibration": "calibration-split empirical residual CDF; checkpoint-pooled with explicit sparse-cell fallback",
            "uncertainty": {
                "method": "whole-series bootstrap",
                "replicates": BOOTSTRAP_REPLICATES,
                "seed": BOOTSTRAP_SEED,
            },
            "no_post_checkpoint_leakage": True,
        },
        "coverage": {
            name: {
                "rows": len(part),
                "series": len({str(row["series_id"]) for row in part}),
                "leagues": {
                    league: sum(str(row["league"]) == league for row in part)
                    for league in LEAGUES
                    if any(str(row["league"]) == league for row in part)
                },
                "start": min(str(row["date"]) for row in part) if part else None,
                "end": max(str(row["date"]) for row in part) if part else None,
            }
            for name, part in split.items()
        },
        "feature_selection": {
            "selected_family": selected_family,
            "candidates": [
                {
                    "family": candidate["family"],
                    "selection": candidate["selection"],
                    "relative_rmse_improvement_vs_prior": (
                        (prior_rmse - float(candidate["selection"]["rmse"])) / prior_rmse
                    ),
                }
                for candidate in candidates
            ],
            "minimum_relative_improvement": MIN_SELECTION_RELATIVE_IMPROVEMENT,
        },
        "selected_model": final_model,
        "calibration_residuals_by_checkpoint": {
            str(checkpoint): [
                float(residual)
                for row, residual in zip(calibration, residuals)
                if int(row["checkpoint"]) == checkpoint
            ]
            for checkpoint in CHECKPOINTS
        },
        "heldout": {
            "selected": selected_test_metrics,
            "prior": prior_test_metrics,
            "selected_rmse_ci_95": rmse_ci,
            "calibration_residual_cdf": cdf,
            "line_metrics": line_metrics,
            "line_summary": line_summary,
        },
        "comparison": {
            "current_model": current_model_summary,
            "grid_synthetic_model": {
                "verified_maps": int((manifest.get("coverage") or {}).get("verified_maps_total") or 0),
                "selected_family": selected_family,
                "heldout_rmse": selected_test_metrics["rmse"],
                "heldout_prior_rmse": prior_test_metrics["rmse"],
                "heldout_rmse_delta": selected_test_metrics["rmse"] - prior_test_metrics["rmse"],
            },
            "comparable": False,
            "not_comparable_reasons": [
                "different frozen populations and date ranges",
                "current model source is maps.parquet rather than receipt-backed GRID/Riot cohort",
                "current model selected gold_difference while GRID cohort has no gold_difference field",
                "GRID synthetic evaluation uses objective_state fields that current model marked unavailable",
            ],
        },
        "synthetic_reference_prices": reference_prices,
        "authority": {
            "status": "research_synthetic_prices_only",
            "synthetic_prices_available": True,
            "betting_classification_authorized": False,
            "model_or_market_authority": "unavailable",
            "blockers": sorted(set(authority_blockers)),
            "required_before_betting_claim": [
                "freeze a current exact-patch/league cohort with verified checkpoint coverage",
                "repeat held-out evaluation after any feature or scope change",
                "predeclare and pass uncertainty/calibration thresholds at the served line range",
                "collect a timestamped, settlement-matched external bookmaker benchmark",
                "compare out-of-sample prices after vig/limits/availability are handled",
                "separately authorize prospective live latency and operational controls",
            ],
        },
    }


def write_artifact(
    manifest_path: Path = DEFAULT_MANIFEST,
    *,
    output_root: Path | None = None,
    current_model_path: Path = DEFAULT_CURRENT_MODEL,
    catalog_path: Path = DEFAULT_CATALOG,
    lines: Sequence[float] = DEFAULT_LINES,
) -> tuple[dict[str, Any], Path]:
    artifact = build_artifact(
        manifest_path,
        current_model_path=current_model_path,
        catalog_path=catalog_path,
        lines=lines,
    )
    payload = dict(artifact)
    digest = _hash(payload)
    payload["artifact_sha256"] = digest
    root = output_root or manifest_path.parent.parent / "evaluations"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"synthetic-total-kills-prices-{digest}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload, path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--current-model", type=Path, default=DEFAULT_CURRENT_MODEL)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    args = parser.parse_args(argv)
    artifact, path = write_artifact(
        args.manifest,
        output_root=args.output_root,
        current_model_path=args.current_model,
        catalog_path=args.catalog,
    )
    print(
        json.dumps(
            {
                "output": str(path),
                "artifact_sha256": artifact["artifact_sha256"],
                "selected_family": artifact["feature_selection"]["selected_family"],
                "heldout": artifact["heldout"],
                "authority": artifact["authority"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
