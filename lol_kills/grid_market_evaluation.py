"""Private, fail-closed evaluation for the frozen GRID market cohort.

This module evaluates descriptive checkpoint forecasts only.  It never serves
predictions and never computes prices, fair odds, edge, EV, or betting claims.
Every split is chronological and whole-series grouped; the cohort manifest is
treated as immutable input and all accepted rows must carry its exact IDs and
verified outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from lol_kills.grid_live_foundation import write_immutable_receipt


SCHEMA_VERSION = "scryglass.grid-market-evaluation.v1"
CHECKPOINTS = (10, 15, 20, 25)
LEAGUES = ("LCS", "LEC", "CBLOL", "LCK")
SPLIT_FRACTIONS = (0.50, 0.15, 0.15, 0.20)
MIN_CALIBRATION_EXAMPLES = 40
MIN_TEST_EXAMPLES = 40
MAX_ECE = 0.10
MAX_CDF_ERROR = 0.10
MIN_SELECTION_RELATIVE_IMPROVEMENT = 0.005

# First tower remains a researched first-event target.  Total towers are not
# represented anywhere in this registry or evaluation.
FIRST_TARGETS = (
    "first_blood",
    "first_tower",
    "first_inhibitor",
    "first_dragon",
    "first_baron",
)
TOTAL_TARGETS = (
    "total_kills",
    "total_dragons",
    "total_barons",
    "total_inhibitor_destructions",
)
TARGETS = FIRST_TARGETS + TOTAL_TARGETS


class EvaluationError(RuntimeError):
    """Raised when the frozen cohort cannot support a truthful evaluation."""


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_time(value: Any) -> datetime:
    text = str(value or "").replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _safe_probability(value: Any) -> float:
    return float(min(max(float(value), 1e-6), 1.0 - 1e-6))


def _log_loss(y: Sequence[float], p: Sequence[float]) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return float(np.mean(-(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp))))


def _ece(y: Sequence[float], p: Sequence[float], bins: int = 10) -> float:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (pp >= edges[index]) & (
            pp <= edges[index + 1]
            if index == bins - 1
            else pp < edges[index + 1]
        )
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(yy[mask].mean()) - float(pp[mask].mean()))
    return float(total)


def _classification_metrics(y: Sequence[int], p: Sequence[float]) -> dict[str, Any]:
    yy = np.asarray(y, dtype=float)
    pp = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return {
        "n": int(len(yy)),
        "brier": float(np.mean((pp - yy) ** 2)),
        "log_loss": _log_loss(yy, pp),
        "ece": _ece(yy, pp),
        "positive_rate": float(yy.mean()) if len(yy) else None,
    }


def _regression_metrics(y: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    yy = np.asarray(y, dtype=float)
    pp = np.asarray(prediction, dtype=float)
    errors = pp - yy
    return {
        "n": int(len(yy)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "mae": float(np.mean(np.abs(errors))),
    }


def _empirical_crps(y: Sequence[float], prediction: Sequence[float], residuals: Sequence[float]) -> float:
    if not residuals:
        return float("nan")
    residual = np.asarray(residuals, dtype=float)
    scores = []
    for actual, point in zip(y, prediction):
        draws = point + residual
        scores.append(float(np.mean(np.abs(draws - actual)) - 0.5 * np.mean(np.abs(draws[:, None] - draws[None, :]))))
    return float(np.mean(scores))


def _cdf_calibration(residuals: Sequence[float], y: Sequence[float], prediction: Sequence[float]) -> dict[str, Any]:
    if not residuals or not y:
        return {"status": "unavailable", "max_absolute_error": None, "points": []}
    ordered = sorted(float(value) for value in residuals)
    points = []
    for nominal in (0.10, 0.25, 0.50, 0.75, 0.90):
        index = max(0, min(len(ordered) - 1, math.ceil(nominal * len(ordered)) - 1))
        threshold = ordered[index]
        observed = sum(
            float(actual) - float(point) <= threshold
            for actual, point in zip(y, prediction)
        ) / len(y)
        points.append(
            {
                "nominal": nominal,
                "observed": observed,
                "absolute_error": abs(observed - nominal),
            }
        )
    maximum = max(point["absolute_error"] for point in points)
    return {
        "status": "passed" if maximum <= MAX_CDF_ERROR else "failed",
        "max_absolute_error": maximum,
        "threshold": MAX_CDF_ERROR,
        "points": points,
    }


def _normalize_record(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    if record.get("status") != "verified":
        return []
    identity = record.get("identity") or {}
    chronology = record.get("chronology") or {}
    series_id = str(identity.get("provider_series_id") or "")
    game_id = str(identity.get("provider_game_id") or "")
    league = str(record.get("league") or "").upper()
    date = chronology.get("series_start_time_scheduled")
    outcomes = record.get("outcomes") or {}
    total_kills = outcomes.get("total_kills")
    if league not in LEAGUES or not series_id or not game_id or not date or not _finite(total_kills):
        return []
    rows = []
    for checkpoint in record.get("checkpoints") or []:
        minute = checkpoint.get("minute")
        values = checkpoint.get("values") or {}
        if minute not in CHECKPOINTS:
            continue
        if not _finite(values.get("current_kills")):
            continue
        row = {
            "series_id": series_id,
            "game_id": game_id,
            "date": str(date),
            "league": league,
            "patch": str(record.get("patch") or ""),
            "checkpoint": int(minute),
            "current_kills": float(values["current_kills"]),
            "total_dragons_now": float(values.get("total_dragons") or 0),
            "total_barons_now": float(values.get("total_barons") or 0),
            "total_inhibitors_now": float(values.get("total_inhibitor_destructions") or 0),
            "first_blood_now": values.get("first_blood"),
            "first_tower_now": values.get("first_tower"),
            "first_inhibitor_now": values.get("first_inhibitor"),
            "first_dragon_now": values.get("first_dragon"),
            "first_baron_now": values.get("first_baron"),
            "total_kills": float(total_kills),
            "total_dragons": record.get("labels", {}).get("total_dragons"),
            "total_barons": record.get("labels", {}).get("total_barons"),
            "total_inhibitor_destructions": record.get("labels", {}).get("total_inhibitor_destructions"),
        }
        for target in FIRST_TARGETS:
            value = (record.get("labels") or {}).get(target)
            if value not in (100, 200):
                row[target] = None
            else:
                row[target] = int(value == 100)
        rows.append(row)
    return rows


def load_cohort(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.is_file():
        raise EvaluationError(f"cohort manifest is missing: {manifest_path}")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    expected = manifest.get("manifest_sha256")
    if not expected:
        raise EvaluationError("cohort manifest hash is missing")
    replay = dict(manifest)
    replay.pop("manifest_sha256", None)
    if _hash(replay) != expected:
        raise EvaluationError("cohort manifest hash does not replay")
    if (manifest.get("scope") or {}).get("models_trained") is True:
        raise EvaluationError("cohort manifest is not a data-only frozen input")
    rows = [row for record in manifest.get("verified_games") or [] for row in _normalize_record(record)]
    if not rows:
        raise EvaluationError("cohort has no evaluation rows with verified outcomes")
    return manifest, rows


def _receipt_path(value: Any, manifest_path: Path) -> Path:
    path = Path(str(value or ""))
    if path.is_absolute():
        return path
    candidate = (Path(__file__).resolve().parents[1] / path).resolve()
    if candidate.is_file():
        return candidate
    return (manifest_path.parent.parent.parent.parent / path).resolve()


def _verified_summary_outcome(record: Mapping[str, Any], manifest_path: Path) -> dict[str, Any]:
    identity = record.get("identity") or {}
    platform = str(identity.get("riot_platform_id") or "")
    game_id = str(identity.get("riot_game_id") or "")
    matches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for file_id, reference in (record.get("source_receipts") or {}).items():
        if not str(file_id).startswith("state-summary-riot-game-"):
            continue
        receipt_file = _receipt_path((reference or {}).get("receipt_path"), manifest_path)
        if not receipt_file.is_file():
            continue
        receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
        raw_path = _receipt_path(receipt.get("raw_path"), manifest_path)
        if not raw_path.is_file() or _file_sha256(raw_path) != receipt.get("raw_sha256"):
            continue
        summary = json.loads(raw_path.read_text(encoding="utf-8"))
        if str(summary.get("platformId") or "") == platform and str(summary.get("gameId") or "") == game_id:
            matches.append((receipt, summary))
    if len(matches) != 1:
        raise EvaluationError("final summary receipt is not uniquely replayable")
    receipt, summary = matches[0]
    if summary.get("endOfGameResult") not in (None, "GameComplete"):
        raise EvaluationError("final summary is not complete")
    if not isinstance(summary.get("gameEndTimestamp"), int) or not isinstance(summary.get("gameDuration"), int):
        raise EvaluationError("final summary game-end fields are incomplete")
    teams = summary.get("teams") or []
    if {team.get("teamId") for team in teams if isinstance(team, Mapping)} != {100, 200}:
        raise EvaluationError("final summary does not contain exact Riot teams")
    participants = summary.get("participants") or []
    if len(participants) != 10:
        raise EvaluationError("final summary does not contain exactly ten participants")
    kills = [participant.get("kills") for participant in participants]
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in kills):
        raise EvaluationError("final summary participant kills are invalid")
    return {
        "total_kills": int(sum(kills)),
        "game_duration_seconds": int(summary["gameDuration"]),
        "final_summary_receipt_sha256": receipt.get("receipt_sha256"),
        "final_summary_raw_sha256": receipt.get("raw_sha256"),
    }


def freeze_evaluation_manifest(manifest_path: Path, output_root: Path | None = None) -> tuple[dict[str, Any], Path]:
    """Add separately replayed final total-kill outcomes to an immutable copy."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("manifest_sha256")
    replay = dict(manifest)
    replay.pop("manifest_sha256", None)
    if not expected or _hash(replay) != expected:
        raise EvaluationError("base cohort manifest hash does not replay")
    derived = json.loads(json.dumps(manifest))
    derived_games = []
    for record in derived.get("verified_games") or []:
        outcome = dict(record.get("outcomes") or {})
        if not _finite(outcome.get("total_kills")):
            outcome.update(_verified_summary_outcome(record, manifest_path))
        derived_record = dict(record)
        derived_record["outcomes"] = outcome
        derived_games.append(derived_record)
    derived["verified_games"] = derived_games
    derived["derivation"] = {
        "kind": "evaluation_input_freeze",
        "base_manifest_sha256": manifest.get("manifest_sha256"),
        "final_outcomes_replayed_from_immutable_summary_receipts": True,
        "derived_at": _now(),
    }
    derived.pop("manifest_sha256", None)
    derived["manifest_sha256"] = _hash(derived)
    root = output_root or manifest_path.parent
    path = root / f"evaluation-cohort-{derived['manifest_sha256']}.json"
    write_immutable_receipt(path, derived)
    return derived, path


def chronological_series_split(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_series: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_series[str(row["series_id"])].append(dict(row))
    ordered = sorted(
        by_series.items(),
        key=lambda item: (max(_parse_time(row["date"]) for row in item[1]), item[0]),
    )
    if len(ordered) < 20:
        raise EvaluationError("at least 20 complete provider series are required")
    n = len(ordered)
    cuts = [
        int(n * SPLIT_FRACTIONS[0]),
        int(n * sum(SPLIT_FRACTIONS[:2])),
        int(n * sum(SPLIT_FRACTIONS[:3])),
    ]
    parts = [ordered[: cuts[0]], ordered[cuts[0] : cuts[1]], ordered[cuts[1] : cuts[2]], ordered[cuts[2] :]]
    names = ("train", "selection", "calibration", "test")
    return {name: [row for _, series_rows in part for row in series_rows] for name, part in zip(names, parts)}


def _feature_names(target: str, family: str) -> list[str]:
    # Reference categories (10 minutes and CBLOL) keep the design full-rank;
    # the omitted categories are represented by the intercept.
    base = [
        "intercept",
        *[f"checkpoint:{m}" for m in CHECKPOINTS if m != 10],
        *[f"league:{league}" for league in LEAGUES if league != "CBLOL"],
    ]
    if family in {"state", "target_state"}:
        base += ["current_kills", "total_dragons_now", "total_barons_now", "total_inhibitors_now"]
    if family == "target_state":
        if target in FIRST_TARGETS:
            base += ["target_known", "target_blue_now"]
        else:
            base += ["target_count_now"]
    return base


def _features(rows: Sequence[Mapping[str, Any]], target: str, family: str) -> np.ndarray:
    names = _feature_names(target, family)
    matrix = []
    for row in rows:
        values = {name: 0.0 for name in names}
        values["intercept"] = 1.0
        checkpoint_name = f"checkpoint:{row['checkpoint']}"
        league_name = f"league:{row['league']}"
        if checkpoint_name in values:
            values[checkpoint_name] = 1.0
        if league_name in values:
            values[league_name] = 1.0
        if family in {"state", "target_state"}:
            values.update(
                {
                    "current_kills": float(row["current_kills"]),
                    "total_dragons_now": float(row["total_dragons_now"]),
                    "total_barons_now": float(row["total_barons_now"]),
                    "total_inhibitors_now": float(row["total_inhibitors_now"]),
                }
            )
        if family == "target_state":
            if target in FIRST_TARGETS:
                current = row[f"{target}_now"]
                values["target_known"] = float(current in (100, 200))
                values["target_blue_now"] = float(current == 100)
            else:
                current_key = {
                    "total_kills": "current_kills",
                    "total_dragons": "total_dragons_now",
                    "total_barons": "total_barons_now",
                    "total_inhibitor_destructions": "total_inhibitors_now",
                }[target]
                values["target_count_now"] = float(row[current_key])
        matrix.append([values[name] for name in names])
    return np.asarray(matrix, dtype=float)


def _standardize(train_x: np.ndarray, *other: np.ndarray) -> tuple[np.ndarray, ...]:
    center = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    center[0] = 0.0
    scale[0] = 1.0
    scale[scale < 1e-9] = 1.0
    return tuple((x - center) / scale for x in (train_x, *other))


def _ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float = 10.0) -> np.ndarray:
    regularizer = np.eye(x.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        matrix = np.einsum("ni,nj->ij", x, x) + regularizer + np.eye(x.shape[1]) * 1e-8
        matrix[0, 0] -= 1e-8
        target = np.einsum("ni,n->i", x, y)
    try:
        beta = np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(matrix, target, rcond=None)[0]
    if not np.isfinite(beta).all():
        raise EvaluationError("non-finite ridge fit")
    return beta


def _ridge_predictions(train: Sequence[Mapping[str, Any]], fit: Sequence[Mapping[str, Any]], target: str, family: str, parts: Sequence[Sequence[Mapping[str, Any]]]) -> list[np.ndarray]:
    train_x = _features(train, target, family)
    fit_x = _features(fit, target, family)
    all_x = [_features(part, target, family) for part in parts]
    train_x, fit_x, *all_x = _standardize(train_x, fit_x, *all_x)
    beta = _ridge_fit(fit_x, np.asarray([float(row[target]) for row in fit], dtype=float))
    return [np.maximum(0.0, np.einsum("ij,j->i", x, beta)) for x in all_x]


def _logistic_fit(x: np.ndarray, y: np.ndarray, penalty: float = 5.0) -> np.ndarray:
    # The explicit intercept column is retained so the same standardized
    # design is used for every target.  sklearn's bounded L2 solver is more
    # stable than hand-rolled Newton updates for one-hot checkpoint/league
    # columns that are intentionally rank deficient.
    model = LogisticRegression(
        C=1.0 / max(float(penalty), 1e-9),
        penalty="l2",
        solver="liblinear",
        fit_intercept=False,
        max_iter=500,
        random_state=0,
    )
    model.fit(x, np.asarray(y, dtype=int))
    beta = np.asarray(model.coef_[0], dtype=float)
    if not np.isfinite(beta).all():
        raise EvaluationError("non-finite logistic fit")
    return beta


def _logistic_predictions(fit: Sequence[Mapping[str, Any]], target: str, family: str, parts: Sequence[Sequence[Mapping[str, Any]]]) -> list[np.ndarray]:
    fit_x = _features(fit, target, family)
    all_x = [_features(part, target, family) for part in parts]
    fit_x, *all_x = _standardize(fit_x, *all_x)
    y = np.asarray([int(row[target]) for row in fit], dtype=float)
    beta = _logistic_fit(fit_x, y)
    predictions = []
    for x in all_x:
        with np.errstate(over="ignore", invalid="ignore"):
            logits = np.einsum("ij,j->i", x, beta)
        if not np.isfinite(logits).all():
            raise EvaluationError("non-finite logistic prediction")
        predictions.append(1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0))))
    return predictions


def _prior_predictions(train: Sequence[Mapping[str, Any]], parts: Sequence[Sequence[Mapping[str, Any]]], target: str) -> list[np.ndarray]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    global_values = []
    for row in train:
        value = row.get(target)
        if value is None:
            continue
        grouped[(str(row["league"]), int(row["checkpoint"]))].append(float(value))
        global_values.append(float(value))
    fallback = float(np.mean(global_values)) if global_values else 0.0
    return [
        np.asarray(
            [float(np.mean(grouped.get((str(row["league"]), int(row["checkpoint"])), [fallback]))) for row in part],
            dtype=float,
        )
        for part in parts
    ]


def _calibrate_binary(cal_y: Sequence[int], cal_p: Sequence[float], other: Sequence[np.ndarray]) -> list[np.ndarray] | None:
    if len(cal_y) < MIN_CALIBRATION_EXAMPLES or len(set(cal_y)) < 2:
        return None
    calibrator = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
    calibrator.fit(np.asarray(cal_p, dtype=float), np.asarray(cal_y, dtype=float))
    return [np.asarray(calibrator.predict(np.asarray(values, dtype=float)), dtype=float) for values in other]


def _split_manifest(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    series = sorted({str(row["series_id"]) for row in rows})
    dates = [_parse_time(row["date"]) for row in rows]
    return {
        "rows": len(rows),
        "series": len(series),
        "start": min(dates).isoformat() if dates else None,
        "end": max(dates).isoformat() if dates else None,
        "series_ids_sha256": hashlib.sha256("\n".join(series).encode()).hexdigest(),
    }


def _target_rows(rows: Sequence[Mapping[str, Any]], target: str) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        current = row.get(target)
        if target in FIRST_TARGETS:
            if current is None:
                continue
            result.append({**row, target: int(current)})
        else:
            final = row.get(target)
            if not _finite(final):
                continue
            if target == "total_kills":
                now = float(row["current_kills"])
            elif target == "total_dragons":
                now = float(row["total_dragons_now"])
            elif target == "total_barons":
                now = float(row["total_barons_now"])
            else:
                now = float(row["total_inhibitors_now"])
            result.append({**row, target: max(0.0, float(final) - now)})
    return result


def _evaluate_target(target: str, rows: Sequence[Mapping[str, Any]], split: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    target_rows = {name: _target_rows(part, target) for name, part in split.items()}
    train = target_rows["train"]
    selection = target_rows["selection"]
    calibration = target_rows["calibration"]
    test = target_rows["test"]
    report: dict[str, Any] = {
        "target": target,
        "kind": "classification" if target in FIRST_TARGETS else "remaining_count_regression",
        "coverage": {name: _split_manifest(part) for name, part in target_rows.items()},
        "status": "unavailable",
        "candidates": {},
        "selected_family": None,
        "authority_blockers": [],
    }
    if min(len(calibration), len(test)) < MIN_CALIBRATION_EXAMPLES:
        report["authority_blockers"].append("insufficient_calibration_or_test_rows")
        return report
    families = ("prior", "state", "target_state")
    if target in FIRST_TARGETS:
        if len(set(int(row[target]) for row in train)) < 2 or len(set(int(row[target]) for row in test)) < 2:
            report["authority_blockers"].append("single_class_train_or_test")
            return report
        train_fit = train
        candidate_predictions: dict[str, dict[str, np.ndarray]] = {}
        for family in families:
            if family == "prior":
                selection_raw = _prior_predictions(train_fit, [selection], target)[0]
                final_raw = _prior_predictions(train_fit + selection, [calibration, test], target)
            else:
                selection_raw = _logistic_predictions(train_fit, target, family, [selection])[0]
                final_raw = _logistic_predictions(train_fit + selection, target, family, [calibration, test])
            candidate_predictions[family] = {
                "selection": selection_raw,
                "calibration": final_raw[0],
                "test": final_raw[1],
            }
            cal_y = [int(row[target]) for row in calibration]
            calibrated = _calibrate_binary(cal_y, final_raw[0], [final_raw[1]])
            selection_metrics = _classification_metrics([int(row[target]) for row in selection], selection_raw)
            test_metrics = _classification_metrics([int(row[target]) for row in test], calibrated[0] if calibrated else raw[2])
            report["candidates"][family] = {
                "selection": selection_metrics,
                "test_calibrated": test_metrics if calibrated else None,
                "calibration_status": "available" if calibrated else "unavailable",
            }
        prior = report["candidates"]["prior"]["selection"]["brier"]
        eligible = [
            family
            for family in families[1:]
            if report["candidates"][family]["calibration_status"] == "available"
            and report["candidates"][family]["selection"]["brier"] < prior * (1.0 - MIN_SELECTION_RELATIVE_IMPROVEMENT)
        ]
        selected = min(eligible, key=lambda family: report["candidates"][family]["selection"]["brier"]) if eligible else "prior"
        report["selected_family"] = selected
        selected_test = report["candidates"][selected]["test_calibrated"]
        prior_test = report["candidates"]["prior"]["test_calibrated"]
        if not selected_test or not prior_test:
            report["authority_blockers"].append("calibration_unavailable")
            return report
        improved = selected_test["brier"] < prior_test["brier"] and selected_test["log_loss"] < prior_test["log_loss"]
        if not improved:
            report["authority_blockers"].append("heldout_improvement_not_supported")
        if selected_test["ece"] > MAX_ECE:
            report["authority_blockers"].append("heldout_calibration_ece_failed")
        report["status"] = "supported" if not report["authority_blockers"] else "unavailable"
        return report

    # Continuous targets: choose by selection RMSE, then evaluate an empirical
    # residual calibration layer on the untouched test set.
    for family in families:
        if family == "prior":
            selection_pred = _prior_predictions(train, [selection], target)[0]
            calibration_pred, test_pred = _prior_predictions(train + selection, [calibration, test], target)
        else:
            selection_pred = _ridge_predictions(train, train, target, family, [selection])[0]
            calibration_pred, test_pred = _ridge_predictions(train, train + selection, target, family, [calibration, test])
        selection_metrics = _regression_metrics([row[target] for row in selection], selection_pred)
        calibration_residuals = [float(row[target]) - float(pred) for row, pred in zip(calibration, calibration_pred)]
        test_metrics = _regression_metrics([row[target] for row in test], test_pred)
        cdf = _cdf_calibration(calibration_residuals, [row[target] for row in test], test_pred)
        test_metrics["crps"] = _empirical_crps([row[target] for row in test], test_pred, calibration_residuals)
        report["candidates"][family] = {
            "selection": selection_metrics,
            "test": test_metrics,
            "residual_calibration": cdf,
        }
    prior_rmse = report["candidates"]["prior"]["selection"]["rmse"]
    eligible = [
        family
        for family in families[1:]
        if report["candidates"][family]["selection"]["rmse"] < prior_rmse * (1.0 - MIN_SELECTION_RELATIVE_IMPROVEMENT)
    ]
    selected = min(eligible, key=lambda family: report["candidates"][family]["selection"]["rmse"]) if eligible else "prior"
    report["selected_family"] = selected
    selected_test = report["candidates"][selected]["test"]
    prior_test = report["candidates"]["prior"]["test"]
    selected_cdf = report["candidates"][selected]["residual_calibration"]
    if selected_test["rmse"] > prior_test["rmse"] or selected_test["crps"] > prior_test["crps"]:
        report["authority_blockers"].append("heldout_improvement_not_supported")
    if selected_cdf.get("status") != "passed":
        report["authority_blockers"].append("heldout_residual_calibration_failed")
    report["status"] = "supported" if not report["authority_blockers"] else "unavailable"
    return report


def evaluate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest, rows = load_cohort(manifest_path)
    split = chronological_series_split(rows)
    target_reports = {
        target: _evaluate_target(target, rows, split)
        for target in TARGETS
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "cohort_manifest": {
            "path": str(manifest_path),
            "sha256": str(manifest["manifest_sha256"]),
            "verified_maps_total": int((manifest.get("coverage") or {}).get("verified_maps_total") or 0),
        },
        "protocol": {
            "split": "chronological_whole_provider_series_train_selection_calibration_test",
            "fractions": list(SPLIT_FRACTIONS),
            "features": "checkpoint, league, current kills, at-or-before objective state; no post-checkpoint or final labels",
            "selection_metric": "RMSE for remaining counts; Brier score for first-event classifications",
            "calibration": "empirical residual CDF for counts; isotonic calibration for classifications fitted on calibration split only",
            "minimum_calibration_rows": MIN_CALIBRATION_EXAMPLES,
            "minimum_test_rows": MIN_TEST_EXAMPLES,
            "total_towers": "excluded",
        },
        "coverage": {name: _split_manifest(part) for name, part in split.items()},
        "targets": target_reports,
        "authority": {
            "status": "research_evidence_only",
            "model_or_betting_authority": "unavailable",
            "probabilities_or_prices_served": False,
            "promoted_targets": [],
            "blockers": [
                "no_external_market_benchmark_collected",
                "no_live_serving_authorized",
                *sorted(
                    f"{target}:{blocker}"
                    for target, report in target_reports.items()
                    for blocker in report.get("authority_blockers") or []
                ),
            ],
        },
    }


def write_evaluation(manifest_path: Path, output_root: Path | None = None) -> tuple[dict[str, Any], Path]:
    report = evaluate_manifest(manifest_path)
    root = output_root or manifest_path.parent.parent / "evaluations"
    # The path is content addressed after excluding the path itself.
    report_hash = _hash(report)
    report["report_sha256"] = report_hash
    path = root / f"market-evaluation-{report_hash}.json"
    write_immutable_receipt(path, report)
    return report, path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    report, path = write_evaluation(args.manifest, args.output_root)
    print(
        json.dumps(
            {
                "output": str(path),
                "report_sha256": report["report_sha256"],
                "coverage": report["coverage"],
                "status_by_target": {target: value["status"] for target, value in report["targets"].items()},
                "authority": report["authority"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
