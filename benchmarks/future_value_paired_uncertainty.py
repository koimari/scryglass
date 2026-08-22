"""Build a deterministic whole-series uncertainty report for four variants.

The command consumes the already evaluated, fold-local prediction ledgers.  It
does not refit a model and it does not select a variant.  Each bootstrap draw
resamples conservative whole-series clusters, so maps from one series stay in
the same draw.  The output is research-only and binds every result to the
source, bundle, fold, and prediction-ledger hashes.

Example::

    python3 benchmarks/future_value_paired_uncertainty.py \
      --evaluation-root /private/tmp/scryglass-four-variant-runs/evaluation-v2 \
      --bundle /private/tmp/scryglass-four-variant-runs/four-variant-feature-ledger-bundle-v2.json \
      --output-dir /private/tmp/scryglass-four-variant-runs/paired-uncertainty-v1 \
      --draws 2000 --seed 461
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-paired-uncertainty:v1"
VARIANTS = (
    "current_only",
    "future_player_form",
    "scaling_curve",
    "both",
)
COMPARISONS = (
    ("future_player_form", "current_only"),
    ("scaling_curve", "current_only"),
    ("both", "current_only"),
    ("both", "future_player_form"),
)
SHA256_LENGTH = 64
LOG_EPSILON = 1e-15


class PairedUncertaintyError(ValueError):
    """The paired uncertainty artifact cannot be built safely."""


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
        raise PairedUncertaintyError("value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    if not path.is_file() or path.is_symlink():
        raise PairedUncertaintyError(f"file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise PairedUncertaintyError(f"file cannot be read: {path}") from error
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PairedUncertaintyError(f"{label} is missing or unsafe")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PairedUncertaintyError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise PairedUncertaintyError(f"{label} must be a JSON object")
    return value


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise PairedUncertaintyError(f"{label} is not a SHA-256 value")
    try:
        int(value, 16)
    except ValueError as error:
        raise PairedUncertaintyError(f"{label} is not a SHA-256 value") from error
    return value.lower()


def _verify_authority(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("research_only") is not True:
        raise PairedUncertaintyError(f"{label} is not research-only")
    if any(bool(flag) for key, flag in value.items() if key != "research_only"):
        raise PairedUncertaintyError(f"{label} grants authority")


def _finite_probability(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PairedUncertaintyError(f"{label} is not numeric") from error
    if not math.isfinite(result) or result < 0.0 or result > 1.0:
        raise PairedUncertaintyError(f"{label} is outside [0, 1]")
    return result


def _finite_target(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise PairedUncertaintyError(f"{label} is not numeric") from error
    if result not in (0.0, 1.0):
        raise PairedUncertaintyError(f"{label} is not binary")
    return result


def _ledger_hash(rows: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_bytes(_canonical_bytes(list(rows)))


def _load_prediction_ledger(
    path: Path,
    variant: str,
    *,
    expected_source: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    document = _load_json(path, f"{variant} model")
    variants = document.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != {variant}:
        raise PairedUncertaintyError(f"{variant} model variant binding changed")
    result = variants[variant]
    if not isinstance(result, Mapping):
        raise PairedUncertaintyError(f"{variant} model result is invalid")
    if result.get("status") != "development_evaluated":
        raise PairedUncertaintyError(f"{variant} model status is not evaluated")
    _verify_authority(result.get("authority"), f"{variant} authority")
    source = result.get("source")
    if not isinstance(source, Mapping):
        raise PairedUncertaintyError(f"{variant} source binding is missing")
    if expected_source is not None:
        for key in (
            "source_as_of",
            "source_game_count",
            "source_identity_sha256",
            "model_eligible_game_count",
            "model_eligible_identity_sha256",
            "source_receipt_sha256",
            "source_receipt_file_sha256",
        ):
            if source.get(key) != expected_source.get(key):
                raise PairedUncertaintyError(f"{variant} source field changed: {key}")
    ledger = result.get("prediction_ledger")
    if not isinstance(ledger, Mapping):
        raise PairedUncertaintyError(f"{variant} prediction ledger is missing")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or not all(isinstance(row, Mapping) for row in rows):
        raise PairedUncertaintyError(f"{variant} prediction ledger rows are invalid")
    if ledger.get("row_count") != len(rows):
        raise PairedUncertaintyError(f"{variant} prediction ledger row count changed")
    claimed_ledger_hash = _require_hash(ledger.get("sha256"), f"{variant} ledger hash")
    if _ledger_hash(rows) != claimed_ledger_hash:
        raise PairedUncertaintyError(f"{variant} prediction ledger hash changed")
    ids: list[str] = []
    parsed: dict[str, Any] = {}
    for row in rows:
        game_id = str(row.get("game_id", "")).strip()
        if not game_id or game_id in parsed:
            raise PairedUncertaintyError(f"{variant} prediction ledger has duplicate identity")
        try:
            fold = int(row["fold"])
        except (KeyError, TypeError, ValueError) as error:
            raise PairedUncertaintyError(f"{variant} ledger fold is invalid") from error
        if fold not in (1, 2, 3):
            raise PairedUncertaintyError(f"{variant} ledger fold is invalid")
        target = _finite_target(row.get("target"), f"{variant} target {game_id}")
        candidate = _finite_probability(
            row.get("candidate"), f"{variant} prediction {game_id}"
        )
        parsed[game_id] = {
            "fold": fold,
            "game_id": game_id,
            "target": target,
            "candidate": candidate,
        }
        ids.append(game_id)
    expected_identity = identity_sha256(ids)
    claimed_identity = _require_hash(
        ledger.get("game_identity_sha256"), f"{variant} ledger game identity"
    )
    if expected_identity != claimed_identity:
        raise PairedUncertaintyError(f"{variant} ledger game identity changed")
    return dict(result), parsed


def _bundle_series_assignments(
    path: Path,
    *,
    source: Mapping[str, Any],
    expected_game_ids: set[str],
) -> tuple[dict[str, str], dict[str, Any]]:
    bundle = _load_json(path, "feature-ledger bundle")
    if bundle.get("schema_version") != "scryglass:future-value-four-variant-ledger-bundle:v1":
        raise PairedUncertaintyError("feature-ledger bundle schema changed")
    if bundle.get("status") != "research_only":
        raise PairedUncertaintyError("feature-ledger bundle status is invalid")
    _verify_authority(bundle.get("authority"), "feature-ledger bundle authority")
    claimed_bundle_hash = _require_hash(bundle.get("bundle_sha256"), "bundle hash")
    payload = dict(bundle)
    payload.pop("bundle_sha256", None)
    if _sha256_bytes(_canonical_bytes(payload)) != claimed_bundle_hash:
        raise PairedUncertaintyError("feature-ledger bundle hash changed")
    bundle_source = bundle.get("source")
    if not isinstance(bundle_source, Mapping):
        raise PairedUncertaintyError("feature-ledger bundle source is missing")
    for key in (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "source_receipt_sha256",
        "source_receipt_file_sha256",
    ):
        if bundle_source.get(key) != source.get(key):
            raise PairedUncertaintyError(f"bundle source field changed: {key}")
    variants = bundle.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(VARIANTS):
        raise PairedUncertaintyError("feature-ledger bundle variants changed")
    assignments: dict[str, str] = {}
    fold_for_game: dict[str, int] = {}
    reference_fold_game_sets: dict[str, set[str]] = {}
    for variant in VARIANTS:
        variant_payload = variants[variant]
        if not isinstance(variant_payload, Mapping) or not isinstance(
            variant_payload.get("folds"), Mapping
        ):
            raise PairedUncertaintyError(f"{variant} bundle folds are missing")
        folds = variant_payload["folds"]
        for fold_key in ("1", "2", "3"):
            fold_payload = folds.get(fold_key)
            if not isinstance(fold_payload, Mapping) or not isinstance(
                fold_payload.get("rows"), list
            ):
                raise PairedUncertaintyError(f"{variant} bundle fold {fold_key} is invalid")
            rows = fold_payload["rows"]
            local: dict[str, tuple[str, str]] = {}
            for row in rows:
                if not isinstance(row, Mapping):
                    raise PairedUncertaintyError(f"{variant} bundle row is invalid")
                game_id = str(row.get("game_id", "")).strip()
                series_id = str(row.get("series_id", "")).strip()
                if not game_id or not series_id or game_id in local:
                    raise PairedUncertaintyError(f"{variant} bundle identity is invalid")
                local[game_id] = (series_id, str(row.get("date", "")))
            if variant == VARIANTS[0]:
                reference_fold_game_sets[fold_key] = set(local)
            elif set(local) != reference_fold_game_sets.get(fold_key):
                raise PairedUncertaintyError(f"{variant} bundle fold coverage changed")
            if variant == VARIANTS[0]:
                for game_id, (series_id, _date) in local.items():
                    if game_id not in expected_game_ids:
                        continue
                    if game_id in assignments:
                        if assignments[game_id] != series_id:
                            raise PairedUncertaintyError("bundle series assignment changed")
                        # Chronological folds can repeat an earlier validation
                        # map in a later training window. Keep its first fold.
                        continue
                    assignments[game_id] = series_id
                    fold_for_game[game_id] = int(fold_key)
            else:
                for game_id, (series_id, _date) in local.items():
                    if game_id in assignments and assignments[game_id] != series_id:
                        raise PairedUncertaintyError("bundle series assignment changed")
    if set(assignments) != expected_game_ids:
        missing = sorted(expected_game_ids - set(assignments))[:3]
        extra = sorted(set(assignments) - expected_game_ids)[:3]
        raise PairedUncertaintyError(
            f"bundle coverage changed; missing={missing}, extra={extra}"
        )
    return assignments, {
        "bundle_sha256": claimed_bundle_hash,
        "fold_for_game": fold_for_game,
        "series_assignment_sha256": _sha256_bytes(
            _canonical_bytes(
                [
                    {"game_id": game_id, "series_id": assignments[game_id]}
                    for game_id in sorted(assignments)
                ]
            )
        ),
    }


def _log_loss(target: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    clipped = np.clip(probability, LOG_EPSILON, 1.0 - LOG_EPSILON)
    values = -(target * np.log(clipped) + (1.0 - target) * np.log1p(-clipped))
    return float(np.sum(weight * values) / np.sum(weight))


def _brier(target: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sum(weight * (probability - target) ** 2) / np.sum(weight))


def _weighted_auc_sorted(
    target_sorted: np.ndarray,
    probability_sorted: np.ndarray,
    weight_sorted: np.ndarray,
) -> float:
    positive_total = float(np.sum(weight_sorted * target_sorted))
    negative_total = float(np.sum(weight_sorted * (1.0 - target_sorted)))
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise PairedUncertaintyError("bootstrap draw has one target class")
    starts = np.r_[True, probability_sorted[1:] != probability_sorted[:-1]]
    start_indices = np.flatnonzero(starts)
    positive = np.add.reduceat(weight_sorted * target_sorted, start_indices)
    negative = np.add.reduceat(weight_sorted * (1.0 - target_sorted), start_indices)
    negative_before = np.cumsum(negative) - negative
    numerator = np.sum(positive * (negative_before + 0.5 * negative))
    return float(numerator / (positive_total * negative_total))


def _auc(target: np.ndarray, probability: np.ndarray, weight: np.ndarray) -> float:
    order = np.argsort(probability, kind="mergesort")
    return _weighted_auc_sorted(target[order], probability[order], weight[order])


def _percentiles(values: np.ndarray) -> dict[str, float]:
    return {
        "lower_2_5": float(np.percentile(values, 2.5)),
        "median": float(np.percentile(values, 50.0)),
        "upper_97_5": float(np.percentile(values, 97.5)),
    }


def _paired_bootstrap(
    target: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    series: Sequence[str],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    if draws < 1000:
        raise PairedUncertaintyError("draws must be at least 1000")
    if not (len(target) == len(candidate) == len(baseline) == len(series)):
        raise PairedUncertaintyError("paired arrays have different lengths")
    if len(target) == 0:
        raise PairedUncertaintyError("paired arrays are empty")
    series_names = tuple(sorted(set(str(value) for value in series)))
    if any(not value for value in series_names):
        raise PairedUncertaintyError("series identity is incomplete")
    series_index = {value: index for index, value in enumerate(series_names)}
    row_cluster = np.asarray([series_index[str(value)] for value in series], dtype=np.int64)
    rng = np.random.default_rng(seed)
    candidate_order = np.argsort(candidate, kind="mergesort")
    baseline_order = np.argsort(baseline, kind="mergesort")
    candidate_sorted_target = target[candidate_order]
    baseline_sorted_target = target[baseline_order]
    candidate_sorted_probability = candidate[candidate_order]
    baseline_sorted_probability = baseline[baseline_order]
    delta_log_loss: list[float] = []
    delta_brier: list[float] = []
    delta_auc: list[float] = []
    rejected = 0
    attempts = 0
    max_attempts = draws * 20
    while len(delta_log_loss) < draws and attempts < max_attempts:
        attempts += 1
        sampled = rng.integers(0, len(series_names), size=len(series_names), dtype=np.int64)
        counts = np.bincount(sampled, minlength=len(series_names)).astype(float)
        weight = counts[row_cluster]
        if not np.any(weight > 0.0):
            rejected += 1
            continue
        try:
            candidate_auc = _weighted_auc_sorted(
                candidate_sorted_target,
                candidate_sorted_probability,
                weight[candidate_order],
            )
            baseline_auc = _weighted_auc_sorted(
                baseline_sorted_target,
                baseline_sorted_probability,
                weight[baseline_order],
            )
        except PairedUncertaintyError:
            rejected += 1
            continue
        delta_log_loss.append(
            _log_loss(target, candidate, weight) - _log_loss(target, baseline, weight)
        )
        delta_brier.append(_brier(target, candidate, weight) - _brier(target, baseline, weight))
        delta_auc.append(candidate_auc - baseline_auc)
    if len(delta_log_loss) != draws:
        raise PairedUncertaintyError(
            f"only {len(delta_log_loss)} of {draws} bootstrap draws were accepted"
        )
    log_values = np.asarray(delta_log_loss, dtype=float)
    brier_values = np.asarray(delta_brier, dtype=float)
    auc_values = np.asarray(delta_auc, dtype=float)
    full_weight = np.ones(len(target), dtype=float)
    observed = {
        "candidate": {
            "log_loss": _log_loss(target, candidate, full_weight),
            "brier": _brier(target, candidate, full_weight),
            "auc": _auc(target, candidate, full_weight),
        },
        "baseline": {
            "log_loss": _log_loss(target, baseline, full_weight),
            "brier": _brier(target, baseline, full_weight),
            "auc": _auc(target, baseline, full_weight),
        },
    }
    return {
        "rows": int(len(target)),
        "series_count": int(len(series_names)),
        "draws_requested": int(draws),
        "draws_accepted": int(draws),
        "draws_rejected": int(rejected),
        "seed": int(seed),
        "observed": observed,
        "metrics": {
            "log_loss": {
                "delta_candidate_minus_baseline": float(observed["candidate"]["log_loss"] - observed["baseline"]["log_loss"]),
                "percentile_interval": _percentiles(log_values),
                "probability_candidate_improves": float(np.mean(log_values < 0.0)),
            },
            "brier": {
                "delta_candidate_minus_baseline": float(observed["candidate"]["brier"] - observed["baseline"]["brier"]),
                "percentile_interval": _percentiles(brier_values),
                "probability_candidate_improves": float(np.mean(brier_values < 0.0)),
            },
            "auc": {
                "delta_candidate_minus_baseline": float(observed["candidate"]["auc"] - observed["baseline"]["auc"]),
                "percentile_interval": _percentiles(auc_values),
                "probability_candidate_improves": float(np.mean(auc_values > 0.0)),
            },
            "all_three_metrics": {
                "probability_candidate_improves": float(
                    np.mean((log_values < 0.0) & (brier_values < 0.0) & (auc_values > 0.0))
                )
            },
        },
    }


def _prediction_frame(
    ledgers: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, str],
    fold_for_game: Mapping[str, int],
) -> list[dict[str, Any]]:
    baseline = ledgers["current_only"]
    game_ids = set(baseline)
    for variant in VARIANTS:
        if set(ledgers[variant]) != game_ids:
            raise PairedUncertaintyError(f"{variant} prediction coverage differs")
    rows: list[dict[str, Any]] = []
    for game_id in sorted(game_ids, key=lambda value: (fold_for_game.get(value, 0), value)):
        base = baseline[game_id]
        if game_id not in assignments or game_id not in fold_for_game:
            raise PairedUncertaintyError(f"series assignment is missing for {game_id}")
        for variant in VARIANTS:
            row = ledgers[variant][game_id]
            if row["fold"] != base["fold"] or row["target"] != base["target"]:
                raise PairedUncertaintyError(f"{variant} target or fold changed for {game_id}")
        rows.append(
            {
                "fold": int(base["fold"]),
                "game_id": game_id,
                "series_id": assignments[game_id],
                "target": float(base["target"]),
            }
        )
    return rows


def build_report(
    *,
    evaluation_root: Path,
    bundle_path: Path,
    draws: int = 2000,
    seed: int = 461,
) -> dict[str, Any]:
    if draws < 1000:
        raise PairedUncertaintyError("draws must be at least 1000")
    model_paths = {
        variant: evaluation_root / variant / "model.json" for variant in VARIANTS
    }
    first_document = _load_json(model_paths[VARIANTS[0]], "current_only model")
    first_result = first_document.get("variants", {}).get(VARIANTS[0])
    if not isinstance(first_result, Mapping) or not isinstance(first_result.get("source"), Mapping):
        raise PairedUncertaintyError("current_only source binding is missing")
    expected_source = first_result["source"]
    result_by_variant: dict[str, dict[str, Any]] = {}
    ledgers: dict[str, dict[str, Any]] = {}
    file_bindings: dict[str, dict[str, Any]] = {}
    for variant in VARIANTS:
        model_result, ledger = _load_prediction_ledger(
            model_paths[variant], variant, expected_source=expected_source
        )
        result_by_variant[variant] = model_result
        ledgers[variant] = ledger
        file_bindings[variant] = {
            "path": str(model_paths[variant]),
            "bytes": int(model_paths[variant].stat().st_size),
            "sha256": _sha256_path(model_paths[variant]),
            "prediction_ledger_sha256": model_result["prediction_ledger"]["sha256"],
        }
    expected_ids = set(ledgers[VARIANTS[0]])
    assignments, bundle_binding = _bundle_series_assignments(
        bundle_path, source=expected_source, expected_game_ids=expected_ids
    )
    fold_for_game = bundle_binding["fold_for_game"]
    frame = _prediction_frame(ledgers, assignments, fold_for_game)
    index_by_game = {row["game_id"]: index for index, row in enumerate(frame)}
    target = np.asarray([row["target"] for row in frame], dtype=float)
    series = [str(row["series_id"]) for row in frame]
    comparisons: dict[str, Any] = {}
    csv_rows: list[dict[str, Any]] = []
    for candidate_name, baseline_name in COMPARISONS:
        candidate = np.asarray(
            [ledgers[candidate_name][game_id]["candidate"] for game_id in index_by_game],
            dtype=float,
        )
        baseline = np.asarray(
            [ledgers[baseline_name][game_id]["candidate"] for game_id in index_by_game],
            dtype=float,
        )
        key = f"{candidate_name}_vs_{baseline_name}"
        comparison = _paired_bootstrap(
            target, candidate, baseline, series, draws=draws, seed=seed
        )
        comparison["candidate"] = candidate_name
        comparison["baseline"] = baseline_name
        comparisons[key] = comparison
        for metric_name, metric in comparison["metrics"].items():
            if metric_name == "all_three_metrics":
                continue
            csv_rows.append(
                {
                    "comparison": key,
                    "candidate": candidate_name,
                    "baseline": baseline_name,
                    "metric": metric_name,
                    "rows": comparison["rows"],
                    "series_count": comparison["series_count"],
                    "draws": comparison["draws_accepted"],
                    "delta_candidate_minus_baseline": metric[
                        "delta_candidate_minus_baseline"
                    ],
                    "ci_lower_2_5": metric["percentile_interval"]["lower_2_5"],
                    "ci_median": metric["percentile_interval"]["median"],
                    "ci_upper_97_5": metric["percentile_interval"]["upper_97_5"],
                    "probability_candidate_improves": metric[
                        "probability_candidate_improves"
                    ],
                }
            )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only",
        "authority": {
            "research_only": True,
            "public_probability": False,
            "public_player_rating": False,
            "public_team_rating": False,
            "promotion": False,
            "deployment": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
        },
        "method": {
            "resampling": "whole_conservative_series_cluster_with_replacement",
            "delta_convention": "candidate_minus_baseline",
            "loss_improvement": "delta_less_than_zero",
            "auc_improvement": "delta_greater_than_zero",
            "percentile_interval": "95_percentile_2_5_to_97_5",
            "seed": int(seed),
            "draws": int(draws),
        },
        "source": {
            key: expected_source[key]
            for key in (
                "source_as_of",
                "source_game_count",
                "source_identity_sha256",
                "model_eligible_game_count",
                "model_eligible_identity_sha256",
                "source_receipt_sha256",
                "source_receipt_file_sha256",
            )
        },
        "coverage": {
            "rows": len(frame),
            "game_identity_sha256": identity_sha256(index_by_game),
            "series_count": len(set(series)),
            "folds": {
                str(fold): sum(1 for row in frame if row["fold"] == fold)
                for fold in (1, 2, 3)
            },
            "series_assignment_sha256": bundle_binding["series_assignment_sha256"],
            "fold_game_identity_sha256": {
                str(fold): identity_sha256(
                    row["game_id"] for row in frame if row["fold"] == fold
                )
                for fold in (1, 2, 3)
            },
        },
        "bindings": {
            "bundle": {
                "path": str(bundle_path),
                "bytes": int(bundle_path.stat().st_size),
                **bundle_binding,
            },
            "models": file_bindings,
        },
        "comparisons": comparisons,
    }


def _write_csv(path: Path, report: Mapping[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for key, comparison in report["comparisons"].items():
        for metric_name, metric in comparison["metrics"].items():
            if metric_name == "all_three_metrics":
                continue
            rows.append(
                {
                    "comparison": key,
                    "candidate": comparison["candidate"],
                    "baseline": comparison["baseline"],
                    "metric": metric_name,
                    "rows": comparison["rows"],
                    "series_count": comparison["series_count"],
                    "draws": comparison["draws_accepted"],
                    "delta_candidate_minus_baseline": metric[
                        "delta_candidate_minus_baseline"
                    ],
                    "ci_lower_2_5": metric["percentile_interval"]["lower_2_5"],
                    "ci_median": metric["percentile_interval"]["median"],
                    "ci_upper_97_5": metric["percentile_interval"]["upper_97_5"],
                    "probability_candidate_improves": metric[
                        "probability_candidate_improves"
                    ],
                }
            )
    columns = tuple(rows[0]) if rows else ()
    lines = [",".join(columns)]
    for row in rows:
        lines.append(",".join(json.dumps(row[column], ensure_ascii=True) for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(report: Mapping[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "paired-uncertainty.json"
    csv_path = output_dir / "paired-uncertainty.csv"
    json_path.write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(csv_path, report)
    return json_path, csv_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=461)
    args = parser.parse_args(argv)
    report = build_report(
        evaluation_root=args.evaluation_root,
        bundle_path=args.bundle,
        draws=args.draws,
        seed=args.seed,
    )
    json_path, csv_path = write_report(report, args.output_dir)
    print(json.dumps({"json": str(json_path), "csv": str(csv_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
