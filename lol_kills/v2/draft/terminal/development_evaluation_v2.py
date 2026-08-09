"""Source-frozen, dependence-clustered terminal Draft Score development v2.

Version 1 accidentally made regularization vanish with sample size and read a
mutable warehouse directly.  This evaluator fixes both defects without
reinterpreting the old result: it consumes exact frozen cohort bytes and uses
the explicitly normalized objective

    mean logistic loss + 0.5 * ridge_strength * ||draft coefficients||^2

while leaving the pre-event team-strength nuisance coefficient unpenalized.
All candidate and ridge choices are made on chronological validation slices;
calibration is fit later; only that fold's selected variant reaches its outer
test.  These are still adaptive development diagnostics, never promotion or
betting authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from .development_evaluation import (
    BASELINE_CONFIG,
    CALIBRATION_ORDER,
    CANDIDATE_ORDER,
    DraftRow,
    _brier,
    _cluster_metrics,
    _design,
    _fit_calibration,
    _league_metrics,
    _log_loss,
    _parse_time,
    _patch_sort_key,
    _probabilities,
    chronological_folds,
    feature_vocabulary,
    pre_event_team_elo_logits,
)
from .development_snapshot import (
    DEFAULT_MANIFEST,
    DevelopmentSnapshotError,
    load_development_snapshot,
)


SCHEMA_VERSION = "draft-terminal-development-evaluation-v2"
SUMMARY_SCHEMA_VERSION = "scryglass:draft-terminal-development-evaluation-summary:v2"
DEFAULT_SUMMARY = Path(
    "data/lol/v2/models/draft-terminal/development-evaluation-summary-v2.json"
)
# Frozen in ascending penalty order before any future/prospective outcome is
# observed.  These are coefficients in a mean-loss objective, so their meaning
# does not change with the number of training maps.
RIDGE_STRENGTH_ORDER = (0.02, 0.05, 0.10, 0.20, 0.40)
OPTIMIZER_MAX_ITERATIONS = 500
OPTIMIZER_GRADIENT_TOLERANCE = 1e-8
OPTIMIZER_ABSOLUTE_PARAMETER_BOUND = 20.0
_VARIANT_RE = re.compile(r"^(?P<candidate>.+)@ridge-(?P<ridge>[0-9.]+)$")


class DevelopmentEvaluationV2Error(ValueError):
    """Raised when the source-frozen v2 development evaluation is invalid."""


@dataclass(frozen=True)
class PenalizedFit:
    candidate_id: str
    ridge_strength: float
    vocabulary: tuple[str, ...]
    beta: np.ndarray
    baseline_coefficient: float
    optimizer_iterations: int
    optimizer_gradient_max_abs: float

    @property
    def variant_id(self) -> str:
        return _variant_id(self.candidate_id, self.ridge_strength)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _variant_id(candidate_id: str, ridge_strength: float) -> str:
    return f"{candidate_id}@ridge-{ridge_strength:.2f}"


def _sigmoid_vector(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _baseline_initial_coefficient(
    baseline: np.ndarray, labels: np.ndarray
) -> float:
    coefficient = 0.0
    for _ in range(100):
        logits = coefficient * baseline
        probabilities = _sigmoid_vector(logits)
        gradient = float(np.mean(baseline * (probabilities - labels)))
        hessian = float(
            np.mean(baseline * baseline * probabilities * (1.0 - probabilities))
        )
        if hessian <= 1e-12:
            break
        step = gradient / hessian
        coefficient -= step
        if abs(step) <= 1e-12:
            break
    if not math.isfinite(coefficient):
        raise DevelopmentEvaluationV2Error(
            "baseline nuisance initialization is non-finite"
        )
    return coefficient


def fit_penalized(
    rows: Sequence[DraftRow],
    candidate_id: str,
    ridge_strength: float,
    baseline_logits: Mapping[str, float],
) -> PenalizedFit:
    """Fit normalized ridge draft terms with an unpenalized nuisance baseline."""

    if candidate_id not in CANDIDATE_ORDER:
        raise DevelopmentEvaluationV2Error(f"unregistered candidate: {candidate_id}")
    if ridge_strength not in RIDGE_STRENGTH_ORDER:
        raise DevelopmentEvaluationV2Error(
            f"unregistered ridge strength: {ridge_strength}"
        )
    if not rows:
        raise DevelopmentEvaluationV2Error("cannot fit an empty development slice")
    vocabulary = feature_vocabulary(rows, candidate_id)
    design = _design(rows, candidate_id, vocabulary).tocsr()
    labels = np.asarray([row.label_a for row in rows], dtype=float)
    nuisance = np.asarray(
        [float(baseline_logits[row.game_id]) for row in rows], dtype=float
    )
    if not np.all(np.isfinite(nuisance)):
        raise DevelopmentEvaluationV2Error("baseline nuisance contains non-finite values")
    initial = np.zeros(len(vocabulary) + 1, dtype=float)
    initial[-1] = _baseline_initial_coefficient(nuisance, labels)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        beta = parameters[:-1]
        baseline_coefficient = float(parameters[-1])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = (
                np.asarray(design @ beta, dtype=float)
                + baseline_coefficient * nuisance
            )
            probabilities = _sigmoid_vector(logits)
            loss = float(
                np.mean(np.logaddexp(0.0, logits) - labels * logits)
                + 0.5 * ridge_strength * float(np.sum(np.square(beta)))
            )
            residual = (probabilities - labels) / len(labels)
            gradient_beta = np.asarray(design.T @ residual, dtype=float).reshape(-1)
            gradient_beta += ridge_strength * beta
            gradient_baseline = float(np.sum(nuisance * residual))
        gradient = np.concatenate(
            [gradient_beta, np.asarray([gradient_baseline], dtype=float)]
        )
        return loss, gradient

    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[
            (-OPTIMIZER_ABSOLUTE_PARAMETER_BOUND, OPTIMIZER_ABSOLUTE_PARAMETER_BOUND)
        ]
        * len(initial),
        options={
            "maxiter": OPTIMIZER_MAX_ITERATIONS,
            "gtol": OPTIMIZER_GRADIENT_TOLERANCE,
            "ftol": 1e-12,
            "maxls": 50,
        },
    )
    parameters = np.asarray(optimized.x, dtype=float)
    _, gradient = objective(parameters)
    gradient_max_abs = float(np.max(np.abs(gradient))) if len(gradient) else 0.0
    if (
        not optimized.success
        or not np.all(np.isfinite(parameters))
        or not math.isfinite(gradient_max_abs)
        or gradient_max_abs > 5e-6
    ):
        raise DevelopmentEvaluationV2Error(
            "penalized fit did not converge "
            f"for {_variant_id(candidate_id, ridge_strength)}: "
            f"{optimized.message}; gradient={gradient_max_abs:.3g}"
        )
    return PenalizedFit(
        candidate_id=candidate_id,
        ridge_strength=ridge_strength,
        vocabulary=vocabulary,
        beta=parameters[:-1],
        baseline_coefficient=float(parameters[-1]),
        optimizer_iterations=int(optimized.nit),
        optimizer_gradient_max_abs=gradient_max_abs,
    )


def composition_logits(rows: Sequence[DraftRow], fit: PenalizedFit) -> np.ndarray:
    result = np.asarray(
        _design(rows, fit.candidate_id, fit.vocabulary) @ fit.beta, dtype=float
    ).reshape(-1)
    if not np.all(np.isfinite(result)):
        raise DevelopmentEvaluationV2Error("composition logits are non-finite")
    return result


def baseline_adjusted_logits(
    rows: Sequence[DraftRow],
    fit: PenalizedFit,
    baseline_logits: Mapping[str, float],
) -> np.ndarray:
    nuisance = np.asarray(
        [float(baseline_logits[row.game_id]) for row in rows], dtype=float
    )
    result = composition_logits(rows, fit) + fit.baseline_coefficient * nuisance
    if not np.all(np.isfinite(result)):
        raise DevelopmentEvaluationV2Error(
            "baseline-adjusted logits are non-finite"
        )
    return result


def _fold_rows(
    rows: Sequence[DraftRow], series_order: Sequence[str], span: tuple[int, int]
) -> list[DraftRow]:
    selected = set(series_order[span[0] : span[1]])
    return [row for row in rows if row.dependence_cluster_id in selected]


def _code_bindings() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = {
        "development_evaluation_v2": Path(__file__),
        "development_evaluation_helpers": directory / "development_evaluation.py",
        "development_snapshot": directory / "development_snapshot.py",
    }
    return {name: _sha256(path.read_bytes()) for name, path in paths.items()}


def _variant_sort_key(report: Mapping[str, Any]) -> tuple[float, float, int, int]:
    candidate_id = str(report["candidate_id"])
    ridge_strength = float(report["ridge_strength"])
    return (
        float(report["validation_equalized_draft"]["log_loss"]),
        float(report["validation_equalized_draft"]["brier_score"]),
        CANDIDATE_ORDER.index(candidate_id),
        RIDGE_STRENGTH_ORDER.index(ridge_strength),
    )


def evaluate(root: Path) -> dict[str, Any]:
    rows, source_snapshot = load_development_snapshot(root)
    cluster_latest: dict[str, Any] = {}
    for row in rows:
        cluster_latest[row.dependence_cluster_id] = max(
            cluster_latest.get(row.dependence_cluster_id, row.date), row.date
        )
    series_order = [
        series_id
        for series_id, _ in sorted(
            cluster_latest.items(), key=lambda item: (item[1], item[0])
        )
    ]
    folds = chronological_folds(len(series_order))
    fold_reports: list[dict[str, Any]] = []
    aggregate_validation: dict[str, list[dict[str, float]]] = {}
    for fold in folds:
        train = _fold_rows(rows, series_order, fold.train)
        validation = _fold_rows(rows, series_order, fold.validation)
        calibration = _fold_rows(rows, series_order, fold.calibration)
        test = _fold_rows(rows, series_order, fold.test)
        test_start = min((row.date for row in test), default=None)
        if test_start is None:
            raise DevelopmentEvaluationV2Error(f"{fold.fold_id} outer test is empty")
        fold_baseline_logits = pre_event_team_elo_logits(rows, freeze_at=test_start)
        variant_reports: list[dict[str, Any]] = []
        fitted: dict[str, PenalizedFit] = {}
        for candidate_id in CANDIDATE_ORDER:
            for ridge_strength in RIDGE_STRENGTH_ORDER:
                fit = fit_penalized(
                    train,
                    candidate_id,
                    ridge_strength,
                    fold_baseline_logits,
                )
                fitted[fit.variant_id] = fit
                validation_equalized = composition_logits(validation, fit)
                validation_adjusted = baseline_adjusted_logits(
                    validation, fit, fold_baseline_logits
                )
                report = {
                    "variant_id": fit.variant_id,
                    "candidate_id": candidate_id,
                    "ridge_strength": ridge_strength,
                    "feature_count": len(fit.vocabulary),
                    "coefficient_l2": float(np.linalg.norm(fit.beta)),
                    "baseline_coefficient": fit.baseline_coefficient,
                    "optimizer_iterations": fit.optimizer_iterations,
                    "optimizer_gradient_max_abs": fit.optimizer_gradient_max_abs,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                    "validation_equalized_draft": _cluster_metrics(
                        validation, _probabilities(validation_equalized, 1.0)
                    ),
                    "validation_baseline_adjusted": _cluster_metrics(
                        validation, _probabilities(validation_adjusted, 1.0)
                    ),
                    "selected_for_outer_test": False,
                }
                variant_reports.append(report)
                aggregate_validation.setdefault(fit.variant_id, []).append(
                    {
                        "log_loss": float(
                            report["validation_equalized_draft"]["log_loss"]
                        ),
                        "brier_score": float(
                            report["validation_equalized_draft"]["brier_score"]
                        ),
                    }
                )
        selected = min(variant_reports, key=_variant_sort_key)
        selected["selected_for_outer_test"] = True
        selected_fit = fitted[str(selected["variant_id"])]
        calibration_logits = composition_logits(calibration, selected_fit)
        calibration_choices: list[tuple[float, int, str, float]] = []
        calibration_reports: list[dict[str, Any]] = []
        for method in CALIBRATION_ORDER:
            parameter, loss = _fit_calibration(
                calibration_logits, [row.label_a for row in calibration], method
            )
            calibration_choices.append(
                (float(loss), CALIBRATION_ORDER.index(method), method, parameter)
            )
            calibration_reports.append(
                {
                    "method": method,
                    "parameter": parameter,
                    "calibration_log_loss": loss,
                }
            )
        _, _, selected_transform, selected_parameter = min(calibration_choices)
        scale = (
            1.0 / selected_parameter
            if selected_transform == "symmetric_temperature"
            else selected_parameter
        )
        for item in calibration_reports:
            item["selected"] = item["method"] == selected_transform
        test_equalized_logits = composition_logits(test, selected_fit)
        test_adjusted_logits = baseline_adjusted_logits(
            test, selected_fit, fold_baseline_logits
        )
        equalized_probabilities = _probabilities(test_equalized_logits, scale)
        adjusted_probabilities = _probabilities(test_adjusted_logits, scale)
        neutral_probabilities = np.full(len(test), 0.5, dtype=float)
        selected["calibration_transforms"] = calibration_reports
        selected["locked_outer_test"] = _cluster_metrics(
            test, equalized_probabilities
        )
        selected["locked_outer_test_by_league"] = _league_metrics(
            test, equalized_probabilities
        )
        selected["baseline_adjusted_locked_outer_test"] = _cluster_metrics(
            test, adjusted_probabilities
        )
        selected["neutral_locked_outer_test"] = _cluster_metrics(
            test, neutral_probabilities
        )
        selected["locked_outer_test_delta_vs_neutral"] = {
            "log_loss": float(
                selected["locked_outer_test"]["log_loss"]
                - selected["neutral_locked_outer_test"]["log_loss"]
            ),
            "brier_score": float(
                selected["locked_outer_test"]["brier_score"]
                - selected["neutral_locked_outer_test"]["brier_score"]
            ),
            "negative_is_better": True,
        }
        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "dependence_cluster_spans": {
                    "train": fold.train,
                    "validation": fold.validation,
                    "calibration": fold.calibration,
                    "test": fold.test,
                },
                "date_ranges": {
                    name: {
                        "start": min((row.date for row in subset), default=None).isoformat()
                        if subset
                        else None,
                        "end": max((row.date for row in subset), default=None).isoformat()
                        if subset
                        else None,
                    }
                    for name, subset in (
                        ("train", train),
                        ("validation", validation),
                        ("calibration", calibration),
                        ("test", test),
                    )
                },
                "selection": {
                    "variant_id": selected["variant_id"],
                    "candidate_id": selected["candidate_id"],
                    "ridge_strength": selected["ridge_strength"],
                    "criterion": "equalized_validation_log_loss_then_brier_then_frozen_order",
                    "calibration_transform": selected_transform,
                    "outer_test_locked": True,
                },
                "baseline_state_policy": {
                    "outer_test_frozen_at": test_start.isoformat(),
                    "outer_test_outcomes_update_baseline": False,
                    "baseline_nuisance_penalized": False,
                },
                "variants": variant_reports,
            }
        )

    aggregate_reports: list[dict[str, Any]] = []
    for variant_id, metrics in aggregate_validation.items():
        match = _VARIANT_RE.fullmatch(variant_id)
        if match is None:
            raise DevelopmentEvaluationV2Error("variant id cannot be parsed")
        ridge_strength = float(match.group("ridge"))
        aggregate_reports.append(
            {
                "variant_id": variant_id,
                "candidate_id": match.group("candidate"),
                "ridge_strength": ridge_strength,
                "folds": len(metrics),
                "mean_validation_log_loss": float(
                    np.mean([item["log_loss"] for item in metrics])
                ),
                "mean_validation_brier_score": float(
                    np.mean([item["brier_score"] for item in metrics])
                ),
            }
        )
    development_candidate = min(
        aggregate_reports,
        key=lambda report: (
            report["mean_validation_log_loss"],
            report["mean_validation_brier_score"],
            CANDIDATE_ORDER.index(report["candidate_id"]),
            RIDGE_STRENGTH_ORDER.index(report["ridge_strength"]),
        ),
    )
    patches = sorted({row.patch for row in rows}, key=_patch_sort_key)
    latest_patch = patches[-1] if patches else None
    international = {"MSI", "EWC", "WORLDS", "WORLD CHAMPIONSHIP"}
    international_rows = [row for row in rows if row.league in international]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only",
        "production_eligible": False,
        "public_probability_authorized": False,
        "claim_ceiling": {
            "development_diagnostic": True,
            "descriptive_premap_association": False,
            "causal_draft_effect": False,
            "recommendation": False,
            "betting": False,
            "reliability": False,
        },
        "code_bindings": _code_bindings(),
        "source_snapshot": source_snapshot,
        "objective": {
            "loss": "mean_binary_log_loss",
            "draft_penalty": "0.5_times_ridge_strength_times_squared_l2",
            "baseline_nuisance_penalized": False,
            "ridge_strength_order": list(RIDGE_STRENGTH_ORDER),
            "sample_size_invariant": True,
            "absolute_parameter_bound": OPTIMIZER_ABSOLUTE_PARAMETER_BOUND,
        },
        "baseline_adjustment": {
            **BASELINE_CONFIG,
            "status": "development_nuisance_only",
            "served_baseline_logit": 0.0,
            "team_identity_in_served_artifact": False,
        },
        "population": {
            "complete_rows": len(rows),
            "dependence_clusters": len(series_order),
            "unclustered_rows": 0,
            "leagues": sorted({row.league for row in rows}),
            "patches": patches,
            "international_event_rows": len(international_rows),
        },
        "split_policy": {
            "folds": len(folds),
            "chronological": True,
            "dependence_clustered": True,
            "unclustered_maps_excluded": True,
            "series_grouped": False,
            "series_identity_status": "outcome_free_proxy_only_not_authoritative",
            "participant_cluster_status": "unavailable",
            "candidate_selection_on_validation_only": True,
            "calibration_fit_after_candidate_selection": True,
            "outer_test_scored_only_for_fold_selected_variant": True,
            "candidate_search_opened_on_outer_test": False,
            "future_candidate_selected_from_validation_only": True,
        },
        "candidate_order": list(CANDIDATE_ORDER),
        "ridge_strength_order": list(RIDGE_STRENGTH_ORDER),
        "calibration_order": list(CALIBRATION_ORDER),
        "development_candidate_for_future_freeze": {
            **development_candidate,
            "selection_scope": "mean_chronological_validation_only",
            "independent_validation": False,
            "authorizes_retrospective_claim": False,
            "authorizes_future_probability": False,
        },
        "aggregate_validation": aggregate_reports,
        "holdouts": {
            "future_patch": {
                "status": "development_diagnostic_only"
                if latest_patch
                else "unavailable",
                "patch_id": latest_patch,
                "rows": sum(row.patch == latest_patch for row in rows)
                if latest_patch
                else 0,
                "dependence_clusters": len(
                    {
                        row.dependence_cluster_id
                        for row in rows
                        if row.patch == latest_patch
                    }
                )
                if latest_patch
                else 0,
                "promotion": False,
            },
            "league": {
                "status": "development_diagnostic_only",
                "scored_within_each_outer_test": True,
                "promotion": False,
            },
            "international_event_or_meta": {
                "status": "development_diagnostic_only"
                if international_rows
                else "unavailable",
                "rows": len(international_rows),
                "leagues": sorted({row.league for row in international_rows}),
                "promotion": False,
            },
            "roster_change": {
                "status": "not_applicable",
                "reason": "neutral estimator contains no player or exact-roster identity terms",
            },
            "sparse_or_new_champion": {
                "status": "development_diagnostic_only",
                "promotion": False,
            },
        },
        "folds": fold_reports,
    }


def build_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    selected_tests: list[dict[str, Any]] = []
    for fold in report["folds"]:
        selected = next(
            item for item in fold["variants"] if item["selected_for_outer_test"]
        )
        selected_tests.append(
            {
                "fold_id": fold["fold_id"],
                "variant_id": selected["variant_id"],
                "candidate_id": selected["candidate_id"],
                "ridge_strength": selected["ridge_strength"],
                "calibration_transform": fold["selection"][
                    "calibration_transform"
                ],
                "locked_outer_test": selected["locked_outer_test"],
                "neutral_locked_outer_test": selected["neutral_locked_outer_test"],
                "locked_outer_test_delta_vs_neutral": selected[
                    "locked_outer_test_delta_vs_neutral"
                ],
                "baseline_adjusted_locked_outer_test": selected[
                    "baseline_adjusted_locked_outer_test"
                ],
            }
        )
    serialized = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_id": "draft-terminal-development-evaluation-summary-v2",
        "status": report["status"],
        "production_eligible": False,
        "public_probability_authorized": False,
        "run_command": "python3 -W error -m lol_kills.v2.draft.terminal.development_evaluation_v2",
        "run_output_sha256": _sha256(serialized.encode("ascii")),
        "code_bindings": report["code_bindings"],
        "source_snapshot": report["source_snapshot"],
        "objective": report["objective"],
        "population": report["population"],
        "split_policy": report["split_policy"],
        "development_candidate_for_future_freeze": report[
            "development_candidate_for_future_freeze"
        ],
        "fold_locked_selected_test": selected_tests,
        "holdouts": report["holdouts"],
        "grid_promotion_gate": {
            "status": "not_passed",
            "baseline_source": "OE",
            "candidate_source": "GRID",
            "primary_source_for_cohort": "OE",
            "public_reproducibility_benchmark": "OE",
            "reason": "no authorized complete hash-verified GRID Draft Score cohort has passed the gate",
        },
        "claim_ceiling": report["claim_ceiling"],
    }


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DevelopmentEvaluationV2Error(
            f"refusing to overwrite development evaluation evidence: {path}"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise DevelopmentEvaluationV2Error(
                f"refusing to overwrite development evaluation evidence: {path}"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_summary(root: Path, report: Mapping[str, Any]) -> Path:
    path = root / DEFAULT_SUMMARY
    summary = build_summary(report)
    _atomic_write_new(
        path,
        (
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("ascii"),
    )
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--write-summary", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = evaluate(args.root)
        if args.write_summary:
            path = write_summary(args.root, report)
            print(
                json.dumps(
                    {
                        "summary": str(path),
                        "summary_raw_sha256": _sha256(path.read_bytes()),
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    except (
        OSError,
        DevelopmentEvaluationV2Error,
        DevelopmentSnapshotError,
        ValueError,
    ) as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
