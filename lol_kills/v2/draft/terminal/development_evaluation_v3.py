"""Evaluate draft terms only by incremental value over pre-event context.

The v2 source/regularization repair still ranked equal-strength composition
logits against observed match outcomes.  That is the wrong validation target:
real teams are unequal.  V3 preserves the exact frozen cohort and normalized
ridge fits, but selects draft variants by whether a contextual model with draft
terms improves on a separately fit model using the same pre-event team-strength
input and training window.  The neutral composition output is retained only as
an equal-strength index; it is not claimed to be outcome-calibrated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .development_evaluation import (
    BASELINE_CONFIG,
    CALIBRATION_ORDER,
    CANDIDATE_ORDER,
    DraftRow,
    _cluster_metrics,
    _fit_calibration,
    _league_metrics,
    _patch_sort_key,
    _probabilities,
    chronological_folds,
    pre_event_team_elo_logits,
)
from .development_evaluation_v2 import (
    RIDGE_STRENGTH_ORDER,
    _baseline_initial_coefficient,
    _fold_rows,
    baseline_adjusted_logits,
    composition_logits,
    fit_penalized,
)
from .development_snapshot import (
    DevelopmentSnapshotError,
    load_development_snapshot,
)


SCHEMA_VERSION = "draft-terminal-development-evaluation-v3"
SUMMARY_SCHEMA_VERSION = "scryglass:draft-terminal-development-evaluation-summary:v3"
DEFAULT_SUMMARY = Path(
    "data/lol/v2/models/draft-terminal/development-evaluation-summary-v3.json"
)


class DevelopmentEvaluationV3Error(ValueError):
    """Raised when incremental-context development evaluation fails closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _code_bindings() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    paths = {
        "development_evaluation_v3": Path(__file__),
        "development_evaluation_v2_fit": directory / "development_evaluation_v2.py",
        "development_evaluation_helpers": directory / "development_evaluation.py",
        "development_snapshot": directory / "development_snapshot.py",
    }
    return {name: _sha256(path.read_bytes()) for name, path in paths.items()}


def fit_baseline_only(
    rows: Sequence[DraftRow], baseline_logits: Mapping[str, float]
) -> float:
    if not rows:
        raise DevelopmentEvaluationV3Error("baseline-only fit slice is empty")
    nuisance = np.asarray(
        [float(baseline_logits[row.game_id]) for row in rows], dtype=float
    )
    labels = np.asarray([row.label_a for row in rows], dtype=float)
    return _baseline_initial_coefficient(nuisance, labels)


def baseline_only_logits(
    rows: Sequence[DraftRow],
    coefficient: float,
    baseline_logits: Mapping[str, float],
) -> np.ndarray:
    return coefficient * np.asarray(
        [float(baseline_logits[row.game_id]) for row in rows], dtype=float
    )


def _calibration(
    logits: np.ndarray, labels: Sequence[int]
) -> tuple[str, float, list[dict[str, Any]]]:
    choices: list[tuple[float, int, str, float]] = []
    reports: list[dict[str, Any]] = []
    for method in CALIBRATION_ORDER:
        parameter, loss = _fit_calibration(logits, labels, method)
        choices.append((float(loss), CALIBRATION_ORDER.index(method), method, parameter))
        reports.append(
            {
                "method": method,
                "parameter": parameter,
                "calibration_log_loss": loss,
            }
        )
    _, _, selected_method, selected_parameter = min(choices)
    scale = (
        1.0 / selected_parameter
        if selected_method == "symmetric_temperature"
        else selected_parameter
    )
    for report in reports:
        report["selected"] = report["method"] == selected_method
    return selected_method, float(scale), reports


def _incremental_delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, Any]:
    brier = float(candidate["brier_score"]) - float(baseline["brier_score"])
    log_loss = float(candidate["log_loss"]) - float(baseline["log_loss"])
    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "pass_rule": "both deltas must be nonpositive",
        "passed": brier <= 0.0 and log_loss <= 0.0,
        "negative_is_better": True,
    }


def _variant_order(report: Mapping[str, Any]) -> tuple[float, float, int, int]:
    return (
        float(report["validation_incremental_vs_baseline_only"]["log_loss"]),
        float(report["validation_incremental_vs_baseline_only"]["brier_score"]),
        CANDIDATE_ORDER.index(str(report["candidate_id"])),
        RIDGE_STRENGTH_ORDER.index(float(report["ridge_strength"])),
    )


def evaluate(root: Path) -> dict[str, Any]:
    rows, source_snapshot = load_development_snapshot(root)
    cluster_latest: dict[str, Any] = {}
    for row in rows:
        cluster_latest[row.dependence_cluster_id] = max(
            cluster_latest.get(row.dependence_cluster_id, row.date), row.date
        )
    cluster_order = [
        cluster_id
        for cluster_id, _ in sorted(
            cluster_latest.items(), key=lambda item: (item[1], item[0])
        )
    ]
    folds = chronological_folds(len(cluster_order))
    fold_reports: list[dict[str, Any]] = []
    aggregate: dict[str, list[dict[str, Any]]] = {}
    for fold in folds:
        train = _fold_rows(rows, cluster_order, fold.train)
        validation = _fold_rows(rows, cluster_order, fold.validation)
        calibration = _fold_rows(rows, cluster_order, fold.calibration)
        test = _fold_rows(rows, cluster_order, fold.test)
        test_start = min((row.date for row in test), default=None)
        if test_start is None:
            raise DevelopmentEvaluationV3Error(f"{fold.fold_id} outer test is empty")
        baseline_logits = pre_event_team_elo_logits(rows, freeze_at=test_start)
        baseline_coefficient = fit_baseline_only(train, baseline_logits)
        validation_baseline_logits = baseline_only_logits(
            validation, baseline_coefficient, baseline_logits
        )
        validation_baseline_metrics = _cluster_metrics(
            validation, _probabilities(validation_baseline_logits, 1.0)
        )
        variant_reports: list[dict[str, Any]] = []
        fitted: dict[str, Any] = {}
        for candidate_id in CANDIDATE_ORDER:
            for ridge_strength in RIDGE_STRENGTH_ORDER:
                fit = fit_penalized(
                    train,
                    candidate_id,
                    ridge_strength,
                    baseline_logits,
                )
                fitted[fit.variant_id] = fit
                validation_context_logits = baseline_adjusted_logits(
                    validation, fit, baseline_logits
                )
                context_metrics = _cluster_metrics(
                    validation, _probabilities(validation_context_logits, 1.0)
                )
                delta = _incremental_delta(
                    context_metrics, validation_baseline_metrics
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
                    "validation_context_with_draft": context_metrics,
                    "validation_context_without_draft": validation_baseline_metrics,
                    "validation_incremental_vs_baseline_only": delta,
                    "equal_strength_composition_index_diagnostic": _cluster_metrics(
                        validation,
                        _probabilities(composition_logits(validation, fit), 1.0),
                    ),
                    "equal_strength_index_is_outcome_calibration_target": False,
                    "selected_for_outer_test": False,
                }
                variant_reports.append(report)
                aggregate.setdefault(fit.variant_id, []).append(
                    {
                        "candidate_id": candidate_id,
                        "ridge_strength": ridge_strength,
                        "delta": delta,
                    }
                )
        eligible = [
            report
            for report in variant_reports
            if report["validation_incremental_vs_baseline_only"]["passed"]
        ]
        if not eligible:
            fold_reports.append(
                {
                    "fold_id": fold.fold_id,
                    "selection": {
                        "status": "baseline_only_no_nonharmful_draft_variant",
                        "variant_id": None,
                        "outer_test_opened": False,
                    },
                    "validation_context_without_draft": validation_baseline_metrics,
                    "variants": variant_reports,
                }
            )
            continue
        selected = min(eligible, key=_variant_order)
        selected["selected_for_outer_test"] = True
        fit = fitted[str(selected["variant_id"])]
        calibration_labels = [row.label_a for row in calibration]
        candidate_method, candidate_scale, candidate_transforms = _calibration(
            baseline_adjusted_logits(calibration, fit, baseline_logits),
            calibration_labels,
        )
        baseline_method, baseline_scale, baseline_transforms = _calibration(
            baseline_only_logits(calibration, baseline_coefficient, baseline_logits),
            calibration_labels,
        )
        candidate_probabilities = _probabilities(
            baseline_adjusted_logits(test, fit, baseline_logits), candidate_scale
        )
        baseline_probabilities = _probabilities(
            baseline_only_logits(test, baseline_coefficient, baseline_logits),
            baseline_scale,
        )
        candidate_metrics = _cluster_metrics(test, candidate_probabilities)
        baseline_metrics = _cluster_metrics(test, baseline_probabilities)
        selected["candidate_calibration_transforms"] = candidate_transforms
        selected["baseline_calibration_transforms"] = baseline_transforms
        selected["locked_outer_test_context_with_draft"] = candidate_metrics
        selected["locked_outer_test_context_without_draft"] = baseline_metrics
        selected["locked_outer_test_incremental_vs_baseline_only"] = (
            _incremental_delta(candidate_metrics, baseline_metrics)
        )
        selected["locked_outer_test_context_with_draft_by_league"] = (
            _league_metrics(test, candidate_probabilities)
        )
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
                    "status": "draft_variant_selected_on_incremental_validation",
                    "variant_id": selected["variant_id"],
                    "candidate_id": selected["candidate_id"],
                    "ridge_strength": selected["ridge_strength"],
                    "criterion": "nonharm_on_both_metrics_then_incremental_log_loss_then_brier_then_frozen_order",
                    "candidate_calibration_transform": candidate_method,
                    "baseline_calibration_transform": baseline_method,
                    "outer_test_locked": True,
                },
                "baseline_state_policy": {
                    "outer_test_frozen_at": test_start.isoformat(),
                    "outer_test_outcomes_update_baseline": False,
                    "candidate_baseline_nuisance_penalized": False,
                    "baseline_comparator_fit_separately_on_same_train_window": True,
                },
                "validation_context_without_draft": validation_baseline_metrics,
                "variants": variant_reports,
            }
        )

    aggregate_reports: list[dict[str, Any]] = []
    for variant_id, records in aggregate.items():
        logloss_deltas = [float(item["delta"]["log_loss"]) for item in records]
        brier_deltas = [float(item["delta"]["brier_score"]) for item in records]
        all_folds_nonharmful = all(bool(item["delta"]["passed"]) for item in records)
        aggregate_reports.append(
            {
                "variant_id": variant_id,
                "candidate_id": records[0]["candidate_id"],
                "ridge_strength": records[0]["ridge_strength"],
                "folds": len(records),
                "mean_validation_log_loss_delta": float(np.mean(logloss_deltas)),
                "mean_validation_brier_delta": float(np.mean(brier_deltas)),
                "all_validation_folds_nonharmful": all_folds_nonharmful,
                "validation_log_loss_deltas": logloss_deltas,
                "validation_brier_deltas": brier_deltas,
            }
        )
    eligible_global = [
        report
        for report in aggregate_reports
        if report["all_validation_folds_nonharmful"]
        and report["mean_validation_log_loss_delta"] <= 0
        and report["mean_validation_brier_delta"] <= 0
    ]
    development_candidate: dict[str, Any] | None = None
    if eligible_global:
        development_candidate = min(
            eligible_global,
            key=lambda report: (
                report["mean_validation_log_loss_delta"],
                report["mean_validation_brier_delta"],
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
            "equal_strength_composition_index": True,
            "outcome_calibrated_neutral_probability": False,
            "causal_draft_effect": False,
            "recommendation": False,
            "betting": False,
            "reliability": False,
        },
        "code_bindings": _code_bindings(),
        "source_snapshot": source_snapshot,
        "estimands": {
            "selection_target": "incremental_predictive_value_of_context_plus_draft_over_same_input_context_only_model",
            "served_neutral_target": "equal_strength_composition_logit_index",
            "neutral_probability_calibration_directly_identified": False,
            "causal_draft_effect_identified": False,
        },
        "objective": {
            "loss": "mean_binary_log_loss",
            "draft_penalty": "0.5_times_ridge_strength_times_squared_l2",
            "candidate_baseline_nuisance_penalized": False,
            "baseline_comparator_fit_separately": True,
            "ridge_strength_order": list(RIDGE_STRENGTH_ORDER),
            "sample_size_invariant": True,
        },
        "baseline_adjustment": {
            **BASELINE_CONFIG,
            "status": "development_nuisance_and_incremental_comparator_only",
            "served_baseline_logit": 0.0,
            "team_identity_in_served_artifact": False,
        },
        "population": {
            "complete_rows": len(rows),
            "dependence_clusters": len(cluster_order),
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
            "candidate_must_be_nonharmful_on_brier_and_log_loss": True,
            "calibration_fit_after_candidate_selection": True,
            "outer_test_scored_only_for_fold_selected_variant": True,
            "candidate_search_opened_on_outer_test": False,
            "future_candidate_selected_from_validation_only": True,
        },
        "candidate_order": list(CANDIDATE_ORDER),
        "ridge_strength_order": list(RIDGE_STRENGTH_ORDER),
        "calibration_order": list(CALIBRATION_ORDER),
        "development_candidate_for_future_freeze": (
            {
                **development_candidate,
                "selection_scope": "incremental_chronological_validation_only",
                "independent_validation": False,
                "authorizes_retrospective_claim": False,
                "authorizes_future_probability": False,
            }
            if development_candidate is not None
            else None
        ),
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
                "promotion": False,
            },
            "league": {
                "status": "development_diagnostic_only",
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
                "reason": "neutral composition terms contain no player identity",
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
        if fold["selection"].get("variant_id") is None:
            selected_tests.append(
                {
                    "fold_id": fold["fold_id"],
                    "status": fold["selection"]["status"],
                    "variant_id": None,
                }
            )
            continue
        selected = next(
            item for item in fold["variants"] if item["selected_for_outer_test"]
        )
        selected_tests.append(
            {
                "fold_id": fold["fold_id"],
                "status": fold["selection"]["status"],
                "variant_id": selected["variant_id"],
                "candidate_id": selected["candidate_id"],
                "ridge_strength": selected["ridge_strength"],
                "candidate_calibration_transform": fold["selection"][
                    "candidate_calibration_transform"
                ],
                "baseline_calibration_transform": fold["selection"][
                    "baseline_calibration_transform"
                ],
                "locked_outer_test_context_with_draft": selected[
                    "locked_outer_test_context_with_draft"
                ],
                "locked_outer_test_context_without_draft": selected[
                    "locked_outer_test_context_without_draft"
                ],
                "locked_outer_test_incremental_vs_baseline_only": selected[
                    "locked_outer_test_incremental_vs_baseline_only"
                ],
            }
        )
    report_raw = (
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "artifact_id": "draft-terminal-development-evaluation-summary-v3",
        "status": report["status"],
        "production_eligible": False,
        "public_probability_authorized": False,
        "run_command": "python3 -W error -m lol_kills.v2.draft.terminal.development_evaluation_v3",
        "run_output_sha256": _sha256(report_raw),
        "code_bindings": report["code_bindings"],
        "source_snapshot": report["source_snapshot"],
        "estimands": report["estimands"],
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
        raise DevelopmentEvaluationV3Error(
            f"refusing to overwrite v3 evaluation evidence: {path}"
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
            raise DevelopmentEvaluationV3Error(
                f"refusing to overwrite v3 evaluation evidence: {path}"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_summary(root: Path, report: Mapping[str, Any]) -> Path:
    path = root / DEFAULT_SUMMARY
    _atomic_write_new(
        path,
        (
            json.dumps(build_summary(report), ensure_ascii=True, indent=2, sort_keys=True)
            + "\n"
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
                    {"summary": str(path), "summary_raw_sha256": _sha256(path.read_bytes())},
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    except (
        OSError,
        DevelopmentEvaluationV3Error,
        DevelopmentSnapshotError,
        ValueError,
    ) as exc:
        print(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

