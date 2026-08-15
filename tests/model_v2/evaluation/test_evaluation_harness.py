"""Integration-style tests for the L2 evaluation harness."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Iterator, Sequence
from pathlib import Path
import json
import unittest

from lol_kills.v2.evaluation import (
    CandidateComparison,
    TransferComparison,
    Decision,
    EvalRow,
    EvaluationRegistry,
    LeakyAdapter,
    MismatchTransformAdapter,
    PrefixRejectAdapter,
    TransferPredictionsAdapter,
    PromotionPlan,
    SnapshotRowsAdapter,
    TerminalRejectAdapter,
    ToyAdapter,
    ValidationFailure,
    build_promotion_report,
    build_synthetic_rows,
    compare_candidate_to_transfer_baselines,
    compare_candidate_to_baseline,
    evaluate_candidate,
    evaluate_candidate_from_snapshot,
    make_model_snapshot,
)
from lol_kills.v2.evaluation import CONTRACT_TREE_SHA256
from lol_kills.v2.evaluation.checks import (
    assert_bootstrap_not_map_level,
    assert_split_partitions_disjoint,
)
from lol_kills.v2.evaluation.pipeline import _as_probability_map, _run_calibration
from lol_kills.v2.evaluation.splitter import (
    SplitConfig,
    SplitPartition,
    SplitPlan,
    build_rolling_origin_plan,
    load_evaluation_registry,
)
from lol_kills.v2.evaluation.types import SealedHoldoutPartition

FROZEN_REGISTRY_PATH = Path("data/lol/v2/evaluation/synthetic-registry-frozen.json")
_FROZEN_REGISTRY_DICT = json.loads(FROZEN_REGISTRY_PATH.read_text(encoding="utf-8"))
SYNTHETIC_SOURCE_TREE_HASH = _FROZEN_REGISTRY_DICT["source_tree_sha256"]


class DuplicateMetricMapping(Mapping[str, float]):
    def __init__(self, name: str, first: float, second: float) -> None:
        self.name = name
        self.first = first
        self.second = second

    def __getitem__(self, key: str) -> float:
        if key != self.name:
            raise KeyError(key)
        return self.first

    def __iter__(self) -> Iterator[str]:
        return iter((self.name,))

    def __len__(self) -> int:
        return 1

    def items(self):
        return ((self.name, self.first), (self.name, self.second))


@dataclass(frozen=True)
class RecordingAdapter(ToyAdapter):
    fit_row_id_calls: list[tuple[str, tuple[str, ...]]] = field(
        default_factory=list,
        compare=False,
    )
    calibration_row_id_calls: list[tuple[str, ...]] = field(
        default_factory=list,
        compare=False,
    )

    def fit(self, rows: Sequence[EvalRow], *, split_name: str):
        self.fit_row_id_calls.append(
            (split_name, tuple(row.row_id for row in rows))
        )
        return super().fit(rows, split_name=split_name)

    def fit_calibration(self, rows, predictions, *, mode="terminal"):
        self.calibration_row_id_calls.append(tuple(row.row_id for row in rows))
        return super().fit_calibration(rows, predictions, mode=mode)


def _make_fixture_rows() -> list[EvalRow]:
    return build_synthetic_rows()


def _load_fixture_registry(path: Path = FROZEN_REGISTRY_PATH) -> EvaluationRegistry:
    return load_evaluation_registry(path)


def _inject_unresolved_test_row(
    registry: EvaluationRegistry,
) -> EvaluationRegistry:
    first_fold = registry.split_plan.folds[0]
    bad_fold = SplitPartition(
        name=first_fold.name,
        train_row_ids=first_fold.train_row_ids,
        validation_row_ids=first_fold.validation_row_ids,
        calibration_row_ids=first_fold.calibration_row_ids,
        test_row_ids=first_fold.test_row_ids + ("row-unresolved-01",),
    )
    return replace(
        registry,
        split_plan=SplitPlan(folds=(bad_fold,), sealed_holdouts=registry.split_plan.sealed_holdouts),
    )


class EvaluationHarnessTests(unittest.TestCase):
    """Exercise the L2 contract gates in deterministic synthetic settings."""

    SOURCE_TREE_HASH = SYNTHETIC_SOURCE_TREE_HASH

    def setUp(self) -> None:
        self.rows = _make_fixture_rows()
        self.registry = _load_fixture_registry()
        self.snapshot = make_model_snapshot(self.rows)

    def _promotion_probe(
        self,
        *,
        registry_sha256: str | None = None,
        candidate_registry_sha256: str | None = None,
        plan_registry_sha256: str | None = None,
        contract_tree_sha256: str = CONTRACT_TREE_SHA256,
        margins: Mapping[str, float] | None = None,
        directions: Mapping[str, bool] | None = None,
        candidate_metrics: Mapping[str, float] | None = None,
        baseline_metrics: Mapping[str, float] | None = None,
        hard_gates: Mapping[str, bool] | None = None,
    ):
        frozen_hash = self.registry.sha256()
        margins = {"log_loss": 0.02} if margins is None else margins
        directions = (
            {name: False for name in margins}
            if directions is None
            else directions
        )
        candidate_metrics = (
            {name: 0.50 for name in margins}
            if candidate_metrics is None
            else candidate_metrics
        )
        baseline_metrics = (
            {name: 0.50 for name in margins}
            if baseline_metrics is None
            else baseline_metrics
        )
        hard_gates = {"registry_frozen": True} if hard_gates is None else hard_gates
        return build_promotion_report(
            model_id="promotion-probe",
            model_version="1",
            registry_sha256=(
                frozen_hash if registry_sha256 is None else registry_sha256
            ),
            candidate_registry_sha256=(
                frozen_hash
                if candidate_registry_sha256 is None
                else candidate_registry_sha256
            ),
            planned=PromotionPlan(
                contract_tree_sha256=contract_tree_sha256,
                split_registry_sha256=(
                    frozen_hash
                    if plan_registry_sha256 is None
                    else plan_registry_sha256
                ),
                metric_noninferiority_margins=margins,
                higher_is_better=directions,
            ),
            candidate_metrics=candidate_metrics,
            baseline_metrics=baseline_metrics,
            hard_gates=hard_gates,
        )

    def test_happy_path_evaluation_runs_and_reports_holdouts(self) -> None:
        report = evaluate_candidate(
            ToyAdapter(adapter_id="toy-l2-happy", source_tree_sha256=self.SOURCE_TREE_HASH),
            self.rows,
            self.registry,
            request_prefixes=("slot_1", "slot_2"),
            sealed_rows_snapshot=self.snapshot,
        )

        self.assertEqual(report.adapter_id, "toy-l2-happy")
        self.assertTrue(all(report.hard_gate_results.values()))
        self.assertIn("temporal", report.holdout_reports)
        self.assertIn("future_patch", report.holdout_reports)
        self.assertTrue(
            any(name.startswith("league_out_") for name in report.holdout_reports),
        )
        self.assertIn("masked_champion_residual", report.holdout_reports)
        self.assertIn("archetype_transfer_true_new_or_zero_play", report.holdout_reports)
        self.assertTrue(report.test_predictions)
        for fold in report.folds:
            self.assertGreater(fold.validation_rows, 0)
            self.assertGreater(fold.validation_raw_metrics.log_loss, 0.0)
            self.assertGreater(fold.validation_calibrated_metrics.log_loss, 0.0)

        # Registry fingerprinting is deterministic for this synthetic fixture.
        self.assertEqual(report.registry_hash, self.registry.sha256())

    def test_compare_candidate_to_baseline_and_promotion_plan(self) -> None:
        baseline_report = evaluate_candidate(
            ToyAdapter(adapter_id="baseline-terminal-v2", adapter_version="1", source_tree_sha256=self.SOURCE_TREE_HASH),
            self.rows,
            self.registry,
            request_prefixes=("slot_1",),
            sealed_rows_snapshot=self.snapshot,
        )
        candidate_report = evaluate_candidate(
            ToyAdapter(adapter_id="candidate-terminal-v2", adapter_version="2", source_tree_sha256=self.SOURCE_TREE_HASH),
            self.rows,
            self.registry,
            request_prefixes=("slot_1",),
            sealed_rows_snapshot=self.snapshot,
        )

        comparison = compare_candidate_to_baseline(
            candidate_report,
            baseline_report,
            self.rows,
        )
        self.assertIsInstance(comparison, CandidateComparison)
        self.assertEqual(comparison.candidate_adapter_id, "candidate-terminal-v2")
        self.assertEqual(comparison.baseline_adapter_id, "baseline-terminal-v2")
        self.assertGreater(comparison.shared_rows, 0)

        promotion_metric_names = ("log_loss", "brier", "ece")
        promotion = build_promotion_report(
            model_id="candidate-terminal-v2",
            model_version="2",
            registry_sha256=self.registry.sha256(),
            candidate_registry_sha256=self.registry.sha256(),
            planned=PromotionPlan(
                contract_tree_sha256=CONTRACT_TREE_SHA256,
                split_registry_sha256=self.registry.sha256(),
                metric_noninferiority_margins={"log_loss": 0.02, "brier": 0.02, "ece": 0.05},
                higher_is_better={"log_loss": False, "brier": False, "ece": False},
            ),
            candidate_metrics={
                name: candidate_report.aggregate_calibrated_metrics[name]
                for name in promotion_metric_names
            },
            baseline_metrics={
                name: baseline_report.aggregate_calibrated_metrics[name]
                for name in promotion_metric_names
            },
            hard_gates=candidate_report.hard_gate_results,
        )
        self.assertEqual(promotion.decision, Decision.BLOCK)

    def test_promotion_blocks_wrong_candidate_or_plan_registry(self) -> None:
        for kwargs in (
            {"candidate_registry_sha256": "1" * 64},
            {"plan_registry_sha256": "2" * 64},
            {"registry_sha256": "malformed"},
        ):
            with self.subTest(kwargs=kwargs):
                self.assertEqual(
                    self._promotion_probe(**kwargs).decision,
                    Decision.BLOCK,
                )

    def test_promotion_blocks_wrong_plan_contract(self) -> None:
        report = self._promotion_probe(contract_tree_sha256="0" * 64)
        self.assertEqual(report.decision, Decision.BLOCK)

    def test_promotion_blocks_negative_infinity_candidate_metric(self) -> None:
        report = self._promotion_probe(
            candidate_metrics={"log_loss": float("-inf")},
        )
        self.assertEqual(report.decision, Decision.BLOCK)

    def test_promotion_blocks_empty_hard_gates(self) -> None:
        report = self._promotion_probe(hard_gates={})
        self.assertEqual(report.decision, Decision.BLOCK)

    def test_promotion_blocks_empty_metric_requirements(self) -> None:
        report = self._promotion_probe(
            margins={},
            directions={},
            candidate_metrics={},
            baseline_metrics={},
        )
        self.assertEqual(report.decision, Decision.BLOCK)

    def test_promotion_blocks_missing_or_extra_metric_evidence(self) -> None:
        margins = {"log_loss": 0.02, "brier": 0.02}
        directions = {"log_loss": False, "brier": False}
        variants = (
            {"candidate_metrics": {"log_loss": 0.5}},
            {
                "candidate_metrics": {
                    "log_loss": 0.5,
                    "brier": 0.2,
                    "extra": 0.1,
                }
            },
            {"baseline_metrics": {"log_loss": 0.5}},
        )
        for evidence in variants:
            with self.subTest(evidence=evidence):
                report = self._promotion_probe(
                    margins=margins,
                    directions=directions,
                    **evidence,
                )
                self.assertEqual(report.decision, Decision.BLOCK)

    def test_promotion_blocks_duplicate_metric_evidence(self) -> None:
        report = self._promotion_probe(
            candidate_metrics=DuplicateMetricMapping("log_loss", 0.50, 0.49),
        )
        self.assertEqual(report.decision, Decision.BLOCK)

    def test_promotion_blocks_nan_or_infinite_metric_values(self) -> None:
        variants: tuple[dict[str, Mapping[str, float]], ...] = (
            {"candidate_metrics": {"log_loss": float("nan")}},
            {"candidate_metrics": {"log_loss": float("inf")}},
            {"baseline_metrics": {"log_loss": float("nan")}},
            {"baseline_metrics": {"log_loss": float("-inf")}},
        )
        for evidence in variants:
            with self.subTest(evidence=evidence):
                self.assertEqual(
                    self._promotion_probe(**evidence).decision,
                    Decision.BLOCK,
                )

    def test_promotion_blocks_negative_or_nonfinite_margin(self) -> None:
        for margin in (-0.01, float("nan"), float("inf"), float("-inf")):
            with self.subTest(margin=margin):
                report = self._promotion_probe(
                    margins={"log_loss": margin},
                )
                self.assertEqual(report.decision, Decision.BLOCK)

    def test_rolling_origin_folds_are_blocked_and_chronological(self) -> None:
        rows = _make_fixture_rows()
        split_plan = build_rolling_origin_plan(rows, config=SplitConfig(development_folds=2))
        self.assertEqual(len(split_plan.folds), 2)

        row_by_id = {row.row_id: row for row in rows}
        test_windows: list[tuple[int, int, str]] = []
        seen_test: set[str] = set()
        for fold in split_plan.folds:
            self.assertTrue(fold.validation_row_ids)
            self.assertTrue(fold.calibration_row_ids)
            self.assertTrue(fold.test_row_ids)
            self.assertGreater(len(fold.train_row_ids), 0)

            # Ensure partitions are disjoint and chronological per fold.
            train_ids = set(fold.train_row_ids)
            validation_ids = set(fold.validation_row_ids)
            calibration_ids = set(fold.calibration_row_ids)
            test_ids = set(fold.test_row_ids)
            self.assertFalse(train_ids & validation_ids)
            self.assertFalse(train_ids & calibration_ids)
            self.assertFalse(train_ids & test_ids)
            self.assertFalse(validation_ids & calibration_ids)
            self.assertFalse(validation_ids & test_ids)
            self.assertFalse(calibration_ids & test_ids)

            fold_train_end = max(row_by_id[row_id].event_start.timestamp() for row_id in fold.train_row_ids)
            fold_validation_start = min(row_by_id[row_id].event_start.timestamp() for row_id in fold.validation_row_ids)
            fold_validation_end = max(row_by_id[row_id].event_start.timestamp() for row_id in fold.validation_row_ids)
            fold_calibration_start = min(row_by_id[row_id].event_start.timestamp() for row_id in fold.calibration_row_ids)
            fold_calibration_end = max(row_by_id[row_id].event_start.timestamp() for row_id in fold.calibration_row_ids)
            fold_test_start = min(row_by_id[row_id].event_start.timestamp() for row_id in fold.test_row_ids)

            self.assertLess(fold_train_end, fold_validation_start)
            self.assertLess(fold_validation_end, fold_calibration_start)
            self.assertLess(fold_calibration_end, fold_test_start)

            fold_test_end = max(row_by_id[row_id].event_start.timestamp() for row_id in fold.test_row_ids)
            test_windows.append((fold_test_start, fold_test_end, fold.name))

            for row_id in fold.test_row_ids:
                self.assertNotIn(row_id, seen_test)
                seen_test.add(row_id)

        test_windows.sort(key=lambda item: item[0])
        for previous, current in zip(test_windows, test_windows[1:]):
            self.assertLess(previous[1], current[0])

        final_test_end = max(
            row_by_id[row_id].event_start
            for row_id in split_plan.folds[-1].test_row_ids
        )
        development_ids = {
            row_id for fold in split_plan.folds for row_id in fold.all_ids
        }
        self.assertEqual(
            final_test_end,
            max(row_by_id[row_id].event_start for row_id in development_ids),
        )

        multi_map_partitions = []
        for fold in split_plan.folds:
            for partition in (
                fold.train_row_ids,
                fold.validation_row_ids,
                fold.calibration_row_ids,
                fold.test_row_ids,
            ):
                if "row-04" in partition or "row-04-map-2" in partition:
                    multi_map_partitions.append(set(partition))
        self.assertTrue(multi_map_partitions)
        self.assertTrue(
            all({"row-04", "row-04-map-2"}.issubset(ids) for ids in multi_map_partitions)
        )

        self.assertIn("future_patch", {holdout.name for holdout in self.registry.split_plan.sealed_holdouts})

    def test_splitter_rejects_mixed_resolution_within_series(self) -> None:
        rows = [
            replace(row, series_resolved=False)
            if row.row_id == "row-04-map-2"
            else row
            for row in self.rows
        ]
        with self.assertRaises(ValueError):
            build_rolling_origin_plan(rows, config=SplitConfig(development_folds=2))

    def test_splitter_rejects_overlapping_series_intervals(self) -> None:
        row_by_id = {row.row_id: row for row in self.rows}
        overlap_start = row_by_id["row-04"].event_start.replace(minute=30)
        rows = [
            replace(row, event_start=overlap_start)
            if row.row_id == "row-05"
            else row
            for row in self.rows
        ]
        with self.assertRaises(ValueError):
            build_rolling_origin_plan(rows, config=SplitConfig(development_folds=2))

    def test_registry_rejects_missing_development_assignment(self) -> None:
        folds = tuple(
            replace(
                fold,
                train_row_ids=tuple(row_id for row_id in fold.train_row_ids if row_id != "row-00"),
                validation_row_ids=tuple(row_id for row_id in fold.validation_row_ids if row_id != "row-00"),
                calibration_row_ids=tuple(row_id for row_id in fold.calibration_row_ids if row_id != "row-00"),
                test_row_ids=tuple(row_id for row_id in fold.test_row_ids if row_id != "row-00"),
            )
            for fold in self.registry.split_plan.folds
        )
        bad_registry = replace(
            self.registry,
            split_plan=replace(self.registry.split_plan, folds=folds),
        )
        with self.assertRaises(ValidationFailure):
            assert_split_partitions_disjoint(
                bad_registry,
                rows_by_id={row.row_id: row for row in self.rows},
            )

    def test_registry_rejects_swapped_validation_and_calibration(self) -> None:
        first, second = self.registry.split_plan.folds
        swapped = replace(
            first,
            validation_row_ids=first.calibration_row_ids,
            calibration_row_ids=first.validation_row_ids,
        )
        bad_registry = replace(
            self.registry,
            split_plan=replace(self.registry.split_plan, folds=(swapped, second)),
        )
        with self.assertRaises(ValidationFailure):
            assert_split_partitions_disjoint(
                bad_registry,
                rows_by_id={row.row_id: row for row in self.rows},
            )

    def test_registry_preserves_declared_fold_order(self) -> None:
        bad_registry = replace(
            self.registry,
            split_plan=replace(
                self.registry.split_plan,
                folds=tuple(reversed(self.registry.split_plan.folds)),
            ),
        )
        with self.assertRaises(ValidationFailure):
            assert_split_partitions_disjoint(
                bad_registry,
                rows_by_id={row.row_id: row for row in self.rows},
            )

    def test_registry_rejects_unresolved_sealed_row(self) -> None:
        temporal = next(
            holdout
            for holdout in self.registry.split_plan.sealed_holdouts
            if holdout.name == "temporal"
        )
        bad_temporal = replace(
            temporal,
            row_ids=temporal.row_ids + ("row-unresolved-01",),
        )
        holdouts = tuple(
            bad_temporal if holdout.name == "temporal" else holdout
            for holdout in self.registry.split_plan.sealed_holdouts
        )
        bad_registry = replace(
            self.registry,
            split_plan=replace(self.registry.split_plan, sealed_holdouts=holdouts),
        )
        with self.assertRaises(ValidationFailure):
            assert_split_partitions_disjoint(
                bad_registry,
                rows_by_id={row.row_id: row for row in self.rows},
            )

    def test_league_holdouts_require_exact_complete_unique_coverage(self) -> None:
        rows_by_id = {row.row_id: row for row in self.rows}
        lcs = next(
            holdout
            for holdout in self.registry.split_plan.sealed_holdouts
            if holdout.name == "league_out_lcs"
        )

        variants = []
        variants.append(
            tuple(
                replace(holdout, row_ids=holdout.row_ids[:-1])
                if holdout.name == "league_out_lcs"
                else holdout
                for holdout in self.registry.split_plan.sealed_holdouts
            )
        )
        variants.append(
            tuple(
                holdout
                for holdout in self.registry.split_plan.sealed_holdouts
                if holdout.name != "league_out_lcs"
            )
        )
        variants.append(self.registry.split_plan.sealed_holdouts + (lcs,))
        variants.append(
            self.registry.split_plan.sealed_holdouts
            + (
                SealedHoldoutPartition(
                    name="league_out_all",
                    row_ids=lcs.row_ids,
                    metadata={
                        "protocol": "league_leave_one_out",
                        "league_id": "all",
                    },
                ),
            )
        )

        for holdouts in variants:
            with self.subTest(holdout_count=len(holdouts)):
                bad_registry = replace(
                    self.registry,
                    split_plan=replace(
                        self.registry.split_plan,
                        sealed_holdouts=holdouts,
                    ),
                )
                with self.assertRaises(ValidationFailure):
                    assert_split_partitions_disjoint(
                        bad_registry,
                        rows_by_id=rows_by_id,
                    )

    def test_sealed_execution_uses_only_registered_fit_and_calibration_ids(self) -> None:
        adapter = RecordingAdapter(source_tree_sha256=self.SOURCE_TREE_HASH)
        report = evaluate_candidate(
            adapter,
            self.rows,
            self.registry,
            sealed_rows_snapshot=self.snapshot,
        )
        rows_by_id = {row.row_id: row for row in self.rows}
        development_ids = {
            row_id
            for fold in self.registry.split_plan.folds
            for row_id in fold.all_ids
        }
        registered_calibration_ids = {
            row_id
            for fold in self.registry.split_plan.folds
            for row_id in fold.calibration_row_ids
        }
        temporal_ids = {
            row_id
            for holdout in self.registry.split_plan.sealed_holdouts
            if holdout.metadata.get("protocol") in {"temporal", "future_patch"}
            for row_id in holdout.row_ids
        }

        for holdout in self.registry.split_plan.sealed_holdouts:
            holdout_report = report.holdout_reports[holdout.name]
            if holdout_report["status"] == "unavailable":
                continue
            fit_ids = set(holdout_report["fit_row_ids"])
            calibration_ids = set(holdout_report["calibration_row_ids"])
            scored_ids = tuple(holdout_report["scored_row_ids"])
            self.assertTrue(fit_ids.issubset(development_ids))
            self.assertTrue(calibration_ids.issubset(registered_calibration_ids))
            self.assertFalse(fit_ids & registered_calibration_ids)
            self.assertFalse(fit_ids & calibration_ids)
            self.assertFalse((fit_ids | calibration_ids) & temporal_ids)
            self.assertFalse((fit_ids | calibration_ids) & set(holdout.row_ids))
            self.assertTrue(all(rows_by_id[row_id].series_resolved for row_id in fit_ids | calibration_ids))
            self.assertEqual(scored_ids, holdout.row_ids)

        available_reports = [
            (name, holdout_report)
            for name, holdout_report in report.holdout_reports.items()
            if holdout_report["status"] != "unavailable"
        ]
        sealed_fit_calls = adapter.fit_row_id_calls[len(self.registry.split_plan.folds):]
        self.assertEqual(
            sealed_fit_calls,
            [
                (
                    f"holdout-{name}",
                    tuple(holdout_report["fit_row_ids"]),
                )
                for name, holdout_report in available_reports
            ],
        )
        fold_calibration_call_count = len(self.registry.split_plan.folds) * 2
        sealed_calibration_calls = adapter.calibration_row_id_calls[
            fold_calibration_call_count:
        ]
        self.assertEqual(
            sealed_calibration_calls,
            [
                tuple(holdout_report["calibration_row_ids"])
                for _, holdout_report in available_reports
            ],
        )

    def test_league_exclusion_applies_to_refit_and_calibration(self) -> None:
        report = evaluate_candidate(
            ToyAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
            self.rows,
            self.registry,
        )
        rows_by_id = {row.row_id: row for row in self.rows}
        for league_id in ("lcs", "lec", "lpl"):
            holdout_report = report.holdout_reports[f"league_out_{league_id}"]
            self.assertEqual(holdout_report["status"], "ok")
            support_ids = (
                tuple(holdout_report["fit_row_ids"])
                + tuple(holdout_report["calibration_row_ids"])
            )
            self.assertTrue(
                all(rows_by_id[row_id].league_id != league_id for row_id in support_ids)
            )

    def test_synthetic_and_placeholder_registries_have_explicit_status(self) -> None:
        frozen = json.loads((Path("data/lol/v2/evaluation/synthetic-registry-frozen.json")).read_text(encoding="utf-8"))
        placeholder = json.loads((Path("data/lol/v2/evaluation/synthetic-registry-placeholder.json")).read_text(encoding="utf-8"))

        self.assertTrue(frozen["is_synthetic_registry"])
        self.assertIn("is_synthetic_placeholder", frozen)
        self.assertFalse(frozen["is_synthetic_placeholder"])

        self.assertFalse(placeholder["is_synthetic_registry"])
        self.assertIn("is_synthetic_placeholder", placeholder)
        self.assertTrue(placeholder["is_synthetic_placeholder"])
        self.assertIn("placeholder_detected", placeholder["invalidation_reasons"])
        self.assertEqual(placeholder["source_tree_sha256"], "0" * 64)

    def test_probability_maps_reject_mismatched_row_set(self) -> None:
        row_pairs = _make_fixture_rows()[:3]
        with self.assertRaises(ValidationFailure):
            _as_probability_map({"row-00": 0.4}, row_pairs, field_name="stub")
        with self.assertRaises(ValidationFailure):
            _as_probability_map(
                {
                    "row-00": 0.2,
                    "row-01": 0.3,
                    "row-does-not-exist": 0.4,
                },
                row_pairs,
                field_name="stub",
            )
        with self.assertRaises(ValidationFailure):
            _as_probability_map({"row-00": 1.2, "row-01": 0.3, "row-02": 0.4}, row_pairs, field_name="stub")
        with self.assertRaises(ValidationFailure):
            _as_probability_map(
                {"row-00": float("nan"), "row-01": 0.3, "row-02": 0.4},
                row_pairs,
                field_name="stub",
            )

    def test_run_calibration_rejects_prediction_row_misalignment(self) -> None:
        rows = _make_fixture_rows()[:4]
        adapter = ToyAdapter(adapter_id="calibration-guard")
        fit_state = adapter.fit(rows, split_name="calibration-guard")

        calibration_rows = tuple(rows[:2])
        calibration_predictions = tuple(adapter.predict(fit_state, calibration_rows, mode="terminal"))
        test_rows = tuple(rows[2:4])
        test_predictions = tuple(adapter.predict(fit_state, test_rows, mode="terminal"))

        reversed_calibration = tuple(reversed(calibration_predictions))
        with self.assertRaises(ValidationFailure):
            _run_calibration(
                adapter,
                fit_state,
                calibration_rows,
                reversed_calibration,
                test_predictions,
            )

        duplicate_calibration = (calibration_predictions[0], calibration_predictions[0])
        with self.assertRaises(ValidationFailure):
            _run_calibration(
                adapter,
                fit_state,
                calibration_rows,
                duplicate_calibration,
                test_predictions,
            )

    def test_compare_candidate_to_transfer_baselines(self) -> None:
        candidate_report = evaluate_candidate(
            ToyAdapter(adapter_id="candidate-transfer-v2", adapter_version="3", source_tree_sha256=self.SOURCE_TREE_HASH),
            self.rows,
            self.registry,
            request_prefixes=("slot_1",),
            sealed_rows_snapshot=self.snapshot,
        )

        transfer_probs = {
            row_id: max(0.05, min(0.95, candidate_report.test_predictions[row_id].final_probability() - 0.02))
            for row_id in candidate_report.test_predictions
        }
        transfer_ablation_probs = {
            row_id: min(0.95, candidate_report.test_predictions[row_id].final_probability() + 0.02)
            for row_id in candidate_report.test_predictions
        }
        comparison = compare_candidate_to_transfer_baselines(
            candidate_report,
            TransferPredictionsAdapter(
                adapter_id="transfer-baseline-v1",
                ontology_free_probabilities=transfer_probs,
                transfer_ablation_probabilities=transfer_ablation_probs,
            ),
            self.rows,
        )

        self.assertIsInstance(comparison, TransferComparison)
        self.assertEqual(comparison.candidate_adapter_id, "candidate-transfer-v2")
        self.assertEqual(comparison.transfer_adapter_id, "transfer-baseline-v1")
        self.assertGreater(comparison.shared_rows, 0)

    def test_transfer_comparison_rejects_incoherent_probability_maps(self) -> None:
        candidate_report = evaluate_candidate(
            ToyAdapter(adapter_id="candidate-transfer-v2", adapter_version="4", source_tree_sha256=self.SOURCE_TREE_HASH),
            self.rows,
            self.registry,
            request_prefixes=("slot_1",),
            sealed_rows_snapshot=self.snapshot,
        )

        missing_row_id = "row-does-not-exist"
        with self.assertRaises(ValidationFailure):
            compare_candidate_to_transfer_baselines(
                candidate_report,
                TransferPredictionsAdapter(
                    ontology_free_probabilities={missing_row_id: 0.4},
                    transfer_ablation_probabilities={missing_row_id: 0.5},
                ),
                self.rows,
            )

    def test_transfer_comparison_rejects_out_of_range_probabilities(self) -> None:
        candidate_report = evaluate_candidate(
            ToyAdapter(adapter_id="candidate-transfer-v2", adapter_version="5", source_tree_sha256=self.SOURCE_TREE_HASH),
            self.rows,
            self.registry,
            request_prefixes=("slot_1",),
            sealed_rows_snapshot=self.snapshot,
        )

        row_id = next(iter(candidate_report.test_predictions.keys()))
        with self.assertRaises(ValidationFailure):
            compare_candidate_to_transfer_baselines(
                candidate_report,
                TransferPredictionsAdapter(
                    ontology_free_probabilities={row_id: 1.25},
                    transfer_ablation_probabilities={row_id: 0.5},
                ),
                self.rows,
            )

    def test_modifying_split_payload_hash_rejects_run(self) -> None:
        first_fold = self.registry.split_plan.folds[0]
        changed_fold = SplitPartition(
            name=first_fold.name,
            train_row_ids=first_fold.train_row_ids,
            validation_row_ids=first_fold.validation_row_ids,
            calibration_row_ids=first_fold.calibration_row_ids[:-1],
            test_row_ids=first_fold.test_row_ids,
        )
        changed_registry = replace(
            self.registry,
            split_plan=SplitPlan(
                folds=(changed_fold,),
                sealed_holdouts=self.registry.split_plan.sealed_holdouts,
            ),
        )
        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                ToyAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                self.rows,
                changed_registry,
                request_prefixes=("slot_1",),
            )

    def test_leaky_adapter_is_rejected(self) -> None:
        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                LeakyAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                self.rows,
                self.registry,
            )

    def test_mismatched_transform_is_rejected(self) -> None:
        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                MismatchTransformAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                self.rows,
                self.registry,
            )

    def test_unapproved_terminal_probability_wording_is_rejected(self) -> None:
        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                TerminalRejectAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                self.rows,
                self.registry,
            )

    def test_unapproved_prefix_probability_wording_is_rejected(self) -> None:
        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                PrefixRejectAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                self.rows,
                self.registry,
                request_prefixes=("slot_1",),
            )

    def test_fake_holdout_label_tamper_fails_snapshot_gate(self) -> None:
        tampered_rows = [
            replace(row, label=(0 if row.label else 1)) if row.row_id == "row-00" else row
            for row in self.rows
        ]
        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                ToyAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                tampered_rows,
                self.registry,
                sealed_rows_snapshot=self.snapshot,
            )

    def test_future_feature_join_is_rejected(self) -> None:
        first = self.rows[0]
        bad_features = dict(first.feature_available_at)
        bad_features["feature_core"] = first.event_start
        bad_row = replace(first, feature_available_at=bad_features)
        bad_rows = [bad_row] + self.rows[1:]

        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                ToyAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                bad_rows,
                self.registry,
            )

    def test_map_independent_bootstrap_rejects_row_level_units(self) -> None:
        first_two = [row for row in self.rows if row.series_resolved][:2]
        with self.assertRaises(ValidationFailure):
            assert_bootstrap_not_map_level(
                cluster_ids=[row.row_id for row in first_two],
                row_ids=[row.row_id for row in first_two],
                rows_by_id={row.row_id: row for row in first_two},
            )

    def test_unresolved_series_are_not_allowed_in_primary_bootstrap(self) -> None:
        unresolved_registry = _inject_unresolved_test_row(self.registry)
        with self.assertRaises(ValidationFailure):
            evaluate_candidate(
                ToyAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                self.rows,
                unresolved_registry,
            )

    def test_snapshot_adapter_rejects_source_tree_mismatch(self) -> None:
        snapshot = SnapshotRowsAdapter(
            rows_payload=tuple(self.rows),
            source_tree_sha256="b" * 64,
            snapshot_id="snapshot-mismatch-v2",
        )
        with self.assertRaises(ValidationFailure):
            evaluate_candidate_from_snapshot(
                ToyAdapter(source_tree_sha256=self.SOURCE_TREE_HASH),
                snapshot,
                self.registry,
            )

    def test_frozen_fixture_registry_can_be_loaded_from_disk(self) -> None:
        fixture_path = Path("data/lol/v2/evaluation/synthetic-registry-frozen.json")
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        self.assertEqual(fixture["contract_tree_sha256"], CONTRACT_TREE_SHA256)
        self.assertEqual(fixture["split_plan_id"], self.registry.split_plan_id)
        self.assertEqual(fixture["source_tree_sha256"], self.SOURCE_TREE_HASH)
        self.assertEqual(fixture["registry_hash"], self.registry.sha256())


class BootstrapAndRegistryFixtureSanityTests(unittest.TestCase):
    def test_frozen_and_placeholder_fixture_payloads_are_distinguishable(self) -> None:
        frozen = json.loads(Path("data/lol/v2/evaluation/synthetic-registry-frozen.json").read_text(encoding="utf-8"))
        placeholder = json.loads(Path("data/lol/v2/evaluation/synthetic-registry-placeholder.json").read_text(encoding="utf-8"))

        self.assertNotEqual(frozen["registry_hash"], placeholder["registry_hash"])
        self.assertNotEqual(frozen["source_tree_sha256"], placeholder["source_tree_sha256"])

    def test_registry_preregisters_transfer_and_masked_residual_holdouts(self) -> None:
        registry = _load_fixture_registry()
        holdout_lookup = {holdout.name: holdout.row_ids for holdout in registry.split_plan.sealed_holdouts}
        rows_by_id = {row.row_id: row for row in _make_fixture_rows()}

        self.assertIn("masked_champion_residual", holdout_lookup)
        self.assertIn("archetype_transfer_true_new_or_zero_play", holdout_lookup)
        self.assertGreater(len(holdout_lookup["masked_champion_residual"]), 0)
        self.assertGreater(len(holdout_lookup["archetype_transfer_true_new_or_zero_play"]), 0)

        expected_masked = sorted(
            row_id
            for row_id, row in rows_by_id.items()
            if row.metadata.get("masked_champion_residual")
        )
        expected_transfer = sorted(
            row_id
            for row_id, row in rows_by_id.items()
            if row.metadata.get("true_new_champion") or row.metadata.get("archetype_transfer")
        )
        self.assertEqual(holdout_lookup["masked_champion_residual"], tuple(expected_masked))
        self.assertEqual(
            holdout_lookup["archetype_transfer_true_new_or_zero_play"],
            tuple(expected_transfer),
        )
