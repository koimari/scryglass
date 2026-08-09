"""PASS-B1 sealed-decision and five-output executable contract probes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import lol_kills.v2.evaluation.sealed as sealed_module
from lol_kills.v2.evaluation import (
    AtomicSealedLedger,
    CONTRACT_TREE_SHA256,
    Decision,
    INVARIANT_DISPATCH,
    PromotionPlan,
    REQUIRED_B1_SEALED_HARD_GATES,
    SEALED_STAGE_NAMES,
    SealedDecisionReceipt,
    SealedStage,
    ToyAdapter,
    ValidationFailure,
    build_promotion_report,
    build_synthetic_rows,
    create_sealed_decision_plan,
    create_sealed_decision_request,
    evaluate_candidate,
    execute_sealed_decision,
    make_frozen_evaluation_snapshot,
    make_model_snapshot,
    run_five_output_validation,
    validate_output_payload,
    verify_five_output_validation_report,
    verify_sealed_decision_receipt,
    write_sealed_decision_plan,
)
from lol_kills.v2.evaluation.fixtures import (
    SYNTHETIC_BASELINE_ARTIFACT_SHA256,
    SYNTHETIC_CANDIDATE_ARTIFACT_SHA256,
    SYNTHETIC_TRANSFORM_SHA256,
)
from lol_kills.v2.evaluation.splitter import (
    load_evaluation_registry,
    registry_to_disk,
)
from lol_kills.v2.evaluation.types import canonical_sha256


DATA_ROOT = Path("data/lol/v2/evaluation")
FROZEN_REGISTRY = DATA_ROOT / "synthetic-registry-frozen.json"
CONTRACT_ROOT = Path("docs/model-v2/contracts")


def _claim_in_process(
    ledger_path: str,
    request,
    signing_seed: bytes,
    registrar_id: str,
    queue,
) -> None:
    ledger = AtomicSealedLedger(
        ledger_path,
        signing_seed=signing_seed,
        registrar_identity=registrar_id,
        registrar_kind="test_only",
    )
    try:
        ledger.claim(request)
        queue.put("ok")
    except ValidationFailure:
        queue.put("blocked")


class FiveOutputContractValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = run_five_output_validation()

    def test_executes_every_declared_invariant_and_mutation(self) -> None:
        self.assertEqual(self.report.invariant_pass_count, 62)
        self.assertEqual(self.report.mutation_pass_count, 42)
        self.assertEqual(self.report.structural_pass_count, 6)
        self.assertEqual(set(self.report.invariant_ids), set(INVARIANT_DISPATCH))
        self.assertEqual(len(self.report.mutation_ids), 42)
        by_id = {item.evidence_id: item for item in self.report.evidence}
        for invariant_id in self.report.invariant_ids:
            with self.subTest(invariant_id=invariant_id):
                positive_id = f"invariant:{invariant_id}:positive"
                counterexample_id = f"invariant:{invariant_id}:counterexample"
                self.assertIn(positive_id, by_id)
                self.assertIn(counterexample_id, by_id)
                self.assertEqual(by_id[positive_id].status, "passed")
                self.assertEqual(by_id[counterexample_id].status, "passed")
                self.assertTrue(by_id[positive_id].applicable)
                self.assertTrue(by_id[counterexample_id].applicable)
                self.assertEqual(
                    self.report.invariant_counts[invariant_id],
                    {
                        "applicable": 2,
                        "passed": 2,
                        "not_applicable": 0,
                    },
                )
        for mutation_id in self.report.mutation_ids:
            with self.subTest(mutation_id=mutation_id):
                self.assertIn(mutation_id, by_id)
                self.assertEqual(by_id[mutation_id].status, "passed")
        self.assertTrue(self.report.all_pass)
        self.assertTrue(self.report.verify_hash())
        frozen_report = json.loads(
            (DATA_ROOT / "five-output-validation-report.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            frozen_report["report_sha256"],
            self.report.report_sha256,
        )

    def test_missing_invariant_dispatch_fails_closed(self) -> None:
        dispatch = dict(INVARIANT_DISPATCH)
        dispatch.pop(next(iter(dispatch)))
        with self.assertRaises(ValidationFailure):
            run_five_output_validation(invariant_dispatch=dispatch)

    def test_false_freshness_and_input_conflict_fail(self) -> None:
        document = json.loads(
            (CONTRACT_ROOT / "examples/draft-score.example.json").read_text(
                encoding="utf-8"
            )
        )
        stale = deepcopy(document)
        stale["provenance"]["freshness_checks"][0]["fresh"] = False
        with self.assertRaises(ValidationFailure):
            validate_output_payload("draft_score", stale)

        conflicted = deepcopy(document)
        conflicted["provenance"]["input_conflicts"] = [
            {
                "input_id": "scryglass:source:analysis-input",
                "field": "patch_id",
                "values": ["26.13", "26.14"],
                "source_ids": [
                    "scryglass:source:a",
                    "scryglass:source:b",
                ],
                "detected_at": "2026-07-27T17:59:00Z",
            }
        ]
        with self.assertRaises(ValidationFailure):
            validate_output_payload("draft_score", conflicted)

    def test_structural_and_semantic_contradictions_fail(self) -> None:
        player = json.loads(
            (CONTRACT_ROOT / "examples/player-rating.example.json").read_text(
                encoding="utf-8"
            )
        )
        player["rating"] = player["posterior_mean"]
        with self.assertRaises(ValidationFailure):
            validate_output_payload("player_rating", player)

        draft = json.loads(
            (CONTRACT_ROOT / "examples/draft-score.example.json").read_text(
                encoding="utf-8"
            )
        )
        draft["score_b"] = 50.0
        with self.assertRaises(ValidationFailure):
            validate_output_payload("draft_score", draft)

    def test_schema_and_example_drift_fail_against_anchors(self) -> None:
        for relative_path in (
            "player-rating.schema.json",
            "examples/player-rating.example.json",
        ):
            with self.subTest(relative_path=relative_path):
                with tempfile.TemporaryDirectory() as temporary:
                    copied = Path(temporary) / "contracts"
                    shutil.copytree(CONTRACT_ROOT, copied)
                    target = copied / relative_path
                    target.write_text(
                        target.read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(ValidationFailure):
                        run_five_output_validation(contract_root=copied)

        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / "contracts"
            shutil.copytree(CONTRACT_ROOT, copied)
            (copied / "common.schema.json").unlink()
            with self.assertRaises(ValidationFailure):
                run_five_output_validation(contract_root=copied)

    def test_tampered_validation_report_fails(self) -> None:
        tampered = replace(self.report, report_sha256="0" * 64)
        with self.assertRaises(ValidationFailure):
            verify_five_output_validation_report(tampered)
        not_all_pass = replace(
            self.report,
            all_pass=False,
            report_sha256=canonical_sha256(
                {**self.report.unsigned_payload(), "all_pass": False}
            ),
        )
        with self.assertRaises(ValidationFailure):
            verify_five_output_validation_report(not_all_pass)


class SealedDecisionIntegrityTests(unittest.TestCase):
    TEST_SIGNING_SEED = bytes.fromhex(
        "169416e755c36f2982c3b604341388e6449a05c26063fb9fb003ecbf3a1024e8"
    )
    TEST_REGISTRAR_ID = "scryglass:test-only:l2-evaluation-registrar-v1"

    @classmethod
    def setUpClass(cls) -> None:
        cls.validation_report = run_five_output_validation()

    def setUp(self) -> None:
        self.rows = [
            replace(
                row,
                metadata={
                    **dict(row.metadata),
                    "sealed_registration_fixture": "independent-b1",
                },
            )
            for row in build_synthetic_rows()
        ]
        self.synthetic_registry = load_evaluation_registry(FROZEN_REGISTRY)
        self.tempdir = tempfile.TemporaryDirectory(dir=DATA_ROOT)
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        source_sha256 = canonical_sha256([row.to_payload() for row in self.rows])
        rows_by_id = {row.row_id: row for row in self.rows}
        development_ids = sorted(
            {
                row_id
                for fold in self.synthetic_registry.split_plan.folds
                for row_id in fold.all_ids
            }
        )
        training_sha256 = canonical_sha256(
            [
                [row_id, rows_by_id[row_id].fingerprint()]
                for row_id in development_ids
            ]
        )
        self.registry = replace(
            self.synthetic_registry,
            is_synthetic_registry=False,
            b2_artifact_refs=(),
            b2_validation_report_sha256="",
            source_snapshot_id="source://registered/v2/evaluation/b1",
            source_snapshot_sha256=source_sha256,
            training_snapshot_id="source://registered/v2/evaluation/b1-training",
            training_snapshot_sha256=training_sha256,
            source_tree_sha256=source_sha256,
            source_crosswalk_sha256={
                "source://registered/v2/evaluation/b1": source_sha256,
            },
            entity_crosswalk_sha256={
                "entity://registered/v2/evaluation/b1": next(
                    iter(self.synthetic_registry.entity_crosswalk_sha256.values())
                ),
            },
            candidate_artifact_hashes={
                "registered-candidate": SYNTHETIC_CANDIDATE_ARTIFACT_SHA256,
            },
            served_transform_identities={
                "transform://registered/v2/evaluation/b1": {
                    "kind": "identity",
                    "sha256": SYNTHETIC_TRANSFORM_SHA256,
                },
            },
            noninferiority_provenance="registered-b1-test-policy",
            invalidation_reasons=("sealed_b1_registered_test_fixture",),
        )
        self.registry_path = self.root / "production-registry.json"
        registry_to_disk(self.registry, self.registry_path)
        registry_raw_sha256 = hashlib.sha256(
            self.registry_path.read_bytes()
        ).hexdigest()
        self.eligibility_path = self.root / "production-eligibility.json"
        self.eligibility_path.write_text(
            json.dumps(
                {
                    "registry_raw_sha256": registry_raw_sha256,
                    "registry_sha256": self.registry.sha256(),
                    "registry_kind": "test_only",
                    "production_eligible": False,
                    "contract_tree_sha256": CONTRACT_TREE_SHA256,
                    "registrar_id": self.TEST_REGISTRAR_ID,
                    "registrar_raw_sha256": sealed_module.REGISTRY_REGISTRAR_TRUST_ROOT_RAW_SHA256,
                    "registrar_object_sha256": sealed_module.REGISTRY_REGISTRAR_TRUST_ROOT_OBJECT_SHA256,
                    "registrar_verifier_public_key_hex": (
                        "85e1bfab0d2d7052dd8acc0d209798e66bfbbeefa4a4438f5284aa720a178e29"
                    ),
                    "source_ancestry": ["synthetic"],
                    "registry_provenance_sha256": canonical_sha256(
                        {
                            "source_snapshot_id": self.registry.source_snapshot_id,
                            "training_snapshot_id": self.registry.training_snapshot_id,
                            "source_tree_sha256": self.registry.source_tree_sha256,
                            "noninferiority_provenance": self.registry.noninferiority_provenance,
                        }
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        self.adapter = ToyAdapter(
            adapter_id="sealed-candidate",
            source_tree_sha256=self.registry.source_tree_sha256,
        )
        self.baseline_adapter = ToyAdapter(
            adapter_id="sealed-baseline",
            source_tree_sha256=self.registry.source_tree_sha256,
        )
        self.snapshot = make_frozen_evaluation_snapshot(
            self.rows,
            self.registry,
            locator=self.root / "sealed-snapshot.json",
        )
        self.plan = create_sealed_decision_plan(
            registry=self.registry,
            registrar_id=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
            registry_locator=self.registry_path,
            eligibility_locator=self.eligibility_path,
            snapshot=self.snapshot,
            candidate_adapter=self.adapter,
            candidate_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            baseline_adapter=self.baseline_adapter,
            baseline_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            transform_locator=Path("lol_kills/v2/evaluation/calibration.py"),
            five_output_validation_sha256=self.validation_report.report_sha256,
            metric_margins={"log_loss": 0.02, "brier": 0.02, "ece": 0.05},
            higher_is_better={
                "log_loss": False,
                "brier": False,
                "ece": False,
            },
        )
        self.plan_path = self.root / "decision-plan.json"
        write_sealed_decision_plan(self.plan, self.plan_path)
        self.ledger = AtomicSealedLedger(
            self.root / "sealed-ledger.jsonl",
            signing_seed=self.TEST_SIGNING_SEED,
            registrar_identity=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
        )
        self.request = self._request("seal-b1-main")

    def _request(
        self,
        seal_id: str,
        *,
        opened_at: datetime | None = None,
    ):
        return create_sealed_decision_request(
            seal_id=seal_id,
            plan=self.plan,
            decision_plan_locator=self.plan_path,
            opened_at=opened_at
            or datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc),
        )

    def _execute(
        self,
        *,
        request=None,
        ledger=None,
        snapshot="default",
        adapter=None,
        baseline_adapter=None,
    ):
        return execute_sealed_decision(
            adapter=adapter or self.adapter,
            baseline_adapter=baseline_adapter or self.baseline_adapter,
            registry=self.registry,
            snapshot=self.snapshot if snapshot == "default" else snapshot,
            plan=self.plan,
            request=request or self.request,
            validation_report=self.validation_report,
            ledger=ledger or self.ledger,
        )

    def test_sealed_decision_executes_once_with_complete_stage_chain(self) -> None:
        receipt = self._execute()
        self.assertEqual(tuple(stage.name for stage in receipt.stages), SEALED_STAGE_NAMES)
        self.assertEqual(
            set(receipt.hard_gates), set(REQUIRED_B1_SEALED_HARD_GATES)
        )
        self.assertTrue(all(receipt.hard_gates.values()))
        self.assertTrue(all(status == "ok" for _, status in receipt.suite_statuses))
        self.assertEqual(
            tuple(name for name, _ in receipt.suite_statuses),
            tuple(
                holdout.name
                for holdout in self.registry.split_plan.sealed_holdouts
            ),
        )
        verify_sealed_decision_receipt(receipt)
        self.ledger.verify_receipt(receipt)

    def test_reopen_same_decision_or_seal_fails(self) -> None:
        self._execute()
        with self.assertRaises(ValidationFailure):
            self._execute()
        second_request = self._request(
            self.request.seal_id,
            opened_at=datetime(2026, 7, 28, 12, 1, tzinfo=timezone.utc),
        )
        with self.assertRaises(ValidationFailure):
            self._execute(request=second_request)

    def test_different_seal_id_cannot_reopen_same_frozen_suites(self) -> None:
        self._execute()
        different_seal = self._request(
            "different-caller-seal",
            opened_at=datetime(2026, 7, 28, 12, 2, tzinfo=timezone.utc),
        )
        self.assertEqual(different_seal.decision_id, self.request.decision_id)
        self.assertEqual(different_seal.consumption_key, self.request.consumption_key)
        with self.assertRaises(ValidationFailure):
            self._execute(request=different_seal)

    def test_sealed_outcomes_are_loaded_only_after_atomic_claim(self) -> None:
        original_loader = sealed_module._load_and_verify_snapshot

        def assert_claim_then_load(snapshot, registry):
            entries = [
                json.loads(line)
                for line in self.ledger.path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            self.assertTrue(
                any(
                    entry.get("kind") == "open"
                    and entry.get("consumption_key") == self.request.consumption_key
                    for entry in entries
                )
            )
            return original_loader(snapshot, registry)

        with patch.object(
            sealed_module,
            "_load_and_verify_snapshot",
            side_effect=assert_claim_then_load,
        ):
            self._execute()

    def test_concurrent_open_allows_exactly_one_claim(self) -> None:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(self.ledger.claim, self.request)
                for _ in range(2)
            ]
        successes = 0
        failures = 0
        for future in futures:
            try:
                future.result()
                successes += 1
            except ValidationFailure:
                failures += 1
        self.assertEqual((successes, failures), (1, 1))

    def test_multiprocess_open_allows_exactly_one_outcome_claim(self) -> None:
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        processes = [
            context.Process(
                target=_claim_in_process,
                args=(
                    str(self.ledger.path),
                    self.request,
                    self.TEST_SIGNING_SEED,
                    self.TEST_REGISTRAR_ID,
                    queue,
                ),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
            self.assertEqual(process.exitcode, 0)
        self.assertEqual(sorted(queue.get() for _ in processes), ["blocked", "ok"])

    def test_policy_candidate_path_and_time_variants_share_outcome_key(self) -> None:
        variants = (
            replace(
                self.plan,
                metric_margins={
                    **self.plan.metric_margins,
                    "log_loss": 0.019,
                },
            ),
            replace(
                self.plan,
                candidate_adapter_version="0.0.2",
            ),
            replace(
                self.plan,
                candidate_sha256="6" * 64,
            ),
            replace(
                self.plan,
                baseline_adapter_version="0.0.2",
            ),
            replace(
                self.plan,
                transform_sha256="9" * 64,
            ),
            replace(
                self.plan,
                registry_sha256="8" * 64,
                registry_raw_sha256="7" * 64,
            ),
        )
        requests = []
        for index, base in enumerate(variants):
            variant = replace(
                base,
                plan_sha256=canonical_sha256(base.unsigned_payload()),
            )
            path = self.root / f"variant-plan-{index}.json"
            write_sealed_decision_plan(variant, path)
            requests.append(
                create_sealed_decision_request(
                    seal_id=f"variant-{index}",
                    plan=variant,
                    decision_plan_locator=path,
                    opened_at=datetime(
                        2026, 7, 28, 13, index, tzinfo=timezone.utc
                    ),
                )
            )
        for index, request in enumerate(requests):
            self.assertEqual(request.consumption_key, self.request.consumption_key)
            if index < len(requests) - 1:
                self.assertNotEqual(
                    request.decision_id, self.request.decision_id
                )
        self.ledger.claim(self.request)
        for request in requests:
            with self.assertRaises(ValidationFailure):
                self.ledger.claim(request)

    def test_semantic_outcome_key_ignores_suite_row_layout_and_display_names(
        self,
    ) -> None:
        suite_names = tuple(self.plan.suite_row_ids)
        variants = []
        for offset in range(len(suite_names)):
            order = suite_names[offset:] + suite_names[:offset]
            variants.append(
                replace(
                    self.plan,
                    critical_strata=order,
                    suite_row_ids={
                        name: self.plan.suite_row_ids[name] for name in order
                    },
                    suite_assignment_hashes={
                        name: self.plan.suite_assignment_hashes[name]
                        for name in order
                    },
                    suite_outcome_locator_hashes={
                        name: self.plan.suite_outcome_locator_hashes[name]
                        for name in order
                    },
                )
            )
        reverse_order = tuple(reversed(suite_names))
        variants.append(
            replace(
                self.plan,
                critical_strata=reverse_order,
                suite_row_ids={
                    name: self.plan.suite_row_ids[name]
                    for name in reverse_order
                },
                suite_assignment_hashes={
                    name: self.plan.suite_assignment_hashes[name]
                    for name in reverse_order
                },
                suite_outcome_locator_hashes={
                    name: self.plan.suite_outcome_locator_hashes[name]
                    for name in reverse_order
                },
            )
        )

        permuted_snapshot = make_frozen_evaluation_snapshot(
            tuple(reversed(self.rows)),
            self.registry,
            locator=self.root / "row-order-permuted-snapshot.json",
        )
        permuted_plan = create_sealed_decision_plan(
            registry=self.registry,
            registrar_id=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
            registry_locator=self.registry_path,
            eligibility_locator=self.eligibility_path,
            snapshot=permuted_snapshot,
            candidate_adapter=self.adapter,
            candidate_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            baseline_adapter=self.baseline_adapter,
            baseline_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            transform_locator=Path("lol_kills/v2/evaluation/calibration.py"),
            five_output_validation_sha256=self.validation_report.report_sha256,
            metric_margins=dict(self.plan.metric_margins),
            higher_is_better=dict(self.plan.higher_is_better),
        )
        variants.append(permuted_plan)

        renamed_snapshot_base = replace(
            self.snapshot,
            source_snapshot_id="source://renamed-display-only",
            training_snapshot_id="training://renamed-display-only",
        )
        renamed_snapshot = replace(
            renamed_snapshot_base,
            snapshot_sha256=canonical_sha256(
                renamed_snapshot_base.unsigned_payload()
            ),
        )
        variants.append(replace(self.plan, snapshot=renamed_snapshot))

        requests = []
        for index, base in enumerate(variants):
            variant = replace(
                base,
                plan_sha256=canonical_sha256(base.unsigned_payload()),
            )
            path = self.root / f"semantic-variant-{index}.json"
            write_sealed_decision_plan(variant, path)
            request = create_sealed_decision_request(
                seal_id=f"semantic-variant-{index}",
                plan=variant,
                decision_plan_locator=path,
            )
            self.assertEqual(
                request.consumption_key, self.request.consumption_key
            )
            requests.append(request)

        self.ledger.claim(self.request)
        restarted = AtomicSealedLedger(
            self.ledger.path,
            verifier_public_key=self.ledger.verifier_public_key,
            registrar_identity=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
        )
        for request in requests:
            with self.assertRaises(ValidationFailure):
                restarted.claim(request)

        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        process = context.Process(
            target=_claim_in_process,
            args=(
                str(self.ledger.path),
                requests[-1],
                self.TEST_SIGNING_SEED,
                self.TEST_REGISTRAR_ID,
                queue,
            ),
        )
        process.start()
        process.join(timeout=10)
        self.assertEqual(process.exitcode, 0)
        self.assertEqual(queue.get(), "blocked")

    def test_changed_outcome_fingerprint_changes_semantic_outcome_key(self) -> None:
        changed_rows = list(self.rows)
        changed_rows[0] = replace(
            changed_rows[0],
            label=1 - changed_rows[0].label,
        )
        changed_snapshot = make_frozen_evaluation_snapshot(
            changed_rows,
            self.registry,
            locator=self.root / "changed-outcome-snapshot.json",
        )
        changed_plan = create_sealed_decision_plan(
            registry=self.registry,
            registrar_id=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
            registry_locator=self.registry_path,
            eligibility_locator=self.eligibility_path,
            snapshot=changed_snapshot,
            candidate_adapter=self.adapter,
            candidate_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            baseline_adapter=self.baseline_adapter,
            baseline_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            transform_locator=Path("lol_kills/v2/evaluation/calibration.py"),
            five_output_validation_sha256=self.validation_report.report_sha256,
            metric_margins=dict(self.plan.metric_margins),
            higher_is_better=dict(self.plan.higher_is_better),
        )
        path = self.root / "changed-outcome-plan.json"
        write_sealed_decision_plan(changed_plan, path)
        changed_request = create_sealed_decision_request(
            seal_id="changed-outcome",
            plan=changed_plan,
            decision_plan_locator=path,
        )
        self.assertNotEqual(
            changed_request.consumption_key,
            self.request.consumption_key,
        )

    def test_semantic_outcome_identity_rejects_duplicates_and_missing_suites(
        self,
    ) -> None:
        suite_name = next(iter(self.plan.suite_row_ids))
        membership = self.plan.suite_row_ids[suite_name]
        invalid_plans = (
            replace(
                self.plan,
                suite_row_ids={
                    **self.plan.suite_row_ids,
                    suite_name: (membership[0], membership[0], *membership[1:]),
                },
            ),
            replace(
                self.plan,
                critical_strata=(
                    self.plan.critical_strata[0],
                    self.plan.critical_strata[0],
                    *self.plan.critical_strata[2:],
                ),
            ),
            replace(
                self.plan,
                suite_row_ids={
                    name: rows
                    for name, rows in self.plan.suite_row_ids.items()
                    if name != suite_name
                },
            ),
            replace(
                self.plan,
                snapshot=replace(
                    self.snapshot,
                    row_fingerprints=(
                        self.snapshot.row_fingerprints[0],
                        self.snapshot.row_fingerprints[0],
                        *self.snapshot.row_fingerprints[2:],
                    ),
                ),
            ),
        )
        for invalid in invalid_plans:
            with self.assertRaises(ValidationFailure):
                sealed_module._consumption_key(invalid)

    def test_noncanonical_or_extra_persisted_plan_fails_before_claim(self) -> None:
        canonical = json.loads(self.plan_path.read_text(encoding="utf-8"))
        malformed_payloads = (
            json.dumps(canonical, indent=2).encode("utf-8"),
            (
                json.dumps(
                    {**canonical, "caller_extra": True},
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            ).encode("utf-8"),
            (
                self.plan_path.read_text(encoding="utf-8")
                .replace(
                    '"plan_sha256":',
                    '"plan_sha256":"duplicate","plan_sha256":',
                    1,
                )
                .encode("utf-8")
            ),
        )
        for index, raw in enumerate(malformed_payloads):
            with self.subTest(index=index):
                path = self.root / f"malformed-plan-{index}.json"
                path.write_bytes(raw)
                with self.assertRaises(ValidationFailure):
                    create_sealed_decision_request(
                        seal_id=f"malformed-{index}",
                        plan=self.plan,
                        decision_plan_locator=path,
                    )
        self.assertFalse(self.ledger.path.exists())

    def test_signed_final_entry_verifies_after_restart_and_tamper_fails(self) -> None:
        receipt = self._execute()
        restarted = AtomicSealedLedger(
            self.ledger.path,
            verifier_public_key=self.ledger.verifier_public_key,
            registrar_identity=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
        )
        restarted.verify_receipt(receipt)

        original_lines = self.ledger.path.read_text(
            encoding="utf-8"
        ).splitlines()
        for mutation in ("signed_field", "missing_signature", "truncated_signature"):
            with self.subTest(mutation=mutation):
                lines = list(original_lines)
                final = json.loads(lines[-1])
                if mutation == "signed_field":
                    final["candidate_identity"]["adapter_version"] = "forged"
                elif mutation == "missing_signature":
                    final.pop("registrar_signature")
                else:
                    final["registrar_signature"] = final[
                        "registrar_signature"
                    ][:16]
                lines[-1] = json.dumps(
                    final, sort_keys=True, separators=(",", ":")
                )
                self.ledger.path.write_text(
                    "\n".join(lines) + "\n", encoding="utf-8"
                )
                with self.assertRaises(ValidationFailure):
                    restarted.verify_receipt(receipt)
        self.ledger.path.write_text(
            "\n".join(original_lines) + "\n", encoding="utf-8"
        )

    def test_registrar_verifier_key_is_pinned_before_open_and_on_receipt(
        self,
    ) -> None:
        invalid_ledger_paths = [
            self.root / "wrong-seed-ledger.jsonl",
            self.root / "wrong-verifier-ledger.jsonl",
            self.root / "missing-verifier-ledger.jsonl",
        ]
        with self.assertRaises(ValidationFailure):
            AtomicSealedLedger(
                invalid_ledger_paths[0],
                signing_seed=b"\x11" * 32,
                registrar_identity=self.TEST_REGISTRAR_ID,
                registrar_kind="test_only",
            )
        with self.assertRaises(ValidationFailure):
            AtomicSealedLedger(
                invalid_ledger_paths[1],
                verifier_public_key=b"\x22" * 32,
                registrar_identity=self.TEST_REGISTRAR_ID,
                registrar_kind="test_only",
            )
        with self.assertRaises(ValidationFailure):
            AtomicSealedLedger(
                invalid_ledger_paths[2],
                registrar_identity=self.TEST_REGISTRAR_ID,
                registrar_kind="test_only",
            )
        self.assertTrue(all(not path.exists() for path in invalid_ledger_paths))

        swapped_plan_base = replace(
            self.plan,
            registrar_verifier_public_key_hex="11" * 32,
        )
        swapped_plan = replace(
            swapped_plan_base,
            plan_sha256=canonical_sha256(swapped_plan_base.unsigned_payload()),
        )
        swapped_path = self.root / "swapped-key-plan.json"
        write_sealed_decision_plan(swapped_plan, swapped_path)
        swapped_request = create_sealed_decision_request(
            seal_id="swapped-key",
            plan=swapped_plan,
            decision_plan_locator=swapped_path,
        )
        swapped_ledger_path = self.root / "swapped-key-ledger.jsonl"
        swapped_ledger = AtomicSealedLedger(
            swapped_ledger_path,
            signing_seed=self.TEST_SIGNING_SEED,
            registrar_identity=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
        )
        with self.assertRaises(ValidationFailure):
            execute_sealed_decision(
                adapter=self.adapter,
                baseline_adapter=self.baseline_adapter,
                registry=self.registry,
                snapshot=self.snapshot,
                plan=swapped_plan,
                request=swapped_request,
                validation_report=self.validation_report,
                ledger=swapped_ledger,
            )
        self.assertFalse(swapped_ledger_path.exists())

        caller_root = self.root / "caller-trust-root.json"
        caller_payload = json.loads(
            sealed_module.REGISTRY_REGISTRAR_TRUST_ROOT.read_text(
                encoding="utf-8"
            )
        )
        caller_payload["test_only"]["registrars"][0][
            "verifier_public_key_hex"
        ] = "11" * 32
        caller_root.write_text(
            json.dumps(caller_payload, sort_keys=True, separators=(",", ":"))
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(ValidationFailure):
            create_sealed_decision_plan(
                registry=self.registry,
                registrar_locator=caller_root,
                registrar_id=self.TEST_REGISTRAR_ID,
                registrar_kind="test_only",
                registry_locator=self.registry_path,
                eligibility_locator=self.eligibility_path,
                snapshot=self.snapshot,
                candidate_adapter=self.adapter,
                candidate_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
                baseline_adapter=self.baseline_adapter,
                baseline_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
                transform_locator=Path("lol_kills/v2/evaluation/calibration.py"),
                five_output_validation_sha256=(
                    self.validation_report.report_sha256
                ),
                metric_margins=dict(self.plan.metric_margins),
                higher_is_better=dict(self.plan.higher_is_better),
            )

        receipt = self._execute()
        tampered = replace(
            receipt,
            registrar_verifier_public_key_hex="11" * 32,
        )
        tampered = replace(
            tampered,
            receipt_sha256=canonical_sha256(tampered.unsigned_payload()),
        )
        with self.assertRaises(ValidationFailure):
            self.ledger.verify_receipt(tampered)

    def test_optional_or_mismatched_snapshot_fails_before_open(self) -> None:
        with self.assertRaises(ValidationFailure):
            execute_sealed_decision(
                adapter=self.adapter,
                baseline_adapter=self.baseline_adapter,
                registry=self.registry,
                snapshot=None,
                plan=self.plan,
                request=self.request,
                validation_report=self.validation_report,
                ledger=self.ledger,
            )
        tampered = replace(
            self.snapshot,
            training_snapshot_sha256="9" * 64,
        )
        tampered = replace(
            tampered,
            snapshot_sha256=canonical_sha256(tampered.unsigned_payload()),
        )
        second_ledger = AtomicSealedLedger(
            self.root / "tampered-snapshot-ledger.jsonl",
            signing_seed=self.TEST_SIGNING_SEED,
            registrar_identity=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
        )
        with self.assertRaises(ValidationFailure):
            self._execute(snapshot=tampered, ledger=second_ledger)

    def test_snapshot_raw_byte_mismatch_fails_after_consuming_claim(self) -> None:
        Path(self.snapshot.locator).write_bytes(
            Path(self.snapshot.locator).read_bytes() + b" "
        )
        with self.assertRaises(ValidationFailure):
            self._execute()
        entries = [
            json.loads(line)
            for line in self.ledger.path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual([entry["kind"] for entry in entries], ["open"])

    def test_snapshot_fingerprint_must_match_actual_loaded_row(self) -> None:
        first_row_id, _ = self.snapshot.row_fingerprints[0]
        forged_snapshot_base = replace(
            self.snapshot,
            row_fingerprints=(
                (first_row_id, "9" * 64),
                *self.snapshot.row_fingerprints[1:],
            ),
        )
        forged_snapshot = replace(
            forged_snapshot_base,
            snapshot_sha256=canonical_sha256(
                forged_snapshot_base.unsigned_payload()
            ),
        )
        forged_plan_base = replace(self.plan, snapshot=forged_snapshot)
        forged_plan = replace(
            forged_plan_base,
            plan_sha256=canonical_sha256(forged_plan_base.unsigned_payload()),
        )
        path = self.root / "forged-fingerprint-plan.json"
        write_sealed_decision_plan(forged_plan, path)
        request = create_sealed_decision_request(
            seal_id="forged-fingerprint",
            plan=forged_plan,
            decision_plan_locator=path,
        )
        ledger = AtomicSealedLedger(
            self.root / "forged-fingerprint-ledger.jsonl",
            signing_seed=self.TEST_SIGNING_SEED,
            registrar_identity=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
        )
        with self.assertRaisesRegex(
            ValidationFailure,
            "row identities are incomplete or changed",
        ):
            execute_sealed_decision(
                adapter=self.adapter,
                baseline_adapter=self.baseline_adapter,
                registry=self.registry,
                snapshot=forged_snapshot,
                plan=forged_plan,
                request=request,
                validation_report=self.validation_report,
                ledger=ledger,
            )
        entries = [
            json.loads(line)
            for line in ledger.path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([entry["kind"] for entry in entries], ["open"])

    def test_unavailable_suite_is_a_hard_failure(self) -> None:
        valid_report = evaluate_candidate(
            self.adapter,
            self.rows,
            self.registry,
            sealed_rows_snapshot=self.snapshot.fingerprint_map(),
        )
        unavailable_holdouts = {
            name: (
                {**payload, "status": "unavailable"}
                if name == "temporal"
                else payload
            )
            for name, payload in valid_report.holdout_reports.items()
        }
        unavailable_report = replace(
            valid_report,
            holdout_reports=unavailable_holdouts,
        )
        with patch(
            "lol_kills.v2.evaluation.sealed.evaluate_candidate",
            return_value=unavailable_report,
        ):
            with self.assertRaises(ValidationFailure):
                self._execute()

    def test_swapped_stage_and_tampered_receipt_fail(self) -> None:
        receipt = self._execute()
        swapped = replace(
            receipt,
            stages=(receipt.stages[1], receipt.stages[0], *receipt.stages[2:]),
        )
        swapped = replace(
            swapped,
            receipt_sha256=canonical_sha256(swapped.unsigned_payload()),
        )
        with self.assertRaises(ValidationFailure):
            verify_sealed_decision_receipt(swapped)

        tampered = replace(
            receipt,
            candidate_metrics={
                **receipt.candidate_metrics,
                "log_loss": receipt.candidate_metrics["log_loss"] + 0.01,
            },
        )
        with self.assertRaises(ValidationFailure):
            verify_sealed_decision_receipt(tampered)

    def test_stage_byte_tamper_and_caller_rehashed_receipt_fail(self) -> None:
        receipt = self._execute()
        Path(receipt.stages[2].artifact_locator).write_bytes(b"forged fitted state\n")
        with self.assertRaises(ValidationFailure):
            self.ledger.verify_receipt(receipt)

        forged = replace(
            receipt,
            candidate_metrics={
                name: 0.0 for name in receipt.candidate_metrics
            },
        )
        forged = replace(
            forged,
            receipt_sha256=canonical_sha256(forged.unsigned_payload()),
        )
        with self.assertRaises(ValidationFailure):
            self.ledger.verify_receipt(forged)

    def test_direct_claim_cannot_fake_finalize(self) -> None:
        claim = self.ledger.claim(self.request)
        fake_stage = SealedStage(
            name="raw",
            input_sha256="1" * 64,
            output_sha256="2" * 64,
            artifact_locator=str(self.root / "fake-raw.json"),
        )
        fake = SealedDecisionReceipt(
            request=self.request,
            stages=(fake_stage,) * len(SEALED_STAGE_NAMES),
            suite_statuses=tuple(
                (name, "ok") for name in self.plan.critical_strata
            ),
            candidate_metrics={name: 0.0 for name in self.plan.metric_names},
            baseline_metrics={name: 1.0 for name in self.plan.metric_names},
            metric_bounds={
                name: {"point": -1.0, "lower_95": -1.0, "upper_95": -1.0}
                for name in self.plan.metric_names
            },
            metric_decisions={name: True for name in self.plan.metric_names},
            critical_stratum_decisions={
                name: True for name in self.plan.critical_strata
            },
            metric_margins=dict(self.plan.metric_margins),
            higher_is_better=dict(self.plan.higher_is_better),
            uncertainty_rule=self.plan.uncertainty_rule,
            multiplicity_rule=self.plan.multiplicity_rule,
            secondary_benefit_rule=self.plan.secondary_benefit_rule,
            hard_gates={
                name: True for name in REQUIRED_B1_SEALED_HARD_GATES
            },
            registrar_verifier_public_key_hex=(
                self.plan.registrar_verifier_public_key_hex
            ),
            evaluation_report_sha256="3" * 64,
            evaluation_report_locator=str(self.root / "fake-report.json"),
            ledger_entry_sha256=claim.ledger_entry_sha256,
            receipt_locator=str(self.root / "fake-receipt.json"),
            receipt_sha256="4" * 64,
        )
        with self.assertRaises(ValidationFailure):
            self.ledger.finalize(fake)

    def test_swapped_independently_passing_baseline_fails(self) -> None:
        other_baseline = replace(
            self.baseline_adapter,
            runtime_artifact_sha256="9" * 64,
        )
        with self.assertRaises(ValidationFailure):
            self._execute(baseline_adapter=other_baseline)

    def test_promotion_consumes_exact_verified_receipt(self) -> None:
        receipt = self._execute()
        plan = PromotionPlan(
            contract_tree_sha256=CONTRACT_TREE_SHA256,
            split_registry_sha256=self.registry.sha256(),
            metric_noninferiority_margins={
                "log_loss": 0.02,
                "brier": 0.02,
                "ece": 0.05,
            },
            higher_is_better={
                "log_loss": False,
                "brier": False,
                "ece": False,
            },
        )
        promotion = build_promotion_report(
            model_id="sealed-candidate",
            model_version="1",
            registry_sha256=self.registry.sha256(),
            candidate_registry_sha256=self.registry.sha256(),
            planned=plan,
            candidate_metrics=receipt.candidate_metrics,
            baseline_metrics=receipt.baseline_metrics,
            hard_gates=receipt.hard_gates,
            sealed_receipt=receipt,
            sealed_ledger=self.ledger,
        )
        self.assertEqual(promotion.decision, Decision.BLOCK)

        caller_gates = {**receipt.hard_gates, "caller_invented_gate": True}
        blocked = build_promotion_report(
            model_id="sealed-candidate",
            model_version="1",
            registry_sha256=self.registry.sha256(),
            candidate_registry_sha256=self.registry.sha256(),
            planned=plan,
            candidate_metrics=receipt.candidate_metrics,
            baseline_metrics=receipt.baseline_metrics,
            hard_gates=caller_gates,
            sealed_receipt=receipt,
            sealed_ledger=self.ledger,
        )
        self.assertEqual(blocked.decision, Decision.BLOCK)

    def test_sealed_terrible_metrics_override_good_development_aggregate(self) -> None:
        candidate_report = evaluate_candidate(
            self.adapter,
            self.rows,
            self.registry,
            sealed_rows_snapshot=self.snapshot.fingerprint_map(),
        )
        baseline_report = evaluate_candidate(
            self.baseline_adapter,
            self.rows,
            self.registry,
            sealed_rows_snapshot=self.snapshot.fingerprint_map(),
        )
        rows_by_id = {row.row_id: row for row in self.rows}
        terrible_holdouts = {}
        for name, payload in candidate_report.holdout_reports.items():
            terrible_holdouts[name] = {
                **payload,
                "scored_probabilities": {
                    row_id: (0.01 if rows_by_id[row_id].label == 1 else 0.99)
                    for row_id in payload["scored_row_ids"]
                },
            }
        terrible_report = replace(
            candidate_report,
            holdout_reports=terrible_holdouts,
            aggregate_calibrated_metrics={
                name: 0.0 for name in candidate_report.aggregate_calibrated_metrics
            },
        )
        with patch.object(
            sealed_module,
            "evaluate_candidate",
            side_effect=(terrible_report, baseline_report),
        ):
            receipt = self._execute()
        promotion = build_promotion_report(
            model_id="sealed-candidate",
            model_version="1",
            registry_sha256=self.registry.sha256(),
            candidate_registry_sha256=self.registry.sha256(),
            planned=PromotionPlan(
                contract_tree_sha256=CONTRACT_TREE_SHA256,
                split_registry_sha256=self.registry.sha256(),
                metric_noninferiority_margins=dict(self.plan.metric_margins),
                higher_is_better=dict(self.plan.higher_is_better),
            ),
            candidate_metrics=receipt.candidate_metrics,
            baseline_metrics=receipt.baseline_metrics,
            hard_gates=receipt.hard_gates,
            sealed_receipt=receipt,
            sealed_ledger=self.ledger,
        )
        self.assertNotEqual(promotion.decision, Decision.ACCEPT)
        self.assertTrue(
            all(value == 0.0 for value in terrible_report.aggregate_calibrated_metrics.values())
        )

    def test_request_and_promotion_cannot_override_frozen_metric_rules(self) -> None:
        with self.assertRaises(ValidationFailure):
            create_sealed_decision_request(
                seal_id="override",
                plan=self.plan,
                decision_plan_locator=self.plan_path,
                metric_names=("auc",),
                metric_margins={"auc": 999.0},
                higher_is_better={"auc": True},
            )
        receipt = self._execute()
        wrong_plan = PromotionPlan(
            contract_tree_sha256=CONTRACT_TREE_SHA256,
            split_registry_sha256=self.registry.sha256(),
            metric_noninferiority_margins={
                **self.plan.metric_margins,
                "log_loss": 999.0,
            },
            higher_is_better={
                **self.plan.higher_is_better,
                "log_loss": True,
            },
        )
        blocked = build_promotion_report(
            model_id="sealed-candidate",
            model_version="1",
            registry_sha256=self.registry.sha256(),
            candidate_registry_sha256=self.registry.sha256(),
            planned=wrong_plan,
            candidate_metrics=receipt.candidate_metrics,
            baseline_metrics=receipt.baseline_metrics,
            hard_gates=receipt.hard_gates,
            sealed_receipt=receipt,
            sealed_ledger=self.ledger,
        )
        self.assertEqual(blocked.decision, Decision.BLOCK)

    def test_synthetic_registry_cannot_open_or_promote(self) -> None:
        flipped = replace(self.synthetic_registry, is_synthetic_registry=False)
        flipped_eligibility = self.root / "flipped-eligibility.json"
        flipped_eligibility.write_text(
            json.dumps(
                {
                    "registry_raw_sha256": hashlib.sha256(
                        FROZEN_REGISTRY.read_bytes()
                    ).hexdigest(),
                    "registry_sha256": flipped.sha256(),
                    "registry_kind": "production",
                    "production_eligible": True,
                    "contract_tree_sha256": CONTRACT_TREE_SHA256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        flipped_snapshot = make_frozen_evaluation_snapshot(
            self.rows,
            flipped,
            locator=self.root / "flipped-snapshot.json",
        )
        flipped_plan = create_sealed_decision_plan(
            registry=flipped,
            registrar_id=self.TEST_REGISTRAR_ID,
            registrar_kind="test_only",
            registry_locator=FROZEN_REGISTRY,
            eligibility_locator=flipped_eligibility,
            snapshot=flipped_snapshot,
            candidate_adapter=ToyAdapter(
                source_tree_sha256=flipped.source_tree_sha256
            ),
            candidate_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            baseline_adapter=ToyAdapter(
                adapter_id="baseline",
                source_tree_sha256=flipped.source_tree_sha256,
            ),
            baseline_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
            transform_locator=Path("lol_kills/v2/evaluation/calibration.py"),
            five_output_validation_sha256=self.validation_report.report_sha256,
            metric_margins={"log_loss": 0.02},
            higher_is_better={"log_loss": False},
        )
        flipped_plan_path = self.root / "flipped-plan.json"
        write_sealed_decision_plan(flipped_plan, flipped_plan_path)
        flipped_request = create_sealed_decision_request(
            seal_id="flipped-synthetic",
            plan=flipped_plan,
            decision_plan_locator=flipped_plan_path,
        )
        with self.assertRaises(ValidationFailure):
            execute_sealed_decision(
                adapter=ToyAdapter(source_tree_sha256=flipped.source_tree_sha256),
                baseline_adapter=ToyAdapter(
                    adapter_id="baseline",
                    source_tree_sha256=flipped.source_tree_sha256,
                ),
                registry=flipped,
                snapshot=flipped_snapshot,
                plan=flipped_plan,
                request=flipped_request,
                validation_report=self.validation_report,
                ledger=self.ledger,
            )
        blocked = build_promotion_report(
            model_id="synthetic",
            model_version="1",
            registry_sha256=self.synthetic_registry.sha256(),
            candidate_registry_sha256=self.synthetic_registry.sha256(),
            planned=PromotionPlan(
                contract_tree_sha256=CONTRACT_TREE_SHA256,
                split_registry_sha256=self.synthetic_registry.sha256(),
                metric_noninferiority_margins={"log_loss": 0.02},
                higher_is_better={"log_loss": False},
            ),
            candidate_metrics={"log_loss": 0.5},
            baseline_metrics={"log_loss": 0.5},
            hard_gates={"caller": True},
        )
        self.assertEqual(blocked.decision, Decision.BLOCK)

    def test_repository_has_no_synthetic_capable_production_registrar(self) -> None:
        trust_root = json.loads(
            sealed_module.REGISTRY_REGISTRAR_TRUST_ROOT.read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(trust_root["production"]["registrars"], [])
        self.assertFalse(
            trust_root["production"]["synthetic_ancestry_allowed"]
        )
        with self.assertRaises(ValidationFailure):
            create_sealed_decision_plan(
                registry=self.registry,
                registrar_id="self-authored-production",
                registrar_kind="production",
                registry_locator=self.registry_path,
                eligibility_locator=self.eligibility_path,
                snapshot=self.snapshot,
                candidate_adapter=self.adapter,
                candidate_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
                baseline_adapter=self.baseline_adapter,
                baseline_locator=Path("lol_kills/v2/evaluation/pipeline.py"),
                transform_locator=Path("lol_kills/v2/evaluation/calibration.py"),
                five_output_validation_sha256=(
                    self.validation_report.report_sha256
                ),
                metric_margins={"log_loss": 0.02},
                higher_is_better={"log_loss": False},
            )

    def test_frozen_synthetic_hashes_are_actual_owned_bytes(self) -> None:
        payload = json.loads(FROZEN_REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(
            payload["baseline_artifact_hashes"]["candidate-baseline"],
            SYNTHETIC_BASELINE_ARTIFACT_SHA256,
        )
        self.assertEqual(
            payload["candidate_artifact_hashes"]["toy-synthetic-candidate"],
            SYNTHETIC_CANDIDATE_ARTIFACT_SHA256,
        )
        self.assertEqual(
            payload["served_transform_identities"][
                "transform://v2/fixture"
            ]["sha256"],
            SYNTHETIC_TRANSFORM_SHA256,
        )
        self.assertEqual(
            SYNTHETIC_BASELINE_ARTIFACT_SHA256,
            hashlib.sha256(
                Path("lol_kills/v2/evaluation/metrics.py").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            SYNTHETIC_CANDIDATE_ARTIFACT_SHA256,
            hashlib.sha256(
                Path("lol_kills/v2/evaluation/pipeline.py").read_bytes()
            ).hexdigest(),
        )
        self.assertEqual(
            SYNTHETIC_TRANSFORM_SHA256,
            hashlib.sha256(
                Path("lol_kills/v2/evaluation/calibration.py").read_bytes()
            ).hexdigest(),
        )
