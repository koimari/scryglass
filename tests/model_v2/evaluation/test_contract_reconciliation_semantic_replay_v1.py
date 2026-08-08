from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import tempfile
from pathlib import Path
import unittest

from lol_kills.v2.evaluation import contract_validation as validation
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.contract_reconciliation_semantic_replay_v1 import (
    AUTHORITY,
    ContractSemanticReplayError,
    build_reference_semantic_replay_v1,
    replay_matches_reference_semantic_report_v1,
    validate_reference_semantic_replay_v1,
    write_reference_semantic_replay_v1,
)
from lol_kills.v2.evaluation.types import canonical_sha256


ROOT = Path(__file__).resolve().parents[3]
FIXED_TIME = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)


class ContractReconciliationSemanticReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_reference_semantic_replay_v1(
            root=ROOT, clock=lambda: FIXED_TIME
        )

    def test_candidate_anchor_reference_replay_is_complete_but_non_authorizing(
        self,
    ) -> None:
        report = self.report
        self.assertTrue(report["five_output_report"]["all_pass"])
        self.assertEqual(report["coverage"]["invariant_pass_count"], 62)
        self.assertEqual(report["coverage"]["mutation_pass_count"], 42)
        self.assertEqual(
            report["coverage"]["structural_mutation_pass_count"], 24
        )
        self.assertTrue(
            report["runner_provenance"]["generated_by_evaluated_system"]
        )
        self.assertFalse(
            report["runner_provenance"]["independent_review_eligible"]
        )
        self.assertEqual(report["authority"], AUTHORITY)
        self.assertFalse(any(report["authority"].values()))

    def test_reference_report_replays_exactly(self) -> None:
        self.assertTrue(
            replay_matches_reference_semantic_report_v1(
                self.report, root=ROOT
            )
        )

    def test_self_rehashed_independence_forgery_is_rejected(self) -> None:
        forged = deepcopy(self.report)
        forged["runner_provenance"]["generated_by_evaluated_system"] = False
        forged["runner_provenance"]["independent_review_eligible"] = True
        forged["artifact_sha256"] = canonical_sha256(
            {key: value for key, value in forged.items() if key != "artifact_sha256"}
        )
        with self.assertRaisesRegex(
            ContractSemanticReplayError, "runner provenance"
        ):
            validate_reference_semantic_replay_v1(forged, root=ROOT)

    def test_default_anchor_factory_remains_the_frozen_production_default(
        self,
    ) -> None:
        anchors = validation.default_contract_validation_anchors()
        self.assertEqual(anchors.contract_tree_sha256, validation.CONTRACT_TREE_SHA256)
        self.assertEqual(
            anchors.schema_sha256, dict(validation.EXPECTED_SCHEMA_SHA256)
        )
        self.assertEqual(
            anchors.contract_validation_trust_root_raw_sha256,
            validation.EXPECTED_CONTRACT_VALIDATION_TRUST_ROOT_RAW_SHA256,
        )

    def test_default_execution_does_not_silently_accept_candidate_anchors(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            ValidationFailure, "canonical source-tree-v1 digest"
        ):
            validation.run_five_output_validation()

    def test_reference_writer_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reference.json"
            write_reference_semantic_replay_v1(root=ROOT, output=output)
            with self.assertRaisesRegex(
                ContractSemanticReplayError, "refusing to overwrite"
            ):
                write_reference_semantic_replay_v1(root=ROOT, output=output)


if __name__ == "__main__":
    unittest.main()
