"""Adversarial regressions for the independent B1 contract-validation remand."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import lol_kills.v2.evaluation.contract_validation as validation
from lol_kills.v2.evaluation import (
    ValidationFailure,
    run_five_output_validation,
    validate_output_payload,
    verify_five_output_validation_report,
)
from lol_kills.v2.evaluation.types import canonical_sha256


CONTRACT_ROOT = Path("docs/model-v2/contracts")


def _example(output: str) -> dict:
    filename = validation.EXAMPLE_FILES[output]
    return json.loads((CONTRACT_ROOT / filename).read_text(encoding="utf-8"))


def _unavailable(output: str, required_status: str = "stale") -> dict:
    schema_name = validation.OUTPUT_SCHEMAS[output]
    schema = json.loads((CONTRACT_ROOT / schema_name).read_text(encoding="utf-8"))
    canonical = _example(output)
    document = {
        key: deepcopy(canonical[key])
        for key in schema["required"]
    }
    document["status"] = "unavailable"
    document["error"] = {
        "code": {
            "missing": "missing_required_input",
            "stale": "stale_context",
            "conflict": "patch_conflict",
        }[required_status],
        "message": "Typed unavailable fixture.",
        "retryable": True,
        "missing_fields": ["required_input"] if required_status == "missing" else [],
        "stale_fields": ["required_input"] if required_status == "stale" else [],
    }
    provenance = document["provenance"]
    provenance["required_input_status"] = required_status
    if required_status == "stale":
        provenance["freshness_checks"][0].update(
            {
                "source_updated_at": "2026-07-20T00:00:00Z",
                "limit_seconds": 60,
                "fresh": False,
            }
        )
        provenance["input_conflicts"] = []
    elif required_status == "missing":
        provenance["freshness_checks"] = []
        provenance["input_conflicts"] = []
    else:
        provenance["freshness_checks"] = []
        provenance["input_conflicts"] = [
            {
                "input_id": "scryglass:source:analysis-input",
                "conflict_type": "patch",
                "source_ids": [
                    "scryglass:source:analysis-input",
                    "scryglass:source:oe-example",
                ],
                "detected_at": "2026-07-27T17:59:00Z",
            }
        ]
    validation._seal_dynamic_identity(document)
    return document


class ContractValidationRemandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schemas = {
            name: json.loads((CONTRACT_ROOT / name).read_text(encoding="utf-8"))
            for name in validation.SCHEMA_FILES
        }
        cls.examples = {
            output: _example(output)
            for output in validation.OUTPUT_SCHEMAS
        }

    def _rejects(self, output: str, document: dict) -> None:
        validation._seal_dynamic_identity(document)
        with self.assertRaises(ValidationFailure):
            validate_output_payload(output, document)

    def test_nonexistent_protocol_taxonomy_and_identity_ids_fail(self) -> None:
        partial = _example("partial_draft_state")
        partial["state"]["protocol_id"] = "scryglass:protocol:nonexistent"
        self._rejects("partial_draft_state", partial)

        tier = _example("tier_list")
        tier["competition_scope_id"] = "scryglass:competition-scope:nonexistent"
        self._rejects("tier_list", tier)

        player = _example("player_rating")
        player["league_id"] = "scryglass:league:nonexistent"
        self._rejects("player_rating", player)

        champion = _example("tier_list")
        champion["entries"][0]["champion_id"] = "riot:champion:999999"
        self._rejects("tier_list", champion)

    def test_arbitrary_terminal_delegation_hash_fails(self) -> None:
        terminal = validation._terminal_partial_fixture(self.examples)
        terminal["terminal_delegation"]["terminal_output_sha256"] = "9" * 64
        self._rejects("partial_draft_state", terminal)

    def test_game_count_and_aggregate_confidence_evidence_fail(self) -> None:
        game_count = _example("player_rating")
        game_count["evidence"]["posterior_displacement"]["method_id"] = (
            "scryglass:evidence-method:game-count-confidence"
        )
        self._rejects("player_rating", game_count)

        aggregate = _example("player_rating")
        aggregate["evidence"]["precision"]["method_id"] = (
            "scryglass:evidence-method:aggregate-confidence"
        )
        self._rejects("player_rating", aggregate)

    def test_zero_or_unregistered_reliability_map_fails(self) -> None:
        for digest in ("0" * 64, "8" * 64):
            with self.subTest(digest=digest):
                player = _example("player_rating")
                player["reliability"]["stratum_mapping_sha256"] = digest
                self._rejects("player_rating", player)

    def test_unevaluated_tie_and_text_only_search_registration_fail(self) -> None:
        tied = validation._tie_fixture(self.examples)
        tied["entries"].reverse()
        for index, entry in enumerate(tied["entries"], start=1):
            entry["rank"] = index
        self._rejects("tier_list", tied)

        exact = validation._exact_search_fixture(self.examples)
        exact["search"]["policy_id"] = "scryglass:policy:text-only-exact"
        self._rejects("partial_draft_state", exact)

    def test_false_team_component_math_fails(self) -> None:
        team = _example("team_rating")
        team["roster_strength_component"] += 1.0
        team["posterior_mean"] += 1.0
        self._rejects("team_rating", team)

    def test_stale_but_fresh_and_conflicting_as_of_or_lineage_fail(self) -> None:
        stale = _example("player_rating")
        stale["provenance"]["freshness_checks"][0].update(
            {
                "source_updated_at": "2026-01-01T00:00:00Z",
                "limit_seconds": 60,
                "fresh": True,
            }
        )
        self._rejects("player_rating", stale)

        wrong_as_of = _example("player_rating")
        wrong_as_of["provenance"]["as_of"] = "2026-07-27T17:00:00Z"
        self._rejects("player_rating", wrong_as_of)

        wrong_lineage = _example("player_rating")
        wrong_lineage["provenance"]["lineage"]["artifact_sha256"] = "9" * 64
        self._rejects("player_rating", wrong_lineage)

    def test_changed_output_with_static_identity_hash_fails(self) -> None:
        player = _example("player_rating")
        player["display_name"] = "Changed without a new identity"
        with self.assertRaises(ValidationFailure):
            validate_output_payload("player_rating", player)

    def test_all_five_typed_unavailable_outputs_validate_without_reliability(self) -> None:
        statuses = ("missing", "stale", "conflict", "stale", "missing")
        for output, required_status in zip(
            validation.OUTPUT_SCHEMAS,
            statuses,
        ):
            with self.subTest(output=output, required_status=required_status):
                document = _unavailable(output, required_status)
                self.assertNotIn("reliability", document)
                evidence = validate_output_payload(output, document)
                self.assertTrue(
                    all(
                        item.status in {"passed", "not_applicable"}
                        for item in evidence
                    )
                )

    def test_embedded_research_only_partial_validates_without_canonical_fields(self) -> None:
        schema = json.loads(
            (CONTRACT_ROOT / "partial-draft-state.schema.json").read_text(
                encoding="utf-8"
            )
        )
        document = schema["examples"][0]
        for forbidden in (
            "interval_95",
            "partial_score_a",
            "partial_score_b",
            "standardized_map_win_probability_a",
            "reliability",
        ):
            self.assertNotIn(forbidden, document)
        evidence = validate_output_payload("partial_draft_state", document)
        self.assertTrue(evidence)

    def test_schema_invalid_semantic_counterexample_fails_report_build(self) -> None:
        original = validation._counterexample_fixture

        def structurally_invalid(invariant_id, positive):
            document = original(invariant_id, positive)
            if invariant_id == "player_interval_contains_posterior_mean":
                document["schema_forbidden_field"] = True
            return document

        with patch.object(
            validation,
            "_counterexample_fixture",
            side_effect=structurally_invalid,
        ):
            with self.assertRaisesRegex(
                ValidationFailure,
                "semantic counterexample is schema-invalid",
            ):
                run_five_output_validation()

    def test_forged_self_rehashed_empty_report_fails(self) -> None:
        report = run_five_output_validation()
        forged = replace(
            report,
            evidence=(),
            invariant_counts={},
            invariant_pass_count=0,
            mutation_pass_count=0,
            structural_pass_count=0,
            report_sha256="",
        )
        forged = replace(
            forged,
            report_sha256=canonical_sha256(forged.unsigned_payload()),
        )
        with self.assertRaises(ValidationFailure):
            verify_five_output_validation_report(forged)

    def test_identifier_authority_covers_all_ok_unavailable_and_research_paths(
        self,
    ) -> None:
        for output in validation.OUTPUT_SCHEMAS:
            with self.subTest(output=output, status="ok"):
                document = _example(output)
                document["season_id"] = "scryglass:season:invented-2099"
                document["calendar_year"] = 2099
                self._rejects(output, document)
            with self.subTest(output=output, status="unavailable"):
                document = _unavailable(output, "stale")
                document["provenance"]["input_snapshot_id"] = (
                    "scryglass:input:invented"
                )
                self._rejects(output, document)

        research_schema = json.loads(
            (CONTRACT_ROOT / "partial-draft-state.schema.json").read_text(
                encoding="utf-8"
            )
        )
        research = deepcopy(research_schema["examples"][0])
        research["archetype_extrapolation"]["support_gap"]["champion_id"] = (
            "riot:champion:invented"
        )
        self._rejects("partial_draft_state", research)

    def test_fixture_authority_rejects_synchronized_lineage_and_freshness_substitution(
        self,
    ) -> None:
        for output in validation.OUTPUT_SCHEMAS:
            document = _example(output)
            document["lineage"]["artifact_sha256"] = "8" * 64
            document["provenance"]["lineage"]["artifact_sha256"] = "8" * 64
            self._rejects(output, document)

        research_schema = json.loads(
            (CONTRACT_ROOT / "partial-draft-state.schema.json").read_text(
                encoding="utf-8"
            )
        )
        research = deepcopy(research_schema["examples"][0])
        check = research["provenance"]["freshness_checks"][0]
        check["source_updated_at"] = research["as_of"]
        check["limit_seconds"] = 10**9
        check["fresh"] = True
        self._rejects("partial_draft_state", research)

    def test_structural_diagnostics_are_exactly_frozen(self) -> None:
        original = validation._schema_error_evidence

        def forged_keyword(*args, **kwargs):
            evidence = list(original(*args, **kwargs))
            if evidence:
                evidence[0] = {**evidence[0], "keyword": "forged_keyword"}
            return tuple(evidence)

        with patch.object(
            validation,
            "_schema_error_evidence",
            side_effect=forged_keyword,
        ):
            with self.assertRaisesRegex(
                ValidationFailure,
                "diagnostic set/path/schema/keyword/reason mismatch",
            ):
                run_five_output_validation()
