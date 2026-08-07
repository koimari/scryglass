"""Unit tests for L3 champion ontology foundations."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from lol_kills.v2.champions import canonical_sha256
from lol_kills.v2.champions import fixtures as fixture_io
from lol_kills.v2.champions.catalog import ChampionOntologyError, load_champion_ontology
from lol_kills.v2.champions.paths import (
    DEFAULT_FIXTURE_PATH,
    DEFAULT_ONTOLOGY_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH,
)


class ChampionOntologyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = load_champion_ontology(
            ontology_path=DEFAULT_ONTOLOGY_PATH,
            source_path=DEFAULT_SOURCE_PATH,
            review_path=DEFAULT_REVIEW_PATH,
        )

    def test_as_of_earlier_than_ontology_snapshot_is_rejected(self) -> None:
        with self.assertRaises(ChampionOntologyError):
            load_champion_ontology(
                ontology_path=DEFAULT_ONTOLOGY_PATH,
                source_path=DEFAULT_SOURCE_PATH,
                review_path=DEFAULT_REVIEW_PATH,
                as_of="2026-07-27T12:15:00Z",
            )

    def test_as_of_later_than_ontology_snapshot_is_allowed(self) -> None:
        ontology = load_champion_ontology(
            ontology_path=DEFAULT_ONTOLOGY_PATH,
            source_path=DEFAULT_SOURCE_PATH,
            review_path=DEFAULT_REVIEW_PATH,
            as_of="2026-08-07T13:00:00Z",
        )
        prior = ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="mid",
            patch_id="26.14",
            league_id="LEC",
        )
        self.assertEqual(prior["review_as_of"], "2026-08-07T13:00:00Z")
        self.assertEqual(prior["as_of"], "2026-08-07T13:00:00Z")
        self.assertEqual(prior["ontology_as_of"], "2026-08-07T12:00:00Z")
        self.assertEqual(prior["source_as_of"], "2026-08-07T12:00:00Z")

    def _write_json(self, payload: dict[str, Any]) -> Path:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            suffix=".json",
            encoding="utf-8",
            dir=DEFAULT_ONTOLOGY_PATH.parent,
        ) as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            return Path(handle.name)

    def _write_jsonl(self, rows: list[dict[str, Any]]) -> Path:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            suffix=".jsonl",
            encoding="utf-8",
            dir=DEFAULT_REVIEW_PATH.parent,
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row))
                handle.write("\n")
            return Path(handle.name)

    def _source_for_review_path(self, review_path: Path, as_of: str | None = None) -> Path:
        repo_root = DEFAULT_ONTOLOGY_PATH.resolve().parents[4]
        with DEFAULT_SOURCE_PATH.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        for row in source_payload["sources"]:
            if row["kind"] == "manual_labels":
                row["locator"] = review_path.resolve().relative_to(repo_root).as_posix()
        if as_of is not None:
            source_payload["as_of"] = as_of
            for row in source_payload["sources"]:
                row["reviewed_at"] = as_of
        return self._write_json(source_payload)

    def _load_leave_one_out_fixture(self) -> dict[str, Any]:
        return fixture_io.leave_one_out_fixtures()[0]

    def _load_fixtures(self) -> list[dict[str, Any]]:
        return fixture_io.masked_champion_fixtures()

    def test_data_file_layout_is_repo_rooted(self) -> None:
        expected = DEFAULT_ONTOLOGY_PATH.exists()
        self.assertTrue(expected)

    def test_canonical_json_hash_is_deterministic(self) -> None:
        payload_a = {"b": 2, "a": 1}
        payload_b = {"a": 1, "b": 2}
        self.assertEqual(canonical_sha256(payload_a), canonical_sha256(payload_b))

    def test_load_ontology_contains_seeded_champions(self) -> None:
        champions = self.ontology.champion_ids()
        for champion_id in ("riot:champion:115", "riot:champion:101", "riot:champion:161"):
            self.assertIn(champion_id, champions)

    def test_alias_resolution_is_stable(self) -> None:
        self.assertEqual(self.ontology.resolve_by_alias("VelKoZ"), "riot:champion:161")
        self.assertIsNone(self.ontology.resolve_by_alias("missing-alias"))

    def test_missing_champion_fallback(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:99999",
            role="mid",
            patch_id="26.14",
            league_id="LEC",
        )
        self.assertEqual(prior["residual"]["status"], "missing_ontology")
        self.assertFalse(prior["tier_list_eligible"])
        self.assertIn("unknown_champion:riot:champion:99999", prior["issues"])
        self.assertTrue(all(value == 0.0 for value in prior["vector"]))

    def test_invalid_champion_id_shape_is_rejected(self) -> None:
        with self.assertRaises(ChampionOntologyError):
            self.ontology.build_archetype_prior(
                champion_id="invalid-id",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )

    def test_invalid_patch_syntax_is_rejected_before_missing_ontology(self) -> None:
        with self.assertRaises(ChampionOntologyError):
            self.ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="mid",
                patch_id="bad.patch",
                league_id="LEC",
            )

    def test_as_of_filtering_is_directional_and_review_time_sensitive(self) -> None:
        rows = [
            {
                "review_id": "r-asof-filter-001",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "artillery",
                "decision": "accepted",
                "confidence": 0.70,
                "reviewed_at": "2026-07-27T12:35:00Z",
            },
            {
                "review_id": "r-asof-filter-002",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "artillery",
                "decision": "accepted",
                "confidence": 0.88,
                "revision_of": "r-asof-filter-001",
                "reviewed_at": "2026-07-27T12:45:00Z",
            },
        ]
        review_path = self._write_jsonl(rows)
        review_source_path = self._source_for_review_path(
            review_path,
            as_of="2026-07-27T12:45:00Z",
        )
        try:
            with DEFAULT_ONTOLOGY_PATH.open("r", encoding="utf-8") as handle:
                base_payload = json.load(handle)
            base_payload["as_of"] = "2026-07-27T12:30:00Z"
            ontology_path = self._write_json(base_payload)

            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=ontology_path,
                    source_path=review_source_path,
                    review_path=review_path,
                    as_of="2026-07-27T12:00:00Z",
                )

            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=ontology_path,
                    source_path=review_source_path,
                    review_path=review_path,
                    as_of="2026-07-27T12:35:00Z",
                )

            cutoff_ontology = load_champion_ontology(
                ontology_path=ontology_path,
                source_path=review_source_path,
                review_path=review_path,
                as_of="2026-07-27T12:45:00Z",
            )
            cutoff_prior = cutoff_ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
            self.assertEqual(cutoff_prior["review_summary"]["damage_profile"]["review_count"], 2)
            self.assertEqual(cutoff_prior["review_summary"]["damage_profile"]["latest_review_id"], "r-asof-filter-002")
            self.assertEqual(cutoff_prior["review_as_of"], "2026-07-27T12:45:00Z")

            full_ontology = load_champion_ontology(
                ontology_path=ontology_path,
                source_path=review_source_path,
                review_path=review_path,
                as_of="2026-07-27T12:45:00Z",
            )
            full_prior = full_ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
            self.assertEqual(full_prior["review_summary"]["damage_profile"]["review_count"], 2)
            self.assertEqual(full_prior["review_summary"]["damage_profile"]["latest_review_id"], "r-asof-filter-002")
        finally:
            review_path.unlink()
            review_source_path.unlink()
            ontology_path.unlink()

    def test_source_snapshot_as_of_gates_review_rows(self) -> None:
        rows = [
            {
                "review_id": "r-source-gate-001",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "artillery",
                "decision": "accepted",
                "confidence": 0.70,
                "reviewed_at": "2026-07-27T12:30:00Z",
            },
            {
                "review_id": "r-source-gate-002",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "poke",
                "decision": "proposed",
                "confidence": 0.80,
                "revision_of": "r-source-gate-001",
                "reviewed_at": "2026-07-27T12:50:00Z",
            },
        ]
        review_path = self._write_jsonl(rows)
        review_source = self._source_for_review_path(
            review_path,
            as_of="2026-07-27T12:40:00Z",
        )
        try:
            ontology = load_champion_ontology(
                ontology_path=DEFAULT_ONTOLOGY_PATH,
                source_path=review_source,
                review_path=review_path,
                as_of="2026-08-07T13:00:00Z",
            )
            prior = ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
            self.assertEqual(prior["review_summary"]["damage_profile"]["review_count"], 1)
            self.assertEqual(
                prior["review_summary"]["damage_profile"]["latest_review_id"],
                "r-source-gate-001",
            )
            self.assertEqual(prior["review_as_of"], "2026-08-07T13:00:00Z")
        finally:
            review_path.unlink()
            review_source.unlink()

    def test_source_snapshot_as_of_future_is_rejected(self) -> None:
        source_payload = json.loads(DEFAULT_SOURCE_PATH.read_text(encoding="utf-8"))
        source_payload["as_of"] = "2026-07-27T13:20:00Z"
        source_path = self._write_json(source_payload)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=source_path,
                    review_path=DEFAULT_REVIEW_PATH,
                    as_of="2026-07-27T13:00:00Z",
                )
        finally:
            source_path.unlink()

    def test_empirical_snapshot_as_of_future_is_rejected(self) -> None:
        fixture = self._load_leave_one_out_fixture()
        empirical_payload = {
            "schema_version": fixture["schema_version"],
            "as_of": "2026-07-27T13:20:00Z",
            "cells": copy.deepcopy(fixture["cells"]),
        }
        empirical_path = self._write_json(empirical_payload)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=DEFAULT_SOURCE_PATH,
                    review_path=DEFAULT_REVIEW_PATH,
                    empirical_path=empirical_path,
                    as_of="2026-07-27T12:40:00Z",
                )
        finally:
            empirical_path.unlink()

    def test_zero_play_champion_uses_prior_only_residual(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:518",
            role="mid",
            patch_id="26.14",
            league_id="LEC",
        )
        self.assertEqual(prior["residual"]["status"], "prior_only")
        self.assertEqual(prior["residual"]["mean"], 0.0)
        self.assertGreaterEqual(prior["residual"]["sigma"], 2.0)
        self.assertFalse(prior["tier_list_eligible"])
        self.assertEqual(prior["tier_list_eligibility_reason"], "no_verified_appearances:LEC")
        self.assertTrue(prior["ontology_coverage"]["has_ontology"])
        self.assertTrue(prior["ontology_coverage"]["has_role_profile"])
        self.assertNotEqual(prior["vector"][0], 0.0)

    def test_older_patch_has_no_prior_profile(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="mid",
            patch_id="26.13",
            league_id="LEC",
        )
        self.assertFalse(prior["tier_list_eligible"])
        self.assertEqual(prior["residual"]["status"], "masked")
        self.assertIsNone(prior["resolved_patch_id"])
        self.assertEqual(prior["fallback_level"], "no_prior_patch")
        self.assertEqual(prior["tier_list_eligibility_reason"], "patch_no_prior:26.13:26.14")

    def test_tier_list_requires_exact_patch_context(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="mid",
            patch_id="26.16",
            league_id="LEC",
        )
        self.assertFalse(prior["tier_list_eligible"])
        self.assertIn(
            prior["tier_list_eligibility_reason"],
            {"tier_requires_exact_patch:26.15", "role_not_legal_for_patch"},
        )
        self.assertEqual(prior["fallback_level"], "future_patch_fallback")
        self.assertFalse(prior["exact_cell_appearances"]["is_exact_patch"])
        self.assertTrue(prior["exact_cell_appearances"]["is_exact_patch_role_legal"])
        self.assertFalse(prior["exact_cell_appearances"]["is_requested_patch_role_legal"])
        self.assertFalse(prior["exact_cell_appearances"]["is_requested_patch_role_legal_available"])
        self.assertEqual(prior["exact_cell_appearances"]["requested_patch_roles"], [])
        self.assertEqual(prior["exact_cell_appearances"]["exact_patch_roles"], ["bot", "mid"])
        self.assertNotEqual(
            prior["exact_cell_appearances"]["requested_patch_roles"],
            prior["exact_cell_appearances"]["exact_patch_roles"],
        )

    def test_archetype_prior_exposes_lineage_hashes(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="mid",
            patch_id="26.14",
            league_id="LEC",
        )
        self.assertIn("snapshot_hashes", prior)
        self.assertIn("artifact_ids", prior)
        self.assertEqual(prior["artifact_ids"]["review_snapshot_id"], prior["snapshot_hashes"]["reviews_sha256"])
        self.assertEqual(prior["artifact_ids"]["source_metadata_sha256"], prior["snapshot_hashes"]["source_metadata_sha256"])
        self.assertEqual(prior["artifact_ids"]["source_as_of"], "2026-08-07T12:00:00Z")
        self.assertIsNone(prior["artifact_ids"]["source_snapshot_id"])
        self.assertEqual(prior["artifact_ids"]["source_snapshot_status"], "pending_l1_snapshot")
        self.assertEqual(
            prior["source_dependency"]["source_snapshot_status"],
            "pending_l1_snapshot",
        )

    def test_review_disagreement_and_revision_trail(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:101",
            role="mid",
            patch_id="26.14",
            league_id="LEC",
        )
        effective_range_review = prior["review_summary"]["effective_range"]
        self.assertTrue(effective_range_review["disagreement"])
        self.assertEqual(effective_range_review["status"], "disputed")
        self.assertEqual(effective_range_review["review_count"], 2)
        self.assertEqual(len(effective_range_review["revisions"]), 2)

    def test_masked_evaluation_fixtures_drive_zero_and_unknown_coverage(self) -> None:
        fixtures = self._load_fixtures()
        self.assertGreaterEqual(len(fixtures), 2)
        for fixture in fixtures:
            expected_resolved_patch = (
                None
                if fixture["expected_residual_status"] == "missing_ontology"
                else (None if fixture["patch_id"] == "26.13" else fixture["patch_id"])
            )
            prior = self.ontology.build_archetype_prior(
                champion_id=fixture["champion_id"],
                role=fixture["role"],
                patch_id=fixture["patch_id"],
                league_id=fixture["league_id"],
            )
            self.assertEqual(prior["tier_list_eligible"], fixture["expected_tier_list_eligible"])
            self.assertEqual(prior["tier_list_eligibility_reason"], fixture["expected_reason"])
            self.assertEqual(prior["residual"]["status"], fixture["expected_residual_status"])
            self.assertEqual(prior["fallback_level"], fixture["expected_fallback_level"])
            self.assertEqual(prior["resolved_patch_id"], expected_resolved_patch)

    def test_transfer_fixture_distances_are_similar_but_not_equal(self) -> None:
        fixture = fixture_io.transfer_fixtures()[0]
        distances = fixture_io.build_transfer_distances(self.ontology, fixture)
        self.assertEqual(len(distances), 3)
        self.assertGreater(len(distances), 0)
        self.assertGreaterEqual(min(distances), fixture["min_pair_distance"])
        self.assertLessEqual(max(distances), fixture["max_pair_distance"])
        self.assertTrue(all(distance > 0.0 for distance in distances))

    def test_review_as_of_excludes_later_reviews(self) -> None:
        review_rows = [
            {
                "review_id": "r-ziggs-mid-dmg-001",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "l3_owner",
                "label": "artillery",
                "decision": "proposed",
                "confidence": 0.81,
                "reviewed_at": "2026-07-27T12:10:00Z",
            },
            {
                "review_id": "r-ziggs-mid-dmg-002",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "l3_reviewer",
                "label": "poke",
                "decision": "proposed",
                "confidence": 0.73,
                "revision_of": "r-ziggs-mid-dmg-001",
                "reviewed_at": "2026-07-27T12:15:00Z",
            },
            {
                "review_id": "r-ziggs-mid-dmg-003",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "l3_owner",
                "label": "artillery",
                "decision": "accepted",
                "confidence": 0.95,
                "revision_of": "r-ziggs-mid-dmg-002",
                "reviewed_at": "2026-07-27T12:20:00Z",
            },
        ]
        temp_path = self._write_jsonl(review_rows)
        temp_source = self._source_for_review_path(
            temp_path,
            as_of="2026-07-27T12:15:00Z",
        )
        with DEFAULT_ONTOLOGY_PATH.open("r", encoding="utf-8") as handle:
            ontology_payload = json.load(handle)
        ontology_payload["as_of"] = "2026-07-27T12:15:00Z"
        temp_ontology = self._write_json(ontology_payload)
        try:
            cut_off_ontology = load_champion_ontology(
                ontology_path=temp_ontology,
                source_path=temp_source,
                review_path=temp_path,
                as_of="2026-07-27T12:15:00Z",
            )
            prior = cut_off_ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
            review_summary = prior["review_summary"]["damage_profile"]
            self.assertEqual(review_summary["review_count"], 2)
            self.assertEqual(review_summary["status"], "disputed")
            self.assertIn(review_summary["latest_review_id"], {"r-ziggs-mid-dmg-002"})

            full_cutoff = load_champion_ontology(
                ontology_path=temp_ontology,
                source_path=temp_source,
                review_path=temp_path,
                as_of="2026-07-27T12:20:00Z",
            )
            full_prior = full_cutoff.build_archetype_prior(
                champion_id="riot:champion:115",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
            self.assertEqual(full_prior["review_summary"]["damage_profile"]["review_count"], 2)
            self.assertEqual(full_prior["review_summary"]["damage_profile"]["latest_review_id"], "r-ziggs-mid-dmg-002")
        finally:
            temp_path.unlink()
            temp_ontology.unlink()
            temp_source.unlink()

    def test_review_log_mutation_changes_lineage_hash(self) -> None:
        with DEFAULT_REVIEW_PATH.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]

        rows.append(
            {
                "review_id": "r-ziggs-mid-dmg-mutate",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "artillery",
                "decision": "accepted",
                "confidence": 0.92,
                "reviewed_at": "2026-07-27T12:35:00Z",
            }
        )
        mutation_path = self._write_jsonl(rows)
        mutation_source = self._source_for_review_path(mutation_path)
        try:
            mutated = load_champion_ontology(
                ontology_path=DEFAULT_ONTOLOGY_PATH,
                source_path=mutation_source,
                review_path=mutation_path,
            )
            self.assertNotEqual(mutated.review_snapshot_hash, self.ontology.review_snapshot_hash)
        finally:
            mutation_path.unlink()
            mutation_source.unlink()

    def test_future_patch_fallback_widens_uncertainty(self) -> None:
        exact = self.ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="mid",
            patch_id="26.15",
            league_id="LEC",
        )
        future = self.ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="mid",
            patch_id="26.16",
            league_id="LEC",
        )
        self.assertEqual(exact["fallback_level"], "none")
        self.assertEqual(future["resolved_patch_id"], "26.15")
        self.assertEqual(future["fallback_level"], "future_patch_fallback")
        self.assertGreater(future["residual"]["sigma"], exact["residual"]["sigma"])
        self.assertIn("patch_fallback_unseen:future_patch_fallback_v1", future["issues"])
        self.assertIn("review_effective_prior_v1", exact["dimension_rules"])
        self.assertIn("review_effective_prior_v1", future["dimension_rules"])
        self.assertIn("future_patch_fallback_v1", future["dimension_rules"])
        self.assertNotEqual(exact["residual"]["mean"], None)
        self.assertNotEqual(future["residual"]["mean"], None)
        self.assertEqual(future["residual"]["mean"], exact["residual"]["mean"])
        for dimension in exact["dimension_uncertainty"]:
            for label, exact_value in exact["dimension_uncertainty"][dimension].items():
                self.assertIn(
                    label,
                    future["dimension_uncertainty"][dimension],
                    f"missing future uncertainty label {dimension}.{label}",
                )
                self.assertGreaterEqual(
                    future["dimension_uncertainty"][dimension][label],
                    exact_value,
                    f"future uncertainty must not shrink for {dimension}.{label}",
                )

    def test_no_league_profile_substitution(self) -> None:
        with DEFAULT_ONTOLOGY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for champion in payload["champions"]:
            if champion["champion_id"] == "riot:champion:115":
                patch_profiles = champion["patch_profiles"]["26.14"]
                bot_profile = copy.deepcopy(patch_profiles["role_profiles"]["mid"])
                bot_profile["residual"] = {"status": "prior_only", "mean": 0.0, "sigma": 1.0, "observation_count": 0}
                bot_profile["verified_appearances"] = {"LEC": 6}
                patch_profiles["league_role_profiles"] = {"LEC": {"bot": bot_profile}}
                break

        temp_path = self._write_json(payload)
        try:
            temp_ontology = load_champion_ontology(
                ontology_path=temp_path,
                source_path=DEFAULT_SOURCE_PATH,
                review_path=DEFAULT_REVIEW_PATH,
            )
            prior = temp_ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="bot",
                patch_id="26.14",
                league_id="LEC",
            )
            self.assertEqual(prior["fallback_level"], "league_profile_only")
            self.assertEqual(prior["residual"]["status"], "masked")
            self.assertEqual(prior["vector"], [0.0 for _ in prior["vector"]])
            self.assertIn("missing_role_profile:bot", prior["issues"])
            self.assertIn("league_profile_only", prior["issues"])
            self.assertEqual(prior["tier_list_eligibility_reason"], "missing_role_profile:26.14")
        finally:
            temp_path.unlink()

    def test_exact_patch_role_legality_does_not_use_fallback_profile_roles(self) -> None:
        with DEFAULT_ONTOLOGY_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        for champion in payload["champions"]:
            if champion["champion_id"] == "riot:champion:115":
                champion["role_legalities"] = ["bot", "mid"]
                # Simulate a partial 26.15 profile that lacks the bot role
                # (authored coverage can be narrower than declared legality).
                patch_26_15 = champion["patch_profiles"].get("26.15", {})
                role_profiles = patch_26_15.get("role_profiles", {})
                if "bot" in role_profiles:
                    del role_profiles["bot"]
                break

        temp_path = self._write_json(payload)
        try:
            temp_ontology = load_champion_ontology(
                ontology_path=temp_path,
                source_path=DEFAULT_SOURCE_PATH,
                review_path=DEFAULT_REVIEW_PATH,
            )
            prior = temp_ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="bot",
                patch_id="26.16",
                league_id="LEC",
            )
            self.assertEqual(prior["fallback_level"], "future_patch_fallback")
            self.assertEqual(prior["resolved_patch_id"], "26.15")
            self.assertFalse(prior["exact_cell_appearances"]["is_role_legal"])
            self.assertIn("missing_role_profile:bot", prior["issues"])
            self.assertFalse(prior["exact_cell_appearances"]["is_eligible"])
        finally:
            temp_path.unlink()

    def test_exact_patch_role_legality_uses_only_patch_profile(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="bot",
            patch_id="26.14",
            league_id="LEC",
        )
        self.assertEqual(prior["resolved_patch_id"], "26.14")
        self.assertTrue(prior["exact_cell_appearances"]["is_exact_patch"])
        self.assertFalse(prior["exact_cell_appearances"]["is_role_legal"])
        self.assertEqual(prior["role_legal_exact_patch"], ["mid"])

    def test_review_coverage_separates_disputed_and_unreviewed(self) -> None:
        rows = [
            {
                "review_id": "r-coverage-accept",
                "champion_id": "riot:champion:101",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "artillery",
                "decision": "accepted",
                "confidence": 0.81,
                "reviewed_at": "2026-07-27T12:31:00Z",
            },
            {
                "review_id": "r-coverage-dispute",
                "champion_id": "riot:champion:101",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "engage",
                "reviewer": "qa",
                "label": "area",
                "decision": "proposed",
                "confidence": 0.75,
                "reviewed_at": "2026-07-27T12:32:00Z",
            },
        ]
        review_path = self._write_jsonl(rows)
        review_source = self._source_for_review_path(review_path)
        try:
            prior = load_champion_ontology(
                ontology_path=DEFAULT_ONTOLOGY_PATH,
                source_path=review_source,
                review_path=review_path,
            ).build_archetype_prior(
                champion_id="riot:champion:101",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
        finally:
            review_path.unlink()
            review_source.unlink()

        coverage = prior["review_coverage"]
        self.assertIn("damage_profile", coverage["accepted_dimensions"])
        self.assertIn("engage", coverage["disputed_dimensions"])
        self.assertGreater(len(coverage["unreviewed_dimensions"]), 0)
        self.assertTrue(coverage["accepted_dimensions"] and coverage["disputed_dimensions"])
        self.assertIn(
            "dimension_review:damage_profile:accepted:review_effective_prior_v1",
            prior["dimension_rules"],
        )
        self.assertIn(
            "dimension_review:engage:disputed:review_effective_prior_v1",
            prior["dimension_rules"],
        )
        self.assertIn(
            f"dimension_review:{coverage['unreviewed_dimensions'][0]}:unreviewed:review_effective_prior_v1",
            prior["dimension_rules"],
        )
        self.assertEqual(
            prior["dimension_uncertainty"]["damage_profile"],
            prior["authored_dimension_uncertainty"]["damage_profile"],
        )
        self.assertNotEqual(
            prior["dimension_uncertainty"]["engage"],
            prior["authored_dimension_uncertainty"]["engage"],
        )
        unreviewed_dimension = coverage["unreviewed_dimensions"][0]
        self.assertNotEqual(
            prior["dimension_uncertainty"][unreviewed_dimension],
            prior["authored_dimension_uncertainty"][unreviewed_dimension],
        )

    def test_accepted_review_alignment_is_preserved_by_author_match(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:518",
            role="mid",
            patch_id="26.14",
            league_id="LEC",
        )
        damage_review = prior["review_summary"]["damage_profile"]
        self.assertEqual(damage_review["status"], "accepted")
        self.assertEqual(damage_review["top_label"], "poke")

    def test_source_locator_must_be_repository_relative(self) -> None:
        with DEFAULT_SOURCE_PATH.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        for row in source_payload["sources"]:
            if row["kind"] == "manual_labels":
                row["locator"] = "/abs/path/to/manual.jsonl"
                break
        bad_path = self._write_json(source_payload)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=bad_path,
                    review_path=DEFAULT_REVIEW_PATH,
                )
        finally:
            bad_path.unlink()

    def test_source_locator_must_match_review_path(self) -> None:
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".jsonl",
            dir=str(Path(__file__).resolve().parents[4]),
            delete=False,
            encoding="utf-8",
        ) as handle:
            for line in DEFAULT_REVIEW_PATH.read_text(encoding="utf-8").splitlines():
                handle.write(f"{line}\n")
            mismatched = Path(handle.name)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=DEFAULT_SOURCE_PATH,
                    review_path=mismatched,
                )
        finally:
            mismatched.unlink()

    def test_source_locator_nonexistent_review_file_is_rejected(self) -> None:
        with DEFAULT_SOURCE_PATH.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        for row in source_payload["sources"]:
            if row["kind"] == "manual_labels":
                row["locator"] = "data/lol/v2/champions/does-not-exist.jsonl"
                break
        bad_path = self._write_json(source_payload)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=bad_path,
                    review_path=DEFAULT_REVIEW_PATH,
                )
        finally:
            bad_path.unlink()

    def test_source_locator_rejects_traversal(self) -> None:
        with DEFAULT_SOURCE_PATH.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        for row in source_payload["sources"]:
            if row["kind"] == "manual_labels":
                row["locator"] = "data/../etc/secrets.jsonl"
                break
        bad_path = self._write_json(source_payload)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=bad_path,
                    review_path=DEFAULT_REVIEW_PATH,
                )
        finally:
            bad_path.unlink()

    def test_ziggs_xerath_velkoz_masked_prior_only_coverage(self) -> None:
        fixtures = [
            "riot:champion:115",
            "riot:champion:101",
            "riot:champion:161",
        ]
        for champion_id in fixtures:
            prior = self.ontology.build_archetype_prior(
                champion_id=champion_id,
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
            self.assertEqual(prior["ontology_coverage"]["has_ontology"], True)
            self.assertEqual(prior["residual"]["status"], "prior_only")
            self.assertEqual(prior["residual"]["mean"], 0.0)
            self.assertGreaterEqual(prior["residual"]["sigma"], 2.0)
            self.assertEqual(prior["residual_evidence"]["observation_count"], 0)
            self.assertEqual(prior["exact_cell_appearances"]["is_exact_patch"], True)
            self.assertTrue(prior["dimension_uncertainty"])

    def test_as_of_review_cutoff_alters_artefact_hash(self) -> None:
        with DEFAULT_REVIEW_PATH.open("r", encoding="utf-8") as handle:
            review_rows = [json.loads(line) for line in handle if line.strip()]

        with DEFAULT_ONTOLOGY_PATH.open("r", encoding="utf-8") as handle:
            ontology_payload = json.load(handle)
        ontology_payload["as_of"] = "2026-07-27T12:15:00Z"
        temp_ontology = self._write_json(ontology_payload)
        cutoff = self._write_jsonl(review_rows)
        cutoff_source = self._source_for_review_path(
            cutoff,
            as_of="2026-07-27T12:15:00Z",
        )
        try:
            cut_off_ontology = load_champion_ontology(
                ontology_path=temp_ontology,
                source_path=cutoff_source,
                review_path=cutoff,
                as_of="2026-07-27T12:15:00Z",
            )
            prior = cut_off_ontology.build_archetype_prior(
                champion_id="riot:champion:115",
                role="mid",
                patch_id="26.14",
                league_id="LEC",
            )
            self.assertEqual(prior["review_as_of"], "2026-07-27T12:15:00Z")
            self.assertNotEqual(cut_off_ontology.review_snapshot_hash, self.ontology.review_snapshot_hash)
            self.assertEqual(prior["artifact_ids"]["review_snapshot_id"], prior["snapshot_hashes"]["reviews_sha256"])
            self.assertLess(prior["review_summary"]["damage_profile"]["review_count"], 3)
            self.assertEqual(prior["review_summary"]["damage_profile"]["status"], "disputed")
        finally:
            cutoff.unlink()
            temp_ontology.unlink()
            cutoff_source.unlink()

    def test_source_decisions_do_not_imply_public_approval(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:115",
            role="mid",
            patch_id="26.14",
            league_id="LEC",
        )
        decisions = prior["source_payload"]["decision"]
        self.assertIn("source:manual-labels-v2", decisions)
        self.assertEqual(decisions["source:manual-labels-v2"], "private_pending_review")
        self.assertIn("source:riot-dd-26.14", decisions)
        self.assertEqual(decisions["source:riot-dd-26.14"], "private_pending_review")

    def test_transfer_fixture_is_executable_without_empirical_residual(self) -> None:
        fixture = fixture_io.transfer_fixtures()[0]
        for champion_id in fixture["champion_ids"]:
            prior = self.ontology.build_archetype_prior(
                champion_id=champion_id,
                role=fixture["role"],
                patch_id=fixture["patch_id"],
                league_id=fixture["league_id"],
            )
            self.assertEqual(prior["exact_cell_appearances"]["is_exact_patch"], True)
            self.assertEqual(prior["residual"]["status"], "prior_only")
            self.assertEqual(prior["residual"]["observation_count"], 0)
            self.assertGreaterEqual(prior["residual"]["sigma"], 2.0)

    def test_leave_one_out_evaluation_is_executable_on_synthetic_contract_data(self) -> None:
        fixture = self._load_leave_one_out_fixture()
        result = fixture_io.run_leave_one_out_prediction_evaluation(
            fixture,
            ontology_path=DEFAULT_ONTOLOGY_PATH,
            source_path=DEFAULT_SOURCE_PATH,
            review_path=DEFAULT_REVIEW_PATH,
        
        as_of="2026-08-07T12:00:00Z",)

        self.assertEqual(result["status"], "synthetic_contract_only")
        self.assertTrue(result["synthetic_contract_only"])
        self.assertEqual(result["result_version"], "leave_one_out_v2")
        self.assertIn("result_hash", result)
        self.assertIn("config_hash", result)
        self.assertEqual(result["coverage"]["requested_holdouts"], 3)
        self.assertEqual(result["coverage"]["covered_holdouts"], 3)
        self.assertEqual(
            sorted(result["coverage"]["covered_champions"]),
            sorted(fixture["target_champion_ids"]),
        )
        self.assertEqual(result["coverage"]["missing_champions"], [])
        self.assertFalse(result["coverage"]["covered_champions"] == [])
        self.assertIn("input_hash", result)
        self.assertIn("fixture_hash", result)
        self.assertIn("transfer_mse", result["metrics"])
        self.assertIn("baseline_mse", result["metrics"])
        self.assertTrue(any(
            entry["transfer_prediction"] != entry["baseline_prediction"]
            for entry in result["per_holdout"]
        ))
        self.assertEqual(
            result["dependency_lineage"]["c0_contract_hash"],
            fixture["c0_contract_hash"],
        )
        self.assertIn("ontology_snapshot_sha256", result["dependency_lineage"])
        self.assertIn("source_metadata_sha256", result["dependency_lineage"])
        self.assertIn("manual_review_snapshot_sha256", result["dependency_lineage"])
        self.assertIn("empirical_snapshot_sha256", result["dependency_lineage"])

    def test_leave_one_out_targets_are_held_out_from_predictions(self) -> None:
        fixture = self._load_leave_one_out_fixture()
        result = fixture_io.run_leave_one_out_prediction_evaluation(fixture,
as_of="2026-08-07T12:00:00Z")
        self.assertTrue(result["per_holdout"])

        for entry in result["per_holdout"]:
            self.assertFalse(entry["training_includes_holdout"])
            self.assertGreater(entry["candidate_count"], 1)
            self.assertEqual(len(entry["candidate_ids"]), entry["candidate_count"])

    def test_leave_one_out_invalid_contract_hash_is_rejected(self) -> None:
        fixture = copy.deepcopy(self._load_leave_one_out_fixture())
        fixture["c0_contract_hash"] = "bad_contract"
        with self.assertRaises(ChampionOntologyError):
            fixture_io.run_leave_one_out_prediction_evaluation(fixture,
as_of="2026-08-07T12:00:00Z")

    def test_leave_one_out_targets_are_held_out_from_all_cells(self) -> None:
        fixture = copy.deepcopy(self._load_leave_one_out_fixture())
        holdout_id = fixture["target_champion_ids"][0]
        fixture["target_champion_ids"] = [holdout_id]
        fixture["cells"].extend(
            [
                {
                    "champion_id": holdout_id,
                    "patch_id": "26.15",
                    "role": "top",
                    "league_id": "LEC",
                    "residual": {
                        "status": "observed",
                        "mean": 0.04,
                        "sigma": 0.73,
                        "observation_count": 7,
                    },
                    "verified_appearance_count": 7,
                },
                {
                    "champion_id": holdout_id,
                    "patch_id": "26.14",
                    "role": "bot",
                    "league_id": "LCK",
                    "residual": {
                        "status": "observed",
                        "mean": 0.01,
                        "sigma": 0.71,
                        "observation_count": 5,
                    },
                    "verified_appearance_count": 5,
                },
            ]
        )

        result = fixture_io.run_leave_one_out_prediction_evaluation(fixture,
as_of="2026-08-07T12:00:00Z")
        entry = result["per_holdout"][0]

        self.assertEqual(entry["holdout_champion_id"], holdout_id)
        self.assertFalse(entry["training_includes_holdout"])
        self.assertEqual(entry["candidate_count"], 2)
        self.assertNotIn(holdout_id, entry["candidate_ids"])
        self.assertEqual(
            entry["training_cell_count"],
            len([row for row in fixture["cells"] if row["champion_id"] != holdout_id]),
        )

    def test_leave_one_out_target_only_affects_scores_not_predictions(self) -> None:
        fixture = self._load_leave_one_out_fixture()
        baseline_result = fixture_io.run_leave_one_out_prediction_evaluation(fixture,
as_of="2026-08-07T12:00:00Z")

        mutated = copy.deepcopy(fixture)
        holdout_id = fixture["target_champion_ids"][0]
        for row in mutated["cells"]:
            if row["champion_id"] == holdout_id:
                row["residual"]["mean"] += 0.41
                break
        mutated_result = fixture_io.run_leave_one_out_prediction_evaluation(mutated,
as_of="2026-08-07T12:00:00Z")

        baseline_entry = next(
            item for item in baseline_result["per_holdout"] if item["holdout_champion_id"] == holdout_id
        )
        mutated_entry = next(
            item for item in mutated_result["per_holdout"] if item["holdout_champion_id"] == holdout_id
        )
        self.assertEqual(
            baseline_entry["transfer_prediction"],
            mutated_entry["transfer_prediction"],
        )
        self.assertEqual(
            baseline_entry["baseline_prediction"],
            mutated_entry["baseline_prediction"],
        )
        self.assertNotEqual(
            baseline_entry["transfer_squared_error"],
            mutated_entry["transfer_squared_error"],
        )
        self.assertNotEqual(
            baseline_entry["baseline_squared_error"],
            mutated_entry["baseline_squared_error"],
        )

    def test_leave_one_out_configuration_and_input_hashes_change_on_mutation(self) -> None:
        fixture = self._load_leave_one_out_fixture()
        baseline_result = fixture_io.run_leave_one_out_prediction_evaluation(fixture,
as_of="2026-08-07T12:00:00Z")

        mutated = copy.deepcopy(fixture)
        mutated["cells"][1]["residual"]["mean"] += 0.02
        mutated_result = fixture_io.run_leave_one_out_prediction_evaluation(mutated,
as_of="2026-08-07T12:00:00Z")
        self.assertNotEqual(baseline_result["input_hash"], mutated_result["input_hash"])
        self.assertNotEqual(baseline_result["fixture_hash"], mutated_result["fixture_hash"])
        self.assertNotEqual(baseline_result["result_hash"], mutated_result["result_hash"])

    def test_leave_one_out_dependency_lineage_tracks_mutations(self) -> None:
        fixture = self._load_leave_one_out_fixture()
        baseline = fixture_io.run_leave_one_out_prediction_evaluation(fixture,
as_of="2026-08-07T12:00:00Z")

        with DEFAULT_ONTOLOGY_PATH.open("r", encoding="utf-8") as handle:
            ontology_payload = json.load(handle)
        ontology_payload["snapshot_id"] = f"{ontology_payload['snapshot_id']}-mut"
        mutated_ontology = self._write_json(ontology_payload)
        try:
            ontology_mutated = fixture_io.run_leave_one_out_prediction_evaluation(
                fixture,
                ontology_path=mutated_ontology,
            
            as_of="2026-08-07T12:00:00Z",)
            self.assertNotEqual(
                baseline["dependency_lineage"]["ontology_snapshot_sha256"],
                ontology_mutated["dependency_lineage"]["ontology_snapshot_sha256"],
            )
            self.assertNotEqual(
                baseline["result_hash"],
                ontology_mutated["result_hash"],
            )
        finally:
            mutated_ontology.unlink()

        with DEFAULT_REVIEW_PATH.open("r", encoding="utf-8") as handle:
            review_rows = [json.loads(line) for line in handle if line.strip()]
        review_rows.append(
            {
                "review_id": "r-dep-review-mutate",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "artillery",
                "decision": "accepted",
                "confidence": 0.91,
                "reviewed_at": "2026-07-27T12:40:00Z",
            }
        )
        mutated_review_path = self._write_jsonl(review_rows)
        mutated_review_source = self._source_for_review_path(mutated_review_path)
        try:
            review_mutated = fixture_io.run_leave_one_out_prediction_evaluation(
                fixture,
                review_path=mutated_review_path,
                source_path=mutated_review_source,
            
            as_of="2026-08-07T12:00:00Z",)
            self.assertNotEqual(
                baseline["dependency_lineage"]["manual_review_snapshot_sha256"],
                review_mutated["dependency_lineage"]["manual_review_snapshot_sha256"],
            )
            self.assertNotEqual(
                baseline["result_hash"],
                review_mutated["result_hash"],
            )
        finally:
            mutated_review_path.unlink()
            mutated_review_source.unlink()

        with DEFAULT_SOURCE_PATH.open("r", encoding="utf-8") as handle:
            source_payload = json.load(handle)
        for source_row in source_payload["sources"]:
            if source_row["kind"] == "manual_labels":
                source_row["publication_decision"] = (
                    "private_pending_review"
                    if source_row["publication_decision"] == "private"
                    else "private"
                )
                break
        mutated_source = self._write_json(source_payload)
        try:
            source_mutated = fixture_io.run_leave_one_out_prediction_evaluation(
                fixture,
                source_path=mutated_source,
            
            as_of="2026-08-07T12:00:00Z",)
            self.assertNotEqual(
                baseline["dependency_lineage"]["source_metadata_sha256"],
                source_mutated["dependency_lineage"]["source_metadata_sha256"],
            )
            self.assertNotEqual(
                baseline["result_hash"],
                source_mutated["result_hash"],
            )
        finally:
            mutated_source.unlink()

        empirical_mutated = copy.deepcopy(fixture)
        empirical_mutated["cells"][0]["residual"]["mean"] += 0.21
        empirical_mutated["cells"][0]["residual"]["observation_count"] += 1
        empirical_result = fixture_io.run_leave_one_out_prediction_evaluation(empirical_mutated,
as_of="2026-08-07T12:00:00Z")
        self.assertNotEqual(
            baseline["dependency_lineage"]["empirical_snapshot_sha256"],
            empirical_result["dependency_lineage"]["empirical_snapshot_sha256"],
        )
        self.assertNotEqual(
            baseline["result_hash"],
            empirical_result["result_hash"],
        )

    def test_leave_one_out_leakage_config_changes_prediction_and_hash(self) -> None:
        fixture = self._load_leave_one_out_fixture()
        no_leak = fixture_io.run_leave_one_out_prediction_evaluation(fixture,
as_of="2026-08-07T12:00:00Z")
        with_leak = fixture_io.run_leave_one_out_prediction_evaluation(
            fixture,
            allow_training_target=True,
        
        as_of="2026-08-07T12:00:00Z",)

        self.assertNotEqual(
            no_leak["config_hash"],
            with_leak["config_hash"],
        )
        self.assertNotEqual(no_leak["result_hash"], with_leak["result_hash"])
        self.assertEqual(with_leak["status"], "invalid_no_score")
        self.assertIsNone(with_leak["metrics"]["transfer_mse"])
        self.assertIsNone(with_leak["metrics"]["baseline_mse"])
        self.assertEqual(with_leak["coverage"]["covered_holdouts"], 0)
        self.assertTrue(all(item["status"] == "invalid_no_score" for item in with_leak["per_holdout"]))
        self.assertTrue(all(item["candidate_count"] == 0 for item in with_leak["per_holdout"]))

    def test_invalid_review_labels_are_rejected(self) -> None:
        rows = [
            {
                "review_id": "r-invalid-label",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "qa",
                "label": "not-a-label",
                "decision": "accepted",
                "confidence": 0.55,
                "reviewed_at": "2026-07-27T12:40:00Z",
            }
        ]
        review_path = self._write_jsonl(rows)
        review_source = self._source_for_review_path(review_path)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=review_source,
                    review_path=review_path,
                )
        finally:
            review_path.unlink()
            review_source.unlink()

    def test_invalid_revision_chains_are_rejected(self) -> None:
        rows = [
            {
                "review_id": "r-invalid-revision-001",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "mid",
                "dimension": "damage_profile",
                "reviewer": "l3_owner",
                "label": "artillery",
                "decision": "proposed",
                "confidence": 0.80,
                "reviewed_at": "2026-07-27T12:10:00Z",
            },
            {
                "review_id": "r-invalid-revision-002",
                "champion_id": "riot:champion:115",
                "patch_id": "26.14",
                "role": "bot",
                "dimension": "damage_profile",
                "reviewer": "l3_owner",
                "label": "artillery",
                "decision": "proposed",
                "confidence": 0.70,
                "revision_of": "r-invalid-revision-001",
                "reviewed_at": "2026-07-27T12:11:00Z",
            },
        ]
        review_path = self._write_jsonl(rows)
        review_source = self._source_for_review_path(review_path)
        try:
            with self.assertRaises(ChampionOntologyError):
                load_champion_ontology(
                    ontology_path=DEFAULT_ONTOLOGY_PATH,
                    source_path=review_source,
                    review_path=review_path,
                )
        finally:
            review_path.unlink()
            review_source.unlink()

    def test_schema_files_parse(self) -> None:
        for path in (DEFAULT_ONTOLOGY_PATH, DEFAULT_SOURCE_PATH, DEFAULT_REVIEW_PATH, DEFAULT_FIXTURE_PATH):
            self.assertTrue(path.exists(), f"missing L3 file: {path}")
            if path.suffix == ".json":
                with path.open("r", encoding="utf-8") as handle:
                    json.load(handle)

    def test_invalid_champion_payload_fails_fast(self) -> None:
        with self.assertRaises(ChampionOntologyError):
            load_champion_ontology(
                ontology_path=Path("/does-not-exist.json"),
                source_path=DEFAULT_SOURCE_PATH,
                review_path=DEFAULT_REVIEW_PATH,
            )
