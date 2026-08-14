"""Tests for the LCC atom bridge -> champion ontology seed builder (v1)."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from lol_kills.v2.champions.atoms.consume import AtomBridge, DEFAULT_ARTIFACT_PATH
from lol_kills.v2.champions.atoms.seed_ontology_v1 import (
    CANONICAL_AS_OF,
    PRIOR_PATCH,
    SOURCE_ID,
    SNAPSHOT_ID,
    build_seed,
    build_sources,
)
from lol_kills.v2.champions.catalog import (
    DEFAULT_ONTOLOGY_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH,
    load_champion_ontology,
)
from lol_kills.v2.champions.schema import DIMENSION_LABELS, REQUIRED_DIMENSIONS
from lol_kills.v2.patch_identity import CURRENT_PUBLIC_PATCH

ROOT = Path(__file__).resolve().parents[3]


class AtomSeedV1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bridge = AtomBridge.load(DEFAULT_ARTIFACT_PATH)
        cls.seed = json.loads(DEFAULT_ONTOLOGY_PATH.read_text(encoding="utf-8"))
        cls.sources = json.loads(DEFAULT_SOURCE_PATH.read_text(encoding="utf-8"))
        cls.ontology = load_champion_ontology(
            ontology_path=DEFAULT_ONTOLOGY_PATH,
            source_path=DEFAULT_SOURCE_PATH,
            review_path=DEFAULT_REVIEW_PATH,
        )

    def test_seed_covers_all_173_bridge_champions(self) -> None:
        by_id = {row["champion_id"] for row in self.seed["champions"]}
        self.assertEqual(len(by_id), 173)
        self.assertEqual(by_id, set(self.bridge.champion_ids()))

    def test_every_champion_has_26_15_profile_for_each_legal_role(self) -> None:
        for row in self.seed["champions"]:
            patch = row["patch_profiles"].get(PRIOR_PATCH)
            self.assertIsNotNone(patch, row["champion_id"])
            roles = patch["role_profiles"]
            # Curated champions can have narrower 26.15 coverage than their
            # authored role_legalities; every listed role must still be legal.
            self.assertTrue(roles)
            self.assertLessEqual(set(roles), set(row["role_legalities"]))
            if row["champion_id"] not in ("riot:champion:115", "riot:champion:101", "riot:champion:161", "riot:champion:518"):
                self.assertEqual(sorted(roles), sorted(row["role_legalities"]))
            for role, profile in roles.items():
                self.assertEqual(
                    sorted(profile["dimensions"]),
                    sorted(REQUIRED_DIMENSIONS),
                )
                self.assertEqual(profile["source_ids"], [SOURCE_ID])
                self.assertEqual(profile["residual"]["status"], "prior_only")

    def test_current_26_16_profile_is_present_for_each_legal_role(self) -> None:
        for row in self.seed["champions"]:
            patch = row["patch_profiles"].get(CURRENT_PUBLIC_PATCH)
            self.assertIsNotNone(patch, row["champion_id"])
            self.assertTrue(patch["role_profiles"])
            self.assertLessEqual(
                set(patch["role_profiles"]), set(row["role_legalities"])
            )

    def test_dimension_labels_are_complete_and_probabilities_bounded(self) -> None:
        for row in self.seed["champions"]:
            for role, profile in row["patch_profiles"][PRIOR_PATCH]["role_profiles"].items():
                for dimension, entry in profile["dimensions"].items():
                    self.assertEqual(
                        set(entry["labels"]),
                        set(DIMENSION_LABELS[dimension]),
                        (row["champion_id"], role, dimension),
                    )
                    for value in entry["labels"].values():
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)
                    for value in entry["uncertainty"].values():
                        self.assertGreaterEqual(value, 0.0)
                        self.assertLessEqual(value, 1.0)

    def test_unavailable_bridge_dimensions_become_uniform_max_uncertainty(self) -> None:
        # Find a champion with at least one unavailable dimension in the bridge.
        target = None
        for champion_id in self.bridge.champion_ids():
            prior = self.bridge.ontology_prior(champion_id)
            if any((d or {}).get("status") == "unavailable" for d in prior.values()):
                target = champion_id
                break
        self.assertIsNotNone(target, "expected at least one unavailable dimension")
        prior = self.bridge.ontology_prior(target)
        seed_row = next(r for r in self.seed["champions"] if r["champion_id"] == target)
        profile = seed_row["patch_profiles"][PRIOR_PATCH]["role_profiles"][
            seed_row["role_legalities"][0]
        ]
        for dimension, entry in prior.items():
            if (entry or {}).get("status") != "unavailable":
                continue
            dim = profile["dimensions"][dimension]
            n = len(DIMENSION_LABELS[dimension])
            self.assertEqual(set(dim["labels"].values()), {round(1.0 / n, 4)})
            self.assertEqual(set(dim["uncertainty"].values()), {1.0})

    def test_curated_champions_keep_26_14_profiles(self) -> None:
        for champion_id in ("riot:champion:115", "riot:champion:101"):
            row = next(r for r in self.seed["champions"] if r["champion_id"] == champion_id)
            self.assertIn("26.14", row["patch_profiles"])

    def test_sources_declare_atom_bridge_source_and_seed_references_it(self) -> None:
        source_ids = {row["source_id"] for row in self.sources["sources"]}
        self.assertIn(SOURCE_ID, source_ids)
        row = next(r for r in self.sources["sources"] if r["source_id"] == SOURCE_ID)
        self.assertEqual(row["kind"], "atom_bridge")
        self.assertEqual(row["publication_decision"], "private_pending_review")
        self.assertTrue(Path(ROOT / row["locator"]).exists())
        self.assertEqual(self.seed["as_of"], CANONICAL_AS_OF)
        self.assertEqual(self.seed["snapshot_id"], SNAPSHOT_ID)
        self.assertLessEqual(self.sources["as_of"], self.seed["as_of"])

    def test_builder_is_reproducible_and_fail_closed(self) -> None:
        rebuilt = build_seed(self.bridge, json.loads(DEFAULT_ONTOLOGY_PATH.read_text()))
        rebuilt_sources = build_sources(json.loads(DEFAULT_SOURCE_PATH.read_text()))
        self.assertEqual(rebuilt, self.seed)
        self.assertEqual(rebuilt_sources, self.sources)

    def test_ontology_loads_seed_and_prior_is_fail_closed(self) -> None:
        prior = self.ontology.build_archetype_prior(
            champion_id="riot:champion:266",
            role="top",
            patch_id=PRIOR_PATCH,
            league_id="LEC",
        )
        self.assertEqual(prior["resolved_patch_id"], PRIOR_PATCH)
        self.assertEqual(prior["fallback_level"], "none")
        self.assertFalse(prior["tier_list_eligible"])  # development-only: no appearances

    def test_cli_regenerates_identical_seed(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-m", "lol_kills.v2.champions.atoms.seed_ontology_v1",
             "--out-seed", str(ROOT / "data/lol/v2/champions/champion-ontology-seed.json"),
             "--out-sources", str(ROOT / "data/lol/v2/champions/champion-ontology-sources.json")],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
        after = json.loads(DEFAULT_ONTOLOGY_PATH.read_text(encoding="utf-8"))
        self.assertEqual(after, self.seed)


if __name__ == "__main__":
    unittest.main()
