"""Focused contract tests for the isolated Champion Representation v2."""

from __future__ import annotations

import copy
import unittest

from lol_kills.v2.champions.catalog import canonical_sha256, load_champion_ontology
from lol_kills.v2.champions.paths import (
    DEFAULT_ONTOLOGY_PATH,
    DEFAULT_REVIEW_PATH,
    DEFAULT_SOURCE_PATH,
)
from lol_kills.v2.champions.representation_v2 import (
    KIT_SEMANTIC_FEATURE_ORDER,
    RESPONSE_FEATURE_ORDER,
    RESPONSE_SNAPSHOT_SCHEMA_ID,
    ROLE_ORDER,
    ChampionRepresentationError,
    build_champion_representation_v2,
    load_representation_contract,
    validate_representation_sha256,
)
from lol_kills.v2.champions.schema import DIMENSION_LABEL_ORDER


class ChampionRepresentationV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.ontology = load_champion_ontology(
            ontology_path=DEFAULT_ONTOLOGY_PATH,
            source_path=DEFAULT_SOURCE_PATH,
            review_path=DEFAULT_REVIEW_PATH,
        )
        self.request = {
            "ontology": self.ontology,
            "champion_id": "riot:champion:115",
            "patch_id": "26.14",
            "league_id": "LEC",
            "requested_as_of": "2026-08-08T00:00:00Z",
        }

    @staticmethod
    def _response_cell(
        *,
        champion_id: str = "riot:champion:115",
        role: str = "mid",
        patch_id: str = "26.14",
        league_id: str = "LEC",
        value: float = 0.25,
    ) -> dict:
        return {
            "champion_id": champion_id,
            "patch_id": patch_id,
            "role": role,
            "league_id": league_id,
            "status": "observed",
            "values": [value],
            "uncertainty": {"sigma": 0.15},
            "evidence": {
                "observation_count": 12,
                "max_event_at": "2026-07-27T20:00:00Z",
            },
        }

    @classmethod
    def _response_snapshot(cls, cells: list[dict] | None = None) -> dict:
        snapshot = {
            "schema_id": RESPONSE_SNAPSHOT_SCHEMA_ID,
            "snapshot_id": "response-test-v1",
            "snapshot_as_of": "2026-07-27T21:00:00Z",
            "feature_order": list(RESPONSE_FEATURE_ORDER),
            "source_sha256": "a" * 64,
            "cells": cells if cells is not None else [cls._response_cell()],
        }
        snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
        return snapshot

    @staticmethod
    def _rehash_response(snapshot: dict) -> dict:
        snapshot.pop("snapshot_sha256", None)
        snapshot["snapshot_sha256"] = canonical_sha256(snapshot)
        return snapshot

    def _build(self, **overrides: object) -> dict:
        return build_champion_representation_v2(**{**self.request, **overrides})

    def test_contract_pins_five_roles_and_48_semantic_features(self) -> None:
        contract = load_representation_contract()
        self.assertEqual(tuple(contract["role_order"]), ROLE_ORDER)
        self.assertEqual(len(contract["kit_semantic_feature_order"]), 48)
        self.assertEqual(
            tuple(contract["kit_semantic_feature_order"]),
            KIT_SEMANTIC_FEATURE_ORDER,
        )
        self.assertEqual(
            KIT_SEMANTIC_FEATURE_ORDER,
            tuple(f"{dimension}.{label}" for dimension, label in DIMENSION_LABEL_ORDER),
        )
        self.assertEqual(
            contract["embedding"]["activation_status"],
            "disabled_pending_independent_evidence_registry",
        )
        self.assertFalse(
            contract["response"]["content_addressing_confers_predictive_authority"]
        )

    def test_hash_and_role_output_are_deterministic(self) -> None:
        first = self._build()
        second = self._build()
        self.assertEqual(first, second)
        self.assertEqual([row["role"] for row in first["roles"]], list(ROLE_ORDER))
        self.assertTrue(validate_representation_sha256(first))

    def test_semantic_layer_matches_exact_requested_role_vector(self) -> None:
        result = self._build()
        by_role = {row["role"]: row for row in result["roles"]}
        for role in ROLE_ORDER:
            expected = self.ontology.build_feature_vector(
                champion_id=self.request["champion_id"],
                role=role,
                patch_id=self.request["patch_id"],
                league_id=self.request["league_id"],
            )
            semantic = by_role[role]["kit_semantic"]
            self.assertEqual(semantic["feature_order"], list(KIT_SEMANTIC_FEATURE_ORDER))
            if expected["ontology_coverage"]["has_role_profile"]:
                self.assertEqual(semantic["values"], expected["vector"])
                self.assertEqual(semantic["status"], "exact")
                self.assertTrue(semantic["exact_requested_patch"])
            else:
                self.assertEqual(semantic["values"], [0.0] * 48)
                self.assertFalse(semantic["available"])

    def test_future_patch_semantic_is_labeled_fallback_prior(self) -> None:
        exact = self._build()
        future = self._build(patch_id="26.16")
        exact_mid = exact["roles"][2]["kit_semantic"]
        future_mid = future["roles"][2]["kit_semantic"]
        self.assertTrue(future_mid["available"])
        self.assertEqual(future_mid["resolved_patch_id"], "26.15")
        self.assertEqual(future_mid["status"], "fallback_prior")
        self.assertFalse(future_mid["exact_requested_patch"])
        self.assertTrue(
            all(
                future_value >= exact_value
                for future_value, exact_value in zip(
                    future_mid["uncertainty"],
                    exact_mid["uncertainty"],
                )
                if future_value is not None and exact_value is not None
            )
        )

    def test_missing_champion_and_missing_roles_never_borrow(self) -> None:
        missing = self._build(champion_id="riot:champion:99999")
        for row in missing["roles"]:
            self.assertFalse(row["kit_semantic"]["available"])
            self.assertEqual(row["kit_semantic"]["values"], [0.0] * 48)
            self.assertFalse(row["response"]["available"])
            self.assertEqual(row["response"]["values"], [0.0])
            self.assertFalse(row["learned_residual_embedding"]["available"])

        known = self._build()
        by_role = {row["role"]: row for row in known["roles"]}
        self.assertTrue(by_role["mid"]["kit_semantic"]["available"])
        for role in ("top", "jungle", "bot", "support"):
            self.assertFalse(by_role[role]["kit_semantic"]["available"])
            self.assertEqual(by_role[role]["kit_semantic"]["values"], [0.0] * 48)

    def test_zero_play_is_not_observed_zero(self) -> None:
        result = self._build(response_snapshot=self._response_snapshot(cells=[]))
        mid = result["roles"][2]
        self.assertTrue(mid["kit_semantic"]["available"])
        self.assertEqual(mid["response"]["status"], "unavailable")
        self.assertFalse(mid["response"]["available"])
        self.assertEqual(mid["response"]["values"], [0.0])
        self.assertEqual(mid["response"]["evidence"]["observation_count"], 0)

        absent = self._build()
        absent_lineage = absent["lineage"]["response"]
        self.assertFalse(absent_lineage["content_addressed"])
        self.assertEqual(
            absent_lineage["predictive_authority_status"], "unavailable"
        )
        self.assertFalse(absent_lineage["predictive_eligible"])

    def test_response_requires_hash_and_rejects_post_hash_mutation(self) -> None:
        missing_hash = self._response_snapshot()
        missing_hash.pop("snapshot_sha256")
        with self.assertRaisesRegex(ChampionRepresentationError, "snapshot_sha256"):
            self._build(response_snapshot=missing_hash)

        null_hash = self._response_snapshot()
        null_hash["snapshot_sha256"] = None
        with self.assertRaisesRegex(ChampionRepresentationError, "snapshot_sha256"):
            self._build(response_snapshot=null_hash)

        mutated = self._response_snapshot()
        mutated["cells"][0]["values"] = [0.99]
        with self.assertRaisesRegex(ChampionRepresentationError, "does not match"):
            self._build(response_snapshot=mutated)

    def test_response_is_blocked_without_same_role_semantic_anchor(self) -> None:
        top_snapshot = self._response_snapshot([self._response_cell(role="top")])
        missing_role = self._build(response_snapshot=top_snapshot)
        top_response = missing_role["roles"][0]["response"]
        self.assertEqual(top_response["status"], "blocked_missing_semantic_anchor")
        self.assertFalse(top_response["available"])
        self.assertFalse(top_response["predictive_eligible"])

        unknown_cell = self._response_cell(champion_id="riot:champion:99999")
        unknown_snapshot = self._response_snapshot([unknown_cell])
        unknown = self._build(
            champion_id="riot:champion:99999",
            response_snapshot=unknown_snapshot,
        )
        unknown_mid = unknown["roles"][2]["response"]
        self.assertEqual(
            unknown_mid["status"], "blocked_missing_semantic_anchor"
        )
        self.assertFalse(unknown_mid["available"])
        self.assertEqual(unknown_mid["values"], [0.0])

    def test_layers_are_independent(self) -> None:
        without = self._build()
        with_response = self._build(
            response_snapshot=self._response_snapshot(),
        )
        self.assertEqual(
            [row["kit_semantic"] for row in without["roles"]],
            [row["kit_semantic"] for row in with_response["roles"]],
        )
        self.assertEqual(
            [row["learned_residual_embedding"] for row in without["roles"]],
            [row["learned_residual_embedding"] for row in with_response["roles"]],
        )
        self.assertFalse(without["roles"][2]["response"]["available"])
        self.assertTrue(with_response["roles"][2]["response"]["available"])

    def test_exact_cell_matching_does_not_substitute(self) -> None:
        for cell in (
            self._response_cell(role="top"),
            self._response_cell(patch_id="26.13"),
            self._response_cell(league_id="LCK"),
        ):
            result = self._build(response_snapshot=self._response_snapshot([cell]))
            self.assertFalse(result["roles"][2]["response"]["available"])

    def test_duplicate_exact_cells_are_rejected(self) -> None:
        cell = self._response_cell()
        with self.assertRaisesRegex(ChampionRepresentationError, "duplicate response"):
            self._build(
                response_snapshot=self._response_snapshot(
                    [cell, copy.deepcopy(cell)]
                )
            )

    def test_response_cutoff_and_observation_gates_fail_closed(self) -> None:
        future_event = self._response_snapshot()
        future_event["cells"][0]["evidence"]["max_event_at"] = "2026-08-09T22:00:00Z"
        self._rehash_response(future_event)
        with self.assertRaisesRegex(ChampionRepresentationError, "max_event_at"):
            self._build(response_snapshot=future_event)

        future_snapshot = self._response_snapshot()
        future_snapshot["snapshot_as_of"] = "2026-08-09T00:00:00Z"
        self._rehash_response(future_snapshot)
        with self.assertRaisesRegex(ChampionRepresentationError, "newer"):
            self._build(response_snapshot=future_snapshot)

        zero_count = self._response_snapshot()
        zero_count["cells"][0]["evidence"]["observation_count"] = 0
        self._rehash_response(zero_count)
        with self.assertRaisesRegex(ChampionRepresentationError, "greater than zero"):
            self._build(response_snapshot=zero_count)

        boolean_value = self._response_snapshot()
        boolean_value["cells"][0]["values"] = [True]
        self._rehash_response(boolean_value)
        with self.assertRaisesRegex(ChampionRepresentationError, "not boolean"):
            self._build(response_snapshot=boolean_value)

        zero_sigma = self._response_snapshot()
        zero_sigma["cells"][0]["uncertainty"]["sigma"] = 0.0
        self._rehash_response(zero_sigma)
        with self.assertRaisesRegex(ChampionRepresentationError, "greater than zero"):
            self._build(response_snapshot=zero_sigma)

    def test_observed_response_is_content_addressed_but_not_predictive(self) -> None:
        result = self._build(response_snapshot=self._response_snapshot())
        response = result["roles"][2]["response"]
        lineage = result["lineage"]["response"]
        self.assertTrue(response["available"])
        self.assertEqual(response["status"], "observed")
        self.assertTrue(response["content_addressed"])
        self.assertEqual(response["predictive_authority_status"], "unavailable")
        self.assertFalse(response["predictive_eligible"])
        self.assertTrue(lineage["content_addressed"])
        self.assertEqual(lineage["predictive_authority_status"], "unavailable")
        self.assertFalse(lineage["predictive_eligible"])

    def test_embedding_channel_is_disabled_before_self_attestation(self) -> None:
        with self.assertRaisesRegex(ChampionRepresentationError, "disabled pending"):
            self._build(embedding_snapshot={"caller_says_safe": True})
        for dimension in (2, 4):
            result = self._build(embedding_dimension=dimension)
            layer = result["roles"][2]["learned_residual_embedding"]
            self.assertEqual(layer["status"], "unavailable")
            self.assertEqual(layer["vector"], [0.0] * dimension)
            self.assertEqual(
                layer["activation_status"],
                "disabled_pending_independent_evidence_registry",
            )
            self.assertFalse(layer["predictive_eligible"])
        for invalid in (2.0, True):
            with self.assertRaisesRegex(ChampionRepresentationError, "must be 2 or 4"):
                self._build(embedding_dimension=invalid)

    def test_representation_hash_is_sensitive_to_each_empirical_layer(self) -> None:
        response_a = self._response_snapshot()
        response_b = self._response_snapshot()
        response_b["cells"][0]["values"] = [0.26]
        self._rehash_response(response_b)
        a = self._build(response_snapshot=response_a)
        b = self._build(response_snapshot=response_b)
        self.assertNotEqual(a["representation_sha256"], b["representation_sha256"])

    def test_legacy_archetype_prior_is_unchanged(self) -> None:
        kwargs = {
            "champion_id": "riot:champion:115",
            "role": "mid",
            "patch_id": "26.14",
            "league_id": "LEC",
        }
        before = copy.deepcopy(self.ontology.build_archetype_prior(**kwargs))
        self._build(
            response_snapshot=self._response_snapshot(),
        )
        after = self.ontology.build_archetype_prior(**kwargs)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
