"""Tests for the isolated fail-closed atom matchup feature resolver."""

from __future__ import annotations

import copy
import json

import pytest

from lol_kills.v2.champions.atoms.consume import AtomBridge
from lol_kills.v2.champions.atoms.schema import canonical_sha256
from lol_kills.v2.tierlists.atom_matchup_features import (
    ATTRIBUTE_FEATURE_NAMES,
    AtomMatchupFeatureError,
    AtomMatchupFeatureResolver,
    ExactAtomSnapshotMapping,
    FEATURE_ORDER,
    FEATURE_SCHEMA_SHA256,
    FAMILY_FEATURE_NAMES,
    ONTOLOGY_REFERENCE_LABELS,
    ONTOLOGY_FEATURE_NAMES,
)


def _bridge() -> AtomBridge:
    return AtomBridge.load()


def _mutated_bridge(mutator) -> AtomBridge:
    # Read through the public artifact path used by AtomBridge.load.  The
    # temporary payload is re-hashed before it is passed to the validator.
    from lol_kills.v2.champions.atoms.consume import DEFAULT_ARTIFACT_PATH

    payload = json.loads(DEFAULT_ARTIFACT_PATH.read_text(encoding="utf-8"))
    unsigned = copy.deepcopy(payload)
    unsigned.pop("artifact_sha256", None)
    mutator(unsigned)
    unsigned["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in unsigned.items() if key != "artifact_sha256"}
    )
    return AtomBridge(unsigned)


def test_schema_is_fixed_and_deterministic() -> None:
    first = AtomMatchupFeatureResolver(_bridge())
    second = AtomMatchupFeatureResolver(_bridge())

    assert first.feature_schema() == second.feature_schema()
    assert first.feature_schema_sha256 == second.feature_schema_sha256 == FEATURE_SCHEMA_SHA256
    schema = first.feature_schema()
    assert schema["feature_order"] == list(FEATURE_ORDER)
    assert schema["availability"]["missing_value"] is None
    assert schema["availability"]["missing_policy"] == "fail_closed_no_zero_imputation"
    assert len(FAMILY_FEATURE_NAMES) == 6
    assert len(ATTRIBUTE_FEATURE_NAMES) == 7
    assert all(
        reference not in name
        for dimension, reference in ONTOLOGY_REFERENCE_LABELS.items()
        for name in ONTOLOGY_FEATURE_NAMES
        if name.startswith(f"ontology.{dimension}.")
    )


def test_pair_features_are_canonical_and_antisymmetric() -> None:
    resolver = AtomMatchupFeatureResolver(_bridge())
    forward = resolver.resolve_pair("riot:champion:266", "riot:champion:103")
    reverse = resolver.resolve_pair("riot:champion:103", "riot:champion:266")

    assert forward["canonical_pair"]["key"] == reverse["canonical_pair"]["key"]
    assert forward["canonical_pair"]["orientation"] == -reverse["canonical_pair"]["orientation"]
    assert forward["feature_order"] == reverse["feature_order"] == list(FEATURE_ORDER)
    for name in FEATURE_ORDER:
        assert forward["availability"][name] == reverse["availability"][name]
        if forward["availability"][name]:
            assert reverse["features"][name] == pytest.approx(-forward["features"][name])
        else:
            assert forward["features"][name] is None
            assert reverse["features"][name] is None


def test_missing_inputs_remain_unavailable_without_zero_imputation() -> None:
    def remove_inputs(payload: dict) -> None:
        profile = payload["champions"][0]
        profile["family_presence"].pop("damage")
        profile["lcc_attribute_ratings"].pop("utility")
        profile["ontology_prior"]["damage_profile"]["labels"] = None

    bridge = _mutated_bridge(remove_inputs)
    result = AtomMatchupFeatureResolver(bridge).resolve("riot:champion:266")

    missing_names = {
        "family_presence.damage",
        "lcc_attribute.utility",
        "ontology.damage_profile.burst",
        "ontology.damage_profile.artillery",
        "ontology.damage_profile.teamfight",
    }
    for name in missing_names:
        assert result["availability"][name] is False
        assert result["features"][name] is None
    # A registered false family is a real zero.  The missing family above is
    # still None, so a zero cannot be a hidden missing-value convention.
    assert result["features"]["family_presence.vision-economy"] == 0.0
    assert result["availability"]["family_presence.vision-economy"] is True


def test_explicit_target_patch_requires_exact_time_safe_mapping() -> None:
    resolver = AtomMatchupFeatureResolver(_bridge())

    with pytest.raises(AtomMatchupFeatureError, match="exact time-safe"):
        resolver.resolve("riot:champion:266", requested_patch="26.16")

    wrong_patch = ExactAtomSnapshotMapping(
        requested_patch="26.16",
        snapshot_patch="26.15",
        snapshot_as_of="2026-08-07T20:55:44.102200Z",
        bridge_artifact_sha256=resolver.bridge.artifact_sha256,
    )
    with pytest.raises(AtomMatchupFeatureError, match="exactly equal"):
        resolver.resolve(
            "riot:champion:266",
            requested_patch="26.16",
            snapshot_mapping=wrong_patch,
        )


def test_exact_mapping_is_bound_to_bridge_and_allows_explicit_patch() -> None:
    resolver = AtomMatchupFeatureResolver(_bridge())
    mapping = ExactAtomSnapshotMapping(
        requested_patch="26.15",
        snapshot_patch="26.15",
        snapshot_as_of=resolver.bridge.generated_at,
        bridge_artifact_sha256=resolver.bridge.artifact_sha256,
    )

    result = resolver.resolve(
        "riot:champion:266",
        requested_patch="26.15",
        snapshot_mapping=mapping,
    )
    assert result["snapshot"]["requested_patch"] == "26.15"
    assert result["provenance"]["bridge_artifact_sha256"] == resolver.bridge.artifact_sha256
