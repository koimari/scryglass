"""Tests for the LCC atom bridge (schema, build, consume, fail-closed rules)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.v2.champions.atoms.bridge_v1 import (
    DEFAULT_ARTIFACT_PATH,
    build_bridge_payload,
)
from lol_kills.v2.champions.atoms.consume import AtomBridge, AtomBridgeError
from lol_kills.v2.champions.atoms.lcc_sources import LccSources
from lol_kills.v2.champions.atoms.mapping_v1 import FAMILY_FALLBACK_V1, MAPPING_V1
from lol_kills.v2.champions.atoms.schema import (
    BRIDGE_SCHEMA_ID,
    BRIDGE_VERSION,
    CHAMPION_ATOM_FAMILIES,
    DIMENSION_LABELS,
    canonical_sha256,
)

def _artifact() -> dict:
    return json.loads(DEFAULT_ARTIFACT_PATH.read_text())


def _sources() -> LccSources:
    sources = LccSources()
    sources.load()
    return sources


def test_artifact_exists_and_is_canonical():
    artifact = _artifact()
    assert artifact["schema_id"] == BRIDGE_SCHEMA_ID
    assert artifact["version"] == BRIDGE_VERSION
    submitted = artifact["artifact_sha256"]
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256")
    assert canonical_sha256(unsigned) == submitted


def test_all_173_champions_profiled():
    artifact = _artifact()
    champions = artifact["champions"]
    assert len(champions) == 173
    ids = [c["champion_id"] for c in champions]
    assert len(set(ids)) == 173
    assert all(cid.startswith("riot:champion:") for cid in ids)
    for c in champions:
        assert set(c["family_presence"]) == set(CHAMPION_ATOM_FAMILIES)
        assert c["profile_status"] in {"atom_detail", "family_only"}


def test_atom_detail_champions_have_counts_and_damage_mix():
    artifact = _artifact()
    detail = [c for c in artifact["champions"] if c["profile_status"] == "atom_detail"]
    assert len(detail) >= 7
    for c in detail:
        counts = c["atom_family_counts"]
        assert counts is not None
        assert sum(counts.values()) == c["atom_count"] > 0
        assert isinstance(c["damage_type_mix"], dict)
        assert isinstance(c["relations"], list)


def test_ontology_prior_never_fabricates_zeros():
    artifact = _artifact()
    for c in artifact["champions"]:
        prior = c["ontology_prior"]
        for dimension, cell in prior.items():
            if cell["status"] == "unavailable":
                assert cell["labels"] is None and cell["uncertainty"] is None
            else:
                labels = cell["labels"]
                assert set(labels) == set(DIMENSION_LABELS[dimension])
                assert abs(sum(labels.values()) - 1.0) < 0.01  # 4-decimal rounding
                assert 0.0 <= cell["uncertainty"] <= 0.6


def test_mapping_rows_are_well_formed():
    for row in MAPPING_V1:
        assert row["atom_id"].count(".") >= 1
        assert row["dimension"] in DIMENSION_LABELS
        assert row["label"] in DIMENSION_LABELS[row["dimension"]]
        assert row["weight"] in (0.1, 0.15, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)
        assert row["evidence_note"]
    for row in FAMILY_FALLBACK_V1:
        assert row["family"] in CHAMPION_ATOM_FAMILIES
        assert row["dimension"] in DIMENSION_LABELS
        assert row["label"] in DIMENSION_LABELS[row["dimension"]]


def test_relations_graph_is_directed_and_consistent():
    artifact = _artifact()
    relations = artifact["atom_relations"]
    assert isinstance(relations, dict) and len(relations) >= 10
    for source, targets in relations.items():
        assert isinstance(source, str) and source.count(".") >= 1
        assert isinstance(targets, list) and targets
        for target in targets:
            assert isinstance(target, str) and target.count(".") >= 1


def test_provenance_pins_every_required_file():
    artifact = _artifact()
    provenance = artifact["provenance"]
    files = provenance["file_sha256"]
    for rel in (
        "data/atoms/atom-summary.json",
        "data/atoms/classification-report.json",
        "data/wiki-atoms/atom-relations.json",
        "data/champions.json",
    ):
        assert rel in files and len(files[rel]) == 64


def test_consume_loads_and_validates():
    bridge = AtomBridge.load()
    assert len(bridge.champion_ids()) == 173
    profile = bridge.profile("riot:champion:266")
    assert profile is not None and profile["display_name"] == "Aatrox"
    assert bridge.family_presence("riot:champion:266") is not None
    assert bridge.atom_family_counts("riot:champion:266") is not None
    assert bridge.ontology_prior("riot:champion:266") is not None
    assert bridge.profile("riot:champion:999999") is None
    assert bridge.mapping_for_atom("heal-shield.heal")


def test_consume_rejects_tampered_artifact(tmp_path):
    artifact = _artifact()
    path = tmp_path / "tampered.json"
    artifact["champions"][0]["display_name"] = "Hacked"
    path.write_text(json.dumps(artifact))
    with pytest.raises(AtomBridgeError):
        AtomBridge.load(path)


def test_builder_is_reproducible_from_pinned_sources():
    sources = _sources()
    payload = build_bridge_payload(sources)
    artifact = _artifact()
    unsigned = dict(artifact)
    unsigned.pop("artifact_sha256")
    # generated_at differs per run; compare everything else structurally
    rebuilt = dict(payload)
    rebuilt.pop("generated_at")
    existing = dict(unsigned)
    existing.pop("generated_at")
    assert rebuilt == existing
