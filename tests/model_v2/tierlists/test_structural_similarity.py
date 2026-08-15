"""Tests for the descriptive champion similarity library."""

from __future__ import annotations

from lol_kills.v2.champions.atoms.consume import AtomBridge
from lol_kills.v2.tierlists.structural_similarity import (
    MINIMUM_SIMILARITY,
    build_structural_similarity,
    champion_structural_similarity,
)


def _profile(bridge: AtomBridge, name: str) -> dict:
    for champion_id in bridge.champion_ids():
        profile = bridge.profile(champion_id)
        if profile and profile["display_name"] == name:
            return profile
    raise AssertionError(f"missing champion profile: {name}")


def test_similarity_is_symmetric_bounded_and_complete() -> None:
    bridge = AtomBridge.load()
    ziggs = _profile(bridge, "Ziggs")
    xerath = _profile(bridge, "Xerath")

    assert champion_structural_similarity(ziggs, ziggs) == 1.0
    assert champion_structural_similarity(ziggs, xerath) == champion_structural_similarity(xerath, ziggs)
    assert 0.0 <= champion_structural_similarity(ziggs, xerath) <= 1.0

    library = build_structural_similarity(bridge)
    assert len(library["champions"]) == 173
    assert len(library["similarity"]) == 173
    assert all(len(row) == 173 for row in library["similarity"])
    assert library["source_atom_bridge_sha256"] == bridge.artifact_sha256


def test_user_examples_pass_the_structural_viability_threshold() -> None:
    bridge = AtomBridge.load()
    ziggs = _profile(bridge, "Ziggs")
    sion = _profile(bridge, "Sion")

    assert champion_structural_similarity(ziggs, _profile(bridge, "Xerath")) >= MINIMUM_SIMILARITY
    assert champion_structural_similarity(ziggs, _profile(bridge, "Vel'Koz")) >= MINIMUM_SIMILARITY
    assert champion_structural_similarity(sion, _profile(bridge, "Malphite")) >= MINIMUM_SIMILARITY


def test_role_positions_and_profile_basis_are_public() -> None:
    library = build_structural_similarity(AtomBridge.load())
    by_name = {profile["champion"]: profile for profile in library["champions"]}

    assert "mid" in by_name["Ziggs"]["positions"]
    assert by_name["Ziggs"]["profile_status"] == "atom_detail"
    assert by_name["Anivia"]["profile_status"] == "atom_detail"
    assert by_name["Ziggs"]["traits"]
