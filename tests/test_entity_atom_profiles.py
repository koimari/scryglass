from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.v2.champions.atoms.entity_atom_profiles import (
    EntityAtomProfileError,
    build_entity_atom_profiles,
    public_projection,
)


ROOT = Path(__file__).resolve().parents[1]


def test_profiles_aggregate_players_and_teams_in_atom_space() -> None:
    result = build_entity_atom_profiles(
        [
            {"entity_type": "player", "entity_name": "Player A", "champion": "Aatrox", "games": 3, "wins": 2, "role": "top"},
            {"entity_type": "player", "entity_name": "Player A", "champion": "Ahri", "games": 1, "wins": 1, "role": "mid"},
            {"entity_type": "team", "entity_name": "Team A", "champion": "Aatrox", "games": 4, "wins": 3, "league": "LCK"},
        ],
        bridge_path=ROOT / "data/lol/v2/champions/lcc-atom-bridge-26.16.json",
        receipt_path=ROOT / "data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json",
    )
    assert result["authority"] == "development_only"
    assert {row["entity_type"] for row in result["entities"]} == {"player", "team"}
    assert result["entity_matrix"]
    assert any(column.startswith("family:") for column in result["feature_columns"])
    assert result["claim_ceiling"]["public_rating"] is False


def test_unknown_champion_is_counted_and_missing_champion_fails_closed() -> None:
    result = build_entity_atom_profiles(
        [
            {"entity_type": "player", "entity_name": "Player A", "champion": "Aatrox", "games": 1},
            {"entity_type": "player", "entity_name": "Player A", "champion": "NotAChampion", "games": 1},
        ],
        bridge_path=ROOT / "data/lol/v2/champions/lcc-atom-bridge-26.16.json",
        receipt_path=ROOT / "data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json",
    )
    assert result["skipped_unknown_champions"] == 1
    assert result["source_rows"] == 2
    assert result["entities"][0]["source_rows"] == 2
    assert result["entities"][0]["contributing_rows"] == 1
    with pytest.raises(EntityAtomProfileError, match="no champion"):
        build_entity_atom_profiles(
            [{"entity_type": "player", "entity_name": "Player A", "games": 1}],
            bridge_path=ROOT / "data/lol/v2/champions/lcc-atom-bridge-26.16.json",
            receipt_path=ROOT / "data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json",
        )


def test_canonical_entity_ids_keep_same_display_names_separate() -> None:
    result = build_entity_atom_profiles(
        [
            {
                "entity_type": "player",
                "entity_name": "Same Name",
                "player_id": "source-a",
                "champion": "Aatrox",
                "games": 1,
            },
            {
                "entity_type": "player",
                "entity_name": "Same Name",
                "player_id": "source-b",
                "champion": "Ahri",
                "games": 1,
            },
        ],
        bridge_path=ROOT / "data/lol/v2/champions/lcc-atom-bridge-26.16.json",
        receipt_path=ROOT / "data/lol/v2/champions/lcc-atom-refresh-26.16-receipt.json",
    )
    assert len(result["entities"]) == 2
    assert {row["identity_source"] for row in result["entities"]} == {"canonical_id"}
    assert len({row["entity_id"] for row in result["entities"]}) == 2


def test_public_projection_withholds_research_vectors() -> None:
    projection = public_projection()
    assert projection["authority"] == "unavailable"
    assert projection["entities"] is None
    assert projection["entity_matrix"] is None
