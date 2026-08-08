from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills import grid_market_foundation as market


def _row(
    schema: str,
    time: int,
    sequence: int,
    **values,
):
    return {
        "rfc461Schema": schema,
        "gameTime": time,
        "sequenceIndex": sequence,
        "rootGameID": "root",
        "generationID": 0,
        **values,
    }


def test_market_events_use_only_at_or_before_checkpoint() -> None:
    rows = [
        _row(
            "building_destroyed",
            599_999,
            1,
            buildingType="turret",
            teamID=200,
            lane="top",
            turretTier="outer",
        ),
        _row(
            "building_destroyed",
            600_001,
            2,
            buildingType="inhibitor",
            teamID=200,
            lane="top",
        ),
    ]
    values = market._riot_market_rows(rows, cutoff_ms=600_000)
    assert values["total_tower_destructions"] == 1
    assert values["first_tower"] == 100
    assert values["total_inhibitor_destructions"] == 0
    assert values["first_inhibitor"] is None


def test_tower_events_and_unique_towers_are_distinct() -> None:
    rows = [
        _row(
            "building_destroyed",
            1_000,
            1,
            buildingType="turret",
            teamID=100,
            lane="mid",
            turretTier="nexus",
            nexusTurretName="nexus1",
        ),
        _row(
            "building_destroyed",
            2_000,
            2,
            buildingType="turret",
            teamID=100,
            lane="mid",
            turretTier="nexus",
            nexusTurretName="nexus1",
        ),
    ]
    values = market._riot_market_rows(rows, cutoff_ms=None)
    assert values["total_tower_destructions"] == 2
    assert values["unique_towers_destroyed"] == 1


def test_first_inhibitor_taker_is_opposite_destroyed_team() -> None:
    rows = [
        _row(
            "building_destroyed",
            1_000,
            1,
            buildingType="inhibitor",
            teamID=100,
            lane="mid",
        )
    ]
    values = market._riot_market_rows(rows, cutoff_ms=None)
    assert values["first_inhibitor"] == 200


def test_canonical_riot_events_requires_unique_game_end(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    rows = [
        {
            **_row("game_info", 0, 1),
            "platformID": "LOLTMNT01",
            "gameID": 1,
            "rootGameID": 2,
        }
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = market._canonical_riot_events(path)
    assert "outcome.riot_game_end_not_unique" in result["blockers"]


def test_safe_file_metadata_removes_signed_capabilities() -> None:
    value = market._safe_file_metadata(
        {
            "id": "events-riot-game-1",
            "fullURL": "https://example.invalid/?token=secret",
            "signature": "secret",
            "status": "ready",
        }
    )
    assert value == {"id": "events-riot-game-1", "status": "ready"}
    assert "secret" not in json.dumps(value)


def test_market_summary_always_withholds_authority() -> None:
    games = [
        {
            "labels": {
                "first_inhibitor": {"status": "verified"},
                "first_tower": {"status": "verified"},
            },
            "checkpoints": [{"status": "eligible"} for _ in range(4)],
        }
    ]
    rows = market._market_summary(games)
    first_inhibitor = next(row for row in rows if row["target"] == "first_inhibitor")
    assert first_inhibitor["structural_feasibility"] == "verified_on_bounded_sample"
    assert first_inhibitor["research_authority_status"] == "unavailable"
    assert first_inhibitor["probability_authorized"] is False
    assert "sample.minimum_30_verified_maps_not_met" in first_inhibitor[
        "research_authority_blockers"
    ]


def test_grid_labels_fail_on_ambiguous_first() -> None:
    final_state = {
        "teams": {
            "100": {"tower_count": 1, "first_tower": True},
            "200": {"tower_count": 1, "first_tower": True},
        }
    }
    assert market._grid_labels(final_state)["first_tower"] is None
