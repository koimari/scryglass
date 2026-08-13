from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.export.leaderboards import (
    LEADERBOARDS_SCHEMA,
    build_leaderboards,
)


def _profile_records() -> dict:
    return {
        "games": {
            "game-1": {
                "date": "2026-08-01T10:00:00Z",
                "blue_win": 1,
                "players": [
                    {"player": "Alice", "side": "Blue", "grade": {"status": "available", "grade": "A"}},
                    {"player": "Bob", "side": "Red", "grade": {"status": "available", "grade": "C"}},
                ],
            },
            "game-2": {
                "date": "2026-08-02T10:00:00Z",
                "blue_win": 0,
                "players": [
                    {"player": "Alice", "side": "Red", "grade": {"status": "available", "grade": "A"}},
                    {"player": "Bob", "side": "Blue", "grade": {"status": "unavailable", "grade": None}},
                ],
            },
        }
    }


def _player_records() -> dict:
    return {
        "Alice": {
            "primary_role": "mid",
            "current_team": "Team A",
            "current_league": "LCS",
            "games": 40,
            "wins": 25,
            "wr": 0.625,
        },
        "Bob": {
            "primary_role": "jng",
            "current_team": "Team B",
            "current_league": "LEC",
            "games": 30,
            "wins": 10,
            "wr": 0.3333,
        },
    }


def _ratings() -> list[dict]:
    return [
        {"player": "Alice", "mu_total": 1643.0},
        {"player": "Bob", "mu_total": 1580.0},
    ]


def _team_ratings() -> list[dict]:
    return [
        {"team": "Team A", "mu_total": 1700.0, "n_maps": 60},
        {"team": "Team B", "mu_total": 1620.0, "n_maps": 55},
    ]


def _team_records() -> dict:
    return {
        "Team A": {"team_key": "team-a", "current_league": "LCS", "games": 60, "wins": 35, "wr": 0.5833},
        "Team B": {"team_key": "team-b", "current_league": "LEC", "games": 55, "wins": 28, "wr": 0.5091},
    }


def _player_champions() -> dict:
    return {
        "Alice": [{"champion": "Orianna", "games": 10, "wins": 7, "wr": 0.7}],
        "Bob": [{"champion": "Orianna", "games": 8, "wins": 4, "wr": 0.5}],
    }


def _match_index() -> dict:
    return {
        "games": [
            {"game_id": "g1", "date": "2026-08-02T10:00:00Z", "league": "LCS", "blue_team": "Team A", "red_team": "Team B", "blue_win": 1},
            {"game_id": "g2", "date": "2026-08-01T10:00:00Z", "league": "LCS", "blue_team": "Team B", "red_team": "Team A", "blue_win": 0},
        ]
    }


def _draft_records() -> dict:
    return {
        "games": {
            f"draft-{index}": {
                "blue_team": "Team A",
                "red_team": "Team B",
                "draft_edge": 0.4,
            }
            for index in range(5)
        }
    }


def test_build_leaderboards_aggregates_all_domains() -> None:
    payload = build_leaderboards(
        _player_records(),
        _profile_records(),
        _ratings(),
        _team_ratings(),
        team_records=_team_records(),
        player_champion_records=_player_champions(),
        match_index=_match_index(),
        draft_records=_draft_records(),
        draft_players=[
            {"player": "Alice", "games": 8, "draft_score": 0.12, "best_pick_rate": 0.625, "role": "mid", "team": "Team A"},
            {"player": "Bob", "games": 10, "draft_score": 0.08, "best_pick_rate": 0.4, "role": "jng", "team": "Team B"},
        ],
    )

    assert payload["schema_version"] == LEADERBOARDS_SCHEMA

    alice = payload["players"]["Alice"]
    assert alice["rating"] == 1643.0
    assert alice["role"] == "mid"
    assert alice["team"] == "Team A"
    assert alice["grade_a_games"] == 2
    assert alice["grade_games"] == 2
    assert alice["games"] == 40
    assert alice["win_rate"] == 0.625

    bob = payload["players"]["Bob"]
    assert bob["grade_a_games"] == 0
    assert bob["grade_games"] == 1

    assert payload["top"]["a_grades"][0]["player"] == "Alice"
    assert payload["top"]["rating"][0]["player"] == "Alice"
    assert payload["top"]["rating_by_role"]["jng"][0]["player"] == "Bob"
    assert payload["top"]["win_rate"][0]["player"] == "Alice"

    assert payload["teams"][0]["team"] == "Team A"
    assert payload["teams"][0]["win_rate"] == 0.5833
    assert len(payload["teams"][0]["recent"]) == 2
    assert payload["teams"][0]["recent"][-1]["won"] is True

    assert payload["champions"]["Orianna"][0]["player"] == "Alice"
    assert payload["indexes"]["players"]["Alice"]["role"] == "mid"
    assert payload["indexes"]["teams"]["Team A"]["team_key"] == "team-a"
    assert "Orianna" in payload["indexes"]["champions"]
    assert "LCS" in payload["indexes"]["leagues"] and "LEC" in payload["indexes"]["leagues"]
    assert payload["teams_draft"][0]["team"] == "Team A"
    assert payload["teams_draft"][0]["draft_win_share"] == 0.5987
    assert payload["players_draft"][0]["player"] == "Alice"
    assert payload["players_draft"][0]["best_pick_rate"] == 0.625


def test_build_leaderboards_handles_missing_optional_payloads() -> None:
    payload = build_leaderboards(_player_records(), _profile_records(), _ratings(), _team_ratings())
    assert payload["champions"] == {}
    assert payload["teams"][0]["recent"] == []
    assert "players" in payload["indexes"]
