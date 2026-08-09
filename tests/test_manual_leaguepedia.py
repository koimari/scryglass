from __future__ import annotations

import pytest

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.manual_leaguepedia import (
    PregameLeakageError,
    attach_score,
    freeze_pregame,
    leaguepedia_api_url,
    reveal_outcome,
    resolve_time_sliced_lineup,
    verify_run,
)


SHA = "a" * 64


def _side(team: str, players: list[str], picks: list[str]) -> dict:
    roles = ["top", "jungle", "mid", "bot", "support"]
    return {
        "team": team,
        "picks": picks,
        "players": [
            {"role": role, "player": player}
            for role, player in zip(roles, players)
        ],
    }


def _pregame(mode: str = "strict") -> dict:
    return {
        "mode": mode,
        "fixture_id": "fixture-001",
        "event_start": "2026-07-20T18:00:00Z",
        "draft_locked_at": "2026-07-20T17:55:00Z",
        "as_of": "2026-07-20T17:56:00Z",
        "competition": {"league": "CBLOL", "scope": "regional"},
        "blue": _side(
            "LYON",
            ["Dhokla", "Inspired", "Saint", "Berserker", "Isles"],
            ["Jayce", "Nocturne", "Viktor", "Ziggs", "Alistar"],
        ),
        "red": _side(
            "LØS",
            ["Zest", "Curse", "Feisty", "Duduhh", "Ackerman"],
            ["Vayne", "Vi", "Taliyah", "Cassiopeia", "Camille"],
        ),
        "source_snapshots": [
            {
                "snapshot_id": "source-001",
                "sha256": SHA,
                "available_at": "2026-07-20T17:40:00Z",
            }
        ],
    }


def test_freeze_is_deterministic_and_excludes_outcome() -> None:
    first = freeze_pregame(_pregame())
    second = freeze_pregame(_pregame())

    assert first["pregame_sha256"] == second["pregame_sha256"]
    assert first["phase"] == "pregame_frozen"
    assert "outcome" not in first
    verify_run(first)


def test_pregame_rejects_nested_winner() -> None:
    payload = _pregame()
    payload["blue"]["winner"] = True

    with pytest.raises(PregameLeakageError, match="outcome fields"):
        freeze_pregame(payload)


def test_strict_pregame_requires_as_of_before_event() -> None:
    payload = _pregame()
    payload["as_of"] = "2026-07-20T18:00:00Z"

    with pytest.raises(PregameLeakageError, match="draft_locked_at <= as_of < event_start"):
        freeze_pregame(payload)


def test_roster_move_selects_temporary_starter_after_leave() -> None:
    events = [
        {
            "team": "LYON",
            "role": "jungle",
            "player": "Inspired",
            "status": "confirmed_starter",
            "effective_from": "2026-01-01T00:00:00Z",
            "available_at": "2026-07-19T12:00:00Z",
            "precedence": 100,
        },
        {
            "team": "LYON",
            "role": "jungle",
            "player": "Inspired",
            "status": "leave",
            "effective_from": "2026-07-20T12:00:00Z",
            "available_at": "2026-07-20T12:05:00Z",
            "precedence": 300,
        },
        {
            "team": "LYON",
            "role": "jungle",
            "player": "Armao",
            "status": "temporary_starter",
            "effective_from": "2026-07-20T12:00:00Z",
            "available_at": "2026-07-20T12:05:00Z",
            "precedence": 400,
        },
    ]
    for role, player in zip(
        ["top", "mid", "bot", "support"],
        ["Dhokla", "Saint", "Berserker", "Isles"],
    ):
        events.append(
            {
                "team": "LYON",
                "role": role,
                "player": player,
                "status": "confirmed_starter",
                "effective_from": "2026-01-01T00:00:00Z",
                "available_at": "2026-07-19T12:00:00Z",
                "precedence": 100,
            }
        )

    resolved = resolve_time_sliced_lineup(
        events,
        "LYON",
        event_start="2026-07-20T18:00:00Z",
        as_of="2026-07-20T17:56:00Z",
    )

    assert resolved["status"] == "ok"
    assert [row["player"] for row in resolved["players"]] == [
        "Dhokla",
        "Armao",
        "Saint",
        "Berserker",
        "Isles",
    ]


def test_roster_resolution_fails_closed_on_equal_candidates() -> None:
    events = []
    for role in ["top", "jungle", "mid", "bot", "support"]:
        for player in ([f"{role}-one", f"{role}-two"] if role == "jungle" else [f"{role}-one"]):
            events.append(
                {
                    "team": "LYON",
                    "role": role,
                    "player": player,
                    "status": "confirmed_starter",
                    "effective_from": "2026-01-01T00:00:00Z",
                    "available_at": "2026-07-19T12:00:00Z",
                    "precedence": 100,
                }
            )

    resolved = resolve_time_sliced_lineup(
        events,
        "LYON",
        event_start="2026-07-20T18:00:00Z",
        as_of="2026-07-20T17:56:00Z",
    )
    assert resolved["status"] == "unavailable"
    assert any("ambiguous active candidates" in error for error in resolved["errors"])


def test_outcome_cannot_be_revealed_before_score() -> None:
    frozen = freeze_pregame(_pregame())
    with pytest.raises(Exception):
        reveal_outcome(
            frozen,
            {"winner": "LØS", "revealed_at": "2026-07-20T19:00:00Z"},
        )


def test_score_attachment_records_input_hash_and_reveal_order() -> None:
    frozen = freeze_pregame(_pregame(mode="retrospective"))
    scored = attach_score(
        frozen,
        {"composite": {"blue_pct": 50.0, "red_pct": 50.0}},
        runtime_as_of="2026-07-18 16:33:48",
        runtime_sha256=SHA,
        runner_sha256=SHA,
        score_module_sha256=SHA,
        scored_at="2026-07-20T19:00:00Z",
    )
    revealed = reveal_outcome(
        scored,
        {"winner": "LØS", "source_sha256": SHA},
        revealed_at="2026-07-20T19:01:00Z",
    )

    assert scored["score"]["input_pregame_sha256"] == frozen["pregame_sha256"]
    assert revealed["phase"] == "outcome_revealed"
    assert revealed["pregame"] == frozen["pregame"]
    verify_run(revealed, require_score=True, require_outcome=True)


def test_strict_score_rejects_runtime_after_fixture() -> None:
    frozen = freeze_pregame(_pregame())
    with pytest.raises(PregameLeakageError, match="runtime as_of"):
        attach_score(
            frozen,
            {"composite": {"blue_pct": 50.0, "red_pct": 50.0}},
            runtime_as_of="2026-07-21T00:00:00Z",
            runtime_sha256=SHA,
            runner_sha256=SHA,
            score_module_sha256=SHA,
        )


def test_leaguepedia_cutoff_url_and_team_aliases_are_stable() -> None:
    url = leaguepedia_api_url(
        "LØS/Match History",
        before="2026-07-15T14:00:00Z",
    )
    assert "rvstart=2026-07-15T14%3A00%3A00Z" in url
    assert "rvdir=older" in url
    assert normalize_team("MIBR.LOS") == "LØS"
    assert normalize_team("Los Grandes") == "LØS"
    assert normalize_team("LYON (2024 American Team)") == "LYON"
