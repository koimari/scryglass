from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lol_kills.v2.data.grid_scheduled_roster import (
    NON_AUTHORITY_STATUS,
    READY_STATUS,
    ScheduledRosterError,
    UNAVAILABLE_STATUS,
    evaluate_pre_event_scheduled_roster,
)


HASH = "a" * 64
OBSERVED = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)


def _series(*, start: str = "2026-08-01T12:00:00Z") -> dict:
    teams = [
        ("team-a", "Alpha"),
        ("team-b", "Bravo"),
    ]
    roles = ("top", "jungle", "mid", "bottom", "support")
    players = []
    for team_id, team_name in teams:
        for index, role in enumerate(roles, start=1):
            players.append(
                {
                    "id": f"{team_id}-player-{index}",
                    "nickname": f"{team_name}-{role}",
                    "updatedAt": "2026-07-20T00:00:00Z",
                    "team": {"id": team_id, "name": team_name},
                    "roles": [{"name": role}],
                }
            )
    return {
        "id": "series-1",
        "startTimeScheduled": start,
        "updatedAt": "2026-07-29T00:00:00Z",
        "teams": [{"baseInfo": {"id": team_id, "name": name}} for team_id, name in teams],
        "players": players,
    }


def _evaluate(series: dict):
    return evaluate_pre_event_scheduled_roster(
        series,
        observed_at=OBSERVED,
        source_payload_sha256=HASH,
    )


def test_valid_shape_is_only_a_non_authorizing_review_candidate() -> None:
    result = _evaluate(_series())

    assert result.status == READY_STATUS
    assert result.authority_status == NON_AUTHORITY_STATUS
    assert result.is_ready_for_review is True
    assert result.can_authorize_roster is False
    assert result.reasons == ()
    assert tuple(player.role for player in result.teams[0].players) == (
        "top",
        "jungle",
        "mid",
        "bot",
        "support",
    )
    assert all(value is False for value in result.claim_ceiling.values())


def test_missing_side_or_extra_player_is_unavailable() -> None:
    series = _series()
    series["players"] = series["players"][:-1]

    result = _evaluate(series)

    assert result.status == UNAVAILABLE_STATUS
    assert "TEAM_PLAYER_COUNT_NOT_EXACT:team-b:4" in result.reasons


def test_duplicate_role_and_duplicate_player_do_not_get_selected() -> None:
    series = _series()
    series["players"][1]["roles"] = [{"name": "top"}]
    series["players"][2]["id"] = series["players"][0]["id"]

    result = _evaluate(series)

    assert result.status == UNAVAILABLE_STATUS
    assert "TEAM_PLAYER_IDS_NOT_UNIQUE:team-a" in result.reasons
    assert "TEAM_ROLE_SET_NOT_EXACT:team-a" in result.reasons


def test_post_start_observation_and_updated_source_are_rejected() -> None:
    series = _series(start="2026-07-30T11:00:00Z")
    series["updatedAt"] = "2026-07-30T11:30:00Z"

    result = _evaluate(series)

    assert result.status == UNAVAILABLE_STATUS
    assert "OBSERVED_AFTER_SCHEDULED_START" in result.reasons
    assert "SERIES_UPDATED_AFTER_SCHEDULED_START" in result.reasons


def test_player_update_after_observation_is_rejected() -> None:
    series = _series()
    series["players"][0]["updatedAt"] = "2026-07-30T12:00:01Z"

    result = _evaluate(series)

    assert result.status == UNAVAILABLE_STATUS
    assert "PLAYER_UPDATED_AFTER_OBSERVATION:team-a-player-1" in result.reasons


def test_malformed_source_hash_is_not_accepted() -> None:
    with pytest.raises(ScheduledRosterError, match="source_payload_sha256"):
        evaluate_pre_event_scheduled_roster(
            _series(),
            observed_at=OBSERVED,
            source_payload_sha256="not-a-hash",
        )


def test_empty_series_players_never_falls_back_to_team_directory() -> None:
    series = _series()
    series["players"] = []

    result = _evaluate(series)

    assert result.status == UNAVAILABLE_STATUS
    assert result.can_authorize_roster is False
    assert any(reason.startswith("TEAM_PLAYER_COUNT_NOT_EXACT:") for reason in result.reasons)

