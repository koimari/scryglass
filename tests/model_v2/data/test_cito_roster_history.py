from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lol_kills.v2.data.cito_roster_history import (
    NON_AUTHORITY_STATUS,
    READY_STATUS,
    CitoRosterError,
    UNAVAILABLE_STATUS,
    evaluate_cito_team_roster,
)


HASH = "b" * 64
OBSERVED = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)
EVENT_START = datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc)


def _payloads(*, ended_role: str | None = None) -> list[dict[str, object]]:
    roles = ("TOP", "JUNGLE", "MID", "BOT", "SUPPORT")
    rows: list[dict[str, object]] = []
    for index, role in enumerate(roles, start=1):
        membership: dict[str, object] = {
            "teamSlug": "alpha",
            "teamName": "Alpha",
            "role": role,
            "startedAt": "2026-01-01T00:00:00Z",
            "endedAt": "2026-07-31T00:00:00Z" if role == ended_role else None,
        }
        rows.append(
            {
                "success": True,
                "data": {
                    "playerId": f"p-{index}",
                    "playerName": f"Player {index}",
                    "teams": [membership],
                },
            }
        )
    return rows


def _evaluate(payloads: list[dict[str, object]]):
    return evaluate_cito_team_roster(
        payloads,
        team_slug="alpha",
        team_name="Alpha",
        event_start=EVENT_START,
        observed_at=OBSERVED,
        source_updated_at=datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
        source_payload_sha256=HASH,
    )


def test_valid_documented_interval_shape_is_non_authorizing() -> None:
    result = _evaluate(_payloads())

    assert result.status == READY_STATUS
    assert result.authority_status == NON_AUTHORITY_STATUS
    assert result.is_ready_for_review is True
    assert result.can_authorize_roster is False
    assert result.reasons == ()
    assert tuple(player.role for player in result.players) == (
        "top",
        "jungle",
        "mid",
        "bot",
        "support",
    )
    assert all(value is False for value in result.claim_ceiling.values())


def test_membership_ending_before_event_is_not_selected() -> None:
    result = _evaluate(_payloads(ended_role="TOP"))

    assert result.status == UNAVAILABLE_STATUS
    assert "TEAM_PLAYER_COUNT_NOT_EXACT:alpha:4" in result.reasons
    assert "TEAM_ROLE_SET_NOT_EXACT:alpha" in result.reasons


def test_duplicate_active_role_and_player_are_unavailable() -> None:
    payloads = _payloads()
    payloads[1]["data"]["teams"][0]["role"] = "TOP"  # type: ignore[index]
    payloads[2]["data"]["playerId"] = "p-1"  # type: ignore[index]

    result = _evaluate(payloads)

    assert result.status == UNAVAILABLE_STATUS
    assert "TEAM_PLAYER_IDS_NOT_UNIQUE:alpha" in result.reasons
    assert "TEAM_ROLE_SET_NOT_EXACT:alpha" in result.reasons


def test_source_times_must_precede_event_and_observation() -> None:
    result = evaluate_cito_team_roster(
        _payloads(),
        team_slug="alpha",
        event_start=EVENT_START,
        observed_at=OBSERVED,
        source_updated_at=datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        source_payload_sha256=HASH,
    )

    assert result.status == UNAVAILABLE_STATUS
    assert "SOURCE_UPDATED_AT_NOT_BEFORE_EVENT_START" in result.reasons


def test_unsuccessful_or_missing_payload_never_falls_back() -> None:
    result = _evaluate([{"success": False, "data": {}}])

    assert result.status == UNAVAILABLE_STATUS
    assert "PLAYER_HISTORY_ROW_UNSUCCESSFUL:0" in result.reasons
    assert "NO_ACTIVE_TEAM_MEMBERSHIP_AT_EVENT_START" in result.reasons


def test_malformed_source_hash_is_rejected() -> None:
    with pytest.raises(CitoRosterError, match="source_payload_sha256"):
        evaluate_cito_team_roster(
            _payloads(),
            team_slug="alpha",
            event_start=EVENT_START,
            observed_at=OBSERVED,
            source_updated_at=datetime(2026, 7, 30, 11, 0, tzinfo=timezone.utc),
            source_payload_sha256="not-a-hash",
        )
