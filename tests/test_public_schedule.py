from __future__ import annotations

import urllib.parse
from datetime import datetime, timezone

import pytest

from lol_kills.export.public_schedule import (
    PublicScheduleError,
    build_public_schedule,
    public_region,
    validate_public_schedule,
)


def _fake_fetch(url: str):
    table = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)["tables"][0]
    if table == "Tournaments":
        return [
            {
                "Name": "LCS 2026 Summer",
                "OverviewPage": "LCS/2026 Season/Summer Season",
                "DateStart": "2026-07-25",
                "Date": "2026-09-06",
                "Region": "North America",
                "League": "League of Legends Championship Series",
                "TournamentLevel": "Primary",
                "IsOfficial": 1,
            },
            {
                "Name": "Worlds 2026",
                "OverviewPage": "2026 Season World Championship",
                "DateStart": "2026-10-01",
                "Date": "2026-11-08",
                "Region": "International",
                "TournamentLevel": "Primary",
                "IsOfficial": 1,
            },
        ]
    assert table == "MatchSchedule"
    return [
        {
            "MatchId": "lcs-next",
            "OverviewPage": "LCS/2026 Season/Summer Season",
            "Team1": "LYON (2024 American Team)",
            "Team2": "Cloud9",
            "DateTime UTC": "2026-08-11 18:00:00",
            "HasTime": 1,
            "BestOf": 3,
            "Tab": "Week 4",
            "Winner": None,
            "N MatchInTab": 1,
        },
        {
            "MatchId": "completed",
            "OverviewPage": "LCS/2026 Season/Summer Season",
            "Team1": "Team Liquid",
            "Team2": "Cloud9",
            "DateTime UTC": "2026-08-11 16:00:00",
            "HasTime": 1,
            "BestOf": 3,
            "Tab": "Week 4",
            "Winner": 1,
            "N MatchInTab": 2,
        },
    ]


def test_public_schedule_keeps_future_series_and_canonical_teams() -> None:
    payload = build_public_schedule(
        now=datetime(2026, 8, 11, 12, tzinfo=timezone.utc),
        fetch_json=_fake_fetch,
    )

    assert payload["refresh_status"] == "fresh"
    assert len(payload["upcoming"]) == 1
    series = payload["upcoming"][0]
    assert series["team1"] == "LYON"
    assert series["team2"] == "Cloud9"
    assert series["region"] == "Americas"
    assert series["best_of"] == 3
    assert payload["tournaments"][0]["status"] == "current"
    assert payload["tournaments"][1]["status"] == "upcoming"


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("Brazil", "Americas"),
        ("EMEA", "EMEA"),
        ("Korea", "Asia"),
        ("International", "International"),
        ("Unknown", "Other"),
    ],
)
def test_public_region_is_small_and_stable(source: str, expected: str) -> None:
    assert public_region(source) == expected


def test_schedule_validator_rejects_duplicate_series() -> None:
    payload = build_public_schedule(
        now=datetime(2026, 8, 11, 12, tzinfo=timezone.utc),
        fetch_json=_fake_fetch,
    )
    payload["upcoming"].append(dict(payload["upcoming"][0]))
    with pytest.raises(PublicScheduleError, match="identity"):
        validate_public_schedule(payload)
