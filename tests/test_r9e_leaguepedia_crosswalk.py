from __future__ import annotations

from tools.build_r9e_leaguepedia_crosswalk import _time, build_crosswalk


def test_crosswalk_time_parser_accepts_space_and_iso_z_timestamps() -> None:
    expected = (2026, 8, 14, 15, 7, 55)
    assert _time("2026-08-14 15:07:55").timetuple()[:6] == expected
    assert _time("2026-08-14T15:07:55Z").timetuple()[:6] == expected


def _evidence_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    games = [
        ("oe-1", "2026-08-14 15:07:55", "SeriesA_1", "Alpha", "Beta", "2026-08-14 15:00:00"),
        ("oe-2", "2026-08-14 15:54:54", "SeriesA_2", "Alpha", "Beta", "2026-08-14 15:00:00"),
        ("oe-3", "2026-08-14 17:02:02", "SeriesB_1", "Gamma", "Delta", "2026-08-14 17:15:00"),
        ("oe-4", "2026-08-14 17:48:14", "SeriesB_2", "Gamma", "Delta", "2026-08-14 17:15:00"),
        ("oe-5", "2026-08-14 18:42:36", "SeriesB_3", "Delta", "Gamma", "2026-08-14 17:15:00"),
        ("oe-6", "2026-08-15 15:07:14", "SeriesC_1", "Epsilon", "Zeta", "2026-08-15 15:00:00"),
        ("oe-7", "2026-08-15 15:58:57", "SeriesC_2", "Epsilon", "Zeta", "2026-08-15 15:00:00"),
    ]
    oe = [
        {"gameid": game_id, "date": date, "league": "LEC", "patch": "16.16", "teams": [team1, team2]}
        for game_id, date, _game_key, team1, team2, _series_date in games
    ]
    scoreboard = [
        {
            "GameId": game_key,
            "DateTime UTC": date,
            "Team1": team1,
            "Team2": team2,
            "Patch": "26.16",
            "OverviewPage": "LEC/2026 Season/Summer Season",
            "Tournament": "LEC 2026 Summer",
        }
        for _oe_id, date, game_key, team1, team2, _series_date in games
    ]
    schedule = [
        {
            "MatchId": series_id,
            "DateTime UTC": series_date,
            "Team1": team1,
            "Team2": team2,
            "Patch": "26.16",
        }
        for series_id, team1, team2, series_date in (
            ("SeriesA", "Alpha", "Beta", "2026-08-14 15:00:00"),
            ("SeriesB", "Gamma", "Delta", "2026-08-14 17:15:00"),
            ("SeriesC", "Epsilon", "Zeta", "2026-08-15 15:00:00"),
        )
    ]
    return oe, scoreboard, schedule


def test_crosswalk_keeps_source_patch_and_resolves_series_without_outcome_inference() -> None:
    oe, scoreboard, schedule = _evidence_rows()

    result = build_crosswalk(
        oe,
        scoreboard,
        schedule,
        source_manifest={"source": "fixture"},
        captured_at="2026-08-15T00:00:00Z",
    )

    assert result["counts"] == {
        "oe_rows": 7,
        "scoreboard_rows": 7,
        "schedule_rows": 3,
        "mapped_rows": 7,
        "mapped_series": 3,
        "issues": 0,
    }
    assert result["patch_identity"] == {
        "source_token": "16.16",
        "public_patch": "26.16",
        "prior_source_token_preserved": "16.15",
        "prior_public_patch": "26.15",
    }
    assert {row["oe_patch_token"] for row in result["rows"]} == {"16.16"}
    assert result["public_probability_authorized"] is False
    assert result["public_draft_authorized"] is False


def test_crosswalk_rejects_ambiguous_scoreboard_identity() -> None:
    oe, scoreboard, schedule = _evidence_rows()
    scoreboard.append(dict(scoreboard[0], GameId="SeriesA_duplicate_1"))

    result = build_crosswalk(
        oe,
        scoreboard,
        schedule,
        source_manifest={"source": "fixture"},
        captured_at="2026-08-15T00:00:00Z",
    )

    assert result["counts"]["issues"] >= 2
    assert any(issue["kind"] == "scoreboard_identity_ambiguous" for issue in result["issues"])
    assert result["public_probability_authorized"] is False
