from __future__ import annotations

import pandas as pd

from lol_kills.ratings.player_map_grades import compute_player_map_grades


ROLES = ("top", "jng", "mid", "bot", "sup")


def _grade_rows() -> pd.DataFrame:
    rows = []
    for game in range(61):
        date = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=game)
        for side, sign in (("Blue", 1), ("Red", -1)):
            for role_index, role in enumerate(ROLES):
                variation = ((game + role_index + (0 if side == "Blue" else 2)) % 9) - 4
                standout = game == 60 and side == "Blue" and role == "jng"
                lift = 4 if standout else 0
                rows.append(
                    {
                        "game_uid": f"grade-{game}",
                        "gameid": f"grade-{game}",
                        "date": date,
                        "league": "LCS",
                        "competition_tier": "tier1",
                        "side": side,
                        "position": role,
                        "playername": f"{side}-{role}",
                        "kills": 3 + max(variation + lift, 0),
                        "deaths": max(1, 4 - variation - lift),
                        "assists": 7 + max(variation + lift, 0),
                        "teamkills": 15,
                        "gamelength": 1800,
                        "dpm": 500 + 25 * variation + 120 * lift,
                        "damageshare": 0.2 + 0.008 * variation + 0.025 * lift,
                        "totalgold": 10000 + 240 * variation + 900 * lift,
                        "cspm": 7 + 0.15 * variation + 0.3 * lift,
                        "wpm": 0.5 + 0.02 * variation + 0.04 * lift,
                        "wcpm": 0.25 + 0.01 * variation + 0.03 * lift,
                        "golddiffat10": sign * variation * 60 + 350 * lift,
                        "result": 1 if side == "Blue" else 0,
                    }
                )
    return pd.DataFrame(rows)


def test_player_map_grades_use_four_comparisons_and_ignore_result() -> None:
    rows = _grade_rows()
    grades = compute_player_map_grades(rows)
    target = grades[(grades["game_id"] == "grade-60") & (grades["player"] == "Blue-jng")].iloc[0]

    assert target["grade_status"] == "available"
    assert target["grade"] in {"A", "A+"}
    assert target["grade_self"] > 0
    assert target["grade_team"] > 0
    assert target["grade_opponent"] > 0
    assert target["grade_league_role"] > 0

    flipped = rows.copy()
    flipped["result"] = 1 - flipped["result"]
    second = compute_player_map_grades(flipped)
    columns = ["game_id", "player", "grade", "grade_score"]
    pd.testing.assert_frame_equal(grades[columns], second[columns])


def test_player_map_grade_fails_closed_without_history() -> None:
    rows = _grade_rows()
    first = rows[rows["game_uid"].eq("grade-0")]
    grades = compute_player_map_grades(first)

    assert grades["grade_status"].eq("unavailable").all()
    assert grades["grade_reason"].str.contains("history|prior", case=False, regex=True).all()
