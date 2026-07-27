from __future__ import annotations

import pandas as pd
import pytest

from lol_kills.etl.series_schedule import (
    SCHEDULE_SOURCE,
    annotate_scheduled_series,
    validate_schedule,
)


def _schedule(*, best_of: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "riot_game_id": "LOLTMNT01_1",
                "leaguepedia_game_id": "LPL/2026_1_1",
                "game_index": 1,
                "match_id": "LPL/2026_1",
                "best_of": best_of,
                "scheduled_at": "2026-07-01T10:00:00Z",
                "team1": "Bilibili Gaming",
                "team2": "Anyone's Legend",
                "overview_page": "LPL/2026",
            }
        ]
    )


def test_schedule_annotation_uses_platform_id_and_verified_team_pair() -> None:
    rows = pd.DataFrame(
        [
            {
                "gameid": "LOLTMNT01_1",
                "side": "Blue",
                "teamname": "BLG",
                "game": 9,
            },
            {
                "gameid": "LOLTMNT01_1",
                "side": "Red",
                "teamname": "Anyone's Legend",
                "game": 9,
            },
        ]
    )

    result = annotate_scheduled_series(rows, _schedule())

    assert result.audit["matched_games"] == 1
    assert result.audit["team_conflicts"] == 0
    assert set(result.rows["source_series_id"]) == {"LPL/2026_1"}
    assert set(result.rows["series_format"]) == {"Bo3"}
    assert set(result.rows["series_format_source"]) == {SCHEDULE_SOURCE}
    assert set(result.rows["game"]) == {1}


def test_schedule_annotation_preserves_team_conflict_provenance() -> None:
    rows = pd.DataFrame(
        [
            {"gameid": "LOLTMNT01_1", "teamname": "Wrong A"},
            {"gameid": "LOLTMNT01_1", "teamname": "Wrong B"},
        ]
    )

    result = annotate_scheduled_series(rows, _schedule())

    assert result.audit["matched_games"] == 1
    assert result.audit["team_conflicts"] == 1
    assert set(result.rows["source_series_id"]) == {"LPL/2026_1"}
    assert set(result.rows["series_format"]) == {"Bo3"}
    assert set(result.rows["series_schedule_team_pair_status"]) == {
        "alias_or_identity_mismatch"
    }
    assert set(result.rows["leaguepedia_team1"]) == {"Bilibili Gaming"}
    assert set(result.rows["leaguepedia_team2"]) == {"Anyone's Legend"}


def test_schedule_annotation_fails_closed_on_date_conflict() -> None:
    rows = pd.DataFrame(
        [
            {
                "gameid": "LOLTMNT01_1",
                "date": "2026-07-10T10:00:00Z",
                "teamname": "BLG",
            },
            {
                "gameid": "LOLTMNT01_1",
                "date": "2026-07-10T10:00:00Z",
                "teamname": "Anyone's Legend",
            },
        ]
    )

    result = annotate_scheduled_series(rows, _schedule())

    assert result.audit["matched_games"] == 0
    assert result.audit["date_conflicts"] == 1
    assert result.rows["source_series_id"].isna().all()
    assert result.rows["series_format"].isna().all()


def test_schedule_annotation_replaces_only_its_own_stale_values() -> None:
    rows = pd.DataFrame(
        [
            {
                "gameid": "LOLTMNT01_1",
                "teamname": "BLG",
                "source_series_id": "stale",
                "series_format": "Bo5",
                "series_format_source": SCHEDULE_SOURCE,
            },
            {
                "gameid": "LOLTMNT01_1",
                "teamname": "Anyone's Legend",
                "source_series_id": "stale",
                "series_format": "Bo5",
                "series_format_source": SCHEDULE_SOURCE,
            },
            {
                "gameid": "GRID_1",
                "teamname": "Other",
                "source_series_id": "grid-series",
                "series_format": "Bo3",
                "series_format_source": "GRID verified summary",
            },
        ]
    )

    result = annotate_scheduled_series(rows, _schedule())

    scheduled = result.rows[result.rows["gameid"].eq("LOLTMNT01_1")]
    grid = result.rows[result.rows["gameid"].eq("GRID_1")]
    assert set(scheduled["source_series_id"]) == {"LPL/2026_1"}
    assert set(scheduled["series_format"]) == {"Bo3"}
    assert set(grid["source_series_id"]) == {"grid-series"}
    assert set(grid["series_format"]) == {"Bo3"}


def test_fixed_two_game_schedule_is_preserved_but_not_called_best_of() -> None:
    rows = pd.DataFrame(
        [
            {
                "game_uid": "LOLTMNT01_1",
                "blue_team": "Bilibili Gaming",
                "red_team": "Anyone's Legend",
            }
        ]
    )

    result = annotate_scheduled_series(rows, _schedule(best_of=2))

    assert result.audit["fixed_game_series_quarantined"] == 1
    assert result.rows.loc[0, "leaguepedia_best_of"] == 2
    assert pd.isna(result.rows.loc[0, "series_format"])


def test_conflicting_platform_game_schedule_is_rejected() -> None:
    schedule = pd.concat(
        [
            _schedule(),
            _schedule().assign(match_id="other-match"),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="conflicting"):
        validate_schedule(schedule)
