"""Adversarial fixtures for explicit series identity (issue #44).

The old fallback grouped maps by ``date floored to four hours | sorted teams``
and collapsed the group with a first-row tie shortcut.  That merged unrelated
same-day Bo1 matches into one observation.  These fixtures pin the new
contract:

* authoritative GRID series id groups maps only when the team pair is stable;
* every map without a safe series id stays its own game-level observation;
* side exposure is preserved for every map of a true series;
* tied/incomplete feeds are audited and excluded from primary inference;
* the first row is never selected as an outcome shortcut.
"""

from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.ratings.hierarchical_bt import _observations, fit_hierarchical_bt


def _map(date: str, blue: str, red: str, y: int, **extra) -> dict[str, object]:
    row: dict[str, object] = {
        "date": date,
        "league": "LEC",
        "blue_team": blue,
        "red_team": red,
        "y_blue_win": y,
        "game_uid": extra.pop("game_uid", f"uid-{date}-{blue}-{red}"),
    }
    row.update(extra)
    return row


class SeriesIdentityTests(unittest.TestCase):
    def test_same_day_bo1_same_teams_are_separate_observations(self) -> None:
        # The issue fixture: Rogue vs SK Gaming twice on the same day, both
        # "game 1", opposite outcomes, inside the old four-hour bucket.
        maps = pd.DataFrame(
            [
                _map("2024-03-25 17:01:20", "Rogue", "SK Gaming", 1, game=1),
                _map("2024-03-25 19:57:18", "Rogue", "SK Gaming", 0, game=1),
            ]
        )
        obs, audit = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 2)
        self.assertEqual(set(obs["series_source"]), {"none"})
        self.assertEqual(int(obs["n_maps"].sum()), 2)
        self.assertEqual(audit["n_unresolved_maps"], 0)
        snapshot, meta = fit_hierarchical_bt(maps, write=False)
        self.assertEqual(meta["n_series"], 2)
        self.assertEqual(meta["n_maps"], 2)
        self.assertEqual(meta["series_identity"]["n_authoritative_series"], 0)
        self.assertEqual(meta["series_identity"]["n_game_level_maps"], 2)
        self.assertFalse(snapshot.empty)

    def test_two_unrelated_matches_inside_four_hours_stay_separate(self) -> None:
        maps = pd.DataFrame(
            [
                _map("2026-01-01 10:10:00", "Team A", "Team B", 1, game=1),
                _map("2026-01-01 12:50:00", "Team C", "Team D", 0, game=1),
            ]
        )
        obs, _ = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 2)
        self.assertEqual(int(obs["n_maps"].sum()), 2)

    def test_repeated_game_numbers_are_distinct_observations(self) -> None:
        # Both matches are "game 1"; only the game uid separates them.
        maps = pd.DataFrame(
            [
                _map("2026-01-01 10:00:00", "Team A", "Team B", 1, game=1),
                _map("2026-01-01 11:00:00", "Team A", "Team B", 0, game=1),
            ]
        )
        obs, _ = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 2)
        self.assertEqual(len(set(obs["series_key"])), 2)

    def test_missing_series_ids_preserve_game_level_observations(self) -> None:
        maps = pd.DataFrame(
            [
                _map("2026-01-01 10:00:00", "Team A", "Team B", 1),
                _map("2026-01-02 10:00:00", "Team A", "Team B", 0),
            ]
        )
        obs, _ = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 2)
        self.assertTrue((obs["series_source"] == "none").all())

    def test_real_series_keeps_side_exposure_for_every_map(self) -> None:
        # Bo3: A wins map 1 from blue, wins map 2 from red, loses map 3.
        maps = pd.DataFrame(
            [
                _map("2026-01-01 10:00:00", "Team A", "Team B", 1, grid_series_id="s1", game=1),
                _map("2026-01-01 10:35:00", "Team B", "Team A", 0, grid_series_id="s1", game=2),
                _map("2026-01-01 11:10:00", "Team A", "Team B", 0, grid_series_id="s1", game=3),
            ]
        )
        obs, audit = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 1)
        row = obs.iloc[0]
        self.assertEqual(row["series_source"], "grid")
        self.assertEqual(int(row["n_maps"]), 3)
        self.assertEqual(float(row["y_a"]), 1.0)  # 2-1 majority, not first row
        self.assertAlmostEqual(float(row["a_blue_share"]), 2.0 / 3.0)
        self.assertAlmostEqual(2.0 * float(row["a_blue_share"]) - 1.0, 1.0 / 3.0)
        self.assertEqual(audit["n_unresolved_maps"], 0)

    def test_tied_multimap_series_is_unresolved_and_audited(self) -> None:
        # A 2-map 1-1 feed cannot be resolved; it must not use the first row.
        maps = pd.DataFrame(
            [
                _map("2026-01-01 10:00:00", "Team A", "Team B", 1, grid_series_id="s-tie", game=1),
                _map("2026-01-01 10:35:00", "Team B", "Team A", 1, grid_series_id="s-tie", game=2),
                _map("2026-01-01 12:00:00", "Team C", "Team D", 1, grid_series_id="s-ok", game=1),
            ]
        )
        obs, audit = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 1)
        self.assertEqual(obs.iloc[0]["series_key"], "grid:s-ok")
        self.assertEqual(audit["n_unresolved_maps"], 2)
        self.assertEqual(audit["n_unresolved_series"], 1)
        self.assertIn("s-tie", audit["unresolved_series_ids"])
        self.assertEqual(len(audit["unresolved_map_uids"]), 2)
        snapshot, meta = fit_hierarchical_bt(maps, write=False)
        self.assertEqual(meta["n_series"], 1)
        self.assertEqual(meta["n_maps"], 1)
        self.assertEqual(meta["series_identity"]["n_unresolved_maps"], 2)
        self.assertFalse(snapshot.empty)

    def test_corrupt_series_id_reused_for_different_pairs_falls_back_to_game_level(self) -> None:
        maps = pd.DataFrame(
            [
                _map("2026-01-01 10:00:00", "Team A", "Team B", 1, grid_series_id="shared"),
                _map("2026-01-01 10:35:00", "Team C", "Team D", 0, grid_series_id="shared"),
            ]
        )
        obs, audit = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 2)
        self.assertTrue((obs["series_source"] == "none").all())
        self.assertEqual(audit["unsafe_series_ids"], ["shared"])
        self.assertEqual(audit["n_unsafe_maps"], 2)

    def test_first_row_is_never_selected_as_a_tie_shortcut(self) -> None:
        # First map says A wins; the 2-1 majority says B wins.  The collapse
        # must follow the majority of all maps.
        maps = pd.DataFrame(
            [
                _map("2026-01-01 10:00:00", "Team A", "Team B", 1, grid_series_id="s-x", game=1),
                _map("2026-01-01 10:35:00", "Team B", "Team A", 1, grid_series_id="s-x", game=2),
                _map("2026-01-01 11:10:00", "Team B", "Team A", 1, grid_series_id="s-x", game=3),
            ]
        )
        obs, _ = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 1)
        self.assertEqual(float(obs.iloc[0]["y_a"]), 0.0)  # B won the series

    def test_empty_input_returns_empty_audit(self) -> None:
        obs, audit = _observations(pd.DataFrame(), None, 365.0)
        self.assertTrue(obs.empty)
        self.assertEqual(audit["n_unresolved_maps"], 0)


class TeamWeeklyRanksMovementTests(unittest.TestCase):
    def test_team_weekly_ranks_use_previous_refresh_baseline(self) -> None:
        from lol_kills.export.pack_records import build_maps_frame_from_team_games
        from lol_kills.ratings.hierarchical_bt import build_team_weekly_ranks

        team_games = pd.DataFrame(
            [
                {"game_uid": "g1", "side": "Blue", "teamname": "Team A", "date": "2026-07-10 10:00:00", "league": "LEC", "tournament": "LEC 2026", "result": 1},
                {"game_uid": "g1", "side": "Red", "teamname": "Team B", "date": "2026-07-10 10:00:00", "league": "LEC", "tournament": "LEC 2026", "result": 0},
                {"game_uid": "g2", "side": "Blue", "teamname": "Team B", "date": "2026-07-20 10:00:00", "league": "LEC", "tournament": "LEC 2026", "result": 1},
                {"game_uid": "g2", "side": "Red", "teamname": "Team A", "date": "2026-07-20 10:00:00", "league": "LEC", "tournament": "LEC 2026", "result": 0},
            ]
        )
        maps = build_maps_frame_from_team_games(team_games)
        cutoff = pd.Timestamp("2026-07-26T12:00:00")
        baseline = build_team_weekly_ranks(maps, as_of=cutoff, min_series=1)
        self.assertEqual(baseline["previous_as_of"], "2026-07-19T00:00:00Z")
        anchored = build_team_weekly_ranks(
            maps, as_of=cutoff, min_series=1,
            previous_as_of=pd.Timestamp("2026-07-24T00:00:00"),
        )
        self.assertEqual(anchored["previous_as_of"], "2026-07-24T00:00:00Z")
        self.assertEqual(anchored["current_through"], "2026-07-26T12:00:00Z")
        self.assertIn("Team A", anchored["by_team"])
        self.assertIn("delta", anchored["by_team"]["Team A"])
        guarded = build_team_weekly_ranks(
            maps, as_of=cutoff, min_series=1,
            previous_as_of=pd.Timestamp("2026-07-26T13:00:00"),
        )
        self.assertEqual(guarded["previous_as_of"], "2026-07-19T00:00:00Z")
        # CLI-style tz-aware ISO input (trailing Z) must not raise: the
        # refresh passes --previous-as-of from the previous bundle verbatim.
        cli_style = build_team_weekly_ranks(
            maps, as_of=cutoff, min_series=1,
            previous_as_of=pd.Timestamp("2026-07-24T00:00:00Z"),
        )
        self.assertEqual(cli_style["previous_as_of"], "2026-07-24T00:00:00Z")
        cli_same_day = build_team_weekly_ranks(
            maps, as_of=pd.Timestamp("2026-07-24T12:00:00"), min_series=1,
            previous_as_of=pd.Timestamp("2026-07-24T12:00:00Z"),
        )
        # Friday cutoff: current week Sunday is 07-19, so the fallback is 07-12.
        self.assertEqual(cli_same_day["previous_as_of"], "2026-07-12T00:00:00Z")


class PackAdapterSeriesIdentityTests(unittest.TestCase):
    def test_team_games_adapter_carries_grid_series_id(self) -> None:
        from lol_kills.export.pack_records import build_maps_frame_from_team_games

        team_games = pd.DataFrame(
            [
                {
                    "game_uid": "g1", "side": "Blue", "teamname": "Team A",
                    "date": "2026-01-01 10:00:00", "league": "LEC", "tournament": "LEC 2026",
                    "result": 1, "grid_series_id": "series-1",
                },
                {
                    "game_uid": "g1", "side": "Red", "teamname": "Team B",
                    "date": "2026-01-01 10:00:00", "league": "LEC", "tournament": "LEC 2026",
                    "result": 0, "grid_series_id": "series-1",
                },
                {
                    "game_uid": "g2", "side": "Blue", "teamname": "Team C",
                    "date": "2026-01-02 10:00:00", "league": "LEC", "tournament": "LEC 2026",
                    "result": 1, "grid_series_id": "",
                },
                {
                    "game_uid": "g2", "side": "Red", "teamname": "Team D",
                    "date": "2026-01-02 10:00:00", "league": "LEC", "tournament": "LEC 2026",
                    "result": 0, "grid_series_id": "",
                },
            ]
        )
        maps = build_maps_frame_from_team_games(team_games)
        self.assertEqual(len(maps), 2)
        self.assertIn("grid_series_id", maps.columns)
        self.assertEqual(maps.loc[maps["game_uid"] == "g1", "grid_series_id"].iloc[0], "series-1")
        self.assertEqual(maps.loc[maps["game_uid"] == "g2", "grid_series_id"].iloc[0], "")
        obs, _ = _observations(maps, None, 365.0)
        self.assertEqual(len(obs), 2)
        self.assertEqual(obs.loc[obs["game_uid"] == "g1", "series_source"].iloc[0], "grid")
        self.assertEqual(obs.loc[obs["game_uid"] == "g2", "series_source"].iloc[0], "none")


if __name__ == "__main__":
    unittest.main()
