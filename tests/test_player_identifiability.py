from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.ratings.player_identifiability import (
    build_player_outcome_identifiability,
)


class PlayerIdentifiabilityTests(unittest.TestCase):
    def test_players_with_identical_signed_map_history_share_group(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "gameid": game,
                    "side": side,
                    "position": role,
                    "playername": player,
                    "teamname": team,
                }
                for game in ("g1", "g2")
                for side, team, names in (
                    ("Blue", "A", (("A-top", "top"), ("A-mid", "mid"))),
                    ("Red", "B", (("B-top", "top"), ("B-mid", "mid"))),
                )
                for player, role in names
            ]
        )
        audit = build_player_outcome_identifiability(players).set_index(
            "player"
        )
        self.assertEqual(
            audit.loc["A-top", "outcome_exposure_group_id"],
            audit.loc["A-mid", "outcome_exposure_group_id"],
        )
        self.assertEqual(
            audit.loc["A-top", "outcome_exposure_group_size"], 2
        )
        self.assertFalse(
            bool(audit.loc["A-top", "outcome_separately_identified"])
        )
        self.assertEqual(
            audit.loc["A-top", "outcome_identical_players"],
            ["A-mid", "A-top"],
        )

    def test_roster_move_creates_distinct_outcome_design_column(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "gameid": "g1",
                    "side": "Blue",
                    "position": "top",
                    "playername": "Mover",
                    "teamname": "A",
                },
                {
                    "gameid": "g1",
                    "side": "Blue",
                    "position": "mid",
                    "playername": "Teammate",
                    "teamname": "A",
                },
                {
                    "gameid": "g2",
                    "side": "Red",
                    "position": "top",
                    "playername": "Mover",
                    "teamname": "B",
                },
            ]
        )
        audit = build_player_outcome_identifiability(players).set_index(
            "player"
        )
        self.assertTrue(
            bool(audit.loc["Mover", "outcome_separately_identified"])
        )
        self.assertEqual(audit.loc["Mover", "n_distinct_teams"], 2)


if __name__ == "__main__":
    unittest.main()
