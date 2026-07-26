from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.export.public_pack import _draft_coverage


ROLES = ("top", "jng", "mid", "bot", "sup")


def participant_rows(gameid: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "gameid": gameid,
                "side": side,
                "position": role,
                "champion": f"{side}-{role}",
            }
            for side in ("Blue", "Red")
            for role in ROLES
        ]
    )


class PublicPackDraftCoverageTests(unittest.TestCase):
    def test_accepts_role_aligned_participant_fallback(self) -> None:
        maps = pd.DataFrame([{"oe_gameid": "grid-1"}])
        result = _draft_coverage(maps, participant_rows("grid-1"))

        self.assertEqual(result["maps"], 1)
        self.assertEqual(result["map_pick_rows"], 0)
        self.assertEqual(result["participant_fallback_rows"], 1)
        self.assertEqual(result["complete_rows"], 1)
        self.assertEqual(result["coverage_rate"], 1.0)

    def test_accepts_complete_map_picks_without_player_rows(self) -> None:
        row = {"oe_gameid": "oe-1"}
        for side in ("blue", "red"):
            for index in range(1, 6):
                row[f"{side}_pick{index}"] = f"{side}-{index}"

        result = _draft_coverage(pd.DataFrame([row]), pd.DataFrame())
        self.assertEqual(result["map_pick_rows"], 1)
        self.assertEqual(result["participant_fallback_rows"], 0)

    def test_rejects_unknown_or_missing_participant_champion(self) -> None:
        players = participant_rows("broken-1")
        players.loc[
            (players["side"] == "Red") & (players["position"] == "sup"),
            "champion",
        ] = "Unknown"

        with self.assertRaisesRegex(ValueError, "draft coverage failed"):
            _draft_coverage(pd.DataFrame([{"oe_gameid": "broken-1"}]), players)

    def test_rejects_duplicate_role_in_participant_fallback(self) -> None:
        players = participant_rows("broken-2")
        players.loc[
            (players["side"] == "Red") & (players["position"] == "sup"),
            "position",
        ] = "mid"

        with self.assertRaisesRegex(ValueError, "draft coverage failed"):
            _draft_coverage(pd.DataFrame([{"oe_gameid": "broken-2"}]), players)


if __name__ == "__main__":
    unittest.main()
