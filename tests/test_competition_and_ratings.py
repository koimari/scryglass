from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.etl.competition import canonicalize_competition_frame, classify_competition
from lol_kills.export.pack_records import build_player_records, build_team_records
from lol_kills.ratings.dual_elo import _is_intl
from lol_kills.ratings.hierarchical_bt import fit_hierarchical_bt
from lol_kills.ratings.validation import audit_rating_inputs


class CompetitionIdentityTests(unittest.TestCase):
    def test_lta_n_maps_to_lcs_but_source_is_retained(self) -> None:
        frame = canonicalize_competition_frame(
            pd.DataFrame(
                [
                    {
                        "league": "LTA N",
                        "tournament": None,
                        "blue_team": "Team Liquid",
                        "red_team": "Cloud9",
                    }
                ]
            )
        )
        row = frame.iloc[0]
        self.assertEqual(row["league"], "LCS")
        self.assertEqual(row["league_source"], "LTA N")
        self.assertEqual(row["competition_scope"], "regional")
        self.assertFalse(bool(row["is_international"]))
        self.assertFalse(bool(row["is_interregional"]))
        self.assertEqual(row["blue_team_key"], "team-liquid")

    def test_lta_s_maps_to_cblol(self) -> None:
        row = canonicalize_competition_frame(
            pd.DataFrame([{"league": "LTA S", "blue_team": "FURIA", "red_team": "LOUD"}])
        ).iloc[0]
        self.assertEqual(row["league"], "CBLOL")
        self.assertEqual(row["league_source"], "LTA S")
        self.assertEqual(row["competition_scope"], "regional")
        self.assertFalse(bool(row["is_international"]))

    def test_generic_lta_is_interregional_not_domestic(self) -> None:
        label = classify_competition("LTA", None)
        self.assertEqual(label.league, "AMERICAS")
        self.assertEqual(label.scope, "interregional")
        self.assertEqual(label.event_kind, "americas_cross_region")
        self.assertFalse(label.is_international)
        self.assertTrue(label.is_interregional)

    def test_road_to_msi_does_not_become_international(self) -> None:
        label = classify_competition("LCK", "LCK 2026 Road to MSI")
        self.assertEqual(label.league, "LCK")
        self.assertEqual(label.scope, "regional")
        self.assertFalse(label.is_international)
        self.assertFalse(_is_intl("LCK", "LCK 2026 Road to MSI"))
        self.assertTrue(_is_intl("MSI", None))

    def test_team_records_merge_regional_and_international_appearances(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "league": "LTA N",
                    "tournament": None,
                    "blue_team": "Team Liquid",
                    "red_team": "Cloud9",
                    "y_blue_win": 1,
                },
                {
                    "date": "2026-01-02",
                    "league": "EWC",
                    "tournament": None,
                    "blue_team": "Team Liquid",
                    "red_team": "Cloud9",
                    "y_blue_win": 0,
                },
            ]
        )
        records = build_team_records(maps)
        self.assertEqual(set(records), {"Team Liquid", "Cloud9"})
        self.assertEqual(records["Team Liquid"]["primary"], "LCS")
        self.assertEqual(records["Team Liquid"]["leagues"], ["EWC", "LCS"])
        self.assertEqual(records["Team Liquid"]["source_leagues"], ["EWC", "LTA N"])

    def test_team_primary_uses_latest_domestic_affiliation(self) -> None:
        maps = pd.DataFrame(
            [
                {"date": "2025-01-01", "league": "PCS", "blue_team": "A", "red_team": "B", "y_blue_win": 1},
                {"date": "2026-01-01", "league": "LCP", "blue_team": "A", "red_team": "C", "y_blue_win": 1},
                {"date": "2026-02-01", "league": "LTA", "blue_team": "A", "red_team": "D", "y_blue_win": 1},
            ]
        )
        records = build_team_records(maps)
        self.assertEqual(records["A"]["primary"], "LCP")
        self.assertTrue(records["A"]["interregional"])

    def test_player_records_use_canonical_latest_domestic_league(self) -> None:
        players = pd.DataFrame(
            [
                {"date": "2025-01-01", "league": "LTA S", "playername": "Bot", "position": "bot", "result": 1},
                {"date": "2026-01-01", "league": "CBLOL", "playername": "Bot", "position": "bot", "result": 0},
                {"date": "2026-02-01", "league": "LTA", "playername": "Bot", "position": "bot", "result": 1},
            ]
        )
        records = build_player_records(players)
        self.assertEqual(records["Bot"]["primary"], "CBLOL")
        self.assertEqual(records["Bot"]["leagues"], ["AMERICAS", "CBLOL"])
        self.assertTrue(records["Bot"]["interregional"])


class HierarchicalRatingTests(unittest.TestCase):
    def test_input_audit_reports_legacy_labels_and_temporal_cutoffs(self) -> None:
        maps = pd.DataFrame(
            [
                {"date": "2026-01-01", "league": "LTA N", "blue_team": "A", "red_team": "B", "y_blue_win": 1},
                {"date": "2026-05-01", "league": "EWC", "blue_team": "A", "red_team": "C", "y_blue_win": 1},
                {"date": "2026-08-01", "league": "LCK", "blue_team": "C", "red_team": "D", "y_blue_win": 0},
            ]
        )
        audit = audit_rating_inputs(maps)
        self.assertTrue(audit["ok"])
        self.assertEqual(audit["deprecated_source_rows"], 1)
        self.assertEqual(audit["n_international_bridge_pairs"], 0)
        self.assertEqual(audit["canonical_leagues"], ["EWC", "LCK", "LCS"])

    def test_series_are_one_observation_and_snapshot_has_conservative_interval(self) -> None:
        maps = pd.DataFrame(
            [
                {"date": "2026-01-01 10:00", "league": "LCS", "blue_team": "A", "red_team": "B", "y_blue_win": 1, "grid_series_id": "s1"},
                {"date": "2026-01-01 10:35", "league": "LCS", "blue_team": "A", "red_team": "B", "y_blue_win": 1, "grid_series_id": "s1"},
                {"date": "2026-01-02 10:00", "league": "LCS", "blue_team": "A", "red_team": "B", "y_blue_win": 0, "grid_series_id": "s2"},
            ]
        )
        snapshot, meta = fit_hierarchical_bt(maps, write=False)
        self.assertEqual(meta["n_series"], 2)
        self.assertEqual(meta["n_maps"], 3)
        self.assertTrue((snapshot["rating_p10"] < snapshot["mu_total"]).all())
        self.assertTrue(set(snapshot["model"]) == {"hierarchical_bt"})


if __name__ == "__main__":
    unittest.main()
