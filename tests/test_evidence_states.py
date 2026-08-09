"""Adversarial fixtures for validated evidence states (issue #46).

The old UI labeled a rating Settled when sigma reached a floor.  These tests
pin the validated contract: interval width, relative precision, stability,
freshness, support coverage, fallback, active, disconnected, and OOD fields
drive a fail-closed evidence state; sigma and map counts stay diagnostics.
"""

from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.ratings.evidence import (
    STALE_DAYS,
    WIDE_INTERVAL_WIDTH,
    attach_player_evidence,
    attach_team_evidence,
    evidence_state,
)


def team_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "team": "Team A",
                "team_key": "team a",
                "mu_total": 1550.0,
                "sigma": 20.0,
                "rating_p10": 1508.9,
                "n_series": 12,
                "n_maps": 30,
                "international_series": 4,
                "home_league": "LCK",
                "last_game_date": "2026-07-28T20:00:00",
                "model": "hierarchical_bt",
            },
            {
                "team": "Team B",
                "team_key": "team b",
                "mu_total": 1520.0,
                "sigma": 60.0,
                "rating_p10": 1421.4,
                "n_series": 3,
                "n_maps": 9,
                "international_series": 0,
                "home_league": "LCS",
                "last_game_date": "2026-01-01T20:00:00",
                "model": "hierarchical_bt",
            },
        ]
    )


def player_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player": "P1",
                "mu_total": 1550.0,
                "sigma": 28.0,
                "n_maps": 40,
                "last_team": "Team A",
                "home_league": "LCK",
                "last_game_date": "2026-07-28T20:00:00",
                "evidence_stability": 0.6,
            },
            {
                "player": "P2",
                "mu_total": 1500.0,
                "sigma": 90.0,
                "n_maps": 0,
                "last_team": None,
                "home_league": None,
                "last_game_date": None,
                "evidence_stability": None,
            },
        ]
    )


class EvidenceStateUnitTests(unittest.TestCase):
    def test_missing_fields_fail_closed_as_unsupported(self) -> None:
        self.assertEqual(evidence_state({}), "unsupported")
        self.assertEqual(
            evidence_state({"evidence_interval_width": 100, "evidence_precision_ratio": 1.0}),
            "unsupported",
        )
        self.assertEqual(
            evidence_state(
                {
                    "evidence_interval_width": float("nan"),
                    "evidence_precision_ratio": 1.0,
                    "evidence_support_coverage": 1.0,
                    "evidence_fallback": 0,
                    "evidence_active": 1,
                    "evidence_disconnected": 0,
                    "evidence_ood": 0,
                }
            ),
            "unsupported",
        )

    def test_exactly_95_percent_precision_is_not_settled(self) -> None:
        fields = {
            "evidence_interval_width": 98.0,
            "evidence_precision_ratio": 0.95,
            "evidence_stability": 0.5,
            "evidence_freshness_days": 1,
            "evidence_support_coverage": 1.0,
            "evidence_fallback": 0,
            "evidence_active": 1,
            "evidence_disconnected": 0,
            "evidence_ood": 0,
        }
        self.assertEqual(evidence_state(fields), "observed")

    def test_precision_above_95_with_all_conditions_is_settled(self) -> None:
        fields = {
            "evidence_interval_width": 98.0,
            "evidence_precision_ratio": 0.951,
            "evidence_stability": 0.5,
            "evidence_freshness_days": 1,
            "evidence_support_coverage": 1.0,
            "evidence_fallback": 0,
            "evidence_active": 1,
            "evidence_disconnected": 0,
            "evidence_ood": 0,
        }
        self.assertEqual(evidence_state(fields), "settled")

    def test_fail_closed_states(self) -> None:
        base = {
            "evidence_interval_width": 98.0,
            "evidence_precision_ratio": 1.0,
            "evidence_stability": 0.5,
            "evidence_freshness_days": 1,
            "evidence_support_coverage": 1.0,
            "evidence_fallback": 0,
            "evidence_active": 1,
            "evidence_disconnected": 0,
            "evidence_ood": 0,
        }
        self.assertEqual(evidence_state({**base, "evidence_freshness_days": STALE_DAYS + 1}), "stale")
        self.assertEqual(evidence_state({**base, "evidence_active": 0}), "inactive")
        self.assertEqual(
            evidence_state({**base, "evidence_interval_width": WIDE_INTERVAL_WIDTH + 1}),
            "wide_interval",
        )
        self.assertEqual(evidence_state({**base, "evidence_fallback": 1}), "fallback")
        self.assertEqual(evidence_state({**base, "evidence_disconnected": 1}), "disconnected")
        self.assertEqual(evidence_state({**base, "evidence_ood": 1}), "ood")
        self.assertEqual(evidence_state({**base, "evidence_stability": None}), "thin")


class TeamEvidenceAttachmentTests(unittest.TestCase):
    def test_team_evidence_fields_and_states(self) -> None:
        source_as_of = pd.Timestamp("2026-07-30T00:00:00")
        weekly = {"team a": 0.3}
        out = attach_team_evidence(
            team_frame(),
            source_as_of=source_as_of,
            weekly_stability=weekly,
        )
        a = out.iloc[0]
        b = out.iloc[1]
        # Interval width is 2 * z95 * sigma, not sigma.
        self.assertAlmostEqual(a["evidence_interval_width"], 2 * 1.959963984540054 * 20.0)
        # Precision ratio: relative information versus the registered
        # reference (scope median sigma floored at the model floor).  Median
        # of (20, 60) is 40, floored at 20 -> reference 40.  Team A (sigma
        # 20) is 1.0; Team B is (40/60)^2.
        self.assertAlmostEqual(a["evidence_precision_ratio"], 1.0)
        self.assertAlmostEqual(b["evidence_precision_ratio"], (40.0 / 60.0) ** 2)
        # Freshness from source as-of (28 hours before the as-of stamp).
        self.assertAlmostEqual(a["evidence_freshness_days"], 28.0 / 24.0, places=4)
        # Support coverage vs the 8-series target.
        self.assertAlmostEqual(a["evidence_support_coverage"], 1.0)
        self.assertAlmostEqual(b["evidence_support_coverage"], 3.0 / 8.0)
        # Stability comes from the weekly movement mapping.
        self.assertAlmostEqual(a["evidence_stability"], 0.3)
        self.assertTrue(pd.isna(b["evidence_stability"]))
        # States: A settled, B stale (last game in January).
        self.assertEqual(a["evidence_state"], "settled")
        self.assertEqual(b["evidence_state"], "stale")

    def test_team_disconnected_league_fails_closed(self) -> None:
        frame = team_frame()
        frame.loc[0, "home_league"] = "UNKNOWN"
        frame.loc[0, "international_series"] = 0
        out = attach_team_evidence(frame, source_as_of=pd.Timestamp("2026-07-30T00:00:00"))
        self.assertEqual(out.iloc[0]["evidence_disconnected"], 1)
        self.assertEqual(out.iloc[0]["evidence_state"], "disconnected")

    def test_unknown_league_with_international_bridge_is_not_disconnected(self) -> None:
        frame = team_frame()
        frame.loc[0, "home_league"] = "UNKNOWN"
        # International series provide a bridge, so the row stays supported
        # (thin) instead of failing to Disconnected.
        out = attach_team_evidence(frame, source_as_of=pd.Timestamp("2026-07-30T00:00:00"))
        self.assertEqual(out.iloc[0]["evidence_disconnected"], 0)
        self.assertNotEqual(out.iloc[0]["evidence_state"], "disconnected")

    def test_empty_frame_gains_evidence_columns(self) -> None:
        out = attach_team_evidence(
            pd.DataFrame(columns=["team", "team_key", "sigma", "n_series", "home_league"]),
            source_as_of=pd.Timestamp("2026-07-30T00:00:00"),
        )
        self.assertTrue(out.empty)
        self.assertIn("evidence_state", out.columns)


class PlayerEvidenceAttachmentTests(unittest.TestCase):
    def test_player_evidence_fields_and_states(self) -> None:
        out = attach_player_evidence(
            player_frame(),
            source_as_of=pd.Timestamp("2026-07-30T00:00:00"),
        )
        p1 = out.iloc[0]
        p2 = out.iloc[1]
        self.assertAlmostEqual(p1["evidence_precision_ratio"], 1.0)
        self.assertAlmostEqual(p1["evidence_stability"], 0.6)
        self.assertEqual(p1["evidence_fallback"], 0)
        self.assertEqual(p1["evidence_active"], 1)
        self.assertEqual(p1["evidence_state"], "settled")
        # Fallback prior row: no games, no team.
        self.assertEqual(p2["evidence_fallback"], 1)
        self.assertEqual(p2["evidence_active"], 0)
        self.assertEqual(p2["evidence_state"], "fallback")

    def test_inactive_player_fails_closed(self) -> None:
        frame = player_frame()
        # 66 days before as-of: inside the stale window, outside the active window.
        frame.loc[0, "last_game_date"] = "2026-05-25T00:00:00"
        out = attach_player_evidence(frame, source_as_of=pd.Timestamp("2026-07-30T00:00:00"))
        self.assertEqual(out.iloc[0]["evidence_active"], 0)
        self.assertEqual(out.iloc[0]["evidence_state"], "inactive")

    def test_disconnected_and_ood_players(self) -> None:
        frame = player_frame()
        frame.loc[0, "last_team"] = None
        frame.loc[1, "n_maps"] = 5
        frame.loc[1, "home_league"] = "UNKNOWN"
        frame.loc[1, "last_game_date"] = "2026-07-28T20:00:00"
        frame.loc[1, "evidence_stability"] = 1.0
        out = attach_player_evidence(frame, source_as_of=pd.Timestamp("2026-07-30T00:00:00"))
        self.assertEqual(out.iloc[0]["evidence_state"], "disconnected")
        self.assertEqual(out.iloc[1]["evidence_state"], "ood")


if __name__ == "__main__":
    unittest.main()
