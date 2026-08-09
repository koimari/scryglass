from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from lol_kills.market_decision import unavailable_authority
from tools.live_fair_odds import model
from tools.live_fair_odds.model import (
    ModelInputError,
    _market_view,
    _validated_live_totals,
    score_manual_state,
)


class MarketViewTests(unittest.TestCase):
    def test_diagnostic_probability_cannot_self_authorize_fair_odds_or_ev(self) -> None:
        view = _market_view(0.60, 2.0, 1.80)
        self.assertEqual(view["diagnostic_probability"], 0.60)
        self.assertIsNone(view["probability"])
        self.assertIsNone(view["fair_odds"])
        self.assertIsNone(view["expected_return_pct"])
        self.assertEqual(view["decision"], "NO_AUTHORIZED_BET")
        self.assertIsNotNone(view["no_vig_break_even_probability"])

    def test_two_sided_quote_is_required_even_for_descriptive_market_context(self) -> None:
        view = _market_view(0.60, 2.0, None)
        self.assertEqual(view["decision"], "NO_AUTHORIZED_BET")
        self.assertIn("two_way_quote_incomplete", view["blockers"])
        self.assertIsNone(view["no_vig_break_even_probability"])

    def test_missing_probability_withholds_winner_even_with_price(self) -> None:
        view = _market_view(None, 1.70, 2.20)
        self.assertEqual(view["decision"], "NO_AUTHORIZED_BET")
        self.assertEqual(view["offered_odds"], 1.70)
        self.assertIsNone(view["expected_return_pct"])


class LiveTotalsTests(unittest.TestCase):
    def test_series_cluster_interval_reaches_market_gate(self) -> None:
        priced = {
            "eligibility": {"status": "supported", "blockers": []},
            "projected_mean": 29.0,
            "uncertainty": {
                "status": "available",
                "method": "series_cluster_weighted_hoeffding",
                "confidence": 0.95,
                "blockers": [],
            },
            "lines": [
                {
                    "line": 28.5,
                    "under_probability": 0.60,
                    "over_probability": 0.40,
                    "under_probability_interval": [0.52, 0.68],
                    "over_probability_interval": [0.32, 0.48],
                    "uncertainty": {"effective_series_n": 31.0},
                }
            ],
        }
        with (
            patch.object(model, "price_live_totals", return_value=priced),
            patch.object(model, "_live_totals_artifact", return_value={}),
            patch.object(
                model,
                "_private_market_authority",
                return_value=unavailable_authority("test_authority_unavailable"),
            ),
            patch.object(
                model,
                "_registered_market_quote",
                return_value={
                    "status": "unavailable",
                    "quote": None,
                    "quote_sha256": None,
                    "blockers": ["test_quote_unavailable"],
                },
            ),
        ):
            result = _validated_live_totals(
                minute=10,
                current_kills=8,
                gold_difference=400.0,
                league="LCK",
                patch="16.14",
                as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
                blue_team="Blue",
                red_team="Red",
                champions=list("ABCDEFGHIJ"),
                lines=[{"line": 28.5, "under_odds": 1.9, "over_odds": 1.9}],
                event_id="event-1-map-1",
            )
        self.assertEqual(
            result["lines"][0]["under"]["probability_interval"],
            [0.52, 0.68],
        )
        self.assertEqual(
            result["lines"][0]["over"]["probability_interval"],
            [0.32, 0.48],
        )
        self.assertEqual(result["effective_n"], 31.0)
        self.assertEqual(result["uncertainty"]["status"], "available")

    def test_22_minutes_is_withheld_instead_of_borrowing_checkpoint_authority(self) -> None:
        priced = _validated_live_totals(
            minute=22,
            current_kills=14,
            gold_difference=1200,
            league="LCK",
            patch="16.14",
            as_of=datetime(2026, 7, 30, tzinfo=timezone.utc),
            blue_team="Dplus Kia",
            red_team="Hanwha Life Esports",
            champions=[
                "Renekton",
                "Skarner",
                "Ryze",
                "Viktor",
                "Pyke",
                "Aatrox",
                "Lee Sin",
                "Annie",
                "Jhin",
                "Bard",
            ],
            lines=[{"line": 32.5, "under_odds": 1.8, "over_odds": ""}],
        )
        self.assertFalse(priced["classification_available"])
        self.assertIn("minute_not_validated:22", priced["eligibility"]["blockers"])
        self.assertEqual(
            priced["lines"][0]["under"]["decision"],
            "NO_AUTHORIZED_BET",
        )


class IntegrationTests(unittest.TestCase):
    def test_registered_roster_does_not_self_authorize_missing_ratings(self) -> None:
        registered = {
            "status": "registered",
            "roster": {"teams": []},
            "receipt_sha256": "a" * 64,
            "registry_sha256": "b" * 64,
            "registry_id": "roster-registry-1",
            "capture_protocol_sha256": "c" * 64,
            "blockers": [],
        }
        with patch.object(
            model, "_registered_pregame_roster", return_value=registered
        ):
            result = score_manual_state(
                {
                    "league": "LPL",
                    "blue_team": "Bilibili Gaming",
                    "red_team": "LGD Gaming",
                    "blue_picks": ["K'Sante", "Skarner", "Orianna", "Yunara", "Lulu"],
                    "red_picks": ["Ambessa", "Lee Sin", "Ryze", "Varus", "Bard"],
                    "minute": 15,
                    "event_id": "lpl-example-map-1",
                    "event_start": "2026-08-01T19:00:00Z",
                    "draft_source_available_at": "2026-08-01T18:59:00Z",
                    "blue_kills": 5,
                    "red_kills": 7,
                    "lines": [{"line": 34.5}],
                }
            )
        strength = result["pregame_win"]["strength_expectation"]
        components = result["winner_reprice"]["component_authority"]
        self.assertTrue(strength["pre_event_roster_authorized"])
        self.assertFalse(strength["team_rating_authorized"])
        self.assertFalse(strength["player_rating_authorized"])
        self.assertTrue(components["pre_event_roster"]["authorized"])
        self.assertFalse(components["team_rating"]["authorized"])
        self.assertEqual(result["winner_reprice"]["mode"], "unavailable")

    def test_real_manual_state_prices_totals_and_withholds_win_without_gold(self) -> None:
        result = score_manual_state(
            {
                "league": "LPL",
                "blue_team": "Bilibili Gaming",
                "red_team": "LGD Gaming",
                "blue_picks": ["K'Sante", "Skarner", "Orianna", "Yunara", "Lulu"],
                "red_picks": ["Ambessa", "Lee Sin", "Ryze", "Varus", "Bard"],
                "minute": 15,
                "event_id": "lpl-example-map-1",
                "event_start": "2026-08-01T19:00:00Z",
                "draft_source_available_at": "2026-08-01T18:59:00Z",
                "blue_kills": 5,
                "red_kills": 7,
                "blue_gold": "",
                "red_gold": "",
                "blue_win_odds": 1.37,
                "red_win_odds": 2.95,
                "lines": [{"line": 34.5, "under_odds": 1.83, "over_odds": ""}],
            }
        )
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(result["live_win"]["p_blue"])
        self.assertEqual(result["winner_reprice"]["mode"], "unavailable")
        self.assertEqual(
            result["pregame_win"]["draft_score"]["model_kind"],
            "canonical_terminal_neutral",
        )
        self.assertFalse(
            result["pregame_win"]["strength_expectation"]["team_rating_authorized"]
        )
        self.assertFalse(
            result["pregame_win"]["strength_expectation"]["player_rating_authorized"]
        )
        self.assertEqual(
            result["winner_reprice"]["component_authority"]["draft_score"]["status"],
            "development_only",
        )
        self.assertFalse(
            result["winner_reprice"]["component_authority"]["team_rating"]["authorized"]
        )
        self.assertIsNotNone(result["pregame_win"]["draft_score"]["blue"])
        self.assertIsNone(result["pregame_win"]["p_blue"])
        self.assertIsNone(result["winner_reprice"]["diagnostic_p_blue"])
        self.assertIsNone(result["winner_reprice"]["blue"]["probability"])
        self.assertIsNone(result["winner_reprice"]["blue"]["expected_return_pct"])
        self.assertEqual(
            result["winner_reprice"]["blue"]["decision"],
            "NO_AUTHORIZED_BET",
        )
        self.assertFalse(result["live_totals"]["classification_available"])
        self.assertIsNone(result["live_totals"]["lines"][0]["under"]["probability"])
        self.assertEqual(len(result["live_totals"]["lines"]), 1)

    def test_rejects_incomplete_draft(self) -> None:
        with self.assertRaises(ModelInputError):
            score_manual_state(
                {
                    "league": "LPL",
                    "blue_team": "Bilibili Gaming",
                    "red_team": "LGD Gaming",
                    "blue_picks": ["K'Sante"],
                    "red_picks": ["Ambessa", "Lee Sin", "Ryze", "Varus", "Bard"],
                    "minute": 15,
                    "blue_kills": 5,
                    "red_kills": 7,
                    "lines": [{"line": 34.5}],
                }
            )

    def test_gold_does_not_create_live_diagnostic_without_rating_authority(self) -> None:
        result = score_manual_state(
            {
                "league": "LPL",
                "blue_team": "LGD Gaming",
                "red_team": "Bilibili Gaming",
                "blue_picks": ["Rumble", "Vi", "Annie", "Jhin", "Seraphine"],
                "red_picks": ["Aurora", "Poppy", "Akali", "Caitlyn", "Lux"],
                "minute": 15,
                "blue_kills": 5,
                "red_kills": 7,
                "blue_gold": 25000,
                "red_gold": 24000,
                "blue_win_odds": 2.95,
                "red_win_odds": 1.37,
                "lines": [{"line": 34.5}],
            }
        )
        self.assertEqual(result["winner_reprice"]["mode"], "unavailable")
        self.assertIsNone(result["live_win"]["diagnostic_p_blue"])
        self.assertIsNone(result["live_win"]["p_blue"])
        self.assertIn(
            "independently_registered_event_rating_unavailable",
            result["live_win"]["decision_blockers"],
        )
        self.assertIsNone(result["winner_reprice"]["diagnostic_p_blue"])
        self.assertIsNone(result["winner_reprice"]["blue"]["expected_return_pct"])
        self.assertEqual(
            result["winner_reprice"]["blue"]["decision"],
            "NO_AUTHORIZED_BET",
        )

    def test_registered_ratings_feed_only_the_live_development_diagnostic(self) -> None:
        registered_roster = {
            "status": "registered",
            "roster": {"teams": []},
            "receipt_sha256": "a" * 64,
            "registry_sha256": "b" * 64,
            "registry_id": "roster-registry-1",
            "capture_protocol_sha256": "c" * 64,
            "blockers": [],
        }
        registered_rating = {
            "status": "registered",
            "player_rating_authorized": True,
            "team_rating_authorized": True,
            "match_probability_authorized": False,
            "betting_decision_authorized": False,
            "ratings": {
                "strength_difference": {
                    "orientation": "blue_minus_red",
                    "posterior_mean": 75.0,
                    "posterior_interval_95": [-25.0, 175.0],
                }
            },
            "receipt_sha256": "d" * 64,
            "registry_sha256": "e" * 64,
            "blockers": [
                "rating_to_match_probability_calibration_unavailable",
                "draft_rating_combination_authority_unavailable",
            ],
        }
        with (
            patch.object(
                model,
                "_registered_pregame_roster",
                return_value=registered_roster,
            ),
            patch.object(
                model,
                "_registered_event_rating",
                return_value=registered_rating,
            ),
        ):
            result = score_manual_state(
                {
                    "league": "LPL",
                    "blue_team": "LGD Gaming",
                    "red_team": "Bilibili Gaming",
                    "blue_picks": ["Rumble", "Vi", "Annie", "Jhin", "Seraphine"],
                    "red_picks": ["Aurora", "Poppy", "Akali", "Caitlyn", "Lux"],
                    "minute": 15,
                    "event_id": "lpl-example-map-1",
                    "event_start": "2026-08-01T21:00:00Z",
                    "draft_source_available_at": "2026-08-01T20:59:00Z",
                    "blue_kills": 5,
                    "red_kills": 7,
                    "blue_gold": 25000,
                    "red_gold": 24000,
                    "lines": [{"line": 34.5}],
                }
            )
        strength = result["pregame_win"]["strength_expectation"]
        components = result["winner_reprice"]["component_authority"]
        self.assertTrue(strength["team_rating_authorized"])
        self.assertTrue(strength["player_rating_authorized"])
        self.assertFalse(strength["match_probability_authorized"])
        self.assertEqual(
            strength["strength_difference"]["posterior_mean"], 75.0
        )
        self.assertIsNotNone(result["live_win"]["diagnostic_p_blue"])
        self.assertIsNone(result["live_win"]["p_blue"])
        self.assertEqual(result["winner_reprice"]["mode"], "unavailable")
        self.assertTrue(components["team_rating"]["authorized"])
        self.assertTrue(components["player_rating"]["authorized"])
        self.assertFalse(components["rating_to_probability"]["authorized"])
        self.assertIsNone(result["winner_reprice"]["blue"]["probability"])
        self.assertIsNone(result["winner_reprice"]["blue"]["expected_return_pct"])


if __name__ == "__main__":
    unittest.main()
