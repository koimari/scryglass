from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.etl.series_ledger import build_canonical_series_ledger


def _map(
    uid: str,
    date: str,
    game: int,
    *,
    blue: str = "A",
    red: str = "B",
    blue_win: int = 1,
    series_format: str | None = "Bo5",
) -> dict:
    return {
        "game_uid": uid,
        "date": date,
        "league": "LPL",
        "split": "Split 3",
        "playoffs": 0,
        "source": "oe",
        "game": game,
        "blue_team": blue,
        "red_team": red,
        "y_blue_win": blue_win,
        "series_format": series_format,
    }


class CanonicalSeriesLedgerTest(unittest.TestCase):
    def test_long_bo5_is_one_series_and_is_row_order_invariant(self) -> None:
        maps = pd.DataFrame(
            [
                _map("g1", "2026-01-23T08:00:00Z", 1),
                _map("g2", "2026-01-23T09:05:00Z", 2, blue_win=0),
                _map("g3", "2026-01-23T10:10:00Z", 3),
                _map("g4", "2026-01-23T11:15:00Z", 4, blue_win=0),
                _map("g5", "2026-01-23T12:20:00Z", 5),
            ]
        )
        first = build_canonical_series_ledger(maps)
        shuffled = build_canonical_series_ledger(
            maps.sample(frac=1.0, random_state=17).reset_index(drop=True)
        )

        self.assertEqual(len(first.series), 1)
        series = first.series.iloc[0]
        self.assertEqual(series["completion_status"], "completed")
        self.assertTrue(series["rating_eligible"])
        self.assertEqual(series["score_a"], 3)
        self.assertEqual(series["score_b"], 2)
        self.assertEqual(
            first.series["canonical_series_id"].tolist(),
            shuffled.series["canonical_series_id"].tolist(),
        )
        self.assertEqual(
            first.maps["canonical_game_index"].tolist(),
            [1, 2, 3, 4, 5],
        )

    def test_missing_scheduled_format_fails_closed(self) -> None:
        maps = pd.DataFrame(
            [
                _map(
                    "g1",
                    "2026-01-01T10:00:00Z",
                    1,
                    series_format=None,
                ),
                _map(
                    "g2",
                    "2026-01-01T11:00:00Z",
                    2,
                    series_format=None,
                ),
            ]
        )
        result = build_canonical_series_ledger(maps)
        series = result.series.iloc[0]
        self.assertFalse(series["rating_eligible"])
        self.assertIn("unverified_series_format", series["quarantine_reasons"])
        self.assertTrue(result.maps["canonical_game_index"].isna().all())

    def test_gapped_explicit_series_is_quarantined(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    **_map("g1", "2026-01-01T10:00:00Z", 1, series_format="Bo3"),
                    "grid_series_id": "series-1",
                    "grid_game_index": 1,
                    "source": "grid",
                },
                {
                    **_map("g3", "2026-01-01T12:00:00Z", 3, series_format="Bo3"),
                    "grid_series_id": "series-1",
                    "grid_game_index": 3,
                    "source": "grid",
                },
            ]
        )
        result = build_canonical_series_ledger(maps)
        series = result.series.iloc[0]
        self.assertFalse(series["rating_eligible"])
        self.assertIn(
            "non_contiguous_source_game_index", series["quarantine_reasons"]
        )

    def test_game_number_reset_creates_distinct_series(self) -> None:
        maps = pd.DataFrame(
            [
                _map("s1g1", "2026-01-01T10:00:00Z", 1, series_format="Bo1"),
                _map("s2g1", "2026-01-01T15:00:00Z", 1, series_format="Bo1"),
            ]
        )
        result = build_canonical_series_ledger(maps)
        self.assertEqual(len(result.series), 2)
        self.assertTrue(result.series["rating_eligible"].all())
        self.assertEqual(result.maps["canonical_series_id"].nunique(), 2)

    def test_rebuild_replaces_stale_canonical_output_columns(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    **_map(
                        "g1",
                        "2026-01-01T10:00:00Z",
                        1,
                        series_format="Bo3",
                    ),
                    "canonical_series_id": "stale-series",
                    "canonical_game_index": 99,
                    "canonical_series_status": "completed",
                    "canonical_series_completion_source": "stale",
                    "canonical_series_winner_team_key": "wrong-team",
                    "scheduled_best_of": 5,
                    "series_quarantine_reasons": ["stale"],
                    "series_rating_eligible": False,
                },
                {
                    **_map(
                        "g2",
                        "2026-01-01T11:00:00Z",
                        2,
                        blue_win=0,
                        series_format="Bo3",
                    ),
                    "canonical_series_id": None,
                    "canonical_game_index": None,
                    "canonical_series_status": None,
                    "canonical_series_completion_source": None,
                    "canonical_series_winner_team_key": None,
                    "scheduled_best_of": None,
                    "series_quarantine_reasons": None,
                    "series_rating_eligible": None,
                },
                {
                    **_map(
                        "g3",
                        "2026-01-01T12:00:00Z",
                        3,
                        series_format="Bo3",
                    ),
                    "canonical_series_id": "another-stale-series",
                    "canonical_game_index": 7,
                    "canonical_series_status": "invalid",
                    "canonical_series_completion_source": "stale",
                    "canonical_series_winner_team_key": "wrong-team",
                    "scheduled_best_of": 1,
                    "series_quarantine_reasons": ["stale"],
                    "series_rating_eligible": False,
                },
            ]
        )

        result = build_canonical_series_ledger(maps)

        self.assertEqual(result.maps["canonical_series_id"].nunique(), 1)
        self.assertEqual(
            result.maps["canonical_game_index"].tolist(),
            [1, 2, 3],
        )
        self.assertTrue(result.maps["series_rating_eligible"].all())
        self.assertEqual(
            set(result.maps["canonical_series_status"]),
            {"completed"},
        )
        self.assertEqual(result.series.iloc[0]["scheduled_best_of"], 3)

    def test_score_exceeding_format_is_invalid(self) -> None:
        maps = pd.DataFrame(
            [
                _map("g1", "2026-01-01T10:00:00Z", 1, series_format="Bo3"),
                _map("g2", "2026-01-01T11:00:00Z", 2, series_format="Bo3"),
                _map("g3", "2026-01-01T12:00:00Z", 3, series_format="Bo3"),
            ]
        )
        result = build_canonical_series_ledger(maps)
        series = result.series.iloc[0]
        self.assertEqual(series["completion_status"], "invalid")
        self.assertIn(
            "score_exceeds_scheduled_format", series["quarantine_reasons"]
        )
        self.assertFalse(series["rating_eligible"])


if __name__ == "__main__":
    unittest.main()
