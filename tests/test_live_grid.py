from __future__ import annotations

import io
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import lol_kills.etl.grid_ingest as grid_ingest
from lol_kills.etl.grid_ingest import _download
from lol_kills.etl.grid_series_events import (
    default_config,
    series_events_url,
    transaction_sequence,
    transaction_state,
)
from lol_kills.live_model import evaluate_live_state
from lol_kills.live_snapshots import LivePublisher, build_live_snapshot


class GridSeriesEventsTests(unittest.TestCase):
    def test_source_merge_normalizes_oe_and_grid_date_types(self) -> None:
        merged = grid_ingest.merge_source_frames(
            pd.DataFrame(
                [{"gameid": "g1", "side": "Blue", "date": pd.Timestamp("2026-07-26 12:00"), "patch": 16.14}]
            ),
            pd.DataFrame(
                [{"gameid": "g2", "side": "Blue", "date": "2026-07-26T13:00:00+00:00", "patch": "16.14.794.9266"}]
            ),
            ["gameid", "side"],
        )
        self.assertTrue(pd.api.types.is_datetime64_dtype(merged["date"]))
        self.assertEqual(merged["patch"].tolist(), ["16.14", "16.14.794.9266"])

    def test_grid_ingest_reuses_verified_cache_when_raw_files_are_missing(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            old_raw = grid_ingest.RAW_GRID_DIR
            old_parquet = grid_ingest.PARQUET_DIR
            try:
                grid_ingest.RAW_GRID_DIR = root / "raw_grid"
                grid_ingest.PARQUET_DIR = root / "parquet"
                grid_ingest.PARQUET_DIR.mkdir(parents=True)
                pd.DataFrame([{"gameid": "g1", "side": "Blue"}]).to_parquet(
                    grid_ingest.PARQUET_DIR / "grid_team_games.parquet", index=False
                )
                pd.DataFrame([{"gameid": "g1", "side": "Blue", "position": "top"}]).to_parquet(
                    grid_ingest.PARQUET_DIR / "grid_player_games.parquet", index=False
                )
                teams, players = grid_ingest.ingest_grid(required=True)
            finally:
                grid_ingest.RAW_GRID_DIR = old_raw
                grid_ingest.PARQUET_DIR = old_parquet
        self.assertEqual(len(teams), 1)
        self.assertEqual(len(players), 1)

    def test_grid_file_download_stops_after_bounded_429_retry(self) -> None:
        error = urllib.error.HTTPError(
            "https://example.test/events.jsonl",
            429,
            "too many requests",
            {"Retry-After": "120"},
            io.BytesIO(b"too many requests"),
        )
        with TemporaryDirectory() as temp, patch(
            "lol_kills.etl.grid_ingest.urllib.request.urlopen", side_effect=error
        ) as urlopen, patch("lol_kills.etl.grid_ingest.time.sleep") as sleep:
            self.assertFalse(_download("https://example.test/events.jsonl", "key", Path(temp) / "events.jsonl"))
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(30)

    def test_url_uses_documented_key_query_without_logging_value(self) -> None:
        url = series_events_url("2970137", "secret", use_config=True, from_sequence_number=9)
        self.assertIn("/live-data-feed/series/2970137", url)
        self.assertIn("key=secret", url)
        self.assertIn("useConfig=true", url)
        self.assertIn("fromSequenceNumber=9", url)
        self.assertEqual(default_config()["rules"][0]["includeFullState"], True)

    def test_transaction_helpers_prefer_full_series_state(self) -> None:
        transaction = {
            "sequenceNumber": "12",
            "events": [{"seriesState": {"id": "2970137", "games": []}}],
        }
        self.assertEqual(transaction_sequence(transaction), 12)
        self.assertEqual(transaction_state(transaction), {"id": "2970137", "games": []})

    def test_live_model_is_explicit_when_draft_is_incomplete(self) -> None:
        state = {
            "games": [
                {
                    "started": True,
                    "finished": False,
                    "clock": {"currentSeconds": 600},
                    "teams": [
                        {
                            "id": "blue-id",
                            "name": "Blue Org",
                            "side": "blue",
                            "netWorth": 35000,
                            "kills": 4,
                            "players": [],
                        },
                        {
                            "id": "red-id",
                            "name": "Red Org",
                            "side": "red",
                            "netWorth": 33000,
                            "kills": 2,
                            "players": [],
                        },
                    ],
                    "draftActions": [],
                }
            ]
        }
        result = evaluate_live_state(state, elo_diff=40)
        self.assertEqual(result.status, "preliminary")
        self.assertEqual(result.draft_status, "incomplete")
        self.assertIsNotNone(result.p_blue)
        self.assertIn("Draft is incomplete", " ".join(result.warnings))

    def test_live_model_fails_closed_without_a_game(self) -> None:
        result = evaluate_live_state({"games": []}, elo_diff=40)
        self.assertEqual(result.status, "unavailable")
        self.assertIsNone(result.p_blue)
        self.assertIn("active_game", result.missing)

    def test_live_model_withholds_late_probability_until_calibrated(self) -> None:
        state = {
            "games": [
                {
                    "started": True,
                    "finished": False,
                    "clock": {"currentSeconds": 1800},
                    "teams": [
                        {"id": "blue-id", "name": "Blue Org", "side": "blue", "netWorth": 50000},
                        {"id": "red-id", "name": "Red Org", "side": "red", "netWorth": 42000},
                    ],
                }
            ]
        }
        result = evaluate_live_state(state, elo_diff=40)
        self.assertEqual(result.status, "preliminary-out-of-calibration")
        self.assertIsNone(result.p_blue)
        self.assertIn("calibration_window", result.missing)

    def test_live_snapshot_writes_versioned_public_pointer_without_raw_transactions(self) -> None:
        state = {
            "games": [
                {
                    "id": "game-1",
                    "started": True,
                    "finished": False,
                    "clock": {"currentSeconds": 600},
                    "teams": [
                        {
                            "id": "blue-id",
                            "name": "Blue Org",
                            "side": "blue",
                            "netWorth": 35000,
                            "kills": 4,
                            "players": [{"name": "Blue Mid", "role": "mid", "champion": "Ahri"}],
                        },
                        {
                            "id": "red-id",
                            "name": "Red Org",
                            "side": "red",
                            "netWorth": 33000,
                            "kills": 2,
                            "players": [{"name": "Red Mid", "role": "mid", "champion": "Orianna"}],
                        },
                    ],
                }
            ]
        }
        snapshot = build_live_snapshot("2970137", state, sequence_number=12, elo_diff=40)
        self.assertEqual(snapshot["schema_version"], "live.v1")
        self.assertNotIn("transactions", snapshot)
        self.assertEqual(snapshot["teams"]["blue"]["players"][0]["champion"], "Ahri")
        self.assertGreater(len(snapshot["evaluation"]["contributions"]), 0)
        with TemporaryDirectory() as temp:
            publisher = LivePublisher(local_root=Path(temp))
            pointer = publisher.publish_snapshot(snapshot)
            publisher.publish_index([pointer])
            self.assertEqual(pointer["snapshot_url"], "/live/series/2970137/snapshots/12.json")
            self.assertTrue((Path(temp) / "series/2970137/snapshots/12.json").is_file())
            self.assertTrue((Path(temp) / "index.json").is_file())


if __name__ == "__main__":
    unittest.main()
