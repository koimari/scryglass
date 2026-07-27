from __future__ import annotations

import io
import json
import unittest
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import lol_kills.etl.grid_ingest as grid_ingest
from lol_kills.etl.grid_ingest import _download
from lol_kills.etl.join import _canonical_player_game_key, _oe_wide
from lol_kills.etl.grid_series_events import (
    default_config,
    series_events_url,
    transaction_sequence,
    transaction_state,
)
from lol_kills.etl.riot_esports_events import (
    analyze_voidgrub_window,
    extract_epic_kills,
    extract_positions,
)
from lol_kills.live_model import _role_assigned_picks, evaluate_live_state
from lol_kills.live_snapshots import LivePublisher, build_live_snapshot


class GridSeriesEventsTests(unittest.TestCase):
    def test_live_draft_uses_explicit_roles_not_pick_or_player_order(self) -> None:
        blue_players = [
            {"role": "support", "champion": "Nautilus"},
            {"role": "mid", "champion": "Ahri"},
            {"role": "top", "champion": "Gnar"},
            {"role": "bottom", "champion": "Jinx"},
            {"role": "jungle", "champion": "Vi"},
        ]
        red_players = [
            {"role": "top", "champion": "Kennen"},
            {"role": "jungle", "champion": "Sejuani"},
            {"role": "mid", "champion": "Orianna"},
            {"role": "bot", "champion": "Aphelios"},
            {"role": "sup", "champion": "Thresh"},
        ]
        blue, red = _role_assigned_picks(
            {
                "blue": {"players": blue_players},
                "red": {"players": red_players},
            }
        )
        self.assertEqual(blue, ["Gnar", "Vi", "Ahri", "Jinx", "Nautilus"])
        self.assertEqual(
            red,
            ["Kennen", "Sejuani", "Orianna", "Aphelios", "Thresh"],
        )
        blue_reordered, _ = _role_assigned_picks(
            {
                "blue": {"players": list(reversed(blue_players))},
                "red": {"players": red_players},
            }
        )
        self.assertEqual(blue_reordered, blue)

    def test_live_stats_preserve_unknown_combat_values_and_real_zeroes(self) -> None:
        rows = extract_positions(
            [
                {
                    "rfc461Schema": "stats_update",
                    "gameTime": 1_000,
                    "participants": [
                        {
                            "participantID": 1,
                            "teamID": 100,
                            "position": {"x": 10, "z": 20},
                            "totalGold": 0,
                            "currentGold": None,
                            "level": None,
                            "health": None,
                            "healthMax": None,
                        }
                    ],
                }
            ]
        )
        self.assertEqual(rows[0]["gameTime_ms"], 1_000)
        self.assertEqual(rows[0]["totalGold"], 0.0)
        self.assertIsNone(rows[0]["currentGold"])
        self.assertIsNone(rows[0]["level"])
        self.assertIsNone(rows[0]["health"])
        self.assertIsNone(rows[0]["healthMax"])
        self.assertIsNone(rows[0]["alive"])

    def test_missing_epic_position_never_becomes_map_origin(self) -> None:
        kills = extract_epic_kills(
            [
                {
                    "rfc461Schema": "epic_monster_kill",
                    "gameTime": 480_000,
                    "monsterType": "VoidGrub",
                    "killerTeamID": 100,
                }
            ]
        )
        self.assertIsNone(kills[0]["x"])
        result = analyze_voidgrub_window([], kills, [], [])
        self.assertEqual(result["status"], "unavailable_missing_pit_position")

    def test_grid_tournament_classification_keeps_developmental_and_unknown_scopes_safe(self) -> None:
        self.assertEqual(grid_ingest._league_for("NACL Summer 2026"), "NACL")
        self.assertEqual(grid_ingest._league_for("LCK Challengers 2026"), "LCKC")
        self.assertEqual(grid_ingest._league_for("LPL Split 3 2026"), "LPL")
        self.assertEqual(grid_ingest._league_for("Circuito Desafiante - Split 2 2026"), "CD")
        self.assertEqual(grid_ingest._league_for("Nonsense Invitational"), "UNKNOWN")

    def test_player_key_normalization_deduplicates_grid_game_uid(self) -> None:
        players = _canonical_player_game_key(
            pd.DataFrame(
                [{"gameid": "oe-id", "game_uid": "grid-id", "playername": "Player"}]
            )
        )
        self.assertEqual(list(players.columns), ["game_uid", "playername"])
        self.assertEqual(players.iloc[0]["game_uid"], "grid-id")

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

    def test_grid_completion_provenance_survives_map_widening(self) -> None:
        maps = _oe_wide(
            pd.DataFrame(
                [
                    {
                        "gameid": "g1",
                        "side": "Blue",
                        "teamname": "Blue Org",
                        "source": "grid",
                        "grid_completion_source": "end_state_summary",
                    },
                    {
                        "gameid": "g1",
                        "side": "Red",
                        "teamname": "Red Org",
                        "source": "grid",
                        "grid_completion_source": "end_state_summary",
                    },
                ]
            )
        )
        self.assertEqual(maps.iloc[0]["grid_completion_source"], "end_state_summary")
        self.assertTrue(bool(maps.iloc[0]["source_grid"]))

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
                teams, players = grid_ingest.ingest_grid(required=False)
                with self.assertRaisesRegex(
                    grid_ingest.GridIngestError,
                    "current run produced no completed game",
                ):
                    grid_ingest.ingest_grid(required=True)
            finally:
                grid_ingest.RAW_GRID_DIR = old_raw
                grid_ingest.PARQUET_DIR = old_parquet
        self.assertEqual(len(teams), 1)
        self.assertEqual(len(players), 1)

    def test_completed_grid_events_use_public_teamkill_columns(self) -> None:
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        participants = [
            {
                "teamID": team_id,
                "riotId": {"displayName": f"{prefix} Player{index}"},
                "role": role,
                "championName": f"Champion{index}",
            }
            for team_id, prefix in ((100, "Blue"), (200, "Red"))
            for index, role in enumerate(roles, start=1)
        ]
        events = [
            {
                "rfc461Schema": "game_info",
                "rfc460Timestamp": "2026-07-26T12:00:00Z",
                "gameID": 42,
                "gameName": "game-42",
                "gameVersion": "16.14.794.5912",
                "platformID": "LOLTMNT01",
                "participants": participants,
            },
            *[
                {"rfc461Schema": "champion_kill", "killerTeamID": 100}
                for _ in range(4)
            ],
            *[
                {"rfc461Schema": "champion_kill", "killerTeamID": 200}
                for _ in range(2)
            ],
            {
                "rfc461Schema": "game_end",
                "winningTeam": 100,
                "gameTime": 1_800_000,
            },
        ]
        with TemporaryDirectory() as temp:
            path = Path(temp) / "events.jsonl"
            path.write_text("\n".join(json.dumps(event) for event in events))
            parsed = grid_ingest._parse_events(
                path,
                series={
                    "id": "series-42",
                    "tournament": "LPL 2026",
                    "teams": ["Blue Org", "Red Org"],
                },
                game_index=1,
            )

        self.assertIsNotNone(parsed)
        team_rows = parsed[0]["team_rows"]
        blue = next(row for row in team_rows if row["side"] == "Blue")
        red = next(row for row in team_rows if row["side"] == "Red")
        self.assertEqual((blue["teamkills"], blue["teamdeaths"]), (4, 2))
        self.assertEqual((red["teamkills"], red["teamdeaths"]), (2, 4))
        self.assertEqual(blue["grid_completion_source"], "events_game_end")

    def test_complete_matching_summary_recovers_missing_game_end(self) -> None:
        roles = ["TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY"]
        participants = [
            {
                "teamID": team_id,
                "riotId": {"displayName": f"{prefix} Player{index}"},
                "role": role,
                "championName": f"Champion{index}",
            }
            for team_id, prefix in ((100, "Blue"), (200, "Red"))
            for index, role in enumerate(roles, start=1)
        ]
        events = [
            {
                "rfc461Schema": "game_info",
                "rfc460Timestamp": "2026-07-26T12:00:00Z",
                "gameID": 42,
                "gameName": "game-42",
                "gameVersion": "16.14.794.5912",
                "platformID": "LOLTMNT01",
                "participants": participants,
            },
            *[{"rfc461Schema": "champion_kill", "killerTeamID": 100} for _ in range(2)],
            *[{"rfc461Schema": "champion_kill", "killerTeamID": 200} for _ in range(4)],
        ]
        summary = {
            "endOfGameResult": "GameComplete",
            "gameId": 42,
            "gameDuration": 1_800,
            "teams": [
                {
                    "teamId": 100,
                    "win": False,
                    "objectives": {"champion": {"kills": 2}},
                },
                {
                    "teamId": 200,
                    "win": True,
                    "objectives": {"champion": {"kills": 4}},
                },
            ],
        }
        with TemporaryDirectory() as temp:
            root = Path(temp)
            event_path = root / "events.jsonl"
            summary_path = root / "summary.json"
            event_path.write_text("\n".join(json.dumps(event) for event in events))
            summary_path.write_text(json.dumps(summary))
            parsed = grid_ingest._parse_events(
                event_path,
                series={
                    "id": "series-42",
                    "tournament": "LPL 2026",
                    "teams": ["Blue Org", "Red Org"],
                },
                game_index=3,
                summary_path=summary_path,
            )

        self.assertIsNotNone(parsed)
        team_rows = parsed[0]["team_rows"]
        self.assertEqual({row["grid_completion_source"] for row in team_rows}, {"end_state_summary"})
        self.assertEqual(next(row for row in team_rows if row["side"] == "Red")["result"], 1)
        self.assertEqual(next(row for row in team_rows if row["side"] == "Blue")["teamdeaths"], 4)

    def test_summary_fallback_rejects_mismatched_or_incomplete_evidence(self) -> None:
        game_info = {"gameID": 42}
        kills = {100: 2, 200: 4}
        base = {
            "endOfGameResult": "GameComplete",
            "gameId": 99,
            "gameDuration": 1_800,
            "teams": [
                {"teamId": 100, "win": False, "objectives": {"champion": {"kills": 2}}},
                {"teamId": 200, "win": True, "objectives": {"champion": {"kills": 4}}},
            ],
        }
        with TemporaryDirectory() as temp:
            summary_path = Path(temp) / "summary.json"
            summary_path.write_text(json.dumps(base))
            self.assertIsNone(
                grid_ingest._verified_summary_game_end(
                    summary_path,
                    game_info=game_info,
                    kills=kills,
                )
            )
            base["gameId"] = 42
            base["teams"][1]["objectives"]["champion"]["kills"] = 3
            summary_path.write_text(json.dumps(base))
            self.assertIsNone(
                grid_ingest._verified_summary_game_end(
                    summary_path,
                    game_info=game_info,
                    kills=kills,
                )
            )

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
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 90)
        sleep.assert_called_once_with(30)

    def test_grid_tournament_filter_expands_discovery_before_download(self) -> None:
        series = [
            {"id": "lpl-series", "tournament": "LPL - Split 3 2026", "teams": ["A", "B"]},
            {"id": "other-series", "tournament": "LEC - Summer 2026", "teams": ["C", "D"]},
        ]
        with TemporaryDirectory() as temp:
            old_raw = grid_ingest.RAW_GRID_DIR
            try:
                grid_ingest.RAW_GRID_DIR = Path(temp) / "raw_grid"
                with patch("lol_kills.etl.grid_ingest._api_key", return_value="key"), patch(
                    "lol_kills.etl.grid_ingest._series_rows", return_value=series
                ) as series_rows, patch(
                    "lol_kills.etl.grid_ingest._file_list", return_value=[]
                ) as file_list:
                    result = grid_ingest._download_recent(
                        days=5,
                        limit=10,
                        tournament="lpl",
                        env_file=None,
                    )
            finally:
                grid_ingest.RAW_GRID_DIR = old_raw
        self.assertEqual(series_rows.call_args.args[3], 100)
        self.assertEqual(file_list.call_args.args[1], "lpl-series")
        self.assertEqual(result["series_seen"], 1)
        self.assertEqual(result["tournament_filter"], "lpl")

    def test_grid_download_includes_verified_end_state_summaries(self) -> None:
        series = [{"id": "series-1", "tournament": "LEC - Summer 2026", "teams": ["A", "B"]}]
        files = [
            {
                "id": "events-riot-game-1",
                "status": "ready",
                "fullURL": "https://example.test/events",
                "fileName": "events_1_1_riot.jsonl",
            },
            {
                "id": "state-summary-riot-game-1",
                "status": "ready",
                "fullURL": "https://example.test/summary",
                "fileName": "end_state_summary_riot_1_1.json",
            },
            {
                "id": "replay-riot-game-1",
                "status": "ready",
                "fullURL": "https://example.test/replay",
                "fileName": "replay.rofl",
            },
        ]
        with TemporaryDirectory() as temp:
            old_raw = grid_ingest.RAW_GRID_DIR
            try:
                grid_ingest.RAW_GRID_DIR = Path(temp) / "raw_grid"
                with patch("lol_kills.etl.grid_ingest._api_key", return_value="key"), patch(
                    "lol_kills.etl.grid_ingest._series_rows", return_value=series
                ), patch("lol_kills.etl.grid_ingest._file_list", return_value=files), patch(
                    "lol_kills.etl.grid_ingest._download", return_value=True
                ) as download:
                    result = grid_ingest._download_recent(
                        days=3,
                        limit=40,
                        tournament=None,
                        env_file=None,
                    )
            finally:
                grid_ingest.RAW_GRID_DIR = old_raw
        self.assertEqual(download.call_count, 2)
        self.assertEqual(result["files_downloaded"], 2)

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
        self.assertEqual(result.status, "unavailable-unvalidated-model")
        self.assertEqual(result.draft_status, "unavailable")
        self.assertIsNone(result.p_blue)
        self.assertIn("probability is withheld", " ".join(result.warnings))
        self.assertIn("validated_live_model", result.missing)

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
        self.assertEqual(result.status, "unavailable-unvalidated-model")
        self.assertIsNone(result.p_blue)
        self.assertIn("validated_live_model", result.missing)

    def test_live_game_time_milliseconds_are_not_inferred_from_magnitude(self) -> None:
        short = {
            "games": [
                {
                    "gameTime": 1_000,
                    "teams": [
                        {"id": "blue-id", "name": "Blue", "side": "blue"},
                        {"id": "red-id", "name": "Red", "side": "red"},
                    ],
                }
            ]
        }
        result = evaluate_live_state(short, elo_diff=0)
        self.assertAlmostEqual(result.minute or -1, 1 / 60)

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
        self.assertEqual(
            snapshot["evaluation"]["status"],
            "unavailable-unvalidated-model",
        )
        self.assertEqual(snapshot["evaluation"]["contributions"], [])
        with TemporaryDirectory() as temp:
            publisher = LivePublisher(local_root=Path(temp))
            pointer = publisher.publish_snapshot(snapshot)
            publisher.publish_index([pointer])
            self.assertEqual(pointer["snapshot_url"], "/live/series/2970137/snapshots/12.json")
            self.assertTrue((Path(temp) / "series/2970137/snapshots/12.json").is_file())
            self.assertTrue((Path(temp) / "index.json").is_file())

    def test_series_score_follows_team_identity_across_side_swaps(self) -> None:
        state = {
            "games": [
                {
                    "id": "g1",
                    "finished": True,
                    "teams": [
                        {
                            "id": "team-a",
                            "name": "Team A",
                            "side": "blue",
                            "won": True,
                        },
                        {
                            "id": "team-b",
                            "name": "Team B",
                            "side": "red",
                            "won": False,
                        },
                    ],
                },
                {
                    "id": "g2",
                    "finished": True,
                    "teams": [
                        {
                            "id": "team-b",
                            "name": "Team B",
                            "side": "blue",
                            "won": False,
                        },
                        {
                            "id": "team-a",
                            "name": "Team A",
                            "side": "red",
                            "won": True,
                        },
                    ],
                },
            ]
        }
        snapshot = build_live_snapshot("series", state, sequence_number=2)
        self.assertEqual(snapshot["teams"]["red"]["name"], "Team A")
        self.assertEqual(snapshot["teams"]["red"]["score"], 2)
        self.assertEqual(snapshot["teams"]["blue"]["name"], "Team B")
        self.assertEqual(snapshot["teams"]["blue"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
