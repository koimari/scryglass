from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.etl.competition import (
    canonicalize_competition_frame,
    classify_competition,
    is_team_affiliation_league,
)
from lol_kills.export.pack_records import (
    build_maps_frame_from_team_games,
    build_player_records,
    build_team_records,
    filter_public_team_rating_maps,
    summarize_player_affiliations,
)
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.dual_elo import _is_intl
from lol_kills.ratings.hierarchical_bt import fit_hierarchical_bt
from lol_kills.ratings.player_elo import build_maps_frame_from_players, build_player_weekly_ranks
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

    def test_challenger_circuit_is_tier_two(self) -> None:
        label = classify_competition("CD", "CD 2026 Split 1")
        self.assertEqual(label.scope, "tier2")
        self.assertEqual(label.tier, "tier2")
        self.assertFalse(label.is_international)

    def test_domestic_cups_are_event_evidence_not_team_affiliations(self) -> None:
        self.assertTrue(is_team_affiliation_league("LCK"))
        self.assertTrue(is_team_affiliation_league("LCKC"))
        self.assertFalse(is_team_affiliation_league("KESPA"))
        self.assertFalse(is_team_affiliation_league("KESPA CUP"))
        self.assertFalse(is_team_affiliation_league("DCUP"))
        self.assertFalse(is_team_affiliation_league("EWC"))

    def test_national_leagues_do_not_enter_tier_one(self) -> None:
        self.assertEqual(classify_competition("TCL").tier, "tier2")
        self.assertEqual(classify_competition("LJL").tier, "tier2")
        self.assertEqual(classify_competition("EM").tier, "international")

    def test_worlds_abbreviation_is_not_a_domestic_tier(self) -> None:
        label = classify_competition("WLDs", None)
        self.assertEqual(label.league, "WORLDS")
        self.assertEqual(label.tier, "international")
        self.assertTrue(label.is_international)

    def test_road_to_msi_does_not_become_international(self) -> None:
        label = classify_competition("LCK", "LCK 2026 Road to MSI")
        self.assertEqual(label.league, "LCK")
        self.assertEqual(label.scope, "regional")
        self.assertFalse(label.is_international)
        self.assertFalse(_is_intl("LCK", "LCK 2026 Road to MSI"))
        self.assertTrue(_is_intl("MSI", None))

    def test_transport_label_falls_back_to_the_real_league(self) -> None:
        row = canonicalize_competition_frame(
            pd.DataFrame(
                [
                    {
                        "league": "LCK",
                        "league_source": "ORACLE_ELIXIR_API",
                        "blue_team": "Gen.G",
                        "red_team": "T1",
                    }
                ]
            )
        ).iloc[0]

        self.assertEqual(row["league"], "LCK")
        self.assertEqual(row["league_source"], "LCK")
        self.assertEqual(row["competition_tier"], "tier1")

    def test_unresolved_transport_label_is_not_tier_three(self) -> None:
        label = classify_competition("ORACLE_ELIXIR_API")
        self.assertEqual(label.league, "UNKNOWN")
        self.assertEqual(label.scope, "other")
        self.assertEqual(label.tier, "other")

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

    def test_gen_g_cup_appearance_does_not_rewrite_team_or_roster_tier(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-07-19",
                    "league": "LCK",
                    "blue_team": "Gen.G",
                    "red_team": "T1",
                    "y_blue_win": 1,
                },
                {
                    "date": "2026-07-28",
                    "league": "KESPA CUP",
                    "blue_team": "Gen.G",
                    "red_team": "T1",
                    "y_blue_win": 1,
                },
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "date": "2026-07-19",
                    "league": "LCK",
                    "playername": "Chovy",
                    "position": "mid",
                    "teamname": "Gen.G",
                    "result": 1,
                },
                {
                    "date": "2026-07-28",
                    "league": "KESPA CUP",
                    "playername": "Chovy",
                    "position": "mid",
                    "teamname": "Gen.G",
                    "result": 1,
                },
            ]
        )

        team_records = build_team_records(maps)
        player_records = build_player_records(players, team_records=team_records)
        team = team_records["Gen.G"]
        player = player_records["Chovy"]

        self.assertEqual(team["current_league"], "LCK")
        self.assertEqual(team["current_tier"], "tier1")
        self.assertEqual(team["last_event_league"], "KESPA CUP")
        self.assertEqual(team["last_event_tier"], "tier3")
        self.assertEqual(player["current_team"], "Gen.G")
        self.assertEqual(player["current_league"], "LCK")
        self.assertEqual(player["current_tier"], "tier1")
        self.assertEqual(player["last_event_league"], "KESPA CUP")
        self.assertTrue(player["affiliation_repaired"])
        self.assertEqual(
            summarize_player_affiliations(player_records, team_records),
            {
                "players": 1,
                "current_team_inherited": 1,
                "repaired_from_team_roster": 1,
                "unresolved_current_teams": 0,
                "remaining_team_player_conflicts": 0,
            },
        )

    def test_player_transfer_inherits_new_team_affiliation(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2025-06-01",
                    "league": "CBLOL",
                    "blue_team": "FURIA",
                    "red_team": "LOUD",
                    "y_blue_win": 1,
                },
                {
                    "date": "2026-06-01",
                    "league": "CD",
                    "blue_team": "KaBuM! Ilha das Lendas",
                    "red_team": "Other",
                    "y_blue_win": 1,
                },
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "date": "2025-06-01",
                    "league": "CBLOL",
                    "playername": "Guigs",
                    "position": "sup",
                    "teamname": "FURIA",
                    "result": 1,
                },
                {
                    "date": "2026-06-01",
                    "league": "CD",
                    "playername": "Guigs",
                    "position": "sup",
                    "teamname": "KaBuM! Ilha das Lendas",
                    "result": 1,
                },
            ]
        )

        records = build_player_records(players, team_records=build_team_records(maps))

        self.assertEqual(records["Guigs"]["current_team"], "KaBuM! Ilha das Lendas")
        self.assertEqual(records["Guigs"]["current_league"], "CD")
        self.assertEqual(records["Guigs"]["current_tier"], "tier2")

    def test_international_transfer_does_not_keep_old_team_league(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "league": "LEC",
                    "blue_team": "Old Team",
                    "red_team": "Other",
                    "y_blue_win": 1,
                },
                {
                    "date": "2026-03-01",
                    "league": "EM",
                    "blue_team": "Witchcraft",
                    "red_team": "Other Two",
                    "y_blue_win": 1,
                },
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "league": "LEC",
                    "playername": "Mover",
                    "position": "top",
                    "teamname": "Old Team",
                    "result": 1,
                },
                {
                    "date": "2026-03-01",
                    "league": "EM",
                    "playername": "Mover",
                    "position": "top",
                    "teamname": "Witchcraft",
                    "result": 1,
                },
            ]
        )

        records = build_player_records(players, team_records=build_team_records(maps))

        self.assertEqual(records["Mover"]["current_team"], "Witchcraft")
        self.assertIsNone(records["Mover"]["current_league"])
        self.assertIsNone(records["Mover"]["current_tier"])

    def test_academy_team_keeps_its_own_affiliation(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-07-28",
                    "league": "LCK",
                    "blue_team": "Gen.G",
                    "red_team": "T1",
                    "y_blue_win": 1,
                },
                {
                    "date": "2026-07-28",
                    "league": "LCKC",
                    "blue_team": "Gen.G Global Academy",
                    "red_team": "T1 Esports Academy",
                    "y_blue_win": 1,
                },
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "date": "2026-07-28",
                    "league": "LCKC",
                    "playername": "Courage",
                    "position": "jng",
                    "teamname": "Gen.G Global Academy",
                    "result": 1,
                }
            ]
        )

        records = build_player_records(players, team_records=build_team_records(maps))

        self.assertEqual(records["Courage"]["current_team"], "Gen.G Global Academy")
        self.assertEqual(records["Courage"]["current_league"], "LCKC")
        self.assertEqual(records["Courage"]["current_tier"], "tier2")

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

    def test_player_current_affiliation_does_not_promote_tier_two_to_cblol(self) -> None:
        players = pd.DataFrame(
            [
                {"date": "2025-06-01", "league": "CBLOL", "playername": "Guigs", "position": "sup", "teamname": "FURIA", "result": 1},
                {"date": "2026-06-01", "league": "CD", "playername": "Guigs", "position": "sup", "teamname": "KaBuM! Ilha das Lendas", "result": 1},
            ]
        )
        records = build_player_records(players)
        self.assertEqual(records["Guigs"]["primary"], "CD")
        self.assertEqual(records["Guigs"]["current_league"], "CD")
        self.assertEqual(records["Guigs"]["current_tier"], "tier2")
        self.assertEqual(records["Guigs"]["current_team"], "KaBuM! Ilha das Lendas")

    def test_player_records_use_latest_observed_team_and_precompute_side_and_role_stats(self) -> None:
        players = pd.DataFrame(
            [
                {"date": "2026-01-01", "league": "LEC", "playername": "Mover", "position": "top", "teamname": "Los Ratones", "side": "Blue", "result": 1},
                {"date": "2026-02-01", "league": "LEC", "playername": "Mover", "position": "top", "teamname": "Los Ratones", "side": "Red", "result": 0},
                {"date": "2026-03-01", "league": "EM", "playername": "Mover", "position": "mid", "teamname": "Witchcraft", "side": "Blue", "result": 1},
            ]
        )
        record = build_player_records(players)["Mover"]
        self.assertEqual(record["current_league"], "LEC")
        self.assertEqual(record["current_team"], "Witchcraft")
        self.assertEqual(record["blue_games"], 2)
        self.assertEqual(record["blue_wins"], 2)
        self.assertEqual(record["blue_wr"], 1.0)
        self.assertEqual(record["red_games"], 1)
        self.assertEqual(record["red_wins"], 0)
        self.assertEqual(record["red_wr"], 0.0)
        self.assertEqual(record["roles"], ["top", "mid"])
        self.assertEqual(record["primary_role"], "top")

    def test_player_records_drop_transport_label_from_league_affiliation(self) -> None:
        players = pd.DataFrame(
            [
                {"date": "2026-07-20", "league": "LCKC", "playername": "Jiwoo", "position": "bot", "teamname": "Kiwoom DRX", "side": "Blue", "result": 1},
                {"date": "2026-07-29", "league": "ORACLE_ELIXIR_API", "playername": "Jiwoo", "position": "bot", "teamname": "Kiwoom DRX Challengers", "side": "Red", "result": 1},
            ]
        )
        record = build_player_records(players)["Jiwoo"]
        self.assertEqual(record["leagues"], ["LCKC"])
        self.assertEqual(record["primary"], "LCKC")
        self.assertEqual(record["current_team"], "Kiwoom DRX Challengers")

    def test_player_records_hide_excluded_team_affiliation(self) -> None:
        players = pd.DataFrame(
            [
                {
                    "date": "2026-08-01",
                    "league": "LEC",
                    "playername": "Baus",
                    "position": "top",
                    "teamname": "Los Ratones",
                    "side": "Blue",
                    "result": 1,
                }
            ]
        )

        record = build_player_records(players)["Baus"]

        self.assertIsNone(record["current_team"])

    def test_full_team_feed_adapter_keeps_developmental_games(self) -> None:
        team_games = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01", "league": "TCL", "side": "Blue", "position": "team", "teamname": "Misa Esports", "result": 1},
                {"gameid": "g1", "date": "2026-01-01", "league": "TCL", "side": "Red", "position": "team", "teamname": "Other", "result": 0},
                {"gameid": "g2", "date": "2026-01-02", "league": "CD", "side": "Blue", "position": "team", "teamname": "KaBuM! Ilha das Lendas", "result": 1},
                {"gameid": "g2", "date": "2026-01-02", "league": "CD", "side": "Red", "position": "team", "teamname": "Other 2", "result": 0},
            ]
        )
        maps = build_maps_frame_from_team_games(team_games)
        self.assertEqual(len(maps), 2)
        self.assertEqual(set(maps["competition_tier"]), {"tier2"})

    def test_team_records_publish_tier_aggregates(self) -> None:
        records = build_team_records(
            pd.DataFrame(
                [
                    {"date": "2026-01-01", "league": "LCK", "blue_team": "A", "red_team": "B", "y_blue_win": 1},
                    {"date": "2026-01-02", "league": "TCL", "blue_team": "A", "red_team": "C", "y_blue_win": 0},
                ]
            )
        )
        self.assertEqual(records["A"]["current_tier"], "tier2")
        self.assertEqual(records["A"]["by_tier"]["tier1"]["games"], 1)
        self.assertEqual(records["A"]["by_tier"]["tier2"]["games"], 1)

    def test_team_record_keeps_tier_one_when_cached_source_label_is_transport(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-08-01",
                    "league": "LCK",
                    "league_source": "ORACLE_ELIXIR_API",
                    "blue_team": "Gen.G",
                    "red_team": "T1",
                    "y_blue_win": 1,
                }
            ]
        )

        records = build_team_records(maps)

        self.assertEqual(records["Gen.G"]["current_league"], "LCK")
        self.assertEqual(records["Gen.G"]["current_tier"], "tier1")
        self.assertNotIn("ORACLE_ELIXIR_API", records["Gen.G"]["source_leagues"])

    def test_weekly_player_rank_payload_uses_sunday_baseline(self) -> None:
        players = []
        roles = ["top", "jng", "mid", "bot", "sup"]
        for game, date, winner in (
            ("g1", "2026-07-10", "A"),
            ("g2", "2026-07-20", "B"),
        ):
            for side, team, result in (("Blue", "A", int(winner == "A")), ("Red", "B", int(winner == "B"))):
                for role_index, role in enumerate(roles):
                    players.append(
                        {
                            "gameid": game,
                            "date": date,
                            "league": "LCS",
                            "side": side,
                            "position": role,
                            "playername": f"{team}{role_index}",
                            "teamname": team,
                            "result": result,
                        }
                    )
        frame = pd.DataFrame(players)
        payload = build_player_weekly_ranks(
            build_maps_frame_from_players(frame),
            frame,
            as_of=pd.Timestamp("2026-07-26T12:00:00Z"),
            min_games=1,
        )
        self.assertEqual(payload["as_of"], "2026-07-26T00:00:00Z")
        self.assertEqual(payload["previous_as_of"], "2026-07-19T00:00:00Z")
        self.assertEqual(payload["current_through"], "2026-07-26T12:00:00Z")
        self.assertIn("A0", payload["by_player"])
        self.assertIn("tier1", payload["by_player"]["A0"])

    def test_player_game_uid_falls_back_per_row_to_gameid(self) -> None:
        rows = []
        roles = ["top", "jng", "mid", "bot", "sup"]
        for side, team, result in (("Blue", "A", 1), ("Red", "B", 0)):
            for role in roles:
                rows.append(
                    {
                        "game_uid": None,
                        "gameid": "g1",
                        "date": "2026-01-01",
                        "league": "LCS",
                        "side": side,
                        "position": role,
                        "playername": f"{team}-{role}",
                        "teamname": team,
                        "result": result,
                    }
                )

        maps = build_maps_frame_from_players(pd.DataFrame(rows))

        self.assertEqual(len(maps), 1)
        self.assertEqual(maps.iloc[0]["game_uid"], "g1")

    def test_source_game_keys_remove_repeated_transport_prefixes(self) -> None:
        self.assertEqual(canonical_source_game_key("oe-api:oracle-elixir-api:game-1"), "game-1")
        self.assertEqual(canonical_source_game_key(None, "oracle-elixir-api:game-2"), "game-2")

    def test_public_team_rating_filter_excludes_los_ratones(self) -> None:
        maps = pd.DataFrame(
            [
                {"blue_team": "Los Ratones", "red_team": "Other", "y_blue_win": 1},
                {"blue_team": "Other", "red_team": "Another", "y_blue_win": 0},
            ]
        )
        filtered = filter_public_team_rating_maps(maps)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]["blue_team"], "Other")


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
