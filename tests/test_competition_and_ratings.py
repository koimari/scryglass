from __future__ import annotations

import unittest

import pandas as pd

from lol_kills.etl.competition import canonicalize_competition_frame, classify_competition
from lol_kills.export.pack_records import (
    build_current_tournament_membership,
    build_maps_frame_from_team_games,
    build_player_records,
    build_team_records,
    tournament_family,
)
from lol_kills.ratings.dual_elo import _is_intl
from lol_kills.ratings.hierarchical_bt import fit_hierarchical_bt
from lol_kills.ratings.player_elo import build_maps_frame_from_players, build_player_weekly_ranks
from lol_kills.ratings.validation import audit_rating_inputs


def _current_lpl_registry() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "snapshot_id": "test-current-lpl",
        "authority": "Riot Games LoL Esports",
        "observed_at": "2026-07-20T00:00:00Z",
        "review_due_at": "2026-07-30T00:00:00Z",
        "tournaments": [
            {
                "tournament_id": "test-lpl-split-3",
                "league": "LPL",
                "name": "LPL - Split 3 2026",
                "status": "current",
                "source_url": "https://lolesports.com/test-lpl",
                "participants": [
                    {
                        "display_name": name,
                        "short_code": f"C{position}",
                    }
                    for position, name in enumerate(
                        ("Current A", "Current B", "Current C", "Current D"),
                        start=1,
                    )
                ],
                "stages": [
                    {
                        "stage_id": "groups",
                        "scheduled_best_of": 3,
                        "format_status": "verified",
                    }
                ],
            }
        ],
    }


class CompetitionIdentityTests(unittest.TestCase):
    def test_export_canonicalization_trims_identity_fields(self) -> None:
        row = canonicalize_competition_frame(
            pd.DataFrame(
                [
                    {
                        "league": "LCK",
                        "playername": " Player ",
                        "playerid": " id-1 ",
                        "teamid": " team-1 ",
                    }
                ]
            )
        ).iloc[0]
        self.assertEqual(row["playername"], "Player")
        self.assertEqual(row["playerid"], "id-1")
        self.assertEqual(row["teamid"], "team-1")

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

    def test_historical_team_display_is_preserved_beside_stable_identity(self) -> None:
        row = canonicalize_competition_frame(
            pd.DataFrame(
                [
                    {
                        "league": "LCK",
                        "blue_team": "Kwangdong Freecs",
                        "red_team": "DRX",
                    }
                ]
            )
        ).iloc[0]
        self.assertEqual(row["blue_team_source"], "Kwangdong Freecs")
        self.assertEqual(row["blue_team"], "DN SOOPers")
        self.assertEqual(row["blue_team_key"], "kwangdong-freecs")
        self.assertEqual(row["red_team_source"], "DRX")
        self.assertEqual(row["red_team_key"], "drx")

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

    def test_national_leagues_do_not_enter_tier_one(self) -> None:
        self.assertEqual(classify_competition("TCL").tier, "tier2")
        self.assertEqual(classify_competition("LJL").tier, "tier2")
        self.assertEqual(classify_competition("EM").tier, "international")

    def test_worlds_abbreviation_is_not_a_domestic_tier(self) -> None:
        label = classify_competition("WLDs", None)
        self.assertEqual(label.league, "WORLDS")
        self.assertEqual(label.tier, "international")
        self.assertTrue(label.is_international)

    def test_legacy_intl_fallback_is_unknown_not_an_international_league(self) -> None:
        label = classify_competition("INTL", "NACL - Summer 2026")
        self.assertEqual(label.league, "UNKNOWN")
        self.assertEqual(label.tier, "tier3")
        self.assertFalse(label.is_international)

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
        self.assertIsNone(records["Bot"]["current_team"])
        self.assertEqual(records["Bot"]["last_observed_league"], "AMERICAS")
        self.assertTrue(records["Bot"]["interregional"])

    def test_player_history_does_not_claim_an_unverified_current_affiliation(self) -> None:
        players = pd.DataFrame(
            [
                {"date": "2025-06-01", "league": "CBLOL", "playername": "Guigs", "position": "sup", "teamname": "FURIA", "result": 1},
                {"date": "2026-06-01", "league": "CD", "playername": "Guigs", "position": "sup", "teamname": "KaBuM! Ilha das Lendas", "result": 1},
            ]
        )
        records = build_player_records(players)
        self.assertEqual(records["Guigs"]["primary"], "CD")
        self.assertIsNone(records["Guigs"]["current_league"])
        self.assertIsNone(records["Guigs"]["current_tier"])
        self.assertIsNone(records["Guigs"]["current_team"])
        self.assertEqual(
            records["Guigs"]["last_observed_team"],
            "KaBuM! Ilha das Lendas",
        )

    def test_player_records_use_provider_identity_and_quarantine_name_collisions(
        self,
    ) -> None:
        players = pd.DataFrame(
            [
                {
                    "date": "2026-01-01",
                    "league": "LCK",
                    "playerid": "stable-1",
                    "playername": "Renamed",
                    "position": "mid",
                    "teamname": "A",
                    "result": 1,
                },
                {
                    "date": "2026-02-01",
                    "league": "LCK",
                    "playerid": "stable-1",
                    "playername": "Current Name",
                    "position": "mid",
                    "teamname": "B",
                    "result": 0,
                },
                {
                    "date": "2026-01-01",
                    "league": "LPL",
                    "playerid": "collision-1",
                    "playername": "Shared",
                    "position": "top",
                    "teamname": "C",
                    "result": 1,
                },
                {
                    "date": "2026-01-02",
                    "league": "LPL",
                    "playerid": "collision-2",
                    "playername": "Shared",
                    "position": "top",
                    "teamname": "D",
                    "result": 0,
                },
                {
                    "date": "2026-01-03",
                    "league": "LPL",
                    "playerid": None,
                    "playername": "Missing ID",
                    "position": "sup",
                    "teamname": "E",
                    "result": 1,
                },
            ]
        )

        records = build_player_records(players)

        self.assertEqual(set(records), {"Current Name"})
        self.assertEqual(records["Current Name"]["player_id"], "stable-1")
        self.assertEqual(
            records["Current Name"]["identity_source"],
            "provider_playerid",
        )
        self.assertEqual(records["Current Name"]["games"], 2)

    def test_full_team_feed_adapter_keeps_developmental_games(self) -> None:
        team_games = pd.DataFrame(
            [
                {"gameid": "g1", "date": "2026-01-01", "league": "TCL", "side": "Blue", "position": "team", "teamname": "Misa Esports", "result": 1, "source": "oe"},
                {"gameid": "g1", "date": "2026-01-01", "league": "TCL", "side": "Red", "position": "team", "teamname": "Other", "result": 0, "source": "oe"},
                {"gameid": "g2", "date": "2026-01-02", "league": "CD", "side": "Blue", "position": "team", "teamname": "KaBuM! Ilha das Lendas", "result": 1, "source": "grid"},
                {"gameid": "g2", "date": "2026-01-02", "league": "CD", "side": "Red", "position": "team", "teamname": "Other 2", "result": 0, "source": "grid"},
            ]
        )
        maps = build_maps_frame_from_team_games(team_games)
        self.assertEqual(len(maps), 2)
        self.assertEqual(set(maps["competition_tier"]), {"tier2"})
        by_game = maps.set_index("game_uid")
        self.assertTrue(bool(by_game.loc["g1", "source_oe"]))
        self.assertEqual(
            by_game.loc["g2", "map_detail_source"],
            "grid_team_aggregate",
        )

    def test_team_records_publish_tier_aggregates(self) -> None:
        records = build_team_records(
            pd.DataFrame(
                [
                    {"date": "2026-01-01", "league": "LCK", "blue_team": "A", "red_team": "B", "y_blue_win": 1},
                    {"date": "2026-01-02", "league": "TCL", "blue_team": "A", "red_team": "C", "y_blue_win": 0},
                ]
            )
        )
        self.assertIsNone(records["A"]["current_tier"])
        self.assertIsNone(records["A"]["current_league"])
        self.assertEqual(records["A"]["by_tier"]["tier1"]["games"], 1)
        self.assertEqual(records["A"]["by_tier"]["tier2"]["games"], 1)

    def test_current_tournament_membership_merges_stage_labels_and_excludes_old_teams(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-07-20",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026 (Group Ascend)",
                    "blue_team": "Current A",
                    "red_team": "Current B",
                    "y_blue_win": 1,
                },
                {
                    "date": "2026-07-21",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026 (Group Nirvana)",
                    "blue_team": "Current C",
                    "red_team": "Current D",
                    "y_blue_win": 0,
                },
                {
                    "date": "2026-06-01",
                    "league": "LPL",
                    "tournament": "LPL - Split 2 2026 (Regular Season)",
                    "blue_team": "Former",
                    "red_team": "Current A",
                    "y_blue_win": 0,
                },
            ]
        )
        membership = build_current_tournament_membership(
            maps,
            as_of="2026-07-26T00:00:00Z",
            registry=_current_lpl_registry(),
        )
        self.assertEqual(tournament_family("LPL - Split 3 2026 (Group Ascend)"), "LPL - Split 3 2026")
        self.assertEqual(membership["leagues"], {"LPL": "LPL - Split 3 2026"})
        self.assertEqual(membership["team_leagues"]["current-a"]["LPL"], "LPL - Split 3 2026")
        self.assertNotIn("former", membership["team_leagues"])
        self.assertEqual(
            membership["observation_audit"]["LPL"][
                "observed_not_registered"
            ],
            [],
        )

        records = build_team_records(maps, membership, tournament_maps=maps)
        self.assertEqual(records["Current A"]["current_tournament"], "LPL - Split 3 2026")
        self.assertIsNone(records["Former"]["current_tournament"])
        self.assertEqual(records["Current A"]["by_tournament"]["LPL|LPL - Split 3 2026"]["games"], 1)

    def test_player_records_inherit_current_tournament_from_current_team(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-07-20",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026 (Group Ascend)",
                    "blue_team": "Current A",
                    "red_team": "Current B",
                    "y_blue_win": 1,
                }
            ]
        )
        players = pd.DataFrame(
            [
                {
                    "date": "2026-07-20",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026 (Group Ascend)",
                    "teamname": "Current A",
                    "playername": "Player A",
                    "position": "mid",
                    "result": 1,
                }
            ]
        )
        membership = build_current_tournament_membership(
            maps,
            as_of="2026-07-26T00:00:00Z",
            registry=_current_lpl_registry(),
        )
        records = build_player_records(players, membership)
        self.assertEqual(records["Player A"]["current_tournament"], "LPL - Split 3 2026")
        self.assertEqual(
            records["Player A"]["current_affiliation_basis"],
            "observed_current_tournament_map",
        )

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
            current_membership={
                "authority": "test",
                "as_of": "2026-07-26T00:00:00Z",
                "team_leagues": {
                    "a": {"LCS": "LCS 2026"},
                    "b": {"LCS": "LCS 2026"},
                },
            },
        )
        self.assertEqual(payload["as_of"], "2026-07-26T00:00:00Z")
        self.assertEqual(payload["previous_as_of"], "2026-07-19T00:00:00Z")
        self.assertIn("A0", payload["by_player"])
        self.assertIn("tier1", payload["by_player"]["A0"])
        self.assertEqual(
            len(
                {
                    payload["by_player"][f"A{role_index}"]["all"]["rank"]
                    for role_index in range(5)
                }
            ),
            1,
        )


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
                {"game_uid": "s1g1", "date": "2026-01-01 10:00", "league": "LCS", "blue_team": "A", "red_team": "B", "y_blue_win": 1, "grid_series_id": "s1", "grid_game_index": 1, "series_format": "Bo3"},
                {"game_uid": "s1g2", "date": "2026-01-01 10:35", "league": "LCS", "blue_team": "A", "red_team": "B", "y_blue_win": 1, "grid_series_id": "s1", "grid_game_index": 2, "series_format": "Bo3"},
                {"game_uid": "s2g1", "date": "2026-01-02 10:00", "league": "LCS", "blue_team": "A", "red_team": "B", "y_blue_win": 0, "grid_series_id": "s2", "grid_game_index": 1, "series_format": "Bo1"},
            ]
        )
        snapshot, meta = fit_hierarchical_bt(maps, write=False)
        self.assertEqual(meta["n_series"], 2)
        self.assertEqual(meta["n_maps"], 3)
        self.assertTrue((snapshot["rating_p05"] < snapshot["mu_total"]).all())
        self.assertTrue(set(snapshot["model"]) == {"hierarchical_bt"})

    def test_gapped_explicit_grid_series_are_excluded_from_rating_fit(self) -> None:
        maps = pd.DataFrame(
            [
                {
                    "date": "2026-01-01 10:00",
                    "league": "LCS",
                    "blue_team": "A",
                    "red_team": "B",
                    "y_blue_win": 1,
                    "grid_series_id": "gapped",
                    "grid_game_index": 1,
                    "series_format": "Bo3",
                },
                {
                    "date": "2026-01-01 10:35",
                    "league": "LCS",
                    "blue_team": "A",
                    "red_team": "B",
                    "y_blue_win": 1,
                    "grid_series_id": "gapped",
                    "grid_game_index": 3,
                    "series_format": "Bo3",
                },
            ]
        )
        _, meta = fit_hierarchical_bt(maps, write=False)
        self.assertEqual(meta["n_series"], 0)
        self.assertEqual(meta["skipped_gapped_series"], 1)


if __name__ == "__main__":
    unittest.main()
