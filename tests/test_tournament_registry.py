from __future__ import annotations

import copy
import unittest

import pandas as pd

from lol_kills.etl.competition import team_identity_key
from lol_kills.etl.series_ledger import build_canonical_series_ledger
from lol_kills.etl.tournament_registry import (
    TournamentRegistryError,
    annotate_maps_with_tournament_registry,
    current_membership_from_registry,
    load_tournament_registry,
    validate_tournament_registry,
)


def _registry() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "snapshot_id": "test-snapshot",
        "authority": "Riot Games LoL Esports",
        "observed_at": "2026-07-26T00:00:00Z",
        "review_due_at": "2026-08-02T00:00:00Z",
        "tournaments": [
            {
                "tournament_id": "current-lck",
                "league": "LCK",
                "name": "LCK - Split 3 2026",
                "status": "current",
                "source_url": "https://lolesports.com/test",
                "participants": [
                    {"display_name": "T1", "short_code": "T1"},
                    {"display_name": "Gen.G", "short_code": "GEN"},
                ],
                "stages": [
                    {
                        "stage_id": "regular",
                        "scheduled_best_of": None,
                        "format_status": "unverified",
                    }
                ],
            }
        ],
    }


class TournamentRegistryTests(unittest.TestCase):
    def test_checked_in_registry_has_exact_current_tier_one_population(self) -> None:
        registry = load_tournament_registry()
        membership = current_membership_from_registry(
            registry,
            as_of="2026-07-27T00:00:00Z",
        )
        self.assertEqual(
            set(membership["leagues"]),
            {"LCK", "LPL", "LEC", "LCS", "CBLOL", "LCP"},
        )
        self.assertEqual(
            sum(
                len(teams)
                for teams in membership["participants_by_league"].values()
            ),
            56,
        )
        self.assertEqual(
            len(membership["participants_by_league"]["LPL"]),
            12,
        )
        self.assertNotIn(
            team_identity_key("Ultra Prime"),
            membership["participants_by_league"]["LPL"],
        )
        self.assertIn(
            team_identity_key("Weibo Gaming"),
            membership["participants_by_league"]["LPL"],
        )

    def test_registry_is_invalid_after_review_deadline(self) -> None:
        with self.assertRaisesRegex(
            TournamentRegistryError, "overdue for authoritative review"
        ):
            current_membership_from_registry(
                _registry(),
                as_of="2026-08-03T00:00:00Z",
            )

    def test_registry_cannot_be_used_before_it_was_observed(self) -> None:
        with self.assertRaisesRegex(
            TournamentRegistryError, "observed after"
        ):
            current_membership_from_registry(
                _registry(),
                as_of="2026-07-25T23:59:59Z",
            )

    def test_unknown_format_must_be_explicit(self) -> None:
        payload = copy.deepcopy(_registry())
        payload["tournaments"][0]["stages"][0]["format_status"] = "verified"
        with self.assertRaisesRegex(
            TournamentRegistryError, "null format must be marked unverified"
        ):
            validate_tournament_registry(payload)

    def test_current_sponsor_names_share_historical_organization_keys(self) -> None:
        self.assertEqual(
            team_identity_key("DN Freecs"),
            team_identity_key("DN SOOPers"),
        )
        self.assertEqual(
            team_identity_key("DRX"),
            team_identity_key("KIWOOM DRX"),
        )
        self.assertEqual(
            team_identity_key("BRION"),
            team_identity_key("HANJIN BRION"),
        )
        self.assertEqual(
            team_identity_key("Cloud9"),
            team_identity_key("Cloud9 Kia"),
        )
        self.assertEqual(
            team_identity_key("Team Liquid"),
            team_identity_key("Team Liquid Alienware"),
        )

    def test_duplicate_canonical_team_identity_is_rejected(self) -> None:
        payload = copy.deepcopy(_registry())
        payload["tournaments"][0]["participants"].append(
            {"display_name": "Gen.G Esports", "short_code": "GEN2"}
        )
        with self.assertRaisesRegex(
            TournamentRegistryError, "duplicate team identity"
        ):
            validate_tournament_registry(payload)

    def test_verified_current_stage_format_completes_series(self) -> None:
        registry = load_tournament_registry()
        maps = pd.DataFrame(
            [
                {
                    "game_uid": "lpl-1",
                    "date": "2026-07-26T10:00:00Z",
                    "source": "oe",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026 (Group Ascend)",
                    "playoffs": 0,
                    "game": 1,
                    "blue_team": "Bilibili Gaming",
                    "red_team": "Top Esports",
                    "y_blue_win": 1,
                },
                {
                    "game_uid": "lpl-2",
                    "date": "2026-07-26T11:00:00Z",
                    "source": "oe",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026 (Group Ascend)",
                    "playoffs": 0,
                    "game": 2,
                    "blue_team": "Top Esports",
                    "red_team": "Bilibili Gaming",
                    "y_blue_win": 0,
                },
            ]
        )
        annotated = annotate_maps_with_tournament_registry(
            maps,
            registry,
            as_of="2026-07-27T00:00:00Z",
        )
        self.assertEqual(set(annotated.maps["series_format"]), {"Bo3"})
        self.assertEqual(annotated.audit["filled_rows"], 2)
        ledger = build_canonical_series_ledger(annotated.maps)
        self.assertEqual(ledger.audit["n_rating_eligible_series"], 1)
        self.assertEqual(
            ledger.series.iloc[0]["series_format_source"],
            "riot-registry:riot-tier1-current-2026-07-26:group-stage",
        )

    def test_unverified_stage_is_not_guessed(self) -> None:
        registry = load_tournament_registry()
        maps = pd.DataFrame(
            [
                {
                    "game_uid": "lec-1",
                    "date": "2026-07-26T10:00:00Z",
                    "source": "oe",
                    "league": "LEC",
                    "tournament": "LEC - Split 3 2026 (Regular Season)",
                    "playoffs": 0,
                    "game": 1,
                    "blue_team": "G2 Esports",
                    "red_team": "Fnatic",
                    "y_blue_win": 1,
                }
            ]
        )
        annotated = annotate_maps_with_tournament_registry(
            maps,
            registry,
            as_of="2026-07-27T00:00:00Z",
        )
        self.assertEqual(annotated.audit["filled_rows"], 0)
        self.assertNotIn("series_format", annotated.maps.columns)

    def test_source_format_conflict_is_quarantined(self) -> None:
        registry = load_tournament_registry()
        maps = pd.DataFrame(
            [
                {
                    "game_uid": "lpl-conflict",
                    "date": "2026-07-26T10:00:00Z",
                    "source": "oe",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026 (Group Ascend)",
                    "playoffs": 0,
                    "game": 1,
                    "series_format": "Bo5",
                    "blue_team": "Bilibili Gaming",
                    "red_team": "Top Esports",
                    "y_blue_win": 1,
                }
            ]
        )
        annotated = annotate_maps_with_tournament_registry(
            maps,
            registry,
            as_of="2026-07-27T00:00:00Z",
        )
        self.assertEqual(annotated.audit["conflict_rows"], 1)
        ledger = build_canonical_series_ledger(annotated.maps)
        self.assertIn(
            "series_format_registry_conflict",
            ledger.series.iloc[0]["quarantine_reasons"],
        )


if __name__ == "__main__":
    unittest.main()
