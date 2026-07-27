from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa

from lol_kills.audit_public_pack import (
    _audit_player_rating_release_contract,
    _audit_rating_release_contract,
    audit_pack,
    require_release_gate,
)
from lol_kills.export import pack_spec
from lol_kills.export.public_pack import (
    _apply_map_provenance_contract,
    _ensure_columns,
    _filter_years,
    _filter_to_game_ids,
    _filter_unverified_grid_games,
    _source_summary,
    require_pinned_model_files,
)
from lol_kills.export.pack_records import complete_public_map_population


class PublicPackAuditTests(unittest.TestCase):
    def test_manifest_requires_one_immutable_data_and_model_bundle_clock(
        self,
    ) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            model_dir = root / "models"
            model_dir.mkdir(parents=True)
            artifact = model_dir / "draft_composition.json"
            artifact.write_text("{}", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "pack_id": "v-test",
                        "schema_version": pack_spec.SCHEMA_VERSION,
                        "files": [
                            {
                                "path": "models/draft_composition.json",
                                "relative": "models/draft_composition.json",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_pack(root)

        self.assertIn(
            "model_bundle_clock_unbound",
            {finding["code"] for finding in report["findings"]},
        )

    def test_player_outcome_release_withholds_rank_and_reconciles_maps(
        self,
    ) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            features = root / "features"
            features.mkdir(parents=True)
            pd.DataFrame([{"player": "A"}, {"player": "B"}]).to_parquet(
                features / "player_ratings_snapshot.parquet",
                index=False,
            )
            meta = {
                "n_maps": 10,
                "n_input_maps": 10,
                "n_identity_eligible_maps": 8,
                "identity_eligible_map_rate": 0.8,
                "n_players": 2,
                "n_unique_outcome_exposure_players": 1,
                "n_shared_outcome_history_players": 1,
                "outcome_ordering_verified": False,
                "individual_skill_estimand": False,
                "identity_audit": {
                    "n_valid_maps": 8,
                    "n_quarantined_maps": 2,
                    "quarantined_game_uid_examples": ["g9", "g10"],
                    "n_display_name_collisions": 1,
                    "display_name_collision_examples": ["shared"],
                },
            }
            meta_path = features / "player_ratings_meta.json"
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            findings: list[dict] = []
            summary = _audit_player_rating_release_contract(
                root, findings
            )
            self.assertTrue(summary["contract_ok"])
            self.assertEqual(findings, [])

            meta["outcome_ordering_verified"] = True
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            findings = []
            summary = _audit_player_rating_release_contract(
                root, findings
            )
            self.assertFalse(summary["contract_ok"])
            self.assertEqual(
                findings[0]["code"],
                "player_rating_claim_or_denominator_invalid",
            )

            meta["outcome_ordering_verified"] = False
            meta["identity_audit"]["display_name_collisions"] = {
                "shared": ["provider-a", "provider-b"]
            }
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            findings = []
            summary = _audit_player_rating_release_contract(root, findings)
            self.assertFalse(summary["contract_ok"])

    def test_year_filter_uses_source_year_before_conflicting_partition_year(self) -> None:
        table = pa.table(
            {
                "row": ["stale", "current", "grid-fallback"],
                "oe_year": pa.array([2024, 2025, None], type=pa.int64()),
                "year": pa.array([2025, 2025, 2026], type=pa.int64()),
            }
        )

        filtered = _filter_years(table, [2025, 2026], ("oe_year", "year"))

        self.assertEqual(filtered["row"].to_pylist(), ["current", "grid-fallback"])
        self.assertEqual(filtered["year"].to_pylist(), [2025, 2026])

    def test_public_map_population_appends_only_missing_team_maps(self) -> None:
        feature_maps = pd.DataFrame(
            [
                {
                    "game_uid": "game-1",
                    "oe_gameid": "game-1",
                    "date": "2026-01-01",
                    "blue_pick1": "Aatrox",
                }
            ]
        )
        team_maps = pd.DataFrame(
            [
                {
                    "game_uid": "game-1",
                    "oe_gameid": "game-1",
                    "date": "2026-01-01",
                    "blue_team": "Alpha",
                },
                {
                    "game_uid": "game-2",
                    "oe_gameid": "game-2",
                    "date": "2026-01-02",
                    "blue_team": "Gamma",
                },
            ]
        )

        completed, audit = complete_public_map_population(feature_maps, team_maps)

        self.assertEqual(completed["game_uid"].tolist(), ["game-1", "game-2"])
        detailed = completed.set_index("game_uid").loc["game-1"]
        aggregate = completed.set_index("game_uid").loc["game-2"]
        self.assertEqual(detailed["blue_pick1"], "Aatrox")
        self.assertEqual(detailed["map_detail_source"], "oe_wide_feature_map")
        self.assertTrue(pd.isna(aggregate["blue_pick1"]))
        self.assertEqual(aggregate["map_detail_source"], "oe_team_aggregate")
        self.assertEqual(audit["appended_team_aggregate_maps"], 1)

    def test_public_map_population_rejects_duplicate_identity(self) -> None:
        feature_maps = pd.DataFrame(
            [
                {"game_uid": "game-1", "date": "2026-01-01"},
                {"game_uid": "game-1", "date": "2026-01-01"},
            ]
        )
        team_maps = pd.DataFrame(
            [{"game_uid": "game-1", "date": "2026-01-01"}]
        )

        with self.assertRaisesRegex(ValueError, "duplicate identities"):
            complete_public_map_population(feature_maps, team_maps)

    def test_export_fails_closed_when_a_pinned_model_is_missing(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for name in pack_spec.PINNED_MODEL_FILES[:-1]:
                (root / name).write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(
                FileNotFoundError,
                pack_spec.PINNED_MODEL_FILES[-1],
            ):
                require_pinned_model_files(root)

    def test_grid_games_require_verified_completion_provenance(self) -> None:
        team = pa.Table.from_pandas(
            pd.DataFrame(
                [
                    {
                        "gameid": "verified",
                        "side": "Blue",
                        "source": "grid",
                        "grid_completion_source": "end_state_summary",
                    },
                    {
                        "gameid": "verified",
                        "side": "Red",
                        "source": "grid",
                        "grid_completion_source": "end_state_summary",
                    },
                    {
                        "gameid": "unverified",
                        "side": "Blue",
                        "source": "grid",
                        "grid_completion_source": None,
                    },
                    {
                        "gameid": "unverified",
                        "side": "Red",
                        "source": "grid",
                        "grid_completion_source": None,
                    },
                    {
                        "gameid": "oe",
                        "side": "Blue",
                        "source": "oe",
                        "grid_completion_source": None,
                    },
                ]
            ),
            preserve_index=False,
        )

        filtered, audit = _filter_unverified_grid_games(team)

        self.assertEqual(
            set(filtered["gameid"].to_pylist()),
            {"verified", "oe"},
        )
        self.assertEqual(audit["grid_games_seen"], 2)
        self.assertEqual(audit["grid_games_retained"], 1)
        self.assertEqual(audit["grid_games_excluded_unverified"], 1)

        players = pa.table(
            {
                "gameid": ["verified", "unverified", "oe"],
                "player": ["A", "B", "C"],
            }
        )
        retained = _filter_to_game_ids(players, {"verified", "oe"})
        self.assertEqual(retained["player"].to_pylist(), ["A", "C"])

    def test_source_summary_names_mixed_sources_and_dedupe_precedence(self) -> None:
        team = pd.DataFrame(
            {
                "gameid": ["oe-1", "oe-1", "grid-1", "grid-1"],
                "source": ["oe", "oe", "grid", "grid"],
            }
        )
        players = pd.DataFrame(
            {
                "gameid": ["oe-1", "grid-1"],
                "source": ["oe", "grid"],
            }
        )
        maps = pd.DataFrame(
            {
                "canonical_map_source": ["oe", "grid_gap_fill"],
                "map_detail_source": [
                    "grid_event_detail",
                    "grid_team_aggregate",
                ],
            }
        )

        summary, attribution = _source_summary(
            team,
            players,
            maps,
            data_as_of="2026-07-26T00:00:00Z",
            grid_completion_gate={"grid_games_retained": 1},
        )

        self.assertIn("Oracle's Elixir", attribution)
        self.assertIn("GRID", attribution)
        self.assertEqual(summary["sources"]["team_games"]["grid"]["maps"], 1)
        self.assertEqual(
            summary["sources"]["canonical_map_inclusion"]["oe"]["maps"],
            1,
        )
        self.assertEqual(
            summary["sources"]["map_detail_enrichment"][
                "grid_event_detail"
            ]["maps"],
            1,
        )
        self.assertEqual(
            summary["canonicalization"]["overlap_precedence"],
            "oracle_elixir_then_verified_grid_gap_fill",
        )

    def test_map_provenance_separates_canonical_origin_from_grid_detail(
        self,
    ) -> None:
        team = pd.DataFrame(
            {
                "gameid": ["same", "same", "gap", "gap"],
                "source": ["oe", "oe", "grid", "grid"],
            }
        )
        maps = pd.DataFrame(
            {
                "oe_gameid": ["same", "gap"],
                "game_uid": ["same", "gap"],
                "map_detail_source": [
                    "grid_event_detail",
                    "grid_event_detail",
                ],
            }
        )
        result = _apply_map_provenance_contract(maps, team)
        self.assertEqual(
            result["canonical_map_source"].tolist(),
            ["oe", "grid_gap_fill"],
        )
        self.assertEqual(result["source_oe"].tolist(), [True, False])
        self.assertEqual(result["source_grid"].tolist(), [True, True])

    def test_public_pack_spec_excludes_uncited_working_artifacts(self) -> None:
        all_paths = (
            *pack_spec.PINNED_MODEL_FILES,
            *pack_spec.GRUBS_MODEL_FILES,
            *pack_spec.GRUBS_PDF_FILES,
        )
        denied_tokens = (
            "tierlist",
            "_brief",
            "_paper.md",
            "ranked_contest_proof",
            "action_graph.png",
        )
        self.assertFalse(
            any(token in path for path in all_paths for token in denied_tokens)
        )
        self.assertIn("draft_composition.json", pack_spec.PINNED_MODEL_FILES)

    def test_public_pack_path_gate_rejects_all_tierlist_download_surfaces(
        self,
    ) -> None:
        blocked = (
            "models/champ_tierlist_16_13_blade_chest.json",
            "models/champ_oe_lenses.json",
            "models/tierlists_csv/renamed.csv",
            "models/otherwise_renamed_model.csv",
        )
        for path in blocked:
            with self.subTest(path=path):
                with self.assertRaisesRegex(ValueError, "quarantine"):
                    pack_spec.require_publication_paths_allowed([path])

        self.assertFalse(hasattr(pack_spec, "TIERLIST_CSV_GLOB"))
        pack_spec.require_publication_paths_allowed(
            [f"models/{name}" for name in pack_spec.PINNED_MODEL_FILES]
        )

    def test_complete_pack_audit_rejects_stale_schema_quarantined_paths_and_missing_registry(
        self,
    ) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            blocked = root / "models" / "tierlists_csv" / "tierlist.csv"
            blocked.parent.mkdir(parents=True)
            blocked.write_text("champion,score\nA,1\n", encoding="utf-8")
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "pack_id": "stale",
                        "schema_version": "1.3.0",
                        "files": [
                            {
                                "path": "models/tierlists_csv/tierlist.csv",
                                "relative": "models/tierlists_csv/tierlist.csv",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = audit_pack(root)

        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("public_schema_version_stale", codes)
        self.assertIn("quarantined_public_artifacts", codes)
        self.assertIn("required_public_artifacts_missing", codes)
        self.assertIn("current_membership_registry_missing", codes)
        self.assertIn("team_rating_release_missing", codes)
        self.assertFalse(report["release_gate"]["ready"])

    def test_pinned_model_resolution_cannot_bypass_tierlist_quarantine(
        self,
    ) -> None:
        original = pack_spec.PINNED_MODEL_FILES
        try:
            pack_spec.PINNED_MODEL_FILES = (
                "champ_tierlist_calibrated_delta_wr.json",
            )
            with TemporaryDirectory() as temp:
                candidate = Path(temp) / pack_spec.PINNED_MODEL_FILES[0]
                candidate.write_text("{}", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "quarantine"):
                    require_pinned_model_files(Path(temp))
        finally:
            pack_spec.PINNED_MODEL_FILES = original

    def test_export_materializes_missing_public_map_columns(self) -> None:
        table = _ensure_columns(pa.table({"game_uid": ["g1"]}), pack_spec.maps_columns())
        self.assertEqual(table.num_rows, 1)
        self.assertEqual(set(table.column_names), set(pack_spec.maps_columns()))

    def test_audit_catches_missing_grid_provenance_and_gapped_series(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            map_dir = root / "maps" / "year=2026"
            map_dir.mkdir(parents=True)
            row = {column: None for column in pack_spec.maps_columns() if column != "grid_completion_source"}
            row.update(
                {
                    "game_uid": "grid-game-1",
                    "oe_gameid": "oe-game-1",
                    "blue_teamname": "Alpha",
                    "red_teamname": "Beta",
                    "blue_result": 1,
                    "red_result": 0,
                    "y_blue_win": 1,
                    "gamelength": 1800,
                    "total_kills": 12,
                    "source_grid": True,
                    "source_oe": False,
                    "map_detail_source": "oe_wide_feature_map",
                    "grid_series_id": "series-1",
                    "grid_game_index": 3,
                    "league": "INTL",
                    "tournament": "NACL - Summer 2026",
                }
            )
            pd.DataFrame([row]).to_parquet(map_dir / "part.parquet", index=False)
            (root / "manifest.json").write_text(
                json.dumps({"pack_id": "test", "schema_version": "1.3.0", "data_as_of": "2026-07-26T00:00:00Z"}),
                encoding="utf-8",
            )

            report = audit_pack(root)

        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("maps_schema_missing_columns", codes)
        self.assertIn("grid_completion_provenance_missing", codes)
        self.assertIn("gapped_grid_series", codes)
        self.assertIn("developmental_league_leaked_to_intl", codes)
        self.assertGreater(report["counts"]["launch blocker"], 0)

    def test_audit_accepts_valid_map_grain(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            map_dir = root / "maps" / "year=2026"
            map_dir.mkdir(parents=True)
            row = {column: None for column in pack_spec.maps_columns()}
            row.update(
                {
                    "game_uid": "game-1",
                    "oe_gameid": "oe-game-1",
                    "blue_teamname": "Alpha",
                    "red_teamname": "Beta",
                    "blue_result": 1,
                    "red_result": 0,
                    "y_blue_win": 1,
                    "gamelength": 1800,
                    "total_kills": 12,
                    "source_grid": True,
                    "source_oe": False,
                    "canonical_map_source": "grid_gap_fill",
                    "map_detail_source": "grid_event_detail",
                    "grid_series_id": "series-1",
                    "grid_game_index": 1,
                    "grid_completion_source": "events_game_end",
                    "league": "LPL",
                    "tournament": "LPL - Split 3 2026",
                }
            )
            pd.DataFrame([row]).to_parquet(map_dir / "part.parquet", index=False)
            (root / "manifest.json").write_text(json.dumps({"pack_id": "test", "schema_version": "1.3.0"}), encoding="utf-8")

            report = audit_pack(root)

        self.assertEqual(report["maps"]["rows"], 1)
        self.assertEqual(report["counts"]["launch blocker"], 0)

    def test_series_audit_scores_teams_across_side_swaps(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            map_dir = root / "maps" / "year=2026"
            map_dir.mkdir(parents=True)
            rows = []
            for index, (blue, red, blue_win) in enumerate(
                (("Alpha", "Beta", 1), ("Beta", "Alpha", 0)),
                start=1,
            ):
                row = {
                    column: None for column in pack_spec.maps_columns()
                }
                row.update(
                    {
                        "game_uid": f"side-swap-{index}",
                        "oe_gameid": f"side-swap-{index}",
                        "blue_teamname": blue,
                        "red_teamname": red,
                        "blue_result": blue_win,
                        "red_result": 1 - blue_win,
                        "y_blue_win": blue_win,
                        "gamelength": 1800,
                        "total_kills": 20,
                        "source_grid": True,
                        "source_oe": False,
                        "map_detail_source": "grid_event_detail",
                        "grid_series_id": "side-swap-series",
                        "grid_game_index": index,
                        "grid_completion_source": "events_game_end",
                        "league": "LPL",
                        "tournament": "LPL - Split 3 2026",
                    }
                )
                rows.append(row)
            pd.DataFrame(rows).to_parquet(
                map_dir / "part.parquet", index=False
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {"pack_id": "test", "schema_version": "1.3.0"}
                ),
                encoding="utf-8",
            )

            report = audit_pack(root)

        self.assertEqual(report["series"]["tied_multi_map_series"], 0)
        self.assertNotIn(
            "tied_grid_series",
            {finding["code"] for finding in report["findings"]},
        )

    def test_gapped_series_is_nonblocking_when_explicitly_quarantined(
        self,
    ) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            map_dir = root / "maps" / "year=2026"
            map_dir.mkdir(parents=True)
            rows = []
            for index in (2, 3):
                row = {
                    column: None for column in pack_spec.maps_columns()
                }
                row.update(
                    {
                        "game_uid": f"gap-safe-{index}",
                        "oe_gameid": f"gap-safe-{index}",
                        "blue_teamname": "Alpha",
                        "red_teamname": "Beta",
                        "blue_result": 1,
                        "red_result": 0,
                        "y_blue_win": 1,
                        "gamelength": 1800,
                        "total_kills": 20,
                        "source_grid": True,
                        "source_oe": False,
                        "map_detail_source": "grid_event_detail",
                        "grid_series_id": "gap-safe",
                        "grid_game_index": index,
                        "grid_completion_source": "events_game_end",
                        "series_rating_eligible": False,
                        "canonical_series_status": "completed",
                        "league": "LPL",
                        "tournament": "LPL - Split 3 2026",
                    }
                )
                rows.append(row)
            pd.DataFrame(rows).to_parquet(
                map_dir / "part.parquet", index=False
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {"pack_id": "test", "schema_version": "1.3.0"}
                ),
                encoding="utf-8",
            )

            report = audit_pack(root)

        self.assertEqual(report["series"]["gapped_series"], 1)
        self.assertEqual(
            report["series"]["quarantined_gapped_series"], 1
        )
        self.assertEqual(report["series"]["unsafe_gapped_series"], 0)
        self.assertNotIn(
            "gapped_grid_series",
            {finding["code"] for finding in report["findings"]},
        )

    def test_audit_catches_current_tournament_record_mismatch(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            (root / "features").mkdir(parents=True)
            (root / "features" / "team_records.json").write_text(
                json.dumps(
                    {
                        "Former": {
                            "current_league": "LPL",
                            "current_tournament": "LPL - Split 2 2026",
                            "current_date": "2026-07-25",
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "pack_id": "test",
                        "current_tournaments": {"LPL": "LPL - Split 3 2026"},
                        "current_tournament_as_of": "2026-07-26T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )

            report = audit_pack(root)

        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("current_tournament_membership_mismatch", codes)
        self.assertEqual(report["membership"]["mismatches"], 1)

    def test_audit_catches_rows_outside_declared_pack_years(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            player_dir = root / "player_games" / "year=2025"
            player_dir.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "gameid": "old-game",
                        "date": "2024-06-05",
                        "oe_year": 2024,
                        "year": 2025,
                        "side": "Blue",
                        "position": "top",
                    }
                ]
            ).to_parquet(player_dir / "part.parquet", index=False)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "pack_id": "test",
                        "filters": {"years": [2025, 2026]},
                    }
                ),
                encoding="utf-8",
            )

            report = audit_pack(root)

        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("player_games_outside_declared_years", codes)
        self.assertIn("player_games_year_field_conflict", codes)

    def test_expired_membership_registry_uses_injected_current_utc(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp) / "pack"
            root.mkdir(parents=True)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "pack_id": "test",
                        "membership_registry": {
                            "snapshot_id": "registry-test",
                            "checked_at": "2026-07-01T00:00:00Z",
                            "review_due_at": "2026-07-20T00:00:00Z",
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = audit_pack(
                root,
                clock=lambda: datetime(2026, 7, 21, tzinfo=timezone.utc),
            )

        codes = {finding["code"] for finding in report["findings"]}
        self.assertIn("current_membership_registry_stale", codes)
        self.assertEqual(
            report["membership"]["audit_current_utc"],
            "2026-07-21T00:00:00+00:00",
        )
        self.assertFalse(report["release_gate"]["ready"])

    def test_major_finding_fails_release_gate(self) -> None:
        report = {
            "counts": {
                "launch blocker": 0,
                "major": 1,
                "minor": 0,
                "informational": 0,
            }
        }

        with self.assertRaisesRegex(RuntimeError, "major=1"):
            require_release_gate(report)

    def test_team_rating_requires_exact_passing_chronological_gate(
        self,
    ) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features"
            models = root / "models"
            features.mkdir()
            models.mkdir()
            meta = {
                "model": "hierarchical_bt",
                "model_id": "hierarchical_bt",
                "model_version": "hierarchical_bt:code:config",
                "model_code_sha256": "code",
                "model_config_sha256": "config",
                "n_series": 600,
                "input_audit": {"ok": True},
                "series_ledger_audit": {
                    "ok": True,
                    "n_rating_eligible_series": 600,
                    "n_rating_eligible_maps": 1200,
                },
            }
            (features / "ratings_meta.json").write_text(
                json.dumps(meta),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "team": "Alpha",
                        "mu_total": 1500.0,
                        "sigma": 25.0,
                        "rating_p05": 1458.0,
                        "model": "hierarchical_bt",
                        "model_version": "hierarchical_bt:code:config",
                    }
                ]
            ).to_parquet(
                features / "ratings_snapshot.parquet",
                index=False,
            )
            (models / "model_validation_2026-07-27.json").write_text(
                json.dumps({"team_rating": {"gate_status": "not_promoted"}}),
                encoding="utf-8",
            )
            findings: list[dict[str, object]] = []

            summary = _audit_rating_release_contract(
                root,
                {"files": [{"path": "placeholder"}]},
                findings,
            )

        self.assertFalse(summary["model_gate_ok"])
        self.assertIn(
            "team_rating_release_empty_or_unvalidated",
            {finding["code"] for finding in findings},
        )

    def test_team_rating_accepts_matching_passing_chronological_gate(
        self,
    ) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            features = root / "features"
            models = root / "models"
            features.mkdir()
            models.mkdir()
            identity = {
                "model_id": "series_dynamic_bt",
                "model_version": "series_dynamic_bt:code:config",
                "model_code_sha256": "code",
                "model_config_sha256": "config",
            }
            meta = {
                "model": "series_dynamic_bt",
                **identity,
                "n_series": 600,
                "input_audit": {"ok": True},
                "series_ledger_audit": {
                    "ok": True,
                    "n_rating_eligible_series": 600,
                    "n_rating_eligible_maps": 1200,
                },
            }
            (features / "ratings_meta.json").write_text(
                json.dumps(meta),
                encoding="utf-8",
            )
            pd.DataFrame(
                [
                    {
                        "team": "Alpha",
                        "team_key": "alpha",
                        "mu_total": 1500.0,
                        "sigma": 25.0,
                        "rating_p05": 1458.0,
                        "model": "series_dynamic_bt",
                        "model_version": "series_dynamic_bt:code:config",
                        "comparison_component_id": "component-test",
                        "comparison_component_size": 1,
                        "cross_component_rankable": False,
                    }
                ]
            ).to_parquet(
                features / "ratings_snapshot.parquet",
                index=False,
            )
            gate = {
                "team_rating": {
                    "gate_status": "passed",
                    "estimand": "pre_series_organization_strength_probability",
                    **identity,
                    "temporal_audit": {"ok": True},
                    "final_test": {
                        "series": 600,
                        "log_loss": 0.62,
                        "brier": 0.215,
                        "ece_10_equal_width": 0.02,
                    },
                    "paired_primary_comparison": {
                        "primary_score": "log_loss",
                        "decision": "noninferior",
                        "confidence_interval": [-0.01, 0.001],
                    },
                }
            }
            (models / "model_validation_2026-07-27.json").write_text(
                json.dumps(gate),
                encoding="utf-8",
            )
            findings: list[dict[str, object]] = []

            summary = _audit_rating_release_contract(
                root,
                {"files": [{"path": "placeholder"}]},
                findings,
            )

        self.assertTrue(summary["model_gate_ok"])
        self.assertEqual(findings, [])


if __name__ == "__main__":
    unittest.main()
