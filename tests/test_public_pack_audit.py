from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import pyarrow as pa

from lol_kills.audit_public_pack import audit_pack
from lol_kills.export import pack_spec
from lol_kills.export.public_pack import _ensure_columns


class PublicPackAuditTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
