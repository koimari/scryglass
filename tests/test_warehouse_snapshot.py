import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lol_kills.export import warehouse_snapshot as snapshot


class WarehouseSnapshotTests(unittest.TestCase):
    def test_pointer_rejects_remote_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pointer = Path(temp) / "pointer.json"
            for url in (
                "file:///tmp/archive.tar.gz",
                "https://example.com/archive.tar.gz",
                "https://token@example.public.blob.vercel-storage.com/archive.tar.gz",
            ):
                pointer.write_text(json.dumps({"url": url}))
                with self.assertRaisesRegex(RuntimeError, "remote warehouse snapshots are disabled"):
                    snapshot._read_pointer(pointer)

    def test_bootstrap_writes_oe_shaped_parquet_from_public_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = root / "packs" / "vtest"
            (pack / "team_games" / "year=2026").mkdir(parents=True)
            (pack / "player_games" / "year=2026").mkdir(parents=True)
            pd.DataFrame(
                [{"gameid": "g1", "year": 2026, "side": "Blue", "source": "oe"}]
            ).to_parquet(pack / "team_games" / "year=2026" / "part.parquet", index=False)
            pd.DataFrame(
                [{"gameid": "g1", "year": 2026, "side": "Blue", "position": "top"}]
            ).to_parquet(pack / "player_games" / "year=2026" / "part.parquet", index=False)

            old_root = snapshot.ROOT
            try:
                snapshot.ROOT = root
                pack_id = snapshot.bootstrap_from_public_pack(root / "packs", "vtest")
                self.assertEqual(pack_id, "vtest")
                self.assertEqual(
                    len(pd.read_parquet(root / "data/lol/warehouse/parquet/oe_team_games.parquet")),
                    1,
                )
                self.assertEqual(
                    len(pd.read_parquet(root / "data/lol/warehouse/parquet/oe_player_games.parquet")),
                    1,
                )
            finally:
                snapshot.ROOT = old_root

    def test_remote_save_is_disabled_before_reading_credentials_or_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parquet_dir = root / "data/lol/warehouse/parquet"
            parquet_dir.mkdir(parents=True)
            pd.DataFrame([{"gameid": "g1", "source": "oe"}]).to_parquet(
                parquet_dir / "oe_team_games.parquet", index=False
            )
            pointer = root / "data/lol/warehouse_snapshot.json"
            old_root = snapshot.ROOT
            old_default_pointer = snapshot.DEFAULT_POINTER
            try:
                snapshot.ROOT = root
                snapshot.DEFAULT_POINTER = pointer
                with patch("urllib.request.urlopen") as request:
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "remote warehouse snapshots are disabled",
                    ):
                        snapshot.save_snapshot(pointer)
                request.assert_not_called()
            finally:
                snapshot.ROOT = old_root
                snapshot.DEFAULT_POINTER = old_default_pointer


if __name__ == "__main__":
    unittest.main()
