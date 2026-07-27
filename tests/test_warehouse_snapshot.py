import io
import json
import os
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lol_kills.export import warehouse_snapshot as snapshot


class WarehouseSnapshotTests(unittest.TestCase):
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

    def test_save_uses_stable_blob_path_and_internal_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parquet_dir = root / "data/lol/warehouse/parquet"
            parquet_dir.mkdir(parents=True)
            pd.DataFrame([{"gameid": "g1", "source": "oe"}]).to_parquet(
                parquet_dir / "oe_team_games.parquet", index=False
            )
            pointer = root / "data/lol/warehouse_snapshot.json"
            uploaded: dict[str, bytes] = {}
            old_root = snapshot.ROOT
            old_default_pointer = snapshot.DEFAULT_POINTER
            try:
                snapshot.ROOT = root
                snapshot.DEFAULT_POINTER = pointer

                def fake_blob_put(_token, pathname, data, _content_type, **_kwargs):
                    uploaded[pathname] = data
                    return "https://example.public.blob.vercel-storage.com/" + pathname

                with patch.dict(os.environ, {"BLOB_READ_WRITE_TOKEN": "test-token"}), patch(
                    "lol_kills.export.upload_pack._blob_put", side_effect=fake_blob_put
                ) as put:
                    snapshot.save_snapshot(pointer)
                    snapshot.save_snapshot(pointer)

                self.assertEqual(put.call_count, 2)
                self.assertEqual(list(uploaded), [snapshot.SNAPSHOT_PATH])
                payload = json.loads(pointer.read_text())
                self.assertEqual(payload["pathname"], snapshot.SNAPSHOT_PATH)
                with tarfile.open(fileobj=io.BytesIO(uploaded[snapshot.SNAPSHOT_PATH]), mode="r:gz") as archive:
                    manifest = json.loads(archive.extractfile("snapshot_manifest.json").read())
                self.assertEqual(manifest["schema"], snapshot.SNAPSHOT_SCHEMA)
                self.assertNotIn("warehouse/raw", " ".join(item["path"] for item in manifest["files"]))
            finally:
                snapshot.ROOT = old_root
                snapshot.DEFAULT_POINTER = old_default_pointer

    def test_restore_verifies_snapshot_before_overwriting_warehouse(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            relative = "data/lol/warehouse/parquet/oe_team_games.parquet"
            live_path = root / relative
            live_path.parent.mkdir(parents=True)
            live_path.write_bytes(b"known-good-live-warehouse")
            replacement = b"untrusted-snapshot"
            manifest = {
                "schema": snapshot.SNAPSHOT_SCHEMA,
                "files": [
                    {
                        "path": relative,
                        "bytes": len(replacement),
                        "sha256": "0" * 64,
                    }
                ],
            }
            archive_buffer = io.BytesIO()
            with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
                payload_info = tarfile.TarInfo(relative)
                payload_info.size = len(replacement)
                archive.addfile(payload_info, io.BytesIO(replacement))
                raw_manifest = json.dumps(manifest).encode("utf-8")
                manifest_info = tarfile.TarInfo("snapshot_manifest.json")
                manifest_info.size = len(raw_manifest)
                archive.addfile(manifest_info, io.BytesIO(raw_manifest))

            pointer = root / "data/lol/warehouse_snapshot.json"
            pointer.write_text(
                json.dumps({"url": "https://example.invalid/snapshot.tar.gz"}),
                encoding="utf-8",
            )
            old_root = snapshot.ROOT
            try:
                snapshot.ROOT = root
                with patch.object(
                    snapshot.urllib.request,
                    "urlopen",
                    return_value=io.BytesIO(archive_buffer.getvalue()),
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "checksum mismatch",
                    ):
                        snapshot.restore_snapshot(pointer)
            finally:
                snapshot.ROOT = old_root

            self.assertEqual(live_path.read_bytes(), b"known-good-live-warehouse")

    def test_fallback_pack_id_rejects_traversal_before_path_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "Unsafe pack_id"):
                snapshot._latest_pack(Path(temp), "../outside")


if __name__ == "__main__":
    unittest.main()
