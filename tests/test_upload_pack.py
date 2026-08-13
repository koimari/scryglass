from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lol_kills import update_public_pack
from lol_kills.export import upload_pack


class UploadPackTests(unittest.TestCase):
    def test_public_pack_blob_paths_fail_before_any_network_request(self) -> None:
        paths = (
            "packs/v2026.08.13.120000/features/draft_records.json",
            "packs/v2026.08.13.120000/features/ratings_snapshot.json",
            "packs/manifest.json",
            "%70acks/v2026.08.13.120000/features/profile_records.json",
            "snapshot/../packs/v2026.08.13.120000/features/team_records.json",
        )
        with patch.object(upload_pack.urllib.request, "urlopen") as request:
            for pathname in paths:
                with self.subTest(pathname=pathname), self.assertRaisesRegex(
                    RuntimeError,
                    "public pack Blob publication is disabled",
                ):
                    upload_pack._blob_put(
                        "token",
                        pathname,
                        b"{}",
                        "application/json",
                    )
        request.assert_not_called()

    def test_all_legacy_pack_publish_entrypoints_fail_closed(self) -> None:
        manifest = {"pack_id": "v2026.08.13.120000"}
        entrypoints = (
            lambda: upload_pack.upload_to_blob(
                Path("candidate"), manifest["pack_id"], "token"
            ),
            lambda: upload_pack.publish_blob_pointers(
                "token",
                manifest["pack_id"],
                manifest,
                base_url="https://example.public.blob.vercel-storage.com/packs/x",
            ),
            lambda: upload_pack.write_atlas_manifest(
                manifest["pack_id"],
                manifest,
                base_url=f"/packs/{manifest['pack_id']}",
            ),
            lambda: upload_pack.main([]),
        )
        for entrypoint in entrypoints:
            with self.subTest(entrypoint=entrypoint), self.assertRaisesRegex(
                (RuntimeError, SystemExit),
                "public pack Blob publication is disabled",
            ):
                entrypoint()

    def test_update_command_rejects_legacy_publication_flags_before_work(self) -> None:
        with patch.object(update_public_pack, "_run_module") as run:
            for flag in ("--publish", "--local-only"):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    update_public_pack.main([flag])
        run.assert_not_called()

    def test_public_pack_callers_have_no_blob_token_or_pointer_lane(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for relative in (
            "lol_kills/postgame_sync.py",
            "lol_kills/update_public_pack.py",
        ):
            source = (root / relative).read_text(encoding="utf-8")
            with self.subTest(relative=relative):
                self.assertNotIn("BLOB_READ_WRITE_TOKEN", source)
                self.assertNotIn("VERCEL_BLOB_READ_WRITE_TOKEN", source)
                self.assertNotIn("publish_blob_pointers", source)
                self.assertNotIn("upload_to_blob", source)

    def test_pack_directory_rejects_symlinks_and_noncanonical_ids(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack = root / "v2026.08.13.120000"
            pack.mkdir()
            self.assertEqual(upload_pack._pack_directory(root, pack.name), pack.resolve())
            with self.assertRaisesRegex(RuntimeError, "pack ID is invalid"):
                upload_pack._pack_directory(root, "../escape")
            alias = root / "v2026.08.13.120001"
            alias.symlink_to(pack, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, "pack directory is invalid"):
                upload_pack._pack_directory(root, alias.name)


if __name__ == "__main__":
    unittest.main()
