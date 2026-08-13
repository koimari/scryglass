from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from lol_kills.export import upload_pack


class UploadPackTests(unittest.TestCase):
    def test_vercel_ignore_keeps_only_older_pack_directories(self) -> None:
        with TemporaryDirectory() as temporary:
            app_root = Path(temporary) / "app"
            public_root = app_root / "public/packs"
            public_root.mkdir(parents=True)
            for pack_id in ("v1", "v2"):
                (public_root / pack_id).mkdir()
            ignore = app_root / ".vercelignore"
            ignore.write_text(
                "custom-private-path/\npublic/packs/*\n!public/packs/manifest.json\n",
                encoding="utf-8",
            )

            upload_pack.write_vercel_pack_ignore(
                "v2",
                public_root=public_root,
                ignore_path=ignore,
            )

            self.assertEqual(
                ignore.read_text(encoding="utf-8").splitlines(),
                ["custom-private-path/", "public/packs/v1/"],
            )

    def test_publish_blob_pointers_uses_stable_overwritable_paths(self) -> None:
        manifest = {
            "pack_id": "v2026.07.26.170000",
            "created_utc": "2026-07-26T17:00:00+00:00",
            "base_url": None,
        }

        objects: dict[str, bytes] = {}

        def fake_get(url: str) -> bytes | None:
            return objects.get(url)

        def fake_put(_token, pathname, data, *_args, **_kwargs):
            objects[f"https://blob/{pathname}"] = data
            return f"https://blob/{pathname}"

        with patch.object(upload_pack, "_blob_get", side_effect=fake_get), patch.object(
            upload_pack,
            "_blob_put",
            side_effect=fake_put,
        ) as put:
            urls = upload_pack.publish_blob_pointers(
                "token",
                manifest["pack_id"],
                manifest,
                base_url="https://blob/packs/v2026.07.26.170000",
            )

        self.assertEqual(set(urls), {"packs/manifest.json", "packs/latest.json"})
        self.assertEqual(put.call_count, 2)
        for call in put.call_args_list:
            self.assertTrue(call.kwargs["allow_overwrite"])
            self.assertEqual(
                call.kwargs["cache_control"],
                "public, max-age=60, must-revalidate",
            )

        manifest_payload = json.loads(put.call_args_list[0].args[2])
        self.assertEqual(
            manifest_payload["base_url"],
            "https://blob/packs/v2026.07.26.170000",
        )

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
