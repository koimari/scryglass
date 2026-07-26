import json
import unittest
from unittest.mock import patch

from lol_kills.export import upload_pack


class UploadPackTests(unittest.TestCase):
    def test_publish_blob_pointers_uses_stable_overwritable_paths(self) -> None:
        manifest = {
            "pack_id": "v2026.07.26.1700",
            "created_utc": "2026-07-26T17:00:00+00:00",
            "base_url": None,
        }

        with patch.object(
            upload_pack,
            "_blob_put",
            side_effect=lambda _token, pathname, *_args, **_kwargs: f"https://blob/{pathname}",
        ) as put:
            urls = upload_pack.publish_blob_pointers(
                "token",
                manifest["pack_id"],
                manifest,
                base_url="https://blob/packs/v2026.07.26.1700",
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
            "https://blob/packs/v2026.07.26.1700",
        )


if __name__ == "__main__":
    unittest.main()
