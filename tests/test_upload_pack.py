import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lol_kills.export import upload_pack


def _release_report(*, blocker: int = 0, major: int = 0) -> dict:
    return {
        "counts": {
            "launch blocker": blocker,
            "major": major,
            "minor": 0,
            "informational": 0,
        },
        "release_gate": {
            "ready": blocker == 0 and major == 0,
            "blocking_severities": ["launch blocker", "major"],
            "blocking_findings": blocker + major,
        },
    }


def _write_pack(root: Path, pack_id: str = "v2026.07.27.1200") -> Path:
    pack = root / pack_id
    data_path = pack / "models" / "model.json"
    data_path.parent.mkdir(parents=True)
    data = b'{"model":"safe"}\n'
    data_path.write_bytes(data)
    (pack / "README.md").write_text("local export note\n", encoding="utf-8")
    manifest = {
        "pack_id": pack_id,
        "schema_version": "1.4.0",
        "created_utc": "2026-07-27T12:00:00+00:00",
        "total_files": 1,
        "total_bytes": len(data),
        "files": [
            {
                "path": "models/model.json",
                "relative": "models/model.json",
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        ],
    }
    (pack / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return pack


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
            side_effect=lambda _token, pathname, *_args, **_kwargs: (
                f"https://blob/{pathname}"
            ),
        ) as put:
            urls = upload_pack.publish_blob_pointers(
                "token",
                manifest["pack_id"],
                manifest,
                base_url="https://blob/packs/v2026.07.26.1700",
                release_report=_release_report(),
            )

        self.assertEqual(set(urls), {"packs/manifest.json", "packs/latest.json"})
        self.assertEqual(put.call_count, 2)
        for call in put.call_args_list:
            self.assertTrue(call.kwargs["allow_overwrite"])
            self.assertEqual(
                call.kwargs["cache_control"],
                "public, max-age=60, must-revalidate",
            )

        manifest_call = next(
            call
            for call in put.call_args_list
            if call.args[1] == "packs/manifest.json"
        )
        manifest_payload = json.loads(manifest_call.args[2])
        self.assertEqual(
            manifest_payload["base_url"],
            "https://blob/packs/v2026.07.26.1700",
        )

    def test_major_release_finding_prevents_pointer_advancement(self) -> None:
        manifest = {
            "pack_id": "v2026.07.26.1700",
            "created_utc": "2026-07-26T17:00:00+00:00",
        }
        with patch.object(upload_pack, "_blob_put") as put:
            with self.assertRaisesRegex(RuntimeError, "major=1"):
                upload_pack.publish_blob_pointers(
                    "token",
                    manifest["pack_id"],
                    manifest,
                    base_url="https://blob/packs/v2026.07.26.1700",
                    release_report=_release_report(major=1),
                )
        put.assert_not_called()

    def test_manifest_allowlist_rejects_extra_and_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = _write_pack(root)
            (pack / "secret.txt").write_text("must not publish", encoding="utf-8")
            with patch.object(upload_pack, "_blob_put") as put:
                with self.assertRaisesRegex(RuntimeError, "undeclared files"):
                    upload_pack.upload_to_blob(pack, pack.name, "token")
            put.assert_not_called()

            (pack / "secret.txt").unlink()
            (pack / "models" / "model.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing or unsafe declared file"):
                upload_pack.validate_pack(pack, pack.name)

    def test_manifest_rejects_size_and_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            size_pack = _write_pack(root, "vsize")
            size_manifest_path = size_pack / "manifest.json"
            size_manifest = json.loads(size_manifest_path.read_text(encoding="utf-8"))
            size_manifest["files"][0]["bytes"] += 1
            size_manifest["total_bytes"] += 1
            size_manifest_path.write_text(json.dumps(size_manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "byte size mismatch"):
                upload_pack.validate_pack(size_pack, "vsize")

            hash_pack = _write_pack(root, "vhash")
            hash_manifest_path = hash_pack / "manifest.json"
            hash_manifest = json.loads(hash_manifest_path.read_text(encoding="utf-8"))
            hash_manifest["files"][0]["sha256"] = "0" * 64
            hash_manifest_path.write_text(json.dumps(hash_manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                upload_pack.validate_pack(hash_pack, "vhash")

    def test_manifest_rejects_traversal_path_and_pack_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = _write_pack(root)
            manifest_path = pack / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["files"][0]["path"] = "../outside.json"
            manifest["files"][0]["relative"] = "../outside.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Unsafe manifest file path"):
                upload_pack.validate_pack(pack, pack.name)

        for unsafe in ("../pack", "nested/pack", "/absolute", ".", "..", ""):
            with self.subTest(pack_id=unsafe):
                with self.assertRaisesRegex(ValueError, "Unsafe pack_id"):
                    upload_pack.validate_pack_id(unsafe)

    def test_blob_upload_uses_only_declared_files_plus_manifest_control(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pack = _write_pack(Path(temp))
            with patch.object(
                upload_pack,
                "audit_pack",
                return_value=_release_report(),
            ), patch.object(
                upload_pack,
                "_blob_put",
                side_effect=lambda _token, pathname, *_args, **_kwargs: (
                    f"https://blob/{pathname}"
                ),
            ) as put:
                urls = upload_pack.upload_to_blob(pack, pack.name, "token")

        self.assertEqual(
            set(urls),
            {"models/model.json", "manifest.json"},
        )
        self.assertEqual(
            [call.args[1] for call in put.call_args_list],
            [
                f"packs/{pack.name}/models/model.json",
                f"packs/{pack.name}/manifest.json",
            ],
        )

    def test_release_audit_failure_prevents_all_publication_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack = _write_pack(root)
            atlas = root / "atlas"
            with patch.dict(
                os.environ,
                {"BLOB_READ_WRITE_TOKEN": "test-token"},
                clear=False,
            ), patch.object(
                upload_pack,
                "ATLAS_PUBLIC",
                atlas,
            ), patch.object(
                upload_pack,
                "_blob_put",
            ) as put:
                with self.assertRaisesRegex(RuntimeError, "failed release gate"):
                    upload_pack.main(
                        ["--pack-root", str(root), "--pack-id", pack.name]
                    )

            put.assert_not_called()
            self.assertFalse(atlas.exists())


if __name__ == "__main__":
    unittest.main()
