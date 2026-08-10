from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lol_kills.export import supabase_publication


class FakeSupabase:
    project_url = "https://example.supabase.co"

    def __init__(self) -> None:
        self.releases: dict[str, dict[str, object]] = {}
        self.assets: dict[tuple[str, str], dict[str, object]] = {}
        self.active_id: str | None = None
        self.stage_calls = 0
        self.storage: dict[str, bytes] = {}

    def release(self, release_id: str):
        return self.releases.get(release_id)

    def active_release(self):
        return self.releases.get(self.active_id or "")

    def asset(self, release_id: str, path: str):
        return self.assets.get((release_id, path))

    def storage_object(self, storage_path: str):
        return self.storage[storage_path]

    def stage_release(self, release, assets, *, storage_objects=None):
        self.stage_calls += 1
        self.releases[release["release_id"]] = dict(release)
        for asset in assets:
            self.assets[(asset["release_id"], asset["path"])] = dict(asset)
        self.storage.update(storage_objects or {})
        return 0

    def activate(self, release_id: str):
        previous = self.active_id
        if previous:
            self.releases[previous]["status"] = "superseded"
        self.active_id = release_id
        self.releases[release_id]["status"] = "active"
        return {
            "status": "active",
            "release_id": release_id,
            "previous_release_id": previous,
        }

    def prune(self, keep: int = 3):
        assert keep == 3
        return 0


def _fixture(root: Path):
    release_id = "v2026.08.10.153000"
    pack = root / release_id
    files = []
    for index, path in enumerate(supabase_publication.PUBLIC_RATING_REQUIRED_FILES):
        destination = pack / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps({"path": path, "index": index}, separators=(",", ":")).encode()
        destination.write_bytes(raw)
        files.append(
            {
                "path": path,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    manifest = {
        "pack_id": release_id,
        "schema_version": "test",
        "base_url": None,
        "created_utc": "2026-08-10T15:30:00Z",
        "ratings": {"source_as_of": "2026-08-10T15:00:00Z"},
        "files": files,
    }
    tier = root / "tierlists.json"
    tier.write_text(json.dumps({"status": "available", "rows": []}), encoding="utf-8")
    return pack, manifest, tier


def test_publish_release_stages_then_activates_complete_snapshot() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        client = FakeSupabase()

        result = supabase_publication.publish_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
            secret_key="sb_secret_unused_because_client_is_injected",
            client=client,
        )

    assert result["status"] == "published"
    assert result["release_id"] == manifest["pack_id"]
    assert result["assets"] == 10
    assert result["reused_assets"] == 0
    assert client.active_id == manifest["pack_id"]
    assert len(client.assets) == 10
    assert list(client.storage) == [
        "v2026.08.10.153000/rankings/tierlists.json"
    ]
    assert client.releases[manifest["pack_id"]]["manifest"]["data_backend"] == "supabase"


def test_publish_release_is_idempotent_after_verified_activation() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        client = FakeSupabase()
        first = supabase_publication.publish_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
            secret_key="sb_secret_unused_because_client_is_injected",
            client=client,
        )
        second = supabase_publication.publish_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
            secret_key="sb_secret_unused_because_client_is_injected",
            client=client,
        )

    assert first["status"] == "published"
    assert second["status"] == "already_active"
    assert client.stage_calls == 1


def test_publish_release_rejects_changed_pack_asset_before_upload() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        changed = pack / supabase_publication.PUBLIC_RATING_REQUIRED_FILES[0]
        changed.write_text("{}", encoding="utf-8")

        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="checksum failed",
        ):
            supabase_publication.publish_release(
                pack,
                manifest,
                tier,
                project_url="https://example.supabase.co",
                secret_key="sb_secret_unused_because_client_is_injected",
                client=FakeSupabase(),
            )


def test_client_repr_redacts_secret_key() -> None:
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    assert "abcdefghijklmnopqrstuvwxyz" not in repr(client)
    assert "<redacted>" in repr(client)
