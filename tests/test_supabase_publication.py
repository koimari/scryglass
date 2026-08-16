from __future__ import annotations

import base64
import hashlib
import json
import urllib.error
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from lol_kills.export import supabase_publication


ROOT = Path(__file__).resolve().parents[1]


class FakeSupabase:
    project_url = "https://example.supabase.co"

    def __init__(self) -> None:
        self.releases: dict[str, dict[str, object]] = {}
        self.assets: dict[tuple[str, str], dict[str, object]] = {}
        self.active_id: str | None = None
        self.stage_calls = 0
        self.storage: dict[str, bytes] = {}
        self.discard_calls: list[str] = []
        self.stale_discard_limits: list[int] = []
        self.drain_calls: list[str] = []

    def release(self, release_id: str):
        return self.releases.get(release_id)

    def active_release(self):
        return self.releases.get(self.active_id or "")

    def asset(self, release_id: str, path: str):
        return self.assets.get((release_id, path))

    def storage_object(self, storage_path: str):
        return self.storage[storage_path]

    def storage_object_metadata(self, storage_path: str):
        raw = self.storage.get(storage_path)
        if raw is None:
            return {}
        return {
            "size": len(raw),
            "etag": "fake",
            "metadata": {
                "sha256": hashlib.sha256(raw).hexdigest(),
                "bytes": len(raw),
                "content_type": "application/json",
            },
        }

    def create_release(self, release):
        self.stage_calls += 1
        self.releases[release["release_id"]] = dict(release)

    def stage_release(self, release, assets, *, storage_objects=None):
        self.create_release(release)
        return self.stage_assets(
            str(release["release_id"]),
            assets,
            storage_objects=storage_objects,
        )

    def drain_staging_cleanup(self, release_id: str) -> int:
        self.drain_calls.append(release_id)
        return 0

    def stage_assets(self, release_id, assets, *, storage_objects=None):
        reused = 0
        for asset in assets:
            key = (asset["release_id"], asset["path"])
            prior = self.assets.get(key)
            if prior and prior.get("bytes") == asset["bytes"] and prior.get("sha256") == asset["sha256"]:
                reused += 1
                continue
            if prior:
                raise supabase_publication.SupabasePublicationError(
                    f"existing public asset has different content: {asset['path']}"
                )
            self.assets[(asset["release_id"], asset["path"])] = dict(asset)
        for path, raw in (storage_objects or {}).items():
            prior = self.storage.get(path)
            if prior is not None and prior != raw:
                raise supabase_publication.SupabasePublicationError(
                    "existing Supabase Storage object has different content"
                )
            self.storage[path] = raw
        return reused

    def discard_staging_release(self, release_id: str) -> int:
        self.discard_calls.append(release_id)
        release = self.releases.get(release_id)
        if release is None:
            return 0
        if release.get("status") != "staging":
            raise supabase_publication.SupabasePublicationError(
                "Only a staging release can be discarded"
            )
        paths = [
            path
            for (stored_release, path) in self.assets
            if stored_release == release_id
        ]
        for key in list(self.assets):
            if key[0] == release_id:
                del self.assets[key]
        for path in list(self.storage):
            if path.startswith(f"{release_id}/"):
                del self.storage[path]
        del self.releases[release_id]
        return len(paths)

    def discard_stale_staging_releases(
        self, *, min_age_minutes: int = 360, limit: int = 10
    ) -> int:
        del min_age_minutes
        self.stale_discard_limits.append(limit)
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

    def restore(self, release_id: str):
        previous = self.active_id
        if previous:
            self.releases[previous]["status"] = "superseded"
        self.active_id = release_id
        self.releases[release_id]["status"] = "active"
        return {
            "status": "restored",
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
        "total_files": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
        "release": {
            "release_id": release_id,
            "artifact_hashes": {
                item["path"]: item["sha256"] for item in files
            },
        },
    }
    tier = root / "tierlists.json"
    tier.write_text(
        json.dumps(
            {
                "status": "available",
                "options": {"patches": ["16.14", "16.15"]},
                "rows": [
                    {"patch": "16.14", "champion": "Old"},
                    {"patch": "16.15", "champion": "Current"},
                ],
                "scopes": [
                    {"patch": "16.14", "role": "mid"},
                    {"patch": "16.15", "role": "mid"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return pack, manifest, tier


def _add_manifest_asset(
    pack: Path,
    manifest: dict[str, object],
    path: str,
    payload: object,
) -> bytes:
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    destination = pack / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(raw)
    file_entry = {
        "path": path,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    files = manifest["files"]
    assert isinstance(files, list)
    files.append(file_entry)
    manifest["total_files"] = len(files)
    manifest["total_bytes"] = sum(int(item["bytes"]) for item in files)
    release = manifest["release"]
    assert isinstance(release, dict)
    artifact_hashes = release["artifact_hashes"]
    assert isinstance(artifact_hashes, dict)
    artifact_hashes[path] = file_entry["sha256"]
    return raw


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

    assert result["status"] == "activated_pending_health"
    assert result["release_id"] == manifest["pack_id"]
    assert result["assets"] == len(supabase_publication.PUBLIC_RATING_REQUIRED_FILES) + 2
    assert result["reused_assets"] == 0
    assert client.stale_discard_limits == [1]
    assert client.active_id == manifest["pack_id"]
    assert len(client.assets) == len(supabase_publication.PUBLIC_RATING_REQUIRED_FILES) + 2
    assert list(client.storage) == [
        *[
            f"v2026.08.10.153000/{path}"
            for path in supabase_publication.PUBLIC_RATING_REQUIRED_FILES
        ],
        "v2026.08.10.153000/rankings/tierlists.json",
        "v2026.08.10.153000/rankings/tierlists-latest.json",
    ]
    assert client.releases[manifest["pack_id"]]["manifest"]["data_backend"] == "supabase"
    assert client.releases[manifest["pack_id"]]["manifest"]["tier"] == {
        "status": "available",
        "as_of": None,
        "latest_path": "rankings/tierlists-latest.json",
    }
    published_manifest = client.releases[manifest["pack_id"]]["manifest"]
    assert published_manifest["total_files"] == result["assets"]
    assert {item["path"] for item in published_manifest["files"]} == {
        path for _, path in client.assets
    }
    assert published_manifest["release"]["artifact_hashes"] == {
        path: asset["sha256"] for (_, path), asset in client.assets.items()
    }
    assert published_manifest["draft_authority"] == {
        "schema_version": "scryglass:draft-authority:v1",
        "status": "unavailable",
        "release_id": manifest["pack_id"],
        "model_version": None,
        "artifact_sha256": None,
        "receipt_sha256": None,
        "issued_utc": None,
        "reason": "model_not_promoted",
    }


def test_publish_release_omits_unpromoted_draft_asset() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        manifest["draft_authority"] = {
            "schema_version": "scryglass:draft-authority:v1",
            "status": "unavailable",
            "release_id": manifest["pack_id"],
            "model_version": None,
            "receipt_sha256": None,
            "issued_utc": None,
            "reason": "model_not_promoted",
        }
        _add_manifest_asset(
            pack,
            manifest,
            supabase_publication.DRAFT_ASSET_PATH,
            {
                "schema_version": "scryglass:draft-records:v1",
                "model_version": "development-only",
                "games": {"game-1": {"draft_edge": 0.5}},
            },
        )
        client = FakeSupabase()

        supabase_publication.publish_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
            secret_key="sb_secret_unused_because_client_is_injected",
            client=client,
        )

    published = client.releases[manifest["pack_id"]]["manifest"]
    assert isinstance(published, dict)
    assert (manifest["pack_id"], supabase_publication.DRAFT_ASSET_PATH) not in client.assets
    assert supabase_publication.DRAFT_ASSET_PATH not in {
        item["path"] for item in published["files"]
    }
    assert supabase_publication.DRAFT_ASSET_PATH not in published["release"]["artifact_hashes"]
    assert published["draft_authority"]["status"] == "unavailable"
    assert published["draft_authority"]["reason"] == "model_not_promoted"


def test_descriptive_draft_asset_rejects_r9e_and_strength_fields() -> None:
    payload = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "model_version": "draft-recommendation-static-v2",
        "games": {"game-1": {"draft_edge": 0.1, "r9e": {"score": 0.2}}},
    }
    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="contains predictive fields",
    ):
        supabase_publication._validate_descriptive_draft_records(payload)


@pytest.mark.parametrize("issued_utc", ["2026-99-99T15:30:00Z", "2026-08-10T15:30:00+00:00"])
def test_descriptive_draft_authority_rejects_invalid_utc_timestamp(issued_utc: str) -> None:
    manifest = {
        "draft_authority": {
            "schema_version": "scryglass:draft-authority:v1",
            "status": "descriptive",
            "authority": "descriptive",
            "release_id": "v2026.08.10.153000",
            "model_version": "draft-recommendation-static-v2",
            "artifact_sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "issued_utc": issued_utc,
            "estimand": "composition_only",
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
            "reason": None,
        }
    }
    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="authority is invalid",
    ):
        supabase_publication._draft_authority(manifest, "v2026.08.10.153000")


def test_descriptive_authority_migration_binds_timestamp_and_model_length() -> None:
    migration = (
        ROOT / "supabase" / "migrations" / "20260815060000_descriptive_draft_authority.sql"
    ).read_text(encoding="utf-8")
    assert "octet_length(" in migration
    assert "draft_authority,model_version" in migration
    assert "draft_authority,issued_utc" in migration
    assert "::timestamptz" in migration
    assert "([.][0-9]{1,6})?Z$" in migration


def test_descriptive_draft_asset_rejects_zero_usable_games() -> None:
    payload = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "model_version": "draft-recommendation-static-v2",
        "games": {},
    }
    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="no usable games",
    ):
        supabase_publication._validate_descriptive_draft_records(payload)


def test_descriptive_draft_asset_requires_complete_pool_and_ten_valid_picks() -> None:
    picks = [
        {
            "side": "Blue" if index <= 5 else "Red",
            "role": ("top", "jungle", "mid", "bot", "support")[(index - 1) % 5],
            "champion": f"Champion {index}",
            "order": index,
            "best_available": True,
            "tier_rank": index,
            "available_count": 10,
        }
        for index in range(1, 11)
    ]
    payload = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "model_version": "draft-recommendation-static-v2",
        "games": {
            "game-1": {
                "draft_edge": 0.1,
                "draft_pool": {
                    "status": "complete",
                    "patch": "26.16",
                    "bans": {
                        "Blue": [f"Blue ban {index}" for index in range(1, 6)],
                        "Red": [f"Red ban {index}" for index in range(1, 6)],
                    },
                    "picked": picks,
                    "evaluated_picks": 10,
                },
            }
        },
    }
    supabase_publication._validate_descriptive_draft_records(payload)

    payload["games"]["game-1"]["draft_pool"]["evaluated_picks"] = 9
    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="pool evaluation is incomplete",
    ):
        supabase_publication._validate_descriptive_draft_records(payload)

    payload["games"]["game-1"] = {"draft_edge": 0.1, "mu_diff": 14}
    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="contains predictive fields",
    ):
        supabase_publication._validate_descriptive_draft_records(payload)


def test_publish_release_rejects_promoted_draft_until_independent_verification() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        model_version = "draft-promoted-v1"
        _add_manifest_asset(
            pack,
            manifest,
            supabase_publication.DRAFT_ASSET_PATH,
            {
                "schema_version": "scryglass:draft-records:v1",
                "model_version": model_version,
                "games": {"game-1": {"draft_edge": 0.5}},
            },
        )
        manifest["draft_authority"] = {
            "schema_version": "scryglass:draft-authority:v1",
            "status": "promoted",
            "release_id": manifest["pack_id"],
            "model_version": model_version,
            "receipt_sha256": "a" * 64,
            "issued_utc": "2026-08-10T15:29:00Z",
        }
        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="independent receipt verifier",
        ):
            supabase_publication.prepare_release(
                pack,
                manifest,
                tier,
                project_url="https://example.supabase.co",
            )


def test_publish_release_rejects_unbound_promoted_draft_authority() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        manifest["draft_authority"] = {
            "schema_version": "scryglass:draft-authority:v1",
            "status": "promoted",
            "release_id": "v2026.08.10.153001",
            "model_version": "draft-promoted-v1",
            "receipt_sha256": "a" * 64,
        }

        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="independent receipt verifier",
        ):
            supabase_publication.prepare_release(
                pack,
                manifest,
                tier,
                project_url="https://example.supabase.co",
            )


def test_publish_release_includes_optional_schedule_when_present() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        path = "features/schedule.json"
        raw = json.dumps({"schema_version": "scryglass:public-schedule:v1", "upcoming": []}).encode()
        destination = pack / path
        destination.write_bytes(raw)
        manifest["files"].append(
            {
                "path": path,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
        manifest["total_files"] = len(manifest["files"])
        manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"])
        manifest["release"]["artifact_hashes"][path] = hashlib.sha256(raw).hexdigest()
        client = FakeSupabase()

        result = supabase_publication.publish_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
            secret_key="sb_secret_unused_because_client_is_injected",
            client=client,
        )

    assert result["assets"] == len(supabase_publication.PUBLIC_RATING_REQUIRED_FILES) + 3
    assert (manifest["pack_id"], path) in client.assets


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

    assert first["status"] == "activated_pending_health"
    assert second["status"] == "already_active"
    assert second["reused_assets"] == second["assets"]
    assert client.stage_calls == 1


def test_publish_release_preserves_another_worker_staging_snapshot() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        client = FakeSupabase()
        prepared_release, _, _ = supabase_publication.prepare_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
        )
        client.releases[manifest["pack_id"]] = {
            "release_id": manifest["pack_id"],
            "status": "staging",
            "manifest": prepared_release["manifest"],
        }
        client.assets[(manifest["pack_id"], "stale.json")] = {
            "release_id": manifest["pack_id"],
            "path": "stale.json",
            "bytes": 1,
            "sha256": "0" * 64,
            "content_type": "application/json",
        }
        client.storage[f"{manifest['pack_id']}/stale.json"] = b"x"

        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="already being staged",
        ):
            supabase_publication.publish_release(
                pack,
                manifest,
                tier,
                project_url=client.project_url,
                secret_key="sb_secret_unused_because_client_is_injected",
                client=client,
            )

    assert client.discard_calls == []
    assert (manifest["pack_id"], "stale.json") in client.assets


def test_publish_release_adds_latest_view_to_existing_active_release() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        client = FakeSupabase()
        supabase_publication.publish_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
            secret_key="sb_secret_unused_because_client_is_injected",
            client=client,
        )
        latest_key = (manifest["pack_id"], supabase_publication.TIER_LATEST_ASSET_PATH)
        latest_storage = f"{manifest['pack_id']}/{supabase_publication.TIER_LATEST_ASSET_PATH}"
        del client.assets[latest_key]
        del client.storage[latest_storage]

        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="readback failed",
        ):
            supabase_publication.publish_release(
                pack,
                manifest,
                tier,
                project_url=client.project_url,
                secret_key="sb_secret_unused_because_client_is_injected",
                client=client,
            )

    assert latest_key not in client.assets
    assert latest_storage not in client.storage
    assert client.active_id == manifest["pack_id"]


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


def test_prepare_release_rejects_conflicting_release_identity_and_hashes() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        manifest["release"]["release_id"] = "v2026.08.10.153001"
        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="release ID does not match",
        ):
            supabase_publication.prepare_release(
                pack,
                manifest,
                tier,
                project_url="https://example.supabase.co",
            )

        manifest["release"]["release_id"] = manifest["pack_id"]
        path = supabase_publication.PUBLIC_RATING_REQUIRED_FILES[0]
        manifest["release"]["artifact_hashes"][path] = "0" * 64
        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="release hash conflicts",
        ):
            supabase_publication.prepare_release(
                pack,
                manifest,
                tier,
                project_url="https://example.supabase.co",
            )


def test_storage_content_digest_is_verified_before_activation() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        database = FakeSupabase()
        original = database.stage_assets

        def stage_corrupt(release_id, assets, *, storage_objects=None):
            reused = original(release_id, assets, storage_objects=storage_objects)
            path = f"{manifest['pack_id']}/{supabase_publication.PUBLIC_RATING_REQUIRED_FILES[0]}"
            database.storage[path] = b"{}"
            return reused

        database.stage_assets = stage_corrupt  # type: ignore[method-assign]
        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="Storage readback failed|Storage checksum failed",
        ):
            supabase_publication.publish_release(
                pack,
                manifest,
                tier,
                project_url=database.project_url,
                secret_key="unused",
                client=database,
            )
        assert database.active_id is None


def test_storage_readback_failure_does_not_activate_staged_release() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        database = FakeSupabase()
        old_release = "v2026.08.09.120000"
        database.active_id = old_release
        database.releases[old_release] = {
            "release_id": old_release,
            "status": "active",
        }
        original = database.storage_object_metadata

        def corrupt(storage_path: str):
            if storage_path.endswith("rankings/tierlists.json"):
                return {"size": -1, "metadata": {}}
            return original(storage_path)

        database.storage_object_metadata = corrupt  # type: ignore[method-assign]

        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="Storage readback failed",
        ):
            supabase_publication.publish_release(
                pack,
                manifest,
                tier,
                project_url=database.project_url,
                secret_key="unused",
                client=database,
            )

        assert database.active_id == old_release


def test_post_activation_readback_failure_restores_previous_release() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
        database = FakeSupabase()
        old_release = "v2026.08.09.120000"
        database.active_id = old_release
        database.releases[old_release] = {
            "release_id": old_release,
            "status": "active",
        }
        original = database.storage_object_metadata
        reads = 0

        def fail_after_activation(storage_path: str):
            nonlocal reads
            reads += 1
            if (
                reads > len(supabase_publication.PUBLIC_RATING_REQUIRED_FILES) + 2
                and storage_path.endswith("rankings/tierlists.json")
            ):
                return {"size": -1, "metadata": {}}
            return original(storage_path)

        database.storage_object_metadata = fail_after_activation  # type: ignore[method-assign]

        with pytest.raises(
            supabase_publication.SupabasePublicationError,
            match="Storage readback failed",
        ):
            supabase_publication.publish_release(
                pack,
                manifest,
                tier,
                project_url=database.project_url,
                secret_key="unused",
                client=database,
            )

        assert database.active_id == old_release
        assert database.releases[old_release]["status"] == "active"


def test_restore_rechecks_storage_and_recovers_previous_release(monkeypatch) -> None:
    target = "v2026.08.13.183000"
    previous = "v2026.08.12.221135"
    path = "features/schedule.json"
    sha256 = "a" * 64
    active = {"release_id": previous}

    def manifest(release_id: str) -> dict[str, object]:
        return {
            "pack_id": release_id,
            "draft_authority": {
                "schema_version": "scryglass:draft-authority:v1",
                "status": "unavailable",
                "release_id": release_id,
            },
            "files": [{"path": path, "bytes": 1, "sha256": sha256}],
            "release": {
                "release_id": release_id,
                "artifact_hashes": {path: sha256},
            },
        }

    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    client.release = lambda release_id: {
        "release_id": release_id,
        "status": "active" if active["release_id"] == release_id else "superseded",
        "manifest": manifest(release_id),
    }  # type: ignore[method-assign]
    client.asset_metadata = lambda _release_id: {path: {}}  # type: ignore[method-assign]
    client.query_receipts = lambda _release_id: {}  # type: ignore[method-assign]

    def restore_request(method, request_path, payload=None, **_kwargs):
        assert method == "POST"
        assert request_path == "rpc/restore_scryglass_public_release"
        restored = payload["p_release_id"]
        replaced = active["release_id"]
        active["release_id"] = restored
        return {
            "status": "restored",
            "release_id": restored,
            "replaced_release_id": replaced,
        }

    client._request = restore_request  # type: ignore[method-assign]
    verify_calls = []

    monkeypatch.setattr(
        supabase_publication,
        "_verify_release_assets",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        supabase_publication,
        "_verify_query_receipts",
        lambda *_args, **_kwargs: None,
    )

    def verify_active(_client, release_id, _assets, *, verify_storage_content):
        verify_calls.append((release_id, verify_storage_content))
        if release_id == target:
            raise supabase_publication.SupabasePublicationError(
                "Supabase Storage checksum changed after restore"
            )

    monkeypatch.setattr(
        supabase_publication,
        "_verify_active_release",
        verify_active,
    )

    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="checksum changed after restore",
    ):
        client.restore(target)

    assert active["release_id"] == previous
    assert verify_calls == [(target, True), (previous, True)]


def test_client_repr_redacts_secret_key() -> None:
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    assert "abcdefghijklmnopqrstuvwxyz" not in repr(client)
    assert "<redacted>" in repr(client)


def test_query_staging_retries_transient_idempotent_batches(monkeypatch) -> None:
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    attempts = 0
    sleeps: list[float] = []

    def request(_method, _path, _payload=None, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < supabase_publication.MAX_QUERY_STAGE_ATTEMPTS:
            try:
                raise TimeoutError("transient TLS timeout")
            except TimeoutError as cause:
                raise supabase_publication.SupabasePublicationError(
                    "Supabase request failed"
                ) from cause
        return 0

    client._request = request  # type: ignore[method-assign]
    monkeypatch.setattr(supabase_publication.time, "sleep", sleeps.append)

    assert client._stage_query_rows("v2026.08.15.120000", "players", []) == 0
    assert attempts == supabase_publication.MAX_QUERY_STAGE_ATTEMPTS
    assert sleeps == [0.5, 1.0, 2.0]


def test_query_staging_retries_gateway_520(monkeypatch) -> None:
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    attempts = 0

    def request(_method, _path, _payload=None, **_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            cause = supabase_publication.urllib.error.HTTPError(
                "https://example.supabase.co/rest/v1/rpc/stage_scryglass_query_rows",
                520,
                "gateway error",
                None,
                None,
            )
            raise supabase_publication.SupabasePublicationError(
                "Supabase request failed with HTTP 520"
            ) from cause
        return 0

    client._request = request  # type: ignore[method-assign]
    monkeypatch.setattr(supabase_publication.time, "sleep", lambda _seconds: None)

    assert client._stage_query_rows("v2026.08.15.120000", "players", []) == 0
    assert attempts == 2


def test_retention_prunes_one_release_per_database_call() -> None:
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    responses = iter(
        [
            {
                "deleted_count": 1,
                "has_more": True,
                "storage_paths": ["v2026.08.10.120000/a.json"],
            },
            {
                "deleted_count": 1,
                "has_more": False,
                "storage_paths": ["v2026.08.11.120000/b.json"],
            },
        ]
    )
    requests: list[tuple[str, str, object]] = []
    deleted: list[list[str]] = []
    acknowledged: list[list[str]] = []

    def request(method, path, payload=None, **_kwargs):
        requests.append((method, path, payload))
        return next(responses)

    def delete_storage_objects(paths):
        deleted.append(list(paths))

    def acknowledge_storage_cleanup(paths):
        acknowledged.append(list(paths))
        return len(paths)

    client._request = request  # type: ignore[method-assign]
    client.delete_storage_objects = delete_storage_objects  # type: ignore[method-assign]
    client.ack_storage_cleanup = acknowledge_storage_cleanup  # type: ignore[method-assign]

    assert client.prune(3) == 2
    assert requests == [
        ("POST", "rpc/prune_scryglass_public_releases_v2", {"p_keep": 3}),
        ("POST", "rpc/prune_scryglass_public_releases_v2", {"p_keep": 3}),
    ]
    assert deleted == [
        ["v2026.08.10.120000/a.json"],
        ["v2026.08.11.120000/b.json"],
    ]
    assert acknowledged == deleted


def test_retention_accepts_the_twentieth_final_deletion() -> None:
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    responses = iter(
        {
            "deleted_count": 1,
            "has_more": index < supabase_publication.MAX_RETENTION_PRUNE_CALLS - 1,
            "storage_paths": [],
        }
        for index in range(supabase_publication.MAX_RETENTION_PRUNE_CALLS)
    )
    requests: list[tuple[str, str, object]] = []

    def request(method, path, payload=None, **_kwargs):
        requests.append((method, path, payload))
        return next(responses)

    client._request = request  # type: ignore[method-assign]
    client.delete_storage_objects = lambda _paths: None  # type: ignore[method-assign]
    client.ack_storage_cleanup = lambda _paths: 0  # type: ignore[method-assign]

    assert client.prune(3) == supabase_publication.MAX_RETENTION_PRUNE_CALLS
    assert len(requests) == supabase_publication.MAX_RETENTION_PRUNE_CALLS


def test_rollback_requires_the_canonical_release_id() -> None:
    assert (
        supabase_publication._rollback_release_id("v2026.08.13.183000")
        == "v2026.08.13.183000"
    )
    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="release ID is invalid",
    ):
        supabase_publication._release_id("v2026.08.13.1830")
    with pytest.raises(
        supabase_publication.SupabasePublicationError,
        match="rollback release ID is invalid",
    ):
        supabase_publication._rollback_release_id("../active")


def test_storage_upload_sends_base64_custom_metadata() -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"{}"

    class Opener:
        request = None

        def open(self, request, *, timeout):
            assert timeout == 180.0
            self.request = request
            return Response()

    opener = Opener()
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
        opener=opener,
    )
    raw = b'{"ok":true}'

    client.put_storage_object(
        "v2026.08.10.153000/features/schedule.json",
        raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        content_type="application/json",
    )

    assert opener.request is not None
    headers = {key.lower(): value for key, value in opener.request.headers.items()}
    metadata = json.loads(base64.b64decode(headers["x-metadata"]))
    assert metadata == {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "content_type": "application/json",
    }
    assert headers["x-upsert"] == "false"


def test_large_storage_upload_uses_resumable_chunks(monkeypatch) -> None:
    raw = b"x" * (supabase_publication.RESUMABLE_UPLOAD_THRESHOLD_BYTES + 3)
    digest = hashlib.sha256(raw).hexdigest()
    requests = []

    class Response:
        def __init__(self, body=b"", headers=None):
            self.body = body
            self.headers = headers or {}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return self.body

    class Opener:
        failed_first_chunk = False

        def open(self, request, *, timeout):
            requests.append(request)
            if request.method == "POST":
                assert timeout == supabase_publication.REQUEST_TIMEOUT_SECONDS
                return Response(
                    headers={"Location": "/storage/v1/upload/resumable/upload-id"}
                )
            if request.method == "PATCH":
                assert timeout == supabase_publication.REQUEST_TIMEOUT_SECONDS
                offset = int(request.headers["Upload-offset"])
                if not self.failed_first_chunk:
                    self.failed_first_chunk = True
                    raise urllib.error.HTTPError(
                        request.full_url,
                        544,
                        "gateway timeout",
                        {},
                        None,
                    )
                return Response(
                    headers={"Upload-Offset": str(offset + len(request.data or b""))}
                )
            if request.method == "HEAD":
                return Response(
                    headers={
                        "Upload-Offset": str(
                            supabase_publication.RESUMABLE_UPLOAD_CHUNK_BYTES
                        )
                    }
                )
            if "/object/info/authenticated/" in request.full_url:
                return Response(
                    json.dumps(
                        {
                            "size": len(raw),
                            "metadata": {
                                "sha256": digest,
                                "bytes": len(raw),
                                "content_type": "application/json",
                            },
                        }
                    ).encode("utf-8")
                )
            if "/object/authenticated/" in request.full_url:
                return Response(raw)
            raise AssertionError(request.full_url)

    monkeypatch.setattr(supabase_publication.time, "sleep", lambda _delay: None)
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
        opener=Opener(),
    )

    client.put_storage_object(
        "v2026.08.10.153000/features/profile_records.json",
        raw,
        sha256=digest,
        content_type="application/json",
    )

    create = requests[0]
    assert create.full_url == (
        "https://example.storage.supabase.co/storage/v1/upload/resumable"
    )
    create_headers = {key.lower(): value for key, value in create.headers.items()}
    assert create_headers["tus-resumable"] == "1.0.0"
    assert create_headers["upload-length"] == str(len(raw))
    assert create_headers["x-upsert"] == "false"
    decoded_metadata = {
        key: base64.b64decode(value).decode("utf-8")
        for key, value in (
            item.split(" ", 1)
            for item in create_headers["upload-metadata"].split(",")
        )
    }
    assert decoded_metadata["bucketName"] == "scryglass-public"
    assert decoded_metadata["objectName"].endswith("/features/profile_records.json")
    assert json.loads(decoded_metadata["metadata"]) == {
        "sha256": digest,
        "bytes": len(raw),
        "content_type": "application/json",
    }
    assert [request.method for request in requests].count("PATCH") == 2
    assert [request.method for request in requests].count("HEAD") == 1


def test_latest_tier_payload_keeps_only_newest_patch_and_all_views() -> None:
    payload = {
        "status": "available",
        "options": {"patches": ["16.9", "16.10"]},
        "rows": [
            {"patch": "16.9", "champion": "Old"},
            {"patch": "16.10", "champion": "New"},
        ],
        "scopes": [
            {"patch": "16.9", "response_matrix": {"old": True}},
            {"patch": "16.10", "response_matrix": {"new": True}},
        ],
        "structural_similarity": {"champions": ["New"]},
    }

    latest = supabase_publication.latest_tier_payload(payload)

    assert latest["latest_patch"] == "26.10"
    assert latest["options"]["patches"] == ["26.09", "26.10"]
    assert latest["rows"] == [{"patch": "26.10", "champion": "New"}]
    assert latest["scopes"] == [{"patch": "26.10", "response_matrix": {"new": True}}]
    assert latest["structural_similarity"] == payload["structural_similarity"]


def test_latest_tier_payload_separates_role_scopes_with_shared_patch_id() -> None:
    payload = {
        "status": "available",
        "options": {"patches": ["26.14"]},
        "rows": [
            {"scope_id": "patch:26.14", "patch": "26.14", "role": "mid", "champion": "Galio"},
            {"scope_id": "patch:26.14", "patch": "26.14", "role": "top", "champion": "Galio"},
        ],
        "scopes": [
            {"scope_id": "patch:26.14", "patch": "26.14", "role": "mid"},
            {"scope_id": "patch:26.14", "patch": "26.14", "role": "top"},
        ],
    }

    latest = supabase_publication.latest_tier_payload(payload)

    assert {row["scope_id"] for row in latest["rows"]} == {"26.14-mid", "26.14-top"}
    assert {scope["scope_id"] for scope in latest["scopes"]} == {"26.14-mid", "26.14-top"}


def test_refresh_ledger_migration_keeps_private_rows_private() -> None:
    migration = (
        ROOT / "supabase/migrations/20260811174735_refresh_run_ledger.sql"
    ).read_text(encoding="utf-8")

    assert "create table public.scryglass_refresh_runs" in migration
    assert "create table public.scryglass_public_health" in migration
    assert "alter table public.scryglass_refresh_runs enable row level security" in migration
    assert "alter table public.scryglass_public_health enable row level security" in migration
    assert (
        "revoke all on public.scryglass_refresh_runs "
        "from public, anon, authenticated, service_role"
    ) in migration
    assert "grant select on public.scryglass_public_health to anon, authenticated" in migration
    assert "grant select on public.scryglass_refresh_runs to anon" not in migration


def test_refresh_ledger_migration_restricts_release_functions() -> None:
    migration = (
        ROOT / "supabase/migrations/20260811174735_refresh_run_ledger.sql"
    ).read_text(encoding="utf-8")

    for function in (
        "activate_scryglass_public_release(text)",
        "restore_scryglass_public_release(text)",
        "prune_scryglass_public_releases(integer)",
    ):
        assert f"alter function public.{function} security invoker" in migration
        assert f"grant execute on function public.{function} to service_role" in migration
    assert "security definer" not in migration


def test_refresh_ledger_indexes_every_foreign_key() -> None:
    migration = (
        ROOT / "supabase/migrations/20260811174735_refresh_run_ledger.sql"
    ).read_text(encoding="utf-8")

    for index in (
        "scryglass_refresh_runs_retry_idx",
        "scryglass_refresh_runs_release_idx",
        "scryglass_public_health_release_idx",
        "scryglass_public_health_run_idx",
    ):
        assert f"create index {index}" in migration


def test_supabase_cli_is_pinned_and_local_seed_is_disabled() -> None:
    package = json.loads((ROOT / "package.json").read_text(encoding="utf-8"))
    config = (ROOT / "supabase/config.toml").read_text(encoding="utf-8")

    assert package["devDependencies"]["supabase"] == "2.113.0"
    assert package["scripts"]["supabase:migrations"] == "supabase migration list"
    seed_section = config.split("[db.seed]", 1)[1].split("\n[", 1)[0]
    assert "enabled = false" in seed_section


def test_database_allowlist_matches_publication_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    migrations = sorted((root / "supabase" / "migrations").glob("*.sql"))
    sql = "\n".join(path.read_text(encoding="utf-8") for path in migrations)
    for path in supabase_publication.PUBLIC_ASSET_PATHS:
        assert f"'{path}'" in sql
    activation_migrations = [
        path
        for path in migrations
        if "required_assets constant text[] := array["
        in path.read_text(encoding="utf-8")
    ]
    assert activation_migrations, "no activation migration found"
    activation = activation_migrations[-1].read_text(encoding="utf-8")
    required_section = activation.split("required_assets constant text[] := array[", 1)[1].split("];", 1)[0]
    for path in (*supabase_publication.PUBLIC_RATING_REQUIRED_FILES, supabase_publication.TIER_ASSET_PATH):
        assert f"'{path}'" in required_section
    assert "'features/schedule.json'" not in required_section


def test_phase_one_publication_migration_keeps_compatibility_boundaries() -> None:
    migration = (
        ROOT
        / "supabase/migrations/20260813010000_public_release_security_hardening.sql"
    ).read_text(encoding="utf-8").lower()

    assert 'create policy "read active scryglass storage assets"' in migration
    assert "release.status = 'active'" in migration
    assert "asset.storage_path = storage.objects.name" in migration
    assert "body is null" in migration
    assert "artifact_hashes" in migration
    assert "grant select on public.scryglass_public_releases to anon, authenticated" in migration
    assert "grant select on public.scryglass_public_assets to anon, authenticated" in migration
    assert "set public = false" not in migration


def test_quarantine_reason_migration_keeps_details_private() -> None:
    migration = (
        ROOT / "supabase/migrations/20260811193000_oe_quarantine_reasons.sql"
    ).read_text(encoding="utf-8")

    assert "add column if not exists quarantined_games jsonb" in migration
    assert "jsonb_typeof(quarantined_games) = 'object'" in migration
    assert "grant" not in migration.lower()


def test_rls_auto_enable_migration_removes_public_execution() -> None:
    migration = (
        ROOT / "supabase/migrations/20260811193736_restrict_rls_auto_enable.sql"
    ).read_text(encoding="utf-8").lower()

    assert "revoke all on function public.rls_auto_enable()" in migration
    assert "from public, anon, authenticated, service_role" in migration
    assert "grant execute" not in migration
