from __future__ import annotations

import base64
import hashlib
import json
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

    def stage_release(self, release, assets, *, storage_objects=None):
        self.stage_calls += 1
        self.releases[release["release_id"]] = dict(release)
        return self.stage_assets(
            str(release["release_id"]),
            assets,
            storage_objects=storage_objects,
        )

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
        "receipt_sha256": None,
        "issued_utc": None,
        "reason": "model_not_promoted",
    }


def test_publish_release_omits_unpromoted_draft_asset() -> None:
    with TemporaryDirectory() as temporary:
        pack, manifest, tier = _fixture(Path(temporary))
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
        original = database.stage_release

        def stage_corrupt(release, assets, *, storage_objects=None):
            reused = original(release, assets, storage_objects=storage_objects)
            path = f"{manifest['pack_id']}/{supabase_publication.PUBLIC_RATING_REQUIRED_FILES[0]}"
            database.storage[path] = b"{}"
            return reused

        database.stage_release = stage_corrupt  # type: ignore[method-assign]
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
        path for path in migrations
        if "activate_scryglass_public_release" in path.read_text(encoding="utf-8")
    ]
    assert activation_migrations, "no activation migration found"
    activation = activation_migrations[-1].read_text(encoding="utf-8")
    required_section = activation.split("required_assets constant text[] := array[", 1)[1].split("];", 1)[0]
    for path in (*supabase_publication.PUBLIC_RATING_REQUIRED_FILES, supabase_publication.TIER_ASSET_PATH):
        assert f"'{path}'" in required_section
    assert "'features/schedule.json'" not in required_section


def test_final_publication_migration_closes_storage_and_function_boundaries() -> None:
    migration = "\n".join(
        [
            (
                ROOT / "supabase/migrations/20260814020000_private_storage_phase.sql"
            ).read_text(encoding="utf-8"),
            (
                ROOT / "supabase/migrations/20260814030000_strict_public_cutover.sql"
            ).read_text(encoding="utf-8"),
        ]
    ).lower()

    assert "set public = false" in migration
    assert 'create policy "read active scryglass storage assets"' in migration
    assert "active scryglass storage objects are immutable" in migration
    assert "revoke all on public.scryglass_public_releases" in migration
    assert "revoke all on public.scryglass_public_assets" in migration
    assert "drop function if exists public.get_scryglass_active_inline_asset(text, text)" in migration


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
