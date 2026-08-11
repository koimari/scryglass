from __future__ import annotations

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
    assert result["assets"] == len(supabase_publication.PUBLIC_RATING_REQUIRED_FILES) + 2
    assert result["reused_assets"] == 0
    assert client.active_id == manifest["pack_id"]
    assert len(client.assets) == len(supabase_publication.PUBLIC_RATING_REQUIRED_FILES) + 2
    assert list(client.storage) == [
        "v2026.08.10.153000/rankings/tierlists.json",
        "v2026.08.10.153000/rankings/tierlists-latest.json",
    ]
    assert client.releases[manifest["pack_id"]]["manifest"]["data_backend"] == "supabase"
    assert client.releases[manifest["pack_id"]]["manifest"]["tier"] == {
        "status": "available",
        "as_of": None,
        "latest_path": "rankings/tierlists-latest.json",
    }


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

    assert first["status"] == "published"
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

        result = supabase_publication.publish_release(
            pack,
            manifest,
            tier,
            project_url=client.project_url,
            secret_key="sb_secret_unused_because_client_is_injected",
            client=client,
        )

    assert result["status"] == "already_active"
    assert result["reused_assets"] == result["assets"] - 1
    assert latest_key in client.assets
    assert latest_storage in client.storage
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


def test_client_repr_redacts_secret_key() -> None:
    client = supabase_publication.SupabasePublicData(
        "https://example.supabase.co",
        "sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    assert "abcdefghijklmnopqrstuvwxyz" not in repr(client)
    assert "<redacted>" in repr(client)


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

    assert latest["latest_patch"] == "16.10"
    assert latest["rows"] == [{"patch": "16.10", "champion": "New"}]
    assert latest["scopes"] == [{"patch": "16.10", "response_matrix": {"new": True}}]
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
    activation = (root / "supabase" / "migrations" / "20260811121932_public_match_assets.sql").read_text(encoding="utf-8")
    required_section = activation.split("required_assets constant text[] := array[", 1)[1].split("];", 1)[0]
    for path in (*supabase_publication.PUBLIC_RATING_REQUIRED_FILES, supabase_publication.TIER_ASSET_PATH):
        assert f"'{path}'" in required_section
    assert "'features/schedule.json'" not in required_section
