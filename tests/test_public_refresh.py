from __future__ import annotations

import json
import hashlib
import urllib.error
from dataclasses import replace
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from lol_kills import public_refresh
from lol_kills.export.public_pack import source_identity_sha256
from lol_kills.postgame_sync import RefreshValidationError, validate_public_identity


NOW = datetime(2026, 8, 10, 0, 0, tzinfo=timezone.utc)


def _config(tmp_path: Path) -> public_refresh.RefreshConfig:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    return public_refresh.RefreshConfig(
        root=tmp_path,
        runtime_root=tmp_path,
        public_root=tmp_path / "packs",
        site="https://example.test",
        manifest_url=None,
        blob_root=None,
        health_path=runtime / "health.json",
        state_path=runtime / "public-state.json",
        lock_path=runtime / "lock",
        production=False,
        publication_backend="blob",
        supabase_url=None,
        supabase_secret_key=None,
    )


def _with_receipt_paths(config: public_refresh.RefreshConfig) -> public_refresh.RefreshConfig:
    source = config.runtime_root / "accepted-source.json"
    imported = config.runtime_root / "accepted-import.json"
    source.write_text("{}", encoding="utf-8")
    imported.write_text("{}", encoding="utf-8")
    return replace(
        config,
        accepted_source_receipt=source,
        accepted_import_receipt=imported,
    )


def test_tier_publication_binds_runtime_payload_to_accepted_source(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), publication_backend="supabase")
    payload_path = (
        config.runtime_root
        / "apps/scryglass/public/rankings/tierlists.json"
    )
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps(
            {
                "schema_version": "rankings-tierlists-v2",
                "status": "available",
                "as_of": "2026-08-09T17:00:00Z",
                "rows": [
                    {"patch": "26.16", "role": role, "champion": f"{role}-champion", "rank": 1}
                    for role in ("top", "jungle", "mid", "bot", "support")
                ],
            }
        ),
        encoding="utf-8",
    )
    source_identity = "c" * 64
    receipt = {
        "status": "production_built",
        "source_observed_through": "2026-08-09T17:00:00Z",
        "candidate_expected_live_as_of": "2026-08-09T17:00:00Z",
        "receipt_canonical_sha256": "d" * 64,
        "candidate": {"artifact_sha256": "e" * 64, "maps_replayed": 22},
    }

    with patch.object(
        public_refresh,
        "validate_live_source",
        return_value={"game_count": 22, "identity_sha256": source_identity},
    ):
        binding = public_refresh._tier_publication_for_source(
            config,
            receipt,
            source_observed_through="2026-08-09T17:00:00Z",
            source={"identity_sha256": source_identity, "game_count": 22},
        )

    assert binding["status"] == "available"
    assert binding["production_status"] == "production_built"
    assert binding["source_identity_sha256"] == source_identity
    assert binding["scope_count"] == 1
    assert len(binding["payload_sha256"]) == 64


def test_tier_publication_rejects_incomplete_role_scope(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), publication_backend="supabase")
    payload_path = config.runtime_root / "apps/scryglass/public/rankings/tierlists.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps(
            {
                "schema_version": "rankings-tierlists-v2",
                "status": "available",
                "as_of": "2026-08-09T17:00:00Z",
                "rows": [{"patch": "26.16", "role": "mid", "champion": "Ahri", "rank": 1}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(public_refresh.PublicRefreshError, match="five-role scope"):
        public_refresh._tier_publication_for_source(
            config,
            {
                "status": "production_built",
                "source_observed_through": "2026-08-09T17:00:00Z",
                "candidate_expected_live_as_of": "2026-08-09T17:00:00Z",
                "receipt_canonical_sha256": "d" * 64,
                "candidate": {"artifact_sha256": "e" * 64},
            },
            source_observed_through="2026-08-09T17:00:00Z",
            source={"identity_sha256": "c" * 64},
        )


def test_tier_publication_rejects_a_different_game_census(tmp_path: Path) -> None:
    config = replace(_config(tmp_path), publication_backend="supabase")
    payload_path = config.runtime_root / "apps/scryglass/public/rankings/tierlists.json"
    payload_path.parent.mkdir(parents=True)
    payload_path.write_text(
        json.dumps(
            {
                "schema_version": "rankings-tierlists-v2",
                "status": "available",
                "as_of": "2026-08-16T21:07:39Z",
                "rows": [
                    {"patch": "26.16", "role": role, "champion": role, "rank": 1}
                    for role in ("top", "jungle", "mid", "bot", "support")
                ],
            }
        ),
        encoding="utf-8",
    )
    source_identity = "c" * 64
    receipt = {
        "status": "production_built",
        "source_observed_through": "2026-08-16T21:07:39Z",
        "candidate_expected_live_as_of": "2026-08-16T21:07:39Z",
        "receipt_canonical_sha256": "d" * 64,
        "candidate": {"artifact_sha256": "e" * 64, "maps_replayed": 12},
    }
    with patch.object(
        public_refresh,
        "validate_live_source",
        return_value={"game_count": 22, "identity_sha256": source_identity},
    ), pytest.raises(public_refresh.PublicRefreshError, match="game census"):
        public_refresh._tier_publication_for_source(
            config,
            receipt,
            source_observed_through="2026-08-16T21:07:39Z",
            source={"identity_sha256": source_identity, "game_count": 22},
        )


def test_environment_defaults_to_supabase_publication(tmp_path: Path) -> None:
    with patch.dict("os.environ", {}, clear=True):
        config = public_refresh.config_from_environment(tmp_path, tmp_path / "packs")

    assert config.publication_backend == "supabase"


def test_tier_failure_keeps_a_smoke_verified_ratings_release(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync.state_path.parent.mkdir(parents=True, exist_ok=True)
    config.sync.state_path.write_text(json.dumps({"pack_id": "old"}), encoding="utf-8")
    ratings = {
        "status": "published",
        "pack_id": "new",
        "publication": {"runtime": "local_staging", "pack_id": "new"},
    }

    with patch.object(public_refresh, "_preflight"), patch.object(
        public_refresh, "_run_with_source_retries", return_value=ratings
    ), patch.object(
        public_refresh,
        "_run_tier_refresh",
        side_effect=RuntimeError("authority is waiting"),
    ), patch.object(public_refresh, "invalidate_public_cache", return_value={"revalidated": True}), patch.object(
        public_refresh,
        "verify_public_release",
        return_value={"pack_id": "new", "files": 9, "tier_status": "available"},
    ), patch.object(public_refresh, "rollback_public_pack") as rollback:
        with pytest.raises(public_refresh.PublicRefreshError, match="authority is waiting"):
            public_refresh.run_once(config, now=NOW)

    rollback.assert_not_called()
    health = json.loads(config.health_path.read_text(encoding="utf-8"))
    assert health["status"] == "error"
    result = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert result["status"] == "partial"


def test_failed_public_smoke_restores_the_previous_pack(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync.state_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {"pack_id": "old", "published_game_ids": ["game-1"]}
    config.sync.state_path.write_text(json.dumps(previous), encoding="utf-8")
    ratings = {
        "status": "published",
        "pack_id": "new",
        "publication": {"runtime": "local_staging", "pack_id": "new"},
    }

    with patch.object(public_refresh, "_preflight"), patch.object(
        public_refresh, "_run_with_source_retries", return_value=ratings
    ), patch.object(
        public_refresh,
        "_run_tier_refresh",
        return_value={"status": "production_promoted"},
    ), patch.object(public_refresh, "invalidate_public_cache", return_value={"revalidated": True}), patch.object(
        public_refresh,
        "verify_public_release",
        side_effect=public_refresh.PublicRefreshError("page smoke failed"),
    ), patch.object(
        public_refresh,
        "rollback_public_pack",
        return_value={"status": "restored"},
    ) as rollback:
        with pytest.raises(public_refresh.PublicRefreshError, match="page smoke failed"):
            public_refresh.run_once(config, now=NOW)

    rollback.assert_called_once_with(ratings["publication"], config.public_root)
    assert json.loads(config.sync.state_path.read_text(encoding="utf-8")) == previous
    health = json.loads(config.health_path.read_text(encoding="utf-8"))
    assert health["rollback"]["status"] == "restored"
    assert health["rollback"]["ratings"] == {"status": "restored"}
    assert health["rollback"]["cache_invalidation"] == {"revalidated": True}


def test_failed_post_rollback_probe_keeps_rollback_failed(tmp_path: Path) -> None:
    previous_release = "v2026.08.10.001500"
    release_id = "v2026.08.11.184500"
    config = replace(_config(tmp_path), production=True)
    config.sync.state_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {"pack_id": previous_release, "published_game_ids": ["game-1"]}
    config.sync.state_path.write_text(json.dumps(previous), encoding="utf-8")
    ratings = {
        "status": "published",
        "pack_id": release_id,
        "publication": {"runtime": "local_staging", "pack_id": release_id},
    }

    with patch.object(public_refresh, "_preflight"), patch.object(
        public_refresh, "_run_with_source_retries", return_value=ratings
    ), patch.object(
        public_refresh,
        "_run_tier_refresh",
        return_value={"status": "production_promoted"},
    ), patch.object(
        public_refresh, "invalidate_public_cache", return_value={"revalidated": True}
    ), patch.object(
        public_refresh,
        "probe_public_release_families",
        side_effect=[
            {"release_id": release_id},
            public_refresh.PublicRefreshError("restored page is stale"),
        ],
    ), patch.object(
        public_refresh,
        "verify_public_release",
        side_effect=public_refresh.PublicRefreshError("page smoke failed"),
    ), patch.object(
        public_refresh,
        "rollback_public_pack",
        return_value={"status": "restored"},
    ):
        with pytest.raises(public_refresh.PublicRefreshError, match="page smoke failed"):
            public_refresh.run_once(config, now=NOW)

    health = json.loads(config.health_path.read_text(encoding="utf-8"))
    assert health["rollback"]["status"] == "failed"
    assert health["rollback"]["cache_invalidation"] == {"revalidated": True}
    assert health["rollback"]["cache_probes"]["status"] == "failed"
    assert "restored page is stale" in health["rollback"]["reason"]


def test_cache_failure_restores_the_new_pack_and_retries_cache_clear(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync.state_path.parent.mkdir(parents=True, exist_ok=True)
    previous = {"pack_id": "old", "published_game_ids": ["game-1"]}
    config.sync.state_path.write_text(json.dumps(previous), encoding="utf-8")
    config.state_path.write_text(
        json.dumps({"tier": {"status": "available"}}), encoding="utf-8"
    )
    ratings = {
        "status": "published",
        "pack_id": "new",
        "publication": {"runtime": "local_staging", "pack_id": "new"},
    }

    with patch.object(public_refresh, "_preflight"), patch.object(
        public_refresh, "_run_with_source_retries", return_value=ratings
    ), patch.object(
        public_refresh,
        "invalidate_public_cache",
        side_effect=[public_refresh.PublicRefreshError("cache is unavailable"), {"revalidated": True}],
    ) as invalidate, patch.object(
        public_refresh,
        "rollback_public_pack",
        return_value={"status": "restored"},
    ) as rollback:
        with pytest.raises(public_refresh.PublicRefreshError, match="cache is unavailable"):
            public_refresh.run_once(config, now=NOW)

    rollback.assert_called_once_with(ratings["publication"], config.public_root)
    assert invalidate.call_count == 2
    assert json.loads(config.sync.state_path.read_text(encoding="utf-8")) == previous
    health = json.loads(config.health_path.read_text(encoding="utf-8"))
    assert health["rollback"]["status"] == "restored"
    assert health["rollback"]["cache_invalidation"] == {"revalidated": True}


def test_identity_gate_allows_exact_case_route_collisions(tmp_path: Path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    (features / "ratings_snapshot.json").write_text(
        json.dumps([{"team": "Team Solid", "team_key": "team-solid"}]), encoding="utf-8"
    )
    (features / "player_ratings_snapshot.json").write_text(
        json.dumps([{"player": "Random"}, {"player": "random"}]), encoding="utf-8"
    )
    (features / "team_records.json").write_text(
        json.dumps({"Team Solid": {"team_key": "team-solid"}}), encoding="utf-8"
    )
    (features / "profile_records.json").write_text(json.dumps({}), encoding="utf-8")

    result = validate_public_identity(tmp_path)

    assert result["team_key_count"] == 1
    assert result["player_case_collision_count"] == 1


def test_identity_gate_rejects_duplicate_team_keys(tmp_path: Path) -> None:
    features = tmp_path / "features"
    features.mkdir()
    (features / "ratings_snapshot.json").write_text(
        json.dumps(
            [
                {"team": "Team Solid", "team_key": "team-solid"},
                {"team": "Team Solid (Brazilian Team)", "team_key": "team-solid"},
            ]
        ),
        encoding="utf-8",
    )
    for name, payload in (
        ("player_ratings_snapshot.json", []),
        ("team_records.json", {}),
        ("profile_records.json", {}),
    ):
        (features / name).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(RefreshValidationError, match="duplicate canonical keys"):
        validate_public_identity(tmp_path)


def test_watchdog_alerts_once_when_the_last_success_is_stale(tmp_path: Path) -> None:
    config = _config(tmp_path)
    old = "2026-08-09T00:00:00Z"
    config.health_path.write_text(
        json.dumps(
            {
                "status": "error",
                "checked_at": "2026-08-09T18:00:00Z",
                "last_success_at": old,
                "reason": "source unavailable",
            }
        ),
        encoding="utf-8",
    )

    with patch.dict("os.environ", {"SCRYGLASS_ALERT_WEBHOOK_URL": "https://alerts.test"}), patch.object(
        public_refresh, "_http_bytes", return_value=b"ok"
    ) as send:
        assert public_refresh.notify_watchdog(config, now=NOW) is True
        assert public_refresh.notify_watchdog(config, now=NOW) is True

    send.assert_called_once()
    health = json.loads(config.health_path.read_text(encoding="utf-8"))
    assert health["stale_alert_sent_at"] == old


def test_watchdog_stays_quiet_when_the_last_success_is_fresh(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.health_path.write_text(
        json.dumps(
            {
                "status": "ok",
                "checked_at": "2026-08-09T18:00:00Z",
                "last_success_at": "2026-08-09T18:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    with patch.object(public_refresh, "_http_bytes") as send:
        assert public_refresh.notify_watchdog(config, now=NOW) is False

    send.assert_not_called()


def test_production_preflight_rejects_the_legacy_blob_backend(tmp_path: Path) -> None:
    base = "https://store-test.public.blob.vercel-storage.com"
    config = _with_receipt_paths(
        replace(
            _config(tmp_path),
            production=True,
            blob_root=base,
            manifest_url=f"{base}/packs/manifest.json",
        )
    )
    with patch.dict(
        "os.environ",
        {
            "SCRYGLASS_DATA_PUBLISH_TOKEN": "publish-key",
            "SCRYGLASS_DIAGNOSTIC_TOKEN": "diagnostic-key",
        },
        clear=True,
    ), pytest.raises(public_refresh.PublicRefreshError, match="Supabase publication backend"):
        public_refresh._preflight(config)


def test_supabase_preflight_does_not_require_blob_credentials(tmp_path: Path) -> None:
    config = _with_receipt_paths(
        replace(
            _config(tmp_path),
            production=True,
            site=public_refresh.DEFAULT_SITE,
            publication_backend="supabase",
            supabase_url="https://example.supabase.co",
            supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
        )
    )
    with patch.dict(
        "os.environ",
        {
            "SCRYGLASS_DATA_PUBLISH_TOKEN": "publish-key",
            "SCRYGLASS_DIAGNOSTIC_TOKEN": "diagnostic-key",
        },
        clear=True,
    ), patch.object(public_refresh, "worker_commit", return_value="a" * 40) as commit:
        public_refresh._preflight(config)
    commit.assert_called_once_with(config.root, require_clean=True)


def test_production_preflight_requires_a_separate_diagnostic_token(
    tmp_path: Path,
) -> None:
    config = _with_receipt_paths(
        replace(
            _config(tmp_path),
            production=True,
            site=public_refresh.DEFAULT_SITE,
            publication_backend="supabase",
            supabase_url="https://example.supabase.co",
            supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
        )
    )
    with patch.dict(
        "os.environ",
        {"SCRYGLASS_DATA_PUBLISH_TOKEN": "publish-key"},
        clear=True,
    ), pytest.raises(public_refresh.PublicRefreshError, match="SCRYGLASS_DIAGNOSTIC_TOKEN"):
        public_refresh._preflight(config)


def test_production_preflight_requires_the_exact_public_origin(tmp_path: Path) -> None:
    config = _with_receipt_paths(
        replace(
            _config(tmp_path),
            production=True,
            site="https://scryglass.xyz.evil.test",
            publication_backend="supabase",
            supabase_url="https://example.supabase.co",
            supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
        )
    )
    with patch.dict(
        "os.environ",
        {
            "SCRYGLASS_DATA_PUBLISH_TOKEN": "publish-key",
            "SCRYGLASS_DIAGNOSTIC_TOKEN": "diagnostic-key",
        },
        clear=True,
    ), pytest.raises(public_refresh.PublicRefreshError, match="exactly"):
        public_refresh._preflight(config)


def test_production_preflight_requires_bound_source_receipts(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        production=True,
        site=public_refresh.DEFAULT_SITE,
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    with patch.dict(
        "os.environ",
        {
            "SCRYGLASS_DATA_PUBLISH_TOKEN": "publish-key",
            "SCRYGLASS_DIAGNOSTIC_TOKEN": "diagnostic-key",
        },
        clear=True,
    ), pytest.raises(public_refresh.PublicRefreshError, match="accepted source"):
        public_refresh._preflight(config)


def test_fresh_supabase_worker_seeds_exact_active_game_ids(tmp_path: Path) -> None:
    game_ids = ["oe:game:a", "oe:game:b"]
    release_id = "v2026.08.10.001500"
    manifest = {
        "pack_id": release_id,
        "ratings": {
            "source_game_count": len(game_ids),
            "source_identity_sha256": source_identity_sha256(game_ids),
        },
    }
    profiles = json.dumps(
        {"games": {game_id: {"game_id": game_id} for game_id in game_ids}}
    ).encode()
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def active_release(self):
            return {"release_id": release_id, "manifest": manifest}

        def asset(self, _release_id: str, path: str):
            assert path == "features/profile_records.json"
            return {"body": None, "storage_path": f"{release_id}/{path}"}

        def storage_object(self, _storage_path: str):
            return profiles

    with patch.object(
        public_refresh.supabase_publication,
        "SupabasePublicData",
        Client,
    ):
        result = public_refresh.seed_supabase_continuity(config)

    assert result == {
        "status": "seeded",
        "pack_id": release_id,
        "game_count": 2,
        "source": "profile_index",
    }
    assert json.loads(config.sync.state_path.read_text())["published_game_ids"] == game_ids
    assert json.loads((config.public_root / "manifest.json").read_text()) == manifest


def test_supabase_bootstrap_uses_a_checksum_verified_full_local_cache(tmp_path: Path) -> None:
    game_ids = ["oe:game:a", "oe:game:b", "oe:game:c"]
    release_id = "v2026.08.10.001500"
    manifest = {
        "pack_id": release_id,
        "ratings": {
            "source_game_count": len(game_ids),
            "source_identity_sha256": source_identity_sha256(game_ids),
        },
    }
    recent_profiles = json.dumps(
        {"games": {game_ids[-1]: {"game_id": game_ids[-1]}}}
    ).encode()
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def active_release(self):
            return {"release_id": release_id, "manifest": manifest}

        def asset(self, _release_id: str, path: str):
            return {"body": None, "storage_path": f"{release_id}/{path}"}

        def storage_object(self, _storage_path: str):
            return recent_profiles

    with patch.object(
        public_refresh.supabase_publication,
        "SupabasePublicData",
        Client,
    ), patch.object(
        public_refresh,
        "validate_live_source",
        return_value={"game_ids": game_ids},
    ):
        result = public_refresh.seed_supabase_continuity(config)

    assert result["source"] == "validated_local_cache"
    assert json.loads(config.sync.state_path.read_text())["published_game_ids"] == game_ids


def test_supabase_bootstrap_recovers_a_worker_ahead_of_the_active_release(tmp_path: Path) -> None:
    # The recovery-seeded local cache is a *contaminated* superset: it covers
    # the release index plus one out-of-window (2024) game that never left
    # worker state. The bootstrap must never seed that unfiltered superset —
    # only the restriction of it to the release's declared publication
    # window years, and only when that restriction reproduces the published
    # binding exactly.
    release_ids = ["oe:game:a", "oe:game:b"]
    contaminated_id = "oe:game:c-2024"
    local_ids = [*release_ids, contaminated_id]
    release_id = "v2026.08.10.001500"
    manifest = {
        "pack_id": release_id,
        "ratings": {
            "source_game_count": len(release_ids),
            "source_identity_sha256": source_identity_sha256(release_ids),
            "window_years": [2025, 2026],
        },
    }
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    live = config.runtime_root / "data/lol/warehouse/parquet/oe_live"
    live.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "game_uid": game_id,
                "date": (
                    "2024-11-03T00:00:00Z"
                    if game_id == contaminated_id
                    else "2026-06-01T00:00:00Z"
                ),
            }
            for game_id in local_ids
        ]
    ).to_parquet(live / "maps.parquet", index=False)

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def active_release(self):
            return {"release_id": release_id, "manifest": manifest}

        def asset(self, _release_id: str, path: str):
            return {"body": None, "storage_path": f"{release_id}/{path}"}

        def storage_object(self, storage_path: str):
            if storage_path.endswith("features/match_index.json"):
                return json.dumps(
                    {"games": [{"game_id": game_id} for game_id in release_ids]}
                ).encode()
            return json.dumps({"games": {"oe:game:a": {"game_id": "oe:game:a"}}}).encode()

    with patch.object(
        public_refresh.supabase_publication,
        "SupabasePublicData",
        Client,
    ), patch.object(
        public_refresh,
        "validate_live_source",
        return_value={"game_ids": local_ids},
    ):
        result = public_refresh.seed_supabase_continuity(config)

    assert result == {
        "status": "seeded",
        "pack_id": release_id,
        "game_count": len(release_ids),
        "source": "validated_local_cache_superset",
    }
    written = json.loads(config.sync.state_path.read_text())
    assert written["published_game_ids"] == release_ids
    assert contaminated_id not in written["published_game_ids"]
    assert written["release_index_game_ids"] == sorted(release_ids)
    assert json.loads((config.public_root / "manifest.json").read_text()) == manifest


def test_supabase_bootstrap_rejects_a_superset_whose_window_filter_still_mismatches(
    tmp_path: Path,
) -> None:
    # The window restriction removes nothing here (every id is in-window),
    # so it still fails to reproduce the two-game published binding. The
    # bootstrap must raise rather than silently seed the unfiltered, or any
    # partially filtered, superset.
    release_ids = ["oe:game:a", "oe:game:b"]
    local_ids = [*release_ids, "oe:game:extra-inwindow"]
    release_id = "v2026.08.10.001500"
    manifest = {
        "pack_id": release_id,
        "ratings": {
            "source_game_count": len(release_ids),
            "source_identity_sha256": source_identity_sha256(release_ids),
            "window_years": [2025, 2026],
        },
    }
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    live = config.runtime_root / "data/lol/warehouse/parquet/oe_live"
    live.mkdir(parents=True)
    pd.DataFrame(
        [{"game_uid": game_id, "date": "2026-06-01T00:00:00Z"} for game_id in local_ids]
    ).to_parquet(live / "maps.parquet", index=False)

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def active_release(self):
            return {"release_id": release_id, "manifest": manifest}

        def asset(self, _release_id: str, path: str):
            return {"body": None, "storage_path": f"{release_id}/{path}"}

        def storage_object(self, storage_path: str):
            if storage_path.endswith("features/match_index.json"):
                return json.dumps(
                    {"games": [{"game_id": game_id} for game_id in release_ids]}
                ).encode()
            return json.dumps({"games": {"oe:game:a": {"game_id": "oe:game:a"}}}).encode()

    with patch.object(
        public_refresh.supabase_publication,
        "SupabasePublicData",
        Client,
    ), patch.object(
        public_refresh,
        "validate_live_source",
        return_value={"game_ids": local_ids},
    ), pytest.raises(public_refresh.PublicRefreshError, match="does not match"):
        public_refresh.seed_supabase_continuity(config)

    assert not config.sync.state_path.exists()


def test_supabase_bootstrap_rejects_a_worker_missing_release_games(tmp_path: Path) -> None:
    release_ids = ["oe:game:a", "oe:game:b"]
    local_ids = ["oe:game:a"]
    release_id = "v2026.08.10.001500"
    manifest = {
        "pack_id": release_id,
        "ratings": {
            "source_game_count": len(release_ids),
            "source_identity_sha256": source_identity_sha256(release_ids),
        },
    }
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def active_release(self):
            return {"release_id": release_id, "manifest": manifest}

        def asset(self, _release_id: str, path: str):
            return {"body": None, "storage_path": f"{release_id}/{path}"}

        def storage_object(self, storage_path: str):
            if storage_path.endswith("features/match_index.json"):
                return json.dumps(
                    {"games": [{"game_id": game_id} for game_id in release_ids]}
                ).encode()
            return json.dumps({"games": {"oe:game:a": {"game_id": "oe:game:a"}}}).encode()

    with patch.object(
        public_refresh.supabase_publication,
        "SupabasePublicData",
        Client,
    ), patch.object(
        public_refresh,
        "validate_live_source",
        return_value={"game_ids": local_ids},
    ), pytest.raises(public_refresh.PublicRefreshError, match="does not match"):
        public_refresh.seed_supabase_continuity(config)


def test_supabase_current_worker_refreshes_its_active_manifest(tmp_path: Path) -> None:
    game_ids = ["oe:game:a", "oe:game:b"]
    release_id = "v2026.08.10.001500"
    manifest = {
        "pack_id": release_id,
        "tier": {"status": "available"},
        "ratings": {
            "source_game_count": len(game_ids),
            "source_identity_sha256": source_identity_sha256(game_ids),
        },
    }
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    config.public_root.mkdir(parents=True)
    (config.public_root / "manifest.json").write_text(
        json.dumps({"pack_id": release_id}),
        encoding="utf-8",
    )
    config.sync.state_path.parent.mkdir(parents=True, exist_ok=True)
    config.sync.state_path.write_text(
        json.dumps({"pack_id": release_id, "published_game_ids": game_ids}),
        encoding="utf-8",
    )

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def active_release(self):
            return {"release_id": release_id, "manifest": manifest}

    with patch.object(public_refresh.supabase_publication, "SupabasePublicData", Client):
        result = public_refresh.seed_supabase_continuity(config)

    assert result["status"] == "current"
    assert json.loads((config.public_root / "manifest.json").read_text()) == manifest


def test_supabase_no_change_reuses_the_active_tier(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    config.public_root.mkdir(parents=True)
    (config.public_root / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "v2026.08.10.001500",
                "tier": {"status": "available", "as_of": "2026-08-08T21:50:46Z"},
            }
        ),
        encoding="utf-8",
    )
    ratings = {
        "status": "no_change",
        "pack_id": "v2026.08.10.001500",
        "source_observed_through": "2026-08-08T21:50:46Z",
    }

    with patch.object(public_refresh, "_preflight"), patch.object(
        public_refresh,
        "seed_supabase_continuity",
        return_value={"status": "current"},
    ), patch.object(
        public_refresh,
        "_run_with_source_retries",
        return_value=ratings,
    ), patch.object(
        public_refresh,
        "_run_tier_refresh",
        side_effect=AssertionError("unchanged source must reuse the active tier"),
    ), patch.object(
        public_refresh,
        "verify_public_release",
        return_value={
            "pack_id": "v2026.08.10.001500",
            "files": 9,
            "tier_status": "available",
        },
    ):
        result = public_refresh.run_once(config, now=NOW)

    assert result["status"] == "no_change"
    assert result["tier"] is None
    assert result["database_publication"] is None
    assert result["cache_invalidation"] is None


def test_supabase_tier_failure_stops_before_publication_and_cache(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    config.public_root.mkdir(parents=True)
    (config.public_root / "manifest.json").write_text(
        json.dumps({"pack_id": "old", "tier": {"status": "available"}}),
        encoding="utf-8",
    )
    ratings = {
        "status": "published",
        "pack_id": "new",
        "source_observed_through": "2026-08-11T10:50:41Z",
    }

    with patch.object(public_refresh, "_preflight"), patch.object(
        public_refresh,
        "seed_supabase_continuity",
        return_value={"status": "current"},
    ), patch.object(
        public_refresh,
        "_run_with_source_retries",
        return_value=ratings,
    ), patch.object(
        public_refresh,
        "_run_tier_refresh",
        side_effect=RuntimeError("bundle failed"),
    ), patch.object(
        public_refresh.supabase_publication,
        "publish_release",
    ) as publish, patch.object(
        public_refresh,
        "invalidate_public_cache",
    ) as invalidate:
        with pytest.raises(public_refresh.PublicRefreshError, match="bundle failed"):
            public_refresh.run_once(config, now=NOW)

    publish.assert_not_called()
    invalidate.assert_not_called()


def test_http_read_retries_transient_statuses(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"ok"

    calls = iter(
        [
            urllib.error.HTTPError("https://example.test", 503, "busy", {}, None),
            Response(),
        ]
    )

    def open_request(*_args, **_kwargs):
        value = next(calls)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(public_refresh.urllib.request, "urlopen", open_request)
    sleeps: list[float] = []
    monkeypatch.setattr(public_refresh.time, "sleep", sleeps.append)

    assert public_refresh._http_bytes("https://example.test", attempts=2) == b"ok"
    assert sleeps == [1.0]


def test_http_read_requires_the_expected_release_header(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Response:
        headers = {"X-Scryglass-Release": "v2026.08.13.120000"}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"ok"

    monkeypatch.setattr(public_refresh.urllib.request, "urlopen", lambda *_args, **_kwargs: Response())

    with pytest.raises(public_refresh.PublicRefreshError, match="expected v2026.08.13.120001"):
        public_refresh._http_bytes(
            "https://example.test",
            expected_release_id="v2026.08.13.120001",
        )


def test_cache_invalidation_is_bound_to_the_activated_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    release_id = "v2026.08.11.184500"
    monkeypatch.setenv("SCRYGLASS_DATA_PUBLISH_TOKEN", "publish-secret")

    def send(url: str, **kwargs):
        assert url == "https://example.test/api/data-published"
        assert kwargs["method"] == "POST"
        assert kwargs["headers"]["authorization"] == "Bearer publish-secret"
        assert json.loads(kwargs["body"]) == {"release_id": release_id}
        return json.dumps(
            {
                "revalidated": True,
                "requested_release_id": release_id,
                "served_release_id": release_id,
                "matches": True,
            }
        ).encode()

    with patch.object(public_refresh, "_http_bytes", side_effect=send):
        result = public_refresh.invalidate_public_cache(_config(tmp_path), release_id)

    assert result["served_release_id"] == release_id


def test_public_release_probes_cover_each_cache_family(tmp_path: Path) -> None:
    release_id = "v2026.08.11.184500"
    config = replace(_config(tmp_path), production=True)
    requested: list[str] = []
    asset_raw = b"asset"
    asset_path = "features/schedule.json"
    asset_url = f"/api/assets/{release_id}/features%2Fschedule.json"

    def read(url: str, **_kwargs):
        requested.append(url)
        if url.endswith("/api/health"):
            return json.dumps({"pack_id": release_id}).encode()
        if url.endswith("/packs/manifest.json"):
            return json.dumps(
                {
                    "release_id": release_id,
                    "files": [
                        {
                            "path": asset_path,
                            "url": asset_url,
                            "bytes": len(asset_raw),
                            "sha256": hashlib.sha256(asset_raw).hexdigest(),
                        }
                    ],
                }
            ).encode()
        if url.endswith(asset_url):
            return asset_raw
        if "/api/chat/matches?team=T1&limit=1" in url:
            return json.dumps(
                {"ok": True, "data": {"matches": [{"game_id": "game-1"}]}}
            ).encode()
        if "/api/" in url:
            return b"{}"
        return f'<main data-scryglass-release="{release_id}">page</main>'.encode()

    with patch.object(public_refresh, "_http_bytes", side_effect=read):
        result = public_refresh.probe_public_release_families(config, release_id)

    assert result == {
        "release_id": release_id,
        "routes": [
            *public_refresh.PUBLIC_RELEASE_PROBES,
            asset_url,
            "/elo/player/Faker",
            "/elo/team/T1",
            "/matches/game-1",
        ],
        "asset": {
            "path": asset_path,
            "url": asset_url,
            "bytes": len(asset_raw),
            "sha256": hashlib.sha256(asset_raw).hexdigest(),
        },
    }
    assert requested[0].endswith("/api/health")
    assert requested[-1].endswith("/api/health")
    assert requested[1 : 1 + len(public_refresh.PUBLIC_RELEASE_PROBES)] == [
        f"{config.site}{path}" for path in public_refresh.PUBLIC_RELEASE_PROBES
    ]
    assert requested[-4:-1] == [
        f"{config.site}/elo/player/Faker",
        f"{config.site}/elo/team/T1",
        f"{config.site}/matches/game-1",
    ]


def test_public_release_probes_skip_the_retired_tier_artifact_route() -> None:
    assert "/tiers" in public_refresh.PUBLIC_RELEASE_PROBES
    assert "/api/chat/tier?limit=1" in public_refresh.PUBLIC_RELEASE_PROBES
    assert (
        "/api/public-data/tierlists?view=latest"
        not in public_refresh.PUBLIC_RELEASE_PROBES
    )


def test_public_release_probes_reject_a_stale_page_marker(tmp_path: Path) -> None:
    release_id = "v2026.08.11.184500"
    stale_release_id = "v2026.08.10.001500"
    config = replace(_config(tmp_path), production=True)

    def read(url: str, **_kwargs):
        if url.endswith("/api/health"):
            return json.dumps({"pack_id": release_id}).encode()
        if url.endswith("/packs/manifest.json"):
            return json.dumps({"release_id": release_id}).encode()
        if "/api/chat/matches?team=T1&limit=1" in url:
            return json.dumps(
                {"ok": True, "data": {"matches": [{"game_id": "game-1"}]}}
            ).encode()
        if "/api/" in url:
            return b"{}"
        marker = stale_release_id if url.endswith("/elo") else release_id
        return f'<main data-scryglass-release="{marker}">page</main>'.encode()

    with patch.object(public_refresh, "_http_bytes", side_effect=read):
        with pytest.raises(
            public_refresh.PublicRefreshError,
            match="public page probe has no marker",
        ):
            public_refresh.probe_public_release_families(config, release_id)


def test_deployed_asset_probe_rejects_equal_size_corruption(tmp_path: Path) -> None:
    release_id = "v2026.08.11.184500"
    asset_raw = b"good"
    manifest = {
        "files": [
            {
                "path": "features/schedule.json",
                "url": f"/api/assets/{release_id}/features%2Fschedule.json",
                "bytes": len(asset_raw),
                "sha256": hashlib.sha256(asset_raw).hexdigest(),
            }
        ]
    }

    with patch.object(public_refresh, "_http_bytes", return_value=b"evil"):
        with pytest.raises(
            public_refresh.PublicRefreshError,
            match="deployed public asset failed verification",
        ):
            public_refresh._probe_deployed_manifest_asset(
                replace(_config(tmp_path), production=True),
                manifest,
                release_id,
            )


def test_health_alignment_failure_restores_previous_supabase_release(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path),
        publication_backend="supabase",
        supabase_url="https://example.supabase.co",
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    previous_release = "v2026.08.10.001500"
    release_id = "v2026.08.11.184500"
    config.sync.state_path.parent.mkdir(parents=True, exist_ok=True)
    config.sync.state_path.write_text(
        json.dumps({"pack_id": previous_release, "published_game_ids": ["game-1"]}),
        encoding="utf-8",
    )
    pack_dir = config.public_root / release_id
    pack_dir.mkdir(parents=True)
    (pack_dir / "manifest.json").write_text(
        json.dumps({"pack_id": release_id, "total_files": 1, "total_bytes": 1}),
        encoding="utf-8",
    )
    (config.public_root / "manifest.json").write_text(
        json.dumps({"pack_id": previous_release, "tier": {"status": "available"}}),
        encoding="utf-8",
    )
    ratings = {
        "status": "published",
        "pack_id": release_id,
        "source_observed_through": "2026-08-11T18:00:00Z",
    }
    accepted = {
        "source_observed_through": "2026-08-11T18:00:00Z",
        "new_games": 1,
        "corrected_games": 0,
    }
    ledger = MagicMock()
    ledger.run_id = "refresh-20260811T180000Z-aaaaaaaaaaaa"
    ledger.worker_git_commit = "a" * 40

    with patch.object(public_refresh, "_preflight"), patch.object(
        public_refresh, "_accepted_run_inputs", return_value=accepted
    ), patch.object(
        public_refresh, "_start_run_ledger", return_value=ledger
    ), patch.object(
        public_refresh, "_write_remote_health"
    ), patch.object(
        public_refresh, "seed_supabase_continuity", return_value={"status": "current"}
    ), patch.object(
        public_refresh, "_run_with_source_retries", return_value=ratings
    ), patch.object(
        public_refresh, "_run_tier_refresh", return_value={"status": "production_promoted"}
    ), patch.object(
        public_refresh.supabase_publication,
        "publish_release",
        return_value={
            "status": "published",
            "release_id": release_id,
            "previous_release_id": previous_release,
            "bytes": 1,
            "assets": 1,
        },
    ), patch.object(
        public_refresh, "invalidate_public_cache", return_value={"revalidated": True}
    ) as invalidate, patch.object(
        public_refresh,
        "probe_public_release_families",
        side_effect=lambda _config, probed_release: {"release_id": probed_release},
    ) as probes, patch.object(
        public_refresh,
        "verify_public_release",
        return_value={"pack_id": release_id, "files": 1, "tier_status": "available"},
    ), patch.object(
        public_refresh,
        "verify_public_health_alignment",
        side_effect=public_refresh.PublicRefreshError("health is stale"),
    ), patch.object(
        public_refresh.supabase_publication,
        "restore_release",
        return_value={"status": "restored", "release_id": previous_release},
    ) as restore:
        with pytest.raises(public_refresh.PublicRefreshError, match="health is stale"):
            public_refresh.run_once(config, now=NOW)

    restore.assert_called_once_with(
        previous_release,
        project_url=config.supabase_url,
        secret_key=config.supabase_secret_key,
    )
    assert invalidate.call_args_list[-1].args[1] == previous_release
    assert [call.args[1] for call in probes.call_args_list] == [
        release_id,
        previous_release,
    ]
    ledger.finish.assert_not_called()
    ledger.fail.assert_called_once()
    health = json.loads(config.health_path.read_text(encoding="utf-8"))
    assert health["rollback"]["status"] == "restored"
    assert health["rollback"]["supabase"]["release_id"] == previous_release
    assert health["rollback"]["cache_probes"] == {"release_id": previous_release}


def test_public_health_alignment_requires_release_watermark_and_worker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_id = "v2026.08.11.184500"
    worker = "a" * 40
    source_as_of = "2026-08-11T18:00:00+00:00"
    config = replace(
        _config(tmp_path),
        production=True,
        publication_backend="supabase",
    )
    monkeypatch.setenv("SCRYGLASS_DIAGNOSTIC_TOKEN", "diagnostic-secret")
    payload = {
        "status": "ok",
        "last_success_at": "2026-08-11T18:45:00Z",
        "stale": False,
        "diagnostics": {
            "release_id": release_id,
            "source_as_of": "2026-08-11T18:00:00Z",
            "refresh_status": "idle",
            "worker_commit": worker,
        },
    }

    def health(_url: str, **kwargs):
        assert kwargs["headers"]["authorization"] == "Bearer diagnostic-secret"
        return json.dumps(payload).encode()

    with patch.object(
        public_refresh,
        "_http_bytes",
        side_effect=health,
    ):
        result = public_refresh.verify_public_health_alignment(
            config,
            release_id=release_id,
            source_as_of=source_as_of,
            expected_worker_commit=worker,
        )

    assert result == payload

    payload["diagnostics"]["worker_commit"] = "b" * 40
    with patch.object(
        public_refresh,
        "_http_bytes",
        return_value=json.dumps(payload).encode(),
    ), pytest.raises(public_refresh.PublicRefreshError, match="worker_commit"):
        public_refresh.verify_public_health_alignment(
            config,
            release_id=release_id,
            source_as_of=source_as_of,
            expected_worker_commit=worker,
        )


def test_local_release_retention_keeps_active_and_two_previous(tmp_path: Path) -> None:
    public_root = tmp_path / "packs"
    public_root.mkdir()
    releases = [f"v2026.08.{day:02d}.120000" for day in range(1, 6)]
    for release_id in releases:
        (public_root / release_id).mkdir()
    (public_root / "manual-notes").mkdir()

    removed = public_refresh.prune_local_releases(public_root, releases[-1])

    assert removed == releases[:2]
    assert sorted(path.name for path in public_root.iterdir()) == [
        "manual-notes",
        *releases[2:],
    ]


def test_production_smoke_requires_the_deployed_app_to_serve_the_new_pack(tmp_path: Path) -> None:
    raw = b"data"
    config = replace(
        _config(tmp_path),
        production=True,
        manifest_url="https://store-test.public.blob.vercel-storage.com/packs/manifest.json",
        blob_root="https://store-test.public.blob.vercel-storage.com",
    )
    manifest = {
        "pack_id": "new",
        "base_url": "https://store-test.public.blob.vercel-storage.com/packs/new",
        "total_files": 1,
        "total_bytes": len(raw),
        "files": [{"path": "features/a.json", "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}],
    }

    def http(url: str, **_kwargs):
        if url.endswith("/api/health"):
            return json.dumps({"status": "ok", "pack_id": "old", "tier": {"status": "available"}}).encode()
        if url.endswith("/rankings/tierlists.json"):
            return b'{"status":"available"}'
        if "/features/" in url:
            return raw
        return b"page"

    with patch.object(public_refresh, "_load_remote_manifest", return_value=(manifest, manifest["base_url"])), patch.object(
        public_refresh, "_http_bytes", side_effect=http
    ):
        with pytest.raises(public_refresh.PublicRefreshError, match="serves old"):
            public_refresh.verify_public_release(config, expected_pack_id="new", tier_expected=True)


def test_supabase_smoke_accepts_a_storage_backed_tier_asset(tmp_path: Path) -> None:
    raw = b"data"
    project_url = "https://example.supabase.co"
    config = replace(
        _config(tmp_path),
        production=True,
        publication_backend="supabase",
        supabase_url=project_url,
        supabase_secret_key="sb_secret_abcdefghijklmnopqrstuvwxyz",
    )
    manifest = {
        "pack_id": "v2026.08.10.001500",
        "base_url": project_url,
        "data_backend": "supabase",
        "tier": {"status": "available", "as_of": "2026-08-07T22:05:18Z"},
        "total_files": 1,
        "total_bytes": len(raw),
        "files": [
            {
                "path": "features/a.json",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        ],
    }

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def asset(self, _release_id: str, path: str):
            if path == "rankings/tierlists.json":
                return {
                    "body": None,
                    "storage_path": "v2026.08.10.001500/rankings/tierlists.json",
                }
            return {
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }

    def http(url: str, **_kwargs):
        if url.endswith("/api/health"):
            return json.dumps(
                {
                    "status": "ok",
                    "diagnostics": {
                        "release_id": manifest["pack_id"],
                        "tier": {"status": "available"},
                    },
                }
            ).encode()
        return b"page"

    with patch.object(
        public_refresh,
        "_load_remote_manifest",
        return_value=(manifest, project_url),
    ), patch.object(
        public_refresh.supabase_publication,
        "SupabasePublicData",
        Client,
    ), patch.object(public_refresh, "_http_bytes", side_effect=http):
        result = public_refresh.verify_public_release(
            config,
            expected_pack_id=manifest["pack_id"],
            tier_expected=True,
        )

    assert result["tier_status"] == "available"


def test_systemd_worker_cannot_start_without_production_environment() -> None:
    root = Path(__file__).parents[1]
    service = (root / "ops/systemd/scryglass-ratings-sync.service").read_text(encoding="utf-8")
    alert = (root / "ops/systemd/scryglass-public-refresh-alert@.service").read_text(encoding="utf-8")
    watchdog = (root / "ops/systemd/scryglass-public-refresh-watchdog.service").read_text(encoding="utf-8")
    public_env = (root / "ops/systemd/public-refresh.env.example").read_text(encoding="utf-8")

    assert "EnvironmentFile=/etc/scryglass/public-refresh.env" in service
    assert "Environment=SCRYGLASS_PUBLIC_RELEASE=1" in service
    assert "EnvironmentFile=-/etc/scryglass/public-refresh.env" not in service
    assert "EnvironmentFile=/etc/scryglass/public-refresh.env" in alert
    assert "EnvironmentFile=/etc/scryglass/public-refresh.env" in watchdog
    assert "ORACLES_ELIXIR_API_KEY=" not in public_env
    assert "SCRYGLASS_ALERT_WEBHOOK_URL=" not in public_env
    assert not (root / "ops/systemd/postgame-sync.env.example").exists()

    launchd = (root / "ops/launchd/run-public-refresh.sh").read_text(encoding="utf-8")
    assert launchd.count('/usr/bin/open -a "Brave Origin"') == 1
    assert "/usr/bin/shlock" in launchd
    assert 'SCRYGLASS_RUNTIME_ROOT="${runtime_root}"' in launchd
    assert '--source-receipt "${source_receipt}"' in launchd
    assert '--result-output "${import_receipt}"' in launchd
    assert '"${repo_root}/data/lol/warehouse' not in launchd
    assert 'runtime/cycles/${cycle_id}' in launchd
    assert "--validate-only" in launchd
    assert launchd.index('cd "${repo_root}"') < launchd.index('"${python}" -m lol_kills.etl.oe_database')
    assert "status --porcelain=v1 --untracked-files=normal" in launchd
    assert '"${SCRYGLASS_WORKER_COMMIT}" != "${real_worker_commit}"' in launchd
    assert 'verify-public-refresh-env.sh" "${repo_root}" "${worker_root}/venv"' in launchd
    assert "ExecStartPre=/srv/scryglass/current/ops/verify-public-refresh-env.sh" in service
    installer = (root / "ops/install-public-refresh-env.sh").read_text(encoding="utf-8")
    verifier = (root / "ops/verify-public-refresh-env.sh").read_text(encoding="utf-8")
    assert "--require-hashes" in installer
    assert "--only-binary=:all:" in installer
    assert ".scryglass-requirements-lock.sha256" in installer
    assert "pip check" in verifier
    plist = (
        root / "ops/launchd/xyz.scryglass.public-refresh.plist.template"
    ).read_text(encoding="utf-8")
    assert "__TESTED_WORKER_COMMIT__" in plist
