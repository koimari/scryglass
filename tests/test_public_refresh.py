from __future__ import annotations

import json
import hashlib
import urllib.error
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from lol_kills import public_refresh
from lol_kills.postgame_sync import RefreshValidationError, validate_public_identity


NOW = datetime(2026, 8, 10, 0, 0, tzinfo=timezone.utc)


def _config(tmp_path: Path) -> public_refresh.RefreshConfig:
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    return public_refresh.RefreshConfig(
        root=tmp_path,
        public_root=tmp_path / "packs",
        site="https://example.test",
        manifest_url=None,
        blob_root=None,
        health_path=runtime / "health.json",
        state_path=runtime / "public-state.json",
        lock_path=runtime / "lock",
        production=False,
    )


def test_tier_failure_keeps_a_smoke_verified_ratings_release(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.sync.state_path.parent.mkdir(parents=True, exist_ok=True)
    config.sync.state_path.write_text(json.dumps({"pack_id": "old"}), encoding="utf-8")
    ratings = {
        "status": "published",
        "pack_id": "new",
        "publication": {"runtime": "blob", "pack_id": "new"},
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
        "publication": {"runtime": "blob", "pack_id": "new"},
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
        "publication": {"runtime": "blob", "pack_id": "new"},
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


def test_production_preflight_uses_public_release_credentials_only(tmp_path: Path) -> None:
    base = "https://store-test.public.blob.vercel-storage.com"
    config = replace(
        _config(tmp_path),
        production=True,
        blob_root=base,
        manifest_url=f"{base}/packs/manifest.json",
    )
    values = {
        "BLOB_READ_WRITE_TOKEN": "blob-key",
        "SCRYGLASS_DATA_PUBLISH_TOKEN": "publish-key",
    }
    with patch.dict("os.environ", values, clear=True):
        public_refresh._preflight(config)


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
