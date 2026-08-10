"""Run the unattended six-hour Oracle's Elixir public refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from lol_kills.etl.oe_api_ingest import OeApiIngestError
from lol_kills.postgame_sync import (
    SyncConfig,
    _load_json,
    _iso,
    _atomic_json,
    exclusive_lock,
    rollback_public_pack,
    sync_once,
)
from lol_kills.v2.tierlists import live_refresh


SCHEMA_VERSION = "scryglass:public-refresh:v1"
HEALTH_FILE = "public-refresh-health.json"
STATE_FILE = "public-refresh.json"
LOCK_FILE = "public-refresh.lock"
DEFAULT_SITE = "https://scryglass.xyz"
DEFAULT_ATTEMPTS = 3
DEFAULT_STALE_AFTER_HOURS = 12
RETRYABLE_ERRORS = (OeApiIngestError, TimeoutError, urllib.error.URLError)
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class PublicRefreshError(RuntimeError):
    """The public release cannot prove a complete end-to-end refresh."""


class PublicRefreshHttpError(PublicRefreshError):
    """A public refresh HTTP request failed, with a safe retry hint."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class RefreshConfig:
    root: Path
    public_root: Path
    site: str
    manifest_url: str | None
    blob_root: str | None
    health_path: Path
    state_path: Path
    lock_path: Path
    production: bool
    attempts: int = DEFAULT_ATTEMPTS

    @property
    def sync(self) -> SyncConfig:
        return SyncConfig(
            root=self.root,
            public_root=self.public_root,
            output_root=self.root / "output/public_pack",
            state_path=self.root / "data/lol/runtime/postgame-sync.json",
            lock_path=self.lock_path,
            health_path=self.root / "data/lol/runtime/postgame-sync-health.json",
        )


def _read_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def config_from_environment(root: Path, public_root: Path) -> RefreshConfig:
    site = (_read_env("SCRYGLASS_PUBLISH_ORIGIN") or DEFAULT_SITE).rstrip("/")
    blob_root = (_read_env("LIVE_BLOB_BASE_URL") or _read_env("SCRYGLASS_TIERLIST_BLOB_BASE_URL"))
    manifest_url = _read_env("SCRYGLASS_PACK_MANIFEST_URL")
    if manifest_url is None and blob_root:
        manifest_url = f"{blob_root.rstrip('/')}/packs/manifest.json"
    runtime = root / "data/lol/runtime"
    try:
        attempts = int(_read_env("SCRYGLASS_REFRESH_ATTEMPTS") or DEFAULT_ATTEMPTS)
    except ValueError as error:
        raise PublicRefreshError("SCRYGLASS_REFRESH_ATTEMPTS must be an integer") from error
    if attempts < 1 or attempts > 5:
        raise PublicRefreshError("SCRYGLASS_REFRESH_ATTEMPTS must be between one and five")
    return RefreshConfig(
        root=root,
        public_root=public_root,
        site=site,
        manifest_url=manifest_url,
        blob_root=blob_root.rstrip("/") if blob_root else None,
        health_path=runtime / HEALTH_FILE,
        state_path=runtime / STATE_FILE,
        lock_path=runtime / LOCK_FILE,
        production=_read_env("SCRYGLASS_PUBLIC_RELEASE") == "1",
        attempts=attempts,
    )


def _write_health(config: RefreshConfig, status: str, checked_at: datetime, **fields: Any) -> None:
    previous = _load_json(config.health_path)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "checked_at": _iso(checked_at),
        "last_success_at": previous.get("last_success_at"),
        "stale_alert_sent_at": previous.get("stale_alert_sent_at"),
        **fields,
    }
    if status not in {"checking", "error", "partial"}:
        payload["last_success_at"] = payload["checked_at"]
        payload["stale_alert_sent_at"] = None
    if status == "error" and previous.get("checked_at") == payload["checked_at"]:
        payload["alert_sent_at"] = previous.get("alert_sent_at")
    _atomic_json(config.health_path, payload)


def _preflight(config: RefreshConfig) -> None:
    if not config.production:
        return
    required = {
        "ORACLES_ELIXIR_API_KEY": _read_env("ORACLES_ELIXIR_API_KEY") or _read_env("OE_API_KEY"),
        "BLOB_READ_WRITE_TOKEN": _read_env("BLOB_READ_WRITE_TOKEN") or _read_env("VERCEL_BLOB_READ_WRITE_TOKEN"),
        "SCRYGLASS_DATA_PUBLISH_TOKEN": _read_env("SCRYGLASS_DATA_PUBLISH_TOKEN"),
        "SCRYGLASS_ALERT_WEBHOOK_URL": _read_env("SCRYGLASS_ALERT_WEBHOOK_URL"),
        "LIVE_BLOB_BASE_URL": config.blob_root,
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise PublicRefreshError("public refresh credentials are incomplete: " + ", ".join(missing))
    if not config.site.startswith("https://"):
        raise PublicRefreshError("SCRYGLASS_PUBLISH_ORIGIN must use HTTPS")
    if not config.blob_root or ".public.blob.vercel-storage.com" not in config.blob_root:
        raise PublicRefreshError("LIVE_BLOB_BASE_URL must be a public Blob root")


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(60.0, 2.0**attempt))


def _run_with_source_retries(config: RefreshConfig, now: datetime, *, force: bool) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(config.attempts):
        try:
            return sync_once(config.sync, now=now, force=force)
        except RETRYABLE_ERRORS as error:
            last_error = error
            if attempt + 1 >= config.attempts:
                raise
            _sleep_before_retry(attempt)
    raise PublicRefreshError("ratings refresh stopped without a result") from last_error


def _run_tier_refresh(config: RefreshConfig, expected_live_as_of: str) -> dict[str, Any]:
    previous = live_refresh.DEFAULT_OUTPUT
    previous_path = config.root / previous if (config.root / previous).is_file() else None
    receipt = live_refresh.refresh_candidate(
        config.root,
        expected_live_as_of=expected_live_as_of,
        previous_path=previous_path,
        source_mode="oe_only",
        promote=True,
        skip_annual_oe=True,
        skip_atom_bridge=True,
    )
    if receipt.get("status") != "production_promoted":
        raise PublicRefreshError(
            "tier-list refresh did not promote: " + str(receipt.get("status") or "unknown")
        )
    return receipt


def _http_bytes(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 1,
) -> bytes:
    if attempts < 1 or attempts > 5:
        raise PublicRefreshError("HTTP attempts must be between one and five")
    request = urllib.request.Request(url, method=method, data=body, headers=headers or {})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_STATUS or attempt + 1 >= attempts:
                raise PublicRefreshHttpError(
                    f"HTTP {error.code} from {url}", status=error.code
                ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            if attempt + 1 >= attempts:
                raise PublicRefreshHttpError(f"HTTP request failed for {url}") from error
        _sleep_before_retry(attempt)
    raise PublicRefreshError("HTTP request stopped without a result")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_remote_manifest(config: RefreshConfig) -> tuple[dict[str, Any], str]:
    if config.manifest_url:
        raw = _http_bytes(
            config.manifest_url,
            headers={"Cache-Control": "no-cache"},
            attempts=config.attempts,
        )
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicRefreshError("public pack manifest is invalid JSON") from error
        if not isinstance(manifest, dict):
            raise PublicRefreshError("public pack manifest is not an object")
        return manifest, str(manifest.get("base_url") or "").rstrip("/")
    manifest = _load_json(config.public_root / "manifest.json")
    if not manifest:
        raise PublicRefreshError("local public pack manifest is missing")
    return manifest, str(manifest.get("base_url") or "").rstrip("/")


def verify_public_release(config: RefreshConfig, *, expected_pack_id: str | None = None, tier_expected: bool = False) -> dict[str, Any]:
    manifest, base_url = _load_remote_manifest(config)
    pack_id = str(manifest.get("pack_id") or "")
    if not pack_id or not base_url:
        raise PublicRefreshError("public pack manifest has no active pack and base URL")
    if config.production and (
        not base_url.startswith("https://")
        or ".public.blob.vercel-storage.com" not in base_url
    ):
        raise PublicRefreshError("public pack base URL is not a public HTTPS Blob URL")
    if expected_pack_id and pack_id != expected_pack_id:
        raise PublicRefreshError(f"public pack pointer is {pack_id}, expected {expected_pack_id}")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PublicRefreshError("public pack manifest has no file inventory")
    if manifest.get("total_files") != len(files):
        raise PublicRefreshError("public pack manifest file total is invalid")
    checked_files = 0
    total_bytes = 0
    local = not config.manifest_url
    for item in files:
        if not isinstance(item, dict):
            raise PublicRefreshError("public pack manifest contains a malformed file entry")
        relative = str(item.get("path") or "")
        relative_path = Path(relative)
        if (
            not relative
            or relative_path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.parts)
        ):
            raise PublicRefreshError("public pack manifest contains an unsafe file path")
        if local:
            path = (config.public_root / pack_id / relative_path).resolve()
            try:
                path.relative_to((config.public_root / pack_id).resolve())
            except ValueError as error:
                raise PublicRefreshError("public pack manifest path leaves the pack") from error
            try:
                raw = path.read_bytes()
            except OSError as error:
                raise PublicRefreshError(f"local public pack file is missing: {relative}") from error
        else:
            raw = _http_bytes(f"{base_url}/{relative}", attempts=config.attempts)
        if len(raw) != int(item.get("bytes", -1)) or _sha256(raw) != item.get("sha256"):
            raise PublicRefreshError(f"public pack checksum failed: {relative}")
        checked_files += 1
        total_bytes += len(raw)
    if manifest.get("total_bytes") != total_bytes:
        raise PublicRefreshError("public pack manifest byte total is invalid")

    if config.production:
        for route in ("/elo", "/matches", "/tiers"):
            _http_bytes(f"{config.site}{route}", attempts=config.attempts)
        health_raw = _http_bytes(f"{config.site}/api/health", attempts=config.attempts)
        try:
            health_payload = json.loads(health_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicRefreshError("public health response is invalid JSON") from error
        if not isinstance(health_payload, dict) or health_payload.get("status") not in {"ok", "partial"}:
            raise PublicRefreshError("public health response is not ready")
        served_pack_id = str(health_payload.get("pack_id") or "")
        if not served_pack_id:
            raise PublicRefreshError("public health response has no pack ID")
        if expected_pack_id and served_pack_id != expected_pack_id:
            raise PublicRefreshError(
                f"public app serves {served_pack_id}, expected {expected_pack_id}"
            )
        if tier_expected:
            served_tier = health_payload.get("tier")
            if not isinstance(served_tier, dict) or served_tier.get("status") != "available":
                raise PublicRefreshError("public health response has no available tier list")

    tier_status = None
    if config.production and config.blob_root:
        tier_raw = _http_bytes(
            f"{config.blob_root}/rankings/tierlists.json",
            attempts=config.attempts,
        )
        try:
            tier_payload = json.loads(tier_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicRefreshError("public tier-list display is invalid JSON") from error
        tier_status = tier_payload.get("status") if isinstance(tier_payload, dict) else None
        if tier_expected and tier_status != "available":
            raise PublicRefreshError("public tier-list display is not available")

    return {"pack_id": pack_id, "files": checked_files, "tier_status": tier_status}


def invalidate_public_cache(config: RefreshConfig) -> dict[str, Any]:
    secret = _read_env("SCRYGLASS_DATA_PUBLISH_TOKEN")
    if not secret:
        raise PublicRefreshError("SCRYGLASS_DATA_PUBLISH_TOKEN is required for cache invalidation")
    body = _http_bytes(
        f"{config.site}/api/data-published",
        method="POST",
        headers={"authorization": f"Bearer {secret}"},
        attempts=config.attempts,
    )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicRefreshError("cache invalidation response is invalid JSON") from error
    if not isinstance(payload, dict) or payload.get("revalidated") is not True:
        raise PublicRefreshError("cache invalidation was not confirmed")
    return payload


def notify_health(config: RefreshConfig, *, failure_unit: str | None = None) -> bool:
    health = _load_json(config.health_path)
    if health.get("status") != "error":
        return False
    if health.get("alert_sent_at") == health.get("checked_at"):
        return True
    url = _read_env("SCRYGLASS_ALERT_WEBHOOK_URL")
    if not url:
        return False
    payload = _alert_payload(
        "Scryglass public refresh failed",
        health,
        unit=failure_unit,
    )
    _http_bytes(
        url,
        method="POST",
        body=(json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"),
        headers={"Content-Type": "application/json"},
        attempts=config.attempts,
    )
    _atomic_json(config.health_path, {**health, "alert_sent_at": health.get("checked_at")})
    return True


def _alert_payload(text: str, health: dict[str, Any], *, unit: str | None = None) -> dict[str, Any]:
    return {
        "text": text,
        "status": health.get("status"),
        "checked_at": health.get("checked_at"),
        "last_success_at": health.get("last_success_at"),
        "reason": health.get("reason"),
        "unit": unit,
    }


def notify_watchdog(config: RefreshConfig, *, now: datetime | None = None) -> bool:
    health = _load_json(config.health_path)
    current = now or datetime.now(timezone.utc)
    raw_hours = _read_env("SCRYGLASS_STALE_AFTER_HOURS") or str(DEFAULT_STALE_AFTER_HOURS)
    try:
        stale_after = float(raw_hours)
    except ValueError as error:
        raise PublicRefreshError("SCRYGLASS_STALE_AFTER_HOURS must be numeric") from error
    if stale_after <= 0:
        raise PublicRefreshError("SCRYGLASS_STALE_AFTER_HOURS must be positive")
    last_success = health.get("last_success_at")
    try:
        last = datetime.fromisoformat(str(last_success).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        last = None
    age_hours = float("inf") if last is None else (current - last).total_seconds() / 3600
    if age_hours <= stale_after:
        return False
    marker = str(last_success or "never")
    if health.get("stale_alert_sent_at") == marker:
        return True
    url = _read_env("SCRYGLASS_ALERT_WEBHOOK_URL")
    if not url:
        return False
    payload = _alert_payload("Scryglass public refresh is stale", health)
    payload.update({"stale_after_hours": stale_after, "age_hours": age_hours})
    _http_bytes(
        url,
        method="POST",
        body=(json.dumps(payload, ensure_ascii=True) + "\n").encode("utf-8"),
        headers={"Content-Type": "application/json"},
        attempts=config.attempts,
    )
    _atomic_json(config.health_path, {**health, "stale_alert_sent_at": marker})
    return True


def run_once(config: RefreshConfig, *, now: datetime | None = None, force: bool = False) -> dict[str, Any]:
    checked_at = now or datetime.now(timezone.utc)
    previous_state = _load_json(config.sync.state_path)
    previous_public_state = _load_json(config.state_path)
    publication: dict[str, Any] | None = None
    post_publication_verified = False
    failure_stage = "preflight"
    _write_health(config, "checking", checked_at)
    try:
        _preflight(config)
        failure_stage = "ratings"
        ratings = _run_with_source_retries(config, checked_at, force=force)
        tier: dict[str, Any] | None = None
        tier_error: str | None = None
        should_run_tier = bool(
            ratings.get("status") == "published"
            or force
            or not previous_public_state.get("tier")
            or previous_public_state.get("tier_error")
        )
        if should_run_tier:
            failure_stage = "tier"
            expected = str(ratings.get("source_observed_through") or _iso(checked_at))
            try:
                tier = _run_tier_refresh(config, expected)
            except Exception as error:  # noqa: BLE001
                tier_error = f"{type(error).__name__}: {str(error)[:500]}"
        publication = ratings.get("publication") if isinstance(ratings.get("publication"), dict) else None
        changed = ratings.get("status") == "published" or tier is not None
        failure_stage = "cache"
        cache = invalidate_public_cache(config) if changed else None
        failure_stage = "smoke"
        smoke = verify_public_release(
            config,
            expected_pack_id=str(ratings.get("pack_id")) if ratings.get("status") == "published" else None,
            tier_expected=tier is not None,
        )
        post_publication_verified = True
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "partial" if tier_error else str(ratings.get("status") or "ok"),
            "checked_at": _iso(checked_at),
            "ratings": ratings,
            "tier": tier,
            "tier_error": tier_error,
            "cache_invalidation": cache,
            "smoke": smoke,
        }
        _atomic_json(config.state_path, result)
        _write_health(config, result["status"], checked_at, pack_id=smoke["pack_id"], tier_error=tier_error)
        if tier_error:
            raise PublicRefreshError(tier_error)
        return result
    except Exception as error:
        if publication is not None and not post_publication_verified and failure_stage == "smoke":
            try:
                rollback = rollback_public_pack(publication, config.public_root)
                _atomic_json(config.sync.state_path, previous_state)
            except Exception as rollback_error:  # noqa: BLE001
                rollback = {"status": "failed", "reason": f"{type(rollback_error).__name__}: {rollback_error}"}
            else:
                rollback = {"status": "restored", **rollback}
        else:
            rollback = None
        _write_health(
            config,
            "error",
            checked_at,
            pack_id=previous_state.get("pack_id"),
            reason=f"{type(error).__name__}: {str(error)[:500]}",
            rollback=rollback,
        )
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public-root", type=Path, default=Path("/srv/scryglass-data/public-packs"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--notify-health", action="store_true")
    parser.add_argument("--watchdog", action="store_true")
    parser.add_argument("--failure-unit", default=None)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = config_from_environment(root, args.public_root.resolve())
    if args.notify_health:
        notify_health(config, failure_unit=args.failure_unit)
        return 0
    if args.watchdog:
        notify_watchdog(config)
        return 0
    with exclusive_lock(config.lock_path):
        result = run_once(config, force=args.force)
    print(json.dumps(result, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
