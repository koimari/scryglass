"""Run the unattended six-hour Oracle's Elixir public refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.etl.oe_ingest import OeDownloadError
from lol_kills.etl.oe_ingest import load_refresh_receipt
from lol_kills.etl.oe_database import TRANSFORM_VERSION
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.export import supabase_publication
from lol_kills.export.public_pack import source_identity_sha256
from lol_kills.postgame_sync import (
    RefreshValidationError,
    SyncConfig,
    _load_json,
    _iso,
    _atomic_json,
    _manifest_window_years,
    _year_filtered_ids,
    exclusive_lock,
    rollback_public_pack,
    sync_once,
    validate_live_source,
)
from lol_kills.v2.tierlists import live_refresh
from lol_kills.refresh_ledger import (
    RefreshRunLedger,
    requirements_lock_sha256,
    worker_commit,
)


SCHEMA_VERSION = "scryglass:public-refresh:v1"
HEALTH_FILE = "public-refresh-health.json"
STATE_FILE = "public-refresh.json"
LOCK_FILE = "public-refresh.lock"
DEFAULT_SITE = "https://scryglass.xyz"
DEFAULT_ATTEMPTS = 3
DEFAULT_STALE_AFTER_HOURS = 12
DEFAULT_STEP_TIMEOUT_MINUTES = 30
RETRYABLE_ERRORS = (OeDownloadError, TimeoutError, urllib.error.URLError)
RETRYABLE_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})
PUBLIC_RELEASE_PROBES = (
    "/",
    "/elo",
    "/matches",
    "/tiers",
    "/chat",
    "/support",
    "/methodology",
    "/sources",
    "/privacy",
    "/legal",
    "/security",
    "/packs/manifest.json",
    "/api/chat/navigation",
    "/api/chat/leaderboards?category=rating&limit=1",
    "/api/chat/leaderboards?category=teams&limit=1",
    "/api/chat/leaderboards?category=teams_draft&limit=1",
    "/api/chat/player?name=Faker",
    "/api/chat/team?name=T1",
    "/api/chat/compare_players?player1=Faker&player2=Chovy",
    "/api/chat/matches?team=T1&limit=1",
    "/api/chat/query_players?q=who+has+the+better+rating+between+Faker+and+Chovy",
    "/api/chat/query_champions?q=who+is+the+best+Galio+player",
    "/api/chat/query_drafts?q=which+team+has+the+best+draft",
    "/api/chat/schedule?limit=1",
    "/api/chat/tier?limit=1",
    "/api/chat/methodology?topic=ratings",
)
RELEASE_BOUND_PAGE_PROBES = frozenset({"/elo", "/matches", "/tiers"})


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
    runtime_root: Path
    public_root: Path
    site: str
    manifest_url: str | None
    blob_root: str | None
    health_path: Path
    state_path: Path
    lock_path: Path
    production: bool
    publication_backend: str
    supabase_url: str | None
    supabase_secret_key: str | None
    accepted_source_receipt: Path | None = None
    accepted_import_receipt: Path | None = None
    attempts: int = DEFAULT_ATTEMPTS
    step_timeout_seconds: float = DEFAULT_STEP_TIMEOUT_MINUTES * 60

    @property
    def sync(self) -> SyncConfig:
        return SyncConfig(
            root=self.root,
            public_root=self.public_root,
            output_root=self.runtime_root / "output/public_pack",
            state_path=self.runtime_root / "data/lol/runtime/postgame-sync.json",
            lock_path=self.lock_path,
            health_path=self.runtime_root / "data/lol/runtime/postgame-sync-health.json",
            runtime_root=self.runtime_root,
            accepted_source_receipt=self.accepted_source_receipt,
            accepted_import_receipt=self.accepted_import_receipt,
        )


def _read_env(name: str) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or None


def config_from_environment(root: Path, public_root: Path) -> RefreshConfig:
    site = _read_env("SCRYGLASS_PUBLISH_ORIGIN") or DEFAULT_SITE
    blob_root = (_read_env("LIVE_BLOB_BASE_URL") or _read_env("SCRYGLASS_TIERLIST_BLOB_BASE_URL"))
    manifest_url = _read_env("SCRYGLASS_PACK_MANIFEST_URL")
    if manifest_url is None and blob_root:
        manifest_url = f"{blob_root.rstrip('/')}/packs/manifest.json"
    runtime_root = Path(_read_env("SCRYGLASS_RUNTIME_ROOT") or root).expanduser().resolve()
    runtime = runtime_root / "data/lol/runtime"
    source_receipt_raw = _read_env("SCRYGLASS_ACCEPTED_SOURCE_RECEIPT")
    import_receipt_raw = _read_env("SCRYGLASS_ACCEPTED_IMPORT_RECEIPT")
    publication_backend = (_read_env("SCRYGLASS_PUBLICATION_BACKEND") or "supabase").lower()
    if publication_backend not in {"blob", "supabase"}:
        raise PublicRefreshError("SCRYGLASS_PUBLICATION_BACKEND must be blob or supabase")
    try:
        attempts = int(_read_env("SCRYGLASS_REFRESH_ATTEMPTS") or DEFAULT_ATTEMPTS)
    except ValueError as error:
        raise PublicRefreshError("SCRYGLASS_REFRESH_ATTEMPTS must be an integer") from error
    if attempts < 1 or attempts > 5:
        raise PublicRefreshError("SCRYGLASS_REFRESH_ATTEMPTS must be between one and five")
    try:
        step_timeout_minutes = float(
            _read_env("SCRYGLASS_STEP_TIMEOUT_MINUTES") or DEFAULT_STEP_TIMEOUT_MINUTES
        )
    except ValueError as error:
        raise PublicRefreshError("SCRYGLASS_STEP_TIMEOUT_MINUTES must be numeric") from error
    if not math.isfinite(step_timeout_minutes) or step_timeout_minutes <= 0 or step_timeout_minutes > 55:
        raise PublicRefreshError(
            "SCRYGLASS_STEP_TIMEOUT_MINUTES must be greater than zero and at most 55"
        )
    return RefreshConfig(
        root=root,
        runtime_root=runtime_root,
        public_root=public_root,
        site=site,
        manifest_url=manifest_url,
        blob_root=blob_root.rstrip("/") if blob_root else None,
        health_path=runtime / HEALTH_FILE,
        state_path=runtime / STATE_FILE,
        lock_path=runtime / LOCK_FILE,
        production=_read_env("SCRYGLASS_PUBLIC_RELEASE") == "1",
        publication_backend=publication_backend,
        supabase_url=_read_env("SCRYGLASS_SUPABASE_URL") or _read_env("SUPABASE_URL"),
        supabase_secret_key=(
            _read_env("SCRYGLASS_SUPABASE_SECRET_KEY")
            or _read_env("SUPABASE_SECRET_KEY")
        ),
        accepted_source_receipt=(
            Path(source_receipt_raw).expanduser().resolve()
            if source_receipt_raw
            else None
        ),
        accepted_import_receipt=(
            Path(import_receipt_raw).expanduser().resolve()
            if import_receipt_raw
            else None
        ),
        attempts=attempts,
        step_timeout_seconds=step_timeout_minutes * 60,
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
    if config.publication_backend != "supabase":
        raise PublicRefreshError("production refresh requires the Supabase publication backend")
    required = {
        "SCRYGLASS_DATA_PUBLISH_TOKEN": _read_env("SCRYGLASS_DATA_PUBLISH_TOKEN"),
        "SCRYGLASS_DIAGNOSTIC_TOKEN": _read_env("SCRYGLASS_DIAGNOSTIC_TOKEN"),
    }
    required.update(
        {
            "SCRYGLASS_SUPABASE_URL": config.supabase_url,
            "SCRYGLASS_SUPABASE_SECRET_KEY": config.supabase_secret_key,
        }
    )
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        raise PublicRefreshError("public refresh credentials are incomplete: " + ", ".join(missing))
    parsed_site = urllib.parse.urlsplit(config.site)
    if (
        parsed_site.scheme != "https"
        or parsed_site.hostname != "scryglass.xyz"
        or parsed_site.port is not None
        or parsed_site.username is not None
        or parsed_site.password is not None
        or parsed_site.path
        or parsed_site.query
        or parsed_site.fragment
        or config.site != DEFAULT_SITE
    ):
        raise PublicRefreshError(
            "SCRYGLASS_PUBLISH_ORIGIN must be exactly https://scryglass.xyz"
        )
    try:
        supabase_publication.SupabasePublicData(
            config.supabase_url or "",
            config.supabase_secret_key or "",
        )
    except ValueError as error:
        raise PublicRefreshError(str(error)) from error
    if config.accepted_source_receipt is None or config.accepted_import_receipt is None:
        raise PublicRefreshError("production refresh requires accepted source and import receipts")
    if not config.accepted_source_receipt.is_file() or not config.accepted_import_receipt.is_file():
        raise PublicRefreshError("accepted source or import receipt is missing")
    try:
        worker_commit(config.root, require_clean=True)
    except RuntimeError as error:
        raise PublicRefreshError(str(error)) from error


def _accepted_run_inputs(config: RefreshConfig) -> dict[str, Any] | None:
    if config.accepted_source_receipt is None or config.accepted_import_receipt is None:
        return None
    source = load_refresh_receipt(config.accepted_source_receipt)
    try:
        imported = json.loads(config.accepted_import_receipt.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicRefreshError("accepted import receipt is invalid") from error
    if not isinstance(imported, dict):
        raise PublicRefreshError("accepted import receipt is not an object")
    candidate = source.get("candidate")
    imported_source = imported.get("accepted_source_receipt")
    if not isinstance(candidate, dict) or not isinstance(imported_source, dict):
        raise PublicRefreshError("accepted source binding is missing")
    source_sha256 = str(candidate.get("raw_sha256") or "")
    if (
        imported.get("source_file_sha256") != source_sha256
        or imported_source.get("raw_sha256") != source_sha256
        or imported_source.get("receipt_canonical_sha256")
        != source.get("receipt_canonical_sha256")
    ):
        raise PublicRefreshError("accepted import receipt does not match the source receipt")
    expected_commit = worker_commit(config.root)
    if imported.get("worker_commit") != expected_commit:
        raise PublicRefreshError("accepted import receipt belongs to a different worker commit")
    return {
        "source_file_sha256": source_sha256,
        "source_observed_through": imported.get("source_observed_through"),
        "accepted_games": int(imported.get("accepted_games", 0)),
        "new_games": int(imported.get("new_games", 0)),
        "corrected_games": int(imported.get("corrected_games", 0)),
        "unchanged_games": int(imported.get("unchanged_games", 0)),
        "quarantined_games": int(imported.get("quarantined_games", 0)),
        "source_receipt_sha256": source.get("receipt_canonical_sha256"),
        "import_status": imported.get("status"),
        "worker_commit": expected_commit,
        "source_bytes": int((source.get("candidate") or {}).get("bytes", candidate.get("bytes", 0))),
        "source_rows": int((source.get("candidate") or {}).get("row_count", candidate.get("row_count", 0))),
        "source_games": int(imported.get("source_games", 0)),
        "statistics_complete_games": int(imported.get("statistics_complete_games", 0)),
    }


def _start_run_ledger(
    config: RefreshConfig,
    checked_at: datetime,
    accepted: dict[str, Any] | None,
) -> RefreshRunLedger | None:
    if accepted is None:
        return None
    remote_write = None
    if config.publication_backend == "supabase":
        client = supabase_publication.SupabasePublicData(
            config.supabase_url or "",
            config.supabase_secret_key or "",
        )
        remote_write = client.write_refresh_run
    counts = {
        key: int(accepted.get(key, 0))
        for key in (
            "accepted_games",
            "new_games",
            "corrected_games",
            "unchanged_games",
            "quarantined_games",
        )
    }
    return RefreshRunLedger(
        runtime_root=config.runtime_root / "data/lol/runtime",
        scheduled_for=checked_at,
        worker_git_commit=worker_commit(config.root),
        transform_version=TRANSFORM_VERSION,
        source_file_sha256=str(accepted["source_file_sha256"]),
        source_observed_through=(
            str(accepted["source_observed_through"])
            if accepted.get("source_observed_through")
            else None
        ),
        requirements_lock_sha256=requirements_lock_sha256(config.root),
        counts=counts,
        remote_write=remote_write,
    )


def _write_remote_health(
    config: RefreshConfig,
    *,
    checked_at: datetime,
    ledger: RefreshRunLedger | None,
    status: str,
    refresh_status: str,
    release_id: str | None,
    source_as_of: str | None,
) -> None:
    if ledger is None or config.publication_backend != "supabase":
        return
    client = supabase_publication.SupabasePublicData(
        config.supabase_url or "",
        config.supabase_secret_key or "",
    )
    diagnostic_token = _read_env("SCRYGLASS_DIAGNOSTIC_TOKEN")
    if not diagnostic_token:
        raise PublicRefreshError("SCRYGLASS_DIAGNOSTIC_TOKEN is required")
    client.write_diagnostic_credential(diagnostic_token)
    if not release_id:
        active = client.active_release()
        if isinstance(active, dict):
            release_id = str(active.get("release_id") or "") or None
    previous = _load_json(config.health_path)
    client.write_public_health(
        {
            "status": status,
            "refresh_status": refresh_status,
            "checked_at": _iso(checked_at),
            "last_success_at": (
                _iso(checked_at)
                if refresh_status == "idle"
                else previous.get("last_success_at")
            ),
            "source_as_of": source_as_of,
            "active_release_id": release_id,
            "last_run_id": ledger.run_id,
            "worker_commit": ledger.worker_git_commit,
            "stale": refresh_status == "stale",
        }
    )


def _supabase_asset_json(
    client: supabase_publication.SupabasePublicData,
    release_id: str,
    path: str,
) -> dict[str, Any]:
    asset = client.asset(release_id, path)
    if not asset:
        raise PublicRefreshError(f"Supabase bootstrap asset is missing: {path}")
    body = asset.get("body")
    if isinstance(body, dict):
        return body
    storage_path = asset.get("storage_path")
    if not isinstance(storage_path, str) or not storage_path:
        raise PublicRefreshError(f"Supabase bootstrap asset has no payload: {path}")
    try:
        payload = json.loads(client.storage_object(storage_path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicRefreshError(f"Supabase bootstrap asset is invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise PublicRefreshError(f"Supabase bootstrap asset is not an object: {path}")
    return payload


def seed_supabase_continuity(config: RefreshConfig) -> dict[str, Any] | None:
    """Seed a fresh worker with the exact source IDs behind the active release."""

    if config.publication_backend != "supabase":
        return None
    client = supabase_publication.SupabasePublicData(
        config.supabase_url or "",
        config.supabase_secret_key or "",
    )
    active = client.active_release()
    manifest = active.get("manifest") if isinstance(active, dict) else None
    if not isinstance(manifest, dict):
        raise PublicRefreshError("Supabase continuity bootstrap has no active manifest")
    release_id = str(manifest.get("pack_id") or "")
    ratings = manifest.get("ratings")
    if not release_id or not isinstance(ratings, dict):
        raise PublicRefreshError("Supabase continuity bootstrap manifest is malformed")
    try:
        expected_count = int(ratings["source_game_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise PublicRefreshError("Supabase continuity bootstrap has no game count") from error
    expected_digest = str(ratings.get("source_identity_sha256") or "")

    state = _load_json(config.sync.state_path)
    local_manifest = _load_json(config.public_root / "manifest.json")
    state_ids = state.get("published_game_ids")
    if (
        local_manifest.get("pack_id") == release_id
        and isinstance(state_ids, list)
        and len(state_ids) == expected_count
        and source_identity_sha256(state_ids) == expected_digest
    ):
        _atomic_json(config.public_root / "manifest.json", manifest)
        return {"status": "current", "pack_id": release_id, "game_count": expected_count}

    profiles = _supabase_asset_json(
        client,
        release_id,
        "features/profile_records.json",
    )
    games = profiles.get("games")
    if not isinstance(games, dict):
        raise PublicRefreshError("Supabase continuity bootstrap has no game index")
    game_ids = sorted(str(game_id) for game_id in games)
    source = "profile_index"
    out_of_window_ids: list[str] = []
    if len(game_ids) != expected_count or source_identity_sha256(game_ids) != expected_digest:
        # The window restriction has to happen while the candidate ids are being
        # read, not after. A contaminated cache is the exact state this bootstrap
        # exists to recover from, so validating the whole cache strictly here
        # would abort before the recovery below could ever discard the
        # out-of-window rows. Exclusion is safe because nothing is seeded until
        # the surviving set reproduces the published binding digest exactly, and
        # the excluded identities are recorded for audit.
        local_source = validate_live_source(
            config.runtime_root,
            [],
            years=config.sync.years,
            exclude_out_of_window=True,
        )
        local_ids = local_source.get("game_ids")
        if not isinstance(local_ids, list):
            raise PublicRefreshError("Supabase continuity bootstrap has no complete source index")
        excluded = local_source.get("out_of_window_game_ids")
        out_of_window_ids = sorted(str(value) for value in excluded) if excluded else []
        game_ids = sorted(str(game_id) for game_id in local_ids)
        source = "validated_local_cache"
    if len(game_ids) != expected_count or source_identity_sha256(game_ids) != expected_digest:
        # Recovery: the local worker can be ahead of the active release when a
        # local pack was published but Supabase activation was interrupted.
        # Verify that the active release's own published game index is fully
        # covered by the validated local source. The raw local superset is
        # never trusted directly — it can be contaminated with games outside
        # the release's declared publication window (the source of the
        # 2026-08 continuity incident) — so only a restriction of it to the
        # release's declared window years is ever seeded, and only when that
        # restriction reproduces the published binding exactly.
        release_index_ids = _release_index_game_ids(client, release_id)
        if release_index_ids and release_index_ids <= set(game_ids):
            try:
                window_years = _manifest_window_years(manifest)
            except RefreshValidationError as error:
                raise PublicRefreshError(str(error)) from error
            filtered_ids = _year_filtered_ids(config.runtime_root, game_ids, window_years)
            if (
                len(filtered_ids) == expected_count
                and source_identity_sha256(filtered_ids) == expected_digest
            ):
                _atomic_json(config.public_root / "manifest.json", manifest)
                _atomic_json(
                    config.sync.state_path,
                    {
                        **state,
                        "pack_id": release_id,
                        "published_game_ids": filtered_ids,
                        "release_index_game_ids": sorted(release_index_ids),
                        "status": "bootstrapped",
                    },
                )
                return {
                    "status": "seeded",
                    "pack_id": release_id,
                    "game_count": len(filtered_ids),
                    "source": "validated_local_cache_superset",
                    "out_of_window_game_count": len(out_of_window_ids),
                    "out_of_window_identity_sha256": source_identity_sha256(out_of_window_ids),
                }
        raise PublicRefreshError("Supabase continuity bootstrap does not match the active release")

    _atomic_json(config.public_root / "manifest.json", manifest)
    _atomic_json(
        config.sync.state_path,
        {
            **state,
            "pack_id": release_id,
            "published_game_ids": game_ids,
            "status": "bootstrapped",
        },
    )
    return {
        "status": "seeded",
        "pack_id": release_id,
        "game_count": expected_count,
        "source": source,
        "out_of_window_game_count": len(out_of_window_ids),
        "out_of_window_identity_sha256": source_identity_sha256(out_of_window_ids),
    }


def _release_index_game_ids(
    client: supabase_publication.SupabasePublicData,
    release_id: str,
) -> set[str]:
    """The active release's own published game index, when available."""

    try:
        payload = _supabase_asset_json(client, release_id, "features/match_index.json")
    except PublicRefreshError:
        return set()
    games = payload.get("games")
    if not isinstance(games, list):
        return set()
    index_ids = {
        canonical_source_game_key(str(game.get("game_id")))
        for game in games
        if isinstance(game, dict)
    }
    index_ids.discard("")
    return index_ids


def _sleep_before_retry(attempt: int) -> None:
    time.sleep(min(60.0, 2.0**attempt))


def _run_with_source_retries(
    config: RefreshConfig,
    now: datetime,
    *,
    force: bool,
    prepare_tier_fn: Any | None = None,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(config.attempts):
        try:
            return sync_once(
                config.sync,
                now=now,
                force=force,
                prepare_tier_fn=prepare_tier_fn,
            )
        except RETRYABLE_ERRORS as error:
            last_error = error
            if attempt + 1 >= config.attempts:
                raise
            _sleep_before_retry(attempt)
    raise PublicRefreshError("ratings refresh stopped without a result") from last_error


def _run_tier_refresh(config: RefreshConfig, expected_live_as_of: str) -> dict[str, Any]:
    previous = live_refresh.DEFAULT_OUTPUT
    previous_path = (
        config.runtime_root / previous
        if (config.runtime_root / previous).is_file()
        else config.root / previous
        if (config.root / previous).is_file()
        else None
    )
    receipt = live_refresh.refresh_candidate(
        config.runtime_root,
        expected_live_as_of=expected_live_as_of,
        previous_path=previous_path,
        source_mode="oe_only",
        promote=True,
        skip_annual_oe=True,
        skip_live_source=True,
        skip_atom_bridge=True,
        step_timeout_seconds=config.step_timeout_seconds,
        publish=config.publication_backend == "blob",
    )
    expected_status = (
        "production_built"
        if config.publication_backend == "supabase"
        else "production_promoted"
    )
    if receipt.get("status") != expected_status:
        raise PublicRefreshError(
            "tier-list refresh did not promote: " + str(receipt.get("status") or "unknown")
        )
    return receipt


def _tier_publication_for_source(
    config: RefreshConfig,
    receipt: Mapping[str, Any],
    *,
    source_observed_through: str | None,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the generated tier board to the exact source accepted by the pack."""

    expected_status = (
        "production_built"
        if config.publication_backend == "supabase"
        else "production_promoted"
    )
    if receipt.get("status") != expected_status:
        raise PublicRefreshError(
            "tier-list refresh did not produce a publishable artifact: "
            + str(receipt.get("status") or "unknown")
        )
    if not isinstance(source_observed_through, str) or not source_observed_through.strip():
        raise PublicRefreshError("tier-list binding has no accepted source watermark")
    source_identity = str(source.get("identity_sha256") or "")
    if not supabase_publication.SHA256_RE.fullmatch(source_identity):
        raise PublicRefreshError("tier-list binding has no accepted source identity digest")

    payload_path = (
        config.runtime_root
        / "apps"
        / "scryglass"
        / "public"
        / "rankings"
        / "tierlists.json"
    ).resolve()
    runtime_root = config.runtime_root.resolve()
    try:
        payload_path.relative_to(runtime_root)
    except ValueError as error:
        raise PublicRefreshError("tier-list payload escaped the runtime root") from error
    try:
        raw = payload_path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicRefreshError("tier-list production payload is unavailable") from error
    if not isinstance(payload, Mapping):
        raise PublicRefreshError("tier-list production payload is malformed")
    if payload.get("schema_version") != "rankings-tierlists-v2" or payload.get("status") != "available":
        raise PublicRefreshError("tier-list production payload is not available")
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PublicRefreshError("tier-list production payload has no rows")
    roles_by_patch: dict[str, set[str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise PublicRefreshError("tier-list production row is malformed")
        patch = str(row.get("patch") or "").strip()
        role = str(row.get("role") or "").strip().casefold()
        if not patch or role not in {"top", "jungle", "mid", "bot", "support"}:
            raise PublicRefreshError("tier-list production row has an invalid patch or role")
        roles_by_patch.setdefault(patch, set()).add(role)
    if not roles_by_patch or any(len(roles) != 5 for roles in roles_by_patch.values()):
        raise PublicRefreshError("tier-list production payload lacks a complete five-role scope")

    receipt_sha256 = str(receipt.get("receipt_canonical_sha256") or "")
    if not supabase_publication.SHA256_RE.fullmatch(receipt_sha256):
        raise PublicRefreshError("tier-list refresh receipt has no canonical digest")
    payload_sha256 = hashlib.sha256(raw).hexdigest()
    candidate = receipt.get("candidate")
    if not isinstance(candidate, Mapping):
        raise PublicRefreshError("tier-list refresh receipt has no candidate binding")
    candidate_artifact = str(candidate.get("artifact_sha256") or "")
    if not supabase_publication.SHA256_RE.fullmatch(candidate_artifact):
        raise PublicRefreshError("tier-list candidate has no artifact digest")
    live_census = validate_live_source(config.runtime_root, [], years=config.sync.years)
    expected_count = int(source.get("game_count", -1))
    candidate_count = int(candidate.get("maps_replayed", -1))
    if (
        live_census.get("game_count") != expected_count
        or live_census.get("identity_sha256") != source_identity
        or candidate_count != expected_count
    ):
        raise PublicRefreshError("tier-list game census does not match the accepted release")

    def timestamp(value: object) -> datetime | None:
        if not isinstance(value, str) or not value.strip():
            return None
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed.astimezone(timezone.utc)

    expected_time = timestamp(source_observed_through)
    receipt_time = timestamp(receipt.get("source_observed_through"))
    candidate_time = timestamp(receipt.get("candidate_expected_live_as_of"))
    if expected_time is None or receipt_time != expected_time or candidate_time != expected_time:
        raise PublicRefreshError("tier-list source watermark does not match the accepted source")
    as_of = str(payload.get("as_of") or "")
    if timestamp(as_of) is None:
        raise PublicRefreshError("tier-list payload has no valid as-of timestamp")

    return {
        "schema_version": "scryglass:tier-publication:v1",
        "status": "available",
        "production_status": expected_status,
        "tier_schema_version": payload["schema_version"],
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha256,
        "receipt_sha256": receipt_sha256,
        "candidate_artifact_sha256": candidate_artifact,
        "source_observed_through": source_observed_through,
        "as_of": as_of,
        "source_identity_sha256": source_identity,
        "patches": sorted(roles_by_patch),
        "scope_count": len(roles_by_patch),
    }


def _http_bytes(
    url: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    headers: dict[str, str] | None = None,
    attempts: int = 1,
    expected_release_id: str | None = None,
) -> bytes:
    if attempts < 1 or attempts > 5:
        raise PublicRefreshError("HTTP attempts must be between one and five")
    request = urllib.request.Request(url, method=method, data=body, headers=headers or {})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=45) as response:
                if expected_release_id is not None:
                    served_release_id = response.headers.get("X-Scryglass-Release", "")
                    if served_release_id != expected_release_id:
                        raise PublicRefreshError(
                            f"HTTP response has release {served_release_id or 'missing'}, "
                            f"expected {expected_release_id}: {url}"
                        )
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


def _assert_html_release_marker(raw: bytes, release_id: str, path: str) -> None:
    marker = f'data-scryglass-release="{release_id}"'.encode("ascii")
    if marker not in raw:
        raise PublicRefreshError(
            f"public page probe has no marker for release {release_id}: {path}"
        )


def _probe_deployed_manifest_asset(
    config: RefreshConfig,
    manifest: dict[str, Any],
    release_id: str,
) -> dict[str, Any]:
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PublicRefreshError("public manifest probe has no asset inventory")
    candidates = [
        item
        for item in files
        if isinstance(item, dict)
        and isinstance(item.get("path"), str)
        and isinstance(item.get("bytes"), int)
        and not isinstance(item.get("bytes"), bool)
        and int(item["bytes"]) >= 0
        and isinstance(item.get("sha256"), str)
    ]
    if not candidates:
        raise PublicRefreshError("public manifest probe has no verifiable asset")
    asset = min(candidates, key=lambda item: int(item["bytes"]))
    path = str(asset["path"])
    expected_url = (
        f"/api/assets/{urllib.parse.quote(release_id, safe='')}/"
        f"{urllib.parse.quote(path, safe='')}"
    )
    if asset.get("url") != expected_url:
        raise PublicRefreshError("public manifest probe has an invalid asset URL")
    raw = _http_bytes(
        f"{config.site}{expected_url}",
        headers={"Cache-Control": "no-cache"},
        attempts=config.attempts,
        expected_release_id=release_id,
    )
    if len(raw) != int(asset["bytes"]) or _sha256(raw) != asset["sha256"]:
        raise PublicRefreshError(f"deployed public asset failed verification: {path}")
    return {
        "path": path,
        "url": expected_url,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _load_remote_manifest(config: RefreshConfig) -> tuple[dict[str, Any], str]:
    if config.publication_backend == "supabase":
        client = supabase_publication.SupabasePublicData(
            config.supabase_url or "",
            config.supabase_secret_key or "",
        )
        release = client.active_release()
        manifest = release.get("manifest") if isinstance(release, dict) else None
        if not isinstance(manifest, dict):
            raise PublicRefreshError("Supabase has no active public release")
        return manifest, str(manifest.get("base_url") or "").rstrip("/")
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


def _diagnostic_headers() -> dict[str, str]:
    token = _read_env("SCRYGLASS_DIAGNOSTIC_TOKEN")
    return {"authorization": f"Bearer {token}"} if token else {}


def _health_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics")
    return diagnostics if isinstance(diagnostics, dict) else payload


def verify_public_release(config: RefreshConfig, *, expected_pack_id: str | None = None, tier_expected: bool = False) -> dict[str, Any]:
    manifest, base_url = _load_remote_manifest(config)
    pack_id = str(manifest.get("pack_id") or "")
    if not pack_id or not base_url:
        raise PublicRefreshError("public pack manifest has no active pack and base URL")
    if config.production:
        if config.publication_backend == "supabase":
            if manifest.get("data_backend") != "supabase" or base_url != config.supabase_url:
                raise PublicRefreshError("public pack is not bound to the configured Supabase project")
        elif not base_url.startswith("https://") or ".public.blob.vercel-storage.com" not in base_url:
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
    local = not config.manifest_url and config.publication_backend != "supabase"
    supabase_client = (
        supabase_publication.SupabasePublicData(
            config.supabase_url or "",
            config.supabase_secret_key or "",
        )
        if config.publication_backend == "supabase"
        else None
    )
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
        if supabase_client is not None:
            asset = supabase_client.asset(pack_id, relative)
            if (
                not asset
                or asset.get("bytes") != item.get("bytes")
                or asset.get("sha256") != item.get("sha256")
            ):
                raise PublicRefreshError(f"Supabase public asset failed verification: {relative}")
            checked_files += 1
            total_bytes += int(item.get("bytes", 0))
            continue
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
        health_raw = _http_bytes(
            f"{config.site}/api/health",
            headers=_diagnostic_headers(),
            attempts=config.attempts,
        )
        try:
            health_payload = json.loads(health_raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicRefreshError("public health response is invalid JSON") from error
        if not isinstance(health_payload, dict) or health_payload.get("status") not in {"ok", "partial"}:
            raise PublicRefreshError("public health response is not ready")
        health_detail = _health_diagnostics(health_payload)
        served_pack_id = str(
            health_detail.get("release_id") or health_payload.get("pack_id") or ""
        )
        if not served_pack_id:
            raise PublicRefreshError("public health response has no pack ID")
        if expected_pack_id and served_pack_id != expected_pack_id:
            raise PublicRefreshError(
                f"public app serves {served_pack_id}, expected {expected_pack_id}"
            )
        if tier_expected:
            served_tier = health_detail.get("tier") or health_payload.get("tier")
            if not isinstance(served_tier, dict) or served_tier.get("status") != "available":
                raise PublicRefreshError("public health response has no available tier list")

    tier_status = None
    if config.production and config.publication_backend == "supabase":
        if supabase_client is None:
            raise PublicRefreshError("Supabase verification client is unavailable")
        tier_asset = supabase_client.asset(pack_id, supabase_publication.TIER_ASSET_PATH)
        if not tier_asset:
            raise PublicRefreshError("Supabase tier-list asset is missing")
        tier_body = tier_asset.get("body") if isinstance(tier_asset, dict) else None
        tier_manifest = manifest.get("tier")
        tier_status = (
            tier_body.get("status")
            if isinstance(tier_body, dict)
            else tier_manifest.get("status")
            if isinstance(tier_manifest, dict)
            else None
        )
        if tier_expected and tier_status != "available":
            raise PublicRefreshError("Supabase tier-list display is not available")
    return {"pack_id": pack_id, "files": checked_files, "tier_status": tier_status}


def verify_public_health_alignment(
    config: RefreshConfig,
    *,
    release_id: str,
    source_as_of: str | None,
    expected_worker_commit: str,
) -> dict[str, Any] | None:
    if not config.production or config.publication_backend != "supabase":
        return None
    raw = _http_bytes(
        f"{config.site}/api/health",
        headers=_diagnostic_headers(),
        attempts=config.attempts,
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicRefreshError("public health alignment response is invalid JSON") from error
    if not isinstance(payload, dict):
        raise PublicRefreshError("public health alignment response is not an object")
    health_detail = _health_diagnostics(payload)
    expected = {
        "release_id": release_id,
        "refresh_status": "idle",
        "worker_commit": expected_worker_commit,
    }
    mismatched = [
        key
        for key, value in expected.items()
        if (
            health_detail.get(key)
            if key != "release_id"
            else health_detail.get("release_id") or payload.get("pack_id")
        )
        != value
    ]
    if payload.get("status") != "ok":
        mismatched.append("status")
    if payload.get("stale") is not False:
        mismatched.append("stale")
    try:
        expected_source = datetime.fromisoformat(str(source_as_of).replace("Z", "+00:00"))
        served_source = datetime.fromisoformat(
            str(health_detail.get("source_as_of")).replace("Z", "+00:00")
        )
    except ValueError:
        expected_source = source_as_of
        served_source = health_detail.get("source_as_of")
    if served_source != expected_source:
        mismatched.append("source_as_of")
    if mismatched or not (
        payload.get("last_success_at") or payload.get("last_refresh_success_at")
    ):
        detail = ", ".join(mismatched or ["last_success_at"])
        raise PublicRefreshError(f"public health does not match the active release: {detail}")
    return payload


def invalidate_public_cache(config: RefreshConfig, release_id: str) -> dict[str, Any]:
    secret = _read_env("SCRYGLASS_DATA_PUBLISH_TOKEN")
    if not secret:
        raise PublicRefreshError("SCRYGLASS_DATA_PUBLISH_TOKEN is required for cache invalidation")
    try:
        release_id = supabase_publication._rollback_release_id(release_id)
    except supabase_publication.SupabasePublicationError as error:
        raise PublicRefreshError(
            "cache invalidation requires a valid activated release ID"
        ) from error
    body = _http_bytes(
        f"{config.site}/api/data-published",
        method="POST",
        body=(json.dumps({"release_id": release_id}) + "\n").encode("utf-8"),
        headers={
            "authorization": f"Bearer {secret}",
            "content-type": "application/json",
        },
        attempts=config.attempts,
    )
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicRefreshError("cache invalidation response is invalid JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("revalidated") is not True
        or payload.get("requested_release_id") != release_id
        or payload.get("served_release_id") != release_id
        or payload.get("matches") is not True
    ):
        raise PublicRefreshError("cache invalidation was not confirmed")
    return payload


def probe_public_release_families(
    config: RefreshConfig,
    release_id: str,
) -> dict[str, Any] | None:
    """Warm each public data family and keep the release watermark stable."""

    if not config.production:
        return None
    if not supabase_publication.PACK_ID_RE.fullmatch(release_id):
        raise PublicRefreshError("public probes require a valid activated release ID")

    def health_watermark() -> str:
        raw = _http_bytes(
            f"{config.site}/api/health",
            headers={**_diagnostic_headers(), "Cache-Control": "no-cache"},
            attempts=config.attempts,
        )
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PublicRefreshError("public probe health response is invalid JSON") from error
        if not isinstance(payload, dict):
            raise PublicRefreshError("public probe health response is not an object")
        detail = _health_diagnostics(payload)
        return str(detail.get("release_id") or payload.get("pack_id") or "")

    if health_watermark() != release_id:
        raise PublicRefreshError("public probes started on a different release")
    checked: list[str] = []
    payloads: dict[str, Any] = {}
    for path in PUBLIC_RELEASE_PROBES:
        raw = _http_bytes(
            f"{config.site}{path}",
            headers={"Cache-Control": "no-cache"},
            attempts=config.attempts,
            expected_release_id=(
                release_id
                if path.startswith("/api/chat/")
                or path.startswith("/api/public-data/")
                or path == "/packs/manifest.json"
                else None
            ),
        )
        if path.startswith("/api/") or path == "/packs/manifest.json":
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise PublicRefreshError(f"public probe returned invalid JSON: {path}") from error
            if not isinstance(payload, (dict, list)):
                raise PublicRefreshError(f"public probe returned an invalid payload: {path}")
            payloads[path] = payload
        if path in RELEASE_BOUND_PAGE_PROBES:
            _assert_html_release_marker(raw, release_id, path)
        checked.append(path)

    manifest_payload = payloads.get("/packs/manifest.json")
    if (
        not isinstance(manifest_payload, dict)
        or manifest_payload.get("release_id") != release_id
    ):
        raise PublicRefreshError("public manifest probe has a different release")
    asset_probe = _probe_deployed_manifest_asset(config, manifest_payload, release_id)
    checked.append(str(asset_probe["url"]))

    match_payload = payloads.get("/api/chat/matches?team=T1&limit=1")
    match_data = match_payload.get("data") if isinstance(match_payload, dict) else None
    matches = match_data.get("matches") if isinstance(match_data, dict) else None
    if not isinstance(matches, list) or not matches:
        raise PublicRefreshError("public match probe could not resolve a current match")
    match_id = matches[0].get("game_id") if isinstance(matches[0], dict) else None
    if not isinstance(match_id, str) or not match_id.strip():
        raise PublicRefreshError("public match probe returned an invalid game ID")
    dynamic_paths = (
        "/elo/player/Faker",
        "/elo/team/T1",
        f"/matches/{urllib.parse.quote(match_id, safe='')}",
    )
    for path in dynamic_paths:
        raw = _http_bytes(
            f"{config.site}{path}",
            headers={"Cache-Control": "no-cache"},
            attempts=config.attempts,
        )
        _assert_html_release_marker(raw, release_id, path)
        checked.append(path)
    if health_watermark() != release_id:
        raise PublicRefreshError("public probes ended on a different release")
    return {"release_id": release_id, "routes": checked, "asset": asset_probe}


def prune_local_releases(public_root: Path, active_release_id: str, keep: int = 3) -> list[str]:
    if keep < 3:
        raise PublicRefreshError("local release retention must keep at least three releases")
    candidates = sorted(
        (
            path
            for path in public_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and supabase_publication.PACK_ID_RE.fullmatch(path.name)
        ),
        key=lambda path: path.name,
        reverse=True,
    ) if public_root.is_dir() else []
    previous = [path.name for path in candidates if path.name != active_release_id]
    protected = {active_release_id, *previous[: keep - 1]}
    removed: list[str] = []
    for path in candidates:
        if path.name in protected:
            continue
        shutil.rmtree(path)
        removed.append(path.name)
    return sorted(removed)


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


def _tier_publication(tier: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(tier, dict):
        return None
    steps = tier.get("promotion_steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict) or step.get("source") not in {
            "publication_coordinator",
            "supabase_publication",
        }:
            continue
        publication = step.get("publication")
        if isinstance(publication, dict):
            return publication
    return None


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
    tier: dict[str, Any] | None = None
    tier_publication: dict[str, Any] | None = None
    database_publication: dict[str, Any] | None = None
    post_publication_verified = False
    failure_stage = "preflight"
    ledger: RefreshRunLedger | None = None
    accepted_inputs: dict[str, Any] | None = None
    _write_health(config, "checking", checked_at)
    try:
        _preflight(config)
        accepted_inputs = _accepted_run_inputs(config)
        ledger = _start_run_ledger(config, checked_at, accepted_inputs)
        _write_remote_health(
            config,
            checked_at=checked_at,
            ledger=ledger,
            status="ok",
            refresh_status="running",
            release_id=None,
            source_as_of=(
                str(accepted_inputs.get("source_observed_through"))
                if accepted_inputs and accepted_inputs.get("source_observed_through")
                else None
            ),
        )
        if ledger is not None:
            ledger.advance(
                "ingest",
                metrics={
                    "bytes": int(accepted_inputs.get("source_bytes", 0)),
                    "rows": int(accepted_inputs.get("source_rows", 0)),
                    "games": int(accepted_inputs.get("source_games", 0)),
                },
            )
        failure_stage = "continuity_bootstrap"
        continuity_bootstrap = seed_supabase_continuity(config)
        if ledger is not None:
            ledger.advance("reconcile")
        failure_stage = "ratings"
        if ledger is not None:
            ledger.advance("derive")
        def prepare_tier(
            source_observed_through: str | None,
            source: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            nonlocal failure_stage, tier, tier_publication
            failure_stage = "tier"
            expected = str(source_observed_through or _iso(checked_at))
            tier = _run_tier_refresh(config, expected)
            tier_publication = _tier_publication_for_source(
                config,
                tier,
                source_observed_through=source_observed_through,
                source=source,
            )
            return tier_publication

        ratings = _run_with_source_retries(
            config,
            checked_at,
            force=force,
            prepare_tier_fn=prepare_tier,
        )
        publication = ratings.get("publication") if isinstance(ratings.get("publication"), dict) else None
        reported_tier_publication = ratings.get("tier_publication")
        if isinstance(reported_tier_publication, Mapping):
            tier_publication = dict(reported_tier_publication)
        tier_error: str | None = None
        active_manifest = _load_json(config.public_root / "manifest.json")
        active_tier = active_manifest.get("tier")
        supabase_tier_available = bool(
            config.publication_backend == "supabase"
            and isinstance(active_tier, dict)
            and active_tier.get("status") == "available"
        )
        should_run_tier = bool(
            tier_publication is None
            and (
                ratings.get("status") == "published"
            or force
            or (
                config.publication_backend == "supabase"
                and not supabase_tier_available
            )
            or (
                config.publication_backend != "supabase"
                and (
                    not previous_public_state.get("tier")
                    or previous_public_state.get("tier_error")
                )
                )
            )
        )
        if should_run_tier:
            failure_stage = "tier"
            expected = str(ratings.get("source_observed_through") or _iso(checked_at))
            try:
                tier = _run_tier_refresh(config, expected)
                source_identity = str(ratings.get("source_identity_sha256") or "")
                if source_identity:
                    tier_publication = _tier_publication_for_source(
                        config,
                        tier,
                        source_observed_through=ratings.get("source_observed_through"),
                        source={"identity_sha256": source_identity},
                    )
                else:
                    tier_publication = _tier_publication(tier)
            except Exception as error:  # noqa: BLE001
                tier_error = f"{type(error).__name__}: {str(error)[:500]}"
        if tier_error and config.publication_backend == "supabase":
            raise PublicRefreshError(tier_error)
        if ledger is not None:
            local_manifest = _load_json(
                config.sync.output_root / str(ratings.get("pack_id") or "") / "manifest.json"
            )
            ledger.advance(
                "validate_artifacts",
                metrics={
                    "bytes": int(local_manifest.get("total_bytes", 0)),
                    "files": int(local_manifest.get("total_files", 0)),
                    "changed_games": int(accepted_inputs.get("new_games", 0))
                    + int(accepted_inputs.get("corrected_games", 0))
                    if accepted_inputs
                    else 0,
                },
            )
        if tier is not None and config.publication_backend == "supabase":
            failure_stage = "publication"
            pack_id = str(ratings.get("pack_id") or "")
            pack_dir = config.public_root / pack_id
            manifest = _load_json(pack_dir / "manifest.json")
            if not manifest:
                raise PublicRefreshError("local release manifest is missing before Supabase publication")
            if ledger is not None:
                ledger.advance("stage_release")
            database_publication = supabase_publication.publish_release(
                pack_dir,
                manifest,
                config.runtime_root / "apps/scryglass/public/rankings/tierlists.json",
                project_url=config.supabase_url or "",
                secret_key=config.supabase_secret_key or "",
            )
            if ledger is not None:
                ledger.advance(
                    "activate_release",
                    release_id=pack_id,
                    metrics={
                        "bytes": int(database_publication.get("bytes", 0)),
                        "assets": int(database_publication.get("assets", 0)),
                    },
                )
        changed = (
            ratings.get("status") == "published"
            or tier is not None
            or database_publication is not None
        )
        ledger_release_id = (
            str(ratings.get("pack_id") or "") or None
            if database_publication is not None or ratings.get("status") != "published"
            else None
        )
        failure_stage = "cache"
        if ledger is not None:
            ledger.advance("invalidate_cache", release_id=ledger_release_id)
        active_release_id = str(ratings.get("pack_id") or "")
        cache = invalidate_public_cache(config, active_release_id) if changed else None
        cache_probes = probe_public_release_families(config, active_release_id)
        failure_stage = "smoke"
        if ledger is not None:
            ledger.advance("smoke", release_id=ledger_release_id)
        smoke = verify_public_release(
            config,
            expected_pack_id=str(ratings.get("pack_id")) if ratings.get("status") == "published" else None,
            tier_expected=tier is not None,
        )
        if config.publication_backend != "supabase":
            post_publication_verified = True
        pruned_local_releases = prune_local_releases(
            config.public_root,
            str(smoke["pack_id"]),
        )
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "partial" if tier_error else str(ratings.get("status") or "ok"),
            "checked_at": _iso(checked_at),
            "continuity_bootstrap": continuity_bootstrap,
            "ratings": ratings,
            "tier": tier,
            "tier_publication": tier_publication,
            "tier_error": tier_error,
            "database_publication": database_publication,
            "cache_invalidation": cache,
            "cache_probes": cache_probes,
            "smoke": smoke,
            "pruned_local_releases": pruned_local_releases,
            "refresh_run_id": ledger.run_id if ledger is not None else None,
            "accepted_source": accepted_inputs,
        }
        if tier_error:
            _atomic_json(config.state_path, result)
            raise PublicRefreshError(tier_error)
        _write_remote_health(
            config,
            checked_at=checked_at,
            ledger=ledger,
            status="ok",
            refresh_status="idle",
            release_id=smoke["pack_id"],
            source_as_of=str(ratings.get("source_observed_through") or "") or None,
        )
        health_alignment = verify_public_health_alignment(
            config,
            release_id=str(smoke["pack_id"]),
            source_as_of=str(ratings.get("source_observed_through") or "") or None,
            expected_worker_commit=(
                ledger.worker_git_commit
                if ledger is not None
                else worker_commit(config.root)
                if config.production
                else ""
            ),
        )
        if (
            isinstance(database_publication, dict)
            and database_publication.get("status") == "activated_pending_health"
        ):
            database_publication["status"] = "published"
        result["public_health"] = health_alignment
        if ledger is not None:
            final_status = "no_change" if result["status"] == "no_change" else "success"
            ledger.finish(final_status, release_id=smoke["pack_id"])
        post_publication_verified = True
        _atomic_json(config.state_path, result)
        _write_health(
            config,
            result["status"],
            checked_at,
            pack_id=smoke["pack_id"],
            tier_error=tier_error,
        )
        return result
    except Exception as error:
        if ledger is not None:
            try:
                ledger.fail(error)
            except Exception:
                pass
        should_rollback = (
            not post_publication_verified
            and failure_stage in {"tier", "publication", "cache", "smoke"}
            and (
                publication is not None
                or tier_publication is not None
                or database_publication is not None
            )
        )
        if should_rollback:
            rollback = {"status": "restoring"}
            rollback_errors: list[str] = []
            restored_release_id: str | None = None
            if publication is not None:
                try:
                    rollback["ratings"] = rollback_public_pack(publication, config.public_root)
                    _atomic_json(config.sync.state_path, previous_state)
                except Exception as rollback_error:  # noqa: BLE001
                    rollback_errors.append(
                        f"ratings: {type(rollback_error).__name__}: {rollback_error}"
                    )
            if tier_publication is not None:
                try:
                    rollback["tier"] = live_refresh.restore_production_bundle(tier_publication)
                except Exception as rollback_error:  # noqa: BLE001
                    rollback_errors.append(
                        f"tier: {type(rollback_error).__name__}: {rollback_error}"
                    )
            if database_publication is not None:
                previous_release_id = database_publication.get("previous_release_id")
                if isinstance(previous_release_id, str) and previous_release_id:
                    try:
                        rollback["supabase"] = supabase_publication.restore_release(
                            previous_release_id,
                            project_url=config.supabase_url or "",
                            secret_key=config.supabase_secret_key or "",
                        )
                        restored_release_id = str(
                            rollback["supabase"].get("release_id") or previous_release_id
                        )
                    except Exception as rollback_error:  # noqa: BLE001
                        rollback_errors.append(
                            f"supabase: {type(rollback_error).__name__}: {rollback_error}"
                        )
            try:
                rollback_release_id = restored_release_id or str(
                    previous_state.get("pack_id") or ""
                )
                rollback["cache_invalidation"] = invalidate_public_cache(
                    config,
                    rollback_release_id,
                )
            except Exception as cache_error:  # noqa: BLE001
                rollback_errors.append(
                    f"cache: {type(cache_error).__name__}: {cache_error}"
                )
                rollback["cache_invalidation"] = {
                    "status": "failed",
                    "reason": f"{type(cache_error).__name__}: {cache_error}",
                }
            else:
                try:
                    rollback["cache_probes"] = probe_public_release_families(
                        config,
                        rollback_release_id,
                    )
                except Exception as probe_error:  # noqa: BLE001
                    rollback_errors.append(
                        f"probe: {type(probe_error).__name__}: {probe_error}"
                    )
                    rollback["cache_probes"] = {
                        "status": "failed",
                        "reason": f"{type(probe_error).__name__}: {probe_error}",
                    }
            if rollback_errors:
                rollback = {
                    **rollback,
                    "status": "failed",
                    "reason": " | ".join(rollback_errors),
                }
            else:
                rollback["status"] = "restored"
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
        try:
            _write_remote_health(
                config,
                checked_at=checked_at,
                ledger=ledger,
                status="error",
                refresh_status="failed",
                release_id=None,
                source_as_of=(
                    str(accepted_inputs.get("source_observed_through"))
                    if accepted_inputs and accepted_inputs.get("source_observed_through")
                    else None
                ),
            )
        except Exception:
            pass
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
