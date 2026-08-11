"""Publish one validated Scryglass release through Supabase Postgres.

The worker uploads page-ready JSON into staging rows. One database function
checks the complete asset set and changes the active release in one transaction.
Readers can therefore see either the previous complete release or the new one.
"""

from __future__ import annotations

import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from lol_kills.export.pack_spec import OPTIONAL_PUBLIC_FILES, PUBLIC_RATING_REQUIRED_FILES


TIER_ASSET_PATH = "rankings/tierlists.json"
TIER_LATEST_ASSET_PATH = "rankings/tierlists-latest.json"
PUBLIC_ASSET_PATHS = (
    *PUBLIC_RATING_REQUIRED_FILES,
    *OPTIONAL_PUBLIC_FILES,
    TIER_ASSET_PATH,
    TIER_LATEST_ASSET_PATH,
)
PACK_ID_RE = re.compile(r"^v\d{4}\.\d{2}\.\d{2}\.\d{6}$")
REQUEST_TIMEOUT_SECONDS = 60.0
INLINE_ASSET_MAX_BYTES = 1_500_000


class SupabasePublicationError(RuntimeError):
    """A Supabase release could not be staged, activated, or verified."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _project_url(value: str) -> str:
    raw = value.strip().rstrip("/")
    parsed = urllib.parse.urlsplit(raw)
    if (
        parsed.scheme != "https"
        or parsed.username
        or parsed.password
        or parsed.port
        or parsed.query
        or parsed.fragment
        or not (parsed.hostname or "").endswith(".supabase.co")
        or parsed.path not in {"", "/"}
    ):
        raise ValueError("Supabase URL must be an HTTPS project URL")
    return raw


def _secret_key(value: str) -> str:
    key = value.strip()
    if not key.startswith("sb_secret_") or len(key) < 24 or any(character.isspace() for character in key):
        raise ValueError("Supabase secret key is malformed")
    return key


class SupabasePublicData:
    """Small PostgREST client for the Scryglass public release tables."""

    def __init__(
        self,
        project_url: str,
        api_key: str,
        *,
        opener: Any | None = None,
    ) -> None:
        self.project_url = _project_url(project_url)
        self._api_key = _secret_key(api_key)
        self._opener = opener or urllib.request.build_opener()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(project_url={self.project_url!r}, api_key=<redacted>)"

    def _request(
        self,
        method: str,
        path: str,
        payload: object | None = None,
        *,
        prefer: str | None = None,
    ) -> Any:
        body = None
        headers = {
            "apikey": self._api_key,
            "Accept": "application/json",
        }
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            headers["Content-Type"] = "application/json"
        if prefer:
            headers["Prefer"] = prefer
        request = urllib.request.Request(
            f"{self.project_url}/rest/v1/{path.lstrip('/')}",
            data=body,
            method=method,
            headers=headers,
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                raw = response.read()
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                parsed = json.loads(error.read().decode("utf-8"))
                detail = str(parsed.get("message") or parsed.get("hint") or "")[:300]
            except Exception:  # noqa: BLE001
                detail = ""
            suffix = f": {detail}" if detail else ""
            raise SupabasePublicationError(
                f"Supabase request failed with HTTP {error.code} for {path.split('?', 1)[0]}{suffix}"
            ) from error
        except (urllib.error.URLError, TimeoutError) as error:
            raise SupabasePublicationError(
                f"Supabase request failed for {path.split('?', 1)[0]}"
            ) from error
        if not raw:
            return None
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SupabasePublicationError("Supabase returned invalid JSON") from error

    def release(self, release_id: str) -> dict[str, Any] | None:
        encoded = urllib.parse.quote(release_id, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_releases"
            f"?release_id=eq.{encoded}&select=release_id,status,manifest,source_as_of,activated_at&limit=1",
        )
        if not isinstance(rows, list):
            raise SupabasePublicationError("Supabase release response is malformed")
        return rows[0] if rows and isinstance(rows[0], dict) else None

    def active_release(self) -> dict[str, Any] | None:
        rows = self._request(
            "GET",
            "scryglass_public_releases"
            "?status=eq.active&select=release_id,status,manifest,source_as_of,activated_at&limit=1",
        )
        if not isinstance(rows, list):
            raise SupabasePublicationError("Supabase active release response is malformed")
        return rows[0] if rows and isinstance(rows[0], dict) else None

    def asset(self, release_id: str, path: str) -> dict[str, Any] | None:
        release = urllib.parse.quote(release_id, safe="")
        asset_path = urllib.parse.quote(path, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_assets"
            f"?release_id=eq.{release}&path=eq.{asset_path}&select=path,body,storage_path,bytes,sha256&limit=1",
        )
        if not isinstance(rows, list):
            raise SupabasePublicationError("Supabase asset response is malformed")
        return rows[0] if rows and isinstance(rows[0], dict) else None

    def asset_metadata(self, release_id: str) -> dict[str, dict[str, Any]]:
        release = urllib.parse.quote(release_id, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_assets"
            f"?release_id=eq.{release}&select=path,bytes,sha256",
        )
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise SupabasePublicationError("Supabase asset inventory is malformed")
        return {
            str(row["path"]): row
            for row in rows
            if isinstance(row.get("path"), str)
        }

    def write_refresh_run(self, payload: dict[str, Any]) -> None:
        fields = {
            key: payload.get(key)
            for key in (
                "run_id",
                "scheduled_for",
                "retry_of",
                "status",
                "stage",
                "input_fingerprint",
                "worker_commit",
                "source_file_sha256",
                "source_observed_through",
                "release_id",
                "accepted_games",
                "new_games",
                "corrected_games",
                "unchanged_games",
                "quarantined_games",
                "stage_durations",
                "failure_code",
                "failure_detail",
                "started_at",
                "completed_at",
            )
        }
        fields["updated_at"] = payload.get("updated_at")
        self._request(
            "POST",
            "scryglass_refresh_runs?on_conflict=run_id",
            [fields],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def write_public_health(self, payload: dict[str, Any]) -> None:
        row = {
            "health_id": "public-refresh",
            "status": payload["status"],
            "refresh_status": payload["refresh_status"],
            "checked_at": payload["checked_at"],
            "last_success_at": payload.get("last_success_at"),
            "source_as_of": payload.get("source_as_of"),
            "active_release_id": payload.get("active_release_id"),
            "last_run_id": payload.get("last_run_id"),
            "worker_commit": payload.get("worker_commit"),
            "stale": bool(payload.get("stale")),
            "updated_at": payload["checked_at"],
        }
        self._request(
            "POST",
            "scryglass_public_health?on_conflict=health_id",
            [row],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def storage_object(self, storage_path: str) -> bytes:
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/public/scryglass-public/{encoded}",
            method="GET",
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                return response.read()
        except Exception as error:  # noqa: BLE001
            raise SupabasePublicationError("Supabase Storage read failed") from error

    def put_storage_object(self, storage_path: str, raw: bytes) -> None:
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/scryglass-public/{encoded}",
            data=raw,
            method="POST",
            headers={
                "apikey": self._api_key,
                "Content-Type": "application/json",
                "Cache-Control": "public, max-age=31536000, immutable",
                "x-upsert": "false",
            },
        )
        try:
            with self._opener.open(request, timeout=180.0) as response:
                response.read()
            return
        except urllib.error.HTTPError as error:
            if error.code not in {400, 409}:
                raise SupabasePublicationError(
                    f"Supabase Storage upload failed with HTTP {error.code}"
                ) from error
        existing = self.storage_object(storage_path)
        if existing != raw:
            raise SupabasePublicationError("existing Supabase Storage object has different content")

    def stage_release(
        self,
        release: dict[str, Any],
        assets: list[dict[str, Any]],
        *,
        storage_objects: dict[str, bytes] | None = None,
    ) -> int:
        self._request(
            "POST",
            "scryglass_public_releases?on_conflict=release_id",
            [release],
            prefer="resolution=merge-duplicates,return=minimal",
        )
        return self.stage_assets(
            str(release["release_id"]),
            assets,
            storage_objects=storage_objects,
        )

    def stage_assets(
        self,
        release_id: str,
        assets: list[dict[str, Any]],
        *,
        storage_objects: dict[str, bytes] | None = None,
    ) -> int:
        """Add immutable assets to an existing release and reuse exact matches."""

        existing = self.asset_metadata(release_id)
        reused = 0
        for asset in assets:
            prior = existing.get(str(asset["path"]))
            if (
                prior
                and prior.get("bytes") == asset["bytes"]
                and prior.get("sha256") == asset["sha256"]
            ):
                reused += 1
                continue
            if prior:
                raise SupabasePublicationError(
                    f"existing public asset has different content: {asset['path']}"
                )
            storage_path = asset.get("storage_path")
            if isinstance(storage_path, str):
                raw = (storage_objects or {}).get(storage_path)
                if raw is None:
                    raise SupabasePublicationError("large public asset has no storage payload")
                self.put_storage_object(storage_path, raw)
            self._request(
                "POST",
                "scryglass_public_assets?on_conflict=release_id,path",
                [asset],
                prefer="resolution=merge-duplicates,return=minimal",
            )
        return reused

    def activate(self, release_id: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            "rpc/activate_scryglass_public_release",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError("Supabase activation response is malformed")
        return result

    def restore(self, release_id: str) -> dict[str, Any]:
        result = self._request(
            "POST",
            "rpc/restore_scryglass_public_release",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError("Supabase rollback response is malformed")
        return result

    def prune(self, keep: int = 3) -> int:
        result = self._request(
            "POST",
            "rpc/prune_scryglass_public_releases",
            {"p_keep": keep},
        )
        if type(result) is not int:
            raise SupabasePublicationError("Supabase retention response is malformed")
        return result


def _asset(path: str, raw: bytes, release_id: str) -> dict[str, Any]:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupabasePublicationError(f"public asset is invalid JSON: {path}") from error
    return {
        "release_id": release_id,
        "path": path,
        "body": body,
        "bytes": len(raw),
        "sha256": _sha256(raw),
    }


def _patch_order(value: object) -> tuple[int, int]:
    parts = str(value or "").split(".", 1)
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def latest_tier_payload(tier_body: dict[str, Any]) -> dict[str, Any]:
    """Keep the newest patch while preserving every view for that patch."""

    options = tier_body.get("options")
    patches = options.get("patches") if isinstance(options, dict) else None
    if not isinstance(patches, list) or not patches:
        raise SupabasePublicationError("tier-list asset has no patch options")
    latest_patch = max((str(value) for value in patches), key=_patch_order)
    rows = tier_body.get("rows")
    scopes = tier_body.get("scopes")
    if not isinstance(rows, list) or not isinstance(scopes, list):
        raise SupabasePublicationError("tier-list asset has invalid rows or scopes")
    latest = dict(tier_body)
    latest["rows"] = [row for row in rows if isinstance(row, dict) and row.get("patch") == latest_patch]
    latest["scopes"] = [scope for scope in scopes if isinstance(scope, dict) and scope.get("patch") == latest_patch]
    latest["latest_patch"] = latest_patch
    if not latest["rows"] or not latest["scopes"]:
        raise SupabasePublicationError("latest tier-list patch has no public data")
    return latest


def prepare_release(
    pack_dir: Path,
    manifest: dict[str, Any],
    tier_path: Path,
    *,
    project_url: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bytes]]:
    release_id = str(manifest.get("pack_id") or "")
    if not PACK_ID_RE.fullmatch(release_id):
        raise SupabasePublicationError("pack ID is invalid")
    if not pack_dir.is_dir():
        raise SupabasePublicationError("pack directory is missing")
    manifest_files = {
        str(item.get("path")): item
        for item in manifest.get("files", [])
        if isinstance(item, dict)
    }
    assets: list[dict[str, Any]] = []
    storage_objects: dict[str, bytes] = {}
    for path in PUBLIC_RATING_REQUIRED_FILES:
        metadata = manifest_files.get(path)
        if not isinstance(metadata, dict):
            raise SupabasePublicationError(f"manifest is missing public asset: {path}")
        raw = (pack_dir / path).read_bytes()
        if len(raw) != metadata.get("bytes") or _sha256(raw) != metadata.get("sha256"):
            raise SupabasePublicationError(f"public asset checksum failed: {path}")
        asset = _asset(path, raw, release_id)
        if len(raw) > INLINE_ASSET_MAX_BYTES:
            storage_path = f"{release_id}/{path}"
            asset["body"] = None
            asset["storage_path"] = storage_path
            storage_objects[storage_path] = raw
        assets.append(asset)

    for path in OPTIONAL_PUBLIC_FILES:
        metadata = manifest_files.get(path)
        if not isinstance(metadata, dict):
            continue
        raw = (pack_dir / path).read_bytes()
        if len(raw) != metadata.get("bytes") or _sha256(raw) != metadata.get("sha256"):
            raise SupabasePublicationError(f"optional public asset checksum failed: {path}")
        asset = _asset(path, raw, release_id)
        if len(raw) > INLINE_ASSET_MAX_BYTES:
            storage_path = f"{release_id}/{path}"
            asset["body"] = None
            asset["storage_path"] = storage_path
            storage_objects[storage_path] = raw
        assets.append(asset)

    tier_raw = tier_path.read_bytes()
    tier_asset = _asset(TIER_ASSET_PATH, tier_raw, release_id)
    tier_body = tier_asset["body"]
    if not isinstance(tier_body, dict) or tier_body.get("status") != "available":
        raise SupabasePublicationError("tier-list asset is unavailable")
    tier_storage_path = f"{release_id}/{TIER_ASSET_PATH}"
    tier_asset["body"] = None
    tier_asset["storage_path"] = tier_storage_path
    assets.append(tier_asset)
    storage_objects[tier_storage_path] = tier_raw

    latest_tier_raw = (
        json.dumps(
            latest_tier_payload(tier_body),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    latest_tier_asset = _asset(TIER_LATEST_ASSET_PATH, latest_tier_raw, release_id)
    latest_tier_storage_path = f"{release_id}/{TIER_LATEST_ASSET_PATH}"
    latest_tier_asset["body"] = None
    latest_tier_asset["storage_path"] = latest_tier_storage_path
    assets.append(latest_tier_asset)
    storage_objects[latest_tier_storage_path] = latest_tier_raw

    published_manifest = dict(manifest)
    published_manifest["base_url"] = _project_url(project_url)
    published_manifest["data_backend"] = "supabase"
    published_manifest["tier"] = {
        "status": "available",
        "as_of": tier_body.get("as_of"),
        "latest_path": TIER_LATEST_ASSET_PATH,
    }
    release_metadata = manifest.get("release")
    if not isinstance(release_metadata, dict):
        release_metadata = {}
    artifact_hashes = {
        str(asset["path"]): str(asset["sha256"])
        for asset in assets
    }
    published_manifest["release"] = {
        **release_metadata,
        "release_id": release_id,
        "tier_list_version": str(tier_body.get("schema_version") or "rankings-tierlists-v2"),
        "artifact_hashes": artifact_hashes,
    }
    release = {
        "release_id": release_id,
        "status": "staging",
        "manifest": published_manifest,
        "source_as_of": (manifest.get("ratings") or {}).get("source_as_of"),
    }
    return release, assets, storage_objects


def _verify_release_assets(
    client: SupabasePublicData,
    release_id: str,
    assets: list[dict[str, Any]],
) -> None:
    for expected in assets:
        actual = client.asset(release_id, str(expected["path"]))
        if (
            not actual
            or actual.get("bytes") != expected["bytes"]
            or actual.get("sha256") != expected["sha256"]
        ):
            raise SupabasePublicationError(
                f"Supabase asset readback failed: {expected['path']}"
            )
        storage_path = expected.get("storage_path")
        if isinstance(storage_path, str):
            raw = client.storage_object(storage_path)
            if len(raw) != expected["bytes"] or _sha256(raw) != expected["sha256"]:
                raise SupabasePublicationError(
                    f"Supabase Storage readback failed: {expected['path']}"
                )
        elif actual.get("body") != expected["body"]:
            raise SupabasePublicationError(
                f"Supabase asset readback failed: {expected['path']}"
            )


def _verify_active_release(
    client: SupabasePublicData,
    release_id: str,
    assets: list[dict[str, Any]],
) -> None:
    active = client.active_release()
    if not active or active.get("release_id") != release_id:
        raise SupabasePublicationError("Supabase active release readback failed")
    _verify_release_assets(client, release_id, assets)


def publish_release(
    pack_dir: Path,
    manifest: dict[str, Any],
    tier_path: Path,
    *,
    project_url: str,
    secret_key: str,
    retention: int = 3,
    client: SupabasePublicData | None = None,
) -> dict[str, Any]:
    database = client or SupabasePublicData(project_url, secret_key)
    release, assets, storage_objects = prepare_release(
        pack_dir,
        manifest,
        tier_path,
        project_url=database.project_url,
    )
    release_id = str(release["release_id"])
    existing = database.release(release_id)
    if existing and existing.get("status") == "superseded":
        raise SupabasePublicationError("a superseded release ID cannot be changed")
    if existing and existing.get("status") == "active":
        reused_assets = database.stage_assets(
            release_id,
            assets,
            storage_objects=storage_objects,
        )
        _verify_active_release(database, release_id, assets)
        return {
            "status": "already_active",
            "release_id": release_id,
            "previous_release_id": None,
            "assets": len(assets),
            "reused_assets": reused_assets,
            "bytes": sum(int(asset["bytes"]) for asset in assets),
            "retained_releases": retention,
        }

    reused_assets = database.stage_release(
        release,
        assets,
        storage_objects=storage_objects,
    )
    staged = database.release(release_id)
    if not staged or staged.get("status") != "staging":
        raise SupabasePublicationError("Supabase staged release readback failed")
    _verify_release_assets(database, release_id, assets)
    activation = database.activate(release_id)
    try:
        _verify_active_release(database, release_id, assets)
    except Exception:
        previous_release_id = activation.get("previous_release_id")
        if isinstance(previous_release_id, str) and previous_release_id:
            try:
                database.restore(previous_release_id)
            except Exception as rollback_error:
                raise SupabasePublicationError(
                    "active release verification and rollback failed"
                ) from rollback_error
        raise
    pruned = database.prune(retention)
    return {
        "status": "published",
        "release_id": release_id,
        "previous_release_id": activation.get("previous_release_id"),
        "assets": len(assets),
        "reused_assets": reused_assets,
        "bytes": sum(int(asset["bytes"]) for asset in assets),
        "pruned_releases": pruned,
        "retained_releases": retention,
    }


def restore_release(
    release_id: str,
    *,
    project_url: str,
    secret_key: str,
) -> dict[str, Any]:
    return SupabasePublicData(project_url, secret_key).restore(release_id)
