"""Publish one validated Scryglass release through Supabase Postgres.

The worker uploads page-ready JSON into staging rows. One database function
checks the complete asset set and changes the active release in one transaction.
Readers can therefore see either the previous complete release or the new one.
"""

from __future__ import annotations

import base64
import hashlib
import json
import time
import re
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from lol_kills.export.pack_spec import (
    OPTIONAL_PUBLIC_FILES,
    PUBLIC_ASSET_PATHS,
    PUBLIC_RATING_REQUIRED_FILES,
)
from lol_kills.export.public_query_projection import (
    QUERY_API_SCHEMA,
    QUERY_DATASETS,
    QUERY_PROJECTION_PATH,
    TIER_QUERY_DATASETS,
    PublicQueryProjectionError,
    build_tier_query_datasets,
    canonical_query_bytes,
    query_dataset_receipt,
    validate_public_query_projection,
)
from lol_kills.v2.patch_identity import PatchIdentityError, public_patch


TIER_ASSET_PATH = "rankings/tierlists.json"
TIER_LATEST_ASSET_PATH = "rankings/tierlists-latest.json"
DRAFT_ASSET_PATH = "features/draft_records.json"
DRAFT_AUTHORITY_SCHEMA = "scryglass:draft-authority:v1"
DRAFT_RECORDS_SCHEMA = "scryglass:draft-records:v1"
PUBLIC_ASSET_PATH_SET = frozenset(PUBLIC_ASSET_PATHS)
PACK_ID_RE = re.compile(r"^v\d{4}\.\d{2}\.\d{2}\.\d{6}$")
REQUEST_TIMEOUT_SECONDS = 60.0
PUBLIC_ASSET_CONTENT_TYPE = "application/json"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
QUERY_TABLE = "scryglass_public_query_rows"
QUERY_RECEIPT_TABLE = "scryglass_public_query_receipts"
QUERY_ROW_FIELDS = frozenset(
    {
        "row_key",
        "player_id",
        "team_id",
        "champion_id",
        "identity_id",
        "name",
        "search_key",
        "kind",
        "alias_key",
        "role",
        "team",
        "league",
        "tier",
        "active",
        "rating",
        "adjusted_rating",
        "movement",
        "games",
        "wins",
        "win_rate",
        "grade_a_games",
        "grade_games",
        "champion",
        "champion_key",
        "score",
        "game_id",
        "played_at",
        "year",
        "blue_team",
        "red_team",
        "blue_team_id",
        "red_team_id",
        "blue_win",
        "champions",
        "ordinal",
        "image_url",
        "patch",
        "region",
        "rank",
        "played_maps",
        "scope_id",
        "reference_id",
        "payload",
        "source_bytes",
        "source_sha256",
        "row_sha256",
    }
)


class SupabasePublicationError(RuntimeError):
    """A Supabase release could not be staged, activated, or verified."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _release_id(value: object) -> str:
    release_id = str(value or "")
    if not PACK_ID_RE.fullmatch(release_id):
        raise SupabasePublicationError("release ID is invalid")
    return release_id


def _rollback_release_id(value: object) -> str:
    try:
        return _release_id(value)
    except SupabasePublicationError as error:
        raise SupabasePublicationError("rollback release ID is invalid") from error


def _public_asset_path(value: object) -> str:
    path = str(value or "")
    if path not in PUBLIC_ASSET_PATH_SET:
        raise SupabasePublicationError("public asset path is not allowed")
    return path


def _storage_path(value: object) -> str:
    storage_path = str(value or "")
    release_id, separator, path = storage_path.partition("/")
    if not separator or _release_id(release_id) + "/" + _public_asset_path(path) != storage_path:
        raise SupabasePublicationError("Supabase Storage path is invalid")
    return storage_path


def _retention_storage_path(value: object) -> str:
    storage_path = str(value or "")
    release_id, separator, path = storage_path.partition("/")
    if (
        not separator
        or not (
            PACK_ID_RE.fullmatch(release_id)
        )
        or f"{release_id}/{_public_asset_path(path)}" != storage_path
    ):
        raise SupabasePublicationError("retained Supabase Storage path is invalid")
    return storage_path


def _draft_authority(manifest: dict[str, Any], release_id: str) -> dict[str, Any]:
    """Keep Draft Score unavailable until an independent verifier exists."""

    candidate = manifest.get("draft_authority")
    if isinstance(candidate, dict) and candidate.get("status") == "promoted":
        raise SupabasePublicationError(
            "Draft Score promotion requires an independent receipt verifier"
        )
    reason = candidate.get("reason") if isinstance(candidate, dict) else None
    return {
        "schema_version": DRAFT_AUTHORITY_SCHEMA,
        "status": "unavailable",
        "release_id": release_id,
        "model_version": None,
        "receipt_sha256": None,
        "issued_utc": None,
        "reason": str(reason or "model_not_promoted"),
    }


def _pack_asset_path(pack_dir: Path, path: str) -> Path:
    root = pack_dir.resolve(strict=True)
    candidate = (root / _public_asset_path(path)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SupabasePublicationError("public asset path leaves the pack") from error
    if not candidate.is_file():
        raise SupabasePublicationError(f"public asset is not a file: {path}")
    return candidate


def _query_projection_path(pack_dir: Path) -> Path:
    root = pack_dir.resolve(strict=True)
    candidate = (root / QUERY_PROJECTION_PATH).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise SupabasePublicationError("query projection path leaves the pack") from error
    if not candidate.is_file():
        raise SupabasePublicationError("query projection is missing")
    return candidate


def _prepare_query_datasets(
    pack_dir: Path,
    manifest: dict[str, Any],
    tier_body: dict[str, Any],
    release_id: str,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any] | None]:
    """Read the internal projection and bind its receipts to publication."""

    query_api = manifest.get("query_api")
    if query_api is None:
        return {}, None
    projection_metadata = query_api.get("projection") if isinstance(query_api, dict) else None
    if (
        not isinstance(query_api, dict)
        or query_api.get("schema_version") != QUERY_API_SCHEMA
        or query_api.get("status") != "available"
        or not isinstance(projection_metadata, dict)
        or projection_metadata.get("path") != QUERY_PROJECTION_PATH
    ):
        raise SupabasePublicationError("query API manifest is invalid")
    path = _query_projection_path(pack_dir)
    raw = path.read_bytes()
    if (
        projection_metadata.get("bytes") != len(raw)
        or projection_metadata.get("sha256") != _sha256(raw)
    ):
        raise SupabasePublicationError("query projection checksum failed")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupabasePublicationError("query projection is invalid JSON") from error
    if not isinstance(parsed, dict):
        raise SupabasePublicationError("query projection is malformed")
    try:
        datasets = validate_public_query_projection(parsed, release_id=release_id)
        tier_datasets = build_tier_query_datasets(tier_body)
    except PublicQueryProjectionError as error:
        raise SupabasePublicationError(str(error)) from error
    datasets.update(tier_datasets)
    receipts = {
        dataset: query_dataset_receipt(dataset, rows)
        for dataset, rows in datasets.items()
    }
    if set(receipts) != {*QUERY_DATASETS, *TIER_QUERY_DATASETS}:
        raise SupabasePublicationError("query dataset inventory is not exact")
    return datasets, {
        "schema_version": QUERY_API_SCHEMA,
        "status": "available",
        "projection_sha256": _sha256(raw),
        "datasets": receipts,
    }


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
        release_id = _release_id(release_id)
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
        release_id = _release_id(release_id)
        path = _public_asset_path(path)
        release = urllib.parse.quote(release_id, safe="")
        asset_path = urllib.parse.quote(path, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_assets"
            f"?release_id=eq.{release}&path=eq.{asset_path}"
            "&select=path,body,storage_path,bytes,sha256,content_type&limit=1",
        )
        if not isinstance(rows, list):
            raise SupabasePublicationError("Supabase asset response is malformed")
        return rows[0] if rows and isinstance(rows[0], dict) else None

    def asset_metadata(self, release_id: str) -> dict[str, dict[str, Any]]:
        release_id = _release_id(release_id)
        release = urllib.parse.quote(release_id, safe="")
        rows = self._request(
            "GET",
            "scryglass_public_assets"
            f"?release_id=eq.{release}&select=path,bytes,sha256,content_type,storage_path",
        )
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise SupabasePublicationError("Supabase asset inventory is malformed")
        return {
            str(row["path"]): row
            for row in rows
            if isinstance(row.get("path"), str)
        }

    def query_receipts(self, release_id: str) -> dict[str, dict[str, Any]]:
        release_id = _release_id(release_id)
        encoded = urllib.parse.quote(release_id, safe="")
        rows = self._request(
            "GET",
            f"{QUERY_RECEIPT_TABLE}?release_id=eq.{encoded}"
            "&select=dataset,row_count,source_bytes,source_sha256,row_digest_sha256,storage_bytes,storage_sha256",
        )
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise SupabasePublicationError("Supabase query receipt response is malformed")
        return {
            str(row["dataset"]): row
            for row in rows
            if isinstance(row.get("dataset"), str)
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
                "requirements_lock_sha256",
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

    def write_diagnostic_credential(self, token: str) -> None:
        if not 32 <= len(token) <= 512 or any(character.isspace() for character in token):
            raise SupabasePublicationError("diagnostic token is malformed")
        self._request(
            "POST",
            "scryglass_diagnostic_credentials?on_conflict=credential_id",
            [
                {
                    "credential_id": "public-health",
                    "token_sha256": _sha256(token.encode("utf-8")),
                    "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
            ],
            prefer="resolution=merge-duplicates,return=minimal",
        )

    def storage_object(self, storage_path: str) -> bytes:
        storage_path = _storage_path(storage_path)
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/authenticated/scryglass-public/{encoded}",
            method="GET",
            headers={
                "apikey": self._api_key,
            },
        )
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    return response.read()
            except Exception as error:  # noqa: BLE001
                last_error = error
                time.sleep(0.5 * (attempt + 1))
        raise SupabasePublicationError("Supabase Storage read failed") from last_error

    def storage_object_metadata(self, storage_path: str) -> dict[str, Any]:
        """Object metadata (size, etag) without downloading the body.

        Used for post-publish verification so a 140MB release is verified by
        metadata instead of being downloaded back (cached-egress cost).
        """
        storage_path = _storage_path(storage_path)
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/info/authenticated/scryglass-public/{encoded}",
            method="GET",
            headers={
                "apikey": self._api_key,
            },
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as error:  # noqa: BLE001
            raise SupabasePublicationError("Supabase Storage metadata read failed") from error
        if not isinstance(payload, dict):
            raise SupabasePublicationError("Supabase Storage metadata is malformed")
        return payload

    def put_storage_object(
        self,
        storage_path: str,
        raw: bytes,
        *,
        sha256: str,
        content_type: str,
    ) -> None:
        storage_path = _storage_path(storage_path)
        if sha256 != _sha256(raw):
            raise SupabasePublicationError("Supabase Storage upload digest is invalid")
        if content_type != PUBLIC_ASSET_CONTENT_TYPE:
            raise SupabasePublicationError("Supabase Storage content type is invalid")
        encoded = "/".join(urllib.parse.quote(part, safe="") for part in storage_path.split("/"))
        custom_metadata = {
            "sha256": sha256,
            "bytes": len(raw),
            "content_type": content_type,
        }
        metadata = base64.b64encode(
            json.dumps(custom_metadata, separators=(",", ":")).encode("utf-8")
        ).decode("ascii")
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/scryglass-public/{encoded}",
            data=raw,
            method="POST",
            headers={
                "apikey": self._api_key,
                "Content-Type": "application/json",
                "Cache-Control": "public, max-age=31536000, immutable",
                "x-upsert": "false",
                "x-metadata": metadata,
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
        existing_metadata = self.storage_object_metadata(storage_path)
        existing_custom = existing_metadata.get("metadata")
        if isinstance(existing_custom, str):
            try:
                existing_custom = json.loads(existing_custom)
            except json.JSONDecodeError:
                existing_custom = None
        if (
            int(existing_metadata.get("size") or -1) != len(raw)
            or not isinstance(existing_custom, dict)
            or existing_custom.get("sha256") != sha256
            or existing_custom.get("bytes") != len(raw)
            or existing_custom.get("content_type") != content_type
        ):
            raise SupabasePublicationError("existing Supabase Storage metadata is different")
        existing = self.storage_object(storage_path)
        if len(existing) != len(raw) or _sha256(existing) != _sha256(raw):
            raise SupabasePublicationError("existing Supabase Storage object has different content")

    def delete_storage_objects(self, storage_paths: list[str]) -> None:
        prefixes = [_retention_storage_path(path) for path in storage_paths]
        if not prefixes:
            return
        body = json.dumps({"prefixes": prefixes}, separators=(",", ":")).encode("utf-8")
        request = urllib.request.Request(
            f"{self.project_url}/storage/v1/object/scryglass-public",
            data=body,
            method="DELETE",
            headers={
                "apikey": self._api_key,
                "Content-Type": "application/json",
            },
        )
        try:
            with self._opener.open(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                response.read()
        except Exception as error:  # noqa: BLE001
            raise SupabasePublicationError("Supabase Storage retention delete failed") from error

    def ack_storage_cleanup(self, storage_paths: list[str]) -> int:
        paths = [_retention_storage_path(path) for path in storage_paths]
        result = self._request(
            "POST",
            "rpc/ack_scryglass_storage_cleanup",
            {"p_storage_paths": paths},
        )
        if type(result) is not int:
            raise SupabasePublicationError(
                "Supabase Storage cleanup acknowledgement is malformed"
            )
        return result

    def stage_release(
        self,
        release: dict[str, Any],
        assets: list[dict[str, Any]],
        *,
        storage_objects: dict[str, bytes] | None = None,
    ) -> int:
        release_id = _release_id(release.get("release_id"))
        self._request(
            "POST",
            "scryglass_public_releases",
            [release],
            prefer="return=minimal",
        )
        return self.stage_assets(
            release_id,
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

        release_id = _release_id(release_id)
        existing = self.asset_metadata(release_id)
        reused = 0
        pending: list[dict[str, Any]] = []
        for asset in assets:
            if _release_id(asset.get("release_id")) != release_id:
                raise SupabasePublicationError("public asset release ID does not match")
            _public_asset_path(asset.get("path"))
            storage_path = asset.get("storage_path")
            if storage_path is not None:
                expected_storage_path = f"{release_id}/{asset['path']}"
                if _storage_path(storage_path) != expected_storage_path:
                    raise SupabasePublicationError("public asset Storage path does not match")
            prior = existing.get(str(asset["path"]))
            if (
                prior
                and prior.get("bytes") == asset["bytes"]
                and prior.get("sha256") == asset["sha256"]
                and prior.get("content_type") == asset["content_type"]
                and prior.get("storage_path") == asset.get("storage_path")
            ):
                reused += 1
                continue
            if prior:
                raise SupabasePublicationError(
                    f"existing public asset has different content: {asset['path']}"
                )
            pending.append(asset)
        # Phase 1: upload all large storage objects concurrently (bounded workers).
        uploads = [
            (asset, (storage_objects or {}).get(str(asset.get("storage_path"))))
            for asset in pending
            if isinstance(asset.get("storage_path"), str)
        ]
        for asset, raw in uploads:
            if raw is None:
                raise SupabasePublicationError("large public asset has no storage payload")
        if uploads:
            with ThreadPoolExecutor(max_workers=min(6, len(uploads))) as executor:
                list(executor.map(
                    lambda item: self.put_storage_object(
                        str(item[0]["storage_path"]),
                        item[1],
                        sha256=str(item[0]["sha256"]),
                        content_type=str(item[0]["content_type"]),
                    ),
                    uploads,
                ))
        # Phase 2: insert the asset rows (inline small assets and storage-backed).
        for asset in pending:
            self._request(
                "POST",
                "scryglass_public_assets",
                [asset],
                prefer="return=minimal",
            )
        return reused

    def stage_query_datasets(
        self,
        release_id: str,
        datasets: dict[str, list[dict[str, Any]]],
        receipts: dict[str, Any],
    ) -> int:
        """Insert immutable bounded rows, then seal each dataset receipt."""

        release_id = _release_id(release_id)
        reused = 0
        existing = self.query_receipts(release_id)
        for dataset in (*QUERY_DATASETS, *TIER_QUERY_DATASETS):
            rows = datasets.get(dataset)
            receipt = receipts.get(dataset)
            if not isinstance(rows, list) or not isinstance(receipt, dict):
                raise SupabasePublicationError(f"query dataset is missing: {dataset}")
            prior = existing.get(dataset)
            if prior:
                if (
                    prior.get("row_count") != receipt.get("rows")
                    or prior.get("source_bytes") != receipt.get("bytes")
                    or prior.get("source_sha256") != receipt.get("sha256")
                ):
                    raise SupabasePublicationError(
                        f"existing query receipt has different content: {dataset}"
                    )
                reused += len(rows)
                continue
            pending: list[dict[str, Any]] = []
            pending_bytes = 0
            for source in rows:
                unknown = set(source) - QUERY_ROW_FIELDS
                if unknown:
                    raise SupabasePublicationError(
                        f"query row has unsupported fields: {dataset}"
                    )
                source_without_digest = {
                    key: value for key, value in source.items() if key != "row_sha256"
                }
                row = {
                    "release_id": release_id,
                    "dataset": dataset,
                    "row_sha256": source.get("row_sha256"),
                    "source_json": canonical_query_bytes(source_without_digest).decode("utf-8"),
                    "payload_json": canonical_query_bytes(source.get("payload")).decode("utf-8"),
                }
                row_bytes = len(canonical_query_bytes(row))
                if pending and (len(pending) >= 100 or pending_bytes + row_bytes > 400_000):
                    staged = self._request(
                        "POST",
                        "rpc/stage_scryglass_query_rows",
                        {"p_release_id": release_id, "p_dataset": dataset, "p_rows": pending},
                    )
                    if type(staged) is not int:
                        raise SupabasePublicationError("Supabase query row staging is malformed")
                    pending = []
                    pending_bytes = 0
                pending.append(row)
                pending_bytes += row_bytes
            if pending:
                staged = self._request(
                    "POST",
                    "rpc/stage_scryglass_query_rows",
                    {"p_release_id": release_id, "p_dataset": dataset, "p_rows": pending},
                )
                if type(staged) is not int:
                    raise SupabasePublicationError("Supabase query row staging is malformed")
            sealed = self._request(
                "POST",
                "rpc/seal_scryglass_query_dataset",
                {
                    "p_release_id": release_id,
                    "p_dataset": dataset,
                    "p_source_rows": receipt.get("rows"),
                    "p_source_bytes": receipt.get("bytes"),
                    "p_source_sha256": receipt.get("sha256"),
                    "p_row_digest_sha256": receipt.get("row_digest_sha256"),
                },
            )
            if not isinstance(sealed, dict) or sealed.get("dataset") != dataset:
                raise SupabasePublicationError(
                    f"Supabase query receipt seal failed: {dataset}"
                )
        return reused

    def activate(self, release_id: str) -> dict[str, Any]:
        release_id = _release_id(release_id)
        result = self._request(
            "POST",
            "rpc/activate_scryglass_public_release",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError("Supabase activation response is malformed")
        return result

    def restore(self, release_id: str, *, _recover: bool = True) -> dict[str, Any]:
        release_id = _rollback_release_id(release_id)
        candidate = self.release(release_id)
        manifest = candidate.get("manifest") if isinstance(candidate, dict) else None
        release_metadata = manifest.get("release") if isinstance(manifest, dict) else None
        draft_authority = manifest.get("draft_authority") if isinstance(manifest, dict) else None
        query_api = manifest.get("query_api") if isinstance(manifest, dict) else None
        raw_files = manifest.get("files") if isinstance(manifest, dict) else None
        artifact_hashes = (
            release_metadata.get("artifact_hashes")
            if isinstance(release_metadata, dict)
            else None
        )
        if (
            not isinstance(candidate, dict)
            or candidate.get("status") not in {"active", "superseded"}
            or not isinstance(manifest, dict)
            or manifest.get("pack_id") != release_id
            or not isinstance(release_metadata, dict)
            or release_metadata.get("release_id") != release_id
            or not isinstance(draft_authority, dict)
            or draft_authority.get("schema_version") != DRAFT_AUTHORITY_SCHEMA
            or draft_authority.get("status") != "unavailable"
            or draft_authority.get("release_id") != release_id
            or not isinstance(raw_files, list)
            or not isinstance(artifact_hashes, dict)
        ):
            raise SupabasePublicationError("Supabase rollback release binding is invalid")
        inventory = self.asset_metadata(release_id)
        expected_assets: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in raw_files:
            if not isinstance(item, dict):
                raise SupabasePublicationError("Supabase rollback manifest inventory is invalid")
            path = _public_asset_path(item.get("path"))
            if path in seen or path == DRAFT_ASSET_PATH:
                raise SupabasePublicationError("Supabase rollback manifest inventory is invalid")
            seen.add(path)
            raw_bytes = item.get("bytes")
            sha256 = item.get("sha256")
            if (
                type(raw_bytes) is not int
                or raw_bytes < 0
                or not isinstance(sha256, str)
                or not SHA256_RE.fullmatch(sha256)
                or artifact_hashes.get(path) != sha256
            ):
                raise SupabasePublicationError("Supabase rollback manifest inventory is invalid")
            expected_assets.append(
                {
                    "release_id": release_id,
                    "path": path,
                    "body": None,
                    "storage_path": f"{release_id}/{path}",
                    "bytes": raw_bytes,
                    "sha256": sha256,
                    "content_type": PUBLIC_ASSET_CONTENT_TYPE,
                }
            )
        if set(inventory) != seen or set(artifact_hashes) != seen:
            raise SupabasePublicationError("Supabase rollback manifest inventory is not exact")
        _verify_release_assets(
            self,
            release_id,
            expected_assets,
            verify_storage_content=True,
        )
        _verify_query_receipts(
            self,
            release_id,
            query_api if isinstance(query_api, dict) else None,
        )
        result = self._request(
            "POST",
            "rpc/restore_scryglass_public_release",
            {"p_release_id": release_id},
        )
        if not isinstance(result, dict) or result.get("release_id") != release_id:
            raise SupabasePublicationError("Supabase rollback response is malformed")
        try:
            # The target was checked before the transition. Check the bytes
            # again after it becomes active. A superseded Storage object may
            # have changed while the transition waited for its database lock.
            _verify_active_release(
                self,
                release_id,
                expected_assets,
                verify_storage_content=True,
            )
            _verify_query_receipts(
                self,
                release_id,
                query_api if isinstance(query_api, dict) else None,
            )
        except Exception:
            replaced_release_id = result.get("replaced_release_id") or result.get(
                "previous_release_id"
            )
            if _recover and isinstance(replaced_release_id, str) and replaced_release_id:
                try:
                    self.restore(replaced_release_id, _recover=False)
                except Exception as recovery_error:
                    raise SupabasePublicationError(
                        "Supabase rollback readback failed and recovery failed"
                    ) from recovery_error
            raise
        return result

    def prune(self, keep: int = 3) -> int:
        if keep < 1 or keep > 10:
            raise SupabasePublicationError("Supabase retention must be between one and ten")
        result = self._request(
            "POST",
            "rpc/prune_scryglass_public_releases_v2",
            {"p_keep": keep},
        )
        if not isinstance(result, dict) or type(result.get("deleted_count")) is not int:
            raise SupabasePublicationError("Supabase retention response is malformed")
        storage_paths = result.get("storage_paths")
        if not isinstance(storage_paths, list) or any(
            not isinstance(path, str) for path in storage_paths
        ):
            raise SupabasePublicationError("Supabase retention Storage inventory is malformed")
        self.delete_storage_objects(storage_paths)
        self.ack_storage_cleanup(storage_paths)
        return int(result["deleted_count"])


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
        "content_type": PUBLIC_ASSET_CONTENT_TYPE,
    }


def _patch_order(value: object) -> tuple[int, int]:
    parts = str(value or "").split(".", 1)
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError):
        return 0, 0


def _public_patch_label(value: object) -> str:
    """Return the public Riot label for a source or client patch token."""

    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return public_patch(text)
    except PatchIdentityError:
        return text


def _public_scope_id(scope_id: object, source_patch: object, patch: str) -> str:
    value = str(scope_id or "").strip()
    source = str(source_patch or "").strip()
    prefix = f"patch:{source}" if source else ""
    if prefix and value.startswith(prefix):
        return f"patch:{patch}{value[len(prefix):]}"
    return value


def latest_tier_payload(tier_body: dict[str, Any]) -> dict[str, Any]:
    """Keep the newest patch while preserving every view for that patch."""

    options = tier_body.get("options")
    patches = options.get("patches") if isinstance(options, dict) else None
    if not isinstance(patches, list) or not patches:
        raise SupabasePublicationError("tier-list asset has no patch options")
    normalized_patches = [_public_patch_label(value) for value in patches]
    latest_patch = max(normalized_patches, key=_patch_order)
    rows = tier_body.get("rows")
    scopes = tier_body.get("scopes")
    if not isinstance(rows, list) or not isinstance(scopes, list):
        raise SupabasePublicationError("tier-list asset has invalid rows or scopes")
    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        item = dict(row)
        source_patch = item.get("patch")
        item["patch"] = _public_patch_label(source_patch)
        if item.get("scope_id") is not None:
            item["scope_id"] = _public_scope_id(item.get("scope_id"), source_patch, item["patch"])
        normalized_rows.append(item)
    normalized_scopes: list[dict[str, Any]] = []
    for scope in scopes:
        if not isinstance(scope, dict):
            continue
        item = dict(scope)
        source_patch = item.get("patch")
        item["patch"] = _public_patch_label(source_patch)
        if item.get("scope_id") is not None:
            item["scope_id"] = _public_scope_id(item.get("scope_id"), source_patch, item["patch"])
        normalized_scopes.append(item)
    latest = dict(tier_body)
    latest_options = dict(options) if isinstance(options, dict) else {}
    latest_options["patches"] = sorted(set(normalized_patches), key=_patch_order)
    latest["options"] = latest_options
    latest["rows"] = [row for row in normalized_rows if row.get("patch") == latest_patch]
    latest["scopes"] = [scope for scope in normalized_scopes if scope.get("patch") == latest_patch]
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
    release_id = _release_id(manifest.get("pack_id"))
    draft_authority = _draft_authority(manifest, release_id)
    if not pack_dir.is_dir():
        raise SupabasePublicationError("pack directory is missing")
    manifest_entries = manifest.get("files")
    if not isinstance(manifest_entries, list) or any(
        not isinstance(item, dict) for item in manifest_entries
    ):
        raise SupabasePublicationError("manifest file inventory is malformed")
    manifest_files: dict[str, dict[str, Any]] = {}
    for item in manifest_entries:
        path = _public_asset_path(item.get("path"))
        if path in manifest_files:
            raise SupabasePublicationError(f"manifest repeats public asset: {path}")
        manifest_files[path] = item
    if manifest.get("total_files") != len(manifest_entries):
        raise SupabasePublicationError("manifest file total does not match its inventory")
    try:
        manifest_total_bytes = sum(int(item["bytes"]) for item in manifest_entries)
    except (KeyError, TypeError, ValueError) as error:
        raise SupabasePublicationError("manifest byte inventory is malformed") from error
    if manifest.get("total_bytes") != manifest_total_bytes:
        raise SupabasePublicationError("manifest byte total does not match its inventory")

    release_metadata = manifest.get("release")
    if not isinstance(release_metadata, dict):
        release_metadata = {}
    bound_release_id = release_metadata.get("release_id")
    if bound_release_id is not None and bound_release_id != release_id:
        raise SupabasePublicationError("manifest release ID does not match its pack ID")
    bound_hashes = release_metadata.get("artifact_hashes")
    if bound_hashes is not None and not isinstance(bound_hashes, dict):
        raise SupabasePublicationError("manifest artifact hash inventory is malformed")
    if isinstance(bound_hashes, dict) and not set(manifest_files).issubset(
        set(map(str, bound_hashes))
    ):
        raise SupabasePublicationError("manifest artifact hashes do not match its file inventory")

    assets: list[dict[str, Any]] = []
    storage_objects: dict[str, bytes] = {}
    for path in PUBLIC_RATING_REQUIRED_FILES:
        metadata = manifest_files.get(path)
        if not isinstance(metadata, dict):
            raise SupabasePublicationError(f"manifest is missing public asset: {path}")
        raw = _pack_asset_path(pack_dir, path).read_bytes()
        if len(raw) != metadata.get("bytes") or _sha256(raw) != metadata.get("sha256"):
            raise SupabasePublicationError(f"public asset checksum failed: {path}")
        if isinstance(bound_hashes, dict) and bound_hashes.get(path) != _sha256(raw):
            raise SupabasePublicationError(f"manifest release hash conflicts with file: {path}")
        asset = _asset(path, raw, release_id)
        storage_path = f"{release_id}/{path}"
        asset["body"] = None
        asset["storage_path"] = storage_path
        storage_objects[storage_path] = raw
        assets.append(asset)

    for path in OPTIONAL_PUBLIC_FILES:
        metadata = manifest_files.get(path)
        if not isinstance(metadata, dict):
            continue
        if path == DRAFT_ASSET_PATH:
            continue
        raw = _pack_asset_path(pack_dir, path).read_bytes()
        if len(raw) != metadata.get("bytes") or _sha256(raw) != metadata.get("sha256"):
            raise SupabasePublicationError(f"optional public asset checksum failed: {path}")
        if isinstance(bound_hashes, dict) and bound_hashes.get(path) != _sha256(raw):
            raise SupabasePublicationError(f"manifest release hash conflicts with file: {path}")
        asset = _asset(path, raw, release_id)
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
    _query_datasets, published_query_api = _prepare_query_datasets(
        pack_dir,
        manifest,
        tier_body,
        release_id,
    )
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
    published_manifest["draft_authority"] = draft_authority
    published_manifest["tier"] = {
        "status": "available",
        "as_of": tier_body.get("as_of"),
        "latest_path": TIER_LATEST_ASSET_PATH,
    }
    if published_query_api is not None:
        published_manifest["query_api"] = published_query_api
    else:
        published_manifest.pop("query_api", None)
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
    public_files = [
        {
            "path": str(asset["path"]),
            "bytes": int(asset["bytes"]),
            "sha256": str(asset["sha256"]),
        }
        for asset in assets
    ]
    published_manifest["files"] = public_files
    published_manifest["total_files"] = len(public_files)
    published_manifest["total_bytes"] = sum(int(asset["bytes"]) for asset in assets)
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
    *,
    verify_storage_content: bool,
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
            metadata = client.storage_object_metadata(storage_path)
            custom = metadata.get("metadata")
            if isinstance(custom, str):
                try:
                    custom = json.loads(custom)
                except json.JSONDecodeError:
                    custom = None
            if (
                int(metadata.get("size") or -1) != expected["bytes"]
                or not isinstance(custom, dict)
                or custom.get("sha256") != expected["sha256"]
                or custom.get("bytes") != expected["bytes"]
                or custom.get("content_type") != expected["content_type"]
            ):
                raise SupabasePublicationError(
                    f"Supabase Storage readback failed: {expected['path']}"
                )
            if verify_storage_content:
                raw = client.storage_object(storage_path)
                if len(raw) != expected["bytes"] or _sha256(raw) != expected["sha256"]:
                    raise SupabasePublicationError(
                        f"Supabase Storage checksum failed: {expected['path']}"
                    )
        elif actual.get("body") != expected["body"]:
            raise SupabasePublicationError(
                f"Supabase asset readback failed: {expected['path']}"
            )


def _verify_query_receipts(
    client: SupabasePublicData,
    release_id: str,
    query_api: dict[str, Any] | None,
) -> None:
    if query_api is None:
        return
    expected = query_api.get("datasets")
    if not isinstance(expected, dict):
        raise SupabasePublicationError("query API receipt inventory is missing")
    actual = client.query_receipts(release_id)
    if set(actual) != set(expected):
        raise SupabasePublicationError("Supabase query receipt inventory is not exact")
    for dataset, receipt in expected.items():
        row = actual.get(dataset)
        if (
            not isinstance(receipt, dict)
            or not isinstance(row, dict)
            or row.get("row_count") != receipt.get("rows")
            or row.get("source_bytes") != receipt.get("bytes")
            or row.get("source_sha256") != receipt.get("sha256")
            or row.get("row_digest_sha256") != receipt.get("row_digest_sha256")
            or type(row.get("storage_bytes")) is not int
            or not isinstance(row.get("storage_sha256"), str)
            or not SHA256_RE.fullmatch(str(row.get("storage_sha256")))
        ):
            raise SupabasePublicationError(
                f"Supabase query receipt readback failed: {dataset}"
            )


def _verify_active_release(
    client: SupabasePublicData,
    release_id: str,
    assets: list[dict[str, Any]],
    *,
    verify_storage_content: bool,
) -> None:
    active = client.active_release()
    if not active or active.get("release_id") != release_id:
        raise SupabasePublicationError("Supabase active release readback failed")
    manifest = active.get("manifest")
    release_metadata = manifest.get("release") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("pack_id") != release_id
        or not isinstance(release_metadata, dict)
        or release_metadata.get("release_id") != release_id
        or release_metadata.get("artifact_hashes")
        != {str(asset["path"]): str(asset["sha256"]) for asset in assets}
    ):
        raise SupabasePublicationError("Supabase active manifest readback failed")
    _verify_release_assets(
        client,
        release_id,
        assets,
        verify_storage_content=verify_storage_content,
    )


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
    try:
        raw_tier = json.loads(tier_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SupabasePublicationError("tier-list asset is invalid JSON") from error
    if not isinstance(raw_tier, dict):
        raise SupabasePublicationError("tier-list asset is malformed")
    query_datasets, query_api = _prepare_query_datasets(
        pack_dir,
        manifest,
        raw_tier,
        release_id,
    )
    existing = database.release(release_id)
    if existing and existing.get("manifest") != release["manifest"]:
        raise SupabasePublicationError("existing release has a different manifest")
    if existing and existing.get("status") == "superseded":
        raise SupabasePublicationError("a superseded release ID cannot be changed")
    if existing and existing.get("status") == "active":
        _verify_active_release(
            database,
            release_id,
            assets,
            verify_storage_content=True,
        )
        _verify_query_receipts(database, release_id, query_api)
        return {
            "status": "already_active",
            "release_id": release_id,
            "previous_release_id": None,
            "assets": len(assets),
            "reused_assets": len(assets),
            "query_rows": sum(len(rows) for rows in query_datasets.values()),
            "reused_query_rows": sum(len(rows) for rows in query_datasets.values()),
            "bytes": sum(int(asset["bytes"]) for asset in assets),
            "retained_releases": retention,
        }

    if existing and existing.get("status") == "staging":
        reused_assets = database.stage_assets(
            release_id,
            assets,
            storage_objects=storage_objects,
        )
    else:
        reused_assets = database.stage_release(
            release,
            assets,
            storage_objects=storage_objects,
        )
    staged = database.release(release_id)
    if not staged or staged.get("status") != "staging":
        raise SupabasePublicationError("Supabase staged release readback failed")
    reused_query_rows = 0
    if query_api is not None:
        receipts = query_api.get("datasets")
        if not isinstance(receipts, dict):
            raise SupabasePublicationError("query API receipt inventory is missing")
        reused_query_rows = database.stage_query_datasets(
            release_id,
            query_datasets,
            receipts,
        )
        _verify_query_receipts(database, release_id, query_api)
    _verify_release_assets(
        database,
        release_id,
        assets,
        verify_storage_content=True,
    )
    activation = database.activate(release_id)
    try:
        _verify_active_release(
            database,
            release_id,
            assets,
            verify_storage_content=False,
        )
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
    try:
        pruned = database.prune(retention)
    except Exception:
        previous_release_id = activation.get("previous_release_id")
        if isinstance(previous_release_id, str) and previous_release_id:
            try:
                database.restore(previous_release_id)
            except Exception as rollback_error:
                raise SupabasePublicationError(
                    "Supabase retention and rollback failed"
                ) from rollback_error
        raise
    return {
        "status": "activated_pending_health",
        "release_id": release_id,
        "previous_release_id": activation.get("previous_release_id"),
        "assets": len(assets),
        "reused_assets": reused_assets,
        "query_rows": sum(len(rows) for rows in query_datasets.values()),
        "reused_query_rows": reused_query_rows,
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
