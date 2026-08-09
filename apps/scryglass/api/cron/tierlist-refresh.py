"""Run one audited tier-list refresh inside a Vercel Python Function."""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


_RUNTIME_BUNDLE_ROOT: Path | None = None


def _private_blob_url(bundle_path: str, token: str) -> str:
    if not re.fullmatch(r"[-A-Za-z0-9._/]+", bundle_path) or bundle_path.startswith("/"):
        raise RuntimeError("private Blob pathname is invalid")
    prefix = "vercel_blob_rw_"
    if not token.startswith(prefix):
        raise RuntimeError("TIER_WORKER_READ_WRITE_TOKEN is invalid")
    store_id = token[len(prefix):].split("_", 1)[0].lower()
    if not store_id:
        raise RuntimeError("TIER_WORKER_READ_WRITE_TOKEN has no store id")
    encoded_path = "/".join(
        urllib.parse.quote(part, safe="") for part in bundle_path.split("/")
    )
    return f"https://{store_id}.private.blob.vercel-storage.com/{encoded_path}"


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _private_blob_token() -> str:
    token = (
        os.environ.get("TIER_WORKER_READ_WRITE_TOKEN")
        or os.environ.get("BLOB_READ_WRITE_TOKEN")
        or os.environ.get("VERCEL_BLOB_READ_WRITE_TOKEN")
        or ""
    ).strip()
    if not token:
        raise WorkerConfigurationError("private Blob token is not configured")
    return token


def _private_blob_upload(
    pathname: str,
    content: bytes,
    *,
    allow_overwrite: bool,
) -> dict[str, Any]:
    token = _private_blob_token()
    private_url = _private_blob_url(pathname, token)
    store_id = token[len("vercel_blob_rw_") :].split("_", 1)[0].lower()
    encoded_path = "/".join(
        urllib.parse.quote(part, safe="") for part in pathname.split("/")
    )
    upload_url = f"https://blob.vercel-storage.com/{encoded_path}"
    content_type = "application/gzip" if pathname.endswith(".gz") else "application/json"
    headers = {
        "Authorization": f"Bearer {token}",
        "X-API-Version": "7",
        "x-vercel-blob-store-id": store_id,
        "x-vercel-blob-access": "private",
        "x-add-random-suffix": "0",
        "Content-Type": content_type,
        "x-content-type": content_type,
    }
    if allow_overwrite:
        headers["x-allow-overwrite"] = "1"
    request = urllib.request.Request(
        upload_url,
        data=content,
        headers=headers,
        method="PUT",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            response.read(64 * 1024)
            status = int(getattr(response, "status", 200))
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"private Blob upload failed: HTTP {error.code}") from error
    if status < 200 or status >= 300:
        raise RuntimeError(f"private Blob upload failed: HTTP {status}")
    return {
        "pathname": pathname,
        "url": private_url,
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def _private_blob_read(
    pathname: str,
    *,
    max_bytes: int,
    cache_bust: bool = False,
) -> bytes:
    token = _private_blob_token()
    url = _private_blob_url(pathname, token)
    if cache_bust:
        url = f"{url}?readback={uuid.uuid4().hex}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = response.read(max_bytes + 1)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            raise FileNotFoundError(pathname) from error
        raise RuntimeError(f"private Blob read failed: HTTP {error.code}") from error
    if len(body) > max_bytes:
        raise RuntimeError(f"private Blob object is larger than {max_bytes} bytes")
    return body


def _download_runtime_bundle() -> Path | None:
    """Fetch the current worker source when Vercel omits includeFiles."""

    bundle_path = os.environ.get(
        "SCRYGLASS_TIER_WORKER_RUNTIME_BUNDLE_PATH",
        "tier-worker/tier-worker-runtime-current.tar.gz",
    )
    token = os.environ.get("TIER_WORKER_READ_WRITE_TOKEN", "").strip()
    if not token:
        return None
    url = _private_blob_url(bundle_path, token)
    root = Path(tempfile.mkdtemp(prefix="scryglass-tier-worker-runtime-", dir="/tmp"))
    try:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                base = root.resolve()
                # Stream the archive directly into the extractor. The
                # function filesystem is small, so keeping the compressed
                # archive beside its extracted files can exhaust /tmp.
                with tarfile.open(fileobj=response, mode="r|gz") as bundle:
                    for member in bundle:
                        destination = (root / member.name).resolve()
                        if os.path.commonpath((str(base), str(destination))) != str(base):
                            raise RuntimeError(
                                "tier worker runtime bundle contains an unsafe path"
                            )
                        bundle.extract(member, root)
        except urllib.error.HTTPError as error:
            raise RuntimeError(
                f"tier worker runtime bundle download failed: HTTP {error.code}"
            ) from error
        return root
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _find_project_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [
        Path.cwd() / ".tier-worker",
        Path.cwd() / "apps/scryglass/.tier-worker",
        Path.cwd(),
        here.parent,
        *here.parents,
    ]
    for candidate in candidates:
        if (
            (candidate / "lol_kills").is_dir()
            and (candidate / "apps/scryglass/public/packs/latest.json").is_file()
            and (
                candidate
                / "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v3.json"
            ).is_file()
        ):
            return candidate
    global _RUNTIME_BUNDLE_ROOT
    _RUNTIME_BUNDLE_ROOT = _download_runtime_bundle()
    if _RUNTIME_BUNDLE_ROOT is not None:
        candidate = _RUNTIME_BUNDLE_ROOT
        if (
            (candidate / "lol_kills").is_dir()
            and (candidate / "apps/scryglass/public/packs/latest.json").is_file()
            and (
                candidate
                / "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v3.json"
            ).is_file()
        ):
            return candidate
    raise RuntimeError("tier refresh source tree is not present in the Vercel bundle")


PROJECT_ROOT = _find_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

from lol_kills.export.blob_retention import (  # noqa: E402
    PlannedWrite,
    RetentionExecutor,
    RetentionPlan,
    WriteMode,
)
from lol_kills.export.vercel_blob_transport import VercelBlobTransport  # noqa: E402


LOCK_PATH = "_scryglass_retention/tierlist-refresh-lock.json"
LOCK_TTL_SECONDS = 20 * 60
RECEIPT_PREFIX = "tierlists/refresh-receipts/"
MOVEMENT_PATH = "tierlists/movement-v1.json"
SOURCE_LOCK_PATH = "_scryglass_retention/source-refresh-lock.json"
RATINGS_LOCK_PATH = "_scryglass_retention/ratings-refresh-lock.json"
RAW_SOURCE_POINTER_PATH = "tier-worker/tierlist-source-raw-current.json"
RAW_SOURCE_BUNDLE_PREFIX = "tier-worker/tierlist-source-raw/"
SOURCE_POINTER_PATH = "tier-worker/tierlist-source-current.json"
SOURCE_BUNDLE_PREFIX = "tier-worker/tierlist-source/"
SOURCE_BUNDLE_MANIFEST = "source_bundle_manifest.json"
SOURCE_BUNDLE_SCHEMA = "scryglass:tierlist-source-bundle:v1"
SOURCE_BUNDLE_MAX_BYTES = 600_000_000
SOURCE_RELATIVE_FILES = (
    "data/lol/warehouse/parquet/oe_player_games.parquet",
    "data/lol/warehouse/parquet/oe_team_games.parquet",
    "data/lol/warehouse/parquet/oe_api_player_games.parquet",
    "data/lol/warehouse/parquet/oe_api_team_games.parquet",
    "data/lol/warehouse/parquet/oe_api_meta.json",
    "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet",
    "data/lol/warehouse/parquet/oe_live/oe_team_games.parquet",
    "data/lol/warehouse/parquet/oe_live/maps.parquet",
    "data/lol/warehouse/parquet/oe_live/meta.json",
    "data/lol/v2/champions/lcc-atom-bridge-v1.json",
    "data/lol/features/ratings_snapshot.parquet",
    "data/lol/features/player_ratings_snapshot.parquet",
    "data/lol/features/team_weekly_ranks.json",
    "data/lol/features/player_weekly_ranks.json",
    "data/lol/v2/tierlists/rating-refresh/rating-refresh-v1.json",
)
RAW_SOURCE_RELATIVE_FILES = SOURCE_RELATIVE_FILES[:10]
PACKS_ROOT = Path("apps/scryglass/public/packs")
PACK_LATEST = PACKS_ROOT / "latest.json"
LIVE_PACK_ID = os.environ.get("SCRYGLASS_LIVE_PACK_ID", "v2026.live")
LIVE_WINDOW_START = "2026-07-18T00:00:00Z"


class WorkerBusy(RuntimeError):
    """A different refresh still holds the distributed Blob lease."""


class WorkerConfigurationError(RuntimeError):
    """The worker lacks a required production configuration value."""


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    raw = (json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n").encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("Content-Length", str(len(raw)))
    handler.end_headers()
    handler.wfile.write(raw)


def _authorized(handler: BaseHTTPRequestHandler) -> bool:
    supplied = handler.headers.get("Authorization", "")
    if not supplied.startswith("Bearer "):
        return False
    presented = supplied[7:].strip()
    expected_values = {
        value.strip()
        for value in (
            os.environ.get("SCRYGLASS_TIERLIST_INGEST_TOKEN", ""),
            os.environ.get("CRON_SECRET", ""),
        )
        if value.strip()
    }
    return any(hmac.compare_digest(presented, expected) for expected in expected_values)


def _safe_pack_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise WorkerConfigurationError("the committed public pack id is malformed")
    return value


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise WorkerConfigurationError(f"required worker input is missing: {source}")
    # The worker bundle and the writable refresh root both live in /tmp on
    # Vercel. Hard links keep the static inputs available without doubling
    # the bundle footprint. Fall back to regular copies for other filesystems.
    try:
        shutil.copytree(source, destination, symlinks=False, copy_function=os.link)
    except OSError as link_error:
        shutil.rmtree(destination, ignore_errors=True)
        try:
            shutil.copytree(source, destination, symlinks=False)
        except OSError:
            raise link_error


def _prepare_runtime_root() -> Path:
    """Copy writable model inputs into Vercel's temporary filesystem."""

    root = Path(tempfile.mkdtemp(prefix="scryglass-tier-refresh-", dir="/tmp"))
    try:
        for relative in (
            Path("data/lol/v2/champions"),
            Path("data/lol/v2/models/draft-terminal"),
        ):
            _copy_tree(PROJECT_ROOT / relative, root / relative)

        # These paths are writable outputs for the refresh. They do not need
        # the historical research files from the main application bundle.
        for relative in (
            Path("data/lol/v2/tierlists"),
            Path("data/lol/features"),
            Path("data/lol/models"),
            Path("output/pdf"),
        ):
            (root / relative).mkdir(parents=True, exist_ok=True)

        latest_source = PROJECT_ROOT / PACK_LATEST
        latest_destination = root / PACK_LATEST
        latest_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest_source, latest_destination)
        latest = json.loads(latest_source.read_text(encoding="utf-8"))
        pack_id = _safe_pack_id(latest.get("pack_id"))
        pack_source = PROJECT_ROOT / PACKS_ROOT / pack_id
        _copy_tree(pack_source, root / PACKS_ROOT / pack_id)

        # The restore step reads the manifest only to bind the pack receipt.
        # Keep the current pointer and its selected immutable pack together.
        manifest_source = PROJECT_ROOT / PACKS_ROOT / "manifest.json"
        if manifest_source.is_file():
            shutil.copy2(manifest_source, root / PACKS_ROOT / "manifest.json")

        (root / "apps/scryglass/public/v2/tierlists/production").mkdir(
            parents=True, exist_ok=True
        )
        (root / "data/lol/warehouse/parquet").mkdir(parents=True, exist_ok=True)
        return root
    except Exception:
        shutil.rmtree(root, ignore_errors=True)
        raise


def _source_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _refresh_source_inputs(
    runtime_root: Path,
    *,
    expected_live_as_of: str,
    include_ratings: bool = True,
) -> dict[str, Any]:
    """Build the private source bundle inputs used by later cron stages."""

    from lol_kills.v2.tierlists.live_refresh import (
        _api_player_detail_complete,
        _api_source_latest,
        _run_step,
        _skipped_step,
        _source_step_failure,
        _verify_prebuilt_atom_bridge,
    )

    oe_step = _skipped_step("oe_annual", "committed_public_pack_baseline")
    oe_api_step = _run_step(
        runtime_root,
        [
            "lol_kills.etl.oe_api_ingest",
            "--root",
            str(runtime_root),
            "--start",
            LIVE_WINDOW_START,
            "--end",
            expected_live_as_of,
            "--lookback-days",
            "120",
        ],
        source="oe_api",
    )
    atom_step = _verify_prebuilt_atom_bridge(runtime_root)
    observed_as_of = _api_source_latest(runtime_root) if oe_api_step["completed"] else None
    live_source_step = (
        _run_step(
            runtime_root,
            ["lol_kills.etl.oe_live_source", "--root", str(runtime_root)],
            source="oe_live_source",
        )
        if oe_api_step["completed"]
        else _skipped_step("oe_live_source", "oe_api_incomplete")
    )
    rating_step = (
        _run_step(
            runtime_root,
            [
                "lol_kills.v2.tierlists.rating_refresh",
                "--root",
                str(runtime_root),
                "--as-of",
                observed_as_of or expected_live_as_of,
            ],
            source="ratings",
        )
        if include_ratings and live_source_step["completed"] and _api_player_detail_complete(runtime_root)
        else _skipped_step(
            "ratings",
            "oe_player_detail_incomplete" if include_ratings else "deferred_to_ratings_refresh",
        )
    )
    grid_step = _skipped_step("grid", "source_mode_oe_only")
    source_steps = [oe_step, oe_api_step, atom_step, live_source_step, rating_step, grid_step]
    required = [oe_api_step, atom_step, live_source_step]
    if include_ratings:
        required.append(rating_step)
    if not all(step["completed"] for step in required):
        raise RuntimeError(
            "tier refresh source preparation failed: "
            + _source_step_failure(source_steps)
        )
    if not isinstance(observed_as_of, str) or not observed_as_of:
        raise RuntimeError("OE API source receipt has no source_latest value")
    return {
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "artifact_kind": "tier_list_source_bundle",
        "bundle_stage": "complete" if include_ratings else "raw_source",
        "source_mode": "oe_only",
        "expected_live_as_of": expected_live_as_of,
        "source_observed_through": observed_as_of,
        "source_steps": source_steps,
    }


def _refresh_rating_inputs(
    runtime_root: Path,
    *,
    source_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Fit live team and player ratings from a completed raw source bundle."""

    from lol_kills.v2.tierlists.live_refresh import (
        _api_player_detail_complete,
        _run_step,
        _source_step_failure,
    )

    observed_as_of = source_manifest.get("source_observed_through")
    expected_as_of = source_manifest.get("expected_live_as_of")
    if not isinstance(observed_as_of, str) or not observed_as_of:
        raise RuntimeError("raw source bundle has no source_latest value")
    if not isinstance(expected_as_of, str) or not expected_as_of:
        expected_as_of = observed_as_of
    if not _api_player_detail_complete(runtime_root):
        raise RuntimeError("OE player detail is incomplete for ratings refresh")
    rating_step = _run_step(
        runtime_root,
        [
            "lol_kills.v2.tierlists.rating_refresh",
            "--root",
            str(runtime_root),
            "--as-of",
            observed_as_of,
        ],
        source="ratings",
    )
    if not rating_step["completed"]:
        raise RuntimeError("ratings refresh failed: " + _source_step_failure([rating_step]))
    source_steps = [
        dict(step)
        for step in source_manifest.get("source_steps", [])
        if isinstance(step, dict) and step.get("source") != "ratings"
    ]
    source_steps.append(rating_step)
    return {
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "artifact_kind": "tier_list_source_bundle",
        "bundle_stage": "complete",
        "source_mode": "oe_only",
        "expected_live_as_of": expected_as_of,
        "source_observed_through": observed_as_of,
        "source_steps": source_steps,
    }


def _source_bundle_files(
    runtime_root: Path,
    *,
    include_ratings: bool,
) -> list[tuple[str, Path]]:
    relatives = SOURCE_RELATIVE_FILES if include_ratings else RAW_SOURCE_RELATIVE_FILES
    files: list[tuple[str, Path]] = []
    for relative in relatives:
        path = runtime_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"source bundle input is missing: {relative}")
        files.append((relative, path))
    return files


def _source_bundle_archive(
    runtime_root: Path,
    *,
    source_receipt: dict[str, Any],
    run_id: str,
    include_ratings: bool,
) -> tuple[bytes, dict[str, Any], str]:
    file_records: list[dict[str, Any]] = []
    for relative, path in _source_bundle_files(runtime_root, include_ratings=include_ratings):
        file_records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _source_file_sha256(path),
            }
        )
    manifest = {
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "artifact_kind": "tier_list_source_bundle",
        "bundle_stage": source_receipt.get("bundle_stage", "complete"),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "source_mode": source_receipt["source_mode"],
        "expected_live_as_of": source_receipt["expected_live_as_of"],
        "source_observed_through": source_receipt["source_observed_through"],
        "source_steps": source_receipt["source_steps"],
        "files": file_records,
    }
    manifest_raw = (
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for relative, path in _source_bundle_files(
            runtime_root,
            include_ratings=include_ratings,
        ):
            archive.add(path, arcname=relative, recursive=False)
        info = tarfile.TarInfo(SOURCE_BUNDLE_MANIFEST)
        info.size = len(manifest_raw)
        info.mode = 0o600
        info.mtime = 0
        archive.addfile(info, fileobj=io.BytesIO(manifest_raw))
    archive_bytes = archive_buffer.getvalue()
    if len(archive_bytes) > SOURCE_BUNDLE_MAX_BYTES:
        raise RuntimeError(
            f"source bundle is larger than {SOURCE_BUNDLE_MAX_BYTES} bytes"
        )
    return archive_bytes, manifest, hashlib.sha256(manifest_raw).hexdigest()


def _publish_source_bundle(
    runtime_root: Path,
    *,
    source_receipt: dict[str, Any],
    run_id: str,
    include_ratings: bool = True,
    pointer_path: str = SOURCE_POINTER_PATH,
    bundle_prefix: str = SOURCE_BUNDLE_PREFIX,
) -> dict[str, Any]:
    archive_bytes, manifest, manifest_sha256 = _source_bundle_archive(
        runtime_root,
        source_receipt=source_receipt,
        run_id=run_id,
        include_ratings=include_ratings,
    )
    archive_sha256 = hashlib.sha256(archive_bytes).hexdigest()
    archive_path = f"{bundle_prefix}{archive_sha256}.tar.gz"
    archive_upload = _private_blob_upload(
        archive_path,
        archive_bytes,
        allow_overwrite=False,
    )
    pointer_unsigned = {
        "schema_version": SOURCE_BUNDLE_SCHEMA,
        "artifact_kind": "tier_list_source_pointer",
        "generated_at": manifest["generated_at"],
        "source_mode": manifest["source_mode"],
        "source_observed_through": manifest["source_observed_through"],
        "archive_path": archive_path,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha256,
        "manifest_sha256": manifest_sha256,
    }
    pointer = dict(pointer_unsigned)
    pointer["pointer_sha256"] = _canonical_sha256(pointer_unsigned)
    pointer_raw = (
        json.dumps(pointer, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    pointer_upload = _private_blob_upload(
        pointer_path,
        pointer_raw,
        allow_overwrite=True,
    )
    pointer_readback = None
    for attempt in range(31):
        try:
            pointer_readback = _private_blob_read(
                pointer_path,
                max_bytes=256 * 1024,
                cache_bust=True,
            )
        except FileNotFoundError:
            pointer_readback = None
        if pointer_readback == pointer_raw:
            break
        if attempt < 30:
            time.sleep(2)
    if pointer_readback != pointer_raw:
        raise RuntimeError("private source pointer failed exact readback")
    return {
        "status": "published",
        "pointer_path": pointer_path,
        "archive_path": archive_path,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha256,
        "manifest_sha256": manifest_sha256,
        "source_observed_through": manifest["source_observed_through"],
        "pointer_sha256": pointer["pointer_sha256"],
        "pointer_readback_verified": True,
        "archive_upload": {
            "pathname": archive_upload["pathname"],
            "bytes": archive_upload["bytes"],
            "sha256": archive_upload["sha256"],
        },
        "pointer_upload": {
            "pathname": pointer_upload["pathname"],
            "bytes": pointer_upload["bytes"],
            "sha256": pointer_upload["sha256"],
        },
    }


def _validate_source_pointer(
    payload: object,
    *,
    bundle_prefix: str,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("private source pointer is not an object")
    submitted = payload.get("pointer_sha256")
    unsigned = dict(payload)
    unsigned.pop("pointer_sha256", None)
    if (
        payload.get("schema_version") != SOURCE_BUNDLE_SCHEMA
        or payload.get("artifact_kind") != "tier_list_source_pointer"
        or not isinstance(submitted, str)
        or not hmac.compare_digest(submitted, _canonical_sha256(unsigned))
    ):
        raise RuntimeError("private source pointer is not an approved artifact")
    archive_path = payload.get("archive_path")
    if not isinstance(archive_path, str) or not re.fullmatch(
        rf"{re.escape(bundle_prefix)}[0-9a-f]{{64}}\.tar\.gz",
        archive_path,
    ):
        raise RuntimeError("private source pointer archive path is invalid")
    if not isinstance(payload.get("archive_bytes"), int) or payload["archive_bytes"] < 1:
        raise RuntimeError("private source pointer archive size is invalid")
    if not isinstance(payload.get("archive_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", payload["archive_sha256"]
    ):
        raise RuntimeError("private source pointer archive digest is invalid")
    if not isinstance(payload.get("manifest_sha256"), str) or not re.fullmatch(
        r"[0-9a-f]{64}", payload["manifest_sha256"]
    ):
        raise RuntimeError("private source pointer manifest digest is invalid")
    return payload


def _download_source_bundle(
    runtime_root: Path,
    *,
    pointer_path: str = SOURCE_POINTER_PATH,
    bundle_prefix: str = SOURCE_BUNDLE_PREFIX,
    required_steps: tuple[str, ...] = (
        "oe_api",
        "champion_atomization",
        "oe_live_source",
        "ratings",
    ),
    include_ratings: bool = True,
) -> dict[str, Any]:
    pointer_raw = _private_blob_read(pointer_path, max_bytes=256 * 1024)
    try:
        pointer = _validate_source_pointer(
            json.loads(pointer_raw.decode("utf-8")),
            bundle_prefix=bundle_prefix,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("private source pointer JSON is invalid") from error
    archive_path = str(pointer["archive_path"])
    token = _private_blob_token()
    url = _private_blob_url(archive_path, token)
    with tempfile.NamedTemporaryFile(
        prefix="scryglass-tier-source-", suffix=".tar.gz", dir="/tmp", delete=False
    ) as temporary:
        archive_name = temporary.name
        digest = hashlib.sha256()
        total = 0
        try:
            request = urllib.request.Request(
                url,
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > SOURCE_BUNDLE_MAX_BYTES:
                        raise RuntimeError("private source bundle exceeds the size limit")
                    digest.update(chunk)
                    temporary.write(chunk)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                raise FileNotFoundError(archive_path) from error
            raise RuntimeError(f"private source bundle read failed: HTTP {error.code}") from error
    try:
        if total != pointer["archive_bytes"] or digest.hexdigest() != pointer["archive_sha256"]:
            raise RuntimeError("private source bundle digest does not match its pointer")
        stage = Path(tempfile.mkdtemp(prefix="scryglass-tier-source-stage-", dir="/tmp"))
        try:
            allowed = set(
                SOURCE_RELATIVE_FILES if include_ratings else RAW_SOURCE_RELATIVE_FILES
            ) | {SOURCE_BUNDLE_MANIFEST}
            seen: set[str] = set()
            with tarfile.open(archive_name, mode="r:gz") as archive:
                for member in archive.getmembers():
                    if member.name not in allowed or member.name in seen:
                        raise RuntimeError("private source bundle contains an unexpected member")
                    if not member.isfile() or member.issym() or member.islnk():
                        raise RuntimeError("private source bundle contains a non-file member")
                    destination = (stage / member.name).resolve()
                    base = stage.resolve()
                    if os.path.commonpath((str(base), str(destination))) != str(base):
                        raise RuntimeError("private source bundle contains an unsafe path")
                    archive.extract(member, stage)
                    seen.add(member.name)
            manifest_path = stage / SOURCE_BUNDLE_MANIFEST
            if not manifest_path.is_file():
                raise RuntimeError("private source bundle has no manifest")
            manifest_raw = manifest_path.read_bytes()
            if hashlib.sha256(manifest_raw).hexdigest() != pointer["manifest_sha256"]:
                raise RuntimeError("private source bundle manifest digest mismatch")
            try:
                manifest = json.loads(manifest_raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("private source bundle manifest is invalid") from error
            if not isinstance(manifest, dict) or manifest.get("schema_version") != SOURCE_BUNDLE_SCHEMA:
                raise RuntimeError("private source bundle schema is invalid")
            if manifest.get("artifact_kind") != "tier_list_source_bundle":
                raise RuntimeError("private source bundle kind is invalid")
            if manifest.get("source_mode") != "oe_only":
                raise RuntimeError("private source bundle source mode is invalid")
            records = manifest.get("files")
            if not isinstance(records, list):
                raise RuntimeError("private source bundle has no file manifest")
            expected_files = SOURCE_RELATIVE_FILES if include_ratings else RAW_SOURCE_RELATIVE_FILES
            record_paths: set[str] = set()
            for record in records:
                if not isinstance(record, dict):
                    raise RuntimeError("private source bundle file record is invalid")
                relative = record.get("path")
                if relative not in expected_files or relative in record_paths:
                    raise RuntimeError("private source bundle file locator is invalid")
                record_paths.add(relative)
                source = stage / relative
                if (
                    not source.is_file()
                    or source.stat().st_size != record.get("bytes")
                    or _source_file_sha256(source) != record.get("sha256")
                ):
                    raise RuntimeError(f"private source bundle checksum failed: {relative}")
            if record_paths != set(expected_files):
                raise RuntimeError("private source bundle file set is incomplete")
            source_steps = manifest.get("source_steps")
            if not isinstance(source_steps, list):
                raise RuntimeError("private source bundle has no source steps")
            for required in required_steps:
                step = next(
                    (item for item in source_steps if isinstance(item, dict) and item.get("source") == required),
                    None,
                )
                if not isinstance(step, dict) or step.get("completed") is not True:
                    raise RuntimeError(f"private source bundle source step is incomplete: {required}")
            Path(archive_name).unlink(missing_ok=True)
            for relative in expected_files:
                destination = runtime_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                source = stage / relative
                try:
                    os.replace(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
            return manifest
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    finally:
        Path(archive_name).unlink(missing_ok=True)


class _RefreshLease:
    def __init__(self) -> None:
        from lol_kills.v2.tierlists.live_refresh import _publication_credentials

        base, token, store_id = _publication_credentials()
        self.transport = VercelBlobTransport(token, store_id)
        self.public_base = base
        self.store_id = store_id
        self.identity = None

    def _read_lock(self, *, deadline_epoch: int) -> tuple[bytes, Any] | None:
        try:
            return self.transport.get_blob(
                self.store_id,
                LOCK_PATH,
                deadline_epoch=deadline_epoch,
            )
        except Exception as error:  # noqa: BLE001
            print(
                f"[tier-refresh] lock authenticated read fallback error={type(error).__name__}",
                flush=True,
            )
            identity = self.transport._lookup(LOCK_PATH, deadline_epoch)
            if identity is None:
                return None
            url = f"{self.public_base}/{urllib.parse.quote(LOCK_PATH, safe='/')}"
            request = urllib.request.Request(url, headers={"Accept": "application/json"})
            try:
                with urllib.request.urlopen(request, timeout=20) as response:
                    body = response.read(64 * 1024 + 1)
            except urllib.error.HTTPError as http_error:
                if http_error.code == 404:
                    return None
                raise
            if len(body) > 64 * 1024 or len(body) != identity.size:
                raise RuntimeError("tier refresh lock body does not match its Blob identity")
            return body, identity

    def acquire(self) -> None:
        now = int(time.time())
        deadline = now + 30
        current = self._read_lock(deadline_epoch=deadline)
        run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex}"
        body = (
            json.dumps(
                {
                    "schema_version": "scryglass:tier-refresh-lock:v1",
                    "run_id": run_id,
                    "started_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "expires_at": now + LOCK_TTL_SECONDS,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        if current is None:
            identity = self.transport.put_if_absent(
                self.store_id,
                LOCK_PATH,
                body,
                deadline_epoch=deadline,
            )
        else:
            try:
                previous = json.loads(current[0].decode("utf-8"))
                expires_at = int(previous.get("expires_at", 0))
            except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
                expires_at = now + LOCK_TTL_SECONDS
            if expires_at > now:
                raise WorkerBusy("another tier refresh holds the Blob lease")
            identity = self.transport.put_if_match(
                self.store_id,
                LOCK_PATH,
                body,
                etag=current[1].etag,
                deadline_epoch=deadline,
            )
        if identity is None:
            raise WorkerBusy("another tier refresh acquired the Blob lease")
        self.identity = identity

    def release(self) -> None:
        if self.identity is None:
            return
        try:
            self.transport.delete_if_match(
                self.store_id,
                LOCK_PATH,
                etag=self.identity.etag,
                deadline_epoch=int(time.time()) + 30,
            )
        finally:
            self.identity = None


def _publish_receipt(receipt_path: Path) -> dict[str, Any]:
    from lol_kills.v2.tierlists.live_refresh import _publication_credentials

    _base, token, store_id = _publication_credentials()
    transport = VercelBlobTransport(token, store_id)
    raw = receipt_path.read_bytes()
    pathname = f"{RECEIPT_PREFIX}{receipt_path.name}"
    identity = transport.put_if_absent(
        store_id,
        pathname,
        raw,
        deadline_epoch=int(time.time()) + 30,
    )
    if identity is None:
        existing = transport.get_blob(store_id, pathname, deadline_epoch=int(time.time()) + 30)
        if existing is None or existing[0] != raw:
            raise RuntimeError("refresh receipt already exists with different content")
        identity = existing[1]
    return {
        "status": "published",
        "pathname": pathname,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "etag": identity.etag,
    }


def _production_digest(payload: dict[str, Any]) -> str:
    from lol_kills.v2.tierlists.production_bundle import _canonical_sha256

    return _canonical_sha256(payload)


def _validate_movement_snapshot(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError("tier-list movement snapshot is not an object")
    if (
        payload.get("schema_version") != "scryglass:tier-list-movement-snapshot:v1"
        or payload.get("artifact_kind") != "tier_list_movement_snapshot"
        or payload.get("status") != "production"
        or payload.get("production_eligible") is not True
        or payload.get("artifact_sha256") != _production_digest(payload)
    ):
        raise RuntimeError("tier-list movement snapshot is not an approved artifact")
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError("tier-list movement snapshot has no cells")
    return payload


def _json_url(url: str, *, timeout: float = 20.0) -> tuple[bytes, dict[str, Any]]:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise RuntimeError("tier-list source URL is invalid")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(8 * 1024 * 1024 + 1)
    if len(raw) > 8 * 1024 * 1024:
        raise RuntimeError("tier-list source JSON is too large")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("tier-list source JSON is invalid") from error
    if not isinstance(payload, dict):
        raise RuntimeError("tier-list source JSON is not an object")
    return raw, payload


def _load_previous_approved(runtime_root: Path, lease: _RefreshLease) -> Path | None:
    """Load the last approved movement rows into the candidate schema."""

    movement_path = runtime_root / "data/lol/v2/tierlists/previous-approved-movement-v1.json"
    movement_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        remote = lease.transport.get_blob(
            lease.store_id,
            MOVEMENT_PATH,
            deadline_epoch=int(time.time()) + 30,
        )
    except Exception as error:  # noqa: BLE001
        print(
            f"[tier-refresh] movement snapshot lookup fallback error={type(error).__name__}",
            flush=True,
        )
        remote = None
    if remote is not None:
        payload = _validate_movement_snapshot(json.loads(remote[0].decode("utf-8")))
        movement_path.write_text(
            json.dumps(
                {
                    "artifact_sha256": payload["artifact_sha256"],
                    "as_of": payload.get("as_of"),
                    "artifact_kind": payload["artifact_kind"],
                    "cells": payload["cells"],
                },
                ensure_ascii=True,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print("[tier-refresh] movement baseline=approved-snapshot", flush=True)
        return movement_path

    index_url = os.environ.get("SCRYGLASS_TIERLIST_INDEX_URL", "").strip()
    if not index_url:
        print("[tier-refresh] movement baseline=unavailable", flush=True)
        return None
    _, pointer = _json_url(index_url)
    if pointer.get("artifact_kind") != "tier_list_index_production" or pointer.get("production_eligible") is not True:
        raise RuntimeError("configured tier-list pointer is not an approved production index")
    if pointer.get("artifact_sha256") != _production_digest(pointer):
        raise RuntimeError("configured tier-list pointer digest is invalid")
    base_url = pointer.get("base_url")
    if not isinstance(base_url, str) or not re.fullmatch(r"\./releases/[0-9a-f]{64}/", base_url):
        raise RuntimeError("configured tier-list pointer base URL is invalid")
    release_url = urllib.parse.urljoin(index_url, base_url)
    cells = pointer.get("cells")
    if not isinstance(cells, list) or not cells:
        raise RuntimeError("configured tier-list pointer has no cells")

    def fetch_cell(meta: object) -> dict[str, Any]:
        if not isinstance(meta, dict):
            raise RuntimeError("tier-list pointer cell metadata is malformed")
        locator = meta.get("locator")
        if not isinstance(locator, str) or not re.fullmatch(r"cells/[A-Za-z0-9._-]+\.json", locator):
            raise RuntimeError("tier-list pointer cell locator is invalid")
        raw, payload = _json_url(urllib.parse.urljoin(release_url, locator))
        if hashlib.sha256(raw).hexdigest() != meta.get("raw_sha256"):
            raise RuntimeError("tier-list pointer cell digest does not match")
        scope = payload.get("scope")
        if not isinstance(scope, dict) or scope.get("scope_id") != meta.get("scope_id") or payload.get("role") != meta.get("role"):
            raise RuntimeError("tier-list pointer cell identity does not match")
        rows = payload.get("rows")
        if not isinstance(rows, list):
            raise RuntimeError("tier-list pointer cell rows are malformed")
        return {
            "scope_id": meta["scope_id"],
            "role": meta["role"],
            "as_of": meta.get("as_of"),
            "rows": [
                {
                    "champion_id": row.get("champion_id"),
                    "champion_name": row.get("champion_name"),
                    "rank": row.get("rank"),
                    "rating": row.get("rating"),
                }
                for row in rows
                if isinstance(row, dict)
            ],
        }

    with ThreadPoolExecutor(max_workers=16) as executor:
        movement_cells = list(executor.map(fetch_cell, cells))
    movement_path.write_text(
        json.dumps(
            {
                "artifact_sha256": pointer["artifact_sha256"],
                "as_of": pointer.get("as_of"),
                "artifact_kind": "tier_list_movement_snapshot",
                "cells": movement_cells,
            },
            ensure_ascii=True,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[tier-refresh] movement baseline=approved-cells cells={len(movement_cells)}", flush=True)
    return movement_path


def _pack_write_mode(pathname: str, existing: dict[str, Any]) -> WriteMode:
    return WriteMode.OVERWRITE if pathname in existing else WriteMode.NEW_IMMUTABLE


def _publish_public_pack(runtime_root: Path, *, run_id: str) -> dict[str, Any]:
    """Export and publish the live ratings pack under one retained Blob plan."""

    from lol_kills.export.public_pack import export_public_pack
    from lol_kills.v2.tierlists.live_refresh import _blob_inventory, _publication_credentials

    warehouse_root = runtime_root / "data/lol/warehouse/parquet/oe_live"
    out_root = runtime_root / "output/public_pack"
    latest = json.loads((runtime_root / PACK_LATEST).read_text(encoding="utf-8"))
    baseline_id = _safe_pack_id(latest.get("pack_id"))
    baseline_root = runtime_root / PACKS_ROOT / baseline_id
    static_model_source = baseline_root / "models"
    static_study_source = baseline_root / "studies" / "grubs"
    if not static_model_source.is_dir() or not static_study_source.is_dir():
        raise RuntimeError(
            "the committed public pack is missing its static model and study assets"
        )
    shutil.copytree(
        static_model_source,
        runtime_root / "data/lol/models",
        dirs_exist_ok=True,
    )
    shutil.copytree(
        static_study_source,
        runtime_root / "output/pdf",
        dirs_exist_ok=True,
    )
    manifest = export_public_pack(
        years=(2025, 2026),
        out_root=out_root,
        pack_id=LIVE_PACK_ID,
        warehouse_root=warehouse_root,
        project_root=runtime_root,
    )
    pack_dir = out_root / LIVE_PACK_ID
    blob_base, token, store_id = _publication_credentials()
    pack_base = f"{blob_base}/packs/{LIVE_PACK_ID}"
    published_manifest = dict(manifest)
    published_manifest["base_url"] = pack_base
    manifest_raw = (json.dumps(published_manifest, indent=2) + "\n").encode("utf-8")
    latest_payload = {
        "pack_id": LIVE_PACK_ID,
        "base_url": pack_base,
        "created_utc": published_manifest.get("created_utc"),
    }
    latest_raw = (json.dumps(latest_payload, indent=2) + "\n").encode("utf-8")
    pointer_raw = (json.dumps(published_manifest, indent=2) + "\n").encode("utf-8")

    transport = VercelBlobTransport(token, store_id)
    existing = _blob_inventory(transport, store_id)
    writes: list[PlannedWrite] = []
    pack_bytes = 0
    for path in sorted(pack_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(pack_dir).as_posix()
        raw = manifest_raw if relative == "manifest.json" else path.read_bytes()
        pathname = f"packs/{LIVE_PACK_ID}/{relative}"
        writes.append(
            PlannedWrite(pathname, raw, _pack_write_mode(pathname, existing))
        )
        pack_bytes += len(raw)
    writes.extend(
        (
            PlannedWrite(
                "packs/manifest.json",
                pointer_raw,
                _pack_write_mode("packs/manifest.json", existing),
            ),
            PlannedWrite(
                "packs/latest.json",
                latest_raw,
                _pack_write_mode("packs/latest.json", existing),
            ),
        )
    )
    plan = RetentionPlan(
        store_id=store_id,
        writer_id=os.environ.get("SCRYGLASS_TIERLIST_WRITER_ID", "scryglass-tierlist-worker"),
        run_id=f"{run_id}-{hashlib.sha256(manifest_raw).hexdigest()[:16]}",
        writes=tuple(writes),
    )
    result = RetentionExecutor(transport).execute(plan)
    if not result.success:
        failed = [operation.pathname for operation in result.operations if not operation.success]
        raise RuntimeError(
            "public pack Blob publication failed: "
            f"state={result.state.value}, failed={failed}"
        )

    readback_paths = (
        "packs/manifest.json",
        "packs/latest.json",
        f"packs/{LIVE_PACK_ID}/manifest.json",
    )
    readback: dict[str, bool] = {}
    for pathname, expected in (
        ("packs/manifest.json", pointer_raw),
        ("packs/latest.json", latest_raw),
        (f"packs/{LIVE_PACK_ID}/manifest.json", manifest_raw),
    ):
        remote = transport.get_blob(
            store_id,
            pathname,
            deadline_epoch=int(time.time()) + 30,
        )
        readback[pathname] = remote is not None and remote[0] == expected
    if not all(readback.values()):
        raise RuntimeError(f"public pack pointer readback failed: {readback}")
    return {
        "status": "published",
        "pack_id": LIVE_PACK_ID,
        "base_url": pack_base,
        "files": len(writes) - 2,
        "bytes": pack_bytes,
        "source_as_of": manifest.get("ratings", {}).get("source_as_of"),
        "readback_verified": readback,
        "retention": {
            "state": result.state.value,
            "current_retained_bytes": result.current_retained_bytes,
            "peak_retained_bytes": result.peak_retained_bytes,
            "projected_final_bytes": result.projected_final_bytes,
            "actual_final_bytes": result.actual_final_bytes,
            "policy_sha256": result.policy_sha256,
        },
    }


def _run_refresh() -> dict[str, Any]:
    started = time.monotonic()
    print("[tier-refresh] phase=prepare start", flush=True)
    runtime_root = _prepare_runtime_root()
    print(f"[tier-refresh] phase=prepare done seconds={time.monotonic() - started:.1f}", flush=True)
    expected = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex}"
    receipt_path = runtime_root / "data/lol/v2/tierlists/refresh-receipts" / f"tierlist-live-refresh-{run_id}.json"
    prior_runtime_root = os.environ.get("SCRYGLASS_RUNTIME_ROOT")
    prior_pythonpath = os.environ.get("PYTHONPATH")
    try:
        os.environ["SCRYGLASS_RUNTIME_ROOT"] = str(runtime_root)
        pythonpath = [str(PROJECT_ROOT), str(runtime_root)]
        if prior_pythonpath:
            pythonpath.append(prior_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
        from lol_kills.v2.tierlists.live_refresh import refresh_candidate

        lease = _RefreshLease()
        lease.acquire()
        print("[tier-refresh] phase=lease acquired", flush=True)
        from lol_kills.etl.restore_oe_pack_baseline import restore_baseline

        baseline = restore_baseline(runtime_root)
        print("[tier-refresh] phase=baseline restored", flush=True)
        source_manifest = _download_source_bundle(runtime_root)
        print(
            "[tier-refresh] phase=source bundle restored "
            f"source_as_of={source_manifest.get('source_observed_through')}",
            flush=True,
        )
        previous_path = _load_previous_approved(runtime_root, lease)
        receipt = refresh_candidate(
            runtime_root,
            expected_live_as_of=expected,
            previous_path=previous_path,
            output_path=Path("data/lol/v2/tierlists/champion-elo-candidate-v1.json"),
            receipt_path=receipt_path,
            source_mode="oe_only",
            promote=True,
            skip_annual_oe=True,
            skip_atom_bridge=True,
            prepared_source=source_manifest,
        )
        print("[tier-refresh] phase=tier-candidate promoted", flush=True)
        receipt_publication = _publish_receipt(receipt_path)
        print("[tier-refresh] phase=receipt published", flush=True)
        if receipt.get("status") != "production_promoted":
            raise RuntimeError(f"tier refresh did not promote: {receipt.get('status')}")
        pack_publication = {
            "status": "scheduled",
            "route": "/api/cron/pack-refresh",
            "schedule": "45 */6 * * *",
        }
        print(f"[tier-refresh] phase=pack deferred seconds={time.monotonic() - started:.1f}", flush=True)
        return {
            "status": "production_promoted",
            "run_id": run_id,
            "baseline": {
                "pack_id": baseline.get("pack_id"),
                "source_latest": baseline.get("source_latest"),
                "player_rows": baseline.get("outputs", {}).get("player_games", {}).get("rows"),
                "team_rows": baseline.get("outputs", {}).get("team_games", {}).get("rows"),
            },
            "receipt": {
                "status": receipt.get("status"),
                "candidate": receipt.get("candidate"),
                "source_observed_through": receipt.get("source_observed_through"),
                "receipt_canonical_sha256": receipt.get("receipt_canonical_sha256"),
            },
            "receipt_publication": receipt_publication,
            "pack_publication": pack_publication,
        }
    finally:
        try:
            if "lease" in locals():
                lease.release()
        finally:
            if prior_runtime_root is None:
                os.environ.pop("SCRYGLASS_RUNTIME_ROOT", None)
            else:
                os.environ["SCRYGLASS_RUNTIME_ROOT"] = prior_runtime_root
            if prior_pythonpath is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = prior_pythonpath
            shutil.rmtree(runtime_root, ignore_errors=True)


class handler(BaseHTTPRequestHandler):
    """Vercel Python runtime entry point."""

    def do_GET(self) -> None:  # noqa: N802
        self._handle()

    def do_POST(self) -> None:  # noqa: N802
        self._handle()

    def _handle(self) -> None:
        if not _authorized(self):
            _json_response(self, 401, {"status": "unauthorized"})
            return
        try:
            result = _run_refresh()
        except WorkerBusy as error:
            _json_response(self, 202, {"status": "busy", "reason": str(error)})
        except FileNotFoundError as error:
            _json_response(
                self,
                503,
                {
                    "status": "unavailable",
                    "code": "source_bundle_unavailable",
                    "reason": str(error),
                },
            )
        except WorkerConfigurationError as error:
            _json_response(self, 503, {"status": "unavailable", "code": "worker_not_configured", "reason": str(error)})
        except Exception as error:  # noqa: BLE001
            _json_response(self, 500, {"status": "failed", "code": "tier_refresh_failed", "reason": f"{type(error).__name__}: {error}"})
        else:
            _json_response(self, 200, result)
