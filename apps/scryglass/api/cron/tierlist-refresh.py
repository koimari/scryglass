"""Run one audited tier-list refresh inside a Vercel Python Function."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


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
PACKS_ROOT = Path("apps/scryglass/public/packs")
PACK_LATEST = PACKS_ROOT / "latest.json"
LIVE_PACK_ID = os.environ.get("SCRYGLASS_LIVE_PACK_ID", "v2026.live")


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
    expected = (
        os.environ.get("SCRYGLASS_TIERLIST_INGEST_TOKEN")
        or os.environ.get("CRON_SECRET")
        or ""
    ).strip()
    supplied = handler.headers.get("Authorization", "")
    if not expected or not supplied.startswith("Bearer "):
        return False
    return hmac.compare_digest(supplied[7:].strip(), expected)


def _safe_pack_id(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise WorkerConfigurationError("the committed public pack id is malformed")
    return value


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir() or source.is_symlink():
        raise WorkerConfigurationError(f"required worker input is missing: {source}")
    shutil.copytree(source, destination, symlinks=False)


def _prepare_runtime_root() -> Path:
    """Copy writable model inputs into Vercel's temporary filesystem."""

    root = Path(tempfile.mkdtemp(prefix="scryglass-tier-refresh-", dir="/tmp"))
    try:
        for relative in (
            Path("data/lol/v2/champions"),
            Path("data/lol/v2/tierlists"),
            Path("data/lol/features"),
            Path("data/lol/models"),
            Path("output/pdf"),
        ):
            _copy_tree(PROJECT_ROOT / relative, root / relative)

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


class _RefreshLease:
    def __init__(self) -> None:
        from lol_kills.v2.tierlists.live_refresh import _publication_credentials

        _base, token, store_id = _publication_credentials()
        self.transport = VercelBlobTransport(token, store_id)
        self.store_id = store_id
        self.identity = None

    def acquire(self) -> None:
        now = int(time.time())
        deadline = now + 30
        current = self.transport.get_blob(
            self.store_id,
            LOCK_PATH,
            deadline_epoch=deadline,
        )
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


def _pack_write_mode(pathname: str, existing: dict[str, Any]) -> WriteMode:
    return WriteMode.OVERWRITE if pathname in existing else WriteMode.NEW_IMMUTABLE


def _publish_public_pack(runtime_root: Path, *, run_id: str) -> dict[str, Any]:
    """Export and publish the live ratings pack under one retained Blob plan."""

    from lol_kills.export.public_pack import export_public_pack
    from lol_kills.v2.tierlists.live_refresh import _blob_inventory, _publication_credentials

    warehouse_root = runtime_root / "data/lol/warehouse/parquet/oe_live"
    out_root = runtime_root / "output/public_pack"
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
    if not os.environ.get("ORACLES_ELIXIR_API_KEY", "").strip():
        raise WorkerConfigurationError("ORACLES_ELIXIR_API_KEY is not configured")
    runtime_root = _prepare_runtime_root()
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
        from lol_kills.etl.restore_oe_pack_baseline import restore_baseline

        baseline = restore_baseline(runtime_root)
        receipt = refresh_candidate(
            runtime_root,
            expected_live_as_of=expected,
            output_path=Path("data/lol/v2/tierlists/champion-elo-candidate-v1.json"),
            receipt_path=receipt_path,
            source_mode="oe_only",
            promote=True,
            skip_annual_oe=True,
            skip_atom_bridge=True,
        )
        receipt_publication = _publish_receipt(receipt_path)
        if receipt.get("status") != "production_promoted":
            raise RuntimeError(f"tier refresh did not promote: {receipt.get('status')}")
        pack_publication = _publish_public_pack(runtime_root, run_id=run_id)
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
        except WorkerConfigurationError as error:
            _json_response(self, 503, {"status": "unavailable", "code": "worker_not_configured", "reason": str(error)})
        except Exception as error:  # noqa: BLE001
            _json_response(self, 500, {"status": "failed", "code": "tier_refresh_failed", "reason": f"{type(error).__name__}: {error}"})
        else:
            _json_response(self, 200, result)
