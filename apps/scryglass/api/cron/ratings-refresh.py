"""Fit live team and player ratings from the private OE source bundle."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any


_WORKER_PATH = Path(__file__).with_name("tierlist-refresh.py")
_SPEC = importlib.util.spec_from_file_location("scryglass_tierlist_worker", _WORKER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError("tier-list worker module cannot be loaded")
_WORKER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _WORKER
_SPEC.loader.exec_module(_WORKER)


RATINGS_LOCK_PATH = _WORKER.RATINGS_LOCK_PATH


def _run_ratings_refresh() -> dict[str, Any]:
    started = time.monotonic()
    print("[ratings-refresh] phase=prepare start", flush=True)
    runtime_root = _WORKER._prepare_runtime_root(include_baseline_pack=False)
    print(
        f"[ratings-refresh] phase=prepare done seconds={time.monotonic() - started:.1f}",
        flush=True,
    )
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex}"
    prior_runtime_root = os.environ.get("SCRYGLASS_RUNTIME_ROOT")
    prior_pythonpath = os.environ.get("PYTHONPATH")
    prior_lock_path = _WORKER.LOCK_PATH
    lease = None
    try:
        os.environ["SCRYGLASS_RUNTIME_ROOT"] = str(runtime_root)
        pythonpath = [str(_WORKER.PROJECT_ROOT), str(runtime_root)]
        if prior_pythonpath:
            pythonpath.append(prior_pythonpath)
        os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)
        _WORKER.LOCK_PATH = RATINGS_LOCK_PATH
        lease = _WORKER._RefreshLease()
        lease.acquire()
        print("[ratings-refresh] phase=lease acquired", flush=True)
        latest = json.loads(
            (runtime_root / _WORKER.PACK_LATEST).read_text(encoding="utf-8")
        )
        baseline = {
            "pack_id": latest.get("pack_id"),
            "source_latest": None,
            "player_rows": None,
            "team_rows": None,
        }
        print("[ratings-refresh] phase=baseline pointer read", flush=True)
        raw_source = _WORKER._download_source_bundle(
            runtime_root,
            pointer_path=_WORKER.RAW_SOURCE_POINTER_PATH,
            bundle_prefix=_WORKER.RAW_SOURCE_BUNDLE_PREFIX,
            required_steps=("oe_api", "champion_atomization", "oe_live_source"),
            include_ratings=False,
        )
        print(
            "[ratings-refresh] phase=raw source restored "
            f"source_as_of={raw_source.get('source_observed_through')}",
            flush=True,
        )
        source_receipt = _WORKER._refresh_rating_inputs(
            runtime_root,
            source_manifest=raw_source,
        )
        print("[ratings-refresh] phase=ratings refreshed", flush=True)
        publication = _WORKER._publish_source_bundle(
            runtime_root,
            source_receipt=source_receipt,
            run_id=run_id,
        )
        print(
            f"[ratings-refresh] phase=source bundle published seconds={time.monotonic() - started:.1f}",
            flush=True,
        )
        return {
            "status": "published",
            "run_id": run_id,
            "source_observed_through": source_receipt["source_observed_through"],
            "baseline": {
                "pack_id": baseline.get("pack_id"),
                "source_latest": baseline.get("source_latest"),
                "player_rows": baseline.get("player_rows"),
                "team_rows": baseline.get("team_rows"),
            },
            "source_steps": source_receipt["source_steps"],
            "publication": publication,
        }
    finally:
        try:
            if lease is not None:
                lease.release()
        finally:
            _WORKER.LOCK_PATH = prior_lock_path
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
        if not _WORKER._authorized(self):
            _WORKER._json_response(self, 401, {"status": "unauthorized"})
            return
        try:
            result = _run_ratings_refresh()
        except _WORKER.WorkerBusy as error:
            _WORKER._json_response(self, 202, {"status": "busy", "reason": str(error)})
        except FileNotFoundError as error:
            _WORKER._json_response(
                self,
                503,
                {
                    "status": "unavailable",
                    "code": "source_bundle_unavailable",
                    "reason": str(error),
                },
            )
        except _WORKER.WorkerConfigurationError as error:
            _WORKER._json_response(
                self,
                503,
                {
                    "status": "unavailable",
                    "code": "worker_not_configured",
                    "reason": str(error),
                },
            )
        except Exception as error:  # noqa: BLE001
            _WORKER._json_response(
                self,
                500,
                {
                    "status": "failed",
                    "code": "ratings_refresh_failed",
                    "reason": f"{type(error).__name__}: {error}",
                },
            )
        else:
            _WORKER._json_response(self, 200, result)
