"""Refresh and publish the live team and player ratings pack."""

from __future__ import annotations

import importlib.util
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


PACK_LOCK_PATH = "_scryglass_retention/pack-refresh-lock.json"


def _run_pack_refresh() -> dict[str, Any]:
    if not os.environ.get("ORACLES_ELIXIR_API_KEY", "").strip():
        raise _WORKER.WorkerConfigurationError("ORACLES_ELIXIR_API_KEY is not configured")
    started = time.monotonic()
    print("[pack-refresh] phase=prepare start", flush=True)
    runtime_root = _WORKER._prepare_runtime_root()
    print(f"[pack-refresh] phase=prepare done seconds={time.monotonic() - started:.1f}", flush=True)
    expected = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
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
        _WORKER.LOCK_PATH = PACK_LOCK_PATH
        lease = _WORKER._RefreshLease()
        lease.acquire()
        print("[pack-refresh] phase=lease acquired", flush=True)
        from lol_kills.etl.restore_oe_pack_baseline import restore_baseline

        baseline = restore_baseline(runtime_root)
        print("[pack-refresh] phase=baseline restored", flush=True)
        api_step = _WORKER._run_step(
            runtime_root,
            [
                "lol_kills.etl.oe_api_ingest",
                "--root",
                str(runtime_root),
                "--start",
                _WORKER.LIVE_WINDOW_START,
                "--end",
                expected,
                "--lookback-days",
                "120",
            ],
            source="oe_api",
        )
        observed_as_of = _WORKER._api_source_latest(runtime_root) if api_step["completed"] else None
        candidate_as_of = observed_as_of or expected
        live_step = (
            _WORKER._run_step(
                runtime_root,
                ["lol_kills.etl.oe_live_source", "--root", str(runtime_root)],
                source="oe_live_source",
            )
            if api_step["completed"]
            else _WORKER._skipped_step("oe_live_source", "oe_api_incomplete")
        )
        rating_step = (
            _WORKER._run_step(
                runtime_root,
                [
                    "lol_kills.v2.tierlists.rating_refresh",
                    "--root",
                    str(runtime_root),
                    "--as-of",
                    candidate_as_of,
                ],
                source="ratings",
            )
            if live_step["completed"] and _WORKER._api_player_detail_complete(runtime_root)
            else _WORKER._skipped_step("ratings", "oe_player_detail_incomplete")
        )
        if not api_step["completed"] or not live_step["completed"] or not rating_step["completed"]:
            raise RuntimeError(
                "ratings pack source preparation failed: "
                + _WORKER._source_step_failure([api_step, live_step, rating_step])
            )
        print("[pack-refresh] phase=ratings refreshed", flush=True)
        publication = _WORKER._publish_public_pack(runtime_root, run_id=run_id)
        print(f"[pack-refresh] phase=pack published seconds={time.monotonic() - started:.1f}", flush=True)
        return {
            "status": "published",
            "run_id": run_id,
            "baseline": {
                "pack_id": baseline.get("pack_id"),
                "source_latest": baseline.get("source_latest"),
                "player_rows": baseline.get("outputs", {}).get("player_games", {}).get("rows"),
                "team_rows": baseline.get("outputs", {}).get("team_games", {}).get("rows"),
            },
            "source_observed_through": observed_as_of,
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
            result = _run_pack_refresh()
        except _WORKER.WorkerBusy as error:
            _WORKER._json_response(self, 202, {"status": "busy", "reason": str(error)})
        except _WORKER.WorkerConfigurationError as error:
            _WORKER._json_response(
                self,
                503,
                {"status": "unavailable", "code": "worker_not_configured", "reason": str(error)},
            )
        except Exception as error:  # noqa: BLE001
            _WORKER._json_response(
                self,
                500,
                {
                    "status": "failed",
                    "code": "pack_refresh_failed",
                    "reason": f"{type(error).__name__}: {error}",
                },
            )
        else:
            _WORKER._json_response(self, 200, result)
