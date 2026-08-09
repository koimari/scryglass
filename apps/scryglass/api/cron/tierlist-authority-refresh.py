"""Run the tier-list authority gates after candidate calculation completes."""

from __future__ import annotations

import importlib.util
import hashlib
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


AUTHORITY_LOCK_PATH = "_scryglass_retention/tierlist-authority-refresh-lock.json"


def _run_authority_refresh() -> dict[str, Any]:
    started = time.monotonic()
    print("[tier-authority] phase=prepare start", flush=True)
    runtime_root = _WORKER._prepare_runtime_root(include_baseline_pack=False)
    print(
        f"[tier-authority] phase=prepare done seconds={time.monotonic() - started:.1f}",
        flush=True,
    )
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex}"
    receipt_path = (
        runtime_root
        / "data/lol/v2/tierlists/refresh-receipts"
        / f"tierlist-authority-refresh-{run_id}.json"
    )
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
        _WORKER.LOCK_PATH = AUTHORITY_LOCK_PATH
        lease = _WORKER._RefreshLease()
        lease.acquire()
        print("[tier-authority] phase=lease acquired", flush=True)

        source_manifest = _WORKER._download_source_bundle(runtime_root)
        print(
            "[tier-authority] phase=source bundle restored "
            f"source_as_of={source_manifest.get('source_observed_through')}",
            flush=True,
        )
        candidate = _WORKER._download_candidate(runtime_root)
        candidate_path = runtime_root / "data/lol/v2/tierlists/champion-elo-candidate-v1.json"
        candidate_raw = candidate_path.read_bytes()
        if candidate.get("source_mode") != "oe_only":
            raise RuntimeError("tier-list candidate source mode is not OE-only")
        if candidate.get("source_complete_through_expected_live_as_of") is not True:
            raise RuntimeError("tier-list candidate source is incomplete")
        print(
            "[tier-authority] phase=candidate restored "
            f"as_of={candidate.get('as_of')}",
            flush=True,
        )

        forward_step = _WORKER._run_step(
            runtime_root,
            [
                "lol_kills.v2.tierlists.forward_evaluation",
                "--root",
                str(runtime_root),
                "--output",
                "data/lol/v2/tierlists/prospective-evaluation-v1.json",
            ],
            source="forward_evaluation",
        )
        if not forward_step["completed"]:
            raise RuntimeError("forward evaluation failed")
        print("[tier-authority] phase=forward evaluation complete", flush=True)

        authority_step = _WORKER._run_step(
            runtime_root,
            [
                "lol_kills.v2.tierlists.independent_authority",
                "--root",
                str(runtime_root),
                "--output",
                "data/lol/v2/tierlists/independent-l2-authority-v1.json",
            ],
            source="independent_authority",
        )
        if not authority_step["completed"]:
            raise RuntimeError("independent authority failed")
        print("[tier-authority] phase=independent authority complete", flush=True)

        bundle_step = _WORKER._run_step(
            runtime_root,
            [
                "lol_kills.v2.tierlists.production_bundle",
                "--root",
                str(runtime_root),
            ],
            source="production_bundle",
        )
        if not bundle_step["completed"]:
            raise RuntimeError("production bundle failed")
        print("[tier-authority] phase=production bundle complete", flush=True)

        from lol_kills.v2.tierlists.live_refresh import publish_production_bundle

        publication = publish_production_bundle(runtime_root)
        print("[tier-authority] phase=production bundle published", flush=True)

        receipt: dict[str, Any] = {
            "schema_version": "scryglass:tierlist-live-refresh-receipt:v1",
            "source_mode": "oe_only",
            "status": "production_promoted",
            "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "source_observed_through": source_manifest.get("source_observed_through"),
            "source_steps": source_manifest.get("source_steps", []),
            "promotion_steps": [forward_step, authority_step, bundle_step],
            "candidate": {
                "locator": "data/lol/v2/tierlists/champion-elo-candidate-v1.json",
                "raw_sha256": hashlib.sha256(candidate_raw).hexdigest(),
                "artifact_sha256": candidate.get("artifact_sha256"),
                "as_of": candidate.get("as_of"),
                "maps_replayed": candidate.get("source", {}).get("maps_replayed"),
                "maps_in_live_window": candidate.get("source", {}).get("maps_in_live_window"),
                "source_complete_through_expected_live_as_of": candidate.get(
                    "source_complete_through_expected_live_as_of"
                ),
                "source_mode": candidate.get("source_mode"),
                "rating_refresh_completed": True,
            },
            "authority": {
                "source_freshness": True,
                "model_validation": True,
                "publication": True,
                "rank_eligibility": True,
                "recommendation": False,
                "betting": False,
            },
            "publication": publication,
            "claim_ceiling": (
                "This receipt records a source-bound descriptive production bundle. "
                "It does not authorize outcome-calibrated probability, causal, "
                "recommendation, or betting claims."
            ),
        }
        receipt["receipt_canonical_sha256"] = _WORKER._canonical_sha256(
            {key: value for key, value in receipt.items() if key != "receipt_canonical_sha256"}
        )
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text(
            json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        receipt_publication = _WORKER._publish_receipt(receipt_path)
        print(
            f"[tier-authority] phase=complete seconds={time.monotonic() - started:.1f}",
            flush=True,
        )
        return {
            "status": "production_promoted",
            "run_id": run_id,
            "source_observed_through": source_manifest.get("source_observed_through"),
            "candidate": receipt["candidate"],
            "promotion": {
                "forward_evaluation": forward_step,
                "independent_authority": authority_step,
                "production_bundle": bundle_step,
            },
            "publication": publication,
            "receipt_publication": receipt_publication,
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
            result = _run_authority_refresh()
        except _WORKER.WorkerBusy as error:
            _WORKER._json_response(self, 202, {"status": "busy", "reason": str(error)})
        except FileNotFoundError as error:
            _WORKER._json_response(
                self,
                503,
                {
                    "status": "unavailable",
                    "code": "candidate_unavailable",
                    "reason": str(error),
                },
            )
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
                    "code": "tier_authority_refresh_failed",
                    "reason": f"{type(error).__name__}: {error}",
                },
            )
        else:
            _WORKER._json_response(self, 200, result)
