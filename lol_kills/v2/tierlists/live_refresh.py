"""Refresh the tier-list candidate from OE with an optional GRID bridge.

This command is for a durable worker. It writes a development candidate and a
hash-bound receipt. With ``--promote``, it runs the descriptive evaluation,
independent authority check, and production bundle after the source gates pass.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from lol_kills.export.blob_retention import (
    BlobIdentity,
    PlannedWrite,
    RetentionExecutor,
    RetentionPlan,
    WriteMode,
)
from lol_kills.export.vercel_blob_transport import VercelBlobTransport
from lol_kills.v2.champions.atoms.consume import AtomBridge

from .champion_elo import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE_MODE,
    SOURCE_MODES,
    build_candidate,
    write_candidate,
)
from .production_bundle import _canonical as _production_canonical
from .production_bundle import _sha256_bytes as _production_sha256_bytes
from .production_bundle import verify_production_index

HISTORY_START = "2025-01-01T00:00:00Z"
LIVE_WINDOW_START = "2026-07-18T00:00:00Z"
RECEIPT_SCHEMA = "scryglass:tierlist-live-refresh:v1"
DEFAULT_RECEIPT = Path("data/lol/v2/tierlists/refresh-receipts")
BLOB_BASE_ENV = "SCRYGLASS_TIERLIST_BLOB_BASE_URL"
BLOB_BASE_FALLBACK_ENV = "LIVE_BLOB_BASE_URL"
BLOB_TOKEN_ENV = "BLOB_READ_WRITE_TOKEN"
BLOB_TOKEN_FALLBACK_ENV = "VERCEL_BLOB_READ_WRITE_TOKEN"
BLOB_STORE_ENV = "BLOB_STORE_ID"
BLOB_STORE_FALLBACK_ENV = "VERCEL_BLOB_STORE_ID"
BLOB_POINTER_PATH = "tierlists/index-v1.json"
BLOB_MOVEMENT_PATH = "tierlists/movement-v1.json"
BLOB_WRITER_ID = "scryglass-tierlist-worker"


class PublicationError(RuntimeError):
    """Raised when the production artifact cannot be published safely."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _production_artifact_bytes(payload: dict[str, Any]) -> bytes:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    payload["artifact_sha256"] = _production_sha256_bytes(_production_canonical(unsigned))
    return _production_canonical(payload) + b"\n"


def _publication_credentials() -> tuple[str, str, str]:
    token = (os.environ.get(BLOB_TOKEN_ENV) or os.environ.get(BLOB_TOKEN_FALLBACK_ENV) or "").strip()
    if not token:
        raise PublicationError(f"{BLOB_TOKEN_ENV} is not configured")
    base = (os.environ.get(BLOB_BASE_ENV) or os.environ.get(BLOB_BASE_FALLBACK_ENV) or "").strip()
    if not base:
        raise PublicationError(
            f"{BLOB_BASE_ENV} or {BLOB_BASE_FALLBACK_ENV} is not configured"
        )
    try:
        parsed = urlsplit(base)
        if parsed.scheme != "https" or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError
        host = parsed.hostname or ""
        suffix = ".public.blob.vercel-storage.com"
        if not host.endswith(suffix) or host == suffix:
            raise ValueError
        host_store = VercelBlobTransport._normalize_store_id(host[: -len(suffix)])
        token_store = VercelBlobTransport._read_write_token_store(token)
    except (ValueError, TypeError):
        raise PublicationError(
            "tier-list Blob configuration must use the matching HTTPS public Blob root and read-write token"
        ) from None
    if host_store != token_store:
        raise PublicationError("tier-list Blob base URL and token identify different stores")
    explicit_store = os.environ.get(BLOB_STORE_ENV) or os.environ.get(BLOB_STORE_FALLBACK_ENV)
    if explicit_store:
        try:
            if VercelBlobTransport._normalize_store_id(explicit_store.strip()) != token_store:
                raise PublicationError("explicit Blob store ID does not match the read-write token")
        except ValueError:
            raise PublicationError("explicit Blob store ID is malformed") from None
    return base.rstrip("/"), token, token_store


def _blob_inventory(transport: VercelBlobTransport, store_id: str) -> dict[str, BlobIdentity]:
    deadline = int(time.time()) + 30
    cursor: str | None = None
    seen_cursors: set[str] = set()
    identities: dict[str, BlobIdentity] = {}
    while True:
        page = transport.list_page(
            store_id,
            cursor=cursor,
            limit=1000,
            deadline_epoch=deadline,
        )
        raw_blobs = page.get("blobs")
        if not isinstance(raw_blobs, list):
            raise PublicationError("Blob inventory response is malformed")
        for raw in raw_blobs:
            if not isinstance(raw, dict):
                raise PublicationError("Blob inventory entry is malformed")
            identity = BlobIdentity(raw["pathname"], raw["size"], raw["etag"])
            if identity.pathname in identities:
                raise PublicationError("Blob inventory contains a duplicate pathname")
            identities[identity.pathname] = identity
        if page.get("hasMore") is not True:
            return identities
        next_cursor = page.get("cursor")
        if not isinstance(next_cursor, str) or not next_cursor or next_cursor in seen_cursors:
            raise PublicationError("Blob inventory pagination is incomplete")
        seen_cursors.add(next_cursor)
        cursor = next_cursor


def _validate_existing_pointer(raw: bytes) -> None:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationError("existing tier-list pointer is not valid JSON") from error
    if (
        not isinstance(payload, dict)
        or payload.get("artifact_kind") != "tier_list_index_production"
        or payload.get("development_only") is not False
        or payload.get("production_eligible") is not True
        or payload.get("publication_eligible") is not True
        or payload.get("artifact_sha256")
        != _production_sha256_bytes(
            _production_canonical({key: value for key, value in payload.items() if key != "artifact_sha256"})
        )
    ):
        raise PublicationError("existing tier-list pointer is not an approved production index")


def _publication_payloads(root: Path) -> dict[str, Any]:
    report = verify_production_index(root)
    index_path = root / "data/lol/v2/tierlists/production/index-v1.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or report["artifact_sha256"] != index.get("artifact_sha256"):
        raise PublicationError("production index changed during publication preparation")
    release_id = str(index["artifact_sha256"])
    release_index = deepcopy(index)
    release_index["base_url"] = "./"
    cell_bytes: dict[str, bytes] = {}
    for cell in release_index["cells"]:
        locator = cell.get("locator")
        if not isinstance(locator, str):
            raise PublicationError("production cell locator is malformed")
        local_path = root / Path(locator)
        raw = local_path.read_bytes()
        if _production_sha256_bytes(raw) != cell.get("raw_sha256"):
            raise PublicationError("production cell digest changed during publication preparation")
        filename = Path(locator).name
        remote_locator = f"cells/{filename}"
        cell["locator"] = remote_locator
        cell_bytes[f"tierlists/releases/{release_id}/{remote_locator}"] = raw
    release_index_raw = _production_artifact_bytes(release_index)
    pointer_index = deepcopy(release_index)
    pointer_index["base_url"] = f"./releases/{release_id}/"
    pointer_raw = _production_artifact_bytes(pointer_index)
    movement_snapshot = {
        "schema_version": "scryglass:tier-list-movement-snapshot:v1",
        "artifact_kind": "tier_list_movement_snapshot",
        "status": "production",
        "production_eligible": True,
        "as_of": index["as_of"],
        "source_index_artifact_sha256": index["artifact_sha256"],
        "cells": [],
    }
    for cell_meta in index["cells"]:
        locator = cell_meta.get("locator")
        if not isinstance(locator, str):
            raise PublicationError("production cell locator is malformed")
        local_path = root / Path(locator)
        cell_payload = json.loads(local_path.read_text(encoding="utf-8"))
        movement_rows = []
        for row in cell_payload.get("rows", []):
            movement_rows.append(
                {
                    "champion_id": row.get("champion_id"),
                    "champion_name": row.get("champion_name"),
                    "rank": row.get("rank"),
                    "rating": row.get("rating"),
                }
            )
        movement_snapshot["cells"].append(
            {
                "scope_id": cell_meta["scope_id"],
                "role": cell_meta["role"],
                "as_of": cell_meta["as_of"],
                "rows": movement_rows,
            }
        )
    movement_raw = _production_artifact_bytes(movement_snapshot)
    return {
        "release_id": release_id,
        "release_index_path": f"tierlists/releases/{release_id}/index-v1.json",
        "release_index_raw": release_index_raw,
        "release_index_artifact_sha256": release_index["artifact_sha256"],
        "cell_bytes": cell_bytes,
        "pointer_raw": pointer_raw,
        "pointer_artifact_sha256": pointer_index["artifact_sha256"],
        "pointer_url_suffix": BLOB_POINTER_PATH,
        "movement_raw": movement_raw,
        "movement_artifact_sha256": movement_snapshot["artifact_sha256"],
        "movement_url_suffix": BLOB_MOVEMENT_PATH,
        "cell_count": len(cell_bytes),
        "source_index_artifact_sha256": index["artifact_sha256"],
    }


def publish_production_bundle(root: Path | str = Path(".")) -> dict[str, Any]:
    """Publish immutable tier-list files, then replace the stable pointer."""

    repo_root = Path(root)
    blob_base, token, store_id = _publication_credentials()
    payloads = _publication_payloads(repo_root)
    transport = VercelBlobTransport(token, store_id)
    inventory = _blob_inventory(transport, store_id)
    pointer_identity = inventory.get(BLOB_POINTER_PATH)
    movement_identity = inventory.get(BLOB_MOVEMENT_PATH)
    pointer_mode = WriteMode.NEW_IMMUTABLE
    if pointer_identity is not None:
        current = transport.get_blob(
            store_id,
            BLOB_POINTER_PATH,
            deadline_epoch=int(time.time()) + 30,
        )
        if current is None or current[1] != pointer_identity:
            raise PublicationError("existing tier-list pointer changed during inventory")
        _validate_existing_pointer(current[0])
        pointer_mode = WriteMode.OVERWRITE
    movement_mode = WriteMode.NEW_IMMUTABLE
    if movement_identity is not None:
        current_movement = transport.get_blob(
            store_id,
            BLOB_MOVEMENT_PATH,
            deadline_epoch=int(time.time()) + 30,
        )
        if current_movement is None or current_movement[1] != movement_identity:
            raise PublicationError("existing tier-list movement snapshot changed during inventory")
        movement_mode = WriteMode.OVERWRITE

    writes = [
        PlannedWrite(pathname, raw, WriteMode.NEW_IMMUTABLE)
        for pathname, raw in sorted(payloads["cell_bytes"].items())
    ]
    writes.append(
        PlannedWrite(
            payloads["release_index_path"],
            payloads["release_index_raw"],
            WriteMode.NEW_IMMUTABLE,
        )
    )
    writes.append(
        PlannedWrite(
            BLOB_MOVEMENT_PATH,
            payloads["movement_raw"],
            movement_mode,
        )
    )
    writes.append(PlannedWrite(BLOB_POINTER_PATH, payloads["pointer_raw"], pointer_mode))
    plan = RetentionPlan(
        store_id=store_id,
        writer_id=os.environ.get("SCRYGLASS_TIERLIST_WRITER_ID", BLOB_WRITER_ID),
        run_id=payloads["release_id"],
        writes=tuple(writes),
    )
    result = RetentionExecutor(transport).execute(plan)
    if not result.success:
        failed = [operation.pathname for operation in result.operations if not operation.success]
        raise PublicationError(
            "Blob publication failed before the stable pointer was proven: "
            f"state={result.state.value}, failed={failed}"
        )
    readback = transport.get_blob(
        store_id,
        BLOB_POINTER_PATH,
        deadline_epoch=int(time.time()) + 30,
    )
    if readback is None or readback[0] != payloads["pointer_raw"]:
        raise PublicationError("published tier-list pointer failed exact readback")
    movement_readback = transport.get_blob(
        store_id,
        BLOB_MOVEMENT_PATH,
        deadline_epoch=int(time.time()) + 30,
    )
    if movement_readback is None or movement_readback[0] != payloads["movement_raw"]:
        raise PublicationError("published tier-list movement snapshot failed exact readback")
    return {
        "status": "published",
        "blob_store_id": store_id,
        "pointer_path": BLOB_POINTER_PATH,
        "index_url": f"{blob_base}/{BLOB_POINTER_PATH}",
        "release_id": payloads["release_id"],
        "release_index_path": payloads["release_index_path"],
        "source_index_artifact_sha256": payloads["source_index_artifact_sha256"],
        "pointer_artifact_sha256": payloads["pointer_artifact_sha256"],
        "release_index_artifact_sha256": payloads["release_index_artifact_sha256"],
        "cell_count": payloads["cell_count"],
        "pointer_mode": pointer_mode.value,
        "movement_path": BLOB_MOVEMENT_PATH,
        "movement_artifact_sha256": payloads["movement_artifact_sha256"],
        "movement_mode": movement_mode.value,
        "movement_readback_verified": True,
        "pointer_readback_verified": True,
        "retention": {
            "state": result.state.value,
            "current_retained_bytes": result.current_retained_bytes,
            "peak_retained_bytes": result.peak_retained_bytes,
            "projected_final_bytes": result.projected_final_bytes,
            "actual_final_bytes": result.actual_final_bytes,
            "policy_sha256": result.policy_sha256,
        },
    }


def _run_step(root: Path, args: list[str], *, source: str) -> dict[str, Any]:
    started = time.monotonic()
    print(f"[tier-refresh] step={source} start", flush=True)
    environment = os.environ.copy()
    inherited_paths = [str(Path(item)) for item in sys.path if item and Path(item).is_dir()]
    configured_paths = [item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item]
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([*inherited_paths, *configured_paths])
    )
    result = subprocess.run(
        [sys.executable, "-m", *args],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    print(
        f"[tier-refresh] step={source} done returncode={result.returncode} seconds={time.monotonic() - started:.1f}",
        flush=True,
    )
    return {
        "source": source,
        "command": args,
        "returncode": result.returncode,
        "completed": result.returncode == 0,
        "stdout_bytes": len(result.stdout.encode("utf-8")),
        "stderr_bytes": len(result.stderr.encode("utf-8")),
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _skipped_step(source: str, reason: str) -> dict[str, Any]:
    return {
        "source": source,
        "command": [],
        "returncode": None,
        "completed": False,
        "skipped": True,
        "reason": reason,
    }


def _source_step_failure(steps: list[dict[str, Any]]) -> str:
    failures: list[str] = []
    for step in steps:
        if step.get("completed") is True:
            continue
        source = str(step.get("source") or "unknown")
        reason = str(step.get("reason") or "step returned a non-zero status")
        stderr = str(step.get("stderr_tail") or "").strip()
        stdout = str(step.get("stdout_tail") or "").strip()
        detail = stderr or stdout
        if detail:
            reason = f"{reason}; {detail[-1200:]}"
        failures.append(f"{source}: {reason}")
    return " | ".join(failures)


def _verify_prebuilt_atom_bridge(root: Path) -> dict[str, Any]:
    """Verify the committed atom bridge when the worker has no LCC checkout."""

    path = root / "data/lol/v2/champions/lcc-atom-bridge-v1.json"
    try:
        bridge = AtomBridge.load(path)
    except Exception as error:  # noqa: BLE001
        return {
            "source": "champion_atomization",
            "command": [],
            "returncode": 2,
            "completed": False,
            "stdout_bytes": 0,
            "stderr_bytes": len(f"{type(error).__name__}: {error}".encode("utf-8")),
            "reason": "prebuilt_atom_bridge_invalid",
        }
    return {
        "source": "champion_atomization",
        "command": [],
        "returncode": 0,
        "completed": True,
        "skipped": True,
        "reason": "prebuilt_atom_bridge_verified",
        "artifact_sha256": bridge.artifact_sha256,
        "generated_at": bridge.generated_at,
    }


def _api_source_latest(root: Path) -> str | None:
    path = root / "data/lol/warehouse/parquet/oe_api_meta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = payload.get("source_latest")
    return value if isinstance(value, str) else None


def _api_player_detail_complete(root: Path) -> bool:
    path = root / "data/lol/warehouse/parquet/oe_api_meta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    return payload.get("player_detail_complete") is True


def _previous_week_start(value: str) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    week_start = stamp.normalize() - pd.Timedelta(days=(stamp.weekday() + 1) % 7)
    return week_start - pd.Timedelta(days=7)


def refresh_candidate(
    root: Path,
    *,
    expected_live_as_of: str,
    previous_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
    receipt_path: Path = DEFAULT_RECEIPT,
    grid_days: int = 21,
    grid_limit: int = 200,
    source_mode: str = DEFAULT_SOURCE_MODE,
    promote: bool = False,
    skip_annual_oe: bool = False,
    skip_atom_bridge: bool = False,
    prepared_source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if grid_days < 1 or grid_limit < 1:
        raise ValueError("grid_days and grid_limit must be positive")
    if source_mode not in SOURCE_MODES:
        raise ValueError(f"source_mode must be one of {', '.join(SOURCE_MODES)}")

    if prepared_source is not None:
        prepared_mode = prepared_source.get("source_mode")
        if prepared_mode != source_mode:
            raise RuntimeError(
                "prepared source mode does not match the requested tier source mode"
            )
        raw_steps = prepared_source.get("source_steps")
        if not isinstance(raw_steps, list):
            raise RuntimeError("prepared source bundle has no source step receipt")
        source_steps = [step for step in raw_steps if isinstance(step, dict)]
        by_source = {
            str(step.get("source")): step
            for step in source_steps
            if isinstance(step.get("source"), str)
        }
        oe_step = by_source.get(
            "oe_annual", _skipped_step("oe_annual", "prepared_source_bundle")
        )
        oe_api_step = by_source.get(
            "oe_api", _skipped_step("oe_api", "prepared_source_bundle")
        )
        atom_step = by_source.get(
            "champion_atomization",
            _skipped_step("champion_atomization", "prepared_source_bundle"),
        )
        live_source_step = by_source.get(
            "oe_live_source", _skipped_step("oe_live_source", "prepared_source_bundle")
        )
        rating_step = by_source.get(
            "ratings", _skipped_step("ratings", "prepared_source_bundle")
        )
        grid_step = by_source.get(
            "grid", _skipped_step("grid", "prepared_source_bundle")
        )
        observed_as_of = prepared_source.get("source_observed_through")
        if not isinstance(observed_as_of, str) or not observed_as_of:
            observed_as_of = _api_source_latest(root)
        candidate_expected_live_as_of = observed_as_of or expected_live_as_of
    else:
        oe_step = (
            _skipped_step("oe_annual", "committed_public_pack_baseline")
            if skip_annual_oe
            else _run_step(
                root,
                [
                    "lol_kills.refresh_warehouse",
                    "--oe-years",
                    "2025",
                    "2026",
                    "--refresh-oe",
                    "--skip-lp",
                    "--skip-grid",
                ],
                source="oe_annual",
            )
        )
        oe_api_step = _run_step(
            root,
            [
                "lol_kills.etl.oe_api_ingest",
                "--root",
                str(root),
                "--start",
                LIVE_WINDOW_START,
                "--end",
                expected_live_as_of,
                "--lookback-days",
                "120",
            ],
            source="oe_api",
        )
        atom_step = (
            _verify_prebuilt_atom_bridge(root)
            if skip_atom_bridge
            else _run_step(
                root,
                ["lol_kills.v2.champions.atoms.bridge_v1"],
                source="champion_atomization",
            )
        )
        observed_as_of = _api_source_latest(root) if oe_api_step["completed"] else None
        candidate_expected_live_as_of = observed_as_of or expected_live_as_of
        live_source_step = (
            _run_step(
                root,
                ["lol_kills.etl.oe_live_source", "--root", str(root)],
                source="oe_live_source",
            )
            if oe_api_step["completed"]
            else _skipped_step("oe_live_source", "oe_api_incomplete")
        )
        rating_step = (
            _run_step(
                root,
                [
                    "lol_kills.v2.tierlists.rating_refresh",
                    "--root",
                    str(root),
                    "--as-of",
                    candidate_expected_live_as_of,
                ],
                source="ratings",
            )
            if live_source_step["completed"] and _api_player_detail_complete(root)
            else _skipped_step("ratings", "oe_player_detail_incomplete")
        )
        if source_mode == "oe_plus_grid":
            grid_step = _run_step(
                root,
                [
                    "lol_kills.refresh_warehouse",
                    "--skip-oe",
                    "--skip-lp",
                    "--download-grid",
                    "--grid-days",
                    str(grid_days),
                    "--grid-limit",
                    str(grid_limit),
                ],
                source="grid",
            )
        else:
            grid_step = _skipped_step("grid", "source_mode_oe_only")

    if prepared_source is None:
        source_steps = [oe_step, oe_api_step, atom_step, live_source_step, rating_step, grid_step]
    required_source_steps = [oe_api_step, atom_step, live_source_step]
    if not all(step["completed"] for step in required_source_steps):
        raise RuntimeError(
            "tier refresh source preparation failed: "
            + _source_step_failure(source_steps)
        )

    previous = None
    movement_baseline: dict[str, Any] = {
        "kind": "previous_approved_artifact",
        "as_of": None,
        "artifact_sha256": None,
    }
    if previous_path is not None:
        previous_file = previous_path if previous_path.is_absolute() else root / previous_path
        previous = json.loads(previous_file.read_text(encoding="utf-8"))
        movement_baseline["as_of"] = previous.get("as_of")
        movement_baseline["artifact_kind"] = previous.get("artifact_kind")
        movement_baseline["artifact_sha256"] = previous.get("artifact_sha256")
    else:
        baseline_as_of = _previous_week_start(candidate_expected_live_as_of)
        previous = build_candidate(
            root,
            as_of=baseline_as_of,
            previous=None,
            source_mode=source_mode,
        )
        movement_baseline = {
            "kind": "previous_sunday_utc",
            "as_of": baseline_as_of.isoformat().replace("+00:00", "Z"),
            "artifact_sha256": previous.get("artifact_sha256"),
        }
    candidate = build_candidate(
        root,
        expected_live_as_of=pd.Timestamp(candidate_expected_live_as_of),
        previous=previous,
        source_mode=source_mode,
    )
    output = output_path if output_path.is_absolute() else root / output_path
    raw_sha256 = write_candidate(output, candidate)

    promotion_steps: list[dict[str, Any]] = []
    promotion_status = "not_requested"
    if promote and output.resolve() != (root / DEFAULT_OUTPUT).resolve():
        promotion_steps.append(
            _skipped_step(
                "production_promotion",
                "promotion_requires_default_candidate_locator",
            )
        )
        promotion_status = "blocked_promotion_locator"
    elif promote and not (
        candidate["source_complete_through_expected_live_as_of"]
        and atom_step["completed"]
        and rating_step["completed"]
        and (source_mode == "oe_only" or grid_step["completed"])
    ):
        promotion_steps.append(_skipped_step("production_promotion", "source_or_rating_incomplete"))
        promotion_status = "blocked_source_incomplete"
    elif promote:
        forward_step = _run_step(
            root,
            [
                "lol_kills.v2.tierlists.forward_evaluation",
                "--root",
                str(root),
                "--output",
                "data/lol/v2/tierlists/prospective-evaluation-v1.json",
            ],
            source="forward_evaluation",
        )
        promotion_steps.append(forward_step)
        if forward_step["completed"]:
            authority_step = _run_step(
                root,
                [
                    "lol_kills.v2.tierlists.independent_authority",
                    "--root",
                    str(root),
                    "--output",
                    "data/lol/v2/tierlists/independent-l2-authority-v1.json",
                ],
                source="independent_authority",
            )
            promotion_steps.append(authority_step)
        else:
            authority_step = _skipped_step("independent_authority", "forward_evaluation_incomplete")
            promotion_steps.append(authority_step)
        if authority_step["completed"]:
            bundle_step = _run_step(
                root,
                [
                    "lol_kills.v2.tierlists.production_bundle",
                    "--root",
                    str(root),
                ],
                source="production_bundle",
            )
            promotion_steps.append(bundle_step)
        else:
            bundle_step = _skipped_step("production_bundle", "independent_authority_incomplete")
            promotion_steps.append(bundle_step)
        if bundle_step["completed"]:
            try:
                publication = publish_production_bundle(root)
                publication_step = {
                    "source": "blob_publication",
                    "command": [],
                    "returncode": 0,
                    "completed": True,
                    "publication": publication,
                }
            except Exception as error:
                publication_step = _skipped_step(
                    "blob_publication",
                    f"{type(error).__name__}: {error}",
                )
                publication_step["returncode"] = 2
            promotion_steps.append(publication_step)
            promotion_status = "promoted" if publication_step["completed"] else "blocked_publication"
        else:
            promotion_status = "blocked_promotion"

    source_ready = (
        candidate["source_complete_through_expected_live_as_of"]
        and atom_step["completed"]
        and rating_step["completed"]
        and (source_mode == "oe_only" or grid_step["completed"])
    )
    if promotion_status == "promoted":
        receipt_status = "production_promoted"
    elif promote and promotion_status == "blocked_publication":
        receipt_status = "blocked_publication"
    elif promote and promotion_status in {"blocked_promotion", "blocked_promotion_locator"}:
        receipt_status = "blocked_promotion"
    elif source_ready:
        receipt_status = "ready_for_authority_review"
    else:
        receipt_status = "blocked_source_incomplete"

    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "source_mode": source_mode,
        "status": receipt_status,
        "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "window_start": HISTORY_START,
        "live_window_start": LIVE_WINDOW_START,
        "expected_live_as_of": expected_live_as_of,
        "candidate_expected_live_as_of": candidate_expected_live_as_of,
        "source_observed_through": observed_as_of,
        "movement_baseline": movement_baseline,
        "source_steps": source_steps,
        "promotion_status": promotion_status,
        "promotion_steps": promotion_steps,
        "candidate": {
            "locator": str(output.relative_to(root)) if output.is_relative_to(root) else str(output),
            "raw_sha256": raw_sha256,
            "artifact_sha256": candidate["artifact_sha256"],
            "as_of": candidate["as_of"],
            "maps_replayed": candidate["source"]["maps_replayed"],
            "maps_in_live_window": candidate["source"]["maps_in_live_window"],
            "source_complete_through_expected_live_as_of": candidate[
                "source_complete_through_expected_live_as_of"
            ],
            "source_mode": candidate["source_mode"],
            "rating_refresh_completed": rating_step["completed"],
        },
        "authority": {
            "source_freshness": True,
            "model_validation": False,
            "publication": promotion_status == "promoted",
            "rank_eligibility": promotion_status == "promoted",
            "recommendation": False,
            "betting": False,
        },
        "claim_ceiling": (
            "This receipt records a source-bound descriptive production bundle. It does not authorize outcome-calibrated probability, causal, recommendation, or betting claims."
            if promotion_status == "promoted"
            else "This receipt records source refresh and a development replay only."
        ),
    }
    receipt["receipt_canonical_sha256"] = _canonical_sha256(receipt)
    receipt_destination = receipt_path if receipt_path.is_absolute() else root / receipt_path
    if receipt_destination.suffix.lower() != ".json":
        stamp = receipt["retrieved_at"].replace("-", "").replace(":", "").replace(".", "")
        receipt_destination = receipt_destination / (
            f"tierlist-live-refresh-{stamp}-{raw_sha256[:16]}.json"
        )
    receipt_destination.parent.mkdir(parents=True, exist_ok=True)
    receipt_destination.write_text(
        json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--expected-live-as-of", required=True)
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--grid-days", type=int, default=21)
    parser.add_argument("--grid-limit", type=int, default=200)
    parser.add_argument(
        "--promote",
        action="store_true",
        help="run the descriptive evaluation, independent authority, and production bundle after refresh",
    )
    parser.add_argument(
        "--source-mode",
        choices=SOURCE_MODES,
        default=DEFAULT_SOURCE_MODE,
        help="oe_only uses the daily OE export; oe_plus_grid adds the same-day bridge",
    )
    parser.add_argument(
        "--skip-annual-oe",
        action="store_true",
        help="use the restored committed OE pack as the historical baseline and only fetch the API freshness bridge",
    )
    parser.add_argument(
        "--skip-atom-bridge",
        action="store_true",
        help="verify the committed atom bridge instead of rebuilding it from a private LCC checkout",
    )
    args = parser.parse_args()
    receipt = refresh_candidate(
        args.root,
        expected_live_as_of=args.expected_live_as_of,
        previous_path=args.previous,
        output_path=args.out,
        receipt_path=args.receipt,
        grid_days=args.grid_days,
        grid_limit=args.grid_limit,
        source_mode=args.source_mode,
        promote=args.promote,
        skip_annual_oe=args.skip_annual_oe,
        skip_atom_bridge=args.skip_atom_bridge,
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {"ready_for_authority_review", "production_promoted"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
