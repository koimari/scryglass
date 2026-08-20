"""Refresh the public tier-list candidate from Oracle's Elixir.

This command is for a durable worker. It writes a development candidate and a
hash-bound receipt. With ``--promote``, it runs the descriptive evaluation,
independent authority check, and production bundle after the source gates pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.v2.champions.atoms.consume import AtomBridge
from lol_kills.v2.patch_identity import CURRENT_PUBLIC_PATCH, PatchIdentityError, public_patch

from .champion_elo import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE_MODE,
    build_candidate,
    write_candidate,
)
from .accepted_census import load_census
from .rating_refresh import OUTPUT as RATING_REFRESH_OUTPUT

HISTORY_START = "2025-01-01T00:00:00Z"
LIVE_WINDOW_START = "2026-07-18T00:00:00Z"
RECEIPT_SCHEMA = "scryglass:tierlist-live-refresh:v1"
DEFAULT_RECEIPT = Path("data/lol/v2/tierlists/refresh-receipts")
# Successful ratings runs measured 682-760s in production 2026-08-11 through
# 2026-08-19 against the prior 900s (15 min) ceiling. Scheduled runs on
# 2026-08-20 were killed at exactly 900.1s as the season dataset grew to
# 7,698 games. 45 min keeps the step well inside the 6-hour cycle budget
# while leaving headroom for continued dataset growth.
DEFAULT_STEP_TIMEOUT_SECONDS = 45 * 60
LEGACY_BLOB_PUBLICATION_DISABLED = (
    "Vercel Blob tier-list publication is retired; use Supabase public release publication"
)


def _step_timeout_seconds() -> float:
    raw = os.environ.get("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS")
    if raw is None or not raw.strip():
        return float(DEFAULT_STEP_TIMEOUT_SECONDS)
    value = float(raw)  # let ValueError propagate: a mis-set override must not silently fall back
    if not math.isfinite(value) or value <= 0:
        raise ValueError("SCRYGLASS_TIER_STEP_TIMEOUT_SECONDS must be positive")
    return value


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


def _regional_refresh_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    """Describe the regional views carried by one candidate receipt.

    Regional views reuse the patch-wide fit. They only change the appearance
    filter and the evidence count for each named league context. The receipt
    records the coverage so a refresh cannot look complete while silently
    dropping the regional layer.
    """

    cells = candidate.get("cells")
    if not isinstance(cells, list):
        return {
            "status": "unavailable",
            "view_count": 0,
            "cell_count": 0,
            "cells_with_views": 0,
            "contexts": [],
        }
    view_count = 0
    cells_with_views = 0
    contexts: set[str] = set()
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        views = cell.get("regional_views")
        if not isinstance(views, list) or not views:
            continue
        cells_with_views += 1
        for view in views:
            if not isinstance(view, dict):
                continue
            view_id = view.get("id")
            if isinstance(view_id, str) and view_id.strip():
                contexts.add(view_id.strip())
                view_count += 1
    return {
        "status": "available" if view_count else "unavailable",
        "view_count": view_count,
        "cell_count": len(cells),
        "cells_with_views": cells_with_views,
        "contexts": sorted(contexts),
        "basis": "patch_wide_model_with_regional_appearance_filter",
    }


def _patch_refresh_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    """Record accepted and pending public patch coverage in the receipt."""

    accepted: set[str] = set()
    cells = candidate.get("cells")
    if isinstance(cells, list):
        for cell in cells:
            if not isinstance(cell, dict) or not isinstance(cell.get("patches"), list):
                continue
            for value in cell["patches"]:
                try:
                    accepted.add(public_patch(str(value)))
                except PatchIdentityError:
                    continue
    ordered = sorted(
        accepted,
        key=lambda value: tuple(int(part) for part in value.split(".")),
    )
    return {
        "current_public_patch": CURRENT_PUBLIC_PATCH,
        "accepted_public_patches": ordered,
        "latest_accepted_public_patch": ordered[-1] if ordered else None,
        "status": "current" if CURRENT_PUBLIC_PATCH in accepted else "awaiting_source_rows",
    }


def restore_production_bundle(publication: dict[str, Any]) -> dict[str, Any]:
    """Restore stable tier-list pointers captured by a successful publication."""

    raise PublicationError(LEGACY_BLOB_PUBLICATION_DISABLED)


def publish_production_bundle(root: Path | str = Path(".")) -> dict[str, Any]:
    """Publish immutable tier-list files, then replace the stable pointer."""

    raise PublicationError(LEGACY_BLOB_PUBLICATION_DISABLED)


def _step_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_step(
    root: Path,
    args: list[str],
    *,
    source: str,
    step_timeout_seconds: float | None = None,
) -> dict[str, Any]:
    if step_timeout_seconds is None:
        step_timeout_seconds = _step_timeout_seconds()
    if step_timeout_seconds <= 0:
        raise ValueError("step timeout must be positive")
    started = time.monotonic()
    print(f"[tier-refresh] step={source} start", flush=True)
    environment = os.environ.copy()
    inherited_paths = [str(Path(item)) for item in sys.path if item and Path(item).is_dir()]
    configured_paths = [item for item in environment.get("PYTHONPATH", "").split(os.pathsep) if item]
    environment["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys([*inherited_paths, *configured_paths])
    )
    try:
        result = subprocess.run(
            [sys.executable, "-m", *args],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=step_timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        stdout = _step_text(error.stdout)
        stderr = _step_text(error.stderr)
        print(
            f"[tier-refresh] step={source} timed_out seconds={time.monotonic() - started:.1f}",
            flush=True,
        )
        return {
            "source": source,
            "command": args,
            "returncode": 124,
            "completed": False,
            "timed_out": True,
            "timeout_seconds": step_timeout_seconds,
            "stdout_bytes": len(stdout.encode("utf-8")),
            "stderr_bytes": len(stderr.encode("utf-8")),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "reason": f"step exceeded {step_timeout_seconds:g} seconds",
        }
    print(
        f"[tier-refresh] step={source} done returncode={result.returncode} seconds={time.monotonic() - started:.1f}",
        flush=True,
    )
    return {
        "source": source,
        "command": args,
        "returncode": result.returncode,
        "completed": result.returncode == 0,
        "timeout_seconds": step_timeout_seconds,
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

    from lol_kills.v2.champions.atoms.consume import DEFAULT_ARTIFACT_PATH

    path = root / "data" / "lol" / "v2" / "champions" / DEFAULT_ARTIFACT_PATH.name
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


def _oe_source_latest(root: Path) -> str | None:
    path = root / "data/lol/warehouse/parquet/oe_live/meta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = payload.get("source_latest")
    return value if isinstance(value, str) else None


def _oe_rating_source_complete(root: Path) -> bool:
    path = root / "data/lol/warehouse/parquet/oe_live/meta.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    source_games = payload.get("source_game_count")
    identity_complete_games = payload.get("identity_complete_game_count")
    if isinstance(source_games, int) and isinstance(identity_complete_games, int):
        return source_games > 0 and identity_complete_games == source_games
    return payload.get("player_statistics_complete") is True


def _verify_prebuilt_oe_live(root: Path, expected_live_as_of: str) -> dict[str, Any]:
    """Keep the accepted release source intact during the tier refresh."""

    observed = _oe_source_latest(root)
    try:
        matches = (
            observed is not None
            and pd.Timestamp(observed).tz_convert("UTC")
            == pd.Timestamp(expected_live_as_of).tz_convert("UTC")
        )
    except (TypeError, ValueError):
        matches = False
    complete = _oe_rating_source_complete(root)
    return {
        "source": "oe_live_source",
        "command": [],
        "returncode": 0 if matches and complete else 2,
        "completed": bool(matches and complete),
        "skipped": True,
        "reason": (
            "accepted_oe_live_verified"
            if matches and complete
            else "accepted_oe_live_mismatch"
        ),
        "source_observed_through": observed,
    }


def _bind_rating_step_to_census(
    root: Path,
    step: dict[str, Any],
    accepted: dict[str, Any] | None,
) -> dict[str, Any]:
    if accepted is None or step.get("completed") is not True:
        return step
    path = root / RATING_REFRESH_OUTPUT
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = payload["source"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError):
        return {
            **step,
            "completed": False,
            "returncode": 2,
            "reason": "rating refresh census receipt is unavailable",
        }
    if (
        source.get("source_game_count") != accepted["game_count"]
        or source.get("source_identity_sha256") != accepted["source_identity_sha256"]
    ):
        return {
            **step,
            "completed": False,
            "returncode": 2,
            "reason": "rating refresh used a different accepted game census",
        }
    return {
        **step,
        "accepted_source_game_count": accepted["game_count"],
        "accepted_source_identity_sha256": accepted["source_identity_sha256"],
    }


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
    source_mode: str = DEFAULT_SOURCE_MODE,
    promote: bool = False,
    skip_annual_oe: bool = False,
    skip_live_source: bool = False,
    skip_atom_bridge: bool = False,
    prepared_source: dict[str, Any] | None = None,
    accepted_census_path: Path | None = None,
    step_timeout_seconds: float | None = None,
    publish: bool = True,
) -> dict[str, Any]:
    if source_mode != "oe_only":
        raise ValueError("public tier refresh source_mode must be oe_only")
    if step_timeout_seconds is None:
        step_timeout_seconds = _step_timeout_seconds()
    if step_timeout_seconds <= 0:
        raise ValueError("step timeout must be positive")
    if accepted_census_path is not None:
        accepted_census_path = (
            accepted_census_path
            if accepted_census_path.is_absolute()
            else root / accepted_census_path
        ).resolve()
        try:
            accepted_census_path.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError("accepted census path escaped the runtime root") from error
    accepted = load_census(accepted_census_path) if accepted_census_path is not None else None
    allowed_game_ids = accepted["game_ids"] if accepted is not None else None

    def run_step(args: list[str], *, source: str) -> dict[str, Any]:
        return _run_step(
            root,
            args,
            source=source,
            step_timeout_seconds=step_timeout_seconds,
        )

    previous_live_as_of: str | None = None
    if previous_path is not None:
        try:
            previous_file = previous_path if previous_path.is_absolute() else root / previous_path
            previous_early = json.loads(previous_file.read_text(encoding="utf-8"))
            previous_live_as_of = str(previous_early.get("as_of") or "") or None
        except (OSError, ValueError, TypeError):
            previous_live_as_of = None

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
        grid_step = _skipped_step("grid", "public_refresh_oe_only")
        observed_as_of = prepared_source.get("source_observed_through")
        if not isinstance(observed_as_of, str) or not observed_as_of:
            observed_as_of = _oe_source_latest(root)
        candidate_expected_live_as_of = observed_as_of or expected_live_as_of
    else:
        oe_step = (
            _skipped_step("oe_annual", "cached_oe_source")
            if skip_annual_oe
            else run_step(
                [
                    "lol_kills.refresh_warehouse",
                    "--oe-years",
                    "2025",
                    "2026",
                    "--download-oe",
                    "--skip-lp",
                    "--skip-grid",
                ],
                source="oe_annual",
            )
        )
        atom_step = (
            _verify_prebuilt_atom_bridge(root)
            if skip_atom_bridge
            else run_step(
                ["lol_kills.v2.champions.atoms.bridge_v1"],
                source="champion_atomization",
            )
        )
        live_source_step = (
            _verify_prebuilt_oe_live(root, expected_live_as_of)
            if skip_live_source
            else run_step(
                ["lol_kills.etl.oe_live_source", "--root", str(root)],
                source="oe_live_source",
            )
        )
        observed_as_of = _oe_source_latest(root) if live_source_step["completed"] else None
        candidate_expected_live_as_of = observed_as_of or expected_live_as_of
        rating_step = (
            run_step(
                [
                    "lol_kills.v2.tierlists.rating_refresh",
                    "--root",
                    str(root),
                    "--as-of",
                    candidate_expected_live_as_of,
                    *(
                        ["--previous-as-of", previous_live_as_of]
                        if previous_live_as_of
                        else []
                    ),
                    *(
                        ["--accepted-census", str(accepted_census_path)]
                        if accepted_census_path is not None
                        else []
                    ),
                ],
                source="ratings",
            )
            if live_source_step["completed"] and _oe_rating_source_complete(root)
            else _skipped_step("ratings", "oe_player_identity_incomplete")
        )
        rating_step = _bind_rating_step_to_census(root, rating_step, accepted)
        grid_step = _skipped_step("grid", "public_refresh_oe_only")

    if prepared_source is None:
        source_steps = [oe_step, atom_step, live_source_step, rating_step, grid_step]
    required_source_steps = [atom_step, live_source_step]
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
            allowed_game_ids=allowed_game_ids,
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
        allowed_game_ids=allowed_game_ids,
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
        and source_mode == "oe_only"
    ):
        promotion_steps.append(_skipped_step("production_promotion", "source_or_rating_incomplete"))
        promotion_status = "blocked_source_incomplete"
    elif promote:
        forward_step = run_step(
            [
                "lol_kills.v2.tierlists.forward_evaluation",
                "--root",
                str(root),
                "--output",
                "data/lol/v2/tierlists/prospective-evaluation-v1.json",
                *(
                    ["--accepted-census", str(accepted_census_path)]
                    if accepted_census_path is not None
                    else []
                ),
            ],
            source="forward_evaluation",
        )
        promotion_steps.append(forward_step)
        if forward_step["completed"]:
            authority_step = run_step(
                [
                    "lol_kills.v2.tierlists.independent_authority",
                    "--root",
                    str(root),
                    "--output",
                    "data/lol/v2/tierlists/independent-l2-authority-v1.json",
                    *(
                        ["--accepted-census", str(accepted_census_path)]
                        if accepted_census_path is not None
                        else []
                    ),
                ],
                source="independent_authority",
            )
            promotion_steps.append(authority_step)
        else:
            authority_step = _skipped_step("independent_authority", "forward_evaluation_incomplete")
            promotion_steps.append(authority_step)
        if authority_step["completed"]:
            bundle_step = run_step(
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
        if bundle_step["completed"] and publish:
            try:
                publication = publish_production_bundle(root)
                publication_step = {
                    "source": "publication_coordinator",
                    "command": [],
                    "returncode": 0,
                    "completed": True,
                    "publication": publication,
                }
            except Exception as error:
                publication_step = _skipped_step(
                    "publication_disabled",
                    f"{type(error).__name__}: {error}",
                )
                publication_step["returncode"] = 2
            promotion_steps.append(publication_step)
            promotion_status = "promoted" if publication_step["completed"] else "blocked_publication"
        elif bundle_step["completed"]:
            promotion_steps.append(
                {
                    "source": "deferred_publication",
                    "command": [],
                    "returncode": 0,
                    "completed": True,
                    "reason": "the release coordinator will publish the complete ratings and tier snapshot",
                }
            )
            promotion_status = "built"
        else:
            promotion_status = "blocked_promotion"

    source_ready = (
        candidate["source_complete_through_expected_live_as_of"]
        and atom_step["completed"]
        and rating_step["completed"]
        and source_mode == "oe_only"
    )
    if promotion_status == "promoted":
        receipt_status = "production_promoted"
    elif promotion_status == "built":
        receipt_status = "production_built"
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
            "source_identity_sha256": candidate["source"].get("source_identity_sha256"),
            "maps_in_live_window": candidate["source"]["maps_in_live_window"],
            "source_complete_through_expected_live_as_of": candidate[
                "source_complete_through_expected_live_as_of"
            ],
            "source_mode": candidate["source_mode"],
            "rating_refresh_completed": rating_step["completed"],
        },
        "accepted_census": (
            {
                "game_count": accepted["game_count"],
                "source_identity_sha256": accepted["source_identity_sha256"],
            }
            if accepted is not None
            else None
        ),
        "regional_refresh": _regional_refresh_summary(candidate),
        "patch_refresh": _patch_refresh_summary(candidate),
        "authority": {
            "source_freshness": True,
            "model_validation": False,
            "publication": promotion_status == "promoted",
            "rank_eligibility": promotion_status in {"promoted", "built"},
            "recommendation": False,
            "betting": False,
        },
        "claim_ceiling": (
            "This receipt records a source-bound descriptive production bundle. It does not authorize outcome-calibrated probability, causal, recommendation, or betting claims."
            if promotion_status in {"promoted", "built"}
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
    parser.add_argument(
        "--promote",
        action="store_true",
        help="run the descriptive evaluation, independent authority, and production bundle after refresh",
    )
    parser.add_argument(
        "--defer-publication",
        action="store_true",
        help="build the approved production bundle and let the release coordinator publish it",
    )
    parser.add_argument(
        "--source-mode",
        choices=("oe_only",),
        default=DEFAULT_SOURCE_MODE,
        help="public tier lists use the OE annual CSV source",
    )
    parser.add_argument(
        "--skip-annual-oe",
        action="store_true",
        help="use the already refreshed OE annual cache",
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
        source_mode=args.source_mode,
        promote=args.promote,
        skip_annual_oe=args.skip_annual_oe,
        skip_atom_bridge=args.skip_atom_bridge,
        publish=not args.defer_publication,
    )
    print(json.dumps(receipt, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if receipt["status"] in {
        "ready_for_authority_review",
        "production_built",
        "production_promoted",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
