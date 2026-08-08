"""Validate and promote the descriptive champion ladder.

The bundle has a separate authority boundary from the private terminal Draft
Score model.  It authorizes source-bound descriptive ranks and movement.  It
keeps outcome-calibrated probability, causal, recommendation, and betting
claims closed.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from .forward_evaluation import (
    CANDIDATE_LOCATOR,
    SOURCE_LOCATOR,
    SOURCE_META_LOCATOR,
)


EVALUATION_LOCATOR = Path("data/lol/v2/tierlists/prospective-evaluation-v1.json")
AUTHORITY_LOCATOR = Path("data/lol/v2/tierlists/independent-l2-authority-v1.json")
MANIFEST_LOCATOR = Path("data/lol/v2/tierlists/production-manifest-v1.json")
PRODUCTION_ROOT = Path("data/lol/v2/tierlists/production")
PUBLIC_PRODUCTION_ROOT = Path("apps/scryglass/public/v2/tierlists/production")
SCHEMA_VERSION = "scryglass:tierlist-production-bundle:v1"
INDEX_SCHEMA_VERSION = "scryglass:tierlist-production-index:v1"
CELL_SCHEMA_VERSION = "scryglass:tierlist-production-cell:v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ROLES = ("top", "jungle", "mid", "bot", "support")
TIER_BUCKETS = ("Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D")


class ProductionBundleError(ValueError):
    """Raised when a production bundle cannot be proven safe."""


def _canonical(value: object) -> bytes:
    def normalize(item: object) -> object:
        if isinstance(item, Mapping):
            return {key: normalize(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ProductionBundleError("canonical production payload contains a non-finite number")
            if item == 0.0:
                return 0
            if item.is_integer():
                return int(item)
        return item

    return json.dumps(normalize(value), ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _python_canonical(value: object) -> bytes:
    """Match the canonical bytes used by the Python candidate artifacts."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return _sha256_bytes(_canonical(unsigned))


def _candidate_canonical_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    return _sha256_bytes(_python_canonical(unsigned))


def _read_json(root: Path, locator: Path) -> tuple[bytes, dict[str, Any]]:
    path = root / locator
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProductionBundleError(f"cannot read JSON: {locator}") from exc
    if not isinstance(payload, dict):
        raise ProductionBundleError(f"JSON object required: {locator}")
    return raw, payload


def _require_sha(value: object, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ProductionBundleError(f"{field} is not a SHA-256")
    return value


def _patch_key(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)", value)
    if match is None:
        return (-1, -1)
    return int(match.group(1)), int(match.group(2))


def _cell_patch(cell: Mapping[str, Any]) -> str:
    patches = [str(value) for value in cell.get("patches", []) if isinstance(value, str)]
    valid = [value for value in patches if _patch_key(value) != (-1, -1)]
    return max(valid, key=_patch_key) if valid else "rolling"


def _cell_slug(scope_id: str, patch_id: str, role: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", f"{scope_id}-{patch_id}-{role}".casefold()).strip("-")
    return f"tierlist-{value}-production-v1.json"


def _image_url(champion_id: str) -> str | None:
    match = re.fullmatch(r"riot:champion:(\d+)", champion_id)
    if match is None:
        return None
    return f"https://cdn.communitydragon.org/latest/champion/{match.group(1)}/square"


def _claim_ceiling() -> dict[str, Any]:
    return {
        "descriptive_pre_map_association": True,
        "rank_eligibility": True,
        "publication": True,
        "production": True,
        "outcome_calibrated_probability": False,
        "causal_draft_effect": False,
        "recommendation": False,
        "betting": False,
    }


def _candidate_and_evaluation(root: Path) -> tuple[bytes, dict[str, Any], bytes, dict[str, Any]]:
    candidate_raw, candidate = _read_json(root, CANDIDATE_LOCATOR)
    evaluation_raw, evaluation = _read_json(root, EVALUATION_LOCATOR)
    if candidate.get("artifact_sha256") != _candidate_canonical_sha256(candidate):
        raise ProductionBundleError("candidate canonical digest is invalid")
    if evaluation.get("artifact_sha256") != _candidate_canonical_sha256(evaluation):
        raise ProductionBundleError("forward evaluation canonical digest is invalid")
    if candidate.get("status") != "development_only" or candidate.get("production_eligible") is not False:
        raise ProductionBundleError("promotion input must remain a development candidate")
    if evaluation.get("status") != "complete" or evaluation.get("decision") != "descriptive_pass":
        raise ProductionBundleError("forward evaluation is not a complete descriptive pass")
    if evaluation.get("production_eligible") is not True:
        raise ProductionBundleError("forward evaluation does not authorize descriptive publication")
    if evaluation.get("candidate", {}).get("artifact_sha256") != candidate.get("artifact_sha256"):
        raise ProductionBundleError("forward evaluation is bound to another candidate")
    for field in (
        "descriptive_replay_complete",
        "descriptive_replay_time_safe",
        "source_identity_complete",
        "all_roles_covered",
        "movement_fields_complete",
        "counterability_policy_validated",
        "counterability_weight_manifested",
    ):
        if evaluation.get(field) is not True:
            raise ProductionBundleError(f"forward evaluation gate failed: {field}")
    return candidate_raw, candidate, evaluation_raw, evaluation


def _validate_candidate_structure(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if candidate.get("unresolved_champion_identities") != []:
        raise ProductionBundleError("candidate contains unresolved champion identities")
    cells = candidate.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ProductionBundleError("candidate contains no cells")
    scope_roles: dict[tuple[str, str], set[str]] = {}
    row_count = 0
    for cell in cells:
        if not isinstance(cell, Mapping):
            raise ProductionBundleError("candidate cell is malformed")
        role = cell.get("role")
        scope_id = cell.get("scope_id")
        if role not in ROLES or not isinstance(scope_id, str) or not scope_id:
            raise ProductionBundleError("candidate scope or role is invalid")
        scope_roles.setdefault((scope_id, str(cell.get("patches"))), set()).add(role)
        rows = cell.get("rows")
        if not isinstance(rows, list) or not rows:
            raise ProductionBundleError(f"candidate cell has no rows: {scope_id} {role}")
        ranks: list[int] = []
        for row in rows:
            if not isinstance(row, Mapping):
                raise ProductionBundleError("candidate champion row is malformed")
            champion_id = row.get("champion_id")
            if not isinstance(champion_id, str) or not re.fullmatch(r"riot:champion:\d+", champion_id):
                raise ProductionBundleError("candidate champion identity is invalid")
            rank = row.get("rank")
            if not isinstance(rank, int) or rank < 1:
                raise ProductionBundleError("candidate rank is invalid")
            ranks.append(rank)
            previous_rank = row.get("previous_rank")
            rank_delta = row.get("rank_delta")
            movement = row.get("movement")
            if previous_rank is None:
                if rank_delta is not None or movement != "new":
                    raise ProductionBundleError("candidate new-row movement is inconsistent")
            else:
                if not isinstance(previous_rank, int) or not isinstance(rank_delta, int):
                    raise ProductionBundleError("candidate previous-rank movement is malformed")
                expected_delta = previous_rank - rank
                expected_movement = "up" if expected_delta > 0 else "down" if expected_delta < 0 else "flat"
                if rank_delta != expected_delta or movement != expected_movement:
                    raise ProductionBundleError("candidate rank movement is inconsistent")
            if row.get("counterability_status") not in {"available", "unavailable"}:
                raise ProductionBundleError("candidate counterability status is invalid")
            row_count += 1
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ProductionBundleError(f"candidate ranks are not contiguous: {scope_id} {role}")
    if any(len(roles) != len(ROLES) for roles in scope_roles.values()):
        raise ProductionBundleError("candidate does not provide all five roles for every scope")
    return {"cell_count": len(cells), "row_count": row_count, "scope_count": len(scope_roles)}


def build_production_index(
    root: Path | str = Path("."),
    *,
    authority_raw_sha256: str,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Build an immutable production index and its cell bytes in memory."""

    repo_root = Path(root)
    candidate_raw, candidate, evaluation_raw, evaluation = _candidate_and_evaluation(repo_root)
    structure = _validate_candidate_structure(candidate)
    authority_raw_sha256 = _require_sha(authority_raw_sha256, "authority_raw_sha256")
    cell_bytes: dict[str, bytes] = {}
    metas: list[dict[str, Any]] = []
    for source_cell in candidate["cells"]:
        scope_id = str(source_cell["scope_id"])
        role = str(source_cell["role"])
        patch_id = _cell_patch(source_cell)
        filename = _cell_slug(scope_id, patch_id, role)
        production_rows: list[dict[str, Any]] = []
        for source_row in source_cell["rows"]:
            champion_id = str(source_row["champion_id"])
            production_rows.append(
                {
                    "champion_id": champion_id,
                    "champion_name": source_row["champion"],
                    "tier_value": source_row["tier_value_pp"],
                    "verified_appearance_count": source_row["played_maps"],
                    "counterability_status": source_row["counterability_status"],
                    "counterability": source_row.get("counterability"),
                    "matchup_maps": source_row.get("matchup_maps"),
                    "matchup_opponents": source_row.get("matchup_opponents"),
                    "blind_score_pp": source_row.get("blind_score_pp"),
                    "counter_score": source_row.get("counter_score"),
                    "expected_counter_breadth": source_row.get("expected_counter_breadth"),
                    "countered_opponent_count": source_row.get("countered_opponent_count"),
                    "countered_opponent_share": source_row.get("countered_opponent_share"),
                    "tier_bucket": source_row["tier_bucket"],
                    "rating": source_row["rating"],
                    "rating_delta": source_row.get("rating_delta"),
                    "previous_rank": source_row.get("previous_rank"),
                    "rank": source_row["rank"],
                    "rank_delta": source_row.get("rank_delta"),
                    "movement": source_row.get("movement"),
                    "champion_image_url": _image_url(champion_id),
                    "atom_profile_status": source_row.get("atom_profile_status"),
                    "atom_patch_last_changed": source_row.get("atom_patch_last_changed"),
                    "legal_opponent_distribution_sha256": source_row.get("legal_opponent_distribution_sha256"),
                    "legal_opponents": source_row.get("legal_opponents"),
                    "legal_opponent_coverage": source_row.get("legal_opponent_coverage"),
                    "tier_membership_probability": source_row.get("tier_membership_probability"),
                    "strength_design_rank_full": source_row.get("strength_design_rank_full"),
                    "strength_design_condition_number": source_row.get("strength_design_condition_number"),
                    "strength_component_identified": source_row.get("strength_component_identified"),
                    "maximum_strength_contrast_sd": source_row.get("maximum_strength_contrast_sd"),
                }
            )
        fail_closed_status = "none"
        if all(row["counterability_status"] == "unavailable" for row in production_rows):
            fail_closed_status = "counterability_unavailable"
        payload: dict[str, Any] = {
            "schema_version": CELL_SCHEMA_VERSION,
            "artifact_kind": "tier_list_production",
            "artifact_id": f"scryglass:tierlist:{scope_id.lower()}:{patch_id}:{role}:production-v1",
            "status": "production",
            "fail_closed_status": fail_closed_status,
            "development_only": False,
            "rank_eligibility": True,
            "publication_eligible": True,
            "production_eligible": True,
            "claim_ceiling": _claim_ceiling(),
            "scope": {
                "scope_id": scope_id,
                "scope_kind": source_cell["scope_kind"],
                "scope_label": source_cell.get("scope_label"),
                "region": source_cell.get("region"),
                "league": source_cell.get("league"),
                "event_kind": source_cell.get("event_kind"),
                "competition_tier": source_cell.get("competition_tier"),
            },
            "role": role,
            "patch_id": patch_id,
            "as_of": source_cell["as_of"],
            "legal_opponents": source_cell.get("legal_opponents"),
            "legal_opponent_distribution_sha256": source_cell.get("legal_opponent_distribution_sha256"),
            "strength_design": source_cell.get("strength_design"),
            "patch_ingestion": candidate.get("patch_ingestion"),
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "rows": production_rows,
            "lineage": {
                "candidate_locator": CANDIDATE_LOCATOR.as_posix(),
                "candidate_raw_sha256": _sha256_bytes(candidate_raw),
                "candidate_artifact_sha256": candidate["artifact_sha256"],
                "forward_evaluation_locator": EVALUATION_LOCATOR.as_posix(),
                "forward_evaluation_raw_sha256": _sha256_bytes(evaluation_raw),
                "independent_authority_raw_sha256": authority_raw_sha256,
                "source_locator": SOURCE_LOCATOR.as_posix(),
                "source_meta_locator": SOURCE_META_LOCATOR.as_posix(),
            },
        }
        payload["artifact_sha256"] = _canonical_sha256(payload)
        raw = _canonical(payload) + b"\n"
        relative = f"production/cells/{filename}"
        cell_bytes[relative] = raw
        metas.append(
            {
                "artifact_id": payload["artifact_id"],
                "scope_id": scope_id,
                "scope_kind": source_cell["scope_kind"],
                "region": source_cell.get("region"),
                "league": source_cell.get("league"),
                "event_kind": source_cell.get("event_kind"),
                "competition_tier": source_cell.get("competition_tier"),
                "role": role,
                "patch_id": patch_id,
                "as_of": source_cell["as_of"],
                "locator": f"data/lol/v2/tierlists/{relative}",
                "raw_sha256": _sha256_bytes(raw),
                "row_count": len(production_rows),
                "status": "production",
                "fail_closed_status": fail_closed_status,
            }
        )
    leagues = sorted({str(value) for value in candidate["options"].get("leagues", []) if value})
    patches = sorted({meta["patch_id"] for meta in metas}, key=_patch_key)
    index: dict[str, Any] = {
        "schema_version": INDEX_SCHEMA_VERSION,
        "artifact_kind": "tier_list_index_production",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "as_of": candidate["as_of"],
        "development_only": False,
        "publication_eligible": True,
        "production_eligible": True,
        "source_mode": candidate["source_mode"],
        "patch_ingestion": candidate.get("patch_ingestion"),
        "claim_ceiling": _claim_ceiling(),
        "candidate_artifact_sha256": candidate["artifact_sha256"],
        "forward_evaluation_artifact_sha256": evaluation["artifact_sha256"],
        "independent_authority_raw_sha256": authority_raw_sha256,
        "cells": metas,
        "options": {
            "leagues": leagues,
            "event_kinds": sorted({str(value) for value in candidate["options"].get("event_kinds", []) if value}),
            "competition_tiers": sorted({str(value) for value in candidate["options"].get("competition_tiers", []) if value}),
            "roles": list(ROLES),
            "patches": patches,
            "tier_buckets": list(TIER_BUCKETS),
        },
        "source": {
            "locator": SOURCE_LOCATOR.as_posix(),
            "meta_locator": SOURCE_META_LOCATOR.as_posix(),
            "candidate_locator": CANDIDATE_LOCATOR.as_posix(),
            "forward_evaluation_locator": EVALUATION_LOCATOR.as_posix(),
        },
    }
    index["artifact_sha256"] = _canonical_sha256(index)
    summary = {
        **structure,
        "index_artifact_sha256": index["artifact_sha256"],
        "cell_count": len(metas),
        "production_cell_count": sum(meta["status"] == "production" for meta in metas),
        "candidate_raw_sha256": _sha256_bytes(candidate_raw),
        "evaluation_raw_sha256": _sha256_bytes(evaluation_raw),
    }
    return index, cell_bytes, summary


def _source_tree_sha256(root: Path, extra: Mapping[str, bytes]) -> str:
    paths = {
        CANDIDATE_LOCATOR.as_posix(): _sha256_path(root / CANDIDATE_LOCATOR),
        EVALUATION_LOCATOR.as_posix(): _sha256_path(root / EVALUATION_LOCATOR),
        AUTHORITY_LOCATOR.as_posix(): _sha256_path(root / AUTHORITY_LOCATOR),
        SOURCE_LOCATOR.as_posix(): _sha256_path(root / SOURCE_LOCATOR),
        SOURCE_META_LOCATOR.as_posix(): _sha256_path(root / SOURCE_META_LOCATOR),
        "lol_kills/v2/tierlists/forward_evaluation.py": _sha256_path(root / "lol_kills/v2/tierlists/forward_evaluation.py"),
        "lol_kills/v2/tierlists/production_bundle.py": _sha256_path(root / "lol_kills/v2/tierlists/production_bundle.py"),
    }
    paths.update(extra)
    return _sha256_bytes(_canonical([{"locator": key, "raw_sha256": paths[key]} for key in sorted(paths)]))


def _commit_sha(root: Path) -> str:
    try:
        value = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ProductionBundleError("cannot resolve git commit for production manifest") from exc
    if not COMMIT_RE.fullmatch(value):
        raise ProductionBundleError("git commit is not a full SHA-1")
    return value


def write_production_bundle(
    root: Path | str = Path("."),
    *,
    authority_path: Path = AUTHORITY_LOCATOR,
) -> dict[str, Any]:
    """Write the approved production cells, index, and manifest."""

    repo_root = Path(root)
    authority_raw, authority = _read_json(repo_root, authority_path)
    authority_raw_sha256 = _sha256_bytes(authority_raw)
    if authority.get("status") != "approved" or authority.get("decision") != "pass" or authority.get("production_eligible") is not True:
        raise ProductionBundleError("independent authority record is not approved")
    if authority.get("tier_list_authority") is not True:
        raise ProductionBundleError("authority record does not authorize tier lists")
    index, cell_bytes, summary = build_production_index(repo_root, authority_raw_sha256=authority_raw_sha256)
    production_dir = repo_root / PRODUCTION_ROOT
    public_dir = repo_root / PUBLIC_PRODUCTION_ROOT
    (production_dir / "cells").mkdir(parents=True, exist_ok=True)
    (public_dir / "cells").mkdir(parents=True, exist_ok=True)
    for relative, raw in cell_bytes.items():
        relative_path = Path(relative).relative_to("production")
        canonical_path = production_dir / relative_path
        public_path = public_dir / relative_path
        canonical_path.parent.mkdir(parents=True, exist_ok=True)
        public_path.parent.mkdir(parents=True, exist_ok=True)
        canonical_path.write_bytes(raw)
        public_path.write_bytes(raw)
    index_raw = _canonical(index) + b"\n"
    (production_dir / "index-v1.json").write_bytes(index_raw)
    (public_dir / "index-v1.json").write_bytes(index_raw)
    source_tree_sha256 = _source_tree_sha256(repo_root, {relative: _sha256_bytes(raw) for relative, raw in cell_bytes.items()})
    manifest: dict[str, Any] = {
        "schema_version": "scryglass:tierlist-production-manifest:v1",
        "status": "approved",
        "decision": "promote",
        "production_eligible": True,
        "artifact_kind": "tier_list_production",
        "independent_l2_authority": True,
        "rollback_manifest_recorded": True,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "production_index_locator": (PRODUCTION_ROOT / "index-v1.json").as_posix(),
        "production_index_sha256": _sha256_bytes(index_raw),
        "source_tree_sha256": source_tree_sha256,
        "commit_sha": _commit_sha(repo_root),
        "candidate": {
            "locator": CANDIDATE_LOCATOR.as_posix(),
            "raw_sha256": summary["candidate_raw_sha256"],
            "artifact_sha256": index["candidate_artifact_sha256"],
        },
        "forward_evaluation": {
            "locator": EVALUATION_LOCATOR.as_posix(),
            "raw_sha256": summary["evaluation_raw_sha256"],
            "artifact_sha256": index["forward_evaluation_artifact_sha256"],
        },
        "independent_authority": {
            "locator": authority_path.as_posix(),
            "raw_sha256": authority_raw_sha256,
            "authority_record_id": authority.get("authority_record_id"),
        },
        "rollback": {
            "previous_index_locator": None,
            "previous_index_sha256": None,
            "empty_pointer_allowed": True,
        },
        "claim_ceiling": _claim_ceiling(),
        "coverage": summary,
    }
    manifest["artifact_sha256"] = _canonical_sha256(manifest)
    manifest_raw = _canonical(manifest) + b"\n"
    (repo_root / MANIFEST_LOCATOR).write_bytes(manifest_raw)
    return {
        "manifest": manifest,
        "manifest_raw_sha256": _sha256_bytes(manifest_raw),
        "index_raw_sha256": _sha256_bytes(index_raw),
        "summary": summary,
    }


def _validate_production_cell(payload: Mapping[str, Any], *, meta: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != CELL_SCHEMA_VERSION:
        raise ProductionBundleError("production cell schema is invalid")
    if payload.get("status") != "production" or payload.get("development_only") is not False:
        raise ProductionBundleError("production cell is not production status")
    if payload.get("publication_eligible") is not True or payload.get("production_eligible") is not True:
        raise ProductionBundleError("production cell eligibility is invalid")
    if payload.get("artifact_id") != meta.get("artifact_id"):
        raise ProductionBundleError("production cell artifact identity does not match index")
    if payload.get("role") != meta.get("role") or payload.get("patch_id") != meta.get("patch_id"):
        raise ProductionBundleError("production cell scope does not match index")
    if payload.get("artifact_sha256") != _canonical_sha256(payload):
        raise ProductionBundleError("production cell canonical digest is invalid")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != meta.get("row_count"):
        raise ProductionBundleError("production cell row count is invalid")


def verify_production_index(root: Path | str = Path(".")) -> dict[str, Any]:
    """Verify the exact production index, cells, and public mirror."""

    repo_root = Path(root)
    index_path = repo_root / PRODUCTION_ROOT / "index-v1.json"
    public_index_path = repo_root / PUBLIC_PRODUCTION_ROOT / "index-v1.json"
    raw = index_path.read_bytes()
    public_raw = public_index_path.read_bytes()
    if raw != public_raw:
        raise ProductionBundleError("public production index differs from canonical index")
    index = json.loads(raw.decode("utf-8"))
    if not isinstance(index, dict) or index.get("artifact_sha256") != _canonical_sha256(index):
        raise ProductionBundleError("production index canonical digest is invalid")
    if index.get("artifact_kind") != "tier_list_index_production" or index.get("development_only") is not False:
        raise ProductionBundleError("production index status is invalid")
    if index.get("publication_eligible") is not True or index.get("production_eligible") is not True:
        raise ProductionBundleError("production index eligibility is invalid")
    cells = index.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ProductionBundleError("production index has no cells")
    role_coverage: dict[tuple[str, str], set[str]] = {}
    for meta in cells:
        if not isinstance(meta, Mapping):
            raise ProductionBundleError("production index cell metadata is malformed")
        locator = meta.get("locator")
        if not isinstance(locator, str) or not locator.startswith("data/lol/v2/tierlists/production/cells/"):
            raise ProductionBundleError("production cell locator is outside the production root")
        path = repo_root / Path(locator)
        public_path = repo_root / PUBLIC_PRODUCTION_ROOT / "cells" / path.name
        cell_raw = path.read_bytes()
        if cell_raw != public_path.read_bytes() or _sha256_bytes(cell_raw) != meta.get("raw_sha256"):
            raise ProductionBundleError("production cell or mirror digest mismatch")
        payload = json.loads(cell_raw.decode("utf-8"))
        if not isinstance(payload, Mapping):
            raise ProductionBundleError("production cell is not an object")
        _validate_production_cell(payload, meta=meta)
        key = (str(meta.get("scope_id")), str(meta.get("patch_id")))
        role_coverage.setdefault(key, set()).add(str(meta.get("role")))
    if any(roles != set(ROLES) for roles in role_coverage.values()):
        raise ProductionBundleError("production index does not cover all roles")
    return {
        "locator": (PRODUCTION_ROOT / "index-v1.json").as_posix(),
        "raw_sha256": _sha256_bytes(raw),
        "artifact_sha256": index["artifact_sha256"],
        "cell_count": len(cells),
        "scope_count": len(role_coverage),
        "production_cell_count": sum(meta.get("status") == "production" for meta in cells),
        "as_of": index.get("as_of"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authority", type=Path, default=AUTHORITY_LOCATOR)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_production_index(args.root), indent=2))
    else:
        print(json.dumps(write_production_bundle(args.root, authority_path=args.authority), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
