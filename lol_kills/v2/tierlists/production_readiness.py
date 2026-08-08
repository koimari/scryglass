"""Production-readiness audit for the descriptive L9 tier list bundle.

The audit checks the exact candidate index, cell bytes, public mirror, model
claim ceiling, prospective evaluation, independent authority, and promotion
manifest.  It keeps the private terminal Draft Score authority boundary
separate from the descriptive tier-list authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

from .champion_elo import SOURCE_MODES
from .model import TERMINAL_MODEL_ARTIFACT
from .production_bundle import (
    AUTHORITY_LOCATOR,
    EVALUATION_LOCATOR,
    MANIFEST_LOCATOR,
    PRODUCTION_ROOT,
    ProductionBundleError,
    verify_production_index,
)

SCHEMA_VERSION = "scryglass:tierlist-production-readiness:v1"
INDEX_LOCATOR = PRODUCTION_ROOT / "index-v1.json"
PUBLIC_INDEX_LOCATOR = Path("apps/scryglass/public/v2/tierlists/production/index-v1.json")
PROSPECTIVE_EVALUATION_LOCATOR = EVALUATION_LOCATOR
INDEPENDENT_AUTHORITY_LOCATOR = AUTHORITY_LOCATOR
PROMOTION_MANIFEST_LOCATOR = MANIFEST_LOCATOR
CANDIDATE_ELO_LOCATOR = Path(
    "data/lol/v2/tierlists/champion-elo-candidate-v1.json"
)
CANDIDATE_REGISTRY_LOCATOR = Path(
    "data/lol/v2/models/draft-terminal/draft-terminal-candidate-registry-v3.json"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")

_RECORD_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "prospective_evaluation": {
        "schema_version": "scryglass:tierlist-forward-evaluation:v1",
        "status": "complete",
        "decision": "descriptive_pass",
        "production_eligible": True,
        "prospective": True,
        "synthetic_only": False,
        "future_observed_outcomes": True,
        "descriptive_replay_complete": True,
        "descriptive_replay_time_safe": True,
        "source_identity_complete": True,
        "all_roles_covered": True,
        "movement_fields_complete": True,
        "counterability_policy_validated": True,
        "counterability_weight_manifested": True,
        "predictive_authority": False,
        "outcome_calibrated_probability": False,
        "roster_strength_time_safe": True,
        "current_patch_verified": True,
    },
    "independent_authority": {
        "schema_version": "scryglass:tierlist-independent-l2-authority:v1",
        "status": "approved",
        "decision": "pass",
        "production_eligible": True,
        "independent_l2_authority": True,
        "tier_list_authority": True,
        "sealed_outer_temporal_holdout_decision": "passed",
    },
    "promotion_manifest": {
        "schema_version": "scryglass:tierlist-production-manifest:v1",
        "status": "approved",
        "decision": "promote",
        "production_eligible": True,
        "artifact_kind": "tier_list_production",
        "independent_l2_authority": True,
        "rollback_manifest_recorded": True,
    },
}


class TierListProductionReadinessError(ValueError):
    """Raised when the readiness audit input is malformed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read_json(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TierListProductionReadinessError(f"cannot read JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise TierListProductionReadinessError(f"JSON object required: {path}")
    return raw, payload


def _canonical_index_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    return _sha256(raw)


def _check_index(root: Path) -> dict[str, Any]:
    try:
        verified = verify_production_index(root)
    except (OSError, ValueError, ProductionBundleError) as exc:
        raise TierListProductionReadinessError(str(exc)) from exc
    raw, index = _read_json(root / INDEX_LOCATOR)
    cells = index.get("cells")
    if not isinstance(cells, list) or not cells:
        raise TierListProductionReadinessError("production tier-list index has no cells")
    status_counts: dict[str, int] = {}
    cell_checks: list[dict[str, Any]] = []
    for cell in cells:
        if not isinstance(cell, dict):
            raise TierListProductionReadinessError("production tier-list index contains a malformed cell")
        status = cell.get("status")
        status_counts[status] = status_counts.get(status, 0) + 1
        cell_checks.append(
            {
                "artifact_id": cell.get("artifact_id"),
                "status": status,
                "development_only": False if status == "production" else None,
                "publication_eligible": True if status == "production" else False,
                "rank_eligibility": True if status == "production" else False,
                "artifact_sha256": cell.get("raw_sha256"),
            }
        )
    return {
        "path": INDEX_LOCATOR.as_posix(),
        "artifact_sha256": index.get("artifact_sha256"),
        "raw_sha256": _sha256(raw),
        "generated_at": index.get("generated_at"),
        "cell_count": len(cells),
        "status_counts": status_counts,
        "cells": cell_checks,
        "verified": verified,
    }


def _check_champion_elo_candidate(root: Path) -> tuple[dict[str, Any], list[str]]:
    """Check the dated all-role replay without granting publication authority."""

    path = root / CANDIDATE_ELO_LOCATOR
    if not path.is_file():
        return {
            "path": CANDIDATE_ELO_LOCATOR.as_posix(),
            "present": False,
        }, ["champion_elo_candidate_missing"]

    try:
        raw, payload = _read_json(path)
    except TierListProductionReadinessError as exc:
        return {
            "path": CANDIDATE_ELO_LOCATOR.as_posix(),
            "present": True,
            "valid": False,
            "error": str(exc),
        }, ["champion_elo_candidate_unreadable"]

    blockers: list[str] = []
    submitted_artifact_sha = payload.get("artifact_sha256")
    if (
        not isinstance(submitted_artifact_sha, str)
        or not _SHA256_RE.fullmatch(submitted_artifact_sha)
        or _canonical_index_sha256(payload) != submitted_artifact_sha
    ):
        blockers.append("champion_elo_candidate_digest_invalid")
    if payload.get("schema_version") not in {
        "scryglass:champion-role-elo-candidate:v1",
        "scryglass:champion-role-elo-candidate:v2",
    }:
        blockers.append("champion_elo_candidate_schema_invalid")
    if payload.get("status") != "development_only" or payload.get("development_only") is not True:
        blockers.append("champion_elo_candidate_status_invalid")
    if payload.get("publication_eligible") is not False or payload.get("production_eligible") is not False:
        blockers.append("champion_elo_candidate_claim_ceiling_invalid")
    if payload.get("source_mode") not in SOURCE_MODES:
        blockers.append("champion_elo_source_mode_invalid")
    if payload.get("history_start") != "2025-01-01T00:00:00Z":
        blockers.append("champion_elo_history_window_invalid")
    if payload.get("live_window_start") != "2026-07-18T00:00:00Z":
        blockers.append("champion_elo_live_window_invalid")
    if payload.get("source_complete_through_expected_live_as_of") is not True:
        blockers.append("champion_elo_source_incomplete")
    joint_model = payload.get("joint_model")
    if not isinstance(joint_model, Mapping):
        blockers.append("champion_elo_joint_model_missing")
    else:
        if joint_model.get("posterior_draws_verified", 0) < 2000:
            blockers.append("champion_elo_joint_posterior_draws_incomplete")
        if joint_model.get("schema_id") != "scryglass.tierlists.joint-pooled-model.v1":
            blockers.append("champion_elo_joint_model_schema_invalid")
    if payload.get("patch_ingestion", {}).get("official_to_oe_patch_mapping", {}).get("status") != "audited":
        blockers.append("champion_elo_patch_mapping_not_audited")
    if payload.get("current_patch_verified") is not True:
        blockers.append("champion_elo_current_patch_unverified")
    stability = payload.get("stability")
    if not isinstance(stability, Mapping) or stability.get("status") != "complete":
        blockers.append("champion_elo_loo_stability_missing")
    if payload.get("unresolved_champion_identities") != []:
        blockers.append("champion_elo_identity_unresolved")

    options = payload.get("options")
    if not isinstance(options, Mapping) or set(options.get("roles") or []) != {
        "top",
        "jungle",
        "mid",
        "bot",
        "support",
    }:
        blockers.append("champion_elo_roles_incomplete")

    cells = payload.get("cells")
    cell_rows = 0
    cell_roles: set[str] = set()
    cell_identity_failures = 0
    movement_fields_complete = True
    if not isinstance(cells, list) or not cells:
        blockers.append("champion_elo_cells_missing")
        cells = []
    for cell in cells:
        if not isinstance(cell, Mapping):
            blockers.append("champion_elo_cell_malformed")
            continue
        role = cell.get("role")
        if isinstance(role, str):
            cell_roles.add(role)
        if cell.get("identity_status") != "complete":
            cell_identity_failures += 1
        rows = cell.get("rows")
        if not isinstance(rows, list):
            blockers.append("champion_elo_cell_rows_missing")
            continue
        cell_rows += len(rows)
        for row in rows:
            if not isinstance(row, Mapping):
                movement_fields_complete = False
                continue
            if not {"champion_id", "rank", "previous_rank", "rank_delta", "rating_delta", "movement"}.issubset(row):
                movement_fields_complete = False
    if cell_roles != {"top", "jungle", "mid", "bot", "support"}:
        blockers.append("champion_elo_cell_roles_incomplete")
    if cell_identity_failures:
        blockers.append("champion_elo_cell_identity_incomplete")
    if not movement_fields_complete:
        blockers.append("champion_elo_movement_fields_incomplete")

    source = payload.get("source")
    if not isinstance(source, Mapping) or not isinstance(source.get("raw_sha256"), str):
        blockers.append("champion_elo_source_binding_missing")

    return {
        "path": CANDIDATE_ELO_LOCATOR.as_posix(),
        "present": True,
        "artifact_sha256": _sha256(raw),
        "artifact_claim_sha256": submitted_artifact_sha,
        "history_start": payload.get("history_start"),
        "live_window_start": payload.get("live_window_start"),
        "source_mode": payload.get("source_mode"),
        "as_of": payload.get("as_of"),
        "expected_live_as_of": payload.get("expected_live_as_of"),
        "source_complete_through_expected_live_as_of": payload.get(
            "source_complete_through_expected_live_as_of"
        ),
        "maps_replayed": source.get("maps_replayed") if isinstance(source, Mapping) else None,
        "maps_in_live_window": source.get("maps_in_live_window") if isinstance(source, Mapping) else None,
        "cell_count": len(cells),
        "row_count": cell_rows,
        "league_count": len(options.get("leagues") or []) if isinstance(options, Mapping) else 0,
        "identity_sources": payload.get("identity_sources"),
        "blockers": sorted(set(blockers)),
    }, blockers


def _check_candidate_model(root: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    model_path = root / TERMINAL_MODEL_ARTIFACT["locator"]
    try:
        model_raw = model_path.read_bytes()
    except OSError:
        return {"path": TERMINAL_MODEL_ARTIFACT["locator"], "present": False}, [
            "terminal_model_missing"
        ]
    model_sha256 = _sha256(model_raw)
    if model_sha256 != TERMINAL_MODEL_ARTIFACT["raw_sha256"]:
        blockers.append("terminal_model_digest_mismatch")
    blockers.append("terminal_model_development_only")

    registry_path = root / CANDIDATE_REGISTRY_LOCATOR
    registry: dict[str, Any] = {}
    registry_raw: bytes | None = None
    if registry_path.is_file():
        registry_raw, registry = _read_json(registry_path)
    else:
        blockers.append("terminal_candidate_registry_missing")
    authority = registry.get("authority")
    if not isinstance(authority, Mapping):
        blockers.append("terminal_candidate_authority_missing")
    else:
        if authority.get("model_validation_authority") is not True:
            blockers.append("terminal_model_validation_authority_missing")
        if authority.get("neutral_probability_authority") is not True:
            blockers.append("terminal_neutral_probability_authority_missing")
    future_state = registry.get("future_state")
    if not isinstance(future_state, Mapping):
        blockers.append("future_capture_state_missing")
    else:
        if future_state.get("prediction_capture_present") is not True:
            blockers.append("future_prediction_capture_missing")
        if future_state.get("future_outcomes_present") is not True:
            blockers.append("future_observed_outcomes_missing")
    return {
        "path": TERMINAL_MODEL_ARTIFACT["locator"],
        "present": True,
        "artifact_sha256": model_sha256,
        "model_version": TERMINAL_MODEL_ARTIFACT["model_version"],
        "candidate_id": TERMINAL_MODEL_ARTIFACT["candidate_id"],
        "registry": {
            "path": CANDIDATE_REGISTRY_LOCATOR.as_posix(),
            "artifact_sha256": _sha256(registry_raw) if registry_raw is not None else None,
            "result_state": registry.get("result_state"),
            "future_state": future_state,
            "authority": authority,
        },
    }, blockers


def _check_terminal_l2(root: Path) -> tuple[dict[str, Any], list[str]]:
    """Include the existing terminal L2 audit without granting authority."""

    try:
        from lol_kills.v2.draft.terminal.l2_readiness import inspect_l2_readiness

        report = inspect_l2_readiness(root)
    except Exception as exc:  # pragma: no cover - protects the fail-closed audit path
        return {
            "status": "unreadable",
            "promotion_eligible": False,
            "error": f"{type(exc).__name__}: {exc}",
        }, ["terminal_l2_readiness_unreadable"]

    summary = {
        "status": report.get("status"),
        "promotion_eligible": report.get("promotion_eligible") is True,
        "blockers": sorted(set(report.get("blockers") or [])),
        "checks": report.get("checks"),
        "future_prediction_ledger": report.get("future_prediction_ledger"),
        "required_next_authority": report.get("required_next_authority"),
    }
    return summary, [] if summary["promotion_eligible"] else ["terminal_l2_readiness_blocked"]


def _record_checks(kind: str, payload: Mapping[str, Any]) -> dict[str, bool]:
    checks = {
        field: payload.get(field) == expected
        for field, expected in _RECORD_EXPECTATIONS[kind].items()
    }
    if kind == "independent_authority":
        ceiling = payload.get("claim_ceiling")
        checks["claim_ceiling_descriptive_only"] = isinstance(ceiling, Mapping) and ceiling.get(
            "descriptive_pre_map_association"
        ) is True and all(ceiling.get(field) is False for field in ("causal_draft_effect", "recommendation", "betting"))
    if kind == "promotion_manifest":
        checks["production_index_locator_present"] = isinstance(payload.get("production_index_locator"), str)
        checks["production_index_sha256_valid"] = isinstance(payload.get("production_index_sha256"), str) and bool(
            _SHA256_RE.fullmatch(payload["production_index_sha256"])
        )
        checks["source_tree_sha256_valid"] = isinstance(payload.get("source_tree_sha256"), str) and bool(
            _SHA256_RE.fullmatch(payload["source_tree_sha256"])
        )
        checks["commit_sha_valid"] = isinstance(payload.get("commit_sha"), str) and bool(
            _COMMIT_RE.fullmatch(payload["commit_sha"])
        )
    return checks


def _check_required_record(root: Path, kind: str, locator: Path, blocker: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        return {"path": locator.as_posix(), "present": False, "blocker": blocker}
    raw, payload = _read_json(path)
    checks = _record_checks(kind, payload)
    invalid = sorted(field for field, passed in checks.items() if not passed)
    return {
        "path": locator.as_posix(),
        "present": True,
        "artifact_sha256": _sha256(raw),
        "schema_version": payload.get("schema_version"),
        "status": payload.get("status"),
        "decision": payload.get("decision"),
        "valid": not invalid,
        "invalid_fields": invalid,
    }


def inspect_production_readiness(root: Path | str = Path(".")) -> dict[str, Any]:
    """Return a deterministic audit for the current production bundle."""

    repo_root = Path(root)
    blockers: list[str] = []
    index = _check_index(repo_root)
    champion_elo, champion_elo_blockers = _check_champion_elo_candidate(repo_root)
    blockers.extend(champion_elo_blockers)
    # The private terminal Draft Score package has a separate authority
    # boundary.  Keep its current state visible for operators, while keeping
    # its blocked predictive gate out of the descriptive tier-list decision.
    model, _model_blockers = _check_candidate_model(repo_root)
    terminal_l2, _terminal_l2_blockers = _check_terminal_l2(repo_root)

    prospective = _check_required_record(
        repo_root,
        "prospective_evaluation",
        PROSPECTIVE_EVALUATION_LOCATOR,
        "prospective_tier_evaluation_missing",
    )
    authority = _check_required_record(
        repo_root,
        "independent_authority",
        INDEPENDENT_AUTHORITY_LOCATOR,
        "independent_l2_authority_missing",
    )
    manifest = _check_required_record(
        repo_root,
        "promotion_manifest",
        PROMOTION_MANIFEST_LOCATOR,
        "production_promotion_manifest_missing",
    )
    for kind, record in (
        ("prospective_evaluation", prospective),
        ("independent_authority", authority),
        ("promotion_manifest", manifest),
    ):
        blocker = record.get("blocker")
        if blocker:
            blockers.append(blocker)
        elif record.get("valid") is not True:
            blockers.append(f"{kind}_record_invalid")

    if prospective.get("valid") is True and authority.get("valid") is True and manifest.get("valid") is True:
        candidate_raw, candidate = _read_json(repo_root / CANDIDATE_ELO_LOCATOR)
        evaluation_raw, evaluation = _read_json(repo_root / PROSPECTIVE_EVALUATION_LOCATOR)
        authority_raw, authority_payload = _read_json(repo_root / INDEPENDENT_AUTHORITY_LOCATOR)
        manifest_raw, manifest_payload = _read_json(repo_root / PROMOTION_MANIFEST_LOCATOR)
        if authority_payload.get("candidate", {}).get("raw_sha256") != _sha256(candidate_raw):
            blockers.append("authority_candidate_binding_invalid")
        if authority_payload.get("prospective_evaluation", {}).get("raw_sha256") != _sha256(evaluation_raw):
            blockers.append("authority_evaluation_binding_invalid")
        if manifest_payload.get("candidate", {}).get("raw_sha256") != _sha256(candidate_raw):
            blockers.append("manifest_candidate_binding_invalid")
        if manifest_payload.get("forward_evaluation", {}).get("raw_sha256") != _sha256(evaluation_raw):
            blockers.append("manifest_evaluation_binding_invalid")
        if manifest_payload.get("independent_authority", {}).get("raw_sha256") != _sha256(authority_raw):
            blockers.append("manifest_authority_binding_invalid")
        production_index_raw = (repo_root / INDEX_LOCATOR).read_bytes()
        if manifest_payload.get("production_index_sha256") != _sha256(production_index_raw):
            blockers.append("manifest_index_binding_invalid")

    if set(index["status_counts"]) - {"production", "unavailable"}:
        blockers.append("production_index_contains_unknown_status")
    if index["status_counts"].get("production", 0) == 0:
        blockers.append("production_index_has_no_numeric_cells")
    if index["status_counts"].get("development_only", 0):
        blockers.append("production_index_contains_development_cells")
    production_index = index.get("verified", {})
    if production_index.get("production_cell_count") != index["status_counts"].get("production", 0):
        blockers.append("production_index_cell_count_mismatch")
    promotion_eligible = not blockers
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "ready_for_promotion_review",
        "promotion_eligible": promotion_eligible,
        "claims": {
            "production": promotion_eligible,
            "publication": promotion_eligible,
            "rank_eligibility": promotion_eligible,
        },
        "candidate_index": index,
        "champion_elo_candidate": champion_elo,
        "candidate_model": model,
        "terminal_l2_readiness": terminal_l2,
        "prospective_evaluation": prospective,
        "independent_authority": authority,
        "promotion_manifest": manifest,
        "draft_score_boundary": {
            "scope": "private_terminal_draft_component",
            "production_authority": False,
            "blocks_descriptive_tier_api": False,
            "model": model,
            "l2_readiness": terminal_l2,
        },
        "blockers": sorted(set(blockers)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--strict", action="store_true", help="return exit code 1 when promotion is blocked")
    args = parser.parse_args()
    report = inspect_production_readiness(args.root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if args.strict and not report["promotion_eligible"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
