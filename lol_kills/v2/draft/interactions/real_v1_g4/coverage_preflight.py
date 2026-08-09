"""Outcome-free coverage preflight for the corrected 52-slot G4 contract.

The preflight authenticates the registered aggregate support artifact, then
replays every G4 slot's chronological feature/fit-availability coverage gate.
It never loads target, M0, outcome, fit, prediction, or final-holdout rows.
When any frozen block is below support, the only valid output is a typed M0
no-winner result; a later permit cannot turn unsupported rows into a fit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from lol_kills.v2.draft.interactions import representation_rank_assay as assay
from lol_kills.v2.draft.interactions import representation_rank_private_runner as private_runner

from . import contract


ROOT = contract.ROOT
OLD_PRIVATE_CONTRACT = ROOT / "data/lol/v2/models/draft-interactions/representation-rank-private-run-contract.json"
OUTPUT_PATH = ROOT / "data/lol/v2/models/draft-interactions/real-v1-g4/coverage-preflight.json"
SCHEMA_PATH = ROOT / "data/lol/v2/models/draft-interactions/real-v1-g4/coverage-preflight.schema.json"
SCHEMA = "scryglass:real-v1-g4-coverage-preflight:v1"


class CoveragePreflightError(ValueError):
    """Raised when the outcome-free G4 coverage replay cannot be trusted."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise CoveragePreflightError("coverage preflight payload is not canonical") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _slot_split(stage: str) -> str:
    if stage == "inner":
        return "train"
    if stage in {"development", "validation"}:
        return stage
    raise CoveragePreflightError("unknown G4 stage")


def _coverage_counts(report: Mapping[str, Any]) -> dict[str, Any]:
    overall = report.get("overall")
    by_month = report.get("by_month")
    if not isinstance(overall, Mapping) or not isinstance(by_month, Mapping) or len(by_month) != 1:
        raise CoveragePreflightError("coverage report aggregate schema changed")
    required = ("maps", "eligible_maps", "clusters", "eligible_clusters")

    def counts(value: Mapping[str, Any]) -> dict[str, int]:
        if any(not isinstance(value.get(key), int) or isinstance(value.get(key), bool) or value[key] < 0 for key in required):
            raise CoveragePreflightError("coverage count is invalid")
        return {key: int(value[key]) for key in required}

    return {"overall": counts(overall), "month": counts(next(iter(by_month.values())))}


def _load_outcome_free_domains() -> tuple[Any, Any, Any]:
    if OLD_PRIVATE_CONTRACT.is_symlink() or not OLD_PRIVATE_CONTRACT.is_file():
        raise CoveragePreflightError("private G4 contract is not a regular file")
    old_contract = private_runner.load_contract(OLD_PRIVATE_CONTRACT, root=ROOT)
    feature = private_runner.load_authoritative_features(old_contract, root=ROOT)
    availability = private_runner.load_fit_availability_domain(old_contract, feature, root=ROOT)
    feature_domain = private_runner.likelihood_feature_domain(feature, availability.ordered_game_ids)
    if any(row[1] == assay.FINAL_SPLIT for row in feature_domain.records):
        raise CoveragePreflightError("final holdout entered outcome-free feature domain")
    return feature_domain, availability, old_contract


def _slot_preflight(slot: Mapping[str, Any], *, feature_domain: Any, availability: Any) -> dict[str, Any]:
    split = _slot_split(str(slot["stage"]))
    month = str(slot["calendar_month"])
    score_ids = [row[0] for row in feature_domain.records if row[1] == split and row[3] == month]
    fit_ids = [row[0] for row in feature_domain.records if row[3] < month]
    if not score_ids or not fit_ids:
        raise CoveragePreflightError("G4 slot has no exact chronological rows")
    try:
        coverage = assay.outcome_free_coverage(
            feature_domain=feature_domain,
            score_game_ids=score_ids,
            fit_game_ids=fit_ids,
            split=split,
            fit_availability_domain=availability,
        )
    except Exception as error:
        raise CoveragePreflightError(f"G4 slot coverage derivation failed: {error}") from error
    counts = _coverage_counts(coverage.report)
    fit_support = coverage.report.get("fit_support")
    if not isinstance(fit_support, Mapping):
        raise CoveragePreflightError("coverage fit-support report missing")
    return {
        "sequence": int(slot["sequence"]),
        "stage": str(slot["stage"]),
        "calendar_month": month,
        "family": str(slot["family"]),
        "penalty": slot["penalty"],
        "width": slot["width"],
        "split": split,
        "score_map_count": len(score_ids),
        "fit_map_count": len(fit_ids),
        "coverage_passed": bool(coverage.report.get("passed")),
        "coverage": counts,
        "fit_support": {
            "input_maps": int(fit_support.get("input_maps", 0)),
            "retained_maps": int(fit_support.get("retained_maps", 0)),
            "eligible_nodes": int(fit_support.get("eligible_nodes", 0)),
            "convergence_checks": int(fit_support.get("convergence_checks", 0)),
            "changing_rounds": int(fit_support.get("changing_rounds", 0)),
        },
        "execution_status": "coverage_pass" if coverage.report.get("passed") else "coverage_gate_failed_before_target_m0_or_outcome_load",
    }


def build_preflight() -> dict[str, Any]:
    """Replay the frozen 52-slot support gate without protected labels."""

    feature_domain, availability, old_contract = _load_outcome_free_domains()
    pending = contract.build_pending_artifacts()
    chronology = pending["chronology-contract.json"]
    support = contract._authenticate_2026_support()
    coverage_cache: dict[tuple[str, str], dict[str, Any]] = {}
    slots: list[dict[str, Any]] = []
    for slot in chronology["execution_slots"]:
        cache_key = (str(slot["stage"]), str(slot["calendar_month"]))
        if cache_key not in coverage_cache:
            coverage_cache[cache_key] = _slot_preflight(
                slot, feature_domain=feature_domain, availability=availability
            )
        cached = dict(coverage_cache[cache_key])
        cached.update(
            {
                "sequence": int(slot["sequence"]),
                "family": str(slot["family"]),
                "penalty": slot["penalty"],
                "width": slot["width"],
            }
        )
        slots.append(cached)
    failures = [slot for slot in slots if not slot["coverage_passed"]]
    first_failure = failures[0] if failures else None
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "status": "NO_INCREMENTAL_DRAFT_WINNER" if failures else "READY_FOR_FRESH_PERMIT_AND_EXECUTION",
        "reason_code": "OUTCOME_FREE_COVERAGE_GATE_FAILED" if failures else None,
        "selected_model": None,
        "selected_width": None,
        "fallback": "M0_NOT_SCORED" if failures else None,
        "chronology_contract_sha256": chronology["artifact_sha256"],
        "review_core_sha256": pending["review-core.json"]["artifact_sha256"],
        "support_first_verified": True,
        "support": support,
        "source": {
            "private_contract_locator": str(OLD_PRIVATE_CONTRACT.relative_to(ROOT)),
            "private_contract_artifact_sha256": old_contract["artifact_sha256"],
            "feature_domain_sha256": feature_domain.artifact_sha256,
            "feature_source_raw_sha256": feature_domain.source_raw_sha256,
            "fit_availability_domain_sha256": availability.artifact_sha256,
            "fit_availability_source_raw_sha256": availability.source_raw_sha256,
            "feature_map_count": len(feature_domain.records),
            "fit_availability_map_count": len(availability.ordered_game_ids),
        },
        "slots": slots,
        "failed_slot_count": len(failures),
        "first_failure": first_failure,
        "target_loader_calls": 0,
        "m0_loader_calls": 0,
        "outcome_loader_calls": 0,
        "fit_execution_calls": 0,
        "final_holdout_loaded": False,
        "claim_ceiling": {
            "outcome_free_support_preflight": True,
            "private_model_fit": False,
            "private_rank_selection": False,
            "prediction": False,
            "publication": False,
            "production": False,
            "promotion": False,
            "sota": False,
            "final_holdout": False,
        },
    }
    body["artifact_sha256"] = _sha256(body)
    return body


def validate_preflight(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise CoveragePreflightError("coverage preflight must be an object")
    claimed = payload.get("artifact_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(claimed, str) or _sha256(unsigned) != claimed:
        raise CoveragePreflightError("coverage preflight digest is invalid")
    expected = build_preflight()
    if dict(payload) != expected:
        raise CoveragePreflightError("coverage preflight differs from replayed evidence")
    if payload.get("target_loader_calls") != 0 or payload.get("m0_loader_calls") != 0 or payload.get("outcome_loader_calls") != 0 or payload.get("fit_execution_calls") != 0:
        raise CoveragePreflightError("protected loader was called")
    if payload.get("final_holdout_loaded") is not False:
        raise CoveragePreflightError("final holdout boundary changed")
    return expected


def canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    validate_preflight(payload)
    return _canonical_bytes(dict(payload)) + b"\n"


__all__ = ["CoveragePreflightError", "OUTPUT_PATH", "SCHEMA_PATH", "SCHEMA", "build_preflight", "validate_preflight", "canonical_bytes"]
