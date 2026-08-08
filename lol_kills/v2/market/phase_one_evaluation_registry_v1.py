"""Independent registration of one terminal phase-one evaluation result."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from . import phase_one_evaluation_v1 as evaluation


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_one_evaluation_registry_v1.py"
SCHEMA_VERSION = "scryglass:phase-one-evaluation-independent-registry:v1"
REGISTRY_LOCATOR = Path(
    "data/lol/private_market_authority/phase-one-evaluation-registry-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PHASE_ONE_EVALUATION_REGISTRY_SHA256"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CLAIM_CEILING = {
    "phase_one_evaluation_independently_registered": True,
    "phase_one_models_independently_passed": "derived_from_exact_result",
    "phase_two_opening_authorized": False,
    "recalibration_authorized": False,
    "production_rating_authorized": False,
    "match_probability_authorized": False,
    "fair_odds_authorized": False,
    "expected_value_authorized": False,
    "recommendation_authorized": False,
    "transaction_authorized": False,
    "betting_authorized": False,
}


class PhaseOneEvaluationRegistryError(RuntimeError):
    """Independent evaluation registration is missing, stale, or malformed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise PhaseOneEvaluationRegistryError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseOneEvaluationRegistryError(f"{field} must be nonempty")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseOneEvaluationRegistryError(f"{field} must be RFC-3339") from exc
    if result.tzinfo is None:
        raise PhaseOneEvaluationRegistryError(f"{field} must include a timezone")
    return result


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseOneEvaluationRegistryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseOneEvaluationRegistryError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PhaseOneEvaluationRegistryError(f"{label} must contain an object")
    return value


def expected_result_binding(
    *, result_locator: str, root: Path = ROOT
) -> dict[str, Any]:
    locator = evaluation._locator(
        result_locator, evaluation.OUTPUT_PREFIX, "result_locator"
    )
    raw = evaluation._read_regular(root, locator, "phase-one evaluation result")
    result = evaluation.validate_phase_one_evaluation_result(
        evaluation._strict_object(raw, "phase-one evaluation result")
    )
    return {
        "result_locator": locator,
        "result_raw_sha256": _sha256(raw),
        "result_artifact_sha256": result["artifact_sha256"],
        "run_id": result["run_id"],
        "opening_authority_binding": result["opening_authority_binding"],
        "phase_one_models_passed": result["phase_one_models_passed"],
        "evaluator_source_locator": evaluation.SOURCE_LOCATOR,
        "evaluator_source_raw_sha256": evaluation._sha256_path(
            root / evaluation.SOURCE_LOCATOR
        ),
    }


def validate_evaluation_registry(
    payload: Mapping[str, Any], *, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneEvaluationRegistryError("evaluation registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "registry_id",
        "status",
        "registered_at_utc",
        "reviews",
        "result_binding",
        "terminal_decision",
        "claim_ceiling",
    }:
        raise PhaseOneEvaluationRegistryError("evaluation registry fields are not exact")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PhaseOneEvaluationRegistryError("evaluation registry schema changed")
    _nonempty(value.get("registry_id"), "registry_id")
    registered_at = _timestamp(value.get("registered_at_utc"), "registered_at_utc")
    if value.get("result_binding") != dict(expected_binding):
        raise PhaseOneEvaluationRegistryError("evaluation result binding changed")
    for key, item in expected_binding.items():
        if key.endswith("sha256"):
            _sha(item, f"result_binding.{key}")
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise PhaseOneEvaluationRegistryError("two independent result reviews are required")
    expected_scopes = {"RATINGS_RESULT", "TERMINAL_DRAFT_RESULT"}
    reviewers: set[str] = set()
    scopes: set[str] = set()
    attestation = {
        "reviewer_not_model_author_candidate_selector_evaluator_author_or_outcome_custodian": True,
        "exact_opening_authority_snapshot_outcome_and_result_hashes_verified": True,
        "registered_bootstrap_seeds_replicates_strata_and_gates_replayed": True,
        "no_post_opening_candidate_threshold_or_cohort_change_found": True,
        "reported_pass_or_failure_reconciles_with_locked_rules": True,
        "review_not_generated_by_the_evaluated_system": True,
    }
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope",
            "reviewer_id",
            "reviewed_at_utc",
            "attestation",
        }:
            raise PhaseOneEvaluationRegistryError("result review structure changed")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        if _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > registered_at:
            raise PhaseOneEvaluationRegistryError("registry predates a result review")
        if review.get("attestation") != attestation:
            raise PhaseOneEvaluationRegistryError("result review attestation is incomplete")
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != expected_scopes:
        raise PhaseOneEvaluationRegistryError("result reviews are not independent and complete")
    passed = expected_binding.get("phase_one_models_passed") is True
    expected_status = "REGISTERED_PASS" if passed else "REGISTERED_TERMINAL_FAILURE"
    if value.get("status") != expected_status:
        raise PhaseOneEvaluationRegistryError("registry status does not match the result")
    expected_decision = {
        "phase_one_evaluation_independently_registered": True,
        "phase_one_models_independently_passed": passed,
        "phase_two_available_for_separate_recalibration_uncertainty_and_opening_work": passed,
        "phase_two_opening_authorized": False,
        "recalibration_authorized": False,
        "failure_is_terminal_no_reopening_or_candidate_substitution": not passed,
    }
    if value.get("terminal_decision") != expected_decision:
        raise PhaseOneEvaluationRegistryError("terminal phase-one decision changed")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseOneEvaluationRegistryError("evaluation registry claim ceiling changed")
    return value


def load_pinned_evaluation_registry(
    *,
    path: Path,
    external_sha256: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    expected_digest = _sha(external_sha256, "external registry digest")
    if path.is_symlink() or not path.is_file():
        raise PhaseOneEvaluationRegistryError("evaluation registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != expected_digest:
        raise PhaseOneEvaluationRegistryError("evaluation registry does not match its external pin")
    receipt = validate_evaluation_registry(
        _object(raw, "evaluation registry"), expected_binding=expected_binding
    )
    passed = receipt["terminal_decision"]["phase_one_models_independently_passed"]
    return {
        "status": "registered_pass" if passed else "registered_terminal_failure",
        "receipt": receipt,
        "receipt_raw_sha256": expected_digest,
        "phase_one_evaluation_independently_registered": True,
        "phase_one_models_independently_passed": passed,
        "phase_two_opening_authorized": False,
        "probability_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "EXTERNAL_SHA256_ENV",
    "REGISTRY_LOCATOR",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "PhaseOneEvaluationRegistryError",
    "expected_result_binding",
    "load_pinned_evaluation_registry",
    "validate_evaluation_registry",
]
