"""Independent registration and exact replay of one phase-two evaluation result."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import phase_one_evaluation_v1 as phase_one
from . import phase_two_evaluation_v1 as evaluation


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = (
    "lol_kills/v2/market/phase_two_evaluation_result_registry_v1.py"
)
SCHEMA_VERSION = "scryglass:phase-two-evaluation-independent-registry:v1"
REGISTRY_LOCATOR = Path(
    "data/lol/private_market_authority/phase-two-evaluation-registry-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PHASE_TWO_EVALUATION_REGISTRY_SHA256"
REVIEW_ATTESTATION = {
    "reviewer_not_model_quote_adapter_evaluator_author_or_outcome_custodian": True,
    "opening_authority_snapshot_outcome_evidence_and_result_hashes_verified": True,
    "all_primary_metrics_bootstraps_calibration_strata_capture_and_shadow_gates_replayed": True,
    "full_prospective_denominator_and_no_post_outcome_exclusion_verified": True,
    "no_post_opening_model_threshold_cohort_or_quote_change_found": True,
    "reported_pass_or_terminal_failure_reconciles_with_frozen_rules": True,
    "review_not_generated_by_the_evaluated_system": True,
}
AUTHORITY = {
    "phase_two_evaluation_identity_authority": True,
    "phase_two_market_gates_independently_passed": "derived_from_exact_result",
    "probability_authority": False,
    "fair_odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent identity and exact replay of one terminal phase-two evaluation. "
    "A registered pass is still not live probability, fair-odds, EV, stake, "
    "transaction, or betting authority; a separate market authority is required."
)


class PhaseTwoEvaluationRegistryError(RuntimeError):
    """The result, exact replay, review, registry, or external pin failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise PhaseTwoEvaluationRegistryError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseTwoEvaluationRegistryError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoEvaluationRegistryError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoEvaluationRegistryError(f"{field} must include a timezone")
    return parsed


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseTwoEvaluationRegistryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"invalid number: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PhaseTwoEvaluationRegistryError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PhaseTwoEvaluationRegistryError(f"{label} must contain an object")
    return value


def expected_result_binding(
    *, result_locator: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    locator = phase_one._locator(
        result_locator, evaluation.RESULT_PREFIX, "result_locator"
    )
    result_raw = phase_one._read_regular(root, locator, "phase-two evaluation result")
    result = evaluation.validate_phase_two_evaluation_result_v1(
        phase_one._strict_object(result_raw, "phase-two evaluation result")
    )
    outcome_locator = result["inputs"]["outcome_cohort_locator"]
    outcome_raw = phase_one._read_regular(
        root, outcome_locator, "phase-two sealed outcome cohort"
    )
    if _sha256(outcome_raw) != result["inputs"]["outcome_cohort_raw_sha256"]:
        raise PhaseTwoEvaluationRegistryError("evaluation outcome binding changed")
    replayed = evaluation.evaluate_phase_two_v1(
        snapshot_locator=result["inputs"]["snapshot_locator"],
        outcome_cohort_raw=outcome_raw,
        outcome_cohort_locator=outcome_locator,
        opening_authority_binding=result["opening_authority_binding"],
        run_id=result["run_id"],
        root=root,
        environment=environment,
        clock=lambda: _timestamp(result["evaluated_at_utc"], "evaluated_at_utc"),
    )
    if replayed != result:
        raise PhaseTwoEvaluationRegistryError(
            "phase-two evaluation does not exactly replay"
        )
    return {
        "result_locator": locator,
        "result_raw_sha256": _sha256(result_raw),
        "result_artifact_sha256": result["artifact_sha256"],
        "run_id": result["run_id"],
        "opening_authority_binding": result["opening_authority_binding"],
        "snapshot_artifact_sha256": result["inputs"]["snapshot_artifact_sha256"],
        "outcome_cohort_artifact_sha256": result["inputs"][
            "outcome_cohort_artifact_sha256"
        ],
        "phase_two_market_gates_passed": result[
            "phase_two_market_gates_passed"
        ],
        "evaluator_source_locator": evaluation.SOURCE_LOCATOR,
        "evaluator_source_raw_sha256": phase_one._sha256_path(
            root / evaluation.SOURCE_LOCATOR
        ),
        "exact_replay_verified": True,
    }


def validate_phase_two_evaluation_registry_v1(
    payload: Mapping[str, Any], *, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoEvaluationRegistryError("evaluation registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "registry_id", "status", "registered_at_utc",
        "reviews", "result_binding", "terminal_decision", "authority",
        "claim_ceiling",
    }:
        raise PhaseTwoEvaluationRegistryError("evaluation registry fields are not exact")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise PhaseTwoEvaluationRegistryError("evaluation registry schema changed")
    _nonempty(value.get("registry_id"), "registry_id")
    registered_at = _timestamp(value.get("registered_at_utc"), "registered_at_utc")
    if value.get("result_binding") != dict(expected_binding):
        raise PhaseTwoEvaluationRegistryError("evaluation result binding changed")
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise PhaseTwoEvaluationRegistryError("two independent result reviews are required")
    reviewers: set[str] = set()
    scopes: set[str] = set()
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope", "reviewer_id", "reviewed_at_utc", "attestation"
        }:
            raise PhaseTwoEvaluationRegistryError("result review structure changed")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        if (
            scope not in {"MODEL_RESULT", "MARKET_RESULT"}
            or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc")
            > registered_at
            or review.get("attestation") != REVIEW_ATTESTATION
        ):
            raise PhaseTwoEvaluationRegistryError("result review is incomplete")
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != {"MODEL_RESULT", "MARKET_RESULT"}:
        raise PhaseTwoEvaluationRegistryError("result reviews are not independent")
    passed = expected_binding.get("phase_two_market_gates_passed") is True
    expected_status = "REGISTERED_PASS" if passed else "REGISTERED_TERMINAL_FAILURE"
    if value.get("status") != expected_status:
        raise PhaseTwoEvaluationRegistryError("registry status does not match result")
    if value.get("terminal_decision") != {
        "phase_two_evaluation_independently_registered": True,
        "phase_two_market_gates_independently_passed": passed,
        "separate_match_winner_market_authority_may_be_considered": passed,
        "probability_or_betting_authorized": False,
        "failure_is_terminal_no_reopening_reselection_or_cohort_substitution": not passed,
    }:
        raise PhaseTwoEvaluationRegistryError("terminal decision changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoEvaluationRegistryError("evaluation registry exceeds authority")
    return value


def load_pinned_phase_two_evaluation_registry_v1(
    *, path: Path, external_sha256: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    digest = _sha(external_sha256, "external evaluation-registry digest")
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoEvaluationRegistryError("evaluation registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise PhaseTwoEvaluationRegistryError(
            "evaluation registry does not match its external pin"
        )
    receipt = validate_phase_two_evaluation_registry_v1(
        _object(raw, "phase-two evaluation registry"),
        expected_binding=expected_binding,
    )
    passed = receipt["terminal_decision"][
        "phase_two_market_gates_independently_passed"
    ]
    return {
        "status": "registered_pass" if passed else "registered_terminal_failure",
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "phase_two_evaluation_independently_registered": True,
        "phase_two_market_gates_independently_passed": passed,
        "probability_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "EXTERNAL_SHA256_ENV", "REGISTRY_LOCATOR", "REVIEW_ATTESTATION",
    "SCHEMA_VERSION", "SOURCE_LOCATOR", "PhaseTwoEvaluationRegistryError",
    "expected_result_binding", "load_pinned_phase_two_evaluation_registry_v1",
    "validate_phase_two_evaluation_registry_v1",
]
