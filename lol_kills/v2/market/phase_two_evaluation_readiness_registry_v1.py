"""Externally pin independently reviewed phase-two evaluation readiness."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import phase_one_evaluation_v1 as phase_one
from . import phase_two_evaluation_readiness_v1 as readiness


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = (
    "lol_kills/v2/market/phase_two_evaluation_readiness_registry_v1.py"
)
SCHEMA_VERSION = "scryglass:phase-two-evaluation-readiness-registry:v1"
REGISTRY_LOCATOR = Path(
    "data/lol/private_market_authority/"
    "phase-two-evaluation-readiness-registry-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PHASE_TWO_EVALUATION_READINESS_REGISTRY_SHA256"
REVIEW_ATTESTATION = {
    "reviewer_independent_of_model_quote_adapter_evaluator_opening_and_outcome_custodian": True,
    "all_source_hashes_signatures_schemas_dependencies_and_empty_state_replayed": True,
    "bootstrap_seed_resamples_metrics_calibration_capture_shadow_and_strata_gates_verified": True,
    "one_time_marker_before_outcome_read_and_exact_replay_registry_verified": True,
    "outcomes_absent_and_unaccessed_during_review": True,
    "all_authority_outputs_remain_false_or_null": True,
    "review_not_generated_by_the_evaluated_system": True,
}
AUTHORITY = {
    "phase_two_evaluation_readiness_identity_authority": True,
    "phase_two_outcome_opening_authority": False,
    "phase_two_evaluation_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent identity registration of the outcome-free phase-two evaluation "
    "implementation freeze. Separate sealed-outcome authority and evaluation "
    "remain required; no model, probability, EV, stake, or betting authority exists."
)


class PhaseTwoEvaluationReadinessRegistryError(RuntimeError):
    """The readiness artifact, independent review, or external pin failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise PhaseTwoEvaluationReadinessRegistryError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseTwoEvaluationReadinessRegistryError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoEvaluationReadinessRegistryError(
            f"{field} must be RFC-3339"
        ) from exc
    if parsed.tzinfo is None:
        raise PhaseTwoEvaluationReadinessRegistryError(
            f"{field} must include a timezone"
        )
    return parsed


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseTwoEvaluationReadinessRegistryError(
                f"duplicate JSON key: {key}"
            )
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseTwoEvaluationReadinessRegistryError(
            f"{label} is not strict JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PhaseTwoEvaluationReadinessRegistryError(
            f"{label} must contain an object"
        )
    return value


def expected_readiness_binding(
    *, root: Path = ROOT, environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    path = root / readiness.DEFAULT_OUTPUT
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoEvaluationReadinessRegistryError(
            "phase-two evaluation readiness is unavailable"
        )
    raw = path.read_bytes()
    try:
        checked = readiness.validate_phase_two_evaluation_readiness_v1(
            phase_one._strict_object(raw, "phase-two evaluation readiness"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise PhaseTwoEvaluationReadinessRegistryError(
            "phase-two evaluation readiness is invalid"
        ) from exc
    return {
        "locator": readiness.DEFAULT_OUTPUT.as_posix(),
        "raw_sha256": _sha256(raw),
        "artifact_sha256": checked["artifact_sha256"],
        "result_state": checked["result_state"],
        "locked_at_utc": checked["locked_at_utc"],
        "outcomes_accessed": False,
    }


def validate_phase_two_evaluation_readiness_registry_v1(
    payload: Mapping[str, Any], *, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoEvaluationReadinessRegistryError("registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "registry_id", "status", "registered_at_utc",
        "review", "readiness_binding", "decision", "authority", "claim_ceiling",
    }:
        raise PhaseTwoEvaluationReadinessRegistryError("registry fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "OUTCOME_FREE_EVALUATION_READINESS_REGISTERED"
    ):
        raise PhaseTwoEvaluationReadinessRegistryError("registry identity changed")
    _nonempty(value.get("registry_id"), "registry_id")
    registered = _timestamp(value.get("registered_at_utc"), "registered_at_utc")
    if value.get("readiness_binding") != dict(expected_binding):
        raise PhaseTwoEvaluationReadinessRegistryError("readiness binding changed")
    review = value.get("review")
    if not isinstance(review, Mapping) or set(review) != {
        "reviewer_id", "reviewed_at_utc", "attestation"
    }:
        raise PhaseTwoEvaluationReadinessRegistryError("review structure changed")
    if (
        not _nonempty(review.get("reviewer_id"), "reviewer_id")
        or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > registered
        or review.get("attestation") != REVIEW_ATTESTATION
    ):
        raise PhaseTwoEvaluationReadinessRegistryError("review is incomplete")
    if value.get("decision") != {
        "phase_two_evaluation_readiness_independently_registered": True,
        "phase_two_outcomes_opened": False,
        "phase_two_evaluation_run": False,
        "probability_or_betting_authorized": False,
    }:
        raise PhaseTwoEvaluationReadinessRegistryError("registry decision changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoEvaluationReadinessRegistryError("registry exceeds authority")
    return value


def load_pinned_phase_two_evaluation_readiness_registry_v1(
    *, path: Path, external_sha256: str,
    expected_binding: Mapping[str, Any],
) -> dict[str, Any]:
    digest = _sha(external_sha256, "external registry digest")
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoEvaluationReadinessRegistryError("registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise PhaseTwoEvaluationReadinessRegistryError(
            "registry does not match its external pin"
        )
    receipt = validate_phase_two_evaluation_readiness_registry_v1(
        _object(raw, "phase-two evaluation readiness registry"),
        expected_binding=expected_binding,
    )
    return {
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "phase_two_evaluation_readiness_independently_registered": True,
        "phase_two_outcome_opening_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "EXTERNAL_SHA256_ENV", "REGISTRY_LOCATOR", "REVIEW_ATTESTATION",
    "SCHEMA_VERSION", "SOURCE_LOCATOR",
    "PhaseTwoEvaluationReadinessRegistryError", "expected_readiness_binding",
    "load_pinned_phase_two_evaluation_readiness_registry_v1",
    "validate_phase_two_evaluation_readiness_registry_v1",
]
