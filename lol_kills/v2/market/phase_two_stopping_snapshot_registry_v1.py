"""Externally pin the first independently reviewed support-met phase-two snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import phase_one_evaluation_v1 as evaluation
from . import phase_two_stopping_snapshot_v1 as snapshot


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_stopping_snapshot_registry_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-stopping-snapshot-registry:v1"
REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/stopping-snapshot-registry-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PRIVATE_MATCH_WINNER_PHASE_TWO_SNAPSHOT_SHA256"
REVIEW_ATTESTATION = {
    "reviewer_independent_of_model_quote_adapter_evaluator_and_outcome_custodian": True,
    "all_plan_completion_quote_failure_and_map_start_bindings_replayed": True,
    "coverage_denominator_includes_every_completed_prospective_plan": True,
    "exact_frozen_metadata_stopping_rule_recomputed": True,
    "this_is_the_first_snapshot_meeting_all_minima": True,
    "phase_two_outcomes_absent_and_unaccessed_during_review": True,
    "manual_post_outcome_exclusion_or_success_only_selection_not_found": True,
    "review_not_generated_by_the_evaluated_system": True,
}
AUTHORITY = {
    "phase_two_snapshot_identity_authority": True,
    "phase_two_outcome_opening_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent identity registration of the first outcome-free phase-two "
    "snapshot meeting the frozen metadata floors. Separate sealed-outcome "
    "authority and evaluation are still required; no betting authority exists."
)


class PhaseTwoSnapshotRegistryError(RuntimeError):
    """The support-met snapshot, review, registry bytes, or external pin failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise PhaseTwoSnapshotRegistryError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseTwoSnapshotRegistryError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoSnapshotRegistryError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoSnapshotRegistryError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseTwoSnapshotRegistryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseTwoSnapshotRegistryError("registry is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PhaseTwoSnapshotRegistryError("registry must be an object")
    return value


def expected_snapshot_binding(
    *, snapshot_locator: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    locator = evaluation._locator(
        snapshot_locator, snapshot.OUTPUT_PREFIX, "snapshot_locator"
    )
    raw = evaluation._read_regular(root, locator, "phase-two stopping snapshot")
    try:
        checked = snapshot.validate_phase_two_stopping_snapshot_v1(
            evaluation._strict_object(raw, "phase-two stopping snapshot"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise PhaseTwoSnapshotRegistryError("stopping snapshot is invalid") from exc
    if checked["support"]["support_met"] is not True or checked["support"]["terminal_shadow_support_failure"] is not False:
        raise PhaseTwoSnapshotRegistryError("snapshot did not meet the frozen support rule")
    return {
        "snapshot_locator": locator,
        "snapshot_raw_sha256": _sha256(raw),
        "snapshot_artifact_sha256": checked["artifact_sha256"],
        "entries_sha256": checked["entries_sha256"],
        "captured_at_utc": checked["captured_at_utc"],
        "eligible_quoted_maps": checked["support"]["eligible_quoted_maps"],
        "otherwise_eligible_maps": checked["support"]["otherwise_eligible_maps"],
        "eligible_series": checked["support"]["eligible_series"],
        "quote_coverage": checked["support"]["quote_coverage"],
        "shadow_policy_qualifying_maps": checked["support"]["shadow_policy_qualifying_maps"],
        "support_met": True,
        "outcomes_accessed": False,
    }


def validate_phase_two_snapshot_registry_v1(
    payload: Mapping[str, Any], *, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoSnapshotRegistryError("registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "registry_id", "status", "issued_at_utc",
        "independent_review", "snapshot_binding", "decision", "authority",
        "claim_ceiling",
    }:
        raise PhaseTwoSnapshotRegistryError("registry fields are not exact")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != "FIRST_SUPPORT_MET_PHASE_TWO_SNAPSHOT_REGISTERED_OUTCOMES_UNOPENED":
        raise PhaseTwoSnapshotRegistryError("registry identity changed")
    _nonempty(value.get("registry_id"), "registry_id")
    issued = _timestamp(value.get("issued_at_utc"), "issued_at_utc")
    binding = dict(expected_binding)
    if value.get("snapshot_binding") != binding or _timestamp(binding["captured_at_utc"], "snapshot.captured_at") > issued:
        raise PhaseTwoSnapshotRegistryError("snapshot binding changed")
    review = value.get("independent_review")
    if not isinstance(review, Mapping) or set(review) != {"reviewer_id", "reviewed_at_utc", "attestation"}:
        raise PhaseTwoSnapshotRegistryError("review structure changed")
    if (
        not _nonempty(review.get("reviewer_id"), "reviewer_id")
        or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > issued
        or review.get("attestation") != REVIEW_ATTESTATION
    ):
        raise PhaseTwoSnapshotRegistryError("review is incomplete")
    if value.get("decision") != {
        "first_support_met_snapshot_independently_registered": True,
        "phase_two_outcomes_opened": False,
        "evaluation_authorized": False,
        "betting_authorized": False,
    }:
        raise PhaseTwoSnapshotRegistryError("registry decision changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoSnapshotRegistryError("registry exceeds authority")
    return value


def load_pinned_phase_two_snapshot_registry_v1(
    *, path: Path, external_sha256: str, expected_binding: Mapping[str, Any]
) -> dict[str, Any]:
    digest = _sha(external_sha256, "external registry digest")
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoSnapshotRegistryError("registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise PhaseTwoSnapshotRegistryError("registry does not match its external pin")
    receipt = validate_phase_two_snapshot_registry_v1(
        _object(raw), expected_binding=expected_binding
    )
    return {
        "status": "first_support_met_snapshot_registered_outcomes_unopened",
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "phase_two_snapshot_identity_authority": True,
        "phase_two_outcome_opening_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "EXTERNAL_SHA256_ENV", "REGISTRY_LOCATOR", "REVIEW_ATTESTATION",
    "SCHEMA_VERSION", "SOURCE_LOCATOR", "PhaseTwoSnapshotRegistryError",
    "expected_snapshot_binding", "load_pinned_phase_two_snapshot_registry_v1",
    "validate_phase_two_snapshot_registry_v1",
]
