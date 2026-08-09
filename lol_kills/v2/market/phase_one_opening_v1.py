"""Independent, externally pinned, one-time phase-one outcome opening gate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any, Callable, Mapping

from . import phase_one_collection_v1 as collection
from . import phase_one_evaluation_v1 as evaluation


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_one_opening_v1.py"
SCHEMA_VERSION = "scryglass:phase-one-independent-opening-authority:v1"
AUTHORITY_LOCATOR = Path(
    "data/lol/private_market_authority/phase-one-opening-authority-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PHASE_ONE_OPENING_AUTHORITY_SHA256"
OPENING_MARKER_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/opening-markers"
)
OUTPUT_PREFIX = evaluation.OUTPUT_PREFIX
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
CLAIM_CEILING = {
    "one_time_phase_one_outcome_opening_authorized": True,
    "phase_one_evaluation_authorized": True,
    "production_rating_authorized": False,
    "match_probability_authorized": False,
    "fair_odds_authorized": False,
    "expected_value_authorized": False,
    "recommendation_authorized": False,
    "transaction_authorized": False,
    "betting_authorized": False,
}


class PhaseOneOpeningError(RuntimeError):
    """Independent opening authority is absent, stale, or already consumed."""


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA_RE.fullmatch(value):
        raise PhaseOneOpeningError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseOneOpeningError(f"{field} must be nonempty")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    try:
        result = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseOneOpeningError(f"{field} must be RFC-3339") from exc
    if result.tzinfo is None:
        raise PhaseOneOpeningError(f"{field} must include a timezone")
    return result.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseOneOpeningError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseOneOpeningError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PhaseOneOpeningError(f"{label} must contain an object")
    return value


def _locator(value: Any, prefix: PurePosixPath, field: str) -> str:
    path = PurePosixPath(_nonempty(value, field))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(prefix.parts)]) != prefix.parts
        or path.suffix != ".json"
    ):
        raise PhaseOneOpeningError(f"{field} is outside its registered root")
    return path.as_posix()


def _regular(root: Path, locator: str, label: str) -> bytes:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseOneOpeningError(f"{label} is not an unaliased regular file")
    return path.read_bytes()


def current_expected_bindings(
    *,
    snapshot_locator: str,
    parity_locator: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Resolve every outcome-free byte binding without reading outcome bytes."""

    from .phase_one_evaluation_readiness_registry_v1 import (
        REGISTERED_READINESS_ARTIFACT_SHA256,
        REGISTERED_READINESS_LOCATOR,
        REGISTERED_READINESS_RAW_SHA256,
        validate_registered_phase_one_evaluation_readiness_v1,
    )

    readiness = validate_registered_phase_one_evaluation_readiness_v1(root=root)
    snapshot_raw, snapshot = evaluation._snapshot(root, snapshot_locator)
    parity_locator = evaluation._locator(
        parity_locator, evaluation.PARITY_PREFIX, "parity_locator"
    )
    parity_raw = evaluation._read_regular(root, parity_locator, "parity registry")
    parity = evaluation.validate_draft_replay_parity_registry(
        evaluation._strict_object(parity_raw, "parity registry"), root=root
    )
    if parity["snapshot"]["artifact_sha256"] != snapshot["artifact_sha256"]:
        raise PhaseOneOpeningError("parity registry is not bound to the snapshot")
    return {
        "joint_snapshot_locator": snapshot_locator,
        "joint_snapshot_raw_sha256": _sha256_bytes(snapshot_raw),
        "joint_snapshot_artifact_sha256": snapshot["artifact_sha256"],
        "draft_parity_locator": parity_locator,
        "draft_parity_raw_sha256": _sha256_bytes(parity_raw),
        "draft_parity_artifact_sha256": parity["artifact_sha256"],
        "protocols": snapshot["protocols"],
        "evaluation_readiness_locator": REGISTERED_READINESS_LOCATOR.as_posix(),
        "evaluation_readiness_raw_sha256": REGISTERED_READINESS_RAW_SHA256,
        "evaluation_readiness_artifact_sha256": REGISTERED_READINESS_ARTIFACT_SHA256,
        "evaluation_readiness_result_state": readiness["result_state"],
        "evaluator_source_locator": evaluation.SOURCE_LOCATOR,
        "evaluator_source_raw_sha256": evaluation._sha256_path(
            root / evaluation.SOURCE_LOCATOR
        ),
        "opening_source_locator": SOURCE_LOCATOR,
        "opening_source_raw_sha256": evaluation._sha256_path(root / SOURCE_LOCATOR),
    }


def validate_opening_authority(
    payload: Mapping[str, Any], *, expected_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseOneOpeningError("opening authority must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "authority_id",
        "status",
        "scope",
        "reviewed_at_utc",
        "reviews",
        "bindings",
        "sealed_outcomes",
        "one_time_run",
        "claim_ceiling",
    }:
        raise PhaseOneOpeningError("opening authority fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "APPROVED"
        or value.get("scope") != "ONE_TIME_JOINT_PHASE_ONE_MODEL_EVALUATION_ONLY"
    ):
        raise PhaseOneOpeningError("opening authority identity changed")
    _nonempty(value.get("authority_id"), "authority_id")
    reviewed_at = _timestamp(value.get("reviewed_at_utc"), "reviewed_at_utc")
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise PhaseOneOpeningError("exactly two independent reviews are required")
    expected_scopes = {"RATINGS_FUTURE_HOLDOUT", "TERMINAL_DRAFT_FUTURE_HOLDOUT"}
    reviewer_ids: set[str] = set()
    observed_scopes: set[str] = set()
    expected_attestation = {
        "reviewer_not_model_author_candidate_selector_or_evaluator_author": True,
        "reviewer_not_outcome_custodian": True,
        "reviewer_used_only_pinned_outcome_free_evidence": True,
        "outcomes_not_accessed_before_approval": True,
        "metadata_stopping_rule_independently_verified": True,
        "no_candidate_or_threshold_reselection_approved": True,
        "approval_not_generated_by_the_evaluated_system": True,
    }
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope",
            "reviewer_id",
            "reviewed_at_utc",
            "independence_attestation",
        }:
            raise PhaseOneOpeningError("independent review structure changed")
        reviewer_id = _nonempty(review.get("reviewer_id"), "reviewer_id")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        if _timestamp(review.get("reviewed_at_utc"), "review.reviewed_at") > reviewed_at:
            raise PhaseOneOpeningError("authority predates an independent review")
        if review.get("independence_attestation") != expected_attestation:
            raise PhaseOneOpeningError("independence attestation is incomplete")
        reviewer_ids.add(reviewer_id)
        observed_scopes.add(scope)
    if len(reviewer_ids) != 2 or observed_scopes != expected_scopes:
        raise PhaseOneOpeningError("reviews are not independent and scope-complete")
    if value.get("bindings") != dict(expected_bindings):
        raise PhaseOneOpeningError("opening bindings do not match current evidence")
    for key, item in expected_bindings.items():
        if key.endswith("sha256"):
            _sha(item, f"bindings.{key}")

    sealed = value.get("sealed_outcomes")
    if not isinstance(sealed, Mapping) or set(sealed) != {
        "cohort_locator",
        "cohort_raw_sha256",
        "custodian_id",
        "sealed_at_utc",
        "custodian_attestation",
    }:
        raise PhaseOneOpeningError("sealed outcome binding is incomplete")
    _locator(sealed.get("cohort_locator"), evaluation.OUTCOME_PREFIX, "cohort_locator")
    _sha(sealed.get("cohort_raw_sha256"), "cohort_raw_sha256")
    _nonempty(sealed.get("custodian_id"), "custodian_id")
    if _timestamp(sealed.get("sealed_at_utc"), "sealed_at_utc") > reviewed_at:
        raise PhaseOneOpeningError("opening review predates the sealed cohort")
    if sealed.get("custodian_attestation") != {
        "digest_created_without_disclosing_outcomes_to_model_authors_or_reviewers": True,
        "cohort_bytes_immutable_after_digest": True,
        "cohort_matches_the_joint_snapshot_without_manual_post_outcome_exclusion": True,
    }:
        raise PhaseOneOpeningError("outcome custodian attestation is incomplete")

    one_time = value.get("one_time_run")
    if not isinstance(one_time, Mapping) or set(one_time) != {
        "run_id",
        "opening_marker_locator",
        "authorized_output_locator",
        "marker_written_before_outcome_read",
        "atomic_no_clobber_output_required",
        "partial_result_publication_prohibited",
        "second_opening_or_replacement_holdout_prohibited",
    }:
        raise PhaseOneOpeningError("one-time run contract is incomplete")
    _nonempty(one_time.get("run_id"), "run_id")
    _locator(
        one_time.get("opening_marker_locator"),
        OPENING_MARKER_PREFIX,
        "opening_marker_locator",
    )
    _locator(
        one_time.get("authorized_output_locator"),
        OUTPUT_PREFIX,
        "authorized_output_locator",
    )
    if any(
        one_time.get(field) is not True
        for field in (
            "marker_written_before_outcome_read",
            "atomic_no_clobber_output_required",
            "partial_result_publication_prohibited",
            "second_opening_or_replacement_holdout_prohibited",
        )
    ):
        raise PhaseOneOpeningError("one-time protections changed")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseOneOpeningError("opening claim ceiling changed")
    return value


def load_pinned_opening_authority(
    *,
    path: Path,
    external_sha256: str,
    expected_bindings: Mapping[str, Any],
) -> dict[str, Any]:
    expected_digest = _sha(external_sha256, "external authority digest")
    if path.is_symlink() or not path.is_file():
        raise PhaseOneOpeningError("opening authority is unavailable")
    raw = path.read_bytes()
    if _sha256_bytes(raw) != expected_digest:
        raise PhaseOneOpeningError("opening authority does not match its external pin")
    receipt = validate_opening_authority(
        _object(raw, "opening authority"), expected_bindings=expected_bindings
    )
    return {
        "receipt": receipt,
        "receipt_raw_sha256": expected_digest,
        "authority_id": receipt["authority_id"],
        "run_id": receipt["one_time_run"]["run_id"],
    }


def _write_marker_no_clobber(path: Path, payload: Mapping[str, Any]) -> None:
    evaluation.write_no_clobber(path, payload)


def run_authorized_phase_one_evaluation(
    *,
    snapshot_locator: str,
    parity_locator: str,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Consume one exact authority, mark the opening, then read outcomes once."""

    external = environment.get(EXTERNAL_SHA256_ENV)
    if not external:
        raise PhaseOneOpeningError("external opening-authority digest is missing")
    expected = current_expected_bindings(
        snapshot_locator=snapshot_locator,
        parity_locator=parity_locator,
        root=root,
    )
    loaded = load_pinned_opening_authority(
        path=root / AUTHORITY_LOCATOR,
        external_sha256=external,
        expected_bindings=expected,
    )
    receipt = loaded["receipt"]
    one_time = receipt["one_time_run"]
    marker_path = root / one_time["opening_marker_locator"]
    output_path = root / one_time["authorized_output_locator"]
    if marker_path.exists() or marker_path.is_symlink() or output_path.exists() or output_path.is_symlink():
        raise PhaseOneOpeningError("one-time opening was already started or consumed")
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PhaseOneOpeningError("opening clock must be timezone-aware")
    observed = observed.astimezone(timezone.utc)
    marker = {
        "schema_version": "scryglass:phase-one-opening-consumption-marker:v1",
        "result_state": "ONE_TIME_OPENING_STARTED_OUTCOMES_NOT_YET_READ",
        "started_at_utc": observed.isoformat(),
        "authority_id": receipt["authority_id"],
        "authority_raw_sha256": loaded["receipt_raw_sha256"],
        "run_id": one_time["run_id"],
        "outcome_cohort_locator": receipt["sealed_outcomes"]["cohort_locator"],
        "outcome_cohort_expected_raw_sha256": receipt["sealed_outcomes"]["cohort_raw_sha256"],
        "second_opening_prohibited": True,
    }
    _write_marker_no_clobber(marker_path, marker)

    # This is intentionally the first read of outcome cohort bytes in this path.
    outcome_locator = receipt["sealed_outcomes"]["cohort_locator"]
    outcome_raw = _regular(root, outcome_locator, "sealed outcome cohort")
    if _sha256_bytes(outcome_raw) != receipt["sealed_outcomes"]["cohort_raw_sha256"]:
        raise PhaseOneOpeningError("sealed outcome cohort does not match authority")
    result = evaluation.evaluate_phase_one(
        snapshot_locator=snapshot_locator,
        parity_locator=parity_locator,
        outcome_cohort_raw=outcome_raw,
        outcome_cohort_locator=outcome_locator,
        opening_authority_binding={
            "authority_id": receipt["authority_id"],
            "authority_raw_sha256": loaded["receipt_raw_sha256"],
            "opening_marker_locator": one_time["opening_marker_locator"],
        },
        run_id=one_time["run_id"],
        root=root,
        clock=clock,
    )
    evaluation.write_no_clobber(output_path, result)
    return result


__all__ = [
    "AUTHORITY_LOCATOR",
    "EXTERNAL_SHA256_ENV",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "PhaseOneOpeningError",
    "current_expected_bindings",
    "load_pinned_opening_authority",
    "run_authorized_phase_one_evaluation",
    "validate_opening_authority",
]
