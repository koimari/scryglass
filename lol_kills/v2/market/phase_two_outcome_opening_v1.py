"""Independent one-time authority for opening sealed phase-two outcomes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from . import phase_one_evaluation_v1 as phase_one
from . import phase_two_evaluation_v1 as evaluation
from . import phase_two_stopping_snapshot_registry_v1 as snapshot_registry


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_outcome_opening_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-independent-outcome-opening-authority:v1"
AUTHORITY_LOCATOR = Path(
    "data/lol/private_market_authority/phase-two-outcome-opening-authority-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PHASE_TWO_OUTCOME_OPENING_AUTHORITY_SHA256"
OPENING_MARKER_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
    "outcome-opening-markers-v1"
)
MARKER_SCHEMA_VERSION = "scryglass:phase-two-outcome-opening-marker:v1"
REVIEW_SCOPES = {
    "MODEL_EVALUATION": {
        "reviewer_not_model_author_candidate_selector_evaluator_author_or_outcome_custodian": True,
        "frozen_probability_and_rating_only_comparator_bindings_verified": True,
        "bootstrap_seed_replicates_metrics_calibration_and_strata_verified": True,
        "outcomes_not_accessed_before_approval": True,
        "no_post_opening_tuning_or_threshold_change_approved": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "MARKET_CAPTURE": {
        "reviewer_not_quote_adapter_capture_evaluator_author_or_outcome_custodian": True,
        "first_support_met_snapshot_and_full_denominator_independently_verified": True,
        "quote_timing_extraction_binding_shadow_and_settlement_rules_verified": True,
        "outcomes_not_accessed_before_approval": True,
        "no_success_only_selection_or_post_outcome_exclusion_approved": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
AUTHORITY = {
    "one_time_phase_two_outcome_opening_authority": True,
    "phase_two_evaluation_authority": True,
    "probability_authority": False,
    "fair_odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "One-time opening of one exact sealed phase-two outcome cohort for the "
    "frozen evaluation only. The result still requires independent registration "
    "and separate market authority; no probability, EV, stake, transaction, or "
    "betting authority is granted."
)


class PhaseTwoOutcomeOpeningError(RuntimeError):
    """The outcome-opening authority is absent, stale, malformed, or consumed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in "0123456789abcdef" for ch in value)
    ):
        raise PhaseTwoOutcomeOpeningError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseTwoOutcomeOpeningError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoOutcomeOpeningError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoOutcomeOpeningError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseTwoOutcomeOpeningError(f"duplicate JSON key: {key}")
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
        raise PhaseTwoOutcomeOpeningError(
            f"{label} is not strict UTF-8 JSON"
        ) from exc
    if not isinstance(value, dict):
        raise PhaseTwoOutcomeOpeningError(f"{label} must contain an object")
    return value


def _locator(value: Any, prefix: PurePosixPath, field: str) -> str:
    try:
        return phase_one._locator(value, prefix, field)
    except Exception as exc:
        raise PhaseTwoOutcomeOpeningError(
            f"{field} is outside its registered root"
        ) from exc


def _regular(root: Path, locator: str, label: str) -> bytes:
    try:
        return phase_one._read_regular(root, locator, label)
    except Exception as exc:
        raise PhaseTwoOutcomeOpeningError(
            f"{label} is not an unaliased regular file"
        ) from exc


def _source_sha(root: Path, locator: str) -> str:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoOutcomeOpeningError(f"source is unavailable: {locator}")
    return phase_one._sha256_path(path)


def current_expected_bindings(
    *, snapshot_locator: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Resolve outcome-free bindings without reading any outcome bytes."""

    expected_snapshot = snapshot_registry.expected_snapshot_binding(
        snapshot_locator=snapshot_locator, root=root, environment=environment
    )
    snapshot_digest = environment.get(snapshot_registry.EXTERNAL_SHA256_ENV)
    if not snapshot_digest:
        raise PhaseTwoOutcomeOpeningError(
            "external stopping-snapshot registry digest is missing"
        )
    registered_snapshot = snapshot_registry.load_pinned_phase_two_snapshot_registry_v1(
        path=root / snapshot_registry.REGISTRY_LOCATOR,
        external_sha256=snapshot_digest,
        expected_binding=expected_snapshot,
    )
    try:
        from .phase_two_evaluation_readiness_registry_v1 import (
            EXTERNAL_SHA256_ENV as READINESS_EXTERNAL_SHA256_ENV,
            REGISTRY_LOCATOR as READINESS_REGISTRY_LOCATOR,
            expected_readiness_binding,
            load_pinned_phase_two_evaluation_readiness_registry_v1,
        )
        readiness_digest = environment.get(READINESS_EXTERNAL_SHA256_ENV)
        if not readiness_digest:
            raise PhaseTwoOutcomeOpeningError(
                "phase-two evaluation readiness registry pin is missing"
            )
        readiness_binding = expected_readiness_binding(
            root=root, environment=environment
        )
        readiness = load_pinned_phase_two_evaluation_readiness_registry_v1(
            path=root / READINESS_REGISTRY_LOCATOR,
            external_sha256=readiness_digest,
            expected_binding=readiness_binding,
        )
    except Exception as exc:
        raise PhaseTwoOutcomeOpeningError(
            "phase-two evaluation readiness is unavailable"
        ) from exc
    return {
        "first_support_met_snapshot_registry": {
            "locator": snapshot_registry.REGISTRY_LOCATOR.as_posix(),
            "raw_sha256": registered_snapshot["receipt_raw_sha256"],
            "registry_id": registered_snapshot["receipt"]["registry_id"],
            "snapshot_binding": expected_snapshot,
        },
        "phase_two_evaluation_readiness": {
            "registry_locator": READINESS_REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": readiness["receipt_raw_sha256"],
            "registry_id": readiness["receipt"]["registry_id"],
            "readiness_binding": readiness_binding,
        },
        "sources": {
            "evaluator": {
                "locator": evaluation.SOURCE_LOCATOR,
                "raw_sha256": _source_sha(root, evaluation.SOURCE_LOCATOR),
            },
            "outcome_opening": {
                "locator": SOURCE_LOCATOR,
                "raw_sha256": _source_sha(root, SOURCE_LOCATOR),
            },
        },
    }


def validate_outcome_opening_authority_v1(
    payload: Mapping[str, Any], *, expected_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoOutcomeOpeningError("opening authority must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "authority_id", "status", "scope",
        "reviewed_at_utc", "reviews", "bindings", "sealed_outcomes",
        "one_time_run", "authority", "claim_ceiling",
    }:
        raise PhaseTwoOutcomeOpeningError("opening authority fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "APPROVED"
        or value.get("scope") != "ONE_TIME_PHASE_TWO_MARKET_EVALUATION_ONLY"
    ):
        raise PhaseTwoOutcomeOpeningError("opening authority identity changed")
    _nonempty(value.get("authority_id"), "authority_id")
    reviewed_at = _timestamp(value.get("reviewed_at_utc"), "reviewed_at_utc")
    if value.get("bindings") != dict(expected_bindings):
        raise PhaseTwoOutcomeOpeningError("opening authority bindings changed")

    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise PhaseTwoOutcomeOpeningError("exactly two independent reviews are required")
    reviewers: set[str] = set()
    scopes: set[str] = set()
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope", "reviewer_id", "reviewed_at_utc", "attestation"
        }:
            raise PhaseTwoOutcomeOpeningError("opening review structure changed")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        if (
            scope not in REVIEW_SCOPES
            or review.get("attestation") != REVIEW_SCOPES[scope]
            or _timestamp(review.get("reviewed_at_utc"), "review.reviewed_at")
            > reviewed_at
        ):
            raise PhaseTwoOutcomeOpeningError("opening review is incomplete")
        scopes.add(scope)
        reviewers.add(reviewer)
    if scopes != set(REVIEW_SCOPES) or len(reviewers) != 2:
        raise PhaseTwoOutcomeOpeningError("opening reviews are not independent")

    sealed = value.get("sealed_outcomes")
    if not isinstance(sealed, Mapping) or set(sealed) != {
        "cohort_locator", "cohort_raw_sha256", "custodian_id",
        "sealed_at_utc", "custodian_attestation",
    }:
        raise PhaseTwoOutcomeOpeningError("sealed outcome binding is incomplete")
    _locator(sealed.get("cohort_locator"), evaluation.OUTCOME_PREFIX, "cohort_locator")
    _sha(sealed.get("cohort_raw_sha256"), "cohort_raw_sha256")
    custodian = _nonempty(sealed.get("custodian_id"), "custodian_id")
    if custodian in reviewers:
        raise PhaseTwoOutcomeOpeningError("outcome custodian cannot be a reviewer")
    if _timestamp(sealed.get("sealed_at_utc"), "sealed_at_utc") > reviewed_at:
        raise PhaseTwoOutcomeOpeningError("review predates the sealed outcome cohort")
    if sealed.get("custodian_attestation") != {
        "digest_created_without_disclosing_outcomes_to_model_or_capture_authors_or_reviewers": True,
        "cohort_bytes_immutable_after_digest": True,
        "cohort_exactly_matches_every_map_in_the_registered_snapshot": True,
        "no_manual_post_outcome_exclusion_or_replacement": True,
    }:
        raise PhaseTwoOutcomeOpeningError("outcome custodian attestation is incomplete")

    run = value.get("one_time_run")
    if not isinstance(run, Mapping) or set(run) != {
        "run_id", "opening_marker_locator", "authorized_output_locator",
        "marker_written_before_outcome_read", "atomic_no_clobber_output_required",
        "partial_result_publication_prohibited",
        "second_opening_or_replacement_cohort_prohibited",
    }:
        raise PhaseTwoOutcomeOpeningError("one-time run contract is incomplete")
    _nonempty(run.get("run_id"), "run_id")
    _locator(
        run.get("opening_marker_locator"), OPENING_MARKER_PREFIX,
        "opening_marker_locator",
    )
    _locator(
        run.get("authorized_output_locator"), evaluation.RESULT_PREFIX,
        "authorized_output_locator",
    )
    if any(
        run.get(field) is not True
        for field in (
            "marker_written_before_outcome_read",
            "atomic_no_clobber_output_required",
            "partial_result_publication_prohibited",
            "second_opening_or_replacement_cohort_prohibited",
        )
    ):
        raise PhaseTwoOutcomeOpeningError("one-time protections changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoOutcomeOpeningError("opening authority exceeds its scope")
    return value


def load_pinned_outcome_opening_authority_v1(
    *, path: Path, external_sha256: str,
    expected_bindings: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    digest = _sha(external_sha256, "external opening-authority digest")
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoOutcomeOpeningError("outcome-opening authority is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise PhaseTwoOutcomeOpeningError(
            "outcome-opening authority does not match its external pin"
        )
    return raw, validate_outcome_opening_authority_v1(
        _object(raw, "outcome-opening authority"),
        expected_bindings=expected_bindings,
    )


def _count_json(root: Path, prefix: PurePosixPath) -> int:
    path = root / prefix
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_dir():
        raise PhaseTwoOutcomeOpeningError("outcome-opening directory is aliased")
    return sum(1 for item in path.rglob("*.json") if item.is_file())


def run_authorized_phase_two_evaluation_v1(
    *, snapshot_locator: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Mark one opening as consumed, then read and evaluate outcomes exactly once."""

    external = environment.get(EXTERNAL_SHA256_ENV)
    if not external:
        raise PhaseTwoOutcomeOpeningError(
            "external outcome-opening authority digest is missing"
        )
    expected = current_expected_bindings(
        snapshot_locator=snapshot_locator, root=root, environment=environment
    )
    authority_raw, authority = load_pinned_outcome_opening_authority_v1(
        path=root / AUTHORITY_LOCATOR,
        external_sha256=external,
        expected_bindings=expected,
    )
    run = authority["one_time_run"]
    marker_path = root / run["opening_marker_locator"]
    output_path = root / run["authorized_output_locator"]
    if (
        _count_json(root, OPENING_MARKER_PREFIX) != 0
        or _count_json(root, evaluation.RESULT_PREFIX) != 0
        or marker_path.exists() or marker_path.is_symlink()
        or output_path.exists() or output_path.is_symlink()
    ):
        raise PhaseTwoOutcomeOpeningError(
            "phase-two outcome opening was already started or consumed"
        )
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PhaseTwoOutcomeOpeningError("opening clock must be timezone-aware")
    observed = observed.astimezone(timezone.utc)
    if observed < _timestamp(authority["reviewed_at_utc"], "reviewed_at_utc"):
        raise PhaseTwoOutcomeOpeningError("opening predates its authority")
    marker = {
        "schema_version": MARKER_SCHEMA_VERSION,
        "result_state": "ONE_TIME_PHASE_TWO_OUTCOME_OPENING_STARTED_OUTCOMES_NOT_YET_READ",
        "started_at_utc": observed.isoformat(),
        "authority_id": authority["authority_id"],
        "authority_raw_sha256": _sha256(authority_raw),
        "run_id": run["run_id"],
        "snapshot_artifact_sha256": expected[
            "first_support_met_snapshot_registry"
        ]["snapshot_binding"]["snapshot_artifact_sha256"],
        "outcome_cohort_locator": authority["sealed_outcomes"]["cohort_locator"],
        "outcome_cohort_expected_raw_sha256": authority["sealed_outcomes"][
            "cohort_raw_sha256"
        ],
        "second_opening_or_replacement_cohort_prohibited": True,
        "probability_or_betting_authority": False,
    }
    evaluation.write_no_clobber(marker_path, marker)

    # Intentionally the first outcome-cohort read in this execution path.
    outcome_locator = authority["sealed_outcomes"]["cohort_locator"]
    outcome_raw = _regular(root, outcome_locator, "sealed phase-two outcome cohort")
    if _sha256(outcome_raw) != authority["sealed_outcomes"]["cohort_raw_sha256"]:
        raise PhaseTwoOutcomeOpeningError(
            "sealed outcome cohort does not match the opening authority"
        )
    result = evaluation.evaluate_phase_two_v1(
        snapshot_locator=snapshot_locator,
        outcome_cohort_raw=outcome_raw,
        outcome_cohort_locator=outcome_locator,
        opening_authority_binding={
            "authority_id": authority["authority_id"],
            "authority_raw_sha256": _sha256(authority_raw),
            "opening_marker_locator": run["opening_marker_locator"],
        },
        run_id=run["run_id"],
        root=root,
        environment=environment,
        clock=clock,
    )
    evaluation.write_no_clobber(output_path, result)
    return result


__all__ = [
    "AUTHORITY_LOCATOR", "EXTERNAL_SHA256_ENV", "MARKER_SCHEMA_VERSION",
    "OPENING_MARKER_PREFIX", "REVIEW_SCOPES", "SCHEMA_VERSION",
    "SOURCE_LOCATOR", "PhaseTwoOutcomeOpeningError",
    "current_expected_bindings", "load_pinned_outcome_opening_authority_v1",
    "run_authorized_phase_two_evaluation_v1",
    "validate_outcome_opening_authority_v1",
]
