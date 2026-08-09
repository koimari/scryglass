"""Externally pinned, one-time opening gate for outcome-free phase two."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from lol_kills import bookmaker_quote_capture

from . import calibration_uncertainty_registry_v1 as calibration_registry
from . import phase_one_evaluation_registry_v1 as evaluation_registry
from .betano_br_quote_adapter_registry_v1 import (
    DEFAULT_REGISTRY as ADAPTER_REGISTRY_LOCATOR,
    load_registered_betano_quote_adapter_v1,
)
from .betano_terms_authority_v1 import (
    REGISTRY_LOCATOR as TERMS_REGISTRY_LOCATOR,
    load_pinned_betano_terms_authority_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_opening_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-independent-opening-authority:v1"
MARKER_SCHEMA_VERSION = "scryglass:phase-two-opening-consumption-marker:v1"
AUTHORITY_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/"
    "phase-two-opening-authority.json"
)
MARKER_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
    "opening-marker-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PRIVATE_MATCH_WINNER_PHASE_TWO_OPENING_SHA256"
EVENT_PROBABILITY_PREFIX = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/event-probabilities-v2"
)
EVENT_PLAN_PREFIX = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/event-plans-v1"
)
QUOTE_FAILURE_PREFIX = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/quote-failures-v1"
)
ATTEMPT_COMPLETION_PREFIX = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/attempt-completions-v1"
)
STOPPING_SNAPSHOT_PREFIX = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/stopping-snapshots-v1"
)
STOPPING_SNAPSHOT_REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/stopping-snapshot-registry-v1.json"
)
BETANO_QUOTE_PREFIX = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/betano-quotes-v2"
)
QUALIFIED_BETANO_QUOTE_PREFIX = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/qualified-betano-quotes-v1"
)
BETANO_QUOTE_REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/betano-quote-registry.json"
)
REVIEW_SCOPES = {
    "MODEL_PIPELINE": {
        "reviewer_independent_of_model_authors_candidate_selectors_and_phase_one_outcome_custodians": True,
        "phase_one_pass_recalibration_uncertainty_and_fast_parity_registries_verified": True,
        "candidates_hyperparameters_thresholds_and_seed_schedule_unchanged": True,
        "verification_event_excluded_from_phase_two": True,
        "event_probability_and_uncertainty_sources_frozen_before_opening": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "MARKET_PIPELINE": {
        "reviewer_independent_of_quote_adapter_terms_capture_and_scryglass_code_authors": True,
        "complete_betano_terms_and_settlement_alignment_registry_verified": True,
        "source_specific_quote_adapter_registry_verified": True,
        "prediction_before_quote_transport_and_freshness_contract_verified": True,
        "outcomes_absent_and_phase_two_artifact_directories_empty": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
AUTHORITY = {
    "outcome_free_phase_two_collection_authority": True,
    "phase_two_outcome_opening_authority": False,
    "event_probability_identity_authority": False,
    "probability_authority": False,
    "quote_identity_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "One-time authority to begin the registered outcome-free phase-two shadow "
    "collection only. It does not open phase-two outcomes or authorize a model "
    "probability, quote identity, expected value, recommendation, transaction, "
    "stake, or wager."
)


class PhaseTwoOpeningError(RuntimeError):
    """The independent opening authority, dependency, or one-time marker failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PhaseTwoOpeningError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PhaseTwoOpeningError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoOpeningError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoOpeningError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PhaseTwoOpeningError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseTwoOpeningError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise PhaseTwoOpeningError(f"{label} must be an object")
    return value


def _regular(root: Path, locator: Path, label: str) -> bytes:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoOpeningError(f"{label} is unavailable")
    return path.read_bytes()


def _load_evaluation(root: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    raw = _regular(root, evaluation_registry.REGISTRY_LOCATOR, "evaluation registry")
    registry = _object(raw, "evaluation registry")
    result_locator = (registry.get("result_binding") or {}).get("result_locator")
    digest = environment.get(evaluation_registry.EXTERNAL_SHA256_ENV)
    if not isinstance(result_locator, str) or not digest:
        raise PhaseTwoOpeningError("evaluation registry pin or result is missing")
    binding = evaluation_registry.expected_result_binding(
        result_locator=result_locator, root=root
    )
    loaded = evaluation_registry.load_pinned_evaluation_registry(
        path=root / evaluation_registry.REGISTRY_LOCATOR,
        external_sha256=digest,
        expected_binding=binding,
    )
    if loaded.get("phase_one_models_independently_passed") is not True:
        raise PhaseTwoOpeningError("phase-one models did not independently pass")
    return {
        "locator": evaluation_registry.REGISTRY_LOCATOR.as_posix(),
        "raw_sha256": loaded["receipt_raw_sha256"],
        "registry_id": loaded["receipt"]["registry_id"],
        "phase_one_models_independently_passed": True,
    }


def _load_calibration(root: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    raw = _regular(root, calibration_registry.REGISTRY_LOCATOR, "calibration registry")
    registry = _object(raw, "calibration registry")
    binding = registry.get("binding") or {}
    recalibration_locator = (binding.get("recalibration") or {}).get("locator")
    slow_locator = (binding.get("uncertainty_verification") or {}).get("locator")
    fast_locator = (binding.get("fast_uncertainty_verification") or {}).get("locator")
    digest = environment.get(calibration_registry.EXTERNAL_SHA256_ENV)
    if not all(isinstance(item, str) for item in (recalibration_locator, slow_locator, fast_locator)) or not digest:
        raise PhaseTwoOpeningError("calibration registry pin or binding is missing")
    expected = calibration_registry.expected_registration_binding(
        recalibration_artifact_locator=recalibration_locator,
        verification_uncertainty_locator=slow_locator,
        verification_fast_uncertainty_locator=fast_locator,
        root=root,
        environment=environment,
    )
    loaded = calibration_registry.load_pinned_calibration_uncertainty_registry(
        path=root / calibration_registry.REGISTRY_LOCATOR,
        external_sha256=digest,
        expected_binding=expected,
    )
    return {
        "locator": calibration_registry.REGISTRY_LOCATOR.as_posix(),
        "raw_sha256": loaded["receipt_raw_sha256"],
        "registry_id": loaded["receipt"]["registry_id"],
        "recalibration_independently_registered": True,
        "uncertainty_and_fast_parity_independently_registered": True,
    }


def current_expected_bindings(
    *, root: Path = ROOT, environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    evaluation_binding = _load_evaluation(root, environment)
    calibration_binding = _load_calibration(root, environment)
    terms_digest = environment.get(
        "SCRYGLASS_PRIVATE_MATCH_WINNER_BOOKMAKER_TERMS_SHA256"
    )
    adapter_digest = environment.get(
        "SCRYGLASS_PRIVATE_MATCH_WINNER_QUOTE_ADAPTER_SHA256"
    )
    if not terms_digest or not adapter_digest:
        raise PhaseTwoOpeningError("terms or adapter external pin is missing")
    terms = load_pinned_betano_terms_authority_v1(
        path=root / TERMS_REGISTRY_LOCATOR,
        external_sha256=terms_digest,
        root=root,
    )
    adapter = load_registered_betano_quote_adapter_v1(
        expected_registry_sha256=adapter_digest,
        root=root,
    )
    try:
        from .phase_two_collection_readiness_registry_v1 import (
            EXTERNAL_SHA256_ENV as READINESS_EXTERNAL_SHA256_ENV,
            REGISTRY_LOCATOR as READINESS_REGISTRY_LOCATOR,
            expected_readiness_binding,
            load_pinned_phase_two_collection_readiness_registry_v1,
        )
        readiness_digest = environment.get(READINESS_EXTERNAL_SHA256_ENV)
        if not readiness_digest:
            raise PhaseTwoOpeningError(
                "phase-two collection readiness registry pin is missing"
            )
        readiness_binding = expected_readiness_binding(
            root=root, environment=environment
        )
        readiness = load_pinned_phase_two_collection_readiness_registry_v1(
            path=root / READINESS_REGISTRY_LOCATOR,
            external_sha256=readiness_digest,
            expected_binding=readiness_binding,
        )
    except Exception as exc:
        raise PhaseTwoOpeningError(
            "phase-two collection implementation readiness is invalid"
        ) from exc
    return {
        "phase_one_evaluation": evaluation_binding,
        "calibration_uncertainty": calibration_binding,
        "bookmaker_terms": {
            "locator": TERMS_REGISTRY_LOCATOR.as_posix(),
            "raw_sha256": terms["receipt_raw_sha256"],
            "registry_id": terms["receipt"]["registry_id"],
            "settlement_contract_resolved": True,
        },
        "quote_adapter": {
            "locator": ADAPTER_REGISTRY_LOCATOR.as_posix(),
            "registry_sha256": adapter["registry_sha256"],
            "registry_id": adapter["registry_id"],
            "source_adapter_identity_authority": True,
        },
        "phase_two_collection_readiness": {
            "registry_locator": READINESS_REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": readiness["receipt_raw_sha256"],
            "registry_id": readiness["receipt"]["registry_id"],
            "readiness_binding": readiness_binding,
        },
    }


def validate_opening_authority(
    payload: Mapping[str, Any], *, expected_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoOpeningError("phase-two authority must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "authority_id",
        "status",
        "issued_at_utc",
        "reviews",
        "bindings",
        "one_time",
        "decision",
        "authority",
        "claim_ceiling",
    }:
        raise PhaseTwoOpeningError("phase-two authority fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "OUTCOME_FREE_PHASE_TWO_COLLECTION_APPROVED"
    ):
        raise PhaseTwoOpeningError("phase-two authority identity changed")
    _nonempty(value.get("authority_id"), "authority_id")
    issued = _timestamp(value.get("issued_at_utc"), "issued_at_utc")
    if value.get("bindings") != dict(expected_bindings):
        raise PhaseTwoOpeningError("phase-two authority bindings changed")
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise PhaseTwoOpeningError("two phase-two opening reviews are required")
    reviewers: set[str] = set()
    scopes: set[str] = set()
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope",
            "reviewer_id",
            "reviewed_at_utc",
            "attestation",
        }:
            raise PhaseTwoOpeningError("phase-two review structure changed")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        reviewed = _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc")
        if (
            scope not in REVIEW_SCOPES
            or review.get("attestation") != REVIEW_SCOPES[scope]
            or reviewed > issued
        ):
            raise PhaseTwoOpeningError("phase-two review is incomplete")
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != set(REVIEW_SCOPES):
        raise PhaseTwoOpeningError("phase-two reviews are not independent")
    if value.get("one_time") != {
        "opening_marker_locator": MARKER_LOCATOR.as_posix(),
        "marker_written_before_first_phase_two_probability_or_quote": True,
        "second_opening_or_marker_replacement_prohibited": True,
        "crash_after_marker_does_not_authorize_reopening": True,
    }:
        raise PhaseTwoOpeningError("phase-two one-time contract changed")
    if value.get("decision") != {
        "outcome_free_phase_two_collection_approved": True,
        "phase_two_outcomes_opened": False,
        "probability_authorized": False,
        "quote_authorized": False,
        "betting_authorized": False,
    }:
        raise PhaseTwoOpeningError("phase-two decision changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoOpeningError("phase-two authority exceeds its scope")
    return value


def load_pinned_opening_authority(
    *,
    root: Path,
    external_sha256: str,
    expected_bindings: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    digest = _sha(external_sha256, "external opening authority digest")
    raw = _regular(root, AUTHORITY_LOCATOR, "phase-two opening authority")
    if _sha256(raw) != digest:
        raise PhaseTwoOpeningError("opening authority does not match its external pin")
    return raw, validate_opening_authority(
        _object(raw, "phase-two opening authority"),
        expected_bindings=expected_bindings,
    )


def _artifact_count(root: Path, prefix: Path) -> int:
    path = root / prefix
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_dir():
        raise PhaseTwoOpeningError("phase-two artifact directory is aliased")
    return sum(1 for item in path.rglob("*.json") if item.is_file())


def _marker_payload(
    *, raw_authority: bytes, authority: Mapping[str, Any], opened_at: datetime
) -> dict[str, Any]:
    return {
        "schema_version": MARKER_SCHEMA_VERSION,
        "status": "PHASE_TWO_OPENING_CONSUMED_COLLECTION_ACTIVE",
        "opened_at_utc": opened_at.isoformat(),
        "authority_id": authority["authority_id"],
        "authority_raw_sha256": _sha256(raw_authority),
        "authority_bindings": authority["bindings"],
        "phase_two_event_probabilities_at_opening": 0,
        "phase_two_quotes_at_opening": 0,
        "outcomes_present": False,
        "outcomes_accessed": False,
        "replacement_or_second_opening_permitted": False,
        "probability_authorized": False,
        "betting_authorized": False,
    }


def validate_opening_marker(
    payload: Mapping[str, Any],
    *,
    raw_authority: bytes,
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    expected = _marker_payload(
        raw_authority=raw_authority,
        authority=authority,
        opened_at=_timestamp(payload.get("opened_at_utc"), "opened_at_utc"),
    )
    if dict(payload) != expected:
        raise PhaseTwoOpeningError("phase-two opening marker changed")
    if _timestamp(payload["opened_at_utc"], "opened_at_utc") < _timestamp(
        authority["issued_at_utc"], "authority.issued_at_utc"
    ):
        raise PhaseTwoOpeningError("phase-two marker predates authority")
    return dict(payload)


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoOpeningError(f"refusing to replace phase-two marker: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseTwoOpeningError(
                f"refusing to replace phase-two marker: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


def consume_phase_two_opening(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    digest = environment.get(EXTERNAL_SHA256_ENV)
    if not digest:
        raise PhaseTwoOpeningError("external opening authority digest is missing")
    expected = current_expected_bindings(root=root, environment=environment)
    raw, authority = load_pinned_opening_authority(
        root=root, external_sha256=digest, expected_bindings=expected
    )
    marker_path = root / MARKER_LOCATOR
    if marker_path.exists() or marker_path.is_symlink():
        raise PhaseTwoOpeningError("phase two was already opened or attempted")
    if (
        _artifact_count(root, EVENT_PROBABILITY_PREFIX) != 0
        or _artifact_count(root, EVENT_PLAN_PREFIX) != 0
        or _artifact_count(root, QUOTE_FAILURE_PREFIX) != 0
        or _artifact_count(root, ATTEMPT_COMPLETION_PREFIX) != 0
        or _artifact_count(root, STOPPING_SNAPSHOT_PREFIX) != 0
        or (root / STOPPING_SNAPSHOT_REGISTRY_LOCATOR).exists()
        or (root / STOPPING_SNAPSHOT_REGISTRY_LOCATOR).is_symlink()
        or _artifact_count(root, BETANO_QUOTE_PREFIX) != 0
        or _artifact_count(root, QUALIFIED_BETANO_QUOTE_PREFIX) != 0
        or _artifact_count(root, Path(bookmaker_quote_capture.RECEIPT_PREFIX)) != 0
        or (root / BETANO_QUOTE_REGISTRY_LOCATOR).exists()
        or (root / BETANO_QUOTE_REGISTRY_LOCATOR).is_symlink()
    ):
        raise PhaseTwoOpeningError("phase-two artifacts predate opening")
    opened_at = clock()
    if not isinstance(opened_at, datetime) or opened_at.tzinfo is None:
        raise PhaseTwoOpeningError("opening clock must be timezone-aware")
    opened_at = opened_at.astimezone(timezone.utc)
    marker = _marker_payload(
        raw_authority=raw, authority=authority, opened_at=opened_at
    )
    validate_opening_marker(marker, raw_authority=raw, authority=authority)
    write_no_clobber(marker_path, marker)
    return marker


def validate_active_phase_two_opening(
    *, root: Path = ROOT, environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    digest = environment.get(EXTERNAL_SHA256_ENV)
    if not digest:
        raise PhaseTwoOpeningError("external opening authority digest is missing")
    expected = current_expected_bindings(root=root, environment=environment)
    raw, authority = load_pinned_opening_authority(
        root=root, external_sha256=digest, expected_bindings=expected
    )
    marker_raw = _regular(root, MARKER_LOCATOR, "phase-two opening marker")
    marker = validate_opening_marker(
        _object(marker_raw, "phase-two opening marker"),
        raw_authority=raw,
        authority=authority,
    )
    return {
        "authority": authority,
        "authority_raw_sha256": _sha256(raw),
        "marker": marker,
        "marker_raw_sha256": _sha256(marker_raw),
        "outcome_free_phase_two_collection_active": True,
        "probability_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "AUTHORITY_LOCATOR",
    "EXTERNAL_SHA256_ENV",
    "MARKER_LOCATOR",
    "REVIEW_SCOPES",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "PhaseTwoOpeningError",
    "consume_phase_two_opening",
    "current_expected_bindings",
    "load_pinned_opening_authority",
    "validate_active_phase_two_opening",
    "validate_opening_authority",
    "validate_opening_marker",
]
