"""Final semantic authority for private match-winner decision support.

This authority is intentionally separate from phase-two evaluation.  It may be
issued only after the terminal phase-two result independently passes and every
production source is frozen.  It never authorizes transaction execution or a
stake size.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from . import calibration_uncertainty_registry_v1 as calibration_registry
from . import phase_one_evaluation_v1 as phase_one
from . import phase_two_evaluation_result_registry_v1 as evaluation_registry
from .betano_br_quote_adapter_registry_v1 import (
    DEFAULT_REGISTRY as ADAPTER_REGISTRY_LOCATOR,
    load_registered_betano_quote_adapter_v1,
)
from .betano_terms_authority_v1 import (
    EXTERNAL_SHA256_ENV as TERMS_EXTERNAL_SHA256_ENV,
    REGISTRY_LOCATOR as TERMS_REGISTRY_LOCATOR,
    load_pinned_betano_terms_authority_v1,
)
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/semantic_market_authority_v1.py"
SCHEMA_VERSION = "scryglass:semantic-match-winner-market-authority:v1"
AUTHORITY_LOCATOR = Path(
    "data/lol/private_market_authority/semantic-match-winner-market-authority-v1.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_SEMANTIC_MATCH_WINNER_MARKET_AUTHORITY_SHA256"
PRODUCTION_SOURCE_LOCATORS = (
    "lol_kills/v2/market/production_event_probability_v1.py",
    "lol_kills/v2/market/production_betano_quote_v1.py",
    "lol_kills/v2/market/semantic_match_winner_decision_v1.py",
)
REVIEW_SCOPES = {
    "MODEL_DEPLOYMENT": {
        "reviewer_independent_of_model_candidate_evaluator_outcome_and_production_code_authors": True,
        "terminal_phase_two_pass_and_exact_replay_registry_verified": True,
        "recalibration_uncertainty_and_production_probability_sources_verified": True,
        "no_post_outcome_model_threshold_or_interval_change_found": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "MARKET_DEPLOYMENT": {
        "reviewer_independent_of_quote_adapter_terms_capture_evaluator_and_production_code_authors": True,
        "complete_terms_adapter_transport_freshness_and_decision_sources_verified": True,
        "decision_authority_excludes_transaction_execution_and_stake_size": True,
        "no_post_outcome_quote_or_selection_rule_change_found": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
AUTHORITY = {
    "private_probability_generation_authority": True,
    "private_fair_odds_authority": True,
    "private_expected_value_authority": True,
    "private_recommendation_authority": True,
    "public_probability_authority": False,
    "transaction_authority": False,
    "stake_authority": False,
    "betting_account_authority": False,
}
CLAIM_CEILING = (
    "Private match-winner calculation and BET/PASS decision support only after "
    "all bound live receipts validate. It does not place a wager, choose a stake, "
    "claim executable price or limit, or authorize public betting content."
)


class SemanticMarketAuthorityError(RuntimeError):
    """A terminal pass, deployment dependency, review, or pin failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise SemanticMarketAuthorityError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SemanticMarketAuthorityError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise SemanticMarketAuthorityError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SemanticMarketAuthorityError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SemanticMarketAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SemanticMarketAuthorityError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise SemanticMarketAuthorityError(f"{label} must contain an object")
    return value


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise SemanticMarketAuthorityError(f"production source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": phase_one._sha256_path(path),
    }


def _phase_two_result(root: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    path = root / evaluation_registry.REGISTRY_LOCATOR
    digest = environment.get(evaluation_registry.EXTERNAL_SHA256_ENV)
    if path.is_symlink() or not path.is_file() or not digest:
        raise SemanticMarketAuthorityError("phase-two evaluation registry unavailable")
    registry = _object(path.read_bytes(), "phase-two evaluation registry")
    result_locator = (registry.get("result_binding") or {}).get("result_locator")
    if not isinstance(result_locator, str):
        raise SemanticMarketAuthorityError("phase-two result locator missing")
    try:
        binding = evaluation_registry.expected_result_binding(
            result_locator=result_locator, root=root, environment=environment
        )
        loaded = evaluation_registry.load_pinned_phase_two_evaluation_registry_v1(
            path=path, external_sha256=digest, expected_binding=binding
        )
    except Exception as exc:
        raise SemanticMarketAuthorityError("phase-two evaluation registry invalid") from exc
    if loaded["phase_two_market_gates_independently_passed"] is not True:
        raise SemanticMarketAuthorityError("phase-two market gates did not pass")
    return {
        "registry_locator": evaluation_registry.REGISTRY_LOCATOR.as_posix(),
        "registry_raw_sha256": loaded["receipt_raw_sha256"],
        "registry_id": loaded["receipt"]["registry_id"],
        "result_binding": binding,
        "market_gates_independently_passed": True,
        "reviewer_ids": sorted(
            review["reviewer_id"] for review in loaded["receipt"]["reviews"]
        ),
    }


def _calibration(root: Path, environment: Mapping[str, str]) -> dict[str, Any]:
    path = root / calibration_registry.REGISTRY_LOCATOR
    digest = environment.get(calibration_registry.EXTERNAL_SHA256_ENV)
    if path.is_symlink() or not path.is_file() or not digest:
        raise SemanticMarketAuthorityError("calibration registry unavailable")
    registry = _object(path.read_bytes(), "calibration registry")
    binding = registry.get("binding") or {}
    try:
        expected = calibration_registry.expected_registration_binding(
            recalibration_artifact_locator=binding["recalibration"]["locator"],
            verification_uncertainty_locator=binding["uncertainty_verification"]["locator"],
            verification_fast_uncertainty_locator=binding["fast_uncertainty_verification"]["locator"],
            root=root,
            environment=environment,
        )
        loaded = calibration_registry.load_pinned_calibration_uncertainty_registry(
            path=path, external_sha256=digest, expected_binding=expected
        )
    except Exception as exc:
        raise SemanticMarketAuthorityError("calibration registry invalid") from exc
    return {
        "registry_locator": calibration_registry.REGISTRY_LOCATOR.as_posix(),
        "registry_raw_sha256": loaded["receipt_raw_sha256"],
        "registry_id": loaded["receipt"]["registry_id"],
        "recalibration_artifact_sha256": expected["recalibration"]["artifact_sha256"],
        "uncertainty_verification_artifact_sha256": expected[
            "uncertainty_verification"
        ]["artifact_sha256"],
        "reviewer_ids": sorted(
            review["reviewer_id"] for review in loaded["receipt"]["reviews"]
        ),
    }


def current_expected_bindings(
    *, root: Path = ROOT, environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    if protocol["artifact_sha256"] != REGISTERED_PROTOCOL_ARTIFACT_SHA256:
        raise SemanticMarketAuthorityError("registered protocol changed")
    result = _phase_two_result(root, environment)
    calibration = _calibration(root, environment)
    terms_digest = environment.get(TERMS_EXTERNAL_SHA256_ENV)
    adapter_digest = environment.get(
        "SCRYGLASS_PRIVATE_MATCH_WINNER_QUOTE_ADAPTER_SHA256"
    )
    if not terms_digest or not adapter_digest:
        raise SemanticMarketAuthorityError("terms or adapter registry pin missing")
    try:
        terms = load_pinned_betano_terms_authority_v1(
            path=root / TERMS_REGISTRY_LOCATOR,
            external_sha256=terms_digest,
            root=root,
        )
        adapter = load_registered_betano_quote_adapter_v1(
            expected_registry_sha256=adapter_digest, root=root
        )
    except Exception as exc:
        raise SemanticMarketAuthorityError("terms or adapter registry invalid") from exc
    excluded_reviewers = set(result["reviewer_ids"]) | set(calibration["reviewer_ids"])
    excluded_reviewers.update(
        review["reviewer_id"] for review in terms["receipt"]["reviews"]
    )
    excluded_reviewers.add(adapter["independent_reviewer_id"])
    return {
        "protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        },
        "phase_two_evaluation": result,
        "calibration_uncertainty": calibration,
        "bookmaker_terms": {
            "registry_locator": TERMS_REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": terms["receipt_raw_sha256"],
            "registry_id": terms["receipt"]["registry_id"],
            "settlement_contract_resolved": True,
        },
        "quote_adapter": {
            "registry_locator": ADAPTER_REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": adapter["registry_sha256"],
            "registry_id": adapter["registry_id"],
            "source_adapter_identity_authority": True,
        },
        "production_sources": [
            _source_record(root, locator)
            for locator in (SOURCE_LOCATOR, *PRODUCTION_SOURCE_LOCATORS)
        ],
        "reviewer_ids_excluded_from_final_authority": sorted(excluded_reviewers),
    }


def validate_semantic_market_authority_v1(
    payload: Mapping[str, Any], *, expected_bindings: Mapping[str, Any]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SemanticMarketAuthorityError("market authority must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "authority_id", "status", "scope", "issued_at_utc",
        "valid_until_utc", "reviews", "bindings", "decision_policy",
        "authority", "claim_ceiling",
    }:
        raise SemanticMarketAuthorityError("market authority fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "APPROVED"
        or value.get("scope") != "PRIVATE_MATCH_WINNER_DECISION_SUPPORT_ONLY"
    ):
        raise SemanticMarketAuthorityError("market authority identity changed")
    _nonempty(value.get("authority_id"), "authority_id")
    issued = _timestamp(value.get("issued_at_utc"), "issued_at_utc")
    valid_until = _timestamp(value.get("valid_until_utc"), "valid_until_utc")
    if valid_until <= issued or (valid_until - issued).total_seconds() > 30 * 86400:
        raise SemanticMarketAuthorityError("authority validity window changed")
    if value.get("bindings") != dict(expected_bindings):
        raise SemanticMarketAuthorityError("market authority bindings changed")
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise SemanticMarketAuthorityError("two deployment reviews are required")
    excluded = set(expected_bindings["reviewer_ids_excluded_from_final_authority"])
    reviewers: set[str] = set()
    scopes: set[str] = set()
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope", "reviewer_id", "reviewed_at_utc", "attestation"
        }:
            raise SemanticMarketAuthorityError("deployment review structure changed")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        if (
            scope not in REVIEW_SCOPES
            or reviewer in excluded
            or review.get("attestation") != REVIEW_SCOPES[scope]
            or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > issued
        ):
            raise SemanticMarketAuthorityError("deployment review is not independent")
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != set(REVIEW_SCOPES):
        raise SemanticMarketAuthorityError("deployment reviews are not scope-complete")
    if value.get("decision_policy") != {
        "market_type": "match_winner",
        "no_vig_method": "two_way_normalized_implied_probability",
        "minimum_lower_bound_expected_return": 0.02,
        "maximum_probability_age_seconds": 60.0,
        "maximum_quote_age_seconds": 30.0,
        "positive_expected_return_haircut_fraction": 0.01,
        "flat_stake_or_bankroll_advice_permitted": False,
        "transaction_execution_permitted": False,
    }:
        raise SemanticMarketAuthorityError("decision policy changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise SemanticMarketAuthorityError("market authority exceeds scope")
    return value


def load_active_semantic_market_authority_v1(
    *, root: Path = ROOT, environment: Mapping[str, str] = os.environ,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    digest = environment.get(EXTERNAL_SHA256_ENV)
    if not digest:
        raise SemanticMarketAuthorityError("external market-authority pin missing")
    _sha(digest, "external market-authority pin")
    path = root / AUTHORITY_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise SemanticMarketAuthorityError("market authority unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise SemanticMarketAuthorityError("market authority external pin changed")
    expected = current_expected_bindings(root=root, environment=environment)
    receipt = validate_semantic_market_authority_v1(
        _object(raw, "semantic market authority"), expected_bindings=expected
    )
    observed = as_of or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise SemanticMarketAuthorityError("authority clock must be timezone-aware")
    observed = observed.astimezone(timezone.utc)
    if not (
        _timestamp(receipt["issued_at_utc"], "issued_at_utc")
        <= observed
        <= _timestamp(receipt["valid_until_utc"], "valid_until_utc")
    ):
        raise SemanticMarketAuthorityError("market authority is not currently valid")
    return {
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "bindings": expected,
        "private_probability_generation_authorized": True,
        "private_decision_support_authorized": True,
        "transaction_authorized": False,
        "stake_authorized": False,
    }


__all__ = [
    "AUTHORITY_LOCATOR", "EXTERNAL_SHA256_ENV", "PRODUCTION_SOURCE_LOCATORS",
    "REVIEW_SCOPES", "SCHEMA_VERSION", "SOURCE_LOCATOR",
    "SemanticMarketAuthorityError", "current_expected_bindings",
    "load_active_semantic_market_authority_v1",
    "validate_semantic_market_authority_v1",
]
