"""Independent, evidence-backed Betano Brazil settlement-terms registry.

The existing public-page snapshot is deliberately incomplete.  This contract
requires exact additional source bytes and two independent reviews before its
resolved rules may be treated as the settlement identity for phase two.  It
does not authorize a quote, model probability, transaction, or wager.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

from .betano_terms_snapshot_registry_v1 import (
    REGISTERED_SNAPSHOT_ARTIFACT_SHA256,
    REGISTERED_SNAPSHOT_LOCATOR,
    REGISTERED_SNAPSHOT_RAW_SHA256,
    validate_registered_betano_terms_snapshot_v1,
)
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_SETTLEMENT_CONTRACT_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "scryglass:betano-br-settlement-terms-authority:v1"
REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/"
    "bookmaker-terms-registry.json"
)
EVIDENCE_PREFIX = PurePosixPath(
    "data/lol/private_market_authority/evidence/betano-terms"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PRIVATE_MATCH_WINNER_BOOKMAKER_TERMS_SHA256"
REQUIRED_COVERAGE = {
    "league_of_legends_single_map_winner_settlement",
    "market_open_close_and_non_start",
    "postponement_and_cancellation",
    "remake_restart_and_resumption",
    "forfeit_walkover_and_disqualification",
    "void_refund_and_cash_odds_treatment",
}
REVIEW_SCOPES = {
    "SOURCE_AND_JURISDICTION": {
        "reviewer_independent_of_scryglass_model_and_market_code_authors": True,
        "exact_source_bytes_urls_capture_times_and_hashes_verified": True,
        "betano_brazil_jurisdiction_and_current_effective_version_verified": True,
        "all_required_rule_sections_covered_by_direct_source_evidence": True,
        "no_request_headers_cookies_credentials_or_account_data_embedded": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
    "SETTLEMENT_ALIGNMENT": {
        "reviewer_independent_of_source_capture_and_scryglass_code_authors": True,
        "single_map_winner_market_and_selection_semantics_verified": True,
        "non_start_postponement_cancellation_remake_resumption_and_forfeit_rules_verified": True,
        "void_refund_cash_odds_and_ambiguous_case_handling_verified": True,
        "each_resolved_rule_directly_supported_by_registered_evidence": True,
        "no_manual_post_outcome_override_or_exclusion_permitted": True,
        "review_not_generated_by_the_evaluated_system": True,
    },
}
AUTHORITY = {
    "bookmaker_terms_identity_authority": True,
    "settlement_rule_identity_authority": True,
    "quote_identity_authority": False,
    "phase_two_opening_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent Betano Brazil map-winner terms and settlement identity only. "
    "It does not prove quote availability or acceptance, open phase two, or "
    "authorize probability, expected value, recommendations, transactions, "
    "stakes, or betting."
)


class BetanoTermsAuthorityError(RuntimeError):
    """The terms evidence, independent reviews, or external pin failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BetanoTermsAuthorityError(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BetanoTermsAuthorityError(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BetanoTermsAuthorityError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise BetanoTermsAuthorityError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BetanoTermsAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetanoTermsAuthorityError("terms registry is not strict JSON") from exc
    if not isinstance(value, dict):
        raise BetanoTermsAuthorityError("terms registry must be an object")
    return value


def _evidence_locator(value: Any) -> str:
    text = _nonempty(value, "evidence.locator")
    locator = PurePosixPath(text)
    if (
        locator.is_absolute()
        or any(part in {"", ".", ".."} for part in locator.parts)
        or tuple(locator.parts[: len(EVIDENCE_PREFIX.parts)])
        != EVIDENCE_PREFIX.parts
        or locator.suffix not in {".html", ".json", ".pdf", ".txt"}
    ):
        raise BetanoTermsAuthorityError("terms evidence locator is invalid")
    return locator.as_posix()


def _validate_evidence(
    records: Any, *, root: Path, registered_at: datetime
) -> list[dict[str, Any]]:
    if not isinstance(records, list) or not records:
        raise BetanoTermsAuthorityError("terms evidence is empty")
    result: list[dict[str, Any]] = []
    source_ids: set[str] = set()
    locators: set[str] = set()
    coverage: set[str] = set()
    expected_keys = {
        "source_id",
        "source_url",
        "locator",
        "raw_sha256",
        "captured_at_utc",
        "effective_at_utc",
        "language",
        "access_method",
        "account_or_credentials_embedded",
        "coverage",
    }
    for record in records:
        if not isinstance(record, Mapping) or set(record) != expected_keys:
            raise BetanoTermsAuthorityError("terms evidence structure changed")
        source_id = _nonempty(record.get("source_id"), "evidence.source_id")
        locator = _evidence_locator(record.get("locator"))
        url = _nonempty(record.get("source_url"), "evidence.source_url")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not (
            parsed.netloc == "betano.bet.br"
            or parsed.netloc.endswith(".betano.bet.br")
        ):
            raise BetanoTermsAuthorityError(
                "terms evidence must be a Betano Brazil HTTPS source"
            )
        path = root / locator
        if path.is_symlink() or not path.is_file():
            raise BetanoTermsAuthorityError("terms evidence bytes are unavailable")
        digest = _sha(record.get("raw_sha256"), "evidence.raw_sha256")
        if _sha256(path.read_bytes()) != digest:
            raise BetanoTermsAuthorityError("terms evidence hash changed")
        captured = _timestamp(record.get("captured_at_utc"), "evidence.captured_at")
        effective = _timestamp(
            record.get("effective_at_utc"), "evidence.effective_at"
        )
        if effective > captured or captured > registered_at:
            raise BetanoTermsAuthorityError("terms evidence chronology changed")
        record_coverage = record.get("coverage")
        if (
            not isinstance(record_coverage, list)
            or not record_coverage
            or any(item not in REQUIRED_COVERAGE for item in record_coverage)
        ):
            raise BetanoTermsAuthorityError("terms evidence coverage changed")
        if (
            record.get("account_or_credentials_embedded") is not False
            or _nonempty(record.get("language"), "evidence.language")
            not in {"pt-BR", "en"}
            or _nonempty(record.get("access_method"), "evidence.access_method")
            not in {
                "fresh_unauthenticated_browser",
                "public_direct_download",
                "independent_manual_export_without_session_data",
            }
        ):
            raise BetanoTermsAuthorityError("terms evidence safety changed")
        if source_id in source_ids or locator in locators:
            raise BetanoTermsAuthorityError("terms evidence identity repeats")
        source_ids.add(source_id)
        locators.add(locator)
        coverage.update(record_coverage)
        result.append(dict(record))
    if coverage != REQUIRED_COVERAGE:
        raise BetanoTermsAuthorityError("terms evidence coverage is incomplete")
    if result != sorted(result, key=lambda item: (item["captured_at_utc"], item["source_id"])):
        raise BetanoTermsAuthorityError("terms evidence is not ordered")
    return result


def validate_betano_terms_authority_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BetanoTermsAuthorityError("terms authority must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "registry_id",
        "status",
        "registered_at_utc",
        "scope",
        "protocol_binding",
        "incomplete_public_snapshot_binding",
        "complete_terms_evidence",
        "resolved_settlement_rules",
        "reviews",
        "decision",
        "authority",
        "claim_ceiling",
    }:
        raise BetanoTermsAuthorityError("terms authority fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "COMPLETE_TERMS_INDEPENDENTLY_REGISTERED"
        or value.get("scope")
        != "betano_brazil_league_of_legends_single_map_winner_cash_odds"
    ):
        raise BetanoTermsAuthorityError("terms authority identity changed")
    _nonempty(value.get("registry_id"), "registry_id")
    registered_at = _timestamp(value.get("registered_at_utc"), "registered_at")
    protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    if value.get("protocol_binding") != {
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "settlement_contract_sha256": REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        "settlement_rule_id": protocol["settlement_contract"][
            "settlement_rule_id"
        ],
    }:
        raise BetanoTermsAuthorityError("terms protocol binding changed")
    public_snapshot = validate_registered_betano_terms_snapshot_v1(root=root)
    if value.get("incomplete_public_snapshot_binding") != {
        "locator": REGISTERED_SNAPSHOT_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_SNAPSHOT_RAW_SHA256,
        "artifact_sha256": REGISTERED_SNAPSHOT_ARTIFACT_SHA256,
        "acknowledged_incomplete": True,
    } or (public_snapshot.get("coverage") or {}).get(
        "complete_bookmaker_terms_snapshot"
    ) is not False:
        raise BetanoTermsAuthorityError("public snapshot binding changed")
    evidence = _validate_evidence(
        value.get("complete_terms_evidence"),
        root=root,
        registered_at=registered_at,
    )
    rules = value.get("resolved_settlement_rules")
    required_rule_keys = {
        "market_label",
        "winning_selection",
        "non_started_map",
        "same_day_resumption",
        "postponement",
        "cancellation",
        "remake_or_restart",
        "forfeit_walkover_or_disqualification",
        "void_refund_cash_odds_treatment",
        "ambiguous_or_conflicting_result",
        "manual_post_outcome_override_permitted",
        "supporting_source_ids_by_rule",
    }
    if not isinstance(rules, Mapping) or set(rules) != required_rule_keys:
        raise BetanoTermsAuthorityError("resolved settlement rules changed")
    for key in required_rule_keys - {
        "manual_post_outcome_override_permitted",
        "supporting_source_ids_by_rule",
    }:
        text = _nonempty(rules.get(key), f"resolved_settlement_rules.{key}")
        if text.upper() in {"UNKNOWN", "TBD", "UNRESOLVED"}:
            raise BetanoTermsAuthorityError("settlement rule remains unresolved")
    if rules.get("manual_post_outcome_override_permitted") is not False:
        raise BetanoTermsAuthorityError("manual post-outcome override is forbidden")
    support = rules.get("supporting_source_ids_by_rule")
    rule_names = required_rule_keys - {
        "manual_post_outcome_override_permitted",
        "supporting_source_ids_by_rule",
    }
    source_ids = {record["source_id"] for record in evidence}
    if (
        not isinstance(support, Mapping)
        or set(support) != rule_names
        or any(
            not isinstance(ids, list)
            or not ids
            or any(item not in source_ids for item in ids)
            for ids in support.values()
        )
    ):
        raise BetanoTermsAuthorityError("settlement rule evidence mapping changed")
    reviews = value.get("reviews")
    if not isinstance(reviews, list) or len(reviews) != 2:
        raise BetanoTermsAuthorityError("two terms reviews are required")
    reviewers: set[str] = set()
    scopes: set[str] = set()
    for review in reviews:
        if not isinstance(review, Mapping) or set(review) != {
            "review_scope",
            "reviewer_id",
            "reviewed_at_utc",
            "attestation",
        }:
            raise BetanoTermsAuthorityError("terms review structure changed")
        scope = _nonempty(review.get("review_scope"), "review_scope")
        reviewer = _nonempty(review.get("reviewer_id"), "reviewer_id")
        reviewed = _timestamp(review.get("reviewed_at_utc"), "reviewed_at")
        if (
            scope not in REVIEW_SCOPES
            or review.get("attestation") != REVIEW_SCOPES[scope]
            or reviewed > registered_at
        ):
            raise BetanoTermsAuthorityError("terms review is incomplete")
        reviewers.add(reviewer)
        scopes.add(scope)
    if len(reviewers) != 2 or scopes != set(REVIEW_SCOPES):
        raise BetanoTermsAuthorityError("terms reviews are not independent")
    if value.get("decision") != {
        "complete_bookmaker_terms_snapshot": True,
        "independent_alignment_review_present": True,
        "settlement_contract_resolved": True,
        "phase_two_opened": False,
        "betting_authorized": False,
    }:
        raise BetanoTermsAuthorityError("terms decision changed")
    if value.get("authority") != AUTHORITY:
        raise BetanoTermsAuthorityError("terms authority exceeds its scope")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise BetanoTermsAuthorityError("terms claim ceiling changed")
    return value


def load_pinned_betano_terms_authority_v1(
    *, path: Path, external_sha256: str, root: Path = ROOT
) -> dict[str, Any]:
    digest = _sha(external_sha256, "external terms registry digest")
    if path.is_symlink() or not path.is_file():
        raise BetanoTermsAuthorityError("terms registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise BetanoTermsAuthorityError(
            "terms registry does not match its external pin"
        )
    receipt = validate_betano_terms_authority_v1(_object(raw), root=root)
    return {
        "status": "complete_terms_independently_registered",
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "bookmaker_terms_snapshot_independently_registered": True,
        "settlement_contract_resolved": True,
        "phase_two_opening_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "AUTHORITY",
    "EVIDENCE_PREFIX",
    "EXTERNAL_SHA256_ENV",
    "REGISTRY_LOCATOR",
    "REQUIRED_COVERAGE",
    "REVIEW_SCOPES",
    "SCHEMA_VERSION",
    "BetanoTermsAuthorityError",
    "load_pinned_betano_terms_authority_v1",
    "validate_betano_terms_authority_v1",
]
