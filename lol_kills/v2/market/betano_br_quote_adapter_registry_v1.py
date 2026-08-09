"""Independent-registration contract for the Betano quote adapter candidate.

The candidate source and manifest can be built by Scryglass, but they do not
become an approved source-adapter identity until a separate reviewer issues a
registry and its exact canonical SHA-256 is supplied out of band.  Even that
registration grants no quote, probability, recommendation, transaction, or
betting authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .betano_br_quote_adapter_v1 import (
    ADAPTER_ID,
    AUTHORITY as CANDIDATE_AUTHORITY,
    BetanoQuoteAdapterError,
    DEFAULT_CANDIDATE_OUTPUT,
    SOURCE_LOCATOR,
    TRANSPORT_ID,
    sha256_json,
)
from .betano_br_quote_adapter_candidate_registry_v1 import (
    BetanoQuoteAdapterCandidateRegistryError,
    REGISTERED_CANDIDATE_ARTIFACT_SHA256,
    REGISTERED_CANDIDATE_RAW_SHA256,
    validate_registered_betano_quote_adapter_candidate_v1,
)
from .match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
    REGISTERED_SETTLEMENT_CONTRACT_SHA256,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "scryglass:betano-br-quote-adapter-registry:v1"
REGISTRY_SCOPE = "private_source_adapter_identity_only"
DEFAULT_REGISTRY = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/quote-adapter-registry.json"
)
AUTHORITY = {
    "source_adapter_identity_authority": True,
    "quote_identity_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
REVIEW = {
    "candidate_exact_raw_and_canonical_hash_reviewed": True,
    "adapter_source_hash_reviewed": True,
    "public_event_url_allowlist_reviewed": True,
    "exact_response_body_boundary_reviewed": True,
    "initial_state_parser_and_duplicate_key_rejection_reviewed": True,
    "event_market_selection_suspension_rule_reviewed": True,
    "future_market_close_rule_reviewed": True,
    "two_participant_selection_and_price_binding_reviewed": True,
    "system_and_monotonic_transport_timing_reviewed": True,
    "fresh_unauthenticated_browser_profile_reviewed": True,
    "headers_cookies_credentials_exclusion_reviewed": True,
    "generic_quote_and_settlement_hash_binding_reviewed": True,
    "outcome_field_rejection_reviewed": True,
}
CLAIM_CEILING = (
    "Independent source-adapter identity registration only. It does not "
    "register any quote, open phase two, resolve bookmaker settlement terms, "
    "or authorize probability, expected value, recommendations, transactions, "
    "stakes, or betting."
)


class BetanoQuoteAdapterRegistryError(ValueError):
    """The independent adapter registry is absent, malformed, or unpinned."""


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BetanoQuoteAdapterRegistryError(f"{label} must be non-empty")
    return value.strip()


def _time(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BetanoQuoteAdapterRegistryError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise BetanoQuoteAdapterRegistryError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BetanoQuoteAdapterRegistryError(
            f"{label} must be a lowercase SHA-256"
        )
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BetanoQuoteAdapterRegistryError(
                f"adapter registry contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _candidate(root: Path) -> tuple[bytes, dict[str, Any]]:
    path = root / DEFAULT_CANDIDATE_OUTPUT
    if not path.is_file() or path.is_symlink():
        raise BetanoQuoteAdapterRegistryError("adapter candidate is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetanoQuoteAdapterRegistryError("adapter candidate is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise BetanoQuoteAdapterRegistryError("adapter candidate must be an object")
    try:
        checked = validate_registered_betano_quote_adapter_candidate_v1(root=root)
    except (
        BetanoQuoteAdapterCandidateRegistryError,
        BetanoQuoteAdapterError,
        OSError,
        ValueError,
    ) as exc:
        raise BetanoQuoteAdapterRegistryError(str(exc)) from exc
    if (
        hashlib.sha256(raw).hexdigest() != REGISTERED_CANDIDATE_RAW_SHA256
        or checked.get("artifact_sha256")
        != REGISTERED_CANDIDATE_ARTIFACT_SHA256
    ):
        raise BetanoQuoteAdapterRegistryError("adapter candidate code pin changed")
    return raw, checked


def build_betano_quote_adapter_registry_v1(
    *,
    independent_reviewer_id: str,
    registry_id: str,
    issued_at: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build a registry candidate whose digest still needs out-of-band pinning."""

    issued = _time(issued_at, "issued_at")
    raw_candidate, candidate = _candidate(root)
    candidate_locked = _time(candidate["locked_at_utc"], "candidate.locked_at_utc")
    if issued < candidate_locked:
        raise BetanoQuoteAdapterRegistryError("adapter registry predates candidate")
    source_lock = candidate["source_lock"]
    registry = {
        "schema_version": SCHEMA_VERSION,
        "status": "approved",
        "scope": REGISTRY_SCOPE,
        "public_or_transactional_use": False,
        "registry_id": _nonempty(registry_id, "registry_id"),
        "independent_reviewer_id": _nonempty(
            independent_reviewer_id, "independent_reviewer_id"
        ),
        "issued_at": issued.isoformat(),
        "candidate": {
            "locator": DEFAULT_CANDIDATE_OUTPUT.as_posix(),
            "raw_sha256": hashlib.sha256(raw_candidate).hexdigest(),
            "artifact_sha256": candidate["artifact_sha256"],
            "locked_at_utc": candidate["locked_at_utc"],
        },
        "adapter": {
            "adapter_id": ADAPTER_ID,
            "transport_id": TRANSPORT_ID,
            "source_locator": SOURCE_LOCATOR,
            "source_sha256": source_lock["raw_sha256"],
        },
        "protocol_bindings": {
            "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "quote_capture_contract_sha256": REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
            "settlement_contract_sha256": REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        },
        "review": dict(REVIEW),
        "candidate_authority_confirmed_non_authorizing": (
            candidate["authority"] == CANDIDATE_AUTHORITY
        ),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    validate_betano_quote_adapter_registry_v1(
        registry,
        expected_registry_sha256=sha256_json(registry),
        root=root,
    )
    return registry


def validate_betano_quote_adapter_registry_v1(
    registry: Mapping[str, Any],
    *,
    expected_registry_sha256: str | None,
    root: Path = ROOT,
) -> dict[str, Any]:
    if expected_registry_sha256 is None:
        raise BetanoQuoteAdapterRegistryError(
            "adapter registry has no out-of-band digest"
        )
    if not isinstance(registry, Mapping):
        raise BetanoQuoteAdapterRegistryError("adapter registry must be an object")
    expected_sha = _sha(expected_registry_sha256, "expected_registry_sha256")
    if sha256_json(registry) != expected_sha:
        raise BetanoQuoteAdapterRegistryError("adapter registry digest mismatch")
    expected_keys = {
        "schema_version",
        "status",
        "scope",
        "public_or_transactional_use",
        "registry_id",
        "independent_reviewer_id",
        "issued_at",
        "candidate",
        "adapter",
        "protocol_bindings",
        "review",
        "candidate_authority_confirmed_non_authorizing",
        "authority",
        "claim_ceiling",
    }
    if set(registry) != expected_keys:
        raise BetanoQuoteAdapterRegistryError("adapter registry keys changed")
    if (
        registry.get("schema_version") != SCHEMA_VERSION
        or registry.get("status") != "approved"
        or registry.get("scope") != REGISTRY_SCOPE
        or registry.get("public_or_transactional_use") is not False
    ):
        raise BetanoQuoteAdapterRegistryError("adapter registry is not approved")
    _nonempty(registry.get("registry_id"), "registry_id")
    _nonempty(registry.get("independent_reviewer_id"), "independent_reviewer_id")
    issued = _time(registry.get("issued_at"), "issued_at")
    raw_candidate, candidate = _candidate(root)
    candidate_record = registry.get("candidate")
    expected_candidate = {
        "locator": DEFAULT_CANDIDATE_OUTPUT.as_posix(),
        "raw_sha256": hashlib.sha256(raw_candidate).hexdigest(),
        "artifact_sha256": candidate["artifact_sha256"],
        "locked_at_utc": candidate["locked_at_utc"],
    }
    if candidate_record != expected_candidate:
        raise BetanoQuoteAdapterRegistryError("adapter candidate binding changed")
    if issued < _time(candidate["locked_at_utc"], "candidate.locked_at_utc"):
        raise BetanoQuoteAdapterRegistryError("adapter registry predates candidate")
    source_lock = candidate["source_lock"]
    if registry.get("adapter") != {
        "adapter_id": ADAPTER_ID,
        "transport_id": TRANSPORT_ID,
        "source_locator": SOURCE_LOCATOR,
        "source_sha256": source_lock["raw_sha256"],
    }:
        raise BetanoQuoteAdapterRegistryError("adapter source identity changed")
    if registry.get("protocol_bindings") != {
        "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "quote_capture_contract_sha256": REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        "settlement_contract_sha256": REGISTERED_SETTLEMENT_CONTRACT_SHA256,
    }:
        raise BetanoQuoteAdapterRegistryError("adapter protocol binding changed")
    if registry.get("review") != REVIEW:
        raise BetanoQuoteAdapterRegistryError("adapter independent review is incomplete")
    if registry.get("candidate_authority_confirmed_non_authorizing") is not True:
        raise BetanoQuoteAdapterRegistryError("candidate authority review is incomplete")
    if candidate.get("authority") != CANDIDATE_AUTHORITY:
        raise BetanoQuoteAdapterRegistryError("candidate authority changed")
    if registry.get("authority") != AUTHORITY:
        raise BetanoQuoteAdapterRegistryError("adapter registry exceeds its authority")
    if registry.get("claim_ceiling") != CLAIM_CEILING:
        raise BetanoQuoteAdapterRegistryError("adapter registry claim ceiling changed")
    return {**dict(registry), "registry_sha256": expected_sha}


def load_registered_betano_quote_adapter_v1(
    *,
    expected_registry_sha256: str | None,
    root: Path = ROOT,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    path = registry_path or (root / DEFAULT_REGISTRY)
    if not path.is_file() or path.is_symlink():
        raise BetanoQuoteAdapterRegistryError("adapter registry is unavailable")
    try:
        raw = path.read_bytes()
        value = json.loads(raw, object_pairs_hook=_strict_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetanoQuoteAdapterRegistryError("adapter registry is invalid JSON") from exc
    return validate_betano_quote_adapter_registry_v1(
        value,
        expected_registry_sha256=expected_registry_sha256,
        root=root,
    )


__all__ = [
    "AUTHORITY",
    "BetanoQuoteAdapterRegistryError",
    "DEFAULT_REGISTRY",
    "REGISTRY_SCOPE",
    "SCHEMA_VERSION",
    "build_betano_quote_adapter_registry_v1",
    "load_registered_betano_quote_adapter_v1",
    "validate_betano_quote_adapter_registry_v1",
]
