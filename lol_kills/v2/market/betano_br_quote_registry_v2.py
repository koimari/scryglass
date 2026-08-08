"""Independent identity registry for map-start-qualified Betano v2 quotes."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import betano_br_quote_qualification_v1 as qualification
from . import event_probability_registry_v2 as probability_registry
from . import phase_one_evaluation_v1 as evaluation


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/betano_br_quote_registry_v2.py"
SCHEMA_VERSION = "scryglass:betano-br-map-winner-quote-registry:v2"
REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/betano-quote-registry.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PRIVATE_MATCH_WINNER_QUOTE_REGISTRY_SHA256"
REVIEW_ATTESTATION = {
    "reviewer_independent_of_probability_quote_adapter_and_market_code_authors": True,
    "every_qualification_quote_map_start_and_exact_response_replayed": True,
    "quote_response_preceded_actual_map_start_by_at_least_five_seconds": True,
    "matching_event_probability_identity_independently_registered": True,
    "phase_two_opening_terms_and_source_adapter_bindings_replayed": True,
    "outcomes_prices_not_used_to_change_model_probability": True,
    "no_receipt_replacement_retrospective_backfill_or_manual_override_found": True,
    "review_not_generated_by_the_evaluated_system": True,
}
AUTHORITY = {
    "quote_identity_authority": True,
    "odds_accuracy_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent identity registration of exact Betano quote responses that "
    "passed the authoritative five-second pre-map boundary. It does not "
    "establish odds accuracy, predictive accuracy, expected value, a "
    "recommendation, transaction acceptance, stakes, or betting authority."
)


class BetanoQuoteRegistryV2Error(RuntimeError):
    """The quote inventory, probability join, review, or external pin failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise BetanoQuoteRegistryV2Error(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BetanoQuoteRegistryV2Error(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise BetanoQuoteRegistryV2Error(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise BetanoQuoteRegistryV2Error(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BetanoQuoteRegistryV2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetanoQuoteRegistryV2Error(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise BetanoQuoteRegistryV2Error(f"{label} must be an object")
    return value


def _probability_index(
    *, root: Path, environment: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    path = root / probability_registry.REGISTRY_LOCATOR
    digest = environment.get(probability_registry.EXTERNAL_SHA256_ENV)
    if path.is_symlink() or not path.is_file() or not digest:
        raise BetanoQuoteRegistryV2Error(
            "independent event-probability registry is unavailable"
        )
    raw_registry = _object(path.read_bytes(), "event-probability registry")
    locators = [
        item.get("receipt_locator")
        for item in (raw_registry.get("entries") or [])
        if isinstance(item, Mapping)
    ]
    if not locators or any(not isinstance(item, str) for item in locators):
        raise BetanoQuoteRegistryV2Error("event-probability inventory is missing")
    try:
        expected = probability_registry.expected_entries(
            receipt_locators=locators, root=root, environment=environment
        )
        loaded = probability_registry.load_pinned_event_probability_registry_v2(
            path=path, external_sha256=digest, expected=expected
        )
    except Exception as exc:
        raise BetanoQuoteRegistryV2Error(
            "event-probability registry is invalid"
        ) from exc
    return {
        str(item["receipt_artifact_sha256"]): dict(item)
        for item in loaded["receipt"]["entries"]
    }


def _qualification(
    *, locator_value: str, root: Path, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, qualification.OUTPUT_PREFIX, "qualification_locator"
    )
    raw = evaluation._read_regular(root, locator, "quote qualification")
    try:
        checked = qualification.validate_betano_quote_qualification_v1(
            evaluation._strict_object(raw, "quote qualification"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise BetanoQuoteRegistryV2Error("quote qualification is invalid") from exc
    return locator, raw, checked


def expected_entries(
    *, qualification_locators: Sequence[str], root: Path = ROOT,
    environment: Mapping[str, str] = os.environ
) -> list[dict[str, Any]]:
    probabilities = _probability_index(root=root, environment=environment)
    entries: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for locator_value in qualification_locators:
        locator, raw, receipt = _qualification(
            locator_value=locator_value, root=root, environment=environment
        )
        if receipt["qualification_output_locator"] != locator:
            raise BetanoQuoteRegistryV2Error(
                "qualification was loaded from an unreserved locator"
            )
        event = receipt["event"]
        identity = (str(event["event_id"]), int(event["game_number"]))
        if identity in identities:
            raise BetanoQuoteRegistryV2Error("qualified quote identity repeats")
        identities.add(identity)
        probability_sha = receipt["quote_binding"][
            "event_probability_artifact_sha256"
        ]
        probability_entry = probabilities.get(probability_sha)
        if probability_entry is None:
            raise BetanoQuoteRegistryV2Error(
                "qualified quote probability is not independently registered"
            )
        for field in (
            "event_id", "series_id", "game_number", "league", "patch",
            "roster_change_stratum", "sparse_or_new_champion_map", "market_type",
            "selection", "opposing_selection",
        ):
            if event[field] != probability_entry[field]:
                raise BetanoQuoteRegistryV2Error(
                    f"quote and probability registry differ: {field}"
                )
        timing = receipt["timing"]
        entries.append(
            {
                **event,
                "qualification_locator": locator,
                "qualification_raw_sha256": _sha256(raw),
                "qualification_artifact_sha256": receipt["artifact_sha256"],
                "event_plan_locator": receipt["event_plan_binding"]["locator"],
                "event_plan_raw_sha256": receipt["event_plan_binding"]["raw_sha256"],
                "event_plan_artifact_sha256": receipt["event_plan_binding"][
                    "artifact_sha256"
                ],
                "quote_locator": receipt["quote_binding"]["locator"],
                "quote_raw_sha256": receipt["quote_binding"]["raw_sha256"],
                "quote_artifact_sha256": receipt["quote_binding"]["artifact_sha256"],
                "generic_quote_receipt_sha256": receipt["quote_binding"][
                    "generic_quote_receipt_sha256"
                ],
                "event_probability_artifact_sha256": probability_sha,
                "map_start_locator": receipt["map_start_binding"]["locator"],
                "map_start_raw_sha256": receipt["map_start_binding"]["raw_sha256"],
                "map_start_artifact_sha256": receipt["map_start_binding"][
                    "artifact_sha256"
                ],
                "quote_response_received_at_utc": timing[
                    "quote_response_received_at_utc"
                ],
                "actual_map_start_utc": timing["actual_map_start_utc"],
                "response_to_start_seconds": timing[
                    "quote_response_to_actual_map_start_seconds"
                ],
            }
        )
    entries.sort(
        key=lambda item: (
            item["quote_response_received_at_utc"], item["event_id"],
            item["game_number"],
        )
    )
    return entries


def validate_betano_quote_registry_v2(
    payload: Mapping[str, Any], *, expected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BetanoQuoteRegistryV2Error("registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "registry_id", "status", "issued_at_utc",
        "independent_review", "entries", "decision", "authority", "claim_ceiling",
    }:
        raise BetanoQuoteRegistryV2Error("registry fields are not exact")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("status") != "QUALIFIED_BETANO_QUOTE_IDENTITIES_REGISTERED":
        raise BetanoQuoteRegistryV2Error("registry identity changed")
    _nonempty(value.get("registry_id"), "registry_id")
    issued = _timestamp(value.get("issued_at_utc"), "issued_at_utc")
    entries = [dict(item) for item in expected]
    if not entries or value.get("entries") != entries:
        raise BetanoQuoteRegistryV2Error("registry entries changed")
    if any(
        _timestamp(item["actual_map_start_utc"], "entry.actual_map_start") > issued
        or float(item["response_to_start_seconds"])
        < qualification.MINIMUM_RESPONSE_TO_START_SECONDS
        for item in entries
    ):
        raise BetanoQuoteRegistryV2Error("registry predates evidence or timing failed")
    review = value.get("independent_review")
    if not isinstance(review, Mapping) or set(review) != {
        "reviewer_id", "reviewed_at_utc", "attestation"
    }:
        raise BetanoQuoteRegistryV2Error("registry review structure changed")
    if (
        not _nonempty(review.get("reviewer_id"), "reviewer_id")
        or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > issued
        or review.get("attestation") != REVIEW_ATTESTATION
    ):
        raise BetanoQuoteRegistryV2Error("registry review is incomplete")
    if value.get("decision") != {
        "qualified_quote_receipts_independently_registered": True,
        "registered_quotes": len(entries),
        "odds_accuracy_authorized": False,
        "betting_authorized": False,
    }:
        raise BetanoQuoteRegistryV2Error("registry decision changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise BetanoQuoteRegistryV2Error("registry exceeds authority")
    return value


def load_pinned_betano_quote_registry_v2(
    *, path: Path, external_sha256: str, expected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    digest = _sha(external_sha256, "external registry digest")
    if path.is_symlink() or not path.is_file():
        raise BetanoQuoteRegistryV2Error("registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise BetanoQuoteRegistryV2Error("registry does not match its external pin")
    receipt = validate_betano_quote_registry_v2(
        _object(raw, "quote registry"), expected=expected
    )
    return {
        "status": "qualified_betano_quote_identities_registered",
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "registered_quotes": len(receipt["entries"]),
        "quote_identity_authority": True,
        "odds_accuracy_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "EXTERNAL_SHA256_ENV", "REGISTRY_LOCATOR", "REVIEW_ATTESTATION",
    "SCHEMA_VERSION", "SOURCE_LOCATOR", "BetanoQuoteRegistryV2Error",
    "expected_entries", "load_pinned_betano_quote_registry_v2",
    "validate_betano_quote_registry_v2",
]
