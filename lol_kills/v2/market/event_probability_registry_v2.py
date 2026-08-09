"""Externally pinned identity registry for phase-two probability v2 receipts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import event_probability_v2 as probability
from . import phase_one_evaluation_v1 as evaluation


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/event_probability_registry_v2.py"
SCHEMA_VERSION = "scryglass:private-event-probability-registry:v2"
REGISTRY_LOCATOR = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/"
    "event-probability-registry.json"
)
EXTERNAL_SHA256_ENV = "SCRYGLASS_PRIVATE_MATCH_WINNER_PROBABILITY_REGISTRY_SHA256"
REVIEW_ATTESTATION = {
    "reviewer_independent_of_model_probability_quote_and_market_code_authors": True,
    "every_receipt_file_raw_and_canonical_hash_replayed": True,
    "active_phase_two_opening_binding_replayed": True,
    "exact_target_recalibration_draws_interval_and_generation_source_replayed": True,
    "percentile_interval_point_containment_not_required_or_falsified": True,
    "event_outcomes_and_market_prices_absent_from_model_receipts": True,
    "no_receipt_replacement_retrospective_backfill_or_manual_override_found": True,
    "review_not_generated_by_the_evaluated_system": True,
}
AUTHORITY = {
    "event_probability_identity_authority": True,
    "probability_accuracy_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Independent identity registration of exact phase-two probability receipts "
    "only. It does not establish predictive accuracy, quote identity, expected "
    "value, recommendation quality, transaction acceptance, stakes, or betting authority."
)


class EventProbabilityRegistryV2Error(RuntimeError):
    """The receipt inventory, independent review, or external pin failed closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise EventProbabilityRegistryV2Error(
            f"{field} must be a lowercase SHA-256"
        )
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EventProbabilityRegistryV2Error(f"{field} must be nonempty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise EventProbabilityRegistryV2Error(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise EventProbabilityRegistryV2Error(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise EventProbabilityRegistryV2Error(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EventProbabilityRegistryV2Error("registry is not strict JSON") from exc
    if not isinstance(value, dict):
        raise EventProbabilityRegistryV2Error("registry must be an object")
    return value


def _receipt(
    *, locator_value: str, root: Path, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, probability.RECEIPT_PREFIX, "probability_receipt_locator"
    )
    raw = evaluation._read_regular(root, locator, "event probability receipt")
    try:
        checked = probability.validate_event_probability_v2(
            evaluation._strict_object(raw, "event probability receipt"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise EventProbabilityRegistryV2Error(
            "event probability receipt is invalid"
        ) from exc
    return locator, raw, checked


def expected_entries(
    *,
    receipt_locators: Sequence[str],
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    identities: set[tuple[str, int]] = set()
    for locator_value in receipt_locators:
        locator, raw, receipt = _receipt(
            locator_value=locator_value, root=root, environment=environment
        )
        event = receipt["event"]
        identity = (str(event["event_id"]), int(event["game_number"]))
        if identity in identities:
            raise EventProbabilityRegistryV2Error(
                "probability event identity repeats"
            )
        identities.add(identity)
        entries.append(
            {
                "event_id": event["event_id"],
                "series_id": event["series_id"],
                "game_number": event["game_number"],
                "league": event["league"],
                "patch": event["patch"],
                "roster_change_stratum": event["roster_change_stratum"],
                "sparse_or_new_champion_map": event[
                    "sparse_or_new_champion_map"
                ],
                "market_type": event["market_type"],
                "selection": event["selection"],
                "opposing_selection": event["opposing_selection"],
                "captured_at_utc": receipt["captured_at_utc"],
                "receipt_locator": locator,
                "receipt_raw_sha256": _sha256(raw),
                "receipt_artifact_sha256": receipt["artifact_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "fast_uncertainty_artifact_sha256": receipt["input_binding"][
                    "fast_uncertainty_artifact_sha256"
                ],
                "draws_sha256": receipt["uncertainty"]["draws_sha256"],
                "probability": receipt["probability"],
                "rating_only_probability": receipt["calculation"][
                    "rating_only_comparator"
                ]["probability_blue"],
                "probability_interval": receipt["probability_interval"],
                "point_inside_percentile_interval": receipt["uncertainty"][
                    "point_inside_percentile_interval"
                ],
            }
        )
    entries.sort(
        key=lambda item: (
            item["captured_at_utc"],
            item["event_id"],
            item["game_number"],
        )
    )
    return entries


def validate_event_probability_registry_v2(
    payload: Mapping[str, Any], *, expected: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise EventProbabilityRegistryV2Error("registry must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "registry_id",
        "status",
        "issued_at_utc",
        "independent_review",
        "entries",
        "decision",
        "authority",
        "claim_ceiling",
    }:
        raise EventProbabilityRegistryV2Error("registry fields are not exact")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("status") != "EVENT_PROBABILITY_IDENTITIES_REGISTERED"
    ):
        raise EventProbabilityRegistryV2Error("registry identity changed")
    _nonempty(value.get("registry_id"), "registry_id")
    issued = _timestamp(value.get("issued_at_utc"), "issued_at_utc")
    entries = [dict(item) for item in expected]
    if not entries or value.get("entries") != entries:
        raise EventProbabilityRegistryV2Error("registry entries changed")
    if any(
        _timestamp(item["captured_at_utc"], "entry.captured_at") > issued
        for item in entries
    ):
        raise EventProbabilityRegistryV2Error("registry predates a receipt")
    review = value.get("independent_review")
    if not isinstance(review, Mapping) or set(review) != {
        "reviewer_id",
        "reviewed_at_utc",
        "attestation",
    }:
        raise EventProbabilityRegistryV2Error("registry review structure changed")
    if (
        not _nonempty(review.get("reviewer_id"), "reviewer_id")
        or _timestamp(review.get("reviewed_at_utc"), "reviewed_at_utc") > issued
        or review.get("attestation") != REVIEW_ATTESTATION
    ):
        raise EventProbabilityRegistryV2Error("registry review is incomplete")
    if value.get("decision") != {
        "event_probability_receipts_independently_registered": True,
        "registered_receipts": len(entries),
        "probability_accuracy_authorized": False,
        "betting_authorized": False,
    }:
        raise EventProbabilityRegistryV2Error("registry decision changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise EventProbabilityRegistryV2Error("registry exceeds authority")
    return value


def load_pinned_event_probability_registry_v2(
    *,
    path: Path,
    external_sha256: str,
    expected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    digest = _sha(external_sha256, "external registry digest")
    if path.is_symlink() or not path.is_file():
        raise EventProbabilityRegistryV2Error("registry is unavailable")
    raw = path.read_bytes()
    if _sha256(raw) != digest:
        raise EventProbabilityRegistryV2Error(
            "registry does not match its external pin"
        )
    receipt = validate_event_probability_registry_v2(
        _object(raw), expected=expected
    )
    return {
        "status": "event_probability_identities_registered",
        "receipt": receipt,
        "receipt_raw_sha256": digest,
        "registered_receipts": len(receipt["entries"]),
        "event_probability_identity_authority": True,
        "probability_accuracy_authorized": False,
        "betting_authorized": False,
    }


__all__ = [
    "EXTERNAL_SHA256_ENV",
    "REGISTRY_LOCATOR",
    "REVIEW_ATTESTATION",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "EventProbabilityRegistryV2Error",
    "expected_entries",
    "load_pinned_event_probability_registry_v2",
    "validate_event_probability_registry_v2",
]
