"""Validate an externally pinned independent side-neutral protocol review.

The review may authorize only prospective collection after its review time.
It cannot authorize retrospective artifacts, outcome opening, ratings,
probabilities, odds, EV, recommendations, or betting.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

from .multileague_v3_side_neutral_protocol_registry_v2 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_side_neutral_protocol_v2,
)
from .multileague_v3_side_neutral_protocol_v2 import INDEPENDENT_REVIEW_ENV
from .side_neutral_collection_implementation_registry_v1 import (
    validate_registered_side_neutral_collection_implementation,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:side-neutral-protocol-independent-review:v1"
REVIEW_LOCATOR = Path(
    "data/lol/v2/authorities/multileague-v3/"
    "side-neutral-protocol-review-v2.json"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUTHORITY_KEYS = (
    "prospective_collection_authority",
    "outcome_opening_authority",
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "draft_validation_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)


class SideNeutralProtocolReviewError(ValueError):
    """The external review is absent, stale, self-authored, or overbroad."""


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SideNeutralProtocolReviewError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SideNeutralProtocolReviewError(f"non-finite number: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralProtocolReviewError("review is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SideNeutralProtocolReviewError("review must be a JSON object")
    return value


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SideNeutralProtocolReviewError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SideNeutralProtocolReviewError(f"{field} must be non-empty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralProtocolReviewError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SideNeutralProtocolReviewError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def validate_side_neutral_protocol_review(
    payload: Mapping[str, Any], *, root: Path = ROOT, as_of: datetime
) -> dict[str, Any]:
    if not isinstance(as_of, datetime) or as_of.tzinfo is None:
        raise SideNeutralProtocolReviewError("as_of must be timezone-aware")
    if not isinstance(payload, Mapping) or set(payload) != {
        "schema_version",
        "review_id",
        "reviewer",
        "reviewed_at_utc",
        "protocol",
        "reviewed_source_locks",
        "reviewed_admission_implementation",
        "findings",
        "authorization",
        "authority",
        "claim_ceiling",
    }:
        raise SideNeutralProtocolReviewError("review structure changed")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise SideNeutralProtocolReviewError("review schema changed")
    _nonempty(value.get("review_id"), "review_id")
    reviewer = value.get("reviewer")
    if not isinstance(reviewer, Mapping) or set(reviewer) != {
        "reviewer_id",
        "reviewer_role",
        "independent_from_implementation",
        "not_the_protocol_author",
        "conflicts_disclosed",
    }:
        raise SideNeutralProtocolReviewError("reviewer structure changed")
    _nonempty(reviewer.get("reviewer_id"), "reviewer_id")
    if (
        reviewer.get("reviewer_role") != "independent-human-reviewer"
        or reviewer.get("independent_from_implementation") is not True
        or reviewer.get("not_the_protocol_author") is not True
        or reviewer.get("conflicts_disclosed") is not True
    ):
        raise SideNeutralProtocolReviewError("reviewer is not independently attested")
    reviewed_at = _timestamp(value.get("reviewed_at_utc"), "reviewed_at_utc")
    if reviewed_at <= _timestamp(REGISTERED_PROTOCOL_LOCKED_AT_UTC, "protocol lock"):
        raise SideNeutralProtocolReviewError("review predates the frozen protocol")
    if reviewed_at > as_of.astimezone(timezone.utc):
        raise SideNeutralProtocolReviewError("review timestamp is in the future")
    protocol = validate_registered_side_neutral_protocol_v2(root=root)
    admission_implementation = (
        validate_registered_side_neutral_collection_implementation(root=root)
    )
    if value.get("protocol") != {
        "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "locked_at_utc": REGISTERED_PROTOCOL_LOCKED_AT_UTC,
    }:
        raise SideNeutralProtocolReviewError("reviewed protocol binding changed")
    if value.get("reviewed_source_locks") != protocol["source_locks"]:
        raise SideNeutralProtocolReviewError("reviewed implementation hashes changed")
    if value.get("reviewed_admission_implementation") != admission_implementation[
        "records"
    ]:
        raise SideNeutralProtocolReviewError(
            "reviewed admission implementation hashes changed"
        )
    expected_findings = {
        "protocol_and_implementation_reviewed": True,
        "source_provenance_and_exact_roster_binding_reviewed": True,
        "side_selection_without_rating_refit_reviewed": True,
        "terminal_draft_and_actual_start_timing_reviewed": True,
        "duplicate_or_ambiguous_side_binding_policy_reviewed": True,
        "outcome_leakage_controls_reviewed": True,
        "no_clobber_persistence_reviewed": True,
        "model_source_boundary_stopping_evaluation_and_uncertainty_unchanged": True,
        "future_outcomes_accessed": False,
        "future_predictions_accessed": False,
        "unresolved_critical_findings": [],
    }
    if value.get("findings") != expected_findings:
        raise SideNeutralProtocolReviewError("independent findings are incomplete")
    expected_authorization = {
        "prospective_collection_authorized": True,
        "effective_at_utc": reviewed_at.isoformat(),
        "captures_before_effective_time_eligible": False,
        "retrospective_backfill_authorized": False,
        "outcome_opening_authorized": False,
        "rating_or_draft_authority_granted": False,
        "probability_odds_ev_or_recommendation_authorized": False,
        "betting_authorized": False,
    }
    if value.get("authorization") != expected_authorization:
        raise SideNeutralProtocolReviewError("review authorization is overbroad")
    expected_authority = {
        "prospective_collection_authority": True,
        **{name: False for name in AUTHORITY_KEYS if name != "prospective_collection_authority"},
    }
    if value.get("authority") != expected_authority:
        raise SideNeutralProtocolReviewError("review authority boundary changed")
    expected_ceiling = (
        "Independent authorization for prospective outcome-free collection only, "
        "effective after this review. No retrospective evidence, outcome opening, "
        "rating, Draft, probability, odds, EV, recommendation, or betting authority."
    )
    if value.get("claim_ceiling") != expected_ceiling:
        raise SideNeutralProtocolReviewError("review claim ceiling changed")
    return value


def load_active_side_neutral_protocol_review(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str],
    as_of: datetime,
) -> dict[str, Any]:
    external_sha = environment.get(INDEPENDENT_REVIEW_ENV)
    if not external_sha:
        raise SideNeutralProtocolReviewError(
            f"missing external review digest: {INDEPENDENT_REVIEW_ENV}"
        )
    expected_sha = _sha(external_sha, INDEPENDENT_REVIEW_ENV)
    path = root / REVIEW_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise SideNeutralProtocolReviewError("independent review file is unavailable")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise SideNeutralProtocolReviewError("external review digest does not match")
    payload = _strict_object(raw)
    return validate_side_neutral_protocol_review(payload, root=root, as_of=as_of)


__all__ = [
    "REVIEW_LOCATOR",
    "SCHEMA_VERSION",
    "SideNeutralProtocolReviewError",
    "load_active_side_neutral_protocol_review",
    "validate_side_neutral_protocol_review",
]
