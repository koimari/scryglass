"""Validate one independent Draft promotion receipt for public publication."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import canonical_sha256
from lol_kills.research.verify_selective_draft_promotion import (
    APPROVED_FIELDS,
    RECEIPT_SCHEMA_VERSION,
)


DRAFT_AUTHORITY_SCHEMA = "scryglass:draft-authority:v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
RELEASE_ID_PATTERN = re.compile(r"v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}")
UTC_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z"
)
BOUND_HASH_FIELDS = (
    "candidate_artifact_sha256",
    "candidate_receipt_sha256",
    "protocol_file_sha256",
    "evaluation_file_sha256",
    "evaluation_receipt_sha256",
    "decision_file_sha256",
    "decision_receipt_sha256",
    "outcomes_sha256",
)


class PromotedDraftAuthorityError(ValueError):
    """Raised when a promotion receipt cannot authorize publication."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PromotedDraftAuthorityError("promotion receipt is unreadable") from error
    if not isinstance(value, dict):
        raise PromotedDraftAuthorityError("promotion receipt is not an object")
    return value


def load_promoted_draft_authority(
    *,
    receipt_path: Path,
    expected_file_sha256: str,
    release_id: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a fixed manifest authority and its verified private receipt."""

    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise PromotedDraftAuthorityError("release ID is invalid")
    if not SHA256_PATTERN.fullmatch(expected_file_sha256):
        raise PromotedDraftAuthorityError("promotion receipt file SHA-256 is invalid")
    if not receipt_path.is_file() or sha256_path(receipt_path) != expected_file_sha256:
        raise PromotedDraftAuthorityError("promotion receipt file changed")

    receipt = _load_json(receipt_path)
    receipt_sha256 = receipt.get("receipt_sha256")
    unsigned = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if (
        not isinstance(receipt_sha256, str)
        or not SHA256_PATTERN.fullmatch(receipt_sha256)
        or receipt_sha256 != canonical_sha256(unsigned)
    ):
        raise PromotedDraftAuthorityError("promotion receipt digest changed")

    model_version = receipt.get("model_version")
    issued_utc = receipt.get("issued_utc")
    paired_receipts = receipt.get("controlled_intervention_receipt_sha256")
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "promoted"
        or receipt.get("authority") != "promoted"
        or not isinstance(model_version, str)
        or not model_version.strip()
        or tuple(receipt.get("approved_public_fields") or ()) != APPROVED_FIELDS
        or receipt.get("public_probability") is not True
        or receipt.get("public_recommendation") is not True
        or receipt.get("betting_odds_ev_stake") is not False
        or not isinstance(receipt.get("reviewer_identity"), str)
        or not receipt["reviewer_identity"].strip()
        or any(
            not isinstance(receipt.get(field), str)
            or not SHA256_PATTERN.fullmatch(receipt[field])
            for field in BOUND_HASH_FIELDS
        )
        or not isinstance(paired_receipts, list)
        or not paired_receipts
        or any(
            not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
            for value in paired_receipts
        )
    ):
        raise PromotedDraftAuthorityError("promotion receipt contract is invalid")
    if not isinstance(issued_utc, str) or not UTC_PATTERN.fullmatch(issued_utc):
        raise PromotedDraftAuthorityError("promotion receipt time is invalid")
    try:
        issued = datetime.fromisoformat(issued_utc.removesuffix("Z") + "+00:00")
    except ValueError as error:
        raise PromotedDraftAuthorityError("promotion receipt time is invalid") from error
    if issued.tzinfo is None or issued.utcoffset() != timezone.utc.utcoffset(issued):
        raise PromotedDraftAuthorityError("promotion receipt time is invalid")

    authority = {
        "schema_version": DRAFT_AUTHORITY_SCHEMA,
        "status": "promoted",
        "authority": "promoted",
        "release_id": release_id,
        "model_version": model_version.strip(),
        "artifact_sha256": receipt["candidate_artifact_sha256"],
        "receipt_sha256": receipt_sha256,
        "issued_utc": issued_utc,
        "estimand": "prematch_map_win_probability_with_controlled_draft_intervention",
        "probability_authority": True,
        "recommendation_authority": True,
        "betting_authority": False,
        "reason": None,
    }
    return authority, receipt
