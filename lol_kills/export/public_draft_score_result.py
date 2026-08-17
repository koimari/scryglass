"""Build one fixed public Draft Score probability result."""

from __future__ import annotations

import math
import re
from datetime import datetime
from typing import Any


SCHEMA_VERSION = "scryglass:public-draft-score-result:v1"
RELEASE_ID_PATTERN = re.compile(r"v[0-9]{4}\.[0-9]{2}\.[0-9]{2}\.[0-9]{6}")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
UTC_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z"
)


class PublicDraftScoreResultError(ValueError):
    """Raised when a promoted public result is not safe to publish."""


def build_public_draft_score_result(
    *,
    release_id: str,
    model_version: str,
    receipt_sha256: str,
    evidence_start: str,
    evidence_end: str,
    blue_win_probability: float,
    controlled_model_units: float,
    controlled_edge_percentage_points: float,
    controlled_explanation: str,
) -> dict[str, Any]:
    """Return the fixed non-betting result envelope."""

    if not RELEASE_ID_PATTERN.fullmatch(release_id):
        raise PublicDraftScoreResultError("release ID is invalid")
    if not model_version.strip():
        raise PublicDraftScoreResultError("model version is empty")
    if not SHA256_PATTERN.fullmatch(receipt_sha256):
        raise PublicDraftScoreResultError("promotion receipt SHA-256 is invalid")
    if not UTC_PATTERN.fullmatch(evidence_start) or not UTC_PATTERN.fullmatch(
        evidence_end
    ):
        raise PublicDraftScoreResultError("evidence window is invalid")
    start = datetime.fromisoformat(evidence_start.removesuffix("Z") + "+00:00")
    end = datetime.fromisoformat(evidence_end.removesuffix("Z") + "+00:00")
    if start >= end:
        raise PublicDraftScoreResultError("evidence window is empty")
    numeric = (
        blue_win_probability,
        controlled_model_units,
        controlled_edge_percentage_points,
    )
    if not all(math.isfinite(value) for value in numeric):
        raise PublicDraftScoreResultError("public Draft result is not finite")
    if not 0.0 <= blue_win_probability <= 1.0:
        raise PublicDraftScoreResultError("win probability is outside zero to one")
    if abs(controlled_edge_percentage_points) > 100.0:
        raise PublicDraftScoreResultError("Draft edge is outside percentage points")
    if (
        controlled_model_units != 0.0
        and controlled_edge_percentage_points == 0.0
    ) or controlled_model_units * controlled_edge_percentage_points < 0.0:
        raise PublicDraftScoreResultError("Draft Score direction is inconsistent")
    if not controlled_explanation.strip():
        raise PublicDraftScoreResultError("Draft Score explanation is empty")
    stronger_draft = (
        "Blue"
        if controlled_model_units > 0.0
        else "Red"
        if controlled_model_units < 0.0
        else "Even"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "promoted",
        "release_id": release_id,
        "model_version": model_version,
        "receipt_sha256": receipt_sha256,
        "evidence_window": {"start": evidence_start, "end": evidence_end},
        "match_win_probability": {
            "Blue": blue_win_probability,
            "Red": 1.0 - blue_win_probability,
        },
        "controlled_draft_score": {
            "model_units": controlled_model_units,
            "edge_percentage_points": controlled_edge_percentage_points,
            "stronger_draft": stronger_draft,
            "explanation": controlled_explanation.strip(),
        },
        "side_recommendation": (
            "Blue" if blue_win_probability >= 0.5 else "Red"
        ),
    }
