"""Independent human authority root for private OE target experiments.

This verifier is deliberately outside the target-evidence generator's
content-addressed dependency boundary.  An approval envelope binds the exact
evidence and split payloads; this module pins the envelope bytes without
creating a hash cycle through those payloads.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .oe_target_evidence import OETargetEvidenceError


DEFAULT_HUMAN_AUTHORITY_PATH = Path(
    "data/lol/v2/models/draft-interactions/oe-private-target-authority.json"
)

# Independently reviewed by KOI_MARI on 2026-07-29. This root is outside the
# evidence generator boundary, so pinning it cannot change the approved
# evidence or split payloads.
PINNED_HUMAN_AUTHORITY_ENVELOPE_RAW_SHA256: str | None = (
    "b0aac33a7d23f05daa14cf8a769fa4cafc44bb15d6a165ec64d5542a49e937d5"
)


def require_exact_human_authority(
    envelope_bytes: bytes | None,
    evidence: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    action: str,
) -> Mapping[str, Any]:
    """Require an exact independently reviewed and pinned approval envelope."""
    if envelope_bytes is None:
        raise OETargetEvidenceError("human authority envelope is missing")
    try:
        envelope = json.loads(envelope_bytes)
    except json.JSONDecodeError as exc:
        raise OETargetEvidenceError("human authority envelope is invalid") from exc
    required_exact = {
        "schema_id": "scryglass.oe-private-target-human-authority.v1",
        "approval_scope": "private_retrospective_oe_target_v1",
        "decision": "approve",
        "source_rights_reviewed": True,
        "target_semantics_reviewed": True,
        "temporal_leakage_reviewed": True,
        "fixed_boundaries_reviewed": True,
    }
    if any(envelope.get(key) != value for key, value in required_exact.items()):
        raise OETargetEvidenceError(
            "human authority envelope review fields are incomplete"
        )
    for field in ("decision_id", "reviewer_identity", "reviewed_at_rfc3339"):
        if not isinstance(envelope.get(field), str) or not envelope[field].strip():
            raise OETargetEvidenceError(
                "human authority envelope identity fields are incomplete"
            )
    reviewed_at = pd.Timestamp(envelope["reviewed_at_rfc3339"])
    if pd.isna(reviewed_at) or reviewed_at.tzinfo is None:
        raise OETargetEvidenceError(
            "human review timestamp must be timezone-aware"
        )
    if (
        envelope.get("generator_authored") is not False
        or envelope.get("independent_from_generator") is not True
    ):
        raise OETargetEvidenceError("generator-authored authority is forbidden")
    if (
        envelope.get("evidence_payload_sha256")
        != evidence.get("artifact_sha256")
        or envelope.get("split_payload_sha256")
        != split.get("artifact_sha256")
    ):
        raise OETargetEvidenceError(
            "human authority envelope does not bind exact evidence"
        )
    approved_actions = envelope.get("approved_actions")
    if not isinstance(approved_actions, list) or action not in approved_actions:
        raise OETargetEvidenceError("requested action lacks human approval")
    if PINNED_HUMAN_AUTHORITY_ENVELOPE_RAW_SHA256 is None:
        raise OETargetEvidenceError(
            "independent human authority envelope hash is not pinned"
        )
    if (
        hashlib.sha256(envelope_bytes).hexdigest()
        != PINNED_HUMAN_AUTHORITY_ENVELOPE_RAW_SHA256
    ):
        raise OETargetEvidenceError(
            "caller-rehashed authority envelope rejected"
        )
    return envelope


def load_and_require_exact_human_authority(
    evidence: Mapping[str, Any],
    split: Mapping[str, Any],
    *,
    action: str,
    authority_path: Path = DEFAULT_HUMAN_AUTHORITY_PATH,
) -> Mapping[str, Any]:
    """Load the pinned non-symlink authority envelope and verify one action."""
    if not authority_path.is_file() or authority_path.is_symlink():
        raise OETargetEvidenceError(
            "human authority envelope must be a regular non-symlink file"
        )
    return require_exact_human_authority(
        authority_path.read_bytes(),
        evidence,
        split,
        action=action,
    )
