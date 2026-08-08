"""Code pin for the non-authorizing Betano quote-adapter candidate.

This pin makes candidate drift visible.  It is not the independent adapter
registry required by the market protocol and grants no authority.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .betano_br_quote_adapter_v1 import (
    ADAPTER_ID,
    DEFAULT_CANDIDATE_OUTPUT,
    BetanoQuoteAdapterError,
    validate_betano_quote_adapter_candidate_v1,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTERED_CANDIDATE_LOCATOR = DEFAULT_CANDIDATE_OUTPUT
REGISTERED_CANDIDATE_RAW_SHA256 = (
    "8de372b4cc5525a0d41f6aa91c5e302e19f0b0781e165165dde5716bed2bedd1"
)
REGISTERED_CANDIDATE_ARTIFACT_SHA256 = (
    "b1fb79a56192c82646d1a7c2f6eed09e88e8ccf4018a2e406aeedcff5cec10a7"
)
REGISTERED_CANDIDATE_LOCKED_AT_UTC = "2026-08-02T05:00:00+00:00"
REGISTERED_ADAPTER_SOURCE_SHA256 = (
    "ef7d011b358c1604c3dde2a570ade988e3dadaaa60a8753f5e3fa90d4d60f2ef"
)


class BetanoQuoteAdapterCandidateRegistryError(RuntimeError):
    """The code-pinned adapter candidate drifted or disappeared."""


def validate_registered_betano_quote_adapter_candidate_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_CANDIDATE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise BetanoQuoteAdapterCandidateRegistryError(
            "registered adapter candidate is unavailable"
        )
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetanoQuoteAdapterCandidateRegistryError(
            "registered adapter candidate is invalid"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_CANDIDATE_RAW_SHA256:
        raise BetanoQuoteAdapterCandidateRegistryError(
            "registered adapter candidate raw hash drifted"
        )
    try:
        checked = validate_betano_quote_adapter_candidate_v1(payload, root=root)
    except (BetanoQuoteAdapterError, OSError, ValueError) as exc:
        raise BetanoQuoteAdapterCandidateRegistryError(str(exc)) from exc
    if (
        checked.get("artifact_sha256")
        != REGISTERED_CANDIDATE_ARTIFACT_SHA256
        or checked.get("locked_at_utc") != REGISTERED_CANDIDATE_LOCKED_AT_UTC
        or checked.get("adapter_contract", {}).get("adapter_id") != ADAPTER_ID
        or checked.get("source_lock", {}).get("raw_sha256")
        != REGISTERED_ADAPTER_SOURCE_SHA256
    ):
        raise BetanoQuoteAdapterCandidateRegistryError(
            "registered adapter candidate identity drifted"
        )
    locked_at = datetime.fromisoformat(REGISTERED_CANDIDATE_LOCKED_AT_UTC)
    if locked_at > datetime.now(timezone.utc):
        raise BetanoQuoteAdapterCandidateRegistryError(
            "registered adapter candidate lock is in the future"
        )
    return checked


__all__ = [
    "BetanoQuoteAdapterCandidateRegistryError",
    "REGISTERED_ADAPTER_SOURCE_SHA256",
    "REGISTERED_CANDIDATE_ARTIFACT_SHA256",
    "REGISTERED_CANDIDATE_LOCATOR",
    "REGISTERED_CANDIDATE_LOCKED_AT_UTC",
    "REGISTERED_CANDIDATE_RAW_SHA256",
    "validate_registered_betano_quote_adapter_candidate_v1",
]
