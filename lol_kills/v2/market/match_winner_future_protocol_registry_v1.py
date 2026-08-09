"""Code pin for the empty two-stage map-winner market protocol."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .match_winner_future_protocol_v1 import (
    DEFAULT_OUTPUT,
    MatchWinnerFutureProtocolError,
    validate_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTERED_PROTOCOL_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PROTOCOL_RAW_SHA256 = (
    "9a0d9df2a68651dfb311200cedf21dd8e3702fa94e97f639ec9ab85362033b9e"
)
REGISTERED_PROTOCOL_ARTIFACT_SHA256 = (
    "32f11f7981d850cd46a953c8b751340fbf90993cbe95a34aed7fc7e6e9228b8a"
)
REGISTERED_PROTOCOL_LOCKED_AT_UTC = "2026-08-02T02:00:00+00:00"
REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256 = (
    "06dc090de7c93d2625bc78c3ba7163eafda598e3d14deaea3f325c829cb55c68"
)
REGISTERED_SETTLEMENT_CONTRACT_SHA256 = (
    "6f1e885f7d49ee27555bdad7babb9579c8ee9b5057951e728957d44dbe253405"
)


class MatchWinnerFutureProtocolRegistryError(RuntimeError):
    """The registered map-winner future protocol drifted."""


def validate_registered_match_winner_future_protocol_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_PROTOCOL_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MatchWinnerFutureProtocolRegistryError(
            "registered map-winner future protocol is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PROTOCOL_RAW_SHA256:
        raise MatchWinnerFutureProtocolRegistryError(
            "registered map-winner future protocol raw hash drifted"
        )
    try:
        checked = validate_match_winner_future_protocol_v1(payload, root=root)
    except (MatchWinnerFutureProtocolError, OSError, ValueError) as exc:
        raise MatchWinnerFutureProtocolRegistryError(str(exc)) from exc
    if (
        checked.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or checked.get("locked_at_utc") != REGISTERED_PROTOCOL_LOCKED_AT_UTC
        or checked.get("quote_capture_contract_sha256")
        != REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256
        or checked.get("settlement_contract_sha256")
        != REGISTERED_SETTLEMENT_CONTRACT_SHA256
    ):
        raise MatchWinnerFutureProtocolRegistryError(
            "registered map-winner future protocol identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_PROTOCOL_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise MatchWinnerFutureProtocolRegistryError(
            "registered map-winner future protocol lock time is in the future"
        )
    return checked


__all__ = [
    "MatchWinnerFutureProtocolRegistryError",
    "REGISTERED_PROTOCOL_ARTIFACT_SHA256",
    "REGISTERED_PROTOCOL_LOCKED_AT_UTC",
    "REGISTERED_PROTOCOL_LOCATOR",
    "REGISTERED_PROTOCOL_RAW_SHA256",
    "REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256",
    "REGISTERED_SETTLEMENT_CONTRACT_SHA256",
    "validate_registered_match_winner_future_protocol_v1",
]

