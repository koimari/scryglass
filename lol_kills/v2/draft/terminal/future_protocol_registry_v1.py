"""Code pin for the empty terminal Draft Score future protocol."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .future_protocol_v1 import (
    DEFAULT_OUTPUT,
    DraftFutureProtocolError,
    validate_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PROTOCOL_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PROTOCOL_RAW_SHA256 = (
    "46826584081e8f9e8bd177c96d7eaf5c2239dba5d9d33b94a50dafe7b3b10a87"
)
REGISTERED_PROTOCOL_ARTIFACT_SHA256 = (
    "c4c9ceb0fbdb3943f684de4983d28d2021fecb5d1c92069c514ff304c9f722f8"
)
REGISTERED_PROTOCOL_LOCKED_AT_UTC = "2026-08-02T01:00:00+00:00"


class DraftFutureProtocolRegistryError(ValueError):
    """The registered terminal Draft Score future protocol drifted."""


def validate_registered_future_protocol_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_PROTOCOL_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DraftFutureProtocolRegistryError(
            "registered terminal future protocol is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PROTOCOL_RAW_SHA256:
        raise DraftFutureProtocolRegistryError(
            "registered terminal future protocol raw hash drifted"
        )
    try:
        validated = validate_future_protocol_v1(payload, root=root)
    except (OSError, ValueError, DraftFutureProtocolError) as exc:
        raise DraftFutureProtocolRegistryError(str(exc)) from exc
    if (
        validated.get("artifact_sha256")
        != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or validated.get("locked_at_utc") != REGISTERED_PROTOCOL_LOCKED_AT_UTC
    ):
        raise DraftFutureProtocolRegistryError(
            "registered terminal future protocol identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_PROTOCOL_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise DraftFutureProtocolRegistryError(
            "registered terminal future protocol lock time is in the future"
        )
    return validated


__all__ = [
    "DraftFutureProtocolRegistryError",
    "REGISTERED_PROTOCOL_ARTIFACT_SHA256",
    "REGISTERED_PROTOCOL_LOCKED_AT_UTC",
    "REGISTERED_PROTOCOL_LOCATOR",
    "REGISTERED_PROTOCOL_RAW_SHA256",
    "validate_registered_future_protocol_v1",
]

