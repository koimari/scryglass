"""Code pin for the clock-corrected empty future ratings protocol."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_future_protocol_v3 import (
    DEFAULT_OUTPUT,
    FutureProtocolV3Error,
    validate_future_protocol_lock_v3,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PROTOCOL_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PROTOCOL_RAW_SHA256 = (
    "db9f2c1557247036ddfdbebdd7cccbd395cc22841cd9d574747fe4f60f446301"
)
REGISTERED_PROTOCOL_ARTIFACT_SHA256 = (
    "996ce4067818799e7345cd49e6a9e450a7675563f3e15daeaa667a8c9846ef24"
)
REGISTERED_PROTOCOL_LOCKED_AT_UTC = "2026-08-01T23:54:00+00:00"


class FutureProtocolRegistryV3Error(RuntimeError):
    """The registered clock-corrected future protocol drifted."""


def validate_registered_future_protocol_v3(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_PROTOCOL_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FutureProtocolRegistryV3Error(
            "registered future protocol v3 is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PROTOCOL_RAW_SHA256:
        raise FutureProtocolRegistryV3Error(
            "registered future protocol v3 raw hash drifted"
        )
    try:
        validated = validate_future_protocol_lock_v3(payload, root=root)
    except (FutureProtocolV3Error, OSError, ValueError) as exc:
        raise FutureProtocolRegistryV3Error(str(exc)) from exc
    if (
        validated.get("artifact_sha256")
        != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or validated.get("locked_at_utc") != REGISTERED_PROTOCOL_LOCKED_AT_UTC
    ):
        raise FutureProtocolRegistryV3Error(
            "registered future protocol v3 identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_PROTOCOL_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise FutureProtocolRegistryV3Error(
            "registered future protocol v3 lock time is still in the future"
        )
    return validated


__all__ = [
    "FutureProtocolRegistryV3Error",
    "REGISTERED_PROTOCOL_ARTIFACT_SHA256",
    "REGISTERED_PROTOCOL_LOCKED_AT_UTC",
    "REGISTERED_PROTOCOL_LOCATOR",
    "REGISTERED_PROTOCOL_RAW_SHA256",
    "validate_registered_future_protocol_v3",
]
