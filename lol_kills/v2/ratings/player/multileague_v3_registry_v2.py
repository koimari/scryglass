"""Code pin for the superseding empty future ratings protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_future_protocol_v2 import (
    DEFAULT_OUTPUT,
    FutureProtocolV2Error,
    validate_future_protocol_lock_v2,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PROTOCOL_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PROTOCOL_RAW_SHA256 = (
    "106a36b0eafa069bee3d48c8e44c70a2e17efa087f4ea41653b812c707510127"
)
REGISTERED_PROTOCOL_ARTIFACT_SHA256 = (
    "49c91c8d1af525514b6125a453185bc41f16717ff60b011712fd7d8ce40425e2"
)


class FutureProtocolRegistryV2Error(RuntimeError):
    """The registered superseding future protocol drifted."""


def validate_registered_future_protocol_v2(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_PROTOCOL_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FutureProtocolRegistryV2Error(
            "registered future protocol v2 is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PROTOCOL_RAW_SHA256:
        raise FutureProtocolRegistryV2Error(
            "registered future protocol v2 raw hash drifted"
        )
    try:
        validated = validate_future_protocol_lock_v2(payload, root=root)
    except (FutureProtocolV2Error, OSError, ValueError) as exc:
        raise FutureProtocolRegistryV2Error(str(exc)) from exc
    if validated.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256:
        raise FutureProtocolRegistryV2Error(
            "registered future protocol v2 identity drifted"
        )
    return validated


__all__ = [
    "FutureProtocolRegistryV2Error",
    "REGISTERED_PROTOCOL_ARTIFACT_SHA256",
    "REGISTERED_PROTOCOL_LOCATOR",
    "REGISTERED_PROTOCOL_RAW_SHA256",
    "validate_registered_future_protocol_v2",
]
