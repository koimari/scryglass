"""Code pin for the empty, future ratings evaluation protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_future_protocol import (
    DEFAULT_OUTPUT,
    FutureProtocolError,
    validate_future_protocol_lock,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PROTOCOL_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PROTOCOL_RAW_SHA256 = (
    "6eb49b1c62ea359d38a38653890c3a52c518b02f3496db490b7f9218f2320c0a"
)
REGISTERED_PROTOCOL_ARTIFACT_SHA256 = (
    "87ebfb6be662820f1597a6583f2bee250dec97f6407523945adfcb2081e69bb4"
)


class FutureProtocolRegistryError(RuntimeError):
    """The code-registered future protocol bytes or semantics drifted."""


def validate_registered_future_protocol(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_PROTOCOL_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FutureProtocolRegistryError("registered future protocol is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PROTOCOL_RAW_SHA256:
        raise FutureProtocolRegistryError("registered future protocol raw hash drifted")
    try:
        validated = validate_future_protocol_lock(payload, root=root)
    except (FutureProtocolError, OSError, ValueError) as exc:
        raise FutureProtocolRegistryError(str(exc)) from exc
    if validated.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256:
        raise FutureProtocolRegistryError(
            "registered future protocol canonical identity drifted"
        )
    return validated


__all__ = [
    "FutureProtocolRegistryError",
    "REGISTERED_PROTOCOL_ARTIFACT_SHA256",
    "REGISTERED_PROTOCOL_LOCATOR",
    "REGISTERED_PROTOCOL_RAW_SHA256",
    "validate_registered_future_protocol",
]
