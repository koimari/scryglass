"""Repository code pin for the side-neutral protocol candidate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_side_neutral_protocol_v1 import (
    DEFAULT_OUTPUT,
    SideNeutralProtocolError,
    validate_side_neutral_protocol_lock,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PROTOCOL_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PROTOCOL_RAW_SHA256 = (
    "2bef4831c615e04e39dd5035cf794b48e62420fc1572e18a5cd5ead1ca693c49"
)
REGISTERED_PROTOCOL_ARTIFACT_SHA256 = (
    "f2f55cac87ba3b1bbb0b1a38f71b6e38d28b3dba2198b52e7f3f63124871f744"
)
REGISTERED_PROTOCOL_LOCKED_AT_UTC = "2026-08-02T00:01:00+00:00"


class SideNeutralProtocolRegistryError(RuntimeError):
    """The repository-pinned side-neutral protocol candidate drifted."""


def validate_registered_side_neutral_protocol(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_PROTOCOL_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralProtocolRegistryError(
            "registered side-neutral protocol is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PROTOCOL_RAW_SHA256:
        raise SideNeutralProtocolRegistryError(
            "registered side-neutral protocol raw hash drifted"
        )
    try:
        checked = validate_side_neutral_protocol_lock(payload, root=root)
    except (SideNeutralProtocolError, OSError, ValueError) as exc:
        raise SideNeutralProtocolRegistryError(str(exc)) from exc
    if (
        checked.get("artifact_sha256")
        != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or checked.get("locked_at_utc") != REGISTERED_PROTOCOL_LOCKED_AT_UTC
    ):
        raise SideNeutralProtocolRegistryError(
            "registered side-neutral protocol identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_PROTOCOL_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise SideNeutralProtocolRegistryError(
            "registered side-neutral protocol lock is still in the future"
        )
    return checked


__all__ = [
    "REGISTERED_PROTOCOL_ARTIFACT_SHA256",
    "REGISTERED_PROTOCOL_LOCATOR",
    "REGISTERED_PROTOCOL_LOCKED_AT_UTC",
    "REGISTERED_PROTOCOL_RAW_SHA256",
    "SideNeutralProtocolRegistryError",
    "validate_registered_side_neutral_protocol",
]
