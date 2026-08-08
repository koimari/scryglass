"""Repository code pin for the complete side-neutral protocol candidate."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_side_neutral_protocol_v2 import (
    DEFAULT_OUTPUT,
    SideNeutralProtocolV2Error,
    validate_side_neutral_protocol_v2,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PROTOCOL_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PROTOCOL_RAW_SHA256 = (
    "0861d751172e032c961553db2e1cde0a83c42b41d8b3e29de79da2121e46f6d5"
)
REGISTERED_PROTOCOL_ARTIFACT_SHA256 = (
    "755538eacecb404437175e27ad9c3c9efe2a6838309c783a79e3f13b9ac67058"
)
REGISTERED_PROTOCOL_LOCKED_AT_UTC = "2026-08-02T00:02:00+00:00"


class SideNeutralProtocolRegistryV2Error(RuntimeError):
    """The repository-pinned complete side-neutral protocol drifted."""


def validate_registered_side_neutral_protocol_v2(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_PROTOCOL_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralProtocolRegistryV2Error(
            "registered side-neutral protocol v2 is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PROTOCOL_RAW_SHA256:
        raise SideNeutralProtocolRegistryV2Error(
            "registered side-neutral protocol v2 raw hash drifted"
        )
    try:
        checked = validate_side_neutral_protocol_v2(payload, root=root)
    except (SideNeutralProtocolV2Error, OSError, ValueError) as exc:
        raise SideNeutralProtocolRegistryV2Error(str(exc)) from exc
    if (
        checked.get("artifact_sha256")
        != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or checked.get("locked_at_utc") != REGISTERED_PROTOCOL_LOCKED_AT_UTC
    ):
        raise SideNeutralProtocolRegistryV2Error(
            "registered side-neutral protocol v2 identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_PROTOCOL_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise SideNeutralProtocolRegistryV2Error(
            "registered side-neutral protocol v2 lock is still in the future"
        )
    return checked


__all__ = [
    "REGISTERED_PROTOCOL_ARTIFACT_SHA256",
    "REGISTERED_PROTOCOL_LOCATOR",
    "REGISTERED_PROTOCOL_LOCKED_AT_UTC",
    "REGISTERED_PROTOCOL_RAW_SHA256",
    "SideNeutralProtocolRegistryV2Error",
    "validate_registered_side_neutral_protocol_v2",
]
