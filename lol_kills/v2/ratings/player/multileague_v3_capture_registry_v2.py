"""Code pin for the clock-corrected empty-ledger capture implementation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_capture_readiness_v2 import (
    DEFAULT_OUTPUT,
    CaptureReadinessV2Error,
    validate_capture_readiness_v2,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_CAPTURE_LOCATOR = DEFAULT_OUTPUT
REGISTERED_CAPTURE_RAW_SHA256 = (
    "a5a0711f80e6234430425e9a6b281c51a3ab1f0367a63487d0cee8463691a0b8"
)
REGISTERED_CAPTURE_ARTIFACT_SHA256 = (
    "02654571004eaae20677e699b57ea65baea8c390d2f2075d0c335621905e0e5e"
)
REGISTERED_CAPTURE_LOCKED_AT_UTC = "2026-08-01T23:56:00+00:00"


class CaptureReadinessRegistryV2Error(RuntimeError):
    """The registered clock-corrected capture receipt drifted."""


def validate_registered_capture_readiness_v2(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_CAPTURE_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureReadinessRegistryV2Error(
            "registered capture readiness v2 is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_CAPTURE_RAW_SHA256:
        raise CaptureReadinessRegistryV2Error(
            "registered capture readiness v2 raw hash drifted"
        )
    try:
        validated = validate_capture_readiness_v2(payload, root=root)
    except (CaptureReadinessV2Error, OSError, ValueError) as exc:
        raise CaptureReadinessRegistryV2Error(str(exc)) from exc
    if (
        validated.get("artifact_sha256")
        != REGISTERED_CAPTURE_ARTIFACT_SHA256
        or validated.get("locked_at_utc") != REGISTERED_CAPTURE_LOCKED_AT_UTC
    ):
        raise CaptureReadinessRegistryV2Error(
            "registered capture readiness v2 identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_CAPTURE_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise CaptureReadinessRegistryV2Error(
            "registered capture readiness v2 lock time is still in the future"
        )
    return validated


__all__ = [
    "CaptureReadinessRegistryV2Error",
    "REGISTERED_CAPTURE_ARTIFACT_SHA256",
    "REGISTERED_CAPTURE_LOCKED_AT_UTC",
    "REGISTERED_CAPTURE_LOCATOR",
    "REGISTERED_CAPTURE_RAW_SHA256",
    "validate_registered_capture_readiness_v2",
]
