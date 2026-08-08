"""Code pin for the system-clocked empty-ledger capture implementation."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_capture_readiness_v3 import (
    DEFAULT_OUTPUT,
    CaptureReadinessV3Error,
    validate_capture_readiness_v3,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_CAPTURE_LOCATOR = DEFAULT_OUTPUT
REGISTERED_CAPTURE_RAW_SHA256 = (
    "7426cb9603b07b8f45dfb78537e4315271436ea20362fcff5b2af2b7b55c0d6e"
)
REGISTERED_CAPTURE_ARTIFACT_SHA256 = (
    "7b744150924ea7a217c524bb71928eb41644f59c1332bbc3de169ba90c18a611"
)
REGISTERED_CAPTURE_LOCKED_AT_UTC = "2026-08-01T23:58:00+00:00"


class CaptureReadinessRegistryV3Error(RuntimeError):
    """The registered system-clocked capture receipt drifted."""


def validate_registered_capture_readiness_v3(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_CAPTURE_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureReadinessRegistryV3Error(
            "registered capture readiness v3 is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_CAPTURE_RAW_SHA256:
        raise CaptureReadinessRegistryV3Error(
            "registered capture readiness v3 raw hash drifted"
        )
    try:
        validated = validate_capture_readiness_v3(payload, root=root)
    except (CaptureReadinessV3Error, OSError, ValueError) as exc:
        raise CaptureReadinessRegistryV3Error(str(exc)) from exc
    if (
        validated.get("artifact_sha256")
        != REGISTERED_CAPTURE_ARTIFACT_SHA256
        or validated.get("locked_at_utc") != REGISTERED_CAPTURE_LOCKED_AT_UTC
    ):
        raise CaptureReadinessRegistryV3Error(
            "registered capture readiness v3 identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_CAPTURE_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise CaptureReadinessRegistryV3Error(
            "registered capture readiness v3 lock time is still in the future"
        )
    return validated


__all__ = [
    "CaptureReadinessRegistryV3Error",
    "REGISTERED_CAPTURE_ARTIFACT_SHA256",
    "REGISTERED_CAPTURE_LOCKED_AT_UTC",
    "REGISTERED_CAPTURE_LOCATOR",
    "REGISTERED_CAPTURE_RAW_SHA256",
    "validate_registered_capture_readiness_v3",
]
