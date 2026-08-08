"""Code pin for the terminal Draft Score future-capture readiness receipt."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .capture_readiness_v1 import (
    DEFAULT_OUTPUT,
    DraftCaptureReadinessError,
    validate_capture_readiness_v1,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_CAPTURE_LOCATOR = DEFAULT_OUTPUT
REGISTERED_CAPTURE_RAW_SHA256 = (
    "84f66287df739ddc57a643ce5612a7933c4ae9caa87835c360ac4c838ed8c11f"
)
REGISTERED_CAPTURE_ARTIFACT_SHA256 = (
    "d60f1d5db16aa06d8e5686c754c74c67905f4c9cc5f528e928328b7b3bc23514"
)
REGISTERED_CAPTURE_LOCKED_AT_UTC = "2026-08-02T01:30:00+00:00"


class DraftCaptureReadinessRegistryError(RuntimeError):
    """The registered terminal Draft Score capture receipt drifted."""


def validate_registered_capture_readiness_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_CAPTURE_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DraftCaptureReadinessRegistryError(
            "registered Draft capture readiness is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_CAPTURE_RAW_SHA256:
        raise DraftCaptureReadinessRegistryError(
            "registered Draft capture readiness raw hash drifted"
        )
    try:
        validated = validate_capture_readiness_v1(payload, root=root)
    except (DraftCaptureReadinessError, OSError, ValueError) as exc:
        raise DraftCaptureReadinessRegistryError(str(exc)) from exc
    if (
        validated.get("artifact_sha256")
        != REGISTERED_CAPTURE_ARTIFACT_SHA256
        or validated.get("locked_at_utc") != REGISTERED_CAPTURE_LOCKED_AT_UTC
    ):
        raise DraftCaptureReadinessRegistryError(
            "registered Draft capture readiness identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_CAPTURE_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise DraftCaptureReadinessRegistryError(
            "registered Draft capture readiness lock time is in the future"
        )
    return validated


__all__ = [
    "DraftCaptureReadinessRegistryError",
    "REGISTERED_CAPTURE_ARTIFACT_SHA256",
    "REGISTERED_CAPTURE_LOCKED_AT_UTC",
    "REGISTERED_CAPTURE_LOCATOR",
    "REGISTERED_CAPTURE_RAW_SHA256",
    "validate_registered_capture_readiness_v1",
]
