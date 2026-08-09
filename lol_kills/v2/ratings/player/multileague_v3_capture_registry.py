"""Code pin for the empty-ledger pre-event capture implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_capture_readiness import (
    DEFAULT_OUTPUT,
    CaptureReadinessError,
    validate_capture_readiness,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_CAPTURE_LOCATOR = DEFAULT_OUTPUT
REGISTERED_CAPTURE_RAW_SHA256 = (
    "75a84655d952f868dcec7eb698e95aab76d105ba03e490c9c7a2186d46cbfd80"
)
REGISTERED_CAPTURE_ARTIFACT_SHA256 = (
    "593f8abb0e4dae2774462ebe4195485990fe07db72ffc31e9622592c57113114"
)


class CaptureReadinessRegistryError(RuntimeError):
    """The registered pre-event capture implementation drifted."""


def validate_registered_capture_readiness(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_CAPTURE_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CaptureReadinessRegistryError(
            "registered capture readiness is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_CAPTURE_RAW_SHA256:
        raise CaptureReadinessRegistryError(
            "registered capture readiness raw hash drifted"
        )
    try:
        validated = validate_capture_readiness(payload, root=root)
    except (CaptureReadinessError, OSError, ValueError) as exc:
        raise CaptureReadinessRegistryError(str(exc)) from exc
    if validated.get("artifact_sha256") != REGISTERED_CAPTURE_ARTIFACT_SHA256:
        raise CaptureReadinessRegistryError(
            "registered capture readiness identity drifted"
        )
    return validated


__all__ = [
    "CaptureReadinessRegistryError",
    "REGISTERED_CAPTURE_ARTIFACT_SHA256",
    "REGISTERED_CAPTURE_LOCATOR",
    "REGISTERED_CAPTURE_RAW_SHA256",
    "validate_registered_capture_readiness",
]
