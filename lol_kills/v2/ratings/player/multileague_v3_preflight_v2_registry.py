"""Code pin for the corrected, non-authorizing ratings-source preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_preflight_v2 import (
    DEFAULT_OUTPUT,
    CorrectedSourcePreflightError,
    validate_corrected_source_preflight,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PREFLIGHT_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PREFLIGHT_RAW_SHA256 = (
    "e86f9c660278bef68d6abe873756050180603da7dfaed07edb5b605ce67f493b"
)
REGISTERED_PREFLIGHT_ARTIFACT_SHA256 = (
    "bb2abaa30f657d15345796a076dfb19a7e116ed235030d29b82b625896b9c6ec"
)


class SourcePreflightRegistryV2Error(RuntimeError):
    """The registered corrected-source preflight drifted."""


def validate_registered_source_preflight_v2(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_PREFLIGHT_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePreflightRegistryV2Error(
            "registered v2 source preflight is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PREFLIGHT_RAW_SHA256:
        raise SourcePreflightRegistryV2Error(
            "registered v2 source preflight raw hash drifted"
        )
    try:
        validated = validate_corrected_source_preflight(payload, root=root)
    except (CorrectedSourcePreflightError, OSError, ValueError) as exc:
        raise SourcePreflightRegistryV2Error(str(exc)) from exc
    if validated.get("artifact_sha256") != REGISTERED_PREFLIGHT_ARTIFACT_SHA256:
        raise SourcePreflightRegistryV2Error(
            "registered v2 source preflight identity drifted"
        )
    return validated


__all__ = [
    "REGISTERED_PREFLIGHT_ARTIFACT_SHA256",
    "REGISTERED_PREFLIGHT_LOCATOR",
    "REGISTERED_PREFLIGHT_RAW_SHA256",
    "SourcePreflightRegistryV2Error",
    "validate_registered_source_preflight_v2",
]
