"""Code pin for the clock-corrected ratings-source preflight."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_preflight_v2 import (
    CorrectedSourcePreflightError,
    validate_corrected_source_preflight,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PREFLIGHT_LOCATOR = Path(
    "data/lol/v2/models/player/multileague-v3/source-preflight-v3.json"
)
REGISTERED_PREFLIGHT_RAW_SHA256 = (
    "01059e3725c4828a4173a09758640f5fe4974ac726955bc36aa4ff1b012bf931"
)
REGISTERED_PREFLIGHT_ARTIFACT_SHA256 = (
    "ef69122f9123d1415de8644a65fa4e100d527bf0e6f47d6353295c15bfc5c4f9"
)
REGISTERED_PREFLIGHT_BUILT_AT_UTC = "2026-08-01T23:51:00+00:00"


class SourcePreflightRegistryV3Error(RuntimeError):
    """The clock-corrected source preflight drifted."""


def validate_registered_source_preflight_v3(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_PREFLIGHT_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePreflightRegistryV3Error(
            "registered clock-corrected preflight is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PREFLIGHT_RAW_SHA256:
        raise SourcePreflightRegistryV3Error(
            "registered clock-corrected preflight raw hash drifted"
        )
    try:
        validated = validate_corrected_source_preflight(payload, root=root)
    except (CorrectedSourcePreflightError, OSError, ValueError) as exc:
        raise SourcePreflightRegistryV3Error(str(exc)) from exc
    if (
        validated.get("artifact_sha256")
        != REGISTERED_PREFLIGHT_ARTIFACT_SHA256
        or validated.get("built_at_utc") != REGISTERED_PREFLIGHT_BUILT_AT_UTC
    ):
        raise SourcePreflightRegistryV3Error(
            "registered clock-corrected preflight identity drifted"
        )
    built = datetime.fromisoformat(REGISTERED_PREFLIGHT_BUILT_AT_UTC)
    if built > datetime.now(timezone.utc):
        raise SourcePreflightRegistryV3Error(
            "registered preflight build time is still in the future"
        )
    return validated


__all__ = [
    "REGISTERED_PREFLIGHT_ARTIFACT_SHA256",
    "REGISTERED_PREFLIGHT_BUILT_AT_UTC",
    "REGISTERED_PREFLIGHT_LOCATOR",
    "REGISTERED_PREFLIGHT_RAW_SHA256",
    "SourcePreflightRegistryV3Error",
    "validate_registered_source_preflight_v3",
]
