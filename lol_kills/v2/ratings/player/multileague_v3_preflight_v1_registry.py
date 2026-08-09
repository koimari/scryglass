"""Code pin for the reproducible v1 ratings-source schema failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_preflight_v1 import (
    DEFAULT_OUTPUT,
    SourcePreflightError,
    validate_source_preflight_failure,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_PREFLIGHT_LOCATOR = DEFAULT_OUTPUT
REGISTERED_PREFLIGHT_RAW_SHA256 = (
    "fca0fb5d591f4ec88fb96e026fc6c243ca0874ecb30804a5fd7ce4044ce956c7"
)
REGISTERED_PREFLIGHT_ARTIFACT_SHA256 = (
    "10f7480bc82fa0711498280ac7318fb5e0a1c9d2cfc233b7b56b94122a3351fd"
)


class SourcePreflightRegistryV1Error(RuntimeError):
    """The registered v1 source-failure evidence drifted."""


def validate_registered_source_preflight_v1(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_PREFLIGHT_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourcePreflightRegistryV1Error(
            "registered v1 source preflight is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_PREFLIGHT_RAW_SHA256:
        raise SourcePreflightRegistryV1Error(
            "registered v1 source preflight raw hash drifted"
        )
    try:
        validated = validate_source_preflight_failure(payload, root=root)
    except (SourcePreflightError, OSError, ValueError) as exc:
        raise SourcePreflightRegistryV1Error(str(exc)) from exc
    if validated.get("artifact_sha256") != REGISTERED_PREFLIGHT_ARTIFACT_SHA256:
        raise SourcePreflightRegistryV1Error(
            "registered v1 source preflight identity drifted"
        )
    return validated


__all__ = [
    "REGISTERED_PREFLIGHT_ARTIFACT_SHA256",
    "REGISTERED_PREFLIGHT_LOCATOR",
    "REGISTERED_PREFLIGHT_RAW_SHA256",
    "SourcePreflightRegistryV1Error",
    "validate_registered_source_preflight_v1",
]
