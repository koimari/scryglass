"""Code pin for the rejected future-dated v2 receipt lineage."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_temporal_failure_v2 import (
    DEFAULT_OUTPUT,
    TemporalFailureError,
    validate_temporal_failure_receipt,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_FAILURE_LOCATOR = DEFAULT_OUTPUT
REGISTERED_FAILURE_RAW_SHA256 = (
    "138916e1b182bfb265acdcb8a00a91a7212781e6672cc2e7c731e5fb5f0f8c16"
)
REGISTERED_FAILURE_ARTIFACT_SHA256 = (
    "b5aeebf52af22c2407a42a1a8dcf524243f67ec4a6351cb288fd4cd944a9f5cc"
)


class TemporalFailureRegistryError(RuntimeError):
    """The registered temporal-failure evidence drifted."""


def validate_registered_temporal_failure(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / REGISTERED_FAILURE_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemporalFailureRegistryError(
            "registered temporal failure is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_FAILURE_RAW_SHA256:
        raise TemporalFailureRegistryError("registered temporal failure raw hash drifted")
    try:
        validated = validate_temporal_failure_receipt(payload, root=root)
    except (TemporalFailureError, OSError, ValueError) as exc:
        raise TemporalFailureRegistryError(str(exc)) from exc
    if validated.get("artifact_sha256") != REGISTERED_FAILURE_ARTIFACT_SHA256:
        raise TemporalFailureRegistryError(
            "registered temporal failure identity drifted"
        )
    return validated


__all__ = [
    "REGISTERED_FAILURE_ARTIFACT_SHA256",
    "REGISTERED_FAILURE_LOCATOR",
    "REGISTERED_FAILURE_RAW_SHA256",
    "TemporalFailureRegistryError",
    "validate_registered_temporal_failure",
]
