"""Code pin for the empty phase-one collection readiness receipt."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .phase_one_collection_readiness_v1 import (
    DEFAULT_OUTPUT,
    PhaseOneCollectionReadinessError,
    validate_phase_one_collection_readiness_v1,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTERED_READINESS_LOCATOR = DEFAULT_OUTPUT
REGISTERED_READINESS_RAW_SHA256 = (
    "54ae822db49d735a84d160e794c5c463fb860dc1e7f9a2170c759375b2ecdf0a"
)
REGISTERED_READINESS_ARTIFACT_SHA256 = (
    "854a12258ca1389b8dae3789bd654dd6ddad76c16bfcbf573c43854c79bb4fe7"
)
REGISTERED_READINESS_LOCKED_AT_UTC = "2026-08-02T03:30:00+00:00"


class PhaseOneCollectionReadinessRegistryError(RuntimeError):
    """The registered phase-one collection readiness receipt drifted."""


def validate_registered_phase_one_collection_readiness_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_READINESS_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseOneCollectionReadinessRegistryError(
            "registered phase-one collection readiness is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_READINESS_RAW_SHA256:
        raise PhaseOneCollectionReadinessRegistryError(
            "registered phase-one collection readiness raw hash drifted"
        )
    try:
        checked = validate_phase_one_collection_readiness_v1(
            payload, root=root
        )
    except (PhaseOneCollectionReadinessError, OSError, ValueError) as exc:
        raise PhaseOneCollectionReadinessRegistryError(str(exc)) from exc
    if (
        checked.get("artifact_sha256")
        != REGISTERED_READINESS_ARTIFACT_SHA256
        or checked.get("locked_at_utc")
        != REGISTERED_READINESS_LOCKED_AT_UTC
    ):
        raise PhaseOneCollectionReadinessRegistryError(
            "registered phase-one collection readiness identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_READINESS_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise PhaseOneCollectionReadinessRegistryError(
            "registered phase-one collection readiness lock time is in the future"
        )
    return checked


__all__ = [
    "PhaseOneCollectionReadinessRegistryError",
    "REGISTERED_READINESS_ARTIFACT_SHA256",
    "REGISTERED_READINESS_LOCATOR",
    "REGISTERED_READINESS_LOCKED_AT_UTC",
    "REGISTERED_READINESS_RAW_SHA256",
    "validate_registered_phase_one_collection_readiness_v1",
]
