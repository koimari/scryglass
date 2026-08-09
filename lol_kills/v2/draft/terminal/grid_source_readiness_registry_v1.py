"""Code pin for the private GRID terminal-draft source readiness receipt."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .grid_source_readiness_v1 import (
    DEFAULT_OUTPUT,
    GridSourceReadinessError,
    validate_grid_source_readiness_v1,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_GRID_SOURCE_LOCATOR = DEFAULT_OUTPUT
REGISTERED_GRID_SOURCE_RAW_SHA256 = (
    "1f06b057adac8c0e35c15dfed40cca01681bf34ab7be61dc780b0b968ae1440c"
)
REGISTERED_GRID_SOURCE_ARTIFACT_SHA256 = (
    "2aa23f24e281fc34926e96851ce5aff36046c31ebef857b41bc8910e51cc732c"
)
REGISTERED_GRID_SOURCE_LOCKED_AT_UTC = "2026-08-02T01:27:57.849474+00:00"


class GridSourceReadinessRegistryError(RuntimeError):
    """The registered private GRID source readiness receipt drifted."""


def validate_registered_grid_source_readiness_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_GRID_SOURCE_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GridSourceReadinessRegistryError(
            "registered GRID source readiness is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_GRID_SOURCE_RAW_SHA256:
        raise GridSourceReadinessRegistryError(
            "registered GRID source readiness raw hash drifted"
        )
    try:
        checked = validate_grid_source_readiness_v1(payload, root=root)
    except (GridSourceReadinessError, OSError, ValueError) as exc:
        raise GridSourceReadinessRegistryError(str(exc)) from exc
    if (
        checked.get("artifact_sha256")
        != REGISTERED_GRID_SOURCE_ARTIFACT_SHA256
        or checked.get("locked_at_utc") != REGISTERED_GRID_SOURCE_LOCKED_AT_UTC
    ):
        raise GridSourceReadinessRegistryError(
            "registered GRID source readiness identity drifted"
        )
    locked = datetime.fromisoformat(REGISTERED_GRID_SOURCE_LOCKED_AT_UTC)
    if locked > datetime.now(timezone.utc):
        raise GridSourceReadinessRegistryError(
            "registered GRID source readiness lock time is in the future"
        )
    return checked


__all__ = [
    "GridSourceReadinessRegistryError",
    "REGISTERED_GRID_SOURCE_ARTIFACT_SHA256",
    "REGISTERED_GRID_SOURCE_LOCKED_AT_UTC",
    "REGISTERED_GRID_SOURCE_LOCATOR",
    "REGISTERED_GRID_SOURCE_RAW_SHA256",
    "validate_registered_grid_source_readiness_v1",
]
