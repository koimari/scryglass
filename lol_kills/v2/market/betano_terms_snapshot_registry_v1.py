"""Code pin for the incomplete public Betano terms snapshot."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .betano_terms_snapshot_v1 import (
    DEFAULT_OUTPUT,
    BetanoTermsSnapshotError,
    validate_betano_terms_snapshot_v1,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTERED_SNAPSHOT_LOCATOR = DEFAULT_OUTPUT
REGISTERED_SNAPSHOT_RAW_SHA256 = (
    "96f5c3228f7b8b1804cd764012677f912a2654da00f3684ec5d1a37bd3a45255"
)
REGISTERED_SNAPSHOT_ARTIFACT_SHA256 = (
    "56407922787b0a954d90447a6ebb10c164005de74d8ad214c0c2520a53b7d9d5"
)
REGISTERED_SNAPSHOT_LOCKED_AT_UTC = "2026-08-02T02:13:59.564289+00:00"


class BetanoTermsSnapshotRegistryError(RuntimeError):
    """The registered incomplete public terms snapshot drifted."""


def validate_registered_betano_terms_snapshot_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_SNAPSHOT_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BetanoTermsSnapshotRegistryError(
            "registered public Betano terms snapshot is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_SNAPSHOT_RAW_SHA256:
        raise BetanoTermsSnapshotRegistryError(
            "registered public Betano terms snapshot raw hash drifted"
        )
    try:
        checked = validate_betano_terms_snapshot_v1(payload, root=root)
    except (BetanoTermsSnapshotError, OSError, ValueError) as exc:
        raise BetanoTermsSnapshotRegistryError(str(exc)) from exc
    if (
        checked.get("artifact_sha256") != REGISTERED_SNAPSHOT_ARTIFACT_SHA256
        or checked.get("locked_at_utc") != REGISTERED_SNAPSHOT_LOCKED_AT_UTC
        or (checked.get("coverage") or {}).get("complete_bookmaker_terms_snapshot")
        is not False
        or (checked.get("coverage") or {}).get(
            "independent_alignment_review_present"
        )
        is not False
    ):
        raise BetanoTermsSnapshotRegistryError(
            "registered public Betano terms snapshot identity changed"
        )
    if datetime.fromisoformat(REGISTERED_SNAPSHOT_LOCKED_AT_UTC) > datetime.now(
        timezone.utc
    ):
        raise BetanoTermsSnapshotRegistryError(
            "registered public Betano terms snapshot is from the future"
        )
    return checked


__all__ = [
    "BetanoTermsSnapshotRegistryError",
    "REGISTERED_SNAPSHOT_ARTIFACT_SHA256",
    "REGISTERED_SNAPSHOT_LOCATOR",
    "REGISTERED_SNAPSHOT_LOCKED_AT_UTC",
    "REGISTERED_SNAPSHOT_RAW_SHA256",
    "validate_registered_betano_terms_snapshot_v1",
]

