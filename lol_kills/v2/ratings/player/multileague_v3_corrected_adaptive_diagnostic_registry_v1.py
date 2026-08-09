"""Code pin for the corrected-source adaptive candidate diagnostic."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .multileague_v3_corrected_adaptive_diagnostic_v1 import (
    CorrectedAdaptiveDiagnosticError,
    DEFAULT_OUTPUT,
    validate_corrected_adaptive_diagnostic,
)


ROOT = Path(__file__).resolve().parents[4]
REGISTERED_DIAGNOSTIC_LOCATOR = DEFAULT_OUTPUT
REGISTERED_DIAGNOSTIC_RAW_SHA256 = (
    "b43250715f07db664d985228f72ee75987e4eb00366596ec13f41e4985bb6577"
)
REGISTERED_DIAGNOSTIC_ARTIFACT_SHA256 = (
    "b9b2987bf0a897810a1785bd9caff3e907c8a88b75e036e85803ad2d88ad5292"
)
REGISTERED_DIAGNOSTIC_BUILT_AT_UTC = "2026-08-01T23:55:00+00:00"


class CorrectedAdaptiveDiagnosticRegistryError(RuntimeError):
    """The registered diagnostic no longer replays its frozen identity."""


def validate_registered_corrected_adaptive_diagnostic_v1(
    *, root: Path = ROOT
) -> dict[str, Any]:
    path = root / REGISTERED_DIAGNOSTIC_LOCATOR
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CorrectedAdaptiveDiagnosticRegistryError(
            "registered corrected adaptive diagnostic is unavailable"
        ) from exc
    if hashlib.sha256(raw).hexdigest() != REGISTERED_DIAGNOSTIC_RAW_SHA256:
        raise CorrectedAdaptiveDiagnosticRegistryError(
            "registered corrected adaptive diagnostic raw hash drifted"
        )
    try:
        validated = validate_corrected_adaptive_diagnostic(payload, root=root)
    except (CorrectedAdaptiveDiagnosticError, OSError, ValueError) as exc:
        raise CorrectedAdaptiveDiagnosticRegistryError(str(exc)) from exc
    if (
        validated.get("artifact_sha256")
        != REGISTERED_DIAGNOSTIC_ARTIFACT_SHA256
        or validated.get("built_at_utc")
        != REGISTERED_DIAGNOSTIC_BUILT_AT_UTC
    ):
        raise CorrectedAdaptiveDiagnosticRegistryError(
            "registered corrected adaptive diagnostic identity drifted"
        )
    built_at = datetime.fromisoformat(REGISTERED_DIAGNOSTIC_BUILT_AT_UTC)
    if built_at > datetime.now(timezone.utc):
        raise CorrectedAdaptiveDiagnosticRegistryError(
            "registered corrected adaptive diagnostic is future dated"
        )
    return validated


__all__ = [
    "CorrectedAdaptiveDiagnosticRegistryError",
    "REGISTERED_DIAGNOSTIC_ARTIFACT_SHA256",
    "REGISTERED_DIAGNOSTIC_BUILT_AT_UTC",
    "REGISTERED_DIAGNOSTIC_LOCATOR",
    "REGISTERED_DIAGNOSTIC_RAW_SHA256",
    "validate_registered_corrected_adaptive_diagnostic_v1",
]
