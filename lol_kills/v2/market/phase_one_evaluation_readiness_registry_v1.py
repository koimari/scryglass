"""Immutable registry pin for phase-one evaluation readiness v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .phase_one_evaluation_readiness_v1 import (
    DEFAULT_OUTPUT,
    RESULT_STATE,
    SCHEMA_VERSION,
    validate_phase_one_evaluation_readiness_v1,
)


REGISTERED_READINESS_LOCATOR = DEFAULT_OUTPUT
REGISTERED_READINESS_RAW_SHA256 = (
    "4298c9e2aba0dca3dee2c34bc09865530aa431ab7123aa5a47e1d6986eb7c4f8"
)
REGISTERED_READINESS_ARTIFACT_SHA256 = (
    "c599b321f07b8471a2aeeaedcfcc9c5883d5faccc44c4bbc85de4c4413278d28"
)
REGISTERED_READINESS_LOCKED_AT_UTC = "2026-08-02T04:00:00+00:00"


class RegisteredPhaseOneEvaluationReadinessError(RuntimeError):
    """The registered readiness artifact no longer matches its immutable pin."""


def validate_registered_phase_one_evaluation_readiness_v1(
    *, root: Path = Path(".")
) -> dict[str, Any]:
    path = root / REGISTERED_READINESS_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise RegisteredPhaseOneEvaluationReadinessError(
            "registered phase-one evaluation readiness is unavailable"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTERED_READINESS_RAW_SHA256:
        raise RegisteredPhaseOneEvaluationReadinessError(
            "registered phase-one evaluation readiness raw hash changed"
        )
    try:
        payload = json.loads(raw)
        checked = validate_phase_one_evaluation_readiness_v1(payload, root=root)
    except (RuntimeError, OSError, ValueError) as exc:
        raise RegisteredPhaseOneEvaluationReadinessError(
            "registered phase-one evaluation readiness is invalid"
        ) from exc
    if (
        checked.get("schema_version") != SCHEMA_VERSION
        or checked.get("result_state") != RESULT_STATE
        or checked.get("artifact_sha256")
        != REGISTERED_READINESS_ARTIFACT_SHA256
        or checked.get("locked_at_utc") != REGISTERED_READINESS_LOCKED_AT_UTC
    ):
        raise RegisteredPhaseOneEvaluationReadinessError(
            "registered phase-one evaluation readiness identity changed"
        )
    return checked


__all__ = [
    "REGISTERED_READINESS_ARTIFACT_SHA256",
    "REGISTERED_READINESS_LOCATOR",
    "REGISTERED_READINESS_LOCKED_AT_UTC",
    "REGISTERED_READINESS_RAW_SHA256",
    "RegisteredPhaseOneEvaluationReadinessError",
    "validate_registered_phase_one_evaluation_readiness_v1",
]
