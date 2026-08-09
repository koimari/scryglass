"""Immutable registry pin for probability-pipeline readiness v1."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .probability_pipeline_readiness_v1 import (
    DEFAULT_OUTPUT,
    RESULT_STATE,
    SCHEMA_VERSION,
    ProbabilityPipelineReadinessError,
    validate_probability_pipeline_readiness_v1,
)


REGISTERED_READINESS_LOCATOR = DEFAULT_OUTPUT
REGISTERED_READINESS_RAW_SHA256 = (
    "535ce12a841cbab527908efdd9419f15f0011b1f350a5651ee9c7839f6a75642"
)
REGISTERED_READINESS_ARTIFACT_SHA256 = (
    "b1b94bd5cf822b70b416ac466af248bff58886135d208bd1de76fb6660bbc5b2"
)
REGISTERED_READINESS_LOCKED_AT_UTC = "2026-08-02T04:30:00+00:00"


class RegisteredProbabilityPipelineReadinessError(RuntimeError):
    """The registered readiness artifact no longer matches its immutable pin."""


def validate_registered_probability_pipeline_readiness_v1(
    *, root: Path = Path(".")
) -> dict[str, Any]:
    path = root / REGISTERED_READINESS_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise RegisteredProbabilityPipelineReadinessError(
            "registered probability-pipeline readiness is unavailable"
        )
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTERED_READINESS_RAW_SHA256:
        raise RegisteredProbabilityPipelineReadinessError(
            "registered probability-pipeline readiness raw hash changed"
        )
    try:
        payload = json.loads(raw)
        checked = validate_probability_pipeline_readiness_v1(
            payload, root=root
        )
    except (json.JSONDecodeError, ProbabilityPipelineReadinessError) as exc:
        raise RegisteredProbabilityPipelineReadinessError(
            "registered probability-pipeline readiness is invalid"
        ) from exc
    if (
        checked.get("schema_version") != SCHEMA_VERSION
        or checked.get("result_state") != RESULT_STATE
        or checked.get("artifact_sha256")
        != REGISTERED_READINESS_ARTIFACT_SHA256
        or checked.get("locked_at_utc")
        != REGISTERED_READINESS_LOCKED_AT_UTC
    ):
        raise RegisteredProbabilityPipelineReadinessError(
            "registered probability-pipeline readiness identity changed"
        )
    return checked


__all__ = [
    "REGISTERED_READINESS_ARTIFACT_SHA256",
    "REGISTERED_READINESS_LOCATOR",
    "REGISTERED_READINESS_LOCKED_AT_UTC",
    "REGISTERED_READINESS_RAW_SHA256",
    "RegisteredProbabilityPipelineReadinessError",
    "validate_registered_probability_pipeline_readiness_v1",
]
