"""Hard gate for the best validated Scryglass composition AUC.

The value in this module is a development reference. It does not grant a
public prediction authority. A refreshed candidate must meet the reference
before its own evaluation and receipt can be considered.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any

BASELINE_MODEL_ID = "scryglass-pi-r9-depth4-state-space"
BASELINE_AUC = 0.70681
BASELINE_BRIER = 0.21708
BASELINE_LOG_LOSS = 0.62330
SCHEMA_VERSION = "scryglass:auc-preservation-gate:v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


class AucGateError(ValueError):
    """Raised when an AUC preservation receipt cannot be trusted."""


def _finite_auc(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise AucGateError(f"{label} must be a finite number") from exc
    if not math.isfinite(result) or not 0.5 <= result <= 1.0:
        raise AucGateError(f"{label} must be a finite AUC in [0.5, 1]")
    return result


def validate_auc(candidate_auc: Any, *, baseline_auc: float = BASELINE_AUC) -> dict[str, Any]:
    """Return a deterministic non-inferiority result for one candidate."""

    baseline = _finite_auc(baseline_auc, "baseline_auc")
    try:
        candidate = _finite_auc(candidate_auc, "candidate_auc")
    except AucGateError as exc:
        return {
            "candidate_auc": None,
            "baseline_auc": baseline,
            "auc_noninferior": False,
            "delta": None,
            "reason": str(exc),
        }
    delta = round(candidate - baseline, 8)
    return {
        "candidate_auc": candidate,
        "baseline_auc": baseline,
        "auc_noninferior": candidate >= baseline,
        "delta": delta,
        "reason": None if candidate >= baseline else "candidate_auc_below_baseline",
    }


def build_reference_receipt(
    *,
    source_sha256: str,
    scorer_sha256: str,
    baseline_auc: float = BASELINE_AUC,
    brier: float = BASELINE_BRIER,
    log_loss: float = BASELINE_LOG_LOSS,
) -> dict[str, Any]:
    """Build the checked-in development receipt for the validated baseline."""

    for value, label in ((source_sha256, "source_sha256"), (scorer_sha256, "scorer_sha256")):
        if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
            raise AucGateError(f"{label} must be a lowercase sha256")
    baseline = _finite_auc(baseline_auc, "baseline_auc")
    try:
        brier_value = float(brier)
        log_loss_value = float(log_loss)
    except (TypeError, ValueError) as exc:
        raise AucGateError("brier and log_loss must be finite numbers") from exc
    if not math.isfinite(brier_value) or not math.isfinite(log_loss_value):
        raise AucGateError("brier and log_loss must be finite numbers")
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "development_only",
        "status": "reference_baseline",
        "model_id": BASELINE_MODEL_ID,
        "metrics": {
            "auc": baseline,
            "brier": round(brier_value, 8),
            "log_loss": round(log_loss_value, 8),
        },
        "source_sha256": source_sha256,
        "scorer_sha256": scorer_sha256,
        "gate": {
            "rule": "candidate_auc >= reference_auc",
            "tolerance": 0.0,
            "promotion_receipt_required": True,
        },
        "claim_ceiling": {
            "public_prediction": False,
            "public_probability": False,
            "model_promotion": False,
        },
    }


def receipt_is_valid(receipt: Mapping[str, Any]) -> bool:
    """Validate the structural parts required by a phase-curve consumer."""

    if receipt.get("schema_version") != SCHEMA_VERSION:
        return False
    if receipt.get("authority") != "development_only":
        return False
    if receipt.get("status") != "reference_baseline":
        return False
    metrics = receipt.get("metrics")
    if not isinstance(metrics, Mapping):
        return False
    try:
        _finite_auc(metrics.get("auc"), "metrics.auc")
    except AucGateError:
        return False
    return all(
        isinstance(receipt.get(key), str) and bool(_HASH_RE.fullmatch(receipt[key]))
        for key in ("source_sha256", "scorer_sha256")
    )


__all__ = [
    "AucGateError",
    "BASELINE_AUC",
    "BASELINE_BRIER",
    "BASELINE_LOG_LOSS",
    "BASELINE_MODEL_ID",
    "SCHEMA_VERSION",
    "build_reference_receipt",
    "receipt_is_valid",
    "validate_auc",
]
