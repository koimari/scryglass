"""Fail-closed validation for the independent Draft Score promotion record.

The coefficient artifact and development report are not authority by
themselves.  A serving result needs an explicitly issued receipt that binds
the model, evaluation, calibration, reliability, and replay evidence bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from lol_kills.v2.data.common import parse_rfc3339


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROMOTION_SCHEMA_VERSION = "draft-terminal-promotion-receipt-v2"


class PromotionReceiptError(ValueError):
    """Raised when a promotion receipt is malformed or incomplete."""


_REQUIRED_KEYS = {
    "schema_version",
    "status",
    "model_version",
    "artifact_sha256",
    "l2_contract_sha256",
    "development_evaluation_sha256",
    "candidate_registry_sha256",
    "calibration_transform_sha256",
    "reliability_artifact_sha256",
    "replay_parity_evidence_sha256",
    "independent_authority_record_sha256",
    "semantic_draft_authority_sha256",
    "semantic_draft_authority_id",
    "independent_l2_authority",
    "final_temporal_holdout_sealed",
    "private_terminal_draft_component_authorized",
    "public_probability_authorized",
    "event_probability_authorized",
    "exact_rating_receipt_required_for_event_probability",
    "replay_parity_verified",
    "reliability_gate_passed",
    "contextual_g1_authority",
    "authority_record_id",
    "issued_at",
}


@dataclass(frozen=True)
class TerminalPromotionBindings:
    """Exact local bytes that an external L2 receipt must bind."""

    development_evaluation_sha256: str
    candidate_registry_sha256: str
    l2_contract_sha256: str
    calibration_transform_sha256: str | None = None
    reliability_artifact_sha256: str | None = None
    replay_parity_evidence_sha256: str | None = None
    independent_authority_record_sha256: str | None = None
    authority_record_id: str | None = None
    semantic_draft_authority_sha256: str | None = None
    semantic_draft_authority_id: str | None = None
    semantic_draft_model_artifact_sha256: str | None = None
    semantic_draft_model_version: str | None = None

    @classmethod
    def from_repo_root(cls, root: Path | str = Path(".")) -> "TerminalPromotionBindings":
        repo_root = Path(root)

        def digest(locator: str) -> str:
            return hashlib.sha256((repo_root / locator).read_bytes()).hexdigest()

        return cls(
            development_evaluation_sha256=digest("data/lol/v2/models/draft-terminal/development-evaluation-summary.json"),
            candidate_registry_sha256=digest("data/lol/v2/models/draft-terminal/draft-terminal-candidate-registry.json"),
            l2_contract_sha256=digest("data/lol/v2/models/draft-terminal/draft-terminal-l2-evaluation-contract.json"),
        )

    def with_authority_record_bytes(
        self,
        raw: bytes,
        *,
        model_artifact_sha256: str,
    ) -> "TerminalPromotionBindings":
        """Validate and bind the exact external authority-record bytes."""

        from .l2_authority import authority_record_payload_sha256, load_l2_authority_record, validate_l2_authority_record

        record = load_l2_authority_record(raw)
        validate_l2_authority_record(
            record,
            expected_bindings={
                "candidate_registry_sha256": self.candidate_registry_sha256,
                "development_evaluation_sha256": self.development_evaluation_sha256,
                "l2_contract_sha256": self.l2_contract_sha256,
                "model_artifact_sha256": model_artifact_sha256,
            },
        )
        return replace(
            self,
            calibration_transform_sha256=record["calibration_transform_sha256"],
            reliability_artifact_sha256=record["reliability_artifact_sha256"],
            replay_parity_evidence_sha256=record["replay_parity_evidence_sha256"],
            independent_authority_record_sha256=authority_record_payload_sha256(raw),
            authority_record_id=record["authority_record_id"],
        )

    def with_active_semantic_draft_authority(
        self,
        *,
        root: Path | str = Path("."),
        environment: Mapping[str, str] = os.environ,
        as_of: datetime | None = None,
    ) -> "TerminalPromotionBindings":
        """Replay and bind the active semantic Draft deployment authority."""

        from .semantic_draft_authority_v1 import (
            load_active_semantic_draft_authority_v1,
        )

        active = load_active_semantic_draft_authority_v1(
            root=Path(root), environment=environment, as_of=as_of
        )
        model = active["deployment_model"]
        return replace(
            self,
            semantic_draft_authority_sha256=active["receipt_raw_sha256"],
            semantic_draft_authority_id=active["receipt"]["authority_id"],
            semantic_draft_model_artifact_sha256=model["artifact_sha256"],
            semantic_draft_model_version=model["model_version"],
        )


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise PromotionReceiptError(f"{field} must be a lowercase SHA-256")
    return value


def validate_promotion_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the exact receipt shape used by the serving boundary.

    This checks structure and explicit authority declarations.  It does not
    manufacture independent authority or recompute evidence from a caller's
    assertions; the receipt must come from the separately reviewed L2 record.
    """

    if not isinstance(receipt, Mapping):
        raise PromotionReceiptError("promotion receipt must be a mapping")
    if set(receipt) != _REQUIRED_KEYS:
        missing = sorted(_REQUIRED_KEYS - set(receipt))
        extra = sorted(set(receipt) - _REQUIRED_KEYS)
        detail = []
        if missing:
            detail.append(f"missing {','.join(missing)}")
        if extra:
            detail.append(f"unexpected {','.join(extra)}")
        raise PromotionReceiptError("promotion receipt keys do not match the frozen contract (" + "; ".join(detail) + ")")
    if receipt["schema_version"] != PROMOTION_SCHEMA_VERSION:
        raise PromotionReceiptError("promotion receipt schema_version is not supported")
    if receipt["status"] != "approved":
        raise PromotionReceiptError("promotion receipt is not approved")
    for field in (
        "artifact_sha256",
        "l2_contract_sha256",
        "development_evaluation_sha256",
        "candidate_registry_sha256",
        "calibration_transform_sha256",
        "reliability_artifact_sha256",
        "replay_parity_evidence_sha256",
        "independent_authority_record_sha256",
        "semantic_draft_authority_sha256",
    ):
        _require_hash(receipt[field], field)
    for field in (
        "model_version",
        "authority_record_id",
        "semantic_draft_authority_id",
        "issued_at",
    ):
        if not isinstance(receipt[field], str) or not receipt[field].strip():
            raise PromotionReceiptError(f"{field} must be a non-empty string")
    try:
        parse_rfc3339(receipt["issued_at"])
    except (TypeError, ValueError) as exc:
        raise PromotionReceiptError("issued_at must be an RFC-3339 timestamp") from exc
    for field in (
        "independent_l2_authority",
        "final_temporal_holdout_sealed",
        "private_terminal_draft_component_authorized",
        "exact_rating_receipt_required_for_event_probability",
        "replay_parity_verified",
        "reliability_gate_passed",
    ):
        if receipt[field] is not True:
            raise PromotionReceiptError(f"{field} must be true")
    if (
        receipt["public_probability_authorized"] is not False
        or receipt["event_probability_authorized"] is not False
    ):
        raise PromotionReceiptError(
            "terminal Draft component receipt cannot authorize probability"
        )
    if receipt["contextual_g1_authority"] not in {"not_applicable", "approved"}:
        raise PromotionReceiptError("contextual_g1_authority must be not_applicable or approved")


def promotion_receipt_authorizes(
    model_version: str,
    artifact_sha256: str,
    receipt: Mapping[str, Any] | None,
    bindings: TerminalPromotionBindings | None = None,
    *,
    root: Path | str = Path("."),
    environment: Mapping[str, str] = os.environ,
    as_of: datetime | None = None,
) -> bool:
    """Return true only for a complete receipt bound to this artifact."""

    if not isinstance(receipt, Mapping):
        return False
    try:
        validate_promotion_receipt(receipt)
    except PromotionReceiptError:
        return False
    try:
        from .semantic_draft_authority_v1 import (
            load_active_semantic_draft_authority_v1,
        )

        active = load_active_semantic_draft_authority_v1(
            root=Path(root), environment=environment, as_of=as_of
        )
    except (OSError, ValueError, RuntimeError):
        return False
    active_model = active.get("deployment_model") or {}
    active_receipt = active.get("receipt") or {}
    return bool(
        bindings is not None
        and isinstance(bindings.independent_authority_record_sha256, str)
        and _SHA256_RE.fullmatch(bindings.independent_authority_record_sha256)
        and receipt["model_version"] == model_version
        and receipt["artifact_sha256"] == artifact_sha256
        and active.get("private_terminal_draft_component_authorized") is True
        and active.get("private_event_probability_authorized") is False
        and active.get("public_probability_authorized") is False
        and active.get("betting_authorized") is False
        and active.get("receipt_raw_sha256")
        == receipt["semantic_draft_authority_sha256"]
        and active_receipt.get("authority_id")
        == receipt["semantic_draft_authority_id"]
        and active_model.get("artifact_sha256") == artifact_sha256
        and active_model.get("model_version") == model_version
        and isinstance(bindings.semantic_draft_authority_sha256, str)
        and _SHA256_RE.fullmatch(bindings.semantic_draft_authority_sha256)
        and receipt["semantic_draft_authority_sha256"]
        == bindings.semantic_draft_authority_sha256
        and isinstance(bindings.semantic_draft_authority_id, str)
        and bool(bindings.semantic_draft_authority_id.strip())
        and receipt["semantic_draft_authority_id"]
        == bindings.semantic_draft_authority_id
        and bindings.semantic_draft_model_artifact_sha256 == artifact_sha256
        and bindings.semantic_draft_model_version == model_version
        and receipt["development_evaluation_sha256"] == bindings.development_evaluation_sha256
        and receipt["candidate_registry_sha256"] == bindings.candidate_registry_sha256
        and receipt["l2_contract_sha256"] == bindings.l2_contract_sha256
        and isinstance(bindings.calibration_transform_sha256, str)
        and _SHA256_RE.fullmatch(bindings.calibration_transform_sha256)
        and receipt["calibration_transform_sha256"] == bindings.calibration_transform_sha256
        and isinstance(bindings.reliability_artifact_sha256, str)
        and _SHA256_RE.fullmatch(bindings.reliability_artifact_sha256)
        and receipt["reliability_artifact_sha256"] == bindings.reliability_artifact_sha256
        and isinstance(bindings.replay_parity_evidence_sha256, str)
        and _SHA256_RE.fullmatch(bindings.replay_parity_evidence_sha256)
        and receipt["replay_parity_evidence_sha256"] == bindings.replay_parity_evidence_sha256
        and receipt["independent_authority_record_sha256"] == bindings.independent_authority_record_sha256
        and isinstance(bindings.authority_record_id, str)
        and bool(bindings.authority_record_id.strip())
        and receipt["authority_record_id"] == bindings.authority_record_id
    )


def receipt_payload_sha256(raw: bytes) -> str:
    """Hash the exact serialized receipt bytes for an external audit record."""

    if not isinstance(raw, bytes) or not raw:
        raise PromotionReceiptError("promotion receipt bytes must be non-empty")
    return hashlib.sha256(raw).hexdigest()


def load_promotion_receipt(raw: bytes) -> dict[str, Any]:
    """Load strict JSON for an authority-owned receipt without self-hashing it."""

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise PromotionReceiptError(f"promotion receipt contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except PromotionReceiptError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise PromotionReceiptError("promotion receipt must be strict UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise PromotionReceiptError("promotion receipt must be a JSON object")
    validate_promotion_receipt(payload)
    return payload


__all__ = [
    "PROMOTION_SCHEMA_VERSION",
    "PromotionReceiptError",
    "TerminalPromotionBindings",
    "load_promotion_receipt",
    "promotion_receipt_authorizes",
    "receipt_payload_sha256",
    "validate_promotion_receipt",
]
