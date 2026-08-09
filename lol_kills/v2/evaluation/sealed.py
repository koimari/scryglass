"""Claim-first, byte-backed, single-use sealed evaluation decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from .checks import ValidationFailure
from .b2_pipeline import B2_REQUIRED_HARD_GATES
from .contract_validation import (
    FiveOutputValidationReport,
    verify_five_output_validation_report,
)
from .metrics import brier_score, expected_calibration_error, log_loss
from .pipeline import EvaluationReport, evaluate_candidate
from .splitter import load_evaluation_registry
from .types import (
    CONTRACT_TREE_SHA256,
    CandidateAdapter,
    EvalRow,
    EvaluationRegistry,
    canonical_json,
    canonical_sha256,
    canonical_timestamp,
    parse_utc_timestamp,
)


SEALED_STAGE_NAMES = (
    "raw",
    "features",
    "state_reconstruction",
    "calibration",
    "serialization",
    "serving",
)
B1_PIPELINE_HARD_GATES = (
    "registry_frozen",
    "source_tree_match",
    "transform_identity",
    "runtime_transform_identity",
    "runtime_artifact_identity",
    "draft_order_diagnostics",
    "future_feature_joins",
    "final_roster",
    "row_cutoff",
    "exact_roster",
    "split_disjoint",
    "required_role_invariance_pairs",
    "required_side_swap_pairs",
    "seal_tamper",
    "terminal_probability_wording",
    "prefix_probability_wording",
    "label_leakage",
    "python_runtime_parity",
    "series_atomicity",
    "test_row_coverage",
    "validation_metrics_present",
)
PIPELINE_HARD_GATES = B1_PIPELINE_HARD_GATES + B2_REQUIRED_HARD_GATES
_SEALED_ONLY_HARD_GATES = (
    "sealed_snapshot_verified",
    "sealed_suites_exact_once",
    "stage_manifest_complete",
    "five_output_validation_all_pass",
    "five_output_validation_hash_match",
    "receipt_content_addressed",
    "sealed_joint_scoring",
    "sealed_uncertainty_rule",
    "sealed_critical_strata",
    "sealed_multiplicity_rule",
)
REQUIRED_B1_SEALED_HARD_GATES = tuple(
    sorted(B1_PIPELINE_HARD_GATES + _SEALED_ONLY_HARD_GATES)
)
REQUIRED_SEALED_HARD_GATES = tuple(
    sorted(
        PIPELINE_HARD_GATES + _SEALED_ONLY_HARD_GATES
    )
)
_EXECUTOR_AUTHORITY = object()
SEALED_OUTCOME_FINGERPRINT_ALGORITHM = {
    "algorithm_id": (
        "scryglass:sealed-outcome-fingerprint:"
        "eval-row-canonical-json-sha256-v1"
    ),
    "canonicalization": "canonical_json_utf8",
    "row_content": (
        "EvalRow.to_payload including outcome label and all registered row content"
    ),
    "suite_identity": (
        "sorted unique suite name and sorted unique member row IDs"
    ),
}
SEALED_OUTCOME_FINGERPRINT_ALGORITHM_SHA256 = (
    "443fcfd85e789ef066c7f1b1f33775ad60e6641d4a45d6e32eb9844f7ca6bbf1"
)
REGISTRY_REGISTRAR_TRUST_ROOT = Path(
    "data/lol/v2/evaluation/registry-registrar-trust-root.json"
)
REGISTRY_REGISTRAR_TRUST_ROOT_RAW_SHA256 = (
    "7ab9ed349af98007d3385b18a769bf6b798a364032c204c354acdbdbb6486590"
)
REGISTRY_REGISTRAR_TRUST_ROOT_OBJECT_SHA256 = (
    "c3328f4bfaeccdf7b9b9e30d9d9576bb689625fa8797bb695d2e19f9a32b5dd2"
)


def _pinned_registrar_verifier_key(
    registrar_id: str,
    registrar_kind: str,
) -> bytes:
    raw = _read_exact_bytes(
        str(REGISTRY_REGISTRAR_TRUST_ROOT.resolve()),
        REGISTRY_REGISTRAR_TRUST_ROOT_RAW_SHA256,
        "registry registrar trust root",
    )
    payload = _strict_json_object(raw, "registry registrar trust root")
    if canonical_sha256(payload) != REGISTRY_REGISTRAR_TRUST_ROOT_OBJECT_SHA256:
        raise ValidationFailure("registry registrar trust root object is stale")
    if registrar_kind not in {"production", "test_only"}:
        raise ValidationFailure("ledger registrar kind is invalid")
    matches = [
        item
        for item in payload[registrar_kind]["registrars"]
        if item.get("registrar_id") == registrar_id
    ]
    if len(matches) != 1:
        raise ValidationFailure("registrar is absent or duplicated in pinned trust root")
    try:
        key = bytes.fromhex(str(matches[0]["verifier_public_key_hex"]))
    except (KeyError, ValueError) as exc:
        raise ValidationFailure("pinned registrar verifier key is malformed") from exc
    if len(key) != 32:
        raise ValidationFailure("pinned registrar verifier key must be 32 bytes")
    return key


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return (canonical_json(value) + "\n").encode("utf-8")


def _valid_hash(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _production_hash(value: object) -> bool:
    return _valid_hash(value) and len(set(str(value).lower())) > 1


def _required_sealed_hard_gates(
    registry: EvaluationRegistry,
) -> tuple[str, ...]:
    return (
        REQUIRED_SEALED_HARD_GATES
        if registry.b2_artifact_refs
        else REQUIRED_B1_SEALED_HARD_GATES
    )


def _required_sealed_hard_gates_for_request(
    request: "SealedDecisionRequest",
) -> tuple[str, ...]:
    plan_raw = _read_exact_bytes(
        request.decision_plan_locator,
        request.decision_plan_raw_sha256,
        "sealed decision plan",
    )
    plan = _strict_json_object(plan_raw, "sealed decision plan")
    if (
        plan.get("plan_sha256") != request.decision_plan_sha256
        or canonical_sha256(
            {key: value for key, value in plan.items() if key != "plan_sha256"}
        )
        != request.decision_plan_sha256
    ):
        raise ValidationFailure("sealed receipt plan identity is stale")
    registry_raw = _read_exact_bytes(
        str(plan.get("registry_locator", "")),
        str(plan.get("registry_raw_sha256", "")),
        "sealed registry",
    )
    registry = load_evaluation_registry(str(plan["registry_locator"]))
    if (
        registry.sha256() != plan.get("registry_sha256")
        or hashlib.sha256(registry_raw).hexdigest()
        != plan.get("registry_raw_sha256")
    ):
        raise ValidationFailure("sealed receipt registry identity is stale")
    return _required_sealed_hard_gates(registry)


def _read_exact_bytes(locator: str, expected_sha256: str, what: str) -> bytes:
    path = Path(locator)
    if not path.is_file():
        raise ValidationFailure(f"{what} locator is missing")
    payload = path.read_bytes()
    if _sha256_bytes(payload) != expected_sha256:
        raise ValidationFailure(f"{what} raw-byte hash mismatch")
    return payload


def _strict_json_object(raw: bytes, what: str) -> dict[str, Any]:
    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationFailure(f"{what} contains duplicate JSON key '{key}'")
            result[key] = value
        return result

    try:
        payload = json.loads(raw, object_pairs_hook=reject_duplicate)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValidationFailure(f"{what} is not valid canonical JSON") from exc
    if not isinstance(payload, dict):
        raise ValidationFailure(f"{what} must be a JSON object")
    if raw != _canonical_bytes(payload):
        raise ValidationFailure(f"{what} encoding is not canonical")
    return payload


def _row_from_payload(payload: Mapping[str, Any]) -> EvalRow:
    return EvalRow(
        row_id=str(payload["row_id"]),
        series_id=str(payload["series_id"]),
        series_resolved=bool(payload["series_resolved"]),
        event_start=parse_utc_timestamp(str(payload["event_start"])),
        patch_id=str(payload["patch_id"]),
        league_id=str(payload["league_id"]),
        league_tier=str(payload["league_tier"]),
        region=str(payload["region"]),
        as_of=parse_utc_timestamp(str(payload["as_of"])),
        label=int(payload["label"]),
        feature_values={
            str(name): float(value)
            for name, value in dict(payload["feature_values"]).items()
        },
        feature_available_at={
            str(name): parse_utc_timestamp(str(value))
            for name, value in dict(payload["feature_available_at"]).items()
        },
        roster_id=str(payload.get("roster_id", "")),
        roster_snapshot_id=payload.get("roster_snapshot_id"),
        roster_snapshot_time=(
            parse_utc_timestamp(str(payload["roster_snapshot_time"]))
            if payload.get("roster_snapshot_time") is not None
            else None
        ),
        roster_snapshot_stage=str(payload.get("roster_snapshot_stage", "operational")),
        is_international_event=bool(payload.get("is_international_event", False)),
        international_event_id=payload.get("international_event_id"),
        is_roster_change=bool(payload.get("is_roster_change", False)),
        champion_ids=tuple(payload.get("champion_ids", ())),
        is_sparse_champion=bool(payload.get("is_sparse_champion", False)),
        metadata=dict(payload.get("metadata", {})),
    )


@dataclass(frozen=True)
class FrozenEvaluationSnapshot:
    locator: str
    raw_bytes_sha256: str
    canonical_rows_sha256: str
    source_snapshot_id: str
    source_snapshot_sha256: str
    training_snapshot_id: str
    training_snapshot_sha256: str
    source_tree_sha256: str
    contract_tree_sha256: str
    row_fingerprints: tuple[tuple[str, str], ...]
    snapshot_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "locator": self.locator,
            "raw_bytes_sha256": self.raw_bytes_sha256,
            "canonical_rows_sha256": self.canonical_rows_sha256,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "training_snapshot_id": self.training_snapshot_id,
            "training_snapshot_sha256": self.training_snapshot_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "contract_tree_sha256": self.contract_tree_sha256,
            "row_fingerprints": [list(item) for item in self.row_fingerprints],
        }

    def verify_hash(self) -> bool:
        return self.snapshot_sha256 == canonical_sha256(self.unsigned_payload())

    def fingerprint_map(self) -> dict[str, str]:
        return dict(self.row_fingerprints)


def make_frozen_evaluation_snapshot(
    rows: Sequence[EvalRow],
    registry: EvaluationRegistry,
    *,
    locator: str | Path,
) -> FrozenEvaluationSnapshot:
    """Write the immutable raw snapshot used by the sole sealed loader."""
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValidationFailure("sealed snapshot contains duplicate row IDs")
    path = Path(locator)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical_bytes([row.to_payload() for row in rows])
    path.write_bytes(raw)
    row_fingerprints = tuple(sorted((row.row_id, row.fingerprint()) for row in rows))
    unsigned = {
        "locator": str(path.resolve()),
        "raw_bytes_sha256": _sha256_bytes(raw),
        "canonical_rows_sha256": canonical_sha256([row.to_payload() for row in rows]),
        "source_snapshot_id": registry.source_snapshot_id,
        "source_snapshot_sha256": registry.source_snapshot_sha256,
        "training_snapshot_id": registry.training_snapshot_id,
        "training_snapshot_sha256": registry.training_snapshot_sha256,
        "source_tree_sha256": registry.source_tree_sha256,
        "contract_tree_sha256": registry.contract_tree_sha256,
        "row_fingerprints": [list(item) for item in row_fingerprints],
    }
    return FrozenEvaluationSnapshot(
        locator=str(path.resolve()),
        raw_bytes_sha256=_sha256_bytes(raw),
        canonical_rows_sha256=unsigned["canonical_rows_sha256"],
        source_snapshot_id=registry.source_snapshot_id,
        source_snapshot_sha256=registry.source_snapshot_sha256,
        training_snapshot_id=registry.training_snapshot_id,
        training_snapshot_sha256=registry.training_snapshot_sha256,
        source_tree_sha256=registry.source_tree_sha256,
        contract_tree_sha256=registry.contract_tree_sha256,
        row_fingerprints=row_fingerprints,
        snapshot_sha256=canonical_sha256(unsigned),
    )


def _load_and_verify_snapshot(
    snapshot: FrozenEvaluationSnapshot | None,
    registry: EvaluationRegistry,
) -> tuple[bytes, tuple[EvalRow, ...]]:
    if snapshot is None:
        raise ValidationFailure("sealed execution requires an immutable snapshot locator")
    if not snapshot.verify_hash():
        raise ValidationFailure("sealed snapshot content hash is invalid")
    raw = _read_exact_bytes(
        snapshot.locator, snapshot.raw_bytes_sha256, "sealed snapshot"
    )
    try:
        payload = json.loads(raw)
        rows = tuple(_row_from_payload(item) for item in payload)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValidationFailure("sealed snapshot bytes are malformed") from exc
    row_ids = [row.row_id for row in rows]
    if len(row_ids) != len(set(row_ids)):
        raise ValidationFailure("sealed snapshot contains duplicate row IDs")
    canonical_rows_sha256 = canonical_sha256([row.to_payload() for row in rows])
    if canonical_rows_sha256 != snapshot.canonical_rows_sha256:
        raise ValidationFailure("sealed snapshot canonical digest mismatch")
    expected = (
        registry.source_snapshot_id,
        registry.source_snapshot_sha256,
        registry.training_snapshot_id,
        registry.training_snapshot_sha256,
        registry.source_tree_sha256,
        CONTRACT_TREE_SHA256,
    )
    actual = (
        snapshot.source_snapshot_id,
        snapshot.source_snapshot_sha256,
        snapshot.training_snapshot_id,
        snapshot.training_snapshot_sha256,
        snapshot.source_tree_sha256,
        snapshot.contract_tree_sha256,
    )
    if actual != expected:
        raise ValidationFailure("sealed snapshot identity/hash/contract mismatch")
    fingerprints = tuple(sorted((row.row_id, row.fingerprint()) for row in rows))
    if len(fingerprints) != len({row_id for row_id, _ in fingerprints}):
        raise ValidationFailure("sealed snapshot fingerprint identities are duplicated")
    if fingerprints != snapshot.row_fingerprints:
        raise ValidationFailure("sealed snapshot row identities are incomplete or changed")
    if canonical_rows_sha256 != registry.source_snapshot_sha256:
        raise ValidationFailure("sealed source snapshot hash does not match raw rows")
    rows_by_id = {row.row_id: row for row in rows}
    development_ids = sorted(
        {row_id for fold in registry.split_plan.folds for row_id in fold.all_ids}
    )
    if any(row_id not in rows_by_id for row_id in development_ids):
        raise ValidationFailure("sealed training snapshot references missing rows")
    training_sha256 = canonical_sha256(
        [[row_id, rows_by_id[row_id].fingerprint()] for row_id in development_ids]
    )
    if training_sha256 != registry.training_snapshot_sha256:
        raise ValidationFailure("sealed training snapshot hash does not match development rows")
    return raw, rows


def verify_frozen_evaluation_snapshot(
    snapshot: FrozenEvaluationSnapshot | None,
    rows: Sequence[EvalRow],
    registry: EvaluationRegistry,
) -> None:
    """Development-only compatibility verifier; sealed execution never calls it."""
    _, loaded = _load_and_verify_snapshot(snapshot, registry)
    if [row.to_payload() for row in loaded] != [row.to_payload() for row in rows]:
        raise ValidationFailure("provided development rows differ from frozen snapshot bytes")


@dataclass(frozen=True)
class SealedDecisionPlan:
    registrar_locator: str
    registrar_raw_sha256: str
    registrar_object_sha256: str
    registrar_id: str
    registrar_kind: str
    registrar_verifier_public_key_hex: str
    registry_locator: str
    registry_raw_sha256: str
    registry_sha256: str
    eligibility_locator: str
    eligibility_sha256: str
    snapshot: FrozenEvaluationSnapshot
    candidate_locator: str
    candidate_sha256: str
    candidate_adapter_id: str
    candidate_adapter_version: str
    baseline_locator: str
    baseline_sha256: str
    baseline_adapter_id: str
    baseline_adapter_version: str
    transform_locator: str
    transform_sha256: str
    five_output_validation_sha256: str
    metric_names: tuple[str, ...]
    metric_margins: Mapping[str, float]
    higher_is_better: Mapping[str, bool]
    uncertainty_rule: str
    multiplicity_rule: str
    critical_strata: tuple[str, ...]
    secondary_benefit_rule: str
    suite_row_ids: Mapping[str, tuple[str, ...]]
    suite_assignment_hashes: Mapping[str, str]
    suite_outcome_locator_hashes: Mapping[str, str]
    contract_tree_sha256: str
    plan_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "registrar_locator": self.registrar_locator,
            "registrar_raw_sha256": self.registrar_raw_sha256,
            "registrar_object_sha256": self.registrar_object_sha256,
            "registrar_id": self.registrar_id,
            "registrar_kind": self.registrar_kind,
            "registrar_verifier_public_key_hex": (
                self.registrar_verifier_public_key_hex
            ),
            "registry_locator": self.registry_locator,
            "registry_raw_sha256": self.registry_raw_sha256,
            "registry_sha256": self.registry_sha256,
            "eligibility_locator": self.eligibility_locator,
            "eligibility_sha256": self.eligibility_sha256,
            "snapshot": {**self.snapshot.unsigned_payload(), "snapshot_sha256": self.snapshot.snapshot_sha256},
            "candidate_locator": self.candidate_locator,
            "candidate_sha256": self.candidate_sha256,
            "candidate_adapter_id": self.candidate_adapter_id,
            "candidate_adapter_version": self.candidate_adapter_version,
            "baseline_locator": self.baseline_locator,
            "baseline_sha256": self.baseline_sha256,
            "baseline_adapter_id": self.baseline_adapter_id,
            "baseline_adapter_version": self.baseline_adapter_version,
            "transform_locator": self.transform_locator,
            "transform_sha256": self.transform_sha256,
            "five_output_validation_sha256": self.five_output_validation_sha256,
            "metric_names": list(self.metric_names),
            "metric_margins": dict(self.metric_margins),
            "higher_is_better": dict(self.higher_is_better),
            "uncertainty_rule": self.uncertainty_rule,
            "multiplicity_rule": self.multiplicity_rule,
            "critical_strata": list(self.critical_strata),
            "secondary_benefit_rule": self.secondary_benefit_rule,
            "suite_row_ids": {
                name: list(row_ids)
                for name, row_ids in self.suite_row_ids.items()
            },
            "suite_assignment_hashes": dict(self.suite_assignment_hashes),
            "suite_outcome_locator_hashes": dict(self.suite_outcome_locator_hashes),
            "contract_tree_sha256": self.contract_tree_sha256,
        }

    def verify_hash(self) -> bool:
        return self.plan_sha256 == canonical_sha256(self.unsigned_payload())


def create_sealed_decision_plan(
    *,
    registry: EvaluationRegistry,
    registrar_locator: str | Path = REGISTRY_REGISTRAR_TRUST_ROOT,
    registrar_id: str,
    registrar_kind: str,
    registry_locator: str | Path,
    eligibility_locator: str | Path,
    snapshot: FrozenEvaluationSnapshot,
    candidate_adapter: CandidateAdapter,
    candidate_locator: str | Path,
    baseline_adapter: CandidateAdapter,
    baseline_locator: str | Path,
    transform_locator: str | Path,
    five_output_validation_sha256: str,
    metric_margins: Mapping[str, float],
    higher_is_better: Mapping[str, bool],
    uncertainty_rule: str = "paired_series_cluster_bootstrap_95",
    multiplicity_rule: str = "all_metrics_and_critical_strata",
    critical_strata: Sequence[str] | None = None,
    secondary_benefit_rule: str = "not_required",
) -> SealedDecisionPlan:
    registrar_path = Path(registrar_locator).resolve()
    registry_path = Path(registry_locator).resolve()
    eligibility_path = Path(eligibility_locator).resolve()
    candidate_path = Path(candidate_locator).resolve()
    baseline_path = Path(baseline_locator).resolve()
    transform_path = Path(transform_locator).resolve()
    registry_raw = registry_path.read_bytes()
    registrar_raw = registrar_path.read_bytes()
    registrar_payload = _strict_json_object(
        registrar_raw, "registry registrar trust root"
    )
    registrar_verifier_key = _pinned_registrar_verifier_key(
        registrar_id, registrar_kind
    )
    if (
        _sha256_bytes(registrar_raw) != REGISTRY_REGISTRAR_TRUST_ROOT_RAW_SHA256
        or canonical_sha256(registrar_payload)
        != REGISTRY_REGISTRAR_TRUST_ROOT_OBJECT_SHA256
        or registrar_kind not in {"production", "test_only"}
        or registrar_id
        not in {
            item["registrar_id"]
            for item in registrar_payload[registrar_kind]["registrars"]
        }
    ):
        raise ValidationFailure("registry registrar authority is not pinned")
    eligibility_raw = eligibility_path.read_bytes()
    names = tuple(metric_margins)
    registered_holdouts = tuple(registry.split_plan.sealed_holdouts)
    registered_suite_names = tuple(item.name for item in registered_holdouts)
    if len(registered_suite_names) != len(set(registered_suite_names)):
        raise ValidationFailure("sealed registry contains duplicate suite IDs")
    suites = tuple(critical_strata or registered_suite_names)
    if (
        not names
        or set(names) != set(higher_is_better)
        or any(not math.isfinite(float(v)) or float(v) < 0 for v in metric_margins.values())
        or any(not isinstance(v, bool) for v in higher_is_better.values())
        or not suites
        or len(suites) != len(set(suites))
        or set(suites) != set(registered_suite_names)
    ):
        raise ValidationFailure("sealed decision plan has invalid metric or stratum rules")
    snapshot_row_ids = [row_id for row_id, _ in snapshot.row_fingerprints]
    if (
        len(snapshot_row_ids) != len(set(snapshot_row_ids))
        or tuple(sorted(snapshot.row_fingerprints)) != snapshot.row_fingerprints
    ):
        raise ValidationFailure(
            "sealed snapshot row fingerprints are duplicate or noncanonical"
        )
    suite_row_ids: dict[str, tuple[str, ...]] = {}
    for holdout in registered_holdouts:
        row_ids = tuple(holdout.row_ids)
        if len(row_ids) != len(set(row_ids)):
            raise ValidationFailure(
                f"sealed suite '{holdout.name}' contains duplicate row membership"
            )
        canonical_ids = tuple(sorted(row_ids))
        if any(row_id not in set(snapshot_row_ids) for row_id in canonical_ids):
            raise ValidationFailure(
                f"sealed suite '{holdout.name}' references an unknown row"
            )
        suite_row_ids[holdout.name] = canonical_ids
    assignment_hashes = {
        suite_name: canonical_sha256(
            {
                "suite_name": suite_name,
                "row_ids": list(row_ids),
            }
        )
        for suite_name, row_ids in suite_row_ids.items()
    }
    outcome_hashes = {
        holdout.name: canonical_sha256(
            {
                "snapshot_raw_sha256": snapshot.raw_bytes_sha256,
                "snapshot_canonical_rows_sha256": snapshot.canonical_rows_sha256,
                "row_ids": list(suite_row_ids[holdout.name]),
            }
        )
        for holdout in registry.split_plan.sealed_holdouts
    }
    fields: dict[str, Any] = {
        "registrar_locator": str(registrar_path),
        "registrar_raw_sha256": _sha256_bytes(registrar_raw),
        "registrar_object_sha256": canonical_sha256(registrar_payload),
        "registrar_id": registrar_id,
        "registrar_kind": registrar_kind,
        "registrar_verifier_public_key_hex": registrar_verifier_key.hex(),
        "registry_locator": str(registry_path),
        "registry_raw_sha256": _sha256_bytes(registry_raw),
        "registry_sha256": registry.sha256(),
        "eligibility_locator": str(eligibility_path),
        "eligibility_sha256": _sha256_bytes(eligibility_raw),
        "snapshot": {**snapshot.unsigned_payload(), "snapshot_sha256": snapshot.snapshot_sha256},
        "candidate_locator": str(candidate_path),
        "candidate_sha256": _sha256_bytes(candidate_path.read_bytes()),
        "candidate_adapter_id": candidate_adapter.adapter_id,
        "candidate_adapter_version": candidate_adapter.adapter_version,
        "baseline_locator": str(baseline_path),
        "baseline_sha256": _sha256_bytes(baseline_path.read_bytes()),
        "baseline_adapter_id": baseline_adapter.adapter_id,
        "baseline_adapter_version": baseline_adapter.adapter_version,
        "transform_locator": str(transform_path),
        "transform_sha256": _sha256_bytes(transform_path.read_bytes()),
        "five_output_validation_sha256": five_output_validation_sha256,
        "metric_names": list(names),
        "metric_margins": dict(metric_margins),
        "higher_is_better": dict(higher_is_better),
        "uncertainty_rule": uncertainty_rule,
        "multiplicity_rule": multiplicity_rule,
        "critical_strata": list(suites),
        "secondary_benefit_rule": secondary_benefit_rule,
        "suite_row_ids": {
            name: list(row_ids) for name, row_ids in suite_row_ids.items()
        },
        "suite_assignment_hashes": assignment_hashes,
        "suite_outcome_locator_hashes": outcome_hashes,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
    }
    return SealedDecisionPlan(
        registrar_locator=str(registrar_path),
        registrar_raw_sha256=fields["registrar_raw_sha256"],
        registrar_object_sha256=fields["registrar_object_sha256"],
        registrar_id=registrar_id,
        registrar_kind=registrar_kind,
        registrar_verifier_public_key_hex=registrar_verifier_key.hex(),
        registry_locator=str(registry_path),
        registry_raw_sha256=fields["registry_raw_sha256"],
        registry_sha256=registry.sha256(),
        eligibility_locator=str(eligibility_path),
        eligibility_sha256=fields["eligibility_sha256"],
        snapshot=snapshot,
        candidate_locator=str(candidate_path),
        candidate_sha256=fields["candidate_sha256"],
        candidate_adapter_id=candidate_adapter.adapter_id,
        candidate_adapter_version=candidate_adapter.adapter_version,
        baseline_locator=str(baseline_path),
        baseline_sha256=fields["baseline_sha256"],
        baseline_adapter_id=baseline_adapter.adapter_id,
        baseline_adapter_version=baseline_adapter.adapter_version,
        transform_locator=str(transform_path),
        transform_sha256=fields["transform_sha256"],
        five_output_validation_sha256=five_output_validation_sha256,
        metric_names=names,
        metric_margins=dict(metric_margins),
        higher_is_better=dict(higher_is_better),
        uncertainty_rule=uncertainty_rule,
        multiplicity_rule=multiplicity_rule,
        critical_strata=suites,
        secondary_benefit_rule=secondary_benefit_rule,
        suite_row_ids=suite_row_ids,
        suite_assignment_hashes=assignment_hashes,
        suite_outcome_locator_hashes=outcome_hashes,
        contract_tree_sha256=CONTRACT_TREE_SHA256,
        plan_sha256=canonical_sha256(fields),
    )


def write_sealed_decision_plan(plan: SealedDecisionPlan, locator: str | Path) -> str:
    if not plan.verify_hash():
        raise ValidationFailure("sealed decision plan is tampered")
    path = Path(locator)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**plan.unsigned_payload(), "plan_sha256": plan.plan_sha256}
    raw = _canonical_bytes(payload)
    path.write_bytes(raw)
    return _sha256_bytes(raw)


@dataclass(frozen=True)
class SealedDecisionRequest:
    seal_id: str
    opened_at: str
    decision_plan_locator: str
    decision_plan_raw_sha256: str
    decision_plan_sha256: str
    consumption_key: str
    decision_id: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "seal_id": self.seal_id,
            "opened_at": self.opened_at,
            "decision_plan_locator": self.decision_plan_locator,
            "decision_plan_raw_sha256": self.decision_plan_raw_sha256,
            "decision_plan_sha256": self.decision_plan_sha256,
            "consumption_key": self.consumption_key,
        }

    def verify_hash(self) -> bool:
        return _valid_hash(self.decision_id) and _valid_hash(self.consumption_key)


def _consumption_key(plan: SealedDecisionPlan) -> str:
    if (
        canonical_sha256(SEALED_OUTCOME_FINGERPRINT_ALGORITHM)
        != SEALED_OUTCOME_FINGERPRINT_ALGORITHM_SHA256
    ):
        raise ValidationFailure("sealed outcome fingerprint algorithm is stale")
    fingerprints = tuple(plan.snapshot.row_fingerprints)
    row_ids = [row_id for row_id, _ in fingerprints]
    if (
        tuple(sorted(fingerprints)) != fingerprints
        or len(row_ids) != len(set(row_ids))
        or any(not row_id or not _valid_hash(digest) for row_id, digest in fingerprints)
    ):
        raise ValidationFailure(
            "sealed outcome row fingerprints are duplicate or noncanonical"
        )
    suite_names = tuple(plan.suite_row_ids)
    if (
        not suite_names
        or len(plan.critical_strata) != len(set(plan.critical_strata))
        or set(suite_names) != set(plan.critical_strata)
        or set(suite_names) != set(plan.suite_assignment_hashes)
        or set(suite_names) != set(plan.suite_outcome_locator_hashes)
    ):
        raise ValidationFailure(
            "sealed outcome suites are missing, extra, or duplicated"
        )
    canonical_suites: list[dict[str, Any]] = []
    known_rows = set(row_ids)
    for suite_name in sorted(suite_names):
        membership = tuple(plan.suite_row_ids[suite_name])
        if (
            not suite_name
            or not membership
            or tuple(sorted(membership)) != membership
            or len(membership) != len(set(membership))
            or any(row_id not in known_rows for row_id in membership)
        ):
            raise ValidationFailure(
                f"sealed suite '{suite_name}' membership is noncanonical"
            )
        canonical_suites.append(
            {
                "suite_name": suite_name,
                "row_ids": list(membership),
            }
        )
    return canonical_sha256(
        {
            "outcome_fingerprint_algorithm_sha256": (
                SEALED_OUTCOME_FINGERPRINT_ALGORITHM_SHA256
            ),
            "rows": [list(item) for item in fingerprints],
            "suites": canonical_suites,
        }
    )


def _candidate_decision_id(plan: SealedDecisionPlan) -> str:
    return canonical_sha256(
        {
            "sealed_outcome_consumption_key": _consumption_key(plan),
            "registrar_benchmark_lineage": {
                "registrar_id": plan.registrar_id,
                "registrar_kind": plan.registrar_kind,
                "registrar_raw_sha256": plan.registrar_raw_sha256,
                "registrar_object_sha256": plan.registrar_object_sha256,
                "registrar_verifier_public_key_hex": (
                    plan.registrar_verifier_public_key_hex
                ),
                "registry_raw_sha256": plan.registry_raw_sha256,
                "registry_sha256": plan.registry_sha256,
                "source_snapshot_id": plan.snapshot.source_snapshot_id,
                "source_snapshot_sha256": plan.snapshot.source_snapshot_sha256,
                "training_snapshot_id": plan.snapshot.training_snapshot_id,
                "training_snapshot_sha256": (
                    plan.snapshot.training_snapshot_sha256
                ),
                "source_tree_sha256": plan.snapshot.source_tree_sha256,
                "contract_tree_sha256": plan.contract_tree_sha256,
            },
            "candidate": {
                "adapter_id": plan.candidate_adapter_id,
                "adapter_version": plan.candidate_adapter_version,
                "artifact_sha256": plan.candidate_sha256,
            },
            "baseline": {
                "adapter_id": plan.baseline_adapter_id,
                "adapter_version": plan.baseline_adapter_version,
                "artifact_sha256": plan.baseline_sha256,
            },
            "transform_sha256": plan.transform_sha256,
            "five_output_validation_sha256": plan.five_output_validation_sha256,
            "decision_policy": {
                "metric_names": list(plan.metric_names),
                "metric_margins": dict(plan.metric_margins),
                "higher_is_better": dict(plan.higher_is_better),
                "uncertainty_rule": plan.uncertainty_rule,
                "multiplicity_rule": plan.multiplicity_rule,
                "critical_strata": list(plan.critical_strata),
                "secondary_benefit_rule": plan.secondary_benefit_rule,
            },
        }
    )


def create_sealed_decision_request(
    *,
    seal_id: str,
    plan: SealedDecisionPlan,
    decision_plan_locator: str | Path,
    opened_at: datetime | None = None,
    **forbidden_overrides: Any,
) -> SealedDecisionRequest:
    if forbidden_overrides:
        raise ValidationFailure("sealed request cannot override preregistered plan fields")
    if not seal_id or not plan.verify_hash():
        raise ValidationFailure("sealed request needs a valid preregistered plan")
    plan_path = Path(decision_plan_locator).resolve()
    raw = _read_exact_bytes(
        str(plan_path),
        _sha256_bytes(plan_path.read_bytes()),
        "sealed decision plan",
    )
    on_disk = _strict_json_object(raw, "sealed decision plan")
    expected_plan = {**plan.unsigned_payload(), "plan_sha256": plan.plan_sha256}
    if on_disk != expected_plan:
        raise ValidationFailure("sealed decision plan locator is stale or swapped")
    opened = canonical_timestamp(opened_at or datetime.now(timezone.utc))
    unsigned = {
        "seal_id": seal_id,
        "opened_at": opened,
        "decision_plan_locator": str(plan_path),
        "decision_plan_raw_sha256": _sha256_bytes(raw),
        "decision_plan_sha256": plan.plan_sha256,
        "consumption_key": _consumption_key(plan),
    }
    return SealedDecisionRequest(
        seal_id=seal_id,
        opened_at=opened,
        decision_plan_locator=str(plan_path),
        decision_plan_raw_sha256=_sha256_bytes(raw),
        decision_plan_sha256=plan.plan_sha256,
        consumption_key=unsigned["consumption_key"],
        decision_id=_candidate_decision_id(plan),
    )


@dataclass(frozen=True)
class SealedStage:
    name: str
    input_sha256: str
    output_sha256: str
    artifact_locator: str

    def to_payload(self) -> dict[str, str]:
        return {
            "name": self.name,
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "artifact_locator": self.artifact_locator,
        }


@dataclass(frozen=True)
class SealedDecisionReceipt:
    request: SealedDecisionRequest
    stages: tuple[SealedStage, ...]
    suite_statuses: tuple[tuple[str, str], ...]
    candidate_metrics: Mapping[str, float]
    baseline_metrics: Mapping[str, float]
    metric_bounds: Mapping[str, Mapping[str, float]]
    metric_decisions: Mapping[str, bool]
    critical_stratum_decisions: Mapping[str, bool]
    metric_margins: Mapping[str, float]
    higher_is_better: Mapping[str, bool]
    uncertainty_rule: str
    multiplicity_rule: str
    secondary_benefit_rule: str
    hard_gates: Mapping[str, bool]
    registrar_verifier_public_key_hex: str
    evaluation_report_sha256: str
    evaluation_report_locator: str
    ledger_entry_sha256: str
    receipt_locator: str
    receipt_sha256: str

    def unsigned_payload(self) -> dict[str, Any]:
        return {
            "request": {**self.request.unsigned_payload(), "decision_id": self.request.decision_id},
            "stages": [stage.to_payload() for stage in self.stages],
            "suite_statuses": [list(item) for item in self.suite_statuses],
            "candidate_metrics": dict(self.candidate_metrics),
            "baseline_metrics": dict(self.baseline_metrics),
            "metric_bounds": {name: dict(value) for name, value in self.metric_bounds.items()},
            "metric_decisions": dict(self.metric_decisions),
            "critical_stratum_decisions": dict(self.critical_stratum_decisions),
            "metric_margins": dict(self.metric_margins),
            "higher_is_better": dict(self.higher_is_better),
            "uncertainty_rule": self.uncertainty_rule,
            "multiplicity_rule": self.multiplicity_rule,
            "secondary_benefit_rule": self.secondary_benefit_rule,
            "hard_gates": dict(self.hard_gates),
            "registrar_verifier_public_key_hex": (
                self.registrar_verifier_public_key_hex
            ),
            "evaluation_report_sha256": self.evaluation_report_sha256,
            "evaluation_report_locator": self.evaluation_report_locator,
            "ledger_entry_sha256": self.ledger_entry_sha256,
            "receipt_locator": self.receipt_locator,
        }

    def verify_hash(self) -> bool:
        return self.receipt_sha256 == canonical_sha256(self.unsigned_payload())


@dataclass(frozen=True)
class ClaimHandle:
    consumption_key: str
    decision_id: str
    ledger_entry_sha256: str


@dataclass(frozen=True)
class ExecutorClaimToken:
    consumption_key: str
    decision_id: str
    artifact_manifest_sha256: str
    evaluation_report_sha256: str
    signature: str


class AtomicSealedLedger:
    """Append-only claim ledger with executor-authenticated finalization."""

    def __init__(
        self,
        path: str | Path,
        *,
        signing_seed: bytes | None = None,
        verifier_public_key: bytes | None = None,
        registrar_identity: str = "production-unconfigured",
        registrar_kind: str = "production",
    ) -> None:
        self.path = Path(path)
        owned_root = Path("data/lol/v2/evaluation").resolve()
        if not self.path.resolve().is_relative_to(owned_root):
            raise ValidationFailure("sealed ledger must live under the L2 evaluation data root")
        if registrar_kind not in {"production", "test_only"}:
            raise ValidationFailure("ledger registrar kind is invalid")
        if signing_seed is not None and len(signing_seed) != 32:
            raise ValidationFailure("Ed25519 signing seed must be exactly 32 bytes")
        self.registrar_identity = registrar_identity
        self.registrar_kind = registrar_kind
        self._signing_seed = signing_seed
        derived_public = (
            Ed25519PrivateKey.from_private_bytes(signing_seed)
            .public_key()
            .public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            if signing_seed is not None
            else None
        )
        if (
            verifier_public_key is not None
            and derived_public is not None
            and verifier_public_key != derived_public
        ):
            raise ValidationFailure("ledger signing and verifier keys do not match")
        pinned_public = _pinned_registrar_verifier_key(
            registrar_identity, registrar_kind
        )
        effective_public = verifier_public_key or derived_public
        if effective_public is None:
            raise ValidationFailure("pinned ledger verifier public key is absent")
        if effective_public != pinned_public:
            raise ValidationFailure(
                "ledger verifier key does not match pinned registrar authority"
            )
        self._verifier_public_key = pinned_public
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._executor_claims: set[tuple[str, str]] = set()

    @property
    def verifier_public_key(self) -> bytes:
        if self._verifier_public_key is None:
            raise ValidationFailure("pinned ledger verifier public key is absent")
        return self._verifier_public_key

    def _assert_pinned_registrar_key(self) -> None:
        pinned = _pinned_registrar_verifier_key(
            self.registrar_identity, self.registrar_kind
        )
        if self.verifier_public_key != pinned:
            raise ValidationFailure(
                "ledger verifier key is detached from pinned registrar authority"
            )

    def _private_key(self) -> Ed25519PrivateKey:
        if self._signing_seed is None:
            raise ValidationFailure(
                "production ledger signing key is absent; fail-closed"
            )
        return Ed25519PrivateKey.from_private_bytes(self._signing_seed)

    def _sign(self, payload: Mapping[str, Any]) -> str:
        return self._private_key().sign(_canonical_bytes(payload)).hex()

    def _verify_signature(self, payload: Mapping[str, Any], signature: object) -> None:
        if not isinstance(signature, str):
            raise ValidationFailure("ledger final signature is missing")
        try:
            Ed25519PublicKey.from_public_bytes(self.verifier_public_key).verify(
                bytes.fromhex(signature),
                _canonical_bytes(payload),
            )
        except (ValueError, InvalidSignature) as exc:
            raise ValidationFailure("ledger final signature is invalid") from exc

    def _entries(self, ledger: Any) -> list[dict[str, Any]]:
        self._assert_pinned_registrar_key()
        ledger.seek(0)
        entries = [
            _strict_json_object(line.encode("utf-8"), "sealed ledger entry")
            for line in ledger
            if line.strip()
        ]
        previous = "0" * 64
        for entry in entries:
            if entry.get("registrar_identity") != self.registrar_identity:
                raise ValidationFailure("sealed ledger registrar identity is swapped")
            if (
                entry.get("registrar_verifier_public_key_hex")
                != self.verifier_public_key.hex()
            ):
                raise ValidationFailure("sealed ledger registrar key is swapped")
            if (
                entry.get("kind") == "receipt"
                and entry.get("registrar_kind") != self.registrar_kind
            ):
                raise ValidationFailure("sealed ledger registrar kind is swapped")
            if entry.get("previous_entry_sha256") != previous:
                raise ValidationFailure("sealed ledger entry chain is broken")
            entry_sha256 = entry.get("entry_sha256")
            unsigned = {
                key: value
                for key, value in entry.items()
                if key not in {"entry_sha256", "registrar_signature"}
            }
            material = dict(unsigned)
            if "registrar_signature" in entry:
                material["registrar_signature"] = entry["registrar_signature"]
            if entry_sha256 != canonical_sha256(material):
                raise ValidationFailure("sealed ledger entry hash is invalid")
            if entry.get("kind") == "receipt":
                self._verify_signature(unsigned, entry.get("registrar_signature"))
            previous = str(entry_sha256)
        return entries

    def claim(self, request: SealedDecisionRequest) -> ClaimHandle:
        self._assert_pinned_registrar_key()
        if not request.verify_hash():
            raise ValidationFailure("sealed request decision_id is invalid")
        entry: dict[str, Any] = {
            "kind": "open",
            "decision_id": request.decision_id,
            "consumption_key": request.consumption_key,
            "seal_id": request.seal_id,
            "opened_at": request.opened_at,
            "request_sha256": canonical_sha256(request.unsigned_payload()),
            "registrar_identity": self.registrar_identity,
            "registrar_verifier_public_key_hex": (
                self.verifier_public_key.hex()
            ),
        }
        with self.path.open("a+", encoding="utf-8") as ledger:
            fcntl.flock(ledger.fileno(), fcntl.LOCK_EX)
            existing = self._entries(ledger)
            if any(
                item.get("decision_id") == request.decision_id
                or item.get("seal_id") == request.seal_id
                or item.get("consumption_key") == request.consumption_key
                for item in existing
            ):
                raise ValidationFailure("frozen sealed suites have already been opened")
            entry["previous_entry_sha256"] = (
                str(existing[-1]["entry_sha256"]) if existing else "0" * 64
            )
            entry["entry_sha256"] = canonical_sha256(entry)
            ledger.seek(0, 2)
            ledger.write(canonical_json(entry) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            fcntl.flock(ledger.fileno(), fcntl.LOCK_UN)
        return ClaimHandle(
            consumption_key=request.consumption_key,
            decision_id=request.decision_id,
            ledger_entry_sha256=str(entry["entry_sha256"]),
        )

    def _claim_for_executor(
        self,
        request: SealedDecisionRequest,
        *,
        authority: object,
    ) -> ClaimHandle:
        if authority is not _EXECUTOR_AUTHORITY:
            raise ValidationFailure("sealed executor authority is required")
        self._private_key()
        claim = self.claim(request)
        self._executor_claims.add((claim.consumption_key, claim.decision_id))
        return claim

    def _issue_executor_token(
        self,
        claim: ClaimHandle,
        *,
        artifact_manifest_sha256: str,
        evaluation_report_sha256: str,
        authority: object,
    ) -> ExecutorClaimToken:
        claim_key = (claim.consumption_key, claim.decision_id)
        if (
            authority is not _EXECUTOR_AUTHORITY
            or claim_key not in self._executor_claims
        ):
            raise ValidationFailure(
                "public ledger claims cannot authorize sealed finalization"
            )
        self._executor_claims.remove(claim_key)
        message = canonical_json(
            {
                "consumption_key": claim.consumption_key,
                "decision_id": claim.decision_id,
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "evaluation_report_sha256": evaluation_report_sha256,
            }
        ).encode("utf-8")
        if self._signing_seed is None:
            raise ValidationFailure("executor signing authority is absent")
        signature = hmac.new(
            self._signing_seed, message, hashlib.sha256
        ).hexdigest()
        return ExecutorClaimToken(
            consumption_key=claim.consumption_key,
            decision_id=claim.decision_id,
            artifact_manifest_sha256=artifact_manifest_sha256,
            evaluation_report_sha256=evaluation_report_sha256,
            signature=signature,
        )

    def _verify_token(self, token: ExecutorClaimToken) -> bool:
        message = canonical_json(
            {
                "consumption_key": token.consumption_key,
                "decision_id": token.decision_id,
                "artifact_manifest_sha256": token.artifact_manifest_sha256,
                "evaluation_report_sha256": token.evaluation_report_sha256,
            }
        ).encode("utf-8")
        if self._signing_seed is None:
            return False
        expected = hmac.new(
            self._signing_seed, message, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, token.signature)

    def finalize(
        self,
        receipt: SealedDecisionReceipt,
        *,
        executor_claim_token: ExecutorClaimToken | None = None,
    ) -> None:
        self._assert_pinned_registrar_key()
        if executor_claim_token is None or not self._verify_token(executor_claim_token):
            raise ValidationFailure("sealed finalize requires an executor-issued claim token")
        if (
            executor_claim_token.consumption_key != receipt.request.consumption_key
            or executor_claim_token.decision_id != receipt.request.decision_id
            or executor_claim_token.evaluation_report_sha256
            != receipt.evaluation_report_sha256
        ):
            raise ValidationFailure("executor token is detached from sealed evidence")
        artifact_manifest_sha256 = canonical_sha256(
            [stage.to_payload() for stage in receipt.stages]
        )
        if artifact_manifest_sha256 != executor_claim_token.artifact_manifest_sha256:
            raise ValidationFailure("executor token does not cover stage artifacts")
        verify_sealed_decision_receipt(receipt, verify_artifact_bytes=True)
        persisted_receipt = _read_exact_bytes(
            receipt.receipt_locator,
            _sha256_bytes(Path(receipt.receipt_locator).read_bytes()),
            "sealed receipt",
        )
        if json.loads(persisted_receipt).get("receipt_sha256") != receipt.receipt_sha256:
            raise ValidationFailure("persisted receipt does not match executor result")
        report_payload = _strict_json_object(
            _read_exact_bytes(
                receipt.evaluation_report_locator,
                receipt.evaluation_report_sha256,
                "canonical sealed evaluation report",
            ),
            "canonical sealed evaluation report",
        )
        identities = report_payload.get("sealed_identities")
        if not isinstance(identities, dict):
            raise ValidationFailure("sealed report lacks immutable identities")
        with self.path.open("a+", encoding="utf-8") as ledger:
            fcntl.flock(ledger.fileno(), fcntl.LOCK_EX)
            existing = self._entries(ledger)
            opens = [
                item for item in existing
                if item.get("kind") == "open"
                and item.get("decision_id") == receipt.request.decision_id
                and item.get("consumption_key") == receipt.request.consumption_key
            ]
            finals = [
                item for item in existing
                if item.get("kind") == "receipt"
                and item.get("consumption_key") == receipt.request.consumption_key
            ]
            if len(opens) != 1 or finals:
                raise ValidationFailure("sealed receipt cannot be durably finalized")
            entry: dict[str, Any] = {
                "kind": "receipt",
                "decision_id": receipt.request.decision_id,
                "consumption_key": receipt.request.consumption_key,
                "receipt_sha256": receipt.receipt_sha256,
                "receipt_locator": receipt.receipt_locator,
                "evaluation_report_sha256": receipt.evaluation_report_sha256,
                "evaluation_report_locator": receipt.evaluation_report_locator,
                "artifact_manifest_sha256": artifact_manifest_sha256,
                "stage_manifest_locator": receipt.receipt_locator,
                "executor_signature": executor_claim_token.signature,
                "open_entry_sha256": receipt.ledger_entry_sha256,
                "candidate_identity": identities["candidate"],
                "baseline_identity": identities["baseline"],
                "registrar_identity": self.registrar_identity,
                "registrar_kind": self.registrar_kind,
                "registrar_verifier_public_key_hex": (
                    self.verifier_public_key.hex()
                ),
                "registrar_trust_root_raw_sha256": (
                    REGISTRY_REGISTRAR_TRUST_ROOT_RAW_SHA256
                ),
                "registrar_trust_root_object_sha256": (
                    REGISTRY_REGISTRAR_TRUST_ROOT_OBJECT_SHA256
                ),
                "previous_entry_sha256": str(existing[-1]["entry_sha256"]),
            }
            entry["registrar_signature"] = self._sign(entry)
            entry["entry_sha256"] = canonical_sha256(entry)
            ledger.seek(0, 2)
            ledger.write(canonical_json(entry) + "\n")
            ledger.flush()
            os.fsync(ledger.fileno())
            fcntl.flock(ledger.fileno(), fcntl.LOCK_UN)

    def _final_entry(self, receipt: SealedDecisionReceipt) -> dict[str, Any]:
        if not self.path.is_file():
            raise ValidationFailure("sealed ledger is missing")
        with self.path.open("r", encoding="utf-8") as ledger:
            fcntl.flock(ledger.fileno(), fcntl.LOCK_SH)
            entries = self._entries(ledger)
            fcntl.flock(ledger.fileno(), fcntl.LOCK_UN)
        opens = [
            item for item in entries
            if item.get("kind") == "open"
            and item.get("decision_id") == receipt.request.decision_id
            and item.get("consumption_key") == receipt.request.consumption_key
        ]
        finals = [
            item for item in entries
            if item.get("kind") == "receipt"
            and item.get("decision_id") == receipt.request.decision_id
            and item.get("consumption_key") == receipt.request.consumption_key
            and item.get("receipt_sha256") == receipt.receipt_sha256
        ]
        if len(opens) != 1 or len(finals) != 1:
            raise ValidationFailure("sealed receipt lacks one durable claim/finalization")
        if opens[0].get("entry_sha256") != receipt.ledger_entry_sha256:
            raise ValidationFailure("sealed receipt ledger proof is tampered")
        return finals[0]

    def verify_receipt(self, receipt: SealedDecisionReceipt) -> None:
        final = self._final_entry(receipt)
        persisted = _read_exact_bytes(
            str(final["receipt_locator"]),
            _sha256_bytes(Path(str(final["receipt_locator"])).read_bytes()),
            "persisted sealed receipt",
        )
        if json.loads(persisted).get("receipt_sha256") != receipt.receipt_sha256:
            raise ValidationFailure("caller receipt differs from persisted executor receipt")
        verify_sealed_decision_receipt(receipt, verify_artifact_bytes=True)
        report_raw = _read_exact_bytes(
            str(final["evaluation_report_locator"]),
            str(final["evaluation_report_sha256"]),
            "canonical sealed evaluation report",
        )
        _verify_report_evidence(json.loads(report_raw))

    def load_trusted_evidence(self, receipt: SealedDecisionReceipt) -> dict[str, Any]:
        self.verify_receipt(receipt)
        final = self._final_entry(receipt)
        raw = _read_exact_bytes(
            str(final["evaluation_report_locator"]),
            str(final["evaluation_report_sha256"]),
            "canonical sealed evaluation report",
        )
        evidence = json.loads(raw)
        _verify_report_evidence(evidence)
        return evidence


def evaluation_report_sha256(report: EvaluationReport) -> str:
    return canonical_sha256(_evaluation_report_payload(report))


def _evaluation_report_payload(report: EvaluationReport) -> dict[str, Any]:
    return {
        "adapter_id": report.adapter_id,
        "adapter_version": report.adapter_version,
        "registry_hash": report.registry_hash,
        "test_predictions": {
            row_id: prediction.final_probability()
            for row_id, prediction in sorted(report.test_predictions.items())
        },
        "holdout_reports": {
            name: dict(payload) for name, payload in sorted(report.holdout_reports.items())
        },
        "hard_gate_results": dict(report.hard_gate_results),
        "registry_bootstrap_seed": report.registry_bootstrap_seed,
        "registry_bootstrap_unit": report.registry_bootstrap_unit,
        "registry_bootstrap_replicates": report.registry_bootstrap_replicates,
        "aggregate_calibrated_metrics": dict(report.aggregate_calibrated_metrics),
        "aggregate_raw_metrics": dict(report.aggregate_raw_metrics),
    }


def _assert_production_registration(
    registry: EvaluationRegistry,
    plan: SealedDecisionPlan,
) -> None:
    registrar_raw = _read_exact_bytes(
        plan.registrar_locator,
        plan.registrar_raw_sha256,
        "registry registrar trust root",
    )
    registrar = _strict_json_object(
        registrar_raw, "registry registrar trust root"
    )
    pinned_registrar_key = _pinned_registrar_verifier_key(
        plan.registrar_id, plan.registrar_kind
    )
    if (
        plan.registrar_raw_sha256 != REGISTRY_REGISTRAR_TRUST_ROOT_RAW_SHA256
        or plan.registrar_object_sha256
        != REGISTRY_REGISTRAR_TRUST_ROOT_OBJECT_SHA256
        or canonical_sha256(registrar) != plan.registrar_object_sha256
        or plan.registrar_id
        not in {
            item["registrar_id"]
            for item in registrar[plan.registrar_kind]["registrars"]
        }
        or plan.registrar_verifier_public_key_hex
        != pinned_registrar_key.hex()
    ):
        raise ValidationFailure("registry registrar trust root is detached")
    registry_raw = _read_exact_bytes(
        plan.registry_locator, plan.registry_raw_sha256, "registered registry"
    )
    loaded = load_evaluation_registry(plan.registry_locator)
    if loaded.to_payload() != registry.to_payload() or loaded.sha256() != plan.registry_sha256:
        raise ValidationFailure("registry object is detached from registered bytes")
    eligibility_raw = _read_exact_bytes(
        plan.eligibility_locator, plan.eligibility_sha256, "production eligibility"
    )
    try:
        eligibility = json.loads(eligibility_raw)
    except json.JSONDecodeError as exc:
        raise ValidationFailure("production eligibility evidence is malformed") from exc
    expected_eligibility: dict[str, Any] = {
        "registry_raw_sha256": _sha256_bytes(registry_raw),
        "registry_sha256": registry.sha256(),
        "registry_kind": plan.registrar_kind,
        "production_eligible": plan.registrar_kind == "production",
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "registrar_id": plan.registrar_id,
        "registrar_raw_sha256": plan.registrar_raw_sha256,
        "registrar_object_sha256": plan.registrar_object_sha256,
        "registrar_verifier_public_key_hex": (
            plan.registrar_verifier_public_key_hex
        ),
        "source_ancestry": ["synthetic"] if plan.registrar_kind == "test_only" else [],
        "registry_provenance_sha256": canonical_sha256(
            {
                "source_snapshot_id": registry.source_snapshot_id,
                "training_snapshot_id": registry.training_snapshot_id,
                "source_tree_sha256": registry.source_tree_sha256,
                "noninferiority_provenance": registry.noninferiority_provenance,
            }
        ),
    }
    if eligibility != expected_eligibility:
        raise ValidationFailure("registry lacks immutable production eligibility evidence")
    if plan.registrar_kind == "production" and (
        registry.is_synthetic_registry or eligibility["source_ancestry"]
    ):
        raise ValidationFailure("synthetic registries are permanently non-promotable")
    provenance_values = (
        registry.source_snapshot_id,
        registry.training_snapshot_id,
        registry.noninferiority_provenance,
        *registry.source_crosswalk_sha256.keys(),
        *registry.entity_crosswalk_sha256.keys(),
        *registry.candidate_artifact_hashes.keys(),
        *registry.served_transform_identities.keys(),
    )
    forbidden_provenance_tokens = ("synthetic", "placeholder", "fixture")
    if plan.registrar_kind == "production" and any(
        token in str(value).lower()
        for value in provenance_values
        for token in forbidden_provenance_tokens
    ):
        raise ValidationFailure(
            "synthetic/placeholder fixture provenance cannot be relabeled production"
        )
    hashes = (
        plan.registry_raw_sha256,
        plan.registry_sha256,
        plan.eligibility_sha256,
        plan.snapshot.raw_bytes_sha256,
        plan.snapshot.snapshot_sha256,
        plan.candidate_sha256,
        plan.baseline_sha256,
        plan.transform_sha256,
        plan.five_output_validation_sha256,
    )
    if not all(_production_hash(value) for value in hashes):
        raise ValidationFailure("production sealed plan contains placeholder hashes")


def _assert_plan(
    plan: SealedDecisionPlan,
    request: SealedDecisionRequest,
    registry: EvaluationRegistry,
    candidate: CandidateAdapter,
    baseline: CandidateAdapter,
    validation_report: FiveOutputValidationReport,
) -> None:
    if not plan.verify_hash() or plan.contract_tree_sha256 != CONTRACT_TREE_SHA256:
        raise ValidationFailure("sealed decision plan hash/contract mismatch")
    plan_raw = _read_exact_bytes(
        request.decision_plan_locator,
        request.decision_plan_raw_sha256,
        "sealed decision plan",
    )
    persisted_plan = _strict_json_object(plan_raw, "sealed decision plan")
    if persisted_plan != {**plan.unsigned_payload(), "plan_sha256": plan.plan_sha256}:
        raise ValidationFailure("request plan locator does not match preregistration")
    if (
        request.decision_plan_sha256 != plan.plan_sha256
        or request.consumption_key != _consumption_key(plan)
        or request.decision_id != _candidate_decision_id(plan)
        or not request.verify_hash()
    ):
        raise ValidationFailure("sealed request is detached from preregistration")
    _assert_production_registration(registry, plan)
    for locator, digest, what in (
        (plan.candidate_locator, plan.candidate_sha256, "candidate artifact"),
        (plan.baseline_locator, plan.baseline_sha256, "baseline artifact"),
        (plan.transform_locator, plan.transform_sha256, "transform artifact"),
    ):
        _read_exact_bytes(locator, digest, what)
    if candidate.runtime_artifact_sha256 != plan.candidate_sha256:
        raise ValidationFailure("candidate runtime is detached from preregistered bytes")
    if baseline.runtime_artifact_sha256 != plan.baseline_sha256:
        raise ValidationFailure("baseline runtime is detached from preregistered bytes")
    if (
        candidate.adapter_id != plan.candidate_adapter_id
        or candidate.adapter_version != plan.candidate_adapter_version
        or baseline.adapter_id != plan.baseline_adapter_id
        or baseline.adapter_version != plan.baseline_adapter_version
    ):
        raise ValidationFailure("candidate/baseline runtime identity was swapped")
    if candidate.serialized_transform_sha256 != plan.transform_sha256:
        raise ValidationFailure("candidate transform is detached from preregistered bytes")
    if validation_report.report_sha256 != plan.five_output_validation_sha256:
        raise ValidationFailure("five-output report is detached from preregistration")
    registered_suites = tuple(item.name for item in registry.split_plan.sealed_holdouts)
    if len(registered_suites) != len(set(registered_suites)):
        raise ValidationFailure("registry contains duplicate sealed suite IDs")
    expected_suite_rows = {
        holdout.name: tuple(sorted(holdout.row_ids))
        for holdout in registry.split_plan.sealed_holdouts
    }
    if any(
        len(holdout.row_ids) != len(set(holdout.row_ids))
        for holdout in registry.split_plan.sealed_holdouts
    ):
        raise ValidationFailure("registry contains duplicate sealed suite membership")
    expected_assignments = {
        holdout.name: canonical_sha256(
            {
                "suite_name": holdout.name,
                "row_ids": list(expected_suite_rows[holdout.name]),
            }
        )
        for holdout in registry.split_plan.sealed_holdouts
    }
    expected_outcome_locators = {
        holdout.name: canonical_sha256(
            {
                "snapshot_raw_sha256": plan.snapshot.raw_bytes_sha256,
                "snapshot_canonical_rows_sha256": (
                    plan.snapshot.canonical_rows_sha256
                ),
                "row_ids": list(expected_suite_rows[holdout.name]),
            }
        )
        for holdout in registry.split_plan.sealed_holdouts
    }
    if (
        dict(plan.suite_row_ids) != expected_suite_rows
        or dict(plan.suite_assignment_hashes) != expected_assignments
    ):
        raise ValidationFailure("preregistered suite assignments are missing or changed")
    if (
        dict(plan.suite_outcome_locator_hashes) != expected_outcome_locators
    ):
        raise ValidationFailure("preregistered outcome locators are missing or changed")
    if (
        len(plan.critical_strata) != len(set(plan.critical_strata))
        or set(plan.critical_strata) != set(registered_suites)
    ):
        raise ValidationFailure("critical strata do not cover every sealed suite")


def _per_row_loss(metric: str, label: int, probability: float) -> float:
    if metric == "log_loss":
        return log_loss([label], [probability])
    if metric == "brier":
        return brier_score([label], [probability])
    raise ValidationFailure(f"metric '{metric}' has no per-row dependence oracle")


def _metric_value(metric: str, labels: Sequence[int], probabilities: Sequence[float]) -> float:
    if metric == "log_loss":
        return log_loss(labels, probabilities)
    if metric == "brier":
        return brier_score(labels, probabilities)
    if metric == "ece":
        return expected_calibration_error(labels, probabilities)
    raise ValidationFailure(f"unsupported sealed metric '{metric}'")


def _paired_cluster_bound(
    *,
    metric: str,
    labels: Sequence[int],
    candidate: Sequence[float],
    baseline: Sequence[float],
    series_ids: Sequence[str],
    seed: int,
    replicates: int,
) -> dict[str, float]:
    clusters: dict[str, list[int]] = {}
    for index, series_id in enumerate(series_ids):
        clusters.setdefault(series_id, []).append(index)
    if not clusters:
        raise ValidationFailure("sealed uncertainty requires resolved series clusters")
    cluster_names = tuple(sorted(clusters))
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=float)
    for draw_index in range(replicates):
        selected = rng.integers(0, len(cluster_names), size=len(cluster_names))
        indices = [
            row_index
            for selected_index in selected
            for row_index in clusters[cluster_names[int(selected_index)]]
        ]
        draw_labels = [labels[index] for index in indices]
        draw_candidate = [candidate[index] for index in indices]
        draw_baseline = [baseline[index] for index in indices]
        draws[draw_index] = (
            _metric_value(metric, draw_labels, draw_candidate)
            - _metric_value(metric, draw_labels, draw_baseline)
        )
    point = _metric_value(metric, labels, candidate) - _metric_value(metric, labels, baseline)
    return {
        "point": float(point),
        "lower_95": float(np.quantile(draws, 0.025)),
        "upper_95": float(np.quantile(draws, 0.975)),
        "cluster_count": float(len(cluster_names)),
    }


def _sealed_scoring_evidence(
    candidate_report: EvaluationReport,
    baseline_report: EvaluationReport,
    rows: Sequence[EvalRow],
    plan: SealedDecisionPlan,
    registry: EvaluationRegistry,
) -> dict[str, Any]:
    rows_by_id = {row.row_id: row for row in rows}
    candidate_probs: list[float] = []
    baseline_probs: list[float] = []
    labels: list[int] = []
    series_ids: list[str] = []
    suite_names: list[str] = []
    row_ids: list[str] = []
    strata: dict[str, dict[str, Any]] = {}
    for suite_name in plan.critical_strata:
        candidate_suite = candidate_report.holdout_reports[suite_name]
        baseline_suite = baseline_report.holdout_reports[suite_name]
        expected_ids = tuple(
            next(
                holdout.row_ids
                for holdout in registry.split_plan.sealed_holdouts
                if holdout.name == suite_name
            )
        )
        if (
            tuple(candidate_suite.get("scored_row_ids", ())) != expected_ids
            or tuple(baseline_suite.get("scored_row_ids", ())) != expected_ids
        ):
            raise ValidationFailure("joint sealed scoring row identity mismatch")
        candidate_map = dict(candidate_suite.get("scored_probabilities", {}))
        baseline_map = dict(baseline_suite.get("scored_probabilities", {}))
        if set(candidate_map) != set(expected_ids) or set(baseline_map) != set(expected_ids):
            raise ValidationFailure("joint sealed scoring probabilities are incomplete")
        suite_candidate = [float(candidate_map[row_id]) for row_id in expected_ids]
        suite_baseline = [float(baseline_map[row_id]) for row_id in expected_ids]
        suite_labels = [int(rows_by_id[row_id].label) for row_id in expected_ids]
        suite_metrics = {
            metric: {
                "candidate": _metric_value(metric, suite_labels, suite_candidate),
                "baseline": _metric_value(metric, suite_labels, suite_baseline),
            }
            for metric in plan.metric_names
        }
        strata[suite_name] = {
            "row_ids": list(expected_ids),
            "metrics": suite_metrics,
        }
        for index, row_id in enumerate(expected_ids):
            row_ids.append(row_id)
            suite_names.append(suite_name)
            labels.append(suite_labels[index])
            candidate_probs.append(suite_candidate[index])
            baseline_probs.append(suite_baseline[index])
            series_ids.append(rows_by_id[row_id].series_id)
    candidate_metrics = {
        metric: _metric_value(metric, labels, candidate_probs)
        for metric in plan.metric_names
    }
    baseline_metrics = {
        metric: _metric_value(metric, labels, baseline_probs)
        for metric in plan.metric_names
    }
    bounds = {
        metric: _paired_cluster_bound(
            metric=metric,
            labels=labels,
            candidate=candidate_probs,
            baseline=baseline_probs,
            series_ids=series_ids,
            seed=registry.bootstrap_seed,
            replicates=registry.bootstrap_cluster_replicates,
        )
        for metric in plan.metric_names
    }
    metric_decisions = {}
    for metric in plan.metric_names:
        margin = float(plan.metric_margins[metric])
        if plan.higher_is_better[metric]:
            metric_decisions[metric] = bounds[metric]["lower_95"] >= -margin
        else:
            metric_decisions[metric] = bounds[metric]["upper_95"] <= margin
    stratum_decisions = {}
    for suite_name, evidence in strata.items():
        stratum_decisions[suite_name] = all(
            (
                values["candidate"] + float(plan.metric_margins[metric])
                >= values["baseline"]
                if plan.higher_is_better[metric]
                else values["candidate"] - float(plan.metric_margins[metric])
                <= values["baseline"]
            )
            for metric, values in evidence["metrics"].items()
        )
    return {
        "row_ids": row_ids,
        "suite_names": suite_names,
        "labels": labels,
        "series_ids": series_ids,
        "candidate_probabilities": candidate_probs,
        "baseline_probabilities": baseline_probs,
        "candidate_metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "metric_margins": dict(plan.metric_margins),
        "higher_is_better": dict(plan.higher_is_better),
        "metric_bounds": bounds,
        "metric_decisions": metric_decisions,
        "critical_strata": strata,
        "critical_stratum_decisions": stratum_decisions,
        "uncertainty_rule": plan.uncertainty_rule,
        "multiplicity_rule": plan.multiplicity_rule,
        "secondary_benefit_rule": plan.secondary_benefit_rule,
    }


def _verify_report_evidence(payload: Mapping[str, Any]) -> None:
    required = {
        "registry_sha256",
        "decision_plan_sha256",
        "sealed_outcome_consumption_key",
        "candidate_decision_id",
        "sealed_identities",
        "candidate_report",
        "baseline_report",
        "sealed_scoring",
        "suite_statuses",
        "hard_gates",
        "contract_tree_sha256",
    }
    if set(payload) != required:
        raise ValidationFailure("canonical sealed report is incomplete or extra")
    identities = payload["sealed_identities"]
    has_b2_identity = (
        isinstance(identities, Mapping)
        and "b2_validation_report_sha256" in identities
    )
    expected_identity_fields = {
        "candidate",
        "baseline",
        "registrar",
        "five_output_validation_sha256",
    } | ({"b2_validation_report_sha256"} if has_b2_identity else set())
    expected_hard_gates = (
        REQUIRED_SEALED_HARD_GATES
        if has_b2_identity
        else REQUIRED_B1_SEALED_HARD_GATES
    )
    if (
        not isinstance(identities, Mapping)
        or set(identities) != expected_identity_fields
        or set(payload["hard_gates"]) != set(expected_hard_gates)
        or not all(
            _valid_hash(value)
            for value in (
                payload["sealed_outcome_consumption_key"],
                payload["candidate_decision_id"],
                identities["candidate"]["artifact_sha256"],
                identities["candidate"]["transform_sha256"],
                identities["baseline"]["artifact_sha256"],
                identities["five_output_validation_sha256"],
                *(
                    (identities["b2_validation_report_sha256"],)
                    if has_b2_identity
                    else ()
                ),
            )
        )
        or identities["registrar"].get("verifier_public_key_hex")
        != _pinned_registrar_verifier_key(
            str(identities["registrar"].get("registrar_id")),
            str(identities["registrar"].get("registrar_kind")),
        ).hex()
    ):
        raise ValidationFailure("canonical sealed report identities are malformed")
    scoring = dict(payload["sealed_scoring"])
    names = tuple(scoring["candidate_metrics"])
    if (
        not names
        or tuple(scoring["baseline_metrics"]) != names
        or tuple(scoring["metric_margins"]) != names
        or tuple(scoring["higher_is_better"]) != names
        or tuple(scoring["metric_bounds"]) != names
        or tuple(scoring["metric_decisions"]) != names
    ):
        raise ValidationFailure("canonical sealed scoring metric identities mismatch")
    if any(
        not math.isfinite(float(value))
        for mapping_name in ("candidate_metrics", "baseline_metrics")
        for value in dict(scoring[mapping_name]).values()
    ):
        raise ValidationFailure("canonical sealed scoring contains non-finite metrics")
    labels = list(scoring["labels"])
    candidate = list(scoring["candidate_probabilities"])
    baseline = list(scoring["baseline_probabilities"])
    if not (len(labels) == len(candidate) == len(baseline) == len(scoring["row_ids"])):
        raise ValidationFailure("canonical sealed scoring row evidence is incomplete")
    for metric in names:
        if not math.isclose(
            float(scoring["candidate_metrics"][metric]),
            _metric_value(metric, labels, candidate),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValidationFailure("candidate sealed metric does not recompute")
        if not math.isclose(
            float(scoring["baseline_metrics"][metric]),
            _metric_value(metric, labels, baseline),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValidationFailure("baseline sealed metric does not recompute")
        recomputed_bound = _paired_cluster_bound(
            metric=metric,
            labels=labels,
            candidate=candidate,
            baseline=baseline,
            series_ids=list(scoring["series_ids"]),
            seed=int(payload["candidate_report"]["registry_bootstrap_seed"])
            if "registry_bootstrap_seed" in payload["candidate_report"]
            else 1776,
            replicates=int(
                payload["candidate_report"].get(
                    "registry_bootstrap_replicates",
                    2000,
                )
            ),
        )
        if dict(scoring["metric_bounds"][metric]) != recomputed_bound:
            raise ValidationFailure("sealed dependence-aware bound does not recompute")
        margin = float(scoring["metric_margins"][metric])
        direction = scoring["higher_is_better"][metric]
        decision = (
            recomputed_bound["lower_95"] >= -margin
            if direction
            else recomputed_bound["upper_95"] <= margin
        )
        if scoring["metric_decisions"][metric] is not decision:
            raise ValidationFailure("sealed noninferiority decision does not recompute")
    suite_names = list(scoring["suite_names"])
    recomputed_strata: dict[str, bool] = {}
    for suite_name in scoring["critical_strata"]:
        indices = [
            index for index, value in enumerate(suite_names) if value == suite_name
        ]
        if not indices:
            raise ValidationFailure("critical stratum has no sealed outcomes")
        recomputed_strata[suite_name] = all(
            (
                _metric_value(
                    metric,
                    [labels[index] for index in indices],
                    [candidate[index] for index in indices],
                )
                + float(scoring["metric_margins"][metric])
                >= _metric_value(
                    metric,
                    [labels[index] for index in indices],
                    [baseline[index] for index in indices],
                )
                if scoring["higher_is_better"][metric]
                else _metric_value(
                    metric,
                    [labels[index] for index in indices],
                    [candidate[index] for index in indices],
                )
                - float(scoring["metric_margins"][metric])
                <= _metric_value(
                    metric,
                    [labels[index] for index in indices],
                    [baseline[index] for index in indices],
                )
            )
            for metric in names
        )
    if dict(scoring["critical_stratum_decisions"]) != recomputed_strata:
        raise ValidationFailure("critical-stratum decisions do not recompute")


def _write_stage(directory: Path, name: str, payload: Any, input_sha256: str) -> SealedStage:
    raw = _canonical_bytes(payload)
    locator = directory / f"{name}.json"
    locator.write_bytes(raw)
    return SealedStage(
        name=name,
        input_sha256=input_sha256,
        output_sha256=_sha256_bytes(raw),
        artifact_locator=str(locator.resolve()),
    )


def _verify_stage_artifacts(
    receipt: SealedDecisionReceipt,
    report_payload: Mapping[str, Any],
) -> None:
    payloads: dict[str, Any] = {}
    for stage in receipt.stages:
        raw = _read_exact_bytes(stage.artifact_locator, stage.output_sha256, stage.name)
        try:
            payloads[stage.name] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValidationFailure(f"{stage.name} stage artifact is malformed") from exc
    raw_rows = tuple(_row_from_payload(item) for item in payloads["raw"])
    expected_features = [
        {
            "row_id": row.row_id,
            "features": dict(row.feature_values),
            "available_at": {
                name: canonical_timestamp(value)
                for name, value in row.feature_available_at.items()
            },
        }
        for row in raw_rows
    ]
    if payloads["features"] != expected_features:
        raise ValidationFailure("feature artifact does not reconstruct from raw bytes")
    state = dict(payloads["state_reconstruction"])
    calibration = dict(payloads["calibration"])
    if (
        calibration.get("candidate_holdouts")
        != dict(state["candidate_report"])["holdout_reports"]
        or calibration.get("baseline_holdouts")
        != dict(state["baseline_report"])["holdout_reports"]
    ):
        raise ValidationFailure("calibration artifact is detached from fitted state")
    if payloads["serialization"] != dict(report_payload):
        raise ValidationFailure("serialized output differs from canonical evaluation report")
    serving = dict(payloads["serving"])
    scoring = dict(report_payload["sealed_scoring"])
    if (
        serving.get("serialized_sha256") != receipt.stages[4].output_sha256
        or serving.get("row_ids") != scoring["row_ids"]
        or serving.get("replayed_candidate_probabilities")
        != scoring["candidate_probabilities"]
    ):
        raise ValidationFailure("serving replay differs from serialized sealed output")


def verify_sealed_decision_receipt(
    receipt: SealedDecisionReceipt,
    *,
    expected_suite_names: Sequence[str] | None = None,
    verify_artifact_bytes: bool = False,
) -> None:
    if not receipt.request.verify_hash() or not receipt.verify_hash():
        raise ValidationFailure("sealed request or receipt is tampered")
    if tuple(stage.name for stage in receipt.stages) != SEALED_STAGE_NAMES:
        raise ValidationFailure("sealed stage manifest is missing or reordered")
    for previous, current in zip(receipt.stages, receipt.stages[1:]):
        if previous.output_sha256 != current.input_sha256:
            raise ValidationFailure("sealed stage manifest chain is broken or swapped")
    if verify_artifact_bytes:
        for stage in receipt.stages:
            _read_exact_bytes(stage.artifact_locator, stage.output_sha256, stage.name)
    suite_names = tuple(name for name, _ in receipt.suite_statuses)
    if len(suite_names) != len(set(suite_names)):
        raise ValidationFailure("sealed suite executed more than once")
    if expected_suite_names is not None and suite_names != tuple(expected_suite_names):
        raise ValidationFailure("sealed suite identities do not exactly match registry")
    if any(status != "ok" for _, status in receipt.suite_statuses):
        raise ValidationFailure("one or more sealed suites are unavailable or failed")
    required_hard_gates = _required_sealed_hard_gates_for_request(
        receipt.request
    )
    if set(receipt.hard_gates) != set(required_hard_gates):
        raise ValidationFailure("sealed hard-gate set is missing, extra, or caller-invented")
    if not all(value is True for value in receipt.hard_gates.values()):
        raise ValidationFailure("sealed hard-gate evidence is not all-pass")
    if not _valid_hash(receipt.ledger_entry_sha256):
        raise ValidationFailure("sealed receipt lacks a durable ledger entry")
    report_raw = _read_exact_bytes(
        receipt.evaluation_report_locator,
        receipt.evaluation_report_sha256,
        "canonical sealed evaluation report",
    )
    report_payload = json.loads(report_raw)
    _verify_report_evidence(report_payload)
    if (
        receipt.registrar_verifier_public_key_hex
        != report_payload["sealed_identities"]["registrar"][
            "verifier_public_key_hex"
        ]
    ):
        raise ValidationFailure("receipt registrar verifier key is detached")
    if verify_artifact_bytes:
        _verify_stage_artifacts(receipt, report_payload)
    if dict(receipt.candidate_metrics) != dict(report_payload["sealed_scoring"]["candidate_metrics"]):
        raise ValidationFailure("receipt candidate metrics are detached from sealed report")
    if dict(receipt.baseline_metrics) != dict(report_payload["sealed_scoring"]["baseline_metrics"]):
        raise ValidationFailure("receipt baseline metrics are detached from sealed report")
    for field_name in (
        "metric_margins",
        "higher_is_better",
        "uncertainty_rule",
        "multiplicity_rule",
        "secondary_benefit_rule",
    ):
        if getattr(receipt, field_name) != report_payload["sealed_scoring"][field_name]:
            raise ValidationFailure(
                f"receipt {field_name} is detached from sealed report"
            )


def execute_sealed_decision(
    *,
    adapter: CandidateAdapter,
    baseline_adapter: CandidateAdapter,
    registry: EvaluationRegistry,
    snapshot: FrozenEvaluationSnapshot | None,
    plan: SealedDecisionPlan,
    request: SealedDecisionRequest,
    validation_report: FiveOutputValidationReport,
    ledger: AtomicSealedLedger,
    rows: Sequence[EvalRow] | None = None,
    baseline_report: EvaluationReport | None = None,
) -> SealedDecisionReceipt:
    """Claim first, then load and jointly score the immutable sealed bytes."""
    if rows is not None or baseline_report is not None:
        raise ValidationFailure("sealed executor does not accept caller-loaded rows or reports")
    verify_five_output_validation_report(validation_report)
    _assert_plan(plan, request, registry, adapter, baseline_adapter, validation_report)
    if (
        ledger.registrar_identity != plan.registrar_id
        or ledger.registrar_kind != plan.registrar_kind
        or ledger.verifier_public_key.hex()
        != plan.registrar_verifier_public_key_hex
    ):
        raise ValidationFailure("ledger registrar is detached from decision plan")
    if snapshot is None or snapshot != plan.snapshot:
        raise ValidationFailure(
            "sealed snapshot locator/identity differs from preregistration"
        )
    claim = ledger._claim_for_executor(
        request,
        authority=_EXECUTOR_AUTHORITY,
    )

    raw_snapshot, loaded_rows = _load_and_verify_snapshot(snapshot, registry)
    candidate_report = evaluate_candidate(
        adapter,
        loaded_rows,
        registry,
        sealed_rows_snapshot=snapshot.fingerprint_map() if snapshot else None,
    )
    baseline_report_actual = evaluate_candidate(
        baseline_adapter,
        loaded_rows,
        registry,
        sealed_rows_snapshot=snapshot.fingerprint_map() if snapshot else None,
    )
    registered_suites = tuple(item.name for item in registry.split_plan.sealed_holdouts)
    for report in (candidate_report, baseline_report_actual):
        if tuple(report.holdout_reports) != registered_suites:
            raise ValidationFailure("sealed suite execution is missing, extra, or reordered")
        if any(
            report.holdout_reports[name].get("status") != "ok"
            for name in registered_suites
        ):
            raise ValidationFailure("every sealed suite must execute once with status=ok")
    suite_statuses = tuple((name, "ok") for name in registered_suites)
    scoring = _sealed_scoring_evidence(
        candidate_report, baseline_report_actual, loaded_rows, plan, registry
    )
    hard_gates = {
        **dict(candidate_report.hard_gate_results),
        "sealed_snapshot_verified": True,
        "sealed_suites_exact_once": True,
        "stage_manifest_complete": True,
        "five_output_validation_all_pass": validation_report.all_pass,
        "five_output_validation_hash_match": (
            validation_report.report_sha256 == plan.five_output_validation_sha256
        ),
        "receipt_content_addressed": True,
        "sealed_joint_scoring": True,
        "sealed_uncertainty_rule": plan.uncertainty_rule
        == "paired_series_cluster_bootstrap_95",
        "sealed_critical_strata": set(plan.critical_strata) == set(registered_suites),
        "sealed_multiplicity_rule": plan.multiplicity_rule
        == "all_metrics_and_critical_strata",
    }
    if set(hard_gates) != set(_required_sealed_hard_gates(registry)):
        raise ValidationFailure("runtime hard gates do not match frozen sealed gate set")

    artifact_dir = ledger.path.parent / "sealed-artifacts" / request.consumption_key
    artifact_dir.mkdir(parents=True, exist_ok=False)
    raw_locator = artifact_dir / "raw.json"
    raw_locator.write_bytes(raw_snapshot)
    raw_stage = SealedStage(
        name="raw",
        input_sha256=plan.snapshot.raw_bytes_sha256,
        output_sha256=_sha256_bytes(raw_snapshot),
        artifact_locator=str(raw_locator.resolve()),
    )
    features_stage = _write_stage(
        artifact_dir,
        "features",
        [
            {
                "row_id": row.row_id,
                "features": dict(row.feature_values),
                "available_at": {
                    name: canonical_timestamp(value)
                    for name, value in row.feature_available_at.items()
                },
            }
            for row in loaded_rows
        ],
        raw_stage.output_sha256,
    )
    state_stage = _write_stage(
        artifact_dir,
        "state_reconstruction",
        {
            "candidate_report": _evaluation_report_payload(candidate_report),
            "baseline_report": _evaluation_report_payload(baseline_report_actual),
            "candidate_artifact_sha256": plan.candidate_sha256,
            "baseline_artifact_sha256": plan.baseline_sha256,
        },
        features_stage.output_sha256,
    )
    calibration_stage = _write_stage(
        artifact_dir,
        "calibration",
        {
            "candidate_holdouts": _evaluation_report_payload(candidate_report)["holdout_reports"],
            "baseline_holdouts": _evaluation_report_payload(baseline_report_actual)["holdout_reports"],
            "transform_sha256": plan.transform_sha256,
        },
        state_stage.output_sha256,
    )
    report_payload = {
        "registry_sha256": registry.sha256(),
        "decision_plan_sha256": plan.plan_sha256,
        "sealed_outcome_consumption_key": request.consumption_key,
        "candidate_decision_id": request.decision_id,
        "sealed_identities": {
            "candidate": {
                "adapter_id": plan.candidate_adapter_id,
                "adapter_version": plan.candidate_adapter_version,
                "artifact_sha256": plan.candidate_sha256,
                "transform_sha256": plan.transform_sha256,
            },
            "baseline": {
                "adapter_id": plan.baseline_adapter_id,
                "adapter_version": plan.baseline_adapter_version,
                "artifact_sha256": plan.baseline_sha256,
            },
            "registrar": {
                "registrar_id": plan.registrar_id,
                "registrar_kind": plan.registrar_kind,
                "verifier_public_key_hex": (
                    plan.registrar_verifier_public_key_hex
                ),
                "trust_root_raw_sha256": plan.registrar_raw_sha256,
                "trust_root_object_sha256": plan.registrar_object_sha256,
            },
            "five_output_validation_sha256": (
                plan.five_output_validation_sha256
            ),
            **(
                {
                    "b2_validation_report_sha256": (
                        registry.b2_validation_report_sha256
                    )
                }
                if registry.b2_artifact_refs
                else {}
            ),
        },
        "candidate_report": _evaluation_report_payload(candidate_report),
        "baseline_report": _evaluation_report_payload(baseline_report_actual),
        "sealed_scoring": scoring,
        "suite_statuses": [list(item) for item in suite_statuses],
        "hard_gates": hard_gates,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
    }
    _verify_report_evidence(report_payload)
    report_locator = artifact_dir / "evaluation-report.json"
    report_raw = _canonical_bytes(report_payload)
    report_locator.write_bytes(report_raw)
    serialization_stage = _write_stage(
        artifact_dir,
        "serialization",
        report_payload,
        calibration_stage.output_sha256,
    )
    serving_stage = _write_stage(
        artifact_dir,
        "serving",
        {
            "serialized_sha256": serialization_stage.output_sha256,
            "replayed_candidate_probabilities": scoring["candidate_probabilities"],
            "row_ids": scoring["row_ids"],
        },
        serialization_stage.output_sha256,
    )
    stages = (
        raw_stage,
        features_stage,
        state_stage,
        calibration_stage,
        serialization_stage,
        serving_stage,
    )
    receipt_locator = artifact_dir / "receipt.json"
    receipt_base = SealedDecisionReceipt(
        request=request,
        stages=stages,
        suite_statuses=suite_statuses,
        candidate_metrics=dict(scoring["candidate_metrics"]),
        baseline_metrics=dict(scoring["baseline_metrics"]),
        metric_bounds=dict(scoring["metric_bounds"]),
        metric_decisions=dict(scoring["metric_decisions"]),
        critical_stratum_decisions=dict(scoring["critical_stratum_decisions"]),
        metric_margins=dict(scoring["metric_margins"]),
        higher_is_better=dict(scoring["higher_is_better"]),
        uncertainty_rule=str(scoring["uncertainty_rule"]),
        multiplicity_rule=str(scoring["multiplicity_rule"]),
        secondary_benefit_rule=str(scoring["secondary_benefit_rule"]),
        hard_gates=hard_gates,
        registrar_verifier_public_key_hex=(
            plan.registrar_verifier_public_key_hex
        ),
        evaluation_report_sha256=_sha256_bytes(report_raw),
        evaluation_report_locator=str(report_locator.resolve()),
        ledger_entry_sha256=claim.ledger_entry_sha256,
        receipt_locator=str(receipt_locator.resolve()),
        receipt_sha256="",
    )
    receipt = SealedDecisionReceipt(
        **{
            **receipt_base.__dict__,
            "receipt_sha256": canonical_sha256(receipt_base.unsigned_payload()),
        }
    )
    receipt_locator.write_bytes(
        _canonical_bytes({**receipt.unsigned_payload(), "receipt_sha256": receipt.receipt_sha256})
    )
    token = ledger._issue_executor_token(
        claim,
        artifact_manifest_sha256=canonical_sha256(
            [stage.to_payload() for stage in stages]
        ),
        evaluation_report_sha256=receipt.evaluation_report_sha256,
        authority=_EXECUTOR_AUTHORITY,
    )
    ledger.finalize(receipt, executor_claim_token=token)
    ledger.verify_receipt(receipt)
    return receipt
