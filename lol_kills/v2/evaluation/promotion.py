"""Promotion policy and deterministic rollback report."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Real
from typing import Any, Mapping

from .checks import ValidationFailure
from .sealed import (
    AtomicSealedLedger,
    SealedDecisionReceipt,
    verify_sealed_decision_receipt,
)
from .types import CONTRACT_TREE_SHA256


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    BLOCK = "BLOCK"
    REMAND = "REMAND"


@dataclass(frozen=True)
class PromotionGate:
    name: str
    passed: bool
    reason: str = ""


@dataclass(frozen=True)
class PromotionPlan:
    contract_tree_sha256: str
    split_registry_sha256: str
    metric_noninferiority_margins: Mapping[str, float]
    higher_is_better: Mapping[str, bool]


@dataclass(frozen=True)
class PromotionReport:
    model_id: str
    model_version: str
    decision: Decision
    hard_gates: Mapping[str, PromotionGate]
    metric_decisions: Mapping[str, bool]
    registry_sha256: str
    previous_manifest_pointer: str | None
    candidate_registry_sha256: str
    candidate_identity: Mapping[str, str] | None = None
    baseline_identity: Mapping[str, str] | None = None
    sealed_outcome_consumption_key: str | None = None
    candidate_decision_id: str | None = None
    five_output_validation_sha256: str | None = None


def _metric_pass(
    value: float,
    baseline: float,
    margin: float,
    higher_is_better: bool,
) -> bool:
    if higher_is_better:
        return value + margin >= baseline
    return value - margin <= baseline


def _unique_items(
    values: Mapping[str, Any],
    *,
    evidence_name: str,
) -> tuple[dict[str, Any] | None, str | None]:
    materialized: dict[str, Any] = {}
    for name, value in values.items():
        if not isinstance(name, str) or not name:
            return None, f"{evidence_name} contains an invalid metric or gate name"
        if name in materialized:
            return None, f"{evidence_name} contains duplicate evidence for '{name}'"
        materialized[name] = value
    return materialized, None


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def build_promotion_report(
    *,
    model_id: str,
    model_version: str,
    registry_sha256: str,
    candidate_registry_sha256: str,
    planned: PromotionPlan,
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
    hard_gates: Mapping[str, bool],
    sealed_receipt: SealedDecisionReceipt | None = None,
    sealed_ledger: AtomicSealedLedger | None = None,
    previous_manifest_pointer: str | None = None,
) -> PromotionReport:
    gate_values, gate_error = _unique_items(hard_gates, evidence_name="hard_gates")
    gate_objs = {
        name: PromotionGate(
            name=name,
            passed=(passed if isinstance(passed, bool) else False),
            reason=("" if isinstance(passed, bool) else "gate evidence must be boolean"),
        )
        for name, passed in (gate_values or {}).items()
    }

    def block(reason: str) -> PromotionReport:
        return PromotionReport(
            model_id=model_id,
            model_version=model_version,
            decision=Decision.BLOCK,
            hard_gates=gate_objs,
            metric_decisions={"promotion_evidence": False, reason: False},
            registry_sha256=registry_sha256,
            previous_manifest_pointer=previous_manifest_pointer,
            candidate_registry_sha256=candidate_registry_sha256,
        )

    def valid_sha256(value: Any) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        try:
            int(value, 16)
        except ValueError:
            return False
        return True

    if not all(
        valid_sha256(value)
        for value in (
            registry_sha256,
            candidate_registry_sha256,
            planned.split_registry_sha256,
        )
    ):
        return block("malformed registry hash evidence")
    if not (
        registry_sha256
        == candidate_registry_sha256
        == planned.split_registry_sha256
    ):
        return block("registry hash mismatch")
    if planned.contract_tree_sha256 != CONTRACT_TREE_SHA256:
        return block("contract tree hash mismatch")
    if gate_error is not None:
        return block(gate_error)
    if not gate_values:
        return block("hard gate evidence is empty")
    if any(not isinstance(passed, bool) for passed in gate_values.values()):
        return block("hard gate evidence must be boolean")
    if not all(gate_values.values()):
        return block("one or more hard gates failed")

    margins, margin_error = _unique_items(
        planned.metric_noninferiority_margins,
        evidence_name="planned metric margins",
    )
    directions, direction_error = _unique_items(
        planned.higher_is_better,
        evidence_name="planned metric directions",
    )
    candidates, candidate_error = _unique_items(
        candidate_metrics,
        evidence_name="candidate metrics",
    )
    baselines, baseline_error = _unique_items(
        baseline_metrics,
        evidence_name="baseline metrics",
    )
    for error in (
        margin_error,
        direction_error,
        candidate_error,
        baseline_error,
    ):
        if error is not None:
            return block(error)
    if not margins:
        return block("planned metric requirements are empty")

    planned_metric_names = set(margins)
    if set(directions or {}) != planned_metric_names:
        return block("planned metric directions do not exactly match margin requirements")
    if set(candidates or {}) != planned_metric_names:
        return block("candidate metric evidence does not exactly match the plan")
    if set(baselines or {}) != planned_metric_names:
        return block("baseline metric evidence does not exactly match the plan")

    metric_decisions: dict[str, bool] = {}
    all_pass = True
    for metric_name, margin in margins.items():
        candidate_value = candidates[metric_name]
        baseline_value = baselines[metric_name]
        higher_is_better = directions[metric_name]
        if not isinstance(higher_is_better, bool):
            return block(f"metric direction for '{metric_name}' must be boolean")
        if not _is_finite_number(margin) or float(margin) < 0:
            return block(f"metric margin for '{metric_name}' must be finite and nonnegative")
        if not _is_finite_number(candidate_value):
            return block(f"candidate metric '{metric_name}' must be finite")
        if not _is_finite_number(baseline_value):
            return block(f"baseline metric '{metric_name}' must be finite")
        pass_metric = _metric_pass(
            float(candidate_value),
            float(baseline_value),
            float(margin),
            higher_is_better,
        )
        metric_decisions[metric_name] = pass_metric
        if not pass_metric:
            all_pass = False

    if sealed_receipt is None or sealed_ledger is None:
        return block("verified sealed receipt and ledger are required")
    try:
        verify_sealed_decision_receipt(sealed_receipt)
        trusted = sealed_ledger.load_trusted_evidence(sealed_receipt)
    except ValidationFailure as exc:
        return block(f"sealed receipt verification failed: {exc}")
    sealed_scoring = dict(trusted["sealed_scoring"])
    trusted_candidate_metrics = dict(sealed_scoring["candidate_metrics"])
    trusted_baseline_metrics = dict(sealed_scoring["baseline_metrics"])
    trusted_hard_gates = dict(trusted["hard_gates"])
    identities = dict(trusted["sealed_identities"])
    trusted_candidate_identity = dict(identities["candidate"])
    trusted_baseline_identity = dict(identities["baseline"])
    if (
        model_id != trusted_candidate_identity["adapter_id"]
        or model_version != trusted_candidate_identity["adapter_version"]
    ):
        return block("promoted model identity is detached from sealed candidate")
    if sealed_receipt.request.consumption_key != trusted[
        "sealed_outcome_consumption_key"
    ]:
        return block("sealed outcome key is detached from trusted evidence")
    if sealed_receipt.request.decision_id != trusted["candidate_decision_id"]:
        return block("candidate decision ID is detached from trusted evidence")
    final_entry = sealed_ledger._final_entry(sealed_receipt)
    if final_entry.get("registrar_kind") != "production":
        return block("test-only registrar can never authorize production promotion")
    if dict(candidate_metrics) != trusted_candidate_metrics:
        return block("candidate metrics are detached from sealed receipt")
    if dict(baseline_metrics) != trusted_baseline_metrics:
        return block("baseline metrics are detached from sealed receipt")
    if dict(hard_gates) != trusted_hard_gates:
        return block("hard gates are detached from sealed receipt")
    if registry_sha256 != trusted["registry_sha256"]:
        return block("registry hash is detached from sealed receipt")
    if candidate_registry_sha256 != trusted["registry_sha256"]:
        return block("candidate registry hash is detached from sealed receipt")
    if planned.contract_tree_sha256 != trusted["contract_tree_sha256"]:
        return block("plan contract is detached from sealed receipt")
    if dict(planned.metric_noninferiority_margins) != dict(
        sealed_scoring["metric_margins"]
    ):
        return block("planned margins are detached from sealed preregistration")
    if dict(planned.higher_is_better) != dict(
        sealed_scoring["higher_is_better"]
    ):
        return block("planned directions are detached from sealed preregistration")
    if not all(dict(sealed_scoring["metric_decisions"]).values()):
        all_pass = False
    if not all(dict(sealed_scoring["critical_stratum_decisions"]).values()):
        all_pass = False

    if all_pass:
        decision = Decision.ACCEPT
    else:
        decision = Decision.REMAND

    return PromotionReport(
        model_id=model_id,
        model_version=model_version,
        decision=decision,
        hard_gates=gate_objs,
        metric_decisions=metric_decisions,
        registry_sha256=registry_sha256,
        previous_manifest_pointer=previous_manifest_pointer,
        candidate_registry_sha256=candidate_registry_sha256,
        candidate_identity=trusted_candidate_identity,
        baseline_identity=trusted_baseline_identity,
        sealed_outcome_consumption_key=trusted[
            "sealed_outcome_consumption_key"
        ],
        candidate_decision_id=trusted["candidate_decision_id"],
        five_output_validation_sha256=identities[
            "five_output_validation_sha256"
        ],
    )
