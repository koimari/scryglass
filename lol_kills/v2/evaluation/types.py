"""Typed contracts and serialization helpers used by the L2 evaluation harness."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple


CONTRACT_TREE_SHA256 = "fb3de56ddec943bc876cb795a8ada5695233f5fe615defe93f952ce299470517"


def ensure_utc(ts: datetime) -> datetime:
    if ts.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return ts.astimezone(timezone.utc)


def canonical_timestamp(ts: datetime) -> str:
    return ensure_utc(ts).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_utc_timestamp(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    parsed = datetime.fromisoformat(ts)
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    return ensure_utc(parsed)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return canonical_sha256(value)


@dataclass(frozen=True)
class EvalRow:
    """One evaluation row for split construction and feature-time audit checks."""

    row_id: str
    series_id: str
    series_resolved: bool
    event_start: datetime
    patch_id: str
    league_id: str
    league_tier: str
    region: str
    as_of: datetime
    label: int
    feature_values: Mapping[str, float]
    feature_available_at: Mapping[str, datetime]
    roster_id: str = ""
    roster_snapshot_id: Optional[str] = None
    roster_snapshot_time: Optional[datetime] = None
    roster_snapshot_stage: str = "operational"
    is_international_event: bool = False
    international_event_id: Optional[str] = None
    is_roster_change: bool = False
    champion_ids: Tuple[str, ...] = ()
    is_sparse_champion: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def features(self) -> Mapping[str, float]:
        """Compatibility read-only alias used by older fixtures."""
        return self.feature_values

    def as_dict(self) -> dict[str, Any]:
        return self.to_payload()

    def to_payload(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "series_id": self.series_id,
            "series_resolved": self.series_resolved,
            "event_start": canonical_timestamp(self.event_start),
            "patch_id": self.patch_id,
            "league_id": self.league_id,
            "league_tier": self.league_tier,
            "region": self.region,
            "as_of": canonical_timestamp(self.as_of),
            "label": self.label,
            "feature_values": dict(self.feature_values),
            "feature_available_at": {
                name: canonical_timestamp(available_at)
                for name, available_at in self.feature_available_at.items()
            },
            "roster_id": self.roster_id,
            "roster_snapshot_id": self.roster_snapshot_id,
            "roster_snapshot_time": (
                canonical_timestamp(self.roster_snapshot_time)
                if self.roster_snapshot_time is not None
                else None
            ),
            "roster_snapshot_stage": self.roster_snapshot_stage,
            "is_international_event": self.is_international_event,
            "international_event_id": self.international_event_id,
            "is_roster_change": self.is_roster_change,
            "champion_ids": list(self.champion_ids),
            "is_sparse_champion": self.is_sparse_champion,
            "metadata": dict(self.metadata),
        }

    def fingerprint(self) -> str:
        return canonical_sha256(self.to_payload())

    def with_mutated_label(self, label: int) -> "EvalRow":
        return replace(self, label=label)


@dataclass(frozen=True)
class MatchPrediction:
    """Structured output from one row-level scoring call."""

    row_id: str
    model_version: str
    mode: str
    raw_logit: Optional[float]
    raw_probability: float
    calibrated_probability: Optional[float] = None
    lower_95: Optional[float] = None
    upper_95: Optional[float] = None
    ledger: Mapping[str, float] = field(default_factory=dict)
    context: Mapping[str, Any] = field(default_factory=dict)

    def final_probability(self) -> float:
        return self.calibrated_probability if self.calibrated_probability is not None else self.raw_probability


MatchPredictionMap = Mapping[str, MatchPrediction]
TransferPredictionMap = Mapping[str, float]


@dataclass(frozen=True)
class SplitPartition:
    """Rolling-origin split with one atomic row block split."""

    name: str
    train_row_ids: Tuple[str, ...]
    validation_row_ids: Tuple[str, ...]
    calibration_row_ids: Tuple[str, ...]
    test_row_ids: Tuple[str, ...]

    @property
    def all_ids(self) -> Tuple[str, ...]:
        return tuple(dict.fromkeys(self.train_row_ids + self.validation_row_ids + self.calibration_row_ids + self.test_row_ids))


@dataclass(frozen=True)
class SealedHoldoutPartition:
    name: str
    row_ids: Tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SplitPlan:
    folds: Tuple[SplitPartition, ...]
    sealed_holdouts: Tuple[SealedHoldoutPartition, ...]


@dataclass(frozen=True)
class ArtifactRef:
    """Externally pinned content address for one executable B2 artifact."""

    artifact_id: str
    locator: str
    raw_sha256: str
    canonical_payload_sha256: str

    def to_payload(self) -> dict[str, str]:
        return {
            "artifact_id": self.artifact_id,
            "locator": self.locator,
            "raw_sha256": self.raw_sha256,
            "canonical_payload_sha256": self.canonical_payload_sha256,
        }


@dataclass(frozen=True)
class EvaluationRegistry:
    """Content-addressed registry manifest used by promotion and rollback checks."""

    contract_tree_sha256: str
    split_plan_id: str
    split_plan_sha256: str
    source_snapshot_id: str
    training_snapshot_id: str
    source_tree_sha256: str
    created_at: str
    bootstrap_seed: int
    split_plan: SplitPlan
    source_snapshot_sha256: str = ""
    training_snapshot_sha256: str = ""

    source_crosswalk_sha256: Mapping[str, str] = field(default_factory=dict)
    entity_crosswalk_sha256: Mapping[str, str] = field(default_factory=dict)
    league_crosswalk_sha256: Mapping[str, str] = field(default_factory=dict)

    metrics: Tuple[str, ...] = ("log_loss", "brier", "ece")
    estimands: Tuple[str, ...] = ("terminal_draft_score",)
    baseline_ids: Mapping[str, str] = field(default_factory=dict)
    baseline_artifact_hashes: Mapping[str, str] = field(default_factory=dict)
    candidate_artifact_hashes: Mapping[str, str] = field(default_factory=dict)

    served_transform_identities: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    subgroup_specs: Mapping[str, Any] = field(default_factory=dict)
    missingness_specs: Mapping[str, Any] = field(default_factory=dict)
    transfer_snapshot_hash: str = ""

    bootstrap_cluster_unit: str = "series"
    bootstrap_cluster_replicates: int = 2000
    bootstrap_cluster_size: int = 2000
    bootstrap_sensitivity_units: Tuple[str, ...] = ("region",)
    noninferiority_rules: Mapping[str, float] = field(default_factory=dict)
    noninferiority_higher_is_better: Mapping[str, bool] = field(default_factory=dict)
    noninferiority_provenance: str = ""
    coverage_procedure: Mapping[str, Any] = field(default_factory=dict)
    parity_tolerance: float = 1e-9
    invalidation_reasons: Tuple[str, ...] = field(default_factory=tuple)

    # Draft-order / protocol diagnostics are part of the registry contract.
    draft_order_analysis: Mapping[str, Any] = field(default_factory=dict)
    required_role_invariance_pairs: Tuple[tuple[str, str], ...] = field(default_factory=tuple)
    required_side_swap_pairs: Tuple[tuple[str, str], ...] = field(default_factory=tuple)

    split_config: Mapping[str, Any] = field(
        default_factory=lambda: {
            "train_ratio": 0.55,
            "validation_ratio": 0.20,
            "calibration_ratio": 0.15,
            "test_ratio": 0.10,
            "development_folds": 2,
            "temporal_holdout_ratio": 0.10,
        }
    )

    is_synthetic_registry: bool = False
    b2_artifact_refs: Tuple[ArtifactRef, ...] = ()
    b2_validation_report_sha256: str = ""

    def frozen_marker(self) -> dict[str, Any]:
        return {
            "contract_tree_sha256": self.contract_tree_sha256,
            "split_plan_id": self.split_plan_id,
            "split_plan_sha256": self.split_plan_sha256,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_sha256": self.source_snapshot_sha256,
            "training_snapshot_id": self.training_snapshot_id,
            "training_snapshot_sha256": self.training_snapshot_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "created_at": self.created_at,
            "bootstrap_seed": self.bootstrap_seed,
            "source_crosswalk_sha256": dict(self.source_crosswalk_sha256),
            "entity_crosswalk_sha256": dict(self.entity_crosswalk_sha256),
            "league_crosswalk_sha256": dict(self.league_crosswalk_sha256),
            "metrics": list(self.metrics),
            "estimands": list(self.estimands),
            "baseline_ids": dict(self.baseline_ids),
            "baseline_artifact_hashes": dict(self.baseline_artifact_hashes),
            "candidate_artifact_hashes": dict(self.candidate_artifact_hashes),
            "served_transform_identities": dict(self.served_transform_identities),
            "subgroup_specs": dict(self.subgroup_specs),
            "missingness_specs": dict(self.missingness_specs),
            "transfer_snapshot_hash": self.transfer_snapshot_hash,
            "bootstrap_cluster_unit": self.bootstrap_cluster_unit,
            "bootstrap_cluster_replicates": self.bootstrap_cluster_replicates,
            "bootstrap_cluster_size": self.bootstrap_cluster_size,
            "bootstrap_sensitivity_units": list(self.bootstrap_sensitivity_units),
            "noninferiority_rules": dict(self.noninferiority_rules),
            "noninferiority_higher_is_better": dict(self.noninferiority_higher_is_better),
            "noninferiority_provenance": self.noninferiority_provenance,
            "coverage_procedure": dict(self.coverage_procedure),
            "parity_tolerance": self.parity_tolerance,
            "invalidation_reasons": list(self.invalidation_reasons),
            "draft_order_analysis": dict(self.draft_order_analysis),
            "required_role_invariance_pairs": [list(pair) for pair in self.required_role_invariance_pairs],
            "required_side_swap_pairs": [list(pair) for pair in self.required_side_swap_pairs],
            "split_config": dict(self.split_config),
            "is_synthetic_registry": self.is_synthetic_registry,
            "b2_artifact_refs": [ref.to_payload() for ref in self.b2_artifact_refs],
            "b2_validation_report_sha256": self.b2_validation_report_sha256,
        }

    def to_payload(self) -> dict[str, Any]:
        return {
            **self.frozen_marker(),
            "split_plan": {
                "folds": [
                    {
                        "name": fold.name,
                        "train_row_ids": list(fold.train_row_ids),
                        "validation_row_ids": list(fold.validation_row_ids),
                        "calibration_row_ids": list(fold.calibration_row_ids),
                        "test_row_ids": list(fold.test_row_ids),
                    }
                    for fold in self.split_plan.folds
                ],
                "sealed_holdouts": [
                    {
                        "name": holdout.name,
                        "row_ids": list(holdout.row_ids),
                        "metadata": dict(holdout.metadata),
                    }
                    for holdout in self.split_plan.sealed_holdouts
                ],
            },
            "source_crosswalk_sha256": dict(self.source_crosswalk_sha256),
            "entity_crosswalk_sha256": dict(self.entity_crosswalk_sha256),
            "league_crosswalk_sha256": dict(self.league_crosswalk_sha256),
            "metrics": list(self.metrics),
            "estimands": list(self.estimands),
            "baseline_ids": dict(self.baseline_ids),
            "baseline_artifact_hashes": dict(self.baseline_artifact_hashes),
            "candidate_artifact_hashes": dict(self.candidate_artifact_hashes),
            "served_transform_identities": dict(self.served_transform_identities),
            "subgroup_specs": dict(self.subgroup_specs),
            "missingness_specs": dict(self.missingness_specs),
            "transfer_snapshot_hash": self.transfer_snapshot_hash,
            "bootstrap_cluster_unit": self.bootstrap_cluster_unit,
            "bootstrap_cluster_replicates": self.bootstrap_cluster_replicates,
            "bootstrap_cluster_size": self.bootstrap_cluster_size,
            "bootstrap_sensitivity_units": list(self.bootstrap_sensitivity_units),
            "noninferiority_rules": dict(self.noninferiority_rules),
            "noninferiority_higher_is_better": dict(self.noninferiority_higher_is_better),
            "noninferiority_provenance": self.noninferiority_provenance,
            "coverage_procedure": dict(self.coverage_procedure),
            "parity_tolerance": self.parity_tolerance,
            "invalidation_reasons": list(self.invalidation_reasons),
            "draft_order_analysis": dict(self.draft_order_analysis),
            "required_role_invariance_pairs": [list(pair) for pair in self.required_role_invariance_pairs],
            "required_side_swap_pairs": [list(pair) for pair in self.required_side_swap_pairs],
            "split_config": dict(self.split_config),
            "is_synthetic_registry": self.is_synthetic_registry,
            "b2_artifact_refs": [ref.to_payload() for ref in self.b2_artifact_refs],
            "b2_validation_report_sha256": self.b2_validation_report_sha256,
        }

    def sha256(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class CalibrationState:
    kind: str
    intercept: float
    slope: float
    model_sha256: str
    status: str = "ok"
    reason: str = ""
    covariance: Tuple[Tuple[float, ...], ...] = ()
    standard_errors: Tuple[float, ...] = ()
    support: int = 0
    parameters: Mapping[str, Any] = field(default_factory=dict)
    boundary_epsilon: float = 1e-9
    symmetry: str = "none"
    calibration_row_sha256: str = ""
    selection_sha256: str = ""
    code_sha256: str = ""
    config_sha256: str = ""


@dataclass(frozen=True)
class CandidateFit:
    adapter_id: str
    adapter_version: str
    fit_digest: str


class CandidateAdapter(Protocol):
    """Protocol required for both baseline and future L4-L10 candidates."""

    adapter_id: str
    adapter_version: str
    source_tree_sha256: str
    runtime_artifact_sha256: str
    runtime_artifact_manifest_sha256: str
    runtime_transform_manifest_sha256: str
    served_transform_sha256: str
    serialized_transform_sha256: str
    terminal_probability_wording_approved: bool
    prefix_probability_wording_approved: Mapping[str, bool]

    def fit(self, rows: Sequence[EvalRow], *, split_name: str) -> CandidateFit:
        ...


class TransferComparisonAdapter(Protocol):
    """Protocol for L6-provided transfer/ablation probability bundles.

    Implementations are expected to be pure, deterministic, and index rows by row_id.
    """

    adapter_id: str
    adapter_version: str
    source_tree_sha256: str
    snapshot_sha256: str

    def predict_ontology_free(self, rows: Sequence[EvalRow]) -> Mapping[str, float]:
        ...

    def predict_transfer_ablation(self, rows: Sequence[EvalRow]) -> Mapping[str, float]:
        ...


class SnapshotAdapter(Protocol):
    """L1 snapshot bridge contract for future adapter wiring."""

    snapshot_id: str
    source_tree_sha256: str

    def rows(self) -> Sequence[EvalRow]:
        ...
