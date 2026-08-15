"""Run L2 candidate evaluations under a frozen registry and hard-gate protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math
import statistics

from .bootstrap import BootstrapResult, grouped_bootstrap_sensitivity, series_cluster_bootstrap
from .calibration import fit_calibration
from .b2_pipeline import (
    B2_REQUIRED_HARD_GATES,
    build_b2_validation_report,
    verify_b2_validation_report,
)
from .checks import (
    ValidationFailure,
    assert_bootstrap_not_map_level,
    assert_partition_disjoint,
    assert_exact_roster,
    assert_invariant_ledger_reconciles,
    assert_no_final_roster_join_backwards,
    assert_no_future_feature_joins,
    assert_no_label_leakage,
    assert_rows_have_binary_labels,
    assert_prefix_probability_wording,
    assert_python_runtime_probabilities,
    assert_row_cutoff,
    assert_row_prediction_values,
    assert_role_invariance,
    assert_runtime_transform_identity,
    assert_runtime_artifact_identity,
    assert_required_role_invariance,
    assert_required_side_swap,
    assert_draft_order_diagnostics,
    assert_sealed_rows_immutable,
    assert_series_atomicity,
    assert_side_swap,
    assert_split_partitions_disjoint,
    assert_terminal_probability_wording,
    assert_transform_identity,
    assert_unresolved_series_not_in_primary_bootstrap,
)
from .metrics import (
    MetricSuite,
    auc_score,
    calibration_intercept_and_slope,
    expected_calibration_error,
    log_loss,
    macro_region_log_loss,
)
from .metrics import brier_score as score_brier
from .types import (
    CandidateAdapter,
    EvalRow,
    EvaluationRegistry,
    SealedHoldoutPartition,
    MatchPrediction,
    MatchPredictionMap,
    TransferComparisonAdapter,
    SnapshotAdapter,
    CONTRACT_TREE_SHA256,
    canonical_sha256,
)
from .splitter import split_plan_payload_for_hash


_SENSITIVITY_UNIT_BUILDERS = {
    "league": lambda row_id, rows_by_id: rows_by_id[row_id].league_id,
    "region": lambda row_id, rows_by_id: rows_by_id[row_id].region,
    "patch": lambda row_id, rows_by_id: rows_by_id[row_id].patch_id,
    "tournament": lambda row_id, rows_by_id: rows_by_id[row_id].metadata.get("tournament_id", rows_by_id[row_id].series_id),
}


def _per_row_log_loss(labels: Sequence[int], probs: Sequence[float]) -> list[float]:
    return [
        -(label * math.log(max(1e-12, prob)) + (1 - label) * math.log(max(1e-12, 1 - prob)))
        for label, prob in zip(labels, probs)
    ]


def _per_row_brier(labels: Sequence[int], probs: Sequence[float]) -> list[float]:
    return [(float(label) - float(prob)) ** 2 for label, prob in zip(labels, probs)]


def _rows_by_id(rows: Sequence[EvalRow]) -> dict[str, EvalRow]:
    mapping = {row.row_id: row for row in rows}
    if len(mapping) != len(rows):
        raise ValidationFailure("row_id duplicates are not allowed")
    return mapping


def _ordered_rows(fold_row_ids: Sequence[str], rows_by_id: Mapping[str, EvalRow]) -> list[EvalRow]:
    missing = [row_id for row_id in fold_row_ids if row_id not in rows_by_id]
    if missing:
        raise ValidationFailure(f"fold references unknown row ids: {missing}")
    return [rows_by_id[row_id] for row_id in fold_row_ids]


def _as_prediction_map(predictions: Sequence[MatchPrediction]) -> MatchPredictionMap:
    mapped: dict[str, MatchPrediction] = {}
    for pred in predictions:
        mapped[pred.row_id] = pred
    return mapped


def _as_probability_map(values: Mapping[str, float], rows: Sequence[EvalRow], *, field_name: str) -> dict[str, float]:
    expected_row_ids = [row.row_id for row in rows]
    if len(set(expected_row_ids)) != len(expected_row_ids):
        raise ValidationFailure(f"{field_name} row payload contains duplicate row IDs")

    value_row_ids = set(values.keys())
    expected_ids = set(expected_row_ids)
    extra_ids = sorted(value_row_ids - expected_ids)
    if extra_ids:
        raise ValidationFailure(f"{field_name} contains unexpected row IDs: {extra_ids}")

    missing_ids = sorted(expected_ids - value_row_ids)
    if missing_ids:
        raise ValidationFailure(f"{field_name} missing row {missing_ids[0] if len(missing_ids) == 1 else missing_ids}")

    mapped: dict[str, float] = {}
    for row_id in expected_row_ids:
        prob = float(values[row_id])
        if not math.isfinite(prob):
            raise ValidationFailure(
                f"{field_name} for row {row_id} is not finite: {prob}"
            )
        if not 0.0 <= prob <= 1.0:
            raise ValidationFailure(
                f"{field_name} for row {row_id} outside [0, 1]: {prob}"
            )
        mapped[row_id] = prob
    return mapped


def _build_fold_metric(
    labels: Sequence[int],
    predictions: Sequence[MatchPrediction],
    row_ids: Sequence[str] | None = None,
) -> MetricSuite:
    if not labels:
        raise ValidationFailure("fold metric computation requires non-empty labels")
    if not predictions:
        raise ValidationFailure("fold metric computation requires non-empty predictions")
    if len(labels) != len(predictions):
        raise ValidationFailure(
            f"label/prediction length mismatch: {len(labels)} labels vs {len(predictions)} predictions"
        )
    if row_ids is not None and len(predictions) != len(row_ids):
        raise ValidationFailure(
            f"row/prediction length mismatch: {len(row_ids)} rows vs {len(predictions)} predictions"
        )
    if row_ids is not None:
        if len(set(row_ids)) != len(row_ids):
            raise ValidationFailure("fold predictions must be row-unique")
        pred_row_ids = [pred.row_id for pred in predictions]
        if len(set(pred_row_ids)) != len(pred_row_ids):
            raise ValidationFailure("fold predictions contain duplicate row IDs")
        if pred_row_ids != list(row_ids):
            raise ValidationFailure("fold predictions are not row-aligned")

    probs = [pred.final_probability() for pred in predictions]
    calibration = calibration_intercept_and_slope(labels, probs)
    return MetricSuite(
        log_loss=log_loss(labels, probs),
        brier=score_brier(labels, probs),
        ece=expected_calibration_error(labels, probs),
        auc=auc_score(labels, probs),
        calibration_status=calibration.status,
        calibration_reason=calibration.reason,
        calibration_support=calibration.support,
        calibration_intercept=calibration.intercept,
        calibration_slope=calibration.slope,
    )


def _assert_frozen_registry(registry: EvaluationRegistry) -> None:
    def _require_non_empty_mapping(name: str, value: Mapping[str, Any], *, allow_zero: bool = False) -> None:
        if not value:
            raise ValidationFailure(f"registry {name} is required")
        if not allow_zero and all(not v for v in value.values()):
            raise ValidationFailure(f"registry {name} must not be empty")

    if registry.contract_tree_sha256 != CONTRACT_TREE_SHA256:
        raise ValidationFailure("registry contract hash does not match docs/model-v2")
    if not registry.split_plan.folds:
        raise ValidationFailure("registry has no development folds")
    if not registry.source_snapshot_id:
        raise ValidationFailure("registry source_snapshot_id is required")
    if len(registry.source_snapshot_sha256) != 64:
        raise ValidationFailure("registry source_snapshot_sha256 is required")
    if not registry.training_snapshot_id:
        raise ValidationFailure("registry training_snapshot_id is required")
    if len(registry.training_snapshot_sha256) != 64:
        raise ValidationFailure("registry training_snapshot_sha256 is required")
    if registry.split_plan_sha256 != canonical_sha256(split_plan_payload_for_hash(registry.split_plan, registry.split_plan_id)):
        raise ValidationFailure("registry split_plan sha256 does not match split payload")
    if not registry.metrics:
        raise ValidationFailure("registry metrics list is required")
    if not registry.estimands:
        raise ValidationFailure("registry estimands list is required")
    if not registry.split_plan_id:
        raise ValidationFailure("registry split_plan_id is required")
    if not registry.source_tree_sha256:
        raise ValidationFailure("registry source_tree_sha256 is required")
    _require_non_empty_mapping("source_crosswalk_sha256", registry.source_crosswalk_sha256)
    _require_non_empty_mapping("entity_crosswalk_sha256", registry.entity_crosswalk_sha256)
    _require_non_empty_mapping("league_crosswalk_sha256", registry.league_crosswalk_sha256)
    if not registry.baseline_ids:
        raise ValidationFailure("registry baseline ids are required")
    if not registry.baseline_artifact_hashes:
        raise ValidationFailure("registry baseline artifact hashes are required")
    if not registry.candidate_artifact_hashes:
        raise ValidationFailure("registry candidate artifact hashes are required")
    _require_non_empty_mapping(
        "served_transform_identities",
        registry.served_transform_identities,
        allow_zero=False,
    )
    for transform_id, transform_record in registry.served_transform_identities.items():
        if not isinstance(transform_record, Mapping):
            raise ValidationFailure(
                f"served_transform_identities[{transform_id}] must be a mapping"
            )
    if registry.subgroup_specs is None or not registry.subgroup_specs:
        raise ValidationFailure("registry subgroup specs are required")
    if registry.missingness_specs is None or not registry.missingness_specs:
        raise ValidationFailure("registry missingness specs are required")
    if registry.bootstrap_cluster_replicates <= 0:
        raise ValidationFailure("registry bootstrap_cluster_replicates must be positive")
    if registry.bootstrap_cluster_size <= 0:
        raise ValidationFailure("registry bootstrap_cluster_size must be positive")
    if registry.bootstrap_seed is None:
        raise ValidationFailure("registry bootstrap_seed is required")
    if not registry.bootstrap_cluster_unit:
        raise ValidationFailure("registry bootstrap_cluster_unit is required")
    if not registry.noninferiority_rules:
        raise ValidationFailure("registry noninferiority rules are required")
    if not registry.noninferiority_higher_is_better:
        raise ValidationFailure("registry noninferiority_higher_is_better is required")
    if not registry.noninferiority_provenance:
        raise ValidationFailure("registry noninferiority_provenance is required")
    if not registry.coverage_procedure:
        raise ValidationFailure("registry coverage_procedure is required")
    if registry.parity_tolerance < 0:
        raise ValidationFailure("registry parity_tolerance must be non-negative")
    if not registry.transfer_snapshot_hash:
        raise ValidationFailure("registry transfer_snapshot_hash is required")
    if not registry.invalidation_reasons:
        raise ValidationFailure("registry invalidation_reasons are required")
    if registry.is_synthetic_registry and registry.source_tree_sha256 == "a" * 64:
        raise ValidationFailure("synthetic registry must not use all-a source_tree hash")
    if registry.is_synthetic_registry and not registry.split_config:
        raise ValidationFailure("synthetic registry must keep split_config")
    if not registry.split_config:
        raise ValidationFailure("registry split_config is required")
    if not registry.bootstrap_sensitivity_units:
        raise ValidationFailure("registry bootstrap_sensitivity_units is required")
    if registry.bootstrap_cluster_size <= 0:
        raise ValidationFailure("registry bootstrap_cluster_size must be positive")
    if registry.bootstrap_seed is None:
        raise ValidationFailure("registry bootstrap_seed is required")
    if not isinstance(registry.required_role_invariance_pairs, tuple):
        raise ValidationFailure("registry required_role_invariance_pairs must be a tuple")
    if not isinstance(registry.required_side_swap_pairs, tuple):
        raise ValidationFailure("registry required_side_swap_pairs must be a tuple")
    assert_split_partitions_disjoint(registry)

def _run_predict(
    adapter: CandidateAdapter,
    fit_state,
    rows: Sequence[EvalRow],
    *,
    mode: str,
    prefix: str | None = None,
) -> tuple[MatchPrediction, ...]:
    predictions = tuple(adapter.predict(fit_state, rows, mode=mode, prefix=prefix))
    if len(predictions) != len(rows):
        raise ValidationFailure(f"adapter {adapter.adapter_id}: predict length mismatch")
    seen: set[str] = set()
    for pred, row in zip(predictions, rows):
        if pred.row_id != row.row_id:
            raise ValidationFailure(
                f"adapter {adapter.adapter_id}: predict row mismatch ({pred.row_id} != {row.row_id})"
            )
        if pred.row_id in seen:
            raise ValidationFailure(f"adapter {adapter.adapter_id}: duplicate row id {pred.row_id}")
        seen.add(pred.row_id)
    return predictions


def _run_calibration(
    adapter: CandidateAdapter,
    fit_state,
    calibration_rows: Sequence[EvalRow],
    calibration_predictions: Sequence[MatchPrediction],
    apply_to_predictions: Sequence[MatchPrediction],
) -> tuple[MatchPrediction, ...]:
    if len(calibration_predictions) != len(calibration_rows):
        raise ValidationFailure(
            "calibration rows and calibration predictions are mismatched"
        )
    calibration_row_ids = [row.row_id for row in calibration_rows]
    calibration_prediction_row_ids = [pred.row_id for pred in calibration_predictions]
    if calibration_prediction_row_ids != calibration_row_ids:
        raise ValidationFailure("calibration predictions are not aligned with calibration rows")
    if len(set(calibration_row_ids)) != len(calibration_row_ids):
        raise ValidationFailure("calibration rows contain duplicate row IDs")
    if len(set(calibration_prediction_row_ids)) != len(calibration_prediction_row_ids):
        raise ValidationFailure("calibration predictions contain duplicate row IDs")

    if not calibration_rows:
        calibration_state = fit_calibration((), (), adapter.adapter_id)
    else:
        calibration_state = adapter.fit_calibration(
            calibration_rows,
            calibration_predictions,
            mode="terminal",
        )

    calibrated = tuple(
        adapter.apply_calibration(calibration_state, apply_to_predictions, mode="terminal")
    ) if apply_to_predictions else ()

    if len(calibrated) != len(apply_to_predictions):
        raise ValidationFailure(f"adapter {adapter.adapter_id}: calibration output length mismatch")

    if apply_to_predictions:
        apply_row_ids = [pred.row_id for pred in apply_to_predictions]
        if len(set(apply_row_ids)) != len(apply_row_ids):
            raise ValidationFailure("calibration target contains duplicate row IDs")
        for original, calibrated_pred in zip(apply_to_predictions, calibrated):
            if original.row_id != calibrated_pred.row_id:
                raise ValidationFailure(
                    "adapter calibration changed row ordering or IDs"
                )
    return calibrated


def _macro_region_metrics(
    predictions: MatchPredictionMap,
    rows_by_id: Mapping[str, EvalRow],
) -> tuple[float, dict[str, float]]:
    per_region: dict[str, list[tuple[int, float]]] = {}
    for row_id, pred in predictions.items():
        row = rows_by_id[row_id]
        per_region.setdefault(row.region, []).append((row.label, pred.final_probability()))

    region_metrics = {
        region: log_loss([row[0] for row in rows], [row[1] for row in rows])
        for region, rows in per_region.items()
    }
    return macro_region_log_loss(region_metrics), region_metrics


def _sensitivity_bootstrap_for_unit(
    unit: str,

    deltas: list[float],
    clusters: list[str],
    resolved: list[bool],
    row_ids: list[str],
    rows_by_id: Mapping[str, EvalRow],
    seed: int,
    *,
    n_boot: int,
) -> BootstrapResult:
    if unit not in _SENSITIVITY_UNIT_BUILDERS:
        raise ValidationFailure(f"unsupported sensitivity unit '{unit}'")

    if not row_ids:
        raise ValidationFailure("sensitivity bootstrap requires rows")
    if not deltas:
        raise ValidationFailure("sensitivity bootstrap requires deltas")

    row_ids_set = set(row_ids)
    for row_id in row_ids_set:
        if row_id not in rows_by_id:
            raise ValidationFailure(f"sensitivity row {row_id} is unknown")

    extractor = _SENSITIVITY_UNIT_BUILDERS[unit]
    group_values = sorted({extractor(row_id, rows_by_id) for row_id in row_ids})
    if len(group_values) <= 1:
        raise ValidationFailure(f"sensitivity unit '{unit}' has fewer than two groups")

    return grouped_bootstrap_sensitivity(
        deltas=deltas,
        cluster_ids=clusters,
        resolved_mask=resolved,
        row_ids=row_ids,
        group_fn=lambda row_id: extractor(row_id, rows_by_id),
        n_boot=n_boot,
        random_seed=seed,
        cluster_unit=unit,
    )


def _parse_patch_to_int(patch_id: str) -> int:
    parts = patch_id.split(".")
    if len(parts) != 2:
        raise ValidationFailure(f"invalid patch id in holdout protocol: {patch_id}")
    return int(parts[0]) * 10000 + int(parts[1])


def _canonical_holdout_protocol(metadata: Mapping[str, Any]) -> str:
    candidate = (
        metadata.get("protocol")
        or metadata.get("mode")
        or metadata.get("protocol_name")
        or ""
    )
    normalized = str(candidate).strip().lower()
    aliases = {
        "latest_block": "temporal",
        "latestblock": "temporal",
        "temporal": "temporal",
        "future_patch": "future_patch",
        "future": "future_patch",
        "future_patch_holdout": "future_patch",
        "league_leave_one_out": "league_leave_one_out",
        "leave_one_out": "league_leave_one_out",
        "leave_one_tier1_league": "league_leave_one_out",
        "international_event": "international_event",
        "international": "international_event",
        "roster_change": "roster_change",
        "new_roster": "roster_change",
        "sparse_new_champion": "sparse_new_champion",
        "sparse_or_zero_play": "sparse_new_champion",
        "masked_champion_residual": "masked_champion_residual",
        "masked_residual": "masked_champion_residual",
        "archetype_transfer": "archetype_transfer",
        "archetype_transfer_residual": "archetype_transfer",
    }
    return aliases.get(normalized, normalized)


def _build_holdout_training_rows(
    protocol: str,
    holdout: SealedHoldoutPartition,
    rows_by_id: Mapping[str, EvalRow],
    registry: EvaluationRegistry,
) -> tuple[tuple[EvalRow, ...], tuple[EvalRow, ...], tuple[str, ...]]:
    holdout_row_ids = tuple(holdout.row_ids)
    holdout_meta = dict(holdout.metadata)
    holdout_set = set(holdout_row_ids)

    development_ids = {
        row_id
        for fold in registry.split_plan.folds
        for row_id in fold.all_ids
    }
    calibration_ids = {
        row_id
        for fold in registry.split_plan.folds
        for row_id in fold.calibration_row_ids
    }
    if not development_ids:
        raise ValidationFailure("sealed execution requires a frozen development union")

    # Sealed execution may only refit from resolved, registry-assigned
    # development identities. Registered calibration identities remain
    # dedicated and are never passed to model fit.
    eligible_ids = development_ids - holdout_set
    training_rows = [
        rows_by_id[row_id]
        for row_id in development_ids - calibration_ids
        if row_id in eligible_ids
        and row_id in rows_by_id
        and rows_by_id[row_id].series_resolved
    ]
    calibration_rows = [
        rows_by_id[row_id]
        for row_id in calibration_ids
        if row_id in eligible_ids
        and row_id in rows_by_id
        and rows_by_id[row_id].series_resolved
    ]

    # Contractual protocol-specific exclusions: these rows are removed from the
    # candidate refit and its dedicated calibration support.
    if protocol == "league_leave_one_out":
        league_id = str(holdout_meta.get("league_id") or "").strip()
        if not league_id:
            raise ValidationFailure(
                f"league holdout '{holdout.name}' is missing league_id metadata"
            )
        training_rows = [row for row in training_rows if row.league_id != league_id]
        calibration_rows = [row for row in calibration_rows if row.league_id != league_id]
        # Keep holdout rows exactly as declared, for explicit coverage accounting.

    elif protocol == "international_event":
        event_id = str(holdout_meta.get("event_id") or holdout_meta.get("international_event_id") or "").strip()
        if not event_id:
            raise ValidationFailure(
                f"international holdout '{holdout.name}' is missing event_id metadata"
            )
        training_rows = [
            row for row in training_rows
            if not (row.is_international_event and row.international_event_id == event_id)
        ]
        calibration_rows = [
            row for row in calibration_rows
            if not (row.is_international_event and row.international_event_id == event_id)
        ]

    elif protocol == "future_patch":
        # Use the explicit temporal boundary from development folds so "future patch"
        # excludes all later patches from refit.
        dev_rows = [
            rows_by_id[row_id]
            for row_id in development_ids
            if row_id in rows_by_id and rows_by_id[row_id].series_resolved
        ]
        if not dev_rows:
            raise ValidationFailure("future-patch holdout requires development rows")
        cutoff = max(_parse_patch_to_int(row.patch_id) for row in dev_rows)
        training_rows = [row for row in training_rows if _parse_patch_to_int(row.patch_id) <= cutoff]
        calibration_rows = [row for row in calibration_rows if _parse_patch_to_int(row.patch_id) <= cutoff]

    elif protocol == "roster_change":
        roster_ids = {
            rows_by_id[row_id].roster_id
            for row_id in holdout_row_ids
            if row_id in rows_by_id and rows_by_id[row_id].roster_id
        }
        if not roster_ids:
            raise ValidationFailure(
                f"roster-change holdout '{holdout.name}' has no roster ids"
            )
        training_rows = [
            row for row in training_rows if row.roster_id not in roster_ids
        ]
        calibration_rows = [
            row for row in calibration_rows if row.roster_id not in roster_ids
        ]

    elif protocol in {"sparse_new_champion", "masked_champion_residual", "archetype_transfer"}:
        # Already enforced by explicit holdout membership for this protocol.
        pass

    elif protocol != "temporal":
        raise ValidationFailure(f"unknown holdout protocol '{protocol}' for '{holdout.name}'")

    ordering = lambda row: (row.event_start, row.row_id)
    return (
        tuple(sorted(training_rows, key=ordering)),
        tuple(sorted(calibration_rows, key=ordering)),
        holdout_row_ids,
    )


def _run_holdout(
    adapter: CandidateAdapter,
    holdout: SealedHoldoutPartition,
    rows_by_id: Mapping[str, EvalRow],
    registry: EvaluationRegistry,
) -> dict[str, Any]:
    protocol = _canonical_holdout_protocol(dict(holdout.metadata))
    training_rows, calibration_rows, holdout_row_ids = _build_holdout_training_rows(
        protocol,
        holdout,
        rows_by_id,
        registry,
    )
    if not holdout_row_ids:
        return {
            "n_total": 0,
            "n_available": 0,
            "coverage": 0.0,
            "available_fraction": 0.0,
            "n_resolved": 0,
            "meta": dict(holdout.metadata),
            "status": "unavailable",
            "fail_reason": "holdout is empty",
        }

    if not training_rows or not calibration_rows:
        missing_support = "model-fit" if not training_rows else "calibration"
        return {
            "n_total": len(holdout_row_ids),
            "n_available": 0,
            "coverage": 0.0,
            "available_fraction": 0.0,
            "n_resolved": 0,
            "meta": dict(holdout.metadata),
            "protocol": protocol,
            "status": "unavailable",
            "fail_reason": f"{missing_support} support unavailable after protocol filtering",
            "fit_row_ids": tuple(row.row_id for row in training_rows),
            "calibration_row_ids": tuple(row.row_id for row in calibration_rows),
            "scored_row_ids": (),
        }

    holdout_rows = _ordered_rows(holdout_row_ids, rows_by_id)
    if not holdout_rows:
        raise ValidationFailure(f"holdout '{holdout.name}' has no resolvable rows")

    fit_state = adapter.fit(training_rows, split_name=f"holdout-{holdout.name}")
    raw_holdout_predictions = _run_predict(adapter, fit_state, holdout_rows, mode="terminal")
    assert_row_prediction_values(raw_holdout_predictions)
    assert_no_label_leakage(adapter, fit_state, holdout_rows)

    calibration_predictions = _run_predict(
        adapter,
        fit_state,
        calibration_rows,
        mode="terminal",
    )
    assert_row_prediction_values(calibration_predictions)

    calibration_state = adapter.fit_calibration(
        calibration_rows,
        calibration_predictions,
        mode="terminal",
    )
    calibrated_predictions = tuple(
        adapter.apply_calibration(
            calibration_state,
            raw_holdout_predictions,
            mode="terminal",
        )
    )
    assert_row_prediction_values(calibrated_predictions)

    n_resolved = sum(1 for row in holdout_rows if row.series_resolved)

    raw_labels = [row.label for row in holdout_rows]
    raw_probs = [pred.final_probability() for pred in calibrated_predictions]

    coverage = len(calibrated_predictions) / len(holdout_rows)
    report: dict[str, Any] = {
        "n_total": len(holdout_rows),
        "n_available": len(calibrated_predictions),
        "coverage": float(coverage),
        "available_fraction": float(coverage),
        "n_resolved": n_resolved,
        "meta": dict(holdout.metadata),
        "protocol": protocol,
        "status": "ok",
        "fit_row_ids": tuple(row.row_id for row in training_rows),
        "calibration_row_ids": tuple(row.row_id for row in calibration_rows),
        "scored_row_ids": tuple(pred.row_id for pred in calibrated_predictions),
        "scored_probabilities": {
            pred.row_id: pred.final_probability()
            for pred in calibrated_predictions
        },
        "fit_state": {
            "adapter_id": fit_state.adapter_id,
            "adapter_version": fit_state.adapter_version,
            "fit_digest": fit_state.fit_digest,
        },
        "calibration_state": {
            "kind": calibration_state.kind,
            "intercept": calibration_state.intercept,
            "slope": calibration_state.slope,
            "model_sha256": calibration_state.model_sha256,
        },
    }
    report.update(
        {
            "log_loss": log_loss(raw_labels, raw_probs),
            "brier": score_brier(raw_labels, raw_probs),
            "status": "ok" if coverage == 1.0 else "degraded",
        }
    )
    if coverage < 1.0:
        report["fail_reason"] = "partial holdout coverage"

    return report


def _collect_holdout_reports(
    adapter: CandidateAdapter,
    registry: EvaluationRegistry,
    rows_by_id: Mapping[str, EvalRow],
) -> dict[str, Mapping[str, Any]]:
    reports: dict[str, Mapping[str, Any]] = {}
    for holdout in registry.split_plan.sealed_holdouts:
        reports[holdout.name] = _run_holdout(adapter, holdout, rows_by_id, registry)
    return reports


def _named_international_metrics(
    predictions: MatchPredictionMap,
    rows_by_id: Mapping[str, EvalRow],
) -> dict[str, dict[str, float]]:
    by_event: dict[str, list[MatchPrediction]] = {}
    for row_id, pred in predictions.items():
        row = rows_by_id[row_id]
        if not row.is_international_event or not row.international_event_id:
            continue
        by_event.setdefault(row.international_event_id, []).append(pred)

    result: dict[str, dict[str, float]] = {}
    for event_id, preds in by_event.items():
        labels = [rows_by_id[pred.row_id].label for pred in preds]
        probs = [pred.final_probability() for pred in preds]
        result[event_id] = {
            "n": len(preds),
            "log_loss": log_loss(labels, probs),
            "brier": score_brier(labels, probs),
        }
    return result


def _derive_pairs(rows: Sequence[EvalRow], *, key: str) -> dict[str, str]:
    pending: dict[str, str] = {}
    for row in rows:
        partner = row.metadata.get(key)
        if isinstance(partner, str):
            pending[row.row_id] = partner

    pairs: dict[str, str] = {}
    used: set[str] = set()
    for left, right in pending.items():
        if left in used:
            continue
        if right not in pending:
            continue
        if pending.get(right) != left:
            continue
        pairs[left] = right
        used.add(left)
        used.add(right)
    return pairs


def _aggregate_metric_suite(folds: Sequence[MetricSuite]) -> dict[str, Any]:
    if not folds:
        return {
            "log_loss": 0.0,
            "brier": 0.0,
            "ece": 0.0,
            "auc": 0.5,
            "calibration_status": "unavailable",
            "calibration_reason": "no_folds",
            "calibration_support": 0,
            "calibration_intercept": None,
            "calibration_slope": None,
        }

    result: dict[str, Any] = {
        key: float(statistics.fmean(float(getattr(fold, key)) for fold in folds))
        for key in ("log_loss", "brier", "ece", "auc")
    }
    result["calibration_support"] = sum(fold.calibration_support for fold in folds)
    unavailable = [fold for fold in folds if fold.calibration_status != "ok"]
    if unavailable:
        result.update(
            {
                "calibration_status": "unavailable",
                "calibration_reason": ";".join(
                    sorted(
                        {
                            fold.calibration_reason or "calibration_fit_unavailable"
                            for fold in unavailable
                        }
                    )
                ),
                "calibration_intercept": None,
                "calibration_slope": None,
            }
        )
    else:
        result.update(
            {
                "calibration_status": "ok",
                "calibration_reason": "",
                "calibration_intercept": float(
                    statistics.fmean(
                        [
                            float(fold.calibration_intercept)
                            for fold in folds
                            if fold.calibration_intercept is not None
                        ]
                    )
                ),
                "calibration_slope": float(
                    statistics.fmean(
                        [
                            float(fold.calibration_slope)
                            for fold in folds
                            if fold.calibration_slope is not None
                        ]
                    )
                ),
            }
        )
    return result


def _run_fold(
    adapter: CandidateAdapter,
    fold,
    rows_by_id: Mapping[str, EvalRow],
) -> tuple[FoldEvaluation, list[MatchPrediction]]:
    train_rows = _ordered_rows(fold.train_row_ids, rows_by_id)
    validation_rows = _ordered_rows(fold.validation_row_ids, rows_by_id)
    calibration_rows = _ordered_rows(fold.calibration_row_ids, rows_by_id)
    test_rows = _ordered_rows(fold.test_row_ids, rows_by_id)

    if not train_rows or not validation_rows or not calibration_rows or not test_rows:
        raise ValidationFailure(f"fold {fold.name} has an empty mandatory partition")

    assert_partition_disjoint(
        train_rows,
        validation_rows,
        calibration_rows,
        test_rows,
        partition_names=("train", "validation", "calibration", "test"),
    )

    assert_series_atomicity(fold.train_row_ids, rows_by_id)
    assert_series_atomicity(fold.validation_row_ids, rows_by_id)
    assert_series_atomicity(fold.calibration_row_ids, rows_by_id)
    assert_series_atomicity(fold.test_row_ids, rows_by_id)

    fit_state = adapter.fit(train_rows, split_name=fold.name)
    if fit_state.adapter_id != adapter.adapter_id:
        raise ValidationFailure(
            f"adapter {adapter.adapter_id}: fit output adapter id mismatch ({fit_state.adapter_id})"
        )

    raw_calibration = _run_predict(adapter, fit_state, calibration_rows, mode="terminal")
    raw_validation = _run_predict(adapter, fit_state, validation_rows, mode="terminal")
    raw_test = _run_predict(adapter, fit_state, test_rows, mode="terminal")
    assert_row_prediction_values(raw_calibration)
    assert_row_prediction_values(raw_validation)
    assert_row_prediction_values(raw_test)

    assert_no_label_leakage(adapter, fit_state, test_rows)

    runtime_test = _run_predict(adapter, fit_state, test_rows, mode="terminal")
    assert_row_prediction_values(runtime_test)
    assert_python_runtime_probabilities(adapter, fit_state, test_rows)
    if runtime_test != raw_test:
        # Keep parity as strict; _run_predict already checked row IDs.
        for runtime_pred, test_pred in zip(runtime_test, raw_test):
            if abs(runtime_pred.final_probability() - test_pred.final_probability()) > 1e-12:
                raise ValidationFailure(
                    f"adapter {adapter.adapter_id}: python and runtime probabilities differ"
                )

    calibrated_test = _run_calibration(
        adapter,
        fit_state,
        calibration_rows,
        raw_calibration,
        raw_test,
    )
    calibrated_validation = _run_calibration(
        adapter,
        fit_state,
        calibration_rows,
        raw_calibration,
        raw_validation,
    )

    if [pred.row_id for pred in calibrated_test] != [row.row_id for row in test_rows]:
        raise ValidationFailure("calibrated predictions are not aligned with test rows")
    assert_row_prediction_values(calibrated_test)
    assert_row_prediction_values(calibrated_validation)

    assert_invariant_ledger_reconciles(raw_calibration)
    assert_invariant_ledger_reconciles(raw_validation)
    assert_invariant_ledger_reconciles(raw_test)
    assert_invariant_ledger_reconciles(calibrated_test)
    assert_invariant_ledger_reconciles(calibrated_validation)

    raw_metrics = _build_fold_metric(
        [row.label for row in validation_rows],
        raw_validation,
        row_ids=[row.row_id for row in validation_rows],
    )
    calibrated_validation_metrics = _build_fold_metric(
        [row.label for row in validation_rows],
        calibrated_validation,
        row_ids=[row.row_id for row in validation_rows],
    )
    raw_test_metrics = _build_fold_metric(
        [row.label for row in test_rows],
        raw_test,
        row_ids=[row.row_id for row in test_rows],
    )
    calibrated_test_metrics = _build_fold_metric(
        [row.label for row in test_rows],
        calibrated_test,
        row_ids=[row.row_id for row in test_rows],
    )

    return (
        FoldEvaluation(
            fold_name=fold.name,
            train_rows=len(train_rows),
            validation_rows=len(validation_rows),
            calibration_rows=len(calibration_rows),
            test_rows=len(test_rows),
            raw_metrics=raw_test_metrics,
            calibrated_metrics=calibrated_test_metrics,
            validation_raw_metrics=raw_metrics,
            validation_calibrated_metrics=calibrated_validation_metrics,
        ),
        list(calibrated_test),
    )


@dataclass(frozen=True)
class FoldEvaluation:
    fold_name: str
    train_rows: int
    validation_rows: int
    calibration_rows: int
    test_rows: int
    raw_metrics: MetricSuite
    validation_raw_metrics: MetricSuite
    validation_calibrated_metrics: MetricSuite
    calibrated_metrics: MetricSuite


@dataclass(frozen=True)
class EvaluationReport:
    adapter_id: str
    adapter_version: str
    registry_hash: str
    folds: tuple[FoldEvaluation, ...]
    bootstrap: BootstrapResult
    sensitivity_bootstrap: Mapping[str, BootstrapResult]
    test_predictions: MatchPredictionMap
    holdout_reports: Mapping[str, Mapping[str, Any]]
    hard_gate_results: Mapping[str, bool]
    registry_bootstrap_seed: int
    registry_bootstrap_unit: str
    registry_bootstrap_replicates: int
    registry_bootstrap_size: int
    registry_bootstrap_sensitivity_units: tuple[str, ...]
    transfer_snapshot_hash: str
    international_metrics: Mapping[str, Mapping[str, float]]
    macro_region_log_loss: float
    aggregate_calibrated_metrics: Mapping[str, Any]
    aggregate_raw_metrics: Mapping[str, Any]


@dataclass(frozen=True)
class CandidateComparison:
    candidate_adapter_id: str
    baseline_adapter_id: str
    shared_rows: int
    candidate_metrics: Mapping[str, float]
    baseline_metrics: Mapping[str, float]
    metric_deltas: Mapping[str, float]
    bootstrap: BootstrapResult
    candidate_hard_gates: Mapping[str, bool]
    baseline_hard_gates: Mapping[str, bool]


@dataclass(frozen=True)
class TransferComparison:
    candidate_adapter_id: str
    transfer_adapter_id: str
    shared_rows: int
    candidate_metrics: Mapping[str, float]
    ontology_free_metrics: Mapping[str, float]
    transfer_ablation_metrics: Mapping[str, float]
    ontology_free_deltas: Mapping[str, float]
    transfer_ablation_deltas: Mapping[str, float]
    ontology_free_bootstrap: BootstrapResult
    transfer_ablation_bootstrap: BootstrapResult
    candidate_hard_gates: Mapping[str, bool]
    transfer_hard_gates: Mapping[str, bool]


def evaluate_candidate(
    adapter: CandidateAdapter,
    rows: Sequence[EvalRow],
    registry: EvaluationRegistry,
    *,
    request_terminal_probability_wording: bool = True,
    request_prefixes: Sequence[str] | None = None,
    sealed_rows_snapshot: Mapping[str, str] | None = None,
) -> EvaluationReport:
    request_prefixes = tuple(request_prefixes or ())
    rows_by_id = _rows_by_id(rows)

    hard_gates: dict[str, bool] = {
        "registry_frozen": False,
        "source_tree_match": False,
        "transform_identity": False,
        "runtime_transform_identity": False,
        "runtime_artifact_identity": False,
        "draft_order_diagnostics": False,
        "future_feature_joins": False,
        "final_roster": False,
        "row_cutoff": False,
        "exact_roster": False,
        "split_disjoint": False,
        "required_role_invariance_pairs": False,
        "required_side_swap_pairs": False,
        "seal_tamper": False,
        "terminal_probability_wording": False,
        "prefix_probability_wording": False,
        "label_leakage": False,
        "python_runtime_parity": False,
        "series_atomicity": False,
        "test_row_coverage": False,
        "validation_metrics_present": False,
    }
    if registry.b2_artifact_refs:
        hard_gates.update({name: False for name in B2_REQUIRED_HARD_GATES})

    fold_row_ids = [row_id for fold in registry.split_plan.folds for row_id in fold.all_ids]
    missing = sorted(set(fold_row_ids) - set(rows_by_id))
    if missing:
        raise ValidationFailure(f"registry references missing rows: {missing}")

    used_rows = [rows_by_id[row_id] for row_id in fold_row_ids]

    all_rows = list(rows_by_id.values())

    try:
        _assert_frozen_registry(registry)
        hard_gates["registry_frozen"] = True
        if registry.b2_artifact_refs:
            b2_report = build_b2_validation_report(registry)
            verify_b2_validation_report(b2_report, registry)
            hard_gates.update(dict(b2_report["hard_gates"]))
        if adapter.source_tree_sha256 != registry.source_tree_sha256:
            raise ValidationFailure(
                f"adapter {adapter.adapter_id}: source tree mismatch "
                f"({adapter.source_tree_sha256} != {registry.source_tree_sha256})"
            )
        hard_gates["source_tree_match"] = True

        assert_split_partitions_disjoint(registry, rows_by_id=rows_by_id)
        hard_gates["split_disjoint"] = True

        assert_rows_have_binary_labels(used_rows)

        assert_no_future_feature_joins(used_rows)
        hard_gates["future_feature_joins"] = True

        assert_no_final_roster_join_backwards(used_rows)
        hard_gates["final_roster"] = True

        for row in all_rows:
            assert_row_cutoff(row)
        hard_gates["row_cutoff"] = True

        for row in all_rows:
            assert_exact_roster(row)
        hard_gates["exact_roster"] = True

        assert_transform_identity(adapter)
        hard_gates["transform_identity"] = True

        assert_runtime_transform_identity(adapter)
        hard_gates["runtime_transform_identity"] = True

        assert_runtime_artifact_identity(adapter)
        hard_gates["runtime_artifact_identity"] = True

        assert_draft_order_diagnostics(registry)
        hard_gates["draft_order_diagnostics"] = True

        assert_terminal_probability_wording(adapter, request_terminal_probability_wording)
        hard_gates["terminal_probability_wording"] = True

        assert_prefix_probability_wording(adapter, request_prefixes)
        hard_gates["prefix_probability_wording"] = True

        if sealed_rows_snapshot is not None:
            assert_sealed_rows_immutable(registry, sealed_rows_snapshot, rows_by_id)
            hard_gates["seal_tamper"] = True

        fold_evaluations: list[FoldEvaluation] = []
        all_test_predictions: dict[str, MatchPrediction] = {}

        for fold in registry.split_plan.folds:
            fold_eval, fold_predictions = _run_fold(adapter, fold, rows_by_id)
            hard_gates["label_leakage"] = True
            hard_gates["python_runtime_parity"] = True
            hard_gates["series_atomicity"] = True
            hard_gates["test_row_coverage"] = True
            hard_gates["validation_metrics_present"] = True
            for pred in fold_predictions:
                if pred.row_id in all_test_predictions:
                    raise ValidationFailure(
                        f"candidate test predictions contain duplicate row id {pred.row_id} across folds"
                    )
                all_test_predictions[pred.row_id] = pred
            fold_evaluations.append(fold_eval)

        assert_required_role_invariance(test_predictions := _as_prediction_map(list(all_test_predictions.values())), registry.required_role_invariance_pairs)
        hard_gates["required_role_invariance_pairs"] = True

        assert_required_side_swap(test_predictions, registry.required_side_swap_pairs)
        hard_gates["required_side_swap_pairs"] = True

        # Apply optional invariance checks from row metadata when provided.
        role_pairs = _derive_pairs(list(rows_by_id.values()), key="role_invariance_pair")
        if role_pairs:
            test_row_ids = set(all_test_predictions.keys())
            filtered_role_pairs = {
                left: right
                for left, right in role_pairs.items()
                if left in test_row_ids and right in test_row_ids
            }
            if filtered_role_pairs:
                assert_role_invariance(_as_prediction_map(list(all_test_predictions.values())), filtered_role_pairs)

        side_pairs = _derive_pairs(list(rows_by_id.values()), key="side_swap_pair")
        if side_pairs:
            filtered_side_pairs = {
                left: right
                for left, right in side_pairs.items()
                if left in all_test_predictions and right in all_test_predictions
            }
            if filtered_side_pairs:
                assert_side_swap(_as_prediction_map(list(all_test_predictions.values())), filtered_side_pairs)

        test_row_ids = sorted(test_predictions)
        if not test_row_ids:
            raise ValidationFailure("candidate produced no test predictions")

        clusters = [rows_by_id[row_id].series_id for row_id in test_row_ids]
        resolved = [rows_by_id[row_id].series_resolved for row_id in test_row_ids]

        # Primary inference: paired series-cluster bootstrap.
        assert_bootstrap_not_map_level(clusters, test_row_ids, rows_by_id=rows_by_id)
        assert_unresolved_series_not_in_primary_bootstrap(test_predictions, rows_by_id)

        shared_labels = [rows_by_id[row_id].label for row_id in test_row_ids]
        shared_probs = [test_predictions[row_id].final_probability() for row_id in test_row_ids]
        candidate_score_deltas = _per_row_log_loss(shared_labels, shared_probs)

        bootstrap = series_cluster_bootstrap(
            deltas=candidate_score_deltas,
            cluster_ids=clusters,
            resolved_mask=resolved,
            row_ids=test_row_ids,
            n_boot=registry.bootstrap_cluster_replicates,
            random_seed=registry.bootstrap_seed,
        )

        sensitivity: dict[str, BootstrapResult] = {}
        for offset, unit in enumerate(registry.bootstrap_sensitivity_units):
            try:
                if unit not in _SENSITIVITY_UNIT_BUILDERS:
                    raise ValidationFailure(f"unsupported bootstrap sensitivity unit '{unit}'")

                sensitivity[unit] = _sensitivity_bootstrap_for_unit(
                    unit=unit,
                    deltas=candidate_score_deltas,
                    clusters=clusters,
                    resolved=resolved,
                    row_ids=test_row_ids,
                    rows_by_id=rows_by_id,
                    seed=registry.bootstrap_seed + offset,
                    n_boot=registry.bootstrap_cluster_replicates,
                )
            except ValidationFailure:
                # registry drives registered sensitivity units; leave unregistered
                # units absent rather than silently changing behavior.
                pass

        macro_log_loss, _ = _macro_region_metrics(test_predictions, rows_by_id)
        return EvaluationReport(
            adapter_id=adapter.adapter_id,
            adapter_version=adapter.adapter_version,
            registry_hash=registry.sha256(),
            folds=tuple(fold_evaluations),
            bootstrap=bootstrap,
            sensitivity_bootstrap=sensitivity,
            test_predictions=test_predictions,
            holdout_reports=_collect_holdout_reports(adapter, registry, rows_by_id),
            hard_gate_results=dict(hard_gates),
            registry_bootstrap_seed=registry.bootstrap_seed,
            registry_bootstrap_unit=registry.bootstrap_cluster_unit,
            registry_bootstrap_replicates=registry.bootstrap_cluster_replicates,
            registry_bootstrap_size=registry.bootstrap_cluster_size,
            registry_bootstrap_sensitivity_units=tuple(registry.bootstrap_sensitivity_units),
            transfer_snapshot_hash=registry.transfer_snapshot_hash,
            international_metrics=_named_international_metrics(test_predictions, rows_by_id),
            macro_region_log_loss=macro_log_loss,
            aggregate_calibrated_metrics=_aggregate_metric_suite(
                [fold.calibrated_metrics for fold in fold_evaluations]
            ),
            aggregate_raw_metrics=_aggregate_metric_suite(
                [fold.raw_metrics for fold in fold_evaluations]
            ),
        )

    except ValidationFailure as exc:
        # Preserve partial gate state for diagnosis.
        raise ValidationFailure(f"evaluation failed: {exc}")


def evaluate_candidate_from_snapshot(
    adapter: CandidateAdapter,
    snapshot: SnapshotAdapter,
    registry: EvaluationRegistry,
    *,
    request_terminal_probability_wording: bool = True,
    request_prefixes: Sequence[str] | None = None,
    sealed_rows_snapshot: Mapping[str, str] | None = None,
) -> EvaluationReport:
    """Evaluate a candidate adapter from an L1 snapshot interface."""
    if snapshot.source_tree_sha256 != registry.source_tree_sha256:
        raise ValidationFailure(
            f"snapshot {snapshot.snapshot_id} source tree mismatch for registry {registry.source_tree_sha256}"
        )

    return evaluate_candidate(
        adapter,
        snapshot.rows(),
        registry,
        request_terminal_probability_wording=request_terminal_probability_wording,
        request_prefixes=request_prefixes,
        sealed_rows_snapshot=sealed_rows_snapshot,
    )


def compare_candidate_to_baseline(
    candidate_report: EvaluationReport,
    baseline_report: EvaluationReport,
    rows: Sequence[EvalRow],
    *,
    row_ids: Sequence[str] | None = None,
) -> CandidateComparison:
    if candidate_report.registry_hash != baseline_report.registry_hash:
        raise ValidationFailure("candidate and baseline reports were produced from different registries")

    rows_by_id = _rows_by_id(rows)
    shared_rows = set(candidate_report.test_predictions) & set(baseline_report.test_predictions)
    if not shared_rows:
        raise ValidationFailure("no overlapping rows between candidate and baseline")

    if row_ids is not None:
        requested = set(row_ids)
        shared_rows &= requested
    if not shared_rows:
        raise ValidationFailure("no overlapping rows after filtering")

    shared_order = sorted(shared_rows)
    candidate_probs = [candidate_report.test_predictions[row_id].final_probability() for row_id in shared_order]
    baseline_probs = [baseline_report.test_predictions[row_id].final_probability() for row_id in shared_order]
    labels = [rows_by_id[row_id].label for row_id in shared_order]

    candidate_metrics = {
        "log_loss": log_loss(labels, candidate_probs),
        "brier": score_brier(labels, candidate_probs),
        "ece": expected_calibration_error(labels, candidate_probs),
        "auc": auc_score(labels, candidate_probs),
    }
    baseline_metrics = {
        "log_loss": log_loss(labels, baseline_probs),
        "brier": score_brier(labels, baseline_probs),
        "ece": expected_calibration_error(labels, baseline_probs),
        "auc": auc_score(labels, baseline_probs),
    }

    candidate_series = [rows_by_id[row_id].series_id for row_id in shared_order]
    resolved = [rows_by_id[row_id].series_resolved for row_id in shared_order]

    candidate_log_loss_deltas = [
        cl - bl for cl, bl in zip(_per_row_log_loss(labels, candidate_probs), _per_row_log_loss(labels, baseline_probs))
    ]

    assert_bootstrap_not_map_level(candidate_series, shared_order, rows_by_id=rows_by_id)
    assert_unresolved_series_not_in_primary_bootstrap(
        {row_id: candidate_report.test_predictions[row_id] for row_id in shared_order},
        rows_by_id,
    )

    bootstrap = series_cluster_bootstrap(
        deltas=candidate_log_loss_deltas,
        cluster_ids=candidate_series,
        resolved_mask=resolved,
        row_ids=shared_order,
        n_boot=candidate_report.registry_bootstrap_replicates,
        random_seed=candidate_report.registry_bootstrap_seed,
        cluster_unit=candidate_report.registry_bootstrap_unit,
    )

    metric_deltas = {
        name: candidate_metrics[name] - baseline_metrics[name]
        for name in candidate_metrics
    }

    return CandidateComparison(
        candidate_adapter_id=candidate_report.adapter_id,
        baseline_adapter_id=baseline_report.adapter_id,
        shared_rows=len(shared_order),
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        metric_deltas=metric_deltas,
        bootstrap=bootstrap,
        candidate_hard_gates=candidate_report.hard_gate_results,
        baseline_hard_gates=baseline_report.hard_gate_results,
    )


def compare_candidate_to_transfer_baselines(
    candidate_report: EvaluationReport,
    transfer_adapter: TransferComparisonAdapter,
    rows: Sequence[EvalRow],
    *,
    row_ids: Sequence[str] | None = None,
) -> TransferComparison:
    rows_by_id = _rows_by_id(rows)

    candidate_rows = set(candidate_report.test_predictions)
    if not candidate_rows:
        raise ValidationFailure("candidate report contains no test predictions")

    shared_rows = set(candidate_rows)
    if row_ids is not None:
        shared_rows &= set(row_ids)
    if not shared_rows:
        raise ValidationFailure("candidate report has no rows after optional filtering")

    if any(row_id not in rows_by_id for row_id in shared_rows):
        raise ValidationFailure("comparison row_id not in provided rows")

    if transfer_adapter.snapshot_sha256 != candidate_report.transfer_snapshot_hash:
        raise ValidationFailure("transfer snapshot hash mismatch")

    ontology_free_probabilities = _as_probability_map(
        transfer_adapter.predict_ontology_free([rows_by_id[row_id] for row_id in sorted(shared_rows)]),
        [rows_by_id[row_id] for row_id in sorted(shared_rows)],
        field_name=f"{transfer_adapter.adapter_id} ontology_free_probability",
    )
    transfer_ablation_probabilities = _as_probability_map(
        transfer_adapter.predict_transfer_ablation([rows_by_id[row_id] for row_id in sorted(shared_rows)]),
        [rows_by_id[row_id] for row_id in sorted(shared_rows)],
        field_name=f"{transfer_adapter.adapter_id} transfer_ablation_probability",
    )

    shared_order = sorted(shared_rows)
    labels = [rows_by_id[row_id].label for row_id in shared_order]
    candidate_probs = [candidate_report.test_predictions[row_id].final_probability() for row_id in shared_order]
    ontology_free_probs = [ontology_free_probabilities[row_id] for row_id in shared_order]
    transfer_ablation_probs = [transfer_ablation_probabilities[row_id] for row_id in shared_order]

    candidate_metrics = {
        "log_loss": log_loss(labels, candidate_probs),
        "brier": score_brier(labels, candidate_probs),
        "ece": expected_calibration_error(labels, candidate_probs),
    }
    ontology_free_metrics = {
        "log_loss": log_loss(labels, ontology_free_probs),
        "brier": score_brier(labels, ontology_free_probs),
        "ece": expected_calibration_error(labels, ontology_free_probs),
    }
    transfer_ablation_metrics = {
        "log_loss": log_loss(labels, transfer_ablation_probs),
        "brier": score_brier(labels, transfer_ablation_probs),
        "ece": expected_calibration_error(labels, transfer_ablation_probs),
    }

    candidate_series = [rows_by_id[row_id].series_id for row_id in shared_order]
    resolved = [rows_by_id[row_id].series_resolved for row_id in shared_order]

    candidate_ontology_log_loss = [
        cl - bl for cl, bl in zip(_per_row_log_loss(labels, candidate_probs), _per_row_log_loss(labels, ontology_free_probs))
    ]
    candidate_transfer_log_loss = [
        cl - tl for cl, tl in zip(_per_row_log_loss(labels, candidate_probs), _per_row_log_loss(labels, transfer_ablation_probs))
    ]

    assert_bootstrap_not_map_level(candidate_series, shared_order, rows_by_id=rows_by_id)
    assert_unresolved_series_not_in_primary_bootstrap(
        {row_id: candidate_report.test_predictions[row_id] for row_id in shared_order},
        rows_by_id,
    )

    ontology_bootstrap = series_cluster_bootstrap(
        deltas=candidate_ontology_log_loss,
        cluster_ids=candidate_series,
        resolved_mask=resolved,
        row_ids=shared_order,
        n_boot=candidate_report.registry_bootstrap_replicates,
        random_seed=candidate_report.registry_bootstrap_seed,
        cluster_unit=candidate_report.registry_bootstrap_unit,
    )
    transfer_bootstrap = series_cluster_bootstrap(
        deltas=candidate_transfer_log_loss,
        cluster_ids=candidate_series,
        resolved_mask=resolved,
        row_ids=shared_order,
        n_boot=candidate_report.registry_bootstrap_replicates,
        random_seed=candidate_report.registry_bootstrap_seed + 1,
        cluster_unit=candidate_report.registry_bootstrap_unit,
    )

    ontology_free_deltas = {
        name: candidate_metrics[name] - ontology_free_metrics[name]
        for name in candidate_metrics
    }
    transfer_ablation_deltas = {
        name: candidate_metrics[name] - transfer_ablation_metrics[name]
        for name in candidate_metrics
    }

    return TransferComparison(
        candidate_adapter_id=candidate_report.adapter_id,
        transfer_adapter_id=transfer_adapter.adapter_id,
        shared_rows=len(shared_order),
        candidate_metrics=candidate_metrics,
        ontology_free_metrics=ontology_free_metrics,
        transfer_ablation_metrics=transfer_ablation_metrics,
        ontology_free_deltas=ontology_free_deltas,
        transfer_ablation_deltas=transfer_ablation_deltas,
        ontology_free_bootstrap=ontology_bootstrap,
        transfer_ablation_bootstrap=transfer_bootstrap,
        candidate_hard_gates=candidate_report.hard_gate_results,
        transfer_hard_gates={"source_tree_match": True},
    )
