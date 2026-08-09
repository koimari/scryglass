"""Split registry construction for the L2 benchmark contract."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .types import (
    ArtifactRef,
    CONTRACT_TREE_SHA256,
    EvaluationRegistry,
    SealedHoldoutPartition,
    SplitPartition,
    SplitPlan,
    EvalRow,
    canonical_sha256,
    canonical_timestamp,
    read_json,
    write_json,
)


def _canonical_holdout_protocol_name(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized in {"latest_block", "temporal"}:
        return "temporal"
    if normalized in {"leave_one_out", "leave_one_tier1_league", "league_leave_one_out", "league_leave_one", "league_leave_out"}:
        return "league_leave_one_out"
    if normalized in {"international", "international_event"}:
        return "international_event"
    if normalized in {"new_roster", "roster_change"}:
        return "roster_change"
    if normalized in {"sparse_or_zero_play", "sparse_new_champion"}:
        return "sparse_new_champion"
    if normalized in {"masked_residual", "masked_champion_residual"}:
        return "masked_champion_residual"
    if normalized in {"archetype_transfer_residual", "archetype_transfer"}:
        return "archetype_transfer"
    if normalized in {"future_patch", "future_patch_holdout"}:
        return "future_patch"
    return normalized


def _normalize_holdout_metadata(
    metadata: Mapping[str, Any],
    *,
    fallback_protocol: str | None = None,
) -> dict[str, Any]:
    normalized = dict(metadata)
    protocol = _canonical_holdout_protocol_name(normalized.get("protocol"))
    if protocol is None:
        protocol = _canonical_holdout_protocol_name(fallback_protocol) or ""
    normalized["protocol"] = protocol
    return normalized



def _flag_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _parse_patch_to_int(patch_id: str) -> int:
    parts = patch_id.split(".")
    if len(parts) != 2:
        raise ValueError(f"patch_id must use major.minor format: {patch_id}")
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError as exc:
        raise ValueError(f"patch_id must use integers: {patch_id}") from exc
    return major * 10000 + minor


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.55
    validation_ratio: float = 0.20
    calibration_ratio: float = 0.15
    test_ratio: float = 0.10

    development_folds: int = 2
    temporal_holdout_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.development_folds < 1:
            raise ValueError("development_folds must be >= 1")
        for name, ratio in (
            ("train_ratio", self.train_ratio),
            ("validation_ratio", self.validation_ratio),
            ("calibration_ratio", self.calibration_ratio),
            ("test_ratio", self.test_ratio),
            ("temporal_holdout_ratio", self.temporal_holdout_ratio),
        ):
            if not 0 < ratio < 1:
                raise ValueError(f"{name} must be between 0 and 1: {ratio}")
        if abs((self.train_ratio + self.validation_ratio + self.calibration_ratio + self.test_ratio) - 1.0) > 1e-12:
            raise ValueError("train/validation/calibration/test ratios must sum to 1.0")


def _series_blocks(rows: Sequence[EvalRow], *, include_unresolved: bool) -> list[tuple[str, tuple[EvalRow, ...]]]:
    by_series: dict[str, list[EvalRow]] = {}
    for row in sorted(rows, key=lambda r: (r.event_start, r.row_id)):
        by_series.setdefault(row.series_id, []).append(row)

    blocks: list[tuple[str, tuple[EvalRow, ...]]] = []
    for series_id, items in by_series.items():
        if not items:
            continue
        if len({item.series_resolved for item in items}) > 1:
            raise ValueError(f"inconsistent series_resolved values for series {series_id}")
        if not include_unresolved and not items[0].series_resolved:
            continue
        blocks.append((series_id, tuple(items)))

    blocks.sort(key=lambda pair: (pair[1][0].event_start, pair[0]))
    return blocks


def _blocks_to_row_ids(blocks: list[tuple[str, tuple[EvalRow, ...]]]) -> tuple[str, ...]:
    ids: list[str] = []
    for _, rows in blocks:
        ids.extend([row.row_id for row in rows])
    return tuple(ids)


def _safe_block_partition_counts(total_blocks: int, config: SplitConfig) -> tuple[int, int, int]:
    if total_blocks <= 0:
        raise ValueError("total_blocks must be > 0")

    validation_blocks = max(1, int(total_blocks * config.validation_ratio))
    calibration_blocks = max(1, int(total_blocks * config.calibration_ratio))
    test_blocks = max(1, int(total_blocks * config.test_ratio))

    initial_train = total_blocks - (
        validation_blocks
        + calibration_blocks
        + (config.development_folds * test_blocks)
    )
    if initial_train < 1:
        raise ValueError(
            f"cannot materialize {config.development_folds} rolling folds with configured ratios"
        )

    return validation_blocks, calibration_blocks, test_blocks


def _build_rolling_windows(total_blocks: int, config: SplitConfig) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]]:
    if total_blocks <= 0:
        return []

    validation_blocks, calibration_blocks, test_blocks = _safe_block_partition_counts(total_blocks, config)
    initial_train = total_blocks - (
        validation_blocks
        + calibration_blocks
        + (config.development_folds * test_blocks)
    )
    if initial_train < 1:
        raise ValueError("development blocks are too short for requested fold count")

    windows = []
    for fold_index in range(config.development_folds):
        train_end = initial_train + (fold_index * test_blocks)
        validation_end = train_end + validation_blocks
        calibration_end = validation_end + calibration_blocks
        test_start = calibration_end
        test_end = test_start + test_blocks

        if test_end > total_blocks:
            raise ValueError("rolling fold window exceeds resolved block range")
        if calibration_end != test_start:
            raise ValueError("invalid rolling split configuration")

        windows.append(
            (
                (0, train_end),
                (train_end, validation_end),
                (validation_end, calibration_end),
                (test_start, test_end),
            )
        )

    if len(windows) != config.development_folds:
        raise ValueError("rolling split did not materialize exactly the declared fold count")
    if windows[-1][-1][1] != total_blocks:
        raise ValueError("final rolling test window does not end at the development boundary")
    return windows


def _first_by_roster_change(rows: Sequence[EvalRow]) -> tuple[str, ...]:
    by_roster: dict[str, EvalRow] = {}
    for row in sorted(rows, key=lambda row: row.event_start):
        if not row.is_roster_change:
            continue
        existing = by_roster.get(row.roster_id)
        if existing is None or row.event_start < existing.event_start:
            by_roster[row.roster_id] = row
    return tuple(row.row_id for row in sorted(by_roster.values(), key=lambda row: row.event_start))


def split_plan_payload_for_hash(split_plan: SplitPlan, split_plan_id: str) -> dict[str, Any]:
    """Canonical split-plan payload used for split integrity hashing."""
    return {
        "split_plan_id": split_plan_id,
        "folds": [
            {
                "name": fold.name,
                "train_row_ids": list(fold.train_row_ids),
                "validation_row_ids": list(fold.validation_row_ids),
                "calibration_row_ids": list(fold.calibration_row_ids),
                "test_row_ids": list(fold.test_row_ids),
            }
            for fold in split_plan.folds
        ],
        "sealed_holdouts": [
            {
                "name": holdout.name,
                "row_ids": list(holdout.row_ids),
                "metadata": dict(holdout.metadata),
            }
            for holdout in split_plan.sealed_holdouts
        ],
    }


def split_plan_sha256(split_plan: SplitPlan, split_plan_id: str) -> str:
    """Canonical split-plan SHA used by frozen-registry validation."""
    return canonical_sha256(split_plan_payload_for_hash(split_plan, split_plan_id))


def build_rolling_origin_plan(
    rows: Sequence[EvalRow],
    *,
    config: SplitConfig = SplitConfig(),
) -> SplitPlan:
    if not rows:
        raise ValueError("no rows supplied to build plan")

    rows_by_id = {row.row_id: row for row in rows}
    if len(rows_by_id) != len(rows):
        raise ValueError("duplicate row IDs are not allowed")

    resolved_blocks = _series_blocks(rows, include_unresolved=False)
    if not resolved_blocks:
        raise ValueError("no resolved series available for development folds")

    # Fail closed on series-level overlaps that would cross split frontiers.
    last_end = None
    for series_id, series_rows in resolved_blocks:
        start = series_rows[0].event_start
        end = series_rows[-1].event_start
        if last_end is not None and start <= last_end:
            raise ValueError(
                f"resolved series '{series_id}' overlaps prior series frontier "
                f"({start.isoformat()} <= {last_end.isoformat()})"
            )
        last_end = end

    # Temporal sealed holdout uses a tail block count by whole-series units.
    temporal_blocks = max(1, int(round(len(resolved_blocks) * config.temporal_holdout_ratio)))
    temporal_blocks = min(temporal_blocks, max(1, len(resolved_blocks) - 1))

    temporal_series_blocks = resolved_blocks[-temporal_blocks:]
    dev_blocks = resolved_blocks[: len(resolved_blocks) - temporal_blocks]
    if not dev_blocks:
        raise ValueError("not enough resolved series after removing temporal holdout")

    # Build rolling folds over resolved development series.
    windows = _build_rolling_windows(len(dev_blocks), config=config)
    all_folds: list[SplitPartition] = []

    for fold_index, (train_slice, val_slice, cal_slice, test_slice) in enumerate(windows):
        fold_train = dev_blocks[train_slice[0] : train_slice[1]]
        fold_validation = dev_blocks[val_slice[0] : val_slice[1]]
        fold_calibration = dev_blocks[cal_slice[0] : cal_slice[1]]
        fold_test = dev_blocks[test_slice[0] : test_slice[1]]

        train_ids = _blocks_to_row_ids(list(fold_train))
        validation_ids = _blocks_to_row_ids(list(fold_validation))
        calibration_ids = _blocks_to_row_ids(list(fold_calibration))
        test_ids = _blocks_to_row_ids(list(fold_test))

        if not (train_ids and validation_ids and calibration_ids and test_ids):
            raise ValueError(f"empty partition in rolling fold {fold_index}")

        all_folds.append(
            SplitPartition(
                name=f"fold_{fold_index}",
                train_row_ids=train_ids,
                validation_row_ids=validation_ids,
                calibration_row_ids=calibration_ids,
                test_row_ids=test_ids,
            )
        )

    # Ensure held-out temporal rows are excluded from all development partitions by construction.
    temporal_row_ids = _blocks_to_row_ids(list(temporal_series_blocks))

    # Future-patch, out-of-time holdout rows are based on max development patch.
    dev_row_ids = _blocks_to_row_ids(dev_blocks)
    dev_patch_max = max(_parse_patch_to_int(rows_by_id[row_id].patch_id) for row_id in dev_row_ids)

    future_patch_ids = tuple(
        row.row_id
        for row in rows
        if row.series_resolved and _parse_patch_to_int(row.patch_id) > dev_patch_max
    )

    # Leave-one-league-out holdouts should only target tier-1 leagues.
    tier1_leagues = sorted({row.league_id for row in rows if row.league_tier == "tier1" and row.league_id})
    league_leaveouts = [
        SealedHoldoutPartition(
            name=f"league_out_{league}",
            row_ids=tuple(
                row.row_id
                for row in rows
                if row.series_resolved and row.league_tier == "tier1" and row.league_id == league
            ),
            metadata=_normalize_holdout_metadata(
                {"mode": "league_leave_out", "league_id": league},
                fallback_protocol="league_leave_one_out",
            ),
        )
        for league in tier1_leagues
    ]

    international_events = sorted({
        row.international_event_id
        for row in rows
        if row.is_international_event and row.international_event_id
    })
    international_holdouts = [
        SealedHoldoutPartition(
            name=f"international_{event_id}",
            row_ids=tuple(
                row.row_id
                for row in rows
                if row.series_resolved and row.international_event_id == event_id
            ),
            metadata=_normalize_holdout_metadata(
                {"mode": "international", "event_id": event_id},
                fallback_protocol="international_event",
            ),
        )
        for event_id in international_events
    ]

    sparse_ids = tuple(
        row.row_id for row in rows if row.series_resolved and (row.is_sparse_champion or len(row.champion_ids) == 0)
    )
    roster_change_ids = _first_by_roster_change(
        [row for row in rows if row.series_resolved]
    )

    masked_residual_ids = tuple(
        row.row_id
        for row in rows
        if row.series_resolved and _flag_truthy(row.metadata.get("masked_champion_residual"))
    )
    archetype_transfer_ids = tuple(
        row.row_id
        for row in rows
        if row.series_resolved and (
            _flag_truthy(row.metadata.get("true_new_champion"))
            or _flag_truthy(row.metadata.get("archetype_transfer"))
        )
    )

    holdouts = [
        SealedHoldoutPartition(
            name="temporal",
            row_ids=temporal_row_ids,
            metadata=_normalize_holdout_metadata(
                {"mode": "latest_block"},
                fallback_protocol="temporal",
            ),
        ),
        SealedHoldoutPartition(
            name="future_patch",
            row_ids=future_patch_ids,
            metadata=_normalize_holdout_metadata(
                {"mode": "future_patch", "no_exact_patch_fallback": True},
                fallback_protocol="future_patch",
            ),
        ),
        SealedHoldoutPartition(
            name="roster_change",
            row_ids=roster_change_ids,
            metadata=_normalize_holdout_metadata(
                {"mode": "new_roster"},
                fallback_protocol="roster_change",
            ),
        ),
        SealedHoldoutPartition(
            name="sparse_new_champion",
            row_ids=sparse_ids,
            metadata=_normalize_holdout_metadata(
                {"mode": "sparse_or_zero_play"},
                fallback_protocol="sparse_new_champion",
            ),
        ),
        SealedHoldoutPartition(
            name="masked_champion_residual",
            row_ids=masked_residual_ids,
            metadata=_normalize_holdout_metadata(
                {"mode": "masked_residual"},
                fallback_protocol="masked_champion_residual",
            ),
        ),
        SealedHoldoutPartition(
            name="archetype_transfer_true_new_or_zero_play",
            row_ids=archetype_transfer_ids,
            metadata=_normalize_holdout_metadata(
                {"mode": "archetype_transfer_residual"},
                fallback_protocol="archetype_transfer",
            ),
        ),
    ]
    holdouts.extend(league_leaveouts)
    holdouts.extend(international_holdouts)

    # canonicalize holdout entries
    deduped = {}
    for holdout in holdouts:
        if holdout.name in deduped:
            raise ValueError(f"duplicate holdout name: {holdout.name}")
        deduped[holdout.name] = SealedHoldoutPartition(
            name=holdout.name,
            row_ids=tuple(
                row_id
                for row_id in dict.fromkeys(holdout.row_ids)
                if row_id in rows_by_id
            ),
            metadata=dict(holdout.metadata),
        )

    # Every resolved row outside an exclusive temporal/future seal or an
    # explicitly content-addressed embargo must be represented in development.
    development_rows: set[str] = set()
    for fold in all_folds:
        development_rows.update(fold.all_ids)

    exclusive_sealed_rows: set[str] = set()
    for holdout in deduped.values():
        protocol = _canonical_holdout_protocol_name(holdout.metadata.get("protocol"))
        embargo_sha256 = str(holdout.metadata.get("embargo_sha256") or "")
        if embargo_sha256:
            if len(embargo_sha256) != 64:
                raise ValueError(f"holdout {holdout.name} has an invalid embargo_sha256")
            try:
                int(embargo_sha256, 16)
            except ValueError as exc:
                raise ValueError(f"holdout {holdout.name} has a non-hex embargo_sha256") from exc
        if protocol in {"temporal", "future_patch"} or embargo_sha256:
            exclusive_sealed_rows.update(holdout.row_ids)

    resolved_non_sealed_rows = {
        row_id
        for row_id in rows_by_id
        if rows_by_id[row_id].series_resolved and row_id not in exclusive_sealed_rows
    }
    unresolved_rows_in_dev = [row_id for row_id in sorted(development_rows) if not rows_by_id[row_id].series_resolved]
    if unresolved_rows_in_dev:
        raise ValueError(
            f"unresolved rows leaked into development: {unresolved_rows_in_dev}"
        )

    missing_assignment = sorted(resolved_non_sealed_rows - development_rows)
    if missing_assignment:
        raise ValueError(
            f"resolved rows outside sealed holdouts are not assigned to development: {missing_assignment}"
        )

    return SplitPlan(folds=tuple(all_folds), sealed_holdouts=tuple(deduped.values()))


def build_evaluation_registry(
    rows: Sequence[EvalRow],
    *,
    contract_tree_sha256: str,
    source_snapshot_id: str,
    training_snapshot_id: str,
    source_tree_sha256: str,
    split_plan_id: str,
    bootstrap_seed: int,
    source_snapshot_sha256: str = "",
    training_snapshot_sha256: str = "",
    split_config: SplitConfig = SplitConfig(),
    split_plan: SplitPlan | None = None,
) -> EvaluationRegistry:
    if contract_tree_sha256 != CONTRACT_TREE_SHA256:
        raise ValueError("contract_tree_sha256 does not match expected S0 contract hash")

    split_plan = split_plan or build_rolling_origin_plan(rows, config=split_config)

    return EvaluationRegistry(
        contract_tree_sha256=contract_tree_sha256,
        split_plan_id=split_plan_id,
        split_plan_sha256=split_plan_sha256(split_plan, split_plan_id),
        source_snapshot_id=source_snapshot_id,
        source_snapshot_sha256=source_snapshot_sha256 or source_tree_sha256,
        training_snapshot_id=training_snapshot_id,
        training_snapshot_sha256=training_snapshot_sha256 or source_tree_sha256,
        source_tree_sha256=source_tree_sha256,
        created_at=canonical_timestamp(datetime.now(timezone.utc)),
        bootstrap_seed=bootstrap_seed,
        split_plan=split_plan,
        split_config={
            "train_ratio": split_config.train_ratio,
            "validation_ratio": split_config.validation_ratio,
            "calibration_ratio": split_config.calibration_ratio,
            "test_ratio": split_config.test_ratio,
            "development_folds": split_config.development_folds,
            "temporal_holdout_ratio": split_config.temporal_holdout_ratio,
        },
    )


def registry_to_disk(registry: EvaluationRegistry, path: str | Path) -> str:
    return write_json(Path(path), registry.to_payload())


def load_evaluation_registry(path: str | Path) -> EvaluationRegistry:
    payload = read_json(Path(path))
    folds = tuple(
        SplitPartition(
            name=fold["name"],
            train_row_ids=tuple(fold["train_row_ids"]),
            validation_row_ids=tuple(fold["validation_row_ids"]),
            calibration_row_ids=tuple(fold["calibration_row_ids"]),
            test_row_ids=tuple(fold["test_row_ids"]),
        )
        for fold in payload["split_plan"]["folds"]
    )
    holdouts = tuple(
        SealedHoldoutPartition(
            name=holdout["name"],
            row_ids=tuple(holdout["row_ids"]),
            metadata=dict(holdout.get("metadata", {})),
        )
        for holdout in payload["split_plan"]["sealed_holdouts"]
    )
    split_plan = SplitPlan(folds=folds, sealed_holdouts=holdouts)

    return EvaluationRegistry(
        contract_tree_sha256=payload["contract_tree_sha256"],
        split_plan_id=payload["split_plan_id"],
        split_plan_sha256=payload["split_plan_sha256"],
        source_snapshot_id=payload["source_snapshot_id"],
        source_snapshot_sha256=str(payload.get("source_snapshot_sha256", "")),
        training_snapshot_id=payload["training_snapshot_id"],
        training_snapshot_sha256=str(payload.get("training_snapshot_sha256", "")),
        source_tree_sha256=payload["source_tree_sha256"],
        created_at=payload["created_at"],
        bootstrap_seed=int(payload["bootstrap_seed"]),
        split_plan=split_plan,
        source_crosswalk_sha256=dict(payload.get("source_crosswalk_sha256", {})),
        entity_crosswalk_sha256=dict(payload.get("entity_crosswalk_sha256", {})),
        league_crosswalk_sha256=dict(payload.get("league_crosswalk_sha256", {})),
        transfer_snapshot_hash=str(payload.get("transfer_snapshot_hash", "")),
        metrics=tuple(payload.get("metrics", ("log_loss", "brier", "ece"))),
        estimands=tuple(payload.get("estimands", ("terminal_draft_score",))),
        baseline_ids=dict(payload.get("baseline_ids", {})),
        baseline_artifact_hashes=dict(payload.get("baseline_artifact_hashes", {})),
        candidate_artifact_hashes=dict(payload.get("candidate_artifact_hashes", {})),
        served_transform_identities=dict(payload.get("served_transform_identities", {})),
        subgroup_specs=dict(payload.get("subgroup_specs", {})),
        missingness_specs=dict(payload.get("missingness_specs", {})),
        bootstrap_cluster_unit=str(payload.get("bootstrap_cluster_unit", "series")),
        bootstrap_cluster_replicates=int(payload.get("bootstrap_cluster_replicates", 2000)),
        bootstrap_cluster_size=int(payload.get("bootstrap_cluster_size", 2000)),
        bootstrap_sensitivity_units=tuple(payload.get("bootstrap_sensitivity_units", ("region",))),
        noninferiority_rules=dict(payload.get("noninferiority_rules", {})),
        noninferiority_higher_is_better=dict(payload.get("noninferiority_higher_is_better", {})),
        noninferiority_provenance=str(payload.get("noninferiority_provenance", "")),
        coverage_procedure=dict(payload.get("coverage_procedure", {})),
        parity_tolerance=float(payload.get("parity_tolerance", 1e-9)),
        invalidation_reasons=tuple(payload.get("invalidation_reasons", ())),
        draft_order_analysis=dict(payload.get("draft_order_analysis", {})),
        required_role_invariance_pairs=tuple(
            tuple(pair)
            for pair in payload.get("required_role_invariance_pairs", ())
        ),
        required_side_swap_pairs=tuple(
            tuple(pair)
            for pair in payload.get("required_side_swap_pairs", ())
        ),
        split_config=dict(
            payload.get(
                "split_config",
                {
                    "train_ratio": 0.55,
                    "validation_ratio": 0.20,
                    "calibration_ratio": 0.15,
                    "test_ratio": 0.10,
                    "development_folds": 2,
                    "temporal_holdout_ratio": 0.10,
                },
            )
        ),
        is_synthetic_registry=bool(payload.get("is_synthetic_registry", False)),
        b2_artifact_refs=tuple(
            ArtifactRef(
                artifact_id=str(ref["artifact_id"]),
                locator=str(ref["locator"]),
                raw_sha256=str(ref["raw_sha256"]),
                canonical_payload_sha256=str(ref["canonical_payload_sha256"]),
            )
            for ref in payload.get("b2_artifact_refs", ())
        ),
        b2_validation_report_sha256=str(
            payload.get("b2_validation_report_sha256", "")
        ),
    )


def assert_registry_roundtrip(registry: EvaluationRegistry, path: str | Path) -> bool:
    roundtrip_hash = registry_to_disk(registry, path)
    loaded = load_evaluation_registry(path)
    return loaded.sha256() == registry.sha256() == roundtrip_hash
