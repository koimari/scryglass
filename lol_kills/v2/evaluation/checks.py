"""Hard gates and audit sentinels for L2 model evaluation."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
import math
from typing import Any

from .types import EvaluationRegistry, EvalRow, MatchPrediction, MatchPredictionMap


_DRAFT_ORDER_KEY = ("protocol", "order", "side")


def _canonical_protocol_name(value: object) -> str:
    if value is None:
        return ""
    normalized = str(value).strip().lower().replace(" ", "_")

    if normalized.startswith("league_out_"):
        return "league_leave_one_out"
    if normalized.startswith("international_"):
        return "international_event"
    if normalized.startswith("temporal"):
        return "temporal"
    if normalized.startswith("future_"):
        return "future_patch"

    aliases = {
        "latest_block": "temporal",
        "latestblock": "temporal",
        "latest-block": "temporal",
        "temporal": "temporal",
        "future_patch": "future_patch",
        "future": "future_patch",
        "future_patch_holdout": "future_patch",
        "league_leave_one_out": "league_leave_one_out",
        "league_leave_out": "league_leave_one_out",
        "leave_one_tier1_league": "league_leave_one_out",
        "league": "league_leave_one_out",
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
        "archetype_transfer_true_new_or_zero_play": "archetype_transfer",
    }
    normalized = aliases.get(normalized, normalized)
    normalized = aliases.get(normalized.replace("-", "_"), normalized.replace("-", "_"))
    return normalized


def _canonical_patch_id(value: object) -> int:
    patch = str(value).strip()
    major_minor = patch.split(".")
    if len(major_minor) != 2:
        raise ValidationFailure(f"invalid patch id: {value!r}")
    try:
        major = int(major_minor[0])
        minor = int(major_minor[1])
    except ValueError as exc:
        raise ValidationFailure(f"invalid patch id: {value!r}") from exc
    return major * 100 + minor


def _ordered_set(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def _get_float(value: object, *, field_name: str) -> float:
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        raise ValidationFailure(f"draft-order analysis field is not numeric: {field_name}={value!r}")
    if not math.isfinite(value_f):
        raise ValidationFailure(f"draft-order analysis field is not finite: {field_name}={value_f}")
    return value_f


def _require_non_empty_sha256(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValidationFailure(f"{field_name} is required")
    if len(value) != 64:
        raise ValidationFailure(f"{field_name} must be a 64-character hex string")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValidationFailure(f"{field_name} is not hex: {value}") from exc
    return value


class ValidationFailure(ValueError):
    """Raised when a hard gate must fail closed."""


def _is_finite_scalar(value: object) -> bool:
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _partition_key(item: object) -> str:
    if hasattr(item, "row_id"):
        return getattr(item, "row_id")
    return str(item)


def assert_partition_disjoint(*partitions: Sequence[object], partition_names: Sequence[str] | None = None) -> None:
    names = list(partition_names or (str(idx) for idx in range(len(partitions))))
    if len(names) < len(partitions):
        names.extend([f"part-{idx}" for idx in range(len(names), len(partitions))])

    seen: dict[str, str] = {}
    for name, partition in zip(names, partitions):
        for row_id in partition:
            key = _partition_key(row_id)
            if key in seen:
                raise ValidationFailure(
                    f"row {key} appears in both {seen[key]} and {name}"
                )
            seen[key] = name


def assert_rows_have_binary_labels(rows: Sequence[EvalRow]) -> None:
    for row in rows:
        if row.label not in (0, 1):
            raise ValidationFailure(
                f"row {row.row_id}: non-binary label {row.label}; expected 0 or 1"
            )


def assert_no_future_feature_joins(rows: Iterable[EvalRow]) -> None:
    """Reject rows where any feature is unavailable at/after event start."""
    for row in rows:
        for feature_name, available_at in row.feature_available_at.items():
            if available_at >= row.event_start:
                raise ValidationFailure(
                    f"row {row.row_id}: feature '{feature_name}' unavailable at prediction time "
                    f"({available_at.isoformat()} >= event_start {row.event_start.isoformat()})"
                )


def assert_no_final_roster_join_backwards(rows: Iterable[EvalRow]) -> None:
    """Block joins from final rosters or future roster snapshots."""
    for row in rows:
        if row.roster_snapshot_stage != "final":
            continue
        if row.roster_snapshot_time is None:
            raise ValidationFailure(
                f"row {row.row_id}: final roster snapshot requested but roster_snapshot_time missing"
            )
        if row.roster_snapshot_time > row.as_of:
            raise ValidationFailure(
                f"row {row.row_id}: final roster snapshot ({row.roster_snapshot_time.isoformat()}) is after as_of "
                f"({row.as_of.isoformat()})"
            )


def assert_row_cutoff(row: EvalRow) -> None:
    """Block post-event joins."""
    if row.as_of > row.event_start:
        raise ValidationFailure(
            f"row {row.row_id}: as_of ({row.as_of.isoformat()}) is after event_start ({row.event_start.isoformat()})"
        )


def assert_series_atomicity(split_row_ids: Sequence[str], rows_by_id: Mapping[str, EvalRow]) -> None:
    """All rows in a fold partition must remain within complete series."""
    if len(split_row_ids) != len(set(split_row_ids)):
        raise ValidationFailure("split contains duplicate row IDs")

    by_series: dict[str, set[str]] = defaultdict(set)
    for row_id, row in rows_by_id.items():
        by_series[row.series_id].add(row_id)

    for row_id in split_row_ids:
        if row_id not in rows_by_id:
            raise ValidationFailure(f"split references unknown row {row_id}")
        series_id = rows_by_id[row_id].series_id
        row_series = by_series[series_id]
        if not row_series.issubset(set(split_row_ids)):
            raise ValidationFailure(
                f"series_atomicity violated: series {series_id} split across partitions"
            )


def assert_split_partitions_disjoint(
    registry: EvaluationRegistry,
    *,
    rows_by_id: Mapping[str, EvalRow] | None = None,
) -> None:
    """Partition checks for rolling folds plus limited holdout constraints.

    We enforce strict per-fold atomicity and chronological test windows but allow
    intentional holdout-overlap for non-temporal holdout protocols (which are
    evaluated by explicit refit logic).
    """

    allowed_protocols = {
        "temporal",
        "future_patch",
        "league_leave_one_out",
        "international_event",
        "roster_change",
        "sparse_new_champion",
        "masked_champion_residual",
        "archetype_transfer",
    }

    if not registry.split_plan.folds:
        raise ValidationFailure("registry has no development folds")
    declared_folds = registry.split_config.get("development_folds")
    if not isinstance(declared_folds, int) or declared_folds < 1:
        raise ValidationFailure("registry development_folds must be a positive integer")
    if len(registry.split_plan.folds) != declared_folds:
        raise ValidationFailure(
            "registry fold count does not match declared development_folds"
        )
    fold_names = [fold.name for fold in registry.split_plan.folds]
    if fold_names != [f"fold_{index}" for index in range(declared_folds)]:
        raise ValidationFailure("registry fold order or names do not match the declared order")

    # 1) Folds: within-fold disjointness and non-empty required partitions.
    for fold in registry.split_plan.folds:
        if not (fold.train_row_ids and fold.validation_row_ids and fold.calibration_row_ids and fold.test_row_ids):
            raise ValidationFailure(f"fold {fold.name} has an empty mandatory partition")

        assert_partition_disjoint(
            fold.train_row_ids,
            fold.validation_row_ids,
            fold.calibration_row_ids,
            fold.test_row_ids,
            partition_names=("train", "validation", "calibration", "test"),
        )
        if rows_by_id is not None:
            partitions = (
                ("train", fold.train_row_ids),
                ("validation", fold.validation_row_ids),
                ("calibration", fold.calibration_row_ids),
                ("test", fold.test_row_ids),
            )
            bounds: dict[str, tuple[float, float]] = {}
            for partition_name, row_ids in partitions:
                timestamps: list[float] = []
                for row_id in row_ids:
                    row = rows_by_id.get(row_id)
                    if row is None:
                        raise ValidationFailure(
                            f"fold {fold.name} references unknown {partition_name} row {row_id}"
                        )
                    timestamps.append(row.event_start.timestamp())
                bounds[partition_name] = (min(timestamps), max(timestamps))
            if not (
                bounds["train"][1] < bounds["validation"][0]
                and bounds["validation"][1] < bounds["calibration"][0]
                and bounds["calibration"][1] < bounds["test"][0]
            ):
                raise ValidationFailure(
                    f"fold {fold.name} partitions are not strictly chronological "
                    "(train < validation < calibration < test)"
                )

    # 2) Temporal constraints for development folds: test windows must be
    # disjoint and chronologically ordered. Roll-forward reuse in other
    # partitions is permitted.
    seen_test: set[str] = set()
    test_windows: list[tuple[int, int, str]] = []

    for fold in registry.split_plan.folds:
        if rows_by_id is not None:
            test_start = None
            test_end = None
            for row_id in fold.test_row_ids:
                row = rows_by_id.get(row_id)
                if row is None:
                    raise ValidationFailure(f"fold {fold.name} references unknown test row {row_id}")
                ts = row.event_start.timestamp()
                if test_start is None or ts < test_start:
                    test_start = ts
                if test_end is None or ts > test_end:
                    test_end = ts
            if test_start is None or test_end is None:
                raise ValidationFailure(f"fold {fold.name} has an empty test window")
            test_windows.append((test_start, test_end, fold.name))

        for row_id in fold.test_row_ids:
            if row_id in seen_test:
                raise ValidationFailure(
                    f"test row {row_id} appears in multiple fold test windows"
                )
            seen_test.add(row_id)

    if rows_by_id is not None and len(test_windows) > 1:
        for previous, current in zip(test_windows, test_windows[1:]):
            if previous[1] >= current[0]:
                raise ValidationFailure(
                    f"test windows are not temporally ordered: {previous[2]} -> {current[2]}"
                )

    # 2b) Keep split windows atomic per-series but avoid globally banning row
    # reuse in train/validation/calibration across folds; rolling-origin schemes
    # commonly do that on purpose.

    # 3) Each sealed holdout must be protocol-consistent with its row identities.
    all_dev_rows: set[str] = set()
    dev_test_rows: set[str] = set()
    dev_train_rows: set[str] = set()
    dev_validation_rows: set[str] = set()
    dev_calibration_rows: set[str] = set()
    for fold in registry.split_plan.folds:
        all_dev_rows.update(fold.train_row_ids)
        all_dev_rows.update(fold.validation_row_ids)
        all_dev_rows.update(fold.calibration_row_ids)
        all_dev_rows.update(fold.test_row_ids)
        dev_test_rows.update(fold.test_row_ids)
        dev_train_rows.update(fold.train_row_ids)
        dev_validation_rows.update(fold.validation_row_ids)
        dev_calibration_rows.update(fold.calibration_row_ids)

    dev_patch_max = None
    if rows_by_id is not None and all_dev_rows:
        dev_patches = [_canonical_patch_id(rows_by_id[row_id].patch_id) for row_id in all_dev_rows if row_id in rows_by_id]
        if dev_patches:
            dev_patch_max = max(dev_patches)

    exclusive_sealed_rows: set[str] = set()
    league_holdouts: dict[str, list[SealedHoldoutPartition]] = defaultdict(list)

    for holdout in registry.split_plan.sealed_holdouts:
        metadata = dict(holdout.metadata)
        protocol = _canonical_protocol_name(metadata.get("protocol") or metadata.get("mode") or metadata.get("name") or "")
        if len(holdout.row_ids) != len(set(holdout.row_ids)):
            raise ValidationFailure(f"holdout '{holdout.name}' contains duplicate row IDs")
        holdout_rows = set(holdout.row_ids)
        if not holdout_rows:
            raise ValidationFailure(f"holdout '{holdout.name}' has no rows")
        if protocol and protocol not in allowed_protocols:
            raise ValidationFailure(f"holdout '{holdout.name}' has unsupported protocol '{protocol}'")
        embargo_sha256 = str(metadata.get("embargo_sha256") or "")
        if embargo_sha256:
            _require_non_empty_sha256(
                embargo_sha256,
                field_name=f"holdout '{holdout.name}' embargo_sha256",
            )
            exclusive_sealed_rows.update(holdout_rows)
        if rows_by_id is not None:
            unknown_or_unresolved = sorted(
                row_id
                for row_id in holdout_rows
                if row_id not in rows_by_id or not rows_by_id[row_id].series_resolved
            )
            if unknown_or_unresolved:
                raise ValidationFailure(
                    f"holdout '{holdout.name}' contains unknown or unresolved rows: "
                    f"{unknown_or_unresolved}"
                )
        if protocol in {"temporal", "future_patch"}:
            exclusive_sealed_rows.update(holdout_rows)
            dev_rows = all_dev_rows
            overlap = holdout_rows & dev_rows
            if overlap:
                raise ValidationFailure(
                    f"{protocol} holdout '{holdout.name}' overlaps development rows: {sorted(overlap)}"
                )

        # Non-temporal holdouts can intentionally share rows with development
        # pools. They are still required to be protocol-explicit and to be
        # evaluated via a dedicated refit path.
        if not protocol:
            raise ValidationFailure(
                f"holdout '{holdout.name}' is missing protocol metadata"
            )

        if protocol in {"future_patch", "league_leave_one_out", "international_event", "roster_change", "sparse_new_champion", "masked_champion_residual", "archetype_transfer"}:
            if not any(1 for _ in holdout_rows):
                raise ValidationFailure(f"holdout '{holdout.name}' is empty")

        if protocol == "temporal":
            if not metadata:
                raise ValidationFailure(f"temporal holdout '{holdout.name}' is missing protocol metadata")

        if protocol == "future_patch":
            if not metadata:
                raise ValidationFailure(f"future patch holdout '{holdout.name}' is missing protocol metadata")
            if dev_patch_max is not None:
                invalid = [
                    row_id
                    for row_id in holdout_rows
                    if rows_by_id is not None
                    and row_id in rows_by_id
                    and _canonical_patch_id(rows_by_id[row_id].patch_id) <= dev_patch_max
                ]
                if invalid:
                    raise ValidationFailure(
                        f"future patch holdout '{holdout.name}' contains non-future rows: {sorted(invalid)}"
                    )

        if protocol == "league_leave_one_out":
            league_id = str(metadata.get("league_id") or "").strip()
            if not league_id:
                raise ValidationFailure(
                    f"league holdout '{holdout.name}' is missing league_id metadata"
                )
            league_holdouts[league_id].append(holdout)
            if rows_by_id is not None:
                invalid = sorted(
                    row_id
                    for row_id in holdout_rows
                    if row_id not in rows_by_id
                    or not rows_by_id[row_id].series_resolved
                    or rows_by_id[row_id].league_tier != "tier1"
                    or rows_by_id[row_id].league_id != league_id
                )
                if invalid:
                    raise ValidationFailure(
                        f"league holdout '{holdout.name}' contains rows outside scope '{league_id}': {sorted(invalid)}"
                    )

        elif protocol == "international_event":
            event_id = str(metadata.get("event_id") or "").strip()
            if not event_id:
                raise ValidationFailure(
                    f"international holdout '{holdout.name}' is missing event_id metadata"
                )
            if rows_by_id is not None:
                invalid = [
                    row_id
                    for row_id in holdout_rows
                    if row_id not in rows_by_id
                    or rows_by_id[row_id].international_event_id != event_id
                ]
                if invalid:
                    raise ValidationFailure(
                        f"international holdout '{holdout.name}' contains rows outside event '{event_id}': {sorted(invalid)}"
                    )

        elif protocol == "roster_change":
            if rows_by_id is not None:
                invalid = [
                    row_id
                    for row_id in holdout_rows
                    if row_id not in rows_by_id or not rows_by_id[row_id].is_roster_change
                ]
                if invalid:
                    raise ValidationFailure(
                        f"roster-change holdout '{holdout.name}' contains non-roster-change rows: {sorted(invalid)}"
                    )

        elif protocol == "sparse_new_champion":
            if rows_by_id is not None:
                invalid = [
                    row_id
                    for row_id in holdout_rows
                    if row_id not in rows_by_id
                    or (
                        not rows_by_id[row_id].is_sparse_champion
                        and len(rows_by_id[row_id].champion_ids) > 0
                    )
                ]
                if invalid:
                    raise ValidationFailure(
                        f"sparse-champion holdout '{holdout.name}' contains ineligible rows: {sorted(invalid)}"
                    )

        elif protocol == "masked_champion_residual":
            if rows_by_id is not None:
                invalid = [
                    row_id
                    for row_id in holdout_rows
                    if row_id not in rows_by_id
                    or not bool(rows_by_id[row_id].metadata.get("masked_champion_residual", False))
                ]
                if invalid:
                    raise ValidationFailure(
                        f"masked-champion-residual holdout '{holdout.name}' contains ineligible rows: {sorted(invalid)}"
                    )

        elif protocol == "archetype_transfer":
            if rows_by_id is not None:
                invalid = [
                    row_id
                    for row_id in holdout_rows
                    if row_id not in rows_by_id
                    or not (
                        bool(rows_by_id[row_id].metadata.get("true_new_champion", False))
                        or bool(rows_by_id[row_id].metadata.get("archetype_transfer", False))
                    )
                ]
                if invalid:
                    raise ValidationFailure(
                        f"archetype-transfer holdout '{holdout.name}' contains ineligible rows: {sorted(invalid)}"
                    )

    if rows_by_id is not None:
        observed_tier1_leagues = {
            row.league_id
            for row in rows_by_id.values()
            if row.series_resolved and row.league_tier == "tier1" and row.league_id
        }
        if set(league_holdouts) != observed_tier1_leagues:
            raise ValidationFailure(
                "league leave-one-out holdouts do not exactly cover observed tier-1 leagues"
            )
        for league_id in sorted(observed_tier1_leagues):
            registered = league_holdouts[league_id]
            if len(registered) != 1:
                raise ValidationFailure(
                    f"league '{league_id}' must have exactly one leave-one-out holdout"
                )
            holdout = registered[0]
            if holdout.name != f"league_out_{league_id}":
                raise ValidationFailure(
                    f"league '{league_id}' holdout must be named league_out_{league_id}"
                )
            expected = {
                row.row_id
                for row in rows_by_id.values()
                if row.series_resolved
                and row.league_tier == "tier1"
                and row.league_id == league_id
            }
            if set(holdout.row_ids) != expected:
                raise ValidationFailure(
                    f"league holdout '{holdout.name}' is partial or contains extra rows"
                )

        resolved_non_sealed = {
            row.row_id
            for row in rows_by_id.values()
            if row.series_resolved and row.row_id not in exclusive_sealed_rows
        }
        missing_assignment = sorted(resolved_non_sealed - all_dev_rows)
        if missing_assignment:
            raise ValidationFailure(
                "resolved non-sealed rows are missing from the frozen development union: "
                f"{missing_assignment}"
            )
        unresolved_assignment = sorted(
            row_id
            for row_id in all_dev_rows
            if row_id not in rows_by_id or not rows_by_id[row_id].series_resolved
        )
        if unresolved_assignment:
            raise ValidationFailure(
                f"unresolved or unknown rows appear in development: {unresolved_assignment}"
            )


def assert_exact_roster(row: EvalRow) -> None:
    """Require explicit exact-roster assignment metadata when roster identity is used."""
    roster_roles = row.metadata.get("roster_roles")
    roster_id = row.roster_id
    if row.roster_id and isinstance(roster_roles, Sequence) and len(roster_roles) > 0:
        if not isinstance(roster_roles, Sequence) or len(roster_roles) != 5 or len(set(roster_roles)) != 5:
            raise ValidationFailure(f"row {row.row_id}: exact roster role assignment is missing or invalid")
    elif row.roster_id and not roster_roles:
        raise ValidationFailure(f"row {row.row_id}: exact roster requires roster_roles metadata")


def assert_transform_identity(adapter) -> None:
    """Transform serving identity is required to pass L2 transform audit."""
    adapter_id = getattr(adapter, "adapter_id", "<unknown>")
    served_transform = getattr(adapter, "served_transform_sha256", None)
    serialized_transform = getattr(adapter, "serialized_transform_sha256", None)

    _require_non_empty_sha256(served_transform, field_name=f"adapter {adapter_id}: served_transform_sha256")
    _require_non_empty_sha256(serialized_transform, field_name=f"adapter {adapter_id}: serialized_transform_sha256")

    if served_transform != serialized_transform:
        raise ValidationFailure(
            f"adapter {adapter_id}: served transform does not match serialized transform"
        )


def assert_runtime_transform_identity(adapter) -> None:
    """Runtime manifest must point at the same serialized transform."""
    adapter_id = getattr(adapter, "adapter_id", "<unknown>")
    runtime_manifest = getattr(adapter, "runtime_transform_manifest_sha256", None)
    serialized_transform = getattr(adapter, "serialized_transform_sha256", None)

    _require_non_empty_sha256(runtime_manifest, field_name=f"adapter {adapter_id}: runtime_transform_manifest_sha256")
    _require_non_empty_sha256(serialized_transform, field_name=f"adapter {adapter_id}: serialized_transform_sha256")

    if runtime_manifest != serialized_transform:
        raise ValidationFailure(
            f"adapter {adapter_id}: runtime transform manifest does not match serialized transform"
        )


def assert_runtime_artifact_identity(adapter) -> None:
    """Runtime artifact hash must match runtime manifest hash."""
    adapter_id = getattr(adapter, "adapter_id", "<unknown>")
    runtime_artifact = getattr(adapter, "runtime_artifact_sha256", None)
    runtime_manifest = getattr(adapter, "runtime_artifact_manifest_sha256", None)

    _require_non_empty_sha256(runtime_artifact, field_name=f"adapter {adapter_id}: runtime_artifact_sha256")
    _require_non_empty_sha256(runtime_manifest, field_name=f"adapter {adapter_id}: runtime_artifact_manifest_sha256")

    if runtime_artifact != runtime_manifest:
        raise ValidationFailure(
            f"adapter {adapter_id}: runtime artifact hash does not match runtime manifest hash"
        )


def _extract_draft_protocol_cells(values: Mapping[str, Any]) -> tuple[tuple[str, ...], ...]:
    """Normalize protocol/order/side support counts into a stable tuple form."""
    protocol_counts = values.get("protocol_order_side_cell_counts")
    if not isinstance(protocol_counts, Mapping):
        return tuple()
    cells: list[tuple[str, ...]] = []
    for protocol, order_map in protocol_counts.items():
        if not isinstance(order_map, Mapping):
            raise ValidationFailure("draft-order protocol counts must map protocols to order buckets")
        for order, side_map in order_map.items():
            if not isinstance(side_map, Mapping):
                raise ValidationFailure("draft-order side counts must map order to side buckets")
            for side in side_map:
                cells.append((str(protocol), str(order), str(side)))
    return tuple(cells)


def _canonical_protocol(protocol: object) -> str:
    if protocol is None:
        return ""
    return str(protocol).strip().lower()


def _require_positive_finite(value: object, *, field_name: str) -> float:
    numeric = _get_float(value, field_name=field_name)
    if not math.isfinite(numeric):
        raise ValidationFailure(f"{field_name} must be finite")
    if numeric <= 0.0:
        raise ValidationFailure(f"{field_name} must be positive")
    return numeric


def assert_draft_order_diagnostics(registry: EvaluationRegistry) -> None:
    """Validate preregistered draft-order identification diagnostics."""
    analysis = dict(registry.draft_order_analysis)
    protocol_counts = analysis.get("protocol_order_side_cell_counts")
    if not isinstance(protocol_counts, Mapping) or not protocol_counts:
        raise ValidationFailure("draft-order analysis is missing cell-count specification")

    cell_supports = analysis.get("protocol_order_side_positive_cells")
    if cell_supports is not None and not isinstance(cell_supports, Mapping):
        raise ValidationFailure("draft-order positive support matrix is malformed")

    required_protocols = set(str(p) for p in analysis.get("protocols", ()) if str(p))
    if required_protocols:
        observed_protocols = set(map(str, protocol_counts.keys()))
        missing = sorted(required_protocols - observed_protocols)
        if missing:
            raise ValidationFailure(
                f"draft-order analysis missing protocol strata: {', '.join(missing)}"
            )

    requested_order_coeff = bool(analysis.get("order_coeff_requested", False))
    design_rank = analysis.get("design_rank")
    design_columns = analysis.get("design_columns")
    condition_number = analysis.get("condition_number")

    for key in _DRAFT_ORDER_KEY:
        if key not in analysis:
            raise ValidationFailure(f"draft-order analysis missing required field '{key}'")

    design_rank_f = _get_float(design_rank, field_name="design_rank")
    design_columns_f = _get_float(design_columns, field_name="design_columns")
    condition_number_f = _get_float(condition_number, field_name="condition_number")

    if design_rank_f <= 0:
        raise ValidationFailure("draft-order design rank must be positive")
    if design_columns_f <= 0:
        raise ValidationFailure("draft-order design columns must be positive")
    if condition_number_f <= 0:
        raise ValidationFailure("draft-order condition number must be positive")

    # Empty support cells fail identification support checks and can indicate
    # missing positivity support.
    positive_cell_count = 0
    for protocol, order_bucket in protocol_counts.items():
        if not isinstance(order_bucket, Mapping):
            continue
        for order, side_bucket in order_bucket.items():
            if not isinstance(side_bucket, Mapping):
                raise ValidationFailure("draft-order side counts must map to side buckets")
            for side, count in side_bucket.items():
                try:
                    cell_count = int(count)
                except (TypeError, ValueError):
                    raise ValidationFailure(
                        f"draft-order cell count {protocol}/{order}/{side} is invalid: {count!r}"
                    )
                if cell_count <= 0:
                    raise ValidationFailure(
                        f"draft-order cell count {protocol}/{order}/{side} is not positive: {cell_count}"
                    )
                positive_cell_count += 1

                if cell_supports is not None:
                    protocol_support = cell_supports.get(str(protocol), {})
                    if not isinstance(protocol_support, Mapping):
                        raise ValidationFailure(
                            f"draft-order positive support map is invalid for protocol {protocol}"
                        )
                    order_support = protocol_support.get(str(order), {})
                    if not isinstance(order_support, Mapping):
                        raise ValidationFailure(
                            f"draft-order positive support map is invalid for protocol={protocol}, order={order}"
                        )
                    side_support = order_support.get(str(side))
                    try:
                        side_support_f = float(side_support)
                    except (TypeError, ValueError):
                        raise ValidationFailure(
                            f"draft-order cell support {protocol}/{order}/{side} is not numeric: {side_support!r}"
                        )
                    if side_support_f <= 0:
                        raise ValidationFailure(
                            f"draft-order positive support {protocol}/{order}/{side} is not positive"
                        )

    if positive_cell_count == 0:
        raise ValidationFailure("draft-order analysis has no positive support cells")

    cells = _extract_draft_protocol_cells(analysis)
    if len(cells) == 0:
        raise ValidationFailure("draft-order support cell specification is empty")

    missing_cells = analysis.get("missing_protocol_order_side_cells")
    if missing_cells is not None:
        if not isinstance(missing_cells, Sequence):
            raise ValidationFailure("draft-order missing cell summary must be a sequence")

    if requested_order_coeff:
        if design_rank_f < design_columns_f:
            raise ValidationFailure("unavailable_collinear: draft-order design is rank-deficient")

    # Posterior correlation/confounding check uses absolute correlation bound.
    confounding = analysis.get("posterior_correlation")
    if confounding is None:
        raise ValidationFailure("draft-order analysis missing posterior_correlation")
    corr = _get_float(confounding, field_name="posterior_correlation")
    if abs(corr) > 1.0:
        raise ValidationFailure(
            f"draft-order posterior correlation out of range: {corr!r}"
        )


def assert_terminal_probability_wording(adapter, requested_terminal: bool) -> None:
    if requested_terminal and not bool(getattr(adapter, "terminal_probability_wording_approved", False)):
        raise ValidationFailure(
            f"adapter {getattr(adapter, 'adapter_id', '<unknown>')}: terminal probability wording not approved"
        )


def assert_prefix_probability_wording(adapter, requested_prefixes: Sequence[str]) -> None:
    approved_prefixes = getattr(adapter, "prefix_probability_wording_approved", {}) or {}
    for prefix in requested_prefixes:
        if approved_prefixes.get(prefix) is not True:
            raise ValidationFailure(
                f"adapter {getattr(adapter, 'adapter_id', '<unknown>')}: prefix '{prefix}' probability wording not approved"
            )


def assert_row_prediction_values(predictions: Sequence[MatchPrediction]) -> None:
    """Ensure every returned row has finite probabilities and ordered intervals."""
    for prediction in predictions:
        if not _is_finite_scalar(prediction.raw_probability):
            raise ValidationFailure(f"row {prediction.row_id}: non-finite raw_probability")
        if not _is_finite_scalar(prediction.final_probability()):
            raise ValidationFailure(f"row {prediction.row_id}: non-finite final_probability")

        p = prediction.final_probability()
        if p < 0.0 or p > 1.0:
            raise ValidationFailure(f"row {prediction.row_id}: final probability out of range {p}")

        lower = prediction.lower_95 if prediction.lower_95 is not None else p
        upper = prediction.upper_95 if prediction.upper_95 is not None else p

        if not _is_finite_scalar(lower):
            raise ValidationFailure(f"row {prediction.row_id}: non-finite lower interval")
        if not _is_finite_scalar(upper):
            raise ValidationFailure(f"row {prediction.row_id}: non-finite upper interval")

        if float(lower) < 0.0 or float(upper) > 1.0:
            raise ValidationFailure(
                f"row {prediction.row_id}: interval bounds out of [0,1] ({lower}, {upper})"
            )
        if float(lower) > float(upper):
            raise ValidationFailure(
                f"row {prediction.row_id}: invalid interval order ({lower} > {upper})"
            )


def _assert_no_nan_or_inf(value: object, label: str, row_id: str) -> None:
    if not _is_finite_scalar(value):
        raise ValidationFailure(f"row {row_id}: {label} is not finite ({value})")


def assert_no_label_leakage(adapter, fit_state, rows: Sequence[EvalRow]) -> None:
    """Predictions must not depend on labels under fixed fit state."""
    if not rows:
        return

    baseline = list(adapter.predict(fit_state, rows, mode="terminal"))
    altered_rows = [
        row.with_mutated_label(1 - row.label if row.label in (0, 1) else row.label)
        for row in rows
    ]
    altered = list(adapter.predict(fit_state, altered_rows, mode="terminal"))

    if len(baseline) != len(altered):
        raise ValidationFailure("prediction counts changed under label mutation")

    for pred_base, pred_alt in zip(baseline, altered):
        if pred_base.row_id != pred_alt.row_id:
            raise ValidationFailure("row-order changed under leakage check")
        if abs(pred_base.final_probability() - pred_alt.final_probability()) > 1e-12:
            raise ValidationFailure(
                f"adapter {getattr(adapter, 'adapter_id', '<unknown>')}: row {pred_base.row_id} is label-sensitive\n"
                f"terminal prediction changed under label mutation"
            )


def assert_invariant_ledger_reconciles(predictions: Sequence[MatchPrediction]) -> None:
    """Reconcile contribution ledger to raw score where provided."""
    for prediction in predictions:
        if not prediction.ledger:
            continue
        total = float(sum(float(v) for v in prediction.ledger.values()))
        if prediction.raw_logit is None:
            raise ValidationFailure(f"row {prediction.row_id}: raw_logit missing while ledger exists")
        if abs(total - float(prediction.raw_logit)) > 1e-6:
            raise ValidationFailure(
                f"row {prediction.row_id}: ledger total {total} != raw_logit {prediction.raw_logit}"
            )


def assert_role_invariance(predictions: MatchPredictionMap, row_pairs: Mapping[str, str]) -> None:
    for left, right in row_pairs.items():
        if left not in predictions or right not in predictions:
            raise ValidationFailure(f"missing role-pair predictions for invariance check: {left}, {right}")
        p_left = predictions[left].final_probability()
        p_right = predictions[right].final_probability()
        if abs(p_left - p_right) > 1e-9:
            raise ValidationFailure(f"role invariance pair ({left},{right}) mismatched: {p_left} != {p_right}")


def assert_side_swap(predictions: MatchPredictionMap, pairings: Mapping[str, str]) -> None:
    for left, right in pairings.items():
        if left not in predictions or right not in predictions:
            raise ValidationFailure(f"missing side-swap pair predictions: {left}, {right}")
        p_left = predictions[left].final_probability()
        p_right = predictions[right].final_probability()
        if abs((p_left + p_right) - 1.0) > 1e-9:
            raise ValidationFailure(f"side-swap pair ({left},{right}) violates complement symmetry")


def assert_bootstrap_not_map_level(
    cluster_ids: Sequence[str],
    row_ids: Sequence[str],
    *,
    rows_by_id: Mapping[str, EvalRow],
) -> None:
    """Reject map-level cluster bootstrap and unknown series clustering."""
    if len(cluster_ids) != len(row_ids):
        raise ValidationFailure("cluster_ids and row_ids must align")

    for row_id, cluster_id in zip(row_ids, cluster_ids):
        if row_id not in rows_by_id:
            raise ValidationFailure(f"bootstrap row {row_id} not found in row index")
        if cluster_id == row_id:
            raise ValidationFailure("map-level bootstrap units detected; series-level clustering is required")
        if rows_by_id[row_id].series_id != cluster_id:
            raise ValidationFailure(
                f"cluster id {cluster_id} for row {row_id} does not match series id {rows_by_id[row_id].series_id}"
            )


def assert_unresolved_series_not_in_primary_bootstrap(
    predictions: MatchPredictionMap, rows_by_id: Mapping[str, EvalRow]
) -> None:
    for row_id in predictions:
        if row_id not in rows_by_id:
            raise ValidationFailure(f"prediction row {row_id} missing from row index")
        if not rows_by_id[row_id].series_resolved:
            raise ValidationFailure(f"unresolved series row {row_id} entered primary bootstrap")


def assert_sealed_rows_immutable(
    registry: EvaluationRegistry,
    snapshot: Mapping[str, str],
    rows_by_id: Mapping[str, EvalRow],
) -> None:
    """Verify sealed rows exactly match a previously anchored snapshot."""
    for holdout in registry.split_plan.sealed_holdouts:
        for row_id in holdout.row_ids:
            if row_id not in rows_by_id:
                raise ValidationFailure(f"sealed holdout {holdout.name} references unknown row {row_id}")
            expected = snapshot.get(row_id)
            if expected is None:
                raise ValidationFailure(
                    f"sealed holdout {holdout.name} missing fingerprint for row {row_id}"
                )
            if rows_by_id[row_id].fingerprint() != expected:
                raise ValidationFailure(
                    f"sealed holdout row {row_id} changed since snapshot"
                )


def assert_python_runtime_probabilities(adapter, fit_state, rows: Sequence[EvalRow]) -> None:
    py = [pred.final_probability() for pred in adapter.predict(fit_state, rows, mode="terminal")]
    rt = [pred.final_probability() for pred in adapter.runtime_predict(fit_state, rows, mode="terminal")]
    if len(py) != len(rt):
        raise ValidationFailure(
            f"runtime output size mismatch for {getattr(adapter, 'adapter_id', '<unknown>')}"
        )
    for row, p_val, r_val in zip(rows, py, rt):
        _assert_no_nan_or_inf(p_val, "python probability", row.row_id)
        _assert_no_nan_or_inf(r_val, "runtime probability", row.row_id)
        if abs(float(p_val) - float(r_val)) > 1e-9:
            raise ValidationFailure(
                f"row {row.row_id}: python/runtime mismatch for {getattr(adapter, 'adapter_id', '<unknown>')}"
            )


def _assert_required_pair_coverage(
    *,
    predictions: MatchPredictionMap,
    required_pairs: Sequence[tuple[str, str]],
    kind: str,
) -> None:
    if not required_pairs:
        return
    missing: list[tuple[str, str]] = []
    for left, right in required_pairs:
        if left not in predictions or right not in predictions:
            missing.append((left, right))
    if missing:
        raise ValidationFailure(
            f"required {kind} pairs missing: {', '.join(f'{left}:{right}' for left, right in missing)}"
        )


def assert_required_role_invariance(predictions: MatchPredictionMap, required_pairs: Sequence[tuple[str, str]]) -> None:
    _assert_required_pair_coverage(predictions=predictions, required_pairs=required_pairs, kind="role-invariance")


def assert_required_side_swap(predictions: MatchPredictionMap, required_pairs: Sequence[tuple[str, str]]) -> None:
    _assert_required_pair_coverage(predictions=predictions, required_pairs=required_pairs, kind="side-swap")
