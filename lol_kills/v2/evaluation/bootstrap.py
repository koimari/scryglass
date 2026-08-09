"""Clustered bootstrap utilities for candidate-vs-baseline deltas."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

from .checks import ValidationFailure


@dataclass(frozen=True)
class BootstrapResult:
    point: float
    lower_95: float
    upper_95: float
    cluster_count: int
    resolved_cluster_count: int
    cluster_unit: str
    cluster_size_distribution: Mapping[str, int] = field(default_factory=dict)


def _cluster_summary(cluster_map: Mapping[str, list[float]]) -> dict[str, float]:
    return {cluster_id: len(values) for cluster_id, values in cluster_map.items()}


def _validate_bootstrap_inputs(
    deltas: Sequence[float],
    cluster_ids: Sequence[str],
    resolved_mask: Sequence[bool],
    *,
    row_ids: Sequence[str] | None = None,
) -> None:
    if len(deltas) != len(cluster_ids) or len(deltas) != len(resolved_mask):
        raise ValueError("deltas, cluster_ids, and resolved_mask must match in length")
    if len(deltas) == 0:
        raise ValueError("bootstrap inputs are empty")
    if row_ids is not None and len(row_ids) != len(deltas):
        raise ValueError("row_ids and deltas length must match")
    if row_ids is not None:
        if len(set(row_ids)) != len(row_ids):
            raise ValidationFailure("map-level bootstrap units detected; row_ids must not repeat")


def series_cluster_bootstrap(
    deltas: Sequence[float],
    cluster_ids: Sequence[str],
    resolved_mask: Sequence[bool],
    *,
    row_ids: Sequence[str] | None = None,
    n_boot: int = 2000,
    random_seed: int = 123,
    cluster_unit: str = "series",
) -> BootstrapResult:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    _validate_bootstrap_inputs(deltas, cluster_ids, resolved_mask, row_ids=row_ids)

    deltas_arr = np.asarray(list(deltas), dtype=float)
    cluster_map: dict[str, list[float]] = {}
    for idx, (delta, cluster_id, is_resolved) in enumerate(zip(deltas_arr, cluster_ids, resolved_mask)):
        if not is_resolved:
            continue

        cluster_map.setdefault(cluster_id, []).append(float(delta))

    if not cluster_map:
        raise ValueError("no resolved-series rows available for bootstrap")

    cluster_values = np.array([float(np.mean(values)) for values in cluster_map.values()])
    rng = np.random.default_rng(random_seed)
    sampled = np.empty(n_boot, dtype=float)
    n_clusters = cluster_values.size

    for i in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        sampled[i] = float(cluster_values[draw].mean())

    point = float(cluster_values.mean())
    return BootstrapResult(
        point=point,
        lower_95=float(np.quantile(sampled, 0.025)),
        upper_95=float(np.quantile(sampled, 0.975)),
        cluster_count=len(cluster_map),
        resolved_cluster_count=n_clusters,
        cluster_unit=cluster_unit,
        cluster_size_distribution=_cluster_summary(cluster_map),
    )


def grouped_bootstrap_sensitivity(
    deltas: Sequence[float],
    cluster_ids: Sequence[str],
    resolved_mask: Sequence[bool],
    row_ids: Sequence[str] | None = None,
    *,
    group_fn: Callable[[str], str],
    n_boot: int = 2000,
    random_seed: int = 123,
    cluster_unit: str = "sensitivity",
) -> BootstrapResult:
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    if random_seed < 0:
        raise ValueError("random_seed must be non-negative")
    _validate_bootstrap_inputs(deltas, cluster_ids, resolved_mask, row_ids=row_ids)

    deltas_arr = np.asarray(list(deltas), dtype=float)
    groups: dict[str, list[float]] = {}
    row_ids_arr = list(row_ids) if row_ids is not None else None

    for idx, (delta, cluster_id, is_resolved) in enumerate(zip(deltas_arr, cluster_ids, resolved_mask)):
        if not is_resolved:
            continue

        group_source = row_ids_arr[idx] if row_ids_arr is not None else cluster_id
        group = group_fn(group_source)
        groups.setdefault(group, []).append(float(delta))

    if not groups:
        raise ValueError("no resolved rows available for sensitivity bootstrap")

    cluster_values = np.array([float(np.mean(values)) for values in groups.values()])
    n_clusters = cluster_values.size
    rng = np.random.default_rng(random_seed)
    sampled = np.empty(n_boot, dtype=float)

    for i in range(n_boot):
        draw = rng.integers(0, n_clusters, size=n_clusters)
        sampled[i] = float(cluster_values[draw].mean())

    point = float(cluster_values.mean())
    return BootstrapResult(
        point=point,
        lower_95=float(np.quantile(sampled, 0.025)),
        upper_95=float(np.quantile(sampled, 0.975)),
        cluster_count=len(cluster_values),
        resolved_cluster_count=n_clusters,
        cluster_unit=cluster_unit,
        cluster_size_distribution={cluster_id: len(values) for cluster_id, values in groups.items()},
    )
