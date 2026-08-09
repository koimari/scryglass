"""Correct simulation-parameter and aggregate forecast coverage."""

from __future__ import annotations

from collections import defaultdict
from statistics import NormalDist
from typing import Any, Mapping, Sequence

import numpy as np

from .checks import ValidationFailure
from .types import canonical_sha256


def _interval(draws: Sequence[float]) -> tuple[float, float]:
    values = np.asarray(draws, dtype=float)
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValidationFailure("coverage draws are missing or nonfinite")
    lo, hi = np.quantile(values, [.025, .975])
    if not float(lo) < float(hi):
        raise ValidationFailure("coverage interval is degenerate")
    return float(lo), float(hi)


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        raise ValidationFailure("coverage aggregation has zero support")
    z = NormalDist().inv_cdf(.975)
    p = successes / total
    denom = 1 + z*z/total
    center = (p + z*z/(2*total))/denom
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return float(max(0, center-half)), float(min(1, center+half))


def simulation_parameter_coverage(
    cases: Sequence[Mapping[str, Any]],
    *,
    generator_sha256: str,
    inference_sha256: str,
    seed: int,
    artifact_sha256: str,
) -> dict[str, Any]:
    if not cases:
        raise ValidationFailure("simulation coverage requires cases")
    evidence = []
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        required = {"case_id", "output_type", "parameter", "stratum", "truth", "posterior_draws"}
        if set(case) != required:
            raise ValidationFailure("simulation coverage case is missing truth/draws")
        truth = float(case["truth"])
        lo, hi = _interval(case["posterior_draws"])
        draws = np.asarray(case["posterior_draws"], dtype=float)
        item = {
            "case_id": case["case_id"],
            "covered": lo <= truth <= hi,
            "lower_95": lo,
            "upper_95": hi,
            "width": hi-lo,
            "sbc_rank": int(np.sum(draws < truth)),
            "draw_count": int(draws.size),
        }
        evidence.append(item)
        groups[(case["output_type"], case["parameter"], case["stratum"])].append(item)
    aggregates = []
    for key, items in sorted(groups.items()):
        widths = np.asarray([item["width"] for item in items])
        covered = sum(item["covered"] for item in items)
        lower, upper = _wilson(covered, len(items))
        aggregates.append({
            "output_type": key[0], "parameter": key[1], "stratum": key[2],
            "nominal_coverage": .95, "empirical_coverage": covered/len(items),
            "coverage_uncertainty": [lower, upper],
            "mean_width": float(widths.mean()), "median_width": float(np.median(widths)),
            "upper_tail_width": float(np.quantile(widths, .95)), "count": len(items),
        })
    report = {
        "kind": "simulation_parameter_coverage",
        "synthetic_only": True,
        "production_eligible": False,
        "generator_sha256": generator_sha256,
        "inference_sha256": inference_sha256,
        "seed": int(seed),
        "artifact_sha256": artifact_sha256,
        "evidence": evidence,
        "aggregates": aggregates,
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def aggregate_forecast_coverage(
    rows: Sequence[Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
    *,
    dependence_design: Mapping[str, Any],
    procedure_sha256: str,
) -> dict[str, Any]:
    if not rows or not cells:
        raise ValidationFailure("aggregate coverage requires rows and cells")
    row_by_id = {str(row["row_id"]): row for row in rows}
    if len(row_by_id) != len(rows):
        raise ValidationFailure("aggregate coverage row IDs are duplicated")
    assigned: list[str] = []
    series_cell: dict[str, str] = {}
    evidence = []
    for cell in cells:
        ids = list(cell.get("row_ids", ()))
        if not ids or len(ids) != len(set(ids)) or any(row_id not in row_by_id for row_id in ids):
            raise ValidationFailure("aggregate forecast cell membership is invalid")
        series_to_cells: dict[str, set[str]] = defaultdict(set)
        for row_id in ids:
            series_id = str(row_by_id[row_id]["series_id"])
            cell_id = str(cell["cell_id"])
            series_to_cells[series_id].add(cell_id)
            if series_id in series_cell and series_cell[series_id] != cell_id:
                raise ValidationFailure("series is split across forecast cells")
            series_cell[series_id] = cell_id
        if any(len(value) != 1 for value in series_to_cells.values()):
            raise ValidationFailure("series is split across forecast cells")
        assigned.extend(ids)
        draw_matrix = np.asarray(cell.get("posterior_predictive_draws"), dtype=float)
        if draw_matrix.ndim != 2 or draw_matrix.shape[1] != len(ids):
            raise ValidationFailure("joint posterior-predictive draws are missing")
        if cell.get("resampling_unit") == "map":
            raise ValidationFailure("map-level Bernoulli resampling is forbidden")
        statistics = draw_matrix.mean(axis=1)
        lo, hi = _interval(statistics)
        observed = float(np.mean([row_by_id[row_id]["outcome"] for row_id in ids]))
        width = hi-lo
        baseline_width = float(cell.get("baseline_width", 0))
        if width >= .999999 or baseline_width <= 0 or width > baseline_width + float(cell.get("width_margin", 0)):
            raise ValidationFailure("aggregate coverage width/noninferiority failed")
        evidence.append({
            "cell_id": cell["cell_id"], "row_ids": ids, "observed_aggregate": observed,
            "lower_95": lo, "upper_95": hi, "width": width, "covered": lo <= observed <= hi,
            "series_support": len(series_to_cells),
            "higher_cluster_support": int(cell.get("higher_cluster_support", 0)),
            "baseline_width": baseline_width,
        })
    if sorted(assigned) != sorted(row_by_id) or len(assigned) != len(set(assigned)):
        raise ValidationFailure("forecast cells overlap or drop eligible rows")
    covered = sum(item["covered"] for item in evidence)
    lower, upper = _wilson(covered, len(evidence))
    report = {
        "kind": "aggregate_forecast_coverage",
        "synthetic_only": True,
        "production_eligible": False,
        "dependence_design": dict(dependence_design),
        "procedure_sha256": procedure_sha256,
        "cells": evidence,
        "empirical_coverage": covered/len(evidence),
        "dependence_aware_uncertainty": [lower, upper],
        "width_distribution": {
            "mean": float(np.mean([item["width"] for item in evidence])),
            "median": float(np.median([item["width"] for item in evidence])),
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    return report
