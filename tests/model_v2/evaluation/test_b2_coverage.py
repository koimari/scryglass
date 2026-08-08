from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.coverage import (
    aggregate_forecast_coverage,
    simulation_parameter_coverage,
)


def simulation_cases():
    rng = np.random.default_rng(2)
    return [
        {
            "case_id": f"c{i}", "output_type": "draft_score",
            "parameter": "p", "stratum": "s", "truth": truth,
            "posterior_draws": rng.normal(truth, .05, 200).tolist(),
        }
        for i, truth in enumerate((.3, .5, .7))
    ]


def test_simulation_parameter_coverage_contains_truth_rank_width_and_hashes() -> None:
    report = simulation_parameter_coverage(
        simulation_cases(),
        generator_sha256="1"*64, inference_sha256="2"*64,
        seed=2, artifact_sha256="3"*64,
    )
    assert report["synthetic_only"] is True
    assert report["production_eligible"] is False
    assert all({"sbc_rank", "width", "covered"} <= set(item) for item in report["evidence"])
    assert len(report["report_sha256"]) == 64


@pytest.mark.parametrize("missing", ["truth", "posterior_draws"])
def test_simulation_missing_truth_or_draws_fails(missing: str) -> None:
    cases = simulation_cases()
    cases[0].pop(missing)
    with pytest.raises(ValidationFailure):
        simulation_parameter_coverage(
            cases, generator_sha256="1"*64, inference_sha256="2"*64,
            seed=2, artifact_sha256="3"*64,
        )


def aggregate_fixture():
    rng = np.random.default_rng(4)
    rows = [
        {"row_id": "a1", "series_id": "a", "outcome": 0},
        {"row_id": "a2", "series_id": "a", "outcome": 1},
        {"row_id": "b1", "series_id": "b", "outcome": 1},
        {"row_id": "b2", "series_id": "b", "outcome": 1},
    ]
    cells = [
        {"cell_id": "a", "row_ids": ["a1", "a2"], "posterior_predictive_draws": rng.binomial(1,.1,(500,2)).tolist(), "baseline_width": 1.0, "width_margin": 0, "higher_cluster_support": 1, "resampling_unit": "series"},
        {"cell_id": "b", "row_ids": ["b1", "b2"], "posterior_predictive_draws": rng.binomial(1,.9,(500,2)).tolist(), "baseline_width": 1.0, "width_margin": 0, "higher_cluster_support": 1, "resampling_unit": "series"},
    ]
    return rows, cells


def run_aggregate(rows=None, cells=None):
    base_rows, base_cells = aggregate_fixture()
    return aggregate_forecast_coverage(
        base_rows if rows is None else rows,
        base_cells if cells is None else cells,
        dependence_design={"id": "synthetic-provisional", "provisional": True},
        procedure_sha256="4"*64,
    )


def test_aggregate_coverage_uses_joint_predictive_aggregate_not_latent_p() -> None:
    report = run_aggregate()
    assert all(item["series_support"] == 1 for item in report["cells"])
    assert all(0 < item["width"] < 1 for item in report["cells"])


@pytest.mark.parametrize("mutation", ["drop", "overlap", "series_split", "missing_interval", "map_level", "trivial"])
def test_aggregate_partition_and_interval_mutations_fail(mutation: str) -> None:
    rows, cells = aggregate_fixture()
    if mutation == "drop":
        cells[0]["row_ids"].pop()
        cells[0]["posterior_predictive_draws"] = [draw[:1] for draw in cells[0]["posterior_predictive_draws"]]
    elif mutation == "overlap":
        cells[1]["row_ids"][0] = "a1"
    elif mutation == "series_split":
        cells[0]["row_ids"] = ["a1", "b1"]
        cells[1]["row_ids"] = ["a2", "b2"]
    elif mutation == "missing_interval":
        cells[0].pop("posterior_predictive_draws")
    elif mutation == "map_level":
        cells[0]["resampling_unit"] = "map"
    else:
        cells[0]["posterior_predictive_draws"] = [[0,0],[1,1]] * 100
    with pytest.raises((ValidationFailure, TypeError)):
        run_aggregate(rows, cells)


def test_point_interval_substitution_fails() -> None:
    cases = simulation_cases()
    cases[0]["posterior_draws"] = [cases[0]["truth"]] * 20
    with pytest.raises(ValidationFailure):
        simulation_parameter_coverage(
            cases, generator_sha256="1"*64, inference_sha256="2"*64,
            seed=2, artifact_sha256="3"*64,
        )
