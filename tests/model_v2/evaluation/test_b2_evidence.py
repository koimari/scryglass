from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import pytest

from lol_kills.v2.evaluation.b2_artifacts import verify_b2_artifact_refs
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.evidence import (
    build_measured_selection_report,
    replay_evidence_value,
    select_recipe,
    verify_evidence_registry,
    verify_measured_selection_report,
    verify_recipe,
)
from lol_kills.v2.evaluation.splitter import load_evaluation_registry


REGISTRY = load_evaluation_registry("data/lol/v2/evaluation/synthetic-registry-frozen.json")
PAYLOAD = verify_b2_artifact_refs(REGISTRY)[
    "scryglass:b2:evidence-candidate-registry:v1"
]
RECIPES = {item["method_id"]: item for item in PAYLOAD["candidates"]}


def test_three_families_and_measured_selection_are_exact() -> None:
    loaded = verify_evidence_registry(PAYLOAD)
    assert set(loaded) == set(RECIPES)
    report = build_measured_selection_report(PAYLOAD)
    assert len(report["selections"]) == 15
    assert all(item["passes"] for item in report["measurements"])
    assert {item["family"] for item in report["selections"]} == {
        "posterior_information", "precision", "source_context_coverage"
    }


def test_values_replay_only_from_verified_dependency_bytes() -> None:
    recipe = RECIPES["standardized_posterior_mean_displacement"]
    result = replay_evidence_value(recipe)
    assert result["value"] > 0
    with pytest.raises(ValidationFailure, match="differ from verified"):
        replay_evidence_value(
            recipe,
            {"posterior_draws": [0, 1], "prior_draws": [-100, 100]},
        )


@pytest.mark.parametrize("field", ["raw_sha256", "canonical_payload_sha256"])
def test_dependency_hash_mutation_rejects(field: str) -> None:
    bad = deepcopy(RECIPES["interval_contraction"])
    bad["dependencies"][0][field] = "f" * 64
    with pytest.raises(ValidationFailure):
        verify_recipe(bad)


@pytest.mark.parametrize("field", ["code_sha256", "config_sha256"])
def test_code_or_config_hash_mutation_rejects(field: str) -> None:
    bad = deepcopy(RECIPES["interval_contraction"])
    bad[field] = "e" * 64
    with pytest.raises(ValidationFailure, match="code or config"):
        verify_recipe(bad)


@pytest.mark.parametrize("mutation", ["missing", "extra", "cardinality", "substitution"])
def test_dependency_dag_is_exact(mutation: str) -> None:
    bad = deepcopy(RECIPES["interval_contraction"])
    if mutation == "missing":
        bad["dependencies"].pop()
    elif mutation == "extra":
        bad["dependencies"].append(deepcopy(bad["dependencies"][0]))
        bad["dependencies"][-1]["role"] = "popularity"
    elif mutation == "cardinality":
        bad["dependencies"][0]["cardinality"] = "one"
    else:
        bad["dependencies"][1]["role"] = "prior_draws"
    with pytest.raises(ValidationFailure):
        verify_recipe(bad)


@pytest.mark.parametrize("mutation", ["family", "units", "formula", "criterion"])
def test_reclassification_or_hidden_volume_dependency_rejects(mutation: str) -> None:
    if mutation == "criterion":
        bad_registry = deepcopy(PAYLOAD)
        bad_registry["selection"]["criteria"] = ["game_count"]
        with pytest.raises(ValidationFailure):
            verify_evidence_registry(bad_registry)
        return
    bad = deepcopy(RECIPES["standardized_posterior_mean_displacement"])
    if mutation == "family":
        bad["family"] = "precision"
    elif mutation == "units":
        bad["units"] = "games"
    else:
        bad["boundaries"]["formula"] = "posterior_displacement / sample_size"
    with pytest.raises(ValidationFailure):
        verify_recipe(bad)


def _fixture_root(tmp_path: Path) -> tuple[Path, dict]:
    config = Path("data/lol/v2/evaluation/b2/evidence-config.json")
    target_config = tmp_path / config
    target_config.parent.mkdir(parents=True)
    shutil.copyfile(config, target_config)
    recipe = deepcopy(RECIPES["interval_contraction"])
    for dependency in recipe["dependencies"]:
        source = Path(dependency["locator"])
        target = tmp_path / source
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return tmp_path, recipe


def test_dependency_symlink_rejects(tmp_path: Path) -> None:
    root, recipe = _fixture_root(tmp_path)
    dependency = recipe["dependencies"][0]
    original = root / dependency["locator"]
    moved = original.with_name("real.json")
    original.rename(moved)
    original.symlink_to(moved.name)
    with pytest.raises(ValidationFailure, match="symlink"):
        verify_recipe(recipe, root)


def test_dependency_hardlink_alias_rejects(tmp_path: Path) -> None:
    root, recipe = _fixture_root(tmp_path)
    first = root / recipe["dependencies"][0]["locator"]
    second = root / recipe["dependencies"][1]["locator"]
    second.unlink()
    os.link(first, second)
    recipe["dependencies"][1].update(
        {
            key: recipe["dependencies"][0][key]
            for key in ("artifact_id", "raw_sha256", "canonical_payload_sha256")
        }
    )
    with pytest.raises(ValidationFailure, match="hard-linked"):
        verify_recipe(recipe, root)


def test_empty_duplicate_ambiguous_and_posthoc_selection_reject() -> None:
    for candidates in ([], PAYLOAD["candidates"][:2], PAYLOAD["candidates"] + [PAYLOAD["candidates"][0]]):
        bad = deepcopy(PAYLOAD)
        bad["candidates"] = deepcopy(candidates)
        with pytest.raises(ValidationFailure):
            build_measured_selection_report(bad)
    report = build_measured_selection_report(PAYLOAD)
    report["selections"][0]["method_id"] = "post-hoc"
    with pytest.raises(ValidationFailure, match="does not replay"):
        verify_measured_selection_report(report, PAYLOAD)
    with pytest.raises(ValidationFailure, match="caller-asserted"):
        select_recipe([{"method_id": "caller"}])


def test_no_aggregate_confidence_scalar_is_emitted() -> None:
    report = build_measured_selection_report(PAYLOAD)
    serialized = json.dumps(report)
    assert '"confidence"' not in serialized
    assert "aggregate_evidence_score" not in serialized
