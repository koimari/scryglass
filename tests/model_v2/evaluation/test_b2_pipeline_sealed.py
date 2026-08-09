from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.evaluation.b2_artifacts import (
    B2_ARTIFACT_IDS,
    verify_b2_artifact_refs,
)
from lol_kills.v2.evaluation.b2_pipeline import (
    B2_REQUIRED_HARD_GATES,
    build_b2_validation_report,
    verify_b2_validation_report,
)
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.fixtures import ToyAdapter, build_synthetic_rows
from lol_kills.v2.evaluation.pipeline import evaluate_candidate
from lol_kills.v2.evaluation.sealed import REQUIRED_SEALED_HARD_GATES
from lol_kills.v2.evaluation.splitter import load_evaluation_registry
from lol_kills.v2.evaluation.types import ArtifactRef


REGISTRY = load_evaluation_registry("data/lol/v2/evaluation/synthetic-registry-frozen.json")


def test_all_four_artifacts_and_twelve_gates_are_exact() -> None:
    assert tuple(ref.artifact_id for ref in REGISTRY.b2_artifact_refs) == B2_ARTIFACT_IDS
    assert len(verify_b2_artifact_refs(REGISTRY)) == 4
    report = build_b2_validation_report(REGISTRY)
    assert set(report["hard_gates"]) == set(B2_REQUIRED_HARD_GATES)
    assert all(report["hard_gates"].values())
    assert set(B2_REQUIRED_HARD_GATES) <= set(REQUIRED_SEALED_HARD_GATES)


@pytest.mark.parametrize("index", range(4))
def test_each_artifact_ref_hash_mutation_fails_exact_artifact_gate(index: int) -> None:
    refs = list(REGISTRY.b2_artifact_refs)
    refs[index] = replace(refs[index], raw_sha256="f"*64)
    with pytest.raises(ValidationFailure, match="raw bytes mismatch"):
        verify_b2_artifact_refs(replace(REGISTRY, b2_artifact_refs=tuple(refs)))


def test_duplicate_or_missing_artifact_ref_fails() -> None:
    with pytest.raises(ValidationFailure):
        verify_b2_artifact_refs(replace(REGISTRY, b2_artifact_refs=REGISTRY.b2_artifact_refs[:3]))
    with pytest.raises(ValidationFailure):
        verify_b2_artifact_refs(replace(REGISTRY, b2_artifact_refs=REGISTRY.b2_artifact_refs[:3] + (REGISTRY.b2_artifact_refs[0],)))


def test_forged_self_rehashed_report_fails_fresh_replay() -> None:
    report = build_b2_validation_report(REGISTRY)
    report["hard_gates"]["calibration_diagnostic_verified"] = False
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    from lol_kills.v2.evaluation.types import canonical_sha256
    report["report_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValidationFailure, match="fresh executable replay"):
        verify_b2_validation_report(report, REGISTRY)


def test_development_pipeline_recomputes_b2_gates() -> None:
    report = evaluate_candidate(
        ToyAdapter(source_tree_sha256=REGISTRY.source_tree_sha256),
        build_synthetic_rows(),
        REGISTRY,
    )
    assert set(B2_REQUIRED_HARD_GATES) <= set(report.hard_gate_results)
    assert all(report.hard_gate_results[name] for name in B2_REQUIRED_HARD_GATES)


def test_synthetic_report_and_registry_are_permanently_nonproduction() -> None:
    report = build_b2_validation_report(REGISTRY)
    assert report["synthetic_only"] is True
    assert report["production_eligible"] is False
    assert REGISTRY.is_synthetic_registry is True
    assert report["production_unavailable_reasons"] == [
        "missing_real_heldout_b2_diagnostics",
        "missing_b3_resolved_cluster_dependence_evidence",
        "missing_l4_l9_model_authorities",
    ]
