from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import pytest

from lol_kills.v2.evaluation.b2_artifacts import (
    B2_ARTIFACT_IDS,
    verify_artifact_ref,
    verify_b2_artifact_refs,
    verify_frozen_b2_registry_authority,
)
from lol_kills.v2.evaluation.b2_pipeline import (
    B2_REQUIRED_HARD_GATES,
    build_b2_validation_report,
    verify_b2_validation_report,
)
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation import sealed as sealed_module
from lol_kills.v2.evaluation.pipeline import _build_fold_metric
from lol_kills.v2.evaluation.splitter import load_evaluation_registry
from lol_kills.v2.evaluation.types import (
    ArtifactRef,
    MatchPrediction,
    canonical_json,
    canonical_sha256,
)


REGISTRY = load_evaluation_registry(
    "data/lol/v2/evaluation/synthetic-registry-frozen.json"
)


def _prediction(row_id: str, probability: float) -> MatchPrediction:
    return MatchPrediction(
        row_id=row_id,
        model_version="checkpoint-1",
        mode="terminal",
        raw_logit=0.0,
        raw_probability=probability,
        lower_95=None,
        upper_95=None,
    )


def test_live_metric_unavailable_calibration_is_typed_and_never_ideal() -> None:
    metric = _build_fold_metric(
        [0, 0, 0],
        [_prediction("r1", 0.2), _prediction("r2", 0.4), _prediction("r3", 0.6)],
        row_ids=["r1", "r2", "r3"],
    )
    assert metric.calibration_status == "unavailable"
    assert metric.calibration_reason == "constant_outcome"
    assert metric.calibration_support == 3
    assert metric.calibration_intercept is None
    assert metric.calibration_slope is None


def test_live_metric_has_no_bernoulli_probability_interval_coverage() -> None:
    metric = _build_fold_metric(
        [0, 1, 0, 1],
        [
            _prediction("r1", 0.2),
            _prediction("r2", 0.8),
            _prediction("r3", 0.4),
            _prediction("r4", 0.6),
        ],
        row_ids=["r1", "r2", "r3", "r4"],
    )
    assert "interval_coverage" not in vars(metric)
    assert metric.log_loss > 0
    assert metric.brier > 0


def _write_canonical(path: Path, payload: dict) -> ArtifactRef:
    raw = (canonical_json(payload) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return ArtifactRef(
        artifact_id=str(payload["artifact_id"]),
        locator=path.name,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        canonical_payload_sha256=canonical_sha256(payload),
    )


@pytest.mark.parametrize(
    "artifact_id,mutation",
    [
        (B2_ARTIFACT_IDS[1], "empty_evidence"),
        (B2_ARTIFACT_IDS[2], "empty_calibration"),
        (B2_ARTIFACT_IDS[3], "truncated_coverage"),
        (B2_ARTIFACT_IDS[0], "unauthorized_stratum"),
    ],
)
def test_artifact_specific_semantic_mutations_reject(
    tmp_path: Path, artifact_id: str, mutation: str
) -> None:
    source = Path(REGISTRY.b2_artifact_refs[B2_ARTIFACT_IDS.index(artifact_id)].locator)
    payload = json.loads(source.read_text())
    if mutation == "empty_evidence":
        payload["candidates"] = []
    elif mutation == "empty_calibration":
        payload["candidates"] = []
    elif mutation == "truncated_coverage":
        del payload["simulation_parameter_coverage"]["required"]
    else:
        payload["validation_strata"].append("caller-friendlier-stratum")
    ref = _write_canonical(tmp_path / source.name, payload)
    with pytest.raises(ValidationFailure):
        verify_artifact_ref(ref, tmp_path)


def test_self_rehashed_registry_and_changed_b1_lineage_reject() -> None:
    changed = replace(
        REGISTRY,
        source_snapshot_id="source://caller/self-rehashed",
        source_snapshot_sha256="a" * 64,
    )
    with pytest.raises(ValidationFailure, match="exact B1-registrar-authorized"):
        build_b2_validation_report(changed)


def test_parent_symlink_component_rejects(tmp_path: Path) -> None:
    source = Path(REGISTRY.b2_artifact_refs[0].locator).resolve()
    target = tmp_path / "real"
    target.mkdir()
    copied = target / source.name
    copied.write_bytes(source.read_bytes())
    (tmp_path / "alias").symlink_to(target, target_is_directory=True)
    original = REGISTRY.b2_artifact_refs[0]
    ref = replace(original, locator=f"alias/{source.name}")
    with pytest.raises(ValidationFailure, match="symlink component"):
        verify_artifact_ref(ref, tmp_path)


def test_hardlink_alias_across_refs_rejects(tmp_path: Path) -> None:
    source = Path(REGISTRY.b2_artifact_refs[0].locator)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(source.read_bytes())
    os.link(first, second)
    refs = list(REGISTRY.b2_artifact_refs)
    refs[0] = replace(refs[0], locator="first.json")
    refs[1] = replace(
        refs[1],
        locator="second.json",
        raw_sha256=refs[0].raw_sha256,
        canonical_payload_sha256=refs[0].canonical_payload_sha256,
    )
    # The remaining paths need only exist: inode alias rejection happens before
    # individual payload identity validation.
    for index in (2, 3):
        path = tmp_path / f"copy-{index}.json"
        path.write_bytes(Path(refs[index].locator).read_bytes())
        refs[index] = replace(refs[index], locator=path.name)
    with pytest.raises(ValidationFailure, match="hard-link alias"):
        verify_b2_artifact_refs(
            replace(REGISTRY, b2_artifact_refs=tuple(refs)), tmp_path
        )


@pytest.mark.parametrize("mutation", ["false", "omitted", "extra"])
def test_false_omitted_or_extra_gate_evidence_rejects(mutation: str) -> None:
    report = build_b2_validation_report(REGISTRY)
    if mutation == "false":
        report["hard_gates"]["calibration_diagnostic_verified"] = False
    elif mutation == "omitted":
        del report["gate_evidence"]["calibration_diagnostic_verified"]
    else:
        report["gate_evidence"]["caller_gate"] = {
            "gate": "caller_gate",
            "status": "pass",
            "predicate_evidence": {},
            "evidence_sha256": "f" * 64,
        }
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    report["report_sha256"] = canonical_sha256(unsigned)
    with pytest.raises(ValidationFailure):
        verify_b2_validation_report(report, REGISTRY)


def test_non_synthetic_registry_cannot_carry_synthetic_b2_authority() -> None:
    with pytest.raises(ValidationFailure):
        verify_b2_artifact_refs(replace(REGISTRY, is_synthetic_registry=False))


def test_forged_b1_compatibility_subset_cannot_authorize_synthetic_b2() -> None:
    forged = replace(
        REGISTRY,
        is_synthetic_registry=False,
        metrics=("caller_metric",),
        source_snapshot_id="source://registered/v2/evaluation/b1",
        source_snapshot_sha256="a" * 64,
        training_snapshot_id="source://registered/v2/evaluation/b1-training",
        training_snapshot_sha256="b" * 64,
        source_tree_sha256="c" * 64,
        noninferiority_provenance="registered-b1-test-policy",
        invalidation_reasons=("sealed_b1_registered_test_fixture",),
    )
    with pytest.raises(ValidationFailure, match="exact B1-registrar-authorized"):
        verify_frozen_b2_registry_authority(forged)
    with pytest.raises(ValidationFailure, match="non-synthetic registry"):
        verify_b2_artifact_refs(forged)


def test_b1_and_b2_sealed_gate_sets_are_layer_conditional() -> None:
    b1_registry = replace(
        REGISTRY,
        b2_artifact_refs=(),
        b2_validation_report_sha256="",
    )
    assert set(sealed_module._required_sealed_hard_gates(b1_registry)) == set(
        sealed_module.REQUIRED_B1_SEALED_HARD_GATES
    )
    assert not set(B2_REQUIRED_HARD_GATES) & set(
        sealed_module._required_sealed_hard_gates(b1_registry)
    )
    assert set(sealed_module._required_sealed_hard_gates(REGISTRY)) == set(
        sealed_module.REQUIRED_SEALED_HARD_GATES
    )
    assert set(B2_REQUIRED_HARD_GATES) <= set(
        sealed_module._required_sealed_hard_gates(REGISTRY)
    )
