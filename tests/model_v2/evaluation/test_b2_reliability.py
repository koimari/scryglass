from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest

from lol_kills.v2.evaluation.b2_artifacts import verify_b2_artifact_refs
from lol_kills.v2.evaluation.b2_artifacts import (
    verify_frozen_b2_registry_authority,
)
from lol_kills.v2.evaluation.b2_pipeline import (
    _build_gate_evidence,
    build_b2_validation_report,
    verify_b2_validation_report,
)
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.reliability import (
    VerifiedReliabilityMapping,
    _validate_diagnostic_record,
    _stable_id,
    _verify_b3_authority,
    audit_mapping_registry,
    load_verified_diagnostics,
    load_verified_mapping,
    resolve_reliability,
    verify_reliability_replay,
)
from lol_kills.v2.evaluation.splitter import load_evaluation_registry
from lol_kills.v2.evaluation.types import canonical_sha256


REGISTRY = load_evaluation_registry("data/lol/v2/evaluation/synthetic-registry-frozen.json")
MAPPING_PAYLOAD = verify_b2_artifact_refs(REGISTRY)[
    "scryglass:b2:reliability-registry:v1"
]
MAPPING = load_verified_mapping(REGISTRY.b2_artifact_refs[0])
DIAGNOSTICS = load_verified_diagnostics()


def output(output_type: str = "draft_score") -> dict:
    contexts = {
        item["output_type"]: item for item in MAPPING.payload["context_universe"]
    }
    context = deepcopy(contexts[output_type])
    mode = context.pop("mode")
    context.pop("output_type")
    context.pop("ood_state")
    return {"output_type": output_type, "mode": mode, "status": "ok", "context": context}


def diagnostic(output_type: str = "draft_score"):
    return next(item for item in DIAGNOSTICS if item.record["output_type"] == output_type)


def rehashed_record(**changes) -> dict:
    record = deepcopy(diagnostic().record)
    record.update(changes)
    record["record_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    return record


def test_mapping_is_total_and_exact_one() -> None:
    assert audit_mapping_registry(MAPPING_PAYLOAD)["registered_context_count"] == 5


@pytest.mark.parametrize("mutation", ["gap", "overlap", "duplicate_context", "duplicate_selector"])
def test_mapping_structural_mutations_fail_closed(mutation: str) -> None:
    bad = deepcopy(MAPPING_PAYLOAD)
    if mutation == "gap":
        bad["rules"] = bad["rules"][1:]
    elif mutation == "overlap":
        bad["rules"].append({"rule_id": "overlap", "selector": {}, "stratum_id": "stratum-draft"})
    elif mutation == "duplicate_context":
        bad["context_universe"].append(deepcopy(bad["context_universe"][0]))
    else:
        bad["rules"].append(deepcopy(bad["rules"][0]))
        bad["rules"][-1]["rule_id"] = "new-id-same-selector"
    with pytest.raises(ValidationFailure):
        audit_mapping_registry(bad)


@pytest.mark.parametrize("field,value", [
    ("rules", []),
    ("synthetic_only", False),
    ("production_eligible", True),
])
def test_detached_mapping_payload_even_if_object_hash_recomputed_rejects(field, value) -> None:
    bad = deepcopy(MAPPING.payload)
    bad[field] = value
    forged = replace(
        MAPPING,
        payload=bad,
        ref=replace(MAPPING.ref, canonical_payload_sha256=canonical_sha256(bad)),
    )
    with pytest.raises(ValidationFailure):
        resolve_reliability(output(), forged, diagnostic())


def test_direct_wrapper_changed_root_locator_or_artifact_id_rejects(tmp_path: Path) -> None:
    direct = VerifiedReliabilityMapping(
        ref=MAPPING.ref,
        repository_root=MAPPING.repository_root,
        authority_id=MAPPING.authority_id,
        payload=MAPPING.payload,
        _authority_token=object(),
    )
    attacks = (
        direct,
        replace(MAPPING, repository_root=str(tmp_path)),
        replace(MAPPING, ref=replace(MAPPING.ref, locator="data/lol/v2/evaluation/b2/coverage-procedure.json")),
        replace(MAPPING, ref=replace(MAPPING.ref, artifact_id="scryglass:b2:wrong:v1")),
    )
    for attack in attacks:
        with pytest.raises(ValidationFailure):
            resolve_reliability(output(), attack, diagnostic())


def test_exact_five_positive_controls_replay_and_are_nonpromotable() -> None:
    for diag in DIAGNOSTICS:
        result = resolve_reliability(output(diag.record["output_type"]), MAPPING, diag)
        assert result.status == "ok"
        assert result.label == "high"
        verify_reliability_replay(
            result, output(diag.record["output_type"]), MAPPING, diag
        )
        assert diag.record["synthetic_positive_control"] is True
        assert diag.record["production_eligible"] is False
    assert _verify_b3_authority()["production_authorities"] == []
    report = build_b2_validation_report(REGISTRY)
    assert report["production_eligible"] is False


def test_mapped_missing_diagnostic_is_limited_not_unrated_or_high() -> None:
    result = resolve_reliability(output(), MAPPING, None)
    assert (result.status, result.label, result.reasons) == (
        "unavailable", "limited", ("diagnostic_missing",)
    )


def test_missing_ood_flags_is_missing_provenance_not_known_empty() -> None:
    changed = output()
    changed["context"].pop("ood_flags")
    result = resolve_reliability(changed, MAPPING, diagnostic())
    assert result.label == "limited"
    assert result.reasons == ("context_provenance_missing",)
    assert result.context["ood_state"] == "missing"


@pytest.mark.parametrize(
    "bad_status",
    [pytest.param(None, id="missing"), "unavailable", "error", False, 123],
)
def test_non_ok_output_status_can_never_be_rated_high(bad_status) -> None:
    changed = output()
    if bad_status is None:
        changed.pop("status")
    else:
        changed["status"] = bad_status
    with pytest.raises(
        ValidationFailure,
        match="exact string output status 'ok'",
    ):
        resolve_reliability(changed, MAPPING, diagnostic())


@pytest.mark.parametrize(
    "changes",
    [
        {"effective_resolved_clusters": float("nan")},
        {"effective_resolved_clusters": float("inf")},
        {"effective_resolved_clusters": None},
        {"transform_approved": "true"},
        {"probability_wording_eligible": "yes"},
        {"candidate_id": ""},
        {"candidate_artifact_sha256": "x" * 64},
        {"candidate_log_loss": float("inf")},
        {"baseline_log_loss": -1},
        {"log_loss_skill": 0.2},
        {"candidate_brier": 0.8, "brier_skill": -0.57},
        {"calibration_slope": -1000},
        {"aggregate_coverage": 0},
        {"transform_id": ""},
        {"transform_sha256": "f" * 64},
    ],
)
def test_strict_diagnostic_semantics_reject_pathologies(changes: dict) -> None:
    with pytest.raises(ValidationFailure):
        _validate_diagnostic_record(rehashed_record(**changes))


def test_caller_self_hashed_diagnostic_cannot_authorize_high() -> None:
    forged = replace(
        diagnostic(),
        record=rehashed_record(candidate_log_loss=.40, log_loss_skill=.20),
        _authority_token=object(),
    )
    with pytest.raises(ValidationFailure, match="not loader-issued"):
        resolve_reliability(output(), MAPPING, forged)


@pytest.mark.parametrize(
    "field,value",
    [
        ("output_type", "team_rating"),
        ("stratum_id", "stratum-team"),
        ("transform_id", "identity-v1"),
    ],
)
def test_wrong_output_stratum_or_transform_rejects(field: str, value: str) -> None:
    forged = replace(diagnostic(), record=rehashed_record(**{field: value}))
    with pytest.raises(ValidationFailure):
        resolve_reliability(output(), MAPPING, forged)


@pytest.mark.parametrize(
    "value,accepted",
    [
        ("x", False),
        ("xx", False),
        ("xxx", True),
        (" padded", False),
        ("padded ", False),
        ("x" * 256, True),
        ("x" * 257, False),
    ],
)
def test_stable_id_exact_schema_boundaries(value: str, accepted: bool) -> None:
    if accepted:
        assert _stable_id(value, "probe_id") == value
    else:
        with pytest.raises(ValidationFailure, match="length 3..256"):
            _stable_id(value, "probe_id")


@pytest.mark.parametrize(
    "field,value",
    [
        ("transform_id", "identity-v1"),
        ("transform_kind", "identity"),
        ("transform_sha256", "83f60c22c56f33a67723ece314c1fbaf4ee65f08a71fa53b18f943b8388e7fb0"),
    ],
)
def test_detached_self_rehashed_transform_substitution_rejects(
    field: str, value: str
) -> None:
    forged = replace(diagnostic(), record=rehashed_record(**{field: value}))
    with pytest.raises(ValidationFailure):
        resolve_reliability(output(), MAPPING, forged)


def test_changed_frozen_registry_transform_record_rejects() -> None:
    records = deepcopy(REGISTRY.served_transform_identities)
    records["symmetrized-platt-v1"] = {
        "kind": "identity",
        "sha256": records["identity-v1"]["sha256"],
    }
    with pytest.raises(ValidationFailure, match="exact B1-registrar-authorized"):
        verify_frozen_b2_registry_authority(
            replace(REGISTRY, served_transform_identities=records)
        )


def test_changed_transform_manifest_bytes_reject(tmp_path: Path) -> None:
    target = tmp_path / "data/lol/v2/evaluation"
    shutil.copytree("data/lol/v2/evaluation", target)
    manifest = target / "b2/transform-symmetrized-platt-v1.json"
    payload = json.loads(manifest.read_text())
    payload["kind"] = "identity"
    manifest.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    with pytest.raises(
        ValidationFailure,
        match="bytes, semantics, and registry record are detached",
    ):
        load_verified_diagnostics(tmp_path)


def test_unregistered_context_is_the_only_unrated_path() -> None:
    changed = output()
    changed["context"]["scope_id"] = "invented"
    result = resolve_reliability(changed, MAPPING, diagnostic())
    assert (result.status, result.label) == ("unrated", "unrated")


def test_literal_placeholder_resolution_or_diagnostic_hash_rejects() -> None:
    result = resolve_reliability(output(), MAPPING, diagnostic())
    for forged in (
        replace(result, resolution_sha256="x"),
        replace(result, diagnostic_sha256="x"),
    ):
        with pytest.raises(ValidationFailure):
            verify_reliability_replay(forged, output(), MAPPING, diagnostic())


def test_self_rehashed_gate_evidence_with_literal_hash_rejects() -> None:
    report = build_b2_validation_report(REGISTRY)
    gate = report["gate_evidence"]["reliability_resolution_replayed"]
    gate["predicate_evidence"]["resolution_hashes"][0] = "x"
    gate["evidence_sha256"] = canonical_sha256(
        {key: value for key, value in gate.items() if key != "evidence_sha256"}
    )
    report["report_sha256"] = canonical_sha256(
        {key: value for key, value in report.items() if key != "report_sha256"}
    )
    with pytest.raises(ValidationFailure, match="fresh executable replay"):
        verify_b2_validation_report(report, REGISTRY)


@pytest.mark.parametrize(
    "attack",
    ["duplicate_minimal_diagnostics", "changed_stratum", "reordered_resolutions"],
)
def test_private_gate_helper_rejects_forged_reliability_summary(attack: str) -> None:
    report = build_b2_validation_report(REGISTRY)
    forged = deepcopy(report["reliability"])
    if attack == "duplicate_minimal_diagnostics":
        minimal = {
            "output_type": "draft_score",
            "stratum_id": "stratum-draft",
            "record_id": "forged",
            "transform_id": "symmetrized-platt-v1",
            "record_sha256": "a1" * 32,
            "synthetic_positive_control": True,
            "production_eligible": False,
        }
        forged["diagnostics"] = [deepcopy(minimal) for _ in range(5)]
        forged["resolutions"] = [deepcopy(forged["resolutions"][0]) for _ in range(5)]
    elif attack == "changed_stratum":
        forged["diagnostics"][0]["stratum_id"] = "stratum-draft"
    else:
        forged["resolutions"] = list(reversed(forged["resolutions"]))
    with pytest.raises(
        ValidationFailure,
        match="differs from exact loader-issued replay",
    ):
        _build_gate_evidence(
            artifacts=verify_b2_artifact_refs(REGISTRY),
            reliability_mapping=MAPPING,
            reliability=forged,
            evidence=report["evidence"],
            calibration=report["calibration"],
            coverage=report["coverage"],
        )
