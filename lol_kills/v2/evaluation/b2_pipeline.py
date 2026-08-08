"""Deterministic PASS-B2 synthetic falsification pipeline and hard gates."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .b2_artifacts import (
    B2_ARTIFACT_IDS,
    verify_b2_artifact_refs,
    verify_frozen_b2_registry_authority,
)
from .calibration import (
    CALIBRATION_FAMILIES,
    apply_registered_transform,
    fit_logistic_calibration,
    select_nested_transform,
)
from .checks import ValidationFailure
from .coverage import aggregate_forecast_coverage, simulation_parameter_coverage
from .evidence import (
    build_measured_selection_report,
    replay_evidence_value,
    verify_measured_selection_report,
)
from .reliability import (
    VerifiedReliabilityMapping,
    audit_mapping_registry,
    load_verified_diagnostics,
    load_verified_mapping,
    resolve_reliability,
    verify_reliability_replay,
)
from .types import CONTRACT_TREE_SHA256, EvaluationRegistry, canonical_sha256, write_json


B2_REQUIRED_HARD_GATES = (
    "b2_artifact_refs_verified",
    "reliability_mapping_total_exact_one",
    "reliability_resolution_replayed",
    "reliability_labels_valid",
    "r20_evidence_recipes_verified",
    "r20_evidence_values_replayed",
    "calibration_selection_nested",
    "calibration_transform_properties_verified",
    "calibration_diagnostic_verified",
    "simulation_coverage_complete",
    "aggregate_coverage_complete",
    "coverage_width_and_wording_consistent",
)
SYNTHETIC_REPORT_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/synthetic-validation-report.json"
)


def _strict_content_hash(value: object) -> bool:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return len(set(value)) > 1


def _output(output_type: str, mode: str, role: str, transform: str, prefix: str) -> dict[str, Any]:
    return {
        "output_type": output_type,
        "mode": mode,
        "status": "ok",
        "context": {
            "scope_id": "synthetic-global",
            "league_id": "synthetic-tier1",
            "role_id": role,
            "patch_relation": "registered",
            "roster_novelty": "known",
            "prefix_slot": prefix,
            "search_policy_id": "exact-v1",
            "transform_id": transform,
            "fallback_profile": "none",
            "ood_flags": [],
        },
    }


def _reliability_evidence(
    mapping: VerifiedReliabilityMapping,
) -> dict[str, Any]:
    audit = audit_mapping_registry(mapping.payload)
    specs = (
        ("player_rating", "terminal", "top", "identity-v1", "terminal"),
        ("team_rating", "terminal", "team", "identity-v1", "terminal"),
        ("draft_score", "terminal", "draft", "symmetrized-platt-v1", "terminal"),
        ("partial_draft_state", "prefix", "draft", "symmetrized-platt-v1", "slot_3"),
        ("tier_list", "terminal", "all", "identity-v1", "terminal"),
    )
    diagnostics = load_verified_diagnostics(mapping.repository_root)
    by_output = {item.record["output_type"]: item for item in diagnostics}
    if set(by_output) != {item[0] for item in specs}:
        raise ValidationFailure("Reliability positive-control output set is not exact")
    resolutions = []
    for spec in specs:
        output = _output(*spec)
        diagnostic = by_output[spec[0]]
        resolution = resolve_reliability(
            output,
            mapping,
            diagnostic,
        )
        verify_reliability_replay(
            resolution, output, mapping, diagnostic
        )
        if resolution.label != "high":
            raise ValidationFailure("synthetic Reliability positive control did not reach high")
        resolutions.append(resolution.unsigned_payload() | {"resolution_sha256": resolution.resolution_sha256})
    real_resolution = resolve_reliability(
        _output(*specs[2]), mapping, None
    )
    if real_resolution.label == "high":
        raise ValidationFailure("real high Reliability is available before B3")
    return {
        "audit": audit,
        "diagnostics": [dict(item.record) for item in diagnostics],
        "diagnostic_artifact_raw_sha256": diagnostics[0].artifact_raw_sha256,
        "diagnostic_artifact_object_sha256": diagnostics[0].artifact_object_sha256,
        "resolutions": resolutions,
        "production_control_label": real_resolution.label,
        "production_control_reasons": list(real_resolution.reasons),
    }


def _evidence_evidence(
    payload: Mapping[str, Any], repo_root: Path
) -> dict[str, Any]:
    results = []
    for recipe in payload["candidates"]:
        results.append(replay_evidence_value(recipe, repo_root=repo_root))
    selection = build_measured_selection_report(payload, repo_root)
    verify_measured_selection_report(selection, payload, repo_root)
    return {"replays": results, "selection": selection}


def _calibration_evidence() -> dict[str, Any]:
    x = [-2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2]
    y = [0, 1, 0, 0, 1, 0, 1, 1, 1]
    oracle = fit_logistic_calibration(x, y, model_sha256="synthetic-oracle")
    if oracle.status != "ok":
        raise ValidationFailure("proper logistic calibration oracle unavailable")
    if abs(oracle.intercept - 0.3075762643523149) > 1e-7 or abs(oracle.slope - 0.9898041959242025) > 1e-7:
        raise ValidationFailure("proper logistic calibration differs from frozen oracle")
    ols = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
    if np.linalg.norm(np.asarray([oracle.intercept, oracle.slope]) - ols) < .1:
        raise ValidationFailure("proper logistic calibration is indistinguishable from OLS sentinel")
    nested_x = [-2.4, -1.7, -1.2, -.8, -.2, .2, .7, 1.1, 1.6, 2.2, 2.5, 2.8]
    nested_y = [0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1]
    nested = select_nested_transform(
        nested_x,
        nested_y,
        [f"s{i//2}" for i in range(len(nested_x))],
        list(range(len(nested_x))),
        [f"r{i}" for i in range(len(nested_x))],
    )
    if nested.status != "ok":
        raise ValidationFailure("nested chronological calibration selection unavailable")
    dense = np.linspace(-100, 100, 2001)
    family_parameters = {
        "identity": {},
        "symmetric_temperature": {"slope": .8},
        "symmetrized_platt": {"intercept": .2, "slope": 1.1},
        "symmetrized_beta": {"intercept": -.1, "a": 1.2, "b": .8},
        "symmetrized_bounded_isotonic": {
            "knots": [-3, -1, 0, 1, 3],
            "levels": [.03, .2, .5, .8, .97],
        },
    }
    properties = {}
    for family in CALIBRATION_FAMILIES:
        values = apply_registered_transform(dense, family, family_parameters[family])
        complement = apply_registered_transform(-dense, family, family_parameters[family])
        passed = (
            np.all(np.isfinite(values))
            and np.all((values > 0) & (values < 1))
            and np.all(np.diff(values) >= -1e-12)
            and np.max(np.abs(values + complement - 1)) <= 1e-12
        )
        if not passed:
            raise ValidationFailure(f"{family} transform property failed")
        properties[family] = True
    return {
        "oracle": {
            "intercept": oracle.intercept,
            "slope": oracle.slope,
            "gradient_inf_norm": oracle.parameters["gradient_inf_norm"],
            "information_eigenvalues": oracle.parameters["information_eigenvalues"],
            "ols_sentinel": [float(v) for v in ols],
        },
        "nested_selection": {
            "family": nested.kind,
            "selection_sha256": nested.selection_sha256,
            "calibration_row_sha256": nested.calibration_row_sha256,
        },
        "properties": properties,
    }


def _coverage_evidence(procedure: Mapping[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(1776)
    cases = []
    for index, truth in enumerate((.25, .4, .55, .7)):
        cases.append({
            "case_id": f"sim-{index}",
            "output_type": "draft_score",
            "parameter": "win_probability",
            "stratum": "stratum-draft",
            "truth": truth,
            "posterior_draws": np.clip(rng.normal(truth, .05, 400), .001, .999).tolist(),
        })
    simulation = simulation_parameter_coverage(
        cases,
        generator_sha256="8" * 64,
        inference_sha256="9" * 64,
        seed=1776,
        artifact_sha256="a" * 64,
    )
    rows = [
        {"row_id": "a1", "series_id": "sa", "outcome": 0},
        {"row_id": "a2", "series_id": "sa", "outcome": 1},
        {"row_id": "b1", "series_id": "sb", "outcome": 1},
        {"row_id": "b2", "series_id": "sb", "outcome": 1},
    ]
    cells = [
        {
            "cell_id": "cell-a", "row_ids": ["a1", "a2"],
            "posterior_predictive_draws": rng.binomial(1, .1, (500, 2)).tolist(),
            "baseline_width": 1.0, "width_margin": 0.0,
            "higher_cluster_support": 1, "resampling_unit": "series",
        },
        {
            "cell_id": "cell-b", "row_ids": ["b1", "b2"],
            "posterior_predictive_draws": rng.binomial(1, .9, (500, 2)).tolist(),
            "baseline_width": 1.0, "width_margin": 0.0,
            "higher_cluster_support": 1, "resampling_unit": "series",
        },
    ]
    aggregate = aggregate_forecast_coverage(
        rows,
        cells,
        dependence_design={"id": procedure["aggregate_forecast_coverage"]["dependence_design"], "provisional": True},
        procedure_sha256=canonical_sha256(procedure),
    )
    return {"simulation": simulation, "aggregate": aggregate, "wording": "95% model range"}


def _gate_evidence(
    name: str, passed: bool, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "gate": name,
        "status": "pass" if passed else "fail",
        "predicate_evidence": dict(evidence),
    }
    return {**payload, "evidence_sha256": canonical_sha256(payload)}


def _verify_reliability_gate_input(
    mapping: VerifiedReliabilityMapping,
    submitted: Mapping[str, Any],
) -> dict[str, Any]:
    """Replay the pinned five-record authority before either Reliability gate."""
    expected = _reliability_evidence(mapping)
    if dict(submitted) != expected:
        raise ValidationFailure(
            "Reliability gate evidence differs from exact loader-issued replay"
        )
    identities = {
        (
            record["output_type"],
            record["stratum_id"],
            record["record_id"],
            record["transform_id"],
        )
        for record in expected["diagnostics"]
    }
    expected_identities = {
        (
            "player_rating",
            "stratum-player",
            "synthetic-reliability-control-player_rating-v1",
            "identity-v1",
        ),
        (
            "team_rating",
            "stratum-team",
            "synthetic-reliability-control-team_rating-v1",
            "identity-v1",
        ),
        (
            "draft_score",
            "stratum-draft",
            "synthetic-reliability-control-draft_score-v1",
            "symmetrized-platt-v1",
        ),
        (
            "partial_draft_state",
            "stratum-prefix",
            "synthetic-reliability-control-partial_draft_state-v1",
            "symmetrized-platt-v1",
        ),
        (
            "tier_list",
            "stratum-tier",
            "synthetic-reliability-control-tier_list-v1",
            "identity-v1",
        ),
    }
    if identities != expected_identities or len(expected["diagnostics"]) != 5:
        raise ValidationFailure("Reliability diagnostic identity set is not exact")
    return expected


def _build_gate_evidence(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    reliability_mapping: VerifiedReliabilityMapping,
    reliability: Mapping[str, Any],
    evidence: Mapping[str, Any],
    calibration: Mapping[str, Any],
    coverage: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    reliability = _verify_reliability_gate_input(
        reliability_mapping, reliability
    )
    resolutions = list(reliability.get("resolutions", ()))
    audit = reliability.get("audit", {})
    properties = calibration.get("properties", {})
    oracle = calibration.get("oracle", {})
    nested = calibration.get("nested_selection", {})
    simulation = coverage.get("simulation", {})
    aggregate = coverage.get("aggregate", {})
    cells = list(aggregate.get("cells", ()))
    evidence_replays = list(evidence.get("replays", ()))
    selection = evidence.get("selection", {})
    predicates: dict[str, tuple[bool, dict[str, Any]]] = {
        "b2_artifact_refs_verified": (
            set(artifacts) == set(B2_ARTIFACT_IDS),
            {
                "artifact_ids": sorted(artifacts),
                "artifact_object_hashes": {
                    artifact_id: canonical_sha256(payload)
                    for artifact_id, payload in sorted(artifacts.items())
                },
            },
        ),
        "reliability_mapping_total_exact_one": (
            audit.get("registered_context_count") == 5
            and len(audit.get("evidence", ())) == 5,
            {"audit_sha256": canonical_sha256(audit)},
        ),
        "reliability_resolution_replayed": (
            len(resolutions) == 5
            and len(reliability.get("diagnostics", ())) == 5
            and _strict_content_hash(
                reliability.get("diagnostic_artifact_raw_sha256")
            )
            and _strict_content_hash(
                reliability.get("diagnostic_artifact_object_sha256")
            )
            and all(
                item.get("status") == "ok"
                and item.get("match_count") == 1
                and _strict_content_hash(item.get("resolution_sha256"))
                and _strict_content_hash(item.get("diagnostic_sha256"))
                for item in resolutions
            ),
            {
                "resolution_hashes": [
                    item.get("resolution_sha256") for item in resolutions
                ],
                "diagnostic_hashes": [
                    item.get("diagnostic_sha256") for item in resolutions
                ],
                "diagnostic_artifact_raw_sha256": reliability.get(
                    "diagnostic_artifact_raw_sha256"
                ),
                "diagnostic_artifact_object_sha256": reliability.get(
                    "diagnostic_artifact_object_sha256"
                ),
            },
        ),
        "reliability_labels_valid": (
            len(resolutions) == 5
            and all(item.get("label") == "high" for item in resolutions)
            and {
                item.get("context", {}).get("output_type") for item in resolutions
            }
            == {
                "player_rating",
                "team_rating",
                "draft_score",
                "partial_draft_state",
                "tier_list",
            }
            and all(
                record.get("synthetic_positive_control") is True
                and record.get("production_eligible") is False
                and _strict_content_hash(record.get("record_sha256"))
                for record in reliability.get("diagnostics", ())
            )
            and reliability.get("production_control_label") == "limited"
            and reliability.get("production_control_reasons")
            == ["diagnostic_missing"],
            {
                "synthetic_labels": [item.get("label") for item in resolutions],
                "production_control_label": reliability.get(
                    "production_control_label"
                ),
            },
        ),
        "r20_evidence_recipes_verified": (
            len(evidence_replays) == 3
            and {item.get("method_id") for item in evidence_replays}
            == {
                "standardized_posterior_mean_displacement",
                "interval_contraction",
                "deterministic_source_context_coverage",
            },
            {
                "recipe_hashes": [
                    item.get("recipe_sha256") for item in evidence_replays
                ],
                "selection_report_sha256": selection.get("report_sha256"),
                "selection_count": len(selection.get("selections", ())),
            },
        ),
        "r20_evidence_values_replayed": (
            len(evidence_replays) == 3
            and all(item.get("result_sha256") for item in evidence_replays)
            and len(selection.get("selections", ())) == 15,
            {
                "result_hashes": [
                    item.get("result_sha256") for item in evidence_replays
                ],
                "selection_report_sha256": selection.get("report_sha256"),
            },
        ),
        "calibration_selection_nested": (
            nested.get("family") in CALIBRATION_FAMILIES
            and bool(nested.get("selection_sha256"))
            and bool(nested.get("calibration_row_sha256")),
            dict(nested),
        ),
        "calibration_transform_properties_verified": (
            set(properties) == set(CALIBRATION_FAMILIES)
            and all(value is True for value in properties.values()),
            {"property_results": dict(properties)},
        ),
        "calibration_diagnostic_verified": (
            abs(float(oracle.get("intercept", float("inf"))) - 0.3075762643523149)
            <= 1e-7
            and abs(float(oracle.get("slope", float("inf"))) - 0.9898041959242025)
            <= 1e-7,
            {
                "intercept": oracle.get("intercept"),
                "slope": oracle.get("slope"),
                "gradient_inf_norm": oracle.get("gradient_inf_norm"),
            },
        ),
        "simulation_coverage_complete": (
            simulation.get("kind") == "simulation_parameter_coverage"
            and bool(simulation.get("evidence"))
            and bool(simulation.get("aggregates")),
            {
                "report_sha256": simulation.get("report_sha256"),
                "case_count": len(simulation.get("evidence", ())),
            },
        ),
        "aggregate_coverage_complete": (
            aggregate.get("kind") == "aggregate_forecast_coverage"
            and bool(cells)
            and all(item.get("row_ids") for item in cells),
            {
                "report_sha256": aggregate.get("report_sha256"),
                "cell_count": len(cells),
            },
        ),
        "coverage_width_and_wording_consistent": (
            coverage.get("wording") == "95% model range"
            and aggregate.get("production_eligible") is False
            and all(float(item.get("width", -1)) >= 0 for item in cells),
            {
                "wording": coverage.get("wording"),
                "widths": [item.get("width") for item in cells],
                "production_eligible": aggregate.get("production_eligible"),
            },
        ),
    }
    if set(predicates) != set(B2_REQUIRED_HARD_GATES):
        raise ValidationFailure("B2 gate predicate set is missing or extra")
    return {
        name: _gate_evidence(name, passed, predicate_evidence)
        for name, (passed, predicate_evidence) in predicates.items()
    }


def build_b2_validation_report(
    registry: EvaluationRegistry,
    repo_root: Path | str = Path("."),
) -> dict[str, Any]:
    root = Path(repo_root)
    authority = verify_frozen_b2_registry_authority(registry, root)
    artifacts = verify_b2_artifact_refs(registry, root)
    refs = {ref.artifact_id: ref for ref in registry.b2_artifact_refs}
    reliability_mapping = load_verified_mapping(
        refs[B2_ARTIFACT_IDS[0]], root
    )
    reliability = _reliability_evidence(reliability_mapping)
    evidence = _evidence_evidence(artifacts[B2_ARTIFACT_IDS[1]], root)
    calibration = _calibration_evidence()
    coverage = _coverage_evidence(artifacts[B2_ARTIFACT_IDS[3]])
    gate_evidence = _build_gate_evidence(
        artifacts=artifacts,
        reliability_mapping=reliability_mapping,
        reliability=reliability,
        evidence=evidence,
        calibration=calibration,
        coverage=coverage,
    )
    gates = {
        name: item["status"] == "pass" for name, item in gate_evidence.items()
    }
    report = {
        "artifact_kind": "scryglass-l2-pass-b2-synthetic-validation",
        "artifact_version": "1",
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "synthetic_only": True,
        "production_eligible": False,
        "production_unavailable_reasons": [
            "missing_real_heldout_b2_diagnostics",
            "missing_b3_resolved_cluster_dependence_evidence",
            "missing_l4_l9_model_authorities",
        ],
        "artifact_refs": [ref.to_payload() for ref in registry.b2_artifact_refs],
        "lineage": {
            **authority,
            "split_plan_id": registry.split_plan_id,
            "split_plan_sha256": registry.split_plan_sha256,
            "source_snapshot_id": registry.source_snapshot_id,
            "source_snapshot_sha256": registry.source_snapshot_sha256,
            "training_snapshot_id": registry.training_snapshot_id,
            "training_snapshot_sha256": registry.training_snapshot_sha256,
            "source_tree_sha256": registry.source_tree_sha256,
            "contract_tree_sha256": registry.contract_tree_sha256,
        },
        "reliability": reliability,
        "evidence": evidence,
        "calibration": calibration,
        "coverage": coverage,
        "gate_evidence": gate_evidence,
        "hard_gates": gates,
        "method_notes": [
            {
                "doi": "10.1093/biomet/asac068",
                "claim": "This source motivates uncertainty-aware calibration assessment; this checkpoint only enforces typed unavailable calibration and does not implement its honest confidence bands.",
            },
            {
                "doi": "10.1073/PNAS.2016191118",
                "claim": "This source motivates reproducible reliability diagnostics; this checkpoint implements content-addressed replay and does not claim to implement the CORP estimator.",
            },
        ],
    }
    report["report_sha256"] = canonical_sha256(report)
    return report


def verify_b2_validation_report(
    report: Mapping[str, Any],
    registry: EvaluationRegistry,
    repo_root: Path | str = Path("."),
) -> None:
    submitted = dict(report)
    report_hash = submitted.pop("report_sha256", None)
    if report_hash != canonical_sha256(submitted):
        raise ValidationFailure("B2 validation report content hash is invalid")
    fresh = build_b2_validation_report(registry, repo_root)
    if dict(report) != fresh:
        raise ValidationFailure("B2 validation report does not match fresh executable replay")
    if (
        set(report.get("hard_gates", {})) != set(B2_REQUIRED_HARD_GATES)
        or set(report.get("gate_evidence", {})) != set(B2_REQUIRED_HARD_GATES)
        or not all(report["hard_gates"].values())
    ):
        raise ValidationFailure("B2 hard gates are missing, extra, or failed")
    for name, evidence in report["gate_evidence"].items():
        submitted_evidence = dict(evidence)
        evidence_hash = submitted_evidence.pop("evidence_sha256", None)
        if (
            evidence_hash != canonical_sha256(submitted_evidence)
            or evidence.get("gate") != name
            or evidence.get("status") != "pass"
            or report["hard_gates"].get(name) is not True
        ):
            raise ValidationFailure("B2 gate evidence is false, detached, or malformed")


def write_b2_validation_report(
    registry: EvaluationRegistry,
    locator: Path | str = SYNTHETIC_REPORT_LOCATOR,
    repo_root: Path | str = Path("."),
) -> str:
    report = build_b2_validation_report(registry, repo_root)
    write_json(Path(locator), report)
    return str(report["report_sha256"])
