"""Strict, aggregate-only result schemas for the private G5 runner."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

from . import contract


SYNTHETIC_SCHEMA = "scryglass:g5-synthetic-execution:v1"
REAL_SCHEMA = "scryglass:g5-private-exploratory-execution:v1"
REAL_STATES = {
    "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER",
    "NO_INCREMENTAL_DRAFT_WINNER",
    "EXECUTION_BLOCKED",
}
WINNER_ONLY_FIELDS = {
    "private_retrospective_exploratory_score_probability",
    "fit_evidence",
    "rank_selection_evidence",
    "B0_probability",
    "D1_logit_increment",
    "neutral_completed_draft_probability",
    "probability_increment_over_B0",
    "D1_conditional_interval",
}
REAL_BASE_FIELDS = {
    "schema_version",
    "state",
    "blocker",
    "selected_candidate",
    "counts",
    "membership_hashes",
    "source_and_feature_review_pins",
    "G2_core_pins",
    "development_metric",
    "validation_metric",
    "bootstrap",
    "objective_gradient_hessian_diagnostics",
    "solver_diagnostics",
    "uncertainty",
    "prior_only_variance_components",
    "coverage_and_prior_only_flags",
    "invariance_tests",
    "contribution_reconciliation",
    "score_subject",
    "context",
    "execution_binding",
    "execution_limitation",
    "claim_ceiling",
    "artifact_sha256",
}
CLAIM_CEILING = {
    "private_retrospective_exploratory": True,
    "prediction": False,
    "forecast": False,
    "publication": False,
    "promotion": False,
    "sota": False,
    "reliability": False,
    "current": False,
    "live": False,
    "calibrated_production": False,
    "public_pack": False,
    "final_holdout": False,
}
BLOCKER_CODES = frozenset({
    "EXECUTION_BLOCKED:APPROVAL_INVALID",
    "EXECUTION_BLOCKED:B0_LATER_STATE_UPDATE",
    "EXECUTION_BLOCKED:B0_PRIMITIVE_RECONCILIATION",
    "EXECUTION_BLOCKED:B0_TRAIN_CHRONOLOGY",
    "EXECUTION_BLOCKED:B0_TRAIN_ORIGIN_FOLD_LEAKAGE",
    "EXECUTION_BLOCKED:B0_TRAIN_ORIGIN_NOT_ELIGIBLE_FOR_LATER_TARGET",
    "EXECUTION_BLOCKED:B0_UNCERTAINTY_UNAVAILABLE",
    "EXECUTION_BLOCKED:BOOTSTRAP_FAILURE",
    "EXECUTION_BLOCKED:BOOTSTRAP_MEMBERSHIP",
    "EXECUTION_BLOCKED:CHAMPION_ABSENT_FROM_TRAIN",
    "EXECUTION_BLOCKED:CLUSTER_ASSIGNMENT_UNAVAILABLE",
    "EXECUTION_BLOCKED:CLUSTER_IDENTITY_MISMATCH",
    "EXECUTION_BLOCKED:CLUSTER_MEMBERSHIP_ALIGNMENT",
    "EXECUTION_BLOCKED:CONDITIONAL_COVARIANCE_UNAVAILABLE",
    "EXECUTION_BLOCKED:CONTRIBUTION_RECONCILIATION_FAILURE",
    "EXECUTION_BLOCKED:FEATURE_IDENTITY_MISMATCH",
    "EXECUTION_BLOCKED:FEATURE_MEMBERSHIP_ORIGIN_DIGEST_MISMATCH",
    "EXECUTION_BLOCKED:FOLD_COVERAGE",
    "EXECUTION_BLOCKED:FOLD_TIME_OR_FEATURE_ALIGNMENT",
    "EXECUTION_BLOCKED:G1_IDENTITY_MISMATCH",
    "EXECUTION_BLOCKED:G1_MEMBERSHIP_ORIGIN_DIGEST_MISMATCH",
    "EXECUTION_BLOCKED:INVARIANCE_FAILURE",
    "EXECUTION_BLOCKED:MAP_MEMBERSHIP_ALIGNMENT",
    "EXECUTION_BLOCKED:INVALID_DUPLICATE",
    "EXECUTION_BLOCKED:LEDGER_INVALID",
    "EXECUTION_BLOCKED:NONFINITE_EVALUATION",
    "EXECUTION_BLOCKED:PICK_IDENTITY_ALIGNMENT",
    "EXECUTION_BLOCKED:PRIOR_ONLY_ROLE_DELTA_DUPLICATED",
    "EXECUTION_BLOCKED:RUN_ID_INVALID",
    "EXECUTION_BLOCKED:SOLVER_OR_OBJECTIVE_FAILURE",
    "EXECUTION_BLOCKED:UNEXPECTED_PIPELINE_FAILURE",
})
EXPECTED_FOLD_COUNTS = {"TRAIN": 805, "DEVELOPMENT": 214, "VALIDATION": 207}


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("result is not canonical JSON") from error


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} exact field set mismatch")
    return value


def _finite(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{label} must be finite")


def _sha(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be lowercase sha256")


def validate_synthetic(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "state",
        "development",
        "validation",
        "solver",
        "claim_ceiling",
        "artifact_sha256",
    }
    _exact(value, expected, "synthetic result")
    unsigned = dict(value)
    claimed = unsigned.pop("artifact_sha256")
    if value.get("schema_version") != SYNTHETIC_SCHEMA or claimed != sha256(unsigned):
        raise ValueError("synthetic result identity/schema mismatch")
    if value.get("state") not in {
        "SYNTHETIC_SELECTION_D1",
        "SYNTHETIC_SELECTION_B0",
        "SYNTHETIC_EXECUTION_BLOCKED",
    }:
        raise ValueError("synthetic result state invalid")
    if value.get("claim_ceiling") != {
        "synthetic_only": True,
        "real_evidence": False,
        "publication": False,
        "prediction": False,
    }:
        raise ValueError("synthetic promotion boundary")
    if any(token in str(value) for token in REAL_STATES):
        raise ValueError("synthetic promotion boundary")


def _validate_real_nested(value: Mapping[str, Any]) -> None:
    counts = _exact(
        value["counts"],
        {"maps", "picks", "TRAIN", "DEVELOPMENT", "VALIDATION"},
        "counts",
    )
    if any(type(counts[key]) is not int or counts[key] < 0 for key in counts):
        raise ValueError("counts must be nonnegative integers")
    if counts["maps"] != counts["TRAIN"] + counts["DEVELOPMENT"] + counts["VALIDATION"]:
        raise ValueError("map count reconciliation failed")
    if counts["picks"] != counts["maps"] * 10:
        raise ValueError("pick count reconciliation failed")
    if value["state"] != "EXECUTION_BLOCKED":
        if (counts["maps"], counts["picks"]) != (1226, 12260):
            raise ValueError("executed result coverage must be exact")
        if {fold: counts[fold] for fold in EXPECTED_FOLD_COUNTS} != EXPECTED_FOLD_COUNTS:
            raise ValueError("executed result fold coverage must be exact")

    hashes = _exact(
        value["membership_hashes"],
        {
            "all_maps_sha256",
            "TRAIN_maps_sha256",
            "DEVELOPMENT_maps_sha256",
            "VALIDATION_maps_sha256",
            "cluster_membership_sha256",
            "origin_membership_sha256",
            "feature_membership_sha256",
        },
        "membership_hashes",
    )
    for key, digest in hashes.items():
        _sha(digest, f"membership_hashes.{key}")

    pins = _exact(
        value["source_and_feature_review_pins"],
        {
            "G1_manifest_sha256",
            "G1_rows_sha256",
            "selected_target_sha256",
            "split_payload_sha256",
            "feature_manifest_sha256",
            "feature_rows_raw_sha256",
            "feature_rows_canonical_sha256",
            "feature_review_sha256",
            "cluster_artifact_sha256",
        },
        "source_and_feature_review_pins",
    )
    for key, digest in pins.items():
        _sha(digest, f"source_and_feature_review_pins.{key}")
    expected_pins = {
        "G1_manifest_sha256": contract.G1["manifest_sha256"],
        "G1_rows_sha256": contract.G1["rows_sha256"],
        "selected_target_sha256": contract.G1["selected_target_sha256"],
        "split_payload_sha256": contract.G1["split_payload_sha256"],
        "feature_manifest_sha256": contract.G1_FEATURES["manifest_canonical_sha256"],
        "feature_rows_raw_sha256": contract.G1_FEATURES["rows_raw_sha256"],
        "feature_rows_canonical_sha256": contract.G1_FEATURES["rows_canonical_sha256"],
        "feature_review_sha256": contract.G1_FEATURES["independent_review_canonical_sha256"],
        "cluster_artifact_sha256": contract.CLUSTERS["artifact_canonical_sha256"],
    }
    if dict(pins) != expected_pins:
        raise ValueError("frozen source/feature pins mismatch")

    g2 = _exact(
        value["G2_core_pins"],
        {"runner_raw_sha256", "model_raw_sha256", "artifact_raw_sha256", "artifact_canonical_sha256", "candidate"},
        "G2_core_pins",
    )
    for key in ("runner_raw_sha256", "model_raw_sha256", "artifact_raw_sha256", "artifact_canonical_sha256"):
        _sha(g2[key], f"G2_core_pins.{key}")
    if g2["candidate"] != "static_baseline":
        raise ValueError("G2 candidate mismatch")
    if dict(g2) != {
        "runner_raw_sha256": contract.G2["runner_raw_sha256"],
        "model_raw_sha256": contract.G2["model_raw_sha256"],
        "artifact_raw_sha256": contract.G2["artifact_raw_sha256"],
        "artifact_canonical_sha256": contract.G2["artifact_canonical_sha256"],
        "candidate": "static_baseline",
    }:
        raise ValueError("frozen G2 pins mismatch")

    development = _exact(
        value["development_metric"],
        {
            "locked_candidate",
            "map_count",
            "evaluations",
            "B0_mean_log_loss",
            "D1_mean_log_loss",
            "mean_LL_B0_minus_LL_D1",
        },
        "development_metric",
    )
    validation = _exact(
        value["validation_metric"],
        {
            "locked_candidate",
            "map_count",
            "evaluations",
            "B0_mean_log_loss",
            "locked_candidate_mean_log_loss",
            "mean_LL_B0_minus_LL_locked_candidate",
        },
        "validation_metric",
    )
    for key, metric in (("development_metric", development), ("validation_metric", validation)):
        if metric["locked_candidate"] not in {"B0", "D1", None}:
            raise ValueError(f"{key} candidate invalid")
        if type(metric["map_count"]) is not int or metric["map_count"] < 0:
            raise ValueError(f"{key} map count invalid")
        if type(metric["evaluations"]) is not int or metric["evaluations"] not in {0, 1}:
            raise ValueError(f"{key} evaluation count invalid")
        for metric_key, metric_value in metric.items():
            if "loss" in metric_key or metric_key.startswith("mean_LL"):
                _finite(metric_value, f"{key}.{metric_key}", nullable=True)

    bootstrap = _exact(
        value["bootstrap"],
        {"status", "replicates", "base_seed", "quantile", "lower_bound", "map_weighted"},
        "bootstrap",
    )
    if bootstrap["status"] not in {"NOT_RUN_B0_LOCKED", "COMPLETED", "BLOCKED"}:
        raise ValueError("bootstrap status invalid")
    if type(bootstrap["replicates"]) is not int or bootstrap["replicates"] not in {0, 2000}:
        raise ValueError("bootstrap replicate count invalid")
    if bootstrap["base_seed"] not in {None, 2026073005} or bootstrap["quantile"] not in {None, 0.05}:
        raise ValueError("bootstrap protocol invalid")
    _finite(bootstrap["lower_bound"], "bootstrap lower bound", nullable=True)
    if type(bootstrap["map_weighted"]) is not bool:
        raise ValueError("bootstrap weighting flag invalid")

    objective = _exact(
        value["objective_gradient_hessian_diagnostics"],
        {
            "objective",
            "gradient_infinity_norm",
            "hessian_dimension",
            "hessian_symmetric_atol_1e_12",
            "hessian_positive_definite",
        },
        "objective_gradient_hessian_diagnostics",
    )
    _finite(objective["objective"], "objective", nullable=True)
    _finite(objective["gradient_infinity_norm"], "gradient norm", nullable=True)
    if type(objective["hessian_dimension"]) is not int or objective["hessian_dimension"] < 0:
        raise ValueError("Hessian dimension invalid")
    if any(type(objective[key]) is not bool for key in ("hessian_symmetric_atol_1e_12", "hessian_positive_definite")):
        raise ValueError("Hessian diagnostics invalid")

    solver = _exact(
        value["solver_diagnostics"],
        {"status", "method", "analytic_jacobian", "iterations", "function_evaluations", "message_sha256"},
        "solver_diagnostics",
    )
    if solver["status"] not in {"CONVERGED", "BLOCKED", "NOT_RUN"} or solver["method"] != "L-BFGS-B" or solver["analytic_jacobian"] is not True:
        raise ValueError("solver diagnostics invalid")
    for key in ("iterations", "function_evaluations"):
        if type(solver[key]) is not int or solver[key] < 0:
            raise ValueError("solver count invalid")
    _sha(solver["message_sha256"], "solver message")

    uncertainty = _exact(
        value["uncertainty"],
        {"B0_latent_mean_available", "B0_latent_variance_available", "D1_conditional_covariance", "total_B0_plus_D1_interval"},
        "uncertainty",
    )
    if any(type(uncertainty[key]) is not bool for key in ("B0_latent_mean_available", "B0_latent_variance_available")):
        raise ValueError("B0 uncertainty flags invalid")
    if uncertainty["D1_conditional_covariance"] not in {"AVAILABLE", "UNAVAILABLE", "NOT_RUN"}:
        raise ValueError("D1 covariance status invalid")
    if uncertainty["total_B0_plus_D1_interval"] != "PROHIBITED":
        raise ValueError("total interval must be prohibited")

    prior = _exact(
        value["prior_only_variance_components"],
        {
            "status",
            "role_delta_count",
            "variance_per_coordinate",
            "total_variance",
            "mean_score_aggregate_variance",
            "conditional_mean_logit_variance",
            "slot_membership_sha256",
            "signed_exposure_sha256",
            "coordinate_exposure_witness",
        },
        "prior_only_variance_components",
    )
    if prior["status"] not in {"EVALUATED", "NOT_EVALUATED", "BLOCKED"}:
        raise ValueError("prior-only status invalid")
    if type(prior["role_delta_count"]) is not int or prior["role_delta_count"] < 0:
        raise ValueError("prior-only count invalid")
    if prior["variance_per_coordinate"] != 0.01:
        raise ValueError("prior-only variance invalid")
    _finite(prior["total_variance"], "prior-only total variance")
    _finite(prior["mean_score_aggregate_variance"], "prior-only mean aggregate variance")
    _finite(prior["conditional_mean_logit_variance"], "conditional mean logit variance")
    if prior["total_variance"] != 0.01 * prior["role_delta_count"]:
        raise ValueError("prior-only total variance does not reconcile")
    if prior["mean_score_aggregate_variance"] < 0.0:
        raise ValueError("prior-only aggregate variance invalid")
    if prior["conditional_mean_logit_variance"] < prior["mean_score_aggregate_variance"]:
        raise ValueError("conditional mean variance cannot omit prior-only component")
    _sha(prior["slot_membership_sha256"], "prior-only membership")
    _sha(prior["signed_exposure_sha256"], "prior-only signed exposure")
    witness = prior["coordinate_exposure_witness"]
    if not isinstance(witness, list):
        raise ValueError("prior-only coordinate exposure witness must be a list")
    commitments: list[str] = []
    signed_records: list[list[Any]] = []
    for record in witness:
        record = _exact(
            record,
            {
                "coordinate_commitment_sha256",
                "blue_count",
                "red_count",
                "net_count",
                "validation_map_count",
            },
            "prior-only coordinate exposure record",
        )
        commitment = record["coordinate_commitment_sha256"]
        _sha(commitment, "prior-only coordinate commitment")
        for key in ("blue_count", "red_count", "net_count", "validation_map_count"):
            if type(record[key]) is not int:
                raise ValueError("prior-only exposure counts must be exact integers")
        blue = record["blue_count"]
        red = record["red_count"]
        net = record["net_count"]
        validation_count = record["validation_map_count"]
        if (
            blue < 0
            or red < 0
            or blue + red < 1
            or validation_count != EXPECTED_FOLD_COUNTS["VALIDATION"]
            or blue + red > validation_count
            or net != blue - red
        ):
            raise ValueError("prior-only coordinate exposure arithmetic invalid")
        commitments.append(commitment)
        signed_records.append([commitment, blue, red, net, validation_count])
    if commitments != sorted(commitments) or len(commitments) != len(set(commitments)):
        raise ValueError("prior-only coordinate commitments must be unique and sorted")
    expected_mean_prior_variance = 0.01 * math.fsum(
        (record[3] / record[4]) ** 2 for record in signed_records
    )
    if (
        prior["role_delta_count"] != len(commitments)
        or prior["total_variance"] != 0.01 * len(commitments)
        or prior["slot_membership_sha256"] != sha256(commitments)
        or prior["signed_exposure_sha256"] != sha256(signed_records)
        or prior["mean_score_aggregate_variance"] != expected_mean_prior_variance
    ):
        raise ValueError("prior-only coordinate witness does not reconcile")

    coverage = _exact(
        value["coverage_and_prior_only_flags"],
        {
            "complete_maps",
            "complete_picks",
            "champion_absent_from_TRAIN",
            "prior_only_role_delta_used",
            "final_holdout_reads",
        },
        "coverage_and_prior_only_flags",
    )
    if any(type(coverage[key]) is not bool for key in ("complete_maps", "complete_picks", "champion_absent_from_TRAIN", "prior_only_role_delta_used")):
        raise ValueError("coverage flags invalid")
    if coverage["final_holdout_reads"] != 0:
        raise ValueError("final holdout boundary violated")

    invariance = _exact(
        value["invariance_tests"],
        {"side_swap", "record_order", "role_relabel"},
        "invariance_tests",
    )
    for name in ("side_swap", "record_order"):
        check = _exact(
            invariance[name],
            {"status", "map_count", "absolute_tolerance", "max_absolute_error"},
            f"invariance.{name}",
        )
        if check["status"] not in {"PASSED", "NOT_RUN", "BLOCKED"}:
            raise ValueError("invariance status invalid")
        if type(check["map_count"]) is not int or check["map_count"] < 0 or check["absolute_tolerance"] != 1e-12:
            raise ValueError("invariance protocol invalid")
        _finite(check["max_absolute_error"], "invariance error", nullable=True)
    if invariance["role_relabel"] != {"status": "NOT_INVARIANT_BY_CONTRACT"}:
        raise ValueError("role relabel contract invalid")
    reconciliation = _exact(
        value["contribution_reconciliation"],
        {"status", "absolute_tolerance", "max_absolute_error"},
        "contribution_reconciliation",
    )
    if reconciliation["status"] not in {"PASSED", "NOT_RUN", "BLOCKED"} or reconciliation["absolute_tolerance"] != 1e-12:
        raise ValueError("contribution reconciliation invalid")
    _finite(reconciliation["max_absolute_error"], "reconciliation error", nullable=True)

    subject = _exact(
        value["score_subject"],
        {"status", "kind", "fold", "map_count", "weighting", "order_invariant"},
        "score_subject",
    )
    if subject["status"] not in {"AVAILABLE", "WITHHELD_NO_WINNER", "NOT_RUN"}:
        raise ValueError("score subject status invalid")
    if subject["kind"] != "VALIDATION_COHORT_AGGREGATE" or subject["fold"] != "VALIDATION" or subject["weighting"] != "MAP_EQUAL" or subject["order_invariant"] is not True:
        raise ValueError("score subject definition invalid")
    if type(subject["map_count"]) is not int or subject["map_count"] < 0:
        raise ValueError("score subject map count invalid")

    context = _exact(value["context"], {"status", "blocker"}, "context")
    if context != {
        "status": "UNAVAILABLE",
        "blocker": "CONTEXTUAL_EXACT_FIVE_OR_PLAYER_CHAMPION_EVIDENCE_UNAVAILABLE",
    }:
        raise ValueError("context authority mismatch")
    binding = _exact(
        value["execution_binding"],
        {
            "run_id_sha256", "runner_core_sha256", "approval_sha256",
            "started_entry_sha256", "result_locator", "uniqueness_enforcement",
        },
        "execution_binding",
    )
    for key in ("run_id_sha256", "runner_core_sha256", "approval_sha256", "started_entry_sha256"):
        _sha(binding[key], f"execution_binding.{key}")
    if binding["result_locator"] != "data/lol/v2/models/draft-interactions/g5-exploratory/execution-result.json":
        raise ValueError("result locator mismatch")
    if binding["uniqueness_enforcement"] != "PROCESS_AND_CONTROL_ONLY":
        raise ValueError("execution uniqueness boundary mismatch")
    if value["execution_limitation"] != (
        "Run uniqueness is process/control enforced only and provides no G9, public, "
        "concurrent, or adversarial single-use authority."
    ):
        raise ValueError("execution limitation mismatch")
    if value["claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("claim ceiling mismatch")


def validate_real(value: Mapping[str, Any]) -> None:
    """Validate the complete immutable real result, including state rules."""

    state = value.get("state")
    if state not in REAL_STATES:
        raise ValueError("real state invalid")
    expected = REAL_BASE_FIELDS | (WINNER_ONLY_FIELDS if state == "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER" else set())
    _exact(value, expected, "real result")
    if value.get("schema_version") != REAL_SCHEMA:
        raise ValueError("real schema mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("artifact_sha256")
    if claimed != sha256(unsigned):
        raise ValueError("real result self hash mismatch")
    if state == "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER":
        if value.get("blocker") is not None or value.get("selected_candidate") != "D1":
            raise ValueError("winner state selection mismatch")
        for key in ("private_retrospective_exploratory_score_probability", "B0_probability", "neutral_completed_draft_probability"):
            _finite(value[key], key)
            if not 0.0 < float(value[key]) < 1.0:
                raise ValueError(f"{key} must be an open probability")
        for key in ("D1_logit_increment", "probability_increment_over_B0"):
            _finite(value[key], key)
        interval = _exact(value["D1_conditional_interval"], {"lower", "upper", "level", "scale"}, "D1 interval")
        _finite(interval["lower"], "D1 interval lower")
        _finite(interval["upper"], "D1 interval upper")
        if interval["lower"] > interval["upper"] or interval["level"] != 0.95 or interval["scale"] != "conditional_mean_validation_logit_increment":
            raise ValueError("D1 interval invalid")
        if value["fit_evidence"] != "TRAIN_ONLY" or value["rank_selection_evidence"] != "DEVELOPMENT_LOCKED_VALIDATION_GATED":
            raise ValueError("winner evidence boundary mismatch")
    elif state == "NO_INCREMENTAL_DRAFT_WINNER":
        if value.get("blocker") is not None or value.get("selected_candidate") not in {"B0", "D1"}:
            raise ValueError("no-winner state mismatch")
    else:
        if value.get("blocker") not in BLOCKER_CODES or value.get("selected_candidate") is not None:
            raise ValueError("blocked state mismatch")
    _validate_real_nested(value)
    _validate_state_semantics(value)
    _reject_identity_leaks(value)


def _validate_state_semantics(value: Mapping[str, Any]) -> None:
    state = value["state"]
    development = value["development_metric"]
    validation = value["validation_metric"]
    bootstrap = value["bootstrap"]
    solver = value["solver_diagnostics"]
    objective = value["objective_gradient_hessian_diagnostics"]
    uncertainty = value["uncertainty"]
    coverage = value["coverage_and_prior_only_flags"]
    reconciliation = value["contribution_reconciliation"]
    invariance = value["invariance_tests"]
    prior = value["prior_only_variance_components"]
    subject = value["score_subject"]
    if prior["status"] in {"NOT_EVALUATED", "BLOCKED"} and prior != {
        "status": prior["status"],
        "role_delta_count": 0,
        "variance_per_coordinate": 0.01,
        "total_variance": 0.0,
        "mean_score_aggregate_variance": 0.0,
        "conditional_mean_logit_variance": 0.0,
        "slot_membership_sha256": sha256([]),
        "signed_exposure_sha256": sha256([]),
        "coordinate_exposure_witness": [],
    }:
        raise ValueError("unevaluated prior-only ledger must be exactly empty")
    if state == "EXECUTION_BLOCKED":
        if solver["status"] not in {"BLOCKED", "NOT_RUN"} or subject["status"] != "NOT_RUN":
            raise ValueError("blocked execution diagnostics overclaim")
        if any(invariance[name]["status"] == "PASSED" for name in ("side_swap", "record_order")):
            raise ValueError("blocked invariance may not pass")
        return

    if (
        solver["status"] != "CONVERGED"
        or uncertainty["D1_conditional_covariance"] != "AVAILABLE"
        or not coverage["complete_maps"]
        or not coverage["complete_picks"]
        or reconciliation["status"] != "PASSED"
        or not objective["hessian_symmetric_atol_1e_12"]
        or not objective["hessian_positive_definite"]
        or any(invariance[name]["status"] != "PASSED" for name in ("side_swap", "record_order"))
        or any(invariance[name]["map_count"] != 1226 for name in ("side_swap", "record_order"))
    ):
        raise ValueError("executed state requires successful diagnostics")
    if (
        objective["gradient_infinity_norm"] is None
        or objective["gradient_infinity_norm"] > 1e-6
        or objective["hessian_dimension"] <= 0
        or reconciliation["max_absolute_error"] is None
        or reconciliation["max_absolute_error"] > 1e-12
        or any(invariance[name]["max_absolute_error"] is None or invariance[name]["max_absolute_error"] > 1e-12 for name in ("side_swap", "record_order"))
        or not uncertainty["B0_latent_mean_available"]
        or not uncertainty["B0_latent_variance_available"]
        or coverage["champion_absent_from_TRAIN"]
        or coverage["prior_only_role_delta_used"] != bool(prior["role_delta_count"])
    ):
        raise ValueError("executed numerical diagnostics do not reconcile")
    if development["map_count"] != EXPECTED_FOLD_COUNTS["DEVELOPMENT"] or development["evaluations"] != 1:
        raise ValueError("development aggregate pass mismatch")
    if validation["map_count"] != EXPECTED_FOLD_COUNTS["VALIDATION"] or validation["evaluations"] != 1:
        raise ValueError("validation aggregate pass mismatch")
    development_delta = development["B0_mean_log_loss"] - development["D1_mean_log_loss"]
    if abs(development_delta - development["mean_LL_B0_minus_LL_D1"]) > 1e-15:
        raise ValueError("development loss reconciliation failed")
    expected_lock = "D1" if development_delta > 0.0 else "B0"
    if development["locked_candidate"] != expected_lock or value["selected_candidate"] != expected_lock:
        raise ValueError("development lock mismatch")
    validation_delta = validation["B0_mean_log_loss"] - validation["locked_candidate_mean_log_loss"]
    if abs(validation_delta - validation["mean_LL_B0_minus_LL_locked_candidate"]) > 1e-15:
        raise ValueError("validation loss reconciliation failed")
    if validation["locked_candidate"] != expected_lock:
        raise ValueError("validation locked candidate mismatch")
    if expected_lock == "B0":
        if (
            state != "NO_INCREMENTAL_DRAFT_WINNER"
            or validation_delta != 0.0
            or bootstrap != {
                "status": "NOT_RUN_B0_LOCKED",
                "replicates": 0,
                "base_seed": None,
                "quantile": None,
                "lower_bound": None,
                "map_weighted": True,
            }
            or prior["status"] != "NOT_EVALUATED"
            or prior["role_delta_count"] != 0
        ):
            raise ValueError("B0 lock semantics invalid")
    else:
        if (
            bootstrap["status"] != "COMPLETED"
            or bootstrap["replicates"] != 2000
            or bootstrap["base_seed"] != 2026073005
            or bootstrap["quantile"] != 0.05
            or bootstrap["lower_bound"] is None
            or bootstrap["map_weighted"] is not True
            or prior["status"] != "EVALUATED"
        ):
            raise ValueError("D1 validation/bootstrap semantics invalid")
        empirical_winner = validation_delta >= 0.005 and bootstrap["lower_bound"] > 0.0
        if (state == "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER") != empirical_winner:
            raise ValueError("winner gate semantics invalid")
    expected_subject = "AVAILABLE" if state == "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER" else "WITHHELD_NO_WINNER"
    if subject["status"] != expected_subject or subject["map_count"] != EXPECTED_FOLD_COUNTS["VALIDATION"]:
        raise ValueError("score subject state mismatch")
    if state == "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER":
        if abs(value["neutral_completed_draft_probability"] - value["B0_probability"] - value["probability_increment_over_B0"]) > 1e-15:
            raise ValueError("winner probability algebra failed")
        if value["private_retrospective_exploratory_score_probability"] != value["neutral_completed_draft_probability"]:
            raise ValueError("winner score probability mismatch")
        interval = value["D1_conditional_interval"]
        if abs((interval["lower"] + interval["upper"]) / 2.0 - value["D1_logit_increment"]) > 1e-15:
            raise ValueError("winner conditional interval center mismatch")
        expected_half_width = 1.959963984540054 * math.sqrt(prior["conditional_mean_logit_variance"])
        if abs((interval["upper"] - interval["lower"]) / 2.0 - expected_half_width) > 1e-15:
            raise ValueError("winner conditional interval variance mismatch")


def validate_real_shape(value: Mapping[str, Any]) -> None:
    """Compatibility name retained, now enforcing the complete strict schema."""

    validate_real(value)


def _reject_identity_leaks(value: Any) -> None:
    forbidden_keys = {
        "game_id",
        "source_game_id",
        "player_id",
        "source_player_id",
        "team_id",
        "champion_id",
        "stable_champion_id",
        "cluster_id",
        "dependence_cluster_id",
        "lineup",
        "lineups",
        "row",
        "rows",
    }
    if isinstance(value, Mapping):
        if forbidden_keys & set(value):
            raise ValueError("real aggregate leaks row identity")
        for child in value.values():
            _reject_identity_leaks(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _reject_identity_leaks(child)
    elif isinstance(value, str):
        lowered = value.casefold()
        if any(
            token in lowered
            for token in (
                "source_game_id",
                "game_id=",
                "player_id",
                "team_id",
                "champion_id",
                "cluster_id",
                "lineup=",
                "map-000",
            )
        ):
            raise ValueError("real aggregate leaks identity-bearing string")
