"""Approval-context-bound result schema for the G5 v2 private-development run."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

from . import contract, result as v1_result, v2_math


REAL_SCHEMA = "scryglass:g5-private-development-v2-result:v2"
PREFIT_SCHEMA = "scryglass:g5-private-development-v2-prefit-diagnostic:v1"
CLAIM_CEILING = v1_result.CLAIM_CEILING
RESULT_LOCATOR = (
    "data/lol/v2/models/draft-interactions/g5-exploratory/v2-execution-result.json"
)
LIMITATION = (
    "Private developmental model-fit and rank-selection evidence only; reviewed local "
    "runner and exact solver config are trusted. Process/control uniqueness does not "
    "authorize G9, adversarial concurrency, final-holdout, prediction, publication, "
    "promotion, reliability, current, live, or production claims."
)
EXPECTED_COUNTS = {"TRAIN": 805, "DEVELOPMENT": 214, "VALIDATION": 207}
NUMERICAL_BLOCKERS = frozenset({
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:NONFINITE",
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:FACTORIZATION",
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:SOLVE_RESIDUAL",
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:COVARIANCE",
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:STAGNATION",
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:ARMIJO_EXHAUSTED",
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:MAX_ITERATIONS",
    "V2_PREFIT_NUMERICAL_UNAVAILABLE:CONFIGURATION",
})
SOURCE_PINS = {
    "G1": contract.G1,
    "G1_features": contract.G1_FEATURES,
    "G2": contract.G2,
    "clusters": contract.CLUSTERS,
}


@dataclass(frozen=True)
class ExpectedBinding:
    contract_sha256: str
    review_core_sha256: str
    approval_sha256: str
    run_id: str
    config_sha256: str
    transform_sha256: str
    scales_sha256: str
    membership_hashes: Mapping[str, str]
    source_pins: Mapping[str, Any]


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        raise ValueError(f"{label} exact field set mismatch")
    return value


def _sha(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase sha256")


def _finite(value: Any, label: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be finite")
    try:
        finite = math.isfinite(float(value))
    except (OverflowError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite") from error
    if not finite:
        raise ValueError(f"{label} must be finite")


def _self_hash(value: Mapping[str, Any], schema: str) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop("artifact_sha256", None)
    if value.get("schema_id") != schema or claimed != sha256(unsigned):
        raise ValueError("schema or self hash mismatch")


def validate_prefit_diagnostic(value: Mapping[str, Any]) -> None:
    expected = {
        "schema_id", "state", "partition", "n", "d", "exposure_range",
        "scale_range", "offset_range", "initial_objective_hex",
        "initial_gradient_inf_hex", "newton_trace_sha256", "config_sha256",
        "transform_sha256", "emits_labels_or_ids", "selection_metrics",
        "artifact_sha256",
    }
    _exact(value, expected, "prefit diagnostic")
    _self_hash(value, PREFIT_SCHEMA)
    if (
        value["state"] not in {"TRAIN_PREFIT_DIAGNOSTIC", "V2_PREFIT_NUMERICAL_UNAVAILABLE"}
        or value["partition"] != "TRAIN"
        or type(value["n"]) is not int
        or type(value["d"]) is not int
        or value["n"] <= 0
        or value["d"] <= 0
        or value["emits_labels_or_ids"] is not False
        or value["selection_metrics"] != "STRUCTURALLY_PROHIBITED"
    ):
        raise ValueError("prefit diagnostic authority boundary")
    for key in ("newton_trace_sha256", "config_sha256", "transform_sha256"):
        _sha(value[key], key)


def _validate_prior(value: Any) -> None:
    prior = _exact(
        value,
        {
            "status", "role_delta_count", "variance_per_coordinate",
            "total_variance", "mean_score_aggregate_variance",
            "conditional_mean_logit_variance", "slot_membership_sha256",
            "signed_exposure_sha256", "coordinate_exposure_witness",
        },
        "prior-only variance components",
    )
    if prior["status"] not in {"EVALUATED", "NOT_EVALUATED"}:
        raise ValueError("prior-only status invalid")
    if type(prior["role_delta_count"]) is not int or prior["role_delta_count"] < 0:
        raise ValueError("prior-only count invalid")
    if prior["variance_per_coordinate"] != 0.01:
        raise ValueError("prior-only marginal variance invalid")
    for key in (
        "total_variance", "mean_score_aggregate_variance",
        "conditional_mean_logit_variance",
    ):
        _finite(prior[key], f"prior.{key}")
    witness = prior["coordinate_exposure_witness"]
    if not isinstance(witness, list):
        raise ValueError("prior-only witness must be a list")
    commitments: list[str] = []
    signed: list[list[Any]] = []
    for item in witness:
        item = _exact(
            item,
            {
                "coordinate_commitment_sha256", "blue_count", "red_count",
                "net_count", "validation_map_count",
            },
            "prior-only coordinate",
        )
        commitment = item["coordinate_commitment_sha256"]
        _sha(commitment, "prior coordinate commitment")
        for key in ("blue_count", "red_count", "net_count", "validation_map_count"):
            if type(item[key]) is not int:
                raise ValueError("prior exposure counts must be exact integers")
        blue, red, net, count = (
            item["blue_count"], item["red_count"], item["net_count"],
            item["validation_map_count"],
        )
        if (
            blue < 0 or red < 0 or blue + red < 1
            or count != EXPECTED_COUNTS["VALIDATION"]
            or blue + red > count or net != blue - red
        ):
            raise ValueError("prior exposure arithmetic invalid")
        commitments.append(commitment)
        signed.append([commitment, blue, red, net, count])
    if commitments != sorted(commitments) or len(commitments) != len(set(commitments)):
        raise ValueError("prior coordinate commitments must be unique and sorted")
    mean_variance = 0.01 * math.fsum((row[3] / row[4]) ** 2 for row in signed)
    if (
        prior["role_delta_count"] != len(commitments)
        or prior["total_variance"] != 0.01 * len(commitments)
        or prior["mean_score_aggregate_variance"] != mean_variance
        or prior["conditional_mean_logit_variance"] < mean_variance
        or prior["slot_membership_sha256"] != sha256(commitments)
        or prior["signed_exposure_sha256"] != sha256(signed)
    ):
        raise ValueError("prior witness does not reconcile")
    if prior["status"] == "NOT_EVALUATED" and (
        commitments
        or prior["conditional_mean_logit_variance"] != 0.0
        or prior["mean_score_aggregate_variance"] != 0.0
    ):
        raise ValueError("not-evaluated prior witness must be empty")


def validate_real(value: Mapping[str, Any], *, expected: ExpectedBinding) -> None:
    fields = {
        "schema_id", "state", "blocker", "selected_candidate", "counts",
        "membership_hashes", "source_pins", "development_metric",
        "validation_metric", "bootstrap", "solver_diagnostics", "uncertainty",
        "invariance_tests", "contribution_reconciliation",
        "prior_only_variance_components", "train_scaling", "execution_binding",
        "winner_evidence", "claim_ceiling", "execution_limitation",
        "final_holdout_reads", "artifact_sha256",
    }
    _exact(value, fields, "v2 real result")
    _self_hash(value, REAL_SCHEMA)
    if expected.config_sha256 != v2_math.config_hash():
        raise ValueError("trusted context config mismatch")
    if value["claim_ceiling"] != CLAIM_CEILING or value["execution_limitation"] != LIMITATION:
        raise ValueError("claim ceiling mismatch")
    if value["final_holdout_reads"] != 0 or value["counts"] != EXPECTED_COUNTS:
        raise ValueError("fold/final-holdout boundary mismatch")
    if value["membership_hashes"] != expected.membership_hashes:
        raise ValueError("membership binding mismatch")
    if value["source_pins"] != expected.source_pins or value["source_pins"] != SOURCE_PINS:
        raise ValueError("source pin binding mismatch")
    scaling = _exact(
        value["train_scaling"],
        {
            "partition", "scales_sha256", "transform_sha256", "config_sha256",
            "definition",
        },
        "train scaling",
    )
    if scaling != {
        "partition": "TRAIN",
        "scales_sha256": expected.scales_sha256,
        "transform_sha256": expected.transform_sha256,
        "config_sha256": expected.config_sha256,
        "definition": "X_s=X/s;gamma=s*beta;lambda_s=lambda/s^2",
    }:
        raise ValueError("train scaling binding mismatch")
    binding = _exact(
        value["execution_binding"],
        {
            "result_locator", "result_schema", "contract_sha256",
            "review_core_sha256", "approval_sha256", "run_id",
            "config_sha256", "transform_sha256", "source_pins_sha256",
            "membership_hashes_sha256", "uniqueness_enforcement",
        },
        "execution binding",
    )
    if binding != {
        "result_locator": RESULT_LOCATOR,
        "result_schema": REAL_SCHEMA,
        "contract_sha256": expected.contract_sha256,
        "review_core_sha256": expected.review_core_sha256,
        "approval_sha256": expected.approval_sha256,
        "run_id": expected.run_id,
        "config_sha256": expected.config_sha256,
        "transform_sha256": expected.transform_sha256,
        "source_pins_sha256": sha256(expected.source_pins),
        "membership_hashes_sha256": sha256(expected.membership_hashes),
        "uniqueness_enforcement": "PROCESS_AND_CONTROL_ONLY",
    }:
        raise ValueError("approval-bound execution identity mismatch")
    state = value["state"]
    if state not in {
        "V2_PREFIT_NUMERICAL_UNAVAILABLE",
        "NO_INCREMENTAL_DRAFT_WINNER",
        "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER",
    }:
        raise ValueError("state invalid")
    development = _exact(
        value["development_metric"],
        {
            "locked_candidate", "map_count", "evaluations",
            "B0_mean_log_loss", "D1_mean_log_loss",
            "mean_LL_B0_minus_LL_D1",
        },
        "development metric",
    )
    validation = _exact(
        value["validation_metric"],
        {
            "locked_candidate", "map_count", "evaluations",
            "B0_mean_log_loss", "locked_candidate_mean_log_loss",
            "mean_LL_B0_minus_LL_locked_candidate",
        },
        "validation metric",
    )
    if state == "V2_PREFIT_NUMERICAL_UNAVAILABLE":
        if value["blocker"] not in NUMERICAL_BLOCKERS or value["selected_candidate"] is not None:
            raise ValueError("unavailable blocker/candidate invalid")
        if development != {
            "locked_candidate": None, "map_count": 214, "evaluations": 0,
            "B0_mean_log_loss": None, "D1_mean_log_loss": None,
            "mean_LL_B0_minus_LL_D1": None,
        } or validation != {
            "locked_candidate": None, "map_count": 207, "evaluations": 0,
            "B0_mean_log_loss": None, "locked_candidate_mean_log_loss": None,
            "mean_LL_B0_minus_LL_locked_candidate": None,
        }:
            raise ValueError("unavailable result contains selection evidence")
        for key in (
            "bootstrap", "solver_diagnostics", "uncertainty", "invariance_tests",
            "contribution_reconciliation", "prior_only_variance_components",
            "winner_evidence",
        ):
            if value[key] is not None:
                raise ValueError("unavailable result contains favorable evidence")
        return
    if value["blocker"] is not None or value["selected_candidate"] not in {"B0", "D1"}:
        raise ValueError("evaluated candidate/blocker invalid")
    selected = value["selected_candidate"]
    for metric, count in ((development, 214), (validation, 207)):
        if metric["locked_candidate"] != selected or metric["map_count"] != count or metric["evaluations"] != 1:
            raise ValueError("selection coverage/lock mismatch")
        for key, item in metric.items():
            if "loss" in key or key.startswith("mean_LL"):
                _finite(item, f"metric.{key}")
    gain_dev = development["mean_LL_B0_minus_LL_D1"]
    if (selected == "D1") != (gain_dev > 0.0):
        raise ValueError("development lock rule mismatch")
    if development["B0_mean_log_loss"] - development["D1_mean_log_loss"] != gain_dev:
        raise ValueError("development loss reconciliation mismatch")
    gain_val = validation["mean_LL_B0_minus_LL_locked_candidate"]
    if validation["B0_mean_log_loss"] - validation["locked_candidate_mean_log_loss"] != gain_val:
        raise ValueError("validation loss reconciliation mismatch")
    bootstrap = _exact(
        value["bootstrap"],
        {
            "status", "replicates", "base_seed", "quantile",
            "lower_bound", "map_weighted",
        },
        "bootstrap",
    )
    if selected == "B0":
        if gain_val != 0.0 or bootstrap != {
            "status": "NOT_RUN_B0_LOCKED", "replicates": 0, "base_seed": None,
            "quantile": None, "lower_bound": None, "map_weighted": True,
        }:
            raise ValueError("B0 validation semantics mismatch")
    else:
        _finite(bootstrap["lower_bound"], "bootstrap lower bound")
        if (
            bootstrap["status"] != "COMPLETED"
            or bootstrap["replicates"] != 2000
            or bootstrap["base_seed"] != 2026073005
            or bootstrap["quantile"] != 0.05
            or bootstrap["map_weighted"] is not True
        ):
            raise ValueError("D1 bootstrap semantics mismatch")
    solver = _exact(
        value["solver_diagnostics"],
        {
            "status", "method", "iterations", "trace_sha256",
            "config_sha256", "gradient_inf", "jitter_used",
        },
        "solver diagnostics",
    )
    if (
        solver["status"] != "CONVERGED"
        or solver["method"] != "DETERMINISTIC_DAMPED_NEWTON_ARMIJO"
        or solver["config_sha256"] != expected.config_sha256
        or solver["jitter_used"] != 0.0
        or type(solver["iterations"]) is not int
        or solver["iterations"] < 0
    ):
        raise ValueError("solver gate mismatch")
    _sha(solver["trace_sha256"], "solver trace")
    _finite(solver["gradient_inf"], "solver gradient")
    if value["uncertainty"] != {
        "status": "AVAILABLE", "hessian_symmetric": True,
        "hessian_strictly_pd": True, "covariance_finite": True,
        "covariance_symmetric": True,
        "covariance_nonnegative_quadratic_forms": True,
        "factorization_residual_pass": True, "solve_residual_pass": True,
        "inverse_residual_pass": True,
    }:
        raise ValueError("uncertainty gate mismatch")
    if value["invariance_tests"] != {
        "side_swap": True, "record_order": True, "role_relabel": "NOT_INVARIANT_BY_CONTRACT"
    }:
        raise ValueError("invariance gate mismatch")
    reconciliation = _exact(
        value["contribution_reconciliation"],
        {"status", "absolute_tolerance", "max_absolute_error"},
        "reconciliation",
    )
    _finite(reconciliation["max_absolute_error"], "reconciliation error")
    if (
        reconciliation["status"] != "PASSED"
        or reconciliation["absolute_tolerance"] != 1e-12
        or not 0.0 <= reconciliation["max_absolute_error"] <= 1e-12
    ):
        raise ValueError("reconciliation gate mismatch")
    _validate_prior(value["prior_only_variance_components"])
    lower = bootstrap["lower_bound"] if selected == "D1" else None
    winner = selected == "D1" and gain_val >= 0.005 and lower is not None and lower > 0.0
    if state == "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER":
        evidence = _exact(
            value["winner_evidence"],
            {
                "candidate", "validation_threshold", "bootstrap_lcb_positive",
                "B0_probability", "D1_logit_increment",
                "neutral_completed_draft_probability",
                "probability_increment_over_B0", "D1_conditional_interval",
            },
            "winner evidence",
        )
        if not winner or evidence["candidate"] != "D1" or evidence["validation_threshold"] != 0.005 or evidence["bootstrap_lcb_positive"] is not True:
            raise ValueError("winner gate mismatch")
        for key in (
            "B0_probability", "D1_logit_increment",
            "neutral_completed_draft_probability", "probability_increment_over_B0",
        ):
            _finite(evidence[key], f"winner.{key}")
        if not 0.0 < evidence["B0_probability"] < 1.0:
            raise ValueError("winner.B0_probability must be a probability")
        if not 0.0 < evidence["neutral_completed_draft_probability"] < 1.0:
            raise ValueError("winner.neutral_completed_draft_probability must be a probability")
        if not -1.0 < evidence["probability_increment_over_B0"] < 1.0:
            raise ValueError("winner.probability_increment_over_B0 out of range")
        interval = _exact(
            evidence["D1_conditional_interval"],
            {"lower", "upper", "level", "scale"},
            "winner conditional interval",
        )
        _finite(interval["lower"], "winner interval lower")
        _finite(interval["upper"], "winner interval upper")
        if (
            interval["level"] != 0.95
            or interval["scale"] != "conditional_mean_validation_logit_increment"
            or interval["lower"] > interval["upper"]
        ):
            raise ValueError("winner conditional interval mismatch")
    elif state == "NO_INCREMENTAL_DRAFT_WINNER":
        if winner or value["winner_evidence"] is not None:
            raise ValueError("no-winner favorable fallback")
