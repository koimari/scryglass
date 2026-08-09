"""Non-authorizing G5 v2 prefit bundle and TRAIN-only diagnostic."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

import numpy as np

from . import (
    contract,
    runner as v1_runner,
    v2_execution_approval,
    v2_math,
    v2_result,
)


ROOT = Path(__file__).resolve().parents[5]
NAMESPACE = ROOT / "data/lol/v2/models/draft-interactions/g5-exploratory"
CONTRACT_LOCATOR = "data/lol/v2/models/draft-interactions/g5-exploratory/v2-contract.json"
CORE_LOCATOR = "data/lol/v2/models/draft-interactions/g5-exploratory/v2-review-core.json"
PENDING_LOCATOR = "data/lol/v2/models/draft-interactions/g5-exploratory/v2-pending-report.json"
EXPECTED_COUNTS = {"TRAIN": 805, "DEVELOPMENT": 214, "VALIDATION": 207}
V1_PRIMITIVES = {
    "locator": "lol_kills/v2/draft/interactions/g5_exploratory/runner.py",
    "raw_sha256": "938c65d7bf6a925edc961d461c7dcdc6db83582c34d314260a66d0300f81102e",
    "trusted_functions": [
        "_load_accepted_g1", "_load_accepted_features", "_load_accepted_clusters",
        "align_inputs", "build_b0_scores", "_design", "score_d1",
        "_measure_invariances", "_prior_aggregate", "validation_bootstrap",
        "_probability_logloss", "_base_evidence",
    ],
}
V1_PROVENANCE = {
    "result_raw_sha256": "52c43445bda3084f43bed835c27284065649760f6e77d3436055fa550f168941",
    "result_artifact_sha256": "88179a5ae3e591bacc1e0e0668b0fddcf096b9dd4b69b5b50410d569970b506f",
    "ledger_raw_sha256": "39731e3c331be279a2b75946d286a3595abaf12786142c9bd2c3fb61cdcd4a7c",
    "completed_entry_sha256": "7df091cdaed28a7980f3a9ac8cd9f73ed38dc66a7d749e1b4c7dd05efffeba65",
    "terminal_state": "EXECUTION_BLOCKED:SOLVER_OR_OBJECTIVE_FAILURE",
    "observed_warning_signature": [
        "runner.py:1049 divide by zero/overflow/invalid in matmul",
        "runner.py:1055 divide by zero/overflow/invalid in matmul",
        "runner.py:1056 divide by zero/overflow/invalid in matmul",
    ],
}
RESEARCH_RECORD = {
    "scispace": {
        "status": "USED_PRIMARY_LITERATURE_SEARCH_ONLY",
        "papers": [
            {
                "title": "Minimization of functions having Lipschitz continuous first partial derivatives",
                "year": 1966,
                "identifier": "doi:10.2140/pjm.1966.16.1",
            },
            {
                "title": "Self-concordant analysis for logistic regression",
                "year": 2010,
                "identifier": "doi:10.1214/09-EJS521",
            },
            {
                "title": "Accurate computation of the log-sum-exp and softmax functions",
                "identifier": "doi:10.1093/imanum/draa038",
            },
            {
                "title": "On the existence of maximum likelihood estimates in logistic regression models",
                "year": 1984,
                "identifier": "doi:10.1093/biomet/71.1.1",
            },
            {
                "title": "Ridge estimators in logistic regression",
                "identifier": "doi:10.2307/2347628",
            },
            {
                "title": "Globally Convergent Newton Methods for Ill-conditioned Generalized Self-concordant Losses",
                "year": 2019,
                "identifier": "arXiv:1907.01771",
            },
            {
                "title": "Newton Method for Sparse Logistic Regression: Quadratic Convergence and Extensive Simulations",
                "year": 2019,
                "identifier": "arXiv:1901.02768",
            },
        ],
        "authority": "METHODOLOGICAL_PRECEDENTS_ONLY_NOT_PERFORMANCE_EVIDENCE",
    },
    "wolfram": {
        "status": "USED_ALGEBRA_ONLY",
        "verified": [
            "X_beta_equals_X_over_s_times_s_beta",
            "quadratic_penalty_equivalence_lambda_over_s_squared",
            "logistic_gradient_identity",
            "logistic_hessian_weight_identity",
        ],
    },
    "academic_writing_toolkit": {
        "status": "USED_CLAIM_CEILING_PROSE_ONLY",
        "issue_count": 0,
        "authority": "PROSE_LOGIC_AUDIT_ONLY_NOT_SCIENTIFIC_EVIDENCE",
    },
}


def _runtime_binding() -> dict[str, Any]:
    numpy_build = getattr(np.__config__, "CONFIG", {})
    thread_variables = {
        name: os.environ.get(name)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "machine": platform.machine(),
        "byteorder": sys.byteorder,
        "numpy_version": importlib.metadata.version("numpy"),
        "scipy_version": importlib.metadata.version("scipy"),
        "numpy_build_sha256": _sha(numpy_build),
        "blas": numpy_build.get("Build Dependencies", {}).get("blas"),
        "lapack": numpy_build.get("Build Dependencies", {}).get("lapack"),
        "thread_environment": thread_variables,
        "algorithmic_determinism": (
            "fixed_initialization_fixed_iteration_order_fixed_Armijo_policy_no_randomness"
        ),
        "trace_bitwise_scope": (
            "ONLY_IDENTICAL_NUMPY_SCIPY_BLAS_LAPACK_ARCHITECTURE_BYTEORDER_"
            "THREAD_ENVIRONMENT_AND_SOLVER_CONFIG"
        ),
        "cross_runtime_claim": "DETERMINISTIC_ALGORITHM_AND_DECLARED_TOLERANCES_ONLY",
        "solver_config_sha256": v2_math.config_hash(),
    }


class V2RunnerError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _raw(path: Path) -> str:
    metadata = os.lstat(path)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise V2RunnerError("unsafe reviewed file")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_contract() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_id": "scryglass:g5-private-development-v2-contract:v1",
        "state": "EXECUTION_READY_BUT_UNAUTHORIZED",
        "versioned_paths": {
            "contract": CONTRACT_LOCATOR,
            "review_core": CORE_LOCATOR,
            "pending": PENDING_LOCATOR,
            "approval": v2_execution_approval.APPROVAL_LOCATOR,
            "ledger": v2_execution_approval.LEDGER_LOCATOR,
            "result": v2_execution_approval.RESULT_LOCATOR,
        },
        "dependency_pins": {
            "G1": contract.G1,
            "G1_features": contract.G1_FEATURES,
            "G2": contract.G2,
            "clusters": contract.CLUSTERS,
            "accepted_v1_orchestration_primitives": V1_PRIMITIVES,
        },
        "scientific_semantics": {
            "B0": "accepted_STATIC_player_model",
            "D1": "same_role_aware_champion_main_effects_and_same_prior_penalty_family",
            "D2": "OMITTED",
            "fold_counts": EXPECTED_COUNTS,
            "development_selection": "select_D1_iff_mean_LL_B0_minus_LL_D1_gt_0_else_B0",
            "validation_winner": "D1_iff_mean_gain_gte_0.005_and_2000_cluster_bootstrap_95pct_LCB_gt_0",
            "final_holdout": False,
        },
        "execution_orchestration": {
            "entrypoint": (
                "lol_kills.v2.draft.interactions.g5_exploratory."
                "v2_runner.execute_real_v2"
            ),
            "approval_required": True,
            "approval_issued": False,
            "ledger": "append_only_STARTED_then_earliest_COMPLETED_no_retries",
            "result": "immutable_v2_path_with_canonical_read_back_validation",
            "uniqueness": "PROCESS_AND_CONTROL_ONLY",
        },
        "numerical_contract": {
            "loss": "label_branch_logaddexp_without_probability_or_logit_clipping",
            "residual": "scipy_expit_minus_label",
            "hessian_weight": "exp(-logaddexp(0,-z)-logaddexp(0,z))",
            "scaling": "TRAIN_only_column_RMS_Xs_equals_X_over_s_gamma_equals_s_beta",
            "penalty_transform": "lambda_scaled_equals_lambda_over_s_squared",
            "centering": "PROHIBITED",
            "solver": "deterministic_damped_Newton_Armijo",
            "config": vars(v2_math.CONFIG),
            "config_sha256": v2_math.config_hash(),
            "cholesky_jitter": 0.0,
            "postfit_checks": {
                "hessian": "finite_symmetric_strictly_PD_unjittered_Cholesky",
                "factorization_residual": "norm_inf(L@L.T-H)_lte_declared_scaled_tolerance",
                "solve_residual": "norm_inf(H@x-b)_lte_declared_scaled_tolerance",
                "covariance": (
                    "finite_symmetric_strictly_PD_nonnegative_diagonal_and_"
                    "quadratic_forms_with_norm_inf(H@C-I)_residual_gate"
                ),
            },
            "stagnation": (
                "after_mandatory_full_accepted_point_reevaluation_step_inf_lte_1e-11_"
                "and_gradient_inf_gt_1e-9_is_typed_STAGNATION"
            ),
            "equations": {
                "loss": "sum_i(logaddexp(0,z_i)-y_i*z_i)+0.5*sum_j(lambda_j*beta_j^2)",
                "gradient": "X.T@(expit(z)-y)+lambda*beta",
                "hessian": "X.T@diag(expit(z)*(1-expit(z)))@X+diag(lambda)",
                "scale_predictor": "X@beta==(X/s)@(s*beta)",
                "scale_penalty": "lambda*beta^2==(lambda/s^2)*(s*beta)^2",
            },
            "failure": v2_math.BLOCKER,
        },
        "v1_failure_provenance_only": V1_PROVENANCE,
        "research_record": RESEARCH_RECORD,
        "claim_ceiling": v2_result.CLAIM_CEILING,
        "claim_limitation": v2_result.LIMITATION,
        "protected_reads": 0,
        "execution_readiness": "EXECUTION_READY_BUT_UNAUTHORIZED",
        "trusted_boundary": {
            "trusted": "reviewed_local_runner_executor_and_exact_reviewed_config",
            "untrusted": "artifacts_data_results_and_paths",
            "limitations": (
                "caller-label injection and generic hostile parser ergonomics are outside "
                "this private non-G9 process/control review"
            ),
        },
    }
    payload["artifact_sha256"] = _sha(payload)
    return payload


def build_review_bundle() -> dict[str, dict[str, Any]]:
    contract.verify_bound_dependencies()
    contract_payload = build_contract()
    subjects = {
        "math": "lol_kills/v2/draft/interactions/g5_exploratory/v2_math.py",
        "runner": "lol_kills/v2/draft/interactions/g5_exploratory/v2_runner.py",
        "result": "lol_kills/v2/draft/interactions/g5_exploratory/v2_result.py",
        "approval": "lol_kills/v2/draft/interactions/g5_exploratory/v2_execution_approval.py",
        "math_test": "tests/model_v2/draft/interactions/test_g5_v2_math.py",
        "contract_test": "tests/model_v2/draft/interactions/test_g5_v2_contract.py",
    }
    core: dict[str, Any] = {
        "schema_id": "scryglass:g5-private-development-v2-review-core:v1",
        "contract_sha256": contract_payload["artifact_sha256"],
        "review_subject_bytes": {
            name: {"locator": locator, "raw_sha256": _raw(ROOT / locator)}
            for name, locator in subjects.items()
        },
        "dependency_pins": contract_payload["dependency_pins"],
        "runtime": _runtime_binding(),
        "approval_contract": {
            "human_root": v2_execution_approval.ROOT_AUTHORITY,
            "scope": v2_execution_approval.SCOPE,
            "approval_locator": v2_execution_approval.APPROVAL_LOCATOR,
            "status": "MISSING_NOT_ISSUED",
        },
        "protected_reads": 0,
        "claim_ceiling": v2_result.CLAIM_CEILING,
    }
    core["artifact_sha256"] = _sha(core)
    pending: dict[str, Any] = {
        "schema_id": "scryglass:g5-private-development-v2-pending-report:v1",
        "state": "EXECUTION_READY_BUT_UNAUTHORIZED",
        "contract_sha256": contract_payload["artifact_sha256"],
        "review_core_sha256": core["artifact_sha256"],
        "protected_reads": 0,
        "real_diagnostic_runs": 0,
        "development_metrics_computed": False,
        "validation_metrics_computed": False,
        "final_holdout_reads": 0,
        "missing": [
            "independent_full_v2_review",
            "canonical_v2_execution_approval",
        ],
        "typed_blocker": "V2_EXECUTION_UNAUTHORIZED",
        "claim_ceiling": v2_result.CLAIM_CEILING,
    }
    pending["artifact_sha256"] = _sha(pending)
    return {
        "v2-contract.json": contract_payload,
        "v2-review-core.json": core,
        "v2-pending-report.json": pending,
    }


def train_only_diagnostic(
    *,
    partition: str,
    design: np.ndarray,
    labels: np.ndarray,
    offsets: np.ndarray,
    precisions: np.ndarray,
) -> dict[str, Any]:
    if partition != "TRAIN":
        raise V2RunnerError("V2_DIAGNOSTIC_REJECTED_NON_TRAIN")
    x, y, o, precision = v2_math.validate_problem(design, labels, offsets, precisions)
    scales = v2_math.train_column_scales(x)
    x_scaled, precision_scaled = v2_math.reparameterize_train(x, precision, scales)
    initial_objective, initial_gradient, _ = v2_math.objective_gradient_hessian(
        np.zeros(x.shape[1]), x_scaled, y, o, precision_scaled
    )
    transform = {
        "partition": "TRAIN",
        "scales_hex": [float(value).hex() for value in scales],
        "precision_scaled_hex": [float(value).hex() for value in precision_scaled],
        "definition": "X_s=X/s;gamma=s*beta;lambda_s=lambda/s^2",
    }
    try:
        fit = v2_math.damped_newton(x_scaled, y, o, precision_scaled)
        state = "TRAIN_PREFIT_DIAGNOSTIC"
        trace_hash = fit["trace_sha256"]
    except v2_math.V2NumericalUnavailable:
        state = v2_math.BLOCKER
        trace_hash = v2_math.sha256([])
    unsigned = {
        "schema_id": v2_result.PREFIT_SCHEMA,
        "state": state,
        "partition": "TRAIN",
        "n": int(x.shape[0]),
        "d": int(x.shape[1]),
        "exposure_range": [float(np.min(np.abs(x))).hex(), float(np.max(np.abs(x))).hex()],
        "scale_range": [float(np.min(scales)).hex(), float(np.max(scales)).hex()],
        "offset_range": [float(np.min(o)).hex(), float(np.max(o)).hex()],
        "initial_objective_hex": initial_objective.hex(),
        "initial_gradient_inf_hex": float(np.max(np.abs(initial_gradient))).hex(),
        "newton_trace_sha256": trace_hash,
        "config_sha256": v2_math.config_hash(),
        "transform_sha256": v2_math.sha256(transform),
        "emits_labels_or_ids": False,
        "selection_metrics": "STRUCTURALLY_PROHIBITED",
    }
    payload = {**unsigned, "artifact_sha256": v2_result.sha256(unsigned)}
    v2_result.validate_prefit_diagnostic(payload)
    return payload


def _verify_v1_primitives() -> None:
    if _raw(ROOT / V1_PRIMITIVES["locator"]) != V1_PRIMITIVES["raw_sha256"]:
        raise V2RunnerError("V2_EXECUTION_BLOCKED:TRUSTED_PRIMITIVE_IDENTITY")


def _fit_d1_v2(
    maps: tuple[Any, ...], labels: Mapping[str, int]
) -> dict[str, Any]:
    train = tuple(item for item in maps if item.fold == "TRAIN")
    keys = {item.map_key for item in train}
    if (
        len(train) != EXPECTED_COUNTS["TRAIN"]
        or set(labels) != keys
        or any(type(value) is not int or value not in (0, 1) for value in labels.values())
    ):
        raise V2RunnerError("V2_EXECUTION_BLOCKED:TRAIN_MEMBERSHIP")
    design, vocabulary, cells = v1_runner._design(train)
    outcomes = np.asarray([labels[item.map_key] for item in train], dtype=float)
    offsets = np.asarray([item.b0_logit_mean for item in train], dtype=float)
    precision = np.asarray(
        [12.5] * len(vocabulary) + [50.0] * len(cells), dtype=float
    )
    scales = v2_math.train_column_scales(design)
    design_scaled, precision_scaled = v2_math.reparameterize_train(
        design, precision, scales
    )
    transform = {
        "partition": "TRAIN",
        "scales_hex": [float(value).hex() for value in scales],
        "precision_scaled_hex": [float(value).hex() for value in precision_scaled],
        "definition": "X_s=X/s;gamma=s*beta;lambda_s=lambda/s^2",
    }
    solved = v2_math.damped_newton(
        design_scaled, outcomes, offsets, precision_scaled
    )
    inverse_scales = 1.0 / scales
    beta = solved["gamma"] * inverse_scales
    covariance = (
        inverse_scales[:, None]
        * solved["covariance_gamma"]
        * inverse_scales[None, :]
    )
    objective, gradient, hessian = v2_math.objective_gradient_hessian(
        solved["gamma"], design_scaled, outcomes, offsets, precision_scaled
    )
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(covariance)):
        raise v2_math.V2NumericalUnavailable(
            f"{v2_math.BLOCKER}:COVARIANCE_NONFINITE"
        )
    return {
        "beta": beta,
        "covariance": covariance,
        "vocabulary": vocabulary,
        "cells": cells,
        "objective": objective,
        "gradient_inf": float(np.max(np.abs(gradient))),
        "hessian": hessian,
        "solver": {
            "iterations": solved["iterations"],
            "function_evaluations": solved["iterations"] + 1,
            "message": solved["status"],
            "trace_sha256": solved["trace_sha256"],
            "config_sha256": solved["config_sha256"],
        },
        "train_scaling": {
            "partition": "TRAIN",
            "scales_sha256": v2_math.sha256(
                [float(value).hex() for value in scales]
            ),
            "transform_sha256": v2_math.sha256(transform),
            "config_sha256": solved["config_sha256"],
            "definition": transform["definition"],
        },
    }


def _v2_evidence_base(**kwargs: Any) -> Any:
    evidence = v1_runner._base_evidence(**kwargs)
    fit = kwargs["fit"]
    return replace(
        evidence,
        solver_diagnostics={
            "status": "CONVERGED",
            "method": "DETERMINISTIC_DAMPED_NEWTON_ARMIJO",
            "iterations": fit["solver"]["iterations"],
            "trace_sha256": fit["solver"]["trace_sha256"],
            "config_sha256": fit["solver"]["config_sha256"],
            "gradient_inf": fit["gradient_inf"],
            "jitter_used": 0.0,
        },
        uncertainty={
            "status": "AVAILABLE",
            "hessian_symmetric": True,
            "hessian_strictly_pd": True,
            "covariance_finite": True,
            "covariance_symmetric": True,
            "covariance_nonnegative_quadratic_forms": True,
            "factorization_residual_pass": True,
            "solve_residual_pass": True,
            "inverse_residual_pass": True,
        },
        invariance_tests={
            "side_swap": evidence.invariance_tests["side_swap"]["status"] == "PASSED",
            "record_order": evidence.invariance_tests["record_order"]["status"] == "PASSED",
            "role_relabel": "NOT_INVARIANT_BY_CONTRACT",
        },
        contribution_reconciliation={
            "status": "PASSED",
            "absolute_tolerance": 1e-12,
            "max_absolute_error": kwargs["reconciliation_error"],
        },
    )


def _compute_evidence_v2(
    aligned: Any,
    *,
    prepared: tuple[tuple[Any, ...], tuple[Any, ...]] | None = None,
) -> tuple[Any, Mapping[str, Any]]:
    """Use accepted semantics around the v2-only TRAIN fit."""
    from lol_kills.v2.ratings.player.model import posterior_predictive_expected_result

    maps, ledgers = (
        v1_runner.build_b0_scores(aligned) if prepared is None else prepared
    )
    if {
        fold: sum(item.fold == fold for item in maps)
        for fold in EXPECTED_COUNTS
    } != EXPECTED_COUNTS:
        raise V2RunnerError("V2_EXECUTION_BLOCKED:FOLD_COVERAGE")
    labels = {
        fold: {item.map_key: item.label for item in ledgers if item.fold == fold}
        for fold in EXPECTED_COUNTS
    }
    fit = _fit_d1_v2(maps, labels["TRAIN"])
    invariance = v1_runner._measure_invariances(maps, fit)
    development_maps = [item for item in maps if item.fold == "DEVELOPMENT"]
    b0_losses: list[float] = []
    d1_losses: list[float] = []
    for item in development_maps:
        scored = v1_runner.score_d1(item, fit)
        label = labels["DEVELOPMENT"][item.map_key]
        probability = posterior_predictive_expected_result(
            item.b0_logit_mean + scored["increment"], item.b0_logit_variance
        )
        b0_losses.append(v1_runner._probability_logloss(label, item.b0_probability))
        d1_losses.append(v1_runner._probability_logloss(label, probability))
    b0_mean = math.fsum(sorted(b0_losses)) / len(b0_losses)
    d1_mean = math.fsum(sorted(d1_losses)) / len(d1_losses)
    development_gain = b0_mean - d1_mean
    locked = "D1" if development_gain > 0.0 else "B0"
    development = {
        "locked_candidate": locked, "map_count": 214, "evaluations": 1,
        "B0_mean_log_loss": b0_mean, "D1_mean_log_loss": d1_mean,
        "mean_LL_B0_minus_LL_D1": development_gain,
    }
    validation_maps = [item for item in maps if item.fold == "VALIDATION"]
    if locked == "B0":
        losses = [
            v1_runner._probability_logloss(
                labels["VALIDATION"][item.map_key], item.b0_probability
            )
            for item in validation_maps
        ]
        mean = math.fsum(sorted(losses)) / len(losses)
        prior, _ = v1_runner._prior_aggregate((), fit)
        evidence = _v2_evidence_base(
            state="NO_INCREMENTAL_DRAFT_WINNER", blocker=None,
            selected_candidate="B0", maps=maps, aligned=aligned, fit=fit,
            development=development,
            validation={
                "locked_candidate": "B0", "map_count": 207, "evaluations": 1,
                "B0_mean_log_loss": mean,
                "locked_candidate_mean_log_loss": mean,
                "mean_LL_B0_minus_LL_locked_candidate": 0.0,
            },
            bootstrap={
                "status": "NOT_RUN_B0_LOCKED", "replicates": 0,
                "base_seed": None, "quantile": None, "lower_bound": None,
                "map_weighted": True,
            },
            prior_summary=prior, invariance=invariance,
            reconciliation_error=0.0,
            score_subject={"status": "WITHHELD_NO_WINNER"},
            winner=None,
        )
        return evidence, fit["train_scaling"]

    deltas: list[tuple[str, str, float]] = []
    scores: list[tuple[Any, Mapping[str, Any]]] = []
    validation_b0: list[float] = []
    validation_d1: list[float] = []
    b0_probabilities: list[float] = []
    d1_probabilities: list[float] = []
    increments: list[float] = []
    max_reconciliation = 0.0
    for item in validation_maps:
        scored = v1_runner.score_d1(item, fit)
        scores.append((item, scored))
        max_reconciliation = max(max_reconciliation, scored["reconciliation_error"])
        label = labels["VALIDATION"][item.map_key]
        probability = posterior_predictive_expected_result(
            item.b0_logit_mean + scored["increment"], item.b0_logit_variance
        )
        loss_b0 = v1_runner._probability_logloss(label, item.b0_probability)
        loss_d1 = v1_runner._probability_logloss(label, probability)
        deltas.append((item.map_key, item.cluster_key, loss_b0 - loss_d1))
        validation_b0.append(loss_b0)
        validation_d1.append(loss_d1)
        b0_probabilities.append(item.b0_probability)
        d1_probabilities.append(probability)
        increments.append(scored["increment"])
    mean_b0 = math.fsum(sorted(validation_b0)) / len(validation_b0)
    mean_d1 = math.fsum(sorted(validation_d1)) / len(validation_d1)
    gain = mean_b0 - mean_d1
    lower, replicates = v1_runner.validation_bootstrap(deltas)
    if replicates.shape != (2000,):
        raise V2RunnerError("V2_EXECUTION_BLOCKED:BOOTSTRAP_COUNT")
    prior, aggregate_variance = v1_runner._prior_aggregate(scores, fit)
    winner = gain >= 0.005 and lower > 0.0
    winner_aggregate = None
    if winner:
        winner_aggregate = v1_runner.WinnerAggregate(
            score_subject={"status": "AVAILABLE"},
            B0_probability=math.fsum(sorted(b0_probabilities)) / 207,
            D1_logit_increment=math.fsum(sorted(increments)) / 207,
            neutral_completed_draft_probability=math.fsum(sorted(d1_probabilities)) / 207,
            probability_increment_over_B0=(
                math.fsum(sorted(d1_probabilities))
                - math.fsum(sorted(b0_probabilities))
            ) / 207,
            D1_conditional_interval={
                "lower": math.fsum(sorted(increments)) / 207
                - 1.959963984540054 * math.sqrt(aggregate_variance),
                "upper": math.fsum(sorted(increments)) / 207
                + 1.959963984540054 * math.sqrt(aggregate_variance),
                "level": 0.95,
                "scale": "conditional_mean_validation_logit_increment",
            },
        )
    evidence = _v2_evidence_base(
        state=(
            "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER"
            if winner else "NO_INCREMENTAL_DRAFT_WINNER"
        ),
        blocker=None, selected_candidate="D1", maps=maps, aligned=aligned, fit=fit,
        development=development,
        validation={
            "locked_candidate": "D1", "map_count": 207, "evaluations": 1,
            "B0_mean_log_loss": mean_b0,
            "locked_candidate_mean_log_loss": mean_d1,
            "mean_LL_B0_minus_LL_locked_candidate": gain,
        },
        bootstrap={
            "status": "COMPLETED", "replicates": 2000,
            "base_seed": 2026073005, "quantile": 0.05,
            "lower_bound": lower, "map_weighted": True,
        },
        prior_summary=prior, invariance=invariance,
        reconciliation_error=max_reconciliation,
        score_subject={"status": "AVAILABLE" if winner else "WITHHELD_NO_WINNER"},
        winner=winner_aggregate,
    )
    return evidence, fit["train_scaling"]


def _load_aligned_v2() -> Any:
    _verify_v1_primitives()
    g1 = v1_runner._load_accepted_g1()
    features, rows = v1_runner._load_accepted_features()
    clusters = v1_runner._load_accepted_clusters()
    return v1_runner.align_inputs(g1, features, rows, clusters)


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _unavailable_scaling(blocker: str) -> dict[str, Any]:
    marker = {"status": "UNAVAILABLE", "blocker": blocker, "partition": "TRAIN"}
    digest = v2_math.sha256(marker)
    return {
        "partition": "TRAIN",
        "scales_sha256": digest,
        "transform_sha256": digest,
        "config_sha256": v2_math.config_hash(),
        "definition": "X_s=X/s;gamma=s*beta;lambda_s=lambda/s^2",
    }


def _result_payload(
    *,
    evidence: Any | None,
    numerical_blocker: str | None,
    membership_hashes: Mapping[str, str],
    scaling: Mapping[str, Any],
    contract_sha256: str,
    review_core_sha256: str,
    approval: Mapping[str, Any],
    run_id: str,
) -> tuple[dict[str, Any], v2_result.ExpectedBinding]:
    expected = v2_result.ExpectedBinding(
        contract_sha256=contract_sha256,
        review_core_sha256=review_core_sha256,
        approval_sha256=approval["approval_sha256"],
        run_id=run_id,
        config_sha256=v2_math.config_hash(),
        transform_sha256=scaling["transform_sha256"],
        scales_sha256=scaling["scales_sha256"],
        membership_hashes=dict(membership_hashes),
        source_pins=v2_result.SOURCE_PINS,
    )
    if evidence is None:
        state = "V2_PREFIT_NUMERICAL_UNAVAILABLE"
        development = {
            "locked_candidate": None, "map_count": 214, "evaluations": 0,
            "B0_mean_log_loss": None, "D1_mean_log_loss": None,
            "mean_LL_B0_minus_LL_D1": None,
        }
        validation = {
            "locked_candidate": None, "map_count": 207, "evaluations": 0,
            "B0_mean_log_loss": None, "locked_candidate_mean_log_loss": None,
            "mean_LL_B0_minus_LL_locked_candidate": None,
        }
        selected = None
        bootstrap = solver = uncertainty = invariance = reconciliation = prior = winner = None
    else:
        state = evidence.state
        numerical_blocker = None
        selected = evidence.selected_candidate
        development = dict(evidence.development_metric)
        validation = dict(evidence.validation_metric)
        bootstrap = dict(evidence.bootstrap)
        solver = dict(evidence.solver_diagnostics)
        uncertainty = dict(evidence.uncertainty)
        invariance = dict(evidence.invariance_tests)
        reconciliation = dict(evidence.contribution_reconciliation)
        prior = {
            **dict(evidence.prior_only_variance_components),
            "coordinate_exposure_witness": [
                dict(item)
                for item in evidence.prior_only_variance_components[
                    "coordinate_exposure_witness"
                ]
            ],
        }
        winner = None
        if evidence.winner is not None:
            winner = {
                "candidate": "D1",
                "validation_threshold": 0.005,
                "bootstrap_lcb_positive": True,
                "B0_probability": evidence.winner.B0_probability,
                "D1_logit_increment": evidence.winner.D1_logit_increment,
                "neutral_completed_draft_probability": (
                    evidence.winner.neutral_completed_draft_probability
                ),
                "probability_increment_over_B0": (
                    evidence.winner.probability_increment_over_B0
                ),
                "D1_conditional_interval": dict(
                    evidence.winner.D1_conditional_interval
                ),
            }
    unsigned = {
        "schema_id": v2_result.REAL_SCHEMA,
        "state": state,
        "blocker": numerical_blocker,
        "selected_candidate": selected,
        "counts": EXPECTED_COUNTS,
        "membership_hashes": dict(membership_hashes),
        "source_pins": v2_result.SOURCE_PINS,
        "development_metric": development,
        "validation_metric": validation,
        "bootstrap": bootstrap,
        "solver_diagnostics": solver,
        "uncertainty": uncertainty,
        "invariance_tests": invariance,
        "contribution_reconciliation": reconciliation,
        "prior_only_variance_components": prior,
        "train_scaling": dict(scaling),
        "execution_binding": {
            "result_locator": v2_result.RESULT_LOCATOR,
            "result_schema": v2_result.REAL_SCHEMA,
            "contract_sha256": contract_sha256,
            "review_core_sha256": review_core_sha256,
            "approval_sha256": approval["approval_sha256"],
            "run_id": run_id,
            "config_sha256": v2_math.config_hash(),
            "transform_sha256": scaling["transform_sha256"],
            "source_pins_sha256": v2_result.sha256(v2_result.SOURCE_PINS),
            "membership_hashes_sha256": v2_result.sha256(membership_hashes),
            "uniqueness_enforcement": "PROCESS_AND_CONTROL_ONLY",
        },
        "winner_evidence": winner,
        "claim_ceiling": v2_result.CLAIM_CEILING,
        "execution_limitation": v2_result.LIMITATION,
        "final_holdout_reads": 0,
    }
    payload = {**unsigned, "artifact_sha256": v2_result.sha256(unsigned)}
    v2_result.validate_real(payload, expected=expected)
    return payload, expected


def execute_real_v2(run_id: str) -> Mapping[str, Any]:
    """Approval-gated v2 orchestration; never called during prefit review."""
    bundle = build_review_bundle()
    contract_sha = bundle["v2-contract.json"]["artifact_sha256"]
    core_sha = bundle["v2-review-core.json"]["artifact_sha256"]
    try:
        approval = v2_execution_approval.load_approval(
            expected_review_core_sha256=core_sha,
            expected_contract_sha256=contract_sha,
            expected_run_id=run_id,
            expected_config_sha256=v2_math.config_hash(),
        )
        ledger = v2_execution_approval.load_ledger()
        ledger_state = v2_execution_approval.validate_ledger_history(
            ledger, approval=approval,
            expected_review_core_sha256=core_sha,
            expected_run_id=run_id,
        )
    except Exception as error:
        raise V2RunnerError("V2_EXECUTION_BLOCKED:APPROVAL_OR_LEDGER") from error
    if ledger_state == "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY":
        raise V2RunnerError("V2_EXECUTION_BLOCKED:INCOMPLETE_NO_RETRY")
    if ledger_state == "COMPLETED_TERMINAL":
        completed = ledger[1]
        try:
            v2_execution_approval.validate_completed_result_from_ledger(
                completed,
                approval=approval,
                expected_contract_sha256=contract_sha,
                expected_review_core_sha256=core_sha,
                expected_run_id=run_id,
            )
        except Exception as error:
            raise V2RunnerError("V2_EXECUTION_BLOCKED:COMPLETED_RESULT") from error
        v2_execution_approval.append_ledger_entry({
            "state": "INVALID_DUPLICATE",
            "approval_id": approval["approval_id"],
            "run_id": run_id,
            "review_core_sha256": core_sha,
            "approval_sha256": approval["approval_sha256"],
            "result_locator": v2_execution_approval.RESULT_LOCATOR,
            "sequence": len(ledger) + 1,
            "authoritative_completed_entry_sha256": completed["entry_sha256"],
            "authoritative_result_artifact_sha256": completed[
                "result_artifact_sha256"
            ],
            "recorded_at": _timestamp(),
        })
        raise V2RunnerError("V2_EXECUTION_BLOCKED:DUPLICATE")
    started = v2_execution_approval.append_ledger_entry({
        "state": "STARTED",
        "approval_id": approval["approval_id"],
        "run_id": run_id,
        "review_core_sha256": core_sha,
        "approval_sha256": approval["approval_sha256"],
        "result_locator": v2_execution_approval.RESULT_LOCATOR,
        "sequence": 1,
        "started_at": _timestamp(),
    })
    # Protected reads begin only after durable STARTED.
    aligned = _load_aligned_v2()
    prepared = v1_runner.build_b0_scores(aligned)
    maps, _ledgers = prepared
    membership_hashes = v1_runner._membership_hashes(maps, aligned)
    try:
        evidence, scaling = _compute_evidence_v2(aligned, prepared=prepared)
        numerical_blocker = None
    except v2_math.V2NumericalUnavailable as error:
        evidence = None
        numerical_blocker = v2_math.closed_result_blocker(error)
        scaling = _unavailable_scaling(numerical_blocker)
    payload, expected = _result_payload(
        evidence=evidence,
        numerical_blocker=numerical_blocker,
        membership_hashes=membership_hashes,
        scaling=scaling,
        contract_sha256=contract_sha,
        review_core_sha256=core_sha,
        approval=approval,
        run_id=run_id,
    )
    result_path = v2_execution_approval._safe_path(
        v2_execution_approval.RESULT_LOCATOR, may_be_missing=True
    )
    data = v2_result.canonical_bytes(payload) + b"\n"
    _write_immutable(result_path, payload)
    completed = v2_execution_approval.append_ledger_entry({
        "state": "COMPLETED",
        "approval_id": approval["approval_id"],
        "run_id": run_id,
        "review_core_sha256": core_sha,
        "approval_sha256": approval["approval_sha256"],
        "result_locator": v2_execution_approval.RESULT_LOCATOR,
        "sequence": 2,
        "started_entry_sha256": started["entry_sha256"],
        "result_artifact_sha256": payload["artifact_sha256"],
        "result_raw_sha256": hashlib.sha256(data).hexdigest(),
        "config_sha256": expected.config_sha256,
        "transform_sha256": expected.transform_sha256,
        "scales_sha256": expected.scales_sha256,
        "membership_hashes_sha256": v2_result.sha256(
            expected.membership_hashes
        ),
        "source_pins_sha256": v2_result.sha256(expected.source_pins),
        "completed_at": max(_timestamp(), started["started_at"]),
    })
    history = v2_execution_approval.load_ledger()
    if v2_execution_approval.validate_ledger_history(
        history, approval=approval,
        expected_review_core_sha256=core_sha,
        expected_run_id=run_id,
    ) != "COMPLETED_TERMINAL":
        raise V2RunnerError("V2_EXECUTION_BLOCKED:LEDGER_FINALIZATION")
    reread = v2_execution_approval.validate_completed_result(
        completed, expected=expected
    )
    if reread != payload:
        raise V2RunnerError("V2_EXECUTION_BLOCKED:RESULT_REREAD")
    return payload


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> None:
    raw = _canonical(payload) + b"\n"
    if path.exists():
        if path.read_bytes() == raw:
            return
        raise V2RunnerError("immutable v2 artifact differs")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def write_review_bundle() -> dict[str, str]:
    bundle = build_review_bundle()
    for name, payload in sorted(bundle.items()):
        _write_immutable(NAMESPACE / name, payload)
    return {name: payload["artifact_sha256"] for name, payload in bundle.items()}
