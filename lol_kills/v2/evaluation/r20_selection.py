"""Chronological synthetic-development selection for the six R-20 candidates.

This namespace deliberately does not consume the 2B1 balanced fixture target.
It issues its own stochastic Bernoulli predictive rows and can establish only
development-selection mechanics, never production predictive evidence.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import types
import weakref
from typing import Any, Mapping, Sequence

import numpy as np
import scipy

from .checks import ValidationFailure
from .r20_foundation_algorithms import (
    METHOD_SPECS as FOUNDATION_METHOD_SPECS,
    replay_foundation_method,
)
from .types import CONTRACT_TREE_SHA256, canonical_json, canonical_sha256


_SCIPY_MINIMIZE = scipy.optimize.minimize


CONFIG_LOCATOR = Path("data/lol/v2/evaluation/b2/r20-selection-config.json")
ROWS_LOCATOR = Path("data/lol/v2/evaluation/b2/r20-selection-predictive-rows.json")
REPORT_LOCATOR = Path("data/lol/v2/evaluation/b2/r20-selection-report.json")
AUTHORITY_LOCATOR = Path("data/lol/v2/evaluation/b2/r20-selection-authority.json")
FOUNDATION_AUTHORITY_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/r20-foundation-authority.json"
)
FOUNDATION_CANDIDATES_LOCATOR = Path(
    "data/lol/v2/evaluation/b2/r20-foundation-evidence-candidate-registry.json"
)

OUTPUT_STRATA = (
    ("player_rating", "stratum-player"),
    ("team_rating", "stratum-team"),
    ("draft_score", "stratum-draft"),
    ("partial_draft_state", "stratum-prefix"),
    ("tier_list", "stratum-tier"),
)
CANDIDATES = (
    ("posterior_mean_displacement_v1", "posterior_information"),
    ("posterior_median_displacement_v1", "posterior_information"),
    ("central_interval_contraction_v2", "precision"),
    ("robust_mad_contraction_v1", "precision"),
    ("source_context_strict_v2", "source_context_coverage"),
    ("source_context_typed_partial_v1", "source_context_coverage"),
)
FAMILY_TIE_ORDER = {
    "posterior_information": (
        "posterior_mean_displacement_v1",
        "posterior_median_displacement_v1",
    ),
    "precision": (
        "central_interval_contraction_v2",
        "robust_mad_contraction_v1",
    ),
    "source_context_coverage": (
        "source_context_strict_v2",
        "source_context_typed_partial_v1",
    ),
}
HARD_GATES = (
    "authority_and_source_closure",
    "proper_predictive_target",
    "foundation_fixture_rejected",
    "chronology_series_atomicity_time_safe",
    "development_only_no_sealed_labels",
    "training_only_preprocessing",
    "exact_volume_baseline",
    "incremental_rank_condition_nonseparability",
    "paired_proper_score_reconciliation",
    "dependence_unavailable_fail_closed",
    "complete_candidate_universe",
    "frozen_family_local_tie_rule",
    "reliability_separate_no_universal_scalar",
    "synthetic_nonpromotion",
)
_FORBIDDEN_CLAIM_KEYS = {
    "confidence",
    "evidence_confidence",
    "correctness",
    "reliability",
    "sota",
    "pass_b2",
    "c1",
    "production_selected",
}
_MAPS_PER_SERIES = 2
_SERIES_PER_CELL = 90
_FOLDS = ((30, 30, 50), (50, 50, 70), (70, 70, 90))
_MIN_EFFECTIVE_SUPPORT = 30.0
_BOOTSTRAP_SEED = 20260731
_BOOTSTRAP_REPLICATES = 2000
_TIE_MARGIN = 0.002
_CONDITION_BOUND = 1.0e8
_RIDGE = 1.0e-6
_CONTROL_SEEDS = tuple(range(20260801, 20260865))
_SOURCE_FILES = (
    Path("lol_kills/v2/evaluation/r20_selection.py"),
    Path("lol_kills/v2/evaluation/generate_r20_selection_artifacts.py"),
    Path("lol_kills/v2/evaluation/r20_foundation_algorithms.py"),
    Path("lol_kills/v2/evaluation/checks.py"),
    Path("lol_kills/v2/evaluation/types.py"),
)


def _fail(message: str) -> None:
    raise ValidationFailure(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_hash(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value.lower() != value
        or len(set(value)) <= 1
    ):
        _fail(f"{name} is not a content hash")
    try:
        int(value, 16)
    except ValueError:
        _fail(f"{name} is not a content hash")
    return value


def _safe_file(root: Path, locator: object) -> Path:
    if not isinstance(locator, str):
        _fail("artifact locator must be a string")
    relative = Path(locator)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        _fail("artifact locator is unsafe")
    cursor = root.resolve()
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail("artifact locator contains a symlink")
    resolved = cursor.resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        _fail("artifact locator escapes repository")
    if not resolved.is_file():
        _fail("artifact locator is missing")
    return resolved


def _canonical_payload(raw: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{name} is not JSON") from exc
    if not isinstance(payload, dict):
        _fail(f"{name} must be an object")
    if raw != (canonical_json(payload) + "\n").encode():
        _fail(f"{name} bytes are not canonical")
    return payload


def _ref(locator: Path, payload: Mapping[str, Any], raw: bytes) -> dict[str, str]:
    return {
        "artifact_id": str(payload["artifact_id"]),
        "locator": locator.as_posix(),
        "raw_sha256": _sha(raw),
        "canonical_payload_sha256": canonical_sha256(payload),
    }


def _read_ref(root: Path, ref: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if set(ref) != {
        "artifact_id",
        "locator",
        "raw_sha256",
        "canonical_payload_sha256",
    }:
        _fail("artifact reference shape is missing or extra")
    path = _safe_file(root, ref["locator"])
    raw = path.read_bytes()
    payload = _canonical_payload(raw, str(ref["artifact_id"]))
    if payload.get("artifact_id") != ref["artifact_id"]:
        _fail("artifact identity mismatch")
    if _sha(raw) != _strict_hash(ref["raw_sha256"], "raw_sha256"):
        _fail("artifact raw hash mismatch")
    if canonical_sha256(payload) != _strict_hash(
        ref["canonical_payload_sha256"], "canonical_payload_sha256"
    ):
        _fail("artifact object hash mismatch")
    return payload, raw


def _source_closure(root: Path) -> dict[str, str]:
    return {
        path.stem: _sha(_safe_file(root, path.as_posix()).read_bytes())
        for path in _SOURCE_FILES
    }


def _runtime() -> dict[str, str]:
    return {
        "python_implementation": platform.python_implementation(),
        "python_major_minor": f"{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}",
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "bit_generator": "PCG64",
    }


def build_selection_config(repo_root: Path | str = Path(".")) -> dict[str, Any]:
    root = Path(repo_root)
    foundation_authority_raw = _safe_file(
        root, FOUNDATION_AUTHORITY_LOCATOR.as_posix()
    ).read_bytes()
    candidate_raw = _safe_file(root, FOUNDATION_CANDIDATES_LOCATOR.as_posix()).read_bytes()
    candidate_registry = _canonical_payload(
        candidate_raw, "foundation candidate registry"
    )
    candidate_execution = {
        item["method_id"]: {
            "family": item["family"],
            "units": item["units"],
            "boundaries": item["boundaries"],
            "boundary_sha256": item["boundary_sha256"],
            "code_sha256": item["code_sha256"],
            "implementation": item["implementation"],
            "simplicity_rank": item["simplicity_rank"],
        }
        for item in candidate_registry["candidates"]
    }
    config = {
        "artifact_id": "scryglass:b2:r20-selection-config:v1",
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "synthetic_only": True,
        "development_only": True,
        "production_eligible": False,
        "target": {
            "kind": "chronological_stochastic_bernoulli_observed_outcome",
            "proper_score_eligible": True,
            "outcome_visibility": "resolution_only",
            "foundation_fixture_label_eligible": False,
        },
        "generator": {
            "generator_id": "scryglass:r20-selection-predictive-generator:v1",
            "seed": 20260731,
            "series_per_cell": _SERIES_PER_CELL,
            "maps_per_series": _MAPS_PER_SERIES,
            "raw_forecast_design": (
                "time-safe signed noisy raw logit whose fidelity varies with "
                "latent unsigned evidence; latent probability and uniform draw "
                "are resolution-only DGP audit fields, never features"
            ),
        },
        "adapter": {
            "adapter_id": "scryglass:r20-positive-modulation-adapter:v2",
            "probability": "sigmoid(raw_logit*exp(unsigned_design@theta))",
            "baseline_terms": [
                "positive_scale_intercept",
                "training_centered_volume",
                "training_centered_volume_squared",
            ],
            "candidate_addition": (
                "one_training_standardized_unsigned_candidate_diagnostic_"
                "inside_positive_log_scale"
            ),
            "candidate_unsigned_main_effect": False,
            "direction_invariants": [
                "raw_zero_probability_exactly_half",
                "side_swap_complement",
                "nonzero_raw_sign_preserved",
                "strictly_positive_exp_scale",
            ],
            "fit": "fold_local_bounded_lbfgsb_proper_log_loss",
            "ridge": _RIDGE,
        },
        "folds": [
            {
                "fold_id": f"selection-fold-{index}",
                "train_series_stop": train_stop,
                "test_series_start": test_start,
                "test_series_stop": test_stop,
            }
            for index, (train_stop, test_start, test_stop) in enumerate(_FOLDS)
        ],
        "metrics": {
            "primary": "log_loss",
            "secondary": "brier",
            "calibration": "descriptive_only_later_phase_owns_transform_selection",
            "map_reconciliation": "mean_within_series_before_inference",
        },
        "dependence": {
            "unit": "series",
            "descriptive_design": (
                "conditional_three_fold_refit_stability_on_frozen_sequence"
            ),
            "interval_method": "none_conditional_observed_fold_envelope",
            "effective_support": None,
            "unconditional_inference": False,
            "series_iid_bootstrap": {
                "status": "non_authoritative_diagnostic_only",
                "seed": _BOOTSTRAP_SEED,
                "replicates": _BOOTSTRAP_REPLICATES,
                "reason": "cross_fold_refit_dependence",
            },
            "sensitivity": "observed_chronological_refit_fold_range",
            "naive_map_standard_error_allowed": False,
        },
        "selection": {
            "scope": "family_local_per_output_registered_stratum",
            "tie_margin_log_loss": _TIE_MARGIN,
            "tie_order": {
                family: list(order) for family, order in FAMILY_TIE_ORDER.items()
            },
            "winner_rule": (
                "no current candidate is selectable without registered "
                "dependence-valid uncertainty; descriptive chronological fold "
                "deltas cannot authorize a winner"
            ),
            "estimand": "conditional_on_frozen_synthetic_sequence",
            "caller_winner_allowed": False,
            "forced_winner_allowed": False,
        },
        "candidate_ids": [method for method, _ in CANDIDATES],
        "candidate_families": {method: family for method, family in CANDIDATES},
        "candidate_execution": candidate_execution,
        "output_strata": [list(cell) for cell in OUTPUT_STRATA],
        "condition_bound": _CONDITION_BOUND,
        "source_closure": _source_closure(root),
        "runtime": _runtime(),
        "foundation_inputs": {
            "authority_locator": FOUNDATION_AUTHORITY_LOCATOR.as_posix(),
            "authority_raw_sha256": _sha(foundation_authority_raw),
            "candidate_registry_locator": FOUNDATION_CANDIDATES_LOCATOR.as_posix(),
            "candidate_registry_raw_sha256": _sha(candidate_raw),
            "role": "candidate identity and family registry only; no foundation rows",
        },
        "methodology_sources": [
            {
                "doi": "10.1007/s10489-021-02735-2",
                "role": "rolling out-of-time predictive development evaluation",
            },
            {
                "doi": "10.1002/qj.456",
                "role": "strictly proper forecast-score framing and separation from reliability",
            },
            {
                "doi": "10.1162/rest_a_01460",
                "role": "dependence-aware cluster inference boundary",
            },
            {
                "doi": "10.48550/arXiv.2505.09090",
                "role": (
                    "sequential scoring-rule comparison context only; the "
                    "artifact makes no time-uniform or unconditional claim"
                ),
            },
        ],
        "wolfram_oracle": {
            "design_rank": 4,
            "singular_values": [
                11.488908016394761,
                3.9413395501504436,
                2.998759594571689,
                1.574254120275471,
            ],
            "series_paired_contributions": [
                -0.07951095648295095,
                -0.07410797215372178,
                -0.09858403245547451,
            ],
            "overall_paired_delta": -0.08406765369738241,
            "effective_support": 3.0,
            "role": "independent numerical fixture; local executable replay is authoritative",
        },
        "wolfram_smoke_interval_oracle": {
            "counts": {"null": 5, "positive": 13, "placebo": 7},
            "trials": 64,
            "wilson_95_intervals": {
                "null": [
                    0.03383117690855046,
                    0.17019537354162906,
                ],
                "positive": [
                    0.12273417819183653,
                    0.3171363573101416,
                ],
                "placebo": [
                    0.054001133444168,
                    0.20898641326896117,
                ],
            },
            "role": (
                "independent arithmetic check for descriptive smoke intervals "
                "only; not selector validation, type-I control, or power evidence"
            ),
        },
        "claim_ceiling": (
            "synthetic development-only R-20 candidate-selection mechanics; "
            "not predictive production evidence, Reliability, SOTA, PASS-B2, promotion, or C1"
        ),
    }
    return config


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _adapter_value(method_id: str, value: object) -> float:
    if isinstance(value, (int, float)) and type(value) is not bool:
        result = float(value)
    elif isinstance(value, Mapping):
        status = value.get("status")
        if method_id == "source_context_strict_v2":
            result = 1.0 if status == "complete" else 0.0
        elif method_id == "source_context_typed_partial_v1":
            result = {"complete": 1.0, "limited": 0.5, "unavailable": 0.0}.get(
                str(status), float("nan")
            )
        else:
            _fail("typed candidate value has no registered forecasting adapter")
    else:
        _fail("candidate value cannot enter the forecasting adapter")
    if not math.isfinite(result):
        _fail("candidate adapter value is nonfinite")
    return result


def _generate_series_observation(
    generator_state: Mapping[str, Any], observation_seed: int
) -> dict[str, Any]:
    if set(generator_state) != {
        "generator_id",
        "latent_success_probability",
        "latent_precision_probability",
        "latent_source_probability",
    }:
        _fail("series latent generator state is missing or extra")
    success_probability = float(generator_state["latent_success_probability"])
    precision_probability = float(generator_state["latent_precision_probability"])
    source_probability = float(generator_state["latent_source_probability"])
    if not all(
        0.0 < value < 1.0
        for value in (
            success_probability,
            precision_probability,
            source_probability,
        )
    ):
        _fail("series latent generator probabilities must lie strictly in (0,1)")
    trial_grid = (24, 36, 48)
    trials = trial_grid[min(2, int(precision_probability * len(trial_grid)))]
    rng = np.random.Generator(np.random.PCG64(observation_seed))
    bernoulli_uniform_draws = rng.random(trials).tolist()
    source_uniform_draws = rng.random(4).tolist()
    observed_outcomes = [
        int(value < success_probability) for value in bernoulli_uniform_draws
    ]
    source_checks = [
        bool(value < source_probability) for value in source_uniform_draws
    ]
    observation = {
        "observation_id": "scryglass:r20-selection-observation:v1",
        "stream": {
            "bit_generator": "PCG64",
            "seed": observation_seed,
            "bernoulli_draw_indices": list(range(trials)),
            "source_draw_indices": list(range(trials, trials + 4)),
        },
        "trials": trials,
        "successes": sum(observed_outcomes),
        "observed_outcomes": observed_outcomes,
        "bernoulli_uniform_draws": bernoulli_uniform_draws,
        "source_uniform_draws": source_uniform_draws,
        "source_checks": source_checks,
    }
    observation["observation_sha256"] = canonical_sha256(observation)
    return observation


def _infer_series_dependencies(
    *,
    observation: Mapping[str, Any],
    inference_seed: int,
    output_type: str,
    stratum_id: str,
) -> dict[str, Any]:
    observation_sha256 = observation.get("observation_sha256")
    unsigned_observation = {
        key: value for key, value in observation.items() if key != "observation_sha256"
    }
    if canonical_sha256(unsigned_observation) != observation_sha256:
        _fail("series observation hash does not replay")
    successes = int(observation["successes"])
    trials = int(observation["trials"])
    if (
        successes != sum(observation["observed_outcomes"])
        or trials != len(observation["observed_outcomes"])
    ):
        _fail("series observation counts do not reconcile")
    posterior_alpha = 2.0 + successes
    posterior_beta = 2.0 + trials - successes
    posterior_rng = np.random.Generator(np.random.PCG64(inference_seed))
    prior_rng = np.random.Generator(np.random.PCG64(inference_seed + 1))
    reference_rng = np.random.Generator(np.random.PCG64(inference_seed + 2))
    source_checks = list(observation["source_checks"])
    dependencies = {
        "posterior_draws": posterior_rng.beta(
            posterior_alpha, posterior_beta, 256
        ).tolist(),
        "prior_draws": prior_rng.beta(2.0, 2.0, 256).tolist(),
        "registered_reference_draws": reference_rng.beta(
            2.0, 2.0, 256
        ).tolist(),
        "source_lineage": {
            "complete": bool(source_checks[0]),
            "registered": True,
        },
        "context_registry": {
            "registered": bool(source_checks[1]),
            "registry_version": "r20-selection-context-v1",
            "path": f"{output_type}:{stratum_id}",
        },
        "fallback_registry": {
            "used": not bool(source_checks[2]),
            "profile": "none" if bool(source_checks[2]) else "fallback",
        },
        "bridge_registry": {
            "registered": bool(source_checks[3]),
            "bridge_id": "r20-selection-bridge-v1",
        },
    }
    inference = {
        "inference_id": "scryglass:r20-selection-beta-binomial-inference:v1",
        "observation_sha256": observation_sha256,
        "registered_prior": {"alpha": 2.0, "beta": 2.0},
        "posterior_parameters": {
            "alpha": posterior_alpha,
            "beta": posterior_beta,
        },
        "streams": {
            "posterior": {"bit_generator": "PCG64", "seed": inference_seed},
            "prior": {"bit_generator": "PCG64", "seed": inference_seed + 1},
            "reference": {"bit_generator": "PCG64", "seed": inference_seed + 2},
        },
        "candidate_dependencies": dependencies,
    }
    inference["inference_output_sha256"] = canonical_sha256(inference)
    return inference


def _execute_candidate_replays(
    config: Mapping[str, Any],
    dependencies: Mapping[str, Any],
    inference_output_sha256: str,
) -> dict[str, Any]:
    replays: dict[str, Any] = {}
    for method_id, family in CANDIDATES:
        execution = config["candidate_execution"][method_id]
        spec = FOUNDATION_METHOD_SPECS[method_id]
        method_dependencies = {
            role: dependencies[role] for role in spec["dependencies"]
        }
        replay = replay_foundation_method(
            method_id=method_id,
            dependencies=method_dependencies,
            boundaries=execution["boundaries"],
        )
        if (
            replay["family"] != family
            or replay["units"] != execution["units"]
            or replay["executed_boundary_sha256"]
            != execution["boundary_sha256"]
            or execution["code_sha256"]
            != execution["implementation"]["source_sha256"]
            or execution["code_sha256"]
            != config["source_closure"]["r20_foundation_algorithms"]
        ):
            _fail("executed candidate implementation or boundary is detached")
        replays[method_id] = {
            "method_id": method_id,
            "family": family,
            "units": replay["units"],
            "value": replay["value"],
            "adapter_value": _adapter_value(method_id, replay["value"]),
            "inference_output_sha256": inference_output_sha256,
            "dependency_sha256": canonical_sha256(method_dependencies),
            "executed_boundary_sha256": replay["executed_boundary_sha256"],
            "implementation_source_sha256": execution["implementation"][
                "source_sha256"
            ],
            "method_complexity": replay["method_complexity"],
        }
    return replays


def _series_candidate_record(
    *,
    config: Mapping[str, Any],
    output_type: str,
    stratum_id: str,
    series_id: str,
    series_seed: int,
    information: float,
    precision: float,
    source_continuity: float,
) -> dict[str, Any]:
    generator_state = {
        "generator_id": "scryglass:r20-selection-latent-generator:v1",
        "latent_success_probability": float(_sigmoid(0.9 * information)),
        "latent_precision_probability": float(_sigmoid(precision)),
        "latent_source_probability": float(
            np.clip(source_continuity, 1.0e-6, 1.0 - 1.0e-6)
        ),
    }
    observation_seed = series_seed
    inference_seed = series_seed + 10_000_000
    observation = _generate_series_observation(generator_state, observation_seed)
    inference = _infer_series_dependencies(
        observation=observation,
        inference_seed=inference_seed,
        output_type=output_type,
        stratum_id=stratum_id,
    )
    dependencies = deepcopy(inference["candidate_dependencies"])
    replays = _execute_candidate_replays(
        config,
        dependencies,
        inference["inference_output_sha256"],
    )
    record = {
        "series_id": series_id,
        "output_type": output_type,
        "stratum_id": stratum_id,
        "generator_state": generator_state,
        "observation": observation,
        "inference": inference,
        "candidate_input": {
            "source": "inference_output_only",
            "inference_output_sha256": inference["inference_output_sha256"],
            "dependencies": dependencies,
            "dependencies_sha256": canonical_sha256(dependencies),
        },
        "dependencies": dependencies,
        "replays": replays,
        "lineage": {
            "generator_state_sha256": canonical_sha256(generator_state),
            "observation_sha256": observation["observation_sha256"],
            "inference_output_sha256": inference["inference_output_sha256"],
            "candidate_dependencies_sha256": canonical_sha256(dependencies),
            "observation_seed": observation_seed,
            "inference_seed": inference_seed,
        },
        "foundation_rows_consumed": False,
    }
    record["record_sha256"] = canonical_sha256(record)
    return record


def build_predictive_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    if config != build_selection_config(Path(".")):
        _fail("predictive generator requires the exact frozen selection config")
    rng = np.random.Generator(np.random.PCG64(config["generator"]["seed"]))
    start = datetime(2026, 8, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    series_records: list[dict[str, Any]] = []
    global_map_index = 0
    for output_index, (output_type, stratum_id) in enumerate(OUTPUT_STRATA):
        output_shift = (-0.16, -0.08, 0.0, 0.08, 0.16)[output_index]
        for series_index in range(_SERIES_PER_CELL):
            volume = float(
                np.clip(
                    0.52
                    + 0.27 * math.sin(series_index * 0.73 + output_index)
                    + rng.normal(0.0, 0.07),
                    0.02,
                    0.98,
                )
            )
            information = float(rng.normal())
            precision = float(rng.normal())
            source_continuity = float(rng.uniform())
            series_id = f"selection:{output_type}:{series_index:03d}"
            series_record = _series_candidate_record(
                config=config,
                output_type=output_type,
                stratum_id=stratum_id,
                series_id=series_id,
                series_seed=(
                    config["generator"]["seed"]
                    + 100_000
                    + output_index * 10_000
                    + series_index
                ),
                information=information,
                precision=precision,
                source_continuity=source_continuity,
            )
            series_records.append(series_record)
            diagnostics = {
                method_id: replay["adapter_value"]
                for method_id, replay in series_record["replays"].items()
            }
            latent_direction = float(rng.normal(0.0, 1.0))
            latent_evidence = float(
                np.clip(
                    0.34 * float(_sigmoid(abs(information)))
                    + 0.33 * float(_sigmoid(precision))
                    + 0.33 * source_continuity,
                    0.02,
                    0.98,
                )
            )
            series_time = start + timedelta(
                minutes=(output_index * _SERIES_PER_CELL + series_index) * 30
            )
            series_shock = float(rng.normal(0.0, 0.18))
            for map_index in range(_MAPS_PER_SERIES):
                issued = series_time + timedelta(minutes=map_index * 10)
                event = issued + timedelta(minutes=2)
                resolved = event + timedelta(minutes=4)
                raw_logit = float(
                    latent_direction * (0.20 + 0.95 * latent_evidence)
                    + rng.normal(0.0, 0.85 - 0.60 * latent_evidence)
                )
                logit = (
                    output_shift
                    + 0.72 * (volume - 0.5)
                    - 0.35 * (volume - 0.5) ** 2
                    + latent_direction
                    + series_shock
                    + (0.04 if map_index else -0.04)
                )
                latent_probability = float(_sigmoid(logit))
                dgp_seed = (
                    config["generator"]["seed"] + 1_000_000 + global_map_index
                )
                uniform_draw = float(
                    np.random.Generator(np.random.PCG64(dgp_seed)).random()
                )
                outcome = int(uniform_draw < latent_probability)
                rows.append(
                    {
                        "row_id": f"selection:{output_type}:{series_index:03d}:{map_index}",
                        "series_id": series_id,
                        "output_type": output_type,
                        "stratum_id": stratum_id,
                        "map_index": map_index,
                        "issued_at": issued.isoformat(),
                        "event_start": event.isoformat(),
                        "resolved_at": resolved.isoformat(),
                        "outcome_visible_at": resolved.isoformat(),
                        "target_kind": "observed_outcome",
                        "proper_score_eligible": True,
                        "observed_outcome": outcome,
                        "dgp": {
                            "dgp_id": "scryglass:r20-selection-bernoulli-dgp:v1",
                            "stream": {
                                "bit_generator": "PCG64",
                                "seed": dgp_seed,
                                "draw_index": 0,
                            },
                            "latent_probability": latent_probability,
                            "uniform_draw": uniform_draw,
                            "outcome_replay": outcome,
                        },
                        "features": {
                            "volume_signal": volume,
                            "raw_logit": raw_logit,
                            "candidate_diagnostics": diagnostics,
                            "candidate_record_sha256": series_record[
                                "record_sha256"
                            ],
                        },
                        "feature_available_at": {
                            "volume_signal": (
                                issued - timedelta(minutes=1)
                            ).isoformat(),
                            "raw_logit": (
                                issued - timedelta(minutes=1)
                            ).isoformat(),
                            "candidate_diagnostics": (
                                issued - timedelta(minutes=1)
                            ).isoformat(),
                            "candidate_record_sha256": (
                                issued - timedelta(minutes=1)
                            ).isoformat(),
                        },
                        "synthetic_only": True,
                        "development_only": True,
                        "production_eligible": False,
                    }
                )
                global_map_index += 1
    payload = {
        "artifact_id": "scryglass:b2:r20-selection-predictive-rows:v1",
        "config_sha256": canonical_sha256(config),
        "generator_id": config["generator"]["generator_id"],
        "synthetic_only": True,
        "development_only": True,
        "production_eligible": False,
        "series_records": series_records,
        "series_records_sha256": canonical_sha256(series_records),
        "rows": rows,
    }
    payload["rows_sha256"] = canonical_sha256(rows)
    return payload


def _validate_predictive_rows_internal_consistency(
    rows: object,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list) or not rows:
        _fail("predictive rows must be a non-empty list")
    submitted_outcomes = [
        item.get("observed_outcome")
        for item in rows
        if isinstance(item, dict)
    ]
    if len(submitted_outcomes) == len(rows) and all(
        type(value) is int and value in (0, 1) for value in submitted_outcomes
    ):
        if set(submitted_outcomes) != {0, 1}:
            _fail("deterministic balanced-class target disguised as outcome")
        if submitted_outcomes == [
            index % 2 for index in range(len(submitted_outcomes))
        ] or submitted_outcomes == [
            1 - index % 2 for index in range(len(submitted_outcomes))
        ]:
            _fail("deterministic alternating class disguised as outcome")
    expected_keys = {
        "row_id",
        "series_id",
        "output_type",
        "stratum_id",
        "map_index",
        "issued_at",
        "event_start",
        "resolved_at",
        "outcome_visible_at",
        "target_kind",
        "proper_score_eligible",
        "observed_outcome",
        "dgp",
        "features",
        "feature_available_at",
        "synthetic_only",
        "development_only",
        "production_eligible",
    }
    seen_rows: set[str] = set()
    series_rows: dict[str, list[dict[str, Any]]] = {}
    for item in rows:
        if not isinstance(item, dict):
            _fail("predictive row must be an object")
        if "fixture_label" in item or "fixture_label_dgp" in item:
            _fail("2B1 fixture ingestion is prohibited")
        if set(item) != expected_keys:
            _fail("predictive row shape is missing or extra")
        if item["target_kind"] != "observed_outcome":
            _fail("predictive target kind is invalid")
        if item["proper_score_eligible"] is not True:
            _fail("predictive target is not proper-score eligible")
        if type(item["observed_outcome"]) is not int or item["observed_outcome"] not in (0, 1):
            _fail("observed outcome must be stochastic Bernoulli resolution")
        if (
            item["synthetic_only"] is not True
            or item["development_only"] is not True
            or item["production_eligible"] is not False
        ):
            _fail("predictive row crosses the synthetic development boundary")
        issued = datetime.fromisoformat(item["issued_at"])
        event = datetime.fromisoformat(item["event_start"])
        resolved = datetime.fromisoformat(item["resolved_at"])
        visible = datetime.fromisoformat(item["outcome_visible_at"])
        if not issued < event < resolved or visible != resolved:
            _fail("outcome chronology or visibility is invalid")
        if set(item["features"]) != {
            "volume_signal",
            "raw_logit",
            "candidate_diagnostics",
            "candidate_record_sha256",
        }:
            _fail("predictive feature shape is invalid")
        if set(item["features"]["candidate_diagnostics"]) != {
            method for method, _ in CANDIDATES
        }:
            _fail("candidate diagnostic universe is incomplete")
        if set(item["feature_available_at"]) != set(item["features"]):
            _fail("feature availability shape is invalid")
        if any(
            datetime.fromisoformat(value) >= event
            for value in item["feature_available_at"].values()
        ):
            _fail("feature availability is not strictly before event start")
        dgp = item["dgp"]
        if not isinstance(dgp, dict) or set(dgp) != {
            "dgp_id",
            "stream",
            "latent_probability",
            "uniform_draw",
            "outcome_replay",
        }:
            _fail("stochastic Bernoulli DGP record is missing or extra")
        if dgp["dgp_id"] != "scryglass:r20-selection-bernoulli-dgp:v1":
            _fail("stochastic Bernoulli DGP identity is invalid")
        stream = dgp["stream"]
        if not isinstance(stream, dict) or set(stream) != {
            "bit_generator",
            "seed",
            "draw_index",
        }:
            _fail("stochastic Bernoulli stream identity is invalid")
        if (
            stream["bit_generator"] != "PCG64"
            or type(stream["seed"]) is not int
            or stream["draw_index"] != 0
        ):
            _fail("stochastic Bernoulli stream identity is invalid")
        probability = dgp["latent_probability"]
        uniform_draw = dgp["uniform_draw"]
        if (
            type(probability) not in (int, float)
            or type(uniform_draw) not in (int, float)
            or not 0.0 < float(probability) < 1.0
            or not 0.0 <= float(uniform_draw) < 1.0
        ):
            _fail("stochastic Bernoulli DGP values are invalid")
        replayed_uniform = float(
            np.random.Generator(np.random.PCG64(stream["seed"])).random()
        )
        replayed_outcome = int(replayed_uniform < float(probability))
        if (
            abs(replayed_uniform - float(uniform_draw)) > 0.0
            or dgp["outcome_replay"] != replayed_outcome
            or item["observed_outcome"] != replayed_outcome
        ):
            _fail("stochastic Bernoulli outcome does not replay")
        row_id = item["row_id"]
        if row_id in seen_rows:
            _fail("predictive row is duplicated")
        seen_rows.add(row_id)
        series_rows.setdefault(item["series_id"], []).append(item)
    for series_id, members in series_rows.items():
        if len(members) != _MAPS_PER_SERIES:
            _fail("series is not map-atomic")
        if {item["map_index"] for item in members} != set(range(_MAPS_PER_SERIES)):
            _fail("series map membership is invalid")
        if len({(item["output_type"], item["stratum_id"]) for item in members}) != 1:
            _fail("series crosses output or stratum")
    return rows


def validate_predictive_rows(
    config: object, rows_payload: object = None
) -> list[dict[str, Any]]:
    """Authenticate a complete payload against the frozen predictive generator."""
    if not isinstance(config, Mapping) or not isinstance(rows_payload, Mapping):
        _fail(
            "authoritative predictive validation requires frozen config and "
            "complete rows payload"
        )
    expected_config = build_selection_config(Path("."))
    if dict(config) != expected_config:
        _fail("authoritative predictive validation requires exact frozen config")
    expected_payload = build_predictive_rows(expected_config)
    if dict(rows_payload) != expected_payload:
        _fail("predictive payload differs from frozen generator replay")
    rows = _validate_predictive_rows_internal_consistency(
        rows_payload.get("rows")
    )
    validate_candidate_replays(expected_config, rows_payload)
    return rows


def _walk_mapping_keys(value: object) -> list[object]:
    keys: list[object] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            keys.append(key)
            keys.extend(_walk_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            keys.extend(_walk_mapping_keys(item))
    return keys


def validate_candidate_replays(
    config: Mapping[str, Any], rows_payload: Mapping[str, Any]
) -> list[dict[str, Any]]:
    records = rows_payload.get("series_records")
    if not isinstance(records, list) or len(records) != len(OUTPUT_STRATA) * _SERIES_PER_CELL:
        _fail("per-series candidate replay lineage is incomplete")
    if canonical_sha256(records) != rows_payload.get("series_records_sha256"):
        _fail("per-series candidate replay lineage hash mismatch")
    by_series: dict[str, dict[str, Any]] = {}
    for record in records:
        if record.get("foundation_rows_consumed") is not False:
            _fail("foundation rows entered candidate replay lineage")
        submitted_hash = record.get("record_sha256")
        unsigned = {key: value for key, value in record.items() if key != "record_sha256"}
        if canonical_sha256(unsigned) != submitted_hash:
            _fail("candidate replay record hash mismatch")
        if set(record.get("replays", {})) != {method for method, _ in CANDIDATES}:
            _fail("candidate replay universe is incomplete")
        lineage = record.get("lineage", {})
        expected_observation = _generate_series_observation(
            record["generator_state"], int(lineage["observation_seed"])
        )
        if record["observation"] != expected_observation:
            _fail("generator to stochastic observation lineage does not replay")
        expected_inference = _infer_series_dependencies(
            observation=record["observation"],
            inference_seed=int(lineage["inference_seed"]),
            output_type=record["output_type"],
            stratum_id=record["stratum_id"],
        )
        if record["inference"] != expected_inference:
            _fail("observation to inference lineage does not replay")
        inferred_dependencies = expected_inference["candidate_dependencies"]
        expected_candidate_input = {
            "source": "inference_output_only",
            "inference_output_sha256": expected_inference[
                "inference_output_sha256"
            ],
            "dependencies": inferred_dependencies,
            "dependencies_sha256": canonical_sha256(inferred_dependencies),
        }
        if (
            record["candidate_input"] != expected_candidate_input
            or record["dependencies"] != inferred_dependencies
        ):
            _fail("inference to candidate-input lineage does not replay")
        if any(
            "latent" in str(key).lower()
            for key in _walk_mapping_keys(record["candidate_input"])
        ):
            _fail("candidate input reads a latent generator field")
        expected_lineage = {
            "generator_state_sha256": canonical_sha256(record["generator_state"]),
            "observation_sha256": expected_observation["observation_sha256"],
            "inference_output_sha256": expected_inference[
                "inference_output_sha256"
            ],
            "candidate_dependencies_sha256": canonical_sha256(
                inferred_dependencies
            ),
            "observation_seed": int(lineage["observation_seed"]),
            "inference_seed": int(lineage["inference_seed"]),
        }
        if lineage != expected_lineage:
            _fail("generator-observation-inference lineage hashes do not reconcile")
        expected_replays = _execute_candidate_replays(
            config,
            inferred_dependencies,
            expected_inference["inference_output_sha256"],
        )
        if record["replays"] != expected_replays:
            _fail("candidate implementation parity or boundary replay failed")
        for method_id, family in CANDIDATES:
            execution = config["candidate_execution"][method_id]
            spec = FOUNDATION_METHOD_SPECS[method_id]
            dependencies = {
                role: inferred_dependencies[role] for role in spec["dependencies"]
            }
            replay = replay_foundation_method(
                method_id=method_id,
                dependencies=dependencies,
                boundaries=execution["boundaries"],
            )
            expected = {
                "method_id": method_id,
                "family": family,
                "units": replay["units"],
                "value": replay["value"],
                "adapter_value": _adapter_value(method_id, replay["value"]),
                "inference_output_sha256": expected_inference[
                    "inference_output_sha256"
                ],
                "dependency_sha256": canonical_sha256(dependencies),
                "executed_boundary_sha256": replay["executed_boundary_sha256"],
                "implementation_source_sha256": execution["implementation"][
                    "source_sha256"
                ],
                "method_complexity": replay["method_complexity"],
            }
            if record["replays"][method_id] != expected:
                _fail("candidate implementation parity or boundary replay failed")
        if record["series_id"] in by_series:
            _fail("candidate replay series is duplicated")
        by_series[record["series_id"]] = record
    for row in rows_payload["rows"]:
        record = by_series.get(row["series_id"])
        if record is None:
            _fail("row has no per-series candidate replay lineage")
        if row["features"]["candidate_record_sha256"] != record["record_sha256"]:
            _fail("row candidate replay lineage is detached")
        values = {
            method_id: replay["adapter_value"]
            for method_id, replay in record["replays"].items()
        }
        if row["features"]["candidate_diagnostics"] != values:
            _fail("generated diagnostic was substituted after exact replay")
    return records


def _fold_ids(
    rows: Sequence[Mapping[str, Any]], output_type: str, fold_index: int
) -> tuple[list[str], list[str]]:
    train_stop, test_start, test_stop = _FOLDS[fold_index]
    series = sorted(
        {
            item["series_id"]
            for item in rows
            if item["output_type"] == output_type
        },
        key=lambda series_id: min(
            item["event_start"] for item in rows if item["series_id"] == series_id
        ),
    )
    return series[:train_stop], series[test_start:test_stop]


def _positive_modulation_probability(
    raw_logit: Sequence[float],
    unsigned_design: np.ndarray,
    parameters: Sequence[float],
) -> np.ndarray:
    raw = np.asarray(raw_logit, dtype=float)
    design = np.asarray(unsigned_design, dtype=float)
    theta = np.asarray(parameters, dtype=float)
    if (
        raw.ndim != 1
        or design.ndim != 2
        or design.shape[0] != raw.size
        or design.shape[1] != theta.size
        or not np.isfinite(raw).all()
        or not np.isfinite(design).all()
        or not np.isfinite(theta).all()
    ):
        _fail("positive modulation probability inputs are invalid")
    log_scale = np.clip(design @ theta, -12.0, 12.0)
    scale = np.exp(log_scale)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        _fail("forecast modulation scale is not strictly positive")
    probability = np.asarray(_sigmoid(raw * scale), dtype=float)
    zero_mask = raw == 0.0
    if np.any(probability[zero_mask] != 0.5):
        _fail("zero signed raw logit did not map exactly to 0.5")
    if np.any(raw > 0) and np.any(probability[raw > 0] <= 0.5):
        _fail("positive raw direction was reversed")
    if np.any(raw < 0) and np.any(probability[raw < 0] >= 0.5):
        _fail("negative raw direction was reversed")
    return probability


def _fit_positive_modulation(
    unsigned_design: np.ndarray,
    raw_logit: Sequence[float],
    outcome: Sequence[float],
) -> np.ndarray:
    design = np.asarray(unsigned_design, dtype=float)
    raw = np.asarray(raw_logit, dtype=float)
    y = np.asarray(outcome, dtype=float)
    if (
        design.ndim != 2
        or raw.shape != (design.shape[0],)
        or y.shape != raw.shape
        or not np.isfinite(design).all()
        or not np.isfinite(raw).all()
        or not np.isfinite(y).all()
        or not np.all((y == 0.0) | (y == 1.0))
    ):
        _fail("positive modulation fit inputs are invalid")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        log_scale = np.clip(design @ theta, -12.0, 12.0)
        scale = np.exp(log_scale)
        eta = raw * scale
        probability = np.asarray(_sigmoid(eta), dtype=float)
        loss = float(
            np.mean(
                -y * np.log(np.clip(probability, 1.0e-12, 1.0))
                - (1.0 - y)
                * np.log(np.clip(1.0 - probability, 1.0e-12, 1.0))
            )
            + _RIDGE * np.sum(theta**2)
        )
        active = (design @ theta > -12.0) & (design @ theta < 12.0)
        gradient = (
            design.T @ ((probability - y) * eta * active.astype(float))
        ) / y.size + 2.0 * _RIDGE * theta
        return loss, np.asarray(gradient, dtype=float)

    result = _SCIPY_MINIMIZE(
        objective,
        np.zeros(design.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-8.0, 8.0)] * design.shape[1],
        options={"ftol": 1.0e-13, "gtol": 1.0e-9, "maxiter": 500},
    )
    theta = np.asarray(result.x, dtype=float)
    if not np.isfinite(theta).all() or not math.isfinite(float(result.fun)):
        _fail("positive modulation forecasting adapter fit is unavailable")
    _positive_modulation_probability(raw, design, theta)
    return theta


def _audit_fitted_direction_invariants(
    parameters: Sequence[float],
    observed_design: np.ndarray,
    observed_raw_logit: Sequence[float],
) -> dict[str, Any]:
    design = np.asarray(observed_design, dtype=float)
    raw = np.asarray(observed_raw_logit, dtype=float)
    observed = _positive_modulation_probability(raw, design, parameters)
    swapped = _positive_modulation_probability(-raw, design, parameters)
    observed_complement_error = float(np.max(np.abs(observed + swapped - 1.0)))
    extreme_design = np.asarray(
        [
            [1.0] + [0.0] * (design.shape[1] - 1),
            [1.0] + [1.0e6] * (design.shape[1] - 1),
            [1.0] + [-1.0e6] * (design.shape[1] - 1),
            [1.0]
            + [(-1.0e6 if index % 2 else 1.0e6) for index in range(design.shape[1] - 1)],
        ],
        dtype=float,
    )
    magnitudes = np.asarray([1.0e-3, 1.0, 10.0, 1.0e6], dtype=float)
    positive = _positive_modulation_probability(
        magnitudes, extreme_design, parameters
    )
    negative = _positive_modulation_probability(
        -magnitudes, extreme_design, parameters
    )
    zero = _positive_modulation_probability(
        np.zeros(extreme_design.shape[0]), extreme_design, parameters
    )
    extreme_complement_error = float(np.max(np.abs(positive + negative - 1.0)))
    zero_deviation = float(np.max(np.abs(zero - 0.5)))
    log_scale = np.clip(
        extreme_design @ np.asarray(parameters, dtype=float), -12.0, 12.0
    )
    minimum_scale = float(np.min(np.exp(log_scale)))
    passed = (
        observed_complement_error <= 1.0e-15
        and extreme_complement_error <= 1.0e-15
        and zero_deviation == 0.0
        and minimum_scale > 0.0
        and np.all(positive > 0.5)
        and np.all(negative < 0.5)
    )
    if not passed:
        _fail("fitted positive modulation direction invariant failed")
    return {
        "status": "pass",
        "observed_complement_error": observed_complement_error,
        "extreme_complement_error": extreme_complement_error,
        "zero_deviation": zero_deviation,
        "minimum_extreme_scale": minimum_scale,
        "positive_direction_preserved": True,
        "negative_direction_preserved": True,
        "adversarial_unsigned_designs": extreme_design.tolist(),
        "adversarial_raw_magnitudes": magnitudes.tolist(),
    }


def evidence_modulation_column(
    raw_logit: Sequence[float],
    diagnostic: Sequence[float],
    *,
    diagnostic_center: float,
    diagnostic_scale: float,
) -> np.ndarray:
    raw = np.asarray(raw_logit, dtype=float)
    diagnostic_values = np.asarray(diagnostic, dtype=float)
    if (
        raw.ndim != 1
        or diagnostic_values.shape != raw.shape
        or not np.isfinite(raw).all()
        or not np.isfinite(diagnostic_values).all()
        or not math.isfinite(diagnostic_center)
        or not math.isfinite(diagnostic_scale)
        or diagnostic_scale <= 0
    ):
        _fail("evidence modulation inputs are invalid")
    return raw * (
        (diagnostic_values - float(diagnostic_center)) / float(diagnostic_scale)
    )


def audit_incremental_design(
    volume: Sequence[float],
    raw_logit: Sequence[float],
    diagnostic: Sequence[float],
) -> dict[str, Any]:
    v = np.asarray(volume, dtype=float)
    raw = np.asarray(raw_logit, dtype=float)
    d = np.asarray(diagnostic, dtype=float)
    if (
        v.ndim != 1
        or raw.shape != v.shape
        or d.shape != v.shape
        or v.size < 8
    ):
        _fail("incremental design support is invalid")
    center, scale = float(np.mean(v)), float(np.std(v, ddof=0))
    if scale <= 0:
        _fail("volume baseline is rank deficient")
    z = (v - center) / scale
    diagnostic_center = float(np.mean(d))
    diagnostic_scale = float(np.std(d, ddof=0))
    if not np.isfinite(raw).all() or not np.isfinite(d).all() or np.any(d < 0):
        _fail("candidate diagnostic must be finite and unsigned")
    if diagnostic_scale <= 0:
        _fail("candidate diagnostic is constant in training")
    volume_only = np.column_stack([np.ones(v.size), z, z**2])
    diagnostic_volume_projection = volume_only @ np.linalg.lstsq(
        volume_only, d, rcond=None
    )[0]
    diagnostic_volume_residual = float(
        np.linalg.norm(d - diagnostic_volume_projection)
    )
    if diagnostic_volume_residual <= 1.0e-8:
        _fail("candidate diagnostic is volume or an exact nonlinear volume proxy")
    base = raw[:, None] * volume_only
    modulation = evidence_modulation_column(
        raw,
        d,
        diagnostic_center=diagnostic_center,
        diagnostic_scale=diagnostic_scale,
    )
    candidate = np.column_stack([base, modulation])
    base_rank = int(np.linalg.matrix_rank(base))
    candidate_rank = int(np.linalg.matrix_rank(candidate))
    condition = float(np.linalg.cond(candidate))
    projection = base @ np.linalg.lstsq(base, modulation, rcond=None)[0]
    residual_norm = float(np.linalg.norm(modulation - projection))
    if candidate_rank - base_rank != 1:
        _fail("candidate has no exact additional rank beyond volume")
    if not math.isfinite(condition) or condition > _CONDITION_BOUND:
        _fail("candidate incremental design condition is inadequate")
    if residual_norm <= 1.0e-8:
        _fail("candidate modulation is contained by the positive-scale baseline")
    return {
        "parameterization": "eta=raw_logit*exp(unsigned_design@theta)",
        "base_rank": base_rank,
        "candidate_rank": candidate_rank,
        "additional_rank": candidate_rank - base_rank,
        "condition_number": condition,
        "nonseparability_residual_norm": residual_norm,
        "diagnostic_volume_residual_norm": diagnostic_volume_residual,
    }


def _losses(probability: np.ndarray, outcome: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.clip(probability, 1.0e-12, 1.0 - 1.0e-12)
    log = -(outcome * np.log(p) + (1.0 - outcome) * np.log(1.0 - p))
    brier = (p - outcome) ** 2
    return log, brier


def replay_cutoff_forecasts(
    rows: Sequence[Mapping[str, Any]], *, series_stop: int = 50
) -> dict[str, Any]:
    """Actually refit and predict the first completed rolling fold per cell."""
    if series_stop != 50:
        _fail("cutoff replay requires the frozen first-fold resolution boundary")
    earlier_rows = [
        row
        for row in rows
        if int(str(row["series_id"]).rsplit(":", 1)[1]) < series_stop
    ]
    training_rows = [
        row
        for row in earlier_rows
        if int(str(row["series_id"]).rsplit(":", 1)[1]) < 30
    ]
    _validate_predictive_rows_internal_consistency(training_rows)
    evidence: list[dict[str, Any]] = []
    for output_type, stratum_id in OUTPUT_STRATA:
        available_series = sorted(
            {
                row["series_id"]
                for row in earlier_rows
                if row["output_type"] == output_type
            }
        )
        required = [
            f"selection:{output_type}:{index:03d}" for index in range(series_stop)
        ]
        if not set(required).issubset(available_series):
            _fail("cutoff replay is missing an earlier required series")
        train_series = set(required[:30])
        test_series = set(required[30:50])
        train = [
            row
            for row in earlier_rows
            if row["output_type"] == output_type
            and row["series_id"] in train_series
        ]
        test = [
            row
            for row in earlier_rows
            if row["output_type"] == output_type and row["series_id"] in test_series
        ]
        for method_id, family in CANDIDATES:
            train_volume = np.asarray(
                [row["features"]["volume_signal"] for row in train], dtype=float
            )
            train_diagnostic = np.asarray(
                [
                    row["features"]["candidate_diagnostics"][method_id]
                    for row in train
                ],
                dtype=float,
            )
            volume_center = float(np.mean(train_volume))
            volume_scale = float(np.std(train_volume, ddof=0))
            diagnostic_center = float(np.mean(train_diagnostic))
            diagnostic_scale = float(np.std(train_diagnostic, ddof=0))

            def design(
                selected: Sequence[Mapping[str, Any]],
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
                volume = np.asarray(
                    [row["features"]["volume_signal"] for row in selected],
                    dtype=float,
                )
                diagnostic = np.asarray(
                    [
                        row["features"]["candidate_diagnostics"][method_id]
                        for row in selected
                    ],
                    dtype=float,
                )
                raw_logit = np.asarray(
                    [row["features"]["raw_logit"] for row in selected],
                    dtype=float,
                )
                z = (volume - volume_center) / volume_scale
                baseline = np.column_stack([np.ones(volume.size), z, z**2])
                candidate = np.column_stack(
                    [
                        baseline,
                        (diagnostic - diagnostic_center) / diagnostic_scale,
                    ]
                )
                return baseline, candidate, raw_logit

            baseline_train, candidate_train, raw_train = design(train)
            baseline_test, candidate_test, raw_test = design(test)
            y_train = np.asarray(
                [row["observed_outcome"] for row in train], dtype=float
            )
            baseline_parameters = _fit_positive_modulation(
                baseline_train, raw_train, y_train
            )
            candidate_parameters = _fit_positive_modulation(
                candidate_train, raw_train, y_train
            )
            baseline_probability = _positive_modulation_probability(
                raw_test, baseline_test, baseline_parameters
            )
            candidate_probability = _positive_modulation_probability(
                raw_test, candidate_test, candidate_parameters
            )
            prediction_payload = [
                {
                    "row_id": row["row_id"],
                    "baseline_probability": float(probability_baseline),
                    "candidate_probability": float(probability_candidate),
                }
                for row, probability_baseline, probability_candidate in zip(
                    test,
                    baseline_probability,
                    candidate_probability,
                )
            ]
            fit_payload = {
                "output_type": output_type,
                "stratum_id": stratum_id,
                "method_id": method_id,
                "family": family,
                "train_row_ids": [row["row_id"] for row in train],
                "train_outcomes": [row["observed_outcome"] for row in train],
                "preprocessing": {
                    "volume_center": volume_center,
                    "volume_scale": volume_scale,
                    "diagnostic_center": diagnostic_center,
                    "diagnostic_scale": diagnostic_scale,
                },
                "baseline_parameters": baseline_parameters.tolist(),
                "candidate_parameters": candidate_parameters.tolist(),
            }
            evidence.append(
                {
                    "output_type": output_type,
                    "stratum_id": stratum_id,
                    "method_id": method_id,
                    "fit_sha256": canonical_sha256(fit_payload),
                    "prediction_bytes_sha256": _sha(
                        (canonical_json(prediction_payload) + "\n").encode()
                    ),
                    "train_row_ids_sha256": canonical_sha256(
                        [row["row_id"] for row in train]
                    ),
                    "test_row_ids_sha256": canonical_sha256(
                        [row["row_id"] for row in test]
                    ),
                }
            )
    result = {
        "cutoff_kind": "first_completed_rolling_fold_per_output",
        "series_stop": series_stop,
        "fit_prediction_evidence": evidence,
    }
    result["replay_sha256"] = canonical_sha256(result)
    return result


def _bootstrap_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.Generator(np.random.PCG64(seed))
    draws = np.mean(
        values[rng.integers(0, values.size, size=(_BOOTSTRAP_REPLICATES, values.size))],
        axis=1,
    )
    return tuple(float(value) for value in np.quantile(draws, [0.025, 0.975]))


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if not 0 <= successes <= trials or trials <= 0:
        _fail("binomial control counts are invalid")
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z**2 / trials
    center = (proportion + z**2 / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z**2 / (4.0 * trials**2)
        )
        / denominator
    )
    return float(center - radius), float(center + radius)


def _control_replication(regime: str, seed: int) -> dict[str, Any]:
    if regime not in {"null", "positive", "placebo"}:
        _fail("procedure-control regime is unregistered")
    rng = np.random.Generator(np.random.PCG64(seed))
    series_count = _FOLDS[-1][2]
    volume = rng.uniform(0.05, 0.95, series_count)
    true_diagnostic = rng.uniform(0.05, 0.95, series_count)
    latent_direction = (
        rng.choice(np.asarray([-1.0, 1.0]), series_count)
        * rng.uniform(0.45, 1.75, series_count)
    )
    if regime == "null":
        raw = latent_direction + rng.normal(0.0, 0.18, series_count)
        diagnostic = rng.uniform(0.05, 0.95, series_count)
    else:
        true_scale = np.exp(-1.5 + 3.0 * true_diagnostic)
        raw = latent_direction / true_scale + rng.normal(
            0.0, 0.01, series_count
        )
        diagnostic = (
            true_diagnostic
            if regime == "positive"
            else np.roll(true_diagnostic, 17)
        )
    map_probability = np.asarray(_sigmoid(8.0 * latent_direction), dtype=float)
    outcomes = np.column_stack(
        [
            (
                rng.random(series_count)
                < np.clip(map_probability + shift, 1.0e-6, 1.0 - 1.0e-6)
            ).astype(float)
            for shift in (-0.02, 0.02)
        ]
    )
    fold_deltas: list[float] = []
    fit_hashes: list[str] = []
    for train_stop, test_start, test_stop in _FOLDS:
        train_slice = slice(0, train_stop)
        test_slice = slice(test_start, test_stop)
        train_volume = volume[train_slice]
        volume_center = float(np.mean(train_volume))
        volume_scale = float(np.std(train_volume, ddof=0))
        diagnostic_center = float(np.mean(diagnostic[train_slice]))
        diagnostic_scale = float(np.std(diagnostic[train_slice], ddof=0))

        def design(selected: slice) -> tuple[np.ndarray, np.ndarray]:
            z = (volume[selected] - volume_center) / volume_scale
            base = np.column_stack([np.ones(z.size), z, z**2])
            candidate = np.column_stack(
                [
                    base,
                    (diagnostic[selected] - diagnostic_center)
                    / diagnostic_scale,
                ]
            )
            return base, candidate

        base_train, candidate_train = design(train_slice)
        base_test, candidate_test = design(test_slice)
        raw_train = np.repeat(raw[train_slice], _MAPS_PER_SERIES)
        raw_test = np.repeat(raw[test_slice], _MAPS_PER_SERIES)
        y_train = outcomes[train_slice].reshape(-1)
        y_test = outcomes[test_slice].reshape(-1)
        base_train_maps = np.repeat(base_train, _MAPS_PER_SERIES, axis=0)
        candidate_train_maps = np.repeat(
            candidate_train, _MAPS_PER_SERIES, axis=0
        )
        base_test_maps = np.repeat(base_test, _MAPS_PER_SERIES, axis=0)
        candidate_test_maps = np.repeat(
            candidate_test, _MAPS_PER_SERIES, axis=0
        )
        base_parameters = _fit_positive_modulation(
            base_train_maps, raw_train, y_train
        )
        candidate_parameters = _fit_positive_modulation(
            candidate_train_maps, raw_train, y_train
        )
        base_probability = _positive_modulation_probability(
            raw_test, base_test_maps, base_parameters
        )
        candidate_probability = _positive_modulation_probability(
            raw_test, candidate_test_maps, candidate_parameters
        )
        base_log, _ = _losses(base_probability, y_test)
        candidate_log, _ = _losses(candidate_probability, y_test)
        series_delta = np.mean(
            (candidate_log - base_log).reshape(-1, _MAPS_PER_SERIES), axis=1
        )
        fold_deltas.append(float(np.mean(series_delta)))
        fit_hashes.append(
            canonical_sha256(
                {
                    "train_series_stop": train_stop,
                    "base_parameters": base_parameters.tolist(),
                    "candidate_parameters": candidate_parameters.tolist(),
                }
            )
        )
    selected = all(value <= 0.0 for value in fold_deltas)
    return {
        "seed": seed,
        "selected": selected,
        "fold_deltas": fold_deltas,
        "full_rolling_refit": True,
        "fit_sha256": canonical_sha256(fit_hashes),
    }


def _run_behavioral_smoke_checks() -> dict[str, Any]:
    regimes: dict[str, Any] = {}
    for regime in ("null", "positive", "placebo"):
        replications = [
            _control_replication(regime, seed) for seed in _CONTROL_SEEDS
        ]
        selected = sum(item["selected"] for item in replications)
        interval = _wilson_interval(selected, len(replications))
        regimes[regime] = {
            "dgp_id": f"scryglass:r20-selection-control-{regime}:v1",
            "estimand": (
                "probability_the_frozen_three_fold_conditional_rule_selects"
            ),
            "seed_count": len(replications),
            "seeds": list(_CONTROL_SEEDS),
            "selected_count": selected,
            "selection_rate": selected / len(replications),
            "descriptive_wilson_95_interval": list(interval),
            "threshold": None,
            "status": "descriptive_only",
            "replications": replications,
        }
    controls = {
        "artifact_role": "non_authoritative_adapter_rule_behavioral_smoke_check",
        "synthetic_only": True,
        "development_only": True,
        "production_eligible": False,
        "neutral_selection_influence": False,
        "selector_validation": False,
        "type_i_error_control": False,
        "power_validation": False,
        "registered_candidate_lineage_exercised": False,
        "selection_rule": (
            "all_three_chronological_full_refit_fold_log_loss_deltas_nonpositive"
        ),
        "future_winner_control_requirement": (
            "controls must traverse generator-observation-inference-foundation "
            "lineage for every family with alpha and power thresholds frozen "
            "by prior authority"
        ),
        "regimes": regimes,
    }
    controls["controls_sha256"] = canonical_sha256(controls)
    return controls


def _candidate_measurement(
    rows: list[dict[str, Any]],
    output_type: str,
    stratum_id: str,
    method_id: str,
    family: str,
) -> dict[str, Any]:
    paired_log: list[float] = []
    paired_brier: list[float] = []
    cluster_sizes: list[int] = []
    fold_evidence: list[dict[str, Any]] = []
    prediction_records: list[dict[str, Any]] = []
    for fold_index in range(len(_FOLDS)):
        train_series, test_series = _fold_ids(rows, output_type, fold_index)
        if set(train_series) & set(test_series):
            _fail("same-series split is prohibited")
        train = [
            item
            for item in rows
            if item["output_type"] == output_type
            and item["series_id"] in set(train_series)
        ]
        test = [
            item
            for item in rows
            if item["output_type"] == output_type
            and item["series_id"] in set(test_series)
        ]
        if max(item["resolved_at"] for item in train) >= min(
            item["issued_at"] for item in test
        ):
            _fail("rolling-origin chronology is swapped")
        train_volume = np.asarray(
            [item["features"]["volume_signal"] for item in train], dtype=float
        )
        train_raw_logit = np.asarray(
            [item["features"]["raw_logit"] for item in train], dtype=float
        )
        train_diagnostic = np.asarray(
            [
                item["features"]["candidate_diagnostics"][method_id]
                for item in train
            ],
            dtype=float,
        )
        design_audit = audit_incremental_design(
            train_volume, train_raw_logit, train_diagnostic
        )
        volume_center = float(np.mean(train_volume))
        volume_scale = float(np.std(train_volume, ddof=0))
        diagnostic_center = float(np.mean(train_diagnostic))
        diagnostic_scale = float(np.std(train_diagnostic, ddof=0))
        if diagnostic_scale <= 0:
            _fail("candidate diagnostic is constant in training")

        def matrices(
            selected: Sequence[Mapping[str, Any]],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            volume = np.asarray(
                [item["features"]["volume_signal"] for item in selected],
                dtype=float,
            )
            diagnostic = np.asarray(
                [
                    item["features"]["candidate_diagnostics"][method_id]
                    for item in selected
                ],
                dtype=float,
            )
            raw_logit = np.asarray(
                [item["features"]["raw_logit"] for item in selected],
                dtype=float,
            )
            z = (volume - volume_center) / volume_scale
            base = np.column_stack([np.ones(volume.size), z, z**2])
            candidate = np.column_stack(
                [
                    base,
                    (diagnostic - diagnostic_center) / diagnostic_scale,
                ]
            )
            return base, candidate, raw_logit

        base_train, candidate_train, raw_train = matrices(train)
        base_test, candidate_test, raw_test = matrices(test)
        y_train = np.asarray([item["observed_outcome"] for item in train], dtype=float)
        y_test = np.asarray([item["observed_outcome"] for item in test], dtype=float)
        baseline_parameters = _fit_positive_modulation(
            base_train, raw_train, y_train
        )
        candidate_parameters = _fit_positive_modulation(
            candidate_train, raw_train, y_train
        )
        baseline_probability = _positive_modulation_probability(
            raw_test, base_test, baseline_parameters
        )
        candidate_probability = _positive_modulation_probability(
            raw_test, candidate_test, candidate_parameters
        )
        baseline_direction_audit = _audit_fitted_direction_invariants(
            baseline_parameters, base_test, raw_test
        )
        candidate_direction_audit = _audit_fitted_direction_invariants(
            candidate_parameters, candidate_test, raw_test
        )
        baseline_log, baseline_brier = _losses(baseline_probability, y_test)
        candidate_log, candidate_brier = _losses(candidate_probability, y_test)
        by_series: dict[str, list[int]] = {}
        for index, row in enumerate(test):
            by_series.setdefault(row["series_id"], []).append(index)
            prediction_records.append(
                {
                    "row_id": row["row_id"],
                    "series_id": row["series_id"],
                    "fold_id": f"selection-fold-{fold_index}",
                    "baseline_probability": float(baseline_probability[index]),
                    "candidate_probability": float(candidate_probability[index]),
                }
            )
        if set(by_series) != set(test_series):
            _fail("test series reconciliation is incomplete")
        for series_id in test_series:
            indices = by_series[series_id]
            if len(indices) != _MAPS_PER_SERIES:
                _fail("map rows were not reconciled into one series contribution")
            paired_log.append(
                float(np.mean(candidate_log[indices] - baseline_log[indices]))
            )
            paired_brier.append(
                float(np.mean(candidate_brier[indices] - baseline_brier[indices]))
            )
            cluster_sizes.append(len(indices))
        fold_evidence.append(
            {
                "fold_id": f"selection-fold-{fold_index}",
                "train_series_ids": train_series,
                "test_series_ids": test_series,
                "fit_scope_series_ids": train_series,
                "training_preprocessing_sha256": canonical_sha256(
                    {
                        "volume_center": volume_center,
                        "volume_scale": volume_scale,
                        "diagnostic_center": diagnostic_center,
                        "diagnostic_scale": diagnostic_scale,
                        "train_series_ids": train_series,
                    }
                ),
                "baseline_design_sha256": canonical_sha256(base_train.tolist()),
                "candidate_design_sha256": canonical_sha256(
                    candidate_train.tolist()
                ),
                "baseline_parameter_sha256": canonical_sha256(
                    baseline_parameters.tolist()
                ),
                "candidate_parameter_sha256": canonical_sha256(
                    candidate_parameters.tolist()
                ),
                "design_audit": design_audit,
                "baseline_direction_audit": baseline_direction_audit,
                "candidate_direction_audit": candidate_direction_audit,
            }
        )
    log_values = np.asarray(paired_log)
    brier_values = np.asarray(paired_brier)
    fold_log_deltas = [
        float(np.mean(values)) for values in np.split(log_values, len(_FOLDS))
    ]
    fold_brier_deltas = [
        float(np.mean(values)) for values in np.split(brier_values, len(_FOLDS))
    ]
    diagnostic_iid_interval = _bootstrap_interval(
        log_values,
        _BOOTSTRAP_SEED
        + list(method for method, _ in CANDIDATES).index(method_id) * 100
        + list(output for output, _ in OUTPUT_STRATA).index(output_type),
    )
    leave_one_cluster_deltas = np.asarray(
        [
            float(np.mean(np.delete(log_values, index)))
            for index in range(log_values.size)
        ]
    )
    sequential_lower = min(fold_log_deltas)
    sequential_upper = max(fold_log_deltas)
    return {
        "output_type": output_type,
        "stratum_id": stratum_id,
        "method_id": method_id,
        "family": family,
        "candidate_eligibility": "eligible",
        "adequacy": "unavailable_dependence_support",
        "selection_status": "not_selected",
        "production_eligibility": "ineligible_synthetic_development_only",
        "primary": {
            "metric": "log_loss",
            "candidate_minus_volume_baseline": float(np.mean(log_values)),
            "paired_series_contributions": paired_log,
            "descriptive_sequential_refit": {
                "estimand": "conditional_on_frozen_synthetic_sequence",
                "fold_deltas": fold_log_deltas,
                "lower_observed_fold_delta": sequential_lower,
                "upper_observed_fold_delta": sequential_upper,
                "maximum_observed_fold_delta": sequential_upper,
                "rule": "all_three_chronological_refit_fold_deltas_must_be_nonpositive",
                "confidence_level": None,
                "unconditional_inference": False,
            },
            "non_authoritative_series_iid_diagnostic": {
                "interval": list(diagnostic_iid_interval),
                "reason": "invalid_for_selection_due_to_cross_fold_refit_dependence",
            },
        },
        "secondary": {
            "metric": "brier",
            "candidate_minus_volume_baseline": float(np.mean(brier_values)),
            "paired_series_contributions": paired_brier,
            "fold_deltas": fold_brier_deltas,
        },
        "calibration_descriptors": {
            "status": "descriptive_only",
            "transform_selection_owner": "later_phase",
        },
        "dependence": {
            "unit": "series",
            "cluster_count": len(cluster_sizes),
            "sequential_refit_block_count": len(_FOLDS),
            "effective_support": None,
            "effective_support_status": (
                "not_identified_under_cross_fold_refit_dependence"
            ),
            "cluster_size_distribution": {
                "minimum": min(cluster_sizes),
                "maximum": max(cluster_sizes),
                "mean": float(np.mean(cluster_sizes)),
            },
            "interval_method": "none_conditional_sequential_fold_envelope",
            "seed": _BOOTSTRAP_SEED,
            "replicates": _BOOTSTRAP_REPLICATES,
            "authoritative_selection_uses_series_iid_bootstrap": False,
            "sensitivity": {
                "method": "full_leave_one_series_cluster",
                "minimum_delta": float(np.min(leave_one_cluster_deltas)),
                "maximum_delta": float(np.max(leave_one_cluster_deltas)),
                "worst_case_delta": float(np.max(leave_one_cluster_deltas)),
                "reconciled_count": int(leave_one_cluster_deltas.size),
            },
            "naive_map_standard_error": None,
        },
        "fold_evidence": fold_evidence,
        "prediction_sha256": canonical_sha256(prediction_records),
    }


def _apply_selection(measurements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selections: list[dict[str, Any]] = []
    for output_type, stratum_id in OUTPUT_STRATA:
        for family, order in FAMILY_TIE_ORDER.items():
            candidates = [
                item
                for item in measurements
                if item["output_type"] == output_type
                and item["stratum_id"] == stratum_id
                and item["family"] == family
            ]
            if {item["method_id"] for item in candidates} != set(order):
                _fail("family-local candidate universe is missing or duplicated")
            eligible = [
                item
                for item in candidates
                if item["candidate_eligibility"] == "eligible"
                and item["adequacy"] == "adequate"
                and item["dependence"]["effective_support"] is not None
                and item["primary"]["descriptive_sequential_refit"][
                    "unconditional_inference"
                ]
                is True
                and item["primary"]["descriptive_sequential_refit"][
                    "maximum_observed_fold_delta"
                ]
                <= 0.0
            ]
            selected: str | None = None
            if eligible:
                best = min(
                    item["primary"]["candidate_minus_volume_baseline"]
                    for item in eligible
                )
                indistinguishable = {
                    item["method_id"]
                    for item in eligible
                    if item["primary"]["candidate_minus_volume_baseline"]
                    <= best + _TIE_MARGIN
                }
                selected = next(
                    method for method in order if method in indistinguishable
                )
            for item in candidates:
                if item["method_id"] == selected:
                    item["selection_status"] = "selected_development"
                elif selected is None:
                    item["selection_status"] = "not_selected_no_eligible_winner"
                else:
                    item["selection_status"] = "not_selected_family_local_rule"
            selections.append(
                {
                    "output_type": output_type,
                    "stratum_id": stratum_id,
                    "family": family,
                    "selected_method_id": selected,
                    "status": (
                        "selected_development"
                        if selected is not None
                        else "unavailable_no_eligible_winner"
                    ),
                    "blocking_reasons": (
                        []
                        if selected is not None
                        else sorted(
                            {
                                item["adequacy"]
                                for item in candidates
                                if item["adequacy"] != "adequate"
                            }
                        )
                    ),
                    "production_eligible": False,
                }
            )
    return selections


def _gate(
    name: str, predicate: bool, evidence: Mapping[str, Any]
) -> dict[str, Any]:
    if type(predicate) is not bool or not predicate:
        _fail(f"hard gate failed: {name}")
    payload = {"gate": name, "status": "pass", "predicate_evidence": dict(evidence)}
    payload["evidence_sha256"] = canonical_sha256(payload)
    return payload


def _contains_forbidden_keys(value: object) -> bool:
    if isinstance(value, Mapping):
        return bool(_FORBIDDEN_CLAIM_KEYS & set(value)) or any(
            _contains_forbidden_keys(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_keys(item) for item in value)
    return False


def build_selection_report(
    config: Mapping[str, Any], rows_payload: Mapping[str, Any]
) -> dict[str, Any]:
    rows = validate_predictive_rows(config, rows_payload)
    measurements = [
        _candidate_measurement(rows, output, stratum, method, family)
        for output, stratum in OUTPUT_STRATA
        for method, family in CANDIDATES
    ]
    selections = _apply_selection(measurements)
    behavioral_smoke_checks = _run_behavioral_smoke_checks()
    oracle = replay_wolfram_oracle()
    cutoff_replay = replay_cutoff_forecasts(rows)
    training_scopes_exact = all(
        fold["fit_scope_series_ids"] == fold["train_series_ids"]
        for item in measurements
        for fold in item["fold_evidence"]
    )
    minimum_rank = min(
        fold["design_audit"]["additional_rank"]
        for item in measurements
        for fold in item["fold_evidence"]
    )
    maximum_condition = max(
        fold["design_audit"]["condition_number"]
        for item in measurements
        for fold in item["fold_evidence"]
    )
    minimum_residual = min(
        fold["design_audit"]["nonseparability_residual_norm"]
        for item in measurements
        for fold in item["fold_evidence"]
    )
    forbidden_absent = not _contains_forbidden_keys(
        {"config": config, "measurements": measurements, "selections": selections}
    )
    lineage_replays = validate_candidate_replays(config, rows_payload)
    direction_audits_pass = all(
        fold[arm]["status"] == "pass"
        for item in measurements
        for fold in item["fold_evidence"]
        for arm in ("baseline_direction_audit", "candidate_direction_audit")
    )
    gate_evidence = {
        "authority_and_source_closure": _gate(
            "authority_and_source_closure",
            config["source_closure"] == _source_closure(Path("."))
            and config["runtime"] == _runtime(),
            {
                "config_sha256": canonical_sha256(config),
                "rows_sha256": canonical_sha256(rows_payload),
                "source_closure": config["source_closure"],
                "runtime": config["runtime"],
                "lineage_record_count": len(lineage_replays),
            },
        ),
        "proper_predictive_target": _gate(
            "proper_predictive_target",
            all(
                row["target_kind"] == "observed_outcome"
                and row["proper_score_eligible"] is True
                and row["outcome_visible_at"] == row["resolved_at"]
                for row in rows
            ),
            {
                "target": config["target"],
                "outcome_count": len(rows),
                "outcome_values": sorted({row["observed_outcome"] for row in rows}),
            },
        ),
        "foundation_fixture_rejected": _gate(
            "foundation_fixture_rejected",
            all(
                "fixture_label" not in row and "fixture_label_dgp" not in row
                for row in rows
            )
            and all(
                record["foundation_rows_consumed"] is False
                for record in rows_payload["series_records"]
            )
            and all(
                record["candidate_input"]["source"] == "inference_output_only"
                for record in lineage_replays
            ),
            {
                "foundation_role": config["foundation_inputs"]["role"],
                "fixture_label_count": sum("fixture_label" in row for row in rows),
            },
        ),
        "chronology_series_atomicity_time_safe": _gate(
            "chronology_series_atomicity_time_safe",
            all(
                datetime.fromisoformat(value)
                < datetime.fromisoformat(row["event_start"])
                for row in rows
                for value in row["feature_available_at"].values()
            )
            and all(
                len([item for item in rows if item["series_id"] == series_id])
                == _MAPS_PER_SERIES
                for series_id in {row["series_id"] for row in rows}
            ),
            {
                "fold_count": len(_FOLDS),
                "maps_per_series": _MAPS_PER_SERIES,
                "strict_feature_availability": True,
            },
        ),
        "development_only_no_sealed_labels": _gate(
            "development_only_no_sealed_labels",
            all(
                row["development_only"] is True
                and row["production_eligible"] is False
                for row in rows
            )
            and not any("sealed" in key for key in rows_payload),
            {
                "development_only": True,
                "artifact_keys_scanned": sorted(rows_payload),
                "cutoff_fit_prediction_replay_sha256": cutoff_replay[
                    "replay_sha256"
                ],
            },
        ),
        "training_only_preprocessing": _gate(
            "training_only_preprocessing",
            training_scopes_exact,
            {
                "all_fit_scopes_equal_train": training_scopes_exact,
                "cutoff_fit_prediction_replay_sha256": cutoff_replay[
                    "replay_sha256"
                ],
            },
        ),
        "exact_volume_baseline": _gate(
            "exact_volume_baseline",
            config["adapter"]["baseline_terms"]
            == [
                "positive_scale_intercept",
                "training_centered_volume",
                "training_centered_volume_squared",
            ]
            and all(
                fold["design_audit"]["base_rank"] == 3
                for item in measurements
                for fold in item["fold_evidence"]
            )
            and direction_audits_pass,
            {"adapter": config["adapter"]},
        ),
        "incremental_rank_condition_nonseparability": _gate(
            "incremental_rank_condition_nonseparability",
            minimum_rank == 1
            and math.isfinite(maximum_condition)
            and maximum_condition <= _CONDITION_BOUND
            and minimum_residual > 1.0e-8,
            {
                "minimum_additional_rank": minimum_rank,
                "maximum_condition": maximum_condition,
                "minimum_nonseparability_residual_norm": minimum_residual,
            },
        ),
        "paired_proper_score_reconciliation": _gate(
            "paired_proper_score_reconciliation",
            all(
                len(item["primary"]["paired_series_contributions"]) == 60
                and len(item["secondary"]["paired_series_contributions"]) == 60
                for item in measurements
            )
            and oracle["design_rank"] == 4,
            {
                "primary": "log_loss",
                "secondary": "brier",
                "series_contribution_count": sorted(
                    {
                        len(item["primary"]["paired_series_contributions"])
                        for item in measurements
                    }
                ),
                "wolfram_oracle_sha256": canonical_sha256(oracle),
                "descriptive_design": config["dependence"][
                    "descriptive_design"
                ],
            },
        ),
        "dependence_unavailable_fail_closed": _gate(
            "dependence_unavailable_fail_closed",
            all(
                item["dependence"]["unit"] == "series"
                and item["dependence"]["naive_map_standard_error"] is None
                and item["dependence"]["effective_support"] is None
                and item["dependence"]["sequential_refit_block_count"]
                == len(_FOLDS)
                and item["dependence"][
                    "authoritative_selection_uses_series_iid_bootstrap"
                ]
                is False
                for item in measurements
            ),
            {
                "unit": "series",
                "effective_support": None,
                "sequential_refit_block_count": len(_FOLDS),
                "unconditional_inference": False,
                "naive_map_standard_error_present": False,
            },
        ),
        "complete_candidate_universe": _gate(
            "complete_candidate_universe",
            len(measurements) == len(OUTPUT_STRATA) * len(CANDIDATES)
            and len(selections) == len(OUTPUT_STRATA) * len(FAMILY_TIE_ORDER)
            and {
                item["method_id"] for item in measurements
            }
            == {method for method, _ in CANDIDATES},
            {
                "candidate_ids": [method for method, _ in CANDIDATES],
                "measurement_count": len(measurements),
                "selection_count": len(selections),
            },
        ),
        "frozen_family_local_tie_rule": _gate(
            "frozen_family_local_tie_rule",
            config["selection"]["tie_order"]
            == {
                family: list(order)
                for family, order in FAMILY_TIE_ORDER.items()
            }
            and config["selection"]["caller_winner_allowed"] is False
            and config["selection"]["forced_winner_allowed"] is False,
            {
                "tie_order": {
                    family: list(order)
                    for family, order in FAMILY_TIE_ORDER.items()
                },
                "tie_margin": _TIE_MARGIN,
                "caller_winner_allowed": False,
                "forced_winner_allowed": False,
            },
        ),
        "reliability_separate_no_universal_scalar": _gate(
            "reliability_separate_no_universal_scalar",
            forbidden_absent,
            {
                "heldout_reliability_in_scope": False,
                "scanned_payload_sha256": canonical_sha256(
                    {
                        "config": config,
                        "measurements": measurements,
                        "selections": selections,
                    }
                ),
            },
        ),
        "synthetic_nonpromotion": _gate(
            "synthetic_nonpromotion",
            config["synthetic_only"] is True
            and config["development_only"] is True
            and config["production_eligible"] is False
            and all(
                item["production_eligible"] is False for item in selections
            ),
            {
                "synthetic_only": True,
                "development_only": True,
                "production_eligible": False,
                "promotion_decision": None,
                "claim_ceiling": config["claim_ceiling"],
            },
        ),
    }
    if tuple(gate_evidence) != HARD_GATES:
        _fail("hard-gate evidence is missing or extra")
    report = {
        "artifact_id": "scryglass:b2:r20-selection-report:v1",
        "kind": "r20_chronological_candidate_selection",
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "synthetic_only": True,
        "development_only": True,
        "production_eligible": False,
        "config_sha256": canonical_sha256(config),
        "rows_artifact_sha256": canonical_sha256(rows_payload),
        "measurements": measurements,
        "selections": selections,
        "behavioral_smoke_checks": behavioral_smoke_checks,
        "hard_gates": gate_evidence,
        "heldout_reliability": {
            "status": "not_evaluated_separate_later_authority"
        },
        "promotion_decision": None,
        "claim_ceiling": config["claim_ceiling"],
    }
    if _FORBIDDEN_CLAIM_KEYS & set(report):
        _fail("report emits a prohibited universal or promotion claim")
    report["report_sha256"] = canonical_sha256(report)
    return report


def forecast_prefix_sha256(
    rows: Sequence[Mapping[str, Any]], cutoff: str
) -> str:
    """Hash all information legally visible before cutoff, excluding outcomes."""
    cutoff_time = datetime.fromisoformat(cutoff)
    visible = [
        {
            key: deepcopy(row[key])
            for key in (
                "row_id",
                "series_id",
                "output_type",
                "stratum_id",
                "issued_at",
                "event_start",
                "features",
                "feature_available_at",
            )
        }
        for row in rows
        if datetime.fromisoformat(row["issued_at"]) < cutoff_time
    ]
    return canonical_sha256(visible)


def replay_wolfram_oracle() -> dict[str, Any]:
    x = np.asarray(
        [
            [1, -2, 4, -1],
            [1, -1, 1, 2],
            [1, 0, 0, -2],
            [1, 1, 1, 1],
            [1, 2, 4, 0],
            [1, 3, 9, 3],
        ],
        dtype=float,
    )
    outcome = np.asarray([0, 1, 0, 1, 1, 0], dtype=float)
    baseline = np.asarray([0.25, 0.55, 0.35, 0.65, 0.72, 0.48])
    candidate = np.asarray([0.22, 0.62, 0.30, 0.70, 0.76, 0.40])
    baseline_loss, _ = _losses(baseline, outcome)
    candidate_loss, _ = _losses(candidate, outcome)
    series = np.mean((candidate_loss - baseline_loss).reshape(3, 2), axis=1)
    weights = np.asarray([2.0, 2.0, 2.0])
    result = {
        "design_rank": int(np.linalg.matrix_rank(x)),
        "singular_values": [float(value) for value in np.linalg.svd(x)[1]],
        "series_paired_contributions": [float(value) for value in series],
        "overall_paired_delta": float(np.mean(series)),
        "effective_support": float(weights.sum() ** 2 / np.sum(weights**2)),
    }
    expected = build_selection_config(Path("."))["wolfram_oracle"]
    for key in (
        "design_rank",
        "series_paired_contributions",
        "overall_paired_delta",
        "effective_support",
    ):
        if not np.allclose(result[key], expected[key], rtol=0.0, atol=1.0e-12):
            _fail("local paired-loss/rank/ESS replay differs from Wolfram oracle")
    if not np.allclose(
        result["singular_values"],
        expected["singular_values"],
        rtol=0.0,
        atol=1.0e-12,
    ):
        _fail("local singular-value replay differs from Wolfram oracle")
    return result


def _require_exact_recomputed_report(
    pinned: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    if dict(recomputed) != dict(pinned):
        _fail("recomputed R-20 selection report differs from loader-pinned report")


def _make_selection_authority_api() -> tuple[type, Any, Any]:
    validation_failure_type = ValidationFailure
    function_type = types.FunctionType
    sha256_type = hashlib.sha256
    json_dumps = json.dumps
    global_namespace = globals()
    issued: "weakref.WeakKeyDictionary[object, dict[str, Any]]" = (
        weakref.WeakKeyDictionary()
    )

    def authority_fail(message: str) -> None:
        raise validation_failure_type(message)

    def stable_value(value: object, active: set[int] | None = None) -> object:
        seen = set() if active is None else active
        if value is None or type(value) in (bool, int, float, str):
            return value
        if isinstance(value, bytes):
            return {"bytes_sha256": sha256_type(value).hexdigest()}
        value_id = id(value)
        if value_id in seen:
            return {
                "cycle_type": f"{type(value).__module__}.{type(value).__qualname__}",
            }
        next_seen = seen | {value_id}
        if isinstance(value, tuple):
            return {
                "tuple": [stable_value(item, next_seen) for item in value]
            }
        if isinstance(value, list):
            return {
                "list": [stable_value(item, next_seen) for item in value]
            }
        if isinstance(value, (set, frozenset)):
            items = [stable_value(item, next_seen) for item in value]
            return {
                type(value).__name__: sorted(
                    items,
                    key=lambda item: json_dumps(
                        item, sort_keys=True, separators=(",", ":")
                    ),
                )
            }
        if isinstance(value, Mapping):
            pairs = [
                (
                    stable_value(key, next_seen),
                    stable_value(item, next_seen),
                )
                for key, item in value.items()
            ]
            pairs.sort(
                key=lambda pair: json_dumps(
                    pair[0], sort_keys=True, separators=(",", ":")
                )
            )
            return {"mapping": pairs}
        if isinstance(value, function_type):
            return {
                "module": value.__module__,
                "qualname": value.__qualname__,
                "code_sha256": code_fingerprint(value.__code__),
                "defaults": stable_value(value.__defaults__, next_seen),
                "kwdefaults": stable_value(value.__kwdefaults__, next_seen),
            }
        rendered = repr(value)
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "repr": None if " at 0x" in rendered else rendered,
        }

    def code_fingerprint(code: types.CodeType) -> str:
        constants = []
        for value in code.co_consts:
            if isinstance(value, types.CodeType):
                constants.append({"nested_code_sha256": code_fingerprint(value)})
            else:
                constants.append(stable_value(value))
        payload = {
            "argcount": code.co_argcount,
            "posonlyargcount": getattr(code, "co_posonlyargcount", 0),
            "kwonlyargcount": code.co_kwonlyargcount,
            "nlocals": code.co_nlocals,
            "stacksize": code.co_stacksize,
            "flags": code.co_flags,
            "bytecode_sha256": sha256_type(code.co_code).hexdigest(),
            "constants": constants,
            "names": list(code.co_names),
            "varnames": list(code.co_varnames),
            "freevars": list(code.co_freevars),
            "cellvars": list(code.co_cellvars),
        }
        return sha256_type(
            json_dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def stable_sha256(value: object) -> str:
        encoded = json_dumps(
            stable_value(value), sort_keys=True, separators=(",", ":")
        ).encode()
        return sha256_type(encoded).hexdigest()

    def callable_fingerprint(function: Any) -> dict[str, object]:
        if not isinstance(function, function_type):
            authority_fail("R-20 executable dependency is not a Python function")
        closure = []
        closure_identities = []
        for cell in function.__closure__ or ():
            try:
                content = cell.cell_contents
            except ValueError:
                content = {"empty_cell": True}
            closure.append(stable_value(content))
            closure_identities.append((id(cell), id(content)))
        return {
            "module": function.__module__,
            "qualname": function.__qualname__,
            "code_identity": id(function.__code__),
            "code_sha256": code_fingerprint(function.__code__),
            "defaults_sha256": stable_sha256(function.__defaults__),
            "kwdefaults_sha256": stable_sha256(function.__kwdefaults__),
            "closure_sha256": stable_sha256(closure),
            "closure_identities": closure_identities,
        }

    bound_globals = {
        "deepcopy": deepcopy,
        "datetime": datetime,
        "timedelta": timedelta,
        "timezone": timezone,
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "math": math,
        "os": os,
        "platform": platform,
        "weakref": weakref,
        "np": np,
        "scipy": scipy,
        "canonical_json": canonical_json,
        "canonical_sha256": canonical_sha256,
        "replay_foundation_method": replay_foundation_method,
        "FOUNDATION_METHOD_SPECS": FOUNDATION_METHOD_SPECS,
        "OUTPUT_STRATA": OUTPUT_STRATA,
        "CANDIDATES": CANDIDATES,
        "FAMILY_TIE_ORDER": FAMILY_TIE_ORDER,
        "HARD_GATES": HARD_GATES,
        "_FOLDS": _FOLDS,
        "_MAPS_PER_SERIES": _MAPS_PER_SERIES,
        "_SERIES_PER_CELL": _SERIES_PER_CELL,
        "_MIN_EFFECTIVE_SUPPORT": _MIN_EFFECTIVE_SUPPORT,
        "_BOOTSTRAP_SEED": _BOOTSTRAP_SEED,
        "_BOOTSTRAP_REPLICATES": _BOOTSTRAP_REPLICATES,
        "_TIE_MARGIN": _TIE_MARGIN,
        "_CONDITION_BOUND": _CONDITION_BOUND,
        "_RIDGE": _RIDGE,
        "_FORBIDDEN_CLAIM_KEYS": _FORBIDDEN_CLAIM_KEYS,
        "_fail": _fail,
        "_sigmoid": _sigmoid,
        "_adapter_value": _adapter_value,
        "_generate_series_observation": _generate_series_observation,
        "_infer_series_dependencies": _infer_series_dependencies,
        "_execute_candidate_replays": _execute_candidate_replays,
        "_series_candidate_record": _series_candidate_record,
        "_walk_mapping_keys": _walk_mapping_keys,
        "_positive_modulation_probability": _positive_modulation_probability,
        "_fit_positive_modulation": _fit_positive_modulation,
        "_audit_fitted_direction_invariants": _audit_fitted_direction_invariants,
        "_SCIPY_MINIMIZE": _SCIPY_MINIMIZE,
        "evidence_modulation_column": evidence_modulation_column,
        "audit_incremental_design": audit_incremental_design,
        "_fold_ids": _fold_ids,
        "_losses": _losses,
        "_bootstrap_interval": _bootstrap_interval,
        "_wilson_interval": _wilson_interval,
        "_control_replication": _control_replication,
        "_run_behavioral_smoke_checks": _run_behavioral_smoke_checks,
        "_candidate_measurement": _candidate_measurement,
        "_apply_selection": _apply_selection,
        "_gate": _gate,
        "_source_closure": _source_closure,
        "_runtime": _runtime,
        "_read_ref": _read_ref,
        "_safe_file": _safe_file,
        "_canonical_payload": _canonical_payload,
        "_validate_predictive_rows_internal_consistency": (
            _validate_predictive_rows_internal_consistency
        ),
        "validate_predictive_rows": validate_predictive_rows,
        "replay_cutoff_forecasts": replay_cutoff_forecasts,
        "replay_wolfram_oracle": replay_wolfram_oracle,
        "build_selection_config": build_selection_config,
        "build_predictive_rows": build_predictive_rows,
        "build_selection_report": build_selection_report,
        "validate_candidate_replays": validate_candidate_replays,
        "_require_exact_recomputed_report": _require_exact_recomputed_report,
    }
    bound_content_names = (
        "OUTPUT_STRATA",
        "CANDIDATES",
        "FAMILY_TIE_ORDER",
        "HARD_GATES",
        "_FOLDS",
    )
    bound_content_hashes = {
        name: stable_sha256(bound_globals[name]) for name in bound_content_names
    }
    bound_forbidden_claim_keys = stable_sha256(_FORBIDDEN_CLAIM_KEYS)
    foundation_specs_hash = stable_sha256(FOUNDATION_METHOD_SPECS)

    recursive_functions: dict[int, Any] = {}

    def register_recursive(function: Any) -> None:
        if not isinstance(function, function_type) or id(function) in recursive_functions:
            return
        recursive_functions[id(function)] = function
        if not function.__module__.startswith("lol_kills."):
            return
        for name in function.__code__.co_names:
            dependency = function.__globals__.get(name)
            if isinstance(dependency, function_type):
                register_recursive(dependency)

    for value in bound_globals.values():
        register_recursive(value)
    for spec in FOUNDATION_METHOD_SPECS.values():
        register_recursive(spec.get("replay"))
    expected_callable_fingerprints = {
        identity: callable_fingerprint(function)
        for identity, function in recursive_functions.items()
    }
    config_call = build_selection_config
    rows_call = build_predictive_rows
    report_call = build_selection_report
    candidate_validation_call = validate_candidate_replays
    predictive_validation_call = validate_predictive_rows
    read_ref_call = _read_ref
    safe_file_call = _safe_file
    canonical_payload_call = _canonical_payload
    sha_call = _sha
    exact_report_call = _require_exact_recomputed_report
    deepcopy_call = deepcopy

    def assert_namespace() -> None:
        for name, value in bound_globals.items():
            if global_namespace.get(name) is not value:
                authority_fail(
                    f"R-20 selection executable dependency changed: {name}"
                )
            if name in bound_content_hashes and stable_sha256(value) != (
                bound_content_hashes[name]
            ):
                authority_fail(
                    f"R-20 selection executable dependency content changed: {name}"
                )
        if stable_sha256(_FORBIDDEN_CLAIM_KEYS) != bound_forbidden_claim_keys:
            authority_fail(
                "R-20 selection executable dependency content changed: "
                "_FORBIDDEN_CLAIM_KEYS"
            )
        if stable_sha256(FOUNDATION_METHOD_SPECS) != foundation_specs_hash:
            authority_fail("accepted foundation method registry content changed")
        for identity, function in recursive_functions.items():
            if id(function) != identity or callable_fingerprint(function) != (
                expected_callable_fingerprints[identity]
            ):
                authority_fail(
                    "R-20 recursive callable executable fingerprint changed: "
                    f"{function.__module__}.{function.__qualname__}"
                )

    class Authority:
        __slots__ = ("__weakref__",)

        def __new__(cls, *args: object, **kwargs: object) -> "Authority":
            raise TypeError("VerifiedR20SelectionAuthority is loader-issued only")

        def __init_subclass__(cls, **kwargs: object) -> None:
            raise TypeError("VerifiedR20SelectionAuthority cannot be subclassed")

        @property
        def config(self) -> dict[str, Any]:
            record = issued.get(self)
            if record is None:
                authority_fail("R-20 selection authority was not loader-issued")
            return deepcopy_call(record["config"])

        @property
        def rows(self) -> dict[str, Any]:
            record = issued.get(self)
            if record is None:
                authority_fail("R-20 selection authority was not loader-issued")
            return deepcopy_call(record["rows"])

        @property
        def report(self) -> dict[str, Any]:
            record = issued.get(self)
            if record is None:
                authority_fail("R-20 selection authority was not loader-issued")
            return deepcopy_call(record["report"])

    Authority.__name__ = "VerifiedR20SelectionAuthority"
    Authority.__qualname__ = "VerifiedR20SelectionAuthority"
    Authority.__module__ = __name__

    def load(
        repo_root: Path | str = Path("."),
    ) -> Authority:
        assert_namespace()
        root = Path(repo_root).resolve()
        authority_raw = safe_file_call(
            root, AUTHORITY_LOCATOR.as_posix()
        ).read_bytes()
        authority_payload = canonical_payload_call(
            authority_raw, "R-20 selection authority"
        )
        if set(authority_payload) != {
            "artifact_id",
            "contract_tree_sha256",
            "synthetic_only",
            "development_only",
            "production_eligible",
            "config_ref",
            "rows_ref",
            "report_ref",
            "foundation_authority_ref",
            "foundation_candidate_registry_ref",
            "claim_ceiling",
        }:
            authority_fail("R-20 selection authority shape is missing or extra")
        config, _ = read_ref_call(root, authority_payload["config_ref"])
        rows, _ = read_ref_call(root, authority_payload["rows_ref"])
        report, _ = read_ref_call(root, authority_payload["report_ref"])
        foundation_authority, foundation_authority_raw = read_ref_call(
            root, authority_payload["foundation_authority_ref"]
        )
        foundation_candidates, foundation_candidates_raw = read_ref_call(
            root, authority_payload["foundation_candidate_registry_ref"]
        )
        expected_config = config_call(root)
        if config != expected_config:
            authority_fail("selection config or executable source closure is detached")
        if rows != rows_call(config):
            authority_fail("selection predictive rows do not replay")
        predictive_validation_call(config, rows)
        recomputed_report = report_call(config, rows)
        exact_report_call(report, recomputed_report)
        registered = [
            (item.get("method_id"), item.get("family"))
            for item in foundation_candidates.get("candidates", ())
        ]
        if registered != list(CANDIDATES):
            authority_fail("accepted foundation candidate universe changed")
        if (
            sha_call(foundation_authority_raw)
            != config["foundation_inputs"]["authority_raw_sha256"]
            or sha_call(foundation_candidates_raw)
            != config["foundation_inputs"]["candidate_registry_raw_sha256"]
            or foundation_authority.get("synthetic_only") is not True
            or foundation_authority.get("production_eligible") is not False
        ):
            authority_fail("foundation identity-only authority is detached")
        if (
            authority_payload["contract_tree_sha256"] != CONTRACT_TREE_SHA256
            or authority_payload["synthetic_only"] is not True
            or authority_payload["development_only"] is not True
            or authority_payload["production_eligible"] is not False
            or authority_payload["claim_ceiling"] != config["claim_ceiling"]
        ):
            authority_fail("selection authority crosses its claim boundary")
        authority = object.__new__(Authority)
        issued[authority] = {
            "config": deepcopy_call(config),
            "rows": deepcopy_call(rows),
            "report": deepcopy_call(report),
            "authority_raw_sha256": sha_call(authority_raw),
        }
        return authority

    def replay(authority: Authority) -> dict[str, Any]:
        assert_namespace()
        if type(authority) is not Authority:
            authority_fail(
                "R-20 selection replay requires exact loader-issued authority"
            )
        record = issued.get(authority)
        if record is None:
            authority_fail("R-20 selection replay requires loader-issued authority")
        recomputed = report_call(record["config"], record["rows"])
        exact_report_call(record["report"], recomputed)
        return deepcopy_call(record["report"])

    return Authority, load, replay


(
    VerifiedR20SelectionAuthority,
    load_r20_selection_authority,
    replay_r20_selection,
) = _make_selection_authority_api()


def write_selection_artifacts(repo_root: Path | str = Path(".")) -> dict[str, str]:
    root = Path(repo_root)
    config = build_selection_config(root)
    rows = build_predictive_rows(config)
    report = build_selection_report(config, rows)
    artifacts = (
        (CONFIG_LOCATOR, config),
        (ROWS_LOCATOR, rows),
        (REPORT_LOCATOR, report),
    )
    refs: dict[str, dict[str, str]] = {}
    hashes: dict[str, str] = {}
    for locator, payload in artifacts:
        path = root / locator
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (canonical_json(payload) + "\n").encode()
        path.write_bytes(raw)
        refs[locator.stem] = _ref(locator, payload, raw)
        hashes[locator.as_posix()] = _sha(raw)
    foundation_authority_path = root / FOUNDATION_AUTHORITY_LOCATOR
    foundation_candidate_path = root / FOUNDATION_CANDIDATES_LOCATOR
    foundation_authority_raw = foundation_authority_path.read_bytes()
    foundation_candidate_raw = foundation_candidate_path.read_bytes()
    foundation_authority = _canonical_payload(
        foundation_authority_raw, "foundation authority"
    )
    foundation_candidates = _canonical_payload(
        foundation_candidate_raw, "foundation candidate registry"
    )
    authority = {
        "artifact_id": "scryglass:b2:r20-selection-authority:v1",
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "synthetic_only": True,
        "development_only": True,
        "production_eligible": False,
        "config_ref": refs[CONFIG_LOCATOR.stem],
        "rows_ref": refs[ROWS_LOCATOR.stem],
        "report_ref": refs[REPORT_LOCATOR.stem],
        "foundation_authority_ref": _ref(
            FOUNDATION_AUTHORITY_LOCATOR,
            foundation_authority,
            foundation_authority_raw,
        ),
        "foundation_candidate_registry_ref": _ref(
            FOUNDATION_CANDIDATES_LOCATOR,
            foundation_candidates,
            foundation_candidate_raw,
        ),
        "claim_ceiling": config["claim_ceiling"],
    }
    authority_path = root / AUTHORITY_LOCATOR
    authority_raw = (canonical_json(authority) + "\n").encode()
    authority_path.write_bytes(authority_raw)
    hashes[AUTHORITY_LOCATOR.as_posix()] = _sha(authority_raw)
    return hashes


__all__ = [
    "AUTHORITY_LOCATOR",
    "CANDIDATES",
    "CONFIG_LOCATOR",
    "FAMILY_TIE_ORDER",
    "HARD_GATES",
    "REPORT_LOCATOR",
    "ROWS_LOCATOR",
    "VerifiedR20SelectionAuthority",
    "audit_incremental_design",
    "build_predictive_rows",
    "build_selection_config",
    "build_selection_report",
    "forecast_prefix_sha256",
    "load_r20_selection_authority",
    "replay_r20_selection",
    "replay_wolfram_oracle",
    "validate_predictive_rows",
    "write_selection_artifacts",
]
