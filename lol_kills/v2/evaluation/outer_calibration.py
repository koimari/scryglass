"""Synthetic-only rolling outer-fold calibration authority.

This module is deliberately standalone.  It does not confer production,
coverage, reliability, promotion, or probability-wording authority.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import stat
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .checks import ValidationFailure


CONTRACT_TREE_SHA256 = "8748bbe48b273593b09304ac80923f11384de808b835f6e83e97c6fef48661dd"
SCHEMA_VERSION = "outer-calibration-v1"
MODEL_ID = "scryglass:model-v2:synthetic-outer-calibration"
CANDIDATE_ORDER = (
    "identity",
    "symmetric_temperature",
    "symmetrized_platt",
    "symmetrized_beta",
    "symmetrized_bounded_isotonic",
)
PAIR_IDENTITIES = tuple(
    f"{CANDIDATE_ORDER[left]}__minus__{CANDIDATE_ORDER[right]}"
    for left in range(len(CANDIDATE_ORDER))
    for right in range(left + 1, len(CANDIDATE_ORDER))
)
OUTPUT_STRATA = (
    ("player_rating", "stratum-player"),
    ("team_rating", "stratum-team"),
    ("draft_score", "stratum-draft"),
    ("partial_draft_state", "stratum-prefix"),
    ("tier_list", "stratum-tier"),
)
OFFSET_STRATA = frozenset({"draft_score", "partial_draft_state"})
CLAIM_CEILING = {
    "scope": "synthetic_calibration_mechanics_only",
    "real_predictive_performance": False,
    "served_approval": False,
    "reliability": False,
    "coverage": False,
    "pass_b2": False,
    "c1": False,
    "promotion": False,
    "probability_wording": False,
    "sota": False,
}
AUTHORITY_THREAT_MODEL = {
    "scope": "process_local_misuse_and_ordinary_forgery_guard_under_honest_interpreter",
    "hostile_same_process_unforgeability": False,
    "closure_cell_mutation_resistant": False,
    "module_or_class_code_mutation_resistant": False,
    "content_hashing_authorizes_promotion": False,
    "singleton_identity_authorizes_promotion": False,
    "production_authority_requirement": "independently_pinned_signature_native_boundary_or_process_os_trust_root",
}
HARD_GATES = (
    "GATE_SYNTHETIC_ONLY_CLAIM_CEILING",
    "GATE_EXACT_FROZEN_CANDIDATES",
    "GATE_SERIES_ATOMIC_CHRONOLOGY",
    "GATE_UPSTREAM_TIME_SAFE",
    "GATE_CALIBRATION_ONLY_FIT",
    "GATE_TEST_LABEL_BLINDNESS",
    "GATE_PAIRED_BLOCK_RECONCILIATION",
    "GATE_DEPENDENCE_SUPPORT",
    "GATE_NONINFERIORITY_BEFORE_SIMPLICITY",
    "GATE_TRANSFORM_SHAPE",
    "GATE_DRAFT_OFFSET_COMPOSITION",
    "GATE_RUNTIME_EXACT_PARITY",
    "GATE_CONTENT_ADDRESSED_CLOSURE",
    "GATE_LOADER_ISSUED_AUTHORITY",
)
DEFAULT_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "seed": 20260728,
    "bit_generator": "PCG64",
    "series_per_stratum": 50,
    "rows_per_series": 16,
    "epsilon": 1e-9,
    "noninferiority_margin_log_loss": 0.005,
    "minimum_calibration_series": 6,
    "minimum_top_level_blocks": 5,
    "minimum_isotonic_distinct_abs_logits": 6,
    "isotonic_root_iterations": 96,
    "folds": (
        {"fold_id": "outer-fold-00", "train": (0, 10), "validation": (10, 16), "calibration": (16, 24), "test": (24, 30)},
        {"fold_id": "outer-fold-01", "train": (0, 16), "validation": (30, 36), "calibration": (36, 44), "test": (44, 50)},
    ),
    "raw_scale_grid": (0.9, 1.0, 1.1),
    "raw_ridge_grid": (0.0, 0.01, 0.1),
    "one_sided_alpha": 0.05,
    "familywise_one_sided_alpha": 0.05,
    "candidate_comparison_count": 10,
    "multiplicity_method": "bonferroni_student_t_over_all_10_predeclared_pairwise_contrasts",
    "reference_limitation": "empirical_best_is_chosen_only_after_all_10_pairwise_bounds_are_frozen",
    "small_cluster_correction": "student_t_over_unique_registered_top_level_blocks",
}


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _strict_json_bytes(raw: bytes, *, label: str) -> Any:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValidationFailure(f"{label}: duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise ValidationFailure(f"{label}: invalid JSON") from exc
    if raw != canonical_json(value):
        raise ValidationFailure(f"{label}: JSON is not canonical")
    return value


def _logit(p: float, epsilon: float) -> float:
    p = min(1.0 - epsilon, max(epsilon, float(p)))
    return math.log(p / (1.0 - p))


def _sigmoid(z: float, epsilon: float) -> float:
    if z >= 0.0:
        p = 1.0 / (1.0 + math.exp(-min(z, 745.0)))
    else:
        e = math.exp(max(z, -745.0))
        p = e / (1.0 + e)
    return min(1.0 - epsilon, max(epsilon, p))


def _loss(y: int, p: float, epsilon: float) -> float:
    p = min(1.0 - epsilon, max(epsilon, p))
    return -(y * math.log(p) + (1 - y) * math.log1p(-p))


def _served_offset_domain(config: Mapping[str, Any]) -> dict[str, Any]:
    policy = config.get("served_offset_policy")
    if not isinstance(policy, Mapping) or set(policy) != {
        "policy_id",
        "quantity",
        "units",
        "maximum_absolute_served_offset",
        "maximum_absolute_odds_multiplier",
        "bound_rationale",
        "evidence_class",
        "production_revalidation",
        "numerical_headroom",
    }:
        raise ValidationFailure("served offset policy is missing or malformed")
    headroom = policy["numerical_headroom"]
    if not isinstance(headroom, Mapping) or set(headroom) != {
        "quantity",
        "units",
        "strict_logit_margin",
        "rationale",
        "evidence_class",
    }:
        raise ValidationFailure("served clamp numerical-headroom policy is missing or malformed")
    epsilon = float(config["epsilon"])
    maximum_offset = float(config["maximum_absolute_served_offset"])
    strict_margin = float(config["served_clamp_strict_logit_margin"])
    theta_upper_bound = float(config["isotonic_theta_upper_bound"])
    if not 0.0 < epsilon < 0.5:
        raise ValidationFailure("served offset domain epsilon is invalid")
    clamp_boundary = math.log((1.0 - epsilon) / epsilon)
    expected_theta_upper_bound = clamp_boundary - maximum_offset - strict_margin
    worst_case_abs_logit = maximum_offset + theta_upper_bound
    actual_margin = clamp_boundary - worst_case_abs_logit
    if (
        not math.isfinite(maximum_offset)
        or maximum_offset <= 0.0
        or not math.isfinite(strict_margin)
        or strict_margin <= 0.0
        or not math.isfinite(theta_upper_bound)
        or theta_upper_bound <= 0.0
        or abs(theta_upper_bound - expected_theta_upper_bound) > 1e-12
        or actual_margin <= 0.0
        or policy["policy_id"] != "synthetic-independent-context-offset-bound-v1"
        or policy["quantity"] != "absolute independent league/context contribution"
        or policy["units"] != "natural-log odds"
        or float(policy["maximum_absolute_served_offset"]) != maximum_offset
        or abs(
            float(policy["maximum_absolute_odds_multiplier"])
            - math.exp(maximum_offset)
        )
        > 1e-15
        or policy["bound_rationale"]
        != (
            "conservative provisional synthetic-mechanics bound for the independently "
            "supplied league/context log-odds term"
        )
        or policy["evidence_class"]
        != "registered synthetic policy; not empirical performance evidence"
        or policy["production_revalidation"]
        != (
            "real-data validation may tighten or replace this bound before any "
            "production use"
        )
        or headroom["quantity"] != "strict clamp-free numerical headroom"
        or headroom["units"] != "natural-log odds"
        or float(headroom["strict_logit_margin"]) != strict_margin
        or headroom["rationale"]
        != (
            "reserved numerical separation from the serving clamp boundary; "
            "not an empirical margin or performance claim"
        )
        or headroom["evidence_class"]
        != "numerical safety policy; not empirical evidence"
    ):
        raise ValidationFailure("served offset domain does not preserve a strict clamp margin")
    evidence = {
        "epsilon": epsilon,
        "clamp_boundary_logit": clamp_boundary,
        "maximum_absolute_served_offset": maximum_offset,
        "isotonic_theta_upper_bound": theta_upper_bound,
        "strict_logit_margin": strict_margin,
        "worst_case_absolute_combined_logit": worst_case_abs_logit,
        "proven_numerical_margin": actual_margin,
        "bound_equation": (
            "abs(offset)+theta <= maximum_absolute_served_offset+"
            "isotonic_theta_upper_bound = log((1-epsilon)/epsilon)-"
            "strict_logit_margin < log((1-epsilon)/epsilon)"
        ),
        "both_offset_signs_and_orientations_covered": True,
        "served_clamp_inactive_for_every_feasible_isotonic_theta": True,
        "served_offset_policy": dict(policy),
    }
    registered = config.get("served_offset_domain")
    if registered is not None and registered != evidence:
        raise ValidationFailure("serialized served offset domain differs from its derivation")
    return evidence


def _validate_served_offset_value(
    offset: Any,
    config: Mapping[str, Any],
    *,
    label: str,
) -> float:
    if not isinstance(offset, (int, float)) or isinstance(offset, bool) or not math.isfinite(float(offset)):
        raise ValidationFailure(f"{label}: served offset must be finite")
    value = float(offset)
    maximum = float(_served_offset_domain(config)["maximum_absolute_served_offset"])
    if abs(value) > maximum:
        raise ValidationFailure(
            f"{label}: absolute served offset exceeds registered maximum {maximum}"
        )
    return value


def _brier(y: int, p: float) -> float:
    return (float(y) - p) ** 2


def build_outer_calibration_config(registry_bytes: bytes, *, regime: str = "nonlinear") -> dict[str, Any]:
    registry = _strict_json_bytes(registry_bytes, label="candidate registry")
    families = tuple(item.get("family") for item in registry.get("candidates", ()))
    ranks = tuple(item.get("simplicity_rank") for item in registry.get("candidates", ()))
    if families != CANDIDATE_ORDER or ranks != tuple(range(len(CANDIDATE_ORDER))):
        raise ValidationFailure("candidate registry does not match the frozen family order and ranks")
    if regime not in {"identity", "temperature", "nonlinear", "sparse"}:
        raise ValidationFailure(f"unknown synthetic regime: {regime}")
    config = dict(DEFAULT_CONFIG)
    config["served_offset_policy"] = {
        "policy_id": "synthetic-independent-context-offset-bound-v1",
        "quantity": "absolute independent league/context contribution",
        "units": "natural-log odds",
        "maximum_absolute_served_offset": 2.0,
        "maximum_absolute_odds_multiplier": math.exp(2.0),
        "bound_rationale": (
            "conservative provisional synthetic-mechanics bound for the independently "
            "supplied league/context log-odds term"
        ),
        "evidence_class": "registered synthetic policy; not empirical performance evidence",
        "production_revalidation": (
            "real-data validation may tighten or replace this bound before any "
            "production use"
        ),
        "numerical_headroom": {
            "quantity": "strict clamp-free numerical headroom",
            "units": "natural-log odds",
            "strict_logit_margin": 0.25,
            "rationale": (
                "reserved numerical separation from the serving clamp boundary; "
                "not an empirical margin or performance claim"
            ),
            "evidence_class": "numerical safety policy; not empirical evidence",
        },
    }
    config["maximum_absolute_served_offset"] = config["served_offset_policy"][
        "maximum_absolute_served_offset"
    ]
    config["served_clamp_strict_logit_margin"] = config["served_offset_policy"][
        "numerical_headroom"
    ]["strict_logit_margin"]
    clamp_boundary = math.log((1.0 - float(config["epsilon"])) / float(config["epsilon"]))
    config["isotonic_theta_upper_bound"] = (
        clamp_boundary
        - float(config["maximum_absolute_served_offset"])
        - float(config["served_clamp_strict_logit_margin"])
    )
    config["served_offset_domain"] = _served_offset_domain(config)
    if regime == "sparse":
        config["series_per_stratum"] = 10
        config["folds"] = (
            {"fold_id": "outer-fold-sparse", "train": (0, 2), "validation": (2, 4), "calibration": (4, 7), "test": (7, 10)},
        )
    config.update(
        {
            "artifact_id": "scryglass:b2:outer-calibration-config:v1",
            "contract_tree_sha256": CONTRACT_TREE_SHA256,
            "model_id": MODEL_ID,
            "regime": regime,
            "candidate_registry_sha256": sha256_bytes(registry_bytes),
            "candidate_order": list(CANDIDATE_ORDER),
            "predeclared_pair_identities": list(PAIR_IDENTITIES),
            "output_strata": [{"output_class": a, "stratum_id": b} for a, b in OUTPUT_STRATA],
            "method_provenance": [
                {
                    "family": "symmetrized_beta",
                    "authors": "Kull, Silva Filho, and Flach",
                    "title": "Beta calibration: a well-founded and easily implemented improvement on logistic calibration for binary classifiers",
                    "venue": "AISTATS, PMLR 54",
                    "year": 2017,
                    "pages": "623-631",
                    "usage": "canonical beta map followed by explicit complement symmetrization",
                },
                {
                    "family": "symmetrized_bounded_isotonic",
                    "authors": "de Leeuw, Hornik, and Mair",
                    "title": "Isotone Optimization in R: Pool-Adjacent-Violators Algorithm (PAVA) and Active Set Methods",
                    "venue": "Journal of Statistical Software 32(5)",
                    "year": 2009,
                    "doi": "10.18637/JSS.V032.I05",
                    "usage": "generalized PAVA for separable convex objectives under chain constraints",
                },
            ],
            "claim_ceiling": CLAIM_CEILING,
            "authority_threat_model": AUTHORITY_THREAT_MODEL,
        }
    )
    config["config_sha256"] = content_hash({k: v for k, v in config.items() if k != "config_sha256"})
    return config


def build_outer_calibration_rows(config: Mapping[str, Any]) -> dict[str, Any]:
    seed = int(config["seed"])
    rng = np.random.Generator(np.random.PCG64(seed))
    block_shocks = [float(value) for value in rng.normal(0.0, 0.28, size=6)]
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for stratum_index, (output_class, stratum_id) in enumerate(OUTPUT_STRATA):
        child_seed = seed + 1009 * (stratum_index + 1)
        child = np.random.Generator(np.random.PCG64(child_seed))
        lineage.append({"stratum_id": stratum_id, "seed": child_seed, "initial_state_sha256": content_hash(child.bit_generator.state)})
        for series_index in range(int(config["series_per_stratum"])):
            series_id = f"{stratum_id}:series-{series_index:03d}"
            block_id = f"generator-block-{series_index % 6:02d}"
            series_effect = float(child.normal(0.0, 0.18))
            block_shock = block_shocks[series_index % 6]
            event_base = start + timedelta(days=series_index * 6, hours=stratum_index * 3)
            stratified_uniforms = [(index + 0.5) / int(config["rows_per_series"]) for index in range(int(config["rows_per_series"]))]
            child.shuffle(stratified_uniforms)
            for row_index in range(int(config["rows_per_series"])):
                row_id = f"{series_id}:row-{row_index}"
                signed_feature = float(child.normal(0.0, 1.35) + series_effect + block_shock + (row_index - 7.5) * 0.06)
                offset = float(child.normal(0.0, 0.35)) if output_class in OFFSET_STRATA else 0.0
                regime = str(config["regime"])
                if regime in {"identity", "sparse"}:
                    calibrated_logit = signed_feature
                elif regime == "temperature":
                    calibrated_logit = 0.25 * signed_feature
                else:
                    calibrated_component = apply_outer_transform(
                        "symmetrized_beta",
                        {"a": 0.1, "b": 4.0, "c": 2.0},
                        signed_feature,
                        epsilon=float(config["epsilon"]),
                    )
                    calibrated_logit = _logit(calibrated_component, float(config["epsilon"]))
                truth_probability = _sigmoid(calibrated_logit + offset, float(config["epsilon"]))
                outcome = int(stratified_uniforms[row_index] < truth_probability)
                issued = event_base - timedelta(days=3, hours=4)
                feature_available = issued - timedelta(hours=2)
                event = event_base + timedelta(hours=row_index)
                resolved = event + timedelta(hours=1)
                rows.append(
                    {
                        "row_id": row_id,
                        "series_id": series_id,
                        "top_level_block_id": block_id,
                        "output_class": output_class,
                        "stratum_id": stratum_id,
                        "series_index": series_index,
                        "timestamps": {
                            "feature_available_at": feature_available.isoformat(),
                            "issued_at": issued.isoformat(),
                            "event_at": event.isoformat(),
                            "resolved_at": resolved.isoformat(),
                        },
                        "generator_truth": {
                            "signed_latent_logit": calibrated_logit,
                            "probability": truth_probability,
                            "top_level_block_shock": block_shock,
                        },
                        "features": {"signed_strength": signed_feature, "league_offset": offset if output_class in OFFSET_STRATA else None},
                        "observation": {"outcome": outcome},
                    }
                )
    _validate_rows(rows, config)
    payload = {
        "artifact_id": "scryglass:b2:outer-calibration-rows:v1",
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "config_sha256": config["config_sha256"],
        "rng_lineage": {
            "algorithm": config["bit_generator"],
            "root_seed": seed,
            "root_initial_state_sha256": content_hash(rng.bit_generator.state),
            "top_level_block_shocks_sha256": content_hash(block_shocks),
            "children": lineage,
        },
        "separation_statement": "generator_truth, observations, and fold-reconstructed transform inputs are distinct",
        "rows": rows,
    }
    payload["rows_sha256"] = content_hash(rows)
    return payload


def _validate_rows(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    _served_offset_domain(config)
    row_ids: set[str] = set()
    series_roles: dict[tuple[str, str], str] = {}
    for row in rows:
        row_id = str(row.get("row_id"))
        if row_id in row_ids:
            raise ValidationFailure(f"duplicate row_id: {row_id}")
        row_ids.add(row_id)
        y = row.get("observation", {}).get("outcome")
        if y not in (0, 1) or isinstance(y, bool):
            raise ValidationFailure(f"{row_id}: outcome must be binary integer")
        feature = row.get("features", {}).get("signed_strength")
        if not isinstance(feature, (int, float)) or not math.isfinite(float(feature)):
            raise ValidationFailure(f"{row_id}: signed strength must be finite")
        if row.get("output_class") in OFFSET_STRATA:
            _validate_served_offset_value(
                row.get("features", {}).get("league_offset"),
                config,
                label=f"{row_id}: time-safe league offset",
            )
        times = row.get("timestamps", {})
        try:
            available = datetime.fromisoformat(times["feature_available_at"])
            issued = datetime.fromisoformat(times["issued_at"])
            event = datetime.fromisoformat(times["event_at"])
            resolved = datetime.fromisoformat(times["resolved_at"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValidationFailure(f"{row_id}: invalid timestamps") from exc
        if not available <= issued < event < resolved:
            raise ValidationFailure(f"{row_id}: timestamp order is not feature <= issued < event < resolved")
        key = (str(row["series_id"]), str(row["stratum_id"]))
        prior = series_roles.setdefault(key, str(row["series_index"]))
        if prior != str(row["series_index"]):
            raise ValidationFailure(f"{row_id}: inconsistent atomic series index")


def _partition(rows: Sequence[Mapping[str, Any]], fold: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    result: dict[str, list[Mapping[str, Any]]] = {}
    seen: dict[str, str] = {}
    for role in ("train", "validation", "calibration", "test"):
        lo, hi = fold[role]
        selected = [row for row in rows if int(lo) <= int(row["series_index"]) < int(hi)]
        for row in selected:
            sid = str(row["series_id"])
            if sid in seen and seen[sid] != role:
                raise ValidationFailure(f"same series split across {seen[sid]} and {role}: {sid}")
            seen[sid] = role
        result[role] = selected
    boundaries = []
    for role in ("train", "validation", "calibration", "test"):
        values = [datetime.fromisoformat(row["timestamps"]["event_at"]) for row in result[role]]
        if not values:
            raise ValidationFailure(f"{fold['fold_id']}: empty {role} partition")
        boundaries.append((min(values), max(values)))
    for left, right in zip(boundaries, boundaries[1:]):
        if not left[1] < right[0]:
            raise ValidationFailure(f"{fold['fold_id']}: chronology overlap")
    return result


def _fit_raw_model(train: Sequence[Mapping[str, Any]], validation: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, float]:
    epsilon = float(config["epsilon"])
    candidates: list[tuple[float, float, float]] = []
    for ridge in config["raw_ridge_grid"]:
        scored: list[tuple[float, float]] = []
        for scale in config["raw_scale_grid"]:
            loss = fmean(
                _loss(int(row["observation"]["outcome"]), _sigmoid(float(scale) * float(row["features"]["signed_strength"]) + _served_offset_for_row(row, config), epsilon), epsilon)
                for row in train
            ) + float(ridge) * (float(scale) - 1.0) ** 2
            scored.append((loss, float(scale)))
        train_loss, scale = min(scored)
        val_loss = fmean(
            _loss(int(row["observation"]["outcome"]), _sigmoid(scale * float(row["features"]["signed_strength"]) + _served_offset_for_row(row, config), epsilon), epsilon)
            for row in validation
        )
        candidates.append((val_loss, float(ridge), scale))
    val_loss, ridge, scale = min(candidates)
    return {"scale": scale, "ridge": ridge, "validation_log_loss": val_loss}


def _transform_input(row: Mapping[str, Any], raw_model: Mapping[str, float]) -> float:
    return float(raw_model["scale"]) * float(row["features"]["signed_strength"])


def _served_offset_for_row(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    if row["output_class"] not in OFFSET_STRATA:
        return 0.0
    return _validate_served_offset_value(
        row["features"].get("league_offset"),
        config,
        label=f"{row['row_id']}: served draft offset",
    )


def _pava(values: Sequence[float], weights: Sequence[int]) -> tuple[list[float], list[int]]:
    means: list[float] = []
    counts: list[int] = []
    for value, weight in zip(values, weights):
        means.append(float(value))
        counts.append(int(weight))
        while len(means) >= 2 and means[-2] > means[-1]:
            combined = (means[-2] * counts[-2] + means[-1] * counts[-1]) / (counts[-2] + counts[-1])
            total = counts[-2] + counts[-1]
            means[-2:] = [combined]
            counts[-2:] = [total]
    return means, counts


def _fit_offset_block_theta(
    outcomes: Sequence[int],
    offsets: Sequence[float],
    *,
    theta_upper_bound: float,
    root_iterations: int,
    epsilon: float,
    maximum_absolute_offset: float,
) -> float:
    if len(outcomes) != len(offsets) or not outcomes:
        raise ValidationFailure("offset-aware isotonic block must be nonempty and aligned")
    if any(outcome not in (0, 1) for outcome in outcomes) or any(not math.isfinite(float(offset)) for offset in offsets):
        raise ValidationFailure("offset-aware isotonic block has invalid outcomes or offsets")
    if not math.isfinite(theta_upper_bound) or theta_upper_bound <= 0.0 or root_iterations < 32:
        raise ValidationFailure("offset-aware isotonic root configuration is invalid")
    clamp_boundary = math.log((1.0 - epsilon) / epsilon)
    if (
        any(abs(float(offset)) > maximum_absolute_offset for offset in offsets)
        or maximum_absolute_offset + theta_upper_bound >= clamp_boundary
    ):
        raise ValidationFailure("offset-aware isotonic block is outside the registered unclamped domain")

    def gradient(theta: float) -> float:
        return sum(_sigmoid(float(offset) + theta, 1e-15) - int(outcome) for outcome, offset in zip(outcomes, offsets))

    if gradient(0.0) >= 0.0:
        return 0.0
    if gradient(theta_upper_bound) <= 0.0:
        return float(theta_upper_bound)
    low, high = 0.0, float(theta_upper_bound)
    for _ in range(int(root_iterations)):
        middle = (low + high) / 2.0
        if gradient(middle) <= 0.0:
            low = middle
        else:
            high = middle
    theta = (low + high) / 2.0
    if not math.isfinite(theta) or not 0.0 <= theta <= theta_upper_bound:
        raise ValidationFailure("offset-aware isotonic block root is nonfinite or out of bounds")
    return theta


def _fit_offset_aware_isotonic(
    rows: Sequence[Mapping[str, Any]],
    logits: Sequence[float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    grouped: dict[float, list[tuple[int, float]]] = {}
    for row, z in zip(rows, logits):
        if float(z) == 0.0:
            continue
        outcome = int(row["observation"]["outcome"])
        offset = _served_offset_for_row(row, config)
        oriented_outcome = outcome if z >= 0.0 else 1 - outcome
        oriented_offset = offset if z >= 0.0 else -offset
        grouped.setdefault(round(abs(float(z)), 12), []).append((oriented_outcome, oriented_offset))
    if len(grouped) < int(config["minimum_isotonic_distinct_abs_logits"]):
        raise ValidationFailure("isotonic support has too few distinct absolute logits")
    upper = float(config["isotonic_theta_upper_bound"])
    iterations = int(config["isotonic_root_iterations"])
    domain = _served_offset_domain(config)
    blocks: list[dict[str, Any]] = []
    for index, knot in enumerate(sorted(grouped)):
        observations = grouped[knot]
        block = {
            "indexes": [index],
            "outcomes": [item[0] for item in observations],
            "offsets": [item[1] for item in observations],
        }
        block["theta"] = _fit_offset_block_theta(
            block["outcomes"],
            block["offsets"],
            theta_upper_bound=upper,
            root_iterations=iterations,
            epsilon=float(config["epsilon"]),
            maximum_absolute_offset=float(domain["maximum_absolute_served_offset"]),
        )
        blocks.append(block)
        while len(blocks) >= 2 and float(blocks[-2]["theta"]) > float(blocks[-1]["theta"]):
            left, right = blocks[-2], blocks[-1]
            merged = {
                "indexes": left["indexes"] + right["indexes"],
                "outcomes": left["outcomes"] + right["outcomes"],
                "offsets": left["offsets"] + right["offsets"],
            }
            merged["theta"] = _fit_offset_block_theta(
                merged["outcomes"],
                merged["offsets"],
                theta_upper_bound=upper,
                root_iterations=iterations,
                epsilon=float(config["epsilon"]),
                maximum_absolute_offset=float(domain["maximum_absolute_served_offset"]),
            )
            blocks[-2:] = [merged]
    knots = sorted(grouped)
    theta_values = [0.0] * len(knots)
    block_diagnostics = []
    for block in blocks:
        for index in block["indexes"]:
            theta_values[index] = float(block["theta"])
        theta = float(block["theta"])
        score = sum(
            _sigmoid(float(offset) + theta, 1e-15) - int(outcome)
            for outcome, offset in zip(block["outcomes"], block["offsets"])
        )
        if theta == 0.0:
            kkt_status = "lower_bound_nonnegative_score"
            kkt_passed = score >= -1e-10
        elif theta == upper:
            kkt_status = "upper_bound_nonpositive_score"
            kkt_passed = score <= 1e-10
        else:
            kkt_status = "interior_zero_score"
            kkt_passed = abs(score) <= 1e-10
        if not kkt_passed:
            raise ValidationFailure("offset-aware isotonic pooled block failed KKT score condition")
        block_diagnostics.append(
            {
                "first_knot_index": min(block["indexes"]),
                "last_knot_index": max(block["indexes"]),
                "theta": theta,
                "score": score,
                "kkt_status": kkt_status,
                "kkt_passed": True,
                "observation_count": len(block["outcomes"]),
            }
        )
    if any(b + 1e-15 < a for a, b in zip(theta_values, theta_values[1:])):
        raise ValidationFailure("offset-aware generalized PAVA did not produce monotone theta")
    return {
        "knots": knots,
        "theta_values": theta_values,
        "theta_upper_bound": upper,
        "root_iterations": iterations,
        "block_diagnostics": block_diagnostics,
        "served_offset_domain": domain,
    }


def _fit_family(family: str, rows: Sequence[Mapping[str, Any]], logits: Sequence[float], config: Mapping[str, Any]) -> dict[str, Any]:
    if family not in CANDIDATE_ORDER:
        raise ValidationFailure(f"unknown calibration family: {family}")
    if len(rows) != len(logits) or not rows:
        raise ValidationFailure("calibration rows and logits must be nonempty and aligned")
    if any(not math.isfinite(float(z)) for z in logits):
        raise ValidationFailure("calibration logits must be finite")
    labels = [int(row["observation"]["outcome"]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValidationFailure("calibration support requires two outcome classes")
    if len(set(round(float(z), 12) for z in logits)) < 2:
        raise ValidationFailure("calibration support requires distinct logits")
    series = {str(row["series_id"]) for row in rows}
    blocks = {str(row["top_level_block_id"]) for row in rows}
    if len(series) < int(config["minimum_calibration_series"]) or len(blocks) < int(config["minimum_top_level_blocks"]):
        raise ValidationFailure("calibration dependence support is inadequate")
    epsilon = float(config["epsilon"])
    domain = _served_offset_domain(config)
    for row in rows:
        _served_offset_for_row(row, config)

    def objective(params: Mapping[str, Any]) -> float:
        return fmean(
            _loss(
                y,
                served_probability(
                    family,
                    params,
                    z,
                    _served_offset_for_row(row, config),
                    epsilon=epsilon,
                    maximum_absolute_offset=float(domain["maximum_absolute_served_offset"]),
                ),
                epsilon,
            )
            for row, y, z in zip(rows, labels, logits)
        )

    if family == "identity":
        params: dict[str, Any] = {}
    elif family == "symmetric_temperature":
        params = min(({"scale": round(0.25 + 0.025 * i, 6)} for i in range(91)), key=objective)
    elif family == "symmetrized_platt":
        params = min(
            ({"scale": round(0.25 + 0.075 * i, 6), "bias": round(0.15 * j, 6)} for i in range(31) for j in range(11)),
            key=objective,
        )
    elif family == "symmetrized_beta":
        beta_a = (0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0)
        beta_b = beta_a
        beta_c = (-2.0, -1.0, 0.0, 1.0, 2.0)
        params = min(
            ({"a": a, "b": b, "c": c} for a in beta_a for b in beta_b for c in beta_c),
            key=objective,
        )
    else:
        params = _fit_offset_aware_isotonic(rows, logits, config)
    result = {
        "family": family,
        "parameters": params,
        "optimizer": {
            "success": True,
            "finite_parameters": True,
            "solver_class": (
                "closed_form_identity"
                if family == "identity"
                else "offset_aware_generalized_pava"
                if family == "symmetrized_bounded_isotonic"
                else "frozen_bounded_grid_search"
            ),
            "gradient_status": "not_applicable",
            "finite_objective": math.isfinite(objective(params)),
            "deterministic_tie_rule": (
                "not_applicable_unique_identity"
                if family == "identity"
                else "generalized_pava_adjacent_pooling_then_canonical_knot_order"
                if family == "symmetrized_bounded_isotonic"
                else "first_lexicographic_frozen_grid_argmin"
            ),
            "feasibility_check": (
                "identity_shape"
                if family == "identity"
                else "finite_monotone_theta_and_offset_block_roots"
                if family == "symmetrized_bounded_isotonic"
                else "finite_objective_over_frozen_grid"
            ),
            "objective_log_loss": objective(params),
            "objective_semantics": "literal_served_probability_epsilon_clamped_log_loss",
            "served_offset_domain": (
                domain
                if family == "symmetrized_bounded_isotonic"
                else "offset_domain_enforced_clamp_may_remain_active_for_parametric_extremes"
            ),
            "kkt_passed": (
                all(block["kkt_passed"] for block in params["block_diagnostics"])
                if family == "symmetrized_bounded_isotonic"
                else "not_applicable"
            ),
        },
        "support": {
            "rows": len(rows),
            "atomic_series": len(series),
            "top_level_blocks": len(blocks),
            "classes": sorted(set(labels)),
            "distinct_logits": len(set(round(float(z), 12) for z in logits)),
        },
    }
    _assert_transform_shape(family, params, epsilon)
    return result


def apply_outer_transform(family: str, parameters: Mapping[str, Any], z: float, *, epsilon: float = 1e-9) -> float:
    z = float(z)
    if not math.isfinite(z):
        raise ValidationFailure("transform input must be finite")
    if z == 0.0:
        return 0.5
    if family == "identity":
        p = _sigmoid(z, epsilon)
    elif family == "symmetric_temperature":
        scale = float(parameters["scale"])
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValidationFailure("temperature scale must be finite and positive")
        p = _sigmoid(scale * z, epsilon)
    elif family == "symmetrized_platt":
        scale, bias = float(parameters["scale"]), float(parameters["bias"])
        if not all(math.isfinite(x) for x in (scale, bias)) or scale <= 0.0 or bias < 0.0:
            raise ValidationFailure("Platt parameters are outside frozen monotone bounds")
        p = 0.5 * (_sigmoid(scale * z + bias, epsilon) + _sigmoid(scale * z - bias, epsilon))
    elif family == "symmetrized_beta":
        a, b, c = float(parameters["a"]), float(parameters["b"]), float(parameters["c"])
        if not all(math.isfinite(x) for x in (a, b, c)) or a < 0.0 or b < 0.0 or a + b <= 0.0:
            raise ValidationFailure("beta parameters are outside frozen monotone bounds")
        base = _sigmoid(abs(z), epsilon)
        log_p = math.log(base)
        log_one_minus_p = math.log1p(-base)
        h_p = _sigmoid(a * log_p - b * log_one_minus_p + c, epsilon)
        h_complement = _sigmoid(a * log_one_minus_p - b * log_p + c, epsilon)
        positive = 0.5 * (h_p + 1.0 - h_complement)
        p = positive if z >= 0.0 else 1.0 - positive
    elif family == "symmetrized_bounded_isotonic":
        knots = [float(x) for x in parameters["knots"]]
        theta_values = [float(x) for x in parameters["theta_values"]]
        upper = float(parameters["theta_upper_bound"])
        if (
            len(knots) != len(theta_values)
            or not knots
            or not math.isfinite(upper)
            or upper <= 0.0
            or any(b <= a for a, b in zip(knots, knots[1:]))
            or any(b < a for a, b in zip(theta_values, theta_values[1:]))
            or any(not math.isfinite(theta) or not 0.0 <= theta <= upper for theta in theta_values)
        ):
            raise ValidationFailure("isotonic parameters are not canonical monotone knots")
        x = abs(z)
        if x <= knots[0]:
            theta = theta_values[0] * (x / knots[0]) if knots[0] > 0.0 else theta_values[0]
        elif x >= knots[-1]:
            theta = theta_values[-1]
        else:
            index = next(i for i in range(1, len(knots)) if x <= knots[i])
            fraction = (x - knots[index - 1]) / (knots[index] - knots[index - 1])
            theta = theta_values[index - 1] + fraction * (theta_values[index] - theta_values[index - 1])
        q = _sigmoid(theta, epsilon)
        p = q if z >= 0.0 else 1.0 - q
    else:
        raise ValidationFailure(f"unknown calibration family: {family}")
    return min(1.0 - epsilon, max(epsilon, p))


def served_probability(
    family: str,
    parameters: Mapping[str, Any],
    z: float,
    offset: float,
    *,
    epsilon: float = 1e-9,
    maximum_absolute_offset: float,
) -> float:
    if (
        not isinstance(offset, (int, float))
        or isinstance(offset, bool)
        or not math.isfinite(float(offset))
        or abs(float(offset)) > float(maximum_absolute_offset)
    ):
        raise ValidationFailure("served offset is outside the registered finite domain")
    calibrated = apply_outer_transform(family, parameters, z, epsilon=epsilon)
    return _sigmoid(float(offset) + _logit(calibrated, epsilon), epsilon)


def _assert_transform_shape(family: str, parameters: Mapping[str, Any], epsilon: float) -> None:
    grid = [-1e6, -100.0, -20.0] + [i / 10 for i in range(-100, 101)] + [20.0, 100.0, 1e6]
    values = [apply_outer_transform(family, parameters, z, epsilon=epsilon) for z in grid]
    if any(not epsilon <= p <= 1.0 - epsilon or not math.isfinite(p) for p in values):
        raise ValidationFailure(f"{family}: transform is not finite and open-bounded")
    if any(b + 1e-15 < a for a, b in zip(values, values[1:])):
        raise ValidationFailure(f"{family}: transform is not monotone nondecreasing")
    if apply_outer_transform(family, parameters, 0.0, epsilon=epsilon) != 0.5:
        raise ValidationFailure(f"{family}: g(0) is not exactly 0.5")
    for z in [0.0, 1e-12, 0.1, 1.0, 7.0, 100.0, 1e6]:
        total = apply_outer_transform(family, parameters, z, epsilon=epsilon) + apply_outer_transform(family, parameters, -z, epsilon=epsilon)
        if abs(total - 1.0) > 2e-15:
            raise ValidationFailure(f"{family}: complement symmetry failed at {z}")


def _candidate_fold_evidence(
    family: str,
    fold_id: str,
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    raw_model: Mapping[str, float],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    epsilon = float(config["epsilon"])
    fits: dict[str, Any] = {}
    unavailable: list[dict[str, str]] = []
    calibration_rows = partitions["calibration"]
    for output_class, stratum_id in OUTPUT_STRATA:
        selected = [row for row in calibration_rows if row["output_class"] == output_class]
        logits = [_transform_input(row, raw_model) for row in selected]
        try:
            fits[stratum_id] = _fit_family(family, selected, logits, config)
        except ValidationFailure as exc:
            unavailable.append({"stratum_id": stratum_id, "reason": str(exc)})
    if unavailable:
        return {"fold_id": fold_id, "family": family, "available": False, "unavailable": unavailable}
    row_evidence: list[dict[str, Any]] = []
    for row in partitions["test"]:
        fit = fits[str(row["stratum_id"])]
        z = _transform_input(row, raw_model)
        offset = _served_offset_for_row(row, config)
        p = served_probability(
            family,
            fit["parameters"],
            z,
            offset,
            epsilon=epsilon,
            maximum_absolute_offset=float(config["maximum_absolute_served_offset"]),
        )
        y = int(row["observation"]["outcome"])
        row_evidence.append(
            {
                "row_id": row["row_id"],
                "series_id": row["series_id"],
                "top_level_block_id": row["top_level_block_id"],
                "stratum_id": row["stratum_id"],
                "raw_logit": z,
                "offset": offset,
                "probability": p,
                "log_loss": _loss(y, p, epsilon),
                "brier": _brier(y, p),
            }
        )
    series_evidence = []
    for series_id in sorted({row["series_id"] for row in row_evidence}):
        selected = [row for row in row_evidence if row["series_id"] == series_id]
        series_evidence.append(
            {
                "series_id": series_id,
                "top_level_block_id": selected[0]["top_level_block_id"],
                "log_loss": fmean(row["log_loss"] for row in selected),
                "brier": fmean(row["brier"] for row in selected),
                "row_count": len(selected),
            }
        )
    block_evidence = []
    for block_id in sorted({row["top_level_block_id"] for row in series_evidence}):
        selected = [row for row in series_evidence if row["top_level_block_id"] == block_id]
        block_evidence.append(
            {
                "top_level_block_id": block_id,
                "log_loss": fmean(row["log_loss"] for row in selected),
                "brier": fmean(row["brier"] for row in selected),
                "series_count": len(selected),
            }
        )
    aggregate = {
        "log_loss": fmean(row["log_loss"] for row in series_evidence),
        "brier": fmean(row["brier"] for row in series_evidence),
        "row_count": len(row_evidence),
        "series_count": len(series_evidence),
        "block_count": len(block_evidence),
        "effective_sample_size": float(len(block_evidence)),
        "largest_cluster_fraction": max(row["series_count"] for row in block_evidence) / len(series_evidence),
        "leave_largest_cluster_log_loss": max(
            fmean(row["log_loss"] for row in block_evidence if row["top_level_block_id"] != omitted["top_level_block_id"])
            for omitted in block_evidence
        ),
    }
    if abs(aggregate["log_loss"] - fmean(row["log_loss"] for row in block_evidence)) > 1e-12:
        raise ValidationFailure("row-series-block aggregate reconciliation failed")
    return {
        "fold_id": fold_id,
        "family": family,
        "available": True,
        "fits": fits,
        "calibration_row_ids": sorted(str(row["row_id"]) for row in calibration_rows),
        "calibration_series_ids": sorted({str(row["series_id"]) for row in calibration_rows}),
        "test_row_ids": sorted(str(row["row_id"]) for row in partitions["test"]),
        "rows": row_evidence,
        "series": series_evidence,
        "blocks": block_evidence,
        "aggregate": aggregate,
    }


def _student_t_critical_95_one_sided(df: int) -> float:
    table = {
        1: 6.314,
        2: 2.920,
        3: 2.353,
        4: 2.132,
        5: 2.015,
        6: 1.943,
        7: 1.895,
        8: 1.860,
        9: 1.833,
        10: 1.812,
        11: 1.796,
        12: 1.782,
        13: 1.771,
        14: 1.761,
        15: 1.753,
        16: 1.746,
        17: 1.740,
        18: 1.734,
        19: 1.729,
        20: 1.725,
    }
    if df not in table:
        raise ValidationFailure(f"no preregistered one-sided t critical for df={df}")
    return table[df]


def _simultaneous_t_critical(df: int, comparison_count: int) -> float:
    if comparison_count != 10:
        raise ValidationFailure("frozen candidate family requires exactly ten predeclared pairwise comparisons")
    table = {
        1: 63.65674116287399,
        2: 9.92484320091807,
        3: 5.840909309733352,
        4: 4.604094871415897,
        5: 4.032142983557535,
        6: 3.707428021324907,
        7: 3.4994832973505026,
        8: 3.3553873313333957,
        9: 3.2498355415921254,
        10: 3.16927267261695,
        11: 3.10580651553928,
        12: 3.0545395893929017,
        13: 3.0122758387165773,
        14: 2.9768427343708344,
        15: 2.946712883485951,
        16: 2.920781622496036,
        17: 2.8982305196347173,
        18: 2.878440472713585,
        19: 2.860934606449914,
        20: 2.845339709776814,
    }
    if df not in table:
        raise ValidationFailure(f"simultaneous support unavailable for df={df}")
    return table[df]


def _select_family(fold_results: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    available = [family for family in CANDIDATE_ORDER if all(next(r for r in fold_results if r["family"] == family and r["fold_id"] == fold)["available"] for fold in sorted({r["fold_id"] for r in fold_results}))]
    if not available:
        return {"status": "unavailable", "selected_family": None, "reason": "no family has complete valid fold support"}
    if tuple(available) != CANDIDATE_ORDER:
        return {
            "status": "unavailable",
            "selected_family": None,
            "reason": "simultaneous family decision requires complete support for all five registered candidates",
        }
    comparison_count = int(config["candidate_comparison_count"])
    means: dict[str, float] = {}
    paired: dict[str, dict[str, float]] = {}
    recurrence: dict[str, dict[str, int]] = {}
    block_sizes: dict[str, dict[str, int]] = {}
    for family in available:
        repeated: dict[str, list[float]] = {}
        sizes: dict[str, int] = {}
        seen_fold_evidence: set[str] = set()
        for result in fold_results:
            if result["family"] == family:
                signature = content_hash({key: value for key, value in result.items() if key != "fold_id"})
                if signature in seen_fold_evidence:
                    continue
                seen_fold_evidence.add(signature)
                for block in result["blocks"]:
                    block_id = str(block["top_level_block_id"])
                    repeated.setdefault(block_id, []).append(float(block["log_loss"]))
                    sizes[block_id] = sizes.get(block_id, 0) + int(block["series_count"])
        paired[family] = {block_id: fmean(values) for block_id, values in repeated.items()}
        recurrence[family] = {block_id: len(values) for block_id, values in repeated.items()}
        block_sizes[family] = sizes
        means[family] = fmean(paired[family].values())
    pairwise_contrasts = []
    for left_index, left in enumerate(CANDIDATE_ORDER):
        for right in CANDIDATE_ORDER[left_index + 1 :]:
            block_ids = sorted(paired[left])
            deltas = [paired[left][block_id] - paired[right][block_id] for block_id in block_ids]
            mean_delta = fmean(deltas)
            df = len(deltas) - 1
            if len(deltas) < int(config["minimum_top_level_blocks"]):
                upper = math.inf
                critical = math.inf
                valid = False
            else:
                variance = sum((value - mean_delta) ** 2 for value in deltas) / df
                critical = _simultaneous_t_critical(df, comparison_count)
                upper = mean_delta + critical * math.sqrt(variance / len(deltas))
                valid = math.isfinite(upper)
            pairwise_contrasts.append(
                {
                    "pair_id": f"{left}__minus__{right}",
                    "left_family": left,
                    "right_family": right,
                    "orientation": "lower_simplicity_rank_minus_higher_simplicity_rank",
                    "mean_delta": mean_delta,
                    "simultaneous_one_sided_upper_bound": upper,
                    "unique_top_level_block_count": len(deltas),
                    "degrees_of_freedom": df,
                    "t_critical": critical,
                    "adjusted_one_sided_alpha": config["familywise_one_sided_alpha"] / comparison_count,
                    "valid": valid,
                }
            )
    if tuple(item["pair_id"] for item in pairwise_contrasts) != tuple(config["predeclared_pair_identities"]):
        raise ValidationFailure("pairwise contrast family differs from the predeclared ten identities")
    best = min(available, key=lambda family: (means[family], CANDIDATE_ORDER.index(family)))
    best_map = paired[best]
    evidence = []
    eligible = []
    for family in available:
        unique_ids = sorted(paired[family])
        if unique_ids != sorted(best_map):
            raise ValidationFailure("candidate top-level block identities do not match")
        deltas_by_block = {block_id: paired[family][block_id] - best_map[block_id] for block_id in unique_ids}
        deltas = list(deltas_by_block.values())
        mean_delta = fmean(deltas)
        if len(deltas) < int(config["minimum_top_level_blocks"]):
            valid = False
            upper = math.inf
            nominal_upper = math.inf
            df = max(0, len(deltas) - 1)
            critical = math.inf
            nominal_critical = math.inf
        else:
            df = len(deltas) - 1
            nominal_critical = _student_t_critical_95_one_sided(df)
            critical = _simultaneous_t_critical(df, comparison_count)
            variance = sum((value - mean_delta) ** 2 for value in deltas) / (len(deltas) - 1)
            nominal_upper = mean_delta + nominal_critical * math.sqrt(variance / len(deltas))
            upper = mean_delta + critical * math.sqrt(variance / len(deltas))
            valid = math.isfinite(upper)
        family_rank = CANDIDATE_ORDER.index(family)
        best_rank = CANDIDATE_ORDER.index(best)
        decision_relevant = family_rank <= best_rank
        if family_rank < best_rank:
            frozen_pair = next(
                item
                for item in pairwise_contrasts
                if item["left_family"] == family and item["right_family"] == best
            )
            upper = frozen_pair["simultaneous_one_sided_upper_bound"]
            critical = frozen_pair["t_critical"]
            valid = frozen_pair["valid"]
        noninferior = valid and decision_relevant and upper <= float(config["noninferiority_margin_log_loss"])
        nominal_noninferior = math.isfinite(nominal_upper) and nominal_upper <= float(config["noninferiority_margin_log_loss"])
        largest_size = max(block_sizes[family].values()) if block_sizes[family] else 0
        largest_ids = sorted(block_id for block_id, size in block_sizes[family].items() if size == largest_size)
        leave_largest = []
        for omitted in largest_ids:
            reduced = [delta for block_id, delta in deltas_by_block.items() if block_id != omitted]
            if len(reduced) < int(config["minimum_top_level_blocks"]):
                reduced_upper = math.inf
                reduced_decision = False
            else:
                reduced_mean = fmean(reduced)
                reduced_variance = sum((value - reduced_mean) ** 2 for value in reduced) / (len(reduced) - 1)
                reduced_upper = reduced_mean + _simultaneous_t_critical(len(reduced) - 1, comparison_count) * math.sqrt(reduced_variance / len(reduced))
                reduced_decision = reduced_upper <= float(config["noninferiority_margin_log_loss"])
            leave_largest.append({"omitted_block_id": omitted, "upper_bound": reduced_upper, "noninferior": reduced_decision})
        if noninferior:
            eligible.append(family)
        evidence.append(
            {
                "family": family,
                "mean_block_log_loss": means[family],
                "paired_delta_to_best": mean_delta,
                "nominal_one_sided_upper_bound": nominal_upper,
                "one_sided_upper_bound": upper,
                "margin": config["noninferiority_margin_log_loss"],
                "unique_top_level_block_count": len(deltas),
                "effective_sample_size": float(len(deltas)),
                "recurrence_by_block": recurrence[family],
                "series_size_by_block": block_sizes[family],
                "degrees_of_freedom": df,
                "one_sided_alpha": config["one_sided_alpha"],
                "familywise_one_sided_alpha": config["familywise_one_sided_alpha"],
                "adjusted_one_sided_alpha": config["familywise_one_sided_alpha"] / comparison_count,
                "candidate_comparison_count": comparison_count,
                "multiplicity_method": config["multiplicity_method"],
                "t_critical": critical,
                "nominal_t_critical": nominal_critical,
                "small_cluster_correction": config["small_cluster_correction"],
                "paired_delta_by_unique_block": deltas_by_block,
                "leave_largest_block_sensitivity": leave_largest,
                "valid": valid,
                "decision_relevant_for_simpler_than_empirical_best": decision_relevant,
                "nominal_noninferior": nominal_noninferior,
                "noninferior": noninferior,
            }
        )
    if not eligible:
        return {"status": "unavailable", "selected_family": None, "reason": "noninferiority evidence unavailable", "evidence": evidence}
    selected = min(eligible, key=CANDIDATE_ORDER.index)
    return {
        "status": "selected_for_synthetic_development_only",
        "selected_family": selected,
        "empirical_best_family": best,
        "selection_rule": "valid simultaneous one-sided paired-block family noninferiority, then lowest frozen simplicity rank",
        "simultaneous_family_rule": {
            "familywise_alpha": config["familywise_one_sided_alpha"],
            "adjusted_alpha": config["familywise_one_sided_alpha"] / comparison_count,
            "comparison_count": comparison_count,
            "method": config["multiplicity_method"],
            "reference_status": config["reference_limitation"],
            "predeclared_pair_identities": list(config["predeclared_pair_identities"]),
        },
        "all_pairwise_contrasts": pairwise_contrasts,
        "evidence": evidence,
    }


def _deduplicate_rows(rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        row_id = str(row["row_id"])
        prior = by_id.setdefault(row_id, row)
        if canonical_json(prior) != canonical_json(row):
            raise ValidationFailure(f"conflicting duplicate row in registered support: {row_id}")
    return [by_id[row_id] for row_id in sorted(by_id)]


def _build_full_presealed_refit(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    selected_family: str,
    *,
    frozen_calibration_lineage: Sequence[Mapping[str, Any]] | None = None,
    frozen_serving_raw_model: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    train_rows = _deduplicate_rows(
        row
        for fold in config["folds"]
        for row in rows
        if int(row["series_index"]) in range(*fold["train"])
    )
    train_ids = {str(row["row_id"]) for row in train_rows}
    validation_rows = _deduplicate_rows(
        row
        for fold in config["folds"]
        for row in rows
        if int(row["series_index"]) in range(*fold["validation"]) and str(row["row_id"]) not in train_ids
    )
    calibration_rows = _deduplicate_rows(
        row
        for fold in config["folds"]
        for row in rows
        if int(row["series_index"]) in range(*fold["calibration"])
    )
    upstream_ids = {str(row["row_id"]) for row in train_rows + validation_rows}
    calibration_ids = {str(row["row_id"]) for row in calibration_rows}
    if upstream_ids & calibration_ids:
        raise ValidationFailure("full calibration support overlaps upstream train/validation support")
    serving_raw_model = (
        dict(frozen_serving_raw_model)
        if frozen_serving_raw_model is not None
        else _fit_raw_model(train_rows, validation_rows, config)
    )
    if frozen_calibration_lineage is None:
        calibration_lineage: list[dict[str, Any]] = []
        seen_calibration_rows: set[str] = set()
        for fold in config["folds"]:
            partitions = _partition(rows, fold)
            fold_raw_model = _fit_raw_model(partitions["train"], partitions["validation"], config)
            fold_raw_model_sha256 = content_hash(fold_raw_model)
            for row in partitions["calibration"]:
                row_id = str(row["row_id"])
                if row_id in seen_calibration_rows:
                    raise ValidationFailure(f"calibration row appears in more than one fold: {row_id}")
                seen_calibration_rows.add(row_id)
                calibration_lineage.append(
                    {
                        "row_id": row_id,
                        "series_id": row["series_id"],
                        "stratum_id": row["stratum_id"],
                        "fold_id": fold["fold_id"],
                        "fold_raw_model_sha256": fold_raw_model_sha256,
                        "raw_logit": _transform_input(row, fold_raw_model),
                    }
                )
        calibration_lineage.sort(key=lambda item: item["row_id"])
    else:
        calibration_lineage = [dict(item) for item in frozen_calibration_lineage]
    lineage_by_row = {str(item["row_id"]): item for item in calibration_lineage}
    if set(lineage_by_row) != calibration_ids or len(lineage_by_row) != len(calibration_lineage):
        raise ValidationFailure("cross-fitted calibration lineage does not exactly cover calibration support")
    calibration_lineage_sha256 = content_hash(calibration_lineage)
    transforms: dict[str, Any] = {}
    for output_class, stratum_id in OUTPUT_STRATA:
        selected_rows = [row for row in calibration_rows if row["output_class"] == output_class]
        fit = _fit_family(selected_family, selected_rows, [float(lineage_by_row[str(row["row_id"])]["raw_logit"]) for row in selected_rows], config)
        transforms[stratum_id] = {
            "output_class": output_class,
            "stratum_id": stratum_id,
            "family": selected_family,
            "parameters": fit["parameters"],
            "epsilon": config["epsilon"],
            "calibration_logit_lineage_sha256": calibration_lineage_sha256,
            "future_serving_raw_model_sha256": content_hash(serving_raw_model),
            "calibration_row_ids": sorted(str(row["row_id"]) for row in selected_rows),
            "calibration_series_ids": sorted({str(row["series_id"]) for row in selected_rows}),
            "support": fit["support"],
        }
    return {
        "status": "available",
        "sealed_opened": False,
        "future_serving_upstream_raw_model": serving_raw_model,
        "upstream_train_row_ids": sorted(str(row["row_id"]) for row in train_rows),
        "upstream_validation_row_ids": sorted(str(row["row_id"]) for row in validation_rows),
        "upstream_support_sha256": content_hash(
            {"train": sorted(str(row["row_id"]) for row in train_rows), "validation": sorted(str(row["row_id"]) for row in validation_rows)}
        ),
        "calibration_row_ids": sorted(calibration_ids),
        "calibration_series_ids": sorted({str(row["series_id"]) for row in calibration_rows}),
        "calibration_logit_lineage": calibration_lineage,
        "calibration_logit_lineage_sha256": calibration_lineage_sha256,
        "calibration_support_sha256": content_hash(
            [
                {
                    "row_id": row["row_id"],
                    "outcome": row["observation"]["outcome"],
                    "fold_id": lineage_by_row[str(row["row_id"])]["fold_id"],
                    "raw_logit": lineage_by_row[str(row["row_id"])]["raw_logit"],
                }
                for row in calibration_rows
            ]
        ),
        "transforms": transforms,
    }


def build_outer_calibration_selection_report(config: Mapping[str, Any], rows_payload: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(rows_payload["rows"])
    _validate_rows(rows, config)
    fold_results: list[dict[str, Any]] = []
    fold_descriptors: list[dict[str, Any]] = []
    for fold in config["folds"]:
        partitions = _partition(rows, fold)
        raw_model = _fit_raw_model(partitions["train"], partitions["validation"], config)
        raw_prediction_sha256 = content_hash(
            [{"row_id": row["row_id"], "raw_logit": _transform_input(row, raw_model)} for role in ("calibration", "test") for row in partitions[role]]
        )
        fold_descriptors.append(
            {
                "fold_id": fold["fold_id"],
                "bounds": {role: list(fold[role]) for role in ("train", "validation", "calibration", "test")},
                "raw_model": raw_model,
                "raw_prediction_sha256": raw_prediction_sha256,
                "source_label_roles": {"raw_model_fit": ["train", "validation"], "transform_fit": ["calibration"], "scoring": ["test"]},
            }
        )
        for family in CANDIDATE_ORDER:
            fold_results.append(_candidate_fold_evidence(family, str(fold["fold_id"]), partitions, raw_model, config))
    selection = _select_family(fold_results, config)
    full_refit: dict[str, Any] = {"status": "unavailable", "sealed_opened": False, "transforms": {}}
    if selection["selected_family"] is not None:
        full_refit = _build_full_presealed_refit(config, rows, str(selection["selected_family"]))
    report = {
        "artifact_id": "scryglass:b2:outer-calibration-selection-report:v1",
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "config_sha256": config["config_sha256"],
        "rows_sha256": rows_payload["rows_sha256"],
        "candidate_registry_sha256": config["candidate_registry_sha256"],
        "candidate_order": list(CANDIDATE_ORDER),
        "folds": fold_descriptors,
        "fold_results": fold_results,
        "selection": selection,
        "full_presealed_refit": full_refit,
        "claim_ceiling": CLAIM_CEILING,
        "authority_threat_model": AUTHORITY_THREAT_MODEL,
    }
    report["selection_report_sha256"] = content_hash({k: v for k, v in report.items() if k != "selection_report_sha256"})
    return report


def _fit_prediction_signature(report: Mapping[str, Any]) -> str:
    return content_hash(
        [
            {
                "fold_id": result["fold_id"],
                "family": result["family"],
                "available": result["available"],
                "fits": result.get("fits"),
                "predictions": [
                    {"row_id": row["row_id"], "raw_logit": row["raw_logit"], "offset": row["offset"], "probability": row["probability"]}
                    for row in result.get("rows", ())
                ],
            }
            for result in report["fold_results"]
        ]
    )


def _evaluate_hard_gates(config: Mapping[str, Any], rows_payload: Mapping[str, Any], report: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(rows_payload["rows"])
    results: dict[str, tuple[str, Any, bool]] = {}

    results["GATE_SYNTHETIC_ONLY_CLAIM_CEILING"] = (
        "claim ceiling equals the frozen synthetic-only ceiling",
        {
            "synthetic": rows_payload.get("synthetic"),
            "claim_ceiling": report.get("claim_ceiling"),
            "authority_threat_model": report.get("authority_threat_model"),
        },
        rows_payload.get("synthetic") is True
        and report.get("claim_ceiling") == CLAIM_CEILING
        and report.get("authority_threat_model") == AUTHORITY_THREAT_MODEL,
    )
    results["GATE_EXACT_FROZEN_CANDIDATES"] = (
        "candidate order exactly equals the immutable five-family registry order",
        {"candidate_order": report.get("candidate_order"), "registry_sha256": config.get("candidate_registry_sha256")},
        tuple(report.get("candidate_order", ())) == CANDIDATE_ORDER,
    )
    chronology_evidence = []
    chronology_passed = True
    for fold in config["folds"]:
        try:
            parts = _partition(rows, fold)
            chronology_evidence.append(
                {
                    "fold_id": fold["fold_id"],
                    "role_series_counts": {role: len({row["series_id"] for row in parts[role]}) for role in ("train", "validation", "calibration", "test")},
                }
            )
        except ValidationFailure as exc:
            chronology_passed = False
            chronology_evidence.append({"fold_id": fold["fold_id"], "failure": str(exc)})
    results["GATE_SERIES_ATOMIC_CHRONOLOGY"] = (
        "every fold is series-atomic with strict train < validation < calibration < test",
        chronology_evidence,
        chronology_passed,
    )
    upstream_evidence = [
        {"fold_id": fold["fold_id"], "source_label_roles": fold["source_label_roles"], "raw_prediction_sha256": fold["raw_prediction_sha256"]}
        for fold in report["folds"]
    ]
    later_validation_mutation = json.loads(json.dumps(rows))
    later_fold = config["folds"][-1]
    later_validation_indexes = set(range(*later_fold["validation"]))
    for row in later_validation_mutation:
        if int(row["series_index"]) in later_validation_indexes:
            row["observation"]["outcome"] = 1 - int(row["observation"]["outcome"])
    mutation_refit = _build_full_presealed_refit(
        config,
        later_validation_mutation,
        str(report["selection"]["selected_family"]),
    )
    original_lineage = report["full_presealed_refit"]["calibration_logit_lineage"]
    mutated_lineage = mutation_refit["calibration_logit_lineage"]
    early_fold_id = config["folds"][0]["fold_id"]
    original_early = [item for item in original_lineage if item["fold_id"] == early_fold_id]
    mutated_early = [item for item in mutated_lineage if item["fold_id"] == early_fold_id]
    changed_lineage_folds = sorted(
        {
            before["fold_id"]
            for before, after in zip(original_lineage, mutated_lineage)
            if canonical_json(before) != canonical_json(after)
        }
    )
    later_mutation_passed = (
        canonical_json(original_early) == canonical_json(mutated_early)
        and set(changed_lineage_folds) <= {str(later_fold["fold_id"])}
    )
    upstream_evidence.append(
        {
            "attack": "flip_later_validation_labels",
            "later_validation_row_ids_sha256": content_hash(
                sorted(row["row_id"] for row in rows if int(row["series_index"]) in later_validation_indexes)
            ),
            "early_calibration_lineage_before_sha256": content_hash(original_early),
            "early_calibration_lineage_after_sha256": content_hash(mutated_early),
            "changed_lineage_folds": changed_lineage_folds,
            "future_serving_raw_model_before_sha256": content_hash(report["full_presealed_refit"]["future_serving_upstream_raw_model"]),
            "future_serving_raw_model_after_sha256": content_hash(mutation_refit["future_serving_upstream_raw_model"]),
            "passed": later_mutation_passed,
        }
    )
    results["GATE_UPSTREAM_TIME_SAFE"] = (
        "upstream raw models use train/validation only and later validation labels cannot rewrite earlier cross-fitted calibration logits",
        upstream_evidence,
        all(fold["source_label_roles"]["raw_model_fit"] == ["train", "validation"] for fold in report["folds"])
        and later_mutation_passed,
    )
    full_refit = report["full_presealed_refit"]
    expected_calibration = sorted(
        {
            str(row["row_id"])
            for fold in config["folds"]
            for row in rows
            if int(row["series_index"]) in range(*fold["calibration"])
        }
    )
    upstream_ids = set(full_refit.get("upstream_train_row_ids", ())) | set(full_refit.get("upstream_validation_row_ids", ()))
    calibration_ids = set(full_refit.get("calibration_row_ids", ()))
    results["GATE_CALIBRATION_ONLY_FIT"] = (
        "final transform support is exactly the deduplicated union of calibration-role rows and excludes upstream support",
        {
            "expected_calibration_row_ids_sha256": content_hash(expected_calibration),
            "actual_calibration_row_ids_sha256": content_hash(full_refit.get("calibration_row_ids", ())),
            "calibration_support_sha256": full_refit.get("calibration_support_sha256"),
            "calibration_logit_lineage_sha256": full_refit.get("calibration_logit_lineage_sha256"),
            "upstream_support_sha256": full_refit.get("upstream_support_sha256"),
            "overlap_count": len(upstream_ids & calibration_ids),
        },
        full_refit.get("calibration_row_ids") == expected_calibration
        and sorted(item["row_id"] for item in full_refit.get("calibration_logit_lineage", ())) == expected_calibration
        and not upstream_ids & calibration_ids,
    )
    mutated_rows = json.loads(json.dumps(rows_payload))
    test_indexes = {index for fold in config["folds"] for index in range(*fold["test"])}
    for row in mutated_rows["rows"]:
        if int(row["series_index"]) in test_indexes:
            row["observation"]["outcome"] = 1 - int(row["observation"]["outcome"])
    mutated_report = build_outer_calibration_selection_report(config, mutated_rows)
    original_signature = _fit_prediction_signature(report)
    mutated_signature = _fit_prediction_signature(mutated_report)
    results["GATE_TEST_LABEL_BLINDNESS"] = (
        "flipping every outer-test label leaves all fit and prediction bytes unchanged",
        {"original_fit_prediction_sha256": original_signature, "mutated_fit_prediction_sha256": mutated_signature},
        original_signature == mutated_signature,
    )
    reconciliation_passed = True
    reconciled = []
    for result in report["fold_results"]:
        if not result["available"]:
            continue
        block_mean = fmean(float(block["log_loss"]) for block in result["blocks"])
        passed = abs(block_mean - float(result["aggregate"]["log_loss"])) <= 1e-12
        reconciliation_passed = reconciliation_passed and passed
        reconciled.append({"fold_id": result["fold_id"], "family": result["family"], "block_count": len(result["blocks"]), "passed": passed})
    results["GATE_PAIRED_BLOCK_RECONCILIATION"] = (
        "candidate identities match and row-series-block-aggregate losses reconcile",
        reconciled,
        reconciliation_passed and bool(reconciled),
    )
    selection_evidence = report["selection"].get("evidence", ())
    unique_counts = [item["unique_top_level_block_count"] for item in selection_evidence]
    dependence_passed = bool(unique_counts) and min(unique_counts) >= int(config["minimum_top_level_blocks"]) and all(
        item["effective_sample_size"] == item["unique_top_level_block_count"] for item in selection_evidence
    )
    results["GATE_DEPENDENCE_SUPPORT"] = (
        "uncertainty uses one contribution per unique registered DGP shock block with small-cluster correction",
        {
            "dgp_block_shocks_sha256": rows_payload["rng_lineage"]["top_level_block_shocks_sha256"],
            "unique_counts": unique_counts,
            "recurrence": [item["recurrence_by_block"] for item in selection_evidence],
            "degrees_of_freedom": [item["degrees_of_freedom"] for item in selection_evidence],
            "leave_largest_block_sensitivity": [item["leave_largest_block_sensitivity"] for item in selection_evidence],
        },
        dependence_passed,
    )
    eligible = [item["family"] for item in selection_evidence if item["valid"] and item["noninferior"]]
    expected_selected = min(eligible, key=CANDIDATE_ORDER.index) if eligible else None
    actual_selected = report["selection"]["selected_family"]
    simultaneous_rule = report["selection"].get("simultaneous_family_rule", {})
    family_rule_passed = (
        simultaneous_rule.get("comparison_count") == 10
        and simultaneous_rule.get("method") == config["multiplicity_method"]
        and simultaneous_rule.get("familywise_alpha") == config["familywise_one_sided_alpha"]
        and simultaneous_rule.get("adjusted_alpha") == config["familywise_one_sided_alpha"] / 10
        and simultaneous_rule.get("reference_status") == config["reference_limitation"]
        and simultaneous_rule.get("predeclared_pair_identities") == list(PAIR_IDENTITIES)
        and [item["pair_id"] for item in report["selection"].get("all_pairwise_contrasts", ())] == list(PAIR_IDENTITIES)
        and all(item["candidate_comparison_count"] == 10 for item in selection_evidence)
    )
    results["GATE_NONINFERIORITY_BEFORE_SIMPLICITY"] = (
        "selection uses simultaneous Bonferroni-Student-t family bounds against the explicitly random empirical reference before simplicity",
        {
            "eligible": eligible,
            "expected_selected": expected_selected,
            "actual_selected": actual_selected,
            "simultaneous_family_rule": simultaneous_rule,
        },
        expected_selected == actual_selected and family_rule_passed,
    )
    transform_shape_passed = True
    shape_evidence = []
    for stratum_id in sorted(full_refit.get("transforms", {})):
        transform = full_refit["transforms"][stratum_id]
        try:
            _assert_transform_shape(transform["family"], transform["parameters"], float(transform["epsilon"]))
            shape_evidence.append({"stratum_id": stratum_id, "passed": True})
        except ValidationFailure as exc:
            transform_shape_passed = False
            shape_evidence.append({"stratum_id": stratum_id, "passed": False, "reason": str(exc)})
    optimizer_checks = []
    optimizer_passed = True
    allowed_solvers = {"closed_form_identity", "offset_aware_generalized_pava", "frozen_bounded_grid_search"}
    for result in report["fold_results"]:
        if not result["available"]:
            continue
        for stratum_id in sorted(result["fits"]):
            fit = result["fits"][stratum_id]
            optimizer = fit["optimizer"]
            passed = (
                optimizer.get("success") is True
                and optimizer.get("finite_parameters") is True
                and optimizer.get("finite_objective") is True
                and optimizer.get("gradient_status") == "not_applicable"
                and optimizer.get("solver_class") in allowed_solvers
                and isinstance(optimizer.get("deterministic_tie_rule"), str)
                and isinstance(optimizer.get("feasibility_check"), str)
                and "finite_gradient" not in optimizer
                and optimizer.get("objective_semantics")
                == "literal_served_probability_epsilon_clamped_log_loss"
                and (
                    optimizer.get("kkt_passed") is True
                    if optimizer.get("solver_class") == "offset_aware_generalized_pava"
                    else optimizer.get("kkt_passed") == "not_applicable"
                )
                and (
                    optimizer.get("served_offset_domain") == _served_offset_domain(config)
                    if optimizer.get("solver_class") == "offset_aware_generalized_pava"
                    else optimizer.get("served_offset_domain")
                    == "offset_domain_enforced_clamp_may_remain_active_for_parametric_extremes"
                )
            )
            optimizer_passed = optimizer_passed and passed
            optimizer_checks.append(
                {
                    "fold_id": result["fold_id"],
                    "family": result["family"],
                    "stratum_id": stratum_id,
                    "solver_class": optimizer.get("solver_class"),
                    "gradient_status": optimizer.get("gradient_status"),
                    "objective_semantics": optimizer.get("objective_semantics"),
                    "served_offset_domain": optimizer.get("served_offset_domain"),
                    "passed": passed,
                }
            )
    results["GATE_TRANSFORM_SHAPE"] = (
        "every transform satisfies shape constraints and every solver reports literal finite deterministic evidence without invented gradients",
        {"shape": shape_evidence, "optimizer": optimizer_checks},
        transform_shape_passed
        and optimizer_passed
        and set(full_refit.get("transforms", ())) == {stratum_id for _, stratum_id in OUTPUT_STRATA},
    )
    draft_checks = []
    draft_passed = True
    for stratum_id in ("stratum-draft", "stratum-prefix"):
        transform = full_refit.get("transforms", {}).get(stratum_id)
        if transform is None:
            draft_passed = False
            continue
        for z, offset in ((0.0, 0.0), (1.7, 0.4), (-20.0, 1.2)):
            p = served_probability(
                transform["family"],
                transform["parameters"],
                z,
                offset,
                epsilon=float(transform["epsilon"]),
                maximum_absolute_offset=float(config["maximum_absolute_served_offset"]),
            )
            swapped = served_probability(
                transform["family"],
                transform["parameters"],
                -z,
                -offset,
                epsilon=float(transform["epsilon"]),
                maximum_absolute_offset=float(config["maximum_absolute_served_offset"]),
            )
            passed = abs(p + swapped - 1.0) <= 2e-15 and (z != 0.0 or offset != 0.0 or p == 0.5)
            draft_passed = draft_passed and passed
            draft_checks.append({"stratum_id": stratum_id, "z": z, "offset": offset, "passed": passed})
    results["GATE_DRAFT_OFFSET_COMPOSITION"] = (
        "draft likelihood composes an independent signed offset and complements under joint side swap",
        draft_checks,
        draft_passed,
    )
    parity_count = 0
    parity_passed = True
    for result in report["fold_results"]:
        if not result["available"]:
            continue
        for row in result["rows"]:
            fit = result["fits"][row["stratum_id"]]
            replayed = served_probability(
                result["family"],
                fit["parameters"],
                row["raw_logit"],
                row["offset"],
                epsilon=float(config["epsilon"]),
                maximum_absolute_offset=float(config["maximum_absolute_served_offset"]),
            )
            parity_passed = parity_passed and canonical_json(replayed) == canonical_json(row["probability"])
            parity_count += 1
    results["GATE_RUNTIME_EXACT_PARITY"] = (
        "runtime evaluation exactly reproduces every outer-test probability and the dense/extreme shape grid",
        {"test_probability_count": parity_count, "fold_evidence_sha256": content_hash(report["fold_results"])},
        parity_passed and parity_count > 0,
    )
    internal_hashes = {
        "config_sha256": config["config_sha256"],
        "rows_sha256": rows_payload["rows_sha256"],
        "selection_report_sha256": report["selection_report_sha256"],
        "source_closure_sha256": content_hash(_source_closure()),
    }
    results["GATE_CONTENT_ADDRESSED_CLOSURE"] = (
        "config, generated rows, fold evidence, report, and source closure are content addressed",
        internal_hashes,
        all(isinstance(value, str) and len(value) == 64 for value in internal_hashes.values()),
    )
    constructor_rejected = False
    try:
        OuterCalibrationAuthority(Path("."), {}, {}, {}, {}, {})
    except ValidationFailure:
        constructor_rejected = True
    results["GATE_LOADER_ISSUED_AUTHORITY"] = (
        "public construction is rejected; exact identity is only a process-local honest-interpreter misuse guard with bundle revalidation",
        {
            "constructor_rejected": constructor_rejected,
            "identity_mechanism": "exact_loader_created_process_singleton",
            "bundle_revalidation": "on_every_authenticate_and_serve",
            "threat_model": AUTHORITY_THREAT_MODEL,
        },
        constructor_rejected and AUTHORITY_THREAT_MODEL["hostile_same_process_unforgeability"] is False,
    )
    if set(results) != set(HARD_GATES):
        raise ValidationFailure("hard-gate evaluator has missing or extra predicates")
    gates: dict[str, Any] = {}
    for name in HARD_GATES:
        predicate, evidence, passed = results[name]
        entry = {"predicate": predicate, "passed": bool(passed), "evidence": evidence}
        entry["evidence_sha256"] = content_hash(evidence)
        gates[name] = entry
    failures = [name for name, entry in gates.items() if not entry["passed"]]
    if failures:
        raise ValidationFailure(f"hard-gate predicates failed: {failures}")
    return gates


def _source_closure() -> dict[str, Any]:
    source_path = Path(__file__).resolve()
    checks_path = source_path.with_name("checks.py")
    numpy_path = Path(np.__file__).resolve()
    dependencies = {
        "python": ".".join(map(str, sys.version_info[:3])),
        "numpy": np.__version__,
        "numpy_source_sha256": sha256_bytes(numpy_path.read_bytes()),
    }
    return {
        "sources": {
            "lol_kills/v2/evaluation/outer_calibration.py": sha256_bytes(source_path.read_bytes()),
            "lol_kills/v2/evaluation/checks.py": sha256_bytes(checks_path.read_bytes()),
        },
        "dependencies": dependencies,
        "dependencies_sha256": content_hash(dependencies),
    }


def _function_fingerprint(function: Any) -> str:
    code = function.__code__
    payload = (
        code.co_code
        + repr((code.co_names, code.co_varnames, code.co_argcount, code.co_kwonlyargcount)).encode("utf-8")
        + repr(function.__defaults__).encode("utf-8")
        + repr(function.__kwdefaults__).encode("utf-8")
    )
    return sha256_bytes(payload)


def _assert_runtime_integrity() -> None:
    """Public attested sentinel; authority paths use a closure-private checker."""
    return None


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _deep_thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _deep_thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_deep_thaw(item) for item in value]
    return value


@dataclass(frozen=True, eq=False)
class OuterCalibrationAuthority:
    root: Path
    authority_relative_path: str
    authority_payload: Mapping[str, Any]
    config: Mapping[str, Any]
    rows: Mapping[str, Any]
    selection_report: Mapping[str, Any]
    transforms: Mapping[str, Mapping[str, Any]]

    def __new__(cls, *args: Any, **kwargs: Any) -> "OuterCalibrationAuthority":
        raise ValidationFailure("OuterCalibrationAuthority is loader-issued only")

    def authenticate(self) -> None:
        raise ValidationFailure("outer calibration authority method installation failed closed")

    def probability(self, stratum_id: str, signed_logit: float, offset: float | None = None) -> float:
        raise ValidationFailure("outer calibration authority method installation failed closed")


def _safe_artifact_path(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or candidate.parts[:3] != ("data", "lol", "v2"):
        raise ValidationFailure(f"unsafe authority path: {relative}")
    try:
        root_lstat = os.lstat(root)
    except OSError as exc:
        raise ValidationFailure("authority repository root is missing or unreadable") from exc
    if stat.S_ISLNK(root_lstat.st_mode) or not stat.S_ISDIR(root_lstat.st_mode):
        raise ValidationFailure("authority repository root must be a real directory, not a symlink")
    path = root
    for index, part in enumerate(candidate.parts):
        path = path / part
        try:
            component_stat = os.lstat(path)
        except OSError as exc:
            raise ValidationFailure(f"artifact path component is missing or unreadable: {relative}") from exc
        is_leaf = index == len(candidate.parts) - 1
        if stat.S_ISLNK(component_stat.st_mode):
            raise ValidationFailure(f"symlink path component rejected: {relative}")
        if is_leaf:
            if not stat.S_ISREG(component_stat.st_mode):
                raise ValidationFailure(f"artifact leaf is not a regular file: {relative}")
            if component_stat.st_nlink != 1:
                raise ValidationFailure(f"hardlinked artifact rejected: {relative}")
        elif not stat.S_ISDIR(component_stat.st_mode):
            raise ValidationFailure(f"artifact parent component is not a directory: {relative}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValidationFailure(f"artifact path is missing or unreadable: {relative}") from exc
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise ValidationFailure(f"artifact path escapes repository: {relative}") from exc
    return resolved


def _load_ref(root: Path, ref: Mapping[str, Any]) -> Any:
    if set(ref) != {"path", "sha256"}:
        raise ValidationFailure("authority reference has missing or extra fields")
    path = _safe_artifact_path(root, str(ref["path"]))
    raw = path.read_bytes()
    if sha256_bytes(raw) != ref["sha256"]:
        raise ValidationFailure(f"artifact hash mismatch: {ref['path']}")
    return _strict_json_bytes(raw, label=str(ref["path"]))


def _validate_complete_report(config: Mapping[str, Any], rows: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    rebuilt = build_outer_calibration_selection_report(config, rows)
    if canonical_json(rebuilt) != canonical_json(report):
        raise ValidationFailure("selection report does not exactly replay from frozen inputs")
    if report.get("claim_ceiling") != CLAIM_CEILING:
        raise ValidationFailure("claim ceiling mismatch")
    if tuple(report.get("candidate_order", ())) != CANDIDATE_ORDER:
        raise ValidationFailure("candidate order mismatch")


def _load_outer_calibration_authority_impl(
    root: Path | str,
    authority_relative_path: str = "data/lol/v2/evaluation/b2/outer-calibration-authority.json",
) -> OuterCalibrationAuthority:
    root = Path(root)
    authority_path = _safe_artifact_path(root, authority_relative_path)
    authority = _strict_json_bytes(authority_path.read_bytes(), label=authority_relative_path)
    if (
        authority.get("schema_version") != SCHEMA_VERSION
        or authority.get("claim_ceiling") != CLAIM_CEILING
        or authority.get("authority_threat_model") != AUTHORITY_THREAT_MODEL
    ):
        raise ValidationFailure("outer calibration authority schema or claim ceiling mismatch")
    if set(authority.get("hard_gates", {})) != set(HARD_GATES):
        raise ValidationFailure("hard gates have missing or extra predicates")
    for name, entry in authority["hard_gates"].items():
        if set(entry) != {"predicate", "passed", "evidence", "evidence_sha256"} or entry["passed"] is not True:
            raise ValidationFailure(f"{name}: hard gate is false or malformed")
        if content_hash(entry["evidence"]) != entry["evidence_sha256"]:
            raise ValidationFailure(f"{name}: hard-gate evidence self-hash mismatch")
    closure = _source_closure()
    if authority.get("source_closure") != closure:
        raise ValidationFailure("source closure or dependency substitution detected")
    config = _load_ref(root, authority["refs"]["config"])
    _served_offset_domain(config)
    registry_path = _safe_artifact_path(root, "data/lol/v2/evaluation/b2/calibration-candidate-registry.json")
    registry_bytes = registry_path.read_bytes()
    registry = _strict_json_bytes(registry_bytes, label="candidate registry")
    if sha256_bytes(registry_bytes) != config.get("candidate_registry_sha256"):
        raise ValidationFailure("immutable candidate registry bytes changed")
    if tuple(item.get("family") for item in registry.get("candidates", ())) != CANDIDATE_ORDER:
        raise ValidationFailure("candidate registry family order changed")
    rows = _load_ref(root, authority["refs"]["rows"])
    report = _load_ref(root, authority["refs"]["selection_report"])
    _validate_complete_report(config, rows, report)
    replayed_gates = _evaluate_hard_gates(config, rows, report)
    if canonical_json(replayed_gates) != canonical_json(authority["hard_gates"]):
        differing = [name for name in HARD_GATES if canonical_json(replayed_gates[name]) != canonical_json(authority["hard_gates"][name])]
        raise ValidationFailure(f"hard-gate evidence does not exactly replay: {differing}")
    transforms: dict[str, Mapping[str, Any]] = {}
    expected_strata = {stratum_id for _, stratum_id in OUTPUT_STRATA}
    if set(authority["refs"]["transforms"]) != expected_strata:
        raise ValidationFailure("transform references do not exactly cover registered strata")
    for stratum_id, ref in authority["refs"]["transforms"].items():
        transform = _load_ref(root, ref)
        expected = report["full_presealed_refit"]["transforms"].get(stratum_id)
        if transform.get("served_transform") != expected:
            raise ValidationFailure(f"{stratum_id}: served transform does not match replayed selection")
        if transform.get("selection_report_sha256") != report["selection_report_sha256"]:
            raise ValidationFailure(f"{stratum_id}: selection report binding mismatch")
        _assert_transform_shape(transform["served_transform"]["family"], transform["served_transform"]["parameters"], float(transform["served_transform"]["epsilon"]))
        transforms[stratum_id] = transform["served_transform"]
    obj = object.__new__(OuterCalibrationAuthority)
    object.__setattr__(obj, "root", root.resolve())
    object.__setattr__(obj, "authority_relative_path", authority_relative_path)
    object.__setattr__(obj, "authority_payload", _deep_freeze(authority))
    object.__setattr__(obj, "config", _deep_freeze(config))
    object.__setattr__(obj, "rows", _deep_freeze(rows))
    object.__setattr__(obj, "selection_report", _deep_freeze(report))
    object.__setattr__(obj, "transforms", _deep_freeze(transforms))
    return obj


def _reauthenticate_loaded_authority(authority: OuterCalibrationAuthority) -> None:
    candidate = _load_outer_calibration_authority_impl(authority.root, authority.authority_relative_path)
    for name in ("authority_payload", "config", "rows", "selection_report", "transforms"):
        if canonical_json(_deep_thaw(getattr(authority, name))) != canonical_json(_deep_thaw(getattr(candidate, name))):
            raise ValidationFailure(f"loaded authority mutable projection changed: {name}")


def _replay_outer_calibration_impl(authority: OuterCalibrationAuthority) -> dict[str, Any]:
    authority.authenticate()
    config = _deep_thaw(authority.config)
    rows_payload = _deep_thaw(authority.rows)
    selection_report = _deep_thaw(authority.selection_report)
    _validate_complete_report(config, rows_payload, selection_report)
    parity_rows = 0
    for result in selection_report["fold_results"]:
        if not result["available"]:
            continue
        for row in result["rows"]:
            fit = result["fits"][row["stratum_id"]]
            actual = served_probability(
                result["family"],
                fit["parameters"],
                row["raw_logit"],
                row["offset"],
                epsilon=float(config["epsilon"]),
                maximum_absolute_offset=float(config["maximum_absolute_served_offset"]),
            )
            if canonical_json(actual) != canonical_json(row["probability"]):
                raise ValidationFailure("runtime/Python test-row probability parity failed")
            parity_rows += 1
    for transform in authority.transforms.values():
        _assert_transform_shape(transform["family"], transform["parameters"], float(transform["epsilon"]))
    return {
        "status": "PASS_SYNTHETIC_MECHANICS_ONLY",
        "selection_status": selection_report["selection"]["status"],
        "selected_family": selection_report["selection"]["selected_family"],
        "parity_row_count": parity_rows,
        "claim_ceiling": CLAIM_CEILING,
    }


def _write_outer_calibration_artifacts_impl(root: Path | str, *, regime: str = "nonlinear") -> dict[str, str]:
    root = Path(root).resolve()
    registry_path = root / "data/lol/v2/evaluation/b2/calibration-candidate-registry.json"
    registry_bytes = registry_path.read_bytes()
    config = build_outer_calibration_config(registry_bytes, regime=regime)
    rows = build_outer_calibration_rows(config)
    report = build_outer_calibration_selection_report(config, rows)
    if report["selection"]["selected_family"] is None:
        raise ValidationFailure("default artifact build requires available synthetic development selection")
    base = root / "data/lol/v2/evaluation/b2"
    transform_dir = base / "outer-calibration/transforms"
    transform_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {
        "data/lol/v2/evaluation/b2/outer-calibration-config.json": config,
        "data/lol/v2/evaluation/b2/outer-calibration-rows.json": rows,
        "data/lol/v2/evaluation/b2/outer-calibration-selection-report.json": report,
    }
    transform_paths: dict[str, str] = {}
    for stratum_id, transform in report["full_presealed_refit"]["transforms"].items():
        relative = f"data/lol/v2/evaluation/b2/outer-calibration/transforms/{stratum_id}.json"
        payload = {
            "artifact_id": f"scryglass:b2:outer-calibration-transform:{stratum_id}:v1",
            "schema_version": SCHEMA_VERSION,
            "model_id": MODEL_ID,
            "candidate_id": transform["family"],
            "registry_sha256": config["candidate_registry_sha256"],
            "split_ids": [fold["fold_id"] for fold in config["folds"]],
            "fold_evidence_sha256": content_hash(report["fold_results"]),
            "config_sha256": config["config_sha256"],
            "rows_sha256": rows["rows_sha256"],
            "selection_report_sha256": report["selection_report_sha256"],
            "source_closure": _source_closure(),
            "served_transform": transform,
            "claim_ceiling": CLAIM_CEILING,
            "authority_threat_model": AUTHORITY_THREAT_MODEL,
        }
        artifacts[relative] = payload
        transform_paths[stratum_id] = relative
    hashes: dict[str, str] = {}
    for relative, payload in artifacts.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = canonical_json(payload)
        path.write_bytes(raw)
        hashes[relative] = sha256_bytes(raw)
    gates = _evaluate_hard_gates(config, rows, report)
    authority_payload = {
        "artifact_id": "scryglass:b2:outer-calibration-authority:v1",
        "schema_version": SCHEMA_VERSION,
        "synthetic": True,
        "contract_tree_sha256": CONTRACT_TREE_SHA256,
        "source_closure": _source_closure(),
        "hard_gates": gates,
        "refs": {
            "config": {"path": "data/lol/v2/evaluation/b2/outer-calibration-config.json", "sha256": hashes["data/lol/v2/evaluation/b2/outer-calibration-config.json"]},
            "rows": {"path": "data/lol/v2/evaluation/b2/outer-calibration-rows.json", "sha256": hashes["data/lol/v2/evaluation/b2/outer-calibration-rows.json"]},
            "selection_report": {"path": "data/lol/v2/evaluation/b2/outer-calibration-selection-report.json", "sha256": hashes["data/lol/v2/evaluation/b2/outer-calibration-selection-report.json"]},
            "transforms": {key: {"path": path, "sha256": hashes[path]} for key, path in transform_paths.items()},
        },
        "claim_ceiling": CLAIM_CEILING,
        "authority_threat_model": AUTHORITY_THREAT_MODEL,
    }
    authority_relative = "data/lol/v2/evaluation/b2/outer-calibration-authority.json"
    authority_raw = canonical_json(authority_payload)
    (root / authority_relative).write_bytes(authority_raw)
    hashes[authority_relative] = sha256_bytes(authority_raw)
    return hashes


def _make_authority_identity() -> tuple[Any, Any, Any, Any]:
    singleton: list[OuterCalibrationAuthority | None] = [None]
    pinned_snapshot: list[Any] = [None]
    checker: list[Any] = [lambda: (_ for _ in ()).throw(ValidationFailure("authority checker is not installed"))]

    def snapshot(authority: OuterCalibrationAuthority) -> tuple[Any, ...]:
        return (
            str(authority.root),
            authority.authority_relative_path,
            *(content_hash(_deep_thaw(getattr(authority, name))) for name in ("authority_payload", "config", "rows", "selection_report", "transforms")),
        )

    def install_checker(value: Any) -> None:
        if checker[0] is not value and getattr(checker[0], "__name__", "") != "<lambda>":
            raise ValidationFailure("authority checker is already installed")
        checker[0] = value

    def register(candidate: OuterCalibrationAuthority) -> OuterCalibrationAuthority:
        checker[0]()
        candidate_snapshot = snapshot(candidate)
        if singleton[0] is None:
            singleton[0] = candidate
            pinned_snapshot[0] = candidate_snapshot
        elif candidate_snapshot != pinned_snapshot[0]:
            raise ValidationFailure("exact-identity singleton already binds different authority bytes")
        return singleton[0]

    def authenticate(self: OuterCalibrationAuthority) -> None:
        checker[0]()
        if self is not singleton[0]:
            raise ValidationFailure("authority is not the exact loader-created process singleton")
        if snapshot(self) != pinned_snapshot[0]:
            raise ValidationFailure("authority singleton projection or pinned root changed")
        _reauthenticate_loaded_authority(self)

    def probability(
        self: OuterCalibrationAuthority,
        stratum_id: str,
        signed_logit: float,
        offset: float | None = None,
    ) -> float:
        authenticate(self)
        if stratum_id not in self.transforms:
            raise ValidationFailure(f"unknown served stratum: {stratum_id}")
        transform = self.transforms[stratum_id]
        if transform["output_class"] in OFFSET_STRATA and offset is None:
            raise ValidationFailure("draft and partial-draft served probabilities require an independent offset")
        use_offset = (
            _validate_served_offset_value(
                offset,
                self.config,
                label=f"{stratum_id}: authority served offset",
            )
            if transform["output_class"] in OFFSET_STRATA
            else 0.0
        )
        return served_probability(
            transform["family"],
            transform["parameters"],
            signed_logit,
            use_offset,
            epsilon=float(transform["epsilon"]),
            maximum_absolute_offset=float(self.config["maximum_absolute_served_offset"]),
        )

    return authenticate, probability, register, install_checker


def _make_runtime_attestor(namespace: dict[str, Any]) -> Any:
    function_names = (
        "canonical_json",
        "sha256_bytes",
        "content_hash",
        "_strict_json_bytes",
        "_logit",
        "_sigmoid",
        "_loss",
        "_brier",
        "build_outer_calibration_config",
        "build_outer_calibration_rows",
        "_validate_rows",
        "_partition",
        "_fit_raw_model",
        "_transform_input",
        "_served_offset_for_row",
        "_pava",
        "_fit_offset_block_theta",
        "_fit_offset_aware_isotonic",
        "_fit_family",
        "apply_outer_transform",
        "served_probability",
        "_assert_transform_shape",
        "_candidate_fold_evidence",
        "_student_t_critical_95_one_sided",
        "_simultaneous_t_critical",
        "_select_family",
        "_deduplicate_rows",
        "_build_full_presealed_refit",
        "build_outer_calibration_selection_report",
        "_fit_prediction_signature",
        "_evaluate_hard_gates",
        "_source_closure",
        "_function_fingerprint",
        "_assert_runtime_integrity",
        "_deep_freeze",
        "_deep_thaw",
        "_safe_artifact_path",
        "_load_ref",
        "_validate_complete_report",
        "_load_outer_calibration_authority_impl",
        "_reauthenticate_loaded_authority",
        "_replay_outer_calibration_impl",
        "_write_outer_calibration_artifacts_impl",
        "_make_runtime_attestor",
        "_make_authority_identity",
        "_make_guarded_entries",
    )
    originals = {name: namespace[name] for name in function_names}
    code_objects = {name: originals[name].__code__ for name in function_names}
    defaults = {name: (repr(originals[name].__defaults__), repr(originals[name].__kwdefaults__)) for name in function_names}
    method_originals = {
        "OuterCalibrationAuthority.__new__": OuterCalibrationAuthority.__new__,
        "OuterCalibrationAuthority.authenticate": OuterCalibrationAuthority.authenticate,
        "OuterCalibrationAuthority.probability": OuterCalibrationAuthority.probability,
    }
    method_codes = {name: function.__code__ for name, function in method_originals.items()}
    method_defaults = {name: (repr(function.__defaults__), repr(function.__kwdefaults__)) for name, function in method_originals.items()}
    registry_names = (
        "CONTRACT_TREE_SHA256",
        "SCHEMA_VERSION",
        "MODEL_ID",
        "CANDIDATE_ORDER",
        "PAIR_IDENTITIES",
        "OUTPUT_STRATA",
        "OFFSET_STRATA",
        "CLAIM_CEILING",
        "AUTHORITY_THREAT_MODEL",
        "HARD_GATES",
        "DEFAULT_CONFIG",
    )
    frozen_dumps = json.dumps
    frozen_registry = {
        name: frozen_dumps(namespace[name], sort_keys=True, separators=(",", ":"), default=sorted)
        for name in registry_names
    }
    dependency_objects = {
        "json.loads": json.loads,
        "json.dumps": json.dumps,
        "hashlib.sha256": hashlib.sha256,
        "Path.resolve": Path.resolve,
        "os.stat": os.stat,
        "os.lstat": os.lstat,
        "stat.module": stat,
        "np.random.PCG64": np.random.PCG64,
        "inspect.isfunction": inspect.isfunction,
        "statistics.fmean": fmean,
        "math.exp": math.exp,
        "math.log": math.log,
        "math.log1p": math.log1p,
        "math.isfinite": math.isfinite,
        "math.sqrt": math.sqrt,
        "math.tanh": math.tanh,
        "datetime.class": datetime,
        "Path.read_bytes": Path.read_bytes,
        "Path.write_bytes": Path.write_bytes,
        "MappingProxyType": MappingProxyType,
    }
    frozen_isfunction = inspect.isfunction

    def attest() -> None:
        for name, original in originals.items():
            current = namespace.get(name)
            if current is not original or not frozen_isfunction(current):
                raise ValidationFailure(f"runtime callable rebound: {name}")
            if current.__code__ is not code_objects[name] or (
                repr(current.__defaults__),
                repr(current.__kwdefaults__),
            ) != defaults[name]:
                raise ValidationFailure(f"runtime code/default/kwdefault mutation: {name}")
        current_methods = {
            "OuterCalibrationAuthority.__new__": OuterCalibrationAuthority.__new__,
            "OuterCalibrationAuthority.authenticate": OuterCalibrationAuthority.authenticate,
            "OuterCalibrationAuthority.probability": OuterCalibrationAuthority.probability,
        }
        for name, original in method_originals.items():
            current = current_methods[name]
            if current is not original or current.__code__ is not method_codes[name] or (
                repr(current.__defaults__),
                repr(current.__kwdefaults__),
            ) != method_defaults[name]:
                raise ValidationFailure(f"runtime authority method mutation: {name}")
        for name, frozen in frozen_registry.items():
            current = frozen_dumps(namespace.get(name), sort_keys=True, separators=(",", ":"), default=sorted)
            if current != frozen:
                raise ValidationFailure(f"runtime registry mutation: {name}")
        current_dependencies = {
            "json.loads": json.loads,
            "json.dumps": json.dumps,
            "hashlib.sha256": hashlib.sha256,
            "Path.resolve": Path.resolve,
            "os.stat": os.stat,
            "os.lstat": os.lstat,
            "stat.module": stat,
            "np.random.PCG64": np.random.PCG64,
            "inspect.isfunction": inspect.isfunction,
            "statistics.fmean": fmean,
            "math.exp": math.exp,
            "math.log": math.log,
            "math.log1p": math.log1p,
            "math.isfinite": math.isfinite,
            "math.sqrt": math.sqrt,
            "math.tanh": math.tanh,
            "datetime.class": datetime,
            "Path.read_bytes": Path.read_bytes,
            "Path.write_bytes": Path.write_bytes,
            "MappingProxyType": MappingProxyType,
        }
        for name, original in dependency_objects.items():
            if current_dependencies[name] is not original:
                raise ValidationFailure(f"runtime dependency substitution: {name}")

    return attest


def _make_guarded_entries(checker: Any, register: Any, loader_impl: Any, replay_impl: Any, writer_impl: Any) -> tuple[Any, Any, Any]:
    def guarded_loader(
        root: Path | str,
        authority_relative_path: str = "data/lol/v2/evaluation/b2/outer-calibration-authority.json",
    ) -> OuterCalibrationAuthority:
        checker()
        candidate = loader_impl(root, authority_relative_path)
        authority = register(candidate)
        authority.authenticate()
        return authority

    def guarded_replay(authority: OuterCalibrationAuthority) -> dict[str, Any]:
        checker()
        return replay_impl(authority)

    def guarded_writer(root: Path | str, *, regime: str = "nonlinear") -> dict[str, str]:
        checker()
        return writer_impl(root, regime=regime)

    return guarded_loader, guarded_replay, guarded_writer


_IDENTITY_AUTHENTICATE, _IDENTITY_PROBABILITY, _IDENTITY_REGISTER, _INSTALL_CHECKER = _make_authority_identity()
OuterCalibrationAuthority.authenticate = _IDENTITY_AUTHENTICATE
OuterCalibrationAuthority.probability = _IDENTITY_PROBABILITY
_PRIVATE_CHECKER = _make_runtime_attestor(globals())
_INSTALL_CHECKER(_PRIVATE_CHECKER)
load_outer_calibration_authority, replay_outer_calibration, write_outer_calibration_artifacts = _make_guarded_entries(
    _PRIVATE_CHECKER,
    _IDENTITY_REGISTER,
    _load_outer_calibration_authority_impl,
    _replay_outer_calibration_impl,
    _write_outer_calibration_artifacts_impl,
)
del _PRIVATE_CHECKER
del _IDENTITY_AUTHENTICATE, _IDENTITY_PROBABILITY, _IDENTITY_REGISTER, _INSTALL_CHECKER
_AUTHORITY_PRIVATE_CHECKER = _assert_runtime_integrity
