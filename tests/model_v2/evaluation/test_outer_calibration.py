from __future__ import annotations

import copy
import functools
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from scipy.optimize import brentq

from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation import outer_calibration as outer
from lol_kills.v2.evaluation.outer_calibration import (
    CANDIDATE_ORDER,
    CLAIM_CEILING,
    HARD_GATES,
    OUTPUT_STRATA,
    OuterCalibrationAuthority,
    _build_full_presealed_refit,
    _fit_family,
    _fit_raw_model,
    _partition,
    _select_family,
    _transform_input,
    apply_outer_transform,
    build_outer_calibration_config,
    build_outer_calibration_rows,
    build_outer_calibration_selection_report,
    canonical_json,
    load_outer_calibration_authority,
    replay_outer_calibration,
    served_probability,
)


ROOT = Path(__file__).resolve().parents[3]
REGISTRY = ROOT / "data/lol/v2/evaluation/b2/calibration-candidate-registry.json"


@functools.lru_cache(maxsize=None)
def _bundle(regime: str = "nonlinear"):
    config = build_outer_calibration_config(REGISTRY.read_bytes(), regime=regime)
    rows = build_outer_calibration_rows(config)
    report = build_outer_calibration_selection_report(config, rows)
    return config, rows, report


def test_cached_bundle_reuses_only_byte_exact_frozen_inputs():
    first = _bundle()
    second = _bundle()
    assert first is second
    assert [canonical_json(value) for value in first] == [
        canonical_json(value) for value in second
    ]
    regenerated_config = build_outer_calibration_config(REGISTRY.read_bytes())
    regenerated_rows = build_outer_calibration_rows(regenerated_config)
    regenerated_report = build_outer_calibration_selection_report(
        regenerated_config, regenerated_rows
    )
    assert [canonical_json(value) for value in first] == [
        canonical_json(value)
        for value in (regenerated_config, regenerated_rows, regenerated_report)
    ]


def test_generator_is_exactly_deterministic_and_separates_truth_observation_inputs():
    config = build_outer_calibration_config(REGISTRY.read_bytes())
    first = build_outer_calibration_rows(config)
    second = build_outer_calibration_rows(config)
    assert canonical_json(first) == canonical_json(second)
    assert {row["output_class"] for row in first["rows"]} == {item[0] for item in OUTPUT_STRATA}
    assert all(set(row) >= {"generator_truth", "observation", "features", "timestamps"} for row in first["rows"])
    assert first["rng_lineage"]["algorithm"] == "PCG64"


def test_rolling_folds_are_atomic_and_strictly_chronological():
    config, payload, _ = _bundle()
    for fold in config["folds"]:
        parts = _partition(payload["rows"], fold)
        role_series = [{row["series_id"] for row in parts[role]} for role in ("train", "validation", "calibration", "test")]
        assert all(left.isdisjoint(right) for i, left in enumerate(role_series) for right in role_series[i + 1 :])
        bounds = [
            (
                min(row["timestamps"]["event_at"] for row in parts[role]),
                max(row["timestamps"]["event_at"] for row in parts[role]),
            )
            for role in ("train", "validation", "calibration", "test")
        ]
        assert all(left[1] < right[0] for left, right in zip(bounds, bounds[1:]))


def test_all_families_are_open_monotone_complement_symmetric():
    _, _, report = _bundle()
    for result in report["fold_results"]:
        assert result["available"]
        for fit in result["fits"].values():
            family, params = fit["family"], fit["parameters"]
            grid = [-1e6, -100, -20, -5, -1, 0, 1, 5, 20, 100, 1e6]
            values = [apply_outer_transform(family, params, z) for z in grid]
            assert all(0 < value < 1 and math.isfinite(value) for value in values)
            assert values == sorted(values)
            assert apply_outer_transform(family, params, 0) == 0.5
            for z in (0, 0.1, 1, 20, 1e6):
                assert apply_outer_transform(family, params, z) + apply_outer_transform(family, params, -z) == pytest.approx(1.0, abs=2e-15)


def test_draft_offset_composition_side_swap_and_zero():
    config, _, report = _bundle()
    transform = next(iter(report["full_presealed_refit"]["transforms"].values()))
    family, params = transform["family"], transform["parameters"]
    maximum = config["maximum_absolute_served_offset"]
    assert served_probability(
        family, params, 0.0, 0.0, maximum_absolute_offset=maximum
    ) == 0.5
    for z, offset in ((1.2, 0.4), (-0.7, 1.1), (20, -2)):
        p = served_probability(
            family, params, z, offset, maximum_absolute_offset=maximum
        )
        assert served_probability(
            family,
            params,
            -z,
            -offset,
            maximum_absolute_offset=maximum,
        ) == pytest.approx(1 - p, abs=2e-15)


def test_symmetrized_beta_is_canonical_beta_map_with_identity_member():
    identity_parameters = {"a": 1.0, "b": 1.0, "c": 0.0}
    for z in (-20.0, -3.0, -0.2, 0.0, 0.2, 3.0, 20.0):
        assert apply_outer_transform("symmetrized_beta", identity_parameters, z) == pytest.approx(
            apply_outer_transform("identity", {}, z),
            abs=2e-15,
        )
    config, rows, report = _bundle("nonlinear")
    beta_results = [item for item in report["fold_results"] if item["family"] == "symmetrized_beta"]
    assert beta_results and all(item["available"] for item in beta_results)
    for result in beta_results:
        for fit in result["fits"].values():
            assert set(fit["parameters"]) == {"a", "b", "c"}
            assert fit["parameters"]["a"] >= 0 and fit["parameters"]["b"] >= 0
            assert fit["optimizer"]["success"] and fit["optimizer"]["finite_parameters"]
            assert fit["optimizer"]["finite_objective"] and fit["optimizer"]["gradient_status"] == "not_applicable"
            assert fit["optimizer"]["solver_class"] == "frozen_bounded_grid_search"
            assert "finite_gradient" not in fit["optimizer"]
            assert fit["support"]["classes"] == [0, 1]
    assert config["method_provenance"][0]["venue"] == "AISTATS, PMLR 54"
    assert any(item.get("doi") == "10.18637/JSS.V032.I05" for item in config["method_provenance"])


def test_offset_aware_fit_matches_independent_temperature_grid_oracle():
    config, payload, _ = _bundle()
    parts = _partition(payload["rows"], config["folds"][0])
    raw_model = _fit_raw_model(parts["train"], parts["validation"], config)
    draft_rows = [row for row in parts["calibration"] if row["output_class"] == "draft_score"]
    logits = [_transform_input(row, raw_model) for row in draft_rows]
    fit = _fit_family("symmetric_temperature", draft_rows, logits, config)
    epsilon = config["epsilon"]

    def logistic(value):
        return 1.0 / (1.0 + math.exp(-value))

    def oracle_loss(scale):
        losses = []
        for row, z in zip(draft_rows, logits):
            p = min(1 - epsilon, max(epsilon, logistic(row["features"]["league_offset"] + scale * z)))
            y = row["observation"]["outcome"]
            losses.append(-(y * math.log(p) + (1 - y) * math.log1p(-p)))
        return sum(losses) / len(losses)

    scales = [round(0.25 + 0.025 * index, 6) for index in range(91)]
    expected_scale = min(scales, key=oracle_loss)
    assert fit["parameters"]["scale"] == expected_scale
    assert fit["optimizer"]["objective_log_loss"] == pytest.approx(oracle_loss(expected_scale), abs=1e-15)


def test_hostile_offset_mutation_changes_draft_fits_but_not_zero_offset_outputs():
    config, payload, _ = _bundle()
    parts = _partition(payload["rows"], config["folds"][0])
    raw_model = _fit_raw_model(parts["train"], parts["validation"], config)
    for output_class in ("draft_score", "partial_draft_state"):
        rows = [row for row in parts["calibration"] if row["output_class"] == output_class]
        logits = [_transform_input(row, raw_model) for row in rows]
        original = _fit_family("symmetric_temperature", rows, logits, config)
        attacked = copy.deepcopy(rows)
        for row, z in zip(attacked, logits):
            row["features"]["league_offset"] = -2.0 if z > 0.0 else 2.0
        mutated = _fit_family("symmetric_temperature", attacked, logits, config)
        assert original["parameters"] != mutated["parameters"]
    player_rows = [row for row in parts["calibration"] if row["output_class"] == "player_rating"]
    player_logits = [_transform_input(row, raw_model) for row in player_rows]
    baseline = _fit_family("symmetric_temperature", player_rows, player_logits, config)
    zero_semantics_attack = copy.deepcopy(player_rows)
    for row in zero_semantics_attack:
        row["features"]["league_offset"] = 999.0
    assert _fit_family("symmetric_temperature", zero_semantics_attack, player_logits, config)["parameters"] == baseline["parameters"]


def test_fold_selection_fit_and_runtime_share_exact_served_objective():
    config, payload, report = _bundle()
    fold = config["folds"][0]
    parts = _partition(payload["rows"], fold)
    raw_model = _fit_raw_model(parts["train"], parts["validation"], config)
    result = next(
        item
        for item in report["fold_results"]
        if item["fold_id"] == fold["fold_id"] and item["family"] == "symmetric_temperature"
    )
    for stratum_id, fit in result["fits"].items():
        rows = [row for row in parts["calibration"] if row["stratum_id"] == stratum_id]
        losses = []
        for row in rows:
            z = _transform_input(row, raw_model)
            offset = row["features"]["league_offset"] if row["output_class"] in {"draft_score", "partial_draft_state"} else 0.0
            p = served_probability(
                "symmetric_temperature",
                fit["parameters"],
                z,
                offset,
                maximum_absolute_offset=config["maximum_absolute_served_offset"],
            )
            y = row["observation"]["outcome"]
            losses.append(-(y * math.log(p) + (1 - y) * math.log1p(-p)))
        assert fit["optimizer"]["objective_log_loss"] == pytest.approx(sum(losses) / len(losses), abs=1e-15)


def test_offset_aware_isotonic_matches_independent_partition_oracle_and_zero_offset_reduction():
    config = build_outer_calibration_config(REGISTRY.read_bytes())
    rows = []
    logits = []
    for knot_index in range(6):
        offset = 1.8 - 0.75 * knot_index
        oriented_cases = ((1.0, 0), (-1.0, 1), (1.0, 1), (-1.0, 0))
        for orientation_index, (sign, oriented_outcome) in enumerate(oriented_cases):
            outcome = oriented_outcome if sign > 0 else 1 - oriented_outcome
            rows.append(
                {
                        "row_id": f"hostile:{knot_index}:{orientation_index}",
                    "series_id": f"hostile-series-{knot_index}",
                    "top_level_block_id": f"generator-block-{knot_index}",
                    "output_class": "draft_score",
                    "stratum_id": "stratum-draft",
                    "features": {"league_offset": offset if sign > 0 else -offset, "signed_strength": sign * float(knot_index + 1)},
                    "observation": {"outcome": outcome},
                }
            )
            logits.append(sign * float(knot_index + 1))
    fit = _fit_family("symmetrized_bounded_isotonic", rows, logits, config)
    repeated = _fit_family("symmetrized_bounded_isotonic", rows, logits, config)
    assert canonical_json(fit) == canonical_json(repeated)
    assert fit["optimizer"]["solver_class"] == "offset_aware_generalized_pava"
    assert fit["optimizer"]["gradient_status"] == "not_applicable"
    assert fit["optimizer"]["kkt_passed"] is True
    assert all(block["kkt_passed"] for block in fit["parameters"]["block_diagnostics"])

    upper = config["isotonic_theta_upper_bound"]

    def block_root(indexes):
        selected = []
        for index in indexes:
            for row_index in range(4 * index, 4 * index + 4):
                row = rows[row_index]
                z = logits[row_index]
                selected.append(
                    (
                        row["observation"]["outcome"] if z >= 0 else 1 - row["observation"]["outcome"],
                        row["features"]["league_offset"] if z >= 0 else -row["features"]["league_offset"],
                    )
                )

        def gradient(theta):
            return sum(1.0 / (1.0 + math.exp(-(offset + theta))) - outcome for outcome, offset in selected)

        if gradient(0.0) >= 0.0:
            return 0.0
        if gradient(upper) <= 0.0:
            return upper
        return brentq(gradient, 0.0, upper, xtol=1e-14, rtol=1e-14)

    oracle_candidates = []
    for cut_mask in range(1 << 5):
        partitions = []
        start = 0
        for boundary in range(5):
            if cut_mask & (1 << boundary):
                partitions.append(list(range(start, boundary + 1)))
                start = boundary + 1
        partitions.append(list(range(start, 6)))
        block_thetas = [block_root(partition) for partition in partitions]
        if any(right + 1e-12 < left for left, right in zip(block_thetas, block_thetas[1:])):
            continue
        theta_by_knot = [0.0] * 6
        for partition, theta in zip(partitions, block_thetas):
            for index in partition:
                theta_by_knot[index] = theta
        losses = []
        for row, z in zip(rows, logits):
            theta = theta_by_knot[int(abs(z)) - 1]
            signed_theta = theta if z >= 0 else -theta
            p = served_probability(
                "symmetrized_bounded_isotonic",
                {
                    "knots": [float(index + 1) for index in range(6)],
                    "theta_values": theta_by_knot,
                    "theta_upper_bound": upper,
                    "root_iterations": config["isotonic_root_iterations"],
                    "block_diagnostics": [],
                    "served_offset_domain": config["served_offset_domain"],
                },
                z,
                row["features"]["league_offset"],
                epsilon=config["epsilon"],
                maximum_absolute_offset=config["maximum_absolute_served_offset"],
            )
            y = row["observation"]["outcome"]
            losses.append(-(y * math.log(p) + (1 - y) * math.log1p(-p)))
        oracle_candidates.append((sum(losses) / len(losses), theta_by_knot))
    oracle_loss, oracle_theta = min(oracle_candidates, key=lambda item: (item[0], item[1]))
    assert fit["parameters"]["theta_values"] == pytest.approx(oracle_theta, abs=2e-12)
    assert fit["optimizer"]["objective_log_loss"] == pytest.approx(oracle_loss, abs=2e-12)
    assert any(theta > 0.5 for theta in fit["parameters"]["theta_values"])

    zero_rows = copy.deepcopy(rows)
    for row in zero_rows:
        row["features"]["league_offset"] = 0.0
    zero_fit = _fit_family("symmetrized_bounded_isotonic", zero_rows, logits, config)
    assert zero_fit["parameters"]["theta_values"] == pytest.approx([0.0] * 6, abs=1e-15)
    fixed_zero_rows = copy.deepcopy(rows)
    fixed_zero_logits = list(logits)
    for outcome, offset in (
        (0, -config["maximum_absolute_served_offset"]),
        (1, config["maximum_absolute_served_offset"]),
    ):
        fixed_zero_rows.append(
            {
                "row_id": f"fixed-zero:{outcome}",
                "series_id": "hostile-series-0",
                "top_level_block_id": "generator-block-0",
                "output_class": "draft_score",
                "stratum_id": "stratum-draft",
                "features": {"league_offset": offset, "signed_strength": 0.0},
                "observation": {"outcome": outcome},
            }
        )
        fixed_zero_logits.append(0.0)
    fixed_zero_fit = _fit_family("symmetrized_bounded_isotonic", fixed_zero_rows, fixed_zero_logits, config)
    assert fixed_zero_fit["parameters"] == fit["parameters"]
    for z in (-6.0, -2.5, 0.0, 2.5, 6.0):
        p = apply_outer_transform("symmetrized_bounded_isotonic", fit["parameters"], z)
        assert apply_outer_transform("symmetrized_bounded_isotonic", fit["parameters"], -z) == pytest.approx(1 - p, abs=2e-15)


def test_reviewer_unclipped_objective_counterexample_is_outside_registered_domain():
    config = build_outer_calibration_config(REGISTRY.read_bytes())
    rows = []
    logits = []
    for knot_index in range(6):
        for case_index, (outcome, offset) in enumerate(((0, 100.0), (1, -1.0))):
            rows.append(
                {
                    "row_id": f"reviewer:{knot_index}:{case_index}",
                    "series_id": f"reviewer-series-{knot_index}",
                    "top_level_block_id": f"generator-block-{knot_index}",
                    "output_class": "draft_score",
                    "stratum_id": "stratum-draft",
                    "features": {
                        "league_offset": offset,
                        "signed_strength": float(knot_index + 1),
                    },
                    "observation": {"outcome": outcome},
                }
            )
            logits.append(float(knot_index + 1))
    with pytest.raises(ValidationFailure, match="registered maximum"):
        _fit_family("symmetrized_bounded_isotonic", rows, logits, config)


def test_served_offset_domain_boundary_both_signs_and_joint_side_swap():
    config = build_outer_calibration_config(REGISTRY.read_bytes())
    maximum = config["maximum_absolute_served_offset"]
    inside = math.nextafter(maximum, 0.0)
    outside = math.nextafter(maximum, math.inf)
    authority = load_outer_calibration_authority(ROOT)
    for offset in (-inside, inside):
        p = authority.probability("stratum-draft", 1.25, offset)
        swapped = authority.probability("stratum-draft", -1.25, -offset)
        assert p + swapped == pytest.approx(1.0, abs=2e-15)
    for offset in (-outside, outside):
        with pytest.raises(ValidationFailure, match="registered maximum"):
            authority.probability("stratum-draft", 1.25, offset)
    assert authority.probability("stratum-player", 1.25, outside) == authority.probability(
        "stratum-player", 1.25, 0.0
    )


def test_registered_isotonic_domain_proves_served_clamp_is_inactive():
    config = build_outer_calibration_config(REGISTRY.read_bytes())
    domain = config["served_offset_domain"]
    policy = config["served_offset_policy"]
    epsilon = config["epsilon"]
    boundary = math.log((1.0 - epsilon) / epsilon)
    maximum = domain["maximum_absolute_served_offset"]
    upper = domain["isotonic_theta_upper_bound"]
    assert maximum + upper == pytest.approx(
        boundary - domain["strict_logit_margin"], abs=1e-15
    )
    assert domain["proven_numerical_margin"] > 0.0
    assert policy["units"] == "natural-log odds"
    assert policy["maximum_absolute_odds_multiplier"] == math.exp(
        policy["maximum_absolute_served_offset"]
    )
    assert "real-data validation may tighten or replace" in policy[
        "production_revalidation"
    ]
    assert policy["numerical_headroom"]["evidence_class"] == (
        "numerical safety policy; not empirical evidence"
    )
    combined_logits = []
    for signed_offset in (-maximum, maximum):
        for theta in (0.0, upper / 2.0, upper):
            for theta_sign in (-1.0, 1.0):
                combined = signed_offset + theta_sign * theta
                combined_logits.append(combined)
                assert abs(combined) < boundary
                unclipped = outer._sigmoid(combined, 1e-15)
                served = outer._sigmoid(combined, epsilon)
                assert unclipped == served
                for outcome in (0, 1):
                    assert outer._loss(outcome, unclipped, epsilon) == outer._loss(
                        outcome, served, epsilon
                    )
    assert min(combined_logits) == pytest.approx(-(maximum + upper), abs=1e-15)
    assert max(combined_logits) == pytest.approx(maximum + upper, abs=1e-15)


def test_future_test_labels_do_not_change_fit_or_prediction_bytes():
    config, rows, report = _bundle()
    changed = copy.deepcopy(rows)
    test_indexes = {index for fold in config["folds"] for index in range(*fold["test"])}
    for row in changed["rows"]:
        if row["series_index"] in test_indexes:
            row["observation"]["outcome"] = 1 - row["observation"]["outcome"]
    replayed = build_outer_calibration_selection_report(config, changed)
    original_fit = [
        {"fold_id": item["fold_id"], "family": item["family"], "fits": item.get("fits"), "rows": [{"row_id": row["row_id"], "raw_logit": row["raw_logit"], "probability": row["probability"]} for row in item.get("rows", [])]}
        for item in report["fold_results"]
    ]
    changed_fit = [
        {"fold_id": item["fold_id"], "family": item["family"], "fits": item.get("fits"), "rows": [{"row_id": row["row_id"], "raw_logit": row["raw_logit"], "probability": row["probability"]} for row in item.get("rows", [])]}
        for item in replayed["fold_results"]
    ]
    assert canonical_json(original_fit) == canonical_json(changed_fit)


def test_calibration_label_mutation_never_changes_raw_forecasts():
    config, rows, report = _bundle()
    changed = copy.deepcopy(rows)
    calibration_indexes = {index for fold in config["folds"] for index in range(*fold["calibration"])}
    for row in changed["rows"]:
        if row["series_index"] in calibration_indexes:
            row["observation"]["outcome"] = 1 - row["observation"]["outcome"]
    replayed = build_outer_calibration_selection_report(config, changed)
    assert [fold["raw_prediction_sha256"] for fold in report["folds"]] == [fold["raw_prediction_sha256"] for fold in replayed["folds"]]


def test_support_failures_are_unavailable_not_identity_fallback():
    config, rows, _ = _bundle()
    selected = rows["rows"][:96]
    logits = [float(row["features"]["signed_strength"]) for row in selected]
    one_class = copy.deepcopy(selected)
    for row in one_class:
        row["observation"]["outcome"] = 0
    with pytest.raises(ValidationFailure, match="two outcome classes"):
        _fit_family("identity", one_class, logits, config)
    with pytest.raises(ValidationFailure, match="distinct logits"):
        _fit_family("identity", selected, [0.0] * len(selected), config)
    with pytest.raises(ValidationFailure, match="finite"):
        _fit_family("identity", selected, [math.nan] + logits[1:], config)
    with pytest.raises(ValidationFailure, match="isotonic support"):
        _fit_family("symmetrized_bounded_isotonic", selected, [float(i % 3) for i in range(len(selected))], config)


def test_duplicate_ids_nonbinary_labels_and_missing_offsets_reject():
    config, rows, _ = _bundle()
    duplicate = copy.deepcopy(rows)
    duplicate["rows"][1]["row_id"] = duplicate["rows"][0]["row_id"]
    with pytest.raises(ValidationFailure, match="duplicate"):
        build_outer_calibration_selection_report(config, duplicate)
    nonbinary = copy.deepcopy(rows)
    nonbinary["rows"][0]["observation"]["outcome"] = 2
    with pytest.raises(ValidationFailure, match="binary"):
        build_outer_calibration_selection_report(config, nonbinary)
    missing = copy.deepcopy(rows)
    draft = next(row for row in missing["rows"] if row["output_class"] == "draft_score")
    draft["features"]["league_offset"] = None
    with pytest.raises(ValidationFailure, match="offset"):
        build_outer_calibration_selection_report(config, missing)


def test_candidate_test_identity_and_loss_aggregation_reconcile():
    _, _, report = _bundle()
    by_fold = {}
    for result in report["fold_results"]:
        assert result["available"]
        by_fold.setdefault(result["fold_id"], []).append(result)
        row_loss = {}
        for row in result["rows"]:
            row_loss.setdefault(row["series_id"], []).append(row["log_loss"])
        assert {row["series_id"] for row in result["series"]} == set(row_loss)
        for series in result["series"]:
            assert series["log_loss"] == pytest.approx(sum(row_loss[series["series_id"]]) / len(row_loss[series["series_id"]]))
        assert result["aggregate"]["log_loss"] == pytest.approx(sum(row["log_loss"] for row in result["blocks"]) / len(result["blocks"]))
    for results in by_fold.values():
        assert [item["family"] for item in results] == list(CANDIDATE_ORDER)
        assert len({tuple(item["test_row_ids"]) for item in results}) == 1


def test_duplicate_fold_identity_cannot_increase_support_or_narrow_interval():
    config, _, report = _bundle()
    original = report["selection"]
    duplicated = copy.deepcopy(report["fold_results"])
    for result in duplicated:
        result["fold_id"] = f"forged-copy:{result['fold_id']}"
    attacked = _select_family(report["fold_results"] + duplicated, config)
    original_by_family = {item["family"]: item for item in original["evidence"]}
    attacked_by_family = {item["family"]: item for item in attacked["evidence"]}
    for family in original_by_family:
        assert attacked_by_family[family]["unique_top_level_block_count"] == original_by_family[family]["unique_top_level_block_count"]
        assert attacked_by_family[family]["effective_sample_size"] == original_by_family[family]["effective_sample_size"]
        assert attacked_by_family[family]["one_sided_upper_bound"] == original_by_family[family]["one_sided_upper_bound"]


def test_nominally_passing_candidate_can_fail_simultaneous_family_rule():
    config = build_outer_calibration_config(REGISTRY.read_bytes())
    block_ids = [f"generator-block-{index:02d}" for index in range(6)]
    identity_deltas = [0.0, 0.0, 0.0, 0.0, 0.006, 0.006]
    fold_results = []
    for family in CANDIDATE_ORDER:
        if family == "symmetrized_beta":
            deltas = [0.0] * 6
        elif family == "identity":
            deltas = identity_deltas
        else:
            deltas = [0.03] * 6
        fold_results.append(
            {
                "fold_id": "synthetic-family-rule",
                "family": family,
                "available": True,
                "blocks": [
                    {"top_level_block_id": block_id, "log_loss": 0.5 + delta, "brier": 0.2, "series_count": 1}
                    for block_id, delta in zip(block_ids, deltas)
                ],
            }
        )
    selection = _select_family(fold_results, config)
    evidence = {item["family"]: item for item in selection["evidence"]}
    assert evidence["identity"]["nominal_noninferior"]
    assert not evidence["identity"]["noninferior"]
    assert evidence["identity"]["adjusted_one_sided_alpha"] == pytest.approx(0.005)
    assert len(selection["all_pairwise_contrasts"]) == 10
    assert selection["selected_family"] == "symmetrized_beta"


def test_selection_uses_noninferiority_then_simplicity_and_sparse_is_unavailable():
    _, _, identity = _bundle("identity")
    selection = identity["selection"]
    assert selection["selected_family"] == min(
        (item["family"] for item in selection["evidence"] if item["noninferior"]),
        key=CANDIDATE_ORDER.index,
    )
    sparse_config, sparse_rows, sparse = _bundle("sparse")
    assert sparse["selection"]["status"] == "unavailable"
    assert sparse["selection"]["selected_family"] is None
    assert sparse_config["regime"] == "sparse" and sparse_rows["synthetic"]


def test_non_authoritative_controls_recover_registered_matching_families():
    _, _, identity = _bundle("identity")
    _, _, temperature = _bundle("temperature")
    _, _, nonlinear = _bundle("nonlinear")
    assert identity["selection"]["selected_family"] == "identity"
    assert temperature["selection"]["selected_family"] == "symmetric_temperature"
    nonlinear_evidence = {item["family"]: item for item in nonlinear["selection"]["evidence"]}
    assert nonlinear_evidence["symmetrized_beta"]["noninferior"]


def test_final_refit_uses_only_calibration_roles_with_frozen_upstream_predictions():
    config, rows, report = _bundle()
    original = report["full_presealed_refit"]
    calibration_indexes = {index for fold in config["folds"] for index in range(*fold["calibration"])}
    expected_ids = sorted(row["row_id"] for row in rows["rows"] if row["series_index"] in calibration_indexes)
    assert original["calibration_row_ids"] == expected_ids
    assert not (
        set(original["calibration_row_ids"])
        & (set(original["upstream_train_row_ids"]) | set(original["upstream_validation_row_ids"]))
    )
    upstream_changed = copy.deepcopy(rows)
    for row in upstream_changed["rows"]:
        if row["row_id"] in set(original["upstream_train_row_ids"]) | set(original["upstream_validation_row_ids"]):
            row["observation"]["outcome"] = 1 - row["observation"]["outcome"]
    held = _build_full_presealed_refit(
        config,
        upstream_changed["rows"],
        report["selection"]["selected_family"],
        frozen_calibration_lineage=original["calibration_logit_lineage"],
        frozen_serving_raw_model=original["future_serving_upstream_raw_model"],
    )
    assert canonical_json(held["transforms"]) == canonical_json(original["transforms"])
    calibration_changed = copy.deepcopy(rows)
    for row in calibration_changed["rows"]:
        if row["row_id"] in set(original["calibration_row_ids"]):
            row["observation"]["outcome"] = 1 - row["observation"]["outcome"]
    refit = _build_full_presealed_refit(
        config,
        calibration_changed["rows"],
        report["selection"]["selected_family"],
        frozen_calibration_lineage=original["calibration_logit_lineage"],
        frozen_serving_raw_model=original["future_serving_upstream_raw_model"],
    )
    assert canonical_json(refit["transforms"]) != canonical_json(original["transforms"])


def test_later_validation_labels_cannot_rewrite_earlier_cross_fitted_calibration_logits():
    config, rows, report = _bundle()
    original = report["full_presealed_refit"]
    changed = copy.deepcopy(rows)
    later_validation = set(range(*config["folds"][1]["validation"]))
    for row in changed["rows"]:
        if row["series_index"] in later_validation:
            row["observation"]["outcome"] = 1 - row["observation"]["outcome"]
    rebuilt = _build_full_presealed_refit(config, changed["rows"], report["selection"]["selected_family"])
    early_fold_id = config["folds"][0]["fold_id"]
    original_early = [item for item in original["calibration_logit_lineage"] if item["fold_id"] == early_fold_id]
    rebuilt_early = [item for item in rebuilt["calibration_logit_lineage"] if item["fold_id"] == early_fold_id]
    assert canonical_json(original_early) == canonical_json(rebuilt_early)
    for before, after in zip(original["calibration_logit_lineage"], rebuilt["calibration_logit_lineage"]):
        if canonical_json(before) != canonical_json(after):
            assert before["fold_id"] == config["folds"][1]["fold_id"]


def test_frozen_artifacts_load_replay_and_claim_ceiling():
    authority = load_outer_calibration_authority(ROOT)
    result = replay_outer_calibration(authority)
    assert result["status"] == "PASS_SYNTHETIC_MECHANICS_ONLY"
    assert result["parity_row_count"] > 0
    assert result["claim_ceiling"] == CLAIM_CEILING
    assert set(authority.authority_payload["hard_gates"]) == set(HARD_GATES)


def test_authority_constructor_and_object_new_forgery_reject():
    with pytest.raises(ValidationFailure, match="loader-issued"):
        OuterCalibrationAuthority(ROOT, {}, {}, {}, {}, {})
    forged = object.__new__(OuterCalibrationAuthority)
    with pytest.raises(ValidationFailure, match="exact loader-created"):
        forged.authenticate()


def test_authority_is_deeply_immutable_and_reauthenticates_projection_changes():
    authority = load_outer_calibration_authority(ROOT)
    transform = authority.transforms["stratum-draft"]
    with pytest.raises(TypeError):
        transform["parameters"]["a"] = 999.0
    original = authority.transforms
    object.__setattr__(authority, "transforms", {"stratum-draft": transform})
    try:
        with pytest.raises(ValidationFailure, match="projection"):
            authority.authenticate()
    finally:
        object.__setattr__(authority, "transforms", original)
    authority.authenticate()


def test_fmean_issuance_alias_checker_and_direct_impl_bypasses_reject():
    original_fmean = outer.fmean
    outer.fmean = lambda values: sum(values) / len(values)
    try:
        with pytest.raises(ValidationFailure, match="dependency"):
            load_outer_calibration_authority(ROOT)
    finally:
        outer.fmean = original_fmean
    outer._is_issued_authority = lambda _value: True
    try:
        forged = object.__new__(OuterCalibrationAuthority)
        with pytest.raises(ValidationFailure, match="exact loader-created"):
            forged.authenticate()
    finally:
        del outer._is_issued_authority
    authority = load_outer_calibration_authority(ROOT)
    object.__setattr__(authority, "_integrity_checker", lambda: None)
    original_served = outer.served_probability
    outer.served_probability = lambda *_args, **_kwargs: 0.5
    try:
        with pytest.raises(ValidationFailure, match="runtime"):
            authority.probability("stratum-player", 1.0, 0.0)
    finally:
        outer.served_probability = original_served
        object.__delattr__(authority, "_integrity_checker")
    direct = outer._load_outer_calibration_authority_impl(ROOT)
    with pytest.raises(ValidationFailure, match="exact loader-created"):
        direct.authenticate()


def test_authority_threat_model_is_literal_and_nonpromotional():
    authority = load_outer_calibration_authority(ROOT)
    threat = authority.authority_payload["authority_threat_model"]
    assert threat["scope"] == "process_local_misuse_and_ordinary_forgery_guard_under_honest_interpreter"
    assert threat["hostile_same_process_unforgeability"] is False
    assert threat["closure_cell_mutation_resistant"] is False
    assert threat["content_hashing_authorizes_promotion"] is False
    assert threat["singleton_identity_authorizes_promotion"] is False
    assert "trust_root" in threat["production_authority_requirement"]


def test_five_authority_bypasses_reject_in_fresh_processes():
    prelude = (
        "from pathlib import Path\n"
        "import lol_kills.v2.evaluation.outer_calibration as o\n"
        "from lol_kills.v2.evaluation.checks import ValidationFailure\n"
        f"ROOT=Path({str(ROOT)!r})\n"
    )
    attacks = [
        "old=o.fmean\no.fmean=lambda v:sum(v)/len(v)\ntry:o.load_outer_calibration_authority(ROOT)\nexcept ValidationFailure:print('REJECT')\n",
        "o._is_issued_authority=lambda _:True\nx=object.__new__(o.OuterCalibrationAuthority)\ntry:x.authenticate()\nexcept ValidationFailure:print('REJECT')\n",
        "a=o.load_outer_calibration_authority(ROOT)\ntry:a.transforms['stratum-draft']['parameters']['a']=999\nexcept TypeError:print('REJECT')\n",
        "a=o.load_outer_calibration_authority(ROOT)\nobject.__setattr__(a,'_integrity_checker',lambda:None)\no.served_probability=lambda *a,**k:.5\ntry:a.probability('stratum-player',1.0,0.0)\nexcept ValidationFailure:print('REJECT')\n",
        "x=o._load_outer_calibration_authority_impl(ROOT)\ntry:x.authenticate()\nexcept ValidationFailure:print('REJECT')\n",
    ]
    for attack in attacks:
        output = subprocess.check_output([sys.executable, "-c", prelude + attack], cwd=ROOT, text=True)
        assert output == "REJECT\n"


def test_in_place_code_and_default_mutations_reject_cleanly():
    original_code = outer._select_family.__code__
    try:
        outer._select_family.__code__ = (lambda *_args, **_kwargs: {}).__code__
        with pytest.raises(ValidationFailure, match="runtime"):
            load_outer_calibration_authority(ROOT)
    finally:
        outer._select_family.__code__ = original_code
    original_kwdefaults = dict(outer.apply_outer_transform.__kwdefaults__)
    try:
        outer.apply_outer_transform.__kwdefaults__["epsilon"] = 1e-6
        with pytest.raises(ValidationFailure, match="runtime"):
            load_outer_calibration_authority(ROOT)
    finally:
        outer.apply_outer_transform.__kwdefaults__.clear()
        outer.apply_outer_transform.__kwdefaults__.update(original_kwdefaults)


@pytest.mark.parametrize("name", ["_partition", "_fit_raw_model", "_candidate_fold_evidence", "_source_closure", "_function_fingerprint", "_assert_runtime_integrity"])
def test_transitive_helper_same_identity_code_mutations_reject(name):
    function = getattr(outer, name)
    original_code = function.__code__
    try:
        function.__code__ = (lambda *_args, **_kwargs: None).__code__
        with pytest.raises(ValidationFailure, match="runtime"):
            load_outer_calibration_authority(ROOT)
    finally:
        function.__code__ = original_code


def test_kwdefault_registry_and_forged_public_baseline_attacks_cannot_authorize():
    original_kwdefaults = dict(outer._build_full_presealed_refit.__kwdefaults__)
    try:
        outer._build_full_presealed_refit.__kwdefaults__["frozen_calibration_lineage"] = ()
        with pytest.raises(ValidationFailure, match="runtime"):
            load_outer_calibration_authority(ROOT)
    finally:
        outer._build_full_presealed_refit.__kwdefaults__.clear()
        outer._build_full_presealed_refit.__kwdefaults__.update(original_kwdefaults)
    original_order = outer.CANDIDATE_ORDER
    try:
        outer.CANDIDATE_ORDER = tuple(reversed(original_order))
        with pytest.raises(ValidationFailure, match="registry"):
            load_outer_calibration_authority(ROOT)
    finally:
        outer.CANDIDATE_ORDER = original_order
    outer._RUNTIME_FUNCTION_BASELINE = {"forged": "accept"}
    try:
        assert replay_outer_calibration(load_outer_calibration_authority(ROOT))["status"] == "PASS_SYNTHETIC_MECHANICS_ONLY"
    finally:
        del outer._RUNTIME_FUNCTION_BASELINE


def test_warmed_and_two_fresh_interpreter_replays_are_exact():
    authority = load_outer_calibration_authority(ROOT)
    warm = replay_outer_calibration(authority)
    assert warm["status"] == "PASS_SYNTHETIC_MECHANICS_ONLY"
    script = (
        "from pathlib import Path;"
        "from lol_kills.v2.evaluation.outer_calibration import load_outer_calibration_authority,replay_outer_calibration,canonical_json;"
        f"a=load_outer_calibration_authority(Path({str(ROOT)!r}));"
        "print(canonical_json(replay_outer_calibration(a)).decode('ascii').strip())"
    )
    first = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True)
    second = subprocess.check_output([sys.executable, "-c", script], cwd=ROOT, text=True)
    assert first == second == canonical_json(warm).decode("ascii")


def test_noncanonical_duplicate_hash_and_path_attacks_reject(tmp_path):
    source = ROOT / "data/lol/v2/evaluation/b2"
    target = tmp_path / "data/lol/v2/evaluation/b2"
    target.mkdir(parents=True)
    for path in source.glob("outer-calibration-*.json"):
        (target / path.name).write_bytes(path.read_bytes())
    transforms = target / "outer-calibration/transforms"
    transforms.mkdir(parents=True)
    for path in (source / "outer-calibration/transforms").glob("*.json"):
        (transforms / path.name).write_bytes(path.read_bytes())
    authority_path = target / "outer-calibration-authority.json"
    authority = json.loads(authority_path.read_text())
    authority["refs"]["config"]["path"] = "../../escape.json"
    authority_path.write_bytes(canonical_json(authority))
    with pytest.raises(ValidationFailure, match="unsafe"):
        load_outer_calibration_authority(tmp_path)


@pytest.mark.parametrize("attack", ["parent_symlink", "leaf_symlink", "hardlink"])
def test_authority_path_component_and_alias_attacks_reject(tmp_path, attack):
    source = ROOT / "data/lol/v2/evaluation/b2"
    target = tmp_path / "data/lol/v2/evaluation/b2"
    shutil.copytree(source / "outer-calibration", target / "outer-calibration")
    for path in source.glob("outer-calibration-*.json"):
        target.mkdir(parents=True, exist_ok=True)
        (target / path.name).write_bytes(path.read_bytes())
    (target / "calibration-candidate-registry.json").write_bytes((source / "calibration-candidate-registry.json").read_bytes())
    if attack == "parent_symlink":
        real = target / "outer-calibration-real"
        (target / "outer-calibration").rename(real)
        (target / "outer-calibration").symlink_to(real.name, target_is_directory=True)
    elif attack == "leaf_symlink":
        config_path = target / "outer-calibration-config.json"
        real = target / "outer-calibration-config-real.json"
        config_path.rename(real)
        config_path.symlink_to(real.name)
    else:
        os.link(target / "outer-calibration-config.json", target / "outer-calibration-config-alias.json")
    with pytest.raises(ValidationFailure, match="symlink|hardlinked"):
        load_outer_calibration_authority(tmp_path)


def test_constant_half_transform_with_copied_bindings_rejects(tmp_path):
    source = ROOT / "data/lol/v2/evaluation/b2"
    target = tmp_path / "data/lol/v2/evaluation/b2"
    target.mkdir(parents=True)
    for path in source.glob("outer-calibration-*.json"):
        (target / path.name).write_bytes(path.read_bytes())
    (target / "calibration-candidate-registry.json").write_bytes((source / "calibration-candidate-registry.json").read_bytes())
    transforms = target / "outer-calibration/transforms"
    transforms.mkdir(parents=True)
    changed_hashes = {}
    for path in (source / "outer-calibration/transforms").glob("*.json"):
        payload = json.loads(path.read_text())
        payload["served_transform"]["family"] = "symmetrized_bounded_isotonic"
        payload["served_transform"]["parameters"] = {"knots": [0.0, 1.0], "values": [0.5, 0.5]}
        raw = canonical_json(payload)
        (transforms / path.name).write_bytes(raw)
        changed_hashes[path.name] = hashlib.sha256(raw).hexdigest()
    authority_path = target / "outer-calibration-authority.json"
    authority = json.loads(authority_path.read_text())
    for ref in authority["refs"]["transforms"].values():
        ref["sha256"] = changed_hashes[Path(ref["path"]).name]
    authority_path.write_bytes(canonical_json(authority))
    with pytest.raises(ValidationFailure):
        load_outer_calibration_authority(tmp_path)


@pytest.mark.parametrize("attack", ["false", "missing", "extra", "self_rehashed"])
def test_hard_gate_attacks_reject(tmp_path, attack):
    source = ROOT / "data/lol/v2/evaluation/b2"
    target = tmp_path / "data/lol/v2/evaluation/b2"
    shutil.copytree(source / "outer-calibration", target / "outer-calibration")
    target.mkdir(parents=True, exist_ok=True)
    for path in source.glob("outer-calibration-*.json"):
        (target / path.name).write_bytes(path.read_bytes())
    (target / "calibration-candidate-registry.json").write_bytes((source / "calibration-candidate-registry.json").read_bytes())
    authority_path = target / "outer-calibration-authority.json"
    authority = json.loads(authority_path.read_text())
    gate_name = next(iter(authority["hard_gates"]))
    if attack == "false":
        authority["hard_gates"][gate_name]["passed"] = False
    elif attack == "missing":
        authority["hard_gates"].pop(gate_name)
    elif attack == "extra":
        authority["hard_gates"]["GATE_FORGED_EXTRA"] = copy.deepcopy(authority["hard_gates"][gate_name])
    else:
        authority["hard_gates"][gate_name]["evidence"] = {"forged": True}
        authority["hard_gates"][gate_name]["evidence_sha256"] = hashlib.sha256(canonical_json({"forged": True})).hexdigest()
    authority_path.write_bytes(canonical_json(authority))
    with pytest.raises(ValidationFailure):
        load_outer_calibration_authority(tmp_path)
