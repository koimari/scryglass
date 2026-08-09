from __future__ import annotations

from copy import deepcopy
from copy import copy
import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import beta as beta_distribution

from lol_kills.v2.evaluation import r20_foundation as foundation
from lol_kills.v2.evaluation import r20_foundation_algorithms as algorithms_module
from lol_kills.v2.evaluation import r20_foundation_generator as generator_module
from lol_kills.v2.evaluation import r20_foundation_inference as inference_module
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.r20_foundation import (
    CANDIDATE_REGISTRY_LOCATOR,
    FOUNDATION_CONFIG_LOCATOR,
    VerifiedFoundationAuthority,
    load_foundation_artifacts,
    replay_foundation_row_candidate,
    volume_basis_design,
)
from lol_kills.v2.evaluation.r20_foundation_algorithms import (
    METHOD_COMPLEXITY,
    METHOD_SPECS,
    replay_foundation_method,
)
from lol_kills.v2.evaluation.r20_foundation_generator import (
    INITIAL_TRAIN_SERIES_PER_CELL,
    OUTPUT_STRATA,
    REGIMES,
    SOURCE_CONTEXT_PATTERNS,
    TEST_SERIES_PER_CELL_PER_FOLD,
    build_prequential_plan,
    build_r20_benchmark,
    verify_prequential_plan,
)
from lol_kills.v2.evaluation.r20_foundation_inference import (
    POSTERIOR_DRAWS,
    infer_beta_binomial,
    monte_carlo_width_design,
)
from lol_kills.v2.evaluation.types import canonical_sha256


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def authority() -> VerifiedFoundationAuthority:
    return load_foundation_artifacts(ROOT)


def test_loader_issues_exact_immutable_synthetic_capability(
    authority: VerifiedFoundationAuthority,
) -> None:
    assert len(authority.benchmark["rows"]) == 1600
    assert len(authority.benchmark["prequential_plan"]["ordered_series_ids"]) == 800
    assert authority.authority["synthetic_only"] is True
    assert authority.authority["production_eligible"] is False
    assert authority.authority["authority_threat_model"] == {
        "boundary": "in_process_public_surface_tamper_resistance",
        "arbitrary_same_process_python_introspection": "out_of_scope",
        "cryptographic_unforgeability": False,
        "production_authority": False,
        "claim_ceiling": (
            "loader-issued synthetic capability under cooperative-process execution"
        ),
    }
    assert authority.config["draw_count"] == 256
    with pytest.raises(TypeError):
        VerifiedFoundationAuthority()


def test_authority_type_cannot_be_constructed_or_subclassed() -> None:
    with pytest.raises(TypeError):
        VerifiedFoundationAuthority({})

    with pytest.raises(TypeError):
        class ForgedAuthority(VerifiedFoundationAuthority):
            pass


def test_detached_attested_source_cannot_execute_imported_code(tmp_path: Path) -> None:
    for relative in (
        "lol_kills/v2/evaluation/r20_foundation_generator.py",
        "lol_kills/v2/evaluation/r20_foundation_inference.py",
        "lol_kills/v2/evaluation/r20_foundation_algorithms.py",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise RuntimeError('detached source attack')\n")
    with pytest.raises(ValidationFailure, match="detached source root"):
        load_foundation_artifacts(tmp_path)


def test_public_copies_cannot_mutate_authorized_state(
    authority: VerifiedFoundationAuthority,
) -> None:
    row = authority.benchmark["rows"][0]
    method = "posterior_mean_displacement_v1"
    before = replay_foundation_row_candidate(
        authority=authority, row_id=row["row_id"], method_id=method
    )
    benchmark_copy = authority.benchmark
    benchmark_copy["rows"][0]["candidate_inputs"]["posterior_draws"] = [0.5] * 19
    candidate_copy = authority.candidate_payloads
    candidate_copy[method]["boundaries"]["minimum_draws"] = 19
    dependencies_copy = authority.dependency_payloads
    dependencies_copy["posterior_draws"]["values_by_row_id"][row["row_id"]] = [0.5] * 19
    after = replay_foundation_row_candidate(
        authority=authority, row_id=row["row_id"], method_id=method
    )
    assert after == before


def test_copy_without_loader_registration_is_rejected_for_replay(
    authority: VerifiedFoundationAuthority,
) -> None:
    copy_of_authority = copy(authority)
    row_id = authority.benchmark["rows"][0]["row_id"]
    with pytest.raises(foundation.ValidationFailure, match="loader-issued"):
        replay_foundation_row_candidate(
            authority=copy_of_authority,
            row_id=row_id,
            method_id="posterior_mean_displacement_v1",
        )


@pytest.mark.parametrize(
    "alias",
    [
        "build_r20_benchmark",
        "infer_beta_binomial",
        "replay_foundation_method",
        "monte_carlo_width_design",
        "_BOUND_METHOD_REPLAY",
    ],
)
def test_replaced_module_foundation_function_is_ignored_for_replay(
    authority: VerifiedFoundationAuthority,
    monkeypatch: pytest.MonkeyPatch,
    alias: str,
) -> None:
    row_id = authority.benchmark["rows"][0]["row_id"]
    method = "posterior_mean_displacement_v1"
    expected = replay_foundation_row_candidate(
        authority=authority, row_id=row_id, method_id=method
    )

    def fake_replay(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"status": "forged", "value": 0.0, "authorized_candidate": False}

    monkeypatch.setattr(foundation, alias, fake_replay, raising=False)
    monkeypatch.setattr(
        foundation,
        f"{alias}_ID",
        id(fake_replay),
        raising=False,
    )
    try:
        mutated = replay_foundation_row_candidate(
            authority=authority, row_id=row_id, method_id=method
        )
    except ValidationFailure as exc:
        assert "namespace changed" in str(exc)
    else:
        assert mutated == expected


def test_public_registrar_forgery_probe_is_closed(
    authority: VerifiedFoundationAuthority,
) -> None:
    assert not hasattr(foundation, "_AUTHORITY_REGISTRY")
    forged = object.__new__(VerifiedFoundationAuthority)
    with pytest.raises(ValidationFailure, match="loader-issued"):
        replay_foundation_row_candidate(
            authority=forged,
            row_id=authority.benchmark["rows"][0]["row_id"],
            method_id="posterior_mean_displacement_v1",
        )


def test_public_replay_impl_rebinding_cannot_bypass_private_api(
    authority: VerifiedFoundationAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forged_impl(**_kwargs: object) -> dict[str, object]:
        return {"value": 999.0, "authorized_candidate": True}

    monkeypatch.setattr(
        foundation,
        "_replay_foundation_row_candidate_impl",
        forged_impl,
    )
    row_id = authority.benchmark["rows"][0]["row_id"]
    with pytest.raises(ValidationFailure, match="namespace changed"):
        foundation.replay_foundation_row_candidate(
            authority=authority,
            row_id=row_id,
            method_id="posterior_mean_displacement_v1",
        )
    with pytest.raises(ValidationFailure, match="namespace changed"):
        foundation.replay_foundation_row_candidate(
            authority=object(),
            row_id=row_id,
            method_id="posterior_mean_displacement_v1",
        )


def test_public_load_impl_rebinding_cannot_mint_private_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forged_impl(*_args: object, **kwargs: object) -> object:
        nonlocal called
        called = True
        return kwargs["issue"]({"forged": True})

    monkeypatch.setattr(foundation, "_load_foundation_artifacts_impl", forged_impl)
    with pytest.raises(ValidationFailure, match="namespace changed"):
        foundation.load_foundation_artifacts(ROOT)
    assert called is False


def test_lowercase_executable_dependency_rebinding_is_rejected(
    authority: VerifiedFoundationAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(algorithms_module, "np", object())
    with pytest.raises(ValidationFailure, match="executable dependency changed: np"):
        replay_foundation_row_candidate(
            authority=authority,
            row_id=authority.benchmark["rows"][0]["row_id"],
            method_id="posterior_mean_displacement_v1",
        )


def test_same_module_name_callable_imitation_cannot_replace_private_execution(
    authority: VerifiedFoundationAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row_id = authority.benchmark["rows"][0]["row_id"]
    expected = replay_foundation_row_candidate(
        authority=authority,
        row_id=row_id,
        method_id="posterior_mean_displacement_v1",
    )

    def fake_replay(**kwargs: object) -> dict[str, object]:
        return {
            "method_id": kwargs["method_id"],
            "family": "posterior_information",
            "units": "prior_standard_deviations",
            "value": 999.0,
            "executed_boundary_sha256": "a" * 64,
            "authorized_candidate": False,
            "replay_ok": True,
            "method_complexity": METHOD_COMPLEXITY[
                "posterior_mean_displacement_v1"
            ],
        }

    fake_replay.__module__ = replay_foundation_method.__module__
    fake_replay.__name__ = replay_foundation_method.__name__
    monkeypatch.setattr(foundation, "replay_foundation_method", fake_replay)
    with pytest.raises(ValidationFailure, match="namespace changed"):
        replay_foundation_row_candidate(
            authority=authority,
            row_id=row_id,
            method_id="posterior_mean_displacement_v1",
        )


@pytest.mark.parametrize(
    ("module", "helper_name"),
    [
        (generator_module, "_build_series_layout"),
        (inference_module, "_row_mad"),
    ],
)
def test_preload_helper_global_replacement_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    module: object,
    helper_name: str,
) -> None:
    original = getattr(module, helper_name)

    def fake(*args: object, **kwargs: object) -> object:
        return original(*args, **kwargs)

    fake.__module__ = original.__module__
    fake.__name__ = original.__name__
    monkeypatch.setattr(module, helper_name, fake)
    with pytest.raises(ValidationFailure, match="helper namespace"):
        load_foundation_artifacts(ROOT)


def test_postload_replay_helper_spec_replacement_is_rejected(
    authority: VerifiedFoundationAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = METHOD_SPECS["posterior_mean_displacement_v1"]["replay"]

    def fake(*args: object, **kwargs: object) -> float:
        return 999.0

    monkeypatch.setitem(
        METHOD_SPECS["posterior_mean_displacement_v1"],
        "replay",
        fake,
    )
    with pytest.raises(ValidationFailure, match="helper/specification"):
        replay_foundation_row_candidate(
            authority=authority,
            row_id=authority.benchmark["rows"][0]["row_id"],
            method_id="posterior_mean_displacement_v1",
        )
    monkeypatch.setitem(
        METHOD_SPECS["posterior_mean_displacement_v1"],
        "replay",
        original,
    )


def test_authorized_replay_has_no_boundary_override_and_binds_identities(
    authority: VerifiedFoundationAuthority,
) -> None:
    signature = inspect.signature(replay_foundation_row_candidate)
    assert "boundaries" not in signature.parameters
    row_id = authority.benchmark["rows"][0]["row_id"]
    with pytest.raises(TypeError):
        replay_foundation_row_candidate(
            authority=authority,
            row_id=row_id,
            method_id="posterior_mean_displacement_v1",
            boundaries={"minimum_draws": 19},
        )
    result = replay_foundation_row_candidate(
        authority=authority,
        row_id=row_id,
        method_id="posterior_mean_displacement_v1",
    )
    assert result["authorized_candidate"] is True
    assert result["synthetic_only"] is True
    assert result["production_eligible"] is False
    assert result["executed_boundary_sha256"] == result["boundary_sha256"]
    assert result["authority_sha256"] == authority.authority_sha256


def test_candidate_boundary_change_is_new_identity_and_not_authorized(
    authority: VerifiedFoundationAuthority,
) -> None:
    registry = authority.candidate_registry
    candidate = next(
        item
        for item in registry["candidates"]
        if item["method_id"] == "central_interval_contraction_v2"
    )
    original_identity = canonical_sha256(candidate)
    candidate["boundaries"]["central_mass"] = 0.50
    candidate["boundary_sha256"] = canonical_sha256(candidate["boundaries"])
    assert canonical_sha256(candidate) != original_identity
    with pytest.raises(ValidationFailure, match="precision boundaries"):
        foundation._validate_candidates(
            ROOT,
            registry,
            authority.benchmark["rows"],
            authority.authority["config_ref"],
            hashlib.sha256((ROOT / FOUNDATION_CONFIG_LOCATOR).read_bytes()).hexdigest(),
            foundation._source_hashes(ROOT),
            replay_call=replay_foundation_method,
            method_specs=METHOD_SPECS,
            method_complexity=METHOD_COMPLEXITY,
        )


def test_low_level_primitive_cannot_masquerade_as_authorized_candidate() -> None:
    draws = np.linspace(0.1, 0.9, POSTERIOR_DRAWS).tolist()
    result = replay_foundation_method(
        method_id="posterior_mean_displacement_v1",
        dependencies={"posterior_draws": draws, "prior_draws": list(reversed(draws))},
        boundaries={"minimum_draws": POSTERIOR_DRAWS},
    )
    assert result["authorized_candidate"] is False
    assert result["executed_boundary_sha256"] == canonical_sha256(
        {"minimum_draws": POSTERIOR_DRAWS}
    )


def test_observation_only_inference_hides_truth() -> None:
    assert "latent_truth" not in inspect.signature(infer_beta_binomial).parameters
    observation = {"successes": 7, "trials": 20}
    first = infer_beta_binomial(
        observation=observation,
        inference_seed=42,
        draw_count=256,
    )
    hidden_record = {"latent_truth": 0.01, "observation": observation}
    hidden_record["latent_truth"] = 0.99
    second = infer_beta_binomial(
        observation=hidden_record["observation"],
        inference_seed=42,
        draw_count=256,
    )
    assert first == second


def test_beta_posterior_matches_frozen_wolfram_quantile_oracle() -> None:
    result = infer_beta_binomial(
        observation={"successes": 7, "trials": 20},
        inference_seed=42,
        draw_count=256,
    )
    assert result["posterior_parameters"] == {"alpha": 9.0, "beta": 15.0}
    quantiles = beta_distribution.ppf([0.025, 0.975], 9, 15)
    assert quantiles[0] == pytest.approx(0.19707642396901415, abs=1e-14)
    assert quantiles[1] == pytest.approx(0.5726560369635058, abs=1e-14)
    assert quantiles[1] - quantiles[0] == pytest.approx(
        0.3755796129944916, abs=1e-14
    )


def test_fresh_generation_replays_observation_and_inference_exactly(
    authority: VerifiedFoundationAuthority,
) -> None:
    generated = build_r20_benchmark()
    assert generated["rows"] == authority.benchmark["rows"]
    assert generated["prequential_plan"] == authority.benchmark["prequential_plan"]
    generator_source = (
        ROOT / "lol_kills/v2/evaluation/r20_foundation_generator.py"
    ).read_text()
    assert not any(method_id in generator_source for method_id in METHOD_SPECS)


def test_monte_carlo_draw_design_is_executable_and_adequate(
    authority: VerifiedFoundationAuthority,
) -> None:
    design = monte_carlo_width_design(
        registered_observation_cells=foundation._registered_observation_cells(
            authority.benchmark["rows"],
        ),
    )
    assert design == authority.config["monte_carlo_design"]
    assert design["draw_count"] == 256
    assert design["draw_count"] >= 128
    assert design["achieved_p90_absolute_width_error"] <= design[
        "target_absolute_width_error"
    ]
    assert design["passes"] is True
    assert set(design["regime_grid"]) == set(REGIMES)
    assert design["trial_grid"] == [12, 24, 36, 48]
    assert "oracle_draw_count" not in design
    assert design["achieved_p90_true_contraction_error"] <= design[
        "target_p90_contraction_error"
    ]
    assert all(
        "central_contraction_error" in report
        and "mad_contraction_error" in report
        and set(report["actual_candidate_replay"])
        == {
            "central_interval_contraction_v2",
            "robust_mad_contraction_v1",
        }
        for case in design["case_results"]
        for report in case["reference_mode_reports"].values()
    )
    assert design["minimum_candidate_agreement_accuracy"] == 1.0
    assert design["minimum_candidate_agreement_wilson_lower_95"] >= design[
        "target_min_candidate_agreement_wilson_lower_95"
    ]
    assert design["minimum_replication_clustered_rank_lower_95"] >= design[
        "target_min_replication_clustered_rank_lower_95"
    ]
    assert design["registered_benchmark_observation_coverage"]["claim"] == (
        "complete_unique_conditional_success_trial_coverage"
    )
    assert design["registered_benchmark_observation_coverage"]["omitted_count"] == 0
    assert (
        design["registered_benchmark_observation_coverage"]["registered_count"]
        == len(
            design["registered_benchmark_observation_coverage"]["mapping"]
        )
    )
    assert all(
        "latent_alpha" not in case and "latent_beta" not in case
        for case in design["cases"]
    )
    compressed_precision = [
        cell
        for cell in design["candidate_reference_mode_regime_matrix"]
        if cell["reference_mode"] == "compressed_reference"
        and cell["method_summary"] in {
            "central_width",
            "mad",
        }
    ]
    assert compressed_precision
    assert all(
        cell["actual_candidate_status"]["expected_statuses"] == ["reject"]
        and cell["actual_candidate_status"]["observed_accept_count"] == 0
        and cell["adequacy_status"]
        in {"pass", "not_applicable_exact_oracle_ties"}
        and cell["cell_status"] == "pass"
        for cell in compressed_precision
    )
    boundary = design["precision_boundary_parity"]
    assert boundary["passes"] is True
    assert {
        method["method_id"] for method in boundary["methods"]
    } == {
        "central_interval_contraction_v2",
        "robust_mad_contraction_v1",
    }
    expected = {
        "just_below_tolerance": "reject",
        "inside_tolerance": "accept",
        "exact_zero": "accept",
        "positive": "accept",
    }
    assert all(
        method["passes"] is True
        and {
            case["case_id"]: case["batch_status"]
            for case in method["cases"]
        }
        == expected
        and {
            case["case_id"]: case["scalar_status"]
            for case in method["cases"]
        }
        == expected
        and all(
            case["position_ok"]
            and case["expected_parity_ok"]
            and case["scalar_batch_parity_ok"]
            for case in method["cases"]
        )
        for method in boundary["methods"]
    )
    assert design["candidate_reference_mode_regime_matrix"]


def test_exact_three_folds_and_support(authority: VerifiedFoundationAuthority) -> None:
    plan = authority.benchmark["prequential_plan"]
    rows = {row["row_id"]: row for row in authority.benchmark["rows"]}
    assert len(plan["folds"]) == 3
    seen_test: set[str] = set()
    for fold in plan["folds"]:
        assert len(fold["test_series_ids"]) == 5 * TEST_SERIES_PER_CELL_PER_FOLD
        assert not seen_test.intersection(fold["test_series_ids"])
        seen_test.update(fold["test_series_ids"])
        assert max(rows[row_id]["resolved"] for row_id in fold["train_row_ids"]) < min(
            rows[row_id]["issued"] for row_id in fold["test_row_ids"]
        )
        for output_type, _ in OUTPUT_STRATA:
            support = fold["test_support_by_output"][output_type]
            assert support == {
                "raw_series": TEST_SERIES_PER_CELL_PER_FOLD,
                "effective_series": TEST_SERIES_PER_CELL_PER_FOLD,
                "rows": 2 * TEST_SERIES_PER_CELL_PER_FOLD,
            }
            labels = {
                rows[row_id]["fixture_label"]
                for row_id in fold["test_row_ids"]
                if rows[row_id]["output_type"] == output_type
            }
            assert labels == {0, 1}
    assert plan["initial_series_per_cell"] == INITIAL_TRAIN_SERIES_PER_CELL


def test_plan_rebuild_is_order_invariant_but_submitted_order_attack_rejects(
    authority: VerifiedFoundationAuthority,
) -> None:
    rows = authority.benchmark["rows"]
    assert build_prequential_plan(list(reversed(rows))) == authority.benchmark[
        "prequential_plan"
    ]
    attacked = deepcopy(authority.benchmark["prequential_plan"])
    attacked["folds"][0]["test_row_ids"].reverse()
    with pytest.raises(ValueError, match="not the exact rebuilt plan"):
        verify_prequential_plan(rows, attacked)


@pytest.mark.parametrize(
    "attack",
    [
        "three_map", "missing_cell", "one_series_cell", "mixed_series",
        "mixed_outcome", "mixed_latent_truth", "mixed_generator_regime",
        "destroyed_class_support", "conditional_volume_separation", "overlap",
    ],
)
def test_plan_builder_rejects_structural_attacks(
    authority: VerifiedFoundationAuthority,
    attack: str,
) -> None:
    rows = deepcopy(authority.benchmark["rows"])
    if attack == "three_map":
        extra = deepcopy(rows[0])
        extra["row_id"] = "three-map-attack"
        extra["case_id"] = "three-map-case"
        extra["issued"] = rows[1]["issued"]
        extra["event"] = rows[1]["event"]
        extra["resolved"] = rows[1]["resolved"]
        rows.append(extra)
    elif attack == "missing_cell":
        rows = [row for row in rows if row["output_type"] != "tier_list"]
    elif attack == "one_series_cell":
        keep = {
            row["series_id"]
            for row in rows
            if row["output_type"] != "player_rating"
        }
        keep.add(next(row["series_id"] for row in rows if row["output_type"] == "player_rating"))
        rows = [row for row in rows if row["series_id"] in keep]
    elif attack == "mixed_series":
        rows[0]["output_type"] = "team_rating"
        rows[0]["stratum_id"] = "stratum-team"
    elif attack == "mixed_outcome":
        rows[0]["fixture_label"] = 1 - rows[0]["fixture_label"]
    elif attack == "mixed_latent_truth":
        rows[1]["latent_truth"] = min(1.0, rows[1]["latent_truth"] + 0.01)
    elif attack == "mixed_generator_regime":
        replacement = next(
            regime
            for regime in REGIMES
            if regime != rows[1]["candidate_inputs"]["generator_regime"]
        )
        rows[1]["candidate_inputs"]["generator_regime"] = replacement
        rows[1]["lineage"]["generator_regime"] = replacement
    elif attack == "destroyed_class_support":
        target = rows[0]
        for row in rows:
            if (
                row["output_type"] == target["output_type"]
                and row["cohort_id"] == target["cohort_id"]
                and row["candidate_inputs"]["generator_regime"]
                == target["candidate_inputs"]["generator_regime"]
            ):
                row["fixture_label"] = 0
    elif attack == "conditional_volume_separation":
        target = rows[0]
        for row in rows:
            if (
                row["output_type"] == target["output_type"]
                and row["cohort_id"] == target["cohort_id"]
                and row["candidate_inputs"]["generator_regime"]
                == target["candidate_inputs"]["generator_regime"]
            ):
                label = row["fixture_label"]
                row["volume_inputs"] = {
                    "volume_signal": 0.1 + 0.8 * label,
                    "sample_size": 10 + 10 * label,
                    "game_count": 10 + 10 * label,
                    "pick_rate": 0.1 + 0.8 * label,
                    "play_rate": 0.1 + 0.8 * label,
                    "popularity": 0.1 + 0.8 * label,
                }
    else:
        second_series = rows[2]["series_id"]
        for row in rows:
            if row["series_id"] == second_series:
                row["issued"] = rows[1]["event"]
    with pytest.raises(ValueError):
        build_prequential_plan(rows)


def test_plan_verifier_rejects_test_reuse_and_changed_fold_count(
    authority: VerifiedFoundationAuthority,
) -> None:
    rows = authority.benchmark["rows"]
    attacked = deepcopy(authority.benchmark["prequential_plan"])
    attacked["folds"][1]["test_series_ids"] = attacked["folds"][0]["test_series_ids"]
    with pytest.raises(ValueError):
        verify_prequential_plan(rows, attacked)
    with pytest.raises(ValueError, match="exactly three"):
        build_prequential_plan(rows, chronological_folds=2)


def test_regimes_are_neutral_and_balanced_per_cell_cohort(
    authority: VerifiedFoundationAuthority,
) -> None:
    rows = authority.benchmark["rows"]
    series_rows = {}
    for row in rows:
        series_rows.setdefault(row["series_id"], row)
    for cohort in ("initial_train", "test_fold_0", "test_fold_1", "test_fold_2"):
        for output_type, _ in OUTPUT_STRATA:
            counts = {regime: 0 for regime in REGIMES}
            for row in series_rows.values():
                if row["cohort_id"] == cohort and row["output_type"] == output_type:
                    counts[row["candidate_inputs"]["generator_regime"]] += 1
            assert max(counts.values()) - min(counts.values()) <= 1
            assert all(value > 0 for value in counts.values())


def test_fixture_label_dgp_is_series_atomic_balanced_and_not_proper_score(
    authority: VerifiedFoundationAuthority,
) -> None:
    by_group: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    by_series: dict[str, list[dict[str, object]]] = {}
    for row in authority.benchmark["rows"]:
        by_series.setdefault(row["series_id"], []).append(row)
    for members in by_series.values():
        assert len(members) == 2
        assert len({row["fixture_label"] for row in members}) == 1
        assert len({canonical_sha256(row["fixture_label_dgp"]) for row in members}) == 1
        first = members[0]
        key = (
            first["output_type"],
            first["cohort_id"],
            first["candidate_inputs"]["generator_regime"],
        )
        by_group.setdefault(key, []).append(first)
    for (_, _, _regime), series in by_group.items():
        ordered = sorted(
            series,
            key=lambda row: row["fixture_label_dgp"]["support_stratum"],
        )
        assert [
            row["fixture_label_dgp"]["support_stratum"] for row in ordered
        ] == list(range(8))
        assert [row["fixture_label"] for row in ordered] == [0] * 4 + [1] * 4
        probabilities = [row["fixture_label_dgp"]["probability"] for row in ordered]
        assert probabilities == sorted(probabilities)
        assert all(
            row["fixture_label_dgp"]["sampling_weight"] == 1.0
            for row in ordered
        )
        assert all(
            row["fixture_label_dgp"]["target_kind"]
            == "balanced_fixture_classification"
            and row["fixture_label_dgp"]["probability_semantics"]
            == "fixture_class_probability"
            and row["fixture_label_dgp"]["proper_score_eligible"] is False
            and row["fixture_label_dgp"]["probability"] != row["latent_truth"]
            for row in ordered
        )


def test_volume_fields_do_not_perfectly_predict_fixture_target_at_any_scope(
    authority: VerifiedFoundationAuthority,
) -> None:
    audit = authority.benchmark["prequential_plan"][
        "volume_target_nonseparability"
    ]
    assert audit["target_kind"] == "balanced_fixture_classification"
    assert audit["passes"] is True
    assert audit["scope_count"] == 101
    assert len(audit["scopes"]) == 101
    assert all(
        scope["passes"] is True
        and all(
            field["perfect_prediction"] is False
            and field["exact_lookup_accuracy"] == 0.5
            and field["mixed_value_count"] > 0
            for field in scope["fields"]
        )
        for scope in audit["scopes"]
    )


def test_fixture_label_cannot_be_retyped_for_proper_scoring(
    authority: VerifiedFoundationAuthority,
) -> None:
    rows = deepcopy(authority.benchmark["rows"])
    rows[0]["fixture_label_dgp"]["proper_score_eligible"] = True
    with pytest.raises(ValidationFailure, match="proper scores"):
        foundation._validate_rows(rows, inference_call=infer_beta_binomial)


def test_fixture_rows_cannot_enter_predictive_target_path(
    authority: VerifiedFoundationAuthority,
) -> None:
    with pytest.raises(
        ValidationFailure,
        match="cannot serve as predictive examples",
    ):
        foundation.require_predictive_target_authority(
            authority.benchmark["rows"],
        )

    attacked = deepcopy(authority.benchmark["rows"])
    attacked[0]["observed_outcome"] = attacked[0]["fixture_label"]
    with pytest.raises(ValidationFailure, match="row shape mismatch"):
        foundation._validate_rows(attacked, inference_call=infer_beta_binomial)


def test_full_foundation_has_no_predictive_or_winner_semantics(
    authority: VerifiedFoundationAuthority,
) -> None:
    payload = {
        "authority": authority.authority,
        "config": authority.config,
        "registry": authority.candidate_registry,
        "benchmark": authority.benchmark,
    }
    serialized = json.dumps(payload, sort_keys=True).lower()
    for forbidden in (
        '"observed_outcome"',
        '"future_outcome"',
        "future game outcome",
        "game outcome",
        '"winner"',
        '"selected"',
        '"candidate_selection_hard_gate"',
    ):
        assert forbidden not in serialized
    assert all(
        row["fixture_label_dgp"]["proper_score_eligible"] is False
        and row["fixture_label_dgp"]["target_kind"]
        == "balanced_fixture_classification"
        and row["fixture_label_dgp"]["probability_semantics"]
        == "fixture_class_probability"
        for row in authority.benchmark["rows"]
    )
    implementation_text = "\n".join(
        (
            (
                ROOT
                / "lol_kills/v2/evaluation/r20_foundation.py"
            ).read_text(),
            (
                ROOT
                / "lol_kills/v2/evaluation/r20_foundation_generator.py"
            ).read_text(),
        ),
    ).lower()
    assert "observed_outcome" not in implementation_text
    assert "future_outcome" not in implementation_text
    assert "future game outcome" not in implementation_text
    assert "game outcome" not in implementation_text


def test_source_context_candidates_are_typed_distinct_and_conservative(
    authority: VerifiedFoundationAuthority,
) -> None:
    observed_pairs = set()
    for row in authority.benchmark["rows"]:
        strict = replay_foundation_row_candidate(
            authority=authority,
            row_id=row["row_id"],
            method_id="source_context_strict_v2",
        )["value"]
        partial = replay_foundation_row_candidate(
            authority=authority,
            row_id=row["row_id"],
            method_id="source_context_typed_partial_v1",
        )["value"]
        observed_pairs.add((strict["status"], partial["status"]))
        if row["candidate_inputs"]["source_context_pattern"] != "all_good":
            assert strict["high_eligible"] is False
            assert partial["high_eligible"] is False
        if row["candidate_inputs"]["fallback_registry"]["used"]:
            assert strict["high_eligible"] is False
            assert partial["status"] == "unavailable"
    assert ("unavailable", "limited") in observed_pairs
    assert len(observed_pairs) >= 3


def test_fallback_used_profile_inconsistency_rejects() -> None:
    dependencies = {
        "source_lineage": {"complete": True, "registered": True},
        "context_registry": {
            "registered": True,
            "registry_version": "r20-context-v1",
            "path": "player_rating:test",
        },
        "fallback_registry": {"used": False, "profile": "fallback"},
        "bridge_registry": {
            "registered": True,
            "bridge_id": "r20-synthetic-bridge-v1",
        },
    }
    with pytest.raises(ValidationFailure, match="inconsistent"):
        replay_foundation_method(
            method_id="source_context_strict_v2",
            dependencies=dependencies,
            boundaries={"fallback_forbids_high": True},
        )


def test_precision_candidates_are_nonaffine_and_rank_distinct(
    authority: VerifiedFoundationAuthority,
) -> None:
    central: list[float] = []
    robust: list[float] = []
    rejected = 0
    equality = 0
    for row in authority.benchmark["rows"]:
        try:
            left = replay_foundation_row_candidate(
                authority=authority,
                row_id=row["row_id"],
                method_id="central_interval_contraction_v2",
            )["value"]
            right = replay_foundation_row_candidate(
                authority=authority,
                row_id=row["row_id"],
                method_id="robust_mad_contraction_v1",
            )["value"]
            central.append(left)
            robust.append(right)
            equality += int(left == 0.0 or right == 0.0)
        except ValidationFailure:
            rejected += 1
    assert rejected > 0
    assert equality > 0
    assert len(central) > 100
    assert not np.array_equal(np.argsort(central), np.argsort(robust))
    coefficients = np.polyfit(np.asarray(central), np.asarray(robust), 1)
    residual = np.asarray(robust) - np.polyval(coefficients, np.asarray(central))
    assert np.max(np.abs(residual)) > 1e-6


def test_probability_draw_support_and_19_draw_attack_reject() -> None:
    posterior = [0.4] * 256
    prior = [0.3] * 256
    for bad in (-0.1, 1.1, float("nan"), True, "0.4"):
        attacked = posterior.copy()
        attacked[0] = bad
        with pytest.raises(ValidationFailure):
            replay_foundation_method(
                method_id="posterior_mean_displacement_v1",
                dependencies={"posterior_draws": attacked, "prior_draws": prior},
                boundaries={"minimum_draws": 256},
            )
    with pytest.raises(ValidationFailure, match="insufficient"):
        replay_foundation_method(
            method_id="posterior_mean_displacement_v1",
            dependencies={
                "posterior_draws": posterior[:19],
                "prior_draws": prior[:19],
            },
            boundaries={"minimum_draws": 256},
        )


def test_config_and_candidate_central_mass_mismatch_rejects(
    authority: VerifiedFoundationAuthority,
) -> None:
    config = authority.config
    config["central_mass"] = 0.50
    with pytest.raises(ValidationFailure, match="central_mass"):
        foundation._validate_config(
            config,
            foundation._source_hashes(ROOT),
            monte_carlo_call=monte_carlo_width_design,
            benchmark_rows=authority.benchmark["rows"],
        )


def test_monte_carlo_design_callable_is_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    authority = load_foundation_artifacts(ROOT)
    forged = deepcopy(authority.config["monte_carlo_design"])
    forged["design_id"] = "forge"

    def fake_design(**kwargs: object) -> dict[str, object]:
        return forged

    monkeypatch.setattr(foundation, "monte_carlo_width_design", fake_design)
    with pytest.raises(ValidationFailure, match="namespace changed"):
        load_foundation_artifacts(ROOT)


def test_volume_quadratic_null_is_baseline_contained_and_full_rank() -> None:
    benchmark = build_r20_benchmark()
    rows_by_id = {row["row_id"]: row for row in benchmark["rows"]}
    for fold in benchmark["prequential_plan"]["folds"]:
        fold_rows = [rows_by_id[row_id] for row_id in fold["train_row_ids"]]
        assert len(fold_rows) > 0
        for output_type, _ in OUTPUT_STRATA:
            series_seen: set[str] = set()
            values: list[float] = []
            for row in fold_rows:
                if (
                    row["output_type"] == output_type
                    and row["candidate_inputs"]["generator_regime"]
                    == "volume_quadratic_null"
                    and row["series_id"] not in series_seen
                ):
                    series_seen.add(row["series_id"])
                    values.append(row["volume_inputs"]["volume_signal"])
            assert len(series_seen) >= 4
            assert len(values) == len(series_seen)
            center = float(np.median(values))
            design = volume_basis_design(values, center=center)
            assert np.linalg.matrix_rank(design) == 3
            assert np.linalg.cond(design) < 500
            centered_square = (np.asarray(values) - 0.5) ** 2
            # External symbolic oracle (Wolfram FullSimplify) confirms this identity for
            # arbitrary fold center c:
            # (v-1/2)^2 = (c-1/2)^2 + 2(c-1/2)(v-c) + (v-c)^2
            theoretical = ((center - 0.5) ** 2) + 2 * (
                center - 0.5
            ) * (np.asarray(values) - center) + (np.asarray(values) - center) ** 2
            assert np.max(np.abs(theoretical - centered_square)) < 1e-14
            explicit_coeff = np.array(
                [
                    (center - 0.5) ** 2,
                    2 * (center - 0.5),
                    1.0,
                ],
            )
            fitted = design @ explicit_coeff
            assert np.max(np.abs(fitted - centered_square)) < 1e-12
            fitted_reg = design @ np.linalg.lstsq(design, centered_square, rcond=None)[0]
            assert np.max(np.abs(fitted_reg - centered_square)) < 1e-12


def test_fold_volume_readiness_is_series_weighted_training_only_and_hashed(
    authority: VerifiedFoundationAuthority,
) -> None:
    readiness = authority.benchmark["volume_readiness"]
    assert len(readiness) == 3
    for fold in readiness:
        assert fold["ready"] is True
        assert fold["weighting"] == "one_equal_weight_observation_per_series"
        assert fold["preprocessing"] == "training_only"
        for output in fold["outputs"]:
            assert len(output["training_series_ids"]) == len(
                set(output["training_series_ids"])
            )
            assert output["retained_rank"] == len(output["retained_terms"])
            assert output["achieved_condition"] <= output["condition_bound"]
            assert output["generator_null"]["additional_rank"] == 0
            assert "volume_signal:linear" in output["retained_terms"]
            assert "volume_signal:quadratic" in output["retained_terms"]
            assert len(output["design_sha256"]) == 64
            assert len(output["test_transform_sha256"]) == 64


def test_same_role_hardlink_and_source_alias_reject(tmp_path: Path) -> None:
    original = tmp_path / "source.py"
    alias = tmp_path / "source-alias.py"
    original.write_text("x = 1\n")
    os.link(original, alias)
    with pytest.raises(ValidationFailure, match="hard-linked"):
        foundation._safe_file(tmp_path.resolve(), "source.py")
    with pytest.raises(ValidationFailure, match="hard-linked"):
        foundation._safe_file(tmp_path.resolve(), "source-alias.py")


def test_all_artifacts_and_replays_remain_nonpromotable(
    authority: VerifiedFoundationAuthority,
) -> None:
    assert authority.authority["production_eligible"] is False
    assert authority.benchmark["production_eligible"] is False
    assert authority.candidate_registry["production_eligible"] is False
    assert all(
        payload["production_eligible"] is False
        for payload in authority.dependency_payloads.values()
    )
    row_id = authority.benchmark["rows"][0]["row_id"]
    assert replay_foundation_row_candidate(
        authority=authority,
        row_id=row_id,
        method_id="posterior_mean_displacement_v1",
    )["production_eligible"] is False


def test_no_winner_or_hard_gate_is_present(authority: VerifiedFoundationAuthority) -> None:
    serialized = json.dumps(
        {
            "config": authority.config,
            "registry": authority.candidate_registry,
            "benchmark": authority.benchmark,
        }
    )
    assert '"selected"' not in serialized
    assert "hard_gate" not in serialized
