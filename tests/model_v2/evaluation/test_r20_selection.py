from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from lol_kills.v2.evaluation import r20_selection as selection_module
from lol_kills.v2.evaluation.checks import ValidationFailure
from lol_kills.v2.evaluation.r20_selection import (
    AUTHORITY_LOCATOR,
    CANDIDATES,
    CONFIG_LOCATOR,
    HARD_GATES,
    REPORT_LOCATOR,
    ROWS_LOCATOR,
    VerifiedR20SelectionAuthority,
    audit_incremental_design,
    build_selection_report,
    evidence_modulation_column,
    forecast_prefix_sha256,
    load_r20_selection_authority,
    replay_r20_selection,
    replay_cutoff_forecasts,
    replay_wolfram_oracle,
    validate_candidate_replays,
    validate_predictive_rows,
)
from lol_kills.v2.evaluation.types import canonical_json, canonical_sha256


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def authority() -> VerifiedR20SelectionAuthority:
    return load_r20_selection_authority(ROOT)


def test_loader_issues_replayable_development_only_authority(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    report = replay_r20_selection(authority)
    assert report == authority.report
    assert report["synthetic_only"] is True
    assert report["development_only"] is True
    assert report["production_eligible"] is False
    assert report["promotion_decision"] is None
    assert len(report["hard_gates"]) == len(HARD_GATES)
    assert set(report["hard_gates"]) == set(HARD_GATES)
    assert all(item["status"] == "pass" for item in report["hard_gates"].values())
    with pytest.raises(TypeError):
        VerifiedR20SelectionAuthority()
    with pytest.raises(ValidationFailure, match="loader-issued"):
        replay_r20_selection(object())  # type: ignore[arg-type]


def test_fresh_process_callable_warmup_keeps_structural_fingerprints_stable() -> None:
    code = """
from pathlib import Path
import lol_kills.v2.evaluation.r20_selection as r
root = Path.cwd()
config = r.build_selection_config(root)
rows = r.build_predictive_rows(config)
r.build_selection_report(config, rows)
authority = r.load_r20_selection_authority(root)
assert r.replay_r20_selection(authority) == authority.report
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(ROOT),
        },
        text=True,
        capture_output=True,
        check=False,
        timeout=90,
    )
    assert completed.returncode == 0, completed.stderr


def test_scipy_lazy_lapack_initialization_keeps_authority_stable() -> None:
    selection_module.scipy.linalg.lapack.get_lapack_funcs(
        ("gesv",), (np.eye(2),)
    )
    loaded = load_r20_selection_authority(ROOT)
    assert replay_r20_selection(loaded) == loaded.report


def test_separate_predictive_authority_has_proper_resolution_only_target(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    rows = validate_predictive_rows(authority.config, authority.rows)
    assert len(rows) == 5 * 90 * 2
    assert {row["target_kind"] for row in rows} == {"observed_outcome"}
    assert {row["observed_outcome"] for row in rows} == {0, 1}
    assert all(row["outcome_visible_at"] == row["resolved_at"] for row in rows)
    assert not any("fixture_label" in row for row in rows)
    outcomes = [row["observed_outcome"] for row in rows]
    assert outcomes != [index % 2 for index in range(len(outcomes))]
    assert outcomes != [1 - index % 2 for index in range(len(outcomes))]


def test_foundation_fixture_injection_and_retyping_are_rejected(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    rows = authority.rows["rows"]
    injected = deepcopy(rows)
    injected[0]["fixture_label"] = injected[0]["observed_outcome"]
    with pytest.raises(ValidationFailure, match="2B1 fixture"):
        selection_module._validate_predictive_rows_internal_consistency(injected)
    retyped = deepcopy(rows)
    retyped[0]["target_kind"] = "fixture_label"
    with pytest.raises(ValidationFailure, match="target kind"):
        selection_module._validate_predictive_rows_internal_consistency(retyped)


def test_deterministic_class_disguised_as_outcome_is_rejected(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    rows = authority.rows["rows"]
    disguised = deepcopy(rows)
    for row in disguised:
        row["observed_outcome"] = 0
    with pytest.raises(ValidationFailure, match="deterministic balanced-class"):
        selection_module._validate_predictive_rows_internal_consistency(disguised)


def test_deterministic_alternating_class_is_rejected(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    alternating = deepcopy(authority.rows["rows"])
    for index, row in enumerate(alternating):
        row["observed_outcome"] = index % 2
    with pytest.raises(ValidationFailure, match="deterministic alternating"):
        selection_module._validate_predictive_rows_internal_consistency(alternating)


def test_fake_dgp_seed_probability_or_draw_is_rejected(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    for field, value in (
        ("latent_probability", 1.0),
        ("uniform_draw", 0.123456789),
    ):
        attacked = deepcopy(authority.rows["rows"])
        attacked[0]["dgp"][field] = value
        with pytest.raises(ValidationFailure, match="DGP|outcome"):
            selection_module._validate_predictive_rows_internal_consistency(attacked)
    attacked = deepcopy(authority.rows["rows"])
    attacked[0]["dgp"]["stream"]["seed"] += 1
    with pytest.raises(ValidationFailure, match="outcome does not replay"):
        selection_module._validate_predictive_rows_internal_consistency(attacked)


def _interior_probability_attack(payload: dict[str, object]) -> None:
    rows = payload["rows"]
    row = next(  # type: ignore[call-overload]
        item
        for item in rows  # type: ignore[union-attr]
        if item["dgp"]["latent_probability"] != item["dgp"]["uniform_draw"]
    )
    probability = float(row["dgp"]["latent_probability"])
    uniform = float(row["dgp"]["uniform_draw"])
    attacked_probability = (probability + uniform) / 2.0
    assert int(uniform < attacked_probability) == row["observed_outcome"]
    row["dgp"]["latent_probability"] = attacked_probability
    row["dgp"]["outcome_replay"] = row["observed_outcome"]
    payload["rows_sha256"] = canonical_sha256(rows)


def test_public_validator_rejects_interior_same_outcome_dgp_and_bad_boundary(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    attacked = authority.rows
    _interior_probability_attack(attacked)
    with pytest.raises(ValidationFailure, match="frozen generator replay"):
        validate_predictive_rows(authority.config, attacked)
    with pytest.raises(ValidationFailure, match="frozen generator replay"):
        build_selection_report(authority.config, attacked)
    wrong_config = authority.config
    wrong_config["generator"]["seed"] += 1
    with pytest.raises(ValidationFailure, match="exact frozen config"):
        validate_predictive_rows(wrong_config, authority.rows)
    truncated = authority.rows
    truncated["rows"].pop()
    truncated["rows_sha256"] = canonical_sha256(truncated["rows"])
    with pytest.raises(ValidationFailure, match="frozen generator replay"):
        validate_predictive_rows(authority.config, truncated)
    with pytest.raises(ValidationFailure, match="complete rows payload"):
        validate_predictive_rows(authority.rows["rows"])


def test_time_safety_and_chronology_attacks_reject(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    rows = authority.rows["rows"]
    future_feature = deepcopy(rows)
    future_feature[0]["feature_available_at"]["volume_signal"] = future_feature[0][
        "event_start"
    ]
    with pytest.raises(ValidationFailure, match="strictly before"):
        selection_module._validate_predictive_rows_internal_consistency(
            future_feature
        )
    swapped = deepcopy(rows)
    swapped[0]["issued_at"], swapped[0]["event_start"] = (
        swapped[0]["event_start"],
        swapped[0]["issued_at"],
    )
    with pytest.raises(ValidationFailure, match="chronology"):
        selection_module._validate_predictive_rows_internal_consistency(swapped)


def test_future_label_deletion_and_shuffle_cannot_change_actual_earlier_fits(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    rows = authority.rows["rows"]
    expected = replay_cutoff_forecasts(rows)
    attacked = deepcopy(rows)
    future = [
        row
        for row in attacked
        if int(row["series_id"].rsplit(":", 1)[1]) >= 50
    ]
    labels = [row["observed_outcome"] for row in future][::-1]
    for row, label in zip(future, labels):
        row["observed_outcome"] = label
    assert replay_cutoff_forecasts(attacked) == expected
    deleted = [
        row
        for row in rows
        if int(row["series_id"].rsplit(":", 1)[1]) < 50
    ]
    assert replay_cutoff_forecasts(deleted) == expected


def test_fold_test_labels_never_enter_own_fit_or_prediction_bytes(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    rows = authority.rows["rows"]
    expected = replay_cutoff_forecasts(rows)
    attacked = deepcopy(rows)
    for row in attacked:
        index = int(row["series_id"].rsplit(":", 1)[1])
        if 30 <= index < 50:
            row["observed_outcome"] = 1 - row["observed_outcome"]
    assert replay_cutoff_forecasts(attacked) == expected


def test_series_atomicity_and_complete_candidate_universe_reject(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    rows = authority.rows["rows"]
    missing_map = deepcopy(rows)
    missing_map.pop(0)
    with pytest.raises(ValidationFailure, match="map-atomic"):
        selection_module._validate_predictive_rows_internal_consistency(missing_map)
    missing_candidate = deepcopy(rows)
    missing_candidate[0]["features"]["candidate_diagnostics"].pop(CANDIDATES[0][0])
    with pytest.raises(ValidationFailure, match="candidate diagnostic universe"):
        selection_module._validate_predictive_rows_internal_consistency(
            missing_candidate
        )
    duplicate_candidate = deepcopy(rows)
    duplicate_candidate[0]["features"]["candidate_diagnostics"]["duplicate"] = 0.0
    with pytest.raises(ValidationFailure, match="candidate diagnostic universe"):
        selection_module._validate_predictive_rows_internal_consistency(
            duplicate_candidate
        )


@pytest.mark.parametrize(
    "diagnostic",
    [
        lambda volume: volume,
        lambda volume: volume**2,
        lambda volume: 3.0 + 2.0 * volume - 4.0 * volume**2,
        lambda volume: np.ones_like(volume),
    ],
)
def test_volume_proxy_and_rank_deficiency_attacks_reject(diagnostic: object) -> None:
    volume = np.linspace(0.05, 0.95, 40)
    raw_logit = np.sin(np.linspace(-2.0, 2.0, 40))
    with pytest.raises(ValidationFailure, match="volume|constant|rank"):
        audit_incremental_design(
            volume,
            raw_logit,
            diagnostic(volume),  # type: ignore[operator]
        )


def test_unsigned_evidence_modulates_but_cannot_invent_or_reverse_direction() -> None:
    raw = np.asarray([-2.0, -0.5, 0.0, 0.5, 2.0])
    diagnostic = np.asarray([0.1, 0.4, 0.8, 0.4, 0.1])
    column = evidence_modulation_column(
        raw,
        diagnostic,
        diagnostic_center=float(np.mean(diagnostic)),
        diagnostic_scale=float(np.std(diagnostic)),
    )
    swapped = evidence_modulation_column(
        -raw,
        diagnostic,
        diagnostic_center=float(np.mean(diagnostic)),
        diagnostic_scale=float(np.std(diagnostic)),
    )
    assert swapped == pytest.approx(-column, abs=1e-12)
    assert column[2] == pytest.approx(0.0, abs=0.0)


def test_fitted_positive_modulation_is_odd_direction_safe_at_extremes() -> None:
    raw_train = np.asarray([-3.0, -1.0, -0.2, 0.2, 1.0, 3.0] * 8)
    unsigned = np.column_stack(
        [
            np.ones(raw_train.size),
            np.linspace(-2.0, 2.0, raw_train.size),
            np.linspace(-2.0, 2.0, raw_train.size) ** 2,
        ]
    )
    outcomes = (raw_train > 0).astype(float)
    parameters = selection_module._fit_positive_modulation(
        unsigned, raw_train, outcomes
    )
    adversarial = np.asarray(
        [[1.0, 0.0, 0.0], [1.0, 1.0e6, -1.0e6], [1.0, -1.0e6, 1.0e6]]
    )
    raw = np.asarray([1.0e-9, 1.0, 1.0e6])
    positive = selection_module._positive_modulation_probability(
        raw, adversarial, parameters
    )
    negative = selection_module._positive_modulation_probability(
        -raw, adversarial, parameters
    )
    zero = selection_module._positive_modulation_probability(
        np.zeros(3), adversarial, parameters
    )
    assert np.all(positive > 0.5)
    assert np.all(negative < 0.5)
    assert positive + negative == pytest.approx(np.ones(3), abs=1e-15)
    assert zero == pytest.approx(np.full(3, 0.5), abs=0.0)


def test_wolfram_paired_loss_rank_and_ess_oracle_replays() -> None:
    oracle = replay_wolfram_oracle()
    assert oracle["design_rank"] == 4
    assert oracle["effective_support"] == pytest.approx(3.0, abs=1e-12)
    assert oracle["overall_paired_delta"] == pytest.approx(
        -0.08406765369738241, abs=1e-12
    )
    assert oracle["series_paired_contributions"] == pytest.approx(
        [
            -0.07951095648295095,
            -0.07410797215372178,
            -0.09858403245547451,
        ],
        abs=1e-12,
    )


def test_every_diagnostic_replays_exact_registered_candidate(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    records = validate_candidate_replays(authority.config, authority.rows)
    assert len(records) == 5 * 90
    for record in records:
        assert record["foundation_rows_consumed"] is False
        assert set(record["replays"]) == {method for method, _ in CANDIDATES}
        for method_id, replay in record["replays"].items():
            execution = authority.config["candidate_execution"][method_id]
            assert replay["executed_boundary_sha256"] == execution[
                "boundary_sha256"
            ]
            assert replay["implementation_source_sha256"] == execution[
                "implementation"
            ]["source_sha256"]
        assert record["candidate_input"]["source"] == "inference_output_only"
        assert not any(
            "latent" in str(key).lower()
            for key in selection_module._walk_mapping_keys(
                record["candidate_input"]
            )
        )


def _rehash_series_records(payload: dict[str, object], index: int = 0) -> None:
    records = payload["series_records"]
    record = records[index]  # type: ignore[index]
    unsigned = {
        key: value
        for key, value in record.items()  # type: ignore[union-attr]
        if key != "record_sha256"
    }
    record["record_sha256"] = canonical_sha256(unsigned)  # type: ignore[index]
    payload["series_records_sha256"] = canonical_sha256(records)


@pytest.mark.parametrize("edge", ["latent", "observation", "inference", "dependency"])
def test_generator_observation_inference_candidate_edges_replay_or_reject(
    authority: VerifiedR20SelectionAuthority, edge: str
) -> None:
    attacked = authority.rows
    record = attacked["series_records"][0]
    if edge == "latent":
        record["generator_state"]["latent_success_probability"] = 0.91
    elif edge == "observation":
        record["observation"]["observed_outcomes"][0] ^= 1
    elif edge == "inference":
        record["inference"]["posterior_parameters"]["alpha"] += 1.0
    else:
        record["candidate_input"]["dependencies"]["posterior_draws"][0] += 0.01
    _rehash_series_records(attacked)
    with pytest.raises(ValidationFailure, match="lineage|replay|hash"):
        validate_candidate_replays(authority.config, attacked)


def test_substituted_diagnostic_with_same_method_id_rejects(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    attacked = authority.rows
    method_id = CANDIDATES[0][0]
    attacked["series_records"][0]["replays"][method_id]["adapter_value"] += 0.01
    unsigned = {
        key: value
        for key, value in attacked["series_records"][0].items()
        if key != "record_sha256"
    }
    attacked["series_records"][0]["record_sha256"] = canonical_sha256(unsigned)
    attacked["series_records_sha256"] = canonical_sha256(
        attacked["series_records"]
    )
    with pytest.raises(ValidationFailure, match="parity"):
        validate_candidate_replays(authority.config, attacked)
    attacked = authority.rows
    attacked["rows"][0]["features"]["candidate_diagnostics"][method_id] += 0.01
    with pytest.raises(ValidationFailure, match="substituted"):
        validate_candidate_replays(authority.config, attacked)


def test_measurements_are_family_local_paired_and_dependence_aware(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    report = authority.report
    assert len(report["measurements"]) == 5 * 6
    assert len(report["selections"]) == 5 * 3
    for measurement in report["measurements"]:
        assert measurement["candidate_eligibility"] == "eligible"
        assert measurement["adequacy"] == "unavailable_dependence_support"
        assert measurement["production_eligibility"].startswith("ineligible_")
        assert measurement["primary"]["metric"] == "log_loss"
        assert measurement["secondary"]["metric"] == "brier"
        assert len(measurement["primary"]["paired_series_contributions"]) == 60
        assert measurement["dependence"]["unit"] == "series"
        assert measurement["dependence"]["effective_support"] is None
        assert (
            measurement["primary"]["descriptive_sequential_refit"][
                "unconditional_inference"
            ]
            is False
        )
        assert (
            measurement["dependence"][
                "authoritative_selection_uses_series_iid_bootstrap"
            ]
            is False
        )
        assert measurement["dependence"]["naive_map_standard_error"] is None
        for fold in measurement["fold_evidence"]:
            assert fold["fit_scope_series_ids"] == fold["train_series_ids"]
            assert not set(fold["train_series_ids"]) & set(fold["test_series_ids"])
            assert fold["design_audit"]["additional_rank"] == 1
            assert fold["design_audit"]["base_rank"] == 3
            assert fold["baseline_direction_audit"]["status"] == "pass"
            assert fold["candidate_direction_audit"]["status"] == "pass"


def test_iid_bootstrap_is_non_authoritative_and_leave_one_reconciles(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    for measurement in authority.report["measurements"]:
        primary = measurement["primary"]
        iid = primary["non_authoritative_series_iid_diagnostic"]["interval"]
        envelope = primary["descriptive_sequential_refit"]
        assert iid != [
            envelope["lower_observed_fold_delta"],
            envelope["upper_observed_fold_delta"],
        ]
        values = np.asarray(primary["paired_series_contributions"], dtype=float)
        leave_one = np.asarray(
            [np.mean(np.delete(values, index)) for index in range(values.size)]
        )
        sensitivity = measurement["dependence"]["sensitivity"]
        assert sensitivity["method"] == "full_leave_one_series_cluster"
        assert sensitivity["reconciled_count"] == values.size
        assert sensitivity["minimum_delta"] == pytest.approx(np.min(leave_one))
        assert sensitivity["maximum_delta"] == pytest.approx(np.max(leave_one))
        assert sensitivity["worst_case_delta"] == pytest.approx(np.max(leave_one))


def test_selection_is_preregistered_and_may_return_no_winner(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    selections = authority.report["selections"]
    assert all(item["selected_method_id"] is None for item in selections)
    assert len(selections) == 15
    assert all(
        item["blocking_reasons"] == ["unavailable_dependence_support"]
        for item in selections
    )
    for item in selections:
        assert item["production_eligible"] is False
        if item["selected_method_id"] is None:
            assert item["status"] == "unavailable_no_eligible_winner"
        else:
            assert item["status"] == "selected_development"


def test_favorable_fold_deltas_cannot_bypass_missing_dependence_support(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    measurements = deepcopy(authority.report["measurements"])
    for measurement in measurements:
        measurement["primary"]["candidate_minus_volume_baseline"] = -1.0
        sequential = measurement["primary"]["descriptive_sequential_refit"]
        sequential["fold_deltas"] = [-1.0, -1.0, -1.0]
        sequential["maximum_observed_fold_delta"] = -1.0
        measurement["adequacy"] = "unavailable_dependence_support"
        measurement["dependence"]["effective_support"] = None
    selections = selection_module._apply_selection(measurements)
    assert all(item["selected_method_id"] is None for item in selections)


def test_behavioral_controls_are_non_authoritative_smoke_checks_only(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    checks = authority.report["behavioral_smoke_checks"]
    assert checks["artifact_role"] == (
        "non_authoritative_adapter_rule_behavioral_smoke_check"
    )
    assert checks["neutral_selection_influence"] is False
    assert checks["selector_validation"] is False
    assert checks["type_i_error_control"] is False
    assert checks["power_validation"] is False
    assert checks["registered_candidate_lineage_exercised"] is False
    assert set(checks["regimes"]) == {"null", "positive", "placebo"}
    assert all(
        item["status"] == "descriptive_only"
        and item["threshold"] is None
        for item in checks["regimes"].values()
    )
    oracle = authority.config["wolfram_smoke_interval_oracle"]
    for regime, item in checks["regimes"].items():
        assert item["selected_count"] == oracle["counts"][regime]
        assert item["seed_count"] == oracle["trials"]
        assert item["descriptive_wilson_95_interval"] == pytest.approx(
            oracle["wilson_95_intervals"][regime], abs=1e-15
        )


def _copy_authority_tree(tmp_path: Path) -> Path:
    selected = [
        "lol_kills/v2/evaluation/r20_selection.py",
        "lol_kills/v2/evaluation/generate_r20_selection_artifacts.py",
        "lol_kills/v2/evaluation/r20_foundation_algorithms.py",
        "lol_kills/v2/evaluation/checks.py",
        "lol_kills/v2/evaluation/types.py",
        "data/lol/v2/evaluation/b2/r20-selection-config.json",
        "data/lol/v2/evaluation/b2/r20-selection-predictive-rows.json",
        "data/lol/v2/evaluation/b2/r20-selection-report.json",
        "data/lol/v2/evaluation/b2/r20-selection-authority.json",
        "data/lol/v2/evaluation/b2/r20-foundation-authority.json",
        "data/lol/v2/evaluation/b2/r20-foundation-evidence-candidate-registry.json",
    ]
    for relative in selected:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    return tmp_path


def _self_rehash_report(root: Path, mutate: object) -> None:
    report_path = root / REPORT_LOCATOR
    report = json.loads(report_path.read_text())
    mutate(report)  # type: ignore[operator]
    report_raw = (canonical_json(report) + "\n").encode()
    report_path.write_bytes(report_raw)
    authority_path = root / AUTHORITY_LOCATOR
    authority = json.loads(authority_path.read_text())
    authority["report_ref"]["raw_sha256"] = hashlib.sha256(report_raw).hexdigest()
    authority["report_ref"]["canonical_payload_sha256"] = canonical_sha256(report)
    authority_path.write_text(canonical_json(authority) + "\n")


def test_loader_rejects_self_rehashed_interior_probability_attack(
    tmp_path: Path,
) -> None:
    root = _copy_authority_tree(tmp_path)
    rows_path = root / ROWS_LOCATOR
    payload = json.loads(rows_path.read_text())
    _interior_probability_attack(payload)
    rows_raw = (canonical_json(payload) + "\n").encode()
    rows_path.write_bytes(rows_raw)
    authority_path = root / AUTHORITY_LOCATOR
    authority_payload = json.loads(authority_path.read_text())
    authority_payload["rows_ref"]["raw_sha256"] = hashlib.sha256(
        rows_raw
    ).hexdigest()
    authority_payload["rows_ref"]["canonical_payload_sha256"] = canonical_sha256(
        payload
    )
    authority_path.write_text(canonical_json(authority_payload) + "\n")
    with pytest.raises(ValidationFailure, match="predictive rows do not replay"):
        load_r20_selection_authority(root)


@pytest.mark.parametrize(
    "attack",
    [
        lambda report: report["measurements"][0]["fold_evidence"][0].__setitem__(
            "fit_scope_series_ids",
            report["measurements"][0]["fold_evidence"][0]["train_series_ids"]
            + report["measurements"][0]["fold_evidence"][0]["test_series_ids"],
        ),
        lambda report: report["measurements"][0]["dependence"].__setitem__(
            "naive_map_standard_error", 0.001
        ),
        lambda report: report["measurements"][0]["primary"].__setitem__(
            "paired_series_contributions",
            report["measurements"][0]["primary"]["paired_series_contributions"]
            * 2,
        ),
        lambda report: report["selections"][0].__setitem__(
            "selected_method_id", CANDIDATES[0][0]
        ),
        lambda report: report.__setitem__("production_eligible", True),
        lambda report: report.__setitem__("sota", True),
        lambda report: report.__setitem__("reliability", "high"),
        lambda report: report.__setitem__("confidence", 0.99),
    ],
)
def test_self_rehashed_detached_gate_and_claim_attacks_reject(
    tmp_path: Path, attack: object
) -> None:
    root = _copy_authority_tree(tmp_path)
    _self_rehash_report(root, attack)
    with pytest.raises(ValidationFailure):
        load_r20_selection_authority(root)


def test_caller_cannot_posthoc_or_force_winner(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    config = authority.config
    rows = authority.rows
    forced = deepcopy(config)
    forced["selection"]["caller_winner"] = CANDIDATES[0][0]
    with pytest.raises(ValidationFailure, match="frozen"):
        build_selection_report(forced, rows)


def test_gate_cannot_stamp_pass_for_false_predicate() -> None:
    with pytest.raises(ValidationFailure, match="hard gate failed"):
        selection_module._gate("attack", False, {"claimed": True})


def test_authority_constructor_object_new_and_namespace_forgery_reject(
    authority: VerifiedR20SelectionAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(TypeError):
        VerifiedR20SelectionAuthority(_loader_issue=True)
    forged = object.__new__(VerifiedR20SelectionAuthority)
    with pytest.raises(ValidationFailure, match="loader-issued"):
        replay_r20_selection(forged)
    assert not hasattr(selection_module, "_ISSUED")
    monkeypatch.setattr(selection_module, "np", object())
    with pytest.raises(ValidationFailure, match="dependency changed: np"):
        replay_r20_selection(authority)


@pytest.mark.parametrize(
    "name",
    [
        "build_selection_report",
        "build_predictive_rows",
        "replay_foundation_method",
        "FOUNDATION_METHOD_SPECS",
        "_bootstrap_interval",
        "_fold_ids",
        "audit_incremental_design",
        "_sigmoid",
        "_fail",
    ],
)
def test_loader_and_replay_dependency_rebinding_reject(
    authority: VerifiedR20SelectionAuthority,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    monkeypatch.setattr(selection_module, name, object())
    with pytest.raises(ValidationFailure, match="executable dependency changed"):
        replay_r20_selection(authority)


def test_exact_post_load_bootstrap_helper_rebind_rejects(
    authority: VerifiedR20SelectionAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selection_module,
        "_bootstrap_interval",
        lambda values, seed: (-999.0, -999.0),
    )
    with pytest.raises(
        ValidationFailure, match="executable dependency changed: _bootstrap_interval"
    ):
        replay_r20_selection(authority)


@pytest.mark.parametrize("name", ["_bootstrap_interval", "_fold_ids"])
def test_same_identity_callable_code_mutation_rejects_closed(
    authority: VerifiedR20SelectionAuthority, name: str
) -> None:
    function = getattr(selection_module, name)
    original_code = function.__code__

    def evil(*args: object, **kwargs: object) -> object:
        return (-999.0, -999.0)

    try:
        function.__code__ = evil.__code__
        with pytest.raises(
            ValidationFailure, match="recursive callable executable fingerprint"
        ):
            replay_r20_selection(authority)
    finally:
        function.__code__ = original_code


@pytest.mark.parametrize("attribute,value", [
    ("__defaults__", (None,)),
    ("__kwdefaults__", {"inert": None}),
])
def test_callable_defaults_and_kwdefaults_mutation_rejects_closed(
    authority: VerifiedR20SelectionAuthority,
    attribute: str,
    value: object,
) -> None:
    function = selection_module._bootstrap_interval
    original = getattr(function, attribute)
    try:
        setattr(function, attribute, value)
        with pytest.raises(
            ValidationFailure, match="recursive callable executable fingerprint"
        ):
            replay_r20_selection(authority)
    finally:
        setattr(function, attribute, original)


def test_foundation_method_registry_in_place_mutation_rejects_closed(
    authority: VerifiedR20SelectionAuthority,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method_id = CANDIDATES[0][0]
    monkeypatch.setitem(
        selection_module.FOUNDATION_METHOD_SPECS[method_id],
        "inert_attack",
        True,
    )
    with pytest.raises(ValidationFailure, match="method registry content changed"):
        replay_r20_selection(authority)


def test_exact_recomputed_report_comparator_rejects_mismatch() -> None:
    with pytest.raises(ValidationFailure, match="loader-pinned report"):
        selection_module._require_exact_recomputed_report(
            {"report_sha256": "pinned"},
            {"report_sha256": "changed"},
        )


def test_no_reliability_or_universal_scalar_conflation(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    report = authority.report
    serialized = canonical_json(report).lower()
    assert '"confidence":' not in serialized
    assert '"evidence_confidence":' not in serialized
    assert '"correctness":' not in serialized
    assert report["heldout_reliability"] == {
        "status": "not_evaluated_separate_later_authority"
    }


def test_accepted_foundation_invariants_and_bytes_remain_pinned(
    authority: VerifiedR20SelectionAuthority,
) -> None:
    config = authority.config
    foundation_config_raw = (
        ROOT / "data/lol/v2/evaluation/b2/r20-foundation-config.json"
    ).read_bytes()
    foundation_config = json.loads(foundation_config_raw)
    audit = foundation_config["monte_carlo_design"][
        "registered_benchmark_observation_coverage"
    ]
    assert audit["registered_count"] == 345
    assert audit["unique_conditional_cell_count"] == 117
    assert foundation_config["volume_only_basis"]["basis_id"] == "r20-volume-basis-v2"
    assert foundation_config["contract_tree_sha256"] == (
        "8748bbe48b273593b09304ac80923f11384de808b835f6e83e97c6fef48661dd"
    )
    assert hashlib.sha256(
        (ROOT / config["foundation_inputs"]["authority_locator"]).read_bytes()
    ).hexdigest() == config["foundation_inputs"]["authority_raw_sha256"]
    assert hashlib.sha256(
        (ROOT / config["foundation_inputs"]["candidate_registry_locator"]).read_bytes()
    ).hexdigest() == config["foundation_inputs"]["candidate_registry_raw_sha256"]
    assert set(config["source_closure"]) == {
        "r20_selection",
        "generate_r20_selection_artifacts",
        "r20_foundation_algorithms",
        "checks",
        "types",
    }
