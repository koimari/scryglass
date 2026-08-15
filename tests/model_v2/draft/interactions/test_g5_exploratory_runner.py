from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import result, runner


CHAMPS = tuple(f"c{number}" for number in range(10))


def _map(key: str, fold: str, *, swap: bool = False, offset: float = 0.0) -> runner.SyntheticMap:
    picks = []
    for index, role in enumerate(runner.ROLES):
        picks.append(("red" if swap else "blue", role, CHAMPS[index]))
        picks.append(("blue" if swap else "red", role, CHAMPS[index + 5]))
    return runner.SyntheticMap(key, fold, f"cluster-{key}", offset, tuple(picks))


def _fit() -> dict[str, object]:
    return runner.fit_d1_train([_map("train", "TRAIN")], {"train": 1})


def _scripted_production_inputs() -> tuple[runner.AlignedInputs, tuple[runner.OutcomeFreeMap, ...], tuple[runner.EvaluationLedger, ...]]:
    folds = ("TRAIN",) * 805 + ("DEVELOPMENT",) * 214 + ("VALIDATION",) * 207
    picks = tuple(
        runner.DraftPick(side, role, CHAMPS[index + (5 if side == "red" else 0)])
        for side in ("blue", "red")
        for index, role in enumerate(runner.ROLES)
    )
    maps = tuple(
        runner.OutcomeFreeMap(
            map_key=f"map-{index:04d}",
            fold=fold,
            source_local_event_start=f"2026-01-{1 + index % 28:02d}T00:00:00",
            cluster_key=f"{fold}-cluster-{index // 2}",
            b0_logit_mean=0.0,
            b0_logit_variance=0.0,
            b0_probability=0.5,
            picks=picks,
        )
        for index, fold in enumerate(folds)
    )
    aligned_maps = tuple(
        SimpleNamespace(source_game_id=item.map_key, ordered_origin_map_ids=())
        for item in maps
    )
    aligned = runner.AlignedInputs(
        maps=aligned_maps,
        feature_rows=tuple({"source_game_id": item.map_key} for item in maps),
        clusters={item.map_key: item.cluster_key for item in maps},
        feature_manifest={},
        cluster_artifact_sha256=runner.contract.CLUSTERS["artifact_canonical_sha256"],
    )
    ledgers = tuple(runner.EvaluationLedger(item.map_key, item.fold, 1) for item in maps)
    return aligned, maps, ledgers


def _scripted_fit() -> dict[str, object]:
    return {
        "beta": np.zeros(1),
        "covariance": np.eye(1),
        "vocabulary": ("c0",),
        "cells": (),
        "objective": 1.0,
        "gradient_inf": 0.0,
        "hessian": np.eye(1),
        "solver": {"iterations": 1, "function_evaluations": 2, "message": "ok"},
    }


def _passed_invariance() -> dict[str, object]:
    return {
        "side_swap": {"status": "PASSED", "map_count": 1226, "absolute_tolerance": 1e-12, "max_absolute_error": 0.0},
        "record_order": {"status": "PASSED", "map_count": 1226, "absolute_tolerance": 1e-12, "max_absolute_error": 0.0},
        "role_relabel": {"status": "NOT_INVARIANT_BY_CONTRACT"},
    }


def _real_payload_from_evidence(evidence: runner.AggregateEvidence) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": result.REAL_SCHEMA,
        "state": evidence.state,
        "blocker": evidence.blocker,
        "selected_candidate": evidence.selected_candidate,
        "counts": dict(evidence.counts),
        "membership_hashes": dict(evidence.membership_hashes),
        "source_and_feature_review_pins": dict(evidence.source_and_feature_review_pins),
        "G2_core_pins": dict(evidence.G2_core_pins),
        "development_metric": dict(evidence.development_metric),
        "validation_metric": dict(evidence.validation_metric),
        "bootstrap": dict(evidence.bootstrap),
        "objective_gradient_hessian_diagnostics": dict(evidence.objective_gradient_hessian_diagnostics),
        "solver_diagnostics": dict(evidence.solver_diagnostics),
        "uncertainty": dict(evidence.uncertainty),
        "prior_only_variance_components": dict(evidence.prior_only_variance_components),
        "coverage_and_prior_only_flags": dict(evidence.coverage_and_prior_only_flags),
        "invariance_tests": dict(evidence.invariance_tests),
        "contribution_reconciliation": dict(evidence.contribution_reconciliation),
        "score_subject": dict(evidence.score_subject),
        "context": dict(evidence.context),
        "execution_binding": {
            "run_id_sha256": "a" * 64,
            "runner_core_sha256": "b" * 64,
            "approval_sha256": "c" * 64,
            "started_entry_sha256": "d" * 64,
            "result_locator": runner.RESULT_LOCATOR,
            "uniqueness_enforcement": "PROCESS_AND_CONTROL_ONLY",
        },
        "execution_limitation": (
            "Run uniqueness is process/control enforced only and provides no G9, public, "
            "concurrent, or adversarial single-use authority."
        ),
        "claim_ceiling": result.CLAIM_CEILING,
    }
    if evidence.winner is not None:
        unsigned.update({
            "private_retrospective_exploratory_score_probability": evidence.winner.neutral_completed_draft_probability,
            "fit_evidence": "TRAIN_ONLY",
            "rank_selection_evidence": "DEVELOPMENT_LOCKED_VALIDATION_GATED",
            "B0_probability": evidence.winner.B0_probability,
            "D1_logit_increment": evidence.winner.D1_logit_increment,
            "neutral_completed_draft_probability": evidence.winner.neutral_completed_draft_probability,
            "probability_increment_over_B0": evidence.winner.probability_increment_over_B0,
            "D1_conditional_interval": dict(evidence.winner.D1_conditional_interval),
        })
    return {**unsigned, "artifact_sha256": result.sha256(unsigned)}


def _scripted_d1_evidence(monkeypatch, *, reverse_validation: bool = False) -> runner.AggregateEvidence:
    aligned, maps, ledgers = _scripted_production_inputs()
    if reverse_validation:
        nonvalidation = tuple(item for item in maps if item.fold != "VALIDATION")
        validation = tuple(reversed([item for item in maps if item.fold == "VALIDATION"]))
        maps = nonvalidation + validation
        ledger_by_key = {item.map_key: item for item in ledgers}
        ledgers = tuple(ledger_by_key[item.map_key] for item in maps)
    monkeypatch.setattr(runner, "build_b0_scores", lambda _aligned: (maps, ledgers))
    monkeypatch.setattr(runner, "fit_d1_train", lambda *_args: _scripted_fit())
    monkeypatch.setattr(runner, "_measure_invariances", lambda *_args: _passed_invariance())

    def score(item: runner.OutcomeFreeMap, _fit: object) -> dict[str, object]:
        prior = (("blue", "top", "c0"),) if item.fold == "VALIDATION" else ()
        return {
            "increment": 0.1,
            "variance": 0.01,
            "prior_only": prior,
            "contributions": (),
            "reconciliation_error": 0.0,
            "fitted_vector": np.zeros(1),
        }

    monkeypatch.setattr(runner, "score_d1", score)
    monkeypatch.setattr(runner, "validation_bootstrap", lambda _deltas: (0.001, np.full(2000, 0.001)))
    return runner.compute_aggregate_evidence(aligned)


def test_objective_gradient_and_hessian_match_finite_difference_oracles() -> None:
    x = np.array([[1.0, -1.0], [-1.0, 1.0], [0.5, 0.25]])
    beta = np.array([0.2, -0.3])
    offset = np.array([0.1, -0.2, 0.4])
    y = np.array([1.0, 0.0, 1.0])
    penalty = np.array([12.5, 50.0])
    value, gradient, hessian = runner.objective_gradient_hessian(beta, x, offset, y, penalty)
    epsilon = 1e-6
    basis = np.eye(2)
    numeric_gradient = np.array([
        (
            runner.objective_gradient_hessian(beta + basis[index] * epsilon, x, offset, y, penalty)[0]
            - runner.objective_gradient_hessian(beta - basis[index] * epsilon, x, offset, y, penalty)[0]
        )
        / (2 * epsilon)
        for index in range(2)
    ])
    numeric_hessian = np.column_stack([
        (
            runner.objective_gradient_hessian(beta + basis[index] * epsilon, x, offset, y, penalty)[1]
            - runner.objective_gradient_hessian(beta - basis[index] * epsilon, x, offset, y, penalty)[1]
        )
        / (2 * epsilon)
        for index in range(2)
    ])
    assert math.isfinite(value)
    assert np.allclose(gradient, numeric_gradient, atol=1e-5)
    assert np.allclose(hessian, numeric_hessian, atol=1e-5)
    assert np.all(np.linalg.eigvalsh(hessian) > 0)


@pytest.mark.parametrize(
    "bad",
    [
        (np.array([np.nan]), np.ones((1, 1)), np.zeros(1), np.zeros(1), np.ones(1)),
        (np.zeros(1), np.ones((1, 1)), np.zeros(1), np.array([2.0]), np.ones(1)),
        (np.zeros(1), np.ones((1, 1)), np.zeros(1), np.zeros(1), np.zeros(1)),
    ],
)
def test_objective_fails_closed_on_nonfinite_label_or_penalty(bad: tuple[np.ndarray, ...]) -> None:
    with pytest.raises(runner.G5RunnerError, match="SOLVER_OR_OBJECTIVE"):
        runner.objective_gradient_hessian(*bad)


def test_train_only_fit_covariance_and_prior_only_variance() -> None:
    fit = _fit()
    assert fit["gradient_inf"] <= 1e-6
    assert np.allclose(fit["covariance"], np.asarray(fit["covariance"]).T, atol=1e-12)
    picks = list(_map("dev", "DEVELOPMENT").picks)
    picks[0], picks[2] = (
        (picks[0][0], picks[0][1], picks[2][2]),
        (picks[2][0], picks[2][1], picks[0][2]),
    )
    scored = runner.score_d1(
        runner.SyntheticMap("dev", "DEVELOPMENT", "cluster", 0.0, tuple(picks)),
        fit,
    )
    assert scored["variance"] >= 0.02
    assert len(scored["prior_only"]) == 2
    assert len(scored["contributions"]) == 10
    assert scored["reconciliation_error"] <= 1e-12


def test_side_swap_and_record_order_invariance() -> None:
    fit = _fit()
    base = _map("dev", "DEVELOPMENT")
    swapped = _map("dev", "DEVELOPMENT", swap=True)
    reversed_order = runner.SyntheticMap(
        "dev",
        "DEVELOPMENT",
        "cluster",
        0.0,
        tuple(reversed(base.picks)),
    )
    left = runner.score_d1(base, fit)
    right = runner.score_d1(swapped, fit)
    reordered = runner.score_d1(reversed_order, fit)
    assert left["increment"] == pytest.approx(-right["increment"], abs=1e-14)
    assert left["variance"] == pytest.approx(right["variance"], abs=1e-14)
    assert left["increment"] == pytest.approx(reordered["increment"], abs=1e-14)


def test_invariance_evidence_is_measured_over_supplied_maps() -> None:
    from lol_kills.v2.ratings.player.model import posterior_predictive_expected_result

    fit = _fit()
    source = _map("dev", "DEVELOPMENT")
    item = runner.OutcomeFreeMap(
        map_key=source.map_key,
        fold=source.fold,
        source_local_event_start="2026-01-01T00:00:00",
        cluster_key=source.cluster_key,
        b0_logit_mean=0.2,
        b0_logit_variance=0.1,
        b0_probability=posterior_predictive_expected_result(0.2, 0.1),
        picks=tuple(runner.DraftPick(*pick) for pick in source.picks),
    )
    measured = runner._measure_invariances((item,), fit)
    assert measured["side_swap"]["status"] == "PASSED"
    assert measured["side_swap"]["map_count"] == 1
    assert measured["record_order"]["max_absolute_error"] <= 1e-12


def test_absent_train_champion_and_evaluation_label_leakage_fail_closed() -> None:
    train = _map("train", "TRAIN")
    with pytest.raises(runner.G5RunnerError, match="strict TRAIN"):
        runner.fit_d1_train([train], {"train": 1, "dev": 0})
    fit = runner.fit_d1_train([train], {"train": 1})
    picks = list(_map("dev", "DEVELOPMENT").picks)
    picks[0] = (picks[0][0], picks[0][1], "never-trained")
    with pytest.raises(runner.G5RunnerError, match="CHAMPION_ABSENT"):
        runner.score_d1(runner.SyntheticMap("dev", "DEVELOPMENT", "c", 0.0, tuple(picks)), fit)


@pytest.mark.parametrize(
    ("mean", "lower", "expected"),
    [
        (0.004999999999, 0.1, False),
        (0.005, 0.1, True),
        (0.005, 0.0, False),
        (0.005, np.nextafter(0.0, 1.0), True),
    ],
)
def test_validation_threshold_edges(mean: float, lower: float, expected: bool) -> None:
    assert runner.d1_validation_wins(mean, lower) is expected


def test_bootstrap_calls_bound_primitive_exactly_2000_times(monkeypatch) -> None:
    seeds: list[int] = []

    def scripted(_summaries: object, seed: int) -> float:
        seeds.append(seed)
        return float(seed % 7) / 100.0

    monkeypatch.setattr(runner, "_bootstrap_replicate", scripted)
    lower, replicates = runner.validation_bootstrap(
        [("m1", "cluster-a", 0.01), ("m2", "cluster-a", 0.02), ("m3", "cluster-b", -0.01)]
    )
    assert seeds == list(range(2026073005, 2026073005 + 2000))
    assert len(replicates) == 2000
    assert lower == np.quantile(replicates, 0.05, method="linear")


def test_bootstrap_missing_or_nonfinite_membership_blocks() -> None:
    with pytest.raises(runner.G5RunnerError, match="BOOTSTRAP_MEMBERSHIP"):
        runner.validation_bootstrap([])
    with pytest.raises(runner.G5RunnerError, match="BOOTSTRAP_MEMBERSHIP"):
        runner.validation_bootstrap([("m", "", 0.0)])
    with pytest.raises(runner.G5RunnerError, match="BOOTSTRAP_MEMBERSHIP"):
        runner.validation_bootstrap([("m", "c", float("nan"))])


def test_synthetic_harness_cannot_promote_or_accept_evaluation_labels_for_fit() -> None:
    maps = [_map("train", "TRAIN"), _map("dev", "DEVELOPMENT"), _map("val", "VALIDATION")]
    outcome = runner.synthetic_execute(
        maps,
        {"train": 1},
        [runner.SyntheticEvaluation("dev", 1), runner.SyntheticEvaluation("val", 1)],
    )
    result.validate_synthetic(outcome)
    assert outcome["state"].startswith("SYNTHETIC_")
    assert outcome["validation"]["called_once"] is True
    forged = dict(outcome)
    forged["state"] = "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER"
    with pytest.raises(ValueError):
        result.validate_synthetic(forged)


def test_fixture_callable_path_has_no_real_schema_capability() -> None:
    assert not hasattr(runner, "write_real_result")
    assert not hasattr(runner, "execute_aligned_pipeline")
    blocked = runner._blocked_evidence(blocker="EXECUTION_BLOCKED:SOLVER_OR_OBJECTIVE_FAILURE")
    assert isinstance(blocked, runner.AggregateEvidence)
    assert not hasattr(blocked, "schema_version")
    assert result.REAL_SCHEMA not in repr(blocked)
    result.validate_real(_real_payload_from_evidence(blocked))


def test_bound_pipeline_calls_only_concrete_runner_owned_loaders_in_order(monkeypatch) -> None:
    events: list[str] = []
    monkeypatch.setattr(runner, "_load_accepted_g1", lambda: events.append("g1") or "g1")
    monkeypatch.setattr(
        runner,
        "_load_accepted_features",
        lambda: events.append("features") or ("manifest", ("row",)),
    )
    monkeypatch.setattr(runner, "_load_accepted_clusters", lambda: events.append("clusters") or "clusters")
    monkeypatch.setattr(
        runner,
        "align_inputs",
        lambda g1, manifest, rows, clusters: events.append(
            f"align:{g1}:{manifest}:{rows[0]}:{clusters}"
        )
        or "aligned",
    )
    monkeypatch.setattr(
        runner,
        "compute_aggregate_evidence",
        lambda aligned: events.append(f"execute:{aligned}") or runner._blocked_evidence(blocker="EXECUTION_BLOCKED:SOLVER_OR_OBJECTIVE_FAILURE"),
    )
    output = runner._execute_bound_pipeline()
    assert isinstance(output, runner.AggregateEvidence)
    assert events == [
        "g1",
        "features",
        "clusters",
        "align:g1:manifest:row:clusters",
        "execute:aligned",
    ]


def test_scripted_production_b0_lock_never_scores_d1_validation(monkeypatch) -> None:
    aligned, maps, ledgers = _scripted_production_inputs()
    score_folds: list[str] = []
    loss_calls: list[tuple[int, float]] = []
    original_loss = runner._probability_logloss
    monkeypatch.setattr(runner, "build_b0_scores", lambda _aligned: (maps, ledgers))
    monkeypatch.setattr(runner, "fit_d1_train", lambda *_args: _scripted_fit())
    monkeypatch.setattr(runner, "_measure_invariances", lambda *_args: _passed_invariance())

    def tied_score(item: runner.OutcomeFreeMap, _fit: object) -> dict[str, object]:
        score_folds.append(item.fold)
        return {"increment": 0.0, "variance": 0.0, "prior_only": (), "contributions": (), "reconciliation_error": 0.0}

    monkeypatch.setattr(runner, "score_d1", tied_score)
    monkeypatch.setattr(
        runner,
        "_probability_logloss",
        lambda label, probability: loss_calls.append((label, probability)) or original_loss(label, probability),
    )
    output = runner.compute_aggregate_evidence(aligned)
    assert output.state == "NO_INCREMENTAL_DRAFT_WINNER"
    assert output.selected_candidate == "B0"
    assert output.bootstrap["status"] == "NOT_RUN_B0_LOCKED"
    assert output.validation_metric["B0_mean_log_loss"] == pytest.approx(-math.log(0.5))
    assert output.validation_metric["map_count"] == 207
    assert set(score_folds) == {"DEVELOPMENT"}
    assert len(loss_calls) == 2 * 214 + 207


@pytest.mark.parametrize(
    ("lower_bound", "expected_state"),
    [(0.001, "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER"), (0.0, "NO_INCREMENTAL_DRAFT_WINNER")],
)
def test_scripted_production_d1_validation_is_scored_once_and_gated(
    lower_bound: float,
    expected_state: str,
    monkeypatch,
) -> None:
    aligned, maps, ledgers = _scripted_production_inputs()
    calls: dict[str, int] = {"DEVELOPMENT": 0, "VALIDATION": 0}
    monkeypatch.setattr(runner, "build_b0_scores", lambda _aligned: (maps, ledgers))
    monkeypatch.setattr(runner, "fit_d1_train", lambda *_args: _scripted_fit())
    monkeypatch.setattr(runner, "_measure_invariances", lambda *_args: _passed_invariance())

    def beneficial_score(item: runner.OutcomeFreeMap, _fit: object) -> dict[str, object]:
        calls[item.fold] += 1
        return {
            "increment": 0.1,
            "variance": 0.01,
            "prior_only": (),
            "contributions": (),
            "reconciliation_error": 0.0,
            "fitted_vector": np.zeros(1),
        }

    monkeypatch.setattr(runner, "score_d1", beneficial_score)
    monkeypatch.setattr(
        runner,
        "validation_bootstrap",
        lambda deltas: (lower_bound, np.full(2000, lower_bound)),
    )
    output = runner.compute_aggregate_evidence(aligned)
    assert output.state == expected_state
    assert calls == {"DEVELOPMENT": 214, "VALIDATION": 207}
    assert output.validation_metric["evaluations"] == 1
    if expected_state.endswith("_WINNER") and expected_state.startswith("PRIVATE"):
        assert output.winner is not None
        assert output.winner.D1_conditional_interval["scale"] == "conditional_mean_validation_logit_increment"
        assert output.winner.score_subject["kind"] == "VALIDATION_COHORT_AGGREGATE"


def test_missing_approval_fails_before_protected_loader(monkeypatch) -> None:
    from lol_kills.v2.draft.interactions.g5_exploratory import execution_approval

    monkeypatch.setattr(runner, "_frozen_prefit", lambda: None)
    monkeypatch.setattr(
        runner,
        "build_runner_review_bundle",
        lambda: {
            "execution-review-core.json": {
                "artifact_sha256": "c" * 64,
            }
        },
    )
    monkeypatch.setattr(
        execution_approval,
        "load_approval",
        lambda **_kwargs: (_ for _ in ()).throw(execution_approval.ApprovalError("missing")),
    )
    called = False

    def protected(**_kwargs: object) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(runner, "_execute_bound_pipeline", protected)
    with pytest.raises(runner.G5RunnerError, match="APPROVAL_INVALID"):
        runner.execute_real("run")
    assert called is False


def test_started_is_appended_before_protected_read(monkeypatch) -> None:
    from lol_kills.v2.draft.interactions.g5_exploratory import execution_approval

    approval = {"approval_id": "approval", "approval_sha256": "a" * 64}
    monkeypatch.setattr(runner, "_frozen_prefit", lambda: None)
    monkeypatch.setattr(
        runner,
        "build_runner_review_bundle",
        lambda: {"execution-review-core.json": {"artifact_sha256": "c" * 64}},
    )
    monkeypatch.setattr(execution_approval, "load_approval", lambda **_kwargs: approval)
    appended: list[dict[str, object]] = []
    monkeypatch.setattr(execution_approval, "load_ledger", lambda: appended)
    monkeypatch.setattr(
        execution_approval,
        "validate_ledger_history",
        lambda entries, **_kwargs: (
            "EMPTY" if not entries else "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY"
        ),
    )

    def append(value: dict[str, object]) -> dict[str, object]:
        payload = {**value, "entry_sha256": "d" * 64}
        appended.append(payload)
        return payload

    def protected() -> object:
        assert appended and appended[0]["state"] == "STARTED"
        raise KeyboardInterrupt

    monkeypatch.setattr(execution_approval, "append_ledger_entry", append)
    monkeypatch.setattr(runner, "_execute_bound_pipeline", protected)
    with pytest.raises(KeyboardInterrupt):
        runner.execute_real("run")


def test_completed_duplicate_records_invalid_and_blocks_before_read(monkeypatch) -> None:
    from lol_kills.v2.draft.interactions.g5_exploratory import execution_approval

    approval = {"approval_id": "approval", "approval_sha256": "a" * 64}
    monkeypatch.setattr(runner, "_frozen_prefit", lambda: None)
    monkeypatch.setattr(
        runner,
        "build_runner_review_bundle",
        lambda: {"execution-review-core.json": {"artifact_sha256": "c" * 64}},
    )
    monkeypatch.setattr(execution_approval, "load_approval", lambda **_kwargs: approval)
    completed = [
        {"state": "STARTED", "approval_id": "approval", "entry_sha256": "d" * 64},
        {
            "state": "COMPLETED",
            "approval_id": "approval",
            "entry_sha256": "e" * 64,
            "result_artifact_sha256": "f" * 64,
            "completed_at": "2026-07-30T00:00:00Z",
        },
    ]
    monkeypatch.setattr(execution_approval, "load_ledger", lambda: completed)
    monkeypatch.setattr(
        execution_approval,
        "validate_ledger_history",
        lambda *_args, **_kwargs: "COMPLETED_TERMINAL",
    )
    monkeypatch.setattr(
        execution_approval,
        "validate_completed_result",
        lambda *_args, **_kwargs: {},
    )
    appended: list[dict[str, object]] = []
    monkeypatch.setattr(
        execution_approval,
        "append_ledger_entry",
        lambda value: appended.append(dict(value)) or dict(value),
    )
    called = False

    def protected() -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(runner, "_execute_bound_pipeline", protected)
    with pytest.raises(runner.G5RunnerError, match="INVALID_DUPLICATE"):
        runner.execute_real("run")
    assert called is False
    assert appended[0]["state"] == "INVALID_DUPLICATE"
    assert appended[0]["authoritative_completed_entry_sha256"] == "e" * 64
    assert appended[0]["authoritative_result_artifact_sha256"] == "f" * 64


def test_strict_real_schema_rejects_minimal_winner_extras_and_identity_leaks() -> None:
    with pytest.raises(ValueError):
        result.validate_real_shape(
            {
                "schema_version": result.REAL_SCHEMA,
                "state": "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER",
            }
        )
    assert "EXECUTION_BLOCKED:SCRIPTED" not in result.BLOCKER_CODES


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["source_and_feature_review_pins"].update({"G1_rows_sha256": "0" * 64}),
        lambda value: value["bootstrap"].update({"replicates": 1999}),
        lambda value: value["solver_diagnostics"].update({"status": "BLOCKED"}),
        lambda value: value["uncertainty"].update({"D1_conditional_covariance": "UNAVAILABLE"}),
        lambda value: value["coverage_and_prior_only_flags"].update({"complete_maps": False}),
        lambda value: value["contribution_reconciliation"].update({"status": "BLOCKED"}),
        lambda value: value["validation_metric"].update({"mean_LL_B0_minus_LL_locked_candidate": 0.004}),
        lambda value: value["objective_gradient_hessian_diagnostics"].update({"hessian_positive_definite": False}),
        lambda value: value["prior_only_variance_components"].update({"total_variance": 0.0}),
        lambda value: value["prior_only_variance_components"].update({"slot_membership_sha256": "0" * 64}),
        lambda value: value["prior_only_variance_components"].update({"signed_exposure_sha256": "0" * 64}),
        lambda value: value["prior_only_variance_components"].update({"role_delta_count": 2, "total_variance": 0.02}),
        lambda value: value.update({"probability_increment_over_B0": 0.0}),
        lambda value: value["counts"].update({"VALIDATION": 206, "TRAIN": 806}),
    ],
)
def test_self_rehashed_semantic_winner_substitution_fails(monkeypatch, mutate) -> None:
    payload = _real_payload_from_evidence(_scripted_d1_evidence(monkeypatch))
    result.validate_real(payload)
    forged = copy.deepcopy(payload)
    forged.pop("artifact_sha256")
    mutate(forged)
    forged["artifact_sha256"] = result.sha256(forged)
    with pytest.raises(ValueError):
        result.validate_real(forged)


def test_string_value_identity_leak_fails_even_when_self_rehashed(monkeypatch) -> None:
    with pytest.raises(ValueError, match="identity-bearing string"):
        result._reject_identity_leaks({"note": "source_game_id=map-0001"})
    payload = _real_payload_from_evidence(_scripted_d1_evidence(monkeypatch))
    forged = copy.deepcopy(payload)
    forged.pop("artifact_sha256")
    forged["context"]["blocker"] = "source_game_id=map-0001"
    forged["artifact_sha256"] = result.sha256(forged)
    with pytest.raises(ValueError):
        result.validate_real(forged)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda witness: witness[0].update({"blue_count": 206}),
        lambda witness: witness[0].update({"net_count": 206}),
        lambda witness: witness[0].update({"validation_map_count": 206}),
        lambda witness: witness.append(copy.deepcopy(witness[0])),
        lambda witness: witness.extend([
            {
                "coordinate_commitment_sha256": "0" * 64,
                "blue_count": 1,
                "red_count": 0,
                "net_count": 1,
                "validation_map_count": 207,
            }
        ]) or witness.reverse(),
    ],
)
def test_self_rehashed_coordinate_witness_mutations_fail(monkeypatch, mutate) -> None:
    payload = _real_payload_from_evidence(_scripted_d1_evidence(monkeypatch))
    forged = copy.deepcopy(payload)
    forged.pop("artifact_sha256")
    mutate(forged["prior_only_variance_components"]["coordinate_exposure_witness"])
    forged["artifact_sha256"] = result.sha256(forged)
    with pytest.raises(ValueError):
        result.validate_real(forged)


def test_validation_order_does_not_change_cohort_score_or_prior_ledger(monkeypatch) -> None:
    first = _scripted_d1_evidence(monkeypatch, reverse_validation=False)
    second = _scripted_d1_evidence(monkeypatch, reverse_validation=True)
    assert first.winner == second.winner
    assert first.prior_only_variance_components == second.prior_only_variance_components
    assert first.prior_only_variance_components["role_delta_count"] == 1
    assert first.prior_only_variance_components["total_variance"] == pytest.approx(0.01)
    assert first.prior_only_variance_components["mean_score_aggregate_variance"] == pytest.approx(0.01)
    commitment = runner._sha({
        "domain": "g5-prior-only-role-champion-coordinate:v1",
        "role": "top",
        "stable_champion_id": "c0",
    })
    assert first.prior_only_variance_components["slot_membership_sha256"] == runner._sha([commitment])
    assert first.prior_only_variance_components["signed_exposure_sha256"] == runner._sha(
        [[commitment, 207, 0, 207, 207]]
    )
    assert first.prior_only_variance_components["coordinate_exposure_witness"] == [{
        "coordinate_commitment_sha256": commitment,
        "blue_count": 207,
        "red_count": 0,
        "net_count": 207,
        "validation_map_count": 207,
    }]


def test_prior_aggregate_counts_one_shared_coordinate_and_separates_signed_exposure() -> None:
    scores = []
    for side in ("blue", "red", "blue", "red", "blue", "red"):
        scores.append((
            None,
            {
                "fitted_vector": np.zeros(1),
                "prior_only": ((side, "top", "c0"),),
            },
        ))
    summary, aggregate_variance = runner._prior_aggregate(scores, _scripted_fit())
    commitment = runner._sha({
        "domain": "g5-prior-only-role-champion-coordinate:v1",
        "role": "top",
        "stable_champion_id": "c0",
    })
    assert summary == {
        "status": "EVALUATED",
        "role_delta_count": 1,
        "variance_per_coordinate": 0.01,
        "total_variance": 0.01,
        "mean_score_aggregate_variance": 0.0,
        "conditional_mean_logit_variance": 0.0,
        "slot_membership_sha256": runner._sha([commitment]),
        "signed_exposure_sha256": runner._sha([[commitment, 3, 3, 0, 6]]),
        "coordinate_exposure_witness": [{
            "coordinate_commitment_sha256": commitment,
            "blue_count": 3,
            "red_count": 3,
            "net_count": 0,
            "validation_map_count": 6,
        }],
    }
    assert aggregate_variance == 0.0
    reversed_summary, reversed_variance = runner._prior_aggregate(
        tuple(reversed(scores)), _scripted_fit()
    )
    assert reversed_summary == summary
    assert reversed_variance == aggregate_variance


def test_prior_aggregate_two_unique_coordinates_use_net_cohort_weights() -> None:
    exposures = (
        (("blue", "top", "c0"),),
        (("blue", "top", "c0"),),
        (("red", "top", "c0"),),
        (("red", "jungle", "c1"),),
    )
    scores = tuple(
        (None, {"fitted_vector": np.zeros(1), "prior_only": prior})
        for prior in exposures
    )
    summary, aggregate_variance = runner._prior_aggregate(scores, _scripted_fit())
    assert summary["role_delta_count"] == 2
    assert summary["total_variance"] == pytest.approx(0.02)
    assert summary["mean_score_aggregate_variance"] == pytest.approx(
        0.01 * ((1 / 4) ** 2 + (-1 / 4) ** 2)
    )
    assert summary["conditional_mean_logit_variance"] == pytest.approx(
        summary["mean_score_aggregate_variance"]
    )
    commitments = {
        ("jungle", "c1"): runner._sha({
            "domain": "g5-prior-only-role-champion-coordinate:v1",
            "role": "jungle",
            "stable_champion_id": "c1",
        }),
        ("top", "c0"): runner._sha({
            "domain": "g5-prior-only-role-champion-coordinate:v1",
            "role": "top",
            "stable_champion_id": "c0",
        }),
    }
    witness = sorted([
        {
            "coordinate_commitment_sha256": commitments[("jungle", "c1")],
            "blue_count": 0,
            "red_count": 1,
            "net_count": -1,
            "validation_map_count": 4,
        },
        {
            "coordinate_commitment_sha256": commitments[("top", "c0")],
            "blue_count": 2,
            "red_count": 1,
            "net_count": 1,
            "validation_map_count": 4,
        },
    ], key=lambda record: record["coordinate_commitment_sha256"])
    assert summary["coordinate_exposure_witness"] == witness
    assert summary["slot_membership_sha256"] == runner._sha([
        record["coordinate_commitment_sha256"] for record in witness
    ])
    assert summary["signed_exposure_sha256"] == runner._sha([
        [
            record["coordinate_commitment_sha256"],
            record["blue_count"],
            record["red_count"],
            record["net_count"],
            record["validation_map_count"],
        ]
        for record in witness
    ])
    assert aggregate_variance == pytest.approx(summary["mean_score_aggregate_variance"])


def test_fake_mapping_cannot_reach_real_result_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    result_path = tmp_path / runner.RESULT_LOCATOR
    result_path.parent.mkdir(parents=True)
    with pytest.raises(runner.G5RunnerError, match="execute_real-only"):
        runner._immutable_write_many(((result_path, b"{}"),))
    assert not result_path.exists()


def test_coherent_witness_substitution_is_only_shape_valid_and_cannot_write(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _real_payload_from_evidence(_scripted_d1_evidence(monkeypatch))
    substituted = copy.deepcopy(payload)
    substituted.pop("artifact_sha256")
    prior = substituted["prior_only_variance_components"]
    record = prior["coordinate_exposure_witness"][0]
    record["coordinate_commitment_sha256"] = "0" * 64
    prior["slot_membership_sha256"] = result.sha256(["0" * 64])
    prior["signed_exposure_sha256"] = result.sha256([
        ["0" * 64, record["blue_count"], record["red_count"], record["net_count"], 207]
    ])
    substituted["artifact_sha256"] = result.sha256(substituted)
    # Standalone validation proves internal witness consistency, not source truth.
    result.validate_real(substituted)
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    result_path = tmp_path / runner.RESULT_LOCATOR
    result_path.parent.mkdir(parents=True)
    with pytest.raises(runner.G5RunnerError, match="execute_real-only"):
        runner._immutable_write_many(
            ((result_path, result.canonical_bytes(substituted) + b"\n"),)
        )
    assert not result_path.exists()


def test_immutable_writer_rejects_partial_mismatch_symlink_hardlink_and_rolls_back(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    directory = tmp_path / "safe"
    directory.mkdir()
    first, second = directory / "one.json", directory / "two.json"
    runner._immutable_write_many(((first, b"one"), (second, b"two")))
    runner._immutable_write_many(((first, b"one"), (second, b"two")))
    with pytest.raises(runner.G5RunnerError, match="different"):
        runner._immutable_write_many(((first, b"changed"), (second, b"two")))
    first.unlink()
    with pytest.raises(runner.G5RunnerError, match="partial"):
        runner._immutable_write_many(((first, b"one"), (second, b"two")))
    second.unlink()
    target = directory / "target"
    target.write_bytes(b"x")
    os.link(target, first)
    with pytest.raises(runner.G5RunnerError, match="unsafe output leaf"):
        runner._immutable_write_many(((first, b"x"),))
    first.unlink()
    target.unlink()
    first.symlink_to(second)
    with pytest.raises(runner.G5RunnerError, match="unsafe output leaf"):
        runner._immutable_write_many(((first, b"x"),))
    first.unlink()

    original_link = os.link
    calls = 0

    def fail_second(source: str, destination: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected")
        original_link(source, destination)

    monkeypatch.setattr(runner.os, "link", fail_second)
    with pytest.raises(OSError, match="injected"):
        runner._immutable_write_many(((first, b"one"), (second, b"two")))
    assert not first.exists() and not second.exists()


def test_runner_review_bundle_stays_closed_after_bound_dependency_drift() -> None:
    with pytest.raises(runner.G5RunnerError, match="frozen dependency identity mismatch"):
        runner.build_runner_review_bundle()

    namespace = Path("data/lol/v2/models/draft-interactions/g5-exploratory")
    core = json.loads((namespace / "execution-review-core.json").read_bytes())
    pending = json.loads((namespace / "execution-pending-report.json").read_bytes())
    for artifact in (core, pending):
        claimed = artifact["artifact_sha256"]
        unsigned = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
        assert result.sha256(unsigned) == claimed
    assert set(core["review_subject_bytes"]) == {
        "runner", "result", "approval_contract", "focused_test", "approval_test"
    }
    assert set(core["execution_dependency_pins"]) == {"G1", "G1_features", "G2", "clusters", "runtime"}
    assert pending["protected_reads"] == 0
    assert pending["missing"] == [
        "final_independent_runner_review",
        "canonical_execution_approval",
    ]
    serialized_core = result.canonical_bytes(core)
    for forbidden in (b"permit", b"environment", b"dynamic authority"):
        assert forbidden not in serialized_core.lower()
    assert pending["approval"]["status"] == "MISSING_NOT_ISSUED"
