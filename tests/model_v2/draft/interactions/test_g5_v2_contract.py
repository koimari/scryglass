from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import (
    contract, v2_execution_approval, v2_result, v2_runner,
)


def _approval(core_sha: str, contract_sha: str) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_id": v2_execution_approval.APPROVAL_SCHEMA,
        "state": "APPROVED_PRIVATE_DEVELOPMENT_V2",
        "approval_id": "future-v2-approval-id",
        "run_id": "future-v2-run-id",
        "reviewer_root": "KOI_MARI",
        "authority_scope": v2_execution_approval.SCOPE,
        "review_core_sha256": core_sha,
        "contract_sha256": contract_sha,
        "numerical_config_sha256": v2_runner.v2_math.config_hash(),
        "dependency_pins": {
            "G1": contract.G1,
            "G1_features": contract.G1_FEATURES,
            "G2": contract.G2,
            "clusters": contract.CLUSTERS,
            "accepted_v1_orchestration_primitives": v2_execution_approval.V1_PRIMITIVES,
        },
        "allowed_partitions": ["TRAIN", "DEVELOPMENT", "VALIDATION"],
        "selection_semantics": {
            "development": "select_D1_iff_mean_LL_B0_minus_LL_D1_gt_0_else_B0",
            "validation": "D1_winner_iff_mean_gain_gte_0.005_and_cluster_bootstrap_95pct_LCB_gt_0",
            "bootstrap_replicates": 2000,
            "D2": "OMITTED",
            "final_holdout": False,
        },
        "paths": {
            "approval": v2_execution_approval.APPROVAL_LOCATOR,
            "ledger": v2_execution_approval.LEDGER_LOCATOR,
            "result": v2_execution_approval.RESULT_LOCATOR,
        },
        "claim_ceiling": v2_result.CLAIM_CEILING,
    }
    return {**unsigned, "approval_sha256": v2_execution_approval.sha256(unsigned)}


def test_v2_bundle_is_stable_versioned_and_non_authorizing() -> None:
    first = v2_runner.build_review_bundle()
    second = v2_runner.build_review_bundle()
    assert first == second
    assert set(first) == {"v2-contract.json", "v2-review-core.json", "v2-pending-report.json"}
    contract_payload = first["v2-contract.json"]
    pending = first["v2-pending-report.json"]
    assert contract_payload["scientific_semantics"]["fold_counts"] == {
        "TRAIN": 805, "DEVELOPMENT": 214, "VALIDATION": 207
    }
    assert contract_payload["scientific_semantics"]["D2"] == "OMITTED"
    assert contract_payload["numerical_contract"]["cholesky_jitter"] == 0.0
    assert pending["protected_reads"] == 0
    assert pending["real_diagnostic_runs"] == 0
    assert pending["development_metrics_computed"] is False
    assert pending["validation_metrics_computed"] is False
    assert pending["final_holdout_reads"] == 0
    for locator in (
        v2_execution_approval.APPROVAL_LOCATOR,
        v2_execution_approval.LEDGER_LOCATOR,
        v2_execution_approval.RESULT_LOCATOR,
    ):
        assert not (v2_runner.ROOT / locator).exists()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"approval_id": "koi-mari-g5-private-development-v1"}),
        lambda value: value.update({"run_id": "other"}),
        lambda value: value.update({"reviewer_root": "OTHER"}),
        lambda value: value.update({"allowed_partitions": ["TRAIN"]}),
        lambda value: value["selection_semantics"].update({"bootstrap_replicates": 1999}),
        lambda value: value["selection_semantics"].update({"final_holdout": True}),
        lambda value: value["paths"].update({"result": "v1-path"}),
        lambda value: value["claim_ceiling"].update({"publication": True}),
        lambda value: value["dependency_pins"]["G2"].update({"artifact_raw_sha256": "0" * 64}),
        lambda value: value.update({"numerical_config_sha256": "0" * 64}),
    ],
)
def test_v2_approval_mutations_fail(mutate) -> None:
    bundle = v2_runner.build_review_bundle()
    core_sha = bundle["v2-review-core.json"]["artifact_sha256"]
    contract_sha = bundle["v2-contract.json"]["artifact_sha256"]
    value = copy.deepcopy(_approval(core_sha, contract_sha))
    mutate(value)
    unsigned = dict(value)
    unsigned.pop("approval_sha256")
    value["approval_sha256"] = v2_execution_approval.sha256(unsigned)
    with pytest.raises(v2_execution_approval.V2ApprovalError):
        v2_execution_approval.validate_approval(
            value,
            expected_review_core_sha256=core_sha,
            expected_contract_sha256=contract_sha,
            expected_run_id="future-v2-run-id",
        )


def _unavailable_result():
    scaling = v2_runner._unavailable_scaling(
        "V2_PREFIT_NUMERICAL_UNAVAILABLE:FACTORIZATION"
    )
    membership = {"all_maps_sha256": "1" * 64}
    approval = {"approval_sha256": "2" * 64}
    return v2_runner._result_payload(
        evidence=None,
        numerical_blocker="V2_PREFIT_NUMERICAL_UNAVAILABLE:FACTORIZATION",
        membership_hashes=membership,
        scaling=scaling,
        contract_sha256="3" * 64,
        review_core_sha256="4" * 64,
        approval=approval,
        run_id="future-v2-run-id",
    )


def _rehash(value: dict[str, object]) -> None:
    unsigned = dict(value)
    unsigned.pop("artifact_sha256")
    value["artifact_sha256"] = v2_result.sha256(unsigned)


def test_unavailable_result_is_closed_and_expected_context_bound() -> None:
    payload, expected = _unavailable_result()
    v2_result.validate_real(payload, expected=expected)
    assert payload["counts"] == {"TRAIN": 805, "DEVELOPMENT": 214, "VALIDATION": 207}
    assert payload["development_metric"]["evaluations"] == 0
    assert payload["validation_metric"]["evaluations"] == 0
    assert payload["prior_only_variance_components"] is None
    assert payload["winner_evidence"] is None
    with pytest.raises(TypeError):
        v2_result.validate_real(payload)


def test_result_finite_validator_rejects_integer_float_overflow() -> None:
    with pytest.raises(ValueError, match="finite"):
        v2_result._finite(10**1000, "overflowing_integer")


@pytest.mark.parametrize(
    "field",
    [
        "config_sha256", "contract_sha256", "review_core_sha256",
        "approval_sha256", "run_id", "transform_sha256",
    ],
)
def test_coordinated_result_identity_substitution_fails(field: str) -> None:
    payload, expected = _unavailable_result()
    forged = copy.deepcopy(payload)
    context = copy.deepcopy(expected)
    if field in forged["execution_binding"]:
        forged["execution_binding"][field] = "9" * 64 if field != "run_id" else "other"
    if field == "config_sha256":
        forged["train_scaling"]["config_sha256"] = "9" * 64
    if field == "transform_sha256":
        forged["train_scaling"]["transform_sha256"] = "9" * 64
    _rehash(forged)
    with pytest.raises(ValueError):
        v2_result.validate_real(forged, expected=context)


def test_coordinated_config_context_substitution_still_fails() -> None:
    payload, expected = _unavailable_result()
    forged = copy.deepcopy(payload)
    forged["train_scaling"]["config_sha256"] = "9" * 64
    forged["execution_binding"]["config_sha256"] = "9" * 64
    _rehash(forged)
    with pytest.raises(ValueError, match="trusted context config"):
        v2_result.validate_real(
            forged, expected=replace(expected, config_sha256="9" * 64)
        )


def test_prior_only_unique_coordinate_witness_exactness() -> None:
    commitment = "a" * 64
    prior = {
        "status": "EVALUATED",
        "role_delta_count": 1,
        "variance_per_coordinate": 0.01,
        "total_variance": 0.01,
        "mean_score_aggregate_variance": 0.01,
        "conditional_mean_logit_variance": 0.02,
        "slot_membership_sha256": v2_result.sha256([commitment]),
        "signed_exposure_sha256": v2_result.sha256(
            [[commitment, 207, 0, 207, 207]]
        ),
        "coordinate_exposure_witness": [{
            "coordinate_commitment_sha256": commitment,
            "blue_count": 207,
            "red_count": 0,
            "net_count": 207,
            "validation_map_count": 207,
        }],
    }
    v2_result._validate_prior(prior)
    for mutate in (
        lambda value: value.update({"role_delta_count": 2}),
        lambda value: value.update({"variance_per_coordinate": 0.02}),
        lambda value: value.update({"slot_membership_sha256": "0" * 64}),
        lambda value: value["coordinate_exposure_witness"][0].update(
            {"net_count": 0}
        ),
        lambda value: value["coordinate_exposure_witness"].append(
            copy.deepcopy(value["coordinate_exposure_witness"][0])
        ),
    ):
        forged = copy.deepcopy(prior)
        mutate(forged)
        with pytest.raises(ValueError):
            v2_result._validate_prior(forged)


def _evaluated_evidence(*, selected: str, winner: bool = False):
    dev_b0 = 0.7
    dev_d1 = 0.69 if selected == "D1" else 0.71
    val_b0 = 0.7
    val_candidate = 0.694 if selected == "D1" else 0.7
    empty = v2_result.sha256([])
    prior = {
        "status": "NOT_EVALUATED" if selected == "B0" else "EVALUATED",
        "role_delta_count": 0,
        "variance_per_coordinate": 0.01,
        "total_variance": 0.0,
        "mean_score_aggregate_variance": 0.0,
        "conditional_mean_logit_variance": 0.0,
        "slot_membership_sha256": empty,
        "signed_exposure_sha256": empty,
        "coordinate_exposure_witness": [],
    }
    winner_value = None
    if winner:
        winner_value = SimpleNamespace(
            B0_probability=0.5,
            D1_logit_increment=0.1,
            neutral_completed_draft_probability=0.52,
            probability_increment_over_B0=0.02,
            D1_conditional_interval={
                "lower": 0.01, "upper": 0.19, "level": 0.95,
                "scale": "conditional_mean_validation_logit_increment",
            },
        )
    return SimpleNamespace(
        state=(
            "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER"
            if winner else "NO_INCREMENTAL_DRAFT_WINNER"
        ),
        blocker=None,
        selected_candidate=selected,
        development_metric={
            "locked_candidate": selected, "map_count": 214, "evaluations": 1,
            "B0_mean_log_loss": dev_b0, "D1_mean_log_loss": dev_d1,
            "mean_LL_B0_minus_LL_D1": dev_b0 - dev_d1,
        },
        validation_metric={
            "locked_candidate": selected, "map_count": 207, "evaluations": 1,
            "B0_mean_log_loss": val_b0,
            "locked_candidate_mean_log_loss": val_candidate,
            "mean_LL_B0_minus_LL_locked_candidate": val_b0 - val_candidate,
        },
        bootstrap=(
            {
                "status": "COMPLETED", "replicates": 2000,
                "base_seed": 2026073005, "quantile": 0.05,
                "lower_bound": 0.001 if winner else -0.001,
                "map_weighted": True,
            }
            if selected == "D1"
            else {
                "status": "NOT_RUN_B0_LOCKED", "replicates": 0,
                "base_seed": None, "quantile": None, "lower_bound": None,
                "map_weighted": True,
            }
        ),
        solver_diagnostics={
            "status": "CONVERGED",
            "method": "DETERMINISTIC_DAMPED_NEWTON_ARMIJO",
            "iterations": 4,
            "trace_sha256": "a" * 64,
            "config_sha256": v2_runner.v2_math.config_hash(),
            "gradient_inf": 1e-10,
            "jitter_used": 0.0,
        },
        uncertainty={
            "status": "AVAILABLE", "hessian_symmetric": True,
            "hessian_strictly_pd": True, "covariance_finite": True,
            "covariance_symmetric": True,
            "covariance_nonnegative_quadratic_forms": True,
            "factorization_residual_pass": True, "solve_residual_pass": True,
            "inverse_residual_pass": True,
        },
        invariance_tests={
            "side_swap": True, "record_order": True,
            "role_relabel": "NOT_INVARIANT_BY_CONTRACT",
        },
        contribution_reconciliation={
            "status": "PASSED", "absolute_tolerance": 1e-12,
            "max_absolute_error": 0.0,
        },
        prior_only_variance_components=prior,
        winner=winner_value,
    )


@pytest.mark.parametrize(
    "selected,winner",
    [("B0", False), ("D1", False), ("D1", True)],
)
def test_b0_d1_no_winner_and_winner_result_gates(
    selected: str, winner: bool
) -> None:
    evidence = _evaluated_evidence(selected=selected, winner=winner)
    scaling = {
        "partition": "TRAIN", "scales_sha256": "5" * 64,
        "transform_sha256": "6" * 64,
        "config_sha256": v2_runner.v2_math.config_hash(),
        "definition": "X_s=X/s;gamma=s*beta;lambda_s=lambda/s^2",
    }
    payload, expected = v2_runner._result_payload(
        evidence=evidence, numerical_blocker=None,
        membership_hashes={"all_maps_sha256": "7" * 64},
        scaling=scaling, contract_sha256="8" * 64,
        review_core_sha256="9" * 64,
        approval={"approval_sha256": "a" * 64},
        run_id="synthetic-local-run",
    )
    v2_result.validate_real(payload, expected=expected)
    assert payload["bootstrap"]["replicates"] == (
        2000 if selected == "D1" else 0
    )
    assert (payload["winner_evidence"] is not None) is winner


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["winner_evidence"].update({"B0_probability": 1.0}),
        lambda value: value["winner_evidence"].update({"probability_increment_over_B0": 1.0}),
        lambda value: value["winner_evidence"]["D1_conditional_interval"].update({"level": 0.90}),
        lambda value: value["winner_evidence"]["D1_conditional_interval"].update({"lower": 2.0, "upper": 1.0}),
    ],
)
def test_winner_probability_and_interval_ranges_fail_closed(mutate) -> None:
    evidence = _evaluated_evidence(selected="D1", winner=True)
    scaling = {
        "partition": "TRAIN", "scales_sha256": "5" * 64,
        "transform_sha256": "6" * 64,
        "config_sha256": v2_runner.v2_math.config_hash(),
        "definition": "X_s=X/s;gamma=s*beta;lambda_s=lambda/s^2",
    }
    payload, _expected = v2_runner._result_payload(
        evidence=evidence, numerical_blocker=None,
        membership_hashes={"all_maps_sha256": "7" * 64},
        scaling=scaling, contract_sha256="8" * 64,
        review_core_sha256="9" * 64,
        approval={"approval_sha256": "a" * 64},
        run_id="synthetic-local-run",
    )
    forged = copy.deepcopy(payload)
    mutate(forged)
    forged["artifact_sha256"] = v2_result.sha256({
        key: value for key, value in forged.items() if key != "artifact_sha256"
    })
    with pytest.raises(ValueError):
        v2_result.validate_real(forged, expected=_expected)


def test_v2_ledger_earliest_completion_and_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(v2_execution_approval, "ROOT", tmp_path)
    directory = tmp_path / v2_execution_approval.NAMESPACE
    directory.mkdir(parents=True)
    approval = {
        "approval_id": "approval",
        "approval_sha256": "a" * 64,
    }
    common = {
        "approval_id": "approval", "run_id": "run",
        "review_core_sha256": "b" * 64,
        "approval_sha256": "a" * 64,
        "result_locator": v2_execution_approval.RESULT_LOCATOR,
    }
    started = v2_execution_approval.append_ledger_entry({
        **common, "state": "STARTED", "sequence": 1,
        "started_at": "2026-07-30T12:00:00Z",
    })
    history = v2_execution_approval.load_ledger()
    assert v2_execution_approval.validate_ledger_history(
        history, approval=approval,
        expected_review_core_sha256="b" * 64, expected_run_id="run",
    ) == "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY"
    completed = v2_execution_approval.append_ledger_entry({
        **common, "state": "COMPLETED", "sequence": 2,
        "started_entry_sha256": started["entry_sha256"],
        "result_artifact_sha256": "c" * 64,
        "result_raw_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "transform_sha256": "f" * 64,
        "scales_sha256": "1" * 64,
        "membership_hashes_sha256": "2" * 64,
        "source_pins_sha256": "3" * 64,
        "completed_at": "2026-07-30T12:00:01Z",
    })
    v2_execution_approval.append_ledger_entry({
        **common, "state": "INVALID_DUPLICATE", "sequence": 3,
        "authoritative_completed_entry_sha256": completed["entry_sha256"],
        "authoritative_result_artifact_sha256": "c" * 64,
        "recorded_at": "2026-07-30T12:00:02Z",
    })
    assert v2_execution_approval.validate_ledger_history(
        v2_execution_approval.load_ledger(), approval=approval,
        expected_review_core_sha256="b" * 64, expected_run_id="run",
    ) == "COMPLETED_TERMINAL"


def test_execute_v2_starts_before_reads_and_numerical_failure_never_selects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = v2_runner.build_review_bundle()
    approval = _approval(
        bundle["v2-review-core.json"]["artifact_sha256"],
        bundle["v2-contract.json"]["artifact_sha256"],
    )
    entries: list[dict[str, object]] = []
    written: dict[str, object] = {}
    monkeypatch.setattr(v2_runner, "build_review_bundle", lambda: bundle)
    monkeypatch.setattr(
        v2_execution_approval, "load_approval", lambda **_kwargs: approval
    )
    monkeypatch.setattr(v2_execution_approval, "load_ledger", lambda: entries)
    monkeypatch.setattr(
        v2_execution_approval,
        "validate_ledger_history",
        lambda current, **_kwargs: (
            "EMPTY" if not current else
            "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY" if len(current) == 1 else
            "COMPLETED_TERMINAL"
        ),
    )

    def append(unsigned):
        value = dict(unsigned)
        value["entry_sha256"] = f"{len(entries) + 1:064x}"
        entries.append(value)
        return value

    monkeypatch.setattr(v2_execution_approval, "append_ledger_entry", append)

    def protected_read():
        assert entries and entries[0]["state"] == "STARTED"
        return object()

    monkeypatch.setattr(v2_runner, "_load_aligned_v2", protected_read)
    maps = tuple(SimpleNamespace() for _ in range(1226))
    monkeypatch.setattr(
        v2_runner.v1_runner, "build_b0_scores", lambda _aligned: (maps, ())
    )
    monkeypatch.setattr(
        v2_runner.v1_runner,
        "_membership_hashes",
        lambda _maps, _aligned: {"all_maps_sha256": "7" * 64},
    )
    monkeypatch.setattr(
        v2_runner,
        "_compute_evidence_v2",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            v2_runner.v2_math.V2NumericalUnavailable(
                "V2_PREFIT_NUMERICAL_UNAVAILABLE:HESSIAN_FACTORIZATION"
            )
        ),
    )
    monkeypatch.setattr(
        v2_runner,
        "_write_immutable",
        lambda _path, payload: written.update(payload),
    )
    monkeypatch.setattr(
        v2_execution_approval,
        "validate_completed_result",
        lambda _completion, *, expected: dict(written),
    )
    payload = v2_runner.execute_real_v2("future-v2-run-id")
    assert [entry["state"] for entry in entries] == ["STARTED", "COMPLETED"]
    assert payload["state"] == "V2_PREFIT_NUMERICAL_UNAVAILABLE"
    assert payload["selected_candidate"] is None
    assert payload["development_metric"]["evaluations"] == 0
    assert payload["validation_metric"]["evaluations"] == 0
    assert payload["bootstrap"] is None
    assert payload["winner_evidence"] is None


def test_v1_paths_are_not_v2_paths() -> None:
    assert "v2-" in Path(v2_execution_approval.APPROVAL_LOCATOR).name
    assert "v2-" in Path(v2_execution_approval.LEDGER_LOCATOR).name
    assert "v2-" in Path(v2_execution_approval.RESULT_LOCATOR).name
