from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lol_kills.v2.draft.interactions import representation_rank_assay as assay
from lol_kills.v2.draft.interactions import representation_rank_private_runner as runner


ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = (
    ROOT
    / "data/lol/v2/models/draft-interactions/"
    "representation-rank-private-run-contract.json"
)


def _rehash(payload: dict) -> dict:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return {**unsigned, "artifact_sha256": assay.canonical_sha256(unsigned)}


def _pass_contract(contract: dict, tmp_path: Path, monkeypatch) -> dict:
    permit = {
        "approved_action": "private_target_m0_load_and_rank_assay",
        "decision": "PASS",
        "final_temporal_holdout_sealed": True,
        "independent_from_runner_and_generator": True,
        "review_core_sha256": contract["runner_review_core_sha256"],
        "schema_id": "scryglass.representation-rank-runner-review-permit.v1",
    }
    path = tmp_path / "synthetic-runner-review-permit.json"
    path.write_bytes(assay.canonical_bytes(permit))
    raw = assay.raw_sha256(path)
    monkeypatch.setattr(
        runner.runner_review_authority,
        "PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256",
        raw,
    )
    return _rehash(
        {
            **contract,
            "runner_review_status": "PASS",
            "status": "private_runner_review_pass",
            "runner_review_permit": {
                "locator": str(path),
                "raw_sha256": raw,
            },
        }
    )


def _gate_result(
    months: tuple[str, ...],
    *,
    passed: bool = True,
) -> dict:
    comparators = {}
    for comparator in ("M0", "M8"):
        metrics = {}
        for metric in ("log_loss", "brier", "calibration"):
            upper = (
                0.0
                if not passed and comparator == "M0" and metric == "log_loss"
                else -0.01
                if metric == "log_loss"
                else 0.0
            )
            decision = assay.metric_gate_decision(
                comparator=comparator,
                metric=metric,
                upper=upper,
            )
            metrics[metric] = {
                "upper": upper,
                **decision,
            }
        blocks = {
            month: {
                "delta": 0.0,
                "passed": assay.block_gate_decision(0.0),
            }
            for month in months
        }
        comparators[comparator] = {
            **metrics,
            "blocks": blocks,
        }
    derived_passed = all(
        gate["passed"]
        for comparator in comparators.values()
        for gate in (
            comparator["log_loss"],
            comparator["brier"],
            comparator["calibration"],
            *comparator["blocks"].values(),
        )
    )
    assert derived_passed is passed
    return {
        "passed": derived_passed,
        "comparators": comparators,
    }


def _coverage_report(*, month: str, count: int) -> dict:
    counts = {
        "maps": count,
        "eligible_maps": count,
        "map_fraction": 1.0,
        "clusters": count,
        "eligible_clusters": count,
        "cluster_fraction": 1.0,
    }
    return {
        "passed": True,
        "overall": dict(counts),
        "by_month": {month: dict(counts)},
        "by_league": {"LEC": dict(counts)},
    }


def _synthetic_world(
    monkeypatch,
    *,
    locked_width: int,
    train_count: int = 300,
):
    train_ids = tuple(
        f"train-{index:04d}" for index in range(train_count)
    )
    development_counts = (128, 230, 421, 202)
    validation_counts = (515, 569)
    development_months = tuple(
        row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["development"]
    )
    validation_months = tuple(
        row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]
    )
    by_key: dict[tuple[str, str], tuple[str, ...]] = {}
    for month in assay.INNER_MONTHS:
        by_key[("train", month)] = train_ids
    for month, count in zip(development_months, development_counts):
        by_key[("development", month)] = tuple(
            f"development-{month}-{index:04d}" for index in range(count)
        )
    for month, count in zip(validation_months, validation_counts):
        by_key[("validation", month)] = tuple(
            f"validation-{month}-{index:04d}" for index in range(count)
        )
    all_ids = tuple(
        dict.fromkeys(
            [
                *train_ids,
                *(
                    game_id
                    for key, ids in by_key.items()
                    if key[0] != "train"
                    for game_id in ids
                ),
            ]
        )
    )
    split_by_id = {
        game_id: (
            "train"
            if game_id.startswith("train")
            else "development"
            if game_id.startswith("development")
            else "validation"
        )
        for game_id in all_ids
    }
    records = tuple(
        (
            game_id,
            split_by_id[game_id],
            f"cluster-{game_id}",
            "2025-01",
            "LEC",
            tuple(range(10)),
        )
        for game_id in all_ids
    )
    domain = SimpleNamespace(
        records=records,
        artifact_sha256="a" * 64,
        node_domain=SimpleNamespace(node_roles=tuple("x" for _ in range(10))),
    )
    feature = runner.FeatureEnvelope(
        domain=domain,
        ordered_rows=(),
        selected_columns=runner.SAFE_FEATURE_COLUMNS,
        source_raw_sha256="b" * 64,
        logical_rows_sha256="c" * 64,
        artifact_sha256="d" * 64,
    )
    availability = SimpleNamespace(
        ordered_game_ids=all_ids,
        artifact_sha256="e" * 64,
    )
    target_domain = SimpleNamespace(
        ordered_targets=tuple(
            (game_id, index % 2) for index, game_id in enumerate(all_ids)
        ),
        artifact_sha256="f" * 64,
    )
    target = runner.TargetM0Envelope(
        target_domain=target_domain,
        m0_by_game_id=tuple((game_id, 0.5) for game_id in all_ids),
        ordered_rows=(),
        source_raw_sha256="0" * 64,
        logical_rows_sha256="1" * 64,
        ordered_logical_rows_sha256="2" * 64,
        artifact_sha256="3" * 64,
    )
    target_by_id = dict(target_domain.ordered_targets)
    contexts: dict[tuple[str, str], runner.MonthRunContext] = {}
    for key, ids in by_key.items():
        split, month = key
        prepared = (
            None
            if split == "train"
            else SimpleNamespace(
                split=split,
                ordered_eligible_game_ids=ids,
                m0_probability=np.full(len(ids), 0.5),
            )
        )
        contexts[key] = runner.MonthRunContext(
            split=split,
            calendar_month=month,
            eligibility_binding=SimpleNamespace(
                artifact_sha256=assay.canonical_sha256([split, month]),
                ordered_source_game_ids=ids,
            ),
            prediction_game_ids=ids,
            prediction_cluster_ids=tuple(
                f"cluster-{game_id}" for game_id in ids
            ),
            m0_probability=np.full(len(ids), 0.5),
            membership_sha256=assay.canonical_sha256([split, month, len(ids)]),
            prepared_fold=prepared,
            coverage_report=_coverage_report(month=month, count=len(ids)),
        )

    def context_builder(**kwargs):
        return contexts[(kwargs["split"], kwargs["calendar_month"])]

    def combine(folds, *, split):
        ids = tuple(
            game_id for fold in folds for game_id in fold.ordered_eligible_game_ids
        )
        return SimpleNamespace(
            split=split,
            ordered_eligible_game_ids=ids,
            m0_probability=np.full(len(ids), 0.5),
        )

    def select_development_width(**kwargs):
        assert set(kwargs["predictions"]) == {1, 2, 4, 8, "M0", "M8"}
        assert len(kwargs["game_ids"]) == 981
        return locked_width, {
            "M8_prerequisite": _gate_result(development_months),
            "widths": {
                str(width): _gate_result(
                    development_months,
                    passed=width == locked_width,
                )
                for width in assay.WIDTHS[
                    : assay.WIDTHS.index(locked_width) + 1
                ]
            },
            "locked_width": locked_width,
        }

    def validate_locked_width(**kwargs):
        assert set(kwargs["predictions"]) == {locked_width, "M0", "M8"}
        assert len(kwargs["game_ids"]) == 1084
        return {
            "passed": True,
            "locked_width": locked_width,
            "M8": _gate_result(validation_months),
            "locked": _gate_result(validation_months),
        }

    monkeypatch.setattr(runner, "likelihood_feature_domain", lambda *args: domain)
    monkeypatch.setattr(runner, "_build_month_context", context_builder)
    monkeypatch.setattr(assay, "combine_prepared_folds", combine)
    monkeypatch.setattr(
        assay, "select_development_width", select_development_width
    )
    monkeypatch.setattr(assay, "validate_locked_width", validate_locked_width)

    def execution(request: runner.FitRequest) -> runner.FitExecution:
        probability = tuple(
            (
                game_id,
                0.7 if target_by_id[game_id] else 0.3,
            )
            for game_id in request.context.prediction_game_ids
        )
        return runner.FitExecution(
            prediction_by_game_id=probability,
            objective=0.5,
            max_gradient=1e-7,
            converged_starts=3,
            stability_rms=1e-4,
        )

    return feature, availability, target, execution


@pytest.mark.parametrize("locked_width,expected_fits", ((8, 54), (2, 56)))
def test_synthetic_pass_coordinator_order_alias_and_determinism(
    tmp_path: Path,
    monkeypatch,
    locked_width: int,
    expected_fits: int,
) -> None:
    contract = runner.load_contract(CONTRACT_PATH, root=ROOT)
    contract = _pass_contract(contract, tmp_path, monkeypatch)
    feature, availability, target, execution = _synthetic_world(
        monkeypatch, locked_width=locked_width
    )
    load_order: list[str] = []
    fit_requests: list[runner.FitRequest] = []
    writes: list[bytes] = []

    def load_feature(*args, **kwargs):
        load_order.append("feature")
        return feature

    def load_availability(*args, **kwargs):
        load_order.append("availability")
        return availability

    def load_target(*args, **kwargs):
        load_order.append("target")
        return target

    def fit(request):
        fit_requests.append(request)
        return execution(request)

    def write(payload, *, path):
        writes.append(assay.canonical_bytes(payload))

    first = runner.run_private(
        contract,
        root=ROOT,
        feature_loader=load_feature,
        availability_loader=load_availability,
        target_loader=load_target,
        fit_executor=fit,
        result_writer=write,
    )
    assert load_order == ["feature", "availability", "target"]
    assert [request.sequence for request in fit_requests] == list(
        range(1, 53)
    ) + list(
        range(53, 57)
        if locked_width != 8
        else (53, 55)
    )
    assert len(fit_requests) == expected_fits
    assert len([request for request in fit_requests if request.width == 8]) == 42
    assert first["fit_counts"]["actual"] == expected_fits
    assert first["run_status"] == "accepted"
    assert len(first["fit_plan"]) == 56
    assert first["penalties"] == {"lambda_ally": 1.0, "lambda_enemy": 1.0}
    assert all(
        set(row)
        == {
            "sequence",
            "stage",
            "split",
            "calendar_month",
            "family",
            "fit_role",
            "lambda_ally",
            "lambda_enemy",
            "width",
            "execution_status",
            "maps",
            "clusters",
            "membership_sha256",
            "objective",
            "max_gradient",
            "converged_starts",
            "stability_rms",
            "log_loss_total",
            "brier_total",
        }
        for row in first["fit_plan"][:36]
    )
    assert all(
        (
            request.lambda_enemy == 1.0
            if request.family == "ally"
            else request.lambda_ally == 1.0
        )
        for request in fit_requests[:36]
    )
    if locked_width == 8:
        assert [
            first["fit_plan"][index]["execution_status"]
            for index in (53, 55)
        ] == ["aliased", "aliased"]
    load_order.clear()
    fit_requests.clear()
    second = runner.run_private(
        contract,
        root=ROOT,
        feature_loader=load_feature,
        availability_loader=load_availability,
        target_loader=load_target,
        fit_executor=fit,
        result_writer=write,
    )
    assert first == second
    assert writes[0] == writes[1]


@pytest.mark.parametrize(
    "failed_sequence,locked_width",
    ((1, 2), (37, 2), (53, 2), (55, 8)),
)
def test_statistical_fit_failure_writes_complete_m0_terminal(
    tmp_path: Path,
    monkeypatch,
    failed_sequence: int,
    locked_width: int,
) -> None:
    contract = runner.load_contract(CONTRACT_PATH, root=ROOT)
    contract = _pass_contract(contract, tmp_path, monkeypatch)
    feature, availability, target, execution = _synthetic_world(
        monkeypatch, locked_width=locked_width
    )
    writes = []

    def fit(request):
        if request.sequence == failed_sequence:
            raise runner.StatisticalRunInconclusive("fit_unavailable")
        return execution(request)

    payload = runner.run_private(
        contract,
        root=ROOT,
        feature_loader=lambda *args, **kwargs: feature,
        availability_loader=lambda *args, **kwargs: availability,
        target_loader=lambda *args, **kwargs: target,
        fit_executor=fit,
        result_writer=lambda value, **kwargs: writes.append(value),
    )
    assert len(writes) == 1
    assert payload["run_status"] == "inconclusive"
    assert payload["fallback"] == "M0"
    assert payload["selected_width"] is None
    failed_row = payload["fit_plan"][failed_sequence - 1]
    assert failed_row["execution_status"] == "failed"
    assert all(
        failed_row[key] is not None
        for key in ("maps", "clusters", "membership_sha256")
    )
    assert all(
        failed_row[key] is None
        for key in (
            "objective",
            "max_gradient",
            "converged_starts",
            "stability_rms",
            "log_loss_total",
            "brier_total",
        )
    )
    assert payload["reason_code"] == "fit_unavailable"
    assert payload["reason_context"] == {
        "stage": failed_row["stage"],
        "sequence": failed_sequence,
        "calendar_month": failed_row["calendar_month"],
        "family": failed_row["family"],
        "width": failed_row["width"],
    }
    assert payload["stage_status"][failed_row["stage"]] == {
        "status": "failed",
        "reason_code": "fit_unavailable",
    }
    assert payload["fit_counts"]["actual"] == sum(
        row["execution_status"] == "passed"
        for row in payload["fit_plan"]
    )
    if failed_sequence == 55:
        assert payload["fit_plan"][53]["execution_status"] == "aliased"
        assert payload["fit_plan"][54]["width"] == 8
        assert payload["fit_plan"][54]["lambda_ally"] == 1.0
        assert payload["fit_plan"][54]["lambda_enemy"] == 1.0
    assert all(
        row["execution_status"] == "not_run"
        for row in payload["fit_plan"][failed_sequence:]
    )
    raw = assay.canonical_bytes(payload)
    assert b"game_id" not in raw
    assert b"prediction" not in raw


def test_coverage_failure_writes_complete_zero_fit_terminal(
    tmp_path: Path, monkeypatch
) -> None:
    contract = runner.load_contract(CONTRACT_PATH, root=ROOT)
    contract = _pass_contract(contract, tmp_path, monkeypatch)
    feature, availability, target, _ = _synthetic_world(
        monkeypatch, locked_width=2
    )
    base_builder = runner._build_month_context

    def failed_coverage(**kwargs):
        context = base_builder(**kwargs)
        if (
            context.split == "development"
            and context.calendar_month == "2025-10"
        ):
            eligible = context.coverage_report["overall"]["eligible_maps"]
            total = 2 * eligible
            failed_counts = {
                "maps": total,
                "eligible_maps": eligible,
                "map_fraction": eligible / total,
                "clusters": total,
                "eligible_clusters": eligible,
                "cluster_fraction": eligible / total,
            }
            failed_report = {
                "passed": assay.coverage_gate_decision(
                    overall=failed_counts,
                    month_rows=[failed_counts],
                    league_rows=[failed_counts],
                ),
                "overall": dict(failed_counts),
                "by_month": {
                    context.calendar_month: dict(failed_counts),
                },
                "by_league": {"LEC": dict(failed_counts)},
            }
            assert failed_report["passed"] is False
            return runner.MonthRunContext(
                **{
                    **context.__dict__,
                    "prepared_fold": None,
                    "coverage_report": failed_report,
                }
            )
        return context

    monkeypatch.setattr(runner, "_build_month_context", failed_coverage)
    writes = []
    payload = runner.run_private(
        contract,
        root=ROOT,
        feature_loader=lambda *args, **kwargs: feature,
        availability_loader=lambda *args, **kwargs: availability,
        target_loader=lambda *args, **kwargs: target,
        fit_executor=lambda request: pytest.fail("fit must not run"),
        result_writer=lambda value, **kwargs: writes.append(value),
    )
    assert len(writes) == 1
    assert payload["run_status"] == "inconclusive"
    assert payload["fit_counts"]["actual"] == 0
    assert payload["stage_status"] == {
        "inner": {"status": "not_run", "reason_code": None},
        "development": {
            "status": "failed",
            "reason_code": "coverage_gate_failed",
        },
        "validation": {"status": "not_run", "reason_code": None},
    }
    assert payload["reason_code"] == "coverage_gate_failed"
    assert payload["reason_context"] == {
        "stage": "development",
        "sequence": None,
        "calendar_month": "2025-10",
        "family": None,
        "width": None,
    }
    assert all(
        row["execution_status"] == "not_run" for row in payload["fit_plan"]
    )


def test_coverage_pass_flag_cannot_disagree_with_count_decision(
    monkeypatch,
) -> None:
    _synthetic_world(monkeypatch, locked_width=2)
    month = assay.ELIGIBLE_GATE_BLOCKS["development"][0][0]
    context = runner._build_month_context(
        split="development",
        calendar_month=month,
    )
    mismatched = runner.MonthRunContext(
        **{
            **context.__dict__,
            "coverage_report": {
                **context.coverage_report,
                "passed": False,
            },
        }
    )
    with pytest.raises(
        runner.PrivateRunnerError,
        match="pass differs from frozen gates",
    ):
        runner._aggregate_coverage([mismatched])


@pytest.mark.parametrize(
    "gate,expected_fits,failed_stage",
    (
        ("penalty", 36, "inner"),
        ("development", 52, "development"),
        ("validation", 56, "validation"),
    ),
)
def test_statistical_gate_failure_is_terminal_m0(
    tmp_path: Path,
    monkeypatch,
    gate: str,
    expected_fits: int,
    failed_stage: str,
) -> None:
    contract = runner.load_contract(CONTRACT_PATH, root=ROOT)
    contract = _pass_contract(contract, tmp_path, monkeypatch)
    feature, availability, target, execution = _synthetic_world(
        monkeypatch,
        locked_width=2,
        train_count=200 if gate == "penalty" else 300,
    )
    if gate == "development":
        development_months = tuple(
            row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["development"]
        )
        monkeypatch.setattr(
            assay,
            "select_development_width",
            lambda **kwargs: (_ for _ in ()).throw(
                assay.RepresentationRankAssayError(
                    "synthetic development gate failure",
                    diagnostics={
                        "M8_prerequisite": _gate_result(
                            development_months
                        ),
                        "widths": {
                            str(width): _gate_result(
                                development_months,
                                passed=False,
                            )
                            for width in assay.WIDTHS
                        },
                        "locked_width": None,
                    },
                )
            ),
        )
    elif gate == "validation":
        validation_months = tuple(
            row[0] for row in assay.ELIGIBLE_GATE_BLOCKS["validation"]
        )
        monkeypatch.setattr(
            assay,
            "validate_locked_width",
            lambda **kwargs: (_ for _ in ()).throw(
                assay.RepresentationRankAssayError(
                    "synthetic validation gate failure",
                    diagnostics={
                        "passed": False,
                        "locked_width": 2,
                        "M8": _gate_result(validation_months),
                        "locked": _gate_result(
                            validation_months,
                            passed=False,
                        ),
                    },
                )
            ),
        )
    writes = []
    payload = runner.run_private(
        contract,
        root=ROOT,
        feature_loader=lambda *args, **kwargs: feature,
        availability_loader=lambda *args, **kwargs: availability,
        target_loader=lambda *args, **kwargs: target,
        fit_executor=execution,
        result_writer=lambda value, **kwargs: writes.append(value),
    )
    assert len(writes) == 1
    assert payload["run_status"] == "inconclusive"
    assert payload["fallback"] == "M0"
    assert payload["fit_counts"]["actual"] == expected_fits
    reason_code = {
        "penalty": "penalty_selection_failed",
        "development": "development_gate_failed",
        "validation": "validation_gate_failed",
    }[gate]
    failed_index = ("inner", "development", "validation").index(failed_stage)
    assert payload["stage_status"] == {
        stage: {
            "status": (
                "passed"
                if index < failed_index
                else "failed"
                if index == failed_index
                else "not_run"
            ),
            "reason_code": reason_code if index == failed_index else None,
        }
        for index, stage in enumerate(
            ("inner", "development", "validation")
        )
    }
    assert payload["reason_code"] == reason_code
    assert payload["reason_context"] == {
        "stage": failed_stage,
        "sequence": None,
        "calendar_month": None,
        "family": None,
        "width": None,
    }
    assert all(
        row["execution_status"] != "failed" for row in payload["fit_plan"]
    )
    assert [
        row["execution_status"] for row in payload["fit_plan"]
    ] == [
        "passed" if index < expected_fits else "not_run"
        for index in range(56)
    ]


def test_programmer_or_final_membership_failure_writes_nothing(
    tmp_path: Path, monkeypatch
) -> None:
    contract = runner.load_contract(CONTRACT_PATH, root=ROOT)
    contract = _pass_contract(contract, tmp_path, monkeypatch)
    feature, availability, target, _ = _synthetic_world(
        monkeypatch, locked_width=2
    )
    writes = []
    with pytest.raises(runner.PrivateRunnerError, match="synthetic programmer"):
        runner.run_private(
            contract,
            root=ROOT,
            feature_loader=lambda *args, **kwargs: feature,
            availability_loader=lambda *args, **kwargs: availability,
            target_loader=lambda *args, **kwargs: target,
            fit_executor=lambda request: (_ for _ in ()).throw(
                runner.PrivateRunnerError("synthetic programmer failure")
            ),
            result_writer=lambda value, **kwargs: writes.append(value),
        )
    assert writes == []

    records = list(feature.domain.records)
    records[0] = (
        records[0][0],
        assay.FINAL_SPLIT,
        *records[0][2:],
    )
    final_feature = runner.FeatureEnvelope(
        **{**feature.__dict__, "domain": SimpleNamespace(
            **{**feature.domain.__dict__, "records": tuple(records)}
        )}
    )
    with pytest.raises(runner.PrivateRunnerError, match="sealed"):
        runner.run_private(
            contract,
            root=ROOT,
            feature_loader=lambda *args, **kwargs: final_feature,
            availability_loader=lambda *args, **kwargs: availability,
            target_loader=lambda *args, **kwargs: target,
            fit_executor=lambda request: pytest.fail("fit must not run"),
            result_writer=lambda value, **kwargs: writes.append(value),
        )
    assert writes == []
