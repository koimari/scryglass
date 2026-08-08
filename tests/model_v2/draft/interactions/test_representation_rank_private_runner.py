from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_kills.v2.draft.interactions import representation_rank_assay as assay
from lol_kills.v2.draft.interactions import representation_rank_private_runner as runner


ROOT = Path(__file__).resolve().parents[4]
CONTRACT_PATH = (
    ROOT
    / "data/lol/v2/models/draft-interactions/"
    "representation-rank-private-run-contract.json"
)
PENDING_PATH = (
    ROOT
    / "data/lol/v2/models/draft-interactions/"
    "representation-rank-private-run-pending-report.json"
)


@pytest.fixture(scope="module")
def contract() -> dict:
    return runner.load_contract(CONTRACT_PATH, root=ROOT)


@pytest.fixture(scope="module")
def feature(contract: dict) -> runner.FeatureEnvelope:
    return runner.load_authoritative_features(contract, root=ROOT)


def _rehash(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("artifact_sha256", None)
    return {**payload, "artifact_sha256": assay.canonical_sha256(payload)}


def _review_core_rehash(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("runner_review_subject_sha256", None)
    payload.pop("runner_review_core_sha256", None)
    payload["runner_review_core_sha256"] = runner.contract_review_core_sha256(
        payload
    )
    return _rehash(payload)


def _synthetic_target(
    feature: runner.FeatureEnvelope, ids: list[str]
) -> runner.TargetM0Envelope:
    domain = assay._build_target_domain(
        {game_id: index % 2 for index, game_id in enumerate(ids)},
        source_raw_sha256="a" * 64,
    )
    p0 = tuple((game_id, 0.4 + 0.01 * (index % 3)) for index, game_id in enumerate(ids))
    return runner.TargetM0Envelope(
        target_domain=domain,
        m0_by_game_id=p0,
        ordered_rows=(),
        source_raw_sha256="a" * 64,
        logical_rows_sha256="b" * 64,
        ordered_logical_rows_sha256="c" * 64,
        artifact_sha256="d" * 64,
    )


def test_contract_and_pending_report_are_canonical_and_null(contract: dict) -> None:
    assert contract["runner_review_status"] == "PASS"
    raw = PENDING_PATH.read_bytes()
    pending = json.loads(raw)
    assert raw == assay.canonical_bytes(pending)
    unsigned = dict(pending)
    claimed = unsigned.pop("artifact_sha256")
    assert claimed == assay.canonical_sha256(unsigned)
    assert pending["development_result"] is None
    assert pending["validation_result"] is None
    assert pending["private_aggregate_artifact"] is None
    assert pending["outcome_columns_loaded"] is False


def test_feature_loader_requests_only_exact_safe_projection(contract: dict) -> None:
    calls = []

    def spy(path, **kwargs):
        calls.append(kwargs)
        return pd.read_parquet(path, **kwargs)

    loaded = runner.load_authoritative_features(
        contract, root=ROOT, read_parquet=spy
    )
    assert len(loaded.domain.records) == 5949
    assert calls == [
        {
            "columns": list(runner.SAFE_FEATURE_COLUMNS),
            "filters": [
                ("split", "in", ["train", "development", "validation"])
            ],
        }
    ]
    assert not (
        runner.FORBIDDEN_READINESS_COLUMNS & set(calls[0]["columns"])
    )
    assert loaded.domain.authoritative_source_verified is False


def test_feature_projection_rejects_final_and_identity_mismatch(
    contract: dict,
) -> None:
    source = contract["source_identity"]["private_feature_materialization"]
    frame = pd.read_parquet(
        ROOT / source["locator"], columns=list(runner.SAFE_FEATURE_COLUMNS)
    )
    final = frame.loc[frame["split"] == assay.FINAL_SPLIT].head(1)

    with pytest.raises(runner.PrivateRunnerError, match="sealed or unknown"):
        runner.load_authoritative_features(
            contract,
            root=ROOT,
            read_parquet=lambda *args, **kwargs: final,
        )

    changed = frame.loc[frame["split"] == "train"].head(1).copy()
    changed["dependence_cluster_id"] = "wrong"
    with pytest.raises(runner.PrivateRunnerError, match="identity mismatch"):
        runner.load_authoritative_features(
            contract,
            root=ROOT,
            read_parquet=lambda *args, **kwargs: changed,
        )


def test_target_loader_is_unreachable_while_pending(
    contract: dict, feature: runner.FeatureEnvelope
) -> None:
    """A PENDING contract must refuse the target loader before any read."""
    pending = _rehash(
        {
            **contract,
            "runner_review_status": "PENDING",
            "status": "private_runner_pending_review",
            "runner_review_permit": None,
        }
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("reader must remain unreachable")

    with pytest.raises(runner.PrivateRunnerError, match="review PASS"):
        runner.load_authoritative_target_m0(
            pending, feature, root=ROOT, read_parquet=forbidden
        )
    assert called is False


def test_caller_rehashed_pass_cannot_self_promote_target_loader(
    contract: dict, feature: runner.FeatureEnvelope
) -> None:
    pending = _rehash(
        {
            **contract,
            "runner_review_status": "PENDING",
            "status": "private_runner_pending_review",
            "runner_review_permit": None,
        }
    )
    promoted = _rehash(
        {
            **pending,
            "runner_review_status": "PASS",
            "status": "private_runner_review_pass",
        }
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("reader must remain unreachable")

    with pytest.raises(
        runner.PrivateRunnerError, match="runner-review permit identity invalid"
    ):
        runner.load_authoritative_target_m0(
            promoted, feature, root=ROOT, read_parquet=forbidden
        )
    assert called is False


def test_target_loader_reloads_koi_mari_authority_before_parquet(
    contract: dict,
    feature: runner.FeatureEnvelope,
    tmp_path: Path,
    monkeypatch,
) -> None:
    permit_payload = {
        "approved_action": "private_target_m0_load_and_rank_assay",
        "decision": "PASS",
        "final_temporal_holdout_sealed": True,
        "independent_from_runner_and_generator": True,
        "review_core_sha256": contract["runner_review_core_sha256"],
        "schema_id": "scryglass.representation-rank-runner-review-permit.v1",
    }
    permit_path = tmp_path / "runner-review-permit.json"
    permit_path.write_bytes(assay.canonical_bytes(permit_payload))
    permit_raw = assay.raw_sha256(permit_path)
    monkeypatch.setattr(
        runner.runner_review_authority,
        "PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256",
        permit_raw,
    )
    authority_path = (
        ROOT / contract["source_identity"]["human_authority"]["locator"]
    )
    invalid_authority = json.loads(authority_path.read_bytes())
    invalid_authority["reviewer_identity"] = "CALLER_REHASH"
    invalid_path = tmp_path / "invalid-target-authority.json"
    invalid_path.write_bytes(assay.canonical_bytes(invalid_authority))
    sources = {
        **contract["source_identity"],
        "human_authority": {
            "locator": str(invalid_path),
            "raw_sha256": assay.raw_sha256(invalid_path),
        },
    }
    changed_contract = {
            **contract,
            "source_identity": sources,
        }
    changed_contract = _review_core_rehash(changed_contract)
    permit_payload["review_core_sha256"] = changed_contract[
        "runner_review_core_sha256"
    ]
    permit_path.write_bytes(assay.canonical_bytes(permit_payload))
    permit_raw = assay.raw_sha256(permit_path)
    monkeypatch.setattr(
        runner.runner_review_authority,
        "PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256",
        permit_raw,
    )
    promoted = _rehash(
        {
            **changed_contract,
            "runner_review_status": "PASS",
            "status": "private_runner_review_pass",
            "runner_review_permit": {
                "locator": str(permit_path),
                "raw_sha256": permit_raw,
            },
        }
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("parquet reader must remain unreachable")

    with pytest.raises(Exception, match="reviewer|authority|KOI_MARI"):
        runner.load_authoritative_target_m0(
            promoted, feature, root=ROOT, read_parquet=forbidden
        )
    assert called is False


def test_permit_binds_complete_contract_review_core(
    contract: dict,
    feature: runner.FeatureEnvelope,
    tmp_path: Path,
    monkeypatch,
) -> None:
    permit_payload = {
        "approved_action": "private_target_m0_load_and_rank_assay",
        "decision": "PASS",
        "final_temporal_holdout_sealed": True,
        "independent_from_runner_and_generator": True,
        "review_core_sha256": contract["runner_review_core_sha256"],
        "schema_id": "scryglass.representation-rank-runner-review-permit.v1",
    }
    permit_path = tmp_path / "review-core-permit.json"
    permit_path.write_bytes(assay.canonical_bytes(permit_payload))
    permit_raw = assay.raw_sha256(permit_path)
    monkeypatch.setattr(
        runner.runner_review_authority,
        "PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256",
        permit_raw,
    )
    sources = {
        **contract["source_identity"],
        "nuisance_oof_materialization": {
            **contract["source_identity"]["nuisance_oof_materialization"],
            "raw_sha256": "f" * 64,
        },
    }
    altered = _review_core_rehash(
        {
            **contract,
            "source_identity": sources,
            "runner_review_status": "PASS",
            "status": "private_runner_review_pass",
            "runner_review_permit": {
                "locator": str(permit_path),
                "raw_sha256": permit_raw,
            },
        }
    )
    called = False

    def forbidden(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("reader must remain unreachable")

    with pytest.raises(runner.PrivateRunnerError, match="permit invalid"):
        runner.load_authoritative_target_m0(
            altered, feature, root=ROOT, read_parquet=forbidden
        )
    assert called is False


def test_fit_availability_projection_and_fixed_point_counts(
    contract: dict, feature: runner.FeatureEnvelope
) -> None:
    calls = []

    def spy(path, **kwargs):
        calls.append(kwargs)
        return pd.read_parquet(path, **kwargs)

    availability = runner.load_fit_availability_domain(
        contract, feature, root=ROOT, read_parquet=spy
    )
    assert len(availability.ordered_game_ids) == 5702
    assert assay.canonical_sha256(list(availability.ordered_game_ids)) == (
        runner.PINNED_OOF_MEMBERSHIP_SHA256
    )
    split_source = contract["source_identity"]["outcome_free_split"]
    split = json.loads((ROOT / split_source["locator"]).read_bytes())
    assignments = {str(row["game_id"]): row for row in split["assignments"]}
    final_ids = {
        game_id
        for game_id, row in assignments.items()
        if row["split"] == assay.FINAL_SPLIT
    }
    assert len(final_ids) == runner.PINNED_FINAL_ROWS == 361
    assert not (set(availability.ordered_game_ids) & final_ids)
    assert all(
        assignments[game_id]["split"] in runner.NONHOLDOUT_SPLITS
        for game_id in availability.ordered_game_ids
    )
    assert calls == [{"columns": list(runner.SAFE_FIT_AVAILABILITY_COLUMNS)}]
    assert not (
        runner.FORBIDDEN_READINESS_COLUMNS & set(calls[0]["columns"])
    )
    likelihood = runner.likelihood_feature_domain(
        feature, availability.ordered_game_ids
    )
    expected = {
        "2025-03": (86, 68),
        "2025-04": (335, 235),
        "2025-05": (391, 252),
        "2025-06": (228, 83),
        "2025-07": (250, 111),
        "2025-08": (427, 217),
        "2025-09": (210, 66),
        "2025-10": (128, 55),
        "2026-01": (230, 152),
        "2026-02": (421, 221),
        "2026-03": (202, 83),
        "2026-04": (515, 288),
        "2026-05": (569, 243),
    }
    for month, (maps, clusters) in expected.items():
        split = (
            "train"
            if month < "2025-10"
            else "development"
            if month < "2026-04"
            else "validation"
        )
        score = [
            row[0]
            for row in likelihood.records
            if row[1] == split and row[3] == month
        ]
        fit = [
            row[0] for row in likelihood.records if row[3] < month
        ]
        coverage = assay.outcome_free_coverage(
            feature_domain=likelihood,
            score_game_ids=score,
            fit_game_ids=fit,
            split=split,
            fit_availability_domain=availability,
        )
        retained_clusters = len(
            {
                cluster
                for cluster, keep in zip(
                    coverage.eligibility_binding.ordered_source_cluster_ids,
                    coverage.eligible_rows,
                )
                if bool(keep)
            }
        )
        assert int(coverage.eligible_rows.sum()) == maps
        assert retained_clusters == clusters
        assert coverage.report["fit_support"]["derivation"] == (
            "maximal_monotone_fixed_point"
        )


def test_exact_fit_population_is_feature_target_m0_intersection(
    feature: runner.FeatureEnvelope,
) -> None:
    eligible = [
        row[0]
        for row in feature.domain.records
        if row[1] == "train" and row[3] < "2025-06"
    ][:12]
    target = _synthetic_target(feature, eligible)
    assert runner.exact_fit_game_ids(
        feature, target, prediction_month="2025-06"
    ) == tuple(
        row[0]
        for row in feature.domain.records
        if row[0] in set(eligible) and row[3] < "2025-06"
    )
    mismatched = replace(target, m0_by_game_id=target.m0_by_game_id[:-1])
    with pytest.raises(runner.PrivateRunnerError, match="exactly equal"):
        runner.exact_fit_game_ids(
            feature, mismatched, prediction_month="2025-06"
        )


def test_score_requires_exact_bound_eligible_prediction_population(
    contract: dict, feature: runner.FeatureEnvelope
) -> None:
    availability = runner.load_fit_availability_domain(
        contract, feature, root=ROOT
    )
    likelihood = runner.likelihood_feature_domain(
        feature, availability.ordered_game_ids
    )
    score_ids = [
        row[0]
        for row in likelihood.records
        if row[1] == "validation" and row[3] == "2026-04"
    ]
    fit_ids = [row[0] for row in likelihood.records if row[3] < "2026-04"]
    binding = assay.outcome_free_coverage(
        feature_domain=likelihood,
        score_game_ids=score_ids,
        fit_game_ids=fit_ids,
        split="validation",
        fit_availability_domain=availability,
    ).eligibility_binding
    records = {row[0]: row for row in likelihood.records}
    eligible = np.asarray(binding.eligible_nodes, dtype=bool)
    expected = tuple(
        game_id
        for game_id in binding.ordered_source_game_ids
        if np.all(eligible[np.asarray(records[game_id][5], dtype=np.int64)])
    )
    excluded_source = next(
        game_id
        for game_id in binding.ordered_source_game_ids
        if game_id not in set(expected)
    )
    dummy_fit = assay.LatentFit(
        width=1,
        ally_centered=np.zeros((len(eligible), 1)),
        enemy_centered=np.zeros((len(eligible), 2)),
        objective=0.0,
        maximum_absolute_gradient=0.0,
        converged_starts=2,
        best_two_interaction_logit_rms=0.0,
        eligibility_binding_sha256=binding.artifact_sha256,
    )
    p0 = {row[0]: 0.5 for row in likelihood.records}
    scored = runner.score_latent_fit(
        fit=dummy_fit,
        eligibility_binding=binding,
        game_ids=expected,
        verified_nuisance_oof=p0,
    )
    assert tuple(scored) == expected
    mutations = {
        "omission": expected[:-1],
        "reordering": (expected[1], expected[0], *expected[2:]),
        "earlier fit substitution": (
            binding.ordered_fit_game_ids[0],
            *expected[1:],
        ),
        "ineligible source substitution": (
            excluded_source,
            *expected[1:],
        ),
    }
    for changed in mutations.values():
        with pytest.raises(
            runner.PrivateRunnerError,
            match="bound eligible prediction population",
        ):
            runner.score_latent_fit(
                fit=dummy_fit,
                eligibility_binding=binding,
                game_ids=changed,
                verified_nuisance_oof=p0,
            )


def test_penalty_family_failure_is_inconclusive_m0_without_cherry_pick() -> None:
    calls = []

    def evaluate(month: str, penalty: float) -> dict:
        calls.append((month, penalty))
        if month == assay.INNER_MONTHS[2] and penalty == assay.PENALTY_GRID[0]:
            raise runner.PrivateRunnerError("fit unavailable")
        return {}

    result = runner.run_penalty_family(family="ally", evaluate=evaluate)
    assert result.status == "inconclusive"
    assert result.selected is None
    assert result.fallback == "M0"
    assert result.reason_code == "penalty_selection_failed"
    assert len(calls) == 3


def test_development_m8_is_exact_width_8_alias(monkeypatch) -> None:
    width8 = np.array([0.4, 0.6])

    def select(**kwargs):
        assert kwargs["predictions"]["M8"] is width8
        assert kwargs["predictions"][8] is width8
        return 2, {"locked_width": 2}

    monkeypatch.setattr(assay, "select_development_width", select)
    locked, result = runner.choose_development_width(
        prepared_fold=object(),
        game_ids=["a", "b"],
        target_domain=object(),
        width_predictions={
            1: np.array([0.5, 0.5]),
            2: np.array([0.5, 0.5]),
            4: np.array([0.5, 0.5]),
            8: width8,
        },
        m0=np.array([0.5, 0.5]),
        m8_optimization_stable=True,
    )
    assert locked == 2
    assert result["status"] == "pass"


def test_validation_forwards_only_locked_width_m0_and_m8(monkeypatch) -> None:
    def validate(**kwargs):
        assert set(kwargs["predictions"]) == {4, "M0", "M8"}
        assert kwargs["locked_width"] == 4
        return {"passed": True}

    monkeypatch.setattr(assay, "validate_locked_width", validate)
    result = runner.validate_locked_candidate(
        prepared_fold=object(),
        game_ids=["g"],
        locked_width=4,
        target_domain=object(),
        locked_prediction=[0.6],
        m0=[0.5],
        m8=[0.61],
        m8_optimization_stable=True,
    )
    assert result == {"status": "pass", "fallback": "none", "passed": True}


def test_selector_failures_expose_only_fixed_reason_codes(monkeypatch) -> None:
    leaked = "game_id=private-123 /tmp/private-target.parquet"
    monkeypatch.setattr(
        assay,
        "select_development_width",
        lambda **kwargs: (_ for _ in ()).throw(
            assay.RepresentationRankAssayError(leaked)
        ),
    )
    locked, development = runner.choose_development_width(
        prepared_fold=object(),
        game_ids=["g"],
        target_domain=object(),
        width_predictions={
            width: np.array([0.5]) for width in assay.WIDTHS
        },
        m0=np.array([0.5]),
        m8_optimization_stable=True,
    )
    assert locked is None
    assert development == {
        "status": "inconclusive",
        "fallback": "M0",
        "reason_code": "development_gate_failed",
    }
    assert leaked not in repr(development)

    monkeypatch.setattr(
        assay,
        "validate_locked_width",
        lambda **kwargs: (_ for _ in ()).throw(
            assay.RepresentationRankAssayError(leaked)
        ),
    )
    validation = runner.validate_locked_candidate(
        prepared_fold=object(),
        game_ids=["g"],
        locked_width=2,
        target_domain=object(),
        locked_prediction=[0.5],
        m0=[0.5],
        m8=[0.5],
        m8_optimization_stable=True,
    )
    assert validation == {
        "status": "inconclusive",
        "fallback": "M0",
        "reason_code": "validation_gate_failed",
    }
    assert leaked not in repr(validation)


def test_legacy_row_level_artifact_surface_is_prohibited(tmp_path: Path) -> None:
    with pytest.raises(runner.PrivateRunnerError, match="row-level/parquet"):
        runner.write_private_artifact(
            pd.DataFrame(
                {
                    "game_id": ["g1"],
                    "split": ["development"],
                    "p_blue_win_selected": [0.5],
                }
            ),
            parquet_path=tmp_path / "x.parquet",
            manifest_path=tmp_path / "x.json",
        )
    assert not list(tmp_path.iterdir())
    with pytest.raises(runner.PrivateRunnerError, match="row-level/parquet"):
        runner.verify_private_artifact(
            parquet_path=tmp_path / "x.parquet",
            manifest_path=tmp_path / "x.json",
        )


@pytest.mark.parametrize("arguments", [[], ["--run-private"]])
def test_cli_refuses_default_and_pending_private_run(arguments: list[str]) -> None:
    """Default fitting is refused; with the (now PASS) runner review, an
    explicit --run-private executes the outcome-free fit and stays
    inconclusive (no target/outcome access, no authority)."""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lol_kills.v2.draft.interactions.representation_rank_private_runner",
            "--contract",
            str(CONTRACT_PATH),
            *arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if not arguments:
        assert completed.returncode != 0
        assert "fitting refused by default" in completed.stderr
    else:
        assert completed.returncode == 0, completed.stderr
        result = json.loads(completed.stdout)
        assert result["run_status"] == "inconclusive"


def test_real_verify_ready_subprocess_loads_no_target_or_outcome() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lol_kills.v2.draft.interactions.representation_rank_private_runner",
            "--contract",
            str(CONTRACT_PATH),
            "--verify-ready",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["feature_rows"] == 5949
    assert result["real_target_loader_invoked"] is False
    assert result["target_rows_loaded"] is False
    assert result["outcome_columns_loaded"] is False
    assert result["final_target_loaded"] is False
    assert result["candidate_fit_started"] is False
    assert "y_blue_win" not in result["safe_feature_projection"]
