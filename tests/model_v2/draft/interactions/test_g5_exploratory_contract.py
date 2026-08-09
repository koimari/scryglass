from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import contract as g5


ROOT = Path(__file__).parents[4]
ARTIFACTS = ROOT / "data/lol/v2/models/draft-interactions/g5-exploratory"


def _read(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def _bundle() -> tuple[dict, dict]:
    bundle = g5.build_prefit_bundle()
    return bundle["contract.json"], bundle["review-core.json"]


def test_prefit_construction_never_reads_rows_targets_or_final_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = Path.read_bytes

    def watched(path: Path) -> bytes:
        calls.append(str(path.relative_to(ROOT)))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", watched)
    bundle = g5.build_prefit_bundle()
    assert bundle["contract.json"]["target_access"]["pre_freeze_target_or_outcome_row_reads"] == 0
    assert bundle["pre-fit-review.json"]["execution_authorization"] is False
    assert all("lpl-private-development-rows.jsonl" not in call for call in calls)
    assert all("target" not in call.lower() and "final" not in call.lower() for call in calls)


def test_dependency_verification_binds_supplementary_input_without_opening_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original = Path.read_bytes

    def watched(path: Path) -> bytes:
        calls.append(str(path.relative_to(ROOT)))
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", watched)
    bindings = g5.verify_bound_dependencies()
    assert {"g1_features_loader_verifier", "g1_features_manifest", "g1_features_independent_review", "g1_features_accepted_metadata", "g2_runner", "g2_model", "clusters_proxy", "runtime_scipy", "runtime_numpy"} <= set(bindings)
    assert all("lpl-private-development-rows.jsonl" not in call for call in calls)
    assert all("lpl-private-draft-features-rows.jsonl" not in call for call in calls)
    assert all("target" not in call.lower() and "final" not in call.lower() for call in calls)


def test_b0_d1_math_train_lock_and_exact_validation_gate_are_frozen() -> None:
    contract, _ = _bundle()
    protocol = contract["candidate_protocol"]
    assert [candidate["candidate_id"] for candidate in protocol["eligible_candidates"]] == ["B0", "D1"]
    assert protocol["D2"] == g5.D2_OMISSION
    b0, d1 = protocol["eligible_candidates"]
    assert "score each TRAIN target before its own" in b0["train_state"]
    assert "no state update" in b0["development_validation_state"]
    assert d1["formula"].startswith("logit(p_D1)=eta_B0")
    assert d1["priors_and_penalty"]["mu"]["lambda"] == 12.5
    assert d1["priors_and_penalty"]["delta"]["lambda"] == 50.0
    assert "no sum-to-zero" in d1["priors_and_penalty"]["identifiability"]
    objective = d1["numerical_objective"]
    assert objective["negative_log_likelihood"] == "sum_i(numpy.logaddexp(0.0,z_i)-y_i*z_i)"
    assert "X^T(scipy.special.expit(z)-y)+Lambda beta" in objective["gradient"]
    assert "X^T diag(p_i*(1-p_i)) X+Lambda" in objective["hessian"]
    assert "no logit or probability clipping" in objective["domain"]
    assert "EXECUTION_BLOCKED" in objective["failure"]
    assert d1["solver"]["gradient_infinity_norm_at_MAP_at_most"] == 1e-6
    assert d1["solver"]["jacobian"] == "analytic_only" and d1["solver"]["bounds"] == "none"
    assert protocol["selection"]["validation_d1_winner_rule"]["all_required"] == ["D1_was_locked_by_DEVELOPMENT", "mean(LL_B0-LL_D1)>=0.005", "one_sided_paired_series_cluster_bootstrap_95_lower_bound>0"]


def test_availability_uncertainty_terminal_and_contextual_boundaries_are_explicit() -> None:
    contract, _ = _bundle()
    math = contract["mathematics_availability_uncertainty"]
    assert math["availability"]["champion_absent_from_TRAIN"].startswith("EXECUTION_BLOCKED")
    assert "PRIOR_ONLY_ROLE_DELTA" in math["availability"]["seen_champion_unseen_exact_role"]
    assert "must not be inserted into fitted covariance" in math["availability"]["prior_only_coordinate_rules"]
    assert "no pair" in math["availability"]["D2"]
    assert math["parameter_uncertainty"]["D1_conditional_variance"] == "Var_D1(d)=x_fitted^T C x_fitted+0.01*N_prior_only_role_deltas; N_prior_only_role_deltas counts unique absent fitted (role,champion) delta coordinates only"
    assert "sqrt(Var_D1(d))" in math["parameter_uncertainty"]["interval"]
    assert "prior_only_role_slot_membership_sha256" in math["parameter_uncertainty"]["prior_only_ledger"]
    assert "do not report a total" in math["parameter_uncertainty"]["prohibition"]
    assert "eta_B0_variance and Var_D1(d) are unchanged" in math["invariances"]["side_swap"]
    assert "absolute tolerance 1e-12" in math["invariances"]["reconciliation"]
    assert math["contextual_score"]["supplementary_rows_not_sufficient"] is True
    assert contract["execution_result_schema"]["terminal_states"] == ["PREFIT_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED", "PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER", "NO_INCREMENTAL_DRAFT_WINNER", "EXECUTION_BLOCKED"]
    assert {"objective_gradient_hessian_diagnostics", "prior_only_variance_components"} <= set(contract["execution_result_schema"]["required_execution_result"])
    assert "EXECUTION_BLOCKED" in contract["execution_result_schema"]["solver_or_uncertainty_failure"]


def test_bootstrap_algorithm_is_exact_and_distinct_from_parameter_uncertainty() -> None:
    contract, _ = _bundle()
    bootstrap = contract["bootstrap"]
    assert bootstrap["replicates"] == 2000
    assert bootstrap["base_seed"] == 2026073005
    assert "seed=2026073005+b" in bootstrap["sampling"]
    assert bootstrap["lower_bound"] == {"confidence": 0.95, "quantile": 0.05, "method": "numpy.quantile(method='linear')", "strict_rule": "lower_bound > 0.0"}
    assert "EXECUTION_BLOCKED" in bootstrap["failure"]


def test_artifacts_are_canonical_content_addressed_and_reviewed() -> None:
    rebuilt = g5.build_prefit_bundle()
    for name, expected in rebuilt.items():
        raw = (ARTIFACTS / name).read_bytes()
        actual = json.loads(raw)
        assert raw == g5.canonical_bytes(actual) + b"\n"
        unsigned = dict(actual)
        claimed = unsigned.pop("artifact_sha256")
        assert claimed == g5.sha256(unsigned)
        assert actual == expected
    review = _read("pre-fit-review.json")
    assert review["disposition"] == "ACCEPT_PREFIT_ONLY"
    assert review["execution_authorization"] is False
    assert all(review["hostile_checks"].values())


def test_review_core_binds_contract_test_and_every_runtime_dependency() -> None:
    core = _read("review-core.json")
    assert set(core["review_subject_bytes"]) == {"package_init", "contract", "focused_test"}
    for item in core["review_subject_bytes"].values():
        assert hashlib.sha256((ROOT / item["locator"]).read_bytes()).hexdigest() == item["raw_sha256"]
    pins = core["dependency_pins"]
    assert pins["G1_features"] == g5.G1_FEATURES
    assert pins["G2"] == g5.G2
    assert pins["clusters"] == g5.CLUSTERS
    assert pins["runtime"] == g5.RUNTIME


@pytest.mark.parametrize(
    ("mutation", "failed_check"),
    [
        (lambda c: c["target_access"].update({"pre_freeze_target_or_outcome_row_reads": 1}), "zero_prefit_target_or_outcome_reads"),
        (lambda c: c["target_access"].update({"final_holdout_reads": 1}), "sealed_final_holdout"),
        (lambda c: c["candidate_protocol"]["eligible_candidates"].append({"candidate_id": "D3"}), "candidate_family_exact"),
        (lambda c: c["candidate_protocol"]["eligible_candidates"][0].update({"development_validation_state": "update all folds"}), "b0_train_prequential_then_frozen"),
        (lambda c: c["candidate_protocol"]["eligible_candidates"][1].update({"fit": "TRAIN_and_DEVELOPMENT"}), "train_development_validation_leakage_blocked"),
        (lambda c: c["candidate_protocol"]["eligible_candidates"][1]["priors_and_penalty"].update({"identifiability": "sum-to-zero"}), "d1_proper_priors_no_false_constraint"),
        (lambda c: c["candidate_protocol"]["eligible_candidates"][1]["numerical_objective"].update({"gradient": "finite difference"}), "d1_objective_gradient_hessian_frozen"),
        (lambda c: c["candidate_protocol"]["selection"].update({"VALIDATION": "refit and tune"}), "development_validation_lock"),
        (lambda c: c["candidate_protocol"]["selection"]["validation_d1_winner_rule"].update({"all_required": ["mean>0"]}), "validation_gate_exact"),
        (lambda c: c["bootstrap"].update({"replicates": 1}), "bootstrap_exact_and_fail_closed"),
        (lambda c: c["mathematics_availability_uncertainty"]["availability"].update({"seen_champion_unseen_exact_role": "use observed delta"}), "availability_prior_only_and_d2_no_pair"),
        (lambda c: c["mathematics_availability_uncertainty"]["availability"].update({"prior_only_coordinate_rules": "insert missing coordinate into C"}), "availability_prior_only_and_d2_no_pair"),
        (lambda c: c["mathematics_availability_uncertainty"]["parameter_uncertainty"].update({"D1_conditional_variance": "x_fitted^T C x_fitted"}), "separate_uncertainty_no_false_total"),
        (lambda c: c["mathematics_availability_uncertainty"]["parameter_uncertainty"].update({"interval": "d +/- 1.96 sqrt(x C x + 0.01*N*2)"}), "separate_uncertainty_no_false_total"),
        (lambda c: c["mathematics_availability_uncertainty"]["parameter_uncertainty"].update({"prohibition": "report total interval"}), "separate_uncertainty_no_false_total"),
        (lambda c: c["mathematics_availability_uncertainty"]["contextual_score"].update({"status": "available_by_default"}), "contextual_overreach_blocked"),
        (lambda c: c["claim_ceiling"].update({"publication": True}), "terminal_and_claim_schema"),
        (lambda c: c["execution_result_schema"].update({"terminal_states": ["PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER"]}), "terminal_and_claim_schema"),
        (lambda c: c["input_identities"].update({"G1_features": {}}), "accepted_supplementary_input_bound"),
    ],
)
def test_hostile_contract_mutations_fail_closed(mutation, failed_check: str) -> None:
    contract, core = _bundle()
    altered = deepcopy(contract)
    mutation(altered)
    review = g5.review_prefit_contract(altered, core)
    assert review["disposition"] == "REMAND"
    assert review["hostile_checks"][failed_check] is False


def test_dependency_substitution_fails_closed() -> None:
    with pytest.raises(g5.G5PreFitError, match="bound dependency changed"):
        g5._verify_bound(g5.G1_FEATURES["loader_verifier_locator"], "0" * 64)
