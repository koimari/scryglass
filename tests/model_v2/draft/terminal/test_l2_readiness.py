from __future__ import annotations

from lol_kills.v2.draft.terminal import inspect_l2_readiness
from lol_kills.v2.draft.terminal import l2_readiness


def test_l2_readiness_audit_keeps_development_package_blocked() -> None:
    report = inspect_l2_readiness()
    assert report["status"] == "blocked"
    assert report["promotion_eligible"] is False
    assert report["public_probability_authorized"] is False
    assert report["prospective_supersession"]["active"] is False
    assert report["prospective_supersession"][
        "historical_nested_outer_all_nonharmful"
    ] is False
    assert report["prospective_supersession"][
        "historical_outer_test_pass_count"
    ] == 2
    assert report["prospective_supersession"]["historical_outer_test_count"] == 3
    assert report["prospective_supersession"]["superseded_blockers"] == []
    assert report["checks"]["artifact_bound_to_registry"] is True
    assert report["checks"]["evaluation_summary_binds_runner_output"] is True
    assert report["checks"]["l2_contract_policy_frozen"] is True
    assert report["checks"]["contextual_standardization_policy_frozen"] is True
    assert report["checks"]["neutral_standardization_policy_frozen"] is True
    assert report["checks"]["roster_change_holdout_available"] is True
    assert report["checks"]["candidate_registry_independent_l2_authority"] is False
    assert report["checks"]["independent_authority_record_present"] is False
    assert report["checks"]["participant_clusters_available"] is False
    assert (
        report["checks"][
            "participant_dependence_method_predeclared_and_valid"
        ]
        is True
    )
    assert report["checks"]["participant_dependence_support_verified"] is False
    assert (
        report["checks"][
            "participant_dependence_diagnostic_present_and_valid"
        ]
        is True
    )
    assert report["artifacts"]["independent_authority_record"]["present"] is False
    assert report["independent_authority_record_error"] is None
    assert "independent_authority_record_present" in report["blockers"]
    assert "participant_clusters_available" not in report["blockers"]
    assert "participant_dependence_support_verified" in report["blockers"]
    assert (
        "participant_dependence_method_predeclared_and_valid"
        not in report["blockers"]
    )
    assert (
        "participant_dependence_diagnostic_present_and_valid"
        not in report["blockers"]
    )
    assert report["required_next_authority"]["must_be_independent"] is True
    assert report["required_next_authority"][
        "historical_development_artifacts_remain_unchanged"
    ] is True
    assert "exact_evaluated_model_locator_raw_sha256_and_version" in report[
        "required_next_authority"
    ]["must_bind"]
    assert report["required_next_authority"]["betting_authorized"] is False
    assert report["g1_contextual_source"]["status"] == "externally_blocked"
    assert report["g1_contextual_source"]["applies_to"] == "contextual_only"
    assert report["g1_contextual_source"]["blocks_neutral"] is False
    assert report["g1_contextual_source"]["authority_available"] is False
    assert report["g1_contextual_source"]["procurement_attempt"]["status"] == "externally_blocked"
    assert "pre_draft_roles_missing_from_GRID_state" in report["g1_contextual_source"]["procurement_attempt"]["blockers"]
    assert report["artifacts"]["g1_procurement_attempt"]["present"] is True
    assert report["grid_promotion_gate"]["status"] == "not_passed"
    assert report["grid_promotion_gate"]["primary_source_for_cohort"] == "OE"
    assert report["grid_promotion_gate"]["public_reproducibility_benchmark"] == "OE"
    assert report["grid_promotion_gate"]["failed_procurement_attempt"]["series_id"] == "2974293"
    assert report["grid_promotion_gate"]["missing_or_invalid_records"]["invalid"][0]["series_id"] == "2974293"
    assert "pre_draft_patch_missing_from_GRID_state" in report["grid_promotion_gate"]["blockers"]
    assert report["checks"]["adaptive_temporal_diagnostic_present_and_valid"] is True
    assert report["checks"]["known_harmful_app_draft_family_quarantined"] is True
    assert report["checks"]["selected_candidate_uses_incremental_context_estimand"] is True
    assert report["checks"]["selected_candidate_all_validation_folds_nonharmful"] is True
    assert report["checks"]["selected_candidate_all_nested_outer_folds_nonharmful"] is False
    assert report["checks"]["future_protocol_locked_and_valid"] is True
    assert report["checks"]["future_capture_readiness_locked_and_valid"] is True
    assert (
        report["checks"][
            "grid_terminal_draft_source_readiness_locked_and_valid"
        ]
        is True
    )
    assert report["checks"]["future_prediction_ledger_present_and_valid"] is False
    assert report["checks"]["future_holdout_support_met"] is False
    assert report["checks"]["future_protocol_independent_review_present"] is False
    assert report["checks"]["semantic_terminal_draft_authority_active"] is False
    assert report["adaptive_temporal_diagnostic"]["status"] == "valid"
    assert report["adaptive_temporal_diagnostic"]["result_state"] == "ADAPTIVE_DRAFT_TERMS_HARM"
    assert report["adaptive_temporal_diagnostic"]["population"]["exact_roster_context_maps"] == 267
    assert report["adaptive_temporal_diagnostic"]["decision"]["independent_validation"] is False
    assert report["participant_dependence_diagnostic"]["status"] == "valid"
    assert (
        report["participant_dependence_diagnostic"]["population"]
        ["maps_with_exact_ten_unique_players_and_roles"]
        == 5751
    )
    assert (
        report["participant_dependence_diagnostic"]["population"]
        ["component_graph"]["all_valid_maps_in_one_component"]
        is True
    )
    assert (
        report["participant_dependence_diagnostic"]["decision"]
        ["participant_dependence_support_verified"]
        is False
    )
    assert report["participant_dependence_method"]["status"] == (
        "predeclared_pending_independent_future_evaluation"
    )
    assert (
        report["participant_dependence_method"]["contract"][
            "locked_before_future_outcomes"
        ]
        is True
    )
    assert (
        report["participant_dependence_method"][
            "atomic_component_split_required"
        ]
        is False
    )
    assert (
        report["participant_dependence_method"][
            "independent_future_support_verified"
        ]
        is False
    )
    assert "selected_candidate_all_nested_outer_folds_nonharmful" in report["blockers"]
    assert "future_holdout_support_met" in report["blockers"]
    assert "future_prediction_ledger_present_and_valid" in report["blockers"]
    assert "future_protocol_independent_review_present" in report["blockers"]
    assert "semantic_terminal_draft_authority_active" in report["blockers"]
    assert report["future_protocol"]["status"] == "valid"
    assert report["future_protocol"]["future_holdout"]["status"] == "EMPTY_NOT_YET_ACQUIRED"
    assert report["future_protocol"]["locked_candidate"]["variant_id"] == "m0-role-additive@ridge-0.05"
    assert report["future_protocol"]["estimands"]["neutral_output_directly_outcome_calibrated"] is False
    assert report["future_capture_readiness"]["status"] == "valid"
    assert report["future_capture_readiness"]["ledger_state_at_lock"]["entries"] == 0
    assert report["future_capture_readiness"]["implementation"]["ready_for_outcome_free_future_capture"] is True
    assert report["grid_source_readiness"]["status"] == "valid"
    assert (
        report["grid_source_readiness"]["capability_conclusion"][
            "terminal_pick_ban_prestart_observed_in_all_archives"
        ]
        is True
    )
    assert (
        report["grid_source_readiness"]["capability_conclusion"][
            "prestart_role_assignment_available_from_grid"
        ]
        is False
    )
    assert report["future_prediction_ledger"]["status"] == "missing"
    assert report["artifacts"]["future_capture_readiness"]["present"] is True
    assert report["artifacts"]["grid_source_readiness"]["present"] is True
    assert report["artifacts"]["future_prediction_ledger"]["present"] is False
    assert report["artifacts"]["semantic_draft_authority"][
        "present_and_valid"
    ] is False
    assert report["artifacts"]["semantic_draft_authority"]["error"] == (
        "external Draft-authority pin missing"
    )
    assert "roster_change_holdout_available" not in report["blockers"]


def test_l2_readiness_audit_is_replayable() -> None:
    assert inspect_l2_readiness() == inspect_l2_readiness()


def test_only_exact_future_semantic_authority_can_supersede_legacy_blockers() -> None:
    checks = {
        name: False for name in l2_readiness._PROSPECTIVE_SUPERSEDED_CHECKS
    }
    checks["unrelated_live_gate"] = False

    inactive_blockers, inactive_superseded = l2_readiness._resolve_blockers(
        checks, prospective_supersession_active=False
    )
    assert "selected_candidate_all_nested_outer_folds_nonharmful" in (
        inactive_blockers
    )
    assert inactive_superseded == []

    active_blockers, active_superseded = l2_readiness._resolve_blockers(
        checks, prospective_supersession_active=True
    )
    assert active_blockers == ["unrelated_live_gate"]
    assert active_superseded == sorted(
        l2_readiness._PROSPECTIVE_SUPERSEDED_CHECKS
    )
    assert checks["selected_candidate_all_nested_outer_folds_nonharmful"] is False
