from __future__ import annotations

from datetime import datetime, timezone

from lol_kills import private_decision_readiness as readiness


AS_OF = datetime(2026, 8, 1, 20, 0, tzinfo=timezone.utc)


def draft_blocked(*_args, **_kwargs) -> dict:
    return {
        "status": "blocked",
        "promotion_eligible": False,
        "public_probability_authorized": False,
        "blockers": ["independent_authority_record_present"],
    }


def test_current_rating_and_total_artifacts_remain_fail_closed(monkeypatch) -> None:
    monkeypatch.setattr(readiness, "inspect_l2_readiness", draft_blocked)
    report = readiness.inspect_private_decision_readiness(
        ".", as_of=AS_OF, environment={}
    )
    assert report["schema_version"] == "scryglass.private-decision-readiness.v15"
    assert report["status"] == "blocked"
    assert report["betting_ready"] is False
    assert report["event_authorization"]["self_authorizing"] is False
    assert report["ratings"]["artifacts"]["player"]["result_state"] == (
        "DEVELOPMENT_CANDIDATE_VALIDATION_GATE_FAILED"
    )
    assert report["ratings"]["artifacts"]["team"]["result_state"] == (
        "DEVELOPMENT_CANDIDATE_VALIDATION_GATE_FAILED"
    )
    assert report["ratings"]["checks"]["player_final_holdout_available"] is False
    assert report["ratings"]["checks"]["overall_validation_uncertainty_gate_passed"] is False
    assert report["ratings"]["checks"]["lcs_validation_uncertainty_gate_passed"] is False
    assert report["ratings"]["checks"]["warehouse_source_pins_match_current_files"] is True
    assert report["ratings"]["checks"][
        "prospective_source_snapshot_replaces_mutable_warehouse_dependency"
    ] is False
    assert (
        "prospective_source_snapshot_replaces_mutable_warehouse_dependency"
        in report["ratings"]["blockers"]
    )
    assert report["ratings"]["checks"][
        "semantic_output_contract_trust_root_current_and_valid"
    ] is False
    assert report["ratings"]["checks"][
        "semantic_output_contract_reconciliation_candidate_present_and_valid"
    ] is False  # C0 contract-tree freeze predates the current docs (L2 re-freeze pending)
    assert report["ratings"]["checks"][
        "semantic_output_contract_candidate_reference_replay_passed"
    ] is False  # C0 contract-tree freeze
    assert report["ratings"]["checks"][
        "semantic_output_contract_exact_prior_tree_recovered"
    ] is False  # C0 contract-tree freeze
    assert report["ratings"]["checks"][
        "semantic_output_contract_reconciliation_independently_reviewed"
    ] is False
    assert (
        report["ratings"]["checks"]["strong_baseline_benchmark_present_and_valid"]
        is True
    )
    assert (
        report["ratings"]["checks"][
            "v3_replayable_joint_source_snapshot_present"
        ]
        is True
    )
    assert report["ratings"]["checks"][
        "v3_failed_v1_source_evidence_preserved"
    ] is True
    assert report["ratings"]["checks"][
        "v3_temporal_failure_receipt_present_and_valid"
    ] is True
    assert report["ratings"]["checks"][
        "v3_future_dated_v2_receipts_rejected"
    ] is True
    assert report["ratings"]["checks"][
        "v3_corrected_source_preflight_passed"
    ] is True
    assert report["ratings"]["checks"][
        "v3_corrected_adaptive_candidate_diagnostic_present_and_valid"
    ] is True
    assert report["ratings"]["checks"][
        "v3_corrected_adaptive_diagnostic_preserves_future_holdout"
    ] is True
    assert report["ratings"]["checks"][
        "v3_future_protocol_lock_present_and_valid"
    ] is True
    assert report["ratings"]["checks"][
        "v3_protocol_supersession_preserves_candidate_and_boundary"
    ] is True
    assert report["ratings"]["checks"]["v3_future_targets_still_unopened"] is True
    assert report["ratings"]["checks"]["v3_future_holdout_support_met"] is False
    assert report["ratings"]["checks"][
        "v3_future_prediction_ledger_present_and_valid"
    ] is False
    assert report["ratings"]["checks"][
        "v3_joint_future_evaluation_independently_registered_and_passed"
    ] is False
    assert report["ratings"]["checks"][
        "v3_pre_event_prediction_ledger_capture_ready"
    ] is False
    assert report["ratings"]["checks"][
        "v3_prediction_and_ledger_system_clock_hardened"
    ] is False
    assert report["ratings"]["checks"][
        "v3_pre_event_prediction_ledger_has_eligible_entries"
    ] is False
    assert report["ratings"]["checks"][
        "v3_independent_protocol_review_present"
    ] is False
    assert report["ratings"]["artifacts"]["v3_source_preflight_v1"][
        "result_state"
    ] == "SOURCE_SCHEMA_PREFLIGHT_FAILED"
    assert report["ratings"]["artifacts"]["v3_temporal_failure"][
        "result_state"
    ] == "FUTURE_DATED_RECEIPTS_REJECTED_AND_SUPERSESSION_REQUIRED"
    assert report["ratings"]["artifacts"]["v3_source_preflight_v2_rejected"][
        "result_state"
    ] == "CORRECTED_SOURCE_PREFLIGHT_PASSED_NON_AUTHORIZING"
    assert report["ratings"]["artifacts"]["v3_source_preflight_v2_rejected"][
        "qualifies_as_future_evidence"
    ] is False
    assert report["ratings"]["artifacts"]["v3_source_preflight_v3"][
        "result_state"
    ] == "CORRECTED_SOURCE_PREFLIGHT_PASSED_NON_AUTHORIZING"
    assert report["ratings"]["artifacts"][
        "v3_corrected_adaptive_candidate_diagnostic"
    ]["result_state"] == "INCUMBENT_RETAINED_NO_ADAPTIVE_SUPERSESSION_EVIDENCE"
    assert report["ratings"]["artifacts"][
        "v3_corrected_adaptive_candidate_diagnostic"
    ]["retention_decision"]["does_not_validate_incumbent"] is True
    assert report["ratings"]["artifacts"]["v3_future_protocol_v1_superseded"][
        "operational_status"
    ] == "SUPERSEDED_AFTER_FAILED_SOURCE_SCHEMA_PREFLIGHT"
    assert report["ratings"]["artifacts"]["v3_future_protocol_v2_rejected"][
        "result_state"
    ] == "SUPERSEDING_FUTURE_HOLDOUT_PROTOCOL_LOCKED_EMPTY"
    assert report["ratings"]["artifacts"]["v3_future_protocol_v2_rejected"][
        "qualifies_as_future_evidence"
    ] is False
    assert report["ratings"]["artifacts"]["v3_future_protocol"][
        "result_state"
    ] == "CLOCK_CORRECTED_FUTURE_HOLDOUT_PROTOCOL_LOCKED_EMPTY"
    assert report["ratings"]["artifacts"]["v3_future_protocol"][
        "prediction_ledger"
    ]["entries"] == 0
    assert report["ratings"]["artifacts"]["v3_capture_readiness_v1_rejected"][
        "result_state"
    ] == "PRE_EVENT_CAPTURE_IMPLEMENTATION_READY_EMPTY_LEDGER"
    assert report["ratings"]["artifacts"]["v3_capture_readiness_v1_rejected"][
        "qualifies_as_future_evidence"
    ] is False
    assert report["ratings"]["artifacts"][
        "v3_capture_readiness_v2_superseded"
    ]["result_state"] == (
        "CLOCK_CORRECTED_PRE_EVENT_CAPTURE_IMPLEMENTATION_READY_EMPTY_LEDGER"
    )
    assert report["ratings"]["artifacts"][
        "v3_capture_readiness_v2_superseded"
    ]["qualifies_as_current_implementation_evidence"] is False
    assert report["ratings"]["artifacts"]["v3_capture_readiness"][
        "result_state"
    ] == "SYSTEM_CLOCKED_PRE_EVENT_CAPTURE_IMPLEMENTATION_READY_EMPTY_LEDGER"
    assert report["ratings"]["artifacts"]["v3_capture_readiness"][
        "clock_attestation"
    ]["user_supplied_timestamp_allowed"] is False
    assert report["ratings"]["artifacts"]["v3_capture_readiness"][
        "implementation"
    ]["ready_for_pre_event_capture"] is True
    assert report["ratings"]["checks"]["v2_protocol_lock_present_and_valid"] is True
    assert report["ratings"]["checks"]["v2_protocol_artifact_integrity_valid"] is True
    assert report["ratings"]["checks"]["v2_protocol_source_replay_valid"] is True
    assert (
        report["ratings"]["checks"][
            "v2_observed_validation_reclassified_as_adaptive_development"
        ]
        is True
    )
    assert (
        report["ratings"]["checks"]["v2_sealed_final_targets_still_unopened"]
        is True
    )
    assert (
        report["ratings"]["checks"][
            "v2_adaptive_candidate_selection_artifact_present"
        ]
        is True
    )
    assert (
        report["ratings"]["checks"]["v2_adaptive_selection_source_replay_valid"]
        is True
    )
    assert report["ratings"]["checks"]["v2_adaptive_candidate_selected"] is True
    assert (
        report["ratings"]["checks"][
            "v2_independent_sealed_opening_approval_present"
        ]
        is False
    )
    assert report["ratings"]["artifacts"]["v2_sealed_opening_authority"][
        "status"
    ] == "unavailable"
    assert report["ratings"]["artifacts"]["v2_protocol_lock"]["error"] is None
    assert report["ratings"]["artifacts"]["v2_adaptive_selection"]["error"] is None
    assert report["ratings"]["artifacts"]["v2_protocol_lock"][
        "result_state"
    ] == "EQUAL_SERIES_PROTOCOL_LOCKED_SEALED_FINAL_UNOPENED"
    assert report["ratings"]["artifacts"]["v2_protocol_lock"][
        "sealed_final_series"
    ] == 398
    assert report["ratings"]["artifacts"]["v2_adaptive_selection"][
        "result_state"
    ] == "EQUAL_SERIES_ADAPTIVE_CANDIDATE_SELECTED_SEALED_FINAL_UNOPENED"
    assert (
        report["ratings"]["checks"][
            "player_beats_strong_organization_baseline_overall"
        ]
        is False
    )
    assert (
        report["ratings"]["checks"][
            "player_beats_strong_organization_baseline_lcs"
        ]
        is False
    )
    assert (
        report["ratings"]["checks"][
            "player_beats_strong_organization_baseline_roster_change"
        ]
        is False
    )
    assert report["ratings"]["artifacts"]["strong_baseline_benchmark"][
        "result_state"
    ] == "PLAYER_DOES_NOT_BEAT_STRONG_BASELINE"
    assert (
        report["ratings"]["checks"][
            "team_identified_estimand_preserves_unavailable_components"
        ]
        is True
    )
    assert (
        report["ratings"]["checks"][
            "team_last_observed_exact_roster_aggregation_available"
        ]
        is True
    )
    assert (
        report["ratings"]["checks"][
            "team_pre_event_exact_roster_aggregation_available"
        ]
        is False
    )
    assert report["live_totals"]["checks"]["series_cluster_schema_current"] is True
    assert report["live_totals"]["checks"]["series_cluster_residuals_present"] is True
    assert report["live_totals"]["checks"]["development_candidate_code_pin_valid"] is True
    assert report["live_totals"]["checks"]["replayable_source_snapshot_present"] is True
    assert report["live_totals"]["checks"]["at_least_one_fresh_league"] is True
    assert report["live_totals"]["artifact"]["fresh_leagues"] == [
        "CBLOL",
        "LCS",
        "LEC",
    ]
    assert report["live_totals"]["artifact"]["latest_lcs_patch"] == "16.14"
    assert report["live_totals"]["artifact"]["latest_lcs_patch_min_test_n"] == 9
    assert report["live_totals"]["checks"]["latest_lcs_patch_holdout_sufficient"] is False
    assert (
        report["live_totals"]["checks"][
            "model_independent_validation_authority_present"
        ]
        is False
    )
    assert report["match_winner_market"]["protocol"]["present_and_valid"] is True
    assert report["match_winner_market"]["protocol"]["result_state"] == (
        "TWO_STAGE_PROSPECTIVE_MARKET_PROTOCOL_LOCKED_EMPTY"
    )
    assert report["match_winner_market"]["checks"][
        "future_market_protocol_locked_and_valid"
    ] is True
    assert report["match_winner_market"]["checks"][
        "phase_one_future_outcomes_still_sealed"
    ] is True
    assert report["match_winner_market"]["checks"][
        "outcome_free_phase_one_collection_contract_present"
    ] is True
    assert report["match_winner_market"]["checks"][
        "side_neutral_pre_side_and_binding_implementations_fail_closed"
    ] is True
    assert report["match_winner_market"]["checks"][
        "side_neutral_review_gated_end_to_end_operator_present"
    ] is True
    assert report["match_winner_market"]["checks"][
        "side_neutral_reviewed_operator_feeds_frozen_phase_one_evaluator"
    ] is True
    assert report["match_winner_market"]["checks"][
        "side_neutral_protocol_supersession_independently_registered"
    ] is False
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["side_binding_selects_existing_child_without_refit"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["artifact_counts"]["invalid_artifacts"] == 0
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["protocol_supersession_candidate_present"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["repository_code_pin_present"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["terminal_draft_adapter_source_present"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["complete_bundle_source_present"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["review_gated_operator_source_present"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["operator_requires_external_review_before_every_stage"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["review_gated_operator_phase_one_bridge_stages"] == [
        "draft",
        "map-start",
        "ledger",
    ]
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["independent_review_present_and_valid"] is False
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["admission_implementation_code_pin_valid"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["independent_review_packet_present_and_valid"] is True
    assert report["match_winner_market"]["checks"][
        "side_neutral_review_packet_and_admission_code_pins_valid"
    ] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "side_neutral_revision"
    ]["independently_registered"] is False
    assert report["match_winner_market"]["checks"][
        "side_neutral_reviewed_ledger_present_and_valid"
    ] is False
    assert report["match_winner_market"]["checks"][
        "side_neutral_reviewed_ledger_has_eligible_entries"
    ] is False
    assert report["match_winner_market"]["checks"][
        "phase_one_collection_readiness_locked_empty_and_valid"
    ] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "registered_readiness"
    ]["present_and_valid"] is True
    assert report["match_winner_market"]["phase_one_collection"][
        "registered_readiness"
    ]["locked_empty_collection_state"]["plans"] == 0
    assert report["match_winner_market"]["phase_one_collection"]["inventory"][
        "unvalidated_event_bundle_files"
    ] == 0
    assert report["match_winner_market"]["phase_one_collection"][
        "opening_authority"
    ] is False
    assert report["match_winner_market"]["checks"][
        "phase_one_evaluation_readiness_locked_empty_and_valid"
    ] is True
    assert report["match_winner_market"]["phase_one_evaluation"][
        "registered_readiness"
    ]["present_and_valid"] is True
    assert report["match_winner_market"]["phase_one_evaluation"][
        "registered_readiness"
    ]["locked_empty_state"]["outcomes_accessed"] is False
    assert report["match_winner_market"]["phase_one_evaluation"][
        "independent_registry"
    ]["present_and_valid"] is False
    assert report["match_winner_market"]["checks"][
        "post_pass_probability_pipeline_implementation_frozen_pre_boundary"
    ] is True
    assert report["match_winner_market"]["post_pass_probability_pipeline"][
        "registered_readiness"
    ]["present_and_valid"] is True
    assert report["match_winner_market"]["post_pass_probability_pipeline"][
        "registered_readiness"
    ]["locked_empty_state"]["outcomes_accessed"] is False
    assert report["match_winner_market"]["post_pass_probability_pipeline"][
        "independent_registry"
    ]["present_and_valid"] is False
    assert report["match_winner_market"]["checks"][
        "phase_one_models_independently_passed"
    ] is False
    assert report["match_winner_market"]["checks"][
        "phase_two_not_started_before_phase_one"
    ] is True
    assert report["match_winner_market"]["checks"][
        "quote_builder_time_not_misrepresented_as_transport_time"
    ] is True
    assert report["match_winner_market"]["checks"][
        "public_bookmaker_terms_snapshot_locked_and_valid"
    ] is True
    assert report["match_winner_market"]["checks"][
        "public_bookmaker_terms_snapshot_honestly_incomplete"
    ] is True
    assert report["match_winner_market"]["checks"][
        "source_specific_quote_adapter_candidate_locked_and_valid"
    ] is True
    assert report["match_winner_market"]["checks"][
        "source_specific_quote_adapter_independently_registered"
    ] is False
    assert report["match_winner_market"]["source_specific_quote_adapter"][
        "candidate"
    ]["present_and_valid"] is True
    assert report["match_winner_market"]["source_specific_quote_adapter"][
        "candidate"
    ]["registration"]["independently_registered"] is False
    assert report["match_winner_market"]["source_specific_quote_adapter"][
        "independent_registry"
    ]["present_and_valid"] is False
    assert report["match_winner_market"]["public_bookmaker_terms_snapshot"][
        "present_and_valid"
    ] is True
    assert report["match_winner_market"]["public_bookmaker_terms_snapshot"][
        "coverage"
    ]["complete_bookmaker_terms_snapshot"] is False
    assert report["match_winner_market"]["bookmaker_terms_authority"][
        "present_and_valid"
    ] is False
    assert report["match_winner_market"]["checks"][
        "event_probability_receipt_and_registry_contract_present"
    ] is True
    assert report["match_winner_market"]["checks"][
        "event_probability_registry_independently_registered"
    ] is False
    assert report["match_winner_market"]["checks"][
        "phase_two_collection_readiness_independently_registered"
    ] is False
    assert report["match_winner_market"]["checks"][
        "phase_two_evaluation_readiness_independently_registered"
    ] is False
    assert report["match_winner_market"]["checks"][
        "phase_two_first_support_met_snapshot_independently_registered"
    ] is False
    assert report["match_winner_market"]["checks"][
        "phase_two_market_evaluation_independently_registered"
    ] is False
    assert report["match_winner_market"]["probability_authorized"] is False
    assert report["match_winner_market"]["expected_value_authorized"] is False
    assert report["match_winner_market"]["betting_authorized"] is False
    assert (
        report["registrations"]["rating_registry"][
            "external_digest_pin_present"
        ]
        is False
    )
    assert (
        "draft_score:independent_authority_record_present" in report["blockers"]
    )
    assert (
        "ratings:warehouse_source_pins_match_current_files"
        not in report["blockers"]
    )
    assert "ratings:v3_future_holdout_support_met" in report["blockers"]
    assert (
        "ratings:semantic_output_contract_trust_root_current_and_valid"
        in report["blockers"]
    )
    assert (
        "ratings:semantic_output_contract_reconciliation_candidate_present_and_valid"
        in report["blockers"]
    )
    assert (
        "ratings:semantic_output_contract_candidate_reference_replay_passed"
        in report["blockers"]
    )
    assert (
        "ratings:semantic_output_contract_exact_prior_tree_recovered"
        in report["blockers"]
    )
    assert (
        "ratings:semantic_output_contract_reconciliation_independently_reviewed"
        in report["blockers"]
    )
    assert "live_totals:latest_lcs_patch_holdout_sufficient" in report["blockers"]
    assert (
        "match_winner_market:phase_one_models_independently_passed"
        in report["blockers"]
    )
    assert (
        "match_winner_market:event_probability_registry_independently_registered"
        in report["blockers"]
    )
    assert (
        "match_winner_market:source_specific_quote_adapter_independently_registered"
        in report["blockers"]
    )
    assert (
        "match_winner_market:source_specific_quote_adapter_candidate_locked_and_valid"
        not in report["blockers"]
    )


def test_legacy_warehouse_drift_blocks_again_if_prospective_snapshot_fails(
    monkeypatch,
) -> None:
    monkeypatch.setattr(readiness, "inspect_l2_readiness", draft_blocked)
    monkeypatch.setattr(
        readiness,
        "validate_registered_source_snapshot_v2",
        lambda **_kwargs: (_ for _ in ()).throw(OSError("snapshot unavailable")),
    )
    report = readiness.inspect_private_decision_readiness(
        ".", as_of=AS_OF, environment={}
    )
    assert report["ratings"]["checks"][
        "prospective_source_snapshot_replaces_mutable_warehouse_dependency"
    ] is False
    # The warehouse pins now match the current files (L4 regeneration, 2026-08-08);
    # the remaining ratings blockers are the missing snapshot + holdout/review gates.
    assert report["ratings"]["checks"]["warehouse_source_pins_match_current_files"] is True
    assert not any(
        blocker.startswith("ratings:warehouse_source_pins_match_current_files")
        for blocker in report["blockers"]
    )
def test_missing_package_is_reported_without_creating_authority(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(readiness, "inspect_l2_readiness", draft_blocked)
    report = readiness.inspect_private_decision_readiness(
        tmp_path, as_of=AS_OF, environment={}
    )
    assert report["ratings"]["checks"]["player_artifact_present_and_valid"] is False
    assert report["ratings"]["checks"]["v2_protocol_lock_present_and_valid"] is False
    assert report["live_totals"]["checks"]["artifact_present_and_valid"] is False
    assert report["match_winner_market"]["checks"][
        "future_market_protocol_locked_and_valid"
    ] is False
    assert report["betting_ready"] is False
    assert all(
        registration["validated"] is False
        for registration in report["registrations"].values()
    )


def test_match_winner_authority_check_requires_semantic_validation(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(readiness, "inspect_l2_readiness", draft_blocked)
    authority_path = tmp_path / readiness.SEMANTIC_MATCH_WINNER_AUTHORITY_LOCATOR
    authority_path.parent.mkdir(parents=True)
    authority_path.write_text("{}\n")
    monkeypatch.setattr(
        readiness,
        "load_active_semantic_market_authority_v1",
        lambda **_kwargs: {
            "receipt": {"authority_id": "semantic-authority-1"},
            "receipt_raw_sha256": "a" * 64,
            "private_probability_generation_authorized": True,
            "private_decision_support_authorized": True,
            "transaction_authorized": False,
            "stake_authorized": False,
        },
    )
    report = readiness.inspect_private_decision_readiness(
        tmp_path,
        as_of=AS_OF,
        environment={readiness.SEMANTIC_MATCH_WINNER_AUTHORITY_ENV: "a" * 64},
    )
    market = report["match_winner_market"]
    assert market["checks"][
        "match_winner_market_authority_independently_registered"
    ] is True
    assert market["semantic_market_authority"]["present_and_valid"] is True
    assert market["semantic_market_authority"]["transaction_authorized"] is False
    assert market["semantic_market_authority"]["stake_authorized"] is False


def test_rating_support_is_read_from_live_validated_ledger_not_empty_protocol(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(readiness, "inspect_l2_readiness", draft_blocked)
    ledger_path = tmp_path / readiness.V3_PREDICTION_LEDGER_LOCATOR
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text('{"placeholder": true}\n')
    monkeypatch.setattr(
        readiness,
        "validate_v3_prediction_ledger",
        lambda _payload, root: {
            "status": "SUPPORT_MET_OUTCOMES_UNOPENED",
            "artifact_sha256": "a" * 64,
            "entries": [{"event_id": "future-event-1"}],
            "metadata_support": {"overall_series": 100},
            "outcomes_present": False,
            "outcomes_accessed": False,
        },
    )
    report = readiness.inspect_private_decision_readiness(
        tmp_path,
        as_of=AS_OF,
        environment={},
    )
    checks = report["ratings"]["checks"]
    assert checks["v3_future_prediction_ledger_present_and_valid"] is True
    assert checks["v3_future_holdout_support_met"] is True
    assert checks["v3_pre_event_prediction_ledger_has_eligible_entries"] is True
    assert report["ratings"]["artifacts"]["v3_future_prediction_ledger"][
        "entries"
    ] == 1


def test_joint_phase_one_pass_is_rating_evidence_but_not_rating_authority(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(readiness, "inspect_l2_readiness", draft_blocked)
    registry_path = tmp_path / readiness.PHASE_ONE_EVALUATION_REGISTRY_LOCATOR
    result_locator = (
        "data/lol/v2/evaluation/match-winner-market-v1/phase-one/"
        "results-v1/result.json"
    )
    result_path = tmp_path / result_locator
    registry_path.parent.mkdir(parents=True)
    result_path.parent.mkdir(parents=True)
    registry_path.write_text(
        '{"result_binding":{"result_locator":"' + result_locator + '"}}\n'
    )
    result_path.write_text('{"placeholder":true}\n')
    monkeypatch.setattr(readiness, "expected_result_binding", lambda **_kwargs: {})
    monkeypatch.setattr(
        readiness,
        "load_pinned_evaluation_registry",
        lambda **_kwargs: {
            "phase_one_evaluation_independently_registered": True,
            "phase_one_models_independently_passed": True,
            "phase_two_opening_authorized": False,
            "probability_authorized": False,
            "betting_authorized": False,
        },
    )
    metric = {"upper_95": -0.001}
    strata = {
        name: {"organization": {"log_loss": metric, "brier": metric}}
        for name in ("overall", "league:LCS", "roster_change")
    }
    monkeypatch.setattr(
        readiness.phase_one_model_evaluation,
        "validate_phase_one_evaluation_result",
        lambda _payload: {
            "result_state": "PHASE_ONE_MODELS_PASSED_PENDING_INDEPENDENT_REGISTRATION",
            "artifact_sha256": "a" * 64,
            "phase_one_models_passed": True,
            "ratings_evaluation": {
                "comparators": ["organization"],
                "metrics_by_stratum": strata,
                "reliability": {"complete": True},
                "reliability_gate_passed": True,
                "passed": True,
            },
            "draft_evaluation": {"passed": True},
        },
    )
    report = readiness.inspect_private_decision_readiness(
        tmp_path,
        as_of=AS_OF,
        environment={readiness.PHASE_ONE_EVALUATION_REGISTRY_ENV: "b" * 64},
    )
    checks = report["ratings"]["checks"]
    assert checks[
        "v3_joint_future_evaluation_independently_registered_and_passed"
    ] is True
    assert checks["player_final_holdout_available"] is True
    assert checks["overall_validation_uncertainty_gate_passed"] is True
    assert checks["lcs_validation_uncertainty_gate_passed"] is True
    assert checks["player_rating_not_development_only"] is False
    assert checks["team_rating_available"] is False


def test_rating_readiness_requires_semantic_deployment_authority(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(readiness, "inspect_l2_readiness", draft_blocked)
    monkeypatch.setattr(
        readiness,
        "load_active_semantic_rating_authority_v1",
        lambda **_kwargs: {
            "receipt": {"authority_id": "semantic-rating-authority-1"},
            "receipt_raw_sha256": "c" * 64,
            "private_player_rating_authorized": True,
            "private_team_rating_authorized": True,
            "match_probability_authorized": False,
            "betting_authorized": False,
        },
    )
    report = readiness.inspect_private_decision_readiness(
        tmp_path,
        as_of=AS_OF,
        environment={readiness.SEMANTIC_RATING_AUTHORITY_ENV: "c" * 64},
    )
    ratings_report = report["ratings"]
    checks = ratings_report["checks"]
    assert checks["semantic_rating_deployment_authority_active"] is True
    assert checks["player_rating_not_development_only"] is True
    assert checks["team_rating_available"] is True
    assert checks["team_pre_event_exact_roster_aggregation_available"] is False
    assert ratings_report["rating_probability_authorized"] is False
    assert ratings_report["artifacts"]["semantic_rating_authority"][
        "betting_authorized"
    ] is False
