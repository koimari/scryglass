"""Current-state audit for the independent L2 Draft Score validation gate.

This module reports what is present and what is missing.  It never turns a
development result into authority and never selects a model.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v3 import (
    CaptureReadinessRegistryV3Error,
)

from .adaptive_temporal_diagnostic import (
    AdaptiveTemporalDiagnosticError,
    validate_adaptive_temporal_diagnostic,
)
from .candidate_registry_v3 import validate_candidate_registry_v3
from .capture_readiness_registry_v1 import (
    DraftCaptureReadinessRegistryError,
    REGISTERED_CAPTURE_LOCATOR as FUTURE_CAPTURE_LOCATOR,
    validate_registered_capture_readiness_v1,
)
from .development_evaluation_v3 import evaluate
from .future_prediction_ledger import (
    DEFAULT_LEDGER as FUTURE_LEDGER_LOCATOR,
    DraftPredictionLedgerError,
    validate_prediction_ledger,
)
from .future_protocol_registry_v1 import (
    DraftFutureProtocolRegistryError,
    REGISTERED_PROTOCOL_LOCATOR as FUTURE_PROTOCOL_LOCATOR,
    validate_registered_future_protocol_v1,
)
from .grid_source_readiness_registry_v1 import (
    GridSourceReadinessRegistryError,
    REGISTERED_GRID_SOURCE_LOCATOR,
    validate_registered_grid_source_readiness_v1,
)
from .l2_authority import (
    L2AuthorityRecordError,
    authority_record_payload_sha256,
    load_l2_authority_record,
    validate_l2_authority_record,
)
from .participant_dependence_diagnostic_v1 import (
    ParticipantDependenceDiagnosticError,
    validate_participant_dependence_diagnostic_v1,
)
SCHEMA_VERSION = "scryglass:draft-terminal-l2-readiness:v10"
_ARTIFACT_LOCATOR = "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v3.json"
_REGISTRY_LOCATOR = "data/lol/v2/models/draft-terminal/draft-terminal-candidate-registry-v3.json"
_EVALUATION_LOCATOR = "data/lol/v2/models/draft-terminal/development-evaluation-summary-v3.json"
_CONTRACT_LOCATOR = "data/lol/v2/models/draft-terminal/draft-terminal-l2-evaluation-contract.json"
_AUTHORITY_LOCATOR = "data/lol/v2/models/draft-terminal/draft-terminal-l2-authority-record.json"
_G1_PROCUREMENT_LOCATOR = "data/lol/v2/models/draft-terminal/g1-grid-procurement-attempt.json"
_ADAPTIVE_TEMPORAL_DIAGNOSTIC_LOCATOR = (
    "data/lol/v2/models/draft-terminal/adaptive-temporal-diagnostic-v1.json"
)
_PARTICIPANT_DEPENDENCE_DIAGNOSTIC_LOCATOR = (
    "data/lol/v2/models/draft-terminal/participant-dependence-diagnostic-v1.json"
)
SEMANTIC_DRAFT_AUTHORITY_LOCATOR = Path(
    "data/lol/private_draft_authority/semantic-terminal-draft-authority-v1.json"
)
SEMANTIC_DRAFT_AUTHORITY_ENV = (
    "SCRYGLASS_SEMANTIC_TERMINAL_DRAFT_AUTHORITY_SHA256"
)
_ALWAYS_DIAGNOSTIC_CHECKS = frozenset(
    {
        "participant_clusters_available",
        "roster_change_holdout_available",
    }
)
_PROSPECTIVE_SUPERSEDED_CHECKS = frozenset(
    {
        "candidate_registry_independent_l2_authority",
        "candidate_registry_production_eligible",
        "evaluation_not_development_only",
        "evaluation_source_time_replayed",
        "future_patch_holdout_available",
        "international_holdout_available",
        "reliability_artifact_present",
        "independent_authority_record_present",
        "selected_candidate_all_nested_outer_folds_nonharmful",
        "future_prediction_ledger_present_and_valid",
        "future_holdout_support_met",
    }
)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _read(root: Path, locator: str) -> tuple[bytes, dict[str, Any]]:
    raw = (root / locator).read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError(f"{locator} must contain a JSON object")
    return raw, payload


def _resolve_blockers(
    checks: Mapping[str, bool], *, prospective_supersession_active: bool
) -> tuple[list[str], list[str]]:
    """Keep historical failures visible while allowing stronger future proof.

    Supersession is available only after the independently registered future
    evaluation and the exact-model semantic deployment authority have both
    replayed successfully. It never changes a historical check from false to
    true; it only classifies the listed legacy checks as non-blocking for the
    narrower private terminal-Draft component.
    """

    superseded = (
        _PROSPECTIVE_SUPERSEDED_CHECKS
        if prospective_supersession_active
        else frozenset()
    )
    blockers = sorted(
        name
        for name, passed in checks.items()
        if not passed
        and name not in _ALWAYS_DIAGNOSTIC_CHECKS
        and name not in superseded
    )
    return blockers, sorted(name for name in superseded if not checks.get(name))


def inspect_l2_readiness(
    root: Path | str = Path("."),
    *,
    environment: Mapping[str, str] | None = None,
    as_of: datetime | None = None,
) -> dict[str, Any]:
    """Return a deterministic, non-authorizing audit of the current package."""

    repo_root = Path(root)
    env = environment if environment is not None else os.environ
    observed = as_of or datetime.now(timezone.utc)
    if observed.tzinfo is None:
        raise ValueError("as_of must include a timezone")
    observed = observed.astimezone(timezone.utc)
    registry_raw, registry = _read(repo_root, _REGISTRY_LOCATOR)
    registry = validate_candidate_registry_v3(registry, root=repo_root)
    selected_candidate = registry.get("selected_candidate", {})
    artifact_locator = selected_candidate.get("artifact_locator", _ARTIFACT_LOCATOR)
    artifact_raw, artifact = _read(repo_root, artifact_locator)
    evaluation_raw, evaluation = _read(repo_root, _EVALUATION_LOCATOR)
    contract_raw, contract = _read(repo_root, _CONTRACT_LOCATOR)
    runner_report = evaluate(repo_root)
    runner_output_sha256 = _sha256((json.dumps(runner_report, sort_keys=True, indent=2) + "\n").encode())
    artifact_sha256 = _sha256(artifact_raw)
    registry_sha256 = _sha256(registry_raw)
    evaluation_sha256 = _sha256(evaluation_raw)
    contract_sha256 = _sha256(contract_raw)
    authority_record: dict[str, Any] | None = None
    authority_record_sha256: str | None = None
    authority_record_error: str | None = None
    authority_path = repo_root / _AUTHORITY_LOCATOR
    if authority_path.exists():
        try:
            authority_raw = authority_path.read_bytes()
            authority_record = load_l2_authority_record(authority_raw)
            validate_l2_authority_record(
                authority_record,
                expected_bindings={
                    "model_artifact_sha256": artifact_sha256,
                    "candidate_registry_sha256": registry_sha256,
                    "development_evaluation_sha256": evaluation_sha256,
                    "l2_contract_sha256": contract_sha256,
                },
            )
            authority_record_sha256 = authority_record_payload_sha256(authority_raw)
        except (OSError, L2AuthorityRecordError) as exc:
            authority_record_error = str(exc)
    procurement_attempt: dict[str, Any] | None = None
    procurement_attempt_sha256: str | None = None
    procurement_attempt_error: str | None = None
    procurement_path = repo_root / _G1_PROCUREMENT_LOCATOR
    if procurement_path.exists():
        try:
            procurement_raw, procurement_payload = _read(repo_root, _G1_PROCUREMENT_LOCATOR)
            if procurement_payload.get("schema_version") != "scryglass:draft-terminal-g1-grid-procurement-attempt:v1":
                raise ValueError("G1 procurement attempt schema_version is not supported")
            if procurement_payload.get("status") != "externally_blocked":
                raise ValueError("G1 procurement attempt is not marked externally_blocked")
            action = procurement_payload.get("action")
            if not isinstance(action, Mapping) or action.get("g1_status") != "externally_blocked":
                raise ValueError("G1 procurement attempt does not preserve the blocked action")
            blockers = procurement_payload.get("blockers")
            if not isinstance(blockers, list) or not blockers or any(not isinstance(item, str) for item in blockers):
                raise ValueError("G1 procurement attempt blockers are missing")
            procurement_attempt = {
                "series_id": procurement_payload.get("series_id"),
                "competition": procurement_payload.get("competition"),
                "scheduled_date": procurement_payload.get("scheduled_date"),
                "source_event_payload_sha256": procurement_payload.get("source_event_payload_sha256"),
                "blockers": list(blockers),
                "status": procurement_payload.get("status"),
            }
            procurement_attempt_sha256 = _sha256(procurement_raw)
        except (OSError, ValueError, TypeError) as exc:
            procurement_attempt_error = str(exc)
    adaptive_temporal_diagnostic: dict[str, Any] | None = None
    adaptive_temporal_diagnostic_sha256: str | None = None
    adaptive_temporal_diagnostic_error: str | None = None
    adaptive_temporal_path = repo_root / _ADAPTIVE_TEMPORAL_DIAGNOSTIC_LOCATOR
    if adaptive_temporal_path.exists():
        try:
            adaptive_temporal_raw, adaptive_temporal_payload = _read(
                repo_root, _ADAPTIVE_TEMPORAL_DIAGNOSTIC_LOCATOR
            )
            adaptive_temporal_diagnostic = validate_adaptive_temporal_diagnostic(
                adaptive_temporal_payload,
                root=repo_root,
            )
            adaptive_temporal_diagnostic_sha256 = _sha256(adaptive_temporal_raw)
        except (OSError, ValueError, AdaptiveTemporalDiagnosticError) as exc:
            adaptive_temporal_diagnostic_error = str(exc)
    participant_dependence_diagnostic: dict[str, Any] | None = None
    participant_dependence_diagnostic_sha256: str | None = None
    participant_dependence_diagnostic_error: str | None = None
    participant_dependence_path = (
        repo_root / _PARTICIPANT_DEPENDENCE_DIAGNOSTIC_LOCATOR
    )
    if participant_dependence_path.exists():
        try:
            participant_dependence_raw, participant_dependence_payload = _read(
                repo_root, _PARTICIPANT_DEPENDENCE_DIAGNOSTIC_LOCATOR
            )
            participant_dependence_diagnostic = (
                validate_participant_dependence_diagnostic_v1(
                    participant_dependence_payload,
                    root=repo_root,
                )
            )
            participant_dependence_diagnostic_sha256 = _sha256(
                participant_dependence_raw
            )
        except (
            OSError,
            ValueError,
            ParticipantDependenceDiagnosticError,
        ) as exc:
            participant_dependence_diagnostic_error = str(exc)
    participant_dependence_method: dict[str, Any] | None = None
    participant_dependence_method_error: str | None = None
    participant_dependence_method_locator: Path | None = None
    participant_dependence_method_raw_sha256: str | None = None
    participant_dependence_method_artifact_sha256: str | None = None
    try:
        # Import at audit time.  The market evaluator imports the Draft package
        # to validate ledger receipts, so a module-level import here would form
        # a circular dependency during evaluator startup.
        from lol_kills.v2.market.phase_one_evaluation_readiness_registry_v1 import (
            REGISTERED_READINESS_ARTIFACT_SHA256,
            REGISTERED_READINESS_LOCATOR,
            REGISTERED_READINESS_RAW_SHA256,
            validate_registered_phase_one_evaluation_readiness_v1,
        )

        participant_method_readiness = (
            validate_registered_phase_one_evaluation_readiness_v1(root=repo_root)
        )
        participant_dependence_method_locator = REGISTERED_READINESS_LOCATOR
        participant_dependence_method_raw_sha256 = REGISTERED_READINESS_RAW_SHA256
        participant_dependence_method_artifact_sha256 = (
            REGISTERED_READINESS_ARTIFACT_SHA256
        )
        participant_dependence_method = dict(
            participant_method_readiness["evaluation_contract"][
                "entity_network_dependence_sensitivity"
            ]
        )
    except (
        OSError,
        KeyError,
        ImportError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        participant_dependence_method_error = str(exc)
    future_protocol: dict[str, Any] | None = None
    future_protocol_sha256: str | None = None
    future_protocol_error: str | None = None
    try:
        future_protocol = validate_registered_future_protocol_v1(root=repo_root)
        future_protocol_sha256 = _sha256(
            (repo_root / FUTURE_PROTOCOL_LOCATOR).read_bytes()
        )
    except (OSError, ValueError, DraftFutureProtocolRegistryError) as exc:
        future_protocol_error = str(exc)
    future_capture: dict[str, Any] | None = None
    future_capture_sha256: str | None = None
    future_capture_error: str | None = None
    try:
        future_capture = validate_registered_capture_readiness_v1(root=repo_root)
        future_capture_sha256 = _sha256(
            (repo_root / FUTURE_CAPTURE_LOCATOR).read_bytes()
        )
    except (
        OSError,
        ValueError,
        DraftCaptureReadinessRegistryError,
        CaptureReadinessRegistryV3Error,
    ) as exc:
        future_capture_error = str(exc)
    grid_source_readiness: dict[str, Any] | None = None
    grid_source_readiness_sha256: str | None = None
    grid_source_readiness_error: str | None = None
    try:
        grid_source_readiness = validate_registered_grid_source_readiness_v1(
            root=repo_root
        )
        grid_source_readiness_sha256 = _sha256(
            (repo_root / REGISTERED_GRID_SOURCE_LOCATOR).read_bytes()
        )
    except (OSError, ValueError, GridSourceReadinessRegistryError) as exc:
        grid_source_readiness_error = str(exc)
    future_ledger: dict[str, Any] | None = None
    future_ledger_sha256: str | None = None
    future_ledger_error: str | None = None
    future_ledger_path = repo_root / FUTURE_LEDGER_LOCATOR
    if future_ledger_path.exists():
        try:
            future_ledger_raw, future_ledger_payload = _read(
                repo_root, FUTURE_LEDGER_LOCATOR.as_posix()
            )
            future_ledger = validate_prediction_ledger(
                future_ledger_payload,
                root=repo_root,
            )
            future_ledger_sha256 = _sha256(future_ledger_raw)
        except (OSError, ValueError, DraftPredictionLedgerError) as exc:
            future_ledger_error = str(exc)
    phase_one_evaluation: dict[str, Any] | None = None
    phase_one_evaluation_error: str | None = None
    phase_one_registry_digest: str | None = None
    phase_one_registry_locator: Path | None = None
    try:
        from lol_kills.v2.market.phase_one_evaluation_registry_v1 import (
            EXTERNAL_SHA256_ENV,
            REGISTRY_LOCATOR,
            PhaseOneEvaluationRegistryError,
            expected_result_binding,
            load_pinned_evaluation_registry,
        )

        phase_one_registry_locator = REGISTRY_LOCATOR
        phase_one_registry_digest = env.get(EXTERNAL_SHA256_ENV)
        registry_path = repo_root / REGISTRY_LOCATOR
        if registry_path.is_file() and phase_one_registry_digest:
            _registry_raw, registry_payload = _read(
                repo_root, REGISTRY_LOCATOR.as_posix()
            )
            result_locator = (
                registry_payload.get("result_binding") or {}
            ).get("result_locator")
            if not isinstance(result_locator, str):
                raise PhaseOneEvaluationRegistryError(
                    "phase-one result locator is missing"
                )
            binding = expected_result_binding(
                result_locator=result_locator,
                root=repo_root,
            )
            phase_one_evaluation = load_pinned_evaluation_registry(
                path=registry_path,
                external_sha256=phase_one_registry_digest,
                expected_binding=binding,
            )
    except (OSError, ValueError, RuntimeError) as exc:
        phase_one_evaluation_error = str(exc)
    semantic_draft_authority: dict[str, Any] | None = None
    semantic_draft_authority_error: str | None = None
    try:
        from .semantic_draft_authority_v1 import (
            load_active_semantic_draft_authority_v1,
        )

        semantic_draft_authority = load_active_semantic_draft_authority_v1(
            root=repo_root,
            environment=env,
            as_of=observed,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        semantic_draft_authority_error = str(exc)
    candidate = selected_candidate
    selected_development_candidate = evaluation.get(
        "development_candidate_for_future_freeze"
    )
    selected_outer_tests = evaluation.get("fold_locked_selected_test", ())
    checks = {
        "artifact_bound_to_registry": candidate.get("artifact_raw_sha256") == artifact_sha256,
        "evaluation_summary_binds_runner_output": evaluation.get("run_output_sha256") == runner_output_sha256,
        "l2_contract_policy_frozen": (
            contract.get("schema_version") == "scryglass:draft-terminal-l2-evaluation-contract:v1"
            and contract.get("status") == "pending_independent_authority"
            and contract.get("production_eligible") is False
            and contract.get("public_probability_authorized") is False
            and set(contract.get("promotion_gates", ()))
            == {
                "independent_l2_authority_record",
                "sealed_outer_temporal_evaluation",
                "approved_calibration_transform",
                "protocol_specific_action_order_validation",
                "reliability_record_with_baseline_and_dependence_support",
                "python_artifact_typescript_replay_parity",
                "source_rights_and_preevent_availability",
                "non_betting_descriptive_claim_ceiling",
                "grid_source_cohort_promotion_gate",
            }
            and contract.get("grid_promotion_gate", {}).get("predeclared_plan", {}).get("required") is True
            and contract.get("grid_promotion_gate", {}).get("predeclared_plan", {}).get("hash_bound") is True
            and contract.get("grid_promotion_gate", {}).get("predeclared_plan", {}).get("must_precede_cohort") is True
            and {
                "exact_source_payload_bytes_and_hash",
                "heldout_results_hash_bound",
                "replay_data_hash_bound_to_verified_records",
                "replay_model_hash_bound_to_exact_model_bytes",
            }.issubset(set(contract.get("grid_promotion_gate", {}).get("required_before_promotion", ())))
        ),
        "contextual_standardization_policy_frozen": (
            contract.get("contextual_policy", {}).get("exact_roster_required") is True
            and contract.get("contextual_policy", {}).get("player_champion_response_required") is True
            and contract.get("contextual_policy", {}).get("team_policy_response_required") is True
            and contract.get("contextual_policy", {}).get("baseline_team_rating_difference_in_served_draft_score") == 0.0
            and contract.get("contextual_policy", {}).get("league_rating_difference_in_served_draft_score") == 0.0
            and contract.get("contextual_policy", {}).get("in_game_side_advantage_in_served_draft_score") == 0.0
            and contract.get("contextual_policy", {}).get("neutral_fallback_when_context_is_missing") is False
        ),
        "neutral_standardization_policy_frozen": (
            contract.get("neutral_policy", {}).get("exact_roster_required") is False
            and contract.get("neutral_policy", {}).get("player_champion_response_required") is False
            and contract.get("neutral_policy", {}).get("team_policy_response_required") is False
            and contract.get("neutral_policy", {}).get("neutral_context_required") is True
            and contract.get("neutral_policy", {}).get("g1_contextual_source_required") is False
        ),
        "candidate_registry_independent_l2_authority": (
            registry.get("authority", {}).get("model_validation_authority") is True
        ),
        "candidate_registry_production_eligible": (
            registry.get("claim_ceiling", {}).get("production_probability") is True
        ),
        "evaluation_not_development_only": evaluation.get("status") != "development_only",
        "evaluation_selected_candidate": isinstance(
            selected_development_candidate, Mapping
        ),
        "evaluation_source_time_replayed": evaluation.get("source_snapshot", {}).get("availability_status") == "verified_preevent",
        "participant_clusters_available": (
            evaluation.get("split_policy", {}).get("participant_cluster_status") in {
                "available",
                "team_or_series_available",
            }
            and evaluation.get("split_policy", {}).get("series_grouped") is True
        ),
        "participant_dependence_method_predeclared_and_valid": (
            participant_dependence_method is not None
            and participant_dependence_method_error is None
            and participant_dependence_method.get("method")
            == (
                "shared_series_participant_or_organization_network_hac_"
                "sandwich_for_mean_paired_loss_delta"
            )
            and participant_dependence_method.get("locked_before_future_outcomes")
            is True
            and participant_dependence_method.get(
                "supplements_not_replaces_whole_series_bootstrap"
            )
            is True
            and participant_dependence_method.get(
                "exact_ten_player_and_two_organization_ids_required"
            )
            is True
            and participant_dependence_method.get("draft_required_strata")
            == ["overall"]
            and participant_dependence_method.get(
                "both_metric_upper_95_bounds_must_be_nonpositive"
            )
            is True
        ),
        "participant_dependence_diagnostic_present_and_valid": (
            participant_dependence_diagnostic is not None
            and participant_dependence_diagnostic_error is None
            and (participant_dependence_diagnostic.get("decision") or {}).get(
                "participant_identity_available_for_development_diagnostic"
            )
            is True
            and (participant_dependence_diagnostic.get("decision") or {}).get(
                "participant_dependence_support_verified"
            )
            is False
        ),
        "future_patch_holdout_available": evaluation.get("holdouts", {}).get("future_patch", {}).get("status") == "passed",
        "international_holdout_available": evaluation.get("holdouts", {}).get("international_event_or_meta", {}).get("status") == "passed",
        "roster_change_holdout_available": (
            contract.get("holdouts", {}).get("roster_change") == "not_required_for_neutral"
            or evaluation.get("holdouts", {}).get("roster_change", {}).get("status") == "passed"
        ),
        "reliability_artifact_present": False,
        "independent_authority_record_present": authority_record is not None and authority_record_error is None,
        "adaptive_temporal_diagnostic_present_and_valid": (
            adaptive_temporal_diagnostic is not None
            and adaptive_temporal_diagnostic_error is None
        ),
        "known_harmful_app_draft_family_quarantined": (
            (adaptive_temporal_diagnostic or {})
            .get("decision", {})
            .get("known_app_draft_family_nonharmful")
            is False
            and (adaptive_temporal_diagnostic or {})
            .get("model_scope", {})
            .get("same_as_terminal_m0_candidate")
            is False
        ),
        "selected_candidate_uses_incremental_context_estimand": (
            evaluation.get("estimands", {}).get("selection_target")
            == "incremental_predictive_value_of_context_plus_draft_over_same_input_context_only_model"
            and evaluation.get("estimands", {}).get(
                "neutral_probability_calibration_directly_identified"
            )
            is False
        ),
        "selected_candidate_all_validation_folds_nonharmful": (
            isinstance(selected_development_candidate, Mapping)
            and selected_development_candidate.get(
                "all_validation_folds_nonharmful"
            )
            is True
        ),
        "selected_candidate_all_nested_outer_folds_nonharmful": (
            bool(selected_outer_tests)
            and all(
                isinstance(item, Mapping)
                and item.get(
                    "locked_outer_test_incremental_vs_baseline_only", {}
                ).get("passed")
                is True
                for item in selected_outer_tests
            )
        ),
        "future_protocol_locked_and_valid": (
            future_protocol is not None and future_protocol_error is None
        ),
        "future_capture_readiness_locked_and_valid": (
            future_capture is not None
            and future_capture_error is None
            and (future_capture.get("implementation") or {}).get(
                "ready_for_outcome_free_future_capture"
            )
            is True
            and (future_capture.get("ledger_state") or {}).get("entries") == 0
        ),
        "grid_terminal_draft_source_readiness_locked_and_valid": (
            grid_source_readiness is not None
            and grid_source_readiness_error is None
            and (grid_source_readiness.get("capability_conclusion") or {}).get(
                "terminal_pick_ban_prestart_observed_in_all_archives"
            )
            is True
            and (grid_source_readiness.get("capability_conclusion") or {}).get(
                "prestart_role_assignment_available_from_grid"
            )
            is False
            and (grid_source_readiness.get("capability_conclusion") or {}).get(
                "reviewed_separate_role_assignment_required"
            )
            is True
            and (grid_source_readiness.get("capability_conclusion") or {}).get(
                "prospective_system_receipts_required"
            )
            is True
            and (grid_source_readiness.get("capability_conclusion") or {}).get(
                "retrospective_archives_qualify_future_evidence"
            )
            is False
        ),
        "future_prediction_ledger_present_and_valid": (
            future_ledger is not None and future_ledger_error is None
        ),
        "future_holdout_support_met": (
            future_ledger is not None
            and future_ledger_error is None
            and (future_ledger.get("metadata_support") or {}).get("support_met")
            is True
            and future_ledger.get("status") == "SUPPORT_MET_OUTCOMES_UNOPENED"
        ),
        "future_protocol_independent_review_present": (
            phase_one_evaluation is not None
            and phase_one_evaluation_error is None
            and phase_one_evaluation.get(
                "phase_one_evaluation_independently_registered"
            )
            is True
            and phase_one_evaluation.get("phase_one_models_independently_passed")
            is True
        ),
        "participant_dependence_support_verified": (
            participant_dependence_method is not None
            and participant_dependence_method_error is None
            and phase_one_evaluation is not None
            and phase_one_evaluation_error is None
            and phase_one_evaluation.get(
                "phase_one_evaluation_independently_registered"
            )
            is True
            and phase_one_evaluation.get("phase_one_models_independently_passed")
            is True
        ),
        "semantic_terminal_draft_authority_active": (
            semantic_draft_authority is not None
            and semantic_draft_authority_error is None
            and semantic_draft_authority.get(
                "private_terminal_draft_component_authorized"
            )
            is True
            and semantic_draft_authority.get(
                "private_event_probability_authorized"
            )
            is False
            and semantic_draft_authority.get("public_probability_authorized")
            is False
            and semantic_draft_authority.get("betting_authorized") is False
        ),
    }
    prospective_supersession_active = bool(
        checks["future_protocol_independent_review_present"]
        and checks["participant_dependence_support_verified"]
        and checks["semantic_terminal_draft_authority_active"]
    )
    blockers, superseded_blockers = _resolve_blockers(
        checks,
        prospective_supersession_active=prospective_supersession_active,
    )
    promotion_eligible = not blockers and prospective_supersession_active
    grid_promotion_gate = evaluation.get("grid_promotion_gate")
    if not isinstance(grid_promotion_gate, Mapping):
        grid_promotion_gate = {
            "status": "not_passed",
            "baseline_source": "OE",
            "candidate_source": "GRID",
            "primary_source_for_cohort": "OE",
            "public_reproducibility_benchmark": "OE",
            "reason": "no authorized complete hash-verified GRID Draft Score cohort has passed the gate",
        }
    else:
        grid_promotion_gate = dict(grid_promotion_gate)
    if grid_promotion_gate.get("status") != "passed" and procurement_attempt is not None:
        attempt_blockers = procurement_attempt.get("blockers", [])
        if isinstance(attempt_blockers, list):
            grid_promotion_gate["blockers"] = list(dict.fromkeys([
                *grid_promotion_gate.get("blockers", []),
                *attempt_blockers,
            ]))
            grid_promotion_gate.setdefault(
                "missing_or_invalid_records",
                {
                    "missing": [],
                    "extra": [],
                    "invalid": [{
                        "record_scope": "series",
                        "series_id": procurement_attempt.get("series_id"),
                        "competition": procurement_attempt.get("competition"),
                        "reason_codes": list(attempt_blockers),
                    }],
                },
            )
        grid_promotion_gate["failed_procurement_attempt"] = {
            "series_id": procurement_attempt.get("series_id"),
            "competition": procurement_attempt.get("competition"),
            "scheduled_date": procurement_attempt.get("scheduled_date"),
            "status": procurement_attempt.get("status"),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "blocked" if blockers else "ready_for_independent_decision",
        "promotion_eligible": promotion_eligible,
        "public_probability_authorized": False,
        "claim_ceiling": {
            "development_diagnostic": True,
            "equal_strength_composition_index": True,
            "outcome_calibrated_neutral_probability": False,
            "descriptive_public_probability": False,
            "causal_draft_effect": False,
            "recommendation": False,
            "betting": False,
        },
        "artifacts": {
            "model": {"locator": artifact_locator, "raw_sha256": artifact_sha256, "model_version": artifact.get("model_version")},
            "candidate_registry": {"locator": _REGISTRY_LOCATOR, "raw_sha256": registry_sha256},
            "evaluation_summary": {"locator": _EVALUATION_LOCATOR, "raw_sha256": evaluation_sha256},
            "l2_contract": {"locator": _CONTRACT_LOCATOR, "raw_sha256": contract_sha256},
            "independent_authority_record": {
                "locator": _AUTHORITY_LOCATOR,
                "raw_sha256": authority_record_sha256,
                "present": authority_record is not None,
                "error": authority_record_error,
            },
            "semantic_draft_authority": {
                "locator": SEMANTIC_DRAFT_AUTHORITY_LOCATOR.as_posix(),
                "external_digest_pin_present": bool(
                    env.get(SEMANTIC_DRAFT_AUTHORITY_ENV)
                ),
                "present_and_valid": semantic_draft_authority is not None,
                "error": semantic_draft_authority_error,
                "authority_id": (
                    (semantic_draft_authority or {}).get("receipt") or {}
                ).get("authority_id"),
                "receipt_raw_sha256": (semantic_draft_authority or {}).get(
                    "receipt_raw_sha256"
                ),
                "private_terminal_draft_component_authorized": (
                    semantic_draft_authority or {}
                ).get("private_terminal_draft_component_authorized"),
                "private_event_probability_authorized": (
                    semantic_draft_authority or {}
                ).get("private_event_probability_authorized"),
                "public_probability_authorized": (
                    semantic_draft_authority or {}
                ).get("public_probability_authorized"),
                "betting_authorized": (semantic_draft_authority or {}).get(
                    "betting_authorized"
                ),
            },
            "g1_procurement_attempt": {
                "locator": _G1_PROCUREMENT_LOCATOR,
                "raw_sha256": procurement_attempt_sha256,
                "present": procurement_attempt is not None,
                "error": procurement_attempt_error,
            },
            "adaptive_temporal_diagnostic": {
                "locator": _ADAPTIVE_TEMPORAL_DIAGNOSTIC_LOCATOR,
                "raw_sha256": adaptive_temporal_diagnostic_sha256,
                "present": adaptive_temporal_diagnostic is not None,
                "error": adaptive_temporal_diagnostic_error,
                "declared_artifact_sha256": (
                    (adaptive_temporal_diagnostic or {}).get("artifact_sha256")
                ),
                "result_state": (
                    (adaptive_temporal_diagnostic or {}).get("result_state")
                ),
            },
            "participant_dependence_diagnostic": {
                "locator": _PARTICIPANT_DEPENDENCE_DIAGNOSTIC_LOCATOR,
                "raw_sha256": participant_dependence_diagnostic_sha256,
                "present": participant_dependence_diagnostic is not None,
                "error": participant_dependence_diagnostic_error,
                "declared_artifact_sha256": (
                    (participant_dependence_diagnostic or {}).get(
                        "artifact_sha256"
                    )
                ),
                "result_state": (
                    (participant_dependence_diagnostic or {}).get(
                        "result_state"
                    )
                ),
            },
            "participant_dependence_method": {
                "locator": (
                    participant_dependence_method_locator.as_posix()
                    if participant_dependence_method_locator is not None
                    else None
                ),
                "raw_sha256": participant_dependence_method_raw_sha256,
                "artifact_sha256": participant_dependence_method_artifact_sha256,
                "present": participant_dependence_method is not None,
                "error": participant_dependence_method_error,
            },
            "future_protocol": {
                "locator": str(FUTURE_PROTOCOL_LOCATOR),
                "raw_sha256": future_protocol_sha256,
                "present": future_protocol is not None,
                "error": future_protocol_error,
                "declared_artifact_sha256": (
                    (future_protocol or {}).get("artifact_sha256")
                ),
                "result_state": (future_protocol or {}).get("result_state"),
            },
            "future_capture_readiness": {
                "locator": str(FUTURE_CAPTURE_LOCATOR),
                "raw_sha256": future_capture_sha256,
                "present": future_capture is not None,
                "error": future_capture_error,
                "declared_artifact_sha256": (
                    (future_capture or {}).get("artifact_sha256")
                ),
                "result_state": (future_capture or {}).get("result_state"),
            },
            "grid_source_readiness": {
                "locator": str(REGISTERED_GRID_SOURCE_LOCATOR),
                "raw_sha256": grid_source_readiness_sha256,
                "present": grid_source_readiness is not None,
                "error": grid_source_readiness_error,
                "declared_artifact_sha256": (
                    (grid_source_readiness or {}).get("artifact_sha256")
                ),
                "result_state": (
                    (grid_source_readiness or {}).get("result_state")
                ),
            },
            "future_prediction_ledger": {
                "locator": str(FUTURE_LEDGER_LOCATOR),
                "raw_sha256": future_ledger_sha256,
                "present": future_ledger is not None,
                "error": future_ledger_error,
                "declared_artifact_sha256": (
                    (future_ledger or {}).get("artifact_sha256")
                ),
                "status": (future_ledger or {}).get("status"),
            },
        },
        "fold_policy": evaluation.get("split_policy"),
        "holdouts": evaluation.get("holdouts"),
        "prospective_supersession": {
            "active": prospective_supersession_active,
            "scope": "private_equal_strength_terminal_draft_component_only",
            "superseded_blockers": superseded_blockers,
            "historical_checks_retained_unchanged": True,
            "historical_nested_outer_all_nonharmful": checks[
                "selected_candidate_all_nested_outer_folds_nonharmful"
            ],
            "historical_outer_test_pass_count": sum(
                isinstance(item, Mapping)
                and item.get(
                    "locked_outer_test_incremental_vs_baseline_only", {}
                ).get("passed")
                is True
                for item in selected_outer_tests
            ),
            "historical_outer_test_count": len(selected_outer_tests),
            "future_result_independently_registered": checks[
                "future_protocol_independent_review_present"
            ],
            "participant_dependence_support_verified": checks[
                "participant_dependence_support_verified"
            ],
            "exact_future_evaluated_model_semantically_authorized": checks[
                "semantic_terminal_draft_authority_active"
            ],
            "event_probability_authorized": False,
            "public_probability_authorized": False,
            "betting_authorized": False,
        },
        "g1_contextual_source": {
            "status": "externally_blocked",
            "focused_procurement_attempt": "failed",
            "applies_to": "contextual_only",
            "blocks_neutral": False,
            "authority_available": False,
            "required_fields": [
                "preevent_payload",
                "source_and_retrieval_timestamps",
                "exact_starters_and_roles",
                "rights_review",
                "verifiable_payload_sha256",
            ],
            "procurement_attempt": procurement_attempt,
            "procurement_attempt_error": procurement_attempt_error,
        },
        "grid_promotion_gate": grid_promotion_gate,
        "adaptive_temporal_diagnostic": {
            "status": (
                "valid"
                if adaptive_temporal_diagnostic is not None
                and adaptive_temporal_diagnostic_error is None
                else "invalid_or_missing"
            ),
            "result_state": (
                (adaptive_temporal_diagnostic or {}).get("result_state")
            ),
            "model_scope": (
                (adaptive_temporal_diagnostic or {}).get("model_scope")
            ),
            "population": (
                (adaptive_temporal_diagnostic or {}).get("population")
            ),
            "metrics": (
                (adaptive_temporal_diagnostic or {}).get("metrics")
            ),
            "decision": (
                (adaptive_temporal_diagnostic or {}).get("decision")
            ),
            "claim_ceiling": (
                (adaptive_temporal_diagnostic or {}).get("claim_ceiling")
            ),
            "error": adaptive_temporal_diagnostic_error,
        },
        "participant_dependence_diagnostic": {
            "status": (
                "valid"
                if participant_dependence_diagnostic is not None
                and participant_dependence_diagnostic_error is None
                else "invalid_or_missing"
            ),
            "result_state": (
                (participant_dependence_diagnostic or {}).get("result_state")
            ),
            "population": (
                (participant_dependence_diagnostic or {}).get("population")
            ),
            "decision": (
                (participant_dependence_diagnostic or {}).get("decision")
            ),
            "claim_ceiling": (
                (participant_dependence_diagnostic or {}).get("claim_ceiling")
            ),
            "error": participant_dependence_diagnostic_error,
        },
        "participant_dependence_method": {
            "status": (
                "predeclared_pending_independent_future_evaluation"
                if participant_dependence_method is not None
                and participant_dependence_method_error is None
                else "invalid_or_missing"
            ),
            "contract": participant_dependence_method,
            "atomic_component_split_required": False,
            "independent_future_support_verified": checks[
                "participant_dependence_support_verified"
            ],
            "error": participant_dependence_method_error,
        },
        "future_protocol": {
            "status": (
                "valid"
                if future_protocol is not None and future_protocol_error is None
                else "invalid_or_missing"
            ),
            "result_state": (future_protocol or {}).get("result_state"),
            "locked_candidate": (future_protocol or {}).get("locked_candidate"),
            "estimands": (future_protocol or {}).get("estimands"),
            "future_holdout": (future_protocol or {}).get("future_holdout"),
            "capture_state": (future_protocol or {}).get("capture_state"),
            "opening_authority": (future_protocol or {}).get(
                "opening_authority"
            ),
            "claim_ceiling": (future_protocol or {}).get("claim_ceiling"),
            "error": future_protocol_error,
        },
        "future_capture_readiness": {
            "status": (
                "valid"
                if future_capture is not None and future_capture_error is None
                else "invalid_or_missing"
            ),
            "result_state": (future_capture or {}).get("result_state"),
            "locked_at_utc": (future_capture or {}).get("locked_at_utc"),
            "capture_contract": (future_capture or {}).get("capture_contract"),
            "ledger_state_at_lock": (future_capture or {}).get("ledger_state"),
            "implementation": (future_capture or {}).get("implementation"),
            "claim_ceiling": (future_capture or {}).get("claim_ceiling"),
            "error": future_capture_error,
        },
        "grid_source_readiness": {
            "status": (
                "valid"
                if grid_source_readiness is not None
                and grid_source_readiness_error is None
                else "invalid_or_missing"
            ),
            "result_state": (grid_source_readiness or {}).get("result_state"),
            "locked_at_utc": (grid_source_readiness or {}).get("locked_at_utc"),
            "capability_conclusion": (
                (grid_source_readiness or {}).get("capability_conclusion")
            ),
            "prospective_capture_contract": (
                (grid_source_readiness or {}).get("prospective_capture_contract")
            ),
            "claim_ceiling": (grid_source_readiness or {}).get("claim_ceiling"),
            "error": grid_source_readiness_error,
        },
        "future_prediction_ledger": {
            "status": (
                "valid"
                if future_ledger is not None and future_ledger_error is None
                else "missing"
                if not future_ledger_path.exists()
                else "invalid"
            ),
            "ledger_status": (future_ledger or {}).get("status"),
            "metadata_support": (future_ledger or {}).get("metadata_support"),
            "outcomes_present": (future_ledger or {}).get("outcomes_present"),
            "outcomes_accessed": (future_ledger or {}).get("outcomes_accessed"),
            "error": future_ledger_error,
        },
        "joint_phase_one_evaluation": {
            "status": (
                (phase_one_evaluation or {}).get("status")
                if phase_one_evaluation is not None
                else "missing_or_unpinned"
            ),
            "registry_locator": (
                phase_one_registry_locator.as_posix()
                if phase_one_registry_locator is not None
                else None
            ),
            "external_digest_pin_present": bool(phase_one_registry_digest),
            "independently_registered": (
                phase_one_evaluation or {}
            ).get("phase_one_evaluation_independently_registered"),
            "models_independently_passed": (
                phase_one_evaluation or {}
            ).get("phase_one_models_independently_passed"),
            "probability_authorized": (
                phase_one_evaluation or {}
            ).get("probability_authorized"),
            "betting_authorized": (
                phase_one_evaluation or {}
            ).get("betting_authorized"),
            "error": phase_one_evaluation_error,
        },
        "checks": checks,
        "blockers": blockers,
        "independent_authority_record_error": authority_record_error,
        "required_next_authority": None if promotion_eligible else {
            "kind": (
                "independently_registered_future_evaluation_then_exact_model_"
                "semantic_terminal_draft_authority"
            ),
            "must_bind": [
                "joint_future_snapshot_raw_and_artifact_sha256",
                "sealed_outcome_cohort_raw_and_artifact_sha256",
                "one_time_opening_authority_and_run_id",
                "independently_registered_phase_one_result_sha256",
                "draft_primary_subgroup_reliability_and_dependence_gates",
                "future_protocol_sha256",
                "future_capture_readiness_sha256",
                "typescript_replay_parity_sha256",
                "exact_evaluated_model_locator_raw_sha256_and_version",
                "two_independent_deployment_and_runtime_reviews",
            ],
            "must_be_independent": True,
            "historical_development_artifacts_remain_unchanged": True,
            "event_probability_authorized": False,
            "public_probability_authorized": False,
            "betting_authorized": False,
        },
    }


__all__ = ["SCHEMA_VERSION", "inspect_l2_readiness"]
