from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import phase_one_evaluation_v1 as evaluation
from lol_kills.v2.market import phase_one_evaluation_readiness_v1 as readiness
from lol_kills.v2.market import phase_one_evaluation_readiness_registry_v1 as readiness_registry
from lol_kills.v2.market import phase_one_evaluation_registry_v1 as result_registry
from lol_kills.v2.market import phase_one_opening_v1 as opening


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    leagues = ("LCS", "LEC", "LCK", "LPL", "MSI")
    for series_index in range(100):
        league = leagues[series_index % len(leagues)]
        for game_number, blue_win in ((1, 0), (2, 1)):
            candidate = 0.25 if blue_win == 0 else 0.75
            comparator = 0.45 if blue_win == 0 else 0.55
            rows.append(
                {
                    "event_id": f"event-{series_index}-{game_number}",
                    "series_id": f"series-{series_index}",
                    "game_number": game_number,
                    "league": league,
                    "patch": f"26.{15 + series_index % 3}",
                    "roster_change": series_index < 30,
                    "sparse_or_new_champion": series_index < 50,
                    "participant_ids": tuple(
                        f"player-{series_index}-{slot}" for slot in range(10)
                    ),
                    "organization_ids": (
                        f"organization-{series_index}-blue",
                        f"organization-{series_index}-red",
                    ),
                    "blue_win": blue_win,
                    "rating_candidate": candidate,
                    evaluation.RATINGS_COMPARATORS[0]: comparator,
                    evaluation.RATINGS_COMPARATORS[1]: comparator,
                    "ratings_only": comparator,
                    "ratings_plus_draft": candidate,
                }
            )
    return rows


def test_cluster_bootstrap_keeps_whole_series_and_finds_clear_improvement() -> None:
    report = evaluation._delta_interval(
        _rows(),
        candidate_key="ratings_plus_draft",
        comparator_key="ratings_only",
        metric="log_loss",
        replicates=200,
        seed=7,
    )
    assert report["maps"] == 200
    assert report["series"] == 100
    assert report["point_delta"] < 0
    assert report["upper_95"] < 0


def test_locked_model_reports_apply_primary_and_reliability_separately() -> None:
    rows = _rows()
    ratings = evaluation._ratings_report(rows, replicates=100)
    draft = evaluation._draft_report(
        rows,
        parity={"artifact_sha256": "a" * 64, "numerical_parity": {"passed": True}},
        replicates=100,
    )
    assert ratings["primary_gate_passed"] is True
    assert ratings["locked_reliability_gate"][
        "candidate_minus_each_comparator_ece_upper_95_maximum"
    ] == 0.01
    assert draft["primary_gate_passed"] is True
    assert draft["subgroup_nonharm_gate_passed"] is True
    assert ratings["entity_network_dependence_gate_passed"] is True
    assert draft["entity_network_dependence_gate_passed"] is True
    assert draft["typescript_parity_artifact_sha256"] == "a" * 64


def test_entity_network_hac_links_shared_players_organizations_and_series() -> None:
    rows = _rows()
    report = evaluation._entity_network_hac_interval(
        rows,
        candidate_key="ratings_plus_draft",
        comparator_key="ratings_only",
        metric="log_loss",
    )
    assert report["complete"] is True
    assert report["series"] == 100
    assert report["participants"] == 1000
    assert report["organizations"] == 200
    assert report["dependent_ordered_pairs"] == 400
    assert report["maximum_dependency_neighborhood"] == 2
    assert report["upper_95"] < 0


def test_entity_network_hac_requires_exact_participant_identity() -> None:
    rows = _rows()
    rows[0]["participant_ids"] = tuple(rows[0]["participant_ids"][:-1])
    with pytest.raises(
        evaluation.PhaseOneEvaluationError,
        match="ten exact participant IDs",
    ):
        evaluation._entity_network_hac_interval(
            rows,
            candidate_key="ratings_plus_draft",
            comparator_key="ratings_only",
            metric="brier_score",
        )


def _ratings_with_roster() -> dict[str, object]:
    return {
        "input_receipts": {
            "roster": {
                "receipt": {
                    "teams": [
                        {
                            "organization_id": f"organization-{side}",
                            "players": [
                                {"player_id": f"player-{side}-{slot}"}
                                for slot in range(5)
                            ],
                        }
                        for side in ("blue", "red")
                    ]
                }
            }
        }
    }


def test_evaluation_entities_extracts_exact_roster_identity() -> None:
    participant_ids, organization_ids = evaluation._evaluation_entities(
        _ratings_with_roster()
    )
    assert participant_ids == tuple(
        f"player-{side}-{slot}"
        for side in ("blue", "red")
        for slot in range(5)
    )
    assert organization_ids == ("organization-blue", "organization-red")


def test_evaluation_entities_fails_closed_on_duplicate_player() -> None:
    ratings = _ratings_with_roster()
    teams = ratings["input_receipts"]["roster"]["receipt"]["teams"]
    teams[1]["players"][4]["player_id"] = "player-blue-0"
    with pytest.raises(
        evaluation.PhaseOneEvaluationError,
        match="ten exact players",
    ):
        evaluation._evaluation_entities(ratings)


def test_outcome_cohort_requires_exact_snapshot_and_evidence_bytes(tmp_path: Path) -> None:
    evidence_locator = (
        evaluation.OUTCOME_EVIDENCE_PREFIX / "event-1-map-1.json"
    ).as_posix()
    evidence_path = tmp_path / evidence_locator
    evidence_path.parent.mkdir(parents=True)
    evidence_raw = b'{"winner":"blue"}\n'
    evidence_path.write_bytes(evidence_raw)
    snapshot = {
        "artifact_sha256": "1" * 64,
        "event_bundles": [
            {
                "event_id": "event-1",
                "series_id": "series-1",
                "game_number": 1,
                "actual_map_start_utc": "2026-08-03T01:00:00+00:00",
            }
        ],
    }
    payload = {
        "schema_version": evaluation.OUTCOME_SCHEMA_VERSION,
        "created_at_utc": "2026-08-03T03:00:00+00:00",
        "snapshot_artifact_sha256": "1" * 64,
        "rows": [
            {
                "event_id": "event-1",
                "series_id": "series-1",
                "game_number": 1,
                "actual_map_start_utc": "2026-08-03T01:00:00+00:00",
                "winning_side": "blue",
                "source_system": "authoritative-results",
                "source_record_id": "result-1",
                "source_revision_id": "revision-1",
                "source_observed_at_utc": "2026-08-03T02:00:00+00:00",
                "evidence_locator": evidence_locator,
                "evidence_raw_sha256": hashlib.sha256(evidence_raw).hexdigest(),
            }
        ],
    }
    payload["artifact_sha256"] = evaluation._canonical_sha256(payload)
    checked = evaluation.validate_outcome_cohort(
        payload, snapshot=snapshot, root=tmp_path
    )
    assert checked["rows"][0]["winning_side"] == "blue"

    changed = json.loads(json.dumps(payload))
    changed["rows"][0]["winning_side"] = "red"
    with pytest.raises(evaluation.PhaseOneEvaluationError, match="hash changed"):
        evaluation.validate_outcome_cohort(changed, snapshot=snapshot, root=tmp_path)


def _authority(expected: dict[str, object]) -> dict[str, object]:
    attestation = {
        "reviewer_not_model_author_candidate_selector_or_evaluator_author": True,
        "reviewer_not_outcome_custodian": True,
        "reviewer_used_only_pinned_outcome_free_evidence": True,
        "outcomes_not_accessed_before_approval": True,
        "metadata_stopping_rule_independently_verified": True,
        "no_candidate_or_threshold_reselection_approved": True,
        "approval_not_generated_by_the_evaluated_system": True,
    }
    return {
        "schema_version": opening.SCHEMA_VERSION,
        "authority_id": "authority-1",
        "status": "APPROVED",
        "scope": "ONE_TIME_JOINT_PHASE_ONE_MODEL_EVALUATION_ONLY",
        "reviewed_at_utc": "2026-12-01T00:00:00+00:00",
        "reviews": [
            {
                "review_scope": "RATINGS_FUTURE_HOLDOUT",
                "reviewer_id": "reviewer-a",
                "reviewed_at_utc": "2026-11-30T20:00:00+00:00",
                "independence_attestation": attestation,
            },
            {
                "review_scope": "TERMINAL_DRAFT_FUTURE_HOLDOUT",
                "reviewer_id": "reviewer-b",
                "reviewed_at_utc": "2026-11-30T21:00:00+00:00",
                "independence_attestation": attestation,
            },
        ],
        "bindings": expected,
        "sealed_outcomes": {
            "cohort_locator": (evaluation.OUTCOME_PREFIX / "cohort-v1.json").as_posix(),
            "cohort_raw_sha256": "b" * 64,
            "custodian_id": "custodian-1",
            "sealed_at_utc": "2026-11-30T19:00:00+00:00",
            "custodian_attestation": {
                "digest_created_without_disclosing_outcomes_to_model_authors_or_reviewers": True,
                "cohort_bytes_immutable_after_digest": True,
                "cohort_matches_the_joint_snapshot_without_manual_post_outcome_exclusion": True,
            },
        },
        "one_time_run": {
            "run_id": "phase-one-run-1",
            "opening_marker_locator": (
                opening.OPENING_MARKER_PREFIX / "phase-one-run-1.json"
            ).as_posix(),
            "authorized_output_locator": (
                evaluation.OUTPUT_PREFIX / "phase-one-run-1.json"
            ).as_posix(),
            "marker_written_before_outcome_read": True,
            "atomic_no_clobber_output_required": True,
            "partial_result_publication_prohibited": True,
            "second_opening_or_replacement_holdout_prohibited": True,
        },
        "claim_ceiling": opening.CLAIM_CEILING,
    }


def test_opening_authority_requires_two_distinct_reviewers_and_exact_bindings() -> None:
    expected = {
        "joint_snapshot_raw_sha256": "1" * 64,
        "draft_parity_raw_sha256": "2" * 64,
    }
    payload = _authority(expected)
    assert opening.validate_opening_authority(
        payload, expected_bindings=expected
    )["status"] == "APPROVED"
    payload["reviews"][1]["reviewer_id"] = "reviewer-a"  # type: ignore[index]
    with pytest.raises(opening.PhaseOneOpeningError, match="not independent"):
        opening.validate_opening_authority(payload, expected_bindings=expected)


def test_pre_boundary_evaluation_readiness_builds_with_zero_future_artifacts(
    historical_capture_root: Path,
) -> None:
    payload = readiness.build_phase_one_evaluation_readiness_v1(
        root=historical_capture_root,
        clock=lambda: datetime(2026, 8, 2, 4, 0, tzinfo=timezone.utc),
    )
    checked = readiness.validate_phase_one_evaluation_readiness_v1(
        payload, root=historical_capture_root
    )
    assert checked["result_state"] == readiness.RESULT_STATE
    assert checked["locked_empty_state"]["outcomes_accessed"] is False
    assert all(value is False for value in checked["authority"].values())

    with pytest.raises(
        readiness_registry.RegisteredPhaseOneEvaluationReadinessError,
        match="invalid",
    ):
        readiness_registry.validate_registered_phase_one_evaluation_readiness_v1(
            root=historical_capture_root
        )


def test_independent_result_registry_preserves_pass_without_opening_phase_two() -> None:
    binding = {
        "result_raw_sha256": "1" * 64,
        "result_artifact_sha256": "2" * 64,
        "phase_one_models_passed": True,
    }
    attestation = {
        "reviewer_not_model_author_candidate_selector_evaluator_author_or_outcome_custodian": True,
        "exact_opening_authority_snapshot_outcome_and_result_hashes_verified": True,
        "registered_bootstrap_seeds_replicates_strata_and_gates_replayed": True,
        "no_post_opening_candidate_threshold_or_cohort_change_found": True,
        "reported_pass_or_failure_reconciles_with_locked_rules": True,
        "review_not_generated_by_the_evaluated_system": True,
    }
    payload = {
        "schema_version": result_registry.SCHEMA_VERSION,
        "registry_id": "registry-1",
        "status": "REGISTERED_PASS",
        "registered_at_utc": "2026-12-02T00:00:00+00:00",
        "reviews": [
            {
                "review_scope": "RATINGS_RESULT",
                "reviewer_id": "reviewer-a",
                "reviewed_at_utc": "2026-12-01T20:00:00+00:00",
                "attestation": attestation,
            },
            {
                "review_scope": "TERMINAL_DRAFT_RESULT",
                "reviewer_id": "reviewer-b",
                "reviewed_at_utc": "2026-12-01T21:00:00+00:00",
                "attestation": attestation,
            },
        ],
        "result_binding": binding,
        "terminal_decision": {
            "phase_one_evaluation_independently_registered": True,
            "phase_one_models_independently_passed": True,
            "phase_two_available_for_separate_recalibration_uncertainty_and_opening_work": True,
            "phase_two_opening_authorized": False,
            "recalibration_authorized": False,
            "failure_is_terminal_no_reopening_or_candidate_substitution": False,
        },
        "claim_ceiling": result_registry.CLAIM_CEILING,
    }
    checked = result_registry.validate_evaluation_registry(
        payload, expected_binding=binding
    )
    assert checked["terminal_decision"]["phase_one_models_independently_passed"] is True
    assert checked["terminal_decision"]["phase_two_opening_authorized"] is False
