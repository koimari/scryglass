from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, RefResolver

from lol_kills.v2.draft.terminal import (
    PROMOTION_SCHEMA_VERSION,
    G1_SCHEMA_VERSION,
    G1RosterError,
    G1RosterEvidence,
    L2AuthorityRecordError,
    TerminalDraft,
    TerminalDraftError,
    TerminalModel,
    TerminalPromotionBindings,
    load_promotion_receipt,
    load_l2_authority_record,
    authority_record_payload_sha256,
    render_terminal_contract,
    score_terminal_draft,
)
from lol_kills.v2.draft.terminal import semantic_draft_authority_v1


SHA = "a" * 64


def draft(*, mode: str = "neutral", side_a=None, side_b=None, available: str = "2026-07-01T11:00:00Z", payload_sha256: str = SHA, source_record_id: str = "source:terminal-fixture", roster_evidence=None, actions=None, final_assignments=None) -> TerminalDraft:
    return TerminalDraft.from_sides(
        side_a or {"top": "Aatrox", "jungle": "Nidalee", "mid": "Ahri", "bot": "Jinx", "support": "Thresh"},
        side_b or {"top": "Gnar", "jungle": "Sejuani", "mid": "Orianna", "bot": "Aphelios", "support": "Rakan"},
        event_start="2026-07-01T12:00:00Z",
        source_available_at=available,
        source_record_id=source_record_id,
        source_payload_sha256=payload_sha256,
        source_rights_status="reviewed",
        mode=mode,
        roster_evidence=roster_evidence,
        actions=actions,
        final_assignments=final_assignments,
    )


def model(*, authorized: bool = False, intercept: float = 0.0, model_as_of: str = "2026-06-30T23:59:59Z") -> TerminalModel:
    return TerminalModel(
        model_version="draft-terminal-dev-v1.0.0",
        model_as_of=model_as_of,
        intercept=intercept,
        calibration_slope=1.0,
        calibration_intercept=0.0,
        uncertainty_logit_sd=0.15,
        champion_role_logit={"top|Aatrox": 0.2, "mid|Ahri": 0.1},
        ally_synergy_logit={"Ahri|Jinx": 0.05},
        counter_logit={"top|Aatrox|Gnar": 0.12},
        artifact_sha256=SHA,
        authorizes_prediction=authorized,
    )


def g1_roster_payload(*, available_at: str = "2026-07-01T08:00:00Z", rights_status: str = "reviewed") -> bytes:
    return json.dumps(
        {
            "schema_version": G1_SCHEMA_VERSION,
            "source_record_id": "source:g1-roster-fixture",
            "event_start": "2026-07-01T12:00:00Z",
            "available_at": available_at,
            "retrieved_at": "2026-07-01T09:00:00Z",
            "rights_status": rights_status,
            "rosters": {
                "A": {
                    "roster_id": "roster:A",
                    "starters": [
                        {"role": "top", "player_id": "player:a-top"},
                        {"role": "jungle", "player_id": "player:a-jungle"},
                        {"role": "mid", "player_id": "player:a-mid"},
                        {"role": "bot", "player_id": "player:a-bot"},
                        {"role": "support", "player_id": "player:a-support"},
                    ],
                },
                "B": {
                    "roster_id": "roster:B",
                    "starters": [
                        {"role": "top", "player_id": "player:b-top"},
                        {"role": "jungle", "player_id": "player:b-jungle"},
                        {"role": "mid", "player_id": "player:b-mid"},
                        {"role": "bot", "player_id": "player:b-bot"},
                        {"role": "support", "player_id": "player:b-support"},
                    ],
                },
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def l2_authority_record_payload(model_artifact_sha256: str, bindings: TerminalPromotionBindings) -> bytes:
    return json.dumps(
        {
            "schema_version": "scryglass:draft-terminal-l2-authority-record:v1",
            "status": "approved",
            "authority_record_id": "test-only:l2-authority",
            "issued_at": "2026-07-01T13:00:00Z",
            "independent_reviewer_id": "test-only:independent-reviewer",
            "model_artifact_sha256": model_artifact_sha256,
            "candidate_registry_sha256": bindings.candidate_registry_sha256,
            "development_evaluation_sha256": bindings.development_evaluation_sha256,
            "l2_contract_sha256": bindings.l2_contract_sha256,
            "calibration_transform_sha256": "2" * 64,
            "reliability_artifact_sha256": "3" * 64,
            "replay_parity_evidence_sha256": "4" * 64,
            "source_snapshot_sha256": "8" * 64,
            "independent_l2_authority": True,
            "sealed_outer_temporal_holdout_decision": "passed",
            "source_snapshot": {
                "availability_status": "verified_preevent",
                "participant_cluster_status": "team_or_series_available",
                "series_grouped": True,
            },
            "holdouts": {
                "future_patch": "passed",
                "league": "passed",
                "international_event_or_meta": "passed",
                "roster_change": "not_required_for_neutral",
                "sparse_or_new_champion": "passed",
            },
            "reliability": {
                "validation_gate_passed": True,
                "probability_wording_approved": True,
                "baseline_support_verified": True,
                "dependence_support_verified": True,
                "interval_coverage_verified": True,
            },
            "claim_ceiling": {
                "descriptive_pre_map_association": True,
                "causal_draft_effect": False,
                "recommendation": False,
                "betting": False,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def test_development_score_reconciles_and_side_swap_complements() -> None:
    first = score_terminal_draft(draft(), model(), development=True)
    swapped = score_terminal_draft(
        draft(
            side_a={"top": "Gnar", "jungle": "Sejuani", "mid": "Orianna", "bot": "Aphelios", "support": "Rakan"},
            side_b={"top": "Aatrox", "jungle": "Nidalee", "mid": "Ahri", "bot": "Jinx", "support": "Thresh"},
        ),
        model(),
        development=True,
    )
    assert first["status"] == "development_only"
    assert first["score_a"] + first["score_b"] == pytest.approx(100.0, abs=1e-12)
    assert first["ledger_logit_sum"] == pytest.approx(first["uncalibrated_logit_a"], abs=1e-12)
    assert swapped["standardized_map_win_probability_a"] == pytest.approx(
        1.0 - first["standardized_map_win_probability_a"], abs=1e-12
    )
    assert swapped["interval_95"]["lower"] == pytest.approx(1.0 - first["interval_95"]["upper"], abs=1e-12)
    assert swapped["interval_95"]["upper"] == pytest.approx(1.0 - first["interval_95"]["lower"], abs=1e-12)


def test_artifact_replay_binds_exact_bytes_and_preserves_score() -> None:
    raw = json.dumps(model().to_artifact_mapping(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    loaded = TerminalModel.from_artifact_bytes(raw, expected_artifact_sha256=hashlib.sha256(raw).hexdigest())
    original = score_terminal_draft(draft(), model(), development=True)
    replayed = score_terminal_draft(draft(), loaded, development=True)
    assert loaded.artifact_sha256 == hashlib.sha256(raw).hexdigest()
    assert replayed["score_a"] == pytest.approx(original["score_a"], abs=1e-12)
    assert replayed["ledger"] == original["ledger"]
    with pytest.raises(TerminalDraftError, match="expected SHA-256"):
        TerminalModel.from_artifact_bytes(raw + b"\n", expected_artifact_sha256=loaded.artifact_sha256)


def test_candidate_registry_is_preregistered_without_selection_authority() -> None:
    registry_path = Path("data/lol/v2/models/draft-terminal/draft-terminal-candidate-registry.json")
    registry = json.loads(registry_path.read_text())
    assert registry["status"] == "preregistered_pending_source"
    assert registry["production_eligible"] is False
    assert registry["public_probability_authorized"] is False
    assert registry["evaluation_policy"]["status"] == "development_diagnostic_complete"
    assert registry["evaluation_policy"]["selected_candidate_id"] == "m0-role-additive"
    assert registry["evaluation_policy"]["selection_artifact"] == "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-report.json"
    assert registry["candidate_order"] == [
        "m0-role-additive",
        "m1-role-additive-allied-synergy",
        "m2-role-additive-allied-and-counter",
    ]
    artifact_path = Path("data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v1.json")
    artifact_raw = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact_raw).hexdigest()
    candidate = next(item for item in registry["candidates"] if item["candidate_id"] == "m0-role-additive")
    assert candidate["artifact_sha256"] == artifact_sha256
    TerminalModel.from_artifact_bytes(artifact_raw, expected_artifact_sha256=candidate["artifact_sha256"])


def test_independent_authority_record_is_strictly_hash_bound() -> None:
    bindings = TerminalPromotionBindings.from_repo_root()
    raw = l2_authority_record_payload(SHA, bindings)
    record = load_l2_authority_record(raw)
    assert record["status"] == "approved"
    assert authority_record_payload_sha256(raw) == hashlib.sha256(raw).hexdigest()
    with pytest.raises(L2AuthorityRecordError, match="does not bind"):
        bindings.with_authority_record_bytes(raw, model_artifact_sha256="b" * 64)


def test_shared_replay_fixture_matches_python_terminal_path() -> None:
    fixture = json.loads(Path("data/lol/v2/models/draft-terminal/terminal-replay-fixture.json").read_text())
    artifact_raw = Path("data/lol/v2/models/draft-terminal/terminal-model-development-v1.json").read_bytes()
    loaded = TerminalModel.from_artifact_bytes(artifact_raw, expected_artifact_sha256=fixture["model_artifact_sha256"])
    fixture_draft = fixture["draft"]
    replay_draft = TerminalDraft.from_sides(
        fixture_draft["side_a"],
        fixture_draft["side_b"],
        event_start=fixture_draft["event_start"],
        source_available_at=fixture_draft["source_available_at"],
        source_record_id=fixture_draft["source_record_id"],
        source_payload_sha256=fixture_draft["source_payload_sha256"],
        source_rights_status=fixture_draft["source_rights_status"],
        mode=fixture_draft["mode"],
    )
    assert score_terminal_draft(replay_draft, loaded, development=True) == fixture["expected_development"]


def test_legal_terminal_history_is_required_for_promoted_mode() -> None:
    side_a = {"top": "Aatrox", "jungle": "Nidalee", "mid": "Ahri", "bot": "Jinx", "support": "Thresh"}
    side_b = {"top": "Gnar", "jungle": "Sejuani", "mid": "Orianna", "bot": "Aphelios", "support": "Rakan"}
    actions = []
    assignments = []
    slot = 1
    for role in ("top", "jungle", "mid", "bot", "support"):
        for canonical_side, champions in (("A", side_a), ("B", side_b)):
            action_id = f"scryglass:action:{slot}"
            actions.append({"action_id": action_id, "slot": slot, "kind": "pick", "canonical_side": canonical_side, "champion_id": champions[role], "role_set": [role]})
            assignments.append({"action_id": action_id, "canonical_side": canonical_side, "champion_id": champions[role], "role": role})
            slot += 1
    complete = draft(actions=actions, final_assignments=assignments)
    assert len(complete.actions) == 10
    assert len(complete.final_assignments) == 10
    blocked = score_terminal_draft(complete, model(authorized=True))
    assert blocked["status"] == "unavailable"
    assert blocked["error"]["code"] == "model_not_promoted"
    with pytest.raises(TerminalDraftError, match="not legal"):
        draft(actions=actions, final_assignments=[*assignments[:-1], {**assignments[-1], "role": "top"}])
    with pytest.raises(TerminalDraftError, match="pick/ban"):
        draft(actions=[*actions[:-1], {**actions[-1], "kind": "draft"}], final_assignments=assignments)
    with pytest.raises(TerminalDraftError, match="slots must be integers"):
        draft(actions=[{**actions[0], "slot": 1.0}, *actions[1:]], final_assignments=assignments)


def test_terminal_history_must_match_scored_side_composition() -> None:
    side_a = {"top": "Camille", "jungle": "Nidalee", "mid": "Ahri", "bot": "Jinx", "support": "Thresh"}
    action_side_a = {**side_a, "top": "Aatrox"}
    side_b = {"top": "Gnar", "jungle": "Sejuani", "mid": "Orianna", "bot": "Aphelios", "support": "Rakan"}
    actions = []
    assignments = []
    slot = 1
    for role in ("top", "jungle", "mid", "bot", "support"):
        for canonical_side, champions in (("A", action_side_a), ("B", side_b)):
            action_id = f"scryglass:history:{slot}"
            actions.append({"action_id": action_id, "slot": slot, "kind": "pick", "canonical_side": canonical_side, "champion_id": champions[role], "role_set": [role]})
            assignments.append({"action_id": action_id, "canonical_side": canonical_side, "champion_id": champions[role], "role": role})
            slot += 1
    with pytest.raises(TerminalDraftError, match="side composition"):
        draft(side_a=side_a, actions=actions, final_assignments=assignments)


def test_terminal_input_identity_binds_source_availability_and_payload_hash() -> None:
    baseline = draft().input_id
    assert draft(available="2026-07-01T10:59:59Z").input_id != baseline
    assert draft(payload_sha256="b" * 64).input_id != baseline


def test_terminal_input_hash_uses_python_ascii_json_canonicalization() -> None:
    assert draft(source_record_id="source:á").input_id == "b8a769fdfed02f3d5fdbf8c55ab595e45b0a96ee1e25d276bab87ca7b96cad69"


def test_authorized_contract_renderer_is_schema_valid_and_neutral(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    side_a = {"top": "Aatrox", "jungle": "Nidalee", "mid": "Ahri", "bot": "Jinx", "support": "Thresh"}
    side_b = {"top": "Gnar", "jungle": "Sejuani", "mid": "Orianna", "bot": "Aphelios", "support": "Rakan"}
    actions = []
    assignments = []
    slot = 1
    for role in ("top", "jungle", "mid", "bot", "support"):
        for canonical_side, champions in (("A", side_a), ("B", side_b)):
            action_id = f"scryglass:action:{slot}"
            actions.append({"action_id": action_id, "slot": slot, "kind": "pick", "canonical_side": canonical_side, "champion_id": champions[role], "role_set": [role]})
            assignments.append({"action_id": action_id, "canonical_side": canonical_side, "champion_id": champions[role], "role": role})
            slot += 1
    terminal = draft(actions=actions, final_assignments=assignments)
    terminal_model = model(authorized=True)
    bindings = TerminalPromotionBindings.from_repo_root()
    authority_raw = l2_authority_record_payload(terminal_model.artifact_sha256, bindings)
    bindings = bindings.with_authority_record_bytes(authority_raw, model_artifact_sha256=terminal_model.artifact_sha256)
    bindings = replace(
        bindings,
        semantic_draft_authority_sha256="7" * 64,
        semantic_draft_authority_id="test-only:semantic-draft-authority",
        semantic_draft_model_artifact_sha256=terminal_model.artifact_sha256,
        semantic_draft_model_version=terminal_model.model_version,
    )
    receipt = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "approved",
        "model_version": terminal_model.model_version,
        "artifact_sha256": terminal_model.artifact_sha256,
        "l2_contract_sha256": bindings.l2_contract_sha256,
        "development_evaluation_sha256": bindings.development_evaluation_sha256,
        "candidate_registry_sha256": bindings.candidate_registry_sha256,
        "calibration_transform_sha256": "2" * 64,
        "reliability_artifact_sha256": "3" * 64,
        "replay_parity_evidence_sha256": "4" * 64,
        "independent_authority_record_sha256": bindings.independent_authority_record_sha256,
        "semantic_draft_authority_sha256": bindings.semantic_draft_authority_sha256,
        "semantic_draft_authority_id": bindings.semantic_draft_authority_id,
        "independent_l2_authority": True,
        "final_temporal_holdout_sealed": True,
        "private_terminal_draft_component_authorized": True,
        "public_probability_authorized": False,
        "event_probability_authorized": False,
        "exact_rating_receipt_required_for_event_probability": True,
        "replay_parity_verified": True,
        "reliability_gate_passed": True,
        "contextual_g1_authority": "not_applicable",
        "authority_record_id": bindings.authority_record_id,
        "issued_at": "2026-07-01T13:00:00Z",
    }
    monkeypatch.setattr(
        semantic_draft_authority_v1,
        "load_active_semantic_draft_authority_v1",
        lambda **_: {
            "receipt": {
                "authority_id": bindings.semantic_draft_authority_id,
            },
            "receipt_raw_sha256": bindings.semantic_draft_authority_sha256,
            "deployment_model": {
                "artifact_sha256": terminal_model.artifact_sha256,
                "model_version": terminal_model.model_version,
            },
            "private_terminal_draft_component_authorized": True,
            "private_event_probability_authorized": False,
            "public_probability_authorized": False,
            "betting_authorized": False,
        },
    )
    evidence = {
        "posterior_displacement": {"diagnostic_id": "scryglass:evidence:displacement", "method_id": "scryglass:method:displacement", "prior_id": "scryglass:prior:terminal", "value": 0.1, "unit": "nats"},
        "precision": {"diagnostic_id": "scryglass:evidence:precision", "method_id": "scryglass:method:precision", "reference_id": "scryglass:prior:terminal", "posterior_dispersion": 0.1, "reference_dispersion": 0.2, "unit": "probability_interval_width"},
        "source_context_coverage": {"coverage_spec_id": "scryglass:coverage:terminal", "supported_source_family_ids": ["scryglass:source-family:drafts"], "supported_context_ids": ["scryglass:context:neutral"], "missing_required_context_ids": [], "identity_terms_status": "not_applicable", "bridge_path_status": "not_applicable", "fallback_levels": [], "coverage_gaps": []},
    }
    reliability = {
        "label": "limited", "validation_stratum_id": "scryglass:stratum:terminal", "stratum_match_status": "matched", "stratum_mapping_sha256": "b" * 64, "benchmark_version": "2.0.0", "baseline_id": "scryglass:baseline:terminal", "probability_wording_approved": True, "validation_gate_passed": True, "out_of_distribution": False, "out_of_distribution_flags": [], "log_loss": 0.69, "baseline_log_loss": 0.70, "log_loss_skill": 0.01, "brier_score": 0.24, "baseline_brier_score": 0.25, "brier_skill": 0.01, "calibration_intercept": 0.0, "calibration_slope": 1.0, "empirical_interval_coverage": 0.95, "nominal_interval_coverage": 0.95, "sample_count": 100, "cluster_count": 50,
    }
    contract = {
        "season_id": "scryglass:season:dev-2026",
        "competition_scope_id": "scryglass:competition-scope:dev",
        "competition_scope_kind": "regional_league",
        "patch_id": "26.14",
        "protocol_id": "scryglass:protocol:dev",
        "event_id": None,
        "side_mapping": {"side_a_game_side": "blue", "side_b_game_side": "red", "side_a_draft_order": "first", "side_b_draft_order": "second", "mapping_source_id": "scryglass:source:protocol", "available_at": "2026-07-01T11:00:00Z", "mapping_basis": "observed"},
        "source_record": {"source_id": "scryglass:source:drafts", "source_record_id": "source:terminal-fixture", "source_revision_id": "scryglass:source-revision:1", "supersedes_source_revision_id": None, "observed_at": "2026-07-01T11:00:00Z", "available_at": "2026-07-01T11:00:00Z", "action_order_source": "observed"},
        "protocol_validation": {"status": "validated", "validator_id": "scryglass:protocol-validator:fixture", "validator_sha256": "6" * 64, "available_at": "2026-07-01T11:00:00Z", "action_order_verified": True, "pick_ban_counts_verified": True, "canonical_side_mapping_verified": True},
        "role_constraint_revisions": [],
        "assignment_revisions": [],
        "evidence": evidence,
        "reliability": reliability,
        "calibration_id": "scryglass:calibration:terminal-dev",
        "provenance": {"schema_version": "2.0.0", "model_version": terminal_model.model_version, "as_of": terminal_model.model_as_of, "prediction_id": "scryglass:prediction:fixture", "mode": "forecast", "created_at": "2026-07-01T11:30:00Z", "event_start": terminal.event_start, "availability_replayed": True, "sealed_before_event_start": True, "input_snapshot_id": "scryglass:input:fixture", "estimator_id": "scryglass:estimator:terminal", "calibration_id": "scryglass:calibration:terminal-dev", "probability_transform": {"transform_sha256": "c" * 64, "probability_domain": "open_0_1", "monotonicity": "nondecreasing", "complement_symmetry_verified": True, "open_support_verified": True, "transform_proof_sha256": "d" * 64}, "required_input_status": "complete", "freshness_checks": [{"input_id": "scryglass:source:drafts", "source_updated_at": "2026-07-01T11:00:00Z", "limit_seconds": 3600, "fresh": True}], "input_conflicts": [], "fallback_levels": [], "out_of_distribution_flags": [], "output_sha256": "e" * 64, "immutable": True},
    }
    response = render_terminal_contract(
        terminal,
        terminal_model,
        contract=contract,
        promotion_receipt=receipt,
        promotion_bindings=bindings,
    )
    schema_path = Path("docs/model-v2/contracts/draft-score.schema.json").resolve()
    schema = json.loads(schema_path.read_text())
    common_path = schema_path.with_name("common.schema.json")
    provenance_path = schema_path.with_name("prediction-provenance.schema.json")
    resolver = RefResolver(base_uri=schema_path.as_uri(), referrer=schema, store={"https://scryglass.xyz/schemas/model-v2/common.schema.json": json.loads(common_path.read_text()), "https://scryglass.xyz/schemas/model-v2/prediction-provenance.schema.json": json.loads(provenance_path.read_text())})
    Draft202012Validator(schema, resolver=resolver).validate(response)
    assert response["status"] == "ok"
    assert response["identity_mode"] == "neutral"
    assert "context" not in response
    assert response["as_of"] == terminal_model.model_as_of
    assert response["provenance"]["created_at"] < terminal.event_start
    assert response["score_a"] + response["score_b"] == pytest.approx(100.0, abs=1e-12)
    tampered_receipt = dict(receipt, calibration_transform_sha256="4" * 64)
    blocked = score_terminal_draft(
        terminal,
        terminal_model,
        promotion_receipt=tampered_receipt,
        promotion_bindings=bindings,
    )
    assert blocked["status"] == "unavailable"
    assert blocked["error"]["code"] == "model_not_promoted"
    mismatched_record = score_terminal_draft(
        terminal,
        terminal_model,
        promotion_receipt={**receipt, "authority_record_id": "test-only:other-authority"},
        promotion_bindings=bindings,
    )
    assert mismatched_record["status"] == "unavailable"
    assert mismatched_record["error"]["code"] == "model_not_promoted"
    without_protocol = dict(contract)
    without_protocol.pop("protocol_validation")
    unavailable = render_terminal_contract(
        terminal,
        terminal_model,
        contract=without_protocol,
        promotion_receipt=receipt,
        promotion_bindings=bindings,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["error"]["code"] == "missing_required_input"
    monkeypatch.setattr(
        semantic_draft_authority_v1,
        "load_active_semantic_draft_authority_v1",
        lambda **_: (_ for _ in ()).throw(
            semantic_draft_authority_v1.SemanticDraftAuthorityError(
                "external authority pin missing"
            )
        ),
    )
    inactive = score_terminal_draft(
        terminal,
        terminal_model,
        promotion_receipt=receipt,
        promotion_bindings=bindings,
    )
    assert inactive["status"] == "unavailable"
    assert inactive["error"]["code"] == "model_not_promoted"


def test_public_boundary_is_unavailable_without_promotion() -> None:
    result = score_terminal_draft(draft(), model())
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "model_not_promoted"
    assert "score_a" not in result


def test_incomplete_promotion_receipt_cannot_authorize_prediction() -> None:
    terminal_model = model(authorized=True)
    receipt = {
        "status": "approved",
        "model_version": terminal_model.model_version,
        "artifact_sha256": terminal_model.artifact_sha256,
        "public_probability_authorized": True,
        "replay_parity_verified": True,
        "reliability_gate_passed": True,
    }
    result = score_terminal_draft(draft(actions=[], final_assignments=[]), terminal_model, promotion_receipt=receipt)
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "model_not_promoted"


def test_promotion_receipt_loader_rejects_duplicate_keys_and_accepts_exact_payload() -> None:
    payload = {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "status": "approved",
        "model_version": "draft-terminal-dev-v1.0.0",
        "artifact_sha256": SHA,
        "l2_contract_sha256": "5" * 64,
        "development_evaluation_sha256": "f" * 64,
        "candidate_registry_sha256": "1" * 64,
        "calibration_transform_sha256": "2" * 64,
        "reliability_artifact_sha256": "3" * 64,
        "replay_parity_evidence_sha256": "4" * 64,
        "independent_authority_record_sha256": "9" * 64,
        "semantic_draft_authority_sha256": "7" * 64,
        "semantic_draft_authority_id": "test-only:semantic-draft-authority",
        "independent_l2_authority": True,
        "final_temporal_holdout_sealed": True,
        "private_terminal_draft_component_authorized": True,
        "public_probability_authorized": False,
        "event_probability_authorized": False,
        "exact_rating_receipt_required_for_event_probability": True,
        "replay_parity_verified": True,
        "reliability_gate_passed": True,
        "contextual_g1_authority": "not_applicable",
        "authority_record_id": "test-only:receipt",
        "issued_at": "2026-07-01T13:00:00Z",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert load_promotion_receipt(raw) == payload
    with pytest.raises(ValueError, match="duplicate key"):
        load_promotion_receipt(raw[:-1] + b',"status":"approved"}')
    invalid_time = dict(payload, issued_at="not-a-timestamp")
    with pytest.raises(ValueError, match="RFC-3339"):
        load_promotion_receipt(json.dumps(invalid_time, sort_keys=True, separators=(",", ":")).encode())
    legacy = dict(
        payload,
        schema_version="draft-terminal-promotion-receipt-v1",
        public_probability_authorized=True,
    )
    with pytest.raises(ValueError, match="schema_version"):
        load_promotion_receipt(
            json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()
        )
    public = dict(payload, public_probability_authorized=True)
    with pytest.raises(ValueError, match="cannot authorize probability"):
        load_promotion_receipt(
            json.dumps(public, sort_keys=True, separators=(",", ":")).encode()
        )


def test_unavailable_result_is_valid_draft_score_contract_without_numeric_output() -> None:
    result = score_terminal_draft(draft(), model())
    schema_path = Path("docs/model-v2/contracts/draft-score.schema.json").resolve()
    schema = json.loads(schema_path.read_text())
    common_path = schema_path.with_name("common.schema.json")
    provenance_path = schema_path.with_name("prediction-provenance.schema.json")
    resolver = RefResolver(
        base_uri=schema_path.as_uri(),
        referrer=schema,
        store={
            "https://scryglass.xyz/schemas/model-v2/common.schema.json": json.loads(common_path.read_text()),
            "https://scryglass.xyz/schemas/model-v2/prediction-provenance.schema.json": json.loads(
                provenance_path.read_text()
            ),
        },
    )
    Draft202012Validator(schema, resolver=resolver).validate(result)
    assert not {"score_a", "score_b", "standardized_map_win_probability_a"}.intersection(result)


def test_contextual_mode_is_gated_by_g1() -> None:
    result = score_terminal_draft(draft(mode="contextual"), model(authorized=True))
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "source_access_blocked"


def test_g1_roster_payload_is_hash_bound_and_contextual_model_stays_closed() -> None:
    raw = g1_roster_payload()
    evidence = G1RosterEvidence.from_payload_bytes(raw)
    assert evidence.source_payload_sha256 == hashlib.sha256(raw).hexdigest()
    assert evidence.is_available_for("2026-07-01T12:00:00Z") is True
    result = score_terminal_draft(draft(mode="contextual", roster_evidence=evidence), model(authorized=True))
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "model_not_promoted"
    assert result["error"]["missing_fields"] == ["contextual_fit_model", "player_champion_response", "team_policy_response"]


def test_g1_roster_payload_rejects_late_or_unreviewed_source() -> None:
    with pytest.raises(G1RosterError, match="not available"):
        G1RosterEvidence.from_payload_bytes(g1_roster_payload(available_at="2026-07-01T12:00:01Z"))
    with pytest.raises(G1RosterError, match="rights"):
        G1RosterEvidence.from_payload_bytes(g1_roster_payload(rights_status="unknown"))


def test_terminal_input_rejects_duplicate_champions_and_late_source() -> None:
    with pytest.raises(TerminalDraftError, match="ten unique"):
        draft(side_b={"top": "Aatrox", "jungle": "Sejuani", "mid": "Orianna", "bot": "Aphelios", "support": "Rakan"})
    with pytest.raises(TerminalDraftError, match="not available"):
        draft(available="2026-07-01T12:00:01Z")


def test_pre_event_source_gates_reject_event_start_timestamp() -> None:
    with pytest.raises(TerminalDraftError, match="not available"):
        draft(available="2026-07-01T12:00:00Z")
    with pytest.raises(G1RosterError, match="not available"):
        G1RosterEvidence.from_payload_bytes(g1_roster_payload(available_at="2026-07-01T12:00:00Z"))


def test_model_as_of_must_be_strictly_before_event_start() -> None:
    result = score_terminal_draft(
        draft(),
        model(model_as_of="2026-07-01T12:00:00Z"),
        development=True,
    )
    assert result["status"] == "unavailable"
    assert result["error"]["code"] == "prediction_time_violation"


def test_neutral_model_rejects_unidentified_intercept() -> None:
    with pytest.raises(TerminalDraftError, match="intercept=0"):
        model(intercept=0.1)
