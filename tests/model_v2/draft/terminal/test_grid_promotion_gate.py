from __future__ import annotations

import base64
import hashlib
import json
from datetime import datetime, timezone

from lol_kills.v2.draft.terminal.grid_promotion_gate import (
    SCHEMA_VERSION,
    evaluate_grid_promotion_gate,
    sha256_canonical,
)


def _record(game_id: str) -> dict:
    picks = []
    champions = {
        "A": {"top": "Aatrox", "jungle": "Nidalee", "mid": "Ahri", "bot": "Jinx", "support": "Thresh"},
        "B": {"top": "Gnar", "jungle": "Sejuani", "mid": "Orianna", "bot": "Aphelios", "support": "Rakan"},
    }
    slot = 1
    for side in ("A", "B"):
        for role in ("top", "jungle", "mid", "bot", "support"):
            picks.append({"slot": slot, "kind": "pick", "canonical_side": side, "role": role, "champion_id": champions[side][role]})
            slot += 1
    payload = {
        "game_id": game_id,
        "event_start": "2026-06-01T12:00:00Z",
        "source_available_at": "2026-06-01T10:00:00Z",
        "source_retrieved_at": "2026-06-01T10:05:00Z",
        "patch": "16.10",
        "side_mapping": {
            "A": {"game_side": "blue", "draft_order": "first"},
            "B": {"game_side": "red", "draft_order": "second"},
        },
        "picks": picks,
        "result": "A",
    }
    source_raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    record = {
        "game_id": game_id,
        "payload": payload,
        "source_payload_base64": base64.b64encode(source_raw).decode("ascii"),
        "source_payload_sha256": hashlib.sha256(source_raw).hexdigest(),
        "identity_checks": {"game_id_matches": True, "teams_are_distinct": True},
        "sequence_checks": {"slots_contiguous": True, "picks_complete": True},
        "leakage_checks": {"pre_event_inputs_only": True, "result_excluded_from_draft_inputs": True},
    }
    record["record_sha256"] = sha256_canonical(record)
    return record


def _manifest() -> dict:
    def held_out(name: str, oe: dict, grid: dict) -> dict:
        tolerances = {"log_loss": 0.0, "brier_score": 0.0, "ece": 0.0}
        plan = {
            "cohort_id": name,
            "baseline_source": "OE",
            "candidate_source": "GRID",
            "metrics": ["log_loss", "brier_score", "ece"],
            "max_allowed_delta": tolerances,
        }
        return {
            "predeclared_plan": plan,
            "predeclared_plan_sha256": sha256_canonical(plan),
            "predeclared_at": "2026-05-01T00:00:00Z",
            "results_recorded_at": "2026-06-03T00:00:00Z",
            "oe": oe,
            "grid": grid,
            "max_allowed_delta": tolerances,
            "results_sha256": sha256_canonical({"oe": oe, "grid": grid, "max_allowed_delta": tolerances}),
        }

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "provider": "GRID",
        "cohort": {
            "competition_id": "test:competition",
            "date_start": "2026-06-01T00:00:00Z",
            "date_end": "2026-06-02T00:00:00Z",
            "game_ids": ["game-1"],
        },
        "records": [_record("game-1")],
        "model": {
            "model_id": "test:grid-model",
            "payload": {"model_version": "test-grid-v1.0.0", "coefficient_count": 12},
        },
        "held_out": {
            "validation": held_out(
                "validation",
                {"log_loss": 0.70, "brier_score": 0.25, "ece": 0.04},
                {"log_loss": 0.69, "brier_score": 0.24, "ece": 0.03},
            ),
            "calibration": held_out(
                "calibration",
                {"log_loss": 0.71, "brier_score": 0.26, "ece": 0.05},
                {"log_loss": 0.71, "brier_score": 0.26, "ece": 0.05},
            ),
        },
    }
    model_raw = json.dumps(manifest["model"]["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    manifest["model"]["payload_base64"] = base64.b64encode(model_raw).decode("ascii")
    manifest["model"]["model_sha256"] = hashlib.sha256(model_raw).hexdigest()
    data_hash = sha256_canonical({"cohort": manifest["cohort"], "records": manifest["records"]})
    manifest["second_replay"] = {
        "status": "identical",
        "first_data_sha256": data_hash,
        "second_data_sha256": data_hash,
        "first_model_sha256": manifest["model"]["model_sha256"],
        "second_model_sha256": manifest["model"]["model_sha256"],
    }
    return manifest


def test_grid_can_replace_oe_only_after_complete_cohort_gate() -> None:
    result = evaluate_grid_promotion_gate(_manifest())
    assert result["status"] == "passed"
    assert result["primary_source_for_cohort"] == "GRID"
    assert result["public_reproducibility_benchmark"] == "OE"
    assert result["oe_remains_active"] is False
    assert result["blockers"] == []


def test_grid_failure_reports_exact_record_and_keeps_oe_active() -> None:
    manifest = _manifest()
    manifest["records"][0]["payload"]["picks"][0]["role"] = "mid"
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert result["primary_source_for_cohort"] == "OE"
    assert result["public_reproducibility_benchmark"] == "OE"
    assert result["oe_remains_active"] is True
    assert "record.game-1.source_payload_content_mismatch" in result["blockers"]
    assert "record.game-1.picks.role_invalid" in result["blockers"]
    assert result["missing_or_invalid_records"]["invalid"]["game-1"]


def test_grid_gate_rejects_held_out_numbers_without_a_predeclared_plan() -> None:
    manifest = _manifest()
    manifest["held_out"]["validation"].pop("predeclared_plan")
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert "held_out.validation.predeclared_plan_missing" in result["blockers"]
    assert result["held_out"]["validation"]["status"] == "failed"


def test_grid_gate_requires_results_after_the_predeclared_plan() -> None:
    manifest = _manifest()
    manifest["held_out"]["validation"]["results_recorded_at"] = manifest["held_out"]["validation"]["predeclared_at"]
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert "held_out.validation.results_recorded_not_after_plan" in result["blockers"]


def test_grid_gate_requires_named_identity_sequence_and_leakage_checks() -> None:
    manifest = _manifest()
    manifest["records"][0]["leakage_checks"] = {"unrelated_check": True}
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert "record.game-1.leakage_checks.pre_event_inputs_only.missing" in result["blockers"]
    assert "record.game-1.leakage_checks.result_excluded_from_draft_inputs.missing" in result["blockers"]


def test_grid_gate_requires_explicit_game_side_mapping() -> None:
    manifest = _manifest()
    manifest["records"][0]["payload"].pop("side_mapping")
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert "record.game-1.side_mapping.missing_or_invalid" in result["blockers"]


def test_grid_gate_requires_exact_source_bytes_not_a_caller_claimed_hash() -> None:
    manifest = _manifest()
    manifest["records"][0]["source_payload_base64"] = base64.b64encode(b"{}").decode("ascii")
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert "record.game-1.source_payload_content_mismatch" in result["blockers"]
    assert result["primary_source_for_cohort"] == "OE"


def test_grid_gate_binds_replay_hashes_to_verified_data_and_model() -> None:
    manifest = _manifest()
    manifest["second_replay"]["first_data_sha256"] = "c" * 64
    manifest["second_replay"]["second_data_sha256"] = "c" * 64
    manifest["second_replay"]["first_model_sha256"] = "d" * 64
    manifest["second_replay"]["second_model_sha256"] = "d" * 64
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert "second_replay.data_hash_not_bound_to_manifest" in result["blockers"]
    assert "second_replay.model_hash_not_bound_to_manifest" in result["blockers"]


def test_grid_gate_binds_held_out_results_to_a_hash() -> None:
    manifest = _manifest()
    manifest["held_out"]["validation"]["grid"]["ece"] = 0.99
    result = evaluate_grid_promotion_gate(manifest)
    assert result["status"] == "blocked"
    assert "held_out.validation.results_hash_mismatch" in result["blockers"]
