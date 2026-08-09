from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from lol_kills.v2.draft.interactions.real_v1_g4 import coverage_preflight as preflight


def _payload() -> dict:
    return json.loads(preflight.OUTPUT_PATH.read_text(encoding="utf-8"))


def test_committed_coverage_preflight_is_target_free_and_replays_exactly() -> None:
    payload = _payload()
    expected = preflight.build_preflight()
    assert preflight.validate_preflight(payload) == expected
    assert preflight.canonical_bytes(payload) == preflight.OUTPUT_PATH.read_bytes()
    assert payload["status"] == "NO_INCREMENTAL_DRAFT_WINNER"
    assert payload["reason_code"] == "OUTCOME_FREE_COVERAGE_GATE_FAILED"
    assert payload["fallback"] == "M0_NOT_SCORED"
    assert payload["selected_model"] is None
    assert payload["target_loader_calls"] == payload["m0_loader_calls"] == payload["outcome_loader_calls"] == payload["fit_execution_calls"] == 0
    assert payload["final_holdout_loaded"] is False
    assert len(payload["slots"]) == 52
    assert payload["failed_slot_count"] == 24
    assert payload["first_failure"]["calendar_month"] == "2025-04"


def test_coverage_preflight_schema_is_closed() -> None:
    schema = json.loads(preflight.SCHEMA_PATH.read_text(encoding="utf-8"))
    assert list(Draft202012Validator(schema).iter_errors(_payload())) == []
    altered = deepcopy(_payload())
    altered["unexpected"] = True
    assert list(Draft202012Validator(schema).iter_errors(altered))


def test_failed_slots_are_outcome_free_support_diagnostics_only() -> None:
    payload = _payload()
    text = json.dumps(payload, sort_keys=True)
    for forbidden in ("y_blue_win", "p_blue_win", "winner", "target_rows"):
        assert forbidden not in text
    assert all(slot["execution_status"] in {"coverage_pass", "coverage_gate_failed_before_target_m0_or_outcome_load"} for slot in payload["slots"])
    assert any(not slot["coverage_passed"] for slot in payload["slots"])


def test_self_rehashed_preflight_cannot_claim_scored_m0_or_fit() -> None:
    altered = deepcopy(_payload())
    altered["fallback"] = "M0_SCORED"
    unsigned = dict(altered)
    unsigned.pop("artifact_sha256")
    altered["artifact_sha256"] = preflight._sha256(unsigned)
    with pytest.raises(preflight.CoveragePreflightError):
        preflight.validate_preflight(altered)
