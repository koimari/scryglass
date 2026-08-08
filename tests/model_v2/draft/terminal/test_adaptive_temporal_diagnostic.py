from __future__ import annotations

import json

from lol_kills.v2.draft.terminal.adaptive_temporal_diagnostic import (
    DEFAULT_OUTPUT,
    build_adaptive_temporal_diagnostic,
    validate_adaptive_temporal_diagnostic,
)


def test_bound_adaptive_temporal_diagnostic_replays_and_preserves_claim_ceiling() -> None:
    payload = json.loads(DEFAULT_OUTPUT.read_text(encoding="utf-8"))

    validated = validate_adaptive_temporal_diagnostic(payload)

    assert validated == build_adaptive_temporal_diagnostic()
    assert validated["result_state"] == "ADAPTIVE_DRAFT_TERMS_HARM"
    assert validated["population"]["maps"] == 997
    assert validated["population"]["exact_roster_context_maps"] == 267
    assert validated["metrics"]["incremental_draft_against_same_context"] == {
        "n": 267,
        "brier_delta": 0.021379,
        "logloss_delta": 0.051739,
        "pass_rule": "both deltas must be nonpositive",
        "passed": False,
    }
    assert validated["model_scope"]["same_as_terminal_m0_candidate"] is False
    assert validated["decision"]["known_app_draft_family_nonharmful"] is False
    assert validated["decision"]["independent_validation"] is False
    assert validated["claim_ceiling"]["probability"] is False
    assert validated["claim_ceiling"]["betting"] is False
