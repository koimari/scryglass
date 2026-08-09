from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import (
    multileague_v3_corrected_adaptive_diagnostic_v1 as diagnostic,
)
from lol_kills.v2.ratings.player.multileague_v3_corrected_adaptive_diagnostic_registry_v1 import (
    validate_registered_corrected_adaptive_diagnostic_v1,
)


ROOT = Path(".").resolve()


def _registered_payload() -> dict:
    return json.loads((ROOT / diagnostic.DEFAULT_OUTPUT).read_text())


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = diagnostic._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_registered_corrected_diagnostic_retains_incumbent_without_authority() -> None:
    payload = validate_registered_corrected_adaptive_diagnostic_v1(root=ROOT)
    assert payload["result_state"] == diagnostic.RESULT_STATE
    assert payload["incumbent"]["candidate_id"] == (
        "hierarchical-orgw100-orgv025-retain100"
    )
    assert payload["adaptive_challenger"]["candidate_id"] == (
        "hierarchical-orgw025-orgv025-retain100"
    )
    assert payload["retention_decision"] == {
        "status": "RETAIN_REGISTERED_INCUMBENT",
        "challenger_superiority_required_strata": [
            "overall",
            "LCS",
            "one_or_both_rosters_changed",
        ],
        "challenger_superiority_gate_passed": False,
        "reason": payload["retention_decision"]["reason"],
        "does_not_validate_incumbent": True,
    }
    assert payload["information_boundary"]["future_holdout_targets_accessed"] is False
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())


def test_corrected_diagnostic_records_actual_unresolved_rating_strata() -> None:
    payload = validate_registered_corrected_adaptive_diagnostic_v1(root=ROOT)
    versus_organization = payload["incumbent"]["versus_comparators"][
        "predecessor-organization-random-walk"
    ]
    versus_player = payload["incumbent"]["versus_comparators"][
        "predecessor-player-random-walk"
    ]
    assert versus_organization["overall"]["status"] == (
        "PASS_NONPOSITIVE_UPPER_95"
    )
    assert versus_organization["LCS"]["status"] == "FAIL_UPPER_95_ABOVE_ZERO"
    assert versus_organization["one_or_both_rosters_changed"]["status"] == (
        "FAIL_UPPER_95_ABOVE_ZERO"
    )
    assert versus_player["overall"]["status"] == "FAIL_UPPER_95_ABOVE_ZERO"
    assert payload["reliability"]["status"] == "UNAVAILABLE_AS_AUTHORITY"


def test_corrected_diagnostic_rejects_forged_rating_authority() -> None:
    forged = deepcopy(_registered_payload())
    forged["authority"]["team_rating_authority"] = True
    _resign(forged)
    with pytest.raises(
        diagnostic.CorrectedAdaptiveDiagnosticError,
        match="exceeded authority",
    ):
        diagnostic.validate_corrected_adaptive_diagnostic(forged, root=ROOT)


def test_corrected_diagnostic_rejects_false_supersession_gate() -> None:
    forged = deepcopy(_registered_payload())
    forged["retention_decision"]["challenger_superiority_gate_passed"] = True
    _resign(forged)
    with pytest.raises(
        diagnostic.CorrectedAdaptiveDiagnosticError,
        match="retention boundary changed",
    ):
        diagnostic.validate_corrected_adaptive_diagnostic(forged, root=ROOT)


def test_corrected_diagnostic_replays_exactly() -> None:
    payload = _registered_payload()
    replayed = diagnostic.replay_corrected_adaptive_diagnostic(
        payload, root=ROOT
    )
    assert replayed["artifact_sha256"] == payload["artifact_sha256"]
