from __future__ import annotations

import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v2_protocol as protocol
from lol_kills.v2.ratings.player import multileague_v2_runner as runner


ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v2/adaptive-development-artifact-v1.json"
)


def candidate(retention: float = 0.5) -> runner.CandidateSpec:
    payload = copy.deepcopy(protocol._candidate_payloads()[0])
    payload["organization_roster_retention"]["floor"] = retention
    return runner.CandidateSpec.from_payload(payload)


def test_roster_transition_is_psd_and_uses_exact_retained_player_fraction() -> None:
    state = runner.HierarchicalGaussianState(candidate(0.5))
    at = datetime(2026, 1, 1)
    state.transition_entities([], ["team"], at)
    key = runner._organization_key("team")
    index = state.index[key]
    state.mean[index] = 1.0
    previous = tuple((role, f"p{index}") for index, role in enumerate(("top", "jungle", "mid", "bot", "support")))
    current = previous[:-1] + (("support", "substitute"),)

    phi = state.apply_roster_transition("team", previous, current)

    assert phi == pytest.approx(0.9)
    assert state.mean[index] == pytest.approx(0.9)
    assert state.assert_psd()["minimum_eigenvalue"] >= 0.0


def test_current_artifact_is_bound_non_authorizing_and_sealed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validated = runner.validate_adaptive_development_artifact(payload)
    assert validated["selection"][
        "selection_is_adaptive_not_independent_validation"
    ] is True
    assert validated["sealed_final"] == {
        "opened": False,
        "targets_accessed": False,
        "series": 398,
        "maps": 1007,
        "opening_authority_present": False,
        "gate_passed": False,
    }
    assert all(value is None for value in validated["decision_outputs"].values())
    if validated["adaptive_posterior"] is not None:
        for team in validated["adaptive_posterior"]["teams"]:
            assert team["components"]["lineup_synergy"]["status"] == "UNAVAILABLE"
            assert team["components"]["lineup_synergy"]["posterior_mean_logit"] is None
            assert team["components"]["team_policy"]["status"] == "UNAVAILABLE"
            assert team["unavailable_components_are_not_zero"] is True


def test_tampered_selection_and_decision_output_fail_closed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(payload)
    tampered["selection"]["selected_candidate_id"] = "not-eligible"
    tampered["artifact_sha256"] = runner._artifact_sha256(tampered)
    with pytest.raises(runner.MultiLeagueV2RunnerError, match="not eligible"):
        runner.validate_adaptive_development_artifact(tampered)

    output_tamper = copy.deepcopy(payload)
    output_tamper["decision_outputs"]["match_probability"] = 0.6
    output_tamper["artifact_sha256"] = runner._artifact_sha256(output_tamper)
    with pytest.raises(runner.MultiLeagueV2RunnerError, match="decision outputs"):
        runner.validate_adaptive_development_artifact(output_tamper)


def test_writer_never_clobbers(tmp_path: Path) -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    output = tmp_path / ARTIFACT.name
    raw_sha256 = runner.write_adaptive_artifact_no_clobber(output, payload)
    assert raw_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="refusing to replace"):
        runner.write_adaptive_artifact_no_clobber(output, payload)
