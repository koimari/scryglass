from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v2_runner_equal_series as runner


ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v2/adaptive-development-artifact-v2.json"
)


def test_current_equal_series_replay_selects_only_adaptive_candidate_and_stays_sealed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validated = runner.validate_equal_series_adaptive_artifact(payload)
    assert validated["result_state"] == runner.RESULT_SELECTED
    assert validated["selection"]["selected_candidate_id"] in validated["selection"][
        "eligible_candidate_ids"
    ]
    assert validated["selection"][
        "selection_is_adaptive_not_independent_validation"
    ] is True
    assert [item["series"] for item in validated["window_manifests"]] == [165, 164, 164]
    assert validated["sealed_final"]["opened"] is False
    assert validated["sealed_final"]["targets_accessed"] is False
    assert all(value is None for value in validated["decision_outputs"].values())

    posterior = validated["adaptive_posterior"]
    assert posterior["candidate_id"] == validated["selection"]["selected_candidate_id"]
    assert posterior["players"]
    assert posterior["teams"]
    for team in posterior["teams"]:
        assert team["components"]["lineup_synergy"]["status"] == "UNAVAILABLE"
        assert team["components"]["lineup_synergy"]["posterior_mean_logit"] is None
        assert team["components"]["team_policy"]["status"] == "UNAVAILABLE"
        assert team["unavailable_components_are_not_zero"] is True


def test_selection_or_sealed_tamper_fails_closed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    selection_tamper = copy.deepcopy(payload)
    selection_tamper["selection"]["selected_candidate_id"] = "not-eligible"
    selection_tamper["artifact_sha256"] = runner._artifact_sha256(selection_tamper)
    with pytest.raises(runner.EqualSeriesRunnerError, match="not eligible"):
        runner.validate_equal_series_adaptive_artifact(selection_tamper)

    sealed_tamper = copy.deepcopy(payload)
    sealed_tamper["sealed_final"]["opened"] = True
    sealed_tamper["artifact_sha256"] = runner._artifact_sha256(sealed_tamper)
    with pytest.raises(runner.EqualSeriesRunnerError, match="opened sealed"):
        runner.validate_equal_series_adaptive_artifact(sealed_tamper)
