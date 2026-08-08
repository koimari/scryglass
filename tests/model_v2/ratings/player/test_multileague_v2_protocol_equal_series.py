from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v2_protocol_equal_series as protocol


ARTIFACT = Path(
    "data/lol/v2/models/player/multileague-v2/protocol-lock-v2.json"
)


def test_current_equal_series_lock_is_disclosed_balanced_and_sealed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validated = protocol.validate_equal_series_protocol_lock(payload)
    assert validated["adaptation_disclosure"]["status"] == (
        "METADATA_SUPPORT_CORRECTION_AFTER_FAILED_LOCK"
    )
    assert [
        item["series"] for item in validated["adaptive_development"]["windows"]
    ] == [165, 164, 164]
    assert validated["adaptive_development"]["assignment_uses_outcomes"] is False
    assert validated["information_boundary"]["sealed_final_targets_accessed"] is False
    assert validated["sealed_final_gate"]["opened"] is False
    assert all(value is None for value in validated["decision_outputs"].values())


def test_window_or_candidate_tamper_fails_closed() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    window_tamper = copy.deepcopy(payload)
    window_tamper["adaptive_development"]["windows"][0]["series"] = 175
    window_tamper["artifact_sha256"] = protocol._artifact_sha256(window_tamper)
    with pytest.raises(protocol.EqualSeriesProtocolError, match="window manifest"):
        protocol.validate_equal_series_protocol_lock(window_tamper)

    candidate_tamper = copy.deepcopy(payload)
    candidate_tamper["candidate_family"]["candidates"][0]["organization_weight"] = 2.0
    candidate_tamper["artifact_sha256"] = protocol._artifact_sha256(candidate_tamper)
    with pytest.raises(protocol.EqualSeriesProtocolError, match="candidate family"):
        protocol.validate_equal_series_protocol_lock(candidate_tamper)
