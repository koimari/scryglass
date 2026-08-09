from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v2_protocol as protocol


LOCKED_AT = "2026-08-01T22:30:00Z"


@pytest.fixture(scope="module")
def locked() -> dict:
    return protocol.build_protocol_lock(locked_at=LOCKED_AT)


def test_observed_validation_is_reclassified_and_final_remains_unopened(
    locked: dict,
) -> None:
    assert locked["result_state"] == protocol.RESULT_STATE
    assert locked["validation_disclosure"]["status"] == (
        "RECLASSIFIED_AS_ADAPTIVE_DEVELOPMENT"
    )
    assert locked["information_boundary"]["sealed_final_targets_accessed"] is False
    assert locked["information_boundary"]["sealed_final_series"] == 398
    assert locked["sealed_final_gate"]["opened"] is False
    assert locked["sealed_final_gate"]["one_time_evaluation"] is True
    assert all(value is None for value in locked["decision_outputs"].values())


def test_candidate_grid_is_exact_unique_and_preserves_unavailable_components(
    locked: dict,
) -> None:
    candidates = locked["candidate_family"]["candidates"]
    assert len(candidates) == 12
    assert len({item["candidate_id"] for item in candidates}) == 12
    assert {item["organization_weight"] for item in candidates} == {0.25, 0.5, 1.0}
    assert {item["organization_prior_variance"] for item in candidates} == {0.25, 1.0}
    assert {
        item["organization_roster_retention"]["floor"] for item in candidates
    } == {0.5, 1.0}
    for candidate in candidates:
        assert candidate["lineup_synergy_component"] == {
            "status": "UNAVAILABLE",
            "value": None,
        }
        assert candidate["team_policy_component"] == {
            "status": "UNAVAILABLE",
            "value": None,
        }


def test_protocol_digest_and_current_source_locks_validate(locked: dict) -> None:
    validated = protocol.validate_protocol_lock(locked)
    assert validated["artifact_sha256"] == locked["artifact_sha256"]
    assert len(validated["source_locks"]) == 8


def test_tampering_candidate_or_source_pin_fails_closed(locked: dict) -> None:
    candidate_tamper = copy.deepcopy(locked)
    candidate_tamper["candidate_family"]["candidates"][0]["organization_weight"] = 9.0
    candidate_tamper["artifact_sha256"] = protocol._artifact_sha256(candidate_tamper)
    with pytest.raises(protocol.MultiLeagueV2ProtocolError, match="candidate family"):
        protocol.validate_protocol_lock(candidate_tamper)

    source_tamper = copy.deepcopy(locked)
    source_tamper["source_locks"][0]["raw_sha256"] = "0" * 64
    source_tamper["artifact_sha256"] = protocol._artifact_sha256(source_tamper)
    with pytest.raises(protocol.MultiLeagueV2ProtocolError, match="source drifted"):
        protocol.validate_protocol_lock(source_tamper)


def test_writer_is_immutable_and_returns_raw_digest(tmp_path: Path, locked: dict) -> None:
    output = tmp_path / "protocol-lock-v1.json"
    raw_sha256 = protocol.write_protocol_lock_no_clobber(output, locked)
    assert raw_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert json.loads(output.read_text(encoding="utf-8")) == locked
    with pytest.raises(FileExistsError, match="refusing to replace"):
        protocol.write_protocol_lock_no_clobber(output, locked)
