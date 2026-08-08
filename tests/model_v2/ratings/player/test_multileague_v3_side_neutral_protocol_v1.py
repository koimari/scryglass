from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import (
    multileague_v3_side_neutral_protocol_v1 as protocol,
)
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_side_neutral_protocol,
)


TEST_CLOCK = datetime(2026, 8, 2, 7, 21, tzinfo=timezone.utc)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = protocol._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


@pytest.fixture(scope="module")
def candidate() -> dict:
    return protocol.build_side_neutral_protocol_lock(
        root=Path(".").resolve(), clock=lambda: TEST_CLOCK
    )


def test_candidate_changes_only_capture_semantics(candidate: dict) -> None:
    assert candidate["result_state"] == protocol.RESULT_STATE
    assert candidate["locked_empty_state"] == {
        "legacy_prediction_receipts": 0,
        "pre_side_envelopes": 0,
        "side_bindings": 0,
        "legacy_prediction_registry_present": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
    }
    assert candidate["supersession"]["capture_semantics_changed"] is True
    for field in (
        "candidate_changed",
        "source_snapshot_changed",
        "future_boundary_changed",
        "support_stopping_rule_changed",
        "evaluation_rule_changed",
        "comparators_changed",
        "uncertainty_rule_changed",
        "opening_rule_changed",
        "future_outcomes_used_to_design_revision",
        "future_conditional_predictions_used_to_design_revision",
    ):
        assert candidate["supersession"][field] is False
    assert candidate["registration"]["collection_authorized"] is False
    assert all(value is False for value in candidate["authority"].values())
    assert all(value is None for value in candidate["decision_outputs"].values())


def test_candidate_rejects_post_boundary_lock() -> None:
    with pytest.raises(protocol.SideNeutralProtocolError, match="future boundary"):
        protocol.build_side_neutral_protocol_lock(
            root=Path(".").resolve(),
            clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


def test_candidate_rejects_resigned_scope_or_authority_forgery(
    candidate: dict,
) -> None:
    changed_model = deepcopy(candidate)
    changed_model["supersession"]["candidate_changed"] = True
    _resign(changed_model)
    with pytest.raises(protocol.SideNeutralProtocolError, match="scope changed"):
        protocol.validate_side_neutral_protocol_lock(
            changed_model, root=Path(".").resolve()
        )

    forged_authority = deepcopy(candidate)
    forged_authority["authority"]["probability_authority"] = True
    _resign(forged_authority)
    with pytest.raises(protocol.SideNeutralProtocolError, match="exceeds authority"):
        protocol.validate_side_neutral_protocol_lock(
            forged_authority, root=Path(".").resolve()
        )


def test_registered_candidate_is_hash_pinned_and_non_authorizing() -> None:
    checked = validate_registered_side_neutral_protocol(root=Path(".").resolve())
    assert checked["artifact_sha256"] == REGISTERED_PROTOCOL_ARTIFACT_SHA256
    assert len(REGISTERED_PROTOCOL_RAW_SHA256) == 64
    assert checked["registration"]["independent_reviewer_digest_present"] is False
    assert checked["registration"]["collection_authorized"] is False


def test_protocol_write_is_no_clobber(candidate: dict, tmp_path: Path) -> None:
    output = tmp_path / "protocol.json"
    protocol.write_no_clobber(output, candidate)
    with pytest.raises(protocol.SideNeutralProtocolError, match="overwrite"):
        protocol.write_no_clobber(output, candidate)
