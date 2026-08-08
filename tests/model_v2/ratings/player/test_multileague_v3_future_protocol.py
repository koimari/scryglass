from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_future_protocol as protocol
from lol_kills.v2.ratings.player.multileague_v3_registry import (
    validate_registered_future_protocol,
)


def test_future_protocol_is_empty_prediction_led_and_non_authorizing(tmp_path: Path) -> None:
    root = Path(".").resolve()
    payload = protocol.build_future_protocol_lock(
        locked_at="2026-08-01T23:30:00Z",
        root=root,
    )
    assert payload["result_state"] == "FUTURE_HOLDOUT_PROTOCOL_LOCKED_EMPTY"
    assert payload["future_holdout"]["start_inclusive_source_time"] == (
        "2026-08-03T00:00:00"
    )
    assert payload["source_snapshot"]["latest_observed_source_time"] < (
        payload["future_holdout"]["start_inclusive_source_time"]
    )
    assert payload["future_holdout"]["eligibility"][
        "pre_event_prediction_ledger_required"
    ] is True
    assert payload["future_holdout"]["eligibility"][
        "retrospective_prediction_generation_qualifies"
    ] is False
    assert all(value is None for value in payload["decision_outputs"].values())
    assert payload["opening_authority"]["self_authorizing"] is False

    output = tmp_path / "protocol.json"
    protocol.write_protocol_lock_no_clobber(output, payload)
    with pytest.raises(FileExistsError):
        protocol.write_protocol_lock_no_clobber(output, payload)

    forged = deepcopy(payload)
    forged["opening_authority"]["independent_opening_approval_present"] = True
    forged["artifact_sha256"] = protocol._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(protocol.FutureProtocolError, match="fabricated"):
        protocol.validate_future_protocol_lock(forged, root=root)


def test_future_protocol_cannot_be_locked_after_boundary() -> None:
    with pytest.raises(protocol.FutureProtocolError, match="before the future boundary"):
        protocol.build_future_protocol_lock(
            locked_at="2026-08-03T00:00:00Z",
            root=Path(".").resolve(),
        )


def test_registered_future_protocol_replays_and_remains_empty() -> None:
    payload = validate_registered_future_protocol(root=Path(".").resolve())
    assert payload["result_state"] == protocol.RESULT_STATE
    assert payload["future_holdout"]["status"] == "EMPTY_NOT_YET_ACQUIRED"
    assert payload["opening_authority"]["independent_protocol_review_present"] is False
    assert all(value is None for value in payload["decision_outputs"].values())
