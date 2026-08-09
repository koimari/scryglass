from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_future_protocol_v3 as protocol
from lol_kills.v2.ratings.player.multileague_v3_registry_v3 import (
    validate_registered_future_protocol_v3,
)


TEST_CLOCK = datetime(2026, 8, 2, 2, 30, 0, tzinfo=timezone.utc)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = protocol._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_clock_checked_protocol_rejects_old_timing_without_changing_candidate() -> None:
    payload = protocol.build_future_protocol_lock_v3(
        root=Path(".").resolve(),
        clock=lambda: TEST_CLOCK,
    )
    assert payload["result_state"] == protocol.RESULT_STATE
    assert payload["locked_at_utc"] == TEST_CLOCK.isoformat()
    assert payload["clock_attestation"]["user_supplied_timestamp_allowed"] is False
    assert payload["supersession"][
        "rejected_artifacts_qualify_as_future_evidence"
    ] is False
    assert payload["supersession"]["candidate_changed"] is False
    assert payload["future_holdout"]["status"] == "EMPTY_NOT_YET_ACQUIRED"
    assert payload["prediction_ledger"]["entries"] == 0
    assert all(value is False for value in payload["authority"].values())


def test_clock_checked_protocol_rejects_future_builder_clock() -> None:
    with pytest.raises(protocol.FutureProtocolV3Error, match="future boundary"):
        protocol.build_future_protocol_lock_v3(
            root=Path(".").resolve(),
            clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


def test_clock_checked_protocol_rejects_rehabilitated_lineage_or_authority() -> None:
    payload = protocol.build_future_protocol_lock_v3(
        root=Path(".").resolve(),
        clock=lambda: TEST_CLOCK,
    )
    forged = deepcopy(payload)
    forged["supersession"]["rejected_artifacts_qualify_as_future_evidence"] = True
    _resign(forged)
    with pytest.raises(protocol.FutureProtocolV3Error, match="lineage changed"):
        protocol.validate_future_protocol_lock_v3(
            forged,
            root=Path(".").resolve(),
        )

    forged_authority = deepcopy(payload)
    forged_authority["authority"]["probability_authority"] = True
    _resign(forged_authority)
    with pytest.raises(protocol.FutureProtocolV3Error, match="exceeds authority"):
        protocol.validate_future_protocol_lock_v3(
            forged_authority,
            root=Path(".").resolve(),
        )


def test_registered_clock_corrected_protocol_replays_and_remains_empty() -> None:
    payload = validate_registered_future_protocol_v3(root=Path(".").resolve())
    assert payload["result_state"] == protocol.RESULT_STATE
    assert payload["clock_attestation"]["user_supplied_timestamp_allowed"] is False
    assert payload["future_holdout"]["status"] == "EMPTY_NOT_YET_ACQUIRED"
    assert payload["prediction_ledger"]["entries"] == 0
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())
