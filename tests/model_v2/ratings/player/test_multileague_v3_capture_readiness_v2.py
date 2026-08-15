from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_capture_readiness_v2 as readiness
from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v2 import (
    CaptureReadinessRegistryV2Error,
    validate_registered_capture_readiness_v2,
)


TEST_CLOCK = datetime(2026, 8, 1, 23, 57, 0, tzinfo=timezone.utc)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = readiness._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_clock_corrected_capture_path_is_ready_while_ledger_is_empty() -> None:
    payload = readiness.build_capture_readiness_v2(
        root=Path(".").resolve(),
        clock=lambda: TEST_CLOCK,
    )
    assert payload["result_state"] == readiness.RESULT_STATE
    assert payload["clock_attestation"]["user_supplied_timestamp_allowed"] is False
    assert payload["supersession"][
        "rejected_capture_qualifies_as_future_evidence"
    ] is False
    assert payload["implementation"]["ready_for_pre_event_capture"] is True
    assert payload["implementation"]["actual_future_prediction_evidence_present"] is False
    assert payload["ledger_state"]["entries"] == 0
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())


def test_clock_corrected_capture_path_rejects_future_builder_clock() -> None:
    with pytest.raises(readiness.CaptureReadinessV2Error, match="future boundary"):
        readiness.build_capture_readiness_v2(
            root=Path(".").resolve(),
            clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


def test_capture_readiness_v2_rejects_fabricated_evidence_or_authority() -> None:
    payload = readiness.build_capture_readiness_v2(
        root=Path(".").resolve(),
        clock=lambda: TEST_CLOCK,
    )
    forged_evidence = deepcopy(payload)
    forged_evidence["implementation"]["actual_future_prediction_evidence_present"] = True
    _resign(forged_evidence)
    with pytest.raises(readiness.CaptureReadinessV2Error, match="status changed"):
        readiness.validate_capture_readiness_v2(
            forged_evidence,
            root=Path(".").resolve(),
        )

    forged_authority = deepcopy(payload)
    forged_authority["authority"]["probability_authority"] = True
    _resign(forged_authority)
    with pytest.raises(readiness.CaptureReadinessV2Error, match="exceeds authority"):
        readiness.validate_capture_readiness_v2(
            forged_authority,
            root=Path(".").resolve(),
        )


def test_v2_capture_receipt_replays_on_sealed_sources(
    historical_capture_root: Path,
) -> None:
    """Replay the archived receipt only against its sealed source bytes."""
    payload = validate_registered_capture_readiness_v2(root=historical_capture_root)
    assert payload["result_state"] == readiness.RESULT_STATE
    assert payload["clock_attestation"]["user_supplied_timestamp_allowed"] is False
    assert payload["supersession"]["rejected_capture_qualifies_as_future_evidence"] is False
    assert payload["implementation"]["ready_for_pre_event_capture"] is True
    assert all(value is False for value in payload["authority"].values())


def test_v2_capture_receipt_rejects_current_source_drift() -> None:
    with pytest.raises(
        CaptureReadinessRegistryV2Error,
        match="capture readiness source drifted",
    ):
        validate_registered_capture_readiness_v2(root=Path(".").resolve())
