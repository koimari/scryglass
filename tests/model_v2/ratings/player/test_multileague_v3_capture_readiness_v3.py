from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_capture_readiness_v3 as readiness
from lol_kills.v2.ratings.player.multileague_v3_capture_registry_v3 import (
    CaptureReadinessRegistryV3Error,
    validate_registered_capture_readiness_v3,
)


TEST_CLOCK = datetime(2026, 8, 1, 23, 59, 0, tzinfo=timezone.utc)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = readiness._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_system_clocked_capture_contract_is_ready_with_empty_ledger() -> None:
    payload = readiness.build_capture_readiness_v3(
        root=Path(".").resolve(),
        clock=lambda: TEST_CLOCK,
    )
    assert payload["result_state"] == readiness.RESULT_STATE
    assert payload["supersession"]["prediction_receipt_schema_changed"] is True
    assert payload["supersession"]["prediction_ledger_schema_changed"] is True
    assert payload["supersession"]["future_outcomes_used_for_hardening"] is False
    contract = payload["capture_contract"]
    assert contract["prediction_cli_user_timestamp_argument_present"] is False
    assert contract["ledger_builder_user_timestamp_argument_present"] is False
    assert payload["implementation"]["ready_for_pre_event_capture"] is True
    assert payload["implementation"]["actual_future_prediction_evidence_present"] is False
    assert payload["ledger_state"]["entries"] == 0
    assert all(value is False for value in payload["authority"].values())


def test_system_clocked_capture_contract_rejects_boundary_clock() -> None:
    with pytest.raises(readiness.CaptureReadinessV3Error, match="future boundary"):
        readiness.build_capture_readiness_v3(
            root=Path(".").resolve(),
            clock=lambda: datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


def test_capture_readiness_v3_rejects_clock_loophole_or_authority() -> None:
    payload = readiness.build_capture_readiness_v3(
        root=Path(".").resolve(),
        clock=lambda: TEST_CLOCK,
    )
    forged_clock = deepcopy(payload)
    forged_clock["capture_contract"][
        "prediction_cli_user_timestamp_argument_present"
    ] = True
    _resign(forged_clock)
    with pytest.raises(readiness.CaptureReadinessV3Error, match="contract changed"):
        readiness.validate_capture_readiness_v3(
            forged_clock,
            root=Path(".").resolve(),
        )

    forged_authority = deepcopy(payload)
    forged_authority["authority"]["probability_authority"] = True
    _resign(forged_authority)
    with pytest.raises(readiness.CaptureReadinessV3Error, match="exceeds authority"):
        readiness.validate_capture_readiness_v3(
            forged_authority,
            root=Path(".").resolve(),
        )


def test_registered_capture_readiness_v3_replays_with_empty_ledger(
    historical_capture_root: Path,
) -> None:
    payload = validate_registered_capture_readiness_v3(root=historical_capture_root)
    assert payload["result_state"] == readiness.RESULT_STATE
    assert payload["capture_contract"][
        "prediction_cli_user_timestamp_argument_present"
    ] is False
    assert payload["capture_contract"][
        "ledger_builder_user_timestamp_argument_present"
    ] is False
    assert payload["implementation"]["ready_for_pre_event_capture"] is True
    assert payload["ledger_state"]["entries"] == 0
    assert all(value is False for value in payload["authority"].values())


def test_registered_capture_readiness_v3_rejects_current_source_drift() -> None:
    with pytest.raises(
        CaptureReadinessRegistryV3Error,
        match="capture readiness source drifted",
    ):
        validate_registered_capture_readiness_v3(root=Path(".").resolve())
