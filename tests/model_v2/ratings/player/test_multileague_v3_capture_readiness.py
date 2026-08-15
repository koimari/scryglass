from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_capture_readiness as readiness
from lol_kills.v2.ratings.player.multileague_v3_capture_registry import (
    CaptureReadinessRegistryError,
    validate_registered_capture_readiness,
)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = readiness._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_capture_implementation_is_ready_while_ledger_remains_empty(
    historical_capture_root: Path,
) -> None:
    payload = readiness.build_capture_readiness(
        locked_at="2026-08-02T00:10:00Z",
        root=historical_capture_root,
    )
    assert payload["result_state"] == readiness.RESULT_STATE
    assert payload["implementation"]["ready_for_pre_event_capture"] is True
    assert payload["implementation"]["actual_future_prediction_evidence_present"] is False
    assert payload["ledger_state"]["entries"] == 0
    assert payload["ledger_state"]["metadata_support_met"] is False
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())


def test_capture_readiness_rejects_fabricated_evidence_or_authority(
    historical_capture_root: Path,
) -> None:
    payload = readiness.build_capture_readiness(
        locked_at="2026-08-02T00:10:00Z",
        root=historical_capture_root,
    )
    forged_evidence = deepcopy(payload)
    forged_evidence["implementation"]["actual_future_prediction_evidence_present"] = True
    _resign(forged_evidence)
    with pytest.raises(readiness.CaptureReadinessError, match="status changed"):
        readiness.validate_capture_readiness(
            forged_evidence,
            root=historical_capture_root,
        )

    forged_authority = deepcopy(payload)
    forged_authority["authority"]["probability_authority"] = True
    _resign(forged_authority)
    with pytest.raises(readiness.CaptureReadinessError, match="exceeds authority"):
        readiness.validate_capture_readiness(
            forged_authority,
            root=historical_capture_root,
        )


def test_v1_capture_receipt_replays_only_on_archived_sources(
    historical_capture_root: Path,
    private_test_root: Path,
) -> None:
    payload = validate_registered_capture_readiness(root=historical_capture_root)
    assert payload["result_state"] == readiness.RESULT_STATE
    assert payload["implementation"]["ready_for_pre_event_capture"] is True
    assert payload["ledger_state"]["entries"] == 0
    assert all(value is False for value in payload["authority"].values())
    with pytest.raises(CaptureReadinessRegistryError, match="source drifted"):
        validate_registered_capture_readiness(root=private_test_root)
