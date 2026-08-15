from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from lol_kills.v2.ratings.player import multileague_v3_temporal_failure_v2 as failure
from lol_kills.v2.ratings.player.multileague_v3_temporal_failure_registry import (
    validate_registered_temporal_failure,
)


OBSERVED_AT = "2026-08-01T23:48:55.873475+00:00"


def _use_recorded_creation_time(monkeypatch: pytest.MonkeyPatch) -> None:
    original_read_target = failure._read_target
    recorded_epoch = datetime.fromisoformat(OBSERVED_AT).timestamp() - 1

    def read_target(root: Path, spec: dict[str, str]) -> tuple[object, dict]:
        _path, payload = original_read_target(root, spec)
        historical_path = SimpleNamespace(
            stat=lambda: SimpleNamespace(st_birthtime=recorded_epoch)
        )
        return historical_path, payload

    monkeypatch.setattr(failure, "_read_target", read_target)


def test_future_dated_receipts_are_rejected_without_future_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_recorded_creation_time(monkeypatch)
    payload = failure.build_temporal_failure_receipt(
        observed_at=OBSERVED_AT,
        root=Path(".").resolve(),
    )
    assert payload["result_state"] == failure.RESULT_STATE
    assert len(payload["failures"]) == 3
    assert all(
        item["filesystem_created_at_utc"] < item["declared_time_utc"]
        for item in payload["failures"]
    )
    assert payload["policy"]["artifacts_qualify_as_future_evidence"] is False
    assert payload["outcome_access"]["future_holdout_targets_accessed"] is False
    assert all(value is False for value in payload["authority"].values())


def test_temporal_failure_rejects_rehabilitation_or_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_recorded_creation_time(monkeypatch)
    payload = failure.build_temporal_failure_receipt(
        observed_at=OBSERVED_AT,
        root=Path(".").resolve(),
    )
    forged = deepcopy(payload)
    forged["policy"]["artifacts_qualify_as_future_evidence"] = True
    forged["artifact_sha256"] = failure._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(failure.TemporalFailureError, match="policy changed"):
        failure.validate_temporal_failure_receipt(
            forged,
            root=Path(".").resolve(),
        )


def test_registered_temporal_failure_preserves_all_three_rejections() -> None:
    payload = validate_registered_temporal_failure(root=Path(".").resolve())
    assert payload["result_state"] == failure.RESULT_STATE
    assert [item["kind"] for item in payload["failures"]] == [
        "corrected_source_preflight_v2",
        "future_protocol_v2",
        "capture_readiness_v1",
    ]
