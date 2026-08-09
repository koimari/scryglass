from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import phase_two_evaluation_readiness_v1 as readiness


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        readiness,
        "_dependencies",
        lambda _root, _environment={}: {
            "phase_two_collection_readiness": {"artifact_sha256": "1" * 64},
            "match_winner_future_protocol": {"artifact_sha256": "2" * 64},
        },
    )
    monkeypatch.setattr(
        readiness,
        "_contract",
        lambda _root=readiness.ROOT: {
            "bootstrap": {
                "replicates": 10_000,
                "seed": 20_260_806,
            },
            "opening_contract": {
                "marker_written_before_first_outcome_read": True
            },
        },
    )
    monkeypatch.setattr(
        readiness,
        "_empty_state",
        lambda _root: {
            "outcome_cohorts": 0,
            "outcome_evidence": 0,
            "opening_markers": 0,
            "evaluation_results": 0,
            "outcome_opening_authority_present": False,
            "evaluation_registry_present": False,
            "outcomes_accessed": False,
        },
    )
    monkeypatch.setattr(
        readiness,
        "_source_record",
        lambda _root, locator: {
            "locator": locator,
            "bytes": 1,
            "raw_sha256": "3" * 64,
        },
    )


def _receipt(monkeypatch: pytest.MonkeyPatch) -> dict:
    _install(monkeypatch)
    return readiness.build_phase_two_evaluation_readiness_v1(
        environment={},
        clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = readiness._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_readiness_freezes_evaluator_opening_and_registry_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(monkeypatch)
    checked = readiness.validate_phase_two_evaluation_readiness_v1(
        receipt, environment={}
    )
    assert checked["evaluation_contract"]["bootstrap"] == {
        "replicates": 10_000,
        "seed": 20_260_806,
    }
    assert checked["locked_empty_state"]["outcomes_accessed"] is False
    assert all(value is False for value in checked["authority"].values())
    assert all(value is None for value in checked["decision_outputs"].values())


def test_readiness_rejects_betting_authority_forgery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forged = deepcopy(_receipt(monkeypatch))
    forged["authority"]["betting_authority"] = True
    _resign(forged)
    with pytest.raises(
        readiness.PhaseTwoEvaluationReadinessError,
        match="exceeds authority",
    ):
        readiness.validate_phase_two_evaluation_readiness_v1(
            forged, environment={}
        )
