from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib

import pytest

from lol_kills.v2.market import phase_two_evaluation_v1 as evaluation


def _outcomes(snapshot: dict, rows: list[dict]) -> dict:
    payload = {
        "schema_version": evaluation.OUTCOME_SCHEMA_VERSION,
        "created_at_utc": "2026-10-01T13:00:00+00:00",
        "snapshot_artifact_sha256": snapshot["artifact_sha256"],
        "rows": rows,
    }
    payload["artifact_sha256"] = evaluation._canonical_sha256(payload)
    return payload


def test_outcome_cohort_requires_exact_snapshot_and_hashed_evidence(tmp_path) -> None:
    snapshot = {
        "artifact_sha256": "1" * 64,
        "entries": [
            {
                "event_id": "event-1",
                "series_id": "series-1",
                "game_number": 1,
                "actual_map_start_utc": "2026-10-01T12:00:00+00:00",
            },
            {
                "event_id": "event-2",
                "series_id": "series-1",
                "game_number": 2,
                "actual_map_start_utc": "2026-10-01T12:30:00+00:00",
            },
        ],
    }
    rows = []
    for index, entry in enumerate(snapshot["entries"], start=1):
        locator = (
            evaluation.OUTCOME_EVIDENCE_PREFIX
            / f"event-{index}.json"
        ).as_posix()
        raw = f'{{"officialWinner":"{index}"}}\n'.encode()
        path = tmp_path / locator
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        rows.append(
            {
                **entry,
                "winning_side": "blue" if index == 1 else "red",
                "source_system": "official-results",
                "source_record_id": f"record-{index}",
                "source_revision_id": f"revision-{index}",
                "source_observed_at_utc": "2026-10-01T12:45:00+00:00",
                "evidence_locator": locator,
                "evidence_raw_sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    checked = evaluation.validate_outcome_cohort_v1(
        _outcomes(snapshot, rows), snapshot=snapshot, root=tmp_path
    )
    assert len(checked["rows"]) == 2

    missing = _outcomes(snapshot, rows[:1])
    with pytest.raises(
        evaluation.PhaseTwoEvaluationError,
        match="exact snapshot cohort",
    ):
        evaluation.validate_outcome_cohort_v1(
            missing, snapshot=snapshot, root=tmp_path
        )


def test_capture_gate_uses_all_received_quotes_and_exact_after_start_count() -> None:
    snapshot = {
        "entries": [
            {
                "qualified_quote": True,
                "prediction_to_response_seconds": 2.0,
            },
            {
                "qualified_quote": False,
                "prediction_to_response_seconds": 28.0,
            },
        ],
        "support": {
            "quote_coverage": 0.8,
            "failure_codes": {},
            "quote_received_after_map_start_maps": 0,
            "quote_response_too_late_maps": 1,
        },
    }
    report = evaluation._capture_report(snapshot)
    assert report["prediction_to_quote_response_p95_seconds"] > 26.0
    assert report["quote_received_after_map_start_count"] == 0
    assert report[
        "quote_received_after_or_within_five_seconds_before_start_count"
    ] == 1
    assert report["passed"] is True

    after_start = deepcopy(snapshot)
    after_start["support"]["quote_received_after_map_start_maps"] = 1
    assert evaluation._capture_report(after_start)["passed"] is False


def test_result_validator_rejects_self_registration(monkeypatch) -> None:
    monkeypatch.setattr(
        evaluation.protocol_source,
        "_evaluation_contract",
        lambda: {"frozen": True},
    )
    payload = {
        "schema_version": evaluation.RESULT_SCHEMA_VERSION,
        "result_state": "PHASE_TWO_MARKET_GATE_FAILED_TERMINALLY",
        "run_id": "run-1",
        "evaluated_at_utc": "2026-10-01T13:00:00+00:00",
        "opening_authority_binding": {
            "authority_id": "authority-1",
            "authority_raw_sha256": "1" * 64,
            "opening_marker_locator": (
                "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
                "outcome-opening-markers-v1/run-1.json"
            ),
        },
        "inputs": {
            "snapshot_locator": (
                "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
                "stopping-snapshots-v1/snapshot.json"
            ),
            "snapshot_raw_sha256": "2" * 64,
            "snapshot_artifact_sha256": "3" * 64,
            "outcome_cohort_locator": (
                "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
                "outcomes-v1/outcomes.json"
            ),
            "outcome_cohort_raw_sha256": "4" * 64,
            "outcome_cohort_artifact_sha256": "5" * 64,
            "otherwise_eligible_maps": 600,
            "qualified_quoted_maps": 500,
            "series": 125,
        },
        "evaluation_contract": {"frozen": True},
        "bootstrap": {
            "method": "paired_series_cluster_bootstrap",
            "replicates": evaluation.BOOTSTRAP_REPLICATES,
            "base_seed": evaluation.BOOTSTRAP_SEED,
            "confidence_level": 0.95,
        },
        "primary_probabilistic_gates": {"passed": False},
        "calibration_gates": {"passed": False},
        "capture_gates": {"passed": True},
        "shadow_policy_gates": {"passed": False},
        "phase_two_market_gates_passed": False,
        "independently_registered": False,
        "authority": dict(evaluation.AUTHORITY),
        "claim_ceiling": evaluation.CLAIM_CEILING,
    }
    payload["artifact_sha256"] = evaluation._canonical_sha256(payload)
    assert evaluation.validate_phase_two_evaluation_result_v1(payload) == payload
    payload["independently_registered"] = True
    payload["artifact_sha256"] = evaluation._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        evaluation.PhaseTwoEvaluationError, match="self-registered"
    ):
        evaluation.validate_phase_two_evaluation_result_v1(payload)
