from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from lol_kills.v2.market import phase_one_evaluation_registry_v1 as evaluation_registry
from lol_kills.v2.market import phase_one_recalibration_v1 as recalibration
from lol_kills.v2.market.match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
)


ROOT = Path(__file__).resolve().parents[3]


def test_bounded_recalibration_uses_exact_protocol_optimizer() -> None:
    raw = np.tile([0.20, 0.40, 0.60, 0.80], 100).tolist()
    outcomes = np.tile([0, 0, 1, 1], 100).tolist()
    result = recalibration.fit_bounded_recalibration(raw, outcomes)

    assert -2.0 <= result["intercept"] <= 2.0
    assert 0.25 <= result["slope"] <= 4.0
    assert result["map_log_loss"] <= result["identity_map_log_loss"]
    assert result["convergence"]["success"] is True
    assert result["convergence"]["status"] == 0


def test_recalibration_rejects_one_class_or_out_of_domain_inputs() -> None:
    with pytest.raises(recalibration.PhaseOneRecalibrationError, match="domain"):
        recalibration.fit_bounded_recalibration([0.4, 0.6], [1, 1])
    with pytest.raises(recalibration.PhaseOneRecalibrationError, match="domain"):
        recalibration.fit_bounded_recalibration([0.0, 0.6], [0, 1])


def artifact() -> dict:
    fit = recalibration.fit_bounded_recalibration(
        np.tile([0.20, 0.40, 0.60, 0.80], 50).tolist(),
        np.tile([0, 0, 1, 1], 50).tolist(),
    )
    payload = {
        "schema_version": recalibration.SCHEMA_VERSION,
        "result_state": recalibration.RESULT_STATE,
        "fitted_at_utc": "2026-12-03T00:00:00+00:00",
        "inputs": {
            "phase_one_result_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/evaluations/run-1.json",
            "phase_one_result_raw_sha256": "1" * 64,
            "phase_one_result_artifact_sha256": "2" * 64,
            "phase_one_evaluation_registry_locator": evaluation_registry.REGISTRY_LOCATOR.as_posix(),
            "phase_one_evaluation_registry_raw_sha256": "3" * 64,
            "phase_one_evaluation_registry_id": "registry-1",
            "joint_snapshot_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/snapshots/snapshot-1.json",
            "joint_snapshot_raw_sha256": "4" * 64,
            "joint_snapshot_artifact_sha256": "5" * 64,
            "outcome_cohort_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/outcomes/cohort-1.json",
            "outcome_cohort_raw_sha256": "6" * 64,
            "outcome_cohort_artifact_sha256": "7" * 64,
            "maps": 1000,
            "series": 250,
        },
        "protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "recalibration_contract": recalibration._validate_protocol(ROOT)[
                "recalibration"
            ],
        },
        "optimization_contract": recalibration._optimization_contract(),
        "models": {
            "ratings_plus_draft": {
                "raw_probability_field": "ratings_plus_draft.p_blue",
                **fit,
            },
            "ratings_only": {
                "raw_probability_field": "ratings_only.p_blue",
                **fit,
            },
        },
        "qualification": {
            "phase_one_evaluation_independently_registered": True,
            "phase_one_models_independently_passed": True,
            "complete_phase_one_draft_cohort_used": True,
            "unweighted_map_objective_used": True,
            "candidate_or_hyperparameter_reselection_performed": False,
            "market_price_used": False,
            "phase_two_event_outcome_used": False,
            "phase_two_started": False,
            "independently_registered": False,
        },
        "authority": recalibration.AUTHORITY,
        "claim_ceiling": recalibration.CLAIM_CEILING,
    }
    payload["artifact_sha256"] = recalibration._canonical_sha256(payload)
    return payload


def test_recalibration_artifact_replays_bounds_and_convergence() -> None:
    value = artifact()
    checked = recalibration.validate_phase_one_recalibration_artifact(value)
    assert checked["models"]["ratings_plus_draft"]["convergence"]["success"] is True
    assert all(authority is False for authority in checked["authority"].values())

    changed = deepcopy(value)
    changed["models"]["ratings_plus_draft"]["slope"] = 9.0
    changed["artifact_sha256"] = recalibration._canonical_sha256(
        {key: item for key, item in changed.items() if key != "artifact_sha256"}
    )
    with pytest.raises(recalibration.PhaseOneRecalibrationError, match="invalid"):
        recalibration.validate_phase_one_recalibration_artifact(changed)
