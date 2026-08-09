from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import event_probability_v2 as probability
from lol_kills.v2.market import phase_two_opening_v1 as opening


def _install(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    candidate = {
        "artifact_sha256": "1" * 64,
        "event": {
            "event_id": "event-1",
            "series_id": "series-1",
            "game_number": 1,
            "league": "LCS",
            "patch": "26.17",
            "roster_change_stratum": "UNCHANGED",
            "blue_organization_id": "blue-team",
            "red_organization_id": "red-team",
            "target_prediction_locator": "prediction.json",
            "target_prediction_artifact_sha256": "2" * 64,
        },
        "point_calculation": {
            "raw_probability_blue": 0.75,
            "recalibration_intercept": 0.1,
            "recalibration_slope": 1.1,
            "probability_blue": 0.8,
        },
        "bootstrap_contract": {"resamples": 2_000},
        "uncertainty": {
            "draws_sha256": "3" * 64,
            "probability_interval_blue": [0.2, 0.4],
        },
    }
    fast = {
        "artifact_sha256": "4" * 64,
        "built_at_utc": "2026-09-01T15:00:00+00:00",
        "frozen_contract_candidate": candidate,
        "decomposition": {"rating_bootstrap_locator": "rating.json"},
        "evaluation_comparator": {
            "model": "recalibrated_rating_only",
            "raw_probability_blue": 0.7,
            "recalibration_intercept": 0.0,
            "recalibration_slope": 1.0,
            "probability_blue": 0.7,
            "phase_two_market_price_used": False,
            "target_event_outcome_used": False,
        },
    }
    active = {
        "authority": {"authority_id": "opening-1"},
        "authority_raw_sha256": "5" * 64,
        "marker_raw_sha256": "6" * 64,
        "marker": {"opened_at_utc": "2026-09-01T14:00:00+00:00"},
    }
    rating = {"event": {"event_start_utc": "2026-09-01T18:00:00+00:00"}}
    monkeypatch.setattr(
        probability.opening,
        "validate_active_phase_two_opening",
        lambda **_kwargs: active,
    )
    monkeypatch.setattr(
        probability,
        "_uncertainty",
        lambda _root, _locator, _environment: ("fast.json", b"fast", fast),
    )
    monkeypatch.setattr(
        probability.fast_uncertainty,
        "_rating_artifact",
        lambda _root, _locator, _environment: ("rating.json", b"rating", rating),
    )
    monkeypatch.setattr(
        probability.fast_uncertainty.frozen,
        "_target",
        lambda _root, _locator: (
            b"target",
            {"draft_index": {"sparse_or_new_champion_map": False}},
            {},
            {},
        ),
    )
    monkeypatch.setattr(
        probability,
        "_source_locks",
        lambda _root: [{"locator": "probability.py", "bytes": 1, "raw_sha256": "7" * 64}],
    )
    monkeypatch.setattr(
        probability.evaluation,
        "_sha256_path",
        lambda _path: "8" * 64,
    )
    return candidate, fast, active


def _receipt(monkeypatch: pytest.MonkeyPatch) -> dict:
    candidate, fast, active = _install(monkeypatch)
    interval = candidate["uncertainty"]["probability_interval_blue"]
    point = candidate["point_calculation"]
    payload = {
        "schema_version": probability.RECEIPT_SCHEMA_VERSION,
        "result_state": probability.RESULT_STATE,
        "captured_at_utc": "2026-09-01T15:00:01+00:00",
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": "2026-09-01T15:00:01+00:00",
            "user_supplied_timestamp_allowed": False,
        },
        "event": {
            "event_id": "event-1",
            "series_id": "series-1",
            "game_number": 1,
            "league": "LCS",
            "patch": "26.17",
            "roster_change_stratum": "UNCHANGED",
            "sparse_or_new_champion_map": False,
            "market_type": probability.MARKET_TYPE,
            "selection": "winner:blue-team",
            "opposing_selection": "winner:red-team",
        },
        "opening_binding": {
            "authority_id": "opening-1",
            "authority_raw_sha256": active["authority_raw_sha256"],
            "marker_locator": opening.MARKER_LOCATOR.as_posix(),
            "marker_raw_sha256": active["marker_raw_sha256"],
            "opened_at_utc": active["marker"]["opened_at_utc"],
            "outcome_free_phase_two_collection_active": True,
        },
        "input_binding": {
            "fast_uncertainty_locator": "fast.json",
            "fast_uncertainty_raw_sha256": probability._sha256_bytes(b"fast"),
            "fast_uncertainty_artifact_sha256": fast["artifact_sha256"],
            "frozen_contract_candidate_artifact_sha256": candidate[
                "artifact_sha256"
            ],
            "target_prediction_locator": "prediction.json",
            "target_prediction_artifact_sha256": "2" * 64,
            "market_protocol_artifact_sha256": probability.REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "generation_source_locator": probability.SOURCE_LOCATOR,
            "generation_source_raw_sha256": "8" * 64,
        },
        "calculation": {
            "method": "bounded_logistic_recalibration",
            "raw_model_probability": point["raw_probability_blue"],
            "calibration_intercept": point["recalibration_intercept"],
            "calibration_slope": point["recalibration_slope"],
            "probability": 0.8,
            "opposing_probability": 1.0 - 0.8,
            "rating_only_comparator": dict(fast["evaluation_comparator"]),
        },
        "uncertainty": {
            "method": "series_cluster_bootstrap_full_prediction_pipeline",
            "confidence_level": 0.95,
            "resamples": 2_000,
            "draws_sha256": "3" * 64,
            "probability_interval": interval,
            "opposing_probability_interval": [0.6, 0.8],
            "point_inside_percentile_interval": False,
            "point_containment_required": False,
            "interval_is_epistemic": True,
            "interval_is_not_binary_outcome_coverage_guarantee": True,
        },
        "qualification": {
            "phase_two_opening_active": True,
            "phase_one_models_independently_passed": True,
            "recalibration_and_fast_uncertainty_independently_registered": True,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
            "market_price_used_as_model_input": False,
            "independently_registered": False,
        },
        "source_locks": probability._source_locks(None),
        "authority": dict(probability.AUTHORITY),
        "claim_ceiling": probability.CLAIM_CEILING,
    }
    payload["artifact_sha256"] = probability._canonical_sha256(payload)
    return payload


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = probability._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_v2_accepts_valid_percentile_interval_that_excludes_plugin_point(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt(monkeypatch)
    checked = probability.validate_event_probability_v2(receipt)
    assert checked["probability"] == 0.8
    assert checked["probability_interval"] == [0.2, 0.4]
    assert checked["uncertainty"]["point_inside_percentile_interval"] is False
    assert checked["uncertainty"]["point_containment_required"] is False
    assert checked["authority"]["probability_authority"] is False


def test_v2_rejects_forged_containment_opening_or_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    containment = deepcopy(_receipt(monkeypatch))
    containment["uncertainty"]["point_inside_percentile_interval"] = True
    _resign(containment)
    with pytest.raises(
        probability.EventProbabilityV2Error,
        match="uncertainty changed",
    ):
        probability.validate_event_probability_v2(containment)

    authority = deepcopy(_receipt(monkeypatch))
    authority["authority"]["probability_authority"] = True
    _resign(authority)
    with pytest.raises(
        probability.EventProbabilityV2Error,
        match="exceeds authority",
    ):
        probability.validate_event_probability_v2(authority)
