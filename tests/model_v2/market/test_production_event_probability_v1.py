from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone

import pytest

from lol_kills.v2.market import production_event_probability_v1 as probability


CAPTURED = datetime(2026, 10, 5, 12, 0, tzinfo=timezone.utc)


def _component() -> dict:
    return {
        "locator": "data/lol/v2/evaluation/match-winner-market-v1/fast-event-uncertainty/event.json",
        "raw": b"fast-bytes",
        "fast": {
            "artifact_sha256": "1" * 64,
            "built_at_utc": "2026-10-05T11:59:00+00:00",
            "evaluation_comparator": {"probability_blue": 0.55},
        },
        "candidate": {
            "artifact_sha256": "2" * 64,
            "bootstrap_contract": {"resamples": 2000},
            "point_calculation": {
                "raw_probability_blue": 0.57,
                "recalibration_intercept": 0.01,
                "recalibration_slope": 0.95,
                "probability_blue": 0.56,
            },
            "uncertainty": {
                "draws_sha256": "3" * 64,
                "probability_interval_blue": [0.51, 0.61],
            },
        },
        "event": {
            "event_id": "event-1",
            "series_id": "series-1",
            "game_number": 1,
            "league": "LCS",
            "patch": "26.20",
            "roster_change_stratum": "UNCHANGED",
            "blue_organization_id": "blue",
            "red_organization_id": "red",
            "target_prediction_locator": "target.json",
            "target_prediction_artifact_sha256": "4" * 64,
        },
        "target": {"draft_index": {"sparse_or_new_champion_map": False}},
        "rating": {"event": {"event_start_utc": "2026-10-05T12:30:00+00:00"}},
        "probability": 0.56,
        "interval": [0.51, 0.61],
    }


def _active() -> dict:
    return {
        "receipt": {
            "authority_id": "authority-1",
            "issued_at_utc": "2026-10-05T00:00:00+00:00",
            "valid_until_utc": "2026-10-06T00:00:00+00:00",
        },
        "receipt_raw_sha256": "5" * 64,
        "bindings": {
            "phase_two_evaluation": {"registry_raw_sha256": "6" * 64}
        },
    }


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(probability, "_semantic_authority", lambda **_kwargs: _active())
    monkeypatch.setattr(probability, "_components", lambda **_kwargs: _component())
    monkeypatch.setattr(probability, "_source_locks", lambda _root: [{"raw_sha256": "7" * 64}])
    monkeypatch.setattr(
        probability,
        "_input_binding",
        lambda _component_value, _root: {
            "fast_uncertainty_locator": _component()["locator"],
            "market_protocol_artifact_sha256": probability.REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "generation_source_locator": probability.SOURCE_LOCATOR,
        },
    )
    monkeypatch.setattr(
        probability,
        "validate_registered_match_winner_future_protocol_v1",
        lambda **_kwargs: {
            "artifact_sha256": probability.REGISTERED_PROTOCOL_ARTIFACT_SHA256
        },
    )


def test_production_probability_requires_semantic_authority_and_no_market_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    receipt = probability.build_production_event_probability_v1(
        fast_uncertainty_locator="ignored.json",
        environment={},
        clock=lambda: CAPTURED,
    )
    checked = probability.validate_production_event_probability_v1(
        receipt, environment={}
    )
    assert checked["probability"] == 0.56
    assert checked["probability_interval"] == [0.51, 0.61]
    assert checked["qualification"]["market_price_used_as_model_input"] is False
    assert checked["authority"]["transaction_authority"] is False

    forged = deepcopy(receipt)
    forged["qualification"]["market_price_used_as_model_input"] = True
    forged["artifact_sha256"] = probability._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        probability.ProductionEventProbabilityError,
        match="qualification changed",
    ):
        probability.validate_production_event_probability_v1(
            forged, environment={}
        )
