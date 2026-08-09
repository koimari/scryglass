from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import event_probability_v1 as probability


NOW = datetime(2026, 8, 4, 12, 0, tzinfo=timezone.utc)
MODEL_SHA = "a" * 64
PROTOCOL_SHA = "b" * 64
CALIBRATION_SHA = "c" * 64
UNCERTAINTY_SHA = "d" * 64
GENERATOR_SHA = "e" * 64


def receipt() -> dict:
    return probability.build_event_probability_receipt(
        event_id="series-1-map-1",
        league="LCS",
        market_type="match_winner",
        selection="winner:blue-team",
        opposing_selection="winner:red-team",
        model_artifact_sha256=MODEL_SHA,
        market_protocol_artifact_sha256=PROTOCOL_SHA,
        calibration_artifact_sha256=CALIBRATION_SHA,
        uncertainty_artifact_sha256=UNCERTAINTY_SHA,
        source_prediction_receipt_sha256="f" * 64,
        source_prediction_registry_sha256="1" * 64,
        generation_code_sha256=GENERATOR_SHA,
        raw_model_probability=0.60,
        calibration_intercept=-0.05,
        calibration_slope=0.90,
        probability_interval=(0.54, 0.65),
        uncertainty_draws_sha256="2" * 64,
        uncertainty_resamples=2000,
        clock=lambda: NOW,
    )


def registry(value: dict) -> dict:
    return probability.build_event_probability_registry(
        receipts=[
            (
                "data/lol/v2/evaluation/match-winner-market-v1/event-probabilities/series-1-map-1-blue.json",
                value,
            )
        ],
        registry_id="probability-registry-1",
        independent_reviewer_id="reviewer-1",
        issued_at=(NOW + timedelta(seconds=1)).isoformat(),
        model_artifact_sha256=MODEL_SHA,
        market_protocol_artifact_sha256=PROTOCOL_SHA,
        calibration_artifact_sha256=CALIBRATION_SHA,
        uncertainty_artifact_sha256=UNCERTAINTY_SHA,
        generation_code_sha256=GENERATOR_SHA,
    )


def registered(value: dict | None = None) -> dict:
    value = value or receipt()
    ledger = registry(value)
    return probability.validate_registered_event_probability(
        receipt=value,
        expected_receipt_sha256=probability.sha256_json(value),
        registry=ledger,
        expected_registry_sha256=probability.sha256_json(ledger),
        event_id="series-1-map-1",
        league="LCS",
        market_type="match_winner",
        selection="winner:blue-team",
        opposing_selection="winner:red-team",
        model_artifact_sha256=MODEL_SHA,
        market_protocol_artifact_sha256=PROTOCOL_SHA,
        calibration_artifact_sha256=CALIBRATION_SHA,
        uncertainty_artifact_sha256=UNCERTAINTY_SHA,
        generation_code_sha256=GENERATOR_SHA,
        as_of=NOW + timedelta(seconds=2),
    )


def test_registered_probability_replays_calibration_and_interval() -> None:
    checked = registered()

    assert 0.0 < checked["probability"] < 1.0
    assert checked["probability_interval"] == [0.54, 0.65]
    assert checked["authority"]["probability_authority"] is False
    assert checked["registry_id"] == "probability-registry-1"


def test_receipt_cannot_self_register() -> None:
    value = receipt()
    ledger = registry(value)

    with pytest.raises(probability.EventProbabilityError, match="not registered"):
        probability.validate_registered_event_probability(
            receipt=value,
            expected_receipt_sha256=probability.sha256_json(value),
            registry=ledger,
            expected_registry_sha256=None,
            event_id="series-1-map-1",
            league="LCS",
            market_type="match_winner",
            selection="winner:blue-team",
            opposing_selection="winner:red-team",
            model_artifact_sha256=MODEL_SHA,
            market_protocol_artifact_sha256=PROTOCOL_SHA,
            calibration_artifact_sha256=CALIBRATION_SHA,
            uncertainty_artifact_sha256=UNCERTAINTY_SHA,
            generation_code_sha256=GENERATOR_SHA,
            as_of=NOW + timedelta(seconds=2),
        )


def test_registry_detects_receipt_tampering() -> None:
    value = receipt()
    ledger = registry(value)
    expected_receipt = probability.sha256_json(value)
    value["uncertainty"]["draws_sha256"] = "3" * 64

    with pytest.raises(
        probability.EventProbabilityError,
        match="artifact hash changed|digest mismatch",
    ):
        probability.validate_registered_event_probability(
            receipt=value,
            expected_receipt_sha256=expected_receipt,
            registry=ledger,
            expected_registry_sha256=probability.sha256_json(ledger),
            event_id="series-1-map-1",
            league="LCS",
            market_type="match_winner",
            selection="winner:blue-team",
            opposing_selection="winner:red-team",
            model_artifact_sha256=MODEL_SHA,
            market_protocol_artifact_sha256=PROTOCOL_SHA,
            calibration_artifact_sha256=CALIBRATION_SHA,
            uncertainty_artifact_sha256=UNCERTAINTY_SHA,
            generation_code_sha256=GENERATOR_SHA,
            as_of=NOW + timedelta(seconds=2),
        )


def test_rehashed_receipt_cannot_fake_calculation() -> None:
    value = deepcopy(receipt())
    value["calculation"]["probability"] = 0.99
    unsigned = dict(value)
    unsigned.pop("artifact_sha256")
    value["artifact_sha256"] = probability.sha256_json(unsigned)

    with pytest.raises(probability.EventProbabilityError, match="does not replay"):
        probability.validate_event_probability_receipt(value)


def test_malformed_calibration_fails_as_typed_error() -> None:
    value = deepcopy(receipt())
    value["calculation"]["calibration_slope"] = "not-a-number"
    unsigned = dict(value)
    unsigned.pop("artifact_sha256")
    value["artifact_sha256"] = probability.sha256_json(unsigned)

    with pytest.raises(probability.EventProbabilityError, match="not numeric"):
        probability.validate_event_probability_receipt(value)


def test_registry_is_event_and_selection_specific() -> None:
    with pytest.raises(probability.EventProbabilityError, match="unavailable"):
        value = receipt()
        ledger = registry(value)
        probability.validate_registered_event_probability(
            receipt=value,
            expected_receipt_sha256=probability.sha256_json(value),
            registry=ledger,
            expected_registry_sha256=probability.sha256_json(ledger),
            event_id="series-1-map-2",
            league="LCS",
            market_type="match_winner",
            selection="winner:blue-team",
            opposing_selection="winner:red-team",
            model_artifact_sha256=MODEL_SHA,
            market_protocol_artifact_sha256=PROTOCOL_SHA,
            calibration_artifact_sha256=CALIBRATION_SHA,
            uncertainty_artifact_sha256=UNCERTAINTY_SHA,
            generation_code_sha256=GENERATOR_SHA,
            as_of=NOW + timedelta(seconds=2),
        )


def test_loader_replays_receipt_from_pinned_registry(tmp_path: Path) -> None:
    value = receipt()
    ledger = registry(value)
    receipt_locator = ledger["entries"][0]["receipt_locator"]
    registry_locator = probability.DEFAULT_REGISTRY.as_posix()
    for locator, payload in (
        (receipt_locator, value),
        (registry_locator, ledger),
    ):
        path = tmp_path / locator
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = probability.load_registered_event_probability(
        registry_locator=registry_locator,
        expected_registry_sha256=probability.sha256_json(ledger),
        event_id="series-1-map-1",
        league="LCS",
        market_type="match_winner",
        selection="winner:blue-team",
        opposing_selection="winner:red-team",
        model_artifact_sha256=MODEL_SHA,
        market_protocol_artifact_sha256=PROTOCOL_SHA,
        calibration_artifact_sha256=CALIBRATION_SHA,
        uncertainty_artifact_sha256=UNCERTAINTY_SHA,
        generation_code_sha256=GENERATOR_SHA,
        as_of=NOW + timedelta(seconds=2),
        root=tmp_path,
    )
    assert loaded["status"] == "registered"
    assert loaded["receipt_sha256"] == probability.sha256_json(value)
    assert loaded["probability_interval"] == [0.54, 0.65]
