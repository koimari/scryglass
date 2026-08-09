from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import betano_br_quote_adapter_v2 as adapter


def _probability() -> dict:
    return {
        "captured_at_utc": "2026-09-01T15:00:00+00:00",
        "artifact_sha256": "1" * 64,
        "receipt_sha256": "2" * 64,
        "probability": 0.8,
        "probability_interval": [0.2, 0.4],
        "event": {
            "event_id": "event-1",
            "league": "LCS",
            "market_type": "match_winner",
            "selection": "winner:blue-team",
            "opposing_selection": "winner:red-team",
        },
        "opening_binding": {"marker_raw_sha256": "3" * 64},
        "input_binding": {
            "target_prediction_artifact_sha256": "4" * 64,
            "market_protocol_artifact_sha256": "5" * 64,
            "frozen_contract_candidate_artifact_sha256": "6" * 64,
            "fast_uncertainty_artifact_sha256": "7" * 64,
            "generation_source_raw_sha256": "8" * 64,
        },
        "calculation": {
            "raw_model_probability": 0.8,
            "calibration_intercept": 0.0,
            "calibration_slope": 1.0,
            "probability": 0.8,
        },
        "uncertainty": {
            "draws_sha256": "9" * 64,
            "resamples": 2_000,
            "probability_interval": [0.2, 0.4],
        },
    }


def test_transport_bridge_never_changes_v2_point_or_evidence_interval() -> None:
    probability = _probability()
    bridge = adapter._transport_bridge(probability)
    assert bridge["original_v2_probability"] == 0.8
    assert bridge["original_v2_probability_interval"] == [0.2, 0.4]
    assert bridge["bridge_probability_interval"] == [0.2, 0.8]
    assert bridge[
        "bridge_interval_widened_only_for_legacy_transport_shape"
    ] is True
    assert bridge["bridge_interval_used_by_transport_extraction_or_quote"] is False
    assert bridge["bridge_persisted_as_probability_evidence"] is False
    assert bridge["receipt"]["calculation"]["probability"] == pytest.approx(0.8)
    assert all(value is False for value in bridge["bridge_authority"].values())


def _bundle(monkeypatch: pytest.MonkeyPatch) -> dict:
    probability = _probability()
    bridge = adapter._transport_bridge(probability)
    quote = {
        "captured_at_utc": "2026-09-01T15:00:02+00:00",
        "transport": {
            "request_started_at_utc": "2026-09-01T15:00:01+00:00"
        },
        "prediction_binding": {
            "event_probability_receipt_sha256": bridge["receipt_sha256"],
            "event_probability_artifact_sha256": bridge["receipt"][
                "artifact_sha256"
            ],
            "prediction_captured_at_utc": probability["captured_at_utc"],
            "scryglass_event_id": "event-1",
            "selection": "winner:blue-team",
            "opposing_selection": "winner:red-team",
        },
    }
    plan = {
        "artifact_sha256": "b" * 64,
        "planned_at_utc": "2026-09-01T15:00:01+00:00",
        "probability_binding": {
            "locator": "probability.json",
            "raw_sha256": adapter._sha256_bytes(b"probability"),
            "artifact_sha256": probability["artifact_sha256"],
        },
    }
    monkeypatch.setattr(
        adapter,
        "_probability",
        lambda **_kwargs: ("probability.json", b"probability", probability),
    )
    monkeypatch.setattr(
        adapter.evaluation,
        "_locator",
        lambda value, _prefix, _field: str(value),
    )
    monkeypatch.setattr(
        adapter.evaluation,
        "_read_regular",
        lambda _root, _locator, _label: b"plan",
    )
    monkeypatch.setattr(
        adapter.evaluation,
        "_strict_object",
        lambda _raw, _label: {},
    )
    monkeypatch.setattr(
        adapter.event_plan,
        "validate_phase_two_event_plan_v1",
        lambda _payload, **_kwargs: plan,
    )
    monkeypatch.setattr(
        adapter.transport,
        "validate_betano_map_winner_quote_v1",
        lambda _quote, **_kwargs: _quote,
    )
    monkeypatch.setattr(
        adapter,
        "_source_locks",
        lambda _root: [{"locator": "adapter-v2.py", "bytes": 1, "raw_sha256": "a" * 64}],
    )
    payload = {
        "schema_version": adapter.SCHEMA_VERSION,
        "result_state": adapter.RESULT_STATE,
        "captured_at_utc": quote["captured_at_utc"],
        "event_probability_v2_binding": {
            "locator": "probability.json",
            "raw_sha256": adapter._sha256_bytes(b"probability"),
            "artifact_sha256": probability["artifact_sha256"],
            "receipt_sha256": probability["receipt_sha256"],
            "captured_at_utc": probability["captured_at_utc"],
        },
        "event_plan_binding": {
            "locator": "plan.json",
            "raw_sha256": adapter._sha256_bytes(b"plan"),
            "artifact_sha256": plan["artifact_sha256"],
            "planned_at_utc": plan["planned_at_utc"],
        },
        "transport_compatibility_bridge": {
            key: value for key, value in bridge.items() if key != "receipt"
        },
        "frozen_v1_transport_quote": quote,
        "qualification": {
            "phase_two_opening_active": True,
            "complete_terms_and_source_adapter_registered": True,
            "v2_probability_preceded_quote_request": True,
            "event_plan_preceded_quote_request": True,
            "legacy_bridge_changed_probability_point": False,
            "legacy_bridge_interval_used_by_quote": False,
            "exact_response_body_and_transport_replay_present": True,
            "actual_map_start_checked": False,
            "quote_independently_registered": False,
            "phase_two_evidence_qualifies": False,
        },
        "source_locks": adapter._source_locks(None),
        "authority": dict(adapter.AUTHORITY),
        "claim_ceiling": adapter.CLAIM_CEILING,
    }
    payload["artifact_sha256"] = adapter._canonical_sha256(payload)
    return payload


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = adapter._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_v2_bundle_rejects_bridge_use_or_probability_substitution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _bundle(monkeypatch)
    checked = adapter.validate_betano_map_winner_quote_v2(payload)
    assert checked["authority"]["betting_authority"] is False

    forged = deepcopy(payload)
    forged["qualification"]["legacy_bridge_interval_used_by_quote"] = True
    _resign(forged)
    with pytest.raises(
        adapter.BetanoQuoteAdapterV2Error,
        match="qualification changed",
    ):
        adapter.validate_betano_map_winner_quote_v2(forged)
