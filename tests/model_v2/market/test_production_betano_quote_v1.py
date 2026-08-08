from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import production_betano_quote_v1 as quote


PROBABILITY = {
    "artifact_sha256": "1" * 64,
    "captured_at_utc": "2026-10-05T12:00:00+00:00",
    "event": {
        "event_id": "event-1",
        "selection": "winner:blue",
        "opposing_selection": "winner:red",
        "scheduled_event_start_utc": "2026-10-05T12:30:00+00:00",
    },
    "semantic_market_authority_binding": {"authority_raw_sha256": "2" * 64},
}
ACTIVE = {
    "receipt": {
        "authority_id": "authority-1",
        "valid_until_utc": "2026-10-06T00:00:00+00:00",
    },
    "receipt_raw_sha256": "2" * 64,
}
BRIDGE = {
    "receipt": {"legacy": True},
    "receipt_sha256": "3" * 64,
    "original_probability": 0.56,
    "original_probability_interval": [0.51, 0.61],
    "bridge_probability_interval": [0.51, 0.61],
    "interval_widened_only_for_legacy_transport_shape": False,
    "bridge_used_as_probability_or_decision_input": False,
    "bridge_persisted_as_probability_evidence": False,
}
TRANSPORT = {
    "captured_at_utc": "2026-10-05T12:00:03+00:00",
    "transport": {
        "request_started_at_utc": "2026-10-05T12:00:01+00:00",
        "response_received_at_utc": "2026-10-05T12:00:02+00:00",
    },
    "prediction_binding": {
        "event_probability_receipt_sha256": "3" * 64,
        "prediction_captured_at_utc": "2026-10-05T12:00:00+00:00",
        "scryglass_event_id": "event-1",
        "selection": "winner:blue",
        "opposing_selection": "winner:red",
    },
    "source_extraction": {
        "betano_event": {
            "scheduled_series_start_epoch_ms": 1791203400000,
        }
    },
    "generic_quote_receipt": {
        "prices": {"winner:blue": 1.95, "winner:red": 1.85}
    },
}


def _install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        quote,
        "_probability",
        lambda **_kwargs: ("probability.json", b"probability", deepcopy(PROBABILITY)),
    )
    monkeypatch.setattr(quote, "_bridge", lambda _value: deepcopy(BRIDGE))
    monkeypatch.setattr(quote, "_semantic_authority", lambda **_kwargs: deepcopy(ACTIVE))
    monkeypatch.setattr(quote, "_source_locks", lambda _root: [{"raw_sha256": "4" * 64}])
    monkeypatch.setattr(
        quote.transport,
        "capture_betano_map_winner_quote_v1",
        lambda **_kwargs: deepcopy(TRANSPORT),
    )
    monkeypatch.setattr(
        quote.transport,
        "validate_betano_map_winner_quote_v1",
        lambda _payload, root: deepcopy(TRANSPORT),
    )


def test_production_quote_binds_probability_authority_and_exact_prices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install(monkeypatch)
    receipt = quote.capture_production_betano_quote_v1(
        production_probability_locator="ignored.json",
        request_url="https://br.betano.com/event",
        betano_event_id="1",
        map_number=1,
        participant_bindings=[],
        fetcher=lambda _url: None,
        environment={},
    )
    checked = quote.validate_production_betano_quote_v1(receipt, environment={})
    assert checked["prices"] == {"winner:blue": 1.95, "winner:red": 1.85}
    assert checked["authority"]["transaction_authority"] is False
    assert checked["qualification"]["cash_acceptance_limit_or_execution_proven"] is False

    forged = deepcopy(receipt)
    forged["qualification"]["cash_acceptance_limit_or_execution_proven"] = True
    forged["artifact_sha256"] = quote._canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(quote.ProductionBetanoQuoteError, match="qualification changed"):
        quote.validate_production_betano_quote_v1(forged, environment={})
