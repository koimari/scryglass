from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from lol_kills import bookmaker_quote_capture as capture
from tools.live_fair_odds import model


NOW = datetime(2026, 8, 1, 20, 0, tzinfo=timezone.utc)
CAPTURE_SHA = "a" * 64
SETTLEMENT_SHA = "b" * 64
EXTRACTOR_SHA = "c" * 64
RECEIPT_LOCATOR = (
    "data/lol/private_market_quotes/receipts/event-1-total-kills.json"
)
REGISTRY_LOCATOR = "data/lol/private_market_quotes/registry.json"


def receipt() -> dict:
    raw = b'{"capture":"source-backed-fixture"}'
    extraction = capture.build_price_extraction_payload(
        raw_source_payload=raw,
        event_id="event-1-map-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        prices={"under:28.5": 1.93, "over:28.5": 1.80},
        capture_protocol_sha256=CAPTURE_SHA,
        settlement_rules_sha256=SETTLEMENT_SHA,
        extractor_id="synthetic-deterministic-extractor-v1",
        extractor_sha256=EXTRACTOR_SHA,
    )
    return capture.build_quote_receipt(
        raw_source_payload=raw,
        extraction_payload_raw=capture.canonical_bytes(extraction),
        source="bookmaker-browser-capture",
        source_url="https://example.invalid/event-1",
        source_record_id="source:event-1:quote-1",
        capture_protocol_sha256=CAPTURE_SHA,
        settlement_rules_sha256=SETTLEMENT_SHA,
        clock=lambda: NOW - timedelta(seconds=5),
    )


def registry(quote: dict) -> dict:
    return capture.build_quote_registry(
        receipts=[(RECEIPT_LOCATOR, quote)],
        registry_id="independent-registry-1",
        independent_reviewer_id="reviewer-1",
        issued_at=(NOW - timedelta(seconds=4)).isoformat(),
        capture_protocol_sha256=CAPTURE_SHA,
        settlement_rules_sha256=SETTLEMENT_SHA,
    )


def write(root: Path, locator: str, value: dict) -> None:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def approved_authority() -> dict:
    return {
        "status": "approved",
        "betting_decision_authorized": True,
        "blockers": [],
        "authority_record_id": "market-authority-1",
        "market_type": "total_kills",
        "capture_protocol_sha256": CAPTURE_SHA,
        "settlement_rules_sha256": SETTLEMENT_SHA,
        "minimum_edge_pp": 2.0,
        "minimum_expected_return": 0.02,
    }


def test_pinned_quote_cannot_authorize_unregistered_probability(
    tmp_path: Path, monkeypatch
) -> None:
    quote = receipt()
    index = registry(quote)
    write(tmp_path, RECEIPT_LOCATOR, quote)
    write(tmp_path, REGISTRY_LOCATOR, index)
    monkeypatch.setattr(model, "ROOT", tmp_path)
    monkeypatch.setenv(
        model.QUOTE_REGISTRY_SHA_ENV,
        capture.sha256_json(index),
    )
    registered = model._registered_market_quote(
        event_id="event-1-map-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        as_of=NOW,
    )
    assert registered["status"] == "registered"
    view = model._market_view(
        0.60,
        1.93,
        1.80,
        probability_interval=(0.57, 0.63),
        quote=registered["quote"],
        expected_quote_sha256=registered["quote_sha256"],
        quote_registry_sha256=registered["registry_sha256"],
        authority=approved_authority(),
        as_of=NOW,
        selection="under:28.5",
        opposing_selection="over:28.5",
        event_id="event-1-map-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
    )
    assert view["status"] == "unavailable"
    assert view["decision"] == "NO_AUTHORIZED_BET"
    assert view["probability"] is None
    assert view["expected_return_pct"] is None
    assert "event_probability_receipt_missing" in view["blockers"]
    assert "event_probability_registry_missing" in view["blockers"]


def test_missing_registry_pin_remains_typed_unavailable(monkeypatch) -> None:
    monkeypatch.delenv(model.QUOTE_REGISTRY_SHA_ENV, raising=False)
    unavailable = model._registered_market_quote(
        event_id="event-1-map-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        as_of=NOW,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["blockers"] == ["quote_registry_not_registered"]


def test_tampered_registered_receipt_cannot_reach_market_view(
    tmp_path: Path, monkeypatch
) -> None:
    quote = receipt()
    index = registry(quote)
    quote["prices"]["under:28.5"] = 8.0
    write(tmp_path, RECEIPT_LOCATOR, quote)
    write(tmp_path, REGISTRY_LOCATOR, index)
    monkeypatch.setattr(model, "ROOT", tmp_path)
    monkeypatch.setenv(model.QUOTE_REGISTRY_SHA_ENV, capture.sha256_json(index))
    unavailable = model._registered_market_quote(
        event_id="event-1-map-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        as_of=NOW,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["quote"] is None
    assert unavailable["blockers"] == ["registered_market_quote_invalid"]
