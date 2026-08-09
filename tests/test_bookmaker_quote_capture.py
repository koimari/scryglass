from __future__ import annotations

import base64
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lol_kills import bookmaker_quote_capture as capture


NOW = datetime(2026, 8, 1, 20, 0, tzinfo=timezone.utc)
CAPTURE_SHA = "a" * 64
SETTLEMENT_SHA = "b" * 64
EXTRACTOR_SHA = "c" * 64
RECEIPT_LOCATOR = (
    "data/lol/private_market_quotes/receipts/event-1-total-kills.json"
)
REGISTRY_LOCATOR = "data/lol/private_market_quotes/registry.json"


def total_kills_receipt() -> dict:
    raw = b'{"source":"captured-browser-payload","event":1}'
    extraction = capture.build_price_extraction_payload(
        raw_source_payload=raw,
        event_id="event-1-map-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        prices={
            "over:28.5": 1.87,
            "under:28.5": 1.87,
            "over:29.5": 2.05,
            "under:29.5": 1.72,
        },
        capture_protocol_sha256=CAPTURE_SHA,
        settlement_rules_sha256=SETTLEMENT_SHA,
        extractor_id="synthetic-deterministic-extractor-v1",
        extractor_sha256=EXTRACTOR_SHA,
    )
    return capture.build_quote_receipt(
        raw_source_payload=raw,
        extraction_payload_raw=capture.canonical_bytes(extraction),
        source="betano-browser-capture",
        source_url="https://example.invalid/event/1",
        source_record_id="betano:event-1:quote-1",
        capture_protocol_sha256=CAPTURE_SHA,
        settlement_rules_sha256=SETTLEMENT_SHA,
        clock=lambda: NOW - timedelta(seconds=5),
    )


def registry(receipt: dict | None = None) -> dict:
    return capture.build_quote_registry(
        receipts=[(RECEIPT_LOCATOR, receipt or total_kills_receipt())],
        registry_id="quote-review-1",
        independent_reviewer_id="reviewer-1",
        issued_at=(NOW - timedelta(seconds=4)).isoformat(),
        capture_protocol_sha256=CAPTURE_SHA,
        settlement_rules_sha256=SETTLEMENT_SHA,
    )


def write_json(root: Path, locator: str, value: dict) -> None:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_receipt_commits_exact_raw_source_bytes_and_complete_two_sided_lines() -> None:
    receipt = total_kills_receipt()
    checked = capture.validate_quote_receipt(receipt)
    assert checked["quote_sha256"] == capture.sha256_json(receipt)
    assert checked["prices"]["over:28.5"] == 1.87
    assert len(checked["source_payload_sha256"]) == 64


def test_total_kills_capture_rejects_one_sided_or_non_half_kill_lines() -> None:
    arguments = dict(
        raw_source_payload=b"payload",
        event_id="event-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        capture_protocol_sha256=CAPTURE_SHA,
        settlement_rules_sha256=SETTLEMENT_SHA,
        extractor_id="synthetic-deterministic-extractor-v1",
        extractor_sha256=EXTRACTOR_SHA,
    )
    with pytest.raises(capture.QuoteCaptureError, match="both over and under"):
        capture.build_price_extraction_payload(
            **arguments, prices={"over:28.5": 1.87}
        )
    with pytest.raises(capture.QuoteCaptureError, match="canonical"):
        capture.build_price_extraction_payload(
            **arguments,
            prices={"over:28": 1.87, "under:28": 1.87},
        )


def test_raw_source_payload_tamper_is_rejected() -> None:
    receipt = total_kills_receipt()
    receipt["source_payload_base64"] = "dGFtcGVyZWQ="
    with pytest.raises(capture.QuoteCaptureError, match="source payload digest mismatch"):
        capture.validate_quote_receipt(receipt)


def test_extraction_payload_tamper_is_rejected() -> None:
    receipt = total_kills_receipt()
    extraction = json.loads(
        base64.b64decode(receipt["extraction_payload_base64"])
    )
    extraction["prices"]["over:28.5"] = 9.99
    receipt["extraction_payload_base64"] = base64.b64encode(
        capture.canonical_bytes(extraction)
    ).decode("ascii")
    with pytest.raises(capture.QuoteCaptureError, match="extraction payload digest"):
        capture.validate_quote_receipt(receipt)


def test_registry_cannot_authorize_itself_without_external_digest() -> None:
    value = registry()
    with pytest.raises(capture.RegisteredQuoteUnavailable) as error:
        capture.validate_quote_registry(value, expected_registry_sha256=None)
    assert error.value.code == "quote_registry_not_registered"


def test_registered_loader_replays_registry_receipt_and_source_bindings(
    tmp_path: Path,
) -> None:
    receipt = total_kills_receipt()
    value = registry(receipt)
    write_json(tmp_path, RECEIPT_LOCATOR, receipt)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    loaded = capture.load_registered_quote(
        registry_locator=REGISTRY_LOCATOR,
        expected_registry_sha256=capture.sha256_json(value),
        event_id="event-1-map-1",
        market_type="total_kills",
        settlement_rule_id="map-total-kills-v1",
        as_of=NOW,
        root=tmp_path,
    )
    assert loaded["status"] == "registered"
    assert loaded["quote_sha256"] == capture.sha256_json(receipt)
    assert loaded["quote"]["source_payload_base64"] == receipt[
        "source_payload_base64"
    ]


def test_post_registration_receipt_tamper_is_rejected(tmp_path: Path) -> None:
    receipt = total_kills_receipt()
    value = registry(receipt)
    receipt["prices"]["over:28.5"] = 9.99
    write_json(tmp_path, RECEIPT_LOCATOR, receipt)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    with pytest.raises(capture.QuoteCaptureError, match="quote receipt digest mismatch"):
        capture.load_registered_quote(
            registry_locator=REGISTRY_LOCATOR,
            expected_registry_sha256=capture.sha256_json(value),
            event_id="event-1-map-1",
            market_type="total_kills",
            settlement_rule_id="map-total-kills-v1",
            as_of=NOW,
            root=tmp_path,
        )


def test_registry_digest_tamper_is_rejected(tmp_path: Path) -> None:
    receipt = total_kills_receipt()
    value = registry(receipt)
    expected = capture.sha256_json(value)
    value["independent_reviewer_id"] = "attacker"
    write_json(tmp_path, RECEIPT_LOCATOR, receipt)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    with pytest.raises(capture.RegisteredQuoteUnavailable) as error:
        capture.load_registered_quote(
            registry_locator=REGISTRY_LOCATOR,
            expected_registry_sha256=expected,
            event_id="event-1-map-1",
            market_type="total_kills",
            settlement_rule_id="map-total-kills-v1",
            as_of=NOW,
            root=tmp_path,
        )
    assert error.value.code == "quote_registry_digest_mismatch"


def test_registry_rejects_ambiguous_market_keys() -> None:
    receipt = total_kills_receipt()
    value = registry(receipt)
    value["entries"].append(dict(value["entries"][0]))
    expected = capture.sha256_json(value)
    with pytest.raises(capture.QuoteCaptureError, match="ambiguous market key"):
        capture.validate_quote_registry(value, expected_registry_sha256=expected)


def test_registry_rejects_receipt_path_escape() -> None:
    receipt = total_kills_receipt()
    with pytest.raises(capture.QuoteCaptureError, match="outside"):
        capture.build_quote_registry(
            receipts=[("../../quote.json", receipt)],
            registry_id="quote-review-1",
            independent_reviewer_id="reviewer-1",
            issued_at=NOW.isoformat(),
            capture_protocol_sha256=CAPTURE_SHA,
            settlement_rules_sha256=SETTLEMENT_SHA,
        )


def test_candidate_writer_never_clobbers_an_existing_receipt(tmp_path: Path) -> None:
    output = tmp_path / "quote.json"
    original = total_kills_receipt()
    capture._atomic_write_json(output, original)
    before = output.read_bytes()
    replacement = total_kills_receipt()
    replacement["source_record_id"] = "replacement"
    with pytest.raises(capture.QuoteCaptureError, match="already exists"):
        capture._atomic_write_json(output, replacement)
    assert output.read_bytes() == before


def test_quote_capture_cli_exposes_no_user_timestamp_argument() -> None:
    assert "--captured-at" not in Path(capture.__file__).read_text()
    receipt = total_kills_receipt()
    assert receipt["clock_attestation"]["user_supplied_timestamp_allowed"] is False
    assert receipt["capture_timing_boundary"][
        "prospective_transport_latency_authority"
    ] is False
