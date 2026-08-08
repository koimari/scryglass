"""Private bookmaker quote capture and independently pinned registry loading.

Capture content identity is deliberately separate from authority.  This module
can build a candidate receipt from raw source bytes, but the runtime accepts it
only through a registry whose SHA-256 was supplied out of band.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA_VERSION = "scryglass.private-bookmaker-quote.v2"
REGISTRY_SCHEMA_VERSION = "scryglass.private-bookmaker-quote-registry.v2"
EXTRACTION_SCHEMA_VERSION = "scryglass.private-bookmaker-price-extraction.v1"
REGISTRY_SCOPE = "private_personal_decision_support"
RECEIPT_PREFIX = PurePosixPath("data/lol/private_market_quotes/receipts")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TOTAL_SELECTION_RE = re.compile(r"^(over|under):(\d+\.5)$")
MAX_SOURCE_PAYLOAD_BYTES = 5_000_000
MAX_EXTRACTION_PAYLOAD_BYTES = 1_000_000
EXTRACTION_METHOD = "deterministic_source_adapter"
CLAIM_CEILING = (
    "System-clocked, exact-byte private quote candidate only. The receipt is "
    "non-authorizing and does not establish probability, edge, expected value, "
    "recommendation, or betting authority."
)


class QuoteCaptureError(ValueError):
    """A quote capture or registry violates its frozen contract."""


class RegisteredQuoteUnavailable(QuoteCaptureError):
    """No independently registered quote can be loaded for the requested market."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise QuoteCaptureError("value is not canonical finite JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise QuoteCaptureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                QuoteCaptureError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QuoteCaptureError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise QuoteCaptureError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise QuoteCaptureError(f"{label} keys do not match the frozen contract")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise QuoteCaptureError(f"{label} must be a non-empty string")
    return value


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise QuoteCaptureError(f"{label} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise QuoteCaptureError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise QuoteCaptureError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise QuoteCaptureError(
            "quote capture clock must return a timezone-aware datetime"
        )
    return observed.astimezone(timezone.utc)


def _price(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise QuoteCaptureError(f"{label} must be numeric decimal odds")
    number = float(value)
    if not math.isfinite(number) or not 1.0 < number <= 100.0:
        raise QuoteCaptureError(f"{label} must be finite decimal odds in (1, 100]")
    return number


def _validate_prices(market_type: str, prices: Any) -> dict[str, float]:
    if not isinstance(prices, Mapping):
        raise QuoteCaptureError("prices must be a mapping")
    normalized: dict[str, float] = {}
    for raw_key, raw_value in prices.items():
        key = _nonempty(raw_key, "selection key")
        if key in normalized:
            raise QuoteCaptureError(f"duplicate selection key: {key}")
        normalized[key] = _price(raw_value, f"price.{key}")
    if market_type == "match_winner":
        if len(normalized) != 2 or any(
            not key.startswith("winner:") or len(key) == len("winner:")
            for key in normalized
        ):
            raise QuoteCaptureError(
                "match_winner requires exactly two winner:<team> selections"
            )
    elif market_type == "total_kills":
        by_line: dict[str, set[str]] = {}
        for key in normalized:
            match = TOTAL_SELECTION_RE.fullmatch(key)
            if not match:
                raise QuoteCaptureError(
                    "total_kills selections must be canonical over:<x.5> or under:<x.5>"
                )
            side, line = match.groups()
            by_line.setdefault(line, set()).add(side)
        if not by_line or any(sides != {"over", "under"} for sides in by_line.values()):
            raise QuoteCaptureError(
                "every total-kills line requires both over and under prices"
            )
    else:
        raise QuoteCaptureError(f"unsupported market_type: {market_type}")
    return dict(sorted(normalized.items()))


def _decode_source_payload(value: Any) -> bytes:
    text = _nonempty(value, "source_payload_base64")
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as exc:
        raise QuoteCaptureError("source_payload_base64 is not strict base64") from exc
    if not raw or len(raw) > MAX_SOURCE_PAYLOAD_BYTES:
        raise QuoteCaptureError("source payload is empty or exceeds the size limit")
    if base64.b64encode(raw).decode("ascii") != text:
        raise QuoteCaptureError("source_payload_base64 is not canonical")
    return raw


def _decode_extraction_payload(value: Any) -> bytes:
    text = _nonempty(value, "extraction_payload_base64")
    try:
        raw = base64.b64decode(text, validate=True)
    except (ValueError, TypeError) as exc:
        raise QuoteCaptureError(
            "extraction_payload_base64 is not strict base64"
        ) from exc
    if not raw or len(raw) > MAX_EXTRACTION_PAYLOAD_BYTES:
        raise QuoteCaptureError(
            "extraction payload is empty or exceeds the size limit"
        )
    if base64.b64encode(raw).decode("ascii") != text:
        raise QuoteCaptureError("extraction_payload_base64 is not canonical")
    return raw


def build_price_extraction_payload(
    *,
    raw_source_payload: bytes,
    event_id: str,
    market_type: str,
    settlement_rule_id: str,
    prices: Mapping[str, float],
    capture_protocol_sha256: str,
    settlement_rules_sha256: str,
    extractor_id: str,
    extractor_sha256: str,
) -> dict[str, Any]:
    """Build a non-authorizing deterministic extraction record.

    This function records what an extractor claims to have read. Independent
    review must still verify the bound extractor and exact source bytes.
    """

    if not isinstance(raw_source_payload, bytes) or not raw_source_payload:
        raise QuoteCaptureError("raw_source_payload must be non-empty bytes")
    if len(raw_source_payload) > MAX_SOURCE_PAYLOAD_BYTES:
        raise QuoteCaptureError("source payload exceeds the size limit")
    payload = {
        "schema_version": EXTRACTION_SCHEMA_VERSION,
        "source_payload_sha256": hashlib.sha256(raw_source_payload).hexdigest(),
        "event_id": _nonempty(event_id, "event_id"),
        "market_type": market_type,
        "settlement_rule_id": _nonempty(
            settlement_rule_id, "settlement_rule_id"
        ),
        "market_status": "open",
        "prices": _validate_prices(market_type, prices),
        "capture_protocol_sha256": _sha(
            capture_protocol_sha256, "capture_protocol_sha256"
        ),
        "settlement_rules_sha256": _sha(
            settlement_rules_sha256, "settlement_rules_sha256"
        ),
        "extraction_method": EXTRACTION_METHOD,
        "extractor_id": _nonempty(extractor_id, "extractor_id"),
        "extractor_sha256": _sha(extractor_sha256, "extractor_sha256"),
    }
    validate_price_extraction_payload(
        canonical_bytes(payload),
        expected_source_payload_sha256=payload["source_payload_sha256"],
        expected_capture_protocol_sha256=payload["capture_protocol_sha256"],
        expected_settlement_rules_sha256=payload["settlement_rules_sha256"],
    )
    return payload


def validate_price_extraction_payload(
    raw: bytes,
    *,
    expected_source_payload_sha256: str,
    expected_capture_protocol_sha256: str,
    expected_settlement_rules_sha256: str,
) -> dict[str, Any]:
    if not isinstance(raw, bytes) or not raw or len(raw) > MAX_EXTRACTION_PAYLOAD_BYTES:
        raise QuoteCaptureError("extraction payload bytes are invalid")
    value = _read_json_bytes(raw, "price extraction payload")
    _exact_keys(
        value,
        {
            "schema_version",
            "source_payload_sha256",
            "event_id",
            "market_type",
            "settlement_rule_id",
            "market_status",
            "prices",
            "capture_protocol_sha256",
            "settlement_rules_sha256",
            "extraction_method",
            "extractor_id",
            "extractor_sha256",
        },
        "price extraction payload",
    )
    if value.get("schema_version") != EXTRACTION_SCHEMA_VERSION:
        raise QuoteCaptureError("price extraction schema is not recognized")
    if value.get("market_status") != "open":
        raise QuoteCaptureError("bookmaker market was not open at capture")
    if value.get("extraction_method") != EXTRACTION_METHOD:
        raise QuoteCaptureError("price extraction method is not deterministic")
    source_sha = _sha(value.get("source_payload_sha256"), "source_payload_sha256")
    if source_sha != _sha(
        expected_source_payload_sha256, "expected_source_payload_sha256"
    ):
        raise QuoteCaptureError("price extraction source payload binding mismatch")
    capture_sha = _sha(
        value.get("capture_protocol_sha256"), "capture_protocol_sha256"
    )
    if capture_sha != _sha(
        expected_capture_protocol_sha256,
        "expected_capture_protocol_sha256",
    ):
        raise QuoteCaptureError("price extraction capture protocol mismatch")
    settlement_sha = _sha(
        value.get("settlement_rules_sha256"), "settlement_rules_sha256"
    )
    if settlement_sha != _sha(
        expected_settlement_rules_sha256,
        "expected_settlement_rules_sha256",
    ):
        raise QuoteCaptureError("price extraction settlement rules mismatch")
    market_type = _nonempty(value.get("market_type"), "market_type")
    return {
        **dict(value),
        "event_id": _nonempty(value.get("event_id"), "event_id"),
        "market_type": market_type,
        "settlement_rule_id": _nonempty(
            value.get("settlement_rule_id"), "settlement_rule_id"
        ),
        "prices": _validate_prices(market_type, value.get("prices")),
        "extractor_id": _nonempty(value.get("extractor_id"), "extractor_id"),
        "extractor_sha256": _sha(
            value.get("extractor_sha256"), "extractor_sha256"
        ),
    }


def build_quote_receipt(
    *,
    raw_source_payload: bytes,
    extraction_payload_raw: bytes,
    source: str,
    source_url: str,
    source_record_id: str,
    capture_protocol_sha256: str,
    settlement_rules_sha256: str,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Build a non-authorizing quote receipt from the exact captured bytes."""
    captured_at = _clock_sample(clock)
    if not isinstance(raw_source_payload, bytes) or not raw_source_payload:
        raise QuoteCaptureError("raw_source_payload must be non-empty bytes")
    if len(raw_source_payload) > MAX_SOURCE_PAYLOAD_BYTES:
        raise QuoteCaptureError("source payload exceeds the size limit")
    parsed_url = urlparse(_nonempty(source_url, "source_url"))
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise QuoteCaptureError("source_url must be an absolute HTTPS URL")
    capture_sha = _sha(capture_protocol_sha256, "capture_protocol_sha256")
    settlement_sha = _sha(settlement_rules_sha256, "settlement_rules_sha256")
    source_payload_sha256 = hashlib.sha256(raw_source_payload).hexdigest()
    extraction = validate_price_extraction_payload(
        extraction_payload_raw,
        expected_source_payload_sha256=source_payload_sha256,
        expected_capture_protocol_sha256=capture_sha,
        expected_settlement_rules_sha256=settlement_sha,
    )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "source": _nonempty(source, "source"),
        "source_url": source_url,
        "source_record_id": _nonempty(source_record_id, "source_record_id"),
        "source_payload_sha256": source_payload_sha256,
        "source_payload_base64": base64.b64encode(raw_source_payload).decode("ascii"),
        "extraction_payload_sha256": hashlib.sha256(
            extraction_payload_raw
        ).hexdigest(),
        "extraction_payload_base64": base64.b64encode(
            extraction_payload_raw
        ).decode("ascii"),
        "captured_at_utc": captured_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_quote_builder",
            "observed_wall_clock_utc": captured_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "capture_timing_boundary": {
            "builder_receipt_time_is_transport_receive_time": False,
            "prospective_transport_latency_authority": False,
            "retrospective_backfill_qualifies": False,
        },
        "event_id": extraction["event_id"],
        "market_type": extraction["market_type"],
        "settlement_rule_id": extraction["settlement_rule_id"],
        "capture_protocol_sha256": capture_sha,
        "settlement_rules_sha256": settlement_sha,
        "extractor_id": extraction["extractor_id"],
        "extractor_sha256": extraction["extractor_sha256"],
        "prices": extraction["prices"],
        "authority": {
            "quote_identity_authority": False,
            "probability_authority": False,
            "expected_value_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": CLAIM_CEILING,
    }
    validate_quote_receipt(receipt)
    return receipt


def validate_quote_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_quote_sha256: str | None = None,
    expected_capture_protocol_sha256: str | None = None,
    expected_settlement_rules_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise QuoteCaptureError("quote receipt must be a mapping")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "source",
            "source_url",
            "source_record_id",
            "source_payload_sha256",
            "source_payload_base64",
            "extraction_payload_sha256",
            "extraction_payload_base64",
            "captured_at_utc",
            "clock_attestation",
            "capture_timing_boundary",
            "event_id",
            "market_type",
            "settlement_rule_id",
            "capture_protocol_sha256",
            "settlement_rules_sha256",
            "extractor_id",
            "extractor_sha256",
            "prices",
            "authority",
            "claim_ceiling",
        },
        "quote receipt",
    )
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise QuoteCaptureError("quote receipt schema is not recognized")
    actual_quote_sha256 = sha256_json(receipt)
    if expected_quote_sha256 is not None:
        _sha(expected_quote_sha256, "expected_quote_sha256")
        if actual_quote_sha256 != expected_quote_sha256:
            raise QuoteCaptureError("quote receipt digest mismatch")
    _nonempty(receipt.get("source"), "source")
    parsed_url = urlparse(_nonempty(receipt.get("source_url"), "source_url"))
    if (
        parsed_url.scheme != "https"
        or not parsed_url.netloc
        or parsed_url.username is not None
        or parsed_url.password is not None
    ):
        raise QuoteCaptureError("source_url must be an absolute HTTPS URL")
    _nonempty(receipt.get("source_record_id"), "source_record_id")
    raw = _decode_source_payload(receipt.get("source_payload_base64"))
    claimed_payload_sha256 = _sha(
        receipt.get("source_payload_sha256"), "source_payload_sha256"
    )
    if hashlib.sha256(raw).hexdigest() != claimed_payload_sha256:
        raise QuoteCaptureError("source payload digest mismatch")
    extraction_raw = _decode_extraction_payload(
        receipt.get("extraction_payload_base64")
    )
    extraction_sha = _sha(
        receipt.get("extraction_payload_sha256"),
        "extraction_payload_sha256",
    )
    if hashlib.sha256(extraction_raw).hexdigest() != extraction_sha:
        raise QuoteCaptureError("extraction payload digest mismatch")
    captured_at = _timestamp(receipt.get("captured_at_utc"), "captured_at_utc")
    if receipt.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_quote_builder",
        "observed_wall_clock_utc": captured_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise QuoteCaptureError("quote clock attestation changed")
    if receipt.get("capture_timing_boundary") != {
        "builder_receipt_time_is_transport_receive_time": False,
        "prospective_transport_latency_authority": False,
        "retrospective_backfill_qualifies": False,
    }:
        raise QuoteCaptureError("quote capture timing boundary changed")
    capture_sha = _sha(
        receipt.get("capture_protocol_sha256"), "capture_protocol_sha256"
    )
    settlement_sha = _sha(
        receipt.get("settlement_rules_sha256"), "settlement_rules_sha256"
    )
    if (
        expected_capture_protocol_sha256 is not None
        and capture_sha
        != _sha(
            expected_capture_protocol_sha256,
            "expected_capture_protocol_sha256",
        )
    ):
        raise QuoteCaptureError("capture protocol binding mismatch")
    if (
        expected_settlement_rules_sha256 is not None
        and settlement_sha
        != _sha(
            expected_settlement_rules_sha256,
            "expected_settlement_rules_sha256",
        )
    ):
        raise QuoteCaptureError("settlement rules binding mismatch")
    extraction = validate_price_extraction_payload(
        extraction_raw,
        expected_source_payload_sha256=claimed_payload_sha256,
        expected_capture_protocol_sha256=capture_sha,
        expected_settlement_rules_sha256=settlement_sha,
    )
    for field in ("event_id", "market_type", "settlement_rule_id"):
        if receipt.get(field) != extraction[field]:
            raise QuoteCaptureError(f"quote {field} extraction binding mismatch")
    if receipt.get("extractor_id") != extraction["extractor_id"]:
        raise QuoteCaptureError("quote extractor id binding mismatch")
    if receipt.get("extractor_sha256") != extraction["extractor_sha256"]:
        raise QuoteCaptureError("quote extractor digest binding mismatch")
    prices = _validate_prices(extraction["market_type"], receipt.get("prices"))
    if prices != extraction["prices"]:
        raise QuoteCaptureError("quote prices differ from extraction payload")
    authority = receipt.get("authority")
    expected_authority = {
        "quote_identity_authority": False,
        "probability_authority": False,
        "expected_value_authority": False,
        "recommendation_authority": False,
        "betting_authority": False,
    }
    if authority != expected_authority:
        raise QuoteCaptureError("quote receipt exceeds its authority")
    if receipt.get("claim_ceiling") != CLAIM_CEILING:
        raise QuoteCaptureError("quote receipt claim ceiling changed")
    return {**dict(receipt), "prices": prices, "quote_sha256": actual_quote_sha256}


def build_quote_registry(
    *,
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    registry_id: str,
    independent_reviewer_id: str,
    issued_at: str,
    capture_protocol_sha256: str,
    settlement_rules_sha256: str,
) -> dict[str, Any]:
    """Build a registry candidate; its digest still requires external registration."""
    issued = _timestamp(issued_at, "issued_at")
    capture_sha = _sha(capture_protocol_sha256, "capture_protocol_sha256")
    settlement_sha = _sha(settlement_rules_sha256, "settlement_rules_sha256")
    entries: list[dict[str, Any]] = []
    for locator, raw_receipt in receipts:
        receipt = validate_quote_receipt(
            raw_receipt,
            expected_capture_protocol_sha256=capture_sha,
            expected_settlement_rules_sha256=settlement_sha,
        )
        _validate_receipt_locator(locator)
        if issued < _timestamp(receipt["captured_at_utc"], "captured_at_utc"):
            raise QuoteCaptureError("quote registry cannot predate its receipt")
        entries.append(
            {
                "event_id": receipt["event_id"],
                "market_type": receipt["market_type"],
                "settlement_rule_id": receipt["settlement_rule_id"],
                "source_record_id": receipt["source_record_id"],
                "captured_at_utc": receipt["captured_at_utc"],
                "extractor_id": receipt["extractor_id"],
                "extractor_sha256": receipt["extractor_sha256"],
                "receipt_locator": locator,
                "quote_sha256": receipt["quote_sha256"],
            }
        )
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "approved",
        "scope": REGISTRY_SCOPE,
        "public_or_transactional_use": False,
        "registry_id": _nonempty(registry_id, "registry_id"),
        "independent_reviewer_id": _nonempty(
            independent_reviewer_id, "independent_reviewer_id"
        ),
        "issued_at": issued_at,
        "capture_protocol_sha256": capture_sha,
        "settlement_rules_sha256": settlement_sha,
        "entries": sorted(
            entries,
            key=lambda item: (
                item["event_id"],
                item["market_type"],
                item["settlement_rule_id"],
            ),
        ),
    }
    validate_quote_registry(registry, expected_registry_sha256=sha256_json(registry))
    return registry


def _validate_receipt_locator(locator: Any) -> PurePosixPath:
    text = _nonempty(locator, "receipt_locator")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(RECEIPT_PREFIX.parts)]) != RECEIPT_PREFIX.parts
        or path.suffix != ".json"
    ):
        raise QuoteCaptureError("receipt_locator is outside the private receipt root")
    return path


def validate_quote_registry(
    registry: Mapping[str, Any], *, expected_registry_sha256: str | None
) -> dict[str, Any]:
    if expected_registry_sha256 is None:
        raise RegisteredQuoteUnavailable("quote_registry_not_registered")
    _sha(expected_registry_sha256, "expected_registry_sha256")
    if not isinstance(registry, Mapping):
        raise QuoteCaptureError("quote registry must be a mapping")
    if sha256_json(registry) != expected_registry_sha256:
        raise RegisteredQuoteUnavailable("quote_registry_digest_mismatch")
    _exact_keys(
        registry,
        {
            "schema_version",
            "status",
            "scope",
            "public_or_transactional_use",
            "registry_id",
            "independent_reviewer_id",
            "issued_at",
            "capture_protocol_sha256",
            "settlement_rules_sha256",
            "entries",
        },
        "quote registry",
    )
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise QuoteCaptureError("quote registry schema is not recognized")
    if (
        registry.get("status") != "approved"
        or registry.get("scope") != REGISTRY_SCOPE
        or registry.get("public_or_transactional_use") is not False
    ):
        raise QuoteCaptureError("quote registry is not approved for private support")
    _nonempty(registry.get("registry_id"), "registry_id")
    _nonempty(registry.get("independent_reviewer_id"), "independent_reviewer_id")
    _timestamp(registry.get("issued_at"), "issued_at")
    _sha(registry.get("capture_protocol_sha256"), "capture_protocol_sha256")
    _sha(registry.get("settlement_rules_sha256"), "settlement_rules_sha256")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise QuoteCaptureError("quote registry entries must be a non-empty list")
    expected_keys = {
        "event_id",
        "market_type",
        "settlement_rule_id",
        "source_record_id",
        "captured_at_utc",
        "extractor_id",
        "extractor_sha256",
        "receipt_locator",
        "quote_sha256",
    }
    seen: set[tuple[str, str, str]] = set()
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise QuoteCaptureError("quote registry entry must be a mapping")
        _exact_keys(entry, expected_keys, "quote registry entry")
        key = (
            _nonempty(entry.get("event_id"), "entry.event_id"),
            _nonempty(entry.get("market_type"), "entry.market_type"),
            _nonempty(
                entry.get("settlement_rule_id"), "entry.settlement_rule_id"
            ),
        )
        if key in seen:
            raise QuoteCaptureError("quote registry contains an ambiguous market key")
        seen.add(key)
        _nonempty(entry.get("source_record_id"), "entry.source_record_id")
        _timestamp(entry.get("captured_at_utc"), "entry.captured_at_utc")
        _nonempty(entry.get("extractor_id"), "entry.extractor_id")
        _sha(entry.get("extractor_sha256"), "entry.extractor_sha256")
        _validate_receipt_locator(entry.get("receipt_locator"))
        _sha(entry.get("quote_sha256"), "entry.quote_sha256")
        normalized.append(dict(entry))
    if normalized != sorted(
        normalized,
        key=lambda item: (
            item["event_id"],
            item["market_type"],
            item["settlement_rule_id"],
        ),
    ):
        raise QuoteCaptureError("quote registry entries are not canonically ordered")
    return {**dict(registry), "entries": normalized}


def _safe_repo_file(root: Path, locator: str) -> Path:
    relative = PurePosixPath(locator)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise RegisteredQuoteUnavailable("quote_artifact_path_invalid")
    root_real = root.resolve(strict=True)
    current = root_real
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise RegisteredQuoteUnavailable("quote_artifact_missing") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RegisteredQuoteUnavailable("quote_artifact_symlink_rejected")
    metadata = os.lstat(current)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RegisteredQuoteUnavailable("quote_artifact_not_unaliased_file")
    try:
        current.resolve(strict=True).relative_to(root_real)
    except ValueError as exc:
        raise RegisteredQuoteUnavailable("quote_artifact_path_escape") from exc
    return current


def load_registered_quote(
    *,
    registry_locator: str,
    expected_registry_sha256: str | None,
    event_id: str,
    market_type: str,
    settlement_rule_id: str,
    as_of: datetime,
    root: Path = ROOT,
) -> dict[str, Any]:
    """Load one quote through an independently pinned registry and exact key."""
    if not expected_registry_sha256:
        raise RegisteredQuoteUnavailable("quote_registry_not_registered")
    registry_path = _safe_repo_file(root, registry_locator)
    registry = validate_quote_registry(
        _read_json_bytes(registry_path.read_bytes(), "quote registry"),
        expected_registry_sha256=expected_registry_sha256,
    )
    if _timestamp(registry["issued_at"], "issued_at") > as_of.astimezone(timezone.utc):
        raise RegisteredQuoteUnavailable("quote_registry_from_future")
    wanted = (event_id, market_type, settlement_rule_id)
    matches = [
        entry
        for entry in registry["entries"]
        if (
            entry["event_id"],
            entry["market_type"],
            entry["settlement_rule_id"],
        )
        == wanted
    ]
    if len(matches) != 1:
        raise RegisteredQuoteUnavailable("registered_market_quote_unavailable")
    entry = matches[0]
    _validate_receipt_locator(entry["receipt_locator"])
    receipt_path = _safe_repo_file(root, entry["receipt_locator"])
    receipt = validate_quote_receipt(
        _read_json_bytes(receipt_path.read_bytes(), "quote receipt"),
        expected_quote_sha256=entry["quote_sha256"],
        expected_capture_protocol_sha256=registry["capture_protocol_sha256"],
        expected_settlement_rules_sha256=registry["settlement_rules_sha256"],
    )
    if receipt["source_record_id"] != entry["source_record_id"]:
        raise RegisteredQuoteUnavailable("quote_source_record_binding_mismatch")
    for field in ("captured_at_utc", "extractor_id", "extractor_sha256"):
        if receipt[field] != entry[field]:
            raise RegisteredQuoteUnavailable(f"quote_{field}_binding_mismatch")
    return {
        "status": "registered",
        "quote": {key: value for key, value in receipt.items() if key != "quote_sha256"},
        "quote_sha256": receipt["quote_sha256"],
        "registry_id": registry["registry_id"],
        "registry_sha256": expected_registry_sha256,
        "capture_protocol_sha256": registry["capture_protocol_sha256"],
        "settlement_rules_sha256": registry["settlement_rules_sha256"],
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise QuoteCaptureError("quote receipt output already exists")
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(json.dumps(value, indent=2, ensure_ascii=False).encode("utf-8"))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise QuoteCaptureError("quote receipt output already exists") from exc
        os.unlink(temporary)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a non-authorizing private bookmaker quote receipt."
    )
    parser.add_argument("--raw-payload", type=Path, required=True)
    parser.add_argument("--extraction-payload", type=Path, required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--source-record-id", required=True)
    parser.add_argument("--capture-protocol-sha256", required=True)
    parser.add_argument("--settlement-rules-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = build_quote_receipt(
        raw_source_payload=args.raw_payload.read_bytes(),
        extraction_payload_raw=args.extraction_payload.read_bytes(),
        source=args.source,
        source_url=args.source_url,
        source_record_id=args.source_record_id,
        capture_protocol_sha256=args.capture_protocol_sha256,
        settlement_rules_sha256=args.settlement_rules_sha256,
    )
    _atomic_write_json(args.output, receipt)
    print(
        json.dumps(
            {
                "status": "candidate_written",
                "authorizing": False,
                "output": str(args.output),
                "quote_sha256": sha256_json(receipt),
                "next_required_step": "independently review and pin a registry containing this digest",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
