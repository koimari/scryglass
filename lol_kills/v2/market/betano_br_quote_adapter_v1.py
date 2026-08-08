"""Replayable Betano Brazil pre-event map-winner quote capture candidate.

This module is intentionally non-authorizing.  It captures the exact public
pre-event HTML response, extracts one exact ``Vencedor do mapa`` market from
Betano's embedded ``window["initial_state"]`` JSON, and binds the result to a
system-clocked and monotonic transport window plus an earlier event-probability
receipt.  Independent source-adapter registration, quote registration,
bookmaker-terms alignment, phase-two opening, actual-map-start evidence, and
market authority are all still required before the receipt can be used.

The live transport launches a fresh, unauthenticated Brave profile through a
pinned Playwright CLI version.  It never submits credentials, clicks a price,
opens a bet slip, or persists request headers, cookies, or account data.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import urlparse

from lol_kills import bookmaker_quote_capture as generic_quote
from lol_kills.v2.data.common import sha256_canonical_object

from .event_probability_v1 import (
    EventProbabilityError,
    validate_event_probability_receipt,
)
from .match_winner_future_protocol_registry_v1 import (
    MatchWinnerFutureProtocolRegistryError,
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC,
    REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
    REGISTERED_SETTLEMENT_CONTRACT_SHA256,
    validate_registered_match_winner_future_protocol_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/betano_br_quote_adapter_v1.py"
SCHEMA_VERSION = "scryglass:betano-br-map-winner-quote-transport:v1"
CANDIDATE_SCHEMA_VERSION = "scryglass:betano-br-quote-adapter-candidate:v1"
RESULT_STATE = "QUOTE_AND_TRANSPORT_CAPTURED_NON_AUTHORIZING"
CANDIDATE_RESULT_STATE = "SOURCE_ADAPTER_IMPLEMENTED_NOT_INDEPENDENTLY_REGISTERED"
ADAPTER_ID = "betano-br-pre-event-html-map-winner-v1"
TRANSPORT_ID = "betano-br-public-brave-playwright-v1"
BOOKMAKER_ID = "betano-brazil"
MARKET_TYPE = "match_winner"
SETTLEMENT_RULE_ID = "betano-br-map-winner-shadow-v1"
BETANO_HOST = "www.betano.bet.br"
PLAYWRIGHT_CLI_PACKAGE = "@playwright/cli"
PLAYWRIGHT_CLI_VERSION = "0.1.17"
BRAVE_EXECUTABLE = Path(
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser"
)
DEFAULT_CANDIDATE_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/quote-adapter-candidate-v1.json"
)
INITIAL_STATE_PREFIX = 'window["initial_state"]='
MAX_RESPONSE_BYTES = generic_quote.MAX_SOURCE_PAYLOAD_BYTES
MAX_SUBPROCESS_STDOUT_BYTES = 12_000_000
MAX_PREDICTION_TO_RESPONSE_SECONDS = 30.0
MIN_RESPONSE_TO_MARKET_CLOSE_SECONDS = 5.0
EVENT_ID_RE = re.compile(r"^[0-9]+$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SELECTION_RE = re.compile(r"^winner:[^\s].*$")
EVENT_PATH_RE = re.compile(r"^/odds/[a-z0-9-]+/([0-9]+)/$")
CONTENT_TYPE_RE = re.compile(
    r"^text/html(?:\s*;\s*charset=(?:utf-8|\"utf-8\"))?$", re.IGNORECASE
)

AUTHORITY = {
    "source_adapter_authority": False,
    "quote_identity_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "betting_authority": False,
}
DECISION_OUTPUTS = {
    "probability": None,
    "fair_odds": None,
    "expected_value": None,
    "edge": None,
    "recommendation": None,
    "stake": None,
}
CLAIM_CEILING = (
    "Exact public Betano pre-event response bytes, deterministic map-winner "
    "extraction, and transport timing candidate only. Independent adapter, "
    "quote, terms, phase-two, actual-map-start, probability, and market "
    "authority remain absent; this receipt cannot authorize a wager."
)
CANDIDATE_CLAIM_CEILING = (
    "Implemented and source-hashed Betano Brazil quote adapter candidate only. "
    "It is not an independent adapter registration, quote, phase-two opening, "
    "or betting authority."
)


class BetanoQuoteAdapterError(ValueError):
    """The source response, extraction, transport, or binding failed closed."""


@dataclass(frozen=True)
class PublicDocumentResponse:
    """Safe public response material returned by a transport implementation."""

    http_status: int
    final_url: str
    content_type: str
    response_body: bytes
    browser_request_started_at_utc: str | None = None
    browser_response_body_read_at_utc: str | None = None


class PublicDocumentFetcher(Protocol):
    """Transport boundary used by :func:`capture_betano_map_winner_quote_v1`."""

    transport_id: str

    def __call__(self, request_url: str) -> PublicDocumentResponse: ...


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BetanoQuoteAdapterError("value is not canonical finite JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BetanoQuoteAdapterError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json(raw: str, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                BetanoQuoteAdapterError(
                    f"non-finite JSON number in {label}: {token}"
                )
            ),
        )
    except BetanoQuoteAdapterError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BetanoQuoteAdapterError(f"{label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise BetanoQuoteAdapterError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise BetanoQuoteAdapterError(f"{label} keys changed")


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BetanoQuoteAdapterError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise BetanoQuoteAdapterError(f"{label} must be a lowercase SHA-256")
    return value


def _time(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BetanoQuoteAdapterError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise BetanoQuoteAdapterError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], label: str) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise BetanoQuoteAdapterError(
            f"{label} clock must return a timezone-aware datetime"
        )
    return observed.astimezone(timezone.utc)


def _monotonic_sample(monotonic_ns: Callable[[], int], label: str) -> int:
    observed = monotonic_ns()
    if isinstance(observed, bool) or not isinstance(observed, int) or observed < 0:
        raise BetanoQuoteAdapterError(f"{label} must return non-negative integer ns")
    return observed


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise BetanoQuoteAdapterError(f"{label} must be an integer >= {minimum}")
    return value


def _price(value: Any, label: str) -> float:
    if type(value) not in (int, float):
        raise BetanoQuoteAdapterError(f"{label} must be numeric decimal odds")
    number = float(value)
    if not math.isfinite(number) or not 1.0 < number <= 100.0:
        raise BetanoQuoteAdapterError(f"{label} must be decimal odds in (1, 100]")
    return number


def _source_file_lock(root: Path = ROOT) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise BetanoQuoteAdapterError("adapter source must be a regular non-symlink")
    raw = path.read_bytes()
    if not raw:
        raise BetanoQuoteAdapterError("adapter source is empty")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": len(raw),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
    }


def _event_url(value: Any, *, expected_event_id: str | None = None) -> str:
    url = _nonempty(value, "request_url")
    parsed = urlparse(url)
    match = EVENT_PATH_RE.fullmatch(parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.hostname != BETANO_HOST
        or parsed.port is not None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.params
        or parsed.query
        or parsed.fragment
        or match is None
    ):
        raise BetanoQuoteAdapterError("request URL is not a frozen public event URL")
    event_id = match.group(1)
    if expected_event_id is not None and event_id != expected_event_id:
        raise BetanoQuoteAdapterError("request URL event id mismatch")
    return url


class _InitialStateScriptParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self._inside_script = False
        self._chunks: list[str] = []
        self.scripts: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        if tag.casefold() == "script":
            if self._inside_script:
                raise BetanoQuoteAdapterError("nested script element is invalid")
            self._inside_script = True
            self._chunks = []

    def handle_endtag(self, tag: str) -> None:
        if tag.casefold() == "script" and self._inside_script:
            self.scripts.append("".join(self._chunks))
            self._inside_script = False
            self._chunks = []

    def handle_data(self, data: str) -> None:
        if self._inside_script:
            self._chunks.append(data)

    def close(self) -> None:
        super().close()
        if self._inside_script:
            raise BetanoQuoteAdapterError("unterminated script element")


def parse_initial_state_v1(raw_response_body: bytes) -> dict[str, Any]:
    """Extract exactly one strict embedded Betano initial-state object."""

    if not isinstance(raw_response_body, bytes):
        raise BetanoQuoteAdapterError("response body must be exact bytes")
    if not raw_response_body or len(raw_response_body) > MAX_RESPONSE_BYTES:
        raise BetanoQuoteAdapterError("response body is empty or exceeds size limit")
    if raw_response_body.startswith(b"\xef\xbb\xbf") or b"\x00" in raw_response_body:
        raise BetanoQuoteAdapterError("response body has a forbidden encoding marker")
    marker = INITIAL_STATE_PREFIX.encode("ascii")
    if raw_response_body.count(marker) != 1:
        raise BetanoQuoteAdapterError("initial-state marker must occur exactly once")
    try:
        html = raw_response_body.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise BetanoQuoteAdapterError("response body is not strict UTF-8") from exc
    parser = _InitialStateScriptParser()
    try:
        parser.feed(html)
        parser.close()
    except BetanoQuoteAdapterError:
        raise
    except Exception as exc:
        raise BetanoQuoteAdapterError("response HTML could not be parsed") from exc
    candidates = [
        script[len(INITIAL_STATE_PREFIX) :]
        for script in parser.scripts
        if script.startswith(INITIAL_STATE_PREFIX)
    ]
    if len(candidates) != 1:
        raise BetanoQuoteAdapterError(
            "exactly one initial-state script at byte-zero content is required"
        )
    return _strict_json(candidates[0], "Betano initial_state")


def _not_suspended(value: Mapping[str, Any], label: str) -> None:
    for key in ("isSuspended", "suspended"):
        state = value.get(key, False)
        if state is not False:
            raise BetanoQuoteAdapterError(f"{label}.{key} is not explicitly active")


def _no_outcome_fields(value: Any, label: str = "event") -> None:
    forbidden = {
        "result",
        "results",
        "winner",
        "winningParticipant",
        "score",
        "scores",
        "homeScore",
        "awayScore",
        "settled",
        "settlement",
        "outcome",
        "outcomes",
    }
    if isinstance(value, Mapping):
        overlap = forbidden.intersection(value)
        if overlap:
            raise BetanoQuoteAdapterError(
                f"{label} contains forbidden outcome field {sorted(overlap)[0]!r}"
            )
        for key, item in value.items():
            _no_outcome_fields(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _no_outcome_fields(item, f"{label}[{index}]")


def _participant_bindings(
    raw: Any,
    *,
    probability_selection: str,
    probability_opposing_selection: str,
) -> list[dict[str, str]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
        raise BetanoQuoteAdapterError("participant bindings must contain exactly two rows")
    expected_selections = {
        probability_selection,
        probability_opposing_selection,
    }
    rows: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise BetanoQuoteAdapterError(f"participant binding {index} is malformed")
        _exact_keys(
            item,
            {"canonical_selection", "bookmaker_participant_id", "bookmaker_name"},
            f"participant binding {index}",
        )
        selection = _nonempty(
            item.get("canonical_selection"),
            f"participant binding {index}.canonical_selection",
        )
        if not SELECTION_RE.fullmatch(selection):
            raise BetanoQuoteAdapterError("canonical selection must be winner:<team>")
        participant_id = _nonempty(
            item.get("bookmaker_participant_id"),
            f"participant binding {index}.bookmaker_participant_id",
        )
        if not EVENT_ID_RE.fullmatch(participant_id):
            raise BetanoQuoteAdapterError("bookmaker participant id must be numeric")
        rows.append(
            {
                "canonical_selection": selection,
                "bookmaker_participant_id": participant_id,
                "bookmaker_name": _nonempty(
                    item.get("bookmaker_name"),
                    f"participant binding {index}.bookmaker_name",
                ),
            }
        )
    if {row["canonical_selection"] for row in rows} != expected_selections:
        raise BetanoQuoteAdapterError(
            "participant bindings do not match the exact probability selections"
        )
    if len({row["bookmaker_participant_id"] for row in rows}) != 2:
        raise BetanoQuoteAdapterError("bookmaker participant ids are duplicated")
    if len({row["bookmaker_name"] for row in rows}) != 2:
        raise BetanoQuoteAdapterError("bookmaker participant names are duplicated")
    return sorted(rows, key=lambda row: row["canonical_selection"])


def extract_map_winner_v1(
    *,
    raw_response_body: bytes,
    response_received_at_utc: datetime,
    expected_betano_event_id: str,
    expected_league: str,
    map_number: int,
    participant_bindings: Sequence[Mapping[str, str]],
    probability_selection: str,
    probability_opposing_selection: str,
) -> dict[str, Any]:
    """Replay one exact open two-way map-winner extraction from response bytes."""

    if response_received_at_utc.tzinfo is None:
        raise BetanoQuoteAdapterError("response receive time must include a timezone")
    response_received = response_received_at_utc.astimezone(timezone.utc)
    event_id = _nonempty(expected_betano_event_id, "expected_betano_event_id")
    if not EVENT_ID_RE.fullmatch(event_id):
        raise BetanoQuoteAdapterError("Betano event id must be numeric")
    league = _nonempty(expected_league, "expected_league")
    map_index = _integer(map_number, "map_number", minimum=1)
    if map_index > 5:
        raise BetanoQuoteAdapterError("map_number exceeds the frozen best-of-five bound")
    bindings = _participant_bindings(
        participant_bindings,
        probability_selection=probability_selection,
        probability_opposing_selection=probability_opposing_selection,
    )
    state = parse_initial_state_v1(raw_response_body)
    data = state.get("data")
    if not isinstance(data, Mapping):
        raise BetanoQuoteAdapterError("initial_state.data is missing")
    event = data.get("event")
    if not isinstance(event, Mapping):
        raise BetanoQuoteAdapterError("initial_state.data.event is missing")
    _no_outcome_fields(event)
    _not_suspended(event, "event")
    if (
        event.get("id") != event_id
        or event.get("sportId") != "ESPS"
        or event.get("regionName") != "League of Legends"
        or event.get("leagueName") != league
        or event.get("leagueDescription") != league
    ):
        raise BetanoQuoteAdapterError("event identity does not match the frozen binding")
    event_name = _nonempty(event.get("name"), "event.name")
    short_name = _nonempty(event.get("shortName"), "event.shortName")
    if event_name != short_name:
        raise BetanoQuoteAdapterError("event name and short name differ")
    event_url = _nonempty(event.get("url"), "event.url")
    url_path = urlparse(event_url).path
    match = EVENT_PATH_RE.fullmatch(url_path)
    if match is None or match.group(1) != event_id:
        raise BetanoQuoteAdapterError("embedded event URL identity changed")
    start_time_ms = _integer(event.get("startTime"), "event.startTime", minimum=1)
    participants = event.get("participants")
    if not isinstance(participants, list) or len(participants) != 2:
        raise BetanoQuoteAdapterError("event must contain exactly two participants")
    source_participants: list[dict[str, str]] = []
    for index, participant in enumerate(participants):
        if not isinstance(participant, Mapping):
            raise BetanoQuoteAdapterError(f"participant {index} is malformed")
        participant_id = _nonempty(participant.get("id"), f"participant {index}.id")
        if not EVENT_ID_RE.fullmatch(participant_id):
            raise BetanoQuoteAdapterError("source participant id must be numeric")
        source_participants.append(
            {
                "bookmaker_participant_id": participant_id,
                "bookmaker_name": _nonempty(
                    participant.get("name"), f"participant {index}.name"
                ),
            }
        )
    if len({item["bookmaker_participant_id"] for item in source_participants}) != 2:
        raise BetanoQuoteAdapterError("source participant ids are duplicated")
    if len({item["bookmaker_name"] for item in source_participants}) != 2:
        raise BetanoQuoteAdapterError("source participant names are duplicated")
    expected_source = {
        (row["bookmaker_participant_id"], row["bookmaker_name"])
        for row in bindings
    }
    observed_source = {
        (row["bookmaker_participant_id"], row["bookmaker_name"])
        for row in source_participants
    }
    if observed_source != expected_source:
        raise BetanoQuoteAdapterError("participant source identity binding mismatch")
    expected_event_names = {
        f"{participants[0]['name']} - {participants[1]['name']}",
        f"{participants[1]['name']} - {participants[0]['name']}",
    }
    if event_name not in expected_event_names:
        raise BetanoQuoteAdapterError("event name does not match its participants")

    markets = event.get("markets")
    if not isinstance(markets, list):
        raise BetanoQuoteAdapterError("event markets are missing")
    expected_market_name = f"Vencedor do mapa (Mapa {map_index})"
    expected_pin_key = f"p_TMPW{map_index - 1}"
    candidates = [
        market
        for market in markets
        if isinstance(market, Mapping)
        and market.get("type") == "TMPW"
        and market.get("typeId") == 3185
        and market.get("name") == expected_market_name
        and market.get("pinKey") == expected_pin_key
    ]
    if len(candidates) != 1:
        raise BetanoQuoteAdapterError("exact map-winner market is missing or ambiguous")
    market = candidates[0]
    _not_suspended(market, "market")
    market_id = _nonempty(market.get("id"), "market.id")
    if (
        not EVENT_ID_RE.fullmatch(market_id)
        or market.get("uniqueId") != market_id
        or type(market.get("handicap")) not in (int, float)
        or float(market["handicap"]) != 0.0
    ):
        raise BetanoQuoteAdapterError("market identity or handicap changed")
    close_time_ms = _integer(
        market.get("marketCloseTimeMillis"), "market.marketCloseTimeMillis", minimum=1
    )
    response_ms = int(response_received.timestamp() * 1000)
    if close_time_ms - response_ms < int(MIN_RESPONSE_TO_MARKET_CLOSE_SECONDS * 1000):
        raise BetanoQuoteAdapterError("market closes too near or before response receipt")
    selections = market.get("selections")
    if not isinstance(selections, list) or len(selections) != 2:
        raise BetanoQuoteAdapterError("map-winner market must have exactly two selections")
    binding_by_name = {row["bookmaker_name"]: row for row in bindings}
    selection_rows: list[dict[str, Any]] = []
    prices: dict[str, float] = {}
    for index, selection in enumerate(selections):
        if not isinstance(selection, Mapping):
            raise BetanoQuoteAdapterError(f"selection {index} is malformed")
        _not_suspended(selection, f"selection {index}")
        source_name = _nonempty(selection.get("name"), f"selection {index}.name")
        binding = binding_by_name.get(source_name)
        if binding is None:
            raise BetanoQuoteAdapterError("selection name is not a bound participant")
        selection_id = _nonempty(selection.get("id"), f"selection {index}.id")
        bet_ref = _nonempty(selection.get("betRef"), f"selection {index}.betRef")
        if (
            not EVENT_ID_RE.fullmatch(selection_id)
            or bet_ref != selection_id
            or type(selection.get("handicap")) not in (int, float)
            or float(selection["handicap"]) != 0.0
        ):
            raise BetanoQuoteAdapterError("selection identity or handicap changed")
        price = _price(selection.get("price"), f"selection {index}.price")
        canonical_selection = binding["canonical_selection"]
        if canonical_selection in prices:
            raise BetanoQuoteAdapterError("canonical selection is duplicated")
        prices[canonical_selection] = price
        selection_rows.append(
            {
                "selection_id": selection_id,
                "bet_ref": bet_ref,
                "bookmaker_name": source_name,
                "bookmaker_participant_id": binding["bookmaker_participant_id"],
                "canonical_selection": canonical_selection,
                "decimal_price": price,
            }
        )
    if set(prices) != {probability_selection, probability_opposing_selection}:
        raise BetanoQuoteAdapterError("extracted prices do not bind both predictions")
    if len({row["selection_id"] for row in selection_rows}) != 2:
        raise BetanoQuoteAdapterError("selection ids are duplicated")
    if market.get("scorerSelections", []) != [] or market.get(
        "exactScoreSelections", []
    ) != []:
        raise BetanoQuoteAdapterError("map-winner market contains unexpected selections")
    return {
        "source_payload_sha256": hashlib.sha256(raw_response_body).hexdigest(),
        "betano_event": {
            "event_id": event_id,
            "event_name": event_name,
            "sport_id": "ESPS",
            "region_name": "League of Legends",
            "region_id": _nonempty(event.get("regionId"), "event.regionId"),
            "league_id": _nonempty(event.get("leagueId"), "event.leagueId"),
            "league_name": league,
            "scheduled_series_start_epoch_ms": start_time_ms,
            "participants": bindings,
        },
        "market": {
            "market_id": market_id,
            "unique_id": market_id,
            "market_name": expected_market_name,
            "market_type_code": "TMPW",
            "market_type_id": 3185,
            "map_number": map_index,
            "pin_key": expected_pin_key,
            "market_close_epoch_ms": close_time_ms,
            "status": "open",
            "open_rule": (
                "event_market_and_selections_not_suspended_with_two_valid_prices_"
                "and_close_at_least_5s_after_response"
            ),
            "selections": sorted(
                selection_rows, key=lambda row: row["canonical_selection"]
            ),
        },
        "prices": dict(sorted(prices.items())),
        "source_schema": {
            "initial_state_assignment": INITIAL_STATE_PREFIX,
            "event_sport_id": "ESPS",
            "map_winner_type": "TMPW",
            "map_winner_type_id": 3185,
            "event_market_and_selection_suspension_checked": True,
            "missing_suspension_field_means_false_in_source_state": True,
            "market_close_time_checked_against_transport_response": True,
            "cash_or_promotional_execution_proven": False,
        },
    }


def _playwright_result(stdout: str) -> dict[str, Any]:
    if not isinstance(stdout, str) or len(stdout.encode("utf-8")) > MAX_SUBPROCESS_STDOUT_BYTES:
        raise BetanoQuoteAdapterError("Playwright output is missing or exceeds limit")
    start_marker = "### Result\n"
    end_marker = "\n### Ran Playwright code"
    start = stdout.find(start_marker)
    end = stdout.find(end_marker, start + len(start_marker))
    if start < 0 or end < 0 or stdout.find(start_marker, start + 1) >= 0:
        raise BetanoQuoteAdapterError("Playwright result framing changed")
    raw = stdout[start + len(start_marker) : end].strip()
    return _strict_json(raw, "Playwright public response result")


class BravePlaywrightPublicDocumentFetcher:
    """Fresh, public, unauthenticated Brave transport for one capture session."""

    transport_id = TRANSPORT_ID

    def __init__(
        self,
        *,
        brave_executable: Path = BRAVE_EXECUTABLE,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self._brave_executable = brave_executable
        self._runner = runner
        self._temporary: tempfile.TemporaryDirectory[str] | None = None
        self._session_name: str | None = None
        self._prepared = False

    def _base_command(self) -> list[str]:
        return [
            "npx",
            "--yes",
            "--package",
            f"{PLAYWRIGHT_CLI_PACKAGE}@{PLAYWRIGHT_CLI_VERSION}",
            "playwright-cli",
        ]

    def _run(
        self, arguments: Sequence[str], *, timeout_seconds: float
    ) -> subprocess.CompletedProcess[str]:
        safe_environment = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL", "SHELL")
            if key in os.environ
        }
        safe_environment["NO_COLOR"] = "1"
        safe_environment["NPM_CONFIG_USERCONFIG"] = "/dev/null"
        safe_environment["NPM_CONFIG_AUDIT"] = "false"
        safe_environment["NPM_CONFIG_FUND"] = "false"
        safe_environment["NPM_CONFIG_UPDATE_NOTIFIER"] = "false"
        try:
            completed = self._runner(
                [*self._base_command(), *arguments],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=safe_environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise BetanoQuoteAdapterError("pinned Playwright transport failed") from exc
        if not isinstance(completed, subprocess.CompletedProcess):
            raise BetanoQuoteAdapterError("Playwright runner returned an invalid result")
        if completed.returncode != 0:
            raise BetanoQuoteAdapterError("pinned Playwright transport returned nonzero")
        if len((completed.stdout or "").encode("utf-8")) > MAX_SUBPROCESS_STDOUT_BYTES:
            raise BetanoQuoteAdapterError("Playwright stdout exceeds the frozen limit")
        if len((completed.stderr or "").encode("utf-8")) > 1_000_000:
            raise BetanoQuoteAdapterError("Playwright stderr exceeds the frozen limit")
        return completed

    def __enter__(self) -> "BravePlaywrightPublicDocumentFetcher":
        if self._prepared:
            raise BetanoQuoteAdapterError("Brave transport is already prepared")
        if (
            not self._brave_executable.is_file()
            or self._brave_executable.is_symlink()
        ):
            raise BetanoQuoteAdapterError("frozen Brave executable is unavailable")
        version = self._run(["--version"], timeout_seconds=30).stdout.strip()
        if version != PLAYWRIGHT_CLI_VERSION:
            raise BetanoQuoteAdapterError("Playwright CLI version drifted")
        self._temporary = tempfile.TemporaryDirectory(
            prefix="scryglass-betano-quote-v1-"
        )
        temporary = Path(self._temporary.name)
        profile = temporary / "brave-profile"
        config_path = temporary / "cli-config.json"
        config = {
            "browser": {
                "browserName": "chromium",
                "isolated": False,
                "userDataDir": str(profile),
                "launchOptions": {
                    "executablePath": str(self._brave_executable),
                    "headless": False,
                },
                "contextOptions": {
                    "locale": "pt-BR",
                    "timezoneId": "America/Sao_Paulo",
                },
            }
        }
        descriptor = os.open(
            config_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical_bytes(config))
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            config_path.unlink(missing_ok=True)
            self._temporary.cleanup()
            self._temporary = None
            raise
        session_entropy = hashlib.sha256(
            f"{time.monotonic_ns()}:{os.getpid()}:{temporary}".encode("utf-8")
        ).hexdigest()[:16]
        self._session_name = f"betano-quote-v1-{session_entropy}"
        try:
            self._run(
                [
                    f"-s={self._session_name}",
                    "open",
                    "about:blank",
                    "--config",
                    str(config_path),
                ],
                timeout_seconds=45,
            )
        except Exception:
            self._temporary.cleanup()
            self._temporary = None
            self._session_name = None
            raise
        self._prepared = True
        return self

    def __call__(self, request_url: str) -> PublicDocumentResponse:
        if not self._prepared or self._session_name is None:
            raise BetanoQuoteAdapterError("Brave transport must be prepared first")
        url = _event_url(request_url)
        encoded_url = json.dumps(url, ensure_ascii=True)
        code = (
            "async (page) => {"
            f"const requestUrl={encoded_url};"
            "const browserRequestStartedAtUtc=new Date().toISOString();"
            "const response=await page.goto(requestUrl,{waitUntil:'commit',timeout:30000});"
            "if(!response)throw new Error('missing document response');"
            "const body=await response.body();"
            "const browserResponseBodyReadAtUtc=new Date().toISOString();"
            "return {"
            "schemaVersion:'scryglass-playwright-public-document-v1',"
            "browserRequestStartedAtUtc,"
            "browserResponseBodyReadAtUtc,"
            "httpStatus:response.status(),"
            "finalUrl:response.url(),"
            "contentType:response.headers()['content-type']||'',"
            "responseBodyBase64:body.toString('base64')"
            "};}"
        )
        completed = self._run(
            [f"-s={self._session_name}", "run-code", code],
            timeout_seconds=45,
        )
        result = _playwright_result(completed.stdout)
        _exact_keys(
            result,
            {
                "schemaVersion",
                "browserRequestStartedAtUtc",
                "browserResponseBodyReadAtUtc",
                "httpStatus",
                "finalUrl",
                "contentType",
                "responseBodyBase64",
            },
            "Playwright public response",
        )
        if result.get("schemaVersion") != "scryglass-playwright-public-document-v1":
            raise BetanoQuoteAdapterError("Playwright response schema changed")
        try:
            response_body = base64.b64decode(
                _nonempty(result.get("responseBodyBase64"), "responseBodyBase64"),
                validate=True,
            )
        except (TypeError, ValueError) as exc:
            raise BetanoQuoteAdapterError("Playwright response body is not base64") from exc
        if (
            not response_body
            or len(response_body) > MAX_RESPONSE_BYTES
            or base64.b64encode(response_body).decode("ascii")
            != result.get("responseBodyBase64")
        ):
            raise BetanoQuoteAdapterError("Playwright response body is invalid")
        return PublicDocumentResponse(
            http_status=_integer(result.get("httpStatus"), "httpStatus", minimum=100),
            final_url=_nonempty(result.get("finalUrl"), "finalUrl"),
            content_type=_nonempty(result.get("contentType"), "contentType"),
            response_body=response_body,
            browser_request_started_at_utc=_time(
                result.get("browserRequestStartedAtUtc"),
                "browserRequestStartedAtUtc",
            ).isoformat(),
            browser_response_body_read_at_utc=_time(
                result.get("browserResponseBodyReadAtUtc"),
                "browserResponseBodyReadAtUtc",
            ).isoformat(),
        )

    def close(self) -> None:
        if not self._prepared:
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
            return
        error: Exception | None = None
        try:
            if self._session_name is not None:
                self._run(
                    [f"-s={self._session_name}", "close"], timeout_seconds=20
                )
        except Exception as exc:  # cleanup still runs
            error = exc
        finally:
            self._prepared = False
            self._session_name = None
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None
        if error is not None:
            raise BetanoQuoteAdapterError("Brave transport cleanup failed") from error

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        try:
            self.close()
        except BetanoQuoteAdapterError:
            if exc is None:
                raise
        return False


def _response(
    value: Any,
    *,
    request_url: str,
    request_started: datetime,
    response_received: datetime,
) -> PublicDocumentResponse:
    if not isinstance(value, PublicDocumentResponse):
        raise BetanoQuoteAdapterError("transport did not return PublicDocumentResponse")
    if value.http_status != 200:
        raise BetanoQuoteAdapterError("Betano public event response was not HTTP 200")
    if value.final_url != request_url:
        raise BetanoQuoteAdapterError("Betano public event response redirected")
    if not CONTENT_TYPE_RE.fullmatch(value.content_type.strip()):
        raise BetanoQuoteAdapterError("Betano response content type changed")
    if not value.response_body or len(value.response_body) > MAX_RESPONSE_BYTES:
        raise BetanoQuoteAdapterError("Betano response body is empty or too large")
    browser_started: datetime | None = None
    browser_read: datetime | None = None
    if (
        value.browser_request_started_at_utc is None
        or value.browser_response_body_read_at_utc is None
    ):
        raise BetanoQuoteAdapterError("browser transport clocks are required")
    browser_started = _time(
        value.browser_request_started_at_utc,
        "browser_request_started_at_utc",
    )
    browser_read = _time(
        value.browser_response_body_read_at_utc,
        "browser_response_body_read_at_utc",
    )
    tolerance = 1.0
    if (
        browser_read < browser_started
        or (browser_started - request_started).total_seconds() < -tolerance
        or (response_received - browser_read).total_seconds() < -tolerance
    ):
        raise BetanoQuoteAdapterError("browser clocks escape the adapter envelope")
    return PublicDocumentResponse(
        http_status=value.http_status,
        final_url=value.final_url,
        content_type=value.content_type.strip(),
        response_body=value.response_body,
        browser_request_started_at_utc=browser_started.isoformat(),
        browser_response_body_read_at_utc=browser_read.isoformat(),
    )


def capture_betano_map_winner_quote_v1(
    *,
    probability_receipt: Mapping[str, Any],
    request_url: str,
    betano_event_id: str,
    map_number: int,
    participant_bindings: Sequence[Mapping[str, str]],
    fetcher: PublicDocumentFetcher,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic_ns: Callable[[], int] = time.monotonic_ns,
) -> dict[str, Any]:
    """Capture a non-authorizing exact-body quote and transport receipt."""

    try:
        protocol = validate_registered_match_winner_future_protocol_v1(root=root)
        probability = validate_event_probability_receipt(probability_receipt)
    except (
        MatchWinnerFutureProtocolRegistryError,
        EventProbabilityError,
        OSError,
        ValueError,
    ) as exc:
        raise BetanoQuoteAdapterError(str(exc)) from exc
    event = probability["event"]
    bindings = probability["bindings"]
    if (
        event["market_type"] != MARKET_TYPE
        or bindings["market_protocol_artifact_sha256"]
        != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or protocol.get("quote_capture_contract_sha256")
        != REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256
        or protocol.get("settlement_contract_sha256")
        != REGISTERED_SETTLEMENT_CONTRACT_SHA256
    ):
        raise BetanoQuoteAdapterError("probability or protocol binding changed")
    source_lock = _source_file_lock(root)
    event_id = _nonempty(betano_event_id, "betano_event_id")
    if not EVENT_ID_RE.fullmatch(event_id):
        raise BetanoQuoteAdapterError("Betano event id must be numeric")
    url = _event_url(request_url, expected_event_id=event_id)
    checked_participants = _participant_bindings(
        participant_bindings,
        probability_selection=event["selection"],
        probability_opposing_selection=event["opposing_selection"],
    )
    if getattr(fetcher, "transport_id", None) != TRANSPORT_ID:
        raise BetanoQuoteAdapterError("transport id is not the frozen public transport")
    prediction_time = _time(probability["captured_at_utc"], "prediction.captured_at")
    request_started = _clock_sample(clock, "transport.request_started")
    if request_started < prediction_time:
        raise BetanoQuoteAdapterError("quote request predates the probability receipt")
    monotonic_started = _monotonic_sample(monotonic_ns, "transport.monotonic_started")
    try:
        raw_response = fetcher(url)
    except BetanoQuoteAdapterError:
        raise
    except Exception as exc:
        raise BetanoQuoteAdapterError("Betano public transport failed") from exc
    monotonic_received = _monotonic_sample(
        monotonic_ns, "transport.monotonic_received"
    )
    response_received = _clock_sample(clock, "transport.response_received")
    if monotonic_received <= monotonic_started:
        raise BetanoQuoteAdapterError("monotonic transport duration is not positive")
    if response_received < request_started:
        raise BetanoQuoteAdapterError("system transport clock moved backwards")
    monotonic_duration_ns = monotonic_received - monotonic_started
    wall_duration_seconds = (response_received - request_started).total_seconds()
    monotonic_duration_seconds = monotonic_duration_ns / 1_000_000_000
    if abs(monotonic_duration_seconds - wall_duration_seconds) > 0.5:
        raise BetanoQuoteAdapterError("system and monotonic transport durations diverge")
    prediction_to_response_seconds = (
        response_received - prediction_time
    ).total_seconds()
    if not 0 <= prediction_to_response_seconds <= MAX_PREDICTION_TO_RESPONSE_SECONDS:
        raise BetanoQuoteAdapterError("prediction-to-quote response window exceeded")
    response = _response(
        raw_response,
        request_url=url,
        request_started=request_started,
        response_received=response_received,
    )
    extraction = extract_map_winner_v1(
        raw_response_body=response.response_body,
        response_received_at_utc=response_received,
        expected_betano_event_id=event_id,
        expected_league=event["league"],
        map_number=map_number,
        participant_bindings=checked_participants,
        probability_selection=event["selection"],
        probability_opposing_selection=event["opposing_selection"],
    )
    price_extraction = generic_quote.build_price_extraction_payload(
        raw_source_payload=response.response_body,
        event_id=event["event_id"],
        market_type=MARKET_TYPE,
        settlement_rule_id=SETTLEMENT_RULE_ID,
        prices=extraction["prices"],
        capture_protocol_sha256=REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        settlement_rules_sha256=REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        extractor_id=ADAPTER_ID,
        extractor_sha256=source_lock["raw_sha256"],
    )
    extraction_raw = generic_quote.canonical_bytes(price_extraction)
    source_record_id = (
        f"betano-brazil:{event_id}:map-{map_number}:"
        f"market-{extraction['market']['market_id']}:"
        f"body-{extraction['source_payload_sha256'][:16]}"
    )
    quote_receipt = generic_quote.build_quote_receipt(
        raw_source_payload=response.response_body,
        extraction_payload_raw=extraction_raw,
        source="betano-br-public-pre-event-html-v1",
        source_url=url,
        source_record_id=source_record_id,
        capture_protocol_sha256=REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        settlement_rules_sha256=REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        clock=clock,
    )
    quote_checked = generic_quote.validate_quote_receipt(
        quote_receipt,
        expected_capture_protocol_sha256=REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        expected_settlement_rules_sha256=REGISTERED_SETTLEMENT_CONTRACT_SHA256,
    )
    quote_built = _time(quote_receipt["captured_at_utc"], "quote.captured_at_utc")
    if quote_built < response_received:
        raise BetanoQuoteAdapterError("generic quote receipt predates response receipt")
    if (quote_built - response_received).total_seconds() > 5.0:
        raise BetanoQuoteAdapterError("generic quote receipt was built too late")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": quote_built.isoformat(),
        "bookmaker_id": BOOKMAKER_ID,
        "market_type": MARKET_TYPE,
        "settlement_rule_id": SETTLEMENT_RULE_ID,
        "protocol_bindings": {
            "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "quote_capture_contract_sha256": REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
            "settlement_contract_sha256": REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        },
        "adapter": {
            "adapter_id": ADAPTER_ID,
            "source_locator": SOURCE_LOCATOR,
            "source_sha256": source_lock["raw_sha256"],
            "transport_id": TRANSPORT_ID,
            "playwright_cli_package": PLAYWRIGHT_CLI_PACKAGE,
            "playwright_cli_version": PLAYWRIGHT_CLI_VERSION,
            "browser_product": "Brave",
            "public_unauthenticated_profile": True,
            "request_headers_cookies_credentials_persisted": False,
            "independently_registered": False,
        },
        "source_lock": source_lock,
        "prediction_binding": {
            "event_probability_receipt_sha256": probability["receipt_sha256"],
            "event_probability_artifact_sha256": probability["artifact_sha256"],
            "prediction_captured_at_utc": prediction_time.isoformat(),
            "scryglass_event_id": event["event_id"],
            "league": event["league"],
            "market_type": event["market_type"],
            "selection": event["selection"],
            "opposing_selection": event["opposing_selection"],
            "prediction_to_request_seconds": (
                request_started - prediction_time
            ).total_seconds(),
            "prediction_to_response_seconds": prediction_to_response_seconds,
        },
        "transport": {
            "transport_id": TRANSPORT_ID,
            "method": "GET",
            "request_url": url,
            "final_url": response.final_url,
            "http_status": response.http_status,
            "content_type": response.content_type,
            "request_started_at_utc": request_started.isoformat(),
            "response_received_at_utc": response_received.isoformat(),
            "wall_transport_duration_seconds": wall_duration_seconds,
            "monotonic_transport_duration_ns": monotonic_duration_ns,
            "browser_request_started_at_utc": response.browser_request_started_at_utc,
            "browser_response_body_read_at_utc": (
                response.browser_response_body_read_at_utc
            ),
            "response_body_bytes": len(response.response_body),
            "response_body_sha256": extraction["source_payload_sha256"],
            "response_body_storage": (
                "generic_quote_receipt.source_payload_base64"
            ),
            "response_body_excludes_request_headers_cookies_credentials": True,
        },
        "source_extraction": extraction,
        "generic_quote_receipt": quote_receipt,
        "generic_quote_receipt_sha256": quote_checked["quote_sha256"],
        "qualification": {
            "prediction_receipt_preceded_quote_request": True,
            "prediction_to_response_within_30_seconds": True,
            "exact_response_body_embedded_once": True,
            "deterministic_source_extraction_replays": True,
            "market_open_under_frozen_source_rule": True,
            "exact_two_way_decimal_prices_present": True,
            "actual_map_start_checked": False,
            "quote_response_preceded_actual_map_start": None,
            "source_adapter_independently_registered": False,
            "quote_independently_registered": False,
            "bookmaker_terms_independently_registered": False,
            "phase_two_opening_independently_registered": False,
            "retrospective_backfill_qualifies": False,
            "phase_two_evidence_qualifies": False,
        },
        "actual_map_start_binding": None,
        "decision_outputs": dict(DECISION_OUTPUTS),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_betano_map_winner_quote_v1(payload, root=root)


def _decode_exact_base64(value: Any, label: str, *, maximum: int) -> bytes:
    text = _nonempty(value, label)
    try:
        raw = base64.b64decode(text, validate=True)
    except (TypeError, ValueError) as exc:
        raise BetanoQuoteAdapterError(f"{label} is not strict base64") from exc
    if (
        not raw
        or len(raw) > maximum
        or base64.b64encode(raw).decode("ascii") != text
    ):
        raise BetanoQuoteAdapterError(f"{label} is not canonical bounded base64")
    return raw


def _expected_adapter(source_lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "adapter_id": ADAPTER_ID,
        "source_locator": SOURCE_LOCATOR,
        "source_sha256": source_lock["raw_sha256"],
        "transport_id": TRANSPORT_ID,
        "playwright_cli_package": PLAYWRIGHT_CLI_PACKAGE,
        "playwright_cli_version": PLAYWRIGHT_CLI_VERSION,
        "browser_product": "Brave",
        "public_unauthenticated_profile": True,
        "request_headers_cookies_credentials_persisted": False,
        "independently_registered": False,
    }


def _expected_qualification() -> dict[str, Any]:
    return {
        "prediction_receipt_preceded_quote_request": True,
        "prediction_to_response_within_30_seconds": True,
        "exact_response_body_embedded_once": True,
        "deterministic_source_extraction_replays": True,
        "market_open_under_frozen_source_rule": True,
        "exact_two_way_decimal_prices_present": True,
        "actual_map_start_checked": False,
        "quote_response_preceded_actual_map_start": None,
        "source_adapter_independently_registered": False,
        "quote_independently_registered": False,
        "bookmaker_terms_independently_registered": False,
        "phase_two_opening_independently_registered": False,
        "retrospective_backfill_qualifies": False,
        "phase_two_evidence_qualifies": False,
    }


def validate_betano_map_winner_quote_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Replay and validate one non-authorizing quote/transport bundle."""

    if not isinstance(payload, Mapping):
        raise BetanoQuoteAdapterError("Betano quote bundle must be an object")
    value = dict(payload)
    _exact_keys(
        value,
        {
            "schema_version",
            "result_state",
            "captured_at_utc",
            "bookmaker_id",
            "market_type",
            "settlement_rule_id",
            "protocol_bindings",
            "adapter",
            "source_lock",
            "prediction_binding",
            "transport",
            "source_extraction",
            "generic_quote_receipt",
            "generic_quote_receipt_sha256",
            "qualification",
            "actual_map_start_binding",
            "decision_outputs",
            "authority",
            "claim_ceiling",
            "artifact_sha256",
        },
        "Betano quote bundle",
    )
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
        or value.get("bookmaker_id") != BOOKMAKER_ID
        or value.get("market_type") != MARKET_TYPE
        or value.get("settlement_rule_id") != SETTLEMENT_RULE_ID
    ):
        raise BetanoQuoteAdapterError("Betano quote bundle identity changed")
    unsigned = dict(value)
    artifact_sha256 = unsigned.pop("artifact_sha256", None)
    if artifact_sha256 != sha256_canonical_object(unsigned):
        raise BetanoQuoteAdapterError("Betano quote bundle artifact hash changed")
    if expected_artifact_sha256 is not None and artifact_sha256 != _sha(
        expected_artifact_sha256, "expected_artifact_sha256"
    ):
        raise BetanoQuoteAdapterError("Betano quote bundle digest mismatch")
    captured_at = _time(value.get("captured_at_utc"), "captured_at_utc")
    try:
        protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    except (MatchWinnerFutureProtocolRegistryError, OSError, ValueError) as exc:
        raise BetanoQuoteAdapterError(str(exc)) from exc
    protocol_bindings = value.get("protocol_bindings")
    expected_protocol_bindings = {
        "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "quote_capture_contract_sha256": REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        "settlement_contract_sha256": REGISTERED_SETTLEMENT_CONTRACT_SHA256,
    }
    if (
        protocol_bindings != expected_protocol_bindings
        or protocol.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or protocol.get("quote_capture_contract_sha256")
        != REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256
        or protocol.get("settlement_contract_sha256")
        != REGISTERED_SETTLEMENT_CONTRACT_SHA256
    ):
        raise BetanoQuoteAdapterError("registered market protocol binding changed")
    source_lock = _source_file_lock(root)
    if value.get("source_lock") != source_lock:
        raise BetanoQuoteAdapterError("Betano adapter source lock changed")
    if value.get("adapter") != _expected_adapter(source_lock):
        raise BetanoQuoteAdapterError("Betano adapter identity changed")

    prediction = value.get("prediction_binding")
    if not isinstance(prediction, Mapping):
        raise BetanoQuoteAdapterError("prediction binding is malformed")
    _exact_keys(
        prediction,
        {
            "event_probability_receipt_sha256",
            "event_probability_artifact_sha256",
            "prediction_captured_at_utc",
            "scryglass_event_id",
            "league",
            "market_type",
            "selection",
            "opposing_selection",
            "prediction_to_request_seconds",
            "prediction_to_response_seconds",
        },
        "prediction binding",
    )
    _sha(
        prediction.get("event_probability_receipt_sha256"),
        "event_probability_receipt_sha256",
    )
    _sha(
        prediction.get("event_probability_artifact_sha256"),
        "event_probability_artifact_sha256",
    )
    prediction_time = _time(
        prediction.get("prediction_captured_at_utc"),
        "prediction_captured_at_utc",
    )
    scryglass_event_id = _nonempty(
        prediction.get("scryglass_event_id"), "scryglass_event_id"
    )
    league = _nonempty(prediction.get("league"), "prediction.league")
    selection = _nonempty(prediction.get("selection"), "prediction.selection")
    opposing_selection = _nonempty(
        prediction.get("opposing_selection"), "prediction.opposing_selection"
    )
    if (
        prediction.get("market_type") != MARKET_TYPE
        or selection == opposing_selection
        or not SELECTION_RE.fullmatch(selection)
        or not SELECTION_RE.fullmatch(opposing_selection)
    ):
        raise BetanoQuoteAdapterError("prediction market or selections changed")

    transport = value.get("transport")
    if not isinstance(transport, Mapping):
        raise BetanoQuoteAdapterError("transport record is malformed")
    _exact_keys(
        transport,
        {
            "transport_id",
            "method",
            "request_url",
            "final_url",
            "http_status",
            "content_type",
            "request_started_at_utc",
            "response_received_at_utc",
            "wall_transport_duration_seconds",
            "monotonic_transport_duration_ns",
            "browser_request_started_at_utc",
            "browser_response_body_read_at_utc",
            "response_body_bytes",
            "response_body_sha256",
            "response_body_storage",
            "response_body_excludes_request_headers_cookies_credentials",
        },
        "transport record",
    )
    extraction = value.get("source_extraction")
    if not isinstance(extraction, Mapping):
        raise BetanoQuoteAdapterError("source extraction is malformed")
    betano_event = extraction.get("betano_event")
    market = extraction.get("market")
    if not isinstance(betano_event, Mapping) or not isinstance(market, Mapping):
        raise BetanoQuoteAdapterError("source extraction bindings are malformed")
    betano_event_id = _nonempty(betano_event.get("event_id"), "betano_event.id")
    request_url = _event_url(
        transport.get("request_url"), expected_event_id=betano_event_id
    )
    if (
        transport.get("transport_id") != TRANSPORT_ID
        or transport.get("method") != "GET"
        or transport.get("final_url") != request_url
        or transport.get("http_status") != 200
        or not isinstance(transport.get("content_type"), str)
        or not CONTENT_TYPE_RE.fullmatch(transport["content_type"].strip())
        or transport.get("response_body_storage")
        != "generic_quote_receipt.source_payload_base64"
        or transport.get(
            "response_body_excludes_request_headers_cookies_credentials"
        )
        is not True
    ):
        raise BetanoQuoteAdapterError("transport identity or safe-body boundary changed")
    request_started = _time(
        transport.get("request_started_at_utc"), "request_started_at_utc"
    )
    response_received = _time(
        transport.get("response_received_at_utc"), "response_received_at_utc"
    )
    wall_duration = transport.get("wall_transport_duration_seconds")
    monotonic_duration_ns = transport.get("monotonic_transport_duration_ns")
    if (
        response_received < request_started
        or type(wall_duration) not in (int, float)
        or not math.isfinite(float(wall_duration))
        or float(wall_duration) < 0
        or abs(
            float(wall_duration)
            - (response_received - request_started).total_seconds()
        )
        > 1e-6
        or isinstance(monotonic_duration_ns, bool)
        or not isinstance(monotonic_duration_ns, int)
        or monotonic_duration_ns <= 0
        or abs(monotonic_duration_ns / 1_000_000_000 - float(wall_duration))
        > 0.5
    ):
        raise BetanoQuoteAdapterError("transport clocks or durations changed")
    browser_started_raw = transport.get("browser_request_started_at_utc")
    browser_read_raw = transport.get("browser_response_body_read_at_utc")
    if browser_started_raw is None or browser_read_raw is None:
        raise BetanoQuoteAdapterError("browser timing envelope is incomplete")
    browser_started = _time(browser_started_raw, "browser_request_started_at_utc")
    browser_read = _time(browser_read_raw, "browser_response_body_read_at_utc")
    if (
        browser_read < browser_started
        or (browser_started - request_started).total_seconds() < -1.0
        or (response_received - browser_read).total_seconds() < -1.0
    ):
        raise BetanoQuoteAdapterError("browser timing escaped transport envelope")
    prediction_to_request = (request_started - prediction_time).total_seconds()
    prediction_to_response = (response_received - prediction_time).total_seconds()
    declared_to_request = prediction.get("prediction_to_request_seconds")
    declared_to_response = prediction.get("prediction_to_response_seconds")
    if (
        type(declared_to_request) not in (int, float)
        or type(declared_to_response) not in (int, float)
        or not math.isclose(
            float(declared_to_request), prediction_to_request, rel_tol=0, abs_tol=1e-6
        )
        or not math.isclose(
            float(declared_to_response),
            prediction_to_response,
            rel_tol=0,
            abs_tol=1e-6,
        )
        or prediction_to_request < 0
        or not 0 <= prediction_to_response <= MAX_PREDICTION_TO_RESPONSE_SECONDS
    ):
        raise BetanoQuoteAdapterError("prediction-to-quote timing changed")

    raw_quote = value.get("generic_quote_receipt")
    if not isinstance(raw_quote, Mapping):
        raise BetanoQuoteAdapterError("generic quote receipt is malformed")
    expected_quote_sha256 = _sha(
        value.get("generic_quote_receipt_sha256"),
        "generic_quote_receipt_sha256",
    )
    try:
        quote = generic_quote.validate_quote_receipt(
            raw_quote,
            expected_quote_sha256=expected_quote_sha256,
            expected_capture_protocol_sha256=REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
            expected_settlement_rules_sha256=REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        )
    except generic_quote.QuoteCaptureError as exc:
        raise BetanoQuoteAdapterError(str(exc)) from exc
    raw_body = _decode_exact_base64(
        quote.get("source_payload_base64"),
        "generic_quote_receipt.source_payload_base64",
        maximum=MAX_RESPONSE_BYTES,
    )
    body_sha256 = hashlib.sha256(raw_body).hexdigest()
    response_body_bytes = transport.get("response_body_bytes")
    if (
        quote.get("source") != "betano-br-public-pre-event-html-v1"
        or quote.get("source_url") != request_url
        or quote.get("event_id") != scryglass_event_id
        or quote.get("market_type") != MARKET_TYPE
        or quote.get("settlement_rule_id") != SETTLEMENT_RULE_ID
        or quote.get("extractor_id") != ADAPTER_ID
        or quote.get("extractor_sha256") != source_lock["raw_sha256"]
        or quote.get("source_payload_sha256") != body_sha256
        or extraction.get("source_payload_sha256") != body_sha256
        or transport.get("response_body_sha256") != body_sha256
        or isinstance(response_body_bytes, bool)
        or not isinstance(response_body_bytes, int)
        or response_body_bytes != len(raw_body)
    ):
        raise BetanoQuoteAdapterError("exact response body or generic quote binding changed")
    quote_built = _time(quote.get("captured_at_utc"), "quote.captured_at_utc")
    if (
        captured_at != quote_built
        or quote_built < response_received
        or (quote_built - response_received).total_seconds() > 5.0
    ):
        raise BetanoQuoteAdapterError("generic quote builder timing changed")
    participants = betano_event.get("participants")
    map_number = _integer(market.get("map_number"), "market.map_number", minimum=1)
    replay = extract_map_winner_v1(
        raw_response_body=raw_body,
        response_received_at_utc=response_received,
        expected_betano_event_id=betano_event_id,
        expected_league=league,
        map_number=map_number,
        participant_bindings=participants,
        probability_selection=selection,
        probability_opposing_selection=opposing_selection,
    )
    if replay != extraction or quote.get("prices") != replay["prices"]:
        raise BetanoQuoteAdapterError("Betano source extraction does not replay")
    expected_record_id = (
        f"betano-brazil:{betano_event_id}:map-{map_number}:"
        f"market-{replay['market']['market_id']}:body-{body_sha256[:16]}"
    )
    if quote.get("source_record_id") != expected_record_id:
        raise BetanoQuoteAdapterError("generic quote source record id changed")
    if value.get("qualification") != _expected_qualification():
        raise BetanoQuoteAdapterError("quote qualification was overstated")
    if value.get("actual_map_start_binding") is not None:
        raise BetanoQuoteAdapterError("candidate quote cannot self-bind actual map start")
    if value.get("decision_outputs") != DECISION_OUTPUTS:
        raise BetanoQuoteAdapterError("quote bundle contains a decision output")
    if value.get("authority") != AUTHORITY:
        raise BetanoQuoteAdapterError("quote bundle exceeds its authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise BetanoQuoteAdapterError("quote bundle claim ceiling changed")
    return value


def _candidate_adapter_contract(source_lock: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "adapter_id": ADAPTER_ID,
        "source_locator": SOURCE_LOCATOR,
        "source_sha256": source_lock["raw_sha256"],
        "bookmaker_id": BOOKMAKER_ID,
        "market_type": MARKET_TYPE,
        "settlement_rule_id": SETTLEMENT_RULE_ID,
        "source_host": BETANO_HOST,
        "source_document": "public_pre_event_html_response_body",
        "embedded_state_assignment": INITIAL_STATE_PREFIX,
        "map_winner_type": "TMPW",
        "map_winner_type_id": 3185,
        "supported_map_numbers": [1, 2, 3, 4, 5],
        "event_market_and_selection_suspension_must_be_absent_or_false": True,
        "market_close_at_least_five_seconds_after_response_required": True,
        "exact_two_participant_and_two_selection_bindings_required": True,
        "deterministic_exact_byte_replay_required": True,
    }


def _candidate_transport_contract() -> dict[str, Any]:
    return {
        "transport_id": TRANSPORT_ID,
        "playwright_cli_package": PLAYWRIGHT_CLI_PACKAGE,
        "playwright_cli_version": PLAYWRIGHT_CLI_VERSION,
        "browser_product": "Brave",
        "fresh_ephemeral_profile_required": True,
        "authentication_permitted": False,
        "credential_submission_permitted": False,
        "bet_slip_or_price_click_permitted": False,
        "system_utc_before_and_after_fetch_required": True,
        "monotonic_duration_required": True,
        "exact_http_200_utf8_html_body_required": True,
        "redirect_permitted": False,
        "persisted_response_material": [
            "final_url",
            "http_status",
            "content_type",
            "response_body_bytes",
            "response_body_sha256",
            "response_body_base64",
        ],
        "request_headers_cookies_credentials_persisted": False,
        "prediction_must_precede_request": True,
        "prediction_to_response_seconds_maximum": MAX_PREDICTION_TO_RESPONSE_SECONDS,
        "generic_quote_builder_time_counts_as_transport_time": False,
    }


def build_betano_quote_adapter_candidate_v1(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Build a source-hashed candidate manifest without independent authority."""

    try:
        protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    except (MatchWinnerFutureProtocolRegistryError, OSError, ValueError) as exc:
        raise BetanoQuoteAdapterError(str(exc)) from exc
    observed = _clock_sample(clock, "candidate.locked_at")
    if observed < _time(REGISTERED_PROTOCOL_LOCKED_AT_UTC, "protocol.locked_at"):
        raise BetanoQuoteAdapterError("adapter candidate predates market protocol")
    source_lock = _source_file_lock(root)
    payload: dict[str, Any] = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "result_state": CANDIDATE_RESULT_STATE,
        "locked_at_utc": observed.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_candidate_builder",
            "observed_wall_clock_utc": observed.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "protocol_bindings": {
            "market_protocol_artifact_sha256": protocol["artifact_sha256"],
            "quote_capture_contract_sha256": protocol[
                "quote_capture_contract_sha256"
            ],
            "settlement_contract_sha256": protocol["settlement_contract_sha256"],
            "generic_quote_receipt_schema_version": (
                generic_quote.RECEIPT_SCHEMA_VERSION
            ),
            "generic_price_extraction_schema_version": (
                generic_quote.EXTRACTION_SCHEMA_VERSION
            ),
        },
        "adapter_contract": _candidate_adapter_contract(source_lock),
        "transport_contract": _candidate_transport_contract(),
        "source_lock": source_lock,
        "registration": {
            "independent_reviewer": None,
            "independent_registry_locator": None,
            "independent_registry_sha256": None,
            "independently_registered": False,
        },
        "live_capture": {
            "response_bytes_embedded_in_candidate_manifest": False,
            "quote_receipt_created_by_candidate_manifest": False,
            "first_phase_two_quote_created": False,
        },
        "decision_outputs": dict(DECISION_OUTPUTS),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CANDIDATE_CLAIM_CEILING,
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_betano_quote_adapter_candidate_v1(payload, root=root)


def validate_betano_quote_adapter_candidate_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    expected_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise BetanoQuoteAdapterError("adapter candidate must be an object")
    value = dict(payload)
    _exact_keys(
        value,
        {
            "schema_version",
            "result_state",
            "locked_at_utc",
            "clock_attestation",
            "protocol_bindings",
            "adapter_contract",
            "transport_contract",
            "source_lock",
            "registration",
            "live_capture",
            "decision_outputs",
            "authority",
            "claim_ceiling",
            "artifact_sha256",
        },
        "adapter candidate",
    )
    if (
        value.get("schema_version") != CANDIDATE_SCHEMA_VERSION
        or value.get("result_state") != CANDIDATE_RESULT_STATE
    ):
        raise BetanoQuoteAdapterError("adapter candidate identity changed")
    unsigned = dict(value)
    artifact_sha256 = unsigned.pop("artifact_sha256", None)
    if artifact_sha256 != sha256_canonical_object(unsigned):
        raise BetanoQuoteAdapterError("adapter candidate artifact hash changed")
    if expected_artifact_sha256 is not None and artifact_sha256 != _sha(
        expected_artifact_sha256, "expected_artifact_sha256"
    ):
        raise BetanoQuoteAdapterError("adapter candidate digest mismatch")
    locked_at = _time(value.get("locked_at_utc"), "locked_at_utc")
    if locked_at > datetime.now(timezone.utc):
        raise BetanoQuoteAdapterError("adapter candidate lock is in the future")
    if locked_at < _time(REGISTERED_PROTOCOL_LOCKED_AT_UTC, "protocol.locked_at"):
        raise BetanoQuoteAdapterError("adapter candidate predates market protocol")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_candidate_builder",
        "observed_wall_clock_utc": locked_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise BetanoQuoteAdapterError("adapter candidate clock attestation changed")
    try:
        protocol = validate_registered_match_winner_future_protocol_v1(root=root)
    except (MatchWinnerFutureProtocolRegistryError, OSError, ValueError) as exc:
        raise BetanoQuoteAdapterError(str(exc)) from exc
    expected_protocol = {
        "market_protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "quote_capture_contract_sha256": REGISTERED_QUOTE_CAPTURE_CONTRACT_SHA256,
        "settlement_contract_sha256": REGISTERED_SETTLEMENT_CONTRACT_SHA256,
        "generic_quote_receipt_schema_version": generic_quote.RECEIPT_SCHEMA_VERSION,
        "generic_price_extraction_schema_version": (
            generic_quote.EXTRACTION_SCHEMA_VERSION
        ),
    }
    if (
        value.get("protocol_bindings") != expected_protocol
        or protocol.get("artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256
    ):
        raise BetanoQuoteAdapterError("adapter candidate protocol binding changed")
    source_lock = _source_file_lock(root)
    if (
        value.get("source_lock") != source_lock
        or value.get("adapter_contract")
        != _candidate_adapter_contract(source_lock)
        or value.get("transport_contract") != _candidate_transport_contract()
    ):
        raise BetanoQuoteAdapterError("adapter candidate source contract changed")
    if value.get("registration") != {
        "independent_reviewer": None,
        "independent_registry_locator": None,
        "independent_registry_sha256": None,
        "independently_registered": False,
    }:
        raise BetanoQuoteAdapterError("adapter candidate self-authorized")
    if value.get("live_capture") != {
        "response_bytes_embedded_in_candidate_manifest": False,
        "quote_receipt_created_by_candidate_manifest": False,
        "first_phase_two_quote_created": False,
    }:
        raise BetanoQuoteAdapterError("adapter candidate live-capture state changed")
    if value.get("decision_outputs") != DECISION_OUTPUTS:
        raise BetanoQuoteAdapterError("adapter candidate contains decision output")
    if value.get("authority") != AUTHORITY:
        raise BetanoQuoteAdapterError("adapter candidate exceeds authority")
    if value.get("claim_ceiling") != CANDIDATE_CLAIM_CEILING:
        raise BetanoQuoteAdapterError("adapter candidate claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace existing artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(raw).hexdigest()


def _read_json_file(path: Path, label: str) -> Any:
    if not path.is_file() or path.is_symlink():
        raise BetanoQuoteAdapterError(f"{label} must be a regular non-symlink file")
    raw = path.read_bytes()
    if not raw or len(raw) > MAX_RESPONSE_BYTES:
        raise BetanoQuoteAdapterError(f"{label} is empty or too large")
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                BetanoQuoteAdapterError(f"non-finite JSON in {label}: {token}")
            ),
        )
    except BetanoQuoteAdapterError:
        raise
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise BetanoQuoteAdapterError(f"{label} is not strict UTF-8 JSON") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    candidate_parser = subparsers.add_parser(
        "candidate", description="Build the non-authorizing adapter candidate."
    )
    candidate_parser.add_argument("--root", type=Path, default=ROOT)
    candidate_parser.add_argument("--out", type=Path, default=DEFAULT_CANDIDATE_OUTPUT)
    capture_parser = subparsers.add_parser(
        "capture", description="Capture a non-authorizing public quote receipt."
    )
    capture_parser.add_argument("--root", type=Path, default=ROOT)
    capture_parser.add_argument("--probability-receipt", type=Path, required=True)
    capture_parser.add_argument("--event-url", required=True)
    capture_parser.add_argument("--betano-event-id", required=True)
    capture_parser.add_argument("--map-number", type=int, required=True)
    capture_parser.add_argument("--participant-bindings", type=Path, required=True)
    capture_parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "candidate":
            output = args.out if args.out.is_absolute() else args.root / args.out
            payload = build_betano_quote_adapter_candidate_v1(root=args.root)
        else:
            output = args.out if args.out.is_absolute() else args.root / args.out
            probability = _read_json_file(
                args.probability_receipt, "probability receipt"
            )
            participant_bindings = _read_json_file(
                args.participant_bindings, "participant bindings"
            )
            if not isinstance(probability, Mapping):
                raise BetanoQuoteAdapterError("probability receipt must be an object")
            if not isinstance(participant_bindings, list):
                raise BetanoQuoteAdapterError("participant bindings must be a list")
            with BravePlaywrightPublicDocumentFetcher() as fetcher:
                payload = capture_betano_map_winner_quote_v1(
                    probability_receipt=probability,
                    request_url=args.event_url,
                    betano_event_id=args.betano_event_id,
                    map_number=args.map_number,
                    participant_bindings=participant_bindings,
                    fetcher=fetcher,
                    root=args.root,
                )
        raw_sha256 = write_no_clobber(output, payload)
    except (
        BetanoQuoteAdapterError,
        FileExistsError,
        OSError,
        ValueError,
    ) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "artifact": str(output),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "authorizing": False,
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ADAPTER_ID",
    "BetanoQuoteAdapterError",
    "BravePlaywrightPublicDocumentFetcher",
    "CANDIDATE_SCHEMA_VERSION",
    "DEFAULT_CANDIDATE_OUTPUT",
    "PublicDocumentResponse",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "TRANSPORT_ID",
    "build_betano_quote_adapter_candidate_v1",
    "capture_betano_map_winner_quote_v1",
    "extract_map_winner_v1",
    "parse_initial_state_v1",
    "validate_betano_map_winner_quote_v1",
    "validate_betano_quote_adapter_candidate_v1",
    "write_no_clobber",
]


if __name__ == "__main__":
    raise SystemExit(main())
