from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.market import betano_br_quote_adapter_v1 as adapter
from lol_kills.v2.market import event_probability_v1 as probability
from lol_kills.v2.market.match_winner_future_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
)


ROOT = Path(__file__).resolve().parents[3]
BETANO_EVENT_ID = "89777520"
REQUEST_URL = "https://www.betano.bet.br/odds/lyon-shopify-rebellion/89777520/"
RESPONSE_TIME = datetime(2026, 8, 2, 12, 0, 1, 200000, tzinfo=timezone.utc)
PREDICTION_TIME = RESPONSE_TIME - timedelta(seconds=1.2)
REQUEST_TIME = RESPONSE_TIME - timedelta(seconds=0.2)
QUOTE_BUILD_TIME = RESPONSE_TIME + timedelta(seconds=0.1)
PARTICIPANT_BINDINGS = [
    {
        "canonical_selection": "winner:lyon",
        "bookmaker_participant_id": "2069028",
        "bookmaker_name": "LYON",
    },
    {
        "canonical_selection": "winner:shopify-rebellion",
        "bookmaker_participant_id": "1986990",
        "bookmaker_name": "Shopify Rebellion",
    },
]


def event_state(*, close_delta_seconds: float = 3600) -> dict:
    close_ms = int((RESPONSE_TIME + timedelta(seconds=close_delta_seconds)).timestamp() * 1000)
    return {
        "data": {
            "event": {
                "stats": [],
                "sportId": "ESPS",
                "shortName": "LYON - Shopify Rebellion",
                "totalMarketsAvailable": 3,
                "regionName": "League of Legends",
                "regionId": "189377",
                "leagueDescription": "LCS",
                "leagueId": "193761",
                "leagueName": "LCS",
                "id": BETANO_EVENT_ID,
                "name": "LYON - Shopify Rebellion",
                "startTime": close_ms,
                "url": "/odds/lyon-shopify-rebellion/89777520/",
                "participants": [
                    {"name": "LYON", "color": "", "id": "2069028"},
                    {
                        "name": "Shopify Rebellion",
                        "color": "",
                        "id": "1986990",
                    },
                ],
                "markets": [
                    {
                        "id": "2877492977",
                        "uniqueId": "2877492977",
                        "name": "Vencedor do mapa (Mapa 1)",
                        "type": "TMPW",
                        "typeId": 3185,
                        "handicap": 0,
                        "displayOrder": 150,
                        "marketCloseTimeMillis": close_ms,
                        "renderingLayout": 2,
                        "pinKey": "p_TMPW0",
                        "selections": [
                            {
                                "id": "10082690249",
                                "name": "LYON",
                                "price": 1.35,
                                "handicap": 0,
                                "betRef": "10082690249",
                                "renderingLayout": 2,
                                "columnIndex": 0,
                            },
                            {
                                "id": "10082690248",
                                "name": "Shopify Rebellion",
                                "price": 3.0,
                                "handicap": 0,
                                "betRef": "10082690248",
                                "renderingLayout": 2,
                                "columnIndex": 1,
                            },
                        ],
                        "scorerSelections": [],
                        "exactScoreSelections": [],
                    }
                ],
            },
            "markets": [],
        },
        "structureComponents": {},
        "user": {"useOptimizedUserInfo": True, "birthYear": 0},
        "languages": {"available": [], "subPath": ""},
        "companyId": 0,
        "regions": {"available": []},
    }


def response_body(state: dict | None = None) -> bytes:
    state = event_state() if state is None else state
    embedded = json.dumps(
        state,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return (
        "<!DOCTYPE html><html lang='pt-BR'><head></head><body>"
        f"<script>{adapter.INITIAL_STATE_PREFIX}{embedded}</script>"
        "<script>window.__translations__={}</script></body></html>"
    ).encode("utf-8")


def probability_receipt(*, captured_at: datetime = PREDICTION_TIME) -> dict:
    return probability.build_event_probability_receipt(
        event_id="scryglass:lcs:series-1:map-1",
        league="LCS",
        market_type="match_winner",
        selection="winner:lyon",
        opposing_selection="winner:shopify-rebellion",
        model_artifact_sha256="1" * 64,
        market_protocol_artifact_sha256=REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        calibration_artifact_sha256="2" * 64,
        uncertainty_artifact_sha256="3" * 64,
        source_prediction_receipt_sha256="4" * 64,
        source_prediction_registry_sha256="5" * 64,
        generation_code_sha256="6" * 64,
        raw_model_probability=0.6,
        calibration_intercept=0.0,
        calibration_slope=1.0,
        probability_interval=[0.5, 0.7],
        uncertainty_draws_sha256="7" * 64,
        uncertainty_resamples=2000,
        clock=lambda: captured_at,
    )


class ScriptedFetcher:
    transport_id = adapter.TRANSPORT_ID

    def __init__(
        self,
        *,
        body: bytes | None = None,
        status: int = 200,
        final_url: str = REQUEST_URL,
        content_type: str = "text/html; charset=utf-8",
        browser_clocks: bool = True,
    ) -> None:
        self.body = response_body() if body is None else body
        self.status = status
        self.final_url = final_url
        self.content_type = content_type
        self.browser_clocks = browser_clocks

    def __call__(self, request_url: str) -> adapter.PublicDocumentResponse:
        assert request_url == REQUEST_URL
        return adapter.PublicDocumentResponse(
            http_status=self.status,
            final_url=self.final_url,
            content_type=self.content_type,
            response_body=self.body,
            browser_request_started_at_utc=(
                (REQUEST_TIME + timedelta(milliseconds=20)).isoformat()
                if self.browser_clocks
                else None
            ),
            browser_response_body_read_at_utc=(
                (RESPONSE_TIME - timedelta(milliseconds=20)).isoformat()
                if self.browser_clocks
                else None
            ),
        )


def capture(
    *,
    root: Path,
    receipt: dict | None = None,
    fetcher: ScriptedFetcher | None = None,
    clocks: list[datetime] | None = None,
    monotonic: list[int] | None = None,
) -> dict:
    clock_values = iter(clocks or [REQUEST_TIME, RESPONSE_TIME, QUOTE_BUILD_TIME])
    monotonic_values = iter(monotonic or [1_000_000_000, 1_200_000_000])
    return adapter.capture_betano_map_winner_quote_v1(
        probability_receipt=probability_receipt() if receipt is None else receipt,
        request_url=REQUEST_URL,
        betano_event_id=BETANO_EVENT_ID,
        map_number=1,
        participant_bindings=PARTICIPANT_BINDINGS,
        fetcher=ScriptedFetcher() if fetcher is None else fetcher,
        root=root,
        clock=lambda: next(clock_values),
        monotonic_ns=lambda: next(monotonic_values),
    )


def test_exact_initial_state_and_map_winner_extract() -> None:
    raw = response_body()
    state = adapter.parse_initial_state_v1(raw)
    assert state["data"]["event"]["id"] == BETANO_EVENT_ID
    extracted = adapter.extract_map_winner_v1(
        raw_response_body=raw,
        response_received_at_utc=RESPONSE_TIME,
        expected_betano_event_id=BETANO_EVENT_ID,
        expected_league="LCS",
        map_number=1,
        participant_bindings=PARTICIPANT_BINDINGS,
        probability_selection="winner:lyon",
        probability_opposing_selection="winner:shopify-rebellion",
    )
    assert extracted["market"]["market_id"] == "2877492977"
    assert extracted["market"]["status"] == "open"
    assert extracted["prices"] == {
        "winner:lyon": 1.35,
        "winner:shopify-rebellion": 3.0,
    }


def test_initial_state_must_be_unique_strict_json() -> None:
    raw = response_body()
    duplicated = raw.replace(
        b"</body>",
        b'<script>window["initial_state"]={}</script></body>',
    )
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="exactly once"):
        adapter.parse_initial_state_v1(duplicated)
    duplicate_key = (
        b'<html><body><script>window["initial_state"]={"data":{},"data":{}}'
        b"</script></body></html>"
    )
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="duplicate JSON key"):
        adapter.parse_initial_state_v1(duplicate_key)


@pytest.mark.parametrize(
    ("level", "field"),
    [
        ("event", "isSuspended"),
        ("market", "suspended"),
        ("selection", "isSuspended"),
    ],
)
def test_suspension_at_any_source_layer_fails_closed(level: str, field: str) -> None:
    state = event_state()
    event = state["data"]["event"]
    if level == "event":
        target = event
    elif level == "market":
        target = event["markets"][0]
    else:
        target = event["markets"][0]["selections"][0]
    target[field] = True
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="not explicitly active"):
        adapter.extract_map_winner_v1(
            raw_response_body=response_body(state),
            response_received_at_utc=RESPONSE_TIME,
            expected_betano_event_id=BETANO_EVENT_ID,
            expected_league="LCS",
            map_number=1,
            participant_bindings=PARTICIPANT_BINDINGS,
            probability_selection="winner:lyon",
            probability_opposing_selection="winner:shopify-rebellion",
        )


def test_close_time_ambiguity_participant_drift_and_outcome_fail_closed() -> None:
    near_close = event_state(close_delta_seconds=4.999)
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="closes too near"):
        adapter.extract_map_winner_v1(
            raw_response_body=response_body(near_close),
            response_received_at_utc=RESPONSE_TIME,
            expected_betano_event_id=BETANO_EVENT_ID,
            expected_league="LCS",
            map_number=1,
            participant_bindings=PARTICIPANT_BINDINGS,
            probability_selection="winner:lyon",
            probability_opposing_selection="winner:shopify-rebellion",
        )
    drifted = deepcopy(PARTICIPANT_BINDINGS)
    drifted[0]["bookmaker_participant_id"] = "999"
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="source identity"):
        adapter.extract_map_winner_v1(
            raw_response_body=response_body(),
            response_received_at_utc=RESPONSE_TIME,
            expected_betano_event_id=BETANO_EVENT_ID,
            expected_league="LCS",
            map_number=1,
            participant_bindings=drifted,
            probability_selection="winner:lyon",
            probability_opposing_selection="winner:shopify-rebellion",
        )
    outcome = event_state()
    outcome["data"]["event"]["result"] = "LYON"
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="forbidden outcome"):
        adapter.extract_map_winner_v1(
            raw_response_body=response_body(outcome),
            response_received_at_utc=RESPONSE_TIME,
            expected_betano_event_id=BETANO_EVENT_ID,
            expected_league="LCS",
            map_number=1,
            participant_bindings=PARTICIPANT_BINDINGS,
            probability_selection="winner:lyon",
            probability_opposing_selection="winner:shopify-rebellion",
        )


def test_capture_binds_exact_body_transport_prediction_and_generic_receipt(
    historical_capture_root: Path,
) -> None:
    value = capture(root=historical_capture_root)
    checked = adapter.validate_betano_map_winner_quote_v1(
        value, root=historical_capture_root
    )
    assert checked["transport"]["monotonic_transport_duration_ns"] == 200_000_000
    assert checked["transport"]["response_body_bytes"] == len(response_body())
    assert checked["prediction_binding"]["prediction_to_response_seconds"] == 1.2
    assert checked["generic_quote_receipt"]["prices"] == {
        "winner:lyon": 1.35,
        "winner:shopify-rebellion": 3.0,
    }
    assert checked["qualification"]["phase_two_evidence_qualifies"] is False
    assert all(authority is False for authority in checked["authority"].values())


def test_capture_rejects_late_prediction_bad_transport_and_clock_drift(
    historical_capture_root: Path,
) -> None:
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="predates"):
        capture(
            root=historical_capture_root,
            receipt=probability_receipt(
                captured_at=REQUEST_TIME + timedelta(seconds=1)
            ),
        )
    too_old = probability_receipt(captured_at=RESPONSE_TIME - timedelta(seconds=31))
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="window exceeded"):
        capture(root=historical_capture_root, receipt=too_old)
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="redirected"):
        capture(
            root=historical_capture_root,
            fetcher=ScriptedFetcher(final_url="https://www.betano.bet.br/"),
        )
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="content type"):
        capture(
            root=historical_capture_root,
            fetcher=ScriptedFetcher(content_type="application/json"),
        )
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="durations diverge"):
        capture(
            root=historical_capture_root,
            monotonic=[1_000_000_000, 3_000_000_000],
        )


def test_bundle_tampering_is_detected_by_hash_and_replay(
    historical_capture_root: Path,
) -> None:
    value = capture(root=historical_capture_root)
    changed = deepcopy(value)
    changed["generic_quote_receipt"]["prices"]["winner:lyon"] = 1.4
    unsigned = dict(changed)
    unsigned.pop("artifact_sha256")
    changed["artifact_sha256"] = adapter.sha256_json(unsigned)
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="quote receipt"):
        adapter.validate_betano_map_winner_quote_v1(
            changed, root=historical_capture_root
        )

    changed = deepcopy(value)
    raw = base64.b64decode(
        changed["generic_quote_receipt"]["source_payload_base64"], validate=True
    )
    changed["generic_quote_receipt"]["source_payload_base64"] = base64.b64encode(
        raw + b" "
    ).decode("ascii")
    unsigned = dict(changed)
    unsigned.pop("artifact_sha256")
    changed["artifact_sha256"] = adapter.sha256_json(unsigned)
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="quote receipt"):
        adapter.validate_betano_map_winner_quote_v1(
            changed, root=historical_capture_root
        )


def test_candidate_manifest_is_source_locked_and_non_authorizing(
    historical_capture_root: Path,
) -> None:
    locked_at = datetime(2026, 8, 2, 2, 0, tzinfo=timezone.utc)
    value = adapter.build_betano_quote_adapter_candidate_v1(
        root=historical_capture_root, clock=lambda: locked_at
    )
    checked = adapter.validate_betano_quote_adapter_candidate_v1(
        value, root=historical_capture_root
    )
    assert checked["locked_at_utc"] == locked_at.isoformat()
    assert checked["registration"]["independently_registered"] is False
    assert checked["live_capture"]["first_phase_two_quote_created"] is False
    assert all(authority is False for authority in checked["authority"].values())


def test_playwright_result_is_strictly_framed() -> None:
    payload = {
        "schemaVersion": "scryglass-playwright-public-document-v1",
        "browserRequestStartedAtUtc": REQUEST_TIME.isoformat(),
        "browserResponseBodyReadAtUtc": RESPONSE_TIME.isoformat(),
        "httpStatus": 200,
        "finalUrl": REQUEST_URL,
        "contentType": "text/html; charset=utf-8",
        "responseBodyBase64": base64.b64encode(response_body()).decode("ascii"),
    }
    stdout = (
        "### Result\n"
        + json.dumps(payload, separators=(",", ":"))
        + "\n### Ran Playwright code\nignored"
    )
    assert adapter._playwright_result(stdout) == payload
    with pytest.raises(adapter.BetanoQuoteAdapterError, match="framing"):
        adapter._playwright_result(stdout + stdout)


def test_cli_exposes_no_timestamp_login_or_credentials_argument() -> None:
    source = (ROOT / adapter.SOURCE_LOCATOR).read_text(encoding="utf-8")
    assert "--captured-at" not in source
    assert "--request-started" not in source
    assert "--response-received" not in source
    assert "--login" not in source
    assert "--username" not in source
    assert "--password" not in source
