from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.etl import grid_series_events
from lol_kills.v2.data.common import ROLES
from lol_kills.v2.draft.terminal import grid_future_source_v1 as source


SERIES_ID = "3000001"
GAME_ID = "grid-game-1"
BLUE_GRID_ID = "grid-blue"
RED_GRID_ID = "grid-red"
BASE_TIME = datetime(2026, 8, 4, 20, 0, tzinfo=timezone.utc)


def _json_raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _draft_transaction(slot: int) -> dict:
    kind = "ban" if slot <= 10 else "pick"
    if slot <= 5 or 11 <= slot <= 15:
        side, team_id, team_name = "blue", BLUE_GRID_ID, "Blue Grid"
    else:
        side, team_id, team_name = "red", RED_GRID_ID, "Red Grid"
    character_id = f"character-{slot}"
    champion_name = f"Champion{slot}"
    event_type = (
        "team-picked-character" if kind == "pick" else "team-banned-character"
    )
    return {
        "id": f"transaction-{slot}",
        "seriesId": SERIES_ID,
        "sequenceNumber": slot,
        "occurredAt": (BASE_TIME + timedelta(seconds=slot)).isoformat(),
        "events": [
            {
                "id": f"event-{slot}",
                "type": event_type,
                "actor": {
                    "type": "team",
                    "id": team_id,
                    "state": {
                        "id": team_id,
                        "name": team_name,
                        "side": side,
                        "game": {"id": GAME_ID},
                    },
                },
                "target": {
                    "type": "character",
                    "id": character_id,
                    "state": {
                        "id": character_id,
                        "type": "character",
                        "name": champion_name,
                    },
                },
                "seriesStateDelta": {
                    "id": SERIES_ID,
                    "games": [
                        {
                            "id": GAME_ID,
                            "draftActions": [
                                {
                                    "id": f"action-{slot}",
                                    "sequenceNumber": slot,
                                    "type": kind,
                                    "drafter": {"id": team_id, "type": "team"},
                                    "draftable": {
                                        "id": character_id,
                                        "type": "character",
                                        "name": champion_name,
                                    },
                                }
                            ],
                        }
                    ],
                },
            }
        ],
    }


def _start_transaction(*, include_full_state_fields: bool = False) -> dict:
    actual_start = BASE_TIME + timedelta(seconds=50)
    event = {
        "id": "event-start",
        "type": "series-started-game",
        "actor": {"type": "series", "id": SERIES_ID},
        "target": {
            "type": "game",
            "id": GAME_ID,
            "stateDelta": {
                "id": GAME_ID,
                "started": True,
                "startedAt": actual_start.isoformat(),
            },
        },
        "seriesStateDelta": {
            "id": SERIES_ID,
            "games": [
                {
                    "id": GAME_ID,
                    "started": True,
                    "startedAt": actual_start.isoformat(),
                }
            ],
        },
    }
    if include_full_state_fields:
        event["seriesState"] = {
            "teams": [{"id": BLUE_GRID_ID, "won": False}],
            "winner": None,
        }
    return {
        "id": "transaction-start",
        "seriesId": SERIES_ID,
        "sequenceNumber": 21,
        "occurredAt": (BASE_TIME + timedelta(seconds=51)).isoformat(),
        "events": [event],
    }


def _envelope(transaction: dict, received: datetime) -> dict:
    return grid_series_events.build_received_transaction_envelope(
        _json_raw(transaction),
        series_id=SERIES_ID,
        clock=lambda: received,
    )


def _receipt_log(*, include_start: bool = False) -> bytes:
    envelopes = [
        _envelope(
            _draft_transaction(slot),
            BASE_TIME + timedelta(seconds=slot, milliseconds=100),
        )
        for slot in range(1, 21)
    ]
    if include_start:
        envelopes.append(
            _envelope(
                _start_transaction(include_full_state_fields=True),
                BASE_TIME + timedelta(seconds=52),
            )
        )
    return b"\n".join(_json_raw(envelope) for envelope in envelopes) + b"\n"


def _context() -> dict:
    picks = [
        ("blue", f"Champion{slot}") for slot in range(11, 16)
    ] + [("red", f"Champion{slot}") for slot in range(16, 21)]
    assignments = [
        {"side": side, "role": role, "champion_name": champion}
        for (side, champion), role in zip(picks, (*ROLES, *ROLES))
    ]
    return {
        "schema_version": source.CONTEXT_SCHEMA_VERSION,
        "event_id": "fixture-1",
        "series_id": SERIES_ID,
        "game_number": 1,
        "league": "LCS",
        "patch": "26.15",
        "grid_game_id": GAME_ID,
        "teams": [
            {
                "side": "blue",
                "grid_team_id": BLUE_GRID_ID,
                "grid_team_name": "Blue Grid",
                "organization_id": "org-blue",
                "organization_name": "Blue Org",
            },
            {
                "side": "red",
                "grid_team_id": RED_GRID_ID,
                "grid_team_name": "Red Grid",
                "organization_id": "org-red",
                "organization_name": "Red Org",
            },
        ],
        "provider_fixture_attestation": {
            "source_id": "grid-central-data-private-v1",
            "source_url": "https://api.grid.gg/central-data/graphql",
            "source_record_id": SERIES_ID,
            "rights_status": "reviewed",
            "series_identity_verified": True,
            "game_identity_verified": True,
            "team_crosswalk_verified": True,
        },
        "role_assignment_attestation": {
            "source_id": "reviewed-live-broadcast-test",
            "source_url": "https://example.test/live-draft",
            "source_record_id": f"{SERIES_ID}:{GAME_ID}:roles",
            "rights_status": "reviewed",
            "observation_method": "reviewed_live_broadcast",
            "observed_before_map_start": True,
        },
        "role_assignments": assignments,
    }


@pytest.fixture(autouse=True)
def _registered_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        source,
        "_registered_readiness",
        lambda root: {
            "locator": "test-grid-readiness.json",
            "raw_sha256": "a" * 64,
            "artifact_sha256": "b" * 64,
            "registry_source": {
                "locator": "test-registry.py",
                "bytes": 1,
                "raw_sha256": "c" * 64,
            },
        },
    )


def test_received_envelope_binds_exact_bytes_and_never_persists_key() -> None:
    raw = _json_raw(_draft_transaction(1))
    envelope = grid_series_events.build_received_transaction_envelope(
        raw,
        series_id=SERIES_ID,
        clock=lambda: BASE_TIME,
    )
    checked, decoded, transaction = (
        grid_series_events.validate_received_transaction_envelope(envelope)
    )
    assert decoded == raw
    assert checked["message"]["raw_sha256"] == hashlib.sha256(raw).hexdigest()
    assert transaction["sequenceNumber"] == 1
    serialized = json.dumps(envelope).lower()
    assert "?key=" not in serialized
    assert "secret" not in serialized


def test_transport_one_shot_detectors_are_exact_to_game_and_event() -> None:
    terminal = _draft_transaction(20)
    start = _start_transaction()
    assert grid_series_events.transaction_has_terminal_draft_for_game(
        terminal, GAME_ID
    )
    assert not grid_series_events.transaction_has_terminal_draft_for_game(
        terminal, "other-game"
    )
    assert not grid_series_events.transaction_has_map_start_for_game(
        terminal, GAME_ID
    )
    assert grid_series_events.transaction_has_map_start_for_game(start, GAME_ID)
    assert not grid_series_events.transaction_has_map_start_for_game(
        start, "other-game"
    )


def test_terminal_adapter_requires_complete_grid_actions_and_reviewed_roles() -> None:
    prepared = source.prepare_terminal_draft_inputs(
        receipt_log_raw=_receipt_log(),
        context_raw=_json_raw(_context()),
        root=Path("."),
        clock=lambda: BASE_TIME + timedelta(seconds=30),
    )
    metadata = json.loads(prepared.metadata_raw)
    payload = json.loads(prepared.source_payload_raw)
    assert [action["slot"] for action in metadata["actions"]] == list(range(1, 21))
    assert len(metadata["final_assignments"]) == 10
    assert set(metadata["blue"]) == set(ROLES)
    assert set(metadata["red"]) == set(ROLES)
    assert payload["validation"]["target_map_start_not_received"] is True
    assert payload["validation"]["raw_grid_messages_embedded_in_source_payload"] is False
    assert "raw_base64" not in prepared.source_payload_raw.decode()


def test_terminal_adapter_refuses_role_guess_or_already_started_map() -> None:
    context = _context()
    context["role_assignments"][0]["champion_name"] = "NotPicked"
    with pytest.raises(source.GridFutureSourceError, match="do not exactly match"):
        source.prepare_terminal_draft_inputs(
            receipt_log_raw=_receipt_log(),
            context_raw=_json_raw(context),
            root=Path("."),
            clock=lambda: BASE_TIME + timedelta(seconds=30),
        )
    with pytest.raises(source.GridFutureSourceError, match="already received"):
        source.prepare_terminal_draft_inputs(
            receipt_log_raw=_receipt_log(include_start=True),
            context_raw=_json_raw(_context()),
            root=Path("."),
            clock=lambda: BASE_TIME + timedelta(seconds=60),
        )


def test_terminal_adapter_rejects_missing_or_reordered_receipt() -> None:
    lines = _receipt_log().splitlines()
    missing = b"\n".join(lines[:9] + lines[10:]) + b"\n"
    with pytest.raises(source.GridFutureSourceError, match="slots 1 through 20"):
        source.prepare_terminal_draft_inputs(
            receipt_log_raw=missing,
            context_raw=_json_raw(_context()),
            root=Path("."),
            clock=lambda: BASE_TIME + timedelta(seconds=30),
        )
    reordered = b"\n".join([lines[1], lines[0], *lines[2:]]) + b"\n"
    with pytest.raises(source.GridFutureSourceError, match="not strictly increasing"):
        source.prepare_terminal_draft_inputs(
            receipt_log_raw=reordered,
            context_raw=_json_raw(_context()),
            root=Path("."),
            clock=lambda: BASE_TIME + timedelta(seconds=30),
        )


def test_map_start_adapter_sanitizes_full_state_and_binds_receive_time() -> None:
    prepared = source.prepare_map_start_inputs(
        receipt_log_raw=_receipt_log(include_start=True),
        context_raw=_json_raw(_context()),
        root=Path("."),
    )
    metadata = json.loads(prepared.metadata_raw)
    payload = json.loads(prepared.source_payload_raw)
    assert metadata["actual_map_start_utc"] == (
        BASE_TIME + timedelta(seconds=50)
    ).isoformat()
    assert metadata["source"]["available_at_utc"] == (
        BASE_TIME + timedelta(seconds=52)
    ).isoformat()
    serialized = json.dumps(payload).lower()
    assert '"winner"' not in serialized
    assert '"won"' not in serialized
    assert "raw_base64" not in serialized
    assert payload["validation"]["provider_start_fields_reconcile"] is True


def test_map_start_adapter_requires_a_log_cut_at_start() -> None:
    later = {
        "id": "transaction-after-start",
        "seriesId": SERIES_ID,
        "sequenceNumber": 22,
        "occurredAt": (BASE_TIME + timedelta(seconds=53)).isoformat(),
        "events": [{"id": "clock-event", "type": "game-clock-started"}],
    }
    raw = _receipt_log(include_start=True) + _json_raw(
        _envelope(later, BASE_TIME + timedelta(seconds=54))
    ) + b"\n"
    with pytest.raises(source.GridFutureSourceError, match="must end"):
        source.prepare_map_start_inputs(
            receipt_log_raw=raw,
            context_raw=_json_raw(_context()),
            root=Path("."),
        )


def test_context_rejects_credential_like_source_urls() -> None:
    context = _context()
    context["role_assignment_attestation"]["source_url"] = (
        "https://example.test/live?token=secret"
    )
    with pytest.raises(source.GridFutureSourceError, match="credential-like"):
        source.validate_capture_context(_json_raw(context))


def test_grid_adapter_cli_exposes_no_user_timestamp_argument() -> None:
    text = Path(source.__file__).read_text()
    assert "--captured-at" not in text
    assert "--received-at" not in text
    assert "--validated-at" not in text
