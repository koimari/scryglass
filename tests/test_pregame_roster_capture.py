from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lol_kills import pregame_roster_capture as roster
from tools.live_fair_odds import model


NOW = datetime(2026, 8, 1, 20, 0, tzinfo=timezone.utc)
EVENT_START = "2026-08-01T20:30:00+00:00"
PROTOCOL_SHA = "a" * 64
RECEIPT_LOCATOR = (
    "data/lol/private_pregame_rosters/receipts/event-1.json"
)
REGISTRY_LOCATOR = "data/lol/private_pregame_rosters/registry.json"


def teams() -> list[dict]:
    result = []
    for side, organization in (("blue", "Alpha"), ("red", "Bravo")):
        result.append(
            {
                "side": side,
                "organization_id": f"org-{organization.lower()}",
                "organization_name": organization,
                "roster_id": f"roster-{organization.lower()}-event-1",
                "players": [
                    {
                        "role": role,
                        "player_id": f"{organization.lower()}-{role}",
                        "display_name": f"{organization} {role}",
                    }
                    for role in roster.ROLES
                ],
            }
        )
    return result


def receipt(**overrides) -> dict:
    values = {
        "raw_source_payload": b'{"provider":"captured-roster","event":1}',
        "source": "provider-scheduled-series",
        "source_url": "https://example.invalid/series/event-1",
        "source_record_id": "provider:series:event-1",
        "source_updated_at": "2026-08-01T19:50:00+00:00",
        "available_at": "2026-08-01T19:50:00+00:00",
        "captured_at": "2026-08-01T19:55:00+00:00",
        "event_id": "event-1-map-1",
        "event_start": EVENT_START,
        "league": "LCS",
        "teams": teams(),
        "capture_protocol_sha256": PROTOCOL_SHA,
    }
    values.update(overrides)
    return roster.build_pregame_roster_receipt(**values)


def registry(candidate: dict | None = None) -> dict:
    return roster.build_pregame_roster_registry(
        receipts=[(RECEIPT_LOCATOR, candidate or receipt())],
        registry_id="roster-review-1",
        independent_reviewer_id="reviewer-1",
        issued_at=(NOW - timedelta(seconds=5)).isoformat(),
        capture_protocol_sha256=PROTOCOL_SHA,
    )


def write_json(root: Path, locator: str, value: dict) -> None:
    path = root / locator
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_receipt_commits_raw_bytes_and_exact_ordered_ten() -> None:
    candidate = receipt()
    checked = roster.validate_pregame_roster_receipt(candidate)
    assert checked["receipt_sha256"] == roster.sha256_json(candidate)
    assert [team["side"] for team in checked["teams"]] == ["blue", "red"]
    assert [player["role"] for player in checked["teams"][0]["players"]] == list(
        roster.ROLES
    )
    assert len(
        {
            player["player_id"]
            for team in checked["teams"]
            for player in team["players"]
        }
    ) == 10


def test_late_or_unreviewed_capture_is_rejected() -> None:
    with pytest.raises(roster.PregameRosterError, match="before event_start"):
        receipt(captured_at=EVENT_START)
    with pytest.raises(roster.PregameRosterError, match="rights"):
        receipt(rights_status="unknown")


def test_missing_role_and_cross_team_duplicate_are_rejected() -> None:
    missing = teams()
    missing[0]["players"] = missing[0]["players"][:-1]
    with pytest.raises(roster.PregameRosterError, match="exactly five"):
        receipt(teams=missing)
    duplicate = teams()
    duplicate[1]["players"][0]["player_id"] = duplicate[0]["players"][0][
        "player_id"
    ]
    with pytest.raises(roster.PregameRosterError, match="repeat a player"):
        receipt(teams=duplicate)


def test_registry_cannot_register_itself_without_external_digest() -> None:
    value = registry()
    with pytest.raises(roster.RegisteredPregameRosterUnavailable) as error:
        roster.validate_pregame_roster_registry(
            value, expected_registry_sha256=None
        )
    assert error.value.code == "roster_registry_not_registered"


def test_registered_loader_replays_all_event_and_team_bindings(tmp_path: Path) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_json(tmp_path, RECEIPT_LOCATOR, candidate)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    loaded = roster.load_registered_pregame_roster(
        registry_locator=REGISTRY_LOCATOR,
        expected_registry_sha256=roster.sha256_json(value),
        event_id="event-1-map-1",
        event_start=EVENT_START,
        league="LCS",
        blue_organization_name="Alpha",
        red_organization_name="Bravo",
        as_of=NOW,
        root=tmp_path,
    )
    assert loaded["status"] == "registered"
    assert loaded["receipt_sha256"] == roster.sha256_json(candidate)
    assert loaded["roster"]["teams"][0]["players"][0]["player_id"] == "alpha-top"


def test_registered_side_swap_cannot_bind_to_requested_match(tmp_path: Path) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_json(tmp_path, RECEIPT_LOCATOR, candidate)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    with pytest.raises(roster.RegisteredPregameRosterUnavailable) as error:
        roster.load_registered_pregame_roster(
            registry_locator=REGISTRY_LOCATOR,
            expected_registry_sha256=roster.sha256_json(value),
            event_id="event-1-map-1",
            event_start=EVENT_START,
            league="LCS",
            blue_organization_name="Bravo",
            red_organization_name="Alpha",
            as_of=NOW,
            root=tmp_path,
        )
    assert error.value.code == "roster_blue_organization_name_binding_mismatch"


def test_post_registration_receipt_tamper_is_rejected(tmp_path: Path) -> None:
    candidate = receipt()
    value = registry(candidate)
    candidate["teams"][0]["players"][0]["player_id"] = "attacker"
    write_json(tmp_path, RECEIPT_LOCATOR, candidate)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    with pytest.raises(roster.PregameRosterError, match="receipt digest mismatch"):
        roster.load_registered_pregame_roster(
            registry_locator=REGISTRY_LOCATOR,
            expected_registry_sha256=roster.sha256_json(value),
            event_id="event-1-map-1",
            event_start=EVENT_START,
            league="LCS",
            blue_organization_name="Alpha",
            red_organization_name="Bravo",
            as_of=NOW,
            root=tmp_path,
        )


def test_registry_path_escape_is_rejected() -> None:
    with pytest.raises(roster.PregameRosterError, match="outside"):
        roster.build_pregame_roster_registry(
            receipts=[("../../roster.json", receipt())],
            registry_id="roster-review-1",
            independent_reviewer_id="reviewer-1",
            issued_at=NOW.isoformat(),
            capture_protocol_sha256=PROTOCOL_SHA,
        )


def test_future_registry_is_not_usable(tmp_path: Path) -> None:
    candidate = receipt()
    value = roster.build_pregame_roster_registry(
        receipts=[(RECEIPT_LOCATOR, candidate)],
        registry_id="roster-review-future",
        independent_reviewer_id="reviewer-1",
        issued_at=(NOW + timedelta(seconds=1)).isoformat(),
        capture_protocol_sha256=PROTOCOL_SHA,
    )
    write_json(tmp_path, RECEIPT_LOCATOR, candidate)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    with pytest.raises(roster.RegisteredPregameRosterUnavailable) as error:
        roster.load_registered_pregame_roster(
            registry_locator=REGISTRY_LOCATOR,
            expected_registry_sha256=roster.sha256_json(value),
            event_id="event-1-map-1",
            event_start=EVENT_START,
            league="LCS",
            blue_organization_name="Alpha",
            red_organization_name="Bravo",
            as_of=NOW,
            root=tmp_path,
        )
    assert error.value.code == "roster_registry_from_future"


def test_private_worksheet_loads_roster_only_through_pinned_registry(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = receipt()
    value = registry(candidate)
    write_json(tmp_path, RECEIPT_LOCATOR, candidate)
    write_json(tmp_path, REGISTRY_LOCATOR, value)
    monkeypatch.setattr(model, "ROOT", tmp_path)
    monkeypatch.setenv(model.ROSTER_REGISTRY_SHA_ENV, roster.sha256_json(value))
    loaded = model._registered_pregame_roster(
        event_id="event-1-map-1",
        event_start="2026-08-01T20:30:00Z",
        league="LCS",
        blue_team="Alpha",
        red_team="Bravo",
        as_of=NOW,
    )
    assert loaded["status"] == "registered"
    assert loaded["receipt_sha256"] == roster.sha256_json(candidate)
    assert loaded["blockers"] == []


def test_private_worksheet_reports_missing_roster_pin(monkeypatch) -> None:
    monkeypatch.delenv(model.ROSTER_REGISTRY_SHA_ENV, raising=False)
    unavailable = model._registered_pregame_roster(
        event_id="event-1-map-1",
        event_start=EVENT_START,
        league="LCS",
        blue_team="Alpha",
        red_team="Bravo",
        as_of=NOW,
    )
    assert unavailable["status"] == "unavailable"
    assert unavailable["blockers"] == ["roster_registry_not_registered"]
