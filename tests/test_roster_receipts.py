from __future__ import annotations

import hashlib
import json

import pytest

from lol_kills.etl.roster_receipts import (
    RosterReceiptError,
    _sha_object,
    _select_revision,
    lineup_receipt,
    load_receipt_manifest,
    parse_active_roster,
)


HTML = """
<table class="wikitable team-members-current">
<tr><th>Player</th><th>Role</th></tr>
<tr><td class="team-members-player"><a>TopPlayer</a></td><td class="team-members-role"><span>Top Laner</span></td></tr>
<tr><td class="team-members-player"><a>JunglePlayer</a></td><td class="team-members-role"><span>Jungler</span></td></tr>
<tr><td class="team-members-player"><a>MidPlayer</a></td><td class="team-members-role"><span>Mid Laner</span></td></tr>
<tr><td class="team-members-player"><a>BotPlayer</a></td><td class="team-members-role"><span>Bot Laner</span></td></tr>
<tr><td class="team-members-player"><a>SupportPlayer</a></td><td class="team-members-role"><span>Support</span></td></tr>
</table>
"""


def test_parser_reads_only_active_team_members_table() -> None:
    assert parse_active_roster(HTML) == (
        {"role": "top", "player": "TopPlayer"},
        {"role": "jungle", "player": "JunglePlayer"},
        {"role": "mid", "player": "MidPlayer"},
        {"role": "bot", "player": "BotPlayer"},
        {"role": "support", "player": "SupportPlayer"},
    )


def test_revision_selection_is_strictly_before_cutoff() -> None:
    history = {
        "revisions": [
            {"revision_id": 2, "revision_timestamp": "2026-07-01T10:00:00Z"},
            {"revision_id": 1, "revision_timestamp": "2026-06-30T10:00:00Z"},
        ]
    }
    selected = _select_revision(history, "2026-07-01T10:00:00Z")
    assert selected == {"revision_id": 1, "revision_timestamp": "2026-06-30T10:00:00Z"}


def test_lineup_receipt_confirms_only_exact_five_role_roster() -> None:
    history = {
        "team_page": "Test Team",
        "resolved_title": "Test Team",
        "manifest_sha256": "a" * 64,
        "revisions": [{"revision_id": 1, "revision_timestamp": "2026-06-30T10:00:00Z"}],
    }
    payload = {
        "revision_payload_sha256": "b" * 64,
        "rendered_payload_sha256": "c" * 64,
        "content_sha256": "d" * 64,
        "rendered_html_sha256": "e" * 64,
        "html": HTML,
    }
    receipt = lineup_receipt(
        fixture_id="fixture",
        team="Test Team",
        event_start="2026-07-01T10:00:00Z",
        as_of="2026-07-01T09:59:59Z",
        history=history,
        payload=payload,
        capture_at="2026-07-31T00:00:00Z",
    )
    assert receipt["authority_status"] == "confirmed"
    assert receipt["blockers"] == []


def _receipt_package(tmp_path, *, inject_outcome: bool = False):
    history = {
        "team_page": "Blue Team",
        "resolved_title": "Blue Team",
        "manifest_sha256": "a" * 64,
        "revisions": [{"revision_id": 1, "revision_timestamp": "2026-06-30T10:00:00Z"}],
    }
    payload = {
        "revision_payload_sha256": "b" * 64,
        "rendered_payload_sha256": "c" * 64,
        "content_sha256": "d" * 64,
        "rendered_html_sha256": "e" * 64,
        "html": HTML,
    }
    values = {
        "fixture_id": "fixture",
        "event_start": "2026-07-01T10:00:00Z",
        "as_of": "2026-07-01T09:59:59Z",
        "history": history,
        "payload": payload,
        "capture_at": "2026-07-31T00:00:00Z",
    }
    blue = lineup_receipt(team="Blue Team", **values)
    red = lineup_receipt(team="Red Team", **values)
    teams = {"blue": blue, "red": red}
    row = {
        "schema_version": "scryglass:roster-receipts:v1",
        "fixture_id": "fixture",
        "event_start": values["event_start"],
        "as_of": values["as_of"],
        "authority_status": "confirmed",
        "blockers": [],
        "teams": teams,
        "evidence_hash": _sha_object(teams),
    }
    if inject_outcome:
        row["winner"] = "Blue Team"
    receipt_path = tmp_path / "lineup-receipts.jsonl"
    receipt_bytes = (json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n").encode()
    receipt_path.write_bytes(receipt_bytes)
    unsigned = {
        "schema_version": "scryglass:roster-receipts:v1",
        "run_dir": str(tmp_path),
        "captured_at": "2026-07-31T00:00:00Z",
        "team_count": 2,
        "fixture_count": 1,
        "confirmed_fixture_count": 1,
        "unavailable_fixture_count": 0,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "teams": ["Blue Team", "Red Team"],
        "claim_ceiling": {
            "pre_event_lineup_authority": True,
            "winner_prediction": False,
            "publication": False,
        },
    }
    manifest = {**unsigned, "manifest_sha256": _sha_object(unsigned)}
    manifest_path = tmp_path / "receipt-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_receipt_manifest_validates_exact_bytes_and_confirmed_lineup(tmp_path) -> None:
    readiness, index = load_receipt_manifest(_receipt_package(tmp_path))

    assert readiness["status"] == "complete"
    assert readiness["confirmed_fixture_count"] == 1
    assert readiness["model_authority"] is False
    assert readiness["betting_authority"] is False
    assert index["fixture"]["authority_status"] == "confirmed"


def test_receipt_manifest_rejects_hash_bound_outcome_field(tmp_path) -> None:
    with pytest.raises(RosterReceiptError, match="contains an outcome field"):
        load_receipt_manifest(_receipt_package(tmp_path, inject_outcome=True))
