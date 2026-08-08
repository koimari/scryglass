from __future__ import annotations

import hashlib
import json

from lol_kills.etl.leaguepedia_patch_revisions import extract_match_patch
from lol_kills.research.mechanics_engine_run import (
    _canonical_hash,
    _load_leaguepedia_patch_revision_receipts,
    _prediction,
)


def test_extract_match_patch_uses_tab_and_match_ordinal() -> None:
    content = """
{{SetPatch|patch=26.13}}
{{MatchSchedule/Start|tab=Week 1}}
{{MatchSchedule|date=2026-07-01}}
{{MatchSchedule|date=2026-07-02}}
{{SetPatch|patch=26.14}}
{{MatchSchedule/Start|tab=Week 2}}
{{MatchSchedule|date=2026-07-10}}
"""

    result = extract_match_patch(content, tab="Week 2", match_ordinal=1)

    assert result["status"] == "exact"
    assert result["patch"] == "26.14"


def test_blank_historical_setpatch_does_not_fallback_to_retrospective_value() -> None:
    content = """
{{SetPatch|patch=}}
{{MatchSchedule/Start|tab=Week 1}}
{{MatchSchedule|date=2026-07-01}}
"""

    result = extract_match_patch(content, tab="Week 1", match_ordinal=1)

    assert result["status"] == "blocked"
    assert result["patch"] is None


def test_revision_manifest_preserves_partial_strict_coverage(tmp_path) -> None:
    receipt = {
        "fixture_id": "fixture-1",
        "authority_status": "pre_event_revision",
        "pregame_authorized": True,
        "patch": "26.13",
        "client_patch": "16.13",
        "blockers": [],
        "evidence": {"revision_timestamp": "2026-07-01T08:00:00Z"},
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
    receipt_path = tmp_path / "patch-receipts.jsonl"
    receipt_path.write_bytes(receipt_bytes)
    unsigned = {
        "schema_version": "scryglass:leaguepedia-patch-revisions:v1",
        "fixture_count": 1,
        "pre_event_revision_fixture_count": 1,
        "pregame_authorized_fixture_count": 1,
        "unavailable_fixture_count": 0,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "outcome_fields_requested": [],
        "outcome_fields_emitted": False,
    }
    manifest_path = tmp_path / "receipt-manifest.json"
    manifest_path.write_text(
        json.dumps({**unsigned, "manifest_sha256": _canonical_hash(unsigned)}),
        encoding="utf-8",
    )

    readiness, index = _load_leaguepedia_patch_revision_receipts(manifest_path)

    assert readiness["status"] == "pregame_authorized"
    assert readiness["pregame_authorized_fixture_count"] == 1
    assert index["fixture-1"]["patch"] == "26.13"


def test_revision_manifest_fails_closed_on_manifest_outcome_flag(tmp_path) -> None:
    receipt = {
        "fixture_id": "fixture-1",
        "authority_status": "pre_event_revision",
        "pregame_authorized": True,
        "patch": "26.13",
        "client_patch": "16.13",
        "blockers": [],
        "evidence": {"revision_timestamp": "2026-07-01T08:00:00Z"},
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
    receipt_path = tmp_path / "patch-receipts.jsonl"
    receipt_path.write_bytes(receipt_bytes)
    unsigned = {
        "schema_version": "scryglass:leaguepedia-patch-revisions:v1",
        "fixture_count": 1,
        "pre_event_revision_fixture_count": 1,
        "pregame_authorized_fixture_count": 1,
        "unavailable_fixture_count": 0,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "outcome_fields_requested": [],
        "outcome_fields_emitted": True,
    }
    manifest_path = tmp_path / "receipt-manifest.json"
    manifest_path.write_text(
        json.dumps({**unsigned, "manifest_sha256": _canonical_hash(unsigned)}),
        encoding="utf-8",
    )

    readiness, _ = _load_leaguepedia_patch_revision_receipts(manifest_path)

    assert readiness["status"] != "pregame_authorized"
    assert "leaguepedia_patch_revision_outcome_field_emitted" in readiness["blockers"]


def test_prediction_uses_only_pre_event_revision_patch_receipt() -> None:
    row = {
        "pregame_sha256": "a" * 64,
        "pregame": {
            "fixture_id": "fixture-1",
            "as_of": "2026-07-01T09:59:59Z",
        },
    }
    prediction = _prediction(
        row,
        {},
        None,
        {
            "fixture-1": {
                "authority_status": "pre_event_revision",
                "pregame_authorized": True,
                "patch": "26.13",
                "client_patch": "16.13",
                "evidence_hash": "b" * 64,
            }
        },
        "c" * 64,
        {"26.13"},
    )

    assert "patch_identity_missing_from_frozen_row" not in prediction.blockers
    assert "patch_packet_not_bound" not in prediction.blockers
