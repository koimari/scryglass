from __future__ import annotations

import json

from lol_kills.research.mechanics_engine_run import (
    _canonical_hash,
    _client_patch_readiness,
    _load_leaguepedia_patch_receipts,
    _load_roster_receipts,
    _prediction,
)


def test_client_matrix_reports_exact_source_blocked_without_fallback(tmp_path) -> None:
    unsigned = {
        "schema_version": "scryglass:cdragon-patch-packet:v1",
        "matrix_kind": "communitydragon_2026_patch_matrix",
        "patches": [
            {
                "patch": "26.13",
                "status": "blocked",
                "exact_patch_source": False,
                "probes": {},
                "row_sha256": "a" * 64,
            }
        ],
    }
    matrix = {**unsigned, "manifest_sha256": _canonical_hash(unsigned)}
    path = tmp_path / "matrix-manifest.json"
    path.write_text(json.dumps(matrix), encoding="utf-8")

    result = _client_patch_readiness(path)

    assert result["status"] == "blocked"
    assert result["patch_count"] == 1
    assert result["exact_source_patch_count"] == 0
    assert "client_patch_source_unavailable" in result["blockers"]


def test_roster_manifest_hashes_receipt_file_and_indexes_fixtures(tmp_path) -> None:
    receipt = {
        "fixture_id": "fixture-1",
        "authority_status": "confirmed",
        "teams": {},
    }
    receipt_path = tmp_path / "lineup-receipts.jsonl"
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    import hashlib

    unsigned = {
        "schema_version": "scryglass:roster-receipts:v1",
        "fixture_count": 1,
        "confirmed_fixture_count": 1,
        "unavailable_fixture_count": 0,
        "team_count": 2,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
    }
    manifest = {**unsigned, "manifest_sha256": _canonical_hash(unsigned)}
    manifest_path = tmp_path / "receipt-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    readiness, index = _load_roster_receipts(manifest_path)

    assert readiness["status"] == "complete"
    assert readiness["confirmed_fixture_count"] == 1
    assert index["fixture-1"]["authority_status"] == "confirmed"


def test_prediction_uses_confirmed_historical_lineups_without_roster_missing_blocker() -> None:
    fixture_id = "fixture-1"
    row = {
        "pregame_sha256": "a" * 64,
        "pregame": {
            "fixture_id": fixture_id,
            "as_of": "2026-07-01T09:59:59Z",
        },
    }
    players = [
        {"role": "top", "player": "Top"},
        {"role": "jungle", "player": "Jungle"},
        {"role": "mid", "player": "Mid"},
        {"role": "bot", "player": "Bot"},
        {"role": "support", "player": "Support"},
    ]
    team = {
        "fixture_id": fixture_id,
        "team": "Test",
        "event_start": "2026-07-01T10:00:00Z",
        "as_of": "2026-07-01T09:59:59Z",
        "players": players,
        "authority_status": "confirmed",
        "blockers": [],
        "evidence_hash": "b" * 64,
    }
    prediction = _prediction(
        row,
        {
            fixture_id: {
                "fixture_id": fixture_id,
                "authority_status": "confirmed",
                "evidence_hash": "c" * 64,
                "teams": {"blue": team, "red": {**team, "team": "Other"}},
            }
        },
        "d" * 64,
    )

    assert "pre_event_roster_receipt_missing" not in prediction.blockers
    assert "pre_event_roster_receipt_unavailable" not in prediction.blockers
    assert "lineup_pair_missing" not in prediction.blockers


def test_leaguepedia_patch_manifest_is_result_blind_and_retrospective(tmp_path) -> None:
    receipt = {
        "fixture_id": "fixture-1",
        "authority_status": "confirmed_metadata",
        "pregame_authorized": False,
        "patch": "26.13",
        "client_patch": "16.13",
        "blockers": ["leaguepedia_source_captured_after_cutoff"],
        "evidence": {"source_kind": "leaguepedia_scoreboardgames_patch"},
    }
    receipt_path = tmp_path / "patch-receipts.jsonl"
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
    receipt_path.write_bytes(receipt_bytes)
    import hashlib

    unsigned = {
        "schema_version": "scryglass:leaguepedia-patch-receipts:v1",
        "fixture_count": 1,
        "confirmed_metadata_fixture_count": 1,
        "pregame_authorized_fixture_count": 0,
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

    readiness, index = _load_leaguepedia_patch_receipts(manifest_path)

    assert readiness["status"] == "retrospective_only"
    assert readiness["exact_patch_fixture_count"] == 1
    assert readiness["pregame_authorized_fixture_count"] == 0
    assert readiness["outcome_fields_emitted"] is False
    assert index["fixture-1"]["patch"] == "26.13"
