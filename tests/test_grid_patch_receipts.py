from __future__ import annotations

import hashlib
import json

from lol_kills.etl.grid_patch_receipts import resolve_fixture
from lol_kills.research.mechanics_engine_run import _canonical_hash, _load_grid_patch_receipts


def test_grid_crosswalk_requires_exact_identity_and_excludes_outcome_fields() -> None:
    pregame = {
        "fixture_id": "fixture-1",
        "event_start": "2026-07-15T14:15:00Z",
        "as_of": "2026-07-15T14:14:59Z",
        "blue": {
            "team": "Alpha",
            "picks": ["Aatrox", "Lee Sin", "Azir", "Jhin", "Leona"],
            "players": [{"player": name} for name in ["A", "B", "C", "D", "E"]],
        },
        "red": {
            "team": "Beta",
            "picks": ["Gnar", "Vi", "Orianna", "Aphelios", "Nautilus"],
            "players": [{"player": name} for name in ["F", "G", "H", "I", "J"]],
        },
    }
    candidate = {
        "series_id": "series-1",
        "game_id": "game-1",
        "date": "2026-07-15T14:15:20Z",
        "tournament_id": "tournament-1",
        "patch": "16.13",
        "team_1_name": "Alpha",
        "team_2_name": "Beta",
        "team_1_champions": json.dumps(pregame["blue"]["picks"]),
        "team_2_champions": json.dumps(pregame["red"]["picks"]),
        "team_1_players": json.dumps(["A", "B", "C", "D", "E"]),
        "team_2_players": json.dumps(["F", "G", "H", "I", "J"]),
        "winner_team_id": "leak-me",
        "complete": True,
    }

    receipt = resolve_fixture(fixture_id="fixture-1", pregame=pregame, candidates=[candidate])

    assert receipt["authority_status"] == "confirmed_metadata"
    assert receipt["patch"] == "26.13"
    assert receipt["pregame_authorized"] is False
    assert "winner_team_id" not in receipt
    assert "complete" not in receipt
    assert not {"winner_team_id", "complete", "won"}.intersection(receipt["evidence"])


def test_grid_manifest_loader_marks_post_event_receipts_retrospective(tmp_path) -> None:
    receipt = {
        "fixture_id": "fixture-1",
        "authority_status": "confirmed_metadata",
        "pregame_authorized": False,
        "blockers": ["grid_source_captured_after_cutoff"],
        "evidence": {"public_patch": "26.13"},
    }
    receipt_bytes = (json.dumps(receipt, sort_keys=True) + "\n").encode("utf-8")
    receipt_path = tmp_path / "patch-receipts.jsonl"
    receipt_path.write_bytes(receipt_bytes)
    unsigned = {
        "schema_version": "scryglass:grid-patch-receipts:v1",
        "fixture_count": 1,
        "confirmed_metadata_fixture_count": 1,
        "pregame_authorized_fixture_count": 0,
        "unavailable_fixture_count": 0,
        "outcome_fields_emitted": False,
        "receipt_file": str(receipt_path),
        "receipt_file_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "source_captured_at": "2026-07-28T02:38:11Z",
    }
    manifest_path = tmp_path / "receipt-manifest.json"
    manifest_path.write_text(
        json.dumps({**unsigned, "manifest_sha256": _canonical_hash(unsigned)}),
        encoding="utf-8",
    )

    readiness, index = _load_grid_patch_receipts(manifest_path)

    assert readiness["status"] == "retrospective_only"
    assert readiness["exact_identity_fixture_count"] == 1
    assert readiness["pregame_authorized_fixture_count"] == 0
    assert "grid_patch_source_captured_after_cutoff" in readiness["blockers"]
    assert index["fixture-1"]["pregame_authorized"] is False
