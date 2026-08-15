from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "data/lol/v2/evaluation/r9e-16.16-ingest-receipt.json"
EVIDENCE = ROOT / "data/lol/v2/evaluation/r9e-16.16-crosswalk-evidence.json"


def test_16_16_ingest_receipt_is_hash_bound_and_fail_closed() -> None:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()

    assert claimed == actual
    assert payload["patch_identity"] == {
        "accepted_public_patch": "26.16",
        "accepted_source_token": "16.16",
        "prior_public_patch": "26.15",
        "prior_source_token_preserved": "16.15",
        "source_game_count": 7,
        "source_latest_utc": "2026-08-15T15:58:57Z",
        "source_patch_row_count": 84,
    }
    assert payload["series_crosswalk"]["mapped_rows"] == 7
    assert payload["series_crosswalk"]["mapped_series"] == 3
    assert payload["series_crosswalk"]["issues"] == 0
    assert payload["authority"]["private_r9e_promotion"] is False
    assert payload["authority"]["public_draft"] is False
    assert payload["authority"]["public_probability"] is False


def test_16_16_crosswalk_evidence_is_durable_and_hash_bound() -> None:
    payload = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")
    actual = hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()

    assert claimed == actual
    assert payload["source_patch_token"] == "16.16"
    assert payload["public_patch"] == "26.16"
    assert payload["counts"] == {
        "issues": 0,
        "mapped_rows": 7,
        "mapped_series": 3,
        "oe_rows": 7,
        "schedule_rows": 4,
        "scoreboard_rows": 9,
    }
    assert len(payload["rows"]) == 7
    assert {row["oe_patch_token"] for row in payload["rows"]} == {"16.16"}
    assert {row["series_patch"] for row in payload["rows"]} == {"26.16"}
    assert payload["public_draft_authorized"] is False
    assert payload["public_probability_authorized"] is False
