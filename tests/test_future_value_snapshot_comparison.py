from __future__ import annotations

import hashlib
import json

import pytest

from lol_kills.research.future_value_snapshot_comparison import (
    SnapshotComparisonError,
    build_snapshot_comparison_report,
)


def _canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    if newline:
        raw += "\n"
    return raw.encode()


def _rank_rows(identity: str, future_value: str) -> list[dict[str, object]]:
    return [
        {
            identity: f"oe:{identity}:a",
            "current_rank": 1,
            "future_rank": 2,
            "rank_delta": -1,
            "current_value": 100.0,
            "future_value": 1.0,
        },
        {
            identity: f"oe:{identity}:b",
            "current_rank": 2,
            "future_rank": 1,
            "rank_delta": 1,
            "current_value": 90.0,
            "future_value": 2.0,
        },
    ]


def _coverage(identity: str, future_value: str, *, current_rows: int, future_rows: int) -> dict[str, object]:
    rows = _rank_rows(identity, future_value)
    ids = sorted(row[identity] for row in rows)
    identity_digest = hashlib.sha256(
        _canonical({"identity": identity, "ids": ids})
    ).hexdigest()
    paired = [
        {
            identity: row[identity],
            "current_rank": row["current_rank"],
            "future_rank": row["future_rank"],
            "rank_delta": row["rank_delta"],
            "current_value": row["current_value"],
            "future_value": row["future_value"],
        }
        for row in rows
    ]
    paired_digest = hashlib.sha256(_canonical(paired)).hexdigest()
    return {
        "future_rows": future_rows,
        "current_rows": current_rows,
        "matched_rows": len(rows),
        "unmatched_rows": future_rows - len(rows),
        "join_rate": len(rows) / future_rows,
        "status": "partial",
        "rank_universe": "common_verified_finite_ids",
        "eligibility_filter": "verified_nonempty_id_and_finite_value",
        "common_universe_size": len(rows),
        "common_identity_sha256": identity_digest,
        "identity_sha256": identity_digest,
        "current_value_field": "mu_effective",
        "future_value_field": future_value,
        "rank_direction": "descending_value_rank_1_highest",
        "paired_row_digest_sha256": paired_digest,
        "paired_row_digest": paired_digest,
        "finite_current_rows": current_rows,
        "finite_future_rows": future_rows - 1,
        "full_snapshot_ranks": {
            "status": "incomparable",
            "reason": "full snapshot ranks use separate universes",
            "current_universe_size": current_rows,
            "future_universe_size": future_rows - 1,
            "current_value_field": "mu_effective",
            "future_value_field": future_value,
            "rank_direction": "descending_value_rank_1_highest",
        },
    }


def _fixtures() -> tuple[dict[str, object], dict[str, object], dict[str, object], dict[str, object]]:
    source_hash = "a" * 64
    current: dict[str, object] = {
        "schema_version": "scryglass:current-rating-snapshot-receipt:v1",
        "source_receipt_sha256": source_hash,
        "source_identity_sha256": "b" * 64,
        "source_as_of": "2026-08-20T00:00:00Z",
        "source_game_count": 10,
        "snapshots": {
            "player": {"value_column": "mu_effective", "verified_rows": 3},
            "team": {"value_column": "mu_effective", "verified_rows": 2},
        },
    }
    current["receipt_sha256"] = hashlib.sha256(_canonical(current, newline=True)).hexdigest()
    future: dict[str, object] = {
        "schema_version": "scryglass:future-value-snapshot-receipt:v1",
        "status": "research_only_partial",
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
        },
        "source": {
            "source_receipt_sha256": source_hash,
            "source_identity_sha256": "b" * 64,
            "source_as_of": "2026-08-20T00:00:00Z",
            "source_game_count": 10,
        },
        "model": {"receipt_sha256": "c" * 64},
        "blockers": [
            "current_rating_player_team_identity_missing_for_rank_diffs",
            "current_player_team_rating_comparison_missing",
        ],
    }
    future["receipt_sha256"] = hashlib.sha256(_canonical(future)).hexdigest()
    player_rows = _rank_rows("player_id", "future_player_value_logit")
    team_rows = _rank_rows("team_id", "future_team_value_logit")
    player = {
        "source_receipt_sha256": source_hash,
        "rows": player_rows,
        "rank_coverage": _coverage("player_id", "future_player_value_logit", current_rows=3, future_rows=3),
    }
    team = {
        "source_receipt_sha256": source_hash,
        "rows": team_rows,
        "rank_coverage": _coverage("team_id", "future_team_value_logit", current_rows=2, future_rows=3),
    }
    return current, future, player, team


def test_snapshot_comparison_binds_common_ids_and_inherited_blocker() -> None:
    current, future, player, team = _fixtures()
    report = build_snapshot_comparison_report(
        current_receipt=current,
        future_receipt=future,
        player_rank_diff_artifact=player,
        team_rank_diff_artifact=team,
        current_receipt_file_sha256="d" * 64,
        future_receipt_file_sha256="e" * 64,
        player_rank_diff_file_sha256="f" * 64,
        team_rank_diff_file_sha256="0" * 64,
        expected_source_receipt_sha256="a" * 64,
    )
    assert report["status"] == "research_only_partial"
    assert report["independent_join"] == {
        "status": "verified",
        "player_rows": 2,
        "team_rows": 2,
        "current_value_fields": {"player": "mu_effective", "team": "mu_effective"},
        "future_value_fields": {
            "player": "future_player_value_logit",
            "team": "future_team_value_logit",
        },
    }
    assert report["blocker_context"]["identity_blocker_is_join_failure"] is False
    assert report["full_snapshot_rank_status"] == "incomparable"
    assert report["snapshot_comparisons"]["player"]["join_rate"] == pytest.approx(2 / 3)
    assert len(report["snapshot_comparisons"]["player"]["common_ids"]) == 2
    assert report["receipts"]["current"]["receipt_sha256"] == current["receipt_sha256"]
    assert report["receipts"]["future"]["receipt_sha256"] == future["receipt_sha256"]
    assert len(report["report_sha256"]) == 64


def test_snapshot_comparison_rejects_changed_paired_digest() -> None:
    current, future, player, team = _fixtures()
    player["rank_coverage"]["paired_row_digest_sha256"] = "1" * 64
    with pytest.raises(SnapshotComparisonError, match="player_id rank coverage changed"):
        build_snapshot_comparison_report(
            current_receipt=current,
            future_receipt=future,
            player_rank_diff_artifact=player,
            team_rank_diff_artifact=team,
        )


def test_snapshot_comparison_rejects_source_mismatch() -> None:
    current, future, player, team = _fixtures()
    player["source_receipt_sha256"] = "2" * 64
    with pytest.raises(SnapshotComparisonError, match="player_id rank artifact source receipt changed"):
        build_snapshot_comparison_report(
            current_receipt=current,
            future_receipt=future,
            player_rank_diff_artifact=player,
            team_rank_diff_artifact=team,
        )
