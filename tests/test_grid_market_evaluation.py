from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from lol_kills.grid_market_evaluation import (
    EvaluationError,
    _target_rows,
    chronological_series_split,
    evaluate_manifest,
    load_cohort,
)


def _row(index: int, series: str | None = None) -> dict:
    series_id = series or f"series-{index}"
    date = (datetime(2024, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)).isoformat().replace("+00:00", "Z")
    return {
        "series_id": series_id,
        "game_id": f"game-{index}",
        "date": date,
        "league": "LCK",
        "checkpoint": 10,
        "current_kills": float(index % 4),
        "total_dragons_now": float(index % 2),
        "total_barons_now": 0.0,
        "total_inhibitors_now": 0.0,
        "first_blood_now": None,
        "first_tower_now": None,
        "first_inhibitor_now": None,
        "first_dragon_now": None,
        "first_baron_now": None,
        "total_kills": 20.0 + (index % 4),
        "total_dragons": 3.0,
        "total_barons": 1.0,
        "total_inhibitor_destructions": 2.0,
        "first_blood": index % 2,
        "first_tower": (index + 1) % 2,
        "first_inhibitor": index % 2,
        "first_dragon": (index + 1) % 2,
        "first_baron": index % 2,
    }


def test_series_split_keeps_all_checkpoint_rows_together() -> None:
    rows = []
    for index in range(40):
        rows.extend([{**_row(index, f"series-{index}"), "checkpoint": minute} for minute in (10, 15, 20, 25)])
    split = chronological_series_split(rows)
    locations = {
        row["series_id"]: name
        for name, part in split.items()
        for row in part
    }
    assert len(locations) == 40
    for name, part in split.items():
        assert {row["series_id"] for row in part} == {
            series for series, location in locations.items() if location == name
        }


def test_remaining_total_target_uses_only_at_or_before_state() -> None:
    row = _row(0)
    row["current_kills"] = 7.0
    row["total_kills"] = 25.0
    assert _target_rows([row], "total_kills")[0]["total_kills"] == 18.0


def test_manifest_hash_is_required_and_replayed(tmp_path: Path) -> None:
    record = {
        "status": "verified",
        "league": "LCK",
        "patch": "14.1",
        "identity": {"provider_series_id": "s", "provider_game_id": "g"},
        "chronology": {"series_start_time_scheduled": "2024-01-01T00:00:00Z"},
        "outcomes": {"total_kills": 25},
        "labels": {
            "first_blood": 100,
            "first_tower": 100,
            "first_inhibitor": 100,
            "first_dragon": 100,
            "first_baron": 100,
            "total_dragons": 3,
            "total_barons": 1,
            "total_inhibitor_destructions": 2,
        },
        "checkpoints": [
            {
                "minute": 10,
                "values": {
                    "current_kills": 4,
                    "total_dragons": 1,
                    "total_barons": 0,
                    "total_inhibitor_destructions": 0,
                    "first_blood": 100,
                    "first_tower": None,
                    "first_inhibitor": None,
                    "first_dragon": 100,
                    "first_baron": None,
                },
            }
        ],
    }
    manifest = {
        "scope": {"models_trained": False},
        "coverage": {"verified_maps_total": 1},
        "verified_games": [record],
    }
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvaluationError, match="20 complete provider series"):
        evaluate_manifest(path)
    loaded, rows = load_cohort(path)
    assert loaded["manifest_sha256"] == manifest["manifest_sha256"]
    assert len(rows) == 1
