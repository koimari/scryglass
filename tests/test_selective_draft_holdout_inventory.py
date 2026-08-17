from __future__ import annotations

import pytest

from lol_kills.research.selective_draft_holdout_inventory import (
    SelectiveDraftHoldoutInventoryError,
    summarize_holdout_inventory,
)
from lol_kills.research.seal_selective_draft_holdout import SCHEMA_VERSION
from lol_kills.research.selective_draft_probability import canonical_sha256


def _batch(
    *,
    day: int,
    rows: int,
    selected: int,
    league: str,
    candidate: str = "a" * 64,
) -> dict[str, object]:
    game_ids = [f"day-{day}-game-{index}" for index in range(rows)]
    receipt: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "waiting_for_minimum_holdout_inventory",
        "outcome_blind": True,
        "window": {
            "start": f"2026-08-{day:02d}T00:00:00+00:00",
            "end_exclusive": f"2026-08-{day + 1:02d}T00:00:00+00:00",
        },
        "candidate_receipt_sha256": candidate,
        "game_ids": game_ids,
        "game_ids_sha256": canonical_sha256(game_ids),
        "rows": rows,
        "selected_rows": selected,
        "coverage": selected / rows,
        "league_rows": {league: rows},
        "selected_league_rows": {league: selected},
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def test_inventory_stays_closed_until_all_count_gates_pass() -> None:
    report = summarize_holdout_inventory(
        [_batch(day=16, rows=14, selected=13, league="LPL")]
    )

    assert report["status"] == "waiting"
    assert report["outcomes_may_be_joined"] is False
    assert report["selected_rows"] == 13


def test_inventory_opens_only_after_three_leagues_reach_twenty() -> None:
    report = summarize_holdout_inventory(
        [
            _batch(day=16, rows=40, selected=36, league="LPL"),
            _batch(day=17, rows=40, selected=34, league="LEC"),
            _batch(day=18, rows=40, selected=32, league="LCK"),
        ]
    )

    assert report["selected_rows"] == 102
    assert report["coverage"] == 0.85
    assert report["gates"]["passed"] is True
    assert report["outcomes_may_be_joined"] is True


def test_inventory_rejects_duplicate_games() -> None:
    first = _batch(day=16, rows=40, selected=36, league="LPL")
    second = _batch(day=17, rows=40, selected=36, league="LEC")
    second["game_ids"] = list(first["game_ids"])
    second["game_ids_sha256"] = canonical_sha256(second["game_ids"])
    second["receipt_sha256"] = canonical_sha256(
        {key: value for key, value in second.items() if key != "receipt_sha256"}
    )

    with pytest.raises(SelectiveDraftHoldoutInventoryError, match="games overlap"):
        summarize_holdout_inventory([first, second])
