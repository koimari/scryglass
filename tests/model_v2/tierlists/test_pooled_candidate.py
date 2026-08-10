"""Tests for the public coach-facing pooled tier fields."""

from __future__ import annotations

from lol_kills.v2.tierlists.pooled_candidate import (
    _build_regional_views,
    _regional_contexts,
)


def test_regional_contexts_use_exact_league_identity() -> None:
    assert _regional_contexts({"league": "LCK"}) == ("LCK",)
    assert _regional_contexts({"league": "MSI", "event_kind": "msi"}) == ("INTERNATIONAL",)
    assert _regional_contexts({"league": "LCKC"}) == ()


def test_regional_view_keeps_patch_wide_strength_order() -> None:
    rows = [
        {
            "champion": "Lower global pick",
            "champion_id": "riot:champion:2",
            "rank": 2,
            "tier_value_pp": 4.0,
        },
        {
            "champion": "Higher global pick",
            "champion_id": "riot:champion:1",
            "rank": 1,
            "tier_value_pp": 10.0,
        },
    ]
    views = _build_regional_views(
        rows=rows,
        scope_id="patch:16.14",
        role="mid",
        regional_counts={
            ("patch:16.14", "LCK", "mid"): {
                "riot:champion:2": 4,
                "riot:champion:1": 1,
            }
        },
        regional_game_ids={("patch:16.14", "LCK"): {"game-1", "game-2"}},
    )

    assert len(views) == 1
    assert views[0]["id"] == "LCK"
    assert views[0]["maps"] == 2
    assert [row["champion_id"] for row in views[0]["rows"]] == [
        "riot:champion:1",
        "riot:champion:2",
    ]
