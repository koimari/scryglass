"""Scope filtering and played-only membership tests for L9 tier lists."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.v2.tierlists.appearances import (
    AppearanceRow,
    AppearanceScope,
    AppearanceTable,
    international_scope,
    league_scope,
)
from lol_kills.v2.tierlists.model import TierListError

ROWS = [
    # LEC tier1, patch 16.14
    {"map_id": "m1", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Aatrox", "event_end": "2026-07-20T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m1", "league": "LEC", "patch_id": "16.14", "role": "jng", "champion_name": "Sejuani", "event_end": "2026-07-20T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m1", "league": "LEC", "patch_id": "16.14", "role": "mid", "champion_name": "Ahri", "event_end": "2026-07-20T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m1", "league": "LEC", "patch_id": "16.14", "role": "bot", "champion_name": "Jinx", "event_end": "2026-07-20T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m1", "league": "LEC", "patch_id": "16.14", "role": "sup", "champion_name": "Thresh", "event_end": "2026-07-20T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m2", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Gnar", "event_end": "2026-07-21T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    {"map_id": "m2", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Aatrox", "event_end": "2026-07-21T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    # LCS tier1 same patch
    {"map_id": "m3", "league": "LCS", "patch_id": "16.14", "role": "top", "champion_name": "Aatrox", "event_end": "2026-07-22T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    # LEC tier2 (development circuit) same patch
    {"map_id": "m4", "league": "LEC", "patch_id": "16.14", "role": "top", "champion_name": "Renekton", "event_end": "2026-07-23T10:00:00Z", "competition_tier": "tier2", "event_kind": None},
    # MSI and EWC stay separate
    {"map_id": "m5", "league": "MSI", "patch_id": "16.13", "role": "top", "champion_name": "Aatrox", "event_end": "2026-07-10T10:00:00Z", "competition_tier": "international", "event_kind": "msi"},
    {"map_id": "m6", "league": "EWC", "patch_id": "16.13", "role": "top", "champion_name": "Gnar", "event_end": "2026-07-11T10:00:00Z", "competition_tier": "international", "event_kind": "ewc"},
    # older patch
    {"map_id": "m7", "league": "LEC", "patch_id": "16.13", "role": "top", "champion_name": "Camille", "event_end": "2026-07-01T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
]


@pytest.fixture
def table() -> AppearanceTable:
    return AppearanceTable.from_rows(ROWS)


def test_league_scope_filters_exactly_one_league(table: AppearanceTable) -> None:
    scope = league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier1")
    cell = table.filter(scope, as_of="2026-08-01T00:00:00Z")
    membership = cell.membership()
    assert set(membership) == {"Aatrox", "Gnar"}
    assert membership["Aatrox"]["distinct_maps"] == 2
    assert membership["Gnar"]["distinct_maps"] == 1
    # LCS rows never leak into LEC and vice versa
    lcs = table.filter(league_scope("LCS", role="top", patch_id="16.14", competition_tier="tier1"), as_of="2026-08-01T00:00:00Z")
    assert set(lcs.membership()) == {"Aatrox"}


def test_international_scopes_msi_and_ewc_stay_separate(table: AppearanceTable) -> None:
    msi = table.filter(international_scope("MSI", role="top", patch_id="16.13"), as_of="2026-08-01T00:00:00Z")
    ewc = table.filter(international_scope("EWC", role="top", patch_id="16.13"), as_of="2026-08-01T00:00:00Z")
    assert set(msi.membership()) == {"Aatrox"}
    assert set(ewc.membership()) == {"Gnar"}
    # an MSI request must never include EWC rows even though both are international
    assert "Gnar" not in msi.membership()
    assert "Aatrox" not in ewc.membership()


def test_competition_tier_filter_tier1_vs_tier2(table: AppearanceTable) -> None:
    tier1 = table.filter(league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier1"), as_of="2026-08-01T00:00:00Z")
    tier2 = table.filter(league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier2"), as_of="2026-08-01T00:00:00Z")
    assert "Renekton" not in tier1.membership()
    assert set(tier2.membership()) == {"Renekton"}


def test_patch_filter_never_mixes_patches(table: AppearanceTable) -> None:
    old = table.filter(league_scope("LEC", role="top", patch_id="16.13", competition_tier="tier1"), as_of="2026-08-01T00:00:00Z")
    assert set(old.membership()) == {"Camille"}
    new = table.filter(league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier1"), as_of="2026-08-01T00:00:00Z")
    assert "Camille" not in new.membership()


def test_zero_play_champion_is_excluded(table: AppearanceTable) -> None:
    scope = league_scope("LEC", role="mid", patch_id="16.14", competition_tier="tier1")
    cell = table.filter(scope, as_of="2026-08-01T00:00:00Z")
    assert set(cell.membership()) == {"Ahri"}
    assert "Aatrox" not in cell.membership()  # played top, not mid


def test_as_of_excludes_future_appearances(table: AppearanceTable) -> None:
    scope = league_scope("LEC", role="top", patch_id="16.14", competition_tier="tier1")
    cell = table.filter(scope, as_of="2026-07-20T12:00:00Z")
    assert set(cell.membership()) == {"Aatrox"}  # Gnar's map is after as_of
    assert cell.membership()["Aatrox"]["distinct_maps"] == 1


def test_role_aliases_canonicalize_jng_and_sup(table: AppearanceTable) -> None:
    jng = table.filter(league_scope("LEC", role="jungle", patch_id="16.14", competition_tier="tier1"), as_of="2026-08-01T00:00:00Z")
    assert set(jng.membership()) == {"Sejuani"}
    sup = table.filter(league_scope("LEC", role="support", patch_id="16.14", competition_tier="tier1"), as_of="2026-08-01T00:00:00Z")
    assert set(sup.membership()) == {"Thresh"}


def test_latest_patch_uses_most_recent_completed_maps(table: AppearanceTable) -> None:
    assert table.latest_patch("LEC", scope_kind="league", competition_tier="tier1") == "16.14"
    assert table.latest_patch("MSI", scope_kind="international") == "16.13"


def test_latest_patch_conflict_fails_closed() -> None:
    conflicting = ROWS + [
        {"map_id": "m8", "league": "LEC", "patch_id": "16.15", "role": "top", "champion_name": "Aatrox", "event_end": "2026-07-21T10:00:00Z", "competition_tier": "tier1", "event_kind": None},
    ]
    table = AppearanceTable.from_rows(conflicting)
    with pytest.raises(TierListError, match="conflicting patches"):
        table.latest_patch("LEC", scope_kind="league", competition_tier="tier1")


def test_missing_source_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(TierListError, match="missing or not a regular file"):
        AppearanceTable.from_oe_player_games(tmp_path, locator="data/lol/warehouse/parquet/oe_player_games.parquet")


def test_invalid_scope_rejected() -> None:
    with pytest.raises(TierListError):
        AppearanceScope(scope_kind="league", scope_id="MSI", role="top", patch_id="16.14")
    with pytest.raises(TierListError):
        AppearanceScope(scope_kind="international", scope_id="MSI", role="top", patch_id="16.14", competition_tier="tier1")
    with pytest.raises(TierListError):
        AppearanceScope(scope_kind="international", scope_id="WORLDSX", role="top", patch_id="16.14")
    with pytest.raises(TierListError):
        AppearanceScope(scope_kind="league", scope_id="LEC", role="top", patch_id="16.14", competition_tier="silver")


def test_duplicate_rows_deduped_or_conflict_rejected() -> None:
    # identical duplicates (known OE ingest artifacts) are deduplicated
    table = AppearanceTable.from_rows([ROWS[0], ROWS[0]])
    assert len(table.rows()) == 1
    # conflicting duplicates for the same identity fail closed
    conflict = dict(ROWS[0])
    conflict["event_end"] = "2026-07-25T10:00:00Z"
    with pytest.raises(TierListError, match="conflicting duplicate"):
        AppearanceTable.from_rows([ROWS[0], conflict])
