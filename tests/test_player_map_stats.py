"""Contract tests for the public per-map Stats artifact.

The banned-key check below is a deliberate reimplementation of
``public.scryglass_json_has_draft_fields`` from
``supabase/migrations/20260815060001_descriptive_draft_query_api.sql``.  That
function is what actually rejects a published asset in Postgres, and the
failure surfaces only at publication time, so the ban list is mirrored here and
pinned against the migration text by
``test_banned_key_list_matches_the_migration``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.export.player_map_stats import (
    DEFAULT_MAP_LIMIT,
    DEFAULT_WINDOW_DAYS,
    SCHEMA_VERSION,
    build_player_map_stats,
    player_map_stats_row_count,
)


ROOT = Path(__file__).resolve().parents[1]
GUARD_MIGRATION = ROOT / "supabase/migrations/20260815060001_descriptive_draft_query_api.sql"

# Mirrors the ``normalized_key = any(array[...])`` list in the guard.
BANNED_KEYS = frozenset({
    "average_win_share", "best_available", "betting", "development_composite",
    "draft_authority", "draft_edge", "draft_pool", "draft_probability",
    "draft_score", "draft_win_share", "elo", "ev", "expected_value",
    "fair_odds", "gold", "live_state", "match_probability",
    "match_win_expectation", "momentum", "mu_diff", "objectives", "odds",
    "p_blue", "p_red", "phase_curve", "player_elo", "player_rating",
    "probability", "r9e", "r9e_state_space", "rating_uncertainty",
    "recommendation", "sigma_pair", "strength", "team_elo", "team_rating",
    "win_probability",
})

# Mirrors the ``normalized_key like '...\_%'`` clauses in the guard.
BANNED_PREFIXES = ("r9e_", "control_", "strength_", "phase_", "live_")

# The guard also treats these two keys as banned unless their value validates
# against a descriptive-authority schema the Stats artifact never carries.
BANNED_UNLESS_DESCRIPTIVE = ("draft_contribution", "draft_metric")


def banned_keys(value: object, path: str = "$") -> list[str]:
    """Return every JSON path whose key the Supabase guard would reject."""

    found: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            here = f"{path}.{key}"
            if (
                normalized in BANNED_KEYS
                or normalized in BANNED_UNLESS_DESCRIPTIVE
                or normalized.startswith(BANNED_PREFIXES)
            ):
                found.append(here)
            found.extend(banned_keys(child, here))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(banned_keys(child, f"{path}[{index}]"))
    return found


def test_banned_key_list_matches_the_migration() -> None:
    """Pin the local ban list to the SQL the database actually enforces."""

    source = GUARD_MIGRATION.read_text(encoding="utf-8")
    guard = source.split("create or replace function public.scryglass_json_has_draft_fields", 1)[1]
    array_literal = guard.split("normalized_key = any(array[", 1)[1].split("])", 1)[0]
    assert set(re.findall(r"'([^']+)'", array_literal)) == BANNED_KEYS
    for prefix in BANNED_PREFIXES:
        escaped = prefix[:-1] + r"\_%"
        assert f"normalized_key like '{escaped}' escape '\\'" in guard


def _find(entries: list[dict], name: str) -> dict:
    """Resolve one identity from the array-shaped payload."""

    match = [entry for entry in entries if entry["name"] == name]
    assert len(match) == 1, f"{name} is not present exactly once"
    return match[0]


def _frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _game(
    game_id: str,
    date: str,
    *,
    blue: str = "Blue Team",
    red: str = "Red Team",
    blue_win: int = 1,
    league: str = "LCK",
    gold: float = 12000.0,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    roles = ("top", "jng", "mid", "bot", "sup")
    for side, team, win in (("Blue", blue, blue_win), ("Red", red, 1 - blue_win)):
        for index, role in enumerate(roles):
            rows.append({
                "gameid": game_id,
                "date": date,
                "league": league,
                "result": win,
                "side": side,
                "position": role,
                "teamname": team,
                "playername": f"{team[:1]}{side}{index}",
                "champion": f"Champ{index}",
                "kills": 3.0,
                "deaths": 2.0,
                "assists": 5.0,
                "teamkills": 15.0,
                "gamelength": 1800.0,
                "dpm": 500.0,
                "damageshare": 0.2,
                "totalgold": gold,
                "cspm": 8.0,
            })
    return rows


def test_payload_shape_and_derivations() -> None:
    payload = build_player_map_stats(_frame(_game("g1", "2026-01-10")))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["window_days"] == DEFAULT_WINDOW_DAYS
    assert payload["map_limit"] == DEFAULT_MAP_LIMIT

    player = _find(payload["players"], "BBlue2")
    assert player["maps"] == 1
    assert player["wins"] == 1
    assert player["losses"] == 0
    assert player["win_rate"] == 1.0
    row = player["games"][0]
    assert row["date"] == "2026-01-10"
    assert row["league"] == "LCK"
    assert row["opponent"] == "Red Team"
    assert row["champion"] == "Champ2"
    assert row["position"] == "mid"
    assert row["win"] is True
    assert row["game_length_min"] == 30.0
    assert row["cs_per_min"] == 8.0
    # 12000 gold over 30 minutes.
    assert row["gold_per_min"] == 400.0
    # One of five equal shares of the side's gold.
    assert row["gold_share_pct"] == 20.0
    assert row["damage_per_min"] == 500.0
    assert row["damage_share_pct"] == 20.0
    # (3 kills + 5 assists) / 2 deaths.
    assert row["kda"] == 4.0

    team = _find(payload["teams"], "Blue Team")
    assert team["maps"] == 1
    team_row = team["games"][0]
    assert team_row["opponent"] == "Red Team"
    assert team_row["win"] is True
    # Five players at 12000 gold over 30 minutes.
    assert team_row["gold_per_min"] == 2000.0
    assert team_row["damage_per_min"] == 2500.0
    assert team_row["kills"] == 15
    assert team_row["deaths"] == 10


def test_emitted_keys_pass_the_supabase_guard() -> None:
    payload = build_player_map_stats(
        _frame([*_game("g1", "2026-01-10"), *_game("g2", "2026-01-11", blue_win=0)])
    )
    assert banned_keys(payload) == []
    # The exact spellings the profile depends on, proven present rather than
    # merely absent from the ban list.
    row = _find(payload["players"], "BBlue0")["games"][0]
    assert {
        "cs_per_min", "gold_per_min", "gold_share_pct", "damage_per_min",
        "damage_share_pct", "kda", "game_length_min", "opponent", "champion",
        "league", "date", "win",
    }.issubset(row)
    assert "gold" not in row


def test_identity_names_are_values_so_a_banned_name_cannot_fail_publication() -> None:
    """A roster is data we do not control; it must not reach the guard as a key.

    The Supabase guard inspects JSON object *keys*.  If identities were the keys
    of a map, a player called ``Gold`` or an org tagged ``EV`` would fail
    publication with an error pointing at the asset rather than at the roster.
    """

    payload = build_player_map_stats(
        _frame(_game("g1", "2026-01-10", blue="Gold", red="EV"))
    )
    assert isinstance(payload["players"], list)
    assert isinstance(payload["teams"], list)
    assert {entry["name"] for entry in payload["teams"]} == {"Gold", "EV"}
    assert banned_keys(payload) == []


def test_window_excludes_older_maps() -> None:
    payload = build_player_map_stats(
        _frame([*_game("old", "2025-01-01"), *_game("new", "2026-01-10")]),
        window_days=30,
    )
    assert [row["game_id"] for row in _find(payload["players"], "BBlue0")["games"]] == ["new"]


def test_rows_are_most_recent_first_and_capped() -> None:
    rows: list[dict[str, object]] = []
    for day in range(1, 8):
        rows.extend(_game(f"g{day}", f"2026-01-0{day}"))
    payload = build_player_map_stats(_frame(rows), map_limit=3)
    games = _find(payload["players"], "BBlue0")["games"]
    assert [row["date"] for row in games] == ["2026-01-07", "2026-01-06", "2026-01-05"]
    # The header describes exactly the capped rows it sits above.
    assert _find(payload["players"], "BBlue0")["maps"] == 3


def test_same_day_series_orders_by_timestamp_not_by_game_id() -> None:
    """A ten-game day must not order ``_10`` ahead of ``_9``.

    ``date`` is truncated to the calendar day for display.  Sorting on that
    truncation leaves the game identity as the only tie-break, and lexically
    ``series_10`` precedes ``series_9``.  The full source timestamp decides.
    """

    rows: list[dict[str, object]] = []
    for game in range(1, 11):
        rows.extend(_game(f"series_{game}", f"2026-01-05T{game + 6:02d}:00:00Z"))
    payload = build_player_map_stats(_frame(rows), map_limit=3)
    games = _find(payload["players"], "BBlue0")["games"]
    assert [row["game_id"] for row in games] == ["series_10", "series_9", "series_8"]
    assert {row["date"] for row in games} == {"2026-01-05"}
    # The team rows follow the same order.
    team_games = _find(payload["teams"], "Blue Team")["games"]
    assert [row["game_id"] for row in team_games] == ["series_10", "series_9", "series_8"]


def test_identical_timestamps_break_ties_on_natural_game_order() -> None:
    """Without a distinguishing timestamp the identity still orders naturally."""

    rows: list[dict[str, object]] = []
    for game in (9, 10):
        rows.extend(_game(f"series_{game}", "2026-01-05T09:00:00Z"))
    payload = build_player_map_stats(_frame(rows))
    games = _find(payload["players"], "BBlue0")["games"]
    assert [row["game_id"] for row in games] == ["series_10", "series_9"]


def test_no_private_ordering_key_reaches_the_payload() -> None:
    rows = [*_game("g1", "2026-01-10T09:00:00Z"), *_game("g2", "2026-01-10T11:00:00Z")]
    payload = build_player_map_stats(_frame(rows))
    for kind in ("players", "teams"):
        for entry in payload[kind]:
            for row in entry["games"]:
                assert not any(str(key).startswith("_") for key in row), row


def test_missing_metric_is_null_never_zero() -> None:
    rows = _game("g1", "2026-01-10")
    for row in rows:
        row["cspm"] = None
        row["dpm"] = float("nan")
    payload = build_player_map_stats(_frame(rows))
    player = _find(payload["players"], "BBlue0")
    assert player["games"][0]["cs_per_min"] is None
    assert player["games"][0]["damage_per_min"] is None
    assert player["cs_per_min"] is None
    assert player["damage_per_min"] is None
    # A metric that is still measurable is unaffected.
    assert player["games"][0]["gold_per_min"] == 400.0


def test_incomplete_side_yields_unavailable_shares_not_wrong_ones() -> None:
    rows = [row for row in _game("g1", "2026-01-10") if row["playername"] != "BBlue4"]
    payload = build_player_map_stats(_frame(rows))
    assert _find(payload["players"], "BBlue0")["games"][0]["gold_share_pct"] is None
    # The team row refuses to sum a four-player side.
    assert _find(payload["teams"], "Blue Team")["games"][0]["gold_per_min"] is None
    assert _find(payload["teams"], "Blue Team")["games"][0]["kills"] is None


def test_zero_length_game_does_not_divide_by_zero() -> None:
    rows = _game("g1", "2026-01-10")
    for row in rows:
        row["gamelength"] = 0.0
    payload = build_player_map_stats(_frame(rows))
    row = _find(payload["players"], "BBlue0")["games"][0]
    assert row["gold_per_min"] is None
    assert row["game_length_min"] is None
    # A per-minute source column is still reported as measured.
    assert row["cs_per_min"] == 8.0


def test_rows_without_a_usable_result_are_dropped() -> None:
    rows = _game("g1", "2026-01-10")
    for row in rows:
        row["result"] = None
    payload = build_player_map_stats(_frame(rows))
    assert payload["players"] == []
    assert payload["teams"] == []


def test_empty_and_incomplete_sources_degrade_to_an_empty_payload() -> None:
    empty = build_player_map_stats(pd.DataFrame())
    assert empty["players"] == [] and empty["teams"] == []
    assert empty["schema_version"] == SCHEMA_VERSION
    missing = build_player_map_stats(_frame([{"playername": "A", "date": "2026-01-01"}]))
    assert missing["players"] == []


def test_invalid_bounds_are_rejected() -> None:
    with pytest.raises(ValueError):
        build_player_map_stats(_frame(_game("g1", "2026-01-10")), window_days=0)
    with pytest.raises(ValueError):
        build_player_map_stats(_frame(_game("g1", "2026-01-10")), map_limit=0)


def test_payload_is_json_serializable_with_native_scalars() -> None:
    payload = build_player_map_stats(_frame(_game("g1", "2026-01-10")))
    encoded = json.dumps(payload)
    assert json.loads(encoded) == payload
    assert player_map_stats_row_count(payload) == 12
