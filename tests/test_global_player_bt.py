from __future__ import annotations

import pandas as pd
import pytest

from lol_kills.ratings.global_player_bt import (
    GlobalPlayerBTConfig,
    GlobalPlayerRatingError,
    fit_global_player_bt,
)
from lol_kills.ratings.player_elo import PlayerEloConfig, _apply_bridge_uncertainty


ROLES = ("top", "jng", "mid", "bot", "sup")


def _fixture(
    games: list[tuple[str, str, str, int, str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = []
    players = []
    for number, (game_id, blue, red, blue_win, league) in enumerate(games):
        date = f"2026-01-{1 + number // 20:02d}T{number % 20:02d}:00:00Z"
        maps.append(
            {
                "gameid": game_id,
                "date": date,
                "league": league,
                "y_blue_win": blue_win,
            }
        )
        for side, team in (("Blue", blue), ("Red", red)):
            for role in ROLES:
                players.append(
                    {
                        "gameid": game_id,
                        "side": side,
                        "position": role,
                        "playername": f"{team}-{role}",
                        "kills": 999 if team == "minor" else 0,
                    }
                )
    return pd.DataFrame(maps), pd.DataFrame(players)


def _config(**changes: object) -> GlobalPlayerBTConfig:
    values = {
        "minimum_maps": 1,
        "minimum_connected_share": 0.95,
        "minimum_holdout_gain": -1.0,
    }
    values.update(changes)
    return GlobalPlayerBTConfig(**values)


def test_cross_pool_games_put_domestic_results_on_one_scale() -> None:
    games = []
    for index in range(30):
        major_win = int(index < 18)
        minor_win = int(index < 27)
        if index % 2 == 0:
            games.append((f"major-{index}", "major", "major-opp", major_win, "LCK"))
            games.append((f"minor-{index}", "minor", "minor-opp", minor_win, "TIER3"))
        else:
            games.append((f"major-{index}", "major-opp", "major", 1 - major_win, "LCK"))
            games.append((f"minor-{index}", "minor-opp", "minor", 1 - minor_win, "TIER3"))
    for index in range(12):
        games.append(
            (f"bridge-{index}", "major", "minor", 1, "INTL")
            if index % 2 == 0
            else (f"bridge-{index}", "minor", "major", 0, "INTL")
        )
    maps, players = _fixture(games)

    snapshot, meta = fit_global_player_bt(maps, players, _config(), validate=False)
    by_player = snapshot.set_index("player")["global_rating"]

    assert by_player["major-mid"] > by_player["minor-mid"]
    assert meta["n_components"] == 1
    assert meta["connected_share"] == 1.0
    assert meta["tier_adjustments"] is False


def test_tier_labels_and_player_box_score_columns_do_not_change_rating() -> None:
    games = [
        (f"game-{index}", "alpha", "beta", int(index % 3 != 0), "LCK")
        for index in range(18)
    ]
    maps, players = _fixture(games)
    first, _ = fit_global_player_bt(maps, players, _config(), validate=False)

    changed_maps = maps.copy()
    changed_maps["league"] = "TIER3"
    changed_players = players.copy()
    changed_players["kills"] = changed_players["kills"] * -100
    second, _ = fit_global_player_bt(changed_maps, changed_players, _config(), validate=False)

    left = first.set_index("player")["global_rating"].sort_index()
    right = second.set_index("player")["global_rating"].sort_index()
    pd.testing.assert_series_equal(left, right)


def test_role_alone_does_not_change_ratings_within_a_fixed_lineup() -> None:
    games = [
        (f"game-{index}", "alpha", "beta", int(index % 3 != 0), "LCK")
        for index in range(18)
    ]
    maps, players = _fixture(games)

    snapshot, _ = fit_global_player_bt(maps, players, _config(), validate=False)
    ratings = snapshot.set_index("player")["global_rating"]

    alpha_ratings = [ratings[f"alpha-{role}"] for role in ROLES]
    beta_ratings = [ratings[f"beta-{role}"] for role in ROLES]
    assert max(alpha_ratings) - min(alpha_ratings) < 1e-8
    assert max(beta_ratings) - min(beta_ratings) < 1e-8


def test_disconnected_player_pools_fail_closed() -> None:
    games = []
    for index in range(10):
        games.append((f"one-{index}", "a", "b", int(index % 2 == 0), "ONE"))
        games.append((f"two-{index}", "c", "d", int(index % 2 == 0), "TWO"))
    maps, players = _fixture(games)

    with pytest.raises(GlobalPlayerRatingError, match="largest player component"):
        fit_global_player_bt(maps, players, _config(), validate=False)


def test_stronger_tier_history_reduces_bridge_uncertainty_without_moving_mean() -> None:
    rows = [{"player": "prospect", "mu_total": 1650.0, "sigma": 28.0}]
    records = {"prospect": {"current_tier": "tier2"}}
    no_bridge = pd.DataFrame(
        [{"gameid": "local", "date": "2026-01-01", "league": "LFL", "playername": "prospect"}]
    )
    bridged = pd.DataFrame(
        [
            {"gameid": f"major-{index}", "date": "2026-01-01", "league": "LEC", "playername": "prospect"}
            for index in range(10)
        ]
    )

    local_row = _apply_bridge_uncertainty(rows, no_bridge, records, PlayerEloConfig())[0]
    bridged_row = _apply_bridge_uncertainty(rows, bridged, records, PlayerEloConfig())[0]

    assert local_row["mu_total"] == bridged_row["mu_total"] == 1650.0
    assert local_row["global_bridge_sigma"] == 45.0
    assert bridged_row["global_bridge_sigma"] < local_row["global_bridge_sigma"]
    assert bridged_row["sigma"] < local_row["sigma"]


def test_holdout_evidence_is_chronological_and_beats_side_only_baseline() -> None:
    games = []
    for index in range(120):
        alpha_win = int(index % 5 != 0)
        if index % 2 == 0:
            games.append((f"game-{index}", "alpha", "beta", alpha_win, "LCS"))
        else:
            games.append((f"game-{index}", "beta", "alpha", 1 - alpha_win, "LCS"))
    maps, players = _fixture(games)

    _, meta = fit_global_player_bt(
        maps,
        players,
        _config(minimum_maps=100, minimum_holdout_gain=0.001),
        validate=True,
    )

    assert meta["holdout"]["train_maps"] == 96
    assert meta["holdout"]["test_maps"] == 24
    assert meta["holdout"]["gain"] >= 0.001
