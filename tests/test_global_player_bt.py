from __future__ import annotations

import pandas as pd
import pytest

from lol_kills.ratings.global_player_bt import (
    GlobalPlayerBTConfig,
    GlobalPlayerRatingError,
    _contribution_metrics,
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
                        # The release projection carries the map date on every
                        # player row, and the anchor baselines are ordered by
                        # it, so the fixture carries it too.
                        "date": date,
                        "side": side,
                        "position": role,
                        "playername": f"{team}-{role}",
                        "kills": 999 if team == "minor" else 0,
                    }
                )
    return pd.DataFrame(maps), pd.DataFrame(players)


def _with_metrics(
    players: pd.DataFrame,
    profiles: dict[str, dict[str, float]],
    *,
    tier: str | None = "tier1",
) -> pd.DataFrame:
    """Attach per player contribution metrics to a fixture player frame."""

    frame = players.copy()
    if tier is not None:
        frame["competition_tier"] = tier
    frame["gamelength"] = 1800.0
    columns = sorted({column for profile in profiles.values() for column in profile})
    for column in columns:
        frame[column] = [
            profiles.get(str(name), {}).get(column, float("nan"))
            for name in frame["playername"]
        ]
    return frame


def _profile(
    cs: float,
    gold: float,
    damage: float,
    share: float,
    kills: float,
    deaths: float,
    assists: float,
) -> dict[str, float]:
    return {
        "cspm": cs,
        "totalgold": gold,
        "dpm": damage,
        "damageshare": share,
        "kills": kills,
        "deaths": deaths,
        "assists": assists,
        "wpm": 0.5,
        "wcpm": 0.2,
    }


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
    # A release-grade fit (validate=True) refuses an unanchored ladder, so the
    # holdout fixture carries contribution metrics exactly as production does.
    frame = _with_metrics(players, _roster_profiles())

    _, meta = fit_global_player_bt(
        maps,
        frame,
        _config(minimum_maps=100, minimum_holdout_gain=0.001),
        validate=True,
    )

    assert meta["holdout"]["train_maps"] == 96
    assert meta["holdout"]["test_maps"] == 24
    assert meta["holdout"]["gain"] >= 0.001


_TEAMS = ("alpha", "beta", "gamma", "delta")

# Role-typical output, so a support is never compared against a top laner.
_ROLE_BASE = {
    "top": _profile(8.0, 12500.0, 520.0, 0.23, 3.0, 3.0, 5.0),
    "jng": _profile(5.0, 11000.0, 400.0, 0.18, 3.0, 3.0, 7.0),
    "mid": _profile(9.1, 13000.0, 650.0, 0.27, 4.0, 3.0, 5.0),
    "bot": _profile(9.8, 13500.0, 700.0, 0.29, 5.0, 3.0, 4.0),
    "sup": _profile(1.0, 8700.0, 210.0, 0.10, 1.0, 4.0, 10.0),
}


def _quality_profile(role: str, quality: float) -> dict[str, float]:
    base = _ROLE_BASE[role]
    profile = {key: value * quality for key, value in base.items()}
    profile["deaths"] = base["deaths"] / quality
    return profile


def _roster_profiles(
    overrides: dict[str, dict[str, float]] | None = None,
    *,
    drop: tuple[str, ...] = (),
) -> dict[str, dict[str, float]]:
    """Four locked rosters, every role peer on a distinct quality level.

    Distinct levels matter: with only two players per role a z-score collapses
    to a sign, which is not a fair exercise of the anchor.
    """

    profiles: dict[str, dict[str, float]] = {}
    for team_index, team in enumerate(_TEAMS):
        for role_index, role in enumerate(ROLES):
            quality = 1.0 + 0.05 * (((team_index * 5 + role_index) % 7) - 3)
            profiles[f"{team}-{role}"] = _quality_profile(role, quality)
    profiles.update(overrides or {})
    for name in drop:
        profiles.pop(name, None)
    return profiles


def _locked_roster_games(rounds: int = 6) -> list[tuple[str, str, str, int, str]]:
    """Locked rosters in a full round robin, so teammates always co-occur."""

    pairs = [
        (left, right)
        for index, left in enumerate(_TEAMS)
        for right in _TEAMS[index + 1 :]
    ]
    games = []
    for round_number in range(rounds):
        for index, (left, right) in enumerate(pairs):
            blue_win = int((round_number + index) % 3 != 0)
            home, away = (left, right) if (round_number + index) % 2 == 0 else (right, left)
            games.append((f"game-{round_number}-{index}", home, away, blue_win, "LCK"))
    return games


def test_always_together_players_separate_by_role_normalized_contribution() -> None:
    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(
        players,
        _roster_profiles(
            {
                "alpha-top": _quality_profile("top", 1.25),
                "alpha-sup": _quality_profile("sup", 0.75),
            }
        ),
    )

    snapshot, meta = fit_global_player_bt(maps, players=frame, cfg=_config(), validate=False)
    ratings = snapshot.set_index("player")["global_rating"]

    assert meta["performance_anchor"]["enabled"] is True
    assert meta["performance_anchor"]["players_anchored"] == 20
    assert meta["performance_anchor"]["players_without_metrics"] == 0
    assert meta["performance_anchor"]["normalization"] == "role+competition_tier"
    assert meta["player_statistics_used"] is True
    # The whole point: identical design columns, different ratings.
    assert ratings["alpha-top"] != ratings["alpha-sup"]
    assert ratings["alpha-top"] > ratings["alpha-sup"]
    # The baseline is a median/MAD, which reads RANK rather than exact spacing.
    # This fixture has only four players per (role, tier) pool, so two of
    # alpha's seats can hold the same rank in pools of the same shape and land
    # on the same z.  Four of the five still separate, and the two overridden
    # seats are ordered correctly, which is the property under test.  At census
    # scale the ladder keeps the same 3506 distinct ratings over 3797 players
    # it had under mean/std.
    assert len({ratings[f"alpha-{role}"] for role in ROLES}) >= 4
    anchors = snapshot.set_index("player")["global_performance_anchor_logit"]
    assert anchors["alpha-top"] > 0.0 > anchors["alpha-sup"]


def test_role_normalization_does_not_punish_a_low_cs_support() -> None:
    maps, players = _fixture(_locked_roster_games())
    # The support farms about 1 cs/min against the top laner's 6, but is the
    # better player *for their own role*.
    frame = _with_metrics(
        players,
        _roster_profiles(
            {
                "alpha-top": _quality_profile("top", 0.75),
                "alpha-sup": _quality_profile("sup", 1.25),
            }
        ),
    )

    snapshot, _ = fit_global_player_bt(maps, players=frame, cfg=_config(), validate=False)
    anchors = snapshot.set_index("player")["global_performance_anchor_logit"]
    ratings = snapshot.set_index("player")["global_rating"]

    assert _ROLE_BASE["sup"]["cspm"] * 1.25 < _ROLE_BASE["top"]["cspm"] * 0.75
    assert anchors["alpha-sup"] > 0.0
    assert anchors["alpha-top"] < 0.0
    assert ratings["alpha-sup"] > ratings["alpha-top"]


def test_disabled_performance_anchor_restores_shrink_to_zero_behaviour() -> None:
    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(
        players,
        _roster_profiles(
            {
                "alpha-top": _quality_profile("top", 1.25),
                "alpha-sup": _quality_profile("sup", 0.75),
            }
        ),
    )
    off = _config(performance_anchor_enabled=False)

    bare, bare_meta = fit_global_player_bt(maps, players=players, cfg=off, validate=False)
    rich, rich_meta = fit_global_player_bt(maps, players=frame, cfg=off, validate=False)

    # Contribution metrics are inert while the anchor is off, and the snapshot
    # carries no anchor columns at all.
    pd.testing.assert_frame_equal(bare, rich)
    assert "global_performance_anchor_logit" not in rich.columns
    assert rich_meta["player_statistics_used"] is False
    assert rich_meta["performance_anchor"]["enabled"] is False
    assert bare_meta["performance_anchor"]["players_anchored"] == 0
    ratings = rich.set_index("player")["global_rating"]
    assert len({ratings[f"alpha-{role}"] for role in ROLES}) == 1


def test_players_without_contribution_metrics_keep_a_zero_anchor() -> None:
    maps, players = _fixture(_locked_roster_games())
    # These two seats carry no usable metric at all, so they must fall back to
    # exactly the old shrink-to-zero behaviour rather than borrow a mean.
    frame = _with_metrics(
        players, _roster_profiles(drop=("alpha-sup", "beta-sup"))
    )

    snapshot, meta = fit_global_player_bt(maps, players=frame, cfg=_config(), validate=False)
    anchored = snapshot.set_index("player")["global_performance_anchored"]
    anchors = snapshot.set_index("player")["global_performance_anchor_logit"]

    assert anchored["alpha-sup"] == 0
    assert anchored["beta-sup"] == 0
    assert anchors["alpha-sup"] == 0.0
    assert anchors["beta-sup"] == 0.0
    assert meta["performance_anchor"]["players_without_metrics"] == 2
    assert meta["performance_anchor"]["players_anchored"] == 18


def test_incomplete_team_gold_withholds_the_share_metric() -> None:
    maps, players = _fixture(_locked_roster_games())
    profiles = _roster_profiles()
    without_gold = dict(profiles["alpha-sup"])
    without_gold.pop("totalgold")
    profiles["alpha-sup"] = without_gold
    frame = _with_metrics(players, profiles)

    metrics = _contribution_metrics(frame)
    alpha_rows = metrics[metrics["_player"].astype(str).str.startswith("alpha-")]
    beta_rows = metrics[metrics["_player"].astype(str).str.startswith("beta-")]

    # One missing seat would otherwise inflate every teammate's gold share.
    assert alpha_rows["gold_share_pct"].isna().all()
    assert beta_rows["gold_share_pct"].notna().all()


def test_performance_anchor_is_zero_mean_and_leaves_the_scale_unchanged() -> None:
    maps, players = _fixture(_locked_roster_games())
    frame = _with_metrics(
        players,
        _roster_profiles(
            {
                "alpha-top": _quality_profile("top", 1.25),
                "alpha-sup": _quality_profile("sup", 0.75),
            }
        ),
    )

    cfg = _config()
    anchored, meta = fit_global_player_bt(maps, players=frame, cfg=cfg, validate=False)
    plain, _ = fit_global_player_bt(
        maps, players=frame, cfg=_config(performance_anchor_enabled=False), validate=False
    )

    assert abs(float(anchored["global_performance_anchor_logit"].sum())) < 1e-12
    assert abs(float(meta["performance_anchor"]["anchor_mean_logit"])) < 1e-12
    assert abs(
        float(anchored["global_rating"].mean()) - float(plain["global_rating"].mean())
    ) < 1e-6
    assert abs(float(anchored["global_rating"].mean()) - cfg.prior_rating) < 1e-6
