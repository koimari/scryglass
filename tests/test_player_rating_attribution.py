"""Per-player attribution for the descriptive player Elo baseline.

The baseline applied one shared team-outcome residual to all five players on a
side, so teammates whose sigma had converged received byte-identical updates
forever.  These fixtures pin the replacement:

* the multipliers average to exactly 1 within a side, so the side's aggregate
  update is unchanged and only the split among teammates moves;
* teammates on the same map no longer receive identical updates;
* the baseline is leakage-safe — a map never contributes to its own
  (role, competition_tier) baseline, and later maps never change an earlier
  map's multiplier;
* every unavailable metric, thin baseline, or missing column falls back to a
  neutral multiplier of exactly 1.0 and is counted, never imputed;
* ``attribution_enabled=False`` reproduces the pre-attribution replay exactly.

Nothing here fits or validates the composite weights.  They are an UNFITTED
equal-weight development default with no promotion authority.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from lol_kills.ratings.player_elo import (
    ATTRIBUTION_FEATURE_WEIGHTS,
    ATTRIBUTION_MIN_BASELINE_OBS,
    ATTRIBUTION_WEIGHTS_STATUS,
    PlayerEloConfig,
    _lineups_by_game,
    _run_player_elo,
    build_maps_frame_from_players,
    player_attribution_multipliers,
)

ROLES = ("top", "jng", "mid", "bot", "sup")


def _players(n_maps: int = 40, *, with_metrics: bool = True) -> pd.DataFrame:
    """Deterministic two-team history with separable per-player box scores."""

    rows = []
    for game in range(n_maps):
        blue_win = game % 2
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=game)
        for side, team, result in (("Blue", "A", blue_win), ("Red", "B", 1 - blue_win)):
            for slot, role in enumerate(ROLES):
                row = {
                    "gameid": f"g{game:03d}",
                    "date": date.isoformat(),
                    "league": "LCK",
                    "competition_tier": "tier1",
                    "side": side,
                    "position": role,
                    "playername": f"{team}{slot}",
                    "teamname": team,
                    "result": result,
                }
                if with_metrics:
                    # Separable, deterministic, and never constant within a
                    # (role, tier) baseline: slot sets the level, game adds
                    # spread so the prior standard deviation is positive.
                    lift = 1.0 + 0.05 * slot + 0.01 * (game % 7)
                    offset = 1 if side == "Red" else 0
                    row.update(
                        {
                            "cspm": 6.0 * lift + 0.1 * offset,
                            "dpm": 500.0 * lift,
                            "damageshare": 0.14 + 0.02 * slot + 0.001 * (game % 5),
                            "totalgold": int(11000 * lift) + 13 * offset,
                            "earnedgold": int(8000 * lift),
                            "kills": (slot + game) % 9,
                            "deaths": (slot + game) % 5,
                            "assists": (2 * slot + game) % 11,
                            "teamkills": 15 + (game % 6),
                            "gamelength": 1800 + 30 * (game % 8),
                            "wpm": 0.4 + 0.15 * slot + 0.01 * (game % 4),
                            "wcpm": 0.2 + 0.05 * slot + 0.005 * (game % 3),
                        }
                    )
                rows.append(row)
    return pd.DataFrame(rows)


def _multipliers(players: pd.DataFrame, cfg: PlayerEloConfig | None = None):
    cfg = cfg or PlayerEloConfig()
    lineups, metrics = _lineups_by_game(players, with_metrics=True)
    mapping, stats = player_attribution_multipliers(metrics, cfg)
    return lineups, mapping, stats


def test_weight_vector_is_a_named_unfitted_development_default() -> None:
    assert ATTRIBUTION_WEIGHTS_STATUS == "unfitted_development_default"
    assert set(ATTRIBUTION_FEATURE_WEIGHTS) == {
        "cs_per_min",
        "gold_per_min",
        "gold_share_pct",
        "damage_per_min",
        "damage_share_pct",
        "kda_role_weighted",
        "wpm",
        "wcpm",
    }
    # Equal weights: the composite is a plain mean of the available z-scores.
    assert len(set(ATTRIBUTION_FEATURE_WEIGHTS.values())) == 1


def test_multipliers_average_to_exactly_one_within_every_side() -> None:
    lineups, mapping, _stats = _multipliers(_players())
    checked = 0
    for gid, sides in lineups.items():
        for side in ("Blue", "Red"):
            lineup = sides[side]
            assert len(lineup) == 5
            values = [mapping.get((gid, side, name), 1.0) for name, _role in lineup]
            assert abs(float(np.mean(values)) - 1.0) <= 1e-12
            checked += 1
    assert checked > 0


def test_side_aggregate_update_is_unchanged_by_reallocation() -> None:
    """Conservation: the side's summed rating movement is preserved.

    Role weights are disabled here so the team aggregate is the plain mean the
    conservation proof is stated against.
    """

    players = _players()
    maps = build_maps_frame_from_players(players)
    base = PlayerEloConfig(use_role_weights=False, attribution_enabled=False)
    on = PlayerEloConfig(use_role_weights=False, attribution_enabled=True)
    _r, states_off, _c, _m = _run_player_elo(maps, players, base)
    _r, states_on, _c, _m = _run_player_elo(maps, players, on)

    for team in ("A", "B"):
        roster = [f"{team}{slot}" for slot in range(5)]
        total_off = sum(
            states_off[name].mu_regional + states_off[name].mu_meta for name in roster
        )
        total_on = sum(
            states_on[name].mu_regional + states_on[name].mu_meta for name in roster
        )
        assert total_off == pytest.approx(total_on, abs=1e-6)


def test_teammates_no_longer_receive_identical_updates() -> None:
    players = _players()
    maps = build_maps_frame_from_players(players)
    _r, states_off, _c, _m = _run_player_elo(
        maps, players, PlayerEloConfig(attribution_enabled=False)
    )
    _r, states_on, _c, _m = _run_player_elo(
        maps, players, PlayerEloConfig(attribution_enabled=True)
    )

    def duplicate_pairs(states) -> int:
        pairs = 0
        for team in ("A", "B"):
            keys = [
                (
                    states[f"{team}{slot}"].mu_regional,
                    states[f"{team}{slot}"].mu_meta,
                    states[f"{team}{slot}"].sigma,
                    states[f"{team}{slot}"].n_maps,
                )
                for slot in range(5)
            ]
            pairs += sum(
                1
                for i in range(len(keys))
                for j in range(i + 1, len(keys))
                if keys[i] == keys[j]
            )
        return pairs

    assert duplicate_pairs(states_off) == 20  # every teammate pair collides
    assert duplicate_pairs(states_on) == 0


def test_disabled_attribution_reproduces_the_shared_update_baseline() -> None:
    players = _players()
    maps = build_maps_frame_from_players(players)
    replay_off, states_off, _c, _m = _run_player_elo(
        maps, players, PlayerEloConfig(attribution_enabled=False)
    )
    replay_bare, states_bare, _c, _m = _run_player_elo(
        maps, _players(with_metrics=False), PlayerEloConfig(attribution_enabled=True)
    )
    # No metric columns at all -> every row fails closed to neutral, so the
    # replay must match the disabled path bit for bit.
    pd.testing.assert_frame_equal(replay_off, replay_bare)
    for name, state in states_off.items():
        assert state.mu_regional == states_bare[name].mu_regional
        assert state.mu_meta == states_bare[name].mu_meta
        assert state.sigma == states_bare[name].sigma


def test_missing_metrics_fail_closed_to_neutral_and_are_counted() -> None:
    _lineups, mapping, stats = _multipliers(_players(with_metrics=False))
    assert mapping == {}
    assert stats["rows_graded"] == 0
    assert stats["rows_neutral_no_composite"] == stats["rows_total"]
    assert stats["unavailable_reason"] == "no_row_cleared_the_baseline_floor"
    for count in stats["feature_available_rows"].values():
        assert count == 0


def test_thin_baselines_fall_back_to_neutral_and_are_counted() -> None:
    lineups, mapping, stats = _multipliers(_players())
    # 5 roles x 1 tier x ATTRIBUTION_MIN_BASELINE_OBS prior observations must
    # elapse before any row can be standardized.
    expected_neutral = len(ROLES) * ATTRIBUTION_MIN_BASELINE_OBS
    assert stats["rows_neutral_no_composite"] == expected_neutral
    assert stats["rows_graded"] == stats["rows_total"] - expected_neutral
    assert stats["rows_neutral_renorm_guard"] == 0

    # The neutral rows are the earliest maps, and they are exactly 1.0.
    early = sorted(lineups)[: ATTRIBUTION_MIN_BASELINE_OBS // 2]
    for gid in early:
        for side in ("Blue", "Red"):
            for name, _role in lineups[gid][side]:
                assert mapping.get((gid, side, name), 1.0) == 1.0


def test_baseline_is_leakage_safe_against_later_maps() -> None:
    """A map's multiplier must not move when later maps are added."""

    short = _players(n_maps=30)
    long = _players(n_maps=40)
    _l_short, short_map, _s = _multipliers(short)
    _l_long, long_map, _s = _multipliers(long)

    shared = [key for key in short_map if key[0] <= "g029"]
    assert shared, "expected overlapping graded maps"
    for key in shared:
        assert long_map[key] == short_map[key]


def test_multipliers_stay_inside_the_configured_band() -> None:
    cfg = PlayerEloConfig()
    _lineups, mapping, _stats = _multipliers(_players(), cfg)
    values = np.array(list(mapping.values()), dtype=float)
    assert values.size > 0
    # 1 +/- beta before renormalization; renormalization keeps it well inside
    # a generous envelope and can never reach zero or flip sign.
    bound = (1.0 + cfg.attribution_beta) / (1.0 - cfg.attribution_beta)
    assert values.min() > 0.0
    assert values.max() <= bound
    assert np.isfinite(values).all()


def test_lineup_tuple_contract_is_unchanged_by_the_metric_extension() -> None:
    players = _players().sample(frac=1, random_state=7).reset_index(drop=True)
    plain = _lineups_by_game(players)
    carried, metrics = _lineups_by_game(players, with_metrics=True)
    assert plain == carried
    expected_rows = sum(len(carried[gid][side]) for gid in carried for side in ("Blue", "Red"))
    assert len(metrics) == expected_rows
    assert {"_gid", "side", "_name", "_role", "_attr_date", "_attr_tier"}.issubset(
        metrics.columns
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"attribution_beta": -0.1},
        {"attribution_beta": 1.0},
        {"attribution_beta": float("nan")},
        {"attribution_clip": 0.0},
        {"attribution_clip": -1.0},
        {"attribution_clip": float("inf")},
        {"attribution_enabled": "yes"},
    ],
)
def test_invalid_attribution_configuration_fails_closed(kwargs) -> None:
    with pytest.raises(ValueError):
        PlayerEloConfig(**kwargs)


def test_gold_enters_only_as_a_rate_and_a_within_team_share() -> None:
    """Raw gold must never be a direct feature."""

    from lol_kills.ratings.player_elo import _attribution_features

    _lineups, metrics = _lineups_by_game(_players(), with_metrics=True)
    features = _attribution_features(metrics)
    assert "totalgold" not in features.columns
    assert "earnedgold" not in features.columns
    share = features["gold_share_pct"].to_numpy(dtype=float)
    assert np.nanmin(share) > 0.0
    assert np.nanmax(share) < 1.0
    # Shares of a side sum to 1.
    grouped = features["gold_share_pct"].groupby(
        [metrics["_gid"], metrics["side"]], sort=False
    ).sum()
    assert grouped.sub(1.0).abs().max() < 1e-9
    per_minute = features["gold_per_min"].to_numpy(dtype=float)
    assert np.isfinite(per_minute).all()
    assert math.isclose(
        float(per_minute[0]),
        float(metrics["totalgold"].iloc[0]) / (float(metrics["gamelength"].iloc[0]) / 60.0),
        rel_tol=1e-12,
    )
