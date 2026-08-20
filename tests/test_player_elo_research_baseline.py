from __future__ import annotations

import pandas as pd
import pytest

from lol_kills.ratings.player_elo import (
    PlayerEloConfig,
    SequentialPlayerEloBaselineError,
    _sequential_baseline_identity,
    build_sequential_player_elo_baseline,
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    game_ids = [f"oe:game:{index}" for index in range(4)]
    maps = pd.DataFrame(
        [
            {
                "game_uid": game_id,
                "date": pd.Timestamp("2026-01-01") + pd.Timedelta(days=index),
                "blue_team": f"blue-{index % 2}",
                "red_team": f"red-{index % 2}",
                "y_blue_win": index % 2,
                "blue_golddiffat15": 100 * index,
                "length_min": 30.0,
                "league": "LCK",
                "tournament": "Spring",
            }
            for index, game_id in enumerate(game_ids)
        ]
    )
    roles = ("top", "jng", "mid", "bot", "sup")
    player_rows: list[dict[str, object]] = []
    for index, game_id in enumerate(game_ids):
        date = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        for side, team in (("Blue", f"blue-{index % 2}"), ("Red", f"red-{index % 2}")):
            for role_index, role in enumerate(roles):
                player_rows.append(
                    {
                        "game_uid": game_id,
                        "date": date,
                        "side": side,
                        "position": role,
                        "playername": f"{side}-{role_index}",
                        "playerid": f"oe:player:{side}-{role_index}",
                        "teamid": f"oe:team:{team}",
                        "league": "LCK",
                        "tournament": "Spring",
                        "competition_tier": "tier1",
                        "champion": f"champion-{role_index}",
                        "cspm": 5.0 + role_index,
                        "dpm": 100.0 + role_index,
                        "totalgold": 1000.0 + role_index,
                        "gamelength": 1800.0,
                        "result": index % 2,
                    }
                )
    players = pd.DataFrame(player_rows)
    receipt = {
        "receipt_sha256": "a" * 64,
        "model_eligible_game_ids": game_ids,
        "model_eligible_identity_sha256": _sequential_baseline_identity(game_ids),
    }
    return maps, players, receipt


def _build(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    receipt: dict[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    return build_sequential_player_elo_baseline(
        maps,
        players,
        train_game_ids=["oe:game:0", "oe:game:1"],
        validation_game_ids=["oe:game:2", "oe:game:3"],
        strict_cutoff="2026-01-03T00:00:00Z",
        source_receipt=receipt,
        cfg=PlayerEloConfig(attribution_enabled=False),
    )


def test_research_baseline_returns_one_finite_pre_map_row_per_validation_game() -> None:
    maps, players, receipt = _fixture()
    output, run_receipt = _build(maps, players, receipt)

    assert output["game_uid"].tolist() == ["oe:game:2", "oe:game:3"]
    assert output["p_player_elo"].notna().all()
    assert output["p_player_elo"].between(0.0, 1.0).all()
    assert run_receipt["output_rows"] == 2
    assert run_receipt["missing_game_ids"] == []
    assert run_receipt["strict_cutoff"] == "2026-01-03T00:00:00Z"
    assert run_receipt["state"] == "fresh_in_memory_replay"
    assert run_receipt["writes_production_artifacts"] is False
    assert run_receipt["train_game_identity_sha256"] == _sequential_baseline_identity(
        ["oe:game:0", "oe:game:1"]
    )
    assert run_receipt["validation_game_identity_sha256"] == _sequential_baseline_identity(
        ["oe:game:2", "oe:game:3"]
    )
    assert len(str(run_receipt["implementation_digest"])) == 64
    assert len(str(run_receipt["output_sha256"])) == 64


def test_validation_outcomes_and_final_player_values_are_masked_before_replay() -> None:
    maps, players, receipt = _fixture()
    baseline, baseline_receipt = _build(maps, players, receipt)

    changed_maps = maps.copy()
    validation_mask = changed_maps["game_uid"].isin(["oe:game:2", "oe:game:3"])
    changed_maps.loc[
        validation_mask,
        ["y_blue_win", "blue_golddiffat15", "length_min"],
    ] = [1, 999999.0, 1.0]
    changed_players = players.copy()
    player_validation_mask = changed_players["game_uid"].isin(
        ["oe:game:2", "oe:game:3"]
    )
    changed_players.loc[
        player_validation_mask,
        ["cspm", "dpm", "totalgold", "gamelength", "result"],
    ] = [999.0, 999999.0, 999999.0, 1.0, 0]
    changed, changed_receipt = _build(changed_maps, changed_players, receipt)

    pd.testing.assert_frame_equal(
        baseline[["game_uid", "p_player_elo"]],
        changed[["game_uid", "p_player_elo"]],
        check_exact=True,
    )
    assert baseline_receipt["output_sha256"] == changed_receipt["output_sha256"]
    assert "y_blue_win" in baseline_receipt["masked_map_columns"]
    assert "blue_golddiffat15" in baseline_receipt["masked_map_columns"]
    assert "cspm" in baseline_receipt["masked_player_columns"]
    assert "result" in baseline_receipt["masked_player_columns"]


def test_replay_rejects_overlap_missing_outcomes_and_non_strict_cutoff() -> None:
    maps, players, receipt = _fixture()
    with pytest.raises(SequentialPlayerEloBaselineError, match="overlap"):
        build_sequential_player_elo_baseline(
            maps,
            players,
            train_game_ids=["oe:game:0"],
            validation_game_ids=["oe:game:0"],
            strict_cutoff="2026-01-03T00:00:00Z",
            source_receipt=receipt,
        )

    missing_maps = maps.loc[maps["game_uid"] != "oe:game:3"].copy()
    with pytest.raises(SequentialPlayerEloBaselineError, match="requested maps are missing"):
        _build(missing_maps, players, receipt)

    bad_maps = maps.copy()
    bad_maps.loc[bad_maps["game_uid"] == "oe:game:0", "y_blue_win"] = pd.NA
    with pytest.raises(SequentialPlayerEloBaselineError, match="training maps"):
        _build(bad_maps, players, receipt)

    with pytest.raises(SequentialPlayerEloBaselineError, match="strictly before"):
        build_sequential_player_elo_baseline(
            maps,
            players,
            train_game_ids=["oe:game:0", "oe:game:1"],
            validation_game_ids=["oe:game:2", "oe:game:3"],
            strict_cutoff="2025-12-31T00:00:00Z",
            source_receipt=receipt,
        )
