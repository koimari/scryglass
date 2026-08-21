from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_kills.research.future_value_rating import bind_accepted_future_value_source
from lol_kills.research.future_value_rating_ledger import (
    CurrentRatingLedgerError,
    build_fold_current_rating_feature_ledger,
    validate_fold_current_rating_feature_ledger,
)


GAME_IDS = ("g1", "g2", "g3", "g4")
ROLES = ("top", "jng", "mid", "bot", "sup")


def _source_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dates = {
        "g1": "2026-01-01T00:00:00Z",
        "g2": "2026-01-01T00:00:00Z",
        "g3": "2026-01-02T00:00:00Z",
        "g4": "2026-01-02T00:00:00Z",
    }
    maps: list[dict[str, object]] = []
    players: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    for index, gid in enumerate(GAME_IDS):
        blue = f"Blue {gid}"
        red = f"Red {gid}"
        maps.append(
            {
                "game_uid": gid,
                "date": dates[gid],
                "blue_team": blue,
                "red_team": red,
                "league": "LCK",
                "tournament": "fixture",
                "series_id": f"series-{gid}",
                "y_blue_win": float(index % 2),
                "blue_golddiffat15": float(100 * index),
                "blue_golddiffat10": float(50 * index),
                "length_min": 30.0,
                "final_map_metric": float(index + 1),
            }
        )
        for side, team, prefix in (("Blue", blue, "b"), ("Red", red, "r")):
            teams.append(
                {
                    "game_uid": gid,
                    "date": dates[gid],
                    "side": side,
                    "teamid": f"oe:team:{prefix}-{gid}",
                    "teamname": team,
                }
            )
            for role in ROLES:
                players.append(
                    {
                        "game_uid": gid,
                        "date": dates[gid],
                        "side": side,
                        "position": role,
                        "playername": f"{prefix}-{role}-{gid}",
                        "playerid": f"oe:player:{prefix}-{role}-{gid}",
                        "teamid": f"oe:team:{prefix}-{gid}",
                        "teamname": team,
                        "champion": f"Champion{prefix}{role}",
                        "dpm": float(index + 1),
                    }
                )
    return pd.DataFrame(maps), pd.DataFrame(players), pd.DataFrame(teams)


def _source() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    maps, players, teams = _source_frames()
    source = bind_accepted_future_value_source(
        maps,
        players,
        teams,
        census={
            "game_ids": list(GAME_IDS),
            "game_count": len(GAME_IDS),
            "source_identity_sha256": __import__(
                "lol_kills.v2.tierlists.accepted_census",
                fromlist=["identity_sha256"],
            ).identity_sha256(GAME_IDS),
        },
        source_as_of="2026-01-03T00:00:00Z",
        source_files={
            label: {"locator": f"fixture/{label}", "bytes": 1, "sha256": "0" * 64}
            for label in ("maps", "players", "teams", "accepted_census")
        },
    )
    return source.maps, source.players, source.teams, source.receipt


def _build(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    teams: pd.DataFrame,
    receipt: dict[str, object],
    destination: Path | None = None,
):
    return build_fold_current_rating_feature_ledger(
        maps,
        players,
        teams,
        source_receipt=receipt,
        train_game_ids=("g1", "g2"),
        validation_game_ids=("g3", "g4"),
        fit_window_end="2026-01-02T00:00:00Z",
        destination=destination,
    )


def test_fold_replay_batches_equal_timestamp_rows_and_emits_four_features() -> None:
    maps, players, teams, receipt = _source()
    ledger, artifact = _build(maps, players, teams, receipt)
    assert len(ledger) == 4
    assert set(ledger["game_id"]) == set(GAME_IDS)
    expected = {
        "base_team_logit",
        "team_rating_diff_scaled",
        "base_player_logit",
        "player_rating_diff_scaled",
    }
    assert expected.issubset(ledger.columns)
    assert np.isfinite(ledger[list(expected)].to_numpy(dtype=float)).all()
    for feature in expected:
        assert ledger.loc[ledger["game_id"].isin(("g1", "g2")), feature].nunique() == 1
        assert ledger.loc[ledger["game_id"].isin(("g3", "g4")), feature].nunique() == 1
    assert artifact["same_timestamp_policy"] == "score_full_utc_timestamp_batch_before_training_updates"
    assert artifact["authority"]["public_player_rating"] is False
    assert artifact["authority"]["public_team_rating"] is False


def test_validation_outcomes_and_final_metrics_cannot_change_validation_features() -> None:
    maps, players, teams, receipt = _source()
    original, _ = _build(maps, players, teams, receipt)
    mutated_maps = maps.copy()
    validation = mutated_maps["game_uid"].isin(("g3", "g4"))
    mutated_maps.loc[validation, "y_blue_win"] = 1.0 - mutated_maps.loc[validation, "y_blue_win"]
    mutated_maps.loc[validation, "blue_golddiffat15"] = 999999.0
    mutated_maps.loc[validation, "final_map_metric"] = 999999.0
    mutated_players = players.copy()
    mutated_players.loc[mutated_players["game_uid"].isin(("g3", "g4")), "dpm"] = 999999.0
    changed, _ = _build(mutated_maps, mutated_players, teams, receipt)
    cols = ["base_team_logit", "team_rating_diff_scaled", "base_player_logit", "player_rating_diff_scaled"]
    left = original.loc[original["game_id"].isin(("g3", "g4")), ["game_id", *cols]].sort_values("game_id")
    right = changed.loc[changed["game_id"].isin(("g3", "g4")), ["game_id", *cols]].sort_values("game_id")
    pd.testing.assert_frame_equal(left.reset_index(drop=True), right.reset_index(drop=True), check_exact=True)


def test_missing_player_row_fails_closed() -> None:
    maps, players, teams, receipt = _source()
    broken = players.loc[~((players["game_uid"] == "g4") & (players["side"] == "Red") & (players["position"] == "sup"))].copy()
    with pytest.raises(CurrentRatingLedgerError, match="exactly ten|closure"):
        _build(maps, broken, teams, receipt)


def test_durable_artifact_and_receipt_are_byte_bound(tmp_path: Path) -> None:
    maps, players, teams, receipt = _source()
    ledger, artifact = _build(maps, players, teams, receipt, tmp_path)
    artifact_path = Path(str(artifact["artifact"]["path"]))
    assert artifact_path.is_file()
    assert artifact["artifact"]["bytes"] == artifact_path.stat().st_size
    validate_fold_current_rating_feature_ledger(
        ledger,
        artifact,
        source_receipt=receipt,
        train_game_ids=("g1", "g2"),
        validation_game_ids=("g3", "g4"),
        fit_window_end="2026-01-02T00:00:00Z",
    )
    artifact_path.write_bytes(artifact_path.read_bytes() + b"x")
    with pytest.raises(CurrentRatingLedgerError, match="artifact bytes"):
        validate_fold_current_rating_feature_ledger(
            ledger,
            artifact,
            source_receipt=receipt,
            train_game_ids=("g1", "g2"),
            validation_game_ids=("g3", "g4"),
            fit_window_end="2026-01-02T00:00:00Z",
        )
