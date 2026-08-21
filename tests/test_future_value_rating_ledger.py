from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_kills.research.future_value_rating import bind_accepted_future_value_source
from lol_kills.research.future_value_rating_ledger import (
    CurrentRatingLedgerError,
    _artifact_digest,
    _canonical_json_bytes,
    build_fold_current_rating_feature_ledger,
    validate_fold_current_rating_feature_ledger,
)
import hashlib
import json


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
    source_frame_sha256: dict[str, str] | None = None,
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
        source_frame_sha256=source_frame_sha256,
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


def test_forged_source_frame_hashes_fail_closed() -> None:
    maps, players, teams, receipt = _source()
    with pytest.raises(CurrentRatingLedgerError, match="do not match"):
        _build(
            maps,
            players,
            teams,
            receipt,
            source_frame_sha256={
                "maps": "a" * 64,
                "players": "b" * 64,
                "teams": "c" * 64,
            },
        )


def test_receipt_validation_recomputes_source_frame_hashes(tmp_path: Path) -> None:
    maps, players, teams, receipt = _source()
    ledger, good_receipt = _build(maps, players, teams, receipt, tmp_path)
    bad_receipt = json.loads(json.dumps(good_receipt))
    bad_receipt["source_frame_sha256"]["maps"] = "a" * 64
    payload = dict(bad_receipt)
    payload.pop("receipt_sha256")
    bad_receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    with pytest.raises(CurrentRatingLedgerError, match="source frame hashes changed"):
        validate_fold_current_rating_feature_ledger(
            ledger,
            bad_receipt,
            source_receipt=receipt,
            train_game_ids=("g1", "g2"),
            validation_game_ids=("g3", "g4"),
            fit_window_end="2026-01-02T00:00:00Z",
            source_frames={"maps": maps, "players": players, "teams": teams},
        )


def test_no_series_source_uses_conservative_proxy_and_rejects_series_split() -> None:
    maps, players, teams, receipt = _source()
    maps = maps.drop(columns=["series_id"]).copy()
    players = players.copy()
    teams = teams.copy()
    maps.loc[maps["game_uid"].eq("g3"), ["blue_team", "red_team"]] = maps.loc[
        maps["game_uid"].eq("g1"), ["blue_team", "red_team"]
    ].to_numpy()[0]
    for side in ("Blue", "Red"):
        player_team_id = players.loc[
            players["game_uid"].eq("g1") & players["side"].eq(side), "teamid"
        ].iloc[0]
        team_name = players.loc[
            players["game_uid"].eq("g1") & players["side"].eq(side), "teamname"
        ].iloc[0]
        players.loc[
            players["game_uid"].eq("g3") & players["side"].eq(side),
            ["teamid", "teamname"],
        ] = [player_team_id, team_name]
        teams.loc[
            teams["game_uid"].eq("g3") & teams["side"].eq(side),
            ["teamid", "teamname"],
        ] = [player_team_id, team_name]
    with pytest.raises(CurrentRatingLedgerError, match="series overlap"):
        _build(maps, players, teams, receipt)


def test_validation_rechecks_series_disjointness_after_receipt_reseal(
    tmp_path: Path,
) -> None:
    maps, players, teams, receipt = _source()
    ledger, good_receipt = _build(maps, players, teams, receipt, tmp_path)
    bad_ledger = ledger.copy()
    g1_series = ledger.loc[ledger["game_id"].eq("g1"), "series_id"].iloc[0]
    bad_ledger.loc[bad_ledger["game_id"].eq("g4"), "series_id"] = g1_series
    bad_receipt = json.loads(json.dumps(good_receipt))
    bad_receipt["ledger_rows_sha256"] = _artifact_digest(
        bad_ledger,
        ("base_team_logit", "team_rating_diff_scaled", "base_player_logit", "player_rating_diff_scaled"),
    )
    payload = dict(bad_receipt)
    payload.pop("receipt_sha256")
    bad_receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(payload)
    ).hexdigest()
    with pytest.raises(CurrentRatingLedgerError, match="series overlap"):
        validate_fold_current_rating_feature_ledger(
            bad_ledger,
            bad_receipt,
            source_receipt=receipt,
            train_game_ids=("g1", "g2"),
            validation_game_ids=("g3", "g4"),
            fit_window_end="2026-01-02T00:00:00Z",
            source_frames={"maps": maps, "players": players, "teams": teams},
        )


def test_stable_player_id_preserves_state_across_display_rename() -> None:
    maps, players, teams, receipt = _source()
    renamed_a = players.copy()
    renamed_b = players.copy()
    source_player = players.loc[
        players["game_uid"].eq("g1")
        & players["side"].eq("Blue")
        & players["position"].eq("top"),
        "playerid",
    ].iloc[0]
    selector = (
        renamed_a["game_uid"].eq("g3")
        & renamed_a["side"].eq("Blue")
        & renamed_a["position"].eq("top")
    )
    renamed_a.loc[selector, "playerid"] = source_player
    renamed_b.loc[selector, "playerid"] = source_player
    renamed_a.loc[selector, "playername"] = "renamed-player"
    renamed_b.loc[selector, "playername"] = "same-player"
    ledger_a, _ = _build(maps, renamed_a, teams, receipt)
    ledger_b, _ = _build(maps, renamed_b, teams, receipt)
    columns = [
        "game_id",
        "base_player_logit",
        "player_rating_diff_scaled",
        "base_team_logit",
        "team_rating_diff_scaled",
    ]
    pd.testing.assert_frame_equal(
        ledger_a[columns].sort_values("game_id").reset_index(drop=True),
        ledger_b[columns].sort_values("game_id").reset_index(drop=True),
        check_exact=True,
    )


def test_display_alias_collision_fails_closed() -> None:
    maps, players, teams, receipt = _source()
    broken = players.copy()
    broken.loc[broken["game_uid"].eq("g3") & broken["position"].eq("top"), "playername"] = (
        players.loc[players["game_uid"].eq("g1") & players["position"].eq("top"), "playername"].iloc[0]
    )
    with pytest.raises(CurrentRatingLedgerError, match="alias collision"):
        _build(maps, broken, teams, receipt)
