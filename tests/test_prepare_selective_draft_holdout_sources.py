from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.prepare_selective_draft_holdout_sources import (
    SelectiveDraftHoldoutSourceError,
    prepare_holdout_sources,
)
from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import canonical_sha256


def _sources(tmp_path: Path, *, complete: bool = True) -> tuple[Path, Path]:
    matrix = pd.DataFrame(
        [
            {
                "game_uid": "game-1",
                "date": "2026-08-16T12:00:00Z",
                "league": "LEC",
                "series_id": "series-1",
                "safe_atom": 0.25,
                "y": 1,
                "target_gold_diff_10": 500.0,
            }
        ]
    )
    roles = ["top", "jng", "mid", "bot", "sup"]
    players = pd.DataFrame(
        [
            {
                "game_uid": "game-1",
                "gameid": "game-1",
                "date": "2026-08-16T12:00:00Z",
                "side": side,
                "position": role,
                "champion": f"{side}-{role}",
                "playername": f"player-{side}-{role}",
                "teamname": f"team-{side}",
                "league": "LEC",
                "result": 1 if side == "Blue" else 0,
                "kills": 99,
                "goldat15": 99999,
            }
            for side in ("Blue", "Red")
            for role in roles
            if complete or not (side == "Red" and role == "sup")
        ]
    )
    matrix_path = tmp_path / "matrix.parquet"
    player_path = tmp_path / "players.parquet"
    matrix.to_parquet(matrix_path, index=False)
    players.to_parquet(player_path, index=False)
    return matrix_path, player_path


def test_prepare_sources_removes_results_and_targets(tmp_path: Path) -> None:
    matrix_path, player_path = _sources(tmp_path)
    features = tmp_path / "features.parquet"
    players = tmp_path / "draft.parquet"
    receipt_path = tmp_path / "receipt.json"

    receipt = prepare_holdout_sources(
        feature_matrix_path=matrix_path,
        players_path=player_path,
        batch_start="2026-08-16T00:00:00Z",
        batch_end_exclusive="2026-08-17T00:00:00Z",
        feature_output_path=features,
        player_output_path=players,
        receipt_output_path=receipt_path,
    )

    feature_frame = pd.read_parquet(features)
    player_frame = pd.read_parquet(players)
    assert feature_frame.columns.tolist() == [
        "game_uid",
        "date",
        "league",
        "series_id",
        "safe_atom",
    ]
    assert set(player_frame.columns) == {
        "game_uid",
        "date",
        "side",
        "position",
        "champion",
        "playername",
        "teamname",
        "league",
    }
    assert len(player_frame) == 10
    assert receipt["evaluation_player_rows"] == 10
    assert receipt["outcome_blind"] is True
    assert receipt["output_sha256"] == {
        "features": sha256_path(features),
        "players": sha256_path(players),
    }
    stored = json.loads(receipt_path.read_text(encoding="utf-8"))
    unsigned = {key: value for key, value in stored.items() if key != "receipt_sha256"}
    assert stored["receipt_sha256"] == canonical_sha256(unsigned)


def test_prepare_sources_rejects_incomplete_draft(tmp_path: Path) -> None:
    matrix_path, player_path = _sources(tmp_path, complete=False)

    with pytest.raises(SelectiveDraftHoldoutSourceError, match="ten unique slots"):
        prepare_holdout_sources(
            feature_matrix_path=matrix_path,
            players_path=player_path,
            batch_start="2026-08-16T00:00:00Z",
            batch_end_exclusive="2026-08-17T00:00:00Z",
            feature_output_path=tmp_path / "features.parquet",
            player_output_path=tmp_path / "draft.parquet",
            receipt_output_path=tmp_path / "receipt.json",
        )
