from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.prepare_controlled_draft_swaps import (
    ControlledDraftSwapPreparationError,
    prepare_controlled_draft_swaps,
)
from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_constituents import (
    DRAFT_INFERENCE_COLUMNS,
)


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    features = pd.DataFrame({"game_uid": ["game-1"], "league": ["LCK"]})
    feature_path = tmp_path / "features.parquet"
    features.to_parquet(feature_path, index=False)
    rows = []
    for side in ("Blue", "Red"):
        for index, role in enumerate(("top", "jng", "mid", "bot", "sup")):
            rows.append(
                {
                    "game_uid": "game-1",
                    "date": pd.Timestamp("2026-08-17T12:00:00Z"),
                    "side": side,
                    "position": role,
                    "champion": f"{side}-champion-{index}",
                    "playername": f"{side}-player-{index}",
                    "teamname": f"{side}-team",
                    "league": "LCK",
                }
            )
    player_path = tmp_path / "players.parquet"
    pd.DataFrame(rows, columns=DRAFT_INFERENCE_COLUMNS).to_parquet(
        player_path, index=False
    )
    return feature_path, player_path


def test_preparer_writes_one_outcome_blind_role_swap(tmp_path: Path) -> None:
    features, players = _inputs(tmp_path)
    output = tmp_path / "swaps.json"
    swapped_players = tmp_path / "swapped-players.parquet"

    result = prepare_controlled_draft_swaps(
        feature_path=features,
        expected_feature_sha256=sha256_path(features),
        player_path=players,
        expected_player_sha256=sha256_path(players),
        output_path=output,
        swapped_player_output_path=swapped_players,
    )

    assert result["outcome_blind"] is True
    assert result["game_ids"] == ["game-1"]
    assert len(result["games"]["game-1"]) == 10
    blue_top = next(
        row
        for row in result["games"]["game-1"]
        if row["side"] == "Blue" and row["position"] == "top"
    )
    assert blue_top["champion"] == "Red-champion-0"
    assert len(result["receipt_sha256"]) == 64
    assert output.is_file()
    swapped_frame = pd.read_parquet(swapped_players)
    assert list(swapped_frame.columns) == list(DRAFT_INFERENCE_COLUMNS)
    assert len(swapped_frame) == 10
    assert sha256_path(swapped_players) == result["swapped_player_file_sha256"]


def test_preparer_rejects_extra_player_field(tmp_path: Path) -> None:
    features, players = _inputs(tmp_path)
    frame = pd.read_parquet(players)
    frame["result"] = 1
    frame.to_parquet(players, index=False)

    with pytest.raises(
        ControlledDraftSwapPreparationError,
        match="outcome-blind draft contract",
    ):
        prepare_controlled_draft_swaps(
            feature_path=features,
            expected_feature_sha256=sha256_path(features),
            player_path=players,
            expected_player_sha256=sha256_path(players),
            output_path=tmp_path / "swaps.json",
        )
