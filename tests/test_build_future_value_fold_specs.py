from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_future_value_fold_specs import FoldSpecError, build_fold_specs
from lol_kills.research.future_value_rating import FutureValueSourceError
from tests.test_future_value_leaguepedia_series import (
    _source_receipt,
    _write_crosswalk,
)


def test_fold_specs_bind_mixed_partition_and_reject_receipt_mutation(
    tmp_path: Path,
) -> None:
    game_ids = [f"g{index:02d}" for index in range(1, 41)]
    maps = pd.DataFrame(
        [
            {
                "game_uid": game_id,
                "date": pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(hours=index),
                "y_blue_win": index % 2,
                "league": "LEC",
                "tournament": f"event-{index // 4}",
                "blue_team_key": f"team-{index % 2}",
                "red_team_key": f"team-{1 - (index % 2)}",
            }
            for index, game_id in enumerate(game_ids)
        ]
    )
    source = _source_receipt(game_ids)
    assignments = [
        {
            "oe_game_id": game_id,
            "series_id": f"series-{index // 2}",
            "normalized_team_set": ["team 0", "team 1"],
            "outcome_used": False,
        }
        for index, game_id in enumerate(game_ids)
    ]
    crosswalk_path, receipt_path, receipt_file_sha = _write_crosswalk(
        tmp_path, source, assignments
    )
    result = build_fold_specs(
        maps=maps,
        source_receipt=source,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=receipt_path,
        expected_crosswalk_receipt_file_sha256=receipt_file_sha,
        n_folds=2,
    )
    assert len(result["folds"]) == 2
    assert result["series_partition"]["authoritative"] is False
    assert result["source"]["model_eligible_game_count"] == 40
    with pytest.raises(FutureValueSourceError, match="receipt file changed"):
        build_fold_specs(
            maps=maps,
            source_receipt=source,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=receipt_path,
            expected_crosswalk_receipt_file_sha256="0" * 64,
            n_folds=2,
        )


def test_fold_specs_reject_empty_model_census(tmp_path: Path) -> None:
    source = _source_receipt(["g1"])
    maps = pd.DataFrame(
        [
            {
                "game_uid": "other",
                "date": "2026-01-01T00:00:00Z",
                "y_blue_win": 1,
                "blue_team_key": "a",
                "red_team_key": "b",
            }
        ]
    )
    with pytest.raises((FoldSpecError, FutureValueSourceError)):
        build_fold_specs(
            maps=maps,
            source_receipt=source,
            crosswalk_path=tmp_path / "missing.json",
            crosswalk_receipt_path=tmp_path / "missing-receipt.json",
            expected_crosswalk_receipt_file_sha256="0" * 64,
            n_folds=1,
        )
