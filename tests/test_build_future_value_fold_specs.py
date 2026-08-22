from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_future_value_fold_specs import FoldSpecError, build_fold_specs
from lol_kills.research.future_value_rating import FutureValueSourceError
from lol_kills.v2.tierlists.accepted_census import identity_sha256
from tests.test_future_value_leaguepedia_series import (
    _source_receipt,
    _write_crosswalk,
)


def _reseal_source(source: dict[str, object]) -> dict[str, object]:
    body = dict(source)
    body.pop("receipt_sha256", None)
    source["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            body,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return source


def _maps(game_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_uid": game_id,
                "date": pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(hours=index),
                "y_blue_win": index % 2,
                "league": "LEC",
                "tournament": f"event-{index // 2}",
                "blue_team_key": "blue",
                "red_team_key": "red",
            }
            for index, game_id in enumerate(game_ids)
        ]
    )


def _assignments(game_ids: list[str]) -> list[dict[str, object]]:
    return [
        {
            "oe_game_id": game_id,
            "series_id": f"series-{index // 2}",
            "normalized_team_set": ["team 0", "team 1"],
            "outcome_used": False,
        }
        for index, game_id in enumerate(game_ids)
    ]


def test_fold_specs_bind_mixed_partition_and_reject_receipt_mutation(
    tmp_path: Path,
) -> None:
    game_ids = [f"g{index:02d}" for index in range(1, 41)]
    maps = _maps(game_ids)
    source = _source_receipt(game_ids)
    assignments = _assignments(game_ids)
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
    assert result["series_partition"]["authoritative"] is True
    assert result["series_partition"]["map_count"] == 40
    assert result["series_partition"]["retained_proxy_game_count"] == 0
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


def test_fold_specs_exclude_declared_raw_source_extras(tmp_path: Path) -> None:
    game_ids = [f"g{index:02d}" for index in range(1, 41)]
    source = _source_receipt(game_ids)
    source["source_extra_game_ids"] = {"maps": ["raw-extra"]}
    _reseal_source(source)
    maps = pd.concat(
        [_maps(game_ids), _maps(["raw-extra", "raw-extra"])],
        ignore_index=True,
    )
    crosswalk_path, receipt_path, receipt_file_sha = _write_crosswalk(
        tmp_path, source, _assignments(game_ids)
    )
    result = build_fold_specs(
        maps=maps,
        source_receipt=source,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=receipt_path,
        expected_crosswalk_receipt_file_sha256=receipt_file_sha,
        n_folds=2,
    )
    assert result["series_partition"]["map_count"] == len(game_ids)
    assert result["series_partition"]["authoritative"] is True


def test_fold_specs_ignore_accepted_unmatched_map_outside_eligible_scope(
    tmp_path: Path,
) -> None:
    accepted_ids = [f"g{index:02d}" for index in range(1, 42)]
    eligible_ids = accepted_ids[:-1]
    source = _source_receipt(accepted_ids)
    source["model_eligible_game_count"] = len(eligible_ids)
    source["model_eligible_game_ids"] = sorted(eligible_ids)
    source["model_eligible_identity_sha256"] = identity_sha256(eligible_ids)
    _reseal_source(source)
    crosswalk_path, receipt_path, receipt_file_sha = _write_crosswalk(
        tmp_path, source, _assignments(eligible_ids)
    )
    result = build_fold_specs(
        maps=_maps(accepted_ids),
        source_receipt=source,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=receipt_path,
        expected_crosswalk_receipt_file_sha256=receipt_file_sha,
        n_folds=1,
    )
    partition = result["series_partition"]
    assert partition["map_count"] == len(eligible_ids)
    assert partition["authoritative"] is True
    assert partition["retained_proxy_game_count"] == 0


def test_fold_specs_keep_eligible_proxy_authority_blocker(tmp_path: Path) -> None:
    game_ids = [f"g{index:02d}" for index in range(1, 42)]
    source = _source_receipt(game_ids)
    crosswalk_path, receipt_path, receipt_file_sha = _write_crosswalk(
        tmp_path, source, _assignments(game_ids[:-1])
    )
    result = build_fold_specs(
        maps=_maps(game_ids),
        source_receipt=source,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=receipt_path,
        expected_crosswalk_receipt_file_sha256=receipt_file_sha,
        n_folds=1,
    )
    partition = result["series_partition"]
    assert partition["authoritative"] is False
    assert partition["retained_proxy_game_count"] == 1
    assert partition["authoritative_series_blocker"] == (
        "authoritative_series_id_missing_proxy_cluster_used"
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
