"""Focused tests for nested feature-ledger bundle construction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from benchmarks.future_value_four_variant_bundle import (
    FourVariantBundleError,
    _derive_inner_fold_spec,
    _prepare_inner_output_root,
    build_bundle,
)


def _model_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"g{index}" for index in range(1, 7)],
            "date": pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC"),
            "series_id": [f"s{index}" for index in range(1, 7)],
        }
    )


def test_nested_fold_covers_outer_train_and_excludes_outer_validation() -> None:
    spec = _derive_inner_fold_spec(
        _model_frame(),
        outer_fold=2,
        outer_train_ids=("g1", "g2", "g3", "g4"),
        outer_validation_ids=("g5", "g6"),
    )

    inner_train = set(spec["train_game_ids"])
    inner_validation = set(spec["validation_game_ids"])
    assert inner_train | inner_validation == {"g1", "g2", "g3", "g4"}
    assert not (inner_train | inner_validation) & {"g5", "g6"}
    assert spec["outer_fold"] == 2
    assert spec["inner_fold"] == 1
    assert spec["fit_window_end"] == spec["validation_start"]


def test_nested_fold_rejects_incomplete_outer_training_census() -> None:
    with pytest.raises(FourVariantBundleError, match="outer training census"):
        _derive_inner_fold_spec(
            _model_frame(),
            outer_fold=1,
            outer_train_ids=("g1", "g2", "missing"),
            outer_validation_ids=("g5", "g6"),
        )


def test_nested_output_root_fails_closed_when_reused(tmp_path: Path) -> None:
    root = tmp_path / "nested-inner"
    root.mkdir()
    _prepare_inner_output_root(root)
    (root / "stale.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FourVariantBundleError, match="must be empty"):
        _prepare_inner_output_root(root)


def test_bundle_requires_complete_crosswalk_binding(tmp_path: Path) -> None:
    source_receipt = tmp_path / "source-receipt.json"
    source_receipt.write_text("{}", encoding="utf-8")

    with pytest.raises(
        FourVariantBundleError, match="crosswalk inputs must be supplied together"
    ):
        build_bundle(
            source_root=tmp_path,
            source_receipt_path=source_receipt,
            folds_root=tmp_path,
            crosswalk_path=tmp_path / "crosswalk.json",
        )
