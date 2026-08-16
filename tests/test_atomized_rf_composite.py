from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.research.atomized_rf_composite import (
    MODEL_COLUMNS,
    AtomizedResearchError,
    RunningStat,
    _expanding_series_folds,
    _cluster_bootstrap_differences,
    _shrunk_metric_mean,
    _validate_no_current_state_features,
    _write_json,
    exact_mechanic_keys,
    normalize_source_patch,
)


def test_receipt_writer_serializes_numpy_scalars(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    _write_json(path, {"rows": np.int64(1664)})
    assert '"rows": 1664' in path.read_text(encoding="utf-8")


def test_float_patch_token_uses_date_to_distinguish_16_1_and_16_10() -> None:
    assert normalize_source_patch("16.1", "2026-01-20T00:00:00Z") == "16.1"
    assert normalize_source_patch("16.1", "2026-05-20T00:00:00Z") == "16.10"
    assert normalize_source_patch("16.15", "2026-08-01T00:00:00Z") == "16.15"


def test_model_columns_exclude_current_state_and_targets() -> None:
    _validate_no_current_state_features(MODEL_COLUMNS)
    assert not any(column.startswith("target_") for column in MODEL_COLUMNS)


def test_expanding_series_folds_are_forward_only() -> None:
    rows = []
    for series in range(40):
        for game in range(10):
            rows.append(
                {
                    "series_id": f"series-{series}",
                    "date": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(days=series, minutes=game),
                    "y": (series + game) % 2,
                }
            )
    frame = pd.DataFrame(rows)
    folds = _expanding_series_folds(frame)
    assert len(folds) >= 2
    for train_index, validation_index, audit in folds:
        assert frame.iloc[train_index]["date"].max() < frame.iloc[validation_index]["date"].min()
        assert set(frame.iloc[train_index]["series_id"]).isdisjoint(
            set(frame.iloc[validation_index]["series_id"])
        )
        assert audit["whole_series"] is True


def test_shrinkage_uses_only_supplied_prior_state() -> None:
    state = {("player", "champion", "gold_diff_10"): RunningStat(total=400.0, count=2)}
    global_state = {"gold_diff_10": RunningStat(total=100.0, count=2)}
    value, support, missing = _shrunk_metric_mean(
        state,
        global_state,
        [("player", "champion")],
        "gold_diff_10",
    )
    assert value == pytest.approx((400.0 + 5.0 * 50.0) / 7.0)
    assert support == 2
    assert missing == 0


def test_exact_mechanic_keys_keep_raw_fields() -> None:
    keys = exact_mechanic_keys(
        [
            {
                "atom_id": "damage.packet",
                "behavior": "SpellQ",
                "trigger": "on_cast",
                "target_policy": "enemy",
                "parameters": {"cooldown_seconds": 8.0, "damage_type": "magic"},
                "relations": ["damage.resistance"],
                "states": [{"state": "active"}],
            }
        ]
    )
    assert any("parameter=cooldown_seconds" in key for key in keys)
    assert any("parameter=damage_type|value=magic" in key for key in keys)


def test_broad_mechanics_label_fails_closed() -> None:
    with pytest.raises(AtomizedResearchError, match="broad mechanic labels"):
        exact_mechanic_keys(
            [
                {
                    "atom_id": "teamfight",
                    "behavior": "summary",
                    "trigger": "on_cast",
                    "target_policy": "enemy",
                    "parameters": {},
                }
            ]
        )


def test_numpy_cluster_bootstrap_matches_dataframe_reference() -> None:
    frame = pd.DataFrame(
        {
            "series_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "y": [0, 1, 0, 1, 1, 0, 1, 0],
        }
    )
    candidate = np.array([0.2, 0.7, 0.3, 0.8, 0.6, 0.4, 0.75, 0.25])
    baseline = np.array([0.3, 0.6, 0.4, 0.7, 0.55, 0.45, 0.65, 0.35])
    actual = _cluster_bootstrap_differences(
        frame, candidate, baseline, repetitions=25
    )

    work = frame.assign(_candidate=candidate, _baseline=baseline)
    clusters = [group for _, group in work.groupby("series_id", sort=True)]
    rng = np.random.default_rng(461)
    values = {"auc": [], "brier": [], "log_loss": []}
    for _ in range(25):
        sample = pd.concat(
            [clusters[index] for index in rng.integers(0, len(clusters), len(clusters))],
            ignore_index=True,
        )
        values["auc"].append(
            roc_auc_score(sample["y"], sample["_candidate"])
            - roc_auc_score(sample["y"], sample["_baseline"])
        )
        values["brier"].append(
            brier_score_loss(sample["y"], sample["_candidate"])
            - brier_score_loss(sample["y"], sample["_baseline"])
        )
        values["log_loss"].append(
            log_loss(sample["y"], sample["_candidate"], labels=[0, 1])
            - log_loss(sample["y"], sample["_baseline"], labels=[0, 1])
        )
    for metric, samples in values.items():
        assert actual[metric]["median"] == pytest.approx(np.median(samples))
        assert actual[metric]["lower_95"] == pytest.approx(np.quantile(samples, 0.025))
        assert actual[metric]["upper_95"] == pytest.approx(np.quantile(samples, 0.975))
