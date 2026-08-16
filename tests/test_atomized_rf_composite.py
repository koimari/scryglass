from __future__ import annotations

import pandas as pd
import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.research.atomized_rf_composite import (
    FEATURE_AVAILABILITY_COLUMNS,
    MODEL_COLUMNS,
    AtomizedResearchError,
    RFConfig,
    RunningStat,
    _calibration_outer_audit,
    _expanding_series_folds,
    _cluster_bootstrap_differences,
    _equal_weight_team_forecast,
    _locked_rating_authority,
    _matched_comparison_config,
    _momentum_features,
    _phase_curve_features,
    _shrunk_metric_mean,
    _unique_player_map_support,
    _validate_no_current_state_features,
    _write_json,
    exact_mechanic_keys,
    feature_group_coverage_report,
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


def test_unique_player_map_support_deduplicates_metric_families() -> None:
    state = {
        ("player-1", "Galio"): {"map-1", "map-2"},
        ("player-2", "Ahri"): {"map-3"},
    }
    assert _unique_player_map_support(
        state,
        [("player-1", "Galio"), ("player-1", "Galio"), ("player-2", "Ahri")],
    ) == 3


def test_phase_forecast_weights_each_current_player_once() -> None:
    keys = [(f"player-{index}", "champion", 10, "gold") for index in range(5)]
    state = {
        keys[0]: RunningStat(total=10_000.0, count=100),
        keys[1]: RunningStat(total=20.0, count=1),
        keys[2]: RunningStat(total=60.0, count=2),
        keys[3]: RunningStat(total=120.0, count=3),
        keys[4]: RunningStat(total=200.0, count=4),
    }
    total, support, coverage, missing = _equal_weight_team_forecast(state, keys)
    assert total == pytest.approx(240.0)
    assert support == 1
    assert coverage == 1.0
    assert missing == 0


def test_phase_curve_is_side_swap_antisymmetric() -> None:
    blue = _phase_curve_features(
        [100.0, 200.0, -300.0, -900.0],
        [50.0, 100.0, -200.0, -500.0],
        available=True,
    )
    red = _phase_curve_features(
        [-100.0, -200.0, 300.0, 900.0],
        [-50.0, -100.0, 200.0, 500.0],
        available=True,
    )
    signed = [
        key
        for key in blue
        if key.startswith("forecast_") and key not in {
            "forecast_curve_available",
            "forecast_curve_missing",
        }
    ]
    for key in signed:
        assert red[key] == pytest.approx(-blue[key])


def test_rating_and_momentum_missingness_is_explicit() -> None:
    rating = _locked_rating_authority(
        {
            "base_team_logit": 0.2,
            "team_rating_diff_scaled": 0.1,
            "base_player_logit": 0.3,
            "player_rating_diff_scaled": 0.2,
            "player_lineup_complete": 1.0,
        },
        identity_recovered=True,
    )
    assert rating["player_rating_available"] == 0.0
    assert rating["player_rating_missing"] == 1.0
    assert rating["base_player_logit"] == 0.0

    momentum = _momentum_features(
        {}, {}, "blue-team", "red-team", [f"b-{i}" for i in range(5)], [f"r-{i}" for i in range(5)]
    )
    assert momentum["team_momentum_missing"] == 1.0
    assert momentum["player_momentum_missing"] == 1.0
    assert momentum["team_momentum_points_diff"] == 0.0


def test_feature_group_coverage_gates_each_split_and_league() -> None:
    rows = []
    for index in range(40):
        rows.append(
            {
                "date": pd.Timestamp("2026-04-01", tz="UTC")
                if index < 20
                else pd.Timestamp("2026-06-01", tz="UTC"),
                "league": "LEC" if index < 20 else "LCK",
                FEATURE_AVAILABILITY_COLUMNS["team_rating"]: 1.0
                if index < 30
                else 0.0,
            }
        )
    report = feature_group_coverage_report(
        pd.DataFrame(rows), thresholds={"team_rating": 0.8}
    )
    failures = {
        (row["dimension"], row["value"]) for row in report["failures"]
    }
    assert ("split", "validation") in failures
    assert ("league", "LCK") in failures


def test_calibration_requires_brier_and_log_loss_improvement_in_every_fold() -> None:
    target = np.tile(np.array([0, 1]), 100)
    underconfident = np.where(target == 1, 0.60, 0.40)
    accepted = _calibration_outer_audit(
        [underconfident, underconfident, underconfident],
        [target, target, target],
    )
    assert accepted["accepted"] is True

    reversed_probability = 1.0 - underconfident
    rejected = _calibration_outer_audit(
        [underconfident, reversed_probability], [target, target]
    )
    assert rejected["accepted"] is False


def test_ablation_comparison_keeps_exact_frozen_learner() -> None:
    config = RFConfig(
        n_estimators=600,
        max_depth=None,
        min_samples_leaf=20,
        max_features=0.25,
        class_weight=None,
        bootstrap=True,
        max_samples=None,
    )
    comparison = _matched_comparison_config(config)
    assert comparison == config
    assert comparison.n_estimators == 600
