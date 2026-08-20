from __future__ import annotations

import pandas as pd
import pytest

from lol_kills.research.future_phase_curve import (
    FuturePhaseCurveError,
    chronological_folds,
    evaluate_phase_curve,
    fit_phase_curve,
    phase_curve_measures,
    prepare_phase_frame,
    score_phase_curve,
    side_swap_frame,
    side_swap_invariance_report,
    strict_prior_final_history,
)
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


def _receipt(ids: list[str]) -> dict[str, object]:
    ids = list(canonical_game_ids(ids))
    return {
        "source_as_of": "2026-08-20T23:59:59Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": ids,
    }


def _phase_frame(rows: int = 12) -> pd.DataFrame:
    values: list[dict[str, object]] = []
    for index in range(rows):
        game = f"oe-api:{index + 1}"
        date = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=index)
        value: dict[str, object] = {
            "game_uid": game,
            "date": date,
            "region": "LCK" if index % 2 else "LEC",
            "patch": "16.1" if index < rows // 2 else "16.2",
            "series_id": f"series-{index // 2}",
            "prior_form_gold_diff": float(index - 5),
            "prior_form_gold_diff_missing": index == 3,
        }
        for phase in (10, 15, 20, 25):
            value[f"gold_diff_{phase}"] = float((index - 5) * phase)
            value[f"xp_diff_{phase}"] = float((index - 4) * phase)
            value[f"gold_diff_{phase}_censored"] = False
            value[f"gold_diff_{phase}_missing"] = False
            value[f"xp_diff_{phase}_censored"] = False
            value[f"xp_diff_{phase}_missing"] = False
        value["gold_diff_25"] = None if index == 0 else value["gold_diff_25"]
        value["gold_diff_25_censored"] = index == 0
        value["gold_diff_25_missing"] = index == 0
        values.append(value)
    return pd.DataFrame(values)


def test_prepare_phase_frame_marks_short_game_targets_as_censored() -> None:
    maps = pd.DataFrame(
        [
            {"game_uid": "oe-api:1", "date": "2026-01-01T00:00:00Z", "gamelength": 19 * 60},
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "game_uid": "oe-api:1",
                "date": "2026-01-01T00:00:00Z",
                "side": side,
                **{f"goldat{phase}": base + phase for phase in (10, 15, 20, 25)},
                **{f"xpat{phase}": base + phase for phase in (10, 15, 20, 25)},
            }
            for side, base in (("Blue", 1000), ("Red", 900))
        ]
    )
    result = prepare_phase_frame(maps, teams)
    assert result.loc[0, "gold_diff_15"] == 100
    assert pd.isna(result.loc[0, "gold_diff_20"])
    assert bool(result.loc[0, "gold_diff_20_censored"])
    assert result.loc[0, "duration_seconds"] == 19 * 60


def test_phase_pregame_feature_gate_rejects_bare_final_metrics() -> None:
    frame = _phase_frame(4)
    receipt = _receipt(frame["game_uid"].tolist())
    with pytest.raises(FuturePhaseCurveError, match="pregame phase features"):
        fit_phase_curve(frame, source_receipt=receipt, feature_columns=["earnedgold"])
    with pytest.raises(FuturePhaseCurveError, match="pregame phase features"):
        fit_phase_curve(frame, source_receipt=receipt, feature_columns=["goldat15"])


def test_strict_prior_history_excludes_same_timestamp_rows() -> None:
    frame = pd.DataFrame(
        [
            {"player": "p", "date": "2026-01-01T00:00:00Z", "dpm": 10.0},
            {"player": "p", "date": "2026-01-02T00:00:00Z", "dpm": 20.0},
            {"player": "p", "date": "2026-01-02T00:00:00Z", "dpm": 30.0},
        ]
    )
    result = strict_prior_final_history(
        frame,
        entity_column="player",
        date_column="date",
        metric_columns=["dpm"],
    )
    assert pd.isna(result.loc[0, "prior_form_dpm"])
    assert result.loc[1, "prior_form_dpm"] == 10.0
    assert result.loc[2, "prior_form_dpm"] == 10.0


def test_fit_reports_support_uncertainty_and_curve_measures() -> None:
    frame = _phase_frame()
    receipt = _receipt(frame["game_uid"].tolist())
    artifact = fit_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
    )
    assert artifact["authority"] == "development_only"
    assert artifact["feature_family"] == "checkpoint_forecasts"
    assert artifact["coverage"]["gold"]["25"]["censored_rows"] == 1
    assert artifact["support"]["gold"]["10"] == len(frame)
    assert artifact["uncertainty"]["gold"]["10"] is not None
    measures = phase_curve_measures([0.0, 1.0, 3.0, 7.0], [0.0, 2.0, 3.0, 7.0])
    assert measures == {"scaling_index": 2.0, "snowball_index": -3.0}
    score = score_phase_curve(artifact, {"prior_form_gold_diff": 1.0})
    assert score["expected_gold_curve"]["10"] is not None
    assert score["uncertainty_scaling_index"] is not None


def test_chronological_folds_keep_series_whole() -> None:
    frame = _phase_frame(12)
    folds = chronological_folds(frame, n_splits=3, min_train_rows=1, cluster_column="series_id")
    seen: set[str] = set()
    for train_indices, test_indices in folds:
        train_series = set(frame.iloc[train_indices]["series_id"])
        test_series = set(frame.iloc[test_indices]["series_id"])
        assert not train_series.intersection(test_series)
        assert not seen.intersection(test_series)
        seen.update(test_series)


def test_side_swap_changes_targets_and_model_sign() -> None:
    frame = _phase_frame()
    receipt = _receipt(frame["game_uid"].tolist())
    artifact = fit_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
    )
    swapped = side_swap_frame(frame)
    assert swapped.loc[1, "gold_diff_15"] == -frame.loc[1, "gold_diff_15"]
    report = side_swap_invariance_report(artifact, frame, ["prior_form_gold_diff"])
    assert report["passed"]


def test_evaluation_has_transfer_and_missingness_sections() -> None:
    frame = _phase_frame(16)
    receipt = _receipt(frame["game_uid"].tolist())
    report = evaluate_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
        n_splits=3,
        cluster_column="series_id",
    )
    assert report["authority"] == "development_only"
    assert report["cluster_safe"]
    assert "groups" in report["regional_transfer"]
    assert "groups" in report["patch_transfer"]
    assert "missingness" in report["metrics"]["gold"]["10"]
