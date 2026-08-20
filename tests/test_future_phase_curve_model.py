from __future__ import annotations

import json
from pathlib import Path

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


def test_prepare_phase_frame_derives_outcome_free_series_provenance() -> None:
    games = [
        "123-123_game_1",
        "123-123_game_2",
        "oe:game:bridge-a",
        "oe:game:bridge-b",
        "oe:game:fallback",
    ]
    date = "2026-01-01T00:00:00Z"
    maps = pd.DataFrame(
        [
            {
                "game_uid": game,
                "date": date,
                "league": "LCK",
                "tournament": "Spring",
                "blue_team_key": "blue",
                "red_team_key": "red",
            }
            for game in games
        ]
    )
    maps.loc[maps["game_uid"].eq("oe:game:fallback"), ["blue_team_key", "red_team_key"]] = None
    teams = pd.DataFrame(
        [
            {
                "game_uid": game,
                "date": date,
                "side": side,
                "teamid": f"{side}-{game}",
                **{f"goldat{phase}": base + phase for phase in (10, 15, 20, 25)},
                **{f"xpat{phase}": base + phase for phase in (10, 15, 20, 25)},
            }
            for game in games
            for side, base in (("Blue", 1000), ("Red", 900))
        ]
    )
    result = prepare_phase_frame(maps, teams)
    source = dict(result["series_id_source"].value_counts())
    assert source == {"team_date_proxy": 2, "exact_id_proxy": 2, "game_fallback": 1}
    numeric = result.loc[result["game_uid"].eq("123-123_game_2"), "series_id"].item()
    assert numeric == "123-123_game"
    bridge = result.loc[result["game_uid"].eq("oe:game:bridge-b"), "series_id"].item()
    assert bridge == result.loc[result["game_uid"].eq("oe:game:bridge-a"), "series_id"].item()
    assert result.loc[result["game_uid"].eq("oe:game:fallback"), "series_id"].item() == (
        "game-fallback:oe:game:fallback"
    )


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


def test_chronological_folds_exclude_clusters_that_continue_into_test() -> None:
    frame = _phase_frame(8)
    frame["date"] = pd.to_datetime(
        [
            "2026-01-01T00:00:00Z",
            "2026-01-04T00:00:00Z",
            "2026-01-02T00:00:00Z",
            "2026-01-05T00:00:00Z",
            "2026-01-03T00:00:00Z",
            "2026-01-06T00:00:00Z",
            "2026-01-07T00:00:00Z",
            "2026-01-08T00:00:00Z",
        ],
        utc=True,
    )
    frame["series_id"] = ["A", "A", "B", "B", "C", "C", "D", "D"]
    folds = chronological_folds(frame, n_splits=2, min_train_rows=0, cluster_column="series_id")
    for train_indices, test_indices in folds:
        test_start = frame.iloc[test_indices]["date"].min()
        test_clusters = set(frame.iloc[test_indices]["series_id"])
        train = frame.iloc[train_indices]
        last_by_cluster = frame.groupby("series_id")["date"].max()
        assert all(
            last_by_cluster[cluster] < test_start
            for cluster in train["series_id"]
            if cluster not in test_clusters
        )


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
    assert report["source"] == "oracle_elixir_only"
    assert report["source_as_of"] == receipt["source_as_of"]
    assert report["source_game_count"] == receipt["source_game_count"]
    assert report["source_identity_sha256"] == receipt["source_identity_sha256"]
    assert report["accepted_game_ids"] == receipt["accepted_game_ids"]
    assert len(report["source_receipt_sha256"]) == 64
    assert report["cluster_safe"]
    assert "groups" in report["regional_transfer"]
    assert "groups" in report["patch_transfer"]
    assert "cluster_boundary_exclusions" in report
    assert "missingness" in report["metrics"]["gold"]["10"]
    assert report["metrics"]["gold"]["10"]["baseline_rows_match"]
    assert report["metrics"]["gold"]["10"]["baseline_zero"]["rows"] == report[
        "metrics"
    ]["gold"]["10"]["rows"]
    for transfer in (report["regional_transfer"], report["patch_transfer"]):
        for group in transfer["groups"].values():
            if not group["available"]:
                continue
            for kind in ("gold", "xp"):
                for phase in ("10", "15", "20", "25"):
                    assert group["metrics"][kind][phase]["baseline_rows_match"]


def test_static_phase_artifacts_bind_the_accepted_census_reference() -> None:
    root = Path(__file__).parents[1]
    candidate = json.loads(
        (root / "data/lol/v2/evaluation/future-phase-candidate.json").read_text()
    )
    evaluation = json.loads(
        (root / "data/lol/v2/evaluation/future-phase-evaluation.json").read_text()
    )
    source = candidate["source"]
    reference = source["accepted_game_ids_artifact"]
    assert source["source_as_of"] == evaluation["source_as_of"]
    assert source["source_game_count"] == evaluation["source_game_count"] == 17756
    assert source["source_identity_sha256"] == evaluation["source_identity_sha256"]
    assert reference == evaluation["accepted_game_ids_artifact"]
    assert len(reference["sha256"]) == 64
    assert reference["game_ids_field"] == "game_ids"
