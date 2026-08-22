from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import lol_kills.research.future_value_rating as future_value_rating
from benchmarks.rebuild_future_phase import _partition_payload

from lol_kills.research.future_phase_curve import (
    FuturePhaseCurveError,
    _VerifiedPhaseSeriesReference,
    _make_verified_phase_series_reference,
    _revalidate_verified_phase_series_reference,
    _phase_partition_map_frame,
    _design,
    _finite_linear_predict,
    _fit_one,
    bind_phase_source,
    chronological_folds,
    evaluate_phase_curve,
    fit_phase_curve,
    phase_series_assignment_sha256,
    phase_curve_measures,
    prepare_phase_frame,
    score_phase_curve,
    side_swap_frame,
    side_swap_invariance_report,
    strict_prior_final_history,
    verify_accepted_census_artifact,
    verify_source_receipt_artifact,
)
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


def _receipt(ids: list[str]) -> dict[str, object]:
    ids = list(canonical_game_ids(ids))
    payload: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "source_as_of": "2026-08-20T23:59:59Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": ids,
        "authority": {
            "research_only": True,
            "deployment": False,
            "merge": False,
            "promotion": False,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
        },
        "source_files": {
            "annual_2025": {"locator": "2025.csv", "bytes": 1, "sha256": "a" * 64, "year": 2025},
            "annual_2026": {"locator": "2026.csv", "bytes": 1, "sha256": "b" * 64, "year": 2026},
            "bridge_oe_api_meta.json": {
                "locator": "oe_api_meta.json",
                "bytes": 1,
                "sha256": "c" * 64,
            },
            "bridge_oe_api_player_games.parquet": {
                "locator": "oe_api_player_games.parquet",
                "bytes": 1,
                "sha256": "d" * 64,
            },
            "bridge_oe_api_team_games.parquet": {
                "locator": "oe_api_team_games.parquet",
                "bytes": 1,
                "sha256": "e" * 64,
            },
            "maps": {"locator": "maps.parquet", "bytes": 1, "sha256": "f" * 64},
            "players": {"locator": "players.parquet", "bytes": 1, "sha256": "0" * 64},
            "teams": {"locator": "teams.parquet", "bytes": 1, "sha256": "1" * 64},
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return payload


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
            "series_id_source": "exact_id_proxy",
            "prior_form_gold_diff": float(index - 5),
            "prior_form_gold_diff_missing": index == 3,
            "league": "LCK",
            "tournament": "fixture",
            "blue_team_key": f"blue-{index // 2}",
            "red_team_key": f"red-{index // 2}",
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


def _crosswalk_receipt(ids: list[str]) -> dict[str, object]:
    receipt = _receipt(ids)
    receipt.update(
        {
            "model_eligible_game_count": len(ids),
            "model_eligible_game_ids": list(canonical_game_ids(ids)),
            "model_eligible_identity_sha256": identity_sha256(ids),
        }
    )
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _rehash_receipt(receipt: dict[str, object]) -> dict[str, object]:
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return receipt


def _fake_crosswalk_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    unmatched_ids: set[str] | None = None,
) -> None:
    unmatched = {str(value) for value in (unmatched_ids or set())}

    def bind(maps: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        result = maps.copy()
        game_ids = result["game_id"].astype(str)
        result.attrs["verified_leaguepedia_series_crosswalk"] = {
            "mapped_game_ids": [
                value for value in game_ids if value not in unmatched
            ],
        }
        crosswalk_path = _kwargs.get("crosswalk_path")
        if crosswalk_path is not None and Path(crosswalk_path).is_file():
            result.attrs["crosswalk_artifact_sha256"] = hashlib.sha256(
                Path(crosswalk_path).read_bytes()
            ).hexdigest()
        receipt_path = _kwargs.get("receipt_path")
        if receipt_path is not None and Path(receipt_path).is_file():
            result.attrs["crosswalk_receipt_file_sha256"] = hashlib.sha256(
                Path(receipt_path).read_bytes()
            ).hexdigest()
        return result

    def model_frame(maps: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        result = maps.copy()
        game_ids = result["game_id"].astype(str)
        result["_series_crosswalk_mapped"] = ~game_ids.isin(unmatched)
        result["series_id"] = [
            (
                f"team-tournament:proxy-{game_id}"
                if game_id in unmatched
                else f"leaguepedia:series-{int(index) // 2}"
            )
            for index, game_id in enumerate(game_ids)
        ]
        source_receipt = _kwargs.get("verified_source_receipt")
        source_receipt_sha256 = (
            str(source_receipt.get("receipt_sha256"))
            if isinstance(source_receipt, dict)
            else ""
        )
        result.attrs["series_cluster_audit"] = {
            "source_receipt_sha256": source_receipt_sha256,
            "key_fields": ["league", "tournament", "unordered_team_pair"],
            "crosswalk_assignment_sha256": "a" * 64,
            "crosswalk_sha256": "b" * 64,
            "crosswalk_artifact_sha256": str(
                result.attrs.get("crosswalk_artifact_sha256") or "c" * 64
            ),
            "crosswalk_receipt_sha256": "d" * 64,
            "partial_series_blocker": bool(unmatched),
        }
        return result

    monkeypatch.setattr(
        future_value_rating,
        "bind_verified_leaguepedia_series_crosswalk",
        bind,
    )
    monkeypatch.setattr(future_value_rating, "_map_model_frame", model_frame)


def test_design_replaces_nonfinite_feature_values_with_zero() -> None:
    frame = pd.DataFrame(
        {"prior_form_gold_diff": [float("inf"), float("-inf"), float("nan"), 1000.0]}
    )
    matrix, columns = _design(frame, ["prior_form_gold_diff"])
    assert columns == ("prior_form_gold_diff", "prior_form_gold_diff__missing")
    assert np.isfinite(matrix).all()
    assert matrix[:, 0].tolist() == [0.0, 0.0, 0.0, 0.1]
    assert matrix[:, 1].tolist() == [1.0, 1.0, 1.0, 0.0]


def test_phase_linear_prediction_is_finite_warning_free_and_parity_safe() -> None:
    matrix = np.asarray(
        [[1.25, -2.0, 0.5], [-3.0, 4.0, 1.5], [0.0, 2.5, -1.0]],
        dtype=float,
    )
    coefficients = np.asarray([2.0, -0.75, 1.5], dtype=float)
    expected = (matrix * coefficients).sum(axis=1) + 0.125
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        predicted = _finite_linear_predict(matrix, coefficients, 0.125)
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]
    np.testing.assert_allclose(predicted, expected, rtol=0.0, atol=1e-12)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = _fit_one(matrix, expected, alpha=10.0)
    assert fit is not None
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]

    invalid = matrix.copy()
    invalid[0, 0] = np.inf
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        invalid_prediction = _finite_linear_predict(invalid, coefficients)
    assert np.isnan(invalid_prediction).all()
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]


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
                "teamid": (
                    None
                    if game == "oe:game:fallback"
                    else f"oe:team:{side.casefold()}"
                ),
                **{f"goldat{phase}": base + phase for phase in (10, 15, 20, 25)},
                **{f"xpat{phase}": base + phase for phase in (10, 15, 20, 25)},
            }
            for game in games
            for side, base in (("Blue", 1000), ("Red", 900))
        ]
    )
    result = prepare_phase_frame(maps, teams)
    source = dict(result["series_id_source"].value_counts())
    assert source == {"team_tournament_proxy": 2, "exact_id_proxy": 2, "game_fallback": 1}
    numeric = result.loc[result["game_uid"].eq("123-123_game_2"), "series_id"].item()
    assert numeric == "123-123_game"
    bridge = result.loc[result["game_uid"].eq("oe:game:bridge-b"), "series_id"].item()
    assert bridge == result.loc[result["game_uid"].eq("oe:game:bridge-a"), "series_id"].item()
    assert result.loc[result["game_uid"].eq("oe:game:fallback"), "series_id"].item() == (
        "game-fallback:oe:game:fallback"
    )


def test_team_tournament_proxy_keeps_cross_date_matchups_together() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": "oe:game:cross-date-a",
                "date": "2026-01-01T00:00:00Z",
                "league": "LCK",
                "tournament": "Spring",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
            {
                "game_uid": "oe:game:cross-date-b",
                "date": "2026-01-03T00:00:00Z",
                "league": "LCK",
                "tournament": "Spring",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "game_uid": game,
                "date": date,
                "side": side,
                **{f"goldat{phase}": base + phase for phase in (10, 15, 20, 25)},
                **{f"xpat{phase}": base + phase for phase in (10, 15, 20, 25)},
            }
            for game, date in (
                ("oe:game:cross-date-a", "2026-01-01T00:00:00Z"),
                ("oe:game:cross-date-b", "2026-01-03T00:00:00Z"),
            )
            for side, base in (("Blue", 1000), ("Red", 900))
        ]
    )
    result = prepare_phase_frame(maps, teams)
    assert set(result["series_id_source"]) == {"team_tournament_proxy"}
    assert result["series_id"].nunique() == 1
    assert result.loc[0, "series_id"] == result.loc[1, "series_id"]


def test_team_tournament_proxy_prefers_stable_oe_team_ids() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": "oe:game:stable-team-ids",
                "date": "2026-01-01T00:00:00Z",
                "league": "LCK",
                "tournament": "Spring",
                "blue_team_key": "old-blue-alias",
                "red_team_key": "old-red-alias",
            }
        ]
    )
    teams = pd.DataFrame(
        [
            {
                "game_uid": "oe:game:stable-team-ids",
                "date": "2026-01-01T00:00:00Z",
                "side": side,
                "teamid": team_id,
                **{f"goldat{phase}": base + phase for phase in (10, 15, 20, 25)},
                **{f"xpat{phase}": base + phase for phase in (10, 15, 20, 25)},
            }
            for side, team_id, base in (
                ("Blue", "oe:blue-stable", 1000),
                ("Red", "oe:red-stable", 900),
            )
        ]
    )
    result = prepare_phase_frame(maps, teams)
    assert result.loc[0, "series_id"] == (
        "team-tournament:LCK|Spring|oe:blue-stable|oe:red-stable"
    )


def test_team_tournament_proxy_reports_possible_collisions() -> None:
    frame = _phase_frame(4)
    frame["series_id"] = "team-tournament:LCK|Spring|blue|red"
    frame["series_id_source"] = "team_tournament_proxy"
    receipt = _receipt(frame["game_uid"].tolist())
    report = evaluate_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
        n_splits=2,
        cluster_column="series_id",
    )
    identity = report["series_identity"]
    assert report["cluster_safe"] is False
    assert identity["source_counts"]["team_tournament_proxy"] == len(frame)
    assert identity["cluster_counts"]["team_tournament_proxy"] == 1
    assert identity["possible_collisions"]["clusters"] == 1
    assert identity["possible_collisions"]["cross_date_clusters"] == 1
    assert identity["blockers"]


def test_phase_pregame_feature_gate_rejects_bare_final_metrics() -> None:
    frame = _phase_frame(4)
    receipt = _receipt(frame["game_uid"].tolist())
    with pytest.raises(FuturePhaseCurveError, match="pregame phase features"):
        fit_phase_curve(frame, source_receipt=receipt, feature_columns=["earnedgold"])
    with pytest.raises(FuturePhaseCurveError, match="pregame phase features"):
        fit_phase_curve(frame, source_receipt=receipt, feature_columns=["goldat15"])


def test_phase_source_rejects_rows_outside_the_accepted_census() -> None:
    frame = _phase_frame(4)
    extra = frame.iloc[[0]].copy()
    extra["game_uid"] = "oe-api:outside-census"
    frame = pd.concat([frame, extra], ignore_index=True)
    receipt = _receipt(frame.loc[frame["game_uid"] != "oe-api:outside-census", "game_uid"].tolist())
    with pytest.raises(FuturePhaseCurveError, match="outside the accepted census"):
        fit_phase_curve(
            frame,
            source_receipt=receipt,
            feature_columns=["prior_form_gold_diff"],
        )


def test_phase_source_rejects_forged_receipts_and_source_file_records() -> None:
    frame = _phase_frame(4)
    receipt = _receipt(frame["game_uid"].tolist())
    forged_hash = dict(receipt)
    forged_hash["source_identity_sha256"] = "0" * 64
    with pytest.raises(FuturePhaseCurveError, match="receipt hash"):
        bind_phase_source(frame, forged_hash, allow_subset=True)

    forged_file = dict(receipt)
    forged_file["source_files"] = {
        key: dict(value) for key, value in receipt["source_files"].items()
    }
    forged_file["source_files"]["annual_2025"]["sha256"] = "not-a-hash"
    forged_file["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in forged_file.items() if key != "receipt_sha256"},
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(FuturePhaseCurveError, match="source file hash"):
        bind_phase_source(frame, forged_file, allow_subset=True)

    missing_bridge = dict(receipt)
    missing_bridge["source_files"] = {
        key: value
        for key, value in receipt["source_files"].items()
        if key != "bridge_oe_api_team_games.parquet"
    }
    missing_bridge["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in missing_bridge.items() if key != "receipt_sha256"},
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(FuturePhaseCurveError, match="required OE transport"):
        bind_phase_source(frame, missing_bridge, allow_subset=True)


def test_durable_source_receipt_artifact_is_verified(tmp_path: Path) -> None:
    receipt = _receipt(["oe-api:1"])
    path = tmp_path / "source-receipt.json"
    raw = json.dumps(receipt, indent=2, sort_keys=True).encode("utf-8")
    path.write_bytes(raw)
    reference = {
        "locator": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "source_receipt_sha256": receipt["receipt_sha256"],
    }
    verified = verify_source_receipt_artifact(
        reference,
        runtime_root=tmp_path,
        expected_source_game_count=1,
        expected_source_identity_sha256=receipt["source_identity_sha256"],
        expected_source_as_of=receipt["source_as_of"],
    )
    assert verified["status"] == "verified"
    assert verified["source_receipt_sha256"] == receipt["receipt_sha256"]
    assert verified["transport"] == (
        "official_public_oracles_elixir_annual_exports_plus_oe_api_bridge"
    )


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


def test_chronological_folds_produce_three_valid_folds_from_four_blocks() -> None:
    frame = _phase_frame(16)
    folds = chronological_folds(
        frame,
        n_splits=4,
        min_train_rows=1,
        cluster_column="series_id",
    )
    assert len(folds) == 3
    seen_test_clusters: set[str] = set()
    last_by_cluster = frame.groupby("series_id")["date"].max()
    for train_indices, test_indices in folds:
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        test_start = test["date"].min()
        train_clusters = set(train["series_id"])
        test_clusters = set(test["series_id"])
        assert not train_clusters.intersection(test_clusters)
        assert not seen_test_clusters.intersection(test_clusters)
        assert train["date"].max() < test_start
        assert all(last_by_cluster[cluster] < test_start for cluster in train_clusters)
        seen_test_clusters.update(test_clusters)
    report = evaluate_phase_curve(
        frame,
        source_receipt=_receipt(frame["game_uid"].tolist()),
        feature_columns=["prior_form_gold_diff"],
        n_splits=4,
        required_validation_folds=3,
        transfer_columns=(),
    )
    assert report["chronological_blocks_requested"] == 4
    assert report["validation_folds_required"] == 3
    assert report["validation_folds_valid"] == 3


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
    expected_end = pd.to_datetime(frame["date"], utc=True).max().isoformat().replace("+00:00", "Z")
    assert report["evaluation_window"]["date_end"] == expected_end
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


def test_evaluation_blocks_non_authoritative_team_tournament_series_proxies() -> None:
    frame = _phase_frame(16)
    frame["series_id_source"] = "team_tournament_proxy"
    receipt = _receipt(frame["game_uid"].tolist())
    report = evaluate_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
        n_splits=3,
        cluster_column="series_id",
    )
    assert report["cluster_safe"] is False
    assert report["series_identity"]["authoritative"] is False
    assert report["series_identity"]["status"] == "blocked"
    assert report["series_identity"]["source_counts"]["team_tournament_proxy"] == len(frame)
    assert report["series_identity"]["blockers"]


def test_verified_mixed_partition_binds_hashes_and_scopes_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _phase_frame(8)
    receipt = _crosswalk_receipt(frame["game_uid"].tolist())
    _fake_crosswalk_loader(monkeypatch)
    artifact = fit_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
        crosswalk_path="fixture/crosswalk.json",
        crosswalk_receipt_path="fixture/crosswalk.receipt.json",
        crosswalk_receipt_file_sha256="e" * 64,
    )
    assert artifact["series_partition_source"] == (
        "mixed:leaguepedia_crosswalk+conservative_series_superset"
    )
    assert artifact["series_partition_mapping_sha256"] == "a" * 64
    assert artifact["series_partition_crosswalk_sha256"] == "b" * 64
    assert artifact["series_partition_eligible_game_ids"] == receipt[
        "model_eligible_game_ids"
    ]
    assert artifact["cross_model_series_partition"]["status"] == "non_comparable"
    assert artifact["cross_model_series_partition"]["proxy_authority_blocker"] is False
    assert artifact["series_partition"]["proxy_authority_blocker"] is False
    assert artifact["series_partition_proxy_authority_blocker"] is False
    assert artifact["series_identity"]["authoritative"] is True
    assert artifact["series_identity"]["blockers"] == []


def test_accepted_unmatched_reference_row_outside_eligible_scope_is_not_a_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_frame = _phase_frame(9)
    phase_frame = reference_frame.iloc[:8].copy()
    accepted_ids = list(reference_frame["game_uid"].astype(str))
    eligible_ids = list(canonical_game_ids(accepted_ids[:8]))
    receipt = _crosswalk_receipt(accepted_ids)
    receipt["model_eligible_game_count"] = len(eligible_ids)
    receipt["model_eligible_game_ids"] = eligible_ids
    receipt["model_eligible_identity_sha256"] = identity_sha256(eligible_ids)
    _rehash_receipt(receipt)
    _fake_crosswalk_loader(monkeypatch, unmatched_ids={accepted_ids[-1]})
    reference = future_value_rating._map_model_frame(
        _phase_partition_map_frame(reference_frame)
    )
    expected_eligible = phase_series_assignment_sha256(
        reference.loc[reference["game_id"].astype(str).isin(set(eligible_ids))],
        game_column="game_id",
    )
    artifact = fit_phase_curve(
        phase_frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
        crosswalk_path="fixture/crosswalk.json",
        crosswalk_receipt_path="fixture/crosswalk.receipt.json",
        crosswalk_receipt_file_sha256="e" * 64,
        series_partition_reference_frame=reference_frame,
        series_partition_assignment_sha256=expected_eligible,
    )
    partition = artifact["series_partition"]
    assert partition["reference_game_count"] == len(accepted_ids)
    assert partition["reference_identity_sha256"] == identity_sha256(accepted_ids)
    assert partition["audit"]["full_source_map_count"] == len(accepted_ids)
    assert partition["audit"]["retained_proxy_game_count"] == 0
    assert partition["reference_audit"]["partial_series_blocker"] is True
    assert partition["proxy_authority_blocker"] is False
    assert partition["authoritative"] is True
    assert artifact["cross_model_series_partition"]["status"] == "comparable"
    assert artifact["series_identity"]["authoritative"] is True


def test_verified_mixed_partition_evaluation_uses_shared_series_clusters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _phase_frame(12)
    receipt = _crosswalk_receipt(frame["game_uid"].tolist())
    _fake_crosswalk_loader(monkeypatch)
    report = evaluate_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
        n_splits=2,
        transfer_columns=(),
        crosswalk_path="fixture/crosswalk.json",
        crosswalk_receipt_path="fixture/crosswalk.receipt.json",
        crosswalk_receipt_file_sha256="e" * 64,
    )
    assert report["cluster_column"] == "series_id"
    assert report["cross_model_series_partition"]["status"] == "non_comparable"
    assert report["cross_model_series_partition"]["proxy_authority_blocker"] is False
    assert report["series_partition"]["proxy_authority_blocker"] is False
    assert report["series_partition_mapping_sha256"] == "a" * 64
    assert report["series_partition_eligible_game_count"] == len(frame)
    assert report["series_identity"]["authoritative"] is True
    assert report["cluster_safe"] is True


def test_verified_mixed_partition_requires_full_reference_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_frame = _phase_frame(9)
    frame = reference_frame.iloc[:8].copy()
    receipt = _crosswalk_receipt(reference_frame["game_uid"].tolist())
    receipt["model_eligible_game_ids"] = list(
        canonical_game_ids(frame["game_uid"].tolist())
    )
    receipt["model_eligible_game_count"] = len(frame)
    receipt["model_eligible_identity_sha256"] = identity_sha256(
        receipt["model_eligible_game_ids"]
    )
    receipt_payload = dict(receipt)
    receipt_payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt_payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _fake_crosswalk_loader(monkeypatch)
    reference = future_value_rating._map_model_frame(
        _phase_partition_map_frame(reference_frame)
    )
    expected_eligible = phase_series_assignment_sha256(
        reference.loc[
            reference["game_id"].astype(str).isin(
                set(receipt["model_eligible_game_ids"])
            )
        ],
        game_column="game_id",
    )
    expected_reference = phase_series_assignment_sha256(
        reference,
        game_column="game_id",
    )
    artifact = fit_phase_curve(
        frame,
        source_receipt=receipt,
        feature_columns=["prior_form_gold_diff"],
        crosswalk_path="fixture/crosswalk.json",
        crosswalk_receipt_path="fixture/crosswalk.receipt.json",
        crosswalk_receipt_file_sha256="e" * 64,
        series_partition_reference_frame=reference_frame,
        series_partition_assignment_sha256=expected_eligible,
    )
    assert artifact["cross_model_series_partition"]["status"] == "comparable"
    assert artifact["cross_model_series_partition"]["proxy_authority_blocker"] is False
    assert artifact["series_partition"]["proxy_authority_blocker"] is False
    assert _partition_payload(artifact)["proxy_authority_blocker"] is False
    assert artifact["series_partition_reference_game_count"] == len(reference_frame)
    assert artifact["series_partition_reference_identity_sha256"] == identity_sha256(
        reference_frame["game_uid"].tolist()
    )
    assert artifact["series_partition_reference_assignment_sha256"] == expected_reference
    assert artifact["series_partition_eligible_assignment_sha256"] == expected_eligible
    assert expected_reference != expected_eligible


def test_phase_assignment_digest_includes_proxy_rows() -> None:
    reference = pd.DataFrame(
        {
            "game_id": ["game-1", "game-2"],
            "series_id": [
                "leaguepedia:series-1",
                "team-tournament:league|event|blue|red",
            ],
        }
    )
    eligible = reference.iloc[[0]].copy()
    assert phase_series_assignment_sha256(reference, game_column="game_id") != (
        phase_series_assignment_sha256(eligible, game_column="game_id")
    )


def test_verified_mixed_partition_rejects_full_digest_as_eligible_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_frame = _phase_frame(9)
    frame = reference_frame.iloc[:8].copy()
    receipt = _crosswalk_receipt(reference_frame["game_uid"].tolist())
    receipt["model_eligible_game_ids"] = list(
        canonical_game_ids(frame["game_uid"].tolist())
    )
    receipt["model_eligible_game_count"] = len(frame)
    receipt["model_eligible_identity_sha256"] = identity_sha256(
        receipt["model_eligible_game_ids"]
    )
    receipt_payload = dict(receipt)
    receipt_payload.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            receipt_payload,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _fake_crosswalk_loader(monkeypatch)
    reference = future_value_rating._map_model_frame(
        _phase_partition_map_frame(reference_frame)
    )
    full_digest = phase_series_assignment_sha256(reference, game_column="game_id")
    with pytest.raises(FuturePhaseCurveError, match="assignments differ"):
        fit_phase_curve(
            frame,
            source_receipt=receipt,
            feature_columns=["prior_form_gold_diff"],
            crosswalk_path="fixture/crosswalk.json",
            crosswalk_receipt_path="fixture/crosswalk.receipt.json",
            crosswalk_receipt_file_sha256="e" * 64,
            series_partition_reference_frame=reference_frame,
            series_partition_assignment_sha256=full_digest,
        )


def test_forged_prebound_series_reference_is_rejected() -> None:
    frame = _phase_frame(2)
    with pytest.raises(FuturePhaseCurveError, match="not verified"):
        _VerifiedPhaseSeriesReference(
            frame=frame,
            source_game_count=2,
            source_identity_sha256="a" * 64,
            source_receipt_sha256="b" * 64,
            crosswalk_artifact_sha256="c" * 64,
            crosswalk_sha256="d" * 64,
            crosswalk_assignment_sha256="e" * 64,
            crosswalk_receipt_sha256="f" * 64,
            crosswalk_receipt_file_sha256="0" * 64,
            eligible_assignment_sha256="1" * 64,
            reference_assignment_sha256="1" * 64,
            reference_game_count=2,
            reference_identity_sha256="2" * 64,
            _factory_token=object(),
        )


def test_cached_series_reference_revalidates_mutated_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["1", "2"],
            "date": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"],
            "blue_team_key": ["blue-1", "blue-2"],
            "red_team_key": ["red-1", "red-2"],
            "series_id": ["leaguepedia:series-1", "leaguepedia:series-1"],
            "_series_crosswalk_mapped": [True, True],
        }
    )
    source_receipt = _crosswalk_receipt(frame["game_id"].tolist())
    crosswalk_path = tmp_path / "crosswalk.json"
    crosswalk_receipt_path = tmp_path / "crosswalk.receipt.json"
    crosswalk_path.write_bytes(b"crosswalk")
    crosswalk_receipt_path.write_bytes(b"crosswalk-receipt")
    crosswalk_artifact_sha256 = hashlib.sha256(crosswalk_path.read_bytes()).hexdigest()
    crosswalk_receipt_file_sha256 = hashlib.sha256(
        crosswalk_receipt_path.read_bytes()
    ).hexdigest()
    frame.attrs["series_cluster_source"] = (
        "mixed:leaguepedia_crosswalk+conservative_series_superset"
    )
    frame.attrs["series_cluster_audit"] = {
        "source_receipt_sha256": source_receipt["receipt_sha256"],
        "crosswalk_artifact_sha256": crosswalk_artifact_sha256,
        "crosswalk_sha256": "a" * 64,
        "crosswalk_assignment_sha256": "b" * 64,
        "crosswalk_receipt_sha256": "c" * 64,
    }
    expected = phase_series_assignment_sha256(frame, game_column="game_id")
    reference = _make_verified_phase_series_reference(
        frame,
        source_receipt=source_receipt,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=crosswalk_receipt_path,
        crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        eligible_ids=frame["game_id"].tolist(),
        eligible_assignment_sha256=expected,
    )

    def bind(maps: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        return maps.copy()

    def model_frame(maps: pd.DataFrame, **_kwargs: object) -> pd.DataFrame:
        result = maps.copy()
        result["series_id"] = "leaguepedia:series-1"
        result["_series_crosswalk_mapped"] = True
        return result

    monkeypatch.setattr(
        future_value_rating,
        "bind_verified_leaguepedia_series_crosswalk",
        bind,
    )
    monkeypatch.setattr(future_value_rating, "_map_model_frame", model_frame)
    _revalidate_verified_phase_series_reference(
        reference,
        source_receipt=source_receipt,
        crosswalk_path=crosswalk_path,
        crosswalk_receipt_path=crosswalk_receipt_path,
        crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
        eligible_ids=frame["game_id"].tolist(),
        expected_assignment_sha256=expected,
    )
    reference.frame.loc[0, "series_id"] = "forged:series"
    forged_expected = phase_series_assignment_sha256(
        reference.frame,
        game_column="game_id",
    )
    with pytest.raises(FuturePhaseCurveError, match="differ from verified crosswalk"):
        _revalidate_verified_phase_series_reference(
            reference,
            source_receipt=source_receipt,
            crosswalk_path=crosswalk_path,
            crosswalk_receipt_path=crosswalk_receipt_path,
            crosswalk_receipt_file_sha256=crosswalk_receipt_file_sha256,
            eligible_ids=frame["game_id"].tolist(),
            expected_assignment_sha256=forged_expected,
        )


def test_verified_mixed_partition_rejects_reference_assignment_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _phase_frame(8)
    receipt = _crosswalk_receipt(frame["game_uid"].tolist())
    _fake_crosswalk_loader(monkeypatch)
    with pytest.raises(FuturePhaseCurveError, match="assignments differ"):
        fit_phase_curve(
            frame,
            source_receipt=receipt,
            feature_columns=["prior_form_gold_diff"],
            crosswalk_path="fixture/crosswalk.json",
            crosswalk_receipt_path="fixture/crosswalk.receipt.json",
            crosswalk_receipt_file_sha256="e" * 64,
            series_partition_reference_frame=frame,
            series_partition_assignment_sha256="f" * 64,
        )


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
    receipt_reference = candidate["source_receipt_artifact"]
    assert source["source_as_of"] == evaluation["source_as_of"]
    assert source["source_game_count"] == evaluation["source_game_count"] == 17756
    assert source["source_identity_sha256"] == evaluation["source_identity_sha256"]
    assert candidate["source_receipt_sha256"] == evaluation["source_receipt_sha256"]
    assert candidate["source_receipt_sha256"] == receipt_reference["source_receipt_sha256"]
    assert source["transport"] == (
        "official_public_oracles_elixir_annual_exports_plus_oe_api_bridge"
    )
    assert evaluation["source_transport"] == source["transport"]
    assert candidate["evaluation_scope"]["date_end"] == source["source_as_of"]
    assert evaluation["evaluation_window"]["date_end"] == evaluation["source_as_of"]
    assert evaluation["cluster_safe"] is False
    assert evaluation["authoritative_series_identity"] is False
    assert reference == evaluation["accepted_game_ids_artifact"]
    assert reference["locator"] == "data/lol/v2/evaluation/future-phase-accepted-census.json"
    census_path = root / reference["locator"]
    assert census_path.is_file()
    assert census_path.stat().st_size == reference["bytes"]
    assert hashlib.sha256(census_path.read_bytes()).hexdigest() == reference["sha256"]
    assert len(reference["sha256"]) == 64
    assert reference["game_ids_field"] == "game_ids"
    verified = verify_accepted_census_artifact(
        reference,
        runtime_root=root,
        expected_source_game_count=evaluation["source_game_count"],
        expected_source_identity_sha256=evaluation["source_identity_sha256"],
    )
    assert verified["status"] == "verified"
    assert verified["game_count"] == evaluation["source_game_count"]
    assert verified["source_identity_sha256"] == evaluation["source_identity_sha256"]
    receipt_path = root / receipt_reference["locator"]
    verified_receipt = verify_source_receipt_artifact(
        receipt_reference,
        runtime_root=root,
        expected_source_game_count=evaluation["source_game_count"],
        expected_source_identity_sha256=evaluation["source_identity_sha256"],
        expected_source_as_of=evaluation["source_as_of"],
    )
    assert receipt_path.is_file()
    assert verified_receipt["source_receipt_sha256"] == candidate["source_receipt_sha256"]
    assert verified_receipt["transport"] == source["transport"]
