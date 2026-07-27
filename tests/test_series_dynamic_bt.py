from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_kills.ratings import series_dynamic_bt
from lol_kills.ratings.series_dynamic_bt import (
    BASE_RATE_ID,
    ELO_BASELINE_ID,
    SeriesDynamicBTConfig,
    SeriesDynamicBradleyTerry,
    SeriesEloConfig,
    SeriesTournamentSpec,
    config_sha256,
    evaluate_promotion_gate,
    model_code_sha256,
    model_identity,
    prepare_series_observations,
    proper_scores,
    run_prequential_series,
    run_series_rating_tournament,
    series_win_probability,
)


def _observation(
    series_key: str,
    prediction_time: str,
    completion_time: str,
    team_a: str,
    team_b: str,
    outcome: int,
    *,
    best_of: int = 1,
    home_a: str = "LPL",
    home_b: str = "LPL",
    team_a_name: str | None = None,
    team_b_name: str | None = None,
) -> dict:
    return {
        "series_key": series_key,
        "prediction_time": prediction_time,
        "date": completion_time,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_name": team_a_name or team_a.upper(),
        "team_b_name": team_b_name or team_b.upper(),
        "home_a": home_a,
        "home_b": home_b,
        "y_a": outcome,
        "n_maps": 1 if best_of == 1 else best_of,
        "international": home_a != home_b,
        "league": home_a if home_a == home_b else "MSI",
        "scheduled_best_of": best_of,
        "series_source": "fixture",
        "source_series_id": series_key,
        "completion_status": "completed",
        "completion_source": "score_to_format_validation",
        "format_source": "fixture_registry",
        "series_provenance": f"fixture:{series_key}",
        "rating_eligible": True,
        "series_weight": 1.0,
    }


def _bo1_map(
    series_id: str,
    timestamp: str,
    blue: str,
    red: str,
    blue_win: int,
) -> dict:
    return {
        "game_uid": f"{series_id}:1",
        "date": timestamp,
        "league": "LPL",
        "split": "Split",
        "playoffs": 0,
        "source": "grid",
        "grid_series_id": series_id,
        "grid_game_index": 1,
        "game": 1,
        "blue_team": blue,
        "red_team": red,
        "y_blue_win": blue_win,
        "series_format": "Bo1",
        "series_format_source": "fixture_registry",
        "series_completion_source": "fixture_registry",
        "series_completion_state": "completed",
    }


def _bo3_maps(
    series_id: str,
    timestamps: tuple[str, str, str],
    *,
    winner: str = "A",
) -> list[dict]:
    rows = []
    results = [winner == "A", winner != "A", winner == "A"]
    sides = [("A", "B"), ("B", "A"), ("A", "B")]
    for index, (timestamp, result, (blue, red)) in enumerate(
        zip(timestamps, results, sides), start=1
    ):
        rows.append(
            {
                "game_uid": f"{series_id}:{index}",
                "date": timestamp,
                "league": "LPL",
                "split": "Split",
                "playoffs": 0,
                "source": "grid",
                "grid_series_id": series_id,
                "grid_game_index": index,
                "game": index,
                "blue_team": blue,
                "red_team": red,
                "y_blue_win": int(result),
                "series_format": "Bo3",
                "series_format_source": "fixture_registry",
                "series_completion_source": "fixture_registry",
                "series_completion_state": "completed",
            }
        )
    return rows


def test_side_and_order_complement_for_every_supported_format() -> None:
    model = SeriesDynamicBradleyTerry(
        SeriesDynamicBTConfig(context_enabled=False)
    )
    for best_of in (1, 3, 5):
        forward = model.predict(
            "a",
            "b",
            timestamp="2026-01-01T00:00:00Z",
            scheduled_best_of=best_of,
        )
        reverse = model.predict(
            "b",
            "a",
            timestamp="2026-01-01T00:00:00Z",
            scheduled_best_of=best_of,
        )
        assert forward.probability == pytest.approx(
            1.0 - reverse.probability, abs=1e-15
        )
        assert forward.map_probability == pytest.approx(
            1.0 - reverse.map_probability, abs=1e-15
        )


def test_exact_series_probability_transforms_and_format_calibration() -> None:
    p = 0.7
    assert series_win_probability(p, 1) == pytest.approx(p)
    assert series_win_probability(p, 3) == pytest.approx(p**2 * (3 - 2 * p))
    assert series_win_probability(p, 5) == pytest.approx(
        p**3 * (10 - 15 * p + 6 * p**2)
    )
    for best_of in (1, 3, 5):
        assert series_win_probability(p, best_of) == pytest.approx(
            1.0 - series_win_probability(1.0 - p, best_of)
        )
    metrics = proper_scores(
        [1, 0, 1, 0, 1, 0],
        [0.7, 0.3, 0.7, 0.3, 0.7, 0.3],
        scheduled_best_of=[1, 1, 3, 3, 5, 5],
        ece_bins=5,
    )
    assert set(metrics["format_stratified_calibration"]) == {
        "Bo1",
        "Bo3",
        "Bo5",
    }


def test_same_timestamp_batch_has_no_peer_outcome_leakage() -> None:
    base = pd.DataFrame(
        [
            _observation(
                "history",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "a",
                "b",
                1,
            ),
            _observation(
                "same-1",
                "2026-01-02T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "a",
                "b",
                1,
            ),
            _observation(
                "same-2",
                "2026-01-02T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "a",
                "c",
                0,
            ),
        ]
    )
    changed = base.copy()
    changed.loc[changed["series_key"].eq("same-1"), "y_a"] = 0
    first = run_prequential_series(base).predictions.set_index("series_key")
    second = run_prequential_series(changed).predictions.set_index("series_key")
    assert first.loc["same-2", "probability"] == second.loc[
        "same-2", "probability"
    ]


def test_pending_series_outcome_is_not_used_before_verified_completion() -> None:
    base = pd.DataFrame(
        [
            _observation(
                "pending",
                "2026-01-01T00:00:00Z",
                "2026-01-04T00:00:00Z",
                "a",
                "b",
                1,
                best_of=3,
            ),
            _observation(
                "intervening",
                "2026-01-03T00:00:00Z",
                "2026-01-03T00:00:00Z",
                "a",
                "c",
                1,
            ),
        ]
    )
    changed = base.copy()
    changed.loc[changed["series_key"].eq("pending"), "y_a"] = 0
    original = run_prequential_series(base).predictions.set_index("series_key")
    counterfactual = run_prequential_series(changed).predictions.set_index(
        "series_key"
    )
    assert original.loc["intervening", "probability"] == counterfactual.loc[
        "intervening", "probability"
    ]
    assert original.loc["pending", "prediction_time"] < original.loc[
        "pending", "completion_time"
    ]


def test_context_term_requires_completed_historical_bridge_support() -> None:
    observations = pd.DataFrame(
        [
            _observation(
                "bridge",
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "a",
                "b",
                1,
                home_a="LPL",
                home_b="LCK",
            ),
            _observation(
                "same-completion-time",
                "2026-01-02T00:00:00Z",
                "2026-01-02T01:00:00Z",
                "c",
                "d",
                1,
                home_a="LPL",
                home_b="LCK",
            ),
            _observation(
                "after-bridge",
                "2026-01-03T00:00:00Z",
                "2026-01-03T01:00:00Z",
                "e",
                "f",
                0,
                home_a="LPL",
                home_b="LCK",
            ),
        ]
    )
    config = SeriesDynamicBTConfig(
        context_enabled=True,
        min_bridge_series=1,
        min_bridge_teams_per_context=1,
    )
    predictions = run_prequential_series(
        observations, config=config
    ).predictions.set_index("series_key")
    assert not bool(predictions.loc["bridge", "context_supported"])
    # Starts at the bridge's completion timestamp, so prediction comes first.
    assert not bool(
        predictions.loc["same-completion-time", "context_supported"]
    )
    assert bool(predictions.loc["after-bridge", "context_supported"])


def test_cutoff_excludes_future_completed_series_without_partial_series() -> None:
    rows = [
        _bo1_map("past", "2026-01-01T10:00:00Z", "A", "B", 1),
        *_bo3_maps(
            "crosses-cutoff",
            (
                "2026-01-02T10:00:00Z",
                "2026-01-02T11:00:00Z",
                "2026-01-03T10:00:00Z",
            ),
        ),
    ]
    frame = pd.DataFrame(rows)
    cutoff = "2026-01-02T12:00:00Z"
    before = prepare_series_observations(frame, data_cutoff=cutoff)
    assert len(before) == 1
    assert before.iloc[0]["scheduled_best_of"] == 1
    assert "crosses-cutoff" not in " ".join(before["series_provenance"])

    future = pd.concat(
        [
            frame,
            pd.DataFrame(
                [
                    _bo1_map(
                        "far-future",
                        "2026-02-01T10:00:00Z",
                        "A",
                        "C",
                        0,
                    )
                ]
            ),
        ],
        ignore_index=True,
    )
    with_future = prepare_series_observations(future, data_cutoff=cutoff)
    pd.testing.assert_frame_equal(before, with_future)


def test_unverified_format_cannot_enter_evaluation() -> None:
    row = _bo1_map("unverified", "2026-01-01T10:00:00Z", "A", "B", 1)
    row["series_format"] = None
    row["series_format_source"] = None
    observations = prepare_series_observations(pd.DataFrame([row]))
    assert observations.empty
    with pytest.raises(ValueError, match="no rating-eligible"):
        run_series_rating_tournament(
            pd.DataFrame([row]),
            spec=SeriesTournamentSpec(
                validation_start="2026-01-02",
                test_start="2026-01-03",
                bootstrap_replicates=100,
            ),
        )


def test_each_completed_series_has_one_unit_weight_and_one_update() -> None:
    observations = pd.DataFrame(
        [
            _observation(
                "long-series",
                "2026-01-01T00:00:00Z",
                "2026-01-01T03:00:00Z",
                "a",
                "b",
                1,
                best_of=5,
            )
        ]
    )
    run = run_prequential_series(observations)
    assert run.predictions["series_weight"].tolist() == [1.0]
    assert run.model.observed_series == 1
    assert run.model.teams["a"].observations == 1
    assert run.model.teams["b"].observations == 1


def test_probability_and_snapshot_bounds_and_uncertainty_labels() -> None:
    observations = pd.DataFrame(
        [
            _observation(
                f"s{index}",
                f"2026-01-{index + 1:02d}T00:00:00Z",
                f"2026-01-{index + 1:02d}T01:00:00Z",
                "a",
                "b",
                1,
                best_of=(1, 3, 5)[index % 3],
                team_a_name=f"A-{index}",
            )
            for index in range(9)
        ]
    )
    run = run_prequential_series(observations)
    assert np.isfinite(run.predictions["probability"]).all()
    assert run.predictions["probability"].between(0.0, 1.0, inclusive="neither").all()
    snapshot = run.model.snapshot()
    assert {"mean", "sigma", "rating_p05", "team", "home_league"}.issubset(
        snapshot.columns
    )
    assert {
        "comparison_component_id",
        "comparison_component_size",
        "cross_component_rankable",
    }.issubset(snapshot.columns)
    assert snapshot["comparison_component_id"].nunique() == 1
    assert snapshot["comparison_component_size"].eq(2).all()
    assert not snapshot["cross_component_rankable"].any()
    assert snapshot["sigma"].gt(0.0).all()
    assert snapshot["sigma_kind"].eq(
        "diagonal_filter_approximation_sd"
    ).all()
    assert snapshot["rating_p05_interpretation"].str.contains(
        "coverage has not been established"
    ).all()
    assert snapshot.loc[snapshot["team_key"].eq("a"), "team"].iloc[0] == "A-8"


def test_exact_model_and_config_hashes() -> None:
    config = SeriesDynamicBTConfig(
        team_variance_per_day=0.007,
        context_enabled=True,
    )
    root = Path(series_dynamic_bt.__file__).resolve().parents[2]
    dependencies = (
        Path(series_dynamic_bt.__file__),
        Path(series_dynamic_bt.__file__).with_name("hierarchical_bt.py"),
        Path(series_dynamic_bt.__file__).parents[1] / "etl" / "series_ledger.py",
        Path(series_dynamic_bt.__file__).parents[1] / "etl" / "competition.py",
        Path(series_dynamic_bt.__file__).parents[1] / "etl" / "aliases.py",
    )
    digest = hashlib.sha256()
    for path in sorted(
        (dependency.resolve() for dependency in dependencies),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    expected_code = digest.hexdigest()
    expected_config = hashlib.sha256(
        json.dumps(
            asdict(config),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    identity = model_identity(config)
    assert model_code_sha256() == expected_code
    assert config_sha256(config) == expected_config
    assert identity["model_code_sha256"] == expected_code
    assert identity["model_config_sha256"] == expected_config
    assert len(identity["model_code_sha256"]) == 64
    assert len(identity["model_config_sha256"]) == 64


def test_gate_fails_for_inconclusive_or_missing_calibration() -> None:
    comparisons = {
        ELO_BASELINE_ID: {"score": "log_loss", "decision": "inconclusive"},
        BASE_RATE_ID: {"score": "log_loss", "decision": "noninferior"},
    }
    metrics = {
        "n": 20,
        "ece": 0.04,
        "calibration_bins": [{"bin": 0, "n": 20}],
    }
    gate = evaluate_promotion_gate(comparisons, metrics)
    assert not gate["passed"]
    assert gate["status"] == "failed"
    assert "inconclusive" in " ".join(gate["reasons"])

    no_calibration = evaluate_promotion_gate(
        {
            ELO_BASELINE_ID: {"score": "log_loss", "decision": "noninferior"},
            BASE_RATE_ID: {"score": "log_loss", "decision": "superior"},
        },
        {"n": 20, "ece": np.nan, "calibration_bins": []},
    )
    assert not no_calibration["passed"]
    assert not no_calibration["calibration_reported"]


def test_tournament_is_validation_selected_and_final_test_is_format_stratified() -> None:
    outcomes = [1, 0, 1, 1, 0, 1, 0, 1, 0]
    maps = pd.DataFrame(
        [
            _bo1_map(
                f"series-{index}",
                f"2026-01-{index + 1:02d}T10:00:00Z",
                "A",
                "B" if index % 2 == 0 else "C",
                outcome,
            )
            for index, outcome in enumerate(outcomes)
        ]
    )
    spec = SeriesTournamentSpec(
        validation_start="2026-01-04T00:00:00Z",
        test_start="2026-01-07T00:00:00Z",
        bootstrap_replicates=100,
        moving_block_size=2,
        random_seed=7,
    )
    candidate_grid = (
        SeriesDynamicBTConfig(team_variance_per_day=0.001),
        SeriesDynamicBTConfig(team_variance_per_day=0.01),
    )
    elo_grid = (SeriesEloConfig(k_factor=12.0),)
    result = run_series_rating_tournament(
        maps,
        spec=spec,
        dynamic_candidates=candidate_grid,
        elo_candidates=elo_grid,
    )
    changed = maps.copy()
    changed.loc[changed["game_uid"].eq("series-8:1"), "y_blue_win"] = 1
    counterfactual = run_series_rating_tournament(
        changed,
        spec=spec,
        dynamic_candidates=(
            SeriesDynamicBTConfig(team_variance_per_day=0.001),
            SeriesDynamicBTConfig(team_variance_per_day=0.01),
        ),
        elo_candidates=elo_grid,
    )
    assert result.metadata["selection"]["test_labels_used_for_selection"] is False
    assert result.metadata["selection"]["hyperparameters_frozen_before_test"]
    assert result.metadata["excluded_model_family"].startswith("static hierarchy")
    assert "Bo1" in result.final_metrics[
        series_dynamic_bt.MODEL_ID
    ]["format_stratified_calibration"]
    assert {
        "mean",
        "sigma",
        "rating_p05",
        "team",
        "home_league",
    }.issubset(result.snapshot.columns)
    assert result.prediction_ledger["series_weight"].eq(1.0).all()
    pd.testing.assert_frame_equal(
        result.validation_scores, counterfactual.validation_scores
    )
    assert result.selected_config == counterfactual.selected_config
    original_last = result.prediction_ledger.sort_values(
        "prediction_time"
    ).iloc[-1]
    changed_last = counterfactual.prediction_ledger.sort_values(
        "prediction_time"
    ).iloc[-1]
    assert original_last[series_dynamic_bt.MODEL_ID] == changed_last[
        series_dynamic_bt.MODEL_ID
    ]
