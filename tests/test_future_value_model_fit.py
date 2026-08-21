from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    CURRENT_RATING_SIGNED_MAP_FEATURES,
    FutureValueSourceError,
    LEAGUEPEDIA_CROSSWALK_SOURCE,
    SIDE_LEVEL_TO_MODEL_FEATURE,
    _side_level_column,
    _map_model_frame,
    _missingness_metrics,
    _group_slice_metrics,
    _roster_change_labels,
    _frame_game_ids,
    _baseline_output_alignment,
    _baseline_source_binding,
    _bind_baseline_fold_series,
    _current_rating_method_comparison,
    _required_current_rating_comparison_blockers,
    _scope_series_cluster_audit_to_frame,
    _sequential_current_rating_baseline,
    _fold_level_imputation_values,
    build_future_value_design,
    build_time_decayed_prior_player_form,
    chronological_whole_series_folds,
    fit_future_value_model,
    fit_rank3_player_champion_role_atoms,
    evaluate_future_value,
    verify_phase_series_partition_binding,
)
from lol_kills.research.future_value_training import (
    FutureValueTrainingError,
    run_model_evaluation,
    verify_bridge_sources,
)
from lol_kills.research import future_value_training as training_module
from lol_kills.v2.tierlists.accepted_census import identity_sha256


METRICS = (
    "cs_per_min",
    "gold_per_min",
    "gold_share_pct",
    "damage_per_min",
    "damage_share_pct",
    "kda_role_weighted",
    "wards_per_min",
    "wards_cleared_per_min",
)
ROLES = ("top", "jungle", "mid", "bot", "support")


def _raw_source(game_count: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = []
    players = []
    for game_index in range(game_count):
        game_id = f"oe-api:{game_index + 1}"
        date = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=game_index)
        maps.append({"game_uid": game_id, "date": date, "y_blue_win": game_index % 2})
        for side_index, side in enumerate(("Blue", "Red")):
            for role_index, role in enumerate(("top", "jng", "mid", "bot", "sup")):
                value = 100.0 + game_index * 2 + side_index + role_index
                players.append(
                    {
                        "game_uid": game_id,
                        "date": date,
                        "side": side,
                        "position": role,
                        "playername": f"Player-{side_index}-{role_index}",
                        "playerid": f"oe:player:{side_index}:{role_index}",
                        "teamid": f"oe:team:{side_index}",
                        "champion": f"Champion-{role_index}",
                        "gamelength": 1800,
                        "cspm": value,
                        "dpm": value + 1,
                        "damageshare": 10 + role_index,
                        "wpm": 1 + role_index / 10,
                        "wcpm": 0.5 + role_index / 20,
                        "totalgold": 1000 + value,
                        "kills": role_index,
                        "deaths": 1,
                        "assists": role_index + 1,
                    }
                )
    return pd.DataFrame(maps), pd.DataFrame(players)


def _manual_form(game_count: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    maps = []
    rows = []
    for game_index in range(game_count):
        game_id = str(game_index + 1)
        date = pd.Timestamp("2026-01-01T00:00:00Z") + pd.Timedelta(days=game_index)
        maps.append(
            {
                "game_uid": game_id,
                "date": date,
                "y_blue_win": game_index % 2,
                "series_id": f"series-{game_index}",
            }
        )
        for side_index, side in enumerate(("blue", "red")):
            for role_index, role in enumerate(ROLES):
                player_id = f"oe:player:{side_index}:{role_index}"
                champion = f"Champion-{(role_index + game_index) % 6}"
                value = float(1 + role_index + game_index / 10 + side_index / 20)
                row = {
                    "game_id": game_id,
                    "date": date,
                    "side": side,
                    "role": role,
                    "player_id": player_id,
                    "team_id": f"oe:team:{side_index}",
                    "champion": champion,
                }
                for metric_index, metric in enumerate(METRICS):
                    row[f"prior_form_{metric}"] = value + metric_index / 10
                    row[f"prior_form_{metric}_support"] = game_index + 1
                    row[f"prior_form_{metric}_effective_support"] = float(game_index + 1)
                rows.append(row)
    return pd.DataFrame(maps), pd.DataFrame(rows)


def _source_receipt(game_ids: list[str], source_as_of: str = "2026-01-24T00:00:00Z") -> dict:
    game_ids = sorted(game_ids)
    source_files = {
        label: {"bytes": 1, "sha256": "0" * 64, "locator": f"fixture/{label}"}
        for label in ("maps", "players", "teams", "accepted_census")
    }
    payload = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": source_as_of,
        "source_game_count": len(game_ids),
        "source_identity_sha256": identity_sha256(game_ids),
        "accepted_game_ids": game_ids,
        "model_eligible_game_count": len(game_ids),
        "model_eligible_identity_sha256": identity_sha256(game_ids),
        "model_eligible_game_ids": game_ids,
        "source_rows": {},
        "source_extra_game_ids": {},
        "identity_coverage": {},
        "checkpoint_coverage": {},
        "model_exclusions": {},
        "source_files": source_files,
        "model_contract": {},
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    return payload


def test_time_decayed_form_excludes_same_timestamp_and_is_append_invariant() -> None:
    maps, players = _raw_source(3)
    # Make the second map share a timestamp with a third map.
    maps.loc[2, "date"] = maps.loc[1, "date"]
    players.loc[players["game_uid"].eq("oe-api:3"), "date"] = maps.loc[1, "date"]
    full = build_time_decayed_prior_player_form(maps, players, half_life_days=10)
    first_player = full[full["player_id"].eq("oe:player:0:0")].sort_values("date")
    assert pd.isna(first_player.iloc[0]["prior_form_cs_per_min"])
    assert first_player.iloc[1]["prior_form_cs_per_min"] == pytest.approx(100.0)
    assert first_player.iloc[2]["prior_form_cs_per_min"] == pytest.approx(
        first_player.iloc[1]["prior_form_cs_per_min"]
    )
    base = build_time_decayed_prior_player_form(
        maps.iloc[:2],
        players[players["game_uid"].isin(["oe-api:1", "oe-api:2"])],
        half_life_days=10,
    )
    pd.testing.assert_frame_equal(
        base.reset_index(drop=True),
        full[full["game_id"].isin(["1", "2"])].reset_index(drop=True),
        check_dtype=False,
    )


def test_rank3_fit_rejects_rows_at_exclusive_boundary() -> None:
    _maps, form = _manual_form(6)
    with pytest.raises(FutureValueSourceError, match="boundary or future"):
        fit_rank3_player_champion_role_atoms(
            form,
            train_game_ids=["1", "2"],
            fit_window_end="2026-01-02T00:00:00Z",
        )


def test_rank3_fit_rejects_a_missing_requested_training_game() -> None:
    _maps, form = _manual_form(6)
    with pytest.raises(FutureValueSourceError, match="training games are missing"):
        fit_rank3_player_champion_role_atoms(
            form,
            train_game_ids=["1", "999"],
        )


def test_rank3_fit_and_design_use_exact_five_rosters() -> None:
    maps, form = _manual_form(24)
    atom = fit_rank3_player_champion_role_atoms(form, train_game_ids=[str(index) for index in range(1, 19)])
    design = build_future_value_design(maps, form, atom)
    assert atom.rank == 3
    assert len(design) == len(maps)
    assert set("rank_3_player_atom_1 rank_3_player_atom_2 rank_3_player_atom_3".split()).issubset(design)
    game_id = "20"
    selected_form = form.loc[form["game_id"].eq(game_id)]
    expected_sum_difference = (
        selected_form.loc[
            selected_form["side"].eq("blue"), "prior_form_cs_per_min"
        ].sum()
        - selected_form.loc[
            selected_form["side"].eq("red"), "prior_form_cs_per_min"
        ].sum()
    )
    actual_sum_difference = design.loc[
        design["game_id"].eq(game_id), "player_form_cs_per_min"
    ].iloc[0]
    assert actual_sum_difference == pytest.approx(expected_sum_difference)
    broken = form.drop(form.index[0]).copy()
    with pytest.raises(FutureValueSourceError, match="exact five-player"):
        build_future_value_design(maps, broken, atom)
    broken_role = form.copy()
    broken_role.loc[broken_role.index[0], "role"] = "mid"
    with pytest.raises(FutureValueSourceError, match="duplicate game-side-role"):
        build_future_value_design(maps, broken_role, atom)


def test_rank3_fit_normalizes_frozen_jng_and_sup_roles() -> None:
    maps, form = _manual_form(24)
    form.loc[form["role"].eq("jungle"), "role"] = "jng"
    form.loc[form["role"].eq("support"), "role"] = "sup"
    for index, row in form.iterrows():
        blue_win = (int(row["game_id"]) - 1) % 2 == 1
        favorable = blue_win if row["side"] == "blue" else not blue_win
        form.loc[index, "champion"] = (
            f"Signal-{'favorable' if favorable else 'unfavorable'}-{row['role']}"
        )
    train_ids = [str(index) for index in range(1, 22)]
    atom = fit_rank3_player_champion_role_atoms(form, train_game_ids=train_ids)
    assert any(key.endswith("|jungle") for key in atom.champion_role_coordinates)
    assert any(key.endswith("|support") for key in atom.champion_role_coordinates)
    assert not any(key.endswith("|jng") for key in atom.champion_role_coordinates)
    assert not any(key.endswith("|sup") for key in atom.champion_role_coordinates)
    model, design = fit_future_value_model(
        maps,
        form,
        train_game_ids=train_ids,
        fit_window_end="2026-01-22T00:00:00Z",
        source_receipt=_source_receipt([str(index) for index in range(1, 25)]),
    )
    champion_columns = [
        _side_level_column(side, f"rank_3_champion_role_atom_{index}")
        for side in ("blue", "red")
        for index in range(1, 4)
    ]
    assert np.isfinite(design[champion_columns].to_numpy(dtype=float)).any()
    champion_coefficients = [
        model.coefficient_map[f"rank_3_champion_role_atom_{index}"]
        for index in range(1, 4)
    ]
    assert any(abs(value) > 1e-12 for value in champion_coefficients)
    broken = form.copy()
    broken.loc[broken.index[0], "role"] = pd.NA
    with pytest.raises(FutureValueSourceError, match="missing champion or role"):
        fit_rank3_player_champion_role_atoms(broken, train_game_ids=train_ids)
    for missing_champion in (None, np.nan, pd.NA):
        broken = form.copy()
        broken.loc[broken.index[0], "champion"] = missing_champion
        with pytest.raises(FutureValueSourceError, match="missing champion or role"):
            fit_rank3_player_champion_role_atoms(broken, train_game_ids=train_ids)


def test_imputation_fails_closed_for_all_missing_non_centered_features() -> None:
    maps, form = _manual_form(24)
    atom = fit_rank3_player_champion_role_atoms(
        form, train_game_ids=[str(index) for index in range(1, 22)]
    )
    design = build_future_value_design(maps, form, atom)
    broken = design.copy()
    for side in ("blue", "red"):
        broken[_side_level_column(side, "team_prior_win")] = np.nan
    with pytest.raises(FutureValueSourceError, match="non-centered.*all missing"):
        _fold_level_imputation_values(broken)
    centered = design.copy()
    atom_sources = [
        f"rank_3_champion_role_atom_{index}" for index in range(1, 4)
    ]
    for source_name in atom_sources:
        for side in ("blue", "red"):
            centered[_side_level_column(side, source_name)] = np.nan
    imputation = _fold_level_imputation_values(centered)
    for source_name in atom_sources:
        position = list(SIDE_LEVEL_TO_MODEL_FEATURE).index(source_name)
        assert imputation[position] == 0.0


def test_future_value_fit_returns_fitted_metric_weights_and_prediction() -> None:
    maps, form = _manual_form(24)
    model, design = fit_future_value_model(
        maps,
        form,
        train_game_ids=[str(index) for index in range(1, 22)],
        fit_window_end="2026-01-22T00:00:00Z",
        source_receipt=_source_receipt([str(index) for index in range(1, 25)]),
    )
    assert set(model.metric_weights) == {f"player_form_{metric}" for metric in METRICS}
    prediction = model.predict_probability(design.iloc[21:])
    assert prediction.notna().all()
    assert ((prediction >= 0) & (prediction <= 1)).all()
    assert model.receipt()["authority"]["public_probability"] is False
    assert model.receipt()["source_binding"]["accepted_game_ids"] == sorted(
        str(index) for index in range(1, 25)
    )
    parameters = model.parameter_receipt()
    assert parameters["intercept"] == pytest.approx(model.intercept)
    assert set(parameters["feature_means"]) == set(model.feature_names)
    assert set(parameters["feature_scales"]) == set(model.feature_names)
    assert len(parameters["parameter_sha256"]) == 64
    assert len(parameters["rank_3"]["parameter_sha256"]) == 64
    assert parameters["rank_3"]["champion_role_coordinates"]
    assert parameters["intercept"] == 0.0
    assert set(parameters["fold_local_side_imputation"]) == set(
        SIDE_LEVEL_TO_MODEL_FEATURE
    )
    assert parameters["antisymmetric_fit"]["fit_intercept"] is False


def test_fold_local_imputation_predicts_incomplete_rows_and_preserves_side_swap() -> None:
    maps, form = _manual_form(24)
    form.loc[
        form["game_id"].eq("10") & form["side"].eq("red") & form["role"].eq("mid"),
        "prior_form_gold_per_min",
    ] = np.nan
    form.loc[
        form["game_id"].eq("22") & form["side"].eq("blue") & form["role"].eq("top"),
        "prior_form_cs_per_min",
    ] = np.nan
    receipt = _source_receipt([str(index) for index in range(1, 25)])
    model, design = fit_future_value_model(
        maps,
        form,
        train_game_ids=[str(index) for index in range(1, 22)],
        fit_window_end="2026-01-22T00:00:00Z",
        source_receipt=receipt,
    )
    incomplete = design.loc[design["game_id"].eq("22")].copy()
    assert incomplete["model_features_complete"].eq(False).all()
    probability = model.predict_probability(incomplete)
    assert probability.notna().all()
    missingness = _missingness_metrics(
        incomplete,
        incomplete["target"].astype(float),
        probability,
    )
    assert missingness["status"] == "imputed_only"
    assert missingness["complete_case_metrics"]["rows"] == 0
    assert missingness["incomplete_case_metrics"]["rows"] == 1
    assert "complete_case_validation_rows_missing" in missingness["blockers"]

    swapped = incomplete.copy()
    for source_name in SIDE_LEVEL_TO_MODEL_FEATURE:
        blue_column = _side_level_column("blue", source_name)
        red_column = _side_level_column("red", source_name)
        blue = swapped[blue_column].copy()
        swapped[blue_column] = swapped[red_column].to_numpy()
        swapped[red_column] = blue.to_numpy()
    swapped_probability = model.predict_probability(swapped)
    assert swapped_probability.iloc[0] == pytest.approx(
        1.0 - probability.iloc[0], abs=1e-12
    )

    changed_future = form.copy()
    future_mask = changed_future["game_id"].eq("24")
    changed_future.loc[future_mask, "prior_form_cs_per_min"] = 1_000_000.0
    future_model, _ = fit_future_value_model(
        maps,
        changed_future,
        train_game_ids=[str(index) for index in range(1, 22)],
        fit_window_end="2026-01-22T00:00:00Z",
        source_receipt=receipt,
    )
    np.testing.assert_array_equal(model.imputation_values, future_model.imputation_values)
    np.testing.assert_allclose(model.coefficients, future_model.coefficients, atol=0.0)
    assert model.regularization_selection == future_model.regularization_selection
    assert model.regularization_selection["method"] == (
        "nested_chronological_whole_series_log_loss"
    )
    assert model.regularization_selection["selected_c"] in model.regularization_selection[
        "candidate_grid"
    ]


def test_player_value_components_reconstruct_full_logit_with_support_records() -> None:
    maps, form = _manual_form(24)
    model, design = fit_future_value_model(
        maps,
        form,
        train_game_ids=[str(index) for index in range(1, 22)],
        fit_window_end="2026-01-22T00:00:00Z",
        source_receipt=_source_receipt([str(index) for index in range(1, 25)]),
    )
    selected_design = design.loc[design["game_id"].isin(["22", "23", "24"])].copy()
    components = model.player_value_logit(form, selected_design)
    reconstructed = (
        components["player_value_logit"]
        + components["team_context_logit"]
        + components["data_quality_logit"]
    )
    np.testing.assert_allclose(
        reconstructed,
        components["full_model_logit"],
        rtol=0.0,
        atol=1e-12,
    )
    assert components["component_reconstruction_error"].abs().max() <= 1e-12
    assert components["player_support_records"].map(len).eq(10).all()
    first_record = components.iloc[0]["player_support_records"][0]
    assert set(first_record["metric_support"]) == set(METRICS)
    assert set(first_record["metric_effective_support"]) == set(METRICS)
    assert "minimum_effective_support" in first_record
    assert "champion_role_atom_support" in first_record
    assert "missing_feature_names" in first_record
    assert first_record["support_status"] in {"adequate", "sparse", "missing"}
    assert model.optimizer_evidence["success"] is True
    assert model.optimizer_evidence["finite_coefficients"] is True
    assert model.regularization_selection["inner_atom_parameter_sha256"]
    assert model.regularization_selection["inner_transform_sha256"]
    assert all(
        row["optimizer"]["success"] and len(row["prediction_sha256"]) == 64
        for row in model.regularization_selection["candidate_scores"]
    )


def test_fit_requires_a_verified_source_receipt() -> None:
    maps, form = _manual_form(24)
    with pytest.raises(FutureValueSourceError, match="verified source receipt is required"):
        fit_future_value_model(
            maps,
            form,
            train_game_ids=[str(index) for index in range(1, 22)],
            fit_window_end="2026-01-22T00:00:00Z",
        )


def test_evaluation_cross_binds_series_receipt_to_validated_source() -> None:
    maps, players = _raw_source(60)
    maps["grid_series_id"] = [f"series-{index}" for index in range(len(maps))]
    maps["blue_teamid"] = "oe:team:0"
    maps["red_teamid"] = "oe:team:1"
    game_ids = list(_frame_game_ids(maps, "maps"))
    source = _source_receipt(game_ids, source_as_of=maps["date"].max().isoformat())
    assignment_rows = [
        {"game_id": str(row.game_uid), "series_id": str(row.grid_series_id)}
        for row in maps.sort_values("game_uid").itertuples(index=False)
    ]
    pair_rows = [
        {
            "game_id": str(row.game_uid),
            "series_id": str(row.grid_series_id),
            "team_pair": "oe:team:0|oe:team:1",
        }
        for row in maps.sort_values("game_uid").itertuples(index=False)
    ]
    series_receipt = {
        "source_type": "verified_grid_series",
        "series_column": "grid_series_id",
        "game_count": len(maps),
        "game_identity_sha256": identity_sha256(game_ids),
        "series_assignment_sha256": hashlib.sha256(
            json.dumps(assignment_rows, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "series_pair_assignment_sha256": hashlib.sha256(
            json.dumps(pair_rows, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "source_receipt_sha256": source["receipt_sha256"],
    }
    series_receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(series_receipt, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    maps.attrs["verified_series_receipt"] = series_receipt
    forged_source = dict(source)
    forged_source["source_files"] = {
        "fixture": {"bytes": 2, "sha256": "1" * 64}
    }
    forged_source.pop("receipt_sha256")
    forged_source["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            forged_source, separators=(",", ":"), sort_keys=True
        ).encode()
    ).hexdigest()
    with pytest.raises(FutureValueSourceError, match="file bindings are incomplete"):
        evaluate_future_value(
            maps,
            players,
            n_folds=1,
            source_receipt=forged_source,
        )


def test_partial_authoritative_series_ids_use_a_proxy_cluster() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": "1",
                "date": "2026-01-01T00:00:00Z",
                "y_blue_win": 1,
                "series_id": "series-1",
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
            {
                "game_uid": "2",
                "date": "2026-01-01T01:00:00Z",
                "y_blue_win": 0,
                "series_id": None,
                "blue_team_key": "blue",
                "red_team_key": "red",
            },
        ]
    )
    frame = _map_model_frame(maps)
    assert frame.attrs["series_cluster_source"] == "conservative_series_superset"
    assert frame["series_id"].str.startswith("proxy:").all()


def test_proxy_series_keeps_repeated_team_tournament_rows_together() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": f"g{index}",
                "date": f"2026-01-{index:02d}T00:00:00Z",
                "y_blue_win": index % 2,
                "league": "LEC",
                "tournament": "Spring",
                "blue_team_key": "team-a",
                "red_team_key": "team-b",
            }
            for index in range(1, 4)
        ]
    )
    frame = _map_model_frame(maps)
    assert frame["series_id"].nunique() == 1
    audit = frame.attrs["series_cluster_audit"]
    assert audit["source"] == "conservative_series_superset"
    assert audit["colliding_cluster_count"] == 1
    assert audit["collision_extra_map_count"] == 2


def test_bare_source_neutral_series_id_does_not_claim_authority() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": f"g{index}",
                "date": f"2026-01-0{index}T00:00:00Z",
                "y_blue_win": index % 2,
                "series_id": "unverified-series",
                "league": "LEC",
                "blue_teamid": "oe:team:a",
                "red_teamid": "oe:team:b",
            }
            for index in range(1, 3)
        ]
    )
    frame = _map_model_frame(maps)
    assert frame.attrs["series_cluster_source"] == "conservative_series_superset"
    assert frame.attrs["series_cluster_audit"]["authoritative"] is False
    assert frame["series_id"].str.startswith("proxy:").all()


def test_authoritative_series_requires_source_bound_assignment_receipt() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": f"g{index}",
                "date": f"2026-01-0{index}T00:00:00Z",
                "y_blue_win": index % 2,
                "grid_series_id": "series-a" if index < 3 else "series-b",
                "blue_teamid": "oe:team:a",
                "red_teamid": "oe:team:b",
            }
            for index in range(1, 4)
        ]
    )
    game_ids = list(maps["game_uid"])
    source_receipt_sha256 = "a" * 64
    assignment_rows = [
        {"game_id": row.game_uid, "series_id": row.grid_series_id}
        for row in maps.sort_values("game_uid").itertuples(index=False)
    ]
    pair_rows = [
        {
            "game_id": row.game_uid,
            "series_id": row.grid_series_id,
            "team_pair": "oe:team:a|oe:team:b",
        }
        for row in maps.sort_values("game_uid").itertuples(index=False)
    ]
    payload = {
        "source_type": "verified_grid_series",
        "series_column": "grid_series_id",
        "game_count": len(maps),
        "game_identity_sha256": identity_sha256(game_ids),
        "series_assignment_sha256": hashlib.sha256(
            json.dumps(assignment_rows, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "series_pair_assignment_sha256": hashlib.sha256(
            json.dumps(pair_rows, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest(),
        "source_receipt_sha256": source_receipt_sha256,
    }
    payload["receipt_sha256"] = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    maps.attrs["verified_series_receipt"] = payload
    frame = _map_model_frame(
        maps, verified_source_receipt_sha256=source_receipt_sha256
    )
    audit = frame.attrs["series_cluster_audit"]
    assert audit["authoritative"] is True
    assert audit["cluster_count"] == 2
    assert audit["colliding_cluster_count"] == 1
    assert audit["collision_extra_map_count"] == 1
    assert audit["max_cluster_size"] == 2
    fold_maps, fold_ids, fold_cluster, series_column = _bind_baseline_fold_series(
        maps,
        requested_game_ids={"g1", "g2"},
        full_map_frame=frame,
    )
    assert series_column == "grid_series_id"
    assert list(fold_maps["grid_series_id"]) == ["series-a", "series-a"]
    assert list(fold_ids) == ["g1", "g2"]
    assert fold_cluster["authoritative"] is True

    mutated = maps.copy()
    mutated.attrs = dict(maps.attrs)
    mutated.loc[0, "grid_series_id"] = "forged"
    with pytest.raises(FutureValueSourceError, match="assignments changed"):
        _map_model_frame(
            mutated, verified_source_receipt_sha256=source_receipt_sha256
        )

    unbound = maps.copy()
    unbound.attrs = {"verified_series_receipt": payload}
    unbound_frame = _map_model_frame(unbound)
    assert unbound_frame.attrs["series_cluster_audit"]["authoritative"] is False
    with pytest.raises(FutureValueSourceError, match="does not bind"):
        _map_model_frame(
            unbound, verified_source_receipt_sha256="b" * 64
        )


def test_proxy_series_prefers_stable_team_ids_over_alias_keys() -> None:
    maps = pd.DataFrame(
        [
            {
                "game_uid": "g1",
                "date": "2026-01-01T00:00:00Z",
                "y_blue_win": 1,
                "league": "LEC",
                "blue_teamid": "oe:team:a",
                "red_teamid": "oe:team:b",
                "blue_team_key": "old-alias-a",
                "red_team_key": "old-alias-b",
            },
            {
                "game_uid": "g2",
                "date": "2026-01-02T00:00:00Z",
                "y_blue_win": 0,
                "league": "LEC",
                "blue_teamid": "oe:team:a",
                "red_teamid": "oe:team:b",
                "blue_team_key": "new-alias-a",
                "red_team_key": "new-alias-b",
            },
        ]
    )
    frame = _map_model_frame(maps)
    assert frame["series_id"].nunique() == 1
    audit = frame.attrs["series_cluster_audit"]
    assert audit["stable_team_ids"] is True
    assert audit["team_identity_columns"] == ["blue_teamid", "red_teamid"]


def _phase_partition_fixture(tmp_path, game_ids: list[str]):
    maps, _players = _manual_form(len(game_ids))
    maps["game_uid"] = game_ids
    frame = _map_model_frame(maps)
    hashes = {
        "mapping_sha256": "a" * 64,
        "crosswalk_sha256": "b" * 64,
        "artifact_sha256": "c" * 64,
        "receipt_sha256": "d" * 64,
        "receipt_file_sha256": "e" * 64,
    }
    frame.attrs["series_cluster_audit"] = {
        "crosswalk_assignment_sha256": hashes["mapping_sha256"],
        "crosswalk_sha256": hashes["crosswalk_sha256"],
        "crosswalk_artifact_sha256": hashes["artifact_sha256"],
        "crosswalk_receipt_sha256": hashes["receipt_sha256"],
        "partial_series_blocker": True,
        "retained_proxy_game_count": 1,
        "retained_proxy_cluster_count": 1,
    }
    frame.attrs["crosswalk_receipt_file_sha256"] = hashes["receipt_file_sha256"]
    source = _source_receipt(game_ids)
    assignment_rows = [
        {"game_id": str(game_id), "series_id": str(series_id)}
        for game_id, series_id in zip(frame["game_id"], frame["series_id"])
    ]
    assignment_rows.sort(key=lambda row: row["game_id"])
    assignment_sha256 = hashlib.sha256(
        json.dumps(
            assignment_rows,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    partition = {
        **hashes,
        "status": "comparable",
        "eligible_game_count": len(game_ids),
        "eligible_identity_sha256": identity_sha256(game_ids),
        "eligible_assignment_sha256": assignment_sha256,
        "reference_game_count": len(game_ids),
        "reference_identity_sha256": identity_sha256(game_ids),
        "reference_assignment_sha256": assignment_sha256,
        "reference_assignment_match": True,
        "proxy_authority_blocker": True,
    }
    artifact_path = tmp_path / "phase-candidate.json"
    artifact_path.write_text(
        json.dumps(
            {
                "source_receipt_sha256": source["receipt_sha256"],
                "cross_model_series_partition": partition,
                "series_partition_reference_game_count": len(game_ids),
                "series_partition_reference_identity_sha256": identity_sha256(game_ids),
                "series_partition_reference_assignment_sha256": assignment_sha256,
                "series_partition_proxy_authority_blocker": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact_record = {
        "path": str(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }
    receipt_path = tmp_path / "phase-run-receipt.json"
    receipt_payload = {
        "source": {
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "partition": partition,
        "outputs": {"candidate": artifact_record},
    }
    receipt_path.write_text(json.dumps(receipt_payload, sort_keys=True), encoding="utf-8")
    receipt_record = {
        "path": str(receipt_path),
        "bytes": receipt_path.stat().st_size,
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
    }
    binding = {
        "phase_artifact": artifact_record,
        "phase_receipt": receipt_record,
        "phase_artifact_sha256": artifact_record["sha256"],
        "phase_receipt_file_sha256": receipt_record["sha256"],
        "eligible_game_count": len(game_ids),
        "eligible_identity_sha256": identity_sha256(game_ids),
        "eligible_assignment_sha256": assignment_sha256,
        "reference_game_count": len(game_ids),
        "reference_identity_sha256": identity_sha256(game_ids),
        "reference_assignment_sha256": assignment_sha256,
        "source_receipt_sha256": source["receipt_sha256"],
    }
    source_path = tmp_path / "source-receipt.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    return (
        frame,
        source,
        binding,
        artifact_record["sha256"],
        receipt_record["sha256"],
        source_path,
        source_hash,
    )


def test_phase_partition_binding_requires_byte_bound_receipt_and_matches_rating_frame(tmp_path) -> None:
    game_ids = ["1", "2", "3", "4"]
    (
        frame,
        source,
        binding,
        artifact_hash,
        receipt_hash,
        source_path,
        source_file_hash,
    ) = _phase_partition_fixture(
        tmp_path, game_ids
    )
    verified = verify_phase_series_partition_binding(
        frame,
        source,
        binding,
        expected_phase_artifact_sha256=artifact_hash,
        expected_phase_receipt_file_sha256=receipt_hash,
        source_receipt_path=source_path,
        source_receipt_file_sha256=source_file_hash,
    )
    assert verified["status"] == "verified"
    assert verified["cross_model_partition_status"] == "comparable"
    assert verified["proxy_authority_blocker"] is True


def test_phase_partition_binding_fails_closed_on_artifact_mutation(tmp_path) -> None:
    game_ids = ["1", "2", "3", "4"]
    (
        frame,
        source,
        binding,
        artifact_hash,
        receipt_hash,
        source_path,
        source_file_hash,
    ) = _phase_partition_fixture(
        tmp_path, game_ids
    )
    artifact_path = binding["phase_artifact"]["path"]
    Path(artifact_path).write_text(
        Path(artifact_path).read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )
    with pytest.raises(FutureValueSourceError, match="phase partition artifact"):
        verify_phase_series_partition_binding(
            frame,
            source,
            binding,
            expected_phase_artifact_sha256=artifact_hash,
            expected_phase_receipt_file_sha256=receipt_hash,
            source_receipt_path=source_path,
            source_receipt_file_sha256=source_file_hash,
        )


def test_phase_partition_binding_rejects_nested_flattened_conflict(tmp_path) -> None:
    frame, source, binding, _artifact_hash, receipt_hash, source_path, source_file_hash = (
        _phase_partition_fixture(tmp_path, ["1", "2", "3", "4"])
    )
    artifact_path = Path(binding["phase_artifact"]["path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["series_partition_eligible_game_count"] = 999
    artifact_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    binding["phase_artifact"]["bytes"] = artifact_path.stat().st_size
    binding["phase_artifact"]["sha256"] = artifact_hash
    binding["phase_artifact_sha256"] = artifact_hash
    with pytest.raises(FutureValueSourceError, match="conflicting eligible_game_count"):
        verify_phase_series_partition_binding(
            frame,
            source,
            binding,
            expected_phase_artifact_sha256=artifact_hash,
            expected_phase_receipt_file_sha256=receipt_hash,
            source_receipt_path=source_path,
            source_receipt_file_sha256=source_file_hash,
        )


def test_phase_partition_binding_requires_full_reference_identity(tmp_path) -> None:
    frame, source, binding, _artifact_hash, receipt_hash, source_path, source_file_hash = (
        _phase_partition_fixture(tmp_path, ["1", "2", "3", "4"])
    )
    artifact_path = Path(binding["phase_artifact"]["path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    partition = payload["cross_model_series_partition"]
    partition.pop("reference_identity_sha256")
    payload.pop("series_partition_reference_identity_sha256")
    artifact_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    binding["phase_artifact"]["bytes"] = artifact_path.stat().st_size
    binding["phase_artifact"]["sha256"] = artifact_hash
    binding["phase_artifact_sha256"] = artifact_hash
    with pytest.raises(FutureValueSourceError, match="reference_identity_sha256"):
        verify_phase_series_partition_binding(
            frame,
            source,
            binding,
            expected_phase_artifact_sha256=artifact_hash,
            expected_phase_receipt_file_sha256=receipt_hash,
            source_receipt_path=source_path,
            source_receipt_file_sha256=source_file_hash,
        )


def test_phase_partition_binding_requires_durable_source_receipt(tmp_path) -> None:
    frame, source, binding, artifact_hash, receipt_hash, _source_path, _source_file_hash = (
        _phase_partition_fixture(tmp_path, ["1", "2", "3", "4"])
    )
    with pytest.raises(FutureValueSourceError, match="durable source receipt"):
        verify_phase_series_partition_binding(
            frame,
            source,
            binding,
            expected_phase_artifact_sha256=artifact_hash,
            expected_phase_receipt_file_sha256=receipt_hash,
        )


def test_phase_partition_binding_requires_proxy_blocker_in_every_copy(tmp_path) -> None:
    frame, source, binding, _artifact_hash, receipt_hash, source_path, source_file_hash = (
        _phase_partition_fixture(tmp_path, ["1", "2", "3", "4"])
    )
    artifact_path = Path(binding["phase_artifact"]["path"])
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload["cross_model_series_partition"].pop("proxy_authority_blocker")
    payload.pop("series_partition_proxy_authority_blocker")
    artifact_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    binding["phase_artifact"]["bytes"] = artifact_path.stat().st_size
    binding["phase_artifact"]["sha256"] = artifact_hash
    binding["phase_artifact_sha256"] = artifact_hash
    with pytest.raises(FutureValueSourceError, match="proxy_authority_blocker"):
        verify_phase_series_partition_binding(
            frame,
            source,
            binding,
            expected_phase_artifact_sha256=artifact_hash,
            expected_phase_receipt_file_sha256=receipt_hash,
            source_receipt_path=source_path,
            source_receipt_file_sha256=source_file_hash,
        )


def test_evaluation_pairs_candidate_and_baseline_on_identical_game_ids() -> None:
    maps, players = _raw_source(60)
    game_ids = list(_frame_game_ids(maps, "maps"))
    receipt = _source_receipt(game_ids, source_as_of=maps["date"].max().isoformat())
    result = evaluate_future_value(
        maps,
        players,
        n_folds=1,
        source_receipt=receipt,
    )
    fold = result["folds"][0]
    assert fold["candidate"]["rows"] == fold["intercept_baseline"]["rows"]
    assert fold["candidate"]["rows"] == fold["paired_rows"]
    assert fold["paired_rows"] == fold["paired_game_id_count"]
    assert fold["calibration"]["status"] == "available"
    assert fold["calibration"]["rows"] == fold["paired_rows"]
    assert fold["missingness"]["status"] == "available"
    assert fold["side_swap"]["status"] == "available"
    assert fold["side_swap"]["within_tolerance"] is True
    assert fold["side_swap"]["max_probability_complement_error"] <= 1e-12
    assert fold["side_swap"]["blockers"] == []
    assert fold["imputation_policy"]["all_missing_non_centered_features"] == (
        "fail_closed"
    )
    components = fold["component_evidence"]
    assert components["row_count"] == fold["paired_rows"]
    assert components["maximum_absolute_reconstruction_error"] <= 1e-12
    assert len(components["sha256"]) == 64
    assert all(len(row["player_support_records"]) == 10 for row in components["rows"])
    assert result["evaluation"]["component_reconstruction_audit"]["status"] == (
        "passed"
    )
    assert fold["regional_transfer"]["status"] == "unavailable"
    assert fold["patch_transfer"]["status"] == "unavailable"
    assert fold["tournament_boundary"]["status"] == "unavailable"
    assert "regional_transfer_slice_missing" in result["blockers"]
    assert "patch_transfer_slice_missing" in result["blockers"]
    assert "tournament_boundary_slice_missing" in result["blockers"]
    assert result["evaluation"]["pooled_calibration"]["rows"] == fold["paired_rows"]
    ledger = result["prediction_ledger"]
    assert ledger["row_count"] == fold["validation_game_id_count"]
    assert ledger["columns"] == [
        "fold",
        "game_id",
        "target",
        "candidate",
        "candidate_raw_probability",
        "candidate_raw_logit",
        "candidate_calibrated_logit",
        "calibration_slope",
            "intercept",
            "sequential_player_elo",
            "sequential_current_rating",
            "hierarchical_bt",
        "minimum_metric_support",
        "minimum_effective_support",
        "minimum_atom_support",
        "missing_feature_names",
        "support_status",
    ]
    assert ledger["sha256"] == hashlib.sha256(
        json.dumps(
            ledger["rows"],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def test_baseline_output_alignment_reports_missing_and_extra_ids() -> None:
    validation = pd.DataFrame({"game_id": ["g1", "g2", "g3"]})
    output = pd.DataFrame(
        {
            "game_uid": ["g1", "g3", "g-extra"],
            "probability": [0.2, 0.8, 0.5],
        }
    )
    aligned, report = _baseline_output_alignment(
        validation,
        output,
        game_id_column="game_uid",
        probability_column="probability",
        method="research_baseline",
    )
    assert aligned.iloc[0] == pytest.approx(0.2)
    assert pd.isna(aligned.iloc[1])
    assert aligned.iloc[2] == pytest.approx(0.8)
    assert report["status"] == "partial"
    assert report["missing_game_ids"] == ["g2"]
    assert report["extra_game_ids"] == ["g-extra"]
    assert report["blockers"] == [
        "research_baseline_coverage_incomplete",
        "research_baseline_extra_prediction_ids",
    ]


def test_current_rating_methods_keep_partial_bt_out_of_shared_cohort() -> None:
    validation = pd.DataFrame(
        {"game_id": ["g1", "g2", "g3", "g4"]}
    )
    target = pd.Series([0.0, 1.0, 1.0, 0.0], index=validation.index)
    paired_mask = pd.Series([True, True, True, True], index=validation.index)
    sequential = pd.Series([0.2, 0.8, 0.7, 0.3], index=validation.index)
    sequential_current = pd.Series([0.22, 0.78, 0.68, 0.32], index=validation.index)
    hierarchical = pd.Series([0.25, 0.75, np.nan, np.nan], index=validation.index)
    reports = {
        "sequential_player_elo": {
            "status": "available",
            "blockers": [],
            "source_binding": {"status": "available", "blockers": []},
        },
        "hierarchical_bt": {
            "status": "partial",
            "blockers": ["hierarchical_bt_coverage_incomplete"],
            "exclusion_reason": "validation rows with unseen teams are excluded",
            "source_binding": {"status": "available", "blockers": []},
        },
        "sequential_current_rating": {
            "status": "available",
            "blockers": [],
            "source_binding": {"status": "available", "blockers": []},
        },
    }

    evidence = _current_rating_method_comparison(
        validation,
        target,
        paired_mask,
        sequential,
        hierarchical,
        reports,
        sequential_current,
    )

    assert evidence["method_specific"]["sequential_player_elo"]["status"] == (
        "available"
    )
    assert evidence["method_specific"]["sequential_player_elo"]["scored_rows"] == 4
    assert evidence["method_specific"]["sequential_player_elo"]["missing_game_ids"] == []
    current = evidence["method_specific"]["sequential_current_rating"]
    assert current["status"] == "available"
    assert current["scored_rows"] == 4
    assert current["missing_game_ids"] == []
    bt = evidence["method_specific"]["hierarchical_bt"]
    assert bt["status"] == "partial"
    assert bt["scored_rows"] == 2
    assert bt["missing_game_ids"] == ["g3", "g4"]
    assert bt["exclusion_reason"] == "validation rows with unseen teams are excluded"
    assert evidence["common_all_method"]["status"] == "blocked"
    assert evidence["common_all_method"]["rows"] == 2
    assert evidence["common_all_method"]["game_ids"] == ["g1", "g2"]
    assert "current_rating_row_id_parity_incomplete" in evidence["blockers"]
    assert "hierarchical_bt_coverage_incomplete" in evidence["blockers"]
    assert pd.isna(hierarchical.iloc[2])


def test_partial_bt_does_not_block_complete_current_rating_controls() -> None:
    reports = [
        {
            "method_specific": {
                "sequential_player_elo": {"status": "available"},
                "sequential_current_rating": {"status": "available"},
                "hierarchical_bt": {"status": "partial"},
            }
        }
    ]

    assert _required_current_rating_comparison_blockers(reports) == []

    reports[0]["method_specific"]["sequential_player_elo"]["status"] = "partial"
    assert _required_current_rating_comparison_blockers(reports) == [
        "current_player_rating_comparison_missing"
    ]


def test_series_audit_scopes_verified_model_frame_and_keeps_full_source_audit() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "series_id": [
                "leaguepedia:series-a",
                "leaguepedia:series-a",
                "proxy:league|tournament|teams",
            ],
        }
    )
    frame.attrs["series_cluster_source"] = LEAGUEPEDIA_CROSSWALK_SOURCE
    frame.attrs["series_cluster_audit"] = {
        "source": "leaguepedia_crosswalk",
        "authoritative": False,
        "map_count": 7,
        "cluster_count": 4,
    }
    frame.attrs["verified_leaguepedia_series_crosswalk"] = {
        "mapped_game_ids": ["g1", "g2", "g4"],
    }

    scoped = _scope_series_cluster_audit_to_frame(frame)

    assert scoped.attrs["full_source_series_cluster_audit"]["map_count"] == 7
    audit = scoped.attrs["series_cluster_audit"]
    assert audit["scope"] == "model_eligible_census"
    assert audit["map_count"] == 3
    assert audit["full_source_map_count"] == 7
    assert audit["mapped_game_count"] == 2
    assert audit["promoted_game_count"] == 2
    assert audit["retained_proxy_game_count"] == 1
    assert audit["cluster_count"] == 2


def test_series_audit_uses_row_bound_crosswalk_assignments() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "series_id": [
                "leaguepedia:series-a",
                "proxy:league|tournament|teams",
                "proxy:other|tournament|teams",
            ],
            "_series_crosswalk_mapped": [True, True, False],
            "_series_crosswalk_assignment": ["series-a", "series-b", pd.NA],
        }
    )
    frame.attrs["series_cluster_source"] = LEAGUEPEDIA_CROSSWALK_SOURCE
    frame.attrs["series_cluster_audit"] = {
        "source": "leaguepedia_crosswalk",
        "authoritative": False,
        "map_count": 7,
        "mapped_game_count": 5,
        "unmatched_game_count": 2,
    }

    scoped = _scope_series_cluster_audit_to_frame(frame)

    audit = scoped.attrs["series_cluster_audit"]
    assert audit["mapped_game_count"] == 2
    assert audit["unmatched_game_count"] == 1
    assert audit["mapped_series_count"] == 2
    assert "_series_crosswalk_mapped" not in scoped.columns
    assert "_series_crosswalk_assignment" not in scoped.columns


def test_series_audit_rejects_mixed_partition_without_row_binding() -> None:
    frame = pd.DataFrame(
        {
            "game_id": ["g1"],
            "series_id": ["leaguepedia:series-a"],
        }
    )
    frame.attrs["series_cluster_source"] = LEAGUEPEDIA_CROSSWALK_SOURCE
    frame.attrs["series_cluster_audit"] = {
        "source": "leaguepedia_crosswalk",
        "authoritative": False,
        "map_count": 1,
    }

    with pytest.raises(FutureValueSourceError, match="crosswalk row binding"):
        _scope_series_cluster_audit_to_frame(frame)


def test_sequential_current_rating_uses_bound_team_logit(monkeypatch: pytest.MonkeyPatch) -> None:
    validation = pd.DataFrame({"game_id": ["g3", "g4"]})
    ledger = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3", "g4"],
            "date": pd.date_range("2026-01-01", periods=4, tz="UTC"),
            "series_id": ["s1", "s2", "s3", "s4"],
            "base_team_logit": [0.0, 0.1, 0.8, -0.8],
            "team_rating_diff_scaled": [0.0, 0.1, 0.2, -0.2],
            "base_player_logit": [0.0, 0.1, 0.2, -0.2],
            "player_rating_diff_scaled": [0.0, 0.1, 0.2, -0.2],
        }
    )
    ledger.attrs["feature_names"] = list(CURRENT_RATING_SIGNED_MAP_FEATURES)
    ledger.attrs["schema_version"] = "scryglass:future-value-rating-ledger:v1"

    monkeypatch.setattr(
        "lol_kills.research.future_value_rating.validate_rating_feature_ledger",
        lambda *args, **kwargs: ledger,
    )
    probabilities, report = _sequential_current_rating_baseline(
        validation,
        ledger,
        train_game_ids=("g1", "g2"),
        validation_game_ids=("g3", "g4"),
        strict_cutoff="2026-01-03T00:00:00Z",
        source_receipt={"receipt_sha256": "a" * 64},
    )

    assert report["status"] == "available"
    assert report["probability_feature"] == "base_team_logit"
    assert probabilities.iloc[0] == pytest.approx(1.0 / (1.0 + np.exp(-0.8)))
    assert probabilities.iloc[1] == pytest.approx(1.0 / (1.0 + np.exp(0.8)))


def test_hierarchical_binding_requires_the_declared_proxy_series_receipt() -> None:
    source = _source_receipt(["g1", "g2", "g3", "g4"])
    train_ids = ["g1", "g2"]
    validation_ids = ["g3", "g4"]
    baseline = {
        "source": {
            "receipt_sha256": source["receipt_sha256"],
            "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
        },
        "train_receipt": {"identity_sha256": identity_sha256(train_ids)},
        "validation_receipt": {"identity_sha256": identity_sha256(validation_ids)},
        "implementation_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "series_identity": {
            "source_types": ["conservative_series_superset"],
            "authoritative": False,
        },
        "fit": {
            "optimizer_success": True,
            "optimizer_status": 0,
            "optimizer_message": "CONVERGENCE",
            "objective_value": 1.0,
            "gradient_inf_norm": 0.01,
            "finite_fit_evidence": True,
            "converged": True,
        },
        "terms": {"side_logit": 0.1, "team_logit": {"a": 0.2}},
    }
    bound = _baseline_source_binding(
        "hierarchical_bt",
        baseline,
        source,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        strict_cutoff="2026-01-02T00:00:00+00:00",
        expected_implementation_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_series_source="conservative_series_superset",
        expected_series_authoritative=False,
    )
    assert bound["status"] == "available"
    assert bound["series_identity"]["authoritative"] is False
    mismatched = _baseline_source_binding(
        "hierarchical_bt",
        {**baseline, "series_identity": {"source_types": ["grid"], "authoritative": True}},
        source,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        strict_cutoff="2026-01-02T00:00:00+00:00",
        expected_implementation_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_series_source="conservative_series_superset",
        expected_series_authoritative=False,
    )
    assert mismatched["status"] == "blocked"
    assert "hierarchical_bt_series_source_mismatch" in mismatched["blockers"]
    assert "hierarchical_bt_series_authority_mismatch" in mismatched["blockers"]
    forged = _baseline_source_binding(
        "hierarchical_bt",
        {**baseline, "implementation_sha256": "c" * 64, "config_sha256": "d" * 64},
        source,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        strict_cutoff="2026-01-02T00:00:00+00:00",
        expected_implementation_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_series_source="conservative_series_superset",
        expected_series_authoritative=False,
    )
    assert "hierarchical_bt_implementation_hash_mismatch" in forged["blockers"]
    assert "hierarchical_bt_config_hash_mismatch" in forged["blockers"]
    failed_fit = _baseline_source_binding(
        "hierarchical_bt",
        {
            **baseline,
            "fit": {
                **baseline["fit"],
                "converged": False,
                "optimizer_status": 2,
                "objective_value": float("nan"),
            },
        },
        source,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        strict_cutoff="2026-01-02T00:00:00+00:00",
        expected_implementation_sha256="a" * 64,
        expected_config_sha256="b" * 64,
        expected_series_source="conservative_series_superset",
        expected_series_authoritative=False,
    )
    assert "hierarchical_bt_optimizer_status_invalid" in failed_fit["blockers"]
    assert "hierarchical_bt_objective_value_nonfinite" in failed_fit["blockers"]


def test_sequential_binding_rejects_forged_implementation_and_config_hashes() -> None:
    source = _source_receipt(["g1", "g2", "g3", "g4"])
    train_ids = ["g1", "g2"]
    validation_ids = ["g3", "g4"]
    config = {"scale": 1.0}
    config_hash = hashlib.sha256(
        json.dumps(config, separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()
    baseline = {
        "source_receipt_sha256": source["receipt_sha256"],
        "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
        "train_game_identity_sha256": identity_sha256(train_ids),
        "validation_game_identity_sha256": identity_sha256(validation_ids),
        "strict_cutoff": "2026-01-02T00:00:00Z",
        "implementation_digest": "a" * 64,
        "rating_config": config,
    }
    valid = _baseline_source_binding(
        "sequential_player_elo",
        baseline,
        source,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        strict_cutoff="2026-01-02T00:00:00Z",
        expected_implementation_sha256="a" * 64,
        expected_config_sha256=config_hash,
    )
    assert valid["status"] == "available"
    forged = _baseline_source_binding(
        "sequential_player_elo",
        {
            **baseline,
            "implementation_digest": "c" * 64,
            "rating_config": {"scale": 2.0},
        },
        source,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        strict_cutoff="2026-01-02T00:00:00Z",
        expected_implementation_sha256="a" * 64,
        expected_config_sha256=config_hash,
    )
    assert "sequential_player_elo_implementation_hash_mismatch" in forged[
        "blockers"
    ]
    assert "sequential_player_elo_config_hash_mismatch" in forged["blockers"]


def test_chronological_folds_keep_series_whole_and_dates_strict() -> None:
    maps, _form = _manual_form(24)
    maps["blue_teamid"] = [f"oe:team:a-{index // 2}" for index in range(len(maps))]
    maps["red_teamid"] = [f"oe:team:b-{index // 2}" for index in range(len(maps))]
    folds = chronological_whole_series_folds(maps, n_folds=2)
    assert len(folds) == 2
    prior_validation_ids: set[str] = set()
    previous_end = None
    for fold in folds:
        assert set(fold["train_series_ids"]).isdisjoint(fold["validation_series_ids"])
        assert fold["train_end"] < fold["validation_start"]
        assert fold["validation_start"] <= fold["validation_end"]
        assert set(fold["train_game_ids"]).isdisjoint(fold["validation_game_ids"])
        assert prior_validation_ids.isdisjoint(fold["validation_game_ids"])
        assert fold["overlap_audit"]["prior_validation_game_overlap_count"] == 0
        if previous_end is not None:
            assert previous_end < fold["validation_start"]
        previous_end = fold["validation_end"]
        prior_validation_ids.update(fold["validation_game_ids"])


def test_roster_continuity_unavailable_blocks_the_slice() -> None:
    frame = pd.DataFrame(
        {
            "blue_roster_continuity": [np.nan] * 20,
            "red_roster_continuity": [np.nan] * 20,
        }
    )
    labels = _roster_change_labels(frame)
    assert labels is None
    report = _group_slice_metrics(
        pd.Series([0, 1] * 10, dtype=float),
        pd.Series([0.4, 0.6] * 10, dtype=float),
        labels,
        labels,
        slice_name="roster_change",
    )
    assert "roster_change_field_missing" in report["blockers"]


def test_chronological_folds_exclude_clusters_that_cross_intervals() -> None:
    rows = []
    for index in range(1, 41):
        bridge = index in {12, 29}
        rows.append(
            {
                "game_uid": f"g{index:02d}",
                "date": pd.Timestamp("2026-01-01T00:00:00Z")
                + pd.Timedelta(days=index - 1),
                "y_blue_win": index % 2,
                "league": "LEC",
                "blue_teamid": "oe:team:bridge-a" if bridge else f"oe:team:a-{index}",
                "red_teamid": "oe:team:bridge-b" if bridge else f"oe:team:b-{index}",
            }
        )
    folds = chronological_whole_series_folds(pd.DataFrame(rows), n_folds=3)
    assert len(folds) == 3
    validation_ids = {
        game_id for fold in folds for game_id in fold["validation_game_ids"]
    }
    assert {"g12", "g29"}.isdisjoint(validation_ids)
    assert {"g12", "g29"}.issubset(set(folds[2]["train_game_ids"]))
    assert any(
        fold["overlap_audit"]["excluded_boundary_cluster_count"] > 0
        for fold in folds[:2]
    )


def test_bridge_receipt_binds_bytes_and_rejects_mutation(tmp_path) -> None:
    payload = b"bridge"
    path = tmp_path / "oe_api_meta.json"
    path.write_bytes(payload)
    freeze = {
        "oe_bridge_sources": [
            {
                "name": path.name,
                "bytes": len(payload),
                "raw_sha256": hashlib.sha256(payload).hexdigest(),
            }
        ]
    }
    assert verify_bridge_sources(tmp_path, freeze)[path.name]["bytes"] == len(payload)
    path.write_bytes(payload + b"changed")
    with pytest.raises(FutureValueTrainingError, match="bridge source changed"):
        verify_bridge_sources(tmp_path, freeze)


def test_model_runtime_receipt_binds_code_source_environment_and_output(
    tmp_path, monkeypatch
) -> None:
    game_ids = ["g1", "g2"]
    source = _source_receipt(game_ids, source_as_of="2026-01-02T00:00:00Z")
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    source_file_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    oe_root = tmp_path / "oe"
    oe_root.mkdir()
    pd.DataFrame({"game_uid": game_ids}).to_parquet(oe_root / "maps.parquet")
    pd.DataFrame({"game_uid": ["g1"] * 10 + ["g2"] * 10}).to_parquet(
        oe_root / "oe_player_games.parquet"
    )
    pd.DataFrame({"game_uid": ["g1", "g1", "g2", "g2"]}).to_parquet(
        oe_root / "oe_team_games.parquet"
    )
    normalized = {}
    for label, name in (
        ("maps", "maps.parquet"),
        ("players", "oe_player_games.parquet"),
        ("teams", "oe_team_games.parquet"),
    ):
        path = oe_root / name
        normalized[label] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "locator": f"warehouse/parquet/oe_live/{name}",
        }
    freeze = {
        "reference_source_receipt_sha256": source["receipt_sha256"],
        "source_receipt_file_sha256": source_file_hash,
        "source_receipt_path": str(source_path),
        "normalized_source_files": normalized,
    }
    monkeypatch.setattr(training_module, "_load_freeze", lambda _path: freeze)
    monkeypatch.setattr(
        training_module,
        "_git_output",
        lambda _root, *args: "" if args[0] == "status" else "f" * 40,
    )

    def fake_evaluation(*_args, **kwargs):
        assert kwargs["source_receipt_path"] == str(source_path)
        assert kwargs["source_receipt_file_sha256"] == source_file_hash
        return {
            "source": {},
            "prediction_ledger": {"sha256": "a" * 64, "row_count": 2},
            "authority": {"research_only": True, "deployment": False},
        }

    monkeypatch.setattr(training_module, "evaluate_future_value", fake_evaluation)
    output_path = tmp_path / "model.json"
    runtime_path = tmp_path / "runtime.json"
    receipt = run_model_evaluation(
        oe_root=oe_root,
        freeze_path=tmp_path / "freeze.json",
        source_receipt_path=source_path,
        model_output_path=output_path,
        runtime_receipt_path=runtime_path,
        command=["python3", "-m", "future_value_training", "--fit-model"],
    )
    assert receipt["code_commit"] == "f" * 40
    assert receipt["output"]["sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert receipt["source"]["source_receipt_file_sha256"] == source_file_hash
    assert receipt["environment"]["logical_cpu_count"]
    assert receipt["authority"]["deployment"] is False
    for field in ("odds", "expected_value", "recommendation", "betting"):
        assert receipt["authority"][field] is False
    assert json.loads(runtime_path.read_text())["receipt_sha256"] == receipt[
        "receipt_sha256"
    ]
    with pytest.raises(FutureValueTrainingError, match="crosswalk inputs"):
        run_model_evaluation(
            oe_root=oe_root,
            freeze_path=tmp_path / "freeze.json",
            source_receipt_path=source_path,
            model_output_path=tmp_path / "unused-model.json",
            runtime_receipt_path=tmp_path / "unused-runtime.json",
            crosswalk_path=tmp_path / "crosswalk.json",
        )


def _phase_training_fixture(tmp_path, source: dict[str, object]):
    game_ids = [str(value) for value in source["model_eligible_game_ids"]]
    assignment = hashlib.sha256(
        json.dumps(
            [
                {"game_id": game_id, "series_id": f"series-{game_id}"}
                for game_id in sorted(game_ids)
            ],
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    partition = {
        "mapping_sha256": "a" * 64,
        "crosswalk_sha256": "b" * 64,
        "artifact_sha256": "c" * 64,
        "receipt_sha256": "d" * 64,
        "receipt_file_sha256": "e" * 64,
        "source_receipt_sha256": source["receipt_sha256"],
        "status": "comparable",
        "eligible_game_count": len(game_ids),
        "eligible_identity_sha256": identity_sha256(game_ids),
        "eligible_assignment_sha256": assignment,
        "reference_game_count": len(game_ids),
        "reference_identity_sha256": identity_sha256(game_ids),
        "reference_assignment_sha256": assignment,
        "reference_assignment_match": True,
        "proxy_authority_blocker": True,
    }
    artifact_path = tmp_path / "phase-candidate.json"
    artifact_path.write_text(
        json.dumps(
            {
                "source_receipt_sha256": source["receipt_sha256"],
                "cross_model_series_partition": partition,
                "series_partition_reference_game_count": len(game_ids),
                "series_partition_reference_identity_sha256": identity_sha256(game_ids),
                "series_partition_reference_assignment_sha256": assignment,
                "series_partition_proxy_authority_blocker": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    artifact_record = {
        "locator": str(artifact_path),
        "bytes": artifact_path.stat().st_size,
        "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
    }
    receipt_path = tmp_path / "phase-run-receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "source": {"source_receipt_sha256": source["receipt_sha256"]},
                "partition": partition,
                "outputs": {"candidate": artifact_record},
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    source_path = tmp_path / "source-receipt.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    return artifact_path, receipt_path


def test_training_phase_binding_verifies_files_and_receipt_output(tmp_path) -> None:
    source = _source_receipt(["g1", "g2"])
    artifact_path, receipt_path = _phase_training_fixture(tmp_path, source)
    binding, runtime = training_module._build_phase_partition_binding(
        artifact_path,
        receipt_path,
        artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        receipt_file_sha256=hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        source_receipt=source,
        source_receipt_path=tmp_path / "source-receipt.json",
        source_receipt_file_sha256=hashlib.sha256(
            (tmp_path / "source-receipt.json").read_bytes()
        ).hexdigest(),
        artifact_kind="candidate",
    )
    assert binding["eligible_game_count"] == 2
    assert binding["phase_artifact_kind"] == "candidate"
    assert runtime["expected_artifact_sha256"] == binding["phase_artifact_sha256"]


def test_training_phase_file_guard_rejects_symlinked_parent(tmp_path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    path = real_root / "phase.json"
    path.write_text("{}", encoding="utf-8")
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    linked_path = linked_root / "phase.json"
    with pytest.raises(FutureValueTrainingError, match="symlink"):
        training_module._phase_file_record(
            linked_path,
            expected_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            label="phase artifact",
        )


def test_training_phase_inputs_fail_closed_when_partial(tmp_path, monkeypatch) -> None:
    source = _source_receipt(["g1", "g2"], source_as_of="2026-01-02T00:00:00Z")
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source, sort_keys=True), encoding="utf-8")
    monkeypatch.setattr(
        training_module,
        "_load_freeze",
        lambda _path: {
            "reference_source_receipt_sha256": source["receipt_sha256"],
            "source_receipt_file_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "source_receipt_path": str(source_path),
        },
    )
    with pytest.raises(FutureValueTrainingError, match="phase partition inputs"):
        run_model_evaluation(
            oe_root=tmp_path / "oe",
            freeze_path=tmp_path / "freeze.json",
            source_receipt_path=source_path,
            model_output_path=tmp_path / "model.json",
            runtime_receipt_path=tmp_path / "runtime.json",
            phase_artifact_path=tmp_path / "phase.json",
        )
