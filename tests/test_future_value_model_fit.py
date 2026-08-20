from __future__ import annotations

import hashlib
import json
import numpy as np
import pandas as pd
import pytest

from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    _map_model_frame,
    _frame_game_ids,
    _baseline_output_alignment,
    _baseline_source_binding,
    build_future_value_design,
    build_time_decayed_prior_player_form,
    chronological_whole_series_folds,
    fit_future_value_model,
    fit_rank3_player_champion_role_atoms,
    evaluate_future_value,
)
from lol_kills.research.future_value_training import (
    FutureValueTrainingError,
    verify_bridge_sources,
)
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
    payload = {
        "source_as_of": source_as_of,
        "source_game_count": len(game_ids),
        "source_identity_sha256": identity_sha256(game_ids),
        "accepted_game_ids": game_ids,
        "model_eligible_game_count": len(game_ids),
        "model_eligible_identity_sha256": identity_sha256(game_ids),
        "model_eligible_game_ids": game_ids,
        "source_files": {"fixture": {"bytes": 1, "sha256": "0" * 64}},
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
    broken = form.drop(form.index[0]).copy()
    with pytest.raises(FutureValueSourceError, match="exact five-player"):
        build_future_value_design(maps, broken, atom)
    broken_role = form.copy()
    broken_role.loc[broken_role.index[0], "role"] = "mid"
    with pytest.raises(FutureValueSourceError, match="duplicate game-side-role"):
        build_future_value_design(maps, broken_role, atom)


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


def test_fit_requires_a_verified_source_receipt() -> None:
    maps, form = _manual_form(24)
    with pytest.raises(FutureValueSourceError, match="verified source receipt is required"):
        fit_future_value_model(
            maps,
            form,
            train_game_ids=[str(index) for index in range(1, 22)],
            fit_window_end="2026-01-22T00:00:00Z",
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
    assert fold["regional_transfer"]["status"] == "unavailable"
    assert fold["patch_transfer"]["status"] == "unavailable"
    assert fold["tournament_boundary"]["status"] == "unavailable"
    assert "regional_transfer_slice_missing" in result["blockers"]
    assert "patch_transfer_slice_missing" in result["blockers"]
    assert "tournament_boundary_slice_missing" in result["blockers"]
    assert result["evaluation"]["pooled_calibration"]["rows"] == fold["paired_rows"]


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
    }
    bound = _baseline_source_binding(
        "hierarchical_bt",
        baseline,
        source,
        train_game_ids=train_ids,
        validation_game_ids=validation_ids,
        strict_cutoff="2026-01-02T00:00:00+00:00",
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
        expected_series_source="conservative_series_superset",
        expected_series_authoritative=False,
    )
    assert mismatched["status"] == "blocked"
    assert "hierarchical_bt_series_source_mismatch" in mismatched["blockers"]
    assert "hierarchical_bt_series_authority_mismatch" in mismatched["blockers"]


def test_chronological_folds_keep_series_whole_and_dates_strict() -> None:
    maps, _form = _manual_form(24)
    maps["series_id"] = [f"series-{index // 2}" for index in range(len(maps))]
    folds = chronological_whole_series_folds(maps, n_folds=2)
    assert folds
    for fold in folds:
        assert set(fold["train_series_ids"]).isdisjoint(fold["validation_series_ids"])
        assert fold["train_end"] < fold["validation_start"]
        assert set(fold["train_game_ids"]).isdisjoint(fold["validation_game_ids"])


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
