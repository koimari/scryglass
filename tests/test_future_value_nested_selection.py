from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

import lol_kills.research.future_value_rating as rating
from benchmarks.build_future_value_final_fit import (
    FinalFitError,
    _canonical_sha,
    _verified_nested_selection,
)
from tests.test_future_value_model_fit import _manual_form
from tests.test_future_value_snapshots import _source_receipt


def _source_for(maps: pd.DataFrame, game_ids: list[str]) -> dict[str, object]:
    source = _source_receipt(game_ids)
    source["source_as_of"] = maps["date"].max().isoformat().replace("+00:00", "Z")
    source.pop("receipt_sha256")
    source["receipt_sha256"] = hashlib.sha256(
        json.dumps(
            source,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return source


def _ledger(
    maps: pd.DataFrame,
    source: dict[str, object],
    game_ids: list[str],
    train_ids: list[str],
) -> pd.DataFrame:
    features = rating.CURRENT_RATING_SIGNED_MAP_FEATURES
    dates = maps.set_index(maps["game_uid"].astype(str))["date"]
    frame = pd.DataFrame(
        {
            "game_id": game_ids,
            "date": [dates.loc[value] for value in game_ids],
            "series_id": [f"series-{value}" for value in game_ids],
            **{
                name: np.linspace(0.1, 1.0, len(game_ids)) + index
                for index, name in enumerate(features)
            },
        }
    )
    validation_ids = sorted(set(game_ids) - set(train_ids))
    frame.attrs = {
        "schema_version": rating.RATING_FEATURE_LEDGER_SCHEMA_VERSION,
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "fit_game_ids": list(train_ids),
        "fit_game_identity_sha256": rating.identity_sha256(train_ids),
        "game_identity_sha256": rating.identity_sha256(game_ids),
        "validation_game_ids": validation_ids,
        "validation_game_identity_sha256": rating.identity_sha256(validation_ids),
        "fit_window_end": "2026-01-12T00:00:00Z",
        "fit_date_min": str(frame.loc[frame["game_id"].isin(train_ids), "date"].min()),
        "fit_date_max": str(frame.loc[frame["game_id"].isin(train_ids), "date"].max()),
        "feature_names": list(features),
        "ledger_rows_sha256": "a" * 64,
        "feature_value_digest": "b" * 64,
        "producer_receipt_sha256": "c" * 64,
        "producer_receipt": {
            "producer_artifacts": {
                "artifact": {
                    "path": "/private/tmp/future-value-test-artifact",
                    "bytes": 1,
                    "sha256": "d" * 64,
                }
            }
        },
    }
    return frame


def test_variant_selector_requires_a_distinct_inner_ledger() -> None:
    maps, form = _manual_form(24)
    map_frame = rating._map_model_frame(maps)
    source = _source_for(maps, [str(value) for value in range(1, 25)])
    selection = rating._select_fold_regularization(
        map_frame,
        form,
        train_game_ids=tuple(str(value) for value in range(1, 22)),
        rank=rating.RANK_3,
        min_cell_support=1,
        variant=rating.RatingVariant.FUTURE_PLAYER_FORM,
        source_receipt=source,
        fit_window_end="2026-01-22T00:00:00Z",
    )
    assert selection["inner_ledger_status"] == "missing"
    assert selection["selected_c"] == rating.PREDECLARED_VARIANT_REGULARIZATION_C
    assert selection["blockers"] == ["nested_inner_feature_ledger_missing_fixed_c_used"]


def test_variant_selector_binds_inner_census_and_candidate_grid(monkeypatch) -> None:
    maps, form = _manual_form(24)
    map_frame = rating._map_model_frame(maps)
    all_ids = [str(value) for value in range(1, 25)]
    outer_train_ids = all_ids[:21]
    inner_train_ids = outer_train_ids[:11]
    source = _source_for(maps, all_ids)
    inner = _ledger(maps, source, outer_train_ids, inner_train_ids)

    monkeypatch.setattr(
        rating,
        "validate_rating_feature_ledger",
        lambda frame, **_kwargs: frame.copy(),
    )
    selection = rating._select_fold_regularization(
        map_frame,
        form,
        train_game_ids=tuple(outer_train_ids),
        rank=rating.RANK_3,
        min_cell_support=1,
        variant=rating.RatingVariant.FUTURE_PLAYER_FORM,
        inner_feature_ledger=inner,
        source_receipt=source,
        fit_window_end="2026-01-22T00:00:00Z",
    )
    assert selection["inner_ledger_status"] == "verified"
    assert selection["blockers"] == []
    assert selection["outer_ledger_reuse"] is False
    assert selection["inner_train_identity_sha256"] == rating.identity_sha256(
        inner_train_ids
    )
    assert selection["inner_validation_identity_sha256"] == rating.identity_sha256(
        outer_train_ids[11:]
    )
    assert selection["candidate_grid"] == list(rating.REGULARIZATION_GRID)
    assert len(selection["candidate_scores"]) == len(rating.REGULARIZATION_GRID)
    assert all(row["optimizer"]["success"] for row in selection["candidate_scores"])
    assert selection["selected_c"] in selection["candidate_grid"]


def test_variant_selector_binds_series_boundary_exclusions(monkeypatch) -> None:
    maps, form = _manual_form(24)
    map_frame = rating._map_model_frame(maps)
    all_ids = [str(value) for value in range(1, 25)]
    outer_train_ids = all_ids[:23]
    inner_train_ids = outer_train_ids[:11]
    excluded_id = outer_train_ids[11]
    inner_validation_ids = outer_train_ids[12:]
    source = _source_for(maps, all_ids)
    inner = _ledger(
        maps,
        source,
        [*inner_train_ids, *inner_validation_ids],
        inner_train_ids,
    )
    inner_fold = {
        "train_game_ids": inner_train_ids,
        "validation_game_ids": inner_validation_ids,
        "validation_start": "2026-01-12T00:00:00Z",
        "validation_end": "2026-01-23T00:00:00Z",
        "overlap_audit": {"excluded_boundary_map_count": 1},
    }
    monkeypatch.setattr(
        rating,
        "chronological_whole_series_folds",
        lambda *_args, **_kwargs: [inner_fold],
    )
    monkeypatch.setattr(
        rating,
        "validate_rating_feature_ledger",
        lambda frame, **_kwargs: frame.copy(),
    )

    selection = rating._select_fold_regularization(
        map_frame,
        form,
        train_game_ids=tuple(outer_train_ids),
        rank=rating.RANK_3,
        min_cell_support=1,
        variant=rating.RatingVariant.FUTURE_PLAYER_FORM,
        inner_feature_ledger=inner,
        source_receipt=source,
        fit_window_end="2026-01-24T00:00:00Z",
    )

    assert selection["inner_boundary_excluded_game_count"] == 1
    assert selection["inner_boundary_excluded_identity_sha256"] == (
        rating.identity_sha256((excluded_id,))
    )


def test_variant_selector_rejects_inner_ledger_with_outer_validation_rows(monkeypatch) -> None:
    maps, form = _manual_form(24)
    map_frame = rating._map_model_frame(maps)
    all_ids = [str(value) for value in range(1, 25)]
    outer_train_ids = all_ids[:21]
    source = _source_for(maps, all_ids)
    inner = _ledger(maps, source, all_ids, all_ids[:11])
    monkeypatch.setattr(
        rating,
        "validate_rating_feature_ledger",
        lambda frame, **_kwargs: frame.copy(),
    )
    with pytest.raises(rating.FutureValueSourceError, match="outer validation IDs"):
        rating._select_fold_regularization(
            map_frame,
            form,
            train_game_ids=tuple(outer_train_ids),
            rank=rating.RANK_3,
            min_cell_support=1,
            variant=rating.RatingVariant.FUTURE_PLAYER_FORM,
            inner_feature_ledger=inner,
            source_receipt=source,
            fit_window_end="2026-01-22T00:00:00Z",
        )


def test_final_fit_accepts_only_verified_nested_selection_evidence(tmp_path) -> None:
    maps, _form = _manual_form(24)
    all_ids = [str(value) for value in range(1, 25)]
    source = _source_for(maps, all_ids)
    artifact = tmp_path / "inner-ledger.parquet"
    artifact.write_bytes(b"bound nested artifact")
    artifact_record = {
        "path": str(artifact),
        "bytes": artifact.stat().st_size,
        "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
    }
    inner_train = all_ids[:11]
    inner_validation = all_ids[11:21]
    binding = {
        "schema_version": rating.RATING_FEATURE_LEDGER_SCHEMA_VERSION,
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "producer_receipt_sha256": "c" * 64,
        "ledger_rows_sha256": "a" * 64,
        "feature_value_digest": "b" * 64,
        "feature_names": list(rating.CURRENT_RATING_SIGNED_MAP_FEATURES),
        "game_identity_sha256": rating.identity_sha256(all_ids[:21]),
        "fit_game_identity_sha256": rating.identity_sha256(inner_train),
        "validation_game_identity_sha256": rating.identity_sha256(inner_validation),
        "fit_window_end": "2026-01-12T00:00:00Z",
        "fit_date_min": "2026-01-01T00:00:00Z",
        "fit_date_max": "2026-01-11T00:00:00Z",
        "validation_date_min": "2026-01-12T00:00:00Z",
        "validation_date_max": "2026-01-21T00:00:00Z",
        "fit_game_ids": inner_train,
        "validation_game_ids": inner_validation,
        "producer_artifacts": {"inner": artifact_record},
    }
    binding["binding_sha256"] = _canonical_sha(binding)
    scores = [
        {
            "c": float(value),
            "log_loss": float(index + 1),
            "optimizer": {"success": True, "finite_coefficients": True},
            "prediction_sha256": f"{index + 1:064x}",
        }
        for index, value in enumerate(rating.REGULARIZATION_GRID)
    ]
    selection = {
        "method": "nested_chronological_whole_series_log_loss",
        "candidate_grid": list(rating.REGULARIZATION_GRID),
        "candidate_scores": scores,
        "selected_c": float(rating.REGULARIZATION_GRID[0]),
        "inner_ledger_status": "verified",
        "blockers": [],
        "variant": rating.RatingVariant.FUTURE_PLAYER_FORM.value,
        "inner_feature_ledger_binding": binding,
        "inner_validation_start": "2026-01-12T00:00:00Z",
        "inner_validation_end": "2026-01-21T00:00:00Z",
    }
    payload = {
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "variants": {
            rating.RatingVariant.FUTURE_PLAYER_FORM.value: {
                "folds": [{"fold": 1, "regularization_selection": selection}]
            }
        },
    }
    path = tmp_path / "nested-selection.json"
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    result = _verified_nested_selection(
        path,
        source,
        expected_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert result["selected_c"] == rating.REGULARIZATION_GRID[0]
    assert result["folds"][0]["inner_train_identity_sha256"] == rating.identity_sha256(
        inner_train
    )

    modal_payload = json.loads(json.dumps(payload))
    variant_payload = modal_payload["variants"][
        rating.RatingVariant.FUTURE_PLAYER_FORM.value
    ]
    variant_payload["feature_names"] = list(
        rating.CURRENT_RATING_SIGNED_MAP_FEATURES
    )
    variant_payload["folds"] = []
    for fold, selected_c in enumerate((0.003, 0.001, 0.001), start=1):
        fold_selection = json.loads(json.dumps(selection))
        fold_selection["selected_c"] = selected_c
        variant_payload["folds"].append(
            {"fold": fold, "regularization_selection": fold_selection}
        )
    modal_path = tmp_path / "nested-selection-modal.json"
    modal_path.write_text(json.dumps(modal_payload, sort_keys=True), encoding="utf-8")
    modal = _verified_nested_selection(
        modal_path,
        source,
        expected_file_sha256=hashlib.sha256(modal_path.read_bytes()).hexdigest(),
    )
    assert modal["selected_c"] == 0.001
    assert modal["final_refit_selection_rule"] == (
        "outer_fold_mode_then_smaller_c_tie"
    )
    assert modal["outer_fold_selected_c_counts"] == {"0.001": 2, "0.003": 1}

    mutated = json.loads(path.read_text(encoding="utf-8"))
    mutated["variants"][rating.RatingVariant.FUTURE_PLAYER_FORM.value]["folds"][0][
        "regularization_selection"
    ]["selected_c"] = 1.0
    path.write_text(json.dumps(mutated, sort_keys=True), encoding="utf-8")
    with pytest.raises(FinalFitError, match="nested selection evidence file changed"):
        _verified_nested_selection(
            path,
            source,
            expected_file_sha256=hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )
