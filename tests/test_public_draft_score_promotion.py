from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from lol_kills.draft_recommendation import ROLES, build_games
from lol_kills.research.public_draft_score_promotion import (
    PublicDraftScorePromotionError,
    _categorical_world_predictions,
    _encoded_categorical_frame,
    _fit_bounded_draft_model,
    _fit_quantum_meta,
    _draft_expert_logits,
    _mirror_categorical_frame,
    _mirror_features,
    _phase_curve_predictions,
    _quantum_meta_probability,
    _quantum_world_predictions,
    _validate_protocol_matrix_binding,
    _validate_matrix_manifest,
)


def test_mirror_features_are_an_involution() -> None:
    frame = pd.DataFrame(
        {
            "blue_side": [1.0],
            "team_rating_diff_scaled": [0.4],
            "history_unique_player_maps_blue": [8.0],
            "history_unique_player_maps_red": [5.0],
            "history_unique_player_maps_min": [5.0],
            "history_player_champion_gold_diff_10": [120.0],
            "history_player_champion_gold_diff_10_support": [12.0],
            "history_player_champion_gold_diff_10_missing": [0.0],
            "context_league_LEC": [1.0],
            "team_momentum_count_difference": [2.0],
        }
    )
    columns = tuple(frame.columns)

    mirrored = _mirror_features(frame, columns)
    restored = _mirror_features(mirrored, columns)

    assert mirrored.loc[0, "blue_side"] == -1.0
    assert mirrored.loc[0, "team_rating_diff_scaled"] == -0.4
    assert mirrored.loc[0, "history_unique_player_maps_blue"] == 5.0
    assert mirrored.loc[0, "history_unique_player_maps_red"] == 8.0
    assert mirrored.loc[0, "history_unique_player_maps_min"] == 5.0
    assert mirrored.loc[0, "history_player_champion_gold_diff_10"] == -120.0
    assert mirrored.loc[0, "history_player_champion_gold_diff_10_support"] == 12.0
    assert mirrored.loc[0, "history_player_champion_gold_diff_10_missing"] == 0.0
    assert mirrored.loc[0, "context_league_LEC"] == 1.0
    assert mirrored.loc[0, "team_momentum_count_difference"] == -2.0
    pd.testing.assert_frame_equal(restored, frame)


def test_frozen_protocol_binds_the_matrix_hash() -> None:
    expected = "a" * 64

    _validate_protocol_matrix_binding(
        {"iteration": {"matrix_sha256": expected}}, expected
    )

    with pytest.raises(
        PublicDraftScorePromotionError,
        match="does not match the frozen protocol",
    ):
        _validate_protocol_matrix_binding(
            {"iteration": {"matrix_sha256": "b" * 64}}, expected
        )


def test_frozen_protocol_binds_the_matrix_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion.MODEL_COLUMNS",
        ("numeric",),
    )
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion.CATEGORICAL_CONTEXT_COLUMNS",
        ("category_team",),
    )
    matrix_sha256 = "a" * 64
    manifest_path = tmp_path / "matrix.manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "matrix:v1",
                "matrix_sha256": matrix_sha256,
                "rows": 10,
                "columns": ["numeric", "category_team"],
                "model_columns": ["numeric"],
                "categorical_columns": ["category_team"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    protocol = {
        "feature_contract": {"schema_version": "matrix:v1"},
        "iteration": {"matrix_manifest_sha256": manifest_sha256},
    }

    receipt = _validate_matrix_manifest(
        protocol=protocol,
        manifest_path=manifest_path,
        expected_manifest_sha256=manifest_sha256,
        expected_matrix_sha256=matrix_sha256,
    )

    assert receipt == {
        "path": str(manifest_path),
        "sha256": manifest_sha256,
        "schema_version": "matrix:v1",
    }
    with pytest.raises(
        PublicDraftScorePromotionError,
        match="matrix manifest does not match the frozen protocol",
    ):
        _validate_matrix_manifest(
            protocol={
                **protocol,
                "iteration": {"matrix_manifest_sha256": "b" * 64},
            },
            manifest_path=manifest_path,
            expected_manifest_sha256=manifest_sha256,
            expected_matrix_sha256=matrix_sha256,
        )


def test_phase_curve_uses_checkpoints_as_targets_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 600
    train = pd.DataFrame(
        {
            "game_uid": [f"train-{index}" for index in range(rows)],
            "x": np.linspace(-1.0, 1.0, rows),
            **{
                f"target_{metric}_diff_{checkpoint}": np.linspace(
                    -checkpoint, checkpoint, rows
                )
                for checkpoint in (10, 15, 20, 25)
                for metric in ("gold", "xp")
            },
        }
    )
    evaluation = pd.DataFrame(
        {"game_uid": ["eval-a", "eval-b"], "x": [-0.5, 0.5]}
    )
    protocol = {
        "selection": {
            "phase_curve_config": {
                "id": "phase-test",
                "n_estimators": 5,
                "max_depth": 3,
                "min_samples_leaf": 2,
                "max_features": 1.0,
                "max_samples": 1.0,
            }
        }
    }
    captured: dict[str, object] = {}

    class FakeRegressor:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit(self, features: pd.DataFrame, target: np.ndarray) -> None:
            captured["columns"] = list(features.columns)
            captured["target_shape"] = target.shape

        def predict(self, features: pd.DataFrame) -> np.ndarray:
            return np.zeros((len(features), 8), dtype=float)

    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion.MODEL_COLUMNS",
        ("x",),
    )
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion.RandomForestRegressor",
        FakeRegressor,
    )

    values, receipt = _phase_curve_predictions(train, evaluation, protocol)

    assert captured == {"columns": ["x"], "target_shape": (rows, 8)}
    assert values.shape == (2, 8)
    assert receipt["training_rows"] == rows


def test_categorical_codes_are_compact_and_keep_columns_discrete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion.CATEGORICAL_CONTEXT_COLUMNS",
        ("category_team", "category_champion"),
    )
    frame = pd.DataFrame(
        {
            "numeric": [0.2, -0.4],
            "category_team": ["oe:team:blue", "oe:team:red"],
            "category_champion": ["Ahri", "Orianna"],
        }
    )

    encoded, cardinality = _encoded_categorical_frame(
        frame, numeric_columns=("numeric",)
    )

    assert cardinality == {"category_team": 2, "category_champion": 2}
    assert encoded["category_team"].dtype == np.dtype("int32")
    assert encoded["category_team"].tolist() == [0, 1]
    assert encoded["category_champion"].tolist() == [0, 1]


def test_categorical_mirror_is_an_involution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion.CATEGORICAL_CONTEXT_COLUMNS",
        (
            "category_blue_team",
            "category_red_team",
            "category_first_pick_side",
            "category_patch",
        ),
    )
    frame = pd.DataFrame(
        {
            "signed_strength": [0.4],
            "category_blue_team": ["blue-id"],
            "category_red_team": ["red-id"],
            "category_first_pick_side": ["blue"],
            "category_patch": ["16.16"],
        }
    )

    mirrored = _mirror_categorical_frame(
        frame, numeric_columns=("signed_strength",)
    )
    restored = _mirror_categorical_frame(
        mirrored, numeric_columns=("signed_strength",)
    )

    assert mirrored.loc[0, "signed_strength"] == -0.4
    assert mirrored.loc[0, "category_blue_team"] == "red-id"
    assert mirrored.loc[0, "category_red_team"] == "blue-id"
    assert mirrored.loc[0, "category_first_pick_side"] == "red"
    assert mirrored.loc[0, "category_patch"] == "16.16"
    pd.testing.assert_frame_equal(restored, frame)


def test_categorical_world_reuses_hash_bound_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion.CATEGORICAL_CONTEXT_COLUMNS",
        ("category_team",),
    )
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion._candidate_columns",
        lambda _groups: ("x",),
    )
    train = pd.DataFrame(
        {
            "game_uid": [f"train-{index}" for index in range(40)],
            "y": [index % 2 for index in range(40)],
            "x": np.linspace(-1.0, 1.0, 40),
            "category_team": [f"team-{index % 4}" for index in range(40)],
        }
    )
    evaluation = pd.DataFrame(
        {
            "game_uid": ["eval-a", "eval-b"],
            "x": [-0.5, 0.5],
            "category_team": ["team-0", "team-new"],
        }
    )
    protocol = {
        "selection": {
            "categorical_world_config": {
                "id": "categorical-test",
                "groups": [],
                "n_estimators": 5,
                "learning_rate": 0.05,
                "num_leaves": 4,
                "max_depth": 3,
                "min_child_samples": 2,
                "subsample": 1.0,
                "colsample_bytree": 1.0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "cat_smooth": 5.0,
                "cat_l2": 1.0,
            }
        }
    }

    first, first_receipt = _categorical_world_predictions(
        train,
        evaluation,
        protocol,
        cache_dir=tmp_path,
        matrix_sha256="a" * 64,
    )
    second, second_receipt = _categorical_world_predictions(
        train,
        evaluation,
        protocol,
        cache_dir=tmp_path,
        matrix_sha256="a" * 64,
    )

    np.testing.assert_allclose(first, second)
    assert first_receipt == second_receipt
    assert first_receipt["train_cardinality"] == {"category_team": 4}
    assert first_receipt["evaluation_cardinality"] == {"category_team": 2}
    assert first_receipt["encoding_cardinality"] == {"category_team": 5}
    assert first_receipt["category_audit"] == {
        "category_team": {
            "train": 4,
            "evaluation": 2,
            "evaluation_unseen": 1,
        }
    }

    protocol["selection"]["categorical_world_config"] = {
        "id": "categorical-extra-trees-test",
        "learner": "extra_trees_onehot",
        "groups": [],
        "n_estimators": 8,
        "max_depth": 4,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "max_samples": 1.0,
        "class_weight": None,
    }
    forest_values, forest_receipt = _categorical_world_predictions(
        train,
        evaluation,
        protocol,
        cache_dir=tmp_path / "extra-trees",
        matrix_sha256="b" * 64,
    )

    assert np.isfinite(forest_values).all()
    assert forest_receipt["learner"] == "extra_trees_onehot"
    assert forest_receipt["train_cardinality"] == {"category_team": 4}
    assert forest_receipt["evaluation_cardinality"] == {"category_team": 2}
    assert forest_receipt["encoding_cardinality"] == {"category_team": 4}


def test_bounded_draft_model_produces_finite_scores() -> None:
    matrix = sparse.csr_matrix(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 0.5],
            [0.0, 0.0, -0.5],
        ]
    )
    model = _fit_bounded_draft_model(
        matrix,
        np.asarray([1, 0, 1, 0]),
        np.ones(4),
        0.001,
    )

    assert np.isfinite(model.coef_).all()
    assert np.isfinite(model.intercept_).all()
    assert np.isfinite(model.decision_function(matrix)).all()


@pytest.mark.parametrize(
    ("outcomes", "weights", "message"),
    [
        ([1, 1], [1.0, 1.0], "both outcomes"),
        ([0, 1], [1.0, float("nan")], "weights are invalid"),
        ([0, 1], [1.0, 0.0], "weights are invalid"),
    ],
)
def test_bounded_draft_model_rejects_invalid_fit_inputs(
    outcomes: list[int], weights: list[float], message: str
) -> None:
    matrix = sparse.csr_matrix([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(PublicDraftScorePromotionError, match=message):
        _fit_bounded_draft_model(
            matrix,
            np.asarray(outcomes),
            np.asarray(weights),
            0.001,
        )


def test_quantum_meta_rejects_non_finite_world_scores() -> None:
    with pytest.raises(PublicDraftScorePromotionError, match="inputs are not finite"):
        _fit_quantum_meta(
            np.asarray([0, 1]),
            np.asarray([[0.0, float("inf")], [0.0, 1.0]]),
            np.asarray([0.0, 0.0]),
        )


def test_quantum_meta_produces_finite_probabilities() -> None:
    target = np.asarray([0, 1, 0, 1])
    worlds = np.asarray(
        [[-0.5, -0.2], [0.3, 0.7], [-0.1, -0.4], [0.8, 0.4]]
    )
    anchor = np.asarray([-0.3, 0.2, -0.2, 0.5])
    model = _fit_quantum_meta(target, worlds, anchor)

    probability = _quantum_meta_probability(model, worlds, anchor)

    assert np.isfinite(probability).all()
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert model.intercept_.tolist() == [0.0]


def test_quantum_world_predictions_reuse_hash_bound_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = pd.DataFrame(
        {"game_uid": ["a", "b", "c", "d"], "y": [0, 1, 0, 1], "x": [0, 1, 2, 3]}
    )
    evaluation = pd.DataFrame(
        {"game_uid": ["e", "f"], "y": [0, 1], "x": [4, 5]}
    )
    protocol = {
        "selection": {
            "quantum_forest_config": {"n_estimators": 5},
            "quantum_worlds": [
                {"id": "first", "groups": ["first_group"]},
                {"id": "second", "groups": ["second_group"]},
            ],
        }
    }
    calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion._quantum_world_columns",
        lambda groups, **_kwargs: ("x",),
    )

    def fake_fit_probability(
        _train: pd.DataFrame,
        _evaluation: pd.DataFrame,
        columns: tuple[str, ...],
        _config: dict[str, object],
        *,
        shuffled: bool = False,
    ) -> np.ndarray:
        calls.append(columns)
        return np.asarray([0.4, 0.6])

    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion._fit_probability",
        fake_fit_probability,
    )
    matrix_sha = "a" * 64
    first, first_receipts = _quantum_world_predictions(
        train,
        evaluation,
        protocol,
        cache_dir=tmp_path,
        matrix_sha256=matrix_sha,
    )
    second, second_receipts = _quantum_world_predictions(
        train,
        evaluation,
        protocol,
        cache_dir=tmp_path,
        matrix_sha256=matrix_sha,
    )

    assert len(calls) == 2
    np.testing.assert_array_equal(first, second)
    assert first_receipts == second_receipts


def test_quantum_world_predictions_reject_tampered_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = pd.DataFrame(
        {"game_uid": ["a", "b"], "y": [0, 1], "x": [0, 1]}
    )
    evaluation = pd.DataFrame(
        {"game_uid": ["c", "d"], "y": [0, 1], "x": [2, 3]}
    )
    protocol = {
        "selection": {
            "quantum_forest_config": {"n_estimators": 5},
            "quantum_worlds": [{"id": "first", "groups": ["first_group"]}],
        }
    }
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion._quantum_world_columns",
        lambda groups, **_kwargs: ("x",),
    )
    monkeypatch.setattr(
        "lol_kills.research.public_draft_score_promotion._fit_probability",
        lambda *_args, **_kwargs: np.asarray([0.4, 0.6]),
    )
    matrix_sha = "b" * 64
    _quantum_world_predictions(
        train,
        evaluation,
        protocol,
        cache_dir=tmp_path,
        matrix_sha256=matrix_sha,
    )
    cache_path = next((tmp_path / "world-predictions-v2").glob("*.npz"))
    np.savez_compressed(
        cache_path,
        logits=np.asarray([[float("nan")]]),
        receipt=np.asarray([], dtype=np.uint8),
    )

    with pytest.raises(PublicDraftScorePromotionError, match="forest cache is invalid"):
        _quantum_world_predictions(
            train,
            evaluation,
            protocol,
            cache_dir=tmp_path,
            matrix_sha256=matrix_sha,
        )


def test_draft_loader_keeps_first_complete_role_rows_and_time_order() -> None:
    rows: list[dict[str, object]] = []
    for game_id, date in (("later", "2026-02-02"), ("earlier", "2026-01-01")):
        for side, result in (("Blue", 1), ("Red", 0)):
            for role in ROLES:
                rows.append(
                    {
                        "game_uid": game_id,
                        "date": date,
                        "side": side,
                        "position": role,
                        "champion": f"{side}-{role}",
                        "playername": f"first-{side}-{role}",
                        "teamname": f"{side}-team",
                        "league": "lec",
                        "result": result,
                    }
                )
        rows.append({**rows[-1], "playername": "duplicate-must-not-win"})

    games = build_games(pd.DataFrame(rows))

    assert [game["game_uid"] for game in games] == ["earlier", "later"]
    assert games[0]["blue_team"] == "Blue-team"
    assert games[0]["red_team"] == "Red-team"
    assert games[0]["red"]["sup"]["player"] == "first-Red-sup"


def test_draft_expert_reads_labels_from_training_matrix() -> None:
    rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    for index in range(20):
        game_id = f"game-{index}"
        training_rows.append({"game_uid": game_id, "y": index % 2})
        for side in ("Blue", "Red"):
            for role in ROLES:
                rows.append(
                    {
                        "game_uid": game_id,
                        "date": f"2026-01-{index + 1:02d}T12:00:00Z",
                        "side": side,
                        "position": role,
                        "champion": f"{side}-{role}-{index % 3}",
                        "playername": f"{side}-{role}-{index % 4}",
                        "teamname": f"{side}-team-{index % 5}",
                        "league": "LEC",
                    }
                )
    games = build_games(pd.DataFrame(rows), require_result=False)
    game_by_id = {game["game_uid"]: game for game in games}
    training = pd.DataFrame(training_rows[:18])
    evaluation = pd.DataFrame(training_rows[18:]).drop(columns="y")

    prediction, receipts = _draft_expert_logits(
        training,
        evaluation,
        game_by_id,
        [{"id": "test", "alpha": 0.01, "half_life_days": 365}],
    )

    assert prediction.shape == (2, 2)
    assert np.isfinite(prediction).all()
    assert receipts[0]["outputs"] == ["full_logit", "composition_only_logit"]
