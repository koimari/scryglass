from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lol_kills.research.public_draft_score_promotion import (
    CATEGORICAL_CONTEXT_COLUMNS,
    STRENGTH_COLUMNS,
    _mirror_categorical_frame,
)
from lol_kills.research.selective_draft_constituents import (
    ALL_ATOM_IDENTITY_SCHEMA,
    FROZEN_SELECTIVE_VOTERS_SCHEMA,
    QUANTUM_VOTER_SCHEMA,
    ROSTER_FOREST_CONFIG,
    ROSTER_FOREST_SCHEMA,
    ROSTER_GROUPS,
    STRENGTH_IDENTITY_SCHEMA,
    V24_QUANTUM_FEATURE_GROUPS_SHA256,
    V24_QUANTUM_PROTOCOL_RESOLVED_SHA256,
    DRAFT_INFERENCE_COLUMNS,
    _normalize_player_game_uids,
    _outcome_blind_draft_source,
    build_draft_games,
    SelectiveDraftConstituentError,
    fit_all_atom_identity_predictions,
    fit_frozen_selective_voters,
    fit_roster_random_forest_predictions,
    fit_quantum_masked_forest_predictions,
    fit_strength_identity_predictions,
    load_v24_quantum_contract,
)
from lol_kills.research.selective_draft_probability import canonical_sha256
from lol_kills.research.atomized_rf_composite import GROUP_COLUMNS


def _constituents_module() -> object:
    return sys.modules[fit_frozen_selective_voters.__module__]


def _training_frame(rows: int = 80) -> pd.DataFrame:
    values = np.linspace(-2.0, 2.0, rows)
    frame = pd.DataFrame(
        {
            "game_uid": [f"train-{index}" for index in range(rows)],
            "y": (values > 0).astype(int),
            "signal": values,
        }
    )
    for column in CATEGORICAL_CONTEXT_COLUMNS:
        if column.startswith("category_blue_"):
            frame[column] = "blue-value"
        elif column.startswith("category_red_"):
            frame[column] = "red-value"
        elif column == "category_first_pick_side":
            frame[column] = "blue"
        else:
            frame[column] = "shared"
    return frame


def test_all_atom_identity_voter_is_side_symmetric_and_outcome_free() -> None:
    training = _training_frame()
    evaluation = training.iloc[[10]].drop(columns="y").copy()
    evaluation["game_uid"] = "evaluation-blue"
    mirrored = _mirror_categorical_frame(
        evaluation, numeric_columns=("signal",)
    )
    mirrored["game_uid"] = "evaluation-red"
    evaluation = pd.concat([evaluation, mirrored], ignore_index=True)

    predictions, receipt = fit_all_atom_identity_predictions(
        training,
        evaluation,
        numeric_columns=("signal",),
    )

    assert predictions.columns.tolist() == ["game_uid", "p"]
    assert predictions["p"].sum() == pytest.approx(1.0, abs=1e-10)
    assert receipt["schema_version"] == ALL_ATOM_IDENTITY_SCHEMA
    assert len(receipt["receipt_sha256"]) == 64


def test_all_atom_identity_voter_rejects_outcome_or_feature_failures() -> None:
    training = _training_frame()
    evaluation = training.iloc[:2].drop(columns=["y", "signal"])
    with pytest.raises(SelectiveDraftConstituentError, match="evaluation columns"):
        fit_all_atom_identity_predictions(
            training,
            evaluation,
            numeric_columns=("signal",),
        )


def test_strength_identity_voter_uses_frozen_public_feature_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _training_frame()
    evaluation = training.iloc[:2].drop(columns="y")
    captured: dict[str, object] = {}

    def fake_fit(
        training_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        *,
        regularization_c: float,
        numeric_columns: tuple[str, ...],
    ) -> tuple[pd.DataFrame, dict[str, object]]:
        captured.update(
            training=training_frame,
            evaluation=evaluation_frame,
            regularization_c=regularization_c,
            numeric_columns=numeric_columns,
        )
        output = pd.DataFrame(
            {"game_uid": evaluation_frame["game_uid"], "p": [0.4, 0.6]}
        )
        return output, {
            "schema_version": ALL_ATOM_IDENTITY_SCHEMA,
            "receipt_sha256": "0" * 64,
        }

    monkeypatch.setattr(_constituents_module(), "fit_all_atom_identity_predictions", fake_fit)
    output, receipt = fit_strength_identity_predictions(training, evaluation)

    expected_numeric = tuple(
        dict.fromkeys(
            (
                *STRENGTH_COLUMNS,
                *GROUP_COLUMNS["match_context"],
                *GROUP_COLUMNS["competition_context"],
            )
        )
    )
    assert captured["regularization_c"] == 0.001
    assert captured["numeric_columns"] == expected_numeric
    assert output["p"].tolist() == [0.4, 0.6]
    assert receipt["schema_version"] == STRENGTH_IDENTITY_SCHEMA
    assert len(receipt["receipt_sha256"]) == 64


def test_roster_forest_voter_is_bound_and_does_not_use_evaluation_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = pd.DataFrame(
        {"game_uid": ["train-a", "train-b"], "y": [0, 1]}
    )
    evaluation = pd.DataFrame(
        {"game_uid": ["eval-a", "eval-b"], "y": [0, 1]}
    )
    captured: dict[str, object] = {}

    def fake_worlds(
        training_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        protocol: dict[str, object],
        *,
        cache_dir: object,
        matrix_sha256: str,
    ) -> tuple[np.ndarray, list[dict[str, str]]]:
        captured.update(
            training=training_frame,
            evaluation=evaluation_frame,
            protocol=protocol,
            cache_dir=cache_dir,
            matrix_sha256=matrix_sha256,
        )
        return np.asarray([[-1.0], [1.0]]), [{"world_id": "roster"}]

    monkeypatch.setattr(_constituents_module(), "_quantum_world_predictions", fake_worlds)
    output, receipt = fit_roster_random_forest_predictions(
        training,
        evaluation,
        matrix_sha256="a" * 64,
    )

    assert output.columns.tolist() == ["game_uid", "p"]
    assert output["p"].sum() == pytest.approx(1.0)
    assert captured["evaluation"] is evaluation
    assert captured["protocol"] == {
        "selection": {
            "quantum_worlds": [
                {"id": "composite-roster-world", "groups": list(ROSTER_GROUPS)}
            ],
            "quantum_forest_config": ROSTER_FOREST_CONFIG,
        }
    }
    assert receipt["schema_version"] == ROSTER_FOREST_SCHEMA
    assert receipt["matrix_sha256"] == "a" * 64
    assert len(receipt["receipt_sha256"]) == 64


@pytest.mark.parametrize("matrix_sha256", ["", "z" * 64, "a" * 63])
def test_roster_forest_rejects_invalid_matrix_identity(matrix_sha256: str) -> None:
    training = pd.DataFrame(
        {"game_uid": ["train-a", "train-b"], "y": [0, 1]}
    )
    evaluation = pd.DataFrame({"game_uid": ["eval-a"]})
    with pytest.raises(SelectiveDraftConstituentError, match="matrix SHA-256"):
        fit_roster_random_forest_predictions(
            training,
            evaluation,
            matrix_sha256=matrix_sha256,
        )


def _quantum_training_frame() -> pd.DataFrame:
    rows = 120
    return pd.DataFrame(
        {
            "game_uid": [f"train-{index}" for index in range(rows)],
            "series_id": [f"series-{index}" for index in range(rows)],
            "date": pd.date_range("2026-01-01", periods=rows, tz="UTC"),
            "y": [index % 2 for index in range(rows)],
            "signal": np.linspace(-2.0, 2.0, rows),
        }
    )


def test_v24_quantum_contract_is_repository_reproducible() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol, feature_groups = load_v24_quantum_contract(
        root
        / "data/lol/v2/evaluation/public-draft-score-promotion-protocol-v24.json"
    )

    assert canonical_sha256(protocol) == V24_QUANTUM_PROTOCOL_RESOLVED_SHA256
    assert canonical_sha256(feature_groups) == V24_QUANTUM_FEATURE_GROUPS_SHA256
    assert "rating_uncertainty" not in feature_groups
    assert sum(len(columns) for columns in feature_groups.values()) == 3326


def test_player_game_identity_uses_gameid_for_appended_rows() -> None:
    players = pd.DataFrame(
        {
            "game_uid": ["frozen-game", None],
            "gameid": ["frozen-game", "appended-game"],
        }
    )

    normalized = _normalize_player_game_uids(players)

    assert normalized["game_uid"].tolist() == [
        "frozen-game",
        "appended-game",
    ]


def test_pre_match_draft_source_excludes_every_result_field() -> None:
    roles = ["top", "jng", "mid", "bot", "sup"]
    players = pd.DataFrame(
        [
            {
                "game_uid": "future-game",
                "gameid": "future-game",
                "date": "2026-08-17T12:00:00Z",
                "side": side,
                "position": role,
                "champion": f"Champion-{side}-{role}",
                "playername": f"Player-{side}-{role}",
                "teamname": f"Team-{side}",
                "league": "LEC",
                "result": 1 if side == "Blue" else 0,
                "kills": 99,
                "goldat15": 99999,
            }
            for side in ("Blue", "Red")
            for role in roles
        ]
    )

    source = _outcome_blind_draft_source(players)
    games = build_draft_games(source, require_result=False)

    assert source.columns.tolist() == list(DRAFT_INFERENCE_COLUMNS)
    assert len(games) == 1
    assert "y" not in games[0]
    assert games[0]["blue"]["top"]["champion"] == "Champion-Blue-top"


def test_quantum_voter_uses_only_training_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _quantum_training_frame()
    evaluation = pd.DataFrame(
        {
            "game_uid": ["future-a", "future-b"],
            "series_id": ["future-series-a", "future-series-b"],
            "date": pd.to_datetime(["2026-05-02", "2026-05-03"], utc=True),
            "signal": [-1.0, 1.0],
            "y": [0, 1],
        }
    )
    protocol = {
        "selection": {
            "architecture": "quantum_masked_forest_v1",
            "phase_curve_config": None,
            "regional_worlds": [],
            "draft_expert_configs": [],
            "categorical_world_config": None,
        }
    }
    target_receipts: list[list[int]] = []

    def fake_stack(
        training_frame: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        **_: object,
    ) -> tuple[np.ndarray, dict[str, object]]:
        target_receipts.append(training_frame["y"].astype(int).tolist())
        return evaluation_frame[["signal"]].to_numpy(), {"rows": len(training_frame)}

    class FakeMeta:
        coef_ = np.asarray([[0.25, 0.75]])
        intercept_ = np.asarray([0.0])

    monkeypatch.setattr(_constituents_module(), "_quantum_stack", fake_stack)
    monkeypatch.setattr(
        _constituents_module(),
        "_select_anchor",
        lambda *_: {
            "team_weight": 0.5,
            "momentum_weight": 0.0,
            "source": "raw_rating_difference",
        },
    )
    monkeypatch.setattr(
        _constituents_module(),
        "_anchor_probability",
        lambda frame, **_: 1.0 / (1.0 + np.exp(-frame["signal"].to_numpy())),
    )
    monkeypatch.setattr(_constituents_module(), "_fit_quantum_meta", lambda *_: FakeMeta())
    monkeypatch.setattr(
        _constituents_module(),
        "_quantum_meta_probability",
        lambda _model, worlds, _anchor: 1.0 / (1.0 + np.exp(-worlds[:, 0])),
    )

    first, receipt = fit_quantum_masked_forest_predictions(
        training,
        evaluation,
        protocol=protocol,
        game_by_id={},
        inner_start="2026-03-01T00:00:00Z",
        evaluation_start="2026-05-01T00:00:00Z",
        matrix_sha256="b" * 64,
        feature_groups={"team_rating": ["signal"]},
    )
    flipped = evaluation.assign(y=1 - evaluation["y"])
    second, _ = fit_quantum_masked_forest_predictions(
        training,
        flipped,
        protocol=protocol,
        game_by_id={},
        inner_start="2026-03-01T00:00:00Z",
        evaluation_start="2026-05-01T00:00:00Z",
        matrix_sha256="b" * 64,
        feature_groups={"team_rating": ["signal"]},
    )

    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == ["game_uid", "p"]
    assert receipt["schema_version"] == QUANTUM_VOTER_SCHEMA
    assert len(receipt["receipt_sha256"]) == 64
    assert all(target_receipts)
    assert len(target_receipts) == 4


def test_quantum_voter_rejects_a_holdout_boundary_violation() -> None:
    training = _quantum_training_frame()
    evaluation = pd.DataFrame(
        {
            "game_uid": ["future-a"],
            "series_id": ["future-series-a"],
            "date": pd.to_datetime(["2026-04-30"], utc=True),
        }
    )
    protocol = {"selection": {"architecture": "quantum_masked_forest_v1"}}
    with pytest.raises(SelectiveDraftConstituentError, match="holdout boundary"):
        fit_quantum_masked_forest_predictions(
            training,
            evaluation,
            protocol=protocol,
            game_by_id={},
            inner_start="2026-03-01T00:00:00Z",
            evaluation_start="2026-05-01T00:00:00Z",
            matrix_sha256="b" * 64,
            feature_groups={"team_rating": ["signal"]},
        )


def test_frozen_voter_builder_removes_future_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = pd.DataFrame({"game_uid": ["train"], "y": [1]})
    evaluation = pd.DataFrame({"game_uid": ["future-a", "future-b"]})
    future_games = {
        "future-a": {"game_uid": "future-a", "y": 1, "winner": "blue"},
        "future-b": {"game_uid": "future-b", "y": 0, "winner": "red"},
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        _constituents_module(),
        "load_v24_quantum_contract",
        lambda _path: ({"selection": {}}, {"team_rating": ["signal"]}),
    )

    def fake_quantum(
        _training: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        **kwargs: object,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        captured["games"] = kwargs["game_by_id"]
        return (
            pd.DataFrame(
                {"game_uid": evaluation_frame["game_uid"], "p": [0.41, 0.59]}
            ),
            {"receipt_sha256": "1" * 64},
        )

    def fake_voter(
        _training: pd.DataFrame,
        evaluation_frame: pd.DataFrame,
        **_kwargs: object,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
        return (
            pd.DataFrame(
                {"game_uid": evaluation_frame["game_uid"], "p": [0.45, 0.55]}
            ),
            {"receipt_sha256": "2" * 64},
        )

    monkeypatch.setattr(
        _constituents_module(), "fit_quantum_masked_forest_predictions", fake_quantum
    )
    monkeypatch.setattr(
        _constituents_module(), "fit_roster_random_forest_predictions", fake_voter
    )
    monkeypatch.setattr(_constituents_module(), "fit_strength_identity_predictions", fake_voter)
    monkeypatch.setattr(_constituents_module(), "fit_all_atom_identity_predictions", fake_voter)

    predictions, receipt = fit_frozen_selective_voters(
        training,
        training,
        evaluation,
        v24_protocol_path=Path("unused.json"),
        game_by_id=future_games,
        inner_start="2026-06-01T00:00:00Z",
        evaluation_start="2026-08-16T00:00:00Z",
        cache_dir=None,
        source_matrix_sha256="a" * 64,
        training_matrix_sha256="b" * 64,
        quantum_training_matrix_sha256="d" * 64,
        evaluation_features_sha256="c" * 64,
    )

    assert predictions.columns.tolist() == [
        "game_uid",
        "quantum",
        "roster",
        "identity",
        "development_composite",
    ]
    assert receipt["schema_version"] == FROZEN_SELECTIVE_VOTERS_SCHEMA
    assert receipt["outcome_blind"] is True
    assert all("y" not in row for row in captured["games"].values())
    assert all("winner" not in row for row in captured["games"].values())


@pytest.mark.parametrize("field", ["y", "winner", "target_gold_diff_10"])
def test_frozen_voter_builder_rejects_revealed_fields(field: str) -> None:
    training = pd.DataFrame({"game_uid": ["train"], "y": [1]})
    evaluation = pd.DataFrame({"game_uid": ["future"], field: [1]})
    with pytest.raises(SelectiveDraftConstituentError, match="forbidden fields"):
        fit_frozen_selective_voters(
            training,
            training,
            evaluation,
            v24_protocol_path=Path("unused.json"),
            game_by_id={},
            inner_start="2026-06-01T00:00:00Z",
            evaluation_start="2026-08-16T00:00:00Z",
            cache_dir=None,
            source_matrix_sha256="a" * 64,
            training_matrix_sha256="b" * 64,
            quantum_training_matrix_sha256="d" * 64,
            evaluation_features_sha256="c" * 64,
        )


def test_frozen_voter_builder_rejects_an_empty_holdout() -> None:
    with pytest.raises(SelectiveDraftConstituentError, match="feature set is empty"):
        fit_frozen_selective_voters(
            pd.DataFrame({"game_uid": ["train"], "y": [1]}),
            pd.DataFrame({"game_uid": ["train"], "y": [1]}),
            pd.DataFrame({"game_uid": []}),
            v24_protocol_path=Path("unused.json"),
            game_by_id={},
            inner_start="2026-06-01T00:00:00Z",
            evaluation_start="2026-08-16T00:00:00Z",
            cache_dir=None,
            source_matrix_sha256="a" * 64,
            training_matrix_sha256="b" * 64,
            quantum_training_matrix_sha256="d" * 64,
            evaluation_features_sha256="c" * 64,
        )
