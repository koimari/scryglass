from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from lol_kills.v2.draft.interactions import oe_nuisance_baseline as baseline
from lol_kills.v2.draft.interactions.oe_target_evidence import (
    canonical_sha256,
)


def _rows(months: int = 5, clusters_per_month: int = 2) -> pd.DataFrame:
    rows = []
    game = 0
    champions = [f"riot:champion:{value}" for value in range(1, 13)]
    for month in range(1, months + 1):
        for cluster in range(clusters_per_month):
            cluster_id = f"cluster:{month}:{cluster}"
            for offset in range(30):
                game += 1
                row = {
                    "game_id": f"game:{game}",
                    "dependence_cluster_id": cluster_id,
                    "split": (
                        "train"
                        if month <= 2
                        else "development"
                        if month <= 4
                        else "validation"
                    ),
                    "oe_date_naive": f"2025-{month:02d}-{1 + offset % 20:02d}T12:00:00",
                    "canonical_league": ("LCK", "LPL")[game % 2],
                    "oe_patch_token": f"15.{month:02d}",
                    "y_blue_win": game % 2,
                }
                for role_index, role in enumerate(baseline.ROLE_ORDER):
                    row[f"blue_{role}_stable_champion_id"] = champions[
                        (game + role_index) % len(champions)
                    ]
                    row[f"red_{role}_stable_champion_id"] = champions[
                        (game + role_index + 3) % len(champions)
                    ]
                rows.append(row)
    return pd.DataFrame(rows).loc[:, baseline.INPUT_COLUMNS]


class _FakeLogisticRegression:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.classes_ = np.array([0, 1])
        self.n_iter_ = np.array([1])

    def fit(self, x: object, y: object) -> "_FakeLogisticRegression":
        width = x.shape[1]
        self.coef_ = np.zeros((1, width))
        self.intercept_ = np.zeros(1)
        if self.mode == "warning":
            warnings.warn("forced", baseline.ConvergenceWarning)
        elif self.mode == "iteration_limit":
            self.n_iter_ = np.array([baseline.MAXIMUM_ITERATIONS])
        elif self.mode == "iteration_shape":
            self.n_iter_ = np.array([[1]])
        elif self.mode == "nonfinite_coefficient":
            self.coef_[0, 0] = np.nan
        return self

    def decision_function(self, x: object) -> np.ndarray:
        values = np.zeros(x.shape[0])
        if self.mode == "nonfinite_decision":
            values[0] = np.inf
        return values

    def predict_proba(self, x: object) -> np.ndarray:
        values = np.tile([0.5, 0.5], (x.shape[0], 1))
        if self.mode == "boundary_probability":
            values[0] = [1.0, 0.0]
        return values


def test_cluster_atomic_rolling_predictions_use_only_strictly_earlier_clusters() -> None:
    rows = _rows()
    oof = baseline.cross_fit_rows(rows)
    assert not oof.empty
    assert oof["game_id"].is_unique
    assert (
        pd.to_datetime(oof["fit_maximum_date_naive"])
        < pd.to_datetime(oof["prediction_fold_month_naive"])
    ).all()
    for _, group in rows.groupby("dependence_cluster_id"):
        predicted = set(group["game_id"]) & set(oof["game_id"])
        assert not predicted or predicted == set(group["game_id"])


def test_future_and_same_fold_label_changes_cannot_change_earlier_predictions() -> None:
    rows = _rows()
    original = baseline.cross_fit_rows(rows)
    changed = rows.copy()
    changed.loc[changed["oe_date_naive"] >= "2025-04", "y_blue_win"] = (
        1 - changed.loc[changed["oe_date_naive"] >= "2025-04", "y_blue_win"]
    )
    replay = baseline.cross_fit_rows(changed)
    earlier = original[original["prediction_fold_month_naive"] < "2025-04"]
    comparison = replay[replay["game_id"].isin(earlier["game_id"])]
    assert earlier["game_id"].tolist() == comparison["game_id"].tolist()
    np.testing.assert_array_equal(
        earlier["p_blue_win_richer_candidate_oof"],
        comparison["p_blue_win_richer_candidate_oof"],
    )
    assert earlier[
        ["selected_regularization_C", "selected_nuisance_method"]
    ].equals(
        comparison[
            ["selected_regularization_C", "selected_nuisance_method"]
        ].reset_index(drop=True)
    )


def test_duration_team_player_and_timestamp_inputs_are_rejected() -> None:
    rows = _rows()
    for column in (
        "gamelength",
        "team_id",
        "player_name",
        "forecast_at",
        "derived_resolution_time_naive",
    ):
        changed = rows.assign(**{column: 1})
        with pytest.raises(
            baseline.OENuisanceBaselineError,
            match="forbidden duration/team/player input",
        ):
            baseline.cross_fit_rows(changed)


def test_shuffled_labels_change_diagnostics_but_not_feature_or_fold_contract() -> None:
    rows = _rows()
    original = baseline.cross_fit_rows(rows)
    shuffled = rows.copy()
    shuffled["y_blue_win"] = np.random.default_rng(7).permutation(
        shuffled["y_blue_win"].to_numpy()
    )
    changed = baseline.cross_fit_rows(shuffled)
    assert original[
        ["game_id", "dependence_cluster_id", "prediction_fold_month_naive"]
    ].equals(
        changed[
            ["game_id", "dependence_cluster_id", "prediction_fold_month_naive"]
        ]
    )
    assert not np.array_equal(
        original["p_blue_win_nuisance_oof"],
        changed["p_blue_win_nuisance_oof"],
    )


def test_split_is_validated_before_any_target_processing() -> None:
    split = json.loads(baseline.DEFAULT_SPLIT_PATH.read_text())
    changed = copy.deepcopy(split)
    changed["outcome_free"] = False
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    malformed_targets = _rows()
    malformed_targets["y_blue_win"] = "do-not-read"
    with pytest.raises(
        baseline.OENuisanceBaselineError, match="split population contract"
    ):
        baseline._validate_against_split(malformed_targets, changed)


def test_holdout_access_is_rejected() -> None:
    rows = _rows()
    rows.loc[0, "split"] = "final_temporal_holdout"
    with pytest.raises(
        baseline.OENuisanceBaselineError, match="sealed final temporal holdout"
    ):
        baseline.cross_fit_rows(rows)


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("warning", "ConvergenceWarning"),
        ("iteration_limit", "iteration limit"),
        ("iteration_shape", "iteration shape"),
        ("nonfinite_coefficient", "coefficients are nonfinite"),
        ("nonfinite_decision", "decision values are invalid"),
        ("boundary_probability", "strictly inside"),
    ],
)
def test_logistic_pathologies_fail_closed(
    monkeypatch: pytest.MonkeyPatch, mode: str, message: str
) -> None:
    rows = _rows()
    fit = rows.iloc[:60]
    predict = rows.iloc[60:65]
    monkeypatch.setattr(
        baseline,
        "LogisticRegression",
        lambda **kwargs: _FakeLogisticRegression(mode),
    )
    with pytest.raises(baseline.OENuisanceBaselineError, match=message):
        baseline._fit_predict(fit, predict, regularization_c=0.001)


def test_regularization_and_method_are_frozen_from_train_only() -> None:
    rows = _rows()
    oof = baseline.cross_fit_rows(rows)
    selection = oof.attrs["regularization_selection"]
    assert selection["selection_population"] == "train_only"
    assert selection["grid"] == list(baseline.REGULARIZATION_C_GRID)
    assert {0.0001, 0.001, 0.003, 0.01}.issubset(
        baseline.REGULARIZATION_C_GRID
    )
    assert set(oof["selected_regularization_C"]) == {selection["selected_C"]}
    assert set(oof["selected_nuisance_method"]) == {selection["selected_method"]}

    changed = rows.copy()
    changed.loc[changed["split"] != "train", "y_blue_win"] = (
        1 - changed.loc[changed["split"] != "train", "y_blue_win"]
    )
    replay = baseline.cross_fit_rows(changed)
    assert oof.attrs["regularization_selection"] == (
        replay.attrs["regularization_selection"]
    )
    assert set(replay["selected_regularization_C"]) == {selection["selected_C"]}


def test_train_inner_failure_freezes_intercept_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows()
    monkeypatch.setattr(baseline, "_improves_both", lambda scores: False)
    oof = baseline.cross_fit_rows(rows)
    assert oof.attrs["regularization_selection"]["selected_method"] == (
        "intercept_only"
    )
    np.testing.assert_array_equal(
        oof["p_blue_win_nuisance_oof"],
        oof["p_blue_win_intercept_oof"],
    )


def test_unavailable_train_inner_support_selects_intercept_only() -> None:
    rows = _rows(months=4)
    rows.loc[rows["oe_date_naive"] >= "2025-02", "split"] = "development"
    rows.loc[rows["oe_date_naive"] >= "2025-04", "split"] = "validation"
    train = rows[rows["split"] == "train"]
    selection = baseline.select_regularization(train)
    assert selection["selected_method"] == "intercept_only"
    assert selection["selected_C"] is None
    assert all(
        candidate == {
            "C": candidate["C"],
            "availability": "unavailable",
            "unavailable_reason": "no fold has sufficient strictly earlier support",
            "inner_oof_rows": 0,
            "scores": None,
            "improves_both_over_intercept": False,
        }
        for candidate in selection["candidates"]
    )
    oof = baseline.cross_fit_rows(rows)
    np.testing.assert_array_equal(
        oof["p_blue_win_nuisance_oof"],
        oof["p_blue_win_intercept_oof"],
    )


def test_persisted_artifact_has_pinned_authority_and_fail_closed_claims() -> None:
    payload = json.loads(baseline.DEFAULT_ARTIFACT_PATH.read_bytes())
    baseline.validate_artifact(payload)
    assert payload["source_identity"]["human_authority"]["reviewer_identity"] == "KOI_MARI"
    assert (
        payload["source_identity"]["human_authority"]["approved_action_used"]
        == "model_fit"
    )
    assert payload["status"] == "private_pending_rank_selection"
    assert payload["representation_rank_selected"] is False
    assert payload["authorizes_prediction"] is False
    assert payload["authorizes_publication"] is False
    assert payload["authorizes_sota_claim"] is False
    assert payload["descriptive_diagnostics"]["selection_use"] is False
    gate = payload["descriptive_diagnostics"]["outer_confirmation_gate"]
    assert gate["changes_frozen_nuisance_predictions"] is False
    assert gate["passed"] is True
    assert gate["eligible_for_downstream_rank_assay"] is True
    assert (
        payload["descriptive_diagnostics"]["outer_scores_do_not_change_predictions"]
        is True
    )
    assert payload["estimator"]["selected_C"] in baseline.REGULARIZATION_C_GRID
    assert payload["estimator"]["selected_C_scope"] == "global_train_frozen"
    assert payload["fold_contract"]["train_frozen_selection"] is True
    assert (
        payload["fold_contract"][
            "posthoc_development_or_validation_prediction_switch"
        ]
        is False
    )
    assert "already-opened non-holdout nuisance-only diagnostics" in payload[
        "design_history_disclosure"
    ]
    assert "No interaction-rank outcomes were inspected" in payload[
        "design_history_disclosure"
    ]
    assert payload["fold_contract"]["final_temporal_holdout"]["targets_read"] is False


def test_persisted_artifact_replays_in_a_clean_subprocess() -> None:
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = "731"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lol_kills.v2.draft.interactions.oe_nuisance_baseline",
            "--verify-existing",
        ],
        cwd=Path(__file__).resolve().parents[4],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["strict_replay_verified"] is True


def test_replay_does_not_mask_regenerated_oof_raw_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    persisted = json.loads(baseline.DEFAULT_ARTIFACT_PATH.read_bytes())

    def fake_build_artifact(**kwargs: object) -> dict[str, object]:
        replay_oof = Path(kwargs["oof_path"])
        replay_oof.write_bytes(b"deliberately-different-parquet-bytes")
        return copy.deepcopy(persisted)

    monkeypatch.setattr(baseline, "build_artifact", fake_build_artifact)
    with pytest.raises(
        baseline.OENuisanceBaselineError,
        match="regenerated OOF raw bytes do not match",
    ):
        baseline.load_and_replay_artifact(source_root=Path.cwd())


def test_rehashed_authority_drift_is_rejected(tmp_path: Path) -> None:
    authority = json.loads(baseline.DEFAULT_HUMAN_AUTHORITY_PATH.read_text())
    authority["decision_id"] += ":drift"
    path = tmp_path / "authority.json"
    path.write_bytes(
        json.dumps(
            authority, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode()
    )
    with pytest.raises(
        baseline.OENuisanceBaselineError, match="caller-rehashed"
    ):
        baseline.build_artifact(
            authority_path=path, oof_path=tmp_path / "oof.parquet"
        )


def test_artifact_mutations_fail_closed() -> None:
    payload = json.loads(baseline.DEFAULT_ARTIFACT_PATH.read_bytes())
    for field in (
        "representation_rank_selected",
        "authorizes_prediction",
        "authorizes_publication",
        "authorizes_sota_claim",
    ):
        changed = copy.deepcopy(payload)
        changed[field] = True
        changed.pop("artifact_sha256")
        changed["artifact_sha256"] = canonical_sha256(changed)
        with pytest.raises(
            baseline.OENuisanceBaselineError, match="authority ceiling"
        ):
            baseline.validate_artifact(changed)
    changed = copy.deepcopy(payload)
    changed["descriptive_diagnostics"]["prediction_selection_use"] = (
        "inner fold scores within each outer fold's strictly earlier support only"
    )
    changed.pop("artifact_sha256")
    changed["artifact_sha256"] = canonical_sha256(changed)
    with pytest.raises(
        baseline.OENuisanceBaselineError,
        match="prediction selection disclosure",
    ):
        baseline.validate_artifact(changed)
