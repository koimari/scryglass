"""Evaluate the public atomized Draft Score under a frozen time protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import optimize, sparse
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder

from lol_kills import draft_recommendation as draft_recommendation_module
from lol_kills.draft_recommendation import (
    _feature_rows as _draft_feature_rows,
    _recency_weights as _draft_recency_weights,
    _vocabulary as _draft_vocabulary,
    build_games as build_draft_games,
)
from lol_kills.research.atomized_rf_composite import (
    CATEGORICAL_CONTEXT_COLUMNS,
    GROUP_COLUMNS,
    MODEL_COLUMNS,
    RANDOM_SEED,
    _cluster_bootstrap_auc,
    _cluster_bootstrap_differences,
    metric_report,
)


SCHEMA_VERSION = "scryglass:public-draft-score-promotion-evaluation:v1"
FOREST_CACHE_SCHEMA_VERSION = "scryglass:public-draft-score-forest-cache:v1"
FOREST_WORLD_CACHE_SCHEMA_VERSION = (
    "scryglass:public-draft-score-forest-world-cache:v2"
)
CATEGORICAL_WORLD_CACHE_SCHEMA_VERSION = (
    "scryglass:public-draft-score-categorical-world-cache:v1"
)
STRENGTH_COLUMNS = (
    "blue_side",
    *GROUP_COLUMNS["team_rating"],
    *GROUP_COLUMNS["player_rating"],
    *GROUP_COLUMNS["rating_uncertainty"],
    *GROUP_COLUMNS["team_momentum"],
)
DRAFT_GROUPS = (
    "player_exact_performance",
    "player_overall_performance",
    "player_role_performance",
    "global_champion_performance",
    "global_champion_interactions",
    "exact_ally_enemy_pairs",
    "checkpoint_forecasts",
    "parity_conditioned_performance",
    "patch_exact_performance",
    "regional_draft_atoms",
)
CROSSFIT_COMPOSITION_COLUMNS = (
    "crossfit_composition_total",
    "crossfit_champion_main",
    "crossfit_role_champion",
    "crossfit_ally_synergy",
    "crossfit_archetype_synergy",
    "crossfit_enemy_counter",
    "crossfit_archetype_counter",
    "crossfit_same_role",
)
PHASE_TARGET_COLUMNS = tuple(
    f"target_{metric}_diff_{checkpoint}"
    for checkpoint in (10, 15, 20, 25)
    for metric in ("gold", "xp")
)


def _fit_bounded_draft_model(
    matrix: Any,
    outcomes: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> LogisticRegression:
    """Fit the sparse draft expert with finite, bounded coefficients."""

    row_count = int(matrix.shape[0])
    if row_count < 1 or matrix.shape[1] < 1:
        raise PublicDraftScorePromotionError("draft expert matrix is empty")
    target = np.asarray(outcomes, dtype=int)
    sample_weight = np.asarray(weights, dtype=float)
    if set(np.unique(target)) != {0, 1}:
        raise PublicDraftScorePromotionError("draft expert needs both outcomes")
    if not np.isfinite(sample_weight).all() or np.any(sample_weight <= 0):
        raise PublicDraftScorePromotionError("draft expert weights are invalid")
    if not math.isfinite(alpha) or alpha <= 0:
        raise PublicDraftScorePromotionError("draft expert alpha is invalid")

    model = LogisticRegression(
        C=1.0 / (float(alpha) * row_count),
        penalty="l2",
        solver="liblinear",
        dual=True,
        fit_intercept=True,
        intercept_scaling=1.0,
        max_iter=3000,
        tol=1e-7,
        random_state=RANDOM_SEED,
    )
    model.fit(matrix, target, sample_weight=sample_weight)
    if model.n_iter_[0] >= model.max_iter:
        raise PublicDraftScorePromotionError("draft expert did not converge")
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise PublicDraftScorePromotionError("draft expert coefficients are not finite")
    decision = np.asarray(model.decision_function(matrix), dtype=float)
    if not np.isfinite(decision).all():
        raise PublicDraftScorePromotionError("draft expert scores are not finite")
    return model


def _draft_component_columns(vocabulary: Mapping[str, int]) -> dict[str, np.ndarray]:
    prefixes = {
        "crossfit_champion_main": ("M|",),
        "crossfit_role_champion": ("MR|",),
        "crossfit_ally_synergy": ("S|",),
        "crossfit_archetype_synergy": ("AS|",),
        "crossfit_enemy_counter": ("C|",),
        "crossfit_archetype_counter": ("AC|",),
        "crossfit_same_role": ("R|",),
    }
    return {
        name: np.asarray(
            [column for key, column in vocabulary.items() if key.startswith(starts)],
            dtype=int,
        )
        for name, starts in prefixes.items()
    }


def _score_draft_components(
    matrix: Any,
    coefficient: np.ndarray,
    component_columns: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for name, columns in component_columns.items():
        output[name] = (
            np.asarray(matrix[:, columns] @ coefficient[columns]).reshape(-1)
            if columns.size
            else np.zeros(matrix.shape[0], dtype=float)
        )
    output["crossfit_composition_total"] = sum(output.values())
    return output


def _games_for_rows(
    rows: pd.DataFrame, game_by_id: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    missing: list[str] = []
    for game_uid in rows["game_uid"].astype(str):
        game = game_by_id.get(game_uid)
        if game is None:
            missing.append(game_uid)
        else:
            games.append(dict(game))
    if missing:
        raise PublicDraftScorePromotionError(
            f"player source misses {len(missing)} promotion drafts"
        )
    return games


def _add_crossfit_composition(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    game_by_id: Mapping[str, Mapping[str, Any]],
    *,
    alpha: float,
    half_life_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit composition terms on train outcomes and score both frames.

    Team, player, and league fields stabilize coefficient estimation. Their
    fitted values never enter the returned composition signal.
    """

    train_games = _games_for_rows(train, game_by_id)
    evaluation_games = _games_for_rows(evaluation, game_by_id)
    if len(train_games) < 300:
        raise PublicDraftScorePromotionError("too few drafts for cross-fit composition")
    vocabulary, _ = _draft_vocabulary(train_games)
    train_matrix = _draft_feature_rows(train_games, vocabulary)
    evaluation_matrix = _draft_feature_rows(evaluation_games, vocabulary)
    target = np.asarray([game["y"] for game in train_games], dtype=int)
    reference = train_games[-1]["date"]
    weights = _draft_recency_weights(train_games, reference, int(half_life_days))
    model = _fit_bounded_draft_model(train_matrix, target, weights, float(alpha))
    components = _draft_component_columns(vocabulary)
    train_values = _score_draft_components(train_matrix, model.coef_[0], components)
    evaluation_values = _score_draft_components(
        evaluation_matrix, model.coef_[0], components
    )
    train_output = train.copy()
    evaluation_output = evaluation.copy()
    for name in CROSSFIT_COMPOSITION_COLUMNS:
        train_output[name] = train_values[name]
        evaluation_output[name] = evaluation_values[name]
    train_output["crossfit_full_logit"] = model.decision_function(train_matrix)
    evaluation_output["crossfit_full_logit"] = model.decision_function(
        evaluation_matrix
    )
    receipt = {
        "alpha": float(alpha),
        "half_life_days": int(half_life_days),
        "training_rows": len(train_games),
        "training_end": pd.Timestamp(reference).isoformat(),
        "vocabulary_sha256": canonical_sha256(sorted(vocabulary)),
        "training_game_ids_sha256": canonical_sha256(
            [game["game_uid"] for game in train_games]
        ),
        "excluded_control_prefixes": ["L|", "T|", "P|"],
    }
    return train_output, evaluation_output, receipt


def _draft_expert_logits(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    game_by_id: Mapping[str, Mapping[str, Any]],
    configs: Sequence[Mapping[str, Any]],
    *,
    shuffled: bool = False,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    train_games = _games_for_rows(train, game_by_id)
    evaluation_games = _games_for_rows(evaluation, game_by_id)
    vocabulary, _ = _draft_vocabulary(train_games)
    train_matrix = _draft_feature_rows(train_games, vocabulary)
    evaluation_matrix = _draft_feature_rows(evaluation_games, vocabulary)
    target = np.asarray([game["y"] for game in train_games], dtype=int)
    if shuffled:
        target = np.random.default_rng(RANDOM_SEED).permutation(target)
    component_columns = _draft_component_columns(vocabulary)
    values: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    for config in configs:
        weights = _draft_recency_weights(
            train_games,
            train_games[-1]["date"],
            int(config["half_life_days"]),
        )
        model = _fit_bounded_draft_model(
            train_matrix, target, weights, float(config["alpha"])
        )
        components = _score_draft_components(
            evaluation_matrix, model.coef_[0], component_columns
        )
        values.extend(
            [
                np.asarray(model.decision_function(evaluation_matrix), dtype=float),
                np.asarray(components["crossfit_composition_total"], dtype=float),
            ]
        )
        receipts.append(
            {
                "id": str(config["id"]),
                "alpha": float(config["alpha"]),
                "half_life_days": int(config["half_life_days"]),
                "vocabulary_sha256": canonical_sha256(sorted(vocabulary)),
                "training_game_ids_sha256": canonical_sha256(
                    [game["game_uid"] for game in train_games]
                ),
                "outputs": ["full_logit", "composition_only_logit"],
            }
        )
    return np.column_stack(values), receipts


def _anchor_probability(
    frame: pd.DataFrame,
    *,
    team_weight: float,
    momentum_weight: float,
    source: str = "shrunk_probability",
) -> np.ndarray:
    if source == "shrunk_probability":
        team = frame["base_team_logit"].astype(float).to_numpy()
        player = frame["base_player_logit"].astype(float).to_numpy()
    elif source == "raw_rating_difference":
        elo_logit_scale = math.log(10.0)
        team = (
            elo_logit_scale
            * frame["team_rating_diff_scaled"].astype(float).to_numpy()
        )
        player = (
            elo_logit_scale
            * frame["player_rating_diff_scaled"].astype(float).to_numpy()
        )
    else:
        raise PublicDraftScorePromotionError(f"unknown anchor source: {source}")
    momentum = (
        frame["team_momentum_points_diff"].astype(float).to_numpy()
        + frame["player_momentum_points_diff"].astype(float).to_numpy()
    ) / 400.0
    return _sigmoid(
        float(team_weight) * team
        + (1.0 - float(team_weight)) * player
        + float(momentum_weight) * momentum
    )


def _candidate_columns(
    groups: Sequence[str],
    *,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    group_columns: Mapping[str, Sequence[str]] = feature_groups or GROUP_COLUMNS
    requested = (
        "team_rating",
        "player_rating",
        "rating_uncertainty",
        "team_momentum",
        "team_macro_form",
        "competition_context",
        "match_context",
        *groups,
    )
    output = ["blue_side"]
    for group in requested:
        if group not in group_columns:
            if group in groups:
                raise PublicDraftScorePromotionError(
                    f"frozen feature inventory misses group: {group}"
                )
            continue
        if group == "crossfit_composition":
            output.extend(CROSSFIT_COMPOSITION_COLUMNS)
        else:
            output.extend(group_columns[group])
    return tuple(dict.fromkeys(output))


class PublicDraftScorePromotionError(ValueError):
    """Raised when the frozen promotion protocol cannot be executed."""


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
            default=_json_default,
        ).encode("utf-8")
    ).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise TypeError(f"unsupported receipt value: {type(value).__name__}")


def _load_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    inherited = document.get("inherits")
    if not inherited:
        return document, {"path": str(path), "sha256": sha256_path(path)}
    base_path = path.parent / str(inherited)
    base, base_receipt = _load_protocol(base_path)
    resolved = {**base, **{key: value for key, value in document.items() if key != "inherits"}}
    selection_extension = resolved.pop("selection_extension", None)
    if selection_extension is not None:
        resolved["selection"] = {**base["selection"], **selection_extension}
    return resolved, {
        "path": str(path),
        "sha256": sha256_path(path),
        "inherits": base_receipt,
    }


def _validate_protocol_matrix_binding(
    protocol: Mapping[str, Any], expected_matrix_sha256: str
) -> None:
    iteration = protocol.get("iteration")
    if not isinstance(iteration, Mapping) or "matrix_sha256" not in iteration:
        return
    frozen_matrix_sha256 = str(iteration["matrix_sha256"])
    if not re.fullmatch(r"[0-9a-f]{64}", frozen_matrix_sha256):
        raise PublicDraftScorePromotionError(
            "promotion protocol matrix SHA-256 is invalid"
        )
    if frozen_matrix_sha256 != expected_matrix_sha256:
        raise PublicDraftScorePromotionError(
            "promotion matrix does not match the frozen protocol"
        )


def _validate_matrix_manifest(
    *,
    protocol: Mapping[str, Any],
    manifest_path: Path | None,
    expected_manifest_sha256: str | None,
    expected_matrix_sha256: str,
) -> dict[str, Any] | None:
    iteration = protocol.get("iteration")
    frozen_manifest_sha256 = (
        str(iteration.get("matrix_manifest_sha256"))
        if isinstance(iteration, Mapping)
        and iteration.get("matrix_manifest_sha256") is not None
        else None
    )
    if manifest_path is None and expected_manifest_sha256 is None:
        if frozen_manifest_sha256 is not None:
            raise PublicDraftScorePromotionError(
                "frozen protocol requires the matrix manifest"
            )
        return None
    if manifest_path is None or expected_manifest_sha256 is None:
        raise PublicDraftScorePromotionError(
            "matrix manifest path and SHA-256 must be supplied together"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise PublicDraftScorePromotionError("matrix manifest SHA-256 is invalid")
    if frozen_manifest_sha256 not in {None, expected_manifest_sha256}:
        raise PublicDraftScorePromotionError(
            "matrix manifest does not match the frozen protocol"
        )
    if sha256_path(manifest_path) != expected_manifest_sha256:
        raise PublicDraftScorePromotionError("matrix manifest SHA-256 changed")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublicDraftScorePromotionError("matrix manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise PublicDraftScorePromotionError("matrix manifest is invalid")
    expected_schema = str(protocol["feature_contract"]["schema_version"])
    if manifest.get("schema_version") != expected_schema:
        raise PublicDraftScorePromotionError("matrix feature schema changed")
    if manifest.get("matrix_sha256") != expected_matrix_sha256:
        raise PublicDraftScorePromotionError("matrix manifest binds another matrix")
    expected_model_columns = list(MODEL_COLUMNS)
    expected_category_columns = list(CATEGORICAL_CONTEXT_COLUMNS)
    if manifest.get("model_columns") != expected_model_columns:
        raise PublicDraftScorePromotionError("matrix model columns changed")
    if manifest.get("categorical_columns") != expected_category_columns:
        raise PublicDraftScorePromotionError("matrix categorical columns changed")
    if manifest.get("columns") != [
        *expected_model_columns,
        *expected_category_columns,
    ]:
        raise PublicDraftScorePromotionError("matrix column inventory changed")
    return {
        "path": str(manifest_path),
        "sha256": expected_manifest_sha256,
        "schema_version": expected_schema,
    }


def _clip(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)


def _logit(probability: np.ndarray) -> np.ndarray:
    value = _clip(probability)
    return np.log(value / (1.0 - value))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-value))


def _blend(
    full_probability: np.ndarray,
    strength_probability: np.ndarray,
    weight: float,
) -> np.ndarray:
    return _sigmoid(
        float(weight) * _logit(full_probability)
        + (1.0 - float(weight)) * _logit(strength_probability)
    )


@dataclass(frozen=True)
class Calibration:
    accepted: bool
    slope: float = 1.0
    intercept: float = 0.0

    def apply(self, probability: np.ndarray) -> np.ndarray:
        if not self.accepted:
            return _clip(probability)
        return _sigmoid(self.slope * _logit(probability) + self.intercept)

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "slope": self.slope,
            "intercept": self.intercept,
        }


@dataclass(frozen=True)
class BoundedLogisticModel:
    coefficient: np.ndarray
    intercept: float

    @property
    def coef_(self) -> np.ndarray:
        return self.coefficient.reshape(1, -1)

    @property
    def intercept_(self) -> np.ndarray:
        return np.asarray([self.intercept], dtype=float)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=float)
        score = np.sum(values * self.coefficient[None, :], axis=1) + self.intercept
        if not np.isfinite(score).all():
            raise PublicDraftScorePromotionError(
                "bounded logistic scores are not finite"
            )
        probability = _sigmoid(score)
        return np.column_stack([1.0 - probability, probability])


def _fit_calibration(target: pd.Series, probability: np.ndarray) -> Calibration:
    raw = _clip(probability)
    if target.nunique() != 2:
        return Calibration(False)
    model = LogisticRegression(
        C=100.0,
        solver="liblinear",
        max_iter=2000,
        tol=1e-7,
        random_state=RANDOM_SEED,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model.fit(_logit(raw).reshape(-1, 1), target.astype(int))
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise PublicDraftScorePromotionError("calibration coefficients are not finite")
    candidate = _clip(model.predict_proba(_logit(raw).reshape(-1, 1))[:, 1])
    raw_brier = brier_score_loss(target, raw)
    raw_log_loss = log_loss(target, raw, labels=[0, 1])
    candidate_brier = brier_score_loss(target, candidate)
    candidate_log_loss = log_loss(target, candidate, labels=[0, 1])
    accepted = candidate_brier <= raw_brier and candidate_log_loss <= raw_log_loss
    return Calibration(
        accepted,
        float(model.coef_[0, 0]) if accepted else 1.0,
        float(model.intercept_[0]) if accepted else 0.0,
    )


def _rf(config: Mapping[str, Any]) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=float(config["max_features"]),
        max_samples=float(config["max_samples"]),
        class_weight=config.get("class_weight"),
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )


def _learner(config: Mapping[str, Any]) -> Any:
    if config.get("learner", "random_forest") == "random_forest":
        return _rf(config)
    if config["learner"] == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            num_leaves=int(config["num_leaves"]),
            max_depth=int(config["max_depth"]),
            min_child_samples=int(config["min_child_samples"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            reg_alpha=float(config["reg_alpha"]),
        reg_lambda=float(config["reg_lambda"]),
        subsample_freq=1,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
        )
    raise PublicDraftScorePromotionError(f"unknown learner: {config['learner']}")


def _mirror_features(
    frame: pd.DataFrame, columns: Sequence[str]
) -> pd.DataFrame:
    """Return the exact red-side view of signed pre-match features."""

    values = frame[list(columns)].astype(float)
    mirrored = values.copy()
    invariant_exact = {
        "player_lineup_complete",
        "series_map_index",
        "h2h_count_g10",
        "h2h_available",
        "forecast_peak_magnitude",
    }
    for column in columns:
        blue_red_pair: str | None = None
        if column.endswith("_blue"):
            candidate = f"{column[:-5]}_red"
            if candidate in values:
                blue_red_pair = candidate
        elif column.endswith("_red"):
            candidate = f"{column[:-4]}_blue"
            if candidate in values:
                blue_red_pair = candidate
        if blue_red_pair is not None:
            mirrored[column] = values[blue_red_pair]
            continue
        invariant = (
            column in invariant_exact
            or column.startswith(("context_", "availability_", "rating_"))
            or any(
                marker in column
                for marker in (
                    "_support",
                    "_coverage",
                    "_missing",
                    "_available",
                )
            )
            or column.endswith("_min")
            or (
                "_count" in column
                and not column.endswith("_count_difference")
            )
        )
        mirrored[column] = values[column] if invariant else -values[column]
    return mirrored


def _fit_probability(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    columns: Sequence[str],
    config: Mapping[str, Any],
    *,
    shuffled: bool = False,
) -> np.ndarray:
    target = train["y"].astype(int).to_numpy()
    if shuffled:
        target = np.random.default_rng(RANDOM_SEED).permutation(target)
    model = _learner(config)
    train_values = train[list(columns)].astype(float)
    evaluation_values = evaluation[list(columns)].astype(float)
    if config.get("side_symmetry_augmentation", False):
        mirrored_train = _mirror_features(train, columns)
        model.fit(
            pd.concat([train_values, mirrored_train], ignore_index=True),
            np.concatenate([target, 1 - target]),
        )
        direct = model.predict_proba(evaluation_values)[:, 1]
        reverse = model.predict_proba(_mirror_features(evaluation, columns))[:, 1]
        return _clip(0.5 * (direct + (1.0 - reverse)))
    model.fit(train_values, target)
    return _clip(model.predict_proba(evaluation_values)[:, 1])


def _series_bounds(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("series_id", sort=False)["date"].agg(["min", "max"])


def _series_slice(
    frame: pd.DataFrame,
    bounds: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if start is None:
        series = bounds[bounds["max"].lt(end)].index
    else:
        series = bounds[bounds["min"].ge(start) & bounds["max"].lt(end)].index
    return frame[frame["series_id"].isin(series)].copy()


def _validate_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    required = set(MODEL_COLUMNS) | set(STRENGTH_COLUMNS) | {
        "game_uid",
        "series_id",
        "date",
        "league",
        "source_patch",
        "y",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise PublicDraftScorePromotionError(f"promotion matrix misses columns: {missing}")
    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], utc=True, errors="raise")
    work = work.sort_values(["date", "game_uid"], kind="stable").reset_index(drop=True)
    if work["game_uid"].astype(str).duplicated().any():
        raise PublicDraftScorePromotionError("promotion matrix contains duplicate maps")
    if not work["y"].isin([0, 1]).all():
        raise PublicDraftScorePromotionError("promotion target is not binary")
    if work[list(MODEL_COLUMNS)].isna().any().any():
        raise PublicDraftScorePromotionError("promotion features contain null values")
    forbidden = [
        column
        for column in MODEL_COLUMNS
        if column.startswith("target_")
        or column.startswith("observed_")
        or column in {"y", "result", "gold_diff", "xp_diff"}
    ]
    if forbidden:
        raise PublicDraftScorePromotionError(
            f"current state or outcome entered model columns: {forbidden}"
        )
    return work


def _select_configuration(
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if protocol["selection"].get("architecture") == "rating_anchor_rf_v1":
        return _select_rating_anchor_configuration(
            inner_train, inner_validation, protocol
        )
    rows: list[dict[str, Any]] = []
    for config in protocol["selection"]["configs"]:
        full = _fit_probability(inner_train, inner_validation, MODEL_COLUMNS, config)
        strength = _fit_probability(
            inner_train, inner_validation, STRENGTH_COLUMNS, config
        )
        for weight in protocol["selection"]["blend_weights"]:
            raw = _blend(full, strength, float(weight))
            calibration = _fit_calibration(inner_validation["y"], raw)
            probability = calibration.apply(raw)
            metrics = metric_report(inner_validation["y"], probability)
            rows.append(
                {
                    "config_id": config["id"],
                    "config": dict(config),
                    "blend_weight": float(weight),
                    "calibration": calibration.as_dict(),
                    **metrics,
                }
            )
    selected = min(
        rows,
        key=lambda row: (
            float(row["log_loss"]),
            -float(row["auc"]),
            float(row["brier"]),
            str(row["config_id"]),
            float(row["blend_weight"]),
        ),
    )
    return selected, {"candidates": rows, "selected": selected}


def _select_rating_anchor_configuration(
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    selection = protocol["selection"]
    anchor_source = str(selection.get("anchor_source", "shrunk_probability"))
    anchors: list[dict[str, Any]] = []
    for team_weight in selection["anchor_team_weights"]:
        for momentum_weight in selection["anchor_momentum_weights"]:
            raw = _anchor_probability(
                inner_validation,
                team_weight=float(team_weight),
                momentum_weight=float(momentum_weight),
                source=anchor_source,
            )
            calibration = _fit_calibration(inner_validation["y"], raw)
            probability = calibration.apply(raw)
            anchors.append(
                {
                    "team_weight": float(team_weight),
                    "momentum_weight": float(momentum_weight),
                    "calibration": calibration.as_dict(),
                    **metric_report(inner_validation["y"], probability),
                }
            )
    baseline = min(
        anchors,
        key=lambda row: (
            float(row["log_loss"]),
            -float(row["auc"]),
            float(row["brier"]),
            float(row["team_weight"]),
            float(row["momentum_weight"]),
        ),
    )
    rows: list[dict[str, Any]] = []
    fitted: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
    for group_set in selection["draft_group_sets"]:
        groups = tuple(str(group) for group in group_set["groups"])
        columns = _candidate_columns(groups)
        for config in selection["configs"]:
            key = (str(config["id"]), columns)
            if key not in fitted:
                fitted[key] = _fit_probability(
                    inner_train, inner_validation, columns, config
                )
            full = fitted[key]
            for anchor in anchors:
                anchor_raw = _anchor_probability(
                    inner_validation,
                    team_weight=anchor["team_weight"],
                    momentum_weight=anchor["momentum_weight"],
                    source=anchor_source,
                )
                for weight in selection["blend_weights"]:
                    raw = _blend(full, anchor_raw, float(weight))
                    calibration = _fit_calibration(inner_validation["y"], raw)
                    probability = calibration.apply(raw)
                    rows.append(
                        {
                            "config_id": config["id"],
                            "config": dict(config),
                            "group_set_id": group_set["id"],
                            "groups": list(groups),
                            "columns": list(columns),
                            "anchor_team_weight": anchor["team_weight"],
                            "anchor_momentum_weight": anchor["momentum_weight"],
                            "blend_weight": float(weight),
                            "calibration": calibration.as_dict(),
                            **metric_report(inner_validation["y"], probability),
                        }
                    )
    selected = min(
        rows,
        key=lambda row: (
            float(row["log_loss"]),
            -float(row["auc"]),
            float(row["brier"]),
            str(row["config_id"]),
            str(row["group_set_id"]),
            float(row["blend_weight"]),
        ),
    )
    return selected, {
        "architecture": "rating_anchor_rf_v1",
        "anchors": anchors,
        "selected_baseline": baseline,
        "candidates": rows,
        "selected": selected,
    }


def _select_anchor(
    inner_validation: pd.DataFrame, protocol: Mapping[str, Any]
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    selection = protocol["selection"]
    anchor_source = str(selection.get("anchor_source", "shrunk_probability"))
    for team_weight in selection["anchor_team_weights"]:
        for momentum_weight in selection["anchor_momentum_weights"]:
            raw = _anchor_probability(
                inner_validation,
                team_weight=float(team_weight),
                momentum_weight=float(momentum_weight),
                source=anchor_source,
            )
            calibration = _fit_calibration(inner_validation["y"], raw)
            probability = calibration.apply(raw)
            rows.append(
                {
                    "source": anchor_source,
                    "team_weight": float(team_weight),
                    "momentum_weight": float(momentum_weight),
                    "calibration": calibration.as_dict(),
                    **metric_report(inner_validation["y"], probability),
                }
            )
    return min(
        rows,
        key=lambda row: (
            float(row["log_loss"]),
            -float(row["auc"]),
            float(row["brier"]),
            float(row["team_weight"]),
            float(row["momentum_weight"]),
        ),
    )


def _quantum_world_columns(
    groups: Sequence[str],
    *,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    return _candidate_columns(
        tuple(str(group) for group in groups), feature_groups=feature_groups
    )


def _quantum_world_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    shuffled: bool = False,
    cache_dir: Path | None = None,
    matrix_sha256: str | None = None,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    config = protocol["selection"]["quantum_forest_config"]
    world_definitions: list[dict[str, Any]] = []
    for world in protocol["selection"]["quantum_worlds"]:
        groups = tuple(str(group) for group in world["groups"])
        columns = _quantum_world_columns(groups, feature_groups=feature_groups)
        world_definitions.append(
            {
                "world_id": str(world["id"]),
                "groups": list(groups),
                "columns_sha256": canonical_sha256(list(columns)),
            }
        )
    receipts = [
        {
            **world,
            "anonymous_id": canonical_sha256(
                {
                    "cache_schema": FOREST_CACHE_SCHEMA_VERSION,
                    **world,
                }
            )[:16],
        }
        for world in world_definitions
    ]
    target = train["y"].astype(int).to_numpy()
    if shuffled:
        target = np.random.default_rng(RANDOM_SEED).permutation(target)
    cache_common: dict[str, Any] | None = None
    world_cache_paths: list[Path] = []
    if cache_dir is not None:
        if matrix_sha256 is None or not re.fullmatch(
            r"[0-9a-f]{64}", str(matrix_sha256)
        ):
            raise PublicDraftScorePromotionError(
                "forest cache requires the bound matrix SHA-256"
            )
        cache_common = {
            "matrix_sha256": matrix_sha256,
            "train_game_ids_sha256": canonical_sha256(
                train["game_uid"].astype(str).tolist()
            ),
            "evaluation_game_ids_sha256": canonical_sha256(
                evaluation["game_uid"].astype(str).tolist()
            ),
            "target_sha256": hashlib.sha256(
                np.asarray(target, dtype="<i8").tobytes()
            ).hexdigest(),
            "config": dict(config),
            "random_seed": RANDOM_SEED,
        }
        world_cache_root = cache_dir / "world-predictions-v2"
        world_cache_root.mkdir(parents=True, exist_ok=True)
        world_cache_paths = [
            world_cache_root
            / f"{canonical_sha256({**cache_common, 'schema_version': FOREST_WORLD_CACHE_SCHEMA_VERSION, 'world': world})}.npz"
            for world in world_definitions
        ]

        # Read the original bundled cache once, then split it into stable
        # per-world entries. This preserves prior computation when a later
        # frozen protocol adds one world without changing existing worlds.
        if not all(path.exists() for path in world_cache_paths):
            legacy_key = canonical_sha256(
                {
                    **cache_common,
                    "schema_version": FOREST_CACHE_SCHEMA_VERSION,
                    "worlds": world_definitions,
                }
            )
            legacy_path = (
                cache_dir / "world-predictions" / f"{legacy_key}.npz"
            )
            if legacy_path.exists():
                try:
                    with np.load(legacy_path, allow_pickle=False) as cached:
                        legacy_values = np.asarray(cached["logits"], dtype=float)
                        legacy_receipts = json.loads(
                            np.asarray(cached["receipts"], dtype=np.uint8)
                            .tobytes()
                            .decode("utf-8")
                        )
                    if legacy_values.shape != (
                        len(evaluation),
                        len(world_definitions),
                    ) or not np.isfinite(legacy_values).all():
                        raise PublicDraftScorePromotionError(
                            "forest cache is invalid"
                        )
                    if legacy_receipts != receipts:
                        raise PublicDraftScorePromotionError(
                            "forest cache receipt changed"
                        )
                    for index, path in enumerate(world_cache_paths):
                        receipt_bytes = np.frombuffer(
                            json.dumps(
                                receipts[index],
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("utf-8"),
                            dtype=np.uint8,
                        )
                        temporary = path.with_suffix(".tmp.npz")
                        np.savez_compressed(
                            temporary,
                            logits=legacy_values[:, index],
                            receipt=receipt_bytes,
                        )
                        temporary.replace(path)
                except PublicDraftScorePromotionError:
                    raise
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    raise PublicDraftScorePromotionError(
                        "forest cache is invalid"
                    ) from error

    predictions: list[np.ndarray] = []
    for index, world in enumerate(world_definitions):
        cache_path = world_cache_paths[index] if world_cache_paths else None
        if cache_path is not None and cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    values = np.asarray(cached["logits"], dtype=float)
                    cached_receipt = json.loads(
                        np.asarray(cached["receipt"], dtype=np.uint8)
                        .tobytes()
                        .decode("utf-8")
                    )
                if values.shape != (len(evaluation),) or not np.isfinite(
                    values
                ).all():
                    raise PublicDraftScorePromotionError(
                        "forest cache is invalid"
                    )
                if cached_receipt != receipts[index]:
                    raise PublicDraftScorePromotionError(
                        "forest cache receipt changed"
                    )
                predictions.append(values)
                continue
            except PublicDraftScorePromotionError:
                raise
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise PublicDraftScorePromotionError(
                    "forest cache is invalid"
                ) from error
        groups = tuple(world["groups"])
        columns = _quantum_world_columns(groups, feature_groups=feature_groups)
        values = _logit(
            _fit_probability(
                train, evaluation, columns, config, shuffled=shuffled
            )
        )
        predictions.append(values)
        if cache_path is not None:
            receipt_bytes = np.frombuffer(
                json.dumps(
                    receipts[index], sort_keys=True, separators=(",", ":")
                ).encode("utf-8"),
                dtype=np.uint8,
            )
            temporary = cache_path.with_suffix(".tmp.npz")
            np.savez_compressed(
                temporary, logits=values, receipt=receipt_bytes
            )
            temporary.replace(cache_path)
    return np.column_stack(predictions), receipts


def _encoded_categorical_frame(
    frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
    vocabularies: Mapping[str, Mapping[str, int]] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    missing = sorted(set(CATEGORICAL_CONTEXT_COLUMNS) - set(frame.columns))
    if missing:
        raise PublicDraftScorePromotionError(
            f"categorical world misses fields: {missing}"
        )
    output = frame[list(numeric_columns)].astype(float).reset_index(drop=True)
    cardinality: dict[str, int] = {}
    for column in CATEGORICAL_CONTEXT_COLUMNS:
        values = frame[column].astype(str)
        vocabulary = (
            dict(vocabularies[column])
            if vocabularies is not None and column in vocabularies
            else {
                value: index
                for index, value in enumerate(sorted(set(values)))
            }
        )
        encoded = values.map(vocabulary)
        if encoded.isna().any():
            raise PublicDraftScorePromotionError(
                f"categorical vocabulary misses a value in {column}"
            )
        output[column] = encoded.astype("int32").to_numpy()
        cardinality[column] = len(vocabulary)
    if not np.isfinite(output[list(numeric_columns)].to_numpy(dtype=float)).all():
        raise PublicDraftScorePromotionError(
            "categorical world numeric inputs are invalid"
        )
    return output, cardinality


def _mirror_categorical_frame(
    frame: pd.DataFrame,
    *,
    numeric_columns: Sequence[str],
) -> pd.DataFrame:
    mirrored = frame.copy().reset_index(drop=True)
    mirrored.loc[:, list(numeric_columns)] = _mirror_features(
        frame, numeric_columns
    ).to_numpy()
    for column in CATEGORICAL_CONTEXT_COLUMNS:
        if column.startswith("category_blue_"):
            counterpart = f"category_red_{column.removeprefix('category_blue_')}"
            mirrored[column] = frame[counterpart].astype(str).to_numpy()
        elif column.startswith("category_red_"):
            counterpart = f"category_blue_{column.removeprefix('category_red_')}"
            mirrored[column] = frame[counterpart].astype(str).to_numpy()
        elif column == "category_first_pick_side":
            mirrored[column] = frame[column].astype(str).map(
                {"blue": "red", "red": "blue"}
            ).fillna(frame[column].astype(str))
        else:
            mirrored[column] = frame[column].astype(str).to_numpy()
    return mirrored


def _categorical_world_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    shuffled: bool = False,
    cache_dir: Path | None = None,
    matrix_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit one fold-local LightGBM world over exact pre-match identities."""

    config = protocol["selection"]["categorical_world_config"]
    groups = tuple(str(group) for group in config["groups"])
    numeric_columns = _candidate_columns(groups)
    side_symmetry = bool(config.get("side_symmetry_augmentation", False))
    fit_frame = (
        pd.concat(
            [
                train,
                _mirror_categorical_frame(
                    train, numeric_columns=numeric_columns
                ),
            ],
            ignore_index=True,
        )
        if side_symmetry
        else train
    )
    evaluation_mirror = (
        _mirror_categorical_frame(
            evaluation, numeric_columns=numeric_columns
        )
        if side_symmetry
        else None
    )
    evaluation_audit_frame = (
        pd.concat([evaluation, evaluation_mirror], ignore_index=True)
        if evaluation_mirror is not None
        else evaluation
    )
    learner = str(config.get("learner", "lightgbm_categorical_splits"))
    if learner not in {
        "lightgbm_categorical_splits",
        "extra_trees_onehot",
    }:
        raise PublicDraftScorePromotionError(
            f"unknown categorical world learner: {learner}"
        )
    category_audit: dict[str, dict[str, int]] = {}
    vocabularies: dict[str, dict[str, int]] = {}
    for column in CATEGORICAL_CONTEXT_COLUMNS:
        if column not in fit_frame.columns or column not in evaluation.columns:
            raise PublicDraftScorePromotionError(
                f"categorical world misses field: {column}"
            )
        train_vocabulary = set(fit_frame[column].astype(str))
        evaluation_vocabulary = set(evaluation_audit_frame[column].astype(str))
        combined_vocabulary = train_vocabulary | evaluation_vocabulary
        encoding_vocabulary = (
            train_vocabulary
            if learner == "extra_trees_onehot"
            else combined_vocabulary
        )
        vocabularies[column] = {
            value: index
            for index, value in enumerate(sorted(encoding_vocabulary))
        }
        category_audit[column] = {
            "train": len(train_vocabulary),
            "evaluation": len(evaluation_vocabulary),
            "evaluation_unseen": len(evaluation_vocabulary - train_vocabulary),
        }
    train_values: pd.DataFrame | sparse.csr_matrix
    evaluation_values: pd.DataFrame | sparse.csr_matrix
    mirror_values: pd.DataFrame | sparse.csr_matrix | None = None
    if learner == "lightgbm_categorical_splits":
        train_values, _ = _encoded_categorical_frame(
            fit_frame,
            numeric_columns=numeric_columns,
            vocabularies=vocabularies,
        )
        evaluation_values, _ = _encoded_categorical_frame(
            evaluation,
            numeric_columns=numeric_columns,
            vocabularies=vocabularies,
        )
        if evaluation_mirror is not None:
            mirror_values, _ = _encoded_categorical_frame(
                evaluation_mirror,
                numeric_columns=numeric_columns,
                vocabularies=vocabularies,
            )
    else:
        categories = [
            sorted(vocabularies[column])
            for column in CATEGORICAL_CONTEXT_COLUMNS
        ]
        encoder = OneHotEncoder(
            categories=categories,
            handle_unknown="ignore",
            sparse_output=True,
            dtype=np.float32,
        )
        train_categories = encoder.fit_transform(
            fit_frame[list(CATEGORICAL_CONTEXT_COLUMNS)].astype(str)
        )
        evaluation_categories = encoder.transform(
            evaluation[list(CATEGORICAL_CONTEXT_COLUMNS)].astype(str)
        )
        train_values = sparse.hstack(
            [
                sparse.csr_matrix(
                    fit_frame[list(numeric_columns)].to_numpy(
                        dtype=np.float32
                    )
                ),
                train_categories,
            ],
            format="csr",
            dtype=np.float32,
        )
        evaluation_values = sparse.hstack(
            [
                sparse.csr_matrix(
                    evaluation[list(numeric_columns)].to_numpy(
                        dtype=np.float32
                    )
                ),
                evaluation_categories,
            ],
            format="csr",
            dtype=np.float32,
        )
        if evaluation_mirror is not None:
            mirror_values = sparse.hstack(
                [
                    sparse.csr_matrix(
                        evaluation_mirror[list(numeric_columns)].to_numpy(
                            dtype=np.float32
                        )
                    ),
                    encoder.transform(
                        evaluation_mirror[
                            list(CATEGORICAL_CONTEXT_COLUMNS)
                        ].astype(str)
                    ),
                ],
                format="csr",
                dtype=np.float32,
            )
    train_cardinality = {
        column: audit["train"] for column, audit in category_audit.items()
    }
    evaluation_cardinality = {
        column: audit["evaluation"]
        for column, audit in category_audit.items()
    }
    encoding_cardinality = {
        column: len(vocabulary)
        for column, vocabulary in vocabularies.items()
    }
    target = train["y"].astype(int).to_numpy()
    if shuffled:
        target = np.random.default_rng(RANDOM_SEED).permutation(target)
    if side_symmetry:
        target = np.concatenate([target, 1 - target])
    receipt = {
        "id": str(config["id"]),
        "learner": learner,
        "groups": list(groups),
        "side_symmetry_augmentation": side_symmetry,
        "numeric_columns_sha256": canonical_sha256(list(numeric_columns)),
        "categorical_columns_sha256": canonical_sha256(
            list(CATEGORICAL_CONTEXT_COLUMNS)
        ),
        "train_cardinality": train_cardinality,
        "evaluation_cardinality": evaluation_cardinality,
        "encoding_cardinality": encoding_cardinality,
        "category_audit": category_audit,
        "category_vocabulary_sha256": canonical_sha256(
            {
                column: sorted(vocabulary)
                for column, vocabulary in vocabularies.items()
            }
        ),
        "training_game_ids_sha256": canonical_sha256(
            train["game_uid"].astype(str).tolist()
        ),
        "evaluation_game_ids_sha256": canonical_sha256(
            evaluation["game_uid"].astype(str).tolist()
        ),
    }
    cache_path: Path | None = None
    if cache_dir is not None:
        if matrix_sha256 is None or not re.fullmatch(
            r"[0-9a-f]{64}", str(matrix_sha256)
        ):
            raise PublicDraftScorePromotionError(
                "categorical world cache requires the bound matrix SHA-256"
            )
        cache_key = canonical_sha256(
            {
                "schema_version": CATEGORICAL_WORLD_CACHE_SCHEMA_VERSION,
                "matrix_sha256": matrix_sha256,
                "config": dict(config),
                "target_sha256": hashlib.sha256(
                    np.asarray(target, dtype="<i8").tobytes()
                ).hexdigest(),
                "receipt": receipt,
                "random_seed": RANDOM_SEED,
            }
        )
        cache_root = cache_dir / "categorical-world-predictions"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{cache_key}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    values = np.asarray(cached["logits"], dtype=float)
                    cached_receipt = json.loads(
                        np.asarray(cached["receipt"], dtype=np.uint8)
                        .tobytes()
                        .decode("utf-8")
                    )
                if values.shape != (len(evaluation),) or not np.isfinite(
                    values
                ).all():
                    raise PublicDraftScorePromotionError(
                        "categorical world cache is invalid"
                    )
                if cached_receipt != receipt:
                    raise PublicDraftScorePromotionError(
                        "categorical world cache receipt changed"
                    )
                return values, receipt
            except PublicDraftScorePromotionError:
                raise
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise PublicDraftScorePromotionError(
                    "categorical world cache is invalid"
                ) from error

    if learner == "lightgbm_categorical_splits":
        from lightgbm import LGBMClassifier

        model: Any = LGBMClassifier(
            n_estimators=int(config["n_estimators"]),
            learning_rate=float(config["learning_rate"]),
            num_leaves=int(config["num_leaves"]),
            max_depth=int(config["max_depth"]),
            min_child_samples=int(config["min_child_samples"]),
            subsample=float(config["subsample"]),
            colsample_bytree=float(config["colsample_bytree"]),
            reg_alpha=float(config["reg_alpha"]),
            reg_lambda=float(config["reg_lambda"]),
            cat_smooth=float(config["cat_smooth"]),
            cat_l2=float(config["cat_l2"]),
            subsample_freq=1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        model.fit(
            train_values,
            target,
            categorical_feature=list(CATEGORICAL_CONTEXT_COLUMNS),
        )
    else:
        model = ExtraTreesClassifier(
            n_estimators=int(config["n_estimators"]),
            max_depth=(
                None
                if config.get("max_depth") is None
                else int(config["max_depth"])
            ),
            min_samples_leaf=int(config["min_samples_leaf"]),
            max_features=float(config["max_features"]),
            max_samples=float(config["max_samples"]),
            class_weight=config.get("class_weight"),
            bootstrap=True,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        )
        model.fit(train_values, target)
    values = _logit(model.predict_proba(evaluation_values)[:, 1])
    if mirror_values is not None:
        direct_probability = _sigmoid(values)
        mirror_probability = model.predict_proba(mirror_values)[:, 1]
        values = _logit(
            0.5 * (direct_probability + (1.0 - mirror_probability))
        )
    if not np.isfinite(values).all():
        raise PublicDraftScorePromotionError(
            "categorical world prediction is invalid"
        )
    if cache_path is not None:
        receipt_bytes = np.frombuffer(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
            dtype=np.uint8,
        )
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, logits=values, receipt=receipt_bytes)
        temporary.replace(cache_path)
    return values, receipt


def _phase_curve_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    shuffled: bool = False,
    cache_dir: Path | None = None,
    matrix_sha256: str | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predict post-draft checkpoints from pre-match fields only."""

    config = protocol["selection"]["phase_curve_config"]
    complete = train.dropna(subset=list(PHASE_TARGET_COLUMNS))
    if len(complete) < 500:
        raise PublicDraftScorePromotionError(
            "phase curve has fewer than 500 complete training rows"
        )
    features = complete[list(MODEL_COLUMNS)].astype(float)
    target = complete[list(PHASE_TARGET_COLUMNS)].astype(float).to_numpy()
    if not np.isfinite(features.to_numpy()).all() or not np.isfinite(target).all():
        raise PublicDraftScorePromotionError("phase curve training data is invalid")
    if shuffled:
        target = target[
            np.random.default_rng(RANDOM_SEED).permutation(len(target))
        ]
    receipt = {
        "id": str(config["id"]),
        "target_columns": list(PHASE_TARGET_COLUMNS),
        "target_scale": 1000.0,
        "feature_columns_sha256": canonical_sha256(list(MODEL_COLUMNS)),
        "training_rows": len(complete),
        "training_game_ids_sha256": canonical_sha256(
            complete["game_uid"].astype(str).tolist()
        ),
    }
    cache_path: Path | None = None
    if cache_dir is not None:
        if matrix_sha256 is None or not re.fullmatch(
            r"[0-9a-f]{64}", str(matrix_sha256)
        ):
            raise PublicDraftScorePromotionError(
                "phase curve cache requires the bound matrix SHA-256"
            )
        cache_key = canonical_sha256(
            {
                "schema_version": "scryglass:public-draft-score-phase-cache:v1",
                "matrix_sha256": matrix_sha256,
                "training_game_ids_sha256": receipt[
                    "training_game_ids_sha256"
                ],
                "evaluation_game_ids_sha256": canonical_sha256(
                    evaluation["game_uid"].astype(str).tolist()
                ),
                "target_sha256": hashlib.sha256(
                    np.asarray(target, dtype="<f8").tobytes()
                ).hexdigest(),
                "config": dict(config),
                "feature_columns_sha256": receipt["feature_columns_sha256"],
                "random_seed": RANDOM_SEED,
            }
        )
        cache_root = cache_dir / "phase-curve-predictions"
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{cache_key}.npz"
        if cache_path.exists():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    values = np.asarray(cached["values"], dtype=float)
                    cached_receipt = json.loads(
                        np.asarray(cached["receipt"], dtype=np.uint8)
                        .tobytes()
                        .decode("utf-8")
                    )
                if values.shape != (
                    len(evaluation),
                    len(PHASE_TARGET_COLUMNS),
                ) or not np.isfinite(values).all():
                    raise PublicDraftScorePromotionError(
                        "phase curve cache is invalid"
                    )
                if cached_receipt != receipt:
                    raise PublicDraftScorePromotionError(
                        "phase curve cache receipt changed"
                    )
                return values, receipt
            except PublicDraftScorePromotionError:
                raise
            except (
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                raise PublicDraftScorePromotionError(
                    "phase curve cache is invalid"
                ) from error
    model = RandomForestRegressor(
        n_estimators=int(config["n_estimators"]),
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=float(config["max_features"]),
        max_samples=float(config["max_samples"]),
        bootstrap=True,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(features, target)
    values = np.asarray(
        model.predict(evaluation[list(MODEL_COLUMNS)].astype(float)),
        dtype=float,
    ) / 1000.0
    if values.shape != (
        len(evaluation),
        len(PHASE_TARGET_COLUMNS),
    ) or not np.isfinite(values).all():
        raise PublicDraftScorePromotionError("phase curve prediction is invalid")
    if cache_path is not None:
        receipt_bytes = np.frombuffer(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            ),
            dtype=np.uint8,
        )
        temporary = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, values=values, receipt=receipt_bytes)
        temporary.replace(cache_path)
    return values, receipt


def _regional_world_predictions(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    protocol: Mapping[str, Any],
    *,
    shuffled: bool = False,
    feature_groups: Mapping[str, Sequence[str]] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    config = protocol["selection"]["quantum_forest_config"]
    outputs: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    for world in protocol["selection"].get("regional_worlds", []):
        columns = _quantum_world_columns(
            world["groups"], feature_groups=feature_groups
        )
        probability = np.full(len(evaluation), np.nan, dtype=float)
        support: dict[str, int] = {}
        for league, evaluation_group in evaluation.groupby("league", sort=True):
            train_group = train[train["league"].eq(league)]
            support[str(league)] = len(train_group)
            if len(train_group) < 100 or train_group["y"].nunique() != 2:
                predicted = _fit_probability(
                    train, evaluation_group, columns, config, shuffled=shuffled
                )
            else:
                predicted = _fit_probability(
                    train_group,
                    evaluation_group,
                    columns,
                    config,
                    shuffled=shuffled,
                )
            probability[evaluation.index.get_indexer(evaluation_group.index)] = predicted
        if not np.isfinite(probability).all():
            raise PublicDraftScorePromotionError("regional world prediction is incomplete")
        outputs.append(_logit(probability))
        receipts.append(
            {
                "world_id": str(world["id"]),
                "groups": list(world["groups"]),
                "training_rows_by_league": support,
                "columns_sha256": canonical_sha256(list(columns)),
            }
        )
    return np.column_stack(outputs), receipts


def _fit_quantum_meta(
    target: pd.Series, world_logits: np.ndarray, anchor_logit: np.ndarray
) -> BoundedLogisticModel:
    features = np.column_stack([anchor_logit, world_logits]).astype(float)
    if not np.isfinite(features).all():
        raise PublicDraftScorePromotionError("quantum meta inputs are not finite")
    features = np.clip(features, -8.0, 8.0)
    outcome = np.asarray(target, dtype=float)
    l2_strength = 10.0

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        coefficient = parameters
        score = np.sum(features * coefficient[None, :], axis=1)
        probability = _sigmoid(score)
        loss = float(
            np.logaddexp(0.0, score).sum()
            - np.sum(outcome * score)
            + 0.5 * l2_strength * np.sum(coefficient * coefficient)
        )
        residual = probability - outcome
        gradient = (
            np.sum(features * residual[:, None], axis=0)
            + l2_strength * coefficient
        )
        return loss, gradient

    result = optimize.minimize(
        objective,
        np.zeros(features.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-8.0, 8.0)] * features.shape[1],
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise PublicDraftScorePromotionError(
            f"quantum meta optimization failed: {result.message}"
        )
    if not np.isfinite(result.x).all() or np.max(np.abs(result.x)) > 8.0:
        raise PublicDraftScorePromotionError("quantum meta coefficients are invalid")
    return BoundedLogisticModel(
        coefficient=np.asarray(result.x, dtype=float),
        intercept=0.0,
    )


def _quantum_meta_probability(
    model: BoundedLogisticModel,
    world_logits: np.ndarray,
    anchor_logit: np.ndarray,
) -> np.ndarray:
    features = np.column_stack([anchor_logit, world_logits]).astype(float)
    if not np.isfinite(features).all():
        raise PublicDraftScorePromotionError("quantum meta inputs are not finite")
    features = np.clip(features, -8.0, 8.0)
    probability = np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    if not np.isfinite(probability).all():
        raise PublicDraftScorePromotionError("quantum meta probabilities are not finite")
    return _clip(probability)


def _select_with_crossfit_composition(
    inner_train: pd.DataFrame,
    inner_validation: pd.DataFrame,
    game_by_id: Mapping[str, Mapping[str, Any]],
    protocol: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates: list[tuple[dict[str, Any], dict[str, Any], dict[str, Any]]] = []
    for composition_config in protocol["selection"]["composition_configs"]:
        augmented_train, augmented_validation, receipt = _add_crossfit_composition(
            inner_train,
            inner_validation,
            game_by_id,
            alpha=float(composition_config["alpha"]),
            half_life_days=int(composition_config["half_life_days"]),
        )
        selected, search = _select_configuration(
            augmented_train, augmented_validation, protocol
        )
        selected = {
            **selected,
            "composition_config": dict(composition_config),
            "composition_receipt": receipt,
        }
        candidates.append((selected, search, receipt))
    selected, selected_search, _ = min(
        candidates,
        key=lambda item: (
            float(item[0]["log_loss"]),
            -float(item[0]["auc"]),
            float(item[0]["brier"]),
            str(item[0]["composition_config"]["id"]),
        ),
    )
    return selected, {
        **selected_search,
        "composition_candidates": [
            {
                "composition_config": item[0]["composition_config"],
                "composition_receipt": item[0]["composition_receipt"],
                "selected_model": item[0],
            }
            for item in candidates
        ],
    }


def _fold_evaluation(
    frame: pd.DataFrame,
    bounds: pd.DataFrame,
    fold: Mapping[str, Any],
    protocol: Mapping[str, Any],
    game_by_id: Mapping[str, Mapping[str, Any]],
    world_cache_dir: Path,
    matrix_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    start = pd.Timestamp(fold["start"])
    end = pd.Timestamp(fold["end"])
    inner_start = pd.Timestamp(fold["inner_start"])
    outer_train = _series_slice(frame, bounds, end=start)
    outer_test = _series_slice(frame, bounds, start=start, end=end)
    inner_train = _series_slice(outer_train, _series_bounds(outer_train), end=inner_start)
    inner_validation = _series_slice(
        outer_train, _series_bounds(outer_train), start=inner_start, end=start
    )
    sizes = {
        "inner_train": len(inner_train),
        "inner_validation": len(inner_validation),
        "outer_train": len(outer_train),
        "outer_test": len(outer_test),
    }
    if min(sizes.values()) < 50 or inner_train["y"].nunique() != 2 or outer_test["y"].nunique() != 2:
        raise PublicDraftScorePromotionError(
            f"fold {fold['id']} is too small or has one target class: {sizes}"
        )
    if protocol["selection"].get("architecture") == "quantum_masked_forest_v1":
        anchor = _select_anchor(inner_validation, protocol)
        inner_anchor_raw = _anchor_probability(
            inner_validation,
            team_weight=anchor["team_weight"],
            momentum_weight=anchor["momentum_weight"],
            source=anchor["source"],
        )
        inner_worlds, world_receipts = _quantum_world_predictions(
            inner_train,
            inner_validation,
            protocol,
            cache_dir=world_cache_dir,
            matrix_sha256=matrix_sha256,
        )
        inner_phase_receipt: dict[str, Any] | None = None
        if protocol["selection"].get("phase_curve_config"):
            inner_phase, inner_phase_receipt = _phase_curve_predictions(
                inner_train,
                inner_validation,
                protocol,
                cache_dir=world_cache_dir,
                matrix_sha256=matrix_sha256,
            )
            inner_worlds = np.column_stack([inner_worlds, inner_phase])
        inner_regional_receipts: list[dict[str, Any]] = []
        if protocol["selection"].get("regional_worlds"):
            inner_regional, inner_regional_receipts = _regional_world_predictions(
                inner_train, inner_validation, protocol
            )
            inner_worlds = np.column_stack([inner_worlds, inner_regional])
        inner_draft_receipts: list[dict[str, Any]] = []
        if protocol["selection"].get("draft_expert_configs"):
            inner_draft_logits, inner_draft_receipts = _draft_expert_logits(
                inner_train,
                inner_validation,
                game_by_id,
                protocol["selection"]["draft_expert_configs"],
            )
            inner_worlds = np.column_stack([inner_worlds, inner_draft_logits])
        inner_categorical_receipt: dict[str, Any] | None = None
        if protocol["selection"].get("categorical_world_config"):
            inner_categorical, inner_categorical_receipt = (
                _categorical_world_predictions(
                    inner_train,
                    inner_validation,
                    protocol,
                    cache_dir=world_cache_dir,
                    matrix_sha256=matrix_sha256,
                )
            )
            inner_worlds = np.column_stack([inner_worlds, inner_categorical])
        meta = _fit_quantum_meta(
            inner_validation["y"], inner_worlds, _logit(inner_anchor_raw)
        )
        outer_worlds, outer_world_receipts = _quantum_world_predictions(
            outer_train,
            outer_test,
            protocol,
            cache_dir=world_cache_dir,
            matrix_sha256=matrix_sha256,
        )
        if world_receipts != outer_world_receipts:
            raise PublicDraftScorePromotionError("quantum world reveal changed")
        outer_phase_receipt: dict[str, Any] | None = None
        if protocol["selection"].get("phase_curve_config"):
            outer_phase, outer_phase_receipt = _phase_curve_predictions(
                outer_train,
                outer_test,
                protocol,
                cache_dir=world_cache_dir,
                matrix_sha256=matrix_sha256,
            )
            outer_worlds = np.column_stack([outer_worlds, outer_phase])
        outer_regional_receipts: list[dict[str, Any]] = []
        if protocol["selection"].get("regional_worlds"):
            outer_regional, outer_regional_receipts = _regional_world_predictions(
                outer_train, outer_test, protocol
            )
            outer_worlds = np.column_stack([outer_worlds, outer_regional])
        outer_draft_receipts: list[dict[str, Any]] = []
        if protocol["selection"].get("draft_expert_configs"):
            outer_draft_logits, outer_draft_receipts = _draft_expert_logits(
                outer_train,
                outer_test,
                game_by_id,
                protocol["selection"]["draft_expert_configs"],
            )
            outer_worlds = np.column_stack([outer_worlds, outer_draft_logits])
        outer_categorical_receipt: dict[str, Any] | None = None
        if protocol["selection"].get("categorical_world_config"):
            outer_categorical, outer_categorical_receipt = (
                _categorical_world_predictions(
                    outer_train,
                    outer_test,
                    protocol,
                    cache_dir=world_cache_dir,
                    matrix_sha256=matrix_sha256,
                )
            )
            outer_worlds = np.column_stack([outer_worlds, outer_categorical])
        outer_anchor_raw = _anchor_probability(
            outer_test,
            team_weight=anchor["team_weight"],
            momentum_weight=anchor["momentum_weight"],
            source=anchor["source"],
        )
        probability = _quantum_meta_probability(
            meta, outer_worlds, _logit(outer_anchor_raw)
        )
        strength_calibration = Calibration(**anchor["calibration"])
        baseline_probability = strength_calibration.apply(outer_anchor_raw)
        shuffled_inner_worlds, _ = _quantum_world_predictions(
            inner_train,
            inner_validation,
            protocol,
            shuffled=True,
            cache_dir=world_cache_dir,
            matrix_sha256=matrix_sha256,
        )
        if protocol["selection"].get("phase_curve_config"):
            shuffled_inner_phase, _ = _phase_curve_predictions(
                inner_train,
                inner_validation,
                protocol,
                shuffled=True,
                cache_dir=world_cache_dir,
                matrix_sha256=matrix_sha256,
            )
            shuffled_inner_worlds = np.column_stack(
                [shuffled_inner_worlds, shuffled_inner_phase]
            )
        if protocol["selection"].get("regional_worlds"):
            shuffled_inner_regional, _ = _regional_world_predictions(
                inner_train, inner_validation, protocol, shuffled=True
            )
            shuffled_inner_worlds = np.column_stack(
                [shuffled_inner_worlds, shuffled_inner_regional]
            )
        if protocol["selection"].get("draft_expert_configs"):
            shuffled_inner_draft, _ = _draft_expert_logits(
                inner_train,
                inner_validation,
                game_by_id,
                protocol["selection"]["draft_expert_configs"],
                shuffled=True,
            )
            shuffled_inner_worlds = np.column_stack(
                [shuffled_inner_worlds, shuffled_inner_draft]
            )
        if protocol["selection"].get("categorical_world_config"):
            shuffled_inner_categorical, _ = _categorical_world_predictions(
                inner_train,
                inner_validation,
                protocol,
                shuffled=True,
                cache_dir=world_cache_dir,
                matrix_sha256=matrix_sha256,
            )
            shuffled_inner_worlds = np.column_stack(
                [shuffled_inner_worlds, shuffled_inner_categorical]
            )
        shuffled_meta = _fit_quantum_meta(
            inner_validation["y"],
            shuffled_inner_worlds,
            np.zeros(len(inner_validation), dtype=float),
        )
        shuffled_outer_worlds, _ = _quantum_world_predictions(
            outer_train,
            outer_test,
            protocol,
            shuffled=True,
            cache_dir=world_cache_dir,
            matrix_sha256=matrix_sha256,
        )
        if protocol["selection"].get("phase_curve_config"):
            shuffled_outer_phase, _ = _phase_curve_predictions(
                outer_train,
                outer_test,
                protocol,
                shuffled=True,
                cache_dir=world_cache_dir,
                matrix_sha256=matrix_sha256,
            )
            shuffled_outer_worlds = np.column_stack(
                [shuffled_outer_worlds, shuffled_outer_phase]
            )
        if protocol["selection"].get("regional_worlds"):
            shuffled_outer_regional, _ = _regional_world_predictions(
                outer_train, outer_test, protocol, shuffled=True
            )
            shuffled_outer_worlds = np.column_stack(
                [shuffled_outer_worlds, shuffled_outer_regional]
            )
        if protocol["selection"].get("draft_expert_configs"):
            shuffled_outer_draft, _ = _draft_expert_logits(
                outer_train,
                outer_test,
                game_by_id,
                protocol["selection"]["draft_expert_configs"],
                shuffled=True,
            )
            shuffled_outer_worlds = np.column_stack(
                [shuffled_outer_worlds, shuffled_outer_draft]
            )
        if protocol["selection"].get("categorical_world_config"):
            shuffled_outer_categorical, _ = _categorical_world_predictions(
                outer_train,
                outer_test,
                protocol,
                shuffled=True,
                cache_dir=world_cache_dir,
                matrix_sha256=matrix_sha256,
            )
            shuffled_outer_worlds = np.column_stack(
                [shuffled_outer_worlds, shuffled_outer_categorical]
            )
        shuffled_probability = _quantum_meta_probability(
            shuffled_meta,
            shuffled_outer_worlds,
            np.zeros(len(outer_test), dtype=float),
        )
        output = outer_test[
            ["game_uid", "series_id", "date", "league", "source_patch", "y"]
        ].copy()
        output["probability"] = probability
        output["baseline_probability"] = baseline_probability
        output["shuffle_probability"] = shuffled_probability
        group_names = tuple(DRAFT_GROUPS)
        reveal = {
            group: {
                "active": [
                    row["anonymous_id"]
                    for row in world_receipts
                    if group in row["groups"]
                ],
                "inactive": [
                    row["anonymous_id"]
                    for row in world_receipts
                    if group not in row["groups"]
                ],
            }
            for group in group_names
        }
        if any(not row["active"] or not row["inactive"] for row in reveal.values()):
            raise PublicDraftScorePromotionError(
                "each atom group must be active and inactive across hidden worlds"
            )
        selected = {
            "architecture": "quantum_masked_forest_v1",
            "anchor": anchor,
            "meta_intercept": float(meta.intercept_[0]),
            "meta_coefficients": [float(value) for value in meta.coef_[0]],
            "world_commitment_sha256": canonical_sha256(
                [
                    {
                        "anonymous_id": row["anonymous_id"],
                        "columns_sha256": row["columns_sha256"],
                    }
                    for row in world_receipts
                ]
            ),
            "prediction_commitment_sha256": hashlib.sha256(
                np.asarray(outer_worlds, dtype="<f8").tobytes()
            ).hexdigest(),
            "post_prediction_reveal": reveal,
            "worlds": world_receipts,
            "inner_regional_worlds": inner_regional_receipts,
            "outer_regional_worlds": outer_regional_receipts,
            "inner_draft_experts": inner_draft_receipts,
            "outer_draft_experts": outer_draft_receipts,
            "inner_phase_curve": inner_phase_receipt,
            "outer_phase_curve": outer_phase_receipt,
            "inner_categorical_world": inner_categorical_receipt,
            "outer_categorical_world": outer_categorical_receipt,
        }
        report = {
            "id": fold["id"],
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sizes": sizes,
            "search": {"architecture": "quantum_masked_forest_v1"},
            "selected": selected,
            "outer_composition_receipt": None,
            "strength_calibration": strength_calibration.as_dict(),
            "candidate": metric_report(output["y"], output["probability"]),
            "baseline": metric_report(output["y"], output["baseline_probability"]),
            "shuffle": metric_report(output["y"], output["shuffle_probability"]),
        }
        return output, report
    if protocol["selection"].get("composition_configs"):
        selected, search = _select_with_crossfit_composition(
            inner_train, inner_validation, game_by_id, protocol
        )
        composition_config = selected["composition_config"]
        outer_train, outer_test, outer_composition_receipt = (
            _add_crossfit_composition(
                outer_train,
                outer_test,
                game_by_id,
                alpha=float(composition_config["alpha"]),
                half_life_days=int(composition_config["half_life_days"]),
            )
        )
    else:
        selected, search = _select_configuration(
            inner_train, inner_validation, protocol
        )
        outer_composition_receipt = None
    config = selected["config"]
    if search.get("architecture") == "rating_anchor_rf_v1":
        columns = tuple(selected["columns"])
        full = _fit_probability(outer_train, outer_test, columns, config)
        anchor = _anchor_probability(
            outer_test,
            team_weight=selected["anchor_team_weight"],
            momentum_weight=selected["anchor_momentum_weight"],
            source=str(
                protocol["selection"].get("anchor_source", "shrunk_probability")
            ),
        )
        raw = _blend(full, anchor, selected["blend_weight"])
        calibration = Calibration(**selected["calibration"])
        probability = calibration.apply(raw)
        baseline_selected = search["selected_baseline"]
        baseline_raw = _anchor_probability(
            outer_test,
            team_weight=baseline_selected["team_weight"],
            momentum_weight=baseline_selected["momentum_weight"],
            source=str(
                protocol["selection"].get("anchor_source", "shrunk_probability")
            ),
        )
        strength_calibration = Calibration(**baseline_selected["calibration"])
        baseline_probability = strength_calibration.apply(baseline_raw)
        shuffle_columns = columns
    else:
        full = _fit_probability(outer_train, outer_test, MODEL_COLUMNS, config)
        strength = _fit_probability(
            outer_train, outer_test, STRENGTH_COLUMNS, config
        )
        raw = _blend(full, strength, selected["blend_weight"])
        calibration = Calibration(**selected["calibration"])
        probability = calibration.apply(raw)
        inner_strength = _fit_probability(
            inner_train, inner_validation, STRENGTH_COLUMNS, config
        )
        strength_calibration = _fit_calibration(inner_validation["y"], inner_strength)
        baseline_probability = strength_calibration.apply(strength)
        shuffle_columns = MODEL_COLUMNS
    shuffled_probability = _fit_probability(
        outer_train,
        outer_test,
        shuffle_columns,
        config,
        shuffled=True,
    )
    output = outer_test[
        ["game_uid", "series_id", "date", "league", "source_patch", "y"]
    ].copy()
    output["probability"] = probability
    output["baseline_probability"] = baseline_probability
    output["shuffle_probability"] = shuffled_probability
    report = {
        "id": fold["id"],
        "start": start.isoformat(),
        "end": end.isoformat(),
        "sizes": sizes,
        "search": search,
        "selected": selected,
        "outer_composition_receipt": outer_composition_receipt,
        "strength_calibration": strength_calibration.as_dict(),
        "candidate": metric_report(output["y"], output["probability"]),
        "baseline": metric_report(output["y"], output["baseline_probability"]),
        "shuffle": metric_report(output["y"], output["shuffle_probability"]),
    }
    return output, report


def _subgroup_metrics(
    predictions: pd.DataFrame, column: str, minimum_rows: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value, group in predictions.groupby(column, sort=True):
        if len(group) < minimum_rows or group["y"].nunique() != 2:
            continue
        result[str(value)] = metric_report(group["y"], group["probability"])
    return result


def evaluate_public_draft_score(
    *,
    matrix_path: Path,
    protocol_path: Path,
    expected_matrix_sha256: str,
    matrix_manifest_path: Path | None = None,
    expected_matrix_manifest_sha256: str | None = None,
    players_path: Path | None = None,
    expected_players_sha256: str | None = None,
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    started = time.perf_counter()
    if sha256_path(matrix_path) != expected_matrix_sha256:
        raise PublicDraftScorePromotionError("promotion matrix SHA-256 changed")
    protocol, protocol_receipt = _load_protocol(protocol_path)
    if protocol.get("status") != "frozen_before_first_v1_model_evaluation":
        raise PublicDraftScorePromotionError("promotion protocol is not frozen")
    _validate_protocol_matrix_binding(protocol, expected_matrix_sha256)
    matrix_manifest_receipt = _validate_matrix_manifest(
        protocol=protocol,
        manifest_path=matrix_manifest_path,
        expected_manifest_sha256=expected_matrix_manifest_sha256,
        expected_matrix_sha256=expected_matrix_sha256,
    )
    frame = _validate_matrix(pd.read_parquet(matrix_path))
    if matrix_manifest_receipt is not None:
        manifest = json.loads(matrix_manifest_path.read_text(encoding="utf-8"))
        if manifest.get("rows") != len(frame):
            raise PublicDraftScorePromotionError("matrix manifest row count changed")
    game_by_id: dict[str, Mapping[str, Any]] = {}
    players_receipt: dict[str, Any] | None = None
    if protocol["selection"].get("composition_configs") or protocol["selection"].get(
        "draft_expert_configs"
    ):
        if players_path is None or expected_players_sha256 is None:
            raise PublicDraftScorePromotionError(
                "cross-fit composition requires a hash-bound player source"
            )
        actual_players_sha256 = sha256_path(players_path)
        if actual_players_sha256 != expected_players_sha256:
            raise PublicDraftScorePromotionError("player source SHA-256 changed")
        players = pd.read_parquet(players_path)
        if "game_uid" not in players and "gameid" in players:
            players = players.assign(game_uid=players["gameid"].astype(str))
        games = build_draft_games(players)
        game_by_id = {str(game["game_uid"]): game for game in games}
        players_receipt = {
            "path": str(players_path),
            "sha256": actual_players_sha256,
            "complete_drafts": len(game_by_id),
        }
    bounds = _series_bounds(frame)
    cache_dir = cache_dir or matrix_path.parent / "public-draft-score-fold-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    prediction_parts: list[pd.DataFrame] = []
    fold_reports: list[dict[str, Any]] = []
    for fold in protocol["outer_folds"]:
        fold_key = canonical_sha256(
            {
                "matrix_sha256": expected_matrix_sha256,
                "protocol_sha256": canonical_sha256(protocol),
                "fold": fold,
                "code_sha256": sha256_path(Path(__file__)),
                "draft_recommendation_code_sha256": sha256_path(
                    Path(draft_recommendation_module.__file__)
                ),
                "players_sha256": expected_players_sha256,
            }
        )
        prediction_cache = cache_dir / f"{fold_key}.parquet"
        report_cache = cache_dir / f"{fold_key}.json"
        if prediction_cache.exists() and report_cache.exists():
            predictions = pd.read_parquet(prediction_cache)
            report = json.loads(report_cache.read_text(encoding="utf-8"))
        else:
            predictions, report = _fold_evaluation(
                frame,
                bounds,
                fold,
                protocol,
                game_by_id,
                cache_dir,
                expected_matrix_sha256,
            )
            predictions.to_parquet(prediction_cache, index=False, compression="zstd")
            report_cache.write_text(
                json.dumps(report, indent=2, sort_keys=True, default=_json_default),
                encoding="utf-8",
            )
        prediction_parts.append(predictions)
        fold_reports.append(report)
        print(
            json.dumps(
                {
                    "fold": report["id"],
                    "outer_rows": report["candidate"]["n"],
                    "candidate_auc": report["candidate"]["auc"],
                    "baseline_auc": report["baseline"]["auc"],
                    "selected": report["selected"].get("group_set_id"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    pooled = pd.concat(prediction_parts, ignore_index=True).sort_values(
        ["date", "game_uid"], kind="stable"
    )
    if pooled["game_uid"].duplicated().any():
        raise PublicDraftScorePromotionError("outer folds overlap")

    candidate = metric_report(pooled["y"], pooled["probability"])
    baseline = metric_report(pooled["y"], pooled["baseline_probability"])
    shuffle = metric_report(pooled["y"], pooled["shuffle_probability"])
    difference_bootstrap = _cluster_bootstrap_differences(
        pooled,
        pooled["probability"].to_numpy(),
        pooled["baseline_probability"].to_numpy(),
        repetitions=2000,
    )
    auc_bootstrap = _cluster_bootstrap_auc(
        pooled, pooled["probability"].to_numpy(), repetitions=2000
    )
    regions = _subgroup_metrics(pooled, "league", 100)
    patches = _subgroup_metrics(pooled, "source_patch", 50)
    gates = protocol["promotion_gates"]
    gate_results = {
        "pooled_outer_auc": {
            "passed": float(candidate["auc"]) >= float(gates["pooled_outer_auc_min"]),
            "value": candidate["auc"],
            "minimum": gates["pooled_outer_auc_min"],
        },
        "brier_vs_baseline": {
            "passed": float(candidate["brier"] - baseline["brier"])
            <= float(gates["pooled_outer_brier_max_delta_vs_baseline"]),
            "delta": float(candidate["brier"] - baseline["brier"]),
            "maximum": gates["pooled_outer_brier_max_delta_vs_baseline"],
        },
        "log_loss_vs_baseline": {
            "passed": float(candidate["log_loss"] - baseline["log_loss"])
            <= float(gates["pooled_outer_log_loss_max_delta_vs_baseline"]),
            "delta": float(candidate["log_loss"] - baseline["log_loss"]),
            "maximum": gates["pooled_outer_log_loss_max_delta_vs_baseline"],
        },
        "calibration": {
            "passed": float(candidate["ece_equal_frequency_10"])
            <= float(gates["pooled_outer_ece_max"]),
            "value": candidate["ece_equal_frequency_10"],
            "maximum": gates["pooled_outer_ece_max"],
        },
        "shuffle": {
            "passed": float(shuffle["auc"]) <= float(gates["shuffle_auc_max"]),
            "value": shuffle["auc"],
            "maximum": gates["shuffle_auc_max"],
        },
        "bootstrap_auc_difference": {
            "passed": float(difference_bootstrap["auc"]["median"])
            >= float(gates["series_bootstrap_auc_difference_median_min"]),
            **difference_bootstrap["auc"],
        },
        "sample_size": {
            "passed": len(pooled) >= int(gates["minimum_outer_rows"]),
            "value": len(pooled),
            "minimum": gates["minimum_outer_rows"],
        },
        "regional_coverage": {
            "passed": len(regions) >= int(gates["minimum_regions_with_100_rows"]),
            "value": len(regions),
            "minimum": gates["minimum_regions_with_100_rows"],
        },
        "final_public_holdout": {
            "passed": not bool(
                protocol.get("authority_policy", {}).get(
                    "final_public_holdout_required", False
                )
            ),
            "required": bool(
                protocol.get("authority_policy", {}).get(
                    "final_public_holdout_required", False
                )
            ),
        },
    }
    passed = all(bool(row["passed"]) for row in gate_results.values())
    report = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            **protocol_receipt,
            "frozen_utc": protocol["frozen_utc"],
            "resolved_sha256": canonical_sha256(protocol),
        },
        "code": {
            "evaluation_sha256": sha256_path(Path(__file__)),
            "draft_recommendation_sha256": sha256_path(
                Path(draft_recommendation_module.__file__)
            ),
        },
        "matrix": {
            "path": str(matrix_path),
            "sha256": expected_matrix_sha256,
            "rows": len(frame),
            "feature_columns": len(MODEL_COLUMNS),
            "manifest": matrix_manifest_receipt,
        },
        "players": players_receipt,
        "estimands": protocol["products"],
        "outer_folds": fold_reports,
        "pooled": {
            "candidate": candidate,
            "baseline": baseline,
            "shuffle": shuffle,
            "auc_series_bootstrap": auc_bootstrap,
            "difference_series_bootstrap": difference_bootstrap,
            "regions": regions,
            "patches": patches,
            "prediction_sha256": hashlib.sha256(
                np.asarray(pooled["probability"], dtype="<f8").tobytes()
            ).hexdigest(),
        },
        "promotion_gates": gate_results,
        "promotion_passed": passed,
        "authority": {
            "status": "promoted" if passed else "research_failed_gate",
            "public_probability": passed,
            "public_recommendation": passed,
            "public_controlled_draft_score": passed,
            "betting": False,
            "odds": False,
            "expected_value": False,
            "stake": False,
            "wager": False,
        },
        "wall_seconds": time.perf_counter() - started,
    }
    report["receipt_sha256"] = canonical_sha256(report)
    return report, pooled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--matrix-sha256", required=True)
    parser.add_argument("--matrix-manifest", type=Path)
    parser.add_argument("--matrix-manifest-sha256")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--players", type=Path)
    parser.add_argument("--players-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    report, predictions = evaluate_public_draft_score(
        matrix_path=args.matrix,
        protocol_path=args.protocol,
        expected_matrix_sha256=args.matrix_sha256,
        matrix_manifest_path=args.matrix_manifest,
        expected_matrix_manifest_sha256=args.matrix_manifest_sha256,
        players_path=args.players,
        expected_players_sha256=args.players_sha256,
        cache_dir=args.cache_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    predictions.to_parquet(args.predictions, index=False, compression="zstd")
    print(
        json.dumps(
            {
                "promotion_passed": report["promotion_passed"],
                "candidate": report["pooled"]["candidate"],
                "baseline": report["pooled"]["baseline"],
                "shuffle": report["pooled"]["shuffle"],
                "gates": report["promotion_gates"],
                "receipt_sha256": report["receipt_sha256"],
                "wall_seconds": report["wall_seconds"],
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
