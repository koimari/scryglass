"""Fit public-reproducible voters for the selective Draft model."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from lol_kills.research.atomized_rf_composite import GROUP_COLUMNS, MODEL_COLUMNS
from lol_kills.research.public_draft_score_promotion import (
    CATEGORICAL_CONTEXT_COLUMNS,
    STRENGTH_COLUMNS,
    _anchor_probability,
    build_draft_games,
    _categorical_world_predictions,
    _draft_expert_logits,
    _fit_quantum_meta,
    _logit,
    _load_protocol,
    _mirror_categorical_frame,
    _phase_curve_predictions,
    _quantum_world_predictions,
    _quantum_meta_probability,
    _regional_world_predictions,
    _select_anchor,
    _series_bounds,
    _series_slice,
    sha256_path,
)
from lol_kills.research.selective_draft_probability import canonical_sha256


ALL_ATOM_IDENTITY_SCHEMA = "scryglass:all-atom-identity-logit:v1"
STRENGTH_IDENTITY_SCHEMA = "scryglass:strength-identity-logit:v1"
ROSTER_FOREST_SCHEMA = "scryglass:roster-random-forest:v1"
QUANTUM_VOTER_SCHEMA = "scryglass:quantum-masked-forest-voter:v1"
FROZEN_SELECTIVE_VOTERS_SCHEMA = "scryglass:frozen-selective-voters:v1"
V24_QUANTUM_PROTOCOL_FILE_SHA256 = (
    "da548e126829ab57a1a90f11387128cb66f839f6fde01ba0062b2feb53881004"
)
V24_QUANTUM_PROTOCOL_RESOLVED_SHA256 = (
    "5ecba3d602394816ef3d3c5a9c9ebac6bf13671092c5872e3e683e1c424faa2b"
)
V24_QUANTUM_FEATURE_GROUPS_SHA256 = (
    "731f88119bc698d428005c3ecf8f455bbba25d5ef9e231880c55ab998de132c8"
)
ROSTER_GROUPS = (
    "team_rating",
    "player_rating",
    "rating_uncertainty",
    "team_momentum",
    "match_context",
    "team_macro_form",
    "player_exact_performance",
    "player_role_performance",
    "global_champion_performance",
    "exact_ally_enemy_pairs",
    "checkpoint_forecasts",
)
ROSTER_FOREST_CONFIG = {
    "id": "composite-roster-rf",
    "n_estimators": 800,
    "max_depth": 10,
    "min_samples_leaf": 8,
    "max_features": 0.2,
    "max_samples": 0.85,
    "class_weight": None,
    "side_symmetry_augmentation": True,
}


class SelectiveDraftConstituentError(ValueError):
    """Raised when a constituent fit is not reproducible."""


def _normalize_player_game_uids(players: pd.DataFrame) -> pd.DataFrame:
    """Use the OE game identifier when a merged row has a null game UID."""

    output = players.copy()
    if "gameid" in output:
        if "game_uid" in output:
            output["game_uid"] = output["game_uid"].where(
                output["game_uid"].notna(), output["gameid"]
            )
        else:
            output["game_uid"] = output["gameid"]
    if "game_uid" not in output or output["game_uid"].isna().any():
        raise SelectiveDraftConstituentError(
            "player source has unresolved game identities"
        )
    output["game_uid"] = output["game_uid"].astype(str)
    return output


def load_v24_quantum_contract(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Load the exact v24 protocol and its frozen feature inventory."""

    if sha256_path(protocol_path) != V24_QUANTUM_PROTOCOL_FILE_SHA256:
        raise SelectiveDraftConstituentError("v24 quantum protocol changed")
    protocol, _ = _load_protocol(protocol_path)
    if canonical_sha256(protocol) != V24_QUANTUM_PROTOCOL_RESOLVED_SHA256:
        raise SelectiveDraftConstituentError("v24 quantum protocol chain changed")
    feature_groups = {
        name: list(columns)
        for name, columns in GROUP_COLUMNS.items()
        if name != "rating_uncertainty"
    }
    if canonical_sha256(feature_groups) != V24_QUANTUM_FEATURE_GROUPS_SHA256:
        raise SelectiveDraftConstituentError("v24 quantum feature inventory changed")
    return protocol, feature_groups


def fit_frozen_selective_voters(
    training: pd.DataFrame,
    quantum_training: pd.DataFrame,
    evaluation_features: pd.DataFrame,
    *,
    v24_protocol_path: Path,
    game_by_id: Mapping[str, Mapping[str, Any]],
    inner_start: Any,
    evaluation_start: Any,
    cache_dir: Path | None,
    source_matrix_sha256: str,
    training_matrix_sha256: str,
    quantum_training_matrix_sha256: str,
    evaluation_features_sha256: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit all four frozen voters before any holdout outcome is revealed."""

    forbidden_exact = {"y", "result", "outcome", "winner", "blue_win"}
    forbidden = sorted(
        column
        for column in evaluation_features.columns
        if column in forbidden_exact
        or column.startswith(("target_", "observed_", "final_"))
    )
    if forbidden:
        raise SelectiveDraftConstituentError(
            f"evaluation contains forbidden fields: {forbidden}"
        )
    if "game_uid" not in evaluation_features:
        raise SelectiveDraftConstituentError("evaluation game identities are missing")
    if evaluation_features.empty:
        raise SelectiveDraftConstituentError("evaluation feature set is empty")
    hashes = {
        "source_matrix": source_matrix_sha256,
        "training_matrix": training_matrix_sha256,
        "quantum_training_matrix": quantum_training_matrix_sha256,
        "evaluation_features": evaluation_features_sha256,
    }
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in hashes.values()
    ):
        raise SelectiveDraftConstituentError("voter input SHA-256 is invalid")
    future_ids = set(evaluation_features["game_uid"].astype(str))
    blind_games: dict[str, Mapping[str, Any]] = {}
    for game_id, game in game_by_id.items():
        if str(game_id) in future_ids:
            blind_games[str(game_id)] = {
                key: value
                for key, value in game.items()
                if key not in forbidden_exact
            }
        else:
            blind_games[str(game_id)] = game

    protocol, quantum_feature_groups = load_v24_quantum_contract(
        v24_protocol_path
    )
    quantum, quantum_receipt = fit_quantum_masked_forest_predictions(
        quantum_training,
        evaluation_features,
        protocol=protocol,
        game_by_id=blind_games,
        inner_start=inner_start,
        evaluation_start=evaluation_start,
        cache_dir=cache_dir,
        matrix_sha256=source_matrix_sha256,
        feature_groups=quantum_feature_groups,
    )
    roster, roster_receipt = fit_roster_random_forest_predictions(
        training,
        evaluation_features,
        cache_dir=cache_dir,
        matrix_sha256=source_matrix_sha256,
    )
    identity, identity_receipt = fit_strength_identity_predictions(
        training, evaluation_features
    )
    atomized_identity, atomized_identity_receipt = (
        fit_all_atom_identity_predictions(training, evaluation_features)
    )
    outputs = {
        "quantum": quantum,
        "roster": roster,
        "identity": identity,
        "development_composite": atomized_identity,
    }
    expected_ids = evaluation_features["game_uid"].astype(str).tolist()
    combined = pd.DataFrame({"game_uid": expected_ids})
    for name, frame in outputs.items():
        if frame["game_uid"].astype(str).tolist() != expected_ids:
            raise SelectiveDraftConstituentError(
                f"{name} prediction identities changed"
            )
        combined[name] = frame["p"].to_numpy(dtype=float)
    probabilities = combined[list(outputs)].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or np.any(
        (probabilities <= 0) | (probabilities >= 1)
    ):
        raise SelectiveDraftConstituentError("voter probabilities are invalid")
    receipt = {
        "schema_version": FROZEN_SELECTIVE_VOTERS_SCHEMA,
        "outcome_blind": True,
        "legacy_slot_semantics": {
            "development_composite": "in_repo_all_atom_fields_identity_logit"
        },
        "input_sha256": hashes,
        "evaluation_rows": len(combined),
        "evaluation_game_ids_sha256": canonical_sha256(expected_ids),
        "voters": {
            "quantum": quantum_receipt,
            "roster": roster_receipt,
            "identity": identity_receipt,
            "development_composite": atomized_identity_receipt,
        },
        "prediction_sha256": canonical_sha256(
            combined.to_dict(orient="records")
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return combined, receipt


def run_frozen_selective_voters(
    *,
    training_matrix_path: Path,
    expected_training_matrix_sha256: str,
    quantum_training_matrix_path: Path,
    expected_quantum_training_matrix_sha256: str,
    evaluation_features_path: Path,
    expected_evaluation_features_sha256: str,
    players_path: Path,
    expected_players_sha256: str,
    v24_protocol_path: Path,
    inner_start: Any,
    evaluation_start: Any,
    cache_dir: Path | None,
    predictions_output: Path,
    receipt_output: Path,
) -> dict[str, Any]:
    """Run the four-voter blind fit from hash-bound file inputs."""

    paths_and_hashes = (
        (training_matrix_path, expected_training_matrix_sha256, "training matrix"),
        (
            quantum_training_matrix_path,
            expected_quantum_training_matrix_sha256,
            "quantum training matrix",
        ),
        (
            evaluation_features_path,
            expected_evaluation_features_sha256,
            "evaluation features",
        ),
        (players_path, expected_players_sha256, "player source"),
    )
    for path, expected, label in paths_and_hashes:
        if not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
            raise SelectiveDraftConstituentError(f"{label} SHA-256 is invalid")
        if not path.is_file() or sha256_path(path) != expected:
            raise SelectiveDraftConstituentError(f"{label} changed")
    if predictions_output.exists() or receipt_output.exists():
        raise SelectiveDraftConstituentError("voter output already exists")

    training = pd.read_parquet(training_matrix_path)
    quantum_training = pd.read_parquet(quantum_training_matrix_path)
    evaluation_features = pd.read_parquet(evaluation_features_path)
    players = pd.read_parquet(players_path)
    players = _normalize_player_game_uids(players)
    game_by_id = {
        str(game["game_uid"]): game for game in build_draft_games(players)
    }
    source_matrix_sha256 = canonical_sha256(
        {
            "training_matrix_sha256": expected_training_matrix_sha256,
            "quantum_training_matrix_sha256": (
                expected_quantum_training_matrix_sha256
            ),
            "evaluation_features_sha256": expected_evaluation_features_sha256,
        }
    )
    predictions, receipt = fit_frozen_selective_voters(
        training,
        quantum_training,
        evaluation_features,
        v24_protocol_path=v24_protocol_path,
        game_by_id=game_by_id,
        inner_start=inner_start,
        evaluation_start=evaluation_start,
        cache_dir=cache_dir,
        source_matrix_sha256=source_matrix_sha256,
        training_matrix_sha256=expected_training_matrix_sha256,
        quantum_training_matrix_sha256=(
            expected_quantum_training_matrix_sha256
        ),
        evaluation_features_sha256=expected_evaluation_features_sha256,
    )
    receipt.pop("receipt_sha256")
    receipt["input_sha256"]["players"] = expected_players_sha256
    receipt["prediction_rows"] = len(predictions)
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    receipt_output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_output, index=False, compression="zstd")
    receipt["prediction_file_sha256"] = sha256_path(predictions_output)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-matrix", type=Path, required=True)
    parser.add_argument("--training-matrix-sha256", required=True)
    parser.add_argument("--quantum-training-matrix", type=Path, required=True)
    parser.add_argument("--quantum-training-matrix-sha256", required=True)
    parser.add_argument("--evaluation-features", type=Path, required=True)
    parser.add_argument("--evaluation-features-sha256", required=True)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--players-sha256", required=True)
    parser.add_argument("--v24-protocol", type=Path, required=True)
    parser.add_argument("--inner-start", required=True)
    parser.add_argument("--evaluation-start", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    receipt = run_frozen_selective_voters(
        training_matrix_path=args.training_matrix,
        expected_training_matrix_sha256=args.training_matrix_sha256,
        quantum_training_matrix_path=args.quantum_training_matrix,
        expected_quantum_training_matrix_sha256=(
            args.quantum_training_matrix_sha256
        ),
        evaluation_features_path=args.evaluation_features,
        expected_evaluation_features_sha256=args.evaluation_features_sha256,
        players_path=args.players,
        expected_players_sha256=args.players_sha256,
        v24_protocol_path=args.v24_protocol,
        inner_start=args.inner_start,
        evaluation_start=args.evaluation_start,
        cache_dir=args.cache_dir,
        predictions_output=args.predictions_output,
        receipt_output=args.receipt_output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


def _quantum_stack(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    game_by_id: Mapping[str, Mapping[str, Any]],
    cache_dir: Path | None,
    matrix_sha256: str,
    feature_groups: Mapping[str, Sequence[str]],
) -> tuple[np.ndarray, dict[str, Any]]:
    worlds, world_receipts = _quantum_world_predictions(
        training,
        evaluation,
        protocol,
        cache_dir=cache_dir,
        matrix_sha256=matrix_sha256,
        feature_groups=feature_groups,
    )
    phase_receipt: dict[str, Any] | None = None
    if protocol["selection"].get("phase_curve_config"):
        phase, phase_receipt = _phase_curve_predictions(
            training,
            evaluation,
            protocol,
            cache_dir=cache_dir,
            matrix_sha256=matrix_sha256,
        )
        worlds = np.column_stack([worlds, phase])
    regional_receipts: list[dict[str, Any]] = []
    if protocol["selection"].get("regional_worlds"):
        regional, regional_receipts = _regional_world_predictions(
            training,
            evaluation,
            protocol,
            feature_groups=feature_groups,
        )
        worlds = np.column_stack([worlds, regional])
    draft_receipts: list[dict[str, Any]] = []
    if protocol["selection"].get("draft_expert_configs"):
        draft, draft_receipts = _draft_expert_logits(
            training,
            evaluation,
            game_by_id,
            protocol["selection"]["draft_expert_configs"],
        )
        worlds = np.column_stack([worlds, draft])
    categorical_receipt: dict[str, Any] | None = None
    if protocol["selection"].get("categorical_world_config"):
        categorical, categorical_receipt = _categorical_world_predictions(
            training,
            evaluation,
            protocol,
            cache_dir=cache_dir,
            matrix_sha256=matrix_sha256,
        )
        worlds = np.column_stack([worlds, categorical])
    if worlds.shape[0] != len(evaluation) or not np.isfinite(worlds).all():
        raise SelectiveDraftConstituentError("quantum world output is invalid")
    return worlds, {
        "worlds": world_receipts,
        "phase_curve": phase_receipt,
        "regional_worlds": regional_receipts,
        "draft_experts": draft_receipts,
        "categorical_world": categorical_receipt,
    }


def fit_quantum_masked_forest_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    protocol: Mapping[str, Any],
    game_by_id: Mapping[str, Mapping[str, Any]],
    inner_start: Any,
    evaluation_start: Any,
    cache_dir: Path | None = None,
    matrix_sha256: str,
    feature_groups: Mapping[str, Sequence[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the frozen quantum voter without reading evaluation outcomes."""

    if protocol.get("selection", {}).get("architecture") != (
        "quantum_masked_forest_v1"
    ):
        raise SelectiveDraftConstituentError("quantum architecture is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", str(matrix_sha256)):
        raise SelectiveDraftConstituentError("matrix SHA-256 is invalid")
    if not feature_groups or any(
        not isinstance(name, str) or not tuple(columns)
        for name, columns in feature_groups.items()
    ):
        raise SelectiveDraftConstituentError("feature-group inventory is invalid")
    required_training = {"game_uid", "series_id", "date", "y"}
    required_evaluation = {"game_uid", "series_id", "date"}
    if not required_training.issubset(training.columns):
        raise SelectiveDraftConstituentError("quantum training columns are incomplete")
    if not required_evaluation.issubset(evaluation.columns):
        raise SelectiveDraftConstituentError("quantum evaluation columns are incomplete")
    train = training.copy()
    future = evaluation.copy()
    train["date"] = pd.to_datetime(train["date"], utc=True, errors="raise")
    future["date"] = pd.to_datetime(future["date"], utc=True, errors="raise")
    inner_cut = pd.Timestamp(inner_start)
    holdout_cut = pd.Timestamp(evaluation_start)
    inner_cut = (
        inner_cut.tz_localize("UTC")
        if inner_cut.tzinfo is None
        else inner_cut.tz_convert("UTC")
    )
    holdout_cut = (
        holdout_cut.tz_localize("UTC")
        if holdout_cut.tzinfo is None
        else holdout_cut.tz_convert("UTC")
    )
    if inner_cut >= holdout_cut:
        raise SelectiveDraftConstituentError("quantum time split is invalid")
    if train["game_uid"].astype(str).duplicated().any() or future[
        "game_uid"
    ].astype(str).duplicated().any():
        raise SelectiveDraftConstituentError("game identities are duplicated")
    if set(train["game_uid"].astype(str)) & set(future["game_uid"].astype(str)):
        raise SelectiveDraftConstituentError("training and evaluation overlap")
    if set(train["y"].astype(int).unique()) != {0, 1}:
        raise SelectiveDraftConstituentError("training outcomes are not binary")
    if not train["date"].lt(holdout_cut).all() or not future["date"].ge(
        holdout_cut
    ).all():
        raise SelectiveDraftConstituentError("quantum holdout boundary is violated")

    evaluation_series = set(future["series_id"].astype(str))
    outer_training = train[
        ~train["series_id"].astype(str).isin(evaluation_series)
    ].copy()
    outer_bounds = _series_bounds(outer_training)
    inner_training = _series_slice(outer_training, outer_bounds, end=inner_cut)
    inner_validation = _series_slice(
        outer_training, outer_bounds, start=inner_cut, end=holdout_cut
    )
    if (
        len(inner_training) < 50
        or len(inner_validation) < 50
        or inner_training["y"].nunique() != 2
        or inner_validation["y"].nunique() != 2
    ):
        raise SelectiveDraftConstituentError("quantum inner split is too small")

    anchor = _select_anchor(inner_validation, protocol)
    inner_worlds, inner_receipts = _quantum_stack(
        inner_training,
        inner_validation,
        protocol=protocol,
        game_by_id=game_by_id,
        cache_dir=cache_dir,
        matrix_sha256=matrix_sha256,
        feature_groups=feature_groups,
    )
    inner_anchor = _anchor_probability(
        inner_validation,
        team_weight=float(anchor["team_weight"]),
        momentum_weight=float(anchor["momentum_weight"]),
        source=str(anchor["source"]),
    )
    meta = _fit_quantum_meta(
        inner_validation["y"], inner_worlds, _logit(inner_anchor)
    )
    future_worlds, future_receipts = _quantum_stack(
        outer_training,
        future,
        protocol=protocol,
        game_by_id=game_by_id,
        cache_dir=cache_dir,
        matrix_sha256=matrix_sha256,
        feature_groups=feature_groups,
    )
    future_anchor = _anchor_probability(
        future,
        team_weight=float(anchor["team_weight"]),
        momentum_weight=float(anchor["momentum_weight"]),
        source=str(anchor["source"]),
    )
    probability = _quantum_meta_probability(
        meta, future_worlds, _logit(future_anchor)
    )
    output = pd.DataFrame(
        {
            "game_uid": future["game_uid"].astype(str).to_numpy(),
            "p": probability,
        }
    )
    receipt = {
        "schema_version": QUANTUM_VOTER_SCHEMA,
        "matrix_sha256": matrix_sha256,
        "feature_groups_sha256": canonical_sha256(
            {name: list(columns) for name, columns in feature_groups.items()}
        ),
        "selection_sha256": canonical_sha256(protocol["selection"]),
        "inner_start": inner_cut.isoformat(),
        "evaluation_start": holdout_cut.isoformat(),
        "training_game_ids_sha256": canonical_sha256(
            sorted(outer_training["game_uid"].astype(str))
        ),
        "input_training_rows": len(train),
        "series_complete_training_rows": len(outer_training),
        "inner_training_game_ids_sha256": canonical_sha256(
            sorted(inner_training["game_uid"].astype(str))
        ),
        "inner_validation_game_ids_sha256": canonical_sha256(
            sorted(inner_validation["game_uid"].astype(str))
        ),
        "evaluation_game_ids_sha256": canonical_sha256(
            sorted(future["game_uid"].astype(str))
        ),
        "anchor": anchor,
        "meta_intercept": float(meta.intercept_[0]),
        "meta_coefficients": [float(value) for value in meta.coef_[0]],
        "inner_receipts": inner_receipts,
        "evaluation_receipts": future_receipts,
        "prediction_sha256": canonical_sha256(
            [
                [str(game_uid), float(value)]
                for game_uid, value in zip(output["game_uid"], output["p"])
            ]
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return output, receipt


def fit_all_atom_identity_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    regularization_c: float = 0.0001,
    numeric_columns: Sequence[str] = MODEL_COLUMNS,
    categorical_columns: Sequence[str] = CATEGORICAL_CONTEXT_COLUMNS,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the frozen side-symmetric all-atom identity voter."""

    numeric = tuple(numeric_columns)
    categorical = tuple(categorical_columns)
    required_training = {"game_uid", "y", *numeric, *categorical}
    required_evaluation = {"game_uid", *numeric, *categorical}
    if not required_training.issubset(training.columns):
        raise SelectiveDraftConstituentError("training columns are incomplete")
    if not required_evaluation.issubset(evaluation.columns):
        raise SelectiveDraftConstituentError("evaluation columns are incomplete")
    if training["game_uid"].astype(str).duplicated().any() or evaluation[
        "game_uid"
    ].astype(str).duplicated().any():
        raise SelectiveDraftConstituentError("game identities are duplicated")
    if set(training["y"].astype(int).unique()) != {0, 1}:
        raise SelectiveDraftConstituentError("training outcomes are not binary")
    if not np.isfinite(float(regularization_c)) or regularization_c <= 0:
        raise SelectiveDraftConstituentError("regularization is invalid")
    training_numeric = training[list(numeric)].to_numpy(dtype=np.float64)
    evaluation_numeric = evaluation[list(numeric)].to_numpy(dtype=np.float64)
    if not np.isfinite(training_numeric).all() or not np.isfinite(
        evaluation_numeric
    ).all():
        raise SelectiveDraftConstituentError("numeric features are invalid")

    mirrored = _mirror_categorical_frame(
        training, numeric_columns=numeric
    )
    augmented = pd.concat([training, mirrored], ignore_index=True)
    target = np.concatenate(
        [training["y"].to_numpy(dtype=int), 1 - training["y"].to_numpy(dtype=int)]
    )
    categories = [
        sorted(augmented[column].astype(str).unique())
        for column in categorical
    ]
    encoder = OneHotEncoder(
        categories=categories,
        handle_unknown="ignore",
        sparse_output=True,
        dtype=np.float32,
    )
    training_categories = encoder.fit_transform(
        augmented[list(categorical)].astype(str)
    )
    evaluation_categories = encoder.transform(
        evaluation[list(categorical)].astype(str)
    )
    scaler = StandardScaler()
    augmented_numeric = scaler.fit_transform(
        augmented[list(numeric)].to_numpy(dtype=np.float64)
    ).astype(np.float32)
    transformed_evaluation = scaler.transform(evaluation_numeric).astype(
        np.float32
    )
    design = sparse.hstack(
        [sparse.csr_matrix(augmented_numeric), training_categories],
        format="csr",
    )
    evaluation_design = sparse.hstack(
        [sparse.csr_matrix(transformed_evaluation), evaluation_categories],
        format="csr",
    )
    model = LogisticRegression(
        C=float(regularization_c),
        max_iter=4000,
        solver="liblinear",
        fit_intercept=False,
    )
    model.fit(design, target)
    probability = model.predict_proba(evaluation_design)[:, 1]
    if not np.isfinite(probability).all() or np.any(
        (probability <= 0) | (probability >= 1)
    ):
        raise SelectiveDraftConstituentError("prediction is invalid")
    output = pd.DataFrame(
        {
            "game_uid": evaluation["game_uid"].astype(str).to_numpy(),
            "p": probability,
        }
    )
    receipt = {
        "schema_version": ALL_ATOM_IDENTITY_SCHEMA,
        "regularization_c": float(regularization_c),
        "solver": "liblinear",
        "fit_intercept": False,
        "side_symmetry_augmentation": True,
        "training_rows": len(training),
        "evaluation_rows": len(evaluation),
        "training_game_ids_sha256": canonical_sha256(
            sorted(training["game_uid"].astype(str))
        ),
        "evaluation_game_ids_sha256": canonical_sha256(
            sorted(evaluation["game_uid"].astype(str))
        ),
        "numeric_columns_sha256": canonical_sha256(list(numeric)),
        "categorical_columns_sha256": canonical_sha256(list(categorical)),
        "prediction_sha256": canonical_sha256(
            [
                [str(game_uid), float(value)]
                for game_uid, value in zip(output["game_uid"], output["p"])
            ]
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return output, receipt


def fit_strength_identity_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the frozen strength, context, and exact-identity logit voter."""

    numeric_columns = tuple(
        dict.fromkeys(
            (
                *STRENGTH_COLUMNS,
                *GROUP_COLUMNS["match_context"],
                *GROUP_COLUMNS["competition_context"],
            )
        )
    )
    output, receipt = fit_all_atom_identity_predictions(
        training,
        evaluation,
        regularization_c=0.001,
        numeric_columns=numeric_columns,
    )
    receipt.pop("receipt_sha256")
    receipt["schema_version"] = STRENGTH_IDENTITY_SCHEMA
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return output, receipt


def fit_roster_random_forest_predictions(
    training: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    cache_dir: Path | None = None,
    matrix_sha256: str = "",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the frozen roster-focused hidden-world forest voter."""

    if "game_uid" not in training or "game_uid" not in evaluation:
        raise SelectiveDraftConstituentError("game identities are missing")
    if "y" not in training:
        raise SelectiveDraftConstituentError("training outcome is missing")
    if training["game_uid"].astype(str).duplicated().any() or evaluation[
        "game_uid"
    ].astype(str).duplicated().any():
        raise SelectiveDraftConstituentError("game identities are duplicated")
    if set(training["y"].astype(int).unique()) != {0, 1}:
        raise SelectiveDraftConstituentError("training outcomes are not binary")
    if not re.fullmatch(r"[0-9a-f]{64}", str(matrix_sha256)):
        raise SelectiveDraftConstituentError("matrix SHA-256 is invalid")
    protocol = {
        "selection": {
            "quantum_worlds": [
                {"id": "composite-roster-world", "groups": list(ROSTER_GROUPS)}
            ],
            "quantum_forest_config": dict(ROSTER_FOREST_CONFIG),
        }
    }
    logits, world_receipts = _quantum_world_predictions(
        training,
        evaluation,
        protocol,
        cache_dir=cache_dir,
        matrix_sha256=matrix_sha256,
    )
    if logits.shape != (len(evaluation), 1):
        raise SelectiveDraftConstituentError("roster forest output is invalid")
    probability = expit(logits[:, 0])
    output = pd.DataFrame(
        {
            "game_uid": evaluation["game_uid"].astype(str).to_numpy(),
            "p": probability,
        }
    )
    receipt = {
        "schema_version": ROSTER_FOREST_SCHEMA,
        "matrix_sha256": matrix_sha256,
        "groups": list(ROSTER_GROUPS),
        "config": dict(ROSTER_FOREST_CONFIG),
        "training_game_ids_sha256": canonical_sha256(
            sorted(training["game_uid"].astype(str))
        ),
        "evaluation_game_ids_sha256": canonical_sha256(
            sorted(evaluation["game_uid"].astype(str))
        ),
        "world_receipts": world_receipts,
        "prediction_sha256": canonical_sha256(
            [
                [str(game_uid), float(value)]
                for game_uid, value in zip(output["game_uid"], output["p"])
            ]
        ),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return output, receipt


if __name__ == "__main__":
    main()
