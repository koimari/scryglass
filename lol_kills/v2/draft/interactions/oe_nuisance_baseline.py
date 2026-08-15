"""Fold-safe private OE nuisance baseline for draft representation assays.

This module fits only an intercept, league, exact OE patch token, and
role-specific champion main effects.  It deliberately has no interaction,
team, player, duration, or invented forecast-time inputs.  Predictions are
retrospective out-of-fold diagnostics, never public forecasts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

from .oe_target_authority import (
    DEFAULT_HUMAN_AUTHORITY_PATH,
    load_and_require_exact_human_authority,
)
from .oe_target_evidence import (
    DEFAULT_EVIDENCE_PATH,
    DEFAULT_PRIVATE_ROWS_PATH,
    DEFAULT_SPLIT_PATH,
    FINAL_SPLIT,
    OETargetEvidenceError,
    ROLE_ORDER,
    canonical_bytes,
    canonical_sha256,
    ordered_input_sha256,
    raw_sha256,
    selected_input_sha256,
    validate_evidence,
    validate_split,
)


SCHEMA_ID = "scryglass.oe-private-nuisance-baseline.v1"
GENERATOR_VERSION = "oe-private-nuisance-baseline-generator.v2"
DEFAULT_ARTIFACT_PATH = Path(
    "data/lol/v2/models/draft-interactions/oe-private-nuisance-baseline.json"
)
DEFAULT_OOF_PATH = Path(
    "data/lol/warehouse/private_v2/draft-interactions/oe-nuisance-oof.parquet"
)
ALLOWED_SPLITS = ("train", "development", "validation")
REGULARIZATION_C_GRID = (
    0.00001,
    0.00003,
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
)
MINIMUM_PRIOR_ROWS = 50
SOLVER = "lbfgs"
MAXIMUM_ITERATIONS = 5000
TOLERANCE = 1e-12
RANDOM_SEED = 0

FEATURE_COLUMNS = (
    "game_id",
    "dependence_cluster_id",
    "split",
    "oe_date_naive",
    "canonical_league",
    "oe_patch_token",
    *(
        f"{side}_{role}_stable_champion_id"
        for side in ("blue", "red")
        for role in ROLE_ORDER
    ),
)
TARGET_COLUMNS = ("y_blue_win",)
INPUT_COLUMNS = FEATURE_COLUMNS + TARGET_COLUMNS
FORBIDDEN_INPUT_TOKENS = (
    "duration",
    "gamelength",
    "team",
    "player",
    "forecast_at",
    "draft_completed_at",
    "derived_resolution",
)


class OENuisanceBaselineError(ValueError):
    """Raised when the baseline's leakage or authority contract fails closed."""


def _runtime_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy")
    }


def _generator_identity() -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    runtime = _runtime_versions()
    return {
        "version": GENERATOR_VERSION,
        "executable_dependency_boundary": [
            {
                "locator": "lol_kills/v2/draft/interactions/oe_nuisance_baseline.py",
                "raw_sha256": raw_sha256(module_path),
            }
        ],
        "runtime_versions": runtime,
        "runtime_identity_sha256": canonical_sha256(runtime),
    }


def _load_json_canonical(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OENuisanceBaselineError(f"{label} is not valid JSON") from exc
    if raw != canonical_bytes(payload):
        raise OENuisanceBaselineError(f"{label} is not canonical JSON")
    return payload, raw


def _month_start(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None:
        raise OENuisanceBaselineError("OE date must be present and timezone-naive")
    return stamp.to_period("M").start_time


def _validate_input_frame(frame: pd.DataFrame) -> pd.DataFrame:
    unexpected = set(frame.columns) - set(INPUT_COLUMNS)
    missing = set(INPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise OENuisanceBaselineError(
            f"baseline input is missing columns: {sorted(missing)}"
        )
    if unexpected:
        if any(
            token in str(column).casefold()
            for column in unexpected
            for token in FORBIDDEN_INPUT_TOKENS
        ):
            raise OENuisanceBaselineError("forbidden duration/team/player input")
        raise OENuisanceBaselineError(
            f"baseline input has unreviewed columns: {sorted(unexpected)}"
        )
    clean = frame.loc[:, INPUT_COLUMNS].copy()
    if clean["game_id"].duplicated().any():
        raise OENuisanceBaselineError("baseline game identity is not unique")
    if (clean["split"] == FINAL_SPLIT).any():
        raise OENuisanceBaselineError("sealed final temporal holdout was accessed")
    if not set(clean["split"]).issubset(ALLOWED_SPLITS):
        raise OENuisanceBaselineError("baseline input has an unknown split")
    if clean["y_blue_win"].map(
        lambda value: isinstance(value, (bool, np.bool_))
        or pd.isna(value)
        or int(value) not in (0, 1)
        or float(value) != int(value)
    ).any():
        raise OENuisanceBaselineError("baseline target is not finite binary")
    for column in FEATURE_COLUMNS:
        if clean[column].map(lambda value: pd.isna(value) or not str(value).strip()).any():
            raise OENuisanceBaselineError(f"baseline feature is missing: {column}")
    clean["oe_date_naive"] = clean["oe_date_naive"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    clean["y_blue_win"] = clean["y_blue_win"].astype(int)
    return clean


def _validate_against_split(
    frame: pd.DataFrame, split: Mapping[str, Any]
) -> pd.DataFrame:
    """Validate the outcome-free split before touching the target column."""
    try:
        validate_split(split)
    except OETargetEvidenceError as exc:
        raise OENuisanceBaselineError(str(exc)) from exc
    assignments = {
        str(row["game_id"]): row
        for row in split.get("assignments", ())
        if row.get("split") != FINAL_SPLIT
    }
    expected_ids = set(assignments)
    observed_ids = set(frame["game_id"].astype(str))
    if observed_ids != expected_ids:
        raise OENuisanceBaselineError(
            "baseline rows do not exactly match non-holdout split membership"
        )
    for row in frame.loc[
        :, ("game_id", "dependence_cluster_id", "split", "oe_date_naive")
    ].itertuples(index=False):
        assigned = assignments[str(row.game_id)]
        if (
            str(row.dependence_cluster_id)
            != str(assigned["dependence_cluster_id"])
            or str(row.split) != str(assigned["split"])
            or pd.Timestamp(row.oe_date_naive)
            != pd.Timestamp(assigned["oe_date_naive"])
        ):
            raise OENuisanceBaselineError(
                "private row disagrees with outcome-free split assignment"
            )
    return _validate_input_frame(frame)


def _levels(fit: pd.DataFrame) -> dict[str, Any]:
    leagues = sorted(fit["canonical_league"].astype(str).unique())
    patches = sorted(fit["oe_patch_token"].astype(str).unique())
    champions = {
        role: sorted(
            set(fit[f"blue_{role}_stable_champion_id"].astype(str))
            | set(fit[f"red_{role}_stable_champion_id"].astype(str))
        )
        for role in ROLE_ORDER
    }
    return {
        "league_reference": leagues[0],
        "league_levels": leagues,
        "patch_reference": patches[0],
        "patch_levels": patches,
        "champion_reference_by_role": {
            role: champions[role][0] for role in ROLE_ORDER
        },
        "champion_levels_by_role": champions,
    }


def _column_spec(levels: Mapping[str, Any]) -> list[tuple[str, str, str | None]]:
    columns: list[tuple[str, str, str | None]] = []
    columns.extend(
        ("league", level, None)
        for level in levels["league_levels"]
        if level != levels["league_reference"]
    )
    columns.extend(
        ("patch", level, None)
        for level in levels["patch_levels"]
        if level != levels["patch_reference"]
    )
    for role in ROLE_ORDER:
        reference = levels["champion_reference_by_role"][role]
        columns.extend(
            ("champion", level, role)
            for level in levels["champion_levels_by_role"][role]
            if level != reference
        )
    return columns


def _design(
    frame: pd.DataFrame,
    levels: Mapping[str, Any],
    columns: Sequence[tuple[str, str, str | None]],
) -> sparse.csr_matrix:
    index = {column: position for position, column in enumerate(columns)}
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for row_number, row in enumerate(frame.itertuples(index=False)):
        league = ("league", str(row.canonical_league), None)
        patch = ("patch", str(row.oe_patch_token), None)
        for key in (league, patch):
            if key in index:
                rows.append(row_number)
                cols.append(index[key])
                values.append(1.0)
        for role in ROLE_ORDER:
            blue = (
                "champion",
                str(getattr(row, f"blue_{role}_stable_champion_id")),
                role,
            )
            red = (
                "champion",
                str(getattr(row, f"red_{role}_stable_champion_id")),
                role,
            )
            for key, sign in ((blue, 1.0), (red, -1.0)):
                if key in index:
                    rows.append(row_number)
                    cols.append(index[key])
                    values.append(sign)
    return sparse.csr_matrix(
        (values, (rows, cols)), shape=(len(frame), len(columns)), dtype=np.float64
    )


def _fit_predict(
    fit: pd.DataFrame, predict: pd.DataFrame, *, regularization_c: float
) -> np.ndarray:
    levels = _levels(fit)
    columns = _column_spec(levels)
    x_fit = _design(fit, levels, columns)
    x_predict = _design(predict, levels, columns)
    model = LogisticRegression(
        penalty="l2",
        C=regularization_c,
        fit_intercept=True,
        solver=SOLVER,
        random_state=RANDOM_SEED,
        max_iter=MAXIMUM_ITERATIONS,
        tol=TOLERANCE,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_fit, fit["y_blue_win"].to_numpy())
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise OENuisanceBaselineError("logistic fit emitted ConvergenceWarning")
    n_iter = np.asarray(getattr(model, "n_iter_", None))
    if n_iter.shape != (1,) or not np.issubdtype(n_iter.dtype, np.integer):
        raise OENuisanceBaselineError("logistic iteration shape changed")
    if int(n_iter[0]) >= MAXIMUM_ITERATIONS:
        raise OENuisanceBaselineError("logistic fit reached iteration limit")
    if model.classes_.tolist() != [0, 1]:
        raise OENuisanceBaselineError("binary class order changed")
    coefficients = np.asarray(model.coef_)
    intercept = np.asarray(model.intercept_)
    if coefficients.shape != (1, x_fit.shape[1]) or intercept.shape != (1,):
        raise OENuisanceBaselineError("logistic coefficient shape changed")
    if not np.isfinite(coefficients).all() or not np.isfinite(intercept).all():
        raise OENuisanceBaselineError("logistic coefficients are nonfinite")
    decision = np.asarray(model.decision_function(x_predict))
    if decision.shape != (len(predict),) or not np.isfinite(decision).all():
        raise OENuisanceBaselineError("logistic decision values are invalid")
    probabilities = np.asarray(model.predict_proba(x_predict))
    if probabilities.shape != (len(predict), 2):
        raise OENuisanceBaselineError("logistic probability shape changed")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities <= 0.0).any()
        or (probabilities >= 1.0).any()
    ):
        raise OENuisanceBaselineError(
            "logistic probabilities must be finite and strictly inside (0,1)"
        )
    return probabilities[:, 1]


def _cross_fit_candidate(
    frame: pd.DataFrame, *, regularization_c: float
) -> pd.DataFrame:
    """Return expanding-window OOF predictions from strictly earlier clusters."""
    clean = _validate_input_frame(frame)
    cluster_dates = (
        clean.groupby("dependence_cluster_id", sort=True)["oe_date_naive"]
        .max()
        .map(pd.Timestamp)
    )
    cluster_month = cluster_dates.map(_month_start)
    output: list[dict[str, Any]] = []
    for month in sorted(cluster_month.unique()):
        prediction_clusters = set(cluster_month[cluster_month == month].index)
        fit_clusters = set(cluster_dates[cluster_dates < month].index)
        prediction = clean[
            clean["dependence_cluster_id"].isin(prediction_clusters)
        ].sort_values("game_id")
        fit = clean[clean["dependence_cluster_id"].isin(fit_clusters)].sort_values(
            "game_id"
        )
        if len(fit) < MINIMUM_PRIOR_ROWS or fit["y_blue_win"].nunique() != 2:
            continue
        if fit["oe_date_naive"].map(pd.Timestamp).max() >= month:
            raise OENuisanceBaselineError("future-fit leakage detected")
        probability = _fit_predict(
            fit, prediction, regularization_c=regularization_c
        )
        intercept_probability = float(fit["y_blue_win"].mean())
        fit_cluster_hash = canonical_sha256(sorted(fit_clusters))
        for row, value in zip(prediction.itertuples(index=False), probability):
            output.append(
                {
                    "game_id": str(row.game_id),
                    "dependence_cluster_id": str(row.dependence_cluster_id),
                    "split": str(row.split),
                    "oe_date_naive": str(row.oe_date_naive),
                    "prediction_fold_month_naive": month.isoformat(),
                    "fit_maximum_date_naive": fit["oe_date_naive"]
                    .map(pd.Timestamp)
                    .max()
                    .isoformat(),
                    "fit_rows": int(len(fit)),
                    "fit_dependence_clusters": int(len(fit_clusters)),
                    "fit_cluster_membership_sha256": fit_cluster_hash,
                    "p_blue_win_nuisance_oof": float(value),
                    "p_blue_win_richer_candidate_oof": float(value),
                    "p_blue_win_intercept_oof": intercept_probability,
                    "y_blue_win": int(row.y_blue_win),
                }
            )
    if not output:
        raise OENuisanceBaselineError("no fold has sufficient strictly earlier support")
    result = pd.DataFrame(output).sort_values(
        ["oe_date_naive", "game_id"], ignore_index=True
    )
    if result["game_id"].duplicated().any():
        raise OENuisanceBaselineError("OOF game identity is not unique")
    return result


def _candidate_vs_intercept_scores(rows: pd.DataFrame) -> dict[str, Any]:
    return {
        "richer_candidate": _score_probability(
            rows, rows["p_blue_win_richer_candidate_oof"].to_numpy()
        ),
        "fold_safe_expanding_intercept_only": _score_probability(
            rows, rows["p_blue_win_intercept_oof"].to_numpy()
        ),
    }


def _improves_both(scores: Mapping[str, Any]) -> bool:
    candidate = scores["richer_candidate"]
    reference = scores["fold_safe_expanding_intercept_only"]
    return bool(
        candidate["log_loss"] < reference["log_loss"]
        and candidate["brier_score"] < reference["brier_score"]
    )


def select_regularization(train: pd.DataFrame) -> dict[str, Any]:
    """Freeze one nuisance method using only chronological train inner folds."""
    clean = _validate_input_frame(train)
    if set(clean["split"]) != {"train"}:
        raise OENuisanceBaselineError(
            "regularization selection may use train split only"
        )
    candidates: list[dict[str, Any]] = []
    for regularization_c in REGULARIZATION_C_GRID:
        try:
            inner_oof = _cross_fit_candidate(
                clean, regularization_c=regularization_c
            )
        except OENuisanceBaselineError as exc:
            if str(exc) != "no fold has sufficient strictly earlier support":
                raise
            candidates.append(
                {
                    "C": regularization_c,
                    "availability": "unavailable",
                    "unavailable_reason": (
                        "no fold has sufficient strictly earlier support"
                    ),
                    "inner_oof_rows": 0,
                    "scores": None,
                    "improves_both_over_intercept": False,
                }
            )
            continue
        scores = _candidate_vs_intercept_scores(inner_oof)
        candidates.append(
            {
                "C": regularization_c,
                "availability": "available",
                "unavailable_reason": None,
                "inner_oof_rows": int(len(inner_oof)),
                "scores": scores,
                "improves_both_over_intercept": _improves_both(scores),
            }
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["improves_both_over_intercept"]
    ]
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["scores"]["richer_candidate"]["log_loss"],
            candidate["scores"]["richer_candidate"]["brier_score"],
            candidate["C"],
        ),
        default=None,
    )
    return {
        "grid": list(REGULARIZATION_C_GRID),
        "selection_population": "train_only",
        "train_rows": int(len(clean)),
        "train_dependence_clusters": int(
            clean["dependence_cluster_id"].nunique()
        ),
        "train_cluster_membership_sha256": canonical_sha256(
            sorted(clean["dependence_cluster_id"].astype(str).unique())
        ),
        "selection_fold_rule": (
            "cluster-atomic calendar-month inner folds fitted only on strictly "
            "earlier clusters within the fixed train split"
        ),
        "rule": (
            "retain candidates improving both inner-fold log loss and Brier "
            "against fold-safe intercept-only; choose minimum log loss, then "
            "minimum Brier, then smallest C; use intercept-only when no "
            "candidate passes or inner support is unavailable"
        ),
        "candidates": candidates,
        "selected_C": None if selected is None else selected["C"],
        "selected_method": (
            "intercept_only" if selected is None else "richer_main_effects"
        ),
    }


def _outer_confirmation_gate(rows: pd.DataFrame) -> dict[str, Any]:
    by_split = {
        split: _candidate_vs_intercept_scores(rows[rows["split"] == split])
        for split in ("development", "validation")
    }
    passes = all(_improves_both(scores) for scores in by_split.values())
    return {
        "rule": (
            "the train-frozen nuisance permits a downstream rank assay only if "
            "development and validation each strictly improve both log loss and "
            "Brier over fold-safe intercept-only"
        ),
        "by_split": by_split,
        "passed": passes,
        "eligible_for_downstream_rank_assay": passes,
        "changes_frozen_nuisance_predictions": False,
        "failure_consequence": (
            None
            if passes
            else "fail closed: do not run or interpret the downstream rank assay"
        ),
    }


def cross_fit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Return monthly OOF predictions using one train-frozen nuisance method."""
    clean = _validate_input_frame(frame)
    selection = select_regularization(clean[clean["split"] == "train"])
    selected_c = selection["selected_C"]
    candidate_c = (
        REGULARIZATION_C_GRID[0] if selected_c is None else selected_c
    )
    selected = _cross_fit_candidate(clean, regularization_c=candidate_c)
    selected["selected_regularization_C"] = selected_c
    selected["selected_nuisance_method"] = selection["selected_method"]
    if selection["selected_method"] == "intercept_only":
        selected["p_blue_win_nuisance_oof"] = selected[
            "p_blue_win_intercept_oof"
        ]
    selected.attrs["regularization_selection"] = selection
    gate = _outer_confirmation_gate(selected)
    if selection["selected_method"] == "intercept_only":
        gate["passed"] = False
        gate["eligible_for_downstream_rank_assay"] = False
        gate["failure_consequence"] = (
            "fail closed: train-only selection chose intercept-only, so the "
            "richer nuisance improvement required for the rank assay is unavailable"
        )
    selected.attrs["outer_confirmation_gate"] = gate
    return selected


def _score(rows: pd.DataFrame) -> dict[str, Any]:
    y = rows["y_blue_win"].to_numpy(dtype=float)
    p = rows["p_blue_win_nuisance_oof"].to_numpy(dtype=float)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "rows": int(len(rows)),
        "blue_win_fraction": float(y.mean()),
        "mean_predicted_blue_win": float(p.mean()),
        "log_loss": float(
            np.mean(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
        ),
        "brier_score": float(np.mean((p - y) ** 2)),
    }


def _score_probability(rows: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    y = rows["y_blue_win"].to_numpy(dtype=float)
    p = np.asarray(probability, dtype=float)
    clipped = np.clip(p, 1e-15, 1 - 1e-15)
    return {
        "rows": int(len(rows)),
        "log_loss": float(
            np.mean(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
        ),
        "brier_score": float(np.mean((p - y) ** 2)),
    }


def _calibration(rows: pd.DataFrame) -> list[dict[str, Any]]:
    bins = np.minimum(
        (rows["p_blue_win_nuisance_oof"].to_numpy() * 10).astype(int), 9
    )
    output: list[dict[str, Any]] = []
    for bin_id in range(10):
        selected = rows.iloc[np.flatnonzero(bins == bin_id)]
        if selected.empty:
            continue
        output.append(
            {
                "bin": bin_id,
                "lower_inclusive": bin_id / 10,
                "upper_inclusive_only_for_final_bin": (bin_id + 1) / 10,
                "rows": int(len(selected)),
                "mean_prediction": float(
                    selected["p_blue_win_nuisance_oof"].mean()
                ),
                "observed_blue_win_fraction": float(
                    selected["y_blue_win"].mean()
                ),
            }
        )
    return output


def _read_non_holdout_private_rows(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise OENuisanceBaselineError(
            "private target rows must be a regular non-symlink file"
        )
    # The predicate is applied by the parquet reader before returning targets.
    # No final-holdout target is materialized into this process.
    return pd.read_parquet(
        path,
        columns=list(INPUT_COLUMNS),
        filters=[("split", "in", list(ALLOWED_SPLITS))],
    )


def build_artifact(
    *,
    split_path: Path = DEFAULT_SPLIT_PATH,
    evidence_path: Path = DEFAULT_EVIDENCE_PATH,
    authority_path: Path = DEFAULT_HUMAN_AUTHORITY_PATH,
    private_rows_path: Path = DEFAULT_PRIVATE_ROWS_PATH,
    oof_path: Path = DEFAULT_OOF_PATH,
) -> dict[str, Any]:
    split, split_bytes = _load_json_canonical(split_path, "split")
    evidence, evidence_bytes = _load_json_canonical(evidence_path, "evidence")
    try:
        validate_split(split)
        validate_evidence(evidence)
        authority = load_and_require_exact_human_authority(
            evidence, split, action="model_fit", authority_path=authority_path
        )
    except OETargetEvidenceError as exc:
        raise OENuisanceBaselineError(str(exc)) from exc
    if authority.get("reviewer_identity") != "KOI_MARI":
        raise OENuisanceBaselineError("model_fit authority is not KOI_MARI")
    if authority.get("final_temporal_holdout_sealed") is not True:
        raise OENuisanceBaselineError("final temporal holdout is not sealed")
    if raw_sha256(private_rows_path) != evidence["private_materialization"]["raw_sha256"]:
        raise OENuisanceBaselineError("private target materialization bytes changed")

    rows = _read_non_holdout_private_rows(private_rows_path)
    rows = _validate_against_split(rows, split)
    oof = cross_fit_rows(rows)
    regularization_selection = oof.attrs["regularization_selection"]
    outer_confirmation_gate = oof.attrs["outer_confirmation_gate"]
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(oof_path, index=False)

    per_split = {
        name: _score(oof[oof["split"] == name])
        for name in ALLOWED_SPLITS
        if (oof["split"] == name).any()
    }
    nuisance_score = _score(oof)
    constant_score = _score_probability(oof, np.full(len(oof), 0.5))
    intercept_score = _score_probability(
        oof, oof["p_blue_win_intercept_oof"].to_numpy()
    )
    runtime = _runtime_versions()
    authority_bytes = authority_path.read_bytes()
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "status": "private_pending_rank_selection",
        "development_only": True,
        "retrospective_out_of_fold_only": True,
        "representation_rank_selected": False,
        "predictive_authority": False,
        "authorizes_rank_selection": False,
        "authorizes_prediction": False,
        "authorizes_publication": False,
        "authorizes_production": False,
        "authorizes_sota_claim": False,
        "content_addressing_confers_authority": False,
        "claim_ceiling": (
            "descriptive private nuisance-baseline scores only; no representation "
            "rank, forecast, publication, production, or SOTA authority"
        ),
        "generator": _generator_identity(),
        "source_identity": {
            "split": {
                "locator": split_path.as_posix(),
                "raw_sha256": hashlib.sha256(split_bytes).hexdigest(),
                "payload_sha256": split["artifact_sha256"],
                "outcome_free_split_sha256": split["outcome_free_split_sha256"],
            },
            "target_evidence": {
                "locator": evidence_path.as_posix(),
                "raw_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
                "payload_sha256": evidence["artifact_sha256"],
                "feature_domain_sha256": evidence["feature_domain_sha256"],
                "target_domain_sha256": evidence["target_domain_sha256"],
                "target_transform_sha256": evidence["target_transform_sha256"],
            },
            "private_rows": {
                "locator": private_rows_path.as_posix(),
                "raw_sha256": evidence["private_materialization"]["raw_sha256"],
                "logical_rows_sha256": evidence["private_materialization"][
                    "logical_rows_sha256"
                ],
            },
            "human_authority": {
                "locator": authority_path.as_posix(),
                "raw_sha256": hashlib.sha256(authority_bytes).hexdigest(),
                "decision_id": authority["decision_id"],
                "reviewer_identity": authority["reviewer_identity"],
                "approved_action_used": "model_fit",
            },
        },
        "feature_contract": {
            "intercept": True,
            "categorical_main_effects": [
                "canonical_league",
                "exact_oe_patch_token",
                *[f"{role}_champion_blue_minus_red" for role in ROLE_ORDER],
            ],
            "champion_coding": (
                "+1 for the blue role champion and -1 for the red role champion"
            ),
            "identifiability_constraints": (
                "lexicographically first observed league and patch are omitted; "
                "within each role the lexicographically first observed champion "
                "is omitted, equivalent to choosing one representative from the "
                "blue-minus-red zero-sum coefficient class"
            ),
            "unseen_level_rule": "zero contribution relative to the fit-fold reference",
            "forbidden": [
                "team controls",
                "player controls",
                "ally interactions",
                "enemy interactions",
                "same-role interactions",
                "duration",
                "draft_completed_at",
                "forecast_at",
                "derived_resolution_time",
            ],
            "feature_contract_sha256": canonical_sha256(
                {
                    "columns": list(FEATURE_COLUMNS),
                    "regularization_c_grid": list(REGULARIZATION_C_GRID),
                    "role_order": list(ROLE_ORDER),
                    "coding": "reference-coded league/patch; role champion blue-minus-red",
                }
            ),
        },
        "estimator": {
            "family": "binary logistic regression",
            "method": (
                "train-frozen choice between deterministic sparse "
                "L2-penalized lbfgs and expanding intercept-only"
            ),
            "regularization_selection": regularization_selection,
            "selected_C": regularization_selection["selected_C"],
            "selected_C_scope": "global_train_frozen",
            "selected_nuisance_offset": regularization_selection[
                "selected_method"
            ],
            "fit_intercept": True,
            "intercept_penalized": False,
            "feature_order": [
                "league levels lexical excluding reference",
                "patch levels lexical excluding reference",
                *[
                    f"{role} champion levels lexical excluding reference"
                    for role in ROLE_ORDER
                ],
            ],
            "class_order": [0, 1],
            "solver": SOLVER,
            "random_state": RANDOM_SEED,
            "maximum_iterations": MAXIMUM_ITERATIONS,
            "tolerance": TOLERANCE,
            "runtime_identity_sha256": canonical_sha256(runtime),
        },
        "fold_contract": {
            "unit": "dependence_cluster_id",
            "prediction_block": "calendar month of cluster maximum OE date",
            "fit_rule": (
                "only complete dependence clusters with maximum OE date strictly "
                "before the prediction month"
            ),
            "minimum_prior_rows": MINIMUM_PRIOR_ROWS,
            "cluster_atomic": True,
            "strictly_earlier": True,
            "nested_selection": False,
            "train_frozen_selection": True,
            "posthoc_development_or_validation_prediction_switch": False,
            "rank_assay_oof_interface": {
                "join_key": "game_id",
                "probability_column": "p_blue_win_nuisance_oof",
                "required_fold_identity_columns": [
                    "prediction_fold_month_naive",
                    "fit_maximum_date_naive",
                    "fit_rows",
                    "fit_dependence_clusters",
                    "fit_cluster_membership_sha256",
                ],
                "consumer_rule": (
                    "every representation candidate must use the exact same OOF "
                    "rows and fold-local fit-cluster membership"
                ),
            },
            "final_temporal_holdout": {
                "status": "sealed_unaccessed",
                "maps": split["counts"]["by_split"][FINAL_SPLIT],
                "targets_read": False,
                "predictions": 0,
                "fit_rows": 0,
                "score_rows": 0,
            },
        },
        "oof_materialization": {
            "locator": oof_path.as_posix(),
            "expected_git_ignored": True,
            "strict_raw_byte_replay_required": True,
            "rows": int(len(oof)),
            "raw_sha256": raw_sha256(oof_path),
            "logical_rows_sha256": selected_input_sha256(oof),
            "ordered_logical_rows_sha256": ordered_input_sha256(oof),
            "predicted_game_membership_sha256": canonical_sha256(
                sorted(oof["game_id"].tolist())
            ),
        },
        "descriptive_diagnostics": {
            "selection_use": False,
            "prediction_selection_use": (
                "one method and C selected from cluster-atomic chronological "
                "inner OOF scores on the fixed train split only, then frozen "
                "before development and validation"
            ),
            "overall": nuisance_score,
            "by_split": per_split,
            "equal_width_calibration": _calibration(oof),
            "reference_scores": {
                "constant_0_5": constant_score,
                "fold_safe_expanding_intercept_only": intercept_score,
            },
            "outer_confirmation_gate": outer_confirmation_gate,
            "outer_scores_do_not_change_predictions": True,
        },
        "design_history_disclosure": (
            "This nuisance redesign was informed by already-opened non-holdout "
            "nuisance-only diagnostics. No interaction-rank outcomes were inspected. "
            "The C grid and deterministic selection rule were frozen before this "
            "artifact regeneration."
        ),
        "literature_and_formal_decisions": [
            {
                "decision": (
                    "use strictly proper log loss and Brier score only to describe "
                    "the frozen baseline, never to select representation rank"
                ),
                "citation": (
                    "Dimitriadis, Gneiting, and Jordan (2021/2022), "
                    "doi:10.1073/pnas.2016191118"
                ),
            },
            {
                "decision": (
                    "separate fit and assessment observations and preserve "
                    "dependence-cluster atomicity under ordered cross-fitting"
                ),
                "citation": (
                    "Chernozhukov et al. (2018), doi:10.1111/ectj.12097; "
                    "temporal ordering is a conservative project constraint"
                ),
            },
            {
                "decision": (
                    "reference constraints identify categorical main effects; "
                    "no interaction estimand is introduced"
                ),
                "citation": (
                    "Wolfram LogitModelFit nominal-variable and constrained-model "
                    "documentation consulted 2026-07-29"
                ),
            },
        ],
        "limitations": [
            "retrospective OE date is not an observed draft-completion or forecast timestamp",
            "the baseline omits team and player strength by design",
            "unseen categorical levels receive zero contribution",
            "monthly expanding folds sacrifice early rows and may reflect temporal drift",
            "".join(
                (
                    "regularization and intercept-only fallback are selected once from ",
                    "train-only inner OOF predictions and frozen before development and validation",
                )
            ),
            "proper scores and calibration are descriptive and do not authorize rank selection",
            "outer development and validation outcomes never switch predictions after the fact",
            "the sealed final temporal holdout remains wholly unevaluated",
        ],
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise OENuisanceBaselineError("baseline artifact hash mismatch")
    if payload.get("status") != "private_pending_rank_selection":
        raise OENuisanceBaselineError("baseline status is not rank-selection pending")
    false_flags = (
        "representation_rank_selected",
        "predictive_authority",
        "authorizes_rank_selection",
        "authorizes_prediction",
        "authorizes_publication",
        "authorizes_production",
        "authorizes_sota_claim",
        "content_addressing_confers_authority",
    )
    if any(payload.get(field) is not False for field in false_flags):
        raise OENuisanceBaselineError("baseline authority ceiling was exceeded")
    holdout = payload.get("fold_contract", {}).get("final_temporal_holdout", {})
    if holdout != {
        "status": "sealed_unaccessed",
        "maps": 361,
        "targets_read": False,
        "predictions": 0,
        "fit_rows": 0,
        "score_rows": 0,
    }:
        raise OENuisanceBaselineError("sealed final temporal holdout contract changed")
    if payload.get("descriptive_diagnostics", {}).get("selection_use") is not False:
        raise OENuisanceBaselineError("baseline diagnostics were used for selection")
    if payload.get("descriptive_diagnostics", {}).get(
        "prediction_selection_use"
    ) != (
        "one method and C selected from cluster-atomic chronological inner OOF "
        "scores on the fixed train split only, then frozen before development "
        "and validation"
    ):
        raise OENuisanceBaselineError("prediction selection disclosure changed")
    if payload.get("oof_materialization", {}).get(
        "strict_raw_byte_replay_required"
    ) is not True:
        raise OENuisanceBaselineError("OOF raw-byte replay contract changed")
    estimator = payload.get("estimator", {})
    selection = estimator.get("regularization_selection", {})
    if (
        selection.get("grid") != list(REGULARIZATION_C_GRID)
        or selection.get("selection_population") != "train_only"
        or estimator.get("selected_C") != selection.get("selected_C")
        or estimator.get("selected_C_scope") != "global_train_frozen"
        or estimator.get("selected_nuisance_offset")
        != selection.get("selected_method")
    ):
        raise OENuisanceBaselineError("regularization selection contract changed")
    if (
        payload.get("fold_contract", {}).get("train_frozen_selection") is not True
        or payload.get("fold_contract", {}).get(
            "posthoc_development_or_validation_prediction_switch"
        )
        is not False
        or payload.get("descriptive_diagnostics", {}).get(
            "outer_scores_do_not_change_predictions"
        )
        is not True
    ):
        raise OENuisanceBaselineError("train-frozen nuisance contract changed")
    gate = payload.get("descriptive_diagnostics", {}).get(
        "outer_confirmation_gate", {}
    )
    if (
        gate.get("changes_frozen_nuisance_predictions") is not False
        or gate.get("eligible_for_downstream_rank_assay")
        is not gate.get("passed")
    ):
        raise OENuisanceBaselineError("outer confirmation gate changed")


def write_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH, **kwargs: Any
) -> dict[str, Any]:
    payload = build_artifact(**kwargs)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(canonical_bytes(payload))
    return payload


def load_and_replay_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    source_root: Path = Path.cwd(),
) -> dict[str, Any]:
    persisted_bytes = artifact_path.read_bytes()
    persisted = json.loads(persisted_bytes)
    validate_artifact(persisted)
    if persisted_bytes != canonical_bytes(persisted):
        raise OENuisanceBaselineError("persisted baseline artifact is not canonical")

    def resolve(locator: str) -> Path:
        path = Path(locator)
        return path if path.is_absolute() else source_root / path

    source = persisted["source_identity"]
    oof_path = resolve(persisted["oof_materialization"]["locator"])
    if not oof_path.is_file() or oof_path.is_symlink():
        raise OENuisanceBaselineError("persisted OOF materialization is unavailable")
    if raw_sha256(oof_path) != persisted["oof_materialization"]["raw_sha256"]:
        raise OENuisanceBaselineError("persisted OOF bytes changed")
    persisted_oof = pd.read_parquet(oof_path)
    if (
        selected_input_sha256(persisted_oof)
        != persisted["oof_materialization"]["logical_rows_sha256"]
        or ordered_input_sha256(persisted_oof)
        != persisted["oof_materialization"]["ordered_logical_rows_sha256"]
    ):
        raise OENuisanceBaselineError("persisted OOF logical rows changed")
    with tempfile.TemporaryDirectory() as tmp:
        replay_oof = Path(tmp) / "oe-nuisance-oof.parquet"
        replay = build_artifact(
            split_path=resolve(source["split"]["locator"]),
            evidence_path=resolve(source["target_evidence"]["locator"]),
            authority_path=resolve(source["human_authority"]["locator"]),
            private_rows_path=resolve(source["private_rows"]["locator"]),
            oof_path=replay_oof,
        )
        regenerated_raw_sha256 = raw_sha256(replay_oof)
        if regenerated_raw_sha256 != persisted["oof_materialization"]["raw_sha256"]:
            raise OENuisanceBaselineError(
                "regenerated OOF raw bytes do not match persisted OOF"
            )
    # The private OOF locator is metadata; replay used a temporary output path.
    replay["oof_materialization"]["locator"] = persisted["oof_materialization"][
        "locator"
    ]
    for source_name in (
        "split",
        "target_evidence",
        "human_authority",
        "private_rows",
    ):
        replay["source_identity"][source_name]["locator"] = persisted[
            "source_identity"
        ][source_name]["locator"]
    replay["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in replay.items() if key != "artifact_sha256"}
    )
    if canonical_bytes(replay) != persisted_bytes:
        raise OENuisanceBaselineError(
            "strict source-backed baseline replay does not match persisted payload"
        )
    return persisted


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args(argv)
    payload = (
        load_and_replay_artifact()
        if args.verify_existing
        else write_artifact()
    )
    print(
        json.dumps(
            {
                "artifact_sha256": payload["artifact_sha256"],
                "oof_rows": payload["oof_materialization"]["rows"],
                "overall": payload["descriptive_diagnostics"]["overall"],
                "strict_replay_verified": bool(args.verify_existing),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
