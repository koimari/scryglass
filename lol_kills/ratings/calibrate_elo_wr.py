#!/usr/bin/env python3
"""
Calibrate Elo→WR mapping on time-safe train only (avoid overfitting).

Fits logistic:
  P(blue win) = sigmoid(a + b * (μ_diff / 400))

so the numeric Elo gap matches empirical WR (player scale was "hot").
Also produces a blended strength prior.

  python3 -m lol_kills.ratings.calibrate_elo_wr
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR
from lol_kills.ml.eval import holdout_cut

OUT = MODELS_DIR / "elo_wr_calibration.json"
MIN_TRAIN_ROWS = 400
MIN_HOLDOUT_ROWS = 100


class CalibrationArtifactError(RuntimeError):
    """Raised when a calibration cannot be represented as time-safe."""


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _finite_number(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CalibrationArtifactError(f"{label} must be numeric") from exc
    if not np.isfinite(number):
        raise CalibrationArtifactError(f"{label} must be finite")
    return number


def _apply_block(
    mu_diff: float | np.ndarray,
    block: dict,
    *,
    label: str,
) -> np.ndarray:
    a = _finite_number(block.get("intercept"), f"{label}.intercept")
    b = _finite_number(block.get("coef"), f"{label}.coef")
    z = a + b * (np.asarray(mu_diff, dtype=float) / 400.0)
    return np.asarray(_sigmoid(z), dtype=float)


def validate_calibration_artifact(cal: dict) -> dict:
    """Require a complete, explicitly time-held-out calibration artifact."""

    if not isinstance(cal, dict):
        raise CalibrationArtifactError("calibration artifact must be an object")
    if cal.get("status") != "validated_time_holdout":
        raise CalibrationArtifactError(
            "calibration artifact is not marked validated_time_holdout"
        )
    split = cal.get("time_split")
    if not isinstance(split, dict):
        raise CalibrationArtifactError("calibration artifact lacks time_split")
    for key in ("train_end", "holdout_start"):
        if not split.get(key):
            raise CalibrationArtifactError(f"time_split.{key} is required")

    for key in ("team", "player"):
        block = cal.get(key)
        if not isinstance(block, dict):
            raise CalibrationArtifactError(
                f"calibration artifact is missing the {key} block"
            )
        _finite_number(block.get("intercept"), f"{key}.intercept")
        _finite_number(block.get("coef"), f"{key}.coef")
        if block.get("fit_split") != "train":
            raise CalibrationArtifactError(f"{key}.fit_split must be train")
        holdout = block.get("holdout")
        if not isinstance(holdout, dict):
            raise CalibrationArtifactError(f"{key}.holdout is required")
        if int(block.get("n_train") or 0) < MIN_TRAIN_ROWS:
            raise CalibrationArtifactError(
                f"{key} block has fewer than {MIN_TRAIN_ROWS} train rows"
            )
        if int(holdout.get("n") or 0) < MIN_HOLDOUT_ROWS:
            raise CalibrationArtifactError(
                f"{key} block has fewer than {MIN_HOLDOUT_ROWS} holdout rows"
            )

    blend = cal.get("strength_blend")
    if not isinstance(blend, dict):
        raise CalibrationArtifactError(
            "calibration artifact is missing strength_blend"
        )
    for field in ("intercept", "coef_team", "coef_player"):
        _finite_number(blend.get(field), f"strength_blend.{field}")
    if blend.get("fit_split") != "train":
        raise CalibrationArtifactError(
            "strength_blend.fit_split must be train"
        )
    if int(blend.get("n_train") or 0) < MIN_TRAIN_ROWS:
        raise CalibrationArtifactError(
            f"strength_blend has fewer than {MIN_TRAIN_ROWS} train rows"
        )
    if int(blend.get("n_holdout") or 0) < MIN_HOLDOUT_ROWS:
        raise CalibrationArtifactError(
            f"strength_blend has fewer than {MIN_HOLDOUT_ROWS} holdout rows"
        )
    return cal


def apply_scale(mu_diff: float | np.ndarray, cal: dict, key: str = "player") -> np.ndarray:
    """Calibrated P(blue); missing blocks are an error, never a default."""

    validate_calibration_artifact(cal)
    block = cal.get(key)
    if not isinstance(block, dict):
        raise CalibrationArtifactError(
            f"calibration artifact is missing the {key} block"
        )
    return _apply_block(mu_diff, block, label=key)


def classic_elo_p(mu_diff: float | np.ndarray) -> np.ndarray:
    d = np.asarray(mu_diff, dtype=float)
    return 1.0 / (1.0 + 10 ** (-d / 400.0))


def _fit_one(diff: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Fit on provided train arrays; return params + in-sample diagnostics."""
    x = (diff / 400.0).reshape(-1, 1)
    # Strong L2 so we don't chase noise in small bins
    lr = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        try:
            lr.fit(x, y)
        except (ConvergenceWarning, RuntimeWarning) as exc:
            raise CalibrationArtifactError(
                f"{name} calibration optimizer emitted {type(exc).__name__}"
            ) from exc
    if (
        not np.isfinite(lr.intercept_).all()
        or not np.isfinite(lr.coef_).all()
        or np.max(lr.n_iter_) >= lr.max_iter
    ):
        raise CalibrationArtifactError(
            f"{name} calibration optimizer did not converge to finite coefficients"
        )
    p = np.clip(lr.predict_proba(x)[:, 1], 1e-4, 1 - 1e-4)
    classic = np.clip(classic_elo_p(diff), 1e-4, 1 - 1e-4)
    return {
        "name": name,
        "intercept": float(lr.intercept_[0]),
        "coef": float(lr.coef_[0][0]),
        # Classic Elo uses ln(10)≈2.302 on (diff/400) in logit space
        "coef_vs_classic": float(lr.coef_[0][0] / np.log(10)),
        "temperature_400": float(np.log(10) * 400.0 / lr.coef_[0][0]) if abs(lr.coef_[0][0]) > 1e-6 else None,
        "train_brier": float(brier_score_loss(y, p)),
        "train_brier_classic": float(brier_score_loss(y, classic)),
        "train_auc": float(roc_auc_score(y, p)),
        "n_train": len(y),
        "fit_split": "train",
    }


def _holdout_metrics(diff: np.ndarray, y: np.ndarray, block: dict) -> dict:
    p = _apply_block(diff, block, label=str(block["name"]))
    p = np.clip(p, 1e-4, 1 - 1e-4)
    classic = np.clip(classic_elo_p(diff), 1e-4, 1 - 1e-4)
    return {
        "n": len(y),
        "brier_cal": float(brier_score_loss(y, p)),
        "brier_classic": float(brier_score_loss(y, classic)),
        "auc_cal": float(roc_auc_score(y, p)),
        "auc_classic": float(roc_auc_score(y, classic)),
        "log_loss_cal": float(log_loss(y, p)),
    }


def fit_elo_wr_calibration(df: pd.DataFrame | None = None) -> dict:
    """
    Time-safe calibration: fit logistic Elo→WR on train cut only, score holdout.
    """
    if df is None:
        maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
        feat = pd.read_parquet(FEATURES_DIR / "maps.parquet")
        pr_path = FEATURES_DIR / "player_ratings.parquet"
        maps["game_uid"] = maps["game_uid"].astype(str)
        feat["game_uid"] = feat["game_uid"].astype(str)
        fcols = [c for c in feat.columns if c == "game_uid" or c not in maps.columns]
        df = maps.merge(feat[fcols], on="game_uid", how="inner")
        if pr_path.exists():
            pr = pd.read_parquet(pr_path)
            pr["game_uid"] = pr["game_uid"].astype(str)
            drop = [c for c in df.columns if c.startswith("player_") or c == "p_player_elo"]
            df = df.drop(columns=drop, errors="ignore")
            df = df.merge(
                pr[["game_uid", "player_mu_diff", "p_player_elo"]],
                on="game_uid",
                how="left",
            )
    required = {"y_blue_win", "date", "mu_diff", "player_mu_diff"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise CalibrationArtifactError(
            f"calibration input is missing required columns: {missing}"
        )
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["y"] = pd.to_numeric(df["y_blue_win"], errors="coerce")
    for column in ("mu_diff", "player_mu_diff"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[~np.isfinite(df[column]), column] = np.nan
    df = (
        df.dropna(subset=["y", "date"])
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    if df.empty or not df["y"].isin([0.0, 1.0]).all():
        raise CalibrationArtifactError(
            "calibration outcomes must be non-empty and binary"
        )
    tr_idx, te_idx = holdout_cut(df["date"], frac=0.85)
    if len(tr_idx) == 0 or len(te_idx) == 0:
        raise CalibrationArtifactError(
            "calibration requires non-empty train and future holdout splits"
        )
    train_end = pd.Timestamp(df.loc[tr_idx, "date"].max())
    te_idx = np.asarray(
        [
            index
            for index in te_idx
            if pd.Timestamp(df.loc[index, "date"]) > train_end
        ],
        dtype=int,
    )
    if len(te_idx) == 0:
        raise CalibrationArtifactError(
            "calibration holdout must begin strictly after the train period"
        )
    holdout_start = pd.Timestamp(df.loc[te_idx, "date"].min())

    populations: dict[str, dict[str, np.ndarray]] = {}
    population_errors: list[str] = []
    for key, columns in {
        "team": ("mu_diff",),
        "player": ("player_mu_diff",),
        "strength_blend": ("mu_diff", "player_mu_diff"),
    }.items():
        mask = df[list(columns)].notna().all(axis=1)
        train_mask = mask & df.index.isin(tr_idx)
        holdout_mask = mask & df.index.isin(te_idx)
        train_rows = df.loc[train_mask]
        holdout_rows = df.loc[holdout_mask]
        if len(train_rows) < MIN_TRAIN_ROWS:
            population_errors.append(
                f"{key} train rows {len(train_rows)} < {MIN_TRAIN_ROWS}"
            )
        if len(holdout_rows) < MIN_HOLDOUT_ROWS:
            population_errors.append(
                f"{key} holdout rows {len(holdout_rows)} < {MIN_HOLDOUT_ROWS}"
            )
        if train_rows["y"].nunique() < 2:
            population_errors.append(f"{key} train split has one outcome class")
        if holdout_rows["y"].nunique() < 2:
            population_errors.append(
                f"{key} holdout split has one outcome class"
            )
        populations[key] = {
            "train_index": train_rows.index.to_numpy(dtype=int),
            "holdout_index": holdout_rows.index.to_numpy(dtype=int),
        }
    if population_errors:
        raise CalibrationArtifactError(
            "incomplete calibration population; refusing to write a partial "
            f"artifact: {'; '.join(population_errors)}"
        )

    art: dict = {
        "version": 2,
        "status": "validated_time_holdout",
        "method": "logit = a + b*(mu_diff/400); C=0.5 L2; fit on time holdout train only",
        "time_split": {
            "train_end": train_end.isoformat(),
            "holdout_start": holdout_start.isoformat(),
            "strictly_future_holdout": True,
        },
        "note": (
            "Team and player mappings plus their strength blend are all fitted "
            "on the chronological train split and evaluated on a strictly "
            "later holdout. Partial artifacts are rejected."
        ),
    }

    for key, col in (("team", "mu_diff"), ("player", "player_mu_diff")):
        population = populations[key]
        train_rows = df.loc[population["train_index"]]
        holdout_rows = df.loc[population["holdout_index"]]
        dtr = train_rows[col].to_numpy(dtype=float)
        ytr = train_rows["y"].to_numpy(dtype=float)
        dte = holdout_rows[col].to_numpy(dtype=float)
        yte = holdout_rows["y"].to_numpy(dtype=float)
        block = _fit_one(dtr, ytr, key)
        block["holdout"] = _holdout_metrics(dte, yte, block)
        # empirical check around |Δ|~45
        band = (np.abs(dte) >= 40) & (np.abs(dte) < 50)
        if band.sum() >= 40:
            fav = np.where(dte[band] > 0, yte[band], 1 - yte[band])
            p_cal = _apply_block(
                np.abs(dte[band]), block, label=key
            )
            p_cl = classic_elo_p(np.abs(dte[band]))
            block["holdout_band_40_50"] = {
                "n": int(band.sum()),
                "fav_wr_actual": float(np.mean(fav)),
                "fav_p_cal_mean": float(np.mean(p_cal)),
                "fav_p_classic_mean": float(np.mean(p_cl)),
            }
        art[key] = block
        temperature = block["temperature_400"]
        temperature_text = (
            f"{temperature:.1f}" if temperature is not None else "undefined"
        )
        print(
            f"[elo_cal] {key}: b={block['coef']:.3f} (classic ln10={np.log(10):.3f}) "
            f"T400≈{temperature_text}  "
            f"holdout brier cal={block['holdout']['brier_cal']:.4f} "
            f"classic={block['holdout']['brier_classic']:.4f}"
        )

    # Blend weights on train only: logistic on [p_team_cal, p_player_cal]
    blend_population = populations["strength_blend"]
    blend_train = df.loc[blend_population["train_index"]]
    blend_holdout = df.loc[blend_population["holdout_index"]]
    p_team_tr = _apply_block(
        blend_train["mu_diff"].values, art["team"], label="team"
    )
    p_pl_tr = _apply_block(
        blend_train["player_mu_diff"].values,
        art["player"],
        label="player",
    )
    p_team_te = _apply_block(
        blend_holdout["mu_diff"].values, art["team"], label="team"
    )
    p_pl_te = _apply_block(
        blend_holdout["player_mu_diff"].values,
        art["player"],
        label="player",
    )
    ytr = blend_train["y"].to_numpy(dtype=float)
    yte = blend_holdout["y"].to_numpy(dtype=float)
    blend = LogisticRegression(
        C=0.5, max_iter=2000, solver="liblinear"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        warnings.simplefilter("error", RuntimeWarning)
        try:
            blend.fit(np.column_stack([p_team_tr, p_pl_tr]), ytr)
        except (ConvergenceWarning, RuntimeWarning) as exc:
            raise CalibrationArtifactError(
                "strength_blend optimizer emitted "
                f"{type(exc).__name__}"
            ) from exc
    if (
        not np.isfinite(blend.intercept_).all()
        or not np.isfinite(blend.coef_).all()
        or np.max(blend.n_iter_) >= blend.max_iter
    ):
        raise CalibrationArtifactError(
            "strength_blend optimizer did not converge to finite coefficients"
        )
    p_te = blend.predict_proba(
        np.column_stack([p_team_te, p_pl_te])
    )[:, 1]
    p_6040 = 0.6 * p_team_te + 0.4 * p_pl_te
    art["strength_blend"] = {
        "intercept": float(blend.intercept_[0]),
        "coef_team": float(blend.coef_[0][0]),
        "coef_player": float(blend.coef_[0][1]),
        "fit_split": "train",
        "holdout_brier": float(
            brier_score_loss(yte, np.clip(p_te, 1e-4, 1 - 1e-4))
        ),
        "holdout_brier_60_40": float(
            brier_score_loss(yte, np.clip(p_6040, 1e-4, 1 - 1e-4))
        ),
        "holdout_brier_team": float(
            brier_score_loss(yte, np.clip(p_team_te, 1e-4, 1 - 1e-4))
        ),
        "holdout_brier_player": float(
            brier_score_loss(yte, np.clip(p_pl_te, 1e-4, 1 - 1e-4))
        ),
        "holdout_auc": float(roc_auc_score(yte, p_te)),
        "n_train": len(blend_train),
        "n_holdout": len(blend_holdout),
    }
    print(
        f"[elo_cal] blend holdout brier={art['strength_blend']['holdout_brier']:.4f} "
        f"(team={art['strength_blend']['holdout_brier_team']:.4f} "
        f"player={art['strength_blend']['holdout_brier_player']:.4f})"
    )

    validate_calibration_artifact(art)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2), encoding="utf-8")
    print(f"[elo_cal] wrote {OUT}")
    return art


def load_calibration(*, required: bool = False) -> dict:
    if not OUT.exists():
        if required:
            raise CalibrationArtifactError(
                f"calibration artifact does not exist: {OUT}"
            )
        return {}
    try:
        artifact = json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationArtifactError(
            f"calibration artifact is unreadable: {OUT}"
        ) from exc
    return validate_calibration_artifact(artifact)


def calibrated_player_p(mu_diff: float, cal: dict | None = None) -> float:
    cal = cal if cal is not None else load_calibration()
    return float(apply_scale(mu_diff, cal, "player"))


def calibrated_team_p(mu_diff: float, cal: dict | None = None) -> float:
    cal = cal if cal is not None else load_calibration()
    return float(apply_scale(mu_diff, cal, "team"))


def calibrated_strength_p(
    team_mu_diff: float,
    player_mu_diff: float,
    cal: dict | None = None,
) -> dict:
    cal = cal if cal is not None else load_calibration()
    validate_calibration_artifact(cal)
    p_t = calibrated_team_p(team_mu_diff, cal)
    p_p = calibrated_player_p(player_mu_diff, cal)
    blend = cal["strength_blend"]
    z = (
        float(blend["intercept"])
        + float(blend["coef_team"]) * p_t
        + float(blend["coef_player"]) * p_p
    )
    p_b = float(_sigmoid(z))
    return {
        "p_team_cal": round(p_t, 4),
        "p_player_cal": round(p_p, 4),
        "p_strength_blend": round(p_b, 4),
    }


def apply_calibration_to_features(feat_path: Path | None = None) -> pd.DataFrame:
    """Rewrite p_player_elo / p_dual-style cols on features maps using calibration."""
    cal = load_calibration(required=True)
    path = feat_path or (FEATURES_DIR / "maps.parquet")
    df = pd.read_parquet(path)
    required_columns = {"mu_diff", "player_mu_diff"}
    missing = sorted(required_columns - set(df.columns))
    if missing:
        raise CalibrationArtifactError(
            f"feature frame is missing calibration inputs: {missing}"
        )

    team_diff = pd.to_numeric(df["mu_diff"], errors="coerce")
    player_diff = pd.to_numeric(df["player_mu_diff"], errors="coerce")
    team_valid = team_diff.notna() & np.isfinite(team_diff)
    player_valid = player_diff.notna() & np.isfinite(player_diff)
    joint_valid = team_valid & player_valid

    df["p_player_elo_raw"] = df.get("p_player_elo", np.nan)
    df["p_dual_elo_raw"] = df.get("p_dual_elo", np.nan)
    df["p_player_elo"] = np.nan
    df["p_dual_elo_cal"] = np.nan
    df["p_dual_elo"] = np.nan
    df["p_strength_blend"] = np.nan

    df.loc[player_valid, "p_player_elo"] = apply_scale(
        player_diff[player_valid].to_numpy(dtype=float), cal, "player"
    )
    team_probability = apply_scale(
        team_diff[team_valid].to_numpy(dtype=float), cal, "team"
    )
    df.loc[team_valid, "p_dual_elo_cal"] = team_probability
    df.loc[team_valid, "p_dual_elo"] = team_probability

    if joint_valid.any():
        p_team = apply_scale(
            team_diff[joint_valid].to_numpy(dtype=float), cal, "team"
        )
        p_player = apply_scale(
            player_diff[joint_valid].to_numpy(dtype=float), cal, "player"
        )
        blend = cal["strength_blend"]
        blend_logit = (
            float(blend["intercept"])
            + float(blend["coef_team"]) * p_team
            + float(blend["coef_player"]) * p_player
        )
        df.loc[joint_valid, "p_strength_blend"] = _sigmoid(blend_logit)
    df.to_parquet(path, index=False)
    print(f"[elo_cal] applied calibration → {path}")
    return df


def main() -> None:
    fit_elo_wr_calibration()
    apply_calibration_to_features()


if __name__ == "__main__":
    main()
