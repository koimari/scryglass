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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR
from lol_kills.ml.eval import holdout_cut

OUT = MODELS_DIR / "elo_wr_calibration.json"


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def apply_scale(mu_diff: float | np.ndarray, cal: dict, key: str = "player") -> np.ndarray:
    """Calibrated P(blue) from μ_diff using saved logistic."""
    block = cal.get(key) or {}
    a = float(block.get("intercept", 0.0))
    b = float(block.get("coef", np.log(10)))  # default ≈ classic Elo logit scale
    z = a + b * (np.asarray(mu_diff, dtype=float) / 400.0)
    return np.asarray(_sigmoid(z), dtype=float)


def classic_elo_p(mu_diff: float | np.ndarray) -> np.ndarray:
    d = np.asarray(mu_diff, dtype=float)
    return 1.0 / (1.0 + 10 ** (-d / 400.0))


def _fit_one(diff: np.ndarray, y: np.ndarray, name: str) -> dict:
    """Fit on provided train arrays; return params + in-sample diagnostics."""
    x = (diff / 400.0).reshape(-1, 1)
    # Strong L2 so we don't chase noise in small bins
    lr = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
    lr.fit(x, y)
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
        "n_train": int(len(y)),
    }


def _holdout_metrics(diff: np.ndarray, y: np.ndarray, block: dict) -> dict:
    p = apply_scale(diff, {block["name"]: block}, key=block["name"])
    p = np.clip(p, 1e-4, 1 - 1e-4)
    classic = np.clip(classic_elo_p(diff), 1e-4, 1 - 1e-4)
    return {
        "n": int(len(y)),
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
    df = df.dropna(subset=["y_blue_win", "date"]).sort_values("date").reset_index(drop=True)
    df["y"] = df["y_blue_win"].astype(float)
    tr_idx, te_idx = holdout_cut(df["date"], frac=0.85)

    art: dict = {
        "version": 1,
        "method": "logit = a + b*(mu_diff/400); C=0.5 L2; fit on time holdout train only",
        "note": "Player μ gaps understate classic 400 WR; logistic b>ln(10) maps them to empirical fav WR. Fit on time-train only (C=0.5 L2).",
    }

    for key, col in (("team", "mu_diff"), ("player", "player_mu_diff")):
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col]).copy()
        # remap indices
        tr = sub.index.isin(df.index[tr_idx])
        te = sub.index.isin(df.index[te_idx])
        dtr = sub.loc[tr, col].astype(float).values
        ytr = sub.loc[tr, "y"].values
        dte = sub.loc[te, col].astype(float).values
        yte = sub.loc[te, "y"].values
        if len(dtr) < 400 or len(dte) < 100:
            continue
        block = _fit_one(dtr, ytr, key)
        block["holdout"] = _holdout_metrics(dte, yte, block)
        # empirical check around |Δ|~45
        band = (np.abs(dte) >= 40) & (np.abs(dte) < 50)
        if band.sum() >= 40:
            fav = np.where(dte[band] > 0, yte[band], 1 - yte[band])
            p_cal = apply_scale(np.abs(dte[band]), {key: block}, key=key)
            p_cl = classic_elo_p(np.abs(dte[band]))
            block["holdout_band_40_50"] = {
                "n": int(band.sum()),
                "fav_wr_actual": float(np.mean(fav)),
                "fav_p_cal_mean": float(np.mean(p_cal)),
                "fav_p_classic_mean": float(np.mean(p_cl)),
            }
        art[key] = block
        print(
            f"[elo_cal] {key}: b={block['coef']:.3f} (classic ln10={np.log(10):.3f}) "
            f"T400≈{block['temperature_400']:.1f}  "
            f"holdout brier cal={block['holdout']['brier_cal']:.4f} "
            f"classic={block['holdout']['brier_classic']:.4f}"
        )

    # Blend weights on train only: logistic on [p_team_cal, p_player_cal]
    if "team" in art and "player" in art:
        sub = df.dropna(subset=["mu_diff", "player_mu_diff"]).copy()
        tr = sub.index.isin(df.index[tr_idx])
        te = sub.index.isin(df.index[te_idx])
        p_team_tr = apply_scale(sub.loc[tr, "mu_diff"].values, art, "team")
        p_pl_tr = apply_scale(sub.loc[tr, "player_mu_diff"].values, art, "player")
        p_team_te = apply_scale(sub.loc[te, "mu_diff"].values, art, "team")
        p_pl_te = apply_scale(sub.loc[te, "player_mu_diff"].values, art, "player")
        ytr = sub.loc[tr, "y"].values
        yte = sub.loc[te, "y"].values
        blend = LogisticRegression(C=0.5, max_iter=2000)
        blend.fit(np.column_stack([p_team_tr, p_pl_tr]), ytr)
        p_te = blend.predict_proba(np.column_stack([p_team_te, p_pl_te]))[:, 1]
        # also fixed 60/40 reference
        p_6040 = 0.6 * p_team_te + 0.4 * p_pl_te
        art["strength_blend"] = {
            "intercept": float(blend.intercept_[0]),
            "coef_team": float(blend.coef_[0][0]),
            "coef_player": float(blend.coef_[0][1]),
            "holdout_brier": float(brier_score_loss(yte, np.clip(p_te, 1e-4, 1 - 1e-4))),
            "holdout_brier_60_40": float(brier_score_loss(yte, np.clip(p_6040, 1e-4, 1 - 1e-4))),
            "holdout_brier_team": float(brier_score_loss(yte, np.clip(p_team_te, 1e-4, 1 - 1e-4))),
            "holdout_brier_player": float(brier_score_loss(yte, np.clip(p_pl_te, 1e-4, 1 - 1e-4))),
            "holdout_auc": float(roc_auc_score(yte, p_te)),
            "n_train": int(tr.sum()),
            "n_holdout": int(te.sum()),
        }
        print(f"[elo_cal] blend holdout brier={art['strength_blend']['holdout_brier']:.4f} "
              f"(team={art['strength_blend']['holdout_brier_team']:.4f} "
              f"player={art['strength_blend']['holdout_brier_player']:.4f})")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2))
    print(f"[elo_cal] wrote {OUT}")
    return art


def load_calibration() -> dict:
    if not OUT.exists():
        return {}
    try:
        return json.loads(OUT.read_text())
    except Exception:
        return {}


def calibrated_player_p(mu_diff: float, cal: dict | None = None) -> float:
    cal = cal if cal is not None else load_calibration()
    if not cal.get("player"):
        return float(classic_elo_p(mu_diff))
    return float(apply_scale(mu_diff, cal, "player"))


def calibrated_team_p(mu_diff: float, cal: dict | None = None) -> float:
    cal = cal if cal is not None else load_calibration()
    if not cal.get("team"):
        return float(classic_elo_p(mu_diff))
    return float(apply_scale(mu_diff, cal, "team"))


def calibrated_strength_p(
    team_mu_diff: float,
    player_mu_diff: float,
    cal: dict | None = None,
) -> dict:
    cal = cal if cal is not None else load_calibration()
    p_t = calibrated_team_p(team_mu_diff, cal)
    p_p = calibrated_player_p(player_mu_diff, cal)
    blend = cal.get("strength_blend") or {}
    if blend.get("coef_team") is not None:
        z = (
            float(blend["intercept"])
            + float(blend["coef_team"]) * p_t
            + float(blend["coef_player"]) * p_p
        )
        p_b = float(_sigmoid(z))
    else:
        p_b = 0.6 * p_t + 0.4 * p_p
    return {
        "p_team_cal": round(p_t, 4),
        "p_player_cal": round(p_p, 4),
        "p_strength_blend": round(p_b, 4),
    }


def apply_calibration_to_features(feat_path: Path | None = None) -> pd.DataFrame:
    """Rewrite p_player_elo / p_dual-style cols on features maps using calibration."""
    cal = load_calibration()
    if not cal:
        cal = fit_elo_wr_calibration()
    path = feat_path or (FEATURES_DIR / "maps.parquet")
    df = pd.read_parquet(path)
    if "player_mu_diff" in df.columns and cal.get("player"):
        df["p_player_elo_raw"] = df.get("p_player_elo", np.nan)
        df["p_player_elo"] = apply_scale(df["player_mu_diff"].fillna(0.0).values, cal, "player")
    if "mu_diff" in df.columns and cal.get("team"):
        df["p_dual_elo_raw"] = df.get("p_dual_elo", np.nan)
        p_team = apply_scale(df["mu_diff"].fillna(0.0).values, cal, "team")
        df["p_dual_elo_cal"] = p_team
        df["p_dual_elo"] = p_team  # stack/GBM see empirically calibrated team prior
    if "mu_diff" in df.columns and "player_mu_diff" in df.columns:
        scored = [
            calibrated_strength_p(float(t), float(p), cal)
            for t, p in zip(
                df["mu_diff"].fillna(0.0),
                df["player_mu_diff"].fillna(0.0),
            )
        ]
        df["p_strength_blend"] = [s["p_strength_blend"] for s in scored]
    elif "p_dual_elo" in df.columns and "p_player_elo" in df.columns:
        df["p_strength_blend"] = 0.60 * df["p_dual_elo"].fillna(0.5) + 0.40 * df["p_player_elo"].fillna(0.5)
    df.to_parquet(path, index=False)
    print(f"[elo_cal] applied calibration → {path}")
    return df


def main() -> None:
    fit_elo_wr_calibration()
    apply_calibration_to_features()


if __name__ == "__main__":
    main()
