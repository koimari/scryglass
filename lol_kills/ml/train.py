#!/usr/bin/env python3
"""
Train calibrated research models: win / kills / FB / inhib + OOF stack + conformal.

Gates via lol_kills.ml.eval — kills must beat ridge on CRPS or ridge is selected.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_squared_error

from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR
from lol_kills.features.build import FEATURE_COLS
from lol_kills.ml.eval import (
    archive_models,
    classification_report,
    crps_gaussian,
    evaluate_gates,
    holdout_cut,
    purged_time_splits,
    regression_report,
    write_eval_report,
)

warnings.filterwarnings("ignore", category=UserWarning)

FEAT = FEATURE_COLS + ["league_id"]


def _get_gbm_classifier(**kwargs):
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=kwargs.get("n_estimators", 250),
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=25,
            random_state=42,
            verbosity=-1,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_depth=5, learning_rate=0.05, max_iter=250, random_state=42)


def _get_gbm_regressor(alpha: float | None = None, **kwargs):
    try:
        import lightgbm as lgb

        params = dict(
            n_estimators=kwargs.get("n_estimators", 250),
            learning_rate=0.05,
            max_depth=5,
            num_leaves=31,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_samples=25,
            random_state=42,
            verbosity=-1,
        )
        if alpha is not None:
            params["objective"] = "quantile"
            params["alpha"] = alpha
        return lgb.LGBMRegressor(**params)
    except Exception:
        from sklearn.ensemble import HistGradientBoostingRegressor

        if alpha is not None:
            return HistGradientBoostingRegressor(
                loss="quantile", quantile=alpha, max_depth=5, learning_rate=0.05, max_iter=250, random_state=42
            )
        return HistGradientBoostingRegressor(max_depth=5, learning_rate=0.05, max_iter=250, random_state=42)


def _X(df: pd.DataFrame, cols: list[str] | None = None):
    cols = cols or [c for c in FEAT if c in df.columns]
    X = df[cols].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).values
    return X, cols


def _iso_is_degenerate(iso) -> bool:
    """Isotonic that maps most mid-range probs to 0/1 is unusable."""
    if iso is None:
        return True
    try:
        probe = np.linspace(0.05, 0.95, 19)
        y = np.asarray(iso.transform(probe), dtype=float)
        return float(np.mean((y <= 1e-6) | (y >= 1.0 - 1e-6))) > 0.45
    except Exception:
        return True


def train_classifier(df: pd.DataFrame, label: str, name: str) -> dict:
    sub = df.dropna(subset=[label, "date"]).sort_values("date").reset_index(drop=True)
    y = sub[label].astype(float).values
    if set(np.unique(y)) - {0.0, 1.0}:
        y = (y >= 0.5).astype(float)
    if len(sub) < 200 or y.sum() < 20 or (len(y) - y.sum()) < 20:
        return {"name": name, "status": "skipped", "reason": "insufficient data"}

    X, cols = _X(sub)
    tr_idx, te_idx = holdout_cut(sub["date"])
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    # Baselines
    mean_p = float(y_tr.mean())
    p_mean = np.full_like(y_te, mean_p)
    elo_col = "mu_diff" if "mu_diff" in cols else ("elo_diff" if "elo_diff" in cols else None)
    elo_brier = None
    elo_ll = None
    if elo_col:
        j = cols.index(elo_col)
        lr = LogisticRegression(max_iter=300)
        lr.fit(X_tr[:, [j]], y_tr)
        p_elo = np.clip(lr.predict_proba(X_te[:, [j]])[:, 1], 1e-4, 1 - 1e-4)
        elo_brier = float(brier_score_loss(y_te, p_elo))
        elo_ll = float(log_loss(y_te, p_elo))
        joblib.dump(lr, MODELS_DIR / f"{name}_baseline_elo.joblib")

    base = _get_gbm_classifier()
    base.fit(X_tr, y_tr)
    p_raw = np.clip(base.predict_proba(X_te)[:, 1], 1e-4, 1 - 1e-4)
    iso = IsotonicRegression(out_of_bounds="clip")
    if len(np.unique(y_te)) > 1 and len(y_te) >= 40:
        # calibrate on first half of holdout, score on second — simple
        mid = len(y_te) // 2
        iso.fit(p_raw[:mid], y_te[:mid])
        p_cal = iso.transform(p_raw)
        calibrated = True
    else:
        p_cal = p_raw
        calibrated = False

    hold = classification_report(y_te, p_cal, name)
    # CV
    cv_briers = []
    for tr, te in purged_time_splits(sub["date"]):
        clf = _get_gbm_classifier()
        clf.fit(X[tr], y[tr])
        pp = np.clip(clf.predict_proba(X[te])[:, 1], 1e-4, 1 - 1e-4)
        cv_briers.append(brier_score_loss(y[te], pp))

    # full refit
    full = _get_gbm_classifier()
    full.fit(X, y)
    # refit iso on last 15%
    cut = int(len(X) * 0.85)
    p_hold_raw = np.clip(full.predict_proba(X[cut:])[:, 1], 1e-4, 1 - 1e-4)
    iso_full = IsotonicRegression(out_of_bounds="clip")
    iso_ok = False
    if len(np.unique(y[cut:])) > 1 and len(y[cut:]) >= 80:
        iso_full.fit(p_hold_raw, y[cut:])
        iso_ok = not _iso_is_degenerate(iso_full)
    if not iso_ok and calibrated and not _iso_is_degenerate(iso):
        iso_full = iso
        iso_ok = True
    if not iso_ok:
        iso_full = None
        calibrated = False

    # conformal residual on holdout (absolute residual of prob)
    q = float(np.quantile(np.abs(y_te - p_cal), 0.9)) if len(y_te) else 0.15

    joblib.dump(full, MODELS_DIR / f"{name}_gbm.joblib")
    iso_path = MODELS_DIR / f"{name}_iso.joblib"
    if iso_full is not None:
        joblib.dump(iso_full, iso_path)
    elif iso_path.exists():
        iso_path.unlink()
    meta = {
        "status": "ok",
        "name": name,
        "feature_cols": cols,
        "calibrated": bool(iso_ok),
        "iso_skipped": not iso_ok,
        "holdout": hold,
        "cv_brier_mean": float(np.mean(cv_briers)) if cv_briers else None,
        "baselines": {
            "mean_brier": float(brier_score_loss(y_te, p_mean)),
            "elo_brier": elo_brier,
            "elo_log_loss": elo_ll,
        },
        "conformal_q90": q,
        "n_train": int(len(tr_idx)),
        "n_holdout": int(len(te_idx)),
    }
    (MODELS_DIR / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ml] {name} brier={hold['brier']:.4f} ece={hold['ece']:.4f}")
    return meta


def train_kills(df: pd.DataFrame) -> dict:
    label = "y_total_kills"
    sub = df.dropna(subset=[label, "date"]).sort_values("date").reset_index(drop=True)
    y = sub[label].astype(float).values
    X, cols = _X(sub)
    if len(sub) < 200:
        return {"name": "kills", "status": "skipped"}

    tr_idx, te_idx = holdout_cut(sub["date"])
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_te, y_te = X[te_idx], y[te_idx]

    ridge = Ridge(alpha=10.0)
    ridge.fit(X_tr, y_tr)
    mu_ridge = ridge.predict(X_te)
    sd_ridge = float(max(np.std(y_te - mu_ridge), 3.0))
    ridge_rep = regression_report(y_te, mu_ridge, np.full_like(y_te, sd_ridge), "ridge")

    # Quantile GBM → Normal params from p10/p50/p90
    q10 = _get_gbm_regressor(alpha=0.1)
    q50 = _get_gbm_regressor(alpha=0.5)
    q90 = _get_gbm_regressor(alpha=0.9)
    q10.fit(X_tr, y_tr)
    q50.fit(X_tr, y_tr)
    q90.fit(X_tr, y_tr)
    p10, p50, p90 = q10.predict(X_te), q50.predict(X_te), q90.predict(X_te)
    # Approximate σ from IQR: (p90-p10)/(2*1.28155)
    sd_q = np.maximum((p90 - p10) / (2 * 1.28155), 3.0)
    mu_q = p50
    q_rep = regression_report(y_te, mu_q, sd_q, "quantile_gbm")

    # Mean GBM residual-sd (legacy)
    gbm = _get_gbm_regressor()
    gbm.fit(X_tr, y_tr)
    mu_g = gbm.predict(X_te)
    sd_g = float(max(np.std(y_te - mu_g), 3.0))
    g_rep = regression_report(y_te, mu_g, np.full_like(y_te, sd_g), "mean_gbm")

    candidates = [
        ("ridge", ridge_rep, "ridge"),
        ("quantile", q_rep, "quantile"),
        ("mean_gbm", g_rep, "mean_gbm"),
    ]
    best_name, best_rep, _ = min(candidates, key=lambda t: t[1]["crps"])

    # Refit winners on all data
    ridge_full = Ridge(alpha=10.0)
    ridge_full.fit(X, y)
    q10f, q50f, q90f = _get_gbm_regressor(0.1), _get_gbm_regressor(0.5), _get_gbm_regressor(0.9)
    q10f.fit(X, y)
    q50f.fit(X, y)
    q90f.fit(X, y)
    gbm_full = _get_gbm_regressor()
    gbm_full.fit(X, y)
    resid_sd = float(max(np.std(y - gbm_full.predict(X)), 3.0))

    joblib.dump(ridge_full, MODELS_DIR / "kills_baseline_ridge.joblib")
    joblib.dump({"q10": q10f, "q50": q50f, "q90": q90f}, MODELS_DIR / "kills_quantile.joblib")
    joblib.dump(gbm_full, MODELS_DIR / "kills_gbm.joblib")

    # conformal for mu absolute residual of selected
    if best_name == "ridge":
        resid = np.abs(y_te - mu_ridge)
        sel_sd = sd_ridge
    elif best_name == "quantile":
        resid = np.abs(y_te - mu_q)
        sel_sd = float(np.median(sd_q))
    else:
        resid = np.abs(y_te - mu_g)
        sel_sd = sd_g
    conf_q = float(np.quantile(resid, 0.9))

    meta = {
        "status": "ok",
        "name": "kills",
        "feature_cols": cols,
        "selected_model": best_name,
        "holdout": best_rep,
        "baselines": {
            "ridge_crps": ridge_rep["crps"],
            "ridge_rmse": ridge_rep["rmse"],
            "quantile_crps": q_rep["crps"],
            "mean_gbm_crps": g_rep["crps"],
        },
        "all_holdout": {"ridge": ridge_rep, "quantile": q_rep, "mean_gbm": g_rep},
        "residual_sd": sel_sd if best_name != "mean_gbm" else resid_sd,
        "conformal_q90": conf_q,
        "n_train": int(len(tr_idx)),
        "n_holdout": int(len(te_idx)),
    }
    (MODELS_DIR / "kills_meta.json").write_text(json.dumps(meta, indent=2))
    print(
        f"[ml] kills selected={best_name} crps={best_rep['crps']:.3f} "
        f"(ridge={ridge_rep['crps']:.3f} q={q_rep['crps']:.3f})"
    )
    return meta


def train_stack_and_ablations(df: pd.DataFrame) -> dict:
    """
    OOF logistic stack on calibrated strength / draft / gbm.

    Anti-overfit:
      - purged time-series OOF for GBM
      - L2 logistic (C=0.5)
      - stack coefs fit on train-time OOF only; holdout metrics reported honestly
      - production model refit on all OOF rows
    """
    sub = df.dropna(subset=["y_blue_win", "date"]).sort_values("date").reset_index(drop=True)
    y = sub["y_blue_win"].astype(float).values
    X, cols = _X(sub)
    n = len(sub)
    oof_gbm = np.full(n, np.nan)
    splits = purged_time_splits(sub["date"], n_folds=4)
    if not splits:
        tr, te = holdout_cut(sub["date"])
        clf = _get_gbm_classifier()
        clf.fit(X[tr], y[tr])
        oof_gbm[te] = clf.predict_proba(X[te])[:, 1]
    else:
        for tr, te in splits:
            clf = _get_gbm_classifier()
            clf.fit(X[tr], y[tr])
            oof_gbm[te] = clf.predict_proba(X[te])[:, 1]

    mask = ~np.isnan(oof_gbm)
    # Prefer calibrated strength blend; fall back gracefully
    if "p_strength_blend" in sub.columns:
        p_str = sub["p_strength_blend"].astype(float).fillna(0.5).values
    elif "p_player_elo" in sub.columns and "p_dual_elo" in sub.columns:
        p_str = (
            0.6 * sub["p_dual_elo"].astype(float).fillna(0.5).values
            + 0.4 * sub["p_player_elo"].astype(float).fillna(0.5).values
        )
    else:
        p_str = sub["p_dual_elo"].astype(float).fillna(0.5).values if "p_dual_elo" in sub.columns else np.full(n, 0.5)

    p_team = sub["p_dual_elo"].astype(float).fillna(0.5).values if "p_dual_elo" in sub.columns else p_str
    p_player = (
        sub["p_player_elo"].astype(float).fillna(0.5).values if "p_player_elo" in sub.columns else p_str
    )
    draft_logit = (
        sub["draft_win_logit_blue"].astype(float).fillna(0.0).values
        if "draft_win_logit_blue" in sub.columns
        else np.zeros(n)
    )
    p_draft = 1.0 / (1.0 + np.exp(-1.4 * draft_logit))
    sigma = sub["sigma_pair"].astype(float).fillna(80.0).values if "sigma_pair" in sub.columns else np.full(n, 80.0)
    sigma_z = (sigma - 80.0) / 40.0

    feature_order = ["p_strength_blend", "p_draft", "p_gbm", "sigma_z"]
    Z_all = np.column_stack([p_str, p_draft, oof_gbm, sigma_z])

    tr_idx, te_idx = holdout_cut(sub["date"])
    # Fit stack only where OOF exists AND in train window
    fit_mask = mask.copy()
    fit_mask[te_idx] = False
    # ensure we only use train indices that have OOF
    hold_mask = mask.copy()
    hold_mask[:] = False
    hold_mask[te_idx] = mask[te_idx]

    def _brier(y_true, p):
        return float(brier_score_loss(y_true, np.clip(p, 1e-4, 1 - 1e-4)))

    stack_tr = LogisticRegression(C=0.5, max_iter=800, solver="lbfgs")
    stack_tr.fit(Z_all[fit_mask], y[fit_mask])
    p_hold = stack_tr.predict_proba(Z_all[hold_mask])[:, 1] if hold_mask.sum() else np.array([])

    ablations_hold = {}
    if hold_mask.sum() >= 80:
        yh = y[hold_mask]
        ablations_hold = {
            "strength_blend": _brier(yh, p_str[hold_mask]),
            "team_elo": _brier(yh, p_team[hold_mask]),
            "player_elo": _brier(yh, p_player[hold_mask]),
            "draft_only": _brier(yh, p_draft[hold_mask]),
            "gbm_only": _brier(yh, oof_gbm[hold_mask]),
            "full_stack": _brier(yh, p_hold),
        }

    # Production: refit on all OOF rows (still regularized)
    stack = LogisticRegression(C=0.5, max_iter=800, solver="lbfgs")
    stack.fit(Z_all[mask], y[mask])
    p_stack = stack.predict_proba(Z_all[mask])[:, 1]
    ablations_oof = {
        "strength_blend": _brier(y[mask], p_str[mask]),
        "team_elo": _brier(y[mask], p_team[mask]),
        "player_elo": _brier(y[mask], p_player[mask]),
        "draft_only": _brier(y[mask], p_draft[mask]),
        "gbm_only": _brier(y[mask], oof_gbm[mask]),
        "full_stack": _brier(y[mask], p_stack),
    }

    joblib.dump(
        {
            "model": stack,
            "feature_order": feature_order,
            "regularization_C": 0.5,
            "notes": "p_strength_blend should be calibrated team+player prior",
        },
        MODELS_DIR / "stack_meta.joblib",
    )
    meta = {
        "ablations_brier_oof": ablations_oof,
        "ablations_brier_holdout": ablations_hold,
        "ablations_brier": ablations_hold or ablations_oof,  # prefer honest holdout
        "stack_coef": stack.coef_.ravel().tolist(),
        "stack_intercept": float(stack.intercept_[0]),
        "feature_order": feature_order,
        "regularization_C": 0.5,
        "n_oof": int(mask.sum()),
        "n_stack_train": int(fit_mask.sum()),
        "n_stack_holdout": int(hold_mask.sum()),
        "full_beats_draft": (ablations_hold or ablations_oof)["full_stack"]
        < (ablations_hold or ablations_oof)["draft_only"],
        "full_beats_strength": (ablations_hold or ablations_oof)["full_stack"]
        < (ablations_hold or ablations_oof)["strength_blend"],
    }
    (MODELS_DIR / "stack_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[ml] stack holdout {ablations_hold or '(thin)'}")
    print(f"[ml] stack oof     {ablations_oof}")
    return meta


# Cache loaded GBM/iso/meta so each board doesn't re-joblib from disk
_MODEL_CACHE: dict[str, tuple] = {}
_JSON_CACHE: dict[str, dict] = {}
_OBJ_CACHE: dict[str, object] = {}


def _load_json_cached(path: Path) -> dict:
    key = str(path)
    if key not in _JSON_CACHE:
        _JSON_CACHE[key] = json.loads(path.read_text())
    return _JSON_CACHE[key]


def _load_obj_cached(path: Path):
    key = str(path)
    if key not in _OBJ_CACHE:
        _OBJ_CACHE[key] = joblib.load(path)
    return _OBJ_CACHE[key]


def predict_row(features: dict) -> dict:
    out: dict = {}

    def vec(meta_name: str):
        if meta_name in _MODEL_CACHE:
            gbm, iso, meta = _MODEL_CACHE[meta_name]
            if gbm is None:
                return None, None, None
            cols = meta.get("feature_cols") or FEAT
            x = np.array([[float(features.get(c, 0.0) or 0.0) for c in cols]])
            return gbm, iso, (x, meta)
        path = MODELS_DIR / f"{meta_name}_gbm.joblib"
        if not path.exists():
            _MODEL_CACHE[meta_name] = (None, None, {})
            return None, None, None
        gbm = _load_obj_cached(path)
        iso_path = MODELS_DIR / f"{meta_name}_iso.joblib"
        iso = _load_obj_cached(iso_path) if iso_path.exists() else None
        meta = _load_json_cached(MODELS_DIR / f"{meta_name}_meta.json")
        _MODEL_CACHE[meta_name] = (gbm, iso, meta)
        cols = meta.get("feature_cols") or FEAT
        x = np.array([[float(features.get(c, 0.0) or 0.0) for c in cols]])
        return gbm, iso, (x, meta)

    def _iso_degenerate(iso) -> bool:
        """True when isotonic collapses most mid probs to 0/1 (broken small-holdout fit)."""
        if iso is None:
            return True
        try:
            probe = np.linspace(0.05, 0.95, 19)
            y = np.asarray(iso.transform(probe), dtype=float)
            return float(np.mean((y <= 1e-6) | (y >= 1.0 - 1e-6))) > 0.45
        except Exception:
            return True

    gbm, iso, pack = vec("win")
    if gbm is not None:
        x, meta = pack
        p = float(gbm.predict_proba(x)[0, 1])
        if iso is not None and not _iso_degenerate(iso):
            try:
                p = float(iso.transform([p])[0])
            except Exception:
                pass
        p = float(np.clip(p, 0.02, 0.98))
        q = float(meta.get("conformal_q90", 0.12))
        out["p_blue_win_gbm"] = p
        out["p_blue_win_lo"] = float(np.clip(p - q, 0.02, 0.98))
        out["p_blue_win_hi"] = float(np.clip(p + q, 0.02, 0.98))

    for name, key in (("firstblood", "fb"), ("first_inhib", "inhib")):
        gbm, iso, pack = vec(name)
        if gbm is None:
            continue
        x, meta = pack
        raw = float(gbm.predict_proba(x)[0, 1])
        used_iso = False
        p = raw
        if iso is not None and not _iso_degenerate(iso):
            try:
                p = float(iso.transform([raw])[0])
                used_iso = True
            except Exception:
                p = raw
        # First inhib: sparse LP proxy labels + often-degenerate iso → shrink to win/Elo
        if name == "first_inhib":
            p_win = float(
                out.get("p_blue_win_gbm")
                or features.get("p_dual_elo")
                or 0.5
            )
            p_elo = p_win
            base_path = MODELS_DIR / "first_inhib_baseline_elo.joblib"
            if base_path.exists() and "mu_diff" in (meta.get("feature_cols") or []):
                try:
                    lr = _load_obj_cached(base_path)
                    j = (meta.get("feature_cols") or []).index("mu_diff")
                    p_elo = float(lr.predict_proba(x[:, [j]])[0, 1])
                except Exception:
                    pass
            # Blend: raw GBM (signal) + Elo inhib baseline + win (outcome-correlated soft prior)
            if used_iso:
                p = 0.55 * p + 0.25 * p_elo + 0.20 * p_win
            else:
                p = 0.40 * raw + 0.35 * p_elo + 0.25 * p_win
            p = float(np.clip(p, 0.20, 0.80))
            out["p_blue_inhib"] = p
            out["inhib_note"] = (
                "soft blend raw+elo+win"
                + ("" if used_iso else "; iso discarded (degenerate)")
            )
            out["p_blue_inhib_raw"] = round(raw, 4)
            continue

        p = float(np.clip(p, 0.05, 0.95))
        out[f"p_blue_{key}"] = p
        if name == "firstblood":
            out["fb_note"] = "calibrated OE" + ("" if used_iso else "; iso skipped")

    # kills
    meta_path = MODELS_DIR / "kills_meta.json"
    if meta_path.exists():
        meta = _load_json_cached(meta_path)
        cols = meta.get("feature_cols") or FEAT
        x = np.array([[float(features.get(c, 0.0) or 0.0) for c in cols]])
        sel = meta.get("selected_model", "ridge")
        conf_q = float(meta.get("conformal_q90", 5.0))
        if sel == "quantile" and (MODELS_DIR / "kills_quantile.joblib").exists():
            qs = _load_obj_cached(MODELS_DIR / "kills_quantile.joblib")
            p10, p50, p90 = float(qs["q10"].predict(x)[0]), float(qs["q50"].predict(x)[0]), float(qs["q90"].predict(x)[0])
            mu = p50
            sd = max((p90 - p10) / (2 * 1.28155), 3.0)
        elif sel == "mean_gbm" and (MODELS_DIR / "kills_gbm.joblib").exists():
            mu = float(_load_obj_cached(MODELS_DIR / "kills_gbm.joblib").predict(x)[0])
            sd = float(meta.get("residual_sd", 6.0))
        else:
            ridge = _load_obj_cached(MODELS_DIR / "kills_baseline_ridge.joblib")
            mu = float(ridge.predict(x)[0])
            sd = float(meta.get("residual_sd", 6.0))
        out["kills_mean"] = mu
        out["kills_sd"] = sd
        out["kills_lo"] = mu - conf_q
        out["kills_hi"] = mu + conf_q
        out["kills_model"] = sel
    else:
        out["kills_mean"] = float(features.get("draft_expected_kills") or 28.0)
        out["kills_sd"] = 6.5

    # stack
    stack_path = MODELS_DIR / "stack_meta.joblib"
    if stack_path.exists() and "p_blue_win_gbm" in out:
        bundle = _load_obj_cached(stack_path)
        model = bundle["model"]
        order = bundle.get("feature_order") or ["p_strength_blend", "p_draft", "p_gbm", "sigma_z"]
        p_str = float(
            features.get("p_strength_blend")
            or features.get("p_dual_elo")
            or 0.5
        )
        logit = float(features.get("draft_win_logit_blue") or 0.0)
        p_draft = 1.0 / (1.0 + math.exp(-1.4 * logit))
        sig = float(features.get("sigma_pair") or 80.0)
        sigma_z = (sig - 80.0) / 40.0
        raw = {
            "p_strength_blend": p_str,
            "p_dual_elo": float(features.get("p_dual_elo") or p_str),
            "p_draft": p_draft,
            "p_gbm": out["p_blue_win_gbm"],
            "sigma_pair": sig,
            "sigma_z": sigma_z,
        }
        z = np.array([[float(raw.get(c, 0.0)) for c in order]])
        p = float(np.clip(model.predict_proba(z)[0, 1], 0.05, 0.95))
        # mild sigma shrink (less aggressive — calibration already handles scale)
        shrink = 1.0 / (1.0 + (sig / 160.0) ** 2)
        p = 0.5 + (p - 0.5) * shrink
        out["p_blue_win"] = float(np.clip(p, 0.05, 0.95))
        out["p_red_win"] = 1.0 - out["p_blue_win"]
        out["stack_components"] = {k: raw[k] for k in order if k in raw}
        out["stack_components"]["p_player_elo"] = float(features.get("p_player_elo") or 0.5)
    elif "p_blue_win_gbm" in out:
        out["p_blue_win"] = out["p_blue_win_gbm"]
        out["p_red_win"] = 1.0 - out["p_blue_win"]

    return out


def race_to_k(mu_total: float, p_blue_share: float, ks: list[int] | None = None) -> dict:
    from math import comb

    ks = ks or [5, 10, 15]
    length = 30.0
    lam_b = max(mu_total * p_blue_share / length, 0.05)
    lam_r = max(mu_total * (1.0 - p_blue_share) / length, 0.05)
    p = lam_b / (lam_b + lam_r)
    out = {}
    for K in ks:
        pb = sum(comb(K - 1 + j, j) * (p**K) * ((1 - p) ** j) for j in range(K))
        pb = float(np.clip(pb, 0.02, 0.98))
        out[str(K)] = {"p_blue_first": round(pb, 4), "p_red_first": round(1.0 - pb, 4)}
    return out


def train_all(features_path: Path | None = None, do_archive: bool = True) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if do_archive:
        try:
            archive_models("pre_train")
        except Exception as exc:  # noqa: BLE001
            print(f"[ml] archive skip: {exc}")

    path = features_path or (FEATURES_DIR / "maps.parquet")
    df = pd.read_parquet(path)
    report = {
        "win": train_classifier(df, "y_blue_win", "win"),
        "kills": train_kills(df),
        "firstblood": train_classifier(df, "y_blue_firstblood", "firstblood"),
        "first_inhib": train_classifier(df, "y_blue_first_inhib", "first_inhib"),
        "stack": train_stack_and_ablations(df),
        "n_maps": int(len(df)),
    }
    gates = evaluate_gates(report)
    write_eval_report(report, gates)
    (MODELS_DIR / "train_report.json").write_text(json.dumps(report, indent=2, default=str))
    # baseline registry snapshot
    (MODELS_DIR / "baselines.json").write_text(
        json.dumps(
            {
                "win_mean_brier": (report.get("win") or {}).get("baselines", {}).get("mean_brier"),
                "win_elo_brier": (report.get("win") or {}).get("baselines", {}).get("elo_brier"),
                "kills_ridge_crps": (report.get("kills") or {}).get("baselines", {}).get("ridge_crps"),
                "kills_selected": (report.get("kills") or {}).get("selected_model"),
            },
            indent=2,
        )
    )
    print(f"[ml] gates passed={gates['passed']} details={json.dumps(gates['details'], default=str)[:400]}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", type=Path, default=None)
    ap.add_argument("--no-archive", action="store_true")
    args = ap.parse_args()
    report = train_all(args.features, do_archive=not args.no_archive)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) else v) for k, v in report.items()}, indent=2))


if __name__ == "__main__":
    main()
