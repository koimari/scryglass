#!/usr/bin/env python3
"""
Deep draft → win-rate study (OE majors, league-aware).

Goals
-----
1. Measure how draft edge maps to *empirical* WR — globally and by league/tier.
2. Show draft is a *dynamic* shifter (length, kills, snowball) even when WR Δ is small.
3. Fit strength-controlled, league-specific draft→WR calibrations for Draft Score.

Run
---
  python3 -m lol_kills.research.draft_wr_study

Writes
------
  data/lol/models/draft_wr_study.json
  data/lol/models/draft_wr_calibration.json
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[2]
OUT_STUDY = MODELS_DIR / "draft_wr_study.json"
OUT_CAL = MODELS_DIR / "draft_wr_calibration.json"

LEAGUE_TIER = {
    "LCK": "tier1",
    "LPL": "tier1",
    "LEC": "west",
    "LCS": "west",
    "CBLOL": "americas",
    "AMERICAS": "americas",
    "PCS": "asia_reg",
    "VCS": "asia_reg",
    "LJL": "asia_reg",
    "LCP": "asia_reg",
    "TCL": "asia_reg",
    "MSI": "intl",
    "EWC": "intl",
    "FST": "intl",
    "Worlds": "intl",
}


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def load_draft_frame() -> pd.DataFrame:
    """One row per map: outcome, strength, draft champs, dynamics."""
    maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    feat = pd.read_parquet(FEATURES_DIR / "maps.parquet") if (FEATURES_DIR / "maps.parquet").exists() else None

    # Keep majors maps with a winner
    maps = maps.dropna(subset=["y_blue_win"]).copy()
    maps["game_uid"] = maps["game_uid"].astype(str)
    maps["date"] = pd.to_datetime(maps["date"], errors="coerce")
    maps = maps.dropna(subset=["date"]).sort_values("date")

    players = players.copy()
    players["game_uid"] = players["game_uid"].astype(str)
    players["champion"] = players["champion"].map(lambda c: normalize_champ(str(c)))
    players["side"] = players["side"].astype(str).str.title()
    pos = players["position"].astype(str).str.lower()
    role_map = {
        "top": "top",
        "jng": "jng",
        "jungle": "jng",
        "mid": "mid",
        "middle": "mid",
        "bot": "bot",
        "adc": "bot",
        "bottom": "bot",
        "sup": "sup",
        "support": "sup",
        "utility": "sup",
    }
    players["role"] = pos.map(lambda r: role_map.get(r, r[:3] if r else ""))

    # Pivot champs: blue_top … red_sup
    rows = []
    for gid, g in players.groupby("game_uid"):
        blue = g[g["side"] == "Blue"]
        red = g[g["side"] == "Red"]
        if len(blue) < 5 or len(red) < 5:
            continue
        rec = {"game_uid": gid}
        ok = True
        for side, sub, prefix in (("Blue", blue, "blue"), ("Red", red, "red")):
            for role in ("top", "jng", "mid", "bot", "sup"):
                hit = sub[sub["role"] == role]["champion"]
                if hit.empty:
                    # fallback: any leftover
                    ok = False
                    break
                rec[f"{prefix}_{role}"] = hit.iloc[0]
            if not ok:
                break
        if ok:
            rows.append(rec)
    draft = pd.DataFrame(rows)
    df = maps.merge(draft, on="game_uid", how="inner")

    if feat is not None:
        keep = [
            c
            for c in [
                "game_uid",
                "mu_diff",
                "elo_diff",
                "sigma_pair",
                "p_dual_elo",
                "draft_win_logit_blue",
                "draft_kills_shift",
                "draft_expected_kills",
                "form_wr_diff",
            ]
            if c in feat.columns
        ]
        feat = feat[keep].copy()
        feat["game_uid"] = feat["game_uid"].astype(str)
        df = df.merge(feat, on="game_uid", how="left")

    df["tier"] = df["league"].map(LEAGUE_TIER).fillna("other")
    df["elo_diff"] = df.get("mu_diff", df.get("elo_diff"))
    if "elo_diff" not in df.columns or df["elo_diff"].isna().all():
        df["elo_diff"] = 0.0
    df["elo_diff"] = df["elo_diff"].fillna(0.0)

    # Dynamics columns when present
    for c in ("length_min", "gamelength", "total_kills", "y_total_kills", "ckpm"):
        if c in df.columns:
            continue
    if "length_min" not in df.columns and "blue_gamelength" in df.columns:
        df["length_min"] = df["blue_gamelength"] / 60.0
    if "total_kills" not in df.columns:
        if "y_total_kills" in df.columns:
            df["total_kills"] = df["y_total_kills"]
        elif "blue_kills" in df.columns and "red_kills" in df.columns:
            df["total_kills"] = df["blue_kills"] + df["red_kills"]

    # gold snowball proxy from players (mean |golddiffat15| on mid/jng)
    if "golddiffat15" in players.columns:
        g15 = (
            players.dropna(subset=["golddiffat15"])
            .groupby("game_uid")["golddiffat15"]
            .apply(lambda s: float(np.nanmean(np.abs(s))))
            .rename("abs_g15")
        )
        df = df.merge(g15, on="game_uid", how="left")

    return df


def fit_champ_logits(
    df: pd.DataFrame,
    *,
    min_games: int = 40,
    C: float = 1.0,
) -> dict:
    """
    Side-aware champ presence logistic: +1 blue / −1 red, controlling Elo.
    Also fit league-tier offsets on a shared champ space (two-stage).
    """
    champs = sorted(
        {
            c
            for role in ("top", "jng", "mid", "bot", "sup")
            for side in ("blue", "red")
            for c in df[f"{side}_{role}"].dropna().unique()
        }
    )
    idx = {c: i for i, c in enumerate(champs)}
    n = len(df)
    p = len(champs)
    X = np.zeros((n, p + 1), dtype=np.float32)  # + elo
    y = df["y_blue_win"].astype(float).values
    elo = df["elo_diff"].astype(float).values
    X[:, -1] = elo / 400.0

    for i, (_, r) in enumerate(df.iterrows()):
        for role in ("top", "jng", "mid", "bot", "sup"):
            bc, rc = r[f"blue_{role}"], r[f"red_{role}"]
            if bc in idx:
                X[i, idx[bc]] += 1.0
            if rc in idx:
                X[i, idx[rc]] -= 1.0

    # Counts for shrinkage
    counts = defaultdict(int)
    for role in ("top", "jng", "mid", "bot", "sup"):
        for side in ("blue", "red"):
            for c, cnt in df[f"{side}_{role}"].value_counts().items():
                counts[c] += int(cnt)

    clf = LogisticRegression(C=C, max_iter=400, solver="lbfgs")
    clf.fit(X, y)
    raw = {c: float(clf.coef_[0][idx[c]]) for c in champs}
    # hierarchical shrink rare champs toward 0
    logits = {}
    for c, beta in raw.items():
        n_c = counts.get(c, 0)
        w = n_c / (n_c + 40.0)
        if n_c < min_games:
            w *= n_c / max(min_games, 1)
        logits[c] = {"logit": beta * w, "raw": beta, "n": n_c, "w": w}

    # Tier-specific residual models: y ~ elo + draft_global + (no new champs — scale draft)
    draft_logit = X[:, :-1] @ np.array([logits[c]["logit"] for c in champs])
    df = df.copy()
    df["_draft_logit"] = draft_logit
    df["_p_draft_raw"] = _sigmoid(1.4 * draft_logit)

    tier_scale = {}
    for tier, sub in df.groupby("tier"):
        if len(sub) < 200:
            continue
        # logit(y) ≈ a + b1*elo + b2*draft
        Z = np.column_stack(
            [
                sub["elo_diff"].values / 400.0,
                sub["_draft_logit"].values,
            ]
        )
        yy = sub["y_blue_win"].astype(float).values
        lr = LogisticRegression(C=0.5, max_iter=300)
        lr.fit(Z, yy)
        tier_scale[tier] = {
            "n": int(len(sub)),
            "intercept": float(lr.intercept_[0]),
            "coef_elo": float(lr.coef_[0][0]),
            "coef_draft": float(lr.coef_[0][1]),
            # implied temperature vs global 1.0
            "draft_temperature": float(lr.coef_[0][1]),
        }

    # League-level scales (top leagues only)
    league_scale = {}
    for lg, sub in df.groupby("league"):
        if len(sub) < 250:
            continue
        Z = np.column_stack([sub["elo_diff"].values / 400.0, sub["_draft_logit"].values])
        yy = sub["y_blue_win"].astype(float).values
        lr = LogisticRegression(C=0.5, max_iter=300)
        lr.fit(Z, yy)
        league_scale[lg] = {
            "n": int(len(sub)),
            "coef_elo": float(lr.coef_[0][0]),
            "coef_draft": float(lr.coef_[0][1]),
            "intercept": float(lr.intercept_[0]),
        }

    return {
        "champs": logits,
        "champ_list": champs,
        "tier_scale": tier_scale,
        "league_scale": league_scale,
        "model_intercept": float(clf.intercept_[0]),
        "model_elo_coef": float(clf.coef_[0][-1]),
        "frame": df,
    }


def calibration_bins(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    out = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() < 30:
            continue
        out.append(
            {
                "bin": i,
                "n": int(m.sum()),
                "p_mean": float(p[m].mean()),
                "y_mean": float(y[m].mean()),
                "gap_pp": float((y[m].mean() - p[m].mean()) * 100),
            }
        )
    return out


def equal_strength_strata(df: pd.DataFrame) -> dict:
    """Draft effect when |Elo| is small vs large — draft should matter more when even."""
    d = df.dropna(subset=["_draft_logit", "elo_diff", "y_blue_win"]).copy()
    d["abs_elo"] = d["elo_diff"].abs()
    d["draft_sign"] = np.sign(d["_draft_logit"])
    # quintiles of |draft|
    d["draft_abs_q"] = pd.qcut(d["_draft_logit"].abs(), 5, labels=False, duplicates="drop")

    results = {}
    for name, mask in (
        ("even_elo_|d|<50", d["abs_elo"] < 50),
        ("mid_elo_50-150", (d["abs_elo"] >= 50) & (d["abs_elo"] < 150)),
        ("gap_elo_>=150", d["abs_elo"] >= 150),
    ):
        sub = d[mask]
        if len(sub) < 200:
            continue
        # WR when draft favors blue vs red
        fav_b = sub[sub["_draft_logit"] > 0.05]
        fav_r = sub[sub["_draft_logit"] < -0.05]
        results[name] = {
            "n": int(len(sub)),
            "wr_when_draft_blue": float(fav_b["y_blue_win"].mean()) if len(fav_b) else None,
            "n_draft_blue": int(len(fav_b)),
            "wr_when_draft_red": float(fav_r["y_blue_win"].mean()) if len(fav_r) else None,
            "n_draft_red": int(len(fav_r)),
            "draft_wr_swing_pp": (
                float(fav_b["y_blue_win"].mean() - fav_r["y_blue_win"].mean()) * 100
                if len(fav_b) and len(fav_r)
                else None
            ),
        }
    return results


def dynamics_vs_draft(df: pd.DataFrame) -> dict:
    """How draft kill/win edges reshape the *game*, not just binary WR."""
    d = df.dropna(subset=["_draft_logit"]).copy()
    out = {}
    # Correlate draft kill shift / win logit with dynamics
    if "draft_kills_shift" in d.columns:
        kill_edge = d["draft_kills_shift"].fillna(0)
    else:
        kill_edge = d["_draft_logit"]  # proxy

    for col, label in (
        ("length_min", "length_min"),
        ("total_kills", "total_kills"),
        ("abs_g15", "abs_golddiff15"),
        ("ckpm", "ckpm"),
    ):
        if col not in d.columns or d[col].isna().all():
            continue
        x = kill_edge.values.astype(float)
        y = d[col].astype(float).values
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 200:
            continue
        corr = float(np.corrcoef(x[m], y[m])[0, 1])
        # Top vs bottom draft-kill quintile means
        q = pd.qcut(x[m], 5, labels=False, duplicates="drop")
        means = {int(qi): float(y[m][q == qi].mean()) for qi in np.unique(q)}
        out[label] = {
            "corr_with_draft_kill_or_logit": corr,
            "quintile_means": means,
            "bottom_q_mean": means.get(0),
            "top_q_mean": means.get(max(means)),
            "top_minus_bottom": (
                means.get(max(means), 0) - means.get(0, 0) if means else None
            ),
        }

    # High |draft win logit| → more decisive / shorter? split by abs draft
    d["abs_draft"] = d["_draft_logit"].abs()
    if "length_min" in d.columns:
        hi = d[d["abs_draft"] >= d["abs_draft"].quantile(0.8)]
        lo = d[d["abs_draft"] <= d["abs_draft"].quantile(0.2)]
        out["length_by_abs_draft_edge"] = {
            "high_edge_mean_min": float(hi["length_min"].mean()),
            "low_edge_mean_min": float(lo["length_min"].mean()),
            "delta_min": float(hi["length_min"].mean() - lo["length_min"].mean()),
            "n_hi": int(len(hi)),
            "n_lo": int(len(lo)),
        }
    if "total_kills" in d.columns:
        hi = d[d["abs_draft"] >= d["abs_draft"].quantile(0.8)]
        lo = d[d["abs_draft"] <= d["abs_draft"].quantile(0.2)]
        out["kills_by_abs_draft_edge"] = {
            "high_edge_mean": float(hi["total_kills"].mean()),
            "low_edge_mean": float(lo["total_kills"].mean()),
            "delta": float(hi["total_kills"].mean() - lo["total_kills"].mean()),
        }
    return out


def time_series_eval(df: pd.DataFrame) -> dict:
    """Compare Elo-only vs Elo+draft vs tier-scaled draft on purged time folds."""
    d = df.dropna(subset=["_draft_logit", "elo_diff", "y_blue_win"]).sort_values("date")
    y = d["y_blue_win"].astype(float).values
    elo = d["elo_diff"].values / 400.0
    draft = d["_draft_logit"].values
    tiers = d["tier"].astype(str).values
    tier_map = {
        t: i
        for i, t in enumerate(sorted(d["tier"].unique()))
    }
    # Build design
    X_elo = elo.reshape(-1, 1)
    X_both = np.column_stack([elo, draft])
    # tier interaction: draft * one-hot tier
    T = np.zeros((len(d), len(tier_map)))
    for i, t in enumerate(tiers):
        T[i, tier_map[t]] = draft[i]
    X_tier = np.column_stack([elo, T])

    tscv = TimeSeriesSplit(n_splits=5)
    scores = {"elo": [], "elo_draft": [], "elo_draft_tier": []}
    aucs = {k: [] for k in scores}

    for tr, te in tscv.split(X_both):
        for name, X in (
            ("elo", X_elo),
            ("elo_draft", X_both),
            ("elo_draft_tier", X_tier),
        ):
            lr = LogisticRegression(C=0.8, max_iter=400)
            lr.fit(X[tr], y[tr])
            p = lr.predict_proba(X[te])[:, 1]
            scores[name].append(float(brier_score_loss(y[te], p)))
            try:
                aucs[name].append(float(roc_auc_score(y[te], p)))
            except ValueError:
                # A single-class fold has no defined ROC AUC.
                pass

    def summarize(xs):
        return {"mean": float(np.mean(xs)), "std": float(np.std(xs)), "folds": xs}

    return {
        "brier": {k: summarize(v) for k, v in scores.items()},
        "auc": {k: summarize(v) for k, v in aucs.items() if v},
        "tier_index": tier_map,
        "delta_brier_elo_to_draft": float(
            np.mean(scores["elo"]) - np.mean(scores["elo_draft"])
        ),
        "delta_brier_draft_to_tier": float(
            np.mean(scores["elo_draft"]) - np.mean(scores["elo_draft_tier"])
        ),
    }


def wr_bump_curve(df: pd.DataFrame, n_bins: int = 12) -> dict:
    """Empirical WR vs draft logit — the mapping Draft Score should use."""
    d = df.dropna(subset=["_draft_logit", "y_blue_win"])
    # residualize Elo first: look at draft within Elo-matched buckets
    d = d.copy()
    d["elo_q"] = pd.qcut(d["elo_diff"], 5, labels=False, duplicates="drop")

    global_bins = []
    # bin by draft logit
    try:
        d["draft_q"] = pd.qcut(d["_draft_logit"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        d["draft_q"] = pd.cut(d["_draft_logit"], n_bins, labels=False)
    for q, sub in d.groupby("draft_q"):
        global_bins.append(
            {
                "q": int(q),
                "n": int(len(sub)),
                "draft_logit_mean": float(sub["_draft_logit"].mean()),
                "p_draft_sigmoid_1_4": float(_sigmoid(1.4 * sub["_draft_logit"].mean())),
                "empirical_wr": float(sub["y_blue_win"].mean()),
                "mean_elo_diff": float(sub["elo_diff"].mean()),
            }
        )

    # Elo-controlled: within each elo quintile, blue-favored vs red-favored draft
    controlled = []
    for eq, sub in d.groupby("elo_q"):
        hi = sub[sub["_draft_logit"] >= sub["_draft_logit"].quantile(0.7)]
        lo = sub[sub["_draft_logit"] <= sub["_draft_logit"].quantile(0.3)]
        if len(hi) < 40 or len(lo) < 40:
            continue
        controlled.append(
            {
                "elo_q": int(eq),
                "n_hi": int(len(hi)),
                "n_lo": int(len(lo)),
                "wr_hi_draft": float(hi["y_blue_win"].mean()),
                "wr_lo_draft": float(lo["y_blue_win"].mean()),
                "wr_bump_pp": float((hi["y_blue_win"].mean() - lo["y_blue_win"].mean()) * 100),
                "draft_logit_hi": float(hi["_draft_logit"].mean()),
                "draft_logit_lo": float(lo["_draft_logit"].mean()),
                "elo_mean": float(sub["elo_diff"].mean()),
            }
        )

    # Fit isotonic-like linear map: empirical_wr ≈ sigmoid(a + b * draft_logit) after Elo residual
    # Residualize y on Elo, then regress residual logit on draft
    lr_elo = LogisticRegression(C=1.0, max_iter=300)
    lr_elo.fit(d[["elo_diff"]].values / 400.0, d["y_blue_win"].astype(float))
    p_elo = lr_elo.predict_proba(d[["elo_diff"]].values / 400.0)[:, 1]
    # Working residual in probability space
    resid = d["y_blue_win"].astype(float).values - p_elo
    # Ridge: resid ≈ b * draft  (through origin-ish)
    ridge = Ridge(alpha=1.0, fit_intercept=True)
    ridge.fit(d[["_draft_logit"]].values, resid)
    # Suggested temperature: map draft_logit → Δp
    # Also logistic on residualized via stacked
    lr2 = LogisticRegression(C=0.8, max_iter=300)
    lr2.fit(
        np.column_stack([d["elo_diff"].values / 400.0, d["_draft_logit"].values]),
        d["y_blue_win"].astype(float),
    )

    return {
        "global_bins": global_bins,
        "elo_controlled_bumps": controlled,
        "mean_controlled_bump_pp": float(np.mean([c["wr_bump_pp"] for c in controlled]))
        if controlled
        else None,
        "residual_ridge": {
            "intercept_dp": float(ridge.intercept_),
            "coef_dp_per_logit": float(ridge.coef_[0]),
            "note": "ΔP(win) ≈ intercept + coef * draft_logit after removing Elo-only prediction",
        },
        "joint_logistic": {
            "intercept": float(lr2.intercept_[0]),
            "coef_elo": float(lr2.coef_[0][0]),
            "coef_draft": float(lr2.coef_[0][1]),
            "suggested_temperature_vs_1_4": float(lr2.coef_[0][1] / 1.4),
        },
    }


def build_calibration_tables(fit: dict) -> dict:
    """Per-tier / per-league maps: p = sigmoid(a + b_elo*elo/400 + b_draft*draft_logit)."""
    return {
        "version": 1,
        "formula": "p_blue = sigmoid(intercept + coef_elo*(elo_diff/400) + coef_draft*draft_logit)",
        "global": {
            "intercept": fit.get("model_intercept"),
            "coef_elo": fit.get("model_elo_coef"),
            # draft logits already in champ vector; joint refit below uses tier tables
        },
        "by_tier": fit["tier_scale"],
        "by_league": fit["league_scale"],
        "usage": (
            "Replace Draft Score raw sigmoid(1.4*win_edge) with "
            "sigmoid(tier.intercept + tier.coef_elo*(μ_diff/400) + tier.coef_draft*win_edge) "
            "OR apply temperature: sigmoid(tier.coef_draft/1.4 * 1.4 * win_edge) after Elo stack."
        ),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[draft_wr_study] loading OE maps + drafts…")
    df = load_draft_frame()
    print(f"[draft_wr_study] n_maps_with_draft={len(df)} leagues={df['league'].nunique()}")

    print("[draft_wr_study] fitting champ logits + league scales…")
    fit = fit_champ_logits(df)
    df = fit["frame"]

    print("[draft_wr_study] calibration / strata / dynamics / CV…")
    y = df["y_blue_win"].astype(float).values
    p_d = df["_p_draft_raw"].values
    lr = LogisticRegression(C=0.8, max_iter=300)
    lr.fit(
        np.column_stack([df["elo_diff"].values / 400.0, df["_draft_logit"].values]),
        y,
    )
    p_joint = lr.predict_proba(
        np.column_stack([df["elo_diff"].values / 400.0, df["_draft_logit"].values])
    )[:, 1]

    study = {
        "n_maps": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "n_champs": len(fit["champs"]),
        "league_counts": df["league"].value_counts().to_dict(),
        "tier_counts": df["tier"].value_counts().to_dict(),
        "metrics": {
            "draft_only_brier": float(brier_score_loss(y, p_d)),
            "draft_only_auc": float(roc_auc_score(y, p_d)),
            "elo_draft_brier": float(brier_score_loss(y, p_joint)),
            "elo_draft_auc": float(roc_auc_score(y, p_joint)),
            "elo_draft_log_loss": float(log_loss(y, np.clip(p_joint, 1e-4, 1 - 1e-4))),
        },
        "calibration_draft_only": calibration_bins(y, p_d),
        "calibration_elo_draft": calibration_bins(y, p_joint),
        "equal_strength_strata": equal_strength_strata(df),
        "dynamics": dynamics_vs_draft(df),
        "time_series_cv": time_series_eval(df),
        "wr_bump": wr_bump_curve(df),
        "tier_scale": fit["tier_scale"],
        "league_scale": fit["league_scale"],
        "top_champ_logits": sorted(
            (
                {"champ": c, **v}
                for c, v in fit["champs"].items()
                if v["n"] >= 80
            ),
            key=lambda x: -abs(x["logit"]),
        )[:40],
        "key_findings_hints": [],
    }

    # Auto narrative hints
    strata = study["equal_strength_strata"]
    if "even_elo_|d|<50" in strata and strata["even_elo_|d|<50"].get("draft_wr_swing_pp") is not None:
        even = strata["even_elo_|d|<50"]["draft_wr_swing_pp"]
        gap = strata.get("gap_elo_>=150", {}).get("draft_wr_swing_pp")
        study["key_findings_hints"].append(
            f"Even-Elo draft swing ≈ {even:.1f}pp WR"
            + (f"; large-Elo gap swing ≈ {gap:.1f}pp" if gap is not None else "")
        )
    cv = study["time_series_cv"]
    study["key_findings_hints"].append(
        f"Time-series Brier: Elo {cv['brier']['elo']['mean']:.4f} → "
        f"+draft {cv['brier']['elo_draft']['mean']:.4f} (Δ{cv['delta_brier_elo_to_draft']:+.4f}) → "
        f"+tier {cv['brier']['elo_draft_tier']['mean']:.4f} (Δ{cv['delta_brier_draft_to_tier']:+.4f})"
    )
    bump = study["wr_bump"]
    if bump.get("mean_controlled_bump_pp") is not None:
        study["key_findings_hints"].append(
            f"Elo-controlled high-vs-low draft bump ≈ {bump['mean_controlled_bump_pp']:.1f}pp mean across Elo quintiles"
        )
        jl = bump["joint_logistic"]
        study["key_findings_hints"].append(
            f"Joint logistic draft coef={jl['coef_draft']:.3f} "
            f"(current Draft Score uses temp 1.4 → suggested scale {jl['suggested_temperature_vs_1_4']:.2f}×)"
        )

    for tier, sc in sorted(fit["tier_scale"].items(), key=lambda x: -abs(x[1]["coef_draft"])):
        study["key_findings_hints"].append(
            f"Tier {tier}: draft_coef={sc['coef_draft']:.3f} elo_coef={sc['coef_elo']:.3f} n={sc['n']}"
        )

    cal = build_calibration_tables(fit)
    cal["joint_logistic_global"] = bump["joint_logistic"]
    cal["residual_ridge"] = bump["residual_ridge"]
    cal["champ_logits_shrunk"] = {
        c: v["logit"] for c, v in fit["champs"].items() if v["n"] >= 40
    }

    # JSON-safe league counts
    study["league_counts"] = {str(k): int(v) for k, v in study["league_counts"].items()}
    study["tier_counts"] = {str(k): int(v) for k, v in study["tier_counts"].items()}

    OUT_STUDY.write_text(json.dumps(study, indent=2, default=str))
    OUT_CAL.write_text(json.dumps(cal, indent=2, default=str))
    print(f"[draft_wr_study] wrote {OUT_STUDY}")
    print(f"[draft_wr_study] wrote {OUT_CAL}")
    for h in study["key_findings_hints"]:
        print(" •", h)


if __name__ == "__main__":
    main()
