#!/usr/bin/env python3
"""
Champion fight-impact from gold → damage conversion (OE players).

For each player-game we observe DPM and earned gold/min. Fit a role-specific
scaling curve:

    E[DPM | role] = a_r + b_r · egpm + c_r · egpm² + d_r · length_min

Impact residual = observed DPM − predicted DPM (gold-expected fight output).

Also track damageshare / earnedgoldshare efficiency, and split residuals by
win vs loss so we can see who *converts* gold into fight results when it matters.

  python3 -m lol_kills.research.champ_impact

Writes
------
  data/lol/models/champ_impact.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import MODELS_DIR, PARQUET_DIR

OUT = MODELS_DIR / "champ_impact.json"
POS_MAP = {
    "top": "top",
    "jungle": "jng",
    "jng": "jng",
    "mid": "mid",
    "bottom": "bot",
    "bot": "bot",
    "adc": "bot",
    "support": "sup",
    "sup": "sup",
}
ROLES = ("top", "jng", "mid", "bot", "sup")
MIN_CHAMP_N = 25
PRIOR_N = 40.0
# Map residual DPM → approximate “pp-like” tier bonus scale
DPM_TO_PP = 0.012  # ~80 DPM residual ≈ 1pp


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_player_impact_frame(
    *,
    patches: tuple[float, ...] | None = None,
    min_patch: float | None = None,
) -> pd.DataFrame:
    p = pd.read_parquet(PARQUET_DIR / "players.parquet")
    need = ["dpm", "earned gpm", "damageshare", "earnedgoldshare", "result", "champion", "position"]
    for c in need:
        if c not in p.columns:
            raise RuntimeError(
                f"players.parquet missing {c!r} — re-run OE ingest (full schema) + warehouse join"
            )
    df = p.copy()
    df["pos"] = df["position"].astype(str).str.lower().map(lambda x: POS_MAP.get(x, x))
    df = df[df["pos"].isin(ROLES)].copy()
    df["champion"] = df["champion"].map(lambda x: normalize_champ(str(x)) if pd.notna(x) else None)
    df["dpm"] = _num(df["dpm"])
    df["egpm"] = _num(df["earned gpm"])
    df["dshare"] = _num(df["damageshare"])
    df["gshare"] = _num(df["earnedgoldshare"])
    df["result"] = _num(df["result"])
    df["patch_f"] = _num(df["patch"])
    if "gamelength" in df.columns:
        df["length_min"] = _num(df["gamelength"]) / 60.0
    else:
        df["length_min"] = np.nan
    df["totalgold"] = _num(df["totalgold"]) if "totalgold" in df.columns else np.nan
    df["damagetochampions"] = (
        _num(df["damagetochampions"]) if "damagetochampions" in df.columns else np.nan
    )

    df = df.dropna(subset=["dpm", "egpm", "champion", "pos", "result"])
    df = df[(df["egpm"] > 50) & (df["dpm"] > 0) & (df["egpm"] < 1200) & (df["dpm"] < 2500)]
    if patches is not None:
        df = df[df["patch_f"].isin(patches)]
    if min_patch is not None:
        df = df[df["patch_f"] >= min_patch]
    # efficiency: damage share per gold share
    df["eff"] = df["dshare"] / df["gshare"].clip(lower=0.04)
    df["eff"] = df["eff"].clip(0.2, 3.5)
    return df.reset_index(drop=True)


def fit_scaling_curves(df: pd.DataFrame) -> dict[str, dict]:
    """Role-specific gold→DPM curves."""
    curves: dict[str, dict] = {}
    for role, sub in df.groupby("pos"):
        sub = sub.dropna(subset=["dpm", "egpm"])
        if len(sub) < 200:
            continue
        length = sub["length_min"].fillna(float(sub["length_min"].median() or 32.0))
        X = np.column_stack(
            [
                np.ones(len(sub)),
                sub["egpm"].values,
                (sub["egpm"].values ** 2) / 1000.0,
                length.values,
            ]
        )
        y = sub["dpm"].values
        model = Ridge(alpha=5.0, fit_intercept=False)
        model.fit(X, y)
        pred = model.predict(X)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        r2 = float(1.0 - np.var(y - pred) / max(np.var(y), 1e-9))
        curves[str(role)] = {
            "coef": [float(c) for c in model.coef_],
            "feature_names": ["intercept", "egpm", "egpm2_over_1000", "length_min"],
            "n": int(len(sub)),
            "rmse": round(rmse, 2),
            "r2": round(r2, 4),
            "mean_dpm": round(float(y.mean()), 2),
            "mean_egpm": round(float(sub["egpm"].mean()), 2),
        }
    return curves


def attach_residuals(df: pd.DataFrame, curves: dict[str, dict]) -> pd.DataFrame:
    out = df.copy()
    resid = np.full(len(out), np.nan)
    for role, cur in curves.items():
        mask = out["pos"].values == role
        if not mask.any():
            continue
        coef = np.array(cur["coef"], dtype=float)
        egpm = out.loc[mask, "egpm"].values
        length = out.loc[mask, "length_min"].fillna(32.0).values
        X = np.column_stack([np.ones(mask.sum()), egpm, (egpm**2) / 1000.0, length])
        pred = X @ coef
        resid[mask] = out.loc[mask, "dpm"].values - pred
        out.loc[mask, "dpm_hat"] = pred
    out["dpm_resid"] = resid
    return out.dropna(subset=["dpm_resid"])


def _shrink(mean: float, n: int, prior: float = PRIOR_N) -> float:
    return float(mean) * (n / (n + prior))


def aggregate_champ_impact(
    df: pd.DataFrame,
    *,
    min_n: int = MIN_CHAMP_N,
) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for champ, g in df.groupby("champion"):
        n = len(g)
        if n < min_n:
            continue
        wins = g[g["result"] == 1]
        losses = g[g["result"] == 0]
        resid_all = float(g["dpm_resid"].mean())
        resid_w = float(wins["dpm_resid"].mean()) if len(wins) >= 8 else None
        resid_l = float(losses["dpm_resid"].mean()) if len(losses) >= 8 else None
        conversion = None
        if resid_w is not None and resid_l is not None:
            conversion = resid_w - resid_l  # + = converts more in wins than losses
        eff = float(g["eff"].mean()) if g["eff"].notna().any() else None
        # Primary role
        role = g["pos"].value_counts().idxmax()
        # Composite impact used in tier_score (DPM residual units, then → pp)
        # Weight conversion so win/loss application of gold matters.
        base = _shrink(resid_all, n)
        conv_term = _shrink(conversion, min(len(wins), len(losses))) if conversion is not None else 0.0
        impact_dpm = 0.65 * base + 0.35 * conv_term
        # efficiency tilt vs role mean later — store raw for now
        rows[str(champ)] = {
            "n": int(n),
            "n_wins": int(len(wins)),
            "n_losses": int(len(losses)),
            "role": str(role),
            "mean_dpm": round(float(g["dpm"].mean()), 2),
            "mean_egpm": round(float(g["egpm"].mean()), 2),
            "mean_eff": None if eff is None else round(eff, 3),
            "dpm_resid": round(resid_all, 2),
            "dpm_resid_shrunk": round(base, 2),
            "dpm_resid_win": None if resid_w is None else round(resid_w, 2),
            "dpm_resid_loss": None if resid_l is None else round(resid_l, 2),
            "conversion_win_minus_loss": None if conversion is None else round(conversion, 2),
            "impact_dpm": round(impact_dpm, 2),
            "impact_pp": round(impact_dpm * DPM_TO_PP, 3),
        }
    return rows


def build_impact(
    *,
    fit_patches: tuple[float, ...] | None = None,
    score_patches: tuple[float, ...] | None = None,
    min_n: int = MIN_CHAMP_N,
) -> dict:
    """
    Fit gold→DPM curves on fit_patches (or all data), score champ impact on
    score_patches (default = fit).
    """
    fit_df = load_player_impact_frame(patches=fit_patches)
    print(f"[champ_impact] fit rows={len(fit_df)} patches={fit_patches or 'all'}")
    curves = fit_scaling_curves(fit_df)
    fit_r = attach_residuals(fit_df, curves)

    if score_patches is None or score_patches == fit_patches:
        score_r = fit_r
    else:
        score_df = load_player_impact_frame(patches=score_patches)
        score_r = attach_residuals(score_df, curves)
        print(f"[champ_impact] score rows={len(score_r)} patches={score_patches}")

    # Role mean efficiency for relative tilt
    role_eff = score_r.groupby("pos")["eff"].mean().to_dict()
    champs = aggregate_champ_impact(score_r, min_n=min_n)
    for c, row in champs.items():
        re = float(role_eff.get(row["role"], 1.0) or 1.0)
        if row["mean_eff"] is not None and re > 0:
            row["eff_vs_role"] = round(row["mean_eff"] / re - 1.0, 3)
            # small efficiency add-on in pp space
            row["impact_pp"] = round(
                row["impact_pp"] + 1.5 * float(np.clip(row["eff_vs_role"], -0.35, 0.45)),
                3,
            )
        else:
            row["eff_vs_role"] = None

    ranked = sorted(champs.items(), key=lambda x: -x[1]["impact_pp"])
    out = {
        "version": 1,
        "estimand": (
            "DPM residual vs role gold-scaling curve; "
            "impact mixes mean residual + (win resid − loss resid) conversion; "
            "plus damageshare/earnedgoldshare efficiency vs role"
        ),
        "dpm_to_pp": DPM_TO_PP,
        "fit_patches": list(fit_patches) if fit_patches else None,
        "score_patches": list(score_patches) if score_patches else list(fit_patches) if fit_patches else None,
        "n_fit_rows": int(len(fit_r)),
        "n_score_rows": int(len(score_r)),
        "curves_by_role": curves,
        "role_mean_eff": {k: round(float(v), 3) for k, v in role_eff.items()},
        "champs": champs,
        "top_impact": [{"champ": c, **v} for c, v in ranked[:25]],
        "bottom_impact": [{"champ": c, **v} for c, v in ranked[-15:]],
    }
    return out


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    # Curve fit on recent patches; report both window + 16.13-only champ table in one artifact.
    fit_patches = (16.08, 16.09, 16.1, 16.11, 16.13)
    out = build_impact(fit_patches=fit_patches, score_patches=fit_patches, min_n=30)
    # Also attach score-patch-only (16.13) table for tierlist
    out_1613 = build_impact(
        fit_patches=fit_patches,
        score_patches=(16.13,),
        min_n=12,
    )
    out["by_score_patch"] = {
        "16.13": {
            "n_score_rows": out_1613["n_score_rows"],
            "champs": out_1613["champs"],
            "top_impact": out_1613["top_impact"][:20],
            "bottom_impact": out_1613["bottom_impact"][:10],
        }
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[champ_impact] wrote {OUT}")
    print("top 16.13:")
    for row in out["by_score_patch"]["16.13"]["top_impact"][:12]:
        print(
            f"  {row['champ']:14} impact_pp={row['impact_pp']:+.2f} "
            f"resid={row['dpm_resid']:+.0f} conv={row['conversion_win_minus_loss']} "
            f"eff={row['mean_eff']} n={row['n']}"
        )


if __name__ == "__main__":
    main()
