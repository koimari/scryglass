#!/usr/bin/env python3
"""
Draft archetype → outcome edges (OE majors).

Builds per-map archetype diffs (control mage, engage, hypercarry, …) and
correlates with win / length / kills / late-game proxies — Elo-residualized.

Hypothesis example: more control mages → longer games + higher WR when long.

  python3 -m lol_kills.research.draft_archetype_edges
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from lol_kills.draft_archetypes import (
    ARCHETYPE_NAMES,
    coverage_report,
    draft_archetype_features,
)
from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR


def _r(a: np.ndarray, b: np.ndarray, min_n: int = 200) -> tuple[float | None, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < min_n:
        return None, n
    aa, bb = a[mask], b[mask]
    if float(aa.std()) < 1e-12 or float(bb.std()) < 1e-12:
        return None, n
    return float(np.corrcoef(aa, bb)[0, 1]), n


def _resid(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < 80:
        return out
    lr = LinearRegression().fit(X[mask], y[mask])
    out[mask] = y[mask] - lr.predict(X[mask])
    return out


def load_maps() -> pd.DataFrame:
    maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
    feat = pd.read_parquet(FEATURES_DIR / "maps.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")

    maps = maps.copy()
    maps["game_uid"] = maps["game_uid"].astype(str)
    feat = feat.copy()
    feat["game_uid"] = feat["game_uid"].astype(str)
    keep = [c for c in ["game_uid", "mu_diff", "draft_win_logit_blue", "draft_kills_shift"] if c in feat.columns]
    df = maps.merge(feat[keep], on="game_uid", how="left")
    df["mu_diff"] = pd.to_numeric(df.get("mu_diff"), errors="coerce").fillna(0.0)
    df["draft_win_logit_blue"] = pd.to_numeric(df.get("draft_win_logit_blue"), errors="coerce").fillna(0.0)
    df["draft_kills_shift"] = pd.to_numeric(df.get("draft_kills_shift"), errors="coerce").fillna(0.0)

    df["y_blue_win"] = pd.to_numeric(df["y_blue_win"], errors="coerce")
    df["gold15"] = pd.to_numeric(df.get("blue_golddiffat15"), errors="coerce")
    df["gold10"] = pd.to_numeric(df.get("blue_golddiffat10"), errors="coerce")
    df["total_kills"] = pd.to_numeric(df.get("total_kills", df.get("y_total_kills")), errors="coerce")
    df["length_min"] = pd.to_numeric(df.get("length_min"), errors="coerce")
    if df["length_min"].isna().all():
        df["length_min"] = pd.to_numeric(df["blue_gamelength"], errors="coerce") / 60.0
    df["ckpm"] = pd.to_numeric(df.get("ckpm"), errors="coerce")
    df["under_29_5"] = (df["total_kills"] <= 29).astype(float)
    df["under_32_5"] = (df["total_kills"] <= 32).astype(float)
    df["long_game_32"] = (df["length_min"] >= 32).astype(float)
    df["long_game_35"] = (df["length_min"] >= 35).astype(float)
    # Late-fight proxy: won despite not snowballing @15
    df["comeback_win"] = ((df["gold15"] < -500) & (df["y_blue_win"] == 1)).astype(float)
    df["snowball_win"] = ((df["gold15"] > 1000) & (df["y_blue_win"] == 1)).astype(float)
    # Conditional: win given long (NaN if not long)
    df["win_if_long_32"] = np.where(df["long_game_32"] == 1, df["y_blue_win"], np.nan)
    df["win_if_long_35"] = np.where(df["long_game_35"] == 1, df["y_blue_win"], np.nan)
    df["win_if_behind_15"] = np.where(df["gold15"] < -1000, df["y_blue_win"], np.nan)
    df["win_if_ahead_15"] = np.where(df["gold15"] > 1000, df["y_blue_win"], np.nan)

    for c in ("blue_firstdragon", "blue_firstherald", "blue_firsttower", "y_blue_firstblood"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drafts from players
    p = players.copy()
    p["game_uid"] = p["game_uid"].astype(str)
    p["champion"] = p["champion"].map(lambda c: normalize_champ(str(c)))
    p["side"] = p["side"].astype(str).str.title()
    rows = []
    all_champs = []
    for gid, g in p.groupby("game_uid"):
        blue = g[g["side"] == "Blue"]["champion"].tolist()
        red = g[g["side"] == "Red"]["champion"].tolist()
        if len(blue) < 5 or len(red) < 5:
            continue
        blue, red = blue[:5], red[:5]
        all_champs.extend(blue + red)
        feat_a = draft_archetype_features(blue, red)
        feat_a["game_uid"] = gid
        rows.append(feat_a)
    adf = pd.DataFrame(rows)
    df = df.merge(adf, on="game_uid", how="inner")
    cov = coverage_report(sorted(set(all_champs)))
    return df, cov


OUTCOMES = [
    "y_blue_win",
    "length_min",
    "long_game_32",
    "long_game_35",
    "total_kills",
    "under_29_5",
    "under_32_5",
    "ckpm",
    "gold15",
    "comeback_win",
    "win_if_long_32",
    "win_if_long_35",
    "win_if_behind_15",
    "win_if_ahead_15",
    "y_blue_firstblood",
    "blue_firstdragon",
    "blue_firsttower",
]


def correlate_archetypes(df: pd.DataFrame) -> dict:
    preds = [f"arch_{a}_diff" for a in ARCHETYPE_NAMES]
    preds += [
        "arch_control_vs_engage",
        "arch_scale_vs_snowball",
        "arch_poke_vs_engage",
    ]
    preds += [
        f"arch_{a}_sum"
        for a in ("control_mage", "engage", "hypercarry_adc", "early_snowball", "scaling_late")
    ]
    preds = [p for p in preds if p in df.columns]
    X_elo = df[["mu_diff"]].astype(float).values

    raw_edges = []
    resid_edges = []
    for pred in preds:
        pv = df[pred].astype(float).values
        pr = _resid(pv, X_elo)
        for outc in OUTCOMES:
            if outc not in df.columns:
                continue
            ov = df[outc].astype(float).values
            r, n = _r(pv, ov, min_n=150)
            if r is not None and abs(r) >= 0.04:
                raw_edges.append(
                    {"predictor": pred, "outcome": outc, "r": round(r, 4), "n": n, "abs_r": abs(r)}
                )
            ovr = _resid(ov, X_elo)
            rr, nn = _r(pr, ovr, min_n=150)
            if rr is not None and abs(rr) >= 0.035:
                resid_edges.append(
                    {
                        "predictor": pred,
                        "outcome": outc,
                        "r_residual": round(rr, 4),
                        "n": nn,
                        "abs_r": abs(rr),
                    }
                )

    raw_edges.sort(key=lambda x: -x["abs_r"])
    resid_edges.sort(key=lambda x: -x["abs_r"])
    return {"raw": raw_edges, "elo_residual": resid_edges}


def control_mage_late_diagnosis(df: pd.DataFrame) -> dict:
    """Explicit test: control_mage_diff → length + WR when long."""
    sub = df.dropna(subset=["arch_control_mage_diff", "y_blue_win", "length_min", "mu_diff"]).copy()
    d = sub["arch_control_mage_diff"].values
    # bins
    bins = []
    for lo, hi, label in [(-99, -0.5, "red_more"), (-0.5, 0.5, "even"), (0.5, 99, "blue_more")]:
        g = sub[(d > lo) & (d <= hi)] if lo > -90 else sub[d <= hi]
        if lo == -99:
            g = sub[sub["arch_control_mage_diff"] <= -1]
        elif label == "even":
            g = sub[sub["arch_control_mage_diff"].abs() < 0.5]
        else:
            g = sub[sub["arch_control_mage_diff"] >= 1]
        if len(g) < 80:
            continue
        long = g[g["length_min"] >= 32]
        bins.append(
            {
                "bucket": label,
                "n": int(len(g)),
                "mean_length": round(float(g["length_min"].mean()), 2),
                "p_long_32": round(float((g["length_min"] >= 32).mean()), 4),
                "p_win": round(float(g["y_blue_win"].mean()), 4),
                "p_win_if_long_32": round(float(long["y_blue_win"].mean()), 4) if len(long) >= 40 else None,
                "n_long": int(len(long)),
                "mean_kills": round(float(g["total_kills"].mean()), 2),
                "p_under_29_5": round(float(g["under_29_5"].mean()), 4),
            }
        )

    # logistic: win | long ~ control_mage_diff + elo
    long = sub[sub["length_min"] >= 32]
    out = {"buckets": bins, "n_long_32": int(len(long))}
    if len(long) >= 200:
        X = np.column_stack([long["mu_diff"].values / 400.0, long["arch_control_mage_diff"].values])
        y = long["y_blue_win"].astype(float).values
        lr = LogisticRegression(C=1.0, max_iter=400)
        lr.fit(X, y)
        out["logistic_win_given_long"] = {
            "coef_elo_z": float(lr.coef_[0][0]),
            "coef_control_mage_diff": float(lr.coef_[0][1]),
            "delta_pp_per_extra_blue_control_mage": round(
                (
                    1 / (1 + np.exp(-(lr.intercept_[0] + lr.coef_[0][1])))
                    - 1 / (1 + np.exp(-lr.intercept_[0]))
                )
                * 100,
                2,
            ),
        }
    # length OLS
    X = np.column_stack([sub["mu_diff"].values / 400.0, sub["arch_control_mage_diff"].values])
    lr = LinearRegression().fit(X, sub["length_min"].values)
    out["length_ols"] = {
        "coef_elo_z": float(lr.coef_[0]),
        "coef_control_mage_diff": float(lr.coef_[1]),
        "minutes_per_extra_blue_control_mage": round(float(lr.coef_[1]), 3),
    }
    return out


def hypercarry_behind_diagnosis(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["arch_hypercarry_adc_diff", "gold15", "y_blue_win"]).copy()
    behind = sub[sub["gold15"] < -1000]
    if len(behind) < 100:
        return {"n": int(len(behind))}
    hi = behind[behind["arch_hypercarry_adc_diff"] >= 1]
    lo = behind[behind["arch_hypercarry_adc_diff"] <= -1]
    mid = behind[behind["arch_hypercarry_adc_diff"].abs() < 0.5]
    return {
        "n_behind": int(len(behind)),
        "p_win_blue_has_more_hypercarry": round(float(hi["y_blue_win"].mean()), 4) if len(hi) >= 40 else None,
        "n_blue_hyper": int(len(hi)),
        "p_win_red_has_more_hypercarry": round(float(lo["y_blue_win"].mean()), 4) if len(lo) >= 40 else None,
        "n_red_hyper": int(len(lo)),
        "p_win_even": round(float(mid["y_blue_win"].mean()), 4) if len(mid) >= 40 else None,
        "gap_pp": (
            round((hi["y_blue_win"].mean() - lo["y_blue_win"].mean()) * 100, 2)
            if len(hi) >= 40 and len(lo) >= 40
            else None
        ),
    }


def early_snowball_diagnosis(df: pd.DataFrame) -> dict:
    sub = df.dropna(subset=["arch_early_snowball_diff", "gold15", "y_blue_win", "total_kills"]).copy()
    X = np.column_stack([sub["mu_diff"].values / 400.0, sub["arch_early_snowball_diff"].values])
    g = LinearRegression().fit(X, sub["gold15"].fillna(0).values)
    k = LinearRegression().fit(X, sub["total_kills"].fillna(28).values)
    return {
        "gold15_per_extra_snowball_champ": round(float(g.coef_[1]), 1),
        "kills_per_extra_snowball_champ": round(float(k.coef_[1]), 3),
        "r_snowball_under29": _r(sub["arch_early_snowball_diff"].values, sub["under_29_5"].values)[0],
    }


def top_board(resid: list[dict], k: int = 30) -> list[dict]:
    """Human labels for compound betting board."""
    labels = {
        "arch_control_mage_diff": "control mages (Anivia/Ori/Viktor…)",
        "arch_engage_diff": "engage (J4/Ornn/Sej…)",
        "arch_hypercarry_adc_diff": "hypercarry ADC",
        "arch_early_snowball_diff": "early snowball",
        "arch_scaling_late_diff": "late scalers",
        "arch_poke_siege_diff": "poke/siege",
        "arch_assassin_diff": "assassins",
        "arch_peel_enchanter_diff": "enchanters",
        "arch_splitpush_diff": "splitpush",
        "arch_teamfight_aoe_diff": "AoE teamfight",
        "arch_pick_diff": "pick/hook",
        "arch_scale_vs_snowball": "scale − snowball index",
        "arch_control_vs_engage": "control − engage index",
    }
    out = []
    for e in resid[:k]:
        pred = e["predictor"]
        out.append(
            {
                **e,
                "label": labels.get(pred, pred),
                "edge_hint": _hint(pred, e["outcome"], e["r_residual"]),
            }
        )
    return out


def _hint(pred: str, outcome: str, r: float) -> str:
    direction = "more on blue" if r > 0 else "more on red (or fewer on blue)"
    if "under_" in outcome:
        return f"{direction} → {'higher' if r > 0 else 'lower'} P(under) — O/U"
    if outcome == "length_min" or "long_game" in outcome:
        return f"{direction} → {'longer' if r > 0 else 'shorter'} games"
    if "win_if_long" in outcome:
        return f"{direction} → {'better' if r > 0 else 'worse'} WR in long games (late TF proxy)"
    if "win_if_behind" in outcome:
        return f"{direction} → {'better' if r > 0 else 'worse'} comeback WR when behind@15"
    if outcome == "y_blue_win":
        return f"{direction} → map WR"
    if outcome == "total_kills" or outcome == "ckpm":
        return f"{direction} → {'higher' if r > 0 else 'lower'} pace/kills"
    if outcome == "comeback_win":
        return f"{direction} → comeback frequency"
    return f"{direction} ↔ {outcome}"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[archetype] loading maps + tagging drafts…")
    df, cov = load_maps()
    print(f"[archetype] n_maps={len(df)} champ_tag_coverage={cov}")

    print("[archetype] correlating…")
    corr = correlate_archetypes(df)
    ctrl = control_mage_late_diagnosis(df)
    hyper = hypercarry_behind_diagnosis(df)
    snow = early_snowball_diagnosis(df)
    board = top_board(corr["elo_residual"], 35)

    # Focus slices for chat
    late_tf = [
        e
        for e in corr["elo_residual"]
        if e["outcome"] in ("win_if_long_32", "win_if_long_35", "length_min", "long_game_35", "comeback_win", "win_if_behind_15")
    ][:20]
    pace = [
        e
        for e in corr["elo_residual"]
        if e["outcome"] in ("under_29_5", "under_32_5", "total_kills", "ckpm")
    ][:20]

    report = {
        "version": 1,
        "n_maps": int(len(df)),
        "champ_tag_coverage": cov,
        "archetypes": ARCHETYPE_NAMES,
        "control_mage_late": ctrl,
        "hypercarry_behind": hyper,
        "early_snowball": snow,
        "compound_edge_board": board,
        "late_teamfight_edges": late_tf,
        "pace_ou_edges": pace,
        "top_raw": corr["raw"][:40],
        "notes": [
            "Archetypes are multi-label curated tags — not Riot official roles.",
            "win_if_long_* is the late teamfight proxy (no timeline fight log in OE).",
            "All residual edges Elo-controlled via mu_diff.",
        ],
    }
    path = MODELS_DIR / "draft_archetype_edges.json"
    path.write_text(json.dumps(report, indent=2))
    print(f"[archetype] wrote {path}")
    print("\n=== CONTROL MAGE → LATE ===")
    print(json.dumps(ctrl, indent=2)[:1200])
    print("\n=== HYPERCARRY BEHIND ===")
    print(json.dumps(hyper, indent=2))
    print("\n=== TOP LATE-TF RESIDUAL EDGES ===")
    for e in late_tf[:12]:
        print(f"  {e['predictor']:32} → {e['outcome']:18} r={e['r_residual']:+.3f}")
    print("\n=== TOP PACE/O-U RESIDUAL ===")
    for e in pace[:12]:
        print(f"  {e['predictor']:32} → {e['outcome']:18} r={e['r_residual']:+.3f}")


if __name__ == "__main__":
    main()
