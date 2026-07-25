#!/usr/bin/env python3
"""
Deep unexpected-correlation hunt for betting edges (OE majors).

Expands beyond the obvious win↔tower/gold cluster:
  - pre-match features (draft, form, rolls, h2h, sigma, map index…)
  - residual correlations after Elo control
  - outcome-focused ranks (win / kills / length / FB)

  python3 -m lol_kills.research.metric_correlation_matrix
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR

# Pre-match / early signals (usable before or early live)
PREMATCH = [
    "draft_win_logit_blue",
    "draft_kills_shift",
    "draft_expected_kills",
    "abs_draft_edge",
    "mu_diff",
    "p_dual_elo",
    "sigma_pair",
    "form_wr_diff",
    "form_kills_diff",
    "form_ckpm_avg",
    "form_ka_diff",
    "h2h_blue_edge",
    "roll_fb_diff",
    "roll_g10_diff",
    "player_mu_diff",
    "roster_changed",
    "map_index",
    "league_strength",
    "playoffs",
    "kill_conc_diff",
    "max_carry_blue",
    "scaling_flag",
]

# Outcomes / in-game (for association; live/post only where noted)
OUTCOMES = [
    "y_blue_win",
    "y_blue_firstblood",
    "y_blue_first_inhib",
    "blue_firstdragon",
    "blue_firstherald",
    "blue_firsttower",
    "blue_firstbaron",
    "gold10",
    "gold15",
    "tower_diff",
    "dragon_diff",
    "baron_diff",
    "total_kills",
    "under_29_5",
    "under_32_5",
    "length_min",
    "ckpm",
    "long_game_35",
]

# Pairs we already know / not "edges"
OBVIOUS = {
    frozenset(p)
    for p in [
        ("y_blue_win", "tower_diff"),
        ("y_blue_win", "dragon_diff"),
        ("y_blue_win", "baron_diff"),
        ("y_blue_win", "gold15"),
        ("y_blue_win", "gold10"),
        ("y_blue_win", "blue_firstbaron"),
        ("y_blue_win", "y_blue_first_inhib"),
        ("y_blue_win", "mu_diff"),
        ("y_blue_win", "p_dual_elo"),
        ("tower_diff", "dragon_diff"),
        ("tower_diff", "gold15"),
        ("tower_diff", "baron_diff"),
        ("gold10", "gold15"),
        ("total_kills", "ckpm"),
        ("total_kills", "length_min"),
        ("under_29_5", "total_kills"),
        ("under_32_5", "total_kills"),
        ("under_29_5", "ckpm"),
        ("under_32_5", "ckpm"),
        ("under_29_5", "under_32_5"),
        ("long_game_35", "length_min"),
        ("mu_diff", "p_dual_elo"),
        ("mu_diff", "player_mu_diff"),
        ("draft_expected_kills", "draft_kills_shift"),
        ("draft_win_logit_blue", "abs_draft_edge"),
        ("form_kills_diff", "form_ka_diff"),
        ("y_blue_first_inhib", "tower_diff"),
        ("y_blue_first_inhib", "baron_diff"),
        ("blue_firstbaron", "baron_diff"),
    ]
}


def _r(a: np.ndarray, b: np.ndarray, min_n: int = 120) -> tuple[float | None, int]:
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < min_n:
        return None, n
    aa, bb = a[mask], b[mask]
    if float(aa.std()) < 1e-12 or float(bb.std()) < 1e-12:
        return None, n
    return float(np.corrcoef(aa, bb)[0, 1]), n


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """OLS residuals of y on columns of X (with intercept)."""
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < 50:
        return out
    lr = LinearRegression()
    lr.fit(X[mask], y[mask])
    out[mask] = y[mask] - lr.predict(X[mask])
    return out


def load_frame() -> pd.DataFrame:
    maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
    feat = pd.read_parquet(FEATURES_DIR / "maps.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")

    df = maps.copy()
    df["game_uid"] = df["game_uid"].astype(str)
    feat = feat.copy()
    feat["game_uid"] = feat["game_uid"].astype(str)
    # prefer feat columns on conflict for model features
    overlap = [c for c in feat.columns if c in df.columns and c != "game_uid"]
    df = df.drop(columns=overlap, errors="ignore").merge(feat, on="game_uid", how="inner")

    def num(col, default=np.nan):
        if col not in df.columns:
            return pd.Series(default, index=df.index, dtype=float)
        return pd.to_numeric(df[col], errors="coerce")

    df["draft_win_logit_blue"] = num("draft_win_logit_blue").fillna(0.0)
    df["draft_kills_shift"] = num("draft_kills_shift").fillna(0.0)
    df["draft_expected_kills"] = num("draft_expected_kills")
    df["abs_draft_edge"] = df["draft_win_logit_blue"].abs()
    df["mu_diff"] = num("mu_diff", 0.0).fillna(num("elo_diff", 0.0)).fillna(0.0)
    df["p_dual_elo"] = num("p_dual_elo")
    df["sigma_pair"] = num("sigma_pair")
    df["form_wr_diff"] = num("form_wr_diff")
    df["form_kills_diff"] = num("form_kills_diff")
    df["form_ckpm_avg"] = num("form_ckpm_avg")
    df["form_ka_diff"] = num("form_ka_blue") - num("form_ka_red")
    df["h2h_blue_edge"] = num("h2h_blue_edge")
    df["roll_fb_diff"] = num("roll_fb_diff")
    df["roll_g10_diff"] = num("roll_g10_diff")
    df["player_mu_diff"] = num("player_mu_diff")
    df["roster_changed"] = num("roster_changed").fillna(0.0)
    df["map_index"] = num("map_index")
    df["league_strength"] = num("league_strength")
    df["playoffs"] = num("playoffs").fillna(0.0)

    df["gold10"] = num("blue_golddiffat10")
    df["gold15"] = num("blue_golddiffat15")
    for c in (
        "y_blue_win",
        "y_blue_firstblood",
        "y_blue_first_inhib",
        "blue_firstdragon",
        "blue_firstherald",
        "blue_firsttower",
        "blue_firstbaron",
    ):
        df[c] = num(c)

    bd, rd = num("blue_dragons").fillna(0), num("red_dragons").fillna(0)
    bt, rt = num("blue_towers").fillna(0), num("red_towers").fillna(0)
    bb, rb = num("blue_barons").fillna(0), num("red_barons").fillna(0)
    df["dragon_diff"] = (bd - rd).clip(-4, 4)
    df["tower_diff"] = (bt - rt).clip(-8, 8)
    df["baron_diff"] = (bb - rb).clip(-2, 2)
    df["total_kills"] = num("total_kills").fillna(num("y_total_kills"))
    df["length_min"] = num("length_min")
    if df["length_min"].isna().all():
        df["length_min"] = num("blue_gamelength") / 60.0
    df["ckpm"] = num("ckpm")
    if df["ckpm"].isna().mean() > 0.5:
        df["ckpm"] = df["total_kills"] / df["length_min"].clip(lower=1)
    df["under_29_5"] = (df["total_kills"] <= 29).astype(float)
    df["under_32_5"] = (df["total_kills"] <= 32).astype(float)
    df["long_game_35"] = (df["length_min"] >= 35).astype(float)

    # concentration from artifact
    conc_path = MODELS_DIR / "champ_kill_concentration.json"
    df["kill_conc_diff"] = np.nan
    df["max_carry_blue"] = np.nan
    df["scaling_flag"] = np.nan
    if conc_path.exists():
        from lol_kills.etl.aliases import normalize_champ
        from lol_kills.research.draft_advantage_matrix import draft_conc_features

        conc = json.loads(conc_path.read_text())
        p = players.copy()
        p["game_uid"] = p["game_uid"].astype(str)
        p["champion"] = p["champion"].map(lambda c: normalize_champ(str(c)))
        p["side"] = p["side"].astype(str).str.title()
        rows = []
        for gid, g in p.groupby("game_uid"):
            blue = g[g["side"] == "Blue"]["champion"].tolist()
            red = g[g["side"] == "Red"]["champion"].tolist()
            if len(blue) < 5 or len(red) < 5:
                continue
            cf = draft_conc_features(blue[:5], red[:5], conc)
            rows.append(
                {
                    "game_uid": gid,
                    "kill_conc_diff": cf["kill_conc_diff"],
                    "max_carry_blue": cf["max_carry_blue"],
                    "scaling_flag": cf["scaling_flag"],
                }
            )
        if rows:
            cdf = pd.DataFrame(rows)
            df = df.drop(columns=["kill_conc_diff", "max_carry_blue", "scaling_flag"], errors="ignore")
            df = df.merge(cdf, on="game_uid", how="left")

    return df


def pairwise(df: pd.DataFrame, cols: list[str], min_n: int = 120) -> list[dict]:
    present = [c for c in cols if c in df.columns]
    edges = []
    for i, a in enumerate(present):
        aa = df[a].astype(float).values
        for b in present[i + 1 :]:
            r, n = _r(aa, df[b].astype(float).values, min_n=min_n)
            if r is None:
                continue
            edges.append({"a": a, "b": b, "r": round(r, 4), "n": n, "abs_r": abs(r)})
    edges.sort(key=lambda x: -x["abs_r"])
    return edges


def residual_edges(
    df: pd.DataFrame,
    predictors: list[str],
    outcomes: list[str],
    controls: list[str],
    min_abs: float = 0.06,
) -> list[dict]:
    """Corr(pred_resid, outcome_resid) after OLS on controls — Elo-purged associations."""
    ctrl_cols = [c for c in controls if c in df.columns]
    if not ctrl_cols:
        return []
    X = df[ctrl_cols].astype(float).values
    out = []
    for pred in predictors:
        if pred not in df.columns or pred in ctrl_cols:
            continue
        pr = _residualize(df[pred].astype(float).values, X)
        for outc in outcomes:
            if outc not in df.columns or outc == pred:
                continue
            # Don't residualize binary win the same way for interpretability on under flags —
            # residualize outcome too so we get partial association
            ovr = _residualize(df[outc].astype(float).values, X)
            r, n = _r(pr, ovr, min_n=200)
            if r is None or abs(r) < min_abs:
                continue
            if frozenset({pred, outc}) in OBVIOUS:
                continue
            out.append(
                {
                    "predictor": pred,
                    "outcome": outc,
                    "r_residual": round(r, 4),
                    "n": n,
                    "controls": ctrl_cols,
                    "abs_r": abs(r),
                }
            )
    out.sort(key=lambda x: -x["abs_r"])
    return out


def close_game_slice(df: pd.DataFrame, thr: float = 80.0) -> pd.DataFrame:
    return df[df["mu_diff"].abs() <= thr].copy()


def betting_rankings(df: pd.DataFrame) -> dict:
    """Rank prematch features by |r| to each betting outcome."""
    targets = {
        "map_win_blue": "y_blue_win",
        "first_blood_blue": "y_blue_firstblood",
        "under_29_5": "under_29_5",
        "under_32_5": "under_32_5",
        "long_game_35": "long_game_35",
        "total_kills": "total_kills",
        "length_min": "length_min",
    }
    ranks = {}
    for name, col in targets.items():
        if col not in df.columns:
            continue
        rows = []
        y = df[col].astype(float).values
        for p in PREMATCH:
            if p not in df.columns:
                continue
            r, n = _r(df[p].astype(float).values, y, min_n=200)
            if r is None:
                continue
            rows.append({"feature": p, "r": round(r, 4), "n": n, "abs_r": abs(r)})
        rows.sort(key=lambda x: -x["abs_r"])
        ranks[name] = rows[:15]
    return ranks


def logistic_draft_pace_unders(df: pd.DataFrame) -> dict:
    """Elo + draft_kills_shift → P(under) — actionable O/U edge size."""
    sub = df.dropna(subset=["under_29_5", "mu_diff", "draft_kills_shift"]).copy()
    if len(sub) < 500:
        return {}
    X = np.column_stack([sub["mu_diff"].values / 400.0, sub["draft_kills_shift"].values])
    out = {}
    for line_col, label in (("under_29_5", "29.5"), ("under_32_5", "32.5")):
        y = sub[line_col].astype(float).values
        lr = LogisticRegression(C=1.0, max_iter=400)
        lr.fit(X, y)
        # effect of +1 kill draft shift on under prob at even elo
        base = float(1 / (1 + np.exp(-(lr.intercept_[0]))))
        up = float(1 / (1 + np.exp(-(lr.intercept_[0] + lr.coef_[0][1]))))
        out[label] = {
            "coef_elo_z": float(lr.coef_[0][0]),
            "coef_draft_kills_shift": float(lr.coef_[0][1]),
            "p_under_at_0_shift": round(base, 4),
            "p_under_at_plus1_shift": round(up, 4),
            "delta_pp_per_plus1_shift": round((up - base) * 100, 2),
            "n": int(len(sub)),
        }
    return out


def flag_edges(edges: list[dict], min_abs: float = 0.08) -> list[dict]:
    interesting = []
    for e in edges:
        if e["abs_r"] < min_abs:
            continue
        key = frozenset({e["a"], e["b"]})
        if key in OBVIOUS:
            continue
        # skip near-duplicate elo family
        if {e["a"], e["b"]} <= {"mu_diff", "p_dual_elo", "player_mu_diff", "elo_diff"}:
            continue
        interesting.append({**e, "why": _why(e["a"], e["b"], e["r"])})
    return interesting


def _why(a: str, b: str, r: float) -> str:
    pair = {a, b}
    sign = "↑ together" if r > 0 else "inverse"
    if "draft_kills_shift" in pair and ("under_" in b or "under_" in a or "total_kills" in pair):
        return f"pace draft → kills O/U ({sign})"
    if "abs_draft_edge" in pair and ("length" in a + b or "long_game" in a + b or "under_" in a + b):
        return f"|draft edge| vs length/kills ({sign}) — blowout vs slog"
    if "form_ckpm_avg" in pair and ("total_kills" in pair or "under_" in a + b or "ckpm" in pair):
        return f"team pace form → totals ({sign})"
    if "sigma_pair" in pair:
        return f"rating uncertainty ({sign}) — fade/lean upset or variance props"
    if "map_index" in pair:
        return f"series map number ({sign})"
    if "roll_g10_diff" in pair and "gold" in a + b:
        return "recent gold form persists"
    if "roll_fb_diff" in pair and "firstblood" in a + b:
        return "FB form → FB"
    if "h2h_blue_edge" in pair:
        return f"H2H residual ({sign})"
    if "kill_conc_diff" in pair or "scaling_flag" in pair or "max_carry" in a + b:
        return f"carry concentration / scaling ({sign})"
    if "roster_changed" in pair:
        return f"roster change noise ({sign})"
    if "playoffs" in pair:
        return f"playoffs regime ({sign})"
    if "form_wr_diff" in pair and "y_blue_win" in pair:
        return "form WR (known but tradeable)"
    if "draft_win_logit_blue" in pair and "y_blue_firstblood" in pair:
        return "draft → FB"
    return f"non-obvious ({sign})"


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[corr] loading expanded frame…")
    df = load_frame()
    print(f"[corr] n={len(df)}")

    all_cols = [c for c in PREMATCH + OUTCOMES if c in df.columns]
    raw_edges = pairwise(df, all_cols)
    unexpected_raw = flag_edges(raw_edges, min_abs=0.07)[:50]

    print("[corr] residual (Elo-controlled)…")
    resid = residual_edges(
        df,
        predictors=[c for c in PREMATCH if c not in ("mu_diff", "p_dual_elo", "player_mu_diff")],
        outcomes=[
            "y_blue_win",
            "y_blue_firstblood",
            "under_29_5",
            "under_32_5",
            "total_kills",
            "length_min",
            "long_game_35",
            "gold15",
            "ckpm",
        ],
        controls=["mu_diff"],
        min_abs=0.05,
    )[:40]

    print("[corr] close games |μ|<80…")
    close = close_game_slice(df, 80.0)
    close_edges = flag_edges(pairwise(close, all_cols, min_n=80), min_abs=0.08)[:30]

    ranks = betting_rankings(df)
    ranks_close = betting_rankings(close)
    pace = logistic_draft_pace_unders(df)

    # Compound score: residual edges toward tradable outcomes
    tradable = {"y_blue_win", "under_29_5", "under_32_5", "total_kills", "y_blue_firstblood", "long_game_35", "length_min"}
    compound = [e for e in resid if e["outcome"] in tradable]
    compound.sort(key=lambda x: -x["abs_r"])

    report = {
        "version": 2,
        "n_maps": int(len(df)),
        "n_close_games": int(len(close)),
        "metrics": all_cols,
        "unexpected_raw_top": unexpected_raw,
        "elo_residual_edges": resid,
        "close_game_unexpected": close_edges,
        "betting_feature_ranks": ranks,
        "betting_feature_ranks_close_elo": ranks_close,
        "draft_pace_under_model": pace,
        "compound_edge_board": compound[:25],
        "notes": [
            "First inhib omitted from residual hunt as a primary target — label ≈ win (r~0.97).",
            "Residual r is after linear Elo control; small |r| still compounds in a log if stable +EV.",
            "Prefer prematch predictors; gold/tower are live/post confirmation only.",
        ],
    }
    path = MODELS_DIR / "metric_correlation_matrix.json"
    path.write_text(json.dumps(report, indent=2))
    # keep a slim readable board
    board_path = MODELS_DIR / "unexpected_edges_board.json"
    board_path.write_text(
        json.dumps(
            {
                "compound_edge_board": compound[:25],
                "draft_pace_under_model": pace,
                "top_under_29_5_features": ranks.get("under_29_5", [])[:10],
                "top_map_win_features_residual_hint": [
                    e for e in resid if e["outcome"] == "y_blue_win"
                ][:10],
                "top_fb_features": ranks.get("first_blood_blue", [])[:8],
            },
            indent=2,
        )
    )
    print(f"[corr] wrote {path}")
    print(f"[corr] wrote {board_path}")
    print("\n=== COMPOUND EDGE BOARD (Elo-residual) ===")
    for e in compound[:15]:
        print(f"  {e['predictor']:24} → {e['outcome']:16} r={e['r_residual']:+.3f} n={e['n']}")
    print("\n=== DRAFT PACE → UNDER ===")
    print(json.dumps(pace, indent=2))
    print("\n=== TOP UNDER 29.5 raw features ===")
    for x in ranks.get("under_29_5", [])[:8]:
        print(f"  {x['feature']:24} r={x['r']:+.3f}")


if __name__ == "__main__":
    main()
