#!/usr/bin/env python3
"""
3-grub era isolation study (IMLS-ready).

Filters to maps where void grubs sum to exactly 3 (post-6-grub patch).
Isolates unique map-WR of a 3–0 sweep after game-state controls, then
measures how often the @10 underdog still sweeps and what that does to WR.

Hard limit: OE has no fight logs. P̂_fight is a gold@10+kills@10 strength
proxy — labeled as such everywhere.

  python3 -m lol_kills.research.grubs_isolation_study
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

from lol_kills.draft_archetypes import draft_archetype_features
from lol_kills.etl.paths import MODELS_DIR, RAW_OE_DIR, WAREHOUSE_DIR
from lol_kills.etl.aliases import normalize_champ
from lol_kills.research.side_objective_edges import engineer

MIN_N = 80
GOLD_BINS = [
    ("behind2k", -1e9, -2000),
    ("behind", -2000, -500),
    ("even", -500, 500),
    ("ahead", 500, 2000),
    ("ahead2k", 2000, 1e9),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_oe_team_maps_extended() -> pd.DataFrame:
    """Like side_objective_edges.load_oe_team_maps but with @10 combat cols."""
    files = sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    want = {
        "gameid",
        "date",
        "league",
        "year",
        "split",
        "playoffs",
        "patch",
        "side",
        "position",
        "teamname",
        "result",
        "gamelength",
        "kills",
        "deaths",
        "assists",
        "teamkills",
        "firstblood",
        "firstdragon",
        "firstherald",
        "firstbaron",
        "firsttower",
        "dragons",
        "barons",
        "towers",
        "heralds",
        "void_grubs",
        "elders",
        "elementaldrakes",
        "infernals",
        "mountains",
        "clouds",
        "oceans",
        "hextechs",
        "chemtechs",
        "golddiffat10",
        "golddiffat15",
        "golddiffat20",
        "golddiffat25",
        "xpdiffat10",
        "xpdiffat15",
        "xpdiffat20",
        "killsat10",
        "killsat15",
        "killsat20",
        "ckpm",
    }
    frames = []
    for fp in files:
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        usecols = [c for c in hdr if c in want]
        df = pd.read_csv(fp, usecols=usecols, low_memory=False)
        df = df[df["position"].astype(str).str.lower() == "team"].copy()
        df["oe_year"] = int(fp.name[:4])
        frames.append(df)
        print(f"[grubs3] loaded {fp.name} team_rows={len(df)}")
    raw = pd.concat(frames, ignore_index=True)
    raw["side"] = raw["side"].astype(str).str.title()
    raw["gameid"] = raw["gameid"].astype(str)

    def pivot_side(side: str, prefix: str) -> pd.DataFrame:
        s = raw[raw["side"] == side].copy()
        s = s.drop_duplicates("gameid", keep="first")
        rename = {c: f"{prefix}{c}" for c in s.columns if c not in ("gameid", "side", "position")}
        return s.rename(columns=rename)

    blue = pivot_side("Blue", "blue_")
    red = pivot_side("Red", "red_")
    m = blue.merge(red, on="gameid", how="inner")
    m = m.rename(
        columns={
            "blue_date": "date",
            "blue_league": "league",
            "blue_year": "year",
            "blue_patch": "patch",
            "blue_playoffs": "playoffs",
            "blue_oe_year": "oe_year",
            "blue_gamelength": "gamelength",
            "blue_result": "y_blue_win",
        }
    )
    m["date"] = pd.to_datetime(m["date"], errors="coerce")
    m["y_blue_win"] = pd.to_numeric(m["y_blue_win"], errors="coerce")
    m["length_min"] = pd.to_numeric(m["gamelength"], errors="coerce") / 60.0
    m["total_kills"] = pd.to_numeric(m["blue_kills"], errors="coerce") + pd.to_numeric(
        m["red_kills"], errors="coerce"
    )
    return m.dropna(subset=["y_blue_win", "date"]).sort_values("date")


def load_drafts() -> pd.DataFrame:
    """gameid → blue_champs / red_champs lists from player rows."""
    files = sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    frames = []
    for fp in files:
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        need = ["gameid", "side", "position", "champion"]
        if not all(c in hdr for c in need):
            continue
        df = pd.read_csv(fp, usecols=need, low_memory=False)
        df = df[df["position"].astype(str).str.lower().isin(["top", "jng", "mid", "bot", "sup"])].copy()
        df["gameid"] = df["gameid"].astype(str)
        df["side"] = df["side"].astype(str).str.title()
        df["champion"] = df["champion"].astype(str)
        frames.append(df)
        print(f"[grubs3] drafts {fp.name} player_rows={len(df)}")
    if not frames:
        return pd.DataFrame(columns=["gameid", "blue_champs", "red_champs"])
    raw = pd.concat(frames, ignore_index=True)

    def side_champs(g: pd.DataFrame, side: str) -> list[str]:
        sub = g[g["side"] == side]
        out = []
        for c in sub["champion"]:
            if not c or c == "nan":
                continue
            try:
                out.append(normalize_champ(c))
            except Exception:
                out.append(str(c))
        return out

    rows = []
    for gid, g in raw.groupby("gameid"):
        b, r = side_champs(g, "Blue"), side_champs(g, "Red")
        if len(b) < 5 or len(r) < 5:
            continue
        rows.append({"gameid": gid, "blue_champs": b[:5], "red_champs": r[:5]})
    return pd.DataFrame(rows)


def merge_elo(df: pd.DataFrame) -> pd.DataFrame:
    feat_path = WAREHOUSE_DIR / "hf" / "features.parquet"
    if not feat_path.exists():
        df["elo_diff"] = np.nan
        df["p_dual_elo"] = np.nan
        return df
    feat = pd.read_parquet(feat_path, columns=["gameid", "elo_diff"])
    feat["gameid"] = feat["gameid"].astype(str)
    out = df.merge(feat.drop_duplicates("gameid"), on="gameid", how="left")
    # dual-elo style prior from elo_diff (same scale as elsewhere ~ /400)
    ed = pd.to_numeric(out["elo_diff"], errors="coerce")
    out["p_dual_elo"] = 1.0 / (1.0 + np.exp(-ed / 400.0 * 2.2))
    return out


def engineer_study(df: pd.DataFrame) -> pd.DataFrame:
    d = engineer(df)
    for col in (
        "blue_killsat10",
        "red_killsat10",
        "blue_xpdiffat10",
        "red_xpdiffat10",
        "blue_killsat15",
        "red_killsat15",
    ):
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        else:
            d[col] = np.nan

    # OE stores golddiff/xpdiff/killsat on each team row as that team's perspective
    # After pivot, blue_golddiffat10 is blue's gold lead (standard OE)
    d["gold10"] = pd.to_numeric(d.get("blue_golddiffat10"), errors="coerce")
    d["gold15"] = pd.to_numeric(d.get("blue_golddiffat15"), errors="coerce")
    d["xp10"] = pd.to_numeric(d.get("blue_xpdiffat10"), errors="coerce")
    d["kills10_diff"] = pd.to_numeric(d.get("blue_killsat10"), errors="coerce") - pd.to_numeric(
        d.get("red_killsat10"), errors="coerce"
    )
    # Some OE dumps put killsat as absolute; if blue_killsat10 is NaN use gold only
    d["kills10_blue"] = pd.to_numeric(d.get("blue_killsat10"), errors="coerce")
    d["kills10_red"] = pd.to_numeric(d.get("red_killsat10"), errors="coerce")

    bg = d["blue_void_grubs"].fillna(0)
    rg = d["red_void_grubs"].fillna(0)
    d["grub_sum"] = bg + rg
    d["blue_all3"] = ((bg == 3) & (rg == 0)).astype(float)
    d["red_all3"] = ((rg == 3) & (bg == 0)).astype(float)
    d["all3_signed"] = d["blue_all3"] - d["red_all3"]  # +1 blue sweep, -1 red, 0 else
    d["blue_majority"] = (bg >= 2).astype(float)
    d["blue_firstherald"] = pd.to_numeric(d.get("blue_firstherald"), errors="coerce")
    d["blue_firsttower"] = pd.to_numeric(d.get("blue_firsttower"), errors="coerce")

    # gold@10 bins
    def bin_gold(x: float) -> str:
        if not np.isfinite(x):
            return "missing"
        for name, lo, hi in GOLD_BINS:
            if lo <= x < hi:
                return name
        return "ahead2k"

    d["gold10_bin"] = d["gold10"].map(bin_gold)
    return d


def attach_draft_features(df: pd.DataFrame, drafts: pd.DataFrame) -> pd.DataFrame:
    if drafts.empty:
        df["arch_scale_vs_snowball"] = np.nan
        df["scaling_fav_blue"] = np.nan
        return df
    m = df.merge(drafts, on="gameid", how="left")
    scale = []
    fav = []
    for _, r in m.iterrows():
        b, rd = r.get("blue_champs"), r.get("red_champs")
        if not isinstance(b, list) or not isinstance(rd, list):
            scale.append(np.nan)
            fav.append(np.nan)
            continue
        feats = draft_archetype_features(b, rd)
        s = float(feats["arch_scale_vs_snowball"])
        scale.append(s)
        fav.append(1.0 if s > 0 else (0.0 if s < 0 else 0.5))
    m["arch_scale_vs_snowball"] = scale
    m["scaling_fav_blue"] = fav
    return m


def filter_3grub(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["grub_sum"] == 3].copy()


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------


def _partial_corr(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> dict:
    """Partial corr of x,y controlling for Z columns."""
    n, k = len(y), Z.shape[1]
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = y - LinearRegression().fit(Z, y).predict(Z)
    if float(np.std(rx)) < 1e-12 or float(np.std(ry)) < 1e-12:
        return {"partial_r": None, "p": None, "n": n, "k": k}
    pr = float(np.corrcoef(rx, ry)[0, 1])
    df = n - k - 2
    if df <= 0 or abs(pr) >= 1:
        p = None
    else:
        t = pr * math.sqrt(df / max(1e-15, 1 - pr * pr))
        p = float(2 * stats.t.sf(abs(t), df))
    return {"partial_r": pr, "p": p, "n": n, "k": k, "df": df}


def _nested_lr(y: np.ndarray, Z: np.ndarray, x: np.ndarray) -> dict:
    """LR test: Z vs Z+x. Also unique Δpp at mean(Z)."""
    lr1 = LogisticRegression(C=1e6, max_iter=2000).fit(Z, y)
    Zx = np.column_stack([Z, x])
    lr2 = LogisticRegression(C=1e6, max_iter=2000).fit(Zx, y)
    p1 = lr1.predict_proba(Z)[:, 1]
    p2 = lr2.predict_proba(Zx)[:, 1]
    ll1 = -log_loss(y, np.clip(p1, 1e-6, 1 - 1e-6), normalize=False)
    ll2 = -log_loss(y, np.clip(p2, 1e-6, 1 - 1e-6), normalize=False)
    lr_stat = float(2 * (ll2 - ll1))
    p_lr = float(stats.chi2.sf(lr_stat, 1))
    coef = float(lr2.coef_[0][-1])
    Zm = Z.mean(axis=0, keepdims=True)

    def pred(xv: float) -> float:
        return float(lr2.predict_proba(np.column_stack([Zm, [[xv]]]))[0, 1])

    p0, p1v = pred(0.0), pred(1.0)
    # bootstrap CI on unique dpp (light)
    rng = np.random.default_rng(42)
    boots = []
    n = len(y)
    for _ in range(200):
        idx = rng.integers(0, n, n)
        try:
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(Zx[idx], y[idx])
            Zm_b = Z[idx].mean(axis=0, keepdims=True)
            a = float(lr.predict_proba(np.column_stack([Zm_b, [[0.0]]]))[0, 1])
            b = float(lr.predict_proba(np.column_stack([Zm_b, [[1.0]]]))[0, 1])
            boots.append((b - a) * 100)
        except Exception:
            continue
    ci = (
        (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
        if len(boots) >= 50
        else (None, None)
    )
    return {
        "lr_stat": lr_stat,
        "lr_p": p_lr,
        "coef": coef,
        "odds_ratio": float(math.exp(coef)),
        "pred_at_mean_x0": p0,
        "pred_at_mean_x1": p1v,
        "unique_dpp": (p1v - p0) * 100,
        "unique_dpp_ci95": ci,
        "n": int(n),
    }


def _wr(mask: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> dict | None:
    m = np.asarray(mask, dtype=bool)
    yy = np.asarray(y, dtype=float)
    n = int(m.sum())
    if n < MIN_N:
        return {"n": n, "wr": None}
    return {"n": n, "wr": float(yy[m].mean())}


# ---------------------------------------------------------------------------
# Block A — controlled isolation
# ---------------------------------------------------------------------------


def raw_ladder(df: pd.DataFrame) -> dict:
    rows = []
    for k in range(0, 4):
        sub = df[df["blue_void_grubs"].fillna(-1) == k]
        if len(sub) < 40:
            continue
        rows.append(
            {
                "blue_grubs": k,
                "n": int(len(sub)),
                "wr_blue": float(sub["y_blue_win"].mean()),
                "mean_gold10": float(sub["gold10"].mean()) if sub["gold10"].notna().any() else None,
            }
        )
    # sweeper-side
    sweep = pd.concat(
        [
            df[df["blue_all3"] == 1].assign(sweeper_win=lambda x: x["y_blue_win"]),
            df[df["red_all3"] == 1].assign(sweeper_win=lambda x: 1 - x["y_blue_win"]),
        ],
        ignore_index=True,
    )
    base = float(df["y_blue_win"].mean())
    return {
        "blue_grub_count_ladder": rows,
        "baseline_blue_wr": base,
        "sweeper_all3": {
            "n": int(len(sweep)),
            "wr": float(sweep["sweeper_win"].mean()) if len(sweep) else None,
            "raw_dpp_vs_blue_baseline": (
                (float(sweep["sweeper_win"].mean()) - base) * 100 if len(sweep) else None
            ),
        },
        "blue_all3": {
            "n": int((df["blue_all3"] == 1).sum()),
            "wr": float(df.loc[df["blue_all3"] == 1, "y_blue_win"].mean())
            if (df["blue_all3"] == 1).any()
            else None,
            "raw_dpp_vs_baseline": (
                (float(df.loc[df["blue_all3"] == 1, "y_blue_win"].mean()) - base) * 100
                if (df["blue_all3"] == 1).any()
                else None
            ),
        },
    }


def control_ladder(df: pd.DataFrame) -> dict:
    """Nested controls on blue_all3 → y_blue_win."""
    need = ["y_blue_win", "blue_all3", "gold10", "kills10_diff", "xp10", "gold15", "blue_firstherald", "blue_firsttower"]
    sub = df.dropna(subset=[c for c in need if c in df.columns]).copy()
    y = sub["y_blue_win"].astype(float).values
    x = sub["blue_all3"].astype(float).values

    steps = []

    def run(name: str, cols: list[str], extra: np.ndarray | None = None):
        mats = []
        for c in cols:
            if c not in sub.columns or sub[c].isna().all():
                return
            v = sub[c].astype(float).values
            if c.startswith("gold") or c == "xp10":
                v = v / 1000.0
            mats.append(v)
        if extra is not None:
            mats.append(extra)
        if not mats:
            return
        Z = np.column_stack(mats)
        # drop rows with nan in Z
        mask = np.isfinite(Z).all(axis=1) & np.isfinite(y) & np.isfinite(x)
        if mask.sum() < MIN_N:
            return
        pc = _partial_corr(y[mask], x[mask], Z[mask])
        lr = _nested_lr(y[mask], Z[mask], x[mask])
        steps.append({"step": name, "controls": cols + (["extra"] if extra is not None else []), **pc, **{k: lr[k] for k in lr if k != "n"}, "n": lr["n"]})

    # 1 raw: intercept-only via constant control of zeros — report simple WR delta
    steps.append(
        {
            "step": "1_raw_side",
            "controls": [],
            "partial_r": float(np.corrcoef(x, y)[0, 1]) if len(y) > 2 else None,
            "p": None,
            "unique_dpp": (
                float(sub.loc[sub["blue_all3"] == 1, "y_blue_win"].mean())
                - float(sub.loc[sub["blue_all3"] == 0, "y_blue_win"].mean())
            )
            * 100,
            "n": int(len(sub)),
            "note": "WR(blue_all3=1) − WR(blue_all3=0)",
        }
    )
    run("2_gold_kills_xp_at10", ["gold10", "kills10_diff", "xp10"])
    run("3_plus_gold15_fh_ft", ["gold10", "kills10_diff", "xp10", "gold15", "blue_firstherald", "blue_firsttower"])

    if "elo_diff" in sub.columns and sub["elo_diff"].notna().sum() >= MIN_N:
        run(
            "4_plus_elo",
            ["gold10", "kills10_diff", "xp10", "gold15", "blue_firstherald", "blue_firsttower", "elo_diff"],
        )
    if "arch_scale_vs_snowball" in sub.columns and sub["arch_scale_vs_snowball"].notna().sum() >= MIN_N:
        run(
            "5_plus_scaling",
            [
                "gold10",
                "kills10_diff",
                "xp10",
                "gold15",
                "blue_firstherald",
                "blue_firsttower",
                "arch_scale_vs_snowball",
            ],
        )
        if "elo_diff" in sub.columns and sub["elo_diff"].notna().sum() >= MIN_N:
            run(
                "5b_full",
                [
                    "gold10",
                    "kills10_diff",
                    "xp10",
                    "gold15",
                    "blue_firstherald",
                    "blue_firsttower",
                    "elo_diff",
                    "arch_scale_vs_snowball",
                ],
            )

    # majority secondary
    maj = sub.dropna(subset=["gold10", "gold15", "blue_firstherald", "blue_firsttower"]).copy()
    maj_x = maj["blue_majority"].astype(float).values
    maj_y = maj["y_blue_win"].astype(float).values
    Z = np.column_stack(
        [
            maj["gold10"].values / 1000,
            maj["gold15"].values / 1000,
            maj["blue_firstherald"].fillna(0).values,
            maj["blue_firsttower"].fillna(0).values,
        ]
    )
    mask = np.isfinite(Z).all(axis=1)
    maj_lr = _nested_lr(maj_y[mask], Z[mask], maj_x[mask]) if mask.sum() >= MIN_N else {}

    # strata by gold10
    strata = []
    for name, lo, hi in GOLD_BINS:
        g = sub[(sub["gold10"] >= lo) & (sub["gold10"] < hi)]
        if len(g) < 60:
            continue
        w_all = g[g["blue_all3"] == 1]
        w_not = g[g["blue_all3"] == 0]
        if len(w_all) < 25 or len(w_not) < 25:
            continue
        strata.append(
            {
                "bin": name,
                "n": int(len(g)),
                "n_all3": int(len(w_all)),
                "wr_all3": float(w_all["y_blue_win"].mean()),
                "wr_not": float(w_not["y_blue_win"].mean()),
                "dpp": (float(w_all["y_blue_win"].mean()) - float(w_not["y_blue_win"].mean())) * 100,
                "mean_gold10": float(g["gold10"].mean()),
            }
        )

    # IPW-ish: match gold10 ±750
    matched = _matched_delta(sub)

    headline = next((s for s in steps if s["step"] == "3_plus_gold15_fh_ft"), steps[-1] if steps else {})
    return {
        "n_complete": int(len(sub)),
        "nested_steps": steps,
        "majority_vs_gold15_fh_ft": maj_lr,
        "gold10_strata": strata,
        "matched_gold10": matched,
        "headline_unique_dpp": headline.get("unique_dpp"),
        "headline_lr_p": headline.get("lr_p"),
        "headline_partial_r": headline.get("partial_r"),
        "headline_ci95": headline.get("unique_dpp_ci95"),
    }


def _matched_delta(sub: pd.DataFrame, tol: float = 750.0) -> dict:
    """1:1 nearest gold10 match: treated blue_all3 vs control."""
    treated = sub[sub["blue_all3"] == 1].dropna(subset=["gold10", "y_blue_win"])
    control = sub[sub["blue_all3"] == 0].dropna(subset=["gold10", "y_blue_win"])
    if len(treated) < MIN_N or len(control) < MIN_N:
        return {"n_pairs": 0}
    c_gold = control["gold10"].values
    c_y = control["y_blue_win"].values
    used = np.zeros(len(control), dtype=bool)
    pairs_t, pairs_c = [], []
    for _, r in treated.iterrows():
        diffs = np.abs(c_gold - r["gold10"])
        diffs[used] = 1e18
        j = int(np.argmin(diffs))
        if diffs[j] > tol:
            continue
        used[j] = True
        pairs_t.append(float(r["y_blue_win"]))
        pairs_c.append(float(c_y[j]))
    if len(pairs_t) < MIN_N:
        return {"n_pairs": len(pairs_t)}
    return {
        "n_pairs": len(pairs_t),
        "wr_treated": float(np.mean(pairs_t)),
        "wr_control": float(np.mean(pairs_c)),
        "dpp": (float(np.mean(pairs_t)) - float(np.mean(pairs_c))) * 100,
        "tol_gold": tol,
    }


# ---------------------------------------------------------------------------
# Block B — underdog vs favorite
# ---------------------------------------------------------------------------


def fit_fight_proxy(df: pd.DataFrame) -> tuple[dict, np.ndarray]:
    """
    Fit sigmoid P̂_fight_blue from gold10 + kills10_diff.
    Calibrated to predict *first tower* (next-obj combat proxy), not map win —
    reduces circularity with final WR. Fallback: calibrate to map win if FT sparse.
    """
    d = df.reset_index(drop=True)
    sub = d.dropna(subset=["gold10", "kills10_diff", "blue_firsttower", "y_blue_win"]).copy()
    X = np.column_stack([sub["gold10"].values / 1000.0, sub["kills10_diff"].fillna(0).values])
    y_ft = sub["blue_firsttower"].astype(float).values
    if y_ft.std() > 0.05 and len(sub) >= 500:
        target = "first_tower"
        y = y_ft
    else:
        target = "map_win_fallback"
        y = sub["y_blue_win"].astype(float).values
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    p = lr.predict_proba(X)[:, 1]
    p_full = np.full(len(d), np.nan)
    p_full[sub.index.to_numpy()] = p
    # also score rows missing FT but with gold using the same model
    miss = d.index.difference(sub.index)
    if len(miss):
        mm = d.loc[miss].dropna(subset=["gold10"])
        if len(mm):
            Xm = np.column_stack(
                [mm["gold10"].values / 1000.0, mm["kills10_diff"].fillna(0).values]
            )
            p_full[mm.index.to_numpy()] = lr.predict_proba(Xm)[:, 1]
    meta = {
        "target": target,
        "coef_gold_per_1k": float(lr.coef_[0][0]),
        "coef_kills10_diff": float(lr.coef_[0][1]),
        "intercept": float(lr.intercept_[0]),
        "n_fit": int(len(sub)),
        "note": "P̂_fight is a combat-strength proxy from gold@10 + kills@10; not observed fight win%.",
    }
    return meta, p_full


def block_b_underdog(df: pd.DataFrame, p_fight: np.ndarray) -> dict:
    d = df.reset_index(drop=True).copy()
    d["p_fight_blue"] = p_fight[: len(d)]
    d = d.dropna(subset=["p_fight_blue", "gold10", "y_blue_win"]).copy()

    d["fight_fav_blue"] = (d["p_fight_blue"] > 0.5).astype(float)
    # underdog = not fight favorite
    d["blue_underdog"] = (d["p_fight_blue"] < 0.5).astype(float)
    d["blue_favorite"] = (d["p_fight_blue"] > 0.5).astype(float)

    # take rates
    und = d[d["blue_underdog"] == 1]
    fav = d[d["blue_favorite"] == 1]
    take = {
        "underdog_blue_all3_rate": float(und["blue_all3"].mean()) if len(und) else None,
        "favorite_blue_all3_rate": float(fav["blue_all3"].mean()) if len(fav) else None,
        "n_underdog": int(len(und)),
        "n_favorite": int(len(fav)),
        "underdog_still_sweeps_n": int((und["blue_all3"] == 1).sum()),
        "favorite_sweeps_n": int((fav["blue_all3"] == 1).sum()),
        # red underdog sweeps (= blue favorite fails, red all3)
        "underdog_red_all3_rate": float(fav["red_all3"].mean()) if len(fav) else None,
    }

    def delta_in_group(g: pd.DataFrame, treat_col: str) -> dict:
        a = g[g[treat_col] == 1]
        b = g[g[treat_col] == 0]
        if len(a) < 40 or len(b) < 40:
            return {"n_treat": int(len(a)), "n_ctrl": int(len(b)), "dpp": None}
        return {
            "n_treat": int(len(a)),
            "n_ctrl": int(len(b)),
            "wr_treat": float(a["y_blue_win"].mean()),
            "wr_ctrl": float(b["y_blue_win"].mean()),
            "dpp": (float(a["y_blue_win"].mean()) - float(b["y_blue_win"].mean())) * 100,
            "mean_gold10_treat": float(a["gold10"].mean()),
            "mean_gold10_ctrl": float(b["gold10"].mean()),
        }

    # Matched within gold10 bin
    und_by_bin = []
    for name, lo, hi in GOLD_BINS:
        g = und[(und["gold10"] >= lo) & (und["gold10"] < hi)]
        if len(g) < 50:
            continue
        row = delta_in_group(g, "blue_all3")
        row["bin"] = name
        und_by_bin.append(row)

    fav_by_bin = []
    for name, lo, hi in GOLD_BINS:
        g = fav[(fav["gold10"] >= lo) & (fav["gold10"] < hi)]
        if len(g) < 50:
            continue
        row = delta_in_group(g, "blue_all3")
        row["bin"] = name
        fav_by_bin.append(row)

    # Sweeper-side: underdog team (whichever) sweeps
    # Define underdog team as the one with p_fight < 0.5
    d["underdog_swept"] = np.where(
        d["blue_underdog"] == 1, d["blue_all3"], d["red_all3"]
    )
    d["favorite_swept"] = np.where(
        d["blue_favorite"] == 1, d["blue_all3"], d["red_all3"]
    )
    d["underdog_won"] = np.where(d["blue_underdog"] == 1, d["y_blue_win"], 1 - d["y_blue_win"])
    d["favorite_won"] = np.where(d["blue_favorite"] == 1, d["y_blue_win"], 1 - d["y_blue_win"])

    und_sweep = d[d["underdog_swept"] == 1]

    underdog_effect = {
        "wr_if_underdog_sweeps": float(und_sweep["underdog_won"].mean()) if len(und_sweep) >= 40 else None,
        "n_underdog_sweeps": int(len(und_sweep)),
        "wr_if_underdog_does_not_sweep": float(
            d.loc[d["underdog_swept"] == 0, "underdog_won"].mean()
        )
        if (d["underdog_swept"] == 0).sum() >= 40
        else None,
        "n_underdog_no_sweep": int((d["underdog_swept"] == 0).sum()),
    }
    if underdog_effect["wr_if_underdog_sweeps"] is not None and underdog_effect[
        "wr_if_underdog_does_not_sweep"
    ] is not None:
        underdog_effect["dpp"] = (
            underdog_effect["wr_if_underdog_sweeps"] - underdog_effect["wr_if_underdog_does_not_sweep"]
        ) * 100

    fav_sweep = d[d["favorite_swept"] == 1]
    favorite_effect = {
        "wr_if_favorite_sweeps": float(fav_sweep["favorite_won"].mean()) if len(fav_sweep) >= 40 else None,
        "n_favorite_sweeps": int(len(fav_sweep)),
        "wr_if_favorite_does_not_sweep": float(
            d.loc[d["favorite_swept"] == 0, "favorite_won"].mean()
        )
        if (d["favorite_swept"] == 0).sum() >= 40
        else None,
        "n_favorite_no_sweep": int((d["favorite_swept"] == 0).sum()),
    }
    if favorite_effect["wr_if_favorite_sweeps"] is not None and favorite_effect[
        "wr_if_favorite_does_not_sweep"
    ] is not None:
        favorite_effect["dpp"] = (
            favorite_effect["wr_if_favorite_sweeps"] - favorite_effect["wr_if_favorite_does_not_sweep"]
        ) * 100

    # Scaling control story: fight-fav takes 3, scaling-fav wins
    scaling_story = {}
    if "scaling_fav_blue" in d.columns and d["scaling_fav_blue"].notna().any():
        # fight fav blue, took all3, but scaling favors red (scaling_fav_blue < 0.5)
        mask = (d["fight_fav_blue"] == 1) & (d["blue_all3"] == 1) & (d["scaling_fav_blue"] < 0.5)
        g = d[mask]
        scaling_story["fight_fav_took_grubs_but_scaling_fav_red"] = {
            "n": int(len(g)),
            "wr_blue": float(g["y_blue_win"].mean()) if len(g) >= 30 else None,
            "wr_scaling_side": float(1 - g["y_blue_win"].mean()) if len(g) >= 30 else None,
        }
        # within scaling-behind (blue is fight fav but scales worse): lead+grubs vs lead-no-grubs
        behind_scale = d[(d["fight_fav_blue"] == 1) & (d["scaling_fav_blue"] < 0.5)]
        with_g = behind_scale[behind_scale["blue_all3"] == 1]
        without = behind_scale[behind_scale["blue_all3"] == 0]
        scaling_story["fight_fav_but_worse_scaling"] = {
            "n_with_grubs": int(len(with_g)),
            "n_without": int(len(without)),
            "wr_with_grubs": float(with_g["y_blue_win"].mean()) if len(with_g) >= 40 else None,
            "wr_without_grubs": float(without["y_blue_win"].mean()) if len(without) >= 40 else None,
        }
        if (
            scaling_story["fight_fav_but_worse_scaling"]["wr_with_grubs"] is not None
            and scaling_story["fight_fav_but_worse_scaling"]["wr_without_grubs"] is not None
        ):
            scaling_story["fight_fav_but_worse_scaling"]["dpp_grubs"] = (
                scaling_story["fight_fav_but_worse_scaling"]["wr_with_grubs"]
                - scaling_story["fight_fav_but_worse_scaling"]["wr_without_grubs"]
            ) * 100

        # How often scaling favorite wins despite fight-fav getting grubs
        both = d[(d["favorite_swept"] == 1) & (d["scaling_fav_blue"].notna())]
        # scaling fav wins: if scaling_fav_blue>0.5 and blue wins, or <0.5 and red wins
        scale_wins = np.where(
            both["scaling_fav_blue"] > 0.5,
            both["y_blue_win"],
            np.where(both["scaling_fav_blue"] < 0.5, 1 - both["y_blue_win"], np.nan),
        )
        valid = np.isfinite(scale_wins)
        scaling_story["when_fight_fav_got_grubs"] = {
            "n": int(valid.sum()),
            "scaling_favorite_still_wins_rate": float(np.nanmean(scale_wins)) if valid.sum() >= 40 else None,
        }

    return {
        "take_rates": take,
        "blue_underdog_all3_by_gold10_bin": und_by_bin,
        "blue_favorite_all3_by_gold10_bin": fav_by_bin,
        "underdog_team_sweep_effect": underdog_effect,
        "favorite_team_sweep_effect": favorite_effect,
        "scaling_control_story": scaling_story,
        "mean_p_fight_when_underdog_sweeps": float(und_sweep["p_fight_blue"].mean())
        if len(und_sweep)
        else None,
    }


# ---------------------------------------------------------------------------
# Block C — contest EV table
# ---------------------------------------------------------------------------


def contest_ev_table(block_a: dict, block_b: dict, fight_meta: dict) -> dict:
    """
    EV_contest ≈ p * V_win − (1−p) * V_gift
    V_win  = underdog dpp from sweeping (block B)
    V_gift = favorite dpp from sweeping (cost of giving it away ≈ that value)
    V_unique = headline controlled unique dpp (block A) as alternate value
    """
    v_und = (block_b.get("underdog_team_sweep_effect") or {}).get("dpp")
    v_fav = (block_b.get("favorite_team_sweep_effect") or {}).get("dpp")
    v_unique = block_a.get("headline_unique_dpp")
    ci = block_a.get("headline_ci95")

    # Use absolute value gifted: if fav gets +Xpp from sweep, dog loses that opportunity
    rows = []
    for p in (0.30, 0.40, 0.50, 0.60, 0.70):
        row: dict[str, Any] = {"p_fight_proxy": p}
        if v_und is not None and v_fav is not None:
            # dog contests: with p gets underdog-sweep bonus path, with 1-p gifts fav-sweep bonus
            ev = p * v_und - (1 - p) * v_fav
            row["ev_pp_using_B_branches"] = ev
            row["branch_win_dpp"] = v_und
            row["branch_gift_dpp"] = v_fav
        if v_unique is not None:
            # alternate: unique value only (symmetric)
            row["ev_pp_using_unique_V"] = p * v_unique - (1 - p) * abs(v_unique)
            row["V_unique_dpp"] = v_unique
        row["verdict"] = None
        ev_use = row.get("ev_pp_using_B_branches")
        if ev_use is not None:
            row["verdict"] = "+EV contest" if ev_use > 0.5 else ("~0" if abs(ev_use) <= 0.5 else "−EV contest")
        rows.append(row)

    return {
        "fight_proxy_meta": fight_meta,
        "V_unique_controlled_dpp": v_unique,
        "V_unique_ci95": ci,
        "V_underdog_sweep_dpp": v_und,
        "V_favorite_sweep_dpp": v_fav,
        "table": rows,
        "one_liner": _ev_oneliner(rows, v_unique, v_und, v_fav),
    }


def _ev_oneliner(rows, v_u, v_und, v_fav) -> str:
    r30 = next((r for r in rows if r["p_fight_proxy"] == 0.30), None)
    if not r30:
        return ""
    ev = r30.get("ev_pp_using_B_branches")
    vu = f"{v_u:.2f}pp" if v_u is not None else "n/a"
    if ev is None:
        return f"Controlled unique grub value ≈ {vu}; fight proxy EV table incomplete."
    return (
        f"If you’re a 30% fight dog and underdog-sweep ΔWR≈{v_und:.2f}pp while "
        f"favorite-sweep ΔWR≈{v_fav:.2f}pp, contest EV≈{ev:+.2f}pp "
        f"(controlled unique V≈{vu})."
    )


# ---------------------------------------------------------------------------
# Robustness by year
# ---------------------------------------------------------------------------


def run_subset(df: pd.DataFrame, label: str) -> dict:
    df = df.reset_index(drop=True)
    print(f"[grubs3] subset={label} n={len(df)}")
    if len(df) < 300:
        return {"label": label, "n": int(len(df)), "skipped": True}
    raw = raw_ladder(df)
    fight_meta, p_fight = fit_fight_proxy(df)
    a = control_ladder(df)
    b = block_b_underdog(df, p_fight)
    c = contest_ev_table(a, b, fight_meta)
    return {
        "label": label,
        "n": int(len(df)),
        "date_min": str(df["date"].min().date()) if "date" in df else None,
        "date_max": str(df["date"].max().date()) if "date" in df else None,
        "raw": raw,
        "controlled": a,
        "underdog_favorite": b,
        "contest_ev": c,
        "fight_proxy": fight_meta,
    }


# ---------------------------------------------------------------------------
# Brief
# ---------------------------------------------------------------------------


def write_brief(report: dict, path: Path) -> None:
    main = report["subsets"]["all_grub_sum_3"]
    a = main["controlled"]
    b = main["underdog_favorite"]
    c = main["contest_ev"]
    raw = main["raw"]

    def fmt(x, nd=4):
        if x is None:
            return "—"
        if isinstance(x, float):
            return f"{x:.{nd}f}"
        return str(x)

    lines = []
    lines.append("# Void Grubs (3-camp era): isolation brief")
    lines.append("")
    lines.append(f"**Sample:** maps with `grub_sum == 3` only · n={main['n']} · {main['date_min']} → {main['date_max']}")
    lines.append("")
    lines.append("## Limits (read first)")
    lines.append("")
    lines.append(
        "Oracle’s Elixir has **no fight logs**, no contest flag, and no grub spawn timestamp. "
        "Closest pre-objective state is gold/kills/xp @10. "
        "**P̂_fight** is a logistic strength proxy from gold@10 + kills@10 "
        f"(calibrated to **{main['fight_proxy']['target']}**), not observed skirmish win%. "
        "“Underdog took all 3” is an *outcome* proxy for contesting while unfavored — not proof they queued the fight."
    )
    lines.append("")
    lines.append("## 1. Raw WR ladder (blue grub count)")
    lines.append("")
    lines.append("| Blue grubs | n | Blue WR | mean gold@10 |")
    lines.append("|------------|---|---------|--------------|")
    for r in raw["blue_grub_count_ladder"]:
        g10 = f"{r['mean_gold10']:.0f}" if r.get("mean_gold10") is not None else "—"
        lines.append(f"| {r['blue_grubs']} | {r['n']} | {r['wr_blue']*100:.2f}% | {g10} |")
    ba = raw["blue_all3"]
    sw = raw["sweeper_all3"]
    lines.append("")
    lines.append(
        f"**Blue 3–0:** WR {fmt(ba.get('wr'), 4)} (n={ba.get('n')}) · "
        f"raw Δ vs baseline {fmt(ba.get('raw_dpp_vs_baseline'), 2)}pp · "
        f"sweeper-side WR {fmt(sw.get('wr'), 4)} (n={sw.get('n')})."
    )
    lines.append("")
    lines.append("## 2. Controlled unique effect (headline)")
    lines.append("")
    lines.append("| Step | Controls | unique Δpp | partial r | LR p | n |")
    lines.append("|------|----------|------------|-----------|------|---|")
    for s in a["nested_steps"]:
        lines.append(
            f"| {s['step']} | {', '.join(s.get('controls') or ['(none)'])} | "
            f"{fmt(s.get('unique_dpp'), 2)} | {fmt(s.get('partial_r'), 4)} | "
            f"{fmt(s.get('lr_p'), 4)} | {s.get('n')} |"
        )
    ci = a.get("headline_ci95") or (None, None)
    lines.append("")
    lines.append(
        f"**Headline (gold@10/15 + FH + FT):** unique Δpp = **{fmt(a.get('headline_unique_dpp'), 2)}** "
        f"(95% CI {fmt(ci[0], 2)} … {fmt(ci[1], 2)}) · "
        f"partial r = {fmt(a.get('headline_partial_r'), 4)} · LR p = **{fmt(a.get('headline_lr_p'), 4)}**."
    )
    if a.get("matched_gold10", {}).get("n_pairs"):
        m = a["matched_gold10"]
        lines.append(
            f"Matched gold@10 ±{m.get('tol_gold')}g: n_pairs={m['n_pairs']} · "
            f"Δpp = {fmt(m.get('dpp'), 2)}."
        )
    lines.append("")
    lines.append("### Gold@10 strata (blue 3–0 vs not)")
    lines.append("")
    lines.append("| Bin | n | n_all3 | WR all3 | WR not | Δpp |")
    lines.append("|-----|---|--------|---------|--------|-----|")
    for s in a.get("gold10_strata") or []:
        lines.append(
            f"| {s['bin']} | {s['n']} | {s['n_all3']} | {s['wr_all3']*100:.1f}% | "
            f"{s['wr_not']*100:.1f}% | {s['dpp']:+.1f} |"
        )
    lines.append("")
    lines.append("## 3. Underdog @10 vs favorite @10")
    lines.append("")
    tr = b["take_rates"]
    lines.append(
        f"Blue underdog (P̂_fight<0.5) still sweeps 3–0 in **{fmt(tr.get('underdog_blue_all3_rate'), 4)}** "
        f"of underdog maps (n_und={tr.get('n_underdog')}, sweeps={tr.get('underdog_still_sweeps_n')}). "
        f"Blue favorite sweeps in **{fmt(tr.get('favorite_blue_all3_rate'), 4)}** "
        f"(n_fav={tr.get('n_favorite')})."
    )
    lines.append("")
    ue = b["underdog_team_sweep_effect"]
    fe = b["favorite_team_sweep_effect"]
    lines.append("| Who sweeps all 3 | n | WR for that side | vs no-sweep Δpp |")
    lines.append("|------------------|---|-------------------|-----------------|")
    lines.append(
        f"| Underdog team | {ue.get('n_underdog_sweeps')} | {fmt(ue.get('wr_if_underdog_sweeps'), 4)} | "
        f"{fmt(ue.get('dpp'), 2)} |"
    )
    lines.append(
        f"| Favorite team | {fe.get('n_favorite_sweeps')} | {fmt(fe.get('wr_if_favorite_sweeps'), 4)} | "
        f"{fmt(fe.get('dpp'), 2)} |"
    )
    lines.append("")
    sc = b.get("scaling_control_story") or {}
    if sc:
        lines.append("### Scaling control story")
        lines.append("")
        w = sc.get("when_fight_fav_got_grubs") or {}
        lines.append(
            f"When fight-favorite got the grubs: scaling-favorite still wins "
            f"**{fmt(w.get('scaling_favorite_still_wins_rate'), 4)}** of the time (n={w.get('n')})."
        )
        fb = sc.get("fight_fav_but_worse_scaling") or {}
        if fb.get("dpp_grubs") is not None:
            lines.append(
                f"Fight-fav but worse scaling: WR with grubs {fmt(fb.get('wr_with_grubs'), 4)} "
                f"vs without {fmt(fb.get('wr_without_grubs'), 4)} · Δ **{fmt(fb.get('dpp_grubs'), 2)}pp** "
                f"(n_with={fb.get('n_with_grubs')}, n_without={fb.get('n_without')})."
            )
        lines.append("")
    lines.append("## 4. Contest EV (proxy)")
    lines.append("")
    lines.append(
        f"Branches: underdog-sweep Δ = {fmt(c.get('V_underdog_sweep_dpp'), 2)}pp · "
        f"favorite-sweep Δ = {fmt(c.get('V_favorite_sweep_dpp'), 2)}pp · "
        f"controlled unique V = {fmt(c.get('V_unique_controlled_dpp'), 2)}pp."
    )
    lines.append("")
    lines.append("| P̂_fight (dog) | EV pp (B branches) | EV pp (unique V) | verdict |")
    lines.append("|---------------|--------------------|------------------|---------|")
    for r in c.get("table") or []:
        lines.append(
            f"| {r['p_fight_proxy']:.0%} | {fmt(r.get('ev_pp_using_B_branches'), 2)} | "
            f"{fmt(r.get('ev_pp_using_unique_V'), 2)} | {r.get('verdict') or '—'} |"
        )
    lines.append("")
    lines.append(f"**One-liner:** {c.get('one_liner')}")
    lines.append("")
    lines.append("## 5. Year robustness")
    lines.append("")
    lines.append("| Subset | n | unique Δpp | LR p | und sweep Δpp | fav sweep Δpp |")
    lines.append("|--------|---|------------|------|---------------|---------------|")
    for key, sub in report["subsets"].items():
        if sub.get("skipped"):
            continue
        ca = sub["controlled"]
        ub = sub["underdog_favorite"]
        lines.append(
            f"| {key} | {sub['n']} | {fmt(ca.get('headline_unique_dpp'), 2)} | "
            f"{fmt(ca.get('headline_lr_p'), 4)} | "
            f"{fmt((ub.get('underdog_team_sweep_effect') or {}).get('dpp'), 2)} | "
            f"{fmt((ub.get('favorite_team_sweep_effect') or {}).get('dpp'), 2)} |"
        )
    lines.append("")
    lines.append("## What we are not claiming")
    lines.append("")
    lines.append(
        "- That teams “chose to fight” (only that the underdog sometimes ends with all 3).\n"
        "- That P̂_fight equals true combat win%.\n"
        "- That end-count grubs are fully causal after controls (Herald path / plates still partially entangled)."
    )
    lines.append("")
    path.write_text("\n".join(lines))
    print(f"[grubs3] wrote brief {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[grubs3] loading OE…")
    raw = load_oe_team_maps_extended()
    print(f"[grubs3] maps={len(raw)}")
    df = engineer_study(raw)
    print("[grubs3] loading drafts…")
    drafts = load_drafts()
    df = attach_draft_features(df, drafts)
    df = merge_elo(df)

    era = filter_3grub(df)
    print(f"[grubs3] grub_sum==3 → n={len(era)}")

    subsets = {
        "all_grub_sum_3": run_subset(era, "all_grub_sum_3"),
        "year_2026_only": run_subset(era[era["oe_year"] == 2026].copy(), "year_2026"),
        "year_2025_grub3": run_subset(era[era["oe_year"] == 2025].copy(), "year_2025_grub3"),
    }

    report = {
        "version": 1,
        "title": "3-grub era void grubs isolation study",
        "data_limits": [
            "OE has no fight logs / contest flag / grub timestamp.",
            "P_fight is gold@10+kills@10 proxy (usually calibrated to first tower).",
            "Underdog sweep = end-count proxy, not observed decision to fight.",
            "Elo merge coverage depends on features.parquet join rate.",
        ],
        "elo_coverage": {
            "n_era": int(len(era)),
            "n_with_elo": int(era["elo_diff"].notna().sum()) if "elo_diff" in era.columns else 0,
            "n_with_draft": int(era["arch_scale_vs_snowball"].notna().sum())
            if "arch_scale_vs_snowball" in era.columns
            else 0,
        },
        "subsets": subsets,
    }

    # headline takeaways
    main_a = subsets["all_grub_sum_3"]["controlled"]
    main_c = subsets["all_grub_sum_3"]["contest_ev"]
    report["takeaways"] = [
        f"Controlled unique Δpp (gold10/15+FH+FT) = {main_a.get('headline_unique_dpp')}",
        f"LR p = {main_a.get('headline_lr_p')}",
        f"partial r = {main_a.get('headline_partial_r')}",
        main_c.get("one_liner"),
    ]

    out = MODELS_DIR / "grubs_isolation_study.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[grubs3] wrote {out}")

    brief = MODELS_DIR / "grubs_isolation_brief.md"
    write_brief(report, brief)

    print("\n=== TAKEAWAYS ===")
    for t in report["takeaways"]:
        print(f" • {t}")


if __name__ == "__main__":
    main()
