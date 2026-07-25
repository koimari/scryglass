#!/usr/bin/env python3
"""
Void grubs v2 — 3-camp era isolation + measurable contest EV (beatdown).

Fixes from adversarial board:
  - Era = patch ≥ 15.09 (not grub_sum alone)
  - Headline = @10-conditional association + mediation ladder (not residual≈0)
  - Like-with-like underdog / role contrasts
  - Contest EV from Flores beatdown roles + gold@10 state (measurable)
  - Optional Riot timeline HORDE contest flags when cache present

  python3 -m lol_kills.research.grubs_contest_study
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
from lol_kills.draft_phase_score import (
    BEATDOWN_WEIGHTS,
    INEVITABILITY_WEIGHTS,
    _axis,
    _load_coefs,
    assign_roles,
    sigmoid,
)
from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import MODELS_DIR, RAW_OE_DIR, WAREHOUSE_DIR
from lol_kills.etl.riot_timelines import cache_path, load_cached, summarize_map_grubs
from lol_kills.research.side_objective_edges import engineer

MIN_N = 80
EARLY_GAP_CLEAR = 0.35
GOLD_BINS = [
    ("behind2k", -1e9, -2000),
    ("behind", -2000, -500),
    ("even", -500, 500),
    ("ahead", 500, 2000),
    ("ahead2k", 2000, 1e9),
]


def _patch_key(p: Any) -> tuple[int, int]:
    try:
        parts = str(p).strip().split(".")
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        return (0, 0)


def is_3grub_era(patch: Any, oe_year: int) -> bool:
    """Three-grub format from OE's 15.09 label (Riot's public 25.09 notes).

    Oracle's Elixir labels the 2025 season as ``15.xx`` and the 2026 season as
    ``16.xx``.  Keeping the source field as a string is essential: numeric CSV
    inference converts ``15.10`` to ``15.1`` and would otherwise misclassify it.
    """
    if int(oe_year) >= 2026:
        return True
    if int(oe_year) < 2025:
        return False
    return _patch_key(patch) >= (15, 9)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_maps() -> pd.DataFrame:
    files = sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    want = {
        "gameid",
        "date",
        "league",
        "year",
        "patch",
        "side",
        "position",
        "teamname",
        "result",
        "gamelength",
        "kills",
        "firstblood",
        "firstdragon",
        "firstherald",
        "firsttower",
        "void_grubs",
        "golddiffat10",
        "golddiffat15",
        "xpdiffat10",
        "killsat10",
        "ckpm",
    }
    frames = []
    for fp in files:
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        usecols = [c for c in hdr if c in want]
        # Patch must stay text.  ``15.10`` is semantically distinct from
        # ``15.1``; parsing it as a float silently excludes valid 3-grub maps.
        dtypes = {"patch": "string"} if "patch" in usecols else None
        df = pd.read_csv(fp, usecols=usecols, dtype=dtypes, low_memory=False)
        df = df[df["position"].astype(str).str.lower() == "team"].copy()
        df["oe_year"] = int(fp.name[:4])
        frames.append(df)
        print(f"[grubs2] loaded {fp.name} team_rows={len(df)}")
    raw = pd.concat(frames, ignore_index=True)
    raw["side"] = raw["side"].astype(str).str.title()
    raw["gameid"] = raw["gameid"].astype(str)

    def pivot(side: str, prefix: str) -> pd.DataFrame:
        s = raw[raw["side"] == side].drop_duplicates("gameid", keep="first")
        return s.rename(columns={c: f"{prefix}{c}" for c in s.columns if c not in ("gameid", "side", "position")})

    m = pivot("Blue", "blue_").merge(pivot("Red", "red_"), on="gameid")
    m = m.rename(
        columns={
            "blue_date": "date",
            "blue_league": "league",
            "blue_year": "year",
            "blue_patch": "patch",
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
    files = sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    frames = []
    for fp in files:
        df = pd.read_csv(fp, usecols=["gameid", "side", "position", "champion"], low_memory=False)
        df = df[df["position"].astype(str).str.lower().isin(["top", "jng", "mid", "bot", "sup"])]
        df["gameid"] = df["gameid"].astype(str)
        df["side"] = df["side"].astype(str).str.title()
        frames.append(df)
    raw = pd.concat(frames, ignore_index=True)
    rows = []
    for gid, g in raw.groupby("gameid"):
        b = [normalize_champ(c) for c in g.loc[g.side == "Blue", "champion"].astype(str) if c and c != "nan"]
        r = [normalize_champ(c) for c in g.loc[g.side == "Red", "champion"].astype(str) if c and c != "nan"]
        if len(b) >= 5 and len(r) >= 5:
            rows.append({"gameid": gid, "blue_champs": b[:5], "red_champs": r[:5]})
    return pd.DataFrame(rows)


def engineer_v2(df: pd.DataFrame) -> pd.DataFrame:
    d = engineer(df)
    for c in ("blue_killsat10", "red_killsat10", "blue_xpdiffat10"):
        d[c] = pd.to_numeric(d.get(c), errors="coerce")
    d["gold10"] = pd.to_numeric(d["blue_golddiffat10"], errors="coerce")
    d["gold15"] = pd.to_numeric(d["blue_golddiffat15"], errors="coerce")
    d["xp10"] = pd.to_numeric(d.get("blue_xpdiffat10"), errors="coerce")
    d["kills10_diff"] = d["blue_killsat10"] - d["red_killsat10"]
    bg = d["blue_void_grubs"].fillna(0)
    rg = d["red_void_grubs"].fillna(0)
    d["grub_sum"] = bg + rg
    d["blue_all3"] = ((bg == 3) & (rg == 0)).astype(float)
    d["red_all3"] = ((rg == 3) & (bg == 0)).astype(float)
    d["neither_all3"] = ((d["blue_all3"] == 0) & (d["red_all3"] == 0)).astype(float)
    d["blue_firstherald"] = pd.to_numeric(d["blue_firstherald"], errors="coerce")
    d["blue_firsttower"] = pd.to_numeric(d["blue_firsttower"], errors="coerce")
    d["era_3grub"] = [
        is_3grub_era(p, y) for p, y in zip(d["patch"], d["oe_year"].astype(int))
    ]
    return d


def attach_beatdown(df: pd.DataFrame, drafts: pd.DataFrame) -> pd.DataFrame:
    """Fast beatdown axes + Flores roles + draft P(ahead@10)."""
    m = df.merge(drafts, on="gameid", how="left")
    coefs = _load_coefs()
    ahead = (coefs.get("buckets") or {}).get("10", {}).get("ahead") or {}
    feat_names = ahead.get("feature_names") or ["elo_z", "draft_win_logit", "beatdown_diff", "inev_diff"]
    acoef = dict(zip(feat_names, ahead.get("coef") or [0, 0, 0, 0]))
    aint = float(ahead.get("intercept") or 0.0)

    rows = {
        "beatdown_diff": [],
        "inev_diff": [],
        "blue_is_beatdown": [],
        "early_gap": [],
        "late_gap": [],
        "clear_roles": [],
        "p_ahead_draft_blue": [],
        "arch_scale_vs_snowball": [],
    }
    for _, r in m.iterrows():
        b, rd = r.get("blue_champs"), r.get("red_champs")
        if not isinstance(b, list) or not isinstance(rd, list):
            for k in rows:
                rows[k].append(np.nan)
            continue
        feats = draft_archetype_features(b, rd)
        powers = {
            "beatdown_blue": _axis(feats, BEATDOWN_WEIGHTS, "blue"),
            "beatdown_red": _axis(feats, BEATDOWN_WEIGHTS, "red"),
            "beatdown_diff": _axis(feats, BEATDOWN_WEIGHTS, "diff"),
            "inev_blue": _axis(feats, INEVITABILITY_WEIGHTS, "blue"),
            "inev_red": _axis(feats, INEVITABILITY_WEIGHTS, "red"),
            "inev_diff": _axis(feats, INEVITABILITY_WEIGHTS, "diff"),
        }
        roles = assign_roles(powers)
        # ahead logit without elo/draft_win (pure beatdown path) + small base
        logit = aint + float(acoef.get("beatdown_diff", 0.0)) * powers["beatdown_diff"]
        logit += float(acoef.get("inev_diff", 0.0)) * powers["inev_diff"]
        rows["beatdown_diff"].append(powers["beatdown_diff"])
        rows["inev_diff"].append(powers["inev_diff"])
        rows["blue_is_beatdown"].append(1.0 if roles["blue_is_beatdown"] else 0.0)
        rows["early_gap"].append(roles["early_gap"])
        rows["late_gap"].append(roles["late_gap"])
        rows["clear_roles"].append(1.0 if roles["early_gap"] >= EARLY_GAP_CLEAR else 0.0)
        rows["p_ahead_draft_blue"].append(float(sigmoid(logit)))
        rows["arch_scale_vs_snowball"].append(
            float(feats["arch_scaling_late_diff"] - feats["arch_early_snowball_diff"])
        )
    for k, v in rows.items():
        m[k] = v

    # Role-perspective features
    m["gold10_beatdown"] = np.where(m["blue_is_beatdown"] == 1, m["gold10"], -m["gold10"])
    m["beatdown_all3"] = np.where(m["blue_is_beatdown"] == 1, m["blue_all3"], m["red_all3"])
    m["control_all3"] = np.where(m["blue_is_beatdown"] == 1, m["red_all3"], m["blue_all3"])
    m["beatdown_won"] = np.where(m["blue_is_beatdown"] == 1, m["y_blue_win"], 1 - m["y_blue_win"])
    m["control_won"] = 1 - m["beatdown_won"]
    m["p_fight_beatdown"] = np.where(
        m["blue_is_beatdown"] == 1, m["p_ahead_draft_blue"], 1 - m["p_ahead_draft_blue"]
    )
    # Blend draft ahead prior with live gold@10 into a fight proxy for beatdown
    # logistic: stronger gold → higher P(beatdown wins river)
    return m


def blend_fight_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Fit P(beatdown gets FT) ~ gold10_beatdown + beatdown_diff on clear-role maps."""
    d = df.copy()
    d["ft_beatdown"] = np.where(
        d["blue_is_beatdown"] == 1, d["blue_firsttower"], 1 - d["blue_firsttower"].fillna(0)
    )
    sub = d[
        (d["clear_roles"] == 1)
        & d["gold10_beatdown"].notna()
        & d["ft_beatdown"].notna()
        & d["beatdown_diff"].notna()
    ].copy()
    if len(sub) < 500:
        d["p_fight_beatdown_live"] = d["p_fight_beatdown"]
        return d
    X = np.column_stack(
        [sub["gold10_beatdown"].values / 1000.0, sub["beatdown_diff"].fillna(0).values]
    )
    y = sub["ft_beatdown"].astype(float).values
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    Xa = np.column_stack(
        [d["gold10_beatdown"].fillna(0).values / 1000.0, d["beatdown_diff"].fillna(0).values]
    )
    p = np.full(len(d), np.nan)
    mask = d["gold10_beatdown"].notna() & d["beatdown_diff"].notna()
    p[mask.to_numpy()] = lr.predict_proba(Xa[mask.to_numpy()])[:, 1]
    d["p_fight_beatdown_live"] = p
    d.attrs["fight_proxy_fit"] = {
        "target": "first_tower_for_beatdown_side",
        "coef_gold_per_1k": float(lr.coef_[0][0]),
        "coef_beatdown_diff": float(lr.coef_[0][1]),
        "intercept": float(lr.intercept_[0]),
        "n": int(len(sub)),
        "note": "P̂_fight = P(beatdown side gets first tower | gold@10_bd, beatdown_diff). Measurable OE proxy.",
    }
    return d


def attach_timelines(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    contested = []
    n_horde = []
    cov = 0
    for gid in d["gameid"].astype(str):
        tl = load_cached(gid)
        if not tl:
            contested.append(np.nan)
            n_horde.append(np.nan)
            continue
        cov += 1
        s = summarize_map_grubs(tl)
        contested.append(1.0 if s["any_contested"] else 0.0)
        n_horde.append(float(s["n_horde_events"]))
    d["timeline_contested"] = contested
    d["timeline_n_horde"] = n_horde
    d.attrs["timeline_coverage"] = {"n_cached_in_sample": cov, "n_sample": int(len(d))}
    return d


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def unique_dpp(y: np.ndarray, x: np.ndarray, Z: np.ndarray) -> dict:
    mask = np.isfinite(Z).all(1) & np.isfinite(y) & np.isfinite(x)
    y, x, Z = y[mask], x[mask], Z[mask]
    if len(y) < MIN_N:
        return {"n": int(len(y))}
    lr1 = LogisticRegression(C=1e6, max_iter=2000).fit(Z, y)
    Zx = np.column_stack([Z, x])
    lr2 = LogisticRegression(C=1e6, max_iter=2000).fit(Zx, y)
    p1 = np.clip(lr1.predict_proba(Z)[:, 1], 1e-6, 1 - 1e-6)
    p2 = np.clip(lr2.predict_proba(Zx)[:, 1], 1e-6, 1 - 1e-6)
    ll1 = -log_loss(y, p1, normalize=False)
    ll2 = -log_loss(y, p2, normalize=False)
    lr_p = float(stats.chi2.sf(2 * (ll2 - ll1), 1))
    Zm = Z.mean(0, keepdims=True)
    a = float(lr2.predict_proba(np.column_stack([Zm, [[0.0]]]))[0, 1])
    b = float(lr2.predict_proba(np.column_stack([Zm, [[1.0]]]))[0, 1])
    rx = x - LinearRegression().fit(Z, x).predict(Z)
    ry = y - LinearRegression().fit(Z, y).predict(Z)
    pr = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 1e-12 and ry.std() > 1e-12 else None
    return {
        "n": int(len(y)),
        "unique_dpp": (b - a) * 100,
        "lr_p": lr_p,
        "partial_r": pr,
        "coef": float(lr2.coef_[0][-1]),
        "pred0": a,
        "pred1": b,
    }


def matched_dpp(df: pd.DataFrame, treat: str, y: str, gold: str, tol: float = 750.0) -> dict:
    t = df[df[treat] == 1].dropna(subset=[gold, y])
    c = df[df[treat] == 0].dropna(subset=[gold, y])
    if len(t) < MIN_N or len(c) < MIN_N:
        return {"n_pairs": 0}
    cg, cy = c[gold].values, c[y].values
    used = np.zeros(len(c), dtype=bool)
    pt, pc = [], []
    for _, r in t.iterrows():
        d = np.abs(cg - r[gold])
        d[used] = 1e18
        j = int(np.argmin(d))
        if d[j] > tol:
            continue
        used[j] = True
        pt.append(float(r[y]))
        pc.append(float(cy[j]))
    if len(pt) < MIN_N:
        return {"n_pairs": len(pt)}
    return {
        "n_pairs": len(pt),
        "wr_treat": float(np.mean(pt)),
        "wr_ctrl": float(np.mean(pc)),
        "dpp": (float(np.mean(pt)) - float(np.mean(pc))) * 100,
        "tol_gold": tol,
    }


# ---------------------------------------------------------------------------
# Block A — mediation ladder
# ---------------------------------------------------------------------------


def block_a(df: pd.DataFrame) -> dict:
    sub = df.dropna(
        subset=["y_blue_win", "blue_all3", "gold10", "kills10_diff", "xp10", "gold15", "blue_firstherald", "blue_firsttower"]
    ).copy()
    y = sub["y_blue_win"].astype(float).values
    x = sub["blue_all3"].astype(float).values
    steps = []

    raw = (float(sub.loc[sub.blue_all3 == 1, "y_blue_win"].mean()) - float(sub.loc[sub.blue_all3 == 0, "y_blue_win"].mean())) * 100
    steps.append({"step": "1_raw", "estimand": "raw association", "unique_dpp": raw, "n": len(sub)})

    Z2 = np.column_stack([sub.gold10 / 1000, sub.kills10_diff.fillna(0), sub.xp10.fillna(0) / 1000])
    s2 = unique_dpp(y, x, Z2)
    s2.update({"step": "2_pre_gold10", "estimand": "total assoc | pre/near-treatment @10", "headline": True})
    steps.append(s2)

    Z3 = np.column_stack(
        [sub.gold10 / 1000, sub.kills10_diff.fillna(0), sub.xp10.fillna(0) / 1000, sub.gold15 / 1000, sub.blue_firstherald.fillna(0), sub.blue_firsttower.fillna(0)]
    )
    s3 = unique_dpp(y, x, Z3)
    s3.update(
        {
            "step": "3_post_mediators",
            "estimand": "residual after gold@15+FH+FT (mediators — NOT unique causal value)",
            "headline": False,
        }
    )
    steps.append(s3)

    # mediator path evidence
    med = LinearRegression().fit(np.column_stack([sub.gold10 / 1000, x]), sub.gold15.values)
    ft = LogisticRegression(C=1e6, max_iter=1000).fit(
        np.column_stack([sub.gold10 / 1000, x]), sub.blue_firsttower.fillna(0).values
    )

    strata = []
    for name, lo, hi in GOLD_BINS:
        g = sub[(sub.gold10 >= lo) & (sub.gold10 < hi)]
        a = g[g.blue_all3 == 1]
        b = g[g.blue_all3 == 0]
        if len(a) < 25 or len(b) < 25:
            continue
        strata.append(
            {
                "bin": name,
                "n": len(g),
                "n_all3": len(a),
                "wr_all3": float(a.y_blue_win.mean()),
                "wr_not": float(b.y_blue_win.mean()),
                "dpp": (float(a.y_blue_win.mean()) - float(b.y_blue_win.mean())) * 100,
            }
        )

    match = matched_dpp(sub, "blue_all3", "y_blue_win", "gold10")
    headline = s2
    return {
        "n": len(sub),
        "nested_steps": steps,
        "headline_dpp": headline.get("unique_dpp"),
        "headline_lr_p": headline.get("lr_p"),
        "headline_partial_r": headline.get("partial_r"),
        "headline_estimand": headline.get("estimand"),
        "mediator_residual_dpp": s3.get("unique_dpp"),
        "mediator_residual_lr_p": s3.get("lr_p"),
        "mediation": {
            "all3_coef_on_gold15_given_gold10": float(med.coef_[-1]),
            "all3_OR_on_FT_given_gold10": float(math.exp(ft.coef_[0][-1])),
        },
        "gold10_strata": strata,
        "matched_gold10": match,
        "ladder_wr": [
            {
                "blue_grubs": int(k),
                "n": int(len(g)),
                "wr": float(g.y_blue_win.mean()),
                "mean_gold10": float(g.gold10.mean()) if g.gold10.notna().any() else None,
            }
            for k, g in sub.groupby(sub.blue_void_grubs.fillna(-1).astype(int))
            if k in (0, 1, 2, 3) and len(g) >= 40
        ],
    }


# ---------------------------------------------------------------------------
# Block B — beatdown contest EV (measurable)
# ---------------------------------------------------------------------------


def _bin_name(g: float) -> str:
    for name, lo, hi in GOLD_BINS:
        if lo <= g < hi:
            return name
    return "missing"


def block_b_beatdown_contest(df: pd.DataFrame) -> dict:
    """
    Measurable contest EV using Flores beatdown roles.

    Identification:
      - Beatdown = draft early-damage seat (must press grubs)
      - Control  = inevitability seat (should not force)
      - State S  = gold@10 from beatdown's POV
      - P̂_fight_beatdown = P(FT for beatdown | gold10_bd, beatdown_diff)
      - Outcomes: beatdown_all3 / control_all3 / neither (like-with-like)

    EV for CONTROL contesting vs gifting to beatdown:
      EV = p_control * (WR_c|control_all3,S − WR_c|beatdown_all3,S)
         + (1−p_control) * 0
      with optional lose-cost from timeline-contested losses when available.

    EV for BEATDOWN forcing when behind:
      EV = p_bd * (WR_bd|beatdown_all3,S − WR_bd|neither,S)
         − (1−p_bd) * (WR_bd|control_all3,S − WR_bd|neither,S)  [gift cost]
    """
    d = df[(df["clear_roles"] == 1) & df["gold10_beatdown"].notna() & df["p_fight_beatdown_live"].notna()].copy()
    d["gold_bin"] = d["gold10_beatdown"].map(_bin_name)

    take = {
        "n_clear_roles": int(len(d)),
        "beatdown_all3_rate": float(d["beatdown_all3"].mean()),
        "control_all3_rate": float(d["control_all3"].mean()),
        "neither_rate": float(d["neither_all3"].mean()),
        "control_steals_while_beatdown_ahead": None,
        "beatdown_takes_while_behind": None,
    }
    ahead = d[d["gold10_beatdown"] > 500]
    behind = d[d["gold10_beatdown"] < -500]
    if len(ahead) >= MIN_N:
        take["control_steals_while_beatdown_ahead"] = float(ahead["control_all3"].mean())
        take["n_beatdown_ahead"] = int(len(ahead))
    if len(behind) >= MIN_N:
        take["beatdown_takes_while_behind"] = float(behind["beatdown_all3"].mean())
        take["n_beatdown_behind"] = int(len(behind))

    # Per gold-bin WR for beatdown / control by who swept
    bins_out = []
    for name, lo, hi in GOLD_BINS:
        g = d[(d["gold10_beatdown"] >= lo) & (d["gold10_beatdown"] < hi)]
        if len(g) < 120:
            continue
        bd_sw = g[g["beatdown_all3"] == 1]
        ct_sw = g[g["control_all3"] == 1]
        nei = g[g["neither_all3"] == 1]
        row = {
            "bin": name,
            "n": int(len(g)),
            "mean_p_fight_beatdown": float(g["p_fight_beatdown_live"].mean()),
            "beatdown_sweep": {
                "n": int(len(bd_sw)),
                "wr_beatdown": float(bd_sw["beatdown_won"].mean()) if len(bd_sw) >= 25 else None,
                "wr_control": float(bd_sw["control_won"].mean()) if len(bd_sw) >= 25 else None,
            },
            "control_sweep": {
                "n": int(len(ct_sw)),
                "wr_beatdown": float(ct_sw["beatdown_won"].mean()) if len(ct_sw) >= 25 else None,
                "wr_control": float(ct_sw["control_won"].mean()) if len(ct_sw) >= 25 else None,
            },
            "neither": {
                "n": int(len(nei)),
                "wr_beatdown": float(nei["beatdown_won"].mean()) if len(nei) >= 25 else None,
                "wr_control": float(nei["control_won"].mean()) if len(nei) >= 25 else None,
            },
        }
        # Like-with-like Δ for control steal vs gift to beatdown
        if row["control_sweep"]["wr_control"] is not None and row["beatdown_sweep"]["wr_control"] is not None:
            row["delta_wr_control_steal_vs_gift_pp"] = (
                row["control_sweep"]["wr_control"] - row["beatdown_sweep"]["wr_control"]
            ) * 100
        if row["beatdown_sweep"]["wr_beatdown"] is not None and row["neither"]["wr_beatdown"] is not None:
            row["delta_wr_beatdown_take_vs_neither_pp"] = (
                row["beatdown_sweep"]["wr_beatdown"] - row["neither"]["wr_beatdown"]
            ) * 100
        if row["control_sweep"]["wr_beatdown"] is not None and row["neither"]["wr_beatdown"] is not None:
            row["delta_wr_beatdown_if_gifted_away_pp"] = (
                row["control_sweep"]["wr_beatdown"] - row["neither"]["wr_beatdown"]
            ) * 100
        bins_out.append(row)

    # Global like-with-like (pooled) for EV table
    bd_sw = d[d["beatdown_all3"] == 1]
    ct_sw = d[d["control_all3"] == 1]
    nei = d[d["neither_all3"] == 1]
    pooled = {
        "wr_control_if_control_sweeps": float(ct_sw["control_won"].mean()) if len(ct_sw) >= MIN_N else None,
        "wr_control_if_beatdown_sweeps": float(bd_sw["control_won"].mean()) if len(bd_sw) >= MIN_N else None,
        "wr_control_if_neither": float(nei["control_won"].mean()) if len(nei) >= MIN_N else None,
        "wr_beatdown_if_beatdown_sweeps": float(bd_sw["beatdown_won"].mean()) if len(bd_sw) >= MIN_N else None,
        "wr_beatdown_if_control_sweeps": float(ct_sw["beatdown_won"].mean()) if len(ct_sw) >= MIN_N else None,
        "wr_beatdown_if_neither": float(nei["beatdown_won"].mean()) if len(nei) >= MIN_N else None,
        "n_beatdown_sweep": int(len(bd_sw)),
        "n_control_sweep": int(len(ct_sw)),
        "n_neither": int(len(nei)),
    }
    if pooled["wr_control_if_control_sweeps"] is not None and pooled["wr_control_if_beatdown_sweeps"] is not None:
        pooled["V_control_steal_vs_gift_pp"] = (
            pooled["wr_control_if_control_sweeps"] - pooled["wr_control_if_beatdown_sweeps"]
        ) * 100
    if pooled["wr_beatdown_if_beatdown_sweeps"] is not None and pooled["wr_beatdown_if_neither"] is not None:
        pooled["V_beatdown_take_vs_neither_pp"] = (
            pooled["wr_beatdown_if_beatdown_sweeps"] - pooled["wr_beatdown_if_neither"]
        ) * 100
    if pooled["wr_beatdown_if_control_sweeps"] is not None and pooled["wr_beatdown_if_neither"] is not None:
        pooled["V_beatdown_cost_of_giving_pp"] = (
            pooled["wr_beatdown_if_neither"] - pooled["wr_beatdown_if_control_sweeps"]
        ) * 100

    # EV tables
    V_steal = pooled.get("V_control_steal_vs_gift_pp")
    V_bd_get = pooled.get("V_beatdown_take_vs_neither_pp")
    V_bd_gift_cost = pooled.get("V_beatdown_cost_of_giving_pp")

    control_ev = []
    for p_bd in (0.30, 0.40, 0.50, 0.60, 0.70):
        p_c = 1.0 - p_bd  # control's fight win proxy
        row: dict[str, Any] = {"p_fight_beatdown": p_bd, "p_fight_control": p_c}
        if V_steal is not None:
            # vs gift baseline: win → gain V_steal, lose → 0 (beatdown still has it)
            ev = p_c * V_steal
            row["ev_control_contest_vs_gift_pp"] = ev
            row["verdict"] = "+EV" if ev > 1.0 else ("~0" if abs(ev) <= 1.0 else "−EV")
        control_ev.append(row)

    beatdown_ev = []
    for p_bd in (0.30, 0.40, 0.50, 0.60, 0.70):
        row = {"p_fight_beatdown": p_bd}
        if V_bd_get is not None and V_bd_gift_cost is not None:
            # force: win → V_get, lose → pay gift cost
            ev = p_bd * V_bd_get - (1 - p_bd) * max(V_bd_gift_cost, 0.0)
            row["ev_beatdown_force_pp"] = ev
            row["V_get"] = V_bd_get
            row["V_gift_cost"] = V_bd_gift_cost
            row["verdict"] = "+EV" if ev > 1.0 else ("~0" if abs(ev) <= 1.0 else "−EV")
        beatdown_ev.append(row)

    # Timeline refinement
    tl = {}
    if "timeline_contested" in d.columns and d["timeline_contested"].notna().sum() >= 50:
        tc = d[d["timeline_contested"] == 1]
        tu = d[d["timeline_contested"] == 0]
        tl = {
            "n_contested_maps": int(len(tc)),
            "n_uncontested_maps": int(len(tu)),
            "beatdown_wr_if_contested_and_beatdown_swept": float(
                tc.loc[tc.beatdown_all3 == 1, "beatdown_won"].mean()
            )
            if (tc.beatdown_all3 == 1).sum() >= 20
            else None,
            "control_wr_if_contested_and_control_swept": float(
                tc.loc[tc.control_all3 == 1, "control_won"].mean()
            )
            if (tc.control_all3 == 1).sum() >= 20
            else None,
            "note": "Timeline HORDE contest flags merged from local cache when present.",
        }

    # Empirical p_fight by gold bin (calibration check)
    calib = []
    for name, lo, hi in GOLD_BINS:
        g = d[(d["gold10_beatdown"] >= lo) & (d["gold10_beatdown"] < hi)]
        if len(g) < 80:
            continue
        calib.append(
            {
                "bin": name,
                "mean_p_hat": float(g["p_fight_beatdown_live"].mean()),
                "empirical_ft_beatdown": float(g["ft_beatdown"].mean()) if "ft_beatdown" in g else None,
                "n": int(len(g)),
            }
        )

    one = ""
    if V_steal is not None:
        one = (
            f"Clear Flores roles: control steal-vs-gift ΔWR={V_steal:.2f}pp "
            f"(camp value only — death cost of losing a contest needs timelines). "
            f"Control contest upper-bound EV = p_control×{V_steal:.2f}: "
            f"at 30% fight≈{0.3*V_steal:.2f}pp, at 50%≈{0.5*V_steal:.2f}pp. "
        )
    if V_bd_get is not None and V_bd_gift_cost is not None:
        one += (
            f"Beatdown force: get-vs-neither={V_bd_get:.2f}pp, gift-cost={V_bd_gift_cost:.2f}pp; "
            f"at 30% fight EV≈{0.3*V_bd_get - 0.7*max(V_bd_gift_cost,0):.2f}pp "
            f"(−EV to force while dog)."
        )

    return {
        "definition": {
            "beatdown": "Higher early-damage archetype axis (Flores)",
            "control": "Opponent / inevitability seat",
            "clear_roles": f"early_gap >= {EARLY_GAP_CLEAR}",
            "p_fight": "P(first tower for beatdown | gold@10_bd, beatdown_diff)",
            "contest_ev_control": (
                "UPPER BOUND: p_control * (WR_c|control_all3 − WR_c|beatdown_all3). "
                "Omits death/tempo cost of losing a contest (needs Riot HORDE timelines)."
            ),
            "like_with_like": "neither_all3 and role-conditional sweeps — not mixed favorite-sweep controls",
        },
        "take_rates": take,
        "pooled_values": pooled,
        "by_gold10_beatdown_bin": bins_out,
        "control_contest_ev_table": control_ev,
        "beatdown_force_ev_table": beatdown_ev,
        "p_fight_calibration": calib,
        "timeline": tl,
        "one_liner": one,
        "fight_proxy_fit": df.attrs.get("fight_proxy_fit"),
    }


# ---------------------------------------------------------------------------
# Brief
# ---------------------------------------------------------------------------


def write_brief(report: dict, path: Path) -> None:
    a = report["controlled"]
    b = report["contest_ev_beatdown"]

    def _pct(x):
        return f"{100*x:.1f}%" if isinstance(x, (int, float)) else "—"

    def _pp(x):
        return f"{x:.2f}pp" if isinstance(x, (int, float)) else "—"

    lines = []
    lines.append("# Void Grubs v2 — isolation + measurable contest EV (beatdown)")
    lines.append("")
    lines.append(
        f"**Sample:** 3-camp era (`patch ≥ 15.09` or 2026) AND `grub_sum==3` · "
        f"n={report['n']} · {report['date_min']} → {report['date_max']}"
    )
    lines.append("")
    lines.append("## Limits")
    lines.append("")
    lines.append(
        "OE has no fight logs. **Contest EV here is measurable under Flores roles + gold@10 + who ended with 3**, "
        "with P̂_fight = P(beatdown gets first tower | state). "
        "Riot Match-V5 HORDE timelines refine contested vs free takes when `RIOT_API_KEY` fills "
        f"`data/lol/warehouse/timelines/` (coverage now: {report.get('timeline_coverage')})."
    )
    lines.append("")
    lines.append("## 1. Raw ladder (blue grub count)")
    lines.append("")
    lines.append("| Blue grubs | n | Blue WR | mean gold@10 |")
    lines.append("|------------|---|---------|--------------|")
    for r in a["ladder_wr"]:
        g = f"{r['mean_gold10']:.0f}" if r.get("mean_gold10") is not None else "—"
        lines.append(f"| {r['blue_grubs']} | {r['n']} | {100*r['wr']:.2f}% | {g} |")
    lines.append("")
    lines.append("## 2. Controlled association (correct estimands)")
    lines.append("")
    lines.append("| Step | Estimand | Δpp | partial r | LR p | n |")
    lines.append("|------|----------|-----|-----------|------|---|")
    for s in a["nested_steps"]:
        dpp = s.get("unique_dpp")
        pr = s.get("partial_r")
        lp = s.get("lr_p")
        dpp_s = f"{dpp:.2f}" if isinstance(dpp, (int, float)) else "—"
        pr_s = f"{pr:.4f}" if isinstance(pr, (int, float)) else "—"
        lp_s = f"{lp:.4f}" if isinstance(lp, (int, float)) else "—"
        lines.append(
            f"| {s['step']} | {s.get('estimand','')} | {dpp_s} | {pr_s} | {lp_s} | {s.get('n')} |"
        )
    lines.append("")
    lines.append(
        f"**Headline (ship this):** {a['headline_estimand']} → "
        f"**{a['headline_dpp']:.2f}pp** (partial r={a.get('headline_partial_r')}, LR p={a.get('headline_lr_p')})."
    )
    lines.append(
        f"**Mediator residual (do not call 'unique value'):** {a['mediator_residual_dpp']:.2f}pp "
        f"(p={a.get('mediator_residual_lr_p')}). "
        f"Path check: all3 → +{a['mediation']['all3_coef_on_gold15_given_gold10']:.0f}g @15 | gold@10; "
        f"FT OR={a['mediation']['all3_OR_on_FT_given_gold10']:.2f}."
    )
    if a.get("matched_gold10", {}).get("n_pairs"):
        m = a["matched_gold10"]
        lines.append(f"Matched gold@10 ±{m['tol_gold']}g: **{m['dpp']:.2f}pp** (n_pairs={m['n_pairs']}).")
    lines.append("")
    lines.append("### Gold@10 strata")
    lines.append("")
    lines.append("| Bin | n | WR all3 | WR not | Δpp |")
    lines.append("|-----|---|---------|--------|-----|")
    for s in a["gold10_strata"]:
        lines.append(
            f"| {s['bin']} | {s['n']} | {100*s['wr_all3']:.1f}% | {100*s['wr_not']:.1f}% | {s['dpp']:+.1f} |"
        )
    lines.append("")
    lines.append("## 3. Measurable contest EV (Flores beatdown)")
    lines.append("")
    lines.append(
        f"Clear roles (early_gap≥{EARLY_GAP_CLEAR}): n={b['take_rates']['n_clear_roles']}. "
        f"Beatdown sweeps {b['take_rates']['beatdown_all3_rate']:.1%}; "
        f"control sweeps {b['take_rates']['control_all3_rate']:.1%}."
    )
    if b["take_rates"].get("control_steals_while_beatdown_ahead") is not None:
        lines.append(
            f"Control still steals all3 while beatdown ahead@10: "
            f"**{b['take_rates']['control_steals_while_beatdown_ahead']:.1%}** "
            f"(n={b['take_rates'].get('n_beatdown_ahead')})."
        )
    if b["take_rates"].get("beatdown_takes_while_behind") is not None:
        lines.append(
            f"Beatdown still takes all3 while behind@10: "
            f"**{b['take_rates']['beatdown_takes_while_behind']:.1%}** "
            f"(n={b['take_rates'].get('n_beatdown_behind')})."
        )
    lines.append("")
    pv = b["pooled_values"]
    lines.append("| Outcome | n | WR beatdown | WR control |")
    lines.append("|---------|---|-------------|------------|")
    lines.append(
        f"| Beatdown all3 | {pv.get('n_beatdown_sweep')} | "
        f"{_pct(pv.get('wr_beatdown_if_beatdown_sweeps'))} | {_pct(pv.get('wr_control_if_beatdown_sweeps'))} |"
    )
    lines.append(
        f"| Control all3 | {pv.get('n_control_sweep')} | "
        f"{_pct(pv.get('wr_beatdown_if_control_sweeps'))} | {_pct(pv.get('wr_control_if_control_sweeps'))} |"
    )
    lines.append(
        f"| Neither all3 | {pv.get('n_neither')} | "
        f"{_pct(pv.get('wr_beatdown_if_neither'))} | {_pct(pv.get('wr_control_if_neither'))} |"
    )
    lines.append("")
    lines.append(
        f"**V_control (steal vs gift)** = {_pp(pv.get('V_control_steal_vs_gift_pp'))} · "
        f"**V_beatdown (take vs neither)** = {_pp(pv.get('V_beatdown_take_vs_neither_pp'))} · "
        f"**V_gift_cost** = {_pp(pv.get('V_beatdown_cost_of_giving_pp'))}."
    )
    lines.append("")
    lines.append("### Control contest EV vs gifting to beatdown")
    lines.append("")
    lines.append("| P̂_fight beatdown | P̂_fight control | EV pp | verdict |")
    lines.append("|------------------|-----------------|-------|---------|")
    for r in b["control_contest_ev_table"]:
        ev = r.get("ev_control_contest_vs_gift_pp")
        ev_s = f"{ev:.2f}" if isinstance(ev, (int, float)) else "—"
        lines.append(
            f"| {r['p_fight_beatdown']:.0%} | {r['p_fight_control']:.0%} | "
            f"{ev_s} | {r.get('verdict')} |"
        )
    lines.append("")
    lines.append(
        "*Control EV is an **upper bound** on camp value (p×ΔWR steal-vs-gift). "
        "Losing a contest may cost extra deaths/tempo — fill Riot timelines to subtract that.*"
    )
    lines.append("")
    lines.append("### Beatdown force EV (behind / contested)")
    lines.append("")
    lines.append("| P̂_fight beatdown | EV pp | verdict |")
    lines.append("|------------------|-------|---------|")
    for r in b["beatdown_force_ev_table"]:
        lines.append(
            f"| {r['p_fight_beatdown']:.0%} | "
            f"{r.get('ev_beatdown_force_pp', float('nan')):.2f} | {r.get('verdict')} |"
        )
    lines.append("")
    lines.append(f"**One-liner:** {b.get('one_liner')}")
    lines.append("")
    if b.get("timeline"):
        lines.append("### Timeline layer")
        lines.append("")
        lines.append(json.dumps(b["timeline"], indent=2))
        lines.append("")
    lines.append("## What we claim / don't")
    lines.append("")
    lines.append(
        "- Claim: @10-conditional association ~ multi-pp; residual after gold@15+FT+Herald ≈ 0 = **mediation**, not worthlessness.\n"
        "- Claim: contest EV is defined for Flores roles with measurable WR branches + FT-calibrated P̂_fight.\n"
        "- Don't: call residual-after-mediators the 'unique grub value'.\n"
        "- Don't: claim P̂_fight is true combat win% without timelines.\n"
        "- Don't: use mixed favorite-sweep controls for underdog Δ."
    )
    path.write_text("\n".join(lines))
    print(f"[grubs2] wrote {path}")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[grubs2] loading…")
    raw = load_maps()
    df = engineer_v2(raw)
    drafts = load_drafts()
    print("[grubs2] beatdown…")
    df = attach_beatdown(df, drafts)
    df = blend_fight_proxy(df)
    df = attach_timelines(df)

    era = df[df["era_3grub"] & (df["grub_sum"] == 3)].reset_index(drop=True)
    print(f"[grubs2] era+sum3 n={len(era)} (dropped non-era / non-sum3)")
    tl_cov = {
        "n_cached_in_sample": int(era["timeline_contested"].notna().sum()),
        "n_sample": int(len(era)),
        "frac": float(era["timeline_contested"].notna().mean()),
        "hint": "export RIOT_API_KEY && python3 -m lol_kills.etl.riot_timelines --from-oe --year-min 2025 --fetch --limit 500",
    }

    a = block_a(era)
    b = block_b_beatdown_contest(era)

    report = {
        "version": 2,
        "title": "3-camp void grubs — mediation + beatdown contest EV",
        "n": int(len(era)),
        "date_min": str(era["date"].min().date()),
        "date_max": str(era["date"].max().date()),
        "era_filter": "patch>=15.09 or oe_year>=2026; grub_sum==3",
        "timeline_coverage": tl_cov,
        "board_fixes": [
            "Headline is @10-conditional association, not mediator residual",
            "Era uses patch>=15.09",
            "Contest EV uses Flores beatdown + neither_all3 like-with-like",
            "Optional Riot HORDE contest merge",
        ],
        "controlled": a,
        "contest_ev_beatdown": b,
        "takeaways": [
            f"Headline @10-conditional Δpp={a.get('headline_dpp')}",
            f"Mediator residual Δpp={a.get('mediator_residual_dpp')}",
            b.get("one_liner"),
        ],
    }

    out = MODELS_DIR / "grubs_isolation_study.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"[grubs2] wrote {out}")
    brief = MODELS_DIR / "grubs_isolation_brief.md"
    write_brief(report, brief)

    # also keep a v2-named copy
    (MODELS_DIR / "grubs_contest_study.json").write_text(json.dumps(report, indent=2, default=str))
    (MODELS_DIR / "grubs_contest_brief.md").write_text(brief.read_text())

    print("\n=== TAKEAWAYS ===")
    for t in report["takeaways"]:
        print(f" • {t}")


if __name__ == "__main__":
    main()
