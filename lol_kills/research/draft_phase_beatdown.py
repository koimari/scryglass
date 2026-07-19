#!/usr/bin/env python3
"""
Fit phase-bucket draft score + beatdown roles from OE (@10/@15/@20/@25).

Mike Flores ("Who's the Beatdown?"): misassignment of role = game loss.
  - Beatdown = more early damage → must convert before inevitability.
  - Control  = more late inevitability → must weather early and win late.

We fit Elo-controlled logits for:
  P(ahead @ t), P(win), and P(win | gold@t)
using beatdown / inevitability archetype axes + classic draft_win_logit.

  python3 -m lol_kills.research.draft_phase_beatdown
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from lol_kills.draft_archetypes import draft_archetype_features
from lol_kills.draft_phase_score import (
    BEATDOWN_WEIGHTS,
    BUCKETS,
    INEVITABILITY_WEIGHTS,
    _axis,
)
from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR, RAW_OE_DIR

OUT = MODELS_DIR / "draft_phase_beatdown.json"

def load_oe_draft_maps() -> pd.DataFrame:
    """One row per map: drafts, gold@t, win, optional Elo."""
    files = sorted(RAW_OE_DIR.glob("*_LoL_esports_match_data_from_OraclesElixir.csv"))
    want = {
        "gameid",
        "date",
        "league",
        "year",
        "side",
        "position",
        "champion",
        "result",
        "golddiffat10",
        "golddiffat15",
        "golddiffat20",
        "golddiffat25",
        "gamelength",
        "teamname",
    }
    frames = []
    for fp in files:
        hdr = pd.read_csv(fp, nrows=0).columns.tolist()
        usecols = [c for c in hdr if c in want]
        df = pd.read_csv(fp, usecols=usecols, low_memory=False)
        df["oe_year"] = int(fp.name[:4])
        frames.append(df)
        print(f"[phase] loaded {fp.name} rows={len(df)}")
    raw = pd.concat(frames, ignore_index=True)
    raw["side"] = raw["side"].astype(str).str.title()
    raw["position"] = raw["position"].astype(str).str.lower()
    raw["gameid"] = raw["gameid"].astype(str)

    # Team gold / result
    team = raw[raw["position"] == "team"].copy()
    blue_t = team[team["side"] == "Blue"].drop_duplicates("gameid")
    red_t = team[team["side"] == "Red"].drop_duplicates("gameid")
    bt = blue_t.rename(
        columns={
            "result": "y_blue_win",
            "golddiffat10": "gold10",
            "golddiffat15": "gold15",
            "golddiffat20": "gold20",
            "golddiffat25": "gold25",
            "gamelength": "gamelength",
            "league": "league",
            "date": "date",
            "teamname": "blue_team",
            "year": "year",
        }
    )[
        [
            "gameid",
            "y_blue_win",
            "gold10",
            "gold15",
            "gold20",
            "gold25",
            "gamelength",
            "league",
            "date",
            "blue_team",
            "year",
            "oe_year",
        ]
    ]
    rt = red_t.rename(columns={"teamname": "red_team"})[["gameid", "red_team"]]
    meta = bt.merge(rt, on="gameid", how="inner")

    # Player drafts (role order top/jng/mid/bot/sup)
    role_order = ["top", "jng", "mid", "bot", "sup"]
    pl = raw[raw["position"].isin(role_order)].copy()
    pl["champion"] = pl["champion"].map(lambda c: normalize_champ(str(c)) if pd.notna(c) else "")
    drafts = []
    for gid, g in pl.groupby("gameid"):
        blue, red = [], []
        for role in role_order:
            b = g[(g["side"] == "Blue") & (g["position"] == role)]["champion"]
            r = g[(g["side"] == "Red") & (g["position"] == role)]["champion"]
            if b.empty or r.empty or not b.iloc[0] or not r.iloc[0]:
                blue, red = [], []
                break
            blue.append(b.iloc[0])
            red.append(r.iloc[0])
        if len(blue) != 5:
            continue
        feat = draft_archetype_features(blue, red)
        feat["gameid"] = gid
        feat["blue_champs"] = "|".join(blue)
        feat["red_champs"] = "|".join(red)
        drafts.append(feat)
    adf = pd.DataFrame(drafts)
    df = meta.merge(adf, on="gameid", how="inner")

    for c in ("y_blue_win", "gold10", "gold15", "gold20", "gold25", "gamelength"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["y_blue_win", "date"]).sort_values("date")

    # Elo from warehouse features when available
    df["mu_diff"] = 0.0
    try:
        maps = pd.read_parquet(PARQUET_DIR / "maps.parquet", columns=["oe_gameid", "game_uid"])
        feat = pd.read_parquet(FEATURES_DIR / "maps.parquet", columns=["game_uid", "mu_diff", "draft_win_logit_blue"])
        maps["oe_gameid"] = maps["oe_gameid"].astype(str)
        feat["game_uid"] = feat["game_uid"].astype(str)
        maps["game_uid"] = maps["game_uid"].astype(str)
        m = maps.merge(feat, on="game_uid", how="left")
        elo = m.drop_duplicates("oe_gameid").set_index("oe_gameid")
        df["mu_diff"] = df["gameid"].map(elo["mu_diff"]).astype(float).fillna(0.0)
        df["draft_win_logit_blue"] = (
            df["gameid"].map(elo["draft_win_logit_blue"]).astype(float).fillna(0.0)
        )
    except Exception as e:
        print(f"[phase] elo merge skipped: {e}")
        df["draft_win_logit_blue"] = 0.0

    # Beatdown / inevitability axes
    rows_bd, rows_inev, rows_bdb, rows_bdr, rows_inb, rows_inr = [], [], [], [], [], []
    for _, row in df.iterrows():
        feats = {k: float(row[k]) for k in row.index if str(k).startswith("arch_")}
        bd = _axis(feats, BEATDOWN_WEIGHTS, "diff")
        inev = _axis(feats, INEVITABILITY_WEIGHTS, "diff")
        rows_bd.append(bd)
        rows_inev.append(inev)
        rows_bdb.append(_axis(feats, BEATDOWN_WEIGHTS, "blue"))
        rows_bdr.append(_axis(feats, BEATDOWN_WEIGHTS, "red"))
        rows_inb.append(_axis(feats, INEVITABILITY_WEIGHTS, "blue"))
        rows_inr.append(_axis(feats, INEVITABILITY_WEIGHTS, "red"))
    df["beatdown_diff"] = rows_bd
    df["inev_diff"] = rows_inev
    df["beatdown_blue"] = rows_bdb
    df["beatdown_red"] = rows_bdr
    df["inev_blue"] = rows_inb
    df["inev_red"] = rows_inr
    # Who should be beatdown: higher early power
    df["blue_is_beatdown"] = (df["beatdown_blue"] >= df["beatdown_red"]).astype(float)
    return df


def _fit_logit(X: np.ndarray, y: np.ndarray) -> dict | None:
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 400:
        return None
    lr = LogisticRegression(C=1e6, max_iter=2000)
    lr.fit(X[mask], y[mask])
    return {
        "coef": [float(c) for c in lr.coef_[0]],
        "intercept": float(lr.intercept_[0]),
        "n": int(mask.sum()),
    }


def _fit_ols(X: np.ndarray, y: np.ndarray) -> dict | None:
    mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    if mask.sum() < 400:
        return None
    lr = LinearRegression().fit(X[mask], y[mask])
    return {
        "coef": [float(c) for c in lr.coef_],
        "intercept": float(lr.intercept_),
        "n": int(mask.sum()),
        "r2": float(lr.score(X[mask], y[mask])),
    }


FEATURE_NAMES = ["elo_z", "draft_win_logit", "beatdown_diff", "inev_diff"]


def fit_buckets(df: pd.DataFrame) -> dict:
    elo_z = (df["mu_diff"].astype(float) / 400.0).values
    draft = df["draft_win_logit_blue"].astype(float).values
    bd = df["beatdown_diff"].astype(float).values
    inev = df["inev_diff"].astype(float).values
    y = df["y_blue_win"].astype(float).values
    X_base = np.column_stack([elo_z, draft, bd, inev])

    buckets = {}
    for t in BUCKETS:
        gcol = f"gold{t}"
        gold = df[gcol].astype(float).values
        ahead = (gold > 500).astype(float)
        # Pregame-style: draft → win (includes gold path)
        win_fit = _fit_logit(X_base, y)
        # Draft → ahead @ t
        ahead_fit = _fit_logit(X_base, ahead)
        # Gold magnitude
        gold_fit = _fit_ols(X_base, gold / 1000.0)
        # Unique draft after gold@t
        X_g = np.column_stack([elo_z, gold / 1000.0, draft, bd, inev])
        win_given_gold = _fit_logit(X_g, y)
        buckets[str(t)] = {
            "minute": t,
            "win": win_fit,
            "ahead": ahead_fit,
            "gold_k": gold_fit,
            "win_given_gold": {
                **(win_given_gold or {}),
                "feature_names": ["elo_z", "gold_k", "draft_win_logit", "beatdown_diff", "inev_diff"],
            }
            if win_given_gold
            else None,
            "feature_names": FEATURE_NAMES,
            "mean_gold": round(float(np.nanmean(gold)), 1),
            "p_ahead_base": round(float(np.nanmean(ahead)), 4),
        }
        print(f"[phase] @{t} win_n={win_fit and win_fit['n']} ahead_n={ahead_fit and ahead_fit['n']}")

    # Role-correct conversion: beatdown ahead@15 → win vs control ahead@15 → win
    sub = df.dropna(subset=["gold15", "y_blue_win"]).copy()
    sub["ahead15"] = sub["gold15"] > 500
    role_diag = {}
    for label, mask in (
        ("blue_beatdown_ahead", (sub["blue_is_beatdown"] == 1) & sub["ahead15"]),
        ("blue_beatdown_behind", (sub["blue_is_beatdown"] == 1) & (sub["gold15"] < -500)),
        ("blue_control_ahead", (sub["blue_is_beatdown"] == 0) & sub["ahead15"]),
        ("blue_control_behind", (sub["blue_is_beatdown"] == 0) & (sub["gold15"] < -500)),
    ):
        g = sub[mask]
        if len(g) < 80:
            role_diag[label] = {"n": int(len(g)), "wr": None}
        else:
            role_diag[label] = {"n": int(len(g)), "wr": round(float(g["y_blue_win"].mean()), 4)}

    return {
        "version": 1,
        "buckets": buckets,
        "beatdown_weights": BEATDOWN_WEIGHTS,
        "inevitability_weights": INEVITABILITY_WEIGHTS,
        "role_conversion": role_diag,
        "n_maps": int(len(df)),
        "note": (
            "Flores beatdown: early-damage side must convert; inevitability side must reach late. "
            "Per-bucket `win` coefs are identical (final WR target) — runtime curve uses "
            "gold_k path × win_given_gold leftovers, which do vary by @10/@15/@20/@25. "
            "win_given_gold: beatdown coef is negative (must have converted already); "
            "inev coef is positive (survives after gold is held)."
        ),
        "citation": "Mike Flores — Who's the Beatdown? (The Dojo / StarCityGames, 1999)",
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_oe_draft_maps()
    print(f"[phase] maps with drafts={len(df)}")
    art = fit_buckets(df)
    OUT.write_text(json.dumps(art, indent=2))
    print(f"[phase] wrote {OUT}")
    # Quick sanity print
    for t, b in art["buckets"].items():
        w = b.get("win") or {}
        coefs = dict(zip(FEATURE_NAMES, w.get("coef") or []))
        print(f"  @{t} beatdown_coef={coefs.get('beatdown_diff')} inev_coef={coefs.get('inev_diff')}")


if __name__ == "__main__":
    main()
