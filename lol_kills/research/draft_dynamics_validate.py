#!/usr/bin/env python3
"""
Validate draft-dynamics lenses on OE / warehouse maps.

Checks that analogy axes correlate with the outcomes we claim for betting:
  - tempo/inev → length & under kills
  - chaos → total kills
  - initiative → first blood / first tower
  - trichotomy (teamfight sum) → long-game WR proxy
  - pilot (hypercarry) → WR when behind @15
  - combo → shorter games when that side is ahead

  python3 -m lol_kills.research.draft_dynamics_validate
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from lol_kills.draft_archetypes import draft_archetype_features
from lol_kills.draft_dynamics import (
    COMBO_W,
    FAIR_W,
    INIT_W,
    REACT_W,
    SIEGE_W,
    SPLIT_W,
    TEAMFIGHT_W,
    _side_w,
)
from lol_kills.draft_phase_score import BEATDOWN_WEIGHTS, INEVITABILITY_WEIGHTS, _axis
from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR
from lol_kills.live_win import kill_conc_from_draft

OUT = MODELS_DIR / "draft_dynamics_validation.json"
MIN_N = 400


def _r(a, b, min_n=MIN_N):
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < min_n:
        return None, n
    aa, bb = a[mask], b[mask]
    if float(aa.std()) < 1e-12 or float(bb.std()) < 1e-12:
        return None, n
    return float(np.corrcoef(aa, bb)[0, 1]), n


def _resid_corr(y, x, elo, min_n=MIN_N):
    """Partial corr of x vs y after removing elo."""
    mask = np.isfinite(y) & np.isfinite(x) & np.isfinite(elo)
    if mask.sum() < min_n:
        return None, int(mask.sum())
    X = elo[mask].reshape(-1, 1)
    ry = y[mask] - LinearRegression().fit(X, y[mask]).predict(X)
    rx = x[mask] - LinearRegression().fit(X, x[mask]).predict(X)
    if float(rx.std()) < 1e-12 or float(ry.std()) < 1e-12:
        return None, int(mask.sum())
    return float(np.corrcoef(rx, ry)[0, 1]), int(mask.sum())


def load() -> pd.DataFrame:
    maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
    feat = pd.read_parquet(FEATURES_DIR / "maps.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    maps["game_uid"] = maps["game_uid"].astype(str)
    feat["game_uid"] = feat["game_uid"].astype(str)
    keep = [c for c in ["game_uid", "mu_diff"] if c in feat.columns]
    df = maps.merge(feat[keep], on="game_uid", how="left")
    df["mu_diff"] = pd.to_numeric(df.get("mu_diff"), errors="coerce").fillna(0.0)
    df["y_blue_win"] = pd.to_numeric(df["y_blue_win"], errors="coerce")
    df["gold15"] = pd.to_numeric(df.get("blue_golddiffat15"), errors="coerce")
    df["total_kills"] = pd.to_numeric(df.get("total_kills", df.get("y_total_kills")), errors="coerce")
    df["length_min"] = pd.to_numeric(df.get("length_min"), errors="coerce")
    if df["length_min"].isna().all():
        df["length_min"] = pd.to_numeric(df.get("blue_gamelength"), errors="coerce") / 60.0
    df["under_29_5"] = (df["total_kills"] <= 29).astype(float)
    df["long_35"] = (df["length_min"] >= 35).astype(float)
    for c in ("blue_firsttower", "y_blue_firstblood", "blue_firstdragon"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    p = players.copy()
    p["game_uid"] = p["game_uid"].astype(str)
    p["champion"] = p["champion"].map(lambda c: normalize_champ(str(c)))
    p["side"] = p["side"].astype(str).str.title()
    rows = []
    for gid, g in p.groupby("game_uid"):
        blue = g[g["side"] == "Blue"]["champion"].tolist()[:5]
        red = g[g["side"] == "Red"]["champion"].tolist()[:5]
        if len(blue) < 5 or len(red) < 5:
            continue
        feat_a = draft_archetype_features(blue, red)
        from lol_kills.draft_archetypes import side_archetype_counts

        bcounts = side_archetype_counts(blue)
        rcounts = side_archetype_counts(red)
        conc = kill_conc_from_draft(blue, red)
        row = {
            "game_uid": gid,
            "tempo_sum": _axis(feat_a, BEATDOWN_WEIGHTS, "blue") + _axis(feat_a, BEATDOWN_WEIGHTS, "red"),
            "inev_sum": _axis(feat_a, INEVITABILITY_WEIGHTS, "blue") + _axis(feat_a, INEVITABILITY_WEIGHTS, "red"),
            "length_lean": (
                _axis(feat_a, INEVITABILITY_WEIGHTS, "blue")
                + _axis(feat_a, INEVITABILITY_WEIGHTS, "red")
                - _axis(feat_a, BEATDOWN_WEIGHTS, "blue")
                - _axis(feat_a, BEATDOWN_WEIGHTS, "red")
            ),
            "chaos_sum": (
                bcounts.get("arch_assassin", 0)
                + rcounts.get("arch_assassin", 0)
                + 0.7 * (bcounts.get("arch_early_snowball", 0) + rcounts.get("arch_early_snowball", 0))
                + 0.5 * (bcounts.get("arch_skirmisher", 0) + rcounts.get("arch_skirmisher", 0))
            ),
            "init_diff": _side_w(bcounts, INIT_W) - _side_w(bcounts, REACT_W) - (
                _side_w(rcounts, INIT_W) - _side_w(rcounts, REACT_W)
            ),
            "tf_sum": _side_w(bcounts, TEAMFIGHT_W) + _side_w(rcounts, TEAMFIGHT_W),
            "siege_sum": _side_w(bcounts, SIEGE_W) + _side_w(rcounts, SIEGE_W),
            "split_sum": _side_w(bcounts, SPLIT_W) + _side_w(rcounts, SPLIT_W),
            "combo_blue": _side_w(bcounts, COMBO_W),
            "fair_blue": _side_w(bcounts, FAIR_W),
            "blue_hypercarry": float(conc.get("blue_hypercarry") or 0),
            "max_carry_blue": float(conc.get("max_carry_blue") or 0),
        }
        row["combo_minus_fair_blue"] = row["combo_blue"] - row["fair_blue"]
        rows.append(row)
    adf = pd.DataFrame(rows)
    return df.merge(adf, on="game_uid", how="inner")


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    print(f"[dyn-val] n={len(df)}")
    elo = df["mu_diff"].astype(float).values / 400.0
    checks = []

    def add(name, claim, r, n, **extra):
        checks.append({"name": name, "claim": claim, "r": None if r is None else round(r, 4), "n": n, **extra})
        print(f"  {name}: r={r if r is None else round(r,4)} n={n}")

    r, n = _resid_corr(df["length_min"].values, df["length_lean"].values, elo)
    add("tempo_inev_length", "inev-tempo lean → longer games", r, n)
    r, n = _resid_corr(df["under_29_5"].values, df["length_lean"].values, elo)
    add("tempo_inev_under29", "inev-tempo lean → more unders (low kill pace)", r, n)
    r, n = _resid_corr(df["total_kills"].values, df["chaos_sum"].values, elo)
    add("chaos_kills", "chaos sum → more kills", r, n)
    if "y_blue_firstblood" in df.columns:
        r, n = _resid_corr(df["y_blue_firstblood"].values, df["init_diff"].values, elo)
        add("init_firstblood", "initiative diff → first blood", r, n)
    if "blue_firsttower" in df.columns:
        r, n = _resid_corr(df["blue_firsttower"].values, df["init_diff"].values, elo)
        add("init_firsttower", "initiative diff → first tower", r, n)
    r, n = _resid_corr(df["long_35"].values, df["tf_sum"].values, elo)
    add("tf_long35", "teamfight sum → long games", r, n)

    # Pilot: hypercarry behind @15 WR vs non
    sub = df.dropna(subset=["gold15", "y_blue_win", "blue_hypercarry"]).copy()
    behind = sub[sub["gold15"] < -1000]
    hyp = behind[behind["blue_hypercarry"] == 1]
    noh = behind[behind["blue_hypercarry"] == 0]
    pilot = {
        "behind_hyper_wr": round(float(hyp["y_blue_win"].mean()), 4) if len(hyp) >= 80 else None,
        "behind_hyper_n": int(len(hyp)),
        "behind_committee_wr": round(float(noh["y_blue_win"].mean()), 4) if len(noh) >= 80 else None,
        "behind_committee_n": int(len(noh)),
        "gap_pp": None,
    }
    if pilot["behind_hyper_wr"] is not None and pilot["behind_committee_wr"] is not None:
        pilot["gap_pp"] = round(100 * (pilot["behind_hyper_wr"] - pilot["behind_committee_wr"]), 2)
    print(f"  pilot_behind: {pilot}")

    # Combo ahead → shorter?
    ahead = sub[sub["gold15"] > 1000].copy()
    if len(ahead) >= 200:
        r, n = _r(ahead["length_min"].values, ahead["combo_minus_fair_blue"].values, min_n=200)
        add("combo_ahead_length", "combo-fair when ahead → shorter (neg r)", r, n)

    # Sign checks vs claims
    passed = 0
    total = 0
    for c in checks:
        if c["r"] is None:
            continue
        total += 1
        name = c["name"]
        r = c["r"]
        ok = False
        if name in ("tempo_inev_length", "chaos_kills", "init_firstblood", "init_firsttower", "tf_long35", "tempo_inev_under29"):
            ok = r > 0
        elif name in ("combo_ahead_length",):
            ok = r < 0
        c["sign_ok"] = ok
        passed += int(ok)

    summary = (
        f"{passed}/{total} directional OE checks passed; "
        f"pilot behind gap_pp={pilot.get('gap_pp')} "
        f"(hyper WR {pilot.get('behind_hyper_wr')} vs committee {pilot.get('behind_committee_wr')})"
    )
    art = {
        "version": 1,
        "n_maps": int(len(df)),
        "checks": checks,
        "pilot_behind": pilot,
        "summary": summary,
        "note": "Elo-residualized correlations where noted. Directional validation for betting lenses — not causal proof.",
    }
    OUT.write_text(json.dumps(art, indent=2))
    print(f"[dyn-val] {summary}")
    print(f"[dyn-val] wrote {OUT}")


if __name__ == "__main__":
    main()
