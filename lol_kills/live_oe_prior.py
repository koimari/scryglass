"""
OE-warehouse live kill priors.

Uses `data/lol/warehouse/parquet/maps.parquet` (not LP games_raw) for:
  - league length distributions (EWC included)
  - team-involved pace (maps featuring either side)
  - ckpm / target totals / NB dispersion

This is the ground-truth aggregation for live Unders.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.paths import PARQUET_DIR

MAJORS = ("LCK", "LPL", "LEC", "LCS", "MSI", "EWC", "FST", "Worlds", "CBLOL")


def _ensure_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "length_min" not in df.columns and "gamelength" in df.columns:
        df["length_min"] = df["gamelength"].astype(float) / 60.0
    if "total_kills" not in df.columns:
        if "y_total_kills" in df.columns:
            df["total_kills"] = df["y_total_kills"]
        elif "blue_kills" in df.columns and "red_kills" in df.columns:
            df["total_kills"] = df["blue_kills"].astype(float) + df["red_kills"].astype(float)
    if "ckpm" not in df.columns and "length_min" in df.columns:
        df["ckpm"] = df["total_kills"].astype(float) / df["length_min"].clip(lower=1e-3)
    for c in ("blue_team", "red_team"):
        if c in df.columns:
            df[c] = df[c].map(lambda x: normalize_team(str(x)) if pd.notna(x) else x)
    if "league" in df.columns:
        df["league"] = df["league"].astype(str)
    return df.dropna(subset=["length_min", "total_kills"])


@lru_cache(maxsize=1)
def _maps() -> pd.DataFrame:
    path = PARQUET_DIR / "maps.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = [
        c
        for c in [
            "game_uid",
            "date",
            "league",
            "blue_team",
            "red_team",
            "length_min",
            "gamelength",
            "total_kills",
            "y_total_kills",
            "blue_kills",
            "red_kills",
            "ckpm",
        ]
        if True
    ]
    # read then filter existing
    raw = pd.read_parquet(path)
    keep = [c for c in cols if c in raw.columns]
    return _ensure_cols(raw[keep])


def _dispersion(totals: np.ndarray, mu: float) -> float:
    if len(totals) < 3 or mu <= 1:
        return 0.08
    var = float(np.var(totals, ddof=1))
    return float(max(0.04, (var - mu) / (mu * mu)))


def load_oe_pace_prior(
    league: str,
    team1: str | None = None,
    team2: str | None = None,
    *,
    min_lengths: int = 40,
) -> dict:
    """
    Priority:
      1) maps involving either team (any league) if n≥min
      2) this league's OE maps
      3) majors pool
    """
    df = _maps()
    if df.empty:
        return {
            "source": "empty_fallback",
            "lengths": [28.0, 30.0, 32.0, 34.0, 36.0],
            "totals": [24, 26, 28, 30, 32],
            "ckpm": 0.90,
            "target_mu": 28.0,
            "mean_length": 32.0,
            "dispersion": 0.08,
            "n": 0,
            "league": league,
        }

    t1 = normalize_team(team1) if team1 else None
    t2 = normalize_team(team2) if team2 else None
    lg = (league or "").upper()

    team_pool = pd.DataFrame()
    if t1 or t2:
        mask = False
        if t1:
            mask = (df["blue_team"] == t1) | (df["red_team"] == t1)
        if t2:
            mask = mask | (df["blue_team"] == t2) | (df["red_team"] == t2)
        team_pool = df[mask]

    league_pool = df[df["league"].str.upper() == lg] if lg else pd.DataFrame()
    majors_pool = df[df["league"].isin(MAJORS)]

    if len(team_pool) >= min_lengths:
        pool, source = team_pool, f"oe_team_maps:{t1}|{t2}"
    elif len(league_pool) >= min_lengths:
        pool, source = league_pool, f"oe_league:{lg}"
    elif len(league_pool) >= 20:
        pool, source = league_pool, f"oe_league_thin:{lg}"
    else:
        pool, source = majors_pool if len(majors_pool) else df, "oe_majors"

    lengths = pool["length_min"].astype(float).tolist()
    totals = pool["total_kills"].astype(float).tolist()
    mean_len = float(np.mean(lengths))
    target_mu = float(np.mean(totals))
    ckpm = float(np.mean(pool["ckpm"])) if "ckpm" in pool.columns else target_mu / max(mean_len, 1e-3)
    # Prefer consistent identity: target / mean_len
    ckpm = target_mu / max(mean_len, 1e-3)

    return {
        "source": source,
        "lengths": lengths,
        "totals": totals,
        "ckpm": round(ckpm, 4),
        "target_mu": round(target_mu, 3),
        "mean_length": round(mean_len, 3),
        "dispersion": round(_dispersion(np.asarray(totals, dtype=float), target_mu), 4),
        "n": int(len(pool)),
        "league": lg,
        "p_under_hist": {
            str(L): round(float(np.mean(np.asarray(totals) <= L)), 4)
            for L in (26, 28, 29, 30, 32, 34, 36)
        },
    }


def blend_pair_model_mu(oe: dict, pair_mu: float | None, w_pair: float = 0.35) -> dict:
    """Optional blend with legacy pair_model mean when available."""
    if pair_mu is None or not math.isfinite(pair_mu) or pair_mu <= 0:
        return oe
    out = dict(oe)
    out["target_mu"] = round((1 - w_pair) * oe["target_mu"] + w_pair * float(pair_mu), 3)
    out["ckpm"] = round(out["target_mu"] / max(oe["mean_length"], 1e-3), 4)
    out["source"] = oe["source"] + "+pair_blend"
    return out
