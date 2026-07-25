#!/usr/bin/env python3
"""
Multi-lens champion OE profile — literature-mapped use of the full OE schema.

Lenses (arxiv / OE usage)
-------------------------
1. Fight (SIDO-style) — arXiv:2403.04873
   Resource *use* residual: DPM | egpm + damagetakenperminute + length.
   Damage-taken covariate standardizes aggression (SIDO Eq. 2).

2. Lane / phase gold (MOBA-Slice + live WR) — arXiv:1807.08360, 2309.02449
   golddiff@10/15/20/25 role residuals → early→late curve (time-slice advantage).

3. Vision — OE wards / visionscore; mid/late vision separates winners in KPI work
   (phase-aware vision literature; OE: vspm, wpm, wardskilled).

4. Structure pressure — towers matter most in early WR models (2309.02449 / OE theses)
   damagetotowers residual vs role+egpm.

5. Objectives (team→champ attribution) — arXiv:2309.02449 / OE theses
   Join OE *team* void_grubs, dragons, atakhans, towers, barons, heralds
   (and opp diffs) onto player rows by game_uid+side; residual vs role.

Composite → oe_pp clipped for tier_score (alongside Elo + Blade-Chest).
Supports last-patch-only + per-region scoring.

  python3 -m lol_kills.research.champ_oe_lenses
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import MODELS_DIR, PARQUET_DIR

OUT = MODELS_DIR / "champ_oe_lenses.json"
LIT = MODELS_DIR / "oe_arxiv_usage.json"

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
MIN_N = 18
PRIOR_N = 35.0

# Scale residuals → approximate pp for tier mixing
FIGHT_TO_PP = 0.010  # ~100 DPM resid ≈ 1pp
LANE_TO_PP = 0.00035  # ~3000 golddiff resid ≈ 1pp
VISION_TO_PP = 0.35  # ~3 vspm resid ≈ 1pp
TOWER_TO_PP = 0.0008  # damagetotowers residual

# Composite weights (sum≈1) — SIDO fight + phase gold + objectives
W_FIGHT = 0.32
W_LANE = 0.24
W_VISION = 0.12
W_TOWER = 0.12
W_OBJ = 0.20
OE_CLIP = 4.0
OBJ_TO_PP = 0.55  # ~1.8 obj_score resid ≈ 1pp

# Board scopes requested by user:
#   europe → LEC only; americas → LCS only; intl → MSI and EWC separated.
# Prefer SCORE_PATCH (16.13); if a league has no games there, use that league's latest OE patch.
SCORE_PATCH_DEFAULT = 16.13
BOARD_SCOPES = (
    {"key": "msi", "label": "MSI", "leagues": ("MSI",), "prefer_patch": SCORE_PATCH_DEFAULT},
    {"key": "ewc", "label": "EWC", "leagues": ("EWC",), "prefer_patch": SCORE_PATCH_DEFAULT},
    {"key": "lec", "label": "LEC", "leagues": ("LEC",), "prefer_patch": SCORE_PATCH_DEFAULT},
    {"key": "lcs", "label": "LCS", "leagues": ("LCS",), "prefer_patch": SCORE_PATCH_DEFAULT},
)
SCOPE_ORDER = tuple(s["key"] for s in BOARD_SCOPES)


def _norm_league(x) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip().upper()


def latest_patch_for_leagues(leagues: tuple[str, ...]) -> float | None:
    """Latest OE patch with ≥1 team-game in any of the leagues."""
    team = pd.read_parquet(PARQUET_DIR / "oe_team_games.parquet", columns=["league", "patch"])
    want = {lg.upper() for lg in leagues}
    team = team[team["league"].map(_norm_league).isin(want)].copy()
    team["patch_f"] = pd.to_numeric(team["patch"], errors="coerce")
    team = team.dropna(subset=["patch_f"])
    if team.empty:
        return None
    return float(team["patch_f"].max())


def resolve_scope_patch(scope: dict) -> float | None:
    prefer = float(scope.get("prefer_patch") or SCORE_PATCH_DEFAULT)
    leagues = tuple(scope["leagues"])
    # Probe players for prefer patch
    p = pd.read_parquet(PARQUET_DIR / "players.parquet", columns=["league", "patch"])
    p["patch_f"] = pd.to_numeric(p["patch"], errors="coerce")
    want = {lg.upper() for lg in leagues}
    n = int(
        (
            p["league"].map(_norm_league).isin(want) & (p["patch_f"] == prefer)
        ).sum()
    )
    if n >= 50:  # ≥5 maps worth of player rows
        return prefer
    return latest_patch_for_leagues(leagues)


# Back-compat aliases used by older callers
REGION_ORDER = SCOPE_ORDER


def league_to_region(league: str | None) -> str:
    """Map a single league code to board key (msi/ewc/lec/lcs) or other."""
    u = _norm_league(league)
    for scope in BOARD_SCOPES:
        if u in {lg.upper() for lg in scope["leagues"]}:
            return scope["key"]
    return "other"

ARXIV_USAGE = {
    "title": "OE full-schema → modeling lenses (arxiv-grounded)",
    "papers": [
        {
            "id": "2403.04873",
            "title": "The SIDO Performance Model for League of Legends",
            "use": (
                "Gold = resource gain; damage = resource use. "
                "Condition damage on damage-taken (aggression). "
                "Champion effects as random effects — we use role Ridge residuals as champ effects."
            ),
            "oe_cols": [
                "earned gpm",
                "totalgold",
                "dpm",
                "damagetochampions",
                "damagetakenperminute",
                "damageshare",
                "earnedgoldshare",
            ],
            "lens": "fight",
        },
        {
            "id": "2309.02449",
            "title": "League of Legends: Real-Time Result Prediction",
            "use": (
                "Feature importance shifts by elapsed time: kills/dragons/towers early; "
                "total gold later. Use multipoint gold diffs + objectives, not a single snapshot."
            ),
            "oe_cols": [
                "golddiffat10",
                "golddiffat15",
                "golddiffat20",
                "golddiffat25",
                "towers",
                "dragons",
                "barons",
                "inhibitors",
                "firsttower",
                "firstdragon",
                "void_grubs",
                "atakhans",
            ],
            "lens": "lane + objectives",
        },
        {
            "id": "1807.08360",
            "title": "MOBA-Slice: Time Slice Relative Advantage",
            "use": "Value network over time slices — phase gold/XP/CS curves as advantage trajectory.",
            "oe_cols": [
                "goldat10",
                "goldat15",
                "goldat20",
                "goldat25",
                "xpat10",
                "xpat15",
                "xpat20",
                "xpat25",
                "csdiffat10",
                "csdiffat15",
                "csdiffat20",
                "csdiffat25",
            ],
            "lens": "lane",
        },
        {
            "id": "1806.10130",
            "title": "The Art of Drafting",
            "use": "Draft as combinatorial game; blind vs counter value — already in Blade-Chest path.",
            "oe_cols": ["champion", "ban1-5", "pick1-5", "firstPick", "side"],
            "lens": "draft (existing)",
        },
        {
            "id": "2501.10049",
            "title": "PandaSkill — role-specific performance from player stats",
            "use": "Role-independent models of in-game stats → performance score. We residualize within role.",
            "oe_cols": "full player OE suite (role-normalized)",
            "lens": "all lenses role-conditional",
        },
        {
            "id": "2204.12750",
            "title": "DraftRec",
            "use": "Synergy/competence from compositions — complements counterability, not OE stats.",
            "oe_cols": ["champion", "team composition"],
            "lens": "draft (existing)",
        },
        {
            "id": "2502.03998",
            "title": "Online Learning of Counter Categories and Ratings",
            "use": "Rating + residual counter table — Blade-Chest counterability path.",
            "oe_cols": ["champion matchups"],
            "lens": "counterability (existing)",
        },
        {
            "id": "1806.02643",
            "title": "Re-evaluating Evaluation (mElo / Nash averaging)",
            "use": "Intransitive evaluation — cyclic matchup term.",
            "oe_cols": ["matchup outcomes"],
            "lens": "counterability (existing)",
        },
    ],
    "objective_note": (
        "void_grubs / atakhans / dragons / towers / barons / heralds joined from OE team rows "
        "onto player games by game_uid+side; champ obj residual vs role."
    ),
    "tier_formula": (
        "tier_score = elo_pp - λ·cb_tax + clip(oe_pp) "
        "where oe_pp = 0.32·fight + 0.24·lane + 0.12·vision + 0.12·tower + 0.20·obj"
    ),
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _shrink(x: float, n: int, prior: float = PRIOR_N) -> float:
    return float(x) * (n / (n + prior))


def _attach_team_objectives(df: pd.DataFrame) -> pd.DataFrame:
    """Join OE team objective counts onto player rows (game_uid + side)."""
    team = pd.read_parquet(PARQUET_DIR / "oe_team_games.parquet")
    team = team.copy()
    team["game_uid"] = team["gameid"].astype(str)
    team["side_key"] = team["side"].astype(str).str.strip().str.title()
    team.loc[team["side_key"].str.lower().isin(["blue", "1"]), "side_key"] = "Blue"
    team.loc[team["side_key"].str.lower().isin(["red", "2"]), "side_key"] = "Red"

    obj_cols = [
        "void_grubs",
        "dragons",
        "atakhans",
        "towers",
        "barons",
        "heralds",
        "opp_void_grubs",
        "opp_dragons",
        "opp_atakhans",
        "opp_towers",
        "opp_barons",
        "opp_heralds",
        "firsttower",
        "firstdragon",
        "firstbaron",
        "firstherald",
    ]
    keep = ["game_uid", "side_key"] + [c for c in obj_cols if c in team.columns]
    t = team[keep].copy()
    for c in keep[2:]:
        t[c] = _num(t[c])

    # Differential / absolute objective score (team-level, attributed to each player on that side)
    vg = t["void_grubs"] if "void_grubs" in t.columns else 0.0
    dr = t["dragons"] if "dragons" in t.columns else 0.0
    at = t["atakhans"] if "atakhans" in t.columns else 0.0
    tw = t["towers"] if "towers" in t.columns else 0.0
    ba = t["barons"] if "barons" in t.columns else 0.0
    he = t["heralds"] if "heralds" in t.columns else 0.0
    ovg = t["opp_void_grubs"] if "opp_void_grubs" in t.columns else 0.0
    odr = t["opp_dragons"] if "opp_dragons" in t.columns else 0.0
    oat = t["opp_atakhans"] if "opp_atakhans" in t.columns else 0.0
    otw = t["opp_towers"] if "opp_towers" in t.columns else 0.0
    # Weighted take + diff (literature: towers/dragons dominate early WR)
    t["obj_score"] = (
        0.12 * (vg.fillna(0) - ovg.fillna(0))
        + 0.22 * (dr.fillna(0) - odr.fillna(0))
        + 0.18 * (at.fillna(0) - oat.fillna(0))
        + 0.28 * (tw.fillna(0) - otw.fillna(0))
        + 0.12 * ba.fillna(0)
        + 0.08 * he.fillna(0)
    )
    if "firsttower" in t.columns:
        t["obj_score"] = t["obj_score"] + 0.15 * t["firsttower"].fillna(0)
    if "firstdragon" in t.columns:
        t["obj_score"] = t["obj_score"] + 0.08 * t["firstdragon"].fillna(0)

    out = df.copy()
    out["game_uid"] = out["game_uid"].astype(str) if "game_uid" in out.columns else out.get("gameid", pd.Series(dtype=str)).astype(str)
    out["side_key"] = out["side"].astype(str).str.strip().str.title()
    out.loc[out["side_key"].str.lower().isin(["blue", "1"]), "side_key"] = "Blue"
    out.loc[out["side_key"].str.lower().isin(["red", "2"]), "side_key"] = "Red"
    merged = out.merge(
        t[["game_uid", "side_key", "obj_score"] + [c for c in ("void_grubs", "dragons", "atakhans", "towers", "barons") if c in t.columns]],
        on=["game_uid", "side_key"],
        how="left",
        suffixes=("", "_team"),
    )
    return merged


def load_player_frame(
    *,
    patches: tuple[float, ...] | None = None,
    region: str | None = None,
    leagues: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    p = pd.read_parquet(PARQUET_DIR / "players.parquet")
    df = p.copy()
    df["pos"] = df["position"].astype(str).str.lower().map(lambda x: POS_MAP.get(x, x))
    df = df[df["pos"].isin(ROLES)].copy()
    df["champion"] = df["champion"].map(lambda x: normalize_champ(str(x)) if pd.notna(x) else None)
    df["patch_f"] = _num(df["patch"])
    df["league_u"] = df["league"].map(_norm_league)
    df["region"] = df["league"].map(league_to_region)
    if patches is not None:
        df = df[df["patch_f"].isin(patches)]
    if leagues is not None:
        want = {lg.upper() for lg in leagues}
        df = df[df["league_u"].isin(want)]
    elif region is not None and region not in ("global", "other"):
        # region key == board scope key (msi/ewc/lec/lcs)
        scope = next((s for s in BOARD_SCOPES if s["key"] == region), None)
        if scope:
            want = {lg.upper() for lg in scope["leagues"]}
            df = df[df["league_u"].isin(want)]
        else:
            df = df[df["region"] == region]

    rename_num = {
        "dpm": "dpm",
        "earned gpm": "egpm",
        "damagetakenperminute": "dtpm",
        "damagetotowers": "tow_dmg",
        "visionscore": "visionscore",
        "vspm": "vspm",
        "wpm": "wpm",
        "wardsplaced": "wardsplaced",
        "wardskilled": "wardskilled",
        "damageshare": "dshare",
        "earnedgoldshare": "gshare",
        "result": "result",
        "gamelength": "gamelength",
        "golddiffat10": "gd10",
        "golddiffat15": "gd15",
        "golddiffat20": "gd20",
        "golddiffat25": "gd25",
    }
    for src, dst in rename_num.items():
        if src in df.columns:
            df[dst] = _num(df[src])
        else:
            df[dst] = np.nan

    df["length_min"] = df["gamelength"] / 60.0
    df = df.dropna(subset=["champion", "pos", "result"])
    df = _attach_team_objectives(df)
    return df.reset_index(drop=True)


def _fit_role_ridge(
    df: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    *,
    min_n: int = 200,
    alpha: float = 5.0,
) -> dict[str, dict]:
    curves: dict[str, dict] = {}
    for role, sub in df.groupby("pos"):
        use = sub.dropna(subset=[y_col] + x_cols)
        if len(use) < min_n:
            continue
        X = np.column_stack([np.ones(len(use))] + [use[c].values for c in x_cols])
        y = use[y_col].values
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(X, y)
        pred = model.predict(X)
        rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        curves[str(role)] = {
            "coef": [float(c) for c in model.coef_],
            "x_cols": list(x_cols),
            "n": int(len(use)),
            "rmse": round(rmse, 3),
            "y_mean": round(float(y.mean()), 3),
        }
    return curves


def _attach_resid(
    df: pd.DataFrame,
    curves: dict[str, dict],
    y_col: str,
    out_col: str,
) -> pd.DataFrame:
    out = df.copy()
    resid = pd.Series(np.nan, index=out.index, dtype=float)
    for role, cur in curves.items():
        x_cols = cur["x_cols"]
        mask = (out["pos"] == role) & out[y_col].notna()
        for c in x_cols:
            mask = mask & out[c].notna()
        if not mask.any():
            continue
        coef = np.array(cur["coef"], dtype=float)
        X = np.column_stack(
            [np.ones(int(mask.sum()))] + [out.loc[mask, c].values for c in x_cols]
        )
        pred = X @ coef
        resid.loc[mask] = out.loc[mask, y_col].values - pred
    out[out_col] = resid
    return out


def _aggregate_champs(score: pd.DataFrame, *, min_n: int) -> dict[str, dict]:
    champs: dict[str, dict] = {}
    for champ, g in score.groupby("champion"):
        n = int(len(g))
        if n < min_n:
            continue
        role = str(g["pos"].value_counts().idxmax())

        def mean_resid(col: str) -> float | None:
            s = g[col].dropna()
            return None if len(s) < max(6, min_n // 3) else float(s.mean())

        fight_r = mean_resid("fight_resid")
        lane_r = mean_resid("lane_resid")
        vis_r = mean_resid("vision_resid")
        tow_r = mean_resid("tower_resid")
        obj_r = mean_resid("obj_resid")

        fight_pp = _shrink(fight_r or 0.0, n) * FIGHT_TO_PP if fight_r is not None else 0.0
        lane_pp = _shrink(lane_r or 0.0, n) * LANE_TO_PP if lane_r is not None else 0.0
        vision_pp = _shrink(vis_r or 0.0, n) * VISION_TO_PP if vis_r is not None else 0.0
        tower_pp = _shrink(tow_r or 0.0, n) * TOWER_TO_PP if tow_r is not None else 0.0
        obj_pp = _shrink(obj_r or 0.0, n) * OBJ_TO_PP if obj_r is not None else 0.0

        phase = {}
        for tcol, label in zip(
            ("gd10", "gd15", "gd20", "gd25"), ("@10", "@15", "@20", "@25")
        ):
            r = mean_resid(f"{tcol}_resid")
            phase[label] = None if r is None else round(r, 1)

        oe_pp = (
            W_FIGHT * fight_pp
            + W_LANE * lane_pp
            + W_VISION * vision_pp
            + W_TOWER * tower_pp
            + W_OBJ * obj_pp
        )
        oe_pp = float(np.clip(oe_pp, -OE_CLIP, OE_CLIP))

        champs[str(champ)] = {
            "n": n,
            "role": role,
            "fight_resid": None if fight_r is None else round(fight_r, 2),
            "lane_resid": None if lane_r is None else round(lane_r, 1),
            "vision_resid": None if vis_r is None else round(vis_r, 3),
            "tower_resid": None if tow_r is None else round(tow_r, 1),
            "obj_resid": None if obj_r is None else round(obj_r, 3),
            "phase_gold_resid": phase,
            "mean_void_grubs": round(float(g["void_grubs"].mean()), 2) if "void_grubs" in g and g["void_grubs"].notna().any() else None,
            "mean_dragons": round(float(g["dragons"].mean()), 2) if "dragons" in g and g["dragons"].notna().any() else None,
            "mean_atakhans": round(float(g["atakhans"].mean()), 2) if "atakhans" in g and g["atakhans"].notna().any() else None,
            "fight_pp": round(fight_pp, 3),
            "lane_pp": round(lane_pp, 3),
            "vision_pp": round(vision_pp, 3),
            "tower_pp": round(tower_pp, 3),
            "obj_pp": round(obj_pp, 3),
            "oe_pp": round(oe_pp, 3),
            "impact_pp": round(oe_pp, 3),
        }
    return champs


def build_lenses(
    *,
    fit_patches: tuple[float, ...] | None = None,
    score_patches: tuple[float, ...] | None = None,
    region: str | None = None,
    leagues: tuple[str, ...] | None = None,
    min_n: int = MIN_N,
    ridge_min_n: int | None = None,
) -> dict:
    """
    Fit role curves, score champs on score_patches × optional league filter.
    """
    fit_df = load_player_frame(patches=fit_patches, region=region, leagues=leagues)
    label = (
        "+".join(leagues) if leagues
        else (region or "global")
    )
    print(f"[oe_lenses] [{label}] fit rows={len(fit_df)} patches={fit_patches or 'all'}")

    rmin = ridge_min_n if ridge_min_n is not None else (80 if region else 150)

    fight_x = ["egpm", "dtpm", "length_min"]
    fight_fit = fit_df.dropna(subset=["dpm", "egpm"]).copy()
    if fight_fit.empty:
        return {
            "region": region,
            "n_fit_rows": 0,
            "n_score_rows": 0,
            "n_champs": 0,
            "champs": {},
            "weights": {
                "fight": W_FIGHT,
                "lane": W_LANE,
                "vision": W_VISION,
                "tower": W_TOWER,
                "obj": W_OBJ,
            },
        }
    fight_fit["dtpm"] = fight_fit["dtpm"].fillna(fight_fit.groupby("pos")["dtpm"].transform("median"))
    fight_fit["length_min"] = fight_fit["length_min"].fillna(32.0)
    fight_curves = _fit_role_ridge(fight_fit, "dpm", fight_x, min_n=rmin)

    lane_curves: dict[str, dict[str, dict]] = {}
    for tcol in ("gd10", "gd15", "gd20", "gd25"):
        lane_curves[tcol] = _fit_role_ridge(
            fit_df.assign(length_min=fit_df["length_min"].fillna(32.0)),
            tcol,
            ["length_min"],
            min_n=max(60, rmin - 20),
            alpha=10.0,
        )

    vis_fit = fit_df.dropna(subset=["vspm"]).copy()
    vis_fit["length_min"] = vis_fit["length_min"].fillna(32.0)
    vision_curves = _fit_role_ridge(vis_fit, "vspm", ["length_min"], min_n=max(60, rmin - 20))

    tow_fit = fit_df.dropna(subset=["tow_dmg", "egpm"]).copy()
    tow_fit["length_min"] = tow_fit["length_min"].fillna(32.0)
    tower_curves = _fit_role_ridge(
        tow_fit, "tow_dmg", ["egpm", "length_min"], min_n=max(60, rmin - 20)
    )

    obj_fit = fit_df.dropna(subset=["obj_score"]).copy()
    obj_fit["length_min"] = obj_fit["length_min"].fillna(32.0)
    obj_curves = _fit_role_ridge(obj_fit, "obj_score", ["length_min"], min_n=max(60, rmin - 20))

    if score_patches is None or score_patches == fit_patches:
        score_df = fit_df
    else:
        score_df = load_player_frame(
            patches=score_patches, region=region, leagues=leagues
        )
        print(f"[oe_lenses] [{label}] score rows={len(score_df)} patches={score_patches}")

    score = score_df.copy()
    score["dtpm"] = score["dtpm"].fillna(score.groupby("pos")["dtpm"].transform("median"))
    score["length_min"] = score["length_min"].fillna(32.0)
    score = _attach_resid(score, fight_curves, "dpm", "fight_resid")
    for tcol in ("gd10", "gd15", "gd20", "gd25"):
        score = _attach_resid(score, lane_curves[tcol], tcol, f"{tcol}_resid")
    score = _attach_resid(score, vision_curves, "vspm", "vision_resid")
    score = _attach_resid(score, tower_curves, "tow_dmg", "tower_resid")
    score = _attach_resid(score, obj_curves, "obj_score", "obj_resid")
    lane_cols = [f"{c}_resid" for c in ("gd10", "gd15", "gd20", "gd25")]
    score["lane_resid"] = score[lane_cols].mean(axis=1, skipna=True)

    champs = _aggregate_champs(score, min_n=min_n)
    return {
        "region": region or "global",
        "fit_patches": list(fit_patches) if fit_patches else None,
        "score_patches": list(score_patches) if score_patches else list(fit_patches or []),
        "n_fit_rows": int(len(fit_df)),
        "n_score_rows": int(len(score)),
        "n_champs": len(champs),
        "leagues": sorted(score["league"].astype(str).unique().tolist()) if len(score) else [],
        "weights": {
            "fight": W_FIGHT,
            "lane": W_LANE,
            "vision": W_VISION,
            "tower": W_TOWER,
            "obj": W_OBJ,
        },
        "clip": OE_CLIP,
        "curves": {
            "fight": fight_curves,
            "lane": {k: v for k, v in lane_curves.items()},
            "vision": vision_curves,
            "tower": tower_curves,
            "obj": obj_curves,
        },
        "champs": champs,
        "citations": [p["id"] for p in ARXIV_USAGE["papers"]],
    }


def build_lenses_for_scope(
    scope: dict,
    *,
    min_n: int = 8,
) -> dict:
    patch = resolve_scope_patch(scope)
    if patch is None:
        return {
            "key": scope["key"],
            "label": scope["label"],
            "leagues": list(scope["leagues"]),
            "patch": None,
            "n_champs": 0,
            "n_score_rows": 0,
            "champs": {},
            "note": "no OE games for this league",
        }
    art = build_lenses(
        fit_patches=(patch,),
        score_patches=(patch,),
        region=None,
        leagues=tuple(scope["leagues"]),
        min_n=min_n,
        ridge_min_n=40,
    )
    art["key"] = scope["key"]
    art["label"] = scope["label"]
    art["leagues"] = list(scope["leagues"])
    art["patch"] = patch
    art["patch_note"] = (
        f"prefer {scope.get('prefer_patch')}; using {patch}"
        if patch != float(scope.get("prefer_patch") or SCORE_PATCH_DEFAULT)
        else f"score patch {patch}"
    )
    return art


def build_lenses_by_region(
    *,
    patches: tuple[float, ...] = (16.13,),
    min_n: int = 8,
) -> dict:
    """Board-scope OE lenses: MSI, EWC, LEC, LCS (each league's resolved patch)."""
    by_region = {}
    for scope in BOARD_SCOPES:
        art = build_lenses_for_scope(scope, min_n=min_n)
        by_region[scope["key"]] = art
        print(
            f"[oe_lenses] {scope['key']} patch={art.get('patch')} "
            f"champs={art.get('n_champs')} rows={art.get('n_score_rows')} "
            f"({art.get('patch_note') or art.get('note')})"
        )
    # optional pooled MSI+EWC on preferred patch for reference
    prefer = float(patches[0]) if patches else SCORE_PATCH_DEFAULT
    pooled = build_lenses(
        fit_patches=(prefer,),
        score_patches=(prefer,),
        leagues=("MSI", "EWC"),
        min_n=min_n,
        ridge_min_n=80,
    )
    pooled["key"] = "intl_pooled"
    pooled["label"] = "MSI+EWC"
    pooled["patch"] = prefer
    by_region["intl_pooled"] = pooled
    return {
        "prefer_patch": prefer,
        "scopes": [dict(s) for s in BOARD_SCOPES],
        "by_region": by_region,
        "arxiv": ARXIV_USAGE,
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LIT.write_text(json.dumps(ARXIV_USAGE, indent=2))
    bundle = build_lenses_by_region(patches=(16.13,), min_n=8)
    OUT.write_text(json.dumps(bundle, indent=2))
    print(f"[oe_lenses] wrote {OUT}")


if __name__ == "__main__":
    main()
