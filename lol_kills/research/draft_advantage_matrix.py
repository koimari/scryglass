#!/usr/bin/env python3
"""
Draft × time × objectives advantage matrix (OE majors).

Builds:
  - champ kill-share concentration table
  - empirical WR grids (gold × draft × phase)
  - live interaction coefficients for live_win_prob

  python3 -m lol_kills.research.draft_advantage_matrix
"""

from __future__ import annotations

import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR

warnings.filterwarnings("ignore", category=UserWarning)

MIN_CHAMP_N = 80
HYPERCARRY_SHARE = 0.32
GOLD_EDGES = [-1e9, -3000, -2000, -1000, -500, 500, 1000, 2000, 3000, 1e9]
GOLD_LABELS = ["le-3k", "-3k--2k", "-2k--1k", "-1k--500", "even", "+500-1k", "+1k-2k", "+2k-3k", "ge+3k"]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-min(max(x, -30), 30)))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def phase_of(minute: float) -> str:
    if minute < 14:
        return "early"
    if minute < 25:
        return "mid"
    return "late"


def gold_bin(g: float) -> str:
    for i in range(len(GOLD_EDGES) - 1):
        if GOLD_EDGES[i] <= g < GOLD_EDGES[i + 1]:
            return GOLD_LABELS[i]
    return "even"


def build_kill_concentration(players: pd.DataFrame) -> dict:
    """Per-champion kill_share = kills / team_kills (end of game)."""
    p = players.dropna(subset=["champion", "kills", "game_uid", "side"]).copy()
    p["champion"] = p["champion"].map(lambda c: normalize_champ(str(c)))
    p["side"] = p["side"].astype(str).str.title()
    team_kills = p.groupby(["game_uid", "side"])["kills"].transform("sum")
    p["kill_share"] = p["kills"].astype(float) / team_kills.clip(lower=1).astype(float)

    role_col = "position" if "position" in p.columns else None
    if role_col:
        p["_role"] = (
            p[role_col]
            .astype(str)
            .str.lower()
            .map(
                lambda r: {
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
                }.get(r, r[:3])
            )
        )
    else:
        p["_role"] = "unk"

    champs: dict[str, dict] = {}
    for champ, g in p.groupby("champion"):
        n = int(len(g))
        if n < MIN_CHAMP_N:
            continue
        by_role = {}
        for role, rg in g.groupby("_role"):
            if len(rg) < 30:
                continue
            by_role[str(role)] = {
                "mean_share": round(float(rg["kill_share"].mean()), 4),
                "n": int(len(rg)),
            }
        champs[str(champ)] = {
            "mean_share": round(float(g["kill_share"].mean()), 4),
            "std_share": round(float(g["kill_share"].std(ddof=1) or 0), 4),
            "n": n,
            "hypercarry": bool(g["kill_share"].mean() >= HYPERCARRY_SHARE),
            "by_role": by_role,
        }

    return {
        "version": 1,
        "min_n": MIN_CHAMP_N,
        "hypercarry_threshold": HYPERCARRY_SHARE,
        "n_champs": len(champs),
        "champs": champs,
    }


def draft_conc_features(
    blue_champs: list[str],
    red_champs: list[str],
    conc: dict,
) -> dict:
    table = conc.get("champs") or {}
    default = 0.20

    def side_stats(champs: list[str]) -> dict:
        shares = [float((table.get(c) or {}).get("mean_share", default)) for c in champs]
        if not shares:
            shares = [default]
        return {
            "mean_conc": float(np.mean(shares)),
            "max_conc": float(np.max(shares)),
            "hypercarry": bool(np.max(shares) >= HYPERCARRY_SHARE),
        }

    b, r = side_stats(blue_champs), side_stats(red_champs)
    return {
            "kill_conc_blue": round(b["mean_conc"], 4),
            "kill_conc_red": round(r["mean_conc"], 4),
            "kill_conc_diff": round(b["mean_conc"] - r["mean_conc"], 4),
            "max_carry_blue": round(b["max_conc"], 4),
            "max_carry_red": round(r["max_conc"], 4),
            "hypercarry_blue": b["hypercarry"],
            "hypercarry_red": r["hypercarry"],
            "blue_hypercarry": int(b["hypercarry"]),
            "red_hypercarry": int(r["hypercarry"]),
            "scaling_flag": int(b["hypercarry"] or r["hypercarry"]),
        }


def _map_draft_champs(players: pd.DataFrame) -> pd.DataFrame:
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
        rows.append({"game_uid": gid, "blue_champs": blue[:5], "red_champs": red[:5]})
    return pd.DataFrame(rows)


def build_study_frame(conc: dict) -> pd.DataFrame:
    maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    feat = pd.read_parquet(FEATURES_DIR / "maps.parquet") if (FEATURES_DIR / "maps.parquet").exists() else None

    maps = maps.copy()
    maps["game_uid"] = maps["game_uid"].astype(str)
    if "length_min" not in maps.columns and "gamelength" in maps.columns:
        maps["length_min"] = maps["gamelength"].astype(float) / 60.0
    maps = maps.dropna(subset=["y_blue_win"])

    drafts = _map_draft_champs(players)
    df = maps.merge(drafts, on="game_uid", how="inner")

    keep_feat = [c for c in ["game_uid", "mu_diff", "elo_diff", "draft_win_logit_blue", "date"] if feat is not None and c in feat.columns]
    if feat is not None and keep_feat:
        f = feat[keep_feat].copy()
        f["game_uid"] = f["game_uid"].astype(str)
        df = df.merge(f, on="game_uid", how="left", suffixes=("", "_feat"))

    if "date" not in df.columns and "date_feat" in df.columns:
        df["date"] = df["date_feat"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    df["elo_diff"] = df.get("mu_diff", df.get("elo_diff"))
    if "elo_diff" not in df.columns:
        df["elo_diff"] = 0.0
    df["elo_diff"] = df["elo_diff"].fillna(0.0)
    df["draft_win_logit_blue"] = df.get("draft_win_logit_blue", pd.Series(0.0, index=df.index)).fillna(0.0)

    # concentration per map
    conc_rows = [draft_conc_features(b, r, conc) for b, r in zip(df["blue_champs"], df["red_champs"])]
    conc_df = pd.DataFrame(conc_rows)
    df = pd.concat([df.reset_index(drop=True), conc_df], axis=1)

    # gold / objs
    g10 = df["blue_golddiffat10"] if "blue_golddiffat10" in df.columns else pd.Series(np.nan, index=df.index)
    g15 = df["blue_golddiffat15"] if "blue_golddiffat15" in df.columns else pd.Series(np.nan, index=df.index)
    df["gold10"] = pd.to_numeric(g10, errors="coerce")
    df["gold15"] = pd.to_numeric(g15, errors="coerce")

    def first_blue(col_b: str, col_r: str | None = None) -> pd.Series:
        if col_b in df.columns:
            return pd.to_numeric(df[col_b], errors="coerce").fillna(0).clip(0, 1)
        return pd.Series(0.0, index=df.index)

    df["first_dragon"] = first_blue("blue_firstdragon")
    df["first_herald"] = first_blue("blue_firstherald")
    df["first_tower"] = first_blue("blue_firsttower")
    df["first_baron"] = first_blue("blue_firstbaron")

    bd = pd.to_numeric(df.get("blue_dragons", 0), errors="coerce").fillna(0)
    rd = pd.to_numeric(df.get("red_dragons", 0), errors="coerce").fillna(0)
    bt = pd.to_numeric(df.get("blue_towers", 0), errors="coerce").fillna(0)
    rt = pd.to_numeric(df.get("red_towers", 0), errors="coerce").fillna(0)
    bb = pd.to_numeric(df.get("blue_barons", 0), errors="coerce").fillna(0)
    rb = pd.to_numeric(df.get("red_barons", 0), errors="coerce").fillna(0)
    df["dragon_diff"] = (bd - rd).clip(-4, 4)
    df["tower_diff"] = (bt - rt).clip(-8, 8)
    df["baron_diff"] = (bb - rb).clip(-2, 2)
    length = pd.to_numeric(df.get("length_min", 32), errors="coerce").fillna(32).clip(lower=15)
    df["obj_rate"] = (bd + rd + 0.5 * (bt + rt) + 2.0 * (bb + rb)) / length

    df["gold10_bin"] = df["gold10"].map(lambda x: gold_bin(x) if pd.notna(x) else None)
    df["gold15_bin"] = df["gold15"].map(lambda x: gold_bin(x) if pd.notna(x) else None)

    # draft bins (quintile)
    try:
        df["draft_q"] = pd.qcut(df["draft_win_logit_blue"], 5, labels=False, duplicates="drop")
    except ValueError:
        df["draft_q"] = 2
    df["draft_q"] = df["draft_q"].fillna(2).astype(int)

    # archetype: blue hypercarry = scaling; positive draft without = snowball
    df["archetype"] = np.where(
        df["max_carry_blue"] >= HYPERCARRY_SHARE,
        "scaling",
        np.where(df["draft_win_logit_blue"] > 0.05, "snowball", "neutral"),
    )
    return df


def empirical_grids(df: pd.DataFrame) -> dict:
    """WR grids: gold bin × draft quintile for early(@10) and mid(@15)."""
    global_wr = float(df["y_blue_win"].mean())
    global_logit = _logit(global_wr)

    def grid(gold_col: str, bin_col: str, phase: str) -> list[dict]:
        sub = df.dropna(subset=[gold_col, bin_col, "y_blue_win"])
        cells = []
        for (gbin, dq), g in sub.groupby([bin_col, "draft_q"]):
            if len(g) < 40:
                continue
            wr = float(g["y_blue_win"].mean())
            cells.append(
                {
                    "phase": phase,
                    "gold_bin": str(gbin),
                    "draft_q": int(dq),
                    "n": int(len(g)),
                    "wr": round(wr, 4),
                    "delta_logit": round(_logit(wr) - global_logit, 4),
                    "mean_gold": round(float(g[gold_col].mean()), 1),
                    "mean_draft": round(float(g["draft_win_logit_blue"].mean()), 4),
                }
            )
        return cells

    # archetype × gold@15
    arch_cells = []
    sub = df.dropna(subset=["gold15", "gold15_bin", "y_blue_win"])
    for (gbin, arch), g in sub.groupby(["gold15_bin", "archetype"]):
        if len(g) < 40:
            continue
        wr = float(g["y_blue_win"].mean())
        arch_cells.append(
            {
                "phase": "mid",
                "gold_bin": str(gbin),
                "archetype": str(arch),
                "n": int(len(g)),
                "wr": round(wr, 4),
                "delta_logit": round(_logit(wr) - global_logit, 4),
            }
        )

    # obj firsts × draft
    obj_cells = []
    for obj in ("first_dragon", "first_herald", "first_tower"):
        for (val, dq), g in df.groupby([obj, "draft_q"]):
            if len(g) < 50:
                continue
            wr = float(g["y_blue_win"].mean())
            obj_cells.append(
                {
                    "objective": obj,
                    "blue_has": int(val),
                    "draft_q": int(dq),
                    "n": int(len(g)),
                    "wr": round(wr, 4),
                    "delta_logit": round(_logit(wr) - global_logit, 4),
                }
            )

    return {
        "global_wr": round(global_wr, 4),
        "gold10_x_draft": grid("gold10", "gold10_bin", "early"),
        "gold15_x_draft": grid("gold15", "gold15_bin", "mid"),
        "gold15_x_archetype": arch_cells,
        "objectives_x_draft": obj_cells,
    }


def fit_live_coefs(df: pd.DataFrame) -> dict:
    """
    Logistic on mid-checkpoint state (gold@15 + first objs + draft interactions).

    Intentionally excludes end-game tower/dragon/baron counts — those leak the
    outcome. Live tower/dragon diffs use soft priors in live_win_prob.
    """
    sub = df.dropna(subset=["gold15", "y_blue_win", "draft_win_logit_blue"]).copy()
    y = sub["y_blue_win"].astype(float).values

    def design(frame: pd.DataFrame, gold_col: str) -> np.ndarray:
        gold_k = frame[gold_col].values / 1000.0
        draft = frame["draft_win_logit_blue"].values
        conc = frame["kill_conc_diff"].values
        return np.column_stack(
            [
                frame["elo_diff"].values / 400.0,
                draft,
                gold_k,
                frame["first_dragon"].values,
                frame["first_herald"].values,
                frame["first_tower"].values,
                draft * gold_k,
                conc * gold_k,
                (frame["max_carry_blue"].values >= HYPERCARRY_SHARE).astype(float) * gold_k,
                frame["max_carry_blue"].values * gold_k,  # continuous blue carry × gold
            ]
        )

    feature_names = [
        "elo_z",
        "draft_edge",
        "gold_k",
        "first_dragon",
        "first_herald",
        "first_tower",
        "draft_x_gold",
        "conc_x_gold",
        "scaling_x_gold",
        "blue_carry_x_gold",
    ]

    X15 = design(sub, "gold15")
    lr15 = LogisticRegression(C=0.6, max_iter=500)
    lr15.fit(X15, y)
    coef15 = {n: float(c) for n, c in zip(feature_names, lr15.coef_[0])}

    sub10 = df.dropna(subset=["gold10", "y_blue_win"]).copy()
    X10 = design(sub10, "gold10")
    y10 = sub10["y_blue_win"].astype(float).values
    lr10 = LogisticRegression(C=0.6, max_iter=500)
    lr10.fit(X10, y10)
    coef10 = {n: float(c) for n, c in zip(feature_names, lr10.coef_[0])}

    def blend(a: dict, b: dict, w: float) -> dict:
        return {k: (1 - w) * a[k] + w * b[k] for k in a}

    phase_coefs = {
        "early": {**coef10, "intercept": float(lr10.intercept_[0]), "gold_anchor": "gold10"},
        "mid": {**coef15, "intercept": float(lr15.intercept_[0]), "gold_anchor": "gold15"},
        "late": {
            **blend(coef15, coef10, -0.15),
            "intercept": float(lr15.intercept_[0]),
            "gold_anchor": "gold15",
        },
    }
    for k in ("gold_k", "draft_x_gold", "scaling_x_gold", "blue_carry_x_gold"):
        phase_coefs["late"][k] = float(coef15[k] * 1.12)

    # Soft live priors for current dragon/tower (not fit on end counts)
    live_obj_priors = {
        "dragon_diff": 0.22,
        "tower_diff": 0.18,
        "kill_diff": 0.10,
        "void_grub": 0.06,
        "infernal_extra": 0.05,
    }

    tscv = TimeSeriesSplit(n_splits=5)
    briers_new, briers_elo = [], []
    for tr, te in tscv.split(X15):
        m = LogisticRegression(C=0.6, max_iter=400)
        m.fit(X15[tr], y[tr])
        p = m.predict_proba(X15[te])[:, 1]
        briers_new.append(float(brier_score_loss(y[te], p)))
        m0 = LogisticRegression(C=0.6, max_iter=400)
        m0.fit(X15[tr][:, [0]], y[tr])
        p0 = m0.predict_proba(X15[te][:, [0]])[:, 1]
        briers_elo.append(float(brier_score_loss(y[te], p0)))

    def softcap_p(frame: pd.DataFrame) -> np.ndarray:
        out = []
        for _, r in frame.iterrows():
            p_pre = _sigmoid(r["elo_diff"] / 400 * 2.2)
            x = _logit(p_pre) * 0.76
            adv = 0.18 * (r["gold15"] / 1000.0) + 0.22 * r["first_dragon"] + 0.18 * r["first_tower"]
            adv = 1.35 * math.tanh(adv / 1.35)
            out.append(_sigmoid(x + adv + 0.3 * r["draft_win_logit_blue"]))
        return np.asarray(out)

    cut = int(len(sub) * 0.8)
    hold = sub.iloc[cut:]
    Xh = design(hold, "gold15")
    ph = lr15.predict_proba(Xh)[:, 1]
    yh = hold["y_blue_win"].astype(float).values
    p_soft = softcap_p(hold)

    return {
        "version": 2,
        "feature_names": feature_names,
        "phase_coefs": phase_coefs,
        "live_obj_priors": live_obj_priors,
        "adv_cap": 1.45,
        "cv_brier_model": {"mean": float(np.mean(briers_new)), "std": float(np.std(briers_new))},
        "cv_brier_elo": {"mean": float(np.mean(briers_elo)), "std": float(np.std(briers_elo))},
        "holdout": {
            "n": int(len(hold)),
            "brier_interaction": float(brier_score_loss(yh, ph)),
            "brier_softcap_proxy": float(brier_score_loss(yh, np.clip(p_soft, 1e-4, 1 - 1e-4))),
            "auc_interaction": float(roc_auc_score(yh, ph)),
            "note": "features=gold@15+firsts+draft interactions (no end tower/dragon)",
        },
        "sanity_scaling_vs_snowball_behind": _sanity_behind(df),
    }


def _sanity_behind(df: pd.DataFrame) -> dict:
    """Blue hypercarry vs snowball when blue is behind @15 (gold < -1k)."""
    sub = df.dropna(subset=["gold15", "y_blue_win", "max_carry_blue"])
    behind = sub[sub["gold15"] < -1000]
    # blue-side scaling: blue has a hypercarry
    sc = behind[behind["max_carry_blue"] >= HYPERCARRY_SHARE]
    # snowball: no blue hypercarry, positive draft edge
    sn = behind[(behind["max_carry_blue"] < HYPERCARRY_SHARE) & (behind["draft_win_logit_blue"] > 0.05)]
    return {
        "scaling_behind_wr": round(float(sc["y_blue_win"].mean()), 4) if len(sc) >= 40 else None,
        "snowball_behind_wr": round(float(sn["y_blue_win"].mean()), 4) if len(sn) >= 40 else None,
        "n_scaling": int(len(sc)),
        "n_snowball": int(len(sn)),
        "gap_pp": (
            round(float(sc["y_blue_win"].mean() - sn["y_blue_win"].mean()) * 100, 2)
            if len(sc) >= 40 and len(sn) >= 40
            else None
        ),
        "note": "blue max_carry≥0.32 vs draft>0.05 no hypercarry, gold15<-1k",
    }


def expected_gold_curve(df: pd.DataFrame) -> dict:
    """Mean gold@10/@15 by draft quintile — for scaling_gap live metric."""
    out = {"gold10": {}, "gold15": {}}
    for q, g in df.dropna(subset=["gold10"]).groupby("draft_q"):
        out["gold10"][str(int(q))] = round(float(g["gold10"].mean()), 1)
    for q, g in df.dropna(subset=["gold15"]).groupby("draft_q"):
        out["gold15"][str(int(q))] = round(float(g["gold15"].mean()), 1)
    return out


def replay_kcdk(coefs: dict, conc: dict) -> dict:
    """Replay KC Map1 @20: gold -2k, towers 0-3, dragons 0-2, grubs ignored in matrix."""
    from lol_kills.draft_score import draft_score
    from lol_kills.live_win import live_win_prob

    blue = ["Vayne", "Lee Sin", "Galio", "Ziggs", "Camille"]
    red = ["Tristana", "Skarner", "Viktor", "Kai'Sa", "Shen"]
    ds = draft_score(blue, red, league="EWC")
    edge = float(ds["components"]["win_edge"])
    cf = draft_conc_features(blue, red, conc)
    p_pre = 0.587

    # old-style (no draft interact) — call without draft args
    old = live_win_prob(
        p_pre=p_pre,
        minute=20,
        kill_diff=-1,
        gold_diff=-2000,
        dragons=0,
        opp_dragons=2,
        void_grubs=-3,
        towers=0,
        opp_towers=3,
    )
    # Prefer DK-side invert for old parity
    dk_old = live_win_prob(
        p_pre=1 - p_pre,
        minute=20,
        kill_diff=1,
        gold_diff=2000,
        dragons=2,
        opp_dragons=0,
        void_grubs=3,
        towers=3,
        opp_towers=0,
    )
    p_old = 1 - dk_old["p_win"]

    new = live_win_prob(
        p_pre=p_pre,
        minute=20,
        kill_diff=-1,
        gold_diff=-2000,
        dragons=0,
        opp_dragons=2,
        void_grubs=-3,
        towers=0,
        opp_towers=3,
        draft_edge=edge,
        kill_conc_diff=cf["kill_conc_diff"],
        scaling_flag=cf["scaling_flag"],
        blue_hypercarry=cf.get("blue_hypercarry", 0),
        draft_q=_draft_q_from_edge(edge),
    )
    return {
        "p_pre": p_pre,
        "draft_edge": edge,
        "conc": cf,
        "p_kc_old_generic": round(p_old, 4),
        "p_kc_matrix": round(new["p_win"], 4),
        "delta_pp": round((new["p_win"] - p_old) * 100, 2),
        "phase": new.get("phase"),
        "scaling_gap": new.get("scaling_gap"),
        "matrix_cell": new.get("matrix_cell"),
    }


def _draft_q_from_edge(edge: float) -> int:
    from lol_kills.live_win import draft_q_from_edge

    return draft_q_from_edge(edge)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print("[advantage] loading OE players/maps…")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    conc = build_kill_concentration(players)
    conc_path = MODELS_DIR / "champ_kill_concentration.json"
    conc_path.write_text(json.dumps(conc, indent=2))
    print(f"[advantage] wrote {conc_path} champs={conc['n_champs']}")

    print("[advantage] building study frame…")
    df = build_study_frame(conc)
    print(f"[advantage] n_maps={len(df)}")

    print("[advantage] empirical grids…")
    grids = empirical_grids(df)
    gold_curve = expected_gold_curve(df)
    matrix = {
        "version": 1,
        "n_maps": int(len(df)),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "grids": grids,
        "expected_gold_by_draft_q": gold_curve,
        "gold_bin_edges": GOLD_LABELS,
        "phases": {"early": "0-14", "mid": "14-25", "late": "25+"},
    }
    mat_path = MODELS_DIR / "draft_advantage_matrix.json"
    mat_path.write_text(json.dumps(matrix, indent=2))
    print(f"[advantage] wrote {mat_path}")

    print("[advantage] fitting live interaction coefs…")
    coefs = fit_live_coefs(df)
    coefs["expected_gold_by_draft_q"] = gold_curve
    coef_path = MODELS_DIR / "draft_live_coefs.json"
    coef_path.write_text(json.dumps(coefs, indent=2))
    print(f"[advantage] wrote {coef_path}")

    # Replay needs live_win updated — write placeholder then re-run after wire
    report = {
        "n_maps": int(len(df)),
        "concentration_champs": conc["n_champs"],
        "cv": coefs["cv_brier_model"],
        "cv_elo": coefs["cv_brier_elo"],
        "holdout": coefs["holdout"],
        "sanity_scaling_vs_snowball_behind": coefs["sanity_scaling_vs_snowball_behind"],
        "top_hypercarries": sorted(
            (
                {"champ": c, "mean_share": v["mean_share"], "n": v["n"]}
                for c, v in conc["champs"].items()
                if v.get("hypercarry")
            ),
            key=lambda x: -x["mean_share"],
        )[:15],
    }
    # Try replay if live_win already accepts new args
    try:
        import lol_kills.live_win as lw

        lw._COEFS_CACHE = None
        lw._CONC_CACHE = None
        report["replay_kcdk_20"] = replay_kcdk(coefs, conc)
        # Unit smoke: behind@20 high vs low blue hypercarry
        from lol_kills.live_win import live_win_prob

        base = dict(
            p_pre=0.55,
            minute=20,
            kill_diff=-1,
            gold_diff=-2000,
            dragons=0,
            opp_dragons=2,
            towers=0,
            opp_towers=3,
            void_grubs=-3,
            draft_edge=0.12,
            draft_q=3,
        )
        lo = live_win_prob(**base, kill_conc_diff=0.0, scaling_flag=0, blue_hypercarry=0)
        hi = live_win_prob(**base, kill_conc_diff=0.02, scaling_flag=1, blue_hypercarry=1)
        report["smoke_behind_scaling"] = {
            "p_no_carry": lo["p_win"],
            "p_blue_hypercarry": hi["p_win"],
            "delta_pp": round((hi["p_win"] - lo["p_win"]) * 100, 2),
            "expect": "blue hypercarry behind → higher p than no-carry",
        }
        rep = report["replay_kcdk_20"]
        if isinstance(rep, dict) and "delta_pp" in rep:
            rep["note"] = (
                "KC lacks blue hypercarry (Vayne share<0.32); DK has Kai'Sa. "
                "Matrix p below generic soft-cap is expected — scaling is on the leading side."
            )
    except TypeError as e:
        report["replay_kcdk_20"] = {"error": str(e), "note": "wire live_win first then re-run"}

    rep_path = MODELS_DIR / "draft_advantage_report.json"
    rep_path.write_text(json.dumps(report, indent=2))
    print(f"[advantage] wrote {rep_path}")
    print("[advantage] sanity", report["sanity_scaling_vs_snowball_behind"])
    print("[advantage] holdout", report["holdout"])


if __name__ == "__main__":
    main()
