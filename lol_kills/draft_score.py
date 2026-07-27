#!/usr/bin/env python3
"""
Draft Score v2 (0–100): side-aware win + pace + role pairwise + hierarchical shrinkage.

One stack input only — never the sole decision.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR
from lol_kills.features.build import _load_champ_betas

ROOT = Path(__file__).resolve().parents[1]
DRAFT_MODEL = ROOT / "data" / "lol" / "draft_model.json"
OUT = MODELS_DIR / "draft_score_meta.json"
PAIR_PATH = MODELS_DIR / "draft_role_pairs.json"
CAL_PATH = MODELS_DIR / "draft_wr_calibration.json"
COMPOSITION_MODEL_PATH = MODELS_DIR / "draft_composition.json"

# Fallback when study artifact missing (legacy Draft Score v2)
DEFAULT_TEMP = 1.4

LEAGUE_TIER = {
    "LCK": "tier1",
    "LPL": "tier1",
    "LEC": "west",
    "LCS": "west",
    "CBLOL": "americas",
    "AMERICAS": "americas",
    "PCS": "asia_reg",
    "VCS": "asia_reg",
    "LJL": "asia_reg",
    "LCP": "asia_reg",
    "TCL": "asia_reg",
    "MSI": "intl",
    "EWC": "intl",
    "FST": "intl",
    "Worlds": "intl",
}


def _load_calibration() -> dict:
    if not CAL_PATH.exists():
        return {}
    try:
        return json.loads(CAL_PATH.read_text())
    except Exception:
        return {}


def draft_temperature(league: str | None = None, cal: dict | None = None) -> dict:
    """
    League-aware map: p ≈ sigmoid(temp * win_edge) after confidence shrink.
    Prefer league table, else tier, else global joint logistic coef.
    """
    cal = cal if cal is not None else _load_calibration()
    lg = (league or "").upper().strip()
    by_lg = cal.get("by_league") or {}
    by_tier = cal.get("by_tier") or {}
    joint = cal.get("joint_logistic_global") or {}
    residual = cal.get("residual_ridge") or {}

    if lg in by_lg:
        row = by_lg[lg]
        return {
            "source": f"league:{lg}",
            "temp": float(row["coef_draft"]),
            "coef_elo": float(row.get("coef_elo") or 0),
            "intercept": float(row.get("intercept") or 0),
            "dp_per_logit": float(residual.get("coef_dp_per_logit") or 0.23),
        }
    tier = LEAGUE_TIER.get(lg)
    if tier and tier in by_tier:
        row = by_tier[tier]
        return {
            "source": f"tier:{tier}",
            "temp": float(row["coef_draft"]),
            "coef_elo": float(row.get("coef_elo") or 0),
            "intercept": float(row.get("intercept") or 0),
            "dp_per_logit": float(residual.get("coef_dp_per_logit") or 0.23),
        }
    if joint.get("coef_draft") is not None:
        return {
            "source": "global_joint",
            "temp": float(joint["coef_draft"]),
            "coef_elo": float(joint.get("coef_elo") or 0),
            "intercept": float(joint.get("intercept") or 0),
            "dp_per_logit": float(residual.get("coef_dp_per_logit") or 0.23),
        }
    return {
        "source": "legacy_1.4",
        "temp": DEFAULT_TEMP,
        "coef_elo": 0.0,
        "intercept": 0.0,
        "dp_per_logit": 0.23,
    }


def sigmoid(x: float) -> float:
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _champ_counts() -> dict[str, int]:
    if not DRAFT_MODEL.exists():
        return {}
    dm = json.loads(DRAFT_MODEL.read_text())
    return {k: int(v) for k, v in (dm.get("champ_game_counts") or {}).items()}


def fit_role_pairs(min_n: int = 40) -> dict:
    """Learn role-matchup win edges from OE/LP players when both sides have same role."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    maps_path = PARQUET_DIR / "maps.parquet"
    players_path = PARQUET_DIR / "players.parquet"
    if not maps_path.exists() or not players_path.exists():
        PAIR_PATH.write_text("{}")
        return {}

    maps = pd.read_parquet(maps_path, columns=["game_uid", "y_blue_win"])
    players = pd.read_parquet(players_path)
    if "champion" not in players.columns:
        PAIR_PATH.write_text("{}")
        return {}

    gcol = "game_uid" if "game_uid" in players.columns else "gameid"
    players = players.copy()
    players["_gid"] = players[gcol].astype(str)
    role_col = "position" if "position" in players.columns else "role"
    if role_col not in players.columns:
        PAIR_PATH.write_text("{}")
        return {}

    # normalize roles
    def norm_role(r: str) -> str:
        r = str(r or "").lower()
        for a, b in [("jng", "jungle"), ("jungler", "jungle"), ("bot", "adc"), ("bottom", "adc"), ("support", "sup"), ("utility", "sup"), ("mid", "mid"), ("top", "top")]:
            if r.startswith(a) or r == a:
                return {"jungle": "jng", "adc": "bot", "sup": "sup", "mid": "mid", "top": "top"}.get(b, b)
        return r[:3]

    wins: dict[str, list[float]] = defaultdict(list)
    win_map = maps.dropna(subset=["y_blue_win"]).set_index(maps["game_uid"].astype(str))["y_blue_win"].to_dict()

    by_g = players.groupby("_gid")
    for gid, g in by_g:
        y = win_map.get(str(gid))
        if y is None or (isinstance(y, float) and math.isnan(y)):
            continue
        blue = g[g["side"].astype(str).str.title() == "Blue"]
        red = g[g["side"].astype(str).str.title() == "Red"]
        for role in ("top", "jng", "mid", "bot", "sup"):
            bc = blue[blue[role_col].map(norm_role) == role]["champion"]
            rc = red[red[role_col].map(norm_role) == role]["champion"]
            if bc.empty or rc.empty:
                continue
            bchamp = normalize_champ(str(bc.iloc[0]))
            rchamp = normalize_champ(str(rc.iloc[0]))
            key = f"{role}|{bchamp}|{rchamp}"
            wins[key].append(float(y))

    pairs = {}
    prior = 0.5
    prior_n = 20.0
    for k, ys in wins.items():
        if len(ys) < min_n:
            continue
        # hierarchical shrink to 0.5
        wr = (sum(ys) + prior_n * prior) / (len(ys) + prior_n)
        pairs[k] = {
            "n": len(ys),
            "wr": wr,
            "logit": math.log(max(wr, 1e-3) / max(1 - wr, 1e-3)),
        }
    PAIR_PATH.write_text(json.dumps(pairs, indent=2))
    print(f"[draft_score] role pairs n={len(pairs)}")
    return pairs


def _posterior_weight(count: int, prior_n: float = 25.0) -> float:
    """Hierarchical shrinkage weight toward 0 for rare champs."""
    return count / (count + prior_n)


def score_side(
    champs: list[str],
    *,
    kill_beta: dict[str, float],
    win_delta: dict[str, float],
    counts: dict[str, int],
    side_prior: float = 0.0,
) -> dict:
    champs = [normalize_champ(c) for c in champs]
    win_logit = side_prior
    pace = 0.0
    known = 0
    conf_parts = []
    for c in champs:
        cnt = counts.get(c, 0)
        w = _posterior_weight(cnt)
        conf_parts.append(w)
        if c in win_delta or c in kill_beta:
            known += 1
        win_logit += w * win_delta.get(c, 0.0)
        pace += w * kill_beta.get(c, 0.0)
    n = max(len(champs), 1)
    known_frac = known / n
    conf = float(np.clip(0.5 * known_frac + 0.5 * (sum(conf_parts) / n), 0.05, 0.98))
    # posterior width proxy
    width = float(np.mean([1.0 - _posterior_weight(counts.get(c, 0)) for c in champs])) if champs else 1.0
    return {
        "champs": champs,
        "win_logit": win_logit,
        "pace_shift": pace,
        "known_frac": known_frac,
        "confidence": conf,
        "posterior_width": width,
    }


def draft_score(
    blue: list[str],
    red: list[str],
    *,
    blue_roles: list[str] | None = None,
    red_roles: list[str] | None = None,
    blue_side_bonus: float = 0.03,
    league: str | None = None,
    elo_diff: float | None = None,
    team_elo_diff: float | None = None,
    player_elo_diff: float | None = None,
    strength_source: str | None = None,
) -> dict:
    """
    Draft Score with league-calibrated WR mapping when
    data/lol/models/draft_wr_calibration.json exists.

    Elo gaps are separate inputs to the contextualized score; they never
    change the pure draft score axis.
    """
    if len(blue) == 5 and len(red) == 5 and COMPOSITION_MODEL_PATH.exists():
        from lol_kills.composition_model import predict_composition

        model = json.loads(COMPOSITION_MODEL_PATH.read_text())
        return predict_composition(
            model,
            blue,
            red,
            blue_roles=blue_roles,
            red_roles=red_roles,
            league=league,
            elo_diff=elo_diff,
            team_elo_diff=team_elo_diff,
            player_elo_diff=player_elo_diff,
            strength_source=strength_source,
        )

    kill_beta, win_delta, _mu = _load_champ_betas()
    counts = _champ_counts()
    pairs = {}
    if PAIR_PATH.exists():
        pairs = json.loads(PAIR_PATH.read_text() or "{}")
    cal = _load_calibration()
    scale = draft_temperature(league, cal)
    effective_team_elo_diff = team_elo_diff if team_elo_diff is not None else elo_diff

    b = score_side(blue, kill_beta=kill_beta, win_delta=win_delta, counts=counts, side_prior=blue_side_bonus)
    r = score_side(red, kill_beta=kill_beta, win_delta=win_delta, counts=counts, side_prior=0.0)

    pair_logit = 0.0
    roles = blue_roles or ["top", "jng", "mid", "bot", "sup"]
    if len(b["champs"]) == 5 and len(r["champs"]) == 5:
        for i, role in enumerate(roles[:5]):
            key = f"{role}|{b['champs'][i]}|{r['champs'][i]}"
            if key in pairs:
                pair_logit += 0.35 * float(pairs[key]["logit"])

    win_edge = b["win_logit"] - r["win_logit"] + pair_logit
    temp = float(scale["temp"])
    # Pure draft → WR (no Elo): sigmoid(temp * edge), then confidence shrink
    p_blue_raw = sigmoid(win_edge * temp)
    conf = float(min(b["confidence"], r["confidence"]))
    width = 0.5 * (b["posterior_width"] + r["posterior_width"])
    conf = float(np.clip(conf * (1.0 - 0.5 * width), 0.05, 0.98))
    p_shrunk = 0.5 + (p_blue_raw - 0.5) * conf
    score_blue = 100.0 * p_shrunk
    score_red = 100.0 - score_blue

    # Approximate ΔWR vs coin-flip from residual ridge (Elo-controlled bump)
    wr_bump_pp = 100.0 * float(scale["dp_per_logit"]) * win_edge * conf

    p_with_strength = None
    if effective_team_elo_diff is not None and scale["source"] != "legacy_1.4":
        p_with_strength = sigmoid(
            scale["intercept"]
            + scale["coef_elo"] * (float(effective_team_elo_diff) / 400.0)
            + temp * win_edge
        )
        # apply same confidence shrink toward 50 on the draft component only is hard;
        # shrink full p slightly toward Elo-only if desired — keep raw joint for transparency

    legacy_p = sigmoid(win_edge * DEFAULT_TEMP)
    legacy_shrunk = 0.5 + (legacy_p - 0.5) * conf

    return {
        "draft_score_blue": round(score_blue, 2),
        "draft_score_red": round(score_red, 2),
        "draft_edge": round(score_blue - score_red, 2),
        "confidence": round(conf, 3),
        "p_blue_draft": round(p_shrunk, 4),
        "wr_bump_pp": round(wr_bump_pp, 2),
        "posterior_width": round(width, 3),
        "calibration": {
            "league": league,
            "source": scale["source"],
            "temperature": round(temp, 4),
            "legacy_temp": DEFAULT_TEMP,
            "p_blue_legacy_1_4": round(legacy_shrunk, 4),
            "p_blue_with_strength": round(p_with_strength, 4) if p_with_strength is not None else None,
            "dp_per_logit": round(float(scale["dp_per_logit"]), 4),
        },
        "components": {
            "win_logit_blue": round(b["win_logit"], 4),
            "win_logit_red": round(r["win_logit"], 4),
            "pair_logit": round(pair_logit, 4),
            "win_edge": round(win_edge, 4),
            "pace_shift_blue": round(b["pace_shift"], 3),
            "pace_shift_red": round(r["pace_shift"], 3),
            "pace_total_shift": round(b["pace_shift"] + r["pace_shift"], 3),
            "known_frac_blue": round(b["known_frac"], 3),
            "known_frac_red": round(r["known_frac"], 3),
        },
        "blue": b["champs"],
        "red": r["champs"],
        "note": (
            "Draft Score v3: league-calibrated WR temperature from OE study; "
            "still one stack input — not a standalone bet. "
            "wr_bump_pp ≈ Elo-controlled ΔP from residual ridge × edge × conf."
        ),
    }


def fit_draft_score_scaler() -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    fit_role_pairs()
    meta = {"version": 2, "shrinkage": "hierarchical + confidence", "blue_side_bonus": 0.03}
    path = FEATURES_DIR / "maps.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        if "draft_win_logit_blue" in df.columns and "y_blue_win" in df.columns:
            sub = df.dropna(subset=["draft_win_logit_blue", "y_blue_win"])
            if len(sub) > 80:
                qs = pd.qcut(sub["draft_win_logit_blue"], q=8, duplicates="drop")
                cal = sub.groupby(qs, observed=True)["y_blue_win"].agg(["mean", "count"]).reset_index()
                meta["empirical_bins"] = [
                    {"bin": str(row["draft_win_logit_blue"]), "wr": float(row["mean"]), "n": int(row["count"])}
                    for _, row in cal.iterrows()
                ]
    OUT.write_text(json.dumps(meta, indent=2))
    print(f"[draft_score] wrote {OUT}")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blue", type=str)
    ap.add_argument("--red", type=str)
    ap.add_argument("--fit", action="store_true")
    args = ap.parse_args()
    if args.fit:
        fit_draft_score_scaler()
    if args.blue and args.red:
        blue = [c.strip() for c in args.blue.split(",") if c.strip()]
        red = [c.strip() for c in args.red.split(",") if c.strip()]
        print(json.dumps(draft_score(blue, red), indent=2))


if __name__ == "__main__":
    main()
