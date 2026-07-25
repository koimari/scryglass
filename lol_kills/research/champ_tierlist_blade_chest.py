#!/usr/bin/env python3
"""
Champ side-tierlist with Blade-Chest counterability in the tier score.

Mathematical stack (arxiv-aligned)
----------------------------------
1. Side-specific Elo-controlled pick value on the *score* patch (presence
   logistic + Elo residual blend) — transitive strength.
2. Blade-Chest-inner role matchups (Chen & Joachims WSDM'16 / related NCT
   residual idea in arXiv:2502.03998):

      M(a,b) = blade_a·chest_b − blade_b·chest_a + γ_a − γ_b
      P(a beats b) = σ(M)

   Fit on a slightly wider OE patch window so residuals are identifiable.
3. Counterability from the predicted matchup field (not raw WR heuristics):

      floor = CVaR_α of {P(c beats opp)} over same-role opps
      cb_score ∈ [0,100] from (0.5 − floor) and residual variance

4. Path Blind vs Counter from cb + spike profile; tier cuts on:

      tier_score = elo_pp − λ(path) · (cb/100) · penalty_pp

Run
---
  python3 -m lol_kills.research.champ_tierlist_blade_chest

Writes
------
  data/lol/models/champ_tierlist_16_13_blade_chest.json
  data/lol/models/blade_chest_role_matchups.json
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR
from lol_kills.research.champ_oe_lenses import (
    BOARD_SCOPES,
    SCOPE_ORDER,
    build_lenses_by_region,
    build_lenses_for_scope,
    league_to_region,
    resolve_scope_patch,
)

SCORE_PATCH = 16.13
# User request: last patch only (objectives + regional boards on 16.13).
FIT_PATCHES = (16.13,)
BC_DIM = 3
# Tuned so residuals survive: L2_VEC≈8 collapses blade/chest to 0 (pure BT).
BC_L2_VEC = 0.05
BC_L2_GAMMA = 0.01
MIN_SIDE = 6
MIN_SIDE_ZS = 8
PRIOR_N = 20.0
PENALTY_PP = 10.0
# Only tax counterability *above* this baseline (MOBA RPS ⇒ everyone is somewhat answerable).
CB_BASELINE = 40.0
BLIND_LAM = 1.0
COUNTER_LAM = 0.45
UNKNOWN_CB = 45.0
UNKNOWN_LAM = 0.55
CVAR_ALPHA = 0.20
# Prefer Blind when model floor is healthy even if raw cb is mid.
BLIND_FLOOR_MIN = 44.0  # CVaR predicted WR %
# Gold→DPM impact bonus clipped so it can't swamp Elo/cb.
# OE multi-lens composite (SIDO fight + phase gold + vision + tower) — clipped in oe_pp
IMPACT_CLIP_PP = 4.0
IMPACT_WEIGHT = 1.0

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

OUT_BOARD = MODELS_DIR / "champ_tierlist_16_13_blade_chest.json"
OUT_BC = MODELS_DIR / "blade_chest_role_matchups.json"


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


def _load_maps_players() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    maps = pd.read_parquet(PARQUET_DIR / "maps.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    feat = pd.read_parquet(FEATURES_DIR / "maps.parquet", columns=["game_uid", "elo_diff"])
    maps = maps.merge(feat, on="game_uid", how="left")
    maps["patch_f"] = maps["patch"].astype(float)
    return maps, players, feat


def build_draft_frame(maps: pd.DataFrame, players: pd.DataFrame, patches: tuple[float, ...]) -> pd.DataFrame:
    recent = maps[maps["patch_f"].isin(patches)].copy()
    pl = players[players["game_uid"].isin(set(recent["game_uid"]))].copy()
    pl["champion"] = pl["champion"].map(lambda x: normalize_champ(str(x)) if pd.notna(x) else None)
    pl["pos"] = pl["position"].astype(str).str.lower().map(lambda x: POS_MAP.get(x, x))
    gmap = {gid: g for gid, g in pl.groupby("game_uid")}
    rows: list[dict] = []
    for _, mrow in recent.iterrows():
        g = gmap.get(mrow["game_uid"])
        if g is None or pd.isna(mrow["y_blue_win"]) or pd.isna(mrow["elo_diff"]):
            continue
        e: dict = {
            "game_uid": mrow["game_uid"],
            "y": int(mrow["y_blue_win"]),
            "elo": float(mrow["elo_diff"]),
            "patch": float(mrow["patch_f"]),
            "league": mrow["league"],
        }
        ok = True
        for side, sk in (("Blue", "blue"), ("Red", "red")):
            sg = g[g["side"].astype(str).str.title() == side]
            for role in ROLES:
                rc = sg[sg["pos"] == role]
                if len(rc) != 1:
                    ok = False
                    break
                e[f"{sk}_{role}"] = rc.iloc[0]["champion"]
            if not ok:
                break
        if ok:
            rows.append(e)
    return pd.DataFrame(rows)


def build_draft_frame_full_oe(
    patches: tuple[float, ...],
    *,
    leagues: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """
    One row per map from *full* OE (not majors-filtered maps.parquet).
    Optional league filter (e.g. LEC only, MSI only).
    elo_diff attached when present in feature store; else 0.
    """
    team = pd.read_parquet(PARQUET_DIR / "oe_team_games.parquet")
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    team = team.copy()
    team["patch_f"] = pd.to_numeric(team["patch"], errors="coerce")
    team = team[team["patch_f"].isin(patches)].copy()
    if leagues is not None:
        want = {lg.upper() for lg in leagues}
        team = team[team["league"].astype(str).str.strip().str.upper().isin(want)].copy()
    team["game_uid"] = team["gameid"].astype(str)
    team["side_t"] = team["side"].astype(str).str.strip().str.title()
    team.loc[team["side_t"].str.lower().isin(["blue", "1"]), "side_t"] = "Blue"
    team.loc[team["side_t"].str.lower().isin(["red", "2"]), "side_t"] = "Red"

    blue = team[team["side_t"] == "Blue"][
        ["game_uid", "date", "league", "patch_f", "teamname", "result"]
    ].rename(columns={"teamname": "blue_team", "result": "blue_result"})
    red = team[team["side_t"] == "Red"][["game_uid", "teamname", "result"]].rename(
        columns={"teamname": "red_team", "result": "red_result"}
    )
    maps = blue.merge(red, on="game_uid", how="inner")
    maps["y"] = pd.to_numeric(maps["blue_result"], errors="coerce")
    maps = maps.dropna(subset=["y"]).copy()
    maps["y"] = maps["y"].astype(int)
    maps["region"] = maps["league"].map(league_to_region)
    maps["elo"] = 0.0

    feat_path = FEATURES_DIR / "maps.parquet"
    if feat_path.exists():
        feat = pd.read_parquet(feat_path, columns=["game_uid", "elo_diff"])
        feat["game_uid"] = feat["game_uid"].astype(str)
        maps = maps.merge(feat, on="game_uid", how="left")
        maps["elo"] = pd.to_numeric(maps["elo_diff"], errors="coerce").fillna(0.0)

    pl = players.copy()
    pl["game_uid"] = pl["game_uid"].astype(str)
    pl = pl[pl["game_uid"].isin(set(maps["game_uid"]))].copy()
    pl["champion"] = pl["champion"].map(lambda x: normalize_champ(str(x)) if pd.notna(x) else None)
    pl["pos"] = pl["position"].astype(str).str.lower().map(lambda x: POS_MAP.get(x, x))
    gmap = {gid: g for gid, g in pl.groupby("game_uid")}

    rows: list[dict] = []
    for _, mrow in maps.iterrows():
        g = gmap.get(mrow["game_uid"])
        if g is None:
            continue
        e: dict = {
            "game_uid": mrow["game_uid"],
            "y": int(mrow["y"]),
            "elo": float(mrow["elo"]),
            "patch": float(mrow["patch_f"]),
            "league": mrow["league"],
            "region": mrow["region"],
        }
        ok = True
        for side, sk in (("Blue", "blue"), ("Red", "red")):
            sg = g[g["side"].astype(str).str.title() == side]
            for role in ROLES:
                rc = sg[sg["pos"] == role]
                if len(rc) != 1:
                    ok = False
                    break
                e[f"{sk}_{role}"] = rc.iloc[0]["champion"]
            if not ok:
                break
        if ok:
            rows.append(e)
    return pd.DataFrame(rows)


@dataclass
class BladeChestModel:
    champs: list[str]
    gamma: np.ndarray
    blade: np.ndarray
    chest: np.ndarray
    dim: int
    n_pairs: int
    train_logloss: float

    def idx(self, c: str) -> int | None:
        try:
            return self.champs.index(c)
        except ValueError:
            return None

    def matchup_logit(self, a: str, b: str) -> float | None:
        ia, ib = self.idx(a), self.idx(b)
        if ia is None or ib is None:
            return None
        m = float(
            self.blade[ia] @ self.chest[ib]
            - self.blade[ib] @ self.chest[ia]
            + self.gamma[ia]
            - self.gamma[ib]
        )
        return m

    def p_win(self, a: str, b: str) -> float | None:
        m = self.matchup_logit(a, b)
        if m is None:
            return None
        return float(_sigmoid(m))


def _role_pair_rows(df: pd.DataFrame) -> list[tuple[str, str, int]]:
    """(champ_a, champ_b, a_won) from blue perspective flipped to picker."""
    pairs: list[tuple[str, str, int]] = []
    for _, r in df.iterrows():
        y = int(r["y"])
        for role in ROLES:
            bc, rc = r[f"blue_{role}"], r[f"red_{role}"]
            pairs.append((bc, rc, y))  # blue beat red?
            pairs.append((rc, bc, 1 - y))
    return pairs


def fit_blade_chest(df: pd.DataFrame, dim: int = BC_DIM) -> BladeChestModel:
    pairs = _role_pair_rows(df)
    champs = sorted({a for a, _, _ in pairs} | {b for _, b, _ in pairs})
    idx = {c: i for i, c in enumerate(champs)}
    n_c = len(champs)
    # params: gamma[n], blade[n*d], chest[n*d]
    n_p = n_c + 2 * n_c * dim

    a_idx = np.array([idx[a] for a, _, _ in pairs], dtype=np.int32)
    b_idx = np.array([idx[b] for _, b, _ in pairs], dtype=np.int32)
    y = np.array([w for _, _, w in pairs], dtype=np.float64)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        gamma = theta[:n_c]
        blade = theta[n_c : n_c + n_c * dim].reshape(n_c, dim)
        chest = theta[n_c + n_c * dim :].reshape(n_c, dim)
        return gamma, blade, chest

    def nll(gamma: np.ndarray, blade: np.ndarray, chest: np.ndarray) -> float:
        m = (
            np.sum(blade[a_idx] * chest[b_idx], axis=1)
            - np.sum(blade[b_idx] * chest[a_idx], axis=1)
            + gamma[a_idx]
            - gamma[b_idx]
        )
        p = np.clip(_sigmoid(m), 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    # Warm-start: Bradley-Terry γ only (transitive axis).
    def bt_loss(g: np.ndarray) -> float:
        m = g[a_idx] - g[b_idx]
        p = np.clip(_sigmoid(m), 1e-6, 1 - 1e-6)
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return float(ll + BC_L2_GAMMA * np.mean(g**2))

    bt = minimize(bt_loss, np.zeros(n_c), method="L-BFGS-B", options={"maxiter": 250})
    gamma0 = bt.x

    def loss(theta: np.ndarray) -> float:
        gamma, blade, chest = unpack(theta)
        ll = nll(gamma, blade, chest)
        reg = BC_L2_GAMMA * float(np.mean(gamma**2)) + BC_L2_VEC * float(
            np.mean(blade**2) + np.mean(chest**2)
        )
        return ll + reg

    rng = np.random.default_rng(42)
    theta0 = rng.normal(0, 0.08, size=n_p)
    theta0[:n_c] = gamma0
    res = minimize(
        loss,
        theta0,
        method="L-BFGS-B",
        options={"maxiter": 600, "ftol": 1e-9, "maxfun": 20000},
    )
    gamma, blade, chest = unpack(res.x)
    ll = nll(gamma, blade, chest)
    # residual std diagnostic (cyclic / counter component)
    m_full = (
        np.sum(blade[a_idx] * chest[b_idx], axis=1)
        - np.sum(blade[b_idx] * chest[a_idx], axis=1)
        + gamma[a_idx]
        - gamma[b_idx]
    )
    transit = gamma[a_idx] - gamma[b_idx]
    res_std = float(np.std(m_full - transit))
    print(
        f"[blade_chest] opt success={res.success} nit={res.nit} "
        f"bt_ll={nll(gamma0, np.zeros((n_c, dim)), np.zeros((n_c, dim))):.4f} "
        f"bc_ll={ll:.4f} residual_std={res_std:.3f} ||blade||={np.linalg.norm(blade):.2f}"
    )
    return BladeChestModel(
        champs=champs,
        gamma=gamma,
        blade=blade,
        chest=chest,
        dim=dim,
        n_pairs=len(pairs),
        train_logloss=ll,
    )


def bc_counterability(
    model: BladeChestModel,
    champ: str,
    role: str,
    field_by_role: dict[str, set[str]],
    *,
    min_field: int = 6,
) -> dict | None:
    """Model-based counterability vs same-role meta field (frequent peers only)."""
    opps = sorted(
        o for o in field_by_role.get(role, set()) if o != champ and model.idx(o) is not None
    )
    if model.idx(champ) is None or len(opps) < min_field:
        return None
    ps = []
    residuals = []  # vs transitive-only γ
    ia = model.idx(champ)
    assert ia is not None
    for opp in opps:
        ib = model.idx(opp)
        assert ib is not None
        full = model.matchup_logit(champ, opp)
        assert full is not None
        p = float(_sigmoid(full))
        ps.append(p)
        transit = float(model.gamma[ia] - model.gamma[ib])
        residuals.append(full - transit)
    arr = np.array(ps)
    # CVaR_alpha of win probs (lower tail = answered)
    k = max(1, int(math.ceil(CVAR_ALPHA * len(arr))))
    floor = float(np.sort(arr)[:k].mean())
    mean_p = float(arr.mean())
    p10 = float(np.quantile(arr, 0.10))
    p90 = float(np.quantile(arr, 0.90))
    spread = p90 - p10
    res = np.array(residuals)
    res_neg = float(np.mean(np.clip(-res, 0, None)))
    # Calibrated so typical meta champs land ~15–70, not all 90+.
    # floor 50% → 0; floor 35% → ~1.0 on that component.
    floor_punish = max(0.0, 0.52 - floor)
    score = 100.0 * (
        0.50 * min(floor_punish / 0.22, 1.0)
        + 0.30 * min(res_neg / 0.55, 1.0)
        + 0.20 * min(max(spread - 0.10, 0.0) / 0.30, 1.0)
    )
    order = np.argsort(arr)
    worst = [(opps[i], round(100 * arr[i], 1)) for i in order[:3]]
    best = [(opps[i], round(100 * arr[i], 1)) for i in order[::-1][:3]]
    if score >= 55:
        label = "highly counterable"
    elif score >= 35:
        label = "somewhat counterable"
    elif score < 20:
        label = "hard to answer"
    else:
        label = "moderately counterable"
    return {
        "score": round(float(score), 1),
        "label": label,
        "mean_wr": round(100 * mean_p, 1),
        "cvar_floor": round(100 * floor, 1),
        "p10": round(100 * p10, 1),
        "p90": round(100 * p90, 1),
        "spread_pp": round(100 * spread, 1),
        "residual_neg": round(res_neg, 3),
        "residual_std": round(float(res.std()), 3),
        "gamma": round(float(model.gamma[ia]), 3),
        "n_opps": len(opps),
        "answered_by": worst,
        "holds_vs": best,
        "blind_pp": round(100 * (mean_p - 0.5), 1),
        "counter_pp": round(100 * (p90 - 0.5), 1),
        "gap": round(100 * (p90 - mean_p), 1),
    }


def side_elo_values(df: pd.DataFrame) -> dict[str, dict]:
    """Side-specific Elo-controlled blend for each champ on df (score patch)."""
    y = df["y"].astype(float).values
    elo = df["elo"].astype(float).values
    if float(np.nanstd(elo)) < 1e-6:
        # Full-OE regional frames may lack feature-store Elo — use base-rate residual.
        p_elo = np.full(len(df), float(y.mean()) if len(y) else 0.5)
    else:
        Z = (elo / 400.0).reshape(-1, 1)
        elo_clf = LogisticRegression(C=5.0, max_iter=400).fit(Z, y)
        p_elo = elo_clf.predict_proba(Z)[:, 1]
    resid = y - p_elo

    champs = sorted(
        {
            c
            for role in ROLES
            for side in ("blue", "red")
            for c in df[f"{side}_{role}"]
        }
    )
    idx = {c: i for i, c in enumerate(champs)}
    n, p = len(df), len(champs)
    X = np.zeros((n, 1 + 2 * p))
    X[:, 0] = df["elo"] / 400.0
    counts_b: dict[str, int] = defaultdict(int)
    counts_r: dict[str, int] = defaultdict(int)
    roles: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for i, (_, r) in enumerate(df.iterrows()):
        for role in ROLES:
            bc, rc = r[f"blue_{role}"], r[f"red_{role}"]
            X[i, 1 + idx[bc]] += 1
            X[i, 1 + p + idx[rc]] += 1
            counts_b[bc] += 1
            counts_r[rc] += 1
            roles[bc][role] += 1
            roles[rc][role] += 1

    clf = LogisticRegression(C=0.35, max_iter=700, solver="lbfgs").fit(X, y)
    coef = clf.coef_[0]

    def shrink(beta: float, nc: int) -> float:
        w = nc / (nc + PRIOR_N)
        if nc < MIN_SIDE:
            w *= nc / max(MIN_SIDE, 1)
        return beta * w

    res_b: dict[str, list[float]] = defaultdict(list)
    res_r: dict[str, list[float]] = defaultdict(list)
    raw_b: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    raw_r: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for i, (_, r) in enumerate(df.iterrows()):
        for role in ROLES:
            bc, rc = r[f"blue_{role}"], r[f"red_{role}"]
            res_b[bc].append(resid[i])
            raw_b[bc][1] += 1
            raw_b[bc][0] += int(r["y"] == 1)
            res_r[rc].append(-resid[i])
            raw_r[rc][1] += 1
            raw_r[rc][0] += int(r["y"] == 0)

    def resid_pp(lst: list[float]) -> float:
        if not lst:
            return 0.0
        m = float(np.mean(lst))
        nc = len(lst)
        return 100.0 * m * (nc / (nc + PRIOR_N))

    out: dict[str, dict] = {}
    for c in champs:
        nb, nr = counts_b[c], counts_r[c]
        b_pp = 100.0 * shrink(float(coef[1 + idx[c]]), nb) * 0.25
        r_pp = 100.0 * shrink(float(-coef[1 + p + idx[c]]), nr) * 0.25
        b_blend = 0.55 * resid_pp(res_b[c]) + 0.45 * b_pp
        r_blend = 0.55 * resid_pp(res_r[c]) + 0.45 * r_pp
        rb, rn = raw_b[c]
        rr, rrn = raw_r[c]
        prim = max(roles[c].items(), key=lambda x: x[1])[0] if roles[c] else "?"
        out[c] = {
            "role": prim,
            "roles": dict(roles[c]),
            "blue": {
                "elo_pp": b_blend,
                "n": nb,
                "raw_wr": (100 * rb / rn) if rn else None,
            },
            "red": {
                "elo_pp": r_blend,
                "n": nr,
                "raw_wr": (100 * rr / rrn) if rrn else None,
            },
        }
    return out


def _tier_score(elo_pp: float, cb_score: float, lam: float, impact_pp: float = 0.0) -> float:
    """Tax excess counterability; add gold→DPM impact (clipped)."""
    excess = max(0.0, cb_score - CB_BASELINE)
    impact = float(np.clip(impact_pp, -IMPACT_CLIP_PP, IMPACT_CLIP_PP)) * IMPACT_WEIGHT
    return elo_pp - lam * (excess / 100.0) * PENALTY_PP + impact


def resolve_path(
    elo_pp: float, cb: dict | None, impact_pp: float = 0.0
) -> tuple[str, float, float, float, str]:
    if cb is None:
        lam = UNKNOWN_LAM
        cb_score = UNKNOWN_CB
        ts = _tier_score(elo_pp, cb_score, lam, impact_pp)
        return "Blind", cb_score, lam, ts, "unknown counterability — mild safety tax"
    cb_score = float(cb["score"])
    floor = float(cb["cvar_floor"])
    reason = (
        f"{cb['label']} (cb {cb_score:.0f}/100; CVaR floor {floor:.0f}%; "
        f"γ={cb['gamma']:+.2f})"
    )
    # Healthy matchup floor → Blind path even in a RPS meta.
    if floor >= BLIND_FLOOR_MIN and cb_score < 70:
        lam = BLIND_LAM
        ts = _tier_score(elo_pp, cb_score, lam, impact_pp)
        return "Blind", cb_score, lam, ts, reason + " → Blind (healthy floor)"
    if cb_score >= 55 or (cb["gap"] >= 10 and cb["counter_pp"] >= 8 and cb_score >= 40):
        lam = COUNTER_LAM
        ts = _tier_score(elo_pp, cb_score, lam, impact_pp)
        return "Counter", cb_score, lam, ts, reason + " → Counter (excess answerability)"
    if cb_score <= CB_BASELINE + 5:
        lam = BLIND_LAM
        ts = _tier_score(elo_pp, cb_score, lam, impact_pp)
        return "Blind", cb_score, lam, ts, reason
    # Mid band: pick higher adjusted score
    tb = _tier_score(elo_pp, cb_score, BLIND_LAM, impact_pp)
    tc = _tier_score(elo_pp, cb_score, COUNTER_LAM, impact_pp)
    if tb >= tc:
        return "Blind", cb_score, BLIND_LAM, tb, reason + " → Blind path"
    return "Counter", cb_score, COUNTER_LAM, tc, reason + " → Counter path"


def _why(
    champ: str,
    side: str,
    info: dict,
    arch: str,
    reason: str,
    elo_pp: float,
    tscore: float,
    cb_score: float,
    lam: float,
    cb: dict | None,
    impact_pp: float = 0.0,
    impact: dict | None = None,
) -> str:
    excess = max(0.0, cb_score - CB_BASELINE)
    cb_tax = lam * (excess / 100.0) * PENALTY_PP
    bits = [
        f"Elo-ctrl {elo_pp:+.1f}pp → tier_score {tscore:+.1f}pp "
        f"(cb tax −{cb_tax:.1f}; impact {impact_pp:+.2f}pp clipped±{IMPACT_CLIP_PP})",
        f"{arch}: {reason}",
    ]
    if impact:
        bits.append(
            f"OE lenses {impact_pp:+.2f}pp "
            f"(fight {impact.get('fight_pp')}; lane {impact.get('lane_pp')}; "
            f"vision {impact.get('vision_pp')}; tower {impact.get('tower_pp')}; "
            f"obj {impact.get('obj_pp')})"
        )
    if cb:
        bits.append(
            f"Blade-Chest counterable {cb['score']:.0f}/100; "
            f"CVaR floor {cb['cvar_floor']:.0f}%; γ={cb['gamma']:+.2f}"
        )
        bits.append(
            "answered by "
            + ", ".join(f"{o} ({w:.0f}%)" for o, w in cb["answered_by"][:2])
        )
        if arch == "Counter":
            bits.append(
                "answers " + ", ".join(f"{o} ({w:.0f}%)" for o, w in cb["holds_vs"][:2])
            )
        else:
            bits.append(
                "holds vs " + ", ".join(f"{o} ({w:.0f}%)" for o, w in cb["holds_vs"][:2])
            )
    else:
        bits.append(f"cb unknown → prior {UNKNOWN_CB:.0f}")
    other = "red" if side == "blue" else "blue"
    o = info[other]
    if o["n"] >= MIN_SIDE and abs(elo_pp - o["elo_pp"]) >= 3.5:
        bits.append(f"side-split: {other} Elo {o['elo_pp']:+.1f}pp n={o['n']}")
    s = info[side]
    if s["n"] < 14:
        bits.append("thin sample — provisional")
    if len(info["roles"]) > 1:
        bits.append(
            "flex "
            + ",".join(f"{r}:{n}" for r, n in sorted(info["roles"].items(), key=lambda x: -x[1]))
        )
    else:
        bits.append(f"mostly {info['role']}")
    raw = s["raw_wr"]
    bits.append(f"n={s['n']}" + (f", raw {raw:.0f}%" if raw is not None else ""))
    return "; ".join(bits)


def build_board(
    side: str,
    champ_info: dict[str, dict],
    cb_by_champ: dict[str, dict | None],
    impact_by_champ: dict[str, dict] | None = None,
) -> tuple[dict, dict]:
    impact_by_champ = impact_by_champ or {}
    scored = []
    for champ, info in champ_info.items():
        s = info[side]
        if s["n"] < MIN_SIDE:
            continue
        cb = cb_by_champ.get(champ)
        impact = impact_by_champ.get(champ)
        impact_pp = float(impact["impact_pp"]) if impact else 0.0
        arch, cb_score, lam, tscore, reason = resolve_path(s["elo_pp"], cb, impact_pp)
        scored.append(
            {
                "champ": champ,
                "info": info,
                "arch": arch,
                "cb_score": cb_score,
                "lam": lam,
                "tscore": tscore,
                "reason": reason,
                "elo_pp": s["elo_pp"],
                "cb": cb,
                "impact": impact,
                "impact_pp": impact_pp,
            }
        )
    scored.sort(key=lambda x: -x["tscore"])
    if not scored:
        empty = {k: [] for k in ["Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D", "E", "F"]}
        return empty, {k: 0.0 for k in ["Z", "S", "A", "B", "C", "D", "E"]}
    tss = np.array([x["tscore"] for x in scored])
    cuts = {
        "Z": max(float(np.quantile(tss, 0.90)), 3.2),
        "S": max(float(np.quantile(tss, 0.75)), 1.6),
        "A": float(np.quantile(tss, 0.60)),
        "B": float(np.quantile(tss, 0.45)),
        "C": float(np.quantile(tss, 0.30)),
        "D": float(np.quantile(tss, 0.18)),
        "E": float(np.quantile(tss, 0.08)),
    }
    board = {k: [] for k in ["Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D", "E", "F"]}
    for row in scored:
        info = row["info"]
        s = info[side]
        ts = row["tscore"]
        arch = row["arch"]
        reason = row["reason"]
        cb = row["cb"]
        impact = row["impact"]
        impact_pp = row["impact_pp"]
        if ts >= cuts["Z"] and s["n"] >= MIN_SIDE_ZS:
            base = "Z"
        elif ts >= cuts["Z"]:
            base = "S"
        elif ts >= cuts["S"]:
            base = "S"
        elif ts >= cuts["A"]:
            base = "A"
        elif ts >= cuts["B"]:
            base = "B"
        elif ts >= cuts["C"]:
            base = "C"
        elif ts >= cuts["D"]:
            base = "D"
        elif ts >= cuts["E"]:
            base = "E"
        else:
            base = "F"
        if base in ("Z", "S") and arch == "Blind" and cb is not None and cb["score"] >= 70:
            arch = "Counter"
            reason = reason + f"; forced Counter bucket (cb {cb['score']:.0f})"
        tier = f"{base} {arch}" if base in ("Z", "S") else base
        cb_out = (
            None
            if cb is None
            else {
                "score": cb["score"],
                "label": cb["label"],
                "cvar_floor": cb["cvar_floor"],
                "gamma": cb["gamma"],
                "answered_by": cb["answered_by"],
                "holds_vs": cb["holds_vs"],
            }
        )
        impact_out = None
        if impact:
            impact_out = {
                "impact_pp": impact.get("impact_pp"),
                "oe_pp": impact.get("oe_pp"),
                "fight_pp": impact.get("fight_pp"),
                "lane_pp": impact.get("lane_pp"),
                "vision_pp": impact.get("vision_pp"),
                "tower_pp": impact.get("tower_pp"),
                "obj_pp": impact.get("obj_pp"),
                "n": impact.get("n"),
            }
        entry = {
            "champ": row["champ"],
            "role": info["role"],
            "elo_pp": round(row["elo_pp"], 2),
            "impact_pp": round(impact_pp, 3),
            "tier_score": round(ts, 2),
            "counterability_tax": round(
                row["lam"] * max(0.0, row["cb_score"] - CB_BASELINE) / 100.0 * PENALTY_PP,
                2,
            ),
            "n": s["n"],
            "raw_wr": None if s["raw_wr"] is None else round(s["raw_wr"], 1),
            "archetype": arch if base in ("Z", "S") else None,
            "counterable": cb_out,
            "impact": impact_out,
            "lambda": row["lam"],
            "why": _why(
                row["champ"],
                side,
                info,
                arch,
                reason,
                row["elo_pp"],
                ts,
                row["cb_score"],
                row["lam"],
                cb,
                impact_pp=impact_pp,
                impact=impact,
            ),
        }
        board[tier].append(entry)
    for k in board:
        board[k].sort(key=lambda e: -e["tier_score"])
    return board, {k: round(v, 2) for k, v in cuts.items()}


def _field_by_role(fit_df: pd.DataFrame, min_role_games: int) -> dict[str, set[str]]:
    field_by_role: dict[str, set[str]] = defaultdict(set)
    role_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, r in fit_df.iterrows():
        for role in ROLES:
            for side in ("blue", "red"):
                c = r[f"{side}_{role}"]
                role_counts[c][role] += 1
    for c, rc in role_counts.items():
        for role, n in rc.items():
            if n >= min_role_games:
                field_by_role[role].add(c)
    return field_by_role


def _lookup_from_scored(
    side: str,
    champ_info: dict[str, dict],
    cb_by_champ: dict[str, dict | None],
    impact_by_champ: dict[str, dict] | None = None,
    *,
    min_n: int = 3,
) -> dict[str, dict]:
    """
    Dense champ→metrics lookup for betting (includes provisional n < board floor).
    """
    impact_by_champ = impact_by_champ or {}
    out: dict[str, dict] = {}
    for champ, info in champ_info.items():
        s = info[side]
        n = int(s.get("n") or 0)
        if n < min_n:
            continue
        cb = cb_by_champ.get(champ)
        impact = impact_by_champ.get(champ)
        impact_pp = float(impact["impact_pp"]) if impact else 0.0
        arch, cb_score, lam, tscore, reason = resolve_path(s["elo_pp"], cb, impact_pp)
        out[champ] = {
            "champ": champ,
            "role": info.get("role"),
            "elo_pp": round(float(s["elo_pp"]), 2),
            "impact_pp": round(impact_pp, 3),
            "tier_score": round(float(tscore), 2),
            "counterability_tax": round(
                lam * max(0.0, cb_score - CB_BASELINE) / 100.0 * PENALTY_PP, 2
            ),
            "n": n,
            "raw_wr": None if s.get("raw_wr") is None else round(float(s["raw_wr"]), 1),
            "archetype": arch,
            "lambda": lam,
            "provisional": n < MIN_SIDE,
            "counterable": None
            if cb is None
            else {
                "score": cb["score"],
                "label": cb["label"],
                "cvar_floor": cb["cvar_floor"],
                "gamma": cb["gamma"],
                "answered_by": cb.get("answered_by"),
                "holds_vs": cb.get("holds_vs"),
            },
            "why": reason,
        }
    return out


def _run_side_boards(
    score_df: pd.DataFrame,
    bc: BladeChestModel,
    field_by_role: dict[str, set[str]],
    impact_by_champ: dict[str, dict],
) -> dict:
    champ_info = side_elo_values(score_df)
    cb_by_champ: dict[str, dict | None] = {}
    for c, info in champ_info.items():
        cb_by_champ[c] = bc_counterability(bc, c, info["role"], field_by_role)
    blue_board, blue_cuts = build_board("blue", champ_info, cb_by_champ, impact_by_champ)
    red_board, red_cuts = build_board("red", champ_info, cb_by_champ, impact_by_champ)
    return {
        "n_games": int(len(score_df)),
        "leagues": sorted(score_df["league"].astype(str).unique().tolist()),
        "blue": {
            "cuts": blue_cuts,
            "board": blue_board,
            "lookup": _lookup_from_scored(
                "blue", champ_info, cb_by_champ, impact_by_champ, min_n=3
            ),
        },
        "red": {
            "cuts": red_cuts,
            "board": red_board,
            "lookup": _lookup_from_scored(
                "red", champ_info, cb_by_champ, impact_by_champ, min_n=3
            ),
        },
    }


def _compact(board: dict) -> dict:
    return {t: [e["champ"] for e in lst] for t, lst in board.items()}


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        "[blade_chest] scopes: europe=LEC only, americas=LCS only, "
        "intl=MSI and EWC separated"
    )

    print("[blade_chest] OE lenses by scope…")
    lens_bundle = build_lenses_by_region(patches=FIT_PATCHES, min_n=5)
    (MODELS_DIR / "champ_oe_lenses.json").write_text(json.dumps(lens_bundle, indent=2))

    by_scope: dict[str, dict] = {}
    for scope in BOARD_SCOPES:
        key = scope["key"]
        patch = resolve_scope_patch(scope)
        if patch is None:
            print(f"[blade_chest] skip {key}: no OE data")
            by_scope[key] = {
                "label": scope["label"],
                "leagues": list(scope["leagues"]),
                "patch": None,
                "n_games": 0,
                "note": "no OE games",
            }
            continue
        df = build_draft_frame_full_oe((patch,), leagues=tuple(scope["leagues"]))
        if len(df) < 12:
            print(f"[blade_chest] skip {key}: only {len(df)} maps on patch {patch}")
            by_scope[key] = {
                "label": scope["label"],
                "leagues": list(scope["leagues"]),
                "patch": patch,
                "n_games": int(len(df)),
                "note": "insufficient maps",
            }
            continue

        print(f"[blade_chest] {key}={scope['label']} patch={patch} games={len(df)}")
        bc = fit_blade_chest(df, dim=min(BC_DIM, 2 if len(df) < 80 else BC_DIM))
        field = _field_by_role(df, min_role_games=6 if len(df) < 80 else 10)
        impact = (lens_bundle["by_region"].get(key) or {}).get("champs") or {}
        # If lenses empty, rebuild for this scope
        if not impact:
            art = build_lenses_for_scope(scope, min_n=6)
            impact = art.get("champs") or {}
            lens_bundle["by_region"][key] = art

        # Small domestic samples: keep more champs on the board
        global MIN_SIDE, MIN_SIDE_ZS
        old_min, old_zs = MIN_SIDE, MIN_SIDE_ZS
        if len(df) < 50:
            MIN_SIDE, MIN_SIDE_ZS = 4, 5
        try:
            block = _run_side_boards(df, bc, field, impact)
        finally:
            MIN_SIDE, MIN_SIDE_ZS = old_min, old_zs
        block["label"] = scope["label"]
        block["patch"] = patch
        block["prefer_patch"] = scope.get("prefer_patch")
        block["patch_note"] = (
            f"16.13 empty for {scope['label']} — using latest OE patch {patch}"
            if patch != float(scope.get("prefer_patch") or SCORE_PATCH)
            else f"patch {patch}"
        )
        block["oe"] = {
            "n_champs": (lens_bundle["by_region"].get(key) or {}).get("n_champs"),
            "n_rows": (lens_bundle["by_region"].get(key) or {}).get("n_score_rows"),
            "weights": (lens_bundle["by_region"].get(key) or {}).get("weights"),
        }
        by_scope[key] = block

    def print_board(key: str, block: dict) -> None:
        if not block.get("blue"):
            print(f"\n=== {key.upper()} === skipped ({block.get('note')})")
            return
        print(
            f"\n=== {block.get('label', key).upper()} BLUE === "
            f"patch={block.get('patch')} n={block['n_games']} "
            f"[{block.get('patch_note')}]"
        )
        for t, names in _compact(block["blue"]["board"]).items():
            print(f"  {t}: {', '.join(names) if names else '—'}")
        print(f"=== {block.get('label', key).upper()} RED ===")
        for t, names in _compact(block["red"]["board"]).items():
            print(f"  {t}: {', '.join(names) if names else '—'}")

    for key in SCOPE_ORDER:
        if key in by_scope:
            print_board(key, by_scope[key])

    # BC dump from MSI (largest intl on 16.13) if present
    msi_df = build_draft_frame_full_oe((SCORE_PATCH,), leagues=("MSI",))
    if len(msi_df) >= 20:
        bc_ref = fit_blade_chest(msi_df, dim=BC_DIM)
        bc_dump = {
            "model": "blade-chest-inner",
            "ref_scope": "msi",
            "fit_patches": [SCORE_PATCH],
            "dim": bc_ref.dim,
            "n_champs": len(bc_ref.champs),
            "n_pairs": bc_ref.n_pairs,
            "train_logloss": bc_ref.train_logloss,
        }
        OUT_BC.write_text(json.dumps(bc_dump, indent=2))

    out = {
        "title": "Side tierlist — LEC / LCS / MSI / EWC (separated)",
        "prefer_patch": SCORE_PATCH,
        "scope_rules": {
            "europe": "LEC only",
            "americas": "LCS only",
            "intl": "MSI and EWC as separate boards",
        },
        "formula": {
            "tier_score": "elo_pp - λ·cb_tax + clip(oe_pp)",
            "oe_pp": (
                "0.32 fight + 0.24 lane + 0.12 vision + 0.12 tower + 0.20 objectives"
            ),
        },
        "by_scope": by_scope,
        # aliases
        "by_region": by_scope,
        "msi": by_scope.get("msi"),
        "ewc": by_scope.get("ewc"),
        "lec": by_scope.get("lec"),
        "lcs": by_scope.get("lcs"),
        "blue": (by_scope.get("msi") or {}).get("blue"),
        "red": (by_scope.get("msi") or {}).get("red"),
    }
    OUT_BOARD.write_text(json.dumps(out, indent=2))
    alias = MODELS_DIR / "champ_tierlist_16_13_side.json"
    alias.write_text(json.dumps(out, indent=2))
    regional_path = MODELS_DIR / "champ_tierlist_16_13_by_region.json"
    regional_path.write_text(json.dumps(out, indent=2))
    print(f"\n[blade_chest] wrote {OUT_BOARD}")
    print(f"[blade_chest] wrote {regional_path}")


if __name__ == "__main__":
    main()
