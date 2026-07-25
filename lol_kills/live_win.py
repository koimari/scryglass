#!/usr/bin/env python3
"""
Live win probability + cashout decision.

Starts from draft/matchup P(win), then applies calibrated log-odds
adjustments for clock, kill diff, gold diff, dragons, grubs.

Coefficients are approximate esports-analytics priors (not Leaguepedia-
labeled minute snapshots — those aren't in Cargo). Marked as such in output.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lol_kills.etl.paths import MODELS_DIR
from lol_kills.live import cashout_decision

ROOT = Path(__file__).resolve().parents[1]
MARKETS = ROOT / "data" / "lol" / "markets_model.json"
GAMES = ROOT / "data" / "lol" / "draft_games.json"
COEFS_PATH = MODELS_DIR / "draft_live_coefs.json"
CONC_PATH = MODELS_DIR / "champ_kill_concentration.json"

# Fallback priors when OE matrix not fit yet
COEF = {
    "kill_diff": 0.10,
    "gold_diff_k": 0.18,
    "dragon": 0.22,
    "infernal_extra": 0.05,
    "void_grub": 0.06,
    "tower": 0.18,
    "time_decay": 0.012,
    "adv_cap": 1.35,
}

_COEFS_CACHE: dict | None = None
_CONC_CACHE: dict | None = None


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _load_coefs() -> dict:
    global _COEFS_CACHE
    if _COEFS_CACHE is not None:
        return _COEFS_CACHE
    if COEFS_PATH.exists():
        _COEFS_CACHE = json.loads(COEFS_PATH.read_text())
    else:
        _COEFS_CACHE = {}
    return _COEFS_CACHE


def _load_conc() -> dict:
    global _CONC_CACHE
    if _CONC_CACHE is not None:
        return _CONC_CACHE
    if CONC_PATH.exists():
        _CONC_CACHE = json.loads(CONC_PATH.read_text())
    else:
        _CONC_CACHE = {"champs": {}}
    return _CONC_CACHE


def phase_of(minute: float) -> str:
    if minute < 14:
        return "early"
    if minute < 25:
        return "mid"
    return "late"


def gold_bin(g: float) -> str:
    edges = [-1e9, -3000, -2000, -1000, -500, 500, 1000, 2000, 3000, 1e9]
    labels = ["le-3k", "-3k--2k", "-2k--1k", "-1k--500", "even", "+500-1k", "+1k-2k", "+2k-3k", "ge+3k"]
    for i in range(len(edges) - 1):
        if edges[i] <= g < edges[i + 1]:
            return labels[i]
    return "even"


def kill_conc_from_draft(blue: list[str], red: list[str]) -> dict:
    """Draft-level kill concentration from champ table."""
    from lol_kills.etl.aliases import normalize_champ

    table = (_load_conc().get("champs") or {})
    default = 0.20
    thr = float(_load_conc().get("hypercarry_threshold") or 0.32)

    def side_stats(champs: list[str]) -> tuple[float, float, bool]:
        shares = [
            float((table.get(normalize_champ(c)) or {}).get("mean_share", default)) for c in champs
        ] or [default]
        mx = max(shares)
        return float(sum(shares) / len(shares)), mx, mx >= thr

    bm, bmax, bhyp = side_stats(blue)
    rm, rmax, rhyp = side_stats(red)
    return {
        "kill_conc_blue": round(bm, 4),
        "kill_conc_red": round(rm, 4),
        "kill_conc_diff": round(bm - rm, 4),
        "max_carry_blue": round(bmax, 4),
        "max_carry_red": round(rmax, 4),
        "blue_hypercarry": int(bhyp),
        "red_hypercarry": int(rhyp),
        "scaling_flag": int(bhyp or rhyp),
    }


def live_win_prob(
    *,
    p_pre: float,
    minute: float,
    kill_diff: float,
    gold_diff: float,
    dragons: int = 0,
    opp_dragons: int = 0,
    infernal: bool = False,
    void_grubs: int = 0,
    towers: int = 0,
    opp_towers: int = 0,
    draft_edge: float | None = None,
    kill_conc_diff: float | None = None,
    scaling_flag: int | None = None,
    blue_hypercarry: int | None = None,
    draft_q: int | None = None,
    first_dragon: int | None = None,
    first_herald: int | None = None,
    first_tower: int | None = None,
) -> dict:
    """
    p_pre = pregame/draft P(this team wins).
    Diffs are this_team - opponent.

    When draft_live_coefs.json exists and draft_edge is passed, uses OE-fit
    phase interaction model (gold × draft × concentration). Otherwise soft-cap fallback.
    """
    phase = phase_of(minute)
    coefs = _load_coefs()
    phase_c = (coefs.get("phase_coefs") or {}).get(phase)

    dragon_diff = float(dragons - opp_dragons)
    tower_diff = float(towers - opp_towers)
    gold_k = float(gold_diff) / 1000.0
    edge = float(draft_edge) if draft_edge is not None else 0.0
    conc = float(kill_conc_diff) if kill_conc_diff is not None else 0.0
    scal = int(scaling_flag or 0)
    bhc = int(blue_hypercarry) if blue_hypercarry is not None else scal

    x = logit(p_pre)
    x *= max(0.40, 1.0 - COEF["time_decay"] * minute)

    method = "softcap_fallback"
    adv = 0.0
    scaling_gap = None
    matrix_cell = None

    if phase_c and draft_edge is not None:
        method = f"oe_matrix:{phase}"
        priors = coefs.get("live_obj_priors") or {}
        # Residual draft beyond p_pre; elo already encoded in p_pre
        adv += float(phase_c.get("draft_edge", 0)) * edge * 0.35
        adv += float(phase_c.get("gold_k", 0.18)) * gold_k
        if first_dragon is not None:
            adv += float(phase_c.get("first_dragon", 0)) * float(first_dragon)
        if first_herald is not None:
            adv += float(phase_c.get("first_herald", 0)) * float(first_herald)
        if first_tower is not None:
            adv += float(phase_c.get("first_tower", 0)) * float(first_tower)
        adv += float(phase_c.get("draft_x_gold", 0)) * edge * gold_k
        adv += float(phase_c.get("conc_x_gold", 0)) * conc * gold_k
        adv += float(phase_c.get("scaling_x_gold", 0)) * scal * gold_k
        # Blue-side hypercarry × gold (forgives deficits / converts leads)
        adv += float(phase_c.get("blue_carry_x_gold", 0)) * bhc * gold_k
        # Live objs: soft priors (fit excluded end-game counts to avoid leakage)
        adv += float(priors.get("dragon_diff", COEF["dragon"])) * dragon_diff
        adv += float(priors.get("tower_diff", COEF["tower"])) * tower_diff
        adv += float(priors.get("void_grub", COEF["void_grub"])) * float(void_grubs)
        adv += float(priors.get("kill_diff", COEF["kill_diff"])) * float(kill_diff)
        if infernal and dragons > 0:
            adv += float(priors.get("infernal_extra", COEF["infernal_extra"]))
        cap = float(coefs.get("adv_cap") or COEF["adv_cap"])
        adv = cap * math.tanh(adv / cap)

        dq = draft_q if draft_q is not None else 2
        exp_map = (coefs.get("expected_gold_by_draft_q") or {}).get(
            "gold15" if phase != "early" else "gold10", {}
        )
        exp_g = exp_map.get(str(int(dq)))
        if exp_g is not None:
            scaling_gap = round(float(gold_diff) - float(exp_g), 1)

        matrix_cell = {
            "phase": phase,
            "gold_bin": gold_bin(gold_diff),
            "draft_q": int(dq),
            "draft_edge": round(edge, 4),
            "kill_conc_diff": round(conc, 4),
            "scaling_flag": scal,
            "blue_hypercarry": bhc,
        }
    else:
        adv += COEF["kill_diff"] * kill_diff
        adv += COEF["gold_diff_k"] * gold_k
        adv += COEF["dragon"] * dragon_diff
        if infernal and dragons > 0:
            adv += COEF["infernal_extra"]
        adv += COEF["void_grub"] * float(void_grubs)
        adv += COEF["tower"] * tower_diff
        cap = COEF["adv_cap"]
        adv = cap * math.tanh(adv / cap)
        matrix_cell = {"phase": phase, "gold_bin": gold_bin(gold_diff), "mode": "fallback"}

    x += adv
    p = sigmoid(x)
    return {
        "p_win": round(p, 4),
        "p_pre": round(p_pre, 4),
        "logit": round(x, 4),
        "adv_logit": round(adv, 4),
        "minute": minute,
        "phase": phase,
        "scaling_gap": scaling_gap,
        "matrix_cell": matrix_cell,
        "features": {
            "kill_diff": kill_diff,
            "gold_diff": gold_diff,
            "dragons": dragons,
            "opp_dragons": opp_dragons,
            "infernal": infernal,
            "void_grubs": void_grubs,
            "towers": towers,
            "opp_towers": opp_towers,
            "draft_edge": draft_edge,
            "kill_conc_diff": kill_conc_diff,
            "scaling_flag": scaling_flag,
            "blue_hypercarry": blue_hypercarry,
        },
        "method": method,
    }


def draft_q_from_edge(edge: float) -> int:
    """Approximate draft quintile from win_edge magnitude."""
    if edge < -0.15:
        return 0
    if edge < -0.05:
        return 1
    if edge < 0.05:
        return 2
    if edge < 0.15:
        return 3
    return 4


def live_win_from_draft(
    *,
    p_pre: float,
    minute: float,
    kill_diff: float,
    gold_diff: float,
    blue: list[str],
    red: list[str],
    league: str | None = None,
    dragons: int = 0,
    opp_dragons: int = 0,
    infernal: bool = False,
    void_grubs: int = 0,
    void_grubs_blue: int | None = None,
    void_grubs_red: int | None = None,
    towers: int = 0,
    opp_towers: int = 0,
    first_dragon: int | None = None,
    first_herald: int | None = None,
    first_tower: int | None = None,
) -> dict:
    """Score live win with phase-bucket draft_edge + kill concentration."""
    from lol_kills.draft_phase_score import (
        draft_edge_at_minute,
        draft_score_composite,
        nearest_bucket,
    )

    kwargs = {"league": league} if league else {}
    ds = draft_score_composite(blue, red, **kwargs)
    edge = draft_edge_at_minute(ds, minute)
    cf = kill_conc_from_draft(blue, red)
    # Prefer explicit side counts → net (this team − opp). Default: blue is "this team".
    if void_grubs_blue is not None or void_grubs_red is not None:
        vb = int(void_grubs_blue or 0)
        vr = int(void_grubs_red or 0)
        void_net = vb - vr
    else:
        void_net = int(void_grubs)
        vb = vr = None
    out = live_win_prob(
        p_pre=p_pre,
        minute=minute,
        kill_diff=kill_diff,
        gold_diff=gold_diff,
        dragons=dragons,
        opp_dragons=opp_dragons,
        infernal=infernal,
        void_grubs=void_net,
        towers=towers,
        opp_towers=opp_towers,
        draft_edge=edge,
        kill_conc_diff=cf["kill_conc_diff"],
        scaling_flag=cf["scaling_flag"],
        blue_hypercarry=cf.get("blue_hypercarry", 0),
        draft_q=draft_q_from_edge(edge),
        first_dragon=first_dragon,
        first_herald=first_herald,
        first_tower=first_tower,
    )
    out["draft_score"] = {
        "win_edge": edge,
        "bucket": nearest_bucket(minute),
        "p_blue_composite": ds.get("p_blue_draft"),
        "curve": ds.get("curve"),
        "beatdown": ds.get("beatdown"),
        "buckets": ds.get("buckets"),
        "concentration": cf,
    }
    if vb is not None:
        out["features"]["void_grubs_blue"] = vb
        out["features"]["void_grubs_red"] = vr
        out["features"]["void_grubs_net"] = void_net
    return out


GRUBS_DECISION_PATH = MODELS_DIR / "grubs_decision_numbers.json"
GRUB_CLOCK_END = 14.75  # void grubs despawn ~14:45


def _grubs_research_row(minute: float, void_net: int) -> dict | None:
    """
    Contest-research estimand (article / OE leave-mix), NOT live model Δpp.
    Only when grub clock is still relevant.
    """
    if minute > GRUB_CLOCK_END:
        return None
    if not GRUBS_DECISION_PATH.exists():
        return None
    try:
        art = json.loads(GRUBS_DECISION_PATH.read_text())
    except Exception:
        return None
    dpp = (art.get("deltas_pp") or {}).get("win_minus_leave_mix")
    if dpp is None:
        return None
    # Sign: if this side trails on grubs (net < 0), research row is informational
    # for the underdog contest EV, not a live map WR bump.
    return {
        "label": "grubs_research",
        "estimand": "win_minus_leave_mix (trailing contest vs leave-mix)",
        "delta_pp": round(float(dpp), 2),
        "note": (
            "Contest research only — not added into live p_win. "
            f"Live void_grub prior stays separate (net={void_net})."
        ),
        "breakeven_p_win_fight_vs_leave": (art.get("breakeven_p_win_fight") or {}).get(
            "vs_leave_mix"
        ),
    }


def objective_delta_pp_breakdown(
    *,
    p_pre: float,
    minute: float,
    kill_diff: float,
    gold_diff: float,
    blue: list[str] | None = None,
    red: list[str] | None = None,
    league: str | None = None,
    dragons: int = 0,
    opp_dragons: int = 0,
    infernal: bool = False,
    void_grubs: int = 0,
    void_grubs_blue: int | None = None,
    void_grubs_red: int | None = None,
    towers: int = 0,
    opp_towers: int = 0,
    first_dragon: int | None = None,
    first_herald: int | None = None,
    first_tower: int | None = None,
    draft_edge: float | None = None,
    kill_conc_diff: float | None = None,
    scaling_flag: int | None = None,
    blue_hypercarry: int | None = None,
    stake: float | None = None,
    odds: float | None = None,
    cashout: float | None = None,
) -> dict:
    """
    Live map WR + per-channel Δpp via ablation (tanh softcap ⇒ not linear).

    Δpp_channel = 100 * (p_full − p_with_channel_zeroed).
    Positive ⇒ that channel currently helps the scored side.
    """
    base_kw: dict = {
        "p_pre": p_pre,
        "minute": minute,
        "kill_diff": kill_diff,
        "gold_diff": gold_diff,
        "dragons": dragons,
        "opp_dragons": opp_dragons,
        "infernal": infernal,
        "void_grubs": void_grubs,
        "void_grubs_blue": void_grubs_blue,
        "void_grubs_red": void_grubs_red,
        "towers": towers,
        "opp_towers": opp_towers,
        "first_dragon": first_dragon,
        "first_herald": first_herald,
        "first_tower": first_tower,
        "league": league,
    }

    use_draft = bool(blue and red and len(blue) >= 3 and len(red) >= 3)

    def _run(overrides: dict | None = None) -> dict:
        kw = dict(base_kw)
        if overrides:
            kw.update(overrides)
        if use_draft:
            call = {k: v for k, v in kw.items() if not (k == "league" and v is None)}
            return live_win_from_draft(
                blue=list(blue or []),
                red=list(red or []),
                **call,
            )
        # No draft: live_win_prob with optional edge
        vb, vr = kw.get("void_grubs_blue"), kw.get("void_grubs_red")
        if vb is not None or vr is not None:
            void_net = int(vb or 0) - int(vr or 0)
        else:
            void_net = int(kw.get("void_grubs") or 0)
        return live_win_prob(
            p_pre=float(kw["p_pre"]),
            minute=float(kw["minute"]),
            kill_diff=float(kw["kill_diff"]),
            gold_diff=float(kw["gold_diff"]),
            dragons=int(kw.get("dragons") or 0),
            opp_dragons=int(kw.get("opp_dragons") or 0),
            infernal=bool(kw.get("infernal")),
            void_grubs=void_net,
            towers=int(kw.get("towers") or 0),
            opp_towers=int(kw.get("opp_towers") or 0),
            draft_edge=draft_edge,
            kill_conc_diff=kill_conc_diff,
            scaling_flag=scaling_flag,
            blue_hypercarry=blue_hypercarry,
            draft_q=draft_q_from_edge(float(draft_edge or 0.0)) if draft_edge is not None else None,
            first_dragon=kw.get("first_dragon"),
            first_herald=kw.get("first_herald"),
            first_tower=kw.get("first_tower"),
        )

    full = _run()
    p_full = float(full["p_win"])

    channels = {
        "gold": {"gold_diff": 0.0},
        "kills": {"kill_diff": 0.0},
        "dragons": {"dragons": 0, "opp_dragons": 0, "infernal": False},
        "towers": {"towers": 0, "opp_towers": 0},
        "void_grubs": {
            "void_grubs": 0,
            "void_grubs_blue": 0 if void_grubs_blue is not None or void_grubs_red is not None else None,
            "void_grubs_red": 0 if void_grubs_blue is not None or void_grubs_red is not None else None,
        },
        "first_dragon": {"first_dragon": None},
        "first_herald": {"first_herald": None},
        "first_tower": {"first_tower": None},
    }
    # Drop first-* ablations when not provided (idle None→None)
    if first_dragon is None:
        channels.pop("first_dragon", None)
    if first_herald is None:
        channels.pop("first_herald", None)
    if first_tower is None:
        channels.pop("first_tower", None)

    deltas: dict[str, float] = {}
    for name, ov in channels.items():
        clean = {k: v for k, v in ov.items() if v is not None or k.startswith("first_")}
        # For firsts, ablate by setting to 0 (neutral) not None
        for fk in ("first_dragon", "first_herald", "first_tower"):
            if fk in ov and ov[fk] is None and name == fk:
                clean[fk] = 0
        ab = _run(clean)
        deltas[name] = round(100.0 * (p_full - float(ab["p_win"])), 2)

    # vs pregame
    delta_vs_pre = round(100.0 * (p_full - float(p_pre)), 2)
    top = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))
    top = [{"channel": k, "delta_pp": v} for k, v in top if abs(v) >= 0.05][:6]

    if void_grubs_blue is not None or void_grubs_red is not None:
        void_net = int(void_grubs_blue or 0) - int(void_grubs_red or 0)
    else:
        void_net = int(void_grubs)

    grubs_research = _grubs_research_row(float(minute), void_net)

    out: dict = {
        "p_win": round(p_full, 4),
        "p_pre": round(float(p_pre), 4),
        "fair_odds": round(1.0 / max(p_full, 1e-6), 3),
        "fair_odds_opp": round(1.0 / max(1.0 - p_full, 1e-6), 3),
        "delta_vs_pre_pp": delta_vs_pre,
        "phase": full.get("phase"),
        "minute": minute,
        "method": full.get("method"),
        "deltas_pp": deltas,
        "top": top,
        "features": full.get("features"),
        "grubs_research": grubs_research,
        "note": (
            "Δpp via ablation of live_win softcap model. "
            "grubs_research is contest estimand — not in p_win."
        ),
    }
    if full.get("draft_score"):
        out["draft_score"] = full["draft_score"]

    if stake is not None and odds is not None:
        payout = float(stake) * float(odds)
        fair_cash = round(p_full * payout, 2)
        out["ticket"] = {
            "stake": float(stake),
            "odds": float(odds),
            "payout": round(payout, 2),
            "fair_cashout": fair_cash,
            "hold_ev": round(p_full * payout - float(stake), 2),
        }
        if cashout is not None:
            out["cashout"] = decide_cashout(
                p_win=p_full, stake=float(stake), odds=float(odds), cashout=float(cashout)
            )
    return out


def series_win_prob(
    *,
    p_map_live: float,
    p_map_future: float,
    best_of: int = 5,
    maps_won: int = 0,
    maps_lost: int = 0,
) -> dict:
    """
    P(win BoX) with current map still live, then iid future maps at p_map_future.

    Future maps should NOT fully inherit the live deficit — new draft each map.
    Typical: p_map_future ≈ 0.75*p_pre + 0.25*p_map_live
    """
    need = best_of // 2 + 1
    need_a = need - maps_won
    need_b = need - maps_lost

    from functools import lru_cache

    @lru_cache(None)
    def reach(a_need: int, b_need: int, p: float) -> float:
        if a_need <= 0:
            return 1.0
        if b_need <= 0:
            return 0.0

        @lru_cache(None)
        def f(a: int, b: int) -> float:
            if a >= a_need:
                return 1.0
            if b >= b_need:
                return 0.0
            return p * f(a + 1, b) + (1 - p) * f(a, b + 1)

        return f(0, 0)

    # Current map undecided: win it → (maps_won+1, maps_lost); lose → opposite
    p_if_win = reach(need_a - 1, need_b, p_map_future) if need_a > 1 else 1.0
    p_if_lose = reach(need_a, need_b - 1, p_map_future) if need_b > 1 else 0.0
    if need_a <= 0:
        p_series = 1.0
    elif need_b <= 0:
        p_series = 0.0
    else:
        p_series = p_map_live * p_if_win + (1 - p_map_live) * p_if_lose

    return {
        "p_series": round(p_series, 4),
        "p_map_live": round(p_map_live, 4),
        "p_map_future": round(p_map_future, 4),
        "best_of": best_of,
        "need": need,
        "maps_won": maps_won,
        "maps_lost": maps_lost,
        "p_if_win_map": round(p_if_win, 4),
        "p_if_lose_map": round(p_if_lose, 4),
        "fair_odds": round(1 / p_series, 3) if p_series > 0.01 else None,
    }


def series_from_live(
    *,
    p_pre: float,
    p_map_live: float,
    best_of: int = 5,
    maps_won: int = 0,
    maps_lost: int = 0,
    future_pre_weight: float = 1.0,
) -> dict:
    """
    Series P with live current map + future maps at pre-draft strength by default.

    Default weight 1.0 on p_pre for maps 2+ — each map is a new draft; live
    deficit on map 1 must not be copy-pasted into the rest of the series.
    """
    w = min(max(future_pre_weight, 0.0), 1.0)
    p_fut = w * p_pre + (1.0 - w) * p_map_live
    out = series_win_prob(
        p_map_live=p_map_live,
        p_map_future=p_fut,
        best_of=best_of,
        maps_won=maps_won,
        maps_lost=maps_lost,
    )
    out["p_pre"] = round(p_pre, 4)
    out["future_pre_weight"] = w
    return out


def decide_cashout(
    *,
    p_win: float,
    stake: float,
    odds: float,
    cashout: float,
) -> dict:
    d = cashout_decision(
        p_under=p_win, stake=stake, original_odds=odds, cashout=cashout
    )
    # Stronger messaging
    edge = d["edge_hold_vs_cashout"]
    if edge > stake * 0.15:
        verdict = "HOLD"
        reason = f"Hold EV exceeds cashout by R${edge:.2f} (>15% of stake)."
    elif edge < -stake * 0.10:
        verdict = "CASHOUT"
        reason = f"Cashout beats hold EV by R${-edge:.2f}."
    elif abs(edge) <= stake * 0.05:
        verdict = "HOLD" if edge >= 0 else "CASHOUT"
        reason = f"Too close (edge R${edge:+.2f}); default to {'hold locked price' if edge >= 0 else 'take cash'}."
    else:
        verdict = "HOLD" if edge > 0 else "CASHOUT"
        reason = f"Edge R${edge:+.2f} vs cashout."
    d["verdict"] = verdict
    d["reason"] = reason
    d["implied_by_cashout"] = round(cashout / (stake * odds), 4)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", required=True, help="Team you bet on")
    ap.add_argument("--p-pre", type=float, help="Pregame win prob for --team (0-1)")
    ap.add_argument("--minute", type=float, required=True)
    ap.add_argument("--kills", type=int, required=True, help="Team kills")
    ap.add_argument("--opp-kills", type=int, required=True)
    ap.add_argument("--gold", type=float, default=0.0, help="Team gold (absolute or leave 0)")
    ap.add_argument("--opp-gold", type=float, default=0.0)
    ap.add_argument("--gold-diff", type=float, default=None, help="Team − opp gold (overrides)")
    ap.add_argument("--dragons", type=int, default=0)
    ap.add_argument("--opp-dragons", type=int, default=0)
    ap.add_argument("--infernal", action="store_true")
    ap.add_argument("--grubs", type=int, default=0)
    ap.add_argument("--opp-grubs", type=int, default=0)
    ap.add_argument("--towers", type=int, default=0)
    ap.add_argument("--opp-towers", type=int, default=0)
    ap.add_argument("--draft-edge", type=float, default=None, help="Draft win_edge (enables OE matrix)")
    ap.add_argument("--kill-conc-diff", type=float, default=None)
    ap.add_argument("--scaling-flag", type=int, default=None)
    ap.add_argument("--draft-q", type=int, default=None)
    ap.add_argument("--blue-draft", nargs=5, metavar="CHAMP", help="Blue 5 champs → auto conc")
    ap.add_argument("--red-draft", nargs=5, metavar="CHAMP", help="Red 5 champs → auto conc")
    ap.add_argument("--stake", type=float, required=True)
    ap.add_argument("--odds", type=float, required=True)
    ap.add_argument("--cashout", type=float, required=True)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    p_pre = args.p_pre
    if p_pre is None:
        raise SystemExit("Pass --p-pre from draft board (e.g. 0.646 for T1)")

    draft_edge = args.draft_edge
    kill_conc_diff = args.kill_conc_diff
    scaling_flag = args.scaling_flag
    draft_q = args.draft_q
    if args.blue_draft and args.red_draft:
        from lol_kills.draft_phase_score import draft_edge_at_minute, draft_score_composite

        ds = draft_score_composite(list(args.blue_draft), list(args.red_draft))
        if draft_edge is None:
            draft_edge = draft_edge_at_minute(ds, args.minute)
        cf = kill_conc_from_draft(list(args.blue_draft), list(args.red_draft))
        if kill_conc_diff is None:
            kill_conc_diff = cf["kill_conc_diff"]
        if scaling_flag is None:
            scaling_flag = cf["scaling_flag"]

    gd = args.gold_diff if args.gold_diff is not None else (args.gold - args.opp_gold)
    live = live_win_prob(
        p_pre=p_pre,
        minute=args.minute,
        kill_diff=args.kills - args.opp_kills,
        gold_diff=gd,
        dragons=args.dragons,
        opp_dragons=args.opp_dragons,
        infernal=args.infernal and args.dragons > 0,
        void_grubs=args.grubs - args.opp_grubs,
        towers=args.towers,
        opp_towers=args.opp_towers,
        draft_edge=draft_edge,
        kill_conc_diff=kill_conc_diff,
        scaling_flag=scaling_flag,
        draft_q=draft_q,
    )
    decision = decide_cashout(
        p_win=live["p_win"],
        stake=args.stake,
        odds=args.odds,
        cashout=args.cashout,
    )
    out = {"team": args.team, "live": live, "cashout": decision}
    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"=== LIVE WIN · {args.team} @ {args.minute:.1f}m ===")
    print(
        f"P(win)= {100*live['p_win']:.1f}%  (pre {100*live['p_pre']:.1f}%)  "
        f"cashout implies ~{100*decision['implied_by_cashout']:.1f}%  "
        f"[{live.get('method')}]"
    )
    if live.get("matrix_cell"):
        mc = live["matrix_cell"]
        print(
            f"matrix cell: phase={mc.get('phase')} gold={mc.get('gold_bin')} "
            f"draft_q={mc.get('draft_q')} scaling_gap={live.get('scaling_gap')}"
        )
    print(
        f"Stake R${args.stake:.2f} @ {args.odds} → win R${decision['payout_if_win']:.2f}  ·  "
        f"cashout R${args.cashout:.2f}"
    )
    print(
        f"EV(hold)=R${decision['ev_hold']:.2f}  vs  cash=R${decision['ev_cashout']:.2f}  "
        f"→ {decision['verdict']}"
    )
    print(decision["reason"])


if __name__ == "__main__":
    main()
