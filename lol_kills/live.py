#!/usr/bin/env python3
"""
Live total-kills survival: P(final ≤ line | minute, kills so far).

Matchup-first: T1 vs Gen.G (or any pair) uses
  1) H2H game lengths + H2H CKPM (recent preferred)
  2) pair_model expected total (form attack×defense + H2H blend)
  3) H2H kill-total dispersion
League averages are only a fallback when H2H is thin.

Also prices cashout vs hold.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics as stats
from pathlib import Path

from lol_kills.build import pair_model
from lol_kills.live_oe_prior import blend_pair_model_mu, load_oe_pace_prior
from lol_kills.recommend import resolve_team

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "data" / "lol" / "games_raw.json"
MODELS = ROOT / "data" / "lol" / "kill_models.json"

BUCKET_EDGES = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 70]
BUCKET_BASE_RATE = [
    0.45,
    0.70,
    0.90,
    1.10,
    1.15,
    1.00,
    0.90,
    0.80,
    0.75,
    0.70,
]


def rate_at(minute: float, scale: float = 1.0) -> float:
    m = max(0.0, minute)
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= m < BUCKET_EDGES[i + 1]:
            return BUCKET_BASE_RATE[min(i, len(BUCKET_BASE_RATE) - 1)] * scale
    return BUCKET_BASE_RATE[-1] * scale


def expected_kills(t0: float, t1: float, scale: float = 1.0, step: float = 0.25) -> float:
    if t1 <= t0:
        return 0.0
    total = 0.0
    t = t0
    while t < t1:
        dt = min(step, t1 - t)
        total += rate_at(t, scale) * dt
        t += dt
    return total


def calibrate_scale(target_ckpm: float, ref_length: float = 32.0) -> float:
    base = expected_kills(0.0, ref_length, scale=1.0) / ref_length
    return target_ckpm / base if base > 0 else 1.0


def sample_nb(mean: float, dispersion: float, rng: random.Random) -> int:
    if mean <= 0:
        return 0
    if dispersion <= 1e-9:
        return _poisson(mean, rng)
    r = 1.0 / dispersion
    p = r / (r + mean)
    lam = rng.gammavariate(r, (1 - p) / p) if p < 1 else 0.0
    return _poisson(lam, rng)


def _poisson(lam: float, rng: random.Random) -> int:
    if lam <= 0:
        return 0
    if lam > 30:
        return max(0, int(rng.gauss(lam, math.sqrt(lam)) + 0.5))
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def load_matchup_prior(
    team1: str,
    team2: str,
    league: str = "LCK",
    recent_since: str = "2025-01-01",
) -> dict:
    """H2H-first pace/length prior, blended with pair_model expected total."""
    raw = json.loads(GAMES.read_text())["games"]
    ds = json.loads(MODELS.read_text())
    known = list(ds["teams"])
    t1 = resolve_team(team1, known)
    t2 = resolve_team(team2, known)

    h2h = [
        g
        for g in raw
        if g.get("length_min")
        and set([g["team1"], g["team2"]]) == {t1, t2}
    ]
    h2h_recent = [g for g in h2h if (g.get("date") or "")[:10] >= recent_since]
    pool = h2h_recent if len(h2h_recent) >= 12 else h2h

    league_games = [
        g
        for g in raw
        if g.get("league") == league and g.get("length_min") and g["length_min"] > 5
    ]

    pm = pair_model(ds, t1, t2)
    matchup_mu = float(pm["model_mean"])  # form + H2H blend

    if len(pool) >= 8:
        lengths = [float(g["length_min"]) for g in pool]
        totals = [int(g["total_kills"]) for g in pool]
        h2h_mu = _mean(totals)
        h2h_len = _mean(lengths)
        h2h_ckpm = _mean([g["total_kills"] / g["length_min"] for g in pool])
        # Anchor CKPM so E[total] ≈ blend(h2h empirical, pair model)
        # Prefer pair_model (already H2H-weighted) for level; H2H for length path
        target_mu = 0.65 * matchup_mu + 0.35 * h2h_mu
        ckpm = target_mu / h2h_len if h2h_len > 0 else h2h_ckpm
        var = stats.variance(totals) if len(totals) >= 3 else target_mu * 1.5
        # NB dispersion: Var = mu + d*mu^2 → d = (var-mu)/mu^2
        dispersion = max(0.04, (var - target_mu) / (target_mu * target_mu)) if target_mu > 1 else 0.1
        source = "h2h_recent" if pool is h2h_recent else "h2h_all"
        p_under_hist = sum(1 for t in totals if t <= 34) / len(totals)
    else:
        lengths = [float(g["length_min"]) for g in league_games]
        totals = [int(g["total_kills"]) for g in league_games]
        h2h_mu = _mean(totals)
        h2h_len = _mean(lengths)
        target_mu = matchup_mu
        ckpm = target_mu / h2h_len if h2h_len else _mean(
            [g["total_kills"] / g["length_min"] for g in league_games]
        )
        var = stats.variance(totals) if len(totals) >= 3 else target_mu * 1.5
        dispersion = max(0.04, (var - target_mu) / (target_mu * target_mu)) if target_mu > 1 else 0.1
        source = "league_fallback"
        p_under_hist = sum(1 for t in totals if t <= 34) / len(totals) if totals else None

    league_ckpm = _mean([g["total_kills"] / g["length_min"] for g in league_games])

    return {
        "team1": t1,
        "team2": t2,
        "league": pm["league"],
        "source": source,
        "n_h2h_all": len(h2h),
        "n_h2h_pool": len(pool) if len(h2h) >= 8 else 0,
        "lengths": lengths,
        "matchup_expected_total": round(matchup_mu, 3),
        "h2h_mean_total": round(h2h_mu, 3) if h2h else None,
        "target_mu": round(target_mu, 3),
        "mean_length": round(h2h_len, 3),
        "ckpm": round(ckpm, 4),
        "league_ckpm": round(league_ckpm, 4),
        "dispersion": round(dispersion, 4),
        "pair_strength_total": pm["strength_total"],
        "pair_h2h_recent_mean": pm["h2h_recent_mean"],
        "pair_h2h_n": pm["h2h_n"],
        "pair_blend_w_h2h": pm["blend_w_h2h"],
        "team1_form": pm["team1_form"],
        "team2_form": pm["team2_form"],
        "h2h_last10": pm["h2h_last10"],
        "h2h_p_under_34_hist": round(p_under_hist, 4) if p_under_hist is not None else None,
        "expected_kills_t1": pm["expected_kills_team1"],
        "expected_kills_t2": pm["expected_kills_team2"],
    }


def live_survival(
    *,
    minute: float,
    kills_so_far: int,
    line: float,
    league: str = "LCK",
    teams: tuple[str, str] = ("T1", "Gen.G"),
    n_sims: int = 80000,
    mean_revert: float = 0.55,
    seed: int = 42,
) -> dict:
    """
    OE-warehouse live kills survival.

    Prior (lengths, ckpm, target μ, dispersion) from Oracle's Elixir maps:
      team-involved maps → league (EWC etc.) → majors.
    Optional blend with legacy pair_model mean when LP form exists.
    """
    oe = load_oe_pace_prior(league, teams[0], teams[1])
    pair_mu = None
    pm = None
    legacy = None
    try:
        ds = json.loads(MODELS.read_text())
        known = list(ds["teams"])
        t1 = resolve_team(teams[0], known)
        t2 = resolve_team(teams[1], known)
        pm = pair_model(ds, t1, t2)
        pair_mu = float(pm.get("model_mean") or 0) or None
        legacy = load_matchup_prior(teams[0], teams[1], league=league)
    except Exception:
        pass

    prior = blend_pair_model_mu(oe, pair_mu, w_pair=0.25 if pair_mu else 0.0)
    lengths_all = list(prior["lengths"])
    rng = random.Random(seed)

    lengths = [L for L in lengths_all if L > minute + 0.5]
    if len(lengths) < 25:
        lengths = [L for L in lengths_all if L > minute] or lengths_all
    if len(lengths) < 5:
        lengths = [28.0, 30.0, 32.0, 34.0, 36.0, 38.0]

    ckpm = float(prior["ckpm"])
    if not math.isfinite(ckpm) or ckpm <= 0.05:
        ckpm = float(prior["target_mu"]) / max(float(prior["mean_length"]), 1.0)
    dispersion = float(prior["dispersion"])
    target_mu = float(prior["target_mu"])
    mean_length = float(prior["mean_length"]) or 32.0
    shape_scale = calibrate_scale(ckpm)

    exp_linear_now = ckpm * minute
    exp_share_now = target_mu * (minute / mean_length) if mean_length else exp_linear_now
    exp_curve_now = expected_kills(0.0, minute, scale=shape_scale)
    exp_now = 0.45 * exp_linear_now + 0.35 * exp_share_now + 0.20 * exp_curve_now

    pace_mult = (kills_so_far / exp_now) if exp_now > 1 else 1.0
    pace_mult = max(0.55, min(1.85, pace_mult))
    rem_mult = (1.0 - mean_revert) * pace_mult + mean_revert * 1.0

    under_cap = int(math.floor(line)) - kills_so_far

    finals: list[int] = []
    under = 0
    rem_means: list[float] = []
    for _ in range(n_sims):
        L = rng.choice(lengths)
        L = max(minute + 1.0, L + rng.uniform(-0.4, 0.4))
        mu_rem_ckpm = ckpm * (L - minute) * rem_mult
        mu_rem_share = max(0.0, target_mu * rem_mult - kills_so_far)
        mu_rem = 0.6 * mu_rem_ckpm + 0.4 * max(
            mu_rem_ckpm * 0.85,
            mu_rem_share * ((L - minute) / max(1.0, mean_length - minute)),
        )
        rem_means.append(mu_rem)
        R = sample_nb(mu_rem, dispersion, rng)
        final = kills_so_far + R
        finals.append(final)
        if final <= math.floor(line):
            under += 1

    finals_sorted = sorted(finals)
    p_under = under / n_sims
    mean_final = sum(finals) / n_sims

    def pct(p: float) -> float:
        return finals_sorted[min(len(finals_sorted) - 1, int(p * len(finals_sorted)))]

    typ_L = sorted(lengths)[len(lengths) // 2]
    target_rem = ckpm * max(0.0, typ_L - minute) * rem_mult
    raw_parts = []
    for i in range(len(BUCKET_EDGES) - 1):
        a, b = BUCKET_EDGES[i], BUCKET_EDGES[i + 1]
        if b <= minute:
            continue
        lo = max(a, minute)
        hi = min(b, typ_L)
        if hi <= lo:
            continue
        part = expected_kills(lo, hi, scale=shape_scale)
        raw_parts.append((a, b, lo, hi, part))
    raw_sum = sum(p for *_, p in raw_parts) or 1.0
    bucket_report = []
    for a, b, lo, hi, part in raw_parts:
        mu = target_rem * (part / raw_sum)
        bucket_report.append(
            {
                "bucket": f"{a}-{b}",
                "minutes_remaining_in_bucket": round(hi - lo, 2),
                "expected_kills": round(mu, 2),
                "rate_per_min": round(mu / (hi - lo), 3) if hi > lo else 0.0,
            }
        )

    return {
        "matchup": f"{teams[0]} vs {teams[1]}",
        "prior_source": prior["source"],
        "n_prior_maps": prior["n"],
        "n_h2h_all": (legacy or {}).get("n_h2h_all"),
        "n_h2h_pool": (legacy or {}).get("n_h2h_pool"),
        "matchup_expected_total": prior["target_mu"],
        "h2h_mean_total": (legacy or {}).get("h2h_mean_total"),
        "target_mu": prior["target_mu"],
        "matchup_ckpm": round(ckpm, 4),
        "league_ckpm": round(ckpm, 4),
        "pair_strength_total": (pm or {}).get("strength_total") if pm else None,
        "pair_h2h_recent_mean": (pm or {}).get("h2h_recent_mean") if pm else None,
        "expected_kills_t1": (pm or {}).get("expected_kills_team1") if pm else None,
        "expected_kills_t2": (pm or {}).get("expected_kills_team2") if pm else None,
        "h2h_p_under_34_hist": (prior.get("p_under_hist") or {}).get("34"),
        "minute": minute,
        "kills_so_far": kills_so_far,
        "line": line,
        "under_means_final_le": math.floor(line),
        "remaining_cap_for_under": under_cap,
        "p_under": round(p_under, 4),
        "p_over": round(1 - p_under, 4),
        "mean_final": round(mean_final, 2),
        "median_final": pct(0.5),
        "p10_final": pct(0.10),
        "p90_final": pct(0.90),
        "mean_remaining": round(sum(rem_means) / n_sims, 2),
        "expected_kills_by_now": round(exp_now, 2),
        "expected_kills_by_now_linear": round(exp_linear_now, 2),
        "expected_kills_by_now_curve": round(exp_curve_now, 2),
        "pace_mult": round(pace_mult, 3),
        "remaining_mult": round(rem_mult, 3),
        "mean_revert": mean_revert,
        "dispersion": round(dispersion, 4),
        "n_length_samples": len(lengths),
        "median_length_prior": round(typ_L, 2),
        "mean_length_prior": round(mean_length, 3),
        "fair_odds_under": round(1 / p_under, 3) if p_under > 0 else None,
        "fair_odds_over": round(1 / (1 - p_under), 3) if p_under < 1 else None,
        "n_sims": n_sims,
        "buckets_from_now": bucket_report,
        "oe_p_under_hist": prior.get("p_under_hist"),
    }


def cashout_decision(
    *,
    p_under: float,
    stake: float,
    original_odds: float,
    cashout: float,
) -> dict:
    payout_if_win = stake * original_odds
    ev_hold = p_under * payout_if_win
    ev_cash = cashout
    breakeven_p = cashout / payout_if_win if payout_if_win else 1.0
    edge_vs_cash = ev_hold - ev_cash
    return {
        "stake": stake,
        "original_odds": original_odds,
        "payout_if_win": round(payout_if_win, 2),
        "cashout": cashout,
        "p_under": round(p_under, 4),
        "ev_hold": round(ev_hold, 2),
        "ev_cashout": round(ev_cash, 2),
        "edge_hold_vs_cashout": round(edge_vs_cash, 2),
        "breakeven_p_to_cash": round(breakeven_p, 4),
        "decision": "HOLD" if ev_hold > ev_cash else "CASHOUT",
        "note": (
            "Cashout looks far below hold EV — book is pricing panic vs early kill pace."
            if ev_hold > ev_cash * 1.5
            else ""
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--minute", type=float, required=True)
    ap.add_argument("--kills", type=int, required=True)
    ap.add_argument("--line", type=float, default=34.5)
    ap.add_argument("--team1", default="T1")
    ap.add_argument("--team2", default="Gen.G")
    ap.add_argument("--league", default="LCK")
    ap.add_argument("--stake", type=float, default=0.0)
    ap.add_argument("--odds", type=float, default=0.0, help="Original decimal odds on Under")
    ap.add_argument("--cashout", type=float, default=0.0)
    ap.add_argument("--mean-revert", type=float, default=0.55)
    ap.add_argument("--sims", type=int, default=80000)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    live = live_survival(
        minute=args.minute,
        kills_so_far=args.kills,
        line=args.line,
        league=args.league,
        teams=(args.team1, args.team2),
        n_sims=args.sims,
        mean_revert=args.mean_revert,
    )
    out: dict = {"live": live}
    if args.stake and args.odds and args.cashout:
        out["cashout"] = cashout_decision(
            p_under=live["p_under"],
            stake=args.stake,
            original_odds=args.odds,
            cashout=args.cashout,
        )

    if args.json:
        print(json.dumps(out, indent=2))
        return

    L = live
    print(f"=== {L['matchup']} · live @ {L['minute']} min · {L['kills_so_far']} kills · Under {L['line']} ===")
    print(
        f"Prior: {L['prior_source']} (H2H n={L['n_h2h_all']}, pool={L['n_h2h_pool']})  "
        f"matchup μ={L['matchup_expected_total']}  H2H μ={L['h2h_mean_total']}  "
        f"target μ={L['target_mu']}"
    )
    print(
        f"Form: {args.team1} {L['team1_form']['kills_for']}/{L['team1_form']['kills_against']}  "
        f"{args.team2} {L['team2_form']['kills_for']}/{L['team2_form']['kills_against']}  "
        f"→ E[kills] {L['expected_kills_t1']}+{L['expected_kills_t2']}  "
        f"CKPM matchup={L['matchup_ckpm']} vs league={L['league_ckpm']}"
    )
    if L["h2h_p_under_34_hist"] is not None:
        print(f"H2H historical P(total≤34) in pool ≈ {100*L['h2h_p_under_34_hist']:.1f}%")
    print(
        f"P(Under)= {100*L['p_under']:.1f}%   P(Over)= {100*L['p_over']:.1f}%   "
        f"fair Under odds ≈ {L['fair_odds_under']}"
    )
    print(
        f"Projected final: mean={L['mean_final']}  median={L['median_final']}  "
        f"p10={L['p10_final']}  p90={L['p90_final']}"
    )
    print(
        f"Need ≤{L['remaining_cap_for_under']} more kills  |  E[remaining]={L['mean_remaining']}  |  "
        f"pace_mult={L['pace_mult']} vs THIS matchup "
        f"({'HOT' if L['pace_mult']>1.15 else 'normal' if L['pace_mult']>0.9 else 'SLOW'})"
    )
    print(
        f"Matchup E[kills by {L['minute']:.0f}]={L['expected_kills_by_now']} "
        f"(linear {L['expected_kills_by_now_linear']}) vs actual {L['kills_so_far']}"
    )
    print(f"Length prior median={L['median_length_prior']} (H2H mean {L['mean_length_prior']})")
    print("Remaining 5-min buckets (matchup CKPM, median H2H length):")
    for b in L["buckets_from_now"]:
        print(
            f"  {b['bucket']:>7}  rate={b['rate_per_min']:.2f}/min  "
            f"E[kills]={b['expected_kills']:.2f}  ({b['minutes_remaining_in_bucket']}m left in bucket)"
        )
    if L["h2h_last10"]:
        print("Recent H2H totals:", ", ".join(str(x["total"]) for x in L["h2h_last10"][:8]))
    if "cashout" in out:
        c = out["cashout"]
        print()
        print(
            f"Cashout: stake R${c['stake']:.2f} @ {c['original_odds']} → win R${c['payout_if_win']:.2f}"
        )
        print(
            f"EV(hold)=R${c['ev_hold']:.2f}  vs  cashout=R${c['ev_cashout']:.2f}  "
            f"→ {c['decision']} (edge R${c['edge_hold_vs_cashout']:+.2f})"
        )
        print(
            f"Breakeven to cash: need P(Under) ≤ {100*c['breakeven_p_to_cash']:.1f}% "
            f"(model has {100*c['p_under']:.1f}%)"
        )
        if c["note"]:
            print(c["note"])


if __name__ == "__main__":
    main()
