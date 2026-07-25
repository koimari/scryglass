#!/usr/bin/env python3
"""Build per-team kill form + H2H distributions from fetched games."""

from __future__ import annotations

import argparse
import json
import math
import statistics as stats
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data" / "lol" / "games_raw.json"
DEFAULT_OUT = ROOT / "data" / "lol" / "kill_models.json"


def mean(xs: list[float]) -> float:
    return stats.mean(xs) if xs else 0.0


def stdev(xs: list[float]) -> float:
    return stats.stdev(xs) if len(xs) >= 2 else 0.0


def fit_nb(xs: list[int]) -> dict:
    """Method-of-moments NegativeBinomial (or Poisson if underdispersed)."""
    if len(xs) < 2:
        mu = mean(xs) if xs else 20.0
        return {"kind": "poisson", "mu": mu, "r": None, "p": None, "var": mu}
    mu = stats.mean(xs)
    var = stats.variance(xs)
    if var <= mu + 1e-9:
        return {"kind": "poisson", "mu": mu, "r": None, "p": None, "var": var}
    r = mu * mu / (var - mu)
    p = r / (r + mu)
    return {"kind": "nb", "mu": mu, "r": r, "p": p, "var": var}


def nb_pmf(r: float, p: float, lo: int = 0, hi: int = 70) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in range(lo, hi + 1):
        out[k] = math.exp(
            math.lgamma(k + r)
            - math.lgamma(k + 1)
            - math.lgamma(r)
            + r * math.log(p)
            + k * math.log(max(1e-15, 1 - p))
        )
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def poisson_pmf(mu: float, lo: int = 0, hi: int = 70) -> dict[int, float]:
    out: dict[int, float] = {}
    for k in range(lo, hi + 1):
        out[k] = math.exp(-mu + k * math.log(max(mu, 1e-15)) - math.lgamma(k + 1))
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def pmf_from_fit(fit: dict, lo: int = 0, hi: int = 70) -> dict[int, float]:
    if fit["kind"] == "poisson" or fit["r"] is None:
        return poisson_pmf(fit["mu"], lo, hi)
    return nb_pmf(fit["r"], fit["p"], lo, hi)


def blend_pmf(a: dict[int, float], b: dict[int, float], w_a: float) -> dict[int, float]:
    keys = set(a) | set(b)
    out = {k: w_a * a.get(k, 0.0) + (1 - w_a) * b.get(k, 0.0) for k in keys}
    s = sum(out.values())
    return {k: v / s for k, v in out.items()}


def summarize_totals(totals: list[int]) -> dict:
    if not totals:
        return {"n": 0}
    fit = fit_nb(totals)
    return {
        "n": len(totals),
        "mean": round(mean(totals), 3),
        "median": stats.median(totals),
        "sd": round(stdev(totals), 3),
        "min": min(totals),
        "max": max(totals),
        "p10": sorted(totals)[max(0, int(0.1 * len(totals)) - 1)],
        "p90": sorted(totals)[min(len(totals) - 1, int(0.9 * len(totals)))],
        "fit": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in fit.items()},
        "hist5": _hist5(totals),
    }


def _hist5(totals: list[int]) -> dict[str, int]:
    c: dict[str, int] = defaultdict(int)
    for t in totals:
        b = (t // 5) * 5
        c[f"{b}-{b+4}"] += 1
    return dict(sorted(c.items(), key=lambda x: int(x[0].split("-")[0])))


def build(games: list[dict], since_pref: str = "2025-01-01") -> dict:
    # Form from domestic leagues only; H2H includes international
    domestic = [g for g in games if g.get("league") in ("LCK", "LEC", "LCS")]
    recent = [g for g in domestic if (g.get("date") or "")[:10] >= since_pref]
    if len(recent) < 200:
        recent = domestic

    # League-wide KPG (per team) for attack/defense model
    league_games: dict[str, list[dict]] = defaultdict(list)
    for g in recent:
        league_games[g["league"]].append(g)

    league_kpg: dict[str, float] = {}
    for lg, gs in league_games.items():
        league_kpg[lg] = mean([g["total_kills"] / 2 for g in gs])

    # Per-team form (domestic recent only)
    team_for: dict[str, list[int]] = defaultdict(list)
    team_against: dict[str, list[int]] = defaultdict(list)
    team_total: dict[str, list[int]] = defaultdict(list)
    team_league: dict[str, str] = {}
    team_games_n: dict[str, int] = defaultdict(int)

    for g in recent:
        for side in ("team1", "team2"):
            team = g[side]
            tk = g["kills1"] if side == "team1" else g["kills2"]
            ok = g["kills2"] if side == "team1" else g["kills1"]
            team_for[team].append(tk)
            team_against[team].append(ok)
            team_total[team].append(g["total_kills"])
            team_league[team] = g["league"]
            team_games_n[team] += 1

    teams: dict[str, dict] = {}
    for team, n in sorted(team_games_n.items(), key=lambda x: (-x[1], x[0])):
        lg = team_league[team]
        atk = mean(team_for[team])
        dn = mean(team_against[team])
        teams[team] = {
            "league": lg,
            "n": n,
            "kills_for_mean": round(atk, 3),
            "kills_against_mean": round(dn, 3),
            "total_kills_mean": round(mean(team_total[team]), 3),
            "total_kills_sd": round(stdev(team_total[team]), 3),
            "kills_for_sd": round(stdev(team_for[team]), 3),
            "kills_against_sd": round(stdev(team_against[team]), 3),
            "total_summary": summarize_totals(team_total[team]),
            "for_fit": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in fit_nb(team_for[team]).items()
            },
            "against_fit": {
                k: (round(v, 6) if isinstance(v, float) else v)
                for k, v in fit_nb(team_against[team]).items()
            },
        }

    # H2H pairs — all games including INT, but only if both teams are known domestic sides
    known = set(teams)
    h2h_all: dict[str, list[dict]] = defaultdict(list)
    for g in games:
        if g["team1"] not in known or g["team2"] not in known:
            continue
        a, b = sorted([g["team1"], g["team2"]])
        key = f"{a}__vs__{b}"
        if g["team1"] == a:
            ka, kb = g["kills1"], g["kills2"]
        else:
            ka, kb = g["kills2"], g["kills1"]
        h2h_all[key].append(
            {
                "date": g["date"],
                "league": g["league"],
                "tournament": g["tournament"],
                "team_a": a,
                "team_b": b,
                "kills_a": ka,
                "kills_b": kb,
                "total": g["total_kills"],
            }
        )

    h2h: dict[str, dict] = {}
    for key, rows in h2h_all.items():
        rows_sorted = sorted(rows, key=lambda r: r["date"] or "", reverse=True)
        totals = [r["total"] for r in rows_sorted]
        recent_rows = [r for r in rows_sorted if (r["date"] or "")[:10] >= since_pref]
        recent_totals = [r["total"] for r in recent_rows] or totals
        a, b = key.split("__vs__")
        # primary league = shared domestic if same
        lg_a = teams[a]["league"]
        lg_b = teams[b]["league"]
        h2h[key] = {
            "team_a": a,
            "team_b": b,
            "league": lg_a if lg_a == lg_b else f"{lg_a}/{lg_b}",
            "n_all": len(totals),
            "n_recent": len(recent_rows),
            "all": summarize_totals(totals),
            "recent": summarize_totals(recent_totals),
            "mean_kills_a_recent": round(
                mean([r["kills_a"] for r in (recent_rows or rows_sorted)]), 3
            ),
            "mean_kills_b_recent": round(
                mean([r["kills_b"] for r in (recent_rows or rows_sorted)]), 3
            ),
            "last10_totals": [r["total"] for r in rows_sorted[:10]],
            "last10": [
                {
                    "date": (r["date"] or "")[:10],
                    "total": r["total"],
                    "score": f"{r['team_a']} {r['kills_a']}-{r['kills_b']} {r['team_b']}",
                    "tournament": r["tournament"],
                }
                for r in rows_sorted[:10]
            ],
        }

    leagues: dict[str, dict] = {}
    for lg, gs in league_games.items():
        totals = [g["total_kills"] for g in gs]
        leagues[lg] = {
            "n_games": len(gs),
            "kpg_per_team": round(league_kpg[lg], 3),
            "total_kills": summarize_totals(totals),
            "teams": sorted(t for t, meta in teams.items() if meta["league"] == lg),
        }

    return {
        "meta": {
            "source": "Leaguepedia Cargo ScoreboardGames",
            "n_games_input": len(games),
            "n_games_domestic": len(domestic),
            "n_games_recent_window": len(recent),
            "recent_since": since_pref,
            "n_teams": len(teams),
            "n_h2h_pairs": len(h2h),
        },
        "leagues": leagues,
        "teams": teams,
        "h2h": h2h,
    }


def pair_model(dataset: dict, team1: str, team2: str, w_h2h: float = 0.75) -> dict:
    """Predictive total-kills PMF for team1 vs team2."""
    teams = dataset["teams"]
    if team1 not in teams or team2 not in teams:
        missing = [t for t in (team1, team2) if t not in teams]
        raise KeyError(f"Unknown team(s): {missing}. Known e.g. {list(teams)[:15]}")

    a, b = team1, team2
    meta_a, meta_b = teams[a], teams[b]
    lg = meta_a["league"] if meta_a["league"] == meta_b["league"] else meta_a["league"]
    kpg = dataset["leagues"].get(lg, {}).get("kpg_per_team") or 14.0

    # Attack × defense prior
    e_a = meta_a["kills_for_mean"] * meta_b["kills_against_mean"] / kpg
    e_b = meta_b["kills_for_mean"] * meta_a["kills_against_mean"] / kpg
    e_tot = e_a + e_b

    # Dispersion from each team's game totals (average var)
    var_a = (meta_a["total_kills_sd"] or 0) ** 2
    var_b = (meta_b["total_kills_sd"] or 0) ** 2
    var_prior = max((var_a + var_b) / 2, e_tot + 1.0)
    prior_fit = {
        "kind": "nb",
        "mu": e_tot,
        "var": var_prior,
        "r": e_tot * e_tot / (var_prior - e_tot),
        "p": None,
    }
    prior_fit["p"] = prior_fit["r"] / (prior_fit["r"] + e_tot)
    prior_pmf = pmf_from_fit(prior_fit)

    # H2H if any
    key = "__vs__".join(sorted([a, b]))
    h2h = dataset["h2h"].get(key)
    if h2h and h2h["recent"]["n"] >= 5:
        h_fit = h2h["recent"]["fit"]
        # rebuild fit dict properly
        h_fit_full = {
            "kind": h_fit["kind"],
            "mu": h_fit["mu"],
            "r": h_fit.get("r"),
            "p": h_fit.get("p"),
            "var": h_fit.get("var"),
        }
        h2h_pmf = pmf_from_fit(h_fit_full)
        # shrink w_h2h if small sample
        n = h2h["recent"]["n"]
        w = w_h2h * min(1.0, n / 20)
        pmf = blend_pmf(h2h_pmf, prior_pmf, w)
        blend_w = w
        h2h_mean = h2h["recent"]["mean"]
    else:
        pmf = prior_pmf
        blend_w = 0.0
        h2h_mean = None

    mu = sum(k * v for k, v in pmf.items())
    var = sum((k - mu) ** 2 * v for k, v in pmf.items())
    mode = max(pmf, key=pmf.get)
    acc = 0.0
    lo = hi = med = None
    for k in range(0, 71):
        acc += pmf.get(k, 0)
        if lo is None and acc >= 0.025:
            lo = k
        if med is None and acc >= 0.5:
            med = k
        if hi is None and acc >= 0.975:
            hi = k

    def p_over(line: float) -> float:
        thr = int(line) + 1
        return sum(pmf.get(k, 0) for k in range(thr, 71))

    return {
        "team1": a,
        "team2": b,
        "league": lg,
        "expected_kills_team1": round(e_a, 3),
        "expected_kills_team2": round(e_b, 3),
        "strength_total": round(e_tot, 3),
        "h2h_recent_mean": h2h_mean,
        "h2h_n": (h2h["recent"]["n"] if h2h else 0),
        "blend_w_h2h": round(blend_w, 3),
        "model_mean": round(mu, 3),
        "model_sd": round(math.sqrt(var), 3),
        "model_median": med,
        "model_mode": mode,
        "ci95": [lo, hi],
        "team1_form": {
            "n": meta_a["n"],
            "kills_for": meta_a["kills_for_mean"],
            "kills_against": meta_a["kills_against_mean"],
            "total_mean": meta_a["total_kills_mean"],
        },
        "team2_form": {
            "n": meta_b["n"],
            "kills_for": meta_b["kills_for_mean"],
            "kills_against": meta_b["kills_against_mean"],
            "total_mean": meta_b["total_kills_mean"],
        },
        "pmf": {str(k): round(pmf.get(k, 0), 6) for k in range(10, 51)},
        "p_over": {str(x): round(p_over(x), 4) for x in [x + 0.5 for x in range(20, 40)]},
        "h2h_last10": (h2h["last10"] if h2h else []),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="infile", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--recent-since", default="2025-01-01")
    args = ap.parse_args()
    raw = json.loads(args.infile.read_text())
    games = raw["games"] if isinstance(raw, dict) else raw
    ds = build(games, since_pref=args.recent_since)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(ds, indent=2))
    print(
        f"wrote {args.out} teams={ds['meta']['n_teams']} "
        f"h2h={ds['meta']['n_h2h_pairs']} leagues={list(ds['leagues'])}"
    )


if __name__ == "__main__":
    main()
