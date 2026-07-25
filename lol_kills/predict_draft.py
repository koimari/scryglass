#!/usr/bin/env python3
"""
Predict total kills from teams + optional draft.

Blends:
  1) matchup prior (lol_kills pair_model / H2H+form)
  2) draft adjustment from ridge champion presence model

Example:
  python -m lol_kills.predict_draft T1 Gen.G --league LCK \\
    --draft "Renekton,Nidalee,Azir,Varus,Nautilus,Gragas,Xin Zhao,Ryze,Sivir,Milio" \\
    --lines "25.5:1.19/4.35,34.5:2.25/1.60"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lol_kills.build import pair_model, nb_pmf, poisson_pmf
from lol_kills.draft_model import predict_total
from lol_kills.recommend import parse_lines, resolve_team

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "data" / "lol" / "kill_models.json"
DRAFT = ROOT / "data" / "lol" / "draft_model.json"

# Common OCR / shorthand → Leaguepedia names
CHAMP_ALIASES = {
    "mf": "Miss Fortune",
    "miss fortune": "Miss Fortune",
    "kaisa": "Kai'Sa",
    "kai'sa": "Kai'Sa",
    "kai sa": "Kai'Sa",
    "reksai": "Rek'Sai",
    "rek'sai": "Rek'Sai",
    "kogmaw": "Kog'Maw",
    "kog'maw": "Kog'Maw",
    "cho": "Cho'Gath",
    "chogath": "Cho'Gath",
    "cho'gath": "Cho'Gath",
    "velkoz": "Vel'Koz",
    "vel'koz": "Vel'Koz",
    "khazix": "Kha'Zix",
    "kha'zix": "Kha'Zix",
    "belveth": "Bel'Veth",
    "bel'veth": "Bel'Veth",
    "jarvan": "Jarvan IV",
    "jarvan iv": "Jarvan IV",
    "j4": "Jarvan IV",
    "monkey king": "Wukong",
    "tf": "Twisted Fate",
    "asol": "Aurelion Sol",
    "aurelion sol": "Aurelion Sol",
    "renata": "Renata Glasc",
    "renata glasc": "Renata Glasc",
    "tahm": "Tahm Kench",
    "tahm kench": "Tahm Kench",
    "lee": "Lee Sin",
    "lee sin": "Lee Sin",
    "xin": "Xin Zhao",
    "xin zhao": "Xin Zhao",
    "dr mundo": "Dr. Mundo",
    "mundo": "Dr. Mundo",
    "master yi": "Master Yi",
    "yi": "Master Yi",
    "nunu": "Nunu & Willump",
}


def normalize_champ(name: str, known: list[str]) -> str:
    raw = name.strip()
    key = raw.lower()
    if key in CHAMP_ALIASES:
        raw = CHAMP_ALIASES[key]
    for k in known:
        if k.lower() == raw.lower():
            return k
    # substring unique
    hits = [k for k in known if raw.lower() in k.lower() or k.lower() in raw.lower()]
    if len(hits) == 1:
        return hits[0]
    return raw


def parse_draft(spec: str, known: list[str]) -> list[str]:
    parts = [p.strip() for p in spec.replace(";", ",").split(",") if p.strip()]
    return [normalize_champ(p, known) for p in parts]


def blend_prediction(
    matchup_mean: float,
    matchup_sd: float,
    draft_mean: float,
    draft_sd: float,
    draft_weight: float,
) -> tuple[float, float]:
    """
    Precision-weighted blend of matchup prior and draft-implied mean.
    draft_weight in [0,1] scales draft precision (partial drafts → lower).
    """
    # Treat means as estimates with variances sd^2 / n_eff; use inverse-variance
    w_m = 1.0 / max(matchup_sd**2, 1.0)
    w_d = draft_weight / max(draft_sd**2, 1.0)
    mu = (w_m * matchup_mean + w_d * draft_mean) / (w_m + w_d)
    var = 1.0 / (w_m + w_d)
    # Keep residual game noise ~ matchup_sd
    sd = math.sqrt(var + 0.5 * matchup_sd**2)
    return mu, sd


def pmf_from_moments(mu: float, sd: float, lo: int = 0, hi: int = 70) -> dict[int, float]:
    var = sd * sd
    if var <= mu + 1e-6:
        return poisson_pmf(mu, lo, hi)
    r = mu * mu / (var - mu)
    p = r / (r + mu)
    return nb_pmf(r, p, lo, hi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("team1")
    ap.add_argument("team2")
    ap.add_argument("--league", default="LCK")
    ap.add_argument("--draft", required=True, help="Comma-separated 10 champions (both teams)")
    ap.add_argument("--lines", default="", help="LINE:OVER/UNDER,...")
    ap.add_argument("--draft-weight", type=float, default=1.0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ds = json.loads(MODELS.read_text())
    draft_payload = json.loads(DRAFT.read_text())
    dmodel = draft_payload["model"]
    known_champs = dmodel["champions"]

    t1 = resolve_team(args.team1, list(ds["teams"]))
    t2 = resolve_team(args.team2, list(ds["teams"]))
    pm = pair_model(ds, t1, t2)

    champs = parse_draft(args.draft, known_champs)
    draft_pred = predict_total(dmodel, champs, league=args.league)
    # Partial draft → downweight
    coverage = draft_pred["n_champs_applied"] / max(10, len(champs) or 10)
    dw = args.draft_weight * min(1.0, coverage)

    mu, sd = blend_prediction(
        matchup_mean=float(pm["model_mean"]),
        matchup_sd=float(pm["model_sd"]),
        draft_mean=float(draft_pred["expected_total"]),
        draft_sd=float(draft_pred["sd"]),
        draft_weight=dw,
    )
    pmf = pmf_from_moments(mu, sd)

    def p_over(line: float) -> float:
        thr = int(line) + 1
        return sum(pmf.get(k, 0.0) for k in range(thr, 71))

    result = {
        "matchup": f"{t1} vs {t2}",
        "league": args.league,
        "draft": champs,
        "matchup_prior": {
            "mean": pm["model_mean"],
            "sd": pm["model_sd"],
            "h2h_n": pm["h2h_n"],
            "h2h_mean": pm["h2h_recent_mean"],
            "strength": pm["strength_total"],
        },
        "draft_prior": draft_pred,
        "blend": {
            "mean": round(mu, 2),
            "sd": round(sd, 2),
            "draft_weight_used": round(dw, 3),
            "coverage": round(coverage, 3),
        },
    }

    if args.lines:
        lines = parse_lines(args.lines)
        bets = []
        for line, o_o, o_u in lines:
            p_o = p_over(line)
            for side, odds, p in (("Over", o_o, p_o), ("Under", o_u, 1 - p_o)):
                impl = 1 / odds
                ev = p * odds - 1
                bets.append(
                    {
                        "line": line,
                        "side": side,
                        "odds": odds,
                        "model_p": round(p, 4),
                        "implied_p": round(impl, 4),
                        "edge_pp": round(100 * (p - impl), 2),
                        "ev_per_1": round(ev, 4),
                    }
                )
        result["bets"] = sorted(bets, key=lambda x: -x["ev_per_1"])
        plus = [b for b in bets if b["ev_per_1"] > 0]
        result["picks"] = {
            "best_ev": max(bets, key=lambda x: x["ev_per_1"]),
            "most_likely_plus_ev": max(plus, key=lambda x: x["model_p"]) if plus else None,
        }

    if args.json:
        print(json.dumps(result, indent=2))
        return

    print(f"=== {result['matchup']} ({args.league}) + draft ===")
    print(
        f"Matchup prior μ={pm['model_mean']} (H2H n={pm['h2h_n']}, strength={pm['strength_total']})"
    )
    print(
        f"Draft prior μ={draft_pred['expected_total']} "
        f"(Δ vs baseline {draft_pred['delta_vs_baseline']:+.2f}, "
        f"{draft_pred['n_champs_applied']}/10 champs known)"
    )
    if draft_pred["unknown_champions"]:
        print("Unknown champs (no effect):", ", ".join(draft_pred["unknown_champions"]))
    print("Largest draft effects:")
    for e in draft_pred["effects"][:8]:
        print(f"  {e['champion']:16} {e['effect']:+.2f}")
    print(
        f"BLEND μ={result['blend']['mean']} sd={result['blend']['sd']} "
        f"(draft_weight={result['blend']['draft_weight_used']})"
    )
    if "bets" in result:
        print()
        print(f"{'Line':>6} {'Side':>6} {'Odds':>5} {'Model%':>7} {'Edge':>7} {'EV/$1':>7}")
        for b in result["bets"]:
            print(
                f"{b['line']:6.1f} {b['side']:>6} {b['odds']:5.2f} "
                f"{100*b['model_p']:6.1f}% {b['edge_pp']:+6.1f}pp {b['ev_per_1']:+7.3f}"
            )
        p = result["picks"]["best_ev"]
        print(
            f"\nBEST EV: {p['side']} {p['line']} @ {p['odds']} "
            f"(P={100*p['model_p']:.1f}%, EV={p['ev_per_1']:+.3f})"
        )


if __name__ == "__main__":
    main()
