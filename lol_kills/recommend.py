#!/usr/bin/env python3
"""Recommend best total-kills O/U bets given team matchup + decimal odds."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lol_kills.build import pair_model

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DS = ROOT / "data" / "lol" / "kill_models.json"

# Common aliases → Leaguepedia-ish names in our dataset
ALIASES = {
    "geng": "Gen.G",
    "gen.g": "Gen.G",
    "gen g": "Gen.G",
    "t1": "T1",
    "skt": "T1",
    "hle": "Hanwha Life Esports",
    "hanwha": "Hanwha Life Esports",
    "dk": "Dplus Kia",
    "dplus": "Dplus Kia",
    "kt": "KT Rolster",
    "bro": "BRION",
    "brion": "BRION",
    "ns": "Nongshim RedForce",
    "nongshim": "Nongshim RedForce",
    "fox": "BNK FEARX",
    "fearx": "BNK FEARX",
    "bfx": "BNK FEARX",
    "dnf": "DN Freecs",
    "freecs": "DN Freecs",
    "soopers": "DN Freecs",
    "g2": "G2 Esports",
    "fnc": "Fnatic",
    "mad": "Movistar KOI",
    "koi": "Movistar KOI",
    "mkoi": "Movistar KOI",
    "gx": "GIANTX",
    "giantx": "GIANTX",
    "vit": "Team Vitality",
    "vitality": "Team Vitality",
    "sk": "SK Gaming",
    "kc": "Karmine Corp",
    "karmine": "Karmine Corp",
    "th": "Team Heretics",
    "heretics": "Team Heretics",
    "bds": "Team BDS",
    "rge": "Rogue",
    "navi": "Natus Vincere",
    "tl": "Team Liquid",
    "liquid": "Team Liquid",
    "c9": "Cloud9",
    "fly": "FlyQuest",
    "flyquest": "FlyQuest",
    "sr": "Shopify Rebellion",
    "shopify": "Shopify Rebellion",
    "dig": "Dignitas",
    "dsy": "Disguised",
    "disguised": "Disguised",
    "sen": "Sentinels",
    "sentinels": "Sentinels",
    "lyon": "LYON",
    "hle": "Hanwha Life Esports",
    "hanwha": "Hanwha Life Esports",
}


def resolve_team(name: str, known: list[str]) -> str:
    raw = name.strip()
    key = re.sub(r"\s+", " ", raw.lower())
    if key in ALIASES:
        cand = ALIASES[key]
        if cand in known:
            return cand
    # exact
    for t in known:
        if t.lower() == key:
            return t
    # substring
    hits = [t for t in known if key in t.lower() or t.lower() in key]
    if len(hits) == 1:
        return hits[0]
    if hits:
        raise SystemExit(f"Ambiguous team '{name}': {hits[:8]}")
    raise SystemExit(f"Unknown team '{name}'. Try one of: {known[:20]}...")


def parse_lines(spec: str) -> list[tuple[float, float, float]]:
    """
    Parse lines like:
      '24.5:1.15/5.10,25.5:1.19/4.35,34.5:2.25/1.60'
    or JSON list of {line, over, under}
    """
    spec = spec.strip()
    if spec.startswith("["):
        data = json.loads(spec)
        return [(float(x["line"]), float(x["over"]), float(x["under"])) for x in data]
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.match(
            r"^(?P<line>\d+(?:\.\d+)?)\s*:\s*(?P<over>\d+(?:\.\d+)?)\s*/\s*(?P<under>\d+(?:\.\d+)?)$",
            part,
        )
        if not m:
            raise SystemExit(f"Bad line spec: {part!r} (want LINE:OVER/UNDER)")
        out.append((float(m["line"]), float(m["over"]), float(m["under"])))
    return out


def score_bets(model: dict, lines: list[tuple[float, float, float]]) -> list[dict]:
    rows = []
    for line, o_over, o_under in lines:
        # use precomputed or compute from pmf
        key = str(line)
        if key in model["p_over"]:
            p_o = model["p_over"][key]
        else:
            thr = int(line) + 1
            p_o = sum(float(model["pmf"].get(str(k), 0)) for k in range(thr, 51))
            # incomplete tail — recompute from full if needed
        p_u = 1.0 - p_o
        for side, odds, p in (("Over", o_over, p_o), ("Under", o_under, p_u)):
            impl = 1.0 / odds
            ev = p * odds - 1.0
            b = odds - 1.0
            kelly = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0
            rows.append(
                {
                    "line": line,
                    "side": side,
                    "odds": odds,
                    "model_p": round(p, 4),
                    "implied_p": round(impl, 4),
                    "edge_pp": round(100 * (p - impl), 2),
                    "ev_per_1": round(ev, 4),
                    "kelly": round(kelly, 4),
                }
            )
    return rows


def recommend(dataset: dict, team1: str, team2: str, lines: list[tuple[float, float, float]]) -> dict:
    known = list(dataset["teams"])
    t1 = resolve_team(team1, known)
    t2 = resolve_team(team2, known)
    model = pair_model(dataset, t1, t2)
    # refresh p_over from full pmf rebuilt
    from lol_kills.build import blend_pmf, pmf_from_fit  # noqa — model already has p_over

    bets = score_bets(model, lines)
    # For accuracy, recompute p from pair_model internals via p_over dict;
    # extend for any line not in 20.5-39.5
    pmf = {int(k): v for k, v in model["pmf"].items()}

    def p_over(line: float) -> float:
        thr = int(line) + 1
        return sum(pmf.get(k, 0.0) for k in range(thr, 71))

    bets = []
    for line, o_over, o_under in lines:
        p_o = p_over(line)
        p_u = 1.0 - p_o
        for side, odds, p in (("Over", o_over, p_o), ("Under", o_under, p_u)):
            impl = 1.0 / odds
            ev = p * odds - 1.0
            b = odds - 1.0
            kelly = max(0.0, (b * p - (1 - p)) / b) if b > 0 else 0.0
            bets.append(
                {
                    "line": line,
                    "side": side,
                    "odds": odds,
                    "model_p": round(p, 4),
                    "implied_p": round(impl, 4),
                    "edge_pp": round(100 * (p - impl), 2),
                    "ev_per_1": round(ev, 4),
                    "kelly": round(kelly, 4),
                }
            )

    plus = [b for b in bets if b["ev_per_1"] > 0]
    best_ev = max(bets, key=lambda x: x["ev_per_1"])
    most_likely_plus = max(plus, key=lambda x: x["model_p"]) if plus else None
    best_edge = max(bets, key=lambda x: x["edge_pp"])

    return {
        "matchup": f"{t1} vs {t2}",
        "model": {
            k: model[k]
            for k in (
                "league",
                "model_mean",
                "model_sd",
                "model_median",
                "model_mode",
                "ci95",
                "strength_total",
                "h2h_recent_mean",
                "h2h_n",
                "blend_w_h2h",
                "expected_kills_team1",
                "expected_kills_team2",
                "team1_form",
                "team2_form",
                "h2h_last10",
            )
        },
        "bets": sorted(bets, key=lambda x: -x["ev_per_1"]),
        "picks": {
            "best_ev": best_ev,
            "best_edge": best_edge,
            "most_likely_among_plus_ev": most_likely_plus,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("team1")
    ap.add_argument("team2")
    ap.add_argument(
        "--lines",
        required=True,
        help="LINE:OVER/UNDER,... e.g. 24.5:1.15/5.10,34.5:2.25/1.60",
    )
    ap.add_argument("--data", type=Path, default=DEFAULT_DS)
    ap.add_argument("--json", action="store_true", help="Print full JSON")
    args = ap.parse_args()
    ds = json.loads(args.data.read_text())
    lines = parse_lines(args.lines)
    result = recommend(ds, args.team1, args.team2, lines)
    if args.json:
        print(json.dumps(result, indent=2))
        return

    m = result["model"]
    print(f"=== {result['matchup']} ({m['league']}) ===")
    print(
        f"Model total kills: mean={m['model_mean']} sd={m['model_sd']} "
        f"median={m['model_median']} mode={m['model_mode']} ci95={m['ci95']}"
    )
    print(
        f"Strength prior={m['strength_total']}  H2H recent mean={m['h2h_recent_mean']} "
        f"(n={m['h2h_n']}, w={m['blend_w_h2h']})"
    )
    print(
        f"Form: {args.team1} for/against {m['team1_form']['kills_for']}/{m['team1_form']['kills_against']} "
        f"| {args.team2} {m['team2_form']['kills_for']}/{m['team2_form']['kills_against']}"
    )
    print()
    print(f"{'Line':>6} {'Side':>6} {'Odds':>5} {'Model%':>7} {'Impl%':>6} {'Edge':>7} {'EV/$1':>7}")
    for b in result["bets"]:
        print(
            f"{b['line']:6.1f} {b['side']:>6} {b['odds']:5.2f} "
            f"{100*b['model_p']:6.1f}% {100*b['implied_p']:5.1f}% "
            f"{b['edge_pp']:+6.1f}pp {b['ev_per_1']:+7.3f}"
        )
    print()
    p = result["picks"]
    print(f"BEST EV:     {p['best_ev']['side']} {p['best_ev']['line']} @ {p['best_ev']['odds']} "
          f"(EV={p['best_ev']['ev_per_1']:+.3f}, P={100*p['best_ev']['model_p']:.1f}%)")
    if p["most_likely_among_plus_ev"]:
        x = p["most_likely_among_plus_ev"]
        print(f"MOST LIKELY (+EV): {x['side']} {x['line']} @ {x['odds']} "
              f"(P={100*x['model_p']:.1f}%, EV={x['ev_per_1']:+.3f})")


if __name__ == "__main__":
    main()
