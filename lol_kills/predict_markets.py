#!/usr/bin/env python3
"""
Post-draft market sheet: winner, FB, first inhib, total kills, race-to-K.

Default output = common bet board + high-confidence leans (no odds required).
When --lines is passed, also rank EV.

  python -m lol_kills.predict_markets T1 Gen.G --league LCK \\
    --blue "Renekton,Nidalee,Azir,Varus,Nautilus" \\
    --red "Ornn,Xin Zhao,Ahri,Kai'Sa,Milio"
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from lol_kills.markets_model import predict_markets, team_strength
from lol_kills.predict_draft import parse_draft
from lol_kills.recommend import parse_lines, resolve_team

ROOT = Path(__file__).resolve().parents[1]
MARKETS = ROOT / "data" / "lol" / "markets_model.json"
KILL_MODELS = ROOT / "data" / "lol" / "kill_models.json"
GAMES = ROOT / "data" / "lol" / "draft_games.json"

# Standard book lines we always surface
KILL_LINES = [22.5, 24.5, 25.5, 26.5, 27.5, 28.5, 29.5, 30.5, 32.5, 34.5, 36.5]
HIGH_CONF = 0.70  # highlight leans ≥ this
MED_CONF = 0.60


def p_under_normal(mu: float, sd: float, line: float) -> float:
    thr = math.floor(line) + 0.5
    z = (thr - mu) / max(sd, 1e-6)
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def build_common_board(sheet: dict) -> list[dict]:
    """Flatten markets into comparable bet rows with model probability."""
    t1, t2 = sheet["team1"], sheet["team2"]
    rows: list[dict] = []

    w = sheet["winner"]
    rows.append({"market": "Winner", "selection": t1, "p": w["p_team1"], "fair_odds": w["fair_odds_team1"]})
    rows.append({"market": "Winner", "selection": t2, "p": w["p_team2"], "fair_odds": w["fair_odds_team2"]})

    fb = sheet["first_blood"]
    rows.append(
        {
            "market": "First Blood",
            "selection": t1,
            "p": fb["p_team1"],
            "fair_odds": round(1 / fb["p_team1"], 3) if fb["p_team1"] else None,
            "note": "proxy",
        }
    )
    rows.append(
        {
            "market": "First Blood",
            "selection": t2,
            "p": fb["p_team2"],
            "fair_odds": round(1 / fb["p_team2"], 3) if fb["p_team2"] else None,
            "note": "proxy",
        }
    )

    fi = sheet["first_inhibitor"]
    rows.append(
        {
            "market": "First Inhibitor",
            "selection": t1,
            "p": fi["p_team1"],
            "fair_odds": round(1 / fi["p_team1"], 3) if fi["p_team1"] else None,
        }
    )
    rows.append(
        {
            "market": "First Inhibitor",
            "selection": t2,
            "p": fi["p_team2"],
            "fair_odds": round(1 / fi["p_team2"], 3) if fi["p_team2"] else None,
        }
    )

    mu = sheet["total_kills"]["mean"]
    sd = sheet["total_kills"]["sd"]
    for line in KILL_LINES:
        pu = p_under_normal(mu, sd, line)
        rows.append(
            {
                "market": f"Total kills O/U {line}",
                "selection": f"Under {line}",
                "p": round(pu, 4),
                "fair_odds": round(1 / pu, 3) if pu > 0.01 else None,
            }
        )
        rows.append(
            {
                "market": f"Total kills O/U {line}",
                "selection": f"Over {line}",
                "p": round(1 - pu, 4),
                "fair_odds": round(1 / (1 - pu), 3) if pu < 0.99 else None,
            }
        )

    for k, v in sheet["race_to_kills"].items():
        rows.append(
            {
                "market": f"Race to {k} kills",
                "selection": t1,
                "p": v["p_team1_first"],
                "fair_odds": round(1 / v["p_team1_first"], 3) if v["p_team1_first"] else None,
            }
        )
        rows.append(
            {
                "market": f"Race to {k} kills",
                "selection": t2,
                "p": v["p_team2_first"],
                "fair_odds": round(1 / v["p_team2_first"], 3) if v["p_team2_first"] else None,
            }
        )

    return rows


def lean_label(p: float) -> str:
    if p >= 0.90:
        return "LOCK-ISH"
    if p >= HIGH_CONF:
        return "STRONG"
    if p >= MED_CONF:
        return "LEAN"
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("team1")
    ap.add_argument("team2")
    ap.add_argument("--league", default="LCK")
    ap.add_argument("--blue", required=True, help="Team1 (blue) 5 champs")
    ap.add_argument("--red", required=True, help="Team2 (red) 5 champs")
    ap.add_argument("--lines", default="", help="Optional LINE:OVER/UNDER for EV ranking")
    ap.add_argument("--race", default="10,15,20")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    payload = json.loads(MARKETS.read_text())
    fitted = payload["model"]
    games = json.loads(GAMES.read_text())["games"]
    strengths = team_strength(games)

    known_teams = list({g["team1"] for g in games} | {g["team2"] for g in games})
    if KILL_MODELS.exists():
        known_teams = list(set(known_teams) | set(json.loads(KILL_MODELS.read_text())["teams"]))

    t1 = resolve_team(args.team1, known_teams)
    t2 = resolve_team(args.team2, known_teams)
    known_champs = fitted["champs"]
    blue = parse_draft(args.blue, known_champs)
    red = parse_draft(args.red, known_champs)
    races = [int(x) for x in args.race.split(",") if x.strip()]

    sheet = predict_markets(
        fitted,
        strengths,
        team1=t1,
        team2=t2,
        blue=blue,
        red=red,
        league=args.league,
        race_targets=races,
    )
    board = build_common_board(sheet)
    sheet["common_board"] = board

    # Unique leans: for binary markets keep only the favorite side; for O/U keep both if interesting
    leans = []
    seen_mk = set()
    for row in sorted(board, key=lambda r: -r["p"]):
        if row["p"] < MED_CONF:
            continue
        # avoid listing both sides of same market
        key = row["market"]
        if key in seen_mk:
            continue
        seen_mk.add(key)
        leans.append({**row, "lean": lean_label(row["p"])})
    sheet["leans"] = leans

    if args.lines:
        mu = sheet["total_kills"]["mean"]
        sd = sheet["total_kills"]["sd"]
        bets = []
        for line, o_o, o_u in parse_lines(args.lines):
            pu = p_under_normal(mu, sd, line)
            for side, odds, p in (("Under", o_u, pu), ("Over", o_o, 1 - pu)):
                bets.append(
                    {
                        "line": line,
                        "side": side,
                        "odds": odds,
                        "model_p": round(p, 4),
                        "ev_per_1": round(p * odds - 1, 4),
                        "edge_pp": round(100 * (p - 1 / odds), 2),
                    }
                )
        sheet["kill_bets"] = sorted(bets, key=lambda b: -b["ev_per_1"])

    if args.json:
        print(json.dumps(sheet, indent=2))
        return

    print(f"=== POST-DRAFT BOARD: {t1} (blue) vs {t2} (red) · {args.league} ===")
    print(f"Blue: {', '.join(blue)}")
    print(f"Red:  {', '.join(red)}")
    print(
        f"Kills model ~ Normal(μ={sheet['total_kills']['mean']}, "
        f"σ={sheet['total_kills']['sd']})  ·  "
        f"team kills {t1}≈{sheet['team_kills']['team1_mean']} / "
        f"{t2}≈{sheet['team_kills']['team2_mean']}"
    )
    print()

    print("--- HIGH-CONFIDENCE / LEANS (≥60%) ---")
    if not leans:
        print("  (none — coin-flip board)")
    for L in leans:
        tag = f"  [{L['lean']}]" if L["lean"] else ""
        note = f"  ({L['note']})" if L.get("note") else ""
        print(
            f"  {100*L['p']:5.1f}%  {L['selection']:22}  "
            f"fair≈{L['fair_odds']}  · {L['market']}{tag}{note}"
        )

    print()
    print("--- COMMON TOTAL KILLS LINES ---")
    print(f"  {'Line':>6}  {'P(Under)':>8}  {'P(Over)':>8}  {'Fav':>12}")
    mu = sheet["total_kills"]["mean"]
    sd = sheet["total_kills"]["sd"]
    for line in KILL_LINES:
        pu = p_under_normal(mu, sd, line)
        fav = f"Under {line}" if pu >= 0.5 else f"Over {line}"
        mark = ""
        if max(pu, 1 - pu) >= 0.90:
            mark = " ★"
        elif max(pu, 1 - pu) >= HIGH_CONF:
            mark = " ●"
        print(f"  {line:6.1f}  {100*pu:7.1f}%  {100*(1-pu):7.1f}%  {fav}{mark}")

    print()
    print("--- OTHER MARKETS ---")
    w = sheet["winner"]
    print(f"  Winner        {t1} {100*w['p_team1']:.1f}%  ·  {t2} {100*w['p_team2']:.1f}%")
    fb = sheet["first_blood"]
    print(f"  First Blood   {t1} {100*fb['p_team1']:.1f}%  ·  {t2} {100*fb['p_team2']:.1f}%  (proxy)")
    fi = sheet["first_inhibitor"]
    print(f"  First Inhib   {t1} {100*fi['p_team1']:.1f}%  ·  {t2} {100*fi['p_team2']:.1f}%")
    for k, v in sheet["race_to_kills"].items():
        print(
            f"  Race to {k:>2}   {t1} {100*v['p_team1_first']:.1f}%  ·  "
            f"{t2} {100*v['p_team2_first']:.1f}%"
        )

    print()
    print("Top pick ΔWR% (pp at 50-50):")
    for row in sheet["pick_wr_contributions"][:8]:
        print(f"  {row['team']:12} {row['champion']:16} {row['delta_wr_pp_at_50']:+.2f}pp")

    if sheet.get("kill_bets"):
        print()
        print("--- EV vs YOUR ODDS ---")
        print(f"  {'Line':>6} {'Side':>6} {'Odds':>5} {'Model%':>7} {'EV/$1':>7}")
        for b in sheet["kill_bets"]:
            print(
                f"  {b['line']:6.1f} {b['side']:>6} {b['odds']:5.2f} "
                f"{100*b['model_p']:6.1f}% {b['ev_per_1']:+7.3f}"
            )
        best = sheet["kill_bets"][0]
        print(
            f"  BEST EV: {best['side']} {best['line']} @ {best['odds']} "
            f"(P={100*best['model_p']:.1f}%, EV={best['ev_per_1']:+.3f})"
        )
    else:
        print()
        print("Send book odds next (e.g. Under 30.5 @ 1.85) to rank EV.")


if __name__ == "__main__":
    main()
