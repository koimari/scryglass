#!/usr/bin/env python3
"""
Live WR glance: checklist → objective Δpp → HOLD/CASHOUT.

  python3 -m lol_kills.skills.live_glance --help

Fast path: no build_board unless --p-pre missing and draft provided.
"""

from __future__ import annotations

import argparse
import re
from typing import Any

from lol_kills.etl.aliases import normalize_champ
from lol_kills.live_win import decide_cashout, objective_delta_pp_breakdown


def _parse_champs(s: str | None) -> list[str]:
    if not s:
        return []
    return [normalize_champ(c.strip()) for c in s.split(",") if c.strip()]


def _clock_to_minute(clock: str) -> float:
    m = re.match(r"^(\d{1,2}):(\d{2})$", clock.strip())
    if not m:
        raise SystemExit(f"Bad clock {clock!r} — want mm:ss")
    return int(m.group(1)) + int(m.group(2)) / 60.0


def _as_gold(g: float) -> float:
    return g * 1000.0 if 0 < abs(g) < 200 else g


def format_glance(br: dict[str, Any], *, team: str, opp: str | None = None) -> str:
    p = float(br["p_win"])
    lines = [
        f"### Live glance · {team}",
        "",
        f"**{team} {100 * p:.0f}%** · fair **{br['fair_odds']:.2f}**"
        + (f" · {opp} {100 * (1 - p):.0f}% / {br['fair_odds_opp']:.2f}" if opp else "")
        + f" · vs pre **{br['delta_vs_pre_pp']:+.1f}pp** · {br.get('phase')} @ {br.get('minute')}'",
        "",
        "**Δpp (ablation):**",
    ]
    for t in br.get("top") or []:
        lines.append(f"- {t['channel']}: **{t['delta_pp']:+.2f}pp**")
    if not br.get("top"):
        lines.append("- (flat)")

    gr = br.get("grubs_research")
    if gr:
        lines.append("")
        lines.append(
            f"**Grubs research** (contest, not in WR): "
            f"win−leave_mix ≈ **+{gr['delta_pp']:.2f}pp** · "
            f"breakeven fight≈{gr.get('breakeven_p_win_fight_vs_leave')}"
        )

    ticket = br.get("ticket")
    if ticket:
        lines.append("")
        lines.append(
            f"**Ticket:** stake R${ticket['stake']:.2f} @ {ticket['odds']:.2f} → "
            f"fair cashout ≈ **R${ticket['fair_cashout']:.2f}**"
        )
    co = br.get("cashout")
    if co:
        lines.append(f"**{co['verdict']}** — {co['reason']}")

    lines.append("")
    lines.append(f"_method {br.get('method')}_")
    return "\n".join(lines)


def checklist_block(
    *,
    clock: str,
    kills: str,
    gold: str | None,
    towers: str | None,
    dragons: str | None,
    grubs: str | None,
) -> str:
    return "\n".join(
        [
            "**State check**",
            f"1. Clock: {clock}",
            f"2. Kills: {kills}",
            f"3. Gold: {gold or '—'}",
            f"4. Towers: {towers or '—'}",
            f"5. Dragons (icons, not rounded): {dragons or '—'}",
            f"6. Grubs (≠ dragons): {grubs or '—'}",
            "7. Champs: from paste / user (OCR loses to user draft)",
            "",
        ]
    )


def _attach_ticket(br: dict, p: float, stake: float | None, odds: float | None, cashout: float | None) -> None:
    if stake is None or odds is None:
        return
    payout = float(stake) * float(odds)
    br["ticket"] = {
        "stake": float(stake),
        "odds": float(odds),
        "payout": round(payout, 2),
        "fair_cashout": round(p * payout, 2),
        "hold_ev": round(p * payout - float(stake), 2),
    }
    if cashout is not None:
        br["cashout"] = decide_cashout(
            p_win=p, stake=float(stake), odds=float(odds), cashout=float(cashout)
        )


def resolve_p_pre(
    *,
    p_pre: float | None,
    team: str,
    opp: str | None,
    blue: list[str] | None,
    red: list[str] | None,
    league: str | None,
    team_is_blue: bool,
) -> float:
    if p_pre is not None:
        return float(p_pre)
    if not (blue and red and len(blue) >= 3 and len(red) >= 3):
        raise SystemExit("Need --p-pre (from last board) or full draft for cold start.")
    from lol_kills.board import build_board

    t1 = team if team_is_blue else (opp or "Opp")
    t2 = (opp or "Opp") if team_is_blue else team
    sheet = build_board(
        t1,
        t2,
        league=league,
        blue=blue,
        red=red,
        journal=False,
    )
    # Board p is typically for team1 (= blue here)
    p_blue = float(
        sheet.get("p_team1")
        or sheet.get("p_win")
        or (sheet.get("fair") or {}).get("p_team1")
        or 0.5
    )
    return p_blue if team_is_blue else (1.0 - p_blue)


def run(
    *,
    team: str,
    p_pre: float | None,
    minute: float,
    kills: int,
    opp_kills: int,
    gold: float = 0.0,
    opp_gold: float = 0.0,
    dragons: int = 0,
    opp_dragons: int = 0,
    grubs: int = 0,
    opp_grubs: int = 0,
    towers: int = 0,
    opp_towers: int = 0,
    blue: list[str] | None = None,
    red: list[str] | None = None,
    league: str | None = None,
    team_is_blue: bool = True,
    stake: float | None = None,
    odds: float | None = None,
    cashout: float | None = None,
    opp: str | None = None,
    skip_checklist: bool = False,
    clock: str | None = None,
) -> str:
    p_pre_team = resolve_p_pre(
        p_pre=p_pre,
        team=team,
        opp=opp,
        blue=blue,
        red=red,
        league=league,
        team_is_blue=team_is_blue,
    )

    has_draft = bool(blue and red and len(blue) >= 3 and len(red) >= 3)

    if has_draft:
        # live_win_from_draft always scores BLUE — frame inputs as blue, invert if needed
        if team_is_blue:
            br = objective_delta_pp_breakdown(
                p_pre=p_pre_team,
                minute=minute,
                kill_diff=kills - opp_kills,
                gold_diff=gold - opp_gold,
                blue=list(blue or []),
                red=list(red or []),
                league=league,
                dragons=dragons,
                opp_dragons=opp_dragons,
                void_grubs_blue=grubs,
                void_grubs_red=opp_grubs,
                towers=towers,
                opp_towers=opp_towers,
                stake=stake,
                odds=odds,
                cashout=cashout,
            )
        else:
            br_b = objective_delta_pp_breakdown(
                p_pre=1.0 - p_pre_team,
                minute=minute,
                kill_diff=opp_kills - kills,
                gold_diff=opp_gold - gold,
                blue=list(blue or []),
                red=list(red or []),
                league=league,
                dragons=opp_dragons,
                opp_dragons=dragons,
                void_grubs_blue=opp_grubs,
                void_grubs_red=grubs,
                towers=opp_towers,
                opp_towers=towers,
            )
            p_team = 1.0 - float(br_b["p_win"])
            # Ablation labels in ticket frame (softcap / OE without side draft)
            edge = (br_b.get("draft_score") or {}).get("win_edge")
            br = objective_delta_pp_breakdown(
                p_pre=p_pre_team,
                minute=minute,
                kill_diff=kills - opp_kills,
                gold_diff=gold - opp_gold,
                dragons=dragons,
                opp_dragons=opp_dragons,
                void_grubs=grubs - opp_grubs,
                towers=towers,
                opp_towers=opp_towers,
                draft_edge=-float(edge) if edge is not None else None,
            )
            br["p_win"] = round(p_team, 4)
            br["fair_odds"] = round(1.0 / max(p_team, 1e-6), 3)
            br["fair_odds_opp"] = round(1.0 / max(1.0 - p_team, 1e-6), 3)
            br["delta_vs_pre_pp"] = round(100.0 * (p_team - p_pre_team), 2)
            br["method"] = br_b.get("method")
            br["grubs_research"] = br_b.get("grubs_research")
            _attach_ticket(br, p_team, stake, odds, cashout)
    else:
        br = objective_delta_pp_breakdown(
            p_pre=p_pre_team,
            minute=minute,
            kill_diff=kills - opp_kills,
            gold_diff=gold - opp_gold,
            dragons=dragons,
            opp_dragons=opp_dragons,
            void_grubs=grubs - opp_grubs,
            towers=towers,
            opp_towers=opp_towers,
            stake=stake,
            odds=odds,
            cashout=cashout,
        )

    parts: list[str] = []
    if not skip_checklist:
        parts.append(
            checklist_block(
                clock=clock or f"{int(minute)}:{int(round((minute % 1) * 60)):02d}",
                kills=f"{kills}-{opp_kills}",
                gold=f"{gold / 1000:.1f}k-{opp_gold / 1000:.1f}k" if gold or opp_gold else None,
                towers=f"{towers}-{opp_towers}",
                dragons=f"{dragons}/{opp_dragons}",
                grubs=f"{grubs}/{opp_grubs}",
            )
        )
    parts.append(format_glance(br, team=team, opp=opp))
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team", required=True, help="Side you want p_win for")
    ap.add_argument("--opp", default=None)
    ap.add_argument("--p-pre", type=float, default=None, help="Pregame P(team) 0-1")
    ap.add_argument("--clock", default=None, help="mm:ss")
    ap.add_argument("--minute", type=float, default=None)
    ap.add_argument("--kills", type=int, required=True)
    ap.add_argument("--opp-kills", type=int, required=True)
    ap.add_argument("--gold", type=float, default=0.0, help="Absolute gold (or k if <200)")
    ap.add_argument("--opp-gold", type=float, default=0.0)
    ap.add_argument("--dragons", type=int, default=0)
    ap.add_argument("--opp-dragons", type=int, default=0)
    ap.add_argument("--grubs", type=int, default=0)
    ap.add_argument("--opp-grubs", type=int, default=0)
    ap.add_argument("--towers", type=int, default=0)
    ap.add_argument("--opp-towers", type=int, default=0)
    ap.add_argument("--blue", default=None, help="Comma champs")
    ap.add_argument("--red", default=None)
    ap.add_argument("--league", default=None)
    ap.add_argument("--team-is-blue", default="true")
    ap.add_argument("--stake", type=float, default=None)
    ap.add_argument("--odds", type=float, default=None)
    ap.add_argument("--cashout", type=float, default=None)
    ap.add_argument("--skip-checklist", action="store_true")
    args = ap.parse_args()

    minute = args.minute
    if minute is None:
        if not args.clock:
            raise SystemExit("Pass --clock mm:ss or --minute")
        minute = _clock_to_minute(args.clock)

    team_is_blue = args.team_is_blue.strip().lower() in {"1", "true", "yes", "y"}
    text = run(
        team=args.team,
        opp=args.opp,
        p_pre=args.p_pre,
        minute=minute,
        kills=args.kills,
        opp_kills=args.opp_kills,
        gold=_as_gold(args.gold),
        opp_gold=_as_gold(args.opp_gold),
        dragons=args.dragons,
        opp_dragons=args.opp_dragons,
        grubs=args.grubs,
        opp_grubs=args.opp_grubs,
        towers=args.towers,
        opp_towers=args.opp_towers,
        blue=_parse_champs(args.blue) or None,
        red=_parse_champs(args.red) or None,
        league=args.league,
        team_is_blue=team_is_blue,
        stake=args.stake,
        odds=args.odds,
        cashout=args.cashout,
        skip_checklist=args.skip_checklist,
        clock=args.clock,
    )
    print(text)


if __name__ == "__main__":
    main()
