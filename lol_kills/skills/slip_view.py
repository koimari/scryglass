#!/usr/bin/env python3
"""
Bet-slip 1-view: favorite, fair odds, kills buckets, decision ladders, grades.

  python3 -m lol_kills.skills.slip_view --help
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable

from lol_kills.board import build_board
from lol_kills.econ import edge_pp, ev_per_unit, grade_bet, p_under_normal
from lol_kills.etl.aliases import normalize_champ


def _parse_champs(s: str) -> list[str]:
    return [normalize_champ(c.strip()) for c in s.split(",") if c.strip()]


def _parse_ml(s: str | None) -> dict[str, float]:
    if not s:
        return {}
    out: dict[str, float] = {}
    for part in s.split(","):
        if ":" not in part:
            continue
        name, odds = part.rsplit(":", 1)
        out[name.strip()] = float(odds)
    return out


def _parse_kills(s: str | None) -> list[tuple[float, float, float]]:
    """'29.5:1.87/1.87' → (line, over, under) — Over first to match board.build_board lines."""
    if not s:
        return []
    rows: list[tuple[float, float, float]] = []
    for part in s.split(","):
        if ":" not in part or "/" not in part:
            continue
        line_s, odds_s = part.split(":", 1)
        over_s, under_s = odds_s.split("/", 1)
        rows.append((float(line_s), float(over_s), float(under_s)))
    return rows


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in {"1", "true", "yes", "y"}


def ladder_thresholds(p: float) -> dict[str, float]:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return {
        "fair": round(1.0 / p, 2),
        "small": round(1.05 / p, 2),
        "yes": round(1.10 / p, 2),
        "punch": round(1.20 / p, 2),
    }


def kills_buckets(mu: float, sd: float) -> dict[str, float]:
    """Discrete-ish buckets via normal CDF on half-lines."""
    def cdf_le(x: float) -> float:
        # P(total <= x) ≈ P(under x+0.5)
        return p_under_normal(mu, sd, x + 0.5)

    p26 = cdf_le(26)
    p29 = cdf_le(29)
    p32 = cdf_le(32)
    return {
        "le_26": round(p26, 4),
        "27_29": round(max(0.0, p29 - p26), 4),
        "30_32": round(max(0.0, p32 - p29), 4),
        "ge_33": round(max(0.0, 1.0 - p32), 4),
        "median": round(mu, 1),  # normal median = mean
    }


def _fmt_pct(p: float) -> str:
    return f"{100.0 * p:.1f}%"


def render_1view(
    *,
    team1: str,
    team2: str,
    league: str,
    map_n: int | None,
    sheet: dict,
    ml_odds: dict[str, float],
    kill_lines: list[tuple[float, float, float]],
    locked: list[tuple[str, str, float, float]] | None = None,
) -> str:
    """locked: list of (market, selection, odds, stake)."""
    w = sheet["winner"]
    p1, p2 = float(w["p_team1"]), float(w["p_team2"])
    fav, dog = (team1, team2) if p1 >= p2 else (team2, team1)
    p_fav, p_dog = (p1, p2) if p1 >= p2 else (p2, p1)
    fair_fav, fair_dog = 1.0 / p_fav, 1.0 / p_dog

    book_fav = ml_odds.get(fav)
    book_dog = ml_odds.get(dog)
    edge_fav = edge_pp(p_fav, book_fav) if book_fav else None
    grade_fav = grade_bet(p_fav, book_fav)["grade"] if book_fav else "—"

    kills = sheet["kills"]
    mu, sd = float(kills["mean"]), float(kills["sd"])
    buckets = kills_buckets(mu, sd)

    main_line = 29.5
    if kill_lines:
        main_line = min(kill_lines, key=lambda t: abs(t[0] - 29.5))[0]
    pu = p_under_normal(mu, sd, main_line)
    po = 1.0 - pu
    fu, fo = 1.0 / pu, 1.0 / po
    lad_fav = ladder_thresholds(p_fav)
    lad_u = ladder_thresholds(pu)
    lad_o = ladder_thresholds(po)

    title = f"{team1} vs {team2}"
    if map_n is not None:
        title += f" · Map {map_n}"
    title += f" · {league}"

    lines: list[str] = []
    lines.append(f"### 1-view · {title}")
    lines.append("")
    fav_book = f"{book_fav:.2f}" if book_fav else "—"
    edge_s = f"{edge_fav:+.1f}pp" if edge_fav is not None else "—"
    lines.append(
        f"**Favorite:** {fav} **{_fmt_pct(p_fav)}** · fair **{fair_fav:.2f}** · "
        f"book **{fav_book}** · edge **{edge_s}** · **{grade_fav}**"
    )
    dog_book = f"{book_dog:.2f}" if book_dog else "—"
    lines.append(f"**Underdog:** {dog} **{_fmt_pct(p_dog)}** · fair **{fair_dog:.2f}** · book **{dog_book}**")
    lines.append("")
    lines.append("#### Kills")
    lines.append(f"μ **{mu:.1f}** · median ≈ **{buckets['median']}** · sd **{sd:.1f}**")
    lines.append(
        "Buckets: "
        f"≤26 {_fmt_pct(buckets['le_26'])} · "
        f"27–29 {_fmt_pct(buckets['27_29'])} · "
        f"30–32 {_fmt_pct(buckets['30_32'])} · "
        f"33+ {_fmt_pct(buckets['ge_33'])}"
    )
    lines.append(
        f"Main line {main_line:g}: Under fair **{fu:.2f}** ({_fmt_pct(pu)}) · "
        f"Over fair **{fo:.2f}** ({_fmt_pct(po)})"
    )
    lines.append("")
    lines.append("#### Ladder (odds move — use this)")
    lines.append(
        f"**Map {fav}:** skip < {lad_fav['fair']:.2f} · small ≥ {lad_fav['small']:.2f} · "
        f"**yes ≥ {lad_fav['yes']:.2f}** · punch ≥ {lad_fav['punch']:.2f}"
    )
    lines.append(
        f"**Under {main_line:g}:** skip < {lad_u['fair']:.2f} · "
        f"**yes ≥ {lad_u['yes']:.2f}** · punch ≥ {lad_u['punch']:.2f}"
    )
    lines.append(
        f"**Over {main_line:g}:** skip < {lad_o['fair']:.2f} · "
        f"**yes ≥ {lad_o['yes']:.2f}** · punch ≥ {lad_o['punch']:.2f}"
    )
    lines.append("")
    lines.append("#### vs book now")
    lines.append("| Market | Book | Fair | Edge | Grade | Action |")
    lines.append("|--------|------|------|------|-------|--------|")

    # Winner rows
    for team, p in ((team1, p1), (team2, p2)):
        odds = ml_odds.get(team)
        if not odds:
            continue
        g = grade_bet(p, odds)
        ev = ev_per_unit(p, odds)
        action = "TAKE" if ev > 0.08 else ("TINY" if ev > 0 else "SKIP")
        lines.append(
            f"| Winner {team} | {odds:.2f} | {1/p:.2f} | {edge_pp(p, odds):+.1f}pp | "
            f"{g['grade']} | {action} |"
        )

    for line, o_over, o_under in kill_lines:
        pu_l = p_under_normal(mu, sd, line)
        for sel, p, odds in (
            (f"Under {line:g}", pu_l, o_under),
            (f"Over {line:g}", 1.0 - pu_l, o_over),
        ):
            g = grade_bet(p, odds)
            ev = ev_per_unit(p, odds)
            action = "TAKE" if ev > 0.08 else ("TINY" if ev > 0 else "SKIP")
            lines.append(
                f"| {sel} | {odds:.2f} | {1/p:.2f} | {edge_pp(p, odds):+.1f}pp | "
                f"{g['grade']} | {action} |"
            )

    # Action sentence
    take = []
    if book_fav and ev_per_unit(p_fav, book_fav) > 0.08:
        take.append(f"{fav} map @ {book_fav:.2f}")
    for line, o_over, o_under in kill_lines:
        pu_l = p_under_normal(mu, sd, line)
        if ev_per_unit(pu_l, o_under) > 0.08:
            take.append(f"Under {line:g} @ {o_under:.2f}")
        if ev_per_unit(1.0 - pu_l, o_over) > 0.08:
            take.append(f"Over {line:g} @ {o_over:.2f}")

    lines.append("")
    if take:
        lines.append(f"**Action:** TAKE → {'; '.join(take)}.")
    else:
        lines.append("**Action:** SKIP kills / thin map — wait for ladder thresholds.")

    if locked:
        lines.append("")
        for market, sel, odds, stake in locked:
            # find p
            p = None
            if "winner" in market.lower() or "vencedor" in market.lower() or market == "Winner":
                p = p1 if sel == team1 else p2 if sel == team2 else None
            if p is None:
                continue
            g = grade_bet(p, odds)
            lines.append(
                f"**Locked:** {sel} @ {odds:.2f} · R${stake:.0f} → R${stake*odds:.0f} · "
                f"grade **{g['grade']}** · EV R${ev_per_unit(p, odds)*stake:+.2f}"
            )

    return "\n".join(lines)


def main(argv: Iterable[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--team1", required=True)
    ap.add_argument("--team2", required=True)
    ap.add_argument("--league", default="EWC")
    ap.add_argument("--map", type=int, default=None, dest="map_n")
    ap.add_argument("--blue", required=True, help="Comma-separated blue champs")
    ap.add_argument("--red", required=True, help="Comma-separated red champs")
    ap.add_argument("--team1-is-blue", default="true", help="true/false")
    ap.add_argument("--ml", default=None, help="Team:odds,Team:odds")
    ap.add_argument(
        "--kills",
        default=None,
        help="line:over/under,...  e.g. 29.5:1.87/1.87",
    )
    ap.add_argument(
        "--locked",
        default=None,
        help="selection:odds:stake  (assumes map winner)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    blue = _parse_champs(args.blue)
    red = _parse_champs(args.red)
    ml = _parse_ml(args.ml)
    kills = _parse_kills(args.kills)
    t1_blue = _parse_bool(args.team1_is_blue)

    sheet = build_board(
        args.team1,
        args.team2,
        league=args.league,
        blue=blue,
        red=red,
        team1_is_blue=t1_blue,
        lines=kills or None,
        ml_odds=ml or None,
        journal=False,
    )

    locked = None
    if args.locked:
        # selection:odds:stake
        parts = args.locked.split(":")
        if len(parts) >= 2:
            sel = parts[0].strip()
            odds = float(parts[1])
            stake = float(parts[2]) if len(parts) > 2 else 0.0
            locked = [("Winner", sel, odds, stake)]

    print(
        render_1view(
            team1=args.team1,
            team2=args.team2,
            league=args.league,
            map_n=args.map_n,
            sheet=sheet,
            ml_odds=ml,
            kill_lines=kills,
            locked=locked,
        )
    )


if __name__ == "__main__":
    main()
