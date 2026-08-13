#!/usr/bin/env python3
"""Fetch ScoreboardGames for LCK / LEC / LCS from Leaguepedia Cargo."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from lol_kills.net import require_https_url

DEFAULT_SINCE = "2024-01-01 00:00:00"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "lol" / "games_raw.json"
UA = "parlay-risk-sim/lol-kills (local research)"


def cargo_export(where: str, offset: int = 0, limit: int = 500) -> list[dict]:
    params = {
        "tables": "ScoreboardGames",
        "fields": (
            "ScoreboardGames.Team1,ScoreboardGames.Team2,"
            "ScoreboardGames.Team1Kills,ScoreboardGames.Team2Kills,"
            "ScoreboardGames.DateTime_UTC,ScoreboardGames.Tournament,"
            "ScoreboardGames.Gamelength_Number,ScoreboardGames.Winner"
        ),
        "where": where,
        "order_by": "ScoreboardGames.DateTime_UTC DESC",
        "limit": str(limit),
        "offset": str(offset),
        "format": "json",
    }
    url = "https://lol.fandom.com/wiki/Special:CargoExport?" + urllib.parse.urlencode(params)
    url = require_https_url(url, hosts={"lol.fandom.com"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=90) as resp:
        return json.loads(resp.read().decode())


def league_of(tournament: str) -> str | None:
    t = tournament or ""
    # Skip academies / challengers / secondary circuits
    skip_tokens = (
        "Challengers",
        "LCK CL",
        "LCKC",
        "LCK AS",  # Academy Series
        "LCS Challengers",
        "NACL",
        "Academy",
        "Eméa Masters",
        "EMEA Masters",
        "LFL",
        "LVP",
        "CBLOL",
        "Scholars",
        "Fan Clash",
    )
    if any(s in t for s in skip_tokens):
        return None
    # Major leagues — check LCK before LCS (no overlap), LEC vs LCS carefully
    if re.search(r"\bLCK\b", t) or t.startswith("LCK"):
        return "LCK"
    if re.search(r"\bLEC\b", t) or t.startswith("LEC"):
        return "LEC"
    if re.search(r"\bLCS\b", t) or t.startswith("LCS"):
        return "LCS"
    # Keep international for H2H enrichment; tagged separately
    if any(x in t for x in ("Worlds", "Mid-Season Invitational", "MSI ", "MSI/", "Esports World Cup", "EWC")):
        return "INT"
    return None


# Canonical name merges (Leaguepedia renames / casing)
TEAM_CANON = {
    "Dplus KIA": "Dplus Kia",
    "OKSavingsBank BRION": "BRION",
    "HANJIN BRION": "BRION",
    "OKSavingsBank Brion": "BRION",
    "DN SOOPers": "DN Freecs",
    "Kiwoom DRX": "DRX",
}


def normalize_team(name: str) -> str:
    """Strip Leaguepedia disambiguators like '(2024 American Team)'."""
    if not name:
        return name
    if " (" in name and name.endswith(")"):
        name = name.split(" (", 1)[0].strip()
    else:
        name = name.strip()
    return TEAM_CANON.get(name, name)


def fetch_all(since: str = DEFAULT_SINCE, sleep_s: float = 0.35) -> list[dict]:
    where = (
        f'ScoreboardGames.DateTime_UTC >= "{since}" AND '
        '('
        'ScoreboardGames.Tournament LIKE "%LCK%" OR '
        'ScoreboardGames.Tournament LIKE "%LEC%" OR '
        'ScoreboardGames.Tournament LIKE "%LCS%" OR '
        'ScoreboardGames.Tournament LIKE "%Worlds%" OR '
        'ScoreboardGames.Tournament LIKE "%MSI%" OR '
        'ScoreboardGames.Tournament LIKE "%Esports World Cup%"'
        ')'
    )
    rows: list[dict] = []
    offset = 0
    while True:
        chunk = cargo_export(where, offset=offset, limit=500)
        if not chunk:
            break
        rows.extend(chunk)
        print(f"fetched offset={offset} +{len(chunk)} total={len(rows)}")
        if len(chunk) < 500:
            break
        offset += 500
        time.sleep(sleep_s)
    return rows


ACADEMY_TEAM_TOKENS = ("Academy", "Scholars", "Youth", "Rookies", " Challengers")


def clean_games(raw: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[tuple] = set()
    for g in raw:
        league = league_of(g.get("Tournament") or "")
        if not league:
            continue
        k1, k2 = g.get("Team1Kills"), g.get("Team2Kills")
        if k1 is None or k2 is None:
            continue
        try:
            k1, k2 = int(k1), int(k2)
        except (TypeError, ValueError):
            continue
        t1 = normalize_team(g.get("Team1") or "")
        t2 = normalize_team(g.get("Team2") or "")
        if not t1 or not t2:
            continue
        if any(tok in t1 or tok in t2 for tok in ACADEMY_TEAM_TOKENS):
            continue
        date = g.get("DateTime UTC") or g.get("DateTime_UTC") or ""
        key = (date, t1, t2, k1, k2, g.get("Tournament"))
        if key in seen:
            continue
        seen.add(key)
        length = g.get("Gamelength Number") or g.get("Gamelength_Number")
        try:
            length_f = float(length) if length is not None else None
        except (TypeError, ValueError):
            length_f = None
        out.append(
            {
                "date": date[:19] if date else "",
                "league": league,
                "tournament": g.get("Tournament") or "",
                "team1": t1,
                "team2": t2,
                "kills1": k1,
                "kills2": k2,
                "total_kills": k1 + k2,
                "length_min": length_f,
                "winner": g.get("Winner"),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default=DEFAULT_SINCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    raw = fetch_all(since=args.since)
    games = clean_games(raw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Leaguepedia Cargo ScoreboardGames",
        "since": args.since,
        "fetched_raw": len(raw),
        "games": games,
        "n_games": len(games),
    }
    args.out.write_text(json.dumps(payload, indent=2))
    by_league: dict[str, int] = {}
    for g in games:
        by_league[g["league"]] = by_league.get(g["league"], 0) + 1
    print(f"wrote {args.out} n={len(games)} by_league={by_league}")


if __name__ == "__main__":
    main()
