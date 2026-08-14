#!/usr/bin/env python3
"""Fetch ScoreboardGames + ScoreboardPlayers for major regions (draft model)."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from pathlib import Path

from lol_kills.net import require_https_url

ROOT = Path(__file__).resolve().parents[1]
OUT_GAMES = ROOT / "data" / "lol" / "draft_games.json"
OUT_PLAYERS = ROOT / "data" / "lol" / "draft_players.json"
UA = "parlay-risk-sim/lol-kills-draft"

REGIONS = {
    "LCK": ["%LCK%"],
    "LCS": ["%LCS%"],
    "LEC": ["%LEC%"],
    "LPL": ["%LPL%"],
    "CBLOL": ["%CBLOL%"],
}

SKIP_TOURNAMENT = (
    "Challengers",
    "LCK CL",
    "LCKC",
    "LCK AS",
    "NACL",
    "Academy",
    "LDL",
    "Fan Clash",
    "Scholars",
    "EMEA Masters",
    "Eméa Masters",
)


def cargo_export(tables: str, fields: str, where: str, offset: int = 0, limit: int = 500) -> list[dict]:
    params = {
        "tables": tables,
        "fields": fields,
        "where": where,
        "order_by": f"{tables.split('=')[0]}.DateTime_UTC DESC"
        if "DateTime" in fields or "DateTime_UTC" in fields
        else None,
        "limit": str(limit),
        "offset": str(offset),
        "format": "json",
    }
    params = {k: v for k, v in params.items() if v is not None}
    # Fix order_by for ScoreboardGames
    if tables == "ScoreboardGames":
        params["order_by"] = "ScoreboardGames.DateTime_UTC DESC"
    elif tables == "ScoreboardPlayers":
        params["order_by"] = "ScoreboardPlayers.DateTime_UTC DESC"
    url = "https://lol.fandom.com/wiki/Special:CargoExport?" + urllib.parse.urlencode(params)
    url = require_https_url(url, hosts={"lol.fandom.com"})
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=120) as resp:
        raw = resp.read().decode()
    if not raw.strip().startswith("["):
        raise RuntimeError(raw[:300])
    return json.loads(raw)


def league_of(tournament: str) -> str | None:
    t = tournament or ""
    if any(s in t for s in SKIP_TOURNAMENT):
        return None
    if re.search(r"\bLCK\b", t) or t.startswith("LCK"):
        return "LCK"
    if re.search(r"\bLEC\b", t) or t.startswith("LEC"):
        return "LEC"
    if re.search(r"\bLCS\b", t) or t.startswith("LCS"):
        return "LCS"
    if re.search(r"\bLPL\b", t) or t.startswith("LPL"):
        return "LPL"
    if re.search(r"\bCBLOL\b", t) or t.startswith("CBLOL"):
        return "CBLOL"
    return None


def normalize_team(name: str) -> str:
    if not name:
        return name
    if " (" in name and name.endswith(")"):
        return name.split(" (", 1)[0].strip()
    return name.strip()


def fetch_region_games(region: str, since: str, min_games: int, sleep_s: float) -> list[dict]:
    patterns = REGIONS[region]
    like = " OR ".join(f'ScoreboardGames.Tournament LIKE "{p}"' for p in patterns)
    where = f'ScoreboardGames.DateTime_UTC >= "{since}" AND ({like})'
    rows: list[dict] = []
    offset = 0
    while len(rows) < max(min_games * 4, 800):  # keep fetching until enough cleaned
        chunk = cargo_export(
            "ScoreboardGames",
            "ScoreboardGames.GameId,ScoreboardGames.Team1,ScoreboardGames.Team2,"
            "ScoreboardGames.Team1Kills,ScoreboardGames.Team2Kills,"
            "ScoreboardGames.Tournament,ScoreboardGames.DateTime_UTC,"
            "ScoreboardGames.Gamelength_Number",
            where,
            offset=offset,
            limit=500,
        )
        if not chunk:
            break
        rows.extend(chunk)
        print(f"  {region} games offset={offset} +{len(chunk)} raw={len(rows)}")
        if len(chunk) < 500:
            break
        offset += 500
        time.sleep(sleep_s)
    out = []
    seen = set()
    for g in rows:
        lg = league_of(g.get("Tournament") or "")
        if lg != region:
            continue
        gid = g.get("GameId")
        if not gid or gid in seen:
            continue
        k1, k2 = g.get("Team1Kills"), g.get("Team2Kills")
        if k1 is None or k2 is None:
            continue
        try:
            k1, k2 = int(k1), int(k2)
        except (TypeError, ValueError):
            continue
        seen.add(gid)
        date = g.get("DateTime UTC") or g.get("DateTime_UTC") or ""
        length = g.get("Gamelength Number") or g.get("Gamelength_Number")
        try:
            length_f = float(length) if length is not None else None
        except (TypeError, ValueError):
            length_f = None
        out.append(
            {
                "game_id": gid,
                "league": region,
                "date": date[:19],
                "tournament": g.get("Tournament") or "",
                "team1": normalize_team(g.get("Team1") or ""),
                "team2": normalize_team(g.get("Team2") or ""),
                "kills1": k1,
                "kills2": k2,
                "total_kills": k1 + k2,
                "length_min": length_f,
            }
        )
        if len(out) >= max(min_games, 300):
            break
    return out


def fetch_players_for_game_ids(game_ids: list[str], sleep_s: float) -> list[dict]:
    """Batch GameId IN (...) queries (Cargo limits IN size — use chunks of 40)."""
    out: list[dict] = []
    chunk_size = 35
    for i in range(0, len(game_ids), chunk_size):
        batch = game_ids[i : i + chunk_size]
        # escape quotes in ids
        clauses = []
        for gid in batch:
            safe = gid.replace('"', '\\"')
            clauses.append(f'ScoreboardPlayers.GameId="{safe}"')
        where = "(" + " OR ".join(clauses) + ")"
        offset = 0
        while True:
            chunk = cargo_export(
                "ScoreboardPlayers",
                "ScoreboardPlayers.GameId,ScoreboardPlayers.Team,"
                "ScoreboardPlayers.Champion,ScoreboardPlayers.Role,"
                "ScoreboardPlayers.DateTime_UTC",
                where,
                offset=offset,
                limit=500,
            )
            if not chunk:
                break
            for p in chunk:
                champ = p.get("Champion")
                role = p.get("Role")
                gid = p.get("GameId")
                if not champ or not gid:
                    continue
                out.append(
                    {
                        "game_id": gid,
                        "team": normalize_team(p.get("Team") or ""),
                        "champion": champ,
                        "role": role or "",
                        "date": (p.get("DateTime UTC") or p.get("DateTime_UTC") or "")[:19],
                    }
                )
            if len(chunk) < 500:
                break
            offset += 500
            time.sleep(sleep_s)
        print(f"  players batch {i//chunk_size+1}/{(len(game_ids)-1)//chunk_size+1} rows={len(out)}")
        time.sleep(sleep_s)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--since", default="2024-06-01 00:00:00")
    ap.add_argument("--min-per-region", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.4)
    ap.add_argument("--regions", default="LCK,LCS,LEC,LPL,CBLOL")
    args = ap.parse_args()

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    all_games: list[dict] = []
    for region in regions:
        print(f"Fetching games {region}...")
        gs = fetch_region_games(region, args.since, args.min_per_region, args.sleep)
        print(f"  kept {len(gs)} {region} games")
        if len(gs) < args.min_per_region:
            print(f"  WARNING: {region} only {len(gs)} < {args.min_per_region}")
        all_games.extend(gs)
        time.sleep(args.sleep)

    OUT_GAMES.parent.mkdir(parents=True, exist_ok=True)
    payload_g = {
        "source": "Leaguepedia Cargo ScoreboardGames",
        "since": args.since,
        "n_games": len(all_games),
        "by_league": {r: sum(1 for g in all_games if g["league"] == r) for r in regions},
        "games": all_games,
    }
    OUT_GAMES.write_text(json.dumps(payload_g, indent=2))
    print(f"wrote {OUT_GAMES} n={len(all_games)} {payload_g['by_league']}")

    gids = [g["game_id"] for g in all_games]
    print(f"Fetching players for {len(gids)} games...")
    players = fetch_players_for_game_ids(gids, args.sleep)
    payload_p = {
        "source": "Leaguepedia Cargo ScoreboardPlayers",
        "n_rows": len(players),
        "n_games_with_rows": len({p["game_id"] for p in players}),
        "players": players,
    }
    OUT_PLAYERS.write_text(json.dumps(payload_p, indent=2))
    print(f"wrote {OUT_PLAYERS} rows={len(players)} games={payload_p['n_games_with_rows']}")


if __name__ == "__main__":
    main()
