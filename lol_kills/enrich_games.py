#!/usr/bin/env python3
"""Enrich draft_games.json with Winner / inhibitors / towers from Leaguepedia."""

from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from lol_kills.net import require_https_url

ROOT = Path(__file__).resolve().parents[1]
GAMES = ROOT / "data" / "lol" / "draft_games.json"
UA = "parlay-risk-sim/lol-markets"


def cargo_by_ids(game_ids: list[str], sleep_s: float = 0.35) -> dict[str, dict]:
    out: dict[str, dict] = {}
    chunk_size = 30
    fields = (
        "ScoreboardGames.GameId,ScoreboardGames.Winner,"
        "ScoreboardGames.Team1Kills,ScoreboardGames.Team2Kills,"
        "ScoreboardGames.Team1Towers,ScoreboardGames.Team2Towers,"
        "ScoreboardGames.Team1Inhibitors,ScoreboardGames.Team2Inhibitors,"
        "ScoreboardGames.Team1Barons,ScoreboardGames.Team2Barons,"
        "ScoreboardGames.Team1Dragons,ScoreboardGames.Team2Dragons,"
        "ScoreboardGames.Team1Gold,ScoreboardGames.Team2Gold"
    )
    for i in range(0, len(game_ids), chunk_size):
        batch = game_ids[i : i + chunk_size]
        clauses = [f'ScoreboardGames.GameId="{gid.replace(chr(34), "")}"' for gid in batch]
        where = "(" + " OR ".join(clauses) + ")"
        params = {
            "tables": "ScoreboardGames",
            "fields": fields,
            "where": where,
            "limit": "500",
            "format": "json",
        }
        url = "https://lol.fandom.com/wiki/Special:CargoExport?" + urllib.parse.urlencode(params)
        url = require_https_url(url, hosts={"lol.fandom.com"})
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
        if not raw.strip().startswith("["):
            raise RuntimeError(raw[:240])
        for row in json.loads(raw):
            gid = row.get("GameId")
            if not gid:
                continue

            def _i(key: str):
                v = row.get(key)
                try:
                    return int(v) if v is not None and v != "" else None
                except (TypeError, ValueError):
                    return None

            out[gid] = {
                "winner": _i("Winner"),  # 1 or 2
                "towers1": _i("Team1Towers"),
                "towers2": _i("Team2Towers"),
                "inhibs1": _i("Team1Inhibitors"),
                "inhibs2": _i("Team2Inhibitors"),
                "barons1": _i("Team1Barons"),
                "barons2": _i("Team2Barons"),
                "dragons1": _i("Team1Dragons"),
                "dragons2": _i("Team2Dragons"),
                "gold1": _i("Team1Gold"),
                "gold2": _i("Team2Gold"),
            }
        print(f"enrich batch {i//chunk_size+1}/{(len(game_ids)-1)//chunk_size+1} have={len(out)}")
        time.sleep(sleep_s)
    return out


def first_inhib_side(inhibs1: int | None, inhibs2: int | None) -> int | None:
    """Proxy: which side took first inhib (1/2). Asymmetric counts → that side; else None."""
    if inhibs1 is None or inhibs2 is None:
        return None
    if inhibs1 > 0 and inhibs2 == 0:
        return 1
    if inhibs2 > 0 and inhibs1 == 0:
        return 2
    if inhibs1 > inhibs2:
        return 1
    if inhibs2 > inhibs1:
        return 2
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sleep", type=float, default=0.35)
    args = ap.parse_args()
    payload = json.loads(GAMES.read_text())
    games = payload["games"]
    meta = cargo_by_ids([g["game_id"] for g in games], sleep_s=args.sleep)
    n_ok = 0
    for g in games:
        m = meta.get(g["game_id"], {})
        g.update(m)
        fi = first_inhib_side(g.get("inhibs1"), g.get("inhibs2"))
        g["first_inhib_side"] = fi
        # first blood not in Cargo — leave null; filled by proxy model later
        g["first_blood_side"] = None
        if g.get("winner") in (1, 2):
            n_ok += 1
    payload["enriched"] = True
    payload["n_with_winner"] = n_ok
    GAMES.write_text(json.dumps(payload, indent=2))
    print(f"enriched {GAMES} winners={n_ok}/{len(games)}")


if __name__ == "__main__":
    main()
