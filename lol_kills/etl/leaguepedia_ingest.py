"""Normalize existing Leaguepedia caches into warehouse parquet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.paths import LP_GAMES, LP_PLAYERS, PARQUET_DIR


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def ingest_leaguepedia(
    games_path: Path = LP_GAMES,
    players_path: Path = LP_PLAYERS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one-row-per-side team games + player rows from draft_* JSON.
    """
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    if not games_path.exists():
        raise FileNotFoundError(
            f"Missing {games_path}. Run: python -m lol_kills.fetch_drafts && "
            "python -m lol_kills.enrich_games"
        )

    gblob = _load_json(games_path)
    games = gblob["games"] if isinstance(gblob, dict) else gblob
    pblob = _load_json(players_path) if players_path.exists() else {"players": []}
    players = pblob["players"] if isinstance(pblob, dict) else pblob

    team_rows: list[dict] = []
    for g in games:
        gid = g["game_id"]
        date = pd.to_datetime(g.get("date"), errors="coerce")
        league = g.get("league")
        t1 = normalize_team(g.get("team1", ""))
        t2 = normalize_team(g.get("team2", ""))
        k1, k2 = g.get("kills1"), g.get("kills2")
        winner = g.get("winner")  # 1 or 2
        length = g.get("length_min")
        # Leaguepedia team1 is typically blue; treat as blue/red
        for side, team, kills, opp_kills, side_idx in (
            ("Blue", t1, k1, k2, 1),
            ("Red", t2, k2, k1, 2),
        ):
            result = None
            if winner in (1, 2):
                result = 1 if winner == side_idx else 0
            first_inhib = g.get("first_inhib_side")
            first_blood = g.get("first_blood_side")
            team_rows.append(
                {
                    "game_uid": gid,
                    "lp_game_id": gid,
                    "date": date,
                    "league": league,
                    "tournament": g.get("tournament"),
                    "patch": None,
                    "side": side,
                    "teamname": team,
                    "opp_teamname": t2 if side_idx == 1 else t1,
                    "result": result,
                    "gamelength": float(length) * 60 if length else None,
                    "length_min": length,
                    "kills": kills,
                    "opp_kills": opp_kills,
                    "teamkills": kills,
                    "total_kills": g.get("total_kills"),
                    "towers": g.get("towers1") if side_idx == 1 else g.get("towers2"),
                    "opp_towers": g.get("towers2") if side_idx == 1 else g.get("towers1"),
                    "inhibitors": g.get("inhibs1") if side_idx == 1 else g.get("inhibs2"),
                    "opp_inhibitors": g.get("inhibs2") if side_idx == 1 else g.get("inhibs1"),
                    "dragons": g.get("dragons1") if side_idx == 1 else g.get("dragons2"),
                    "barons": g.get("barons1") if side_idx == 1 else g.get("barons2"),
                    "gold": g.get("gold1") if side_idx == 1 else g.get("gold2"),
                    "firstblood": (
                        1
                        if first_blood == side_idx
                        else (0 if first_blood in (1, 2) else None)
                    ),
                    "first_inhib": (
                        1
                        if first_inhib == side_idx
                        else (0 if first_inhib in (1, 2) else None)
                    ),
                    "ckpm": (
                        (g.get("total_kills") / length)
                        if (g.get("total_kills") and length)
                        else None
                    ),
                    "source": "leaguepedia",
                }
            )

    player_rows: list[dict] = []
    for p in players:
        player_rows.append(
            {
                "game_uid": p.get("game_id"),
                "lp_game_id": p.get("game_id"),
                "date": pd.to_datetime(p.get("date"), errors="coerce"),
                "teamname": normalize_team(p.get("team", "")),
                "champion": normalize_champ(p.get("champion", "")),
                "position": p.get("role"),
                "side": None,
                "source": "leaguepedia",
            }
        )

    team_df = pd.DataFrame(team_rows)
    player_df = pd.DataFrame(player_rows)

    if not team_df.empty and not player_df.empty:
        side_map = team_df.set_index(["lp_game_id", "teamname"])["side"].to_dict()
        player_df["side"] = [
            side_map.get((gid, team))
            for gid, team in zip(player_df["lp_game_id"], player_df["teamname"])
        ]

    team_path = PARQUET_DIR / "lp_team_games.parquet"
    player_path = PARQUET_DIR / "lp_player_games.parquet"
    team_df.to_parquet(team_path, index=False)
    player_df.to_parquet(player_path, index=False)
    print(
        f"[lp] wrote {team_path.name} sides={len(team_df)} "
        f"games={team_df['lp_game_id'].nunique() if len(team_df) else 0} "
        f"players={len(player_df)}"
    )
    return team_df, player_df


if __name__ == "__main__":
    ingest_leaguepedia()
