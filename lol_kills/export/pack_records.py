"""Windowed public team/player records built from canonical map rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lol_kills.etl.competition import canonicalize_competition_frame, team_identity_key


def _wr(wins: int, games: int) -> float | None:
    return round(wins / games, 4) if games else None


def _team_rows(maps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in maps.iterrows():
        y = pd.to_numeric(row.get("y_blue_win"), errors="coerce")
        if pd.isna(y):
            continue
        source_league = str(row.get("league_source") or row.get("league") or "")
        league = str(row.get("league") or "UNKNOWN")
        intl = bool(row.get("is_international", False))
        for side, team in (("blue", row.get("blue_team")), ("red", row.get("red_team"))):
            if not team or pd.isna(team):
                continue
            win = float(y) if side == "blue" else 1.0 - float(y)
            rows.append(
                {
                    "team": str(team),
                    "team_key": team_identity_key(team),
                    "league": league,
                    "league_source": source_league,
                    "is_international": intl,
                    "win": win,
                }
            )
    return pd.DataFrame(rows)


def build_team_records(maps: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Aggregate one record per canonical team identity for one pack window."""

    if maps is None or maps.empty:
        return {}
    frame = canonicalize_competition_frame(maps)
    rows = _team_rows(frame)
    if rows.empty:
        return {}

    records: dict[str, dict[str, Any]] = {}
    for key, group in rows.groupby("team_key", sort=True):
        names = group["team"].value_counts()
        display = str(names.index[0])
        by_league: dict[str, dict[str, Any]] = {}
        for league, lg in group.groupby("league", sort=True):
            wins = int(round(float(lg["win"].sum())))
            games = int(len(lg))
            by_league[str(league)] = {"wins": wins, "games": games, "wr": _wr(wins, games)}

        regional = group[~group["is_international"]]
        if regional.empty:
            primary = str(group["league"].value_counts().index[0])
        else:
            primary = str(regional["league"].value_counts().index[0])
        wins = int(round(float(group["win"].sum())))
        games = int(len(group))
        records[display] = {
            "team_key": key,
            "leagues": sorted(str(x) for x in group["league"].unique()),
            "source_leagues": sorted(str(x) for x in group["league_source"].unique() if x),
            "primary": primary,
            "intl": bool(group["is_international"].any()),
            "wins": wins,
            "games": games,
            "wr": _wr(wins, games),
            "by_league": by_league,
        }
  return records

def build_player_records(players: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Aggregate player results from one pack window without team aggregate rows."""

    if players is None or players.empty or "playername" not in players.columns:
        return {}
    frame = canonicalize_competition_frame(players)
    frame = frame[frame["playername"].notna()].copy()
    if "position" in frame.columns:
        frame = frame[frame["position"].astype(str).str.lower().ne("team")]
    if frame.empty:
        return {}
    frame["result"] = pd.to_numeric(frame.get("result"), errors="coerce")
    frame = frame.dropna(subset=["result"])
    if frame.empty:
        return {}

    records: dict[str, dict[str, Any]] = {}
    for player, group in frame.groupby(frame["playername"].astype(str), sort=True):
        wins = int(round(float(group["result"].sum())))
        games = int(len(group))
        leagues = sorted(str(x) for x in group["league"].dropna().unique())
        regional = group[~group["is_international"]]
        primary = str(regional["league"].value_counts().index[0]) if not regional.empty else (leagues[0] if leagues else None)
        records[player] = {
            "wins": wins,
            "games": games,
            "wr": _wr(wins, games),
            "leagues": leagues,
            "primary": primary,
            "intl": bool(group["is_international"].any()),
        }
    return records
