"""Windowed public team/player records built from canonical map rows."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lol_kills.etl.competition import canonicalize_competition_frame, team_identity_key
from lol_kills.etl.source_keys import canonical_source_game_key


PUBLIC_TEAM_RATING_EXCLUSIONS = frozenset({"los-ratones"})
INVALID_COMPETITION_LABELS = frozenset({"", "UNKNOWN", "ORACLE_ELIXIR_API", "OE_API"})
PUBLIC_ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
PUBLIC_ROLE_ALIASES = {
    "top": "top",
    "jng": "jungle",
    "jungle": "jungle",
    "jungler": "jungle",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "adc": "bot",
    "bottom": "bot",
    "sup": "support",
    "support": "support",
    "utility": "support",
}


def _wr(wins: int, games: int) -> float | None:
    return round(wins / games, 4) if games else None


def build_maps_frame_from_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """Build one canonical map row per OE team-game pair.

    ``maps.parquet`` is intentionally a feature-focused major-event artifact,
    so it cannot be the population for the public team ladder.  The full OE
    team feed has one aggregate row per side for every domestic and
    developmental game; this adapter restores that coverage without pulling
    player rows or feature columns into the rating fit.
    """

    if team_games is None or team_games.empty:
        return pd.DataFrame()
    frame = team_games.copy()
    if "position" in frame.columns:
        frame = frame[frame["position"].astype(str).str.lower().eq("team")]
    if frame.empty or "teamname" not in frame.columns or "side" not in frame.columns:
        return pd.DataFrame()
    if "game_uid" not in frame.columns and "gameid" not in frame.columns:
        return pd.DataFrame()
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        frame["_game_uid"] = [
            canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
            for index, value in frame["game_uid"].items()
        ]
    else:
        frame["_game_uid"] = frame["gameid"].map(canonical_source_game_key)
    frame = frame[frame["_game_uid"].notna() & frame["_game_uid"].str.strip().ne("")]
    frame["_side"] = frame["side"].astype(str).str.title()
    frame = frame[frame["_side"].isin({"Blue", "Red"})]
    blue = frame[frame["_side"].eq("Blue")].drop_duplicates("_game_uid")
    red = frame[frame["_side"].eq("Red")].drop_duplicates("_game_uid")
    if blue.empty or red.empty:
        return pd.DataFrame()

    blue_columns = [c for c in ("_game_uid", "date", "league", "tournament", "result", "teamname") if c in blue.columns]
    red_columns = [c for c in ("_game_uid", "teamname") if c in red.columns]
    maps = blue[blue_columns].rename(
        columns={"_game_uid": "game_uid", "result": "y_blue_win", "teamname": "blue_team"}
    )
    maps = maps.merge(
        red[red_columns].rename(columns={"_game_uid": "game_uid", "teamname": "red_team"}),
        on="game_uid",
        how="inner",
    )
    maps["date"] = pd.to_datetime(maps.get("date"), errors="coerce")
    maps["y_blue_win"] = pd.to_numeric(maps.get("y_blue_win"), errors="coerce")
    maps = maps.dropna(subset=["date", "y_blue_win", "blue_team", "red_team"])
    return canonicalize_competition_frame(maps).sort_values("date").reset_index(drop=True)


def filter_public_team_rating_maps(maps: pd.DataFrame) -> pd.DataFrame:
    """Remove teams that are outside the public team-rating population."""

    if maps is None or maps.empty:
        return maps.copy() if maps is not None else pd.DataFrame()
    blue_column = "blue_team" if "blue_team" in maps.columns else "blue_teamname"
    red_column = "red_team" if "red_team" in maps.columns else "red_teamname"
    if blue_column not in maps.columns or red_column not in maps.columns:
        return maps.copy()
    keep = ~maps[blue_column].map(team_identity_key).isin(PUBLIC_TEAM_RATING_EXCLUSIONS)
    keep &= ~maps[red_column].map(team_identity_key).isin(PUBLIC_TEAM_RATING_EXCLUSIONS)
    return maps.loc[keep].copy()


def _primary_league(group: pd.DataFrame) -> str | None:
    """Return the latest domestic affiliation, with a frequency fallback.

    Cross-region events provide useful evidence but must not overwrite the
    team's domestic league.  A latest-date rule also reflects migrations such
    as LTA South → CBLOL and PCS → LCP in the public label.
    """

    candidates = group[group["competition_tier"].isin({"tier1", "tier2", "tier3"})]
    if candidates.empty:
        candidates = group
    dates = (
        pd.to_datetime(candidates["date"], errors="coerce")
        if "date" in candidates.columns
        else pd.Series(pd.NaT, index=candidates.index)
    )
    if dates.notna().any():
        latest = candidates.loc[dates.idxmax()]
        return str(latest["league"])
    counts = candidates["league"].value_counts()
    return str(counts.index[0]) if not counts.empty else None


def _team_rows(maps: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in maps.iterrows():
        y = pd.to_numeric(row.get("y_blue_win"), errors="coerce")
        if pd.isna(y):
            continue
        source_league = str(row.get("league_source") or row.get("league") or "")
        league = str(row.get("league") or "UNKNOWN")
        intl = bool(row.get("is_international", False))
        scope = str(row.get("competition_scope") or ("international" if intl else "other"))
        interregional = bool(row.get("is_interregional", False))
        tier = str(row.get("competition_tier") or ("international" if intl else "tier3"))
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
                    "competition_scope": scope,
                    "is_interregional": interregional,
                    "competition_tier": tier,
                    "date": row.get("date"),
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

        by_tier: dict[str, dict[str, Any]] = {}
        for tier, tg in group[group["competition_tier"].isin({"tier1", "tier2", "tier3"})].groupby("competition_tier", sort=True):
            wins = int(round(float(tg["win"].sum())))
            games = int(len(tg))
            by_tier[str(tier)] = {"wins": wins, "games": games, "wr": _wr(wins, games)}

        primary = _primary_league(group)
        current = group[group["competition_tier"].isin({"tier1", "tier2", "tier3"})]
        current_row = current.loc[pd.to_datetime(current["date"], errors="coerce").idxmax()] if not current.empty and pd.to_datetime(current["date"], errors="coerce").notna().any() else None
        wins = int(round(float(group["win"].sum())))
        games = int(len(group))
        records[display] = {
            "team_key": key,
            "leagues": sorted(str(x) for x in group["league"].unique()),
            "source_leagues": sorted(str(x) for x in group["league_source"].unique() if x),
            "primary": primary,
            "current_league": str(current_row["league"]) if current_row is not None else primary,
            "current_tier": str(current_row["competition_tier"]) if current_row is not None else None,
            "current_team": display,
            "current_date": str(current_row["date"]) if current_row is not None else None,
            "intl": bool(group["is_international"].any()),
            "interregional": bool(group["is_interregional"].any()),
            "wins": wins,
            "games": games,
            "wr": _wr(wins, games),
            "by_league": by_league,
            "by_tier": by_tier,
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
        valid_league = ~group["league"].astype(str).str.upper().isin(INVALID_COMPETITION_LABELS)
        classified = group[valid_league]
        leagues = sorted(str(x) for x in classified["league"].dropna().unique())
        current = classified[classified["competition_tier"].isin({"tier1", "tier2", "tier3"})]
        current_row = None
        if not current.empty:
            dates = pd.to_datetime(current["date"], errors="coerce") if "date" in current.columns else pd.Series(pd.NaT, index=current.index)
            if dates.notna().any():
                current_row = current.loc[dates.idxmax()]
        latest_valid = None
        if not classified.empty:
            dates = pd.to_datetime(classified["date"], errors="coerce") if "date" in classified.columns else pd.Series(pd.NaT, index=classified.index)
            if dates.notna().any():
                latest_valid = classified.loc[dates.idxmax()]
        observed = group
        observed_dates = pd.to_datetime(observed["date"], errors="coerce") if "date" in observed.columns else pd.Series(pd.NaT, index=observed.index)
        observed_row = observed.loc[observed_dates.idxmax()] if observed_dates.notna().any() else None
        primary = str(current_row["league"]) if current_row is not None else (
            str(latest_valid["league"]) if latest_valid is not None else None
        )
        role_counts: dict[str, int] = {}
        if "position" in group.columns:
            for raw_role, count in group["position"].astype(str).str.lower().value_counts().items():
                role = PUBLIC_ROLE_ALIASES.get(raw_role)
                if role:
                    role_counts[role] = role_counts.get(role, 0) + int(count)
        roles = sorted(
            role_counts,
            key=lambda role: (-role_counts[role], PUBLIC_ROLE_ORDER.index(role)),
        )
        side_values = group["side"].astype(str).str.lower() if "side" in group.columns else pd.Series("", index=group.index)
        blue = group[side_values.eq("blue")]
        red = group[side_values.eq("red")]
        blue_wins = int(round(float(blue["result"].sum())))
        red_wins = int(round(float(red["result"].sum())))
        records[player] = {
            "wins": wins,
            "games": games,
            "wr": _wr(wins, games),
            "leagues": leagues,
            "primary": primary,
            "current_league": primary,
            "current_tier": str(current_row["competition_tier"]) if current_row is not None else None,
            "current_team": str(observed_row["teamname"]) if observed_row is not None and pd.notna(observed_row.get("teamname")) else None,
            "current_date": str(observed_row["date"]) if observed_row is not None else None,
            "intl": bool(group["is_international"].any()),
            "interregional": bool(group.get("is_interregional", pd.Series(dtype=bool)).any()),
            "blue_games": int(len(blue)),
            "blue_wins": blue_wins,
            "blue_wr": _wr(blue_wins, len(blue)),
            "red_games": int(len(red)),
            "red_wins": red_wins,
            "red_wr": _wr(red_wins, len(red)),
            "roles": roles,
            "primary_role": roles[0] if roles else None,
        }
    return records
