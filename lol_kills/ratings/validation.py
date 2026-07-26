"""Input audits and temporal split helpers for rating validation."""

from __future__ import annotations

from typing import Any

import pandas as pd

from lol_kills.etl.competition import (
    DEPRECATED_LEAGUE_MAP,
    REGIONAL_LEAGUES,
    canonicalize_competition_frame,
    team_identity_key,
)


def temporal_cutoffs(
    maps: pd.DataFrame,
    n_splits: int = 3,
    min_train_days: int = 90,
) -> list[pd.Timestamp]:
    """Return chronological cutoffs; never randomize games or split maps."""

    if maps is None or maps.empty or n_splits < 1:
        return []
    dates = pd.to_datetime(maps["date"], errors="coerce").dropna().sort_values().unique()
    if len(dates) < 2:
        return []
    start = pd.Timestamp(dates[0]) + pd.Timedelta(days=min_train_days)
    eligible = [pd.Timestamp(value) for value in dates if pd.Timestamp(value) >= start]
    if not eligible:
        return []
    positions = [round((i + 1) * (len(eligible) - 1) / (n_splits + 1)) for i in range(n_splits)]
    return sorted({eligible[min(max(pos, 0), len(eligible) - 1)] for pos in positions})


def audit_rating_inputs(maps: pd.DataFrame) -> dict[str, Any]:
    """Produce inspectable data-quality diagnostics before fitting a ladder."""

    if maps is None or maps.empty:
        return {"ok": False, "reason": "empty_input"}
    frame = canonicalize_competition_frame(maps)
    dates = pd.to_datetime(frame.get("date"), errors="coerce")
    teams: dict[str, set[str]] = {}
    for column in ("blue_team", "red_team"):
        if column not in frame.columns:
            continue
        for value in frame[column].dropna().astype(str):
            teams.setdefault(team_identity_key(value), set()).add(value)
    display_collisions = {
        key: sorted(values) for key, values in teams.items() if len(values) > 1
    }
    duplicate_game_uids = 0
    if "game_uid" in frame.columns:
        duplicate_game_uids = int(frame["game_uid"].duplicated(keep=False).sum())
    source_leagues = frame.get("league_source", pd.Series(dtype=object)).astype(str)
    deprecated_rows = int(source_leagues.isin(DEPRECATED_LEAGUE_MAP).sum())
    canonical_leagues = sorted(str(value) for value in frame.get("league", pd.Series(dtype=object)).dropna().unique())
    regional_rows = int(frame.get("competition_scope", pd.Series(dtype=object)).eq("regional").sum())
    intl_rows = int(frame.get("is_international", pd.Series(dtype=bool)).fillna(False).sum())
    bridge_pairs: set[tuple[str, str]] = set()
    home: dict[str, str] = {}
    ordered = frame.assign(_date=dates).sort_values("_date")
    for _, row in ordered.iterrows():
        blue = team_identity_key(row.get("blue_team"))
        red = team_identity_key(row.get("red_team"))
        league = str(row.get("league") or "")
        blue_home = home.get(blue, league if league in REGIONAL_LEAGUES else "")
        red_home = home.get(red, league if league in REGIONAL_LEAGUES else "")
        if bool(row.get("is_international", False)) and blue_home in REGIONAL_LEAGUES and red_home in REGIONAL_LEAGUES and blue_home != red_home:
            bridge_pairs.add(tuple(sorted((blue_home, red_home))))
        if league in REGIONAL_LEAGUES:
            home[blue] = league
            home[red] = league
    return {
        "ok": bool(len(frame) > 0 and duplicate_game_uids == 0),
        "n_rows": int(len(frame)),
        "date_min": dates.min().isoformat() if dates.notna().any() else None,
        "date_max": dates.max().isoformat() if dates.notna().any() else None,
        "n_duplicate_game_uid_rows": duplicate_game_uids,
        "n_team_identity_collisions": len(display_collisions),
        "team_identity_collisions": display_collisions,
        "deprecated_source_rows": deprecated_rows,
        "canonical_leagues": canonical_leagues,
        "regional_rows": regional_rows,
        "international_rows": intl_rows,
        "international_bridge_pairs": [list(pair) for pair in sorted(bridge_pairs)],
        "n_international_bridge_pairs": len(bridge_pairs),
        "temporal_cutoffs": [str(value) for value in temporal_cutoffs(frame)],
        "note": "Use chronological series-level holdouts and cluster uncertainty by series/event; random map splits are not valid.",
    }
