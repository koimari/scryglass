"""Per-map contribution statistics for the public player and team profiles.

The rows here are the *same* census the published ratings are fit on: this
module consumes the canonical player archive frame that
:func:`lol_kills.export.pack_records.build_profile_records` reads, so a profile
can never show a map the ratings did not see.

FAIL CLOSED.  Every metric is emitted as ``null`` when its source column is
absent or unusable.  A missing metric is never coerced to ``0``, because a
zero-value CS/min or damage share reads as a real (terrible) performance rather
than as an absent measurement.

Field naming is load bearing.  ``supabase/migrations/
20260815060001_descriptive_draft_query_api.sql`` installs
``public.scryglass_json_has_draft_fields``, which rejects any published asset
carrying a banned key.  ``gold`` is a banned *exact* key and ``strength_``,
``phase_``, ``live_``, ``control_`` and ``r9e_`` are banned prefixes, so the
descriptive names below (``gold_per_min``, ``gold_share_pct``, ...) are the only
safe spellings.  ``tests/test_player_map_stats.py`` reimplements the SQL ban
list and asserts every emitted key against it.

For the same reason ``players`` and ``teams`` are *arrays carrying a ``name``
field* rather than the name-keyed objects the neighbouring artifacts use.  The
guard inspects JSON object **keys**, so a name-keyed map would hand player and
team names to it: a roster with a player called ``Gold`` or an org tagged ``EV``
would silently fail publication with no signal pointing back at the roster. No
current identity collides, but nothing stops one appearing, and a name is data
we do not control.  As values, names are never inspected.
"""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from lol_kills.export.pack_records import (
    PUBLIC_ROLE_ALIASES,
    public_team_affiliation,
)
from lol_kills.etl.source_keys import canonical_source_game_key


SCHEMA_VERSION = "scryglass:player-map-stats:v1"

# Matches ``build_profile_records(recent_window_days=120)`` so the Stats tab and
# the rest of the profile describe the same window of play.
DEFAULT_WINDOW_DAYS = 120

# Per-identity row cap.  Measured against the real census, the 120-day window
# holds a median of 17 maps per player, so a cap of 20 keeps the typical player
# whole while trimming the stacked academy/main-roster tail. The whole artifact
# is loaded and parsed by the profile pages, and an uncapped build measures
# 13.1MB against a ~15MB budget; at 20 it measures 9.5MB.
#
# The aggregate header is computed over exactly the rows that survive this cap,
# so the header always describes the table printed beneath it.
DEFAULT_MAP_LIMIT = 20

_PLAYER_METRIC_KEYS = (
    "cs_per_min",
    "gold_per_min",
    "gold_share_pct",
    "damage_per_min",
    "damage_share_pct",
    "kda",
)

_TEAM_METRIC_KEYS = (
    "gold_per_min",
    "damage_per_min",
    "kills",
    "deaths",
    "game_length_min",
)


def _round(value: Any, digits: int = 2) -> float | None:
    """Round to a JSON-safe float, or fail closed to ``None``."""

    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return round(number, digits)


def _positive(value: float | None) -> float | None:
    """Keep only strictly positive denominators."""

    return value if value is not None and value > 0 else None


def _mean(values: list[float | None]) -> float | None:
    """Mean over the *available* observations only, or ``None``."""

    present = [value for value in values if value is not None]
    if not present:
        return None
    return round(sum(present) / len(present), 2)


def _empty_payload(window_days: int, map_limit: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "window_days": window_days,
        "map_limit": map_limit,
        "players": [],
        "teams": [],
    }


def _prepared_frame(players: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Restrict the archive to complete, windowed, identifiable player maps."""

    frame = players.copy()
    identity_source = frame.get("game_uid", frame.get("gameid"))
    fallback = frame["gameid"] if "gameid" in frame.columns else None
    frame["_game_id"] = [
        canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
        for index, value in identity_source.items()
    ]
    frame["_date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["_side"] = frame["side"].astype(str).str.title()
    frame["_role"] = frame["position"].astype(str).str.casefold().map(PUBLIC_ROLE_ALIASES)
    frame["_player"] = frame["playername"].astype("string").str.strip()
    frame["_team"] = frame["teamname"].astype("string").str.strip()
    frame["_win"] = pd.to_numeric(frame["result"], errors="coerce")
    frame = frame[
        frame["_game_id"].astype(str).str.strip().ne("")
        & frame["_date"].notna()
        & frame["_side"].isin({"Blue", "Red"})
        & frame["_role"].notna()
        & frame["_player"].notna()
        & frame["_player"].ne("")
        & frame["_team"].notna()
        & frame["_team"].ne("")
        & frame["_win"].isin({0, 1})
    ].copy()
    if frame.empty:
        return frame
    latest = frame["_date"].max()
    if pd.notna(latest):
        frame = frame[frame["_date"].ge(latest - pd.Timedelta(days=window_days))].copy()
    return frame


def _side_gold_totals(frame: pd.DataFrame) -> dict[tuple[str, str], float | None]:
    """Team gold per (game, side), or ``None`` when the side is not complete.

    A gold share computed against a partial side is wrong rather than merely
    imprecise, so an incomplete side yields an unavailable share.
    """

    if "totalgold" not in frame.columns:
        return {}
    totals: dict[tuple[str, str], float | None] = {}
    gold = pd.to_numeric(frame["totalgold"], errors="coerce")
    for key, index in frame.groupby(["_game_id", "_side"], sort=False).groups.items():
        values = gold.loc[index]
        if len(values) != 5 or values.isna().any():
            totals[(str(key[0]), str(key[1]))] = None
            continue
        total = float(values.sum())
        totals[(str(key[0]), str(key[1]))] = total if total > 0 else None
    return totals


def _opponents(frame: pd.DataFrame) -> dict[tuple[str, str], str | None]:
    """Map (game, side) to the other side's display team name."""

    by_side: dict[str, dict[str, str]] = {}
    for (game_id, side), group in frame.groupby(["_game_id", "_side"], sort=False):
        names = {str(value) for value in group["_team"].dropna().unique()}
        if len(names) != 1:
            continue
        by_side.setdefault(str(game_id), {})[str(side)] = names.pop()
    opponents: dict[tuple[str, str], str | None] = {}
    for game_id, sides in by_side.items():
        for side in ("Blue", "Red"):
            other = "Red" if side == "Blue" else "Blue"
            opponents[(game_id, side)] = public_team_affiliation(sides.get(other))
    return opponents


def build_player_map_stats(
    players: pd.DataFrame,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    map_limit: int = DEFAULT_MAP_LIMIT,
) -> dict[str, Any]:
    """Build the per-map Stats payload for every windowed player and team."""

    if window_days < 1 or map_limit < 1:
        raise ValueError("window_days and map_limit must be positive")
    required = {"playername", "teamname", "side", "position", "result", "date"}
    if players is None or players.empty or not required.issubset(players.columns):
        return _empty_payload(window_days, map_limit)
    if "gameid" not in players.columns and "game_uid" not in players.columns:
        return _empty_payload(window_days, map_limit)

    frame = _prepared_frame(players, window_days)
    if frame.empty:
        return _empty_payload(window_days, map_limit)

    side_gold = _side_gold_totals(frame)
    opponents = _opponents(frame)

    numeric = {
        column: pd.to_numeric(frame[column], errors="coerce")
        if column in frame.columns
        else pd.Series(index=frame.index, dtype="float64")
        for column in (
            "cspm", "dpm", "damageshare", "totalgold", "gamelength",
            "kills", "deaths", "assists",
        )
    }
    has_champion = "champion" in frame.columns
    has_league = "league" in frame.columns

    player_rows: dict[str, list[dict[str, Any]]] = {}
    team_rows: dict[str, dict[str, dict[str, Any]]] = {}

    for index, row in frame.iterrows():
        game_id = str(row["_game_id"])
        side = str(row["_side"])
        minutes = _positive(_round(numeric["gamelength"].loc[index] / 60.0, 4))
        total_gold = _round(numeric["totalgold"].loc[index], 4)
        side_total = side_gold.get((game_id, side))
        kills = _round(numeric["kills"].loc[index], 0)
        deaths = _round(numeric["deaths"].loc[index], 0)
        assists = _round(numeric["assists"].loc[index], 0)
        damage_share = _round(numeric["damageshare"].loc[index], 6)
        won = int(row["_win"]) == 1

        entry = {
            "game_id": game_id,
            "date": row["_date"].date().isoformat(),
            "league": str(row["league"]) if has_league and pd.notna(row["league"]) else None,
            "opponent": opponents.get((game_id, side)),
            "champion": str(row["champion"]) if has_champion and pd.notna(row["champion"]) else None,
            "position": str(row["_role"]),
            "win": won,
            "cs_per_min": _round(numeric["cspm"].loc[index]),
            "gold_per_min": (
                _round(total_gold / minutes) if total_gold is not None and minutes else None
            ),
            "gold_share_pct": (
                _round(total_gold / side_total * 100.0)
                if total_gold is not None and side_total
                else None
            ),
            "damage_per_min": _round(numeric["dpm"].loc[index]),
            "damage_share_pct": _round(damage_share * 100.0) if damage_share is not None else None,
            "kda": (
                _round((kills + assists) / max(deaths, 1.0))
                if kills is not None and assists is not None and deaths is not None
                else None
            ),
            "game_length_min": _round(minutes) if minutes else None,
        }
        player_rows.setdefault(str(row["_player"]), []).append(entry)

        team = public_team_affiliation(row["_team"])
        if team is None:
            continue
        bucket = team_rows.setdefault(team, {})
        aggregate = bucket.get(game_id)
        if aggregate is None:
            aggregate = {
                "game_id": game_id,
                "date": entry["date"],
                "league": entry["league"],
                "opponent": entry["opponent"],
                "win": won,
                "game_length_min": entry["game_length_min"],
                "_gold": [],
                "_damage": [],
                "_kills": [],
                "_deaths": [],
            }
            bucket[game_id] = aggregate
        aggregate["_gold"].append(total_gold)
        aggregate["_damage"].append(
            _round(numeric["dpm"].loc[index] * minutes, 4)
            if minutes and _round(numeric["dpm"].loc[index]) is not None
            else None
        )
        aggregate["_kills"].append(kills)
        aggregate["_deaths"].append(deaths)

    players_payload = [
        _identity_payload(name, rows, _PLAYER_METRIC_KEYS, map_limit)
        for name, rows in sorted(player_rows.items())
    ]
    teams_payload = [
        _identity_payload(
            team,
            [_finalize_team_row(row) for row in bucket.values()],
            _TEAM_METRIC_KEYS,
            map_limit,
        )
        for team, bucket in sorted(team_rows.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "window_days": window_days,
        "map_limit": map_limit,
        "players": players_payload,
        "teams": teams_payload,
    }


def _finalize_team_row(row: dict[str, Any]) -> dict[str, Any]:
    """Collapse a team's five player rows into one team-level map row."""

    gold = [value for value in row.pop("_gold") if value is not None]
    damage = [value for value in row.pop("_damage") if value is not None]
    kills = [value for value in row.pop("_kills") if value is not None]
    deaths = [value for value in row.pop("_deaths") if value is not None]
    minutes = row.get("game_length_min")
    # A partial side would understate a team total, so only a complete side of
    # five measured players yields a team sum.
    row["gold_per_min"] = (
        _round(sum(gold) / minutes) if len(gold) == 5 and minutes else None
    )
    row["damage_per_min"] = (
        _round(sum(damage) / minutes) if len(damage) == 5 and minutes else None
    )
    row["kills"] = int(sum(kills)) if len(kills) == 5 else None
    row["deaths"] = int(sum(deaths)) if len(deaths) == 5 else None
    return row


def _identity_payload(
    name: str,
    rows: list[dict[str, Any]],
    metric_keys: tuple[str, ...],
    map_limit: int,
) -> dict[str, Any]:
    """Aggregate header plus the most recent capped map rows."""

    ordered = sorted(
        rows,
        key=lambda row: (str(row.get("date") or ""), str(row.get("game_id") or "")),
        reverse=True,
    )[:map_limit]
    maps = len(ordered)
    wins = sum(1 for row in ordered if row.get("win") is True)
    payload: dict[str, Any] = {
        "name": name,
        "maps": maps,
        "wins": wins,
        "losses": maps - wins,
        "win_rate": round(wins / maps, 4) if maps else None,
    }
    for key in metric_keys:
        payload[key] = _mean([row.get(key) for row in ordered])
    payload["games"] = ordered
    return payload


def player_map_stats_row_count(payload: Mapping[str, Any]) -> int:
    """Total emitted map rows across both identity kinds."""

    total = 0
    for kind in ("players", "teams"):
        for entry in payload.get(kind) or []:
            total += len(entry.get("games") or [])
    return total
