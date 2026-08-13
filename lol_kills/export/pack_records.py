"""Windowed public team/player records built from canonical map rows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import pandas as pd

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.competition import (
    TRANSPORT_LEAGUE_LABELS,
    canonicalize_competition_frame,
    competition_tier as classify_competition_tier,
    is_team_affiliation_league,
    source_league,
    team_identity_key,
)
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.player_map_grades import GRADE_CONTRACT, compute_player_map_grades, grade_payload


PUBLIC_TEAM_RATING_EXCLUSIONS = frozenset({"los-ratones"})
INVALID_COMPETITION_LABELS = frozenset({"", "UNKNOWN", *TRANSPORT_LEAGUE_LABELS})
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

DRAFT_PICK_SLOTS = (1, 2, 3, 4, 5)


def _draft_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _draft_champion_key(value: Any) -> str:
    return normalize_champ(_draft_text(value)).casefold()


def _draft_slots(row: pd.Series, prefix: str) -> list[str]:
    return [
        _draft_text(row.get(f"{prefix}{slot}"))
        for slot in DRAFT_PICK_SLOTS
        if _draft_text(row.get(f"{prefix}{slot}"))
    ]


def _first_pick_value(group: pd.DataFrame, metadata: Mapping[str, Any] | None) -> bool | None:
    candidate = (metadata or {}).get("blue_first_pick")
    if not _draft_text(candidate):
        for column in ("blue_firstPick", "firstPick"):
            if column in group.columns:
                candidate = group.iloc[0].get(column)
                if _draft_text(candidate):
                    break
    if isinstance(candidate, bool):
        return candidate
    if isinstance(candidate, (int, float)) and not pd.isna(candidate) and candidate in (0, 1):
        return bool(candidate)
    text = _draft_text(candidate).casefold()
    if text in {"blue", "1", "true", "yes", "first"}:
        return True
    if text in {"red", "0", "false", "no", "second"}:
        return False
    return None


def _public_draft_payload(
    group: pd.DataFrame,
    participants: list[dict[str, Any]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Keep the source draft facts public before tier scoring is attached.

    The source repeats bans and team pick slots on every player row. We read
    one side row and preserve the values only when they are present. Pick
    order is reconstructed from the source team pick slots and first-pick
    flag. Missing order remains explicit so downstream rates can fail closed.
    """

    bans: dict[str, list[str]] = {}
    picks_by_side: dict[str, list[str]] = {}
    for side in ("Blue", "Red"):
        side_frame = group[group["_side"].eq(side)]
        if side_frame.empty:
            continue
        first = side_frame.iloc[0]
        bans[side] = _draft_slots(first, "ban")
        if len(bans[side]) != 5 and metadata:
            bans[side] = [
                _draft_text(value)
                for value in (metadata.get(f"{side.lower()}_bans") or [])
                if _draft_text(value)
            ]
        picks = _draft_slots(first, "pick")
        if len(picks) != 5 and metadata:
            picks = [
                _draft_text(value)
                for value in (metadata.get(f"{side.lower()}_picks") or [])
                if _draft_text(value)
            ]
        picks_by_side[side] = picks

    roster_by_side_champion = {
        (str(row.get("side") or ""), _draft_champion_key(row.get("champion"))): row
        for row in participants
        if row.get("side") and row.get("champion")
    }
    first_pick = _first_pick_value(group, metadata)
    picked: list[dict[str, Any]] = []
    if first_pick is not None and all(len(picks_by_side.get(side, [])) == 5 for side in ("Blue", "Red")):
        first_side = "Blue" if first_pick else "Red"
        second_side = "Red" if first_side == "Blue" else "Blue"
        sequence = [
            (first_side, 1), (second_side, 1), (second_side, 2),
            (first_side, 2), (first_side, 3), (second_side, 3),
            (second_side, 4), (first_side, 4), (first_side, 5),
            (second_side, 5),
        ]
        for order, (side, slot) in enumerate(sequence, start=1):
            champion = picks_by_side[side][slot - 1]
            participant = roster_by_side_champion.get((side, _draft_champion_key(champion)))
            picked.append(
                {
                    "side": side,
                    "role": participant.get("role") if participant else None,
                    "champion": champion,
                    "order": order,
                }
            )
    else:
        for participant in participants:
            champion = _draft_text(participant.get("champion"))
            if champion:
                picked.append(
                    {
                        "side": participant.get("side"),
                        "role": participant.get("role"),
                        "champion": champion,
                        "order": None,
                    }
                )

    patch = _draft_text((metadata or {}).get("patch"))
    if not patch and "patch" in group.columns:
        patch = _draft_text(group.iloc[0].get("patch"))
    complete_bans = all(len(bans.get(side, [])) == 5 for side in ("Blue", "Red"))
    complete_picks = len(picked) == 10 and len({_draft_champion_key(item.get("champion")) for item in picked}) == 10
    complete_order = complete_picks and all(item.get("order") is not None for item in picked)
    status = "complete" if complete_bans and complete_order and patch else "limited" if complete_bans or picked else "unavailable"
    return {
        "schema_version": "scryglass:draft-pool:v1",
        "status": status,
        "source": "oracle-elixir",
        "patch": patch or None,
        "bans": {"Blue": bans.get("Blue", []), "Red": bans.get("Red", [])},
        "picked": picked,
        "unpicked": [],
        "reason": None if status == "complete" else "Complete bans, pick order, and patch identity are required for best-available rates.",
    }


def _wr(wins: int, games: int) -> float | None:
    return round(wins / games, 4) if games else None


def _mean(group: pd.DataFrame, column: str) -> float | None:
    if column not in group.columns:
        return None
    values = pd.to_numeric(group[column], errors="coerce").dropna()
    return round(float(values.mean()), 2) if not values.empty else None


def public_team_affiliation(value: Any) -> str | None:
    """Return a displayable team name under the public rating contract."""

    if value is None or pd.isna(value):
        return None
    display = normalize_team(str(value).strip())
    if not display or team_identity_key(display) in PUBLIC_TEAM_RATING_EXCLUSIONS:
        return None
    return display


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

    draft_columns = (
        "patch",
        "firstPick",
        "ban1", "ban2", "ban3", "ban4", "ban5",
        "pick1", "pick2", "pick3", "pick4", "pick5",
    )
    blue_columns = [
        c
        for c in (
            "_game_uid", "date", "league", "tournament", "result", "teamname", "grid_series_id",
            *draft_columns,
        )
        if c in blue.columns
    ]
    red_columns = [
        c
        for c in ("_game_uid", "teamname", "firstPick", *draft_columns[2:])
        if c in red.columns
    ]
    blue_renames = {
        "_game_uid": "game_uid",
        "result": "y_blue_win",
        "teamname": "blue_team",
        "firstPick": "blue_firstPick",
        **{f"ban{slot}": f"blue_ban{slot}" for slot in DRAFT_PICK_SLOTS},
        **{f"pick{slot}": f"blue_pick{slot}" for slot in DRAFT_PICK_SLOTS},
    }
    red_renames = {
        "_game_uid": "game_uid",
        "teamname": "red_team",
        "firstPick": "red_firstPick",
        **{f"ban{slot}": f"red_ban{slot}" for slot in DRAFT_PICK_SLOTS},
        **{f"pick{slot}": f"red_pick{slot}" for slot in DRAFT_PICK_SLOTS},
    }
    maps = blue[blue_columns].rename(columns=blue_renames)
    maps = maps.merge(
        red[red_columns].rename(columns=red_renames),
        on="game_uid",
        how="inner",
    )
    maps["date"] = pd.to_datetime(maps.get("date"), errors="coerce")
    maps["y_blue_win"] = pd.to_numeric(maps.get("y_blue_win"), errors="coerce")
    maps = maps.dropna(subset=["date", "y_blue_win", "blue_team", "red_team"])
    # Carry the authoritative GRID series id through to the rating fit so the
    # hierarchical ladder can group true series instead of falling back to
    # derived keys (issue #44).  OE-only rows keep an empty id and remain
    # game-level observations.
    if "grid_series_id" in maps.columns:
        maps["grid_series_id"] = maps["grid_series_id"].fillna("").astype(str).str.strip()
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
    if "league" in maps.columns:
        keep &= ~maps["league"].map(source_league).isin(INVALID_COMPETITION_LABELS)
    if "competition_tier" in maps.columns:
        keep &= maps["competition_tier"].astype(str).ne("other")
    return maps.loc[keep].copy()


def _affiliation_rows(group: pd.DataFrame) -> pd.DataFrame:
    """Return rows that can define a team's current league membership."""

    if group.empty or "league" not in group.columns:
        return group.iloc[0:0].copy()
    return group[group["league"].map(is_team_affiliation_league)]


def _latest_row(group: pd.DataFrame) -> pd.Series | None:
    """Return the latest dated row, with input order as the tie-break."""

    if group.empty:
        return None
    dates = (
        pd.to_datetime(group["date"], errors="coerce", utc=True)
        if "date" in group.columns
        else pd.Series(pd.NaT, index=group.index)
    )
    if not dates.notna().any():
        return None
    return group.loc[dates.idxmax()]


def _primary_league(group: pd.DataFrame) -> str | None:
    """Return the latest observed league-membership affiliation.

    Cross-region events provide useful evidence but must not overwrite the
    team's domestic league. Domestic cups follow the same rule. A latest-date
    rule still reflects migrations such as LTA South to CBLOL and PCS to LCP.
    """

    candidates = _affiliation_rows(group)
    latest = _latest_row(candidates)
    if latest is not None:
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
        current_row = _latest_row(_affiliation_rows(group))
        last_event_row = _latest_row(group)
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
            "last_event_league": str(last_event_row["league"]) if last_event_row is not None else None,
            "last_event_tier": str(last_event_row["competition_tier"]) if last_event_row is not None else None,
            "last_event_date": str(last_event_row["date"]) if last_event_row is not None else None,
            "intl": bool(group["is_international"].any()),
            "interregional": bool(group["is_interregional"].any()),
            "wins": wins,
            "games": games,
            "wr": _wr(wins, games),
            "by_league": by_league,
            "by_tier": by_tier,
        }
    return records


def build_player_records(
    players: pd.DataFrame,
    *,
    team_records: dict[str, dict[str, Any]] | None = None,
    canonicalized: bool = False,
) -> dict[str, dict[str, Any]]:
    """Aggregate player results and align current affiliation to current team."""

    if players is None or players.empty or "playername" not in players.columns:
        return {}
    source = players if canonicalized else canonicalize_competition_frame(players)
    frame = source
    named = frame["playername"].notna()
    if not named.all():
        frame = frame[named].copy()
    if "position" in frame.columns:
        player_rows = frame["position"].astype(str).str.lower().ne("team")
        if not player_rows.all():
            frame = frame[player_rows].copy()
    if frame.empty:
        return {}
    frame["result"] = pd.to_numeric(frame.get("result"), errors="coerce")
    frame = frame.dropna(subset=["result"])
    if frame.empty:
        return {}

    team_by_key = {
        str(record.get("team_key") or team_identity_key(team)): record
        for team, record in (team_records or {}).items()
    }
    records: dict[str, dict[str, Any]] = {}
    for player, group in frame.groupby(frame["playername"].astype(str), sort=True):
        wins = int(round(float(group["result"].sum())))
        games = int(len(group))
        valid_league = ~group["league"].astype(str).str.upper().isin(INVALID_COMPETITION_LABELS)
        classified = group[valid_league]
        leagues = sorted(str(x) for x in classified["league"].dropna().unique())
        player_affiliation_row = _latest_row(_affiliation_rows(classified))
        event_affiliation_row = _latest_row(
            classified[classified["competition_tier"].isin({"tier1", "tier2", "tier3"})]
        )
        latest_valid = _latest_row(classified)
        observed_row = _latest_row(group)
        current_team = (
            public_team_affiliation(observed_row.get("teamname"))
            if observed_row is not None
            else None
        )
        team_record = team_by_key.get(team_identity_key(current_team)) if current_team else None
        player_league = (
            str(player_affiliation_row["league"])
            if player_affiliation_row is not None
            else None
        )
        player_tier = (
            str(player_affiliation_row["competition_tier"])
            if player_affiliation_row is not None
            else None
        )
        event_league = (
            str(event_affiliation_row["league"])
            if event_affiliation_row is not None
            else None
        )
        event_tier = (
            str(event_affiliation_row["competition_tier"])
            if event_affiliation_row is not None
            else None
        )
        if team_record is not None:
            primary = team_record.get("current_league")
            current_tier = team_record.get("current_tier")
            affiliation_source = "current_team"
        else:
            primary = player_league
            current_tier = player_tier
            affiliation_source = "player_history" if player_affiliation_row is not None else None
        affiliation_repaired = bool(
            team_record is not None
            and (primary != event_league or current_tier != event_tier)
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
            "current_tier": current_tier,
            "current_team": current_team,
            "current_date": str(observed_row["date"]) if observed_row is not None else None,
            "affiliation_source": affiliation_source,
            "affiliation_repaired": affiliation_repaired,
            "last_event_league": str(latest_valid["league"]) if latest_valid is not None else None,
            "last_event_tier": str(latest_valid["competition_tier"]) if latest_valid is not None else None,
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


def build_player_champion_records(
    players: pd.DataFrame,
) -> dict[str, list[dict[str, Any]]]:
    """Precompute compact champion records for each public player profile."""

    required = {"playername", "champion", "result"}
    if players is None or players.empty or not required.issubset(players.columns):
        return {}
    frame = players.copy()
    if "position" in frame.columns:
        frame = frame[frame["position"].astype(str).str.lower().ne("team")]
    frame["playername"] = frame["playername"].astype("string").str.strip()
    frame["champion"] = frame["champion"].astype("string").str.strip()
    frame["result"] = pd.to_numeric(frame["result"], errors="coerce")
    frame = frame[
        frame["playername"].notna()
        & frame["playername"].ne("")
        & frame["champion"].notna()
        & frame["champion"].ne("")
        & frame["result"].isin({0, 1})
    ]
    if frame.empty:
        return {}

    records: dict[str, list[dict[str, Any]]] = {}
    for player, player_group in frame.groupby("playername", sort=True):
        champions: list[dict[str, Any]] = []
        for champion, group in player_group.groupby("champion", sort=True):
            games = int(len(group))
            wins = int(round(float(group["result"].sum())))
            champions.append(
                {
                    "champion": str(champion),
                    "games": games,
                    "wins": wins,
                    "losses": games - wins,
                    "wr": _wr(wins, games),
                    "kills": _mean(group, "kills"),
                    "deaths": _mean(group, "deaths"),
                    "assists": _mean(group, "assists"),
                }
            )
        records[str(player)] = sorted(
            champions,
            key=lambda row: (-row["games"], -row["wins"], row["champion"].casefold()),
        )
    return records


def _profile_number(value: Any) -> float | int | None:
    number = pd.to_numeric(value, errors="coerce")
    if pd.isna(number):
        return None
    result = float(number)
    return int(result) if result.is_integer() else round(result, 2)


def _profile_sum(*values: Any) -> float | int | None:
    numbers = [_profile_number(value) for value in values]
    present = [number for number in numbers if number is not None]
    return _profile_number(sum(present)) if present else None


def _profile_game_identity(game: Mapping[str, Any]) -> tuple[object, ...]:
    players = game.get("players")
    rows = players if isinstance(players, list) else []
    roster = tuple(
        sorted(
            (
                str(row.get("player") or ""),
                str(row.get("side") or ""),
                str(row.get("role") or ""),
                str(row.get("champion") or ""),
            )
            for row in rows
            if isinstance(row, Mapping)
        )
    )
    return (
        str(game.get("date") or ""),
        str(game.get("blue_team") or ""),
        str(game.get("red_team") or ""),
        game.get("blue_win"),
        roster,
    )


def profile_game_has_complete_stats(game: Mapping[str, Any]) -> bool:
    players = game.get("players")
    if not isinstance(players, list) or len(players) != 10:
        return False
    return all(
        isinstance(row, Mapping)
        and row.get("kills") is not None
        and row.get("deaths") is not None
        and row.get("assists") is not None
        for row in players
    )


def merge_accepted_profile_games(
    candidate_games: Mapping[str, Mapping[str, Any]],
    accepted_games: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Keep complete candidates and preserve matching accepted KDA or grades."""

    accepted_games = accepted_games or {}
    merged: dict[str, dict[str, Any]] = {}
    for game_id, candidate in candidate_games.items():
        accepted = accepted_games.get(game_id)
        identities_match = (
            isinstance(accepted, Mapping)
            and _profile_game_identity(candidate) == _profile_game_identity(accepted)
        )
        if not profile_game_has_complete_stats(candidate):
            if identities_match and profile_game_has_complete_stats(accepted):
                selected = deepcopy(dict(accepted))
                for field in ("patch", "draft_pool", "draft_contribution"):
                    if field in candidate:
                        selected[field] = deepcopy(candidate[field])
                merged[game_id] = selected
            continue

        selected = deepcopy(dict(candidate))
        if identities_match:
            accepted_by_player = {
                str(row.get("player") or ""): row
                for row in accepted.get("players", [])
                if isinstance(row, Mapping)
            }
            for row in selected.get("players", []):
                previous = accepted_by_player.get(str(row.get("player") or ""))
                if not isinstance(previous, Mapping):
                    continue
                current_grade = row.get("grade") or {}
                previous_grade = previous.get("grade") or {}
                if (
                    current_grade.get("status") != "available"
                    and previous_grade.get("status") == "available"
                ):
                    row["grade"] = deepcopy(previous_grade)
        merged[game_id] = selected
    return merged


def build_profile_records(
    players: pd.DataFrame,
    *,
    champion_image_urls: Mapping[str, str] | None = None,
    composition_signals: Mapping[str, Mapping[str, Any]] | None = None,
    draft_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    recent_limit: int = 10,
    recent_window_days: int = 120,
    include_archive: bool = False,
) -> dict[str, Any]:
    """Build recent profile records and, when requested, the accepted map archive."""

    required = {"playername", "teamname", "side", "position", "result", "date"}
    if players is None or players.empty or not required.issubset(players.columns):
        return {"schema_version": "scryglass:profile-records:v3", "grade_contract": GRADE_CONTRACT, "window_days": recent_window_days, "champion_images": {}, "games": {}, "players": {}, "teams": {}}
    if recent_limit < 1 or recent_window_days < 1:
        raise ValueError("recent_limit and recent_window_days must be positive")

    useful_columns = [
        column
        for column in (
            "game_uid",
            "gameid",
            "league",
            "league_source",
            "tournament",
            "playername",
            "teamname",
            "side",
            "position",
            "result",
            "date",
            "champion",
            "kills",
            "deaths",
            "assists",
            "competition_tier",
            "teamkills",
            "gamelength",
            "dpm",
            "damageshare",
            "totalgold",
            "total cs",
            "minionkills",
            "monsterkills",
            "cspm",
            "visionscore",
            "wardsplaced",
            "wpm",
            "wcpm",
            "golddiffat10",
            "dragons",
            "heralds",
            "void_grubs",
            "barons",
            "atakhans",
            "towers",
            "inhibitors",
            "patch",
            "ban1",
            "ban2",
            "ban3",
            "ban4",
            "ban5",
            "pick1",
            "pick2",
            "pick3",
            "pick4",
            "pick5",
            "firstPick",
            "blue_firstPick",
        )
        if column in players.columns
    ]
    frame = canonicalize_competition_frame(players[useful_columns].copy())
    identity_source = frame.get("game_uid", frame.get("gameid"))
    if identity_source is None:
        return {"schema_version": "scryglass:profile-records:v3", "grade_contract": GRADE_CONTRACT, "window_days": recent_window_days, "champion_images": {}, "games": {}, "players": {}, "teams": {}}
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
    frame = frame[
        frame["_game_id"].astype(str).str.strip().ne("")
        & frame["_date"].notna()
        & frame["_side"].isin({"Blue", "Red"})
        & frame["_role"].notna()
        & frame["_player"].notna()
        & frame["_player"].ne("")
        & frame["_team"].notna()
        & frame["_team"].ne("")
    ].copy()
    grades = compute_player_map_grades(frame)
    grade_lookup = {
        (str(row["game_id"]), str(row["player"]).casefold()): row
        for _, row in grades.iterrows()
    }
    archive_frame = frame.copy() if include_archive else None
    latest_date = frame["_date"].max()
    if pd.notna(latest_date):
        frame = frame[
            frame["_date"].ge(latest_date - pd.Timedelta(days=recent_window_days))
        ].copy()

    player_recent = (
        frame[["_player", "_game_id", "_date"]]
        .drop_duplicates(["_player", "_game_id"])
        .sort_values(["_player", "_date"], ascending=[True, False])
        .groupby("_player", sort=True)
        .head(recent_limit)
    )
    team_recent = (
        frame[["_team", "_game_id", "_date"]]
        .drop_duplicates(["_team", "_game_id"])
        .sort_values(["_team", "_date"], ascending=[True, False])
        .groupby("_team", sort=True)
        .head(recent_limit)
    )
    player_index = {
        str(player): group["_game_id"].astype(str).tolist()
        for player, group in player_recent.groupby("_player", sort=True)
    }
    team_index = {
        str(team): group["_game_id"].astype(str).tolist()
        for team, group in team_recent.groupby("_team", sort=True)
        if public_team_affiliation(team)
    }
    selected = {
        game_id
        for values in (*player_index.values(), *team_index.values())
        for game_id in values
    }
    frame = frame[frame["_game_id"].astype(str).isin(selected)].copy()
    game_frame = archive_frame if archive_frame is not None else frame

    games: dict[str, dict[str, Any]] = {}
    images = champion_image_urls or {}
    role_order = {role: index for index, role in enumerate(PUBLIC_ROLE_ORDER)}
    for game_id, group in game_frame.groupby("_game_id", sort=False):
        blue = group[group["_side"].eq("Blue")]
        red = group[group["_side"].eq("Red")]
        if len(group) != 10 or len(blue) != 5 or len(red) != 5 or group["_player"].nunique() != 10:
            continue
        if set(blue["_role"]) != set(PUBLIC_ROLE_ORDER) or set(red["_role"]) != set(PUBLIC_ROLE_ORDER):
            continue
        blue_teams = blue["_team"].dropna().unique()
        red_teams = red["_team"].dropna().unique()
        if len(blue_teams) != 1 or len(red_teams) != 1:
            continue
        blue_team = public_team_affiliation(blue_teams[0])
        red_team = public_team_affiliation(red_teams[0])
        if not blue_team or not red_team or blue_team == red_team:
            continue
        blue_result = _profile_number(blue["result"].iloc[0])
        if blue_result not in (0, 1):
            continue
        date = group["_date"].max()
        league = str(group["league"].iloc[0]) if "league" in group.columns else "UNKNOWN"
        tier_values = group.get("competition_tier", pd.Series(dtype="string")).dropna()
        tier = str(tier_values.iloc[0]).strip().casefold() if not tier_values.empty else classify_competition_tier(league)
        participants: list[dict[str, Any]] = []
        for _, row in group.sort_values(["_side", "_role"], key=lambda values: values.map(role_order) if values.name == "_role" else values).iterrows():
            champion = str(row.get("champion") or "").strip()
            farm = _profile_number(row.get("total cs"))
            if farm is None:
                farm = _profile_sum(row.get("minionkills"), row.get("monsterkills"))
            participants.append(
                {
                    "player": str(row["_player"]),
                    "side": str(row["_side"]),
                    "role": str(row["_role"]),
                    "champion": champion or None,
                    "kills": _profile_number(row.get("kills")),
                    "deaths": _profile_number(row.get("deaths")),
                    "assists": _profile_number(row.get("assists")),
                    "team_kills": _profile_number(row.get("teamkills")),
                    "cs": farm,
                    "cs_per_minute": _profile_number(row.get("cspm")),
                    "damage_per_minute": _profile_number(row.get("dpm")),
                    "damage_share": _profile_number(row.get("damageshare")),
                    "gold": _profile_number(row.get("totalgold")),
                    "gold_diff_at_10": _profile_number(row.get("golddiffat10")),
                    "vision_score": _profile_number(row.get("visionscore")),
                    "wards_placed": _profile_number(row.get("wardsplaced")),
                    "grade": grade_payload(
                        grade_lookup.get((str(game_id), str(row["_player"]).casefold()))
                    ),
                }
            )
        team_stats: dict[str, dict[str, Any]] = {}
        for side_name, side_frame in (("Blue", blue), ("Red", red)):
            first = side_frame.iloc[0]
            team_stats[side_name] = {
                "kills": _profile_number(first.get("teamkills")),
                "gold": _profile_number(pd.to_numeric(side_frame.get("totalgold"), errors="coerce").sum(min_count=1)) if "totalgold" in side_frame.columns else None,
                "dragons": _profile_number(first.get("dragons")),
                "heralds": _profile_number(first.get("heralds")),
                "void_grubs": _profile_number(first.get("void_grubs")),
                "barons": _profile_number(first.get("barons")),
                "atakhans": _profile_number(first.get("atakhans")),
                "towers": _profile_number(first.get("towers")),
                "inhibitors": _profile_number(first.get("inhibitors")),
            }
        key = str(game_id)
        game_draft_metadata = (draft_metadata or {}).get(key)
        if not isinstance(game_draft_metadata, Mapping):
            game_draft_metadata = None
        games[key] = {
            "game_id": key,
            "date": date.isoformat().replace("+00:00", "Z"),
            "league": league,
            "competition_tier": tier,
            "patch": _draft_text((game_draft_metadata or {}).get("patch")) or _draft_text(group.iloc[0].get("patch")) or None,
            "blue_team": blue_team,
            "red_team": red_team,
            "blue_win": int(blue_result),
            "duration_seconds": _profile_number(group["gamelength"].iloc[0]) if "gamelength" in group.columns else None,
            "team_stats": team_stats,
            "players": participants,
        }
        games[key]["draft_pool"] = _public_draft_payload(
            group,
            participants,
            game_draft_metadata,
        )
        if composition_signals and key in composition_signals:
            games[key]["draft_contribution"] = deepcopy(dict(composition_signals[key]))
    archive_games = games
    available = set(games).intersection(selected)
    player_index = {
        identity: [game_id for game_id in values if game_id in available]
        for identity, values in player_index.items()
        if any(game_id in available for game_id in values)
    }
    team_index = {
        identity: [game_id for game_id in values if game_id in available]
        for identity, values in team_index.items()
        if any(game_id in available for game_id in values)
    }
    payload = {
        "schema_version": "scryglass:profile-records:v3",
        "grade_contract": GRADE_CONTRACT,
        "window_days": recent_window_days,
        "champion_images": dict(sorted(images.items())),
        "games": {game_id: games[game_id] for game_id in sorted(available)},
        "players": player_index,
        "teams": team_index,
    }
    if include_archive:
        payload["_archive_games"] = {
            game_id: archive_games[game_id]
            for game_id in sorted(archive_games)
        }
    return payload


def summarize_player_affiliations(
    player_records: dict[str, dict[str, Any]],
    team_records: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Summarize roster-affiliation repairs without blocking pack creation."""

    team_by_key = {
        str(record.get("team_key") or team_identity_key(team)): record
        for team, record in team_records.items()
    }
    inherited = 0
    repaired = 0
    unresolved_teams = 0
    remaining_conflicts = 0
    for record in player_records.values():
        current_team = record.get("current_team")
        if not current_team:
            continue
        team_record = team_by_key.get(team_identity_key(current_team))
        if team_record is None:
            unresolved_teams += 1
            continue
        inherited += 1
        repaired += int(bool(record.get("affiliation_repaired")))
        if (
            record.get("current_league") != team_record.get("current_league")
            or record.get("current_tier") != team_record.get("current_tier")
        ):
            remaining_conflicts += 1
    return {
        "players": len(player_records),
        "current_team_inherited": inherited,
        "repaired_from_team_roster": repaired,
        "unresolved_current_teams": unresolved_teams,
        "remaining_team_player_conflicts": remaining_conflicts,
    }
