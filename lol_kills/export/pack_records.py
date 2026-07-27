"""Windowed public team/player records built from canonical map rows."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from lol_kills.etl.competition import canonicalize_competition_frame, team_identity_key
from lol_kills.export import pack_spec
from lol_kills.etl.tournament_registry import (
    DEFAULT_REGISTRY_PATH,
    current_membership_from_registry,
    load_tournament_registry,
    validate_tournament_registry,
)


def _wr(wins: int, games: int) -> float | None:
    return round(wins / games, 4) if games else None


def tournament_family(value: Any) -> str | None:
    """Normalize a tournament label to its competition family.

    Source rows often append a stage in parentheses (for example, the two
    LPL Split 3 groups).  The stage is useful row provenance, but it must not
    make one league's current tournament look like several memberships.
    """

    if value is None or pd.isna(value):
        return None
    raw = re.sub(r"\s+", " ", str(value).strip())
    if not raw or raw.casefold() in {"nan", "none", "null"}:
        return None
    family = re.sub(r"\s+\([^()]*\)\s*$", "", raw).strip()
    return family or None


def _tournament_key(league: Any, family: Any) -> str:
    return f"{str(league)}|{str(family)}"


def build_current_tournament_membership(
    maps: pd.DataFrame | None = None,
    *,
    as_of: Any = None,
    window_days: int = 90,
    registry: dict[str, Any] | None = None,
    registry_path: Any = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Build current membership from Riot's reviewed tournament registry.

    ``maps`` are used only for a reconciliation diagnostic.  An appearance can
    never create or remove membership.  ``as_of`` is the publication/check
    clock, deliberately separate from the latest match/model-data timestamp.
    """

    checked_at = (
        pd.Timestamp.now(tz="UTC")
        if as_of is None
        else pd.to_datetime(as_of, errors="coerce", utc=True)
    )
    payload = (
        validate_tournament_registry(registry)
        if registry is not None
        else load_tournament_registry(registry_path)
    )
    result = current_membership_from_registry(payload, as_of=checked_at)
    result["window_days"] = int(window_days)

    observed_by_league: dict[str, set[str]] = {}
    if (
        maps is not None
        and not maps.empty
        and "tournament" in maps.columns
        and "league" in maps.columns
    ):
        frame = canonicalize_competition_frame(maps).copy()
        frame["_date"] = pd.to_datetime(
            frame.get("date"), errors="coerce", utc=True
        )
        frame["_tournament_family"] = frame["tournament"].map(
            tournament_family
        )
        observation_end = frame["_date"].max()
        if pd.notna(observation_end):
            start = observation_end - pd.Timedelta(
                days=max(0, int(window_days))
            )
            frame = frame[
                frame["_date"].between(
                    start, observation_end, inclusive="both"
                )
            ]
        for _, row in frame.iterrows():
            league = str(row.get("league") or "")
            if (
                result["leagues"].get(league)
                != row.get("_tournament_family")
            ):
                continue
            for column in (
                "blue_team",
                "red_team",
                "blue_teamname",
                "red_teamname",
            ):
                team = row.get(column)
                if team is None or pd.isna(team) or not str(team).strip():
                    continue
                observed_by_league.setdefault(league, set()).add(
                    team_identity_key(team)
                )

    registered = {
        league: set(keys)
        for league, keys in result["participants_by_league"].items()
    }
    result["observation_audit"] = {
        league: {
            "registered_participants": len(registered.get(league, set())),
            "observed_participants": len(
                observed_by_league.get(league, set())
            ),
            "observed_not_registered": sorted(
                observed_by_league.get(league, set())
                - registered.get(league, set())
            ),
            "registered_not_observed": sorted(
                registered.get(league, set())
                - observed_by_league.get(league, set())
            ),
        }
        for league in sorted(result["leagues"])
    }
    return result


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
    game_column = "game_uid" if "game_uid" in frame.columns else "gameid"
    if game_column not in frame.columns:
        return pd.DataFrame()
    frame["_game_uid"] = frame[game_column].astype(str)
    frame["_side"] = frame["side"].astype(str).str.title()
    frame = frame[frame["_side"].isin({"Blue", "Red"})]
    side_counts = frame.groupby(["_game_uid", "_side"], sort=False).size()
    duplicate_sides = side_counts[side_counts.gt(1)]
    if not duplicate_sides.empty:
        examples = [
            f"{game_id}:{side}"
            for game_id, side in duplicate_sides.index[:10]
        ]
        raise ValueError(
            "team-game population contains duplicate side rows; "
            f"examples={examples}"
        )
    blue = frame[frame["_side"].eq("Blue")]
    red = frame[frame["_side"].eq("Red")]
    if blue.empty or red.empty:
        return pd.DataFrame()

    blue_columns = [
        column
        for column in (
            "_game_uid",
            "date",
            "league",
            "tournament",
            "result",
            "teamname",
            "game",
            "source",
            "year",
            "oe_year",
            "split",
            "playoffs",
            "patch",
            "gamelength",
            "ckpm",
            "datacompleteness",
            "grid_series_id",
            "grid_game_index",
            "series_format",
            "series_format_source",
            "series_format_stage_id",
            "series_format_registry_snapshot_id",
            "series_format_registry_verified",
            "series_format_registry_conflict",
            "best_of",
            "series_completion_status",
            "series_completion_source",
            "completion_source",
        )
        if column in blue.columns
    ]
    side_source_fields = [
        field for field in pack_spec.MAPS_SIDE_FIELDS if field in frame.columns
    ]
    blue_columns.extend(
        field for field in side_source_fields if field not in blue_columns
    )
    red_columns = [
        column
        for column in ("_game_uid", "teamname", *side_source_fields)
        if column in red.columns
    ]
    maps = blue[blue_columns].rename(
        columns={
            "_game_uid": "game_uid",
            "teamname": "blue_team",
            **{field: f"blue_{field}" for field in side_source_fields},
        }
    )
    maps = maps.merge(
        red[red_columns].rename(
            columns={
                "_game_uid": "game_uid",
                "teamname": "red_team",
                **{field: f"red_{field}" for field in side_source_fields},
            }
        ),
        on="game_uid",
        how="inner",
    )
    maps["date"] = pd.to_datetime(maps.get("date"), errors="coerce")
    maps["gamelength"] = pd.to_numeric(
        maps.get("gamelength"),
        errors="coerce",
    )
    maps["length_min"] = maps["gamelength"] / 60.0
    maps["y_blue_win"] = pd.to_numeric(
        maps.get("blue_result"),
        errors="coerce",
    )
    maps = maps.dropna(subset=["date", "y_blue_win", "blue_team", "red_team"])
    maps["oe_gameid"] = maps["game_uid"]
    maps["blue_teamname"] = maps["blue_team"]
    maps["red_teamname"] = maps["red_team"]
    if "red_result" not in maps.columns:
        maps["red_result"] = 1.0 - maps["y_blue_win"]
    blue_kills = pd.to_numeric(maps.get("blue_teamkills"), errors="coerce")
    red_kills = pd.to_numeric(maps.get("red_teamkills"), errors="coerce")
    maps["total_kills"] = blue_kills + red_kills
    maps["y_total_kills"] = maps["total_kills"]
    source = maps.get(
        "source",
        pd.Series("oe", index=maps.index),
    ).fillna("").astype(str).str.casefold()
    maps["source_grid"] = source.eq("grid")
    maps["source_oe"] = ~maps["source_grid"]
    maps["map_detail_source"] = source.map(
        lambda value: (
            "grid_team_aggregate"
            if value == "grid"
            else "oe_team_aggregate"
        )
    )
    return canonicalize_competition_frame(maps).sort_values("date").reset_index(drop=True)


def complete_public_map_population(
    feature_maps: pd.DataFrame,
    team_maps: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Append full-feed maps absent from the feature-focused major slice.

    The wide warehouse map is intentionally major-event focused. Public
    records, ratings, detail pages, and downloads must nevertheless agree on
    game identities. Missing games are therefore appended from the canonical
    two-team aggregate feed with detail provenance made explicit; unknown
    draft/objective fields remain null rather than being fabricated.
    """

    detailed = feature_maps.copy() if feature_maps is not None else pd.DataFrame()
    complete = team_maps.copy() if team_maps is not None else pd.DataFrame()
    if complete.empty:
        return detailed, {
            "feature_maps": int(len(detailed)),
            "full_team_maps": 0,
            "appended_team_aggregate_maps": 0,
        }
    if detailed.empty:
        out = complete.copy()
        out["map_detail_source"] = out.get(
            "map_detail_source",
            "oe_team_aggregate",
        )
        return out, {
            "feature_maps": 0,
            "full_team_maps": int(len(complete)),
            "appended_team_aggregate_maps": int(len(complete)),
        }

    def identity(frame: pd.DataFrame) -> pd.Series:
        if "oe_gameid" in frame.columns:
            values = frame["oe_gameid"]
            if "game_uid" in frame.columns:
                values = values.where(values.notna(), frame["game_uid"])
            return values.astype(str)
        return frame["game_uid"].astype(str)

    detailed = detailed.copy()
    detailed["_public_map_identity"] = identity(detailed)
    complete = complete.copy()
    complete["_public_map_identity"] = identity(complete)
    if detailed["_public_map_identity"].duplicated().any():
        raise ValueError("feature map population contains duplicate identities")
    if complete["_public_map_identity"].duplicated().any():
        raise ValueError("team map population contains duplicate identities")

    oe_backed = detailed.get(
        "source_oe",
        pd.Series(False, index=detailed.index),
    ).fillna(False).astype(bool)
    grid_backed = detailed.get(
        "source_grid",
        pd.Series(False, index=detailed.index),
    ).fillna(False).astype(bool)
    inferred_detail_source = pd.Series(
        "oe_wide_feature_map",
        index=detailed.index,
    )
    inferred_detail_source.loc[grid_backed & ~oe_backed] = "grid_event_detail"
    detailed["map_detail_source"] = detailed.get(
        "map_detail_source",
        inferred_detail_source,
    ).where(
        detailed.get(
            "map_detail_source",
            inferred_detail_source,
        ).notna(),
        inferred_detail_source,
    )
    missing = complete[
        ~complete["_public_map_identity"].isin(
            set(detailed["_public_map_identity"])
        )
    ].copy()
    missing["map_detail_source"] = missing.get(
        "map_detail_source",
        pd.Series("oe_team_aggregate", index=missing.index),
    ).fillna("oe_team_aggregate")
    columns = sorted(set(detailed.columns) | set(missing.columns))
    out = pd.concat(
        [
            detailed.reindex(columns=columns),
            missing.reindex(columns=columns),
        ],
        ignore_index=True,
    )
    out = out.drop(columns=["_public_map_identity"]).sort_values(
        ["date", "game_uid"],
        kind="mergesort",
    )
    return out.reset_index(drop=True), {
        "feature_maps": int(len(detailed)),
        "full_team_maps": int(len(complete)),
        "appended_team_aggregate_maps": int(len(missing)),
    }


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
                    "tournament": row.get("tournament"),
                    "tournament_family": tournament_family(row.get("tournament")),
                    "win": win,
                }
            )
    return pd.DataFrame(rows)


def build_team_records(
    maps: pd.DataFrame,
    current_membership: dict[str, Any] | None = None,
    *,
    tournament_maps: pd.DataFrame | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate one record per canonical team identity for one pack window."""

    if maps is None or maps.empty:
        return {}
    frame = canonicalize_competition_frame(maps)
    rows = _team_rows(frame)
    if rows.empty:
        return {}

    tournament_rows = rows
    if tournament_maps is not None and not tournament_maps.empty:
        tournament_rows = _team_rows(canonicalize_competition_frame(tournament_maps))
    membership_teams = (current_membership or {}).get("team_leagues", {})
    membership_display = (current_membership or {}).get(
        "team_display_names", {}
    )

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

        by_tournament: dict[str, dict[str, Any]] = {}
        tournament_group = tournament_rows[tournament_rows["team_key"].eq(key)]
        tournament_group = tournament_group[tournament_group["tournament_family"].notna()]
        for (league, family), tg in tournament_group.groupby(["league", "tournament_family"], sort=True):
            wins = int(round(float(tg["win"].sum())))
            games = int(len(tg))
            by_tournament[_tournament_key(league, family)] = {
                "wins": wins,
                "games": games,
                "wr": _wr(wins, games),
            }

        primary = _primary_league(group)
        observed = group[
            group["competition_tier"].isin({"tier1", "tier2", "tier3"})
        ]
        observed_row = (
            observed.loc[
                pd.to_datetime(
                    observed["date"], errors="coerce"
                ).idxmax()
            ]
            if not observed.empty
            and pd.to_datetime(
                observed["date"], errors="coerce"
            ).notna().any()
            else None
        )
        registered_leagues = membership_teams.get(key, {})
        if len(registered_leagues) > 1:
            raise ValueError(
                f"team {key!r} has multiple current Tier 1 leagues"
            )
        current_league = (
            next(iter(registered_leagues))
            if registered_leagues
            else None
        )
        current_tournament = (
            registered_leagues.get(current_league)
            if current_league is not None
            else None
        )
        wins = int(round(float(group["win"].sum())))
        games = int(len(group))
        records[display] = {
            "team_key": key,
            "leagues": sorted(str(x) for x in group["league"].unique()),
            "source_leagues": sorted(str(x) for x in group["league_source"].unique() if x),
            "primary": primary,
            "current_league": current_league,
            "current_tier": "tier1" if current_league is not None else None,
            "current_team": membership_display.get(key, display)
            if current_league is not None
            else None,
            "current_date": (
                str(observed_row["date"])
                if observed_row is not None
                else None
            ),
            "current_tournament": current_tournament,
            "membership_as_of": (current_membership or {}).get("as_of")
            if current_league is not None
            else None,
            "membership_source": (current_membership or {}).get(
                "authority"
            )
            if current_league is not None
            else None,
            "intl": bool(group["is_international"].any()),
            "interregional": bool(group["is_interregional"].any()),
            "wins": wins,
            "games": games,
            "wr": _wr(wins, games),
            "by_league": by_league,
            "by_tier": by_tier,
            "by_tournament": by_tournament,
        }
    return records


def build_player_records(
    players: pd.DataFrame,
    current_membership: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Aggregate player results without claiming an unverified current roster.

    Historical rows establish only a player's last observed affiliation. A
    current affiliation is published only when the player has an observed map
    for a team in the active authoritative tournament registry. That proves
    tournament participation, not a contract or starter/substitute status.
    """

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

    identity_source = "name_fallback_no_playerid_column"
    identity_column = "playername"
    if "playerid" in frame.columns:
        identity_source = "provider_playerid"
        frame["_player_id"] = frame["playerid"].map(
            lambda value: (
                str(value).strip()
                if value is not None
                and not pd.isna(value)
                and str(value).strip()
                and str(value).strip().casefold() not in {"nan", "none", "null"}
                else None
            )
        )
        frame = frame[frame["_player_id"].notna()].copy()
        if frame.empty:
            return {}
        frame["_player_name_key"] = (
            frame["playername"].astype(str).str.strip().str.casefold()
        )
        ids_by_name = frame.groupby("_player_name_key")["_player_id"].agg(
            lambda values: {str(value) for value in values}
        )
        colliding_names = set(ids_by_name[ids_by_name.map(len).gt(1)].index)
        if colliding_names:
            colliding_ids = set(
                frame.loc[
                    frame["_player_name_key"].isin(colliding_names),
                    "_player_id",
                ].astype(str)
            )
            frame = frame[~frame["_player_id"].astype(str).isin(colliding_ids)]
        if frame.empty:
            return {}
        identity_column = "_player_id"

    records: dict[str, dict[str, Any]] = {}
    membership_teams = (current_membership or {}).get("team_leagues", {})
    for player_identity, group in frame.groupby(identity_column, sort=True):
        wins = int(round(float(group["result"].sum())))
        games = int(len(group))
        leagues = sorted(str(x) for x in group["league"].dropna().unique())
        dates = (
            pd.to_datetime(group["date"], errors="coerce")
            if "date" in group.columns
            else pd.Series(pd.NaT, index=group.index)
        )
        last_row = group.loc[dates.idxmax()] if dates.notna().any() else None
        player = (
            str(last_row["playername"]).strip()
            if last_row is not None and pd.notna(last_row.get("playername"))
            else str(group["playername"].iloc[-1]).strip()
        )
        if not player:
            continue
        primary = _primary_league(group)
        last_observed_team = (
            str(last_row["teamname"])
            if last_row is not None and pd.notna(last_row.get("teamname"))
            else None
        )
        last_observed_league = (
            str(last_row["league"])
            if last_row is not None and pd.notna(last_row.get("league"))
            else None
        )

        current_candidates: list[tuple[pd.Timestamp, Any, str, str]] = []
        for index, row in group.iterrows():
            team = (
                str(row["teamname"])
                if pd.notna(row.get("teamname"))
                else ""
            )
            league = str(row["league"]) if pd.notna(row.get("league")) else ""
            tournament = membership_teams.get(
                team_identity_key(team),
                {},
            ).get(league)
            observed_at = pd.to_datetime(row.get("date"), errors="coerce")
            if team and league and tournament and pd.notna(observed_at):
                current_candidates.append((observed_at, index, team, tournament))

        current_row = None
        current_team = None
        current_tournament = None
        if current_candidates:
            _, current_index, current_team, current_tournament = max(
                current_candidates,
                key=lambda value: value[0],
            )
            current_row = group.loc[current_index]
        current_league = (
            str(current_row["league"])
            if current_row is not None and pd.notna(current_row.get("league"))
            else None
        )
        records[player] = {
            "player_id": (
                str(player_identity)
                if identity_source == "provider_playerid"
                else None
            ),
            "identity_source": identity_source,
            "wins": wins,
            "games": games,
            "wr": _wr(wins, games),
            "leagues": leagues,
            "primary": primary,
            "last_observed_team": last_observed_team,
            "last_observed_league": last_observed_league,
            "last_observed_date": (
                str(last_row["date"]) if last_row is not None else None
            ),
            "current_league": current_league,
            "current_tier": "tier1" if current_row is not None else None,
            "current_team": current_team,
            "current_date": str(current_row["date"]) if current_row is not None else None,
            "current_tournament": current_tournament,
            "current_affiliation_basis": (
                "observed_current_tournament_map"
                if current_row is not None
                else None
            ),
            "membership_as_of": (current_membership or {}).get("as_of")
            if current_row is not None
            else None,
            "membership_source": (current_membership or {}).get("authority")
            if current_row is not None
            else None,
            "intl": bool(group["is_international"].any()),
            "interregional": bool(group.get("is_interregional", pd.Series(dtype=bool)).any()),
        }
    return records
