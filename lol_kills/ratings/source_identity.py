"""Verified Oracle's Elixir identity joins for rating artifacts.

Source IDs are authoritative for joins. Display names remain presentation
fields. A display alias receives an ID only when the accepted source shows one
stable ID for that alias and one alias for that stable ID. Ambiguous aliases
and missing IDs stay empty. The coverage receipt records the reason.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from lol_kills.etl.competition import team_identity_key


PLAYER_ID_PREFIX = "oe:player:"
TEAM_ID_PREFIX = "oe:team:"
IDENTITY_SCHEMA_VERSION = "scryglass:rating-source-identity:v1"
_SPACE_RE = re.compile(r"\s+")


class RatingIdentityError(ValueError):
    """Raised when an identity frame cannot be interpreted safely."""


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def stable_oe_id(value: Any, prefix: str) -> str | None:
    """Return a non-empty stable OE ID with the expected prefix."""

    text = _text(value)
    if not text.startswith(prefix) or not text[len(prefix) :].strip():
        return None
    return text


def player_display_key(value: Any) -> str:
    """Normalize a player display alias for an exact, case-insensitive join."""

    text = unicodedata.normalize("NFKC", _text(value))
    return _SPACE_RE.sub(" ", text).casefold()


def _id_column(frame: pd.DataFrame | None, candidates: Sequence[str]) -> str | None:
    if frame is None:
        return None
    return next((column for column in candidates if column in frame.columns), None)


def _name_column(frame: pd.DataFrame | None, candidates: Sequence[str]) -> str | None:
    if frame is None:
        return None
    return next((column for column in candidates if column in frame.columns), None)


def _identity_mapping(
    frame: pd.DataFrame | None,
    *,
    name_candidates: Sequence[str],
    id_candidates: Sequence[str],
    prefix: str,
    key_fn,
    label: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    name_column = _name_column(frame, name_candidates)
    id_column = _id_column(frame, id_candidates)
    if frame is None or name_column is None or id_column is None:
        return {}, {
            "source_rows": int(len(frame)) if frame is not None else 0,
            "source_rows_with_name": 0,
            "source_rows_with_stable_id": 0,
            "unique_display_keys": 0,
            "unique_stable_ids": 0,
            "mapped_display_keys": 0,
            "mapped_rows": 0,
            "missing_id_rows": int(len(frame)) if frame is not None else 0,
            "invalid_id_rows": 0,
            "ambiguous_display_keys": {},
            "ambiguous_stable_ids": {},
            "status": "unavailable",
            "id_column": id_column,
            "name_column": name_column,
        }

    alias_to_ids: dict[str, set[str]] = {}
    id_to_aliases: dict[str, set[str]] = {}
    source_rows_with_name = 0
    source_rows_with_stable_id = 0
    missing_id_rows = 0
    invalid_id_rows = 0
    for raw_name, raw_id in zip(frame[name_column], frame[id_column]):
        name = _text(raw_name)
        alias = key_fn(name)
        if not alias:
            continue
        source_rows_with_name += 1
        stable_id = stable_oe_id(raw_id, prefix)
        if stable_id is None:
            if _text(raw_id):
                invalid_id_rows += 1
            else:
                missing_id_rows += 1
            continue
        source_rows_with_stable_id += 1
        alias_to_ids.setdefault(alias, set()).add(stable_id)
        id_to_aliases.setdefault(stable_id, set()).add(alias)

    ambiguous_aliases = {
        alias: sorted(ids)
        for alias, ids in sorted(alias_to_ids.items())
        if len(ids) != 1
    }
    ambiguous_ids = {
        stable_id: sorted(aliases)
        for stable_id, aliases in sorted(id_to_aliases.items())
        if len(aliases) != 1
    }
    mapping = {
        alias: next(iter(ids))
        for alias, ids in alias_to_ids.items()
        if len(ids) == 1 and next(iter(ids)) not in ambiguous_ids
    }
    mapped_rows = 0
    for raw_name, raw_id in zip(frame[name_column], frame[id_column]):
        alias = key_fn(raw_name)
        stable_id = stable_oe_id(raw_id, prefix)
        if alias in mapping and stable_id == mapping[alias]:
            mapped_rows += 1
    total_rows = int(len(frame))
    status = "complete" if total_rows and mapped_rows == total_rows else "partial"
    if not total_rows:
        status = "unavailable"
    details = {
        "source_rows": total_rows,
        "source_rows_with_name": source_rows_with_name,
        "source_rows_with_stable_id": source_rows_with_stable_id,
        "unique_display_keys": len(alias_to_ids),
        "unique_stable_ids": len(id_to_aliases),
        "mapped_display_keys": len(mapping),
        "mapped_rows": mapped_rows,
        "missing_id_rows": missing_id_rows,
        "invalid_id_rows": invalid_id_rows,
        "ambiguous_display_keys": ambiguous_aliases,
        "ambiguous_stable_ids": ambiguous_ids,
        "status": status,
        "id_column": id_column,
        "name_column": name_column,
        "label": label,
    }
    return mapping, details


def _concat_sources(
    first: pd.DataFrame | None,
    second: pd.DataFrame | None,
    *,
    columns: Sequence[str],
) -> pd.DataFrame:
    frames = []
    for frame in (first, second):
        if frame is None or frame.empty:
            continue
        present = [column for column in columns if column in frame.columns]
        if present:
            frames.append(frame[present].copy())
    if not frames:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(frames, ignore_index=True, sort=False)


@dataclass(frozen=True)
class RatingIdentityMaps:
    """Verified display-to-ID maps and their source-bound coverage receipt."""

    player_by_name: Mapping[str, str]
    team_by_name: Mapping[str, str]
    coverage: Mapping[str, Any]

    def player_id_for(self, value: Any) -> str | None:
        return self.player_by_name.get(player_display_key(value))

    def team_id_for(self, value: Any) -> str | None:
        return self.team_by_name.get(team_identity_key(value))


def build_rating_identity_maps(
    players: pd.DataFrame | None,
    teams: pd.DataFrame | None = None,
    *,
    source_identity_sha256: str | None = None,
    source_game_count: int | None = None,
) -> RatingIdentityMaps:
    """Build fail-closed one-to-one joins from an accepted OE source.

    ``playerid`` and ``teamid`` are the preferred OE columns. The underscored
    spellings are accepted for small fixtures and normalized source frames.
    """

    player_frame = players if players is not None else pd.DataFrame()
    team_frame = teams if teams is not None else pd.DataFrame()
    player_by_name, player_details = _identity_mapping(
        player_frame,
        name_candidates=("playername", "player"),
        id_candidates=("playerid", "player_id"),
        prefix=PLAYER_ID_PREFIX,
        key_fn=player_display_key,
        label="player",
    )
    team_source = _concat_sources(
        team_frame,
        player_frame,
        columns=("teamname", "team", "teamid", "team_id"),
    )
    team_by_name, team_details = _identity_mapping(
        team_source,
        name_candidates=("teamname", "team"),
        id_candidates=("teamid", "team_id"),
        prefix=TEAM_ID_PREFIX,
        key_fn=team_identity_key,
        label="team",
    )
    source_identity = str(source_identity_sha256 or "")
    coverage: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "source_identity_sha256": source_identity or None,
        "source_game_count": int(source_game_count) if source_game_count is not None else None,
        "player": player_details,
        "team": team_details,
    }
    statuses = {str(player_details.get("status")), str(team_details.get("status"))}
    coverage["status"] = (
        "complete"
        if statuses == {"complete"}
        else "unavailable"
        if statuses == {"unavailable"}
        else "partial"
    )
    return RatingIdentityMaps(player_by_name, team_by_name, coverage)


def attach_team_ids(frame: pd.DataFrame, identities: RatingIdentityMaps) -> pd.DataFrame:
    """Attach ``team_id`` to a team snapshot or team-keyed frame."""

    output = frame.copy()
    if output.empty:
        output["team_id"] = pd.Series(dtype="string")
        return output
    source = "team_key" if "team_key" in output.columns else "team"
    if source in output.columns:
        team_ids = output[source].map(identities.team_id_for).astype("string")
        duplicate_ids = team_ids.notna() & team_ids.duplicated(keep=False)
        team_ids.loc[duplicate_ids] = pd.NA
        output["team_id"] = team_ids
    else:
        output["team_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
    return output


def attach_player_ids(frame: pd.DataFrame, identities: RatingIdentityMaps) -> pd.DataFrame:
    """Attach ``player_id`` and a stable ID for the current team."""

    output = frame.copy()
    if output.empty:
        output["player_id"] = pd.Series(dtype="string")
        output["team_id"] = pd.Series(dtype="string")
        return output
    if "player" in output.columns:
        player_ids = output["player"].map(identities.player_id_for).astype("string")
        duplicate_ids = player_ids.notna() & player_ids.duplicated(keep=False)
        player_ids.loc[duplicate_ids] = pd.NA
        output["player_id"] = player_ids
    else:
        output["player_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
    team_column = "last_team" if "last_team" in output.columns else "team"
    if team_column in output.columns:
        output["team_id"] = output[team_column].map(identities.team_id_for).astype("string")
    else:
        output["team_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
    return output


def attach_sequential_team_ids(frame: pd.DataFrame, identities: RatingIdentityMaps) -> pd.DataFrame:
    """Attach side-specific team IDs to a per-map team rating frame."""

    output = frame.copy()
    for name in ("blue_team", "red_team"):
        if name in output.columns:
            output[f"{name}_id"] = output[name].map(identities.team_id_for).astype("string")
    return output


def _source_side_ids(
    players: pd.DataFrame,
    identities: RatingIdentityMaps,
) -> dict[tuple[str, str], dict[str, Any]]:
    if players is None or players.empty or "playername" not in players.columns:
        return {}
    game_column = "game_uid" if "game_uid" in players.columns else "gameid"
    if game_column not in players.columns or "side" not in players.columns:
        return {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for (game, side), group in players.groupby([game_column, "side"], sort=False, dropna=False):
        key = (str(game), str(side).title())
        ids: list[str] = []
        team_ids: set[str] = set()
        valid = True
        for _, row in group.iterrows():
            player_id = stable_oe_id(row.get("playerid", row.get("player_id")), PLAYER_ID_PREFIX)
            mapped = identities.player_id_for(row.get("playername"))
            if player_id is None or (mapped is not None and mapped != player_id):
                valid = False
            else:
                ids.append(player_id)
            team_id = stable_oe_id(row.get("teamid", row.get("team_id")), TEAM_ID_PREFIX)
            if team_id is not None:
                team_ids.add(team_id)
        if len(ids) != len(set(ids)) or len(team_ids) != 1:
            valid = False
        rows[key] = {
            "player_ids": ids if valid and ids else None,
            "team_id": next(iter(team_ids)) if valid and len(team_ids) == 1 else None,
        }
    return rows


def attach_player_rating_ids(
    frame: pd.DataFrame,
    players: pd.DataFrame,
    identities: RatingIdentityMaps,
) -> pd.DataFrame:
    """Attach stable side/team IDs to ``player_ratings.parquet`` rows."""

    output = frame.copy()
    side_ids = _source_side_ids(players, identities)
    game_column = "game_uid" if "game_uid" in output.columns else "gameid"
    if game_column not in output.columns:
        output["blue_team_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
        output["red_team_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
        output["blue_player_ids"] = None
        output["red_player_ids"] = None
        return output
    output["blue_team_id"] = [
        side_ids.get((str(game), "Blue"), {}).get("team_id")
        for game in output[game_column]
    ]
    output["red_team_id"] = [
        side_ids.get((str(game), "Red"), {}).get("team_id")
        for game in output[game_column]
    ]
    output["blue_player_ids"] = [
        side_ids.get((str(game), "Blue"), {}).get("player_ids")
        for game in output[game_column]
    ]
    output["red_player_ids"] = [
        side_ids.get((str(game), "Red"), {}).get("player_ids")
        for game in output[game_column]
    ]
    return output


def attach_weekly_ids(
    payload: Mapping[str, Any],
    identities: RatingIdentityMaps,
    *,
    kind: str,
) -> dict[str, Any]:
    """Add stable IDs inside existing weekly rank records."""

    output = dict(payload)
    by_key_name = "by_player" if kind == "player" else "by_team"
    source = output.get(by_key_name)
    if not isinstance(source, Mapping):
        return output
    by_key: dict[str, Any] = {}
    for display, value in source.items():
        item = dict(value) if isinstance(value, Mapping) else value
        stable_id = (
            identities.player_id_for(display)
            if kind == "player"
            else identities.team_id_for(display)
        )
        if isinstance(item, dict):
            item["player_id" if kind == "player" else "team_id"] = stable_id
        by_key[str(display)] = item
    output[by_key_name] = by_key
    output["identity"] = {
        "kind": kind,
        "source_identity_sha256": identities.coverage.get("source_identity_sha256"),
        "mapping_status": identities.coverage.get(kind, {}).get("status"),
    }
    return output


def attach_record_ids(
    records: Mapping[str, Mapping[str, Any]],
    identities: RatingIdentityMaps,
    *,
    kind: str,
) -> dict[str, dict[str, Any]]:
    """Attach stable IDs to display-keyed team/player records."""

    output: dict[str, dict[str, Any]] = {}
    for display, raw in records.items():
        item = dict(raw)
        item["player_id" if kind == "player" else "team_id"] = (
            identities.player_id_for(display)
            if kind == "player"
            else identities.team_id_for(display)
        )
        output[display] = item
    return output
