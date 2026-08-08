"""Fail-closed pre-event GRID roster candidate validation.

GRID Central Data sometimes exposes a ``series.players`` collection before a
scheduled series.  That collection is not automatically an exact active
roster: it can be empty, include substitutes, omit a side, or contain more
than five players.  This module only validates whether a response is
*eligible for independent source review*.  It never creates roster authority
and never returns ``RosterRow`` objects that a rating runner could consume.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from .common import ContractError, ROLES, canonicalize_role, parse_rfc3339, to_rfc3339


SCHEMA_VERSION = "scryglass:grid-scheduled-roster-candidate:v1"
SOURCE_ID = "grid:central_data:series"
NON_AUTHORITY_STATUS = "UNVERIFIED_PROVIDER_SCHEDULED_ASSERTION"
READY_STATUS = "CANDIDATE_READY_FOR_INDEPENDENT_SOURCE_REVIEW"
UNAVAILABLE_STATUS = "UNAVAILABLE"


class ScheduledRosterError(ContractError):
    """Raised when a scheduled-series roster probe is malformed."""


@dataclass(frozen=True)
class ScheduledRosterPlayer:
    """A normalized player assertion retained only inside a candidate result."""

    player_id: str
    player_name: str
    team_id: str
    role: str
    source_updated_at: str


@dataclass(frozen=True)
class ScheduledRosterTeam:
    """One side of a strict five-player candidate."""

    team_id: str
    team_name: str
    players: tuple[ScheduledRosterPlayer, ...]


@dataclass(frozen=True)
class ScheduledRosterCandidate:
    """A non-authorizing result from a pre-event GRID roster probe."""

    schema_version: str
    status: str
    authority_status: str
    source_id: str
    series_id: str
    scheduled_at: str | None
    observed_at: str
    series_source_updated_at: str | None
    source_payload_sha256: str
    teams: tuple[ScheduledRosterTeam, ...]
    reasons: tuple[str, ...]
    claim_ceiling: Mapping[str, bool]

    @property
    def is_ready_for_review(self) -> bool:
        """Whether the shape is suitable for a separately owned review."""

        return self.status == READY_STATUS

    @property
    def can_authorize_roster(self) -> bool:
        """Always false: this package is deliberately non-authorizing."""

        return False


def evaluate_pre_event_scheduled_roster(
    series: Mapping[str, Any],
    *,
    observed_at: datetime,
    source_payload_sha256: str,
) -> ScheduledRosterCandidate:
    """Evaluate a GRID ``series`` response without granting authority.

    The candidate is ready for independent review only when the response was
    observed before the scheduled start, the provider's series and player
    timestamps are no later than that start and no later than observation, and
    both sides contain exactly one uniquely identified player for each of the
    five canonical roles.  Any failure returns ``UNAVAILABLE`` with typed
    reasons rather than selecting a fallback or inferred player.
    """

    if not isinstance(series, Mapping):
        raise ScheduledRosterError("series response must be an object")
    observed = _coerce_utc(observed_at, "observed_at")
    if not _is_sha256(source_payload_sha256):
        raise ScheduledRosterError("source_payload_sha256 must be a 64-character lowercase hex digest")

    series_id = _text(series.get("id"), "series.id")
    scheduled_raw = series.get("startTimeScheduled")
    updated_raw = series.get("updatedAt")
    reasons: list[str] = []
    scheduled: datetime | None = None
    series_updated: datetime | None = None

    if not isinstance(scheduled_raw, str) or not scheduled_raw.strip():
        reasons.append("SERIES_SCHEDULED_START_MISSING")
    else:
        try:
            scheduled = parse_rfc3339(scheduled_raw)
        except ContractError:
            reasons.append("SERIES_SCHEDULED_START_INVALID")

    if not isinstance(updated_raw, str) or not updated_raw.strip():
        reasons.append("SERIES_SOURCE_UPDATED_AT_MISSING")
    else:
        try:
            series_updated = parse_rfc3339(updated_raw)
        except ContractError:
            reasons.append("SERIES_SOURCE_UPDATED_AT_INVALID")

    if scheduled is not None and observed > scheduled:
        reasons.append("OBSERVED_AFTER_SCHEDULED_START")
    if scheduled is not None and series_updated is not None and series_updated > scheduled:
        reasons.append("SERIES_UPDATED_AFTER_SCHEDULED_START")
    if series_updated is not None and series_updated > observed:
        reasons.append("SERIES_UPDATED_AFTER_OBSERVATION")

    teams = _parse_teams(series.get("teams"), reasons)
    team_ids = {team_id for team_id, _ in teams}
    players_by_team: dict[str, list[ScheduledRosterPlayer]] = {team_id: [] for team_id, _ in teams}
    _parse_players(
        series.get("players"),
        team_ids=team_ids,
        observed=observed,
        scheduled=scheduled,
        players_by_team=players_by_team,
        reasons=reasons,
    )

    normalized_teams: list[ScheduledRosterTeam] = []
    for team_id, team_name in teams:
        players = players_by_team.get(team_id, [])
        if len(players) != len(ROLES):
            reasons.append(f"TEAM_PLAYER_COUNT_NOT_EXACT:{team_id}:{len(players)}")
        player_ids = [player.player_id for player in players]
        if len(set(player_ids)) != len(player_ids):
            reasons.append(f"TEAM_PLAYER_IDS_NOT_UNIQUE:{team_id}")
        roles = [player.role for player in players]
        if len(roles) != len(set(roles)) or set(roles) != set(ROLES):
            reasons.append(f"TEAM_ROLE_SET_NOT_EXACT:{team_id}")
        normalized_teams.append(
            ScheduledRosterTeam(
                team_id=team_id,
                team_name=team_name,
                players=tuple(sorted(players, key=lambda player: ROLES.index(player.role))),
            )
        )

    unique_reasons = tuple(sorted(dict.fromkeys(reasons)))
    status = READY_STATUS if not unique_reasons and len(normalized_teams) == 2 else UNAVAILABLE_STATUS
    return ScheduledRosterCandidate(
        schema_version=SCHEMA_VERSION,
        status=status,
        authority_status=NON_AUTHORITY_STATUS,
        source_id=SOURCE_ID,
        series_id=series_id,
        scheduled_at=to_rfc3339(scheduled) if scheduled is not None else None,
        observed_at=to_rfc3339(observed),
        series_source_updated_at=to_rfc3339(series_updated) if series_updated is not None else None,
        source_payload_sha256=source_payload_sha256,
        teams=tuple(normalized_teams),
        reasons=unique_reasons,
        claim_ceiling={
            "current_roster": False,
            "pre_event_roster": False,
            "model_fit": False,
            "prediction": False,
            "production": False,
            "publication": False,
            "promotion": False,
            "final_holdout": False,
        },
    )


def _parse_teams(value: Any, reasons: list[str]) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        reasons.append("SERIES_TEAMS_MISSING")
        return []
    teams: list[tuple[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            reasons.append(f"TEAM_ROW_INVALID:{index}")
            continue
        base = raw.get("baseInfo")
        if not isinstance(base, Mapping):
            reasons.append(f"TEAM_BASE_INFO_MISSING:{index}")
            continue
        team_id = _optional_text(base.get("id"))
        team_name = _optional_text(base.get("name"))
        if team_id is None or team_name is None:
            reasons.append(f"TEAM_ID_OR_NAME_MISSING:{index}")
            continue
        if team_id in seen:
            reasons.append(f"TEAM_IDS_NOT_UNIQUE:{team_id}")
        seen.add(team_id)
        teams.append((team_id, team_name))
    if len(teams) != 2:
        reasons.append(f"SERIES_TEAM_COUNT_NOT_EXACT:{len(teams)}")
    return teams


def _parse_players(
    value: Any,
    *,
    team_ids: set[str],
    observed: datetime,
    scheduled: datetime | None,
    players_by_team: dict[str, list[ScheduledRosterPlayer]],
    reasons: list[str],
) -> None:
    if not isinstance(value, list):
        reasons.append("SERIES_PLAYERS_MISSING")
        return
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            reasons.append(f"PLAYER_ROW_INVALID:{index}")
            continue
        player_id = _optional_text(raw.get("id"))
        player_name = _optional_text(raw.get("nickname")) or _optional_text(raw.get("fullName"))
        team = raw.get("team")
        team_id = _optional_text(team.get("id")) if isinstance(team, Mapping) else None
        updated_raw = raw.get("updatedAt")
        if player_id is None or player_name is None or team_id is None:
            reasons.append(f"PLAYER_ID_NAME_TEAM_MISSING:{index}")
            continue
        if team_id not in team_ids:
            reasons.append(f"PLAYER_TEAM_NOT_IN_SERIES:{player_id}")
            continue
        if not isinstance(updated_raw, str) or not updated_raw.strip():
            reasons.append(f"PLAYER_SOURCE_UPDATED_AT_MISSING:{player_id}")
            continue
        try:
            updated = parse_rfc3339(updated_raw)
        except ContractError:
            reasons.append(f"PLAYER_SOURCE_UPDATED_AT_INVALID:{player_id}")
            continue
        if updated > observed:
            reasons.append(f"PLAYER_UPDATED_AFTER_OBSERVATION:{player_id}")
        if scheduled is not None and updated > scheduled:
            reasons.append(f"PLAYER_UPDATED_AFTER_SCHEDULED_START:{player_id}")

        roles = raw.get("roles")
        if not isinstance(roles, list) or len(roles) != 1:
            reasons.append(f"PLAYER_ROLE_ARITY_NOT_EXACT:{player_id}")
            continue
        role_raw = roles[0].get("name") if isinstance(roles[0], Mapping) else None
        try:
            role = canonicalize_role(_text(role_raw, f"players[{index}].roles[0].name"))
        except ContractError:
            reasons.append(f"PLAYER_ROLE_INVALID:{player_id}")
            continue
        if role not in ROLES:
            reasons.append(f"PLAYER_ROLE_INVALID:{player_id}")
            continue
        players_by_team[team_id].append(
            ScheduledRosterPlayer(
                player_id=player_id,
                player_name=player_name,
                team_id=team_id,
                role=role,
                source_updated_at=to_rfc3339(updated),
            )
        )


def _coerce_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ScheduledRosterError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _text(value: Any, field_name: str) -> str:
    result = _optional_text(value)
    if result is None:
        raise ScheduledRosterError(f"{field_name} is required")
    return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


__all__ = [
    "NON_AUTHORITY_STATUS",
    "READY_STATUS",
    "SCHEMA_VERSION",
    "SOURCE_ID",
    "ScheduledRosterCandidate",
    "ScheduledRosterError",
    "ScheduledRosterPlayer",
    "ScheduledRosterTeam",
    "UNAVAILABLE_STATUS",
    "evaluate_pre_event_scheduled_roster",
]
