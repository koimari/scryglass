"""Fail-closed candidate validation for a Cito roster-history response.

Cito documents player/team membership intervals with ``startedAt``, ``endedAt``,
and ``role`` fields.  This adapter only checks whether a payload could be sent
to independent source review.  It never creates ``RosterRow`` objects and
never grants roster, model, prediction, or publication authority.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

from .common import ContractError, ROLES, canonicalize_role, parse_rfc3339, to_rfc3339


SCHEMA_VERSION = "scryglass:cito-roster-history-candidate:v1"
SOURCE_ID = "cito:lol:team-roster-history"
NON_AUTHORITY_STATUS = "UNVERIFIED_PROVIDER_ROSTER_HISTORY_ASSERTION"
READY_STATUS = "CANDIDATE_READY_FOR_INDEPENDENT_SOURCE_REVIEW"
UNAVAILABLE_STATUS = "UNAVAILABLE"


class CitoRosterError(ContractError):
    """Raised when a Cito-like roster-history payload is malformed."""


@dataclass(frozen=True)
class CitoRosterPlayer:
    """One active interval retained inside a non-authorizing candidate."""

    player_id: str
    player_name: str
    team_slug: str
    team_name: str
    role: str
    started_at: str
    ended_at: str | None


@dataclass(frozen=True)
class CitoRosterCandidate:
    """A strict five-player candidate that still requires independent review."""

    schema_version: str
    status: str
    authority_status: str
    source_id: str
    team_slug: str
    team_name: str | None
    event_start: str
    observed_at: str
    source_updated_at: str
    source_payload_sha256: str
    players: tuple[CitoRosterPlayer, ...]
    reasons: tuple[str, ...]
    claim_ceiling: Mapping[str, bool]

    @property
    def is_ready_for_review(self) -> bool:
        return self.status == READY_STATUS

    @property
    def can_authorize_roster(self) -> bool:
        return False


def evaluate_cito_team_roster(
    payloads: Iterable[Mapping[str, Any]],
    *,
    team_slug: str,
    event_start: datetime,
    observed_at: datetime,
    source_updated_at: datetime,
    source_payload_sha256: str,
    team_name: str | None = None,
) -> CitoRosterCandidate:
    """Evaluate Cito-style player history rows for one scheduled team.

    The caller must supply the provider's source-update and retrieval times.
    Both must precede the scheduled event.  Only membership intervals active at
    ``event_start`` are retained, and the result is ready for review only when
    exactly one player occupies each canonical role.
    """

    slug = _text(team_slug, "team_slug")
    event = _coerce_utc(event_start, "event_start")
    observed = _coerce_utc(observed_at, "observed_at")
    updated = _coerce_utc(source_updated_at, "source_updated_at")
    if not _is_sha256(source_payload_sha256):
        raise CitoRosterError("source_payload_sha256 must be a 64-character lowercase hex digest")

    reasons: list[str] = []
    if observed >= event:
        reasons.append("OBSERVED_AT_NOT_BEFORE_EVENT_START")
    if updated > observed:
        reasons.append("SOURCE_UPDATED_AFTER_OBSERVATION")
    if updated >= event:
        reasons.append("SOURCE_UPDATED_AT_NOT_BEFORE_EVENT_START")

    active: list[CitoRosterPlayer] = []
    raw_payloads = list(payloads) if not isinstance(payloads, (str, bytes, bytearray)) else []
    if not raw_payloads:
        reasons.append("PLAYER_HISTORY_PAYLOADS_MISSING")

    for index, raw in enumerate(raw_payloads):
        if not isinstance(raw, Mapping):
            reasons.append(f"PLAYER_HISTORY_ROW_INVALID:{index}")
            continue
        if raw.get("success") is False:
            reasons.append(f"PLAYER_HISTORY_ROW_UNSUCCESSFUL:{index}")
            continue
        data = raw.get("data", raw)
        if not isinstance(data, Mapping):
            reasons.append(f"PLAYER_HISTORY_DATA_INVALID:{index}")
            continue
        player_id = _optional_text(data.get("playerId"))
        player_name = _optional_text(data.get("playerName"))
        memberships = data.get("teams")
        if player_id is None or player_name is None:
            reasons.append(f"PLAYER_ID_OR_NAME_MISSING:{index}")
            continue
        if not isinstance(memberships, list):
            reasons.append(f"PLAYER_TEAM_HISTORY_MISSING:{player_id}")
            continue
        for membership_index, membership in enumerate(memberships):
            if not isinstance(membership, Mapping):
                reasons.append(f"TEAM_HISTORY_ROW_INVALID:{player_id}:{membership_index}")
                continue
            membership_slug = _optional_text(membership.get("teamSlug"))
            if membership_slug != slug:
                continue
            membership_name = _optional_text(membership.get("teamName"))
            role_raw = membership.get("role")
            started_raw = membership.get("startedAt")
            ended_raw = membership.get("endedAt")
            if membership_name is None:
                reasons.append(f"TEAM_NAME_MISSING:{player_id}")
                continue
            try:
                role = canonicalize_role(_text(role_raw, f"teams[{membership_index}].role"))
            except ContractError:
                reasons.append(f"PLAYER_ROLE_INVALID:{player_id}")
                continue
            try:
                started = parse_rfc3339(_text(started_raw, f"teams[{membership_index}].startedAt"))
                ended = parse_rfc3339(ended_raw) if ended_raw is not None else None
            except ContractError:
                reasons.append(f"TEAM_INTERVAL_INVALID:{player_id}")
                continue
            if ended is not None and ended <= started:
                reasons.append(f"TEAM_INTERVAL_NOT_FORWARD:{player_id}")
                continue
            if team_name is not None and membership_name != team_name:
                reasons.append(f"TEAM_NAME_CONFLICT:{slug}")
            if started <= event and (ended is None or event < ended):
                active.append(
                    CitoRosterPlayer(
                        player_id=player_id,
                        player_name=player_name,
                        team_slug=slug,
                        team_name=membership_name,
                        role=role,
                        started_at=to_rfc3339(started),
                        ended_at=to_rfc3339(ended) if ended is not None else None,
                    )
                )

    if not active:
        reasons.append("NO_ACTIVE_TEAM_MEMBERSHIP_AT_EVENT_START")
    player_ids = [player.player_id for player in active]
    if len(set(player_ids)) != len(player_ids):
        reasons.append(f"TEAM_PLAYER_IDS_NOT_UNIQUE:{slug}")
    roles = [player.role for player in active]
    if len(active) != len(ROLES):
        reasons.append(f"TEAM_PLAYER_COUNT_NOT_EXACT:{slug}:{len(active)}")
    if len(set(roles)) != len(roles) or set(roles) != set(ROLES):
        reasons.append(f"TEAM_ROLE_SET_NOT_EXACT:{slug}")

    unique_reasons = tuple(sorted(dict.fromkeys(reasons)))
    status = READY_STATUS if not unique_reasons else UNAVAILABLE_STATUS
    return CitoRosterCandidate(
        schema_version=SCHEMA_VERSION,
        status=status,
        authority_status=NON_AUTHORITY_STATUS,
        source_id=SOURCE_ID,
        team_slug=slug,
        team_name=team_name,
        event_start=to_rfc3339(event),
        observed_at=to_rfc3339(observed),
        source_updated_at=to_rfc3339(updated),
        source_payload_sha256=source_payload_sha256,
        players=tuple(sorted(active, key=lambda player: ROLES.index(player.role))),
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


def _coerce_utc(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise CitoRosterError(f"{field_name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _text(value: Any, field_name: str) -> str:
    result = _optional_text(value)
    if result is None:
        raise CitoRosterError(f"{field_name} is required")
    return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


__all__ = [
    "CitoRosterCandidate",
    "CitoRosterError",
    "CitoRosterPlayer",
    "NON_AUTHORITY_STATUS",
    "READY_STATUS",
    "SCHEMA_VERSION",
    "SOURCE_ID",
    "UNAVAILABLE_STATUS",
    "evaluate_cito_team_roster",
]
