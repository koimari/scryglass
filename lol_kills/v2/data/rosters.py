"""Exact active roster registry and resolver foundations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable, Mapping

from .common import (
    ContractError,
    ContractTimePoint,
    ROLES,
    canonicalize_role,
    enforce_as_of_order,
    parse_rfc3339,
    to_rfc3339,
)


class RosterError(ContractError):
    """Raised when roster rows or resolution inputs are invalid."""


class AmbiguousRosterError(RosterError):
    """Raised when active roster cannot be resolved uniquely."""


class RosterUnavailableError(RosterError):
    """Raised when no valid active roster exists."""


DraftRoleTuple = tuple[str, str, str, str, str]


@dataclass(frozen=True)
class RosterRow:
    """Single source assertion that a player can occupy one roster slot."""

    row_id: str
    roster_id: str
    organization_id: str
    organization_name: str
    role: str
    player_id: str
    player_name: str
    source_id: str
    source_name: str
    source_record_id: str
    source_snapshot_id: str
    source_snapshot_row_id: str
    source_snapshot_content_sha256: str
    effective_from: str = "1970-01-01T00:00:00Z"
    effective_to: str | None = None
    precedence: int = 0
    source_updated_at: str | None = None
    observed_at: str | None = None
    available_at: str | None = None
    is_substitute: bool = False
    is_provisional: bool = False


@dataclass(frozen=True)
class RosterResolution:
    """Resolved active roster with validation metadata for a request."""

    status: str
    organization_id: str
    as_of: str
    roster_id: str | None
    roster: DraftRoleTuple | None
    player_ids: tuple[str, ...] | None
    role_to_player_id: Mapping[str, str] | None
    roles_exact: bool
    hypothetical: bool
    errors: tuple[str, ...]
    source_rows: tuple[RosterRow, ...] = ()
    source_row_ids: tuple[str, ...] = ()

    def is_ok(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True)
class RosterRegistry:
    """Append-only roster row registry with fail-closed exact resolution."""

    rows: tuple[RosterRow, ...] = ()

    @classmethod
    def empty(cls) -> "RosterRegistry":
        return cls(rows=())

    def append(self, row: RosterRow) -> "RosterRegistry":
        if not row.row_id:
            raise RosterError("row_id required")
        _validate_roster_row(row, require_evidence=True)
        for existing in self.rows:
            if existing.row_id == row.row_id:
                raise RosterError(f"duplicate row_id: {row.row_id}")
        return RosterRegistry(rows=self.rows + (row,))

    def resolve_exact_roster(
        self,
        organization_id: str,
        *,
        as_of: datetime,
        fail_closed: bool = True,
        allow_hypothetical: bool = False,
        hypothetical_rows: Iterable[RosterRow] | None = None,
    ) -> RosterResolution:
        as_of = _coerce_utc(as_of, field_name="as_of")
        organization_key = _require_non_empty(organization_id, "organization_id")

        if hypothetical_rows is not None:
            return _resolve_rows(
                organization_key,
                as_of=as_of,
                rows=tuple(hypothetical_rows),
                include_substitutes=False,
                fail_closed=fail_closed,
                hypothetical=True,
                require_evidence=False,
            )

        candidates = [
            row
            for row in self.rows
            if row.organization_id == organization_key and not row.is_substitute and _row_active(row, as_of)
        ]
        if not candidates:
            if fail_closed and not allow_hypothetical:
                raise RosterUnavailableError(f"no active roster rows for organization={organization_key}")
            return _unavailable(
                organization_key,
                as_of,
                message="no active roster rows for organization",
                hypothetical=False,
            )

        return _resolve_rows(
            organization_key,
            as_of=as_of,
            rows=tuple(candidates),
            include_substitutes=False,
            fail_closed=fail_closed,
            hypothetical=False,
            require_evidence=True,
        )

    def resolve_hypothetical_roster(
        self,
        organization_id: str,
        roster_id: str,
        role_to_player: Mapping[str, tuple[str, str]],
        *,
        as_of: datetime,
        fail_closed: bool = True,
    ) -> RosterResolution:
        as_of = _coerce_utc(as_of, field_name="as_of")
        organization_key = _require_non_empty(organization_id, "organization_id")
        roster_key = _require_non_empty(roster_id, "roster_id")

        if not role_to_player:
            raise RosterError("hypothetical roster requires role_to_player mapping")

        normalized_roles = tuple(_normalize_role(role) for role in role_to_player)
        if len(set(normalized_roles)) != len(ROLES):
            missing = tuple(role for role in ROLES if role not in set(normalized_roles))
            extra = tuple(sorted(set(normalized_roles) - set(ROLES)))
            message = "hypothetical roster requires exactly one player per exact role"
            if missing:
                message = f"{message}; missing={','.join(missing)}"
            if extra:
                message = f"{message}; invalid={','.join(extra)}"
            if fail_closed:
                raise RosterError(message)
            return _unavailable(organization_key, as_of, message=message, hypothetical=True)

        rows: list[RosterRow] = []
        for role, payload in role_to_player.items():
            normalized_role = _normalize_role(role)
            if not isinstance(payload, tuple) or len(payload) != 2:
                raise RosterError(
                    f"hypothetical roster role {normalized_role} must be (player_id, player_name)"
                )
            player_id, player_name = payload
            rows.append(
                RosterRow(
                    row_id=_stable_row_id(organization_key, roster_key, normalized_role, player_id),
                    roster_id=roster_key,
                    organization_id=organization_key,
                    organization_name=organization_key,
                    role=normalized_role,
                    player_id=_require_non_empty(player_id, "player_id"),
                    player_name=_require_non_empty(player_name, "player_name"),
                    source_id="hypothetical",
                    source_name="hypothetical",
                    source_record_id=f"hypo:{roster_key}:{normalized_role}",
                    source_snapshot_id=f"hypo:{roster_key}",
                    source_snapshot_row_id=f"hypo-row:{roster_key}:{normalized_role}",
                    source_snapshot_content_sha256="0" * 64,
                    effective_from=to_rfc3339(as_of),
                    effective_to=None,
                    precedence=999_999,
                    source_updated_at=to_rfc3339(as_of),
                    observed_at=to_rfc3339(as_of),
                    available_at=to_rfc3339(as_of),
                    is_substitute=False,
                    is_provisional=True,
                )
            )

        return _resolve_rows(
            organization_key,
            as_of=as_of,
            rows=tuple(rows),
            include_substitutes=False,
            fail_closed=fail_closed,
            hypothetical=True,
            require_evidence=False,
        )


def _stable_row_id(organization_id: str, roster_id: str, role: str, player_id: str) -> str:
    return f"scryglass:roster-row:{organization_id}:{roster_id}:{role}:{player_id}"


def _resolve_rows(
    organization_id: str,
    *,
    as_of: datetime,
    rows: tuple[RosterRow, ...],
    include_substitutes: bool,
    fail_closed: bool,
    hypothetical: bool = False,
    require_evidence: bool = True,
) -> RosterResolution:
    if not rows:
        if fail_closed:
            raise RosterUnavailableError(f"no rows for organization={organization_id}")
        return _unavailable(
            organization_id,
            as_of,
            message="no roster rows supplied",
            hypothetical=hypothetical,
        )

    normalized_rows: list[RosterRow] = []
    for row in rows:
        _validate_roster_row(row, require_evidence=require_evidence)
        if row.is_substitute and not include_substitutes:
            continue
        normalized_rows.append(row)

    if not normalized_rows:
        if fail_closed:
            raise RosterUnavailableError(
                f"no active non-substitute roster rows for organization={organization_id}"
            )
        return _unavailable(
            organization_id,
            as_of,
            message="only substitute rows available",
            hypothetical=hypothetical,
        )

    by_role: dict[str, list[RosterRow]] = {role: [] for role in ROLES}
    for row in normalized_rows:
        normalized_role = _normalize_role(row.role, row=row)
        if _row_filter_by_as_of(row, as_of):
            by_role[normalized_role].append(row)

    missing_roles = tuple(role for role in ROLES if not by_role[role])
    if missing_roles:
        message = f"missing required roles: {','.join(missing_roles)}"
        if fail_closed and not hypothetical:
            raise AmbiguousRosterError(message)
        return _unavailable(
            organization_id,
            as_of,
            message,
            hypothetical,
        )

    selected: list[RosterRow] = []
    for role in ROLES:
        candidates = sorted(by_role[role], key=_roster_sort_key, reverse=True)
        if len(candidates) == 0:
            message = f"missing required role: {role}"
            if fail_closed and not hypothetical:
                raise AmbiguousRosterError(message)
            return _unavailable(organization_id, as_of, message=message, hypothetical=hypothetical)

        top = candidates[0]
        top_key = _roster_sort_key(top)[:-1]
        tie_players = [
            c.player_id
            for c in candidates
            if _roster_sort_key(c)[:-1] == top_key
        ]
        tie_players = tuple(sorted(dict.fromkeys(tie_players)))
        if len(tie_players) > 1:
            message = f"ambiguous role assignment: role={role}; players={','.join(tie_players)}"
            if fail_closed and not hypothetical:
                raise AmbiguousRosterError(message)
            return _unavailable(organization_id, as_of, message=message, hypothetical=hypothetical)
        selected.append(top)

    roster_ids = {row.roster_id for row in selected}
    if len(roster_ids) != 1:
        message = "inconsistent roster_id across resolved roles"
        if fail_closed and not hypothetical:
            raise AmbiguousRosterError(message)
        return _unavailable(organization_id, as_of, message, hypothetical=hypothetical)

    player_ids = tuple(row.player_id for row in selected)
    if len(set(player_ids)) != len(player_ids):
        message = "non-unique players in selected roster"
        if fail_closed and not hypothetical:
            raise AmbiguousRosterError(message)
        return _unavailable(organization_id, as_of, message, hypothetical=hypothetical)

    role_to_player = {row.role: row.player_id for row in selected}
    if len(role_to_player) != len(ROLES):
        message = "incomplete role coverage"
        if fail_closed and not hypothetical:
            raise AmbiguousRosterError(message)
        return _unavailable(organization_id, as_of, message, hypothetical=hypothetical)

    role_tuple = tuple(role_to_player[role] for role in ROLES)
    return RosterResolution(
        status="ok",
        organization_id=organization_id,
        as_of=to_rfc3339(as_of),
        roster_id=next(iter(roster_ids)),
        roster=role_tuple,
        player_ids=player_ids,
        role_to_player_id=role_to_player,
        roles_exact=True,
        hypothetical=hypothetical,
        errors=(),
        source_rows=tuple(selected),
        source_row_ids=tuple(sorted(row.source_snapshot_row_id for row in selected)),
    )


def _unavailable(
    organization_id: str,
    as_of: datetime,
    message: str,
    hypothetical: bool,
) -> RosterResolution:
    return RosterResolution(
        status="unavailable",
        organization_id=organization_id,
        as_of=to_rfc3339(as_of),
        roster_id=None,
        roster=None,
        player_ids=None,
        role_to_player_id=None,
        roles_exact=False,
        hypothetical=hypothetical,
        errors=(message,),
        source_rows=(),
        source_row_ids=(),
    )


def _row_filter_by_as_of(row: RosterRow, as_of: datetime) -> bool:
    effective_from = parse_rfc3339(_require_non_empty(row.effective_from, "effective_from"))
    if as_of < effective_from:
        return False
    if row.effective_to is not None:
        if as_of > parse_rfc3339(row.effective_to):
            return False
    return True


def _row_active(row: RosterRow, as_of: datetime) -> bool:
    try:
        _validate_roster_row(row, require_evidence=True, as_of=as_of)
    except RosterError:
        return False

    return _row_filter_by_as_of(row, as_of)


def _validate_roster_row(row: RosterRow, *, require_evidence: bool, as_of: datetime | None = None) -> None:
    if not row.row_id:
        raise RosterError("row_id required")
    _require_non_empty(row.roster_id, "roster_id")
    _require_non_empty(row.organization_id, "organization_id")
    _require_non_empty(row.organization_name, "organization_name")
    _normalize_role(row.role, row=row)
    _require_non_empty(row.player_id, "player_id")
    _require_non_empty(row.player_name, "player_name")
    _require_non_empty(row.source_id, "source_id")
    _require_non_empty(row.source_name, "source_name")
    _require_non_empty(row.source_record_id, "source_record_id")
    _require_non_empty(row.source_snapshot_id, "source_snapshot_id")
    _require_non_empty(row.source_snapshot_row_id, "source_snapshot_row_id")
    _require_non_empty(row.source_snapshot_content_sha256, "source_snapshot_content_sha256")

    if row.precedence < 0:
        raise RosterError("precedence must be non-negative")
    if not _is_sha256(row.source_snapshot_content_sha256):
        raise RosterError("source_snapshot_content_sha256 must be 64-char hex")
    _validate_role_id(row.role)

    effective_from = parse_rfc3339(_require_non_empty(row.effective_from, "effective_from"))
    if row.effective_to is not None:
        effective_to = parse_rfc3339(row.effective_to)
        if effective_to < effective_from:
            raise RosterError("effective_to cannot be before effective_from")

    if not require_evidence:
        return

    source_updated = _require_non_empty(row.source_updated_at, "source_updated_at")
    observed = _require_non_empty(row.observed_at, "observed_at")
    available = _require_non_empty(row.available_at, "available_at")
    source_updated_ts = parse_rfc3339(source_updated)
    observed_ts = parse_rfc3339(observed)
    available_ts = parse_rfc3339(available)

    if as_of is None:
        as_of = observed_ts
    enforce_as_of_order(
        as_of=as_of,
        source_updated_at=source_updated_ts,
        observed_at=observed_ts,
        available_at=available_ts,
    )
    ContractTimePoint(
        source_updated_at=source_updated_ts,
        observed_at=observed_ts,
        available_at=available_ts,
    )


def _sort_roster_candidates(candidates: list[RosterRow]) -> list[RosterRow]:
    return sorted(candidates, key=_roster_sort_key, reverse=True)


def _roster_sort_key(row: RosterRow) -> tuple[int, datetime, datetime, datetime, str]:
    updated = parse_rfc3339(_require_non_empty(row.source_updated_at, "source_updated_at"))
    observed = parse_rfc3339(_require_non_empty(row.observed_at, "observed_at"))
    available = parse_rfc3339(_require_non_empty(row.available_at, "available_at"))
    return (
        row.precedence,
        available,
        observed,
        updated,
        row.row_id,
    )


def _normalize_role(raw_role: str, row: RosterRow | None = None) -> str:
    role = canonicalize_role(raw_role)
    if role not in ROLES:
        if row is None:
            raise RosterError(f"invalid role: {raw_role!r}")
        raise RosterError(f"invalid role {raw_role!r} in row {row.row_id}")
    return role


def _validate_role_id(value: str) -> bool:
    return value in ROLES


def _coerce_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise RosterError(f"{field_name} must include timezone")
    return value.astimezone(timezone.utc)


def _require_non_empty(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise RosterError(f"{field_name} is required")
    return str(value).strip()


def _is_sha256(value: str) -> bool:
    return bool(re.fullmatch(r"^[a-f0-9]{64}$", value))


__all__ = [
    "AmbiguousRosterError",
    "DraftRoleTuple",
    "RosterError",
    "RosterRegistry",
    "RosterRow",
    "RosterResolution",
    "RosterUnavailableError",
]
