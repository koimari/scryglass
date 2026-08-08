"""Series resolution primitives for L1 foundations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .common import ContractError, parse_rfc3339


_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class SeriesError(ContractError):
    """Raised for series resolution violations."""


@dataclass(frozen=True)
class SeriesCrosswalkRow:
    row_id: str
    league_id: str
    tournament_id: str
    source_id: str
    source_record_id: str
    source_snapshot_id: str
    source_snapshot_row_id: str
    source_snapshot_content_sha256: str
    source_series_id: str
    series_id: str
    participants: tuple[str, str]
    effective_from: str
    effective_to: str | None = None
    precedence: int = 1
    observed_at: str | None = None
    available_at: str | None = None
    source_updated_at: str | None = None


@dataclass(frozen=True)
class MapRecord:
    map_id: str
    league_id: str
    tournament_id: str
    participants: tuple[str, str]
    source_series_id: str | None = None
    source_id: str = ""
    source_record_id: str = ""
    source_snapshot_id: str = ""
    source_snapshot_row_id: str = ""
    source_snapshot_content_sha256: str = ""
    source_updated_at: str = ""
    observed_at: str = ""
    available_at: str = ""
    scheduled_start: str | None = None
    event_start: str | None = None
    event_end: str | None = None
    result: int | None = None
    patch_id: str | None = None
    source_updated_by: str | None = None
    season_id: str | None = None
    calendar_year: int | None = None


@dataclass(frozen=True)
class SeriesResolution:
    map_id: str
    league_id: str
    tournament_id: str
    series_id: str | None
    resolution: str
    reason: str
    source_series_id: str | None = None
    source_id: str | None = None
    conflict: bool = False
    source_snapshot_id: str | None = None
    source_snapshot_row_id: str | None = None


@dataclass(frozen=True)
class SeriesIdResolution:
    map_id: str
    series_id: str
    resolved: bool
    reason: str


@dataclass(frozen=True)
class UnresolvedMap:
    map_id: str
    league_id: str
    reason: str


@dataclass(frozen=True)
class SeriesRegistry:
    rows: tuple[SeriesCrosswalkRow, ...] = ()

    @classmethod
    def empty(cls) -> "SeriesRegistry":
        return cls(rows=())

    def append(self, row: SeriesCrosswalkRow) -> "SeriesRegistry":
        _validate_crosswalk_row(row)
        for existing in self.rows:
            if existing.row_id == row.row_id:
                raise SeriesError(f"duplicate row_id: {row.row_id}")
        return SeriesRegistry(rows=self.rows + (row,))

    def resolve(self, record: MapRecord, *, as_of: datetime | None = None) -> SeriesResolution:
        if as_of is None:
            raise SeriesError("as_of is required for series resolution")
        as_of = _coerce_utc(as_of, field_name="as_of")
        _validate_map_record(record)
        _validate_map_time_axes(record, as_of)

        candidates = [
            row
            for row in self.rows
            if row.league_id == record.league_id
            and row.tournament_id == record.tournament_id
            and row.source_id == record.source_id
            and _row_active(row, as_of)
        ]

        if not candidates:
            return SeriesResolution(
                map_id=record.map_id,
                league_id=record.league_id,
                tournament_id=record.tournament_id,
                series_id=None,
                resolution="unresolved",
                reason="no_series_crosswalk_match",
                source_series_id=None,
                source_id=record.source_id,
                conflict=True,
            )

        if record.source_series_id is not None:
            candidates = [row for row in candidates if row.source_series_id == record.source_series_id]
            if not candidates:
                return SeriesResolution(
                    map_id=record.map_id,
                    league_id=record.league_id,
                    tournament_id=record.tournament_id,
                    series_id=None,
                    resolution="unresolved",
                    reason="source_series_id_mismatch",
                    source_series_id=record.source_series_id,
                    source_id=record.source_id,
                    conflict=True,
                )

        participants = _participants_key(record.participants)
        candidates = [row for row in candidates if _participants_key(row.participants) == participants]
        if not candidates:
            return SeriesResolution(
                map_id=record.map_id,
                league_id=record.league_id,
                tournament_id=record.tournament_id,
                series_id=None,
                resolution="unresolved",
                reason="participants_mismatch",
                source_series_id=record.source_series_id,
                source_id=record.source_id,
                conflict=True,
            )

        candidates = sorted(candidates, key=_crosswalk_sort_key, reverse=True)
        best_precedence = candidates[0].precedence
        top = [row for row in candidates if row.precedence == best_precedence]

        top_groups = {
            (row.series_id, row.source_series_id, _participants_key(row.participants)) for row in top
        }
        if len(top_groups) > 1:
            return SeriesResolution(
                map_id=record.map_id,
                league_id=record.league_id,
                tournament_id=record.tournament_id,
                series_id=None,
                resolution="unresolved",
                reason="conflicting_authoritative_series",
                source_series_id=record.source_series_id,
                source_id=record.source_id,
                conflict=True,
            )

        chosen = top[0]
        return SeriesResolution(
            map_id=record.map_id,
            league_id=record.league_id,
            tournament_id=record.tournament_id,
            series_id=chosen.series_id,
            resolution="resolved",
            reason="authoritative_crosswalk_match",
            source_series_id=chosen.source_series_id,
            source_id=chosen.source_id,
            conflict=False,
            source_snapshot_id=chosen.source_snapshot_id,
            source_snapshot_row_id=chosen.source_snapshot_row_id,
        )

    def resolve_many(self, records: Iterable[MapRecord], *, as_of: datetime | None = None) -> tuple[SeriesResolution, ...]:
        return tuple(self.resolve(record, as_of=as_of) for record in records)



def filter_resolved_series(
    rows: Iterable[SeriesResolution], *, allow_unresolved_conflict: bool = False
) -> tuple[SeriesIdResolution, ...]:
    accepted: list[SeriesIdResolution] = []
    for row in rows:
        if row.series_id is None or row.resolution != "resolved":
            if row.conflict and not allow_unresolved_conflict:
                continue
            continue
        accepted.append(
            SeriesIdResolution(
                map_id=row.map_id,
                series_id=row.series_id,
                resolved=True,
                reason=row.reason,
            )
        )
    return tuple(accepted)


def unresolved_maps(rows: Iterable[SeriesResolution]) -> tuple[UnresolvedMap, ...]:
    return tuple(
        UnresolvedMap(map_id=row.map_id, league_id=row.league_id, reason=row.reason)
        for row in rows
        if row.series_id is None
    )


def require_series_resolved_for_primary(rows: Iterable[SeriesResolution]) -> tuple[SeriesIdResolution, ...]:
    rows = tuple(rows)
    unresolved = [row.map_id for row in rows if row.series_id is None or row.resolution != "resolved"]
    if unresolved:
        raise SeriesError(
            "primary inference requires resolved_series; unresolved maps: "
            + ",".join(sorted(unresolved))
        )

    return tuple(
        SeriesIdResolution(map_id=row.map_id, series_id=row.series_id or "", resolved=True, reason=row.reason)
        for row in rows
    )


def derive_calendar_year(event_start: str) -> int:
    return parse_rfc3339(event_start).year


def _row_active(row: SeriesCrosswalkRow, as_of: datetime) -> bool:
    start = parse_rfc3339(row.effective_from)
    if as_of < start:
        return False
    if row.effective_to is not None:
        if as_of > parse_rfc3339(row.effective_to):
            return False

    if row.available_at is None or row.source_updated_at is None or row.observed_at is None:
        return False

    available = parse_rfc3339(row.available_at)
    source_updated = parse_rfc3339(row.source_updated_at)
    observed = parse_rfc3339(row.observed_at)
    if available > as_of or source_updated > as_of or observed > as_of:
        return False

    return True


def _validate_map_record(record: MapRecord) -> None:
    _require_non_empty(record.map_id, "map_id")
    _require_non_empty(record.league_id, "league_id")
    _require_non_empty(record.tournament_id, "tournament_id")
    _require_non_empty(record.source_id, "source_id")
    _require_non_empty(record.source_record_id, "source_record_id")
    _validate_snapshot_identity(
        record.source_snapshot_id,
        record.source_snapshot_row_id,
        record.source_snapshot_content_sha256,
    )

    if not isinstance(record.participants, tuple) or len(record.participants) != 2:
        raise SeriesError("participants must be exactly two teams")
    if record.participants[0] == record.participants[1]:
        raise SeriesError("participants must be distinct teams")

    if record.source_series_id is not None:
        _require_non_empty(record.source_series_id, "source_series_id")

    _validate_time_value(record.source_updated_at, "source_updated_at")
    _validate_time_value(record.observed_at, "observed_at")
    _validate_time_value(record.available_at, "available_at")
    _validate_ordered_times(
        parse_rfc3339(record.source_updated_at),
        parse_rfc3339(record.observed_at),
        parse_rfc3339(record.available_at),
    )

    if record.result is not None and record.result not in {0, 1}:
        raise SeriesError("result must be 0 or 1")

    if record.season_id is None:
        raise SeriesError("season_id is required")

    if record.patch_id is not None:
        from .competitions import _validate_patch_id

        _validate_patch_id(record.patch_id)

    if record.event_start is not None:
        parse_rfc3339(record.event_start)
    if record.event_end is not None:
        parse_rfc3339(record.event_end)

    if record.calendar_year is not None and record.event_start is not None:
        event_year = derive_calendar_year(record.event_start)
        if record.calendar_year != event_year:
            raise SeriesError(
                "calendar_year must be derived from event_start year; "
                f"got {record.calendar_year} for event_year={event_year}"
            )
    if record.season_id is not None and record.event_start is not None:
        event_year = derive_calendar_year(record.event_start)
        if record.season_id == str(event_year):
            raise SeriesError(
                "season_id must be authoritative and distinct from calendar_year"
            )
    if record.calendar_year is not None and record.event_start is not None:
        event_year = derive_calendar_year(record.event_start)
        if record.calendar_year != event_year:
            raise SeriesError(
                "calendar_year must be derived from event_start year;"
                f" got {record.calendar_year} for event_year={event_year}"
            )


def _validate_map_time_axes(record: MapRecord, as_of: datetime) -> None:
    if record.scheduled_start is not None:
        scheduled = parse_rfc3339(record.scheduled_start)
        if scheduled > as_of:
            raise SeriesError(f"cannot resolve from future scheduled_start for map {record.map_id}")
    if record.event_start is not None:
        event_start = parse_rfc3339(record.event_start)
        if event_start > as_of:
            raise SeriesError(f"cannot resolve from future event_start for map {record.map_id}")
    if record.event_end is not None and parse_rfc3339(record.event_end) > as_of:
        raise SeriesError(f"cannot resolve from future event_end for map {record.map_id}")



def _validate_crosswalk_row(row: SeriesCrosswalkRow) -> None:
    _require_non_empty(row.row_id, "row_id")
    _require_non_empty(row.league_id, "league_id")
    _require_non_empty(row.tournament_id, "tournament_id")
    _require_non_empty(row.source_id, "source_id")
    _require_non_empty(row.source_record_id, "source_record_id")
    _validate_snapshot_identity(
        row.source_snapshot_id,
        row.source_snapshot_row_id,
        row.source_snapshot_content_sha256,
    )
    _require_non_empty(row.source_series_id, "source_series_id")
    _require_non_empty(row.series_id, "series_id")

    if not isinstance(row.participants, tuple) or len(row.participants) != 2 or row.participants[0] == row.participants[1]:
        raise SeriesError("participants must be exactly two distinct teams")

    if row.precedence < 0:
        raise SeriesError("precedence must be non-negative")

    parse_rfc3339(_require_non_empty(row.effective_from, "effective_from"))
    if row.effective_to is not None:
        start = parse_rfc3339(row.effective_from)
        end = parse_rfc3339(row.effective_to)
        if end < start:
            raise SeriesError("effective_to cannot be before effective_from")

    if row.observed_at is None or row.source_updated_at is None or row.available_at is None:
        raise SeriesError("observed_at, source_updated_at, and available_at are required")

    observed = parse_rfc3339(row.observed_at)
    source_updated = parse_rfc3339(row.source_updated_at)
    available = parse_rfc3339(row.available_at)
    _validate_ordered_times(source_updated, observed, available)


def _validate_time_value(value: str, field_name: str) -> None:
    parse_rfc3339(_require_non_empty(value, field_name))


def _validate_snapshot_identity(snapshot_id: str | None, snapshot_row_id: str | None, source_snapshot_content_sha256: str | None) -> None:
    _require_non_empty(snapshot_id, "source_snapshot_id")
    _require_non_empty(snapshot_row_id, "source_snapshot_row_id")
    if source_snapshot_content_sha256 is None:
        raise SeriesError("source_snapshot_content_sha256 is required")
    if not _SHA256_RE.fullmatch(source_snapshot_content_sha256):
        raise SeriesError("source_snapshot_content_sha256 must be a 64-char hex digest")


def _validate_ordered_times(
    source_updated_at: datetime,
    observed_at: datetime,
    available_at: datetime,
) -> None:
    if source_updated_at > observed_at:
        raise SeriesError("source_updated_at cannot be after observed_at")
    if available_at > observed_at:
        raise SeriesError("available_at cannot be after observed_at")


def _participants_key(participants: tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(participants))


def _crosswalk_sort_key(row: SeriesCrosswalkRow) -> tuple[int, datetime, datetime, datetime, str]:
    return (
        row.precedence,
        parse_rfc3339(row.effective_from),
        parse_rfc3339(_require_non_empty(row.source_updated_at, "source_updated_at")),
        parse_rfc3339(_require_non_empty(row.observed_at, "observed_at")),
        row.row_id,
    )


def _coerce_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise SeriesError(f"{field_name} must include timezone")
    return value.astimezone(timezone.utc)


def _require_non_empty(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise SeriesError(f"{field_name} is required")
    return value.strip()


__all__ = [
    "SeriesCrosswalkRow",
    "SeriesError",
    "SeriesIdResolution",
    "SeriesRegistry",
    "SeriesResolution",
    "UnresolvedMap",
    "MapRecord",
    "filter_resolved_series",
    "require_series_resolved_for_primary",
    "unresolved_maps",
]
