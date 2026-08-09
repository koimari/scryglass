"""Competition taxonomy and patch resolvers for L1 foundations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from .common import (
    ContractError,
    enforce_as_of_order,
    parse_rfc3339,
    to_rfc3339,
)


_PATCH_RE = re.compile(r"^\d{1,2}\.\d{1,2}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")


class CompetitionError(ContractError):
    """Raised for competition contract violations."""


class PatchConflictError(CompetitionError):
    """Raised when equal precedence rows conflict."""


@dataclass(frozen=True)
class CompetitionTaxonomyRow:
    row_id: str
    league_id: str
    competition_tier: str
    structurally_globally_eligible: bool
    source_id: str
    source_name: str
    source_record_id: str
    source_snapshot_id: str
    source_snapshot_row_id: str
    source_snapshot_content_sha256: str
    effective_from: str
    effective_to: str | None
    internationally_connectable: bool
    qualification_rule_id: str
    precedence: int
    observed_at: str
    source_updated_at: str
    available_at: str
    taxonomy_version: str = "unknown"


@dataclass(frozen=True)
class LeaguePatchRecord:
    row_id: str
    league_id: str
    patch_id: str
    source_id: str
    source_name: str
    source_record_id: str
    source_snapshot_id: str
    source_snapshot_row_id: str
    source_snapshot_content_sha256: str
    source_updated_at: str
    observed_at: str
    available_at: str
    announced_at: str
    is_authoritative: bool = False
    effective_from: str | None = None
    effective_to: str | None = None
    precedence: int = 0


@dataclass(frozen=True)
class MapPatchRecord:
    map_id: str
    league_id: str
    patch_id: str
    event_end: str
    is_map_complete: bool
    source_id: str
    source_name: str
    source_record_id: str
    source_snapshot_id: str
    source_snapshot_row_id: str
    source_snapshot_content_sha256: str
    source_updated_at: str
    observed_at: str
    available_at: str
    event_start: str | None = None
    result: int | None = None
    effective_from: str | None = None
    effective_to: str | None = None


@dataclass(frozen=True)
class LeaguePatchResolution:
    league_id: str
    patch_id: str | None
    status: str
    reason: str
    source_id: str | None
    freshness_state: str
    as_of: str
    as_of_conflicts: tuple[str, ...] = ()
    source_tree_conflict_ids: tuple[str, ...] = ()
    source_row_id: str | None = None
    source_snapshot_id: str | None = None
    source_snapshot_row_id: str | None = None
    source_snapshot_content_sha256: str | None = None


@dataclass(frozen=True)
class PatchRecordRegistry:
    """Append-only patch record collection."""

    rows: tuple[LeaguePatchRecord, ...] = ()

    def append(self, row: LeaguePatchRecord) -> "PatchRecordRegistry":
        _validate_patch_row(row)
        for existing in self.rows:
            if existing.row_id == row.row_id:
                raise CompetitionError(f"duplicate patch row_id: {row.row_id}")
        return PatchRecordRegistry(rows=self.rows + (row,))


@dataclass(frozen=True)
class CompetitionTaxonomy:
    """Versioned competition taxonomy registry."""

    version: str
    rows: tuple[CompetitionTaxonomyRow, ...]

    @classmethod
    def empty(cls, *, version: str = "v2.0.0") -> "CompetitionTaxonomy":
        return cls(version=version, rows=())

    def append(self, row: CompetitionTaxonomyRow) -> "CompetitionTaxonomy":
        _validate_taxonomy_row(row)
        for existing in self.rows:
            if existing.row_id == row.row_id:
                raise CompetitionError(f"duplicate row_id: {row.row_id}")
        return CompetitionTaxonomy(version=self.version, rows=self.rows + (row,))

    def active_rows(self, as_of: datetime) -> tuple[CompetitionTaxonomyRow, ...]:
        as_of = _coerce_utc(as_of, field_name="as_of")
        return tuple(
            row
            for row in self.rows
            if _is_row_active(
                effective_from=row.effective_from,
                effective_to=row.effective_to,
                available_at=row.available_at,
                source_updated_at=row.source_updated_at,
                observed_at=row.observed_at,
                as_of=as_of,
            )
        )

    def resolve_tier_profile(self, league_id: str, as_of: datetime) -> CompetitionTaxonomyRow | None:
        as_of = _coerce_utc(as_of, field_name="as_of")
        league = _require_non_empty(league_id, "league_id")
        candidates = [row for row in self.active_rows(as_of) if row.league_id == league]
        if not candidates:
            return None

        best_precedence = max(row.precedence for row in candidates)
        best = [row for row in candidates if row.precedence == best_precedence]
        if len(best) > 1:
            profiles = {
                (
                    row.competition_tier,
                    row.structurally_globally_eligible,
                    row.internationally_connectable,
                    row.qualification_rule_id,
                )
                for row in best
            }
            if len(profiles) > 1:
                raise PatchConflictError(
                    f"conflicting_tier_profile for league={league_id} at precedence={best_precedence}: {sorted(profiles)}"
                )
        return sorted(best, key=_taxonomy_sort_key)[-1]

    def resolve_league_tier(self, league_id: str, as_of: datetime) -> tuple[str, str]:
        profile = self.resolve_tier_profile(league_id, as_of)
        if profile is None:
            return "unavailable", "no_active_tier_record"
        return profile.competition_tier, profile.qualification_rule_id

    def is_internationally_connectable(self, league_id: str, as_of: datetime) -> bool:
        profile = self.resolve_tier_profile(league_id, as_of)
        return bool(profile and profile.internationally_connectable)

    def is_global_eligible(self, league_id: str, as_of: datetime) -> bool:
        profile = self.resolve_tier_profile(league_id, as_of)
        if profile is None:
            return False
        if profile.competition_tier != "tier1":
            return False
        return bool(profile.structurally_globally_eligible)

    def is_structurally_globally_eligible(self, league_id: str, as_of: datetime) -> bool:
        return self.is_global_eligible(league_id, as_of)

    def is_bridge_connected(self, league_id: str, as_of: datetime) -> bool:
        profile = self.resolve_tier_profile(league_id, as_of)
        if profile is None:
            return False
        return bool(profile.qualification_rule_id)


@dataclass(frozen=True)
class PatchResolver:
    taxonomy: CompetitionTaxonomy
    patch_records: tuple[LeaguePatchRecord, ...]
    map_records: tuple[MapPatchRecord, ...]

    def __init__(
        self,
        *,
        taxonomy: CompetitionTaxonomy,
        patch_records: Iterable[LeaguePatchRecord] = (),
        map_records: Iterable[MapPatchRecord] = (),
    ) -> None:
        object.__setattr__(self, "taxonomy", taxonomy)

        registry = PatchRecordRegistry()
        for row in tuple(patch_records):
            registry = registry.append(row)
        object.__setattr__(self, "patch_records", tuple(sorted(registry.rows, key=_patch_record_sort_key)))

        validated_map_rows: list[MapPatchRecord] = []
        for row in tuple(map_records):
            _validate_map_patch_row(row)
            validated_map_rows.append(row)
        object.__setattr__(self, "map_records", tuple(sorted(validated_map_rows, key=_map_patch_sort_key)))

    def resolve_current_patch(
        self,
        league_id: str,
        *,
        as_of: datetime,
        freshness_limit_days: int | None = 7,
        freshness_limit_seconds: int | None = None,
    ) -> LeaguePatchResolution:
        as_of = _coerce_utc(as_of, field_name="as_of")
        league = _require_non_empty(league_id, "league_id")
        freshness_days = 7 if freshness_limit_days is None else freshness_limit_days

        try:
            tier = self.taxonomy.resolve_tier_profile(league, as_of)
        except PatchConflictError as err:
            return LeaguePatchResolution(
                league_id=league,
                patch_id=None,
                status="conflict",
                reason=str(err),
                source_id=None,
                freshness_state="conflict",
                as_of=to_rfc3339(as_of),
            )

        if tier is None:
            return LeaguePatchResolution(
                league_id=league,
                patch_id=None,
                status="missing",
                reason="missing_league_tier_profile",
                source_id=None,
                freshness_state="missing",
                as_of=to_rfc3339(as_of),
            )

        authoritative = [
            row
            for row in self.patch_records
            if row.league_id == league
            and row.is_authoritative
            and _patch_record_active(row, as_of)
        ]
        if authoritative:
            chosen = _resolve_authoritative_records(authoritative)
            if chosen is None:
                return LeaguePatchResolution(
                    league_id=league,
                    patch_id=None,
                    status="conflict",
                    reason="conflicting_authoritative_patch_records",
                    source_id=None,
                    freshness_state="conflict",
                    as_of=to_rfc3339(as_of),
                    as_of_conflicts=tuple(r.row_id for r in authoritative),
                )

            freshness = _freshness_state(
                source_available=parse_rfc3339(chosen.available_at),
                as_of=as_of,
                freshness_limit_days=freshness_days,
                freshness_limit_seconds=freshness_limit_seconds,
            )
            return LeaguePatchResolution(
                league_id=league,
                patch_id=chosen.patch_id,
                status="authoritative",
                reason="official patch source",
                source_id=chosen.source_id,
                freshness_state=freshness,
                as_of=to_rfc3339(as_of),
                source_row_id=chosen.row_id,
                source_snapshot_id=chosen.source_snapshot_id,
                source_snapshot_row_id=chosen.source_snapshot_row_id,
                source_snapshot_content_sha256=chosen.source_snapshot_content_sha256,
            )

        map_candidates = [
            row
            for row in self.map_records
            if row.league_id == league
            and row.is_map_complete
            and _map_patch_active(row, as_of)
            and _record_matches_league_tier(self.taxonomy, league, as_of, row.patch_id)
        ]
        if not map_candidates:
            return LeaguePatchResolution(
                league_id=league,
                patch_id=None,
                status="missing",
                reason="no_authoritative_patch and no completed eligible map evidence",
                source_id=None,
                freshness_state="missing",
                as_of=to_rfc3339(as_of),
            )

        latest_end = max(parse_rfc3339(row.event_end) for row in map_candidates)
        latest_maps = [row for row in map_candidates if parse_rfc3339(row.event_end) == latest_end]
        if not latest_maps:
            return LeaguePatchResolution(
                league_id=league,
                patch_id=None,
                status="missing",
                reason="no map evidence after filtering",
                source_id=None,
                freshness_state="missing",
                as_of=to_rfc3339(as_of),
            )

        patch_ids = {row.patch_id for row in latest_maps}
        if len(patch_ids) > 1:
            return LeaguePatchResolution(
                league_id=league,
                patch_id=None,
                status="conflict",
                reason="latest_map_patch_conflict",
                source_id=_first(row.source_id for row in latest_maps),
                freshness_state="conflict",
                as_of=to_rfc3339(as_of),
                as_of_conflicts=tuple(sorted(row.row_id for row in latest_maps)),
                source_tree_conflict_ids=tuple(sorted({row.source_snapshot_id for row in latest_maps})),
                source_snapshot_id=latest_maps[0].source_snapshot_id,
                source_snapshot_row_id=latest_maps[0].source_snapshot_row_id,
                source_snapshot_content_sha256=latest_maps[0].source_snapshot_content_sha256,
            )

        chosen = sorted(latest_maps, key=_map_patch_sort_key, reverse=True)[0]
        freshness = _freshness_state(
            source_available=parse_rfc3339(chosen.available_at),
            as_of=as_of,
            freshness_limit_days=freshness_days,
            freshness_limit_seconds=freshness_limit_seconds,
        )
        return LeaguePatchResolution(
            league_id=league,
            patch_id=chosen.patch_id,
            status="inferred",
            reason="latest completed league map",
            source_id=chosen.source_id,
            freshness_state=freshness,
            as_of=to_rfc3339(as_of),
            source_row_id=chosen.map_id,
            source_snapshot_id=chosen.source_snapshot_id,
            source_snapshot_row_id=chosen.source_snapshot_row_id,
            source_snapshot_content_sha256=chosen.source_snapshot_content_sha256,
        )


def resolve_competition_patch(
    league_id: str,
    taxonomy: CompetitionTaxonomy,
    patch_records: Iterable[LeaguePatchRecord],
    map_records: Iterable[MapPatchRecord],
    *,
    as_of: datetime,
) -> LeaguePatchResolution:
    resolver = PatchResolver(
        taxonomy=taxonomy,
        patch_records=patch_records,
        map_records=map_records,
    )
    return resolver.resolve_current_patch(league_id, as_of=as_of)


def _resolve_authoritative_records(rows: list[LeaguePatchRecord]) -> LeaguePatchRecord | None:
    if not rows:
        return None
    top_precedence = max(row.precedence for row in rows)
    top = [row for row in rows if row.precedence == top_precedence]
    if len(top) > 1:
        signatures = {
            (row.patch_id, row.source_snapshot_id, row.source_snapshot_row_id, row.source_snapshot_content_sha256)
            for row in top
        }
        if len(signatures) > 1:
            return None
    return max(top, key=_patch_record_sort_key)


def _patch_record_active(row: LeaguePatchRecord, as_of: datetime) -> bool:
    if not _is_row_active(
        effective_from=row.effective_from,
        effective_to=row.effective_to,
        available_at=row.available_at,
        source_updated_at=row.source_updated_at,
        observed_at=row.observed_at,
        as_of=as_of,
    ):
        return False

    try:
        _validate_patch_id(row.patch_id)
    except CompetitionError:
        return False
    return True


def _map_patch_active(row: MapPatchRecord, as_of: datetime) -> bool:
    if row.event_start is not None:
        event_start = parse_rfc3339(row.event_start)
        if event_start > as_of:
            return False
    if row.event_end is not None:
        if parse_rfc3339(row.event_end) > as_of:
            return False

    return _is_row_active(
        effective_from=row.effective_from,
        effective_to=row.effective_to,
        available_at=row.available_at,
        source_updated_at=row.source_updated_at,
        observed_at=row.observed_at,
        as_of=as_of,
    )


def _is_row_active(
    effective_from: str | None,
    effective_to: str | None,
    available_at: str,
    source_updated_at: str,
    observed_at: str,
    as_of: datetime,
) -> bool:
    parsed_available = parse_rfc3339(_require_non_empty(available_at, "available_at"))
    parsed_updated = parse_rfc3339(_require_non_empty(source_updated_at, "source_updated_at"))
    parsed_observed = parse_rfc3339(_require_non_empty(observed_at, "observed_at"))

    if effective_from is not None:
        start = parse_rfc3339(effective_from)
        if as_of < start:
            return False
    if effective_to is not None:
        end = parse_rfc3339(effective_to)
        if as_of > end:
            return False

    if parsed_available > as_of or parsed_updated > as_of or parsed_observed > as_of:
        return False
    if parsed_updated > parsed_observed:
        return False
    return True


def _record_matches_league_tier(
    taxonomy: CompetitionTaxonomy,
    league_id: str,
    as_of: datetime,
    patch_id: str,
) -> bool:
    _validate_patch_id(patch_id)
    try:
        _ = _record_matches_tier_profile(taxonomy, league_id, as_of)
    except PatchConflictError:
        return False
    return True


def _record_matches_tier_profile(
    taxonomy: CompetitionTaxonomy,
    league_id: str,
    as_of: datetime,
) -> bool:
    _ = taxonomy.resolve_tier_profile(league_id, as_of)
    return True


def _is_patch_id_valid(value: str) -> bool:
    return bool(_PATCH_RE.fullmatch(value))


def _freshness_state(
    source_available: datetime,
    as_of: datetime,
    freshness_limit_days: int,
    freshness_limit_seconds: int | None = None,
) -> str:
    source_available = _coerce_utc(source_available, field_name="source_available")
    as_of = _coerce_utc(as_of, field_name="as_of")
    if source_available > as_of:
        raise CompetitionError("source availability cannot be after as_of")

    if freshness_limit_seconds is not None:
        age_seconds = int((as_of - source_available).total_seconds())
        if age_seconds < 0:
            raise CompetitionError("as_of cannot be before source availability")
        return "fresh" if age_seconds <= freshness_limit_seconds else "stale"

    age_days = (as_of - source_available).days
    return "fresh" if age_days <= freshness_limit_days else "stale"


def _patch_record_sort_key(row: LeaguePatchRecord) -> tuple[int, datetime, datetime, str]:
    return (
        row.precedence,
        parse_rfc3339(row.announced_at),
        parse_rfc3339(row.available_at),
        row.row_id,
    )


def _map_patch_sort_key(row: MapPatchRecord) -> tuple[datetime, datetime, datetime, str]:
    return (
        parse_rfc3339(row.event_end),
        parse_rfc3339(row.available_at),
        parse_rfc3339(row.observed_at),
        row.map_id,
    )


def _taxonomy_sort_key(row: CompetitionTaxonomyRow) -> tuple[int, datetime, datetime, datetime, str]:
    return (
        row.precedence,
        parse_rfc3339(_require_non_empty(row.available_at, "available_at")),
        parse_rfc3339(row.observed_at),
        parse_rfc3339(row.source_updated_at),
        row.row_id,
    )


def _validate_taxonomy_row(row: CompetitionTaxonomyRow) -> None:
    _require_non_empty(row.row_id, "row_id")
    _require_non_empty(row.league_id, "league_id")
    if row.competition_tier not in {"tier1", "tier2", "tier3", "international"}:
        raise CompetitionError(f"invalid competition_tier: {row.competition_tier}")

    _require_non_empty(row.source_id, "source_id")
    _require_non_empty(row.source_name, "source_name")
    _require_non_empty(row.source_record_id, "source_record_id")
    _validate_snapshot_identity(
        row.source_snapshot_id,
        row.source_snapshot_row_id,
        row.source_snapshot_content_sha256,
    )
    _validate_time_axis(row.source_updated_at, row.observed_at, row.available_at)

    enforce_as_of_order(
        as_of=parse_rfc3339(row.observed_at),
        source_updated_at=parse_rfc3339(row.source_updated_at),
        observed_at=parse_rfc3339(row.observed_at),
        available_at=parse_rfc3339(row.available_at),
    )

    _require_non_empty(row.qualification_rule_id, "qualification_rule_id")
    if not isinstance(row.internationally_connectable, bool):
        raise CompetitionError("internationally_connectable must be a boolean")
    if not isinstance(row.structurally_globally_eligible, bool):
        raise CompetitionError("structurally_globally_eligible must be a boolean")


def _validate_patch_row(row: LeaguePatchRecord) -> None:
    _require_non_empty(row.row_id, "row_id")
    _require_non_empty(row.league_id, "league_id")
    _validate_patch_id(row.patch_id)
    _validate_snapshot_identity(
        row.source_snapshot_id,
        row.source_snapshot_row_id,
        row.source_snapshot_content_sha256,
    )

    if row.precedence < 0:
        raise CompetitionError("precedence must be non-negative")

    _require_non_empty(row.source_id, "source_id")
    _require_non_empty(row.source_name, "source_name")
    _require_non_empty(row.source_record_id, "source_record_id")

    parse_rfc3339(_require_non_empty(row.announced_at, "announced_at"))

    if row.effective_from is not None:
        parse_rfc3339(row.effective_from)
    if row.effective_to is not None:
        parse_rfc3339(row.effective_to)
    if row.effective_from and row.effective_to and parse_rfc3339(row.effective_from) > parse_rfc3339(row.effective_to):
        raise CompetitionError("effective_to cannot be before effective_from")

    _validate_time_axis(row.source_updated_at, row.observed_at, row.available_at)
    enforce_as_of_order(
        as_of=parse_rfc3339(row.observed_at),
        source_updated_at=parse_rfc3339(row.source_updated_at),
        observed_at=parse_rfc3339(row.observed_at),
        available_at=parse_rfc3339(row.available_at),
    )


def _validate_map_patch_row(row: MapPatchRecord) -> None:
    _validate_patch_id(_require_non_empty(row.patch_id, "patch_id"))
    _require_non_empty(row.map_id, "map_id")
    _require_non_empty(row.league_id, "league_id")
    parse_rfc3339(_require_non_empty(row.event_end, "event_end"))
    _require_non_empty(row.source_id, "source_id")
    _require_non_empty(row.source_name, "source_name")
    _require_non_empty(row.source_record_id, "source_record_id")
    _validate_snapshot_identity(
        row.source_snapshot_id,
        row.source_snapshot_row_id,
        row.source_snapshot_content_sha256,
    )

    _validate_time_axis(row.source_updated_at, row.observed_at, row.available_at)
    enforce_as_of_order(
        as_of=parse_rfc3339(row.observed_at),
        source_updated_at=parse_rfc3339(row.source_updated_at),
        observed_at=parse_rfc3339(row.observed_at),
        available_at=parse_rfc3339(row.available_at),
    )

    if row.result is not None and row.result not in {0, 1}:
        raise CompetitionError("map result must be 0 or 1")
    if row.event_start is not None:
        parse_rfc3339(row.event_start)


def _validate_time_axis(source_updated_at: str, observed_at: str, available_at: str) -> None:
    source_updated = parse_rfc3339(_require_non_empty(source_updated_at, "source_updated_at"))
    observed = parse_rfc3339(_require_non_empty(observed_at, "observed_at"))
    available = parse_rfc3339(_require_non_empty(available_at, "available_at"))

    if source_updated > observed:
        raise CompetitionError("source_updated_at must be <= observed_at")
    if available > observed:
        raise CompetitionError("available_at must be <= observed_at")


def _validate_snapshot_identity(snapshot_id: str | None, snapshot_row_id: str | None, content_sha256: str | None) -> None:
    _require_non_empty(snapshot_id, "source_snapshot_id")
    _require_non_empty(snapshot_row_id, "source_snapshot_row_id")
    if content_sha256 is None:
        raise CompetitionError("source_snapshot_content_sha256 is required")
    if not _SHA256_RE.fullmatch(content_sha256):
        raise CompetitionError("source_snapshot_content_sha256 must be a 64-char hex digest")


def _validate_patch_id(patch_id: str) -> str:
    value = _require_non_empty(patch_id, "patch_id")
    if not _PATCH_RE.fullmatch(value):
        raise CompetitionError(f"invalid patch_id format: {value}")
    return value


def _coerce_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise CompetitionError(f"{field_name} must include timezone")
    return value.astimezone(timezone.utc)


def _require_non_empty(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise CompetitionError(f"{field_name} is required")
    return value.strip()


def _first(value: str | None | Iterable[str]) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return next(iter(value))
    except StopIteration:
        return None


__all__ = [
    "CompetitionError",
    "CompetitionTaxonomy",
    "CompetitionTaxonomyRow",
    "LeaguePatchRecord",
    "LeaguePatchResolution",
    "LeaguePatchRecord",
    "MapPatchRecord",
    "PatchConflictError",
    "PatchRecordRegistry",
    "PatchResolver",
    "resolve_competition_patch",
]
