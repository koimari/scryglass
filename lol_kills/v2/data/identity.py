"""Append-only identity and crosswalk registry for L1 foundations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import (
    ContractError,
    canonicalize_json,
    parse_rfc3339,
    sha256_hex,
    stable_id,
    to_rfc3339,
)


class IdentityRegistryError(ContractError):
    """Raised when identity registry invariants fail."""


@dataclass(frozen=True)
class IdentityCrosswalkRow:
    """Append-only mapping from alias to canonical identity."""

    row_id: str
    entity_type: str
    canonical_id: str
    canonical_name: str
    source_name: str
    source_id: str
    source_snapshot_id: str
    source_snapshot_row_id: str
    source_snapshot_content_sha256: str
    source_record_id: str
    alias: str
    effective_from: str
    effective_to: str | None
    precedence: int
    observed_at: str
    source_updated_at: str
    available_at: str


@dataclass(frozen=True)
class IdentityCollision:
    alias: str
    entity_type: str
    canonical_ids: tuple[str, ...]
    row_ids: tuple[str, ...]
    sources: tuple[str, ...]


@dataclass(frozen=True)
class IdentityRegistry:
    """An append-only registry mapping aliases to canonical identities."""

    rows: tuple[IdentityCrosswalkRow, ...] = ()
    version: str = "2026.07.27.1"

    @classmethod
    def empty(cls, *, version: str = "2026.07.27.1") -> "IdentityRegistry":
        return cls(rows=(), version=version)

    @classmethod
    def from_rows(
        cls, rows: list[IdentityCrosswalkRow], *, version: str = "2026.07.27.1"
    ) -> "IdentityRegistry":
        for row in rows:
            _validate_row(row)
        if len(rows) != len({row.row_id for row in rows}):
            raise IdentityRegistryError("duplicate row IDs in identity registry")
        return cls(rows=tuple(rows), version=version)

    def append(self, row: IdentityCrosswalkRow) -> "IdentityRegistry":
        if not row.row_id:
            raise IdentityRegistryError("row_id is required")
        _validate_row(row)
        if any(existing.row_id == row.row_id for existing in self.rows):
            raise IdentityRegistryError(f"row_id already exists: {row.row_id}")
        return IdentityRegistry(rows=self.rows + (row,), version=self.version)

    def append_with_collision_check(
        self,
        *,
        entity_type: str,
        canonical_id: str,
        canonical_name: str,
        source_name: str,
        source_id: str,
        source_snapshot_id: str,
        source_snapshot_row_id: str,
        source_snapshot_content_sha256: str,
        source_record_id: str,
        alias: str,
        effective_from: str | datetime,
        effective_to: str | datetime | None = None,
        precedence: int = 0,
        observed_at: str | datetime | None = None,
        source_updated_at: str | datetime | None = None,
        available_at: str | datetime | None = None,
    ) -> "IdentityRegistry":
        effective_from_ts = _normalize_timestamp(effective_from)
        effective_to_ts = _normalize_timestamp_or_none(effective_to)
        observed_ts = _normalize_timestamp(_coalesce_dt(observed_at, effective_from_ts))
        source_updated_ts = _normalize_timestamp_or_none(source_updated_at) or observed_ts
        available_ts = _normalize_timestamp_or_none(available_at) or observed_ts

        row = IdentityCrosswalkRow(
            row_id=stable_id(
                "identity-crosswalk",
                f"{entity_type}|{source_id}|{canonical_id}|{_normalize_alias(alias)}|{observed_ts}",
            ),
            entity_type=_normalize_entity_type(entity_type),
            canonical_id=_require_non_empty(canonical_id, "canonical_id"),
            canonical_name=_require_non_empty(canonical_name, "canonical_name"),
            source_name=_require_non_empty(source_name, "source_name"),
            source_id=_require_non_empty(source_id, "source_id"),
            source_snapshot_id=_require_non_empty(source_snapshot_id, "source_snapshot_id"),
            source_snapshot_row_id=_require_non_empty(source_snapshot_row_id, "source_snapshot_row_id"),
            source_snapshot_content_sha256=_require_non_empty(
                source_snapshot_content_sha256,
                "source_snapshot_content_sha256",
            ),
            source_record_id=_require_non_empty(source_record_id, "source_record_id"),
            alias=_normalize_alias(alias),
            effective_from=effective_from_ts,
            effective_to=effective_to_ts,
            precedence=_validate_precedence(precedence),
            observed_at=observed_ts,
            source_updated_at=source_updated_ts,
            available_at=available_ts,
        )
        return self.append(row)

    def _active_rows(self, as_of: datetime) -> tuple[IdentityCrosswalkRow, ...]:
        as_of = _coerce_utc(as_of, field_name="as_of")
        active: list[IdentityCrosswalkRow] = []
        for row in self.rows:
            start = parse_rfc3339(_require_non_empty(row.effective_from, "effective_from"))
            if as_of < start:
                continue
            if row.effective_to is not None:
                end = parse_rfc3339(row.effective_to)
                if as_of > end:
                    continue
            if parse_rfc3339(row.available_at) > as_of:
                continue
            if parse_rfc3339(row.observed_at) > as_of:
                continue
            if parse_rfc3339(row.source_updated_at) > as_of:
                continue
            if parse_rfc3339(row.source_updated_at) > parse_rfc3339(row.observed_at):
                continue
            if parse_rfc3339(row.available_at) > parse_rfc3339(row.observed_at):
                continue
            active.append(row)
        return tuple(active)

    def audit_collisions(
        self,
        as_of: datetime,
        *,
        entity_type: str | None = None,
    ) -> tuple[IdentityCollision, ...]:
        as_of = _coerce_utc(as_of, field_name="as_of")
        index: dict[tuple[str, str], dict[str, list[IdentityCrosswalkRow]]] = {}

        for row in self._active_rows(as_of):
            if entity_type is not None and row.entity_type != _normalize_entity_type(entity_type):
                continue
            key = (row.entity_type, row.alias)
            index.setdefault(key, {}).setdefault(row.canonical_id, []).append(row)

        collisions: list[IdentityCollision] = []
        for (resolved_entity_type, alias), by_canonical in index.items():
            if len(by_canonical) <= 1:
                continue
            canonical_ids = tuple(sorted(by_canonical.keys()))
            row_ids = tuple(
                row_id
                for canonical_id in canonical_ids
                for row_id in sorted(r.row_id for r in by_canonical[canonical_id])
            )
            sources = tuple(
                row.source_id
                for canonical_id in canonical_ids
                for row in sorted(by_canonical[canonical_id], key=lambda item: item.row_id)
            )
            collisions.append(
                IdentityCollision(
                    alias=alias,
                    entity_type=resolved_entity_type,
                    canonical_ids=canonical_ids,
                    row_ids=tuple(dict.fromkeys(row_ids)),
                    sources=tuple(dict.fromkeys(sources)),
                )
            )

        collisions.sort(key=lambda item: (item.entity_type, item.alias))
        return tuple(collisions)

    def lookup(
        self,
        alias: str,
        *,
        as_of: datetime,
        entity_type: str | None = None,
        strict: bool = False,
        fail_on_collision: bool = False,
    ) -> str | None:
        normalized_alias = _normalize_alias(alias)
        if not normalized_alias:
            if strict:
                raise IdentityRegistryError("alias must be non-empty")
            return None

        as_of = _coerce_utc(as_of, field_name="as_of")
        collisions = self.audit_collisions(as_of, entity_type=entity_type)
        for collision in collisions:
            if collision.alias == normalized_alias and (
                entity_type is None or collision.entity_type == _normalize_entity_type(entity_type)
            ):
                if strict or fail_on_collision:
                    raise IdentityRegistryError(
                        f"ambiguous alias '{alias}' maps to {len(collision.canonical_ids)} candidates"
                    )
                return None

        candidates = [
            row
            for row in self._active_rows(as_of)
            if row.alias == normalized_alias
            and (entity_type is None or row.entity_type == _normalize_entity_type(entity_type))
        ]
        if not candidates:
            if strict:
                raise IdentityRegistryError(f"unknown alias: {alias}")
            return None

        if len(candidates) > 1:
            best_precedence = max(row.precedence for row in candidates)
            winner_ties = sorted(
                [row for row in candidates if row.precedence == best_precedence],
                key=_candidate_sort_key,
            )
            top_row = winner_ties[-1]
            same_precedence = [row for row in candidates if row.precedence == best_precedence]
            canonical_ids = {row.canonical_id for row in same_precedence}
            if len(canonical_ids) > 1:
                if strict:
                    raise IdentityRegistryError(
                        f"ambiguous alias '{alias}' maps to {len(canonical_ids)} candidates"
                    )
                return None
        else:
            top_row = candidates[0]

        return top_row.canonical_id

    def canonical_ids(self) -> tuple[str, ...]:
        return tuple(sorted({row.canonical_id for row in self.rows}))

    def to_payload(self) -> dict[str, Any]:
        return {
            "registry_version": self.version,
            "row_count": len(self.rows),
            "rows": [row.__dict__ for row in sorted(self.rows, key=lambda row: row.row_id)],
        }

    def to_payload_canonical(self) -> str:
        return canonicalize_json(self.to_payload())

    def sha256(self) -> str:
        return sha256_hex(self.to_payload())

    @classmethod
    def read(cls, path: Path) -> "IdentityRegistry":
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = [
            IdentityCrosswalkRow(
                row_id=row["row_id"],
                entity_type=row["entity_type"],
                canonical_id=row["canonical_id"],
                canonical_name=row["canonical_name"],
                source_name=row["source_name"],
                source_id=row["source_id"],
                source_snapshot_id=row["source_snapshot_id"],
                source_snapshot_row_id=row["source_snapshot_row_id"],
                source_snapshot_content_sha256=row["source_snapshot_content_sha256"],
                source_record_id=row["source_record_id"],
                alias=row["alias"],
                effective_from=row["effective_from"],
                effective_to=row.get("effective_to"),
                precedence=int(row.get("precedence", 0)),
                observed_at=row["observed_at"],
                source_updated_at=row["source_updated_at"],
                available_at=row["available_at"],
            )
            for row in payload.get("rows", [])
        ]
        return cls(rows=tuple(rows), version=payload.get("registry_version", "2026.07.27.1"))

    def write(self, path: Path) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_payload_canonical() + "\n"
        path.write_text(payload, encoding="utf-8")
        return payload


def _candidate_sort_key(row: IdentityCrosswalkRow) -> tuple[int, datetime, datetime, datetime, str]:
    return (
        row.precedence,
        parse_rfc3339(row.available_at),
        parse_rfc3339(row.observed_at),
        parse_rfc3339(row.source_updated_at),
        row.row_id,
    )


def _normalize_alias(alias: str) -> str:
    if not isinstance(alias, str):
        raise IdentityRegistryError("alias must be a string")
    return " ".join(alias.strip().lower().split())


def _normalize_entity_type(entity_type: str) -> str:
    normalized = _require_non_empty(entity_type, "entity_type").strip().lower()
    if normalized not in {"player", "organization", "champion", "league", "team", "roster", "tournament"}:
        raise IdentityRegistryError(f"unsupported entity_type: {entity_type!r}")
    return normalized


def _require_non_empty(value: str | None, field_name: str) -> str:
    if value is None or not str(value).strip():
        raise IdentityRegistryError(f"{field_name} is required")
    return str(value).strip()


def _coerce_utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise IdentityRegistryError(f"{field_name} must include timezone")
    return value.astimezone(timezone.utc)


def _coalesce_dt(*values: datetime | str | None) -> datetime:
    for value in values:
        if value is None:
            continue
        if isinstance(value, datetime):
            return _coerce_utc(value, field_name="timestamp")
        if isinstance(value, str) and value.strip():
            return parse_rfc3339(value)
    raise IdentityRegistryError("timestamp could not be inferred")


def _normalize_timestamp(value: str | datetime) -> str:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise IdentityRegistryError("timestamp must include timezone")
        return to_rfc3339(value.astimezone(timezone.utc))
    if not isinstance(value, str):
        raise IdentityRegistryError("timestamp must be a string or datetime")
    return to_rfc3339(parse_rfc3339(value))


def _normalize_timestamp_or_none(value: str | datetime | None) -> str | None:
    if value is None:
        return None
    return _normalize_timestamp(value)


def _validate_precedence(value: int) -> int:
    if not isinstance(value, int) or value < 0:
        raise IdentityRegistryError("precedence must be a non-negative integer")
    return value


def _validate_row(row: IdentityCrosswalkRow) -> None:
    if not isinstance(row.row_id, str) or not row.row_id:
        raise IdentityRegistryError("row_id is required")
    if not row.entity_type:
        raise IdentityRegistryError("entity_type is required")
    _normalize_entity_type(row.entity_type)
    _require_non_empty(row.canonical_id, "canonical_id")
    _require_non_empty(row.canonical_name, "canonical_name")
    _require_non_empty(row.source_name, "source_name")
    _require_non_empty(row.source_id, "source_id")
    _require_non_empty(row.source_snapshot_id, "source_snapshot_id")
    _require_non_empty(row.source_snapshot_row_id, "source_snapshot_row_id")
    _require_non_empty(row.source_snapshot_content_sha256, "source_snapshot_content_sha256")
    if not _is_sha256(row.source_snapshot_content_sha256):
        raise IdentityRegistryError("source_snapshot_content_sha256 must be a 64-char hex digest")
    _normalize_alias(row.alias)
    _require_non_empty(row.source_record_id, "source_record_id")

    effective_from = parse_rfc3339(_require_non_empty(row.effective_from, "effective_from"))
    if row.effective_to is not None:
        end = parse_rfc3339(row.effective_to)
        if end < effective_from:
            raise IdentityRegistryError("effective_to cannot be before effective_from")

    observed = parse_rfc3339(_require_non_empty(row.observed_at, "observed_at"))
    source_updated = parse_rfc3339(_require_non_empty(row.source_updated_at, "source_updated_at"))
    available = parse_rfc3339(_require_non_empty(row.available_at, "available_at"))

    if source_updated > observed:
        raise IdentityRegistryError("source_updated_at must be <= observed_at")
    if available > observed:
        raise IdentityRegistryError("available_at must be <= observed_at")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


__all__ = [
    "IdentityCollision",
    "IdentityCrosswalkRow",
    "IdentityRegistry",
    "IdentityRegistryError",
]
