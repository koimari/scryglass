"""Snapshot and lineage foundations for L1 data manifests."""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from math import isfinite
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..data import ROLES, parse_rfc3339, to_rfc3339
from ..data.common import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_hex,
    sha256_hex_bytes,
    sha256_canonical_object_hash,
)
from ..data.source_tree import (
    canonical_source_tree_sha256,
    normalize_source_tree_path,
    resolve_repository_file,
)

CONTRACT_TREE_SHA256 = "8748bbe48b273593b09304ac80923f11384de808b835f6e83e97c6fef48661dd"

_SOURCE_SNAPSHOT_PREFIX = "scryglass:source-snapshot"
_TRAINING_SNAPSHOT_PREFIX = "scryglass:training-snapshot"
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_SOURCE_SNAPSHOT_ALLOWED_STATUS = {"ok", "degraded", "stale", "invalid"}
_FORBIDDEN_NORMALIZED_TOKENS = (
    "grub",
    "market",
)
_FORBIDDEN_STANDALONE_TOKENS = {
    "bet",
    "betting",
    "wager",
    "wagering",
}
_FORBIDDEN_COMPACT_ALIAS_TOKENS = {
    "totalkills",
    "underover",
    "overunder",
}
_FORBIDDEN_ORDERED_TOKEN_PAIRS = (
    ("total", "kills"),
    ("under", "over"),
    ("over", "under"),
)
_GIT_REPO_ROOT = Path(__file__).resolve().parents[3]


def _normalize_repo_path(value: str) -> str:
    normalized = value.replace("\\", "/").strip()
    if not normalized:
        raise SourceSnapshotSnapshotError("path locator cannot be empty")
    if normalized.startswith("/") or normalized.startswith("~"):
        raise SourceSnapshotSnapshotError("locators must be repository-relative")
    if normalized.startswith("./") or normalized.startswith("../") or normalized == "..":
        raise SourceSnapshotSnapshotError(f"path traversal is not allowed: {value!r}")
    if ".." in Path(normalized).parts:
        raise SourceSnapshotSnapshotError(f"invalid path traversal: {value!r}")
    return normalized


class SourceSnapshotSnapshotError(ValueError):
    """Raised when source snapshot inputs violate invariants."""


class SourceTreeMismatchError(ValueError):
    """Raised when computed source-tree hashes diverge."""


class TrainingSnapshotError(SourceSnapshotSnapshotError):
    """Raised when training snapshot invariants are violated."""


class SourceSnapshotMismatch(ValueError):
    """Raised when lineage hashes mismatch."""


class SourceSnapshotError(SourceSnapshotSnapshotError):
    pass


@dataclass(frozen=True)
class ChampionPatchRoleAppearanceRow:
    """Verified champion appearances for one league-patch-role cell."""

    league_id: str
    patch_id: str
    role: str
    champion_id: str
    verified_appearance_count: int
    source_snapshot_id: str
    source_snapshot_row_id: str
    source_id: str
    source_snapshot_content_sha256: str
    as_of: str
    source_tree_sha256: str
    available_at: str
    status: str
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        _validate_patch_role_row(self)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "league_id": self.league_id,
            "patch_id": self.patch_id,
            "role": self.role,
            "champion_id": self.champion_id,
            "verified_appearance_count": self.verified_appearance_count,
            "source_snapshot_id": self.source_snapshot_id,
            "source_snapshot_row_id": self.source_snapshot_row_id,
            "source_id": self.source_id,
            "source_snapshot_content_sha256": self.source_snapshot_content_sha256,
            "as_of": self.as_of,
            "source_tree_sha256": self.source_tree_sha256,
            "available_at": self.available_at,
            "status": self.status,
        }
        if self.unavailable_reason is not None:
            payload["unavailable_reason"] = self.unavailable_reason
        return payload


@dataclass(frozen=True)
class SourceSnapshotRow:
    source_snapshot_id: str
    source_id: str
    source_name: str
    source_snapshot_row_id: str
    source_record_id: str
    source_content_sha256: str
    source_content_object_sha256: str | None
    source_content_path: str
    source_content_size_bytes: int
    source_tree_sha256: str
    row_count: int
    source_updated_at: str
    observed_at: str
    available_at: str
    source_commit: str | None = None
    is_stale: bool = False
    freshness_limit_seconds: int | None = None
    issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_snapshot_row(self)

    def to_payload(self) -> dict[str, Any]:
        payload = {
            "source_snapshot_id": self.source_snapshot_id,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_snapshot_row_id": self.source_snapshot_row_id,
            "source_record_id": self.source_record_id,
            "source_content_sha256": self.source_content_sha256,
            "source_content_object_sha256": self.source_content_object_sha256,
            "source_content_path": self.source_content_path,
            "source_content_size_bytes": self.source_content_size_bytes,
            "source_tree_sha256": self.source_tree_sha256,
            "row_count": self.row_count,
            "source_updated_at": self.source_updated_at,
            "observed_at": self.observed_at,
            "available_at": self.available_at,
            "is_stale": self.is_stale,
        }
        if self.source_commit is not None:
            payload["source_commit"] = self.source_commit
        if self.freshness_limit_seconds is not None:
            payload["freshness_limit_seconds"] = self.freshness_limit_seconds
        if self.issues:
            payload["issues"] = list(self.issues)
        return payload


@dataclass(frozen=True)
class SourceSnapshot:
    """Immutable source-level snapshot manifest."""

    schema_version: str
    model_version: str
    adapter_version: str
    code_version: str
    as_of: str
    snapshot_id: str = ""
    reviewed_at: str = ""
    rows: tuple[SourceSnapshotRow, ...] = ()
    source_tree_sha256: str = ""
    source_tree_allowlist: tuple[str, ...] = ()
    created_at: str = ""
    contract_tree_sha256: str = CONTRACT_TREE_SHA256
    champion_patch_role_counts: tuple[ChampionPatchRoleAppearanceRow, ...] = ()
    status: str = "ok"

    def __post_init__(self) -> None:
        parsed_as_of = parse_rfc3339(self.as_of)
        parse_rfc3339(self.created_at)
        parse_rfc3339(self.reviewed_at)
        if not self.schema_version.strip():
            raise SourceSnapshotError("schema_version required")
        if not self.adapter_version.strip():
            raise SourceSnapshotError("adapter_version required")
        if not self.code_version.strip():
            raise SourceSnapshotError("code_version required")
        if not self.rows:
            raise SourceSnapshotError("source snapshot requires at least one source row")
        if self.status not in _SOURCE_SNAPSHOT_ALLOWED_STATUS:
            raise SourceSnapshotError(f"invalid status: {self.status}")
        ordered_rows = tuple(
            sorted(
                self.rows,
                key=lambda item: (item.source_snapshot_row_id, item.source_id, item.source_record_id),
            )
        )
        if ordered_rows != self.rows:
            object.__setattr__(self, "rows", ordered_rows)

        normalized_allowlist = tuple(sorted(set(self.source_tree_allowlist)))
        if len(self.source_tree_allowlist) != len(normalized_allowlist):
            raise SourceSnapshotError("source_tree_allowlist must be unique")
        if any(not item.strip() for item in normalized_allowlist):
            raise SourceSnapshotError("source_tree_allowlist entries must be non-empty")
        object.__setattr__(self, "source_tree_allowlist", normalized_allowlist)

        if not _is_sha256(self.source_tree_sha256):
            raise SourceSnapshotError("source_tree_sha256 must be a 64-char sha256 digest")
        _validate_contract_tree_hash(self.contract_tree_sha256)
        verify_source_tree(self, self.source_tree_allowlist, _GIT_REPO_ROOT)

        row_paths = tuple(row.source_content_path for row in self.rows)
        if set(row_paths) != set(self.source_tree_allowlist):
            raise SourceSnapshotError("source_tree_allowlist must match source row paths exactly")
        if len(set(row.source_snapshot_row_id for row in self.rows)) != len(self.rows):
            raise SourceSnapshotError("source_snapshot_row_id values must be unique")

        by_snapshot_row = {row.source_snapshot_row_id: row for row in self.rows}
        by_snapshot_id = {row.source_snapshot_id for row in self.rows}
        if len(by_snapshot_id) != 1:
            raise SourceSnapshotError("snapshot rows must share one source_snapshot_id")

        row_snapshot_id = next(iter(by_snapshot_id))
        expected_id = _make_source_snapshot_id(self)
        if self.snapshot_id:
            if self.snapshot_id != expected_id:
                raise SourceSnapshotError("snapshot_id must be derived from ordered canonical snapshot payload")
            if row_snapshot_id != self.snapshot_id:
                raise SourceSnapshotError("source_snapshot_id on rows must match snapshot id")
            active_snapshot_id = self.snapshot_id
        else:
            active_snapshot_id = expected_id
            object.__setattr__(self, "snapshot_id", expected_id)
            if row_snapshot_id != expected_id:
                object.__setattr__(
                    self,
                    "rows",
                    tuple(replace(row, source_snapshot_id=expected_id) for row in self.rows),
                )
                object.__setattr__(
                    self,
                    "champion_patch_role_counts",
                    tuple(
                        replace(row, source_snapshot_id=expected_id)
                        for row in self.champion_patch_role_counts
                    ),
                )
                by_snapshot_row = {row.source_snapshot_row_id: row for row in self.rows}
                row_snapshot_id = expected_id

        for row in self.rows:
            source_updated = parse_rfc3339(row.source_updated_at)
            observed_at = parse_rfc3339(row.observed_at)
            available_at = parse_rfc3339(row.available_at)
            if source_updated > observed_at:
                raise SourceSnapshotError("source_updated_at cannot be after observed_at")
            if available_at > observed_at:
                raise SourceSnapshotError("available_at cannot be after observed_at")
            if source_updated > parsed_as_of:
                raise SourceSnapshotError("source_updated_at cannot be later than snapshot as_of")
            if observed_at > parsed_as_of:
                raise SourceSnapshotError("source row observed_at cannot exceed snapshot as_of")
            if available_at > parsed_as_of:
                raise SourceSnapshotError("source row availability cannot exceed snapshot as_of")
            if row.source_tree_sha256 != self.source_tree_sha256:
                raise SourceSnapshotError("source row tree hash must match snapshot tree hash")
            if row.source_snapshot_id != row_snapshot_id:
                raise SourceSnapshotError("source_snapshot_id on rows must match snapshot id")
            if row.freshness_limit_seconds is not None and (
                isinstance(row.freshness_limit_seconds, bool)
                or not isinstance(row.freshness_limit_seconds, (int, float))
                or not isfinite(row.freshness_limit_seconds)
                or row.freshness_limit_seconds <= 0
            ):
                raise SourceSnapshotError("freshness_limit_seconds must be positive and finite")
            if self.status == "ok" and row.freshness_limit_seconds is None:
                raise SourceSnapshotError(
                    "status 'ok' requires a positive registered freshness SLO for every source"
                )
            stale_derived = _is_source_row_stale(
                row,
                as_of=parsed_as_of,
                source_updated=source_updated,
            )
            if row.is_stale != stale_derived:
                raise SourceSnapshotError("source row is_stale must match derived freshness")
            if self.status == "ok" and stale_derived:
                raise SourceSnapshotError("status 'ok' cannot include stale rows")
            if self.status == "ok" and row.issues:
                raise SourceSnapshotError("status 'ok' requires source rows to be complete")

        ordered_counts = tuple(
            sorted(self.champion_patch_role_counts, key=_appearance_key)
        )
        if ordered_counts != self.champion_patch_role_counts:
            object.__setattr__(self, "champion_patch_role_counts", ordered_counts)

        allowed_cells: set[tuple[str, str, str, str]] = set()
        for row in self.champion_patch_role_counts:
            if row.source_snapshot_id != active_snapshot_id:
                raise SourceSnapshotError(
                    "champion patch role rows must reference this source snapshot id"
                )
            cell_key = (row.league_id, row.patch_id, row.role, row.champion_id)
            if cell_key in allowed_cells:
                raise SourceSnapshotError(f"duplicate appearance row for {cell_key}")
            allowed_cells.add(cell_key)

            source_row = by_snapshot_row.get(row.source_snapshot_row_id)
            if source_row is None:
                raise SourceSnapshotError(
                    f"appearance row references unknown source_snapshot_row_id={row.source_snapshot_row_id}"
                )
            if row.source_snapshot_content_sha256 != source_row.source_content_sha256:
                raise SourceSnapshotError("appearance row content hash must match source row hash")
            if row.source_id != source_row.source_id:
                raise SourceSnapshotError("appearance row source id must match source row source id")

            row_as_of = parse_rfc3339(row.as_of)
            row_available = parse_rfc3339(row.available_at)
            source_available = parse_rfc3339(source_row.available_at)
            if source_available > row_as_of:
                raise SourceSnapshotError(
                    "source row available_at cannot exceed appearance as_of"
                )
            if row_as_of > parsed_as_of:
                raise SourceSnapshotError("appearance as_of cannot exceed snapshot as_of")
            if row_available != source_available:
                raise SourceSnapshotError(
                    "appearance available_at must equal its referenced source row available_at"
                )
            if row_available > row_as_of:
                raise SourceSnapshotError("appearance available_at cannot exceed appearance as_of")
            if row_available > parsed_as_of:
                raise SourceSnapshotError("appearance available_at cannot exceed snapshot as_of")
            if row.source_tree_sha256 != self.source_tree_sha256:
                raise SourceSnapshotError("appearance row tree hash must match snapshot tree hash")

        if active_snapshot_id != expected_id:
            raise SourceSnapshotError("source_snapshot_id must be derived from ordered canonical snapshot payload")
        _reject_forbidden_recursive(self._payload_for_id(), "source snapshot payload")

    @property
    def row_count(self) -> int:
        return sum(row.row_count for row in self.rows)

    def _ordered_rows(self) -> tuple[SourceSnapshotRow, ...]:
        return tuple(
            sorted(
                self.rows,
                key=lambda item: (item.source_snapshot_row_id, item.source_id, item.source_record_id),
            )
        )

    def _ordered_champion_counts(self) -> tuple[ChampionPatchRoleAppearanceRow, ...]:
        return tuple(sorted(self.champion_patch_role_counts, key=_appearance_key))

    def _payload_for_id(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "adapter_version": self.adapter_version,
            "code_version": self.code_version,
            "as_of": self.as_of,
            "reviewed_at": self.reviewed_at,
            "created_at": self.created_at,
            "contract_tree_sha256": self.contract_tree_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "source_tree_allowlist": sorted(self.source_tree_allowlist),
            "rows": [
                _snapshot_row_payload_for_id(row)
                for row in self._ordered_rows()
            ],
            "champion_patch_role_counts": [
                _champion_count_payload_for_id(row)
                for row in self._ordered_champion_counts()
            ],
            "status": self.status,
        }
        return payload

    def to_payload(self) -> dict[str, Any]:
        payload = self._payload_for_id()
        payload["snapshot_id"] = self.snapshot_id
        payload["row_count"] = self.row_count
        return payload

    def sha256(self) -> str:
        return sha256_hex(self.to_payload())

    def write(self, path: Path) -> "ArtifactWriteResult":
        payload = self.to_payload()
        canonical_bytes = canonical_json_bytes(payload) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes)
        object_digest = sha256_bytes(canonical_json_bytes(payload))
        file_digest = sha256_bytes(canonical_bytes)
        return ArtifactWriteResult(object_sha256=object_digest, file_sha256=file_digest, path=path)


def _snapshot_row_payload_for_id(row: SourceSnapshotRow) -> dict[str, Any]:
    payload = row.to_payload()
    payload.pop("source_snapshot_id", None)
    return payload


def _champion_count_payload_for_id(row: ChampionPatchRoleAppearanceRow) -> dict[str, Any]:
    payload = row.to_payload()
    payload.pop("source_snapshot_id", None)
    payload.pop("source_snapshot_row_id", None)
    return payload


SourceSnapshotManifest = SourceSnapshot


@dataclass(frozen=True)
class SourceSnapshotRowSummary:
    """Aggregated row summaries used in training manifests and lineage."""

    source_id: str
    source_name: str
    row_count: int
    stale_count: int
    latest_available_at: str | None = None
    source_snapshot_id: str | None = None
    source_snapshot_content_sha256: str | None = None


@dataclass(frozen=True)
class TrainingSnapshot:
    """Immutable training snapshot manifest."""

    schema_version: str
    model_version: str
    adapter_version: str
    code_version: str
    as_of: str
    train_cutoff: str
    source_manifest_locator: str
    source_manifest_object_sha256: str
    source_snapshot_pairs: tuple[tuple[str, str], ...]
    source_tree_sha256: str
    source_tree_allowlist: tuple[str, ...]
    row_count_evidence_locator: str
    row_count_evidence_sha256: str
    row_count_by_year: Mapping[str, int]
    row_count_by_league: Mapping[str, int]
    row_count_by_tier: Mapping[str, int]
    row_count_by_patch: Mapping[str, int]
    row_count_by_source: Mapping[str, int]
    source_rows: tuple[SourceSnapshotRowSummary, ...]
    created_at: str
    taxonomy_version: str = "unknown"
    crosswalk_version: str = "unknown"
    inclusion_filters: tuple[str, ...] = ()
    exclusion_filters: tuple[str, ...] = ()
    min_event_at: str | None = None
    max_event_at: str | None = None
    min_available_at: str | None = None
    max_available_at: str | None = None
    duplicate_count: int = 0
    correction_count: int = 0
    missingness_count: int = 0
    conflict_count: int = 0
    identity_audit_count: int = 0
    split_assignment_ids: tuple[str, ...] = ()
    split_assignment_locators: tuple[str, ...] = ()
    split_assignment_sha256s: tuple[str, ...] = ()
    environment_lock_sha256: str | None = None
    environment_lock_locator: str | None = None
    candidate_code_commit: str | None = None
    code_commit: str | None = None
    supersession_lines: tuple[str, ...] = ()
    correction_lines: tuple[str, ...] = ()
    contract_tree_sha256: str = CONTRACT_TREE_SHA256
    row_count: int = 0
    require_no_stale_required: bool = False
    required_source_snapshot_pairs: tuple[tuple[str, str], ...] = ()
    optional_source_snapshot_pairs: tuple[tuple[str, str], ...] = ()
    status: str = "ok"
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        parsed_as_of = parse_rfc3339(self.as_of)
        parsed_train_cutoff = parse_rfc3339(self.train_cutoff)
        parse_rfc3339(self.created_at)
        if not self.schema_version.strip():
            raise TrainingSnapshotError("schema_version required")
        if not self.model_version.strip():
            raise TrainingSnapshotError("model_version required")
        if not self.adapter_version.strip():
            raise TrainingSnapshotError("adapter_version required")
        if not self.code_version.strip():
            raise TrainingSnapshotError("code_version required")
        if self.status not in _SOURCE_SNAPSHOT_ALLOWED_STATUS:
            raise TrainingSnapshotError(f"invalid status: {self.status}")
        if not _is_sha256(self.source_tree_sha256):
            raise TrainingSnapshotError("source_tree_sha256 must be a 64-char sha256 digest")
        try:
            _validate_contract_tree_hash(self.contract_tree_sha256)
        except SourceSnapshotError as err:
            raise TrainingSnapshotError(str(err)) from err
        verify_source_tree(self, self.source_tree_allowlist, _GIT_REPO_ROOT)

        normalized_allowlist = tuple(sorted(set(self.source_tree_allowlist)))
        if len(self.source_tree_allowlist) != len(normalized_allowlist):
            raise TrainingSnapshotError("source_tree_allowlist must be unique")
        if any(not item.strip() for item in normalized_allowlist):
            raise TrainingSnapshotError("source_tree_allowlist entries must be non-empty")
        object.__setattr__(self, "source_tree_allowlist", normalized_allowlist)

        try:
            source_manifest_locator = _require_exact_repo_locator(
                self.source_manifest_locator, "source_manifest_locator"
            )
            _require_hash(
                self.source_manifest_object_sha256,
                field_name="source_manifest_object_sha256",
            )
            resolved_source_snapshot = _load_source_snapshot_manifest(
                source_manifest_locator,
                self.source_manifest_object_sha256,
            )
        except SourceSnapshotError as err:
            raise TrainingSnapshotError(str(err)) from err
        object.__setattr__(self, "source_manifest_locator", source_manifest_locator)
        if resolved_source_snapshot.source_tree_sha256 != self.source_tree_sha256:
            raise TrainingSnapshotError(
                "source manifest source_tree_sha256 must match training snapshot"
            )
        if resolved_source_snapshot.source_tree_allowlist != normalized_allowlist:
            raise TrainingSnapshotError(
                "source manifest source_tree_allowlist must match training snapshot exactly"
            )

        ordered_pairs = tuple(sorted(set(self.source_snapshot_pairs), key=_source_pair_key))
        if len(ordered_pairs) != len(self.source_snapshot_pairs):
            raise TrainingSnapshotError("source_snapshot_pairs must be unique")
        if not ordered_pairs:
            raise TrainingSnapshotError("training snapshot requires source snapshot identifiers")
        pair_ids = [pair[0] for pair in ordered_pairs]
        if len(pair_ids) != len(set(pair_ids)):
            raise TrainingSnapshotError("source_snapshot_pairs must not repeat source_snapshot_id")
        pair_hashes_by_snapshot = {source_id: source_hash for source_id, source_hash in ordered_pairs}
        if len(pair_hashes_by_snapshot) != len(ordered_pairs):
            raise TrainingSnapshotError("source_snapshot_pairs cannot contain duplicate source ids")
        for source_id, source_hash in ordered_pairs:
            if not source_id.strip():
                raise TrainingSnapshotError("source_snapshot_ids are required")
            _require_hash(source_hash, field_name="source_snapshot_content_sha256")
        object.__setattr__(self, "source_snapshot_pairs", ordered_pairs)
        exact_manifest_pair = (
            resolved_source_snapshot.snapshot_id,
            self.source_manifest_object_sha256,
        )
        if ordered_pairs != (exact_manifest_pair,):
            raise TrainingSnapshotError(
                "source_snapshot_pairs must equal the resolved source manifest ID/object hash pair"
            )

        try:
            include_filters = _validate_forbidden_filters(
                self.inclusion_filters, "inclusion_filters"
            )
            exclusion_filters = _validate_forbidden_filters(
                self.exclusion_filters, "exclusion_filters"
            )
        except SourceSnapshotError as err:
            raise TrainingSnapshotError(str(err)) from err
        object.__setattr__(self, "inclusion_filters", tuple(sorted(set(include_filters))))
        object.__setattr__(self, "exclusion_filters", tuple(sorted(set(exclusion_filters))))
        try:
            _validate_forbidden_filters(self.correction_lines, "correction_lines")
            _validate_forbidden_filters(self.supersession_lines, "supersession_lines")
        except SourceSnapshotError as err:
            raise TrainingSnapshotError(str(err)) from err

        if not (
            len(self.split_assignment_ids)
            == len(self.split_assignment_locators)
            == len(self.split_assignment_sha256s)
        ):
            raise TrainingSnapshotError(
                "split assignment IDs, locators, and hashes must have equal lengths"
            )
        if self.status == "ok" and not self.split_assignment_ids:
            raise TrainingSnapshotError("status 'ok' requires nonempty split assignment evidence")
        split_evidence: list[tuple[str, str, str]] = []
        for assignment_id, locator, digest in zip(
            self.split_assignment_ids,
            self.split_assignment_locators,
            self.split_assignment_sha256s,
        ):
            try:
                normalized_locator = _require_exact_repo_locator(
                    locator, "split_assignment_locator"
                )
                _require_hash(digest, field_name="split_assignment_sha256")
                evidence_bytes = resolve_repository_file(
                    _GIT_REPO_ROOT, normalized_locator
                ).read_bytes()
            except (SourceSnapshotError, ValueError) as err:
                raise TrainingSnapshotError(str(err)) from err
            actual_digest = sha256_hex_bytes(evidence_bytes)
            if actual_digest != digest:
                raise TrainingSnapshotError(
                    "split_assignment_sha256 must match repository file raw bytes"
                )
            try:
                split_payload = json.loads(evidence_bytes.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as err:
                raise TrainingSnapshotError(
                    "split assignment evidence must contain JSON"
                ) from err
            _reject_forbidden_recursive(
                split_payload, "split assignment evidence"
            )
            expected_assignment_id = canonicalize_snapshot_id(
                "scryglass:split-assignment", actual_digest
            )
            if assignment_id != expected_assignment_id:
                raise TrainingSnapshotError(
                    "split_assignment_id must be derived from split assignment raw bytes"
                )
            split_evidence.append((assignment_id, normalized_locator, digest))
        if len(set(split_evidence)) != len(split_evidence):
            raise TrainingSnapshotError("split assignment evidence must be unique")
        ordered_split_evidence = tuple(sorted(split_evidence))
        object.__setattr__(
            self, "split_assignment_ids", tuple(item[0] for item in ordered_split_evidence)
        )
        object.__setattr__(
            self,
            "split_assignment_locators",
            tuple(item[1] for item in ordered_split_evidence),
        )
        object.__setattr__(
            self,
            "split_assignment_sha256s",
            tuple(item[2] for item in ordered_split_evidence),
        )

        for mapping in (self.row_count_by_year, self.row_count_by_league, self.row_count_by_tier, self.row_count_by_patch, self.row_count_by_source):
            for key, value in mapping.items():
                if not isinstance(key, str) or not key.strip():
                    raise TrainingSnapshotError("count map keys must be non-empty")
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise TrainingSnapshotError("count map values must be non-negative")

        required_pairs = tuple(sorted(set(self.required_source_snapshot_pairs), key=_source_pair_key))
        if len(required_pairs) != len(self.required_source_snapshot_pairs):
            raise TrainingSnapshotError("required_source_snapshot_pairs must be unique")
        required_pairs = tuple(required_pairs)
        source_pairs_set = set(ordered_pairs)
        for pair in required_pairs:
            if pair not in source_pairs_set:
                raise TrainingSnapshotError("required_source_snapshot_pairs must reference declared source_snapshot_pairs")
        optional_pairs = tuple(sorted(set(self.optional_source_snapshot_pairs), key=_source_pair_key))
        if len(optional_pairs) != len(self.optional_source_snapshot_pairs):
            raise TrainingSnapshotError("optional_source_snapshot_pairs must be unique")
        for pair in optional_pairs:
            if pair not in source_pairs_set:
                raise TrainingSnapshotError("optional_source_snapshot_pairs must reference declared source_snapshot_pairs")
        object.__setattr__(self, "required_source_snapshot_pairs", required_pairs)
        object.__setattr__(self, "optional_source_snapshot_pairs", optional_pairs)

        ordered_rows = tuple(
            sorted(self.source_rows, key=lambda row: (row.source_id, row.source_snapshot_id))
        )
        if ordered_rows != self.source_rows:
            object.__setattr__(self, "source_rows", ordered_rows)

        source_rows_by_id: dict[str, SourceSnapshotRowSummary] = {}
        source_rows_by_snapshot_id: dict[str, list[SourceSnapshotRowSummary]] = {}
        for row in self.source_rows:
            if row.source_id in source_rows_by_id:
                raise TrainingSnapshotError("source_rows must reference unique source ids")
            source_rows_by_id[row.source_id] = row
            source_rows_by_snapshot_id.setdefault(row.source_snapshot_id or "", []).append(row)

            if row.source_id.strip() == "":
                raise TrainingSnapshotError("source_rows require source_id")
            if row.source_name.strip() == "":
                raise TrainingSnapshotError("source_rows require source_name")
            if row.source_snapshot_id is None:
                raise TrainingSnapshotError("source_rows require source_snapshot_id")
            if row.source_snapshot_id not in pair_hashes_by_snapshot:
                raise TrainingSnapshotError("source_rows contain unexpected source snapshot pair id")
            if row.source_snapshot_content_sha256 == pair_hashes_by_snapshot[row.source_snapshot_id]:
                raise TrainingSnapshotError(
                    "source_rows must not use source snapshot manifest hash as content hash"
                )
            if row.source_snapshot_content_sha256 is None:
                raise TrainingSnapshotError("source_rows require source_snapshot_content_sha256")
            _require_hash(row.source_snapshot_content_sha256, field_name="source_snapshot_content_sha256")
            if row.row_count < 0:
                raise TrainingSnapshotError("summary row_count cannot be negative")
            if row.stale_count < 0:
                raise TrainingSnapshotError("summary stale_count cannot be negative")
            if row.latest_available_at is None:
                raise TrainingSnapshotError("source row summaries require latest_available_at")
            latest_available = parse_rfc3339(row.latest_available_at)
            if latest_available > parsed_train_cutoff:
                raise TrainingSnapshotError("source row summary latest_available_at must be <= train_cutoff")
            if latest_available > parsed_as_of:
                raise TrainingSnapshotError("source row summary latest_available_at must be <= as_of")
            if row.source_snapshot_id == "" or row.source_id == "":
                raise TrainingSnapshotError("source rows require stable identifiers")

        expected_source_rows = source_rows_from_snapshot_rows(
            resolved_source_snapshot.rows
        )
        if self.source_rows != expected_source_rows:
            raise TrainingSnapshotError(
                "source_rows must exactly match summaries derived from the resolved source manifest"
            )

        if self.status == "ok" and resolved_source_snapshot.status != "ok":
            raise TrainingSnapshotError(
                "status 'ok' requires the resolved source manifest to have status 'ok'"
            )

        if self.status == "ok" and (
            self.environment_lock_locator is None
            or self.environment_lock_sha256 is None
        ):
            raise TrainingSnapshotError(
                "status 'ok' requires environment lock locator and raw-byte hash"
            )
        if self.environment_lock_locator is not None:
            try:
                if self.environment_lock_sha256 is None:
                    raise TrainingSnapshotError(
                        "environment_lock_locator requires environment_lock_sha256"
                    )
                _require_hash(self.environment_lock_sha256, field_name="environment_lock_sha256")
                if self.environment_lock_sha256 in pair_hashes_by_snapshot.values():
                    raise TrainingSnapshotError(
                        "environment_lock_sha256 cannot reuse source snapshot manifest hash"
                    )
                normalized_locator = _normalize_repo_path(self.environment_lock_locator)
                if normalized_locator != self.environment_lock_locator:
                    raise TrainingSnapshotError(
                        "environment_lock_locator must use normalized repository-relative path"
                    )
                environment_lock_path = _resolve_repo_relative_path(normalized_locator)
                if not environment_lock_path.is_file():
                    raise TrainingSnapshotError(
                        "environment_lock_locator must reference a repository file"
                    )
                if sha256_hex_bytes(environment_lock_path.read_bytes()) != self.environment_lock_sha256:
                    raise TrainingSnapshotError(
                        "environment_lock_sha256 must match repository file"
                    )
            except SourceSnapshotError as err:
                raise TrainingSnapshotError(str(err)) from err
        elif self.environment_lock_sha256 is not None:
            raise TrainingSnapshotError(
                "environment_lock_sha256 requires environment_lock_locator"
            )

        for field_name, commit_hash in (("code_commit", self.code_commit), ("candidate_code_commit", self.candidate_code_commit)):
            if commit_hash is None:
                continue
            if not re.fullmatch(r"^[a-f0-9]{40}$", commit_hash):
                raise TrainingSnapshotError(f"{field_name} must be a 40-hex hash when present")
            try:
                _assert_commit_tree_matches_snapshot(
                    commit_hash, self.source_tree_allowlist, self.source_tree_sha256
                )
            except SourceSnapshotError as err:
                raise TrainingSnapshotError(str(err)) from err

        if self.duplicate_count < 0:
            raise TrainingSnapshotError("duplicate_count cannot be negative")
        if self.correction_count < 0:
            raise TrainingSnapshotError("correction_count cannot be negative")
        if self.missingness_count < 0:
            raise TrainingSnapshotError("missingness_count cannot be negative")
        if self.conflict_count < 0:
            raise TrainingSnapshotError("conflict_count cannot be negative")
        if self.identity_audit_count < 0:
            raise TrainingSnapshotError("identity_audit_count cannot be negative")

        if len(self.correction_lines) != self.correction_count:
            raise TrainingSnapshotError("correction_count must match length of correction_lines")
        if any(not item.strip() for item in self.correction_lines):
            raise TrainingSnapshotError("correction_lines require non-empty values")
        object.__setattr__(self, "correction_lines", tuple(self.correction_lines))
        object.__setattr__(self, "supersession_lines", tuple(self.supersession_lines))
        if any(not item.strip() for item in self.supersession_lines):
            raise TrainingSnapshotError("supersession_lines require non-empty values")

        min_event_at = parse_rfc3339(self.min_event_at) if self.min_event_at is not None else None
        max_event_at = parse_rfc3339(self.max_event_at) if self.max_event_at is not None else None
        min_available_at = parse_rfc3339(self.min_available_at) if self.min_available_at is not None else None
        max_available_at = parse_rfc3339(self.max_available_at) if self.max_available_at is not None else None

        if (min_event_at is None) != (max_event_at is None):
            raise TrainingSnapshotError("status requires both min_event_at and max_event_at when either is set")
        if (min_available_at is None) != (max_available_at is None):
            raise TrainingSnapshotError("status requires both min_available_at and max_available_at when either is set")
        if min_event_at is not None and max_event_at is not None and min_event_at > max_event_at:
            raise TrainingSnapshotError("min_event_at cannot be later than max_event_at")
        if min_available_at is not None and max_available_at is not None and min_available_at > max_available_at:
            raise TrainingSnapshotError("min_available_at cannot be later than max_available_at")
        if min_event_at is not None and min_event_at > parsed_train_cutoff:
            raise TrainingSnapshotError("min_event_at cannot be after train_cutoff")
        if min_available_at is not None and min_available_at > parsed_train_cutoff:
            raise TrainingSnapshotError("min_available_at cannot be after train_cutoff")

        if parsed_train_cutoff > parsed_as_of:
            raise TrainingSnapshotError("train_cutoff cannot be later than snapshot as_of")
        if max_event_at is not None and max_event_at > parsed_train_cutoff:
            raise TrainingSnapshotError("max_event_at must be <= train_cutoff")
        if max_available_at is not None and max_available_at > parsed_train_cutoff:
            raise TrainingSnapshotError("max_available_at must be <= train_cutoff")

        required_for_status = required_pairs or ordered_pairs
        if self.status == "ok":
            if not required_for_status:
                raise TrainingSnapshotError("status 'ok' requires at least one required source snapshot pair")
            required_snapshot_ids = [pair[0] for pair in required_for_status]
            stale_snapshot_ids = [
                pair_id for pair_id in required_snapshot_ids
                if not source_rows_by_snapshot_id.get(pair_id) or any(row.stale_count > 0 for row in source_rows_by_snapshot_id[pair_id])
            ]
            if stale_snapshot_ids:
                raise TrainingSnapshotError("status 'ok' requires required source rows to be fresh")
            if self.duplicate_count > 0:
                raise TrainingSnapshotError("status 'ok' cannot include duplicate_count")
            if self.correction_count > 0:
                raise TrainingSnapshotError("status 'ok' cannot include correction_count")
            if self.missingness_count > 0:
                raise TrainingSnapshotError("status 'ok' cannot include missingness_count")
            if self.conflict_count > 0:
                raise TrainingSnapshotError("status 'ok' cannot include conflict_count")
            if self.min_event_at is None or self.max_event_at is None:
                raise TrainingSnapshotError("status 'ok' requires min/max event bounds")
            if self.min_available_at is None or self.max_available_at is None:
                raise TrainingSnapshotError("status 'ok' requires min/max available bounds")
            if any(row.row_count <= 0 for row in self.source_rows):
                raise TrainingSnapshotError("status 'ok' requires complete source row summaries")

        if self.require_no_stale_required:
            stale_snapshot_ids = [
                pair_id for pair_id in [pair[0] for pair in required_for_status]
                if not source_rows_by_snapshot_id.get(pair_id) or any(row.stale_count > 0 for row in source_rows_by_snapshot_id[pair_id])
            ]
            if stale_snapshot_ids:
                raise TrainingSnapshotError(
                    "require_no_stale_required cannot be true when required source snapshot pairs are stale"
                )

        computed_count = self.row_count_effective
        if self.row_count != computed_count:
            raise TrainingSnapshotError("row_count must equal source-row aggregate row_count")

        try:
            count_locator = _require_exact_repo_locator(
                self.row_count_evidence_locator, "row_count_evidence_locator"
            )
            _require_hash(
                self.row_count_evidence_sha256,
                field_name="row_count_evidence_sha256",
            )
            count_rows = _load_count_evidence(
                count_locator, self.row_count_evidence_sha256
            )
        except SourceSnapshotError as err:
            raise TrainingSnapshotError(str(err)) from err
        object.__setattr__(self, "row_count_evidence_locator", count_locator)

        derived_count_maps = _derive_count_maps(count_rows)
        declared_count_maps = {
            "year": dict(self.row_count_by_year),
            "league": dict(self.row_count_by_league),
            "tier": dict(self.row_count_by_tier),
            "patch": dict(self.row_count_by_patch),
            "source": dict(self.row_count_by_source),
        }
        for key, derived in derived_count_maps.items():
            if declared_count_maps[key] != derived:
                raise TrainingSnapshotError(
                    f"row_count_by_{key} must equal the hashed row-membership evidence"
                )
        if sum(row["row_count"] for row in count_rows) != computed_count:
            raise TrainingSnapshotError(
                "hashed row-membership evidence total must equal row_count"
            )

        if set(self.row_count_by_source.keys()) != set(source_rows_by_id.keys()):
            raise TrainingSnapshotError("row_count_by_source keys must align with source_rows source ids")
        for source_id, summary in source_rows_by_id.items():
            if self.row_count_by_source[source_id] != summary.row_count:
                raise TrainingSnapshotError("row_count_by_source must align with source row counts")

        if self.status == "ok":
            if required_pairs != ordered_pairs:
                raise TrainingSnapshotError(
                    "status 'ok' requires exact required source manifest pair coverage"
                )
            if optional_pairs:
                raise TrainingSnapshotError(
                    "status 'ok' cannot relabel the sole exact source manifest as optional"
                )

        expected_id = _make_training_snapshot_id(self)
        if self.snapshot_id:
            if self.snapshot_id != expected_id:
                raise TrainingSnapshotError("snapshot_id must be derived from ordered canonical snapshot payload")
        else:
            object.__setattr__(self, "snapshot_id", expected_id)
        _reject_forbidden_recursive(self._payload_for_id(), "training snapshot payload")

    @property
    def source_snapshot_ids(self) -> tuple[str, ...]:
        return tuple(source_id for source_id, _ in self.source_snapshot_pairs)

    @property
    def row_count_effective(self) -> int:
        return sum(source_row.row_count for source_row in self.source_rows)

    @property
    def source_pairs_sorted(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted(self.source_snapshot_pairs, key=_source_pair_key))

    def _ordered_sources(self) -> tuple[str]:
        return tuple(sorted(self.source_snapshot_ids))

    def _payload_for_id(self) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "model_version": self.model_version,
            "adapter_version": self.adapter_version,
            "code_version": self.code_version,
            "as_of": self.as_of,
            "train_cutoff": self.train_cutoff,
            "source_manifest_locator": self.source_manifest_locator,
            "source_manifest_object_sha256": self.source_manifest_object_sha256,
            "source_snapshot_pairs": [
                {"source_snapshot_id": source_id, "source_snapshot_sha256": source_hash}
                for source_id, source_hash in self.source_pairs_sorted
            ],
            "contract_tree_sha256": self.contract_tree_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "source_tree_allowlist": sorted(self.source_tree_allowlist),
            "row_count_evidence_locator": self.row_count_evidence_locator,
            "row_count_evidence_sha256": self.row_count_evidence_sha256,
            "row_count": self.row_count,
            "row_count_by_year": dict(sorted(self.row_count_by_year.items(), key=lambda item: item[0])),
            "row_count_by_league": dict(sorted(self.row_count_by_league.items(), key=lambda item: item[0])),
            "row_count_by_tier": dict(sorted(self.row_count_by_tier.items(), key=lambda item: item[0])),
            "row_count_by_patch": dict(sorted(self.row_count_by_patch.items(), key=lambda item: item[0])),
            "row_count_by_source": dict(sorted(self.row_count_by_source.items(), key=lambda item: item[0])),
            "created_at": self.created_at,
            "taxonomy_version": self.taxonomy_version,
            "crosswalk_version": self.crosswalk_version,
            "inclusion_filters": sorted(self.inclusion_filters),
            "exclusion_filters": sorted(self.exclusion_filters),
            "min_event_at": self.min_event_at,
            "max_event_at": self.max_event_at,
            "min_available_at": self.min_available_at,
            "max_available_at": self.max_available_at,
            "duplicate_count": self.duplicate_count,
            "correction_count": self.correction_count,
            "missingness_count": self.missingness_count,
            "conflict_count": self.conflict_count,
            "identity_audit_count": self.identity_audit_count,
            "split_assignment_ids": sorted(self.split_assignment_ids),
            "split_assignment_locators": list(self.split_assignment_locators),
            "split_assignment_sha256s": list(self.split_assignment_sha256s),
            "environment_lock_sha256": self.environment_lock_sha256,
            "environment_lock_locator": self.environment_lock_locator,
            "candidate_code_commit": self.candidate_code_commit,
            "code_commit": self.code_commit,
            "supersession_lines": list(self.supersession_lines),
            "correction_lines": list(self.correction_lines),
            "required_source_snapshot_pairs": sorted(self.required_source_snapshot_pairs),
            "optional_source_snapshot_pairs": sorted(self.optional_source_snapshot_pairs),
            "source_rows": [
                {
                    "source_id": source.source_id,
                    "source_name": source.source_name,
                    "row_count": source.row_count,
                    "stale_count": source.stale_count,
                    "latest_available_at": source.latest_available_at,
                    "source_snapshot_id": source.source_snapshot_id,
                    "source_snapshot_content_sha256": source.source_snapshot_content_sha256,
                }
                for source in sorted(self.source_rows, key=_summary_key)
            ],
            "source_rows_order": self._ordered_sources(),
            "status": self.status,
            "require_no_stale_required": self.require_no_stale_required,
        }
        return payload

    def to_payload(self) -> dict[str, Any]:
        payload = self._payload_for_id()
        payload["snapshot_id"] = self.snapshot_id
        return payload

    def sha256(self) -> str:
        return sha256_hex(self.to_payload())

    def write(self, path: Path) -> "ArtifactWriteResult":
        payload = self.to_payload()
        canonical_bytes = canonical_json_bytes(payload) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes)
        object_digest = sha256_bytes(canonical_json_bytes(payload))
        file_digest = sha256_bytes(canonical_bytes)
        return ArtifactWriteResult(object_sha256=object_digest, file_sha256=file_digest, path=path)


@dataclass(frozen=True)
class LineageReport:
    """Cross-manifest lineage and freshness checks."""

    manifest_id: str
    model_version: str
    source_snapshot_id: str
    training_snapshot_id: str
    source_manifest_locator: str
    source_manifest_object_sha256: str
    training_manifest_locator: str
    training_manifest_object_sha256: str
    source_snapshot_tree_sha256: str
    training_snapshot_tree_sha256: str
    source_tree_sha256: str
    source_tree_allowlist: tuple[str, ...]
    as_of: str
    generated_at: str
    source_snapshot_ids: tuple[str, ...]
    source_snapshot_hashes: tuple[str, ...]
    source_snapshot_pairs: tuple[tuple[str, str], ...]
    source_snapshot_row_count: int
    training_row_count: int
    freshness_report: tuple[SourceSnapshotRowSummary, ...] = ()
    artifact_manifest_id: str | None = None
    source_tree_match: bool = True
    require_tree_match: bool = True
    duplicate_count: int = 0
    correction_count: int = 0
    missingness_count: int = 0
    conflict_count: int = 0
    missing_required_sources: tuple[str, ...] = ()
    required_source_snapshot_pairs: tuple[tuple[str, str], ...] = ()
    optional_source_snapshot_pairs: tuple[tuple[str, str], ...] = ()
    map_appearance_evidence: tuple[ChampionPatchRoleAppearanceRow, ...] = ()
    status: str = "ok"
    completeness_issues: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        as_of = parse_rfc3339(self.as_of)
        generated_at = parse_rfc3339(self.generated_at)
        if generated_at < as_of:
            raise SourceSnapshotError("generated_at cannot be earlier than lineage as_of")
        if not self.manifest_id:
            raise SourceSnapshotError("manifest_id required")
        if not self.model_version:
            raise SourceSnapshotError("model_version required")
        if not _is_sha256(self.source_snapshot_tree_sha256):
            raise SourceSnapshotError("invalid source_snapshot_tree_sha256")
        if not _is_sha256(self.training_snapshot_tree_sha256):
            raise SourceSnapshotError("invalid training_snapshot_tree_sha256")
        if not _is_sha256(self.source_tree_sha256):
            raise SourceSnapshotError("invalid source_tree_sha256")
        if self.status not in _SOURCE_SNAPSHOT_ALLOWED_STATUS:
            raise SourceSnapshotError(f"invalid status: {self.status}")
        _validate_forbidden_filters(self.completeness_issues, "completeness_issues")

        source_snapshot = _load_source_snapshot_manifest(
            _require_exact_repo_locator(
                self.source_manifest_locator, "source_manifest_locator"
            ),
            self.source_manifest_object_sha256,
        )
        training_snapshot = _load_training_snapshot_manifest(
            _require_exact_repo_locator(
                self.training_manifest_locator, "training_manifest_locator"
            ),
            self.training_manifest_object_sha256,
        )
        if (
            self.source_manifest_locator != training_snapshot.source_manifest_locator
            or self.source_manifest_object_sha256
            != training_snapshot.source_manifest_object_sha256
        ):
            raise SourceSnapshotError(
                "lineage source manifest reference must match training manifest exactly"
            )
        exact_pairs = training_snapshot.source_snapshot_pairs
        exact_ids = tuple(pair[0] for pair in exact_pairs)
        exact_hashes = tuple(pair[1] for pair in exact_pairs)
        if self.source_snapshot_pairs != exact_pairs:
            raise SourceSnapshotError(
                "lineage source_snapshot_pairs must exactly match training manifest"
            )
        if self.source_snapshot_ids != exact_ids:
            raise SourceSnapshotError(
                "lineage source_snapshot_ids must exactly match canonical pairs"
            )
        if self.source_snapshot_hashes != exact_hashes:
            raise SourceSnapshotError(
                "lineage source_snapshot_hashes must exactly match canonical pairs"
            )
        if self.source_snapshot_id != source_snapshot.snapshot_id:
            raise SourceSnapshotError(
                "lineage source_snapshot_id must match referenced source manifest"
            )
        if self.training_snapshot_id != training_snapshot.snapshot_id:
            raise SourceSnapshotError(
                "lineage training_snapshot_id must match referenced training manifest"
            )
        if not (
            self.model_version
            == source_snapshot.model_version
            == training_snapshot.model_version
        ):
            raise SourceSnapshotError("lineage model versions must agree exactly")
        exact_tree = source_snapshot.source_tree_sha256
        if not (
            self.source_snapshot_tree_sha256
            == self.training_snapshot_tree_sha256
            == self.source_tree_sha256
            == training_snapshot.source_tree_sha256
            == exact_tree
        ):
            raise SourceTreeMismatchError(
                "every source-tree digest in lineage must agree exactly"
            )
        if tuple(self.source_tree_allowlist) != source_snapshot.source_tree_allowlist:
            raise SourceSnapshotError(
                "lineage source_tree_allowlist must match referenced manifests exactly"
            )
        if self.require_tree_match and not self.source_tree_match:
            raise SourceSnapshotError("source_tree_match is required")
        if self.source_snapshot_row_count != source_snapshot.row_count:
            raise SourceSnapshotError(
                "source_snapshot_row_count must match referenced source manifest"
            )
        if self.training_row_count != training_snapshot.row_count:
            raise SourceSnapshotError(
                "training_row_count must match referenced training manifest"
            )
        if self.required_source_snapshot_pairs != training_snapshot.required_source_snapshot_pairs:
            raise SourceSnapshotError(
                "required source pairs must match training manifest exactly"
            )
        if self.optional_source_snapshot_pairs != training_snapshot.optional_source_snapshot_pairs:
            raise SourceSnapshotError(
                "optional source pairs must match training manifest exactly"
            )
        if parse_rfc3339(training_snapshot.as_of) != as_of:
            raise SourceSnapshotError("lineage as_of must match training manifest as_of")

        declared_pairs = set(exact_pairs)
        parsed_missing_required_sources: list[tuple[str, str]] = []
        for pair in self.missing_required_sources:
            if isinstance(pair, Mapping):
                pair_id = pair.get("source_snapshot_id")
                pair_hash = (
                    pair.get("source_snapshot_content_sha256")
                    or pair.get("source_snapshot_sha")
                    or pair.get("source_snapshot_sha256")
                )
            else:
                try:
                    pair_id, pair_hash = pair  # type: ignore[misc]
                except Exception:
                    raise SourceSnapshotError(
                        "missing_required_sources entries must be source snapshot id/hash pairs"
                    )
            if pair_id is None or pair_hash is None:
                raise SourceSnapshotError(
                    "missing_required_sources entries must be source snapshot id/hash pairs"
                )
            if not isinstance(pair_id, str) or not isinstance(pair_hash, str):
                raise SourceSnapshotError(
                    "missing_required_sources entries must be source snapshot id/hash pairs"
                )
            source_pair = (pair_id, pair_hash)
            if source_pair not in declared_pairs:
                raise SourceSnapshotError(
                    "missing_required_sources must reference declared source snapshot pairs"
                )
            parsed_missing_required_sources.append(source_pair)
        if parsed_missing_required_sources:
            object.__setattr__(self, "missing_required_sources", tuple(parsed_missing_required_sources))

        if self.status == "ok":
            if self.source_tree_match is not True:
                raise SourceSnapshotError("status 'ok' requires source_tree_match")
            if self.missingness_count > 0:
                raise SourceSnapshotError("status 'ok' cannot include missingness_count")
            if self.duplicate_count > 0:
                raise SourceSnapshotError("status 'ok' cannot include duplicate_count")
            if self.correction_count > 0:
                raise SourceSnapshotError("status 'ok' cannot include correction_count")
            if self.conflict_count > 0:
                raise SourceSnapshotError("status 'ok' cannot include conflict_count")
            if self.missing_required_sources:
                raise SourceSnapshotError("status 'ok' cannot include missing required sources")
            if self.completeness_issues:
                raise SourceSnapshotError("status 'ok' cannot include completeness_issues")
            for row in self.freshness_report:
                if not row.latest_available_at:
                    raise SourceSnapshotError(
                        "status 'ok' requires freshness rows to include latest_available_at"
                    )
                parse_rfc3339(row.latest_available_at)
                if row.row_count <= 0:
                    raise SourceSnapshotError("status 'ok' requires freshness rows to be complete")
                if row.stale_count:
                    raise SourceSnapshotError("status 'ok' requires freshness rows to be complete")
            if not self.freshness_report:
                raise SourceSnapshotError(
                    "status 'ok' requires nonempty exact freshness coverage"
                )
            if self.freshness_report != training_snapshot.source_rows:
                raise SourceSnapshotError(
                    "status 'ok' freshness coverage must match every required source with no extras"
                )
            if self.map_appearance_evidence != source_snapshot.champion_patch_role_counts:
                raise SourceSnapshotError(
                    "status 'ok' appearance evidence must match referenced source manifest"
                )
        _reject_forbidden_recursive(self.to_payload(), "lineage payload")

    def to_payload(self) -> dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "model_version": self.model_version,
            "source_snapshot_id": self.source_snapshot_id,
            "training_snapshot_id": self.training_snapshot_id,
            "source_manifest_locator": self.source_manifest_locator,
            "source_manifest_object_sha256": self.source_manifest_object_sha256,
            "training_manifest_locator": self.training_manifest_locator,
            "training_manifest_object_sha256": self.training_manifest_object_sha256,
            "source_snapshot_tree_sha256": self.source_snapshot_tree_sha256,
            "training_snapshot_tree_sha256": self.training_snapshot_tree_sha256,
            "source_tree_sha256": self.source_tree_sha256,
            "source_tree_allowlist": list(sorted(self.source_tree_allowlist)),
            "as_of": self.as_of,
            "generated_at": self.generated_at,
            "source_snapshot_pairs": [
                {"source_snapshot_id": source_id, "source_snapshot_sha256": source_hash}
                for source_id, source_hash in sorted(self.source_snapshot_pairs, key=_source_pair_key)
            ],
            "source_snapshot_ids": list(sorted(self.source_snapshot_ids)),
            "source_snapshot_hashes": list(sorted(self.source_snapshot_hashes)),
            "source_snapshot_row_count": self.source_snapshot_row_count,
            "training_row_count": self.training_row_count,
            "source_tree_match": self.source_tree_match,
            "artifact_manifest_id": self.artifact_manifest_id,
            "status": self.status,
            "duplicate_count": self.duplicate_count,
            "correction_count": self.correction_count,
            "missingness_count": self.missingness_count,
            "conflict_count": self.conflict_count,
            "missing_required_sources": list(self.missing_required_sources),
            "required_source_snapshot_pairs": list(self.required_source_snapshot_pairs),
            "optional_source_snapshot_pairs": list(self.optional_source_snapshot_pairs),
            "freshness_report": [
                {
                    "source_id": source.source_id,
                    "source_name": source.source_name,
                    "row_count": source.row_count,
                    "stale_count": source.stale_count,
                    "latest_available_at": source.latest_available_at,
                    "source_snapshot_id": source.source_snapshot_id,
                    "source_snapshot_content_sha256": source.source_snapshot_content_sha256,
                }
                for source in sorted(self.freshness_report, key=_summary_key)
            ],
            "map_appearance_evidence": [row.to_payload() for row in sorted(self.map_appearance_evidence, key=_appearance_key)],
            "completeness_issues": list(self.completeness_issues),
        }

    def write(self, path: Path) -> "ArtifactWriteResult":
        payload = self.to_payload()
        canonical_bytes = canonical_json_bytes(payload) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_bytes)
        object_digest = sha256_bytes(canonical_json_bytes(payload))
        file_digest = sha256_bytes(canonical_bytes)
        return ArtifactWriteResult(object_sha256=object_digest, file_sha256=file_digest, path=path)


@dataclass(frozen=True)
class ArtifactWriteResult:
    """Hash output from manifest writes."""

    object_sha256: str
    file_sha256: str
    path: Path

    def as_payload(self) -> dict[str, str]:
        return {
            "object_sha256": self.object_sha256,
            "file_sha256": self.file_sha256,
            "path": str(self.path),
        }


def _coerce_utc_datetime(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise SourceSnapshotSnapshotError("timestamp must include timezone")
        return value.astimezone(timezone.utc)
    parsed = datetime.fromisoformat(_normalize_dt(value))
    if parsed.tzinfo is None:
        raise SourceSnapshotSnapshotError("timestamp must RFC3339 UTC")
    return parsed.astimezone(timezone.utc)


def _normalize_dt(value: str) -> str:
    if value.endswith("Z"):
        return value.replace("Z", "+00:00")
    return value


def _validate_snapshot_row(row: SourceSnapshotRow) -> None:
    _validate_source_content_path(row.source_content_path)
    _require_non_empty(row.source_snapshot_id, "source_snapshot_id")
    _require_non_empty(row.source_id, "source_id")
    _require_non_empty(row.source_name, "source_name")
    _require_non_empty(row.source_snapshot_row_id, "source_snapshot_row_id")
    _require_non_empty(row.source_record_id, "source_record_id")
    _require_hash(row.source_content_sha256, field_name="source_content_sha256")
    if row.source_content_object_sha256 is not None:
        _require_hash(row.source_content_object_sha256, field_name="source_content_object_sha256")
    _require_non_empty(row.source_content_path, "source_content_path")
    if row.source_content_size_bytes < 0:
        raise SourceSnapshotSnapshotError("source_content_size_bytes cannot be negative")
    _require_hash(row.source_tree_sha256, field_name="source_tree_sha256")
    if row.row_count < 0:
        raise SourceSnapshotSnapshotError("row_count cannot be negative")
    source_updated_at = parse_rfc3339(row.source_updated_at)
    observed_at = parse_rfc3339(row.observed_at)
    available_at = parse_rfc3339(row.available_at)

    if source_updated_at > observed_at:
        raise SourceSnapshotSnapshotError("source_updated_at cannot be after observed_at")
    if available_at > observed_at:
        raise SourceSnapshotSnapshotError("available_at cannot be after observed_at")

    resolved = _resolve_source_content_path(row.source_content_path)
    if not resolved.is_file():
        raise SourceSnapshotError(f"source content path must reference an existing file: {row.source_content_path}")

    actual_bytes = resolved.read_bytes()
    actual_size = len(actual_bytes)
    actual_hash = sha256_hex_bytes(actual_bytes)
    if actual_size != row.source_content_size_bytes:
        raise SourceSnapshotError("source_content_size_bytes mismatch")
    if actual_hash != row.source_content_sha256:
        raise SourceSnapshotError("source_content_sha256 mismatch with local content")

    if row.source_content_object_sha256 is not None:
        if resolved.suffix.lower() != ".json":
            raise SourceSnapshotSnapshotError(
                "source_content_object_sha256 requires a JSON source file"
            )
        try:
            payload_obj = json.loads(actual_bytes.decode("utf-8"))
        except Exception:
            raise SourceSnapshotSnapshotError("source_content_object_sha256 requires JSON content")
        computed = sha256_canonical_object_hash(payload_obj)
        if computed != row.source_content_object_sha256:
            raise SourceSnapshotSnapshotError("source_content_object_sha256 mismatch")


def _validate_patch_role_row(row: ChampionPatchRoleAppearanceRow) -> None:
    _require_non_empty(row.league_id, "league_id")
    _require_non_empty(row.patch_id, "patch_id")
    if row.role not in ROLES:
        raise SourceSnapshotSnapshotError(f"invalid role: {row.role}")
    _require_non_empty(row.champion_id, "champion_id")
    if row.verified_appearance_count < 0:
        raise SourceSnapshotSnapshotError("verified_appearance_count cannot be negative")
    _require_non_empty(row.source_snapshot_id, "source_snapshot_id")
    _require_non_empty(row.source_snapshot_row_id, "source_snapshot_row_id")
    _require_non_empty(row.source_id, "source_id")
    _validate_contract_tree_hash(CONTRACT_TREE_SHA256)
    _require_hash(row.source_snapshot_content_sha256, field_name="source_snapshot_content_sha256")
    _require_hash(row.source_tree_sha256, field_name="source_tree_sha256")

    parse_rfc3339(row.as_of)
    parse_rfc3339(row.available_at)

    if row.status not in {"available", "unavailable"}:
        raise SourceSnapshotSnapshotError("status must be 'available' or 'unavailable'")
    if row.status == "available":
        if row.unavailable_reason is not None:
            raise SourceSnapshotSnapshotError("available rows must not include unavailable_reason")
    else:
        if row.verified_appearance_count != 0:
            raise SourceSnapshotSnapshotError("unavailable rows must keep verified_appearance_count == 0")
        if not row.unavailable_reason:
            raise SourceSnapshotSnapshotError("unavailable rows require unavailable_reason")


def _validate_contract_tree_hash(value: str) -> None:
    if value != CONTRACT_TREE_SHA256:
        raise SourceSnapshotError("contract_tree_sha256 mismatch")


def _is_sha256(value: str) -> bool:
    return bool(_SHA256_RE.fullmatch(value))


def _require_hash(value: str, field_name: str) -> None:
    if not _is_sha256(value):
        raise SourceSnapshotSnapshotError(f"{field_name} must be a 64-char sha256 hash")


def _require_non_empty(value: str | None, field_name: str) -> str:
    if value is None or not value.strip():
        raise SourceSnapshotError(f"{field_name} is required")
    return value.strip()


def _appearance_key(row: ChampionPatchRoleAppearanceRow) -> tuple[str, str, str, str]:
    return (row.league_id, row.patch_id, row.role, row.champion_id)


def _summary_key(summary: SourceSnapshotRowSummary) -> tuple[str, str]:
    return (summary.source_id, summary.source_name)


def source_snapshot_row_lookup(rows: Iterable[SourceSnapshotRow]) -> dict[str, SourceSnapshotRow]:
    return {row.source_snapshot_row_id: row for row in rows}


def _source_row_lookup(rows: Iterable[SourceSnapshotRow]) -> dict[str, SourceSnapshotRow]:
    return source_snapshot_row_lookup(rows)


def canonicalize_snapshot_id(prefix: str, hashed_payload: str) -> str:
    return f"{prefix}:{hashed_payload}"


def _source_pair_key(item: tuple[str, str]) -> str:
    source_id, source_hash = item
    return f"{source_id}|{source_hash}"


def leaf_source_row_evidence(
    rows: Sequence[SourceSnapshotRow],
) -> tuple[tuple[str, str, str], ...]:
    """Return leaf/source-row evidence, never source-manifest ID/hash pairs."""

    return tuple(
        sorted(
            {
                (
                    row.source_snapshot_row_id,
                    row.source_id,
                    row.source_content_sha256,
                )
                for row in rows
            },
        )
    )


def source_rows_from_snapshot_rows(rows: Sequence[SourceSnapshotRow]) -> tuple[SourceSnapshotRowSummary, ...]:
    rows_by_source: dict[str, SourceSnapshotRowSummary] = {}
    for row in rows:
        key = row.source_id
        current = rows_by_source.get(key)
        if current is None:
            rows_by_source[key] = SourceSnapshotRowSummary(
                source_id=row.source_id,
                source_name=row.source_name,
                row_count=row.row_count,
                stale_count=1 if row.is_stale else 0,
                latest_available_at=row.available_at,
                source_snapshot_id=row.source_snapshot_id,
                source_snapshot_content_sha256=row.source_content_sha256,
            )
            continue

        latest_available_at = max(
            parse_rfc3339(current.latest_available_at),
            parse_rfc3339(row.available_at),
        )
        latest_available_at = to_rfc3339(latest_available_at)
        if current.source_snapshot_content_sha256 != row.source_content_sha256:
            raise SourceSnapshotError("duplicate source ids cannot have multiple source content hashes")
        rows_by_source[key] = SourceSnapshotRowSummary(
            source_id=current.source_id,
            source_name=current.source_name,
            row_count=current.row_count + row.row_count,
            stale_count=current.stale_count + (1 if row.is_stale else 0),
            latest_available_at=latest_available_at,
            source_snapshot_id=current.source_snapshot_id,
            source_snapshot_content_sha256=current.source_snapshot_content_sha256,
        )

    return tuple(
        SourceSnapshotRowSummary(
            source_id=summary.source_id,
            source_name=summary.source_name,
            row_count=summary.row_count,
            stale_count=summary.stale_count,
            latest_available_at=summary.latest_available_at,
            source_snapshot_id=summary.source_snapshot_id,
            source_snapshot_content_sha256=summary.source_snapshot_content_sha256,
        )
        for _, summary in sorted(rows_by_source.items(), key=lambda item: item[0])
    )


def _validate_forbidden_filters(values: Iterable[str], field_name: str) -> tuple[str, ...]:
    for value in values:
        if not isinstance(value, str):
            raise SourceSnapshotError(f"{field_name} entries must be strings")
        camel_split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
        casefolded = camel_split.casefold()
        normalized = re.sub(r"[^a-z0-9]+", "", casefolded)
        tokens = tuple(re.findall(r"[a-z0-9]+", casefolded))
        has_forbidden_legacy_token = any(
            token in normalized for token in _FORBIDDEN_NORMALIZED_TOKENS
        )
        has_forbidden_standalone_token = bool(
            set(tokens) & _FORBIDDEN_STANDALONE_TOKENS
        )
        has_forbidden_compact_alias = bool(
            set(tokens) & _FORBIDDEN_COMPACT_ALIAS_TOKENS
        )
        has_forbidden_ordered_pair = any(
            _contains_ordered_token_pair(tokens, first, second)
            for first, second in _FORBIDDEN_ORDERED_TOKEN_PAIRS
        )
        if (
            has_forbidden_legacy_token
            or has_forbidden_standalone_token
            or has_forbidden_compact_alias
            or has_forbidden_ordered_pair
        ):
            raise SourceSnapshotError(f"{field_name} contains forbidden field: {value}")
    return tuple(values)


def _contains_ordered_token_pair(
    tokens: Sequence[str], first: str, second: str
) -> bool:
    first_seen = False
    for token in tokens:
        if first_seen and token == second:
            return True
        if token == first:
            first_seen = True
    return False


def _reject_forbidden_recursive(value: Any, field_name: str) -> None:
    if isinstance(value, str):
        _validate_forbidden_filters((value,), field_name)
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_forbidden_recursive(key, field_name)
            _reject_forbidden_recursive(child, field_name)
        return
    if isinstance(value, (list, tuple, set)):
        for child in value:
            _reject_forbidden_recursive(child, field_name)


def _is_source_row_stale(
    row: SourceSnapshotRow,
    *,
    as_of: datetime,
    source_updated: datetime | None = None,
) -> bool:
    source_updated_at = source_updated or parse_rfc3339(row.source_updated_at)
    if source_updated_at > as_of:
        return True
    if row.freshness_limit_seconds is None:
        return False
    age_seconds = int((as_of - source_updated_at).total_seconds())
    return age_seconds > row.freshness_limit_seconds


def _resolve_repo_relative_path(locator: str) -> Path:
    normalized = _normalize_repo_path(locator)
    base = _GIT_REPO_ROOT
    current = base
    for part in normalized.split("/"):
        current = current / part
        if current.is_symlink():
            raise SourceSnapshotError(f"environment_lock_locator cannot include symlink components: {locator}")
        if not current.exists():
            raise SourceSnapshotError(f"environment_lock_locator must reference an existing file: {locator}")

    resolved = current.resolve()
    if not str(resolved).startswith(str(base)):
        raise SourceSnapshotError(f"environment_lock_locator escapes repository root: {locator}")
    return resolved


def _assert_commit_tree_matches_snapshot(
    commit_hash: str,
    allowlist: Sequence[str],
    source_tree_sha256: str,
) -> None:
    try:
        type_result = subprocess.run(
            ["git", "-C", str(_GIT_REPO_ROOT), "cat-file", "-t", commit_hash],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        commit_type = type_result.stdout.strip()
    except subprocess.CalledProcessError as err:
        raise SourceSnapshotError("commit hash cannot be resolved") from err
    if commit_type != "commit":
        raise SourceSnapshotError(f"commit {commit_hash} is not a commit object")

    paths = tuple(sorted(set(_normalize_repo_path(path) for path in allowlist)))
    if not paths:
        raise SourceSnapshotError("commit check requires a non-empty source tree allowlist")
    allowlist_tree_sha = canonical_source_tree_sha256(_GIT_REPO_ROOT, paths)
    if allowlist_tree_sha != source_tree_sha256:
        raise SourceSnapshotError("commit tree check requires a valid source tree digest")

    for path in paths:
        try:
            result = subprocess.run(
                ["git", "-C", str(_GIT_REPO_ROOT), "cat-file", "-p", f"{commit_hash}:{path}"],
                check=True,
                capture_output=True,
                text=False,
                timeout=30,
            )
        except subprocess.CalledProcessError as err:
            raise SourceSnapshotError(
                f"commit {commit_hash} does not contain required source tree path {path}"
            ) from err

        commit_bytes = result.stdout
        local_path = _resolve_repo_relative_path(path)
        if not local_path.is_file():
            raise SourceSnapshotError(f"source tree path is not a file in working tree: {path}")
        if sha256_hex_bytes(commit_bytes) != sha256_hex_bytes(local_path.read_bytes()):
            raise SourceSnapshotError(
                f"commit {commit_hash} file mismatch for source tree path {path}"
            )



def make_default_freshness_report(
    source_rows: Iterable[SourceSnapshotRow],
    *,
    as_of: str | datetime,
    staleness_limit_seconds: int | None = None,
) -> tuple[SourceSnapshotRowSummary, ...]:
    as_of_dt = _coerce_utc_datetime(as_of)
    by_source: dict[str, list[SourceSnapshotRow]] = {}
    for row in source_rows:
        by_source.setdefault(row.source_id, []).append(row)

    reports: list[SourceSnapshotRowSummary] = []
    for source_id, rows in sorted(by_source.items()):
        stale_count = 0
        latest = None
        source_name = rows[0].source_name
        for row in rows:
            row_available = parse_rfc3339(row.available_at)
            row_freshness_limit = (
                row.freshness_limit_seconds if row.freshness_limit_seconds is not None else staleness_limit_seconds
            )
            if row_freshness_limit is not None:
                if _is_source_row_stale(
                    row,
                    as_of=as_of_dt,
                    source_updated=parse_rfc3339(row.source_updated_at),
                ):
                    stale_count += 1
            if row_available > as_of_dt:
                stale_count += 1
            if latest is None or row_available > parse_rfc3339(latest):
                latest = row.available_at
        reports.append(
            SourceSnapshotRowSummary(
                source_id=source_id,
                source_name=source_name,
                row_count=sum(row.row_count for row in rows),
                stale_count=stale_count,
                latest_available_at=latest,
                source_snapshot_id=rows[0].source_snapshot_id,
                source_snapshot_content_sha256=rows[0].source_content_sha256,
            )
        )
    return tuple(reports)


def _make_source_snapshot_id(snapshot: SourceSnapshot) -> str:
    return canonicalize_snapshot_id(_SOURCE_SNAPSHOT_PREFIX, sha256_hex(snapshot._payload_for_id()))


def _make_training_snapshot_id(snapshot: TrainingSnapshot) -> str:
    return canonicalize_snapshot_id(_TRAINING_SNAPSHOT_PREFIX, sha256_hex(snapshot._payload_for_id()))


def make_default_source_snapshot_rows(
    *,
    model_version: str,
    as_of: str | datetime,
    source_tree_sha256: str,
    source_tree_allowlist: tuple[str, ...],
    source_rows: Iterable[SourceSnapshotRow],
    champion_counts: Iterable[ChampionPatchRoleAppearanceRow],
    reviewed_at: str | None = None,
    adapter_version: str = "unknown",
    code_version: str = "unknown",
    status: str = "ok",
) -> SourceSnapshot:
    as_of_dt = _coerce_utc_datetime(as_of)
    snapshot_id = ""
    reviewed = reviewed_at or to_rfc3339(as_of_dt)
    return SourceSnapshot(
        schema_version="2.0.0",
        model_version=model_version,
        adapter_version=adapter_version,
        code_version=code_version,
        as_of=to_rfc3339(as_of_dt),
        snapshot_id=snapshot_id,
        reviewed_at=to_rfc3339(_coerce_utc_datetime(reviewed)),
        rows=tuple(source_rows),
        source_tree_sha256=source_tree_sha256,
        source_tree_allowlist=source_tree_allowlist,
        created_at=to_rfc3339(as_of_dt),
        champion_patch_role_counts=tuple(champion_counts),
        status=status,
    )


def _resolve_source_content_path(value: str) -> Path:
    path = _normalize_relative_snapshot_path(value)
    if not path.is_absolute():
        base = Path(__file__).resolve().parents[3]
        path = (base / value).resolve()
    return path


def _normalize_relative_snapshot_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise SourceSnapshotError("source content path must be relative to repository root")
    if ".." in path.parts:
        raise SourceSnapshotError("source content path cannot include traversal segments")
    if "\\" in value:
        raise SourceSnapshotError("source content path must use POSIX separators")
    return path


def _validate_source_content_path(value: str) -> None:
    path = _normalize_relative_snapshot_path(value)
    base = Path(__file__).resolve().parents[3]
    path_obj = base / path
    resolved = path_obj.resolve()
    if not str(resolved).startswith(str(base)):
        raise SourceSnapshotError("source content path escapes repository root")

    current = path_obj
    while current != base:
        if current.is_symlink():
            raise SourceSnapshotError("source content path cannot include symlinks")
        current = current.parent


def _require_exact_repo_locator(locator: str, field_name: str) -> str:
    if not isinstance(locator, str) or not locator:
        raise SourceSnapshotError(f"{field_name} is required")
    try:
        normalized = normalize_source_tree_path(locator)
    except ValueError as err:
        raise SourceSnapshotError(f"{field_name} is invalid: {err}") from err
    if normalized != locator:
        raise SourceSnapshotError(
            f"{field_name} must be a normalized repository-relative POSIX path"
        )
    try:
        resolve_repository_file(_GIT_REPO_ROOT, normalized)
    except ValueError as err:
        raise SourceSnapshotError(f"{field_name} must reference an existing file: {err}") from err
    return normalized


def _source_snapshot_from_manifest_payload(payload: Mapping[str, Any]) -> SourceSnapshot:
    try:
        snapshot_id = payload["snapshot_id"]
        rows = tuple(
            SourceSnapshotRow(
                **({"source_snapshot_id": snapshot_id} | dict(row))
            )
            for row in payload["rows"]
        )
        row_ids_by_hash = {
            row.source_content_sha256: row.source_snapshot_row_id for row in rows
        }
        appearances = []
        for appearance in payload.get("champion_patch_role_counts", ()):
            appearance_payload = dict(appearance)
            appearance_payload.setdefault("source_snapshot_id", snapshot_id)
            appearance_payload.setdefault(
                "source_snapshot_row_id",
                row_ids_by_hash[appearance_payload["source_snapshot_content_sha256"]],
            )
            appearances.append(ChampionPatchRoleAppearanceRow(**appearance_payload))
        return SourceSnapshot(
            schema_version=payload["schema_version"],
            model_version=payload["model_version"],
            adapter_version=payload["adapter_version"],
            code_version=payload["code_version"],
            as_of=payload["as_of"],
            snapshot_id=snapshot_id,
            reviewed_at=payload["reviewed_at"],
            rows=rows,
            source_tree_sha256=payload["source_tree_sha256"],
            source_tree_allowlist=tuple(payload["source_tree_allowlist"]),
            created_at=payload["created_at"],
            contract_tree_sha256=payload["contract_tree_sha256"],
            champion_patch_role_counts=tuple(appearances),
            status=payload["status"],
        )
    except (KeyError, TypeError, ValueError) as err:
        if isinstance(err, SourceSnapshotError):
            raise
        raise SourceSnapshotError(f"invalid source snapshot manifest: {err}") from err


def _load_source_snapshot_manifest(
    locator: str, expected_object_sha256: str
) -> SourceSnapshot:
    path = resolve_repository_file(_GIT_REPO_ROOT, locator)
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise SourceSnapshotError("source_manifest_locator must contain JSON") from err
    if not isinstance(payload, Mapping):
        raise SourceSnapshotError("source manifest must be a JSON object")
    actual_object_sha256 = sha256_canonical_object_hash(payload)
    if actual_object_sha256 != expected_object_sha256:
        raise SourceSnapshotError(
            "source_manifest_object_sha256 must match the canonical manifest object"
        )
    source_snapshot = _source_snapshot_from_manifest_payload(payload)
    if source_snapshot.snapshot_id != payload.get("snapshot_id"):
        raise SourceSnapshotError(
            "resolved source manifest snapshot_id failed independent recomputation"
        )
    if source_snapshot.sha256() != actual_object_sha256:
        raise SourceSnapshotError(
            "resolved source manifest object hash failed independent recomputation"
        )
    return source_snapshot


def _decode_manifest_pairs(values: Iterable[Any]) -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for value in values:
        if isinstance(value, Mapping):
            pairs.append(
                (
                    value["source_snapshot_id"],
                    value.get("source_snapshot_sha256")
                    or value["source_snapshot_content_sha256"],
                )
            )
        else:
            source_id, source_hash = value
            pairs.append((source_id, source_hash))
    return tuple(pairs)


def _training_snapshot_from_manifest_payload(
    payload: Mapping[str, Any],
) -> TrainingSnapshot:
    try:
        return TrainingSnapshot(
            schema_version=payload["schema_version"],
            model_version=payload["model_version"],
            adapter_version=payload["adapter_version"],
            code_version=payload["code_version"],
            as_of=payload["as_of"],
            train_cutoff=payload["train_cutoff"],
            source_manifest_locator=payload["source_manifest_locator"],
            source_manifest_object_sha256=payload["source_manifest_object_sha256"],
            source_snapshot_pairs=_decode_manifest_pairs(
                payload["source_snapshot_pairs"]
            ),
            source_tree_sha256=payload["source_tree_sha256"],
            source_tree_allowlist=tuple(payload["source_tree_allowlist"]),
            row_count_evidence_locator=payload["row_count_evidence_locator"],
            row_count_evidence_sha256=payload["row_count_evidence_sha256"],
            row_count_by_year=dict(payload["row_count_by_year"]),
            row_count_by_league=dict(payload["row_count_by_league"]),
            row_count_by_tier=dict(payload["row_count_by_tier"]),
            row_count_by_patch=dict(payload["row_count_by_patch"]),
            row_count_by_source=dict(payload["row_count_by_source"]),
            source_rows=tuple(
                SourceSnapshotRowSummary(**row) for row in payload["source_rows"]
            ),
            created_at=payload["created_at"],
            taxonomy_version=payload["taxonomy_version"],
            crosswalk_version=payload["crosswalk_version"],
            inclusion_filters=tuple(payload["inclusion_filters"]),
            exclusion_filters=tuple(payload["exclusion_filters"]),
            min_event_at=payload["min_event_at"],
            max_event_at=payload["max_event_at"],
            min_available_at=payload["min_available_at"],
            max_available_at=payload["max_available_at"],
            duplicate_count=payload["duplicate_count"],
            correction_count=payload["correction_count"],
            missingness_count=payload["missingness_count"],
            conflict_count=payload["conflict_count"],
            identity_audit_count=payload["identity_audit_count"],
            split_assignment_ids=tuple(payload["split_assignment_ids"]),
            split_assignment_locators=tuple(payload["split_assignment_locators"]),
            split_assignment_sha256s=tuple(payload["split_assignment_sha256s"]),
            environment_lock_sha256=payload["environment_lock_sha256"],
            environment_lock_locator=payload["environment_lock_locator"],
            candidate_code_commit=payload["candidate_code_commit"],
            code_commit=payload["code_commit"],
            supersession_lines=tuple(payload["supersession_lines"]),
            correction_lines=tuple(payload["correction_lines"]),
            contract_tree_sha256=payload["contract_tree_sha256"],
            row_count=payload["row_count"],
            require_no_stale_required=payload["require_no_stale_required"],
            required_source_snapshot_pairs=_decode_manifest_pairs(
                payload["required_source_snapshot_pairs"]
            ),
            optional_source_snapshot_pairs=_decode_manifest_pairs(
                payload["optional_source_snapshot_pairs"]
            ),
            status=payload["status"],
            snapshot_id=payload["snapshot_id"],
        )
    except (KeyError, TypeError, ValueError) as err:
        if isinstance(err, TrainingSnapshotError):
            raise
        raise TrainingSnapshotError(f"invalid training snapshot manifest: {err}") from err


def _load_training_snapshot_manifest(
    locator: str, expected_object_sha256: str
) -> TrainingSnapshot:
    path = resolve_repository_file(_GIT_REPO_ROOT, locator)
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise SourceSnapshotError("training_manifest_locator must contain JSON") from err
    if not isinstance(payload, Mapping):
        raise SourceSnapshotError("training manifest must be a JSON object")
    actual_object_sha256 = sha256_canonical_object_hash(payload)
    if actual_object_sha256 != expected_object_sha256:
        raise SourceSnapshotError(
            "training_manifest_object_sha256 must match canonical manifest object"
        )
    training_snapshot = _training_snapshot_from_manifest_payload(payload)
    if training_snapshot.snapshot_id != payload.get("snapshot_id"):
        raise SourceSnapshotError(
            "resolved training manifest snapshot_id failed independent recomputation"
        )
    if training_snapshot.sha256() != actual_object_sha256:
        raise SourceSnapshotError(
            "resolved training manifest object hash failed independent recomputation"
        )
    return training_snapshot


_COUNT_EVIDENCE_FIELDS = (
    "calendar_year",
    "league_id",
    "league_tier",
    "patch",
    "source",
    "row_count",
)


def _load_count_evidence(
    locator: str, expected_sha256: str
) -> tuple[dict[str, Any], ...]:
    path = resolve_repository_file(_GIT_REPO_ROOT, locator)
    raw_bytes = path.read_bytes()
    if sha256_hex_bytes(raw_bytes) != expected_sha256:
        raise SourceSnapshotError(
            "row_count_evidence_sha256 must match repository file raw bytes"
        )
    try:
        payload = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise SourceSnapshotError("row count evidence must contain JSON") from err
    if not isinstance(payload, list) or not payload:
        raise SourceSnapshotError("row count evidence must be a nonempty JSON array")
    _reject_forbidden_recursive(payload, "row count evidence")
    rows: list[dict[str, Any]] = []
    seen_grains: set[tuple[str, str, str, str, str]] = set()
    for raw_row in payload:
        if not isinstance(raw_row, Mapping) or set(raw_row) != set(_COUNT_EVIDENCE_FIELDS):
            raise SourceSnapshotError(
                "row count evidence rows require exactly calendar_year, league_id, "
                "league_tier, patch, source, and row_count"
            )
        row = dict(raw_row)
        for field_name in _COUNT_EVIDENCE_FIELDS[:-1]:
            if not isinstance(row[field_name], str) or not row[field_name].strip():
                raise SourceSnapshotError(
                    f"row count evidence {field_name} must be nonempty"
                )
        if (
            isinstance(row["row_count"], bool)
            or not isinstance(row["row_count"], int)
            or row["row_count"] <= 0
        ):
            raise SourceSnapshotError(
                "row count evidence row_count must be a positive integer"
            )
        grain = tuple(row[field] for field in _COUNT_EVIDENCE_FIELDS[:-1])
        if grain in seen_grains:
            raise SourceSnapshotError("row count evidence grain must be unique")
        seen_grains.add(grain)
        rows.append(row)
    return tuple(sorted(rows, key=lambda row: tuple(row[field] for field in _COUNT_EVIDENCE_FIELDS[:-1])))


def _derive_count_maps(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    dimensions = {
        "year": "calendar_year",
        "league": "league_id",
        "tier": "league_tier",
        "patch": "patch",
        "source": "source",
    }
    result: dict[str, dict[str, int]] = {}
    for output_name, field_name in dimensions.items():
        counts: dict[str, int] = {}
        for row in rows:
            key = row[field_name]
            counts[key] = counts.get(key, 0) + row["row_count"]
        result[output_name] = dict(sorted(counts.items()))
    return result


def make_training_snapshot(
    *,
    model_version: str,
    as_of: str | datetime,
    source_snapshot: SourceSnapshot,
    source_manifest_locator: str,
    row_count_evidence_locator: str,
    environment_lock_locator: str,
    environment_lock_sha256: str,
    split_assignment_ids: tuple[str, ...],
    split_assignment_locators: tuple[str, ...],
    split_assignment_sha256s: tuple[str, ...],
    min_event_at: str,
    max_event_at: str,
    min_available_at: str,
    max_available_at: str,
    row_count_by_year: Mapping[str, int],
    row_count_by_league: Mapping[str, int],
    row_count_by_tier: Mapping[str, int],
    row_count_by_patch: Mapping[str, int],
    row_count_by_source: Mapping[str, int],
    source_tree_sha256: str,
    source_tree_allowlist: tuple[str, ...],
    train_cutoff: str,
    taxonomy_version: str = "unknown",
    crosswalk_version: str = "unknown",
    inclusion_filters: tuple[str, ...] = (),
    exclusion_filters: tuple[str, ...] = (),
    code_commit: str | None = None,
    candidate_code_commit: str | None = None,
    require_no_stale_required: bool = False,
    adapter_version: str = "unknown",
    code_version: str = "unknown",
    status: str = "ok",
    correction_lines: tuple[str, ...] = (),
    required_source_snapshot_pairs: tuple[tuple[str, str], ...] = (),
    optional_source_snapshot_pairs: tuple[tuple[str, str], ...] = (),
) -> TrainingSnapshot:
    as_of_dt = _coerce_utc_datetime(as_of)
    for bound_name, bound in (
        ("min_event_at", min_event_at),
        ("max_event_at", max_event_at),
        ("min_available_at", min_available_at),
        ("max_available_at", max_available_at),
    ):
        try:
            parse_rfc3339(bound)
        except ValueError as err:
            raise TrainingSnapshotError(f"{bound_name} must be parsed RFC3339") from err
    source_manifest_path = resolve_repository_file(
        _GIT_REPO_ROOT,
        _require_exact_repo_locator(
            source_manifest_locator, "source_manifest_locator"
        ),
    )
    source_manifest_payload = json.loads(
        source_manifest_path.read_bytes().decode("utf-8")
    )
    source_manifest_object_sha256 = sha256_canonical_object_hash(
        source_manifest_payload
    )
    if source_manifest_object_sha256 != source_snapshot.sha256():
        raise TrainingSnapshotError(
            "source_manifest_locator does not resolve to the supplied source_snapshot"
        )
    count_evidence_path = resolve_repository_file(
        _GIT_REPO_ROOT,
        _require_exact_repo_locator(
            row_count_evidence_locator, "row_count_evidence_locator"
        ),
    )
    row_count_evidence_sha256 = sha256_hex_bytes(count_evidence_path.read_bytes())
    row_summary = source_rows_from_snapshot_rows(source_snapshot.rows)
    total_rows = sum(summary.row_count for summary in row_summary)
    return TrainingSnapshot(
        schema_version="2.0.0",
        model_version=model_version,
        adapter_version=adapter_version,
        code_version=code_version,
        as_of=to_rfc3339(as_of_dt),
        train_cutoff=train_cutoff,
        source_manifest_locator=source_manifest_locator,
        source_manifest_object_sha256=source_manifest_object_sha256,
        source_snapshot_pairs=((source_snapshot.snapshot_id, source_snapshot.sha256()),),
        source_tree_sha256=source_tree_sha256,
        source_tree_allowlist=source_tree_allowlist,
        row_count_evidence_locator=row_count_evidence_locator,
        row_count_evidence_sha256=row_count_evidence_sha256,
        row_count_by_year=dict(row_count_by_year),
        row_count_by_league=dict(row_count_by_league),
        row_count_by_tier=dict(row_count_by_tier),
        row_count_by_patch=dict(row_count_by_patch),
        row_count_by_source=dict(row_count_by_source),
        source_rows=row_summary,
        created_at=to_rfc3339(as_of_dt),
        taxonomy_version=taxonomy_version,
        crosswalk_version=crosswalk_version,
        inclusion_filters=inclusion_filters,
        exclusion_filters=exclusion_filters,
        min_event_at=min_event_at,
        max_event_at=max_event_at,
        min_available_at=min_available_at,
        max_available_at=max_available_at,
        split_assignment_ids=split_assignment_ids,
        split_assignment_locators=split_assignment_locators,
        split_assignment_sha256s=split_assignment_sha256s,
        environment_lock_sha256=environment_lock_sha256,
        environment_lock_locator=environment_lock_locator,
        code_commit=code_commit,
        candidate_code_commit=candidate_code_commit,
        require_no_stale_required=require_no_stale_required,
        status=status,
        row_count=total_rows,
        correction_lines=correction_lines,
        required_source_snapshot_pairs=required_source_snapshot_pairs,
        optional_source_snapshot_pairs=optional_source_snapshot_pairs,
    )


def write_source_snapshot(snapshot: SourceSnapshot, path: Path) -> ArtifactWriteResult:
    """Persist a source snapshot and return digest pairs."""

    return snapshot.write(path)


def write_training_snapshot(snapshot: TrainingSnapshot, path: Path) -> ArtifactWriteResult:
    """Persist a training snapshot and return digest pairs."""

    return snapshot.write(path)


def write_lineage_report(report: LineageReport, path: Path) -> ArtifactWriteResult:
    """Persist a lineage report and return digest pairs."""

    return report.write(path)


__all__ = [
    "CONTRACT_TREE_SHA256",
    "ChampionPatchRoleAppearanceRow",
    "SourceSnapshot",
    "SourceSnapshotManifest",
    "SourceSnapshotRow",
    "SourceSnapshotRowSummary",
    "SourceSnapshotSnapshotError",
    "SourceTreeMismatchError",
    "TrainingSnapshot",
    "TrainingSnapshotError",
    "LineageReport",
    "ArtifactWriteResult",
    "verify_source_tree",
    "make_default_freshness_report",
    "make_default_source_snapshot_rows",
    "write_source_snapshot",
    "write_training_snapshot",
    "write_lineage_report",
    "source_rows_from_snapshot_rows",
    "leaf_source_row_evidence",
    "source_snapshot_row_lookup",
    "make_training_snapshot",
    "SourceSnapshotError",
    "SourceSnapshotMismatch",
    "verify_source_tree",
]


def verify_source_tree(
    manifest: SourceSnapshot | TrainingSnapshot,
    allowlist: Iterable[str],
    root: Path,
) -> None:
    """Assert that a manifest hash matches the computed source-tree digest."""

    actual = canonical_source_tree_sha256(root, tuple(allowlist))
    if actual != manifest.source_tree_sha256:
        raise SourceTreeMismatchError(
            f"source tree mismatch for {manifest.snapshot_id}: expected {manifest.source_tree_sha256}, got {actual}"
        )
