"""Patch-pinned source authority for the mechanics-first League engine.

This module deliberately separates source evidence from executable mechanics.
Wiki pages can establish that a rule or change exists, but a semantic-only
cell is not executable.  Exact and reviewed cells must point to immutable
source bindings; blocked cells carry no fallback value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from lol_kills.v2.data.common import parse_rfc3339


SCHEMA_VERSION = "scryglass:patch-authority:v1"
PACKET_MATRIX_SCHEMA_VERSION = "scryglass:patch-authority-matrix:v1"
CELL_STATUSES = frozenset({"exact", "semantic_only", "blocked", "reconciled"})
EXECUTABLE_STATUSES = frozenset({"exact", "reconciled"})
PATCH_RE = re.compile(r"^26\.(?:\d{1,2}|S\d+\.\d+)$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PATCH_TOKEN_RE = re.compile(r"\bV(26\.(?:\d{1,2}|S\d+\.\d+))\b")


class PatchAuthorityError(ValueError):
    """Raised when a patch packet violates the source contract."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_object(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PatchAuthorityError(f"{field_name} must be a non-empty string")
    return value.strip()


def _require_sha(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PatchAuthorityError(f"{field_name} must be a lowercase SHA-256")
    return value


def _require_timestamp(value: Any, field_name: str) -> str:
    text = _require_text(value, field_name)
    try:
        parse_rfc3339(text)
    except Exception as exc:  # keep the public error type stable
        raise PatchAuthorityError(f"{field_name} must be RFC-3339 UTC") from exc
    return text


def _patch_sort_key(patch: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"26\.(\d{1,2})", patch)
    if match:
        return (0, int(match.group(1)), 0)
    match = re.fullmatch(r"26\.S(\d+)\.(\d+)", patch)
    if match:
        return (1, int(match.group(1)), int(match.group(2)))
    return (2, 999, 999)


@dataclass(frozen=True)
class SourceBinding:
    """An exact source payload or revision used by a packet cell."""

    source_id: str
    source_kind: str
    source_url: str
    retrieved_at: str
    payload_sha256: str
    revision_id: str | None = None
    source_updated_at: str | None = None
    content_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SourceBinding":
        if not isinstance(value, Mapping):
            raise PatchAuthorityError("source binding must be an object")
        source_updated = value.get("source_updated_at")
        content_sha = value.get("content_sha256")
        return cls(
            source_id=_require_text(value.get("source_id"), "source_id"),
            source_kind=_require_text(value.get("source_kind"), "source_kind"),
            source_url=_require_text(value.get("source_url"), "source_url"),
            retrieved_at=_require_timestamp(value.get("retrieved_at"), "retrieved_at"),
            payload_sha256=_require_sha(value.get("payload_sha256"), "payload_sha256"),
            revision_id=(str(value["revision_id"]) if value.get("revision_id") is not None else None),
            source_updated_at=(
                _require_timestamp(source_updated, "source_updated_at")
                if source_updated is not None
                else None
            ),
            content_sha256=(
                _require_sha(content_sha, "content_sha256")
                if content_sha is not None
                else None
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_kind": self.source_kind,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "payload_sha256": self.payload_sha256,
            "revision_id": self.revision_id,
            "source_updated_at": self.source_updated_at,
            "content_sha256": self.content_sha256,
        }


@dataclass(frozen=True)
class MechanicCell:
    """One patch-scoped mechanic claim and its executable status."""

    key: str
    domain: str
    status: str
    source_ids: tuple[str, ...] = ()
    value: Any = None
    formula: Mapping[str, Any] | None = None
    reason: str | None = None
    evidence: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MechanicCell":
        if not isinstance(value, Mapping):
            raise PatchAuthorityError("mechanic cell must be an object")
        raw_source_ids = value.get("source_ids", [])
        raw_evidence = value.get("evidence", [])
        if not isinstance(raw_source_ids, list) or any(
            not isinstance(item, str) or not item.strip() for item in raw_source_ids
        ):
            raise PatchAuthorityError("mechanic cell source_ids must be non-empty strings")
        if not isinstance(raw_evidence, list) or any(not isinstance(item, str) for item in raw_evidence):
            raise PatchAuthorityError("mechanic cell evidence must be strings")
        status = _require_text(value.get("status"), "cell.status")
        if status not in CELL_STATUSES:
            raise PatchAuthorityError(f"unsupported mechanic cell status: {status}")
        formula = value.get("formula")
        if formula is not None and not isinstance(formula, Mapping):
            raise PatchAuthorityError("cell.formula must be an object when present")
        return cls(
            key=_require_text(value.get("key"), "cell.key"),
            domain=_require_text(value.get("domain"), "cell.domain"),
            status=status,
            source_ids=tuple(sorted(set(raw_source_ids))),
            value=value.get("value"),
            formula=dict(formula) if formula is not None else None,
            reason=(str(value["reason"]) if value.get("reason") is not None else None),
            evidence=tuple(str(item) for item in raw_evidence),
        )

    @property
    def executable(self) -> bool:
        return self.status in EXECUTABLE_STATUSES

    def to_mapping(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "domain": self.domain,
            "status": self.status,
            "source_ids": list(self.source_ids),
            "value": self.value,
            "formula": dict(self.formula) if self.formula is not None else None,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ReconciliationRecord:
    """A conflict or review decision that cannot be averaged away."""

    cell_key: str
    status: str
    source_ids: tuple[str, ...]
    reason: str
    resolution: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReconciliationRecord":
        if not isinstance(value, Mapping):
            raise PatchAuthorityError("reconciliation record must be an object")
        status = _require_text(value.get("status"), "reconciliation.status")
        if status not in {"blocked", "reviewed"}:
            raise PatchAuthorityError("reconciliation status must be blocked or reviewed")
        ids = value.get("source_ids", [])
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            raise PatchAuthorityError("reconciliation source_ids must be a string list")
        return cls(
            cell_key=_require_text(value.get("cell_key"), "reconciliation.cell_key"),
            status=status,
            source_ids=tuple(sorted(set(ids))),
            reason=_require_text(value.get("reason"), "reconciliation.reason"),
            resolution=(str(value["resolution"]) if value.get("resolution") is not None else None),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "cell_key": self.cell_key,
            "status": self.status,
            "source_ids": list(self.source_ids),
            "reason": self.reason,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class PatchPacket:
    """Immutable, hash-addressed packet for one patch."""

    patch: str
    sources: tuple[SourceBinding, ...]
    cells: tuple[MechanicCell, ...]
    reconciliations: tuple[ReconciliationRecord, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not PATCH_RE.fullmatch(self.patch):
            raise PatchAuthorityError(f"unsupported 2026 patch identifier: {self.patch}")
        source_ids = [source.source_id for source in self.sources]
        if len(set(source_ids)) != len(source_ids):
            raise PatchAuthorityError("patch packet source IDs must be unique")
        cell_keys = [cell.key for cell in self.cells]
        if len(set(cell_keys)) != len(cell_keys):
            raise PatchAuthorityError("patch packet cell keys must be unique")
        source_set = set(source_ids)
        review_by_key = {row.cell_key: row for row in self.reconciliations}
        for cell in self.cells:
            if not set(cell.source_ids).issubset(source_set):
                raise PatchAuthorityError(f"cell {cell.key} references an unknown source")
            if cell.status in {"exact", "reconciled"} and cell.value is None and cell.formula is None:
                raise PatchAuthorityError(f"executable cell {cell.key} has no value or formula")
            if cell.status == "blocked" and (cell.value is not None or cell.formula is not None):
                raise PatchAuthorityError(f"blocked cell {cell.key} cannot carry a fallback value")
            if cell.status == "reconciled":
                review = review_by_key.get(cell.key)
                if review is None or review.status != "reviewed":
                    raise PatchAuthorityError(f"reconciled cell {cell.key} lacks a reviewed record")
        for record in self.reconciliations:
            if not set(record.source_ids).issubset(source_set):
                raise PatchAuthorityError(f"reconciliation {record.cell_key} references an unknown source")

    @property
    def executable_cells(self) -> tuple[MechanicCell, ...]:
        return tuple(cell for cell in self.cells if cell.executable)

    @property
    def blocked_cells(self) -> tuple[MechanicCell, ...]:
        return tuple(cell for cell in self.cells if cell.status == "blocked")

    @property
    def semantic_only_cells(self) -> tuple[MechanicCell, ...]:
        return tuple(cell for cell in self.cells if cell.status == "semantic_only")

    def cell(self, key: str) -> MechanicCell | None:
        return next((cell for cell in self.cells if cell.key == key), None)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "patch": self.patch,
            "sources": [source.to_mapping() for source in self.sources],
            "cells": [cell.to_mapping() for cell in self.cells],
            "reconciliations": [row.to_mapping() for row in self.reconciliations],
            "metadata": dict(self.metadata),
            "claim_ceiling": {
                "mechanics_execution": bool(self.executable_cells),
                "full_game_emulation": False,
                "prediction": False,
                "publication": False,
                "promotion": False,
            },
        }

    @property
    def payload_sha256(self) -> str:
        return _sha256_object(self.to_mapping())

    def write(self, path: Path) -> str:
        payload = self.to_mapping()
        payload["packet_sha256"] = self.payload_sha256
        payload_bytes = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return self.payload_sha256

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PatchPacket":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise PatchAuthorityError("invalid patch packet schema")
        packet_sha = value.get("packet_sha256")
        payload = dict(value)
        payload.pop("packet_sha256", None)
        if packet_sha is not None and packet_sha != _sha256_object(payload):
            raise PatchAuthorityError("patch packet hash mismatch")
        sources = tuple(SourceBinding.from_mapping(item) for item in value.get("sources", []))
        cells = tuple(MechanicCell.from_mapping(item) for item in value.get("cells", []))
        reconciliations = tuple(
            ReconciliationRecord.from_mapping(item)
            for item in value.get("reconciliations", [])
        )
        metadata = value.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise PatchAuthorityError("patch packet metadata must be an object")
        return cls(
            patch=_require_text(value.get("patch"), "patch"),
            sources=sources,
            cells=cells,
            reconciliations=reconciliations,
            metadata=dict(metadata),
        )


def _load_latest(vault: Path) -> list[dict[str, Any]]:
    path = vault / "latest.jsonl"
    if not path.exists():
        raise PatchAuthorityError(f"Wiki vault latest checkpoint is missing: {path}")
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PatchAuthorityError("Wiki latest checkpoint contains invalid JSON") from exc
        if not isinstance(row, Mapping):
            raise PatchAuthorityError("Wiki latest checkpoint row is not an object")
        key = (int(row.get("namespace", -1)), str(row.get("title", "")))
        latest[key] = dict(row)
    return sorted(latest.values(), key=lambda row: (int(row["namespace"]), str(row["title"])))


def discover_2026_patches(vault: Path) -> tuple[str, ...]:
    """Discover patch labels from captured LoL patch-history pages only."""

    patches: set[str] = set()
    for row in _load_latest(vault):
        if int(row.get("namespace", -1)) != 0:
            continue
        title = str(row.get("title", ""))
        if "patch history" not in title.casefold():
            continue
        document_path = row.get("document_path")
        if not isinstance(document_path, str):
            continue
        path = vault / document_path
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        patches.update(match.group(1) for match in PATCH_TOKEN_RE.finditer(content))
    return tuple(sorted(patches, key=_patch_sort_key))


def _wiki_source(row: Mapping[str, Any]) -> SourceBinding:
    revision_id = row.get("revision_id")
    return SourceBinding(
        source_id=f"wiki:ns{int(row['namespace'])}:page{int(row['page_id'])}:rev{revision_id}",
        source_kind="league_wiki_revision",
        source_url=_require_text(row.get("source_url"), "source_url"),
        retrieved_at=_require_timestamp(row.get("retrieved_at"), "retrieved_at"),
        payload_sha256=_require_sha(row.get("document_sha256"), "document_sha256"),
        revision_id=str(revision_id) if revision_id is not None else None,
        source_updated_at=(
            _require_timestamp(row["revision_timestamp"], "revision_timestamp")
            if row.get("revision_timestamp") is not None
            else None
        ),
        content_sha256=_require_sha(row.get("content_sha256"), "content_sha256"),
    )


def build_wiki_patch_packet(vault: Path, patch: str) -> PatchPacket:
    """Create a source-indexed packet from captured Wiki patch histories.

    The extractor intentionally emits semantic evidence plus blocked domain
    cells.  It does not pretend that prose or a patch-history sentence is an
    executable numeric formula.  A later client-data reconciler can replace a
    blocked cell with an exact value while preserving the Wiki source rows.
    """

    if not PATCH_RE.fullmatch(patch):
        raise PatchAuthorityError(f"unsupported 2026 patch identifier: {patch}")
    rows = _load_latest(vault)
    sources: list[SourceBinding] = []
    cells: list[MechanicCell] = []
    seen_sources: set[str] = set()
    evidence_pages = 0
    for row in rows:
        if int(row.get("namespace", -1)) != 0:
            continue
        title = str(row.get("title", ""))
        if "patch history" not in title.casefold():
            continue
        document_path = row.get("document_path")
        if not isinstance(document_path, str):
            continue
        path = vault / document_path
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        if not re.search(rf"\bV{re.escape(patch)}\b", content):
            continue
        source = _wiki_source(row)
        if source.source_id not in seen_sources:
            sources.append(source)
            seen_sources.add(source.source_id)
        evidence_pages += 1
        cells.append(
            MechanicCell(
                key=f"wiki.patch-history:{title}:{patch}",
                domain="wiki.patch_history",
                status="semantic_only",
                source_ids=(source.source_id,),
                value={"page_title": title, "patch": patch},
                evidence=("captured_patch_history_page",),
            )
        )

    blocked_reason = (
        "no patch-pinned executable client formula has been reconciled; "
        "Wiki semantics remain review evidence only"
    )
    for domain in ("champions", "items", "runes", "game_systems"):
        cells.append(
            MechanicCell(
                key=f"domain:{domain}",
                domain=domain,
                status="blocked",
                reason=blocked_reason,
            )
        )
    if not sources:
        cells.append(
            MechanicCell(
                key="packet:source_coverage",
                domain="packet",
                status="blocked",
                reason=f"no captured Wiki patch-history page mentions V{patch}",
            )
        )
    return PatchPacket(
        patch=patch,
        sources=tuple(sorted(sources, key=lambda item: item.source_id)),
        cells=tuple(sorted(cells, key=lambda item: item.key)),
        metadata={
            "reconstruction_method": "league_wiki_latest_patch_history",
            "wiki_patch_history_page_count": evidence_pages,
            "exact_client_source_present": False,
            "semantic_cells_are_non_executable": True,
        },
    )


def build_2026_wiki_packets(vault: Path, output_root: Path) -> dict[str, Any]:
    """Materialize one explicit packet for every discovered 2026 patch."""

    patches = discover_2026_patches(vault)
    if not patches:
        raise PatchAuthorityError("no 2026 patches were discoverable from the Wiki vault")
    rows: list[dict[str, Any]] = []
    for patch in patches:
        packet = build_wiki_patch_packet(vault, patch)
        path = output_root / "2026" / patch / "packet.json"
        packet_sha = packet.write(path)
        rows.append(
            {
                "patch": patch,
                "packet_path": str(path),
                "packet_sha256": packet_sha,
                "source_count": len(packet.sources),
                "cell_count": len(packet.cells),
                "executable_cell_count": len(packet.executable_cells),
                "semantic_only_cell_count": len(packet.semantic_only_cells),
                "blocked_cell_count": len(packet.blocked_cells),
                "wiki_patch_history_page_count": packet.metadata.get("wiki_patch_history_page_count", 0),
            }
        )
    manifest = {
        "schema_version": PACKET_MATRIX_SCHEMA_VERSION,
        "patch_year": 2026,
        "source_vault": str(vault),
        "patches": rows,
        "claim_ceiling": {
            "exact_mechanics": False,
            "full_game_emulation": False,
            "prediction": False,
            "publication": False,
        },
    }
    manifest["manifest_sha256"] = _sha256_object(manifest)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "2026" / "matrix-manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    return _sha256_object(unsigned)


def build_cdragon_patch_packet(
    index_path: Path,
    *,
    expected_patch: str,
    manifest_path: Path | None = None,
) -> PatchPacket:
    """Convert an exact-patch CommunityDragon index into a partial packet.

    The bridge intentionally exposes only the values that the capture step
    extracted as numeric client base values.  Spell formula graphs and raw
    item payloads remain semantic-only until their execution semantics have
    been implemented and micro-tested.  Most importantly, the index patch
    must equal ``expected_patch`` byte-for-byte; an older client packet cannot
    satisfy a newer 2026 packet.
    """

    if not PATCH_RE.fullmatch(expected_patch):
        raise PatchAuthorityError(f"unsupported 2026 patch identifier: {expected_patch}")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchAuthorityError(f"cannot read CommunityDragon mechanics index: {index_path}") from exc
    if not isinstance(index, Mapping) or index.get("schema_version") != "scryglass:cdragon-patch-packet:v1":
        raise PatchAuthorityError("invalid CommunityDragon mechanics index schema")
    if index.get("patch") != expected_patch:
        raise PatchAuthorityError(
            f"CommunityDragon patch mismatch: payload={index.get('patch')!r}, expected={expected_patch!r}"
        )
    retrieved_at = _require_timestamp(index.get("retrieved_at"), "index.retrieved_at")
    root_url = _require_text(index.get("source_root"), "index.source_root").rstrip("/") + "/"
    if manifest_path is None:
        manifest_path = index_path.parent / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchAuthorityError(f"cannot read CommunityDragon manifest: {manifest_path}") from exc
    if not isinstance(manifest, Mapping):
        raise PatchAuthorityError("CommunityDragon manifest must be an object")
    if manifest.get("patch") != expected_patch or manifest.get("exact_patch_source") is not True:
        raise PatchAuthorityError("CommunityDragon manifest is not an exact source for the requested patch")
    if manifest.get("manifest_sha256") != _manifest_payload_sha256(manifest):
        raise PatchAuthorityError("CommunityDragon manifest hash mismatch")
    raw_files = manifest.get("files", [])
    if not isinstance(raw_files, list):
        raise PatchAuthorityError("CommunityDragon manifest files must be a list")
    file_rows = {
        str(row.get("path")): row
        for row in raw_files
        if isinstance(row, Mapping) and isinstance(row.get("path"), str)
    }

    index_source_id = f"cdragon:{expected_patch}:mechanics-index"
    manifest_source_id = f"cdragon:{expected_patch}:manifest"
    sources: dict[str, SourceBinding] = {
        index_source_id: SourceBinding(
            source_id=index_source_id,
            source_kind="communitydragon_mechanics_index",
            source_url=root_url + "mechanics-index.json",
            retrieved_at=retrieved_at,
            payload_sha256=_sha256(index_path.read_bytes()),
        ),
        manifest_source_id: SourceBinding(
            source_id=manifest_source_id,
            source_kind="communitydragon_manifest",
            source_url=root_url + "manifest.json",
            retrieved_at=retrieved_at,
            payload_sha256=_sha256(manifest_path.read_bytes()),
        ),
    }

    def file_source(relative_path: str) -> str | None:
        row = file_rows.get(relative_path)
        local_path = index_path.parent / relative_path
        if not isinstance(row, Mapping) or not local_path.exists():
            return None
        payload_sha = row.get("sha256")
        source_url = row.get("url")
        if not isinstance(payload_sha, str) or not SHA256_RE.fullmatch(payload_sha):
            raise PatchAuthorityError(f"invalid CommunityDragon file hash: {relative_path}")
        if _sha256(local_path.read_bytes()) != payload_sha:
            raise PatchAuthorityError(f"CommunityDragon file hash mismatch: {relative_path}")
        if not isinstance(source_url, str) or not source_url.strip():
            raise PatchAuthorityError(f"CommunityDragon file URL missing: {relative_path}")
        source_id = f"cdragon:{expected_patch}:{relative_path}"
        sources[source_id] = SourceBinding(
            source_id=source_id,
            source_kind="communitydragon_raw_file",
            source_url=source_url,
            retrieved_at=retrieved_at,
            payload_sha256=payload_sha,
        )
        return source_id

    cells: list[MechanicCell] = []
    champions = index.get("champions", [])
    if not isinstance(champions, list):
        raise PatchAuthorityError("CommunityDragon index champions must be a list")
    for champion in champions:
        if not isinstance(champion, Mapping):
            continue
        champion_id = champion.get("id")
        champion_name = _require_text(champion.get("name"), "champion.name")
        mechanics = champion.get("mechanics")
        bin_path = champion.get("bin_json_path")
        bin_source = file_source(str(bin_path)) if isinstance(bin_path, str) else None
        if not isinstance(mechanics, Mapping) or bin_source is None:
            cells.append(
                MechanicCell(
                    key=f"champion:{champion_id or champion_name}:mechanics",
                    domain="champions",
                    status="blocked",
                    reason="missing patch-pinned raw champion mechanics payload",
                )
            )
            continue
        stats = mechanics.get("stats", {})
        if isinstance(stats, Mapping):
            for stat_name, stat_value in sorted(stats.items()):
                if isinstance(stat_value, bool) or not isinstance(stat_value, (int, float)) or not math.isfinite(float(stat_value)):
                    continue
                cells.append(
                    MechanicCell(
                        key=f"champion:{champion_id or champion_name}:stat:{stat_name}",
                        domain="champions.stats",
                        status="exact",
                        source_ids=(bin_source, index_source_id),
                        value={"champion_id": champion_id, "champion": champion_name, "stat": stat_name, "base_value": float(stat_value)},
                        formula={"kind": "client_base_value", "level_scaling": "not_included"},
                        evidence=("CharacterRecords/Root",),
                    )
                )
        spells = mechanics.get("spells", [])
        if isinstance(spells, list):
            for spell in spells:
                if not isinstance(spell, Mapping):
                    continue
                spell_path = _require_text(spell.get("path"), "spell.path")
                cells.append(
                    MechanicCell(
                        key=f"champion:{champion_id or champion_name}:spell:{spell_path}",
                        domain="champions.abilities",
                        status="semantic_only",
                        source_ids=(bin_source, index_source_id),
                        value={"champion_id": champion_id, "champion": champion_name, "spell_path": spell_path},
                        reason="raw client formula graph is preserved but execution semantics are not yet complete",
                        evidence=("raw_formula_graph_preserved",),
                    )
                )

    item_source = file_source("raw/items.json")
    if item_source is not None:
        cells.append(
            MechanicCell(
                key="domain:items:raw",
                domain="items",
                status="semantic_only",
                source_ids=(item_source,),
                value={"payload": "raw_items_json"},
                reason="item normalization and executable modifier formulas are not yet complete",
            )
        )
    else:
        cells.append(MechanicCell(key="domain:items", domain="items", status="blocked", reason="patch-pinned items payload is missing"))
    cells.extend(
        MechanicCell(key=f"domain:{domain}", domain=domain, status="blocked", reason="patch-pinned executable rules are not yet reconciled")
        for domain in ("runes", "game_systems")
    )
    return PatchPacket(
        patch=expected_patch,
        sources=tuple(sorted(sources.values(), key=lambda item: item.source_id)),
        cells=tuple(sorted(cells, key=lambda item: item.key)),
        metadata={
            "reconstruction_method": "communitydragon_mechanics_index",
            "exact_patch_source_present": True,
            "execution_status": "champion_base_stats_only",
            "source_manifest_sha256": manifest.get("manifest_sha256"),
        },
    )


def load_patch_packet(path: Path) -> PatchPacket:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PatchAuthorityError(f"cannot read patch packet: {path}") from exc
    return PatchPacket.from_mapping(payload)


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    discover = sub.add_parser("discover")
    discover.add_argument("--vault", type=Path, required=True)
    build = sub.add_parser("build-wiki-2026")
    build.add_argument("--vault", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    cdragon = sub.add_parser("build-cdragon")
    cdragon.add_argument("--index", type=Path, required=True)
    cdragon.add_argument("--manifest", type=Path)
    cdragon.add_argument("--patch", required=True)
    cdragon.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "discover":
        print(json.dumps({"patches": discover_2026_patches(args.vault)}, sort_keys=True))
        return 0
    if args.command == "build-cdragon":
        packet = build_cdragon_patch_packet(args.index, expected_patch=args.patch, manifest_path=args.manifest)
        print(json.dumps({"patch": packet.patch, "output": str(args.output), "packet_sha256": packet.write(args.output), "executable_cell_count": len(packet.executable_cells)}, sort_keys=True))
        return 0
    manifest = build_2026_wiki_packets(args.vault, args.output)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
