"""Superseding code pin for the boolean-normalized ratings source snapshot."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .multileague_source_snapshot import (
    MultiLeagueSourceSnapshotError,
    validate_source_snapshot,
)


ROOT = Path(__file__).resolve().parents[4]
PACKAGE_ID = "90cbaccb8fbc1ac21a7ea433441e65375d7012ca7f92f60d2b3ba1b74a208438"
MANIFEST_LOCATOR = Path(
    f"data/lol/v2/snapshots/multileague-v3/{PACKAGE_ID}/source-snapshot-manifest.json"
)
MANIFEST_RAW_SHA256 = "5e52b14a641c38e8d6a5e82963df1de5b2de0d9794790b077aba04492aecb61f"
MANIFEST_CANONICAL_SHA256 = (
    "011c955af8190e032a8fdf433102d9f3955d2acfd11b363895d67135781713d3"
)


class SourceRegistryV2Error(RuntimeError):
    """The superseding ratings source snapshot no longer replays exactly."""


def validate_registered_source_snapshot_v2(*, root: Path = ROOT) -> dict[str, Any]:
    path = root / MANIFEST_LOCATOR
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SourceRegistryV2Error("registered v2 source manifest is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != MANIFEST_RAW_SHA256:
        raise SourceRegistryV2Error("registered v2 source manifest raw hash drifted")
    try:
        manifest = validate_source_snapshot(path, root=root)
    except (MultiLeagueSourceSnapshotError, OSError, ValueError) as exc:
        raise SourceRegistryV2Error(str(exc)) from exc
    if (
        manifest.get("package_id") != PACKAGE_ID
        or manifest.get("manifest_canonical_sha256") != MANIFEST_CANONICAL_SHA256
    ):
        raise SourceRegistryV2Error("registered v2 source identity drifted")
    return manifest


__all__ = [
    "MANIFEST_CANONICAL_SHA256",
    "MANIFEST_LOCATOR",
    "MANIFEST_RAW_SHA256",
    "PACKAGE_ID",
    "SourceRegistryV2Error",
    "validate_registered_source_snapshot_v2",
]
