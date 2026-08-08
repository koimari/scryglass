"""Code-pinned, development-only live total-kills candidate registration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from lol_kills.live_totals_model import validate_source_snapshot_manifest


ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CANDIDATE_LOCATOR = Path(
    "data/lol/models/live_totals_model_v2_20260801T230204Z.json"
)
DEVELOPMENT_CANDIDATE_RAW_SHA256 = (
    "1a44d048dc8c24ececbd2a9157f542ce2b7eaaa8530fd6f3221d6888bc19a6c4"
)
SOURCE_SNAPSHOT_LOCATOR = Path(
    "data/lol/warehouse/snapshots/live_totals/"
    "04d4d7016bc1639fecddd613c1af6de94c6222a9b77cc2daaebbc51f8223402f/"
    "maps.parquet"
)
SOURCE_SNAPSHOT_RAW_SHA256 = (
    "04d4d7016bc1639fecddd613c1af6de94c6222a9b77cc2daaebbc51f8223402f"
)
SOURCE_MANIFEST_LOCATOR = SOURCE_SNAPSHOT_LOCATOR.with_name("snapshot-manifest.json")
SOURCE_MANIFEST_RAW_SHA256 = (
    "7dcaddc7fd9ffdd6c8c83bc52b528063129fbd1b80bfbc456786006d1e64b96c"
)
SOURCE_MANIFEST_CANONICAL_SHA256 = (
    "6b4f833f8c23d018c3d1d526999cbc89b1571ae331ffcd19c44497bbade3d613"
)


class LiveTotalsCandidateError(RuntimeError):
    """The registered development candidate or its replay source is invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def development_candidate_path(root: Path = ROOT) -> Path:
    return root / DEVELOPMENT_CANDIDATE_LOCATOR


def validate_development_candidate(root: Path = ROOT) -> dict[str, Any]:
    artifact_path = development_candidate_path(root)
    source_path = root / SOURCE_SNAPSHOT_LOCATOR
    source_manifest_path = root / SOURCE_MANIFEST_LOCATOR
    expected = (
        (artifact_path, DEVELOPMENT_CANDIDATE_RAW_SHA256, "candidate"),
        (source_path, SOURCE_SNAPSHOT_RAW_SHA256, "source snapshot"),
        (source_manifest_path, SOURCE_MANIFEST_RAW_SHA256, "source manifest"),
    )
    for path, digest, label in expected:
        if not path.is_file() or _sha256(path) != digest:
            raise LiveTotalsCandidateError(f"registered {label} bytes do not match")
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveTotalsCandidateError("registered candidate could not be parsed") from exc
    if not isinstance(artifact, dict):
        raise LiveTotalsCandidateError("registered candidate root must be an object")
    if artifact.get("schema_version") != "scryglass.live-total-kills.v2":
        raise LiveTotalsCandidateError("registered candidate schema is not supported")
    try:
        source_manifest = validate_source_snapshot_manifest(
            source_path, source_manifest_path
        )
    except ValueError as exc:
        raise LiveTotalsCandidateError(str(exc)) from exc
    if (
        source_manifest.get("snapshot_canonical_sha256")
        != SOURCE_MANIFEST_CANONICAL_SHA256
    ):
        raise LiveTotalsCandidateError("source manifest canonical pin does not match")
    source = (artifact.get("meta") or {}).get("source") or {}
    if source != {
        "path": SOURCE_SNAPSHOT_LOCATOR.as_posix(),
        "bytes": source_path.stat().st_size,
        "sha256": SOURCE_SNAPSHOT_RAW_SHA256,
        "snapshot_manifest": {
            "path": SOURCE_MANIFEST_LOCATOR.as_posix(),
            "bytes": source_manifest_path.stat().st_size,
            "sha256": SOURCE_MANIFEST_RAW_SHA256,
        },
    }:
        raise LiveTotalsCandidateError("candidate source binding does not match registry")
    authority = artifact.get("authority") or {}
    if (
        authority.get("content_addressing_confers_authority") is not False
        or authority.get("betting_decision_authorized") is not False
        or (authority.get("dependence_interval") or {}).get("status")
        != "development_only"
    ):
        raise LiveTotalsCandidateError("candidate exceeds its development authority")
    return artifact


__all__ = [
    "DEVELOPMENT_CANDIDATE_LOCATOR",
    "DEVELOPMENT_CANDIDATE_RAW_SHA256",
    "LiveTotalsCandidateError",
    "SOURCE_MANIFEST_LOCATOR",
    "SOURCE_MANIFEST_RAW_SHA256",
    "SOURCE_SNAPSHOT_LOCATOR",
    "SOURCE_SNAPSHOT_RAW_SHA256",
    "development_candidate_path",
    "validate_development_candidate",
]
