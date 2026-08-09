"""Path constants for v2 data foundations."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

V2_DATA_ROOT = REPO_ROOT / "data" / "lol" / "v2"

SNAPSHOT_ROOT = V2_DATA_ROOT / "snapshots"

DEFAULT_SOURCE_SNAPSHOT_PATH = SNAPSHOT_ROOT / "source_snapshot_manifest.json"
DEFAULT_TRAINING_SNAPSHOT_PATH = SNAPSHOT_ROOT / "training_snapshot_manifest.json"
DEFAULT_PUBLICATION_MATRIX_PATH = SNAPSHOT_ROOT / "publication_matrix.json"
DEFAULT_PUBLIC_ALLOWLIST_PATH = SNAPSHOT_ROOT / "artifact_allowlist.json"
DEFAULT_LINEAGE_REPORT_PATH = SNAPSHOT_ROOT / "lineage_report.json"

