"""Locate, load, and hash the League Combat Calculator (LCC) atom sources.

The LCC repo is an external project (default ~/Projects/league-combat-calculator,
overridable via the SCRYGLASS_LCC_REPO environment variable).  This module only
reads LCC data; it never writes there.  Every consumed file is pinned by
SHA-256 so the bridge artifact can fail closed when LCC data drifts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .schema import AtomBridgeError, require_object

DEFAULT_LCC_REPO = Path(os.environ.get("SCRYGLASS_LCC_REPO", "/Users/river/Projects/league-combat-calculator"))

# Tracked LCC data files consumed by the bridge (relative to the LCC repo).
REQUIRED_DATA_FILES: tuple[str, ...] = (
    "data/atoms/atom-summary.json",
    "data/atoms/classification-report.json",
    "data/wiki-atoms/atom-relations.json",
    "data/champions.json",
)
# Canonical data-patch marker: the wiki cache is the authority (26.15-era);
# the ddragon 16.15.1 label in static/js/app.js is a stale CDN artifact.
# Coordination answer from the LCC thread (2026-08-07).
PATCH_MARKER_FILE = "data/.champions.json.meta"
OPTIONAL_DATA_FILES: tuple[str, ...] = (
    "data/atoms/items.json",
    "data/wiki-atoms/champion-spell-atoms.json",
    "data/wiki-atoms/champion-passive-atoms.json",
)
CHAMPION_ATOM_GLOB = "data/atoms/*.atoms.json"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise AtomBridgeError(f"cannot read {label} at {path}: {exc}") from exc
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise AtomBridgeError(f"invalid JSON in {label} at {path}: {exc}") from exc
    return payload


class LccSources:
    """Loaded + hashed LCC atom sources for one bridge build."""

    def __init__(self, repo: Path = DEFAULT_LCC_REPO) -> None:
        self.repo = repo.resolve()
        if not (self.repo / ".git").exists() and not (self.repo / "data").exists():
            raise AtomBridgeError(f"LCC repo not found at {self.repo} (set SCRYGLASS_LCC_REPO)")
        self.commit: str | None = None
        self.files: dict[str, str] = {}  # relative path -> sha256
        self.payloads: dict[str, Any] = {}
        self.champion_atom_files: dict[str, list[dict[str, Any]]] = {}

    def load(self) -> None:
        self.commit = self._git_head()
        if (self.repo / PATCH_MARKER_FILE).exists():
            raw = (self.repo / PATCH_MARKER_FILE).read_bytes()
            self.files[PATCH_MARKER_FILE] = sha256_bytes(raw)
            self.payloads[PATCH_MARKER_FILE] = json.loads(raw.decode("utf-8"))
        for rel in REQUIRED_DATA_FILES:
            path = self.repo / rel
            raw = path.read_bytes()
            self.files[rel] = sha256_bytes(raw)
            self.payloads[rel] = json.loads(raw.decode("utf-8"))
        for rel in OPTIONAL_DATA_FILES:
            path = self.repo / rel
            if path.exists():
                raw = path.read_bytes()
                self.files[rel] = sha256_bytes(raw)
                self.payloads[rel] = json.loads(raw.decode("utf-8"))
        for path in sorted(self.repo.glob(CHAMPION_ATOM_GLOB)):
            rel = str(path.relative_to(self.repo))
            raw = path.read_bytes()
            self.files[rel] = sha256_bytes(raw)
            payload = json.loads(raw.decode("utf-8"))
            if not isinstance(payload, list):
                raise AtomBridgeError(f"{rel} must be a JSON list of atoms")
            self.champion_atom_files[path.stem.split(".")[0]] = payload

    def _git_head(self) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(self.repo), "rev-parse", "HEAD"],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
            out = completed.stdout.strip()
            return out if completed.returncode == 0 and out else None
        except (OSError, subprocess.SubprocessError):
            return None

    def source_provenance(self) -> dict[str, Any]:
        data_patch: str | None = None
        meta = self.payloads.get(PATCH_MARKER_FILE)
        if isinstance(meta, dict):
            data_patch = "26.15" if str(meta.get("fetched_at", "")).startswith("1786") else None
        return {
            "lcc_repo": str(self.repo),
            "lcc_commit": self.commit,
            "data_patch": data_patch or "unknown",
            "data_patch_note": (
                "wiki cache is the authority (LCC thread, 2026-08-07); "
                "ddragon 16.15.1 label is a stale CDN artifact"
            ),
            "file_sha256": dict(sorted(self.files.items())),
        }
