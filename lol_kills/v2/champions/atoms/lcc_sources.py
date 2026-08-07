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
            proc = os.popen(f"git -C {self.repo} rev-parse HEAD 2>/dev/null")
            out = proc.read().strip()
            proc.close()
            return out or None
        except Exception:  # noqa: BLE001
            return None

    def source_provenance(self) -> dict[str, Any]:
        return {
            "lcc_repo": str(self.repo),
            "lcc_commit": self.commit,
            "file_sha256": dict(sorted(self.files.items())),
        }
