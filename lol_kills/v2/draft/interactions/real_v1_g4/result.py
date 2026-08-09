"""Isolated aggregate-only result schema for the corrected 52-slot runner."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping


SCHEMA = "scryglass:real-v1-g4-repair-execution-output:v1"


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def validate_result(value: Mapping[str, Any]) -> None:
    unsigned = dict(value)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != sha256(unsigned) or value.get("schema_version") != SCHEMA:
        raise ValueError("isolated result identity/schema mismatch")
    if value.get("final_holdout_loaded") is not False or value.get("claim_ceiling", {}).get("prediction") is not False:
        raise ValueError("isolated result claim boundary mismatch")
    ledger = value.get("ledger")
    if not isinstance(ledger, list) or len(ledger) != 52 or [row.get("sequence") for row in ledger] != list(range(1, 53)):
        raise ValueError("isolated result ledger mismatch")
    def metric_tree(report: Any) -> None:
        if report is None: return
        if not isinstance(report, Mapping): raise ValueError("isolated result metric invalid")
        if {"log_loss", "brier"} <= set(report):
            if any(not math.isfinite(float(report[key])) for key in ("log_loss", "brier")): raise ValueError("isolated result metric invalid")
            return
        for child in report.values(): metric_tree(child)
    metric_tree(value.get("metrics", {}))


def _safe_parent_chain(path: Path, root: Path) -> None:
    absolute = path.absolute()
    current, parts = Path(absolute.anchor), absolute.parts[1:-1]
    for anchor in (root, Path(tempfile.gettempdir())):
        try:
            relative = absolute.relative_to(anchor.absolute())
        except ValueError:
            continue
        current, parts = anchor.resolve(), relative.parts[:-1]
        break
    for part in parts:
        current = current / part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("isolated result unsafe parent")


def write_result(value: Mapping[str, Any], *, path: Path, root: Path) -> str:
    validate_result(value)
    _safe_parent_chain(path, root)
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None and (stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1):
        raise ValueError("isolated result unsafe leaf")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_bytes(value) + b"\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
    return str(value["artifact_sha256"])
