"""Durable local receipts for one resumable public refresh run."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "scryglass:refresh-run:v1"
STAGES = (
    "acquire",
    "validate_source",
    "ingest",
    "reconcile",
    "derive",
    "validate_artifacts",
    "stage_release",
    "activate_release",
    "invalidate_cache",
    "smoke",
    "complete",
)
RELEASE_REFERENCE_STAGES = {
    "activate_release",
    "invalidate_cache",
    "smoke",
    "complete",
}


def canonical_sha256(value: object) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def worker_commit(root: Path) -> str:
    configured = os.environ.get("SCRYGLASS_WORKER_COMMIT", "").strip().lower()
    if len(configured) == 40 and all(character in "0123456789abcdef" for character in configured):
        return configured
    value = ""
    for candidate in (root, Path(__file__).resolve().parents[1]):
        try:
            value = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=candidate,
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).strip().lower()
        except (OSError, subprocess.SubprocessError):
            value = ""
        if len(value) == 40 and all(character in "0123456789abcdef" for character in value):
            break
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError("worker Git commit cannot be resolved")
    return value


def input_fingerprint(
    *,
    source_file_sha256: str,
    transform_version: str,
    worker_git_commit: str,
) -> str:
    return canonical_sha256(
        {
            "source_file_sha256": source_file_sha256,
            "transform_version": transform_version,
            "worker_commit": worker_git_commit,
        }
    )


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def reusable_stage_receipt(
    path: Path,
    *,
    stage: str,
    fingerprint: str,
    transform_version: str,
    worker_git_commit: str,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    expected = {
        "schema_version": SCHEMA_VERSION,
        "stage": stage,
        "status": "success",
        "input_fingerprint": fingerprint,
        "transform_version": transform_version,
        "worker_commit": worker_git_commit,
    }
    return payload if all(payload.get(key) == value for key, value in expected.items()) else None


def latest_failed_run(runtime_root: Path, fingerprint: str) -> str | None:
    receipt_root = runtime_root / "receipts"
    paths = sorted(
        receipt_root.glob("refresh-*/run.json"),
        key=lambda path: path.stat().st_mtime_ns,
        reverse=True,
    )
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("status") == "error"
            and payload.get("input_fingerprint") == fingerprint
            and isinstance(payload.get("run_id"), str)
        ):
            return str(payload["run_id"])
    return None


@dataclass
class RefreshRunLedger:
    runtime_root: Path
    scheduled_for: datetime
    worker_git_commit: str
    transform_version: str
    source_file_sha256: str
    source_observed_through: str | None
    counts: dict[str, int] = field(default_factory=dict)
    remote_write: Callable[[dict[str, Any]], None] | None = None
    retry_of: str | None = None

    def __post_init__(self) -> None:
        stamp = self.scheduled_for.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"refresh-{stamp}-{uuid.uuid4().hex[:12]}"
        self.fingerprint = input_fingerprint(
            source_file_sha256=self.source_file_sha256,
            transform_version=self.transform_version,
            worker_git_commit=self.worker_git_commit,
        )
        if self.retry_of is None:
            self.retry_of = latest_failed_run(self.runtime_root, self.fingerprint)
        self.started_at = datetime.now(timezone.utc)
        self.stage_started_wall = time.monotonic()
        self.stage_started_cpu = resource.getrusage(resource.RUSAGE_SELF)
        self.stage_durations: dict[str, dict[str, float]] = {}
        self.stage = "validate_source"
        self.status = "checking"
        self.release_id: str | None = None
        self.failure_code: str | None = None
        self.failure_detail: str | None = None
        self.completed_at: str | None = None
        self.path = self.runtime_root / "receipts" / self.run_id / "run.json"
        self._write()

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "scheduled_for": self.scheduled_for.astimezone(timezone.utc).isoformat(),
            "retry_of": self.retry_of,
            "status": self.status,
            "stage": self.stage,
            "input_fingerprint": self.fingerprint,
            "worker_commit": self.worker_git_commit,
            "transform_version": self.transform_version,
            "source_file_sha256": self.source_file_sha256,
            "source_observed_through": self.source_observed_through,
            "release_id": self.release_id,
            "accepted_games": int(self.counts.get("accepted_games", 0)),
            "new_games": int(self.counts.get("new_games", 0)),
            "corrected_games": int(self.counts.get("corrected_games", 0)),
            "unchanged_games": int(self.counts.get("unchanged_games", 0)),
            "quarantined_games": int(self.counts.get("quarantined_games", 0)),
            "stage_durations": self.stage_durations,
            "failure_code": self.failure_code,
            "failure_detail": self.failure_detail,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

    def _write(self) -> None:
        payload = self._payload()
        _atomic_json(self.path, payload)
        if self.remote_write is not None:
            self.remote_write(payload)

    def advance(self, next_stage: str, **updates: Any) -> None:
        if next_stage not in STAGES:
            raise ValueError(f"unknown refresh stage: {next_stage}")
        if updates.get("release_id") and next_stage not in RELEASE_REFERENCE_STAGES:
            raise ValueError("release ID cannot be recorded before the release is staged")
        usage = resource.getrusage(resource.RUSAGE_SELF)
        completed_stage = self.stage
        self.stage_durations[completed_stage] = {
            "wall_seconds": round(time.monotonic() - self.stage_started_wall, 6),
            "cpu_seconds": round(
                (usage.ru_utime + usage.ru_stime)
                - (self.stage_started_cpu.ru_utime + self.stage_started_cpu.ru_stime),
                6,
            ),
        }
        metrics = updates.get("metrics")
        if isinstance(metrics, dict):
            self.stage_durations[completed_stage].update(
                {
                    str(key): value
                    for key, value in metrics.items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                }
            )
        stage_payload = {
            **self._payload(),
            "stage": completed_stage,
            "status": "success",
        }
        _atomic_json(self.path.parent / f"{completed_stage}.json", stage_payload)
        self.stage = next_stage
        self.release_id = updates.get("release_id", self.release_id)
        for key in self.counts:
            if key in updates:
                self.counts[key] = int(updates[key])
        self.stage_started_wall = time.monotonic()
        self.stage_started_cpu = resource.getrusage(resource.RUSAGE_SELF)
        self._write()

    def finish(self, status: str, *, release_id: str | None = None) -> None:
        self.advance("complete", release_id=release_id)
        self.status = status
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self._write()

    def fail(self, error: Exception) -> None:
        self.status = "error"
        self.failure_code = type(error).__name__[:80]
        self.failure_detail = str(error)[:2000]
        self.completed_at = datetime.now(timezone.utc).isoformat()
        self._write()
