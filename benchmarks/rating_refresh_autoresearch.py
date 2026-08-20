#!/usr/bin/env python3
"""Run a fixed, source-bound rating-refresh comparison.

The runner owns the benchmark contract. A refresh adapter owns the production
call. This split keeps a performance experiment from changing the worker.

The runner copies the accepted input and census into a fixture, executes one
baseline and one candidate adapter for the cold and append-only phases, then
compares source bindings and output descriptors. Commands are argv arrays. No
shell is used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FREEZE_SCHEMA = "scryglass:rating-autoresearch-freeze:v1"
OUTPUT_SCHEMA = "scryglass:rating-autoresearch-output:v1"
REPORT_SCHEMA = "scryglass:rating-autoresearch-report:v1"
CENSUS_SCHEMA = "scryglass:accepted-game-census:v1"
DEFAULT_BUDGET_SECONDS = 60.0
DEFAULT_TIMEOUT_SECONDS = 1_800.0
MAX_LOG_BYTES = 256 * 1024
CALL_MARKER = re.compile(r"^\[rating-autoresearch\] call name=([A-Za-z0-9_.:-]+)$", re.MULTILINE)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


class HarnessError(ValueError):
    """Raised when a benchmark fixture or adapter violates its contract."""


def _canonical_source_game_key(value: object) -> str:
    """Apply the source transport normalization used by accepted census data."""

    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.casefold() in {"nan", "nat", "none", "<na>"}:
        return ""
    prefixes = ("oe-api:", "oracle-elixir-api:")
    changed = True
    while changed:
        changed = False
        lowered = text.casefold()
        for prefix in prefixes:
            if lowered.startswith(prefix):
                text = text[len(prefix) :].strip()
                changed = True
                break
    return text


def canonical_game_ids(values: Iterable[object]) -> tuple[str, ...]:
    """Return sorted, unique source IDs."""

    return tuple(sorted({key for value in values if (key := _canonical_source_game_key(value))}))


def source_identity_sha256(values: Iterable[object]) -> str:
    """Return the accepted-census identity digest."""

    game_ids = canonical_game_ids(values)
    return hashlib.sha256(("\n".join(game_ids) + "\n").encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one regular file without loading it into memory."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise HarnessError(f"cannot read file: {path}") from error
    return digest.hexdigest()


def _require_regular_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise HarnessError(f"{label} must be a regular file: {path}")


def _safe_relative_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not value.strip():
        raise HarnessError(f"input path must be relative and stay inside its root: {value!r}")
    return path


def load_census(path: Path) -> dict[str, Any]:
    """Load and validate the exact accepted-game census."""

    _require_regular_file(path, "census")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"census cannot be read: {path}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != CENSUS_SCHEMA:
        raise HarnessError(f"census schema is invalid: {path}")
    raw_ids = payload.get("game_ids")
    if not isinstance(raw_ids, list) or not all(isinstance(value, str) for value in raw_ids):
        raise HarnessError(f"census game IDs are invalid: {path}")
    game_ids = canonical_game_ids(raw_ids)
    if not game_ids:
        raise HarnessError(f"census is empty: {path}")
    expected = {
        "schema_version": CENSUS_SCHEMA,
        "game_count": len(game_ids),
        "source_identity_sha256": source_identity_sha256(game_ids),
        "game_ids": list(game_ids),
    }
    if payload != expected:
        raise HarnessError(f"census binding is invalid: {path}")
    return {
        **expected,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def _file_record(path: Path, relative_path: Path) -> dict[str, Any]:
    _require_regular_file(path, "input")
    try:
        byte_count = path.stat().st_size
    except OSError as error:
        raise HarnessError(f"cannot stat input: {path}") from error
    return {
        "path": relative_path.as_posix(),
        "bytes": int(byte_count),
        "sha256": sha256_file(path),
    }


def _manifest_digest(manifest: Mapping[str, Any]) -> str:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    return _sha256_bytes(_canonical_json_bytes(unsigned))


def freeze_phase(
    *,
    source_root: Path,
    census_path: Path,
    destination_root: Path,
    input_relative_paths: Sequence[str],
    phase: str,
) -> dict[str, Any]:
    """Copy one phase into the benchmark-owned fixture directory."""

    if phase not in {"cold", "append_only"}:
        raise HarnessError(f"unknown fixture phase: {phase}")
    if not source_root.is_dir() or source_root.is_symlink():
        raise HarnessError(f"input root must be a directory: {source_root}")
    if destination_root.exists():
        if destination_root.is_symlink() or not destination_root.is_dir() or any(destination_root.iterdir()):
            raise HarnessError(f"fixture destination must be an empty directory: {destination_root}")
    census = load_census(census_path)
    inputs_root = destination_root / "inputs"
    inputs_root.mkdir(parents=True, exist_ok=True)
    file_records: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for raw_relative_path in input_relative_paths:
        relative_path = _safe_relative_path(raw_relative_path)
        key = relative_path.as_posix()
        if key in seen_paths:
            raise HarnessError(f"duplicate input path: {key}")
        seen_paths.add(key)
        source_path = source_root / relative_path
        destination_path = inputs_root / relative_path
        _require_regular_file(source_path, "input")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_path, destination_path)
        except OSError as error:
            raise HarnessError(f"cannot copy input: {source_path}") from error
        record = _file_record(destination_path, Path("inputs") / relative_path)
        source_record = _file_record(source_path, relative_path)
        if record["sha256"] != source_record["sha256"] or record["bytes"] != source_record["bytes"]:
            raise HarnessError(f"copied input changed during freeze: {relative_path}")
        file_records.append(record)

    frozen_census_path = destination_root / "accepted-census.json"
    try:
        shutil.copy2(census_path, frozen_census_path)
    except OSError as error:
        raise HarnessError(f"cannot copy census: {census_path}") from error
    copied_census = load_census(frozen_census_path)
    if copied_census["sha256"] != census["sha256"]:
        raise HarnessError(f"copied census changed during freeze: {census_path}")

    manifest: dict[str, Any] = {
        "schema_version": FREEZE_SCHEMA,
        "phase": phase,
        "input_root": ".",
        "census": {
            "path": "accepted-census.json",
            "bytes": frozen_census_path.stat().st_size,
            "sha256": copied_census["sha256"],
            "game_count": copied_census["game_count"],
            "source_identity_sha256": copied_census["source_identity_sha256"],
        },
        "files": sorted(file_records, key=lambda item: str(item["path"])),
    }
    manifest["manifest_sha256"] = _manifest_digest(manifest)
    _write_json(destination_root / "manifest.json", manifest)
    return manifest


def append_only_contract(base: Mapping[str, Any], append: Mapping[str, Any]) -> dict[str, Any]:
    """Check the census and file-set conditions for an append-only phase."""

    base_ids = set(base["census"]["game_ids"]) if "game_ids" in base["census"] else set()
    append_ids = set(append["census"]["game_ids"]) if "game_ids" in append["census"] else set()
    # The freeze manifest omits full IDs from its public section. The caller
    # supplies the ID sets through the private keys below.
    base_ids = set(base.get("_game_ids", base_ids))
    append_ids = set(append.get("_game_ids", append_ids))
    base_files = {str(item["path"]) for item in base.get("files", [])}
    append_files = {str(item["path"]) for item in append.get("files", [])}
    added_ids = sorted(append_ids - base_ids)
    removed_ids = sorted(base_ids - append_ids)
    checks = {
        "base_ids_subset_append_ids": not removed_ids,
        "same_input_file_set": base_files == append_files,
        "append_census_count_not_smaller": len(append_ids) >= len(base_ids),
        "new_game_count": len(added_ids),
        "removed_game_count": len(removed_ids),
        "added_identity_sha256": source_identity_sha256(added_ids),
    }
    checks["valid"] = bool(
        checks["base_ids_subset_append_ids"]
        and checks["same_input_file_set"]
        and checks["append_census_count_not_smaller"]
    )
    return checks


def freeze_inputs(
    *,
    base_root: Path,
    base_census: Path,
    append_root: Path,
    append_census: Path,
    output_root: Path,
    input_relative_paths: Sequence[str],
) -> dict[str, Any]:
    """Freeze both benchmark phases and write the binding manifest."""

    output_root.mkdir(parents=True, exist_ok=True)
    base_census_payload = load_census(base_census)
    append_census_payload = load_census(append_census)
    base_manifest = freeze_phase(
        source_root=base_root,
        census_path=base_census,
        destination_root=output_root / "frozen" / "cold",
        input_relative_paths=input_relative_paths,
        phase="cold",
    )
    append_manifest = freeze_phase(
        source_root=append_root,
        census_path=append_census,
        destination_root=output_root / "frozen" / "append_only",
        input_relative_paths=input_relative_paths,
        phase="append_only",
    )
    base_manifest["_game_ids"] = list(base_census_payload["game_ids"])
    append_manifest["_game_ids"] = list(append_census_payload["game_ids"])
    append_checks = append_only_contract(base_manifest, append_manifest)
    # The private IDs are used for validation only. Keep them out of the
    # persisted manifest so the input digest stays small and stable.
    base_manifest.pop("_game_ids", None)
    append_manifest.pop("_game_ids", None)
    freeze_manifest = {
        "schema_version": FREEZE_SCHEMA,
        "base": base_manifest,
        "append_only": append_manifest,
        "append_only_checks": append_checks,
    }
    freeze_manifest["freeze_sha256"] = _manifest_digest(freeze_manifest)
    _write_json(output_root / "freeze.json", freeze_manifest)
    return freeze_manifest


def verify_frozen_fixture(fixture_root: Path, manifest: Mapping[str, Any]) -> None:
    """Verify that an adapter did not mutate the copied fixture."""

    manifest_path = fixture_root / "manifest.json"
    _require_regular_file(manifest_path, "fixture manifest")
    try:
        persisted_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"fixture manifest cannot be read: {manifest_path}") from error
    if persisted_manifest != dict(manifest) or _manifest_digest(persisted_manifest) != manifest.get("manifest_sha256"):
        raise HarnessError("adapter changed the frozen fixture manifest")
    census = manifest["census"]
    census_path = fixture_root / str(census["path"])
    _require_regular_file(census_path, "frozen census")
    if census_path.stat().st_size != int(census["bytes"]) or sha256_file(census_path) != census["sha256"]:
        raise HarnessError("adapter changed the frozen census")
    for record in manifest.get("files", []):
        input_path = fixture_root / str(record["path"])
        _require_regular_file(input_path, "frozen input")
        if input_path.stat().st_size != int(record["bytes"]) or sha256_file(input_path) != record["sha256"]:
            raise HarnessError(f"adapter changed the frozen input: {record['path']}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(_canonical_json_bytes(value))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_log(path: Path, value: str | bytes, *, max_bytes: int = MAX_LOG_BYTES) -> dict[str, Any]:
    """Persist bounded subprocess output and describe the stored bytes."""

    if max_bytes <= 0:
        raise ValueError("log byte limit must be positive")
    if isinstance(value, bytes):
        raw = value
    else:
        raw = value.encode("utf-8", errors="replace")
    original_bytes = len(raw)
    truncated = original_bytes > max_bytes
    if truncated:
        marker = f"\n...[output truncated; original_bytes={original_bytes}]...\n".encode("utf-8")
        if len(marker) >= max_bytes:
            payload = marker[:max_bytes]
        else:
            available = max_bytes - len(marker)
            head_bytes = available // 2
            tail_bytes = available - head_bytes
            payload = raw[:head_bytes] + marker + raw[-tail_bytes:]
    else:
        payload = raw
    if path.is_symlink():
        raise HarnessError(f"log destination must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_bytes(payload)
    except OSError as error:
        raise HarnessError(f"cannot write subprocess log: {path}") from error
    return {
        "path": str(path),
        "sha256": _sha256_bytes(payload),
        "bytes": len(payload),
        "original_bytes": original_bytes,
        "truncated": truncated,
    }


def _parse_command(raw: str, label: str) -> list[str]:
    try:
        command = json.loads(raw)
    except json.JSONDecodeError as error:
        raise HarnessError(f"{label} must be a JSON argv array") from error
    if not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command):
        raise HarnessError(f"{label} must be a non-empty JSON argv array")
    return command


def _normalise_counts(value: object) -> dict[str, int]:
    if isinstance(value, Mapping) and isinstance(value.get("counts"), Mapping):
        value = value["counts"]
    if not isinstance(value, Mapping):
        raise HarnessError("call-count payload must be an object or an object with a counts object")
    result: dict[str, int] = {}
    for raw_name, raw_count in value.items():
        name = str(raw_name)
        if not name or isinstance(raw_count, bool) or not isinstance(raw_count, int) or raw_count < 0:
            raise HarnessError(f"invalid call count for {name!r}")
        result[name] = raw_count
    return dict(sorted(result.items()))


def _call_counts(path: Path, stdout: str, stderr: str) -> dict[str, Any]:
    if path.exists():
        _require_regular_file(path, "call-count file")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise HarnessError(f"call-count file cannot be read: {path}") from error
        return {"status": "file", "counts": _normalise_counts(payload)}
    markers: dict[str, int] = {}
    for name in CALL_MARKER.findall(stdout + "\n" + stderr):
        markers[name] = markers.get(name, 0) + 1
    if markers:
        return {"status": "marker", "counts": dict(sorted(markers.items()))}
    return {"status": "unavailable", "counts": {}}


def _output_view(outputs: Mapping[str, Any]) -> dict[str, Any]:
    """Remove paths that differ between baseline and candidate work dirs."""

    view: dict[str, Any] = {}
    for name, descriptor in outputs.items():
        if not isinstance(descriptor, Mapping):
            raise HarnessError(f"output descriptor is not an object: {name}")
        clean = {key: value for key, value in descriptor.items() if key != "path"}
        view[str(name)] = clean
    return dict(sorted(view.items()))


def _validate_output_manifest(path: Path, expected_binding: Mapping[str, Any]) -> dict[str, Any]:
    _require_regular_file(path, "output manifest")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HarnessError(f"output manifest cannot be read: {path}") from error
    if not isinstance(payload, Mapping) or payload.get("schema_version") != OUTPUT_SCHEMA:
        raise HarnessError(f"output manifest schema is invalid: {path}")
    binding = payload.get("source")
    if not isinstance(binding, Mapping):
        raise HarnessError(f"output manifest source binding is missing: {path}")
    actual_binding = {key: binding.get(key) for key in expected_binding}
    if actual_binding != dict(expected_binding):
        raise HarnessError(
            "output source binding does not match fixture: "
            + json.dumps({"expected": expected_binding, "actual": dict(binding)}, sort_keys=True)
        )
    outputs = payload.get("outputs")
    if not isinstance(outputs, Mapping) or not outputs:
        raise HarnessError(f"output manifest has no outputs: {path}")
    for name, descriptor in outputs.items():
        if not isinstance(name, str) or not name:
            raise HarnessError(f"output name is invalid: {name!r}")
        if not isinstance(descriptor, Mapping):
            raise HarnessError(f"output descriptor is invalid: {name}")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise HarnessError(f"output sha256 is invalid: {name}")
        artifact_path = descriptor.get("path")
        if artifact_path is not None:
            if not isinstance(artifact_path, str) or not artifact_path:
                raise HarnessError(f"output path is invalid: {name}")
            resolved = Path(artifact_path)
            if not resolved.is_absolute():
                resolved = path.parent / resolved
            _require_regular_file(resolved, f"output artifact {name}")
            if sha256_file(resolved) != digest:
                raise HarnessError(f"output artifact hash does not match descriptor: {name}")
        for key in ("bytes", "rows"):
            if key in descriptor and (
                isinstance(descriptor[key], bool)
                or not isinstance(descriptor[key], int)
                or descriptor[key] < 0
            ):
                raise HarnessError(f"output {key} is invalid: {name}")
    semantic = payload.get("semantic", {})
    if not isinstance(semantic, Mapping):
        raise HarnessError(f"output semantic section is invalid: {path}")
    run_metadata = payload.get("run", {})
    if not isinstance(run_metadata, Mapping):
        raise HarnessError(f"output run section is invalid: {path}")
    return {
        "binding": dict(expected_binding),
        "outputs": _output_view(outputs),
        "semantic": dict(semantic),
        "run": dict(run_metadata),
    }


def _public_phase_result(result: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if not str(key).startswith("_")}


def _run_adapter(
    *,
    command: Sequence[str],
    command_cwd: Path,
    fixture_root: Path,
    fixture_manifest: Mapping[str, Any],
    runtime_root: Path,
    runtime_owner: str,
    phase: str,
    variant: str,
    expected_binding: Mapping[str, Any],
    run_root: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Execute one adapter and collect timing, call counts, and outputs."""

    run_root.mkdir(parents=True, exist_ok=True)
    output_path = run_root / f"{variant}.output.json"
    calls_path = run_root / f"{variant}.calls.json"
    output_path.unlink(missing_ok=True)
    calls_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "TZ": "UTC",
            "LC_ALL": "C",
            "SCRYGLASS_RATING_AUTORESEARCH_INPUT_ROOT": str(fixture_root),
            "SCRYGLASS_RATING_AUTORESEARCH_CENSUS_PATH": str(fixture_root / "accepted-census.json"),
            "SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST": str(fixture_root / "manifest.json"),
            "SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST_SHA256": str(expected_binding["input_manifest_sha256"]),
            "SCRYGLASS_RATING_AUTORESEARCH_OUTPUT_MANIFEST": str(output_path),
            "SCRYGLASS_RATING_AUTORESEARCH_CALL_COUNTS_PATH": str(calls_path),
            "SCRYGLASS_RATING_AUTORESEARCH_RUNTIME_ROOT": str(runtime_root),
            "SCRYGLASS_RATING_AUTORESEARCH_RUNTIME_OWNER": runtime_owner,
            "SCRYGLASS_RUNTIME_ROOT": str(runtime_root),
            "SCRYGLASS_RATING_AUTORESEARCH_PHASE": phase,
            "SCRYGLASS_RATING_AUTORESEARCH_VARIANT": variant,
        }
    )
    started = time.perf_counter()
    stdout = ""
    stderr = ""
    returncode: int | None = None
    timed_out = False
    error: str | None = None
    try:
        completed = subprocess.run(
            list(command),
            cwd=command_cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
        returncode = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as timeout_error:
        timed_out = True
        error = f"timed out after {timeout_seconds:.3f}s"
        stdout = _decode_output(timeout_error.stdout)
        stderr = _decode_output(timeout_error.stderr)
    except OSError as os_error:
        error = str(os_error)
    elapsed_seconds = time.perf_counter() - started
    log_error: str | None = None
    stdout_log: dict[str, Any] | None = None
    stderr_log: dict[str, Any] | None = None
    try:
        stdout_log = _write_log(run_root / f"{variant}.stdout.log", stdout)
        stderr_log = _write_log(run_root / f"{variant}.stderr.log", stderr)
    except HarnessError as logging_error:
        log_error = str(logging_error)
    result: dict[str, Any] = {
        "variant": variant,
        "phase": phase,
        "command": list(command),
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": round(elapsed_seconds, 6),
        "wall_ms": round(elapsed_seconds * 1000.0, 3),
        "stdout_sha256": _sha256_bytes(stdout.encode("utf-8", errors="replace")),
        "stderr_sha256": _sha256_bytes(stderr.encode("utf-8", errors="replace")),
        "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
        "stderr_bytes": len(stderr.encode("utf-8", errors="replace")),
        "stdout_log": stdout_log,
        "stderr_log": stderr_log,
        "fixture_integrity": "unknown",
        "call_counts": {"status": "unavailable", "counts": {}},
        "status": "failed",
    }
    if log_error:
        result["log_error"] = log_error
        if error is None:
            error = log_error
    if error:
        result["error"] = error
        return result
    try:
        result["call_counts"] = _call_counts(calls_path, stdout, stderr)
    except HarnessError as count_error:
        result["error"] = str(count_error)
        return result
    try:
        verify_frozen_fixture(fixture_root, fixture_manifest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, HarnessError, KeyError) as fixture_error:
        result["error"] = f"fixture integrity check failed: {fixture_error}"
        result["fixture_integrity"] = "failed"
        return result
    result["fixture_integrity"] = "ok"
    if returncode != 0 or timed_out:
        result["error"] = f"adapter returned {returncode}" if not timed_out else error
        return result
    try:
        validated = _validate_output_manifest(output_path, expected_binding)
    except HarnessError as output_error:
        result["error"] = str(output_error)
        return result
    result["status"] = "ok"
    result["output_digest"] = _sha256_bytes(_canonical_json_bytes(validated["outputs"]))
    result["semantic_digest"] = _sha256_bytes(_canonical_json_bytes(validated["semantic"]))
    result["run_metadata"] = validated["run"]
    result["artifact_names"] = sorted(validated["outputs"])
    result["_validated_output"] = validated
    return result


def compare_outputs(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Compare the exact source bindings and output descriptors."""

    reasons: list[str] = []
    if baseline.get("status") != "ok":
        reasons.append("baseline adapter did not produce a valid result")
    if candidate.get("status") != "ok":
        reasons.append("candidate adapter did not produce a valid result")
    baseline_output = baseline.get("_validated_output")
    candidate_output = candidate.get("_validated_output")
    if isinstance(baseline_output, Mapping) and isinstance(candidate_output, Mapping):
        if baseline_output.get("binding") != candidate_output.get("binding"):
            reasons.append("baseline and candidate source bindings differ")
        if baseline_output.get("outputs") != candidate_output.get("outputs"):
            reasons.append("baseline and candidate output descriptors differ")
        if baseline_output.get("semantic") != candidate_output.get("semantic"):
            reasons.append("baseline and candidate semantic outputs differ")
    else:
        reasons.append("one or both output manifests are unavailable")
    return {
        "correct": not reasons,
        "reasons": reasons,
        "baseline_output_digest": baseline.get("output_digest"),
        "candidate_output_digest": candidate.get("output_digest"),
        "baseline_semantic_digest": baseline.get("semantic_digest"),
        "candidate_semantic_digest": candidate.get("semantic_digest"),
    }


def _decode_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _phase_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    census = manifest["census"]
    return {
        "phase": manifest["phase"],
        "source_game_count": int(census["game_count"]),
        "source_identity_sha256": str(census["source_identity_sha256"]),
        "census_sha256": str(census["sha256"]),
        "input_manifest_sha256": str(manifest["manifest_sha256"]),
    }


def run_benchmark(
    *,
    freeze_manifest: Mapping[str, Any],
    output_root: Path,
    baseline_command: Sequence[str],
    candidate_command: Sequence[str],
    command_cwd: Path,
    budget_seconds: float = DEFAULT_BUDGET_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    require_speedup: bool = False,
) -> dict[str, Any]:
    """Run the fixed cold and append-only comparison."""

    if not freeze_manifest.get("append_only_checks", {}).get("valid"):
        raise HarnessError("append-only fixture contract is invalid")
    phases: dict[str, Any] = {}
    runtime_roots = {
        variant: output_root / "runtimes" / variant
        for variant in ("baseline", "candidate")
    }
    runtime_owners = {
        variant: f"{freeze_manifest.get('freeze_sha256')}:{variant}"
        for variant in ("baseline", "candidate")
    }
    for phase_name, fixture_key in (("cold", "base"), ("append_only", "append_only")):
        fixture_manifest = freeze_manifest[fixture_key]
        fixture_root = output_root / "frozen" / phase_name
        expected_binding = _phase_binding(fixture_manifest)
        run_root = output_root / "runs" / phase_name
        baseline = _run_adapter(
            command=baseline_command,
            command_cwd=command_cwd,
            fixture_root=fixture_root,
            fixture_manifest=fixture_manifest,
            runtime_root=runtime_roots["baseline"],
            runtime_owner=runtime_owners["baseline"],
            phase=phase_name,
            variant="baseline",
            expected_binding=expected_binding,
            run_root=run_root,
            timeout_seconds=timeout_seconds,
        )
        candidate = _run_adapter(
            command=candidate_command,
            command_cwd=command_cwd,
            fixture_root=fixture_root,
            fixture_manifest=fixture_manifest,
            runtime_root=runtime_roots["candidate"],
            runtime_owner=runtime_owners["candidate"],
            phase=phase_name,
            variant="candidate",
            expected_binding=expected_binding,
            run_root=run_root,
            timeout_seconds=timeout_seconds,
        )
        comparison = compare_outputs(baseline, candidate)
        candidate_seconds = float(candidate.get("wall_seconds") or 0.0)
        baseline_seconds = float(baseline.get("wall_seconds") or 0.0)
        comparison["candidate_within_budget"] = candidate.get("status") == "ok" and candidate_seconds <= budget_seconds
        comparison["candidate_not_slower"] = (
            baseline.get("status") == "ok"
            and candidate.get("status") == "ok"
            and candidate_seconds <= baseline_seconds
        )
        comparison["speedup"] = (
            round(baseline_seconds / candidate_seconds, 6)
            if candidate_seconds > 0 and baseline.get("status") == "ok"
            else None
        )
        comparison["accepted_for_phase"] = bool(
            comparison["correct"]
            and comparison["candidate_within_budget"]
            and (comparison["candidate_not_slower"] if require_speedup else True)
        )
        phases[phase_name] = {
            "input_manifest_sha256": fixture_manifest["manifest_sha256"],
            "expected_binding": expected_binding,
            "baseline": _public_phase_result(baseline),
            "candidate": _public_phase_result(candidate),
            "comparison": comparison,
        }
    accepted = all(bool(phase["comparison"]["accepted_for_phase"]) for phase in phases.values())
    return {
        "schema_version": REPORT_SCHEMA,
        "budget_seconds": budget_seconds,
        "timeout_seconds": timeout_seconds,
        "require_speedup": require_speedup,
        "freeze_sha256": freeze_manifest.get("freeze_sha256"),
        "runtime_roots": {variant: str(path) for variant, path in runtime_roots.items()},
        "append_only_checks": freeze_manifest.get("append_only_checks"),
        "invocation_budget": {"phases": ["cold", "append_only"], "variants_per_phase": 2, "total_adapter_calls": 4},
        "phases": phases,
        "accepted": accepted,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--base-census", type=Path, required=True)
    parser.add_argument("--append-root", type=Path, required=True)
    parser.add_argument("--append-census", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--input-relative",
        action="append",
        required=True,
        help="Input path relative to each source root. Repeat for every input file.",
    )
    parser.add_argument("--baseline-command-json", help="JSON argv array for the baseline adapter")
    parser.add_argument("--candidate-command-json", help="JSON argv array for the candidate adapter")
    parser.add_argument("--command-cwd", type=Path, default=ROOT)
    parser.add_argument("--budget-seconds", type=float, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--require-speedup", action="store_true")
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.budget_seconds <= 0 or args.timeout_seconds <= 0:
        raise SystemExit("budget and timeout must be positive")
    freeze_manifest = freeze_inputs(
        base_root=args.base_root,
        base_census=args.base_census,
        append_root=args.append_root,
        append_census=args.append_census,
        output_root=args.output_root,
        input_relative_paths=args.input_relative,
    )
    if args.freeze_only:
        print(json.dumps(freeze_manifest, ensure_ascii=True, indent=2, sort_keys=True))
        return 0
    if not args.baseline_command_json or not args.candidate_command_json:
        raise SystemExit("baseline and candidate commands are required unless --freeze-only is set")
    baseline_command = _parse_command(args.baseline_command_json, "baseline command")
    candidate_command = _parse_command(args.candidate_command_json, "candidate command")
    report = run_benchmark(
        freeze_manifest=freeze_manifest,
        output_root=args.output_root,
        baseline_command=baseline_command,
        candidate_command=candidate_command,
        command_cwd=args.command_cwd,
        budget_seconds=args.budget_seconds,
        timeout_seconds=args.timeout_seconds,
        require_speedup=args.require_speedup,
    )
    report_path = args.report or args.output_root / "report.json"
    _write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
