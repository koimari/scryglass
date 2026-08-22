"""Run the four future-value rating variants in one isolated research run.

The runner wires the existing phase, fold, producer, bundle, calibration,
model, and uncertainty commands together.  It owns the output layout and
records a receipt for every stage.  The runner never writes a public pack and
never grants model authority.

The source, freeze, and series crosswalk are checked before any child process
starts.  Each child receives the exact paths and hashes recorded in the stage
receipt.  A resumed run skips a stage only after its receipt and every output
hash pass validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    load_verified_leaguepedia_series_crosswalk,
    validate_future_value_source_receipt_payload,
)


SCHEMA_VERSION = "scryglass:future-value-fourway-run:v1"
STAGE_RECEIPT_SCHEMA_VERSION = "scryglass:future-value-fourway-stage:v1"
RUN_CONFIG_SCHEMA_VERSION = "scryglass:future-value-fourway-config:v1"
TRUST_MANIFEST_SCHEMA_VERSION = "scryglass:future-value-fourway-trust:v1"
VARIANTS = ("current_only", "future_player_form", "scaling_curve", "both")
DEFAULT_WORKERS = max(1, min(4, int(os.cpu_count() or 1)))
DEFAULT_DRAWS = 2000
DEFAULT_SEED = 461
DEFAULT_MAX_LOG_BYTES = 1_048_576
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
SOURCE_FILES = {
    "maps": "maps.parquet",
    "players": "oe_player_games.parquet",
    "teams": "oe_team_games.parquet",
}
PHASE_FILES = (
    "future-phase-candidate.json",
    "future-phase-evaluation.json",
    "run-receipt.json",
)
FOLD_FILES = ("fold-spec-bundle.json", "fold-1-spec.json", "fold-2-spec.json", "fold-3-spec.json")
CURRENT_FILES = (
    "current-rating-feature-ledger.parquet",
    "current-rating-feature-ledger.receipt.json",
    "current-rating-producer-receipt.json",
    "current-rating-adapter.json",
    "current-rating-producer-manifest.json",
    "current-rating-run.json",
)
SCALING_FILES = (
    "scaling-native.parquet",
    "scaling-native-receipt.json",
    "scaling-features.parquet",
    "scaling-producer-receipt.json",
    "scaling-adapter.json",
    "scaling-producer-manifest.json",
    "scaling-run.json",
    "scaling-series-binding.json",
)
DEFAULT_TRUST_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "data/lol/v2/evaluation/future-value-fourway-run-trust-v1.json"
)


class FourwayRunError(RuntimeError):
    """The four-variant run cannot continue safely."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FourwayRunError("receipt value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise FourwayRunError(f"file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise FourwayRunError(f"file cannot be read: {path}") from error
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_path(path)}


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FourwayRunError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FourwayRunError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise FourwayRunError(f"{label} must be a JSON object")
    return value


def _require_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise FourwayRunError(f"{label} is not a SHA-256 value")
    return value.lower()


def _safe_path(path: Path, label: str, *, directory: bool = False) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise FourwayRunError(f"{label} must be an absolute path without '..'")
    if path.is_symlink():
        raise FourwayRunError(f"{label} is a symlink: {path}")
    if directory:
        if not path.is_dir():
            raise FourwayRunError(f"{label} is not a directory: {path}")
    elif not path.is_file():
        raise FourwayRunError(f"{label} is not a file: {path}")
    return path.resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _bounded_bytes(value: bytes, limit: int) -> tuple[bytes, bool, int]:
    if limit < 1:
        raise FourwayRunError("max log bytes must be positive")
    original = len(value)
    if original <= limit:
        return value, False, original
    return value[:limit], True, original


def _write_bounded_log(path: Path, value: bytes, limit: int) -> dict[str, Any]:
    stored, truncated, original = _bounded_bytes(value, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stored)
    return {
        "path": str(path),
        "bytes": len(stored),
        "original_bytes": original,
        "truncated": truncated,
        "sha256": _sha256_path(path),
    }


@dataclass(frozen=True)
class Job:
    """One child process in a stage."""

    name: str
    command: tuple[str, ...]
    output_roots: tuple[Path, ...]
    expected_files: tuple[Path, ...]
    input_paths: tuple[Path, ...]


@dataclass(frozen=True)
class Stage:
    """A sequential stage, with optional parallel child jobs."""

    name: str
    jobs: tuple[Job, ...]
    output_roots: tuple[Path, ...]
    expected_files: tuple[Path, ...]


@dataclass(frozen=True)
class RunConfig:
    source_root: Path
    source_receipt: Path
    freeze: Path
    freeze_root: Path
    crosswalk: Path
    crosswalk_receipt: Path
    crosswalk_receipt_file_sha256: str
    trust_manifest: Path
    output_root: Path
    outer_evaluation_start: str
    workers: int = DEFAULT_WORKERS
    folds: int = 3
    calibration_folds: int = 4
    draws: int = DEFAULT_DRAWS
    seed: int = DEFAULT_SEED
    max_log_bytes: int = DEFAULT_MAX_LOG_BYTES

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise FourwayRunError("workers must be positive")
        if self.folds != 3:
            raise FourwayRunError("the fourway bundle requires exactly three outer folds")
        if self.calibration_folds < 2:
            raise FourwayRunError("calibration fold count must be at least two")
        if self.draws < 1:
            raise FourwayRunError("uncertainty draws must be positive")
        if self.max_log_bytes < 1:
            raise FourwayRunError("max log bytes must be positive")
        _require_hash(self.crosswalk_receipt_file_sha256, "crosswalk receipt file hash")


def _stage_dir(config: RunConfig, name: str) -> Path:
    return config.output_root / "stages" / name


def _receipt_dir(config: RunConfig) -> Path:
    return config.output_root / "receipts"


def _logs_dir(config: RunConfig) -> Path:
    return config.output_root / "logs"


def _python_module(module: str, *args: object) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *(str(arg) for arg in args))


def _common_crosswalk_args(config: RunConfig) -> tuple[str, ...]:
    return (
        "--crosswalk",
        str(config.crosswalk),
        "--crosswalk-receipt",
        str(config.crosswalk_receipt),
        "--crosswalk-receipt-file-sha256",
        config.crosswalk_receipt_file_sha256,
    )


def build_stage_plan(
    config: RunConfig,
    *,
    phase_artifact_sha256: str | None = None,
    phase_receipt_file_sha256: str | None = None,
    outer_evaluation_start: str | None = None,
) -> tuple[Stage, ...]:
    """Return the exact child command plan without running a child process."""

    source = config.source_root
    receipt = config.source_receipt
    evaluation_start = (
        config.outer_evaluation_start
        if outer_evaluation_start is None
        else outer_evaluation_start
    )
    phase_root = _stage_dir(config, "phase")
    phase = Stage(
        name="phase",
        jobs=(
            Job(
                name="phase",
                command=_python_module(
                    "benchmarks.rebuild_future_phase",
                    "--receipt",
                    receipt,
                    "--freeze-root",
                    config.freeze_root,
                    "--source-root",
                    source,
                    *_common_crosswalk_args(config),
                    "--output-root",
                    phase_root,
                ),
                output_roots=(phase_root,),
                expected_files=tuple(phase_root / name for name in PHASE_FILES),
                input_paths=(config.freeze, receipt, *[source / name for name in SOURCE_FILES.values()], config.crosswalk, config.crosswalk_receipt),
            ),
        ),
        output_roots=(phase_root,),
        expected_files=tuple(phase_root / name for name in PHASE_FILES),
    )

    folds_root = _stage_dir(config, "fold-specs")
    folds = Stage(
        name="fold_specs",
        jobs=(
            Job(
                name="fold_specs",
                command=_python_module(
                    "benchmarks.build_future_value_fold_specs",
                    "--maps",
                    source / SOURCE_FILES["maps"],
                    "--source-receipt",
                    receipt,
                    *_common_crosswalk_args(config),
                    "--output-root",
                    folds_root,
                    "--folds",
                    config.folds,
                ),
                output_roots=(folds_root,),
                expected_files=tuple(folds_root / name for name in FOLD_FILES),
                input_paths=(receipt, source / SOURCE_FILES["maps"], config.crosswalk, config.crosswalk_receipt),
            ),
        ),
        output_roots=(folds_root,),
        expected_files=tuple(folds_root / name for name in FOLD_FILES),
    )

    producer_jobs: list[Job] = []
    producer_root = _stage_dir(config, "fold-producers")
    for fold in range(1, config.folds + 1):
        spec = folds_root / f"fold-{fold}-spec.json"
        current_root = producer_root / f"fold-{fold}" / "current-v2"
        scaling_root = producer_root / f"fold-{fold}" / "scaling-v2"
        producer_jobs.append(
            Job(
                name=f"fold-{fold}-current",
                command=_python_module(
                    "benchmarks.build_current_rating_fold_artifact",
                    "--source-root",
                    source,
                    "--source-receipt",
                    receipt,
                    "--fold-spec",
                    spec,
                    "--output-dir",
                    current_root,
                    *_common_crosswalk_args(config),
                ),
                output_roots=(current_root,),
                expected_files=tuple(current_root / name for name in CURRENT_FILES),
                input_paths=(receipt, spec, *[source / name for name in SOURCE_FILES.values()], config.crosswalk, config.crosswalk_receipt),
            )
        )
        producer_jobs.append(
            Job(
                name=f"fold-{fold}-scaling",
                command=_python_module(
                    "benchmarks.build_scaling_fold_artifact",
                    "--source-root",
                    source,
                    "--source-receipt",
                    receipt,
                    "--fold-spec",
                    spec,
                    "--output-dir",
                    scaling_root,
                    *_common_crosswalk_args(config),
                ),
                output_roots=(scaling_root,),
                expected_files=tuple(scaling_root / name for name in SCALING_FILES),
                input_paths=(receipt, spec, *[source / name for name in SOURCE_FILES.values()], config.crosswalk, config.crosswalk_receipt),
            )
        )
    producers = Stage(
        name="fold_producers",
        jobs=tuple(producer_jobs),
        output_roots=(producer_root,),
        expected_files=tuple(path for job in producer_jobs for path in job.expected_files),
    )

    bundle_root = _stage_dir(config, "bundle")
    bundle_path = bundle_root / "four-variant-feature-ledger-bundle.json"
    bundle = Stage(
        name="bundle",
        jobs=(
            Job(
                name="bundle",
                command=_python_module(
                    "benchmarks.future_value_four_variant_bundle",
                    "--source-root",
                    source,
                    "--source-receipt",
                    receipt,
                    "--folds-root",
                    producer_root,
                    "--fold-specs-root",
                    folds_root,
                    *_common_crosswalk_args(config),
                    "--inner-output-root",
                    bundle_root / "nested-inner-v1",
                    "--output",
                    bundle_path,
                ),
                output_roots=(bundle_root,),
                expected_files=(bundle_path,),
                input_paths=(receipt, *[source / name for name in SOURCE_FILES.values()], config.crosswalk, config.crosswalk_receipt, *[folds_root / f"fold-{fold}-spec.json" for fold in range(1, config.folds + 1)], *[producer_root / f"fold-{fold}" / "current-v2" / name for fold in range(1, config.folds + 1) for name in CURRENT_FILES], *[producer_root / f"fold-{fold}" / "scaling-v2" / name for fold in range(1, config.folds + 1) for name in SCALING_FILES]),
            ),
        ),
        output_roots=(bundle_root,),
        expected_files=(bundle_path,),
    )

    calibration_root = _stage_dir(config, "calibration")
    calibration_path = calibration_root / "calibration-prelude.json"
    calibration = Stage(
        name="calibration",
        jobs=(
            Job(
                name="calibration",
                command=_python_module(
                    "benchmarks.build_future_value_calibration_prelude",
                    "--source-root",
                    source,
                    "--source-receipt",
                    receipt,
                    *_common_crosswalk_args(config),
                    "--producer-root",
                    calibration_root / "producer",
                    "--outer-evaluation-start",
                    evaluation_start,
                    "--fold-count",
                    config.calibration_folds,
                    "--output",
                    calibration_path,
                ),
                output_roots=(calibration_root,),
                expected_files=(calibration_path,),
                input_paths=(receipt, *[source / name for name in SOURCE_FILES.values()], config.crosswalk, config.crosswalk_receipt),
            ),
        ),
        output_roots=(calibration_root,),
        expected_files=(calibration_path,),
    )

    evaluation_root = _stage_dir(config, "evaluation")
    phase_artifact_hash_arg = (
        "__PHASE_ARTIFACT_SHA256__"
        if phase_artifact_sha256 is None
        else str(phase_artifact_sha256)
    )
    phase_receipt_hash_arg = (
        "__PHASE_RECEIPT_FILE_SHA256__"
        if phase_receipt_file_sha256 is None
        else str(phase_receipt_file_sha256)
    )
    evaluation_jobs: list[Job] = []
    for variant in VARIANTS:
        variant_root = evaluation_root / variant
        evaluation_jobs.append(
            Job(
                name=f"evaluation-{variant}",
                command=_python_module(
                    "lol_kills.research.future_value_training",
                    "--fit-model",
                    "--oe-root",
                    source,
                    "--freeze",
                    config.freeze,
                    "--source-receipt",
                    receipt,
                    "--model-output",
                    variant_root / "model.json",
                    "--runtime-receipt",
                    variant_root / "runtime.json",
                    "--n-folds",
                    config.folds,
                    "--rating-variant",
                    variant,
                    "--feature-ledger-bundle",
                    bundle_path,
                    *_common_crosswalk_args(config),
                    "--phase-artifact",
                    phase_root / "future-phase-candidate.json",
                    "--phase-receipt",
                    phase_root / "run-receipt.json",
                    "--phase-artifact-sha256",
                    phase_artifact_hash_arg,
                    "--phase-receipt-file-sha256",
                    phase_receipt_hash_arg,
                    "--phase-artifact-kind",
                    "candidate",
                    "--calibration-prior",
                    calibration_path,
                ),
                output_roots=(variant_root,),
                expected_files=(variant_root / "model.json", variant_root / "runtime.json"),
                input_paths=(config.freeze, receipt, bundle_path, calibration_path, phase_root / "future-phase-candidate.json", phase_root / "run-receipt.json", config.crosswalk, config.crosswalk_receipt),
            )
        )
    evaluations = Stage(
        name="evaluations",
        jobs=tuple(evaluation_jobs),
        output_roots=(evaluation_root,),
        expected_files=tuple(path for job in evaluation_jobs for path in job.expected_files),
    )

    uncertainty_root = _stage_dir(config, "paired-uncertainty")
    uncertainty = Stage(
        name="paired_uncertainty",
        jobs=(
            Job(
                name="paired_uncertainty",
                command=_python_module(
                    "benchmarks.future_value_paired_uncertainty",
                    "--evaluation-root",
                    evaluation_root,
                    "--bundle",
                    bundle_path,
                    "--output-dir",
                    uncertainty_root,
                    "--draws",
                    config.draws,
                    "--seed",
                    config.seed,
                ),
                output_roots=(uncertainty_root,),
                expected_files=(uncertainty_root / "paired-uncertainty.json", uncertainty_root / "paired-uncertainty.csv"),
                input_paths=(bundle_path, *[evaluation_root / variant / "model.json" for variant in VARIANTS]),
            ),
        ),
        output_roots=(uncertainty_root,),
        expected_files=(uncertainty_root / "paired-uncertainty.json", uncertainty_root / "paired-uncertainty.csv"),
    )
    return (phase, folds, producers, bundle, calibration, evaluations, uncertainty)


def _plan_digest(stage: Stage) -> str:
    payload = {
        "stage": stage.name,
        "jobs": [
            {
                "name": job.name,
                "command": list(job.command),
                "output_roots": [str(path) for path in job.output_roots],
                "expected_files": [str(path) for path in job.expected_files],
                "input_paths": [str(path) for path in job.input_paths],
            }
            for job in stage.jobs
        ],
    }
    return _sha256_bytes(_canonical_bytes(payload))


def _stage_input_records(stage: Stage) -> dict[str, dict[str, Any]]:
    paths = sorted({path.resolve() for job in stage.jobs for path in job.input_paths}, key=str)
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        records[str(path)] = _file_record(path)
    return records


def _collect_output_records(roots: Iterable[Path]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_symlink():
            raise FourwayRunError(f"stage output root is a symlink: {root}")
        if not root.is_dir():
            raise FourwayRunError(f"stage output root is missing: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise FourwayRunError(f"stage output contains a symlink: {path}")
            if path.is_file():
                paths.add(path)
    return [_file_record(path) for path in sorted(paths, key=str)]


def _ensure_empty_roots(roots: Iterable[Path]) -> None:
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if root.exists() and root.is_symlink():
            raise FourwayRunError(f"stage output root is a symlink: {root}")
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FourwayRunError(f"stage output root is not empty: {root}")


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise FourwayRunError(f"receipt already exists: {path}")
    body = dict(payload)
    body["receipt_sha256"] = _sha256_bytes(_canonical_bytes(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(body))
    return body


def _validate_receipt_hash(path: Path, label: str) -> dict[str, Any]:
    value = _load_json(path, label)
    claimed = _require_hash(value.get("receipt_sha256"), f"{label} receipt hash")
    body = dict(value)
    body.pop("receipt_sha256", None)
    if _sha256_bytes(_canonical_bytes(body)) != claimed:
        raise FourwayRunError(f"{label} receipt hash changed")
    return value


def _validate_stage_receipt(stage: Stage, receipt_path: Path, config: RunConfig) -> dict[str, Any]:
    receipt = _validate_receipt_hash(receipt_path, f"stage {stage.name}")
    if receipt.get("schema_version") != STAGE_RECEIPT_SCHEMA_VERSION:
        raise FourwayRunError(f"stage {stage.name} receipt schema changed")
    if receipt.get("status") != "completed":
        raise FourwayRunError(f"stage {stage.name} is not complete")
    if receipt.get("stage") != stage.name or receipt.get("stage_plan_sha256") != _plan_digest(stage):
        raise FourwayRunError(f"stage {stage.name} command plan changed")
    source = receipt.get("source")
    if not isinstance(source, Mapping) or source.get("source_receipt_path") != str(config.source_receipt):
        raise FourwayRunError(f"stage {stage.name} source binding changed")
    for path in stage.expected_files:
        if not path.is_file() or path.is_symlink():
            raise FourwayRunError(f"stage {stage.name} output is missing: {path}")
    recorded = receipt.get("outputs")
    if not isinstance(recorded, list):
        raise FourwayRunError(f"stage {stage.name} output records are missing")
    actual = {record["path"]: record for record in _collect_output_records(stage.output_roots)}
    if set(actual) != {str(path) for path in _collect_output_records_from_records(recorded)}:
        raise FourwayRunError(f"stage {stage.name} output file set changed")
    for record in recorded:
        if not isinstance(record, Mapping):
            raise FourwayRunError(f"stage {stage.name} output record is invalid")
        path = Path(str(record.get("path")))
        current = _file_record(path)
        if current.get("bytes") != record.get("bytes") or current.get("sha256") != record.get("sha256"):
            raise FourwayRunError(f"stage {stage.name} output hash changed: {path}")
    inputs = receipt.get("inputs")
    if not isinstance(inputs, Mapping):
        raise FourwayRunError(f"stage {stage.name} input records are missing")
    for path_text, record in inputs.items():
        path = Path(str(path_text))
        current = _file_record(path)
        if current.get("bytes") != record.get("bytes") or current.get("sha256") != record.get("sha256"):
            raise FourwayRunError(f"stage {stage.name} input changed: {path}")
    logs = receipt.get("logs")
    if not isinstance(logs, Mapping):
        raise FourwayRunError(f"stage {stage.name} log records are missing")
    for record in _iter_log_records(logs):
        path = Path(str(record.get("path")))
        current = _file_record(path)
        if current.get("bytes") != record.get("bytes") or current.get("sha256") != record.get("sha256"):
            raise FourwayRunError(f"stage {stage.name} log changed: {path}")
    return receipt


def _collect_output_records_from_records(records: Sequence[Mapping[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("path"), str):
            raise FourwayRunError("output record is invalid")
        paths.append(Path(record["path"]))
    return paths


def _iter_log_records(logs: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for value in logs.values():
        if isinstance(value, Mapping) and "path" in value:
            yield value
        elif isinstance(value, Mapping):
            yield from _iter_log_records(value)


def _run_job(job: Job, config: RunConfig, stage_name: str) -> dict[str, Any]:
    for root in job.output_roots:
        _ensure_empty_roots((root,))
    log_root = _logs_dir(config) / stage_name
    stdout_path = log_root / f"{job.name}.stdout.log"
    stderr_path = log_root / f"{job.name}.stderr.log"
    started_at = _utc_now()
    started = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(name, "1")
    try:
        completed = subprocess.run(
            list(job.command),
            cwd=str(Path(__file__).resolve().parents[1]),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        stdout = completed.stdout if isinstance(completed.stdout, bytes) else str(completed.stdout or "").encode()
        stderr = completed.stderr if isinstance(completed.stderr, bytes) else str(completed.stderr or "").encode()
        exit_code = int(completed.returncode)
    except OSError as error:
        stdout = b""
        stderr = str(error).encode("utf-8", errors="replace")
        exit_code = 127
    stdout_record = _write_bounded_log(stdout_path, stdout, config.max_log_bytes)
    stderr_record = _write_bounded_log(stderr_path, stderr, config.max_log_bytes)
    elapsed = time.perf_counter() - started
    result = {
        "name": job.name,
        "command": list(job.command),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": elapsed,
        "exit_code": exit_code,
        "stdout": stdout_record,
        "stderr": stderr_record,
    }
    if exit_code != 0:
        result["error"] = "child command failed"
    return result


def _execute_stage(stage: Stage, config: RunConfig, *, resume: bool) -> dict[str, Any]:
    receipt_path = _receipt_dir(config) / f"{stage.name}.json"
    if resume and receipt_path.exists():
        return _validate_stage_receipt(stage, receipt_path, config)
    if resume and any(root.exists() and any(root.iterdir()) for root in stage.output_roots):
        raise FourwayRunError(f"stage {stage.name} has output without a valid completed receipt")
    if receipt_path.exists() or receipt_path.is_symlink():
        raise FourwayRunError(f"stage {stage.name} has an existing receipt")
    _ensure_empty_roots(stage.output_roots)
    input_records = _stage_input_records(stage)
    started_at = _utc_now()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    if len(stage.jobs) == 1:
        results.append(_run_job(stage.jobs[0], config, stage.name))
    else:
        workers = min(config.workers, len(stage.jobs))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"fourway-{stage.name}") as executor:
            futures = {
                executor.submit(_run_job, job, config, stage.name): job.name
                for job in stage.jobs
            }
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda value: str(value["name"]))
    for path_text, record in input_records.items():
        current = _file_record(Path(path_text))
        if current.get("bytes") != record.get("bytes") or current.get("sha256") != record.get("sha256"):
            raise FourwayRunError(f"stage {stage.name} input changed during execution: {path_text}")
    outputs = _collect_output_records(stage.output_roots)
    missing = [path for path in stage.expected_files if not path.is_file() or path.is_symlink()]
    status = "completed" if all(int(result["exit_code"]) == 0 for result in results) else "failed"
    if missing:
        status = "failed"
    payload = {
        "schema_version": STAGE_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "stage": stage.name,
        "stage_plan_sha256": _plan_digest(stage),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "jobs": results,
        "inputs": input_records,
        "outputs": outputs,
        "missing_outputs": [str(path) for path in missing],
        "logs": {result["name"]: {"stdout": result["stdout"], "stderr": result["stderr"]} for result in results},
        "source": {
            "source_receipt_path": str(config.source_receipt),
            "source_receipt_sha256": str(
                _load_json(config.source_receipt, "source receipt").get("receipt_sha256")
            ),
            "source_receipt_file_sha256": _sha256_path(config.source_receipt),
            "crosswalk_receipt_file_sha256": config.crosswalk_receipt_file_sha256,
        },
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
        },
    }
    receipt = _write_receipt(receipt_path, payload)
    if status != "completed":
        raise FourwayRunError(f"stage {stage.name} failed; see {receipt_path}")
    return receipt


def _validate_source_and_crosswalk(config: RunConfig) -> dict[str, Any]:
    source_root = _safe_path(config.source_root, "source root", directory=True)
    source_receipt_path = _safe_path(config.source_receipt, "source receipt")
    freeze_path = _safe_path(config.freeze, "source freeze")
    _safe_path(config.freeze_root, "freeze root", directory=True)
    crosswalk = _safe_path(config.crosswalk, "series crosswalk")
    crosswalk_receipt = _safe_path(config.crosswalk_receipt, "series crosswalk receipt")
    receipt = _load_json(source_receipt_path, "source receipt")
    freeze = _load_json(freeze_path, "source freeze")
    if freeze.get("schema_version") not in {
        "scryglass:future-value-source-freeze:v1",
        "scryglass:future-value-source-freeze:v2",
    }:
        raise FourwayRunError("source freeze schema is not supported")
    if freeze.get("source_mode") != "oe_only":
        raise FourwayRunError("source freeze is not OE-only")
    freeze_authority = freeze.get("authority")
    if not isinstance(freeze_authority, Mapping) or freeze_authority.get("research_only") is not True or any(bool(value) for key, value in freeze_authority.items() if key != "research_only"):
        raise FourwayRunError("source freeze grants authority")
    try:
        validate_future_value_source_receipt_payload(
            receipt,
            expected_receipt_sha256=str(freeze.get("reference_source_receipt_sha256")),
        )
    except FutureValueSourceError as error:
        raise FourwayRunError(f"source receipt is not verified: {error}") from error
    receipt_hash = _sha256_path(source_receipt_path)
    if freeze.get("source_receipt_file_sha256") != receipt_hash:
        raise FourwayRunError("source freeze source-receipt file hash does not match")
    freeze_receipt_path = freeze.get("source_receipt_path")
    if not isinstance(freeze_receipt_path, str) or Path(freeze_receipt_path).resolve() != source_receipt_path:
        raise FourwayRunError("source freeze receipt path does not match supplied receipt")
    if freeze.get("accepted_census", {}).get("source_game_count") != receipt.get("source_game_count") or freeze.get("accepted_census", {}).get("source_identity_sha256") != receipt.get("source_identity_sha256"):
        raise FourwayRunError("source freeze accepted census differs from receipt")
    model_census = freeze.get("model_eligible_census", {})
    if model_census.get("game_count") != receipt.get("model_eligible_game_count") or model_census.get("source_identity_sha256") != receipt.get("model_eligible_identity_sha256"):
        raise FourwayRunError("source freeze model census differs from receipt")
    source_records = receipt.get("source_files")
    if not isinstance(source_records, Mapping):
        raise FourwayRunError("source receipt file bindings are missing")
    for label, filename in SOURCE_FILES.items():
        path = source_root / filename
        record = source_records.get(label)
        if not isinstance(record, Mapping):
            raise FourwayRunError(f"source receipt is missing {label} binding")
        actual = _file_record(path)
        if actual["bytes"] != record.get("bytes") or actual["sha256"] != record.get("sha256"):
            raise FourwayRunError(f"source file does not match receipt: {label}")
        frozen = freeze.get("normalized_source_files", {}).get(label)
        if isinstance(frozen, Mapping) and (actual["bytes"] != frozen.get("bytes") or actual["sha256"] != frozen.get("sha256")):
            raise FourwayRunError(f"source file does not match freeze: {label}")
    crosswalk_binding = _validate_crosswalk_binding(
        crosswalk,
        crosswalk_receipt,
        source_receipt=receipt,
        expected_receipt_file_sha256=config.crosswalk_receipt_file_sha256,
    )
    trust = _validate_trust_manifest(
        config,
        source_receipt=receipt,
        source_receipt_file_sha256=receipt_hash,
        freeze_file_sha256=_sha256_path(freeze_path),
        crosswalk_binding=crosswalk_binding,
    )
    return {
        "trust_manifest": trust,
        "source_receipt": _file_record(source_receipt_path),
        "source_receipt_sha256": str(receipt["receipt_sha256"]),
        "source_freeze": _file_record(freeze_path),
        "source_root": str(source_root),
        "source_game_count": receipt["source_game_count"],
        "source_identity_sha256": receipt["source_identity_sha256"],
        "model_eligible_game_count": receipt["model_eligible_game_count"],
        "model_eligible_identity_sha256": receipt["model_eligible_identity_sha256"],
        "crosswalk": crosswalk_binding["artifact"],
        "crosswalk_receipt": crosswalk_binding["receipt"],
        "crosswalk_sha256": crosswalk_binding["crosswalk_sha256"],
        "crosswalk_receipt_file_sha256": config.crosswalk_receipt_file_sha256,
    }


def _validate_trust_manifest(
    config: RunConfig,
    *,
    source_receipt: Mapping[str, Any],
    source_receipt_file_sha256: str,
    freeze_file_sha256: str,
    crosswalk_binding: Mapping[str, Any],
) -> dict[str, Any]:
    path = _safe_path(config.trust_manifest, "fourway trust manifest")
    manifest = _load_json(path, "fourway trust manifest")
    body = dict(manifest)
    claimed = body.pop("manifest_sha256", None)
    if manifest.get("schema_version") != TRUST_MANIFEST_SCHEMA_VERSION:
        raise FourwayRunError("fourway trust manifest schema changed")
    if claimed != _sha256_bytes(_canonical_bytes(body)):
        raise FourwayRunError("fourway trust manifest self hash changed")
    if manifest.get("status") != "research_only":
        raise FourwayRunError("fourway trust manifest status changed")
    authority = manifest.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True or any(
        bool(value) for key, value in authority.items() if key != "research_only"
    ):
        raise FourwayRunError("fourway trust manifest grants authority")
    source = manifest.get("source")
    series = manifest.get("series")
    if not isinstance(source, Mapping) or not isinstance(series, Mapping):
        raise FourwayRunError("fourway trust manifest bindings are missing")
    expected_source = {
        "source_game_count": source_receipt.get("source_game_count"),
        "source_identity_sha256": source_receipt.get("source_identity_sha256"),
        "model_eligible_game_count": source_receipt.get("model_eligible_game_count"),
        "model_eligible_identity_sha256": source_receipt.get("model_eligible_identity_sha256"),
        "source_receipt_sha256": source_receipt.get("receipt_sha256"),
        "source_receipt_file_sha256": source_receipt_file_sha256,
        "freeze_file_sha256": freeze_file_sha256,
    }
    if dict(source) != expected_source:
        raise FourwayRunError("source does not match the immutable fourway trust manifest")
    expected_series = {
        "crosswalk_receipt_file_sha256": config.crosswalk_receipt_file_sha256,
        "crosswalk_sha256": crosswalk_binding.get("crosswalk_sha256"),
        "crosswalk_assignment_sha256": crosswalk_binding.get("assignment_sha256"),
    }
    if dict(series) != expected_series:
        raise FourwayRunError("series input does not match the immutable fourway trust manifest")
    return {**_file_record(path), "manifest_sha256": claimed}


def _validate_crosswalk_binding(
    crosswalk: Path,
    crosswalk_receipt: Path,
    *,
    source_receipt: Mapping[str, Any],
    expected_receipt_file_sha256: str,
) -> dict[str, Any]:
    """Validate artifact bytes and the artifact's canonical self-hash.

    The receipt has two different SHA-256 fields.  ``artifact.sha256`` hashes
    the JSON file bytes.  The top-level ``crosswalk_sha256`` hashes the
    canonical crosswalk payload after its self-hash field is removed.  The
    repository verifier checks both contracts and all census bindings.
    """

    artifact_record = _file_record(crosswalk)
    receipt_record = _file_record(crosswalk_receipt)
    try:
        binding = load_verified_leaguepedia_series_crosswalk(
            crosswalk,
            crosswalk_receipt,
            source_receipt=source_receipt,
            expected_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
            expected_receipt_file_sha256=expected_receipt_file_sha256,
        )
    except FutureValueSourceError as error:
        raise FourwayRunError(f"series crosswalk verification failed: {error}") from error
    if binding.get("artifact_sha256") != artifact_record["sha256"]:
        raise FourwayRunError("series crosswalk artifact byte hash changed")
    if binding.get("crosswalk_sha256") != _load_json(crosswalk, "series crosswalk artifact").get("crosswalk_sha256"):
        raise FourwayRunError("series crosswalk canonical self-hash changed")
    receipt_payload = _load_json(crosswalk_receipt, "series crosswalk receipt")
    receipt_artifact = receipt_payload.get("artifact")
    if not isinstance(receipt_artifact, Mapping):
        raise FourwayRunError("series crosswalk artifact receipt binding is missing")
    if receipt_artifact.get("sha256") != artifact_record["sha256"]:
        raise FourwayRunError("series crosswalk artifact receipt hash changed")
    if receipt_payload.get("crosswalk_sha256") != binding.get("crosswalk_sha256"):
        raise FourwayRunError("series crosswalk receipt self-hash binding changed")
    return {
        "artifact": artifact_record,
        "receipt": receipt_record,
        "artifact_sha256": binding["artifact_sha256"],
        "crosswalk_sha256": binding["crosswalk_sha256"],
        "receipt_sha256": binding["receipt_sha256"],
        "assignment_sha256": binding["assignment_sha256"],
    }


def _normalize_utc(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FourwayRunError(f"{label} is missing")
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError as error:
        raise FourwayRunError(f"{label} is not an ISO timestamp") from error
    if parsed.tzinfo is None:
        raise FourwayRunError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _validate_fold_plan_cutoff(config: RunConfig) -> str:
    root = _stage_dir(config, "fold-specs")
    bundle = _load_json(root / "fold-spec-bundle.json", "fold spec bundle")
    body = dict(bundle)
    claimed = body.pop("bundle_sha256", None)
    if claimed != _sha256_bytes(_canonical_bytes(body)):
        raise FourwayRunError("fold spec bundle self hash changed")
    source = bundle.get("source")
    if not isinstance(source, Mapping):
        raise FourwayRunError("fold spec bundle source binding is missing")
    source_receipt = _load_json(config.source_receipt, "source receipt")
    if source.get("source_receipt_sha256") != source_receipt.get("receipt_sha256"):
        raise FourwayRunError("fold spec bundle source receipt changed")
    folds = bundle.get("folds")
    if not isinstance(folds, list) or len(folds) != config.folds:
        raise FourwayRunError("fold spec bundle fold count changed")
    starts: list[str] = []
    for expected_fold, record in enumerate(folds, start=1):
        if not isinstance(record, Mapping) or record.get("fold") != expected_fold:
            raise FourwayRunError("fold spec bundle order changed")
        start = _normalize_utc(record.get("validation_start"), "fold validation start")
        if _normalize_utc(record.get("fit_window_end"), "fold fit cutoff") != start:
            raise FourwayRunError("fold validation start and fit cutoff differ")
        spec = _load_json(root / f"fold-{expected_fold}-spec.json", "fold spec")
        if _normalize_utc(spec.get("fit_window_end"), "fold spec cutoff") != start:
            raise FourwayRunError("fold spec cutoff differs from bundle")
        starts.append(start)
    derived = min(starts)
    supplied = _normalize_utc(config.outer_evaluation_start, "outer evaluation start")
    if supplied != derived:
        raise FourwayRunError(
            "outer evaluation start differs from the generated fold plan: "
            f"expected {derived}"
        )
    return derived


def _validate_scaling_series_bindings(config: RunConfig) -> None:
    source_receipt = _load_json(config.source_receipt, "source receipt")
    expected_eligible_assignment: str | None = None
    for fold in range(1, config.folds + 1):
        spec = _load_json(
            _stage_dir(config, "fold-specs") / f"fold-{fold}-spec.json",
            "fold spec",
        )
        path = (
            _stage_dir(config, "fold-producers")
            / f"fold-{fold}/scaling-v2/scaling-series-binding.json"
        )
        binding = _load_json(path, "scaling series binding")
        body = dict(binding)
        claimed = body.pop("receipt_sha256", None)
        if claimed != _sha256_bytes(_canonical_bytes(body)):
            raise FourwayRunError("scaling series binding self hash changed")
        if (
            binding.get("schema_version")
            != "scryglass:future-value-scaling-series-binding:v1"
            or binding.get("status") != "research_only"
        ):
            raise FourwayRunError("scaling series binding status changed")
        authority = binding.get("authority")
        if not isinstance(authority, Mapping) or authority.get("research_only") is not True or any(
            bool(value) for key, value in authority.items() if key != "research_only"
        ):
            raise FourwayRunError("scaling series binding grants authority")
        expected = {
            "fold": fold,
            "source_receipt_sha256": source_receipt.get("receipt_sha256"),
            "source_identity_sha256": source_receipt.get("source_identity_sha256"),
            "model_eligible_game_count": source_receipt.get("model_eligible_game_count"),
            "model_eligible_identity_sha256": source_receipt.get(
                "model_eligible_identity_sha256"
            ),
            "crosswalk_receipt_file_sha256": config.crosswalk_receipt_file_sha256,
        }
        if any(binding.get(key) != value for key, value in expected.items()):
            raise FourwayRunError("scaling series binding provenance changed")
        trust_series = _load_json(
            config.trust_manifest, "fourway trust manifest"
        )["series"]
        if binding.get("crosswalk_sha256") != trust_series["crosswalk_sha256"]:
            raise FourwayRunError("scaling series crosswalk changed")
        if binding.get("crosswalk_assignment_sha256") != trust_series[
            "crosswalk_assignment_sha256"
        ]:
            raise FourwayRunError("scaling crosswalk assignments changed")
        if tuple(binding.get("train_game_ids") or ()) != tuple(spec["train_game_ids"]):
            raise FourwayRunError("scaling series training IDs changed")
        if tuple(binding.get("validation_game_ids") or ()) != tuple(spec["validation_game_ids"]):
            raise FourwayRunError("scaling series validation IDs changed")
        if binding.get("fit_window_end") != spec.get("fit_window_end"):
            raise FourwayRunError("scaling series cutoff changed")
        for prefix in ("train", "validation"):
            for suffix in (
                "series_ids",
                "series_count",
                "series_identity_sha256",
            ):
                key = f"{prefix}_{suffix}"
                if binding.get(key) != spec.get(key):
                    raise FourwayRunError(
                        "scaling series partition differs from fold spec"
                    )
        eligible_assignment = _require_hash(
            binding.get("eligible_series_assignment_sha256"),
            "eligible series assignment hash",
        )
        _require_hash(binding.get("fold_series_assignment_sha256"), "fold series assignment hash")
        if expected_eligible_assignment is None:
            expected_eligible_assignment = eligible_assignment
        elif eligible_assignment != expected_eligible_assignment:
            raise FourwayRunError("scaling producers used different series assignments")


def _config_payload(config: RunConfig, bindings: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "source": dict(bindings),
        "inputs": {
            "source_root": str(config.source_root),
            "source_receipt": str(config.source_receipt),
            "freeze": str(config.freeze),
            "freeze_root": str(config.freeze_root),
            "crosswalk": str(config.crosswalk),
            "crosswalk_receipt": str(config.crosswalk_receipt),
            "crosswalk_receipt_file_sha256": config.crosswalk_receipt_file_sha256,
            "trust_manifest": str(config.trust_manifest),
        },
        "outer_evaluation_start": config.outer_evaluation_start,
        "workers": config.workers,
        "folds": config.folds,
        "calibration_folds": config.calibration_folds,
        "draws": config.draws,
        "seed": config.seed,
        "max_log_bytes": config.max_log_bytes,
        "variants": list(VARIANTS),
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
        },
    }


def _write_or_validate_config(config: RunConfig, bindings: Mapping[str, Any], *, resume: bool) -> dict[str, Any]:
    path = config.output_root / "run-config.json"
    expected = _config_payload(config, bindings)
    expected_hash = _sha256_bytes(_canonical_bytes(expected))
    expected["config_sha256"] = expected_hash
    if resume:
        actual = _load_json(path, "run config")
        if actual.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION or actual.get("config_sha256") != expected_hash:
            raise FourwayRunError("run configuration changed")
        body = dict(actual)
        claimed = body.pop("config_sha256", None)
        if _sha256_bytes(_canonical_bytes(body)) != claimed:
            raise FourwayRunError("run configuration hash changed")
        return actual
    if path.exists() or path.is_symlink():
        raise FourwayRunError("run configuration already exists")
    path.write_bytes(_canonical_bytes(expected))
    return expected


def _write_final_receipt(config: RunConfig, bindings: Mapping[str, Any], stage_receipts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    path = config.output_root / "run-receipt.json"
    if path.exists() or path.is_symlink():
        return _validate_receipt_hash(path, "final run")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only_complete",
        "completed_at": _utc_now(),
        "source": dict(bindings),
        "stages": [
            {
                "stage": receipt.get("stage"),
                "status": receipt.get("status"),
                "receipt_sha256": receipt.get("receipt_sha256"),
                "outputs": receipt.get("outputs"),
            }
            for receipt in stage_receipts
        ],
        "workers": config.workers,
        "variants": list(VARIANTS),
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
        },
    }
    return _write_receipt(path, payload)


def run(config: RunConfig, *, resume: bool = False, plan_only: bool = False) -> dict[str, Any]:
    """Run or plan the complete research workflow."""

    bindings = _validate_source_and_crosswalk(config)
    output_root = config.output_root
    if output_root.exists() and output_root.is_symlink():
        raise FourwayRunError("output root is a symlink")
    if output_root.exists() and not output_root.is_dir():
        raise FourwayRunError("output root is not a directory")
    if not resume and output_root.exists() and any(output_root.iterdir()):
        raise FourwayRunError("output root must be empty unless --resume is set")
    output_root.mkdir(parents=True, exist_ok=True)
    config_payload = _write_or_validate_config(config, bindings, resume=resume)
    stages = build_stage_plan(config)
    if plan_only:
        return {
            "status": "plan_only",
            "config": config_payload,
            "stages": [
                {"name": stage.name, "plan_sha256": _plan_digest(stage), "jobs": [list(job.command) for job in stage.jobs]}
                for stage in stages
            ],
        }
    receipts: list[dict[str, Any]] = []
    derived_outer_start: str | None = None
    stage_index = 0
    while stage_index < len(stages):
        stage = stages[stage_index]
        receipt = _execute_stage(stage, config, resume=resume)
        receipts.append(receipt)
        if stage.name == "phase":
            phase_root = _stage_dir(config, "phase")
            phase_artifact_hash = _sha256_path(
                phase_root / "future-phase-candidate.json"
            )
            phase_receipt_file_hash = _sha256_path(phase_root / "run-receipt.json")
            stages = build_stage_plan(
                config,
                phase_artifact_sha256=phase_artifact_hash,
                phase_receipt_file_sha256=phase_receipt_file_hash,
                outer_evaluation_start=derived_outer_start,
            )
        elif stage.name == "fold_specs":
            derived_outer_start = _validate_fold_plan_cutoff(config)
            stages = build_stage_plan(
                config,
                phase_artifact_sha256=_sha256_path(
                    _stage_dir(config, "phase") / "future-phase-candidate.json"
                ),
                phase_receipt_file_sha256=_sha256_path(
                    _stage_dir(config, "phase") / "run-receipt.json"
                ),
                outer_evaluation_start=derived_outer_start,
            )
        elif stage.name == "fold_producers":
            _validate_scaling_series_bindings(config)
        stage_index += 1
    final = _write_final_receipt(config, bindings, receipts)
    return final


def _parse_args(argv: Sequence[str] | None = None) -> tuple[RunConfig, bool, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--freeze", "--source-freeze", required=True, type=Path)
    parser.add_argument("--freeze-root", type=Path)
    parser.add_argument("--crosswalk", required=True, type=Path)
    parser.add_argument("--crosswalk-receipt", required=True, type=Path)
    parser.add_argument(
        "--trust-manifest", type=Path, default=DEFAULT_TRUST_MANIFEST
    )
    parser.add_argument(
        "--crosswalk-receipt-file-sha256",
        "--independent-receipt-file-sha256",
        required=True,
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--outer-evaluation-start", required=True)
    parser.add_argument("--workers", "--cpu-workers", dest="workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--calibration-folds", type=int, default=4)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-log-bytes", type=int, default=DEFAULT_MAX_LOG_BYTES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    freeze = args.freeze.expanduser().resolve()
    freeze_root = (args.freeze_root if args.freeze_root is not None else freeze.parent).expanduser().resolve()
    config = RunConfig(
        source_root=args.source_root.expanduser().resolve(),
        source_receipt=args.source_receipt.expanduser().resolve(),
        freeze=freeze,
        freeze_root=freeze_root,
        crosswalk=args.crosswalk.expanduser().resolve(),
        crosswalk_receipt=args.crosswalk_receipt.expanduser().resolve(),
        crosswalk_receipt_file_sha256=str(args.crosswalk_receipt_file_sha256),
        trust_manifest=args.trust_manifest.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        outer_evaluation_start=str(args.outer_evaluation_start),
        workers=int(args.workers),
        folds=int(args.folds),
        calibration_folds=int(args.calibration_folds),
        draws=int(args.draws),
        seed=int(args.seed),
        max_log_bytes=int(args.max_log_bytes),
    )
    return config, bool(args.resume), bool(args.plan_only)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        config, resume, plan_only = _parse_args(argv)
        result = run(config, resume=resume, plan_only=plan_only)
    except FourwayRunError as error:
        print(f"fourway run failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_MAX_LOG_BYTES",
    "DEFAULT_WORKERS",
    "FourwayRunError",
    "Job",
    "RunConfig",
    "Stage",
    "VARIANTS",
    "build_stage_plan",
    "main",
    "run",
]
