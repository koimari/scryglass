"""Run the closed downstream stages for a completed four-way experiment.

The runner is a research receipt coordinator.  It builds evidence for every
registered rating variant.  The selected variant is a caller annotation for
later manual review.  It never gates model evaluation or downstream evidence.
Missing producer inputs create a receipt blocker.  The runner does not publish
a pack and it never changes an authority flag.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.run_future_value_fourway import VARIANTS as FOURWAY_VARIANTS
from lol_kills.research.future_value_rating import (
    FutureValueSourceError,
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-downstream-run:v1"
RUN_CONFIG_SCHEMA_VERSION = "scryglass:future-value-downstream-config:v1"
STAGE_RECEIPT_SCHEMA_VERSION = "scryglass:future-value-downstream-stage:v1"
SELECTION_SCHEMA_VERSION = "scryglass:future-value-downstream-selection:v1"
EVALUATION_RECEIPT_SCHEMA_VERSION = "scryglass:future-value-evaluation-receipt:v1"
FINAL_FIT_MANIFEST_SCHEMA_VERSION = "scryglass:future-value-final-fit-manifest:v1"
# The all-variant manifest uses the capability schema introduced by the
# snapshot producer.  Keep the name local to the runner so callers do not
# need to import the producer module just to inspect a plan.
SNAPSHOT_CAPABILITY_MANIFEST_SCHEMA_VERSION = "scryglass:future-value-snapshot-capability:v1"
SCALING_ARTIFACT_NAME = "scaling-feature-ledger-online.parquet"
SCALING_RECEIPT_NAME = "scaling-feature-ledger-online-receipt.json"
SCALING_MANIFEST_NAME = "scaling-feature-ledger-online-manifest.json"
NESTED_SELECTION_BUNDLE_NAME = "nested-selection-bundle.json"
SOURCE_FREEZE_RECEIPT_NAME = "future-value-source-receipt.json"
VARIANTS = tuple(str(value) for value in FOURWAY_VARIANTS)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
DEFAULT_MAX_LOG_BYTES = 1_048_576

AUTHORITY: dict[str, bool] = {
    "research_only": True,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_tierlist": False,
    "public_draft_score": False,
    "public_probability": False,
    "promotion": False,
    "deployment": False,
    "merge": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
}


class DownstreamRunError(RuntimeError):
    """The downstream run cannot prove a required input."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise DownstreamRunError("receipt value is not canonical JSON") from error


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise DownstreamRunError(f"file is missing or unsafe: {path}")
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DownstreamRunError(f"file cannot be read: {path}") from error
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_path(path)}


def _require_hash(value: object, label: str) -> str:
    text = str(value or "").lower()
    if SHA256_RE.fullmatch(text) is None:
        raise DownstreamRunError(f"{label} is not a SHA-256 value")
    return text


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise DownstreamRunError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DownstreamRunError(f"{label} is not valid JSON: {path}") from error
    if not isinstance(value, dict):
        raise DownstreamRunError(f"{label} must be a JSON object")
    return value


def _safe_path(value: Path | str, label: str, *, directory: bool = False) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise DownstreamRunError(f"{label} must be absolute and contain no '..'")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise DownstreamRunError(f"{label} contains a symlink")
    if directory:
        if not path.is_dir():
            raise DownstreamRunError(f"{label} is not a directory: {path}")
    elif not path.is_file():
        raise DownstreamRunError(f"{label} is not a file: {path}")
    return path


def _safe_output_root(value: Path | str, label: str) -> Path:
    """Validate a writable root before the runner creates any files."""

    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise DownstreamRunError(f"{label} must be absolute and contain no '..'")
    if any(parent.is_symlink() for parent in (path, *path.parents)):
        raise DownstreamRunError(f"{label} contains a symlink")
    if path.exists() and not path.is_dir():
        raise DownstreamRunError(f"{label} is not a directory: {path}")
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _authority_is_closed(value: object, label: str) -> None:
    if not isinstance(value, Mapping) or value.get("research_only") is not True:
        raise DownstreamRunError(f"{label} does not declare research-only authority")
    enabled = [
        str(key)
        for key, flag in value.items()
        if key != "research_only" and flag is True
    ]
    if enabled:
        raise DownstreamRunError(
            f"{label} grants authority: {', '.join(sorted(enabled))}"
        )


def _receipt_hash(path: Path, label: str) -> dict[str, Any]:
    value = _load_json(path, label)
    claimed = _require_hash(value.get("receipt_sha256"), f"{label} receipt hash")
    body = dict(value)
    body.pop("receipt_sha256", None)
    if _sha256_bytes(_canonical(body)) != claimed:
        raise DownstreamRunError(f"{label} receipt hash changed")
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise DownstreamRunError(f"output already exists: {path}")
    if path.is_symlink():
        raise DownstreamRunError(f"output is a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


@dataclass(frozen=True)
class Job:
    """One child process in a downstream stage."""

    name: str
    command: tuple[str, ...]
    output_roots: tuple[Path, ...]
    expected_files: tuple[Path, ...]
    input_paths: tuple[Path, ...] = ()
    output_dir_policy: str = "empty"

    def __post_init__(self) -> None:
        if self.output_dir_policy not in {"empty", "absent"}:
            raise DownstreamRunError(
                "job output directory policy must be 'empty' or 'absent'"
            )
        # Builder plans can contain Path values while they are assembled.  A
        # plan is also a public, canonical JSON record, so freeze every token
        # as text at the Job boundary.
        object.__setattr__(self, "command", tuple(str(token) for token in self.command))


@dataclass(frozen=True)
class Stage:
    """One sequential stage."""

    name: str
    jobs: tuple[Job, ...] = ()
    output_roots: tuple[Path, ...] = ()
    expected_files: tuple[Path, ...] = ()
    blockers: tuple[str, ...] = ()


@dataclass(frozen=True)
class RunConfig:
    """CLI and run identity for the downstream coordinator."""

    fourway_root: Path
    output_root: Path
    selected_variant: str
    repository_root: Path
    nested_selection: Path | None = None
    nested_selection_sha256: str | None = None
    baseline_cache: Path | None = None
    scaling_root: Path | None = None
    scaling_artifact: Path | None = None
    scaling_artifact_sha256: str | None = None
    scaling_receipt: Path | None = None
    scaling_receipt_sha256: str | None = None
    scaling_manifest: Path | None = None
    scaling_manifest_sha256: str | None = None
    tier_source_root: Path | None = None
    tier_repository_root: Path | None = None
    tier_trust_manifest: Path | None = None
    tier_trust_manifest_sha256: str | None = None
    tier_baseline_candidate: Path | None = None
    tier_production_manifest: Path | None = None
    tier_prospective_evaluation: Path | None = None
    tier_build_pooled_candidate: bool = False
    draft_trust_root: Path | None = None
    draft_trust_root_sha256: str | None = None
    draft_folds_root: Path | None = None
    draft_public_pack_root: Path | None = None
    draft_manifest_sha256: str | None = None
    draft_authority_path: Path | None = None
    draft_authority_sha256: str | None = None
    draft_model_artifact: Path | None = None
    draft_model_artifact_sha256: str | None = None
    draft_strict_atom: Path | None = None
    draft_strict_atom_sha256: str | None = None
    draft_strict_atom_code_sha256: str | None = None
    draft_strict_form: Path | None = None
    draft_strict_form_sha256: str | None = None
    draft_strict_form_code_sha256: str | None = None
    draft_strict_fold_root: Path | None = None
    workers: int = 1
    max_log_bytes: int = DEFAULT_MAX_LOG_BYTES

    def __post_init__(self) -> None:
        if self.selected_variant not in VARIANTS:
            raise DownstreamRunError(
                "selected variant must be one of: " + ", ".join(VARIANTS)
            )
        if self.workers < 1:
            raise DownstreamRunError("workers must be positive")
        if self.max_log_bytes < 1:
            raise DownstreamRunError("max log bytes must be positive")
        if self.nested_selection is not None and self.nested_selection_sha256 is None:
            raise DownstreamRunError("nested selection hash is required")


@dataclass(frozen=True)
class ResolvedInputs:
    """Verified paths and exact source identities from the four-way run."""

    fourway_root: Path
    source_root: Path
    source_receipt: Path
    source_receipt_file_sha256: str
    source_receipt_sha256: str
    source_as_of: str
    accepted_game_ids: tuple[str, ...]
    accepted_identity_sha256: str
    eligible_game_ids: tuple[str, ...]
    eligible_identity_sha256: str
    evaluation_paths: dict[str, Path]
    evaluation_runtime_paths: dict[str, Path]
    evaluation_stage_receipt: Path
    evaluation_receipt_paths: dict[str, Path]
    paired_uncertainty: Path
    paired_identity_sha256: str
    paired_rows: int
    freeze_root: Path
    paired_uncertainty_csv: Path | None = None
    crosswalk: Path | None = None
    crosswalk_receipt: Path | None = None
    crosswalk_receipt_file_sha256: str | None = None
    source_freeze_candidates: dict[str, Path] = field(default_factory=dict)


def _fourway_config(root: Path) -> dict[str, Any]:
    config_path = root / "run-config.json"
    value = _load_json(config_path, "fourway run config")
    claimed = _require_hash(value.get("config_sha256"), "fourway run config hash")
    unsigned = dict(value)
    unsigned.pop("config_sha256", None)
    if _sha256_bytes(_canonical(unsigned)) != claimed:
        raise DownstreamRunError("fourway run config hash changed")
    if value.get("schema_version") != "scryglass:future-value-fourway-config:v1":
        raise DownstreamRunError("fourway run config schema is unsupported")
    _authority_is_closed(value.get("authority"), "fourway run config")
    raw_variants = value.get("variants")
    if not isinstance(raw_variants, list) or tuple(str(item) for item in raw_variants) != VARIANTS:
        raise DownstreamRunError("fourway run config variant set changed")
    return value


def _stage_outputs(root: Path, stage_name: str) -> tuple[dict[str, Any], Path]:
    receipt_path = root / "receipts" / f"{stage_name}.json"
    receipt = _receipt_hash(receipt_path, f"fourway stage {stage_name}")
    if receipt.get("status") != "completed":
        raise DownstreamRunError(f"fourway stage {stage_name} is not completed")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        raise DownstreamRunError(f"fourway stage {stage_name} output records are missing")
    for raw in outputs:
        if not isinstance(raw, Mapping):
            raise DownstreamRunError(f"fourway stage {stage_name} output record is invalid")
        path = _safe_path(str(raw.get("path") or ""), f"fourway {stage_name} output")
        try:
            path.relative_to(root.resolve())
        except ValueError as error:
            raise DownstreamRunError(
                f"fourway stage {stage_name} output escapes fourway root"
            ) from error
        if int(raw.get("bytes", -1)) != path.stat().st_size or str(raw.get("sha256")) != _sha256_path(path):
            raise DownstreamRunError(f"fourway stage {stage_name} output hash changed")
    return receipt, receipt_path


def _validate_stage_source(
    receipt: Mapping[str, Any],
    *,
    source_receipt: Path,
    source_receipt_sha256: str,
    source_receipt_file_sha256: str,
    label: str,
) -> None:
    binding = receipt.get("source")
    if not isinstance(binding, Mapping):
        raise DownstreamRunError(f"{label} source binding is missing")
    if binding.get("source_receipt_path") != str(source_receipt):
        raise DownstreamRunError(f"{label} source receipt path changed")
    if binding.get("source_receipt_sha256") != source_receipt_sha256:
        raise DownstreamRunError(f"{label} source receipt hash changed")
    if binding.get("source_receipt_file_sha256") != source_receipt_file_sha256:
        raise DownstreamRunError(f"{label} source receipt file hash changed")


def _source_paths_from_receipt(
    source_root: Path,
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
) -> None:
    records = source_receipt.get("source_files")
    if not isinstance(records, Mapping):
        raise DownstreamRunError("source receipt file bindings are missing")
    expected_names = {"maps": "maps.parquet", "players": "oe_player_games.parquet", "teams": "oe_team_games.parquet"}
    source_root = Path(source_root).expanduser()
    if source_root.is_symlink() or not source_root.is_dir():
        raise DownstreamRunError(f"source root is missing or unsafe: {source_root}")
    source_root = source_root.resolve()
    for label, name in expected_names.items():
        record = records.get(label)
        if not isinstance(record, Mapping):
            raise DownstreamRunError(f"source receipt binding is missing: {label}")
        # Source locators are relative to the explicit source root.  The
        # receipt can live in a separate run directory, so its parent is not
        # a source-data root.
        locator = record.get("locator") or record.get("path")
        if not isinstance(locator, str) or not locator.strip():
            raise DownstreamRunError(f"source receipt locator is missing: {label}")
        relative = Path(locator)
        if relative.is_absolute() or ".." in relative.parts:
            raise DownstreamRunError(f"source receipt locator is unsafe: {label}")
        candidate = source_root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise DownstreamRunError(f"source {label} file does not match source root")
        path = candidate.resolve()
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise DownstreamRunError(
                f"source receipt locator escapes source root: {label}"
            ) from error
        expected_path = (source_root / name).resolve()
        if path != expected_path:
            raise DownstreamRunError(f"source {label} file does not match source root")
        if int(record.get("bytes", -1)) != path.stat().st_size or str(record.get("sha256") or "").lower() != _sha256_path(path):
            raise DownstreamRunError(f"source {label} file hash changed")


def _source_freeze_candidates(
    source_root: Path,
    source_receipt_path: Path,
    source_receipt: Mapping[str, Any],
) -> dict[str, Path]:
    """Resolve every receipt locator from one exact, portable source root.

    A four-way receipt can point at core parquet files below the declared
    source root and auxiliary files beside the receipt.  The downstream
    stage copies both families into one sealed root.  A locator must have one
    exact byte-bound candidate.  A second candidate, a symlink, or a changed
    candidate blocks the stage.
    """

    records = source_receipt.get("source_files")
    if not isinstance(records, Mapping):
        raise DownstreamRunError("source receipt file bindings are missing")
    source_root = _safe_path(source_root, "source root", directory=True)
    source_receipt_path = _safe_path(source_receipt_path, "source receipt")
    origins = (source_root, source_receipt_path.parent)
    expected_names = {
        "maps": "maps.parquet",
        "players": "oe_player_games.parquet",
        "teams": "oe_team_games.parquet",
    }
    result: dict[str, Path] = {}
    used: dict[Path, str] = {}
    for label, raw_record in sorted(records.items(), key=lambda item: str(item[0])):
        if not isinstance(label, str) or not isinstance(raw_record, Mapping):
            raise DownstreamRunError("source receipt file binding is invalid")
        locator = raw_record.get("locator") or raw_record.get("path")
        if not isinstance(locator, str) or not locator.strip():
            raise DownstreamRunError(f"source receipt locator is missing: {label}")
        relative = Path(locator)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise DownstreamRunError(f"source receipt locator is unsafe: {label}")
        if label in expected_names and relative != Path(expected_names[label]):
            raise DownstreamRunError(
                f"source {label} locator is incompatible with the portable source root"
            )
        candidates: list[Path] = []
        for origin in origins:
            candidate = origin / relative
            if any(parent.is_symlink() for parent in (candidate, *candidate.parents)):
                raise DownstreamRunError(f"source {label} candidate is a symlink")
            if candidate.is_symlink():
                raise DownstreamRunError(f"source {label} candidate is a symlink")
            if not candidate.exists():
                continue
            if not candidate.is_file():
                raise DownstreamRunError(f"source {label} candidate is not a file")
            checked = _safe_path(candidate, f"source {label} candidate")
            declared_bytes = raw_record.get("bytes")
            declared_sha = str(raw_record.get("sha256") or "").lower()
            if (
                isinstance(declared_bytes, bool)
                or not isinstance(declared_bytes, int)
                or declared_bytes < 0
                or not SHA256_RE.fullmatch(declared_sha)
            ):
                raise DownstreamRunError(f"source file binding is invalid: {label}")
            if checked.stat().st_size != declared_bytes or _sha256_path(checked) != declared_sha:
                raise DownstreamRunError(f"source {label} candidate bytes changed")
            candidates.append(checked.resolve())
        unique = sorted({str(path): path for path in candidates}.values(), key=str)
        if not unique:
            raise DownstreamRunError(f"source {label} file is missing from source freeze")
        if len(unique) != 1:
            raise DownstreamRunError(f"source {label} file has ambiguous source candidates")
        path = unique[0]
        previous = used.get(path)
        if previous is not None and previous != label:
            raise DownstreamRunError(f"source files share one candidate: {previous}, {label}")
        used[path] = label
        result[label] = path
    return result


def _source_freeze_root(config: RunConfig) -> Path:
    return _root(config, "source-freeze")


def _source_freeze_receipt(config: RunConfig) -> Path:
    return _source_freeze_root(config) / SOURCE_FREEZE_RECEIPT_NAME


def _source_freeze_expected_files(
    config: RunConfig,
    source_receipt: Mapping[str, Any],
) -> tuple[Path, ...]:
    records = source_receipt.get("source_files")
    if not isinstance(records, Mapping):
        return (_source_freeze_receipt(config),)
    paths = [_source_freeze_receipt(config)]
    for raw_record in records.values():
        if not isinstance(raw_record, Mapping):
            continue
        locator = raw_record.get("locator") or raw_record.get("path")
        if isinstance(locator, str) and locator.strip():
            paths.append(_source_freeze_root(config) / Path(locator))
    return tuple(dict.fromkeys(paths))


def _source_freeze_stage(config: RunConfig, inputs: ResolvedInputs) -> Stage:
    root = _source_freeze_root(config)
    receipt_path = _source_freeze_receipt(config)
    try:
        source_receipt = _load_json(inputs.source_receipt, "source receipt")
        expected_files = _source_freeze_expected_files(config, source_receipt)
    except DownstreamRunError:
        source_receipt = {}
        expected_files = (receipt_path,)
    blockers: list[str] = []
    candidates = dict(inputs.source_freeze_candidates)
    if not candidates:
        blockers.append("source_freeze_input_bindings_missing")
    command = _python_module(
        "benchmarks.run_future_value_downstream",
        "--source-freeze-worker",
        "--source-root", inputs.source_root,
        "--source-receipt", inputs.source_receipt,
        "--source-receipt-file-sha256", inputs.source_receipt_file_sha256,
        "--source-receipt-sha256", inputs.source_receipt_sha256,
        "--output-root", root,
    )
    input_paths = [inputs.source_receipt, *candidates.values()]
    return Stage(
        name="source_freeze",
        jobs=()
        if blockers
        else (
            Job(
                name="source_freeze",
                command=command,
                output_roots=(root,),
                expected_files=expected_files,
                input_paths=tuple(dict.fromkeys(input_paths)),
                output_dir_policy="absent",
            ),
        ),
        output_roots=(root,),
        expected_files=expected_files,
        blockers=tuple(sorted(set(blockers))),
    )


def _source_binding(
    fourway_root: Path,
    config: Mapping[str, Any],
    final: Mapping[str, Any],
) -> tuple[ResolvedInputs, dict[str, Any]]:
    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise DownstreamRunError("fourway run config inputs are missing")
    config_source = config.get("source")
    if not isinstance(config_source, Mapping):
        raise DownstreamRunError("fourway run config source binding is missing")
    source_root = _safe_path(str(inputs.get("source_root") or ""), "source root", directory=True)
    source_receipt_path = _safe_path(str(inputs.get("source_receipt") or ""), "source receipt")
    config_source_receipt = config_source.get("source_receipt")
    source_file_sha = (
        str(config_source_receipt.get("sha256") or "")
        if isinstance(config_source_receipt, Mapping)
        else ""
    )
    if not SHA256_RE.fullmatch(source_file_sha):
        source_file_sha = str(config_source.get("source_receipt_file_sha256") or "")
    source_file_sha = _require_hash(source_file_sha, "source receipt file hash")
    if _sha256_path(source_receipt_path) != source_file_sha:
        raise DownstreamRunError("source receipt file hash changed")
    source = _load_json(source_receipt_path, "source receipt")
    try:
        accepted, eligible = validate_future_value_source_receipt_payload(source)
    except FutureValueSourceError as error:
        raise DownstreamRunError(f"source receipt is not verified: {error}") from error
    _authority_is_closed(source.get("authority"), "source receipt")
    source_freeze_candidates = _source_freeze_candidates(
        source_root,
        source_receipt_path,
        source,
    )
    accepted = tuple(str(value) for value in accepted)
    eligible = tuple(str(value) for value in eligible)
    if tuple(sorted(accepted)) != accepted or tuple(sorted(eligible)) != eligible:
        raise DownstreamRunError("source receipt identities are not canonically ordered")
    if source.get("source_game_count") != len(accepted) or source.get("source_identity_sha256") != identity_sha256(accepted):
        raise DownstreamRunError("accepted source census identity changed")
    if source.get("model_eligible_game_count") != len(eligible) or source.get("model_eligible_identity_sha256") != identity_sha256(eligible):
        raise DownstreamRunError("model-eligible source census identity changed")

    config_receipt = config_source.get("source_receipt")
    if not isinstance(config_receipt, Mapping):
        raise DownstreamRunError("fourway run config source receipt binding is missing")
    if (
        config_receipt.get("path") != str(source_receipt_path)
        or int(config_receipt.get("bytes", -1)) != source_receipt_path.stat().st_size
        or str(config_receipt.get("sha256") or "").lower() != source_file_sha
    ):
        raise DownstreamRunError("fourway run config source receipt file binding changed")
    for field, expected in (
        ("source_receipt_sha256", source["receipt_sha256"]),
        ("source_game_count", len(accepted)),
        ("source_identity_sha256", identity_sha256(accepted)),
        ("model_eligible_game_count", len(eligible)),
        ("model_eligible_identity_sha256", identity_sha256(eligible)),
    ):
        if config_source.get(field) != expected:
            raise DownstreamRunError(f"fourway run config source binding changed: {field}")

    final_source = final.get("source")
    if not isinstance(final_source, Mapping):
        raise DownstreamRunError("fourway final receipt source binding is missing")
    expected_final = {
        "source_game_count": len(accepted),
        "source_identity_sha256": identity_sha256(accepted),
        "model_eligible_game_count": len(eligible),
        "model_eligible_identity_sha256": identity_sha256(eligible),
        "source_receipt_sha256": str(source.get("receipt_sha256")),
    }
    for key, expected in expected_final.items():
        if final_source.get(key) != expected:
            raise DownstreamRunError(f"fourway final source binding changed: {key}")

    evaluation_stage, evaluation_stage_receipt = _stage_outputs(fourway_root, "evaluations")
    uncertainty_stage, _ = _stage_outputs(fourway_root, "paired_uncertainty")
    _validate_stage_source(
        evaluation_stage,
        source_receipt=source_receipt_path,
        source_receipt_sha256=str(source["receipt_sha256"]),
        source_receipt_file_sha256=source_file_sha,
        label="fourway evaluations stage",
    )
    _validate_stage_source(
        uncertainty_stage,
        source_receipt=source_receipt_path,
        source_receipt_sha256=str(source["receipt_sha256"]),
        source_receipt_file_sha256=source_file_sha,
        label="fourway paired uncertainty stage",
    )
    eval_paths: dict[str, Path] = {}
    runtime_paths: dict[str, Path] = {}
    raw_eval_outputs = evaluation_stage.get("outputs", [])
    eval_outputs = {str(record.get("path")): record for record in raw_eval_outputs if isinstance(record, Mapping)}
    if len(eval_outputs) != sum(1 for record in raw_eval_outputs if isinstance(record, Mapping)):
        raise DownstreamRunError("fourway evaluation output records contain duplicate paths")
    for variant in VARIANTS:
        # The fourway receipt keeps the historical stage name ``evaluations``.
        # Its verified artifacts live in the singular ``evaluation`` root.
        path = fourway_root / "stages" / "evaluation" / variant / "model.json"
        runtime = fourway_root / "stages" / "evaluation" / variant / "runtime.json"
        if not path.is_file() or path.is_symlink() or not runtime.is_file() or runtime.is_symlink():
            raise DownstreamRunError(f"fourway evaluation outputs are incomplete: {variant}")
        for candidate in (path, runtime):
            record = eval_outputs.get(str(candidate))
            if record is None or int(record.get("bytes", -1)) != candidate.stat().st_size or str(record.get("sha256") or "").lower() != _sha256_path(candidate):
                raise DownstreamRunError(f"fourway evaluation output hash changed: {variant}")
        model = _load_json(path, f"fourway {variant} evaluation")
        runtime_value = _load_json(runtime, f"fourway {variant} runtime receipt")
        if runtime_value.get("schema_version") != "scryglass:future-value-model-runtime:v1":
            raise DownstreamRunError(f"fourway {variant} runtime schema changed")
        runtime_claimed = _require_hash(runtime_value.get("receipt_sha256"), f"fourway {variant} runtime receipt hash")
        runtime_body = dict(runtime_value)
        runtime_body.pop("receipt_sha256", None)
        if _sha256_bytes(_canonical(runtime_body)) != runtime_claimed:
            raise DownstreamRunError(f"fourway {variant} runtime receipt hash changed")
        _authority_is_closed(runtime_value.get("authority"), f"fourway {variant} runtime receipt")
        _validate_stage_source(
            {"source": runtime_value.get("source")},
            source_receipt=source_receipt_path,
            source_receipt_sha256=str(source["receipt_sha256"]),
            source_receipt_file_sha256=source_file_sha,
            label=f"fourway {variant} runtime receipt",
        )
        if model.get("schema_version") != "scryglass:future-value-four-variant-evaluation:v1":
            raise DownstreamRunError(f"fourway {variant} evaluation schema changed")
        model_source = model.get("source")
        if not isinstance(model_source, Mapping):
            raise DownstreamRunError(f"fourway {variant} evaluation source is missing")
        for key, expected in {
            "source_as_of": source.get("source_as_of"),
            "source_game_count": source.get("source_game_count"),
            "source_identity_sha256": source.get("source_identity_sha256"),
            "source_receipt_sha256": source.get("receipt_sha256"),
        }.items():
            if model_source.get(key) != expected:
                raise DownstreamRunError(f"fourway {variant} evaluation source changed: {key}")
        _authority_is_closed(model.get("authority"), f"fourway {variant} evaluation")
        payload = model.get("variants", {}).get(variant) if isinstance(model.get("variants"), Mapping) else None
        if not isinstance(payload, Mapping):
            raise DownstreamRunError(f"fourway {variant} evaluation payload is missing")
        _authority_is_closed(payload.get("authority"), f"fourway {variant} evaluation payload")
        payload_source = payload.get("source")
        if not isinstance(payload_source, Mapping):
            raise DownstreamRunError(f"fourway {variant} evaluation payload source is missing")
        for key, expected in {
            "source_as_of": source.get("source_as_of"),
            "source_game_count": source.get("source_game_count"),
            "source_identity_sha256": source.get("source_identity_sha256"),
            "source_receipt_sha256": source.get("receipt_sha256"),
            "accepted_game_ids": list(accepted),
        }.items():
            if payload_source.get(key) != expected:
                raise DownstreamRunError(f"fourway {variant} evaluation payload source changed: {key}")
        ledger = payload.get("prediction_ledger")
        if not isinstance(ledger, Mapping) or not isinstance(ledger.get("rows"), list):
            raise DownstreamRunError(f"fourway {variant} prediction ledger is missing")
        ids = tuple(str(row.get("game_id") or "") for row in ledger["rows"] if isinstance(row, Mapping))
        if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
            raise DownstreamRunError(f"fourway {variant} prediction ledger identities are invalid")
        if ledger.get("row_count") != len(ids):
            raise DownstreamRunError(f"fourway {variant} prediction ledger count changed")
        if ledger.get("game_identity_sha256") != identity_sha256(ids):
            raise DownstreamRunError(f"fourway {variant} prediction ledger identity changed")
        if not set(ids).issubset(set(eligible)):
            raise DownstreamRunError(
                f"fourway {variant} prediction ledger contains IDs outside eligible census"
            )
        fold_ids: list[set[str]] = []
        folds = payload.get("folds")
        if not isinstance(folds, list) or not folds:
            raise DownstreamRunError(f"fourway {variant} evaluation folds are missing")
        for fold in folds:
            if not isinstance(fold, Mapping):
                raise DownstreamRunError(f"fourway {variant} evaluation fold is invalid")
            raw_fold_ids = fold.get("paired_game_ids")
            if not isinstance(raw_fold_ids, list) or not raw_fold_ids:
                raise DownstreamRunError(f"fourway {variant} evaluation fold IDs are missing")
            fold_values = [str(value) for value in raw_fold_ids]
            if any(not value for value in fold_values) or len(set(fold_values)) != len(fold_values):
                raise DownstreamRunError(f"fourway {variant} evaluation fold IDs are invalid")
            if fold.get("paired_game_id_count") != len(fold_values):
                raise DownstreamRunError(f"fourway {variant} evaluation fold count changed")
            if fold.get("paired_game_identity_sha256") not in {None, identity_sha256(fold_values)}:
                raise DownstreamRunError(f"fourway {variant} evaluation fold identity changed")
            if not set(fold_values).issubset(set(eligible)):
                raise DownstreamRunError(f"fourway {variant} evaluation fold exceeds eligible census")
            fold_ids.append(set(fold_values))
        if any(left & right for index, left in enumerate(fold_ids) for right in fold_ids[index + 1:]):
            raise DownstreamRunError(f"fourway {variant} evaluation folds overlap")
        if set().union(*fold_ids) != set(ids):
            raise DownstreamRunError(
                f"fourway {variant} prediction ledger differs from evaluation folds"
            )
        runtime_output = runtime_value.get("output")
        if not isinstance(runtime_output, Mapping):
            raise DownstreamRunError(f"fourway {variant} runtime ledger binding is missing")
        runtime_rows = runtime_output.get("prediction_ledger_rows")
        runtime_hashes = runtime_output.get("prediction_ledger_sha256")
        if not isinstance(runtime_rows, Mapping) or runtime_rows.get(variant) != len(ids):
            raise DownstreamRunError(f"fourway {variant} runtime ledger count changed")
        if not isinstance(runtime_hashes, Mapping) or runtime_hashes.get(variant) != ledger.get("sha256"):
            raise DownstreamRunError(f"fourway {variant} runtime ledger hash changed")
        eval_paths[variant] = path
        runtime_paths[variant] = runtime

    uncertainty_path = fourway_root / "stages" / "paired-uncertainty" / "paired-uncertainty.json"
    uncertainty_csv = fourway_root / "stages" / "paired-uncertainty" / "paired-uncertainty.csv"
    uncertainty_outputs = {str(record.get("path")): record for record in uncertainty_stage.get("outputs", []) if isinstance(record, Mapping)}
    record = uncertainty_outputs.get(str(uncertainty_path))
    if record is None or int(record.get("bytes", -1)) != uncertainty_path.stat().st_size or str(record.get("sha256") or "").lower() != _sha256_path(uncertainty_path):
        raise DownstreamRunError("paired uncertainty output hash changed")
    uncertainty_csv_path: Path | None = None
    csv_record = uncertainty_outputs.get(str(uncertainty_csv))
    if csv_record is None or not uncertainty_csv.is_file() or uncertainty_csv.is_symlink():
        raise DownstreamRunError("paired uncertainty CSV output is missing")
    if int(csv_record.get("bytes", -1)) != uncertainty_csv.stat().st_size or str(csv_record.get("sha256") or "").lower() != _sha256_path(uncertainty_csv):
        raise DownstreamRunError("paired uncertainty CSV output hash changed")
    uncertainty_csv_path = uncertainty_csv
    uncertainty = _load_json(uncertainty_path, "paired uncertainty")
    if uncertainty.get("schema_version") != "scryglass:future-value-paired-uncertainty:v1":
        raise DownstreamRunError("paired uncertainty schema changed")
    _authority_is_closed(uncertainty.get("authority"), "paired uncertainty")
    uncertainty_source = uncertainty.get("source")
    if not isinstance(uncertainty_source, Mapping):
        raise DownstreamRunError("paired uncertainty source binding is missing")
    for key, expected in expected_final.items():
        if uncertainty_source.get(key) != expected:
            raise DownstreamRunError(f"paired uncertainty source changed: {key}")
    coverage = uncertainty.get("coverage")
    if not isinstance(coverage, Mapping):
        raise DownstreamRunError("paired uncertainty coverage is missing")
    common_ids: tuple[str, ...] = ()
    for variant, path in eval_paths.items():
        model = _load_json(path, f"fourway {variant} evaluation")
        payload = model["variants"][variant]
        ids = tuple(str(row["game_id"]) for row in payload["prediction_ledger"]["rows"])
        if not common_ids:
            common_ids = ids
        elif ids != common_ids:
            raise DownstreamRunError("fourway evaluations do not share one paired identity universe")
    if coverage.get("game_identity_sha256") != identity_sha256(common_ids) or coverage.get("rows") != len(common_ids):
        raise DownstreamRunError("paired uncertainty identity changed")

    # The four-way run stores a freeze root and optional crosswalk in config.
    freeze_root = Path(str(inputs.get("freeze_root") or source_root)).expanduser().resolve()
    crosswalk = Path(str(inputs.get("crosswalk"))).expanduser().resolve() if inputs.get("crosswalk") else None
    crosswalk_receipt = Path(str(inputs.get("crosswalk_receipt"))).expanduser().resolve() if inputs.get("crosswalk_receipt") else None
    crosswalk_hash = str(inputs.get("crosswalk_receipt_file_sha256") or "") or None
    return (
        ResolvedInputs(
            fourway_root=fourway_root,
            source_root=source_root,
            source_receipt=source_receipt_path,
            source_receipt_file_sha256=source_file_sha,
            source_receipt_sha256=str(source["receipt_sha256"]),
            source_as_of=str(source["source_as_of"]),
            accepted_game_ids=accepted,
            accepted_identity_sha256=str(source["source_identity_sha256"]),
            eligible_game_ids=eligible,
            eligible_identity_sha256=str(source["model_eligible_identity_sha256"]),
            evaluation_paths=eval_paths,
            evaluation_runtime_paths=runtime_paths,
            evaluation_stage_receipt=evaluation_stage_receipt,
            evaluation_receipt_paths={},
            paired_uncertainty=uncertainty_path,
            paired_uncertainty_csv=uncertainty_csv_path,
            paired_identity_sha256=str(coverage["game_identity_sha256"]),
            paired_rows=int(coverage["rows"]),
            freeze_root=freeze_root,
            crosswalk=crosswalk,
            crosswalk_receipt=crosswalk_receipt,
            crosswalk_receipt_file_sha256=crosswalk_hash,
            source_freeze_candidates=source_freeze_candidates,
        ),
        source,
    )


def _validate_optional_inputs(config: RunConfig, inputs: ResolvedInputs | None = None) -> None:
    """Check caller-provided trust bytes before any child process starts."""

    hashed = (
        (config.tier_trust_manifest, config.tier_trust_manifest_sha256, "Tier trust manifest"),
        (config.draft_trust_root, config.draft_trust_root_sha256, "Draft Score trust root"),
        (config.draft_authority_path, config.draft_authority_sha256, "Draft Score authority"),
        (config.draft_model_artifact, config.draft_model_artifact_sha256, "Draft Score model artifact"),
        (config.nested_selection, config.nested_selection_sha256, "nested selection evidence"),
        (config.scaling_artifact, config.scaling_artifact_sha256, "scaling feature artifact"),
        (config.scaling_receipt, config.scaling_receipt_sha256, "scaling feature receipt"),
        (config.scaling_manifest, config.scaling_manifest_sha256, "scaling feature manifest"),
        (config.draft_strict_atom, config.draft_strict_atom_sha256, "Draft Score strict atom"),
        (config.draft_strict_form, config.draft_strict_form_sha256, "Draft Score strict form"),
    )
    for path, expected, label in hashed:
        if path is None:
            continue
        actual_path = _safe_path(path, label)
        if expected is None:
            raise DownstreamRunError(f"{label} hash is required")
        if _sha256_path(actual_path) != _require_hash(expected, f"{label} file hash"):
            raise DownstreamRunError(f"{label} file hash changed")
    for path, label in (
        (config.tier_source_root, "Tier source root"),
        (config.tier_repository_root, "Tier repository root"),
        (config.tier_baseline_candidate, "Tier baseline candidate"),
        (config.tier_production_manifest, "Tier production manifest"),
        (config.tier_prospective_evaluation, "Tier prospective evaluation"),
        (config.draft_folds_root, "Draft Score folds root"),
        (config.draft_public_pack_root, "Draft Score public-pack root"),
        (config.draft_strict_fold_root, "Draft Score strict fold root"),
        (config.baseline_cache, "baseline cache"),
        (config.scaling_root, "scaling feature root"),
    ):
        if path is None:
            continue
        if label.endswith("root"):
            _safe_path(path, label, directory=True)
        else:
            _safe_path(path, label)
    if config.draft_public_pack_root is not None:
        manifest = _safe_path(config.draft_public_pack_root / "manifest.json", "Draft Score public manifest")
        expected = _require_hash(config.draft_manifest_sha256, "Draft Score public manifest hash")
        if _sha256_path(manifest) != expected:
            raise DownstreamRunError("Draft Score public manifest file hash changed")
    if config.draft_manifest_sha256 is not None and config.draft_public_pack_root is None:
        raise DownstreamRunError("Draft Score public manifest root is required")
    if config.scaling_artifact_sha256 is not None and config.scaling_artifact is None:
        raise DownstreamRunError("scaling feature artifact is required for its hash")
    if config.scaling_receipt_sha256 is not None and config.scaling_receipt is None:
        raise DownstreamRunError("scaling feature receipt is required for its hash")
    if config.scaling_manifest_sha256 is not None and config.scaling_manifest is None:
        raise DownstreamRunError("scaling feature manifest is required for its hash")


def _python_module(module: str, *args: object) -> tuple[str, ...]:
    return (sys.executable, "-m", module, *(str(arg) for arg in args))


def _digest_or_placeholder(path: Path, label: str) -> str:
    """Return a child hash argument when the dependency already exists."""

    if path.is_file() and not path.is_symlink():
        return _sha256_path(path)
    return f"__{label.upper().replace('-', '_')}_SHA256__"


def _nested_selection_blockers(
    path: Path,
    inputs: ResolvedInputs,
) -> tuple[str, ...]:
    """Check the small source binding that the final-fit CLI consumes.

    The nested evidence lives inside a model artifact in the existing
    evaluation layout.  The final-fit builder performs the detailed fold
    checks.  This preflight keeps the source and variant binding visible in
    the downstream plan.
    """

    try:
        value = _load_json(path, "nested selection evidence")
    except DownstreamRunError:
        return ("nested_selection_input_invalid",)
    claimed_artifact_hash = value.get("artifact_sha256")
    if claimed_artifact_hash is not None:
        try:
            claimed = _require_hash(claimed_artifact_hash, "nested selection artifact hash")
            unsigned = dict(value)
            unsigned.pop("artifact_sha256", None)
            if _sha256_bytes(_canonical(unsigned)) != claimed:
                return ("nested_selection_artifact_hash_changed",)
        except DownstreamRunError:
            return ("nested_selection_artifact_hash_invalid",)
    try:
        _authority_is_closed(value.get("authority"), "nested selection evidence")
    except DownstreamRunError:
        return ("nested_selection_authority_invalid",)
    raw_blockers = value.get("blockers")
    if isinstance(raw_blockers, list) and any(str(item).strip() for item in raw_blockers):
        return tuple(sorted({f"nested_selection_semantic_blocker:{item}" for item in raw_blockers}))
    if value.get("status") in {"blocked", "invalid", "failed", "error"}:
        return ("nested_selection_semantic_status",)
    source = value.get("source")
    if not isinstance(source, Mapping):
        return ("nested_selection_source_binding_missing",)
    blockers: list[str] = []
    for field, expected in (
        ("source_as_of", inputs.source_as_of),
        ("source_game_count", len(inputs.accepted_game_ids)),
        ("source_identity_sha256", inputs.accepted_identity_sha256),
        ("source_receipt_sha256", inputs.source_receipt_sha256),
    ):
        if source.get(field) != expected:
            blockers.append(f"nested_selection_{field}_mismatch")
    variants = value.get("variants")
    if isinstance(variants, Mapping):
        for variant in VARIANTS:
            payload = variants.get(variant)
            if not isinstance(payload, Mapping) or not isinstance(payload.get("folds"), list):
                blockers.append(f"nested_selection_{variant}_missing")
    elif value.get("variant") in VARIANTS and isinstance(value.get("folds"), list):
        # A single-variant nested receipt is valid only for its own final fit.
        # The all-variant runner keeps the receipt visible as a blocker for the
        # other three jobs instead of silently reusing V2 evidence.
        for variant in VARIANTS:
            if variant != value.get("variant"):
                blockers.append(f"nested_selection_{variant}_missing")
    else:
        blockers.append("nested_selection_variants_missing")
    return tuple(sorted(set(blockers)))


def _root(config: RunConfig, name: str) -> Path:
    return config.output_root / "stages" / name


def _selection_stage(config: RunConfig, inputs: ResolvedInputs, source: Mapping[str, Any]) -> Stage:
    output = _root(config, "selection")
    return Stage(
        name="selection",
        jobs=(),
        output_roots=(output,),
        expected_files=(output / "selected-variant.json",),
    )


def _snapshot_comparison_worker(argv: Sequence[str]) -> int:
    """Build a comparison from the current run's accepted-census artifacts."""

    parser = argparse.ArgumentParser(description="Build a source-bound snapshot comparison")
    parser.add_argument("--snapshot-comparison-worker", action="store_true")
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--current-receipt", required=True, type=Path)
    parser.add_argument("--future-receipt", required=True, type=Path)
    parser.add_argument("--player-rank-diffs", required=True, type=Path)
    parser.add_argument("--team-rank-diffs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(list(argv))
    try:
        from benchmarks.build_future_value_snapshot_comparison import (
            _verify_current_snapshot_trust_root,
        )
        from lol_kills.research.future_value_snapshot_comparison import (
            build_snapshot_comparison_report,
        )

        source_receipt_path = args.source_receipt.resolve()
        source_receipt = _load_json(source_receipt_path, "source receipt")
        validate_future_value_source_receipt_payload(source_receipt)
        _authority_is_closed(source_receipt.get("authority"), "snapshot comparison source receipt")
        if _sha256_path(source_receipt_path) != str(args.source_receipt_file_sha256).lower():
            raise DownstreamRunError("snapshot comparison source receipt file hash changed")
        current_receipt = _load_json(args.current_receipt.resolve(), "current snapshot receipt")
        future_receipt = _load_json(args.future_receipt.resolve(), "future snapshot receipt")
        player = _load_json(args.player_rank_diffs.resolve(), "player rank diffs")
        team = _load_json(args.team_rank_diffs.resolve(), "team rank diffs")
        future_source = future_receipt.get("source")
        if not isinstance(future_source, Mapping):
            raise DownstreamRunError("future snapshot source binding is missing")
        for field in ("source_receipt_sha256", "source_identity_sha256", "source_game_count"):
            if future_source.get(field) != source_receipt.get(
                "receipt_sha256" if field == "source_receipt_sha256" else field
            ):
                raise DownstreamRunError(
                    f"snapshot comparison future source binding changed: {field}"
                )
        trust_root = _verify_current_snapshot_trust_root(
            current_receipt_path=args.current_receipt.resolve(),
            current_receipt=current_receipt,
            current_receipt_file_sha256=_sha256_path(args.current_receipt.resolve()),
            source=dict(future_source),
        )
        report = build_snapshot_comparison_report(
            current_receipt=current_receipt,
            future_receipt=future_receipt,
            player_rank_diff_artifact=player,
            team_rank_diff_artifact=team,
            current_receipt_file_sha256=_sha256_path(args.current_receipt.resolve()),
            future_receipt_file_sha256=_sha256_path(args.future_receipt.resolve()),
            player_rank_diff_file_sha256=_sha256_path(args.player_rank_diffs.resolve()),
            team_rank_diff_file_sha256=_sha256_path(args.team_rank_diffs.resolve()),
            expected_source_receipt_sha256=str(future_source.get("source_receipt_sha256") or ""),
            current_snapshot_trust_root=trust_root,
        )
        _write_json(args.output.resolve(), report)
        print(json.dumps({"status": report.get("status"), "blockers": report.get("blockers", [])}, sort_keys=True))
        return 0
    except (DownstreamRunError, OSError, ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        print(f"snapshot comparison worker failed: {error}", file=sys.stderr)
        return 1


def _variant_directory_arguments(values: Sequence[str], label: str) -> dict[str, Path]:
    """Parse the closed ``variant=directory`` worker argument set."""

    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise DownstreamRunError(f"{label} must use VARIANT=PATH")
        variant, value = raw.split("=", 1)
        variant = variant.strip()
        if variant not in VARIANTS or not value.strip() or variant in result:
            raise DownstreamRunError(f"{label} has an invalid variant binding")
        result[variant] = Path(value).expanduser().resolve()
    if set(result) != set(VARIANTS):
        raise DownstreamRunError(
            f"{label} must cover: {', '.join(VARIANTS)}"
        )
    return result


def _final_fit_manifest_worker(argv: Sequence[str]) -> int:
    """Seal four final-fit directories into one source-bound manifest."""

    parser = argparse.ArgumentParser(description="Build an all-variant final-fit manifest")
    parser.add_argument("--final-fit-manifest-worker", action="store_true")
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--variant-output", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(list(argv))
    try:
        source_path = args.source_receipt.resolve()
        source = _load_json(source_path, "source receipt")
        validate_future_value_source_receipt_payload(source)
        _authority_is_closed(source.get("authority"), "final-fit manifest source")
        expected_source_file = _require_hash(
            args.source_receipt_file_sha256, "source receipt file hash"
        )
        if _sha256_path(source_path) != expected_source_file:
            raise DownstreamRunError("final-fit manifest source receipt file changed")
        directories = _variant_directory_arguments(args.variant_output, "--variant-output")
        variants: dict[str, Any] = {}
        manifest_blockers: list[str] = []
        for ordinal, variant in enumerate(VARIANTS, start=1):
            root = _safe_path(directories[variant], f"{variant} final-fit root", directory=True)
            model_path = _safe_path(root / f"final-v{ordinal}-model.json", f"{variant} final-fit model")
            receipt_path = _safe_path(root / f"final-v{ordinal}-model-receipt.json", f"{variant} final-fit receipt")
            run_path = _safe_path(root / "final-fit-run.json", f"{variant} final-fit run")
            model = _load_json(model_path, f"{variant} final-fit model")
            receipt = _receipt_hash(receipt_path, f"{variant} final-fit receipt")
            run = _load_json(run_path, f"{variant} final-fit run")
            if model.get("schema_version") != "scryglass:future-value-final-fit:v1":
                raise DownstreamRunError(f"{variant} final-fit model schema changed")
            if model.get("variant") != variant or receipt.get("variant") != variant or run.get("variant") != variant:
                raise DownstreamRunError(f"{variant} final-fit variant binding changed")
            for label, value in (("model", model), ("receipt", receipt), ("run", run)):
                binding = value.get("source") or value.get("source_binding")
                if isinstance(binding, Mapping):
                    for field, expected in (
                        ("source_as_of", source.get("source_as_of")),
                        ("source_game_count", source.get("source_game_count")),
                        ("source_identity_sha256", source.get("source_identity_sha256")),
                        ("source_receipt_sha256", source.get("receipt_sha256")),
                    ):
                        if field in binding and binding.get(field) != expected:
                            raise DownstreamRunError(f"{variant} final-fit {label} source changed: {field}")
            if run.get("model_artifact_sha256") not in {None, _sha256_path(model_path)}:
                raise DownstreamRunError(f"{variant} final-fit model artifact hash changed")
            variant_blockers = (
                list(receipt.get("blockers", []))
                if isinstance(receipt.get("blockers"), list)
                else []
            )
            manifest_blockers.extend(
                f"{variant}:{value}" for value in variant_blockers if str(value).strip()
            )
            variants[variant] = {
                "status": receipt.get("status"),
                "blockers": variant_blockers,
                "model": _file_record(model_path),
                "receipt": _file_record(receipt_path),
                "run": _file_record(run_path),
                "model_receipt_sha256": receipt.get("receipt_sha256"),
                "fit_game_count": run.get("fit_game_count"),
                "fit_game_identity_sha256": run.get("fit_game_identity_sha256"),
            }
        payload: dict[str, Any] = {
            "schema_version": FINAL_FIT_MANIFEST_SCHEMA_VERSION,
            "status": "blocked" if manifest_blockers else "research_only",
            "authority": dict(AUTHORITY),
            "source": {
                "source_as_of": source.get("source_as_of"),
                "source_game_count": source.get("source_game_count"),
                "source_identity_sha256": source.get("source_identity_sha256"),
                "source_receipt_sha256": source.get("receipt_sha256"),
                "source_receipt_file_sha256": expected_source_file,
                "model_eligible_game_count": source.get("model_eligible_game_count"),
                "model_eligible_identity_sha256": source.get("model_eligible_identity_sha256"),
                "accepted_game_ids": list(source.get("accepted_game_ids", [])),
                "model_eligible_game_ids": list(source.get("model_eligible_game_ids", [])),
            },
            "variants": variants,
            "blockers": sorted(set(manifest_blockers)),
        }
        payload["manifest_sha256"] = _sha256_bytes(_canonical(payload))
        _write_json(args.output.resolve(), payload)
        print(json.dumps({"status": payload["status"], "variants": list(VARIANTS)}, sort_keys=True))
        return 0
    except (DownstreamRunError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"final-fit manifest worker failed: {error}", file=sys.stderr)
        return 1


def _variant_file_arguments(values: Sequence[str], label: str) -> dict[str, Path]:
    """Parse the closed ``variant=file`` worker argument set."""

    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise DownstreamRunError(f"{label} must use VARIANT=PATH")
        variant, value = raw.split("=", 1)
        variant = variant.strip()
        if variant not in VARIANTS or not value.strip() or variant in result:
            raise DownstreamRunError(f"{label} has an invalid variant binding")
        result[variant] = Path(value).expanduser()
    if set(result) != set(VARIANTS):
        raise DownstreamRunError(f"{label} must cover: {', '.join(VARIANTS)}")
    return result


def _source_freeze_worker(argv: Sequence[str]) -> int:
    """Copy the exact source receipt files into one portable freeze root."""

    parser = argparse.ArgumentParser(description="Stage a portable source freeze")
    parser.add_argument("--source-freeze-worker", action="store_true")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(list(argv))
    try:
        source_root = _safe_path(args.source_root, "source freeze source root", directory=True)
        source_receipt_path = _safe_path(args.source_receipt, "source freeze source receipt")
        output_root = _safe_output_root(args.output_root, "source freeze output root")
        expected_file_hash = _require_hash(
            args.source_receipt_file_sha256,
            "source freeze source receipt file hash",
        )
        if _sha256_path(source_receipt_path) != expected_file_hash:
            raise DownstreamRunError("source freeze source receipt file hash changed")
        source = _load_json(source_receipt_path, "source freeze source receipt")
        try:
            validate_future_value_source_receipt_payload(
                source,
                expected_receipt_sha256=_require_hash(
                    args.source_receipt_sha256,
                    "source freeze source receipt hash",
                ),
            )
        except FutureValueSourceError as error:
            raise DownstreamRunError("source freeze source receipt is not verified") from error
        _authority_is_closed(source.get("authority"), "source freeze source receipt")
        candidates = _source_freeze_candidates(source_root, source_receipt_path, source)
        records = source.get("source_files")
        if not isinstance(records, Mapping):
            raise DownstreamRunError("source freeze source file bindings are missing")
        for label, candidate in candidates.items():
            record = records.get(label)
            if not isinstance(record, Mapping):
                raise DownstreamRunError(f"source freeze source file binding is missing: {label}")
            locator = record.get("locator") or record.get("path")
            if not isinstance(locator, str) or not locator.strip():
                raise DownstreamRunError(f"source freeze source locator is missing: {label}")
            relative = Path(locator)
            target = output_root / relative
            if target == output_root / SOURCE_FREEZE_RECEIPT_NAME:
                raise DownstreamRunError("source freeze locator collides with copied receipt")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() or target.is_symlink():
                raise DownstreamRunError(f"source freeze target already exists: {relative}")
            shutil.copyfile(candidate, target)
            if _file_record(target)["bytes"] != int(record.get("bytes", -1)) or _file_record(target)["sha256"] != str(record.get("sha256") or "").lower():
                raise DownstreamRunError(f"source freeze copied bytes changed: {label}")
        copied_receipt = output_root / SOURCE_FREEZE_RECEIPT_NAME
        shutil.copyfile(source_receipt_path, copied_receipt)
        if _sha256_path(copied_receipt) != expected_file_hash or copied_receipt.stat().st_size != source_receipt_path.stat().st_size:
            raise DownstreamRunError("source freeze copied receipt changed")
        print(
            json.dumps(
                {
                    "status": "research_only",
                    "source_files": len(candidates),
                    "source_receipt_file_sha256": expected_file_hash,
                },
                sort_keys=True,
            )
        )
        return 0
    except (DownstreamRunError, FutureValueSourceError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"source freeze worker failed: {error}", file=sys.stderr)
        return 1


def _scaling_online_worker(argv: Sequence[str]) -> int:
    """Build the source-bound online full-census scaling feature ledger."""

    parser = argparse.ArgumentParser(description="Build an online scaling feature ledger")
    parser.add_argument("--scaling-online-worker", action="store_true")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(list(argv))
    try:
        import pandas as pd

        from lol_kills.research.atomized_rf_composite import build_scaling_feature_ledger

        source_root = _safe_path(args.source_root, "scaling source root", directory=True)
        source_path = _safe_path(args.source_receipt, "scaling source receipt")
        output_root = _safe_output_root(args.output_root, "scaling output root")
        source = _load_json(source_path, "scaling source receipt")
        try:
            validate_future_value_source_receipt_payload(source)
        except FutureValueSourceError as error:
            raise DownstreamRunError("scaling source receipt is not verified") from error
        _authority_is_closed(source.get("authority"), "scaling source receipt")
        expected_file_hash = _require_hash(
            args.source_receipt_file_sha256, "scaling source receipt file hash"
        )
        if _sha256_path(source_path) != expected_file_hash:
            raise DownstreamRunError("scaling source receipt file hash changed")
        expected_receipt_hash = _require_hash(
            args.source_receipt_sha256, "scaling source receipt hash"
        )
        if str(source.get("receipt_sha256") or "").lower() != expected_receipt_hash:
            raise DownstreamRunError("scaling source receipt hash changed")
        _source_paths_from_receipt(source_root, source_path, source)
        maps = pd.read_parquet(source_root / "maps.parquet")
        players = pd.read_parquet(source_root / "oe_player_games.parquet")
        teams = pd.read_parquet(source_root / "oe_team_games.parquet")
        ledger, receipt = build_scaling_feature_ledger(
            maps,
            players,
            teams,
            source_receipt=source,
            source_receipt_sha256=expected_receipt_hash,
            model_eligible_only=True,
        )
        output_root.mkdir(parents=True, exist_ok=True)
        artifact_path = output_root / SCALING_ARTIFACT_NAME
        ledger.to_parquet(artifact_path, index=False)
        receipt_path = output_root / SCALING_RECEIPT_NAME
        _write_json(receipt_path, receipt)
        artifact_hash = _sha256_path(artifact_path)
        receipt_hash = _sha256_path(receipt_path)
        manifest: dict[str, Any] = {
            "schema_version": "scryglass:scaling-ledger-artifact:v1",
            "status": "research_only",
            "authority": dict(AUTHORITY),
            "artifact_path": str(artifact_path),
            "artifact_bytes": artifact_path.stat().st_size,
            "artifact_sha256": artifact_hash,
            "producer_receipt_path": str(receipt_path),
            "producer_receipt_file_sha256": receipt_hash,
            "producer_receipt_sha256": receipt.get("receipt_sha256"),
            "source": {
                "source_as_of": source.get("source_as_of"),
                "source_game_count": source.get("source_game_count"),
                "source_identity_sha256": source.get("source_identity_sha256"),
                "source_receipt_sha256": source.get("receipt_sha256"),
                "source_receipt_file_sha256": expected_file_hash,
            },
            "rows": len(ledger),
        }
        manifest["receipt_sha256"] = _sha256_bytes(_canonical(manifest))
        _write_json(output_root / SCALING_MANIFEST_NAME, manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "rows": len(ledger),
                    "artifact_sha256": artifact_hash,
                    "receipt_sha256": receipt_hash,
                },
                sort_keys=True,
            )
        )
        return 0
    except (DownstreamRunError, FutureValueSourceError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"scaling online worker failed: {error}", file=sys.stderr)
        return 1


def _nested_selection_bundle_worker(argv: Sequence[str]) -> int:
    """Seal nested-selection evidence from all verified evaluation models."""

    parser = argparse.ArgumentParser(description="Build an all-variant nested-selection bundle")
    parser.add_argument("--nested-selection-bundle-worker", action="store_true")
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--evaluation", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(list(argv))
    try:
        source_path = _safe_path(args.source_receipt, "nested selection source receipt")
        source = _load_json(source_path, "nested selection source receipt")
        try:
            validate_future_value_source_receipt_payload(source)
        except FutureValueSourceError as error:
            raise DownstreamRunError("nested selection source receipt is not verified") from error
        _authority_is_closed(source.get("authority"), "nested selection source receipt")
        expected_source_file_hash = _require_hash(
            args.source_receipt_file_sha256,
            "nested selection source receipt file hash",
        )
        if _sha256_path(source_path) != expected_source_file_hash:
            raise DownstreamRunError("nested selection source receipt file hash changed")
        evaluation_paths = _variant_file_arguments(args.evaluation, "--evaluation")
        records: dict[str, dict[str, Any]] = {}
        variant_payloads: dict[str, dict[str, Any]] = {}
        top_source: Mapping[str, Any] | None = None
        source_contract = {
            "source_as_of": source.get("source_as_of"),
            "source_game_count": source.get("source_game_count"),
            "source_identity_sha256": source.get("source_identity_sha256"),
            "source_receipt_sha256": source.get("receipt_sha256"),
            "model_eligible_game_count": source.get("model_eligible_game_count"),
            "model_eligible_identity_sha256": source.get("model_eligible_identity_sha256"),
            "accepted_game_ids": list(source.get("accepted_game_ids", [])),
            "model_eligible_game_ids": list(source.get("model_eligible_game_ids", [])),
        }

        def verify_artifact_records(value: Any, label: str) -> int:
            if isinstance(value, Mapping):
                if set(value) == {"path", "bytes", "sha256"}:
                    artifact_path = _safe_path(str(value.get("path") or ""), label)
                    if (
                        int(value.get("bytes", -1)) != artifact_path.stat().st_size
                        or str(value.get("sha256") or "").lower() != _sha256_path(artifact_path)
                    ):
                        raise DownstreamRunError(f"{label} bytes changed")
                    return 1
                return sum(
                    verify_artifact_records(child, f"{label}.{key}")
                    for key, child in value.items()
                )
            if isinstance(value, list):
                return sum(
                    verify_artifact_records(child, f"{label}[{index}]")
                    for index, child in enumerate(value)
                )
            return 0

        for variant in VARIANTS:
            evaluation_path = _safe_path(
                evaluation_paths[variant],
                f"{variant} evaluation model",
            )
            before_hash = _sha256_path(evaluation_path)
            model = _load_json(evaluation_path, f"{variant} evaluation model")
            after_hash = _sha256_path(evaluation_path)
            if before_hash != after_hash:
                raise DownstreamRunError(f"{variant} evaluation model changed during read")
            if model.get("schema_version") != "scryglass:future-value-four-variant-evaluation:v1":
                raise DownstreamRunError(f"{variant} evaluation model schema changed")
            _authority_is_closed(model.get("authority"), f"{variant} evaluation model")
            model_source = model.get("source")
            if not isinstance(model_source, Mapping):
                raise DownstreamRunError(f"{variant} evaluation model source is missing")
            if any(
                model_source.get(field) != source_contract[field]
                for field in (
                    "source_as_of",
                    "source_game_count",
                    "source_identity_sha256",
                    "source_receipt_sha256",
                )
            ):
                raise DownstreamRunError(f"{variant} evaluation model source changed")
            calibration_prior = model_source.get("calibration_prior")
            if not isinstance(calibration_prior, Mapping):
                raise DownstreamRunError(f"{variant} calibration prior binding is missing")
            if calibration_prior.get("variant_keys") != [variant]:
                raise DownstreamRunError(f"{variant} calibration prior variant binding changed")
            if calibration_prior.get("source_receipt_sha256") != source_contract["source_receipt_sha256"]:
                raise DownstreamRunError(f"{variant} calibration prior source changed")
            prior_path = _safe_path(
                str(calibration_prior.get("path") or ""),
                f"{variant} calibration prior",
            )
            if (
                int(calibration_prior.get("bytes", -1)) != prior_path.stat().st_size
                or str(calibration_prior.get("sha256") or "").lower() != _sha256_path(prior_path)
            ):
                raise DownstreamRunError(f"{variant} calibration prior bytes changed")
            prior_payload = _load_json(prior_path, f"{variant} calibration prior")
            if prior_payload.get("receipt_sha256") != calibration_prior.get("payload_receipt_sha256"):
                raise DownstreamRunError(f"{variant} calibration prior receipt changed")
            payloads = model.get("variants")
            payload = payloads.get(variant) if isinstance(payloads, Mapping) else None
            if not isinstance(payload, Mapping) or payload.get("variant") != variant:
                raise DownstreamRunError(f"{variant} evaluation payload is missing")
            _authority_is_closed(payload.get("authority"), f"{variant} evaluation payload")
            payload_blockers = payload.get("blockers")
            if isinstance(payload_blockers, list) and any(str(value).strip() for value in payload_blockers):
                raise DownstreamRunError(f"{variant} evaluation payload carries blockers")
            current_payload_source = payload.get("source")
            if not isinstance(current_payload_source, Mapping):
                raise DownstreamRunError(f"{variant} evaluation payload source is missing")
            if any(
                current_payload_source.get(field) != expected
                for field, expected in source_contract.items()
                if field in {"source_as_of", "source_game_count", "source_identity_sha256", "source_receipt_sha256", "model_eligible_game_count", "model_eligible_identity_sha256"}
            ):
                raise DownstreamRunError(f"{variant} evaluation payload source changed")
            if current_payload_source.get("accepted_game_ids") != source_contract["accepted_game_ids"]:
                raise DownstreamRunError(f"{variant} evaluation payload accepted census changed")
            if current_payload_source.get("model_eligible_game_ids") != source_contract["model_eligible_game_ids"]:
                raise DownstreamRunError(f"{variant} evaluation payload eligible census changed")
            variant_receipt = payload.get("variant_receipt")
            if not isinstance(variant_receipt, Mapping):
                raise DownstreamRunError(f"{variant} evaluation variant receipt is missing")
            variant_receipt_hash = _require_hash(
                variant_receipt.get("receipt_sha256"),
                f"{variant} evaluation variant receipt hash",
            )
            variant_receipt_body = dict(variant_receipt)
            variant_receipt_body.pop("receipt_sha256", None)
            if _sha256_bytes(_canonical(variant_receipt_body)) != variant_receipt_hash:
                raise DownstreamRunError(f"{variant} evaluation variant receipt hash changed")
            folds = payload.get("folds")
            if not isinstance(folds, list) or not folds:
                raise DownstreamRunError(f"{variant} nested folds are missing")
            feature_names: list[str] | None = None
            copied_folds: list[Any] = []
            for fold in folds:
                if not isinstance(fold, Mapping):
                    raise DownstreamRunError(f"{variant} nested fold is invalid")
                selection = fold.get("regularization_selection")
                if not isinstance(selection, Mapping):
                    raise DownstreamRunError(f"{variant} nested selection is missing from a fold")
                if fold.get("variant_receipt") != dict(variant_receipt):
                    raise DownstreamRunError(f"{variant} fold variant receipt changed")
                if selection.get("variant") != variant:
                    raise DownstreamRunError(f"{variant} nested selection variant changed")
                if selection.get("method") != "nested_chronological_whole_series_log_loss":
                    raise DownstreamRunError(f"{variant} nested selection method changed")
                if selection.get("inner_ledger_status") != "verified":
                    raise DownstreamRunError(f"{variant} nested inner ledger is not verified")
                if selection.get("blockers"):
                    raise DownstreamRunError(f"{variant} nested selection carries blockers")
                binding = selection.get("inner_feature_ledger_binding")
                if not isinstance(binding, Mapping):
                    raise DownstreamRunError(f"{variant} nested inner ledger binding is missing")
                names = binding.get("feature_names")
                if not isinstance(names, list) or not names or any(not isinstance(item, str) for item in names):
                    raise DownstreamRunError(f"{variant} nested feature names are missing")
                if feature_names is None:
                    feature_names = list(names)
                elif feature_names != list(names):
                    raise DownstreamRunError(f"{variant} nested feature order changed across folds")
                artifacts = binding.get("producer_artifacts")
                if not isinstance(artifacts, Mapping) or not artifacts:
                    raise DownstreamRunError(f"{variant} nested producer artifacts are missing")
                for field in (
                    "producer_receipt_sha256",
                    "ledger_rows_sha256",
                    "feature_value_digest",
                    "game_identity_sha256",
                    "fit_game_identity_sha256",
                    "validation_game_identity_sha256",
                    "binding_sha256",
                ):
                    if not SHA256_RE.fullmatch(str(binding.get(field) or "")):
                        raise DownstreamRunError(
                            f"{variant} nested inner ledger binding is incomplete: {field}"
                        )
                binding_body = dict(binding)
                binding_body.pop("binding_sha256", None)
                if _sha256_bytes(_canonical(binding_body)) != str(binding.get("binding_sha256")).lower():
                    raise DownstreamRunError(f"{variant} nested inner ledger binding hash changed")
                if binding.get("source_receipt_sha256") != source_contract["source_receipt_sha256"] or binding.get("source_identity_sha256") != source_contract["source_identity_sha256"]:
                    raise DownstreamRunError(f"{variant} nested inner ledger source changed")
                if verify_artifact_records(artifacts, f"{variant} nested producer artifacts") < 1:
                    raise DownstreamRunError(f"{variant} nested producer artifacts are missing")
                copied_folds.append(json.loads(json.dumps(fold, allow_nan=False)))
            if feature_names is None:
                raise DownstreamRunError(f"{variant} nested feature names are missing")
            if top_source is None:
                top_source = dict(source_contract)
            elif dict(top_source) != dict(source_contract):
                raise DownstreamRunError("evaluation model source bindings differ")
            records[variant] = {
                "path": str(evaluation_path),
                "bytes": evaluation_path.stat().st_size,
                "sha256": before_hash,
            }
            variant_payloads[variant] = {
                "authority": json.loads(json.dumps(payload.get("authority"), allow_nan=False)),
                "status": payload.get("status"),
                "variant": variant,
                "variant_receipt": json.loads(json.dumps(payload.get("variant_receipt"), allow_nan=False)),
                "source": json.loads(json.dumps(current_payload_source, allow_nan=False)),
                "feature_names": feature_names,
                "folds": copied_folds,
                "evaluation_artifact": dict(records[variant]),
            }
        if top_source is None:
            raise DownstreamRunError("nested selection source bindings are missing")
        body: dict[str, Any] = {
            "schema_version": "scryglass:future-value-nested-selection-bundle:v1",
            "status": "research_only",
            "authority": dict(AUTHORITY),
            "source": dict(top_source),
            "source_receipt_file_sha256": expected_source_file_hash,
            "evaluation_artifacts": records,
            "variants": variant_payloads,
            "blockers": [],
        }
        body["artifact_sha256"] = _sha256_bytes(_canonical(body))
        output = _safe_output_root(args.output, "nested selection output")
        _write_json(output, body)
        print(
            json.dumps(
                {
                    "status": body["status"],
                    "variants": list(VARIANTS),
                    "artifact_sha256": body["artifact_sha256"],
                    "file_sha256": _sha256_path(output),
                },
                sort_keys=True,
            )
        )
        return 0
    except (DownstreamRunError, FutureValueSourceError, OSError, ValueError, TypeError, KeyError) as error:
        print(f"nested selection bundle worker failed: {error}", file=sys.stderr)
        return 1


def _snapshot_capability_manifest_worker(argv: Sequence[str]) -> int:
    """Seal per-variant snapshot bundles and accept typed N/A capability rows."""

    parser = argparse.ArgumentParser(description="Build an all-variant snapshot capability manifest")
    parser.add_argument("--snapshot-capability-manifest-worker", action="store_true")
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--variant-output", action="append", default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(list(argv))
    try:
        from lol_kills.research.future_value_snapshots import (
            SNAPSHOT_CAPABILITY_SCHEMA_VERSION,
            SNAPSHOT_CAPABILITY_MATRIX,
        )

        source_path = args.source_receipt.resolve()
        source = _load_json(source_path, "source receipt")
        validate_future_value_source_receipt_payload(source)
        _authority_is_closed(source.get("authority"), "snapshot capability source")
        expected_source_file = _require_hash(
            args.source_receipt_file_sha256, "source receipt file hash"
        )
        if _sha256_path(source_path) != expected_source_file:
            raise DownstreamRunError("snapshot capability source receipt file changed")
        directories = _variant_directory_arguments(args.variant_output, "--variant-output")
        source_binding = {
            "source_as_of": source.get("source_as_of"),
            "source_game_count": source.get("source_game_count"),
            "source_identity_sha256": source.get("source_identity_sha256"),
            "source_receipt_sha256": source.get("receipt_sha256"),
            "source_receipt_file_sha256": expected_source_file,
            "model_eligible_game_count": source.get("model_eligible_game_count"),
            "model_eligible_identity_sha256": source.get("model_eligible_identity_sha256"),
        }
        variants: dict[str, Any] = {}
        manifest_blockers: list[str] = []
        for variant in VARIANTS:
            root = _safe_path(directories[variant], f"{variant} snapshot root", directory=True)
            receipt_path = _safe_path(root / "future-value-snapshot-receipt.json", f"{variant} snapshot receipt")
            manifest_path = _safe_path(root / "manifest.json", f"{variant} snapshot manifest")
            receipt = _receipt_hash(receipt_path, f"{variant} snapshot receipt")
            manifest = _load_json(manifest_path, f"{variant} snapshot manifest")
            if manifest.get("schema_version") != "scryglass:future-value-snapshot:v1":
                raise DownstreamRunError(f"{variant} snapshot manifest schema changed")
            if receipt.get("capability_schema_version") != SNAPSHOT_CAPABILITY_SCHEMA_VERSION:
                raise DownstreamRunError(f"{variant} snapshot capability schema changed")
            manifest_hash = _require_hash(manifest.get("manifest_sha256"), f"{variant} snapshot manifest hash")
            body = dict(manifest)
            body.pop("manifest_sha256", None)
            if _sha256_bytes(_canonical(body)) != manifest_hash:
                raise DownstreamRunError(f"{variant} snapshot manifest hash changed")
            if receipt.get("variant") != variant or manifest.get("variant") != variant:
                raise DownstreamRunError(f"{variant} snapshot variant binding changed")
            blockers: list[str] = []
            receipt_source = receipt.get("source")
            if not isinstance(receipt_source, Mapping):
                raise DownstreamRunError(f"{variant} snapshot source binding is missing")
            required_source_fields = {
                "source_as_of",
                "source_game_count",
                "source_identity_sha256",
                "source_receipt_sha256",
                "model_eligible_game_count",
                "model_eligible_identity_sha256",
            }
            if not required_source_fields.issubset(receipt_source):
                raise DownstreamRunError(f"{variant} snapshot source binding is incomplete")
            for field, expected in source_binding.items():
                if field in receipt_source and receipt_source.get(field) != expected:
                    raise DownstreamRunError(f"{variant} snapshot source changed: {field}")
            for field, expected in source_binding.items():
                if field in manifest and manifest.get(field) != expected:
                    raise DownstreamRunError(f"{variant} snapshot manifest source changed: {field}")
            files = manifest.get("files")
            if not isinstance(files, Mapping):
                raise DownstreamRunError(f"{variant} snapshot manifest files are missing")
            required_file_keys = {
                "player_snapshot",
                "team_snapshot",
                "player_rank_diffs",
                "team_rank_diffs",
                "receipt",
            }
            if not required_file_keys.issubset(files):
                missing_files = ", ".join(sorted(required_file_keys.difference(files)))
                raise DownstreamRunError(f"{variant} snapshot manifest files are incomplete: {missing_files}")
            file_records: dict[str, Any] = {}
            for name, raw in files.items():
                # Use the runner's stricter file binding helper.  The local
                # manifest records are absolute and remain inside the bundle.
                if not isinstance(raw, Mapping):
                    raise DownstreamRunError(f"{variant} snapshot file record is invalid: {name}")
                path = _safe_path(str(raw.get("path") or ""), f"{variant} snapshot {name}")
                try:
                    path.relative_to(root.resolve())
                except ValueError as error:
                    raise DownstreamRunError(f"{variant} snapshot file escapes root: {name}") from error
                if int(raw.get("bytes", -1)) != path.stat().st_size or str(raw.get("sha256") or "").lower() != _sha256_path(path):
                    raise DownstreamRunError(f"{variant} snapshot file changed: {name}")
                file_records[str(name)] = {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_path(path)}
            capability = manifest.get("capability")
            if not isinstance(capability, Mapping):
                capability = receipt.get("capability")
            if not isinstance(capability, Mapping):
                raise DownstreamRunError(f"{variant} snapshot capability is missing")
            if variant == "scaling_curve":
                for kind in ("player", "team"):
                    cap = capability.get(f"{kind}_ranks")
                    coverage = receipt.get("rank_coverage", {}).get(kind, {})
                    if not isinstance(cap, Mapping) or cap.get("status") != "not_applicable":
                        raise DownstreamRunError(f"{variant} {kind} capability is not typed N/A")
                    if not isinstance(coverage, Mapping) or coverage.get("status") != "not_applicable" or coverage.get("row_policy") != "no_rows":
                        raise DownstreamRunError(f"{variant} {kind} N/A coverage is incomplete")
            variant_blockers = (
                list(receipt.get("blockers", []))
                if isinstance(receipt.get("blockers"), list)
                else blockers
            )
            manifest_blockers.extend(
                f"{variant}:{value}" for value in variant_blockers if str(value).strip()
            )
            variants[variant] = {
                "status": receipt.get("status"),
                "variant": variant,
                "capability_schema_version": receipt.get("capability_schema_version"),
                "capability": capability,
                "blockers": variant_blockers,
                "player_row_count": receipt.get("player_row_count"),
                "team_row_count": receipt.get("team_row_count"),
                "player_rank_diff_count": receipt.get("player_rank_diff_count"),
                "team_rank_diff_count": receipt.get("team_rank_diff_count"),
                "rank_coverage": receipt.get("rank_coverage", {}),
                "receipt": {**_file_record(receipt_path), "receipt_sha256": receipt.get("receipt_sha256")},
                "manifest": {**_file_record(manifest_path), "manifest_sha256": manifest_hash},
                "files": file_records,
            }
        payload: dict[str, Any] = {
            "schema_version": SNAPSHOT_CAPABILITY_MANIFEST_SCHEMA_VERSION,
            "capability_schema_version": SNAPSHOT_CAPABILITY_SCHEMA_VERSION,
            "status": "blocked" if manifest_blockers else "research_only",
            "authority": dict(AUTHORITY),
            "source": source_binding,
            "variants": variants,
            "capability_matrix": json.loads(json.dumps(SNAPSHOT_CAPABILITY_MATRIX, sort_keys=True)),
            "blockers": sorted(set(manifest_blockers)),
        }
        payload["manifest_sha256"] = _sha256_bytes(_canonical(payload))
        _write_json(args.output.resolve(), payload)
        print(json.dumps({"status": payload["status"], "variants": list(VARIANTS)}, sort_keys=True))
        return 0
    except (DownstreamRunError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"snapshot capability manifest worker failed: {error}", file=sys.stderr)
        return 1


def _tier_shadow_fourway_worker(argv: Sequence[str]) -> int:
    """Run the variant-neutral retrospective Tier shadow producer."""

    parser = argparse.ArgumentParser(description="Build a four-way full-census Tier shadow")
    parser.add_argument("--tier-shadow-fourway-worker", action="store_true")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--current-ledger", required=True, type=Path)
    parser.add_argument("--current-ledger-sha256", required=True)
    parser.add_argument("--current-receipt", required=True, type=Path)
    parser.add_argument("--current-receipt-file-sha256", required=True)
    parser.add_argument("--variant-model", action="append", default=[])
    parser.add_argument("--scaling-ledger", type=Path)
    parser.add_argument("--scaling-ledger-sha256")
    parser.add_argument("--scaling-receipt", type=Path)
    parser.add_argument("--scaling-receipt-file-sha256")
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args(list(argv))
    try:
        from lol_kills.research.future_value_tier_shadow import build_frozen_fourway_tier_shadow

        models = _variant_directory_arguments(args.variant_model, "--variant-model")
        model_inputs: dict[str, dict[str, Any]] = {}
        destinations: dict[str, Path] = {}
        for ordinal, variant in enumerate(VARIANTS, start=1):
            root = _safe_path(models[variant], f"{variant} final-fit root", directory=True)
            model = _safe_path(root / f"final-v{ordinal}-model.json", f"{variant} final-fit model")
            receipt = _safe_path(root / f"final-v{ordinal}-model-receipt.json", f"{variant} final-fit receipt")
            run = _safe_path(root / "final-fit-run.json", f"{variant} final-fit run")
            model_inputs[variant] = {
                "model_path": model,
                "model_receipt_path": receipt,
                "run_receipt_path": run,
                "expected_model_sha256": _sha256_path(model),
                "expected_model_receipt_file_sha256": _sha256_path(receipt),
                "expected_run_receipt_sha256": _sha256_path(run),
            }
            destinations[variant] = args.output_root.resolve() / variant
        results = build_frozen_fourway_tier_shadow(
            source_root=args.source_root.resolve(),
            source_receipt_path=args.source_receipt.resolve(),
            expected_source_receipt_file_sha256=_require_hash(args.source_receipt_file_sha256, "source receipt file hash"),
            model_inputs=model_inputs,
            current_ledger_path=args.current_ledger.resolve(),
            current_receipt_path=args.current_receipt.resolve(),
            expected_current_ledger_sha256=_require_hash(args.current_ledger_sha256, "current ledger hash"),
            expected_current_receipt_file_sha256=_require_hash(args.current_receipt_file_sha256, "current receipt file hash"),
            scaling_ledger_path=args.scaling_ledger.resolve() if args.scaling_ledger else None,
            scaling_receipt_path=args.scaling_receipt.resolve() if args.scaling_receipt else None,
            expected_scaling_ledger_sha256=args.scaling_ledger_sha256,
            expected_scaling_receipt_file_sha256=args.scaling_receipt_file_sha256,
            destinations=destinations,
        )
        output_root = args.output_root.resolve()
        output_root.mkdir(parents=True, exist_ok=True)
        source_payload = _load_json(args.source_receipt.resolve(), "Tier shadow source receipt")
        variants_payload = {
            variant: {
                "variant": variant,
                "ledger": _file_record(result.ledger_path),
                "receipt": _file_record(result.receipt_path),
                "game_count": len(result.game_ids),
                "game_identity_sha256": identity_sha256(result.game_ids),
                "provenance": dict(result.provenance),
            }
            for variant, result in sorted(results.items())
        }
        payload: dict[str, Any] = {
            "schema_version": "scryglass:future-value-tier-shadow-fourway:v1",
            "status": "research_only",
            "authority": dict(AUTHORITY),
            "variants": variants_payload,
            "source": {
                "source_as_of": source_payload.get("source_as_of"),
                "source_game_count": source_payload.get("source_game_count"),
                "source_identity_sha256": source_payload.get("source_identity_sha256"),
                "source_receipt_sha256": source_payload.get("receipt_sha256"),
                "model_eligible_game_count": source_payload.get("model_eligible_game_count"),
                "model_eligible_identity_sha256": source_payload.get("model_eligible_identity_sha256"),
                "source_receipt_file_sha256": _sha256_path(args.source_receipt.resolve()),
            },
            "blockers": ["retrospective_full_census_model_fit_not_chronological_evaluation"],
        }
        payload["manifest_sha256"] = _sha256_bytes(_canonical(payload))
        _write_json(output_root / "fourway-tier-shadow-manifest.json", payload)
        print(json.dumps({"status": payload["status"], "variants": list(VARIANTS)}, sort_keys=True))
        return 0
    except (DownstreamRunError, OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        print(f"Tier shadow worker failed: {error}", file=sys.stderr)
        return 1


def _scaling_bindings(config: RunConfig) -> dict[str, Any]:
    """Resolve internal or caller-supplied online scaling evidence."""

    supplied = (
        config.scaling_root,
        config.scaling_artifact,
        config.scaling_artifact_sha256,
        config.scaling_receipt,
        config.scaling_receipt_sha256,
        config.scaling_manifest,
        config.scaling_manifest_sha256,
    )
    if not any(value is not None for value in supplied):
        root = _root(config, "scaling-online")
        return {
            "external": False,
            "root": root,
            "artifact": root / SCALING_ARTIFACT_NAME,
            "artifact_sha256": None,
            "receipt": root / SCALING_RECEIPT_NAME,
            "receipt_sha256": None,
            "manifest": root / SCALING_MANIFEST_NAME,
            "manifest_sha256": None,
            "blockers": (),
        }
    complete = all(
        value is not None
        for value in (
            config.scaling_artifact,
            config.scaling_artifact_sha256,
            config.scaling_receipt,
            config.scaling_receipt_sha256,
            config.scaling_manifest,
            config.scaling_manifest_sha256,
        )
    )
    root = config.scaling_root or (
        config.scaling_artifact.parent
        if config.scaling_artifact is not None
        else _root(config, "scaling-online")
    )
    return {
        "external": complete,
        "root": root,
        "artifact": config.scaling_artifact or root / SCALING_ARTIFACT_NAME,
        "artifact_sha256": config.scaling_artifact_sha256,
        "receipt": config.scaling_receipt or root / SCALING_RECEIPT_NAME,
        "receipt_sha256": config.scaling_receipt_sha256,
        "manifest": config.scaling_manifest or root / SCALING_MANIFEST_NAME,
        "manifest_sha256": config.scaling_manifest_sha256,
        "blockers": () if complete else ("scaling_external_inputs_incomplete",),
    }


def _nested_selection_bindings(config: RunConfig) -> dict[str, Any]:
    """Resolve caller evidence or the runner-owned combined bundle."""

    if config.nested_selection is None and config.nested_selection_sha256 is None:
        path = _root(config, "nested-selection") / NESTED_SELECTION_BUNDLE_NAME
        return {
            "external": False,
            "path": path,
            "sha256": None,
            "blockers": (),
        }
    if config.nested_selection is None or config.nested_selection_sha256 is None:
        return {
            "external": True,
            "path": config.nested_selection or _root(config, "nested-selection") / NESTED_SELECTION_BUNDLE_NAME,
            "sha256": config.nested_selection_sha256,
            "blockers": ("nested_selection_external_input_incomplete",),
        }
    return {
        "external": True,
        "path": config.nested_selection,
        "sha256": config.nested_selection_sha256,
        "blockers": (),
    }


def _core_stage_plan(config: RunConfig, inputs: ResolvedInputs) -> tuple[Stage, ...]:
    source = _source_freeze_root(config)
    receipt = _source_freeze_receipt(config)
    current = _root(config, "current-rating-trust")
    final = _root(config, "final-fit")
    snapshots = _root(config, "snapshots")
    final_manifest_root = _root(config, "final-fit-manifest")
    snapshot_manifest_root = _root(config, "snapshot-capabilities")
    current_receipt = current / "current-rating-ledger-receipt.json"
    current_artifact = current / "current-rating-ledger.parquet"
    current_snapshot_receipt = current / "current-rating-snapshot-receipt-v1.json"
    scaling = _scaling_bindings(config)
    nested_binding = _nested_selection_bindings(config)
    nested = nested_binding["path"]
    nested_hash = nested_binding["sha256"]
    nested_blockers = list(nested_binding["blockers"])
    if nested_binding["external"]:
        if not nested.is_file() or nested.is_symlink():
            nested_blockers.append("nested_selection_input_missing")
        elif nested_binding["sha256"] is None:
            nested_blockers.append("nested_selection_input_missing")
        else:
            nested_blockers.extend(_nested_selection_blockers(nested, inputs))
    current_stage = Stage(
        name="current_rating_trust",
        jobs=(Job(
            name="current_rating_trust",
            command=_python_module(
                "benchmarks.build_full_current_rating_trust",
                "--source-root", source,
                "--source-receipt", receipt,
                "--source-receipt-file-sha256", inputs.source_receipt_file_sha256,
                "--expected-source-receipt-sha256", inputs.source_receipt_sha256,
                "--output-root", current,
            ),
            output_roots=(current,),
            expected_files=(current_artifact, current_receipt, current_snapshot_receipt),
            input_paths=(source / "maps.parquet", source / "oe_player_games.parquet", source / "oe_team_games.parquet", receipt),
        ),),
        output_roots=(current,),
        expected_files=(current_artifact, current_receipt, current_snapshot_receipt),
    )
    scaling_stage: Stage | None = None
    if not scaling["external"]:
        scaling_root = scaling["root"]
        scaling_stage = Stage(
            name="scaling_online",
            jobs=(Job(
                name="scaling_online",
                command=_python_module(
                    "benchmarks.run_future_value_downstream",
                    "--scaling-online-worker",
                    "--source-root", source,
                    "--source-receipt", receipt,
                    "--source-receipt-file-sha256", inputs.source_receipt_file_sha256,
                    "--source-receipt-sha256", inputs.source_receipt_sha256,
                    "--output-root", scaling_root,
                ),
                output_roots=(scaling_root,),
                expected_files=(scaling["artifact"], scaling["receipt"], scaling["manifest"]),
                input_paths=(
                    source / "maps.parquet",
                    source / "oe_player_games.parquet",
                    source / "oe_team_games.parquet",
                    receipt,
                ),
                output_dir_policy="absent",
            ),),
            output_roots=(scaling_root,),
            expected_files=(scaling["artifact"], scaling["receipt"], scaling["manifest"]),
            blockers=tuple(scaling["blockers"]),
        )
    elif scaling["blockers"]:
        scaling_stage = Stage(
            name="scaling_online",
            output_roots=(scaling["root"],),
            expected_files=(scaling["artifact"], scaling["receipt"], scaling["manifest"]),
            blockers=tuple(scaling["blockers"]),
        )
    nested_stage: Stage | None = None
    if not nested_binding["external"]:
        nested_command: list[str] = list(_python_module(
            "benchmarks.run_future_value_downstream",
            "--nested-selection-bundle-worker",
        )) + [
            "--source-receipt", receipt,
            "--source-receipt-file-sha256", inputs.source_receipt_file_sha256,
        ]
        for variant in VARIANTS:
            nested_command.extend(("--evaluation", f"{variant}={inputs.evaluation_paths[variant]}"))
        nested_command.extend(("--output", nested))
        nested_stage = Stage(
            name="nested_selection",
            jobs=(Job(
                name="nested_selection",
                command=tuple(nested_command),
                output_roots=(nested.parent,),
                expected_files=(nested,),
                input_paths=(receipt, *inputs.evaluation_paths.values()),
                output_dir_policy="absent",
            ),),
            output_roots=(nested.parent,),
            expected_files=(nested,),
        )
    final_jobs: list[Job] = []
    final_roots: list[Path] = []
    final_expected: list[Path] = []
    for ordinal, variant in enumerate(VARIANTS, start=1):
        variant_root = final / variant
        model = variant_root / f"final-v{ordinal}-model.json"
        model_receipt = variant_root / f"final-v{ordinal}-model-receipt.json"
        run_receipt = variant_root / "final-fit-run.json"
        command = list(
            _python_module(
                "benchmarks.build_future_value_final_fit",
                "--variant", variant,
                "--source-root", source,
                "--source-receipt", receipt,
                "--current-root", current,
                "--evaluation", inputs.evaluation_paths[variant],
                "--evaluation-sha256", _digest_or_placeholder(inputs.evaluation_paths[variant], f"{variant}-evaluation"),
                *( ("--baseline-cache", config.baseline_cache) if config.baseline_cache is not None else () ),
                "--source-receipt-sha256", inputs.source_receipt_file_sha256,
                "--current-receipt-sha256", _digest_or_placeholder(current_receipt, "current-receipt-file"),
                "--current-artifact-sha256", _digest_or_placeholder(current_artifact, "current-artifact"),
                "--nested-selection", nested,
                "--nested-selection-sha256", nested_hash or _digest_or_placeholder(nested, "nested-selection"),
                "--output-dir", variant_root,
            )
        )
        if variant in {"scaling_curve", "both"}:
            command.extend((
                "--scaling-root", scaling["root"],
                "--scaling-artifact", scaling["artifact"],
                "--scaling-receipt", scaling["receipt"],
                "--scaling-manifest", scaling["manifest"],
                "--scaling-artifact-sha256", scaling["artifact_sha256"] or _digest_or_placeholder(scaling["artifact"], "scaling-artifact"),
                "--scaling-receipt-sha256", scaling["receipt_sha256"] or _digest_or_placeholder(scaling["receipt"], "scaling-receipt"),
                "--scaling-manifest-sha256", scaling["manifest_sha256"] or _digest_or_placeholder(scaling["manifest"], "scaling-manifest"),
            ))
        final_jobs.append(
            Job(
                name=f"final_fit_{variant}",
                command=tuple(command),
                output_roots=(variant_root,),
                expected_files=(model, model_receipt, run_receipt),
                input_paths=(
                    source / "maps.parquet", source / "oe_player_games.parquet",
                    source / "oe_team_games.parquet", receipt, current_artifact,
                    current_receipt, inputs.evaluation_paths[variant],
                    nested,
                    *( (scaling["artifact"], scaling["receipt"], scaling["manifest"])
                       if variant in {"scaling_curve", "both"} else () ),
                ),
                output_dir_policy="absent",
            )
        )
        final_roots.append(variant_root)
        final_expected.extend((model, model_receipt, run_receipt))
    final_stage = Stage(
        name="final_fit",
        jobs=() if nested_blockers or scaling["blockers"] else tuple(final_jobs),
        output_roots=tuple(final_roots),
        expected_files=tuple(final_expected),
        blockers=tuple(sorted(set(nested_blockers).union(scaling["blockers"]))),
    )
    manifest_command: list[str] = list(
        _python_module("benchmarks.run_future_value_downstream", "--final-fit-manifest-worker")
    ) + ["--source-receipt", receipt, "--source-receipt-file-sha256", inputs.source_receipt_file_sha256]
    for variant in VARIANTS:
        manifest_command.extend(("--variant-output", f"{variant}={final / variant}"))
    final_manifest = final_manifest_root / "final-fit-manifest.json"
    final_manifest_stage = Stage(
        name="final_fit_manifest",
        jobs=(Job(
            name="final_fit_manifest",
            command=tuple(manifest_command + ["--output", final_manifest]),
            output_roots=(final_manifest_root,),
            expected_files=(final_manifest,),
            input_paths=tuple(
                [receipt]
                + [path for path in final_expected]
            ),
            output_dir_policy="absent",
        ),),
        output_roots=(final_manifest_root,),
        expected_files=(final_manifest,),
    )
    snapshot_jobs: list[Job] = []
    snapshot_roots: list[Path] = []
    snapshot_expected: list[Path] = []
    for ordinal, variant in enumerate(VARIANTS, start=1):
        snapshot_root = snapshots / variant
        snapshot_receipt = snapshot_root / "future-value-snapshot-receipt.json"
        snapshot_expected_files = (
            snapshot_receipt,
            snapshot_root / "future-player-value-snapshot.json",
            snapshot_root / "future-team-value-snapshot.json",
            snapshot_root / "future-player-rank-diffs.json",
            snapshot_root / "future-team-rank-diffs.json",
            snapshot_root / "manifest.json",
        )
        command = list(_python_module(
            "benchmarks.build_future_value_snapshots",
            "--variant", variant,
            "--source-root", source,
            "--source-receipt", receipt,
            "--source-receipt-sha256", inputs.source_receipt_file_sha256,
            "--current-root", current,
            "--current-receipt", current_snapshot_receipt,
            "--current-receipt-sha256", _digest_or_placeholder(current_snapshot_receipt, "current-snapshot-receipt-file"),
            "--output-root", snapshot_root,
        ))
        variant_model_root = final / variant
        model = variant_model_root / f"final-v{ordinal}-model.json"
        model_receipt = variant_model_root / f"final-v{ordinal}-model-receipt.json"
        command.extend(("--model-receipt", model_receipt, "--model-artifact", model))
        snapshot_jobs.append(Job(
            name=f"snapshot_{variant}",
            command=tuple(command),
            output_roots=(snapshot_root,),
            expected_files=snapshot_expected_files,
            input_paths=tuple(
                [source / "maps.parquet", source / "oe_player_games.parquet", source / "oe_team_games.parquet",
                 receipt, current_snapshot_receipt,
                 current / "player/player_ratings_snapshot.parquet",
                 current / "team/ratings_snapshot.parquet"]
                + [inputs.evaluation_paths[variant], model, model_receipt]
            ),
            output_dir_policy="absent",
        ))
        snapshot_roots.append(snapshot_root)
        snapshot_expected.extend(snapshot_expected_files)
    snapshot_stage = Stage(
        name="snapshots",
        jobs=tuple(snapshot_jobs),
        output_roots=tuple(snapshot_roots),
        expected_files=tuple(snapshot_expected),
    )
    capability_manifest = snapshot_manifest_root / "snapshot-capability-manifest.json"
    capability_command: list[str] = list(
        _python_module("benchmarks.run_future_value_downstream", "--snapshot-capability-manifest-worker")
    ) + ["--source-receipt", receipt, "--source-receipt-file-sha256", inputs.source_receipt_file_sha256]
    for variant in VARIANTS:
        capability_command.extend(("--variant-output", f"{variant}={snapshots / variant}"))
    capability_inputs = [receipt]
    capability_inputs.extend(snapshot_expected)
    comparison_stage = Stage(
        name="snapshot_capabilities",
        jobs=(Job(
            name="snapshot_capabilities",
            command=tuple(capability_command + ["--output", capability_manifest]),
            output_roots=(snapshot_manifest_root,),
            expected_files=(capability_manifest,),
            input_paths=tuple(capability_inputs),
        ),),
        output_roots=(snapshot_manifest_root,),
        expected_files=(capability_manifest,),
    )
    stages: list[Stage] = [_source_freeze_stage(config, inputs), current_stage]
    if scaling_stage is not None:
        stages.append(scaling_stage)
    if nested_stage is not None:
        stages.append(nested_stage)
    stages.extend((final_stage, final_manifest_stage, snapshot_stage, comparison_stage))
    return tuple(stages)


def build_stage_plan(config: RunConfig, inputs: ResolvedInputs | None = None) -> tuple[Stage, ...]:
    """Return the downstream child command plan.

    ``inputs`` is optional for command-plan inspection.  A real run always
    validates the four-way receipt before it constructs this plan.
    """

    if inputs is None:
        root = config.fourway_root
        source_root = root / "source"
        source_receipt = root / "future-value-source-receipt.json"
        inputs = ResolvedInputs(
            fourway_root=root,
            source_root=source_root,
            source_receipt=source_receipt,
            source_receipt_file_sha256="0" * 64,
            source_receipt_sha256="0" * 64,
            source_as_of="",
            accepted_game_ids=(),
            accepted_identity_sha256="0" * 64,
            eligible_game_ids=(),
            eligible_identity_sha256="0" * 64,
            evaluation_paths={variant: root / "stages/evaluation" / variant / "model.json" for variant in VARIANTS},
            evaluation_runtime_paths={variant: root / "stages/evaluation" / variant / "runtime.json" for variant in VARIANTS},
            evaluation_stage_receipt=root / "receipts/evaluations.json",
            evaluation_receipt_paths={},
            paired_uncertainty=root / "stages/paired-uncertainty/paired-uncertainty.json",
            paired_uncertainty_csv=root / "stages/paired-uncertainty/paired-uncertainty.csv",
            paired_identity_sha256="0" * 64,
            paired_rows=0,
            freeze_root=root,
        )
    stages = list(_core_stage_plan(config, inputs))
    return tuple(stages)


def _plan_digest(stage: Stage) -> str:
    payload = {
        "stage": stage.name,
        "blockers": list(stage.blockers),
        "jobs": [
            {
                "name": job.name,
                "command": [str(token) for token in job.command],
                "output_roots": [str(path) for path in job.output_roots],
                "expected_files": [str(path) for path in job.expected_files],
                "input_paths": [str(path) for path in job.input_paths],
                "output_dir_policy": job.output_dir_policy,
            }
            for job in stage.jobs
        ],
    }
    return _sha256_bytes(_canonical(payload))


def _blocked_stage(stage: Stage, blockers: Iterable[str]) -> Stage:
    """Carry an upstream blocker into a dependent stage plan."""

    values = tuple(sorted({str(value) for value in (*stage.blockers, *blockers) if str(value).strip()}))
    return Stage(
        name=stage.name,
        jobs=() if values else stage.jobs,
        output_roots=stage.output_roots,
        expected_files=stage.expected_files,
        blockers=values,
    )


def _ensure_empty(root: Path) -> None:
    if root.exists() and root.is_symlink():
        raise DownstreamRunError(f"stage output root is a symlink: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise DownstreamRunError(f"stage output root is not empty: {root}")


def _collect_outputs(roots: Iterable[Path]) -> list[dict[str, Any]]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_symlink() or not root.is_dir():
            raise DownstreamRunError(f"stage output root is missing: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise DownstreamRunError(f"stage output contains a symlink: {path}")
            if path.is_file():
                paths.add(path)
    return [_file_record(path) for path in sorted(paths, key=str)]


def _semantic_blockers(stage: Stage) -> list[str]:
    """Read producer status fields after a successful child exit."""

    # ``research_only_blocked`` is a valid authority status for a completed
    # research artifact.  Its semantic blocker list determines computation
    # availability.  Execution failures use the statuses below.
    blocked_statuses = {"blocked", "invalid", "failed", "error"}
    blockers: list[str] = []
    for path in stage.expected_files:
        if path.suffix.lower() != ".json" or not path.is_file() or path.is_symlink():
            continue
        try:
            value = _load_json(path, f"{stage.name} semantic output")
        except DownstreamRunError as error:
            blockers.append(f"{stage.name}:semantic_output_invalid:{path.name}")
            continue
        status = value.get("status")
        if isinstance(status, str) and status in blocked_statuses:
            blockers.append(f"{stage.name}:semantic_status:{path.name}:{status}")
        raw = value.get("blockers")
        if isinstance(raw, list):
            blockers.extend(
                f"{stage.name}:semantic_blocker:{path.name}:{item}"
                for item in raw
                if str(item).strip()
            )
    return sorted(set(blockers))


def _input_records(stage: Stage) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted({path.resolve() for job in stage.jobs for path in job.input_paths}, key=str):
        records[str(path)] = _file_record(path)
    return records


def _log_record(path: Path, value: bytes, limit: int) -> dict[str, Any]:
    original = len(value)
    stored = value[:limit]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(stored)
    return {"path": str(path), "bytes": len(stored), "original_bytes": original, "truncated": original > limit, "sha256": _sha256_path(path)}


def _run_job(job: Job, config: RunConfig, stage_name: str) -> dict[str, Any]:
    for root in job.output_roots:
        if job.output_dir_policy == "absent":
            if root.exists() or root.is_symlink():
                raise DownstreamRunError(
                    f"job output root must not exist: {root}"
                )
        else:
            _ensure_empty(root)
    log_root = config.output_root / "logs" / stage_name
    started = time.perf_counter()
    started_at = _utc_now()
    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "1"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env.setdefault(name, "1")
    try:
        completed = subprocess.run(list(job.command), cwd=str(config.repository_root), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        stdout = completed.stdout if isinstance(completed.stdout, bytes) else str(completed.stdout or "").encode()
        stderr = completed.stderr if isinstance(completed.stderr, bytes) else str(completed.stderr or "").encode()
        code = int(completed.returncode)
    except OSError as error:
        stdout, stderr, code = b"", str(error).encode("utf-8", errors="replace"), 127
    stdout_record = _log_record(log_root / f"{job.name}.stdout.log", stdout, config.max_log_bytes)
    stderr_record = _log_record(log_root / f"{job.name}.stderr.log", stderr, config.max_log_bytes)
    result = {
        "name": job.name,
        "command": list(job.command),
        "started_at": started_at,
        "completed_at": _utc_now(),
        "duration_seconds": time.perf_counter() - started,
        "exit_code": code,
        "stdout": stdout_record,
        "stderr": stderr_record,
    }
    if code != 0:
        result["error"] = "child command failed"
    return result


def _stage_receipt_path(config: RunConfig, stage: Stage) -> Path:
    return config.output_root / "receipts" / f"{stage.name}.json"


def _write_stage_receipt(config: RunConfig, stage: Stage, *, status: str, blockers: Sequence[str] = (), jobs: Sequence[Mapping[str, Any]] = (), inputs: Mapping[str, Any] | None = None, outputs: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    path = _stage_receipt_path(config, stage)
    if path.exists() or path.is_symlink():
        raise DownstreamRunError(f"stage receipt already exists: {path}")
    job_values = [
        float(job.get("duration_seconds", 0.0))
        for job in jobs
        if isinstance(job, Mapping)
    ]
    payload: dict[str, Any] = {
        "schema_version": STAGE_RECEIPT_SCHEMA_VERSION,
        "status": status,
        "stage": stage.name,
        "stage_plan_sha256": _plan_digest(stage),
        "started_at": _utc_now(),
        "completed_at": _utc_now(),
        "duration_seconds": sum(job_values),
        "jobs": [dict(job) for job in jobs],
        "inputs": dict(inputs or {}),
        "outputs": [dict(output) for output in outputs],
        "blockers": sorted({str(value) for value in (*stage.blockers, *blockers) if str(value).strip()}),
        "authority": dict(AUTHORITY),
    }
    payload["receipt_sha256"] = _sha256_bytes(_canonical(payload))
    _write_json(path, payload)
    return payload


def _validate_stage_receipt(config: RunConfig, stage: Stage) -> dict[str, Any]:
    receipt = _receipt_hash(_stage_receipt_path(config, stage), f"downstream stage {stage.name}")
    if receipt.get("schema_version") != STAGE_RECEIPT_SCHEMA_VERSION or receipt.get("stage") != stage.name:
        raise DownstreamRunError(f"downstream stage {stage.name} receipt schema changed")
    if receipt.get("stage_plan_sha256") != _plan_digest(stage):
        raise DownstreamRunError(f"downstream stage {stage.name} command plan changed")
    if receipt.get("status") not in {"completed", "blocked"}:
        raise DownstreamRunError(f"downstream stage {stage.name} status is invalid")
    if receipt.get("authority") != AUTHORITY:
        raise DownstreamRunError(f"downstream stage {stage.name} authority changed")
    outputs = receipt.get("outputs", [])
    if not isinstance(outputs, list):
        raise DownstreamRunError(f"downstream stage {stage.name} output records are missing")
    output_paths: set[str] = set()
    for raw in outputs:
        if not isinstance(raw, Mapping):
            raise DownstreamRunError(f"downstream stage {stage.name} output record is invalid")
        path = _safe_path(str(raw.get("path") or ""), f"downstream stage {stage.name} output")
        inside_root = False
        for root in stage.output_roots:
            try:
                path.relative_to(root.resolve())
            except ValueError:
                continue
            inside_root = True
            break
        if not inside_root:
            raise DownstreamRunError(
                f"downstream stage {stage.name} output escapes stage root"
            )
        output_paths.add(str(path))
        if int(raw.get("bytes", -1)) != path.stat().st_size or str(raw.get("sha256") or "").lower() != _sha256_path(path):
            raise DownstreamRunError(f"downstream stage {stage.name} output hash changed")
    if receipt.get("status") == "completed":
        for expected in stage.expected_files:
            if not expected.is_file() or expected.is_symlink() or str(expected) not in output_paths:
                raise DownstreamRunError(f"downstream stage {stage.name} expected output is missing")
    jobs = receipt.get("jobs", [])
    if not isinstance(jobs, list):
        raise DownstreamRunError(f"downstream stage {stage.name} job records are invalid")
    planned_jobs = {job.name: job for job in stage.jobs}
    recorded_names: set[str] = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            raise DownstreamRunError(f"downstream stage {stage.name} job record is invalid")
        name = str(job.get("name") or "")
        planned = planned_jobs.get(name)
        if planned is None or name in recorded_names:
            raise DownstreamRunError(f"downstream stage {stage.name} job binding changed")
        command = job.get("command")
        if not isinstance(command, list) or tuple(str(value) for value in command) != planned.command:
            raise DownstreamRunError(f"downstream stage {stage.name} command binding changed")
        duration = job.get("duration_seconds")
        if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(float(duration)) or float(duration) < 0:
            raise DownstreamRunError(f"downstream stage {stage.name} job duration is invalid")
        if isinstance(job.get("exit_code"), bool) or not isinstance(job.get("exit_code"), int):
            raise DownstreamRunError(f"downstream stage {stage.name} job exit code is invalid")
        if receipt.get("status") == "completed" and (
            not isinstance(job.get("stdout"), Mapping)
            or not isinstance(job.get("stderr"), Mapping)
        ):
            raise DownstreamRunError(f"downstream stage {stage.name} job logs are incomplete")
        recorded_names.add(name)
    if receipt.get("status") == "completed" and recorded_names != set(planned_jobs):
        raise DownstreamRunError(f"downstream stage {stage.name} job records are incomplete")
    input_values = receipt.get("inputs", {})
    if not isinstance(input_values, Mapping):
        raise DownstreamRunError(f"downstream stage {stage.name} input records are missing")
    for key, raw in input_values.items():
        if not isinstance(raw, Mapping):
            raise DownstreamRunError(f"downstream stage {stage.name} input record is invalid")
        path = _safe_path(str(raw.get("path") or ""), f"downstream stage {stage.name} input")
        if str(key) != str(path):
            raise DownstreamRunError(f"downstream stage {stage.name} input path binding changed")
        if int(raw.get("bytes", -1)) != path.stat().st_size or str(raw.get("sha256") or "").lower() != _sha256_path(path):
            raise DownstreamRunError(f"downstream stage {stage.name} input hash changed")
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        for stream in ("stdout", "stderr"):
            raw = job.get(stream)
            if not isinstance(raw, Mapping):
                continue
            path = _safe_path(str(raw.get("path") or ""), f"downstream stage {stage.name} log")
            try:
                path.relative_to((config.output_root / "logs" / stage.name).resolve())
            except ValueError as error:
                raise DownstreamRunError(
                    f"downstream stage {stage.name} log escapes log root"
                ) from error
            if int(raw.get("bytes", -1)) != path.stat().st_size or str(raw.get("sha256") or "").lower() != _sha256_path(path):
                raise DownstreamRunError(f"downstream stage {stage.name} log changed")
    return receipt


def _execute_stage(config: RunConfig, stage: Stage, *, resume: bool) -> dict[str, Any]:
    receipt_path = _stage_receipt_path(config, stage)
    if resume and receipt_path.exists():
        return _validate_stage_receipt(config, stage)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise DownstreamRunError(f"downstream stage {stage.name} has an existing receipt")
    for root in stage.output_roots:
        if root.exists() and any(root.iterdir()):
            if resume:
                raise DownstreamRunError(f"downstream stage {stage.name} has output without a valid receipt")
            raise DownstreamRunError(f"downstream stage {stage.name} output root is not empty")
    if stage.blockers:
        return _write_stage_receipt(config, stage, status="blocked", blockers=stage.blockers)
    if not stage.jobs:
        return _write_stage_receipt(config, stage, status="blocked", blockers=("stage_has_no_job",))
    inputs = _input_records(stage)
    results = [_run_job(job, config, stage.name) for job in stage.jobs]
    input_changed = False
    for path_text, record in inputs.items():
        try:
            current = _file_record(Path(path_text))
        except DownstreamRunError:
            input_changed = True
            continue
        if current.get("bytes") != record.get("bytes") or current.get("sha256") != record.get("sha256"):
            input_changed = True
    outputs = _collect_outputs(stage.output_roots) if all(int(job["exit_code"]) == 0 for job in results) and all(root.is_dir() for root in stage.output_roots) else []
    missing = [str(path) for path in stage.expected_files if not path.is_file() or path.is_symlink()]
    blockers = ["child_command_failed"] if any(int(job["exit_code"]) != 0 for job in results) else []
    if not blockers:
        blockers.extend(_semantic_blockers(stage))
    if input_changed:
        blockers.append("input_changed_during_execution")
    if missing:
        blockers.append("expected_output_missing")
    status = "completed" if not blockers else "blocked"
    return _write_stage_receipt(config, stage, status=status, blockers=blockers, jobs=results, inputs=inputs, outputs=outputs)


def _write_selection(config: RunConfig, inputs: ResolvedInputs, source: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(config, "selection")
    _ensure_empty(root)
    # The selection is a manual review annotation.  It never suppresses one
    # of the four evaluation chains.
    selected_blockers: list[str] = []
    selected_payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "caller_selected_blocked" if selected_blockers else "caller_selected_research_only",
        "selected_variant": config.selected_variant,
        "selection_method": "explicit_caller_choice",
        "auto_promotion": False,
        "source": {
            "source_as_of": inputs.source_as_of,
            "source_game_count": len(inputs.accepted_game_ids),
            "source_identity_sha256": inputs.accepted_identity_sha256,
            "model_eligible_game_count": len(inputs.eligible_game_ids),
            "model_eligible_identity_sha256": inputs.eligible_identity_sha256,
            "source_receipt_sha256": inputs.source_receipt_sha256,
            "source_receipt_file_sha256": inputs.source_receipt_file_sha256,
            "accepted_game_ids": list(inputs.accepted_game_ids),
            "model_eligible_game_ids": list(inputs.eligible_game_ids),
        },
        "evaluation": {
            variant: {"path": str(path), **_file_record(path)}
            for variant, path in sorted(inputs.evaluation_paths.items())
        },
        "evaluation_runtime": {
            variant: {"path": str(path), **_file_record(path)}
            for variant, path in sorted(inputs.evaluation_runtime_paths.items())
        },
        "evaluation_stage_receipt": _file_record(inputs.evaluation_stage_receipt),
        "paired_uncertainty": {**_file_record(inputs.paired_uncertainty), "rows": inputs.paired_rows, "game_identity_sha256": inputs.paired_identity_sha256},
        "blockers": selected_blockers,
        "authority": dict(AUTHORITY),
    }
    if inputs.paired_uncertainty_csv is not None:
        selected_payload["paired_uncertainty_csv"] = _file_record(inputs.paired_uncertainty_csv)
    selected_path = root / "selected-variant.json"
    _write_json(selected_path, selected_payload)
    # The downstream impact producer requires a per-variant byte receipt.  The
    # sidecars carry only values already verified above.
    receipt_paths: dict[str, Path] = {}
    for variant, artifact in sorted(inputs.evaluation_paths.items()):
        model = _load_json(artifact, f"{variant} evaluation")
        variant_blockers: list[str] = []
        payload = {
            "schema_version": EVALUATION_RECEIPT_SCHEMA_VERSION,
            "status": "blocked" if variant_blockers else "research_only",
            "variant": variant,
            "source": selected_payload["source"],
            "artifact": _file_record(artifact),
            "runtime": _file_record(inputs.evaluation_runtime_paths[variant]),
            "authority": dict(AUTHORITY),
            "model_status": model.get("variants", {}).get(variant, {}).get("status") if isinstance(model.get("variants"), Mapping) else None,
            "blockers": variant_blockers,
        }
        payload["receipt_sha256"] = _sha256_bytes(_canonical(payload))
        path = root / "evaluation-receipts" / f"{variant}.json"
        _write_json(path, payload)
        receipt_paths[variant] = path
    return {"path": selected_path, "receipt_paths": receipt_paths, "payload": selected_payload}


def _validate_selection_payload(
    config: RunConfig,
    inputs: ResolvedInputs,
    path: Path,
) -> dict[str, Any]:
    value = _load_json(path, "selected variant artifact")
    if value.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise DownstreamRunError("selected variant schema changed")
    if value.get("selected_variant") != config.selected_variant:
        raise DownstreamRunError("selected variant choice changed")
    if value.get("selection_method") != "explicit_caller_choice" or value.get("auto_promotion") is not False:
        raise DownstreamRunError("selected variant was not caller-selected")
    expected_selection_blockers: list[str] = []
    if value.get("status") != (
        "caller_selected_blocked" if expected_selection_blockers else "caller_selected_research_only"
    ):
        raise DownstreamRunError("selected variant status changed")
    if value.get("blockers") != expected_selection_blockers:
        raise DownstreamRunError("selected variant blockers changed")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise DownstreamRunError("selected variant source binding is missing")
    expected_source = {
        "source_as_of": inputs.source_as_of,
        "source_game_count": len(inputs.accepted_game_ids),
        "source_identity_sha256": inputs.accepted_identity_sha256,
        "model_eligible_game_count": len(inputs.eligible_game_ids),
        "model_eligible_identity_sha256": inputs.eligible_identity_sha256,
        "source_receipt_sha256": inputs.source_receipt_sha256,
        "source_receipt_file_sha256": inputs.source_receipt_file_sha256,
        "accepted_game_ids": list(inputs.accepted_game_ids),
        "model_eligible_game_ids": list(inputs.eligible_game_ids),
    }
    for field, expected in expected_source.items():
        if source.get(field) != expected:
            raise DownstreamRunError(f"selected variant source binding changed: {field}")
    evaluations = value.get("evaluation")
    runtimes = value.get("evaluation_runtime")
    if not isinstance(evaluations, Mapping) or not isinstance(runtimes, Mapping):
        raise DownstreamRunError("selected variant evaluation bindings are missing")
    for variant in VARIANTS:
        for bindings, expected_path, label in (
            (evaluations, inputs.evaluation_paths[variant], "evaluation"),
            (runtimes, inputs.evaluation_runtime_paths[variant], "evaluation runtime"),
        ):
            record = bindings.get(variant)
            if not isinstance(record, Mapping):
                raise DownstreamRunError(f"selected variant {label} binding is missing: {variant}")
            actual = _file_record(expected_path)
            if record.get("path") != str(expected_path) or record.get("bytes") != actual["bytes"] or record.get("sha256") != actual["sha256"]:
                raise DownstreamRunError(f"selected variant {label} binding changed: {variant}")
    paired = value.get("paired_uncertainty")
    if not isinstance(paired, Mapping):
        raise DownstreamRunError("selected variant paired uncertainty binding is missing")
    paired_actual = _file_record(inputs.paired_uncertainty)
    if any(paired.get(field) != paired_actual[field] for field in ("path", "bytes", "sha256")):
        raise DownstreamRunError("selected variant paired uncertainty binding changed")
    selection_root = path.parent
    for variant in VARIANTS:
        receipt_path = selection_root / "evaluation-receipts" / f"{variant}.json"
        receipt = _receipt_hash(receipt_path, f"selected {variant} evaluation receipt")
        if (
            receipt.get("schema_version") != EVALUATION_RECEIPT_SCHEMA_VERSION
            or receipt.get("variant") != variant
            or receipt.get("authority") != AUTHORITY
            or receipt.get("source") != source
        ):
            raise DownstreamRunError(f"selected {variant} evaluation receipt binding changed")
        artifact = _file_record(inputs.evaluation_paths[variant])
        runtime = _file_record(inputs.evaluation_runtime_paths[variant])
        if receipt.get("artifact") != artifact or receipt.get("runtime") != runtime:
            raise DownstreamRunError(f"selected {variant} evaluation receipt artifacts changed")
        expected_variant_blockers: list[str] = []
        if receipt.get("status") != (
            "blocked" if expected_variant_blockers else "research_only"
        ):
            raise DownstreamRunError(f"selected {variant} evaluation receipt status changed")
        if receipt.get("blockers") != expected_variant_blockers:
            raise DownstreamRunError(f"selected {variant} evaluation receipt blockers changed")
    if value.get("authority") != AUTHORITY:
        raise DownstreamRunError("selected variant authority changed")
    return value


def _selected_variant_receipt_blockers(
    config: RunConfig,
    receipt_paths: Mapping[str, Path],
) -> list[str]:
    """Return blockers for the caller-selected capability receipt only."""

    path = receipt_paths.get(config.selected_variant)
    if path is None:
        raise DownstreamRunError("selected variant capability receipt is missing")
    receipt = _receipt_hash(path, f"selected {config.selected_variant} evaluation receipt")
    raw = receipt.get("blockers", [])
    if not isinstance(raw, list):
        raise DownstreamRunError("selected variant capability receipt blockers are invalid")
    return [
        f"selection:{config.selected_variant}:{value}"
        for value in raw
        if str(value).strip()
    ]


def _blocked_impact(path: Path, inputs: ResolvedInputs, blockers: Sequence[str]) -> dict[str, Any]:
    payload = {
        "schema_version": "scryglass:future-value-downstream-impact:v1",
        "status": "blocked",
        "generated_utc": _utc_now(),
        "source": {
            "source_as_of": inputs.source_as_of,
            "source_game_count": len(inputs.accepted_game_ids),
            "source_identity_sha256": inputs.accepted_identity_sha256,
            "model_eligible_game_count": len(inputs.eligible_game_ids),
            "model_eligible_identity_sha256": inputs.eligible_identity_sha256,
            "source_receipt_sha256": inputs.source_receipt_sha256,
            "source_receipt_file_sha256": inputs.source_receipt_file_sha256,
            "receipt_path": str(inputs.source_receipt),
            "receipt_file": _file_record(inputs.source_receipt),
        },
        "downstream_public_change_flags": {
            "player_ratings": False,
            "team_ratings": False,
            "tierlists": False,
            "draft_score": False,
            "matches": False,
            "probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "deployment": False,
            "measured_changes_present": {},
        },
        "authority": dict(AUTHORITY),
        "blockers": sorted({str(value) for value in blockers}),
        "claim_ceiling": "source-bound research impact review; public outputs remain disabled",
    }
    _write_json(path, payload)
    return payload


def _downstream_stage(config: RunConfig, inputs: ResolvedInputs, selection: Mapping[str, Any], previous_blockers: Sequence[str]) -> Stage:
    root = _root(config, "downstream-impact")
    output = root / "downstream-impact.json"
    required: list[Path] = [_source_freeze_receipt(config)] + list(inputs.evaluation_paths.values()) + list(selection["receipt_paths"].values())
    required.append(inputs.paired_uncertainty)
    required.extend((_root(config, "final-fit-manifest") / "final-fit-manifest.json", _root(config, "snapshot-capabilities") / "snapshot-capability-manifest.json"))
    for variant in VARIANTS:
        snapshot_root = _root(config, "snapshots") / variant
        required.extend(
            snapshot_root / name
            for name in (
                "future-value-snapshot-receipt.json",
                "future-player-rank-diffs.json",
                "future-team-rank-diffs.json",
                "manifest.json",
            )
        )
    blockers = list(previous_blockers)
    tier_diff = _root(config, "tier-diff") / "current-fourway-full-census-tier-diff.json"
    tier_receipt = _root(config, "tier-diff") / "current-fourway-full-census-tier-diff-receipt.json"
    tier_fourway = _root(config, "tier-fourway") / "fourway-tierlist-report.json"
    tier_fourway_receipt = _root(config, "tier-fourway") / "run-receipt.json"
    tier_shadow = _root(config, "tier-shadow") / "fourway-tier-shadow-manifest.json"
    draft_report = _root(config, "draft-score") / "fourway-report.json"
    required.extend((tier_diff, tier_receipt, tier_fourway, tier_fourway_receipt, tier_shadow, draft_report))
    if any(not path.is_file() or path.is_symlink() for path in required):
        blockers.append("downstream_impact_required_input_missing")
    eval_args: list[str] = []
    for variant in VARIANTS:
        eval_args.extend(("--evaluation", f"{variant}={inputs.evaluation_paths[variant]}"))
        eval_args.extend(("--evaluation-receipt", f"{variant}={selection['receipt_paths'][variant]}"))
    command = _python_module(
        "benchmarks.build_future_value_downstream_impact",
        "--source-receipt", _source_freeze_receipt(config),
        *eval_args,
        "--snapshot-capability-manifest", _root(config, "snapshot-capabilities") / "snapshot-capability-manifest.json",
        "--tier-diff", tier_diff,
        "--tier-receipt", tier_receipt,
        "--tier-fourway-report", tier_fourway,
        "--tier-fourway-receipt", tier_fourway_receipt,
        "--tier-shadow-manifest", tier_shadow,
        "--draft-score-report", draft_report,
        "--paired-uncertainty", inputs.paired_uncertainty,
        "--output", output,
    )
    for variant in VARIANTS:
        snapshot_root = _root(config, "snapshots") / variant
        command += ("--snapshot-variant", f"{variant}={snapshot_root / 'future-value-snapshot-receipt.json'}")
        command += ("--snapshot-manifest-variant", f"{variant}={snapshot_root / 'manifest.json'}")
    return Stage(name="downstream_impact", jobs=() if blockers else (Job(name="downstream_impact", command=command, output_roots=(root,), expected_files=(output,), input_paths=tuple(required)),), output_roots=(root,), expected_files=(output,), blockers=tuple(sorted(set(blockers))))


def _optional_stage_plan(
    config: RunConfig,
    inputs: ResolvedInputs,
    *,
    tier_candidate_sha256: str | None = None,
) -> tuple[Stage, ...]:
    final = _root(config, "final-fit")
    current = _root(config, "current-rating-trust")
    stage_list: list[Stage] = []
    staged_source = _source_freeze_root(config)
    staged_receipt = _source_freeze_receipt(config)
    tier_source = config.tier_source_root or staged_source
    if config.tier_source_root is not None:
        candidate_tier_receipt = config.tier_source_root / SOURCE_FREEZE_RECEIPT_NAME
        tier_receipt = (
            candidate_tier_receipt
            if candidate_tier_receipt.is_file()
            else inputs.source_receipt
        )
    else:
        tier_receipt = staged_receipt
    tier_root = _root(config, "tier-shadow")
    scaling = _scaling_bindings(config)
    tier_blockers: list[str] = []
    if config.tier_source_root is None or config.tier_trust_manifest is None or config.tier_trust_manifest_sha256 is None:
        tier_blockers.append("tier_shadow_exact_inputs_missing")
    if config.tier_source_root is not None and any(
        not path.is_file() or path.is_symlink()
        for path in (
            config.tier_source_root / "source" / "oe_player_games.parquet",
            config.tier_source_root / "source" / "meta.json",
        )
    ):
        tier_blockers.append("tier_shadow_source_files_missing")
    tier_blockers.extend(str(value) for value in scaling["blockers"])
    tier_expected_files: list[Path] = [tier_root / "fourway-tier-shadow-manifest.json"]
    for variant in VARIANTS:
        tier_expected_files.extend((
            tier_root / variant / "tier-offset-ledger.json",
            tier_root / variant / "tier-offset-ledger-receipt.json",
        ))
    tier_command: list[str] = list(
        _python_module("benchmarks.run_future_value_downstream", "--tier-shadow-fourway-worker")
    ) + [
        "--source-root", tier_source,
        "--source-receipt", tier_receipt,
        "--source-receipt-file-sha256", _digest_or_placeholder(tier_receipt, "tier-source-receipt-file"),
        "--current-ledger", current / "current-rating-ledger.parquet",
        "--current-ledger-sha256", _digest_or_placeholder(current / "current-rating-ledger.parquet", "current-ledger"),
        "--current-receipt", current / "current-rating-ledger-receipt.json",
        "--current-receipt-file-sha256", _digest_or_placeholder(current / "current-rating-ledger-receipt.json", "current-receipt-file"),
        "--output-root", tier_root,
    ]
    for ordinal, variant in enumerate(VARIANTS, start=1):
        tier_command.extend(("--variant-model", f"{variant}={final / variant}"))
    tier_command.extend((
        "--scaling-ledger", scaling["artifact"],
        "--scaling-ledger-sha256", scaling["artifact_sha256"] or _digest_or_placeholder(scaling["artifact"], "scaling-ledger"),
        "--scaling-receipt", scaling["receipt"],
        "--scaling-receipt-file-sha256", scaling["receipt_sha256"] or _digest_or_placeholder(scaling["receipt"], "scaling-receipt-file"),
    ))
    tier_inputs = [tier_receipt, current / "current-rating-ledger.parquet", current / "current-rating-ledger-receipt.json"]
    tier_inputs.extend(
        final / variant / name
        for variant in VARIANTS
        for name in (f"final-v{VARIANTS.index(variant) + 1}-model.json", f"final-v{VARIANTS.index(variant) + 1}-model-receipt.json", "final-fit-run.json")
    )
    tier_inputs.extend(path for path in (config.tier_trust_manifest, scaling["artifact"], scaling["receipt"]) if path is not None)
    stage_list.append(Stage(
        name="tier_shadow",
        jobs=() if tier_blockers else (Job(name="tier_shadow", command=tuple(tier_command), output_roots=(tier_root,), expected_files=tuple(tier_expected_files), input_paths=tuple(tier_inputs), output_dir_policy="absent"),),
        output_roots=(tier_root,),
        expected_files=tuple(tier_expected_files),
        blockers=tuple(sorted(set(tier_blockers))),
    ))

    chronological_root = _root(config, "tier-fourway")
    chronological_report = chronological_root / "fourway-tierlist-report.json"
    chronological_receipt = chronological_root / "run-receipt.json"
    chronological_blockers: list[str] = []
    if config.tier_source_root is None or config.tier_repository_root is None or config.tier_trust_manifest is None or config.tier_trust_manifest_sha256 is None:
        chronological_blockers.append("tier_fourway_exact_inputs_missing")
    chronological_command = _python_module(
        "benchmarks.future_value_tierlist_fourway",
        "--repo-root", config.tier_repository_root or config.repository_root,
        "--source-root", tier_source,
        "--evaluation-root", inputs.fourway_root / "stages/evaluation",
        "--trust-manifest", config.tier_trust_manifest or "",
        "--expected-trust-manifest-sha256", config.tier_trust_manifest_sha256 or "",
        "--output-root", chronological_root,
        "--workers", config.workers,
    )
    chronological_inputs = [tier_receipt, config.tier_trust_manifest] + list(inputs.evaluation_paths.values())
    stage_list.append(Stage(
        name="tier_fourway",
        jobs=() if chronological_blockers else (Job(name="tier_fourway", command=chronological_command, output_roots=(chronological_root,), expected_files=(chronological_report, chronological_receipt), input_paths=tuple(path for path in chronological_inputs if path is not None), output_dir_policy="absent"),),
        output_roots=(chronological_root,),
        expected_files=(chronological_report, chronological_receipt),
        blockers=tuple(sorted(set(chronological_blockers))),
    ))

    diff_root = _root(config, "tier-diff")
    diff_report = diff_root / "current-fourway-full-census-tier-diff.json"
    diff_receipt = diff_root / "current-fourway-full-census-tier-diff-receipt.json"
    diff_blockers = list(chronological_blockers)
    if config.tier_source_root is None or config.tier_trust_manifest is None or config.tier_trust_manifest_sha256 is None:
        diff_blockers.append("tier_diff_exact_inputs_missing")
    diff_command: list[str] = list(_python_module(
        "benchmarks.future_value_tierlist_full_census_diff",
        "--trust-manifest", config.tier_trust_manifest or "",
        "--expected-trust-manifest-sha256", config.tier_trust_manifest_sha256 or "",
        "--source-root", tier_source,
        "--output-root", diff_root,
    ))
    for variant in VARIANTS:
        candidate = chronological_root / "candidates" / f"{variant}.json"
        diff_command.extend(("--variant-candidate", f"{variant}={candidate}", "--expected-variant-candidate-sha256", f"{variant}={_digest_or_placeholder(candidate, f'{variant}-tier-candidate')}"))
    diff_inputs = [chronological_report, *[chronological_root / "candidates" / f"{variant}.json" for variant in VARIANTS], config.tier_trust_manifest, tier_receipt]
    stage_list.append(Stage(
        name="tier_diff",
        jobs=() if diff_blockers else (Job(name="tier_diff", command=tuple(diff_command), output_roots=(diff_root,), expected_files=(diff_report, diff_receipt), input_paths=tuple(path for path in diff_inputs if path is not None), output_dir_policy="absent"),),
        output_roots=(diff_root,),
        expected_files=(diff_report, diff_receipt),
        blockers=tuple(sorted(set(diff_blockers))),
    ))

    draft_root = _root(config, "draft-score")
    draft_values = (
        config.draft_trust_root,
        config.draft_trust_root_sha256,
        config.draft_folds_root,
        config.draft_public_pack_root,
        config.draft_manifest_sha256,
    )
    draft_blockers = [] if all(value is not None for value in draft_values) else ["draft_score_exact_inputs_missing"]
    draft_fold_files: list[Path] = []
    if config.draft_folds_root is not None:
        draft_fold_files = [
            config.draft_folds_root / f"fold-{fold}-spec.json"
            for fold in (1, 2, 3)
        ]
        if any(not path.is_file() or path.is_symlink() for path in draft_fold_files):
            draft_blockers.append("draft_score_folds_root_incomplete")
    draft_command = _python_module(
        "benchmarks.future_value_draft_score_fourway",
        "--source-receipt", staged_receipt,
        "--expected-source-receipt-sha256", inputs.source_receipt_file_sha256,
        "--trust-root", config.draft_trust_root or "",
        "--expected-trust-root-sha256", config.draft_trust_root_sha256 or "",
        "--source-root", staged_source,
        "--folds-root", config.draft_folds_root or "",
        "--evaluation-root", inputs.fourway_root / "stages/evaluation",
        "--public-pack-root", config.draft_public_pack_root or "",
        "--expected-manifest-sha256", config.draft_manifest_sha256 or "",
        "--output-dir", draft_root,
    )
    if config.draft_authority_path is not None:
        draft_command += ("--authority-path", config.draft_authority_path, "--expected-authority-sha256", config.draft_authority_sha256 or "")
    if config.draft_model_artifact is not None:
        draft_command += (
            "--model-artifact-path", config.draft_model_artifact,
            "--expected-future-model-sha256", config.draft_model_artifact_sha256 or "",
        )
    if config.draft_strict_atom is not None:
        draft_command += (
            "--strict-atom-path", config.draft_strict_atom,
            "--expected-strict-atom-sha256", config.draft_strict_atom_sha256 or "",
        )
        if config.draft_strict_atom_code_sha256 is not None:
            draft_command += ("--expected-strict-atom-code-sha256", config.draft_strict_atom_code_sha256)
    if config.draft_strict_form is not None:
        draft_command += (
            "--strict-form-path", config.draft_strict_form,
            "--expected-strict-form-sha256", config.draft_strict_form_sha256 or "",
        )
        if config.draft_strict_form_code_sha256 is not None:
            draft_command += ("--expected-strict-form-code-sha256", config.draft_strict_form_code_sha256)
    if config.draft_strict_fold_root is not None:
        draft_command += ("--strict-fold-root", config.draft_strict_fold_root)
    draft_inputs = [staged_receipt, *inputs.evaluation_paths.values()]
    for path in (
        config.draft_trust_root,
        config.draft_public_pack_root / "manifest.json" if config.draft_public_pack_root is not None else None,
        config.draft_authority_path,
        config.draft_model_artifact,
        config.draft_strict_atom,
        config.draft_strict_form,
    ):
        if path is not None:
            draft_inputs.append(path)
    draft_inputs.extend(draft_fold_files)
    stage_list.append(
        Stage(
            name="draft_score",
            jobs=()
            if draft_blockers
            else (
                Job(
                    name="draft_score",
                    command=draft_command,
                    output_roots=(draft_root,),
                    expected_files=(
                        draft_root / "fourway-report.json",
                        draft_root / "descriptive-subset.json",
                    ),
                    input_paths=tuple(draft_inputs),
                    output_dir_policy="absent",
                ),
            ),
            output_roots=(draft_root,),
            expected_files=(
                draft_root / "fourway-report.json",
                draft_root / "descriptive-subset.json",
            ),
            blockers=tuple(sorted(set(draft_blockers))),
        )
    )
    return tuple(stage_list)


def _config_payload(config: RunConfig, inputs: ResolvedInputs) -> dict[str, Any]:
    return {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "fourway_root": str(config.fourway_root),
        "output_root": str(config.output_root),
        "selected_variant": config.selected_variant,
        "repository_root": str(config.repository_root),
        "source": {
            "source_receipt": _file_record(inputs.source_receipt),
            "source_receipt_sha256": inputs.source_receipt_sha256,
            "source_as_of": inputs.source_as_of,
            "accepted_game_count": len(inputs.accepted_game_ids),
            "accepted_identity_sha256": inputs.accepted_identity_sha256,
            "accepted_game_ids": list(inputs.accepted_game_ids),
            "model_eligible_game_count": len(inputs.eligible_game_ids),
            "model_eligible_identity_sha256": inputs.eligible_identity_sha256,
            "model_eligible_game_ids": list(inputs.eligible_game_ids),
            "evaluation_artifacts": {
                variant: {
                    "model": _file_record(inputs.evaluation_paths[variant]),
                    "runtime": _file_record(inputs.evaluation_runtime_paths[variant]),
                }
                for variant in VARIANTS
            },
            "paired_uncertainty": _file_record(inputs.paired_uncertainty),
            "paired_uncertainty_csv": (
                _file_record(inputs.paired_uncertainty_csv)
                if inputs.paired_uncertainty_csv is not None
                else None
            ),
        },
        "inputs": {
            "nested_selection": str(config.nested_selection) if config.nested_selection else None,
            "nested_selection_sha256": config.nested_selection_sha256,
            "baseline_cache": str(config.baseline_cache) if config.baseline_cache else None,
            "scaling_root": str(config.scaling_root) if config.scaling_root else None,
            "scaling_artifact": str(config.scaling_artifact) if config.scaling_artifact else None,
            "scaling_artifact_sha256": config.scaling_artifact_sha256,
            "scaling_receipt": str(config.scaling_receipt) if config.scaling_receipt else None,
            "scaling_receipt_sha256": config.scaling_receipt_sha256,
            "scaling_manifest": str(config.scaling_manifest) if config.scaling_manifest else None,
            "scaling_manifest_sha256": config.scaling_manifest_sha256,
            "tier_source_root": str(config.tier_source_root) if config.tier_source_root else None,
            "tier_repository_root": str(config.tier_repository_root) if config.tier_repository_root else None,
            "tier_trust_manifest": str(config.tier_trust_manifest) if config.tier_trust_manifest else None,
            "tier_trust_manifest_sha256": config.tier_trust_manifest_sha256,
            "tier_baseline_candidate": str(config.tier_baseline_candidate) if config.tier_baseline_candidate else None,
            "tier_production_manifest": str(config.tier_production_manifest) if config.tier_production_manifest else None,
            "tier_prospective_evaluation": str(config.tier_prospective_evaluation) if config.tier_prospective_evaluation else None,
            "tier_build_pooled_candidate": config.tier_build_pooled_candidate,
            "draft_trust_root": str(config.draft_trust_root) if config.draft_trust_root else None,
            "draft_trust_root_sha256": config.draft_trust_root_sha256,
            "draft_folds_root": str(config.draft_folds_root) if config.draft_folds_root else None,
            "draft_public_pack_root": str(config.draft_public_pack_root) if config.draft_public_pack_root else None,
            "draft_manifest_sha256": config.draft_manifest_sha256,
            "draft_authority_path": str(config.draft_authority_path) if config.draft_authority_path else None,
            "draft_authority_sha256": config.draft_authority_sha256,
            "draft_model_artifact": str(config.draft_model_artifact) if config.draft_model_artifact else None,
            "draft_model_artifact_sha256": config.draft_model_artifact_sha256,
            "draft_strict_atom": str(config.draft_strict_atom) if config.draft_strict_atom else None,
            "draft_strict_atom_sha256": config.draft_strict_atom_sha256,
            "draft_strict_atom_code_sha256": config.draft_strict_atom_code_sha256,
            "draft_strict_form": str(config.draft_strict_form) if config.draft_strict_form else None,
            "draft_strict_form_sha256": config.draft_strict_form_sha256,
            "draft_strict_form_code_sha256": config.draft_strict_form_code_sha256,
            "draft_strict_fold_root": str(config.draft_strict_fold_root) if config.draft_strict_fold_root else None,
            },
        "authority": dict(AUTHORITY),
    }


def _write_or_validate_config(config: RunConfig, inputs: ResolvedInputs, *, resume: bool) -> dict[str, Any]:
    path = config.output_root / "run-config.json"
    expected = _config_payload(config, inputs)
    expected_hash = _sha256_bytes(_canonical(expected))
    expected["config_sha256"] = expected_hash
    if resume:
        actual = _receipt_hash(path, "downstream run config")
        if actual.get("schema_version") != RUN_CONFIG_SCHEMA_VERSION or actual.get("config_sha256") != expected_hash:
            raise DownstreamRunError("downstream run configuration changed")
        return actual
    if path.exists() or path.is_symlink():
        raise DownstreamRunError("downstream run configuration already exists")
    _write_json(path, expected)
    return expected


def _write_final(config: RunConfig, inputs: ResolvedInputs, receipts: Sequence[Mapping[str, Any]], blockers: Sequence[str]) -> dict[str, Any]:
    path = config.output_root / "run-receipt.json"
    effective_blockers = list(blockers)
    for receipt in receipts:
        if receipt.get("status") != "completed":
            effective_blockers.append(
                f"stage_not_completed:{receipt.get('stage') or 'unknown'}"
            )
    effective_blockers = sorted(
        {str(value) for value in effective_blockers if str(value).strip()}
    )
    if path.exists():
        existing = _receipt_hash(path, "downstream final receipt")
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise DownstreamRunError("downstream final receipt schema changed")
        if (
            existing.get("selected_variant") != config.selected_variant
            or existing.get("selection_method") != "explicit_caller_choice"
            or existing.get("auto_promotion") is not False
        ):
            raise DownstreamRunError("downstream final selection binding changed")
        if existing.get("authority") != AUTHORITY:
            raise DownstreamRunError("downstream final receipt authority changed")
        expected_status = "research_only_complete" if not effective_blockers else "research_only_blocked"
        if existing.get("status") != expected_status:
            raise DownstreamRunError("downstream final receipt status changed")
        expected_source = {
            "source_receipt_sha256": inputs.source_receipt_sha256,
            "source_receipt_file_sha256": inputs.source_receipt_file_sha256,
            "source_as_of": inputs.source_as_of,
            "source_game_count": len(inputs.accepted_game_ids),
            "source_identity_sha256": inputs.accepted_identity_sha256,
            "accepted_game_ids": list(inputs.accepted_game_ids),
            "model_eligible_game_count": len(inputs.eligible_game_ids),
            "model_eligible_identity_sha256": inputs.eligible_identity_sha256,
            "model_eligible_game_ids": list(inputs.eligible_game_ids),
            "evaluation_artifacts": {
                variant: {
                    "model": _file_record(inputs.evaluation_paths[variant]),
                    "runtime": _file_record(inputs.evaluation_runtime_paths[variant]),
                }
                for variant in VARIANTS
            },
            "paired_uncertainty": _file_record(inputs.paired_uncertainty),
            "paired_uncertainty_csv": (
                _file_record(inputs.paired_uncertainty_csv)
                if inputs.paired_uncertainty_csv is not None
                else None
            ),
        }
        if existing.get("source") != expected_source:
            raise DownstreamRunError("downstream final source binding changed")
        current_stages = [
            {
                "stage": receipt.get("stage"),
                "status": receipt.get("status"),
                "receipt_sha256": receipt.get("receipt_sha256"),
            }
            for receipt in receipts
        ]
        existing_stages = existing.get("stages")
        if not isinstance(existing_stages, list) or any(
            not isinstance(item, Mapping) for item in existing_stages
        ) or [
            {
                "stage": item.get("stage"),
                "status": item.get("status"),
                "receipt_sha256": item.get("receipt_sha256"),
            }
            for item in existing_stages
        ] != current_stages:
            raise DownstreamRunError("downstream final stage bindings changed")
        if existing.get("blockers") != effective_blockers:
            raise DownstreamRunError("downstream final blocker list changed")
        return existing
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only_complete" if not effective_blockers else "research_only_blocked",
        "completed_at": _utc_now(),
        "selected_variant": config.selected_variant,
        "selection_method": "explicit_caller_choice",
        "auto_promotion": False,
        "source": {
            "source_receipt_sha256": inputs.source_receipt_sha256,
            "source_receipt_file_sha256": inputs.source_receipt_file_sha256,
            "source_as_of": inputs.source_as_of,
            "source_game_count": len(inputs.accepted_game_ids),
            "source_identity_sha256": inputs.accepted_identity_sha256,
            "accepted_game_ids": list(inputs.accepted_game_ids),
            "model_eligible_game_count": len(inputs.eligible_game_ids),
            "model_eligible_identity_sha256": inputs.eligible_identity_sha256,
            "model_eligible_game_ids": list(inputs.eligible_game_ids),
            "evaluation_artifacts": {
                variant: {
                    "model": _file_record(inputs.evaluation_paths[variant]),
                    "runtime": _file_record(inputs.evaluation_runtime_paths[variant]),
                }
                for variant in VARIANTS
            },
            "paired_uncertainty": _file_record(inputs.paired_uncertainty),
            "paired_uncertainty_csv": (
                _file_record(inputs.paired_uncertainty_csv)
                if inputs.paired_uncertainty_csv is not None
                else None
            ),
        },
        "stages": [
            {"stage": receipt.get("stage"), "status": receipt.get("status"), "receipt_sha256": receipt.get("receipt_sha256"), "outputs": receipt.get("outputs", []), "blockers": receipt.get("blockers", [])}
            for receipt in receipts
        ],
        "blockers": effective_blockers,
        "authority": dict(AUTHORITY),
    }
    payload["receipt_sha256"] = _sha256_bytes(_canonical(payload))
    _write_json(path, payload)
    return payload


def run(config: RunConfig, *, resume: bool = False, plan_only: bool = False) -> dict[str, Any]:
    """Validate the four-way run and execute the research-only downstream plan."""

    fourway_root = _safe_path(config.fourway_root, "fourway root", directory=True)
    _safe_path(config.repository_root, "repository root", directory=True)
    output_root = _safe_output_root(config.output_root, "downstream output root")
    fourway_config = _fourway_config(fourway_root)
    final = _receipt_hash(fourway_root / "run-receipt.json", "fourway final receipt")
    if final.get("schema_version") != "scryglass:future-value-fourway-run:v1" or final.get("status") != "research_only_complete":
        raise DownstreamRunError("fourway run is not complete")
    _authority_is_closed(final.get("authority"), "fourway final receipt")
    final_stages = final.get("stages")
    if not isinstance(final_stages, list):
        raise DownstreamRunError("fourway final stage receipt list is missing")
    if any(not isinstance(item, Mapping) or not str(item.get("stage") or "") for item in final_stages):
        raise DownstreamRunError("fourway final stage receipt list is invalid")
    final_stage_map = {str(item.get("stage")): item for item in final_stages if isinstance(item, Mapping)}
    if len(final_stage_map) != len(final_stages):
        raise DownstreamRunError("fourway final stage receipt list has duplicate stages")
    for required_stage in ("evaluations", "paired_uncertainty"):
        item = final_stage_map.get(required_stage)
        if not isinstance(item, Mapping) or item.get("status") != "completed":
            raise DownstreamRunError(f"fourway final receipt lacks completed {required_stage} stage")
        stage_receipt_path = fourway_root / "receipts" / f"{required_stage}.json"
        stage_receipt = _receipt_hash(stage_receipt_path, f"fourway stage {required_stage}")
        if item.get("receipt_sha256") != stage_receipt.get("receipt_sha256"):
            raise DownstreamRunError(f"fourway final receipt {required_stage} binding changed")
    inputs, source = _source_binding(fourway_root, fourway_config, final)
    _validate_optional_inputs(config, inputs)
    if not resume and output_root.exists() and any(output_root.iterdir()):
        raise DownstreamRunError("downstream output root must be empty unless --resume is set")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_or_validate_config(config, inputs, resume=resume)
    if plan_only:
        stages = build_stage_plan(config, inputs) + _optional_stage_plan(config, inputs)
        return {"status": "plan_only", "selected_variant": config.selected_variant, "stages": [{"name": stage.name, "plan_sha256": _plan_digest(stage), "blockers": list(stage.blockers), "jobs": [list(job.command) for job in stage.jobs]} for stage in stages], "authority": dict(AUTHORITY)}

    source_freeze_stage = _source_freeze_stage(config, inputs)
    if resume and _stage_receipt_path(config, source_freeze_stage).exists():
        source_freeze_receipt = _validate_stage_receipt(config, source_freeze_stage)
    else:
        source_freeze_receipt = _execute_stage(config, source_freeze_stage, resume=resume)

    selection_path = _root(config, "selection") / "selected-variant.json"
    selection_receipt_path = _stage_receipt_path(config, _selection_stage(config, inputs, source))
    if resume:
        if selection_receipt_path.exists():
            _validate_selection_payload(config, inputs, selection_path)
            selection = {
                "path": selection_path,
                "receipt_paths": {
                    variant: _root(config, "selection") / "evaluation-receipts" / f"{variant}.json"
                    for variant in VARIANTS
                },
            }
        elif _root(config, "selection").exists() and any(_root(config, "selection").iterdir()):
            raise DownstreamRunError("selection output exists without a valid receipt")
        else:
            selection = _write_selection(config, inputs, source)
    else:
        selection = _write_selection(config, inputs, source)
    receipts: list[dict[str, Any]] = [source_freeze_receipt]
    blockers: list[str] = _selected_variant_receipt_blockers(
        config,
        selection["receipt_paths"],
    )
    blockers.extend(str(value) for value in source_freeze_receipt.get("blockers", []))
    for variant, receipt_path in sorted(selection["receipt_paths"].items()):
        _receipt_hash(
            receipt_path,
            f"selected {variant} evaluation receipt",
        )
    selection_stage = _selection_stage(config, inputs, source)
    if resume and _stage_receipt_path(config, selection_stage).exists():
        receipts.append(_validate_stage_receipt(config, selection_stage))
    else:
        outputs = _collect_outputs((_root(config, "selection"),))
        selection_inputs = {
            str(path): _file_record(path)
            for path in (
                inputs.source_receipt,
                inputs.evaluation_stage_receipt,
                *inputs.evaluation_paths.values(),
                *inputs.evaluation_runtime_paths.values(),
                inputs.paired_uncertainty,
                *( (inputs.paired_uncertainty_csv,) if inputs.paired_uncertainty_csv is not None else () ),
            )
        }
        selection_receipt_blockers = list(
            value for value in blockers if str(value).startswith("selection:")
        )
        receipts.append(
            _write_stage_receipt(
                config,
                selection_stage,
                status="blocked" if selection_receipt_blockers else "completed",
                blockers=selection_receipt_blockers,
                outputs=outputs,
                inputs=selection_inputs,
            )
        )
    # Rebuild the plan after each dependency.  Child CLIs receive the raw
    # hashes of receipts produced by the preceding stage.
    failed_core_stages: set[str] = set()
    if source_freeze_receipt.get("status") != "completed":
        failed_core_stages.add("source_freeze")
    core_dependencies = {
        "current_rating_trust": {"source_freeze"},
        "scaling_online": {"source_freeze"},
        "nested_selection": {"source_freeze"},
        "final_fit": {"current_rating_trust", "scaling_online", "nested_selection", "source_freeze"},
        "final_fit_manifest": {"final_fit"},
        "snapshots": {"current_rating_trust", "final_fit"},
        "snapshot_capabilities": {"snapshots"},
    }
    for stage_name in (
        "current_rating_trust",
        "scaling_online",
        "nested_selection",
        "final_fit",
        "final_fit_manifest",
        "snapshots",
        "snapshot_capabilities",
    ):
        planned_core = build_stage_plan(config, inputs)
        stage = next((item for item in planned_core if item.name == stage_name), None)
        if stage is None:
            continue
        dependency_blockers = [
            f"dependency_{name}_blocked"
            for name in sorted(core_dependencies.get(stage_name, set()) & failed_core_stages)
        ]
        stage = _blocked_stage(stage, dependency_blockers)
        try:
            receipt = _execute_stage(config, stage, resume=resume)
        except DownstreamRunError as error:
            blockers.append(f"{stage.name}:{error}")
            receipt = _write_stage_receipt(config, stage, status="blocked", blockers=(str(error),)) if not _stage_receipt_path(config, stage).exists() else _validate_stage_receipt(config, stage)
        receipts.append(receipt)
        blockers.extend(str(value) for value in receipt.get("blockers", []))
        if receipt.get("status") != "completed":
            failed_core_stages.add(stage_name)

    failed_optional_stages: set[str] = set()
    optional_dependencies = {
        "tier_shadow": {"current_rating_trust", "final_fit", "source_freeze"},
        "tier_fourway": {"source_freeze"},
        "tier_diff": {"tier_fourway"},
        "draft_score": {"source_freeze"},
    }
    for optional_name in ("tier_shadow", "tier_fourway", "tier_diff", "draft_score"):
        stage = next(
            item
            for item in _optional_stage_plan(
                config,
                inputs,
                tier_candidate_sha256=None,
            )
            if item.name == optional_name
        )
        if stage.name in {"tier_shadow", "tier_fourway", "tier_diff"}:
            stage = _blocked_stage(
                stage,
                [
                    f"dependency_{name}_blocked"
                    for name in (
                        *sorted(optional_dependencies.get(optional_name, set())),
                    )
                    if name in failed_optional_stages or name in failed_core_stages
                ],
            )
        try:
            receipt = _execute_stage(config, stage, resume=resume)
        except DownstreamRunError as error:
            blockers.append(f"{stage.name}:{error}")
            receipt = _write_stage_receipt(config, stage, status="blocked", blockers=(str(error),)) if not _stage_receipt_path(config, stage).exists() else _validate_stage_receipt(config, stage)
        receipts.append(receipt)
        blockers.extend(str(value) for value in receipt.get("blockers", []))
        if receipt.get("status") != "completed":
            failed_optional_stages.add(optional_name)

    impact_stage = _downstream_stage(config, inputs, selection, blockers)
    if impact_stage.blockers:
        if resume and _stage_receipt_path(config, impact_stage).exists():
            impact_receipt = _validate_stage_receipt(config, impact_stage)
        else:
            impact_root = _root(config, "downstream-impact")
            _ensure_empty(impact_root)
            _blocked_impact(impact_root / "downstream-impact.json", inputs, impact_stage.blockers)
            impact_receipt = _write_stage_receipt(config, impact_stage, status="blocked", blockers=impact_stage.blockers, outputs=_collect_outputs((impact_root,)))
    else:
        impact_receipt = _execute_stage(config, impact_stage, resume=resume)
    receipts.append(impact_receipt)
    blockers.extend(str(value) for value in impact_receipt.get("blockers", []))
    return _write_final(config, inputs, receipts, blockers)


def _parse_args(argv: Sequence[str] | None = None) -> tuple[RunConfig, bool, bool]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fourway-root", "--fourway-run", dest="fourway_root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--selected-variant", required=True, choices=VARIANTS)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--nested-selection", type=Path)
    parser.add_argument("--nested-selection-sha256")
    parser.add_argument("--baseline-cache", type=Path)
    parser.add_argument("--scaling-root", type=Path)
    parser.add_argument("--scaling-artifact", type=Path)
    parser.add_argument("--scaling-artifact-sha256")
    parser.add_argument("--scaling-receipt", type=Path)
    parser.add_argument("--scaling-receipt-sha256")
    parser.add_argument("--scaling-manifest", type=Path)
    parser.add_argument("--scaling-manifest-sha256")
    parser.add_argument("--tier-source-root", type=Path)
    parser.add_argument("--tier-repository-root", type=Path)
    parser.add_argument("--tier-trust-manifest", type=Path)
    parser.add_argument("--tier-trust-manifest-sha256")
    parser.add_argument("--tier-baseline-candidate", type=Path)
    parser.add_argument("--tier-production-manifest", type=Path)
    parser.add_argument("--tier-prospective-evaluation", type=Path)
    parser.add_argument("--tier-build-pooled-candidate", action="store_true")
    parser.add_argument("--draft-trust-root", type=Path)
    parser.add_argument("--draft-trust-root-sha256")
    parser.add_argument("--draft-folds-root", type=Path)
    parser.add_argument("--draft-public-pack-root", type=Path)
    parser.add_argument("--draft-manifest-sha256")
    parser.add_argument("--draft-authority-path", type=Path)
    parser.add_argument("--draft-authority-sha256")
    parser.add_argument("--draft-model-artifact", type=Path)
    parser.add_argument("--draft-model-artifact-sha256")
    parser.add_argument("--draft-strict-atom", type=Path)
    parser.add_argument("--draft-strict-atom-sha256")
    parser.add_argument("--draft-strict-atom-code-sha256")
    parser.add_argument("--draft-strict-form", type=Path)
    parser.add_argument("--draft-strict-form-sha256")
    parser.add_argument("--draft-strict-form-code-sha256")
    parser.add_argument("--draft-strict-fold-root", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-log-bytes", type=int, default=DEFAULT_MAX_LOG_BYTES)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    config = RunConfig(
        fourway_root=args.fourway_root.expanduser().resolve(),
        output_root=args.output_root.expanduser().resolve(),
        selected_variant=str(args.selected_variant),
        repository_root=args.repository_root.expanduser().resolve(),
        nested_selection=args.nested_selection.expanduser().resolve() if args.nested_selection else None,
        nested_selection_sha256=args.nested_selection_sha256,
        baseline_cache=args.baseline_cache.expanduser().resolve() if args.baseline_cache else None,
        scaling_root=args.scaling_root.expanduser().resolve() if args.scaling_root else None,
        scaling_artifact=args.scaling_artifact.expanduser().resolve() if args.scaling_artifact else None,
        scaling_artifact_sha256=args.scaling_artifact_sha256,
        scaling_receipt=args.scaling_receipt.expanduser().resolve() if args.scaling_receipt else None,
        scaling_receipt_sha256=args.scaling_receipt_sha256,
        scaling_manifest=args.scaling_manifest.expanduser().resolve() if args.scaling_manifest else None,
        scaling_manifest_sha256=args.scaling_manifest_sha256,
        tier_source_root=args.tier_source_root.expanduser().resolve() if args.tier_source_root else None,
        tier_repository_root=args.tier_repository_root.expanduser().resolve() if args.tier_repository_root else None,
        tier_trust_manifest=args.tier_trust_manifest.expanduser().resolve() if args.tier_trust_manifest else None,
        tier_trust_manifest_sha256=args.tier_trust_manifest_sha256,
        tier_baseline_candidate=args.tier_baseline_candidate.expanduser().resolve() if args.tier_baseline_candidate else None,
        tier_production_manifest=args.tier_production_manifest.expanduser().resolve() if args.tier_production_manifest else None,
        tier_prospective_evaluation=args.tier_prospective_evaluation.expanduser().resolve() if args.tier_prospective_evaluation else None,
        tier_build_pooled_candidate=bool(args.tier_build_pooled_candidate),
        draft_trust_root=args.draft_trust_root.expanduser().resolve() if args.draft_trust_root else None,
        draft_trust_root_sha256=args.draft_trust_root_sha256,
        draft_folds_root=args.draft_folds_root.expanduser().resolve() if args.draft_folds_root else None,
        draft_public_pack_root=args.draft_public_pack_root.expanduser().resolve() if args.draft_public_pack_root else None,
        draft_manifest_sha256=args.draft_manifest_sha256,
        draft_authority_path=args.draft_authority_path.expanduser().resolve() if args.draft_authority_path else None,
        draft_authority_sha256=args.draft_authority_sha256,
        draft_model_artifact=args.draft_model_artifact.expanduser().resolve() if args.draft_model_artifact else None,
        draft_model_artifact_sha256=args.draft_model_artifact_sha256,
        draft_strict_atom=args.draft_strict_atom.expanduser().resolve() if args.draft_strict_atom else None,
        draft_strict_atom_sha256=args.draft_strict_atom_sha256,
        draft_strict_atom_code_sha256=args.draft_strict_atom_code_sha256,
        draft_strict_form=args.draft_strict_form.expanduser().resolve() if args.draft_strict_form else None,
        draft_strict_form_sha256=args.draft_strict_form_sha256,
        draft_strict_form_code_sha256=args.draft_strict_form_code_sha256,
        draft_strict_fold_root=args.draft_strict_fold_root.expanduser().resolve() if args.draft_strict_fold_root else None,
        workers=int(args.workers),
        max_log_bytes=int(args.max_log_bytes),
    )
    return config, bool(args.resume), bool(args.plan_only)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        raw_argv = list(sys.argv[1:] if argv is None else argv)
        if "--snapshot-comparison-worker" in raw_argv:
            return _snapshot_comparison_worker(raw_argv)
        if "--final-fit-manifest-worker" in raw_argv:
            return _final_fit_manifest_worker(raw_argv)
        if "--scaling-online-worker" in raw_argv:
            return _scaling_online_worker(raw_argv)
        if "--source-freeze-worker" in raw_argv:
            return _source_freeze_worker(raw_argv)
        if "--nested-selection-bundle-worker" in raw_argv:
            return _nested_selection_bundle_worker(raw_argv)
        if "--snapshot-capability-manifest-worker" in raw_argv:
            return _snapshot_capability_manifest_worker(raw_argv)
        if "--tier-shadow-fourway-worker" in raw_argv:
            return _tier_shadow_fourway_worker(raw_argv)
        config, resume, plan_only = _parse_args(argv)
        result = run(config, resume=resume, plan_only=plan_only)
    except DownstreamRunError as error:
        print(f"future-value downstream run failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY",
    "DownstreamRunError",
    "Job",
    "ResolvedInputs",
    "RunConfig",
    "SCHEMA_VERSION",
    "Stage",
    "VARIANTS",
    "build_stage_plan",
    "main",
    "run",
]
