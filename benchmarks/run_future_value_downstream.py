"""Run the closed downstream stages for a completed four-way experiment.

The runner is a research receipt coordinator.  It binds one completed
``run_future_value_fourway`` result to a caller-selected variant, then runs
the available downstream producers.  A variant is never selected from a
metric.  Missing producer inputs create a receipt blocker.  The runner does
not publish a pack and it never changes an authority flag.
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
        if int(raw.get("bytes") or -1) != path.stat().st_size or str(raw.get("sha256")) != _sha256_path(path):
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
    receipt_parent = source_receipt_path.parent.resolve()
    source_root = source_root.resolve()
    for label, name in expected_names.items():
        record = records.get(label)
        if not isinstance(record, Mapping):
            raise DownstreamRunError(f"source receipt binding is missing: {label}")
        locator = record.get("locator")
        if isinstance(locator, str) and locator.strip():
            rel = Path(locator)
            if rel.is_absolute() or ".." in rel.parts:
                raise DownstreamRunError(f"source receipt locator is unsafe: {label}")
            bound = (receipt_parent / rel).resolve()
        else:
            bound = Path(str(record.get("path") or "")).expanduser().resolve()
        path = (source_root / name).resolve()
        if bound != path or path.is_symlink() or not path.is_file():
            raise DownstreamRunError(f"source {label} file does not match source root")
        if int(record.get("bytes") or -1) != path.stat().st_size or str(record.get("sha256") or "").lower() != _sha256_path(path):
            raise DownstreamRunError(f"source {label} file hash changed")


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
    _source_paths_from_receipt(source_root, source_receipt_path, source)
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
        or int(config_receipt.get("bytes") or -1) != source_receipt_path.stat().st_size
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
        path = fourway_root / "stages" / "evaluations" / variant / "model.json"
        runtime = fourway_root / "stages" / "evaluations" / variant / "runtime.json"
        if not path.is_file() or path.is_symlink() or not runtime.is_file() or runtime.is_symlink():
            raise DownstreamRunError(f"fourway evaluation outputs are incomplete: {variant}")
        for candidate in (path, runtime):
            record = eval_outputs.get(str(candidate))
            if record is None or int(record.get("bytes") or -1) != candidate.stat().st_size or str(record.get("sha256") or "").lower() != _sha256_path(candidate):
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
        if tuple(sorted(ids)) != eligible:
            raise DownstreamRunError(
                f"fourway {variant} prediction ledger identity differs from eligible census"
            )
        if ledger.get("game_identity_sha256") != identity_sha256(ids):
            raise DownstreamRunError(f"fourway {variant} prediction ledger identity changed")
        eval_paths[variant] = path
        runtime_paths[variant] = runtime

    uncertainty_path = fourway_root / "stages" / "paired-uncertainty" / "paired-uncertainty.json"
    uncertainty_csv = fourway_root / "stages" / "paired-uncertainty" / "paired-uncertainty.csv"
    uncertainty_outputs = {str(record.get("path")): record for record in uncertainty_stage.get("outputs", []) if isinstance(record, Mapping)}
    record = uncertainty_outputs.get(str(uncertainty_path))
    if record is None or int(record.get("bytes") or -1) != uncertainty_path.stat().st_size or str(record.get("sha256") or "").lower() != _sha256_path(uncertainty_path):
        raise DownstreamRunError("paired uncertainty output hash changed")
    uncertainty_csv_path: Path | None = None
    csv_record = uncertainty_outputs.get(str(uncertainty_csv))
    if csv_record is None or not uncertainty_csv.is_file() or uncertainty_csv.is_symlink():
        raise DownstreamRunError("paired uncertainty CSV output is missing")
    if int(csv_record.get("bytes") or -1) != uncertainty_csv.stat().st_size or str(csv_record.get("sha256") or "").lower() != _sha256_path(uncertainty_csv):
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
        payload = variants.get("future_player_form")
        if not isinstance(payload, Mapping) or not isinstance(payload.get("folds"), list):
            blockers.append("nested_selection_future_player_form_missing")
    elif value.get("variant") != "future_player_form" or not isinstance(value.get("folds"), list):
        blockers.append("nested_selection_future_player_form_missing")
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


def _core_stage_plan(config: RunConfig, inputs: ResolvedInputs) -> tuple[Stage, ...]:
    source = inputs.source_root
    receipt = inputs.source_receipt
    current = _root(config, "current-rating-trust")
    final = _root(config, "final-fit")
    team = _root(config, "future-team-context")
    snapshots = _root(config, "snapshots")
    comparison = _root(config, "snapshot-comparison")
    current_receipt = current / "current-rating-ledger-receipt.json"
    current_artifact = current / "current-rating-ledger.parquet"
    current_snapshot_receipt = current / "current-rating-snapshot-receipt-v1.json"
    model = final / "final-v2-model.json"
    model_receipt = final / "final-v2-model-receipt.json"
    nested_blockers: list[str] = []
    nested = config.nested_selection
    nested_hash = config.nested_selection_sha256
    if nested is None or nested_hash is None:
        nested_blockers.append("nested_selection_input_missing")
    elif not nested.is_file() or nested.is_symlink():
        nested_blockers.append("nested_selection_input_missing")
    elif config.selected_variant != "future_player_form":
        nested_blockers.append("final_fit_builder_supports_future_player_form_only")
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
    final_job = Job(
        name="final_fit",
        command=_python_module(
            "benchmarks.build_future_value_final_fit",
            "--source-root", source,
            "--source-receipt", receipt,
            "--current-root", current,
            "--evaluation", inputs.evaluation_paths[config.selected_variant],
            *( ("--baseline-cache", config.baseline_cache) if config.baseline_cache is not None else () ),
            "--source-receipt-sha256", inputs.source_receipt_file_sha256,
            "--current-receipt-sha256", _digest_or_placeholder(current_receipt, "current-receipt-file"),
            "--current-artifact-sha256", _digest_or_placeholder(current_artifact, "current-artifact"),
            "--nested-selection", nested or "",
            "--nested-selection-sha256", nested_hash or "__NESTED_SELECTION_SHA256__",
            "--output-dir", final,
        ),
        output_roots=(final,),
        expected_files=(model, model_receipt, final / "final-fit-run.json"),
        input_paths=(source / "maps.parquet", source / "oe_player_games.parquet", source / "oe_team_games.parquet", receipt, current_artifact, current_receipt, inputs.evaluation_paths[config.selected_variant], *( (nested,) if nested is not None else () )),
    )
    final_stage = Stage(
        name="final_fit",
        jobs=() if nested_blockers else (final_job,),
        output_roots=(final,),
        expected_files=(model, model_receipt, final / "final-fit-run.json"),
        blockers=tuple(sorted(set(nested_blockers))),
    )
    dependent_blockers = tuple(sorted(set(nested_blockers)))
    team_stage = Stage(
        name="future_team_context",
        jobs=() if dependent_blockers else (Job(
            name="future_team_context",
            command=_python_module(
                "benchmarks.build_future_team_context",
                "--source-root", source,
                "--source-receipt", receipt,
                "--source-receipt-sha256", inputs.source_receipt_file_sha256,
                "--model-receipt", model,
                "--model-receipt-sha256", _digest_or_placeholder(model_receipt, "model-receipt-file"),
                "--model-artifact", model,
                "--model-artifact-sha256", _digest_or_placeholder(model, "model-artifact"),
                "--output-root", team,
            ),
            output_roots=(team,),
            expected_files=(team / "future-team-context.json", team / "future-team-context-receipt.json", team / "manifest.json"),
            input_paths=(source / "maps.parquet", source / "oe_player_games.parquet", source / "oe_team_games.parquet", receipt, model, model_receipt),
        ),),
        output_roots=(team,),
        expected_files=(team / "future-team-context.json", team / "future-team-context-receipt.json", team / "manifest.json"),
        blockers=dependent_blockers,
    )
    snapshot_stage = Stage(
        name="snapshots",
        jobs=() if dependent_blockers else (Job(
            name="snapshots",
            command=_python_module(
                "benchmarks.build_future_value_snapshots",
                "--source-root", source,
                "--source-receipt", receipt,
                "--source-receipt-sha256", inputs.source_receipt_file_sha256,
                "--current-root", current,
                "--current-receipt", current / "current-rating-snapshot-receipt-v1.json",
                "--current-receipt-sha256", _digest_or_placeholder(current_snapshot_receipt, "current-snapshot-receipt-file"),
                "--model-receipt", model_receipt,
                "--model-artifact", model,
                "--output-root", snapshots,
            ),
            output_roots=(snapshots,),
            expected_files=(snapshots / "future-value-snapshot-receipt.json", snapshots / "future-player-rank-diffs.json", snapshots / "future-team-rank-diffs.json", snapshots / "manifest.json"),
            input_paths=(source / "maps.parquet", source / "oe_player_games.parquet", source / "oe_team_games.parquet", receipt, current_snapshot_receipt, current / "player/player_ratings_snapshot.parquet", current / "team/ratings_snapshot.parquet", model, model_receipt),
        ),),
        output_roots=(snapshots,),
        expected_files=(snapshots / "future-value-snapshot-receipt.json", snapshots / "future-player-rank-diffs.json", snapshots / "future-team-rank-diffs.json", snapshots / "manifest.json"),
        blockers=dependent_blockers,
    )
    comparison_blockers: tuple[str, ...] = ()
    try:
        from benchmarks.build_future_value_snapshot_comparison import (
            TRUSTED_V15_INPUT_HASHES,
            TRUSTED_V15_SOURCE_RECEIPT_SHA256,
        )

        comparison_inputs = {
            "current_receipt": current_snapshot_receipt,
            "future_receipt": snapshots / "future-value-snapshot-receipt.json",
            "player_rank_diffs": snapshots / "future-player-rank-diffs.json",
            "team_rank_diffs": snapshots / "future-team-rank-diffs.json",
        }
        if not all(path.is_file() and not path.is_symlink() for path in comparison_inputs.values()):
            comparison_blockers = ("snapshot_comparison_inputs_not_available",)
        elif any(_sha256_path(path) != TRUSTED_V15_INPUT_HASHES[key] for key, path in comparison_inputs.items()):
            comparison_blockers = ("snapshot_comparison_cli_pinned_v15_inputs",)
        elif inputs.source_receipt_sha256 != TRUSTED_V15_SOURCE_RECEIPT_SHA256:
            comparison_blockers = ("snapshot_comparison_cli_pinned_v15_source",)
    except Exception:
        comparison_blockers = ("snapshot_comparison_builder_contract_unavailable",)
    comparison_stage = Stage(
        name="snapshot_comparison",
        jobs=(Job(
            name="snapshot_comparison",
            command=_python_module(
                "benchmarks.build_future_value_snapshot_comparison",
                "--current-receipt", current_snapshot_receipt,
                "--future-receipt", snapshots / "future-value-snapshot-receipt.json",
                "--player-rank-diffs", snapshots / "future-player-rank-diffs.json",
                "--team-rank-diffs", snapshots / "future-team-rank-diffs.json",
                "--output", comparison / "snapshot-comparison.json",
            ),
            output_roots=(comparison,),
            expected_files=(comparison / "snapshot-comparison.json",),
            input_paths=(current_snapshot_receipt, snapshots / "future-value-snapshot-receipt.json", snapshots / "future-player-rank-diffs.json", snapshots / "future-team-rank-diffs.json"),
        ),),
        output_roots=(comparison,),
        expected_files=(comparison / "snapshot-comparison.json",),
        blockers=comparison_blockers,
    )
    return (current_stage, final_stage, team_stage, snapshot_stage, comparison_stage)


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
            evaluation_paths={variant: root / "stages/evaluations" / variant / "model.json" for variant in VARIANTS},
            evaluation_runtime_paths={variant: root / "stages/evaluations" / variant / "runtime.json" for variant in VARIANTS},
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
                "command": list(job.command),
                "output_roots": [str(path) for path in job.output_roots],
                "expected_files": [str(path) for path in job.expected_files],
                "input_paths": [str(path) for path in job.input_paths],
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
        if int(raw.get("bytes") or -1) != path.stat().st_size or str(raw.get("sha256") or "").lower() != _sha256_path(path):
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
        if int(raw.get("bytes") or -1) != path.stat().st_size or str(raw.get("sha256") or "").lower() != _sha256_path(path):
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
            if int(raw.get("bytes") or -1) != path.stat().st_size or str(raw.get("sha256") or "").lower() != _sha256_path(path):
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
    if input_changed:
        blockers.append("input_changed_during_execution")
    if missing:
        blockers.append("expected_output_missing")
    status = "completed" if not blockers else "blocked"
    return _write_stage_receipt(config, stage, status=status, blockers=blockers, jobs=results, inputs=inputs, outputs=outputs)


def _write_selection(config: RunConfig, inputs: ResolvedInputs, source: Mapping[str, Any]) -> dict[str, Any]:
    root = _root(config, "selection")
    _ensure_empty(root)
    selected_payload = {
        "schema_version": SELECTION_SCHEMA_VERSION,
        "status": "caller_selected_research_only",
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
        payload = {
            "schema_version": EVALUATION_RECEIPT_SCHEMA_VERSION,
            "status": "research_only",
            "variant": variant,
            "source": selected_payload["source"],
            "artifact": _file_record(artifact),
            "runtime": _file_record(inputs.evaluation_runtime_paths[variant]),
            "authority": dict(AUTHORITY),
            "model_status": model.get("variants", {}).get(variant, {}).get("status") if isinstance(model.get("variants"), Mapping) else None,
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
    if value.get("authority") != AUTHORITY:
        raise DownstreamRunError("selected variant authority changed")
    return value


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
    required: list[Path] = list(inputs.evaluation_paths.values()) + list(selection["receipt_paths"].values())
    required.extend(
        (
            inputs.paired_uncertainty,
            _root(config, "snapshots") / "future-value-snapshot-receipt.json",
            _root(config, "snapshots") / "manifest.json",
            _root(config, "snapshot-comparison") / "snapshot-comparison.json",
        )
    )
    blockers = list(previous_blockers)
    tier_diff = _root(config, "tier-diff") / "current-v2-full-census-tier-diff.json"
    tier_receipt = _root(config, "tier-diff") / "current-v2-full-census-tier-diff-receipt.json"
    draft_report = _root(config, "draft-score") / "fourway-report.json"
    required.extend((tier_diff, tier_receipt, draft_report))
    if any(not path.is_file() or path.is_symlink() for path in required):
        blockers.append("downstream_impact_required_input_missing")
    eval_args: list[str] = []
    for variant in VARIANTS:
        eval_args.extend(("--evaluation", f"{variant}={inputs.evaluation_paths[variant]}"))
        eval_args.extend(("--evaluation-receipt", f"{variant}={selection['receipt_paths'][variant]}"))
    command = _python_module(
        "benchmarks.build_future_value_downstream_impact",
        "--source-receipt", inputs.source_receipt,
        *eval_args,
        "--snapshot-receipt", _root(config, "snapshots") / "future-value-snapshot-receipt.json",
        "--snapshot-comparison", _root(config, "snapshot-comparison") / "snapshot-comparison.json",
        "--snapshot-manifest", _root(config, "snapshots") / "manifest.json",
        "--tier-diff", tier_diff,
        "--tier-receipt", tier_receipt,
        "--draft-score-report", draft_report,
        "--paired-uncertainty", inputs.paired_uncertainty,
        "--output", output,
    )
    return Stage(name="downstream_impact", jobs=() if blockers else (Job(name="downstream_impact", command=command, output_roots=(root,), expected_files=(output,), input_paths=tuple(required)),), output_roots=(root,), expected_files=(output,), blockers=tuple(sorted(set(blockers))))


def _optional_stage_plan(config: RunConfig, inputs: ResolvedInputs) -> tuple[Stage, ...]:
    final = _root(config, "final-fit")
    model = final / "final-v2-model.json"
    model_receipt = final / "final-v2-model-receipt.json"
    current = _root(config, "current-rating-trust")
    stage_list: list[Stage] = []

    tier_root = _root(config, "tier-shadow")
    tier_available = config.tier_source_root is not None
    tier_blockers: list[str] = []
    if not tier_available:
        tier_blockers.append("tier_shadow_exact_inputs_missing")
    elif any(
        not path.is_file() or path.is_symlink()
        for path in (
            config.tier_source_root / "source" / "oe_player_games.parquet",
            config.tier_source_root / "source" / "meta.json",
        )
    ):
        tier_blockers.append("tier_shadow_source_files_missing")
    if config.tier_build_pooled_candidate and (
        config.tier_trust_manifest is None or config.tier_trust_manifest_sha256 is None
    ):
        tier_blockers.append("tier_shadow_pooled_trust_input_missing")
    tier_expected_files = [
        tier_root / "v2-tier-offset-ledger.json",
        tier_root / "v2-tier-offset-ledger-receipt.json",
        tier_root / "run-receipt.json",
    ]
    if config.tier_build_pooled_candidate:
        tier_expected_files.append(tier_root / "v2-tier-candidate.json")
    tier_command = _python_module(
        "benchmarks.build_future_value_v2_tier_shadow",
        "--repository-root", config.tier_repository_root or config.repository_root,
        "--source-root", config.tier_source_root or inputs.freeze_root,
        "--source-receipt", inputs.source_receipt,
        "--source-receipt-file-sha256", inputs.source_receipt_file_sha256,
        "--model", model,
        "--model-sha256", _digest_or_placeholder(model, "model-artifact"),
        "--model-receipt", model_receipt,
        "--model-receipt-file-sha256", _digest_or_placeholder(model_receipt, "model-receipt-file"),
        "--run-receipt", final / "final-fit-run.json",
        "--run-receipt-sha256", _digest_or_placeholder(final / "final-fit-run.json", "final-fit-run"),
        "--current-ledger", current / "current-rating-ledger.parquet",
        "--current-ledger-sha256", _digest_or_placeholder(current / "current-rating-ledger.parquet", "current-ledger"),
        "--current-receipt", current / "current-rating-ledger-receipt.json",
        "--current-receipt-file-sha256", _digest_or_placeholder(current / "current-rating-ledger-receipt.json", "current-receipt-file"),
        "--output-root", tier_root,
    )
    if config.tier_build_pooled_candidate:
        tier_command += (
            "--build-pooled-candidate",
            "--tier-trust-manifest", config.tier_trust_manifest or "",
            "--tier-trust-manifest-sha256", config.tier_trust_manifest_sha256 or "",
        )
    tier_inputs = [
        model,
        model_receipt,
        final / "final-fit-run.json",
        current / "current-rating-ledger.parquet",
        current / "current-rating-ledger-receipt.json",
        inputs.source_receipt,
    ]
    if config.tier_trust_manifest is not None:
        tier_inputs.append(config.tier_trust_manifest)
    stage_list.append(Stage(name="tier_shadow", jobs=() if tier_blockers else (Job(name="tier_shadow", command=tier_command, output_roots=(tier_root,), expected_files=tuple(tier_expected_files), input_paths=tuple(tier_inputs)),), output_roots=(tier_root,), expected_files=tuple(tier_expected_files), blockers=tuple(sorted(set(tier_blockers)))))

    diff_root = _root(config, "tier-diff")
    diff_values = (
        config.tier_trust_manifest,
        config.tier_trust_manifest_sha256,
        config.tier_source_root,
    )
    diff_blockers = list(tier_blockers)
    if any(value is None for value in diff_values):
        diff_blockers.append("tier_diff_exact_inputs_missing")
    if not config.tier_build_pooled_candidate:
        diff_blockers.append("tier_diff_pooled_candidate_required")
    if config.tier_source_root is not None and (
        not (config.tier_source_root / "future-value-source-receipt.json").is_file()
        or (config.tier_source_root / "future-value-source-receipt.json").is_symlink()
    ):
        diff_blockers.append("tier_diff_source_receipt_missing")
    diff_command = _python_module(
        "benchmarks.future_value_tierlist_full_census_diff",
        "--trust-manifest", config.tier_trust_manifest or "",
        "--expected-trust-manifest-sha256", config.tier_trust_manifest_sha256 or "",
        "--source-root", config.tier_source_root or inputs.freeze_root,
        "--v2-candidate", tier_root / "v2-tier-candidate.json",
        "--expected-v2-candidate-sha256", _digest_or_placeholder(tier_root / "v2-tier-candidate.json", "v2-tier-candidate"),
        "--output-root", diff_root,
    )
    diff_job = Job(
        name="tier_diff",
        command=diff_command,
        output_roots=(diff_root,),
        expected_files=(
            diff_root / "current-v2-full-census-tier-diff.json",
            diff_root / "current-v2-full-census-tier-diff-receipt.json",
        ),
        input_paths=tuple(
            path
            for path in (
                tier_root / "v2-tier-candidate.json",
                config.tier_trust_manifest,
                config.tier_source_root / "future-value-source-receipt.json" if config.tier_source_root is not None else None,
                config.tier_baseline_candidate,
                config.tier_production_manifest,
                config.tier_prospective_evaluation,
            )
            if path is not None
        ),
    )
    stage_list.append(
        Stage(
            name="tier_diff",
            jobs=() if diff_blockers else (diff_job,),
            output_roots=(diff_root,),
            expected_files=(
                diff_root / "current-v2-full-census-tier-diff.json",
                diff_root / "current-v2-full-census-tier-diff-receipt.json",
            ),
            blockers=tuple(sorted(set(diff_blockers))),
        )
    )

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
        "--source-receipt", inputs.source_receipt,
        "--expected-source-receipt-sha256", inputs.source_receipt_file_sha256,
        "--trust-root", config.draft_trust_root or "",
        "--expected-trust-root-sha256", config.draft_trust_root_sha256 or "",
        "--source-root", inputs.source_root,
        "--folds-root", config.draft_folds_root or "",
        "--evaluation-root", inputs.fourway_root / "stages/evaluations",
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
    draft_inputs = [inputs.source_receipt, *inputs.evaluation_paths.values()]
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
        expected_status = "research_only_complete" if not blockers else "research_only_blocked"
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
        expected_blockers = sorted({str(value) for value in blockers if str(value).strip()})
        if existing.get("blockers") != expected_blockers:
            raise DownstreamRunError("downstream final blocker list changed")
        return existing
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only_complete" if not blockers else "research_only_blocked",
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
        "blockers": sorted({str(value) for value in blockers if str(value).strip()}),
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
    receipts: list[dict[str, Any]] = []
    blockers: list[str] = []
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
        receipts.append(_write_stage_receipt(config, selection_stage, status="completed", outputs=outputs, inputs=selection_inputs))
    # Rebuild the plan after each dependency.  Child CLIs receive the raw
    # hashes of receipts produced by the preceding stage.
    failed_core_stages: set[str] = set()
    core_dependencies = {
        "final_fit": {"current_rating_trust"},
        "future_team_context": {"current_rating_trust", "final_fit"},
        "snapshots": {"current_rating_trust", "final_fit"},
        "snapshot_comparison": {"current_rating_trust", "final_fit", "snapshots"},
    }
    for stage_name in ("current_rating_trust", "final_fit", "future_team_context", "snapshots", "snapshot_comparison"):
        stage = next(item for item in build_stage_plan(config, inputs) if item.name == stage_name)
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

    for stage in _optional_stage_plan(config, inputs):
        if stage.name in {"tier_shadow", "tier_diff"}:
            stage = _blocked_stage(
                stage,
                [
                    f"dependency_{name}_blocked"
                    for name in ("current_rating_trust", "final_fit")
                    if name in failed_core_stages
                ],
            )
        try:
            receipt = _execute_stage(config, stage, resume=resume)
        except DownstreamRunError as error:
            blockers.append(f"{stage.name}:{error}")
            receipt = _write_stage_receipt(config, stage, status="blocked", blockers=(str(error),)) if not _stage_receipt_path(config, stage).exists() else _validate_stage_receipt(config, stage)
        receipts.append(receipt)
        blockers.extend(str(value) for value in receipt.get("blockers", []))

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
