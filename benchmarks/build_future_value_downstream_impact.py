"""Build a closed downstream impact report for the four rating variants.

This command consumes source-bound research artifacts.  It does not fit a
model, write a public pack, or change a production data file.  The report
keeps measured changes separate from public authority.  A report can contain
useful measurements while every public-change flag stays false.

The input contract is deliberately explicit:

* one canonical accepted-source receipt;
* one evaluation model file and optional receipt for each rating variant;
* one snapshot capability bundle for each rating variant;
* a four-way full-census Tier comparison and its optional receipt;
* the chronological four-way Tier report;
* a four-way Draft Score report; and
* a series-cluster bootstrap receipt for paired metric deltas.

The bootstrap input is optional at the Python API boundary so a partial report
can be inspected during development.  The CLI records a blocker when it is
absent.  All paths are regular files.  Every supplied byte binding is checked
before values are used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SCHEMA_VERSION = "scryglass:future-value-downstream-impact:v1"
SOURCE_SCHEMA_VERSION = "scryglass:future-value-rating-source:v1"
EVALUATION_SCHEMA_VERSION = "scryglass:future-value-four-variant-evaluation:v1"
SNAPSHOT_RECEIPT_SCHEMA_VERSION = "scryglass:future-value-snapshot-receipt:v1"
SNAPSHOT_COMPARISON_SCHEMA_VERSION = "scryglass:future-value-snapshot-comparison:v1"
SNAPSHOT_CAPABILITY_SCHEMA_VERSION = "scryglass:future-value-snapshot-capability:v1"
# c44b5019 uses the capability schema for its all-variant manifest.  The
# bundle spelling is accepted for older runner output during resume checks.
SNAPSHOT_CAPABILITY_MANIFEST_SCHEMA_VERSION = SNAPSHOT_CAPABILITY_SCHEMA_VERSION
SNAPSHOT_CAPABILITY_BUNDLE_MANIFEST_SCHEMA_VERSION = "scryglass:future-value-snapshot-capability-manifest:v1"
TIER_DIFF_SCHEMA_VERSION = "scryglass:future-value-tierlist-full-census-diff:v1"
TIER_FOURWAY_SCHEMA_VERSION = "scryglass:future-value-tierlist-full-census-fourway:v1"
TIER_CHRONOLOGICAL_SCHEMA_VERSION = "scryglass:future-value-tierlist-fourway:v1"
DRAFT_SCHEMA_VERSION = "scryglass:future-value-draft-score-fourway:v1"
DRAFT_SCHEMA_V1 = "scryglass:future-value-draft-score-fourway:v1"
DRAFT_SCHEMA_V2 = "scryglass:future-value-draft-score-fourway:v2"
BOOTSTRAP_SCHEMA_VERSION = "scryglass:future-value-paired-uncertainty:v1"
VARIANTS = ("current_only", "future_player_form", "scaling_curve", "both")
BOOTSTRAP_COMPARISONS = (
    "future_player_form_vs_current_only",
    "scaling_curve_vs_current_only",
    "both_vs_current_only",
    "both_vs_future_player_form",
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class DownstreamImpactError(ValueError):
    """Raised when the report cannot prove an input binding."""


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
        raise DownstreamImpactError("input is not canonical JSON") from error


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise DownstreamImpactError(f"cannot read input: {path}") from error
    return digest.hexdigest()


def _safe_file(value: Path | str, label: str) -> Path:
    """Return an absolute regular file and reject symlink path components."""

    path = Path(value).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise DownstreamImpactError(f"{label} must be absolute and contain no '..'")
    current = Path(path.anchor)
    try:
        for part in path.parts[1:]:
            current = current / part
            if current.is_symlink():
                raise DownstreamImpactError(f"{label} contains a symlink")
    except OSError as error:
        raise DownstreamImpactError(f"cannot inspect {label}") from error
    if not path.is_file() or path.is_symlink():
        raise DownstreamImpactError(f"{label} is missing or unsafe: {path}")
    return path


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DownstreamImpactError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise DownstreamImpactError(f"{label} must be a JSON object")
    return dict(value)


def _hash_self(value: Mapping[str, Any], field: str) -> str:
    claimed = value.get(field)
    if not isinstance(claimed, str) or SHA256_RE.fullmatch(claimed) is None:
        raise DownstreamImpactError(f"{field} is missing or invalid")
    unsigned = dict(value)
    unsigned.pop(field, None)
    expected = hashlib.sha256(_canonical(unsigned)).hexdigest()
    if expected != claimed.lower():
        raise DownstreamImpactError(f"{field} does not match the JSON payload")
    return claimed.lower()


def _authority_blockers(value: Any, label: str) -> list[str]:
    if not isinstance(value, Mapping) or value.get("research_only") is not True:
        return [f"{label}_research_only_authority_missing"]
    return [
        f"{label}_authority_{key}_enabled"
        for key, flag in value.items()
        if key != "research_only" and flag is True
    ]


def _source_fields(receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_as_of": receipt["source_as_of"],
        "source_game_count": int(receipt["source_game_count"]),
        "source_identity_sha256": str(receipt["source_identity_sha256"]),
        "source_receipt_sha256": str(receipt["receipt_sha256"]),
        "model_eligible_game_count": int(receipt["model_eligible_game_count"]),
        "model_eligible_identity_sha256": str(
            receipt["model_eligible_identity_sha256"]
        ),
    }


def _source_match(value: Mapping[str, Any], source: Mapping[str, Any], label: str) -> list[str]:
    """Check all source fields present in a nested artifact binding."""

    aliases = {
        "source_receipt_sha256": ("source_receipt_sha256",),
        "source_as_of": ("source_as_of",),
        "source_game_count": ("source_game_count", "accepted_game_count"),
        "source_identity_sha256": (
            "source_identity_sha256",
            "accepted_identity_sha256",
        ),
        "model_eligible_game_count": (
            "model_eligible_game_count",
            "eligible_game_count",
        ),
        "model_eligible_identity_sha256": (
            "model_eligible_identity_sha256",
            "eligible_identity_sha256",
        ),
    }
    blockers: list[str] = []
    for field, names in aliases.items():
        present = next((name for name in names if name in value), None)
        if present is not None and value[present] != source[field]:
            blockers.append(f"{label}_{field}_mismatch")
    if "accepted_game_ids" in value:
        raw_ids = value.get("accepted_game_ids")
        if raw_ids != source.get("accepted_game_ids"):
            blockers.append(f"{label}_accepted_game_ids_mismatch")
    if "model_eligible_game_ids" in value:
        raw_ids = value.get("model_eligible_game_ids")
        if raw_ids != source.get("model_eligible_game_ids"):
            blockers.append(f"{label}_model_eligible_game_ids_mismatch")
    return blockers


def _load_source(path_value: Path | str) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = _safe_file(path_value, "source receipt")
    receipt = _read_json(path, "source receipt")
    try:
        accepted_ids, eligible_ids = validate_future_value_source_receipt_payload(receipt)
    except Exception as error:
        raise DownstreamImpactError("source receipt failed canonical validation") from error
    if receipt.get("schema_version") != SOURCE_SCHEMA_VERSION:
        raise DownstreamImpactError("source receipt schema is not the canonical rating source")
    file_sha = _sha256_file(path)
    source = _source_fields(receipt)
    summary = {
        **source,
        "accepted_game_ids": list(accepted_ids),
        "model_eligible_game_ids": list(eligible_ids),
        "accepted_game_count": len(accepted_ids),
        "accepted_identity_sha256": identity_sha256(accepted_ids),
        "model_eligible_game_count": len(eligible_ids),
        "model_eligible_identity_sha256": identity_sha256(eligible_ids),
        "source_receipt_file_sha256": file_sha,
    }
    return path, receipt, summary


def _artifact_binding(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _sha256_file(path)}


def _verify_file_record(record: Any, label: str) -> tuple[Path, dict[str, Any]]:
    if not isinstance(record, Mapping):
        raise DownstreamImpactError(f"{label} file record is invalid")
    raw_path = record.get("path", record.get("locator"))
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise DownstreamImpactError(f"{label} file path is missing")
    path = _safe_file(raw_path, label)
    expected_sha = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(expected_sha, str) or SHA256_RE.fullmatch(expected_sha) is None:
        raise DownstreamImpactError(f"{label} file hash is invalid")
    if isinstance(expected_bytes, bool) or not isinstance(expected_bytes, int):
        raise DownstreamImpactError(f"{label} byte count is invalid")
    actual = _artifact_binding(path)
    if actual["bytes"] != expected_bytes or actual["sha256"] != expected_sha.lower():
        raise DownstreamImpactError(f"{label} bytes changed")
    return path, actual


def _authority_summary(value: Any) -> dict[str, bool]:
    """Return a stable authority surface with every public flag disabled."""

    flags = {
        "research_only": True,
        "public_player_rating": False,
        "public_team_rating": False,
        "public_tierlist": False,
        "public_draft_score": False,
        "public_probability": False,
        "promotion": False,
        "deployment": False,
        "odds": False,
        "expected_value": False,
        "recommendation": False,
        "betting": False,
    }
    if isinstance(value, Mapping) and value.get("research_only") is not True:
        flags["research_only"] = False
    return flags


def _external_receipt(
    path_value: Path | str,
    *,
    source: Mapping[str, Any],
    artifact_path: Path,
    label: str,
) -> dict[str, Any]:
    """Verify a separate model receipt without prescribing its producer shape."""

    path = _safe_file(path_value, f"{label} receipt")
    receipt = _read_json(path, f"{label} receipt")
    bindings: list[Mapping[str, Any]] = []

    def walk_bindings(value: Any, depth: int = 0) -> Iterable[Mapping[str, Any]]:
        if depth > 5:
            return
        if isinstance(value, Mapping):
            if any(
                key in value
                for key in (
                    "source_receipt_sha256",
                    "source_identity_sha256",
                    "source_game_count",
                )
            ):
                yield value
            for key, child in value.items():
                if key in {"source", "source_binding", "source_receipt", "binding"}:
                    yield from walk_bindings(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                yield from walk_bindings(child, depth + 1)

    bindings.extend(walk_bindings(receipt))
    blockers = _source_match(receipt, source, label) if bindings == [] else []
    if not bindings:
        blockers.append(f"{label}_receipt_source_binding_missing")
    if bindings:
        for binding in bindings:
            blockers.extend(_source_match(binding, source, label))
    blockers.extend(_authority_blockers(receipt.get("authority"), label))
    self_hash = None
    if "receipt_sha256" in receipt:
        self_hash = _hash_self(receipt, "receipt_sha256")
    expected = _sha256_file(artifact_path)
    expected_bytes = artifact_path.stat().st_size
    found_hash = False
    found_bytes = False

    def walk(value: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(value, Mapping):
            yield value
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)

    for record in walk(receipt):
        for key in ("sha256", "artifact_sha256", "model_artifact_sha256", "file_sha256"):
            if record.get(key) == expected:
                found_hash = True
        for key in ("bytes", "artifact_bytes", "file_bytes"):
            if record.get(key) == expected_bytes:
                found_bytes = True
    if not found_hash:
        blockers.append(f"{label}_receipt_artifact_hash_missing")
    if not found_bytes:
        blockers.append(f"{label}_receipt_artifact_bytes_missing")
    if blockers:
        raise DownstreamImpactError("; ".join(sorted(set(blockers))))
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "receipt_sha256": self_hash,
    }


def _verify_evaluation(
    path_value: Path | str,
    *,
    variant: str,
    source: Mapping[str, Any],
    receipt_path: Path | str | None,
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    path = _safe_file(path_value, f"{variant} evaluation")
    model = _read_json(path, f"{variant} evaluation")
    blockers: list[str] = []
    if model.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        blockers.append(f"{variant}_evaluation_schema_invalid")
    model_source = model.get("source")
    if not isinstance(model_source, Mapping):
        blockers.append(f"{variant}_evaluation_source_missing")
    else:
        blockers.extend(_source_match(model_source, source, f"{variant}_evaluation"))
        if model_source.get("source_receipt_file_sha256") not in {
            None,
            source["source_receipt_file_sha256"],
        }:
            blockers.append(f"{variant}_evaluation_source_receipt_file_mismatch")
    variants = model.get("variants")
    payload = variants.get(variant) if isinstance(variants, Mapping) else None
    if not isinstance(payload, Mapping):
        blockers.append(f"{variant}_evaluation_variant_missing")
        payload = {}
    blockers.extend(_authority_blockers(payload.get("authority"), f"{variant}_evaluation"))
    if payload.get("status") != "development_evaluated":
        blockers.append(f"{variant}_evaluation_status_not_complete")
    ledger = payload.get("prediction_ledger")
    ledger_ids: list[str] = []
    ledger_hash = None
    if not isinstance(ledger, Mapping) or not isinstance(ledger.get("rows"), list):
        blockers.append(f"{variant}_prediction_ledger_missing")
    else:
        rows = ledger["rows"]
        if ledger.get("row_count") != len(rows):
            blockers.append(f"{variant}_prediction_ledger_count_mismatch")
        try:
            ledger_hash = hashlib.sha256(_canonical(rows)).hexdigest()
        except DownstreamImpactError:
            blockers.append(f"{variant}_prediction_ledger_not_canonical")
        if ledger.get("sha256") != ledger_hash:
            blockers.append(f"{variant}_prediction_ledger_hash_mismatch")
        for row in rows:
            if not isinstance(row, Mapping) or not str(row.get("game_id") or "").strip():
                blockers.append(f"{variant}_prediction_ledger_identity_missing")
                continue
            ledger_ids.append(str(row["game_id"]))
        if len(ledger_ids) != len(set(ledger_ids)):
            blockers.append(f"{variant}_prediction_ledger_duplicate_ids")
        claimed_identity = identity_sha256(ledger_ids)
        if ledger.get("game_identity_sha256") != claimed_identity:
            blockers.append(f"{variant}_prediction_ledger_identity_hash_mismatch")
    if receipt_path is not None:
        try:
            receipt_binding = _external_receipt(
                receipt_path,
                source=source,
                artifact_path=path,
                label=f"{variant}_evaluation",
            )
        except DownstreamImpactError as error:
            blockers.append(str(error))
            receipt_binding = {"path": str(receipt_path), "status": "invalid"}
    else:
        receipt_binding = None
        blockers.append(f"{variant}_evaluation_receipt_missing")
    evaluation = payload.get("evaluation") if isinstance(payload, Mapping) else {}
    if not isinstance(evaluation, Mapping):
        evaluation = {}
        blockers.append(f"{variant}_evaluation_metrics_missing")
    pooled = evaluation.get("pooled_candidate")
    if not isinstance(pooled, Mapping):
        blockers.append(f"{variant}_pooled_metrics_missing")
        pooled = {}
    metrics = {
        key: pooled.get(key)
        for key in ("auc", "brier", "log_loss", "rows")
        if key in pooled
    }
    if any(
        key in metrics
        and (not isinstance(metrics[key], (int, float)) or not math.isfinite(float(metrics[key])))
        for key in ("auc", "brier", "log_loss")
    ):
        blockers.append(f"{variant}_pooled_metrics_non_finite")
    coverage = {
        "ledger_rows": len(ledger_ids),
        "ledger_identity_sha256": identity_sha256(ledger_ids) if ledger_ids else None,
        "pooled_rows": evaluation.get("pooled_rows"),
        "valid_folds": evaluation.get("valid_folds"),
        "requested_folds": evaluation.get("requested_folds"),
    }
    summary = {
        "variant": variant,
        "status": payload.get("status"),
        "artifact": _artifact_binding(path),
        "receipt": receipt_binding,
        "metrics": metrics,
        "raw_metrics": evaluation.get("pooled_raw_candidate"),
        "calibration": {
            key: evaluation.get("pooled_calibration", {}).get(key)
            for key in ("status", "expected_calibration_error", "max_absolute_error", "rows")
            if isinstance(evaluation.get("pooled_calibration"), Mapping)
            and key in evaluation["pooled_calibration"]
        },
        "coverage": coverage,
        "model_blockers": list(payload.get("blockers", []))
        if isinstance(payload.get("blockers"), list)
        else [],
    }
    return model, summary, sorted(set(blockers))


def _verify_bootstrap(
    path_value: Path | str | None,
    *,
    source: Mapping[str, Any],
    evaluation_summaries: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[str]]:
    if path_value is None:
        return None, ["paired_bootstrap_receipt_missing"]
    path = _safe_file(path_value, "paired bootstrap receipt")
    value = _read_json(path, "paired bootstrap receipt")
    blockers: list[str] = []
    if value.get("schema_version") != BOOTSTRAP_SCHEMA_VERSION:
        blockers.append("paired_bootstrap_schema_invalid")
    blockers.extend(_source_match(value.get("source", {}), source, "paired_bootstrap"))
    blockers.extend(_authority_blockers(value.get("authority"), "paired_bootstrap"))
    coverage = value.get("coverage")
    if not isinstance(coverage, Mapping):
        blockers.append("paired_bootstrap_coverage_missing")
    else:
        claimed_identity = coverage.get("game_identity_sha256")
        for variant, summary in evaluation_summaries.items():
            if summary.get("coverage", {}).get("ledger_identity_sha256") != claimed_identity:
                blockers.append(f"paired_bootstrap_{variant}_identity_mismatch")
            if summary.get("coverage", {}).get("ledger_rows") != coverage.get("rows"):
                blockers.append(f"paired_bootstrap_{variant}_row_count_mismatch")
    comparisons = value.get("comparisons")
    if not isinstance(comparisons, Mapping):
        blockers.append("paired_bootstrap_comparisons_missing")
    else:
        for name in BOOTSTRAP_COMPARISONS:
            item = comparisons.get(name)
            if not isinstance(item, Mapping):
                blockers.append(f"paired_bootstrap_{name}_missing")
                continue
            if int(item.get("draws_accepted", 0) or 0) <= 0:
                blockers.append(f"paired_bootstrap_{name}_has_no_accepted_draws")
            if not isinstance(item.get("metrics"), Mapping):
                blockers.append(f"paired_bootstrap_{name}_metrics_missing")
    summary = {
        "artifact": _artifact_binding(path),
        "status": value.get("status"),
        "method": value.get("method"),
        "coverage": coverage,
        "comparisons": value.get("comparisons", {}),
    }
    return summary, sorted(set(blockers))


def _verify_snapshot_bundle(
    receipt_path_value: Path | str,
    comparison_path_value: Path | str,
    manifest_path_value: Path | str | None,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    receipt_path = _safe_file(receipt_path_value, "snapshot receipt")
    comparison_path = _safe_file(comparison_path_value, "snapshot comparison")
    receipt = _read_json(receipt_path, "snapshot receipt")
    comparison = _read_json(comparison_path, "snapshot comparison")
    blockers: list[str] = []
    if receipt.get("schema_version") != SNAPSHOT_RECEIPT_SCHEMA_VERSION:
        blockers.append("snapshot_receipt_schema_invalid")
    blockers.extend(_source_match(receipt.get("source", {}), source, "snapshot_receipt"))
    current_inputs = receipt.get("current_rating_inputs", {})
    blockers.extend(_source_match(current_inputs, source, "snapshot_current_rating"))
    if isinstance(current_inputs, Mapping):
        current_snapshots = current_inputs.get("snapshots")
        if not isinstance(current_snapshots, Mapping):
            blockers.append("snapshot_current_rating_snapshots_missing")
        else:
            for kind, record in current_snapshots.items():
                try:
                    _verify_file_record(record, f"snapshot current {kind}")
                except DownstreamImpactError as error:
                    blockers.append(str(error))
        current_receipt = current_inputs.get("receipt")
        if current_receipt is not None:
            try:
                _verify_file_record(current_receipt, "snapshot current trust receipt")
            except DownstreamImpactError as error:
                blockers.append(str(error))
    blockers.extend(_authority_blockers(receipt.get("authority"), "snapshot_receipt"))
    try:
        receipt_hash = _hash_self(receipt, "receipt_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        receipt_hash = None
    if comparison.get("schema_version") != SNAPSHOT_COMPARISON_SCHEMA_VERSION:
        blockers.append("snapshot_comparison_schema_invalid")
    blockers.extend(_source_match(comparison.get("source", {}), source, "snapshot_comparison"))
    independent_join = comparison.get("independent_join")
    if isinstance(independent_join, Mapping):
        blockers.extend(
            _source_match(
                independent_join.get("current_snapshot_trust_root", {}),
                source,
                "snapshot_current_trust_root",
            )
        )
    blockers.extend(_authority_blockers(comparison.get("authority"), "snapshot_comparison"))
    try:
        comparison_hash = _hash_self(comparison, "report_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        comparison_hash = None
    if comparison.get("independent_join", {}).get("status") != "verified":
        blockers.append("snapshot_independent_join_not_verified")
    root = receipt_path.parent
    manifest_path = (
        _safe_file(manifest_path_value, "snapshot manifest")
        if manifest_path_value is not None
        else _safe_file(root / "manifest.json", "snapshot manifest")
    )
    manifest = _read_json(manifest_path, "snapshot manifest")
    if manifest.get("schema_version") != "scryglass:future-value-snapshot:v1":
        blockers.append("snapshot_manifest_schema_invalid")
    blockers.extend(_source_match(manifest, source, "snapshot_manifest"))
    blockers.extend(_authority_blockers(manifest.get("authority"), "snapshot_manifest"))
    try:
        manifest_hash = _hash_self(manifest, "manifest_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        manifest_hash = None
    files = manifest.get("files")
    file_bindings: dict[str, Any] = {}
    if not isinstance(files, Mapping):
        blockers.append("snapshot_manifest_files_missing")
    else:
        for name, record in files.items():
            try:
                file_path, binding = _verify_file_record(record, f"snapshot {name}")
            except DownstreamImpactError as error:
                blockers.append(str(error))
                continue
            file_bindings[str(name)] = binding
            if name == "receipt" and file_path != receipt_path:
                blockers.append("snapshot_manifest_receipt_path_mismatch")
            if name in {"player_snapshot", "team_snapshot"}:
                try:
                    payload = _read_json(file_path, f"snapshot {name}")
                except DownstreamImpactError as error:
                    blockers.append(str(error))
                else:
                    blockers.extend(_source_match(payload, source, f"snapshot_{name}"))
                    blockers.extend(_authority_blockers(payload.get("authority"), f"snapshot_{name}"))
    # Rank-diff rows provide the exact movement counts.  The receipt and
    # comparison summary must agree with the bytes in the bundle.
    movement: dict[str, Any] = {}
    for kind, file_key, count_key in (
        ("player", "player_rank_diffs", "player_rank_diff_count"),
        ("team", "team_rank_diffs", "team_rank_diff_count"),
    ):
        record = files.get(file_key) if isinstance(files, Mapping) else None
        if not isinstance(record, Mapping):
            blockers.append(f"snapshot_{kind}_rank_diff_file_missing")
            continue
        try:
            rank_path, _ = _verify_file_record(record, f"snapshot {kind} rank diff")
            rank_data = _read_json(rank_path, f"snapshot {kind} rank diff")
        except DownstreamImpactError as error:
            blockers.append(str(error))
            continue
        blockers.extend(_source_match(rank_data, source, f"snapshot_{kind}_rank_diff"))
        rows = rank_data.get("rows")
        if not isinstance(rows, list):
            blockers.append(f"snapshot_{kind}_rank_diff_rows_missing")
            continue
        deltas: list[int] = []
        ids: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                blockers.append(f"snapshot_{kind}_rank_diff_row_invalid")
                continue
            identifier = row.get("player_id" if kind == "player" else "team_id")
            delta = row.get("rank_delta")
            if not isinstance(identifier, str) or not identifier.strip():
                blockers.append(f"snapshot_{kind}_rank_diff_identity_missing")
            else:
                ids.append(identifier)
            if isinstance(delta, bool) or not isinstance(delta, int):
                blockers.append(f"snapshot_{kind}_rank_diff_delta_invalid")
            else:
                deltas.append(delta)
        if len(ids) != len(set(ids)):
            blockers.append(f"snapshot_{kind}_rank_diff_duplicate_identity")
        if receipt.get(count_key) != len(rows):
            blockers.append(f"snapshot_{kind}_rank_diff_count_mismatch")
        coverage = receipt.get("rank_coverage", {}).get(kind, {})
        if isinstance(coverage, Mapping) and coverage.get("matched_rows") != len(rows):
            blockers.append(f"snapshot_{kind}_rank_diff_coverage_mismatch")
        full_rank_status = coverage.get("full_snapshot_rank_status") if isinstance(coverage, Mapping) else None
        if full_rank_status is None and isinstance(coverage, Mapping):
            nested_full = coverage.get("full_snapshot_ranks")
            if isinstance(nested_full, Mapping):
                full_rank_status = nested_full.get("status")
        movement[kind] = {
            "rows": len(rows),
            "changed_rank_count": sum(delta != 0 for delta in deltas),
            "mean_absolute_rank_movement": (
                sum(abs(delta) for delta in deltas) / len(deltas) if deltas else None
            ),
            "maximum_absolute_rank_movement": max((abs(delta) for delta in deltas), default=None),
            "identity_sha256": identity_sha256(ids),
            "rank_direction": coverage.get("rank_direction") if isinstance(coverage, Mapping) else None,
            "rank_universe": coverage.get("rank_universe") if isinstance(coverage, Mapping) else None,
            "full_snapshot_rank_status": full_rank_status,
        }
        if movement[kind]["full_snapshot_rank_status"] != "incomparable":
            blockers.append(f"snapshot_{kind}_full_snapshot_rank_status_not_incomparable")
    summary = {
        "receipt": {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256_file(receipt_path),
            "receipt_sha256": receipt_hash,
        },
        "comparison": {
            "path": str(comparison_path),
            "bytes": comparison_path.stat().st_size,
            "sha256": _sha256_file(comparison_path),
            "report_sha256": comparison_hash,
            "status": comparison.get("status"),
            "full_snapshot_rank_status": comparison.get("full_snapshot_rank_status"),
            "independent_join": comparison.get("independent_join"),
        },
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest_hash,
            "files": file_bindings,
        },
        "rank_movement": movement,
        "blockers_from_receipt": receipt.get("blockers", []),
        "blockers_from_comparison": comparison.get("blockers", []),
    }
    return summary, sorted(set(blockers))


def _verify_snapshot_variant(
    receipt_path_value: Path | str,
    manifest_path_value: Path | str,
    *,
    variant: str,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Verify one variant snapshot bundle and preserve typed N/A rows."""

    receipt_path = _safe_file(receipt_path_value, f"{variant} snapshot receipt")
    manifest_path = _safe_file(manifest_path_value, f"{variant} snapshot manifest")
    receipt = _read_json(receipt_path, f"{variant} snapshot receipt")
    manifest = _read_json(manifest_path, f"{variant} snapshot manifest")
    blockers: list[str] = []
    if receipt.get("schema_version") != SNAPSHOT_RECEIPT_SCHEMA_VERSION:
        blockers.append(f"{variant}_snapshot_receipt_schema_invalid")
    if receipt.get("capability_schema_version") not in {None, SNAPSHOT_CAPABILITY_SCHEMA_VERSION}:
        blockers.append(f"{variant}_snapshot_capability_schema_invalid")
    if receipt.get("variant") != variant:
        blockers.append(f"{variant}_snapshot_variant_mismatch")
    blockers.extend(_source_match(receipt.get("source", {}), source, f"{variant}_snapshot"))
    receipt_source = receipt.get("source")
    if isinstance(receipt_source, Mapping):
        for field in (
            "source_as_of",
            "source_game_count",
            "source_identity_sha256",
            "model_eligible_game_count",
            "model_eligible_identity_sha256",
            "source_receipt_sha256",
        ):
            if field not in receipt_source:
                blockers.append(f"{variant}_snapshot_{field}_missing")
    else:
        blockers.append(f"{variant}_snapshot_source_missing")
    blockers.extend(_authority_blockers(receipt.get("authority"), f"{variant}_snapshot_receipt"))
    try:
        receipt_hash = _hash_self(receipt, "receipt_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        receipt_hash = None
    if receipt.get("status") not in {"research_only", "research_only_partial", "research_only_blocked"}:
        blockers.append(f"{variant}_snapshot_status_invalid")
    if manifest.get("schema_version") != "scryglass:future-value-snapshot:v1":
        blockers.append(f"{variant}_snapshot_manifest_schema_invalid")
    if manifest.get("variant") != variant:
        blockers.append(f"{variant}_snapshot_manifest_variant_mismatch")
    blockers.extend(_authority_blockers(manifest.get("authority"), f"{variant}_snapshot_manifest"))
    try:
        manifest_hash = _hash_self(manifest, "manifest_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        manifest_hash = None
    if manifest.get("source_receipt_sha256") not in {None, source.get("source_receipt_sha256")}:
        blockers.append(f"{variant}_snapshot_manifest_source_receipt_mismatch")
    files = manifest.get("files")
    file_bindings: dict[str, dict[str, Any]] = {}
    if not isinstance(files, Mapping):
        blockers.append(f"{variant}_snapshot_manifest_files_missing")
    else:
        required_file_keys = {
            "player_snapshot",
            "team_snapshot",
            "player_rank_diffs",
            "team_rank_diffs",
            "receipt",
        }
        for missing_file in sorted(required_file_keys.difference(files)):
            blockers.append(f"{variant}_snapshot_manifest_{missing_file}_missing")
        for name, record in files.items():
            try:
                path, binding = _verify_file_record(record, f"{variant} snapshot {name}")
            except DownstreamImpactError as error:
                blockers.append(str(error))
                continue
            try:
                path.relative_to(receipt_path.parent.resolve())
            except ValueError:
                blockers.append(f"{variant}_snapshot_file_escapes_bundle")
            file_bindings[str(name)] = binding
            if name == "receipt" and path != receipt_path:
                blockers.append(f"{variant}_snapshot_manifest_receipt_path_mismatch")
    capability = receipt.get("capability")
    if not isinstance(capability, Mapping):
        capability = manifest.get("capability")
    if not isinstance(capability, Mapping):
        blockers.append(f"{variant}_snapshot_capability_missing")
        capability = {}
    expected_component_scope = {
        "current_only": "current_mu_effective",
        "future_player_form": "future_player_form_component",
        "both": "future_player_form_component",
    }.get(variant)
    for kind in ("player", "team"):
        component = capability.get(kind)
        rank_capability = capability.get(f"{kind}_ranks")
        if variant == "scaling_curve":
            if isinstance(component, Mapping) and component.get("status") != "not_applicable":
                blockers.append(f"{variant}_{kind}_snapshot_component_na_invalid")
        elif (
            not isinstance(component, Mapping)
            or component.get("status") != "available"
            or component.get("scope") != expected_component_scope
            or component.get("full_composite_rating") is True
        ):
            blockers.append(f"{variant}_{kind}_snapshot_component_scope_invalid")
        if not isinstance(rank_capability, Mapping):
            blockers.append(f"{variant}_{kind}_snapshot_rank_capability_missing")
    rank_movement: dict[str, Any] = {}
    rank_coverage = receipt.get("rank_coverage")
    if not isinstance(rank_coverage, Mapping):
        rank_coverage = {}
        blockers.append(f"{variant}_snapshot_rank_coverage_missing")
    for kind, file_key, count_key, identity_key in (
        ("player", "player_rank_diffs", "player_rank_diff_count", "player_id"),
        ("team", "team_rank_diffs", "team_rank_diff_count", "team_id"),
    ):
        coverage = rank_coverage.get(kind)
        if not isinstance(coverage, Mapping):
            coverage = {}
            blockers.append(f"{variant}_{kind}_snapshot_rank_coverage_missing")
        status = str(coverage.get("status") or "")
        if variant == "scaling_curve":
            # V3 has no intrinsic player or team value at this endpoint.  A
            # typed N/A is complete evidence and does not create a blocker.
            if status != "not_applicable" or coverage.get("row_policy") != "no_rows":
                blockers.append(f"{variant}_{kind}_snapshot_na_contract_invalid")
            rank_movement[kind] = {
                "status": "not_applicable",
                "rows": 0,
                "changed_rank_count": 0,
                "mean_absolute_rank_movement": None,
                "maximum_absolute_rank_movement": None,
                "rank_universe": coverage.get("rank_universe"),
                "scope": capability.get(f"{kind}_ranks", {}).get("scope") if isinstance(capability.get(f"{kind}_ranks"), Mapping) else None,
            }
            continue
        record = files.get(file_key) if isinstance(files, Mapping) else None
        if not isinstance(record, Mapping):
            blockers.append(f"{variant}_{kind}_snapshot_rank_diff_file_missing")
            rows: list[Any] = []
        else:
            try:
                rank_path, _ = _verify_file_record(record, f"{variant} snapshot {kind} rank diff")
                rank_data = _read_json(rank_path, f"{variant} snapshot {kind} rank diff")
                rows = rank_data.get("rows") if isinstance(rank_data.get("rows"), list) else []
                if not isinstance(rank_data.get("rows"), list):
                    blockers.append(f"{variant}_{kind}_snapshot_rank_diff_rows_missing")
                if rank_data.get("source_receipt_sha256") not in {None, source.get("source_receipt_sha256")}:
                    blockers.append(f"{variant}_{kind}_snapshot_rank_diff_source_mismatch")
            except DownstreamImpactError as error:
                blockers.append(str(error))
                rows = []
        if receipt.get(count_key) != len(rows):
            blockers.append(f"{variant}_{kind}_snapshot_rank_diff_count_mismatch")
        deltas: list[int] = []
        ids: list[str] = []
        for row in rows:
            if not isinstance(row, Mapping):
                blockers.append(f"{variant}_{kind}_snapshot_rank_diff_row_invalid")
                continue
            identity = str(row.get(identity_key) or "")
            if not identity:
                blockers.append(f"{variant}_{kind}_snapshot_rank_diff_identity_missing")
            else:
                ids.append(identity)
            delta = row.get("rank_delta")
            if isinstance(delta, bool) or not isinstance(delta, int):
                blockers.append(f"{variant}_{kind}_snapshot_rank_diff_delta_invalid")
            else:
                deltas.append(delta)
        if len(ids) != len(set(ids)):
            blockers.append(f"{variant}_{kind}_snapshot_rank_diff_duplicate_identity")
        rank_movement[kind] = {
            "status": status or "available",
            "rows": len(rows),
            "changed_rank_count": sum(delta != 0 for delta in deltas),
            "mean_absolute_rank_movement": sum(abs(delta) for delta in deltas) / len(deltas) if deltas else None,
            "maximum_absolute_rank_movement": max((abs(delta) for delta in deltas), default=None),
            "identity_sha256": identity_sha256(ids),
            "rank_universe": coverage.get("rank_universe"),
            "scope": capability.get(f"{kind}_ranks", {}).get("scope") if isinstance(capability.get(f"{kind}_ranks"), Mapping) else None,
        }
    summary = {
        "variant": variant,
        "status": receipt.get("status"),
        "capability": capability,
        "receipt": {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256_file(receipt_path),
            "receipt_sha256": receipt_hash,
        },
        "manifest": {
            "path": str(manifest_path),
            "bytes": manifest_path.stat().st_size,
            "sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest_hash,
            "files": file_bindings,
        },
        "rank_movement": rank_movement,
        "blockers_from_artifact": receipt.get("blockers", []),
    }
    return summary, sorted(set(blockers))


def _verify_tier_shadow_manifest(
    path_value: Path | str,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    path = _safe_file(path_value, "four-way Tier shadow manifest")
    value = _read_json(path, "four-way Tier shadow manifest")
    blockers: list[str] = []
    if value.get("schema_version") != "scryglass:future-value-tier-shadow-fourway:v1":
        blockers.append("tier_shadow_manifest_schema_invalid")
    blockers.extend(_authority_blockers(value.get("authority"), "tier_shadow_manifest"))
    try:
        _hash_self(value, "manifest_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
    source_binding = value.get("source")
    if not isinstance(source_binding, Mapping):
        blockers.append("tier_shadow_source_binding_missing")
    else:
        if source_binding.get("source_receipt_sha256") not in {None, source.get("source_receipt_sha256")}:
            blockers.append("tier_shadow_source_receipt_mismatch")
        if source_binding.get("source_identity_sha256") not in {None, source.get("source_identity_sha256")}:
            blockers.append("tier_shadow_source_identity_mismatch")
    variants = value.get("variants")
    if not isinstance(variants, Mapping):
        blockers.append("tier_shadow_manifest_variants_missing")
        variants = {}
    summary: dict[str, Any] = {"artifact": _artifact_binding(path), "status": value.get("status"), "variants": {}}
    for variant in VARIANTS:
        record = variants.get(variant)
        if not isinstance(record, Mapping):
            blockers.append(f"tier_shadow_{variant}_missing")
            continue
        ledger = record.get("ledger")
        receipt = record.get("receipt")
        try:
            ledger_path, ledger_file = _verify_file_record(ledger, f"tier shadow {variant} ledger")
            receipt_path, receipt_file = _verify_file_record(receipt, f"tier shadow {variant} receipt")
        except DownstreamImpactError as error:
            blockers.append(str(error))
            continue
        if record.get("game_count") != source.get("model_eligible_game_count") or record.get("game_identity_sha256") != source.get("model_eligible_identity_sha256"):
            blockers.append(f"tier_shadow_{variant}_universe_mismatch")
        summary["variants"][variant] = {"ledger": ledger_file, "receipt": receipt_file, "game_count": record.get("game_count"), "game_identity_sha256": record.get("game_identity_sha256"), "path": str(ledger_path), "receipt_path": str(receipt_path)}
    return summary, sorted(set(blockers))


def _verify_tier(
    path_value: Path | str,
    receipt_path_value: Path | str | None,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    path = _safe_file(path_value, "full-census Tier diff")
    report = _read_json(path, "full-census Tier diff")
    blockers: list[str] = []
    is_fourway = report.get("schema_version") == TIER_FOURWAY_SCHEMA_VERSION
    if report.get("schema_version") not in {TIER_DIFF_SCHEMA_VERSION, TIER_FOURWAY_SCHEMA_VERSION}:
        blockers.append("tier_diff_schema_invalid")
    blockers.extend(_authority_blockers(report.get("authority"), "tier_diff"))
    report_source = report.get("source", {})
    blockers.extend(_source_match(report_source, source, "tier_diff"))
    # The Tier producer uses accepted_* and model_eligible_* names.  Check both
    # counts and identities explicitly because the accepted and eligible
    # universes can differ.
    expected = {
        "accepted_game_count": source["source_game_count"],
        "accepted_identity_sha256": source["source_identity_sha256"],
        "model_eligible_game_count": source["model_eligible_game_count"],
        "model_eligible_identity_sha256": source["model_eligible_identity_sha256"],
        "source_receipt_sha256": source["source_receipt_sha256"],
        "source_as_of": source["source_as_of"],
    }
    for field, expected_value in expected.items():
        if field in report_source and report_source.get(field) != expected_value:
            blockers.append(f"tier_diff_{field}_mismatch")
    try:
        report_hash = _hash_self(report, "report_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        report_hash = None
    rows = report.get("rows")
    comparison = report.get("comparison")
    comparisons = report.get("comparisons") if is_fourway else None
    if is_fourway:
        if not isinstance(comparisons, Mapping) or set(comparisons) != set(VARIANTS):
            blockers.append("tier_diff_variant_comparisons_missing")
            comparisons = {}
        rows = []
        comparison = report.get("candidate_universe", {})
    elif not isinstance(rows, list) or not isinstance(comparison, Mapping):
        blockers.append("tier_diff_rows_or_comparison_missing")
        rows = []
        comparison = {}
    if not is_fourway and comparison.get("common_row_count") != len(rows):
        blockers.append("tier_diff_common_row_count_mismatch")
    keys: list[str] = []
    key_payloads: list[dict[str, str]] = []
    rank_changed = 0
    tier_changed = 0
    movements: list[int] = []
    for row in rows:
        key = row.get("key") if isinstance(row, Mapping) else None
        if not isinstance(key, Mapping):
            blockers.append("tier_diff_row_identity_missing")
            continue
        identity_fields = ("scope_id", "patch", "role", "champion_id")
        identity_values = {field: str(key.get(field, "")) for field in identity_fields}
        identity = "|".join(identity_values.values())
        if not all(identity.split("|")):
            blockers.append("tier_diff_row_identity_invalid")
        keys.append(identity)
        key_payloads.append(identity_values)
        delta = row.get("delta") if isinstance(row, Mapping) else None
        if not isinstance(delta, Mapping):
            blockers.append("tier_diff_row_delta_missing")
            continue
        rank_delta = delta.get("rank_delta")
        if isinstance(rank_delta, bool) or not isinstance(rank_delta, int):
            blockers.append("tier_diff_rank_delta_invalid")
        else:
            movements.append(rank_delta)
            rank_changed += rank_delta != 0
        tier_changed += delta.get("tier_changed") is True
    if len(keys) != len(set(keys)):
        blockers.append("tier_diff_duplicate_identity")
    if not is_fourway:
        expected_identity = comparison.get("common_identity_sha256")
        if isinstance(expected_identity, str) and SHA256_RE.fullmatch(expected_identity):
            actual_identity = hashlib.sha256(_canonical(key_payloads)).hexdigest()
            if actual_identity != expected_identity:
                # Tier identity hashes use canonical JSON row identities in the
                # producer.  Preserve the mismatch as a blocker instead of
                # treating a different hashing convention as equivalent.
                blockers.append("tier_diff_common_identity_hash_mismatch")
        if comparison.get("changed_rank_count") != rank_changed:
            blockers.append("tier_diff_changed_rank_count_mismatch")
        if comparison.get("changed_tier_count") != tier_changed:
            blockers.append("tier_diff_changed_tier_count_mismatch")
    else:
        universe = report.get("candidate_universe")
        if not isinstance(universe, Mapping) or universe.get("identical") is not True:
            blockers.append("tier_diff_candidate_universe_not_identical")
        elif universe.get("game_count") != source.get("model_eligible_game_count") or universe.get("game_identity_sha256") != source.get("model_eligible_identity_sha256"):
            blockers.append("tier_diff_candidate_universe_source_mismatch")
    receipt_summary = None
    if receipt_path_value is not None:
        receipt_path = _safe_file(receipt_path_value, "Tier diff receipt")
        receipt = _read_json(receipt_path, "Tier diff receipt")
        blockers.extend(_authority_blockers(receipt.get("authority"), "tier_diff_receipt"))
        blockers.extend(_source_match(receipt.get("source", {}), source, "tier_diff_receipt"))
        try:
            receipt_hash = _hash_self(receipt, "receipt_sha256")
        except DownstreamImpactError as error:
            blockers.append(str(error))
            receipt_hash = None
        record = receipt.get("report")
        if isinstance(record, Mapping):
            if record.get("sha256") != _sha256_file(path) or record.get("bytes") != path.stat().st_size:
                blockers.append("tier_diff_receipt_report_bytes_mismatch")
            if record.get("report_sha256") != report.get("report_sha256"):
                blockers.append("tier_diff_receipt_report_hash_mismatch")
        receipt_summary = {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256_file(receipt_path),
            "receipt_sha256": receipt_hash,
        }
    summary = {
        "artifact": {**_artifact_binding(path), "report_sha256": report_hash},
        "receipt": receipt_summary,
        "status": report.get("status"),
        "comparison": {
            key: comparison.get(key)
            for key in (
                "reference",
                "candidate",
                "common_row_count",
                "baseline_only_row_count",
                "v2_only_row_count",
                "changed_rank_count",
                "changed_tier_count",
                "mean_absolute_rank_movement",
                "maximum_absolute_rank_movement",
                "common_identity_sha256",
                "paired_rows_sha256",
            )
            if key in comparison
        },
        "variants": {
            variant: dict(comparisons.get(variant, {}))
            for variant in VARIANTS
        }
        if is_fourway and isinstance(comparisons, Mapping)
        else {},
        "blockers_from_artifact": report.get("blockers", []),
    }
    return summary, sorted(set(blockers))


def _verify_tier_chronological(
    path_value: Path | str,
    receipt_path_value: Path | str | None,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Verify the existing chronological four-way Tier report."""

    path = _safe_file(path_value, "chronological four-way Tier report")
    report = _read_json(path, "chronological four-way Tier report")
    blockers: list[str] = []
    if report.get("schema_version") != TIER_CHRONOLOGICAL_SCHEMA_VERSION:
        blockers.append("tier_fourway_schema_invalid")
    blockers.extend(_authority_blockers(report.get("authority"), "tier_fourway"))
    blockers.extend(_source_match(report.get("source", {}), source, "tier_fourway"))
    try:
        report_hash = _hash_self(report, "report_sha256")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        report_hash = None
    universe = report.get("evaluation_universe")
    if not isinstance(universe, Mapping):
        blockers.append("tier_fourway_universe_missing")
        universe = {}
    if universe.get("game_count") not in {None, source.get("model_eligible_game_count")} or universe.get("game_identity_sha256") not in {None, source.get("model_eligible_identity_sha256")}:
        blockers.append("tier_fourway_universe_source_mismatch")
    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != set(VARIANTS):
        blockers.append("tier_fourway_variant_comparisons_missing")
        comparisons = {}
    variants: dict[str, Any] = {}
    for variant in VARIANTS:
        item = comparisons.get(variant)
        if not isinstance(item, Mapping):
            blockers.append(f"tier_fourway_{variant}_comparison_missing")
            continue
        variants[variant] = {
            key: item.get(key)
            for key in (
                "reference_variant",
                "row_count",
                "changed_rank_count",
                "changed_tier_count",
                "mean_absolute_rank_movement",
                "maximum_absolute_rank_movement",
                "rank_correlation",
                "paired_row_digest_sha256",
            )
            if key in item
        }
    receipt_summary = None
    if receipt_path_value is not None:
        receipt_path = _safe_file(receipt_path_value, "chronological Tier receipt")
        receipt = _read_json(receipt_path, "chronological Tier receipt")
        blockers.extend(_authority_blockers(receipt.get("authority"), "tier_fourway_receipt"))
        try:
            receipt_hash = _hash_self(receipt, "receipt_sha256")
        except DownstreamImpactError as error:
            blockers.append(str(error))
            receipt_hash = None
        if receipt.get("report_raw_sha256") not in {None, _sha256_file(path)}:
            blockers.append("tier_fourway_receipt_report_bytes_mismatch")
        receipt_summary = {
            "path": str(receipt_path),
            "bytes": receipt_path.stat().st_size,
            "sha256": _sha256_file(receipt_path),
            "receipt_sha256": receipt_hash,
        }
    return {
        "artifact": {**_artifact_binding(path), "report_sha256": report_hash},
        "receipt": receipt_summary,
        "status": report.get("status"),
        "universe": dict(universe),
        "variants": variants,
        "blockers_from_artifact": report.get("blockers", []),
    }, sorted(set(blockers))


def _verify_draft(
    path_value: Path | str,
    *,
    source: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    path = _safe_file(path_value, "Draft Score four-way report")
    report = _read_json(path, "Draft Score four-way report")
    blockers: list[str] = []
    if report.get("schema_version") not in {DRAFT_SCHEMA_V1, DRAFT_SCHEMA_V2}:
        blockers.append("draft_score_schema_invalid")
    blockers.extend(_source_match(report.get("source", {}), source, "draft_score"))
    blockers.extend(_authority_blockers(report.get("authority"), "draft_score"))
    try:
        claimed = report.get("report_sha256")
        if not isinstance(claimed, str) or SHA256_RE.fullmatch(claimed) is None:
            raise DownstreamImpactError("report_sha256 is missing or invalid")
        unsigned = dict(report)
        # The producer writes descriptive_rows as a separate artifact and
        # excludes it from the report digest.  Keep that binding explicit.
        unsigned.pop("descriptive_rows", None)
        unsigned.pop("report_sha256", None)
        report_hash = claimed.lower()
        if hashlib.sha256(_canonical(unsigned)).hexdigest() != report_hash:
            raise DownstreamImpactError("report_sha256 does not match the JSON payload")
    except DownstreamImpactError as error:
        blockers.append(str(error))
        report_hash = None
    coverage = report.get("coverage")
    if not isinstance(coverage, Mapping):
        blockers.append("draft_score_coverage_missing")
        coverage = {}
    if coverage.get("accepted_game_count") != source["source_game_count"]:
        blockers.append("draft_score_accepted_census_count_mismatch")
    subset_count = coverage.get("descriptive_subset_game_count")
    if isinstance(subset_count, int) and subset_count != source["source_game_count"]:
        blockers.append("draft_score_uses_subset_census")
    variants = report.get("variants")
    variant_summary: dict[str, Any] = {}
    prediction_maps: dict[str, dict[str, dict[str, Any]]] = {}
    if not isinstance(variants, Mapping):
        blockers.append("draft_score_variants_missing")
    else:
        for variant in VARIANTS:
            payload = variants.get(variant)
            if not isinstance(payload, Mapping):
                blockers.append(f"draft_score_{variant}_missing")
                continue
            fold_rows = payload.get("folds")
            variant_blockers = list(payload.get("blockers", [])) if isinstance(payload.get("blockers"), list) else []
            if isinstance(fold_rows, list):
                for fold in fold_rows:
                    if isinstance(fold, Mapping):
                        variant_blockers.extend(fold.get("blockers", []))
            else:
                variant_blockers.append("folds_missing")
            predictions: dict[str, dict[str, Any]] = {}
            if isinstance(fold_rows, list):
                for fold in fold_rows:
                    if not isinstance(fold, Mapping):
                        continue
                    raw_predictions = fold.get("predictions", [])
                    if raw_predictions is None:
                        continue
                    if not isinstance(raw_predictions, list):
                        variant_blockers.append("predictions_not_a_list")
                        continue
                    for prediction in raw_predictions:
                        if not isinstance(prediction, Mapping):
                            variant_blockers.append("prediction_row_invalid")
                            continue
                        game_id = str(prediction.get("game_id") or "").strip()
                        if not game_id:
                            variant_blockers.append("prediction_identity_missing")
                            continue
                        if game_id in predictions:
                            variant_blockers.append("prediction_identity_duplicate")
                            continue
                        predictions[game_id] = {
                            key: prediction.get(key)
                            for key in ("logit", "probability")
                            if key in prediction
                        }
            prediction_maps[variant] = predictions
            variant_summary[variant] = {
                "status": payload.get("status"),
                "valid_fold_count": payload.get("valid_fold_count"),
                "fold_count": len(fold_rows) if isinstance(fold_rows, list) else 0,
                "feature_names": payload.get("feature_names", []),
                "producer_requirements": payload.get("producer_requirements", []),
                "blockers": sorted({str(item) for item in variant_blockers}),
                "prediction_rows": len(predictions),
                "metrics": [
                    fold.get("metrics")
                    for fold in fold_rows
                    if isinstance(fold, Mapping) and isinstance(fold.get("metrics"), Mapping)
                ]
                if isinstance(fold_rows, list)
                else [],
            }
    prediction_changes: dict[str, Any] = {}
    baseline_predictions = prediction_maps.get("current_only", {})
    for variant in VARIANTS:
        if variant == "current_only":
            continue
        candidate_predictions = prediction_maps.get(variant, {})
        common = sorted(set(baseline_predictions) & set(candidate_predictions))
        logit_deltas: list[float] = []
        probability_deltas: list[float] = []
        for game_id in common:
            for field, target in (("logit", logit_deltas), ("probability", probability_deltas)):
                left = baseline_predictions[game_id].get(field)
                right = candidate_predictions[game_id].get(field)
                if isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
                    if math.isfinite(float(left)) and math.isfinite(float(right)):
                        target.append(float(right) - float(left))
        prediction_changes[variant] = {
            "baseline": "current_only",
            "candidate": variant,
            "baseline_rows": len(baseline_predictions),
            "candidate_rows": len(candidate_predictions),
            "common_rows": len(common),
            "logit": {
                "changed_count": sum(abs(value) > 1e-12 for value in logit_deltas),
                "mean_delta": sum(logit_deltas) / len(logit_deltas) if logit_deltas else None,
                "mean_abs_delta": sum(abs(value) for value in logit_deltas) / len(logit_deltas) if logit_deltas else None,
                "max_abs_delta": max((abs(value) for value in logit_deltas), default=None),
            },
            "probability": {
                "changed_count": sum(abs(value) > 1e-12 for value in probability_deltas),
                "mean_delta": sum(probability_deltas) / len(probability_deltas) if probability_deltas else None,
                "mean_abs_delta": sum(abs(value) for value in probability_deltas) / len(probability_deltas) if probability_deltas else None,
                "max_abs_delta": max((abs(value) for value in probability_deltas), default=None),
            },
        }
    summary = {
        "artifact": {**_artifact_binding(path), "report_sha256": report_hash},
        "status": report.get("status"),
        "coverage": dict(coverage),
        "variants": variant_summary,
        "changes_vs_current_only": prediction_changes,
        "blockers_from_artifact": report.get("blockers", []),
    }
    return summary, sorted(set(blockers))


def build_downstream_impact_report(
    *,
    source_receipt: Path | str,
    evaluations: Mapping[str, Path | str],
    evaluation_receipts: Mapping[str, Path | str] | None = None,
    snapshot_receipt: Path | str | None = None,
    snapshot_comparison: Path | str | None = None,
    snapshot_manifest: Path | str | None = None,
    snapshot_variants: Mapping[str, Path | str] | None = None,
    snapshot_manifests: Mapping[str, Path | str] | None = None,
    snapshot_capability_manifest: Path | str | None = None,
    tier_diff: Path | str | None = None,
    tier_receipt: Path | str | None = None,
    tier_fourway_report: Path | str | None = None,
    tier_fourway_receipt: Path | str | None = None,
    tier_shadow_manifest: Path | str | None = None,
    draft_score_report: Path | str | None = None,
    paired_uncertainty: Path | str | None = None,
) -> dict[str, Any]:
    """Read and compare all required downstream artifacts.

    Fundamental source errors raise ``DownstreamImpactError``.  Artifact-level
    errors are retained as blockers so the returned report remains useful for
    research review.
    """

    missing = [variant for variant in VARIANTS if variant not in evaluations]
    if missing:
        raise DownstreamImpactError(f"evaluation inputs missing: {', '.join(missing)}")
    source_path, source_receipt_value, source = _load_source(source_receipt)
    all_blockers: list[str] = []
    models: dict[str, dict[str, Any]] = {}
    evaluation_payloads: dict[str, dict[str, Any]] = {}
    evaluation_receipts = evaluation_receipts or {}
    for variant in VARIANTS:
        model, summary, blockers = _verify_evaluation(
            evaluations[variant],
            variant=variant,
            source=source,
            receipt_path=evaluation_receipts.get(variant),
        )
        evaluation_payloads[variant] = model
        models[variant] = summary
        all_blockers.extend(blockers)
        all_blockers.extend(
            f"{variant}_{str(item)}"
            for item in summary.get("model_blockers", [])
            if isinstance(item, str)
        )
    bootstrap, bootstrap_blockers = _verify_bootstrap(
        paired_uncertainty,
        source=source,
        evaluation_summaries=models,
    )
    all_blockers.extend(bootstrap_blockers)
    snapshots: dict[str, Any]
    snapshot_blockers: list[str] = []
    if snapshot_variants is not None:
        if set(snapshot_variants) != set(VARIANTS):
            raise DownstreamImpactError("snapshot variant inputs must cover all four variants")
        manifests = snapshot_manifests or {}
        if set(manifests) != set(VARIANTS):
            raise DownstreamImpactError("snapshot manifest inputs must cover all four variants")
        variant_snapshots: dict[str, Any] = {}
        for variant in VARIANTS:
            try:
                summary, blockers = _verify_snapshot_variant(
                    snapshot_variants[variant],
                    manifests[variant],
                    variant=variant,
                    source=source,
                )
            except DownstreamImpactError as error:
                summary, blockers = {"status": "invalid", "error": str(error)}, [
                    f"{variant}_snapshot_input_validation_failed"
                ]
            variant_snapshots[variant] = summary
            snapshot_blockers.extend(blockers)
        snapshots = {
            "status": "research_only" if not snapshot_blockers else "blocked",
            "variants": variant_snapshots,
            "capability_manifest": None,
        }
        if snapshot_capability_manifest is None:
            snapshot_blockers.append("snapshot_capability_manifest_missing")
        else:
            cap_path = _safe_file(snapshot_capability_manifest, "snapshot capability manifest")
            cap = _read_json(cap_path, "snapshot capability manifest")
            try:
                _hash_self(cap, "manifest_sha256")
            except DownstreamImpactError as error:
                snapshot_blockers.append(str(error))
            snapshot_blockers.extend(_authority_blockers(cap.get("authority"), "snapshot_capability_manifest"))
            snapshot_cap_source = cap.get("source", {})
            snapshot_blockers.extend(_source_match(snapshot_cap_source, source, "snapshot_capability_manifest"))
            if cap.get("schema_version") not in {
                SNAPSHOT_CAPABILITY_MANIFEST_SCHEMA_VERSION,
                SNAPSHOT_CAPABILITY_BUNDLE_MANIFEST_SCHEMA_VERSION,
            }:
                snapshot_blockers.append("snapshot_capability_manifest_schema_invalid")
            if cap.get("capability_schema_version") not in {
                None,
                SNAPSHOT_CAPABILITY_SCHEMA_VERSION,
            }:
                snapshot_blockers.append("snapshot_capability_manifest_capability_schema_invalid")
            cap_variants = cap.get("variants")
            if not isinstance(cap_variants, Mapping) or set(cap_variants) != set(VARIANTS):
                snapshot_blockers.append("snapshot_capability_manifest_variants_missing")
            else:
                for variant in VARIANTS:
                    record = cap_variants.get(variant)
                    if not isinstance(record, Mapping):
                        snapshot_blockers.append(f"snapshot_capability_{variant}_missing")
                        continue
                    if record.get("variant") not in {None, variant}:
                        snapshot_blockers.append(f"snapshot_capability_{variant}_variant_mismatch")
                    if variant == "scaling_curve":
                        coverage = record.get("rank_coverage", {})
                        if not isinstance(coverage, Mapping):
                            coverage = {}
                        if not coverage:
                            capability = record.get("capability", {})
                            if isinstance(capability, Mapping):
                                coverage = {
                                    kind: capability.get(f"{kind}_ranks", {})
                                    for kind in ("player", "team")
                                }
                        if not all(isinstance(coverage.get(kind), Mapping) and coverage[kind].get("status") == "not_applicable" and coverage[kind].get("row_policy") == "no_rows" for kind in ("player", "team")):
                            snapshot_blockers.append("snapshot_capability_scaling_curve_na_missing")
            snapshots["capability_manifest"] = {"path": str(cap_path), **_artifact_binding(cap_path), "variants": cap_variants if isinstance(cap_variants, Mapping) else {}}
        snapshots["status"] = "research_only" if not snapshot_blockers else "blocked"
    elif snapshot_receipt is not None and snapshot_comparison is not None:
        try:
            snapshots, snapshot_blockers = _verify_snapshot_bundle(
                snapshot_receipt,
                snapshot_comparison,
                snapshot_manifest,
                source=source,
            )
        except DownstreamImpactError as error:
            snapshots = {"status": "invalid", "error": str(error)}
            snapshot_blockers = ["snapshot_input_validation_failed"]
    else:
        snapshots = {"status": "invalid", "error": "snapshot inputs are missing"}
        snapshot_blockers = ["snapshot_input_validation_failed"]
    all_blockers.extend(snapshot_blockers)
    if tier_diff is None:
        tier = {"status": "invalid", "error": "Tier diff input is missing"}
        tier_blockers = ["tier_input_validation_failed"]
    else:
        try:
            tier, tier_blockers = _verify_tier(tier_diff, tier_receipt, source=source)
        except DownstreamImpactError as error:
            tier = {"status": "invalid", "error": str(error)}
            tier_blockers = ["tier_input_validation_failed"]
    all_blockers.extend(tier_blockers)
    chronological = None
    chronological_blockers: list[str] = []
    if tier_fourway_report is not None:
        try:
            chronological, chronological_blockers = _verify_tier_chronological(
                tier_fourway_report, tier_fourway_receipt, source=source
            )
        except DownstreamImpactError as error:
            chronological = {"status": "invalid", "error": str(error)}
            chronological_blockers = ["tier_fourway_input_validation_failed"]
        all_blockers.extend(chronological_blockers)
    shadow = None
    shadow_blockers: list[str] = []
    if tier_shadow_manifest is not None:
        try:
            shadow, shadow_blockers = _verify_tier_shadow_manifest(
                tier_shadow_manifest, source=source
            )
        except DownstreamImpactError as error:
            shadow = {"status": "invalid", "error": str(error)}
            shadow_blockers = ["tier_shadow_input_validation_failed"]
        all_blockers.extend(shadow_blockers)
    if draft_score_report is None:
        draft = {"status": "invalid", "error": "Draft Score input is missing"}
        draft_blockers = ["draft_score_input_validation_failed"]
    else:
        try:
            draft, draft_blockers = _verify_draft(draft_score_report, source=source)
        except DownstreamImpactError as error:
            draft = {"status": "invalid", "error": str(error)}
            draft_blockers = ["draft_score_input_validation_failed"]
    all_blockers.extend(draft_blockers)
    measured = {
        "player_rank_movement_available": bool(snapshots.get("rank_movement", {}).get("player") or snapshots.get("variants")),
        "team_rank_movement_available": bool(snapshots.get("rank_movement", {}).get("team") or snapshots.get("variants")),
        "tier_rank_changes_available": bool(tier.get("comparison") or tier.get("variants")),
        "draft_score_changes_available": bool(draft.get("variants")),
    }
    downstream_public_change_flags = {
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
        "measured_changes_present": measured,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_research_only" if not all_blockers else "blocked",
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": {
            **source,
            "receipt_path": str(source_path),
            "receipt_file": _artifact_binding(source_path),
        },
        "evaluations": {
            "variants": models,
            "bootstrap_deltas": bootstrap,
        },
        "snapshots": snapshots,
        "tierlist": {
            **tier,
            "chronological_fourway": chronological,
            "full_census_shadow": shadow,
        },
        "draft_score": draft,
        "downstream_public_change_flags": downstream_public_change_flags,
        "blockers": sorted(set(str(item) for item in all_blockers if str(item).strip())),
        "authority": _authority_summary(source_receipt_value.get("authority")),
        "claim_ceiling": "source-bound research impact review; public outputs remain disabled",
    }
    variant_impact: dict[str, Any] = {}
    for variant in VARIANTS:
        snapshot_variant = snapshots.get("variants", {}).get(variant, {}) if isinstance(snapshots.get("variants"), Mapping) else {}
        snapshot_capability = snapshot_variant.get("capability", {}) if isinstance(snapshot_variant, Mapping) else {}
        tier_variant = tier.get("variants", {}).get(variant, {}) if isinstance(tier.get("variants"), Mapping) else {}
        chronological_variant = chronological.get("variants", {}).get(variant, {}) if isinstance(chronological, Mapping) and isinstance(chronological.get("variants"), Mapping) else {}
        draft_variant = draft.get("variants", {}).get(variant, {}) if isinstance(draft.get("variants"), Mapping) else {}
        variant_impact[variant] = {
            "prediction": models.get(variant, {}),
            "snapshot": {
                "status": snapshot_variant.get("status"),
                "capability": snapshot_capability,
                "reference": {
                    "player": snapshot_capability.get("player", {}).get("scope") if isinstance(snapshot_capability.get("player"), Mapping) else None,
                    "team": snapshot_capability.get("team", {}).get("scope") if isinstance(snapshot_capability.get("team"), Mapping) else None,
                    "rank": snapshot_capability.get("player_ranks", {}).get("scope") if isinstance(snapshot_capability.get("player_ranks"), Mapping) else None,
                },
                "player_ranks": snapshot_variant.get("rank_movement", {}).get("player") if isinstance(snapshot_variant.get("rank_movement"), Mapping) else None,
                "team_ranks": snapshot_variant.get("rank_movement", {}).get("team") if isinstance(snapshot_variant.get("rank_movement"), Mapping) else None,
                "rank_scope": {
                    "player": snapshot_capability.get("player", {}).get("scope") if isinstance(snapshot_capability.get("player"), Mapping) else None,
                    "team": snapshot_capability.get("team", {}).get("scope") if isinstance(snapshot_capability.get("team"), Mapping) else None,
                },
                "intrinsic_rank_status": "not_applicable" if variant == "scaling_curve" else "available",
            },
            "tier": {
                "full_census": tier_variant,
                "chronological": chronological_variant,
            },
            "draft": draft_variant,
        }
    report["variant_impacts"] = variant_impact
    return report


def write_report(path: Path | str, report: Mapping[str, Any]) -> Path:
    """Write one JSON report and return its path."""

    output = Path(path).expanduser()
    if not output.is_absolute() or ".." in output.parts:
        raise DownstreamImpactError("output path must be absolute and contain no '..'")
    if output.exists() and output.is_symlink():
        raise DownstreamImpactError("output path is a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(report), ensure_ascii=True, allow_nan=False, sort_keys=True, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return output


def _parse_bindings(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise DownstreamImpactError(f"{label} must use VAR=PATH")
        variant, raw_path = value.split("=", 1)
        variant = variant.strip()
        if variant not in VARIANTS or not raw_path.strip():
            raise DownstreamImpactError(f"{label} has an invalid variant")
        if variant in result:
            raise DownstreamImpactError(f"{label} repeats {variant}")
        result[variant] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--evaluation", action="append", default=[], metavar="VAR=PATH")
    parser.add_argument("--evaluation-receipt", action="append", default=[], metavar="VAR=PATH")
    parser.add_argument("--snapshot-receipt", type=Path)
    parser.add_argument("--snapshot-comparison", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--snapshot-variant", action="append", default=[], metavar="VAR=PATH")
    parser.add_argument("--snapshot-manifest-variant", action="append", default=[], metavar="VAR=PATH")
    parser.add_argument("--snapshot-capability-manifest", type=Path)
    parser.add_argument("--tier-diff", type=Path)
    parser.add_argument("--tier-receipt", type=Path)
    parser.add_argument("--tier-fourway-report", type=Path)
    parser.add_argument("--tier-fourway-receipt", type=Path)
    parser.add_argument("--tier-shadow-manifest", type=Path)
    parser.add_argument("--draft-score-report", type=Path)
    parser.add_argument("--paired-uncertainty", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        evaluations = _parse_bindings(args.evaluation, "--evaluation")
        evaluation_receipts = _parse_bindings(args.evaluation_receipt, "--evaluation-receipt")
        snapshot_variants = _parse_bindings(args.snapshot_variant, "--snapshot-variant")
        snapshot_manifests = _parse_bindings(args.snapshot_manifest_variant, "--snapshot-manifest-variant")
        report = build_downstream_impact_report(
            source_receipt=args.source_receipt,
            evaluations=evaluations,
            evaluation_receipts=evaluation_receipts,
            snapshot_receipt=args.snapshot_receipt,
            snapshot_comparison=args.snapshot_comparison,
            snapshot_manifest=args.snapshot_manifest,
            snapshot_variants=snapshot_variants or None,
            snapshot_manifests=snapshot_manifests or None,
            snapshot_capability_manifest=args.snapshot_capability_manifest,
            tier_diff=args.tier_diff,
            tier_receipt=args.tier_receipt,
            tier_fourway_report=args.tier_fourway_report,
            tier_fourway_receipt=args.tier_fourway_receipt,
            tier_shadow_manifest=args.tier_shadow_manifest,
            draft_score_report=args.draft_score_report,
            paired_uncertainty=args.paired_uncertainty,
        )
        write_report(args.output, report)
    except DownstreamImpactError as error:
        blocked = {
            "schema_version": SCHEMA_VERSION,
            "status": "blocked",
            "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "authority": _authority_summary(None),
            "blockers": ["input_validation_failed"],
            "error": str(error),
            "claim_ceiling": "source-bound research impact review; public outputs remain disabled",
        }
        try:
            write_report(args.output, blocked)
        except Exception:
            pass
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_COMPARISONS",
    "DownstreamImpactError",
    "SCHEMA_VERSION",
    "VARIANTS",
    "build_downstream_impact_report",
    "write_report",
]
