"""Versioned KOI_MARI approval contract for G5 v2; creates no approval."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any
from datetime import datetime, timezone

from . import contract, v2_math, v2_result

V1_PRIMITIVES = {
    "locator": "lol_kills/v2/draft/interactions/g5_exploratory/runner.py",
    "raw_sha256": "938c65d7bf6a925edc961d461c7dcdc6db83582c34d314260a66d0300f81102e",
    "trusted_functions": [
        "_load_accepted_g1", "_load_accepted_features", "_load_accepted_clusters",
        "align_inputs", "build_b0_scores", "_design", "score_d1",
        "_measure_invariances", "_prior_aggregate", "validation_bootstrap",
        "_probability_logloss", "_base_evidence",
    ],
}


ROOT = Path(__file__).resolve().parents[5]
NAMESPACE = "data/lol/v2/models/draft-interactions/g5-exploratory"
APPROVAL_LOCATOR = f"{NAMESPACE}/v2-execution-approval.json"
LEDGER_LOCATOR = f"{NAMESPACE}/v2-execution-run-ledger.jsonl"
RESULT_LOCATOR = f"{NAMESPACE}/v2-execution-result.json"
APPROVAL_SCHEMA = "scryglass:g5-private-development-v2-approval:v1"
LEDGER_SCHEMA = "scryglass:g5-private-development-v2-ledger-entry:v1"
ROOT_AUTHORITY = "KOI_MARI"
SCOPE = {
    "private_model_fit": True,
    "private_rank_selection": True,
    "final_holdout": False,
    "public": False,
    "prediction": False,
    "publication": False,
    "promotion": False,
}


class V2ApprovalError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _safe_path(locator: str, *, may_be_missing: bool) -> Path:
    if locator not in {APPROVAL_LOCATOR, LEDGER_LOCATOR, RESULT_LOCATOR}:
        raise V2ApprovalError("unapproved locator")
    path = ROOT / locator
    current = ROOT.absolute()
    for part in path.absolute().relative_to(current).parts[:-1]:
        current /= part
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise V2ApprovalError("unsafe approval parent")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        if may_be_missing:
            return path
        raise V2ApprovalError("required fixed file missing")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise V2ApprovalError("unsafe approval leaf")
    return path


def validate_approval(
    value: Mapping[str, Any],
    *,
    expected_review_core_sha256: str,
    expected_contract_sha256: str,
    expected_run_id: str,
    expected_config_sha256: str = v2_math.config_hash(),
) -> None:
    expected = {
        "schema_id", "state", "approval_id", "run_id", "reviewer_root",
        "authority_scope", "review_core_sha256", "contract_sha256",
        "numerical_config_sha256",
        "dependency_pins", "allowed_partitions", "selection_semantics",
        "paths", "claim_ceiling", "approval_sha256",
    }
    if set(value) != expected:
        raise V2ApprovalError("approval exact field set mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("approval_sha256", None)
    if (
        value.get("schema_id") != APPROVAL_SCHEMA
        or value.get("state") != "APPROVED_PRIVATE_DEVELOPMENT_V2"
        or claimed != sha256(unsigned)
        or value.get("approval_id") == "koi-mari-g5-private-development-v1"
        or value.get("run_id") != expected_run_id
        or value.get("review_core_sha256") != expected_review_core_sha256
        or value.get("contract_sha256") != expected_contract_sha256
        or value.get("numerical_config_sha256") != expected_config_sha256
        or value.get("reviewer_root") != ROOT_AUTHORITY
        or value.get("authority_scope") != SCOPE
        or value.get("dependency_pins") != {
            "G1": contract.G1,
            "G1_features": contract.G1_FEATURES,
            "G2": contract.G2,
            "clusters": contract.CLUSTERS,
            "accepted_v1_orchestration_primitives": V1_PRIMITIVES,
        }
        or value.get("allowed_partitions") != ["TRAIN", "DEVELOPMENT", "VALIDATION"]
        or value.get("selection_semantics") != {
            "development": "select_D1_iff_mean_LL_B0_minus_LL_D1_gt_0_else_B0",
            "validation": "D1_winner_iff_mean_gain_gte_0.005_and_cluster_bootstrap_95pct_LCB_gt_0",
            "bootstrap_replicates": 2000,
            "D2": "OMITTED",
            "final_holdout": False,
        }
        or value.get("paths") != {
            "approval": APPROVAL_LOCATOR,
            "ledger": LEDGER_LOCATOR,
            "result": RESULT_LOCATOR,
        }
        or value.get("claim_ceiling") != v2_result.CLAIM_CEILING
    ):
        raise V2ApprovalError("approval frozen binding mismatch")


def load_approval(
    *, expected_review_core_sha256: str, expected_contract_sha256: str,
    expected_run_id: str, expected_config_sha256: str = v2_math.config_hash()
) -> dict[str, Any]:
    raw = _safe_path(APPROVAL_LOCATOR, may_be_missing=False).read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise V2ApprovalError("approval not JSON") from error
    if not isinstance(value, Mapping) or raw != canonical_bytes(value) + b"\n":
        raise V2ApprovalError("approval not canonical newline JSON")
    validate_approval(
        value,
        expected_review_core_sha256=expected_review_core_sha256,
        expected_contract_sha256=expected_contract_sha256,
        expected_run_id=expected_run_id,
        expected_config_sha256=expected_config_sha256,
    )
    return dict(value)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise V2ApprovalError("ledger timestamp invalid")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise V2ApprovalError("ledger timestamp invalid") from error


def _validate_ledger_entry(value: Mapping[str, Any]) -> None:
    common = {
        "schema_id", "state", "approval_id", "run_id", "review_core_sha256",
        "approval_sha256", "result_locator", "sequence", "entry_sha256",
    }
    state_fields = {
        "STARTED": {"started_at"},
        "COMPLETED": {
            "started_entry_sha256", "result_artifact_sha256",
            "result_raw_sha256", "config_sha256", "transform_sha256",
            "scales_sha256", "membership_hashes_sha256",
            "source_pins_sha256", "completed_at",
        },
        "INVALID_DUPLICATE": {
            "authoritative_completed_entry_sha256",
            "authoritative_result_artifact_sha256", "recorded_at",
        },
    }
    state = value.get("state")
    if state not in state_fields or set(value) != common | state_fields[state]:
        raise V2ApprovalError("ledger exact field set mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("entry_sha256", None)
    if value.get("schema_id") != LEDGER_SCHEMA or claimed != sha256(unsigned):
        raise V2ApprovalError("ledger identity mismatch")
    for field in (
        "review_core_sha256", "approval_sha256", "started_entry_sha256",
        "result_artifact_sha256", "result_raw_sha256",
        "config_sha256", "transform_sha256", "scales_sha256",
        "membership_hashes_sha256", "source_pins_sha256",
        "authoritative_completed_entry_sha256",
        "authoritative_result_artifact_sha256",
    ):
        if field in value and (
            not isinstance(value[field], str)
            or len(value[field]) != 64
            or any(character not in "0123456789abcdef" for character in value[field])
        ):
            raise V2ApprovalError("ledger sha invalid")
    if (
        not isinstance(value["approval_id"], str) or not value["approval_id"]
        or not isinstance(value["run_id"], str) or not value["run_id"]
        or value["result_locator"] != RESULT_LOCATOR
        or type(value["sequence"]) is not int or value["sequence"] < 1
    ):
        raise V2ApprovalError("ledger binding invalid")
    _timestamp(value[{
        "STARTED": "started_at",
        "COMPLETED": "completed_at",
        "INVALID_DUPLICATE": "recorded_at",
    }[state]])


def load_ledger() -> list[dict[str, Any]]:
    path = _safe_path(LEDGER_LOCATOR, may_be_missing=True)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise V2ApprovalError("ledger missing terminal newline")
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise V2ApprovalError("ledger line not JSON") from error
        if not isinstance(value, Mapping) or line != canonical_bytes(value):
            raise V2ApprovalError("ledger line not canonical")
        _validate_ledger_entry(value)
        entries.append(dict(value))
    return entries


def validate_ledger_history(
    entries: list[dict[str, Any]],
    *,
    approval: Mapping[str, Any],
    expected_review_core_sha256: str,
    expected_run_id: str,
) -> str:
    previous: datetime | None = None
    completed: Mapping[str, Any] | None = None
    for index, entry in enumerate(entries):
        _validate_ledger_entry(entry)
        if (
            entry["sequence"] != index + 1
            or entry["approval_id"] != approval["approval_id"]
            or entry["approval_sha256"] != approval["approval_sha256"]
            or entry["run_id"] != expected_run_id
            or entry["review_core_sha256"] != expected_review_core_sha256
        ):
            raise V2ApprovalError("foreign or reordered ledger")
        field = {
            "STARTED": "started_at", "COMPLETED": "completed_at",
            "INVALID_DUPLICATE": "recorded_at",
        }[entry["state"]]
        current = _timestamp(entry[field])
        if previous is not None and current < previous:
            raise V2ApprovalError("ledger timestamp regression")
        previous = current
        if index == 0 and entry["state"] != "STARTED":
            raise V2ApprovalError("ledger must begin STARTED")
        if index == 1:
            if (
                entry["state"] != "COMPLETED"
                or entry["started_entry_sha256"] != entries[0]["entry_sha256"]
            ):
                raise V2ApprovalError("STARTED must transition to bound COMPLETED")
            completed = entry
        if index > 1 and (
            entry["state"] != "INVALID_DUPLICATE"
            or completed is None
            or entry["authoritative_completed_entry_sha256"] != completed["entry_sha256"]
            or entry["authoritative_result_artifact_sha256"]
            != completed["result_artifact_sha256"]
        ):
            raise V2ApprovalError("earliest completion binding invalid")
    if not entries:
        return "EMPTY"
    if len(entries) == 1:
        return "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY"
    return "COMPLETED_TERMINAL"


def append_ledger_entry(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_path(LEDGER_LOCATOR, may_be_missing=True)
    payload = {"schema_id": LEDGER_SCHEMA, **dict(unsigned)}
    payload["entry_sha256"] = sha256(payload)
    _validate_ledger_entry(payload)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise V2ApprovalError("unsafe ledger file")
        os.write(descriptor, canonical_bytes(payload) + b"\n")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def validate_completed_result(
    completion: Mapping[str, Any],
    *,
    expected: v2_result.ExpectedBinding,
) -> dict[str, Any]:
    path = _safe_path(RESULT_LOCATOR, may_be_missing=False)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise V2ApprovalError("completed result not JSON") from error
    if (
        not isinstance(value, Mapping)
        or raw != canonical_bytes(value) + b"\n"
        or hashlib.sha256(raw).hexdigest() != completion["result_raw_sha256"]
        or value.get("artifact_sha256") != completion["result_artifact_sha256"]
    ):
        raise V2ApprovalError("completed result byte identity mismatch")
    if (
        completion["config_sha256"] != expected.config_sha256
        or completion["transform_sha256"] != expected.transform_sha256
        or completion["scales_sha256"] != expected.scales_sha256
        or completion["membership_hashes_sha256"]
        != v2_result.sha256(expected.membership_hashes)
        or completion["source_pins_sha256"] != v2_result.sha256(expected.source_pins)
    ):
        raise V2ApprovalError("completed result expected binding mismatch")
    try:
        v2_result.validate_real(value, expected=expected)
    except ValueError as error:
        raise V2ApprovalError("completed result schema invalid") from error
    return dict(value)


def validate_completed_result_from_ledger(
    completion: Mapping[str, Any],
    *,
    approval: Mapping[str, Any],
    expected_contract_sha256: str,
    expected_review_core_sha256: str,
    expected_run_id: str,
) -> dict[str, Any]:
    """Reconstruct only ledger-authenticated runtime bindings on duplicate checks."""
    path = _safe_path(RESULT_LOCATOR, may_be_missing=False)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise V2ApprovalError("completed result not JSON") from error
    if (
        not isinstance(value, Mapping)
        or hashlib.sha256(raw).hexdigest() != completion["result_raw_sha256"]
        or value.get("artifact_sha256") != completion["result_artifact_sha256"]
        or value.get("source_pins") != v2_result.SOURCE_PINS
        or v2_result.sha256(value.get("membership_hashes"))
        != completion["membership_hashes_sha256"]
    ):
        raise V2ApprovalError("completed result ledger binding invalid")
    scaling = value.get("train_scaling")
    if not isinstance(scaling, Mapping):
        raise V2ApprovalError("completed result scaling missing")
    expected = v2_result.ExpectedBinding(
        contract_sha256=expected_contract_sha256,
        review_core_sha256=expected_review_core_sha256,
        approval_sha256=approval["approval_sha256"],
        run_id=expected_run_id,
        config_sha256=completion["config_sha256"],
        transform_sha256=completion["transform_sha256"],
        scales_sha256=completion["scales_sha256"],
        membership_hashes=dict(value["membership_hashes"]),
        source_pins=v2_result.SOURCE_PINS,
    )
    return validate_completed_result(completion, expected=expected)
