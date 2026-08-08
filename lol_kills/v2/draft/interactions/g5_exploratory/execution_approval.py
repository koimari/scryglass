"""Plain human-rooted approval and process-ledger contracts for private G5."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any
from datetime import datetime, timezone

from . import contract, result


ROOT = Path(__file__).resolve().parents[5]
NAMESPACE = "data/lol/v2/models/draft-interactions/g5-exploratory"
APPROVAL_LOCATOR = f"{NAMESPACE}/execution-approval.json"
LEDGER_LOCATOR = f"{NAMESPACE}/execution-run-ledger.jsonl"
RESULT_LOCATOR = f"{NAMESPACE}/execution-result.json"
APPROVAL_SCHEMA = "scryglass:g5-execution-approval:v1"
LEDGER_SCHEMA = "scryglass:g5-execution-run-ledger-entry:v1"
PREFIT = {
    "contract_sha256": "993a9e8e6184e8f2e2b7c1eed244f28ed6eb5d749f067979881d35e715a3a1f0",
    "core_sha256": "25df39ad248fed2565aed7f501b935f1992020fa5441ccff3b6e6ee99cf15ab8",
    "review_sha256": "f869b509abe2ba17bee66eff9a44d72f4ca6422d6ded78f1459e3976180d21ec",
}
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


class ApprovalError(ValueError):
    pass


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sha(value: Any, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ApprovalError(f"{label} must be lowercase sha256")


def _safe_path(locator: str, *, may_be_missing: bool) -> Path:
    if locator not in {APPROVAL_LOCATOR, LEDGER_LOCATOR, RESULT_LOCATOR}:
        raise ApprovalError("unapproved fixed locator")
    path = ROOT / locator
    current = ROOT.absolute()
    relative = path.absolute().relative_to(current)
    for part in relative.parts[:-1]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise ApprovalError("fixed path parent missing") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ApprovalError("unsafe fixed path parent")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        if may_be_missing:
            return path
        raise ApprovalError("required fixed file missing")
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_nlink != 1
    ):
        raise ApprovalError("unsafe fixed path leaf")
    return path


def validate_approval_payload(
    value: Mapping[str, Any], *, expected_runner_core_sha256: str, expected_run_id: str
) -> None:
    expected = {
        "schema_id", "approval_id", "run_id", "state", "reviewer_root",
        "authority_scope", "runner_core_sha256", "prefit", "source_pins",
        "allowed_partitions", "final_holdout", "paths", "claim_ceiling",
        "approval_sha256",
    }
    if set(value) != expected:
        raise ApprovalError("approval exact field set mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("approval_sha256", None)
    if (
        value.get("schema_id") != APPROVAL_SCHEMA
        or value.get("state") != "APPROVED_PRIVATE_DEVELOPMENT_EXECUTION"
        or claimed != sha256(unsigned)
    ):
        raise ApprovalError("approval schema/state/self hash mismatch")
    _sha(expected_runner_core_sha256, "runner core")
    if value.get("runner_core_sha256") != expected_runner_core_sha256:
        raise ApprovalError("approval runner core mismatch")
    for field in ("approval_id", "run_id"):
        item = value.get(field)
        if not isinstance(item, str) or not item or len(item) > 160 or any(c in item for c in "/\\\x00"):
            raise ApprovalError(f"approval {field} invalid")
    if value["run_id"] != expected_run_id:
        raise ApprovalError("approval run id mismatch")
    if (
        value.get("reviewer_root") != ROOT_AUTHORITY
        or value.get("authority_scope") != SCOPE
        or value.get("prefit") != PREFIT
        or value.get("source_pins") != {
            "G1": contract.G1,
            "G1_features": contract.G1_FEATURES,
            "G2": contract.G2,
            "clusters": contract.CLUSTERS,
        }
        or value.get("allowed_partitions") != ["TRAIN", "DEVELOPMENT", "VALIDATION"]
        or value.get("final_holdout") is not False
        or value.get("paths") != {
            "approval_locator": APPROVAL_LOCATOR,
            "ledger_locator": LEDGER_LOCATOR,
            "result_locator": RESULT_LOCATOR,
        }
        or value.get("claim_ceiling") != result.CLAIM_CEILING
    ):
        raise ApprovalError("approval frozen binding mismatch")


def load_approval(*, expected_runner_core_sha256: str, expected_run_id: str) -> dict[str, Any]:
    path = _safe_path(APPROVAL_LOCATOR, may_be_missing=False)
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ApprovalError("approval is not JSON") from error
    if not isinstance(value, Mapping) or raw != canonical_bytes(value) + b"\n":
        raise ApprovalError("approval is not canonical newline JSON")
    validate_approval_payload(
        value,
        expected_runner_core_sha256=expected_runner_core_sha256,
        expected_run_id=expected_run_id,
    )
    return dict(value)


def load_ledger() -> list[dict[str, Any]]:
    path = _safe_path(LEDGER_LOCATOR, may_be_missing=True)
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ApprovalError("ledger must end with newline")
    entries: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ApprovalError("ledger line is not JSON") from error
        if not isinstance(value, Mapping) or line != canonical_bytes(value):
            raise ApprovalError("ledger line is not canonical JSON")
        _validate_ledger_entry(value)
        entries.append(dict(value))
    return entries


def _validate_ledger_entry(value: Mapping[str, Any]) -> None:
    common = {
        "schema_id", "state", "approval_id", "run_id", "runner_core_sha256",
        "approval_sha256", "result_locator", "sequence", "entry_sha256",
    }
    state_fields = {
        "STARTED": {"started_at"},
        "COMPLETED": {
            "started_entry_sha256", "result_artifact_sha256",
            "result_raw_sha256", "completed_at",
        },
        "INVALID_DUPLICATE": {
            "authoritative_completed_entry_sha256",
            "authoritative_result_artifact_sha256", "recorded_at",
        },
    }
    state = value.get("state")
    if state not in state_fields or set(value) != common | state_fields[state]:
        raise ApprovalError("ledger entry exact field set mismatch")
    unsigned = dict(value)
    claimed = unsigned.pop("entry_sha256", None)
    if value.get("schema_id") != LEDGER_SCHEMA or claimed != sha256(unsigned):
        raise ApprovalError("ledger entry identity mismatch")
    for field in (
        "runner_core_sha256", "approval_sha256", "started_entry_sha256",
        "result_artifact_sha256", "result_raw_sha256",
        "authoritative_completed_entry_sha256",
        "authoritative_result_artifact_sha256",
    ):
        if field in value:
            _sha(value[field], f"ledger {field}")
    for field in ("approval_id", "run_id"):
        if not isinstance(value[field], str) or not value[field]:
            raise ApprovalError(f"ledger {field} invalid")
    if value["result_locator"] != RESULT_LOCATOR:
        raise ApprovalError("ledger result locator mismatch")
    if type(value["sequence"]) is not int or value["sequence"] < 1:
        raise ApprovalError("ledger sequence invalid")
    timestamp_field = {
        "STARTED": "started_at",
        "COMPLETED": "completed_at",
        "INVALID_DUPLICATE": "recorded_at",
    }[state]
    _timestamp(value[timestamp_field])


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise ApprovalError("ledger timestamp must be canonical UTC RFC3339")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc
        )
    except ValueError as error:
        raise ApprovalError("ledger timestamp must be canonical UTC RFC3339") from error


def validate_ledger_history(
    entries: list[dict[str, Any]],
    *,
    approval: Mapping[str, Any],
    expected_runner_core_sha256: str,
    expected_run_id: str,
) -> str:
    """Validate the complete one-approval process history before any read."""

    previous_time: datetime | None = None
    completion: Mapping[str, Any] | None = None
    for index, entry in enumerate(entries):
        _validate_ledger_entry(entry)
        if (
            entry["sequence"] != index + 1
            or entry["approval_id"] != approval["approval_id"]
            or entry["approval_sha256"] != approval["approval_sha256"]
            or entry["run_id"] != expected_run_id
            or entry["runner_core_sha256"] != expected_runner_core_sha256
            or entry["result_locator"] != RESULT_LOCATOR
        ):
            raise ApprovalError("foreign or reordered ledger entry")
        timestamp_field = {
            "STARTED": "started_at",
            "COMPLETED": "completed_at",
            "INVALID_DUPLICATE": "recorded_at",
        }[entry["state"]]
        current_time = _timestamp(entry[timestamp_field])
        if previous_time is not None and current_time < previous_time:
            raise ApprovalError("ledger timestamps regress")
        previous_time = current_time
        if index == 0:
            if entry["state"] != "STARTED":
                raise ApprovalError("ledger must begin with STARTED")
            continue
        if index == 1:
            if entry["state"] != "COMPLETED":
                raise ApprovalError("STARTED may only transition to COMPLETED")
            if entry["started_entry_sha256"] != entries[0]["entry_sha256"]:
                raise ApprovalError("completion does not bind STARTED entry")
            completion = entry
            continue
        if entry["state"] != "INVALID_DUPLICATE" or completion is None:
            raise ApprovalError("ledger terminal transition invalid")
        if (
            entry["authoritative_completed_entry_sha256"]
            != completion["entry_sha256"]
            or entry["authoritative_result_artifact_sha256"]
            != completion["result_artifact_sha256"]
        ):
            raise ApprovalError("duplicate does not bind earliest completion")
    if not entries:
        return "EMPTY"
    if len(entries) == 1:
        return "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY"
    return "COMPLETED_TERMINAL"


def validate_completed_result(
    completion: Mapping[str, Any],
    *,
    approval: Mapping[str, Any],
    expected_runner_core_sha256: str,
    expected_run_id: str,
    started_entry_sha256: str,
) -> dict[str, Any]:
    """Authenticate the immutable result referenced by a terminal completion."""

    path = _safe_path(RESULT_LOCATOR, may_be_missing=False)
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ApprovalError("completed result is not JSON") from error
    if (
        not isinstance(payload, Mapping)
        or raw != canonical_bytes(payload) + b"\n"
        or hashlib.sha256(raw).hexdigest() != completion["result_raw_sha256"]
        or payload.get("artifact_sha256") != completion["result_artifact_sha256"]
    ):
        raise ApprovalError("completed result byte identity mismatch")
    try:
        result.validate_real(payload)
    except ValueError as error:
        raise ApprovalError("completed result strict schema invalid") from error
    binding = payload["execution_binding"]
    if (
        binding["approval_sha256"] != approval["approval_sha256"]
        or binding["runner_core_sha256"] != expected_runner_core_sha256
        or binding["run_id_sha256"] != sha256(expected_run_id)
        or binding["started_entry_sha256"] != started_entry_sha256
        or binding["result_locator"] != RESULT_LOCATOR
    ):
        raise ApprovalError("completed result execution binding mismatch")
    return dict(payload)


def append_ledger_entry(unsigned: Mapping[str, Any]) -> dict[str, Any]:
    path = _safe_path(LEDGER_LOCATOR, may_be_missing=True)
    payload = dict(unsigned)
    payload["schema_id"] = LEDGER_SCHEMA
    payload["entry_sha256"] = sha256(payload)
    _validate_ledger_entry(payload)
    data = canonical_bytes(payload) + b"\n"
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT
    descriptor = os.open(path, flags, 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ApprovalError("unsafe ledger file")
        os.write(descriptor, data)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    parent = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)
    return payload
