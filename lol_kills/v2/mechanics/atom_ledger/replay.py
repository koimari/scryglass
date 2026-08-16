"""Apply ordered patch delta events and produce a deterministic replay receipt."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

from .base import validate_base_snapshot
from .schema import (
    DELTA_SCHEMA_ID,
    PARTIAL_DELTA_AUTHORITY_STATUS,
    RECEIPT_SCHEMA_ID,
    AtomLedgerConflictError,
    AtomLedgerCoverageError,
    AtomLedgerFutureDataError,
    AtomLedgerIntegrityError,
    canonical_sha256,
    parse_patch,
    parse_utc,
    signed_hash,
    stable_atom_id,
    validate_signed_hash,
)

DEFAULT_DELTA_PATH = Path(__file__).with_name("deltas") / "26.16-wiki-pilot.json"


def load_delta_event(path: Path = DEFAULT_DELTA_PATH) -> dict[str, Any]:
    if not path.is_file():
        raise AtomLedgerIntegrityError(f"missing delta event: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AtomLedgerIntegrityError(f"cannot load delta event: {path}") from exc
    if not isinstance(payload, dict):
        raise AtomLedgerIntegrityError("delta event must be an object")
    validate_delta_event(payload)
    return payload


def validate_delta_event(event: Mapping[str, Any]) -> None:
    if event.get("schema_id") != DELTA_SCHEMA_ID:
        raise AtomLedgerIntegrityError("delta event schema is invalid")
    validate_signed_hash(event, "event_hash", "delta event")
    base_patch = parse_patch(event.get("base_patch"), "delta base_patch")
    target_patch = parse_patch(event.get("target_patch"), "delta target_patch")
    if target_patch <= base_patch:
        raise AtomLedgerIntegrityError("delta target patch must follow its base patch")
    parse_utc(event.get("effective_at"), "delta effective_at")
    source_receipt = event.get("source_receipt")
    if not isinstance(source_receipt, dict):
        raise AtomLedgerIntegrityError("delta source_receipt must be an object")
    parse_utc(
        source_receipt.get("revision_timestamp"), "delta source revision_timestamp"
    )
    revision_id = source_receipt.get("revision_id")
    if not isinstance(revision_id, int) or revision_id <= 0:
        raise AtomLedgerIntegrityError("delta source revision_id must be positive")
    operations = event.get("operations")
    if not isinstance(operations, list) or not operations:
        raise AtomLedgerIntegrityError("delta event must contain operations")
    coverage = event.get("coverage")
    if not isinstance(coverage, dict):
        raise AtomLedgerIntegrityError("delta coverage must be an object")
    parsed_count = coverage.get("parsed_change_count")
    unsupported_count = coverage.get("unparsed_or_unsupported_change_count")
    candidate_count = coverage.get("patch_page_candidate_change_count")
    unsupported = coverage.get("unparsed_or_unsupported_changes")
    if coverage.get("status") not in {"partial", "complete"}:
        raise AtomLedgerIntegrityError("delta coverage status is invalid")
    if not isinstance(coverage.get("model_ready"), bool):
        raise AtomLedgerIntegrityError("delta model_ready must be boolean")
    if not all(
        isinstance(value, int) and value >= 0
        for value in (parsed_count, unsupported_count, candidate_count)
    ):
        raise AtomLedgerIntegrityError(
            "delta coverage counts must be non-negative integers"
        )
    if parsed_count != len(operations):
        raise AtomLedgerIntegrityError(
            "delta parsed change count must equal operation count"
        )
    if candidate_count != parsed_count + unsupported_count:
        raise AtomLedgerIntegrityError("delta coverage counts do not reconcile")
    if not isinstance(unsupported, list):
        raise AtomLedgerIntegrityError("delta unsupported change list must be an array")
    listed_count = sum(
        row.get("change_count", -1) if isinstance(row, dict) else -1
        for row in unsupported
    )
    if listed_count != unsupported_count:
        raise AtomLedgerIntegrityError(
            "delta unsupported change list count does not reconcile"
        )
    if coverage["status"] == "partial" and coverage["model_ready"] is not False:
        raise AtomLedgerIntegrityError("partial delta coverage cannot be model ready")
    if (
        coverage["status"] == "partial"
        and event.get("authority_status") != PARTIAL_DELTA_AUTHORITY_STATUS
    ):
        raise AtomLedgerIntegrityError("partial delta authority status is invalid")
    operation_ids: set[str] = set()
    atom_ids: set[str] = set()
    for operation in operations:
        if not isinstance(operation, dict):
            raise AtomLedgerIntegrityError("delta operation must be an object")
        operation_id = operation.get("operation_id")
        atom_id = operation.get("atom_id")
        if not isinstance(operation_id, str) or not operation_id:
            raise AtomLedgerIntegrityError("delta operation_id must be a string")
        if not isinstance(atom_id, str) or not atom_id:
            raise AtomLedgerIntegrityError("delta atom_id must be a string")
        if operation_id in operation_ids or atom_id in atom_ids:
            raise AtomLedgerConflictError("delta event repeats an operation or atom")
        operation_ids.add(operation_id)
        atom_ids.add(atom_id)
        if operation.get("op") not in {"add", "change", "deactivate"}:
            raise AtomLedgerIntegrityError(
                f"unsupported delta operation: {operation.get('op')}"
            )


def _record_hash(record: Mapping[str, Any]) -> str:
    return signed_hash(record, "record_hash")


def _apply_add(atoms: dict[str, dict[str, Any]], operation: Mapping[str, Any]) -> None:
    atom_id = str(operation["atom_id"])
    if atom_id in atoms:
        raise AtomLedgerConflictError(f"add operation targets existing atom {atom_id}")
    record = operation.get("record")
    if not isinstance(record, dict) or record.get("atom_id") != atom_id:
        raise AtomLedgerIntegrityError(
            f"add operation {operation['operation_id']} has an invalid record"
        )
    identity = record.get("identity")
    primary = record.get("primary_category")
    categories = record.get("categories")
    if (
        not isinstance(identity, dict)
        or not isinstance(primary, str)
        or not isinstance(categories, list)
        or not categories
    ):
        raise AtomLedgerIntegrityError(
            f"add operation {operation['operation_id']} has invalid identity"
        )
    if stable_atom_id(primary, identity) != atom_id:
        raise AtomLedgerIntegrityError(
            f"add operation {operation['operation_id']} has an unstable atom ID"
        )
    candidate = deepcopy(record)
    candidate["record_hash"] = _record_hash(candidate)
    atoms[atom_id] = candidate


def _require_expected_record(
    atoms: dict[str, dict[str, Any]], operation: Mapping[str, Any]
) -> dict[str, Any]:
    atom_id = str(operation["atom_id"])
    record = atoms.get(atom_id)
    if record is None:
        raise AtomLedgerConflictError(f"operation targets missing atom {atom_id}")
    expected = operation.get("expected_record_hash")
    if expected != record.get("record_hash"):
        raise AtomLedgerConflictError(
            f"operation prior hash differs for atom {atom_id}"
        )
    return record


def _apply_change(
    atoms: dict[str, dict[str, Any]], operation: Mapping[str, Any]
) -> None:
    record = _require_expected_record(atoms, operation)
    if not record.get("active"):
        raise AtomLedgerConflictError(
            f"change operation targets inactive atom {operation['atom_id']}"
        )
    changes = operation.get("fields")
    if not isinstance(changes, dict) or not changes:
        raise AtomLedgerIntegrityError("change operation must contain fields")
    candidate = deepcopy(record)
    for field_name, cell in changes.items():
        if field_name not in candidate["fields"]:
            raise AtomLedgerConflictError(
                f"change operation targets missing field {field_name}"
            )
        required = {"value", "source", "unit", "confidence", "missing", "authority"}
        if not isinstance(cell, dict) or set(cell) != required:
            raise AtomLedgerIntegrityError(
                f"change field {field_name} has invalid schema"
            )
        candidate["fields"][field_name] = deepcopy(cell)
        candidate["missing_mask"][field_name] = cell["missing"]
    candidate["record_hash"] = _record_hash(candidate)
    atoms[str(operation["atom_id"])] = candidate


def _apply_deactivate(
    atoms: dict[str, dict[str, Any]], operation: Mapping[str, Any]
) -> None:
    record = _require_expected_record(atoms, operation)
    if not record.get("active"):
        raise AtomLedgerConflictError(
            f"deactivate operation targets inactive atom {operation['atom_id']}"
        )
    candidate = deepcopy(record)
    candidate["active"] = False
    candidate["record_hash"] = _record_hash(candidate)
    atoms[str(operation["atom_id"])] = candidate


def _derived_snapshot(
    current: Mapping[str, Any],
    event: Mapping[str, Any],
    atoms: Mapping[str, dict[str, Any]],
    field_status_overrides: Mapping[str, str],
) -> dict[str, Any]:
    records = sorted(atoms.values(), key=lambda atom: atom["atom_id"])
    snapshot = {
        "schema_id": current["schema_id"],
        "ledger_schema_id": current["ledger_schema_id"],
        "snapshot_kind": "derived",
        "patch": event["target_patch"],
        "authority_status": event["authority_status"],
        "coverage": event["coverage"],
        "field_status_index": {
            "default_status": "unchanged_with_prior_authority",
            "overrides": dict(sorted(field_status_overrides.items())),
        },
        "base_snapshot_hash": current.get(
            "base_snapshot_hash", current["snapshot_hash"]
        ),
        "parent_snapshot_hash": current["snapshot_hash"],
        "event_hash": event["event_hash"],
        "atom_count": len(records),
        "atoms": records,
    }
    snapshot["snapshot_hash"] = canonical_sha256(snapshot)
    return snapshot


def replay_events(
    base_snapshot: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    as_of_patch: str,
    knowledge_cutoff: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay a caller-ordered chain and reject any future patch or source evidence."""

    current = validate_base_snapshot(base_snapshot)
    if parse_patch(current["patch"]) > parse_patch(as_of_patch, "as_of_patch"):
        raise AtomLedgerFutureDataError("base snapshot is later than as_of_patch")
    cutoff = parse_utc(knowledge_cutoff, "knowledge_cutoff")
    atoms = {atom["atom_id"]: deepcopy(atom) for atom in current["atoms"]}
    original_hashes = {atom_id: atom["record_hash"] for atom_id, atom in atoms.items()}
    seen_events: set[str] = set()
    seen_operations: set[str] = set()
    event_hashes: list[str] = []
    previous_event_hash: str | None = None
    changed_atom_ids: set[str] = set()
    latest_field_status_overrides: dict[str, str] = {}
    for event in events:
        validate_delta_event(event)
        event_hash = str(event["event_hash"])
        if event_hash in seen_events:
            raise AtomLedgerConflictError(
                f"delta event was applied twice: {event_hash}"
            )
        if event.get("base_patch") != current["patch"]:
            raise AtomLedgerIntegrityError("delta base patch breaks the ordered chain")
        if event.get("previous_snapshot_hash") != current["snapshot_hash"]:
            raise AtomLedgerIntegrityError(
                "delta previous snapshot hash breaks the chain"
            )
        if event.get("previous_event_hash") != previous_event_hash:
            raise AtomLedgerIntegrityError("delta previous event hash breaks the chain")
        if parse_patch(event["target_patch"]) > parse_patch(as_of_patch, "as_of_patch"):
            raise AtomLedgerFutureDataError(
                "delta target patch is later than as_of_patch"
            )
        source_time = parse_utc(
            event["source_receipt"]["revision_timestamp"], "source revision_timestamp"
        )
        if source_time > cutoff:
            raise AtomLedgerFutureDataError(
                "delta source revision is later than knowledge_cutoff"
            )
        event_field_status_overrides: dict[str, str] = {}
        for operation in event["operations"]:
            operation_id = str(operation["operation_id"])
            if operation_id in seen_operations:
                raise AtomLedgerConflictError(
                    f"delta operation was applied twice: {operation_id}"
                )
            seen_operations.add(operation_id)
            if operation["op"] == "add":
                _apply_add(atoms, operation)
                for field_name in atoms[str(operation["atom_id"])]["fields"]:
                    event_field_status_overrides[
                        f"{operation['atom_id']}/{field_name}"
                    ] = "added_by_delta"
            elif operation["op"] == "change":
                _apply_change(atoms, operation)
                for field_name in operation["fields"]:
                    event_field_status_overrides[
                        f"{operation['atom_id']}/{field_name}"
                    ] = "refreshed_by_delta"
            else:
                _apply_deactivate(atoms, operation)
            changed_atom_ids.add(str(operation["atom_id"]))
        latest_field_status_overrides = event_field_status_overrides
        current = _derived_snapshot(current, event, atoms, event_field_status_overrides)
        seen_events.add(event_hash)
        event_hashes.append(event_hash)
        previous_event_hash = event_hash
    unchanged_ids = sorted(set(original_hashes) - changed_atom_ids)
    for atom_id in unchanged_ids:
        if atoms[atom_id]["record_hash"] != original_hashes[atom_id]:
            raise AtomLedgerIntegrityError(
                f"unchanged atom did not carry forward: {atom_id}"
            )
    prior_binary_fields = sorted(
        f"{atom_id}/{field_name}"
        for atom_id in unchanged_ids
        for field_name, cell in atoms[atom_id]["fields"].items()
        if cell["authority"] == "binary_patch_bound"
    )
    all_field_refs = sorted(
        f"{atom_id}/{field_name}"
        for atom_id, atom in atoms.items()
        for field_name in atom["fields"]
    )
    unchanged_field_refs = sorted(
        set(all_field_refs) - set(latest_field_status_overrides)
    )
    final_field_status_index = current.get(
        "field_status_index",
        {"default_status": "base_patch_bound", "overrides": {}},
    )
    receipt = {
        "schema_id": RECEIPT_SCHEMA_ID,
        "authority_status": current["authority_status"],
        "coverage": current["coverage"],
        "field_status_index": final_field_status_index,
        "as_of_patch": as_of_patch,
        "knowledge_cutoff": knowledge_cutoff,
        "base_snapshot_hash": base_snapshot["snapshot_hash"],
        "event_hashes": event_hashes,
        "final_snapshot_hash": current["snapshot_hash"],
        "atom_count": len(atoms),
        "active_atom_count": sum(bool(atom["active"]) for atom in atoms.values()),
        "changed_atom_count": len(changed_atom_ids),
        "unchanged_atom_count": len(unchanged_ids),
        "unchanged_field_count": len(unchanged_field_refs),
        "unchanged_fields_sha256": canonical_sha256(unchanged_field_refs),
        "binary_only_unchanged_field_status": "unchanged_with_prior_authority",
        "binary_only_unchanged_field_count": len(prior_binary_fields),
        "binary_only_unchanged_fields_sha256": canonical_sha256(prior_binary_fields),
    }
    receipt["receipt_hash"] = canonical_sha256(receipt)
    return current, receipt


def resolve_model_ready_snapshot(
    base_snapshot: Mapping[str, Any],
    events: Iterable[Mapping[str, Any]],
    *,
    as_of_patch: str,
    knowledge_cutoff: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one model-ready chain within the declared LCC vertical-slice scope."""

    event_list = list(events)
    for event in event_list:
        validate_delta_event(event)
        coverage = event["coverage"]
        if coverage["status"] != "complete" or coverage["model_ready"] is not True:
            raise AtomLedgerCoverageError(
                f"patch {event['target_patch']} delta is incomplete for model use"
            )
    return replay_events(
        base_snapshot,
        event_list,
        as_of_patch=as_of_patch,
        knowledge_cutoff=knowledge_cutoff,
    )
