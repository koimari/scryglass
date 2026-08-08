from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import (
    contract, execution_approval, result, runner,
)


CORE = "c" * 64


def _approval() -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_id": execution_approval.APPROVAL_SCHEMA,
        "approval_id": "approval-1",
        "run_id": "run-1",
        "state": "APPROVED_PRIVATE_DEVELOPMENT_EXECUTION",
        "reviewer_root": "KOI_MARI",
        "authority_scope": execution_approval.SCOPE,
        "runner_core_sha256": CORE,
        "prefit": execution_approval.PREFIT,
        "source_pins": {
            "G1": contract.G1,
            "G1_features": contract.G1_FEATURES,
            "G2": contract.G2,
            "clusters": contract.CLUSTERS,
        },
        "allowed_partitions": ["TRAIN", "DEVELOPMENT", "VALIDATION"],
        "final_holdout": False,
        "paths": {
            "approval_locator": execution_approval.APPROVAL_LOCATOR,
            "ledger_locator": execution_approval.LEDGER_LOCATOR,
            "result_locator": execution_approval.RESULT_LOCATOR,
        },
        "claim_ceiling": result.CLAIM_CEILING,
    }
    return {**unsigned, "approval_sha256": execution_approval.sha256(unsigned)}


def _entry(state: str, sequence: int, **extra: object) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_id": execution_approval.LEDGER_SCHEMA,
        "state": state,
        "approval_id": "approval-1",
        "run_id": "run-1",
        "runner_core_sha256": CORE,
        "approval_sha256": _approval()["approval_sha256"],
        "result_locator": execution_approval.RESULT_LOCATOR,
        "sequence": sequence,
        **extra,
    }
    return {**unsigned, "entry_sha256": execution_approval.sha256(unsigned)}


def _history() -> list[dict[str, object]]:
    started = _entry("STARTED", 1, started_at="2026-07-30T00:00:00Z")
    completed = _entry(
        "COMPLETED",
        2,
        started_entry_sha256=started["entry_sha256"],
        result_artifact_sha256="d" * 64,
        result_raw_sha256="e" * 64,
        completed_at="2026-07-30T00:00:01Z",
    )
    return [started, completed]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value.update({"runner_core_sha256": "0" * 64}),
        lambda value: value.update({"run_id": "other"}),
        lambda value: value.update({"reviewer_root": "OTHER"}),
        lambda value: value.update({"allowed_partitions": ["TRAIN", "DEVELOPMENT"]}),
        lambda value: value.update({"final_holdout": True}),
        lambda value: value["paths"].update({"result_locator": "elsewhere"}),
        lambda value: value["claim_ceiling"].update({"publication": True}),
        lambda value: value["source_pins"]["G2"].update({"artifact_raw_sha256": "0" * 64}),
        lambda value: value["authority_scope"].update({"private_model_fit": False}),
    ],
)
def test_approval_mutations_fail(mutate) -> None:
    value = copy.deepcopy(_approval())
    mutate(value)
    unsigned = dict(value)
    unsigned.pop("approval_sha256")
    value["approval_sha256"] = execution_approval.sha256(unsigned)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_approval_payload(
            value, expected_runner_core_sha256=CORE, expected_run_id="run-1"
        )


def test_approval_requires_canonical_newline_and_safe_single_link(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(execution_approval, "ROOT", tmp_path)
    path = tmp_path / execution_approval.APPROVAL_LOCATOR
    path.parent.mkdir(parents=True)
    path.write_bytes(execution_approval.canonical_bytes(_approval()) + b"\n")
    assert execution_approval.load_approval(
        expected_runner_core_sha256=CORE, expected_run_id="run-1"
    )["approval_id"] == "approval-1"
    target = path.with_name("target.json")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.load_approval(
            expected_runner_core_sha256=CORE, expected_run_id="run-1"
        )
    path.unlink()
    os.link(target, path)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.load_approval(
            expected_runner_core_sha256=CORE, expected_run_id="run-1"
        )


def test_ledger_canonical_malformed_symlink_and_hardlink_fail(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(execution_approval, "ROOT", tmp_path)
    path = tmp_path / execution_approval.LEDGER_LOCATOR
    path.parent.mkdir(parents=True)
    path.write_bytes(b"not-json\n")
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.load_ledger()
    path.unlink()
    target = path.with_name("target")
    target.write_bytes(b"")
    path.symlink_to(target)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.load_ledger()
    path.unlink()
    os.link(target, path)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.load_ledger()


def test_ledger_append_and_earliest_completed_order(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(execution_approval, "ROOT", tmp_path)
    path = tmp_path / execution_approval.LEDGER_LOCATOR
    path.parent.mkdir(parents=True)
    source = _history()
    written = []
    for item in source:
        unsigned = dict(item)
        unsigned.pop("schema_id")
        unsigned.pop("entry_sha256")
        written.append(execution_approval.append_ledger_entry(unsigned))
    loaded = execution_approval.load_ledger()
    assert loaded == written
    assert execution_approval.validate_ledger_history(
        loaded,
        approval=_approval(),
        expected_runner_core_sha256=CORE,
        expected_run_id="run-1",
    ) == "COMPLETED_TERMINAL"
    assert runner._earliest_completed(loaded, approval_id="approval-1") == written[1]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda entries: entries[0].update({"run_id": "foreign"}),
        lambda entries: entries[0].update({"runner_core_sha256": "0" * 64}),
        lambda entries: entries[0].update({"approval_id": "foreign"}),
        lambda entries: entries[0].update({"approval_sha256": "0" * 64}),
        lambda entries: entries[0].update({"started_at": None}),
        lambda entries: entries[0].update({"started_at": "bad"}),
        lambda entries: entries[1].update({"completed_at": "2026-07-29T23:59:59Z"}),
        lambda entries: entries[1].update({"started_entry_sha256": "0" * 64}),
        lambda entries: entries.__setitem__(0, entries[1]),
        lambda entries: entries.append(copy.deepcopy(entries[0])),
        lambda entries: entries.append(copy.deepcopy(entries[1])),
        lambda entries: entries[1].update({"sequence": 3}),
    ],
)
def test_hostile_ledger_history_mutations_fail(mutate) -> None:
    entries = copy.deepcopy(_history())
    mutate(entries)
    for entry in entries:
        if isinstance(entry, dict):
            unsigned = dict(entry)
            unsigned.pop("entry_sha256", None)
            entry["entry_sha256"] = execution_approval.sha256(unsigned)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_ledger_history(
            entries,
            approval=_approval(),
            expected_runner_core_sha256=CORE,
            expected_run_id="run-1",
        )


def test_orphan_duplicate_and_wrong_authoritative_references_fail() -> None:
    completed = _history()
    duplicate = _entry(
        "INVALID_DUPLICATE",
        3,
        authoritative_completed_entry_sha256="0" * 64,
        authoritative_result_artifact_sha256="0" * 64,
        recorded_at="2026-07-30T00:00:02Z",
    )
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_ledger_history(
            [duplicate],
            approval=_approval(),
            expected_runner_core_sha256=CORE,
            expected_run_id="run-1",
        )
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_ledger_history(
            [*completed, duplicate],
            approval=_approval(),
            expected_runner_core_sha256=CORE,
            expected_run_id="run-1",
        )


def test_valid_terminal_history_and_duplicate_bind_earliest_completion() -> None:
    entries = _history()
    completion = entries[1]
    entries.append(_entry(
        "INVALID_DUPLICATE",
        3,
        authoritative_completed_entry_sha256=completion["entry_sha256"],
        authoritative_result_artifact_sha256=completion["result_artifact_sha256"],
        recorded_at="2026-07-30T00:00:02Z",
    ))
    assert execution_approval.validate_ledger_history(
        entries,
        approval=_approval(),
        expected_runner_core_sha256=CORE,
        expected_run_id="run-1",
    ) == "COMPLETED_TERMINAL"


def _write_result(
    tmp_path: Path,
    *,
    approval_sha256: str,
    core_sha256: str = CORE,
    run_id: str = "run-1",
    started_sha256: str,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    path = tmp_path / execution_approval.RESULT_LOCATOR
    path.parent.mkdir(parents=True, exist_ok=True)
    unsigned: dict[str, object] = {
        "execution_binding": {
            "approval_sha256": approval_sha256,
            "runner_core_sha256": core_sha256,
            "run_id_sha256": execution_approval.sha256(run_id),
            "started_entry_sha256": started_sha256,
            "result_locator": execution_approval.RESULT_LOCATOR,
        }
    }
    payload = {**unsigned, "artifact_sha256": execution_approval.sha256(unsigned)}
    raw = execution_approval.canonical_bytes(payload) + b"\n"
    path.write_bytes(raw)
    completion = {
        "result_raw_sha256": __import__("hashlib").sha256(raw).hexdigest(),
        "result_artifact_sha256": payload["artifact_sha256"],
    }
    return path, payload, completion


def test_completed_result_missing_hash_and_binding_failures(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(execution_approval, "ROOT", tmp_path)
    monkeypatch.setattr(result, "validate_real", lambda _payload: None)
    approval = _approval()
    started = "a" * 64
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_completed_result(
            {"result_raw_sha256": "b" * 64, "result_artifact_sha256": "c" * 64},
            approval=approval,
            expected_runner_core_sha256=CORE,
            expected_run_id="run-1",
            started_entry_sha256=started,
        )
    path, payload, completion = _write_result(
        tmp_path, approval_sha256=approval["approval_sha256"], started_sha256=started
    )
    assert execution_approval.validate_completed_result(
        completion,
        approval=approval,
        expected_runner_core_sha256=CORE,
        expected_run_id="run-1",
        started_entry_sha256=started,
    ) == payload
    for field in ("result_raw_sha256", "result_artifact_sha256"):
        forged = dict(completion)
        forged[field] = "0" * 64
        with pytest.raises(execution_approval.ApprovalError):
            execution_approval.validate_completed_result(
                forged,
                approval=approval,
                expected_runner_core_sha256=CORE,
                expected_run_id="run-1",
                started_entry_sha256=started,
            )
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_completed_result(
            completion,
            approval=approval,
            expected_runner_core_sha256=CORE,
            expected_run_id="run-1",
            started_entry_sha256=started,
        )


@pytest.mark.parametrize("binding_field", ["approval_sha256", "runner_core_sha256", "run_id_sha256", "started_entry_sha256"])
def test_completed_result_self_rehashed_binding_mismatch_fails(
    binding_field: str, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(execution_approval, "ROOT", tmp_path)
    monkeypatch.setattr(result, "validate_real", lambda _payload: None)
    approval = _approval()
    started = "a" * 64
    path, payload, completion = _write_result(
        tmp_path, approval_sha256=approval["approval_sha256"], started_sha256=started
    )
    payload["execution_binding"][binding_field] = "0" * 64
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256")
    payload["artifact_sha256"] = execution_approval.sha256(unsigned)
    raw = execution_approval.canonical_bytes(payload) + b"\n"
    path.write_bytes(raw)
    completion["result_artifact_sha256"] = payload["artifact_sha256"]
    completion["result_raw_sha256"] = __import__("hashlib").sha256(raw).hexdigest()
    with pytest.raises(execution_approval.ApprovalError, match="binding"):
        execution_approval.validate_completed_result(
            completion,
            approval=approval,
            expected_runner_core_sha256=CORE,
            expected_run_id="run-1",
            started_entry_sha256=started,
        )


def test_completed_result_symlink_and_hardlink_fail(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(execution_approval, "ROOT", tmp_path)
    approval = _approval()
    started = "a" * 64
    path, _payload, completion = _write_result(
        tmp_path, approval_sha256=approval["approval_sha256"], started_sha256=started
    )
    target = path.with_name("result-target.json")
    path.rename(target)
    path.symlink_to(target)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_completed_result(
            completion, approval=approval, expected_runner_core_sha256=CORE,
            expected_run_id="run-1", started_entry_sha256=started,
        )
    path.unlink()
    os.link(target, path)
    with pytest.raises(execution_approval.ApprovalError):
        execution_approval.validate_completed_result(
            completion, approval=approval, expected_runner_core_sha256=CORE,
            expected_run_id="run-1", started_entry_sha256=started,
        )
