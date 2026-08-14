from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from lol_kills import refresh_ledger
from lol_kills.refresh_ledger import RefreshRunLedger, reusable_stage_receipt


def test_stage_receipt_is_reusable_only_for_exact_inputs(tmp_path: Path) -> None:
    commit = "a" * 40
    ledger = RefreshRunLedger(
        runtime_root=tmp_path,
        scheduled_for=datetime(2026, 8, 11, tzinfo=timezone.utc),
        worker_git_commit=commit,
        transform_version="transform-v1",
        source_file_sha256="b" * 64,
        source_observed_through="2026-08-11T10:50:41Z",
        requirements_lock_sha256="c" * 64,
    )
    ledger.advance("ingest")
    receipt = ledger.path.parent / "validate_source.json"

    assert reusable_stage_receipt(
        receipt,
        stage="validate_source",
        fingerprint=ledger.fingerprint,
        transform_version="transform-v1",
        worker_git_commit=commit,
        requirements_lock_sha256="c" * 64,
    ) is not None
    assert reusable_stage_receipt(
        receipt,
        stage="validate_source",
        fingerprint="c" * 64,
        transform_version="transform-v1",
        worker_git_commit=commit,
        requirements_lock_sha256="c" * 64,
    ) is None


def test_failed_run_keeps_bounded_failure_and_stage_metrics(tmp_path: Path) -> None:
    ledger = RefreshRunLedger(
        runtime_root=tmp_path,
        scheduled_for=datetime(2026, 8, 11, tzinfo=timezone.utc),
        worker_git_commit="a" * 40,
        transform_version="transform-v1",
        source_file_sha256="b" * 64,
        source_observed_through=None,
        requirements_lock_sha256="c" * 64,
    )
    ledger.advance("ingest")
    ledger.fail(RuntimeError("x" * 3000))
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))

    assert payload["status"] == "error"
    assert payload["stage"] == "ingest"
    assert len(payload["failure_detail"]) == 2000
    assert payload["stage_durations"]["validate_source"]["wall_seconds"] >= 0
    assert payload["requirements_lock_sha256"] == "c" * 64


def test_release_reference_is_rejected_before_staging_completes(tmp_path: Path) -> None:
    ledger = RefreshRunLedger(
        runtime_root=tmp_path,
        scheduled_for=datetime(2026, 8, 11, tzinfo=timezone.utc),
        worker_git_commit="a" * 40,
        transform_version="transform-v1",
        source_file_sha256="b" * 64,
        source_observed_through=None,
        requirements_lock_sha256="c" * 64,
    )

    with pytest.raises(ValueError, match="before the release is staged"):
        ledger.advance("stage_release", release_id="v2026.08.11.120000")

    ledger.advance("stage_release")
    ledger.advance("activate_release", release_id="v2026.08.11.120000")
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))
    assert payload["release_id"] == "v2026.08.11.120000"


def test_same_input_after_failure_records_retry_relationship(tmp_path: Path) -> None:
    inputs = {
        "runtime_root": tmp_path,
        "scheduled_for": datetime(2026, 8, 11, tzinfo=timezone.utc),
        "worker_git_commit": "a" * 40,
        "transform_version": "transform-v1",
        "source_file_sha256": "b" * 64,
        "source_observed_through": None,
        "requirements_lock_sha256": "c" * 64,
    }
    first = RefreshRunLedger(**inputs)
    first.fail(RuntimeError("temporary failure"))

    second = RefreshRunLedger(**inputs)
    payload = json.loads(second.path.read_text(encoding="utf-8"))

    assert second.retry_of == first.run_id
    assert payload["retry_of"] == first.run_id


def test_worker_commit_uses_real_head_and_requires_a_clean_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    monkeypatch.setenv("SCRYGLASS_WORKER_COMMIT", commit)
    with patch.object(
        refresh_ledger,
        "_git_output",
        side_effect=[str(tmp_path.resolve()), commit, ""],
    ) as git:
        assert refresh_ledger.worker_commit(tmp_path, require_clean=True) == commit
    assert git.call_args_list[-1].args[1:] == (
        "status",
        "--porcelain=v1",
        "--untracked-files=normal",
    )


def test_worker_commit_rejects_an_environment_value_that_differs_from_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SCRYGLASS_WORKER_COMMIT", "b" * 40)
    with patch.object(
        refresh_ledger,
        "_git_output",
        side_effect=[str(tmp_path.resolve()), "a" * 40],
    ), pytest.raises(RuntimeError, match="does not match HEAD"):
        refresh_ledger.worker_commit(tmp_path)


def test_worker_commit_rejects_dirty_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "a" * 40
    monkeypatch.setenv("SCRYGLASS_WORKER_COMMIT", commit)
    with patch.object(
        refresh_ledger,
        "_git_output",
        side_effect=[str(tmp_path.resolve()), commit, " M tracked.py"],
    ), pytest.raises(RuntimeError, match="uncommitted"):
        refresh_ledger.worker_commit(tmp_path, require_clean=True)
