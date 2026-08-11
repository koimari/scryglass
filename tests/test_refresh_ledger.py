from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

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
    )
    ledger.advance("ingest")
    receipt = ledger.path.parent / "validate_source.json"

    assert reusable_stage_receipt(
        receipt,
        stage="validate_source",
        fingerprint=ledger.fingerprint,
        transform_version="transform-v1",
        worker_git_commit=commit,
    ) is not None
    assert reusable_stage_receipt(
        receipt,
        stage="validate_source",
        fingerprint="c" * 64,
        transform_version="transform-v1",
        worker_git_commit=commit,
    ) is None


def test_failed_run_keeps_bounded_failure_and_stage_metrics(tmp_path: Path) -> None:
    ledger = RefreshRunLedger(
        runtime_root=tmp_path,
        scheduled_for=datetime(2026, 8, 11, tzinfo=timezone.utc),
        worker_git_commit="a" * 40,
        transform_version="transform-v1",
        source_file_sha256="b" * 64,
        source_observed_through=None,
    )
    ledger.advance("ingest")
    ledger.fail(RuntimeError("x" * 3000))
    payload = json.loads(ledger.path.read_text(encoding="utf-8"))

    assert payload["status"] == "error"
    assert payload["stage"] == "ingest"
    assert len(payload["failure_detail"]) == 2000
    assert payload["stage_durations"]["validate_source"]["wall_seconds"] >= 0


def test_release_reference_is_rejected_before_staging_completes(tmp_path: Path) -> None:
    ledger = RefreshRunLedger(
        runtime_root=tmp_path,
        scheduled_for=datetime(2026, 8, 11, tzinfo=timezone.utc),
        worker_git_commit="a" * 40,
        transform_version="transform-v1",
        source_file_sha256="b" * 64,
        source_observed_through=None,
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
    }
    first = RefreshRunLedger(**inputs)
    first.fail(RuntimeError("temporary failure"))

    second = RefreshRunLedger(**inputs)
    payload = json.loads(second.path.read_text(encoding="utf-8"))

    assert second.retry_of == first.run_id
    assert payload["retry_of"] == first.run_id
