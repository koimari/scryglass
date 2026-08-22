from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from benchmarks import run_future_value_fourway as fourway


def _config(tmp_path: Path) -> fourway.RunConfig:
    return fourway.RunConfig(
        source_root=tmp_path / "source",
        source_receipt=tmp_path / "source-receipt.json",
        freeze=tmp_path / "freeze.json",
        freeze_root=tmp_path,
        crosswalk=tmp_path / "crosswalk.json",
        crosswalk_receipt=tmp_path / "crosswalk-receipt.json",
        crosswalk_receipt_file_sha256="a" * 64,
        output_root=tmp_path / "run",
        outer_evaluation_start="2026-08-01T00:00:00Z",
        workers=2,
        max_log_bytes=8,
    )


def test_plan_has_sequential_stages_and_parallel_jobs(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stages = fourway.build_stage_plan(config)

    assert [stage.name for stage in stages] == [
        "phase",
        "fold_specs",
        "fold_producers",
        "bundle",
        "calibration",
        "evaluations",
        "paired_uncertainty",
    ]
    assert len(stages[2].jobs) == 6
    assert len(stages[5].jobs) == 4
    assert stages[2].jobs[0].command[:3] == (
        fourway.sys.executable,
        "-m",
        "benchmarks.build_current_rating_fold_artifact",
    )
    assert "--crosswalk-receipt-file-sha256" in stages[2].jobs[0].command
    assert "__PHASE_ARTIFACT_SHA256__" in stages[5].jobs[0].command
    assert "--rating-variant" in stages[5].jobs[-1].command
    assert stages[5].jobs[-1].command[stages[5].jobs[-1].command.index("--rating-variant") + 1] == "both"


def test_plan_binds_runtime_phase_hashes_into_each_evaluation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stages = fourway.build_stage_plan(
        config,
        phase_artifact_sha256="b" * 64,
        phase_receipt_file_sha256="c" * 64,
    )
    for job in stages[5].jobs:
        command = list(job.command)
        assert command[command.index("--phase-artifact-sha256") + 1] == "b" * 64
        assert command[command.index("--phase-receipt-file-sha256") + 1] == "c" * 64


def test_stage_failure_writes_bounded_logs_and_failed_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    config.source_receipt.write_text('{"receipt_sha256":"' + "a" * 64 + '"}', encoding="utf-8")
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"input")
    output_root = config.output_root / "stage"
    expected = output_root / "result.json"
    job = fourway.Job(
        name="synthetic",
        command=("synthetic", "--fail"),
        output_roots=(output_root,),
        expected_files=(expected,),
        input_paths=(input_path,),
    )
    stage = fourway.Stage(
        name="synthetic",
        jobs=(job,),
        output_roots=(output_root,),
        expected_files=(expected,),
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(
            args=args[0] if args else [],
            returncode=9,
            stdout=b"0123456789abcdef",
            stderr=b"error output",
        )

    monkeypatch.setattr(fourway.subprocess, "run", fake_run)
    with pytest.raises(fourway.FourwayRunError, match="stage synthetic failed"):
        fourway._execute_stage(stage, config, resume=False)

    receipt_path = config.output_root / "receipts" / "synthetic.json"
    receipt = fourway._validate_receipt_hash(receipt_path, "synthetic stage")
    assert receipt["status"] == "failed"
    assert receipt["jobs"][0]["exit_code"] == 9
    assert receipt["logs"]["synthetic"]["stdout"]["truncated"] is True
    assert receipt["logs"]["synthetic"]["stdout"]["bytes"] == 8
    assert receipt["missing_outputs"] == [str(expected)]


def test_resume_requires_receipt_and_validates_output_hashes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    input_path = tmp_path / "input.bin"
    input_path.write_bytes(b"input")
    output_root = config.output_root / "stage"
    output = output_root / "result.json"
    output_root.mkdir()
    output.write_bytes(b"result")
    job = fourway.Job(
        name="synthetic",
        command=("synthetic",),
        output_roots=(output_root,),
        expected_files=(output,),
        input_paths=(input_path,),
    )
    stage = fourway.Stage(
        name="synthetic",
        jobs=(job,),
        output_roots=(output_root,),
        expected_files=(output,),
    )
    receipt_payload = {
        "schema_version": fourway.STAGE_RECEIPT_SCHEMA_VERSION,
        "status": "completed",
        "stage": "synthetic",
        "stage_plan_sha256": fourway._plan_digest(stage),
        "source": {"source_receipt_path": str(config.source_receipt)},
        "inputs": {str(input_path): fourway._file_record(input_path)},
        "outputs": [fourway._file_record(output)],
        "logs": {},
    }
    receipt_path = config.output_root / "receipts" / "synthetic.json"
    fourway._write_receipt(receipt_path, receipt_payload)
    resumed = fourway._execute_stage(stage, config, resume=True)
    assert resumed["status"] == "completed"

    output.write_bytes(b"changed")
    with pytest.raises(fourway.FourwayRunError, match="output hash changed"):
        fourway._execute_stage(stage, config, resume=True)


def test_resume_rejects_output_without_completed_receipt(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.output_root.mkdir()
    output_root = config.output_root / "stage"
    output_root.mkdir(parents=True)
    (output_root / "partial").write_text("partial", encoding="utf-8")
    job = fourway.Job(
        name="synthetic",
        command=("synthetic",),
        output_roots=(output_root,),
        expected_files=(output_root / "result.json",),
        input_paths=(),
    )
    stage = fourway.Stage(
        name="synthetic",
        jobs=(job,),
        output_roots=(output_root,),
        expected_files=job.expected_files,
    )
    with pytest.raises(fourway.FourwayRunError, match="output without a valid"):
        fourway._execute_stage(stage, config, resume=True)
