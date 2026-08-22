from __future__ import annotations

from pathlib import Path
import subprocess
import json

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
        trust_manifest=tmp_path / "trust.json",
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
    assert "--crosswalk-receipt-file-sha256" in stages[2].jobs[1].command
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


def test_real_v14_receipt_separates_payload_and_file_hashes() -> None:
    crosswalk = Path(
        "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
        "oe-leaguepedia-series-crosswalk-v14-dedup-portable.json"
    )
    crosswalk_receipt = Path(
        "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
        "oe-leaguepedia-series-crosswalk-v14-dedup-portable.receipt.json"
    )
    source_receipt = Path(
        "/private/tmp/scryglass-source-verify-dedup-v4/"
        "future-value-source-receipt.json"
    )
    if not all(path.is_file() for path in (crosswalk, crosswalk_receipt, source_receipt)):
        pytest.skip("v14 local crosswalk fixture is not available")

    receipt = json.loads(crosswalk_receipt.read_text(encoding="utf-8"))
    assert receipt["crosswalk_sha256"] != receipt["artifact"]["sha256"]
    binding = fourway._validate_crosswalk_binding(
        crosswalk,
        crosswalk_receipt,
        source_receipt=json.loads(source_receipt.read_text(encoding="utf-8")),
        expected_receipt_file_sha256=fourway._sha256_path(crosswalk_receipt),
    )
    assert binding["artifact_sha256"] == receipt["artifact"]["sha256"]
    assert binding["crosswalk_sha256"] == receipt["crosswalk_sha256"]


def test_trust_manifest_pins_source_and_series(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.trust_manifest.write_text("{}", encoding="utf-8")
    source = {
        "source_game_count": 2,
        "source_identity_sha256": "1" * 64,
        "model_eligible_game_count": 1,
        "model_eligible_identity_sha256": "2" * 64,
        "receipt_sha256": "3" * 64,
    }
    payload = {
        "schema_version": fourway.TRUST_MANIFEST_SCHEMA_VERSION,
        "status": "research_only",
        "source": {
            "source_game_count": 2,
            "source_identity_sha256": "1" * 64,
            "model_eligible_game_count": 1,
            "model_eligible_identity_sha256": "2" * 64,
            "source_receipt_sha256": "3" * 64,
            "source_receipt_file_sha256": "4" * 64,
            "freeze_file_sha256": "5" * 64,
        },
        "series": {
            "crosswalk_receipt_file_sha256": "a" * 64,
            "crosswalk_sha256": "6" * 64,
            "crosswalk_assignment_sha256": "7" * 64,
        },
        "authority": {"research_only": True, "deployment": False},
    }
    payload["manifest_sha256"] = fourway._sha256_bytes(
        fourway._canonical_bytes(payload)
    )
    config.trust_manifest.write_bytes(fourway._canonical_bytes(payload))
    record = fourway._validate_trust_manifest(
        config,
        source_receipt=source,
        source_receipt_file_sha256="4" * 64,
        freeze_file_sha256="5" * 64,
        crosswalk_binding={
            "crosswalk_sha256": "6" * 64,
            "assignment_sha256": "7" * 64,
        },
    )
    assert record["manifest_sha256"] == payload["manifest_sha256"]

    payload["source"]["source_game_count"] = 3
    body = dict(payload)
    body.pop("manifest_sha256")
    payload["manifest_sha256"] = fourway._sha256_bytes(
        fourway._canonical_bytes(body)
    )
    config.trust_manifest.write_bytes(fourway._canonical_bytes(payload))
    with pytest.raises(fourway.FourwayRunError, match="immutable"):
        fourway._validate_trust_manifest(
            config,
            source_receipt=source,
            source_receipt_file_sha256="4" * 64,
            freeze_file_sha256="5" * 64,
            crosswalk_binding={
                "crosswalk_sha256": "6" * 64,
                "assignment_sha256": "7" * 64,
            },
        )


def test_outer_cutoff_must_equal_generated_first_validation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    root = config.output_root / "stages/fold-specs"
    root.mkdir(parents=True)
    folds = []
    for fold, day in enumerate((1, 5, 9), start=1):
        cutoff = f"2026-08-{day:02d}T00:00:00Z"
        record = {
            "fold": fold,
            "fit_window_end": cutoff,
            "validation_start": cutoff,
        }
        folds.append(record)
        (root / f"fold-{fold}-spec.json").write_text(
            json.dumps({"fold": fold, "fit_window_end": cutoff}), encoding="utf-8"
        )
    config.source_receipt.write_text(
        json.dumps({"receipt_sha256": "b" * 64}), encoding="utf-8"
    )
    bundle = {"source": {"source_receipt_sha256": "b" * 64}, "folds": folds}
    bundle["bundle_sha256"] = fourway._sha256_bytes(
        fourway._canonical_bytes(bundle)
    )
    (root / "fold-spec-bundle.json").write_bytes(fourway._canonical_bytes(bundle))
    matching = fourway.RunConfig(
        **{**config.__dict__, "outer_evaluation_start": "2026-08-01T00:00:00+00:00"}
    )
    assert fourway._validate_fold_plan_cutoff(matching) == "2026-08-01T00:00:00Z"
    mismatched = fourway.RunConfig(
        **{**config.__dict__, "outer_evaluation_start": "2026-08-02T00:00:00Z"}
    )
    with pytest.raises(fourway.FourwayRunError, match="differs from the generated"):
        fourway._validate_fold_plan_cutoff(mismatched)


def test_scaling_binding_rejects_crosswalk_or_assignment_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.source_receipt.write_text(
        json.dumps(
            {
                "receipt_sha256": "b" * 64,
                "source_identity_sha256": "c" * 64,
                "model_eligible_game_count": 2,
                "model_eligible_identity_sha256": "1" * 64,
            }
        ),
        encoding="utf-8",
    )
    trust = {
        "series": {
            "crosswalk_sha256": "d" * 64,
            "crosswalk_assignment_sha256": "8" * 64,
        }
    }
    config.trust_manifest.write_text(json.dumps(trust), encoding="utf-8")
    folds_root = config.output_root / "stages/fold-specs"
    producers = config.output_root / "stages/fold-producers"
    folds_root.mkdir(parents=True)
    for fold in range(1, 4):
        spec = {
            "fold": fold,
            "fit_window_end": "2026-08-01T00:00:00Z",
            "train_game_ids": [f"t{fold}"],
            "validation_game_ids": [f"v{fold}"],
        }
        (folds_root / f"fold-{fold}-spec.json").write_text(
            json.dumps(spec), encoding="utf-8"
        )
        binding = {
            "schema_version": "scryglass:future-value-scaling-series-binding:v1",
            "status": "research_only",
            "fold": fold,
            "fit_window_end": spec["fit_window_end"],
            "train_game_ids": spec["train_game_ids"],
            "validation_game_ids": spec["validation_game_ids"],
            "source_receipt_sha256": "b" * 64,
            "source_identity_sha256": "c" * 64,
            "model_eligible_game_count": 2,
            "model_eligible_identity_sha256": "1" * 64,
            "crosswalk_receipt_file_sha256": "a" * 64,
            "crosswalk_sha256": "d" * 64,
            "crosswalk_assignment_sha256": "8" * 64,
            "eligible_series_assignment_sha256": "e" * 64,
            "fold_series_assignment_sha256": "f" * 64,
            "authority": {"research_only": True, "deployment": False},
        }
        binding["receipt_sha256"] = fourway._sha256_bytes(
            fourway._canonical_bytes(binding)
        )
        target = producers / f"fold-{fold}/scaling-v2"
        target.mkdir(parents=True)
        (target / "scaling-series-binding.json").write_bytes(
            fourway._canonical_bytes(binding)
        )
    fourway._validate_scaling_series_bindings(config)

    path = producers / "fold-2/scaling-v2/scaling-series-binding.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["eligible_series_assignment_sha256"] = "9" * 64
    body = dict(changed)
    body.pop("receipt_sha256")
    changed["receipt_sha256"] = fourway._sha256_bytes(
        fourway._canonical_bytes(body)
    )
    path.write_bytes(fourway._canonical_bytes(changed))
    with pytest.raises(fourway.FourwayRunError, match="different series assignments"):
        fourway._validate_scaling_series_bindings(config)
