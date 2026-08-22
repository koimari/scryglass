from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from benchmarks import run_future_value_downstream as downstream


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _write(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")
    return path


def _config(tmp_path: Path, *, variant: str = "future_player_form") -> downstream.RunConfig:
    return downstream.RunConfig(
        fourway_root=tmp_path / "fourway",
        output_root=tmp_path / "downstream",
        selected_variant=variant,
        repository_root=Path.cwd(),
    )


def _inputs(tmp_path: Path) -> downstream.ResolvedInputs:
    root = tmp_path / "fourway"
    evaluations: dict[str, Path] = {}
    runtimes: dict[str, Path] = {}
    for variant in downstream.VARIANTS:
        evaluations[variant] = _write(root / "stages" / "evaluations" / variant / "model.json", {"variant": variant})
        runtimes[variant] = _write(root / "stages" / "evaluations" / variant / "runtime.json", {"variant": variant})
    evaluation_receipt = _write(root / "receipts" / "evaluations.json", {"stage": "evaluations"})
    uncertainty = _write(root / "stages" / "paired-uncertainty" / "paired-uncertainty.json", {"rows": 1})
    return downstream.ResolvedInputs(
        fourway_root=root,
        source_root=root / "source",
        source_receipt=_write(root / "source-receipt.json", {"source": True}),
        source_receipt_file_sha256="a" * 64,
        source_receipt_sha256="b" * 64,
        source_as_of="2026-08-20T00:00:00Z",
        accepted_game_ids=("game-1",),
        accepted_identity_sha256="c" * 64,
        eligible_game_ids=("game-1",),
        eligible_identity_sha256="d" * 64,
        evaluation_paths=evaluations,
        evaluation_runtime_paths=runtimes,
        evaluation_stage_receipt=evaluation_receipt,
        evaluation_receipt_paths={},
        paired_uncertainty=uncertainty,
        paired_identity_sha256="e" * 64,
        paired_rows=1,
        freeze_root=root,
    )


def test_variant_choice_is_explicit_and_closed() -> None:
    with pytest.raises(downstream.DownstreamRunError, match="selected variant"):
        downstream.RunConfig(
            fourway_root=Path("/tmp/fourway"),
            output_root=Path("/tmp/downstream"),
            selected_variant="automatic",  # type: ignore[arg-type]
            repository_root=Path("/tmp/repo"),
        )


def test_plan_contains_no_release_command_and_keeps_authority_false(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stages = downstream.build_stage_plan(config, _inputs(tmp_path))

    assert [stage.name for stage in stages] == [
        "current_rating_trust",
        "final_fit",
        "future_team_context",
        "snapshots",
        "snapshot_comparison",
    ]
    commands = [token for stage in stages for job in stage.jobs for token in job.command]
    assert "public_refresh" not in commands
    assert "--selected-variant" not in commands
    assert all(value is False for key, value in downstream.AUTHORITY.items() if key != "research_only")
    assert stages[0].jobs[0].command[:3] == (
        downstream.sys.executable,
        "-m",
        "benchmarks.build_full_current_rating_trust",
    )


def test_missing_tier_and_draft_inputs_are_blockers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stages = downstream._optional_stage_plan(config, _inputs(tmp_path))
    by_name = {stage.name: stage for stage in stages}

    assert "tier_shadow_exact_inputs_missing" in by_name["tier_shadow"].blockers
    assert "tier_diff_exact_inputs_missing" in by_name["tier_diff"].blockers
    assert by_name["draft_score"].blockers == ("draft_score_exact_inputs_missing",)
    assert all(value is False for key, value in downstream.AUTHORITY.items() if key != "research_only")


def test_draft_plan_uses_explicit_fold_root_when_exact_inputs_exist(tmp_path: Path) -> None:
    folds = tmp_path / "folds"
    for fold in (1, 2, 3):
        _write(folds / f"fold-{fold}-spec.json", {"fold": fold})
    trust = _write(tmp_path / "draft-trust.json", {"trust": True})
    public_pack = tmp_path / "public-pack"
    _write(public_pack / "manifest.json", {"manifest": True})
    config = replace(
        _config(tmp_path),
        draft_trust_root=trust,
        draft_trust_root_sha256="a" * 64,
        draft_folds_root=folds,
        draft_public_pack_root=public_pack,
        draft_manifest_sha256="b" * 64,
    )
    draft = next(stage for stage in downstream._optional_stage_plan(config, _inputs(tmp_path)) if stage.name == "draft_score")
    assert draft.blockers == ()
    command = list(draft.jobs[0].command)
    assert command[command.index("--folds-root") + 1] == str(folds)
    assert str(tmp_path / "fourway" / "stages" / "evaluations") in command


def test_selection_sidecars_bind_exact_artifact_bytes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    inputs = _inputs(tmp_path)
    result = downstream._write_selection(config, inputs, {})

    selected = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert selected["selection_method"] == "explicit_caller_choice"
    assert selected["auto_promotion"] is False
    assert downstream._validate_selection_payload(config, inputs, Path(result["path"]))["selected_variant"] == config.selected_variant
    for variant, receipt_path in result["receipt_paths"].items():
        receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
        assert receipt["variant"] == variant
        assert receipt["artifact"]["sha256"] == hashlib.sha256(inputs.evaluation_paths[variant].read_bytes()).hexdigest()
        body = dict(receipt)
        claimed = body.pop("receipt_sha256")
        assert hashlib.sha256(_canonical(body)).hexdigest() == claimed


def test_resume_receipt_detects_output_mutation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    output_root = config.output_root / "stages" / "synthetic"
    output = output_root / "result.json"
    _write(output, {"value": 1})
    stage = downstream.Stage(
        name="synthetic",
        output_roots=(output_root,),
        expected_files=(output,),
    )
    receipt = downstream._write_stage_receipt(
        config,
        stage,
        status="completed",
        outputs=downstream._collect_outputs((output_root,)),
    )
    assert receipt["status"] == "completed"
    assert downstream._validate_stage_receipt(config, stage)["status"] == "completed"

    output.write_text("changed", encoding="utf-8")
    with pytest.raises(downstream.DownstreamRunError, match="output hash changed"):
        downstream._validate_stage_receipt(config, stage)
