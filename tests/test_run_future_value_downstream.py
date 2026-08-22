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


def test_source_records_resolve_from_explicit_source_root(tmp_path: Path) -> None:
    source_root = tmp_path / "freeze" / "source"
    receipt_path = tmp_path / "run" / "future-value-source-receipt.json"
    source_root.mkdir(parents=True)
    receipt_path.parent.mkdir(parents=True)
    records: dict[str, dict[str, object]] = {}
    for label, name in {
        "maps": "maps.parquet",
        "players": "oe_player_games.parquet",
        "teams": "oe_team_games.parquet",
    }.items():
        path = source_root / name
        path.write_bytes(f"{label}-source".encode("ascii"))
        records[label] = {
            "locator": name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        # A same-named file beside the receipt must not be selected.
        (receipt_path.parent / name).write_bytes(b"receipt-parent-decoy")

    downstream._source_paths_from_receipt(
        source_root,
        receipt_path,
        {"source_files": records},
    )

    (source_root / "maps.parquet").write_bytes(b"mutated-source")
    with pytest.raises(downstream.DownstreamRunError, match="file hash changed"):
        downstream._source_paths_from_receipt(source_root, receipt_path, {"source_files": records})

    records["maps"]["locator"] = "../maps.parquet"
    with pytest.raises(downstream.DownstreamRunError, match="locator is unsafe"):
        downstream._source_paths_from_receipt(source_root, receipt_path, {"source_files": records})


def test_plan_contains_no_release_command_and_keeps_authority_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path)
    stages = downstream.build_stage_plan(config, _inputs(tmp_path))

    assert [stage.name for stage in stages] == [
        "current_rating_trust",
        "final_fit",
        "final_fit_manifest",
        "snapshots",
        "snapshot_capabilities",
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
    # The synthetic input has no nested evidence, so inspect output policies
    # through a small plan with that preflight patched open.
    config_with_nested = replace(
        config,
        nested_selection=_inputs(tmp_path).evaluation_paths["future_player_form"],
        nested_selection_sha256="a" * 64,
    )
    monkeypatch.setattr(downstream, "_nested_selection_blockers", lambda path, inputs: ())
    stages = downstream._core_stage_plan(config_with_nested, _inputs(tmp_path))
    final = stages[1]
    assert len(final.jobs) == 4
    assert final.jobs[1].output_dir_policy == "absent"
    assert all(
        f"--variant" in job.command and downstream.VARIANTS[index] in job.command
        for index, job in enumerate(final.jobs)
    )
    manifest = stages[2].jobs[0]
    assert "--final-fit-manifest-worker" in manifest.command
    assert all(variant in " ".join(manifest.command) for variant in downstream.VARIANTS)
    snapshots = stages[3]
    assert [job.name for job in snapshots.jobs] == [f"snapshot_{variant}" for variant in downstream.VARIANTS]
    capabilities = stages[4].jobs[0]
    assert "--snapshot-capability-manifest-worker" in capabilities.command
    assert all(variant in " ".join(capabilities.command) for variant in downstream.VARIANTS)


def test_plan_only_commands_are_canonical_json(tmp_path: Path) -> None:
    config = _config(tmp_path)
    stages = downstream.build_stage_plan(config, _inputs(tmp_path)) + downstream._optional_stage_plan(config, _inputs(tmp_path))

    payload = {
        "stages": [
            {
                "name": stage.name,
                "jobs": [list(job.command) for job in stage.jobs],
                "plan_sha256": downstream._plan_digest(stage),
            }
            for stage in stages
        ]
    }
    encoded = json.dumps(payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))
    assert encoded
    assert all(isinstance(token, str) for stage in stages for job in stage.jobs for token in job.command)


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
        assert receipt["status"] == "research_only"
        assert receipt["blockers"] == []
        body = dict(receipt)
        claimed = body.pop("receipt_sha256")
        assert hashlib.sha256(_canonical(body)).hexdigest() == claimed


def test_selected_variant_is_manual_annotation_and_never_blocks_any_chain(tmp_path: Path) -> None:
    config = _config(tmp_path, variant="future_player_form")
    inputs = _inputs(tmp_path)
    selection = downstream._write_selection(config, inputs, {})

    assert downstream._selected_variant_receipt_blockers(config, selection["receipt_paths"]) == []
    blocked_config = replace(config, selected_variant="current_only")
    assert downstream._selected_variant_receipt_blockers(blocked_config, selection["receipt_paths"]) == []


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


def test_stub_builder_chain_honors_output_policy_and_semantic_blockers(tmp_path: Path) -> None:
    config = _config(tmp_path)
    writer = (
        "from pathlib import Path; import sys; "
        "root=Path(sys.argv[1]); mode=sys.argv[2]; "
        "assert root.exists() == (mode == 'empty'); "
        "root.mkdir(parents=True, exist_ok=True); "
        "(root/'result.json').write_text(sys.argv[3], encoding='utf-8')"
    )
    absent_root = config.output_root / "stages" / "absent"
    absent_output = absent_root / "result.json"
    absent = downstream.Stage(
        name="absent",
        jobs=(downstream.Job(
            name="stub_absent",
            command=(downstream.sys.executable, "-c", writer, str(absent_root), "absent", '{"status":"research_only","blockers":[]}'),
            output_roots=(absent_root,),
            expected_files=(absent_output,),
            output_dir_policy="absent",
        ),),
        output_roots=(absent_root,),
        expected_files=(absent_output,),
    )
    absent_receipt = downstream._execute_stage(config, absent, resume=False)
    assert absent_receipt["status"] == "completed"
    assert downstream._validate_stage_receipt(config, absent)["status"] == "completed"

    research_root = config.output_root / "stages" / "research-only"
    research_output = research_root / "result.json"
    research = downstream.Stage(
        name="research-only",
        jobs=(downstream.Job(
            name="stub_research_only",
            command=(downstream.sys.executable, "-c", writer, str(research_root), "empty", '{"status":"research_only_blocked","blockers":[]}'),
            output_roots=(research_root,),
            expected_files=(research_output,),
        ),),
        output_roots=(research_root,),
        expected_files=(research_output,),
    )
    research_receipt = downstream._execute_stage(config, research, resume=False)
    assert research_receipt["status"] == "completed"
    assert research_receipt["blockers"] == []

    blocked_root = config.output_root / "stages" / "semantic-blocked"
    blocked_output = blocked_root / "result.json"
    blocked = downstream.Stage(
        name="semantic-blocked",
        jobs=(downstream.Job(
            name="stub_blocked",
            command=(downstream.sys.executable, "-c", writer, str(blocked_root), "empty", '{"status":"blocked","blockers":["stub_missing_input"]}'),
            output_roots=(blocked_root,),
            expected_files=(blocked_output,),
        ),),
        output_roots=(blocked_root,),
        expected_files=(blocked_output,),
    )
    blocked_receipt = downstream._execute_stage(config, blocked, resume=False)
    assert blocked_receipt["status"] == "blocked"
    assert any("semantic_status" in value for value in blocked_receipt["blockers"])
    assert any("stub_missing_input" in value for value in blocked_receipt["blockers"])


def test_tier_diff_binds_candidate_hash_after_shadow(tmp_path: Path) -> None:
    config = replace(
        _config(tmp_path),
        tier_source_root=tmp_path / "tier-source",
        tier_repository_root=tmp_path / "tier-repo",
        tier_trust_manifest=tmp_path / "tier-trust.json",
        tier_trust_manifest_sha256="a" * 64,
        tier_build_pooled_candidate=True,
    )
    config.tier_source_root.mkdir()
    config.tier_repository_root.mkdir()
    (config.tier_source_root / "source").mkdir()
    (config.tier_source_root / "source" / "oe_player_games.parquet").write_bytes(b"")
    (config.tier_source_root / "source" / "meta.json").write_bytes(b"{}")
    _write(config.tier_source_root / "future-value-source-receipt.json", {})
    candidate_root = config.output_root / "stages" / "tier-fourway" / "candidates"
    candidates = {}
    for variant in downstream.VARIANTS:
        candidate = candidate_root / f"{variant}.json"
        candidate.parent.mkdir(parents=True, exist_ok=True)
        candidate.write_text(f"candidate:{variant}", encoding="utf-8")
        candidates[variant] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    stages = {stage.name: stage for stage in downstream._optional_stage_plan(config, _inputs(tmp_path))}
    command = list(stages["tier_diff"].jobs[0].command)
    for variant in downstream.VARIANTS:
        values = [
            command[index + 1]
            for index, token in enumerate(command)
            if token == "--expected-variant-candidate-sha256"
            and command[index + 1].startswith(f"{variant}=")
        ]
        assert values == [f"{variant}={candidates[variant]}"]


def test_final_receipt_stays_blocked_when_stage_status_is_blocked(tmp_path: Path) -> None:
    config = _config(tmp_path)
    inputs = _inputs(tmp_path)
    result = downstream._write_final(
        config,
        inputs,
        [{"stage": "stub", "status": "blocked", "receipt_sha256": "f" * 64}],
        (),
    )
    assert result["status"] == "research_only_blocked"
    assert "stage_not_completed:stub" in result["blockers"]
