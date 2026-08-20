from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from benchmarks.rating_refresh_autoresearch import (
    CENSUS_SCHEMA,
    HarnessError,
    freeze_inputs,
    run_benchmark,
    source_identity_sha256,
)


def _write_census(path: Path, game_ids: list[str]) -> None:
    canonical = sorted(set(game_ids))
    path.write_text(
        json.dumps(
            {
                "schema_version": CENSUS_SCHEMA,
                "game_count": len(canonical),
                "source_identity_sha256": source_identity_sha256(canonical),
                "game_ids": canonical,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    base_root = tmp_path / "base-source"
    append_root = tmp_path / "append-source"
    base_root.mkdir()
    append_root.mkdir()
    (base_root / "maps.jsonl").write_text('{"game_id":"game-a"}\n', encoding="utf-8")
    (append_root / "maps.jsonl").write_text(
        '{"game_id":"game-a"}\n{"game_id":"game-b"}\n', encoding="utf-8"
    )
    base_census = tmp_path / "base-census.json"
    append_census = tmp_path / "append-census.json"
    _write_census(base_census, ["game-a"])
    _write_census(append_census, ["game-a", "game-b"])
    return base_root, base_census, append_root, append_census


def _adapter_command(*, output_hash: str = "a", count_delta: int = 0) -> list[str]:
    code = """
import hashlib, json, os
from pathlib import Path
census_path = Path(os.environ['SCRYGLASS_RATING_AUTORESEARCH_CENSUS_PATH'])
census = json.loads(census_path.read_text())
binding = {
    'phase': os.environ['SCRYGLASS_RATING_AUTORESEARCH_PHASE'],
    'source_game_count': census['game_count'] + COUNT_DELTA,
    'source_identity_sha256': census['source_identity_sha256'],
    'census_sha256': hashlib.sha256(census_path.read_bytes()).hexdigest(),
    'input_manifest_sha256': os.environ['SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST_SHA256'],
}
payload = {
    'schema_version': OUTPUT_SCHEMA,
    'source': binding,
    'outputs': {'player_ratings': {'sha256': OUTPUT_HASH * 64, 'rows': census['game_count']}},
    'semantic': {'source_game_count': census['game_count']},
}
Path(os.environ['SCRYGLASS_RATING_AUTORESEARCH_OUTPUT_MANIFEST']).write_text(json.dumps(payload))
Path(os.environ['SCRYGLASS_RATING_AUTORESEARCH_CALL_COUNTS_PATH']).write_text(json.dumps({'counts': {'fit': 1, 'baseline': 1}}))
""".replace("OUTPUT_SCHEMA", repr("scryglass:rating-autoresearch-output:v1")).replace(
        "OUTPUT_HASH", repr(output_hash)
    ).replace("COUNT_DELTA", str(count_delta))
    return [sys.executable, "-c", code]


def test_freeze_copies_input_and_binds_append_census(tmp_path: Path) -> None:
    base_root, base_census, append_root, append_census = _fixture_inputs(tmp_path)
    output_root = tmp_path / "benchmark"
    manifest = freeze_inputs(
        base_root=base_root,
        base_census=base_census,
        append_root=append_root,
        append_census=append_census,
        output_root=output_root,
        input_relative_paths=["maps.jsonl"],
    )

    frozen = output_root / "frozen" / "cold" / "inputs" / "maps.jsonl"
    assert frozen.read_text(encoding="utf-8") == '{"game_id":"game-a"}\n'
    assert manifest["append_only_checks"]["valid"] is True
    assert manifest["append_only_checks"]["new_game_count"] == 1
    assert manifest["append_only_checks"]["removed_game_count"] == 0
    assert manifest["base"]["census"]["source_identity_sha256"] == source_identity_sha256(["game-a"])
    assert len(manifest["base"]["manifest_sha256"]) == 64

    (base_root / "maps.jsonl").write_text("changed\n", encoding="utf-8")
    assert frozen.read_text(encoding="utf-8") == '{"game_id":"game-a"}\n'


def test_freeze_marks_removed_game_as_invalid_append(tmp_path: Path) -> None:
    base_root, base_census, append_root, append_census = _fixture_inputs(tmp_path)
    _write_census(append_census, ["game-b"])
    manifest = freeze_inputs(
        base_root=base_root,
        base_census=base_census,
        append_root=append_root,
        append_census=append_census,
        output_root=tmp_path / "benchmark",
        input_relative_paths=["maps.jsonl"],
    )
    checks = manifest["append_only_checks"]
    assert checks["valid"] is False
    assert checks["base_ids_subset_append_ids"] is False
    with pytest.raises(HarnessError, match="append-only fixture contract"):
        run_benchmark(
            freeze_manifest=manifest,
            output_root=tmp_path / "benchmark",
            baseline_command=_adapter_command(),
            candidate_command=_adapter_command(),
            command_cwd=Path.cwd(),
        )


def test_benchmark_reports_cold_and_append_timings_calls_and_exact_outputs(tmp_path: Path) -> None:
    base_root, base_census, append_root, append_census = _fixture_inputs(tmp_path)
    output_root = tmp_path / "benchmark"
    manifest = freeze_inputs(
        base_root=base_root,
        base_census=base_census,
        append_root=append_root,
        append_census=append_census,
        output_root=output_root,
        input_relative_paths=["maps.jsonl"],
    )
    report = run_benchmark(
        freeze_manifest=manifest,
        output_root=output_root,
        baseline_command=_adapter_command(),
        candidate_command=_adapter_command(),
        command_cwd=Path.cwd(),
        budget_seconds=60.0,
        timeout_seconds=10.0,
    )

    assert report["accepted"] is True
    assert report["invocation_budget"]["total_adapter_calls"] == 4
    for phase_name in ("cold", "append_only"):
        phase = report["phases"][phase_name]
        assert phase["baseline"]["status"] == "ok"
        assert phase["candidate"]["status"] == "ok"
        assert phase["baseline"]["wall_seconds"] >= 0
        assert phase["candidate"]["wall_seconds"] >= 0
        assert phase["baseline"]["call_counts"] == {"status": "file", "counts": {"baseline": 1, "fit": 1}}
        assert phase["candidate"]["call_counts"] == {"status": "file", "counts": {"baseline": 1, "fit": 1}}
        assert phase["comparison"]["correct"] is True
        assert phase["comparison"]["candidate_within_budget"] is True


def test_benchmark_rejects_output_difference(tmp_path: Path) -> None:
    base_root, base_census, append_root, append_census = _fixture_inputs(tmp_path)
    output_root = tmp_path / "benchmark"
    manifest = freeze_inputs(
        base_root=base_root,
        base_census=base_census,
        append_root=append_root,
        append_census=append_census,
        output_root=output_root,
        input_relative_paths=["maps.jsonl"],
    )
    report = run_benchmark(
        freeze_manifest=manifest,
        output_root=output_root,
        baseline_command=_adapter_command(output_hash="a"),
        candidate_command=_adapter_command(output_hash="b"),
        command_cwd=Path.cwd(),
        budget_seconds=60.0,
        timeout_seconds=10.0,
    )

    assert report["accepted"] is False
    assert report["phases"]["cold"]["comparison"]["correct"] is False
    assert "baseline and candidate output descriptors differ" in report["phases"]["cold"]["comparison"]["reasons"]


def test_benchmark_rejects_binding_difference(tmp_path: Path) -> None:
    base_root, base_census, append_root, append_census = _fixture_inputs(tmp_path)
    output_root = tmp_path / "benchmark"
    manifest = freeze_inputs(
        base_root=base_root,
        base_census=base_census,
        append_root=append_root,
        append_census=append_census,
        output_root=output_root,
        input_relative_paths=["maps.jsonl"],
    )
    report = run_benchmark(
        freeze_manifest=manifest,
        output_root=output_root,
        baseline_command=_adapter_command(),
        candidate_command=_adapter_command(count_delta=1),
        command_cwd=Path.cwd(),
        budget_seconds=60.0,
        timeout_seconds=10.0,
    )

    assert report["accepted"] is False
    assert report["phases"]["cold"]["candidate"]["status"] == "failed"
    assert "source binding" in report["phases"]["cold"]["candidate"]["error"]
