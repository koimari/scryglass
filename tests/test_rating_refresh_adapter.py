from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks import rating_refresh_adapter as adapter
from benchmarks.rating_refresh_autoresearch import CENSUS_SCHEMA, freeze_inputs, source_identity_sha256


def _write_census(path: Path, game_ids: list[str]) -> None:
    ids = sorted(set(game_ids))
    path.write_text(
        json.dumps(
            {
                "schema_version": CENSUS_SCHEMA,
                "game_count": len(ids),
                "source_identity_sha256": source_identity_sha256(ids),
                "game_ids": ids,
            }
        ),
        encoding="utf-8",
    )


def _freeze_runtime_fixture(tmp_path: Path) -> tuple[dict, Path]:
    base_source = tmp_path / "base-source"
    append_source = tmp_path / "append-source"
    for root in (base_source, append_source):
        for relative in (
            "data/lol/warehouse/parquet/oe_live/maps.parquet",
            "data/lol/warehouse/parquet/oe_live/oe_team_games.parquet",
            "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("utf-8"))
    base_census = tmp_path / "base-census.json"
    append_census = tmp_path / "append-census.json"
    _write_census(base_census, ["game-a"])
    _write_census(append_census, ["game-a", "game-b"])
    output_root = tmp_path / "benchmark"
    freeze = freeze_inputs(
        base_root=base_source,
        base_census=base_census,
        append_root=append_source,
        append_census=append_census,
        output_root=output_root,
        input_relative_paths=[
            "data/lol/warehouse/parquet/oe_live/maps.parquet",
            "data/lol/warehouse/parquet/oe_live/oe_team_games.parquet",
            "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet",
        ],
    )
    return freeze, output_root


def _fake_refresh(root: Path, **kwargs) -> dict:
    census = json.loads((root / "accepted-census.json").read_text(encoding="utf-8"))
    feature_root = root / "data/lol/features"
    feature_root.mkdir(parents=True, exist_ok=True)
    artifact_names = {
        "team_snapshot": "ratings_snapshot.parquet",
        "player_snapshot": "player_ratings_snapshot.parquet",
        "team_weekly": "team_weekly_ranks.json",
        "player_weekly": "player_weekly_ranks.json",
    }
    artifacts = {}
    for name, filename in artifact_names.items():
        path = feature_root / filename
        path.write_text(f"{name}\n", encoding="utf-8")
        artifacts[name] = {
            "locator": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": 1,
        }
    manifest_path = root / "data/lol/v2/tierlists/rating-refresh/rating-refresh-v1.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "scryglass:rating-refresh:v1",
        "source": {
            "source_game_count": census["game_count"],
            "source_identity_sha256": census["source_identity_sha256"],
            "as_of": "2026-08-01T00:00:00Z",
        },
        "team": {"snapshot_rows": 1, "weekly_rows": 1},
        "player": {"snapshot_rows": 1, "weekly_rows": 1},
        "artifacts": artifacts,
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _set_adapter_environment(
    monkeypatch,
    output_root: Path,
    freeze: dict,
    *,
    phase: str = "cold",
    variant: str = "candidate",
) -> tuple[Path, Path]:
    fixture_root = output_root / "frozen" / ("cold" if phase == "cold" else "append_only")
    run_root = output_root / "runs" / phase
    run_root.mkdir(parents=True, exist_ok=True)
    output_path = run_root / f"{variant}.output.json"
    calls_path = run_root / f"{variant}.calls.json"
    monkeypatch.setenv("SCRYGLASS_RATING_AUTORESEARCH_INPUT_ROOT", str(fixture_root))
    monkeypatch.setenv("SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST", str(fixture_root / "manifest.json"))
    monkeypatch.setenv(
        "SCRYGLASS_RATING_AUTORESEARCH_FIXTURE_MANIFEST_SHA256",
        freeze["base" if phase == "cold" else "append_only"]["manifest_sha256"],
    )
    monkeypatch.setenv("SCRYGLASS_RATING_AUTORESEARCH_OUTPUT_MANIFEST", str(output_path))
    monkeypatch.setenv("SCRYGLASS_RATING_AUTORESEARCH_CALL_COUNTS_PATH", str(calls_path))
    monkeypatch.setenv("SCRYGLASS_RATING_AUTORESEARCH_RUNTIME_ROOT", str(output_root / "runtimes" / variant))
    monkeypatch.setenv(
        "SCRYGLASS_RATING_AUTORESEARCH_RUNTIME_OWNER",
        f"{freeze['freeze_sha256']}:{variant}",
    )
    monkeypatch.setenv("SCRYGLASS_RATING_AUTORESEARCH_PHASE", phase)
    monkeypatch.setenv("SCRYGLASS_RATING_AUTORESEARCH_VARIANT", variant)
    return output_path, calls_path


def test_adapter_stages_fixture_invokes_refresh_and_emits_contract(tmp_path: Path, monkeypatch) -> None:
    freeze, output_root = _freeze_runtime_fixture(tmp_path)
    output_path, calls_path = _set_adapter_environment(monkeypatch, output_root, freeze)
    seen: dict = {}

    def fake(root: Path, **kwargs):
        seen["root"] = root
        seen["kwargs"] = kwargs
        return _fake_refresh(root, **kwargs)

    monkeypatch.setattr(adapter, "refresh_ratings", fake)
    result = adapter.run_from_environment(min_games=1, min_series=1)

    assert seen["root"] == output_root / "runtimes" / "candidate"
    assert seen["kwargs"]["allowed_game_ids"] == ["game-a"]
    assert (seen["root"] / "data/lol/warehouse/parquet/oe_live/maps.parquet").is_file()
    assert result["source"]["source_game_count"] == 1
    assert result["run"]["entrypoint"].endswith("rating_refresh.refresh_ratings")
    assert result["run"]["runtime_isolated"] is True
    assert result["run"]["runtime_persistent"] is True
    assert result["run"]["timings"]["refresh_seconds"] >= 0
    assert result["run"]["timings"]["artifact_copy_hash_seconds"] >= 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["outputs"]["rating_manifest"]["sha256"]
    assert json.loads(calls_path.read_text(encoding="utf-8"))["counts"] == {
        "load_census": 1,
        "refresh_ratings": 1,
    }


def test_variant_runtime_cache_reuses_own_cold_state_for_append(tmp_path: Path, monkeypatch) -> None:
    freeze, output_root = _freeze_runtime_fixture(tmp_path)
    seen: list[tuple[str, Path]] = []

    def cached_refresh(root: Path, **kwargs):
        variant = os.environ["SCRYGLASS_RATING_AUTORESEARCH_VARIANT"]
        phase = os.environ["SCRYGLASS_RATING_AUTORESEARCH_PHASE"]
        cache_marker = root / "cache-marker.txt"
        if phase == "cold":
            assert not cache_marker.exists(), f"{variant} cold phase saw stale cache"
            cache_marker.write_text(variant, encoding="utf-8")
        else:
            assert cache_marker.read_text(encoding="utf-8") == variant
        seen.append((phase, root))
        return _fake_refresh(root, **kwargs)

    monkeypatch.setattr(adapter, "refresh_ratings", cached_refresh)
    _set_adapter_environment(monkeypatch, output_root, freeze, phase="cold", variant="baseline")
    adapter.run_from_environment(min_games=1, min_series=1)
    _set_adapter_environment(monkeypatch, output_root, freeze, phase="cold", variant="candidate")
    adapter.run_from_environment(min_games=1, min_series=1)
    candidate_map = output_root / "runtimes" / "candidate" / "data/lol/warehouse/parquet/oe_live/maps.parquet"
    candidate_map.unlink()
    _set_adapter_environment(monkeypatch, output_root, freeze, phase="append_only", variant="candidate")
    adapter.run_from_environment(min_games=1, min_series=1)

    assert [phase for phase, _root in seen] == ["cold", "cold", "append_only"]
    assert seen[0][1] == output_root / "runtimes" / "baseline"
    assert seen[1][1] == output_root / "runtimes" / "candidate"
    assert seen[2][1] == output_root / "runtimes" / "candidate"
    assert (output_root / "runtimes" / "baseline" / "cache-marker.txt").read_text() == "baseline"
    assert (output_root / "runtimes" / "candidate" / "cache-marker.txt").read_text() == "candidate"
    assert candidate_map.is_file()
    assert (output_root / "runs" / "cold" / "candidate.output.artifacts").is_dir()
    assert (output_root / "runs" / "append_only" / "candidate.output.artifacts").is_dir()


def test_adapter_rejects_refresh_identity_mismatch(tmp_path: Path, monkeypatch) -> None:
    freeze, output_root = _freeze_runtime_fixture(tmp_path)
    _set_adapter_environment(monkeypatch, output_root, freeze)

    def wrong_refresh(root: Path, **kwargs):
        payload = _fake_refresh(root, **kwargs)
        payload["source"]["source_identity_sha256"] = "0" * 64
        return payload

    monkeypatch.setattr(adapter, "refresh_ratings", wrong_refresh)
    with pytest.raises(adapter.AdapterError, match="identity differs"):
        adapter.run_from_environment(min_games=1, min_series=1)
