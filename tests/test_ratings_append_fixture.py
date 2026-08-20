from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_ratings_append_fixture import build_fixture
from lol_kills.v2.tierlists.accepted_census import identity_sha256, load_census, write_census


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source(root: Path, game_ids: list[str]) -> None:
    source = root / "data/lol/warehouse/parquet/oe_live"
    source.mkdir(parents=True)
    dates = pd.date_range("2026-08-01", periods=len(game_ids), tz="UTC")
    pd.DataFrame({"game_uid": game_ids, "date": dates}).to_parquet(source / "maps.parquet")
    pd.DataFrame({"gameid": game_ids}).to_parquet(source / "oe_team_games.parquet")
    pd.DataFrame({"gameid": game_ids}).to_parquet(source / "oe_player_games.parquet")
    (source / "meta.json").write_text(json.dumps({"source_game_count": len(game_ids)}), encoding="utf-8")


def test_build_fixture_binds_chronological_suffix_and_copies_source(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_ids = [f"game-{index}" for index in range(5)]
    _write_source(source_root, source_ids)
    accepted_path = tmp_path / "accepted.json"
    write_census(accepted_path, source_ids)
    output_root = tmp_path / "fixture"

    manifest = build_fixture(source_root, accepted_path, output_root, suffix_count=2)

    assert manifest["source"]["accepted_game_count"] == 5
    assert manifest["source"]["accepted_source_identity_sha256"] == identity_sha256(source_ids)
    assert manifest["suffix"]["append_game_ids"] == ["game-3", "game-4"]
    assert manifest["phases"]["base"]["game_count"] == 3
    assert manifest["phases"]["current"]["game_count"] == 5
    assert manifest["phases"]["append"]["game_count"] == 2
    assert load_census(Path(manifest["phases"]["base"]["census"]))["game_count"] == 3
    assert load_census(Path(manifest["phases"]["current"]["census"]))["game_count"] == 5
    assert load_census(Path(manifest["phases"]["append"]["census"]))["game_count"] == 2

    relative = Path("data/lol/warehouse/parquet/oe_live/maps.parquet")
    source_hash = _sha256(source_root / relative)
    assert _sha256(output_root / "base" / relative) == source_hash
    assert _sha256(output_root / "current" / relative) == source_hash
    assert manifest["copied_files"]["base"][str(relative)]["sha256"] == source_hash
    assert manifest["copied_files"]["current"][str(relative)]["copied_sha256"] == source_hash


def test_build_fixture_rejects_nonempty_output_and_invalid_suffix(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_ids = [f"game-{index}" for index in range(3)]
    _write_source(source_root, source_ids)
    accepted_path = tmp_path / "accepted.json"
    write_census(accepted_path, source_ids)

    with pytest.raises(ValueError, match="suffix_count"):
        build_fixture(source_root, accepted_path, tmp_path / "invalid", suffix_count=3)

    output_root = tmp_path / "fixture"
    output_root.mkdir()
    (output_root / "existing").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not empty"):
        build_fixture(source_root, accepted_path, output_root, suffix_count=1)
