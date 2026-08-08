from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_source_snapshot as snapshot
from lol_kills.v2.ratings.player.multileague_v3_source_registry_v2 import (
    validate_registered_source_snapshot_v2,
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def test_maps_and_players_are_frozen_together_without_authority(tmp_path: Path) -> None:
    maps_locator = Path("warehouse/maps.parquet")
    players_locator = Path("warehouse/players.parquet")
    refresh_locator = Path("warehouse/refresh_meta.json")
    (tmp_path / "warehouse").mkdir()
    maps = b"maps source"
    players = b"players source"
    (tmp_path / maps_locator).write_bytes(maps)
    (tmp_path / players_locator).write_bytes(players)
    refresh = {
        "schema_version": "scryglass:warehouse-refresh-manifest:v2",
        "refreshed_at": "2026-08-01T23:17:35+00:00",
        "outputs": {
            "maps": {"raw_sha256": _sha(maps), "rows": 12},
            "rating_players": {"raw_sha256": _sha(players), "rows": 120},
        },
        "authority": {
            "descriptive_warehouse_provenance": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
    }
    refresh["manifest_canonical_sha256"] = snapshot._canonical_sha256(refresh)
    (tmp_path / refresh_locator).write_text(json.dumps(refresh))

    result = snapshot.build_source_snapshot(
        root=tmp_path,
        maps_locator=maps_locator,
        players_locator=players_locator,
        refresh_manifest_locator=refresh_locator,
        snapshot_root=Path("snapshots"),
    )
    manifest = snapshot.validate_source_snapshot(
        result["manifest_path"], root=tmp_path
    )
    assert manifest["information_boundary"][
        "all_outcomes_present_in_snapshot_are_adaptive_development"
    ] is True
    assert manifest["information_boundary"]["future_sealed_targets_present"] is False
    assert manifest["authority"]["rating_authority"] is False
    assert manifest["authority"]["betting_authority"] is False

    (tmp_path / maps_locator).write_bytes(b"later mutable maps")
    frozen_maps = tmp_path / manifest["files"]["maps"]["locator"]
    assert frozen_maps.read_bytes() == maps

    tampered = json.loads(result["manifest_path"].read_text())
    tampered["authority"]["rating_authority"] = True
    result["manifest_path"].write_text(json.dumps(tampered))
    with pytest.raises(snapshot.MultiLeagueSourceSnapshotError):
        snapshot.validate_source_snapshot(result["manifest_path"], root=tmp_path)


def test_current_ratings_source_snapshot_is_code_pinned_and_non_authorizing() -> None:
    manifest = snapshot.validate_current_source_snapshot(root=Path(".").resolve())
    assert manifest["package_id"] == snapshot.CURRENT_PACKAGE_ID
    assert manifest["files"]["maps"]["raw_sha256"] == (
        "04d4d7016bc1639fecddd613c1af6de94c6222a9b77cc2daaebbc51f8223402f"
    )
    assert manifest["files"]["players"]["raw_sha256"] == (
        "77d8df205fd88e78d23061fc7b9c6171362673f517311fe371ee1c27b7de5701"
    )
    assert manifest["authority"]["rating_authority"] is False


def test_superseding_snapshot_binds_boolean_normalized_maps_and_players() -> None:
    manifest = validate_registered_source_snapshot_v2(root=Path(".").resolve())
    assert manifest["files"]["maps"]["raw_sha256"] == (
        "04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08"
    )
    assert manifest["files"]["players"]["raw_sha256"] == (
        "12f1cca978d683a0df8ceec0772999aeb03c723b4465f98674247f327dea71fa"
    )
    assert manifest["authority"]["rating_authority"] is False
