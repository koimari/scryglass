from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.export import pack_spec
from lol_kills.export.public_pack import source_identity_sha256
from lol_kills.postgame_sync import RefreshValidationError, SyncConfig, sync_once, validate_live_source, validate_pack


NOW = datetime(2026, 8, 9, 18, tzinfo=timezone.utc)


def _config(root: Path) -> SyncConfig:
    return SyncConfig(
        root=root,
        public_root=root / "served/packs",
        output_root=root / "output/public_pack",
        state_path=root / "runtime/state.json",
        lock_path=root / "runtime/lock",
        health_path=root / "runtime/health.json",
    )


def _write_receipt(root: Path, game_ids: list[str]) -> None:
    path = root / "data/lol/warehouse/raw/oe_api/tierlist-live-v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"games": [{"oe_game_id": value} for value in game_ids]}), encoding="utf-8")


def _ingest(root: Path, game_ids: list[str], complete: bool = True):
    def run(*_args, **_kwargs):
        _write_receipt(root, game_ids)
        return {"player_detail_complete": complete, "source_latest": "2026-08-09T17:00:00Z"}

    return run


def _write_live(root: Path, game_id: str, missing_player: bool = False) -> None:
    live = root / "data/lol/warehouse/parquet/oe_live"
    live.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"game_uid": game_id}]).to_parquet(live / "maps.parquet", index=False)
    pd.DataFrame(
        [
            {"game_uid": game_id, "side": "Blue", "result": 1, "teamname": "Blue Team"},
            {"game_uid": game_id, "side": "Red", "result": 0, "teamname": "Red Team"},
        ]
    ).to_parquet(live / "oe_team_games.parquet", index=False)
    players = pd.DataFrame(
        [
            {"game_uid": game_id, "side": side, "position": role, "playername": f"{side}-{role}"}
            for side in ("Blue", "Red")
            for role in ("top", "jng", "mid", "bot", "sup")
        ]
    )
    if missing_player:
        players = players.iloc[:-1]
    players.to_parquet(live / "oe_player_games.parquet", index=False)


def _manifest(pack_dir: Path, game_ids: list[str]) -> dict:
    files = []
    for relative in pack_spec.PUBLIC_RATING_REQUIRED_FILES:
        path = pack_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")
        files.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        )
    return {
        "pack_id": pack_dir.name,
        "ratings": {"source_game_count": len(game_ids), "source_identity_sha256": source_identity_sha256(game_ids)},
        "files": files,
        "total_files": len(files),
        "total_bytes": sum(item["bytes"] for item in files),
    }


def test_no_new_game_skips_all_rebuild_work(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"]}), encoding="utf-8")

    def unexpected(*_args, **_kwargs):
        raise AssertionError("refresh work must stay idle")

    result = sync_once(config, now=NOW, ingest_fn=_ingest(tmp_path, ["game-1"]), build_live_fn=unexpected)
    assert result["status"] == "no_change"


def test_incomplete_details_keep_the_previous_pack(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"], "pack_id": "old"}), encoding="utf-8")
    result = sync_once(config, now=NOW, ingest_fn=_ingest(tmp_path, ["game-1", "game-2"], False))
    assert result["status"] == "waiting_for_details"
    assert json.loads(config.state_path.read_text())["published_game_ids"] == ["game-1"]


def test_complete_new_game_publishes_manifest_last(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"]}), encoding="utf-8")

    def build(root: Path):
        _write_live(root, "game-2")
        return {}

    def export(*, out_root: Path, pack_id: str, **_kwargs):
        pack_dir = out_root / pack_id
        return _manifest(pack_dir, ["game-2"])

    result = sync_once(
        config,
        now=NOW,
        ingest_fn=_ingest(tmp_path, ["game-1", "game-2"]),
        build_live_fn=build,
        export_pack_fn=export,
    )
    assert result["status"] == "published"
    pointer = json.loads((config.public_root / "manifest.json").read_text())
    assert pointer["pack_id"] == result["pack_id"]
    assert not (config.public_root / "latest.json").exists()


def test_live_validation_rejects_incomplete_players(tmp_path: Path) -> None:
    _write_live(tmp_path, "game-2", missing_player=True)
    with pytest.raises(RefreshValidationError, match="malformed rows"):
        validate_live_source(tmp_path, ["game-2"])


def test_live_validation_uses_fallback_game_identity_columns(tmp_path: Path) -> None:
    _write_live(tmp_path, "game-2")
    live = tmp_path / "data/lol/warehouse/parquet/oe_live"
    for filename in ("maps.parquet", "oe_team_games.parquet", "oe_player_games.parquet"):
        path = live / filename
        frame = pd.read_parquet(path)
        frame["gameid"] = "oe-api:game-2"
        frame["game_uid"] = ""
        frame.to_parquet(path, index=False)

    result = validate_live_source(tmp_path, ["game-2"])

    assert result["game_ids"] == ["game-2"]


def test_pack_validation_rejects_a_changed_file(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    manifest = _manifest(pack_dir, ["game-2"])
    (pack_dir / pack_spec.PUBLIC_RATING_REQUIRED_FILES[0]).write_text("changed", encoding="utf-8")
    source = {"game_ids": ["game-2"], "game_count": 1, "identity_sha256": source_identity_sha256(["game-2"])}
    with pytest.raises(RefreshValidationError, match="wrong size"):
        validate_pack(pack_dir, manifest, source)


def test_pack_validation_rejects_excluded_team_affiliation(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    manifest = _manifest(pack_dir, ["game-2"])
    relative = "features/player_records.json"
    path = pack_dir / relative
    path.write_text(json.dumps({"Player": {"current_team": "Los Ratones"}}), encoding="utf-8")
    item = next(value for value in manifest["files"] if value["path"] == relative)
    manifest["total_bytes"] += path.stat().st_size - item["bytes"]
    item["bytes"] = path.stat().st_size
    item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {"game_ids": ["game-2"], "game_count": 1, "identity_sha256": source_identity_sha256(["game-2"])}

    with pytest.raises(RefreshValidationError, match="excluded team affiliation"):
        validate_pack(pack_dir, manifest, source)


def test_pack_validation_rejects_transport_label_as_league(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    manifest = _manifest(pack_dir, ["game-2"])
    relative = "features/team_records.json"
    path = pack_dir / relative
    path.write_text(
        json.dumps({"Gen.G": {"current_league": "ORACLE_ELIXIR_API", "current_tier": "tier3"}}),
        encoding="utf-8",
    )
    item = next(value for value in manifest["files"] if value["path"] == relative)
    manifest["total_bytes"] += path.stat().st_size - item["bytes"]
    item["bytes"] = path.stat().st_size
    item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = {"game_ids": ["game-2"], "game_count": 1, "identity_sha256": source_identity_sha256(["game-2"])}

    with pytest.raises(RefreshValidationError, match="transport label"):
        validate_pack(pack_dir, manifest, source)
