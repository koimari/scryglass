from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.export import pack_spec
from lol_kills.export.public_pack import source_identity_sha256
from lol_kills.postgame_sync import (
    REVIEWED_QUARANTINED_GAME_IDS,
    RefreshValidationError,
    SyncConfig,
    publish_pack,
    sync_once,
    validate_live_source,
    validate_pack,
    validate_source_continuity,
)


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
    path = root / "data/lol/warehouse/parquet/oe_team_games.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"gameid": value, "game_uid": value}
            for value in game_ids
        ]
    ).to_parquet(path, index=False)


def _ingest(root: Path, game_ids: list[str], complete: bool = True):
    def run(*_args, **_kwargs):
        _write_receipt(root, game_ids)
        return {"player_statistics_complete": complete, "source_latest": "2026-08-09T17:00:00Z"}

    return run


def _write_live(
    root: Path,
    game_id: str | list[str],
    missing_player: bool = False,
    missing_statistics: bool = False,
    partial_source: bool = False,
) -> None:
    game_ids = [game_id] if isinstance(game_id, str) else game_id
    live = root / "data/lol/warehouse/parquet/oe_live"
    live.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"game_uid": value} for value in game_ids]).to_parquet(
        live / "maps.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "game_uid": value,
                "side": side,
                "result": 1 if side == "Blue" else 0,
                "teamname": f"{side} Team",
            }
            for value in game_ids
            for side in ("Blue", "Red")
        ]
    ).to_parquet(live / "oe_team_games.parquet", index=False)
    players = pd.DataFrame(
        [
            {
                "game_uid": value,
                "side": side,
                "position": role,
                "playername": f"{side}-{role}",
                "kills": role_index + 1,
                "deaths": 2,
                "assists": 8,
                "teamkills": 15,
                "gamelength": 1800,
                "dpm": 400 + role_index * 50,
                "damageshare": (400 + role_index * 50) / 2500,
                "totalgold": 9000 + role_index * 500,
                "cspm": 6 + role_index * 0.25,
                "wpm": 0.4 + role_index * 0.05,
                "wcpm": 0.2 + role_index * 0.02,
                "golddiffat10": (1 if side == "Blue" else -1) * role_index * 40,
                "datacompleteness": "partial" if partial_source else "complete",
            }
            for value in game_ids
            for side in ("Blue", "Red")
            for role_index, role in enumerate(("top", "jng", "mid", "bot", "sup"))
        ]
    )
    if missing_player:
        players = players.iloc[:-1]
    if missing_statistics:
        players.loc[players.index[0], "kills"] = pd.NA
    players.to_parquet(live / "oe_player_games.parquet", index=False)


def _manifest(pack_dir: Path, game_ids: list[str]) -> dict:
    files = []
    for relative in pack_spec.PUBLIC_RATING_REQUIRED_FILES:
        path = pack_dir / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if relative == "features/match_index.json":
            payload = {"schema_version": "scryglass:match-index:v1", "games": []}
        elif relative.startswith("features/match_records_"):
            payload = {"schema_version": "scryglass:match-records:v1", "games": {}}
        else:
            payload = []
        path.write_text(json.dumps(payload), encoding="utf-8")
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


def test_no_new_game_still_rejects_a_disappearing_published_game(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-2"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"]}), encoding="utf-8")

    with pytest.raises(RefreshValidationError, match="published completed maps disappeared"):
        sync_once(config, now=NOW, ingest_fn=_ingest(tmp_path, ["game-2"]))


def test_incomplete_details_keep_the_previous_pack(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"], "pack_id": "old"}), encoding="utf-8")
    def build(root: Path):
        _write_live(root, ["game-1", "game-2"])
        live = root / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
        players = pd.read_parquet(live)
        players.loc[players["game_uid"].eq("game-2"), "kills"] = pd.NA
        players.to_parquet(live, index=False)
        return {}

    result = sync_once(
        config,
        now=NOW,
        ingest_fn=_ingest(tmp_path, ["game-1", "game-2"], False),
        build_live_fn=build,
    )
    assert result["status"] == "waiting_for_details"
    assert result["pending_game_ids"] == ["game-2"]
    assert json.loads(config.state_path.read_text())["published_game_ids"] == ["game-1"]


def test_complete_new_game_publishes_manifest_last(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"]}), encoding="utf-8")

    def build(root: Path):
        _write_live(root, ["game-1", "game-2"])
        return {}

    def export(*, out_root: Path, pack_id: str, **_kwargs):
        pack_dir = out_root / pack_id
        return _manifest(pack_dir, ["game-1", "game-2"])

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


def test_pack_publication_updates_blob_pointer_without_a_site_build(tmp_path: Path, monkeypatch) -> None:
    pack_dir = tmp_path / "output/v1"
    manifest = _manifest(pack_dir, ["game-1"])
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "test-token")
    calls: list[str] = []

    def upload(_pack_dir: Path, pack_id: str, token: str) -> dict[str, str]:
        calls.append("files")
        assert token == "test-token"
        return {"features/ratings_snapshot.json": f"https://store.example/packs/{pack_id}/features/ratings_snapshot.json"}

    def pointers(token: str, pack_id: str, _manifest: dict, *, base_url: str) -> dict[str, str]:
        calls.append("pointer")
        assert token == "test-token"
        assert base_url == f"https://store.example/packs/{pack_id}"
        return {"packs/manifest.json": "https://store.example/packs/manifest.json"}

    monkeypatch.setattr("lol_kills.postgame_sync.upload_to_blob", upload)
    monkeypatch.setattr("lol_kills.postgame_sync.publish_blob_pointers", pointers)
    monkeypatch.setattr("lol_kills.postgame_sync._blob_get", lambda _url: None)
    result = publish_pack(pack_dir, manifest, tmp_path / "served/packs")

    assert calls == ["files", "pointer"]
    assert result["runtime"] == "blob"
    pointer = json.loads((tmp_path / "served/packs/manifest.json").read_text())
    assert pointer["base_url"] == "https://store.example/packs/v1"


def test_live_validation_rejects_incomplete_players(tmp_path: Path) -> None:
    _write_live(tmp_path, "game-2", missing_player=True)
    with pytest.raises(RefreshValidationError, match="complete OE source set"):
        validate_live_source(tmp_path, ["game-2"])


def test_live_validation_rejects_incomplete_player_statistics(tmp_path: Path) -> None:
    _write_live(tmp_path, "game-2", missing_statistics=True)
    with pytest.raises(RefreshValidationError, match="complete OE source set"):
        validate_live_source(tmp_path, ["game-2"])


def test_live_validation_keeps_source_labelled_partial_map_pending(
    tmp_path: Path,
) -> None:
    _write_live(tmp_path, "game-2", partial_source=True)

    source = validate_live_source(tmp_path, [])

    assert source["game_ids"] == ["game-2"]
    assert source["statistics_complete_game_ids"] == []


def test_source_continuity_rejects_a_disappearing_published_game(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps({"published_game_ids": ["game-1"], "pack_id": "old"}),
        encoding="utf-8",
    )

    def build(root: Path):
        _write_live(root, "game-2")
        return {}

    with pytest.raises(RefreshValidationError, match="published completed maps disappeared"):
        sync_once(
            config,
            now=NOW,
            ingest_fn=_ingest(tmp_path, ["game-1", "game-2"]),
            build_live_fn=build,
        )


def test_source_continuity_allows_only_reviewed_unknown_player_maps() -> None:
    quarantined = sorted(REVIEWED_QUARANTINED_GAME_IDS)
    source = {
        "game_ids": ["game-1"],
        "game_count": 1,
        "identity_sha256": source_identity_sha256(["game-1"]),
    }

    result = validate_source_continuity(["game-1", *quarantined], source, None)

    assert result["missing_game_count"] == 5
    assert result["quarantined_game_ids"] == quarantined


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


def test_incomplete_discovered_map_waits_while_complete_map_publishes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"]}), encoding="utf-8")

    def build(root: Path):
        _write_live(root, ["game-1", "game-2", "game-3"])
        live = root / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
        players = pd.read_parquet(live)
        players.loc[players["game_uid"].eq("game-3"), "kills"] = pd.NA
        players.to_parquet(live, index=False)
        return {}

    def export(*, out_root: Path, pack_id: str, **_kwargs):
        pack_dir = out_root / pack_id
        assert _kwargs["allowed_game_ids"] == ["game-1", "game-2"]
        return _manifest(pack_dir, ["game-1", "game-2"])

    result = sync_once(
        config,
        now=NOW,
        ingest_fn=_ingest(tmp_path, ["game-1", "game-2", "game-3"]),
        build_live_fn=build,
        export_pack_fn=export,
    )

    assert result["status"] == "published"
    assert result["new_game_ids"] == ["game-2"]
    assert result["pending_game_ids"] == ["game-3"]


def test_replaced_raw_identity_is_not_counted_as_a_new_or_pending_map(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps({"published_game_ids": ["bridge-id"], "pack_id": "old"}),
        encoding="utf-8",
    )

    def build(root: Path):
        _write_live(root, ["bridge-id", "new-game"])
        return {}

    def export(*, out_root: Path, pack_id: str, **kwargs):
        assert kwargs["allowed_game_ids"] == ["bridge-id", "new-game"]
        return _manifest(out_root / pack_id, kwargs["allowed_game_ids"])

    result = sync_once(
        config,
        now=NOW,
        ingest_fn=_ingest(tmp_path, ["annual-replacement-id", "bridge-id", "new-game"]),
        build_live_fn=build,
        export_pack_fn=export,
    )

    assert result["status"] == "published"
    assert result["new_game_ids"] == ["new-game"]
    assert result["pending_game_ids"] == []


def test_pack_validation_rejects_a_changed_file(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    manifest = _manifest(pack_dir, ["game-2"])
    (pack_dir / pack_spec.PUBLIC_RATING_REQUIRED_FILES[0]).write_text("changed", encoding="utf-8")
    source = {"game_ids": ["game-2"], "game_count": 1, "identity_sha256": source_identity_sha256(["game-2"])}
    with pytest.raises(RefreshValidationError, match="wrong size"):
        validate_pack(pack_dir, manifest, source)


def test_pack_validation_accepts_optional_schedule(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    manifest = _manifest(pack_dir, ["game-2"])
    relative = "features/schedule.json"
    path = pack_dir / relative
    path.write_text(
        json.dumps({"schema_version": "scryglass:public-schedule:v1", "upcoming": [{"team1": "Los Ratones"}]}),
        encoding="utf-8",
    )
    manifest["files"].append(
        {
            "path": relative,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    manifest["total_files"] += 1
    manifest["total_bytes"] += path.stat().st_size
    source = {"game_ids": ["game-2"], "game_count": 1, "identity_sha256": source_identity_sha256(["game-2"])}

    result = validate_pack(pack_dir, manifest, source)

    assert result["game_count"] == 1
    assert result["files"] == len(pack_spec.PUBLIC_RATING_REQUIRED_FILES) + 1


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
