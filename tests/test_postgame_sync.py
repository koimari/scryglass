from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import lol_kills.postgame_sync as postgame_sync
from lol_kills.export import pack_spec
from lol_kills.export.public_pack import source_identity_sha256
from lol_kills.postgame_sync import (
    REVIEWED_QUARANTINED_GAME_IDS,
    RefreshValidationError,
    SyncConfig,
    publish_pack,
    sync_once,
    validate_live_source,
    validate_match_archive,
    validate_pack,
    validate_source_continuity,
    exclusive_lock,
    SyncAlreadyRunning,
)


NOW = datetime(2026, 8, 9, 18, tzinfo=timezone.utc)


def test_exclusive_lock_rejects_overlapping_refreshes(tmp_path: Path) -> None:
    lock = tmp_path / "runtime/refresh.lock"
    with exclusive_lock(lock):
        with pytest.raises(SyncAlreadyRunning):
            with exclusive_lock(lock):
                pass


def test_browser_refreshed_source_skips_network_download(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def ingest(**kwargs):
        calls.append(kwargs)

    monkeypatch.setenv("SCRYGLASS_OE_BROWSER_REFRESHED", "1")
    monkeypatch.setattr(postgame_sync, "ingest_oe", ingest)
    monkeypatch.setattr(postgame_sync, "_source_game_ids", lambda _root: {"game-1"})
    monkeypatch.setattr(
        postgame_sync,
        "_annual_source_latest",
        lambda _root: "2026-08-11T10:50:41Z",
    )

    receipt = postgame_sync.ingest_oe_csv(tmp_path)

    assert calls == [{"years": ["2025", "2026"], "download": False, "force_download": False}]
    # SCRYGLASS_OE_TRANSPORT is unset here, so the legacy browser flag still
    # yields the Brave label for backward compatibility.
    assert receipt["source_transport"] == "brave_origin_browser_download"
    assert receipt["source_game_count"] == 1


def test_database_refreshed_source_skips_all_annual_parsing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def unexpected(**_kwargs):
        raise AssertionError("database-refreshed source must not parse annual CSVs again")

    monkeypatch.setenv("SCRYGLASS_OE_DATABASE_REFRESHED", "1")
    monkeypatch.setattr(postgame_sync, "ingest_oe", unexpected)
    monkeypatch.setattr(postgame_sync, "_source_game_ids", lambda _root: {"game-1"})
    monkeypatch.setattr(
        postgame_sync,
        "_annual_source_latest",
        lambda _root: "2026-08-11T10:50:41Z",
    )

    receipt = postgame_sync.ingest_oe_csv(tmp_path)

    assert receipt["source_transport"] == "supabase_incremental_oe"
    assert receipt["source_game_count"] == 1


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


DEFAULT_LIVE_DATE = "2026-06-01T00:00:00Z"


def _write_live(
    root: Path,
    game_id: str | list[str],
    missing_player: bool = False,
    missing_statistics: bool = False,
    partial_source: bool = False,
    dates: str | dict[str, str] | None = None,
) -> None:
    game_ids = [game_id] if isinstance(game_id, str) else game_id
    live = root / "data/lol/warehouse/parquet/oe_live"
    live.mkdir(parents=True, exist_ok=True)

    def date_for(value: str) -> str:
        if isinstance(dates, dict):
            return dates.get(value, DEFAULT_LIVE_DATE)
        return dates or DEFAULT_LIVE_DATE

    pd.DataFrame(
        [{"game_uid": value, "date": date_for(value)} for value in game_ids]
    ).to_parquet(live / "maps.parquet", index=False)
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
        "source_game_count": len(game_ids),
        "source_identity_sha256": source_identity_sha256(game_ids),
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


def test_continuity_baseline_restricts_a_recovery_seeded_superset_to_the_window(
    tmp_path: Path,
) -> None:
    """A contaminated recovery seed is trusted only after window restriction.

    Reproduces the 2026-08-17 release blocker: worker state held the published
    maps plus out-of-window prior-year maps that never left the local cache.
    The unfiltered superset must never become the continuity baseline; only the
    window-restricted set that reproduces the published binding may.
    """

    release_ids = ["oe:game:a", "oe:game:b"]
    out_of_window_id = "oe:game:c"
    seeded_ids = [*release_ids, out_of_window_id]
    config = _config(tmp_path)
    _write_live(
        tmp_path,
        seeded_ids,
        dates={
            "oe:game:a": "2026-03-01T00:00:00Z",
            "oe:game:b": "2026-03-02T00:00:00Z",
            out_of_window_id: "2024-03-03T00:00:00Z",
        },
    )
    config.public_root.mkdir(parents=True)
    (config.public_root / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "v2026.08.10.001500",
                "ratings": {
                    "source_game_count": len(release_ids),
                    "source_identity_sha256": source_identity_sha256(release_ids),
                    "window_years": [2026],
                },
            }
        ),
        encoding="utf-8",
    )
    state = {
        "pack_id": "v2026.08.10.001500",
        "published_game_ids": seeded_ids,
        "release_index_game_ids": release_ids,
    }
    with patch.object(
        postgame_sync,
        "validate_live_source",
        return_value={
            "game_ids": seeded_ids,
            "game_count": len(seeded_ids),
            "identity_sha256": source_identity_sha256(seeded_ids),
            "legacy_identity_game_ids": [],
        },
    ):
        baseline, binding = postgame_sync._continuity_baseline(config, state)

    assert baseline == release_ids
    assert out_of_window_id not in baseline
    assert source_identity_sha256(baseline) == source_identity_sha256(release_ids)
    assert binding == {
        "pack_id": "v2026.08.10.001500",
        "game_count": len(release_ids),
        "identity_sha256": source_identity_sha256(release_ids),
    }


def test_continuity_baseline_rejects_a_superset_without_a_declared_window(
    tmp_path: Path,
) -> None:
    """An undeclared publication window fails closed instead of guessing."""

    release_ids = ["oe:game:a", "oe:game:b"]
    seeded_ids = [*release_ids, "oe:game:c"]
    config = _config(tmp_path)
    _write_live(tmp_path, seeded_ids)
    config.public_root.mkdir(parents=True)
    (config.public_root / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "v2026.08.10.001500",
                "ratings": {
                    "source_game_count": len(release_ids),
                    "source_identity_sha256": source_identity_sha256(release_ids),
                },
            }
        ),
        encoding="utf-8",
    )
    state = {
        "pack_id": "v2026.08.10.001500",
        "published_game_ids": seeded_ids,
        "release_index_game_ids": release_ids,
    }
    with patch.object(
        postgame_sync,
        "validate_live_source",
        return_value={
            "game_ids": seeded_ids,
            "game_count": len(seeded_ids),
            "identity_sha256": source_identity_sha256(seeded_ids),
            "legacy_identity_game_ids": [],
        },
    ):
        with pytest.raises(RefreshValidationError, match="publication window years"):
            postgame_sync._continuity_baseline(config, state)


def test_continuity_baseline_rejects_a_state_without_a_release_index(tmp_path: Path) -> None:
    release_ids = ["oe:game:a", "oe:game:b"]
    seeded_ids = ["oe:game:a", "oe:game:b", "oe:game:c"]
    config = _config(tmp_path)
    config.public_root.mkdir(parents=True)
    (config.public_root / "manifest.json").write_text(
        json.dumps(
            {
                "pack_id": "v2026.08.10.001500",
                "ratings": {
                    "source_game_count": len(release_ids),
                    "source_identity_sha256": source_identity_sha256(release_ids),
                },
            }
        ),
        encoding="utf-8",
    )
    state = {
        "pack_id": "v2026.08.10.001500",
        "published_game_ids": seeded_ids,
    }
    with patch.object(
        postgame_sync,
        "validate_live_source",
        return_value={
            "game_ids": seeded_ids,
            "game_count": len(seeded_ids),
            "identity_sha256": source_identity_sha256(seeded_ids),
            "legacy_identity_game_ids": [],
        },
    ), pytest.raises(RefreshValidationError, match="exact continuity cannot be proved"):
        postgame_sync._continuity_baseline(config, state)


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
    assert pointer["release"]["release_id"] == result["pack_id"]
    assert len(pointer["release"]["worker_commit"]) == 40
    assert pointer["release"]["canonical_game_identity_digest"]


def test_tier_preparation_is_bound_before_pack_export(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(json.dumps({"published_game_ids": ["game-1"]}), encoding="utf-8")

    def build(root: Path):
        _write_live(root, ["game-1", "game-2"])
        return {}

    seen: dict[str, object] = {}

    def prepare(source_latest: str | None, source: dict[str, object]):
        seen["source_latest"] = source_latest
        seen["source_identity"] = source["identity_sha256"]
        return {
            "status": "available",
            "production_status": "production_built",
            "tier_schema_version": "rankings-tierlists-v2",
            "payload_sha256": "a" * 64,
            "receipt_sha256": "b" * 64,
            "source_observed_through": source_latest,
        }

    def export(*, out_root: Path, pack_id: str, tier_publication=None, **_kwargs):
        seen["tier_publication"] = tier_publication
        return _manifest(out_root / pack_id, ["game-1", "game-2"])

    result = sync_once(
        config,
        now=NOW,
        ingest_fn=_ingest(tmp_path, ["game-1", "game-2"]),
        build_live_fn=build,
        export_pack_fn=export,
        prepare_tier_fn=prepare,
    )

    assert seen["source_latest"] == "2026-08-09T17:00:00Z"
    assert seen["source_identity"] == source_identity_sha256(["game-1", "game-2"])
    assert seen["tier_publication"]["production_status"] == "production_built"
    assert result["tier_publication"]["receipt_sha256"] == "b" * 64


def test_corrected_game_rebuilds_when_no_identity_is_new(tmp_path: Path) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps({"published_game_ids": ["game-1"]}),
        encoding="utf-8",
    )

    def ingest(root: Path, **_kwargs):
        _write_receipt(root, ["game-1"])
        return {
            "source_latest": "2026-08-09T17:00:00Z",
            "accepted_import_receipt": {
                "accepted_games": 1,
                "new_games": 0,
                "corrected_games": 1,
                "unchanged_games": 0,
                "quarantined_games": 0,
            },
        }

    def build(root: Path):
        _write_live(root, ["game-1"])
        return {}

    def export(*, out_root: Path, pack_id: str, **kwargs):
        assert kwargs["allowed_game_ids"] == ["game-1"]
        return _manifest(out_root / pack_id, ["game-1"])

    result = sync_once(
        config,
        now=NOW,
        ingest_fn=ingest,
        build_live_fn=build,
        export_pack_fn=export,
    )

    assert result["status"] == "published"
    manifest = json.loads((config.public_root / "manifest.json").read_text())
    assert manifest["release"]["corrected_games"] == 1


def test_unpublished_source_hash_retries_after_import_receipt_becomes_current(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    _write_receipt(tmp_path, ["game-1"])
    config.state_path.parent.mkdir(parents=True)
    config.state_path.write_text(
        json.dumps(
            {
                "published_game_ids": ["game-1"],
                "source_file_sha256": "a" * 64,
            }
        ),
        encoding="utf-8",
    )

    def ingest(root: Path, **_kwargs):
        _write_receipt(root, ["game-1"])
        return {
            "source_latest": "2026-08-09T17:00:00Z",
            "accepted_import_receipt": {
                "source_file_sha256": "b" * 64,
                "accepted_games": 1,
                "new_games": 0,
                "corrected_games": 0,
                "unchanged_games": 1,
                "quarantined_games": 0,
            },
        }

    def build(root: Path):
        _write_live(root, ["game-1"])
        return {}

    def export(*, out_root: Path, pack_id: str, **_kwargs):
        return _manifest(out_root / pack_id, ["game-1"])

    result = sync_once(
        config,
        now=NOW,
        ingest_fn=ingest,
        build_live_fn=build,
        export_pack_fn=export,
    )

    assert result["status"] == "published"
    state = json.loads(config.state_path.read_text(encoding="utf-8"))
    assert state["source_file_sha256"] == "b" * 64


def test_pack_publication_stays_local_when_blob_token_exists(tmp_path: Path, monkeypatch) -> None:
    pack_dir = tmp_path / "output/v1"
    manifest = _manifest(pack_dir, ["game-1"])
    monkeypatch.setenv("BLOB_READ_WRITE_TOKEN", "test-token")
    with patch("lol_kills.export.upload_pack._blob_put") as blob_put:
        result = publish_pack(pack_dir, manifest, tmp_path / "staging/packs")

    blob_put.assert_not_called()
    assert result["runtime"] == "local_staging"
    assert "blob_pointers" not in result
    pointer = json.loads((tmp_path / "staging/packs/manifest.json").read_text())
    assert pointer["base_url"] == "/packs/v1"


def test_pack_candidate_cannot_stage_inside_web_public_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    pack_dir = tmp_path / "output/v1"
    manifest = _manifest(pack_dir, ["game-1"])
    app_public = tmp_path / "apps/scryglass/public"
    monkeypatch.setattr(postgame_sync, "STATIC_APP_PUBLIC_ROOT", app_public.resolve())

    with pytest.raises(RefreshValidationError, match="web app public directory"):
        publish_pack(pack_dir, manifest, app_public / "packs")

    assert not (app_public / "packs").exists()


def test_live_validation_rejects_incomplete_players(tmp_path: Path) -> None:
    _write_live(tmp_path, "game-2", missing_player=True)
    with pytest.raises(RefreshValidationError, match="complete OE source set"):
        validate_live_source(tmp_path, ["game-2"], years=(2025, 2026))


def test_live_validation_rejects_incomplete_player_statistics(tmp_path: Path) -> None:
    _write_live(tmp_path, "game-2", missing_statistics=True)
    with pytest.raises(RefreshValidationError, match="complete OE source set"):
        validate_live_source(tmp_path, ["game-2"], years=(2025, 2026))


def test_live_validation_keeps_source_labelled_partial_map_pending(
    tmp_path: Path,
) -> None:
    _write_live(tmp_path, "game-2", partial_source=True)

    source = validate_live_source(tmp_path, [], years=(2025, 2026))

    assert source["game_ids"] == ["game-2"]
    assert source["statistics_complete_game_ids"] == []


def test_live_validation_rejects_a_map_outside_the_publication_window(
    tmp_path: Path,
) -> None:
    _write_live(
        tmp_path,
        ["game-1", "game-2"],
        dates={"game-1": "2026-06-01T00:00:00Z", "game-2": "2024-11-03T00:00:00Z"},
    )

    with pytest.raises(RefreshValidationError) as excinfo:
        validate_live_source(tmp_path, [], years=(2025, 2026))

    message = str(excinfo.value)
    assert "outside the publication window" in message
    assert "1 games" in message
    assert "2024" in message


def test_live_validation_can_exclude_out_of_window_maps_for_the_bootstrap(
    tmp_path: Path,
) -> None:
    """The bootstrap must be able to read a contaminated cache, unmocked.

    This exercises the real validator against a real contaminated cache. The
    continuity bootstrap has to obtain the in-window population from a cache it
    already knows holds out-of-window rows; if this mode were absent or ordered
    after the strict whole-cache check, the bootstrap could never recover from
    the exact scenario it exists for.
    """

    _write_live(
        tmp_path,
        ["game-1", "game-2", "game-3"],
        dates={
            "game-1": "2026-06-01T00:00:00Z",
            "game-2": "2025-04-02T00:00:00Z",
            "game-3": "2024-11-03T00:00:00Z",
        },
    )

    source = validate_live_source(
        tmp_path,
        [],
        years=(2025, 2026),
        exclude_out_of_window=True,
    )

    assert source["game_ids"] == ["game-1", "game-2"]
    assert "game-3" not in source["game_ids"]
    assert source["game_count"] == 2
    assert source["identity_sha256"] == source_identity_sha256(["game-1", "game-2"])
    # The exclusion is auditable, never silent.
    assert source["out_of_window_game_ids"] == ["game-3"]
    assert source["out_of_window_identity_sha256"] == source_identity_sha256(["game-3"])

    # The same cache still fails closed by default.
    with pytest.raises(RefreshValidationError, match="outside the publication window"):
        validate_live_source(tmp_path, [], years=(2025, 2026))


def test_live_validation_reports_no_exclusions_for_a_clean_cache(tmp_path: Path) -> None:
    """A clean cache reports an empty exclusion set in both modes."""

    _write_live(tmp_path, ["game-1", "game-2"])

    for kwargs in ({}, {"exclude_out_of_window": True}):
        source = validate_live_source(tmp_path, [], years=(2025, 2026), **kwargs)
        assert source["out_of_window_game_ids"] == []
        assert source["game_count"] == 2


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

    assert result["missing_game_count"] == len(quarantined)
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

    result = validate_live_source(tmp_path, ["game-2"], years=(2025, 2026))

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


def test_pack_validation_rejects_a_missing_top_level_source_binding(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    manifest = _manifest(pack_dir, ["game-2"])
    manifest["source_game_count"] = None
    source = {
        "game_ids": ["game-2"],
        "game_count": 1,
        "identity_sha256": source_identity_sha256(["game-2"]),
    }

    with pytest.raises(RefreshValidationError, match="manifest source game count"):
        validate_pack(pack_dir, manifest, source)


def test_match_archive_validation_collects_all_quarter_files(tmp_path: Path) -> None:
    pack_dir = tmp_path / "pack"
    features = pack_dir / "features"
    features.mkdir(parents=True)

    def game(game_id: str, date: str) -> dict:
        players = []
        for side in ("Blue", "Red"):
            for role in ("top", "jungle", "mid", "bot", "support"):
                players.append(
                    {
                        "player": f"{side}-{role}-{game_id}",
                        "side": side,
                        "role": role,
                        "champion": f"Champion-{side}-{role}",
                        "kills": 1,
                        "deaths": 1,
                        "assists": 1,
                    }
                )
        return {
            "game_id": game_id,
            "date": date,
            "blue_team": "Blue Team",
            "red_team": "Red Team",
            "blue_win": True,
            "players": players,
        }

    games = {
        "game-2025-q1": game("game-2025-q1", "2025-01-10T00:00:00Z"),
        "game-2025-q2": game("game-2025-q2", "2025-04-10T00:00:00Z"),
        "game-2026-q3": game("game-2026-q3", "2026-07-10T00:00:00Z"),
    }
    (features / "match_index.json").write_text(
        json.dumps(
            {
                "schema_version": "scryglass:match-index:v1",
                "games": [
                    {"game_id": game_id}
                    for game_id in sorted(games)
                ],
            }
        ),
        encoding="utf-8",
    )
    for year in (2025, 2026):
        for quarter in (1, 2, 3, 4):
            quarter_games = {
                game_id: value
                for game_id, value in games.items()
                if value["date"].startswith(f"{year}-")
                and ((value["date"][5:7] in {"01", "02", "03"} and quarter == 1)
                     or (value["date"][5:7] in {"04", "05", "06"} and quarter == 2)
                     or (value["date"][5:7] in {"07", "08", "09"} and quarter == 3)
                     or (value["date"][5:7] in {"10", "11", "12"} and quarter == 4))
            }
            (features / f"match_records_{year}_q{quarter}.json").write_text(
                json.dumps({"schema_version": "scryglass:match-records:v1", "games": quarter_games}),
                encoding="utf-8",
            )

    result = validate_match_archive(pack_dir)

    assert result == {"maps": 3, "2025": 2, "2026": 1}


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
