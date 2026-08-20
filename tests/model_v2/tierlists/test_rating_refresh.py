"""Runtime-root isolation tests for the rating refresh worker."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lol_kills.v2.tierlists import rating_refresh
from lol_kills.export.public_pack import source_identity_sha256


def test_refresh_writes_rating_artifacts_under_runtime_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    runtime_root = tmp_path / "runtime"
    worker_cwd = tmp_path / "worker"
    worker_cwd.mkdir()
    monkeypatch.chdir(worker_cwd)

    source_root = runtime_root / "data/lol/warehouse/parquet/oe_live"
    source_root.mkdir(parents=True)
    maps_path = source_root / "maps.parquet"
    team_path = source_root / "team_games.parquet"
    player_path = source_root / "player_games.parquet"
    game = {
        "game_uid": "g1",
        "date": "2026-08-01T00:00:00Z",
        "blue_team": "Blue",
        "red_team": "Red",
        "y_blue_win": 1,
    }
    pd.DataFrame([game]).to_parquet(maps_path, index=False)
    pd.DataFrame([game | {"side": "Blue"}, game | {"side": "Red"}]).to_parquet(
        team_path,
        index=False,
    )
    pd.DataFrame(
        [game | {"side": side, "playername": f"p{index}"} for side in ("Blue", "Red") for index in range(5)]
    ).to_parquet(player_path, index=False)

    monkeypatch.setattr(rating_refresh, "LIVE_MAP_OUTPUT", Path("data/lol/warehouse/parquet/oe_live/maps.parquet"))
    monkeypatch.setattr(rating_refresh, "LIVE_TEAM_OUTPUT", Path("data/lol/warehouse/parquet/oe_live/team_games.parquet"))
    monkeypatch.setattr(rating_refresh, "LIVE_PLAYER_OUTPUT", Path("data/lol/warehouse/parquet/oe_live/player_games.parquet"))

    def write_frame(path: Path, rows: int = 1) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"row": list(range(rows))}).to_parquet(path, index=False)

    def fake_dual(_maps, *, cfg, lineup_by_game, output_dir):
        write_frame(output_dir / "ratings.parquet")
        write_frame(output_dir / "ratings_snapshot.parquet")
        pd.DataFrame(
            [{"team": "Blue", "momentum_residual": 0.0}]
        ).to_parquet(output_dir / "ratings_dual_snapshot.parquet", index=False)
        (output_dir / "ratings_meta.json").write_text("{}", encoding="utf-8")

    def fake_player(_maps, _players, *, cfg, output_dir):
        write_frame(output_dir / "player_ratings.parquet")
        write_frame(output_dir / "player_ratings_snapshot.parquet")
        (output_dir / "player_ratings_meta.json").write_text("{}", encoding="utf-8")

    def fake_hierarchical(_maps, *, write, output_dir):
        write_frame(output_dir / "ratings_snapshot.parquet")
        return pd.DataFrame({"team": ["Blue"], "mu_total": [1500.0]}), {"model": "test"}

    monkeypatch.setattr(rating_refresh, "build_maps_frame_from_players", lambda players: maps_frame)
    monkeypatch.setattr(rating_refresh, "lineup_hashes_from_players", lambda players: {})
    monkeypatch.setattr(rating_refresh, "build_dual_ratings", fake_dual)
    monkeypatch.setattr(rating_refresh, "build_player_ratings", fake_player)
    monkeypatch.setattr(rating_refresh, "fit_hierarchical_bt", fake_hierarchical)
    monkeypatch.setattr(rating_refresh, "build_team_weekly_ranks", lambda *args, **kwargs: {"by_team": {}})
    monkeypatch.setattr(rating_refresh, "build_player_weekly_ranks", lambda *args, **kwargs: {"by_player": {}})

    maps_frame = pd.DataFrame([game])
    payload = rating_refresh.refresh_ratings(
        runtime_root,
        as_of=pd.Timestamp("2026-08-01T00:00:00Z"),
        allowed_game_ids={"g1"},
    )

    expected = runtime_root / "data/lol/v2/tierlists/rating-refresh/rating-refresh-v1.json"
    assert expected.is_file()
    assert all(
        str(item["locator"]).startswith("data/lol/")
        for item in payload["artifacts"].values()
    )
    assert not (worker_cwd / "data").exists()
    assert payload["source"]["source_game_count"] == 1
    assert payload["source"]["source_identity_sha256"] == source_identity_sha256(["g1"])
    written_manifest = json.loads(expected.read_text(encoding="utf-8"))
    ratings_meta = json.loads(
        (runtime_root / rating_refresh.FEATURES_RELATIVE / "ratings_meta.json").read_text(
            encoding="utf-8"
        )
    )
    for metadata in (payload, written_manifest, ratings_meta):
        assert metadata["momentum"]["selected"]["window_games"] == 0
        assert metadata["momentum"]["selected"]["scale"] == 0.0
        assert metadata["momentum"]["registered"]["active"]["window_games"] == 0
        assert metadata["momentum"]["registered"]["candidate"]["window_games"] == 7
        assert metadata["momentum"]["registered"]["candidate"]["scale"] == 80.0
        assert metadata["momentum"]["registered"]["promotion"]["status"] == "unavailable"
