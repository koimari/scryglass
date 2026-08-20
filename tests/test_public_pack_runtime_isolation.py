"""Runtime-root isolation for public-pack rating builders."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.export import public_pack


class _StopAfterWeekly(RuntimeError):
    """Stop the export after the call under test."""


def _write_rating_sources(runtime_root: Path) -> pd.DataFrame:
    warehouse = runtime_root / "data" / "lol" / "warehouse" / "parquet"
    warehouse.mkdir(parents=True)
    map_row = {
        "game_uid": "g1",
        "gameid": "g1",
        "date": "2026-01-01",
        "year": 2026,
        "league": "LCK",
        "blue_team": "A",
        "red_team": "B",
        "y_blue_win": 1,
    }
    maps = pd.DataFrame([map_row])
    maps.to_parquet(warehouse / "maps.parquet", index=False)

    roles = ("top", "jng", "mid", "bot", "sup")
    team_rows = []
    player_rows = []
    for side, team, result in (("Blue", "A", 1), ("Red", "B", 0)):
        for index, role in enumerate(roles):
            shared = {
                "game_uid": "g1",
                "gameid": "g1",
                "date": "2026-01-01",
                "year": 2026,
                "league": "LCK",
                "side": side,
                "position": role,
                "teamname": team,
                "result": result,
            }
            team_rows.append(shared)
            player_rows.append(shared | {"playername": f"{team}{index}"})
    pd.DataFrame(team_rows).to_parquet(
        warehouse / "oe_team_games.parquet", index=False
    )
    pd.DataFrame(player_rows).to_parquet(
        warehouse / "oe_player_games.parquet", index=False
    )
    return maps


def test_public_pack_weekly_player_build_uses_runtime_features_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime_root = tmp_path / "runtime"
    worker_root = tmp_path / "worker"
    worker_root.mkdir()
    monkeypatch.chdir(worker_root)
    maps = _write_rating_sources(runtime_root)
    seen: dict[str, object] = {}

    def write_frame(path: Path, frame: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)

    def fake_dual(*_args, output_dir: Path, **_kwargs) -> None:
        write_frame(
            output_dir / "ratings_dual_snapshot.parquet",
            pd.DataFrame({"team": ["A"]}),
        )
        write_frame(
            output_dir / "ratings_snapshot.parquet",
            pd.DataFrame({"team": ["A"]}),
        )
        (output_dir / "ratings_meta.json").write_text("{}", encoding="utf-8")

    def fake_player(*_args, output_dir: Path, **_kwargs) -> None:
        write_frame(
            output_dir / "player_ratings.parquet",
            pd.DataFrame({"player": ["A0"]}),
        )
        write_frame(
            output_dir / "player_ratings_snapshot.parquet",
            pd.DataFrame({"player": ["A0"]}),
        )
        (output_dir / "player_ratings_meta.json").write_text(
            json.dumps(
                {
                    "global_rating": {
                        "performance_anchor": {
                            "enabled": True,
                            "players_anchored": 1,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def fake_hierarchical(*_args, output_dir: Path, **_kwargs):
        write_frame(
            output_dir / "ratings_snapshot.parquet",
            pd.DataFrame({"team": ["A"]}),
        )
        return pd.DataFrame({"team": ["A"]}), {}

    def fake_weekly(*_args, **kwargs):
        seen.update(kwargs)
        raise _StopAfterWeekly()

    monkeypatch.setattr(
        public_pack, "build_maps_frame_from_team_games", lambda _frame: maps.copy()
    )
    monkeypatch.setattr(
        public_pack, "build_maps_frame_from_players", lambda _frame: maps.copy()
    )
    monkeypatch.setattr(public_pack, "build_team_records", lambda _frame: {})
    monkeypatch.setattr(public_pack, "build_player_records", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(public_pack, "filter_public_team_rating_maps", lambda frame: frame)
    monkeypatch.setattr(public_pack, "lineup_hashes_from_players", lambda _frame: {})
    monkeypatch.setattr(public_pack, "build_dual_ratings", fake_dual)
    monkeypatch.setattr(public_pack, "build_player_ratings", fake_player)
    monkeypatch.setattr(public_pack, "fit_hierarchical_bt", fake_hierarchical)
    monkeypatch.setattr(public_pack, "apply_team_momentum_snapshot", lambda frame, *_args: frame)
    monkeypatch.setattr(public_pack, "attach_player_evidence", lambda frame, **_kwargs: frame)
    monkeypatch.setattr(
        public_pack,
        "build_team_weekly_ranks",
        lambda *_args, **_kwargs: {"by_team": {}},
    )
    monkeypatch.setattr(
        public_pack,
        "_attach_public_team_evidence",
        lambda frame, **_kwargs: frame,
    )
    monkeypatch.setattr(public_pack, "build_player_weekly_ranks", fake_weekly)

    with pytest.raises(_StopAfterWeekly):
        public_pack.export_public_pack(
            years=(2026,),
            project_root=runtime_root,
            runtime_root=runtime_root,
            out_root=runtime_root / "output",
            pack_id="runtime-test",
        )

    assert seen["output_dir"] == runtime_root / "data" / "lol" / "features"
    assert not (worker_root / "data").exists()
