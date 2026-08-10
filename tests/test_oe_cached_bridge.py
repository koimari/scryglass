from __future__ import annotations

from pathlib import Path

import pandas as pd

from lol_kills.etl.oe_live_source import build_live_source
from lol_kills.postgame_sync import _source_game_ids


ROLES = ("top", "jng", "mid", "bot", "sup")


def _team_rows(game_id: str, date: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_uid": f"oe-api:{game_id}",
                "gameid": f"oe-api:{game_id}",
                "date": date,
                "league": "LCK",
                "tournament": "LCK 2026",
                "patch": 16.15,
                "side": side,
                "teamname": f"{game_id}-{side}",
                "result": int(side == "Blue"),
                "position": "team",
            }
            for side in ("Blue", "Red")
        ]
    )


def _player_rows(game_id: str, date: str, *, complete: bool = True) -> pd.DataFrame:
    rows = []
    for side in ("Blue", "Red"):
        for role_index, role in enumerate(ROLES):
            rows.append(
                {
                    "game_uid": f"oe-api:{game_id}",
                    "gameid": f"oe-api:{game_id}",
                    "date": date,
                    "league": "LCK",
                    "tournament": "LCK 2026",
                    "patch": 16.15,
                    "side": side,
                    "teamname": f"{game_id}-{side}",
                    "result": int(side == "Blue"),
                    "position": role,
                    "playername": (
                        f"{game_id}-{side}-{role}"
                        if complete
                        else "unknown player"
                    ),
                    "champion": f"Champion-{role}",
                    "kills": role_index + 1,
                    "deaths": 2,
                    "assists": 8,
                    "teamkills": 15,
                    "gamelength": 1800,
                    "dpm": 400 + role_index * 50,
                    "damageshare": 0.2,
                    "totalgold": 9000 + role_index * 500,
                    "cspm": 6 + role_index * 0.25,
                    "wpm": 0.4 + role_index * 0.05,
                    "wcpm": 0.2 + role_index * 0.02,
                    "golddiffat10": role_index * 40,
                }
            )
    return pd.DataFrame(rows)


def _write_annual(root: Path) -> None:
    parquet = root / "data/lol/warehouse/parquet"
    parquet.mkdir(parents=True)
    _team_rows("annual", "2026-08-01").to_parquet(
        parquet / "oe_team_games.parquet", index=False
    )
    _player_rows("annual", "2026-08-01").to_parquet(
        parquet / "oe_player_games.parquet", index=False
    )


def test_cached_bridge_is_optional(tmp_path: Path) -> None:
    _write_annual(tmp_path)

    meta = build_live_source(tmp_path)

    assert meta["cached_bridge_used"] is False
    assert meta["source_game_count"] == 1
    assert set(_source_game_ids(tmp_path)) == {"annual"}


def test_cached_bridge_preserves_complete_maps_without_an_api_key(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("ORACLES_ELIXIR_API_KEY", raising=False)
    monkeypatch.delenv("OE_API_KEY", raising=False)
    _write_annual(tmp_path)
    parquet = tmp_path / "data/lol/warehouse/parquet"
    _team_rows("bridge", "2026-08-08").to_parquet(
        parquet / "oe_api_team_games.parquet", index=False
    )
    _player_rows("bridge", "2026-08-08").to_parquet(
        parquet / "oe_api_player_games.parquet", index=False
    )

    assert set(_source_game_ids(tmp_path)) == {"annual", "bridge"}
    meta = build_live_source(tmp_path)
    maps = pd.read_parquet(
        tmp_path / "data/lol/warehouse/parquet/oe_live/maps.parquet"
    )

    assert meta["cached_bridge_used"] is True
    assert meta["cached_bridge_game_count"] == 1
    assert meta["source_game_count"] == 2
    assert meta["statistics_complete_source_latest"].startswith("2026-08-08")
    assert set(maps["game_uid"]) == {"annual", "bridge"}
    live_players = pd.read_parquet(
        tmp_path / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
    )
    assert str(live_players["patch"].dtype) == "string"
    assert set(live_players["patch"]) == {"16.15"}


def test_cached_bridge_excludes_incomplete_player_identity(tmp_path: Path) -> None:
    _write_annual(tmp_path)
    parquet = tmp_path / "data/lol/warehouse/parquet"
    _team_rows("incomplete", "2026-08-08").to_parquet(
        parquet / "oe_api_team_games.parquet", index=False
    )
    _player_rows("incomplete", "2026-08-08", complete=False).to_parquet(
        parquet / "oe_api_player_games.parquet", index=False
    )

    assert set(_source_game_ids(tmp_path)) == {"annual"}
    meta = build_live_source(tmp_path)

    assert meta["cached_bridge_used"] is False
    assert meta["source_game_count"] == 1
