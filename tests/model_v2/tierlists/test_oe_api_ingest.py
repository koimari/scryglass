from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from lol_kills.etl.oe_api_ingest import SCHEMA_VERSION, _cached_full_games, _fetch_games


def test_cached_full_games_rehydrates_complete_player_details(tmp_path: Path) -> None:
    path = tmp_path / "tierlist-live-v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "games": [
                    {
                        "oe_game_id": "game-complete",
                        "players": {
                            side: {role: f"{side}-{role}" for role in ("top", "jng", "mid", "bot", "sup")}
                            for side in ("blue", "red")
                        },
                    },
                    {
                        "oe_game_id": "game-incomplete",
                        "players": {"blue": {}, "red": {}},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    cached = _cached_full_games(path)

    assert sorted(cached) == ["game-complete"]
    assert cached["game-complete"]["blueTeam"]["players"]["mid"]["name"] == "blue-mid"
    assert cached["game-complete"]["redTeam"]["players"]["sup"]["name"] == "red-sup"


def test_cached_full_games_ignores_wrong_receipt_schema(tmp_path: Path) -> None:
    path = tmp_path / "tierlist-live-v1.json"
    path.write_text(json.dumps({"schema_version": "wrong", "games": []}), encoding="utf-8")

    assert _cached_full_games(path) == {}


def test_fetch_games_drops_old_rows_before_deduplication() -> None:
    body = [
        {"oeGameId": "old", "gameCreation": "2026-07-01T00:00:00Z"},
        {"oeGameId": "new", "gameCreation": "2026-08-08T00:00:00Z"},
    ]
    with patch("lol_kills.etl.oe_api_ingest._request_json", return_value=body):
        games = _fetch_games(
            ["team-1"],
            api_key="test",
            end=pd.Timestamp("2026-08-09T00:00:00Z"),
            not_before=pd.Timestamp("2026-08-01T00:00:00Z"),
            max_workers=1,
        )

    assert [game["oeGameId"] for game in games] == ["new"]
