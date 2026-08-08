from __future__ import annotations

import json
from pathlib import Path

from lol_kills.etl.oe_api_ingest import SCHEMA_VERSION, _cached_full_games


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
