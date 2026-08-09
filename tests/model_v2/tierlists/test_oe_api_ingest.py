from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from lol_kills.etl.oe_api_ingest import (
    DISCOVERY_CACHE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    OeApiIngestError,
    _cached_full_games,
    _fetch_games,
    _read_discovery_cache,
    _rows_from_games,
    _write_discovery_cache,
)


def _full_detail() -> dict:
    return {
        team_key: {
            "players": {
                role: {"name": f"{side}-{role}"}
                for role in ("top", "jng", "mid", "bot", "sup")
            }
        }
        for side, team_key in (("blue", "blueTeam"), ("red", "redTeam"))
    }


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


def test_cached_full_games_rejects_placeholder_or_duplicate_players(tmp_path: Path) -> None:
    path = tmp_path / "tierlist-live-v1.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "games": [
                    {
                        "oe_game_id": "bad",
                        "players": {
                            "blue": {role: "unknown player" for role in ("top", "jng", "mid", "bot", "sup")},
                            "red": {role: f"red-{role}" for role in ("top", "jng", "mid", "bot", "sup")},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

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


def test_discovery_cache_reuses_tournaments_and_teams_within_ttl(tmp_path: Path) -> None:
    path = tmp_path / "discovery-v1.json"
    generated_at = pd.Timestamp("2026-08-09T12:00:00Z")
    tournaments = [{"tournament_id": "LCK/2026", "league": "LCK"}]

    _write_discovery_cache(
        path,
        generated_at=generated_at,
        discovered_through=pd.Timestamp("2026-08-09T12:00:00Z"),
        tournaments=tournaments,
        team_ids=["team-2", "team-1", "team-1"],
    )
    cached = _read_discovery_cache(
        path,
        now=pd.Timestamp("2026-08-09T15:00:00Z"),
        requested_end=pd.Timestamp("2026-08-09T15:00:00Z"),
        max_age=pd.Timedelta(hours=6),
    )

    assert cached == (tournaments, ["team-1", "team-2"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == DISCOVERY_CACHE_SCHEMA_VERSION


def test_discovery_cache_expires_after_ttl(tmp_path: Path) -> None:
    path = tmp_path / "discovery-v1.json"
    _write_discovery_cache(
        path,
        generated_at=pd.Timestamp("2026-08-09T08:00:00Z"),
        discovered_through=pd.Timestamp("2026-08-09T08:00:00Z"),
        tournaments=[{"tournament_id": "LCK/2026"}],
        team_ids=["team-1"],
    )

    cached = _read_discovery_cache(
        path,
        now=pd.Timestamp("2026-08-09T15:00:01Z"),
        requested_end=pd.Timestamp("2026-08-09T15:00:01Z"),
        max_age=pd.Timedelta(hours=6),
    )

    assert cached is None


def test_rows_from_games_writes_canonical_game_ids_and_source_league() -> None:
    game = {
        "oeGameId": "oe:game:1",
        "gameCreation": "2026-08-08T00:00:00Z",
        "side": "blue",
        "ownId": "Blue Team",
        "opponentTeam": "Red Team",
        "result": 1,
        "tournament": "LCK 2026",
        "patch": "16.15",
        **{f"{side}{role}": f"{side}-{role}" for side in ("blue", "red") for role in ("top", "jng", "mid", "bot", "sup")},
    }
    tournaments = [
        {
            "tournament_id": "LCK/2026",
            "tournament_name": "LCK 2026",
            "league": "LCK",
            "competition_tier": "tier1",
            "event_kind": None,
        }
    ]
    team, player, accepted = _rows_from_games(
        [game],
        tournaments=tournaments,
        full_games={"oe:game:1": _full_detail()},
        start=pd.Timestamp("2026-08-01T00:00:00Z"),
        end=pd.Timestamp("2026-08-09T00:00:00Z"),
    )

    assert set(team["gameid"]) == {"oe:game:1"}
    assert set(team["game_uid"]) == {"oe:game:1"}
    assert set(player["gameid"]) == {"oe:game:1"}
    assert set(player["league"]) == {"LCK"}
    assert accepted[0]["game_id"] == "oe:game:1"


def test_rows_from_games_rejects_a_game_without_named_player_details() -> None:
    game = {
        "oeGameId": "oe:game:1",
        "gameCreation": "2026-08-08T00:00:00Z",
        "side": "blue",
        "ownId": "Blue Team",
        "opponentTeam": "Red Team",
        "result": 1,
        "tournament": "LCK 2026",
        **{f"{side}{role}": f"{side}-{role}" for side in ("blue", "red") for role in ("top", "jng", "mid", "bot", "sup")},
    }
    tournaments = [{
        "tournament_id": "LCK/2026",
        "tournament_name": "LCK 2026",
        "league": "LCK",
        "competition_tier": "tier1",
        "event_kind": None,
    }]

    with pytest.raises(OeApiIngestError, match="no complete five-role games"):
        _rows_from_games(
            [game],
            tournaments=tournaments,
            full_games={},
            start=pd.Timestamp("2026-08-01T00:00:00Z"),
            end=pd.Timestamp("2026-08-09T00:00:00Z"),
        )
