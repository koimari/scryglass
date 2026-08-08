from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

import lol_kills.grid_market_cohort as cohort
from lol_kills.grid_market_cohort import (
    GridMarketCohortError,
    GridMarketQuotaStop,
    _CentralDataClient,
    _grid_labels,
    _player_crosswalk,
    _summary_labels,
    extract_checkpoints,
    index_grid_events,
    retrieve_series_files,
)


def _write_events(path: Path, events: list[dict]) -> None:
    transaction = {"sequenceNumber": 1, "events": events}
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("events.jsonl", json.dumps(transaction) + "\n")


def _event(event_id: str, kind: str, second: int, side: str = "blue") -> dict:
    return {
        "id": event_id,
        "type": kind,
        "actor": {"state": {"side": side}},
        "seriesState": {
            "games": [
                {
                    "id": "game-1",
                    "clock": {"currentSeconds": second},
                }
            ]
        },
    }


def test_checkpoint_excludes_equal_second_and_post_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "events.zip"
    _write_events(
        path,
        [
            _event("before", "player-killed-player", 599),
            _event("equal", "player-killed-player", 600, "red"),
            _event("after", "player-completed-slayDragon", 601),
        ],
    )
    result = extract_checkpoints(path, provider_game_id="game-1", checkpoints=(10,))
    assert result["status"] == "eligible"
    assert result["checkpoints"][0]["values"]["current_kills"] == 1
    assert result["checkpoints"][0]["values"]["first_blood"] == 100
    assert result["checkpoints"][0]["values"]["total_dragons"] == 0
    assert result["final_event_values"]["current_kills"] == 2


def test_duplicate_event_is_deduplicated(tmp_path: Path) -> None:
    path = tmp_path / "events.zip"
    event = _event("same", "player-completed-slayDragon", 500)
    _write_events(path, [event, event])
    result = extract_checkpoints(path, provider_game_id="game-1", checkpoints=(10,))
    assert result["status"] == "eligible"
    assert result["checkpoints"][0]["values"]["total_dragons"] == 1
    assert result["event_receipt"]["duplicate_event_count"] == 1


def test_conflicting_event_revision_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "events.zip"
    _write_events(
        path,
        [
            _event("same", "player-completed-slayDragon", 500),
            _event("same", "player-completed-slayDragon", 501),
        ],
    )
    result = extract_checkpoints(path, provider_game_id="game-1", checkpoints=(10,))
    assert result["status"] == "unavailable"
    assert "events.conflicting_or_missing_event_identity" in result["blockers"]


def test_missing_clock_or_team_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "events.zip"
    event = _event("bad", "player-killed-player", 500)
    event["actor"]["state"].pop("side")
    event["seriesState"]["games"][0]["clock"].pop("currentSeconds")
    _write_events(path, [event])
    result = extract_checkpoints(path, provider_game_id="game-1", checkpoints=(10,))
    assert result["status"] == "unavailable"
    assert "checkpoint.event_clock_missing" in result["blockers"]
    assert "identity.event_actor_team_missing" in result["blockers"]


def test_events_from_other_games_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "events.zip"
    other = _event("other", "player-killed-player", 100)
    other["seriesState"]["games"][0]["id"] = "game-2"
    _write_events(
        path,
        [other, _event("target", "player-killed-player", 200)],
    )
    result = extract_checkpoints(path, provider_game_id="game-1", checkpoints=(10,))
    assert result["status"] == "eligible"
    assert result["checkpoints"][0]["values"]["current_kills"] == 1


def test_older_elemental_drakes_and_provider_team_mapping(tmp_path: Path) -> None:
    path = tmp_path / "events.zip"
    event = _event("dragon", "player-completed-slayCloudDrake", 500)
    event["actor"]["state"] = {"teamId": "provider-red"}
    _write_events(path, [event])
    result = extract_checkpoints(
        path,
        provider_game_id="game-1",
        provider_team_to_riot={"provider-red": 200},
        checkpoints=(10,),
    )
    assert result["status"] == "eligible"
    assert result["checkpoints"][0]["values"]["first_dragon"] == 200
    assert result["checkpoints"][0]["values"]["total_dragons"] == 1


def test_indexed_and_direct_checkpoint_replay_are_identical(tmp_path: Path) -> None:
    path = tmp_path / "events.zip"
    _write_events(
        path,
        [
            _event("kill", "player-killed-player", 500),
            _event("dragon", "player-completed-slayDragon", 700),
        ],
    )
    direct = extract_checkpoints(path, provider_game_id="game-1")
    indexed = extract_checkpoints(
        path,
        provider_game_id="game-1",
        event_index=index_grid_events(path),
    )
    assert indexed == direct


def test_player_query_rejects_nonnumeric_id_before_call() -> None:
    client = _CentralDataClient("not-used", minimum_interval_seconds=0)
    called = False

    def query(_query, _variables):
        nonlocal called
        called = True
        return {}

    client.query = query  # type: ignore[method-assign]
    try:
        client.players(['1") { mutation'])
    except GridMarketCohortError:
        pass
    else:
        raise AssertionError("nonnumeric provider ID was accepted")
    assert called is False


def test_player_crosswalk_rejects_team_side_conflict() -> None:
    class Client:
        def players(self, ids):
            return {
                provider_id: {
                    "id": provider_id,
                    "externalLinks": [
                        {
                            "dataProvider": {"name": "LOL_LIVE"},
                            "externalEntity": {
                                "id": str(500 + int(provider_id))
                            },
                        }
                    ],
                }
                for provider_id in ids
            }

    provider_ids = [str(value) for value in range(10, 20)]
    crosswalk, blockers = _player_crosswalk(
        Client(),  # type: ignore[arg-type]
        provider_player_ids=provider_ids,
        provider_player_to_riot_team={
            provider_id: 100 if index < 5 else 200
            for index, provider_id in enumerate(provider_ids)
        },
        summary={
            "participants": [
                {
                    "summonerId": 500 + int(provider_id),
                    "puuid": f"puuid-{provider_id}",
                    "teamId": (
                        200
                        if provider_id == "10"
                        else 100
                        if index < 5
                        else 200
                    ),
                    "participantId": index + 1,
                }
                for index, provider_id in enumerate(provider_ids)
            ]
        },
    )
    assert len(crosswalk) == 9
    assert "identity.player_team_side_conflict" in blockers


def test_file_preflight_downloads_nothing_without_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cohort,
        "_file_list",
        lambda _key, _series_id: [
            {"id": "events-grid", "status": "ready"},
            {"id": "state-grid", "status": "ready"},
        ],
    )
    downloaded = False

    def download_file(**_kwargs):
        nonlocal downloaded
        downloaded = True
        return {}

    monkeypatch.setattr(cohort, "_download_file", download_file)
    with pytest.raises(GridMarketCohortError):
        retrieve_series_files(key="unused", series_id="1", root=tmp_path)
    assert downloaded is False


def test_file_preflight_stops_before_crossing_whole_series_quota(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        cohort,
        "_file_list",
        lambda _key, _series_id: [
            {"id": "events-grid", "status": "ready"},
            {"id": "state-grid", "status": "ready"},
            {"id": "state-summary-riot-game-1", "status": "ready"},
            {"id": "state-summary-riot-game-2", "status": "ready"},
        ],
    )
    with pytest.raises(GridMarketQuotaStop):
        retrieve_series_files(
            key="unused",
            series_id="1",
            root=tmp_path,
            maximum_summaries=1,
        )


def test_summary_requires_complete_game_and_two_exact_teams() -> None:
    labels, blockers = _summary_labels(
        {
            "endOfGameResult": "GameComplete",
            "gameEndTimestamp": 123,
            "teams": [
                {
                    "teamId": 100,
                    "objectives": {
                        name: {"kills": 0, "first": False}
                        for name in ("tower", "inhibitor", "dragon", "baron", "champion")
                    },
                }
            ],
        }
    )
    assert labels == {}
    assert "identity.riot_teams_not_exactly_100_200" in blockers


def test_grid_final_requires_finished_exact_sides() -> None:
    labels, blockers = _grid_labels(
        {"finished": False, "teams": [{"side": "blue"}, {"side": "red"}]}
    )
    assert labels == {}
    assert blockers == ["outcome.grid_final_state_incomplete"]
