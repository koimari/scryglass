from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

import lol_kills.grid_live_foundation as foundation


PUUIDS = [f"puuid-{index}" for index in range(1, 11)]
PROVIDER_GAME_ID = "provider-game-1"
SERIES_ID = "series-1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _player(index: int) -> dict:
    return {
        "id": f"provider-player-{index}",
        "name": f"Player {index}",
        "externalLinks": [
            {
                "dataProvider": {"name": "RIOT_PUUID"},
                "externalEntity": {"id": PUUIDS[index - 1]},
            }
        ],
    }


def _provider_team(
    team_id: str,
    name: str,
    side: str,
    indices: range,
    kills: int,
) -> dict:
    return {
        "id": team_id,
        "name": name,
        "side": side,
        "kills": kills,
        "players": [_player(index) for index in indices],
    }


def _provider_state(*, blue_kills: int = 2, red_kills: int = 1) -> dict:
    teams = [
        _provider_team("provider-blue", "Blue Team", "blue", range(1, 6), blue_kills),
        _provider_team("provider-red", "Red Team", "red", range(6, 11), red_kills),
    ]
    return {
        "id": SERIES_ID,
        "teams": teams,
        "games": [
            {
                "id": PROVIDER_GAME_ID,
                "started": True,
                "startedAt": "2026-07-27T12:00:00Z",
                "titleVersion": {"name": "16.14"},
                "teams": teams,
            }
        ],
    }


def _provider_transactions(
    *,
    winner_team_id: str = "provider-blue",
    duplicate_puuid: bool = False,
) -> list[dict]:
    state = _provider_state()
    if duplicate_puuid:
        state["games"][0]["teams"][1]["players"][0]["externalLinks"][0][
            "externalEntity"
        ]["id"] = PUUIDS[0]
    return [
        {
            "id": "transaction-1",
            "seriesId": SERIES_ID,
            "sequenceNumber": 1,
            "occurredAt": "2026-07-27T12:00:00Z",
            "events": [
                {
                    "type": "series-started-game",
                    "actor": {"type": "series", "id": SERIES_ID},
                    "target": {"type": "game", "id": PROVIDER_GAME_ID},
                    "seriesState": state,
                }
            ],
        },
        {
            "id": "transaction-2",
            "seriesId": SERIES_ID,
            "sequenceNumber": 2,
            "occurredAt": "2026-07-27T12:27:00Z",
            "events": [
                {
                    "type": "team-won-game",
                    "actor": {"type": "team", "id": winner_team_id},
                    "target": {"type": "game", "id": PROVIDER_GAME_ID},
                    "seriesState": _provider_state(),
                }
            ],
        },
        {
            "id": "transaction-3",
            "seriesId": SERIES_ID,
            "sequenceNumber": 3,
            "occurredAt": "2026-07-27T12:27:01Z",
            "events": [
                {
                    "type": "series-ended-game",
                    "actor": {"type": "series", "id": SERIES_ID},
                    "target": {"type": "game", "id": PROVIDER_GAME_ID},
                    "seriesState": _provider_state(),
                }
            ],
        },
    ]


def _stats(sequence: int, game_time: int, blue_kills: int, red_kills: int) -> dict:
    return {
        "rfc461Schema": "stats_update",
        "sequenceIndex": sequence,
        "gameTime": game_time,
        "rfc460Timestamp": f"2026-07-27T12:{sequence:02d}:00Z",
        "teams": [
            {"teamID": 100, "championsKills": blue_kills, "totalGold": 10_000 + sequence},
            {"teamID": 200, "championsKills": red_kills, "totalGold": 9_000 + sequence},
        ],
    }


def _riot_rows(*, winning_team: int = 100, include_game_end: bool = True) -> list[dict]:
    participants = [
        {
            "participantID": index,
            "teamID": 100 if index <= 5 else 200,
            "puuid": PUUIDS[index - 1],
            "role": ("Top", "Jungle", "Middle", "Bottom", "Support")[(index - 1) % 5],
            "championName": f"Champion{index}",
        }
        for index in range(1, 11)
    ]
    rows = [
        {
            "rfc461Schema": "game_info",
            "sequenceIndex": 0,
            "rfc460Timestamp": "2026-07-27T11:59:59Z",
            "platformID": "LOLTEST01",
            "gameID": 12345,
            "rootGameID": 12345,
            "gameVersion": "16.14.1.2",
            "participants": participants,
        },
        _stats(1, 599_000, 0, 0),
        _stats(2, 899_000, 0, 0),
        _stats(3, 1_199_000, 1, 0),
        _stats(4, 1_499_000, 1, 1),
        {
            "rfc461Schema": "champion_kill",
            "sequenceIndex": 5,
            "gameTime": 1_510_000,
            "killerTeamID": 100,
        },
        {
            "rfc461Schema": "champion_kill",
            "sequenceIndex": 6,
            "gameTime": 1_520_000,
            "killerTeamID": 100,
        },
        {
            "rfc461Schema": "champion_kill",
            "sequenceIndex": 7,
            "gameTime": 1_530_000,
            "killerTeamID": 200,
        },
        _stats(8, 1_600_000, 2, 1),
    ]
    if include_game_end:
        rows.append(
            {
                "rfc461Schema": "game_end",
                "sequenceIndex": 9,
                "gameTime": 1_600_000,
                "winningTeam": winning_team,
                "rfc460Timestamp": "2026-07-27T12:27:00Z",
            }
        )
    return rows


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, separators=(",", ":")) for row in rows) + "\n",
        encoding="utf-8",
    )


def _fixture(
    tmp_path: Path,
    *,
    provider_rows: list[dict] | None = None,
    riot_rows: list[dict] | None = None,
) -> tuple[Path, Path, Path, dict]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    provider_path = tmp_path / "provider.jsonl.zip"
    provider_payload = "\n".join(
        json.dumps(row, separators=(",", ":"))
        for row in (provider_rows or _provider_transactions())
    )
    with zipfile.ZipFile(provider_path, "w") as archive:
        archive.writestr("provider.jsonl", provider_payload)
    riot_path = tmp_path / "riot.jsonl"
    _write_jsonl(riot_path, riot_rows or _riot_rows())
    metadata_path = tmp_path / "series.json"
    metadata_path.write_text(
        json.dumps(
            {
                "id": SERIES_ID,
                "date": "2026-07-27T12:00:00Z",
                "tournament": "Test League 2026",
                "teams": ["Blue Team", "Red Team"],
            }
        ),
        encoding="utf-8",
    )
    context_source = tmp_path / "context-source.json"
    context_source.write_text('{"source":"test"}\n', encoding="utf-8")
    record = {
        "provider_series_id": SERIES_ID,
        "provider_game_id": PROVIDER_GAME_ID,
        "tournament": "Test League 2026",
        "league": "TEST",
        "patch": "16.14",
        "complete": True,
        "winner_provider_team_id": "provider-blue",
        "teams": [
            {"provider_team_id": "provider-blue", "name": "Blue Team", "side": "blue"},
            {"provider_team_id": "provider-red", "name": "Red Team", "side": "red"},
        ],
    }
    context = {
        "schema_version": "scryglass.grid-live.context-evidence.v1",
        "source_path": str(context_source),
        "source_sha256": _sha256(context_source),
        "record": record,
        "record_sha256": foundation._hash_value(record),
    }
    return provider_path, riot_path, metadata_path, context


def _build(tmp_path: Path, **kwargs: object) -> dict:
    provider, riot, metadata, context = _fixture(tmp_path, **kwargs)
    return foundation.build_foundation_artifact(
        provider_events_path=provider,
        riot_events_path=riot,
        series_metadata_path=metadata,
        context=context,
    )


def test_valid_offline_replay_extracts_identity_outcome_and_checkpoints(
    tmp_path: Path,
) -> None:
    provider, riot, metadata, context = _fixture(tmp_path)
    artifact = foundation.build_foundation_artifact(
        provider_events_path=provider,
        riot_events_path=riot,
        series_metadata_path=metadata,
        context=context,
    )
    replayed = foundation.build_foundation_artifact(
        provider_events_path=provider,
        riot_events_path=riot,
        series_metadata_path=metadata,
        context=context,
    )

    assert replayed == artifact
    assert artifact["verified_game"]["status"] == "verified"
    assert artifact["verified_game"]["total_kills"] == 3
    assert artifact["verified_game"]["team_kills"] == ((100, 2), (200, 1))
    assert [state["state_status"] for state in artifact["checkpoint_states"]] == [
        "valid",
        "valid",
        "valid",
        "valid",
    ]
    assert all(
        state["state_age_ms"] == 1_000 for state in artifact["checkpoint_states"]
    )
    authority = artifact["cohort_manifest"]["authority_receipt"]
    assert authority["status"] == "unavailable"
    assert authority["historical_replay_evidence_status"] == "eligible"
    assert authority["historical_replay_evidence_blockers"] == ()
    assert authority["model_evaluation_status"] == "unavailable"
    assert authority["prospective_live_latency_status"] == "unavailable"
    assert authority["market_comparison_status"] == "unavailable"
    assert authority["probability_authorized"] is False
    assert "latency.prospective_receive_time_missing" not in authority[
        "historical_replay_evidence_blockers"
    ]
    assert "latency.prospective_receive_time_missing" in authority[
        "prospective_live_latency_blockers"
    ]
    assert "model.coverage_threshold_not_met" in authority[
        "model_evaluation_blockers"
    ]
    assert "market.price_or_external_benchmark_missing" in authority[
        "market_comparison_blockers"
    ]


def test_ambiguous_provider_identity_fails_closed(tmp_path: Path) -> None:
    artifact = _build(
        tmp_path,
        provider_rows=_provider_transactions(duplicate_puuid=True),
    )

    assert artifact["verified_game"]["status"] == "unavailable"
    assert "identity.provider_puuid_invalid_or_ambiguous" in artifact["verified_game"][
        "blockers"
    ]
    assert all(
        state["state_status"] == "unavailable"
        for state in artifact["checkpoint_states"]
    )


def test_missing_and_conflicting_game_end_fail_closed(tmp_path: Path) -> None:
    missing = _build(tmp_path / "missing", riot_rows=_riot_rows(include_game_end=False))
    assert "outcome.riot_game_end_missing" in missing["verified_game"]["blockers"]
    assert missing["verified_game"]["total_kills"] is None

    conflicting_rows = _riot_rows()
    conflicting_rows.append(
        {
            **conflicting_rows[-1],
            "sequenceIndex": 10,
            "winningTeam": 200,
        }
    )
    conflicting_end = _build(
        tmp_path / "conflicting-end",
        riot_rows=conflicting_rows,
    )
    assert "outcome.riot_game_end_conflicting" in conflicting_end["verified_game"][
        "blockers"
    ]

    conflicting = _build(
        tmp_path / "conflicting",
        riot_rows=_riot_rows(winning_team=200),
    )
    assert "outcome.provider_riot_winner_conflict" in conflicting["verified_game"][
        "blockers"
    ]
    assert conflicting["verified_game"]["status"] == "unavailable"


def test_duplicate_late_and_revised_events_are_explicit(tmp_path: Path) -> None:
    rows = _riot_rows()
    identical_duplicate = dict(rows[1])
    late_row = dict(rows[2])
    revised = {**rows[3], "gameTime": 1_198_000}
    ordered = [rows[0], rows[1], rows[3], late_row, identical_duplicate, revised, *rows[4:]]
    path = tmp_path / "riot.jsonl"
    _write_jsonl(path, ordered)

    replay = foundation._riot_archive(path)

    assert replay["duplicate_count"] == 1
    assert replay["late_count"] >= 2
    assert replay["conflicting_sequences"] == [3]


def test_sequence_gap_blocks_affected_checkpoint(tmp_path: Path) -> None:
    rows = [row for row in _riot_rows() if row.get("sequenceIndex") != 2]
    artifact = _build(tmp_path, riot_rows=rows)
    checkpoint_20 = next(
        state for state in artifact["checkpoint_states"] if state["minute"] == 20
    )

    assert checkpoint_20["state_status"] == "unavailable"
    assert "checkpoint.sequence_gap_before_state" in checkpoint_20[
        "historical_model_evidence_blockers"
    ]


def test_stale_checkpoint_state_is_unavailable(tmp_path: Path) -> None:
    rows = _riot_rows()
    rows[1] = _stats(1, 590_000, 0, 0)
    artifact = _build(tmp_path, riot_rows=rows)
    checkpoint_10 = artifact["checkpoint_states"][0]

    assert checkpoint_10["state_status"] == "unavailable"
    assert checkpoint_10["state_age_ms"] == 10_000
    assert "checkpoint.state_stale" in checkpoint_10[
        "historical_model_evidence_blockers"
    ]


def test_checkpoint_never_reads_post_checkpoint_state() -> None:
    verified = foundation.VerifiedGame(
        schema_version=foundation.VERIFIED_GAME_SCHEMA,
        status="verified",
        blockers=(),
        provider_series_id=SERIES_ID,
        provider_game_id=PROVIDER_GAME_ID,
        riot_platform_id="LOLTEST01",
        riot_game_id="12345",
        riot_root_game_id="12345",
        tournament="Test",
        league="TEST",
        patch="16.14",
        provider_to_riot_team_ids=(("provider-blue", 100), ("provider-red", 200)),
        teams=(),
        players=(),
        started_at=None,
        ended_at=None,
        game_end_time_ms=1_600_000,
        winner_provider_team_id="provider-blue",
        winner_riot_team_id=100,
        team_kills=((100, 2), (200, 1)),
        total_kills=3,
        evidence_capture_ids=(),
    )
    before = {
        "sequence": 1,
        "game_time_ms": 599_000,
        "teams": [
            {"team_id": 100, "kills": 1, "gold": 10_000},
            {"team_id": 200, "kills": 0, "gold": 9_000},
        ],
        "event_sha256": "before",
    }
    after = {
        "sequence": 2,
        "game_time_ms": 600_001,
        "teams": [
            {"team_id": 100, "kills": 99, "gold": 99_000},
            {"team_id": 200, "kills": 99, "gold": 99_000},
        ],
        "event_sha256": "after",
    }
    state = foundation._checkpoint_states(
        {
            "stats": [before, after],
            "sequences": [0, 1, 2],
            "nonnegative_sequence_gaps": [],
            "conflicting_sequences": [],
        },
        verified_game=verified,
        checkpoints=(10,),
        maximum_state_age_ms=5_000,
    )[0]

    assert state.state_status == "valid"
    assert state.source_event_sha256 == "before"
    assert state.current_kills == 1
    assert state.gold_difference == 1_000


def test_immutable_receipt_is_idempotent_but_never_overwritten(
    tmp_path: Path,
) -> None:
    path = tmp_path / "receipt.json"
    first = foundation.write_immutable_receipt(path, {"value": 1})
    second = foundation.write_immutable_receipt(path, {"value": 1})

    assert first == second
    with pytest.raises(foundation.ImmutableReceiptConflict):
        foundation.write_immutable_receipt(path, {"value": 2})


def test_already_local_completed_overlap_is_verified_when_available() -> None:
    warehouse = Path("data/lol/warehouse")
    provider = warehouse / "raw_grid/events_2974295_grid.jsonl.zip"
    riot = warehouse / "raw_grid/events_2974295_1_riot.jsonl"
    metadata = warehouse / "raw_grid/series_2974295.json"
    context_path = warehouse / "grid_drakes/games.parquet"
    if not all(path.is_file() for path in (provider, riot, metadata, context_path)):
        pytest.skip("already-local GRID overlap is not present")
    context = foundation.context_from_grid_games_parquet(
        context_path,
        series_id="2974295",
        provider_game_id="e5b5d5bf-26b2-4ec5-aa8e-1163038d491a",
    )

    artifact = foundation.build_foundation_artifact(
        provider_events_path=provider,
        riot_events_path=riot,
        series_metadata_path=metadata,
        context=context,
    )

    assert artifact["verified_game"]["status"] == "verified"
    assert artifact["verified_game"]["total_kills"] == 35
    assert artifact["verified_game"]["team_kills"] == ((100, 14), (200, 21))
    assert all(
        state["state_status"] == "valid"
        and state["state_age_ms"] <= foundation.PROVISIONAL_MAX_STATE_AGE_MS
        for state in artifact["checkpoint_states"]
    )
    assert artifact["cohort_manifest"]["authority_receipt"]["status"] == "unavailable"
