from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from lol_kills.v2.ratings.player.multileague_development import (
    CLAIM_CEILING,
    MAP_DEVELOPMENT_COLUMNS,
    MAP_METADATA_COLUMNS,
    PLAYER_METADATA_COLUMNS,
    ROOT,
    MultiLeagueDevelopmentError,
    load_multileague_development_input,
)


# These are test-fixture identities only.  Production/development callers must
# still provide their own expected digests; the adapter contains no accepted
# pin and this test module is not an authority registry.
CURRENT_TEST_MAPS_SHA256 = "04c0cce1d86a4358d9eeb5937f61d5288358953e66c693a1ce88b0b650295d08"
CURRENT_TEST_PLAYERS_SHA256 = "12f1cca978d683a0df8ceec0772999aeb03c723b4465f98674247f327dea71fa"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _map(
    game_id: str,
    at: str,
    game: int,
    *,
    league: str = "LCS",
    blue_win: int = 1,
    blue_team_id: str = "oe:team:blue",
    red_team_id: str = "oe:team:red",
) -> dict[str, Any]:
    return {
        "game_uid": game_id,
        "oe_gameid": game_id,
        "url": None,
        "league": league,
        "year": int(at[:4]),
        "split": "Spring",
        "playoffs": 0,
        "date": pd.Timestamp(at),
        "game": game,
        "patch": 15.10,
        "competition_scope": "regional" if league == "LCS" else "international",
        "event_kind": "domestic" if league == "LCS" else "international",
        "is_international": league != "LCS",
        "blue_team_key": blue_team_id.removeprefix("oe:team:"),
        "red_team_key": red_team_id.removeprefix("oe:team:"),
        "blue_team": "Blue Team",
        "red_team": "Red Team",
        "blue_teamid": blue_team_id,
        "red_teamid": red_team_id,
        "source_lp": False,
        "lp_matched": False,
        "lp_game_id": None,
        "y_blue_win": blue_win,
    }


def _players(item: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for side, team_id, team_name in (
        ("Blue", item["blue_teamid"], item["blue_team"]),
        ("Red", item["red_teamid"], item["red_team"]),
    ):
        for role in ("top", "jng", "mid", "bot", "sup"):
            player_id = f"oe:player:{team_id.rsplit(':', 1)[-1]}:{role}"
            values.append(
                {
                    "game_uid": item["game_uid"],
                    "league": item["league"],
                    "date": item["date"],
                    "game": item["game"],
                    "side": side,
                    "position": role,
                    "playername": player_id.rsplit(":", 1)[-1],
                    "playerid": player_id,
                    "teamname": team_name,
                    "teamid": team_id,
                    "team_key": team_id.removeprefix("oe:team:"),
                }
            )
    return values


class FakeParquetReader:
    def __init__(self, maps: list[dict[str, Any]], players: list[dict[str, Any]]):
        self.maps = pd.DataFrame(maps)
        self.players = pd.DataFrame(players)
        self.calls: list[dict[str, Any]] = []

    def __call__(
        self,
        path: Path,
        *,
        columns: list[str],
        filters: list[tuple[str, str, Any]],
        engine: str,
    ) -> pd.DataFrame:
        self.calls.append(
            {"path": Path(path), "columns": tuple(columns), "filters": tuple(filters), "engine": engine}
        )
        frame = self.players.copy() if "players" in Path(path).name else self.maps.copy()
        for column, operation, value in filters:
            if operation == "in":
                frame = frame[frame[column].isin(value)]
            elif operation == ">=":
                frame = frame[frame[column] >= value]
            elif operation == "<":
                frame = frame[frame[column] < value]
            else:  # pragma: no cover - the adapter owns this fixed filter language
                raise AssertionError(operation)
        return frame.loc[:, columns].reset_index(drop=True)


def _fixture(tmp_path: Path, maps: list[dict[str, Any]]) -> tuple[FakeParquetReader, dict[str, Any]]:
    (tmp_path / "warehouse").mkdir()
    maps_path = tmp_path / "warehouse/maps.parquet"
    players_path = tmp_path / "warehouse/players.parquet"
    maps_path.write_bytes(b"maps fixture\n")
    players_path.write_bytes(b"players fixture\n")
    players = [row for item in maps for row in _players(item)]
    reader = FakeParquetReader(maps, players)
    arguments = {
        "root": tmp_path,
        "maps_locator": "warehouse/maps.parquet",
        "players_locator": "warehouse/players.parquet",
        "expected_maps_sha256": _sha(maps_path),
        "expected_players_sha256": _sha(players_path),
        "parquet_reader": reader,
    }
    return reader, arguments


def _four_period_fixture() -> list[dict[str, Any]]:
    return [
        _map("train-1", "2025-06-01T10:00:00", 1),
        _map("train-2", "2025-06-01T11:00:00", 2, blue_win=0),
        _map("development-1", "2025-08-01T10:00:00", 1),
        _map("validation-1", "2026-02-01T10:00:00", 1, blue_win=0),
        _map("sealed-1", "2026-05-01T10:00:00", 1),
    ]


def test_sealed_lane_never_projects_an_outcome_column(tmp_path: Path) -> None:
    reader, arguments = _fixture(tmp_path, _four_period_fixture())
    value = load_multileague_development_input(**arguments)

    assert [series.fold_id for series in value.development_series] == [
        "TRAIN",
        "DEVELOPMENT",
        "VALIDATION",
    ]
    assert len(value.development_series[0].maps) == 2
    assert len(value.sealed_series_metadata) == 1
    assert not hasattr(value.sealed_series_metadata[0].maps[0], "blue_win")
    assert value.claim_ceiling == CLAIM_CEILING
    assert value.claim_ceiling["sealed_final_targets_accessed"] is False
    assert value.claim_ceiling["prediction"] is False

    assert [call["columns"] for call in reader.calls] == [
        MAP_DEVELOPMENT_COLUMNS,
        MAP_METADATA_COLUMNS,
        PLAYER_METADATA_COLUMNS,
    ]
    sealed_call = reader.calls[1]
    assert "y_blue_win" not in sealed_call["columns"]
    assert ("date", ">=", pd.Timestamp("2026-04-01T00:00:00")) in sealed_call["filters"]


def test_digest_mismatch_fails_before_any_parquet_projection(tmp_path: Path) -> None:
    reader, arguments = _fixture(tmp_path, _four_period_fixture())
    arguments["expected_maps_sha256"] = "0" * 64

    with pytest.raises(MultiLeagueDevelopmentError, match="independent expected digest"):
        load_multileague_development_input(**arguments)
    assert reader.calls == []


def test_bad_lineup_quarantines_the_complete_series(tmp_path: Path) -> None:
    maps = _four_period_fixture()
    reader, arguments = _fixture(tmp_path, maps)
    duplicate = reader.players[
        (reader.players["game_uid"] == "train-2")
        & (reader.players["side"] == "Red")
        & (reader.players["position"] == "top")
    ].index[0]
    source = reader.players[
        (reader.players["game_uid"] == "train-2")
        & (reader.players["side"] == "Blue")
        & (reader.players["position"] == "top")
    ].iloc[0]
    reader.players.loc[duplicate, "playerid"] = source["playerid"]

    value = load_multileague_development_input(**arguments)

    assert not any(series.fold_id == "TRAIN" for series in value.development_series)
    assert any(
        set(item.game_ids) == {"train-1", "train-2"}
        and "lineup_player_identity_not_globally_distinct" in item.reasons
        for item in value.quarantined_clusters
    )


def test_series_crossing_a_fold_is_not_split_or_reassigned(tmp_path: Path) -> None:
    maps = [
        _map("boundary-1", "2025-06-30T23:30:00", 1),
        _map("boundary-2", "2025-07-01T00:30:00", 2),
        _map("validation-1", "2026-02-01T10:00:00", 1),
        _map("sealed-1", "2026-05-01T10:00:00", 1),
    ]
    _reader, arguments = _fixture(tmp_path, maps)

    value = load_multileague_development_input(**arguments)

    assert any(
        set(item.game_ids) == {"boundary-1", "boundary-2"}
        and "series_crosses_temporal_fold" in item.reasons
        for item in value.quarantined_clusters
    )
    accepted = {
        item.game_id for series in value.development_series for item in series.maps
    }
    assert accepted.isdisjoint({"boundary-1", "boundary-2"})


def test_exact_context_timestamp_counter_collision_is_quarantined(tmp_path: Path) -> None:
    maps = _four_period_fixture()
    maps.extend(
        [
            _map("collision-a", "2025-09-01T10:00:00", 1),
            _map("collision-b", "2025-09-01T10:00:00", 1),
        ]
    )
    _reader, arguments = _fixture(tmp_path, maps)

    value = load_multileague_development_input(**arguments)

    collisions = {
        item.game_ids[0]: item.reasons
        for item in value.quarantined_clusters
        if item.game_ids[0].startswith("collision-")
    }
    assert set(collisions) == {"collision-a", "collision-b"}
    assert all("exact_context_time_counter_collision" in reasons for reasons in collisions.values())


def test_noncanonical_or_symlinked_locator_is_rejected(tmp_path: Path) -> None:
    _reader, arguments = _fixture(tmp_path, _four_period_fixture())
    arguments["maps_locator"] = "warehouse/../warehouse/maps.parquet"
    with pytest.raises(MultiLeagueDevelopmentError, match="canonical relative POSIX"):
        load_multileague_development_input(**arguments)

    arguments["maps_locator"] = "warehouse/maps-link.parquet"
    (tmp_path / "warehouse/maps-link.parquet").symlink_to(tmp_path / "warehouse/maps.parquet")
    with pytest.raises(MultiLeagueDevelopmentError, match="symlink"):
        load_multileague_development_input(**arguments)


@pytest.mark.skipif(
    not (ROOT / "data/lol/warehouse/parquet/maps.parquet").exists(),
    reason="private warehouse snapshot is absent",
)
def test_current_snapshot_reconciles_without_accessing_sealed_targets() -> None:
    value = load_multileague_development_input(
        expected_maps_sha256=CURRENT_TEST_MAPS_SHA256,
        expected_players_sha256=CURRENT_TEST_PLAYERS_SHA256,
    )

    assert value.coverage["selected_maps"] == 3524
    assert value.coverage["accepted_maps"] == 3521
    assert value.coverage["development_maps"] == 2514
    assert value.coverage["sealed_metadata_maps"] == 1007
    assert value.coverage["quarantined_maps"] == 3
    assert len(value.development_series) == 1021
    assert len(value.sealed_series_metadata) == 398
    assert len(value.quarantined_clusters) == 2
    assert all(
        not hasattr(item, "blue_win")
        for series in value.sealed_series_metadata
        for item in series.maps
    )
    assert value.cluster_partition_sha256 == (
        "d7c59b5eb0fb278f6c7c8aadf4193d8d5f42993f717dc95932b37822c6c48906"
    )
