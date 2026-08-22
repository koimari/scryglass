from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_full_current_rating_trust import (
    FullCurrentRatingTrustError,
    _team_snapshot_replay,
    build_full_current_rating_trust,
)
from lol_kills.ratings.dual_elo import DualEloConfig
from benchmarks.build_future_value_snapshots import _verify_current_rating_inputs
from lol_kills.research.future_value_rating import (
    _canonical_json_bytes,
    bind_accepted_future_value_source,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256
from tests.test_future_value_rating_ledger import GAME_IDS, _source_frames


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> dict[str, object]:
    root = tmp_path / "fixture"
    source_root = root / "source"
    source_root.mkdir(parents=True)
    maps, players, teams = _source_frames()
    census_path = root / "accepted-census.json"
    census = {
        "game_ids": list(GAME_IDS),
        "game_count": len(GAME_IDS),
        "source_identity_sha256": identity_sha256(GAME_IDS),
    }
    census_path.write_bytes(_canonical_json_bytes(census))
    paths = {
        "maps": source_root / "maps.parquet",
        "players": source_root / "oe_player_games.parquet",
        "teams": source_root / "oe_team_games.parquet",
        "accepted_census": census_path,
    }
    maps.to_parquet(paths["maps"], index=False)
    players.to_parquet(paths["players"], index=False)
    teams.to_parquet(paths["teams"], index=False)
    source_files = {
        label: {
            "locator": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for label, path in paths.items()
    }
    source = bind_accepted_future_value_source(
        maps,
        players,
        teams,
        census=census,
        source_as_of="2026-01-03T00:00:00Z",
        source_files=source_files,
    )
    source_receipt_path = root / "future-value-source-receipt.json"
    source_receipt_path.write_bytes(_canonical_json_bytes(source.receipt))
    return {
        "root": root,
        "source_root": source_root,
        "source_receipt_path": source_receipt_path,
        "source_receipt": source.receipt,
        "output_root": root / "current-trust",
        "players_path": paths["players"],
        "teams_path": paths["teams"],
    }


def _build(fixture: dict[str, object]) -> dict[str, object]:
    source_receipt_path = fixture["source_receipt_path"]
    assert isinstance(source_receipt_path, Path)
    source_receipt = fixture["source_receipt"]
    assert isinstance(source_receipt, dict)
    return build_full_current_rating_trust(
        source_root=fixture["source_root"],
        source_receipt_path=source_receipt_path,
        source_receipt_file_sha256=_sha256(source_receipt_path),
        expected_source_receipt_sha256=str(source_receipt["receipt_sha256"]),
        output_root=fixture["output_root"],
    )


def _reseal_source_files(fixture: dict[str, object], *labels: str) -> None:
    source_receipt_path = fixture["source_receipt_path"]
    source_receipt = fixture["source_receipt"]
    assert isinstance(source_receipt_path, Path)
    assert isinstance(source_receipt, dict)
    value = json.loads(json.dumps(source_receipt))
    path_by_label = {
        "players": fixture["players_path"],
        "teams": fixture["teams_path"],
    }
    for label in labels:
        path = path_by_label[label]
        assert isinstance(path, Path)
        value["source_files"][label]["bytes"] = path.stat().st_size
        value["source_files"][label]["sha256"] = _sha256(path)
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    source_receipt_path.write_bytes(_canonical_json_bytes(value))
    fixture["source_receipt"] = value


def test_build_binds_source_and_emits_verified_player_team_snapshots(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    result = _build(fixture)
    assert result["status"] == "research_only"
    assert result["rows"] == len(GAME_IDS)
    assert result["authority"]["public_player_rating"] is False
    assert result["authority"]["public_team_rating"] is False
    root = fixture["output_root"]
    assert isinstance(root, Path)
    receipt_path = root / "current-rating-snapshot-receipt-v1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _verify_current_rating_inputs(
        root,
        receipt_path,
        receipt,
        source_receipt=fixture["source_receipt"],
        expected_current_receipt_sha256=_sha256(receipt_path),
    )
    ledger = pd.read_parquet(root / "current-rating-ledger.parquet")
    assert set(ledger["game_id"]) == set(GAME_IDS)
    assert ledger["game_id"].is_unique
    assert ledger["series_id"].notna().all()


def test_same_timestamp_maps_share_the_same_strict_prior_features(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    _build(fixture)
    root = fixture["output_root"]
    ledger = pd.read_parquet(root / "current-rating-ledger.parquet")
    features = [
        "base_team_logit",
        "team_rating_diff_scaled",
        "base_player_logit",
        "player_rating_diff_scaled",
    ]
    same_timestamp = ledger[ledger["game_id"].isin(("g1", "g2"))]
    for feature in features:
        assert same_timestamp[feature].nunique() == 1


def test_team_snapshot_uses_unknown_home_league_for_international_only_team() -> None:
    maps, players, _teams = _source_frames()
    maps = maps.copy()
    maps["league"] = "MSI"
    maps["tournament"] = "Mid-Season Invitational"

    _features, snapshot = _team_snapshot_replay(
        maps,
        players,
        cfg=DualEloConfig(),
    )

    assert set(snapshot["home_league"]) == {"UNKNOWN"}


def test_source_file_mutation_fails_before_replay(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    players_path = fixture["players_path"]
    assert isinstance(players_path, Path)
    players = pd.read_parquet(players_path)
    players.loc[0, "playername"] = "changed"
    players.to_parquet(players_path, index=False)
    with pytest.raises(
        FullCurrentRatingTrustError, match="source file (bytes|hash) changed: players"
    ):
        _build(fixture)


def test_missing_source_ids_use_labeled_fallback_and_stable_only_snapshots(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    players_path = fixture["players_path"]
    teams_path = fixture["teams_path"]
    assert isinstance(players_path, Path)
    assert isinstance(teams_path, Path)
    players = pd.read_parquet(players_path)
    teams = pd.read_parquet(teams_path)
    player_mask = (
        players["game_uid"].eq("g1")
        & players["side"].eq("Blue")
        & players["position"].eq("top")
    )
    team_mask = players["game_uid"].eq("g1") & players["side"].eq("Red")
    players.loc[player_mask, "playerid"] = None
    players.loc[team_mask, "teamid"] = None
    teams.loc[teams["game_uid"].eq("g1") & teams["side"].eq("Red"), "teamid"] = None
    players.to_parquet(players_path, index=False)
    teams.to_parquet(teams_path, index=False)
    _reseal_source_files(fixture, "players", "teams")

    result = _build(fixture)
    assert result["rows"] == len(GAME_IDS)
    audit = result["identity_resolution"]
    assert audit["player_rows_fallback"] == 1
    assert audit["team_rows_fallback"] == 6
    assert audit["player_states_fallback_excluded"] == 6
    assert audit["team_states_fallback_excluded"] == 1
    root = fixture["output_root"]
    assert isinstance(root, Path)
    player_snapshot = pd.read_parquet(root / "player/player_ratings_snapshot.parquet")
    team_snapshot = pd.read_parquet(root / "team/ratings_snapshot.parquet")
    assert player_snapshot["player_id"].str.startswith("oe:player:").all()
    assert team_snapshot["team_id"].str.startswith("oe:team:").all()
    ledger = pd.read_parquet(root / "current-rating-ledger.parquet")
    assert len(ledger) == len(GAME_IDS)
    assert ledger["game_id"].is_unique


def test_reused_name_does_not_infer_one_missing_stable_player_id(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    players_path = fixture["players_path"]
    assert isinstance(players_path, Path)
    players = pd.read_parquet(players_path)
    source_row = players.index[
        players["game_uid"].eq("g1")
        & players["side"].eq("Blue")
        & players["position"].eq("top")
    ][0]
    stable_id = players.loc[source_row, "playerid"]
    target_row = players.index[
        players["game_uid"].eq("g3")
        & players["side"].eq("Blue")
        & players["position"].eq("top")
    ][0]
    players.loc[target_row, "playername"] = players.loc[source_row, "playername"]
    players.loc[target_row, "playerid"] = None
    players.to_parquet(players_path, index=False)
    _reseal_source_files(fixture, "players")

    result = _build(fixture)
    audit = result["identity_resolution"]
    assert audit["player_rows_fallback"] == 1
    root = fixture["output_root"]
    assert isinstance(root, Path)
    snapshot = pd.read_parquet(root / "player/player_ratings_snapshot.parquet")
    assert stable_id in set(snapshot["player_id"])
    assert int(snapshot.loc[snapshot["player_id"].eq(stable_id), "n_maps"].iloc[0]) == 1


def test_reader_uses_the_exact_receipt_bound_source_path(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    root = fixture["root"]
    players_path = fixture["players_path"]
    source_receipt_path = fixture["source_receipt_path"]
    source_receipt = fixture["source_receipt"]
    assert isinstance(root, Path)
    assert isinstance(players_path, Path)
    assert isinstance(source_receipt_path, Path)
    assert isinstance(source_receipt, dict)
    bound_path = players_path.with_name("players.parquet")
    bound_path.write_bytes(players_path.read_bytes())
    players_path.write_bytes(b"unbound file must not be read")
    value = json.loads(json.dumps(source_receipt))
    value["source_files"]["players"] = {
        "locator": bound_path.relative_to(root).as_posix(),
        "bytes": bound_path.stat().st_size,
        "sha256": _sha256(bound_path),
    }
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = hashlib.sha256(_canonical_json_bytes(value)).hexdigest()
    source_receipt_path.write_bytes(_canonical_json_bytes(value))
    fixture["source_receipt"] = value

    result = _build(fixture)

    assert result["rows"] == len(GAME_IDS)


def test_all_fallback_identities_fail_closed_on_empty_stable_snapshots(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    players_path = fixture["players_path"]
    teams_path = fixture["teams_path"]
    assert isinstance(players_path, Path)
    assert isinstance(teams_path, Path)
    players = pd.read_parquet(players_path)
    teams = pd.read_parquet(teams_path)
    players["playerid"] = None
    players["teamid"] = None
    teams["teamid"] = None
    players.to_parquet(players_path, index=False)
    teams.to_parquet(teams_path, index=False)
    _reseal_source_files(fixture, "players", "teams")

    with pytest.raises(
        FullCurrentRatingTrustError, match="(player|team) stable snapshot is empty"
    ):
        _build(fixture)


def test_exact_five_validation_fails_closed_after_bound_source_update(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    players_path = fixture["players_path"]
    assert isinstance(players_path, Path)
    players = pd.read_parquet(players_path)
    players = players.loc[
        ~(
            players["game_uid"].eq("g4")
            & players["side"].eq("Red")
            & players["position"].eq("sup")
        )
    ]
    players.to_parquet(players_path, index=False)

    source_receipt_path = fixture["source_receipt_path"]
    assert isinstance(source_receipt_path, Path)
    source_receipt = fixture["source_receipt"]
    assert isinstance(source_receipt, dict)
    source_receipt = json.loads(json.dumps(source_receipt))
    source_receipt["source_files"]["players"]["bytes"] = players_path.stat().st_size
    source_receipt["source_files"]["players"]["sha256"] = _sha256(players_path)
    source_receipt.pop("receipt_sha256")
    source_receipt["receipt_sha256"] = hashlib.sha256(
        _canonical_json_bytes(source_receipt)
    ).hexdigest()
    source_receipt_path.write_bytes(_canonical_json_bytes(source_receipt))
    fixture["source_receipt"] = source_receipt
    with pytest.raises(FullCurrentRatingTrustError, match="exactly ten|exact five"):
        _build(fixture)


def test_output_root_must_be_empty_and_must_not_be_a_symlink(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output_root = fixture["output_root"]
    assert isinstance(output_root, Path)
    output_root.mkdir()
    (output_root / "owned.txt").write_text("x", encoding="utf-8")
    with pytest.raises(FullCurrentRatingTrustError, match="safe and empty"):
        _build(fixture)

    fixture = _fixture(tmp_path / "symlink")
    target = fixture["root"] / "target"
    target.mkdir()
    output_root = fixture["output_root"]
    output_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(FullCurrentRatingTrustError, match="symlink"):
        _build(fixture)
