from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_full_current_rating_trust import (
    FullCurrentRatingTrustError,
    build_full_current_rating_trust,
)
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
