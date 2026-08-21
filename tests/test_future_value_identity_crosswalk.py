from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.future_value_identity_crosswalk import (
    IdentityCrosswalkError,
    build_identity_crosswalk,
    verify_identity_crosswalk,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _record(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {"bytes": len(raw), "locator": path.name, "path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()}


def _receipt(ids: list[str], source_files: dict[str, dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-08-10T00:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": sorted(ids),
        "source_files": source_files,
        "authority": {
            "deployment": False, "merge": False, "promotion": False,
            "public_player_rating": False, "public_probability": False,
            "public_team_rating": False, "research_only": True,
        },
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _bind(tmp_path: Path, maps: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame) -> tuple[dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    supplied: dict[str, dict[str, object]] = {}
    receipt_records: dict[str, dict[str, object]] = {}
    for label, frame in (("maps", maps), ("players", players), ("teams", teams)):
        path = tmp_path / f"{label}.parquet"
        frame.to_parquet(path, index=False)
        supplied[label] = _record(path)
        receipt_records[label] = {key: value for key, value in supplied[label].items() if key != "path"}
    receipt = _receipt(["game-1", "game-2"], receipt_records)
    receipt_path = tmp_path / "source-receipt.json"
    receipt_path.write_bytes(_canonical(receipt))
    return receipt, supplied, _record(receipt_path)


def _fixture(tmp_path: Path, *, target_date: str = "2026-01-02T00:00:00Z", missing: str = "all") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for game_id, date, historical in (("game-1", "2026-01-01T00:00:00Z", True), ("game-2", target_date, False)):
        for side in ("Blue", "Red"):
            for role in ("top", "jng", "mid", "bot", "sup"):
                name = f"{side}-{role}"
                rows.append({
                    "gameid": game_id, "date": date, "league": "TEST", "side": side,
                    "position": role, "playername": name,
                    "playerid": f"oe:player:{name}" if historical or missing == "team" else None,
                    "teamname": f"{side} Team",
                    "teamid": f"oe:team:{side}" if historical or missing == "player" else None,
                    "champion": f"{name}-champion",
                })
    team_rows: list[dict[str, object]] = []
    for game_id, date, historical in (("game-1", "2026-01-01T00:00:00Z", True), ("game-2", target_date, False)):
        for side in ("Blue", "Red"):
            team_rows.append({
                "gameid": game_id, "date": date, "league": "TEST", "side": side,
                "position": "team", "teamname": f"{side} Team",
                "teamid": f"oe:team:{side}" if historical or missing == "player" else None,
            })
    maps = pd.DataFrame([{"gameid": "game-1", "date": "2026-01-01T00:00:00Z"}, {"gameid": "game-2", "date": target_date}])
    players, teams = pd.DataFrame(rows), pd.DataFrame(team_rows)
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    return maps, players, teams, receipt, files, receipt_file


def _call(maps: pd.DataFrame, players: pd.DataFrame, teams: pd.DataFrame, receipt: dict[str, object], files: dict[str, dict[str, object]], receipt_file: dict[str, object]) -> dict[str, object]:
    return build_identity_crosswalk(
        maps=maps, players=players, teams=teams, source_receipt=receipt,
        source_receipt_file_record=receipt_file,
        trusted_source_receipt_file_sha256=str(receipt_file["sha256"]),
        source_file_records=files,
    )


def _build(tmp_path: Path, **kwargs: object) -> dict[str, object]:
    return _call(*_fixture(tmp_path, **kwargs))


def test_strict_prior_candidates_recover_missing_player_and_team_ids(tmp_path: Path) -> None:
    result = _build(tmp_path, missing="all")
    assert result["counts"]["candidate_game_count"] == 1
    assignment = result["assignments"][0]
    assert all(row["candidate_key_type"] != "source_row_stable_id" for row in assignment["player_assignments"])
    assert all(row["candidate_key_type"] != "source_row_stable_id" for row in assignment["team_assignments"])
    assert all(evidence["timestamp"] < assignment["target_timestamp"] for row in assignment["player_assignments"] + assignment["team_assignments"] for evidence in row["evidence"])


def test_future_only_identity_is_rejected(tmp_path: Path) -> None:
    maps, players, teams, _, _, _ = _fixture(tmp_path)
    players.loc[players["gameid"] == "game-1", "playerid"] = None
    players.loc[players["gameid"] == "game-2", "playerid"] = [f"oe:player:future-{i}" for i in range(10)]
    teams.loc[teams["gameid"] == "game-1", "teamid"] = None
    teams.loc[teams["gameid"] == "game-2", "teamid"] = ["oe:team:future-blue", "oe:team:future-red"]
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    result = _call(maps, players, teams, receipt, files, receipt_file)
    assert result["counts"]["candidate_game_count"] == 0
    assert any("strict_prior" in reason for reason in result["rejected"][0]["reasons"])


def test_repeated_player_name_is_rejected_when_context_is_ambiguous(tmp_path: Path) -> None:
    maps, players, teams, _, _, _ = _fixture(tmp_path, missing="player")
    players.loc[(players["gameid"] == "game-1") & (players["playername"] == "Blue-top"), "playername"] = "Shared"
    players.loc[(players["gameid"] == "game-1") & (players["playername"] == "Blue-jng"), "playername"] = "Shared"
    players.loc[players["gameid"] == "game-2", "playername"] = players.loc[players["gameid"] == "game-2", "playername"].replace({"Blue-top": "Shared"})
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    result = _call(maps, players, teams, receipt, files, receipt_file)
    assert result["counts"]["candidate_game_count"] == 0
    assert any("ambiguous_player_identity" in item["reasons"] for item in result["rejected"])


def test_role_side_and_champion_closure_fail_closed(tmp_path: Path) -> None:
    maps, players, teams, _, _, _ = _fixture(tmp_path, missing="player")
    players.loc[(players["gameid"] == "game-2") & (players["position"] == "sup"), "position"] = "mid"
    players.loc[(players["gameid"] == "game-2") & (players["position"] == "bot"), "champion"] = "Blue-top-champion"
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    result = _call(maps, players, teams, receipt, files, receipt_file)
    assert result["counts"]["candidate_game_count"] == 0
    assert {"player_role_or_side_closure_invalid", "champion_identity_not_unique"} <= set(result["rejected"][0]["reasons"])


@pytest.mark.parametrize("same_side", [False, True])
def test_champion_mirror_is_side_scoped(tmp_path: Path, same_side: bool) -> None:
    maps, players, teams, _, _, _ = _fixture(tmp_path, missing="player")
    side = "Blue" if same_side else "Red"
    role = "bot" if same_side else "top"
    players.loc[(players["gameid"] == "game-2") & (players["side"] == side) & (players["position"] == role), "champion"] = "Blue-top-champion"
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    result = _call(maps, players, teams, receipt, files, receipt_file)
    assert result["counts"]["candidate_game_count"] == (0 if same_side else 1)


def test_source_receipt_and_file_binding_are_required(tmp_path: Path) -> None:
    maps, players, teams, receipt, files, receipt_file = _fixture(tmp_path, missing="player")
    bad_receipt = dict(receipt, source_identity_sha256="0" * 64)
    bad_path = tmp_path / "bad-receipt.json"
    bad_path.write_bytes(_canonical(bad_receipt))
    with pytest.raises(IdentityCrosswalkError, match="source receipt identity"):
        _call(maps, players, teams, bad_receipt, files, _record(bad_path))
    bad_files = {key: dict(value) for key, value in files.items()}
    bad_files["players"]["sha256"] = "0" * 64
    with pytest.raises(IdentityCrosswalkError, match="source file hash changed"):
        _call(maps, players, teams, receipt, bad_files, receipt_file)
    locator_only = {key: {k: v for k, v in value.items() if k != "path"} for key, value in files.items()}
    with pytest.raises(IdentityCrosswalkError, match="path is required"):
        _call(maps, players, teams, receipt, locator_only, receipt_file)


def test_supplied_rows_must_match_verified_source_bytes(tmp_path: Path) -> None:
    maps, players, teams, receipt, files, receipt_file = _fixture(tmp_path, missing="player")
    players.loc[players["gameid"] == "game-1", "playerid"] = "oe:player:forged"
    with pytest.raises(IdentityCrosswalkError, match="players rows do not match"):
        _call(maps, players, teams, receipt, files, receipt_file)


@pytest.mark.parametrize("mutation", ["missing", "duplicate"])
def test_accepted_map_rows_must_be_complete_and_unique(
    tmp_path: Path, mutation: str
) -> None:
    maps, players, teams, _, _, _ = _fixture(tmp_path, missing="player")
    if mutation == "missing":
        maps = maps[maps["gameid"] != "game-2"].reset_index(drop=True)
        message = "accepted maps are missing"
    else:
        maps = pd.concat([maps, maps[maps["gameid"] == "game-2"]], ignore_index=True)
        message = "duplicate map rows"
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    with pytest.raises(IdentityCrosswalkError, match=message):
        _call(maps, players, teams, receipt, files, receipt_file)


def test_six_unaccepted_history_maps_cannot_change_candidates(tmp_path: Path) -> None:
    maps, players, teams, _, _, _ = _fixture(tmp_path, missing="player")
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    baseline = _call(maps, players, teams, receipt, files, receipt_file)
    extra_maps, extra_players, extra_teams = [], [], []
    for index in range(6):
        game_id = f"extra-{index}"
        extra_maps.append({"gameid": game_id, "date": "2025-12-01T00:00:00Z"})
        player_rows = players[players["gameid"] == "game-1"].copy()
        player_rows[["gameid", "date"]] = [game_id, "2025-12-01T00:00:00Z"]
        player_rows["playerid"] = player_rows["playername"].map(lambda name: f"oe:player:extra:{index}:{name}")
        extra_players.extend(player_rows.to_dict("records"))
        team_rows = teams[teams["gameid"] == "game-1"].copy()
        team_rows[["gameid", "date"]] = [game_id, "2025-12-01T00:00:00Z"]
        team_rows["teamid"] = team_rows["side"].map(lambda side: f"oe:team:extra:{index}:{side}")
        extra_teams.extend(team_rows.to_dict("records"))
    maps = pd.concat([maps, pd.DataFrame(extra_maps)], ignore_index=True)
    players = pd.concat([players, pd.DataFrame(extra_players)], ignore_index=True)
    teams = pd.concat([teams, pd.DataFrame(extra_teams)], ignore_index=True)
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    with_extras = _call(maps, players, teams, receipt, files, receipt_file)
    assert with_extras["assignments"] == baseline["assignments"]
    assert with_extras["rejected"] == baseline["rejected"]
    assert with_extras["counts"] == baseline["counts"]


def test_outcome_columns_do_not_enter_identity_assignments(tmp_path: Path) -> None:
    maps, players, teams, _, _, _ = _fixture(tmp_path, missing="player")
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    baseline = _call(maps, players, teams, receipt, files, receipt_file)
    players["result"], players["winner"], teams["result"] = 1, "forged", 0
    receipt, files, receipt_file = _bind(tmp_path, maps, players, teams)
    mutated = _call(maps, players, teams, receipt, files, receipt_file)
    assert baseline["assignments"] == mutated["assignments"]
    assert baseline["rejected"] == mutated["rejected"]


def test_resealed_crosswalk_ids_and_outcome_fields_fail_replay(tmp_path: Path) -> None:
    result = _build(tmp_path, missing="all")
    _, _, _, receipt, files, receipt_file = _fixture(tmp_path, missing="all")
    forged = json.loads(json.dumps(result))
    forged["assignments"][0]["player_assignments"][0]["player_id"] = "oe:player:forged"
    body = dict(forged); body.pop("receipt_sha256")
    forged["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(IdentityCrosswalkError, match="source-row replay"):
        verify_identity_crosswalk(forged, source_receipt=receipt, source_receipt_file_record=receipt_file, trusted_source_receipt_file_sha256=str(receipt_file["sha256"]), source_file_records=files)
    outcome = json.loads(json.dumps(result))
    outcome["final_gold_diff"] = 12345
    body = dict(outcome); body.pop("receipt_sha256")
    outcome["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(IdentityCrosswalkError, match="outcome fields"):
        verify_identity_crosswalk(outcome, source_receipt=receipt, source_receipt_file_record=receipt_file, trusted_source_receipt_file_sha256=str(receipt_file["sha256"]), source_file_records=files)


def test_resealed_source_receipt_unknown_fields_and_authority_fail(tmp_path: Path) -> None:
    maps, players, teams, receipt, files, trusted_receipt_file = _fixture(tmp_path, missing="player")
    mutations = [({"forged_by": "caller"}, "unknown fields"), ({"authority": {**receipt["authority"], "public": False}}, "authority")]
    for index, (mutation, message) in enumerate(mutations):
        forged = dict(receipt); forged.update(mutation)
        body = dict(forged); body.pop("receipt_sha256")
        forged["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
        path = tmp_path / f"forged-{index}.json"; path.write_bytes(_canonical(forged))
        with pytest.raises(IdentityCrosswalkError, match=message):
            _call(maps, players, teams, forged, files, _record(path))
    resealed = dict(receipt, source_as_of="2026-08-09T00:00:00Z")
    body = dict(resealed); body.pop("receipt_sha256")
    resealed["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    path = tmp_path / "resealed-known-fields.json"; path.write_bytes(_canonical(resealed))
    resealed_record = _record(path)
    with pytest.raises(IdentityCrosswalkError, match="trust digest"):
        build_identity_crosswalk(
            maps=maps, players=players, teams=teams, source_receipt=resealed,
            source_receipt_file_record=resealed_record,
            trusted_source_receipt_file_sha256=str(trusted_receipt_file["sha256"]),
            source_file_records=files,
        )


def test_anonymous_ids_fail_verification(tmp_path: Path) -> None:
    result = _build(tmp_path, missing="all")
    _, _, _, receipt, files, receipt_file = _fixture(tmp_path, missing="all")
    anonymous = json.loads(json.dumps(result))
    anonymous["assignments"][0]["player_assignments"][0]["player_id"] = "anon:game-2:blue:top"
    body = dict(anonymous); body.pop("receipt_sha256")
    anonymous["receipt_sha256"] = hashlib.sha256(_canonical(body)).hexdigest()
    with pytest.raises(IdentityCrosswalkError, match="player ID is invalid"):
        verify_identity_crosswalk(anonymous, source_receipt=receipt, source_receipt_file_record=receipt_file, trusted_source_receipt_file_sha256=str(receipt_file["sha256"]), source_file_records=files)
