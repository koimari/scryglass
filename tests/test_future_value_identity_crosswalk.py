from __future__ import annotations

import hashlib
import json

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


def _receipt(ids: list[str], *, source_files: dict[str, dict[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-08-10T00:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": identity_sha256(ids),
        "accepted_game_ids": sorted(ids),
        "source_files": source_files,
        "authority": {"research_only": True, "public": False},
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _fixture(*, target_date: str = "2026-01-02T00:00:00Z", missing: str = "all") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object], dict[str, dict[str, object]]]:
    ids = ["game-1", "game-2"]
    raw = b"fixture-source"
    files = {
        label: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest(), "locator": f"fixture/{label}"}
        for label in ("maps", "players", "teams")
    }
    rows: list[dict[str, object]] = []
    for game_id, date, historical in (
        ("game-1", "2026-01-01T00:00:00Z", True),
        ("game-2", target_date, False),
    ):
        for side in ("Blue", "Red"):
            for role in ("top", "jng", "mid", "bot", "sup"):
                name = f"{side}-{role}"
                rows.append(
                    {
                        "gameid": game_id,
                        "date": date,
                        "league": "TEST",
                        "side": side,
                        "position": role,
                        "playername": name,
                        "playerid": f"oe:player:{name}" if historical or missing == "team" else None,
                        "teamname": f"{side} Team",
                        "teamid": f"oe:team:{side}" if historical or missing == "player" else None,
                        "champion": f"{name}-champion",
                    }
                )
    team_rows: list[dict[str, object]] = []
    for game_id, date, historical in (("game-1", "2026-01-01T00:00:00Z", True), ("game-2", target_date, False)):
        for side in ("Blue", "Red"):
            team_rows.append(
                {
                    "gameid": game_id,
                    "date": date,
                    "league": "TEST",
                    "side": side,
                    "position": "team",
                    "teamname": f"{side} Team",
                    "teamid": f"oe:team:{side}" if historical or missing == "player" else None,
                }
            )
    maps = pd.DataFrame([{"gameid": "game-1", "date": "2026-01-01T00:00:00Z"}, {"gameid": "game-2", "date": target_date}])
    receipt = _receipt(ids, source_files=files)
    return maps, pd.DataFrame(rows), pd.DataFrame(team_rows), receipt, files


def _build(**kwargs: object) -> dict[str, object]:
    maps, players, teams, receipt, files = _fixture(**kwargs)
    return build_identity_crosswalk(
        maps=maps,
        players=players,
        teams=teams,
        source_receipt=receipt,
        source_file_records=files,
    )


def test_strict_prior_candidates_recover_missing_player_and_team_ids() -> None:
    result = _build(missing="all")
    assert result["counts"]["candidate_game_count"] == 1
    assignment = result["assignments"][0]
    assert all(row["candidate_key_type"] != "source_row_stable_id" for row in assignment["player_assignments"])
    assert all(row["candidate_key_type"] != "source_row_stable_id" for row in assignment["team_assignments"])
    assert all(
        evidence["timestamp"] < assignment["target_timestamp"]
        for row in assignment["player_assignments"] + assignment["team_assignments"]
        for evidence in row["evidence"]
    )


def test_future_only_identity_is_rejected() -> None:
    maps, players, teams, receipt, files = _fixture()
    players.loc[players["gameid"] == "game-1", "playerid"] = None
    players.loc[players["gameid"] == "game-2", "playerid"] = [f"oe:player:future-{i}" for i in range(10)]
    teams.loc[teams["gameid"] == "game-1", "teamid"] = None
    teams.loc[teams["gameid"] == "game-2", "teamid"] = ["oe:team:future-blue", "oe:team:future-red"]
    result = build_identity_crosswalk(maps=maps, players=players, teams=teams, source_receipt=receipt, source_file_records=files)
    assert result["counts"]["candidate_game_count"] == 0
    assert any("strict_prior" in reason for reason in result["rejected"][0]["reasons"])


def test_repeated_player_name_is_rejected_when_context_is_ambiguous() -> None:
    maps, players, teams, receipt, files = _fixture(missing="player")
    players.loc[(players["gameid"] == "game-1") & (players["playername"] == "Blue-top"), "playername"] = "Shared"
    players.loc[(players["gameid"] == "game-1") & (players["playername"] == "Blue-jng"), "playername"] = "Shared"
    players.loc[players["gameid"] == "game-2", "playername"] = players.loc[players["gameid"] == "game-2", "playername"].replace({"Blue-top": "Shared"})
    result = build_identity_crosswalk(maps=maps, players=players, teams=teams, source_receipt=receipt, source_file_records=files)
    assert result["counts"]["candidate_game_count"] == 0
    assert any("ambiguous_player_identity" in item["reasons"] for item in result["rejected"])


def test_role_side_and_champion_closure_fail_closed() -> None:
    maps, players, teams, receipt, files = _fixture(missing="player")
    players.loc[(players["gameid"] == "game-2") & (players["position"] == "sup"), "position"] = "mid"
    players.loc[(players["gameid"] == "game-2") & (players["position"] == "bot"), "champion"] = "Blue-top-champion"
    result = build_identity_crosswalk(maps=maps, players=players, teams=teams, source_receipt=receipt, source_file_records=files)
    assert result["counts"]["candidate_game_count"] == 0
    reasons = set(result["rejected"][0]["reasons"])
    assert "player_role_or_side_closure_invalid" in reasons
    assert "champion_identity_not_unique" in reasons


def test_source_receipt_and_file_binding_are_required() -> None:
    maps, players, teams, receipt, files = _fixture(missing="player")
    bad_receipt = dict(receipt)
    bad_receipt["source_identity_sha256"] = "0" * 64
    with pytest.raises(IdentityCrosswalkError, match="source receipt identity"):
        build_identity_crosswalk(maps=maps, players=players, teams=teams, source_receipt=bad_receipt, source_file_records=files)
    bad_files = dict(files)
    bad_files["players"] = dict(files["players"], sha256="0" * 64)
    with pytest.raises(IdentityCrosswalkError, match="source file binding"):
        build_identity_crosswalk(maps=maps, players=players, teams=teams, source_receipt=receipt, source_file_records=bad_files)


def test_outcome_mutation_does_not_change_identity_receipt() -> None:
    maps, players, teams, receipt, files = _fixture(missing="player")
    baseline = build_identity_crosswalk(maps=maps, players=players, teams=teams, source_receipt=receipt, source_file_records=files)
    players["result"] = 1
    players["winner"] = "forged"
    teams["result"] = 0
    mutated = build_identity_crosswalk(maps=maps, players=players, teams=teams, source_receipt=receipt, source_file_records=files)
    assert baseline == mutated


def test_tamper_and_anonymous_ids_fail_verification() -> None:
    result = _build(missing="all")
    maps, _players, _teams, receipt, files = _fixture(missing="all")
    tampered = dict(result)
    tampered["source_identity_sha256"] = "0" * 64
    with pytest.raises(IdentityCrosswalkError, match="hash does not match"):
        verify_identity_crosswalk(tampered, source_receipt=receipt, source_file_records=files)
    anonymous = json.loads(json.dumps(result))
    anonymous["assignments"][0]["player_assignments"][0]["player_id"] = "anon:game-2:blue:top"
    anonymous_body = dict(anonymous)
    anonymous_body.pop("receipt_sha256")
    anonymous["receipt_sha256"] = hashlib.sha256(_canonical(anonymous_body)).hexdigest()
    with pytest.raises(IdentityCrosswalkError, match="player ID is invalid"):
        verify_identity_crosswalk(anonymous, source_receipt=receipt, source_file_records=files)
