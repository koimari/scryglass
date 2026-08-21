from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from benchmarks.build_future_value_source_trust import (
    CENSUS_FILE,
    FREEZE_FILE,
    RUN_FILE,
    SOURCE_RECEIPT_FILE,
    SourceTrustError,
    build_source_trust,
)
from lol_kills.research.future_value_rating import validate_future_value_source_receipt_payload


def _write_json(path: Path, value: object) -> tuple[int, str]:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    path.write_bytes(raw)
    return len(raw), hashlib.sha256(raw).hexdigest()


def _source(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    root = tmp_path / "source"
    raw = root / "raw"
    bridge = root / "bridge"
    raw.mkdir(parents=True)
    bridge.mkdir()
    maps: list[dict[str, object]] = []
    players: list[dict[str, object]] = []
    teams: list[dict[str, object]] = []
    rows = [
        ("bridge-1", "annual-1", 1),
        ("annual-1", "annual-1", 1),
        ("annual-2", "annual-2", 0),
    ]
    for game_id, _survivor, result in rows:
        maps.append(
            {
                "game_uid": game_id,
                "date": pd.Timestamp("2026-08-20T10:00:00Z"),
                "league": "TEST",
                "tournament": "Test Cup",
                "patch": "16.15",
                "y_blue_win": result,
                "blue_team": "Blue",
                "red_team": "Red",
                "blue_team_key": "blue",
                "red_team_key": "red",
            }
        )
        for side, team_id in (("Blue", "oe:team:blue"), ("Red", "oe:team:red")):
            teams.append(
                {
                    "game_uid": game_id,
                    "date": pd.Timestamp("2026-08-20T10:00:00Z"),
                    "side": side,
                    "teamid": team_id,
                    "teamname": side,
                }
            )
        for side, team_id in (("Blue", "oe:team:blue"), ("Red", "oe:team:red")):
            for position, role in enumerate(("top", "jng", "mid", "bot", "sup"), 1):
                players.append(
                    {
                        "game_uid": game_id,
                        "date": pd.Timestamp("2026-08-20T10:00:00Z"),
                        "side": side,
                        "position": role,
                        "playerid": f"oe:player:{game_id}-{side}-{position}",
                        "playername": f"{game_id}-{side}-{position}",
                        "teamid": team_id,
                        "teamname": side,
                        "champion": f"Champion{position}",
                    }
                )
    pd.DataFrame(maps).to_parquet(root / "maps.parquet")
    pd.DataFrame(players).to_parquet(root / "oe_player_games.parquet")
    pd.DataFrame(teams).to_parquet(root / "oe_team_games.parquet")
    (raw / "2026_LoL_esports_match_data_from_OraclesElixir.csv").write_text(
        "gameid,date\nbridge-1,2026-08-20T10:00:00Z\nannual-1,2026-08-20T10:00:00Z\nannual-2,2026-08-20T10:00:00Z\n",
        encoding="utf-8",
    )
    audit = {
        "schema_version": "scryglass:duplicate-audit:v1",
        "assignments": [
            {
                "bridge_game_id": "bridge-1",
                "annual_survivor_game_id": "annual-1",
                "scoreboard_game_id": "test/series_1",
                "scoreboard_riot_platform_game_id": "LOLTEST_1",
            }
        ],
    }
    audit_body = dict(audit)
    audit["crosswalk_sha256"] = hashlib.sha256(
        json.dumps(audit_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact_path = bridge / "duplicate-audit.json"
    artifact_bytes, artifact_hash = _write_json(artifact_path, audit)
    receipt = {
        "schema_version": "scryglass:duplicate-audit-receipt:v1",
        "artifact": {"path": str(artifact_path), "bytes": artifact_bytes, "sha256": artifact_hash},
        "source_identity_sha256": "placeholder",
        "authority": {"research_only": True},
    }
    # The source identity is filled after the canonical accepted IDs are known.
    spec = {
        "schema_version": "scryglass:future-value-source-trust-spec:v1",
        "source_as_of": "2026-08-20T10:00:00Z",
        "exclude_game_ids": ["bridge-1"],
        "duplicate_resolution": {
            "survivor_rule": "annual_row_is_survivor_verified_external_identity",
            "mappings": [
                {
                    "bridge_game_id": "bridge-1",
                    "annual_survivor_game_id": "annual-1",
                    "evidence": {
                        "semantic_fields": ["date", "league", "patch"],
                    },
                }
            ],
            "source_binding": {
                "artifact": {"path": str(artifact_path)},
                "receipt": {"path": str(bridge / "duplicate-audit.receipt.json")},
            },
        },
    }
    return root, {"spec": spec, "receipt": receipt, "artifact": artifact_path}


def _make_fixture(tmp_path: Path) -> tuple[Path, dict[str, object], Path]:
    root, values = _source(tmp_path)
    maps = pd.read_parquet(root / "maps.parquet")
    accepted_ids = sorted(set(maps.game_uid) - {"bridge-1"})
    accepted_identity = hashlib.sha256(("\n".join(accepted_ids) + "\n").encode()).hexdigest()
    receipt = dict(values["receipt"])
    receipt["source_identity_sha256"] = accepted_identity
    receipt_body = dict(receipt)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_path = root / "bridge" / "duplicate-audit.receipt.json"
    _write_json(receipt_path, receipt)
    return root, values, receipt_path


def test_build_source_trust_binds_all_identities_and_map_projection(tmp_path: Path) -> None:
    root, values, _ = _make_fixture(tmp_path)
    output = tmp_path / "out"
    run = build_source_trust(
        source_root=root,
        output_root=output,
        resolution_spec=values["spec"],
        expected_unfiltered_count=3,
        expected_accepted_count=2,
    )
    assert run["source_game_count"] == 2
    assert run["unfiltered_source_game_count"] == 3
    assert run["duplicate_audit"]["receipt_sha256"]
    assert run["authority"]["research_only"] is True
    assert run["authority"]["public"] is False
    assert (output / CENSUS_FILE).is_file()
    assert (output / SOURCE_RECEIPT_FILE).is_file()
    assert (output / FREEZE_FILE).is_file()
    assert (output / RUN_FILE).is_file()
    rows = json.loads((output / "accepted-oe-map-rows.json").read_text())
    assert rows == [
        {
            "date": "2026-08-20T10:00:00Z",
            "gameid": "annual-1",
            "league": "TEST",
            "patch": "16.15",
            "team_keys": ["blue", "red"],
            "teams": ["Blue", "Red"],
            "tournament": "Test Cup",
        },
        {
            "date": "2026-08-20T10:00:00Z",
            "gameid": "annual-2",
            "league": "TEST",
            "patch": "16.15",
            "team_keys": ["blue", "red"],
            "teams": ["Blue", "Red"],
            "tournament": "Test Cup",
        },
    ]
    receipt = json.loads((output / SOURCE_RECEIPT_FILE).read_text())
    validate_future_value_source_receipt_payload(receipt)
    assert receipt["source_game_count"] == 2


def test_source_trust_rejects_changed_audit_receipt(tmp_path: Path) -> None:
    root, values, receipt_path = _make_fixture(tmp_path)
    receipt_path.write_bytes(receipt_path.read_bytes() + b"tampered")
    with pytest.raises(SourceTrustError, match="duplicate audit receipt self-hash|duplicate audit receipt"):
        build_source_trust(
            source_root=root,
            output_root=tmp_path / "out",
            resolution_spec=values["spec"],
        )


def test_source_trust_rejects_exclusion_mutation(tmp_path: Path) -> None:
    root, values, _ = _make_fixture(tmp_path)
    spec = copy.deepcopy(values["spec"])
    spec["exclude_game_ids"] = ["annual-2"]
    with pytest.raises(SourceTrustError, match="duplicate mapping exclusion|mappings do not cover|semantic field|source identity changed"):
        build_source_trust(
            source_root=root,
            output_root=tmp_path / "out",
            resolution_spec=spec,
        )


def test_source_trust_rejects_symlink_source(tmp_path: Path) -> None:
    root, values, _ = _make_fixture(tmp_path)
    original = root / "maps.parquet"
    moved = root / "maps.real.parquet"
    original.rename(moved)
    original.symlink_to(moved)
    with pytest.raises(SourceTrustError, match="symlink"):
        build_source_trust(
            source_root=root,
            output_root=tmp_path / "out",
            resolution_spec=values["spec"],
        )


def test_source_trust_rejects_forged_audit_identity(tmp_path: Path) -> None:
    root, values, receipt_path = _make_fixture(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["source_identity_sha256"] = "a" * 64
    body = dict(receipt)
    body.pop("receipt_sha256", None)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _write_json(receipt_path, receipt)
    with pytest.raises(SourceTrustError, match="source identity changed"):
        build_source_trust(
            source_root=root,
            output_root=tmp_path / "out",
            resolution_spec=values["spec"],
        )


def test_source_trust_generates_new_identity_audit_from_old_crosswalk(tmp_path: Path) -> None:
    root, values, _ = _make_fixture(tmp_path)
    old_dir = tmp_path / "old"
    old_dir.mkdir()
    artifact = {
        "schema_version": "scryglass:verified-oe-leaguepedia-series-crosswalk:v2",
        "assignments": [
            {
                "oe_game_id": "annual-1",
                "scoreboard_game_id": "series_1",
                "scoreboard_riot_platform_game_id": "LOLTEST_1",
                "evidence": {
                    "identity": {
                        "exact": True,
                        "source_field": "OE.gameid",
                        "target_field": "ScoreboardGames.RiotPlatformGameId",
                        "value": "LOLTEST_1",
                    }
                },
            }
        ],
        "issues": [
            {
                "kind": "duplicate_source_assignment",
                "oe_game_id": "bridge-1",
                "scoreboard_game_id": "series_1",
            }
        ],
    }
    artifact_body = dict(artifact)
    artifact["crosswalk_sha256"] = hashlib.sha256(
        json.dumps(artifact_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    artifact_path = old_dir / "crosswalk.json"
    artifact_bytes, artifact_hash = _write_json(artifact_path, artifact)
    receipt = {
        "schema_version": "scryglass:verified-oe-leaguepedia-series-crosswalk-receipt:v1",
        "artifact": {"path": str(artifact_path), "bytes": artifact_bytes, "sha256": artifact_hash},
        "crosswalk_sha256": artifact["crosswalk_sha256"],
        "authority": {key: value for key, value in {"research_only": True, "public": False}.items()},
    }
    receipt_body = dict(receipt)
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt_body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    receipt_path = old_dir / "crosswalk.receipt.json"
    _, receipt_hash = _write_json(receipt_path, receipt)
    spec = copy.deepcopy(values["spec"])
    block = spec["duplicate_resolution"]
    block.pop("source_binding")
    block["crosswalk"] = {
        "artifact": {"path": str(artifact_path)},
        "receipt": {"path": str(receipt_path)},
        "expected_receipt_file_sha256": receipt_hash,
    }
    run = build_source_trust(
        source_root=root,
        output_root=tmp_path / "out",
        resolution_spec=spec,
    )
    assert run["duplicate_audit"]["receipt_sha256"]
    audit = json.loads((tmp_path / "out" / "duplicate-audit.json").read_text())
    assert audit["source_identity_sha256"] == run["source_identity_sha256"]
    assert audit["outcome_free"] is True
    assert audit["assignments"][0]["scoreboard_riot_platform_game_id"] == "LOLTEST_1"
