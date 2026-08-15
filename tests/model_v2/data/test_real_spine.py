from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest
import pandas as pd
import pyarrow as pa

import lol_kills.v2.data.real_spine as spine


def _sha(value: bytes = b"evidence") -> str:
    return hashlib.sha256(value).hexdigest()


def _receipt(*, locator: str = "receipts/oe-lpl.json", digest: str | None = None) -> dict[str, object]:
    return {
        "receipt_id": "oe-lpl",
        "source_id": "oracle-elixir",
        "source_snapshot_id": "oe-lpl-2025-2026-private-v1",
        "source_snapshot_row_id": "oe-row-1",
        "source_record_id": "11995-11995_game_1",
        "source_content_sha256": "a" * 64,
        "source_observed_at": "2025-01-01T00:00:00Z",
        "rights_status": "PRIVATE_REVIEWED",
        "evidence_locator": locator,
        "evidence_sha256": _sha() if digest is None else digest,
    }


def _identity_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for canonical_id in ("team-a", "team-b", *[f"a-{role}" for role in ("top", "jungle", "mid", "bot", "support")], *[f"b-{role}" for role in ("top", "jungle", "mid", "bot", "support")]):
        rows.append(
            {
                "row_id": f"identity-{canonical_id}",
                "receipt_id": "oe-lpl",
                "entity_type": "team" if canonical_id.startswith("team-") else "player",
                "canonical_id": canonical_id,
                "canonical_name": canonical_id,
                "alias": canonical_id,
                "effective_from": "2025-01-01T00:00:00Z",
                "effective_to": None,
                "precedence": 1,
                "observed_at": "2025-01-01T00:00:00Z",
                "source_updated_at": "2025-01-01T00:00:00Z",
                "available_at": "2025-01-01T00:00:00Z",
            }
        )
    return rows


def _roster_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for team in ("a", "b"):
        for role in ("top", "jungle", "mid", "bot", "support"):
            rows.append(
                {
                    "row_id": f"roster-{team}-{role}",
                    "receipt_id": "oe-lpl",
                    "roster_id": f"roster-{team}",
                    "organization_id": f"team-{team}",
                    "organization_name": f"Team {team.upper()}",
                    "role": role,
                    "player_id": f"{team}-{role}",
                    "player_name": f"{team}-{role}",
                    "effective_from": "2025-01-01T00:00:00Z",
                    "effective_to": None,
                    "precedence": 1,
                    "source_updated_at": "2025-01-01T00:00:00Z",
                    "observed_at": "2025-01-01T00:00:00Z",
                    "available_at": "2025-01-01T00:00:00Z",
                    "is_substitute": False,
                    "is_provisional": False,
                }
            )
    return rows


def _record(
    *,
    bmid: str = "11995",
    game_number: int = 1,
    event_start: str = "2025-01-10T10:00:00Z",
    partition: str = "TRAIN",
    prior: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    game_id = f"{bmid}-{bmid}_game_{game_number}"
    return {
        "map_id": f"map:{game_id}",
        "canonical_series_id": f"source:oe:lpl:bmid:{bmid}",
        "source_series_id": bmid,
        "source_game_id": game_id,
        # OE's LPL URL has the bmid/series family, not the full map identifier.
        "source_game_url": f"https://lpl.qq.com/es/stats.shtml?bmid={bmid}",
        "source_game_number": game_number,
        "league_id": "LPL",
        "tournament_id": "LPL-2025-S1",
        "season_id": "2025",
        "event_start": event_start,
        "event_end": "2025-01-10T10:30:00Z",
        "pre_event_as_of": "2025-01-10T09:59:59Z",
        "partition": partition,
        "team_ids": ["team-a", "team-b"],
        "player_ids": [f"a-{role}" for role in ("top", "jungle", "mid", "bot", "support")] + [f"b-{role}" for role in ("top", "jungle", "mid", "bot", "support")],
        "roster_ids": {"team-a": "roster-a", "team-b": "roster-b"},
        "side_mapping": {"A_game_side": "blue", "B_game_side": "red", "A_draft_order": "first", "B_draft_order": "second"},
        "source_receipt_ids": ["oe-lpl"],
        "target_authority_receipt_id": "target-1",
        "prior_map_receipts": [] if prior is None else prior,
    }


def _payload() -> dict[str, object]:
    record = _record(
        prior=[
            {
                "origin_map_id": "map:older",
                "origin_event_end": "2025-01-07T09:00:00Z",
                "origin_source_series_id": "11994",
                "source_receipt_id": "oe-lpl",
                "value_sha256": "b" * 64,
            },
            {
                "origin_map_id": "map:future",
                "origin_event_end": "2025-01-11T09:00:00Z",
                "origin_source_series_id": "11996",
                "source_receipt_id": "oe-lpl",
                "value_sha256": "c" * 64,
            },
        ]
    )
    return {
        "schema_version": "scryglass:real-v1-pre-event-input:v1",
        "snapshot_id": "private-lpl-real-v1",
        "source_receipts": [_receipt()],
        "identity_rows": _identity_rows(),
        "roster_rows": _roster_rows(),
        "target_authority_receipts": [
            {
                "receipt_id": "target-1",
                "source_receipt_id": "oe-lpl",
                "target_record_id": "11995-11995_game_1",
                "target_payload_sha256": "d" * 64,
                "target_available_at": "2025-01-10T10:30:01Z",
                "correction_status": "ORIGINAL",
                "authority_status": "PRIVATE_VERIFIED",
                "authority_locator": "data/lol/v2/models/draft-interactions/oe-private-target-authority.json",
                "authority_raw_sha256": "b1d0a6e37abb9a74dee8689dc19ab54d30fd15516bd4ee454906a075d8f20788",
                "evidence_payload_sha256": "6697ed142324f86e9b233c4a2b36dd501584e7e64449bb6cd9404f6a367d74f9",
                "split_payload_sha256": "469c8d2c568a6a4480db277bf41f7eacf72964e33997f0a4e1f53f60285cd3e4",
            }
        ],
        "records": [record],
        "split_assignments": [{"source_series_id": "11995", "partition": "TRAIN"}],
        "final_holdout": {"status": "SEALED_UNREAD", "cutoff": "2026-06-01T00:00:00Z", "receipt_sha256": "e" * 64},
        "availability_policy": {"kind": "RETROSPECTIVE_FIXED_EMBARGO", "embargo_hours": spine.EMBARGO_HOURS, "development_only": True},
    }


def test_lpl_source_family_url_and_ordinal_are_accepted_without_a_time_heuristic() -> None:
    packet = spine.build_real_v1_packet(_payload())
    assert packet["coverage"]["map_count"] == 1
    assert packet["coverage"]["source_series_family_count"] == 1
    assert packet["coverage"]["prior_map_eligible_after_embargo_count"] == 1
    assert packet["coverage"]["prior_map_excluded_by_embargo_count"] == 1
    assert packet["coverage"]["maps_with_at_least_one_eligible_prior"] == 1
    assert packet["final_holdout"] == {
        "status": "SEALED_UNREAD",
        "cutoff": "2026-06-01T00:00:00Z",
        "receipt_sha256": "e" * 64,
        "accessed": False,
    }


def test_legacy_authority_locator_resolves_only_to_the_exact_archived_receipt() -> None:
    archived = spine._resolve_koi_mari_authority_path(
        spine.LEGACY_KOI_MARI_AUTHORITY_LOCATOR,
        expected_raw_sha256=spine.KOI_MARI_AUTHORITY_RAW_SHA256,
    )
    assert archived == spine.REPO_ROOT / spine.KOI_MARI_AUTHORITY_LOCATOR
    assert spine._raw_file_sha256(archived) == spine.KOI_MARI_AUTHORITY_RAW_SHA256

    current = spine.REPO_ROOT / spine.LEGACY_KOI_MARI_AUTHORITY_LOCATOR
    assert spine._raw_file_sha256(current) != spine.KOI_MARI_AUTHORITY_RAW_SHA256
    unresolved = spine._resolve_koi_mari_authority_path(
        spine.LEGACY_KOI_MARI_AUTHORITY_LOCATOR,
        expected_raw_sha256="f" * 64,
    )
    assert unresolved == current


def test_future_prior_mutation_never_becomes_eligible_for_an_earlier_map() -> None:
    first = spine.build_real_v1_packet(_payload())
    payload = _payload()
    payload["records"][0]["prior_map_receipts"][1]["value_sha256"] = "f" * 64
    second = spine.build_real_v1_packet(payload)
    assert first["coverage"]["prior_map_eligible_after_embargo_count"] == 1
    assert second["coverage"]["prior_map_eligible_after_embargo_count"] == 1
    assert second["coverage"]["prior_map_excluded_by_embargo_count"] == 1
    assert first["input_binding"]["records_sha256"] == second["input_binding"]["records_sha256"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda value: value["identity_rows"].append({**value["identity_rows"][0], "row_id": "collision", "canonical_id": "other-player"}), "identity collision"),
        (lambda value: value["records"][0].update(player_ids=value["records"][0]["player_ids"][:-1]), "ten distinct player_ids"),
        (lambda value: next(row for row in value["roster_rows"] if row["row_id"] == "roster-a-top").update(is_substitute=True), "exact active roster"),
        (lambda value: value["records"][0].update(canonical_series_id="dependence-cluster:fake"), "must not be a dependence cluster"),
        (lambda value: value["records"][0].update(pre_event_as_of="2025-01-10T10:00:00Z"), "before event_start"),
        (lambda value: value["target_authority_receipts"][0].update(authority_status="UNTRUSTED"), "PRIVATE_VERIFIED"),
    ],
)
def test_identity_roster_series_time_and_target_escape_routes_fail_closed(mutate, match: str) -> None:
    payload = _payload()
    mutate(payload)
    with pytest.raises(spine.RealSpineError, match=match):
        spine.build_real_v1_packet(payload)


def test_source_series_cannot_cross_partitions() -> None:
    payload = _payload()
    payload["split_assignments"].append({"source_series_id": "11995", "partition": "VALIDATION"})
    with pytest.raises(spine.RealSpineError, match="crosses split"):
        spine.build_real_v1_packet(payload)


def test_same_source_series_prior_is_explicitly_excluded() -> None:
    payload = _payload()
    payload["records"][0]["prior_map_receipts"][0]["origin_source_series_id"] = "11995"
    packet = spine.build_real_v1_packet(payload)
    assert packet["coverage"]["prior_map_eligible_after_embargo_count"] == 0
    assert packet["coverage"]["prior_map_excluded_same_source_series_count"] == 1


@pytest.mark.parametrize("kind", ("traversal", "symlink", "hardlink", "hash_drift"))
def test_receipt_path_and_hash_attacks_fail_closed(tmp_path: Path, kind: str) -> None:
    root = tmp_path / "evidence"
    root.mkdir()
    receipts = root / "receipts"
    receipts.mkdir()
    target = receipts / "oe-lpl.json"
    target.write_bytes(b"evidence")
    payload = _payload()
    if kind == "traversal":
        payload["source_receipts"][0]["evidence_locator"] = "../outside.json"
    elif kind == "symlink":
        link = receipts / "linked.json"
        link.symlink_to(target)
        payload["source_receipts"][0]["evidence_locator"] = "receipts/linked.json"
    elif kind == "hardlink":
        hard = receipts / "hard.json"
        os.link(target, hard)
        payload["source_receipts"][0]["evidence_locator"] = "receipts/hard.json"
    else:
        payload["source_receipts"][0]["evidence_sha256"] = "0" * 64
    with pytest.raises(spine.RealSpineError):
        spine.build_real_v1_packet(payload, evidence_root=root)


def test_packet_is_byte_identical_across_fresh_processes_and_rejects_tampering(tmp_path: Path) -> None:
    payload = _payload()
    local = spine.build_real_v1_packet(payload)
    payload_path = tmp_path / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    code = (
        "import json,sys; "
        "from lol_kills.v2.data.real_spine import build_real_v1_packet,canonical_packet_bytes; "
        "p=json.load(open(sys.argv[1])); "
        "sys.stdout.buffer.write(canonical_packet_bytes(build_real_v1_packet(p)))"
    )
    command = [sys.executable, "-c", code, str(payload_path)]
    first = subprocess.run(command, cwd=Path(__file__).parents[3], check=True, capture_output=True).stdout
    second = subprocess.run(command, cwd=Path(__file__).parents[3], check=True, capture_output=True).stdout
    assert first == second == spine.canonical_packet_bytes(local)
    tampered = dict(local)
    tampered["claim_scope"] = {"state": "PUBLIC"}
    with pytest.raises(spine.RealSpineError, match="packet_sha256"):
        spine.canonical_packet_bytes(tampered)
    output = tmp_path / "packet.json"
    assert spine.write_real_v1_packet(local, output) == _sha(output.read_bytes())


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_output_path_aliases_are_rejected_before_any_packet_write(tmp_path: Path, kind: str) -> None:
    """A readiness packet must not overwrite an attacker-selected alias."""

    packet = spine.build_real_v1_packet(_payload())
    backing = tmp_path / "backing.json"
    backing.write_bytes(b"leave-me-alone")
    output = tmp_path / f"{kind}.json"
    if kind == "symlink":
        output.symlink_to(backing)
    else:
        os.link(backing, output)
    before = backing.read_bytes()
    with pytest.raises(spine.RealSpineError, match="symlink|hard-linked|regular"):
        spine.write_real_v1_packet(packet, output)
    assert backing.read_bytes() == before


def test_cli_rejects_an_unrecognized_command_before_printing_a_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        spine.argparse.ArgumentParser,
        "parse_args",
        lambda _parser, _argv: spine.argparse.Namespace(command="future-command"),
    )
    with pytest.raises(SystemExit):
        spine.main([])


def _adapter_tables() -> tuple[pa.Table, pa.Table]:
    maps = pa.Table.from_pylist(
        [
            {
                "oe_gameid": "11995-11995_game_1",
                "game_uid": "11995-11995_game_1",
                "url": "https://lpl.qq.com/es/stats.shtml?bmid=11995",
                "league": "LPL",
                "date": pd.Timestamp("2025-01-10T10:00:00"),
                "game": 1,
                "year": 2025,
                "split": "Split 1",
                "playoffs": 0,
                "patch": "15.1",
                "datacompleteness": "complete",
                "blue_firstPick": 1.0,
            }
        ]
    )
    players = []
    for side, team in (("Blue", "team-a"), ("Red", "team-b")):
        for role in ("top", "jng", "mid", "bot", "sup"):
            players.append(
                {
                    "gameid": "11995-11995_game_1",
                    "league": "LPL",
                    "date": pd.Timestamp("2025-01-10T10:00:00"),
                    "game": 1,
                    "position": role,
                    "playerid": f"{team}-{role}",
                    "teamid": team,
                    "teamname": team,
                    "side": side,
                    "datacompleteness": "complete",
                }
            )
    return maps, pa.Table.from_pylist(players)


def _adapter_target_rows(*, split: str = "validation", outcome: int = 1) -> pa.Table:
    return pa.Table.from_pylist([
        {
            "game_id": "11995-11995_game_1",
            "split": split,
            "oe_date_naive": "2025-01-10T10:00:00",
            "y_blue_win": outcome,
            "source_blue_result_id": "oe-team-row:11995-11995_game_1:100",
            "source_red_result_id": "oe-team-row:11995-11995_game_1:200",
            "dependence_cluster_id": "diagnostic-only:11995",
        }
    ])


def _patch_g2_private_authority(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate adapter-shape attacks; real authority is checked separately."""

    monkeypatch.setattr(spine, "_repo_relative_locator", lambda path: Path(path).name)
    monkeypatch.setattr(spine, "_safe_repo_input_file", lambda path, _name: Path(path))
    monkeypatch.setattr(spine, "EXPECTED_LPL_PRIVATE_PARTITION_COUNTS", {"VALIDATION": 1})
    monkeypatch.setattr(spine, "EXPECTED_LPL_PRIVATE_SOURCE_FAMILIES", 1)
    original_raw_sha256 = spine._raw_file_sha256
    monkeypatch.setattr(
        spine,
        "_raw_file_sha256",
        lambda path: spine.KOI_MARI_AUTHORITY_RAW_SHA256
        if Path(path).name == "authority.json"
        else original_raw_sha256(Path(path)),
    )
    monkeypatch.setattr(
        spine,
        "validate_koi_mari_authority",
        lambda _path, *, expected_raw_sha256: {
            "evidence_payload_sha256": spine.KOI_MARI_EVIDENCE_PAYLOAD_SHA256,
            "split_payload_sha256": spine.KOI_MARI_SPLIT_PAYLOAD_SHA256,
        },
    )
    monkeypatch.setattr(
        spine,
        "_validate_target_evidence",
        lambda *_args, **_kwargs: {"private_materialization": {"raw_sha256": "a" * 64}},
    )


def test_g2_adapter_pushes_nonfinal_target_filter_and_uses_frozen_target_split(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A target split wins over any calendar rule; final cannot be read."""

    import pyarrow.parquet as pq

    maps, players = _adapter_tables()
    targets = _adapter_target_rows(split="validation")
    calls: list[tuple[str, object]] = []

    def controlled_read(path, *, columns, filters=None, **kwargs):
        calls.append((str(path), filters))
        if str(path).endswith("target.parquet"):
            assert filters == [
                ("canonical_league", "=", "LPL"),
                ("split", "in", ["train", "development", "validation"]),
            ]
            assert "final_temporal_holdout" not in str(filters)
            return targets
        assert filters is not None, "source adapter must push the sealed date cutoff into parquet"
        assert any(item[0] == "date" and item[1] == "<" for item in filters)
        return maps if str(path).endswith("maps.parquet") else players

    monkeypatch.setattr(pq, "read_table", controlled_read)
    _patch_g2_private_authority(monkeypatch)
    rows = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    (tmp_path / "maps.parquet").write_bytes(b"adapter-test")
    (tmp_path / "players.parquet").write_bytes(b"adapter-test")
    (tmp_path / "target.parquet").write_bytes(b"adapter-target-test")
    (tmp_path / "authority.json").write_bytes(b"authority")
    (tmp_path / "evidence.json").write_bytes(b"evidence")
    manifest = spine.extract_lpl_private_development_snapshot(
        maps_path=tmp_path / "maps.parquet",
        player_games_path=tmp_path / "players.parquet",
        output_rows_path=rows,
        output_manifest_path=manifest_path,
        target_rows_path=tmp_path / "target.parquet",
        authority_path=tmp_path / "authority.json",
        target_evidence_path=tmp_path / "evidence.json",
    )
    assert len(calls) == 3
    assert {key: manifest["coverage"][key] for key in (
        "map_count", "source_series_family_count", "partition_counts", "calendar_year_counts",
        "target_partition_counts", "families_with_atomic_fixed_partition", "eligible_prior_origin_count",
    )} == {
        "map_count": 1,
        "source_series_family_count": 1,
        "partition_counts": {"VALIDATION": 1},
        "calendar_year_counts": {"2025": 1},
        "target_partition_counts": {"VALIDATION": 1},
        "families_with_atomic_fixed_partition": 1,
        "eligible_prior_origin_count": 0,
    }
    assert manifest["final_holdout"] == {
        "status": "SEALED_UNREAD",
        "cutoff_local_naive": "2026-06-01T00:00:00",
        "accessed": False,
    }
    blocker_codes = {row["code"] for row in manifest["typed_blockers"]}
    assert {
        "OE_SOURCE_LOCAL_TIME_NOT_HISTORICAL_INGEST_AUTHORITY",
        "OBSERVED_MAP_PARTICIPANTS_NOT_PRE_EVENT_ROSTER_AUTHORITY",
        "SOURCE_OBSERVED_AT_NOT_BOUND",
        "G1_018_BASELINES_TYPED_UNAVAILABLE",
    } <= blocker_codes
    row = json.loads(rows.read_text(encoding="utf-8"))
    assert row["source_series_id"] == "oe:lpl:bmid:11995"
    assert row["source_game_number"] == 1
    assert row["participant_lineup_kind"] == "OBSERVED_MAP_PARTICIPANTS_NOT_PRE_EVENT_ROSTER_AUTHORITY"
    assert row["source_local_event_start"].endswith("Z") is False
    assert row["retrospective_embargo_after_local_naive"] == "2025-01-12T10:00:00"
    assert row["partition"] == "VALIDATION"  # target is frozen; calendar would be TRAIN.
    assert row["target"]["y_blue_win"] == 1
    assert row["target"]["authority"] == "KOI_MARI_PRIVATE_RETROSPECTIVE_MODEL_FIT_AND_RANK_SELECTION"
    assert row["eligible_prior_origin_map_ids"] == []
    assert manifest["claim_scope"]["available_claims"] == ["private_model_fit", "private_rank_selection"]


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda rows: rows[0].update(split="final_temporal_holdout"), "sealed final"),
        (lambda rows: rows[0].update(y_blue_win=2), "binary"),
        (lambda rows: rows[0].update(oe_date_naive="2025-01-10T10:00:01"), "timestamp disagrees"),
        (lambda rows: rows[0].update(game_id="missing-map"), "membership"),
    ],
)
def test_g2_target_escape_routes_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate, match: str,
) -> None:
    """A bypassed parquet predicate still cannot promote an invalid target row."""

    import pyarrow.parquet as pq

    maps, players = _adapter_tables()
    target_rows = _adapter_target_rows().to_pylist()
    mutate(target_rows)
    targets = pa.Table.from_pylist(target_rows)

    def controlled_read(path, *, columns, filters=None, **kwargs):
        return targets if str(path).endswith("target.parquet") else maps if str(path).endswith("maps.parquet") else players

    monkeypatch.setattr(pq, "read_table", controlled_read)
    _patch_g2_private_authority(monkeypatch)
    for name in ("maps.parquet", "players.parquet", "target.parquet", "authority.json", "evidence.json"):
        (tmp_path / name).write_bytes(b"test")
    with pytest.raises(spine.RealSpineError, match=match):
        spine.extract_lpl_private_development_snapshot(
            maps_path=tmp_path / "maps.parquet",
            player_games_path=tmp_path / "players.parquet",
            target_rows_path=tmp_path / "target.parquet",
            authority_path=tmp_path / "authority.json",
            target_evidence_path=tmp_path / "evidence.json",
            output_rows_path=tmp_path / "rows.jsonl",
            output_manifest_path=tmp_path / "manifest.json",
        )


def test_g2_source_family_cannot_cross_frozen_target_partitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bmid is a true source series and must remain in one frozen split."""

    import pyarrow.parquet as pq

    maps, players = _adapter_tables()
    second_map = dict(maps.to_pylist()[0])
    second_map.update(
        oe_gameid="11995-11995_game_2",
        game_uid="11995-11995_game_2",
        game=2,
        date=pd.Timestamp("2025-01-11T10:00:00"),
    )
    maps = pa.Table.from_pylist([*maps.to_pylist(), second_map])
    second_players = []
    for row in players.to_pylist():
        copied = dict(row)
        copied["gameid"] = "11995-11995_game_2"
        copied["date"] = pd.Timestamp("2025-01-11T10:00:00")
        copied["game"] = 2
        second_players.append(copied)
    players = pa.Table.from_pylist([*players.to_pylist(), *second_players])
    targets = pa.Table.from_pylist([
        *_adapter_target_rows(split="train").to_pylist(),
        {
            **_adapter_target_rows(split="development").to_pylist()[0],
            "game_id": "11995-11995_game_2",
            "oe_date_naive": "2025-01-11T10:00:00",
        },
    ])

    def controlled_read(path, *, columns, filters=None, **kwargs):
        return targets if str(path).endswith("target.parquet") else maps if str(path).endswith("maps.parquet") else players

    monkeypatch.setattr(pq, "read_table", controlled_read)
    _patch_g2_private_authority(monkeypatch)
    for name in ("maps.parquet", "players.parquet", "target.parquet", "authority.json", "evidence.json"):
        (tmp_path / name).write_bytes(b"test")
    with pytest.raises(spine.RealSpineError, match="family crosses"):
        spine.extract_lpl_private_development_snapshot(
            maps_path=tmp_path / "maps.parquet",
            player_games_path=tmp_path / "players.parquet",
            target_rows_path=tmp_path / "target.parquet",
            authority_path=tmp_path / "authority.json",
            target_evidence_path=tmp_path / "evidence.json",
            output_rows_path=tmp_path / "rows.jsonl",
            output_manifest_path=tmp_path / "manifest.json",
        )


def test_g2_preflights_all_output_destinations_before_replacing_either(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected manifest destination must not leave a newly written rows file."""

    import pyarrow.parquet as pq

    maps, players = _adapter_tables()
    targets = _adapter_target_rows()

    def controlled_read(path, *, columns, filters=None, **kwargs):
        return targets if str(path).endswith("target.parquet") else maps if str(path).endswith("maps.parquet") else players

    monkeypatch.setattr(pq, "read_table", controlled_read)
    _patch_g2_private_authority(monkeypatch)
    for name in ("maps.parquet", "players.parquet", "target.parquet", "authority.json", "evidence.json"):
        (tmp_path / name).write_bytes(b"test")
    rows_path = tmp_path / "rows.jsonl"
    rows_path.write_bytes(b"rows-before")
    backing_manifest = tmp_path / "manifest-backing.json"
    backing_manifest.write_bytes(b"manifest-before")
    manifest_alias = tmp_path / "manifest.json"
    manifest_alias.symlink_to(backing_manifest)
    with pytest.raises(spine.RealSpineError, match="symlink|regular"):
        spine.extract_lpl_private_development_snapshot(
            maps_path=tmp_path / "maps.parquet",
            player_games_path=tmp_path / "players.parquet",
            target_rows_path=tmp_path / "target.parquet",
            authority_path=tmp_path / "authority.json",
            target_evidence_path=tmp_path / "evidence.json",
            output_rows_path=rows_path,
            output_manifest_path=manifest_alias,
        )
    assert rows_path.read_bytes() == b"rows-before"
    assert backing_manifest.read_bytes() == b"manifest-before"


def test_persisted_g2_artifact_has_exact_authorized_counts_and_no_final_rows() -> None:
    root = Path(__file__).parents[3]
    rows_path = root / "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl"
    manifest_path = root / "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
    manifest = spine.verify_lpl_private_development_snapshot(
        rows_path=rows_path,
        manifest_path=manifest_path,
        expected_manifest_sha256="3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72",
    )
    assert manifest["rows_sha256"] == "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846"
    assert manifest["canonical_selected_target_rows_sha256"] == "4c332fa4e6cb155341bcffd83bd0ee1be2e04f3b5950b8a7745931253dd8bd2d"
    assert manifest["canonical_selected_target_row_count"] == 1226
    assert manifest["coverage"]["partition_counts"] == {"DEVELOPMENT": 214, "TRAIN": 805, "VALIDATION": 207}
    assert manifest["coverage"]["source_series_family_count"] == 471
    assert manifest["coverage"]["families_with_atomic_fixed_partition"] == 471
    assert manifest["final_holdout"]["accessed"] is False
    assert manifest["claim_scope"]["available_claims"] == ["private_model_fit", "private_rank_selection"]
    assert {"forecast", "prediction", "production", "publication", "sota"} <= set(manifest["claim_scope"]["blocked_claims"])

    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1226 == len({row["source_game_id"] for row in rows})
    assert {row["partition"] for row in rows} == {"TRAIN", "DEVELOPMENT", "VALIDATION"}
    family_partition = {row["source_series_id"]: row["partition"] for row in rows}
    assert all(family_partition[row["source_series_id"]] == row["partition"] for row in rows)
    by_game = {row["source_game_id"]: row for row in rows}
    for row in rows:
        event_start = datetime.fromisoformat(row["source_local_event_start"])
        for origin_id in row["eligible_prior_origin_map_ids"]:
            origin = by_game[origin_id]
            assert origin["source_series_id"] != row["source_series_id"]
            assert datetime.fromisoformat(origin["source_local_event_start"]) + timedelta(hours=48) < event_start


def test_persisted_g2_artifact_is_byte_identical_across_fresh_process_verification() -> None:
    root = Path(__file__).parents[3]
    rows_path = root / "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl"
    manifest_path = root / "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
    code = (
        "from pathlib import Path; import sys; "
        "from lol_kills.v2.data.real_spine import verify_lpl_private_development_snapshot; "
        "m=verify_lpl_private_development_snapshot(rows_path=Path(sys.argv[1]),manifest_path=Path(sys.argv[2]),expected_manifest_sha256=sys.argv[3]); "
        "print(m['rows_sha256']+' '+m['manifest_sha256'])"
    )
    command = [sys.executable, "-c", code, str(rows_path), str(manifest_path), "3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72"]
    first = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    second = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True).stdout
    assert first == second == "4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846 3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72\n"


@pytest.mark.parametrize("kind", ("symlink", "hardlink"))
def test_g2_input_receipt_aliases_are_rejected(kind: str) -> None:
    """All authority/source inputs must be direct repository-owned files."""

    root = Path(__file__).parents[3]
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        directory = Path(temporary)
        regular = directory / "regular.json"
        regular.write_bytes(b"receipt")
        alias = directory / f"{kind}.json"
        if kind == "symlink":
            alias.symlink_to(regular)
        else:
            os.link(regular, alias)
        with pytest.raises(spine.RealSpineError, match="symlink|hard-linked"):
            spine._safe_repo_input_file(alias, "authority_path")


def test_g2_verifier_rejects_a_self_rehashed_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Changing rows and both manifest self-hashes cannot forge the pinned artifact."""

    root = Path(__file__).parents[3]
    source_rows = root / "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl"
    source_manifest = root / "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
    rows_path = tmp_path / "rows.jsonl"
    manifest_path = tmp_path / "manifest.json"
    mutated_rows = source_rows.read_bytes() + b"\n"
    rows_path.write_bytes(mutated_rows)
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    manifest["rows_sha256"] = _sha(mutated_rows)
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256")
    monkeypatch.setattr(spine, "_safe_repo_input_file", lambda path, _name: Path(path))
    manifest["manifest_sha256"] = spine.sha256_bytes(spine.canonical_json_bytes(unsigned))
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(spine.RealSpineError, match="independently pinned"):
        spine.verify_lpl_private_development_snapshot(
            rows_path=rows_path,
            manifest_path=manifest_path,
            expected_manifest_sha256="3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72",
        )
