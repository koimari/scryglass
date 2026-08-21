from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.research.oe_leaguepedia_alias_derivation import (
    AliasDerivationError,
    derive_team_alias_mapping,
    load_verified_alias_mapping,
    verify_alias_mapping,
)
from tools.derive_oe_leaguepedia_team_aliases import main as derive_cli


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _record(label: str, rows: list[dict[str, object]]) -> tuple[dict[str, object], bytes]:
    payload = _canonical(rows)
    raw = b"captured-" + label.encode() + b"\n" + payload
    return {
        "url": f"https://example.test/{label}.json",
        "retrieved_at": "2026-08-15T00:00:00Z",
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
    }, raw


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, object], dict[str, bytes]]:
    oe = [
        {"gameid": "oe-1", "date": "2026-08-01T12:00:00Z", "teams": ["Alpha Old", "Beta"], "result": "blue"},
        {"gameid": "oe-2", "date": "2026-08-02T12:00:00Z", "teams": ["Alpha Old", "Beta"], "result": "red"},
        {"gameid": "oe-3", "date": "2026-08-03T12:00:00Z", "teams": ["Alpha Old", "Gamma"], "result": "blue"},
        {"gameid": "oe-singleton", "date": "2026-08-04T12:00:00Z", "teams": ["Solo Old", "Delta"], "result": "blue"},
        {"gameid": "oe-ambiguous-pair", "date": "2026-08-05T12:00:00Z", "teams": ["X Old", "Y Old"], "result": "blue"},
    ]
    scoreboard = [
        {"GameId": "lp-1", "DateTime UTC": "2026-08-01 12:01:00", "Team1": "Alpha Prime", "Team2": "Beta", "Winner": 1},
        {"GameId": "lp-2", "DateTime UTC": "2026-08-02 11:59:00", "Team1": "Beta", "Team2": "Alpha Prime", "Winner": 2},
        {"GameId": "lp-3", "DateTime UTC": "2026-08-03 12:02:00", "Team1": "Gamma", "Team2": "Alpha Prime", "Winner": 1},
        {"GameId": "lp-singleton", "DateTime UTC": "2026-08-04 12:00:00", "Team1": "Delta", "Team2": "Solo Prime", "Winner": 2},
        {"GameId": "lp-ambiguous-pair", "DateTime UTC": "2026-08-05 12:00:00", "Team1": "X New", "Team2": "Y New", "Winner": 1},
    ]
    oe_record, oe_raw = _record("oe", oe)
    scoreboard_record, scoreboard_raw = _record("scoreboardgames", scoreboard)
    return oe, scoreboard, oe_record, scoreboard_record, {"oe": oe_raw, "scoreboardgames": scoreboard_raw}


def _derive(**kwargs):
    oe, scoreboard, oe_record, scoreboard_record, raw = _fixture()
    return derive_team_alias_mapping(
        oe,
        scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=scoreboard_record,
        raw_source_bytes=raw,
        captured_at="2026-08-15T00:00:00Z",
        **kwargs,
    )


def _schedule_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, object],
    dict[str, bytes],
]:
    oe = [
        {"gameid": "oe-schedule-1", "date": "2026-08-10T12:00:00Z", "teams": ["Alpha Old", "Beta"]},
        {"gameid": "oe-schedule-2", "date": "2026-08-11T12:00:00Z", "teams": ["Alpha Old", "Beta"]},
    ]
    scoreboard = [
        {"GameId": "lp-match-1_1", "DateTime UTC": "2026-08-10 12:00:30", "Team1": "Alpha Prime", "Team2": "Beta"},
        {"GameId": "lp-match-2_1", "DateTime UTC": "2026-08-11 12:00:30", "Team1": "Alpha Prime", "Team2": "Beta"},
    ]
    schedule = [
        {"MatchId": "lp-match-1", "Team1": "Alpha Canonical", "Team2": "Beta"},
        {"MatchId": "lp-match-2", "Team1": "Alpha Canonical", "Team2": "Beta"},
    ]
    oe_record, oe_raw = _record("oe", oe)
    scoreboard_record, scoreboard_raw = _record("scoreboardgames", scoreboard)
    schedule_record, schedule_raw = _record("matchschedule", schedule)
    return (
        oe,
        scoreboard,
        schedule,
        oe_record,
        scoreboard_record,
        schedule_record,
        {"oe": oe_raw, "scoreboardgames": scoreboard_raw, "matchschedule": schedule_raw},
    )


def _stable_fixture() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    dict[str, object],
    dict[str, bytes],
]:
    oe = [
        {
            "gameid": "oe-stable-1",
            "date": "2026-08-20T12:00:00Z",
            "teams": ["Alpha Old", "Beta"],
            "team_keys": ["oe-alpha", "oe-beta"],
        },
        {
            "gameid": "oe-stable-2",
            "date": "2026-08-21T12:00:00Z",
            "teams": ["Alpha Rebrand", "Beta"],
            "team_keys": ["oe-alpha", "oe-beta"],
        },
        {
            "gameid": "oe-stable-3",
            "date": "2026-08-22T12:00:00Z",
            "teams": ["Alpha Old", "Beta"],
            "team_keys": ["oe-alpha", "oe-beta"],
        },
    ]
    scoreboard = [
        {"GameId": "lp-stable-1", "DateTime UTC": "2026-08-20 12:00:30", "Team1": "Alpha Prime", "Team2": "Beta"},
        {"GameId": "lp-stable-2", "DateTime UTC": "2026-08-21 12:00:30", "Team1": "Alpha Prime", "Team2": "Beta"},
        {"GameId": "lp-stable-3", "DateTime UTC": "2026-08-22 12:00:30", "Team1": "Alpha Prime", "Team2": "Beta"},
    ]
    oe_record, oe_raw = _record("oe", oe)
    scoreboard_record, scoreboard_raw = _record("scoreboardgames", scoreboard)
    return oe, scoreboard, oe_record, scoreboard_record, {"oe": oe_raw, "scoreboardgames": scoreboard_raw}


def test_repeated_timestamp_evidence_is_accepted_and_singletons_stay_review_only() -> None:
    result = _derive()

    accepted = {(row["oe_team"], row["leaguepedia_team"]): row for row in result["mapping"]}
    assert accepted[("Alpha Old", "Alpha Prime")]["evidence_count"] == 3
    assert all(row["status"] == "accepted" for row in result["mapping"])
    singleton = {(row["oe_team"], row["leaguepedia_team"]): row for row in result["review_only"]}
    assert singleton[("Solo Old", "Solo Prime")]["reason"] == "singleton_or_insufficient_repeated_evidence"
    assert result["status"] == "review_required"
    assert result["derivation_contract"]["outcome_used"] is False
    verify_alias_mapping(result)


def test_no_shared_name_anchor_is_blocked_without_row_order_inference() -> None:
    result = _derive()

    assert any(issue["kind"] == "team_pair_ambiguous" for issue in result["issues"])
    assert not any(row["oe_team"] in {"X Old", "Y Old"} for row in result["mapping"])
    assert result["audit"]["outcome_used"] is False


def test_timestamp_ambiguity_is_rejected() -> None:
    oe, scoreboard, oe_record, scoreboard_record, raw = _fixture()
    scoreboard.append(dict(scoreboard[0], GameId="lp-close", **{"DateTime UTC": "2026-08-01 12:02:00"}))
    scoreboard_record, raw_scoreboard = _record("scoreboardgames", scoreboard)
    result = derive_team_alias_mapping(
        oe,
        scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=scoreboard_record,
        raw_source_bytes={"oe": raw["oe"], "scoreboardgames": raw_scoreboard},
        captured_at="2026-08-15T00:00:00Z",
    )
    assert any(issue["kind"] == "timestamp_ambiguous" and issue["oe_game_id"] == "oe-1" for issue in result["issues"])
    alpha = next(row for row in result["mapping"] if row["oe_team"] == "Alpha Old")
    assert "oe-1" not in alpha["oe_game_ids"]


def test_source_and_target_one_to_one_conflicts_are_blocked() -> None:
    oe, scoreboard, oe_record, scoreboard_record, raw = _fixture()
    oe.append({"gameid": "oe-conflict", "date": "2026-08-06T12:00:00Z", "teams": ["Alpha Old", "Epsilon"]})
    scoreboard.append({"GameId": "lp-conflict", "DateTime UTC": "2026-08-06 12:00:00", "Team1": "Alpha Prime Two", "Team2": "Epsilon"})
    oe_record, oe_raw = _record("oe", oe)
    scoreboard_record, scoreboard_raw = _record("scoreboardgames", scoreboard)
    result = derive_team_alias_mapping(
        oe, scoreboard,
        oe_source_record=oe_record, scoreboard_source_record=scoreboard_record,
        raw_source_bytes={"oe": oe_raw, "scoreboardgames": scoreboard_raw},
        captured_at="2026-08-15T00:00:00Z",
    )
    assert any(conflict["kind"] == "source_alias_conflict" for conflict in result["conflicts"])
    assert not any(row["leaguepedia_team"] == "Alpha Prime Two" for row in result["mapping"])


def test_hash_mutations_fail_closed() -> None:
    oe, scoreboard, oe_record, scoreboard_record, raw = _fixture()
    oe_record["sha256"] = "0" * 64
    with pytest.raises(AliasDerivationError, match="raw source hash"):
        derive_team_alias_mapping(
            oe, scoreboard,
            oe_source_record=oe_record, scoreboard_source_record=scoreboard_record,
            raw_source_bytes=raw,
        )
    oe, scoreboard, oe_record, scoreboard_record, raw = _fixture()
    oe[0]["teams"] = ["forged", "Beta"]
    with pytest.raises(AliasDerivationError, match="payload hash"):
        derive_team_alias_mapping(
            oe, scoreboard,
            oe_source_record=oe_record, scoreboard_source_record=scoreboard_record,
            raw_source_bytes=raw,
        )


def test_single_evidence_threshold_is_rejected() -> None:
    with pytest.raises(AliasDerivationError, match="at least two"):
        _derive(minimum_repeated_evidence=1)


def test_stable_oe_team_keys_accept_display_rebrands_and_bind_digest(tmp_path: Path) -> None:
    oe, scoreboard, oe_record, scoreboard_record, raw = _stable_fixture()
    result = derive_team_alias_mapping(
        oe,
        scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=scoreboard_record,
        raw_source_bytes=raw,
        captured_at="2026-08-15T00:00:00Z",
    )
    alpha = next(row for row in result["mapping"] if row["oe_team_key"] == "oe-alpha")
    assert alpha["allowed_source_names"] == ["Alpha Old", "Alpha Rebrand"]
    assert alpha["stable_oe_team_keys"] == ["oe-alpha"]
    binding = result["input_bindings"]["oe"]["stable_team_key_binding"]
    assert binding["row_count"] == len(oe)
    assert result["stable_oe_team_key_binding"] == binding
    assert result["coverage"]["accepted_stable_team_key_coverage"] == 1.0
    verify_alias_mapping(result)

    artifact = tmp_path / "stable-aliases.json"
    artifact.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    loaded = load_verified_alias_mapping(
        artifact,
        expected_oe_payload_sha256=result["input_bindings"]["oe"]["payload_sha256"],
        expected_scoreboard_payload_sha256=result["input_bindings"]["scoreboardgames"]["payload_sha256"],
        expected_stable_team_key_rows_sha256=binding["rows_sha256"],
    )
    assert loaded["source_to_allowed_targets"]["ScoreboardGames"]["oe-alpha"] == ["Alpha Prime"]
    with pytest.raises(AliasDerivationError, match="stable OE team-key identity"):
        load_verified_alias_mapping(
            artifact,
            expected_oe_payload_sha256=result["input_bindings"]["oe"]["payload_sha256"],
            expected_scoreboard_payload_sha256=result["input_bindings"]["scoreboardgames"]["payload_sha256"],
            expected_stable_team_key_rows_sha256="f" * 64,
        )


def test_display_name_spanning_stable_ids_is_rejected() -> None:
    oe, scoreboard, oe_record, scoreboard_record, raw = _stable_fixture()
    oe[0]["teams"] = ["Alpha Old", "Beta"]
    oe[1]["teams"] = ["Alpha Old", "Beta"]
    oe[1]["team_keys"] = ["oe-alpha-2", "oe-beta"]
    oe[2]["teams"] = ["Alpha Old", "Beta"]
    oe_record, oe_raw = _record("oe", oe)
    result = derive_team_alias_mapping(
        oe,
        scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=scoreboard_record,
        raw_source_bytes={"oe": oe_raw, "scoreboardgames": raw["scoreboardgames"]},
        captured_at="2026-08-15T00:00:00Z",
    )
    assert any(item["kind"] == "oe_display_name_stable_key_conflict" for item in result["issues"])
    assert not any(row["oe_team"] == "Alpha Old" for row in result["mapping"])
    assert result["coverage"]["stable_display_name_conflict_count"] == 1


def test_multiple_stable_ids_converging_on_one_target_are_blocked() -> None:
    oe, scoreboard, oe_record, scoreboard_record, raw = _stable_fixture()
    oe[0]["teams"] = ["Alpha First", "Beta"]
    oe[0]["team_keys"] = ["oe-alpha-one", "oe-beta"]
    oe[1]["team_keys"] = ["oe-alpha-two", "oe-beta"]
    oe[2]["team_keys"] = ["oe-alpha-two", "oe-beta"]
    oe_record, oe_raw = _record("oe", oe)
    result = derive_team_alias_mapping(
        oe,
        scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=scoreboard_record,
        raw_source_bytes={"oe": oe_raw, "scoreboardgames": raw["scoreboardgames"]},
        captured_at="2026-08-15T00:00:00Z",
    )
    assert any(item["kind"] == "target_alias_conflict" for item in result["conflicts"])
    assert not any(row["leaguepedia_team"] == "Alpha Prime" for row in result["mapping"])


def test_partial_or_misaligned_stable_team_keys_fail_closed() -> None:
    oe, scoreboard, oe_record, scoreboard_record, raw = _stable_fixture()
    oe[1].pop("team_keys")
    oe_record, oe_raw = _record("oe", oe)
    with pytest.raises(AliasDerivationError, match="present for every row"):
        derive_team_alias_mapping(
            oe,
            scoreboard,
            oe_source_record=oe_record,
            scoreboard_source_record=scoreboard_record,
            raw_source_bytes={"oe": oe_raw, "scoreboardgames": raw["scoreboardgames"]},
        )

    oe, scoreboard, oe_record, scoreboard_record, raw = _stable_fixture()
    oe[0]["team_keys"] = ["oe-alpha"]
    oe_record, oe_raw = _record("oe", oe)
    with pytest.raises(AliasDerivationError, match="pair exactly"):
        derive_team_alias_mapping(
            oe,
            scoreboard,
            oe_source_record=oe_record,
            scoreboard_source_record=scoreboard_record,
            raw_source_bytes={"oe": oe_raw, "scoreboardgames": raw["scoreboardgames"]},
        )


def test_unique_game_prefix_bridges_scoreboard_to_schedule_and_is_transitive(tmp_path: Path) -> None:
    oe, scoreboard, schedule, oe_record, scoreboard_record, schedule_record, raw = _schedule_fixture()
    result = derive_team_alias_mapping(
        oe,
        scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=scoreboard_record,
        schedule_rows=schedule,
        schedule_source_record=schedule_record,
        raw_source_bytes=raw,
        captured_at="2026-08-15T00:00:00Z",
    )
    direct = {(row["oe_team"], row["leaguepedia_team"]): row for row in result["scoreboard_schedule_mapping"]}
    transitive = {(row["oe_team"], row["leaguepedia_team"]): row for row in result["oe_to_matchschedule_mapping"]}
    assert direct[("Alpha Prime", "Alpha Canonical")]["evidence_count"] == 2
    assert transitive[("Alpha Old", "Alpha Canonical")]["evidence_count"] == 2
    assert result["coverage"]["matchschedule_prefix_matched_rows"] == 2
    assert result["coverage"]["oe_matchschedule_alias_count"] >= 2
    assert result["input_bindings"]["matchschedule"]["integrity_verified"] is True
    assert any(item["target_system"] == "MatchSchedule" for item in result["canonical_aliases"])
    verify_alias_mapping(result)
    artifact_path = tmp_path / "aliases.json"
    artifact_path.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    loaded = load_verified_alias_mapping(
        artifact_path,
        expected_oe_payload_sha256=result["input_bindings"]["oe"]["payload_sha256"],
        expected_scoreboard_payload_sha256=result["input_bindings"]["scoreboardgames"]["payload_sha256"],
        expected_matchschedule_payload_sha256=result["input_bindings"]["matchschedule"]["payload_sha256"],
    )
    assert loaded["source_to_allowed_targets"]["MatchSchedule"]["alpha prime"] == ["Alpha Canonical"]


def test_schedule_prefix_duplicates_and_missing_rows_are_reviewed() -> None:
    oe, scoreboard, schedule, oe_record, scoreboard_record, schedule_record, raw = _schedule_fixture()
    duplicate_schedule = schedule + [dict(schedule[0], Team1="Forged Duplicate")]
    duplicate_record, duplicate_raw = _record("matchschedule", duplicate_schedule)
    duplicate = derive_team_alias_mapping(
        oe,
        scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=scoreboard_record,
        schedule_rows=duplicate_schedule,
        schedule_source_record=duplicate_record,
        raw_source_bytes={**raw, "matchschedule": duplicate_raw},
        captured_at="2026-08-15T00:00:00Z",
    )
    assert any(item["kind"] == "invalid_matchschedule_row" for item in duplicate["issues"])
    assert duplicate["status"] == "review_required"

    missing_scoreboard = [dict(scoreboard[0], GameId="lp-missing_1"), scoreboard[1]]
    missing_record, missing_raw = _record("scoreboardgames", missing_scoreboard)
    missing = derive_team_alias_mapping(
        oe,
        missing_scoreboard,
        oe_source_record=oe_record,
        scoreboard_source_record=missing_record,
        schedule_rows=schedule,
        schedule_source_record=schedule_record,
        raw_source_bytes={**raw, "scoreboardgames": missing_raw},
        captured_at="2026-08-15T00:00:00Z",
    )
    assert any(item["kind"] == "matchschedule_prefix_missing" for item in missing["issues"])
    assert missing["coverage"]["matchschedule_prefix_unmatched_rows"] == 1


def test_verified_alias_loader_requires_expected_source_payloads(tmp_path: Path) -> None:
    result = _derive()
    artifact = tmp_path / "aliases.json"
    artifact.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    with pytest.raises(AliasDerivationError, match="expected oe payload hash"):
        load_verified_alias_mapping(artifact, allow_review_only=True)
    loaded = load_verified_alias_mapping(
        artifact,
        expected_oe_payload_sha256=result["input_bindings"]["oe"]["payload_sha256"],
        expected_scoreboard_payload_sha256=result["input_bindings"]["scoreboardgames"]["payload_sha256"],
        allow_review_only=True,
    )
    assert loaded["mapping_sha256"] == result["mapping_sha256"]
    assert loaded["source_to_allowed_targets"]["ScoreboardGames"]["alpha old"] == ["Alpha Prime"]
    with pytest.raises(AliasDerivationError, match="scoreboardgames source payload identity"):
        load_verified_alias_mapping(
            artifact,
            expected_oe_payload_sha256=result["input_bindings"]["oe"]["payload_sha256"],
            expected_scoreboard_payload_sha256="f" * 64,
            allow_review_only=True,
        )


def test_cli_reads_local_files_and_returns_review_status(tmp_path: Path) -> None:
    oe, scoreboard, oe_record, scoreboard_record, _raw = _fixture()
    oe_path = tmp_path / "oe.json"
    scoreboard_path = tmp_path / "scoreboard.json"
    oe_path.write_bytes(_canonical(oe))
    scoreboard_path.write_bytes(_canonical(scoreboard))
    # The CLI reads exact files, so records must be rebuilt for their bytes.
    oe_record, _ = _record("oe", oe)
    scoreboard_record, _ = _record("scoreboardgames", scoreboard)
    oe_record["sha256"] = hashlib.sha256(oe_path.read_bytes()).hexdigest()
    oe_record["bytes"] = oe_path.stat().st_size
    scoreboard_record["sha256"] = hashlib.sha256(scoreboard_path.read_bytes()).hexdigest()
    scoreboard_record["bytes"] = scoreboard_path.stat().st_size
    oe_record_path = tmp_path / "oe-record.json"
    scoreboard_record_path = tmp_path / "scoreboard-record.json"
    oe_record_path.write_text(json.dumps(oe_record))
    scoreboard_record_path.write_text(json.dumps(scoreboard_record))
    output = tmp_path / "aliases.json"
    assert derive_cli([
        "--oe", str(oe_path), "--scoreboardgames", str(scoreboard_path),
        "--oe-record", str(oe_record_path), "--scoreboard-record", str(scoreboard_record_path),
        "--output", str(output), "--allow-review-only",
    ]) == 0
    assert json.loads(output.read_text())["status"] == "review_required"


def test_cli_accepts_matchschedule_capture_and_binds_its_hash(tmp_path: Path) -> None:
    oe, scoreboard, schedule, oe_record, scoreboard_record, schedule_record, _raw = _schedule_fixture()
    paths = {
        "oe": tmp_path / "oe.json",
        "scoreboardgames": tmp_path / "scoreboard.json",
        "matchschedule": tmp_path / "schedule.json",
    }
    for label, rows in (("oe", oe), ("scoreboardgames", scoreboard), ("matchschedule", schedule)):
        paths[label].write_bytes(_canonical(rows))
    records = {
        "oe": oe_record,
        "scoreboardgames": scoreboard_record,
        "matchschedule": schedule_record,
    }
    record_paths: dict[str, Path] = {}
    for label, record in records.items():
        record["sha256"] = hashlib.sha256(paths[label].read_bytes()).hexdigest()
        record["bytes"] = paths[label].stat().st_size
        record_path = tmp_path / f"{label}-record.json"
        record_path.write_text(json.dumps(record), encoding="utf-8")
        record_paths[label] = record_path
    output = tmp_path / "aliases.json"
    assert derive_cli([
        "--oe", str(paths["oe"]),
        "--scoreboardgames", str(paths["scoreboardgames"]),
        "--matchschedule", str(paths["matchschedule"]),
        "--oe-record", str(record_paths["oe"]),
        "--scoreboard-record", str(record_paths["scoreboardgames"]),
        "--matchschedule-record", str(record_paths["matchschedule"]),
        "--output", str(output),
    ]) == 0
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["input_bindings"]["matchschedule"]["sha256"] == records["matchschedule"]["sha256"]
