from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.research.oe_leaguepedia_series_crosswalk import (
    CrosswalkError,
    build_oe_leaguepedia_series_crosswalk,
    verify_crosswalk,
)
from tools.build_oe_leaguepedia_series_crosswalk import (
    _safe_capture_path,
    main as build_crosswalk_cli,
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()


def _receipt(ids: list[str]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-08-15T00:00:00Z",
        "source_game_count": len(ids),
        "source_identity_sha256": hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest(),
        "accepted_game_ids": sorted(ids),
        "authority": {"research_only": True, "public": False},
    }
    payload["receipt_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    return payload


def _fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], dict[str, object], dict[str, dict[str, object]], dict[str, bytes]]:
    oe = [
        {"gameid": "oe-1", "date": "2026-08-14T15:07:55Z", "league": "LEC", "patch": "16.16", "teams": ["G2", "Fnatic"], "result": "blue"},
        {"gameid": "oe-2", "date": "2026-08-14T15:54:54Z", "league": "LEC", "patch": "16.16", "teams": ["Fnatic", "G2 Esports"], "result": "red"},
    ]
    scoreboard = [
        {"GameId": "match-1_1", "DateTime UTC": "2026-08-14 15:07:55", "Team1": "G2 Esports", "Team2": "Fnatic", "Patch": "26.16", "OverviewPage": "LEC/2026 Summer", "Tournament": "LEC 2026 Summer", "Winner": "G2"},
        {"GameId": "match-1_2", "DateTime UTC": "2026-08-14 15:54:54", "Team1": "Fnatic", "Team2": "G2 Esports", "Patch": "26.16", "OverviewPage": "LEC/2026 Summer", "Tournament": "LEC 2026 Summer", "Winner": "Fnatic"},
    ]
    schedule = [
        {"MatchId": "match-1", "DateTime UTC": "2026-08-14 15:00:00", "Team1": "G2 Esports", "Team2": "Fnatic", "Patch": "26.16", "OverviewPage": "LEC/2026 Summer", "Winner": "G2"},
    ]
    tournaments = [
        {"Name": "LEC 2026 Summer", "OverviewPage": "LEC/2026 Summer", "League": "LEC"},
    ]
    receipt = _receipt(["oe-1", "oe-2"])
    mapping = {
        "LEC": {
            "scoreboard": {"overview_pages": ["LEC/2026 Summer"]},
            "schedule": {"overview_pages": ["LEC/2026 Summer"]},
            "patches": {"16.16": ["26.16"]},
        }
    }
    payloads = {
        "oe": oe,
        "scoreboardgames": scoreboard,
        "matchschedule": schedule,
        "tournaments": tournaments,
    }
    records: dict[str, dict[str, object]] = {}
    raw: dict[str, bytes] = {}
    for label, rows in payloads.items():
        content = _canonical(rows)
        raw_bytes = b"downloaded-" + label.encode() + b"\n" + content
        raw[label] = raw_bytes
        records[label] = {
            "url": f"https://example.test/{label}.json",
            "retrieved_at": "2026-08-15T00:00:00Z",
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "bytes": len(raw_bytes),
            "payload_sha256": hashlib.sha256(content).hexdigest(),
            "payload_bytes": len(content),
        }
    return oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw


def _build(**kwargs):
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    return build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
        **kwargs,
    )


def _write_source_files(
    root: Path,
    payloads: dict[str, list[dict[str, object]]],
    records: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    root.mkdir(parents=True, exist_ok=True)
    for label, rows in payloads.items():
        path = root / f"{label}.json"
        raw = _canonical(rows)
        path.write_bytes(raw)
        records[label].update(
            path=str(path.resolve()),
            sha256=hashlib.sha256(raw).hexdigest(),
            bytes=len(raw),
        )
    return records


def _build_verifiable(tmp_path: Path, **kwargs):
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, _raw = _fixture()
    records = _write_source_files(
        tmp_path / "source-files",
        {
            "oe": oe,
            "scoreboardgames": scoreboard,
            "matchschedule": schedule,
            "tournaments": tournaments,
        },
        records,
    )
    return build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        **kwargs,
    )


def _refresh_records(
    records: dict[str, dict[str, object]], raw: dict[str, bytes], payloads: dict[str, list[dict[str, object]]]
) -> tuple[dict[str, dict[str, object]], dict[str, bytes]]:
    for label, rows in payloads.items():
        content = _canonical(rows)
        raw_bytes = b"downloaded-" + label.encode() + b"\n" + content
        raw[label] = raw_bytes
        records[label].update(
            sha256=hashlib.sha256(raw_bytes).hexdigest(),
            bytes=len(raw_bytes),
            payload_sha256=hashlib.sha256(content).hexdigest(),
            payload_bytes=len(content),
        )
    return records, raw


def test_complete_crosswalk_binds_exact_prefix_and_keeps_outcomes_out(
    tmp_path: Path,
) -> None:
    result = _build_verifiable(tmp_path)

    assert result["status"] == "complete_authoritative_coverage"
    assert result["coverage"]["mapped_game_count"] == 2
    assert [row["series_id"] for row in result["assignments"]] == ["match-1", "match-1"]
    assert [row["scoreboard_tournament"] for row in result["assignments"]] == [
        "LEC 2026 Summer",
        "LEC 2026 Summer",
    ]
    assert result["series"][0]["scoreboard_tournament"] == "LEC 2026 Summer"
    assert result["join_contract"]["tournament_binding"]["source"] == "ScoreboardGames.Tournament"
    assert result["assignment_sha256"]
    assert [row["scoreboard_game_order"] for row in result["assignments"]] == [1, 2]
    assert all(row["outcome_used"] is False for row in result["assignments"])
    assert result["source_records"]["oe"]["integrity_verified"] is True
    assert "result" not in result["raw_sources"]["oe"][0]
    assert "Winner" not in result["raw_sources"]["scoreboardgames"][0]
    assert "Winner" not in result["raw_sources"]["matchschedule"][0]
    assert (
        result["source_records"]["matchschedule"]["source_payload_sha256"]
        != result["source_records"]["matchschedule"]["payload_sha256"]
    )
    verify_crosswalk(result)


def test_direct_riot_platform_identity_survives_team_and_timestamp_drift(
    tmp_path: Path,
) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    for index, row in enumerate(scoreboard, start=1):
        row["RiotPlatformGameId"] = f"oe-{index}"
        row["DateTime UTC"] = f"2026-07-0{index} 01:00:00"
        row["Team1"] = f"Renamed Alpha {index}"
        row["Team2"] = f"Renamed Beta {index}"
    schedule[0]["DateTime UTC"] = "2026-07-01 00:00:00"
    schedule[0]["Team1"] = "Renamed Alpha"
    schedule[0]["Team2"] = "Renamed Beta"
    payloads = {
        "oe": oe,
        "scoreboardgames": scoreboard,
        "matchschedule": schedule,
        "tournaments": tournaments,
    }
    records, _ = _refresh_records(records, raw, payloads)
    records = _write_source_files(tmp_path / "direct-source-files", payloads, records)

    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
    )

    assert result["status"] == "complete_authoritative_coverage"
    assert [row["scoreboard_riot_platform_game_id"] for row in result["assignments"]] == [
        "oe-1",
        "oe-2",
    ]
    assert {
        row["assignment_method"] for row in result["assignments"]
    } == {"exact_riot_platform_game_id_then_exact_game_id_prefix"}
    assert all(
        row["evidence"]["schedule"]["timestamp_bound_used_for_identity"] is False
        for row in result["assignments"]
    )
    verify_crosswalk(result)

    result["assignments"][0]["scoreboard_riot_platform_game_id"] = "forged"
    result["assignment_sha256"] = hashlib.sha256(
        _canonical(
            sorted(result["assignments"], key=lambda row: str(row["oe_game_id"]))
        )
    ).hexdigest()
    result.pop("crosswalk_sha256")
    result["crosswalk_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    with pytest.raises(CrosswalkError, match="game identity changed"):
        verify_crosswalk(result)


def test_direct_riot_platform_identity_keeps_patch_mismatch_as_diagnostic(
    tmp_path: Path,
) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    for index, row in enumerate(scoreboard, start=1):
        row["RiotPlatformGameId"] = f"oe-{index}"
        row["Patch"] = "26.15"
    schedule[0]["Patch"] = "26.15"
    payloads = {
        "oe": oe,
        "scoreboardgames": scoreboard,
        "matchschedule": schedule,
        "tournaments": tournaments,
    }
    records, raw = _refresh_records(records, raw, payloads)
    records = _write_source_files(tmp_path / "direct-patch-source-files", payloads, records)

    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
    )

    assert result["status"] == "complete_authoritative_coverage"
    assert all(
        row["evidence"]["patch"]["matched"] is False
        and row["evidence"]["patch"]["identity_gate_enforced"] is False
        and row["evidence"]["schedule"]["patch"]["matched"] is False
        and row["evidence"]["schedule"]["patch"]["identity_gate_enforced"] is False
        for row in result["assignments"]
    )
    verify_crosswalk(result)


def test_duplicate_direct_riot_id_selects_only_unique_schedule_prefix(
    tmp_path: Path,
) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard[0]["RiotPlatformGameId"] = "oe-1"
    scoreboard[1]["RiotPlatformGameId"] = "oe-2"
    duplicate = dict(scoreboard[0])
    duplicate["GameId"] = "match-other_1"
    scoreboard.append(duplicate)
    payloads = {
        "oe": oe,
        "scoreboardgames": scoreboard,
        "matchschedule": schedule,
        "tournaments": tournaments,
    }
    records, raw = _refresh_records(records, raw, payloads)
    records = _write_source_files(tmp_path / "direct-duplicate-source-files", payloads, records)

    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
    )

    first = next(row for row in result["assignments"] if row["oe_game_id"] == "oe-1")
    assert result["status"] == "complete_authoritative_coverage"
    assert first["scoreboard_game_id"] == "match-1_1"
    assert first["assignment_method"] == (
        "exact_riot_platform_game_id_disambiguated_by_unique_"
        "matchschedule_prefix_then_exact_game_id_prefix"
    )
    assert first["evidence"]["identity_disambiguation"]["eligible_game_id_prefix"] == "match-1"
    verify_crosswalk(result)


def test_duplicate_direct_riot_id_stays_ambiguous_when_each_prefix_has_schedule(
    tmp_path: Path,
) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard[0]["RiotPlatformGameId"] = "oe-1"
    scoreboard[1]["RiotPlatformGameId"] = "oe-2"
    duplicate = dict(scoreboard[0])
    duplicate["GameId"] = "match-other_1"
    scoreboard.append(duplicate)
    schedule.append(dict(schedule[0], MatchId="match-other"))
    payloads = {
        "oe": oe,
        "scoreboardgames": scoreboard,
        "matchschedule": schedule,
        "tournaments": tournaments,
    }
    records, raw = _refresh_records(records, raw, payloads)
    records = _write_source_files(tmp_path / "direct-ambiguous-source-files", payloads, records)

    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        allow_partial=True,
    )

    assert result["status"] == "partial_authoritative_coverage"
    assert [row["oe_game_id"] for row in result["assignments"]] == ["oe-2"]
    assert any(
        issue["kind"] == "riot_platform_identity_ambiguous"
        and issue["eligible_candidate_count"] == 2
        for issue in result["issues"]
    )
    verify_crosswalk(result)


def test_resealed_embedded_outcome_field_fails_verification(tmp_path: Path) -> None:
    result = _build_verifiable(tmp_path)
    result["raw_sources"]["matchschedule"][0]["Winner"] = "forged"
    projected = _canonical(result["raw_sources"]["matchschedule"])
    result["source_records"]["matchschedule"]["payload_sha256"] = hashlib.sha256(
        projected
    ).hexdigest()
    result["source_records"]["matchschedule"]["payload_bytes"] = len(projected)
    result.pop("crosswalk_sha256")
    result["crosswalk_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()

    with pytest.raises(CrosswalkError, match="contains outcome fields"):
        verify_crosswalk(result)


def test_missing_scoreboard_tournament_is_unmapped_and_fail_closed() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    del scoreboard[0]["Tournament"]
    records, raw = _refresh_records(
        records, raw, {"scoreboardgames": scoreboard}
    )
    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
    )
    assert result["status"] == "rejected_incomplete"
    assert [row["oe_game_id"] for row in result["assignments"]] == ["oe-2"]
    assert any(issue["kind"] == "scoreboard_tournament_missing" for issue in result["issues"])

    partial = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
        allow_partial=True,
    )
    assert partial["status"] == "partial_authoritative_coverage"
    assert all(row["scoreboard_tournament"] for row in partial["assignments"])


def test_conflicting_scoreboard_tournaments_reject_the_series() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard[1]["Tournament"] = "LEC 2026 Playoffs"
    tournaments.append(
        {
            "Name": "LEC 2026 Playoffs",
            "OverviewPage": "LEC/2026 Summer",
            "League": "LEC",
        }
    )
    records, raw = _refresh_records(
        records, raw, {"scoreboardgames": scoreboard, "tournaments": tournaments}
    )
    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
        allow_partial=True,
    )
    assert result["status"] == "partial_authoritative_coverage"
    assert result["assignments"] == []
    conflict = [
        issue for issue in result["issues"] if issue["kind"] == "series_tournament_conflict"
    ]
    assert len(conflict) == 1
    assert conflict[0]["scoreboard_tournaments"] == [
        "LEC 2026 Playoffs",
        "LEC 2026 Summer",
    ]


def test_resealed_wrong_scoreboard_tournament_stays_unmapped() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard[0]["Tournament"] = "Forged Tournament"
    records, raw = _refresh_records(records, raw, {"scoreboardgames": scoreboard})
    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
        allow_partial=True,
    )
    assert [row["oe_game_id"] for row in result["assignments"]] == ["oe-2"]
    assert any(
        issue["kind"] == "tournament_identity_missing"
        and issue["scoreboard_tournament"] == "Forged Tournament"
        for issue in result["issues"]
    )


def test_tampered_tournament_overview_fails_competition_binding() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    tournaments[0]["OverviewPage"] = "Forged/Overview"
    records, raw = _refresh_records(records, raw, {"tournaments": tournaments})
    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
        allow_partial=True,
    )
    assert result["assignments"] == []
    assert all(
        issue["kind"] == "tournament_competition_mismatch"
        for issue in result["issues"]
        if issue["kind"].startswith("tournament_")
    )


def test_complete_crosswalk_requires_tournament_binding() -> None:
    result = _build()
    result["join_contract"].pop("tournament_binding")
    result.pop("crosswalk_sha256")
    result["crosswalk_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    with pytest.raises(CrosswalkError, match="tournament binding"):
        verify_crosswalk(result)


def test_legacy_partial_tournament_binding_remains_research_readable() -> None:
    result = _build()
    result["status"] = "partial_authoritative_coverage"
    result["coverage"]["complete"] = False
    result["join_contract"]["tournament_binding"] = {
        "source": "ScoreboardGames.Tournament",
        "assignment_field": "scoreboard_tournament",
        "series_policy": "one_non_empty_value_per_series",
        "conflict_policy": "reject_series",
    }
    result.pop("crosswalk_sha256")
    result["crosswalk_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    verify_crosswalk(result)


def test_tournament_assignment_tamper_changes_hash_and_fails_verification() -> None:
    result = _build()
    result["assignments"][0]["scoreboard_tournament"] = "forged"
    with pytest.raises(CrosswalkError, match="self-hash"):
        verify_crosswalk(result)

    replay = _build()
    replay["assignments"][0]["scoreboard_tournament"] = "forged"
    replay.pop("crosswalk_sha256")
    replay["assignment_sha256"] = hashlib.sha256(
        _canonical(
            sorted(replay["assignments"], key=lambda row: str(row["oe_game_id"]))
        )
    ).hexdigest()
    replay["crosswalk_sha256"] = hashlib.sha256(_canonical(replay)).hexdigest()
    with pytest.raises(CrosswalkError, match="conflicting"):
        verify_crosswalk(replay)


def test_resealed_raw_tournament_overview_mutation_fails_verification(
    tmp_path: Path,
) -> None:
    result = _build_verifiable(tmp_path)
    result["raw_sources"]["tournaments"][0]["OverviewPage"] = "Forged/Overview"
    tournament_payload = _canonical(result["raw_sources"]["tournaments"])
    result["source_records"]["tournaments"]["payload_sha256"] = hashlib.sha256(
        tournament_payload
    ).hexdigest()
    result["source_records"]["tournaments"]["payload_bytes"] = len(
        tournament_payload
    )
    result.pop("crosswalk_sha256")
    result["crosswalk_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()

    with pytest.raises(CrosswalkError, match="projection differs from source file"):
        verify_crosswalk(result)


def test_resealed_non_outcome_projection_field_fails_verification(
    tmp_path: Path,
) -> None:
    result = _build_verifiable(tmp_path)
    result["raw_sources"]["tournaments"][0]["ForgedProjectionField"] = "forged"
    projected = _canonical(result["raw_sources"]["tournaments"])
    result["source_records"]["tournaments"]["payload_sha256"] = hashlib.sha256(
        projected
    ).hexdigest()
    result["source_records"]["tournaments"]["payload_bytes"] = len(projected)
    result.pop("crosswalk_sha256")
    result["crosswalk_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()

    with pytest.raises(CrosswalkError, match="projection differs from source file"):
        verify_crosswalk(result)


def test_invalid_raw_tournament_rows_replay_exact_builder_issues(
    tmp_path: Path,
) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, _raw = _fixture()
    tournaments.append({"Name": "Incomplete Tournament"})
    records, _raw = _refresh_records(records, _raw, {"tournaments": tournaments})
    records = _write_source_files(
        tmp_path / "source-files",
        {
            "oe": oe,
            "scoreboardgames": scoreboard,
            "matchschedule": schedule,
            "tournaments": tournaments,
        },
        records,
    )
    result = build_oe_leaguepedia_series_crosswalk(
        oe,
        scoreboard,
        schedule,
        tournaments,
        source_receipt=receipt,
        source_records=records,
        competition_mapping=mapping,
        captured_at="2026-08-15T00:00:00Z",
        allow_partial=True,
    )

    assert result["status"] == "partial_authoritative_coverage"
    assert len(result["assignments"]) == 2
    assert [
        issue
        for issue in result["issues"]
        if issue["kind"] == "invalid_tournament_row"
    ] == [
        {
            "kind": "invalid_tournament_row",
            "index": 1,
            "reason": "Name, OverviewPage, and League are required",
        }
    ]
    verify_crosswalk(result)

    omitted = json.loads(json.dumps(result))
    omitted["issues"] = [
        issue
        for issue in omitted["issues"]
        if issue["kind"] != "invalid_tournament_row"
    ]
    omitted.pop("crosswalk_sha256")
    omitted["crosswalk_sha256"] = hashlib.sha256(_canonical(omitted)).hexdigest()
    with pytest.raises(CrosswalkError, match="invalid tournament issues differ"):
        verify_crosswalk(omitted)

    added = json.loads(json.dumps(result))
    added["raw_sources"]["tournaments"].append({"League": "LEC"})
    tournament_payload = _canonical(added["raw_sources"]["tournaments"])
    added["source_records"]["tournaments"]["payload_sha256"] = hashlib.sha256(
        tournament_payload
    ).hexdigest()
    added["source_records"]["tournaments"]["payload_bytes"] = len(
        tournament_payload
    )
    added.pop("crosswalk_sha256")
    added["crosswalk_sha256"] = hashlib.sha256(_canonical(added)).hexdigest()
    with pytest.raises(CrosswalkError, match="projection differs from source file"):
        verify_crosswalk(added)


def test_source_replay_rejects_symlink_leaf_and_ancestor(tmp_path: Path) -> None:
    result = _build_verifiable(tmp_path / "baseline")
    target = Path(result["source_records"]["tournaments"]["path"])
    link_root = tmp_path / "links"
    link_root.mkdir()
    leaf = link_root / "tournaments.json"
    ancestor = link_root / "source-dir"
    try:
        leaf.symlink_to(target)
        ancestor.symlink_to(target.parent, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    for unsafe in (leaf, ancestor / target.name):
        mutated = json.loads(json.dumps(result))
        mutated["source_records"]["tournaments"]["path"] = str(unsafe)
        mutated.pop("crosswalk_sha256")
        mutated["crosswalk_sha256"] = hashlib.sha256(_canonical(mutated)).hexdigest()
        with pytest.raises(CrosswalkError, match="contains a symlink"):
            verify_crosswalk(mutated)


def test_safe_capture_path_rejects_symlink_leaf_and_ancestor(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target = target_dir / "source.json"
    target.write_text("{}")
    leaf = root / "leaf.json"
    ancestor = root / "linked-directory"
    linked_root = tmp_path / "linked-root"
    try:
        leaf.symlink_to(target)
        ancestor.symlink_to(target_dir, target_is_directory=True)
        linked_root.symlink_to(root, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(CrosswalkError, match="missing or unsafe"):
        _safe_capture_path(root, "leaf.json", label="leaf")
    with pytest.raises(CrosswalkError, match="missing or unsafe"):
        _safe_capture_path(root, "linked-directory/source.json", label="ancestor")
    with pytest.raises(CrosswalkError, match="missing or unsafe"):
        _safe_capture_path(linked_root, "leaf.json", label="root")


def test_outcome_mutations_do_not_change_assignments() -> None:
    baseline = _build()
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    for row in oe + scoreboard + schedule:
        row["Winner"] = "forged-outcome"
        row["result"] = "forged-outcome"
    records, raw = _refresh_records(records, raw, {"oe": oe, "scoreboardgames": scoreboard, "matchschedule": schedule})
    mutated = build_oe_leaguepedia_series_crosswalk(
        oe, scoreboard, schedule, tournaments,
        source_receipt=receipt, source_records=records,
        competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
    )
    assert [row["oe_game_id"] for row in mutated["assignments"]] == [row["oe_game_id"] for row in baseline["assignments"]]
    assert mutated["join_contract"]["outcome_used"] is False


def test_ambiguous_game_identity_is_rejected() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard.append(dict(scoreboard[0], **{"GameId": "match-other_1"}))
    records, raw = _refresh_records(records, raw, {"oe": oe, "scoreboardgames": scoreboard, "matchschedule": schedule})
    result = build_oe_leaguepedia_series_crosswalk(
        oe, scoreboard, schedule, tournaments,
        source_receipt=receipt, source_records=records,
        competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw, allow_partial=True,
    )
    assert result["status"] == "partial_authoritative_coverage"
    assert any(issue["kind"] == "scoreboard_identity_ambiguous" for issue in result["issues"])
    assert result["coverage"]["complete"] is False


def test_duplicate_ids_and_missing_prefix_are_not_assigned() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard[1]["GameId"] = scoreboard[0]["GameId"]
    schedule[0]["MatchId"] = "wrong-prefix"
    records, raw = _refresh_records(records, raw, {"oe": oe, "scoreboardgames": scoreboard, "matchschedule": schedule})
    result = build_oe_leaguepedia_series_crosswalk(
        oe, scoreboard, schedule, tournaments,
        source_receipt=receipt, source_records=records,
        competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw, allow_partial=True,
    )
    assert result["assignments"] == []
    assert any(issue["kind"] == "invalid_scoreboard_row" for issue in result["issues"])
    assert any(issue["kind"] == "schedule_identity_missing" for issue in result["issues"])


def test_patch_mismatch_is_unmatched() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard[0]["Patch"] = "26.15"
    records, raw = _refresh_records(records, raw, {"oe": oe, "scoreboardgames": scoreboard, "matchschedule": schedule})
    result = build_oe_leaguepedia_series_crosswalk(
        oe, scoreboard, schedule, tournaments,
        source_receipt=receipt, source_records=records,
        competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw, allow_partial=True,
    )
    assert result["coverage"]["mapped_game_count"] == 1
    assert any(issue["kind"] == "scoreboard_identity_missing" for issue in result["issues"])


def test_partial_coverage_is_explicit_and_never_full_census() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    oe.append({"gameid": "oe-unmatched", "date": "2026-08-14T16:40:00Z", "league": "LEC", "patch": "16.16", "teams": ["G2", "Fnatic"]})
    # The selected row remains within the accepted receipt only after the
    # receipt is rebuilt.  This models a selected subset of a larger census.
    receipt = _receipt(["oe-1", "oe-2", "oe-unmatched"])
    records, raw = _refresh_records(records, raw, {"oe": oe, "scoreboardgames": scoreboard, "matchschedule": schedule})
    result = build_oe_leaguepedia_series_crosswalk(
        oe, scoreboard, schedule, tournaments,
        source_receipt=receipt, source_records=records,
        competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw, allow_partial=True,
    )
    assert result["status"] == "partial_authoritative_coverage"
    assert result["coverage"]["authority"] == "research_only_authoritative_for_mapped_rows"
    assert result["coverage"]["mapped_is_full_accepted_census"] is False


def test_hash_and_receipt_tampering_fails_closed() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    records["oe"]["sha256"] = "0" * 64
    with pytest.raises(CrosswalkError, match="source file hash"):
        build_oe_leaguepedia_series_crosswalk(
            oe, scoreboard, schedule, tournaments,
            source_receipt=receipt, source_records=records,
            competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
            raw_source_bytes=raw,
        )
    receipt["source_game_count"] = 999
    with pytest.raises(CrosswalkError, match="source receipt hash"):
        build_oe_leaguepedia_series_crosswalk(
            oe, scoreboard, schedule, tournaments,
            source_receipt=receipt, source_records=records,
            competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
            raw_source_bytes=raw,
        )


def test_unmatched_rows_fail_closed_without_partial_opt_in() -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, raw = _fixture()
    scoreboard[0]["Team1"] = "Unknown Team"
    records, raw = _refresh_records(records, raw, {"oe": oe, "scoreboardgames": scoreboard, "matchschedule": schedule})
    result = build_oe_leaguepedia_series_crosswalk(
        oe, scoreboard, schedule, tournaments,
        source_receipt=receipt, source_records=records,
        competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
        raw_source_bytes=raw,
    )
    assert result["status"] == "rejected_incomplete"
    assert result["assignments"]


def test_source_path_can_verify_downloaded_file(tmp_path: Path) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, records, _raw = _fixture()
    paths: dict[str, Path] = {}
    for label, rows in (("oe", oe), ("scoreboardgames", scoreboard), ("matchschedule", schedule), ("tournaments", tournaments)):
        path = tmp_path / f"{label}.json"
        raw = _canonical(rows)
        path.write_bytes(raw)
        records[label]["path"] = str(path)
        records[label]["url"] = f"https://example.test/{label}.json"
        records[label]["sha256"] = hashlib.sha256(raw).hexdigest()
        records[label]["bytes"] = len(raw)
        paths[label] = path
    result = build_oe_leaguepedia_series_crosswalk(
        oe, scoreboard, schedule, tournaments,
        source_receipt=receipt, source_records=records,
        competition_mapping=mapping, captured_at="2026-08-15T00:00:00Z",
    )
    assert result["status"] == "complete_authoritative_coverage"


def test_cli_reads_downloaded_json_and_never_needs_network(tmp_path: Path) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, _records, _raw = _fixture()
    paths: dict[str, Path] = {}
    for label, rows in (("oe", oe), ("scoreboardgames", scoreboard), ("matchschedule", schedule), ("tournaments", tournaments)):
        path = tmp_path / f"{label}.json"
        path.write_bytes(_canonical(rows))
        paths[label] = path
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(json.dumps(receipt, sort_keys=True).encode())
    map_path = tmp_path / "competition-map.json"
    map_path.write_bytes(json.dumps(mapping, sort_keys=True).encode())
    manifest_path = tmp_path / "capture-manifest.json"
    manifest_path.write_bytes(
        json.dumps(
            {
                "captured_at": "2026-08-15T00:00:00Z",
                "sources": {
                    label: {"url": f"https://example.test/{label}.json", "retrieved_at": "2026-08-15T00:00:00Z"}
                    for label in ("oe", "scoreboardgames", "matchschedule", "tournaments")
                },
            },
            sort_keys=True,
        ).encode()
    )
    output = tmp_path / "crosswalk.json"
    assert build_crosswalk_cli(
        [
            "--oe", str(paths["oe"]),
            "--scoreboardgames", str(paths["scoreboardgames"]),
            "--matchschedule", str(paths["matchschedule"]),
            "--tournaments", str(paths["tournaments"]),
            "--capture-manifest", str(manifest_path),
            "--source-receipt", str(receipt_path),
            "--competition-map", str(map_path),
            "--output", str(output),
        ]
    ) == 0
    emitted = json.loads(output.read_text())
    assert emitted["status"] == "complete_authoritative_coverage"
    assert emitted["source_records"]["oe"]["integrity_verified"] is True


def test_cli_rejects_forged_tournaments_outside_genuine_capture(
    tmp_path: Path,
) -> None:
    oe, scoreboard, schedule, tournaments, receipt, mapping, _records, _raw = _fixture()
    capture_root = tmp_path / "capture"
    assembled_root = capture_root / "assembled"
    responses_root = capture_root / "responses"
    assembled_root.mkdir(parents=True)
    responses_root.mkdir()
    assembled_rows = {
        "ScoreboardGames": scoreboard,
        "MatchSchedule": schedule,
        "Tournaments": tournaments,
    }
    assembled: dict[str, dict[str, object]] = {}
    for label, rows in assembled_rows.items():
        path = assembled_root / f"{label}.json"
        path.write_bytes(_canonical(rows))
        assembled[label] = {
            "path": f"assembled/{label}.json",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    response_path = responses_root / "response.json"
    response_path.write_bytes(b"{}")
    response_records = [
        {
            "path": "responses/response.json",
            "bytes": 2,
            "sha256": hashlib.sha256(b"{}").hexdigest(),
        }
    ]
    manifest: dict[str, object] = {
        "captured_at": "2026-08-15T00:00:00Z",
        "sources": {
            label: {
                "url": f"https://example.test/{label}.json",
                "retrieved_at": "2026-08-15T00:00:00Z",
            }
            for label in ("oe", "scoreboardgames", "matchschedule", "tournaments")
        },
        "assembled": assembled,
        "response_records": response_records,
    }
    manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    manifest_path = capture_root / "capture-manifest.json"
    manifest_path.write_bytes(_canonical(manifest))

    oe_path = tmp_path / "oe.json"
    oe_path.write_bytes(_canonical(oe))
    forged_tournaments = tmp_path / "forged-tournaments.json"
    forged_tournaments.write_bytes(_canonical(tournaments))
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_bytes(_canonical(receipt))
    mapping_path = tmp_path / "mapping.json"
    mapping_path.write_bytes(_canonical(mapping))
    source_paths = {
        "oe": oe_path,
        "scoreboardgames": assembled_root / "ScoreboardGames.json",
        "matchschedule": assembled_root / "MatchSchedule.json",
        "tournaments": forged_tournaments,
    }
    source_rows = {
        "oe": oe,
        "scoreboardgames": scoreboard,
        "matchschedule": schedule,
        "tournaments": tournaments,
    }
    source_records_path = tmp_path / "source-records.json"
    source_records_path.write_bytes(
        _canonical(
            {
                "source_records": {
                    label: {
                        "path": str(path.resolve()),
                        "locator": str(path.resolve()),
                        "retrieved_at": "2026-08-15T00:00:00Z",
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size,
                        "payload_sha256": hashlib.sha256(
                            _canonical(source_rows[label])
                        ).hexdigest(),
                        "payload_bytes": len(_canonical(source_rows[label])),
                    }
                    for label, path in source_paths.items()
                }
            }
        )
    )

    with pytest.raises(SystemExit) as exc_info:
        build_crosswalk_cli(
            [
                "--oe",
                str(oe_path),
                "--scoreboardgames",
                str(assembled_root / "ScoreboardGames.json"),
                "--matchschedule",
                str(assembled_root / "MatchSchedule.json"),
                "--tournaments",
                str(forged_tournaments),
                "--capture-manifest",
                str(manifest_path),
                "--source-records",
                str(source_records_path),
                "--source-receipt",
                str(receipt_path),
                "--competition-map",
                str(mapping_path),
                "--output",
                str(tmp_path / "crosswalk.json"),
            ]
        )
    assert exc_info.value.code == 2
