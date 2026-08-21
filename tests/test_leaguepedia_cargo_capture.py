from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from lol_kills.research.leaguepedia_cargo_capture import (
    CargoCaptureError,
    QUERY_CONTRACT_VERSION,
    USER_AGENT,
    capture_leaguepedia_sources,
    verify_capture_manifest,
)


def _fake_rows() -> dict[str, list[dict[str, object]]]:
    return {
        "ScoreboardGames": [
            {"GameId": "match-a_1", "RiotPlatformGameId": "LOLTMNT01_1", "DateTime UTC": "2026-08-01 12:00:00", "Team1": "Alpha", "Team2": "Beta", "Patch": "26.15", "Tournament": "LEC Summer", "OverviewPage": "LEC/2026 Summer"},
            {"GameId": "match-b_1", "RiotPlatformGameId": "LOLTMNT01_2", "DateTime UTC": "2026-08-02 13:00:00", "Team1": "Gamma", "Team2": "Delta", "Patch": "26.15", "Tournament": "LEC Summer", "OverviewPage": "LEC/2026 Summer"},
        ],
        "MatchSchedule": [
            {"MatchId": "match-a", "DateTime UTC": "2026-08-01 11:55:00", "Team1": "Alpha", "Team2": "Beta", "Patch": "26.15", "OverviewPage": "LEC/2026 Summer", "Winner": None},
            {"MatchId": "match-b", "DateTime UTC": "2026-08-02 12:55:00", "Team1": "Gamma", "Team2": "Delta", "Patch": "26.15", "OverviewPage": "LEC/2026 Summer", "Winner": None},
        ],
        "Tournaments": [
            {"Name": "LEC 2026 Summer", "OverviewPage": "LEC/2026 Summer", "DateStart": "2026-07-01", "Date": "2026-08-01", "Region": "Europe", "League": "LEC", "TournamentLevel": "Primary", "IsOfficial": 1},
            {"Name": "LCK 2026 Playoffs", "OverviewPage": "LCK/2026 Season/Playoffs", "DateStart": "2026-08-02", "Date": "2026-09-15", "Region": "Korea", "League": "LCK", "TournamentLevel": "Primary", "IsOfficial": 1},
        ],
    }


def _json_bytes(rows: list[dict[str, object]]) -> bytes:
    return json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode() + b"\n"


def _fetcher(rows_by_table: dict[str, list[dict[str, object]]], calls: list[tuple[str, dict[str, str]]]):
    def fetch(url: str, headers: dict[str, str]) -> bytes:
        calls.append((url, dict(headers)))
        assert url.startswith("https://lol.fandom.com/wiki/Special:CargoExport?")
        assert headers["User-Agent"] == USER_AGENT
        assert "Authorization" not in headers
        assert "Cookie" not in headers
        query = parse_qs(urlsplit(url).query)
        table = query["tables"][0]
        rows = rows_by_table[table]
        if table == "Tournaments":
            where = query["where"][0]
            if " OR " in where:
                dates = re.findall(r'"(\d{4}-\d{2}-\d{2})"', where)
                assert len(dates) == 4
                requested_start, active_end, partition_start, partition_end = dates
                assert requested_start == active_end == partition_start
                selected = [
                    row
                    for row in rows
                    if (
                        str(row.get("DateStart", "")) < requested_start
                        and str(row.get("Date", "")) >= requested_start
                    )
                    or partition_start <= str(row.get("DateStart", "")) < partition_end
                ]
            else:
                dates = re.findall(r'"(\d{4}-\d{2}-\d{2})"', where)
                assert len(dates) == 2
                partition_start, partition_end = dates
                selected = [
                    row
                    for row in rows
                    if partition_start <= str(row.get("DateStart", "")) < partition_end
                ]
            return _json_bytes(selected)
        match = re.search(r'where=.*?%3E%3D\+%22(\d{4}-\d{2}-\d{2})', url)
        if match is None:
            # parse_qs decodes the where clause, so use the decoded value.
            match = re.search(r'>= "(\d{4}-\d{2}-\d{2})', query["where"][0])
        assert match is not None
        day = match.group(1)
        selected = [row for row in rows if str(row.get("DateTime UTC", "")).startswith(day)]
        return _json_bytes(selected)

    return fetch


def test_capture_saves_raw_responses_assembles_tables_and_hashes_manifest(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    manifest = capture_leaguepedia_sources(
        start_date="2026-08-01",
        end_date="2026-08-02",
        root=tmp_path,
        fetcher=_fetcher(_fake_rows(), calls),
        captured_at=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )

    assert manifest["status"] == "complete_raw_capture"
    assert manifest["coverage"] == {
        "scoreboardgames_rows": 2,
        "matchschedule_rows": 2,
        "tournaments_rows": 2,
        "requests": 6,
        "cache_hits": 0,
    }
    assert len(calls) == 6
    assert (tmp_path / "capture-manifest.json").is_file()
    assert len([path for path in (tmp_path / "raw").glob("**/*.json") if not path.name.endswith(".meta.json")]) == 6
    assert json.loads((tmp_path / "assembled" / "ScoreboardGames.json").read_text())[0]["GameId"] == "match-a_1"
    assert json.loads((tmp_path / "assembled" / "Tournaments.json").read_text())[0]["OverviewPage"] == "LEC/2026 Summer"
    tournament_rows = json.loads((tmp_path / "assembled" / "Tournaments.json").read_text())
    assert [row["OverviewPage"] for row in tournament_rows] == [
        "LEC/2026 Summer",
        "LCK/2026 Season/Playoffs",
    ]
    assert manifest["query_contract"]["tournament_partition_field"] == "DateStart"
    assert manifest["query_contract"]["schema_version"] == QUERY_CONTRACT_VERSION
    assert manifest["query_contract"]["scoreboard_direct_identity_field"] == "RiotPlatformGameId"
    scoreboard_urls = [
        url
        for url, _headers in calls
        if parse_qs(urlsplit(url).query)["tables"] == ["ScoreboardGames"]
    ]
    assert scoreboard_urls
    assert all("ScoreboardGames.RiotPlatformGameId" in parse_qs(urlsplit(url).query)["fields"][0] for url in scoreboard_urls)
    assert manifest["query_contract"]["tournament_partition_policy"] == "half_open_non_overlapping_windows"
    assert manifest["query_contract"]["tournament_end_date_upper_bound"] is None
    verify_capture_manifest(json.loads((tmp_path / "capture-manifest.json").read_text()))


def test_tournament_windows_include_active_boundary_and_later_end_date(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    manifest = capture_leaguepedia_sources(
        start_date="2026-08-01",
        end_date="2026-08-02",
        root=tmp_path,
        fetcher=_fetcher(_fake_rows(), calls),
    )

    tournament_rows = json.loads((tmp_path / "assembled" / "Tournaments.json").read_text())
    by_page = {row["OverviewPage"]: row for row in tournament_rows}
    assert by_page["LEC/2026 Summer"]["DateStart"] == "2026-07-01"
    assert by_page["LCK/2026 Season/Playoffs"]["Date"] == "2026-09-15"
    assert len(by_page) == manifest["coverage"]["tournaments_rows"] == 2

    tournament_where = [
        parse_qs(urlsplit(url).query)["where"][0]
        for url, _headers in calls
        if parse_qs(urlsplit(url).query)["tables"] == ["Tournaments"]
    ]
    assert tournament_where == [
        '((Tournaments.DateStart < "2026-08-01" AND Tournaments.Date >= "2026-08-01") OR '
        '(Tournaments.DateStart >= "2026-08-01" AND Tournaments.DateStart < "2026-08-02"))',
        'Tournaments.DateStart >= "2026-08-02" AND Tournaments.DateStart < "2026-08-03"',
    ]


def test_capture_resume_uses_verified_cache_without_fetching(tmp_path: Path) -> None:
    first_calls: list[tuple[str, dict[str, str]]] = []
    capture_leaguepedia_sources(
        start_date="2026-08-01",
        end_date="2026-08-02",
        root=tmp_path,
        fetcher=_fetcher(_fake_rows(), first_calls),
    )
    assert len(first_calls) == 6

    def fail_fetch(_url: str, _headers: dict[str, str]) -> bytes:
        raise AssertionError("resume unexpectedly performed a network fetch")

    resumed = capture_leaguepedia_sources(
        start_date="2026-08-01",
        end_date="2026-08-02",
        root=tmp_path,
        fetcher=fail_fetch,
        captured_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
    )
    assert resumed["coverage"]["cache_hits"] == 6
    assert resumed["manifest_sha256"] != ""


def test_verify_rejects_obsolete_tournament_query_contract(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    manifest = capture_leaguepedia_sources(
        start_date="2026-08-01",
        end_date="2026-08-01",
        root=tmp_path,
        fetcher=_fetcher(_fake_rows(), calls),
    )
    obsolete = dict(manifest)
    obsolete.pop("manifest_path", None)
    obsolete["query_contract"] = {
        **obsolete["query_contract"],
        "schema_version": "scryglass:leaguepedia-cargo-query-contract:v1",
    }
    obsolete.pop("manifest_sha256")
    obsolete["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            obsolete,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    with pytest.raises(CargoCaptureError, match="query contract is obsolete"):
        verify_capture_manifest(obsolete)


def test_capture_rejects_row_limit_as_possible_truncation(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    rows = _fake_rows()

    def full_fetch(url: str, headers: dict[str, str]) -> bytes:
        calls.append((url, headers))
        table = parse_qs(urlsplit(url).query)["tables"][0]
        return _json_bytes(rows[table])

    with pytest.raises(CargoCaptureError, match="truncated"):
        capture_leaguepedia_sources(
            start_date="2026-08-01",
            end_date="2026-08-01",
            root=tmp_path,
            limit=1,
            fetcher=full_fetch,
        )
    assert len([path for path in (tmp_path / "raw").glob("**/*.json") if not path.name.endswith(".meta.json")]) == 1
    rejected_meta = list((tmp_path / "raw").glob("**/*.meta.json"))
    assert len(rejected_meta) == 1
    assert json.loads(rejected_meta[0].read_text())["status"] == "rejected"


def test_capture_rejects_duplicate_identity_in_response(tmp_path: Path) -> None:
    rows = _fake_rows()
    rows["ScoreboardGames"].append(dict(rows["ScoreboardGames"][0]))

    def duplicate_fetch(url: str, _headers: dict[str, str]) -> bytes:
        table = parse_qs(urlsplit(url).query)["tables"][0]
        return _json_bytes(rows[table])

    with pytest.raises(CargoCaptureError, match="duplicate ScoreboardGames"):
        capture_leaguepedia_sources(
            start_date="2026-08-01",
            end_date="2026-08-01",
            root=tmp_path,
            fetcher=duplicate_fetch,
        )


def test_capture_rejects_changed_cached_bytes(tmp_path: Path) -> None:
    calls: list[tuple[str, dict[str, str]]] = []
    capture_leaguepedia_sources(
        start_date="2026-08-01",
        end_date="2026-08-01",
        root=tmp_path,
        fetcher=_fetcher(_fake_rows(), calls),
    )
    raw_path = next(
        path
        for path in (tmp_path / "raw" / "ScoreboardGames").glob("*.json")
        if not path.name.endswith(".meta.json")
    )
    raw_path.write_bytes(b"[]")
    with pytest.raises(CargoCaptureError, match="hash changed"):
        capture_leaguepedia_sources(
            start_date="2026-08-01",
            end_date="2026-08-01",
            root=tmp_path,
            fetcher=lambda _url, _headers: b"[]",
        )
