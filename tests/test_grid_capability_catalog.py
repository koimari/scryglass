from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from lol_kills import grid_capability_catalog as catalog


def _type_ref(name: str) -> dict[str, object]:
    return {"kind": "SCALAR", "name": name, "ofType": None}


def _schema(root_fields: list[str], *, state: bool = False) -> dict[str, object]:
    fields = [
        {
            "name": name,
            "description": None,
            "isDeprecated": False,
            "deprecationReason": None,
            "args": [],
            "type": _type_ref("String"),
        }
        for name in root_fields
    ]
    types: list[dict[str, object]] = [
        {
            "kind": "OBJECT",
            "name": "Query",
            "description": None,
            "fields": fields,
            "inputFields": None,
            "interfaces": [],
            "enumValues": None,
            "possibleTypes": None,
        },
        {
            "kind": "SCALAR",
            "name": "String",
            "description": None,
            "fields": None,
            "inputFields": None,
            "interfaces": None,
            "enumValues": None,
            "possibleTypes": None,
        },
    ]
    if state:
        types.append(
            {
                "kind": "OBJECT",
                "name": "GameState",
                "description": None,
                "fields": [
                    {
                        "name": "finished",
                        "description": None,
                        "isDeprecated": False,
                        "deprecationReason": None,
                        "args": [],
                        "type": _type_ref("Boolean"),
                    },
                    {
                        "name": "teams",
                        "description": None,
                        "isDeprecated": False,
                        "deprecationReason": None,
                        "args": [],
                        "type": _type_ref("String"),
                    },
                ],
                "inputFields": None,
                "interfaces": [],
                "enumValues": None,
                "possibleTypes": None,
            }
        )
    return {
        "queryType": {"name": "Query"},
        "mutationType": None,
        "subscriptionType": None,
        "types": types,
        "directives": [],
    }


def test_safe_headers_never_persist_auth_material() -> None:
    result = catalog._safe_headers(
        {
            "X-RateLimit-Remaining": "9",
            "x-api-key": "secret",
            "Authorization": "Bearer secret",
            "Set-Cookie": "secret",
        }
    )
    assert result == {"x-ratelimit-remaining": "9"}
    assert "secret" not in json.dumps(result)


def test_field_availability_marks_outcome_signals_as_leakage() -> None:
    schema = _schema(["seriesState"], state=True)
    fields = catalog._annotated_fields("series_state", schema)
    by_name = {(row["type"], row["field"]): row for row in fields}
    assert (
        by_name[("GameState", "finished")]["availability"]["class"]
        == "final_outcome_signal"
    )
    assert (
        by_name[("GameState", "teams")]["availability"]["class"]
        == "live_and_final_snapshot_or_unknown"
    )
    assert (
        catalog._field_availability("series_state", "GameTeamStateLol", "won")[
            "class"
        ]
        == "final_outcome_signal"
    )


def test_local_event_scan_is_receipted_and_non_exhaustive(tmp_path: Path) -> None:
    path = tmp_path / "events_1_grid.jsonl.zip"
    rows = [
        {
            "id": "tx-1",
            "seriesId": "1",
            "sequenceNumber": 1,
            "events": [
                {
                    "id": "e-1",
                    "actor": "player",
                    "action": "killed",
                    "target": "player",
                    "seriesState": {"id": "1"},
                }
            ],
        }
    ]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("events.jsonl", "\n".join(json.dumps(row) for row in rows))
    result = catalog.scan_local_event_archives(tmp_path)
    assert result["archive_count"] == 1
    assert result["transaction_count"] == 1
    assert result["event_count"] == 1
    assert result["event_families"] == [
        {
            "family": "player.killed.player",
            "observed_count": 1,
            "fields": [
                "action",
                "actor",
                "id",
                "seriesState",
                "target",
            ],
            "coverage_status": "observed_in_local_fixture_not_exhaustive",
        }
    ]
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["archive_receipts"][0]["sha256"] == expected


def test_file_listing_strips_urls_and_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(*args, **kwargs):
        return (
            {
                "files": [
                    {
                        "id": "events-grid",
                        "name": "events.jsonl",
                        "url": "https://example.invalid/?token=secret",
                        "size": 123,
                    }
                ]
            },
            {"x-ratelimit-remaining": "4"},
            200,
        )

    monkeypatch.setattr(catalog, "_request_json_with_headers", fake_request)
    result = catalog._probe_file_listing("super-secret", "123")
    dumped = json.dumps(result)
    assert result["status"] == "confirmed"
    assert result["download_attempted"] is False
    assert result["signed_urls_retained"] is False
    assert "url" not in result["files"][0]["metadata"]
    assert result["files"][0]["temporal_class"] == "timestamped_event_stream"
    assert "secret" not in dumped


def test_build_catalog_is_fail_closed_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    central = _schema(
        [
            "allSeries",
            "gameIdByExternalId",
            "playerIdByExternalId",
            "seriesIdByExternalId",
            "teamIdByExternalId",
        ]
    )
    state = _schema(["seriesState"], state=True)

    def fake_graphql(endpoint, key, query, variables=None):
        if "__schema" in query:
            value = central if endpoint == catalog.GRAPHQL_ENDPOINT else state
            return {"__schema": value}, {}, 200
        if endpoint == catalog.GRAPHQL_ENDPOINT:
            return {"title": {"id": "3", "name": "League of Legends"}}, {}, 200
        return {"seriesState": {"id": "123", "finished": True}}, {}, 200

    monkeypatch.setattr(catalog, "_graphql", fake_graphql)
    monkeypatch.setattr(
        catalog,
        "_probe_file_listing",
        lambda key, series_id: {
            "status": "confirmed",
            "series_id": series_id,
            "download_attempted": False,
        },
    )
    result = catalog.build_catalog(
        key="secret",
        local_events_dir=tmp_path,
        probe_series_id="123",
        generated_at="2026-07-30T00:00:00Z",
    )
    assert result["scope"]["model_authority"] is False
    assert result["scope"]["market_edge_claim_authority"] is False
    assert result["provenance"]["match_files_downloaded"] is False
    assert result["capabilities"][-1]["status"] == "not_exposed_in_introspected_graphql_schemas"
    unhashed = {key: value for key, value in result.items() if key != "catalog_sha256"}
    assert result["catalog_sha256"] == catalog._sha256(unhashed)


def test_write_catalog_is_stable_json(tmp_path: Path) -> None:
    path = tmp_path / "catalog.json"
    value = {"schema_version": catalog.CATALOG_SCHEMA, "catalog_sha256": "a" * 64}
    catalog.write_catalog(path, value)
    assert json.loads(path.read_text()) == value
    assert path.read_bytes().endswith(b"\n")
