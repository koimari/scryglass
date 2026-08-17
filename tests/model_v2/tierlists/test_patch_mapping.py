from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.v2.tierlists.patch_mapping import (
    MappingArtifact,
    PatchMappingError,
    _live_source_binding,
    load_mapping,
    normalize_oe_token,
    resolve_atom_snapshot_patch,
    resolve_official_patch,
    resolve_oe_patch,
)


def test_sidecar_loads_with_all_source_tokens_and_hashes() -> None:
    artifact = load_mapping()

    assert len(artifact.rows) == 40
    assert artifact.payload["source_window"]["start"] == "2025-01-01T00:00:00Z"
    assert artifact.payload["source_window"]["end"] == "2026-08-08T12:13:56Z"
    assert artifact.payload["atom_snapshots"][0]["patch"] == "26.15"
    assert all(len(source["sha256"]) == 64 for source in artifact.payload["sources"])


def test_single_digit_oe_tokens_are_trailing_zero_tokens() -> None:
    assert normalize_oe_token("15.1") == "15.10"
    assert normalize_oe_token("15.2") == "15.20"
    assert normalize_oe_token("16.01") == "16.01"
    with pytest.raises(PatchMappingError):
        normalize_oe_token(15.1)


def test_current_token_resolves_to_the_exact_atom_snapshot() -> None:
    result = resolve_oe_patch("16.15", "2026-08-01T00:00:00Z")

    assert result.status == "resolved"
    assert result.exact_official_patch is True
    assert result.exact_atom_snapshot is True
    assert result.official_patch == "26.15"
    assert result.atom_snapshot_patch == "26.15"
    assert resolve_atom_snapshot_patch("16.15", "2026-08-08T12:13:56Z") == "26.15"


def test_live_binding_extends_the_audited_interval_without_changing_the_mapping() -> None:
    artifact = load_mapping()

    assert artifact.live_source is not None
    assert artifact.live_source["source_mode"] == "oe_only"
    assert artifact.live_source["source_latest"] >= artifact.payload["source_window"]["end"]
    assert artifact.rows["16.15"]["oe_observed_interval"]["end"] >= "2026-08-08T12:13:56Z"
    current = resolve_oe_patch("16.15", artifact.live_source["source_latest"])

    assert current.exact_official_patch is True
    assert current.exact_atom_snapshot is True


def test_historical_token_keeps_official_resolution_and_withholds_atom_snapshot() -> None:
    result = resolve_oe_patch("16.14", "2026-07-24T08:30:21Z")

    assert result.status == "resolved"
    assert result.official_patch == "26.14"
    assert result.atom_snapshot_patch is None
    assert resolve_official_patch("15.1", "2025-05-20T00:00:00Z") == "25.10"
    assert resolve_atom_snapshot_patch("16.14", "2026-07-24T08:30:21Z") is None


def test_time_safe_lookup_requires_as_of_inside_the_source_interval() -> None:
    missing_time = resolve_oe_patch("16.15", None)
    outside = resolve_oe_patch("16.15", "2026-07-30T00:00:00Z")
    after_source = resolve_oe_patch("16.15", "2026-08-09T00:00:00Z")

    assert missing_time.status == "unavailable"
    assert missing_time.reason == "as_of_required"
    assert outside.status == "unavailable"
    assert outside.reason == "as_of_outside_oe_source_interval"
    assert after_source.status == "unavailable"
    assert after_source.reason == "as_of_outside_oe_source_interval"


def test_unknown_token_has_no_nearest_patch_fallback() -> None:
    result = resolve_oe_patch("16.17", "2026-08-08T00:00:00Z")

    assert result.status == "unavailable"
    assert result.reason == "unknown_oe_token"
    assert result.official_patch is None
    assert result.atom_snapshot_patch is None


def test_26_16_atom_mapping_stays_unavailable_without_accepted_rows() -> None:
    path = Path("data/lol/v2/champions/oe-atom-patch-map-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = MappingArtifact(
        payload=payload,
        rows={row["oe_token"]: row for row in payload["mappings"]},
        path=path,
        repo_root=path.parents[3],
    )
    result = resolve_oe_patch(
        "16.16",
        "2026-08-11T18:00:00Z",
        mapping=mapping,
    )

    assert result.status == "unavailable"
    assert result.reason == "no_accepted_oe_rows"
    assert result.official_patch is None
    assert result.atom_snapshot_patch is None


def test_tampered_sidecar_fails_before_resolution(tmp_path) -> None:
    source = load_mapping().payload
    source["mappings"][0]["official_patch"] = "99.99"
    path = tmp_path / "mapping.json"
    path.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(PatchMappingError, match="hash mismatch"):
        load_mapping(path, verify_source_hashes=False)


def test_every_audited_row_has_release_order_and_two_evidence_sources() -> None:
    artifact = load_mapping()

    for row in artifact.rows.values():
        assert row["audit_status"] == "audited"
        assert row["confidence"] == "high"
        assert row["oe_observed_interval"]["start"] >= row["official_release_at"]
        assert {source["kind"] for source in row["evidence"]} == {
            "oe_source",
            "riot_patch_notes",
        }


def test_live_binding_uses_canonical_game_uid_for_map_count(tmp_path: Path) -> None:
    player_path = tmp_path / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    player_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "gameid": "annual-1",
                "game_uid": None,
                "date": "2026-08-08T12:00:00Z",
                "patch": "16.15",
            },
            {
                "gameid": "annual-1",
                "game_uid": "canonical-1",
                "date": "2026-08-08T12:01:00Z",
                "patch": "16.15",
            },
            {
                "gameid": "api-2",
                "game_uid": "canonical-2",
                "date": "2026-08-08T12:02:00Z",
                "patch": "16.15",
            },
        ]
    ).to_parquet(player_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "scryglass:oe-live-source:v1",
                "source_mode": "oe_only",
                "source_latest": "2026-08-08T12:02:00Z",
                "maps": 3,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "source_window": {"start": "2026-08-08T00:00:00Z"},
        "sources": [
            {
                "kind": "oe_live_player_games",
                "locator": "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet",
                "mutable_live_source": True,
            },
            {
                "kind": "oe_live_meta",
                "locator": "data/lol/warehouse/parquet/oe_live/meta.json",
                "mutable_live_source": True,
            },
        ],
        "mappings": [{"oe_token": "16.15"}],
    }

    _intervals, binding = _live_source_binding(payload, repo_root=tmp_path)

    assert binding["source_game_count"] == 3


def test_live_binding_accepts_zero_padded_alias_for_float_like_oe_token(tmp_path: Path) -> None:
    player_path = tmp_path / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    player_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "gameid": "map-16-02",
                "date": "2026-01-25T12:00:00Z",
                "patch": "16.2",
            }
        ]
    ).to_parquet(player_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "scryglass:oe-live-source:v1",
                "source_mode": "oe_only",
                "source_latest": "2026-01-25T12:00:00Z",
                "maps": 1,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "source_window": {"start": "2026-01-01T00:00:00Z"},
        "sources": [
            {
                "kind": "oe_live_player_games",
                "locator": "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet",
                "mutable_live_source": True,
            },
            {
                "kind": "oe_live_meta",
                "locator": "data/lol/warehouse/parquet/oe_live/meta.json",
                "mutable_live_source": True,
            },
        ],
        "mappings": [{"oe_token": "16.02"}],
    }

    intervals, binding = _live_source_binding(payload, repo_root=tmp_path)

    assert list(intervals) == ["16.02"]
    assert binding["source_token_count"] == 1


def test_live_binding_compares_metadata_before_the_public_history_window(tmp_path: Path) -> None:
    player_path = tmp_path / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    player_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {"gameid": "old-map", "date": "2024-12-20T12:00:00Z", "patch": "14.24"},
            {"gameid": "public-map", "date": "2026-01-25T12:00:00Z", "patch": "16.02"},
        ]
    ).to_parquet(player_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "scryglass:oe-live-source:v1",
                "source_mode": "oe_only",
                "source_latest": "2026-01-25T12:00:00Z",
                "maps": 2,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "source_window": {"start": "2026-01-01T00:00:00Z"},
        "sources": [
            {"kind": "oe_live_player_games", "locator": "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet", "mutable_live_source": True},
            {"kind": "oe_live_meta", "locator": "data/lol/warehouse/parquet/oe_live/meta.json", "mutable_live_source": True},
        ],
        "mappings": [{"oe_token": "16.02"}],
    }

    intervals, binding = _live_source_binding(payload, repo_root=tmp_path)

    assert list(intervals) == ["16.02"]
    assert binding["source_game_count"] == 1


def test_float_like_token_uses_event_time_when_both_audited_aliases_exist() -> None:
    path = Path("data/lol/v2/champions/oe-atom-patch-map-v1.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["oe_token"]: row for row in payload["mappings"]}
    early = deepcopy(rows["16.02"])
    late = deepcopy(rows["16.02"])
    late["oe_token"] = "16.20"
    late["official_patch"] = "26.20"
    late["official_release_at"] = "2026-08-01T00:00:00Z"
    late["oe_observed_interval"] = {
        "start": "2026-08-01T00:00:00Z",
        "end": "2026-08-10T00:00:00Z",
    }
    mapping = MappingArtifact(
        payload=payload,
        rows={"16.02": early, "16.20": late},
        path=path,
        repo_root=path.parents[3],
    )

    early_result = resolve_oe_patch("16.2", "2026-01-25T12:00:00Z", mapping=mapping)
    late_result = resolve_oe_patch("16.2", "2026-08-05T12:00:00Z", mapping=mapping)

    assert early_result.status == "resolved"
    assert early_result.oe_token == "16.02"
    assert early_result.official_patch == "26.02"
    assert late_result.status == "resolved"
    assert late_result.oe_token == "16.20"
    assert late_result.official_patch == "26.20"


def test_live_binding_uses_event_time_when_both_aliases_are_present(tmp_path: Path) -> None:
    player_path = tmp_path / "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
    meta_path = tmp_path / "data/lol/warehouse/parquet/oe_live/meta.json"
    player_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "gameid": "early",
                "date": "2026-01-25T12:00:00Z",
                "patch": "16.2",
            },
            {
                "gameid": "late",
                "date": "2026-08-05T12:00:00Z",
                "patch": "16.2",
            },
        ]
    ).to_parquet(player_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "schema_version": "scryglass:oe-live-source:v1",
                "source_mode": "oe_only",
                "source_latest": "2026-08-05T12:00:00Z",
                "maps": 2,
            }
        ),
        encoding="utf-8",
    )
    payload = {
        "source_window": {"start": "2026-01-01T00:00:00Z"},
        "sources": [
            {
                "kind": "oe_live_player_games",
                "locator": "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet",
                "mutable_live_source": True,
            },
            {
                "kind": "oe_live_meta",
                "locator": "data/lol/warehouse/parquet/oe_live/meta.json",
                "mutable_live_source": True,
            },
        ],
        "mappings": [
            {
                "oe_token": "16.02",
                "oe_observed_interval": {
                    "start": "2026-01-20T00:00:00Z",
                    "end": "2026-01-30T00:00:00Z",
                },
            },
            {
                "oe_token": "16.20",
                "oe_observed_interval": {
                    "start": "2026-08-01T00:00:00Z",
                    "end": "2026-08-10T00:00:00Z",
                },
            },
        ],
    }

    intervals, _binding = _live_source_binding(payload, repo_root=tmp_path)

    assert set(intervals) == {"16.02", "16.20"}
    assert intervals["16.02"]["observed_game_count"] == 1
    assert intervals["16.20"]["observed_game_count"] == 1
