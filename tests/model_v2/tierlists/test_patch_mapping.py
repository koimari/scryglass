from __future__ import annotations

import json

import pytest

from lol_kills.v2.tierlists.patch_mapping import (
    PatchMappingError,
    load_mapping,
    normalize_oe_token,
    resolve_atom_snapshot_patch,
    resolve_official_patch,
    resolve_oe_patch,
)


def test_sidecar_loads_with_all_source_tokens_and_hashes() -> None:
    artifact = load_mapping()

    assert len(artifact.rows) == 39
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
    result = resolve_oe_patch("16.16", "2026-08-08T00:00:00Z")

    assert result.status == "unavailable"
    assert result.reason == "unknown_oe_token"
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
