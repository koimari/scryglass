from __future__ import annotations

import json

import pytest

from lol_kills.v2.data import g1_draft_features_v3 as features


def test_v3_replay_rebinds_player_identity_without_outcome_rows() -> None:
    manifest = features.build()
    checked = features.verify(expected_manifest_sha256=manifest["manifest_sha256"])
    assert checked["schema_version"] == features.SCHEMA
    assert checked["coverage"] == {
        "accepted_map_count": 1226,
        "feature_row_count": 1226,
        "pick_count": 12260,
        "identity_unavailable_map_count": 0,
    }
    assert checked["accepted_membership_origin"]["g2_artifact_canonical_sha256"] == features.G2_ARTIFACT_CANONICAL_SHA256
    assert checked["upstream_rebind"]["player_posterior_values_consumed_by_transform"] is False


def test_v3_manifest_claim_mutation_fails_closed(tmp_path) -> None:
    manifest = features.build()
    altered = dict(manifest)
    altered["claim_ceiling"] = dict(altered["claim_ceiling"])
    altered["claim_ceiling"]["prediction"] = True
    altered.pop("manifest_sha256")
    altered["manifest_sha256"] = features._sha256(altered)
    path = tmp_path / "manifest.json"
    path.write_bytes(features._canonical_bytes(altered) + b"\n")
    rows = tmp_path / "rows.jsonl"
    rows.write_bytes(features.OUTPUT_ROWS.read_bytes())
    with pytest.raises(features.G1DraftFeatureV3Error):
        features.verify(rows_path=rows, manifest_path=path, expected_manifest_sha256=altered["manifest_sha256"])

