"""Tests for the patch-wide descriptive production bundle contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.v2.tierlists.production_bundle import (
    AUTHORITY_LOCATOR,
    CANDIDATE_LOCATOR,
    EVALUATION_LOCATOR,
    SOURCE_LOCATOR,
    SOURCE_META_LOCATOR,
    ProductionBundleError,
    _public_structural_similarity,
    _cell_patch,
    _public_scope_label,
    _public_patch_scope_id,
    _require_public_source_mode,
    _source_tree_sha256,
    _validate_matchup_profile,
    _validate_regional_views,
    _validate_response_matrix,
    verify_production_index,
)


ROOT = Path(__file__).resolve().parents[3]


def test_patch_wide_bundle_verifies() -> None:
    report = verify_production_index(ROOT)
    assert report["scope_count"] == 39
    assert report["cell_count"] == 195
    assert report["production_cell_count"] == 195


def test_public_bundle_uses_riot_patch_namespace_and_scope_ids() -> None:
    assert _cell_patch({"patches": ["16.15"]}) == "26.15"
    assert _cell_patch({"patches": ["16.16"]}) == "26.16"
    assert _public_patch_scope_id("patch:16.16", "16.16", "26.16") == "patch:26.16"
    assert _public_scope_label("Patch 16.15", "26.15") == "Patch 26.15"


def test_source_tree_reads_code_from_the_worker_checkout(tmp_path: Path) -> None:
    for locator in (
        CANDIDATE_LOCATOR,
        EVALUATION_LOCATOR,
        AUTHORITY_LOCATOR,
        SOURCE_LOCATOR,
        SOURCE_META_LOCATOR,
        Path("data/lol/v2/champions/lcc-atom-bridge-v1.json"),
    ):
        target = tmp_path / locator
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(locator.as_posix().encode("utf-8"))

    digest = _source_tree_sha256(tmp_path, {}, code_root=ROOT)

    assert len(digest) == 64


def test_public_bundle_rejects_a_grid_backed_candidate() -> None:
    with pytest.raises(ProductionBundleError, match="OE-only candidate"):
        _require_public_source_mode({"source_mode": "oe_plus_grid"})


def test_public_similarity_library_uses_the_validated_atom_bridge() -> None:
    library = _public_structural_similarity(ROOT)
    assert library["schema_version"] == "scryglass:champion-structural-similarity:v1"
    assert len(library["champions"]) == 173
    assert all(profile["champion_image_url"] for profile in library["champions"])


def test_coach_board_matchups_and_regional_views_are_validated() -> None:
    row = {
        "champion_id": "riot:champion:1",
        "matchup_profile": [
            {
                "champion_id": "riot:champion:2",
                "model_edge_pp": 8.5,
                "posterior_interval_pp": {"low": -2.0, "high": 18.0},
                "posterior_positive_probability": 0.82,
                "effective_maps": 4.5,
                "series_count": 3,
                "evidence_status": "supported",
            }
        ],
    }
    _validate_matchup_profile(row)
    _validate_regional_views(
        {
            "regional_views": [
                {
                    "id": "LCK",
                    "maps": 12,
                    "rows": [
                        {
                            "champion_id": "riot:champion:1",
                            "regional_rank": 1,
                            "global_rank": 4,
                            "strength_score_pp": 8.5,
                            "played_maps": 6,
                        }
                    ],
                }
            ]
        }
    )

    invalid = dict(row)
    invalid["matchup_profile"] = [
        {
            **row["matchup_profile"][0],
            "posterior_interval_pp": {"low": 10.0, "high": -1.0},
        }
    ]
    with pytest.raises(ProductionBundleError, match="interval is inverted"):
        _validate_matchup_profile(invalid)


def test_response_matrix_requires_complete_square_matchups() -> None:
    matrix = {
        "response_matrix": {
            "champions": [
                {"champion_id": "riot:champion:1", "champion": "Annie"},
                {"champion_id": "riot:champion:2", "champion": "Olaf"},
            ],
            "edge_pp": [[None, 4.2], [-4.2, None]],
            "interval_low_pp": [[None, -2.0], [-10.0, None]],
            "interval_high_pp": [[None, 10.0], [2.0, None]],
            "evidence": [[None, "supported"], ["supported", None]],
            "effective_maps": [[None, 12.5], [12.5, None]],
            "basis": [[None, "observed_pair_plus_model"], ["observed_pair_plus_model", None]],
        }
    }
    _validate_response_matrix(matrix)

    invalid = {"response_matrix": {**matrix["response_matrix"], "edge_pp": [[None, 4.2]]}}
    with pytest.raises(ProductionBundleError, match="edge_pp is malformed"):
        _validate_response_matrix(invalid)

    invalid_basis = {
        "response_matrix": {
            **matrix["response_matrix"],
            "basis": [[None, "unknown"], ["observed_pair_plus_model", None]],
        }
    }
    with pytest.raises(ProductionBundleError, match="basis is invalid"):
        _validate_response_matrix(invalid_basis)
