"""Tests for the patch-wide descriptive production bundle contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.v2.tierlists.production_bundle import (
    ProductionBundleError,
    _require_public_source_mode,
    _validate_matchup_profile,
    _validate_regional_views,
    verify_production_index,
)


ROOT = Path(__file__).resolve().parents[3]


def test_patch_wide_bundle_verifies() -> None:
    report = verify_production_index(ROOT)
    assert report["scope_count"] == 39
    assert report["cell_count"] == 195
    assert report["production_cell_count"] == 195


def test_public_bundle_rejects_a_grid_backed_candidate() -> None:
    with pytest.raises(ProductionBundleError, match="OE-only candidate"):
        _require_public_source_mode({"source_mode": "oe_plus_grid"})


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
