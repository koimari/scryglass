"""Tests for the patch-wide descriptive production bundle contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.v2.tierlists.production_bundle import (
    ProductionBundleError,
    _require_public_source_mode,
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
