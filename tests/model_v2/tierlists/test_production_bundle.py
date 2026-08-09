"""Tests for the patch-wide descriptive production bundle contract."""

from __future__ import annotations

from pathlib import Path

from lol_kills.v2.tierlists.production_bundle import (
    verify_production_index,
)


ROOT = Path(__file__).resolve().parents[3]


def test_patch_wide_bundle_verifies() -> None:
    report = verify_production_index(ROOT)
    assert report["scope_count"] == 39
    assert report["cell_count"] == 195
    assert report["production_cell_count"] == 195
