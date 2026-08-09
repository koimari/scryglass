"""Tests for the descriptive tier-list production-readiness audit."""

from __future__ import annotations

from pathlib import Path

from lol_kills.v2.tierlists.production_readiness import (
    inspect_production_readiness,
)


ROOT = Path(__file__).resolve().parents[3]


def test_patch_wide_package_is_ready() -> None:
    report = inspect_production_readiness(ROOT)
    assert report["status"] == "ready_for_promotion_review"
    assert report["promotion_eligible"] is True
    assert report["claims"]["production"] is True
