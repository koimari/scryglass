"""Tests for the patch-wide descriptive production bundle contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.v2.tierlists.production_bundle import (
    ProductionBundleError,
    verify_production_index,
)


ROOT = Path(__file__).resolve().parents[3]


def test_retired_league_specific_bundle_fails_closed() -> None:
    with pytest.raises(ProductionBundleError, match="retired competition filters"):
        verify_production_index(ROOT)
