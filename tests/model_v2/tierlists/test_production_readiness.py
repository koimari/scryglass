"""Tests for the descriptive tier-list production-readiness audit."""

from __future__ import annotations

from pathlib import Path

import pytest

from lol_kills.v2.tierlists.production_readiness import (
    TierListProductionReadinessError,
    inspect_production_readiness,
)


ROOT = Path(__file__).resolve().parents[3]


def test_retired_league_specific_package_is_not_ready() -> None:
    with pytest.raises(TierListProductionReadinessError, match="retired competition filters"):
        inspect_production_readiness(ROOT)
