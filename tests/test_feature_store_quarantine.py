from __future__ import annotations

import pytest

from lol_kills.features.build import build_feature_store


def test_legacy_feature_store_fails_closed_without_explicit_research_opt_in() -> None:
    with pytest.raises(RuntimeError, match="quarantined"):
        build_feature_store()
