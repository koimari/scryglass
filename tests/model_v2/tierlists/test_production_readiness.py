"""Tests for the descriptive tier-list production-readiness audit."""

from __future__ import annotations

from pathlib import Path

from lol_kills.v2.tierlists.production_readiness import inspect_production_readiness


ROOT = Path(__file__).resolve().parents[3]


def test_current_tierlist_package_is_ready_for_descriptive_promotion() -> None:
    report = inspect_production_readiness(ROOT)

    assert report["schema_version"] == "scryglass:tierlist-production-readiness:v1"
    assert report["status"] == "ready_for_promotion_review"
    assert report["promotion_eligible"] is True
    assert report["candidate_index"]["cell_count"] == 285
    assert report["candidate_index"]["status_counts"] == {"production": 285}
    assert report["terminal_l2_readiness"]["promotion_eligible"] is False
    assert report["terminal_l2_readiness"]["future_prediction_ledger"]["status"] == "missing"
    assert report["candidate_index"]["artifact_sha256"]
    assert report["blockers"] == []
    assert report["draft_score_boundary"]["blocks_descriptive_tier_api"] is False


def test_unavailable_cells_remain_fail_closed_without_becoming_blockers() -> None:
    report = inspect_production_readiness(ROOT)

    assert "production_index_contains_unknown_status" not in report["blockers"]
    assert "production_index_has_no_numeric_cells" not in report["blockers"]
    assert report["claims"] == {
        "production": True,
        "publication": True,
        "rank_eligibility": True,
    }
