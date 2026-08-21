from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.research.future_value_series_authority import (
    TARGET_PROXY_MAP_COUNT,
    SeriesAuthorityAuditError,
    build_series_authority_audit,
    canonical_sha256,
    file_record,
    verify_series_authority_audit,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data/lol/v2/evaluation/future-value-source-receipt-20260820.json"
CENSUS_PATH = ROOT / "data/lol/v2/evaluation/future-phase-accepted-census.json"
PHASE_PATH = ROOT / "data/lol/v2/evaluation/future-phase-evaluation.json"
PROXY_PATH = ROOT / "data/lol/v2/models/draft-interactions/series-cluster-proxy.json"
AUDIT_PATH = ROOT / "data/lol/v2/evaluation/future-value-series-authority-audit-v1.json"
EXACT_AUDIT_PATH = ROOT / "data/lol/v2/evaluation/future-value-exact-series-subset-audit-v1.json"


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _current_audit() -> dict[str, object]:
    return build_series_authority_audit(
        source_receipt=_load(SOURCE_PATH),
        accepted_census=_load(CENSUS_PATH),
        phase_evaluation=_load(PHASE_PATH),
        proxy_artifact=_load(PROXY_PATH),
        source_receipt_artifact=file_record(SOURCE_PATH),
        accepted_census_artifact=file_record(CENSUS_PATH),
        phase_evaluation_artifact=file_record(PHASE_PATH),
        proxy_artifact_file=file_record(PROXY_PATH),
    )


def test_frozen_census_has_no_authoritative_series_or_tournament_binding() -> None:
    audit = _current_audit()

    assert audit["status"] == "blocked_research_only"
    assert audit["authority"]["authoritative_series"] is False  # type: ignore[index]
    decision = audit["decision"]
    assert decision["fail_closed"] is True  # type: ignore[index]
    assert decision["can_assign_authoritative_series"] is False  # type: ignore[index]
    assert decision["can_populate_tournament_boundary"] is False  # type: ignore[index]
    assert decision["can_promote_tier_evaluation"] is False  # type: ignore[index]
    assert "accepted_source_receipt_has_no_series_identity" in audit["blockers"]
    assert "accepted_census_has_no_series_assignment" in audit["blockers"]
    assert "tournament_assignment_not_source_bound" in audit["blockers"]


def test_proxy_counts_and_collisions_are_retained_without_promotion() -> None:
    audit = _current_audit()
    phase = audit["series_authority"]["phase_evaluation"]  # type: ignore[index]
    assert phase["source_counts"] == {  # type: ignore[index]
        "exact_id_proxy": 1349,
        "team_tournament_proxy": 16407,
    }
    assert phase["possible_collisions"] == {  # type: ignore[index]
        "clusters": 2721,
        "rows": 15644,
        "cross_date_clusters": 1658,
        "cross_date_rows": 12902,
    }
    cohort = audit["proxy_cohort"]  # type: ignore[index]
    assert cohort["requested_map_count"] == TARGET_PROXY_MAP_COUNT  # type: ignore[index]
    assert cohort["requested_count_is_currently_bound"] is False  # type: ignore[index]
    assert cohort["observed_counts"]["legacy_lpl_gameid_audit_maps"] == 2730  # type: ignore[index]
    assert (
        cohort["observed_counts"]["legacy_leaguepedia_game_id_audit_maps"] == 1074
    )  # type: ignore[index]


def test_audit_hash_is_immutable_and_tampering_fails() -> None:
    audit = _current_audit()
    verify_series_authority_audit(audit)
    assert audit["receipt_sha256"] == canonical_sha256(  # type: ignore[index]
        {key: value for key, value in audit.items() if key != "receipt_sha256"}
    )

    changed = dict(audit)
    changed["status"] = "verified"
    with pytest.raises(SeriesAuthorityAuditError, match="status"):
        verify_series_authority_audit(changed)


def test_unbound_bridge_does_not_claim_full_coverage() -> None:
    audit = _current_audit()
    bridge = audit["series_authority"]["leaguepedia_oe_bridge"]  # type: ignore[index]
    assert bridge["artifact_present"] is False  # type: ignore[index]
    assert bridge["receipt_present"] is False  # type: ignore[index]
    assert bridge["authoritative_for_accepted_census"] is False  # type: ignore[index]
    assert "leaguepedia_oe_crosswalk_artifact_missing" in audit["blockers"]
    assert "leaguepedia_oe_crosswalk_receipt_missing" in audit["blockers"]


def test_external_bundle_binds_the_2788_cohort_to_another_census() -> None:
    audit = _load(AUDIT_PATH)
    verify_series_authority_audit(audit)
    cohort = audit["proxy_cohort"]
    external = cohort["external_variant_bundle"]
    assert cohort["requested_map_count"] == 2788
    assert cohort["observed_counts"]["external_variant_bundle_retained_proxy_maps"] == 2788
    assert external["retained_proxy_cluster_count"] == 377
    assert external["mapped_game_count"] == 15912
    assert external["source_game_count"] == 17764
    assert external["source_census_matches_accepted"] is False
    assert "requested_proxy_cohort_is_bound_to_different_source_census" in audit["blockers"]


def test_exact_series_subset_receipt_binds_counts_and_current_source_blocker() -> None:
    audit = _load(EXACT_AUDIT_PATH)
    claimed = audit["receipt_sha256"]
    body = {key: value for key, value in audit.items() if key != "receipt_sha256"}
    assert claimed == canonical_sha256(body)
    assert audit["status"] == "verified_external_subset_blocked_current_census"
    assert audit["decision"]["external_subset_assignments_verified"] is True
    assert audit["decision"]["current_census_series_gate_closed"] is False
    subset = audit["subset"]
    assert subset["validation_game_count"] == 10776
    assert subset["exact_series_game_count"] == 10523
    assert subset["proxy_game_count"] == 253
    assert subset["exact_series_count"] == 4759
    assert audit["external_series_source"]["source_census_matches_current"] is False
    assert "external_subset_source_receipt_differs_from_current_accepted_receipt" in audit[
        "blockers"
    ]
