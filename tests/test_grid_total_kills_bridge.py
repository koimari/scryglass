from __future__ import annotations

import json
import hashlib
from pathlib import Path


BRIDGE = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "lol"
    / "warehouse"
    / "private_grid"
    / "market_cohort"
    / "v1"
    / "bridges"
    / "scryglass-main-goal-total-kills-bridge-v1.json"
)


def _payload() -> dict:
    return json.loads(BRIDGE.read_text(encoding="utf-8"))


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_total_kills_bridge_is_private_and_fail_closed() -> None:
    payload = _payload()

    assert payload["bridge_status"] == "REVALIDATION_REQUIRED"
    assert payload["privacy"] == "private_personal_research_only"
    assert payload["no_raw_grid_rows_embedded"] is True
    assert payload["population"]["verified_maps"] == 764
    assert payload["population_boundary"]["join_status"] == (
        "not_joinable_without_new_exact_crosswalk_and_scope_review"
    )
    assert payload["claim_ceiling"]["bookmaker_comparison"] == "unavailable"
    assert payload["claim_ceiling"]["edge_ev"] == "unavailable"
    assert payload["claim_ceiling"]["prospective_live_latency"] == "unavailable"


def test_total_kills_bridge_keeps_catalog_drift_visible() -> None:
    payload = _payload()
    source = payload["source"]

    assert source["catalog_hash_convention"] == "raw_file_sha256"
    assert source["catalog_hash_used_by_source"] == source["catalog_hash_current"]
    assert source["catalog_canonical_sha256_current"] == (
        "94fb8703d8bcdaab416c1b5f8ce727d5f486789267ae66e5b06f784766d127ed"
    )
    assert "catalog_receipt_requires_current_revalidation" in payload[
        "population_boundary"
    ]["reason_codes"]
    assert source["catalog_schema_hashes"] == {
        "central_data": "9950e21b2986a87ac202f5bc87aa2007ccb52b40fe2e0975647502a7e648f4b0",
        "series_state": "a1786edb0624fa162b85d8e9ecf1422fa6699ff6d608da1be1cd00fc18c33632",
    }


def test_total_kills_bridge_binds_local_source_bytes() -> None:
    payload = _payload()
    source = payload["source"]
    repo = Path(__file__).resolve().parents[1]

    manifest = repo / source["cohort_manifest_locator"]
    evaluation = repo / source["evaluation_locator"]
    assert _raw_sha256(manifest) == source["cohort_manifest_raw_sha256"]
    assert _raw_sha256(evaluation) == source["evaluation_raw_sha256"]

    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
    assert manifest_payload["manifest_sha256"] == source["cohort_manifest_id"]
    assert evaluation_payload["artifact_sha256"] == source["evaluation_artifact_id"]


def test_total_kills_bridge_does_not_promote_state_to_rating_or_draft_input() -> None:
    compatibility = _payload()["main_goal_compatibility"]

    assert compatibility["player_rating"] == "not_an_input"
    assert compatibility["team_rating"] == "not_an_input"
    assert compatibility["terminal_draft_score"] == "not_an_input"
    assert compatibility["partial_draft_score"] == "not_an_input"
