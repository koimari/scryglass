from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.research.future_value_census_drift import (
    CensusDriftAuditError,
    verify_census_drift_audit,
)
from lol_kills.research.future_value_series_authority import canonical_sha256
from tools.build_future_value_census_drift_audit import build_from_paths


ROOT = Path(__file__).resolve().parents[1]
AUDIT_PATH = ROOT / "data/lol/v2/evaluation/future-value-census-drift-audit-v1.json"


def _load() -> dict[str, object]:
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def test_eight_external_only_ids_are_bound_to_local_bridge_rows() -> None:
    audit = _load()
    verify_census_drift_audit(audit)

    diff = audit["census_diff"]
    assert diff["current_game_count"] == 17_756
    assert diff["external_game_count"] == 17_764
    assert diff["overlap_game_count"] == 17_756
    assert diff["external_only_game_count"] == 8
    assert diff["current_only_game_count"] == 0
    assert diff["external_only_game_ids"] == [
        "LOLTMNT01_445121",
        "LOLTMNT01_445123",
        "LOLTMNT01_446026",
        "LOLTMNT02_453185",
        "LOLTMNT02_453274",
        "LOLTMNT02_454077",
        "LOLTMNT02_454095",
        "LOLTMNT02_454159",
    ]
    assert all(
        all(row["checks"].values())
        and row["bridge_assignment"]["outcome_used"] is False
        for row in audit["drift_rows"]
    )


def test_drift_source_dates_and_series_ids_are_exact() -> None:
    audit = _load()
    rows = {row["game_id"]: row for row in audit["drift_rows"]}
    assert {
        game_id: row["source_date"] for game_id, row in rows.items()
    } == {
        "LOLTMNT01_445121": "2026-08-20T14:00:08Z",
        "LOLTMNT01_445123": "2026-08-20T14:45:16Z",
        "LOLTMNT01_446026": "2026-08-20T13:13:37Z",
        "LOLTMNT02_453185": "2026-08-20T09:12:07Z",
        "LOLTMNT02_453274": "2026-08-20T14:11:04Z",
        "LOLTMNT02_454077": "2026-08-20T10:07:30Z",
        "LOLTMNT02_454095": "2026-08-20T11:03:29Z",
        "LOLTMNT02_454159": "2026-08-20T14:51:29Z",
    }
    assert {
        game_id: row["series_id"] for game_id, row in rows.items()
    } == {
        "LOLTMNT01_445121": "LES/2026 Season/Summer Playoffs_Third-Place Match_1",
        "LOLTMNT01_445123": "LES/2026 Season/Summer Playoffs_Third-Place Match_1",
        "LOLTMNT01_446026": "LES/2026 Season/Summer Playoffs_Third-Place Match_1",
        "LOLTMNT02_453185": "LCP/2026 Season/Split 3_Quarterfinals_1",
        "LOLTMNT02_453274": "Hitpoint Masters/2026 Season/Summer Split_Week 2_5",
        "LOLTMNT02_454077": "LCP/2026 Season/Split 3_Quarterfinals_1",
        "LOLTMNT02_454095": "LCP/2026 Season/Split 3_Quarterfinals_1",
        "LOLTMNT02_454159": "Hitpoint Masters/2026 Season/Summer Split_Week 2_5",
    }
    assert audit["series_summary"]["verified_series_count"] == 3
    assert audit["series_summary"]["source_tournament_non_null_count"] == 0


def test_drift_receipt_keeps_current_gate_fail_closed() -> None:
    audit = _load()
    decision = audit["decision"]
    assert decision["fail_closed"] is True
    assert decision["drift_rows_series_assignments_verified"] is True
    assert decision["drift_rows_tournament_fields_verified"] is False
    assert decision["current_census_series_gate_closed"] is False
    assert decision["can_promote_tier_evaluation"] is False
    assert "external_bridge_source_receipt_differs_from_current_accepted_receipt" in audit[
        "blockers"
    ]
    assert "tournament_assignment_not_source_bound" in audit["blockers"]


def test_drift_receipt_hash_tampering_fails_closed() -> None:
    audit = _load()
    verify_census_drift_audit(audit)
    assert audit["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in audit.items() if key != "receipt_sha256"}
    )

    changed = dict(audit)
    changed["status"] = "verified"
    with pytest.raises(CensusDriftAuditError, match="status"):
        verify_census_drift_audit(changed)


def test_local_capture_rebuilds_the_committed_receipt_when_available(
    tmp_path: Path,
) -> None:
    paths = {
        "external_source_path": Path(
            "/private/tmp/scryglass-four-variant-freeze-20260820T145129/"
            "future-value-source-receipt.json"
        ),
        "bridge_oe_rows_path": Path(
            "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
            "crosswalk-inputs/oe-games-v2.json"
        ),
        "crosswalk_path": Path(
            "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
            "oe-leaguepedia-series-crosswalk-v5.json"
        ),
        "crosswalk_receipt_path": Path(
            "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
            "oe-leaguepedia-series-crosswalk-v5.receipt.json"
        ),
    }
    if not all(path.is_file() for path in paths.values()):
        pytest.skip("captured external bridge inputs are not available")
    rebuilt = build_from_paths(output_path=tmp_path / "drift.json", **paths)
    assert rebuilt == _load()
