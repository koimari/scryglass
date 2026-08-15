from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path

from lol_kills.v2.draft.terminal import TerminalDraft, TerminalModel, score_terminal_draft
from lol_kills.v2.draft.terminal.development_evaluation import (
    CANDIDATE_ORDER,
    DraftRow,
    evaluate,
    load_snapshot,
    pre_event_team_elo_logits,
)


ROOT = Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def _current_report() -> dict[str, object]:
    return evaluate(ROOT)


def test_development_evaluation_is_chronological_and_non_authoritative() -> None:
    report = _current_report()
    assert report["status"] == "development_only"
    assert report["production_eligible"] is False
    assert report["public_probability_authorized"] is False
    assert report["population"]["complete_rows"] >= 16324
    assert report["population"]["series_clusters"] == report["population"]["dependence_clusters"]
    assert report["population"]["proxy_clustered_rows"] == 6310
    assert (
        report["population"]["complete_rows"]
        == report["population"]["proxy_clustered_rows"]
        + report["population"]["unclustered_single_game_rows"]
    )
    assert report["split_policy"]["chronological"] is True
    assert report["split_policy"]["series_grouped"] is False
    assert report["split_policy"]["dependence_clustered"] is True
    assert report["split_policy"]["series_identity_status"].startswith("authoritative_series_ids_unavailable")
    assert report["split_policy"]["candidate_search_opened_on_outer_test"] is False
    assert report["split_policy"]["calibration_fit_on_outer_test"] is False
    assert report["split_policy"]["candidate_selection_on_validation_only"] is True
    assert report["split_policy"]["outer_test_scored_only_for_selected_candidate"] is True
    assert report["split_policy"]["baseline_fit_uses_only_pre_event_results"] is True
    assert report["split_policy"]["outer_test_baseline_frozen_before_test"] is True
    assert report["split_policy"]["outer_test_outcomes_update_baseline"] is False
    assert report["split_policy"]["served_neutral_baseline_equalized"] is True
    assert report["baseline_adjustment"]["status"] == "development_nuisance_only"
    assert report["baseline_adjustment"]["team_identity_in_served_artifact"] is False
    assert report["candidate_order"] == list(CANDIDATE_ORDER)
    assert report["selection"] == {"status": "not_selected", "winner_candidate_id": None, "winner_transform": None}
    assert report["holdouts"]["future_patch"]["status"] == "development_diagnostic_only"
    assert report["holdouts"]["future_patch"]["patch_id"] == report["population"]["patches"][-1]
    assert report["holdouts"]["international_event_or_meta"]["status"] == "development_diagnostic_only"
    assert report["holdouts"]["roster_change"]["status"] == "not_applicable"
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["selection"]["outer_test_locked"] is True
        selected = [candidate for candidate in fold["candidates"] if candidate["selected_for_outer_test"]]
        assert len(selected) == 1
        assert "locked_outer_test" in selected[0]
        assert selected[0]["baseline_feature_count"] == 1
        assert selected[0]["baseline_strength_separation"]["baseline_is_not_serialized_in_served_artifact"] is True
        assert selected[0]["baseline_strength_separation"]["served_baseline_logit"] == 0.0
        assert fold["baseline_state_policy"]["outer_test_baseline_frozen_before_test"] is True
        assert fold["baseline_state_policy"]["outer_test_outcomes_update_baseline"] is False
        assert all("locked_outer_test" not in candidate for candidate in fold["candidates"] if not candidate["selected_for_outer_test"])


def test_development_snapshot_has_complete_role_assignments() -> None:
    rows, hashes = load_snapshot(ROOT)
    assert len(rows) >= 16324
    assert set(hashes) == {
        "oe_players_snapshot_sha256",
        "oe_metadata_snapshot_sha256",
        "dependence_cluster_proxy_raw_sha256",
        "dependence_cluster_proxy_artifact_sha256",
    }
    assert all(len(row.side_a) == 5 and len(row.side_b) == 5 for row in rows)
    assert all(len({champion for _, champion in (*row.side_a, *row.side_b)}) == 10 for row in rows)
    assert all(len(row.patch.split(".")[-1]) == 2 for row in rows)
    assert len({row.dependence_cluster_id for row in rows if row.game_id.startswith("LOLTMNT01_")}) > 1


def test_pre_event_baseline_waits_until_a_dependence_cluster_finishes() -> None:
    side_a = (("top", "Aatrox"), ("jungle", "Nidalee"), ("mid", "Ahri"), ("bot", "Jinx"), ("support", "Thresh"))
    side_b = (("top", "Gnar"), ("jungle", "Sejuani"), ("mid", "Orianna"), ("bot", "Aphelios"), ("support", "Rakan"))
    rows = [
        DraftRow("game-1", "series-1", datetime(2026, 1, 1, tzinfo=timezone.utc), "16.01", "LCK", "A", "B", side_a, side_b, 1),
        DraftRow("game-2", "series-1", datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc), "16.01", "LCK", "A", "B", side_a, side_b, 1),
        DraftRow("game-3", "series-2", datetime(2026, 1, 2, tzinfo=timezone.utc), "16.01", "LCK", "A", "B", side_a, side_b, 1),
        DraftRow("game-4", "series-3", datetime(2026, 1, 3, tzinfo=timezone.utc), "16.01", "LCK", "A", "B", side_a, side_b, 1),
    ]
    baseline = pre_event_team_elo_logits(rows)
    assert baseline["game-1"] == baseline["game-2"] == 0.0
    assert baseline["game-3"] > 0.0
    frozen = pre_event_team_elo_logits(rows, freeze_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert frozen["game-4"] == frozen["game-3"]
    assert baseline["game-4"] > baseline["game-3"]


def test_outer_test_freeze_rejects_updates_from_a_cluster_crossing_the_boundary() -> None:
    side_a = (("top", "Aatrox"), ("jungle", "Nidalee"), ("mid", "Ahri"), ("bot", "Jinx"), ("support", "Thresh"))
    side_b = (("top", "Gnar"), ("jungle", "Sejuani"), ("mid", "Orianna"), ("bot", "Aphelios"), ("support", "Rakan"))
    rows = [
        DraftRow("game-1", "series-crossing", datetime(2026, 1, 1, tzinfo=timezone.utc), "16.01", "LCK", "A", "B", side_a, side_b, 1),
        DraftRow("game-2", "series-crossing", datetime(2026, 1, 2, tzinfo=timezone.utc), "16.01", "LCK", "A", "B", side_a, side_b, 1),
        DraftRow("game-3", "series-after", datetime(2026, 1, 3, tzinfo=timezone.utc), "16.01", "LCK", "A", "B", side_a, side_b, 1),
    ]
    frozen = pre_event_team_elo_logits(rows, freeze_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
    assert frozen["game-3"] == 0.0


def test_legacy_summary_is_preserved_as_stale_historical_evidence() -> None:
    report = _current_report()
    summary = json.loads((ROOT / "data/lol/v2/models/draft-terminal/development-evaluation-summary.json").read_text())
    serialized = json.dumps(report, sort_keys=True, indent=2) + "\n"
    assert summary["run_output_sha256"] != hashlib.sha256(serialized.encode()).hexdigest()
    assert (
        summary["source_snapshot"]["oe_players_snapshot_sha256"]
        != report["source_snapshot"]["oe_players_snapshot_sha256"]
    )
    assert summary["status"] == "development_only"
    assert summary["production_eligible"] is False
    assert summary["grid_promotion_gate"]["status"] == "not_passed"
    assert summary["grid_promotion_gate"]["primary_source_for_cohort"] == "OE"


def test_refit_neutral_development_artifact_replays_exactly() -> None:
    fixture = json.loads((ROOT / "data/lol/v2/models/draft-terminal/terminal-neutral-development-replay-fixture.json").read_text())
    artifact_path = ROOT / "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v1.json"
    raw = artifact_path.read_bytes()
    model = TerminalModel.from_artifact_bytes(raw, expected_artifact_sha256=hashlib.sha256(raw).hexdigest())
    fixture_draft = fixture["draft"]
    draft = TerminalDraft.from_sides(
        fixture_draft["side_a"],
        fixture_draft["side_b"],
        event_start=fixture_draft["event_start"],
        source_available_at=fixture_draft["source_available_at"],
        source_record_id=fixture_draft["source_record_id"],
        source_payload_sha256=fixture_draft["source_payload_sha256"],
        source_rights_status=fixture_draft["source_rights_status"],
        actions=fixture_draft.get("actions"),
        final_assignments=fixture_draft.get("final_assignments"),
    )
    assert score_terminal_draft(draft, model, development=True) == fixture["expected_development"]
