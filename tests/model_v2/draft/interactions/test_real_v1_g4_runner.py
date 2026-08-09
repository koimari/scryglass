from __future__ import annotations

import pytest

from lol_kills.v2.draft.interactions.real_v1_g4 import runner
from lol_kills.v2.draft.interactions.real_v1_g4.contract import G4RepairBlocked
from lol_kills.v2.draft.interactions.real_v1_g4.fixtures import synthetic_loaders


def _permit() -> dict[str, object]:
    return {
        "approved_action": "private_target_m0_load_and_rank_assay",
        "decision": "PASS",
        "final_temporal_holdout_sealed": True,
        "independent_from_runner_and_generator": True,
        "review_core_sha256": runner.dry_run()["review_core_sha256"],
        "schema_id": "scryglass.representation-rank-runner-review-permit.v1",
    }


def test_dry_run_is_support_first_and_zero_protected_reads() -> None:
    value = runner.dry_run()
    assert value["call_order"][:3] == ["verify_review_core", "verify_registered_2026_support_PASS", "verify_fresh_independent_permit"]
    assert all(value[key] == 0 for key in ("target_loader_calls", "m0_loader_calls", "outcome_loader_calls", "fit_availability_loader_calls", "fit_execution_calls"))


def test_incremental_fixture_executes_exact_52_slots(tmp_path) -> None:
    output = tmp_path / "isolated-result.json"
    value = runner.execute_once_after_permit(_permit(), loaders=synthetic_loaders(incremental=True), result_path=output)
    assert value["run_status"] == "accepted" and value["selected_width"] == 1
    assert len(value["ledger"]) == 52 and [row["sequence"] for row in value["ledger"]] == list(range(1, 53))
    assert all(row["execution_status"] == "passed" for row in value["ledger"])
    assert output.exists()


def test_m0_fixture_returns_no_incremental_winner() -> None:
    value = runner.execute_once_after_permit(_permit(), loaders=synthetic_loaders(incremental=False))
    assert value["run_status"] == "NO_INCREMENTAL_DRAFT_WINNER" and value["selected_model"] == "M0"


def test_invalid_permit_stops_before_protected_loader() -> None:
    calls = {"target": 0}; loaders = synthetic_loaders(incremental=True)
    original = loaders["load_target_m0"]

    def target():
        calls["target"] += 1
        return original()

    loaders["load_target_m0"] = target
    with pytest.raises(G4RepairBlocked):
        runner.execute_once_after_permit({}, loaders=loaders)
    assert calls["target"] == 0


def test_runner_rejects_non_three_start_fit_adapter() -> None:
    loaders = synthetic_loaders(incremental=True)
    original = loaders["fit"]

    def fit_with_wrong_start_count(slot, rows):
        response = dict(original(slot, rows))
        response["optimization_start_count"] = 4
        return response

    loaders["fit"] = fit_with_wrong_start_count
    with pytest.raises(G4RepairBlocked, match="FIT_ADAPTER_START_BUDGET_MISMATCH"):
        runner.execute_once_after_permit(_permit(), loaders=loaders)
