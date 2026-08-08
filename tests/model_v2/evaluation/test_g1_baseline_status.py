from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.evaluation import benchmark_contract as bc
from lol_kills.v2.evaluation import g1_baseline_status as status


def test_materializes_authenticated_non_authorizing_typed_unavailable_status() -> None:
    bundle = status.materialize_g1_018_baseline_status()

    assert bundle["status"] == "TYPED_UNAVAILABLE"
    assert bundle["authority_status"] == "NONAUTHORIZING_STATUS_ONLY"
    assert bundle["final_labels_read"] is False
    assert bundle["source_rows_decoded"] is False
    assert bundle["source_snapshot"]["map_count"] == 1226
    assert bundle["source_snapshot"]["partition_counts"] == {
        "DEVELOPMENT": 214,
        "TRAIN": 805,
        "VALIDATION": 207,
    }
    assert bundle["source_snapshot"]["final_holdout_status"] == "SEALED_UNREAD"
    assert bundle["baseline_inventory"]["baseline_count"] == 61
    assert bundle["baseline_inventory"]["output_counts"] == {
        "partial_draft_score": 18,
        "player_rating": 7,
        "team_rating": 7,
        "terminal_draft_score": 29,
    }
    assert bundle["baseline_inventory"][
        "source_dependent_execution_bindings_status"
    ] == "UNBOUND"
    assert all(
        item["status"] == "TYPED_UNAVAILABLE"
        for item in bundle["baseline_inventory"]["baselines"]
    )
    assert bundle["claim_ceiling"] == {
        "baseline_execution": False,
        "baseline_score": False,
        "final_holdout": False,
        "prediction": False,
        "promotion": False,
        "publication": False,
        "sota": False,
    }
    assert status.validate_g1_018_baseline_status(bundle) == bundle[
        "bundle_sha256"
    ]


@pytest.mark.parametrize(
    "mutation",
    ("claim", "execution", "score", "source", "digest"),
)
def test_mutated_or_authority_broadening_status_rejects(mutation: str) -> None:
    bundle = status.materialize_g1_018_baseline_status()
    mutated = deepcopy(bundle)
    if mutation == "claim":
        mutated["claim_ceiling"]["promotion"] = True
    elif mutation == "execution":
        mutated["baseline_inventory"]["baselines"][0]["status"] = (
            "EXECUTABLE_PREBOUND"
        )
    elif mutation == "score":
        mutated["baseline_inventory"]["baselines"][0]["score"] = 0.0
    elif mutation == "source":
        mutated["source_snapshot"]["rows_raw_sha256"] = "f" * 64
    else:
        mutated["bundle_sha256"] = "f" * 64

    with pytest.raises(bc.BenchmarkContractError) as error:
        status.validate_g1_018_baseline_status(mutated)
    assert error.value.code == "G1_BASELINE_STATUS_MISMATCH"


def test_snapshot_row_drift_rejects_before_any_row_decode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = status._read_repo_file

    def drifted(root, locator, *, purpose):
        raw = original(root, locator, purpose=purpose)
        if locator == status.SNAPSHOT_ROWS_LOCATOR:
            return raw + b"{}\n"
        return raw

    monkeypatch.setattr(status, "_read_repo_file", drifted)
    with pytest.raises(bc.BenchmarkContractError) as error:
        status.materialize_g1_018_baseline_status()
    assert error.value.code == "G1_BASELINE_SNAPSHOT_MISMATCH"
