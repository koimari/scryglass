from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.draft.terminal import participant_dependence_diagnostic_v1 as diagnostic


ROOT = Path(__file__).resolve().parents[4]


@pytest.fixture(scope="module")
def payload() -> dict:
    return diagnostic.build_participant_dependence_diagnostic_v1(
        root=ROOT,
        clock=lambda: datetime(2026, 8, 2, 8, 35, tzinfo=timezone.utc),
    )


def test_participant_diagnostic_quantifies_coverage_without_claiming_independence(
    payload: dict,
) -> None:
    checked = diagnostic.validate_participant_dependence_diagnostic_v1(
        payload, root=ROOT
    )
    population = checked["population"]
    assert population["snapshot_maps"] == 6194
    assert population["maps_with_exact_ten_unique_players_and_roles"] == 5751
    assert population["unique_participants"] == 970
    assert population["component_graph"]["transitive_component_count"] == 1
    assert population["component_graph"]["all_valid_maps_in_one_component"] is True
    assert checked["decision"]["participant_dependence_support_verified"] is False
    assert checked["decision"]["promotion_eligible"] is False
    assert all(value is False for value in checked["authority"].values())


def test_participant_diagnostic_rejects_a_forged_splittable_conclusion(
    payload: dict,
) -> None:
    forged = deepcopy(payload)
    forged["population"]["component_graph"][
        "atomic_component_temporal_split_available"
    ] = True
    forged["artifact_sha256"] = diagnostic.sha256_canonical_object(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        diagnostic.ParticipantDependenceDiagnosticError,
        match="population drifted|component conclusion changed",
    ):
        diagnostic.validate_participant_dependence_diagnostic_v1(
            forged, root=ROOT
        )
