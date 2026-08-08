from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest

from lol_kills.v2.market import phase_one_collection_readiness_v1 as readiness
from lol_kills.v2.market import (
    phase_one_collection_readiness_registry_v1 as registry,
)


ROOT = Path(".").resolve()
LOCKED_AT = datetime(2026, 8, 2, 2, 40, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def receipt() -> dict:
    return readiness.build_phase_one_collection_readiness_v1(
        root=ROOT,
        clock=lambda: LOCKED_AT,
    )


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = readiness._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_readiness_is_pre_boundary_empty_and_non_authorizing(receipt: dict) -> None:
    checked = readiness.validate_phase_one_collection_readiness_v1(
        receipt, root=ROOT
    )
    assert checked["result_state"] == readiness.RESULT_STATE
    assert checked["locked_empty_collection_state"]["plans"] == 0
    assert checked["locked_empty_collection_state"]["event_bundles"] == 0
    assert checked["locked_empty_collection_state"]["joint_snapshots"] == 0
    assert checked["implementation"][
        "ready_for_outcome_free_phase_one_collection"
    ] is True
    assert checked["implementation"]["actual_future_evidence_present"] is False
    assert checked["clock_attestation"]["user_supplied_timestamp_allowed"] is False
    assert all(value is False for value in checked["authority"].values())
    assert all(value is None for value in checked["decision_outputs"].values())


def test_readiness_rejects_boundary_lock() -> None:
    with pytest.raises(
        readiness.PhaseOneCollectionReadinessError,
        match="before the future boundary",
    ):
        readiness.build_phase_one_collection_readiness_v1(
            root=ROOT,
            clock=lambda: datetime(2026, 8, 3, 0, 0, tzinfo=timezone.utc),
        )


def test_readiness_rejects_forged_future_evidence_or_authority(
    receipt: dict,
) -> None:
    forged_evidence = deepcopy(receipt)
    forged_evidence["implementation"]["actual_future_evidence_present"] = True
    _resign(forged_evidence)
    with pytest.raises(
        readiness.PhaseOneCollectionReadinessError,
        match="implementation claim changed",
    ):
        readiness.validate_phase_one_collection_readiness_v1(
            forged_evidence, root=ROOT
        )

    forged_authority = deepcopy(receipt)
    forged_authority["authority"]["betting_authority"] = True
    _resign(forged_authority)
    with pytest.raises(
        readiness.PhaseOneCollectionReadinessError,
        match="exceeds authority",
    ):
        readiness.validate_phase_one_collection_readiness_v1(
            forged_authority, root=ROOT
        )


def test_readiness_cli_exposes_no_user_lock_timestamp(
    capsys: pytest.CaptureFixture,
) -> None:
    with pytest.raises(SystemExit) as exc:
        readiness.main(["--help"])
    assert exc.value.code == 0
    help_text = capsys.readouterr().out
    assert "--locked-at" not in help_text
    assert "--timestamp" not in help_text


def test_registered_readiness_is_hash_pinned_and_empty() -> None:
    checked = registry.validate_registered_phase_one_collection_readiness_v1(
        root=ROOT
    )
    assert (
        checked["artifact_sha256"]
        == registry.REGISTERED_READINESS_ARTIFACT_SHA256
    )
    assert checked["locked_at_utc"] == registry.REGISTERED_READINESS_LOCKED_AT_UTC
    assert checked["locked_empty_collection_state"]["plans"] == 0
    assert checked["locked_empty_collection_state"]["event_bundles"] == 0
    assert checked["locked_empty_collection_state"]["joint_snapshots"] == 0
    assert all(value is False for value in checked["authority"].values())
