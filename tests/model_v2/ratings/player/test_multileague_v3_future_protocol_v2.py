from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_future_protocol_v2 as protocol
from lol_kills.v2.ratings.player.multileague_v3_preflight_v2_registry import (
    validate_registered_source_preflight_v2,
)
from lol_kills.v2.ratings.player.multileague_v3_registry_v2 import (
    validate_registered_future_protocol_v2,
)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = protocol._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_corrected_source_preflight_is_registered_and_non_authorizing() -> None:
    payload = validate_registered_source_preflight_v2(root=Path(".").resolve())
    assert payload["result_state"] == (
        "CORRECTED_SOURCE_PREFLIGHT_PASSED_NON_AUTHORIZING"
    )
    assert payload["authority"]["player_rating_authority"] is False


def test_future_protocol_v2_supersedes_only_the_failed_source_package() -> None:
    payload = protocol.build_future_protocol_lock_v2(
        locked_at="2026-08-01T23:55:00Z",
        root=Path(".").resolve(),
    )
    assert payload["result_state"] == protocol.RESULT_STATE
    assert payload["supersession"]["candidate_changed"] is False
    assert payload["supersession"]["future_boundary_changed"] is False
    assert payload["supersession"]["evaluation_rule_changed"] is False
    assert payload["supersession"]["future_outcomes_used_for_remediation"] is False
    assert payload["future_holdout"]["status"] == "EMPTY_NOT_YET_ACQUIRED"
    assert payload["prediction_ledger"]["status"] == "NOT_YET_CREATED"
    assert payload["prediction_ledger"]["entries"] == 0
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())


def test_future_protocol_v2_rejects_forged_authority_and_ledger() -> None:
    payload = protocol.build_future_protocol_lock_v2(
        locked_at="2026-08-01T23:55:00Z",
        root=Path(".").resolve(),
    )
    forged_authority = deepcopy(payload)
    forged_authority["authority"]["player_rating_authority"] = True
    _resign(forged_authority)
    with pytest.raises(protocol.FutureProtocolV2Error, match="exceeds authority"):
        protocol.validate_future_protocol_lock_v2(
            forged_authority,
            root=Path(".").resolve(),
        )

    forged_ledger = deepcopy(payload)
    forged_ledger["prediction_ledger"]["entries"] = 1
    _resign(forged_ledger)
    with pytest.raises(protocol.FutureProtocolV2Error, match="ledger state changed"):
        protocol.validate_future_protocol_lock_v2(
            forged_ledger,
            root=Path(".").resolve(),
        )


def test_registered_future_protocol_v2_replays_and_remains_empty() -> None:
    payload = validate_registered_future_protocol_v2(root=Path(".").resolve())
    assert payload["result_state"] == protocol.RESULT_STATE
    assert payload["future_holdout"]["status"] == "EMPTY_NOT_YET_ACQUIRED"
    assert payload["prediction_ledger"]["entries"] == 0
    assert all(value is None for value in payload["decision_outputs"].values())
