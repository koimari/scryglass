from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_preflight_v1 as preflight


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = preflight._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_v1_source_failure_reproduces_without_future_outcome_access() -> None:
    payload = preflight.build_source_preflight_failure(root=Path(".").resolve())
    assert payload["result_state"] == preflight.RESULT_STATE
    assert payload["diagnostic"]["observed_dtype"] == "float64"
    assert payload["diagnostic"]["adapter_error"] == preflight.EXPECTED_ADAPTER_ERROR
    assert payload["outcome_access"] == {
        "future_holdout_maps_present": 0,
        "future_holdout_targets_accessed": False,
    }
    assert all(value is False for value in payload["authority"].values())


def test_v1_source_failure_rejects_forged_authority_and_diagnostic() -> None:
    payload = preflight.build_source_preflight_failure(root=Path(".").resolve())

    forged_authority = deepcopy(payload)
    forged_authority["authority"]["rating_authority"] = True
    _resign(forged_authority)
    with pytest.raises(preflight.SourcePreflightError, match="exceeds authority"):
        preflight.validate_source_preflight_failure(
            forged_authority,
            root=Path(".").resolve(),
        )

    forged_diagnostic = deepcopy(payload)
    forged_diagnostic["diagnostic"]["observed_dtype"] = "boolean"
    _resign(forged_diagnostic)
    with pytest.raises(preflight.SourcePreflightError, match="diagnostic changed"):
        preflight.validate_source_preflight_failure(
            forged_diagnostic,
            root=Path(".").resolve(),
        )
