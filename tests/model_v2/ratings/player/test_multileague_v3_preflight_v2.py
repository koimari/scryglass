from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from lol_kills.v2.ratings.player import multileague_v3_preflight_v2 as preflight
from lol_kills.v2.ratings.player.multileague_v3_preflight_v1_registry import (
    validate_registered_source_preflight_v1,
)
from lol_kills.v2.ratings.player.multileague_v3_preflight_v3_registry import (
    validate_registered_source_preflight_v3,
)


def _resign(payload: dict) -> None:
    payload["artifact_sha256"] = preflight._canonical_sha256(
        {key: value for key, value in payload.items() if key != "artifact_sha256"}
    )


def test_failed_v1_preflight_is_code_pinned() -> None:
    payload = validate_registered_source_preflight_v1(root=Path(".").resolve())
    assert payload["result_state"] == "SOURCE_SCHEMA_PREFLIGHT_FAILED"
    assert payload["outcome_access"]["future_holdout_targets_accessed"] is False


def test_corrected_source_rehearsal_passes_but_grants_no_authority() -> None:
    payload = preflight.build_corrected_source_preflight(
        built_at="2026-08-01T23:50:00Z",
        root=Path(".").resolve(),
    )
    assert payload["result_state"] == preflight.RESULT_STATE
    assert payload["adapter_preflight"]["coverage"]["development_maps"] == 3521
    assert payload["numerical_preflight"]["applied_series"] == 1419
    assert payload["numerical_preflight"]["latent_dimension"] == 529
    assert payload["future_boundary"]["future_holdout_targets_accessed"] is False
    assert all(value is False for value in payload["authority"].values())
    assert all(value is None for value in payload["decision_outputs"].values())


def test_corrected_source_rehearsal_rejects_forged_authority() -> None:
    payload = preflight.build_corrected_source_preflight(
        built_at="2026-08-01T23:50:00Z",
        root=Path(".").resolve(),
    )
    forged = deepcopy(payload)
    forged["authority"]["player_rating_authority"] = True
    _resign(forged)
    with pytest.raises(preflight.CorrectedSourcePreflightError, match="exceeds authority"):
        preflight.validate_corrected_source_preflight(
            forged,
            root=Path(".").resolve(),
        )


def test_clock_corrected_preflight_is_registered_and_non_authorizing() -> None:
    payload = validate_registered_source_preflight_v3(root=Path(".").resolve())
    assert payload["built_at_utc"] == "2026-08-01T23:51:00+00:00"
    assert payload["result_state"] == preflight.RESULT_STATE
    assert payload["authority"]["player_rating_authority"] is False
