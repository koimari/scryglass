from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import v3_prefit


def test_v3_prefit_binds_corrected_g2_and_refuses_execution() -> None:
    bundle = v3_prefit.build_v3_prefit_bundle()
    assert bundle["schema_id"] == v3_prefit.SCHEMA
    assert bundle["state"] == "PREFIT_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED"
    assert bundle["execution_authorization"] is False
    assert bundle["dependencies"]["protected_reads"] == 0
    assert bundle["dependencies"]["final_holdout_reads"] == 0
    assert bundle["input_identities"]["G2"]["artifact_canonical_sha256"] == v3_prefit.G2_ARTIFACT_CANONICAL_SHA256
    assert bundle["input_identities"]["compatibility"]["status"] == "V3_FEATURE_REPLAY_BOUND_INDEPENDENT_REVIEW_REQUIRED"
    assert v3_prefit.validate_v3_prefit_bundle(bundle) == bundle["artifact_sha256"]


@pytest.mark.parametrize("path", [
    ("claim_ceiling", "promotion"),
    ("target_access", "final_holdout_reads"),
])
def test_v3_prefit_mutations_fail_closed(path: tuple[str, str]) -> None:
    bundle = deepcopy(v3_prefit.build_v3_prefit_bundle())
    bundle[path[0]][path[1]] = True if path[1] == "promotion" else 1
    with pytest.raises(v3_prefit.G5V3PreFitError):
        v3_prefit.validate_v3_prefit_bundle(bundle)


def test_v3_prefit_artifact_replay_is_byte_stable(tmp_path) -> None:
    path = tmp_path / "prefit-contract.json"
    first = v3_prefit.write_v3_prefit_bundle(path)
    first_bytes = path.read_bytes()
    second = v3_prefit.write_v3_prefit_bundle(path)
    assert first == second
    assert first_bytes == path.read_bytes()
