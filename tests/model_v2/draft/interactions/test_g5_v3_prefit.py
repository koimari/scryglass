from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from lol_kills.v2.draft.interactions.g5_exploratory import v3_prefit


def _archived_v3_prefit() -> dict[str, object]:
    path = Path("data/lol/v2/models/draft-interactions/g5-exploratory-v3/prefit-contract.json")
    raw = path.read_bytes()
    bundle = json.loads(raw)
    assert raw == v3_prefit.canonical_bytes(bundle) + b"\n"
    claimed = bundle["artifact_sha256"]
    unsigned = {key: value for key, value in bundle.items() if key != "artifact_sha256"}
    assert v3_prefit.sha256(unsigned) == claimed
    return bundle


def _validate_archived_v3_prefit(bundle: dict[str, object]) -> None:
    if bundle != _archived_v3_prefit():
        raise v3_prefit.G5V3PreFitError("archived v3 prefit bundle changed")


def test_v3_prefit_stays_closed_after_bound_dependency_drift() -> None:
    with pytest.raises(v3_prefit.G5V3PreFitError, match="v3 bound dependency changed"):
        v3_prefit.build_v3_prefit_bundle()
    bundle = _archived_v3_prefit()
    assert bundle["schema_id"] == v3_prefit.SCHEMA
    assert bundle["state"] == "PREFIT_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED"
    assert bundle["execution_authorization"] is False
    assert bundle["dependencies"]["protected_reads"] == 0
    assert bundle["dependencies"]["final_holdout_reads"] == 0
    assert bundle["input_identities"]["G2"]["artifact_canonical_sha256"] == v3_prefit.G2_ARTIFACT_CANONICAL_SHA256
    assert bundle["input_identities"]["compatibility"]["status"] == "V3_FEATURE_REPLAY_BOUND_INDEPENDENT_REVIEW_REQUIRED"
    with pytest.raises(v3_prefit.G5V3PreFitError, match="v3 bound dependency changed"):
        v3_prefit.validate_v3_prefit_bundle(bundle)


@pytest.mark.parametrize("path", [
    ("claim_ceiling", "promotion"),
    ("target_access", "final_holdout_reads"),
])
def test_v3_prefit_mutations_fail_closed(path: tuple[str, str]) -> None:
    bundle = deepcopy(_archived_v3_prefit())
    bundle[path[0]][path[1]] = True if path[1] == "promotion" else 1
    with pytest.raises(v3_prefit.G5V3PreFitError):
        _validate_archived_v3_prefit(bundle)


def test_v3_prefit_writer_refuses_dependency_drift(tmp_path) -> None:
    path = tmp_path / "prefit-contract.json"
    with pytest.raises(v3_prefit.G5V3PreFitError, match="v3 bound dependency changed"):
        v3_prefit.write_v3_prefit_bundle(path)
    assert not path.exists()
