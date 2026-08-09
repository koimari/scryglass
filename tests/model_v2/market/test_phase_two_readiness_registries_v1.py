from __future__ import annotations

from copy import deepcopy

import pytest

from lol_kills.v2.market import phase_two_collection_readiness_registry_v1 as collection
from lol_kills.v2.market import phase_two_evaluation_readiness_registry_v1 as evaluation


def _collection_receipt(binding: dict) -> dict:
    return {
        "schema_version": collection.SCHEMA_VERSION,
        "registry_id": "collection-registry-1",
        "status": "OUTCOME_FREE_COLLECTION_READINESS_REGISTERED",
        "registered_at_utc": "2026-08-02T13:00:00+00:00",
        "review": {
            "reviewer_id": "independent-collection-reviewer",
            "reviewed_at_utc": "2026-08-02T12:00:00+00:00",
            "attestation": collection.REVIEW_ATTESTATION,
        },
        "readiness_binding": binding,
        "decision": {
            "phase_two_collection_readiness_independently_registered": True,
            "phase_two_collection_opened": False,
            "probability_or_quote_authorized": False,
            "betting_authorized": False,
        },
        "authority": dict(collection.AUTHORITY),
        "claim_ceiling": collection.CLAIM_CEILING,
    }


def _evaluation_receipt(binding: dict) -> dict:
    return {
        "schema_version": evaluation.SCHEMA_VERSION,
        "registry_id": "evaluation-registry-1",
        "status": "OUTCOME_FREE_EVALUATION_READINESS_REGISTERED",
        "registered_at_utc": "2026-08-02T15:00:00+00:00",
        "review": {
            "reviewer_id": "independent-evaluation-reviewer",
            "reviewed_at_utc": "2026-08-02T14:00:00+00:00",
            "attestation": evaluation.REVIEW_ATTESTATION,
        },
        "readiness_binding": binding,
        "decision": {
            "phase_two_evaluation_readiness_independently_registered": True,
            "phase_two_outcomes_opened": False,
            "phase_two_evaluation_run": False,
            "probability_or_betting_authorized": False,
        },
        "authority": dict(evaluation.AUTHORITY),
        "claim_ceiling": evaluation.CLAIM_CEILING,
    }


@pytest.mark.parametrize(
    ("module", "builder", "validator"),
    [
        (collection, _collection_receipt, collection.validate_phase_two_collection_readiness_registry_v1),
        (evaluation, _evaluation_receipt, evaluation.validate_phase_two_evaluation_readiness_registry_v1),
    ],
)
def test_readiness_registry_only_grants_identity(
    module, builder, validator
) -> None:
    binding = {
        "raw_sha256": "1" * 64,
        "artifact_sha256": "2" * 64,
        "outcomes_accessed": False,
    }
    receipt = builder(binding)
    checked = validator(receipt, expected_binding=binding)
    assert checked["authority"]["betting_authority"] is False

    forged = deepcopy(receipt)
    forged["authority"]["betting_authority"] = True
    with pytest.raises(module.__dict__[next(
        name for name in module.__dict__ if name.endswith("RegistryError")
    )], match="exceeds authority"):
        validator(forged, expected_binding=binding)
