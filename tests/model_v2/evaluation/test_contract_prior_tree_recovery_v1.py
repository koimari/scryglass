from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import pytest

from lol_kills.v2.evaluation import contract_validation as validation
from lol_kills.v2.evaluation.contract_prior_tree_recovery_v1 import (
    ContractPriorTreeRecoveryError,
    load_prior_tree_recovery_v1,
    validate_prior_tree_recovery_v1,
    write_prior_tree_recovery_v1,
)
from lol_kills.v2.evaluation.types import canonical_sha256


ROOT = Path(__file__).resolve().parents[3]
SESSION_ROOT = Path("/Users/river/.codex/sessions")
PRIME_SESSION_ROOT = Path("/Users/river/.prime/agent/sessions")


def test_materialized_recovery_preserves_exact_frozen_prior_tree() -> None:
    manifest = load_prior_tree_recovery_v1(root=ROOT)
    evidence_root = ROOT / manifest["evidence_root"]
    assert manifest["file_count"] == 25
    assert manifest["reconstruction"]["reversed_hunk_count"] == 0
    assert manifest["reconstruction"]["reversed_hunks"] == []
    assert manifest["reconstruction"]["cutoff_evidence"][
        "reported_contract_tree_sha256"
    ] == validation.CONTRACT_TREE_SHA256
    for name, expected in validation.EXPECTED_SCHEMA_SHA256.items():
        locator = f"docs/model-v2/contracts/{name}"
        actual = hashlib.sha256((evidence_root / locator).read_bytes()).hexdigest()
        assert actual == expected
    for name, expected in validation.EXPECTED_EXAMPLE_SHA256.items():
        locator = f"docs/model-v2/contracts/{name}"
        actual = hashlib.sha256((evidence_root / locator).read_bytes()).hexdigest()
        assert actual == expected


def test_materialized_recovery_is_exact_and_non_authorizing() -> None:
    manifest = load_prior_tree_recovery_v1(root=ROOT)
    assert manifest["contract_tree_sha256"] == validation.CONTRACT_TREE_SHA256
    assert manifest["runner_provenance"]["generated_by_evaluated_system"] is True
    assert manifest["runner_provenance"]["independent_review_eligible"] is False
    assert all(value is False for value in manifest["authority"].values())


def test_self_rehashed_independence_forgery_is_rejected() -> None:
    manifest = load_prior_tree_recovery_v1(root=ROOT)
    forged = deepcopy(manifest)
    forged["runner_provenance"]["generated_by_evaluated_system"] = False
    forged["runner_provenance"]["independent_review_eligible"] = True
    forged["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        ContractPriorTreeRecoveryError, match="recovery provenance"
    ):
        validate_prior_tree_recovery_v1(forged, root=ROOT)


def test_recovery_writer_is_no_clobber() -> None:
    with pytest.raises(
        ContractPriorTreeRecoveryError, match="refusing to overwrite"
    ):
        write_prior_tree_recovery_v1(
            root=ROOT, session_root=SESSION_ROOT, extra_roots=[PRIME_SESSION_ROOT]
        )
