from __future__ import annotations

from copy import deepcopy
import os
import shutil
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest

from lol_kills.v2.evaluation import contract_reconciliation_semantic_replay_v1 as replay
from lol_kills.v2.evaluation import contract_reconciliation_v1 as reconciliation
from lol_kills.v2.evaluation import contract_reconciliation_review_v1 as review
from lol_kills.v2.evaluation.types import canonical_sha256


ROOT = Path(__file__).resolve().parents[3]

def _sandbox_ignore(dirpath: str, names: list[str]) -> set[str]:
    ignored = {"experiments"}
    if os.path.basename(os.path.normpath(dirpath)) == "snapshots":
        ignored |= {"multileague-v3", "real-v1"}
    return ignored


def _drift_sandbox(tmp_path: Path) -> Path:
    """A self-consistent repo copy whose docs tree drifts from the frozen C0."""
    sandbox = tmp_path / "repo"
    for rel in (
        "docs/model-v2",
        "data/lol/v2/evaluation",
        "data/lol/v2/champions",
        "data/lol/v2/publication",
        "data/lol/v2/tierlists",
        "data/lol/v2/models",
        "data/lol/v2/review",
        "data/lol/v2/snapshots/b1",
        reconciliation.SOURCE_LOCATOR,
        reconciliation.VALIDATOR_SOURCE_LOCATOR,
        "lol_kills/v2/evaluation/contract_reconciliation_semantic_replay_v1.py",
        "lol_kills/v2/evaluation/checks.py",
    ):
        src = ROOT / rel
        dst = sandbox / rel
        if src.is_dir():
            shutil.copytree(src, dst, ignore=_sandbox_ignore)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    readme = sandbox / "docs/model-v2/README.md"
    readme.write_bytes(readme.read_bytes() + b"\n<!-- drift sandbox marker -->\n")
    replay_payload = replay.build_reference_semantic_replay_v1(root=sandbox)
    out = sandbox / replay.DEFAULT_OUTPUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(
        (
            json.dumps(
                replay_payload, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True
            )
            + "\n"
        ).encode("ascii")
    )
    return sandbox


@pytest.fixture(scope="module")
def candidate_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _drift_sandbox(tmp_path_factory.mktemp("c0-sandbox"))


@pytest.fixture(scope="module")
def candidate(candidate_root: Path) -> dict:
    return reconciliation.build_contract_reconciliation_candidate_v1(
        root=candidate_root,
        clock=lambda: datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc),
    )


def test_candidate_inventories_drift_without_authorizing_it(
    candidate: dict, candidate_root: Path
) -> None:
    checked = reconciliation.validate_contract_reconciliation_candidate_v1(
        candidate, root=candidate_root
    )
    assert checked["structural_replay"]["schemas_checked"] == 9
    assert checked["structural_replay"]["examples_checked"] == 5
    assert checked["structural_replay"]["passed"] is True
    assert checked["drift"]["changed_runtime_schema_files"] == []
    assert checked["drift"][
        "auxiliary_schema_files_without_individual_prior_hashes"
    ] == ["model-manifest.schema.json", "publication-matrix.schema.json"]
    assert checked["drift"]["changed_example_files"] == []
    assert checked["drift"]["semantic_artifact_changed"] is False
    assert checked["drift"]["candidate_anchor_semantic_harness_passed"] is True
    assert checked["drift"][
        "candidate_anchor_semantic_harness_independently_replayed"
    ] is False
    assert checked["reference_semantic_replay"]["all_pass"] is True
    assert checked["reference_semantic_replay"][
        "generated_by_evaluated_system"
    ] is True
    assert checked["reference_semantic_replay"][
        "independent_review_eligible"
    ] is False
    assert checked["decision"]["reference_semantic_replay_passed"] is True
    assert checked["decision"]["activation_eligible"] is False
    assert all(value is False for value in checked["authority"].values())


def test_candidate_rejects_forged_semantic_completion(
    candidate: dict, candidate_root: Path
) -> None:
    forged = deepcopy(candidate)
    forged["reference_semantic_replay"]["generated_by_evaluated_system"] = False
    forged["reference_semantic_replay"]["independent_review_eligible"] = True
    forged["decision"]["semantic_review_complete"] = True
    forged["decision"]["activation_eligible"] = True
    forged["artifact_sha256"] = canonical_sha256(
        {key: value for key, value in forged.items() if key != "artifact_sha256"}
    )
    with pytest.raises(
        reconciliation.ContractReconciliationError,
        match="drifted|decision changed",
    ):
        reconciliation.validate_contract_reconciliation_candidate_v1(
            forged, root=candidate_root
        )


def test_candidate_writer_is_no_clobber(
    tmp_path: Path, candidate_root: Path
) -> None:
    output = tmp_path / "candidate.json"
    written = reconciliation.write_contract_reconciliation_candidate_v1(
        root=candidate_root, output=output
    )
    payload = json.loads(written.read_text())
    reconciliation.validate_contract_reconciliation_candidate_v1(
        payload, root=candidate_root
    )
    with pytest.raises(
        reconciliation.ContractReconciliationError, match="refusing to overwrite"
    ):
        reconciliation.write_contract_reconciliation_candidate_v1(
            root=candidate_root, output=output
        )


def test_review_loader_requires_an_external_digest_pin() -> None:
    with pytest.raises(
        review.ContractReconciliationReviewError,
        match="external review digest",
    ):
        review.load_pinned_contract_reconciliation_review_v1(
            root=ROOT, environment={}
        )


def test_review_cannot_proceed_without_exact_prior_contract_tree(
    tmp_path: Path, monkeypatch
) -> None:
    candidate = json.loads((ROOT / reconciliation.DEFAULT_OUTPUT).read_text())
    binding = {
        "locator": reconciliation.DEFAULT_OUTPUT.as_posix(),
        "raw_sha256": review.CANDIDATE_RAW_SHA256,
        "artifact_sha256": review.CANDIDATE_ARTIFACT_SHA256,
        "created_at_utc": candidate["created_at_utc"],
        "prior_contract_tree_sha256": candidate["active_trust_root"][
            "contract_tree_sha256"
        ],
        "candidate_contract_tree_sha256": candidate["current_contracts"][
            "contract_tree_sha256"
        ],
    }
    monkeypatch.setattr(review, "_candidate", lambda _root: (candidate, binding))
    payload = {
        "schema_version": review.SCHEMA_VERSION,
        "registry_id": "review-fixture",
        "status": "INDEPENDENT_REVIEW_REGISTERED_PENDING_ACTIVATION",
        "registered_at_utc": "2026-08-02T13:00:00+00:00",
        "candidate_binding": binding,
        "prior_contract_tree_evidence": {
            "root_locator": review.PRIOR_TREE_EVIDENCE_ROOT.as_posix(),
            "contract_tree_sha256": candidate["active_trust_root"][
                "contract_tree_sha256"
            ],
            "allowlist": list(
                reconciliation.validation.CONTRACT_SOURCE_TREE_ALLOWLIST
            ),
        },
        "semantic_replay_evidence": {},
        "reviews": [],
        "decision": {},
        "authority": {},
        "claim_ceiling": review.CLAIM_CEILING,
    }
    with pytest.raises(
        review.ContractReconciliationReviewError,
        match="prior contract-tree evidence is incomplete",
    ):
        review.validate_contract_reconciliation_review_v1(payload, root=tmp_path)


def _review_fixture(tmp_path: Path, monkeypatch) -> dict:
    candidate = json.loads((ROOT / reconciliation.DEFAULT_OUTPUT).read_text())
    binding = {
        "locator": reconciliation.DEFAULT_OUTPUT.as_posix(),
        "raw_sha256": review.CANDIDATE_RAW_SHA256,
        "artifact_sha256": review.CANDIDATE_ARTIFACT_SHA256,
        "created_at_utc": candidate["created_at_utc"],
        "prior_contract_tree_sha256": candidate["active_trust_root"][
            "contract_tree_sha256"
        ],
        "candidate_contract_tree_sha256": candidate["current_contracts"][
            "contract_tree_sha256"
        ],
    }
    monkeypatch.setattr(review, "_candidate", lambda _root: (candidate, binding))
    monkeypatch.setattr(
        review,
        "canonical_source_tree_sha256",
        lambda _root, _allowlist: candidate["active_trust_root"][
            "contract_tree_sha256"
        ],
    )
    coverage = review._expected_semantic_coverage(ROOT)
    monkeypatch.setattr(review, "_expected_semantic_coverage", lambda _root: coverage)
    evidence_root = tmp_path / review.REPLAY_EVIDENCE_PREFIX
    evidence_root.mkdir(parents=True)
    runner_path = evidence_root / "independent-runner.py"
    environment_path = evidence_root / "environment-lock.txt"
    runner_path.write_text("# independent replay fixture\n")
    environment_path.write_text("python=test-fixture\n")
    runner_locator = runner_path.relative_to(tmp_path).as_posix()
    environment_locator = environment_path.relative_to(tmp_path).as_posix()
    runner_raw = runner_path.read_bytes()
    environment_raw = environment_path.read_bytes()
    report = {
        "schema_version": (
            "scryglass:contract-validation-candidate-semantic-replay:v1"
        ),
        "report_id": "independent-semantic-replay-fixture",
        "executed_at_utc": "2026-08-02T12:00:00+00:00",
        "candidate_binding": binding,
        "runner_provenance": {
            "implementation_id": "independent-runner-fixture",
            "source_locator": runner_locator,
            "source_raw_sha256": review._sha256(runner_raw),
            "environment_lock_locator": environment_locator,
            "environment_lock_raw_sha256": review._sha256(environment_raw),
            "generated_by_evaluated_system": False,
        },
        "coverage": coverage,
        "authority": {
            "production_model_authority": False,
            "probability_authority": False,
            "betting_authority": False,
        },
    }
    report["report_sha256"] = canonical_sha256(report)
    report_path = evidence_root / "semantic-replay.json"
    report_path.write_text(json.dumps(report, sort_keys=True) + "\n")
    return {
        "schema_version": review.SCHEMA_VERSION,
        "registry_id": "independent-review-fixture",
        "status": "INDEPENDENT_REVIEW_REGISTERED_PENDING_ACTIVATION",
        "registered_at_utc": "2026-08-02T13:00:00+00:00",
        "candidate_binding": binding,
        "prior_contract_tree_evidence": {
            "root_locator": review.PRIOR_TREE_EVIDENCE_ROOT.as_posix(),
            "contract_tree_sha256": candidate["active_trust_root"][
                "contract_tree_sha256"
            ],
            "allowlist": list(
                reconciliation.validation.CONTRACT_SOURCE_TREE_ALLOWLIST
            ),
        },
        "semantic_replay_evidence": {
            "locator": report_path.relative_to(tmp_path).as_posix(),
            "raw_sha256": review._sha256(report_path.read_bytes()),
            "report_sha256": report["report_sha256"],
        },
        "reviews": [
            {
                "scope": scope,
                "reviewer_id": f"reviewer-{index}",
                "reviewed_at_utc": "2026-08-02T12:30:00+00:00",
                "attestation": attestation,
            }
            for index, (scope, attestation) in enumerate(
                review.REVIEW_SCOPES.items(), start=1
            )
        ],
        "decision": {
            "candidate_independently_reviewed": True,
            "prior_contract_tree_replayed": True,
            "candidate_semantic_harness_passed": True,
            "active_trust_root_changed": False,
            "separate_activation_required": True,
        },
        "authority": dict(review.AUTHORITY),
        "claim_ceiling": review.CLAIM_CEILING,
    }


def test_review_registry_accepts_complete_independent_evidence(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _review_fixture(tmp_path, monkeypatch)
    checked = review.validate_contract_reconciliation_review_v1(
        payload, root=tmp_path
    )
    assert checked["decision"]["candidate_independently_reviewed"] is True
    assert checked["decision"]["active_trust_root_changed"] is False
    assert checked["authority"]["contract_reconciliation_review_authority"] is True
    assert checked["authority"]["contract_trust_root_activation_authority"] is False
    assert checked["authority"]["betting_authority"] is False


def test_review_registry_rejects_same_reviewer_twice(
    tmp_path: Path, monkeypatch
) -> None:
    payload = _review_fixture(tmp_path, monkeypatch)
    payload["reviews"][1]["reviewer_id"] = payload["reviews"][0]["reviewer_id"]
    with pytest.raises(
        review.ContractReconciliationReviewError,
        match="reviewer independence",
    ):
        review.validate_contract_reconciliation_review_v1(
            payload, root=tmp_path
        )
