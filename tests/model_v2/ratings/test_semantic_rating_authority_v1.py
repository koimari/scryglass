from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.ratings import semantic_rating_authority_v1 as authority


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _bindings() -> dict:
    return {
        "phase_one_evaluation": {
            "registry_locator": "data/lol/private_market_authority/phase-one-evaluation-registry-v1.json",
            "registry_raw_sha256": "1" * 64,
            "registry_id": "phase-one-result-review-1",
            "result_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/evaluations/result.json",
            "result_raw_sha256": "2" * 64,
            "result_artifact_sha256": "3" * 64,
            "run_id": "phase-one-run-1",
            "snapshot_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/snapshots/snapshot.json",
            "snapshot_raw_sha256": "4" * 64,
            "snapshot_artifact_sha256": "5" * 64,
            "maps": 400,
            "series": 180,
            "ratings_primary_gate_passed": True,
            "ratings_reliability_gate_passed": True,
            "ratings_future_evaluation_independently_passed": True,
        },
        "evaluated_rating_runtime": {
            "runtime_locator": authority.ratings_ledger.SOURCE_LOCATOR,
            "runtime_raw_sha256": "7" * 64,
            "runtime_bytes": 200,
            "lineage_sha256": "8" * 64,
            "protocol": "multileague-v3-future-evaluation-v1",
            "source_snapshot": {
                "locator": "data/lol/v2/source-snapshots/example.json",
                "raw_sha256": "9" * 64,
            },
            "source_preflight": {
                "status": "PASS",
                "checked_at_utc": "2026-08-01T00:00:00+00:00",
            },
            "source_locks_sha256": "a" * 64,
            "model_ids": ["player", "team"],
            "prediction_receipts": 10,
            "same_exact_runtime_lineage_used_for_every_evaluated_prediction": True,
            "player_and_team_outputs_are_joint_views_of_one_runtime": True,
        },
        "production_sources": [
            {
                "locator": authority.SOURCE_LOCATOR,
                "bytes": 100,
                "raw_sha256": "6" * 64,
            }
        ],
        "reviewer_ids_excluded_from_deployment_authority": [
            "ratings-result-reviewer"
        ],
    }


def _artifacts(bindings: dict, *, model_sha: str | None = None) -> dict:
    phase_one = bindings["phase_one_evaluation"]
    runtime = bindings["evaluated_rating_runtime"]
    runtime_reference = {
        "locator": runtime["runtime_locator"],
        "raw_sha256": model_sha or runtime["runtime_raw_sha256"],
    }
    result = {
        "locator": phase_one["result_locator"],
        "raw_sha256": phase_one["result_raw_sha256"],
    }
    return {
        "source_snapshot": {
            "locator": phase_one["snapshot_locator"],
            "raw_sha256": phase_one["snapshot_raw_sha256"],
        },
        "player_model": dict(runtime_reference),
        "team_model": dict(runtime_reference),
        "evaluation": dict(result),
        "reliability": dict(result),
        "uncertainty": dict(result),
    }


def _receipt(bindings: dict, artifacts: dict | None = None) -> dict:
    return {
        "schema_version": authority.SCHEMA_VERSION,
        "authority_id": "semantic-rating-authority-1",
        "status": "APPROVED",
        "scope": "PRIVATE_PLAYER_TEAM_RATINGS_ONLY",
        "issued_at_utc": "2026-08-02T12:00:00+00:00",
        "valid_until_utc": "2026-08-16T12:00:00+00:00",
        "reviews": [
            {
                "review_scope": scope,
                "reviewer_id": f"rating-deployment-reviewer-{index}",
                "reviewed_at_utc": "2026-08-02T11:00:00+00:00",
                "attestation": attestation,
            }
            for index, (scope, attestation) in enumerate(
                authority.REVIEW_SCOPES.items(), start=1
            )
        ],
        "bindings": bindings,
        "deployment_artifacts": artifacts or _artifacts(bindings),
        "deployment_policy": dict(authority.DEPLOYMENT_POLICY),
        "authority": dict(authority.AUTHORITY),
        "claim_ceiling": authority.CLAIM_CEILING,
    }


def test_semantic_rating_authority_requires_independent_deployment_reviews() -> None:
    bindings = _bindings()
    receipt = _receipt(bindings)
    checked = authority.validate_semantic_rating_authority_v1(
        receipt, expected_bindings=bindings
    )
    assert checked["authority"]["private_player_rating_authority"] is True
    assert checked["authority"]["private_team_rating_authority"] is True
    assert checked["authority"]["match_probability_authority"] is False
    assert checked["authority"]["betting_authority"] is False

    forged = deepcopy(receipt)
    forged["reviews"][0]["reviewer_id"] = "ratings-result-reviewer"
    with pytest.raises(
        authority.SemanticRatingAuthorityError, match="not independent"
    ):
        authority.validate_semantic_rating_authority_v1(
            forged, expected_bindings=bindings
        )


def test_semantic_rating_authority_requires_evaluated_artifact_lineage() -> None:
    bindings = _bindings()
    forged = _receipt(bindings)
    forged["deployment_artifacts"]["evaluation"] = {
        "locator": "data/lol/private_rating_authority/evidence/claimed.json",
        "raw_sha256": "8" * 64,
    }
    with pytest.raises(
        authority.SemanticRatingAuthorityError,
        match="not the independently registered result",
    ):
        authority.validate_semantic_rating_authority_v1(
            forged, expected_bindings=bindings
        )


def test_semantic_rating_authority_rejects_arbitrary_rating_model_files() -> None:
    bindings = _bindings()
    forged = _receipt(bindings)
    forged["deployment_artifacts"]["player_model"] = {
        "locator": "data/lol/v2/models/player/promoted-v1.json",
        "raw_sha256": "b" * 64,
    }
    with pytest.raises(
        authority.SemanticRatingAuthorityError,
        match="not the exact future-evaluated runtime",
    ):
        authority.validate_semantic_rating_authority_v1(
            forged, expected_bindings=bindings
        )


def test_evaluated_runtime_binding_rejects_mixed_rating_lineages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = {
        "ratings_ledger_candidate": {
            "entries": [
                {"receipt_locator": "first.json"},
                {"receipt_locator": "second.json"},
            ]
        }
    }
    receipts = {
        b"first": {
            "protocol": {"artifact_sha256": "1" * 64},
            "source_snapshot": {"artifact_sha256": "2" * 64},
            "source_preflight": {"artifact_sha256": "3" * 64},
            "source_locks": [
                {
                    "locator": authority.ratings_ledger.SOURCE_LOCATOR,
                    "raw_sha256": "4" * 64,
                    "bytes": 100,
                }
            ],
            "evaluation_predictions": {"model-a": {}},
        },
        b"second": {
            "protocol": {"artifact_sha256": "9" * 64},
            "source_snapshot": {"artifact_sha256": "2" * 64},
            "source_preflight": {"artifact_sha256": "3" * 64},
            "source_locks": [
                {
                    "locator": authority.ratings_ledger.SOURCE_LOCATOR,
                    "raw_sha256": "4" * 64,
                    "bytes": 100,
                }
            ],
            "evaluation_predictions": {"model-a": {}},
        },
    }
    monkeypatch.setattr(
        authority.evaluation,
        "_read_regular",
        lambda _root, locator, _label: locator.removesuffix(".json").encode(),
    )
    monkeypatch.setattr(
        authority.evaluation,
        "_strict_object",
        lambda raw, _label: receipts[raw],
    )
    monkeypatch.setattr(
        authority.ratings_ledger,
        "validate_pre_event_prediction_receipt",
        lambda value, **_: value,
    )
    with pytest.raises(
        authority.SemanticRatingAuthorityError,
        match="does not bind exactly one runtime lineage",
    ):
        authority._evaluated_rating_runtime_binding(snapshot, root=tmp_path)


def test_semantic_rating_authority_cannot_expand_to_betting() -> None:
    bindings = _bindings()
    forged = _receipt(bindings)
    forged["authority"]["betting_authority"] = True
    with pytest.raises(
        authority.SemanticRatingAuthorityError, match="exceeds scope"
    ):
        authority.validate_semantic_rating_authority_v1(
            forged, expected_bindings=bindings
        )


def test_active_loader_replays_pin_validity_and_every_approved_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "snapshot": b"snapshot\n",
        "result": b"result\n",
        "model": b"model\n",
    }
    bindings = _bindings()
    phase_one = bindings["phase_one_evaluation"]
    phase_one["snapshot_raw_sha256"] = _sha(payloads["snapshot"])
    phase_one["result_raw_sha256"] = _sha(payloads["result"])
    bindings["evaluated_rating_runtime"]["runtime_raw_sha256"] = _sha(
        payloads["model"]
    )
    artifacts = _artifacts(bindings, model_sha=_sha(payloads["model"]))
    for reference in artifacts.values():
        path = tmp_path / reference["locator"]
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = (
            payloads["snapshot"]
            if reference == artifacts["source_snapshot"]
            else payloads["model"]
            if reference in (artifacts["player_model"], artifacts["team_model"])
            else payloads["result"]
        )
        path.write_bytes(raw)
    receipt = _receipt(bindings, artifacts)
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    authority_path = tmp_path / authority.AUTHORITY_LOCATOR
    authority_path.parent.mkdir(parents=True, exist_ok=True)
    authority_path.write_bytes(raw)
    monkeypatch.setattr(
        authority,
        "current_expected_bindings",
        lambda **_: bindings,
    )
    environment = {authority.EXTERNAL_SHA256_ENV: _sha(raw)}
    loaded = authority.load_active_semantic_rating_authority_v1(
        root=tmp_path,
        environment=environment,
        as_of=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )
    assert loaded["private_player_rating_authorized"] is True
    assert loaded["private_team_rating_authorized"] is True
    assert loaded["match_probability_authorized"] is False

    (tmp_path / artifacts["player_model"]["locator"]).write_bytes(b"changed\n")
    with pytest.raises(
        authority.SemanticRatingAuthorityError,
        match="approved deployment artifact changed: player_model",
    ):
        authority.load_active_semantic_rating_authority_v1(
            root=tmp_path,
            environment=environment,
            as_of=datetime(2026, 8, 3, tzinfo=timezone.utc),
        )
