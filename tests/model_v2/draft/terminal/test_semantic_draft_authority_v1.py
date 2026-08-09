from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from lol_kills.v2.draft.terminal import semantic_draft_authority_v1 as authority
from lol_kills.v2.draft.terminal.promotion import TerminalPromotionBindings


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
            "snapshot_artifact_sha256": "4" * 64,
            "parity_locator": "data/lol/v2/evaluation/match-winner-market-v1/phase-one/parity/parity.json",
            "parity_raw_sha256": "5" * 64,
            "parity_artifact_sha256": "6" * 64,
            "maps": 400,
            "series": 180,
            "draft_primary_gate_passed": True,
            "draft_subgroup_nonharm_gate_passed": True,
            "draft_reliability_and_runtime_parity_gate_passed": True,
            "draft_future_evaluation_independently_passed": True,
        },
        "evaluated_draft_model": {
            "locator": "data/lol/v2/models/draft-terminal/promoted-v1.json",
            "raw_sha256": "8" * 64,
            "model_version": "draft-terminal-promoted-v1.0.0",
            "candidate_id": "m0-role-additive",
            "variant_id": "m0-role-additive@ridge-0.05",
            "prediction_receipts": 400,
            "same_exact_model_used_for_every_evaluated_prediction": True,
        },
        "production_sources": [
            {
                "locator": authority.SOURCE_LOCATOR,
                "bytes": 100,
                "raw_sha256": "7" * 64,
            }
        ],
        "reviewer_ids_excluded_from_deployment_authority": [
            "draft-result-reviewer"
        ],
    }


def _model(*, raw_sha256: str = "8" * 64) -> dict:
    return {
        "locator": "data/lol/v2/models/draft-terminal/promoted-v1.json",
        "raw_sha256": raw_sha256,
        "artifact_sha256": raw_sha256,
        "model_version": "draft-terminal-promoted-v1.0.0",
    }


def _receipt(bindings: dict, model: dict | None = None) -> dict:
    return {
        "schema_version": authority.SCHEMA_VERSION,
        "authority_id": "semantic-draft-authority-1",
        "status": "APPROVED",
        "scope": "PRIVATE_TERMINAL_DRAFT_COMPONENT_ONLY",
        "issued_at_utc": "2026-08-02T12:00:00+00:00",
        "valid_until_utc": "2026-08-16T12:00:00+00:00",
        "reviews": [
            {
                "review_scope": scope,
                "reviewer_id": f"draft-deployment-reviewer-{index}",
                "reviewed_at_utc": "2026-08-02T11:00:00+00:00",
                "attestation": attestation,
            }
            for index, (scope, attestation) in enumerate(
                authority.REVIEW_SCOPES.items(), start=1
            )
        ],
        "bindings": bindings,
        "deployment_model": model or _model(),
        "deployment_policy": dict(authority.DEPLOYMENT_POLICY),
        "authority": dict(authority.AUTHORITY),
        "claim_ceiling": authority.CLAIM_CEILING,
    }


def test_semantic_draft_authority_is_private_component_only() -> None:
    bindings = _bindings()
    checked = authority.validate_semantic_draft_authority_v1(
        _receipt(bindings), expected_bindings=bindings
    )
    assert checked["authority"]["private_terminal_draft_component_authority"] is True
    assert checked["authority"]["private_event_probability_authority"] is False
    assert checked["authority"]["public_probability_authority"] is False
    assert checked["authority"]["betting_authority"] is False
    assert checked["deployment_policy"][
        "combined_probability_requires_exact_semantically_authorized_rating_receipt"
    ] is True


def test_semantic_draft_authority_requires_new_independent_reviewers() -> None:
    bindings = _bindings()
    forged = _receipt(bindings)
    forged["reviews"][0]["reviewer_id"] = "draft-result-reviewer"
    with pytest.raises(
        authority.SemanticDraftAuthorityError, match="not independent"
    ):
        authority.validate_semantic_draft_authority_v1(
            forged, expected_bindings=bindings
        )


def test_semantic_draft_authority_rejects_probability_scope_expansion() -> None:
    bindings = _bindings()
    forged = _receipt(bindings)
    forged["authority"]["private_event_probability_authority"] = True
    with pytest.raises(
        authority.SemanticDraftAuthorityError, match="exceeds scope"
    ):
        authority.validate_semantic_draft_authority_v1(
            forged, expected_bindings=bindings
        )


def test_semantic_draft_authority_rejects_unevaluated_deployment_model() -> None:
    bindings = _bindings()
    forged_model = _model(raw_sha256="f" * 64)
    with pytest.raises(
        authority.SemanticDraftAuthorityError,
        match="not the exact future-evaluated",
    ):
        authority.validate_semantic_draft_authority_v1(
            _receipt(bindings, forged_model), expected_bindings=bindings
        )


def test_evaluated_model_binding_rejects_mixed_model_cohort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    snapshot = {
        "draft_ledger_candidate": {
            "entries": [
                {"prediction_locator": "prediction-a.json"},
                {"prediction_locator": "prediction-b.json"},
            ]
        }
    }
    monkeypatch.setattr(
        authority.evaluation,
        "_read_regular",
        lambda _root, locator, _label: locator.encode(),
    )
    monkeypatch.setattr(
        authority.evaluation,
        "_strict_object",
        lambda raw, _label: {"locator": raw.decode()},
    )

    def validate_prediction(payload: dict, **_kwargs: object) -> dict:
        suffix = payload["locator"].split("-")[-1].split(".")[0]
        return {
            "model": {
                "artifact_locator": (
                    f"data/lol/v2/models/draft-terminal/model-{suffix}.json"
                ),
                "artifact_raw_sha256": ("a" if suffix == "a" else "b") * 64,
                "model_version": f"draft-terminal-{suffix}-v1.0.0",
                "candidate_id": f"candidate-{suffix}",
                "variant_id": f"variant-{suffix}",
            }
        }

    monkeypatch.setattr(
        authority.draft_ledger,
        "validate_draft_prediction_receipt",
        validate_prediction,
    )
    with pytest.raises(
        authority.SemanticDraftAuthorityError,
        match="exactly one model identity",
    ):
        authority._evaluated_draft_model_binding(snapshot, root=tmp_path)


def test_active_loader_replays_external_pin_model_bytes_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bindings = _bindings()
    model_payload = {
        "model_version": "draft-terminal-promoted-v1.0.0",
        "model_as_of": "2026-07-18T16:33:48Z",
        "intercept": 0.0,
        "calibration_slope": 0.8,
        "calibration_intercept": 0.0,
        "uncertainty_logit_sd": 0.1,
        "champion_role_logit": {},
        "ally_synergy_logit": {},
        "counter_logit": {},
    }
    model_raw = (json.dumps(model_payload, sort_keys=True) + "\n").encode()
    model = _model(raw_sha256=_sha(model_raw))
    bindings["evaluated_draft_model"].update(
        {
            "raw_sha256": _sha(model_raw),
            "model_version": model["model_version"],
        }
    )
    model_path = tmp_path / model["locator"]
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(model_raw)
    receipt = _receipt(bindings, model)
    raw = (json.dumps(receipt, sort_keys=True) + "\n").encode()
    path = tmp_path / authority.AUTHORITY_LOCATOR
    path.parent.mkdir(parents=True)
    path.write_bytes(raw)
    monkeypatch.setattr(
        authority, "current_expected_bindings", lambda **_: bindings
    )
    environment = {authority.EXTERNAL_SHA256_ENV: _sha(raw)}
    loaded = authority.load_active_semantic_draft_authority_v1(
        root=tmp_path,
        environment=environment,
        as_of=datetime(2026, 8, 3, tzinfo=timezone.utc),
    )
    assert loaded["private_terminal_draft_component_authorized"] is True
    assert loaded["private_event_probability_authorized"] is False
    assert loaded["public_probability_authorized"] is False

    changed = deepcopy(model_payload)
    changed["model_version"] = "changed"
    model_path.write_text(json.dumps(changed))
    with pytest.raises(
        authority.SemanticDraftAuthorityError, match="approved Draft model changed"
    ):
        authority.load_active_semantic_draft_authority_v1(
            root=tmp_path,
            environment=environment,
            as_of=datetime(2026, 8, 3, tzinfo=timezone.utc),
        )


def test_promotion_bindings_are_derived_from_active_semantic_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        authority,
        "load_active_semantic_draft_authority_v1",
        lambda **_: {
            "receipt": {"authority_id": "semantic-draft-authority-1"},
            "receipt_raw_sha256": "a" * 64,
            "deployment_model": {
                "artifact_sha256": "b" * 64,
                "model_version": "draft-terminal-promoted-v1",
            },
        },
    )
    bindings = TerminalPromotionBindings(
        development_evaluation_sha256="1" * 64,
        candidate_registry_sha256="2" * 64,
        l2_contract_sha256="3" * 64,
    ).with_active_semantic_draft_authority(environment={})
    assert bindings.semantic_draft_authority_sha256 == "a" * 64
    assert bindings.semantic_draft_authority_id == "semantic-draft-authority-1"
    assert bindings.semantic_draft_model_artifact_sha256 == "b" * 64
    assert bindings.semantic_draft_model_version == "draft-terminal-promoted-v1"
