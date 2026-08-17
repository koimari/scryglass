from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import canonical_sha256
from lol_kills.research.verify_selective_draft_promotion import (
    APPROVED_FIELDS,
    SelectiveDraftPromotionVerificationError,
    verify_promotion_decision,
)


def _write_signed(path: Path, value: dict[str, object]) -> tuple[Path, str]:
    payload = dict(value)
    payload["receipt_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, sha256_path(path)


def _inputs(tmp_path: Path) -> tuple[Path, str, Path, str]:
    evaluation_path, evaluation_sha = _write_signed(
        tmp_path / "evaluation.json",
        {
            "schema_version": "scryglass:selective-draft-holdout-evaluation:v1",
            "status": "independent_promotion_receipt_required",
            "gates": {"passed": True},
            "candidate_receipt_sha256": "a" * 64,
            "protocol_file_sha256": "c" * 64,
            "outcomes_sha256": "b" * 64,
            "controlled_intervention_receipt_sha256": ["d" * 64],
            "public_probability": False,
            "public_recommendation": False,
        },
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    decision_path, decision_sha = _write_signed(
        tmp_path / "decision.json",
        {
            "schema_version": "scryglass:selective-draft-promotion-decision:v1",
            "decision": "promoted",
            "independent_from_model_development": True,
            "reviewer_identity": "independent-reviewer-1",
            "issued_utc": "2026-09-01T12:00:00Z",
            "evaluation_file_sha256": evaluation_sha,
            "evaluation_receipt_sha256": evaluation["receipt_sha256"],
            "candidate_receipt_sha256": "a" * 64,
            "outcomes_sha256": "b" * 64,
            "approved_public_fields": list(APPROVED_FIELDS),
            "model_version": "public-draft-score-v1",
            "betting_odds_ev_stake": False,
        },
    )
    return evaluation_path, evaluation_sha, decision_path, decision_sha


def test_independent_decision_creates_promoted_receipt(tmp_path: Path) -> None:
    evaluation_path, evaluation_sha, decision_path, decision_sha = _inputs(tmp_path)

    receipt = verify_promotion_decision(
        evaluation_path=evaluation_path,
        expected_evaluation_sha256=evaluation_sha,
        decision_path=decision_path,
        expected_decision_sha256=decision_sha,
        output_path=tmp_path / "promotion.json",
    )

    assert receipt["authority"] == "promoted"
    assert receipt["public_probability"] is True
    assert receipt["public_recommendation"] is True
    assert receipt["betting_odds_ev_stake"] is False
    assert receipt["controlled_intervention_receipt_sha256"] == ["d" * 64]


def test_missing_paired_intervention_evidence_fails_closed(tmp_path: Path) -> None:
    evaluation_path, _evaluation_sha, decision_path, _decision_sha = _inputs(tmp_path)
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    evaluation.pop("controlled_intervention_receipt_sha256")
    evaluation.pop("receipt_sha256")
    evaluation["receipt_sha256"] = canonical_sha256(evaluation)
    evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")

    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["evaluation_file_sha256"] = sha256_path(evaluation_path)
    decision["evaluation_receipt_sha256"] = evaluation["receipt_sha256"]
    decision.pop("receipt_sha256")
    decision["receipt_sha256"] = canonical_sha256(decision)
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(
        SelectiveDraftPromotionVerificationError,
        match="paired Draft intervention",
    ):
        verify_promotion_decision(
            evaluation_path=evaluation_path,
            expected_evaluation_sha256=sha256_path(evaluation_path),
            decision_path=decision_path,
            expected_decision_sha256=sha256_path(decision_path),
            output_path=tmp_path / "promotion.json",
        )


def test_developer_authored_decision_fails_closed(tmp_path: Path) -> None:
    evaluation_path, evaluation_sha, decision_path, _decision_sha = _inputs(tmp_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["independent_from_model_development"] = False
    decision.pop("receipt_sha256")
    decision["receipt_sha256"] = canonical_sha256(decision)
    decision_path.write_text(json.dumps(decision), encoding="utf-8")

    with pytest.raises(
        SelectiveDraftPromotionVerificationError,
        match="independent decision",
    ):
        verify_promotion_decision(
            evaluation_path=evaluation_path,
            expected_evaluation_sha256=evaluation_sha,
            decision_path=decision_path,
            expected_decision_sha256=sha256_path(decision_path),
            output_path=tmp_path / "promotion.json",
        )
