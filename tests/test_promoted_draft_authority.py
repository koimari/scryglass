from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.export.promoted_draft_authority import (
    PromotedDraftAuthorityError,
    load_promoted_draft_authority,
)
from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import canonical_sha256
from lol_kills.research.verify_selective_draft_promotion import APPROVED_FIELDS


def _receipt(path: Path) -> tuple[Path, str]:
    value: dict[str, object] = {
        "schema_version": "scryglass:public-draft-score-promotion-receipt:v1",
        "status": "promoted",
        "authority": "promoted",
        "model_version": "public-draft-score-v1",
        "candidate_artifact_sha256": "a" * 64,
        "candidate_receipt_sha256": "b" * 64,
        "protocol_file_sha256": "c" * 64,
        "evaluation_file_sha256": "d" * 64,
        "evaluation_receipt_sha256": "e" * 64,
        "decision_file_sha256": "f" * 64,
        "decision_receipt_sha256": "1" * 64,
        "outcomes_sha256": "2" * 64,
        "controlled_intervention_receipt_sha256": ["3" * 64],
        "reviewer_identity": "independent-reviewer",
        "issued_utc": "2026-09-01T12:00:00Z",
        "approved_public_fields": list(APPROVED_FIELDS),
        "public_probability": True,
        "public_recommendation": True,
        "betting_odds_ev_stake": False,
    }
    value["receipt_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, sha256_path(path)


def test_verified_receipt_builds_release_bound_promoted_authority(
    tmp_path: Path,
) -> None:
    path, file_sha256 = _receipt(tmp_path / "promotion.json")

    authority, receipt = load_promoted_draft_authority(
        receipt_path=path,
        expected_file_sha256=file_sha256,
        release_id="v2026.09.01.120000",
    )

    assert authority == {
        "schema_version": "scryglass:draft-authority:v1",
        "status": "promoted",
        "authority": "promoted",
        "release_id": "v2026.09.01.120000",
        "model_version": "public-draft-score-v1",
        "artifact_sha256": "a" * 64,
        "receipt_sha256": receipt["receipt_sha256"],
        "issued_utc": "2026-09-01T12:00:00Z",
        "estimand": "prematch_map_win_probability_with_controlled_draft_intervention",
        "probability_authority": True,
        "recommendation_authority": True,
        "betting_authority": False,
        "reason": None,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("candidate_artifact_sha256", "x" * 64),
        ("controlled_intervention_receipt_sha256", []),
        ("public_probability", False),
        ("betting_odds_ev_stake", True),
        ("approved_public_fields", ["match_win_probability"]),
    ],
)
def test_changed_promotion_contract_fails_closed(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    path, _file_sha256 = _receipt(tmp_path / "promotion.json")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    receipt[field] = value
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(PromotedDraftAuthorityError, match="contract"):
        load_promoted_draft_authority(
            receipt_path=path,
            expected_file_sha256=sha256_path(path),
            release_id="v2026.09.01.120000",
        )


def test_changed_receipt_file_fails_before_parsing(tmp_path: Path) -> None:
    path, file_sha256 = _receipt(tmp_path / "promotion.json")
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(PromotedDraftAuthorityError, match="file changed"):
        load_promoted_draft_authority(
            receipt_path=path,
            expected_file_sha256=file_sha256,
            release_id="v2026.09.01.120000",
        )
