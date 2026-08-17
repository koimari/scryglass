from __future__ import annotations

import json
from pathlib import Path

import pytest

from lol_kills.export.promoted_draft_authority import (
    PromotedDraftAuthorityError,
    load_promoted_draft_authority,
    validate_promoted_results_payload,
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


def test_owner_release_receipt_binds_checked_in_candidate_and_evaluation() -> None:
    root = Path(__file__).resolve().parents[1]
    path = (
        root
        / "data/lol/v2/evaluation/public-draft-score-v34-owner-release-receipt.json"
    )

    authority, receipt = load_promoted_draft_authority(
        receipt_path=path,
        expected_file_sha256=sha256_path(path),
        release_id="v2026.08.17.072859",
    )

    assert authority["status"] == "promoted"
    assert authority["model_version"] == "public-draft-score-v34"
    assert authority["receipt_sha256"] == receipt["receipt_sha256"]
    assert authority["probability_authority"] is True


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


def _promoted_payload(authority: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": "scryglass:promoted-draft-results:v1",
        "authority": "promoted",
        "release_id": authority["release_id"],
        "model_version": authority["model_version"],
        "receipt_sha256": authority["receipt_sha256"],
        "results": {
            "game-1": {
                "schema_version": "scryglass:public-draft-score-result:v1",
                "authority": "promoted",
                "release_id": authority["release_id"],
                "model_version": authority["model_version"],
                "receipt_sha256": authority["receipt_sha256"],
                "evidence_window": {
                    "start": "2025-01-01T00:00:00Z",
                    "end": "2026-08-16T00:00:00Z",
                },
                "match_win_probability": {"Blue": 0.61, "Red": 0.39},
                "controlled_draft_score": {
                    "model_units": -0.18,
                    "edge_percentage_points": -1.9,
                    "stronger_draft": "Red",
                    "explanation": "Role-matched champion swap with strength held fixed.",
                    "method": "role_matched_champion_swap",
                    "intervention_receipt_sha256": "4" * 64,
                    "isolated_blue_draft_probability": 0.455,
                    "fixed_strength_blue_win_probability": 0.56,
                },
                "side_recommendation": "Blue",
            }
        },
    }


def test_promoted_result_asset_is_release_and_receipt_bound(tmp_path: Path) -> None:
    path, file_sha256 = _receipt(tmp_path / "promotion.json")
    authority, _receipt_value = load_promoted_draft_authority(
        receipt_path=path,
        expected_file_sha256=file_sha256,
        release_id="v2026.09.01.120000",
    )

    payload = _promoted_payload(authority)

    assert validate_promoted_results_payload(payload, authority=authority) is payload


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload["results"]["game-1"].update({"side_recommendation": "Red"}), "direction"),
        (lambda payload: payload["results"]["game-1"].update({"odds": 1.7}), "asset"),
        (lambda payload: payload["results"]["game-1"].update({"internal_vector": [1, 2]}), "row"),
        (lambda payload: payload["results"]["game-1"]["controlled_draft_score"].update({"internal": 1}), "row"),
        (lambda payload: payload["results"]["game-1"]["match_win_probability"].update({"Red": 0.4}), "numbers"),
        (lambda payload: payload.update({"receipt_sha256": "9" * 64}), "asset"),
    ],
)
def test_promoted_result_asset_tampering_fails_closed(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    path, file_sha256 = _receipt(tmp_path / "promotion.json")
    authority, _receipt_value = load_promoted_draft_authority(
        receipt_path=path,
        expected_file_sha256=file_sha256,
        release_id="v2026.09.01.120000",
    )
    payload = _promoted_payload(authority)
    mutation(payload)

    with pytest.raises(PromotedDraftAuthorityError, match=message):
        validate_promoted_results_payload(payload, authority=authority)


def test_promoted_result_rejects_nonzero_edge_for_even_score(tmp_path: Path) -> None:
    path, file_sha256 = _receipt(tmp_path / "promotion.json")
    authority, _receipt_value = load_promoted_draft_authority(
        receipt_path=path,
        expected_file_sha256=file_sha256,
        release_id="v2026.09.01.120000",
    )
    payload = _promoted_payload(authority)
    score = payload["results"]["game-1"]["controlled_draft_score"]
    score["model_units"] = 0.0

    with pytest.raises(PromotedDraftAuthorityError, match="direction"):
        validate_promoted_results_payload(payload, authority=authority)
