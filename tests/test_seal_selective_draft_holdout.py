from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import lol_kills.research.seal_selective_draft_holdout as seal
from lol_kills.research.selective_draft_probability import canonical_sha256


def _sha(path: Path) -> str:
    return seal.sha256_path(path)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    candidate = {
        "schema_version": "test",
        "receipt_sha256": "candidate-receipt",
    }
    candidate_path = tmp_path / "candidate.json"
    candidate_path.write_text(json.dumps(candidate))
    root = Path(__file__).resolve().parents[1]
    implementations = {
        "implementation_sha256": "lol_kills/research/selective_draft_probability.py",
        "constituent_implementation_sha256": "lol_kills/research/selective_draft_constituents.py",
        "quantum_implementation_sha256": "lol_kills/research/public_draft_score_promotion.py",
        "draft_builder_implementation_sha256": "lol_kills/draft_recommendation.py",
        "holdout_source_preparer_sha256": "lol_kills/research/prepare_selective_draft_holdout_sources.py",
        "holdout_sealer_sha256": "lol_kills/research/seal_selective_draft_holdout.py",
        "holdout_inventory_sha256": "lol_kills/research/selective_draft_holdout_inventory.py",
        "holdout_evaluator_sha256": "lol_kills/research/evaluate_selective_draft_holdout.py",
        "promotion_verifier_sha256": "lol_kills/research/verify_selective_draft_promotion.py",
        "public_result_builder_sha256": "lol_kills/export/public_draft_score_result.py",
        "controlled_contribution_sha256": "lol_kills/research/controlled_draft_contribution.py",
        "paired_public_result_builder_sha256": "lol_kills/export/paired_public_draft_score.py",
    }
    protocol = {
        "iteration": {
            key: _sha(root / relative) for key, relative in implementations.items()
        },
        "next_holdout": {
            "candidate_artifact_sha256": _sha(candidate_path),
            "candidate_receipt_sha256": candidate["receipt_sha256"],
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol))
    features = pd.DataFrame(
        {
            "game_uid": ["game-a", "game-b"],
            "date": pd.to_datetime(
                ["2026-08-16T01:00:00Z", "2026-08-16T02:00:00Z"]
            ),
            "league": ["LPL", "LEC"],
            "series_id": ["series-a", "series-b"],
        }
    )
    features_path = tmp_path / "features.parquet"
    features.to_parquet(features_path, index=False)
    voters = pd.DataFrame(
        {
            "game_uid": ["game-a", "game-b"],
            "quantum": [0.6, 0.4],
            "roster": [0.6, 0.4],
            "identity": [0.6, 0.4],
            "development_composite": [0.6, 0.4],
        }
    )
    voters_path = tmp_path / "voters.parquet"
    voters.to_parquet(voters_path, index=False)
    voter_receipt = {
        "outcome_blind": True,
        "prediction_rows": 2,
        "prediction_file_sha256": _sha(voters_path),
    }
    voter_receipt["receipt_sha256"] = canonical_sha256(voter_receipt)
    voter_receipt_path = tmp_path / "voter-receipt.json"
    voter_receipt_path.write_text(json.dumps(voter_receipt))
    return {
        "protocol": protocol_path,
        "candidate": candidate_path,
        "features": features_path,
        "voters": voters_path,
        "voter_receipt": voter_receipt_path,
    }


def _fake_apply(_candidate: object, frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["ensemble_probability_uncalibrated"] = [0.6, 0.4]
    output["ensemble_probability"] = [0.62, 0.38]
    output["confidence_score"] = [1.0, -1.0]
    output["probability_authorized"] = [True, False]
    return output


def test_seal_holdout_batch_is_outcome_blind_and_receipt_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_inputs(tmp_path)
    monkeypatch.setattr(seal, "apply_selective_candidate", _fake_apply)
    output = tmp_path / "sealed.parquet"
    receipt_output = tmp_path / "sealed.receipt.json"

    receipt = seal.seal_holdout_batch(
        protocol_path=paths["protocol"],
        expected_protocol_sha256=_sha(paths["protocol"]),
        candidate_path=paths["candidate"],
        expected_candidate_sha256=_sha(paths["candidate"]),
        features_path=paths["features"],
        expected_features_sha256=_sha(paths["features"]),
        voters_path=paths["voters"],
        expected_voters_sha256=_sha(paths["voters"]),
        voter_receipt_path=paths["voter_receipt"],
        expected_voter_receipt_sha256=_sha(paths["voter_receipt"]),
        batch_start="2026-08-16T00:00:00Z",
        batch_end_exclusive="2026-08-17T00:00:00Z",
        output_path=output,
        receipt_output_path=receipt_output,
    )

    sealed = pd.read_parquet(output)
    assert sealed.columns.tolist() == list(seal.OUTPUT_COLUMNS)
    assert receipt["outcome_blind"] is True
    assert receipt["selected_rows"] == 1
    assert receipt["coverage"] == 0.5
    assert receipt["game_ids"] == ["game-a", "game-b"]
    assert receipt["game_ids_sha256"] == canonical_sha256(receipt["game_ids"])
    assert receipt["output_sha256"] == _sha(output)
    assert receipt["receipt_sha256"] == canonical_sha256(
        {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    )


@pytest.mark.parametrize("field", ["y", "target_gold_diff_10", "final_gold"])
def test_seal_holdout_batch_rejects_result_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    paths = _write_inputs(tmp_path)
    features = pd.read_parquet(paths["features"])
    features[field] = 1
    features.to_parquet(paths["features"], index=False)
    monkeypatch.setattr(seal, "apply_selective_candidate", _fake_apply)

    with pytest.raises(
        seal.SelectiveDraftHoldoutSealError, match="forbidden fields"
    ):
        seal.seal_holdout_batch(
            protocol_path=paths["protocol"],
            expected_protocol_sha256=_sha(paths["protocol"]),
            candidate_path=paths["candidate"],
            expected_candidate_sha256=_sha(paths["candidate"]),
            features_path=paths["features"],
            expected_features_sha256=_sha(paths["features"]),
            voters_path=paths["voters"],
            expected_voters_sha256=_sha(paths["voters"]),
            voter_receipt_path=paths["voter_receipt"],
            expected_voter_receipt_sha256=_sha(paths["voter_receipt"]),
            batch_start="2026-08-16T00:00:00Z",
            batch_end_exclusive="2026-08-17T00:00:00Z",
            output_path=tmp_path / "sealed.parquet",
            receipt_output_path=tmp_path / "sealed.receipt.json",
        )
