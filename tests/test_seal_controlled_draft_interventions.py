from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import lol_kills.research.seal_controlled_draft_interventions as paired_seal
from lol_kills.research.selective_draft_probability import canonical_sha256


def _write_receipt(path: Path, value: dict[str, object]) -> None:
    value["receipt_sha256"] = canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_paired_seal_is_outcome_blind_and_receipt_bound(
    tmp_path: Path, monkeypatch
) -> None:
    candidate_path = tmp_path / "candidate.json"
    candidate: dict[str, object] = {}
    candidate["receipt_sha256"] = canonical_sha256(candidate)
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    candidate_sha = paired_seal.sha256_path(candidate_path)
    protocol_path = tmp_path / "protocol.json"
    protocol = {
        "controlled_draft_contribution": {
            "method": "role_matched_champion_swap",
            "outcome_blind": True,
        },
        "next_holdout": {
            "candidate_artifact_sha256": candidate_sha,
            "candidate_receipt_sha256": candidate["receipt_sha256"],
        },
    }
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    protocol_sha = paired_seal.sha256_path(protocol_path)
    observed_path = tmp_path / "observed.parquet"
    observed = pd.DataFrame(
        {
            "game_uid": ["game-a", "game-b"],
            "date": pd.to_datetime(
                ["2026-08-16T01:00:00Z", "2026-08-16T02:00:00Z"]
            ),
            "league": ["LPL", "LEC"],
            "series_id": ["series-a", "series-b"],
            "ensemble_probability": [0.62, 0.43],
            "probability_authorized": [True, False],
        }
    )
    observed.to_parquet(observed_path, index=False)
    observed_receipt_path = tmp_path / "observed-receipt.json"
    _write_receipt(
        observed_receipt_path,
        {
            "output_sha256": paired_seal.sha256_path(observed_path),
            "protocol_file_sha256": protocol_sha,
            "candidate_receipt_sha256": candidate["receipt_sha256"],
            "input_sha256": {"features": "observed-features"},
            "game_ids": ["game-a", "game-b"],
        },
    )
    swap_path = tmp_path / "swap.json"
    _write_receipt(
        swap_path,
        {
            "outcome_blind": True,
            "input_sha256": {"features": "observed-features"},
            "game_ids": ["game-a", "game-b"],
            "game_receipt_sha256": {
                "game-a": "a" * 64,
                "game-b": "b" * 64,
            },
        },
    )
    features_path = tmp_path / "swapped-features.parquet"
    pd.DataFrame({"game_uid": ["game-a", "game-b"]}).to_parquet(
        features_path, index=False
    )
    voters_path = tmp_path / "swapped-voters.parquet"
    pd.DataFrame({"game_uid": ["game-a", "game-b"]}).to_parquet(
        voters_path, index=False
    )
    voter_receipt_path = tmp_path / "voter-receipt.json"
    _write_receipt(
        voter_receipt_path,
        {
            "outcome_blind": True,
            "prediction_file_sha256": paired_seal.sha256_path(voters_path),
        },
    )

    def fake_apply(_candidate, frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        output["ensemble_probability"] = [0.48, 0.51]
        output["probability_authorized"] = [True, True]
        return output

    monkeypatch.setattr(paired_seal, "apply_selective_candidate", fake_apply)
    output_path = tmp_path / "paired.parquet"
    receipt_path = tmp_path / "paired-receipt.json"
    receipt = paired_seal.seal_controlled_draft_interventions(
        protocol_path=protocol_path,
        expected_protocol_sha256=protocol_sha,
        candidate_path=candidate_path,
        expected_candidate_sha256=candidate_sha,
        observed_path=observed_path,
        expected_observed_sha256=paired_seal.sha256_path(observed_path),
        observed_receipt_path=observed_receipt_path,
        expected_observed_receipt_sha256=paired_seal.sha256_path(
            observed_receipt_path
        ),
        swap_path=swap_path,
        expected_swap_sha256=paired_seal.sha256_path(swap_path),
        swapped_features_path=features_path,
        expected_swapped_features_sha256=paired_seal.sha256_path(features_path),
        swapped_voters_path=voters_path,
        expected_swapped_voters_sha256=paired_seal.sha256_path(voters_path),
        swapped_voter_receipt_path=voter_receipt_path,
        expected_swapped_voter_receipt_sha256=paired_seal.sha256_path(
            voter_receipt_path
        ),
        output_path=output_path,
        receipt_output_path=receipt_path,
    )

    output = pd.read_parquet(output_path)
    assert receipt["outcome_blind"] is True
    assert receipt["outcomes_opened"] is False
    assert receipt["observed_selected_rows"] == 1
    assert receipt["swapped_selected_rows"] == 2
    assert output.loc[0, "stronger_draft"] == "Blue"
    assert output.loc[1, "stronger_draft"] == "Red"
    assert "y" not in output
