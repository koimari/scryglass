from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from lol_kills.research.evaluate_selective_draft_holdout import (
    SelectiveDraftHoldoutEvaluationError,
    evaluate_sealed_holdout,
)
from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import canonical_sha256


def _signed(value: dict[str, object]) -> dict[str, object]:
    output = dict(value)
    output["receipt_sha256"] = canonical_sha256(output)
    return output


def _candidate(path: Path) -> tuple[Path, str, str]:
    value = _signed(
        {
            "schema_version": "scryglass:selective-draft-probability-candidate:v1",
            "evidence": {"end_exclusive": "2026-08-16T00:00:00Z"},
        }
    )
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, sha256_path(path), str(value["receipt_sha256"])


def _protocol(
    path: Path, candidate_sha: str, candidate_receipt: str
) -> tuple[Path, str]:
    value = {
        "next_holdout": {
            "candidate_artifact_sha256": candidate_sha,
            "candidate_receipt_sha256": candidate_receipt,
            "minimum_selected_rows": 100,
            "minimum_eligible_coverage": 0.75,
            "minimum_auc": 0.710,
            "maximum_brier_delta_vs_quantum": 0.0,
            "maximum_log_loss_delta_vs_quantum": 0.0,
            "maximum_ece_10": 0.08,
            "minimum_leagues_with_20_selected_rows": 3,
            "series_cluster_bootstrap_median_auc_minimum": 0.710,
        }
    }
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, sha256_path(path)


def _receipt(
    candidate_receipt: str,
    protocol_sha: str,
    game_ids: list[str],
    selected: int,
) -> dict[str, object]:
    leagues = {"LCK": 40, "LEC": 40, "LPL": 40}
    return _signed(
        {
            "schema_version": "scryglass:sealed-selective-draft-holdout-batch:v1",
            "outcome_blind": True,
            "window": {
                "start": "2026-08-16T00:00:00+00:00",
                "end_exclusive": "2026-09-01T00:00:00+00:00",
            },
            "candidate_receipt_sha256": candidate_receipt,
            "protocol_file_sha256": protocol_sha,
            "rows": len(game_ids),
            "selected_rows": selected,
            "coverage": selected / len(game_ids),
            "game_ids": game_ids,
            "game_ids_sha256": canonical_sha256(game_ids),
            "league_rows": leagues,
            "selected_league_rows": leagues if selected == len(game_ids) else {},
            "output_sha256": "0" * 64,
        }
    )


def test_outcomes_remain_unread_before_inventory_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path, candidate_sha, candidate_receipt = _candidate(
        tmp_path / "candidate.json"
    )
    protocol_path, protocol_sha = _protocol(
        tmp_path / "protocol.json", candidate_sha, candidate_receipt
    )
    receipt = _receipt(
        candidate_receipt,
        protocol_sha,
        [f"game-{index}" for index in range(14)],
        13,
    )
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.setattr(pd, "read_parquet", lambda *_args, **_kwargs: pytest.fail("read"))

    with pytest.raises(
        SelectiveDraftHoldoutEvaluationError, match="outcomes remain sealed"
    ):
        evaluate_sealed_holdout(
            protocol_path=protocol_path,
            expected_protocol_sha256=protocol_sha,
            candidate_path=candidate_path,
            expected_candidate_sha256=candidate_sha,
            receipt_paths=[receipt_path],
            sealed_paths=[tmp_path / "sealed.parquet"],
            outcomes_path=tmp_path / "outcomes.parquet",
            expected_outcomes_sha256="0" * 64,
            output_path=tmp_path / "evaluation.json",
        )


def test_ready_inventory_produces_independent_review_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_path, candidate_sha, candidate_receipt = _candidate(
        tmp_path / "candidate.json"
    )
    protocol_path, protocol_sha = _protocol(
        tmp_path / "protocol.json", candidate_sha, candidate_receipt
    )
    game_ids = [f"game-{index:03d}" for index in range(120)]
    leagues = ["LCK", "LEC", "LPL"]
    outcomes = pd.DataFrame(
        {"game_uid": game_ids, "y": [index % 2 for index in range(120)]}
    )
    probability = [0.05 if value == 0 else 0.95 for value in outcomes["y"]]
    sealed = pd.DataFrame(
        {
            "game_uid": game_ids,
            "date": ["2026-08-17T12:00:00Z"] * 120,
            "league": [leagues[index // 40] for index in range(120)],
            "series_id": [f"series-{index // 2:03d}" for index in range(120)],
            "quantum": [0.4 if value == 0 else 0.6 for value in outcomes["y"]],
            "roster": probability,
            "identity": probability,
            "development_composite": probability,
            "ensemble_probability_uncalibrated": probability,
            "ensemble_probability": probability,
            "confidence_score": [1.0] * 120,
            "probability_authorized": [True] * 120,
        }
    )
    sealed_path = tmp_path / "sealed.parquet"
    sealed.to_parquet(sealed_path, index=False)
    receipt = _receipt(candidate_receipt, protocol_sha, game_ids, 120)
    receipt["output_sha256"] = sha256_path(sealed_path)
    receipt.pop("receipt_sha256")
    receipt = _signed(receipt)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    outcome_path = tmp_path / "outcomes.parquet"
    outcomes.to_parquet(outcome_path, index=False)
    monkeypatch.setattr(
        "lol_kills.research.evaluate_selective_draft_holdout._cluster_bootstrap_auc",
        lambda _frame: {"repetitions": 2000, "median": 1.0, "lower_95": 1.0, "upper_95": 1.0},
    )

    report = evaluate_sealed_holdout(
        protocol_path=protocol_path,
        expected_protocol_sha256=protocol_sha,
        candidate_path=candidate_path,
        expected_candidate_sha256=candidate_sha,
        receipt_paths=[receipt_path],
        sealed_paths=[sealed_path],
        outcomes_path=outcome_path,
        expected_outcomes_sha256=sha256_path(outcome_path),
        output_path=tmp_path / "evaluation.json",
    )

    assert report["gates"]["passed"] is True
    assert report["status"] == "independent_promotion_receipt_required"
    assert report["authority"] == "unavailable"
    assert report["public_probability"] is False
    assert report["metrics"]["auc"] == 1.0
