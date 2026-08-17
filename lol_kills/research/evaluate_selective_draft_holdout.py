"""Evaluate a complete sealed Draft holdout one time."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_holdout_inventory import (
    summarize_holdout_inventory,
)
from lol_kills.research.selective_draft_probability import (
    _cluster_bootstrap_auc,
    _group_metrics,
    _metrics,
    canonical_sha256,
)


SCHEMA_VERSION = "scryglass:selective-draft-holdout-evaluation:v1"
MINIMUM_AUC = 0.710
MAXIMUM_ECE = 0.08


class SelectiveDraftHoldoutEvaluationError(ValueError):
    """Raised when the final holdout cannot be evaluated safely."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectiveDraftHoldoutEvaluationError("JSON input is not an object")
    return value


def _receipt_matches(value: Mapping[str, Any]) -> bool:
    expected = value.get("receipt_sha256")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return expected == canonical_sha256(unsigned)


def evaluate_sealed_holdout(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    candidate_path: Path,
    expected_candidate_sha256: str,
    receipt_paths: Sequence[Path],
    sealed_paths: Sequence[Path],
    paired_receipt_paths: Sequence[Path] = (),
    paired_sealed_paths: Sequence[Path] = (),
    outcomes_path: Path,
    expected_outcomes_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Join outcomes only after the blind inventory reaches every size gate."""

    if output_path.exists():
        raise SelectiveDraftHoldoutEvaluationError("evaluation output already exists")
    if len(receipt_paths) != len(sealed_paths) or not receipt_paths:
        raise SelectiveDraftHoldoutEvaluationError("batch inputs do not align")
    for expected, label in (
        (expected_protocol_sha256, "protocol"),
        (expected_candidate_sha256, "candidate"),
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SelectiveDraftHoldoutEvaluationError(f"{label} SHA-256 is invalid")
    if not protocol_path.is_file() or sha256_path(protocol_path) != expected_protocol_sha256:
        raise SelectiveDraftHoldoutEvaluationError("protocol changed")
    if (
        not candidate_path.is_file()
        or sha256_path(candidate_path) != expected_candidate_sha256
    ):
        raise SelectiveDraftHoldoutEvaluationError("candidate changed")
    candidate = _json(candidate_path)
    protocol = _json(protocol_path)
    if not _receipt_matches(candidate):
        raise SelectiveDraftHoldoutEvaluationError("candidate receipt changed")

    receipts = [_json(path) for path in receipt_paths]
    inventory = summarize_holdout_inventory(receipts)
    if inventory.get("outcomes_may_be_joined") is not True:
        raise SelectiveDraftHoldoutEvaluationError(
            "holdout inventory is incomplete; outcomes remain sealed"
        )
    if inventory.get("candidate_receipt_sha256") != candidate.get("receipt_sha256"):
        raise SelectiveDraftHoldoutEvaluationError("candidate binding changed")
    if inventory.get("protocol_file_sha256") != expected_protocol_sha256:
        raise SelectiveDraftHoldoutEvaluationError("protocol binding changed")
    holdout = protocol.get("next_holdout")
    required_gates = {
        "minimum_selected_rows": 100,
        "minimum_eligible_coverage": 0.75,
        "minimum_auc": 0.710,
        "maximum_brier_delta_vs_quantum": 0.0,
        "maximum_log_loss_delta_vs_quantum": 0.0,
        "maximum_ece_10": 0.08,
        "minimum_leagues_with_20_selected_rows": 3,
        "series_cluster_bootstrap_median_auc_minimum": 0.710,
    }
    if (
        not isinstance(holdout, dict)
        or any(holdout.get(key) != value for key, value in required_gates.items())
        or holdout.get("candidate_artifact_sha256") != expected_candidate_sha256
        or holdout.get("candidate_receipt_sha256") != candidate.get("receipt_sha256")
    ):
        raise SelectiveDraftHoldoutEvaluationError("protocol gates changed")

    paired_receipts: list[dict[str, Any]] = []
    if protocol.get("controlled_draft_contribution") is not None:
        if (
            len(paired_receipt_paths) != len(receipt_paths)
            or len(paired_sealed_paths) != len(receipt_paths)
        ):
            raise SelectiveDraftHoldoutEvaluationError(
                "paired Draft intervention batches do not align"
            )
        for observed_receipt, paired_receipt_path, paired_path in zip(
            receipts, paired_receipt_paths, paired_sealed_paths
        ):
            paired_receipt = _json(paired_receipt_path)
            if not _receipt_matches(paired_receipt):
                raise SelectiveDraftHoldoutEvaluationError(
                    "paired Draft intervention receipt changed"
                )
            if (
                paired_receipt.get("schema_version")
                != "scryglass:sealed-controlled-draft-interventions:v1"
                or paired_receipt.get("protocol_file_sha256")
                != expected_protocol_sha256
                or paired_receipt.get("candidate_receipt_sha256")
                != candidate.get("receipt_sha256")
                or paired_receipt.get("game_ids") != observed_receipt.get("game_ids")
                or paired_receipt.get("outcome_blind") is not True
                or paired_receipt.get("outcomes_opened") is not False
                or paired_receipt.get("public_probability") is not False
                or paired_receipt.get("public_recommendation") is not False
                or not paired_path.is_file()
                or sha256_path(paired_path) != paired_receipt.get("output_sha256")
            ):
                raise SelectiveDraftHoldoutEvaluationError(
                    "paired Draft intervention binding changed"
                )
            paired_receipts.append(paired_receipt)

    batches: list[pd.DataFrame] = []
    for receipt, sealed_path in zip(receipts, sealed_paths):
        if (
            not sealed_path.is_file()
            or sha256_path(sealed_path) != receipt.get("output_sha256")
        ):
            raise SelectiveDraftHoldoutEvaluationError("sealed batch changed")
        frame = pd.read_parquet(sealed_path)
        if frame["game_uid"].astype(str).tolist() != receipt.get("game_ids"):
            raise SelectiveDraftHoldoutEvaluationError("sealed game inventory changed")
        batches.append(frame)
    predictions = pd.concat(batches, ignore_index=True)
    if predictions["game_uid"].duplicated().any():
        raise SelectiveDraftHoldoutEvaluationError("sealed games overlap")

    if not re.fullmatch(r"[0-9a-f]{64}", expected_outcomes_sha256):
        raise SelectiveDraftHoldoutEvaluationError("outcome SHA-256 is invalid")
    if (
        not outcomes_path.is_file()
        or sha256_path(outcomes_path) != expected_outcomes_sha256
    ):
        raise SelectiveDraftHoldoutEvaluationError("outcomes changed")
    outcomes = pd.read_parquet(outcomes_path)
    if set(outcomes.columns) != {"game_uid", "y"}:
        raise SelectiveDraftHoldoutEvaluationError("outcome columns are not minimal")
    outcomes["game_uid"] = outcomes["game_uid"].astype(str)
    if outcomes["game_uid"].duplicated().any():
        raise SelectiveDraftHoldoutEvaluationError("outcome games overlap")
    if set(outcomes["game_uid"]) != set(predictions["game_uid"].astype(str)):
        raise SelectiveDraftHoldoutEvaluationError("outcome inventory changed")
    target = pd.to_numeric(outcomes["y"], errors="raise")
    if not target.isin((0, 1)).all():
        raise SelectiveDraftHoldoutEvaluationError("outcomes are not binary")
    outcomes["y"] = target.astype(int)

    evaluated = predictions.merge(outcomes, on="game_uid", validate="one_to_one")
    selected = evaluated.loc[evaluated["probability_authorized"]].copy()
    metrics = _metrics(selected)
    quantum = selected.copy()
    quantum["ensemble_probability"] = quantum["quantum"]
    quantum_metrics = _metrics(quantum)
    leagues = _group_metrics(selected, "league", minimum_rows=20)
    bootstrap = _cluster_bootstrap_auc(selected)
    gates = {
        "minimum_selected_rows": len(selected) >= 100,
        "coverage": len(selected) / len(evaluated) >= 0.75,
        "both_outcomes": selected["y"].nunique() == 2,
        "auc": metrics["auc"] > MINIMUM_AUC,
        "brier": metrics["brier"] <= quantum_metrics["brier"],
        "log_loss": metrics["log_loss"] <= quantum_metrics["log_loss"],
        "ece": metrics["ece_10"] <= MAXIMUM_ECE,
        "regional_coverage": len(leagues) >= 3,
        "bootstrap_median_auc": bootstrap["median"] > MINIMUM_AUC,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "independent_promotion_receipt_required"
            if all(gates.values())
            else "promotion_gate_failed"
        ),
        "authority": "unavailable",
        "candidate_receipt_sha256": candidate["receipt_sha256"],
        "protocol_file_sha256": expected_protocol_sha256,
        "inventory_receipt_sha256": inventory["receipt_sha256"],
        "batch_receipt_sha256": inventory["batch_receipt_sha256"],
        "controlled_intervention_receipt_sha256": [
            receipt["receipt_sha256"] for receipt in paired_receipts
        ],
        "outcomes_sha256": expected_outcomes_sha256,
        "eligible_rows": len(evaluated),
        "selected_rows": len(selected),
        "coverage": len(selected) / len(evaluated),
        "metrics": metrics,
        "same_rows_quantum_baseline": quantum_metrics,
        "leagues": leagues,
        "series_bootstrap_auc": bootstrap,
        "gates": {**gates, "passed": all(gates.values())},
        "public_probability": False,
        "public_recommendation": False,
        "betting_odds_ev_stake": False,
    }
    report["receipt_sha256"] = canonical_sha256(report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    parser.add_argument("--sealed", type=Path, action="append", required=True)
    parser.add_argument("--paired-receipt", type=Path, action="append", default=[])
    parser.add_argument("--paired-sealed", type=Path, action="append", default=[])
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--outcomes-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_sealed_holdout(
        protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
        candidate_path=args.candidate,
        expected_candidate_sha256=args.candidate_sha256,
        receipt_paths=args.receipt,
        sealed_paths=args.sealed,
        paired_receipt_paths=args.paired_receipt,
        paired_sealed_paths=args.paired_sealed,
        outcomes_path=args.outcomes,
        expected_outcomes_sha256=args.outcomes_sha256,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
