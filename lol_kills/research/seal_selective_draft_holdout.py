"""Seal one outcome-blind batch for the frozen selective Draft candidate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import (
    apply_selective_candidate,
    canonical_sha256,
)


SCHEMA_VERSION = "scryglass:sealed-selective-draft-holdout-batch:v1"
FORBIDDEN_EXACT = {"y", "result", "outcome", "winner", "blue_win"}
FORBIDDEN_PREFIXES = ("target_", "observed_", "final_")
OUTPUT_COLUMNS = (
    "game_uid",
    "date",
    "league",
    "series_id",
    "quantum",
    "roster",
    "identity",
    "development_composite",
    "ensemble_probability_uncalibrated",
    "ensemble_probability",
    "confidence_score",
    "probability_authorized",
)


class SelectiveDraftHoldoutSealError(ValueError):
    """Raised when a blind holdout batch cannot be sealed."""


def _verified_json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise SelectiveDraftHoldoutSealError(f"{label} SHA-256 is invalid")
    if not path.is_file() or sha256_path(path) != expected_sha256:
        raise SelectiveDraftHoldoutSealError(f"{label} changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectiveDraftHoldoutSealError(f"{label} is not an object")
    return value


def _verify_receipt(value: Mapping[str, Any], label: str) -> None:
    expected = value.get("receipt_sha256")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    if expected != canonical_sha256(unsigned):
        raise SelectiveDraftHoldoutSealError(f"{label} receipt does not match")


def seal_holdout_batch(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    candidate_path: Path,
    expected_candidate_sha256: str,
    features_path: Path,
    expected_features_sha256: str,
    voters_path: Path,
    expected_voters_sha256: str,
    voter_receipt_path: Path,
    expected_voter_receipt_sha256: str,
    batch_start: Any,
    batch_end_exclusive: Any,
    output_path: Path,
    receipt_output_path: Path,
) -> dict[str, Any]:
    """Apply one frozen candidate without reading a game result."""

    if output_path.exists() or receipt_output_path.exists():
        raise SelectiveDraftHoldoutSealError("holdout output already exists")
    protocol = _verified_json(
        protocol_path, expected_protocol_sha256, "protocol"
    )
    candidate = _verified_json(
        candidate_path, expected_candidate_sha256, "candidate"
    )
    holdout = protocol.get("next_holdout")
    if (
        not isinstance(holdout, dict)
        or holdout.get("candidate_artifact_sha256") != expected_candidate_sha256
        or holdout.get("candidate_receipt_sha256")
        != candidate.get("receipt_sha256")
    ):
        raise SelectiveDraftHoldoutSealError("protocol candidate binding changed")
    project_root = Path(__file__).resolve().parents[2]
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
    iteration = protocol.get("iteration")
    if not isinstance(iteration, dict) or any(
        iteration.get(key) != sha256_path(project_root / relative)
        for key, relative in implementations.items()
    ):
        raise SelectiveDraftHoldoutSealError("protocol implementation binding changed")
    voter_receipt = _verified_json(
        voter_receipt_path,
        expected_voter_receipt_sha256,
        "voter receipt",
    )
    _verify_receipt(voter_receipt, "voter")
    paths = (
        (features_path, expected_features_sha256, "feature batch"),
        (voters_path, expected_voters_sha256, "voter predictions"),
    )
    for path, expected, label in paths:
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise SelectiveDraftHoldoutSealError(f"{label} SHA-256 is invalid")
        if not path.is_file() or sha256_path(path) != expected:
            raise SelectiveDraftHoldoutSealError(f"{label} changed")
    if voter_receipt.get("prediction_file_sha256") != expected_voters_sha256:
        raise SelectiveDraftHoldoutSealError("voter file is not receipt-bound")
    if voter_receipt.get("outcome_blind") is not True:
        raise SelectiveDraftHoldoutSealError("voter run is not outcome-blind")

    features = pd.read_parquet(features_path)
    voters = pd.read_parquet(voters_path)
    forbidden = sorted(
        column
        for column in features.columns
        if column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES)
    )
    if forbidden:
        raise SelectiveDraftHoldoutSealError(
            f"feature batch contains forbidden fields: {forbidden}"
        )
    for frame, label in ((features, "feature"), (voters, "voter")):
        if "game_uid" not in frame or frame["game_uid"].duplicated().any():
            raise SelectiveDraftHoldoutSealError(
                f"{label} game identities are invalid"
            )
        frame["game_uid"] = frame["game_uid"].astype(str)
    feature_ids = features["game_uid"].tolist()
    if voters["game_uid"].tolist() != feature_ids:
        raise SelectiveDraftHoldoutSealError("voter game inventory changed")
    if voter_receipt.get("prediction_rows") != len(voters):
        raise SelectiveDraftHoldoutSealError("voter row count changed")

    start = pd.Timestamp(batch_start)
    end = pd.Timestamp(batch_end_exclusive)
    if start.tzinfo is None or end.tzinfo is None or end <= start:
        raise SelectiveDraftHoldoutSealError("holdout window is invalid")
    dates = pd.to_datetime(features.get("date"), utc=True, errors="raise")
    if features.empty or not dates.ge(start).all() or not dates.lt(end).all():
        raise SelectiveDraftHoldoutSealError("feature dates leave the holdout window")

    joined = features.merge(voters, on="game_uid", how="inner", validate="one_to_one")
    inference = apply_selective_candidate(candidate, joined)
    missing = [column for column in OUTPUT_COLUMNS if column not in inference]
    if missing:
        raise SelectiveDraftHoldoutSealError(
            f"sealed prediction columns are incomplete: {missing}"
        )
    sealed = inference[list(OUTPUT_COLUMNS)].copy()
    if sealed.empty or sealed["probability_authorized"].isna().any():
        raise SelectiveDraftHoldoutSealError("sealed prediction batch is empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_receipt = receipt_output_path.with_suffix(
        receipt_output_path.suffix + ".tmp"
    )
    sealed.to_parquet(temporary_output, index=False, compression="zstd")
    output_sha256 = sha256_path(temporary_output)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "waiting_for_minimum_holdout_inventory",
        "outcome_blind": True,
        "window": {
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
        },
        "candidate_receipt_sha256": candidate.get("receipt_sha256"),
        "protocol_file_sha256": expected_protocol_sha256,
        "voter_receipt_sha256": voter_receipt["receipt_sha256"],
        "input_sha256": {
            "candidate": expected_candidate_sha256,
            "features": expected_features_sha256,
            "voters": expected_voters_sha256,
            "voter_receipt": expected_voter_receipt_sha256,
        },
        "output_sha256": output_sha256,
        "rows": len(sealed),
        "game_ids": feature_ids,
        "game_ids_sha256": canonical_sha256(feature_ids),
        "selected_rows": int(sealed["probability_authorized"].sum()),
        "coverage": float(sealed["probability_authorized"].mean()),
        "league_rows": {
            str(key): int(value)
            for key, value in sealed["league"].value_counts().sort_index().items()
        },
        "selected_league_rows": {
            str(key): int(value)
            for key, value in sealed.loc[
                sealed["probability_authorized"], "league"
            ].value_counts().sort_index().items()
        },
        "date_min": dates.min().isoformat(),
        "date_max": dates.max().isoformat(),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output_path)
    temporary_receipt.replace(receipt_output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--candidate-sha256", required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--voters", type=Path, required=True)
    parser.add_argument("--voters-sha256", required=True)
    parser.add_argument("--voter-receipt", type=Path, required=True)
    parser.add_argument("--voter-receipt-sha256", required=True)
    parser.add_argument("--batch-start", required=True)
    parser.add_argument("--batch-end-exclusive", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    receipt = seal_holdout_batch(
        protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
        candidate_path=args.candidate,
        expected_candidate_sha256=args.candidate_sha256,
        features_path=args.features,
        expected_features_sha256=args.features_sha256,
        voters_path=args.voters,
        expected_voters_sha256=args.voters_sha256,
        voter_receipt_path=args.voter_receipt,
        expected_voter_receipt_sha256=args.voter_receipt_sha256,
        batch_start=args.batch_start,
        batch_end_exclusive=args.batch_end_exclusive,
        output_path=args.output,
        receipt_output_path=args.receipt_output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
