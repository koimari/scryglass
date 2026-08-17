"""Build a release-bound promoted Draft Score result asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.export.paired_public_draft_score import (
    build_paired_public_draft_score_result,
)
from lol_kills.export.promoted_draft_authority import (
    load_promoted_draft_authority,
    validate_promoted_results_payload,
)
from lol_kills.research.public_draft_score_promotion import sha256_path


def build_promoted_results(
    *,
    release_id: str,
    receipt_path: Path,
    observed_rows_path: Path,
    paired_predictions_path: Path,
    swapped_rows_path: Path,
) -> dict[str, Any]:
    """Return the fixed result asset from outcome-blind paired inputs."""

    authority, receipt = load_promoted_draft_authority(
        receipt_path=receipt_path,
        expected_file_sha256=sha256_path(receipt_path),
        release_id=release_id,
    )
    evidence = receipt["evidence_window"]
    observed = pd.read_parquet(observed_rows_path)
    paired = pd.read_parquet(paired_predictions_path)
    swapped_payload = json.loads(swapped_rows_path.read_text(encoding="utf-8"))
    swapped_by_game = swapped_payload.get("games")
    if not isinstance(swapped_by_game, dict):
        raise ValueError("swapped Draft rows are invalid")

    results: dict[str, Any] = {}
    for prediction in paired.to_dict("records"):
        game_uid = str(prediction["game_uid"])
        observed_rows = observed.loc[
            observed["game_uid"].astype(str) == game_uid
        ].to_dict("records")
        swapped_rows = swapped_by_game.get(game_uid)
        if len(observed_rows) != 10 or not isinstance(swapped_rows, list):
            raise ValueError(f"complete paired Draft rows are missing for {game_uid}")
        results[game_uid] = build_paired_public_draft_score_result(
            release_id=release_id,
            model_version=str(authority["model_version"]),
            promotion_receipt=receipt,
            evidence_start=str(evidence["start"]),
            evidence_end=str(evidence["end_exclusive"]),
            observed_rows=observed_rows,
            swapped_rows=swapped_rows,
            observed_blue_win_probability=float(
                prediction["ensemble_probability_observed"]
            ),
            swapped_draft_blue_win_probability=float(
                prediction["ensemble_probability_swapped"]
            ),
        )

    payload = {
        "schema_version": "scryglass:promoted-draft-results:v1",
        "authority": "promoted",
        "release_id": release_id,
        "model_version": authority["model_version"],
        "receipt_sha256": authority["receipt_sha256"],
        "results": results,
    }
    return validate_promoted_results_payload(payload, authority=authority)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--observed-rows", type=Path, required=True)
    parser.add_argument("--paired-predictions", type=Path, required=True)
    parser.add_argument("--swapped-rows", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_promoted_results(
        release_id=args.release_id,
        receipt_path=args.receipt,
        observed_rows_path=args.observed_rows,
        paired_predictions_path=args.paired_predictions,
        swapped_rows_path=args.swapped_rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
