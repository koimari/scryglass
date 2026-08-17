"""Seal paired observed and swapped Draft predictions before outcomes open."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.research.controlled_draft_contribution import (
    isolate_controlled_draft_contribution,
)
from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import (
    apply_selective_candidate,
    canonical_sha256,
)


SCHEMA_VERSION = "scryglass:sealed-controlled-draft-interventions:v1"
FORBIDDEN_EXACT = {"y", "result", "outcome", "winner", "blue_win"}
FORBIDDEN_PREFIXES = ("target_", "observed_", "final_")


class ControlledDraftInterventionSealError(ValueError):
    """Raised when paired blind predictions cannot be sealed."""


def _json(path: Path, expected_sha256: str, label: str) -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise ControlledDraftInterventionSealError(f"{label} SHA-256 is invalid")
    if not path.is_file() or sha256_path(path) != expected_sha256:
        raise ControlledDraftInterventionSealError(f"{label} changed")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ControlledDraftInterventionSealError(f"{label} is invalid")
    return value


def _receipt_matches(value: Mapping[str, Any]) -> bool:
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return value.get("receipt_sha256") == canonical_sha256(unsigned)


def seal_controlled_draft_interventions(
    *,
    protocol_path: Path,
    expected_protocol_sha256: str,
    candidate_path: Path,
    expected_candidate_sha256: str,
    observed_path: Path,
    expected_observed_sha256: str,
    observed_receipt_path: Path,
    expected_observed_receipt_sha256: str,
    swap_path: Path,
    expected_swap_sha256: str,
    swapped_features_path: Path,
    expected_swapped_features_sha256: str,
    swapped_voters_path: Path,
    expected_swapped_voters_sha256: str,
    swapped_voter_receipt_path: Path,
    expected_swapped_voter_receipt_sha256: str,
    output_path: Path,
    receipt_output_path: Path,
) -> dict[str, Any]:
    """Bind paired outcome-blind predictions and their controlled Draft values."""

    if output_path.exists() or receipt_output_path.exists():
        raise ControlledDraftInterventionSealError("paired output already exists")
    protocol = _json(protocol_path, expected_protocol_sha256, "protocol")
    candidate = _json(candidate_path, expected_candidate_sha256, "candidate")
    observed_receipt = _json(
        observed_receipt_path,
        expected_observed_receipt_sha256,
        "observed receipt",
    )
    swap = _json(swap_path, expected_swap_sha256, "swap batch")
    voter_receipt = _json(
        swapped_voter_receipt_path,
        expected_swapped_voter_receipt_sha256,
        "swapped voter receipt",
    )
    if not all(
        _receipt_matches(value)
        for value in (candidate, observed_receipt, swap, voter_receipt)
    ):
        raise ControlledDraftInterventionSealError("input receipt changed")
    controlled = protocol.get("controlled_draft_contribution")
    holdout = protocol.get("next_holdout")
    if (
        not isinstance(controlled, dict)
        or controlled.get("method") != "role_matched_champion_swap"
        or controlled.get("outcome_blind") is not True
        or not isinstance(holdout, dict)
        or holdout.get("candidate_artifact_sha256")
        != expected_candidate_sha256
        or holdout.get("candidate_receipt_sha256")
        != candidate.get("receipt_sha256")
    ):
        raise ControlledDraftInterventionSealError(
            "protocol controlled Draft binding changed"
        )
    paths = (
        (observed_path, expected_observed_sha256, "observed predictions"),
        (
            swapped_features_path,
            expected_swapped_features_sha256,
            "swapped features",
        ),
        (swapped_voters_path, expected_swapped_voters_sha256, "swapped voters"),
    )
    for path, expected, label in paths:
        if not path.is_file() or sha256_path(path) != expected:
            raise ControlledDraftInterventionSealError(f"{label} changed")
    if (
        observed_receipt.get("output_sha256") != expected_observed_sha256
        or observed_receipt.get("protocol_file_sha256")
        != expected_protocol_sha256
        or observed_receipt.get("candidate_receipt_sha256")
        != candidate.get("receipt_sha256")
        or swap.get("outcome_blind") is not True
        or swap.get("input_sha256", {}).get("features")
        != observed_receipt.get("input_sha256", {}).get("features")
        or voter_receipt.get("outcome_blind") is not True
        or voter_receipt.get("prediction_file_sha256")
        != expected_swapped_voters_sha256
    ):
        raise ControlledDraftInterventionSealError("paired input binding changed")

    observed = pd.read_parquet(observed_path)
    features = pd.read_parquet(swapped_features_path)
    voters = pd.read_parquet(swapped_voters_path)
    forbidden = sorted(
        column
        for column in features.columns
        if column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES)
    )
    if forbidden:
        raise ControlledDraftInterventionSealError(
            f"swapped features contain forbidden fields: {forbidden}"
        )
    for frame, label in (
        (observed, "observed"),
        (features, "swapped feature"),
        (voters, "swapped voter"),
    ):
        if "game_uid" not in frame or frame["game_uid"].astype(str).duplicated().any():
            raise ControlledDraftInterventionSealError(
                f"{label} game identities are invalid"
            )
        frame["game_uid"] = frame["game_uid"].astype(str)
    game_ids = observed["game_uid"].tolist()
    if (
        features["game_uid"].tolist() != game_ids
        or voters["game_uid"].tolist() != game_ids
        or swap.get("game_ids") != game_ids
        or observed_receipt.get("game_ids") != game_ids
    ):
        raise ControlledDraftInterventionSealError("paired game inventory changed")
    swapped = apply_selective_candidate(
        candidate,
        features.merge(voters, on="game_uid", how="inner", validate="one_to_one"),
    )
    paired = observed[
        ["game_uid", "date", "league", "series_id", "ensemble_probability", "probability_authorized"]
    ].merge(
        swapped[["game_uid", "ensemble_probability", "probability_authorized"]],
        on="game_uid",
        validate="one_to_one",
        suffixes=("_observed", "_swapped"),
    )
    rows = []
    for row in paired.to_dict("records"):
        contribution = isolate_controlled_draft_contribution(
            observed_blue_win_probability=float(
                row["ensemble_probability_observed"]
            ),
            swapped_draft_blue_win_probability=float(
                row["ensemble_probability_swapped"]
            ),
        )
        rows.append(
            {
                **row,
                "controlled_model_units": contribution["model_units"],
                "controlled_edge_percentage_points": contribution[
                    "edge_percentage_points"
                ],
                "stronger_draft": contribution["stronger_draft"],
                "intervention_receipt_sha256": swap[
                    "game_receipt_sha256"
                ][row["game_uid"]],
            }
        )
    output = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    output.to_parquet(temporary_output, index=False, compression="zstd")
    output_sha256 = sha256_path(temporary_output)
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "waiting_for_holdout_outcomes",
        "outcome_blind": True,
        "protocol_file_sha256": expected_protocol_sha256,
        "candidate_receipt_sha256": candidate["receipt_sha256"],
        "input_sha256": {
            "observed": expected_observed_sha256,
            "observed_receipt": expected_observed_receipt_sha256,
            "swap_batch": expected_swap_sha256,
            "swapped_features": expected_swapped_features_sha256,
            "swapped_voters": expected_swapped_voters_sha256,
            "swapped_voter_receipt": expected_swapped_voter_receipt_sha256,
        },
        "output_sha256": output_sha256,
        "rows": len(output),
        "game_ids": game_ids,
        "game_ids_sha256": canonical_sha256(game_ids),
        "observed_selected_rows": int(output["probability_authorized_observed"].sum()),
        "swapped_selected_rows": int(output["probability_authorized_swapped"].sum()),
        "outcomes_opened": False,
        "public_probability": False,
        "public_recommendation": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    temporary_receipt = receipt_output_path.with_suffix(
        receipt_output_path.suffix + ".tmp"
    )
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_output.replace(output_path)
    temporary_receipt.replace(receipt_output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "protocol",
        "candidate",
        "observed",
        "observed-receipt",
        "swap",
        "swapped-features",
        "swapped-voters",
        "swapped-voter-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--{name}-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    receipt = seal_controlled_draft_interventions(
        protocol_path=args.protocol,
        expected_protocol_sha256=args.protocol_sha256,
        candidate_path=args.candidate,
        expected_candidate_sha256=args.candidate_sha256,
        observed_path=args.observed,
        expected_observed_sha256=args.observed_sha256,
        observed_receipt_path=args.observed_receipt,
        expected_observed_receipt_sha256=args.observed_receipt_sha256,
        swap_path=args.swap,
        expected_swap_sha256=args.swap_sha256,
        swapped_features_path=args.swapped_features,
        expected_swapped_features_sha256=args.swapped_features_sha256,
        swapped_voters_path=args.swapped_voters,
        expected_swapped_voters_sha256=args.swapped_voters_sha256,
        swapped_voter_receipt_path=args.swapped_voter_receipt,
        expected_swapped_voter_receipt_sha256=args.swapped_voter_receipt_sha256,
        output_path=args.output,
        receipt_output_path=args.receipt_output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
