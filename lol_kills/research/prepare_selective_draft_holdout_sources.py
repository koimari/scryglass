"""Prepare one outcome-free source batch for frozen Draft inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.seal_selective_draft_holdout import (
    FORBIDDEN_EXACT,
    FORBIDDEN_PREFIXES,
)
from lol_kills.research.selective_draft_constituents import (
    _outcome_blind_draft_source,
)
from lol_kills.research.selective_draft_probability import canonical_sha256


SCHEMA_VERSION = "scryglass:selective-draft-holdout-sources:v1"


class SelectiveDraftHoldoutSourceError(ValueError):
    """Raised when a blind inference source cannot be prepared."""


def _window(value: Any, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise SelectiveDraftHoldoutSourceError(f"{label} must include a timezone")
    return timestamp.tz_convert("UTC")


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> str:
    if path.exists():
        raise SelectiveDraftHoldoutSourceError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    digest = sha256_path(temporary)
    temporary.replace(path)
    return digest


def prepare_holdout_sources(
    *,
    feature_matrix_path: Path,
    players_path: Path,
    batch_start: Any,
    batch_end_exclusive: Any,
    feature_output_path: Path,
    player_output_path: Path,
    receipt_output_path: Path,
) -> dict[str, Any]:
    """Write feature and draft inputs that contain no game outcome."""

    if receipt_output_path.exists():
        raise SelectiveDraftHoldoutSourceError("receipt output already exists")
    if not feature_matrix_path.is_file() or not players_path.is_file():
        raise SelectiveDraftHoldoutSourceError("input source is missing")
    start = _window(batch_start, "batch start")
    end = _window(batch_end_exclusive, "batch end")
    if end <= start:
        raise SelectiveDraftHoldoutSourceError("batch window is invalid")

    matrix = pd.read_parquet(feature_matrix_path)
    if "game_uid" not in matrix or "date" not in matrix:
        raise SelectiveDraftHoldoutSourceError("feature identities are incomplete")
    matrix["game_uid"] = matrix["game_uid"].astype(str)
    dates = pd.to_datetime(matrix["date"], utc=True, errors="raise")
    selected = matrix.loc[dates.ge(start) & dates.lt(end)].copy()
    if selected.empty or selected["game_uid"].duplicated().any():
        raise SelectiveDraftHoldoutSourceError("feature batch identities are invalid")
    forbidden = [
        column
        for column in selected.columns
        if column in FORBIDDEN_EXACT or column.startswith(FORBIDDEN_PREFIXES)
    ]
    features = selected.drop(columns=forbidden).reset_index(drop=True)
    if not forbidden:
        raise SelectiveDraftHoldoutSourceError(
            "feature source has no explicit outcome or target fields"
        )

    players = _outcome_blind_draft_source(pd.read_parquet(players_path))
    player_dates = pd.to_datetime(players["date"], utc=True, errors="raise")
    draft = players.loc[player_dates.lt(end)].copy()
    game_ids = features["game_uid"].tolist()
    evaluation_draft = draft.loc[draft["game_uid"].isin(game_ids)].copy()
    if set(evaluation_draft["game_uid"]) != set(game_ids):
        raise SelectiveDraftHoldoutSourceError("draft source misses feature games")
    unique_slots = evaluation_draft[
        ["game_uid", "side", "position"]
    ].drop_duplicates()
    slot_count = unique_slots.groupby("game_uid", sort=False).size()
    if not slot_count.eq(10).all() or len(evaluation_draft) != 10 * len(game_ids):
        raise SelectiveDraftHoldoutSourceError("draft source is not ten unique slots")
    draft = draft.sort_values(
        ["date", "game_uid", "side", "position"], kind="stable"
    ).reset_index(drop=True)

    feature_sha256 = _atomic_parquet(features, feature_output_path)
    try:
        player_sha256 = _atomic_parquet(draft, player_output_path)
    except Exception:
        feature_output_path.unlink(missing_ok=True)
        raise
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "outcome_blind": True,
        "window": {
            "start": start.isoformat(),
            "end_exclusive": end.isoformat(),
        },
        "input_sha256": {
            "feature_matrix": sha256_path(feature_matrix_path),
            "players": sha256_path(players_path),
        },
        "output_sha256": {
            "features": feature_sha256,
            "players": player_sha256,
        },
        "removed_feature_columns": sorted(forbidden),
        "feature_rows": len(features),
        "player_rows": len(draft),
        "evaluation_player_rows": len(evaluation_draft),
        "game_ids": game_ids,
        "game_ids_sha256": canonical_sha256(game_ids),
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    receipt_output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_receipt = receipt_output_path.with_suffix(
        receipt_output_path.suffix + ".tmp"
    )
    temporary_receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_receipt.replace(receipt_output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-matrix", type=Path, required=True)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--batch-start", required=True)
    parser.add_argument("--batch-end-exclusive", required=True)
    parser.add_argument("--feature-output", type=Path, required=True)
    parser.add_argument("--player-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    receipt = prepare_holdout_sources(
        feature_matrix_path=args.feature_matrix,
        players_path=args.players,
        batch_start=args.batch_start,
        batch_end_exclusive=args.batch_end_exclusive,
        feature_output_path=args.feature_output,
        player_output_path=args.player_output,
        receipt_output_path=args.receipt_output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
