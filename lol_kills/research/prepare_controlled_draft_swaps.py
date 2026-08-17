"""Prepare outcome-blind role-matched champion swaps for a holdout batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.research.controlled_draft_contribution import (
    validate_role_matched_champion_swap,
)
from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_constituents import (
    DRAFT_INFERENCE_COLUMNS,
)
from lol_kills.research.selective_draft_probability import canonical_sha256


SCHEMA_VERSION = "scryglass:controlled-draft-swap-batch:v1"


class ControlledDraftSwapPreparationError(ValueError):
    """Raised when a blind swap batch is not safe to create."""


def _text(value: Any) -> str:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value).strip()


def prepare_controlled_draft_swaps(
    *,
    feature_path: Path,
    expected_feature_sha256: str,
    player_path: Path,
    expected_player_sha256: str,
    output_path: Path,
    swapped_player_output_path: Path | None = None,
) -> dict[str, Any]:
    """Write one exact paired Draft intervention for each feature row."""

    if output_path.exists():
        raise ControlledDraftSwapPreparationError("swap output already exists")
    if (
        swapped_player_output_path is not None
        and swapped_player_output_path.exists()
    ):
        raise ControlledDraftSwapPreparationError(
            "swapped player output already exists"
        )
    sources = (
        (feature_path, expected_feature_sha256, "feature batch"),
        (player_path, expected_player_sha256, "player batch"),
    )
    for path, expected, label in sources:
        if not path.is_file() or sha256_path(path) != expected:
            raise ControlledDraftSwapPreparationError(f"{label} changed")
    features = pd.read_parquet(feature_path)
    players = pd.read_parquet(player_path)
    if "game_uid" not in features or features["game_uid"].astype(str).duplicated().any():
        raise ControlledDraftSwapPreparationError("feature game identities are invalid")
    if set(players.columns) != set(DRAFT_INFERENCE_COLUMNS):
        raise ControlledDraftSwapPreparationError(
            "player batch is not the exact outcome-blind draft contract"
        )
    players = players.copy()
    players["game_uid"] = players["game_uid"].astype(str)
    game_ids = features["game_uid"].astype(str).tolist()
    grouped = {key: value for key, value in players.groupby("game_uid", sort=False)}
    games: dict[str, list[dict[str, str]]] = {}
    game_receipts: dict[str, str] = {}
    for game_uid in game_ids:
        source = grouped.get(game_uid)
        if source is None or len(source) != 10:
            raise ControlledDraftSwapPreparationError(
                "player batch misses an exact ten-player draft"
            )
        observed = [
            {column: _text(row[column]) for column in DRAFT_INFERENCE_COLUMNS}
            for row in source.to_dict("records")
        ]
        champions = {
            (row["side"].title(), row["position"].casefold()): row["champion"]
            for row in observed
        }
        swapped = []
        for row in observed:
            side = row["side"].title()
            role = row["position"].casefold()
            other_side = "Red" if side == "Blue" else "Blue"
            if (other_side, role) not in champions:
                raise ControlledDraftSwapPreparationError(
                    "player batch has an invalid role layout"
                )
            swapped.append({**row, "champion": champions[(other_side, role)]})
        try:
            receipt = validate_role_matched_champion_swap(
                observed_rows=observed,
                swapped_rows=swapped,
            )
        except ValueError as error:
            raise ControlledDraftSwapPreparationError(
                "player batch cannot form a controlled Draft swap"
            ) from error
        games[game_uid] = swapped
        game_receipts[game_uid] = receipt["receipt_sha256"]

    swapped_player_sha256 = None
    if swapped_player_output_path is not None:
        swapped_players = players.copy()
        for game_uid, swapped in games.items():
            champion_by_slot = {
                (row["side"].title(), row["position"].casefold()): row[
                    "champion"
                ]
                for row in swapped
            }
            mask = swapped_players["game_uid"].eq(game_uid)
            if int(mask.sum()) != 10:
                raise ControlledDraftSwapPreparationError(
                    "swapped player output misses an exact draft"
                )
            swapped_players.loc[mask, "champion"] = [
                champion_by_slot[(str(side).title(), str(role).casefold())]
                for side, role in zip(
                    swapped_players.loc[mask, "side"],
                    swapped_players.loc[mask, "position"],
                )
            ]
        swapped_player_output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_players = swapped_player_output_path.with_suffix(
            swapped_player_output_path.suffix + ".tmp"
        )
        swapped_players.to_parquet(
            temporary_players,
            index=False,
            compression="zstd",
        )
        temporary_players.replace(swapped_player_output_path)
        swapped_player_sha256 = sha256_path(swapped_player_output_path)

    document = {
        "schema_version": SCHEMA_VERSION,
        "outcome_blind": True,
        "input_sha256": {
            "features": expected_feature_sha256,
            "players": expected_player_sha256,
        },
        "game_ids": game_ids,
        "game_ids_sha256": canonical_sha256(game_ids),
        "game_receipt_sha256": game_receipts,
        "games": games,
        "swapped_player_file_sha256": swapped_player_sha256,
    }
    document["receipt_sha256"] = canonical_sha256(document)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--players-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--swapped-players", type=Path)
    args = parser.parse_args()
    receipt = prepare_controlled_draft_swaps(
        feature_path=args.features,
        expected_feature_sha256=args.features_sha256,
        player_path=args.players,
        expected_player_sha256=args.players_sha256,
        output_path=args.output,
        swapped_player_output_path=args.swapped_players,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
