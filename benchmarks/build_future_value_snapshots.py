"""Build a research-only future player/team snapshot bundle.

The command validates the frozen source and emits a blocked receipt when no
authorized final model is supplied.  It does not publish ratings, Draft Score,
Tier Lists, matches, or probability outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.research.future_value_snapshots import (
    FutureValueSnapshotError,
    build_future_value_snapshots,
    write_snapshot_bundle,
)


DEFAULT_OUTPUT = Path("/private/tmp/scryglass-four-variant-runs/future-value-snapshots-v1")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FutureValueSnapshotError(f"{label} is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FutureValueSnapshotError(f"{label} must be a JSON object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    receipt_path = args.source_receipt.resolve()
    source_receipt = _load_json(receipt_path, "source receipt")
    model_receipt = (
        _load_json(args.model_receipt.resolve(), "model receipt")
        if args.model_receipt is not None
        else None
    )
    maps = pd.read_parquet(source_root / "maps.parquet")
    players = pd.read_parquet(source_root / "oe_player_games.parquet")
    teams = pd.read_parquet(source_root / "oe_team_games.parquet")

    player_snapshot_path = args.current_root / "player/player_ratings_snapshot.parquet"
    team_snapshot_path = args.current_root / "team/ratings_snapshot.parquet"
    current_player = pd.read_parquet(player_snapshot_path) if player_snapshot_path.is_file() else None
    current_team = pd.read_parquet(team_snapshot_path) if team_snapshot_path.is_file() else None

    result = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        model_receipt=model_receipt,
        current_player_ratings=current_player,
        current_team_ratings=current_team,
    )
    manifest = write_snapshot_bundle(args.output_root.resolve(), result)
    print(
        json.dumps(
            {
                "status": result.status,
                "blockers": list(result.blockers),
                "player_rows": len(result.player_rows),
                "team_rows": len(result.team_rows),
                "manifest_sha256": manifest["manifest_sha256"],
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
