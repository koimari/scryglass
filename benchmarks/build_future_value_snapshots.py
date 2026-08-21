"""Build a research-only future player/team snapshot bundle.

The command can score a source-bound development model when promotion gates
remain open.  It never grants public authority and never publishes ratings,
Draft Score, Tier Lists, matches, or probability outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd

from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_snapshots import (
    FutureValueSnapshotError,
    build_future_value_snapshots,
    load_final_fit_model,
    write_snapshot_bundle,
)


DEFAULT_OUTPUT = Path("/private/tmp/scryglass-four-variant-runs/future-value-snapshots-v1")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FutureValueSnapshotError(f"{label} is missing or unsafe: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise FutureValueSnapshotError(f"{label} must be a JSON object")
    return value


def _verify_source_inputs(
    source_root: Path,
    source_receipt_path: Path,
    source_receipt: dict[str, Any],
    *,
    expected_source_receipt_sha256: str | None,
) -> dict[str, Path]:
    expected_receipt_hash = str(expected_source_receipt_sha256 or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_receipt_hash) is None:
        raise FutureValueSnapshotError("independent source receipt file hash is required")
    if (
        source_receipt_path.is_symlink()
        or not source_receipt_path.is_file()
        or _sha256_path(source_receipt_path) != expected_receipt_hash
    ):
        raise FutureValueSnapshotError("source receipt file hash changed")
    try:
        validate_future_value_source_receipt_payload(source_receipt)
    except Exception as error:
        raise FutureValueSnapshotError("source receipt failed validation") from error
    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, dict):
        raise FutureValueSnapshotError("source receipt file bindings are missing")
    expected_names = {
        "maps": "maps.parquet",
        "players": "oe_player_games.parquet",
        "teams": "oe_team_games.parquet",
    }
    root = source_root.resolve()
    receipt_root = source_receipt_path.parent.resolve()
    verified: dict[str, Path] = {}
    for label, name in expected_names.items():
        record = source_files.get(label)
        if not isinstance(record, dict):
            raise FutureValueSnapshotError(f"source {label} file binding is missing")
        locator = Path(str(record.get("locator") or ""))
        if locator.is_absolute() or not locator.parts or ".." in locator.parts:
            raise FutureValueSnapshotError(f"source {label} file locator is unsafe")
        bound_path = (receipt_root / locator).resolve()
        path = (root / name).resolve()
        if bound_path != path:
            raise FutureValueSnapshotError(f"source {label} file path changed")
        if path.is_symlink() or not path.is_file():
            raise FutureValueSnapshotError(f"source {label} file is missing")
        if int(record.get("bytes") or -1) != path.stat().st_size:
            raise FutureValueSnapshotError(f"source {label} file bytes changed")
        expected_hash = str(record.get("sha256") or "").lower()
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_hash) is None
            or _sha256_path(path) != expected_hash
        ):
            raise FutureValueSnapshotError(f"source {label} file hash changed")
        verified[label] = path
    return verified


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--model-artifact", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    receipt_path = args.source_receipt.resolve()
    source_receipt = _load_json(receipt_path, "source receipt")
    source_paths = _verify_source_inputs(
        source_root,
        receipt_path,
        source_receipt,
        expected_source_receipt_sha256=args.source_receipt_sha256,
    )
    model_receipt = (
        _load_json(args.model_receipt.resolve(), "model receipt")
        if args.model_receipt is not None
        else None
    )
    model = None
    if model_receipt is not None:
        artifact_path = (
            args.model_artifact.resolve()
            if args.model_artifact is not None
            else args.model_receipt.resolve().with_name("final-v2-model.json")
        )
        model, loaded_receipt = load_final_fit_model(
            artifact_path,
            args.model_receipt.resolve(),
            source_receipt=source_receipt,
        )
        if loaded_receipt.get("receipt_sha256") != model_receipt.get("receipt_sha256"):
            raise FutureValueSnapshotError("model receipt changed while loading artifact")
    maps = pd.read_parquet(source_paths["maps"])
    players = pd.read_parquet(source_paths["players"])
    teams = pd.read_parquet(source_paths["teams"])

    player_snapshot_path = args.current_root / "player/player_ratings_snapshot.parquet"
    team_snapshot_path = args.current_root / "team/ratings_snapshot.parquet"
    current_player = pd.read_parquet(player_snapshot_path) if player_snapshot_path.is_file() else None
    current_team = pd.read_parquet(team_snapshot_path) if team_snapshot_path.is_file() else None

    result = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        model=model,
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
                "rank_coverage": result.receipt.get("rank_coverage", {}),
                "rank_diff_extremes": result.receipt.get("rank_diff_extremes", {}),
                "manifest_sha256": manifest["manifest_sha256"],
                "output_root": str(args.output_root.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
