"""Build a source-bound current-versus-future snapshot comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from lol_kills.research.future_value_snapshot_comparison import (
    SnapshotComparisonError,
    build_snapshot_comparison_report,
)


DEFAULT_ROOT = Path("/private/tmp/scryglass-four-variant-runs")
DEFAULT_CURRENT_RECEIPT = DEFAULT_ROOT / "current-ratings-final-fit-v2/current-rating-snapshot-receipt-v1.json"
DEFAULT_FUTURE_ROOT = DEFAULT_ROOT / "future-value-snapshots-v13"
DEFAULT_FUTURE_RECEIPT = DEFAULT_FUTURE_ROOT / "future-value-snapshot-receipt.json"
DEFAULT_PLAYER_DIFF = DEFAULT_FUTURE_ROOT / "future-player-rank-diffs.json"
DEFAULT_TEAM_DIFF = DEFAULT_FUTURE_ROOT / "future-team-rank-diffs.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "future-value-snapshot-comparisons-v1.json"

# This report is a checked comparison of the frozen v13 bundle.  The hashes
# stay in the command so a changed artifact cannot silently reseal the report.
TRUSTED_V13_INPUT_HASHES = {
    "current_receipt": "89ed6c1692b393fdbf5c3bf2ca4c1b0c22bb4ea1a71f4283f7a8d8102070957b",
    "future_receipt": "dbd7de9fd5c813bd07396796e4f27435df28ae42b9a513ea293c01b9d639f511",
    "player_rank_diffs": "2c72020ad5897a23952f868a86c5c865b82d52ca8e615845a124fb921c0cd3d3",
    "team_rank_diffs": "d93c908c173e6af502512f7c25ee43dbb9237ebbec046cf80b2a29bf2287c73e",
}
TRUSTED_V13_SOURCE_RECEIPT_SHA256 = (
    "41325d71332147347bf915798cf6d6c9e8d0b3db1796487d84e167a1056a0212"
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SnapshotComparisonError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SnapshotComparisonError(f"{label} cannot be read") from error
    if not isinstance(value, dict):
        raise SnapshotComparisonError(f"{label} must be a JSON object")
    return value


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SnapshotComparisonError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def build_report(
    *,
    current_receipt_path: Path,
    future_receipt_path: Path,
    player_rank_diff_path: Path,
    team_rank_diff_path: Path,
) -> dict[str, Any]:
    paths_and_keys = (
        (current_receipt_path, "current_receipt"),
        (future_receipt_path, "future_receipt"),
        (player_rank_diff_path, "player_rank_diffs"),
        (team_rank_diff_path, "team_rank_diffs"),
    )
    for path, key in paths_and_keys:
        actual = _sha256_path(path)
        if actual != TRUSTED_V13_INPUT_HASHES[key]:
            raise SnapshotComparisonError(f"trusted v13 {key} artifact changed")
    current = _load(current_receipt_path, "current snapshot receipt")
    future = _load(future_receipt_path, "future snapshot receipt")
    player = _load(player_rank_diff_path, "player rank diff artifact")
    team = _load(team_rank_diff_path, "team rank diff artifact")
    return build_snapshot_comparison_report(
        current_receipt=current,
        future_receipt=future,
        player_rank_diff_artifact=player,
        team_rank_diff_artifact=team,
        current_receipt_file_sha256=_sha256_path(current_receipt_path),
        future_receipt_file_sha256=_sha256_path(future_receipt_path),
        player_rank_diff_file_sha256=_sha256_path(player_rank_diff_path),
        team_rank_diff_file_sha256=_sha256_path(team_rank_diff_path),
        expected_source_receipt_sha256=TRUSTED_V13_SOURCE_RECEIPT_SHA256,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-receipt", type=Path, default=DEFAULT_CURRENT_RECEIPT)
    parser.add_argument("--future-receipt", type=Path, default=DEFAULT_FUTURE_RECEIPT)
    parser.add_argument("--player-rank-diffs", type=Path, default=DEFAULT_PLAYER_DIFF)
    parser.add_argument("--team-rank-diffs", type=Path, default=DEFAULT_TEAM_DIFF)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = build_report(
        current_receipt_path=args.current_receipt.resolve(),
        future_receipt_path=args.future_receipt.resolve(),
        player_rank_diff_path=args.player_rank_diffs.resolve(),
        team_rank_diff_path=args.team_rank_diffs.resolve(),
    )
    _write(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report_sha256": report["report_sha256"],
                "output": str(args.output.resolve()),
                "player_rows": report["snapshot_comparisons"]["player"]["matched_rows"],
                "team_rows": report["snapshot_comparisons"]["team"]["matched_rows"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
