"""Build a source-bound current-versus-future snapshot comparison report."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.build_future_value_snapshots import _verify_current_rating_inputs
from lol_kills.research.future_value_snapshot_comparison import (
    CURRENT_RATING_INPUT_BINDING_SCHEMA,
    SnapshotComparisonError,
    build_snapshot_comparison_report,
)


DEFAULT_ROOT = Path("/private/tmp/scryglass-four-variant-runs")
DEFAULT_CURRENT_RECEIPT = DEFAULT_ROOT / "current-ratings-final-fit-v2/current-rating-snapshot-receipt-v1.json"
DEFAULT_FUTURE_ROOT = DEFAULT_ROOT / "future-value-snapshots-v15-calibrated"
DEFAULT_FUTURE_RECEIPT = DEFAULT_FUTURE_ROOT / "future-value-snapshot-receipt.json"
DEFAULT_PLAYER_DIFF = DEFAULT_FUTURE_ROOT / "future-player-rank-diffs.json"
DEFAULT_TEAM_DIFF = DEFAULT_FUTURE_ROOT / "future-team-rank-diffs.json"
DEFAULT_OUTPUT = DEFAULT_ROOT / "future-value-snapshot-comparisons-v15-calibrated.json"

# This report is a checked comparison of the frozen v15 calibrated bundle.  The hashes
# stay in the command so a changed artifact cannot silently reseal the report.
TRUSTED_V15_INPUT_HASHES = {
    "current_receipt": "89ed6c1692b393fdbf5c3bf2ca4c1b0c22bb4ea1a71f4283f7a8d8102070957b",
    "future_receipt": "5cf2f4958df57b7183e46b2aa41491c4976c86a7a9c2a6e0ce51d8deb972359c",
    "player_rank_diffs": "06435b00aeb67bfc24998f53ee62b2227c44cf15f21cfb8542ff10f69647bc6c",
    "team_rank_diffs": "2daf16db4345166adf0f0c3fa51894299ea3505a88b85f8bc02a30f82b46a2f2",
}
TRUSTED_V15_SOURCE_RECEIPT_SHA256 = (
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


def _identity_digest(identity: str, values: list[str]) -> str:
    payload = {"identity": identity, "ids": sorted(values)}
    canonical = json.dumps(
        payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _verify_current_snapshot_trust_root(
    *,
    current_receipt_path: Path,
    current_receipt: dict[str, Any],
    current_receipt_file_sha256: str,
    source: dict[str, Any],
) -> dict[str, Any]:
    """Verify current snapshot files and retain their finite ID sets in memory."""

    try:
        player_frame, team_frame, binding = _verify_current_rating_inputs(
            current_receipt_path.parent,
            current_receipt_path,
            current_receipt,
            source_receipt={
                "receipt_sha256": source["source_receipt_sha256"],
                "source_identity_sha256": source["source_identity_sha256"],
                "source_as_of": source["source_as_of"],
                "source_game_count": source["source_game_count"],
            },
            expected_current_receipt_sha256=current_receipt_file_sha256,
        )
    except Exception as error:
        raise SnapshotComparisonError(
            "current snapshot trust root verification failed"
        ) from error

    for scope, frame, identity, prefix in (
        ("player", player_frame, "player_id", "oe:player:"),
        ("team", team_frame, "team_id", "oe:team:"),
    ):
        ids = frame[identity].astype("string")
        values = frame["mu_effective"]
        finite = values.map(_finite_number)
        valid = ids.notna() & ids.str.strip().ne("") & finite
        verified_ids = [str(value) for value in ids[valid].tolist()]
        if len(verified_ids) != len(set(verified_ids)):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust IDs are ambiguous"
            )
        if any(not value.startswith(prefix) for value in verified_ids):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust IDs are not stable"
            )
        record = binding["snapshots"].get(scope)
        if not isinstance(record, dict):
            raise SnapshotComparisonError(
                f"current {scope} snapshot trust binding is missing"
            )
        record["identity_ids"] = sorted(verified_ids)
        record["identity_sha256"] = _identity_digest(identity, verified_ids)
        record["finite_rows"] = len(verified_ids)
    binding["schema_version"] = CURRENT_RATING_INPUT_BINDING_SCHEMA
    return binding


def _finite_number(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


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
        if actual != TRUSTED_V15_INPUT_HASHES[key]:
            raise SnapshotComparisonError(f"trusted v15 {key} artifact changed")
    current = _load(current_receipt_path, "current snapshot receipt")
    future = _load(future_receipt_path, "future snapshot receipt")
    player = _load(player_rank_diff_path, "player rank diff artifact")
    team = _load(team_rank_diff_path, "team rank diff artifact")
    future_source = future.get("source")
    if not isinstance(future_source, dict):
        raise SnapshotComparisonError("future snapshot source binding is missing")
    current_trust_root = _verify_current_snapshot_trust_root(
        current_receipt_path=current_receipt_path,
        current_receipt=current,
        current_receipt_file_sha256=_sha256_path(current_receipt_path),
        source=future_source,
    )
    return build_snapshot_comparison_report(
        current_receipt=current,
        future_receipt=future,
        player_rank_diff_artifact=player,
        team_rank_diff_artifact=team,
        current_receipt_file_sha256=_sha256_path(current_receipt_path),
        future_receipt_file_sha256=_sha256_path(future_receipt_path),
        player_rank_diff_file_sha256=_sha256_path(player_rank_diff_path),
        team_rank_diff_file_sha256=_sha256_path(team_rank_diff_path),
        expected_source_receipt_sha256=TRUSTED_V15_SOURCE_RECEIPT_SHA256,
        current_snapshot_trust_root=current_trust_root,
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
