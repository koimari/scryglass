"""Build a research-only future player/team snapshot bundle.

The command can score a source-bound development model when promotion gates
remain open.  It never grants public authority and never publishes ratings,
Draft Score, Tier Lists, matches, or probability outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import pandas as pd

from lol_kills.research.future_value_rating import (
    validate_future_value_source_receipt_payload,
)
from lol_kills.research.future_value_snapshots import (
    SNAPSHOT_VARIANTS,
    FutureValueSnapshotError,
    build_future_value_snapshots,
    load_final_fit_model,
    write_snapshot_bundle,
)


DEFAULT_OUTPUT = Path("/private/tmp/scryglass-four-variant-runs/future-value-snapshots-v1")
CURRENT_RATING_INPUT_SCHEMA_VERSION = "scryglass:current-rating-snapshot-receipt:v1"
CURRENT_RATING_INPUT_BINDING_SCHEMA_VERSION = (
    "scryglass:future-value-current-rating-input-binding:v1"
)


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


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _snapshot_schema_digest(frame: pd.DataFrame) -> str:
    payload = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": {str(column): str(frame[column].dtype) for column in frame.columns},
    }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _snapshot_value_digest(
    frame: pd.DataFrame,
    *,
    identity_column: str,
    value_column: str,
) -> str:
    rows: list[dict[str, Any]] = []
    ordered = frame[[identity_column, value_column]].copy()
    ordered[identity_column] = ordered[identity_column].astype("string")
    ordered = ordered[
        ordered[identity_column].notna()
        & ordered[identity_column].str.strip().ne("")
    ].copy()
    ordered[identity_column] = ordered[identity_column].astype(str)
    ordered[value_column] = pd.to_numeric(ordered[value_column], errors="coerce")
    ordered = ordered.sort_values(identity_column, kind="mergesort")
    for raw_identity, raw_value in ordered.itertuples(index=False, name=None):
        value = float(raw_value)
        if not math.isfinite(value):
            raise FutureValueSnapshotError(
                f"current {identity_column} snapshot contains a non-finite value"
            )
        rows.append({identity_column: str(raw_identity), value_column: value})
    return _sha256_bytes(_canonical_json_bytes(rows))


def _verify_snapshot_frame(
    frame: pd.DataFrame,
    *,
    kind: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    if kind == "player":
        identity_column = "player_id"
        required_columns = {"player", "player_id", "team_id", "mu_effective"}
        identity_prefix = "oe:player:"
        secondary_id = "team_id"
        secondary_prefix = "oe:team:"
    elif kind == "team":
        identity_column = "team_id"
        required_columns = {"team", "team_id", "mu_effective"}
        identity_prefix = "oe:team:"
        secondary_id = None
        secondary_prefix = None
    else:
        raise FutureValueSnapshotError(f"unknown current rating snapshot kind: {kind}")
    if record.get("identity_column") != identity_column or record.get("value_column") != "mu_effective":
        raise FutureValueSnapshotError(f"current {kind} snapshot value schema changed")
    if not required_columns.issubset(frame.columns):
        raise FutureValueSnapshotError(f"current {kind} snapshot schema is incomplete")
    declared_columns = record.get("columns")
    actual_columns = [str(column) for column in frame.columns]
    if declared_columns != actual_columns:
        raise FutureValueSnapshotError(f"current {kind} snapshot schema changed")
    declared_dtypes = record.get("dtypes")
    actual_dtypes = {str(column): str(frame[column].dtype) for column in frame.columns}
    if declared_dtypes != actual_dtypes:
        raise FutureValueSnapshotError(f"current {kind} snapshot dtypes changed")
    expected_schema_hash = str(record.get("schema_sha256") or "").lower()
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_schema_hash) is None
        or _snapshot_schema_digest(frame) != expected_schema_hash
    ):
        raise FutureValueSnapshotError(f"current {kind} snapshot schema digest changed")
    if int(record.get("rows") or -1) != len(frame):
        raise FutureValueSnapshotError(f"current {kind} snapshot row count changed")
    ids = frame[identity_column].astype("string")
    present_ids = ids.notna() & ids.str.strip().ne("")
    if not ids.fillna("").map(lambda value: str(value).startswith(identity_prefix) or not str(value)).all():
        raise FutureValueSnapshotError(f"current {kind} snapshot has invalid stable IDs")
    if ids[present_ids].duplicated().any():
        raise FutureValueSnapshotError(f"current {kind} snapshot has duplicate stable IDs")
    if "verified_rows" in record and int(record.get("verified_rows") or -1) != int(
        present_ids.sum()
    ):
        raise FutureValueSnapshotError(f"current {kind} snapshot identity coverage changed")
    if "unverified_rows" in record and int(record.get("unverified_rows") or -1) != int(
        (~present_ids).sum()
    ):
        raise FutureValueSnapshotError(f"current {kind} snapshot identity coverage changed")
    if secondary_id is not None:
        secondary_ids = frame[secondary_id].astype("string")
        if not secondary_ids.fillna("").map(
            lambda value: str(value).startswith(str(secondary_prefix)) or not str(value)
        ).all():
            raise FutureValueSnapshotError(
                f"current {kind} snapshot has invalid stable team IDs"
            )
    values = pd.to_numeric(frame["mu_effective"], errors="coerce")
    if not values.map(lambda value: math.isfinite(float(value))).all():
        raise FutureValueSnapshotError(f"current {kind} snapshot has non-finite values")
    value_digest = _snapshot_value_digest(
        frame, identity_column=identity_column, value_column="mu_effective"
    )
    declared_value_digest = str(record.get("value_digest_sha256") or "").lower()
    if (
        re.fullmatch(r"[0-9a-f]{64}", declared_value_digest) is None
        or value_digest != declared_value_digest
    ):
        raise FutureValueSnapshotError(f"current {kind} snapshot value digest changed")
    return {
        "kind": kind,
        "identity_column": identity_column,
        "value_column": "mu_effective",
        "rows": len(frame),
        "columns": actual_columns,
        "dtypes": actual_dtypes,
        "schema_sha256": expected_schema_hash,
        "value_digest_sha256": value_digest,
        "verified_rows": int(present_ids.sum()),
        "unverified_rows": int((~present_ids).sum()),
    }


def _verify_current_rating_inputs(
    current_root: Path,
    current_receipt_path: Path,
    current_receipt: dict[str, Any],
    *,
    source_receipt: dict[str, Any],
    expected_current_receipt_sha256: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Verify the independent current snapshot trust root before ranking."""

    expected_receipt_hash = str(expected_current_receipt_sha256 or "").lower()
    if re.fullmatch(r"[0-9a-f]{64}", expected_receipt_hash) is None:
        raise FutureValueSnapshotError("independent current rating receipt hash is required")
    if (
        current_receipt_path.is_symlink()
        or not current_receipt_path.is_file()
        or _sha256_path(current_receipt_path) != expected_receipt_hash
    ):
        raise FutureValueSnapshotError("current rating receipt file hash changed")
    schema_version = str(current_receipt.get("schema_version") or "")
    if schema_version != CURRENT_RATING_INPUT_SCHEMA_VERSION:
        raise FutureValueSnapshotError("current rating receipt schema changed")
    claimed_receipt_hash = str(current_receipt.get("receipt_sha256") or "").lower()
    unsigned = dict(current_receipt)
    unsigned.pop("receipt_sha256", None)
    if (
        re.fullmatch(r"[0-9a-f]{64}", claimed_receipt_hash) is None
        or _sha256_bytes(_canonical_json_bytes(unsigned)) != claimed_receipt_hash
    ):
        raise FutureValueSnapshotError("current rating receipt self-hash changed")
    for key in ("source_receipt_sha256", "source_identity_sha256"):
        if current_receipt.get(key) != source_receipt.get(
            "receipt_sha256" if key == "source_receipt_sha256" else key
        ):
            raise FutureValueSnapshotError(f"current rating {key} changed")
    if str(current_receipt.get("source_as_of")) != str(source_receipt.get("source_as_of")):
        raise FutureValueSnapshotError("current rating source timestamp changed")
    if int(current_receipt.get("source_game_count") or -1) != int(
        source_receipt.get("source_game_count") or -2
    ):
        raise FutureValueSnapshotError("current rating source game count changed")
    snapshots = current_receipt.get("snapshots")
    if not isinstance(snapshots, dict) or set(snapshots) != {"player", "team"}:
        raise FutureValueSnapshotError("current rating player/team snapshot bindings are missing")
    root = current_root.resolve()
    if current_root.is_symlink() or not current_root.is_dir():
        raise FutureValueSnapshotError(f"current rating root is missing or unsafe: {current_root}")
    receipt_root = current_receipt_path.parent.resolve()
    frames: dict[str, pd.DataFrame] = {}
    verified_snapshots: dict[str, dict[str, Any]] = {}
    for kind, relative_name in (
        ("player", "player/player_ratings_snapshot.parquet"),
        ("team", "team/ratings_snapshot.parquet"),
    ):
        record = snapshots.get(kind)
        if not isinstance(record, dict):
            raise FutureValueSnapshotError(f"current {kind} snapshot binding is missing")
        locator = Path(str(record.get("locator") or ""))
        if locator.is_absolute() or not locator.parts or ".." in locator.parts:
            raise FutureValueSnapshotError(f"current {kind} snapshot locator is unsafe")
        bound_path = (receipt_root / locator).resolve()
        expected_path = (root / relative_name).resolve()
        if bound_path != expected_path:
            raise FutureValueSnapshotError(f"current {kind} snapshot path changed")
        if expected_path.is_symlink() or not expected_path.is_file():
            raise FutureValueSnapshotError(f"current {kind} snapshot file is missing")
        if int(record.get("bytes") or -1) != expected_path.stat().st_size:
            raise FutureValueSnapshotError(f"current {kind} snapshot bytes changed")
        expected_file_hash = str(record.get("sha256") or "").lower()
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_file_hash) is None
            or _sha256_path(expected_path) != expected_file_hash
        ):
            raise FutureValueSnapshotError(f"current {kind} snapshot file hash changed")
        frame = pd.read_parquet(expected_path)
        frames[kind] = frame
        verified = _verify_snapshot_frame(frame, kind=kind, record=record)
        verified.update(
            {
                "locator": locator.as_posix(),
                "path": str(expected_path),
                "bytes": expected_path.stat().st_size,
                "sha256": expected_file_hash,
            }
        )
        verified_snapshots[kind] = verified
    binding = {
        "schema_version": CURRENT_RATING_INPUT_BINDING_SCHEMA_VERSION,
        "receipt": {
            "path": str(current_receipt_path),
            "bytes": current_receipt_path.stat().st_size,
            "sha256": expected_receipt_hash,
            "receipt_sha256": claimed_receipt_hash,
            "schema_version": schema_version,
        },
        "source_receipt_sha256": str(source_receipt["receipt_sha256"]),
        "source_identity_sha256": str(source_receipt["source_identity_sha256"]),
        "source_as_of": str(source_receipt["source_as_of"]),
        "source_game_count": int(source_receipt["source_game_count"]),
        "snapshots": verified_snapshots,
    }
    return frames["player"], frames["team"], binding


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
    parser.add_argument(
        "--current-receipt",
        "--current-manifest",
        "--current-rating-receipt",
        dest="current_receipt",
        required=True,
        type=Path,
        help="independent current player/team snapshot receipt or manifest",
    )
    parser.add_argument(
        "--current-receipt-sha256",
        "--current-manifest-sha256",
        "--current-rating-receipt-sha256",
        dest="current_receipt_sha256",
        required=True,
        help="independent raw SHA-256 of the current snapshot receipt",
    )
    parser.add_argument("--model-receipt", type=Path)
    parser.add_argument("--model-artifact", type=Path)
    parser.add_argument(
        "--variant",
        choices=SNAPSHOT_VARIANTS,
        default="future_player_form",
        help="snapshot capability to emit; defaults to the future form component",
    )
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
    current_receipt_path = args.current_receipt.resolve()
    current_receipt = _load_json(current_receipt_path, "current rating receipt")
    current_player, current_team, current_rating_inputs = _verify_current_rating_inputs(
        args.current_root.resolve(),
        current_receipt_path,
        current_receipt,
        source_receipt=source_receipt,
        expected_current_receipt_sha256=args.current_receipt_sha256,
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

    result = build_future_value_snapshots(
        maps,
        players,
        teams,
        source_receipt=source_receipt,
        model=model,
        model_receipt=model_receipt,
        current_player_ratings=current_player,
        current_team_ratings=current_team,
        current_rating_inputs=current_rating_inputs,
        variant=args.variant,
    )
    manifest = write_snapshot_bundle(args.output_root.resolve(), result)
    print(
        json.dumps(
            {
                "status": result.status,
                "variant": args.variant,
                "blockers": list(result.blockers),
                "player_rows": len(result.player_rows),
                "team_rows": len(result.team_rows),
                "rank_coverage": result.receipt.get("rank_coverage", {}),
                "team_context": result.receipt.get("team_context", {}),
                "capability": result.receipt.get("capability", {}),
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
