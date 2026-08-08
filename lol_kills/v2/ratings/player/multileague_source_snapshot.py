"""Immutable maps-plus-players source package for future ratings protocols."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MAPS = Path("data/lol/warehouse/parquet/maps.parquet")
DEFAULT_PLAYERS = Path("data/lol/warehouse/parquet/players.parquet")
DEFAULT_REFRESH_MANIFEST = Path("data/lol/warehouse/parquet/refresh_meta.json")
DEFAULT_SNAPSHOT_ROOT = Path("data/lol/v2/snapshots/multileague-v3")
SCHEMA_VERSION = "scryglass:multileague-rating-source-snapshot:v1"
CURRENT_PACKAGE_ID = (
    "3e739d17476589bc23009a3cb126ab1b2afbcc6071924758be21135e2f88974e"
)
CURRENT_MANIFEST_LOCATOR = (
    DEFAULT_SNAPSHOT_ROOT / CURRENT_PACKAGE_ID / "source-snapshot-manifest.json"
)
CURRENT_MANIFEST_RAW_SHA256 = (
    "8eaba46aff306c678fd9cd17fbb1404c6d88da42c2444e0373df502ae772db63"
)
CURRENT_MANIFEST_CANONICAL_SHA256 = (
    "588cef19c45184f9baa2076eeee7fab3086c43733d02a608928215842e85385c"
)


class MultiLeagueSourceSnapshotError(RuntimeError):
    """The ratings source package is incomplete, mutable, or hash-invalid."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise MultiLeagueSourceSnapshotError("snapshot value is not canonical") from exc
    return hashlib.sha256(raw).hexdigest()


def _locator(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise MultiLeagueSourceSnapshotError(
            f"snapshot file is outside repository root: {path}"
        ) from exc


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiLeagueSourceSnapshotError(f"cannot parse JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise MultiLeagueSourceSnapshotError(f"JSON artifact is not an object: {path}")
    return value


def _copy_no_clobber(source: Path, destination: Path, expected_sha256: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != expected_sha256:
            raise MultiLeagueSourceSnapshotError(
                f"content-addressed destination conflict: {destination}"
            )
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with source.open("rb") as input_stream, os.fdopen(descriptor, "wb") as output:
            shutil.copyfileobj(input_stream, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        if _sha256(temporary) != expected_sha256:
            raise MultiLeagueSourceSnapshotError(
                f"staged source hash mismatch: {source}"
            )
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if _sha256(destination) != expected_sha256:
                raise
    finally:
        temporary.unlink(missing_ok=True)


def _write_manifest_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != raw:
            raise MultiLeagueSourceSnapshotError(
                f"refusing to replace source snapshot manifest: {path}"
            )
        return digest
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def build_source_snapshot(
    *,
    root: Path = ROOT,
    maps_locator: Path = DEFAULT_MAPS,
    players_locator: Path = DEFAULT_PLAYERS,
    refresh_manifest_locator: Path = DEFAULT_REFRESH_MANIFEST,
    snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
) -> dict[str, Any]:
    maps_path = root / maps_locator
    players_path = root / players_locator
    refresh_path = root / refresh_manifest_locator
    for path in (maps_path, players_path, refresh_path):
        if not path.is_file():
            raise MultiLeagueSourceSnapshotError(f"source file is unavailable: {path}")
    refresh = _read_object(refresh_path)
    if refresh.get("schema_version") != "scryglass:warehouse-refresh-manifest:v2":
        raise MultiLeagueSourceSnapshotError("warehouse refresh manifest schema changed")
    claimed_refresh = refresh.get("manifest_canonical_sha256")
    unsigned_refresh = dict(refresh)
    unsigned_refresh.pop("manifest_canonical_sha256", None)
    if (
        not isinstance(claimed_refresh, str)
        or claimed_refresh != _canonical_sha256(unsigned_refresh)
    ):
        raise MultiLeagueSourceSnapshotError(
            "warehouse refresh manifest canonical hash mismatch"
        )
    maps_sha256 = _sha256(maps_path)
    players_sha256 = _sha256(players_path)
    outputs = refresh.get("outputs") or {}
    if (outputs.get("maps") or {}).get("raw_sha256") != maps_sha256:
        raise MultiLeagueSourceSnapshotError("refresh manifest does not bind maps")
    if (outputs.get("rating_players") or {}).get("raw_sha256") != players_sha256:
        raise MultiLeagueSourceSnapshotError("refresh manifest does not bind rating players")
    refresh_authority = refresh.get("authority") or {}
    if any(
        refresh_authority.get(name) is not False
        for name in (
            "model_validation_authority",
            "probability_authority",
            "recommendation_authority",
            "betting_authority",
        )
    ):
        raise MultiLeagueSourceSnapshotError(
            "warehouse refresh manifest exceeds its authority ceiling"
        )

    package_id = _canonical_sha256(
        {
            "maps_raw_sha256": maps_sha256,
            "players_raw_sha256": players_sha256,
            "warehouse_refresh_canonical_sha256": claimed_refresh,
        }
    )
    package = root / snapshot_root / package_id
    snapshot_maps = package / "maps.parquet"
    snapshot_players = package / "players.parquet"
    snapshot_refresh = package / "warehouse-refresh-manifest.json"
    _copy_no_clobber(maps_path, snapshot_maps, maps_sha256)
    _copy_no_clobber(players_path, snapshot_players, players_sha256)
    refresh_raw_sha256 = _sha256(refresh_path)
    _copy_no_clobber(refresh_path, snapshot_refresh, refresh_raw_sha256)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "package_id": package_id,
        "created_from_refresh_at_utc": refresh.get("refreshed_at"),
        "files": {
            "maps": {
                "locator": _locator(root, snapshot_maps),
                "bytes": snapshot_maps.stat().st_size,
                "rows": (outputs.get("maps") or {}).get("rows"),
                "raw_sha256": maps_sha256,
            },
            "players": {
                "locator": _locator(root, snapshot_players),
                "bytes": snapshot_players.stat().st_size,
                "rows": (outputs.get("rating_players") or {}).get("rows"),
                "raw_sha256": players_sha256,
            },
            "warehouse_refresh_manifest": {
                "locator": _locator(root, snapshot_refresh),
                "bytes": snapshot_refresh.stat().st_size,
                "raw_sha256": refresh_raw_sha256,
                "canonical_sha256": claimed_refresh,
            },
        },
        "information_boundary": {
            "all_outcomes_present_in_snapshot_are_adaptive_development": True,
            "future_sealed_targets_present": False,
            "future_sealed_start": None,
            "note": (
                "A separate protocol lock must choose a future boundary after this "
                "snapshot; this source package does not create or open a holdout."
            ),
        },
        "authority": {
            "replayable_source_provenance": True,
            "model_validation_authority": False,
            "rating_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "This package proves exact source replay only. It does not validate or "
            "authorize a player rating, team rating, probability, or wager."
        ),
    }
    manifest["manifest_canonical_sha256"] = _canonical_sha256(manifest)
    manifest_path = package / "source-snapshot-manifest.json"
    raw_sha256 = _write_manifest_no_clobber(manifest_path, manifest)
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_raw_sha256": raw_sha256,
    }


def validate_source_snapshot(
    manifest_path: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    manifest = _read_object(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise MultiLeagueSourceSnapshotError("source snapshot schema changed")
    claimed = manifest.get("manifest_canonical_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_canonical_sha256", None)
    if not isinstance(claimed, str) or claimed != _canonical_sha256(unsigned):
        raise MultiLeagueSourceSnapshotError("source snapshot canonical hash mismatch")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != {
        "maps",
        "players",
        "warehouse_refresh_manifest",
    }:
        raise MultiLeagueSourceSnapshotError("source snapshot file inventory changed")
    for label, record in files.items():
        if not isinstance(record, Mapping):
            raise MultiLeagueSourceSnapshotError(f"{label} record is malformed")
        locator = record.get("locator")
        expected = record.get("raw_sha256")
        if not isinstance(locator, str) or not isinstance(expected, str):
            raise MultiLeagueSourceSnapshotError(f"{label} binding is malformed")
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256(path) != expected
        ):
            raise MultiLeagueSourceSnapshotError(f"{label} bytes drifted")
    authority = manifest.get("authority") or {}
    if any(
        authority.get(name) is not False
        for name in (
            "model_validation_authority",
            "rating_authority",
            "probability_authority",
            "recommendation_authority",
            "betting_authority",
        )
    ):
        raise MultiLeagueSourceSnapshotError("source snapshot exceeds authority")
    return manifest


def validate_current_source_snapshot(*, root: Path = ROOT) -> dict[str, Any]:
    manifest_path = root / CURRENT_MANIFEST_LOCATOR
    if not manifest_path.is_file() or _sha256(manifest_path) != CURRENT_MANIFEST_RAW_SHA256:
        raise MultiLeagueSourceSnapshotError(
            "code-pinned current source snapshot manifest does not match"
        )
    manifest = validate_source_snapshot(manifest_path, root=root)
    if (
        manifest.get("package_id") != CURRENT_PACKAGE_ID
        or manifest.get("manifest_canonical_sha256")
        != CURRENT_MANIFEST_CANONICAL_SHA256
    ):
        raise MultiLeagueSourceSnapshotError(
            "code-pinned current source snapshot identity does not match"
        )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = build_source_snapshot()
    print(
        json.dumps(
            {
                "manifest": str(result["manifest_path"]),
                "raw_sha256": result["manifest_raw_sha256"],
                "canonical_sha256": result["manifest"][
                    "manifest_canonical_sha256"
                ],
                "package_id": result["manifest"]["package_id"],
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CURRENT_MANIFEST_LOCATOR",
    "CURRENT_MANIFEST_RAW_SHA256",
    "CURRENT_PACKAGE_ID",
    "DEFAULT_SNAPSHOT_ROOT",
    "MultiLeagueSourceSnapshotError",
    "SCHEMA_VERSION",
    "build_source_snapshot",
    "validate_source_snapshot",
    "validate_current_source_snapshot",
]
