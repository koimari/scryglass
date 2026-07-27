"""Persist the normalized refresh state between ephemeral CI workers.

The fast refresh does not need to redownload the large OE CSVs.  It needs the
last normalized warehouse plus the generated feature tables, then layers new
completed GRID rows on top.  This module stores that small derived state in a
single Vercel Blob object and keeps a stable pointer in the repository.

The snapshot deliberately excludes raw OE CSVs, Riot timelines, joblib
models, and research artifacts.  The committed public pack is the first-run
fallback; after the first successful refresh, the Blob snapshot becomes the
source of truth for the next worker.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import pandas as pd

from lol_kills.export.upload_pack import validate_pack_id


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = ROOT / "data" / "lol" / "warehouse_snapshot.json"
DEFAULT_PACK_ROOT = ROOT / "apps" / "lol-atlas" / "public" / "packs"
SNAPSHOT_SCHEMA = 1
SNAPSHOT_PATH = "state/scryglass-warehouse-v1.tar.gz"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_SNAPSHOT_MANIFEST = "snapshot_manifest.json"
_SNAPSHOT_ROOTS = (
    PurePosixPath("data/lol/warehouse/parquet"),
    PurePosixPath("data/lol/features"),
)
_SNAPSHOT_EXACT_FILES = frozenset(
    {
        "data/lol/draft_games.json",
        "data/lol/draft_players.json",
        "data/lol/draft_model.json",
        "data/lol/markets_model.json",
        "data/lol/kill_models.json",
    }
)


def _pointer_path(value: Path | None) -> Path:
    return (value or DEFAULT_POINTER).expanduser()


def _snapshot_files() -> list[Path]:
    """Return only derived inputs needed by the next refresh."""
    candidates: list[Path] = []
    parquet_dir = ROOT / "data" / "lol" / "warehouse" / "parquet"
    features_dir = ROOT / "data" / "lol" / "features"
    for directory in (parquet_dir, features_dir):
        if directory.exists():
            candidates.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in {".parquet", ".json"}
            )

    # These optional caches make Leaguepedia enrichment available when it has
    # already been populated, but the refresh remains valid without them.
    for name in (
        "draft_games.json",
        "draft_players.json",
        "draft_model.json",
        "markets_model.json",
        "kill_models.json",
    ):
        path = ROOT / "data" / "lol" / name
        if path.exists():
            candidates.append(path)
    return sorted(set(candidates))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_relative_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"Invalid warehouse snapshot path: {value!r}")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise RuntimeError(f"Invalid warehouse snapshot path: {value!r}")
    allowed = value in _SNAPSHOT_EXACT_FILES or any(
        path.is_relative_to(root) and path != root for root in _SNAPSHOT_ROOTS
    )
    if not allowed:
        raise RuntimeError(f"Snapshot path is outside the derived-state allowlist: {value}")
    return value


def _validated_snapshot_records(
    archive: tarfile.TarFile,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    if manifest.get("schema") != SNAPSHOT_SCHEMA:
        raise RuntimeError(
            f"Unsupported warehouse snapshot schema: {manifest.get('schema')}"
        )
    raw_records = manifest.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise RuntimeError("Warehouse snapshot manifest has no files")

    members = archive.getmembers()
    member_names = [member.name for member in members]
    if len(member_names) != len(set(member_names)):
        raise RuntimeError("Warehouse snapshot contains duplicate archive members")
    members_by_name = {member.name: member for member in members}

    records: list[dict[str, Any]] = []
    declared_paths: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise RuntimeError(f"Snapshot file record {index} must be an object")
        relative = _snapshot_relative_path(raw_record.get("path"))
        if relative in declared_paths:
            raise RuntimeError(f"Duplicate snapshot manifest path: {relative}")
        size = raw_record.get("bytes")
        digest = raw_record.get("sha256")
        if type(size) is not int or size < 0:
            raise RuntimeError(f"Invalid snapshot byte size for {relative}")
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise RuntimeError(f"Invalid snapshot SHA-256 for {relative}")
        member = members_by_name.get(relative)
        if member is None:
            raise RuntimeError(f"Warehouse snapshot is missing {relative}")
        if not member.isfile() or member.issym() or member.islnk():
            raise RuntimeError(f"Refusing non-file snapshot member: {relative}")
        if member.size != size:
            raise RuntimeError(f"Warehouse snapshot size mismatch for {relative}")
        declared_paths.add(relative)
        records.append({"path": relative, "bytes": size, "sha256": digest})

    expected_members = declared_paths | {_SNAPSHOT_MANIFEST}
    actual_members = set(member_names)
    missing = sorted(expected_members - actual_members)
    extra = sorted(actual_members - expected_members)
    if missing:
        raise RuntimeError(f"Warehouse snapshot is missing members: {missing}")
    if extra:
        raise RuntimeError(f"Warehouse snapshot has undeclared members: {extra}")
    manifest_member = members_by_name.get(_SNAPSHOT_MANIFEST)
    if manifest_member is None or not manifest_member.isfile():
        raise RuntimeError("Warehouse snapshot has no regular internal manifest")
    return records


def _extract_verified_snapshot(
    archive: tarfile.TarFile,
    records: list[dict[str, Any]],
    staging_root: Path,
) -> None:
    """Extract and verify every input while all live warehouse files are untouched."""

    for record in records:
        relative = str(record["path"])
        member = archive.getmember(relative)
        source = archive.extractfile(member)
        if source is None:
            raise RuntimeError(f"Warehouse snapshot is unreadable: {relative}")
        destination = staging_root.joinpath(*PurePosixPath(relative).parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        size = 0
        with source, destination.open("wb") as output:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                size += len(chunk)
                digest.update(chunk)
                output.write(chunk)
        if size != record["bytes"]:
            raise RuntimeError(f"Warehouse snapshot size mismatch for {relative}")
        if digest.hexdigest() != record["sha256"]:
            raise RuntimeError(f"Warehouse snapshot checksum mismatch for {relative}")


def _install_staged_files(
    staging_root: Path,
    relative_paths: list[str],
) -> None:
    """Install verified files with rollback if an atomic replacement fails."""

    backup_root = staging_root.parent / "backup"
    touched: list[tuple[Path, Path | None]] = []
    try:
        for relative in relative_paths:
            source = staging_root.joinpath(*PurePosixPath(relative).parts)
            destination = ROOT.joinpath(*PurePosixPath(relative).parts)
            if destination.is_symlink() or (
                destination.exists() and not destination.is_file()
            ):
                raise RuntimeError(f"Unsafe warehouse snapshot destination: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            backup: Path | None = None
            if destination.exists():
                backup = backup_root.joinpath(*PurePosixPath(relative).parts)
                backup.parent.mkdir(parents=True, exist_ok=True)
                os.replace(destination, backup)
            touched.append((destination, backup))
            os.replace(source, destination)
    except Exception:
        for destination, backup in reversed(touched):
            if destination.exists():
                destination.unlink()
            if backup is not None and backup.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.replace(backup, destination)
        raise


def _read_pointer(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid warehouse snapshot pointer: {path}") from exc
    if not isinstance(payload, dict) or not payload.get("url"):
        raise RuntimeError(f"Warehouse snapshot pointer has no URL: {path}")
    return payload


def _latest_pack(pack_root: Path, pack_id: str | None = None) -> Path:
    if pack_id:
        safe_pack_id = validate_pack_id(pack_id)
        candidate = pack_root / safe_pack_id
        if candidate.is_dir() and not candidate.is_symlink():
            return candidate
        raise RuntimeError(f"Missing fallback public pack: {candidate}")

    latest_path = pack_root / "latest.json"
    if latest_path.exists():
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        latest_id = payload.get("pack_id") if isinstance(payload, dict) else None
        if latest_id:
            safe_latest_id = validate_pack_id(latest_id)
            candidate = pack_root / safe_latest_id
            if candidate.is_dir() and not candidate.is_symlink():
                return candidate

    candidates: list[Path] = []
    for path in pack_root.glob("v*"):
        if not path.is_dir() or path.is_symlink():
            continue
        try:
            validate_pack_id(path.name)
        except ValueError:
            continue
        candidates.append(path)
    candidates.sort()
    if not candidates:
        raise RuntimeError(f"No fallback public pack found under {pack_root}")
    return candidates[-1]


def _verified_pack_parts(pack_dir: Path, source_dir: Path) -> list[Path]:
    pack_root = pack_dir.resolve()
    parts = sorted(source_dir.glob("year=*/part.parquet"))
    for part in parts:
        if part.is_symlink() or not part.is_file():
            raise RuntimeError(f"Unsafe fallback pack input: {part}")
        resolved = part.resolve()
        if pack_root not in resolved.parents:
            raise RuntimeError(f"Fallback pack input escapes its pack: {part}")
    return parts


def bootstrap_from_public_pack(
    pack_root: Path = DEFAULT_PACK_ROOT,
    pack_id: str | None = None,
) -> str:
    """Seed normalized OE-shaped parquet from the committed public pack."""
    pack_dir = _latest_pack(pack_root, pack_id)
    outputs: list[tuple[str, pd.DataFrame]] = []
    restored = 0
    for source_dir, target_name in (
        (pack_dir / "team_games", "oe_team_games.parquet"),
        (pack_dir / "player_games", "oe_player_games.parquet"),
    ):
        parts = _verified_pack_parts(pack_dir, source_dir)
        if not parts:
            continue
        frames = [pd.read_parquet(part) for part in parts]
        frame = pd.concat(frames, ignore_index=True, sort=False)
        outputs.append(
            (f"data/lol/warehouse/parquet/{target_name}", frame)
        )
        restored += len(frame)

    # The next refresh rebuilds maps and players from the normalized sources;
    # these pack tables still make the fallback useful for local inspection.
    maps_dir = pack_dir / "maps"
    map_parts = _verified_pack_parts(pack_dir, maps_dir)
    if map_parts:
        maps = pd.concat(
            [pd.read_parquet(part) for part in map_parts],
            ignore_index=True,
            sort=False,
        )
        outputs.append(("data/lol/warehouse/parquet/maps.parquet", maps))

    if restored == 0:
        raise RuntimeError(f"Fallback pack contains no team/player game rows: {pack_dir}")
    with tempfile.TemporaryDirectory(
        prefix=".scryglass-bootstrap-",
        dir=ROOT,
    ) as temp:
        transaction_root = Path(temp)
        staging_root = transaction_root / "payload"
        relative_paths: list[str] = []
        for relative, frame in outputs:
            destination = staging_root.joinpath(*PurePosixPath(relative).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(destination, index=False)
            # Verify each staged parquet before any live warehouse replacement.
            pd.read_parquet(destination)
            relative_paths.append(relative)
        _install_staged_files(staging_root, relative_paths)
    print(f"[snapshot] bootstrapped normalized rows={restored} from {pack_dir.name}")
    return pack_dir.name


def restore_snapshot(
    pointer: Path = DEFAULT_POINTER,
    *,
    pack_root: Path = DEFAULT_PACK_ROOT,
    pack_id: str | None = None,
) -> str:
    """Restore the Blob snapshot, or bootstrap from the committed pack."""
    pointer = _pointer_path(pointer)
    if not pointer.exists():
        return bootstrap_from_public_pack(pack_root, pack_id)

    payload = _read_pointer(pointer)
    url = str(payload["url"])
    request = urllib.request.Request(url, headers={"Accept": "application/gzip"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()

    with tempfile.TemporaryDirectory(
        prefix=".scryglass-restore-",
        dir=ROOT,
    ) as temp:
        transaction_root = Path(temp)
        staging_root = transaction_root / "payload"
        with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as archive:
            manifest_member = archive.extractfile(_SNAPSHOT_MANIFEST)
            if manifest_member is None:
                raise RuntimeError("Warehouse snapshot has no internal manifest")
            try:
                manifest = json.loads(manifest_member.read().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RuntimeError("Invalid warehouse snapshot manifest") from exc
            if not isinstance(manifest, dict):
                raise RuntimeError("Warehouse snapshot manifest must be an object")
            records = _validated_snapshot_records(archive, manifest)
            _extract_verified_snapshot(archive, records, staging_root)
        _install_staged_files(
            staging_root,
            [str(record["path"]) for record in records],
        )

    print(
        "[snapshot] restored "
        f"{len(archive_bytes) / 1024 / 1024:.1f} MB from {payload.get('pathname', url)}"
    )
    return "blob"


def save_snapshot(pointer: Path = DEFAULT_POINTER) -> Path:
    """Upload the current derived state and create/update the stable pointer."""
    token = os.environ.get("BLOB_READ_WRITE_TOKEN") or os.environ.get(
        "VERCEL_BLOB_READ_WRITE_TOKEN"
    )
    if not token:
        raise RuntimeError("BLOB_READ_WRITE_TOKEN is required to save the warehouse snapshot")

    pointer = _pointer_path(pointer)
    existing: dict[str, Any] = {}
    if pointer.exists():
        try:
            candidate = json.loads(pointer.read_text(encoding="utf-8"))
            if isinstance(candidate, dict):
                existing = candidate
        except json.JSONDecodeError:
            existing = {}

    files = _snapshot_files()
    if not files:
        raise RuntimeError("No derived warehouse files are available to snapshot")

    with tempfile.NamedTemporaryFile(prefix="scryglass-warehouse-", suffix=".tar.gz") as temp:
        manifest = {
            "schema": SNAPSHOT_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "files": [],
        }
        with tarfile.open(temp.name, mode="w:gz") as archive:
            for path in files:
                rel = path.relative_to(ROOT).as_posix()
                archive.add(path, arcname=rel, recursive=False)
                manifest["files"].append(
                    {"path": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)}
                )
            raw_manifest = json.dumps(manifest, indent=2).encode("utf-8")
            info = tarfile.TarInfo("snapshot_manifest.json")
            info.size = len(raw_manifest)
            archive.addfile(info, fileobj=io.BytesIO(raw_manifest))
        temp.flush()

        pathname = str(existing.get("pathname") or SNAPSHOT_PATH)
        from lol_kills.export.upload_pack import _blob_put

        url = _blob_put(
            token,
            pathname,
            Path(temp.name).read_bytes(),
            "application/gzip",
            cache_control="no-cache, max-age=0",
            allow_overwrite=True,
        )

    pointer_payload = {
        "schema": SNAPSHOT_SCHEMA,
        "pathname": pathname,
        "url": url,
        "description": "Derived warehouse state for the Scryglass GRID freshness worker",
    }
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(json.dumps(pointer_payload, indent=2) + "\n", encoding="utf-8")
    print(f"[snapshot] uploaded {len(files)} files to {pathname}")
    return pointer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    restore = sub.add_parser("restore")
    restore.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    restore.add_argument("--pack-root", type=Path, default=DEFAULT_PACK_ROOT)
    restore.add_argument("--pack-id", default=None)

    save = sub.add_parser("save")
    save.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)

    args = parser.parse_args(argv)
    if args.command == "restore":
        restore_snapshot(args.pointer, pack_root=args.pack_root, pack_id=args.pack_id)
    else:
        save_snapshot(args.pointer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
