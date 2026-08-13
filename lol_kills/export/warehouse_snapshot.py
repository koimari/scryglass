"""Build and restore local warehouse snapshots.

Remote publication is disabled. Warehouse and feature tables are private
research inputs and must stay on an authenticated worker volume or backup.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = ROOT / "data" / "lol" / "warehouse_snapshot.json"
DEFAULT_PACK_ROOT = ROOT / "apps" / "scryglass" / "public" / "packs"
SNAPSHOT_SCHEMA = 1
SNAPSHOT_PATH = "state/scryglass-warehouse-v1.tar.gz"
MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
REMOTE_SNAPSHOT_DISABLED = "remote warehouse snapshots are disabled"


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


def _safe_members(tar: tarfile.TarFile) -> Iterable[tarfile.TarInfo]:
    for member in tar.getmembers():
        if member.name == "snapshot_manifest.json":
            yield member
            continue
        if member.issym() or member.islnk():
            raise RuntimeError(f"Refusing snapshot link member: {member.name}")
        destination = (ROOT / member.name).resolve()
        if ROOT.resolve() not in destination.parents:
            raise RuntimeError(f"Refusing snapshot path outside repository: {member.name}")
        yield member


def _read_pointer(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid warehouse snapshot pointer: {path}") from exc
    if not isinstance(payload, dict) or payload.get("transport") != "local":
        raise RuntimeError(REMOTE_SNAPSHOT_DISABLED)
    snapshot_path = Path(str(payload.get("path") or ""))
    if not snapshot_path.is_absolute() or snapshot_path != snapshot_path.resolve():
        raise RuntimeError("Warehouse snapshot path must be an absolute local path")
    payload["path"] = str(snapshot_path)
    return payload


def _read_bounded_response(response: Any, limit: int = MAX_SNAPSHOT_BYTES) -> bytes:
    declared = response.headers.get("Content-Length")
    if declared is not None and int(declared) > limit:
        raise RuntimeError("Warehouse snapshot exceeds the byte limit")
    chunks: list[bytes] = []
    total = 0
    while chunk := response.read(1 << 20):
        total += len(chunk)
        if total > limit:
            raise RuntimeError("Warehouse snapshot exceeds the byte limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _latest_pack(pack_root: Path, pack_id: str | None = None) -> Path:
    if pack_id:
        candidate = pack_root / pack_id
        if candidate.is_dir():
            return candidate
        raise RuntimeError(f"Missing fallback public pack: {candidate}")

    latest_path = pack_root / "latest.json"
    if latest_path.exists():
        payload = json.loads(latest_path.read_text(encoding="utf-8"))
        latest_id = payload.get("pack_id") if isinstance(payload, dict) else None
        if latest_id and (pack_root / str(latest_id)).is_dir():
            return pack_root / str(latest_id)

    candidates = sorted(path for path in pack_root.glob("v*") if path.is_dir())
    if not candidates:
        raise RuntimeError(f"No fallback public pack found under {pack_root}")
    return candidates[-1]


def bootstrap_from_public_pack(
    pack_root: Path = DEFAULT_PACK_ROOT,
    pack_id: str | None = None,
) -> str:
    """Seed normalized OE-shaped parquet from the committed public pack."""
    pack_dir = _latest_pack(pack_root, pack_id)
    parquet_dir = ROOT / "data" / "lol" / "warehouse" / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)

    restored = 0
    for source_dir, target_name in (
        (pack_dir / "team_games", "oe_team_games.parquet"),
        (pack_dir / "player_games", "oe_player_games.parquet"),
    ):
        parts = sorted(source_dir.glob("year=*/part.parquet"))
        if not parts:
            continue
        frames = [pd.read_parquet(part) for part in parts]
        frame = pd.concat(frames, ignore_index=True, sort=False)
        frame.to_parquet(parquet_dir / target_name, index=False)
        restored += len(frame)

    # The next refresh rebuilds maps and players from the normalized sources;
    # these pack tables still make the fallback useful for local inspection.
    maps_dir = pack_dir / "maps"
    map_parts = sorted(maps_dir.glob("year=*/part.parquet"))
    if map_parts:
        maps = pd.concat([pd.read_parquet(part) for part in map_parts], ignore_index=True, sort=False)
        maps.to_parquet(parquet_dir / "maps.parquet", index=False)

    if restored == 0:
        raise RuntimeError(f"Fallback pack contains no team/player game rows: {pack_dir}")
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
    snapshot_path = Path(str(payload["path"]))
    if not snapshot_path.is_file() or snapshot_path.stat().st_size > MAX_SNAPSHOT_BYTES:
        raise RuntimeError("Local warehouse snapshot is unavailable or too large")
    with snapshot_path.open("rb") as source:
        archive_bytes = source.read(MAX_SNAPSHOT_BYTES + 1)
    if len(archive_bytes) > MAX_SNAPSHOT_BYTES:
        raise RuntimeError("Warehouse snapshot exceeds the byte limit")

    with tempfile.NamedTemporaryFile(prefix="scryglass-warehouse-", suffix=".tar.gz") as temp:
        temp.write(archive_bytes)
        temp.flush()
        with tarfile.open(temp.name, mode="r:gz") as archive:
            members = list(_safe_members(archive))
            manifest_member = archive.extractfile("snapshot_manifest.json")
            if manifest_member is None:
                raise RuntimeError("Warehouse snapshot has no internal manifest")
            manifest = json.loads(manifest_member.read().decode("utf-8"))
            if manifest.get("schema") != SNAPSHOT_SCHEMA:
                raise RuntimeError(
                    f"Unsupported warehouse snapshot schema: {manifest.get('schema')}"
                )
            for member in members:
                if member.name == "snapshot_manifest.json":
                    continue
                archive.extract(member, ROOT, filter="data")

    for record in manifest.get("files", []):
        path = ROOT / str(record["path"])
        if not path.is_file():
            raise RuntimeError(f"Warehouse snapshot is missing {record['path']}")
        if path.stat().st_size != int(record["bytes"]):
            raise RuntimeError(f"Warehouse snapshot size mismatch for {record['path']}")
        if _sha256(path) != str(record["sha256"]):
            raise RuntimeError(f"Warehouse snapshot checksum mismatch for {record['path']}")

    print(
        "[snapshot] restored "
        f"{len(archive_bytes) / 1024 / 1024:.1f} MB from local storage"
    )
    return "local"


def save_snapshot(pointer: Path = DEFAULT_POINTER) -> Path:
    del pointer
    raise RuntimeError(REMOTE_SNAPSHOT_DISABLED)


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
