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
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = ROOT / "data" / "lol" / "warehouse_snapshot.json"
DEFAULT_PACK_ROOT = ROOT / "apps" / "scryglass" / "public" / "packs"
SNAPSHOT_SCHEMA = 1
SNAPSHOT_PATH = "state/scryglass-warehouse-v1.tar.gz"


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
    if not isinstance(payload, dict) or not payload.get("url"):
        raise RuntimeError(f"Warehouse snapshot pointer has no URL: {path}")
    return payload


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
    url = str(payload["url"])
    request = urllib.request.Request(url, headers={"Accept": "application/gzip"})
    with urllib.request.urlopen(request, timeout=120) as response:
        archive_bytes = response.read()

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
            archive.extractall(
                ROOT,
                members=[member for member in members if member.name != "snapshot_manifest.json"],
            )

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
