"""Publish descriptive ratings after complete new OE games appear."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import pandas as pd
import pyarrow.parquet as pq

from lol_kills.etl.oe_api_ingest import ingest_oe_api
from lol_kills.etl.oe_live_source import build_live_source
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.export import pack_spec
from lol_kills.export.public_pack import export_public_pack, source_identity_sha256


RAW_RECEIPT = Path("data/lol/warehouse/raw/oe_api/tierlist-live-v1.json")
LIVE_ROOT = Path("data/lol/warehouse/parquet/oe_live")
LIVE_MAPS = LIVE_ROOT / "maps.parquet"
LIVE_TEAMS = LIVE_ROOT / "oe_team_games.parquet"
LIVE_PLAYERS = LIVE_ROOT / "oe_player_games.parquet"


class RefreshValidationError(RuntimeError):
    """A refresh cannot prove complete source-to-pack integration."""


class SyncAlreadyRunning(RuntimeError):
    """Another process owns the refresh lock."""


@dataclass(frozen=True)
class SyncConfig:
    root: Path
    public_root: Path
    output_root: Path
    state_path: Path
    lock_path: Path
    health_path: Path
    years: tuple[int, ...] = (2025, 2026)
    discovery_cache_hours: int = 6
    window_hours: int = 24 * 120
    lookback_days: int = 120
    max_workers: int = 8


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=True, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _canonical_ids(values: Sequence[Any]) -> list[str]:
    return sorted({key for value in values if (key := canonical_source_game_key(value))})


def _receipt_game_ids(root: Path) -> list[str]:
    values = []
    for game in _load_json(root / RAW_RECEIPT).get("games", []):
        if isinstance(game, dict):
            values.append(game.get("game_uid") or game.get("oe_game_id") or game.get("gameid"))
    return _canonical_ids(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_frame(path: Path, required: Sequence[str] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise RefreshValidationError(f"missing live source file: {path}")
    columns = pq.ParquetFile(path).schema_arrow.names
    identity_columns = [name for name in ("game_uid", "gameid", "oe_gameid") if name in columns]
    if not identity_columns or any(name not in columns for name in required):
        raise RefreshValidationError(f"live source schema is incomplete: {path}")
    frame = pd.read_parquet(path, columns=[*identity_columns, *required])
    frame["_game_id"] = [
        next(
            (
                key
                for column in identity_columns
                if (key := canonical_source_game_key(row[column]))
            ),
            "",
        )
        for _, row in frame.iterrows()
    ]
    if frame["_game_id"].eq("").any():
        raise RefreshValidationError(f"live source contains an empty game identity: {path}")
    return frame


def validate_live_source(root: Path, new_game_ids: Sequence[str]) -> dict[str, Any]:
    """Require complete canonical rows for every new game."""

    requested = set(_canonical_ids(list(new_game_ids)))
    if len(requested) != len(new_game_ids) or not requested:
        raise RefreshValidationError("new game identities are empty or duplicated")
    maps = _identity_frame(root / LIVE_MAPS)
    teams = _identity_frame(root / LIVE_TEAMS, ("side", "result", "teamname"))
    players = _identity_frame(root / LIVE_PLAYERS, ("side", "position", "playername"))
    roles = {"top", "jng", "mid", "bot", "sup"}
    for game_id in sorted(requested):
        map_rows = maps[maps["_game_id"].eq(game_id)]
        team_rows = teams[teams["_game_id"].eq(game_id)]
        player_rows = players[players["_game_id"].eq(game_id)]
        if (len(map_rows), len(team_rows), len(player_rows)) != (1, 2, 10):
            raise RefreshValidationError(
                f"game {game_id} has malformed rows: maps={len(map_rows)} teams={len(team_rows)} players={len(player_rows)}"
            )
        sides = set(team_rows["side"].astype(str).str.title())
        results = set(pd.to_numeric(team_rows["result"], errors="coerce").dropna().astype(int))
        team_names = team_rows["teamname"].astype("string").fillna("").str.strip()
        if sides != {"Blue", "Red"} or results != {0, 1} or team_names.eq("").any() or team_names.nunique() != 2:
            raise RefreshValidationError(f"game {game_id} has malformed teams, sides, or results")
        names = player_rows["playername"].astype("string").fillna("").str.strip()
        if names.eq("").any() or names.nunique() != 10:
            raise RefreshValidationError(f"game {game_id} has malformed player identities")
        for side in ("Blue", "Red"):
            side_rows = player_rows[player_rows["side"].astype(str).str.title().eq(side)]
            if len(side_rows) != 5 or set(side_rows["position"].astype(str).str.casefold()) != roles:
                raise RefreshValidationError(f"game {game_id} has malformed {side} roles")
    all_ids = _canonical_ids(maps["_game_id"].tolist())
    if len(all_ids) != len(maps):
        raise RefreshValidationError("live maps are not one row per canonical game identity")
    return {"game_ids": all_ids, "game_count": len(all_ids), "identity_sha256": source_identity_sha256(all_ids)}


def validate_pack(pack_dir: Path, manifest: dict[str, Any], source: dict[str, Any]) -> dict[str, Any]:
    """Verify the compact ratings pack and its source identity binding."""

    files = manifest.get("files")
    if not isinstance(files, list):
        raise RefreshValidationError("pack manifest has no file inventory")
    paths = {str(item.get("path")) for item in files if isinstance(item, dict)}
    if paths != set(pack_spec.PUBLIC_RATING_REQUIRED_FILES):
        raise RefreshValidationError("pack inventory differs from the public ratings contract")
    total_bytes = 0
    root = pack_dir.resolve()
    for item in files:
        relative = Path(str(item["path"]))
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise RefreshValidationError(f"pack path leaves its directory: {relative}") from error
        if not path.is_file() or path.stat().st_size != int(item.get("bytes", -1)):
            raise RefreshValidationError(f"pack file is missing or has the wrong size: {relative}")
        if _sha256(path) != item.get("sha256"):
            raise RefreshValidationError(f"pack checksum mismatch: {relative}")
        public_text = path.read_text(encoding="utf-8").casefold()
        if "los ratones" in public_text:
            raise RefreshValidationError(f"excluded team affiliation appears in public pack: {relative}")
        if "oracle_elixir_api" in public_text or "public_datalisk_api" in public_text:
            raise RefreshValidationError(f"transport label appears as a public league: {relative}")
        total_bytes += path.stat().st_size
    ratings = manifest.get("ratings") or {}
    if ratings.get("source_game_count") != source["game_count"]:
        raise RefreshValidationError("pack source game count does not match the live source")
    if ratings.get("source_identity_sha256") != source["identity_sha256"]:
        raise RefreshValidationError("pack source identity digest does not match the live source")
    if manifest.get("total_files") != len(files) or manifest.get("total_bytes") != total_bytes:
        raise RefreshValidationError("pack inventory totals are invalid")
    return {"files": len(files), "bytes": total_bytes, **source}


def publish_pack(pack_dir: Path, manifest: dict[str, Any], public_root: Path) -> dict[str, Any]:
    """Publish an immutable directory, then replace manifest.json last."""

    pack_id = str(manifest.get("pack_id") or "")
    if not pack_id or not pack_dir.is_dir():
        raise RefreshValidationError("pack publication has no valid source directory")
    public_root.mkdir(parents=True, exist_ok=True)
    destination = public_root / pack_id
    if destination.exists():
        raise FileExistsError(f"immutable pack already exists: {destination}")
    staging = public_root / f".{pack_id}.{uuid.uuid4().hex}.incoming"
    published = dict(manifest)
    published["base_url"] = f"/packs/{pack_id}"
    try:
        shutil.copytree(pack_dir, staging)
        _atomic_json(staging / "manifest.json", published)
        os.replace(staging, destination)
        _atomic_json(public_root / "manifest.json", published)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"pack_id": pack_id, "destination": str(destination)}


def _write_health(config: SyncConfig, status: str, now: datetime, **fields: Any) -> None:
    _atomic_json(config.health_path, {"status": status, "checked_at": _iso(now), **fields})


def sync_once(
    config: SyncConfig,
    *,
    now: datetime | None = None,
    ingest_fn: Callable[..., dict[str, Any]] = ingest_oe_api,
    build_live_fn: Callable[..., dict[str, Any]] = build_live_source,
    export_pack_fn: Callable[..., dict[str, Any]] = export_public_pack,
    validate_live_fn: Callable[..., dict[str, Any]] = validate_live_source,
    validate_pack_fn: Callable[..., dict[str, Any]] = validate_pack,
    publish_pack_fn: Callable[..., dict[str, Any]] = publish_pack,
) -> dict[str, Any]:
    checked_at = now or datetime.now(timezone.utc)
    state = _load_json(config.state_path)
    known = set(_canonical_ids(state.get("published_game_ids", [])))
    _write_health(config, "checking", checked_at)
    try:
        source_meta = ingest_fn(
            config.root,
            start=pd.Timestamp(checked_at - timedelta(hours=config.window_hours)),
            end=pd.Timestamp(checked_at),
            lookback_days=config.lookback_days,
            discovery_cache_hours=config.discovery_cache_hours,
            max_workers=config.max_workers,
        )
        observed = set(_receipt_game_ids(config.root))
        new_ids = sorted(observed - known)
        if not new_ids:
            result = {"status": "no_change", "new_game_ids": [], "pack_id": state.get("pack_id")}
            _atomic_json(config.state_path, {**state, **result, "published_game_ids": sorted(known | observed)})
            _write_health(config, "ok", checked_at, pack_id=result["pack_id"])
            return result
        if source_meta.get("player_detail_complete") is not True:
            result = {"status": "waiting_for_details", "new_game_ids": new_ids, "pack_id": state.get("pack_id")}
            _atomic_json(config.state_path, {**state, **result, "published_game_ids": sorted(known), "pending_game_ids": new_ids})
            _write_health(config, "waiting_for_details", checked_at, new_game_ids=new_ids)
            return result
        build_live_fn(config.root)
        source = validate_live_fn(config.root, new_ids)
        pack_id = f"v{checked_at.strftime('%Y.%m.%d.%H%M%S')}"
        manifest = export_pack_fn(years=config.years, out_root=config.output_root, pack_id=pack_id, project_root=config.root)
        validation = validate_pack_fn(config.output_root / pack_id, manifest, source)
        publication = publish_pack_fn(config.output_root / pack_id, manifest, config.public_root)
        result = {"status": "published", "new_game_ids": new_ids, "pack_id": pack_id, "validation": validation, "publication": publication}
        _atomic_json(config.state_path, {**result, "published_game_ids": sorted(known | observed), "pending_game_ids": []})
        _write_health(config, "ok", checked_at, pack_id=pack_id, new_game_ids=new_ids)
        return result
    except Exception as error:
        _write_health(config, "error", checked_at, pack_id=state.get("pack_id"), reason=f"{type(error).__name__}: {str(error)[:500]}")
        raise


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SyncAlreadyRunning(f"another ratings sync owns {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--public-root", type=Path, default=Path("/srv/scryglass-data/public-packs"))
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    config = SyncConfig(
        root=root,
        public_root=args.public_root.resolve(),
        output_root=(root / "output/public_pack").resolve(),
        state_path=(root / "data/lol/runtime/postgame-sync.json").resolve(),
        lock_path=(root / "data/lol/runtime/postgame-sync.lock").resolve(),
        health_path=(root / "data/lol/runtime/postgame-sync-health.json").resolve(),
    )
    with exclusive_lock(config.lock_path):
        result = sync_once(config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
