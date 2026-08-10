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
from lol_kills.etl.oe_live_source import (
    _complete_player_game_ids,
    _identity_complete_player_game_ids,
    build_live_source,
)
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.export import pack_spec
from lol_kills.export.public_pack import (
    _complete_player_game_ids as _complete_identity_game_ids,
    export_public_pack,
    source_identity_sha256,
)
from lol_kills.export.upload_pack import publish_blob_pointers, upload_to_blob
from lol_kills.ratings.player_map_grades import CORE_INPUTS


RAW_RECEIPT = Path("data/lol/warehouse/raw/oe_api/tierlist-live-v1.json")
LIVE_ROOT = Path("data/lol/warehouse/parquet/oe_live")
LIVE_MAPS = LIVE_ROOT / "maps.parquet"
LIVE_TEAMS = LIVE_ROOT / "oe_team_games.parquet"
LIVE_PLAYERS = LIVE_ROOT / "oe_player_games.parquet"
REVIEWED_QUARANTINED_GAME_IDS = frozenset(
    {
        "oe:game:40e753290593c20c4cc990b1abfaf674",
        "oe:game:52db934e7e36169f9bf455bb00110045",
        "oe:game:9b9c8c1375765cffab98cacea34a3259",
        "oe:game:b53fb5b013e28c288688f5bb39d518b7",
        "oe:game:b91ad221257bc4357a3e0d9bc6ea75cd",
    }
)


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
    """Return the complete canonical source set and verify every new game."""

    requested = set(_canonical_ids(list(new_game_ids)))
    if len(requested) != len(new_game_ids):
        raise RefreshValidationError("new game identities are duplicated or malformed")
    maps = _identity_frame(root / LIVE_MAPS)
    teams = _identity_frame(root / LIVE_TEAMS, ("side", "result", "teamname"))
    players = _identity_frame(
        root / LIVE_PLAYERS,
        ("side", "position", "playername", *CORE_INPUTS),
    )
    map_ids = _canonical_ids(maps["_game_id"].tolist())
    if len(map_ids) != len(maps):
        raise RefreshValidationError("live maps are not one row per canonical game identity")
    valid_team_ids: set[str] = set()
    for game_id, team_rows in teams.groupby("_game_id", sort=False):
        sides = set(team_rows["side"].astype(str).str.title())
        results = set(pd.to_numeric(team_rows["result"], errors="coerce").dropna().astype(int))
        team_names = team_rows["teamname"].astype("string").fillna("").str.strip()
        if (
            len(team_rows) == 2
            and sides == {"Blue", "Red"}
            and results == {0, 1}
            and not team_names.eq("").any()
            and team_names.nunique() == 2
        ):
            valid_team_ids.add(str(game_id))
    legacy_identity_complete_ids = _complete_identity_game_ids(
        players.assign(game_uid=players["_game_id"])
    )
    identity_complete_ids = _identity_complete_player_game_ids(
        players.assign(game_uid=players["_game_id"])
    )
    statistics_complete_ids = _complete_player_game_ids(players)
    accepted_ids = sorted(set(map_ids).intersection(valid_team_ids, identity_complete_ids))
    missing_requested = sorted(requested.difference(statistics_complete_ids))
    if missing_requested:
        game_id = missing_requested[0]
        raise RefreshValidationError(
            f"game {game_id} is absent from the complete OE source set"
        )
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
        statistics = player_rows[list(CORE_INPUTS)].apply(pd.to_numeric, errors="coerce")
        if statistics.isna().any().any():
            raise RefreshValidationError(f"game {game_id} has incomplete player statistics")
        nonnegative = (
            "kills", "deaths", "assists", "teamkills", "dpm", "damageshare",
            "totalgold", "cspm", "wpm", "wcpm",
        )
        if statistics[list(nonnegative)].lt(0).any().any() or statistics["gamelength"].le(0).any():
            raise RefreshValidationError(f"game {game_id} has invalid player statistics")
        if statistics["kills"].gt(statistics["teamkills"]).any() or statistics["damageshare"].gt(1).any():
            raise RefreshValidationError(f"game {game_id} has invalid player statistics")
        for side in ("Blue", "Red"):
            side_rows = player_rows[player_rows["side"].astype(str).str.title().eq(side)]
            if len(side_rows) != 5 or set(side_rows["position"].astype(str).str.casefold()) != roles:
                raise RefreshValidationError(f"game {game_id} has malformed {side} roles")
            damage_share = pd.to_numeric(side_rows["damageshare"], errors="coerce").sum()
            if abs(float(damage_share) - 1) > 1e-6:
                raise RefreshValidationError(f"game {game_id} has malformed {side} damage share")
    return {
        "game_ids": accepted_ids,
        "statistics_complete_game_ids": sorted(statistics_complete_ids),
        "legacy_identity_game_ids": sorted(
            set(map_ids).intersection(valid_team_ids, legacy_identity_complete_ids)
        ),
        "game_count": len(accepted_ids),
        "identity_sha256": source_identity_sha256(accepted_ids),
        "candidate_game_count": len(map_ids),
        "rejected_incomplete_game_count": len(set(map_ids).difference(accepted_ids)),
    }


def _published_pack_source(public_root: Path) -> dict[str, Any] | None:
    manifest = _load_json(public_root / "manifest.json")
    if not manifest:
        return None
    ratings = manifest.get("ratings") or {}
    try:
        game_count = int(ratings["source_game_count"])
    except (KeyError, TypeError, ValueError) as error:
        raise RefreshValidationError("published pack has no valid source game count") from error
    identity_sha256 = str(ratings.get("source_identity_sha256") or "")
    if game_count < 1 or len(identity_sha256) != 64:
        raise RefreshValidationError("published pack has no valid source identity binding")
    return {
        "pack_id": str(manifest.get("pack_id") or ""),
        "game_count": game_count,
        "identity_sha256": identity_sha256,
    }


def _continuity_baseline(
    config: SyncConfig,
    state: dict[str, Any],
) -> tuple[list[str], dict[str, Any] | None]:
    """Resolve the exact source set behind the currently published pack."""

    binding = _published_pack_source(config.public_root)
    state_ids = _canonical_ids(state.get("published_game_ids", []))
    if binding is None:
        return state_ids, None
    if (
        len(state_ids) == binding["game_count"]
        and source_identity_sha256(state_ids) == binding["identity_sha256"]
    ):
        return state_ids, binding
    current_source = validate_live_source(config.root, [])
    if (
        current_source["game_count"] != binding["game_count"]
        or current_source["identity_sha256"] != binding["identity_sha256"]
    ):
        legacy_ids = _canonical_ids(current_source.get("legacy_identity_game_ids", []))
        if (
            len(legacy_ids) != binding["game_count"]
            or source_identity_sha256(legacy_ids) != binding["identity_sha256"]
        ):
            raise RefreshValidationError(
                "current source cache does not match the published pack; exact continuity cannot be proved"
            )
        return legacy_ids, binding
    return current_source["game_ids"], binding


def validate_source_continuity(
    baseline_game_ids: Sequence[str],
    source: dict[str, Any],
    published: dict[str, Any] | None,
) -> dict[str, Any]:
    """Reject a candidate source when a completed published map disappears."""

    baseline = set(_canonical_ids(list(baseline_game_ids)))
    candidate = set(_canonical_ids(source.get("game_ids", [])))
    missing = baseline.difference(candidate)
    quarantined = sorted(missing.intersection(REVIEWED_QUARANTINED_GAME_IDS))
    unexpected = sorted(missing.difference(REVIEWED_QUARANTINED_GAME_IDS))
    if unexpected:
        raise RefreshValidationError(
            f"source continuity failed; {len(unexpected)} published completed maps disappeared; first={unexpected[0]}"
        )
    if (
        published is not None
        and source["game_count"] + len(quarantined) < published["game_count"]
    ):
        raise RefreshValidationError("source continuity failed; candidate game count decreased")
    return {
        "previous_pack_id": published.get("pack_id") if published else None,
        "previous_game_count": len(baseline),
        "candidate_game_count": source["game_count"],
        "retained_game_count": len(baseline.intersection(candidate)),
        "added_game_count": len(candidate.difference(baseline)),
        "missing_game_count": len(missing),
        "quarantined_game_ids": quarantined,
        "quarantined_identity_sha256": source_identity_sha256(quarantined),
    }


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
    """Publish an immutable directory, then replace local and Blob pointers last."""

    pack_id = str(manifest.get("pack_id") or "")
    if not pack_id or not pack_dir.is_dir():
        raise RefreshValidationError("pack publication has no valid source directory")
    public_root.mkdir(parents=True, exist_ok=True)
    destination = public_root / pack_id
    if destination.exists():
        raise FileExistsError(f"immutable pack already exists: {destination}")
    staging = public_root / f".{pack_id}.{uuid.uuid4().hex}.incoming"
    token = (
        os.environ.get("BLOB_READ_WRITE_TOKEN")
        or os.environ.get("VERCEL_BLOB_READ_WRITE_TOKEN")
        or ""
    ).strip()
    blob_base = ""
    if token:
        urls = upload_to_blob(pack_dir, pack_id, token)
        if not urls:
            raise RefreshValidationError("pack Blob publication uploaded no files")
        sample = next(iter(urls.values()))
        marker = f"/packs/{pack_id}/"
        if marker not in sample:
            raise RefreshValidationError("pack Blob publication returned an invalid URL")
        blob_base = sample.rsplit(marker, 1)[0] + f"/packs/{pack_id}"
    published = dict(manifest)
    published["base_url"] = blob_base or f"/packs/{pack_id}"
    try:
        shutil.copytree(pack_dir, staging)
        _atomic_json(staging / "manifest.json", published)
        os.replace(staging, destination)
        _atomic_json(public_root / "manifest.json", published)
        blob_pointers = (
            publish_blob_pointers(token, pack_id, manifest, base_url=blob_base)
            if token
            else {}
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "pack_id": pack_id,
        "destination": str(destination),
        "runtime": "blob" if token else "local_only",
        "base_url": published["base_url"],
        "blob_pointers": blob_pointers,
    }


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
    force: bool = False,
) -> dict[str, Any]:
    checked_at = now or datetime.now(timezone.utc)
    state = _load_json(config.state_path)
    baseline_ids, published_source = _continuity_baseline(config, state)
    known = set(baseline_ids)
    _write_health(config, "checking", checked_at)
    try:
        ingest_fn(
            config.root,
            start=pd.Timestamp(checked_at - timedelta(hours=config.window_hours)),
            end=pd.Timestamp(checked_at),
            lookback_days=config.lookback_days,
            discovery_cache_hours=config.discovery_cache_hours,
            max_workers=config.max_workers,
        )
        observed = set(_receipt_game_ids(config.root))
        discovered_ids = sorted(observed - known)
        if not discovered_ids and not force:
            result = {"status": "no_change", "new_game_ids": [], "pack_id": state.get("pack_id")}
            _atomic_json(config.state_path, {**state, **result, "published_game_ids": sorted(known)})
            _write_health(config, "ok", checked_at, pack_id=result["pack_id"])
            return result
        build_live_fn(config.root)
        candidate_source = validate_live_fn(config.root, [])
        candidate_continuity = validate_source_continuity(
            baseline_ids, candidate_source, published_source
        )
        statistics_complete = set(candidate_source.get("statistics_complete_game_ids", []))
        new_ids = sorted(set(discovered_ids).intersection(statistics_complete))
        pending_ids = sorted(set(discovered_ids).difference(statistics_complete))
        if not new_ids and not force:
            status = "waiting_for_details" if pending_ids else "no_change"
            result = {
                "status": status,
                "new_game_ids": [],
                "pending_game_ids": pending_ids,
                "pack_id": state.get("pack_id"),
            }
            _atomic_json(config.state_path, {**state, **result, "published_game_ids": sorted(known)})
            _write_health(config, status if status != "no_change" else "ok", checked_at, pending_game_ids=pending_ids)
            return result
        source = validate_live_fn(config.root, new_ids)
        publish_ids = sorted(
            set(baseline_ids)
            .difference(candidate_continuity["quarantined_game_ids"])
            .union(new_ids)
        )
        source_ids = set(source["game_ids"])
        if not set(publish_ids).issubset(source_ids):
            raise RefreshValidationError("accepted OE source is missing a publishable completed map")
        source = {
            **source,
            "game_ids": publish_ids,
            "game_count": len(publish_ids),
            "identity_sha256": source_identity_sha256(publish_ids),
        }
        continuity = validate_source_continuity(baseline_ids, source, published_source)
        pack_id = f"v{checked_at.strftime('%Y.%m.%d.%H%M%S')}"
        manifest = export_pack_fn(
            years=config.years,
            out_root=config.output_root,
            pack_id=pack_id,
            project_root=config.root,
            allowed_game_ids=publish_ids,
        )
        validation = validate_pack_fn(config.output_root / pack_id, manifest, source)
        publication = publish_pack_fn(config.output_root / pack_id, manifest, config.public_root)
        validation["continuity"] = continuity
        result = {
            "status": "published",
            "new_game_ids": new_ids,
            "pending_game_ids": pending_ids,
            "pack_id": pack_id,
            "validation": validation,
            "publication": publication,
        }
        _atomic_json(config.state_path, {**result, "published_game_ids": source["game_ids"]})
        _write_health(config, "ok", checked_at, pack_id=pack_id, new_game_ids=new_ids, pending_game_ids=pending_ids)
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
    parser.add_argument("--force", action="store_true", help="publish a validated pack even when no new game ID appears")
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
        result = sync_once(config, force=args.force)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
