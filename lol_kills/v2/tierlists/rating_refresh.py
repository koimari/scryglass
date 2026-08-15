"""Refresh team and player rating artifacts from the current OE source."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.etl.oe_live_source import LIVE_MAP_OUTPUT, LIVE_PLAYER_OUTPUT, LIVE_TEAM_OUTPUT
from lol_kills.ratings.dual_elo import (
    DualEloConfig,
    apply_team_momentum_snapshot,
    build_dual_ratings,
    lineup_hashes_from_players,
)
from lol_kills.ratings.hierarchical_bt import build_team_weekly_ranks, fit_hierarchical_bt
from lol_kills.ratings.player_elo import (
    PlayerEloConfig,
    build_maps_frame_from_players,
    build_player_ratings,
    build_player_weekly_ranks,
)
from lol_kills.ratings.momentum_config import (
    DEFAULT_MOMENTUM_SCALE,
    DEFAULT_MOMENTUM_WINDOW_GAMES,
    momentum_manifest_metadata,
    require_public_momentum_disabled,
)

RATING_WINDOW_START = pd.Timestamp("2025-01-01T00:00:00Z")
RATING_YEARS = (2025, 2026)
OUTPUT_ROOT = Path("data/lol/v2/tierlists/rating-refresh")
OUTPUT = OUTPUT_ROOT / "rating-refresh-v1.json"
FEATURES_RELATIVE = Path("data/lol/features")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_timestamp(value: object) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def _normalized_game_uid(frame: pd.DataFrame) -> pd.Series:
    if "game_uid" not in frame.columns and "gameid" not in frame.columns:
        raise ValueError("OE rating input has no game identity column")
    if "game_uid" in frame.columns:
        game_uid = frame["game_uid"].astype("string")
        fallback = (
            frame["gameid"].astype("string")
            if "gameid" in frame.columns
            else pd.Series("", index=frame.index, dtype="string")
        )
        value = game_uid.where(game_uid.notna() & game_uid.str.strip().ne(""), fallback)
    else:
        value = frame["gameid"].astype("string")
    return value.where(value.notna() & value.str.strip().ne(""), pd.NA)


def _window_frame(
    frame: pd.DataFrame,
    *,
    cutoff: pd.Timestamp,
    label: str,
) -> pd.DataFrame:
    if "date" not in frame.columns:
        raise ValueError(f"OE {label} input has no date column")
    dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    mask = dates.ge(RATING_WINDOW_START) & dates.le(cutoff)
    result = frame.loc[mask].copy()
    result["date"] = dates.loc[mask]
    return result


def _bind_game_ids(
    frame: pd.DataFrame,
    *,
    map_ids: set[str],
    expected_rows_per_game: int,
    label: str,
) -> pd.DataFrame:
    result = frame.copy()
    result["game_uid"] = _normalized_game_uid(result)
    result = result[result["game_uid"].isin(map_ids)].copy()
    counts = result.groupby("game_uid", dropna=False).size()
    missing = sorted(map_ids.difference(set(counts.index.astype(str))))
    incomplete = counts[counts.ne(expected_rows_per_game)]
    if missing or not incomplete.empty:
        sample = [str(value) for value in incomplete.index[:5]]
        raise ValueError(
            f"OE {label} input is not complete for the deduplicated map set; "
            f"missing_games={len(missing)} incomplete_games={len(incomplete)} sample={sample}"
        )
    return result


def _artifact_meta(path: Path, repo_root: Path, *, rows: int | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "locator": str(path.relative_to(repo_root)),
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }
    if rows is not None:
        payload["rows"] = int(rows)
    return payload


def refresh_ratings(
    root: Path | str = Path("."),
    *,
    as_of: pd.Timestamp | None = None,
    min_games: int = 20,
    min_series: int = 5,
    previous_as_of: pd.Timestamp | None = None,
    momentum_window_games: int = DEFAULT_MOMENTUM_WINDOW_GAMES,
    momentum_scale: float = DEFAULT_MOMENTUM_SCALE,
) -> dict[str, Any]:
    require_public_momentum_disabled(
        window_games=momentum_window_games,
        scale=momentum_scale,
        entrypoint="refresh_ratings",
    )
    repo_root = Path(root).resolve()
    team_path = repo_root / LIVE_TEAM_OUTPUT
    player_path = repo_root / LIVE_PLAYER_OUTPUT
    maps_path = repo_root / LIVE_MAP_OUTPUT
    for path in (team_path, player_path, maps_path):
        if not path.is_file():
            raise FileNotFoundError(f"OE live rating input is missing: {path}")

    team_games = pd.read_parquet(team_path)
    players = pd.read_parquet(player_path)
    maps = pd.read_parquet(maps_path)
    if maps.empty or players.empty:
        raise ValueError("OE live rating input is empty")

    map_dates = pd.to_datetime(maps["date"], errors="coerce", utc=True)
    source_latest = map_dates.max()
    if pd.isna(source_latest):
        raise ValueError("OE live map input has no usable dates")
    cutoff = _utc_timestamp(as_of) if as_of is not None else source_latest
    cutoff = min(cutoff, source_latest)
    if cutoff < RATING_WINDOW_START:
        raise ValueError("OE live source has no games in the 2025–2026 rating window")

    maps = _window_frame(maps, cutoff=cutoff, label="map")
    maps["game_uid"] = _normalized_game_uid(maps)
    if maps["game_uid"].isna().any() or maps["game_uid"].duplicated().any():
        raise ValueError("OE live map input is not a deduplicated one-row-per-game source")
    map_ids = set(maps["game_uid"].astype(str))
    team_games = _bind_game_ids(
        _window_frame(team_games, cutoff=cutoff, label="team"),
        map_ids=map_ids,
        expected_rows_per_game=2,
        label="team",
    )
    players = _bind_game_ids(
        _window_frame(players, cutoff=cutoff, label="player"),
        map_ids=map_ids,
        expected_rows_per_game=10,
        label="player",
    )

    # ``FEATURES_DIR`` is resolved when the worker imports the ETL path
    # module.  It can point at the code checkout, so use the runtime-relative
    # locator for this refresh output.
    features_dir = repo_root / FEATURES_RELATIVE
    features_dir.mkdir(parents=True, exist_ok=True)

    player_maps = build_maps_frame_from_players(players)
    if player_maps.empty or set(player_maps["game_uid"].astype(str)) != map_ids:
        raise ValueError("OE live player rows do not form the complete deduplicated map set")
    lineup_hashes = lineup_hashes_from_players(players)
    # Keep every generated rating artifact under the requested runtime root.
    # The worker imports this module from its checkout, while ``root`` points
    # to the isolated runtime data tree.  The default output locations in the
    # rating modules follow the process working directory, which can point at
    # the checkout and make the refresh non-reproducible.
    team_rating_cfg = DualEloConfig(
        momentum_window_games=momentum_window_games,
        momentum_scale=momentum_scale,
    )
    player_rating_cfg = PlayerEloConfig(
        momentum_window_games=momentum_window_games,
        momentum_scale=momentum_scale,
    )
    build_dual_ratings(
        maps,
        cfg=team_rating_cfg,
        lineup_by_game=lineup_hashes,
        output_dir=features_dir,
    )
    build_player_ratings(
        player_maps,
        players,
        cfg=player_rating_cfg,
        output_dir=features_dir,
    )
    team_snapshot, team_meta = fit_hierarchical_bt(
        maps,
        write=True,
        output_dir=features_dir,
    )
    sequential_team_snapshot = pd.read_parquet(features_dir / "ratings_dual_snapshot.parquet")
    team_snapshot = apply_team_momentum_snapshot(
        team_snapshot,
        sequential_team_snapshot,
        team_rating_cfg,
    )
    team_snapshot.to_parquet(features_dir / "ratings_snapshot.parquet", index=False)
    team_meta["momentum"] = momentum_manifest_metadata(
        window_games=momentum_window_games,
        scale=momentum_scale,
    )
    (features_dir / "ratings_meta.json").write_text(
        json.dumps(team_meta, indent=2),
        encoding="utf-8",
    )
    team_weekly = build_team_weekly_ranks(
        maps,
        as_of=cutoff,
        min_series=min_series,
        previous_as_of=previous_as_of,
        current=team_snapshot,
    )
    player_weekly = build_player_weekly_ranks(
        player_maps,
        players,
        cfg=player_rating_cfg,
        as_of=cutoff,
        min_games=min_games,
        previous_as_of=previous_as_of,
    )
    (features_dir / "team_weekly_ranks.json").write_text(
        json.dumps(team_weekly, indent=2) + "\n",
        encoding="utf-8",
    )
    (features_dir / "player_weekly_ranks.json").write_text(
        json.dumps(player_weekly, indent=2) + "\n",
        encoding="utf-8",
    )

    output_path = repo_root / OUTPUT
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_as_of = cutoff.isoformat().replace("+00:00", "Z")
    maps_by_year = {
        str(year): int((pd.to_datetime(maps["date"], utc=True).dt.year == year).sum())
        for year in RATING_YEARS
    }
    player_rows_by_year = {
        str(year): int((pd.to_datetime(players["date"], utc=True).dt.year == year).sum())
        for year in RATING_YEARS
    }
    payload = {
        "schema_version": "scryglass:rating-refresh:v1",
        "source": {
            "mode": "oe_only",
            "window_years": list(RATING_YEARS),
            "window_start": RATING_WINDOW_START.isoformat().replace("+00:00", "Z"),
            "window_end": source_as_of,
            "player_locator": str(player_path.relative_to(repo_root)),
            "team_locator": str(team_path.relative_to(repo_root)),
            "maps_locator": str(maps_path.relative_to(repo_root)),
            "player_raw_sha256": _sha256(player_path),
            "team_raw_sha256": _sha256(team_path),
            "maps_raw_sha256": _sha256(maps_path),
            "as_of": source_as_of,
            "maps": int(len(maps)),
            "maps_by_year": maps_by_year,
            "team_rows": int(len(team_games)),
            "player_rows": int(len(players)),
            "player_rows_by_year": player_rows_by_year,
        },
        "team": {
            "snapshot_rows": int(len(team_snapshot)),
            "model": team_meta.get("model"),
            "weekly_rows": len(team_weekly.get("by_team", {})),
            "weekly_locator": "features/team_weekly_ranks.json",
            "movement_baseline": str(team_weekly.get("previous_as_of") or ""),
        },
        "player": {
            "snapshot_rows": int(len(pd.read_parquet(features_dir / "player_ratings_snapshot.parquet"))),
            "weekly_rows": len(player_weekly.get("by_player", {})),
            "weekly_locator": "features/player_weekly_ranks.json",
            "movement_baseline": str(player_weekly.get("previous_as_of") or ""),
        },
        "momentum": momentum_manifest_metadata(
            window_games=momentum_window_games,
            scale=momentum_scale,
        ),
        "artifacts": {
            "team_snapshot": _artifact_meta(
                features_dir / "ratings_snapshot.parquet",
                repo_root,
                rows=len(team_snapshot),
            ),
            "player_snapshot": _artifact_meta(
                features_dir / "player_ratings_snapshot.parquet",
                repo_root,
                rows=len(pd.read_parquet(features_dir / "player_ratings_snapshot.parquet")),
            ),
            "team_weekly": _artifact_meta(
                features_dir / "team_weekly_ranks.json",
                repo_root,
                rows=len(team_weekly.get("by_team", {})),
            ),
            "player_weekly": _artifact_meta(
                features_dir / "player_weekly_ranks.json",
                repo_root,
                rows=len(player_weekly.get("by_player", {})),
            ),
        },
        "weekly": {
            "team": team_weekly,
            "player": player_weekly,
        },
        "claim_ceiling": "These are source-bound descriptive rating refresh artifacts. They do not grant predictive, recommendation, or betting authority.",
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--min-games", type=int, default=20)
    parser.add_argument("--min-series", type=int, default=5)
    parser.add_argument("--previous-as-of", default=None)
    parser.add_argument(
        "--momentum-window-games",
        type=int,
        default=DEFAULT_MOMENTUM_WINDOW_GAMES,
        help="Explicit research momentum window in prior maps; default is zero",
    )
    parser.add_argument(
        "--momentum-scale",
        type=float,
        default=DEFAULT_MOMENTUM_SCALE,
        help="Explicit research momentum scale in rating points; default is zero",
    )
    args = parser.parse_args()
    payload = refresh_ratings(
        args.root,
        as_of=pd.Timestamp(args.as_of) if args.as_of else None,
        min_games=args.min_games,
        min_series=args.min_series,
        previous_as_of=pd.Timestamp(args.previous_as_of) if args.previous_as_of else None,
        momentum_window_games=args.momentum_window_games,
        momentum_scale=args.momentum_scale,
    )
    print(json.dumps({
        "source_as_of": payload["source"]["as_of"],
        "maps": payload["source"]["maps"],
        "team_snapshot_rows": payload["team"]["snapshot_rows"],
        "player_snapshot_rows": payload["player"]["snapshot_rows"],
        "team_weekly_rows": payload["team"]["weekly_rows"],
        "player_weekly_rows": payload["player"]["weekly_rows"],
        "output": str(OUTPUT),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
