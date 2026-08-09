"""Export a compressed, versioned public reproduction pack (2025–2026 default).

Usage:
  python3 -m lol_kills.export.public_pack
  python3 -m lol_kills.export.public_pack --years 2025,2026 --out output/public_pack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lol_kills.export import pack_spec as spec
from lol_kills.export.pack_records import (
    build_maps_frame_from_team_games,
    build_player_records,
    build_team_records,
    filter_public_team_rating_maps,
)
from lol_kills.export.player_metadata import build_player_metadata
from lol_kills.etl.competition import TAXONOMY_VERSION, canonicalize_competition_frame
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.dual_elo import build_dual_ratings, lineup_hashes_from_players
from lol_kills.ratings.hierarchical_bt import build_team_weekly_ranks, fit_hierarchical_bt
from lol_kills.ratings.player_elo import (
    build_maps_frame_from_players,
    build_player_ratings,
    build_player_weekly_ranks,
)

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "data" / "lol" / "warehouse" / "parquet"
LIVE_WAREHOUSE = WAREHOUSE / "oe_live"
FEATURES = ROOT / "data" / "lol" / "features"
MODELS = ROOT / "data" / "lol" / "models"
TEAMS_JSON = ROOT / "web" / "composer" / "teams.json"
DEFAULT_OUT = ROOT / "output" / "public_pack"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _present(cols: Sequence[str], available: Iterable[str]) -> list[str]:
    avail = set(available)
    return [c for c in cols if c in avail]


def _filter_years(table: pa.Table, years: Sequence[int], year_cols: Sequence[str]) -> pa.Table:
    years_list = list(years)
    # The live OE overlay can carry both the original ``year`` and the
    # normalized ``oe_year``.  They differ for a small API overlay.  Prefer
    # the normalized source year instead of widening the map set with OR.
    available = set(table.column_names)
    col = "oe_year" if "oe_year" in available else next(
        (value for value in year_cols if value in available),
        None,
    )
    if col is None:
        return table
    arr = table[col]
    try:
        as_int = pc.cast(arr, pa.int64(), safe=False)
    except Exception:
        as_int = pc.cast(pc.utf8_to_int(pc.cast(arr, pa.string())), pa.int64(), safe=False)
    mask = pc.is_in(as_int, value_set=pa.array(years_list, type=pa.int64()))
    return table.filter(mask)


def _ensure_year_column(table: pa.Table) -> pa.Table:
    """Add a UTC-derived year when a live map overlay has no source year."""
    if "year" in table.column_names or "oe_year" in table.column_names:
        return table
    if "date" not in table.column_names:
        return table
    frame = table.to_pandas()
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["year"] = dates.dt.year.astype("Int64")
    return pa.Table.from_pandas(frame, preserve_index=False)


_CANONICAL_COMPETITION_COLUMNS = frozenset(
    {
        "league_source",
        "competition_scope",
        "event_kind",
        "is_international",
        "is_interregional",
        "competition_tier",
    }
)


def _canonical_pack_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep canonical source frames without copying wide OE columns twice."""

    if _CANONICAL_COMPETITION_COLUMNS.issubset(frame.columns):
        return frame
    return canonicalize_competition_frame(frame)


def _normalized_game_uid(frame: pd.DataFrame) -> pd.Series:
    """Use the source game UID and fall back to the OE game ID per row."""
    if not {"game_uid", "gameid", "oe_gameid"}.intersection(frame.columns):
        raise ValueError("rating source has no game identity column")
    fallback = frame["gameid"] if "gameid" in frame.columns else None
    values: list[str] = []
    if "game_uid" in frame.columns:
        source = frame["game_uid"]
    elif "gameid" in frame.columns:
        source = frame["gameid"]
    else:
        source = frame["oe_gameid"]
    for index, value in source.items():
        fallback_value = fallback.loc[index] if fallback is not None else None
        values.append(canonical_source_game_key(value, fallback_value))
    return pd.Series(values, index=frame.index, dtype="string").replace("", pd.NA)


def _canonicalize_game_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply one canonical source key to every game identity column."""

    if not {"game_uid", "gameid", "oe_gameid"}.intersection(frame.columns):
        return frame
    normalized = _normalized_game_uid(frame)
    for column in ("game_uid", "gameid", "oe_gameid"):
        if column in frame.columns:
            frame[column] = normalized
    return frame


def _source_file_meta(path: Path) -> dict[str, Any]:
    return {
        "locator": path.name,
        "bytes": int(path.stat().st_size),
        "sha256": _sha256(path),
    }


def _write_parquet(table: pa.Table, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=6,
        use_dictionary=True,
    )
    return {
        "path": str(path.name) if path.parent == path.parent else path.as_posix(),
        "relative": None,  # filled by caller
        "rows": table.num_rows,
        "cols": table.num_columns,
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "columns": table.column_names,
    }


def _join_history_years(
    history: pa.Table,
    maps: pa.Table,
    years: Sequence[int],
) -> pa.Table:
    """Keep history rows whose game_uid appears in year-filtered maps."""
    if "game_uid" not in history.column_names or "game_uid" not in maps.column_names:
        return history
    uids = maps.column("game_uid").unique()
    return history.filter(pc.is_in(history["game_uid"], value_set=uids))


def _draft_coverage(maps: pd.DataFrame, players: pd.DataFrame) -> dict[str, Any]:
    """Require every public map to have a verified ten-champion composition.

    OE's map-level pick columns preserve draft order, while GRID and a few OE
    anomalies only carry the final champion lineup on participant rows. Both
    are valid composition sources, but participant fallbacks must contain one
    champion in each canonical role on both sides.
    """

    roles = ("top", "jng", "mid", "bot", "sup")

    def known(value: Any) -> bool:
        if pd.isna(value):
            return False
        return str(value).strip().lower() not in {"", "unknown", "nan", "none", "null"}

    def role(value: Any) -> str | None:
        raw = str(value or "").strip().lower()
        if raw == "top" or raw.startswith("top"):
            return "top"
        if raw in {"jng", "jungle"} or raw.startswith("jungler"):
            return "jng"
        if raw == "mid" or raw.startswith("middle"):
            return "mid"
        if raw in {"bot", "adc"} or raw.startswith("bottom"):
            return "bot"
        if raw in {"sup", "utility"} or raw.startswith("support"):
            return "sup"
        return None

    if maps.empty:
        return {
            "maps": 0,
            "map_pick_rows": 0,
            "participant_fallback_rows": 0,
            "complete_rows": 0,
            "coverage_rate": 1.0,
        }

    pick_columns = [
        f"{side}_pick{index}"
        for side in ("blue", "red")
        for index in range(1, 6)
    ]
    map_pick_complete = pd.Series(False, index=maps.index)
    if all(column in maps.columns for column in pick_columns):
        map_pick_complete = maps[pick_columns].map(known).sum(axis=1).eq(10)

    player_key = "gameid" if "gameid" in players.columns else "game_uid"
    map_id = maps.get("oe_gameid", pd.Series(index=maps.index, dtype=object))
    map_fallback = maps.get("game_uid", pd.Series(index=maps.index, dtype=object))
    map_keys = pd.Series(
        [
            canonical_source_game_key(value, map_fallback.loc[index])
            for index, value in map_id.items()
        ],
        index=maps.index,
        dtype="string",
    )
    complete_participant_games: set[str] = set()
    required_player_columns = {player_key, "side", "position", "champion"}
    if not players.empty and required_player_columns.issubset(players.columns):
        relevant = players[
            players["champion"].map(known)
            & players["side"].astype(str).str.title().isin(["Blue", "Red"])
        ].copy()
        relevant["_side"] = relevant["side"].astype(str).str.title()
        relevant["_role"] = relevant["position"].map(role)
        relevant = relevant[relevant["_role"].notna()]
        side_summary = (
            relevant.groupby([player_key, "_side"], sort=False)
            .agg(
                rows=("champion", "size"),
                roles=("_role", "nunique"),
                champions=("champion", "nunique"),
            )
            .reset_index()
        )
        complete_sides = side_summary[
            side_summary["rows"].eq(5)
            & side_summary["roles"].eq(len(roles))
            & side_summary["champions"].eq(5)
        ]
        complete_games = complete_sides.groupby(player_key, sort=False)["_side"].agg(
            lambda sides: set(sides) == {"Blue", "Red"}
        )
        complete_participant_games = {
            canonical_source_game_key(game_id)
            for game_id, valid in complete_games.items()
            if valid and canonical_source_game_key(game_id)
        }

    participant_complete = map_keys.isin(complete_participant_games)
    complete = map_pick_complete | participant_complete
    if not complete.all():
        missing = map_keys[~complete].head(10).tolist()
        raise ValueError(
            "Public pack draft coverage failed: "
            f"{int((~complete).sum())} map(s) have neither ten map picks nor "
            f"five role-aligned participant champions per side; examples={missing}"
        )

    fallback = ~map_pick_complete & participant_complete
    return {
        "maps": int(len(maps)),
        "map_pick_rows": int(map_pick_complete.sum()),
        "participant_fallback_rows": int(fallback.sum()),
        "complete_rows": int(complete.sum()),
        "coverage_rate": float(complete.mean()),
    }


def export_public_pack(
    *,
    years: Sequence[int] | None = None,
    out_root: Path | None = None,
    pack_id: str | None = None,
    warehouse_root: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    years = tuple(years or spec.DEFAULT_YEARS)
    project = Path(project_root or ROOT).resolve()
    features_root = project / "data" / "lol" / "features"
    models_root = project / "data" / "lol" / "models"
    teams_json_path = project / "web" / "composer" / "teams.json"
    pdf_root = project / "output" / "pdf"
    warehouse = Path(
        warehouse_root
        if warehouse_root is not None
        else project / "data" / "lol" / "warehouse" / "parquet" / "oe_live"
        if (project / "data" / "lol" / "warehouse" / "parquet" / "oe_live" / "meta.json").exists()
        else project / "data" / "lol" / "warehouse" / "parquet"
    )
    # Include UTC time so the 15-minute freshness workflow can publish more
    # than one immutable pack per day without colliding in Blob storage.
    stamp = datetime.now(timezone.utc).strftime("%Y.%m.%d.%H%M")
    pack_id = pack_id or f"v{stamp}"
    out_root = Path(out_root or project / "output" / "public_pack")
    pack_dir = out_root / pack_id
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True)

    files_meta: list[dict[str, Any]] = []

    def register(meta: dict[str, Any], rel: str) -> None:
        meta = dict(meta)
        meta["relative"] = rel
        meta["path"] = rel
        files_meta.append(meta)

    # --- team games (partition by year) ---
    team_path = warehouse / "oe_team_games.parquet"
    team_source = _canonicalize_game_ids(_canonical_pack_frame(pq.read_table(team_path).to_pandas()))
    team_table = pa.Table.from_pandas(team_source, preserve_index=False)
    team_table = _filter_years(team_table, years, ("year", "oe_year"))
    team_rating_frame = team_table.to_pandas()
    team_cols = _present(spec.TEAM_COLS, team_table.column_names)
    team = team_table.select(team_cols)
    for y in years:
        part = team
        if "year" in team.column_names:
            part = team.filter(pc.equal(pc.cast(team["year"], pa.int64(), safe=False), y))
        elif "oe_year" in team.column_names:
            part = team.filter(pc.equal(pc.cast(team["oe_year"], pa.int64(), safe=False), y))
        dest = pack_dir / "team_games" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"team_games/year={y}/part.parquet")
    team_for_records = team_rating_frame
    team_maps_for_ratings = build_maps_frame_from_team_games(team_for_records)
    del team_for_records, team_rating_frame, team_source, team_table, team

    # --- player games ---
    player_path = warehouse / "oe_player_games.parquet"
    player_source = _canonicalize_game_ids(_canonical_pack_frame(pq.read_table(player_path).to_pandas()))
    player_table = pa.Table.from_pandas(player_source, preserve_index=False)
    player_table = _filter_years(player_table, years, ("year", "oe_year"))
    player_rating_frame = player_table.to_pandas()
    player_rating_columns = [
        column
        for column in (
            "gameid",
            "game_uid",
            "date",
            "league",
            "result",
            "side",
            "position",
            "teamname",
            "playername",
        )
        if column in player_rating_frame.columns
    ]
    player_rating_input = player_rating_frame[player_rating_columns].copy()
    player_record_columns = [
        column
        for column in (
            "league",
            "league_source",
            "competition_scope",
            "event_kind",
            "is_international",
            "is_interregional",
            "competition_tier",
            "date",
            "position",
            "side",
            "playername",
            "teamname",
            "result",
            "tournament",
        )
        if column in player_rating_frame.columns
    ]
    player_records_frame = player_rating_frame[player_record_columns].copy()
    player_cols = _present(spec.PLAYER_COLS, player_table.column_names)
    player = player_table.select(player_cols)
    for y in years:
        if "year" in player.column_names:
            part = player.filter(pc.equal(pc.cast(player["year"], pa.int64(), safe=False), y))
        else:
            part = player.filter(pc.equal(pc.cast(player["oe_year"], pa.int64(), safe=False), y))
        dest = pack_dir / "player_games" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"player_games/year={y}/part.parquet")
    del player_source, player_table, player

    # --- maps ---
    maps_path = warehouse / "maps.parquet"
    maps = pq.read_table(maps_path)
    # Re-apply the canonical map contract only for older sources. Live maps
    # already come from the canonical team-game adapter.
    if not _CANONICAL_COMPETITION_COLUMNS.issubset(maps.column_names):
        maps = pa.Table.from_pandas(canonicalize_competition_frame(maps.to_pandas()), preserve_index=False)
    maps = _ensure_year_column(maps)
    map_cols = spec.maps_columns(maps.column_names)
    maps = maps.select(map_cols)
    maps = _filter_years(maps, years, ("year", "oe_year"))
    maps_for_records = _canonicalize_game_ids(maps.to_pandas())
    maps = pa.Table.from_pandas(maps_for_records, preserve_index=False)
    draft_coverage = _draft_coverage(maps_for_records, player_rating_frame)
    source_as_of = pd.to_datetime(maps_for_records["date"], utc=True, errors="coerce").max()
    if pd.isna(source_as_of):
        raise RuntimeError("public pack source has no usable map dates")
    # The feature-oriented maps table intentionally covers the major/public
    # event slice.  Team ladders need the full OE team-game population so
    # Tier 2 and Tier 3 organizations receive both records and estimates.
    live_source = (warehouse / "meta.json").exists()
    rating_input = (
        maps_for_records
        if live_source
        else team_maps_for_ratings if not team_maps_for_ratings.empty else maps_for_records
    )
    rating_input = filter_public_team_rating_maps(rating_input)
    if rating_input.empty:
        raise RuntimeError("public pack team rating source has no eligible team maps")
    player_maps_for_ratings = build_maps_frame_from_players(player_rating_input)
    if player_maps_for_ratings.empty:
        raise RuntimeError("public pack rating source has no complete player maps")
    if (warehouse / "meta.json").exists():
        map_ids = set(_normalized_game_uid(maps_for_records).dropna().astype(str))
        team_ids = set(_normalized_game_uid(team_maps_for_ratings).dropna().astype(str))
        player_ids = set(_normalized_game_uid(player_maps_for_ratings).dropna().astype(str))
        if team_ids != map_ids or player_ids != map_ids:
            raise RuntimeError(
                "OE live public pack inputs do not share the deduplicated map set; "
                f"maps={len(map_ids)} team={len(team_ids)} player={len(player_ids)}"
            )
    player_rating_input["game_uid"] = _normalized_game_uid(player_rating_input)
    if player_rating_input["game_uid"].isna().any():
        raise RuntimeError("public pack rating source has rows without a game identity")
    build_dual_ratings(
        rating_input,
        lineup_by_game=lineup_hashes_from_players(player_rating_input),
    )
    build_player_ratings(player_maps_for_ratings, player_rating_input)
    player_snapshot_path = features_root / "player_ratings_snapshot.parquet"
    if player_snapshot_path.exists():
        player_snapshot = pd.read_parquet(player_snapshot_path)
        player_snapshot = attach_player_evidence(
            player_snapshot,
            source_as_of=source_as_of,
        )
        player_snapshot.to_parquet(player_snapshot_path, index=False)
    public_ratings, public_ratings_meta = fit_hierarchical_bt(rating_input, write=True)
    public_ratings_meta["pack_years"] = list(years)
    public_ratings_meta["rating_window"] = "full canonical OE team-game window as this pack"
    public_ratings_meta["source_as_of"] = source_as_of.isoformat().replace("+00:00", "Z")
    public_ratings_meta["source_mode"] = "oe_live" if live_source else "warehouse"
    public_ratings_meta["evidence_contract"] = "2026-08-09.1"
    features_root.mkdir(parents=True, exist_ok=True)
    (features_root / "ratings_meta.json").write_text(
        json.dumps(public_ratings_meta, indent=2),
        encoding="utf-8",
    )
    (features_root / "ratings_hierarchical_meta.json").write_text(
        json.dumps(public_ratings_meta, indent=2),
        encoding="utf-8",
    )
    for y in years:
        if "year" in maps.column_names:
            part = maps.filter(pc.equal(pc.cast(maps["year"], pa.int64(), safe=False), y))
        else:
            part = maps.filter(pc.equal(pc.cast(maps["oe_year"], pa.int64(), safe=False), y))
        dest = pack_dir / "maps" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"maps/year={y}/part.parquet")

    # --- features snapshots (team ladder uses the same pack window; player
    # snapshot remains the full roster-history artifact) ---
    feat_dir = pack_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    player_records_payload = build_player_records(player_records_frame)
    team_records_payload = build_team_records(rating_input)

    team_weekly_ranks = build_team_weekly_ranks(
        rating_input,
        as_of=source_as_of,
        min_series=5,
    )
    team_weekly_dest = feat_dir / "team_weekly_ranks.json"
    team_weekly_dest.write_text(json.dumps(team_weekly_ranks, indent=2), encoding="utf-8")
    register(
        {
            "rows": len(team_weekly_ranks.get("by_team", {})),
            "cols": None,
            "bytes": team_weekly_dest.stat().st_size,
            "sha256": _sha256(team_weekly_dest),
            "columns": None,
        },
        "features/team_weekly_ranks.json",
    )

    weekly_ranks = build_player_weekly_ranks(
        player_maps_for_ratings,
        player_rating_input,
        as_of=pd.to_datetime(maps_for_records["date"], utc=True, errors="coerce").max(),
        min_games=20,
    )
    player_meta_path = features_root / "player_ratings_meta.json"
    if player_meta_path.exists():
        player_meta = json.loads(player_meta_path.read_text(encoding="utf-8"))
        player_meta["source_as_of"] = source_as_of.isoformat().replace("+00:00", "Z")
        player_meta["source_mode"] = "oe_live" if live_source else "warehouse"
        player_meta["window_years"] = list(years)
        player_meta["evidence_contract"] = "2026-08-09.1"
        player_meta_path.write_text(json.dumps(player_meta, indent=2), encoding="utf-8")
    weekly_dest = feat_dir / "player_weekly_ranks.json"
    weekly_dest.write_text(json.dumps(weekly_ranks, indent=2), encoding="utf-8")
    register(
        {
            "rows": len(weekly_ranks.get("by_player", {})),
            "cols": None,
            "bytes": weekly_dest.stat().st_size,
            "sha256": _sha256(weekly_dest),
            "columns": None,
        },
        "features/player_weekly_ranks.json",
    )

    player_metadata = build_player_metadata(
        player_records_frame["playername"].dropna().astype(str).unique(),
        player_context={
            player: record.get("current_team")
            for player, record in player_records_payload.items()
        },
    )
    metadata_dest = feat_dir / "player_metadata.json"
    metadata_dest.write_text(json.dumps(player_metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    register(
        {
            "rows": len(player_metadata),
            "cols": None,
            "bytes": metadata_dest.stat().st_size,
            "sha256": _sha256(metadata_dest),
            "columns": None,
        },
        "features/player_metadata.json",
    )

    # These records are intentionally built from the same year-filtered
    # canonical rows that are exported above.  This avoids mixing a full
    # history snapshot with a different pack-window win rate.
    for filename, payload in (
        ("team_records.json", team_records_payload),
        ("player_records.json", player_records_payload),
    ):
        dest = feat_dir / filename
        dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        register(
            {
                "rows": len(payload),
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"features/{filename}",
        )

    for src_name, cols, out_name in (
        ("ratings_snapshot.parquet", spec.RATINGS_SNAPSHOT_COLS, "ratings_snapshot.parquet"),
        (
            "player_ratings_snapshot.parquet",
            spec.PLAYER_RATINGS_SNAPSHOT_COLS,
            "player_ratings_snapshot.parquet",
        ),
    ):
        src = features_root / src_name
        if src_name == "ratings_snapshot.parquet":
            t = pa.Table.from_pandas(public_ratings, preserve_index=False)
        elif not src.exists():
            continue
        else:
            t = pq.read_table(src)
        t = t.select(_present(cols, t.column_names))
        dest = feat_dir / out_name
        register(_write_parquet(t, dest), f"features/{out_name}")
        # JSON twin for fast atlas ladders (no WASM)
        if out_name.endswith("_snapshot.parquet"):
            jdest = feat_dir / out_name.replace(".parquet", ".json")
            rows = t.to_pylist()
            jdest.write_text(json.dumps(rows), encoding="utf-8")
            register(
                {
                    "rows": len(rows),
                    "cols": t.num_columns,
                    "bytes": jdest.stat().st_size,
                    "sha256": _sha256(jdest),
                    "columns": t.column_names,
                },
                f"features/{jdest.name}",
            )

    # --- features history year-filtered via maps game_uid ---
    maps_all = maps

    for src_name, cols, out_name in (
        ("ratings.parquet", spec.RATINGS_HISTORY_COLS, "ratings_history.parquet"),
        (
            "player_ratings.parquet",
            spec.PLAYER_RATINGS_HISTORY_COLS,
            "player_ratings_history.parquet",
        ),
    ):
        src = features_root / src_name
        if not src.exists():
            continue
        t = pq.read_table(src)
        t = t.select(_present(cols, t.column_names))
        t = _join_history_years(t, maps_all, years)
        dest = feat_dir / out_name
        register(_write_parquet(t, dest), f"features/{out_name}")

    for meta_name in ("ratings_meta.json", "player_ratings_meta.json"):
        if meta_name == "ratings_meta.json":
            dest = feat_dir / meta_name
            dest.write_text(json.dumps(public_ratings_meta, indent=2), encoding="utf-8")
            register(
                {
                    "rows": None,
                    "cols": None,
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "columns": None,
                },
                f"features/{meta_name}",
            )
            continue
        src = features_root / meta_name
        if src.exists():
            dest = feat_dir / meta_name
            shutil.copy2(src, dest)
            register(
                {
                    "rows": None,
                    "cols": None,
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "columns": None,
                },
                f"features/{meta_name}",
            )

    # --- models ---
    models_dir = pack_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in spec.PINNED_MODEL_FILES:
        src = models_root / name
        if not src.exists():
            continue
        dest = models_dir / name
        shutil.copy2(src, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"models/{name}",
        )

    tier_dir = models_root / "tierlists_csv"
    if tier_dir.is_dir():
        out_tier = models_dir / "tierlists_csv"
        out_tier.mkdir(exist_ok=True)
        for csv in sorted(tier_dir.glob("*.csv")):
            dest = out_tier / csv.name
            shutil.copy2(csv, dest)
            register(
                {
                    "rows": None,
                    "cols": None,
                    "bytes": dest.stat().st_size,
                    "sha256": _sha256(dest),
                    "columns": None,
                },
                f"models/tierlists_csv/{csv.name}",
            )

    # --- void grubs study bundle ---
    grubs_dir = pack_dir / "studies" / "grubs"
    grubs_dir.mkdir(parents=True, exist_ok=True)
    for name in spec.GRUBS_MODEL_FILES:
        src = models_root / name
        if not src.exists():
            continue
        dest = grubs_dir / name
        shutil.copy2(src, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"studies/grubs/{name}",
        )
    for name in spec.GRUBS_PDF_FILES:
        src = pdf_root / name
        if not src.exists():
            continue
        dest = grubs_dir / name
        shutil.copy2(src, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            f"studies/grubs/{name}",
        )
    (grubs_dir / "STUDY_NOTE.txt").write_text(spec.GRUBS_STUDY_NOTE + "\n", encoding="utf-8")
    register(
        {
            "rows": None,
            "cols": None,
            "bytes": (grubs_dir / "STUDY_NOTE.txt").stat().st_size,
            "sha256": _sha256(grubs_dir / "STUDY_NOTE.txt"),
            "columns": None,
        },
        "studies/grubs/STUDY_NOTE.txt",
    )

    # Do not emit an apparently valid pack when the public Reproduce page
    # would show a cited file as missing.  In particular, publication refreshes
    # must carry the paper PDF and article inputs together with the ratings
    # files; a stale/partial pack is not an acceptable fallback.
    present_paths = {str(item["path"]) for item in files_meta}
    missing_public_files = sorted(
        set(spec.PUBLIC_REPRODUCTION_REQUIRED_FILES) - present_paths
    )
    if missing_public_files:
        raise RuntimeError(
            "public reproduction contract incomplete; missing: "
            + ", ".join(missing_public_files)
        )
    leaked_public_files = sorted(
        present_paths.intersection(spec.WITHHELD_PUBLIC_FILES)
    )
    if leaked_public_files:
        raise RuntimeError(
            "withheld public artifacts present: " + ", ".join(leaked_public_files)
        )

    # --- meta ---
    meta_dir = pack_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if teams_json_path.exists():
        dest = meta_dir / "teams.json"
        shutil.copy2(teams_json_path, dest)
        register(
            {
                "rows": None,
                "cols": None,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": None,
            },
            "meta/teams.json",
        )

    readme = pack_dir / "README.md"
    readme.write_text(
        spec.PACK_README.format(
            years=", ".join(str(y) for y in years),
            attribution=spec.ATTRIBUTION,
        ),
        encoding="utf-8",
    )

    oe_meta = warehouse / "oe_meta.json"
    refresh_meta = warehouse / "refresh_meta.json"
    ingest = {}
    grid_meta = warehouse / "grid_meta.json"
    live_meta = warehouse / "meta.json"
    for p in (live_meta, oe_meta, refresh_meta, grid_meta):
        if p.exists():
            try:
                key = "oe_live_meta" if p == live_meta else p.stem
                ingest[key] = json.loads(p.read_text())
            except json.JSONDecodeError:
                key = "oe_live_meta" if p == live_meta else p.stem
                ingest[key] = {"raw": p.read_text()[:500]}

    rating_source_files = {
        name: _source_file_meta(warehouse / name)
        for name in ("maps.parquet", "oe_team_games.parquet", "oe_player_games.parquet")
        if (warehouse / name).exists()
    }
    rating_artifact_paths = {
        item["path"]: {
            key: item[key]
            for key in ("sha256", "bytes", "rows")
            if key in item
        }
        for item in files_meta
        if item["path"]
        in {
            "features/ratings_snapshot.parquet",
            "features/ratings_snapshot.json",
            "features/player_ratings_snapshot.parquet",
            "features/player_ratings_snapshot.json",
            "features/team_weekly_ranks.json",
            "features/player_weekly_ranks.json",
        }
    }

    total_bytes = sum(f["bytes"] for f in files_meta)
    manifest: dict[str, Any] = {
        "pack_id": pack_id,
        "schema_version": spec.SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "filters": {
            "years": list(years),
            "leagues": "all_in_year_window",
            "leagues_note": spec.DEFAULT_LEAGUES_NOTE,
        },
        "identity": {
            "taxonomy_version": TAXONOMY_VERSION,
            "team_key": "one canonical organization identity across regional and international events",
            "league_source": "raw source label retained on rows for auditability",
            "competition_tier": "tier1/tier2/tier3 assigned from the canonical league taxonomy; international and interregional are separate scopes",
            "deprecated_leagues": {
                "LTA": "AMERICAS",
                "LTA N": "LCS",
                "LTA NORTH": "LCS",
                "LTA S": "CBLOL",
                "LTA SOUTH": "CBLOL",
            },
        },
        "attribution": spec.ATTRIBUTION,
        "quality": {"draft_coverage": draft_coverage},
        "excluded": [
            "warehouse/timelines",
            "warehouse/raw OE CSVs",
            "betting fair-odds / Slip Composer",
            "joblib models",
        ],
        "studies": {
            "grubs": {
                "path": "studies/grubs/",
                "note": spec.GRUBS_STUDY_NOTE,
                "entrypoints": [
                    "studies/grubs/grubs_article_contest_ev.json",
                    "studies/grubs/void_grubs_scrap_value_and_contest_rationality.pdf",
                    "studies/grubs/grubs_decision_numbers.json",
                ],
            }
        },
        "ingest": ingest,
        "ratings": {
            "source_mode": "oe_live" if live_meta.exists() else "warehouse",
            "source_as_of": source_as_of.isoformat().replace("+00:00", "Z"),
            "window_years": list(years),
            "map_rows": int(len(maps_for_records)),
            "team_rating_rows": int(len(rating_input)),
            "player_rating_rows": int(len(player_rating_frame)),
            "source_files": rating_source_files,
            "artifacts": rating_artifact_paths,
            "claim_ceiling": "Source-bound descriptive ratings and weekly rank movement only.",
        },
        "base_url": None,  # filled by upload / atlas config
        "total_bytes": total_bytes,
        "total_files": len(files_meta),
        "files": files_meta,
    }
    man_path = pack_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # also write latest pointer
    latest = out_root / "latest.json"
    latest.write_text(
        json.dumps({"pack_id": pack_id, "path": pack_id, "created_utc": manifest["created_utc"]}, indent=2),
        encoding="utf-8",
    )

    # atlas-facing copy of manifest (for static deploy before Blob upload)
    return manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export public LoL research reproduction pack")
    ap.add_argument("--years", default="2025,2026", help="Comma-separated years (default 2025,2026)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output root directory")
    ap.add_argument("--pack-id", default=None, help="Override pack id (default vYYYY.MM.DD)")
    ap.add_argument("--warehouse-root", type=Path, default=None, help="Use a source-root overlay for live refreshes")
    args = ap.parse_args(argv)
    years = tuple(int(x.strip()) for x in args.years.split(",") if x.strip())
    man = export_public_pack(
        years=years,
        out_root=args.out,
        pack_id=args.pack_id,
        warehouse_root=args.warehouse_root,
    )
    mb = man["total_bytes"] / (1024 * 1024)
    print(f"Wrote pack {man['pack_id']} → {args.out / man['pack_id']}")
    print(f"Files: {man['total_files']}  Size: {mb:.1f} MB  schema={man['schema_version']}")
    for f in man["files"]:
        print(f"  {f['path']}: {f['bytes']/1024:.0f} KB  rows={f.get('rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
