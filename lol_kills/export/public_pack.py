"""Export the small public ratings payload (2025–2026 default).

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
    build_player_champion_records,
    build_player_records,
    build_team_records,
    filter_public_team_rating_maps,
    public_team_affiliation,
    summarize_player_affiliations,
)
from lol_kills.export.player_metadata import build_player_metadata
from lol_kills.etl.competition import TAXONOMY_VERSION, canonicalize_competition_frame, competition_tier
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.dual_elo import build_dual_ratings, lineup_hashes_from_players
from lol_kills.ratings.evidence import attach_player_evidence
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


def source_identity_sha256(game_ids: Iterable[str]) -> str:
    """Bind a pack to its sorted, canonical source game identities."""

    canonical = sorted(
        {
            game_id
            for value in game_ids
            if (game_id := canonical_source_game_key(value))
        }
    )
    raw = ("\n".join(canonical) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _present(cols: Sequence[str], available: Iterable[str]) -> list[str]:
    avail = set(available)
    return [c for c in cols if c in avail]


def _public_player_rating_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Exclude players whose league graph cannot support a public rank."""

    return [
        row
        for row in rows
        if row.get("evidence_disconnected") != 1
        and str(row.get("evidence_state") or "").lower() != "disconnected"
    ]


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


def _canonical_pack_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Recheck competition labels, including older cached canonical columns."""

    return canonicalize_competition_frame(frame)


def _validate_public_record_tiers(records: dict[str, dict[str, Any]], *, label: str) -> None:
    invalid = {"ORACLE_ELIXIR_API", "OE_API", "PUBLIC_DATALISK_API"}
    for identity, record in records.items():
        leagues = {str(value).upper() for value in record.get("leagues", [])}
        if leagues.intersection(invalid):
            raise RuntimeError(f"{label} {identity} exposes a transport label as a league")
        league = record.get("current_league")
        tier = record.get("current_tier")
        expected = competition_tier(league)
        if tier is not None and expected != tier:
            raise RuntimeError(
                f"{label} {identity} has inconsistent league tier: "
                f"league={league} tier={tier} expected={expected}"
            )


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

    # Read full local sources to calculate ratings. Raw rows stay local.
    team_path = warehouse / "oe_team_games.parquet"
    team_source = _canonicalize_game_ids(_canonical_pack_frame(pq.read_table(team_path).to_pandas()))
    team_table = pa.Table.from_pandas(team_source, preserve_index=False)
    team_table = _filter_years(team_table, years, ("year", "oe_year"))
    team_rating_frame = team_table.to_pandas()
    team_maps_for_ratings = build_maps_frame_from_team_games(team_rating_frame)
    del team_rating_frame, team_source, team_table

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
    del player_source, player_table

    # --- maps ---
    maps_path = warehouse / "maps.parquet"
    maps = pq.read_table(maps_path)
    maps = pa.Table.from_pandas(canonicalize_competition_frame(maps.to_pandas()), preserve_index=False)
    maps = _ensure_year_column(maps)
    map_cols = spec.maps_columns(maps.column_names)
    maps = maps.select(map_cols)
    maps = _filter_years(maps, years, ("year", "oe_year"))
    maps_for_records = _canonicalize_game_ids(maps.to_pandas())
    maps = pa.Table.from_pandas(maps_for_records, preserve_index=False)
    source_as_of = pd.to_datetime(maps_for_records["date"], utc=True, errors="coerce").max()
    if pd.isna(source_as_of):
        raise RuntimeError("public pack source has no usable map dates")
    source_game_ids = sorted(set(_normalized_game_uid(maps_for_records).dropna().astype(str)))
    if len(source_game_ids) != len(maps_for_records):
        raise RuntimeError("public pack source is not one row per canonical game identity")
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
    team_records_payload = build_team_records(rating_input)
    player_records_payload = build_player_records(
        player_records_frame,
        team_records=team_records_payload,
    )
    player_champions_payload = build_player_champion_records(player_rating_frame)
    affiliation_audit = summarize_player_affiliations(
        player_records_payload,
        team_records_payload,
    )
    build_dual_ratings(
        rating_input,
        lineup_by_game=lineup_hashes_from_players(player_rating_input),
        output_dir=features_root,
    )
    build_player_ratings(
        player_maps_for_ratings,
        player_rating_input,
        output_dir=features_root,
        player_records=player_records_payload,
    )
    player_snapshot_path = features_root / "player_ratings_snapshot.parquet"
    if player_snapshot_path.exists():
        player_snapshot = pd.read_parquet(player_snapshot_path)
        player_snapshot = attach_player_evidence(
            player_snapshot,
            source_as_of=source_as_of,
        )
        player_snapshot.to_parquet(player_snapshot_path, index=False)
    public_ratings, public_ratings_meta = fit_hierarchical_bt(
        rating_input,
        write=True,
        output_dir=features_root,
    )
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
    # Write only the display JSON used by ratings pages.
    feat_dir = pack_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    _validate_public_record_tiers(team_records_payload, label="team")
    _validate_public_record_tiers(player_records_payload, label="player")

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
        player_records=player_records_payload,
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

    player_champions_dest = feat_dir / "player_champion_records.json"
    player_champions_dest.write_text(
        json.dumps(player_champions_payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
    register(
        {
            "rows": sum(len(rows) for rows in player_champions_payload.values()),
            "cols": 8,
            "bytes": player_champions_dest.stat().st_size,
            "sha256": _sha256(player_champions_dest),
            "columns": [
                "champion",
                "games",
                "wins",
                "losses",
                "wr",
                "kills",
                "deaths",
                "assists",
            ],
        },
        "features/player_champion_records.json",
    )

    for src_name, cols, out_name in (
        ("ratings_snapshot.parquet", spec.RATINGS_SNAPSHOT_COLS, "ratings_snapshot.json"),
        (
            "player_ratings_snapshot.parquet",
            spec.PLAYER_RATINGS_SNAPSHOT_COLS,
            "player_ratings_snapshot.json",
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
        rows = t.to_pylist()
        if out_name == "player_ratings_snapshot.json":
            rows = _public_player_rating_rows(rows)
            for row in rows:
                row["last_team"] = public_team_affiliation(row.get("last_team"))
        dest.write_text(json.dumps(rows), encoding="utf-8")
        register(
            {
                "rows": len(rows),
                "cols": t.num_columns,
                "bytes": dest.stat().st_size,
                "sha256": _sha256(dest),
                "columns": t.column_names,
            },
            f"features/{dest.name}",
        )

    present_paths = {str(item["path"]) for item in files_meta}
    missing_public_files = sorted(
        set(spec.PUBLIC_RATING_REQUIRED_FILES) - present_paths
    )
    if missing_public_files:
        raise RuntimeError(
            "public ratings contract incomplete; missing: "
            + ", ".join(missing_public_files)
        )
    leaked_public_files = sorted(
        present_paths.intersection(spec.WITHHELD_PUBLIC_FILES)
    )
    if leaked_public_files:
        raise RuntimeError(
            "withheld public artifacts present: " + ", ".join(leaked_public_files)
        )

    live_meta = warehouse / "meta.json"
    rating_artifact_paths = {
        item["path"]: {
            key: item[key]
            for key in ("sha256", "bytes", "rows")
            if key in item
        }
        for item in files_meta
        if item["path"]
        in {
            "features/ratings_snapshot.json",
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
            "team_affiliation": "current league membership excludes cup and cross-league event labels; players inherit league and tier from their current team",
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
        "excluded": [
            "raw game rows",
            "research studies",
            "training and prediction artifacts",
        ],
        "ratings": {
            "source_mode": "oe_live" if live_meta.exists() else "warehouse",
            "source_as_of": source_as_of.isoformat().replace("+00:00", "Z"),
            "window_years": list(years),
            "map_rows": int(len(maps_for_records)),
            "source_game_count": len(source_game_ids),
            "source_identity_sha256": source_identity_sha256(source_game_ids),
            "team_rating_rows": int(len(rating_input)),
            "player_rating_rows": int(len(player_rating_frame)),
            "affiliation_audit": affiliation_audit,
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
    ap = argparse.ArgumentParser(description="Export the public LoL ratings payload")
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
