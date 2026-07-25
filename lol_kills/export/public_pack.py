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

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from lol_kills.export import pack_spec as spec

ROOT = Path(__file__).resolve().parents[2]
WAREHOUSE = ROOT / "data" / "lol" / "warehouse" / "parquet"
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
    mask = None
    for col in year_cols:
        if col not in table.column_names:
            continue
        arr = table[col]
        # year may be int or string
        try:
            as_int = pc.cast(arr, pa.int64(), safe=False)
        except Exception:
            as_int = pc.cast(pc.utf8_to_int(pc.cast(arr, pa.string())), pa.int64(), safe=False)
        m = pc.is_in(as_int, value_set=pa.array(years_list, type=pa.int64()))
        mask = m if mask is None else pc.or_(mask, m)
    if mask is None:
        return table
    return table.filter(mask)


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


def export_public_pack(
    *,
    years: Sequence[int] | None = None,
    out_root: Path | None = None,
    pack_id: str | None = None,
) -> dict[str, Any]:
    years = tuple(years or spec.DEFAULT_YEARS)
    stamp = datetime.now(timezone.utc).strftime("%Y.%m.%d")
    pack_id = pack_id or f"v{stamp}"
    out_root = Path(out_root or DEFAULT_OUT)
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
    team_path = WAREHOUSE / "oe_team_games.parquet"
    team = pq.read_table(team_path)
    team_cols = _present(spec.TEAM_COLS, team.column_names)
    team = team.select(team_cols)
    team = _filter_years(team, years, ("year", "oe_year"))
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

    # --- player games ---
    player_path = WAREHOUSE / "oe_player_games.parquet"
    player = pq.read_table(player_path)
    player_cols = _present(spec.PLAYER_COLS, player.column_names)
    player = player.select(player_cols)
    player = _filter_years(player, years, ("year", "oe_year"))
    for y in years:
        if "year" in player.column_names:
            part = player.filter(pc.equal(pc.cast(player["year"], pa.int64(), safe=False), y))
        else:
            part = player.filter(pc.equal(pc.cast(player["oe_year"], pa.int64(), safe=False), y))
        dest = pack_dir / "player_games" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"player_games/year={y}/part.parquet")

    # --- maps ---
    maps_path = WAREHOUSE / "maps.parquet"
    maps = pq.read_table(maps_path)
    map_cols = spec.maps_columns(maps.column_names)
    maps = maps.select(map_cols)
    maps = _filter_years(maps, years, ("year", "oe_year"))
    for y in years:
        if "year" in maps.column_names:
            part = maps.filter(pc.equal(pc.cast(maps["year"], pa.int64(), safe=False), y))
        else:
            part = maps.filter(pc.equal(pc.cast(maps["oe_year"], pa.int64(), safe=False), y))
        dest = pack_dir / "maps" / f"year={y}" / "part.parquet"
        if part.num_rows == 0:
            continue
        register(_write_parquet(part, dest), f"maps/year={y}/part.parquet")

    # --- features snapshots (full — small) ---
    feat_dir = pack_dir / "features"
    feat_dir.mkdir(parents=True, exist_ok=True)
    for src_name, cols, out_name in (
        ("ratings_snapshot.parquet", spec.RATINGS_SNAPSHOT_COLS, "ratings_snapshot.parquet"),
        (
            "player_ratings_snapshot.parquet",
            spec.PLAYER_RATINGS_SNAPSHOT_COLS,
            "player_ratings_snapshot.parquet",
        ),
    ):
        src = FEATURES / src_name
        if not src.exists():
            continue
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
    maps_all = pq.read_table(maps_path)
    maps_all = maps_all.select(spec.maps_columns(maps_all.column_names))
    maps_all = _filter_years(maps_all, years, ("year", "oe_year"))

    for src_name, cols, out_name in (
        ("ratings.parquet", spec.RATINGS_HISTORY_COLS, "ratings_history.parquet"),
        (
            "player_ratings.parquet",
            spec.PLAYER_RATINGS_HISTORY_COLS,
            "player_ratings_history.parquet",
        ),
    ):
        src = FEATURES / src_name
        if not src.exists():
            continue
        t = pq.read_table(src)
        t = t.select(_present(cols, t.column_names))
        t = _join_history_years(t, maps_all, years)
        dest = feat_dir / out_name
        register(_write_parquet(t, dest), f"features/{out_name}")

    for meta_name in ("ratings_meta.json", "player_ratings_meta.json"):
        src = FEATURES / meta_name
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
        src = MODELS / name
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

    tier_dir = MODELS / "tierlists_csv"
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
    pdf_root = ROOT / "output" / "pdf"
    grubs_dir = pack_dir / "studies" / "grubs"
    grubs_dir.mkdir(parents=True, exist_ok=True)
    for name in spec.GRUBS_MODEL_FILES:
        src = MODELS / name
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

    # --- meta ---
    meta_dir = pack_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    if TEAMS_JSON.exists():
        dest = meta_dir / "teams.json"
        shutil.copy2(TEAMS_JSON, dest)
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

    oe_meta = WAREHOUSE / "oe_meta.json"
    refresh_meta = WAREHOUSE / "refresh_meta.json"
    ingest = {}
    for p in (oe_meta, refresh_meta):
        if p.exists():
            try:
                ingest[p.stem] = json.loads(p.read_text())
            except json.JSONDecodeError:
                ingest[p.stem] = {"raw": p.read_text()[:500]}

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
        "attribution": spec.ATTRIBUTION,
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
    args = ap.parse_args(argv)
    years = tuple(int(x.strip()) for x in args.years.split(",") if x.strip())
    man = export_public_pack(years=years, out_root=args.out, pack_id=args.pack_id)
    mb = man["total_bytes"] / (1024 * 1024)
    print(f"Wrote pack {man['pack_id']} → {args.out / man['pack_id']}")
    print(f"Files: {man['total_files']}  Size: {mb:.1f} MB  schema={man['schema_version']}")
    for f in man["files"]:
        print(f"  {f['path']}: {f['bytes']/1024:.0f} KB  rows={f.get('rows')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
