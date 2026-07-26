#!/usr/bin/env python3
"""Idempotent warehouse refresh: OE (optional download) + Leaguepedia → parquet maps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lol_kills.etl.grid_ingest import ingest_grid, merge_source_frames
from lol_kills.etl.competition import canonicalize_competition_frame
from lol_kills.etl.join import build_map_warehouse
from lol_kills.etl.leaguepedia_ingest import ingest_leaguepedia
from lol_kills.etl.oe_ingest import ingest_oe, load_cached_oe
from lol_kills.etl.paths import PARQUET_DIR, WAREHOUSE_DIR


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--download-oe",
        action="store_true",
        help="Attempt Google Drive download of OE annual CSVs (often rate-limited)",
    )
    ap.add_argument(
        "--oe-years",
        nargs="*",
        default=None,
        help="OE years to load/download (default: all local CSVs; download defaults to last 4)",
    )
    ap.add_argument("--skip-oe", action="store_true", help="Skip OE ingest entirely")
    ap.add_argument("--skip-lp", action="store_true", help="Skip Leaguepedia ingest")
    ap.add_argument(
        "--allow-missing-lp",
        action="store_true",
        help="Continue with an empty LP enrichment when the draft cache is absent",
    )
    ap.add_argument(
        "--download-grid",
        action="store_true",
        help="Download recent completed professional series from GRID",
    )
    ap.add_argument("--grid-days", type=int, default=3, help="GRID lookback window")
    ap.add_argument("--grid-limit", type=int, default=40, help="Maximum recent GRID series to inspect")
    ap.add_argument("--grid-env-file", type=str, default=None, help="Optional local .env containing GRID_API_KEY")
    ap.add_argument(
        "--grid-required",
        action="store_true",
        help="Fail the refresh if GRID was requested but no completed games are available",
    )
    ap.add_argument("--skip-grid", action="store_true", help="Skip local GRID rows entirely")
    args = ap.parse_args()

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    oe_team = oe_player = None
    if not args.skip_oe:
        oe_team, oe_player = ingest_oe(years=args.oe_years, download=args.download_oe)
    else:
        # Fast GRID workers restore this normalized cache from the durable
        # snapshot, then layer newly completed GRID rows on top of it.
        oe_team, oe_player = load_cached_oe(args.oe_years)
        if oe_team.empty and oe_player.empty:
            print("[oe] --skip-oe requested but no cached normalized OE rows exist")

    grid_team = grid_player = pd.DataFrame()
    if not args.skip_grid:
        grid_team, grid_player = ingest_grid(
            download=args.download_grid,
            days=args.grid_days,
            limit=args.grid_limit,
            env_file=Path(args.grid_env_file) if args.grid_env_file else None,
            required=args.grid_required,
        )

    # OE is the reconciled primary source. GRID fills the freshness gap until
    # the next OE export includes the same game; duplicate game/side rows are
    # therefore resolved in OE's favor.
    combined_team = merge_source_frames(oe_team, grid_team, ["gameid", "side"])
    combined_player = merge_source_frames(
        oe_player,
        grid_player,
        ["gameid", "side", "position"],
    )

    # Canonical identity is applied after source precedence is resolved.  This
    # keeps OE/GRID provenance intact while ensuring one team key across LCS,
    # MSI, EWC, and legacy LTA labels.
    combined_team = canonicalize_competition_frame(combined_team)
    combined_player = canonicalize_competition_frame(combined_player)

    # Keep the join module's established input paths while making the source
    # precedence explicit in the rows themselves.
    combined_team.to_parquet(PARQUET_DIR / "oe_team_games.parquet", index=False)
    combined_player.to_parquet(PARQUET_DIR / "oe_player_games.parquet", index=False)

    lp_team = lp_player = None
    if not args.skip_lp:
        try:
            lp_team, lp_player = ingest_leaguepedia()
        except FileNotFoundError as exc:
            if not args.allow_missing_lp:
                raise
            print(f"[lp] optional enrichment unavailable; continuing: {exc}")
            lp_team, lp_player = pd.DataFrame(), pd.DataFrame()

    maps = build_map_warehouse(
        lp_team=lp_team,
        oe_team=combined_team,
        lp_players=lp_player,
    )

    source_counts = {}
    if "source" in combined_team.columns:
        source_counts = {
            str(source): int(count)
            for source, count in combined_team["source"].value_counts(dropna=False).items()
        }

    meta = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "n_maps": int(len(maps)),
        "oe_rows": int(len(oe_team)) if oe_team is not None else 0,
        "grid_rows": int(len(grid_team)),
        "combined_team_rows": int(len(combined_team)),
        "combined_player_rows": int(len(combined_player)),
        "source_counts": source_counts,
        "lp_side_rows": int(len(lp_team)) if lp_team is not None else 0,
        "oe_matched": int(maps["oe_matched"].sum()) if "oe_matched" in maps.columns else 0,
    }
    (PARQUET_DIR / "refresh_meta.json").write_text(json.dumps(meta, indent=2))
    print("[refresh]", json.dumps(meta))


if __name__ == "__main__":
    main()
