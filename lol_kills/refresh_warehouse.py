#!/usr/bin/env python3
"""Idempotent warehouse refresh: OE (optional download) + Leaguepedia → parquet maps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from lol_kills.etl.join import build_map_warehouse
from lol_kills.etl.leaguepedia_ingest import ingest_leaguepedia
from lol_kills.etl.oe_ingest import ingest_oe
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
    args = ap.parse_args()

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    oe_team = oe_player = None
    if not args.skip_oe:
        oe_team, oe_player = ingest_oe(years=args.oe_years, download=args.download_oe)

    lp_team = lp_player = None
    if not args.skip_lp:
        lp_team, lp_player = ingest_leaguepedia()

    maps = build_map_warehouse(lp_team=lp_team, oe_team=oe_team, lp_players=lp_player)

    meta = {
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "n_maps": int(len(maps)),
        "oe_rows": int(len(oe_team)) if oe_team is not None else 0,
        "lp_side_rows": int(len(lp_team)) if lp_team is not None else 0,
        "oe_matched": int(maps["oe_matched"].sum()) if "oe_matched" in maps.columns else 0,
    }
    (PARQUET_DIR / "refresh_meta.json").write_text(json.dumps(meta, indent=2))
    print("[refresh]", json.dumps(meta))


if __name__ == "__main__":
    main()
