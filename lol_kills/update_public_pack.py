"""Refresh research data, rebuild governed ratings, export, and publish a pack.

This is the automation entrypoint for the public site. It intentionally stops
short of model promotion. Data refreshes may rebuild rating snapshots whose
contracts are enforced by the exporter, but they never refit or silently
replace draft/calibration artifacts. Those require their separate frozen
tournaments and explicit promotion decisions.

Examples:

  python3 -m lol_kills.update_public_pack --download-oe --download-grid
  python3 -m lol_kills.update_public_pack --skip-oe --download-grid --publish
  python3 -m lol_kills.update_public_pack --skip-oe --allow-missing-lp --download-grid --publish
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]


def _run_module(module: str, args: list[str]) -> None:
    import subprocess

    command = [sys.executable, "-m", module, *args]
    print("[update]", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", default="2025,2026")
    parser.add_argument("--download-oe", action="store_true")
    parser.add_argument("--skip-oe", action="store_true")
    parser.add_argument("--download-grid", action="store_true")
    parser.add_argument("--skip-grid", action="store_true")
    parser.add_argument("--grid-days", type=int, default=3)
    parser.add_argument("--grid-limit", type=int, default=40)
    parser.add_argument("--grid-tournament", default=None)
    parser.add_argument("--grid-env-file", type=Path, default=None)
    parser.add_argument("--grid-required", action="store_true")
    parser.add_argument("--skip-lp", action="store_true")
    parser.add_argument(
        "--allow-missing-lp",
        action="store_true",
        help="Continue with an empty Leaguepedia enrichment when draft caches are absent",
    )
    parser.add_argument("--pack-id", default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "output" / "public_pack")
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Run upload_pack after export; uses Blob when BLOB_READ_WRITE_TOKEN exists",
    )
    parser.add_argument("--local-only", action="store_true")
    args = parser.parse_args(argv)

    from lol_kills.export.upload_pack import validate_pack_id

    requested_pack_id = (
        validate_pack_id(args.pack_id) if args.pack_id is not None else None
    )

    refresh_args = ["--oe-years", *[x.strip() for x in args.years.split(",") if x.strip()]]
    if args.download_oe:
        refresh_args.append("--download-oe")
    if args.skip_oe:
        refresh_args.append("--skip-oe")
    if args.skip_lp:
        refresh_args.append("--skip-lp")
    if args.allow_missing_lp:
        refresh_args.append("--allow-missing-lp")
    if args.download_grid:
        refresh_args.append("--download-grid")
    if args.skip_grid:
        refresh_args.append("--skip-grid")
    refresh_args.extend(["--grid-days", str(args.grid_days), "--grid-limit", str(args.grid_limit)])
    if args.grid_tournament:
        refresh_args.extend(["--grid-tournament", args.grid_tournament])
    if args.grid_env_file:
        refresh_args.extend(["--grid-env-file", str(args.grid_env_file)])
    if args.grid_required:
        refresh_args.append("--grid-required")
    _run_module("lol_kills.refresh_warehouse", refresh_args)

    import pandas as pd

    from lol_kills.etl.paths import PARQUET_DIR
    from lol_kills.export.public_pack import export_public_pack
    from lol_kills.ratings.player_elo import (
        build_maps_frame_from_players,
        build_player_ratings,
    )

    print("[update] rebuilding the explicitly labelled Player Dual Elo benchmark")
    player_rows = pd.read_parquet(PARQUET_DIR / "oe_player_games.parquet")
    build_player_ratings(
        build_maps_frame_from_players(player_rows),
        player_rows,
    )
    print(
        "[update] draft and calibration artifacts remain frozen; "
        "this refresh cannot promote a model"
    )
    print("[update] exporting public pack")
    manifest = export_public_pack(
        years=tuple(int(x.strip()) for x in args.years.split(",") if x.strip()),
        out_root=args.out,
        pack_id=requested_pack_id,
    )
    pack_id = validate_pack_id(manifest["pack_id"])
    print(f"[update] exported {pack_id}")

    if args.publish:
        from lol_kills.audit_public_pack import audit_pack, require_release_gate

        print("[update] running full public-pack release audit")
        report = audit_pack(args.out / pack_id)
        require_release_gate(report)
        counts = report["counts"]
        print(
            "[update] release audit passed "
            f"(launch blocker={counts['launch blocker']}, major={counts['major']})"
        )
        upload_args = ["--pack-root", str(args.out), "--pack-id", pack_id]
        if args.local_only:
            upload_args.append("--local-only")
        _run_module("lol_kills.export.upload_pack", upload_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
