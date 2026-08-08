"""Refresh research data, rebuild ratings, export, and publish a public pack.

This is the automation entrypoint for the public site. It intentionally stops
short of the broader internal model-training command: current ratings and
match rows only need a warehouse refresh, feature-store rebuild, and the
time-safe calibration/draft-score refresh that feeds the public pack.

Examples:

  python3 -m lol_kills.update_public_pack --refresh-oe --download-grid
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
    oe_download = parser.add_mutually_exclusive_group()
    oe_download.add_argument("--download-oe", action="store_true")
    oe_download.add_argument("--refresh-oe", action="store_true")
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

    refresh_args = ["--oe-years", *[x.strip() for x in args.years.split(",") if x.strip()]]
    if args.download_oe:
        refresh_args.append("--download-oe")
    if args.refresh_oe:
        refresh_args.append("--refresh-oe")
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

    from lol_kills.draft_score import fit_draft_score_scaler
    from lol_kills.draft_recommendation import write_recommendation_model
    from lol_kills.export.public_pack import export_public_pack
    from lol_kills.features.build import build_feature_store
    from lol_kills.ratings.calibrate_elo_wr import (
        apply_calibration_to_features,
        fit_elo_wr_calibration,
    )

    print("[update] rebuilding feature store and player-aggregated ratings")
    build_feature_store()
    print("[update] recalibrating Elo win probability")
    fit_elo_wr_calibration()
    apply_calibration_to_features()
    print("[update] refreshing draft-score scaler")
    fit_draft_score_scaler()
    print("[update] fitting draft synergy, counter, and player-controlled interactions")
    write_recommendation_model()
    print("[update] exporting public pack")
    manifest = export_public_pack(
        years=tuple(int(x.strip()) for x in args.years.split(",") if x.strip()),
        out_root=args.out,
        pack_id=args.pack_id,
    )
    pack_id = str(manifest["pack_id"])
    print(f"[update] exported {pack_id}")

    if args.publish:
        upload_args = ["--pack-root", str(args.out), "--pack-id", pack_id]
        if args.local_only:
            upload_args.append("--local-only")
        _run_module("lol_kills.export.upload_pack", upload_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
