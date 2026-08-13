"""Refresh research data, rebuild ratings, and export a candidate pack.

This is the automation entrypoint for the public site. It intentionally stops
short of the broader internal model-training command: current ratings and
match rows only need a warehouse refresh, feature-store rebuild, and the
time-safe calibration/draft-score refresh that feeds the public pack.

Examples:

  python3 -m lol_kills.update_public_pack --refresh-oe
  python3 -m lol_kills.update_public_pack --skip-oe
  python3 -m lol_kills.update_public_pack --skip-oe --allow-missing-lp

Public refreshes always pass ``--skip-grid`` to the warehouse build. GRID
ingestion remains available through its private research modules.
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
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--local-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.publish or args.local_only:
        parser.error(
            "legacy public pack publication is disabled; use the private Supabase refresh"
        )

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
    refresh_args.append("--skip-grid")
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
