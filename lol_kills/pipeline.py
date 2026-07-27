#!/usr/bin/env python3
"""
Research pipeline orchestrator.

  # Legacy training is quarantined; use the governed model tournaments.
  python -m lol_kills.pipeline report
  python -m lol_kills.pipeline refresh
"""

from __future__ import annotations

import argparse
import json


def cmd_refresh(args: argparse.Namespace) -> None:
    from lol_kills.refresh_warehouse import main as refresh_main
    import sys

    forwarded = ["lol_kills.refresh_warehouse"]
    if args.oe_years:
        forwarded += ["--oe-years", *args.oe_years]
    if args.download_oe:
        forwarded.append("--download-oe")
    if args.skip_oe:
        forwarded.append("--skip-oe")
    if args.download_grid:
        forwarded.append("--download-grid")
    if args.grid_required:
        forwarded.append("--grid-required")
    if args.grid_env_file:
        forwarded += ["--grid-env-file", args.grid_env_file]
    sys.argv = forwarded
    refresh_main()


def cmd_train(args: argparse.Namespace) -> None:
    raise RuntimeError(
        "The legacy aggregate trainer is quarantined because its historical "
        "feature and calibration paths do not satisfy the current leakage "
        "contract. Run the governed draft, team, and player model tournaments "
        "instead; no legacy artifact was written."
    )


def cmd_report(_args: argparse.Namespace) -> None:
    from pathlib import Path

    from lol_kills.etl.paths import MODELS_DIR

    eval_path = MODELS_DIR / "eval_report.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        print("[pipeline] gates", json.dumps(ev.get("gates"), indent=2, default=str)[:2000])
    else:
        print("[pipeline] no governed evaluation report is available")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_r = sub.add_parser("refresh", help="Refresh OE+GRID+LP warehouse")
    p_r.add_argument("--oe-years", nargs="*", default=None)
    p_r.add_argument("--download-oe", action="store_true")
    p_r.add_argument("--skip-oe", action="store_true")
    p_r.add_argument("--download-grid", action="store_true")
    p_r.add_argument("--grid-required", action="store_true")
    p_r.add_argument("--grid-env-file", default=None)
    p_r.set_defaults(func=cmd_refresh)

    p_t = sub.add_parser("train", help="Quarantined legacy aggregate trainer")
    p_t.add_argument("--no-archive", action="store_true")
    p_t.set_defaults(func=cmd_train)

    p_rep = sub.add_parser("report", help="Read the local evaluation gate report")
    p_rep.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
