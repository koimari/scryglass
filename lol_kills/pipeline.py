#!/usr/bin/env python3
"""
Research pipeline orchestrator.

  python -m lol_kills.pipeline train
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
    from lol_kills.features.build import build_feature_store
    from lol_kills.draft_score import fit_draft_score_scaler
    from lol_kills.ml.train import train_all
    from lol_kills.ratings.calibrate_elo_wr import apply_calibration_to_features, fit_elo_wr_calibration

    print("[pipeline] features…")
    build_feature_store()
    print("[pipeline] Elo→WR calibration (time-safe, L2)…")
    fit_elo_wr_calibration()
    apply_calibration_to_features()
    print("[pipeline] draft score fit…")
    fit_draft_score_scaler()
    print("[pipeline] train + gates…")
    report = train_all(do_archive=not args.no_archive)
    print(json.dumps({k: (v.get("status") if isinstance(v, dict) and "status" in v else "ok") for k, v in report.items()}, indent=2))


def cmd_report(_args: argparse.Namespace) -> None:
    from pathlib import Path

    from lol_kills.econ import betting_report
    from lol_kills.etl.paths import MODELS_DIR

    br = betting_report()
    eval_path = MODELS_DIR / "eval_report.json"
    if eval_path.exists():
        ev = json.loads(eval_path.read_text())
        print("[pipeline] gates", json.dumps(ev.get("gates"), indent=2, default=str)[:2000])
    print("[pipeline] betting", json.dumps(br, indent=2))


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

    p_t = sub.add_parser("train", help="Features + ratings + train + gates")
    p_t.add_argument("--no-archive", action="store_true")
    p_t.set_defaults(func=cmd_train)

    p_rep = sub.add_parser("report", help="Eval gates + betting/CLV report")
    p_rep.set_defaults(func=cmd_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
