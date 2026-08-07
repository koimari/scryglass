#!/usr/bin/env python3
"""Run the Scryglass private GRID capability catalog builder."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--probe-series-id")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = args.repo.expanduser().resolve()
    output = args.out or (
        Path.home()
        / ".codex"
        / "skills"
        / "query-grid-research"
        / "assets"
        / "grid-capability-catalog.v1.json"
    )
    command = [
        sys.executable,
        "-m",
        "lol_kills.grid_capability_catalog",
        "--out",
        str(output),
        "--local-events-dir",
        str(repo / "data/lol/warehouse/raw_grid"),
    ]
    if args.probe_series_id:
        command.extend(["--probe-series-id", args.probe_series_id])
    environment = dict(os.environ)
    return subprocess.run(command, cwd=repo, env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
