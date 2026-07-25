#!/usr/bin/env python3
"""List teams / H2H coverage in the kill models dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DS = ROOT / "data" / "lol" / "kill_models.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DS)
    ap.add_argument("--league", choices=["LCK", "LEC", "LCS"])
    args = ap.parse_args()
    ds = json.loads(args.data.read_text())
    print("meta:", json.dumps(ds["meta"], indent=2))
    for lg, meta in ds["leagues"].items():
        if args.league and lg != args.league:
            continue
        print(f"\n{lg}  games={meta['n_games']}  league_kpg={meta['kpg_per_team']}")
        for t in meta["teams"]:
            tm = ds["teams"][t]
            print(
                f"  {t:28} n={tm['n']:3d}  for={tm['kills_for_mean']:5.1f}  "
                f"against={tm['kills_against_mean']:5.1f}  total={tm['total_kills_mean']:5.1f}"
            )


if __name__ == "__main__":
    main()
