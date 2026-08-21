"""Capture Leaguepedia Cargo inputs for an OE series crosswalk.

This command performs a local research capture.  It does not inspect browser
state and it does not send credentials.  The output root contains exact raw
responses, deterministic assembled arrays, cache metadata, and a
self-hashed capture manifest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lol_kills.research.leaguepedia_cargo_capture import (
    CargoCaptureError,
    MAX_CARGO_LIMIT,
    capture_leaguepedia_sources,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True, help="inclusive ISO date, YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="inclusive ISO date, YYYY-MM-DD")
    parser.add_argument("--root", type=Path, required=True, help="stable local capture/cache root")
    parser.add_argument("--window-days", type=int, default=1, help="half-open ScoreboardGames/MatchSchedule window size, 1 to 31")
    parser.add_argument("--limit", type=int, default=MAX_CARGO_LIMIT, help="Cargo row limit per request, at most 500")
    parser.add_argument("--no-resume", action="store_true", help="fetch requests again instead of using verified cache entries")
    args = parser.parse_args(argv)
    try:
        manifest = capture_leaguepedia_sources(
            start_date=args.start_date,
            end_date=args.end_date,
            root=args.root,
            window_days=args.window_days,
            limit=args.limit,
            resume=not args.no_resume,
        )
    except (CargoCaptureError, OSError) as error:
        parser.error(str(error))
        return 2
    print(
        json.dumps(
            {
                "manifest_path": manifest.get("manifest_path", str(args.root / "capture-manifest.json")),
                "manifest_sha256": manifest["manifest_sha256"],
                "coverage": manifest["coverage"],
                "status": manifest["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
