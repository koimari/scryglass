"""Generate the frozen synthetic-only outer-calibration evidence bundle."""

from __future__ import annotations

import json
from pathlib import Path

from .outer_calibration import write_outer_calibration_artifacts


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    hashes = write_outer_calibration_artifacts(root)
    print(json.dumps(hashes, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

