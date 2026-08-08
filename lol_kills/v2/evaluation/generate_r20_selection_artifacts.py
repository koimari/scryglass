"""Generate canonical synthetic-development R-20 selection artifacts."""

from __future__ import annotations

from pathlib import Path

from .r20_selection import load_r20_selection_authority, write_selection_artifacts


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    hashes = write_selection_artifacts(root)
    load_r20_selection_authority(root)
    for locator, digest in sorted(hashes.items()):
        print(f"{digest}  {locator}")


if __name__ == "__main__":
    main()
