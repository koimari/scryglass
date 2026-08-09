"""Generate deterministic provisional L5 development artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .model import build_development_candidate, canonical_json


def generate(config_path: Path, fixtures_path: Path, output_path: Path) -> dict:
    with config_path.open("rb") as handle:
        config = json.load(handle)
    with fixtures_path.open("rb") as handle:
        fixtures = json.load(handle)
    candidate = build_development_candidate(config, fixtures)
    output_path.write_bytes(canonical_json(candidate) + b"\n")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.config, args.fixtures, args.output)


if __name__ == "__main__":
    main()
