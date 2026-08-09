"""Generate the canonical synthetic B3 coverage authority bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from .b3_coverage import ARTIFACT_ROOT, ARTIFACT_ROLES, _canonical_bytes, build_frozen_payloads


def generate(project_root: Path) -> dict[str, str]:
    root = project_root.resolve(strict=True)
    payloads = build_frozen_payloads(root)
    output_root = root / ARTIFACT_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}
    for role in (*ARTIFACT_ROLES, "authority"):
        name = "coverage-authority.json" if role == "authority" else f"{role.replace('_', '-')}.json"
        path = output_root / name
        path.write_bytes(_canonical_bytes(payloads[role]))
        written[role] = path.relative_to(root).as_posix()
    return written


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    generate(args.project_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
