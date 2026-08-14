"""Create a migration-only checkout for one public-release cutover phase.

The repository keeps all phases for review. Supabase CLI applies every pending
migration in its workdir, so production operators must push from this derived
checkout. Later phases are excluded by construction.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path


PHASE_CUTOFFS = {
    "additive": 20260814160000,
    "storage": 20260814170000,
    "strict": 99999999999999,
}


def _migration_version(path: Path) -> int:
    try:
        return int(path.name.split("_", 1)[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid migration filename: {path.name}") from exc


def prepare_phase(repo_root: Path, phase: str, output: Path | None = None) -> Path:
    if phase not in PHASE_CUTOFFS:
        raise ValueError(f"unknown phase: {phase}")
    source_root = repo_root.expanduser().resolve()
    source_supabase = source_root / "supabase"
    source_migrations = source_supabase / "migrations"
    if not source_migrations.is_dir():
        raise ValueError(f"Supabase migrations are missing: {source_migrations}")
    destination = (
        output.expanduser().resolve()
        if output is not None
        else Path(tempfile.mkdtemp(prefix=f"scryglass-supabase-{phase}-"))
    )
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f"phase checkout is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    destination_supabase = destination / "supabase"
    destination_supabase.mkdir()
    config = source_supabase / "config.toml"
    if config.exists():
        shutil.copy2(config, destination_supabase / config.name)
    destination_migrations = destination_supabase / "migrations"
    destination_migrations.mkdir()
    cutoff = PHASE_CUTOFFS[phase]
    kept: list[str] = []
    for migration in sorted(source_migrations.glob("*.sql")):
        version = _migration_version(migration)
        if version < cutoff:
            shutil.copy2(migration, destination_migrations / migration.name)
            kept.append(migration.name)
    if not kept:
        raise ValueError("no migrations selected for the requested phase")
    later = [
        path.name
        for path in source_migrations.glob("*.sql")
        if _migration_version(path) >= cutoff and path.name not in kept
    ]
    marker = destination / "PHASE.txt"
    marker.write_text(
        f"phase={phase}\n"
        f"source={source_root}\n"
        f"migrations={len(kept)}\n"
        f"excluded_later_migrations={len(later)}\n",
        encoding="utf-8",
    )
    print(destination)
    return destination


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=sorted(PHASE_CUTOFFS), required=True)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    path = prepare_phase(args.repo_root, args.phase, args.output)
    if os.environ.get("SCRYGLASS_PHASE_VERBOSE") == "1":
        print(f"prepared Supabase {args.phase} workdir: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
