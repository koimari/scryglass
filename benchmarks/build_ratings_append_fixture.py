#!/usr/bin/env python3
"""Build an isolated source-bound fixture for ratings timing runs.

The helper reads one worker source snapshot and one accepted game census. It
copies the source files into separate base and current phase roots. Each phase
uses its own census, so the same parquet bytes can serve both runs without
changing the worker source.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.v2.tierlists.accepted_census import (
    identity_sha256,
    load_census,
    write_census,
)


SOURCE_PARQUETS = (
    Path("data/lol/warehouse/parquet/oe_live/maps.parquet"),
    Path("data/lol/warehouse/parquet/oe_live/oe_team_games.parquet"),
    Path("data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"),
)
SOURCE_META = Path("data/lol/warehouse/parquet/oe_live/meta.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_files(source_root: Path) -> tuple[Path, ...]:
    paths = tuple(source_root / relative for relative in SOURCE_PARQUETS)
    missing = [str(path) for path in paths if not path.is_file() or path.is_symlink()]
    if missing:
        raise FileNotFoundError(f"worker source parquet is missing: {missing}")
    meta = source_root / SOURCE_META
    return (*paths, meta) if meta.is_file() and not meta.is_symlink() else paths


def _copy_source(source_root: Path, phase_root: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for source_path in _source_files(source_root):
        relative = source_path.relative_to(source_root)
        destination = phase_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        files[str(relative)] = {
            "bytes": int(source_path.stat().st_size),
            "sha256": _sha256(source_path),
            "copied_sha256": _sha256(destination),
        }
    return files


def _ordered_accepted_maps(
    source_root: Path,
    accepted_ids: tuple[str, ...],
) -> pd.DataFrame:
    maps_path = source_root / SOURCE_PARQUETS[0]
    maps = pd.read_parquet(maps_path, columns=["game_uid", "date"])
    if maps.empty:
        raise ValueError("worker map source is empty")
    maps["_fixture_game_id"] = maps["game_uid"].map(canonical_source_game_key)
    accepted_set = set(accepted_ids)
    accepted = maps[maps["_fixture_game_id"].isin(accepted_set)].copy()
    if len(accepted) != len(accepted_ids):
        counts = accepted["_fixture_game_id"].value_counts()
        duplicate_ids = sorted(str(value) for value in counts[counts.gt(1)].index)
        missing_ids = sorted(accepted_set.difference(set(accepted["_fixture_game_id"])))
        raise ValueError(
            "accepted census does not bind to one source map per game; "
            f"missing_games={len(missing_ids)} duplicate_games={len(duplicate_ids)} "
            f"duplicate_sample={duplicate_ids[:5]} missing_sample={missing_ids[:5]}"
        )
    dates = pd.to_datetime(accepted["date"], errors="coerce", utc=True)
    if dates.isna().any():
        raise ValueError("accepted source maps contain unusable dates")
    accepted["_fixture_date"] = dates
    return accepted.sort_values(
        ["_fixture_date", "_fixture_game_id"],
        kind="mergesort",
    ).reset_index(drop=True)


def _phase_payload(
    *,
    name: str,
    root: Path,
    census_path: Path,
    game_ids: list[str],
) -> dict[str, Any]:
    return {
        "name": name,
        "root": str(root),
        "census": str(census_path),
        "game_count": len(game_ids),
        "source_identity_sha256": identity_sha256(game_ids),
        "game_ids": game_ids,
    }


def build_fixture(
    source_root: Path | str,
    accepted_census: Path | str,
    output_root: Path | str,
    *,
    suffix_count: int = 6,
) -> dict[str, Any]:
    """Create base and current phase roots from a worker source snapshot."""

    source_root = Path(source_root).expanduser().resolve()
    accepted_census = Path(accepted_census).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    if suffix_count < 1:
        raise ValueError("suffix_count must be positive")
    if output_root == source_root or source_root in output_root.parents:
        raise ValueError("output_root must be outside source_root")
    if output_root.exists():
        if not output_root.is_dir():
            raise FileExistsError(f"fixture output is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise FileExistsError(f"fixture output is not empty: {output_root}")
    source_files = _source_files(source_root)
    census = load_census(accepted_census)
    accepted_ids = tuple(census["game_ids"])
    ordered = _ordered_accepted_maps(source_root, accepted_ids)
    if suffix_count >= len(ordered):
        raise ValueError("suffix_count must leave at least one base game")

    base = ordered.iloc[:-suffix_count]
    append = ordered.iloc[-suffix_count:]
    if base["_fixture_date"].max() >= append["_fixture_date"].min():
        raise ValueError("suffix is not strictly later than the base census")
    base_ids = [str(value) for value in base["_fixture_game_id"]]
    append_ids = [str(value) for value in append["_fixture_game_id"]]
    current_ids = list(accepted_ids)

    output_root.mkdir(parents=True, exist_ok=True)
    base_root = output_root / "base"
    current_root = output_root / "current"
    base_files = _copy_source(source_root, base_root)
    current_files = _copy_source(source_root, current_root)
    census_root = output_root / "censuses"
    base_census = census_root / "base.json"
    current_census = census_root / "current.json"
    append_census = census_root / "append.json"
    write_census(base_census, base_ids)
    write_census(current_census, current_ids)
    write_census(append_census, append_ids)

    first_append_date = pd.Timestamp(append["_fixture_date"].iloc[0]).isoformat()
    last_append_date = pd.Timestamp(append["_fixture_date"].iloc[-1]).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": "scryglass:ratings-production-fixture:v1",
        "source": {
            "root": str(source_root),
            "accepted_census": str(accepted_census),
            "accepted_game_count": len(accepted_ids),
            "accepted_source_identity_sha256": identity_sha256(accepted_ids),
            "files": {
                str(path.relative_to(source_root)): {
                    "bytes": int(path.stat().st_size),
                    "sha256": _sha256(path),
                }
                for path in source_files
            },
        },
        "suffix": {
            "count": len(append_ids),
            "first_date": first_append_date,
            "last_date": last_append_date,
            "base_last_date": pd.Timestamp(base["_fixture_date"].iloc[-1]).isoformat(),
            "append_source_identity_sha256": identity_sha256(append_ids),
            "append_game_ids": append_ids,
        },
        "phases": {
            "base": _phase_payload(
                name="base",
                root=base_root,
                census_path=base_census,
                game_ids=base_ids,
            ),
            "current": _phase_payload(
                name="current",
                root=current_root,
                census_path=current_census,
                game_ids=current_ids,
            ),
            "append": _phase_payload(
                name="append",
                root=current_root,
                census_path=append_census,
                game_ids=append_ids,
            ),
        },
        "copied_files": {
            "base": base_files,
            "current": current_files,
        },
        "refresh_commands": {
            "base": (
                "python3 -m lol_kills.v2.tierlists.rating_refresh "
                f"--root {base_root} --accepted-census {base_census}"
            ),
            "current": (
                "python3 -m lol_kills.v2.tierlists.rating_refresh "
                f"--root {current_root} --accepted-census {current_census}"
            ),
            "append": (
                "python3 -m lol_kills.v2.tierlists.rating_refresh "
                f"--root {current_root} --accepted-census {append_census}"
            ),
        },
    }
    manifest_path = output_root / "fixture.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--accepted-census", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--suffix-count", type=int, default=6)
    args = parser.parse_args()
    manifest = build_fixture(
        args.source_root,
        args.accepted_census,
        args.output_root,
        suffix_count=args.suffix_count,
    )
    print(
        json.dumps(
            {
                "fixture": str(args.output_root.expanduser().resolve() / "fixture.json"),
                "accepted_game_count": manifest["source"]["accepted_game_count"],
                "accepted_source_identity_sha256": manifest["source"][
                    "accepted_source_identity_sha256"
                ],
                "suffix": manifest["suffix"],
                "phases": {
                    name: {
                        "root": phase["root"],
                        "census": phase["census"],
                        "game_count": phase["game_count"],
                        "source_identity_sha256": phase["source_identity_sha256"],
                    }
                    for name, phase in manifest["phases"].items()
                },
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
