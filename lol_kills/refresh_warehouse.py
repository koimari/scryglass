#!/usr/bin/env python3
"""Idempotent warehouse refresh: OE (optional download) + Leaguepedia → parquet maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from lol_kills.etl.grid_ingest import ingest_grid, merge_source_frames
from lol_kills.etl.competition import canonicalize_competition_frame
from lol_kills.etl.join import build_map_warehouse
from lol_kills.etl.leaguepedia_ingest import ingest_leaguepedia
from lol_kills.etl.oe_ingest import (
    OeDownloadError,
    ingest_oe,
    load_cached_oe,
    load_refresh_receipt,
)
from lol_kills.etl.paths import (
    LP_GAMES,
    LP_PLAYERS,
    OE_RECEIPT_DIR,
    PARQUET_DIR,
    ROOT,
    WAREHOUSE_DIR,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _locator(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _file_record(path: Path, *, rows: int | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "locator": _locator(path),
        "bytes": int(path.stat().st_size),
        "raw_sha256": _sha256(path),
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _matching_oe_receipts(source_files: list[object]) -> list[dict[str, object]]:
    expected = {
        (str(item.get("year")), str(item.get("raw_sha256")))
        for item in source_files
        if isinstance(item, dict)
    }
    matches: list[dict[str, object]] = []
    for path in sorted(OE_RECEIPT_DIR.glob("*.json")):
        try:
            receipt = load_refresh_receipt(path)
        except OeDownloadError:
            continue
        candidate = receipt.get("candidate") or {}
        key = (str(receipt.get("year")), str(candidate.get("raw_sha256")))
        if key not in expected:
            continue
        matches.append(
            {
                **_file_record(path),
                "year": receipt["year"],
                "retrieved_at_utc": receipt["retrieved_at_utc"],
                "source_raw_sha256": candidate["raw_sha256"],
                "receipt_canonical_sha256": receipt[
                    "receipt_canonical_sha256"
                ],
            }
        )
    return matches


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    oe_download = ap.add_mutually_exclusive_group()
    oe_download.add_argument(
        "--download-oe",
        action="store_true",
        help="Download missing OE annual CSVs; keep validated existing files unchanged",
    )
    oe_download.add_argument(
        "--refresh-oe",
        action="store_true",
        help=(
            "Strictly refresh OE annual CSVs via staged validation, prior-byte archive, "
            "atomic replacement, and a hash-bound receipt"
        ),
    )
    ap.add_argument(
        "--oe-years",
        nargs="*",
        default=None,
        help="OE years to load/download (default: all local CSVs; download defaults to last 4)",
    )
    ap.add_argument("--skip-oe", action="store_true", help="Skip OE ingest entirely")
    ap.add_argument("--skip-lp", action="store_true", help="Skip Leaguepedia ingest")
    ap.add_argument(
        "--allow-missing-lp",
        action="store_true",
        help="Continue with an empty LP enrichment when the draft cache is absent",
    )
    ap.add_argument(
        "--download-grid",
        action="store_true",
        help="Download recent completed professional series from GRID",
    )
    ap.add_argument("--grid-days", type=int, default=3, help="GRID lookback window")
    ap.add_argument("--grid-limit", type=int, default=40, help="Maximum recent GRID series to inspect")
    ap.add_argument(
        "--grid-tournament",
        type=str,
        default=None,
        help="Optional case-insensitive tournament substring filter for manual GRID catch-up",
    )
    ap.add_argument("--grid-env-file", type=str, default=None, help="Optional local .env containing GRID_API_KEY")
    ap.add_argument(
        "--grid-required",
        action="store_true",
        help="Fail the refresh if GRID was requested but no completed games are available",
    )
    ap.add_argument("--skip-grid", action="store_true", help="Skip local GRID rows entirely")
    args = ap.parse_args()

    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    oe_team = oe_player = None
    if not args.skip_oe:
        oe_team, oe_player = ingest_oe(
            years=args.oe_years,
            download=args.download_oe or args.refresh_oe,
            force_download=args.refresh_oe,
        )
    else:
        # Fast GRID workers restore this normalized cache from the durable
        # snapshot, then layer newly completed GRID rows on top of it.
        oe_team, oe_player = load_cached_oe(args.oe_years)
        if oe_team.empty and oe_player.empty:
            print("[oe] --skip-oe requested but no cached normalized OE rows exist")

    grid_team = grid_player = pd.DataFrame()
    if not args.skip_grid:
        grid_team, grid_player = ingest_grid(
            download=args.download_grid,
            days=args.grid_days,
            limit=args.grid_limit,
            tournament=args.grid_tournament,
            env_file=Path(args.grid_env_file) if args.grid_env_file else None,
            required=args.grid_required,
        )

    # OE is the reconciled primary source. GRID fills the freshness gap until
    # the next OE export includes the same game; duplicate game/side rows are
    # therefore resolved in OE's favor.
    combined_team = merge_source_frames(oe_team, grid_team, ["gameid", "side"])
    combined_player = merge_source_frames(
        oe_player,
        grid_player,
        ["gameid", "side", "position"],
    )

    # Canonical identity is applied after source precedence is resolved.  This
    # keeps OE/GRID provenance intact while ensuring one team key across LCS,
    # MSI, EWC, and legacy LTA labels.
    combined_team = canonicalize_competition_frame(combined_team)
    combined_player = canonicalize_competition_frame(combined_player)

    # Keep the join module's established input paths while making the source
    # precedence explicit in the rows themselves.
    combined_team.to_parquet(PARQUET_DIR / "oe_team_games.parquet", index=False)
    combined_player.to_parquet(PARQUET_DIR / "oe_player_games.parquet", index=False)

    lp_team = lp_player = None
    if not args.skip_lp:
        try:
            lp_team, lp_player = ingest_leaguepedia()
        except FileNotFoundError as exc:
            if not args.allow_missing_lp:
                raise
            print(f"[lp] optional enrichment unavailable; continuing: {exc}")
            lp_team, lp_player = pd.DataFrame(), pd.DataFrame()

    maps = build_map_warehouse(
        lp_team=lp_team,
        oe_team=combined_team,
        lp_players=lp_player,
    )

    source_counts = {}
    if "source" in combined_team.columns:
        source_counts = {
            str(source): int(count)
            for source, count in combined_team["source"].value_counts(dropna=False).items()
        }

    oe_meta_path = PARQUET_DIR / "oe_meta.json"
    try:
        oe_meta = json.loads(oe_meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        oe_meta = {}
    oe_source_files = (
        oe_meta.get("source_files")
        if isinstance(oe_meta.get("source_files"), list)
        else []
    )
    output_paths = {
        "team_games": PARQUET_DIR / "oe_team_games.parquet",
        "player_games": PARQUET_DIR / "oe_player_games.parquet",
        "maps": PARQUET_DIR / "maps.parquet",
        "rating_players": PARQUET_DIR / "players.parquet",
    }
    meta = {
        "schema_version": "scryglass:warehouse-refresh-manifest:v2",
        "refreshed_at": datetime.now(timezone.utc).isoformat(),
        "input_control": {
            "oe_years_requested": args.oe_years,
            "oe_full_raw_rebuild": bool(not args.skip_oe and args.oe_years is None),
            "oe_download_requested": bool(args.download_oe),
            "oe_strict_refresh_requested": bool(args.refresh_oe),
            "grid_download_requested": bool(args.download_grid),
            "grid_local_ingest_enabled": bool(not args.skip_grid),
            "leaguepedia_local_ingest_enabled": bool(not args.skip_lp),
        },
        "n_maps": int(len(maps)),
        "oe_rows": int(len(oe_team)) if oe_team is not None else 0,
        "grid_rows": int(len(grid_team)),
        "combined_team_rows": int(len(combined_team)),
        "combined_player_rows": int(len(combined_player)),
        "source_counts": source_counts,
        "lp_side_rows": int(len(lp_team)) if lp_team is not None else 0,
        "oe_matched": int(maps["oe_matched"].sum()) if "oe_matched" in maps.columns else 0,
        "sources": {
            "oe_normalization_manifest": (
                _file_record(oe_meta_path) if oe_meta_path.is_file() else None
            ),
            "oe_annual_files": oe_source_files,
            "oe_transport_receipts": _matching_oe_receipts(oe_source_files),
            "leaguepedia_caches": [
                _file_record(path)
                for path in (LP_GAMES, LP_PLAYERS)
                if path.is_file()
            ],
            "grid_normalization_manifest": (
                _file_record(PARQUET_DIR / "grid_meta.json")
                if (PARQUET_DIR / "grid_meta.json").is_file()
                else None
            ),
        },
        "outputs": {
            "team_games": _file_record(
                output_paths["team_games"], rows=len(combined_team)
            ),
            "player_games": _file_record(
                output_paths["player_games"], rows=len(combined_player)
            ),
            "maps": _file_record(output_paths["maps"], rows=len(maps)),
            "rating_players": (
                _file_record(
                    output_paths["rating_players"], rows=len(combined_player)
                )
                if output_paths["rating_players"].is_file()
                else None
            ),
        },
        "authority": {
            "descriptive_warehouse_provenance": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "This manifest binds warehouse inputs and outputs only. It does not "
            "validate a model, probability, odds, recommendation, or wager."
        ),
    }
    meta["manifest_canonical_sha256"] = _canonical_sha256(meta)
    (PARQUET_DIR / "refresh_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[refresh]", json.dumps(meta))


if __name__ == "__main__":
    main()
