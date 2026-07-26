"""Fail-closed data-quality audit for a public Scryglass pack.

The audit intentionally works on an already-built pack.  It does not infer
missing source data, repair rows, or treat a page rendering as evidence.  The
JSON result is suitable for CI and the human-readable output preserves grain,
counts, rates, and representative keys for launch review.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from lol_kills.export import pack_spec


SEVERITIES = ("launch blocker", "major", "minor", "informational")


def _read_parts(root: Path, prefix: str) -> pd.DataFrame:
    paths = sorted(root.glob(f"{prefix}/year=*/part.parquet"))
    if not paths:
        direct = root / f"{prefix}.parquet"
        paths = [direct] if direct.exists() else []
    if not paths:
        return pd.DataFrame()
    # Read each file directly.  Dataset discovery would also infer the
    # ``year=`` directory as a partition column, which conflicts with the
    # explicit year column stored in these files.
    return pd.concat([pq.ParquetFile(path).read().to_pandas() for path in paths], ignore_index=True)


def _examples(values: Any, limit: int = 5) -> list[str]:
    if values is None:
        return []
    if isinstance(values, pd.Series):
        values = values.tolist()
    return [str(value) for value in list(values)[:limit]]


def _count_true(mask: pd.Series) -> int:
    return int(mask.fillna(False).sum()) if not mask.empty else 0


def _add(
    findings: list[dict[str, Any]],
    *,
    severity: str,
    code: str,
    grain: str,
    evidence: str,
    count: int,
    examples: list[str] | None = None,
) -> None:
    if severity not in SEVERITIES:
        raise ValueError(f"unknown severity: {severity}")
    findings.append(
        {
            "severity": severity,
            "code": code,
            "grain": grain,
            "evidence": evidence,
            "count": int(count),
            "examples": examples or [],
        }
    )


def _audit_maps(maps: pd.DataFrame, findings: list[dict[str, Any]]) -> dict[str, Any]:
    required = set(pack_spec.maps_columns())
    missing = sorted(required - set(maps.columns))
    if missing:
        _add(
            findings,
            severity="launch blocker" if "grid_completion_source" in missing else "major",
            code="maps_schema_missing_columns",
            grain="map",
            evidence="The published maps schema is missing stable identity/provenance columns.",
            count=len(missing),
            examples=missing,
        )

    key = "game_uid" if "game_uid" in maps.columns else "oe_gameid"
    duplicate_count = int(maps.duplicated(key, keep=False).sum()) if key in maps.columns else len(maps)
    if duplicate_count:
        _add(
            findings,
            severity="launch blocker",
            code="duplicate_map_identity",
            grain="map",
            evidence=f"{duplicate_count} rows share a {key} identity.",
            count=duplicate_count,
            examples=_examples(maps.loc[maps.duplicated(key, keep=False), key].drop_duplicates()),
        )

    if {"blue_teamname", "red_teamname"}.issubset(maps.columns):
        same_side = maps["blue_teamname"].astype(str).eq(maps["red_teamname"].astype(str))
        if _count_true(same_side):
            _add(
                findings,
                severity="launch blocker",
                code="same_team_sides",
                grain="map",
                evidence="A map has the same canonical team on both sides.",
                count=_count_true(same_side),
                examples=_examples(maps.loc[same_side, key] if key in maps.columns else None),
            )

    if {"blue_result", "red_result"}.issubset(maps.columns):
        result_sum = pd.to_numeric(maps["blue_result"], errors="coerce") + pd.to_numeric(
            maps["red_result"], errors="coerce"
        )
        bad_results = ~result_sum.eq(1)
        if _count_true(bad_results):
            _add(
                findings,
                severity="launch blocker",
                code="invalid_map_result",
                grain="map",
                evidence="Completed maps do not have exactly one winning side.",
                count=_count_true(bad_results),
                examples=_examples(maps.loc[bad_results, key] if key in maps.columns else None),
            )

    if {"y_blue_win", "blue_result"}.issubset(maps.columns):
        y = pd.to_numeric(maps["y_blue_win"], errors="coerce")
        blue_result = pd.to_numeric(maps["blue_result"], errors="coerce")
        mismatch = y.notna() & blue_result.notna() & ~y.eq(blue_result)
        if _count_true(mismatch):
            _add(
                findings,
                severity="launch blocker",
                code="result_target_mismatch",
                grain="map",
                evidence="The model target y_blue_win disagrees with the published blue result.",
                count=_count_true(mismatch),
                examples=_examples(maps.loc[mismatch, key] if key in maps.columns else None),
            )

    for column in ("gamelength", "total_kills"):
        if column not in maps.columns:
            continue
        values = pd.to_numeric(maps[column], errors="coerce")
        bad = values.isna() | values.lt(0)
        if column == "gamelength":
            bad |= values.eq(0)
        if _count_true(bad):
            _add(
                findings,
                severity="major",
                code=f"invalid_{column}",
                grain="map",
                evidence=f"{column} contains missing, negative, or zero values where a completed map is required.",
                count=_count_true(bad),
                examples=_examples(maps.loc[bad, key] if key in maps.columns else None),
            )

    if "source_grid" in maps.columns:
        grid = maps["source_grid"].fillna(False).astype(bool)
        if grid.any() and "grid_completion_source" not in maps.columns:
            _add(
                findings,
                severity="launch blocker",
                code="grid_completion_provenance_missing",
                grain="map",
                evidence="GRID-backed maps are published without their verified completion-source field.",
                count=int(grid.sum()),
                examples=_examples(maps.loc[grid, key] if key in maps.columns else None),
            )
        elif grid.any():
            missing_provenance = grid & maps["grid_completion_source"].isna()
            if _count_true(missing_provenance):
                _add(
                    findings,
                    severity="launch blocker",
                    code="grid_completion_provenance_null",
                    grain="map",
                    evidence="GRID-backed maps have no completion provenance and cannot support a verified end state.",
                    count=_count_true(missing_provenance),
                    examples=_examples(maps.loc[missing_provenance, key] if key in maps.columns else None),
                )

    if {"league", "tournament"}.issubset(maps.columns):
        tournament = maps["tournament"].astype(str).str.upper()
        suspect_intl = maps["league"].astype(str).str.upper().eq("INTL") & (
            tournament.str.contains(r"\bNACL\b", regex=True, na=False)
            | tournament.str.contains("CIRCUITO DESAFIANTE", regex=False, na=False)
        )
        if _count_true(suspect_intl):
            _add(
                findings,
                severity="major",
                code="developmental_league_leaked_to_intl",
                grain="map/competition",
                evidence="Developmental tournament titles are labeled INTL and would leak into international scope filters.",
                count=_count_true(suspect_intl),
                examples=_examples(maps.loc[suspect_intl, key] if key in maps.columns else None),
            )

    return {
        "rows": int(len(maps)),
        "unique_game_uid": int(maps["game_uid"].nunique()) if "game_uid" in maps.columns else None,
        "unique_oe_gameid": int(maps["oe_gameid"].nunique()) if "oe_gameid" in maps.columns else None,
        "source_grid_rows": int(maps["source_grid"].fillna(False).astype(bool).sum()) if "source_grid" in maps.columns else 0,
        "source_oe_rows": int(maps["source_oe"].fillna(False).astype(bool).sum()) if "source_oe" in maps.columns else 0,
        "years": sorted(int(value) for value in pd.to_numeric(maps.get("year"), errors="coerce").dropna().unique()) if "year" in maps.columns else [],
    }


def _audit_series(maps: pd.DataFrame, findings: list[dict[str, Any]]) -> dict[str, Any]:
    if not {"grid_series_id", "grid_game_index"}.issubset(maps.columns):
        return {"grid_series": 0, "gapped_series": 0, "tied_multi_map_series": 0}
    grid = maps[maps["grid_series_id"].notna()].copy()
    gapped: list[str] = []
    tied: list[str] = []
    for series_id, group in grid.groupby("grid_series_id", dropna=True, sort=True):
        indices = pd.to_numeric(group["grid_game_index"], errors="coerce")
        ordered = sorted(int(value) for value in indices.dropna().tolist()) if indices.notna().all() else []
        if ordered != list(range(1, len(ordered) + 1)) or len(ordered) != len(group):
            gapped.append(str(series_id))
        if "y_blue_win" in group.columns:
            wins = pd.to_numeric(group["y_blue_win"], errors="coerce").sum()
            if len(group) > 1 and wins == len(group) / 2:
                tied.append(str(series_id))
    if gapped:
        _add(
            findings,
            severity="launch blocker",
            code="gapped_grid_series",
            grain="series",
            evidence="GRID series contain missing, duplicate, or non-positive game indices; format inference must fail closed.",
            count=len(gapped),
            examples=gapped,
        )
    if tied:
        _add(
            findings,
            severity="major",
            code="tied_grid_series",
            grain="series",
            evidence="A multi-map GRID group has no series winner and cannot be weighted as a completed series.",
            count=len(tied),
            examples=tied,
        )
    return {
        "grid_series": int(grid["grid_series_id"].nunique()),
        "gapped_series": len(gapped),
        "tied_multi_map_series": len(tied),
    }


def _audit_side_grains(root: Path, maps: pd.DataFrame, findings: list[dict[str, Any]]) -> dict[str, Any]:
    team = _read_parts(root, "team_games")
    player = _read_parts(root, "player_games")
    out: dict[str, Any] = {}
    if not team.empty and {"gameid", "side"}.issubset(team.columns):
        duplicated = team.duplicated(["gameid", "side"], keep=False)
        if _count_true(duplicated):
            _add(
                findings,
                severity="launch blocker",
                code="duplicate_team_side_rows",
                grain="game-side",
                evidence="Team feed has more than one aggregate row for a game and side.",
                count=_count_true(duplicated),
                examples=_examples(team.loc[duplicated, "gameid"].drop_duplicates()),
            )
        out["team_rows"] = int(len(team))
        out["team_unique_games"] = int(team["gameid"].nunique())
    if not player.empty and {"gameid", "side", "position"}.issubset(player.columns):
        duplicated = player.duplicated(["gameid", "side", "position"], keep=False)
        if _count_true(duplicated):
            _add(
                findings,
                severity="launch blocker",
                code="duplicate_player_position_rows",
                grain="game-side-position",
                evidence="Player feed has more than one row for a game, side, and role.",
                count=_count_true(duplicated),
                examples=_examples(player.loc[duplicated, "gameid"].drop_duplicates()),
            )
        out["player_rows"] = int(len(player))
        out["player_unique_games"] = int(player["gameid"].nunique())
    return out


def _audit_history(root: Path, maps: pd.DataFrame, findings: list[dict[str, Any]]) -> dict[str, Any]:
    history_path = root / "features" / "ratings_history.parquet"
    if not history_path.exists() or "game_uid" not in maps.columns:
        return {}
    history = pq.read_table(history_path).to_pandas()
    map_ids = set(maps["game_uid"].dropna().astype(str))
    hist_ids = set(history.get("game_uid", pd.Series(dtype=str)).dropna().astype(str))
    missing = sorted(map_ids - hist_ids)
    extra = sorted(hist_ids - map_ids)
    duplicate = int(history.duplicated("game_uid", keep=False).sum()) if "game_uid" in history.columns else len(history)
    if missing or extra or duplicate:
        _add(
            findings,
            severity="major",
            code="ratings_history_map_mismatch",
            grain="map/rating-history",
            evidence="Rating history is not a one-to-one projection of published map identities.",
            count=len(missing) + len(extra) + duplicate,
            examples=missing[:3] + extra[:3],
        )
    return {"history_rows": int(len(history)), "history_missing_maps": len(missing), "history_extra_maps": len(extra), "history_duplicate_rows": duplicate}


def _audit_current_membership(
    root: Path,
    manifest: dict[str, Any],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    tournaments = manifest.get("current_tournaments") or {}
    if not isinstance(tournaments, dict) or not tournaments:
        return {"current_tournaments": {}}
    records_path = root / "features" / "team_records.json"
    if not records_path.exists():
        return {"current_tournaments": tournaments, "checked_team_records": 0, "mismatches": 0}
    records = json.loads(records_path.read_text(encoding="utf-8"))
    mismatches: list[str] = []
    checked = 0
    as_of = pd.to_datetime(
        manifest.get("current_tournament_as_of") or manifest.get("data_as_of"),
        errors="coerce",
        utc=True,
    )
    window_days = int(manifest.get("recent_activity_window_days") or 90)
    for display, record in records.items():
        league = record.get("current_league")
        expected = tournaments.get(league)
        if not expected:
            continue
        observed = pd.to_datetime(record.get("current_date"), errors="coerce", utc=True)
        if pd.isna(as_of) or pd.isna(observed):
            continue
        age_days = (as_of - observed).total_seconds() / 86400
        if age_days < 0 or age_days > window_days:
            continue
        checked += 1
        if record.get("current_tournament") != expected:
            mismatches.append(str(display))
    if mismatches:
        _add(
            findings,
            severity="major",
            code="current_tournament_membership_mismatch",
            grain="team-record/league-tournament",
            evidence="A record is currently affiliated with a league whose pack-declared current tournament does not match its observed tournament membership.",
            count=len(mismatches),
            examples=mismatches,
        )
    return {
        "current_tournaments": tournaments,
        "current_tournament_as_of": manifest.get("current_tournament_as_of"),
        "checked_team_records": checked,
        "mismatches": len(mismatches),
    }


def audit_pack(root: Path) -> dict[str, Any]:
    """Return a machine-readable launch audit for ``root``."""

    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    findings: list[dict[str, Any]] = []
    if not manifest.get("quality"):
        _add(
            findings,
            severity="major",
            code="manifest_quality_missing",
            grain="pack-manifest",
            evidence="The pack manifest does not expose its draft/data-quality summary.",
            count=1,
        )
    maps = _read_parts(root, "maps")
    map_summary = _audit_maps(maps, findings)
    series_summary = _audit_series(maps, findings)
    side_summary = _audit_side_grains(root, maps, findings)
    history_summary = _audit_history(root, maps, findings)
    membership_summary = _audit_current_membership(root, manifest, findings)

    records_path = root / "features" / "team_records.json"
    stale_records = 0
    if records_path.exists():
        records = json.loads(records_path.read_text(encoding="utf-8"))
        as_of = (
            pd.to_datetime(manifest.get("data_as_of"), errors="coerce", utc=True)
            if manifest.get("data_as_of")
            else (
                pd.to_datetime(maps["date"], errors="coerce", utc=True).max()
                if "date" in maps.columns
                else pd.NaT
            )
        )
        if pd.notna(as_of):
            dates = pd.to_datetime(
                pd.Series([record.get("current_date") for record in records.values()]),
                errors="coerce",
                utc=True,
            )
            age_days = (as_of - dates).dt.total_seconds() / 86400
            stale_records = int((age_days > 90).fillna(False).sum())
    if stale_records:
        _add(
            findings,
            severity="informational",
            code="stale_record_history",
            grain="team-record",
            evidence="Historical team records are older than the scoped ladder recency guard; they remain available outside scoped views.",
            count=stale_records,
        )

    counts = {severity: sum(f["severity"] == severity for f in findings) for severity in SEVERITIES}
    return {
        "pack_id": manifest.get("pack_id"),
        "schema_version": manifest.get("schema_version"),
        "manifest": {
            "data_as_of": manifest.get("data_as_of"),
            "total_files": manifest.get("total_files"),
            "quality_present": bool(manifest.get("quality")),
            "current_tournaments": membership_summary.get("current_tournaments", {}),
            "current_tournament_as_of": membership_summary.get("current_tournament_as_of"),
        },
        "counts": counts,
        "maps": map_summary,
        "series": series_summary,
        "side_grains": side_summary,
        "history": history_summary,
        "membership": membership_summary,
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path, help="path to one unpacked public pack")
    parser.add_argument("--json", action="store_true", help="emit JSON only")
    args = parser.parse_args(argv)
    report = audit_pack(args.pack)
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(f"pack={report['pack_id']} schema={report['schema_version']}")
        print("findings=" + ", ".join(f"{key}={value}" for key, value in report["counts"].items()))
        for finding in report["findings"]:
            print(f"[{finding['severity']}] {finding['code']} ({finding['count']} {finding['grain']}): {finding['evidence']}")
    return 1 if report["counts"]["launch blocker"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
