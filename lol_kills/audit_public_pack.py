"""Fail-closed data-quality audit for a public Scryglass pack.

The audit intentionally works on an already-built pack.  It does not infer
missing source data, repair rows, or treat a page rendering as evidence.  The
JSON result is suitable for CI and the human-readable output preserves grain,
counts, rates, and representative keys for launch review.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from lol_kills.etl.competition import team_identity_key
from lol_kills.export import pack_spec


SEVERITIES = ("launch blocker", "major", "minor", "informational")
RELEASE_BLOCKING_SEVERITIES = ("launch blocker", "major")


def _current_utc(
    clock: Callable[[], datetime] | None = None,
) -> pd.Timestamp:
    value = (clock or (lambda: datetime.now(timezone.utc)))()
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def release_gate(counts: dict[str, int]) -> dict[str, Any]:
    """Return the publication gate derived from audit severity counts."""

    blocking_findings = sum(
        int(counts.get(severity, 0)) for severity in RELEASE_BLOCKING_SEVERITIES
    )
    return {
        "ready": blocking_findings == 0,
        "blocking_severities": list(RELEASE_BLOCKING_SEVERITIES),
        "blocking_findings": blocking_findings,
    }


def require_release_gate(report: dict[str, Any]) -> None:
    """Raise unless a full audit has zero blocker and major findings."""

    counts = report.get("counts")
    if not isinstance(counts, dict):
        raise RuntimeError("Public pack audit report has no severity counts")
    gate = release_gate(counts)
    reported_gate = report.get("release_gate")
    if isinstance(reported_gate, dict) and bool(
        reported_gate.get("ready")
    ) != bool(gate["ready"]):
        raise RuntimeError("Public pack audit report has an inconsistent release gate")
    if not gate.get("ready"):
        raise RuntimeError(
            "Public pack failed release gate: "
            + ", ".join(
                f"{severity}={int(counts.get(severity, 0))}"
                for severity in RELEASE_BLOCKING_SEVERITIES
            )
        )


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


def _false_like(value: Any) -> bool:
    return value is False or value == 0 or str(value).strip().casefold() in {
        "false",
        "0",
    }


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
            severity=(
                "launch blocker"
                if {
                    "grid_completion_source",
                    "canonical_map_source",
                    "map_detail_source",
                }
                & set(missing)
                else "major"
            ),
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

    if "map_detail_source" in maps.columns:
        allowed_detail_sources = {
            "oe_wide_feature_map",
            "oe_team_aggregate",
            "grid_event_detail",
            "grid_team_aggregate",
        }
        detail_source = maps["map_detail_source"].fillna("").astype(str)
        invalid_detail_source = ~detail_source.isin(allowed_detail_sources)
        if _count_true(invalid_detail_source):
            _add(
                findings,
                severity="launch blocker",
                code="map_detail_provenance_invalid",
                grain="map",
                evidence=(
                    "A public map does not identify which source grain supplied "
                    "its detail fields; missing values cannot be interpreted "
                    "safely."
                ),
                count=_count_true(invalid_detail_source),
                examples=_examples(
                    maps.loc[invalid_detail_source, key]
                    if key in maps.columns
                    else None
                ),
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

    if "canonical_map_source" in maps.columns:
        canonical_source = (
            maps["canonical_map_source"].fillna("").astype(str)
        )
        invalid = ~canonical_source.isin({"oe", "grid_gap_fill"})
        if _count_true(invalid):
            _add(
                findings,
                severity="launch blocker",
                code="canonical_map_source_invalid",
                grain="map",
                evidence=(
                    "Canonical map inclusion must be explicitly OE or verified "
                    "GRID gap fill; detail enrichment is a separate field."
                ),
                count=_count_true(invalid),
                examples=_examples(
                    maps.loc[invalid, key] if key in maps.columns else None
                ),
            )
    else:
        canonical_source = pd.Series("", index=maps.index)

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
        canonical_grid = canonical_source.eq("grid_gap_fill")
        if _count_true(canonical_grid & ~grid):
            _add(
                findings,
                severity="launch blocker",
                code="grid_gap_fill_source_flag_mismatch",
                grain="map",
                evidence=(
                    "A canonical GRID gap-fill map does not retain the GRID "
                    "contribution flag."
                ),
                count=_count_true(canonical_grid & ~grid),
                examples=_examples(
                    maps.loc[canonical_grid & ~grid, key]
                    if key in maps.columns
                    else None
                ),
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
        "canonical_map_sources": (
            maps["canonical_map_source"]
            .fillna("")
            .astype(str)
            .value_counts()
            .sort_index()
            .to_dict()
            if "canonical_map_source" in maps.columns
            else {}
        ),
        "years": sorted(int(value) for value in pd.to_numeric(maps.get("year"), errors="coerce").dropna().unique()) if "year" in maps.columns else [],
    }


def _audit_series(maps: pd.DataFrame, findings: list[dict[str, Any]]) -> dict[str, Any]:
    if not {"grid_series_id", "grid_game_index"}.issubset(maps.columns):
        return {
            "grid_series": 0,
            "gapped_series": 0,
            "quarantined_gapped_series": 0,
            "unsafe_gapped_series": 0,
            "tied_multi_map_series": 0,
        }
    grid = maps[maps["grid_series_id"].notna()].copy()
    gapped: list[str] = []
    quarantined_gapped: list[str] = []
    unsafe_gapped: list[str] = []
    tied: list[str] = []
    for series_id, group in grid.groupby("grid_series_id", dropna=True, sort=True):
        indices = pd.to_numeric(group["grid_game_index"], errors="coerce")
        ordered = sorted(int(value) for value in indices.dropna().tolist()) if indices.notna().all() else []
        if ordered != list(range(1, len(ordered) + 1)) or len(ordered) != len(group):
            gapped.append(str(series_id))
            explicitly_quarantined = (
                "series_rating_eligible" in group.columns
                and group["series_rating_eligible"].map(_false_like).all()
            )
            (
                quarantined_gapped
                if explicitly_quarantined
                else unsafe_gapped
            ).append(str(series_id))
        team_columns = (
            ("blue_teamname", "red_teamname")
            if {"blue_teamname", "red_teamname"}.issubset(group.columns)
            else ("blue_team", "red_team")
            if {"blue_team", "red_team"}.issubset(group.columns)
            else None
        )
        if team_columns and "y_blue_win" in group.columns:
            blue_column, red_column = team_columns
            scores: dict[str, int] = {}
            complete_results = True
            for _, row in group.iterrows():
                result = pd.to_numeric(
                    pd.Series([row["y_blue_win"]]), errors="coerce"
                ).iloc[0]
                blue = str(row[blue_column] or "").strip()
                red = str(row[red_column] or "").strip()
                if pd.isna(result) or result not in (0, 1) or not blue or not red:
                    complete_results = False
                    break
                winner = blue if int(result) == 1 else red
                scores[winner] = scores.get(winner, 0) + 1
            score_values = sorted(scores.values())
            if (
                complete_results
                and len(group) > 1
                and len(score_values) == 2
                and score_values[0] == score_values[1]
            ):
                tied.append(str(series_id))
    if unsafe_gapped:
        _add(
            findings,
            severity="launch blocker",
            code="gapped_grid_series",
            grain="series",
            evidence=(
                "GRID series contain missing, duplicate, or non-positive game "
                "indices without an explicit series-rating quarantine."
            ),
            count=len(unsafe_gapped),
            examples=unsafe_gapped,
        )
    if quarantined_gapped:
        _add(
            findings,
            severity="informational",
            code="quarantined_gapped_grid_series",
            grain="series",
            evidence=(
                "GRID map history is retained, but the canonical contract "
                "excludes the gapped series from ratings and series surfaces."
            ),
            count=len(quarantined_gapped),
            examples=quarantined_gapped,
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
        "quarantined_gapped_series": len(quarantined_gapped),
        "unsafe_gapped_series": len(unsafe_gapped),
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
    map_key = (
        "oe_gameid"
        if "oe_gameid" in maps.columns
        else ("game_uid" if "game_uid" in maps.columns else None)
    )
    if map_key is not None:
        map_ids = set(maps[map_key].dropna().astype(str))
        for label, frame in (("team", team), ("player", player)):
            if frame.empty or "gameid" not in frame.columns:
                continue
            side_ids = set(frame["gameid"].dropna().astype(str))
            missing_maps = sorted(side_ids - map_ids)
            missing_side_rows = sorted(map_ids - side_ids)
            if missing_maps or missing_side_rows:
                _add(
                    findings,
                    severity="launch blocker",
                    code=f"{label}_game_map_population_mismatch",
                    grain=f"{label}-game/map",
                    evidence=(
                        f"The published {label} feed and map table do not cover "
                        "the same game identities, so pages and ratings use "
                        "different populations."
                    ),
                    count=len(missing_maps) + len(missing_side_rows),
                    examples=missing_maps[:3] + missing_side_rows[:2],
                )
            out[f"{label}_games_missing_maps"] = len(missing_maps)
            out[f"maps_missing_{label}_games"] = len(missing_side_rows)
    return out


def _audit_declared_years(
    root: Path,
    manifest: dict[str, Any],
    maps: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    declared = {
        int(value)
        for value in (manifest.get("filters") or {}).get("years", [])
        if str(value).strip()
    }
    if not declared:
        return {"declared_years": []}
    summary: dict[str, Any] = {"declared_years": sorted(declared)}
    for label, frame in (
        ("maps", maps),
        ("team_games", _read_parts(root, "team_games")),
        ("player_games", _read_parts(root, "player_games")),
    ):
        if frame.empty or "date" not in frame.columns:
            continue
        dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        outside = dates.notna() & ~dates.dt.year.isin(declared)
        outside_rows = int(outside.sum())
        game_column = (
            "gameid"
            if "gameid" in frame.columns
            else ("oe_gameid" if "oe_gameid" in frame.columns else "game_uid")
        )
        outside_games = (
            int(frame.loc[outside, game_column].astype(str).nunique())
            if outside_rows and game_column in frame.columns
            else outside_rows
        )
        summary[f"{label}_outside_year_rows"] = outside_rows
        summary[f"{label}_outside_year_games"] = outside_games
        if outside_rows:
            _add(
                findings,
                severity="launch blocker",
                code=f"{label}_outside_declared_years",
                grain=f"{label} row",
                evidence=(
                    f"The pack declares years {sorted(declared)} but publishes "
                    f"{outside_rows} {label} rows from another calendar year."
                ),
                count=outside_rows,
                examples=_examples(
                    frame.loc[outside, game_column].drop_duplicates()
                    if game_column in frame.columns
                    else None
                ),
            )
        if {"oe_year", "year"}.issubset(frame.columns):
            oe_year = pd.to_numeric(frame["oe_year"], errors="coerce")
            partition_year = pd.to_numeric(frame["year"], errors="coerce")
            conflict = (
                oe_year.notna()
                & partition_year.notna()
                & ~oe_year.eq(partition_year)
            )
            conflicts = int(conflict.sum())
            summary[f"{label}_year_field_conflicts"] = conflicts
            if conflicts:
                _add(
                    findings,
                    severity="major",
                    code=f"{label}_year_field_conflict",
                    grain=f"{label} row",
                    evidence=(
                        "Source year and derived partition year disagree; the "
                        "exporter must use one precedence rule rather than OR."
                    ),
                    count=conflicts,
                    examples=_examples(
                        frame.loc[conflict, game_column].drop_duplicates()
                        if game_column in frame.columns
                        else None
                    ),
                )
    return summary


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


def _audit_rating_release_contract(
    root: Path,
    manifest: dict[str, Any],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require a non-empty snapshot and matching chronological model gate."""

    if not isinstance(manifest.get("files"), list) or not manifest.get("files"):
        return {}
    meta_path = root / "features" / "ratings_meta.json"
    snapshot_path = root / "features" / "ratings_snapshot.parquet"
    if not meta_path.is_file() or not snapshot_path.is_file():
        _add(
            findings,
            severity="launch blocker",
            code="team_rating_release_missing",
            grain="team-rating release",
            evidence=(
                "The public pack is missing the team-rating snapshot or its "
                "model metadata."
            ),
            count=1,
        )
        return {}

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        snapshot = pq.read_table(snapshot_path).to_pandas()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _add(
            findings,
            severity="launch blocker",
            code="team_rating_release_unreadable",
            grain="team-rating release",
            evidence="The public team-rating release cannot be parsed.",
            count=1,
            examples=[type(exc).__name__],
        )
        return {}

    validation_path = (
        root / "models" / "model_validation_2026-07-27.json"
    )
    try:
        validation = json.loads(
            validation_path.read_text(encoding="utf-8")
        )
        team_gate = validation.get("team_rating") or {}
    except (OSError, json.JSONDecodeError, ValueError):
        team_gate = {}

    final_test = team_gate.get("final_test") or {}
    paired = team_gate.get("paired_primary_comparison") or {}

    def finite_number(value: Any) -> bool:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return False
        return bool(pd.notna(number)) and number not in {
            float("inf"),
            float("-inf"),
        }

    probability_metrics_ok = (
        int(final_test.get("series") or 0) >= 500
        and all(
            finite_number(final_test.get(metric))
            for metric in ("log_loss", "brier", "ece_10_equal_width")
        )
        and 0.0 <= float(final_test.get("brier", -1.0)) <= 1.0
        and 0.0 <= float(final_test.get("ece_10_equal_width", -1.0)) <= 1.0
    )
    interval = paired.get("confidence_interval")
    paired_ok = (
        paired.get("primary_score") == "log_loss"
        and paired.get("decision") in {"superior", "noninferior"}
        and isinstance(interval, list)
        and len(interval) == 2
        and all(finite_number(value) for value in interval)
    )
    model_gate_ok = (
        team_gate.get("gate_status") == "passed"
        and team_gate.get("estimand")
        == "pre_series_organization_strength_probability"
        and team_gate.get("model_id") == meta.get("model_id")
        and team_gate.get("model_version") == meta.get("model_version")
        and team_gate.get("model_code_sha256")
        == meta.get("model_code_sha256")
        and team_gate.get("model_config_sha256")
        == meta.get("model_config_sha256")
        and bool((team_gate.get("temporal_audit") or {}).get("ok"))
        and probability_metrics_ok
        and paired_ok
    )

    ledger = meta.get("series_ledger_audit") or {}
    eligible_series = int(ledger.get("n_rating_eligible_series") or 0)
    eligible_maps = int(ledger.get("n_rating_eligible_maps") or 0)
    required_columns = {
        "team",
        "team_key",
        "mu_total",
        "sigma",
        "rating_p05",
        "model",
        "model_version",
        "comparison_component_id",
        "comparison_component_size",
        "cross_component_rankable",
    }
    missing_columns = sorted(required_columns - set(snapshot.columns))
    numeric_ok = (
        not snapshot.empty
        and not missing_columns
        and snapshot[["mu_total", "sigma", "rating_p05"]]
        .apply(pd.to_numeric, errors="coerce")
        .notna()
        .all()
        .all()
    )
    row_version_ok = bool(
        not snapshot.empty
        and not missing_columns
        and snapshot["model_version"]
        .astype(str)
        .eq(str(meta.get("model_version") or ""))
        .all()
    )
    component_ok = bool(
        not snapshot.empty
        and not missing_columns
        and snapshot["comparison_component_id"]
        .astype(str)
        .str.strip()
        .ne("")
        .all()
        and pd.to_numeric(
            snapshot["comparison_component_size"], errors="coerce"
        )
        .gt(0)
        .all()
        and snapshot["cross_component_rankable"].eq(False).all()
    )
    if (
        meta.get("model") != "series_dynamic_bt"
        or not bool(ledger.get("ok"))
        or eligible_series <= 0
        or eligible_maps <= 0
        or int(meta.get("n_series") or 0) <= 0
        or not bool((meta.get("input_audit") or {}).get("ok"))
        or not numeric_ok
        or not row_version_ok
        or not component_ok
        or not model_gate_ok
    ):
        _add(
            findings,
            severity="launch blocker",
            code="team_rating_release_empty_or_unvalidated",
            grain="team-rating release",
            evidence=(
                "The dynamic series team snapshot is empty, non-finite, or was "
                "fit without format-verified completed series, or is not tied "
                "to a passing chronological model gate for the exact code and "
                "configuration, or lacks connected-comparison boundaries."
            ),
            count=1,
            examples=[
                f"snapshot_rows={len(snapshot)}",
                f"eligible_series={eligible_series}",
                f"eligible_maps={eligible_maps}",
                f"missing_columns={missing_columns}",
                f"row_version_ok={row_version_ok}",
                f"component_ok={component_ok}",
                f"model_gate_status={team_gate.get('gate_status')}",
                f"model_version_match={team_gate.get('model_version') == meta.get('model_version')}",
            ],
        )

    return {
        "snapshot_rows": int(len(snapshot)),
        "eligible_series": eligible_series,
        "eligible_maps": eligible_maps,
        "model": meta.get("model"),
        "model_version": meta.get("model_version"),
        "model_gate_status": team_gate.get("gate_status"),
        "model_gate_ok": model_gate_ok,
        "row_version_ok": row_version_ok,
        "component_ok": component_ok,
        "missing_columns": missing_columns,
    }


def _audit_manifest_release_contract(
    root: Path,
    manifest: dict[str, Any],
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit the governed allowlist on a complete exported public pack."""

    declared_files = manifest.get("files")
    if not isinstance(declared_files, list) or not declared_files:
        return {}

    declared_paths = {
        str(item.get("relative") or item.get("path") or "").strip()
        for item in declared_files
        if isinstance(item, dict)
    }
    declared_paths.discard("")

    if manifest.get("schema_version") != pack_spec.SCHEMA_VERSION:
        _add(
            findings,
            severity="launch blocker",
            code="public_schema_version_stale",
            grain="pack-manifest",
            evidence=(
                "The immutable pack schema does not match the application "
                "release contract."
            ),
            count=1,
            examples=[
                f"declared={manifest.get('schema_version')}",
                f"required={pack_spec.SCHEMA_VERSION}",
            ],
        )

    one_bundle_clock = (
        isinstance(manifest.get("pack_id"), str)
        and manifest.get("pack_id")
        and manifest.get("model_pack_id") == manifest.get("pack_id")
    )
    if not one_bundle_clock:
        _add(
            findings,
            severity="launch blocker",
            code="model_bundle_clock_unbound",
            grain="pack-manifest",
            evidence=(
                "Data and packaged model artifacts do not declare the same "
                "immutable bundle ID."
            ),
            count=1,
            examples=[
                f"pack_id={manifest.get('pack_id')}",
                f"model_pack_id={manifest.get('model_pack_id')}",
            ],
        )

    quarantined = sorted(
        path
        for path in declared_paths
        if pack_spec.public_path_quarantine_reason(path) is not None
    )
    if quarantined:
        _add(
            findings,
            severity="launch blocker",
            code="quarantined_public_artifacts",
            grain="public-manifest path",
            evidence=(
                "The public manifest exposes artifacts that the governed "
                "allowlist explicitly quarantines."
            ),
            count=len(quarantined),
            examples=quarantined,
        )

    required_paths = {
        *(f"models/{name}" for name in pack_spec.PINNED_MODEL_FILES),
        *(f"studies/grubs/{name}" for name in pack_spec.GRUBS_MODEL_FILES),
        "meta/source_summary.json",
    }
    missing_required = sorted(required_paths - declared_paths)
    if missing_required:
        _add(
            findings,
            severity="launch blocker",
            code="required_public_artifacts_missing",
            grain="public-manifest path",
            evidence=(
                "The immutable pack omits one or more pinned model, article, "
                "or source-provenance artifacts required by the public contract."
            ),
            count=len(missing_required),
            examples=missing_required,
        )

    missing_declared = sorted(
        path for path in declared_paths if not (root / path).is_file()
    )
    if missing_declared:
        _add(
            findings,
            severity="launch blocker",
            code="declared_public_file_missing",
            grain="public-manifest path",
            evidence="A file declared by the immutable manifest is absent on disk.",
            count=len(missing_declared),
            examples=missing_declared,
        )

    return {
        "declared_paths": len(declared_paths),
        "quarantined_paths": len(quarantined),
        "missing_required_paths": len(missing_required),
        "missing_declared_files": len(missing_declared),
        "one_bundle_clock": one_bundle_clock,
    }


def _audit_player_rating_release_contract(
    root: Path,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    """Require honest player-outcome denominators and no rank claim."""

    meta_path = root / "features" / "player_ratings_meta.json"
    snapshot_path = (
        root / "features" / "player_ratings_snapshot.parquet"
    )
    if not meta_path.exists() and not snapshot_path.exists():
        return {}
    if not meta_path.is_file() or not snapshot_path.is_file():
        _add(
            findings,
            severity="launch blocker",
            code="player_rating_contract_incomplete",
            grain="player-rating release",
            evidence=(
                "Player outcome snapshot and metadata must be published "
                "together or withheld together."
            ),
            count=1,
        )
        return {}
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        snapshot = pq.read_table(snapshot_path).to_pandas()
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _add(
            findings,
            severity="launch blocker",
            code="player_rating_contract_unreadable",
            grain="player-rating release",
            evidence=f"Player outcome artifacts are unreadable: {type(exc).__name__}.",
            count=1,
        )
        return {}

    identity = meta.get("identity_audit") or {}
    input_maps = int(meta.get("n_input_maps") or -1)
    eligible_maps = int(meta.get("n_identity_eligible_maps") or -1)
    quarantined_maps = int(identity.get("n_quarantined_maps") or 0)
    valid_maps = int(identity.get("n_valid_maps") or -1)
    players = int(meta.get("n_players") or -1)
    unique_exposure = int(
        meta.get("n_unique_outcome_exposure_players") or 0
    )
    shared_exposure = int(
        meta.get("n_shared_outcome_history_players") or 0
    )
    declared_rate = pd.to_numeric(
        pd.Series([meta.get("identity_eligible_map_rate")]),
        errors="coerce",
    ).iloc[0]
    expected_rate = eligible_maps / input_maps if input_maps > 0 else -1.0
    examples = identity.get("quarantined_game_uid_examples")
    collision_examples = identity.get("display_name_collision_examples")
    collision_count = identity.get("n_display_name_collisions")
    contract_ok = bool(
        meta.get("outcome_ordering_verified") is False
        and meta.get("individual_skill_estimand") is False
        and input_maps > 0
        and int(meta.get("n_maps") or -1) == input_maps
        and 0 <= eligible_maps <= input_maps
        and valid_maps == eligible_maps
        and quarantined_maps == input_maps - eligible_maps
        and pd.notna(declared_rate)
        and abs(float(declared_rate) - expected_rate) <= 1e-12
        and players == len(snapshot)
        and unique_exposure + shared_exposure == players
        and "quarantined_game_uids" not in identity
        and isinstance(examples, list)
        and len(examples) <= 20
        and "display_name_collisions" not in identity
        and isinstance(collision_count, int)
        and collision_count >= 0
        and isinstance(collision_examples, list)
        and len(collision_examples) <= 20
        and all(
            isinstance(name, str) and name
            for name in collision_examples
        )
    )
    if not contract_ok:
        _add(
            findings,
            severity="launch blocker",
            code="player_rating_claim_or_denominator_invalid",
            grain="player-rating release",
            evidence=(
                "The shared team-outcome surface must withhold individual "
                "ordering, deny an individual-skill estimand, reconcile input/"
                "eligible/quarantined map counts, and publish only bounded "
                "quarantine and alias-collision examples without provider-ID "
                "maps."
            ),
            count=1,
        )
    weekly_rank_path = root / "features" / "player_weekly_ranks.json"
    if (
        meta.get("outcome_ordering_verified") is False
        and weekly_rank_path.exists()
    ):
        _add(
            findings,
            severity="launch blocker",
            code="player_rank_artifact_published_without_ordering_gate",
            grain="player-ranking artifact",
            evidence=(
                "Weekly rank movement is published even though the player "
                "outcome model explicitly withholds individual ordering."
            ),
            count=1,
            examples=["features/player_weekly_ranks.json"],
        )
    return {
        "snapshot_rows": int(len(snapshot)),
        "input_maps": input_maps,
        "identity_eligible_maps": eligible_maps,
        "identity_eligible_map_rate": (
            float(declared_rate) if pd.notna(declared_rate) else None
        ),
        "quarantined_maps": quarantined_maps,
        "unique_outcome_exposure_players": unique_exposure,
        "shared_outcome_history_players": shared_exposure,
        "outcome_ordering_verified": meta.get(
            "outcome_ordering_verified"
        ),
        "individual_skill_estimand": meta.get(
            "individual_skill_estimand"
        ),
        "contract_ok": contract_ok,
    }


def _audit_source_summary(
    root: Path,
    manifest: dict[str, Any],
    maps: pd.DataFrame,
    findings: list[dict[str, Any]],
) -> dict[str, Any]:
    path = root / "meta" / "source_summary.json"
    if not path.is_file():
        return {"status": "missing"}
    try:
        file_summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        _add(
            findings,
            severity="launch blocker",
            code="source_summary_invalid",
            grain="source-provenance artifact",
            evidence="The public source summary is not valid JSON.",
            count=1,
        )
        return {"status": "invalid"}
    manifest_summary = manifest.get("source_summary")
    if manifest_summary != file_summary:
        _add(
            findings,
            severity="launch blocker",
            code="source_summary_manifest_mismatch",
            grain="source-provenance artifact",
            evidence=(
                "The manifest and downloadable source summary do not declare "
                "the same provenance contract."
            ),
            count=1,
        )
    sources = file_summary.get("sources") or {}
    canonical = sources.get("canonical_map_inclusion") or {}
    detail = sources.get("map_detail_enrichment") or {}
    expected_canonical = (
        maps["canonical_map_source"]
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict()
        if "canonical_map_source" in maps.columns
        else {}
    )
    expected_detail = (
        maps["map_detail_source"]
        .fillna("")
        .astype(str)
        .value_counts()
        .to_dict()
        if "map_detail_source" in maps.columns
        else {}
    )

    def declared_counts(block: Any) -> dict[str, int]:
        if not isinstance(block, dict):
            return {}
        output: dict[str, int] = {}
        for source, values in block.items():
            if isinstance(values, dict):
                try:
                    output[str(source)] = int(values.get("maps"))
                except (TypeError, ValueError):
                    continue
        return output

    declared_canonical = declared_counts(canonical)
    declared_detail = declared_counts(detail)
    contract_ok = bool(
        file_summary.get("schema_version") == 2
        and declared_canonical == expected_canonical
        and declared_detail == expected_detail
        and file_summary.get("attribution") == manifest.get("attribution")
    )
    if not contract_ok:
        _add(
            findings,
            severity="launch blocker",
            code="source_summary_counts_or_roles_invalid",
            grain="source-provenance artifact",
            evidence=(
                "Canonical map inclusion and detail enrichment must be "
                "separate, exhaustive, and reconcile exactly to public maps."
            ),
            count=1,
        )
    return {
        "status": "verified" if contract_ok else "invalid",
        "canonical_map_inclusion": declared_canonical,
        "map_detail_enrichment": declared_detail,
    }


def _audit_current_membership(
    root: Path,
    manifest: dict[str, Any],
    findings: list[dict[str, Any]],
    *,
    now_utc: pd.Timestamp,
) -> dict[str, Any]:
    tournaments = manifest.get("current_tournaments") or {}
    registry = manifest.get("membership_registry") or {}
    complete_pack = isinstance(manifest.get("files"), list) and bool(
        manifest.get("files")
    )
    if complete_pack and (
        not isinstance(registry, dict)
        or not registry.get("snapshot_id")
        or not registry.get("authority")
    ):
        _add(
            findings,
            severity="launch blocker",
            code="current_membership_registry_missing",
            grain="membership-registry snapshot",
            evidence=(
                "A complete public pack must carry an authoritative, versioned "
                "current-tournament membership registry."
            ),
            count=1,
        )
    checked_at = pd.to_datetime(
        registry.get("checked_at"), errors="coerce", utc=True
    )
    review_due_at = pd.to_datetime(
        registry.get("review_due_at"), errors="coerce", utc=True
    )
    if registry and (
        pd.isna(checked_at)
        or pd.isna(review_due_at)
        or checked_at > review_due_at
        or review_due_at <= now_utc
    ):
        _add(
            findings,
            severity="launch blocker",
            code="current_membership_registry_stale",
            grain="membership-registry snapshot",
            evidence=(
                "The current-membership registry lacks valid review timestamps, "
                "was checked after its deadline, or its review deadline has "
                "expired relative to current UTC."
            ),
            count=1,
            examples=[
                f"review_due_at={review_due_at}",
                f"current_utc={now_utc}",
            ],
        )
    freshness_summary = {
        "registry_snapshot_id": registry.get("snapshot_id"),
        "registry_review_due_at": (
            None if pd.isna(review_due_at) else review_due_at.isoformat()
        ),
        "audit_current_utc": now_utc.isoformat(),
    }
    if not isinstance(tournaments, dict) or not tournaments:
        return {"current_tournaments": {}, **freshness_summary}
    records_path = root / "features" / "team_records.json"
    if not records_path.exists():
        return {
            "current_tournaments": tournaments,
            "checked_team_records": 0,
            "mismatches": 0,
            **freshness_summary,
        }
    records = json.loads(records_path.read_text(encoding="utf-8"))
    participants_by_league = registry.get("participants_by_league") or {}
    participant_pairs = {
        (str(league), str(team_key))
        for league, team_keys in participants_by_league.items()
        for team_key in team_keys
    }
    registered_team_keys = {team_key for _, team_key in participant_pairs}
    mismatches: list[str] = []
    checked = 0
    represented_team_keys: set[str] = set()
    for display, record in records.items():
        team_key = str(record.get("team_key") or team_identity_key(display))
        represented_team_keys.add(team_key)
        league = record.get("current_league")
        if participants_by_league:
            if team_key in registered_team_keys:
                checked += 1
                if (str(league), team_key) not in participant_pairs:
                    mismatches.append(str(display))
                    continue
            elif league is not None:
                checked += 1
                mismatches.append(str(display))
                continue
        expected = tournaments.get(str(league)) if league is not None else None
        if expected is None:
            continue
        if not participants_by_league:
            checked += 1
        if record.get("current_tournament") != expected:
            mismatches.append(str(display))
    if mismatches:
        _add(
            findings,
            severity="launch blocker",
            code="current_tournament_membership_mismatch",
            grain="team-record/league-tournament",
            evidence="A team record disagrees with the pack's authoritative current tournament participant registry.",
            count=len(mismatches),
            examples=mismatches,
        )

    missing_records = sorted(registered_team_keys - represented_team_keys)
    if missing_records:
        _add(
            findings,
            severity="major",
            code="current_member_missing_team_record",
            grain="current-tournament participant/team-record",
            evidence="A current Riot Tier 1 tournament participant has no public team record in this pack.",
            count=len(missing_records),
            examples=missing_records,
        )

    return {
        "current_tournaments": tournaments,
        "current_tournament_as_of": manifest.get("current_tournament_as_of"),
        "checked_team_records": checked,
        "mismatches": len(mismatches),
        "missing_current_member_records": len(missing_records),
        **freshness_summary,
    }


def audit_pack(
    root: Path,
    *,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Return a machine-readable launch audit for ``root``."""

    root = Path(root)
    now_utc = _current_utc(clock)
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
    manifest_contract = _audit_manifest_release_contract(
        root,
        manifest,
        findings,
    )
    map_summary = _audit_maps(maps, findings)
    source_summary = _audit_source_summary(
        root,
        manifest,
        maps,
        findings,
    )
    series_summary = _audit_series(maps, findings)
    side_summary = _audit_side_grains(root, maps, findings)
    year_summary = _audit_declared_years(root, manifest, maps, findings)
    history_summary = _audit_history(root, maps, findings)
    ratings_summary = _audit_rating_release_contract(
        root,
        manifest,
        findings,
    )
    player_ratings_summary = _audit_player_rating_release_contract(
        root,
        findings,
    )
    membership_summary = _audit_current_membership(
        root,
        manifest,
        findings,
        now_utc=now_utc,
    )

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
    gate = release_gate(counts)
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
        "source_provenance": source_summary,
        "series": series_summary,
        "side_grains": side_summary,
        "years": year_summary,
        "history": history_summary,
        "ratings": ratings_summary,
        "player_ratings": player_ratings_summary,
        "membership": membership_summary,
        "manifest_contract": manifest_contract,
        "findings": findings,
        "release_gate": gate,
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
        print(f"release_ready={report['release_gate']['ready']}")
        for finding in report["findings"]:
            print(f"[{finding['severity']}] {finding['code']} ({finding['count']} {finding['grain']}): {finding['evidence']}")
    return 0 if report["release_gate"]["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
