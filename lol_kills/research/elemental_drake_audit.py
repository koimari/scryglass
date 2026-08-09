#!/usr/bin/env python3
"""Audit the compact GRID elemental-drake cohort before public modeling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from lol_kills.draft_archetypes import champ_tags
from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.grid_ingest import RAW_GRID_DIR
from lol_kills.research.elemental_drakes import (
    COMPACT_EVENTS_PARQUET,
    COMPACT_GAMES_PARQUET,
    MECHANICS,
    _reconcile_pilot_names,
    parse_normalized_cohort,
    parse_raw_pilot,
)
from lol_kills.research.elemental_drake_explorer_model import (
    _capture_path_is_legal,
    _side_assignment_is_valid,
)

MIN_PUBLIC_GAMES = 6_000
ELEMENTS = {row["id"] for row in MECHANICS}


def _json_length(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return 0
    return len(parsed) if isinstance(parsed, list) else 0


def _json_values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed] if isinstance(parsed, list) else []


def audit_compact_cohort(
    games_path: Path = COMPACT_GAMES_PARQUET,
    events_path: Path = COMPACT_EVENTS_PARQUET,
    *,
    required_games: int = MIN_PUBLIC_GAMES,
) -> dict[str, Any]:
    games = pd.read_parquet(games_path)
    events = pd.read_parquet(events_path)
    errors: list[str] = []
    warnings: list[str] = []

    game_key = ["series_id", "game_id"]
    event_key = [*game_key, "global_index"]
    duplicate_games = int(games.duplicated(game_key).sum())
    duplicate_events = int(events.duplicated(event_key).sum())
    if duplicate_games:
        errors.append(f"{duplicate_games} duplicate game keys")
    if duplicate_events:
        errors.append(f"{duplicate_events} duplicate drake-event keys")

    team_ids_distinct = games["team_1_id"].astype(str) != games["team_2_id"].astype(str)
    winner_known = (
        (games["winner_team_id"].astype(str) == games["team_1_id"].astype(str))
        | (games["winner_team_id"].astype(str) == games["team_2_id"].astype(str))
    )
    completed = games["complete"].astype(bool) & winner_known & team_ids_distinct
    games = games.assign(_eligible=completed)
    eligible_games = games.loc[games["_eligible"]]
    completed_games = int(games.loc[completed, game_key].drop_duplicates().shape[0])
    if completed_games < required_games:
        errors.append(
            f"{completed_games} completed games; {required_games} required"
        )

    composition_complete = (
        eligible_games["team_1_champions"].map(_json_length).eq(5)
        & eligible_games["team_2_champions"].map(_json_length).eq(5)
    )
    roster_complete = (
        eligible_games["team_1_player_ids"].map(_json_length).eq(5)
        & eligible_games["team_2_player_ids"].map(_json_length).eq(5)
    )
    champions = [
        champion
        for column in ("team_1_champions", "team_2_champions")
        for value in eligible_games[column]
        for champion in _json_values(value)
    ]
    tagged_champions = sum(
        bool(champ_tags(normalize_champ(champion))) for champion in champions
    )
    champion_tag_coverage = (
        tagged_champions / len(champions) if champions else 0.0
    )
    patch_present = (
        eligible_games["patch"]
        .astype(str)
        .str.fullmatch(r"\d+\.\d+")
        .fillna(False)
    )

    joined = events.merge(
        games[
            [
                *game_key,
                "team_1_id",
                "team_1_side",
                "team_2_id",
                "team_2_side",
                "complete",
                "_eligible",
            ]
        ],
        on=game_key,
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    owner_known = (
        (joined["owner_team_id"].astype(str) == joined["team_1_id"].astype(str))
        | (joined["owner_team_id"].astype(str) == joined["team_2_id"].astype(str))
    )
    expected_side = joined["team_1_side"].where(
        joined["owner_team_id"].astype(str) == joined["team_1_id"].astype(str),
        joined["team_2_side"],
    )
    side_matches = (
        joined["owner_side"].astype(str).str.lower()
        == expected_side.astype(str).str.lower()
    )
    state_valid = (
        joined["state_timing"].eq("previous-envelope")
        & joined["state_lag_seconds"].between(0, 60)
        & joined["owner_net_worth"].gt(0)
        & joined["opponent_net_worth"].gt(0)
    )
    element_valid = joined["element"].isin(ELEMENTS)

    if not joined["_merge"].eq("both").all():
        errors.append("some drake events do not join to a compact game")
    if not owner_known.all():
        errors.append("some drake owners do not match either game team")
    if not side_matches.all():
        errors.append("some drake-owner sides disagree with the game team mapping")
    if not element_valid.all():
        errors.append("unknown elemental-drake labels are present")

    first = joined.loc[joined["global_index"].eq(1)].copy()
    valid_first = first.loc[
        first["_eligible"].fillna(False).astype(bool)
        & owner_known.loc[first.index]
        & state_valid.loc[first.index]
        & element_valid.loc[first.index]
    ]
    valid_first_games = int(valid_first[game_key].drop_duplicates().shape[0])
    minimum_first = int(required_games * 0.8)
    if valid_first_games < minimum_first:
        errors.append(
            f"{valid_first_games} validated first-drake rows; "
            f"{minimum_first} required"
        )

    first_distribution = {
        element: int((valid_first["element"] == element).sum())
        for element in sorted(ELEMENTS)
    }
    sparse_elements = [
        element for element, count in first_distribution.items() if count < 200
    ]
    if sparse_elements:
        warnings.append(
            "fewer than 200 validated first captures for: "
            + ", ".join(sparse_elements)
        )

    patch_coverage = float(patch_present.mean()) if len(games) else 0.0
    composition_coverage = (
        float(composition_complete.mean()) if len(games) else 0.0
    )
    roster_coverage = float(roster_complete.mean()) if len(games) else 0.0
    eligible_event_mask = joined["_eligible"].fillna(False).astype(bool)
    state_coverage = (
        float(state_valid.loc[eligible_event_mask].mean())
        if eligible_event_mask.any()
        else 0.0
    )
    if patch_coverage < 0.95:
        errors.append(f"exact patch coverage is only {patch_coverage:.1%}")
    if composition_coverage < 0.95:
        errors.append(
            f"complete 5v5 composition coverage is only {composition_coverage:.1%}"
        )
    if champion_tag_coverage < 0.95:
        errors.append(
            f"champion archetype coverage is only {champion_tag_coverage:.1%}"
        )
    if state_coverage < 0.95:
        errors.append(
            f"validated pre-capture state coverage is only {state_coverage:.1%}"
        )
    if roster_coverage < 0.95:
        warnings.append(
            f"complete five-player roster coverage is {roster_coverage:.1%}"
        )

    invalid_side_games = 0
    illegal_path_games = 0
    excluded_capture_rows = 0
    eligible_lookup = {
        (str(row.series_id), str(row.game_id)): row._asdict()
        for row in eligible_games.itertuples(index=False)
    }
    event_groups = {
        (str(series_id), str(game_id)): group
        for (series_id, game_id), group in events.groupby(
            ["series_id", "game_id"],
            sort=False,
        )
    }
    for key, game in eligible_lookup.items():
        game_events = event_groups.get(key, events.iloc[0:0]).sort_values(
            [
                column
                for column in ("global_index", "time_seconds", "occurred_at")
                if column in events.columns
            ]
        )
        if not _side_assignment_is_valid(game):
            invalid_side_games += 1
            excluded_capture_rows += len(game_events)
            continue
        team_ids = (
            str(game.get("team_1_id") or ""),
            str(game.get("team_2_id") or ""),
        )
        if not _capture_path_is_legal(game_events, team_ids):
            illegal_path_games += 1
            excluded_capture_rows += len(game_events)
    if invalid_side_games or illegal_path_games:
        warnings.append(
            "model eligibility excludes "
            f"{invalid_side_games} games with invalid side assignments and "
            f"{illegal_path_games} games with incomplete or illegal elemental paths"
        )

    competition_coverage: dict[str, Any] = {"status": "not-available"}
    competition_columns = {"region", "league", "competition_level"}
    if competition_columns.issubset(eligible_games.columns):
        unclassified = eligible_games["region"].astype(str).eq("other")
        tier_one = eligible_games["competition_level"].astype(str).eq("tier1")
        tier_one_regions = (
            eligible_games.loc[tier_one]
            .groupby("region")
            .size()
            .astype(int)
            .to_dict()
        )
        required_tier_one_regions = {
            "china",
            "emea",
            "korea",
            "north-america",
            "pacific",
            "south-america",
        }
        missing_tier_one_regions = sorted(
            required_tier_one_regions - set(tier_one_regions)
        )
        if unclassified.any():
            errors.append(
                f"{int(unclassified.sum())} eligible games lack a classified region"
            )
        if missing_tier_one_regions:
            errors.append(
                "Tier 1 coverage is absent for: "
                + ", ".join(missing_tier_one_regions)
            )
        competition_coverage = {
            "status": (
                "pass"
                if not unclassified.any() and not missing_tier_one_regions
                else "fail"
            ),
            "tierOneGames": int(tier_one.sum()),
            "internationalGames": int(
                eligible_games["competition_level"]
                .astype(str)
                .eq("international")
                .sum()
            ),
            "otherProGames": int(
                eligible_games["competition_level"]
                .astype(str)
                .eq("other-pro")
                .sum()
            ),
            "unclassifiedGames": int(unclassified.sum()),
            "tierOneByRegion": tier_one_regions,
        }

    return {
        "status": "pass" if not errors else "fail",
        "requiredGames": required_games,
        "games": {
            "rows": len(games),
            "completed": completed_games,
            "series": int(games["series_id"].nunique()),
            "duplicateKeys": duplicate_games,
        },
        "events": {
            "rows": len(events),
            "validatedFirstDrakes": valid_first_games,
            "duplicateKeys": duplicate_events,
            "firstElementDistribution": first_distribution,
            "medianStateLagSeconds": (
                round(float(valid_first["state_lag_seconds"].median()), 3)
                if len(valid_first)
                else None
            ),
        },
        "coverage": {
            "exactPatch": round(patch_coverage, 4),
            "completeComposition": round(composition_coverage, 4),
            "championArchetypes": round(champion_tag_coverage, 4),
            "completeRoster": round(roster_coverage, 4),
            "validatedPreCaptureState": round(state_coverage, 4),
        },
        "modelEligibility": {
            "invalidSideAssignmentGames": invalid_side_games,
            "illegalOrPartialCapturePathGames": illegal_path_games,
            "excludedCaptureRows": excluded_capture_rows,
            "policy": (
                "Exclude the complete game from joint modeling rather than "
                "repairing provider side or capture-path state."
            ),
        },
        "competitionCoverage": competition_coverage,
        "storageBytes": {
            "gamesParquet": games_path.stat().st_size,
            "eventsParquet": events_path.stat().st_size,
        },
        "errors": errors,
        "warnings": warnings,
    }


def reconcile_raw_pilot(raw_dir: Path = RAW_GRID_DIR) -> dict[str, Any]:
    """Cross-check compact GRID events against independent Riot event files."""
    raw_games = parse_raw_pilot(raw_dir)
    normalized_games = parse_normalized_cohort(raw_dir)
    _reconcile_pilot_names(raw_games, normalized_games)
    normalized_by_series: dict[str, list[dict[str, Any]]] = {}
    for game in normalized_games:
        normalized_by_series.setdefault(str(game.get("seriesId") or ""), []).append(game)

    rows = []
    for raw in raw_games:
        series_id = str(raw.get("seriesId") or "")
        candidates = normalized_by_series.get(series_id, [])
        if not candidates:
            continue
        raw_events = list(raw.get("dragonEvents") or [])

        def candidate_score(candidate: dict[str, Any]) -> tuple[int, int]:
            candidate_events = list(candidate.get("dragonEvents") or [])
            pairs = zip(raw_events, candidate_events)
            element_matches = sum(
                left.get("element") == right.get("element")
                for left, right in pairs
            )
            return element_matches, -abs(len(raw_events) - len(candidate_events))

        normalized = max(candidates, key=candidate_score)
        normalized_events = list(normalized.get("dragonEvents") or [])
        team_names = {
            str(team.get("id") or ""): str(team.get("name") or "")
            for team in normalized.get("teams") or []
        }
        event_rows = []
        for left, right in zip(raw_events, normalized_events):
            event_rows.append(
                {
                    "element": left.get("element") == right.get("element"),
                    "timeWithinTwoSeconds": abs(
                        int(left.get("timeSeconds") or 0)
                        - int(right.get("timeSeconds") or 0)
                    )
                    <= 2,
                    "owner": str(left.get("teamName") or "")
                    == team_names.get(str(right.get("ownerTeamId") or ""), ""),
                }
            )
        rows.append(
            {
                "seriesId": series_id,
                "rawEvents": len(raw_events),
                "normalizedEvents": len(normalized_events),
                "events": event_rows,
            }
        )

    compared = [event for row in rows for event in row["events"]]
    raw_series = len(raw_games)
    compared_series = len(rows)
    exact = bool(compared) and all(
        event["element"] and event["timeWithinTwoSeconds"] and event["owner"]
        for event in compared
    )
    return {
        "status": (
            "pass"
            if exact and compared_series == raw_series
            else "partial"
            if exact
            else "fail"
        ),
        "rawSeries": raw_series,
        "comparedSeries": compared_series,
        "eventsCompared": len(compared),
        "elementMatches": sum(event["element"] for event in compared),
        "ownerMatches": sum(event["owner"] for event in compared),
        "timeWithinTwoSeconds": sum(
            event["timeWithinTwoSeconds"] for event in compared
        ),
        "series": rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, default=COMPACT_GAMES_PARQUET)
    parser.add_argument("--events", type=Path, default=COMPACT_EVENTS_PARQUET)
    parser.add_argument("--required-games", type=int, default=MIN_PUBLIC_GAMES)
    parser.add_argument("--reconcile-raw", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = audit_compact_cohort(
        args.games,
        args.events,
        required_games=args.required_games,
    )
    if args.reconcile_raw:
        report["rawReconciliation"] = reconcile_raw_pilot()
        if report["rawReconciliation"]["status"] == "fail":
            report["errors"].append(
                "raw Riot and normalized GRID pilot events did not reconcile"
            )
            report["status"] = "fail"
        elif report["rawReconciliation"]["status"] == "partial":
            report["warnings"].append(
                "some raw Riot pilot series were not yet present in the compact cohort"
            )
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
