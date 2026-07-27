"""Authoritative scheduled-series formats from Leaguepedia MatchSchedule.

The schedule is joined by Riot platform game ID.  An exact platform ID proves
the scheduled match identity and format; the independently normalized team
pair is retained as a reconciliation check, not used to redefine organization
identity.  Date conflicts, duplicate platform IDs, and unmatched games remain
unverified.  Final scores are never used to infer Bo1/Bo3/Bo5.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.paths import PARQUET_DIR


SCHEDULE_CACHE = PARQUET_DIR / "leaguepedia_match_schedule.parquet"
SCHEDULE_SOURCE = "Leaguepedia MatchSchedule/MatchScheduleGame"
SCHEDULE_URL = "https://lol.fandom.com/wiki/Special:CargoExport"
SCHEDULE_FIELDS = (
    "MSG.RiotPlatformGameId=riot_game_id,"
    "MSG.GameId=leaguepedia_game_id,"
    "MSG.N_GameInMatch=game_index,"
    "MS.MatchId=match_id,"
    "MS.BestOf=best_of,"
    "MS.DateTime_UTC=scheduled_at,"
    "MS.Team1=team1,"
    "MS.Team2=team2,"
    "MS.OverviewPage=overview_page"
)
VALID_SCHEDULE_LENGTHS = frozenset({1, 2, 3, 5})
KNOCKOUT_BEST_OF = frozenset({1, 3, 5})
MAX_SCHEDULE_DATE_DELTA_HOURS = 48.0


@dataclass(frozen=True)
class ScheduleAnnotation:
    rows: pd.DataFrame
    audit: dict[str, Any]


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def validate_schedule(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a unique, typed platform-game schedule or raise."""

    required = {
        "riot_game_id",
        "leaguepedia_game_id",
        "game_index",
        "match_id",
        "best_of",
        "scheduled_at",
        "team1",
        "team2",
        "overview_page",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Leaguepedia schedule missing columns: {missing}")
    out = frame.copy()
    for column in (
        "riot_game_id",
        "leaguepedia_game_id",
        "match_id",
        "team1",
        "team2",
        "overview_page",
    ):
        out[column] = out[column].map(_clean)
    out["game_index"] = pd.to_numeric(
        out["game_index"], errors="coerce"
    ).astype("Int64")
    out["best_of"] = pd.to_numeric(
        out["best_of"], errors="coerce"
    ).astype("Int64")
    out["scheduled_at"] = pd.to_datetime(
        out["scheduled_at"], errors="coerce", utc=True
    )
    out = out[
        out["riot_game_id"].ne("")
        & out["match_id"].ne("")
        & out["game_index"].gt(0)
        & out["best_of"].isin(VALID_SCHEDULE_LENGTHS)
        & out["team1"].ne("")
        & out["team2"].ne("")
    ].copy()
    comparison_columns = [
        "leaguepedia_game_id",
        "game_index",
        "match_id",
        "best_of",
        "team1",
        "team2",
        "overview_page",
    ]
    conflicting = (
        out.groupby("riot_game_id")[comparison_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if conflicting.any():
        examples = conflicting[conflicting].index.astype(str).tolist()[:10]
        raise ValueError(
            "Leaguepedia schedule has conflicting Riot platform game IDs: "
            f"{examples}"
        )
    return (
        out.sort_values(
            ["scheduled_at", "match_id", "game_index"],
            kind="mergesort",
        )
        .drop_duplicates("riot_game_id", keep="last")
        .reset_index(drop=True)
    )


def fetch_schedule(
    *,
    start: str,
    end: str,
    cache_path: Path = SCHEDULE_CACHE,
    page_size: int = 5_000,
) -> pd.DataFrame:
    """Download and cache the authoritative game-to-match schedule."""

    rows: list[dict[str, Any]] = []
    for offset in range(0, 100_000, page_size):
        params = {
            "tables": "MatchScheduleGame=MSG,MatchSchedule=MS",
            "fields": SCHEDULE_FIELDS,
            "join_on": "MSG.MatchId=MS.MatchId",
            "where": (
                f'MS.DateTime_UTC >= "{start}" '
                f'AND MS.DateTime_UTC < "{end}" '
                "AND MSG.RiotPlatformGameId IS NOT NULL"
            ),
            "order_by": "MS.DateTime_UTC ASC",
            "limit": str(page_size),
            "offset": str(offset),
            "format": "json",
        }
        request = urllib.request.Request(
            f"{SCHEDULE_URL}?{urllib.parse.urlencode(params)}",
            headers={"User-Agent": "Scryglass public data audit/1.0"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            page = json.load(response)
        if not isinstance(page, list):
            raise RuntimeError("Leaguepedia schedule response is not a row list")
        rows.extend(row for row in page if isinstance(row, dict))
        if len(page) < page_size:
            break
    else:
        raise RuntimeError("Leaguepedia schedule pagination exceeded safety cap")
    schedule = validate_schedule(pd.DataFrame(rows))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(cache_path, index=False)
    return schedule


def load_schedule(cache_path: Path = SCHEDULE_CACHE) -> pd.DataFrame:
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Missing authoritative series schedule cache: {cache_path}"
        )
    return validate_schedule(pd.read_parquet(cache_path))


def annotate_scheduled_series(
    frame: pd.DataFrame,
    schedule: pd.DataFrame,
    *,
    max_date_delta_hours: float = MAX_SCHEDULE_DATE_DELTA_HOURS,
) -> ScheduleAnnotation:
    """Attach scheduled match identity and format by exact Riot game ID.

    A schedule/team-name mismatch is exposed as provenance but does not negate
    the exact game-ID match.  This matters for historical rebrands and
    disambiguated Leaguepedia names.  The schedule names are never copied into
    canonical team identity fields.
    """

    if frame is None or frame.empty:
        return ScheduleAnnotation(
            pd.DataFrame() if frame is None else frame.copy(),
            {"rows": 0, "matched_games": 0, "team_conflicts": 0},
        )
    schedule = validate_schedule(schedule)
    out = frame.copy()
    game_column = next(
        (
            column
            for column in ("gameid", "game_uid", "oe_gameid")
            if column in out.columns
        ),
        None,
    )
    if game_column is None:
        raise ValueError("Schedule annotation requires a canonical game ID")
    out["_schedule_game_id"] = out[game_column].map(_clean)

    blue_column = next(
        (
            column
            for column in ("blue_team", "blue_teamname")
            if column in out.columns
        ),
        None,
    )
    red_column = next(
        (
            column
            for column in ("red_team", "red_teamname")
            if column in out.columns
        ),
        None,
    )
    if blue_column and red_column:
        pairs = {
            game_id: frozenset(
                {
                    normalize_team(_clean(group.iloc[0][blue_column])),
                    normalize_team(_clean(group.iloc[0][red_column])),
                }
            )
            for game_id, group in out.groupby("_schedule_game_id", sort=False)
        }
    elif "teamname" in out.columns:
        pairs = {
            game_id: frozenset(
                normalize_team(_clean(value))
                for value in group["teamname"]
                if _clean(value)
            )
            for game_id, group in out.groupby("_schedule_game_id", sort=False)
        }
    else:
        raise ValueError("Schedule annotation requires map or side team names")

    schedule_rows = schedule.set_index("riot_game_id").to_dict("index")
    matched_games: set[str] = set()
    team_pair_matches: set[str] = set()
    team_conflicts: set[str] = set()
    date_conflicts: set[str] = set()
    fixed_game_series: set[str] = set()
    previous_schedule_rows = (
        out.get(
            "series_format_source",
            pd.Series(pd.NA, index=out.index, dtype=object),
        )
        .map(_clean)
        .eq(SCHEDULE_SOURCE)
    )
    # Remove only prior annotations produced by this source. Other explicit
    # source series IDs (notably GRID) remain untouched.
    for column in ("source_series_id", "series_format"):
        if column in out.columns:
            out.loc[previous_schedule_rows, column] = pd.NA
    for column in (
        "leaguepedia_match_id",
        "leaguepedia_game_id",
        "leaguepedia_game_index",
        "leaguepedia_best_of",
        "leaguepedia_overview_page",
        "leaguepedia_scheduled_at",
        "leaguepedia_team1",
        "leaguepedia_team2",
        "series_schedule_team_pair_status",
        "series_schedule_date_status",
        "series_format_source",
    ):
        out[column] = pd.NA
    for column in ("source_series_id", "series_format"):
        if column not in out.columns:
            out[column] = pd.NA

    for game_id, indexes in out.groupby("_schedule_game_id").groups.items():
        schedule_row = schedule_rows.get(str(game_id))
        if schedule_row is None:
            continue
        schedule_time = pd.to_datetime(
            schedule_row["scheduled_at"], errors="coerce", utc=True
        )
        frame_times = (
            pd.to_datetime(out.loc[indexes, "date"], errors="coerce", utc=True)
            if "date" in out.columns
            else pd.Series(pd.NaT, index=indexes, dtype="datetime64[ns, UTC]")
        )
        valid_frame_times = frame_times.dropna()
        date_ok = (
            pd.isna(schedule_time)
            or valid_frame_times.empty
            or (
                (valid_frame_times - schedule_time)
                .dt.total_seconds()
                .abs()
                .le(max(max_date_delta_hours, 0.0) * 3600.0)
                .all()
            )
        )
        if not date_ok:
            date_conflicts.add(str(game_id))
            continue
        expected_pair = frozenset(
            {
                normalize_team(schedule_row["team1"]),
                normalize_team(schedule_row["team2"]),
            }
        )
        observed_pair = pairs.get(str(game_id), frozenset())
        pair_matches = len(observed_pair) == 2 and observed_pair == expected_pair
        if pair_matches:
            team_pair_matches.add(str(game_id))
            team_pair_status = "matched"
        else:
            team_conflicts.add(str(game_id))
            team_pair_status = "alias_or_identity_mismatch"
        matched_games.add(str(game_id))
        best_of = int(schedule_row["best_of"])
        values = {
            "source_series_id": schedule_row["match_id"],
            "leaguepedia_match_id": schedule_row["match_id"],
            "leaguepedia_game_id": schedule_row["leaguepedia_game_id"],
            "leaguepedia_game_index": int(schedule_row["game_index"]),
            "leaguepedia_best_of": best_of,
            "leaguepedia_overview_page": schedule_row["overview_page"],
            "leaguepedia_scheduled_at": schedule_time,
            "leaguepedia_team1": schedule_row["team1"],
            "leaguepedia_team2": schedule_row["team2"],
            "series_schedule_team_pair_status": team_pair_status,
            "series_schedule_date_status": "within_tolerance_or_unavailable",
            "series_format_source": SCHEDULE_SOURCE,
        }
        if best_of in KNOCKOUT_BEST_OF:
            values["series_format"] = f"Bo{best_of}"
        else:
            fixed_game_series.add(str(schedule_row["match_id"]))
        for column, value in values.items():
            out.loc[indexes, column] = value
        if "game" in out.columns:
            out.loc[indexes, "game"] = int(schedule_row["game_index"])

    out = out.drop(columns=["_schedule_game_id"])
    unique_games = out[game_column].map(_clean).nunique()
    return ScheduleAnnotation(
        out,
        {
            "source": SCHEDULE_SOURCE,
            "schedule_rows": int(len(schedule)),
            "rows": int(len(out)),
            "unique_games": int(unique_games),
            "matched_games": len(matched_games),
            "coverage_rate": (
                len(matched_games) / unique_games if unique_games else 0.0
            ),
            "team_pair_matches": len(team_pair_matches),
            "team_conflicts": len(team_conflicts),
            "team_conflict_examples": sorted(team_conflicts)[:10],
            "date_conflicts": len(date_conflicts),
            "date_conflict_examples": sorted(date_conflicts)[:10],
            "max_schedule_date_delta_hours": float(max_date_delta_hours),
            "fixed_game_series_quarantined": len(fixed_game_series),
        },
    )
