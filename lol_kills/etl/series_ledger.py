"""Canonical map-to-series identity with explicit completion provenance.

Clock buckets are never series identifiers.  Explicit source series IDs take
priority; otherwise source game-number resets define candidate series.  A
candidate is rating-eligible only when its map order and scheduled format prove
that it is a completed, decisive series.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from lol_kills.etl.competition import (
    canonicalize_competition_frame,
    team_identity_key,
)


FORMAT_RE = re.compile(r"(?:bo|best\s*of)\s*([135])", re.IGNORECASE)
COMPLETED_SOURCE_STATES = frozenset({"complete", "completed", "final", "verified"})
DERIVED_CANONICAL_COLUMNS = frozenset(
    {
        "canonical_series_id",
        "canonical_game_index",
        "canonical_series_status",
        "canonical_series_completion_source",
        "canonical_series_winner_team_key",
        "scheduled_best_of",
        "series_quarantine_reasons",
        "series_rating_eligible",
        "raw_source_game_index",
        "raw_source_game_uid",
    }
)


@dataclass(frozen=True)
class CanonicalSeriesResult:
    maps: pd.DataFrame
    series: pd.DataFrame
    audit: dict[str, Any]


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "null"} else text


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "\x1f".join(_clean(part) for part in parts).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(payload).hexdigest()[:24]}"


def _scheduled_best_of(value: Any) -> int | None:
    text = _clean(value)
    if not text:
        return None
    match = FORMAT_RE.search(text.replace("-", " "))
    if match:
        return int(match.group(1))
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number in {1, 3, 5} else None


def _first_available(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    values = pd.Series(pd.NA, index=frame.index, dtype=object)
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = frame[column]
        missing = values.isna() | values.astype(str).str.strip().isin(
            {"", "nan", "None", "<NA>"}
        )
        values = values.where(~missing, candidate)
    return values


def _prepare_maps(maps: pd.DataFrame) -> pd.DataFrame:
    frame = canonicalize_competition_frame(maps).copy()
    frame["_input_order"] = np.arange(len(frame))
    frame["date"] = pd.to_datetime(
        frame.get("date"),
        errors="coerce",
        utc=True,
        format="mixed",
    )
    frame["y_blue_win"] = pd.to_numeric(frame.get("y_blue_win"), errors="coerce")
    frame["_source"] = _first_available(
        frame, ("source", "source_name")
    ).map(_clean)
    explicit_grid = (
        frame.get("grid_series_id", pd.Series(pd.NA, index=frame.index))
        .map(_clean)
        .ne("")
    )
    frame.loc[frame["_source"].eq("") & explicit_grid, "_source"] = "grid"
    frame.loc[frame["_source"].eq(""), "_source"] = "oe"
    frame["_source_series_id"] = _first_available(
        frame,
        (
            "source_series_id",
            "grid_series_id",
            "series_id",
        ),
    ).map(_clean)
    frame["_source_game_index"] = pd.to_numeric(
        _first_available(frame, ("grid_game_index", "game", "game_index")),
        errors="coerce",
    )
    frame["_source_game_index"] = frame["_source_game_index"].where(
        frame["_source_game_index"].isna(),
        frame["_source_game_index"].astype("Int64"),
    )
    frame["_source_completion"] = _first_available(
        frame,
        (
            "series_completion_status",
            "series_status",
            "completion_status",
        ),
    ).map(_clean)
    frame["_series_format"] = _first_available(
        frame, ("series_format", "best_of", "format")
    ).map(_clean)
    frame["_completion_source"] = _first_available(
        frame, ("series_completion_source", "completion_source")
    ).map(_clean)

    blue_key = frame.get(
        "blue_team_key",
        frame.get("blue_team", pd.Series("", index=frame.index)).map(
            team_identity_key
        ),
    ).astype(str)
    red_key = frame.get(
        "red_team_key",
        frame.get("red_team", pd.Series("", index=frame.index)).map(
            team_identity_key
        ),
    ).astype(str)
    frame["_team_a"] = [
        min(blue, red) for blue, red in zip(blue_key, red_key)
    ]
    frame["_team_b"] = [
        max(blue, red) for blue, red in zip(blue_key, red_key)
    ]

    source_uid = _first_available(
        frame, ("game_uid", "gameid", "oe_gameid", "grid_game_id")
    ).map(_clean)
    missing_uid = source_uid.eq("")
    if missing_uid.any():
        derived = [
            _stable_id(
                "map",
                source,
                date.isoformat() if pd.notna(date) else "",
                team_a,
                team_b,
                source_index,
            )
            for source, date, team_a, team_b, source_index in zip(
                frame.loc[missing_uid, "_source"],
                frame.loc[missing_uid, "date"],
                frame.loc[missing_uid, "_team_a"],
                frame.loc[missing_uid, "_team_b"],
                frame.loc[missing_uid, "_source_game_index"],
            )
        ]
        source_uid.loc[missing_uid] = derived
    frame["_source_game_uid"] = source_uid
    return frame


def _scope_key(row: pd.Series) -> tuple[str, ...]:
    return (
        str(row["_source"]),
        _clean(row.get("league")),
        _clean(row.get("tournament")),
        _clean(row.get("split")),
        _clean(row.get("playoffs")),
        str(row["_team_a"]),
        str(row["_team_b"]),
    )


def _assign_candidate_series(
    frame: pd.DataFrame,
    *,
    max_gap_hours: float,
) -> pd.Series:
    assignments = pd.Series("", index=frame.index, dtype=object)

    explicit = frame["_source_series_id"].ne("")
    for source_id, group in frame.loc[explicit].groupby(
        ["_source", "_source_series_id"], sort=True
    ):
        source, series_id = source_id
        assignments.loc[group.index] = _stable_id(
            "series", source, "explicit", series_id
        )

    derived = frame.loc[~explicit].copy()
    if derived.empty:
        return assignments
    derived["_scope_key"] = derived.apply(_scope_key, axis=1)
    for scope, group in derived.groupby("_scope_key", sort=True):
        ordered = group.sort_values(
            ["date", "_source_game_index", "_source_game_uid"],
            kind="mergesort",
            na_position="last",
        )
        segment = -1
        segment_anchor = ""
        previous_index: int | None = None
        previous_date: pd.Timestamp | None = None
        for index, row in ordered.iterrows():
            source_index = row["_source_game_index"]
            current_index = int(source_index) if pd.notna(source_index) else None
            current_date = row["date"] if pd.notna(row["date"]) else None
            gap_hours = (
                (current_date - previous_date).total_seconds() / 3600.0
                if current_date is not None and previous_date is not None
                else None
            )
            starts_new = (
                segment < 0
                or current_index is None
                or current_index == 1
                or (
                    gap_hours is not None
                    and gap_hours > max(max_gap_hours, 0.0)
                )
                or (
                    previous_index is not None
                    and current_index is not None
                    and current_index <= previous_index
                )
            )
            if starts_new:
                segment += 1
                segment_anchor = str(row["_source_game_uid"])
            assignments.loc[index] = _stable_id(
                "series",
                *scope,
                segment,
                segment_anchor,
            )
            previous_index = current_index
            previous_date = current_date
    return assignments


def _series_summary(
    series_id: str,
    group: pd.DataFrame,
) -> tuple[dict[str, Any], dict[Any, int | None]]:
    ordered = group.sort_values(
        ["_source_game_index", "date", "_source_game_uid"],
        kind="mergesort",
        na_position="last",
    )
    reasons: list[str] = []
    if ordered["date"].isna().any():
        reasons.append("invalid_date")
    if ordered["y_blue_win"].isna().any() or not ordered["y_blue_win"].isin(
        [0, 1]
    ).all():
        reasons.append("invalid_map_result")
    if ordered["_source_game_uid"].duplicated().any():
        reasons.append("duplicate_source_game_uid")
    if ordered[["_team_a", "_team_b"]].drop_duplicates().shape[0] != 1:
        reasons.append("mixed_team_pair")
    if ordered["league"].astype(str).nunique(dropna=False) != 1:
        reasons.append("mixed_competition")
    registry_conflict = ordered.get(
        "series_format_registry_conflict",
        pd.Series(False, index=ordered.index),
    ).fillna(False)
    if registry_conflict.astype(bool).any():
        reasons.append("series_format_registry_conflict")

    indices = pd.to_numeric(
        ordered["_source_game_index"], errors="coerce"
    )
    contiguous = False
    if indices.isna().any():
        reasons.append("missing_source_game_index")
    else:
        integer_indices = [int(value) for value in indices]
        if len(set(integer_indices)) != len(integer_indices):
            reasons.append("duplicate_source_game_index")
        expected = list(range(1, len(integer_indices) + 1))
        contiguous = integer_indices == expected
        if not contiguous:
            reasons.append("non_contiguous_source_game_index")

    formats = sorted(
        {
            value
            for value in ordered["_series_format"].map(_clean)
            if value
        }
    )
    if len(formats) > 1:
        reasons.append("conflicting_series_format")
    best_of = _scheduled_best_of(formats[0]) if len(formats) == 1 else None
    if best_of is None:
        reasons.append("unverified_series_format")

    team_a = str(ordered.iloc[0]["_team_a"])
    team_b = str(ordered.iloc[0]["_team_b"])
    blue_key = ordered.get(
        "blue_team_key",
        ordered.get("blue_team", pd.Series("", index=ordered.index)).map(
            team_identity_key
        ),
    ).astype(str)
    red_key = ordered.get(
        "red_team_key",
        ordered.get("red_team", pd.Series("", index=ordered.index)).map(
            team_identity_key
        ),
    ).astype(str)
    result = ordered["y_blue_win"].astype(float)
    team_a_wins = int(
        ((blue_key.eq(team_a)) & result.eq(1.0)).sum()
        + ((red_key.eq(team_a)) & result.eq(0.0)).sum()
    )
    team_b_wins = int(len(ordered) - team_a_wins)

    completion_status = "quarantined"
    winner_team_key: str | None = None
    if best_of is not None:
        wins_required = best_of // 2 + 1
        high = max(team_a_wins, team_b_wins)
        low = min(team_a_wins, team_b_wins)
        if len(ordered) > best_of or high > wins_required:
            reasons.append("score_exceeds_scheduled_format")
            completion_status = "invalid"
        elif high == wins_required and low < wins_required:
            completion_status = "completed"
            winner_team_key = team_a if team_a_wins > team_b_wins else team_b
        elif high < wins_required:
            completion_status = "incomplete"
            reasons.append("winner_below_format_threshold")
        else:
            completion_status = "invalid"
            reasons.append("impossible_series_score")
    elif not reasons:
        completion_status = "unverified"

    source_states = {
        value.casefold()
        for value in ordered["_source_completion"].map(_clean)
        if value
    }
    if source_states and len(source_states) > 1:
        reasons.append("conflicting_source_completion_state")
    source_claims_completed = bool(
        source_states.intersection(COMPLETED_SOURCE_STATES)
    )
    if source_states and not source_claims_completed:
        reasons.append("source_does_not_claim_completion")
        if completion_status == "completed":
            completion_status = "incomplete"
    if source_claims_completed and completion_status != "completed":
        reasons.append("source_completion_conflicts_with_series_validation")

    blocking_reasons = {
        reason
        for reason in reasons
        if reason
        not in {
            "unverified_series_format",
        }
    }
    rating_eligible = bool(
        completion_status == "completed"
        and contiguous
        and not blocking_reasons
    )
    canonical_indices: dict[Any, int | None] = {
        index: (
            int(source_index)
            if rating_eligible and pd.notna(source_index)
            else None
        )
        for index, source_index in zip(
            ordered.index, ordered["_source_game_index"]
        )
    }
    completion_sources = sorted(
        {
            value
            for value in ordered["_completion_source"].map(_clean)
            if value
        }
    )
    format_sources = sorted(
        {
            value
            for value in ordered.get(
                "series_format_source",
                pd.Series(pd.NA, index=ordered.index),
            ).map(_clean)
            if value
        }
    )
    return (
        {
            "canonical_series_id": series_id,
            "source": str(ordered.iloc[0]["_source"]),
            "source_series_id": (
                str(ordered.iloc[0]["_source_series_id"])
                if ordered["_source_series_id"].ne("").all()
                else None
            ),
            "league": _clean(ordered.iloc[0].get("league")),
            "tournament": _clean(ordered.iloc[0].get("tournament")) or None,
            "split": _clean(ordered.iloc[0].get("split")) or None,
            "playoffs": _clean(ordered.iloc[0].get("playoffs")) or None,
            "team_a_key": team_a,
            "team_b_key": team_b,
            "date_start": (
                ordered["date"].min().isoformat()
                if ordered["date"].notna().any()
                else None
            ),
            "date_end": (
                ordered["date"].max().isoformat()
                if ordered["date"].notna().any()
                else None
            ),
            "source_game_indices": [
                int(value) if pd.notna(value) else None
                for value in ordered["_source_game_index"]
            ],
            "source_game_uids": ordered["_source_game_uid"].astype(str).tolist(),
            "n_maps": int(len(ordered)),
            "score_a": team_a_wins,
            "score_b": team_b_wins,
            "scheduled_best_of": best_of,
            "series_format_source": (
                format_sources[0] if len(format_sources) == 1 else None
            ),
            "completion_status": completion_status,
            "completion_source": (
                completion_sources[0]
                if len(completion_sources) == 1
                else (
                    "score_to_format_validation"
                    if completion_status == "completed"
                    else None
                )
            ),
            "winner_team_key": winner_team_key,
            "rating_eligible": rating_eligible,
            "quarantine_reasons": sorted(set(reasons)),
        },
        canonical_indices,
    )


def build_canonical_series_ledger(
    maps: pd.DataFrame,
    *,
    max_gap_hours: float = 18.0,
) -> CanonicalSeriesResult:
    """Attach canonical series IDs and return one validated row per series."""

    if maps is None or maps.empty:
        return CanonicalSeriesResult(
            maps=pd.DataFrame(),
            series=pd.DataFrame(),
            audit={"ok": False, "reason": "empty_input"},
        )
    frame = _prepare_maps(maps)
    # Canonical fields are outputs of this ledger, never trusted inputs. This
    # makes rebuilding an older/schema-expanded pack idempotent and prevents
    # null or stale derived columns from colliding with the new join.
    frame = frame.drop(
        columns=[
            column
            for column in DERIVED_CANONICAL_COLUMNS
            if column in frame.columns
        ],
        errors="ignore",
    )
    frame["canonical_series_id"] = _assign_candidate_series(
        frame, max_gap_hours=max_gap_hours
    )

    summaries: list[dict[str, Any]] = []
    canonical_game_index = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for series_id, group in frame.groupby("canonical_series_id", sort=True):
        summary, index_by_row = _series_summary(str(series_id), group)
        summaries.append(summary)
        for row_index, game_index in index_by_row.items():
            if game_index is not None:
                canonical_game_index.loc[row_index] = game_index

    ledger = pd.DataFrame(summaries).sort_values(
        ["date_start", "canonical_series_id"], kind="mergesort"
    ).reset_index(drop=True)
    frame["canonical_game_index"] = canonical_game_index
    status_by_series = ledger.set_index("canonical_series_id")[
        [
            "scheduled_best_of",
            "completion_status",
            "completion_source",
            "rating_eligible",
            "winner_team_key",
            "quarantine_reasons",
        ]
    ].rename(
        columns={
            "completion_status": "canonical_series_status",
            "completion_source": "canonical_series_completion_source",
            "rating_eligible": "series_rating_eligible",
            "winner_team_key": "canonical_series_winner_team_key",
            "quarantine_reasons": "series_quarantine_reasons",
        }
    )
    frame = frame.join(status_by_series, on="canonical_series_id")
    frame = frame.rename(
        columns={
            "_source_game_index": "raw_source_game_index",
            "_source_game_uid": "raw_source_game_uid",
        }
    )
    public_columns = [
        column for column in frame.columns if not column.startswith("_")
    ]
    frame = frame[public_columns].sort_values(
        ["date", "canonical_series_id", "raw_source_game_index"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)

    statuses = (
        ledger["completion_status"].value_counts().sort_index().to_dict()
        if not ledger.empty
        else {}
    )
    audit = {
        "ok": bool(not ledger.empty and ledger["rating_eligible"].any()),
        "n_maps": int(len(frame)),
        "n_series": int(len(ledger)),
        "n_rating_eligible_series": int(ledger["rating_eligible"].sum()),
        "n_rating_eligible_maps": int(
            frame["series_rating_eligible"].fillna(False).sum()
        ),
        "completion_statuses": {
            str(key): int(value) for key, value in statuses.items()
        },
        "n_quarantined_series": int((~ledger["rating_eligible"]).sum()),
        "note": (
            "Series identity uses explicit source IDs or source game-number "
            "resets. Rating eligibility additionally requires contiguous map "
            "indices and a completed score compatible with a verified format."
        ),
    }
    return CanonicalSeriesResult(maps=frame, series=ledger, audit=audit)
