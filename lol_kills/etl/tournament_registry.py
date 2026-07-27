"""Authoritative, time-bounded tournament membership and format facts.

The match warehouse is evidence that a team played; it is not an authority on
whether the team currently belongs to a league.  This module loads a reviewed
Riot LoL Esports registry snapshot and fails closed when that snapshot is
missing, malformed, not yet valid, or overdue for review.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.etl.competition import canonical_league, team_identity_key


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY_PATH = (
    ROOT / "data" / "lol" / "registries" / "tier1-current-2026-07-26.json"
)
REGISTRY_SCHEMA_VERSION = "1.0.0"
_BEST_OF_RE = re.compile(r"(?:bo|best\s*of)\s*([135])", re.IGNORECASE)


class TournamentRegistryError(ValueError):
    """Raised when current-membership facts cannot be verified safely."""


@dataclass(frozen=True)
class RegistryFormatAnnotationResult:
    maps: pd.DataFrame
    audit: dict[str, Any]


def _required_text(
    value: Any,
    *,
    field: str,
    context: str = "registry",
) -> str:
    text = "" if value is None else str(value).strip()
    if not text:
        raise TournamentRegistryError(f"{context}: missing {field}")
    return text


def _utc_timestamp(value: Any, *, field: str) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise TournamentRegistryError(f"registry: invalid {field}")
    return timestamp


def validate_tournament_registry(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one immutable registry snapshot."""

    if not isinstance(payload, Mapping):
        raise TournamentRegistryError("registry payload must be an object")
    normalized = dict(payload)
    schema_version = _required_text(
        normalized.get("schema_version"), field="schema_version"
    )
    if schema_version != REGISTRY_SCHEMA_VERSION:
        raise TournamentRegistryError(
            f"unsupported tournament registry schema {schema_version!r}"
        )
    _required_text(normalized.get("snapshot_id"), field="snapshot_id")
    authority = _required_text(normalized.get("authority"), field="authority")
    if authority != "Riot Games LoL Esports":
        raise TournamentRegistryError(
            "registry authority must be Riot Games LoL Esports"
        )

    observed_at = _utc_timestamp(
        normalized.get("observed_at"), field="observed_at"
    )
    review_due_at = _utc_timestamp(
        normalized.get("review_due_at"), field="review_due_at"
    )
    if review_due_at <= observed_at:
        raise TournamentRegistryError(
            "registry review_due_at must be after observed_at"
        )

    tournaments = normalized.get("tournaments")
    if not isinstance(tournaments, list) or not tournaments:
        raise TournamentRegistryError("registry: tournaments must be non-empty")

    seen_tournaments: set[str] = set()
    seen_leagues: set[str] = set()
    seen_team_leagues: set[tuple[str, str]] = set()
    normalized_tournaments: list[dict[str, Any]] = []
    for position, raw_tournament in enumerate(tournaments):
        context = f"tournaments[{position}]"
        if not isinstance(raw_tournament, Mapping):
            raise TournamentRegistryError(f"{context}: must be an object")
        tournament = dict(raw_tournament)
        tournament_id = _required_text(
            tournament.get("tournament_id"),
            field="tournament_id",
            context=context,
        )
        if tournament_id in seen_tournaments:
            raise TournamentRegistryError(
                f"{context}: duplicate tournament_id {tournament_id!r}"
            )
        seen_tournaments.add(tournament_id)

        league = canonical_league(
            _required_text(
                tournament.get("league"), field="league", context=context
            )
        )
        if league in seen_leagues:
            raise TournamentRegistryError(
                f"{context}: multiple current tournaments for league {league}"
            )
        seen_leagues.add(league)
        tournament["league"] = league
        _required_text(tournament.get("name"), field="name", context=context)
        source_url = _required_text(
            tournament.get("source_url"),
            field="source_url",
            context=context,
        )
        if not source_url.startswith("https://lolesports.com/"):
            raise TournamentRegistryError(
                f"{context}: source_url is not an official LoL Esports URL"
            )
        if tournament.get("status") != "current":
            raise TournamentRegistryError(
                f"{context}: status must be 'current'"
            )

        participants = tournament.get("participants")
        if not isinstance(participants, list) or not participants:
            raise TournamentRegistryError(
                f"{context}: participants must be non-empty"
            )
        normalized_participants: list[dict[str, Any]] = []
        tournament_team_keys: set[str] = set()
        for team_position, raw_team in enumerate(participants):
            team_context = f"{context}.participants[{team_position}]"
            if not isinstance(raw_team, Mapping):
                raise TournamentRegistryError(
                    f"{team_context}: must be an object"
                )
            team = dict(raw_team)
            display_name = _required_text(
                team.get("display_name"),
                field="display_name",
                context=team_context,
            )
            _required_text(
                team.get("short_code"),
                field="short_code",
                context=team_context,
            )
            key = team_identity_key(display_name)
            if key in tournament_team_keys:
                raise TournamentRegistryError(
                    f"{team_context}: duplicate team identity {key!r}"
                )
            if (league, key) in seen_team_leagues:
                raise TournamentRegistryError(
                    f"{team_context}: duplicate league/team identity {key!r}"
                )
            tournament_team_keys.add(key)
            seen_team_leagues.add((league, key))
            team["team_key"] = key
            normalized_participants.append(team)
        tournament["participants"] = normalized_participants

        stages = tournament.get("stages", [])
        if not isinstance(stages, list):
            raise TournamentRegistryError(f"{context}: stages must be a list")
        for stage_position, raw_stage in enumerate(stages):
            stage_context = f"{context}.stages[{stage_position}]"
            if not isinstance(raw_stage, Mapping):
                raise TournamentRegistryError(
                    f"{stage_context}: must be an object"
                )
            stage = dict(raw_stage)
            _required_text(
                stage.get("stage_id"), field="stage_id", context=stage_context
            )
            best_of = stage.get("scheduled_best_of")
            if best_of is not None and best_of not in {1, 3, 5}:
                raise TournamentRegistryError(
                    f"{stage_context}: scheduled_best_of must be 1, 3, 5, or null"
                )
            if best_of is None and stage.get("format_status") != "unverified":
                raise TournamentRegistryError(
                    f"{stage_context}: null format must be marked unverified"
                )
            if best_of is not None and stage.get("format_status") != "verified":
                raise TournamentRegistryError(
                    f"{stage_context}: known format must be marked verified"
                )
            patterns = stage.get("tournament_label_patterns", [])
            if not isinstance(patterns, list) or not all(
                isinstance(pattern, str) and pattern.strip()
                for pattern in patterns
            ):
                raise TournamentRegistryError(
                    f"{stage_context}: tournament_label_patterns must be non-empty strings"
                )
            for pattern in patterns:
                try:
                    re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    raise TournamentRegistryError(
                        f"{stage_context}: invalid tournament label pattern"
                    ) from exc
        normalized_tournaments.append(tournament)

    normalized["observed_at"] = observed_at.isoformat()
    normalized["review_due_at"] = review_due_at.isoformat()
    normalized["tournaments"] = normalized_tournaments
    return normalized


def load_tournament_registry(
    path: Path | str = DEFAULT_REGISTRY_PATH,
) -> dict[str, Any]:
    """Load and validate a reviewed registry snapshot from disk."""

    registry_path = Path(path)
    if not registry_path.exists():
        raise TournamentRegistryError(
            f"tournament registry not found: {registry_path}"
        )
    try:
        payload = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TournamentRegistryError(
            f"could not read tournament registry: {registry_path}"
        ) from exc
    return validate_tournament_registry(payload)


def current_membership_from_registry(
    registry: Mapping[str, Any],
    *,
    as_of: Any,
) -> dict[str, Any]:
    """Return current league membership, refusing stale or future snapshots."""

    payload = validate_tournament_registry(registry)
    query_at = _utc_timestamp(as_of, field="as_of")
    observed_at = _utc_timestamp(payload["observed_at"], field="observed_at")
    review_due_at = _utc_timestamp(
        payload["review_due_at"], field="review_due_at"
    )
    if query_at < observed_at:
        raise TournamentRegistryError(
            "current-membership snapshot was observed after the requested as_of"
        )
    if query_at > review_due_at:
        raise TournamentRegistryError(
            "current-membership snapshot is overdue for authoritative review"
        )

    leagues: dict[str, str] = {}
    team_leagues: dict[str, dict[str, str]] = {}
    sources: dict[str, str] = {}
    participants_by_league: dict[str, list[str]] = {}
    team_display_names: dict[str, str] = {}
    for tournament in payload["tournaments"]:
        league = str(tournament["league"])
        name = str(tournament["name"])
        leagues[league] = name
        sources[league] = str(tournament["source_url"])
        participants: list[str] = []
        for team in tournament["participants"]:
            key = str(team["team_key"])
            participants.append(key)
            team_display_names[key] = str(team["display_name"])
            team_leagues.setdefault(key, {})[league] = name
        participants_by_league[league] = sorted(participants)

    return {
        "as_of": observed_at.isoformat(),
        "checked_at": query_at.isoformat(),
        "review_due_at": review_due_at.isoformat(),
        "window_days": None,
        "registry_snapshot_id": payload["snapshot_id"],
        "authority": payload["authority"],
        "leagues": dict(sorted(leagues.items())),
        "team_leagues": {
            key: dict(sorted(value.items()))
            for key, value in sorted(team_leagues.items())
        },
        "participants_by_league": dict(sorted(participants_by_league.items())),
        "team_display_names": dict(sorted(team_display_names.items())),
        "sources": dict(sorted(sources.items())),
    }


def _tournament_family(value: Any) -> str:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return re.sub(r"\s+\([^()]*\)\s*$", "", text).strip()


def _safe_text(value: Any) -> str:
    return "" if value is None or pd.isna(value) else str(value)


def _best_of(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    match = _BEST_OF_RE.search(text.replace("-", " "))
    if match:
        return int(match.group(1))
    try:
        number = int(float(text))
    except ValueError:
        return None
    return number if number in {1, 3, 5} else None


def _explicit_playoffs_value(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y", "playoffs", "postseason"}:
        return True
    if text in {"0", "false", "no", "n", "regular", "regular season"}:
        return False
    return None


def annotate_maps_with_tournament_registry(
    maps: pd.DataFrame,
    registry: Mapping[str, Any],
    *,
    as_of: Any,
) -> RegistryFormatAnnotationResult:
    """Fill only formats that a reviewed stage rule identifies uniquely."""

    payload = validate_tournament_registry(registry)
    current_membership_from_registry(payload, as_of=as_of)
    frame = maps.copy()
    if frame.empty:
        return RegistryFormatAnnotationResult(
            maps=frame,
            audit={
                "registry_snapshot_id": payload["snapshot_id"],
                "rows": 0,
                "matched_rows": 0,
                "filled_rows": 0,
                "conflict_rows": 0,
                "unmatched_current_tournament_rows": 0,
            },
        )

    frame["series_format_registry_conflict"] = False
    frame["series_format_registry_verified"] = False
    if "series_format_source" not in frame.columns:
        frame["series_format_source"] = pd.NA
    if "series_format_stage_id" not in frame.columns:
        frame["series_format_stage_id"] = pd.NA
    if "series_format_registry_snapshot_id" not in frame.columns:
        frame["series_format_registry_snapshot_id"] = pd.NA

    tournament_lookup = {
        (str(tournament["league"]), str(tournament["name"])): tournament
        for tournament in payload["tournaments"]
    }
    matched_rows = 0
    filled_rows = 0
    conflict_rows = 0
    unmatched_current_rows = 0
    for index, row in frame.iterrows():
        league = canonical_league(row.get("league"))
        family = _tournament_family(row.get("tournament"))
        tournament = tournament_lookup.get((league, family))
        if tournament is None:
            continue

        label = " ".join(
            _safe_text(row.get(column))
            for column in ("tournament", "stage", "phase", "round")
            if column in frame.columns
        )
        matching_stages: list[dict[str, Any]] = []
        for stage in tournament.get("stages", []):
            if stage.get("format_status") != "verified":
                continue
            patterns = stage.get("tournament_label_patterns", [])
            if patterns and any(
                re.search(pattern, label, flags=re.IGNORECASE)
                for pattern in patterns
            ):
                matching_stages.append(stage)

        if not matching_stages:
            playoffs = _explicit_playoffs_value(row.get("playoffs"))
            if playoffs is not None:
                matching_stages = [
                    stage
                    for stage in tournament.get("stages", [])
                    if stage.get("format_status") == "verified"
                    and stage.get("default_when_playoffs") is playoffs
                ]
        unique_formats = {
            int(stage["scheduled_best_of"])
            for stage in matching_stages
            if stage.get("scheduled_best_of") is not None
        }
        if len(matching_stages) != 1 or len(unique_formats) != 1:
            unmatched_current_rows += 1
            continue

        stage = matching_stages[0]
        verified_best_of = next(iter(unique_formats))
        existing_values = [
            row.get(column)
            for column in ("series_format", "best_of", "format")
            if column in frame.columns
        ]
        existing_formats = {
            parsed
            for parsed in (_best_of(value) for value in existing_values)
            if parsed is not None
        }
        matched_rows += 1
        frame.at[index, "series_format_registry_verified"] = True
        frame.at[index, "series_format_stage_id"] = stage["stage_id"]
        frame.at[
            index, "series_format_registry_snapshot_id"
        ] = payload["snapshot_id"]
        if existing_formats and existing_formats != {verified_best_of}:
            frame.at[index, "series_format_registry_conflict"] = True
            conflict_rows += 1
            continue
        if not existing_formats:
            frame.at[index, "series_format"] = f"Bo{verified_best_of}"
            frame.at[index, "series_format_source"] = (
                f"riot-registry:{payload['snapshot_id']}:{stage['stage_id']}"
            )
            filled_rows += 1

    return RegistryFormatAnnotationResult(
        maps=frame,
        audit={
            "registry_snapshot_id": payload["snapshot_id"],
            "rows": int(len(frame)),
            "matched_rows": matched_rows,
            "filled_rows": filled_rows,
            "conflict_rows": conflict_rows,
            "unmatched_current_tournament_rows": unmatched_current_rows,
        },
    )
