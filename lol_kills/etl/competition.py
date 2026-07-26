"""Canonical competition and team-identity labels for the public data contract.

The source feeds use competition labels as both provenance and modeling
features.  Those are different concerns: a historical ``LTA`` row should stay
auditable as LTA, while a current public ladder should not expose LTA as a
live league or create a second Americas ladder.

This module therefore keeps ``league_source`` and writes a canonical ``league``
plus an explicit event scope.  Team identity is independent of competition;
the same organization in LCS, MSI, or EWC receives one key.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import pandas as pd

from lol_kills.etl.aliases import normalize_team


TAXONOMY_VERSION = "2026-07-26.1"

# LTA was the 2025 Americas competition.  LTA North/South are source labels,
# not separate current public rating scopes; they are mapped to LCS while the
# original value remains in league_source.
DEPRECATED_LEAGUE_MAP: dict[str, str] = {
    "LTA": "LCS",
    "LTA N": "LCS",
    "LTA S": "LCS",
}

REGIONAL_LEAGUES = frozenset(
    {
        "LCK",
        "LPL",
        "LEC",
        "LCS",
        "CBLOL",
        "PCS",
        "VCS",
        "LJL",
        "LCP",
        "TCL",
    }
)

INTERNATIONAL_LEAGUES = frozenset({"MSI", "EWC", "FST", "WORLDS", "IWC", "MSC"})

_EVENT_TOKENS: tuple[tuple[str, str], ...] = (
    ("FIRST STAND", "FST"),
    ("WORLDS", "WORLDS"),
    ("WORLD CHAMPIONSHIP", "WORLDS"),
    ("MSI", "MSI"),
    ("EWC", "EWC"),
    ("ESPORTS WORLD CUP", "EWC"),
    ("IWC", "IWC"),
    ("MSC", "MSC"),
)

_WS_RE = re.compile(r"\s+")
_KEY_RE = re.compile(r"[^a-z0-9]+")


def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def source_league(value: Any) -> str:
    """Return a stable, audit-friendly source label."""

    return _WS_RE.sub(" ", _text(value).upper())


def canonical_league(value: Any) -> str:
    """Map source aliases to the public/model competition taxonomy."""

    raw = source_league(value)
    if raw in DEPRECATED_LEAGUE_MAP:
        return DEPRECATED_LEAGUE_MAP[raw]
    if raw in {"WORLD", "WORLD CHAMPIONSHIP", "WORLDS"}:
        return "WORLDS"
    if raw == "FIRST STAND":
        return "FST"
    return raw


@dataclass(frozen=True)
class CompetitionLabel:
    source: str
    league: str
    scope: str
    event_kind: str
    is_international: bool


def classify_competition(league: Any, tournament: Any = None) -> CompetitionLabel:
    """Classify a row without substring false positives.

    A regional source league takes precedence over event words in its
    tournament title.  Thus ``LCK / LCK 2026 Road to MSI`` remains regional;
    only a source league explicitly identifying an international event, or an
    otherwise unknown source with an exact event token, becomes international.
    """

    source = source_league(league)
    canonical = canonical_league(source)
    if canonical in REGIONAL_LEAGUES:
        return CompetitionLabel(source, canonical, "regional", "domestic", False)
    if canonical in INTERNATIONAL_LEAGUES:
        return CompetitionLabel(source, canonical, "international", canonical.lower(), True)

    tournament_text = source_league(tournament)
    for token, event in _EVENT_TOKENS:
        if re.search(rf"(?<![A-Z0-9]){re.escape(token)}(?![A-Z0-9])", tournament_text):
            return CompetitionLabel(source, event, "international", event.lower(), True)

    return CompetitionLabel(source, canonical or "UNKNOWN", "other", "other", False)


def team_identity_key(name: Any) -> str:
    """Return a stable identity key independent of league/event context."""

    canonical = normalize_team(_text(name))
    value = unicodedata.normalize("NFKD", canonical).encode("ascii", "ignore").decode("ascii")
    value = _KEY_RE.sub("-", value.casefold()).strip("-")
    return value or "unknown-team"


def canonicalize_competition_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Add canonical identity/event columns while preserving source labels.

    Existing display columns are normalized in place for backwards
    compatibility.  New columns are deliberately explicit so joins and
    models can use keys instead of display strings.
    """

    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()

    out = frame.copy()
    if "league" in out.columns:
        if "league_source" not in out.columns:
            out["league_source"] = out["league"]
        labels = [
            classify_competition(
                row.get("league_source") if _text(row.get("league_source")) else row.get("league"),
                row.get("tournament"),
            )
            for _, row in out.iterrows()
        ]
        out["league_source"] = [label.source for label in labels]
        out["league"] = [label.league for label in labels]
        out["competition_scope"] = [label.scope for label in labels]
        out["event_kind"] = [label.event_kind for label in labels]
        out["is_international"] = [label.is_international for label in labels]

    team_columns = ("teamname", "blue_team", "red_team", "blue_teamname", "red_teamname")
    for column in team_columns:
        if column not in out.columns:
            continue
        out[column] = out[column].map(lambda value: normalize_team(_text(value)) if _text(value) else value)
        key_column = {
            "teamname": "team_key",
            "blue_team": "blue_team_key",
            "red_team": "red_team_key",
            "blue_teamname": "blue_team_key",
            "red_teamname": "red_team_key",
        }[column]
        if key_column not in out.columns or column in ("blue_teamname", "red_teamname"):
            out[key_column] = out[column].map(team_identity_key)

    return out
