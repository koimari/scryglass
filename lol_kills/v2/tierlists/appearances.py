"""Played-only membership and scope filtering for L9 tier lists.

Appearances come from the OE warehouse player-games table
(oe_player_games.parquet), which carries league, champion, position, patch,
competition_tier, event_kind, and date per player-game.  Membership is
played-only: a champion enters a cell only with at least one verified
completed map in the exact league/event x patch x role scope.  Zero-play
champions are excluded; archetype transfer never places them on a tier list.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from lol_kills.v2.data.common import canonical_json_bytes, canonicalize_role, parse_rfc3339, to_rfc3339

from .model import (
    APPEARANCE_SOURCE,
    COMPETITION_TIERS,
    INTERNATIONAL_SCOPES,
    PATCH_RE,
    REGIONS,
    TierListError,
)
from .schema import validate_patch

SOURCE_COLUMNS = (
    "gameid",
    "league",
    "patch",
    "position",
    "champion",
    "date",
    "competition_tier",
    "event_kind",
)


@dataclass(frozen=True)
class AppearanceScope:
    """One league or international event cell filter: scope x tier x role x patch."""

    scope_kind: str
    scope_id: str
    role: str
    patch_id: str
    competition_tier: str | None = None
    region: str | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if self.scope_kind not in {"league", "international"}:
            raise TierListError("scope_kind must be league or international")
        scope_id = self.scope_id.strip().upper()
        if not scope_id:
            raise TierListError("scope_id is required")
        role = self.role.strip().lower()
        if role not in {"top", "jungle", "mid", "bot", "support"}:
            raise TierListError(f"unknown role: {self.role!r}")
        patch_id = validate_patch(self.patch_id)
        tier = self.competition_tier
        if tier is not None:
            if self.scope_kind == "international" and tier != "international":
                raise TierListError("international scope requires competition_tier=None or 'international'")
            if tier not in COMPETITION_TIERS and tier != "international":
                raise TierListError(f"unknown competition tier: {tier!r}")
        if self.scope_kind == "league":
            league_id = scope_id
            if league_id in INTERNATIONAL_SCOPES:
                raise TierListError("league scope cannot use an international scope_id")
            international_event = None
        else:
            if scope_id not in INTERNATIONAL_SCOPES:
                raise TierListError(f"international scope must be one of {sorted(INTERNATIONAL_SCOPES)}")
            league_id = None
            international_event = scope_id
        object.__setattr__(self, "scope_id", scope_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "patch_id", patch_id)
        object.__setattr__(self, "league_id", league_id)
        object.__setattr__(self, "international_event", international_event)
        object.__setattr__(self, "competition_scope_id", f"scryglass:scope:{self.scope_kind}-{scope_id.lower()}")
        if self.region is not None and self.region not in REGIONS:
            raise TierListError(f"unknown region: {self.region!r}")

    def as_mapping(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind,
            "scope_id": self.scope_id,
            "league_id": getattr(self, "league_id", None),
            "international_event": getattr(self, "international_event", None),
            "competition_scope_id": getattr(self, "competition_scope_id", None),
            "competition_tier": self.competition_tier,
            "competition_tier_source": "appearance_source",
            "region": self.region,
            "role": self.role,
            "patch_id": self.patch_id,
            "label": self.label or f"{self.scope_id} {self.patch_id} {self.role}",
        }


def league_scope(league_id: str, *, role: str, patch_id: str, competition_tier: str | None = None) -> AppearanceScope:
    return AppearanceScope(scope_kind="league", scope_id=league_id, role=role, patch_id=patch_id, competition_tier=competition_tier)


def international_scope(event_id: str, *, role: str, patch_id: str) -> AppearanceScope:
    return AppearanceScope(scope_kind="international", scope_id=event_id, role=role, patch_id=patch_id, competition_tier="international")


@dataclass(frozen=True)
class AppearanceRow:
    map_id: str
    league: str
    patch_id: str
    role: str
    champion_name: str
    event_end: str
    competition_tier: str | None
    event_kind: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AppearanceRow":
        required = {
            "map_id",
            "league",
            "patch_id",
            "role",
            "champion_name",
            "event_end",
            "competition_tier",
            "event_kind",
        }
        if set(value) != required:
            raise TierListError(f"appearance row fields must be exactly {sorted(required)}")
        map_id = value["map_id"]
        league = value["league"]
        patch_id = validate_patch(value["patch_id"])
        role = canonicalize_role(value["role"])
        champion_name = value["champion_name"]
        event_end = to_rfc3339(parse_rfc3339(value["event_end"]))
        if not isinstance(map_id, str) or not map_id:
            raise TierListError("appearance map_id is required")
        if not isinstance(league, str) or not league.strip():
            raise TierListError("appearance league is required")
        if not isinstance(champion_name, str) or not champion_name.strip():
            raise TierListError("appearance champion_name is required")
        tier = value["competition_tier"]
        if tier is not None and tier not in COMPETITION_TIERS and tier != "international":
            raise TierListError(f"unknown appearance competition tier: {tier!r}")
        kind = value["event_kind"]
        if kind is not None and not isinstance(kind, str):
            raise TierListError("appearance event_kind must be a string or null")
        return cls(
            map_id=map_id,
            league=league.strip(),
            patch_id=patch_id,
            role=role,
            champion_name=champion_name.strip(),
            event_end=event_end,
            competition_tier=tier,
            event_kind=kind.strip().lower() if isinstance(kind, str) else None,
        )


@dataclass(frozen=True)
class CellAppearances:
    """Filtered, verified appearances for exactly one tier-list cell."""

    scope: AppearanceScope
    as_of: str
    rows: tuple[AppearanceRow, ...]

    def membership(self) -> dict[str, dict[str, Any]]:
        """Played-only membership: champion_name -> verified counts in the cell."""
        counts: dict[str, dict[str, Any]] = {}
        for row in self.rows:
            entry = counts.setdefault(
                row.champion_name,
                {"distinct_maps": 0, "earliest_event_end": None, "latest_event_end": None},
            )
            entry["distinct_maps"] += 1
            entry["earliest_event_end"] = min(entry["earliest_event_end"] or row.event_end, row.event_end)
            entry["latest_event_end"] = max(entry["latest_event_end"] or row.event_end, row.event_end)
        return counts

    @property
    def played_champion_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.membership()))

    def window(self) -> dict[str, Any]:
        membership = self.membership()
        if not membership:
            return {"earliest_event_end": None, "latest_event_end": None, "distinct_maps": 0}
        earliest = min(entry["earliest_event_end"] for entry in membership.values())
        latest = max(entry["latest_event_end"] for entry in membership.values())
        return {"earliest_event_end": earliest, "latest_event_end": latest, "distinct_maps": len(self.rows)}


class AppearanceTable:
    """Immutable verified-appearance table with source identity."""

    def __init__(self, rows: Sequence[AppearanceRow], *, source_locator: str, raw_sha256: str | None) -> None:
        if not raw_sha256 or not isinstance(raw_sha256, str) or len(raw_sha256) != 64:
            raise TierListError("appearance source raw_sha256 is required")
        # Identical duplicate rows (known OE ingest artifacts) are deduplicated;
        # conflicting duplicates for the same identity fail closed.
        seen: dict[tuple[str, str, str, str], tuple[AppearanceRow, int]] = {}
        deduped: list[AppearanceRow] = []
        for row in rows:
            key = (row.map_id, row.role, row.champion_name, row.league)
            prior = seen.get(key)
            if prior is None:
                seen[key] = (row, len(deduped))
                deduped.append(row)
            elif prior[0] != row:
                raise TierListError(f"conflicting duplicate appearance row: {key}")
        self._rows = tuple(deduped)
        self.source_locator = source_locator
        self.raw_sha256 = raw_sha256

    @classmethod
    def from_rows(
        cls,
        rows: Iterable[Mapping[str, Any]],
        *,
        source_locator: str = "synthetic",
        raw_sha256: str | None = None,
    ) -> "AppearanceTable":
        parsed = tuple(AppearanceRow.from_mapping(dict(row)) for row in rows)
        digest = raw_sha256 or hashlib.sha256(canonical_json_bytes([row.__dict__ for row in parsed])).hexdigest()
        return cls(parsed, source_locator=source_locator, raw_sha256=digest)

    @classmethod
    def from_oe_player_games(
        cls,
        root: Path,
        *,
        locator: str | None = None,
        expected_raw_sha256: str | None = None,
    ) -> "AppearanceTable":
        """Load and verify the OE warehouse player-games table (fail closed)."""
        path = root / (locator or APPEARANCE_SOURCE["locator"])
        if not path.is_file() or path.is_symlink():
            raise TierListError("oe player-games source is missing or not a regular file")
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_raw_sha256 is not None and expected_raw_sha256 != digest:
            raise TierListError("oe player-games source bytes do not match the expected sha256")
        try:
            frame = pd.read_parquet(path, columns=list(SOURCE_COLUMNS))
        except (KeyError, ValueError) as exc:
            raise TierListError(f"oe player-games missing required columns: {exc}") from exc
        rows: list[AppearanceRow] = []
        for record in frame.itertuples(index=False):
            if record.position is None or pd.isna(record.position):
                continue
            if record.champion is None or pd.isna(record.champion):
                continue
            if record.date is None or pd.isna(record.date):
                continue
            if record.patch is None or pd.isna(record.patch):
                continue
            if record.league is None or pd.isna(record.league):
                continue
            patch_token = str(record.patch).strip()
            if not PATCH_RE.fullmatch(patch_token):
                raise TierListError(f"oe player-games contains a malformed patch token: {patch_token!r}")
            # OE stores the map datetime as naive UTC; attach UTC explicitly
            event_end = to_rfc3339(record.date.to_pydatetime().replace(tzinfo=timezone.utc))
            tier = None if pd.isna(record.competition_tier) else str(record.competition_tier)
            kind = None if pd.isna(record.event_kind) else str(record.event_kind)
            rows.append(
                AppearanceRow(
                    map_id=str(record.gameid),
                    league=str(record.league).strip(),
                    patch_id=patch_token,
                    role=canonicalize_role(str(record.position)),
                    champion_name=str(record.champion).strip(),
                    event_end=event_end,
                    competition_tier=tier,
                    event_kind=kind,
                )
            )
        return cls(rows, source_locator=str(path), raw_sha256=digest)

    def rows(self) -> tuple[AppearanceRow, ...]:
        return self._rows

    def filter(self, scope: AppearanceScope, *, as_of: str) -> CellAppearances:
        """Filter to exactly one cell; champions with zero play are excluded."""
        as_of_utc = parse_rfc3339(as_of)
        selected: list[AppearanceRow] = []
        for row in self._rows:
            if parse_rfc3339(row.event_end) > as_of_utc:
                continue
            if row.patch_id != scope.patch_id:
                continue
            if row.role != scope.role:
                continue
            if scope.scope_kind == "league":
                if row.league != scope.scope_id:
                    continue
            else:
                if row.league != scope.scope_id or row.event_kind != scope.scope_id.lower():
                    continue
            if scope.competition_tier is not None and row.competition_tier != scope.competition_tier:
                continue
            selected.append(row)
        return CellAppearances(scope=scope, as_of=to_rfc3339(as_of_utc), rows=tuple(selected))

    def latest_patch(
        self,
        scope_id: str,
        *,
        scope_kind: str,
        competition_tier: str | None = None,
        as_of: str | None = None,
    ) -> str:
        """Current patch: patch of the most recent completed eligible maps (L1 rule 2)."""
        as_of_utc = parse_rfc3339(as_of) if as_of is not None else datetime(9999, 1, 1, tzinfo=timezone.utc)
        by_date: dict[str, set[str]] = {}
        for row in self._rows:
            if parse_rfc3339(row.event_end) > as_of_utc:
                continue
            if scope_kind == "league" and row.league != scope_id:
                continue
            if scope_kind == "international" and (row.league != scope_id or row.event_kind != scope_id.lower()):
                continue
            if competition_tier is not None and row.competition_tier != competition_tier:
                continue
            by_date.setdefault(row.event_end, set()).add(row.patch_id)
        if not by_date:
            raise TierListError(f"no completed eligible maps for scope {scope_id}")
        latest_date = max(by_date)
        patches = by_date[latest_date]
        if len(patches) > 1:
            raise TierListError(f"conflicting patches on the most recent completed maps: {sorted(patches)}")
        return validate_patch(next(iter(patches)))
