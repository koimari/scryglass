#!/usr/bin/env python3
"""Build the public elemental-drake study artifact from GRID telemetry.

The artifact intentionally separates:

1. deterministic Dragon Slayer mechanics;
2. composition and game-state snapshots observed in GRID pro games; and
3. outcome-effect estimates, which stay disabled until the cohort is large
   enough for state-controlled, held-out analysis.

Raw GRID provider files remain in the private warehouse. Only the compact
derived JSON written by this module is intended for the public web app.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import re
import statistics
import subprocess
import tempfile
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import urljoin, urlparse

import httpx
import pandas as pd

from lol_kills.draft_archetypes import champ_tags
from lol_kills.draft_recommendation import CURRENT_CHAMPIONS
from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.competition import classify_competition
from lol_kills.etl.grid_ingest import (
    ALLOWED_SERIES_TYPE,
    GRAPHQL_ENDPOINT,
    GridIngestError,
    LOL_TITLE_ID,
    RAW_GRID_DIR,
    _assert_pro,
    _api_key,
    _file_list,
    _iter_jsonl,
    _resolve_sides,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "apps" / "elemental-drakes" / "src" / "data" / "drake-study.json"
DEFAULT_EXPLORER_MODEL_OUTPUT = (
    ROOT / "data" / "lol" / "models" / "elemental_drake_explorer_model.json"
)
DEFAULT_AUDIT_OUTPUT = ROOT / "data" / "lol" / "models" / "elemental_drake_audit.json"
DEFAULT_SNAPSHOTS_OUTPUT = (
    ROOT / "data" / "lol" / "models" / "elemental_drake_tier_one_snapshots.json"
)
DEFAULT_ROLE_CATALOG_INPUT = ROOT / "data" / "lol" / "draft_players.json"
COMPACT_GRID_DIR = ROOT / "data" / "lol" / "warehouse" / "grid_drakes"
COMPACT_SERIES_DIR = COMPACT_GRID_DIR / "series"
SERIES_CATALOG_PATH = COMPACT_GRID_DIR / "series_catalog.json"
COMPACT_GAMES_PARQUET = COMPACT_GRID_DIR / "games.parquet"
COMPACT_EVENTS_PARQUET = COMPACT_GRID_DIR / "events.parquet"
COMPACT_SCHEMA_VERSION = 8
GRID_NORMALIZED_EVENTS_URL = "https://api.grid.gg/file-download/events/grid/series"
_GRID_DOWNLOAD_LOCK = threading.Lock()
_GRID_NEXT_DOWNLOAD_AT = 0.0
_GRID_DOWNLOAD_INTERVAL_SECONDS = 3.2
_GRID_HTTP_CLIENTS = threading.local()

DRAGON_TYPE_ALIASES = {
    "air": "cloud",
    "cloud": "cloud",
    "chemtech": "chemtech",
    "earth": "mountain",
    "fire": "infernal",
    "hextech": "hextech",
    "infernal": "infernal",
    "mountain": "mountain",
    "ocean": "ocean",
    "water": "ocean",
}

GRID_DRAKE_EVENTS = {
    "player-completed-slayChemtechDrake": "chemtech",
    "player-completed-slayCloudDrake": "cloud",
    "player-completed-slayHextechDrake": "hextech",
    "player-completed-slayInfernalDrake": "infernal",
    "player-completed-slayMountainDrake": "mountain",
    "player-completed-slayOceanDrake": "ocean",
}

MECHANICS = [
    {
        "id": "infernal",
        "name": "Infernal",
        "short": "More attack damage and ability power",
        "perStack": "3% attack damage and ability power",
        "unit": "% AD + AP",
        "value": 3,
        "directTags": ["burst_mage", "hypercarry_adc", "poke_siege", "scaling_late"],
        "source": "Riot Games, Patch 13.20",
        "sourceUrl": "https://www.leagueoflegends.com/en-sg/news/game-updates/patch-13-20-notes/",
    },
    {
        "id": "mountain",
        "name": "Mountain",
        "short": "More bonus armor and magic resistance",
        "perStack": "5% bonus armor and magic resistance",
        "unit": "% bonus resists",
        "value": 5,
        "directTags": ["tank_frontline", "engage", "skirmisher"],
        "source": "Riot Games, Patch 13.20",
        "sourceUrl": "https://www.leagueoflegends.com/en-sg/news/game-updates/patch-13-20-notes/",
    },
    {
        "id": "ocean",
        "name": "Ocean",
        "short": "Regenerate missing health",
        "perStack": "2.5% missing health every 5 seconds",
        "unit": "% missing HP / 5s",
        "value": 2.5,
        "directTags": ["tank_frontline", "engage", "skirmisher", "peel_enchanter"],
        "source": "Riot Games, Patch 13.6",
        "sourceUrl": "https://www.leagueoflegends.com/en-au/news/game-updates/patch-13-6-notes/",
    },
    {
        "id": "cloud",
        "name": "Cloud",
        "short": "Move faster out of combat and resist slows",
        "perStack": "5% out-of-combat movement speed and slow resistance",
        "unit": "% OOC move + slow resist",
        "value": 5,
        "directTags": ["roam", "pick", "engage", "splitpush"],
        "source": "Riot Games, Patch 13.20",
        "sourceUrl": "https://www.leagueoflegends.com/en-sg/news/game-updates/patch-13-20-notes/",
    },
    {
        "id": "hextech",
        "name": "Hextech",
        "short": "Cast and attack more often",
        "perStack": "5 ability haste and 5% bonus attack speed",
        "unit": "AH + % bonus AS",
        "value": 5,
        "directTags": ["hypercarry_adc", "control_mage", "skirmisher", "teamfight_aoe"],
        "source": "Riot Games, Patch 13.20",
        "sourceUrl": "https://www.leagueoflegends.com/en-sg/news/game-updates/patch-13-20-notes/",
    },
    {
        "id": "chemtech",
        "name": "Chemtech",
        "short": "Resist crowd control and amplify healing and shields",
        "perStack": "5% tenacity and 5% healing and shielding power",
        "unit": "% tenacity + heal/shield",
        "value": 5,
        "directTags": ["peel_enchanter", "tank_frontline", "engage", "teamfight_aoe"],
        "source": "Riot Games, Patch 12.22",
        "sourceUrl": "https://www.leagueoflegends.com/en-us/news/game-updates/patch-12-22-notes/",
    },
]

MECHANICS_BY_ID = {row["id"]: row for row in MECHANICS}
CURRENT_CHAMPION_LOOKUP = {
    re.sub(r"[^a-z0-9]", "", champion.casefold()): champion
    for champion in CURRENT_CHAMPIONS
}
ROLE_ORDER = ("Top", "Jungle", "Mid", "Bot", "Support")
MIN_RANDOM_ROLE_APPEARANCES = 5


def _canonical_champion_name(value: Any) -> str:
    raw = str(value or "Unknown")
    special = {"monkeyking": "Wukong"}
    key = re.sub(r"[^a-z0-9]", "", raw.casefold())
    return special.get(
        key,
        CURRENT_CHAMPION_LOOKUP.get(key, normalize_champ(raw)),
    )


def load_role_catalog(
    path: Path = DEFAULT_ROLE_CATALOG_INPUT,
) -> dict[str, Any]:
    """Return a compact, provider-safe champion pool for role-aware controls."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": "unavailable",
            "source": None,
            "appearances": 0,
            "games": 0,
            "minimumRandomAppearances": MIN_RANDOM_ROLE_APPEARANCES,
            "roles": [],
        }
    rows = payload.get("players") if isinstance(payload, Mapping) else None
    role_counts: dict[str, Counter[str]] = {
        role: Counter() for role in ROLE_ORDER
    }
    game_ids: set[str] = set()
    appearances = 0
    for row in rows or []:
        if not isinstance(row, Mapping):
            continue
        role = str(row.get("role") or "")
        if role not in role_counts:
            continue
        champion = _canonical_champion_name(row.get("champion"))
        if champion not in CURRENT_CHAMPIONS:
            continue
        role_counts[role][champion] += 1
        appearances += 1
        game_id = str(row.get("game_id") or "")
        if game_id:
            game_ids.add(game_id)
    return {
        "status": "ready" if appearances else "unavailable",
        "source": str(payload.get("source") or "reviewed pro draft role rows"),
        "appearances": appearances,
        "games": len(game_ids),
        "minimumRandomAppearances": MIN_RANDOM_ROLE_APPEARANCES,
        "roles": [
            {
                "role": role,
                "champions": [
                    {"name": champion, "appearances": count}
                    for champion, count in sorted(
                        role_counts[role].items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
            }
            for role in ROLE_ORDER
        ],
    }

# GRID tournament titles do not include canonical league, tier, or region
# columns. Keep this projection explicit and versioned instead of relying on
# substring matching such as "LCK" in "LCK Challengers".
GRID_TOURNAMENT_LEAGUES: dict[str, str] = {
    "LCK": "LCK",
    "LPL": "LPL",
    "LEC": "LEC",
    "LCS": "LCS",
    "CBLOL": "CBLOL",
    "LCP": "LCP",
    "LCK CHALLENGERS": "LCKC",
    "EMEA MASTERS": "EM",
    "PRIME LEAGUE": "PRM",
    "LA LIGUE FRANÇAISE": "LFL",
    "ARABIAN LEAGUE": "AL",
    "HITPOINT MASTERS": "HM",
    "NACL": "NACL",
    "LRS": "LRS",
    "LRN": "LRN",
    "CIRCUITO DESAFIANTE": "CD",
    "HELLENIC LEGENDS LEAGUE": "HLL",
    "LOL ITALIAN TOURNAMENT": "LIT",
    "TCL": "TCL",
    "ESPORTS BALKAN LEAGUE": "EBL",
    "NLC": "NLC",
    "LIGA PORTUGUESA": "LPLOL",
    "LJL": "LJL",
    "VCS": "VCS",
    "PCS": "PCS",
    "ROAD OF LEGENDS": "ROL",
    "KESPA CUP": "KESPA",
    "KESPA CUP 2025": "KESPA",
    "WORLDS": "WORLDS",
    "MSI": "MSI",
    "FIRST STAND": "FST",
    "ESPORTS WORLD CUP": "EWC",
    "LTA CROSS-CONFERENCE": "AMERICAS",
}

REGION_BY_LEAGUE: dict[str, tuple[str, str]] = {
    "LCK": ("korea", "Korea"),
    "LCKC": ("korea", "Korea"),
    "KESPA": ("korea", "Korea"),
    "LPL": ("china", "China"),
    "LEC": ("emea", "EMEA"),
    "EM": ("emea", "EMEA"),
    "PRM": ("emea", "EMEA"),
    "LFL": ("emea", "EMEA"),
    "AL": ("emea", "EMEA"),
    "HM": ("emea", "EMEA"),
    "HLL": ("emea", "EMEA"),
    "LIT": ("emea", "EMEA"),
    "TCL": ("emea", "EMEA"),
    "EBL": ("emea", "EMEA"),
    "NLC": ("emea", "EMEA"),
    "LPLOL": ("emea", "EMEA"),
    "LCS": ("north-america", "North America"),
    "NACL": ("north-america", "North America"),
    "CBLOL": ("south-america", "South America"),
    "LRS": ("south-america", "South America"),
    "LRN": ("south-america", "South America"),
    "CD": ("south-america", "South America"),
    "AMERICAS": ("south-america", "South America"),
    "LCP": ("pacific", "Pacific"),
    "LJL": ("pacific", "Pacific"),
    "VCS": ("pacific", "Pacific"),
    "PCS": ("pacific", "Pacific"),
    "ROL": ("pacific", "Pacific"),
    "WORLDS": ("international", "International"),
    "MSI": ("international", "International"),
    "FST": ("international", "International"),
    "EWC": ("international", "International"),
}

REGION_ORDER = (
    "korea",
    "china",
    "emea",
    "north-america",
    "south-america",
    "pacific",
    "international",
    "other",
)

TIER_ONE_SNAPSHOT_SPECS: tuple[dict[str, Any], ...] = (
    {"region": "international", "seriesId": "2930124", "gameIndex": 1},
    {"region": "korea", "seriesId": "2954868", "gameIndex": 1},
    {"region": "china", "seriesId": "2975401", "gameIndex": 1},
    {"region": "emea", "seriesId": "2966871", "gameIndex": 1},
    {"region": "north-america", "seriesId": "2964589", "gameIndex": 1},
    {"region": "south-america", "seriesId": "2973238", "gameIndex": 1},
    {"region": "pacific", "seriesId": "2978925", "gameIndex": 1},
)

FALLBACK_REGION_PREFIXES: dict[str, set[str]] = {
    "china": {"DEMACIA CUP 2025"},
    "emea": {
        "ESPORTS WORLD CUP 2026 ONLINE QUALIFIER: EMEA",
        "ESPORTS NATIONS CUP 2026 EUROPE EAST QUALIFIER",
        "ESPORTS NATIONS CUP 2026 EUROPE WEST QUALIFIER",
        "ESPORTS NATIONS CUP 2026 MIDDLE EAST & AFRICA QUALIFIER",
        "NOVA SERIES 2025 PRÉLUDE",
        "ELMILLOR INVITATIONAL",
    },
    "korea": {"ESPORTS WORLD CUP 2026 ONLINE QUALIFIER: KOREA"},
    "north-america": {
        "COMEDY CENTRAL WINTER SNOWDOWN",
        "NACL 2026 SUMMER PROMOTION",
        "ESPORTS WORLD CUP 2026 ONLINE QUALIFIER: NORTH AMERICA",
        "ESPORTS NATIONS CUP 2026 NORTH AMERICA QUALIFIER",
    },
    "south-america": {
        "LES",
        "AMERICAS CUP",
        "ESPORTS WORLD CUP 2026 ONLINE QUALIFIER: SOUTH AMERICA",
        "ESPORTS NATIONS CUP 2026 SOUTH AMERICA QUALIFIER",
    },
    "pacific": {
        "RIFT LEGENDS",
        "ESPORTS WORLD CUP 2026 ONLINE QUALIFIER: APAC",
        "ESPORTS NATIONS CUP 2026 ASIA QUALIFIER",
        "ESPORTS NATIONS CUP 2026 SOUTHEAST ASIA & OCEANIA QUALIFIER",
    },
}


def _tournament_prefix(tournament: Any) -> str:
    return str(tournament or "").split(" - ", 1)[0].strip()


def competition_metadata(tournament: Any) -> dict[str, str]:
    """Return canonical competition fields for one GRID tournament title."""
    prefix = _tournament_prefix(tournament)
    source_league = GRID_TOURNAMENT_LEAGUES.get(prefix.upper(), prefix)
    label = classify_competition(source_league, tournament)
    fallback_region = next(
        (
            candidate
            for candidate, prefixes in FALLBACK_REGION_PREFIXES.items()
            if prefix.upper() in prefixes
        ),
        None,
    )
    region_id, region_label = REGION_BY_LEAGUE.get(
        label.league,
        ("other", "Other / mixed"),
    )
    if fallback_region:
        region_id = fallback_region
    elif region_id == "other" and label.is_international:
        region_id, region_label = "international", "International"
    if region_id != "other" and isinstance(region_label, str):
        canonical_region_labels = {
            "china": "China",
            "emea": "EMEA",
            "korea": "Korea",
            "north-america": "North America",
            "south-america": "South America",
            "pacific": "Pacific",
            "international": "International",
        }
        region_label = canonical_region_labels[region_id]
    level = (
        "other-pro"
        if fallback_region
        else
        "tier1"
        if label.tier == "tier1"
        else "international"
        if label.is_international
        else "other-pro"
    )
    return {
        "competition": prefix or label.league,
        "league": label.league,
        "region": region_id,
        "regionLabel": region_label,
        "level": level,
        "levelLabel": {
            "tier1": "Tier 1",
            "international": "International",
            "other-pro": "Tier 2 / other pro",
        }[level],
    }


def _eligible_games(
    games: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    eligible = []
    for game in games:
        team_ids = {
            str(team.get("id") or "")
            for team in game.get("teams") or []
            if str(team.get("id") or "")
        }
        winner = str(game.get("winnerTeamId") or "")
        if bool(game.get("complete")) and len(team_ids) == 2 and winner in team_ids:
            eligible.append(game)
    return eligible


def summarize_competition_coverage(
    games: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Summarize the model-eligible cohort by region and competition level."""
    eligible = _eligible_games(games)
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    labels: dict[str, str] = {}
    competitions: dict[str, Counter[str]] = defaultdict(Counter)
    for game in eligible:
        meta = competition_metadata(game.get("tournament"))
        region = meta["region"]
        labels[region] = meta["regionLabel"]
        buckets[region][meta["level"]] += 1
        competitions[region][meta["competition"]] += 1
    rows = []
    for region in REGION_ORDER:
        counts = buckets.get(region)
        if not counts:
            continue
        rows.append(
            {
                "id": region,
                "label": labels.get(region, region.replace("-", " ").title()),
                "games": sum(counts.values()),
                "tierOneGames": counts.get("tier1", 0),
                "internationalGames": counts.get("international", 0),
                "otherProGames": counts.get("other-pro", 0),
                "competitions": [
                    {"name": name, "games": count}
                    for name, count in competitions[region].most_common()
                ],
            }
        )
    return {
        "eligibleGames": len(eligible),
        "tierOneGames": sum(row["tierOneGames"] for row in rows),
        "internationalGames": sum(row["internationalGames"] for row in rows),
        "otherProGames": sum(row["otherProGames"] for row in rows),
        "unclassifiedGames": next(
            (row["games"] for row in rows if row["id"] == "other"),
            0,
        ),
        "regions": rows,
        "taxonomy": (
            "Tier 1 means the six canonical regional leagues: LCK, LPL, LEC, "
            "LCS, CBLOL, and LCP. International events and developmental or "
            "other professional circuits are reported separately."
        ),
    }


def normalize_dragon_type(value: Any) -> str | None:
    """Return the public element name for a Riot or GRID dragon label."""
    raw = re.sub(r"[^a-z]", "", str(value or "").lower())
    raw = raw.removesuffix("drake").removesuffix("dragon")
    return DRAGON_TYPE_ALIASES.get(raw)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _round(value: Any, digits: int = 1) -> float:
    return round(_as_float(value), digits)


def _role(value: Any) -> str:
    raw = re.sub(r"[^a-z]", "", str(value or "").lower())
    return {
        "top": "Top",
        "jungle": "Jungle",
        "middle": "Mid",
        "mid": "Mid",
        "bottom": "Bot",
        "bot": "Bot",
        "utility": "Support",
        "support": "Support",
    }.get(raw, str(value or "Unknown").title())


def _team_stats(snapshot: Mapping[str, Any] | None, team_id: int) -> Mapping[str, Any]:
    if not snapshot:
        return {}
    for team in snapshot.get("teams") or []:
        if _as_int(team.get("teamID")) == team_id:
            return team
    return {}


def _participant_snapshot(
    snapshot: Mapping[str, Any] | None,
    team_id: int,
) -> list[dict[str, Any]]:
    if not snapshot:
        return []
    out: list[dict[str, Any]] = []
    for participant in snapshot.get("participants") or []:
        if _as_int(participant.get("teamID")) != team_id:
            continue
        health_max = participant.get("healthMax")
        attack_speed = _as_float(participant.get("attackSpeed"))
        if attack_speed > 10:
            attack_speed /= 100
        champion = _canonical_champion_name(participant.get("championName"))
        row = {
            "champion": champion,
            "role": _role(participant.get("role")),
            "level": _as_int(participant.get("level"), 1),
            "attackDamage": _round(participant.get("attackDamage")),
            "abilityPower": _round(participant.get("abilityPower")),
            "armor": _round(participant.get("armor")),
            "magicResist": _round(participant.get("magicResist")),
            "attackSpeed": round(attack_speed, 3),
            "health": _round(participant.get("health")),
            "healthMax": _round(health_max),
            "currentGold": _as_int(participant.get("currentGold")),
            "totalGold": _as_int(participant.get("totalGold")),
        }
        row["tags"] = sorted(champ_tags(row["champion"]))
        out.append(row)
    role_order = {"Top": 0, "Jungle": 1, "Mid": 2, "Bot": 3, "Support": 4}
    return sorted(out, key=lambda row: role_order.get(row["role"], 9))


def _snapshot_state(
    snapshot: Mapping[str, Any] | None,
    *,
    team_id: int,
    opponent_id: int,
) -> dict[str, Any]:
    own = _team_stats(snapshot, team_id)
    enemy = _team_stats(snapshot, opponent_id)
    own_gold = _as_int(own.get("totalGold"))
    enemy_gold = _as_int(enemy.get("totalGold"))
    return {
        "teamGold": own_gold,
        "opponentGold": enemy_gold,
        "goldDiff": own_gold - enemy_gold,
        "teamKills": _as_int(own.get("championsKills")),
        "opponentKills": _as_int(enemy.get("championsKills")),
        "teamTowers": _as_int(own.get("towerKills")),
        "opponentTowers": _as_int(enemy.get("towerKills")),
    }


def _fallback_sides(
    participants: Sequence[Mapping[str, Any]],
    candidates: Sequence[str],
) -> dict[int, str] | None:
    """Resolve small-league team prefixes not yet present in the alias table."""
    prefixes: dict[int, set[str]] = {100: set(), 200: set()}
    for participant in participants:
        team_id = _as_int(participant.get("teamID"))
        riot_id = participant.get("riotId") or {}
        display = (
            riot_id.get("displayName")
            if isinstance(riot_id, Mapping)
            else participant.get("summonerName")
        )
        prefix = re.sub(r"[^a-z0-9]", "", str(display or "").split()[0].lower())
        if team_id in prefixes and prefix:
            prefixes[team_id].add(prefix)
    scored: dict[int, list[tuple[int, str]]] = {}
    for team_id, observed in prefixes.items():
        values = []
        for candidate in candidates:
            words = [
                re.sub(r"[^a-z0-9]", "", word.lower())
                for word in str(candidate).split()
                if word.lower() not in {"esports", "gaming", "team"}
            ]
            tokens = {word for word in words if word}
            tokens |= {word[:4] for word in words if len(word) >= 4}
            tokens |= {word[:3] for word in words if len(word) >= 3}
            score = sum(prefix in tokens for prefix in observed)
            values.append((score, str(candidate)))
        scored[team_id] = sorted(values, reverse=True)
    if any(not values or values[0][0] == 0 for values in scored.values()):
        return None
    result = {team_id: values[0][1] for team_id, values in scored.items()}
    return result if result.get(100) != result.get(200) else None


def _parse_raw_game(path: Path) -> dict[str, Any] | None:
    match = re.match(r"events_(\d+)_(\d+)_riot\.jsonl$", path.name)
    if not match:
        return None
    series_id, game_index = match.groups()
    meta_path = path.parent / f"series_{series_id}.json"
    if not meta_path.exists():
        return None
    series = json.loads(meta_path.read_text(encoding="utf-8"))
    game_info: Mapping[str, Any] | None = None
    game_end: Mapping[str, Any] | None = None
    latest_stats: Mapping[str, Any] | None = None
    dragon_events: list[dict[str, Any]] = []

    for row in _iter_jsonl(path):
        schema = row.get("rfc461Schema")
        if schema == "game_info" and game_info is None:
            game_info = row
        elif schema == "stats_update":
            latest_stats = row
        elif schema == "game_end":
            game_end = row
        elif (
            schema == "epic_monster_kill"
            and str(row.get("monsterType") or "").lower() == "dragon"
        ):
            element = normalize_dragon_type(row.get("dragonType"))
            owner = _as_int(row.get("killerTeamID"))
            if element and owner in (100, 200):
                dragon_events.append(
                    {
                        "element": element,
                        "timeSeconds": round(_as_int(row.get("gameTime")) / 1000),
                        "ownerTeamId": owner,
                        "_snapshot": latest_stats,
                    }
                )

    participants = list((game_info or {}).get("participants") or [])
    if not game_info or len(participants) != 10 or not dragon_events:
        return None
    sides = _resolve_sides(participants, series.get("teams") or [])
    if not sides:
        sides = _fallback_sides(participants, series.get("teams") or [])
    if not sides:
        sides = {100: "Blue team", 200: "Red team"}
    winner = _as_int((game_end or {}).get("winningTeam"))
    team_stacks = Counter()
    public_events = []
    first_snapshot = dragon_events[0].get("_snapshot")
    for global_index, event in enumerate(dragon_events, start=1):
        owner = event["ownerTeamId"]
        team_stacks[owner] += 1
        opponent = 200 if owner == 100 else 100
        snapshot = event.pop("_snapshot")
        public_events.append(
            {
                **event,
                "globalIndex": global_index,
                "ownerStack": team_stacks[owner],
                "teamName": sides[owner],
                "opponentName": sides[opponent],
                "state": _snapshot_state(snapshot, team_id=owner, opponent_id=opponent),
                "composition": _participant_snapshot(snapshot, owner),
                "opponentComposition": _participant_snapshot(snapshot, opponent),
            }
        )

    platform = str(game_info.get("platformID") or "")
    game_id = str(game_info.get("gameID") or "")
    return {
        "id": f"{series_id}-{game_index}",
        "providerGameId": f"{platform}-{game_id}",
        "seriesId": series_id,
        "gameIndex": _as_int(game_index, 1),
        "tournament": str(series.get("tournament") or "Professional series"),
        "patch": str(game_info.get("gameVersion") or "").split(".")[0:2],
        "winnerTeamId": winner if winner in (100, 200) else None,
        "complete": game_end is not None,
        "teams": [
            {
                "id": team_id,
                "name": sides[team_id],
                "side": "Blue" if team_id == 100 else "Red",
                "won": winner == team_id if winner in (100, 200) else None,
                "composition": _participant_snapshot(first_snapshot, team_id),
            }
            for team_id in (100, 200)
        ],
        "dragonEvents": public_events,
    }


def parse_raw_pilot(raw_dir: Path = RAW_GRID_DIR) -> list[dict[str, Any]]:
    games = []
    for path in sorted(raw_dir.glob("events_*_*_riot.jsonl")):
        parsed = _parse_raw_game(path)
        if parsed:
            parsed["patch"] = ".".join(parsed["patch"])
            games.append(parsed)
    return games


def _stream_grid_jsonl(url: str, key: str) -> Iterator[Mapping[str, Any]]:
    """Yield one GRID JSONL response without persisting the provider file."""
    global _GRID_NEXT_DOWNLOAD_AT

    headers = {
        "x-api-key": key,
        "Accept": "application/json,application/octet-stream,*/*",
        "User-Agent": "scryglass/elemental-drake-snapshot-projector",
    }
    current_url = url
    for _ in range(4):
        host = (urlparse(current_url).hostname or "").lower()
        if host != "grid.gg" and not host.endswith(".grid.gg"):
            raise RuntimeError(f"blocked non-GRID snapshot host={host or 'unknown'}")
        with _GRID_DOWNLOAD_LOCK:
            now = time.monotonic()
            delay = max(0.0, _GRID_NEXT_DOWNLOAD_AT - now)
            if delay:
                time.sleep(delay)
            _GRID_NEXT_DOWNLOAD_AT = (
                time.monotonic() + _GRID_DOWNLOAD_INTERVAL_SECONDS
            )
        client = _grid_http_client()
        with client.stream("GET", current_url, headers=headers) as response:
            if response.status_code in {301, 302, 303, 307, 308}:
                location = response.headers.get("location")
                if not location:
                    raise RuntimeError("GRID snapshot redirect omitted location")
                current_url = urljoin(current_url, location)
                continue
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, Mapping):
                    yield payload
            return
    raise RuntimeError("GRID snapshot download exceeded redirect limit")


def _first_drake_snapshot_from_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    series: Mapping[str, Any],
    game_index: int,
    complete: bool,
    side_names: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Project only the state required for one observed first-drake example."""
    game_info: Mapping[str, Any] | None = None
    latest_stats: Mapping[str, Any] | None = None
    first_drake: dict[str, Any] | None = None
    for row in rows:
        schema = row.get("rfc461Schema")
        if schema == "game_info" and game_info is None:
            game_info = row
        elif schema == "stats_update":
            latest_stats = row
        elif (
            schema == "epic_monster_kill"
            and str(row.get("monsterType") or "").lower() == "dragon"
        ):
            element = normalize_dragon_type(row.get("dragonType"))
            owner = _as_int(row.get("killerTeamID"))
            if element and owner in (100, 200) and latest_stats:
                first_drake = {
                    "element": element,
                    "timeSeconds": round(_as_int(row.get("gameTime")) / 1000),
                    "ownerTeamId": owner,
                    "_snapshot": latest_stats,
                }
                break
    participants = list((game_info or {}).get("participants") or [])
    if not game_info or len(participants) != 10 or not first_drake:
        raise RuntimeError("GRID raw stream did not yield a complete first-drake snapshot")
    sides = _resolve_sides(participants, series.get("teams") or [])
    if not sides:
        sides = _fallback_sides(participants, series.get("teams") or [])
    if not sides and side_names:
        blue = str(side_names.get("blue") or "")
        red = str(side_names.get("red") or "")
        if blue and red and blue != red:
            sides = {100: blue, 200: red}
    if not sides:
        raise RuntimeError("GRID raw stream team sides could not be resolved")
    owner = int(first_drake["ownerTeamId"])
    opponent = 200 if owner == 100 else 100
    snapshot = first_drake.pop("_snapshot")
    event = {
        **first_drake,
        "globalIndex": 1,
        "ownerStack": 1,
        "teamName": sides[owner],
        "opponentName": sides[opponent],
        "state": _snapshot_state(snapshot, team_id=owner, opponent_id=opponent),
        "composition": _participant_snapshot(snapshot, owner),
        "opponentComposition": _participant_snapshot(snapshot, opponent),
    }
    if len(event["composition"]) != 5 or len(event["opponentComposition"]) != 5:
        raise RuntimeError("GRID first-drake snapshot did not contain both 5v5 compositions")
    patch_parts = str(game_info.get("gameVersion") or "").split(".")[:2]
    return {
        "id": f"projected-{game_index}",
        "date": str(series.get("date") or ""),
        "tournament": str(series.get("tournament") or "Professional series"),
        "patch": ".".join(patch_parts),
        "complete": complete,
        "teams": [
            {
                "side": "blue",
                "name": sides[100],
                "composition": (
                    event["composition"]
                    if owner == 100
                    else event["opponentComposition"]
                ),
            },
            {
                "side": "red",
                "name": sides[200],
                "composition": (
                    event["composition"]
                    if owner == 200
                    else event["opponentComposition"]
                ),
            },
        ],
        "dragonEvents": [event],
    }


def refresh_tier_one_snapshots(
    *,
    output: Path = DEFAULT_SNAPSHOTS_OUTPUT,
    env_file: Path | None = None,
) -> dict[str, Any]:
    """Stream one verified Tier 1 example per region and retain only its projection."""
    key = _api_key(env_file)
    existing_games = load_tier_one_snapshots(output)
    public_by_region = {
        str(game.get("region") or ""): game
        for game in existing_games
        if str(game.get("region") or "")
    }
    checks: list[dict[str, Any]] = []

    def checkpoint() -> dict[str, Any]:
        ordered = [
            public_by_region[str(spec["region"])]
            for spec in TIER_ONE_SNAPSHOT_SPECS
            if str(spec["region"]) in public_by_region
        ]
        for index, game in enumerate(ordered, start=1):
            game["id"] = f"snapshot-{index}"
            for event in game.get("dragonEvents") or []:
                if isinstance(event, dict):
                    event.pop("ownerTeamId", None)
        payload = {
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "games": ordered,
            "checks": checks,
            "rawFilesRetained": False,
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    for spec in TIER_ONE_SNAPSHOT_SPECS:
        if str(spec["region"]) in public_by_region:
            continue
        series_id = str(spec["seriesId"])
        game_index = _as_int(spec["gameIndex"], 1)
        compact_path = _compact_path(series_id)
        compact = json.loads(compact_path.read_text(encoding="utf-8"))
        series = compact.get("series") or {}
        competition = competition_metadata(series.get("tournament"))
        if competition["region"] != spec["region"]:
            raise RuntimeError(
                f"snapshot {series_id} region changed from {spec['region']} "
                f"to {competition['region']}"
            )
        if competition["level"] not in {"tier1", "international"}:
            raise RuntimeError(f"snapshot {series_id} is not top-flight competition")
        compact_games = list(compact.get("games") or [])
        if game_index < 1 or game_index > len(compact_games):
            raise RuntimeError(f"snapshot {series_id} game {game_index} is unavailable")
        compact_game = compact_games[game_index - 1]
        if not compact_game.get("complete") or not compact_game.get("dragonEvents"):
            raise RuntimeError(f"snapshot {series_id} game {game_index} is incomplete")
        files: list[dict[str, Any]] | None = None
        file_list_error: Exception | None = None
        for attempt in range(4):
            try:
                files = _file_list(key, series_id)
                break
            except GridIngestError as exc:
                file_list_error = exc
                if attempt < 3:
                    time.sleep(min(12.0, 2.0 * (2**attempt)))
        if files is None:
            raise RuntimeError(
                f"snapshot {series_id} file list failed after retries: "
                f"{file_list_error}"
            )
        wanted_id = f"events-riot-game-{game_index}"
        source = next(
            (
                file
                for file in files
                if str(file.get("id") or "") == wanted_id
                and str(file.get("status") or "") == "ready"
                and str(file.get("fullURL") or "")
            ),
            None,
        )
        if not source:
            raise RuntimeError(f"snapshot {series_id} has no ready {wanted_id} file")
        compact_side_names = {
            str(team.get("side") or "").lower(): str(team.get("name") or "")
            for team in compact_game.get("teams") or []
            if str(team.get("side") or "").lower() in {"blue", "red"}
        }
        last_error: Exception | None = None
        game: dict[str, Any] | None = None
        for attempt in range(4):
            try:
                game = _first_drake_snapshot_from_rows(
                    _stream_grid_jsonl(str(source["fullURL"]), key),
                    series=series,
                    game_index=game_index,
                    complete=True,
                    side_names=compact_side_names,
                )
                break
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(min(12.0, 2.0 * (2**attempt)))
        if game is None:
            raise RuntimeError(
                f"snapshot {series_id} stream failed after retries: {last_error}"
            )
        compact_first = compact_game["dragonEvents"][0]
        raw_first = game["dragonEvents"][0]
        matches = {
            "element": raw_first["element"] == compact_first["element"],
            "timeWithinTwoSeconds": abs(
                int(raw_first["timeSeconds"]) - int(compact_first["timeSeconds"])
            )
            <= 2,
        }
        if not all(matches.values()):
            raise RuntimeError(f"snapshot {series_id} failed compact reconciliation")
        compact_team_names = {
            str(team.get("id") or ""): str(team.get("name") or "")
            for team in compact_game.get("teams") or []
        }
        compact_team_sides = {
            str(team.get("id") or ""): str(team.get("side") or "").lower()
            for team in compact_game.get("teams") or []
        }
        names_by_side = {
            compact_team_sides[team_id]: name
            for team_id, name in compact_team_names.items()
            if compact_team_sides.get(team_id) in {"blue", "red"}
        }
        for team in game["teams"]:
            team["name"] = names_by_side.get(team["side"], team["name"])
        game["observedCaptures"] = [
            {
                "globalIndex": _as_int(event.get("globalIndex")),
                "element": str(event.get("element") or ""),
                "timeSeconds": _as_int(event.get("timeSeconds")),
                "ownerSide": compact_team_sides.get(
                    str(event.get("ownerTeamId") or ""),
                    str(event.get("ownerSide") or ""),
                ),
                "ownerName": compact_team_names.get(
                    str(event.get("ownerTeamId") or ""),
                    "Unknown team",
                ),
                "ownerStack": _as_int(event.get("ownerStack")),
            }
            for event in compact_game.get("dragonEvents") or []
        ]
        prepared = _prepare_pilot_games([game])
        public_by_region[str(spec["region"])] = _public_pilot_games(prepared)[0]
        checks.append(
            {
                "region": competition["region"],
                "element": matches["element"],
                "timeWithinTwoSeconds": matches["timeWithinTwoSeconds"],
            }
        )
        print(
            f"[elemental-drakes] projected_top_flight_snapshot="
            f"{competition['region']} {series.get('tournament')}",
            flush=True,
        )
        checkpoint()
    return checkpoint()


def load_tier_one_snapshots(
    path: Path = DEFAULT_SNAPSHOTS_OUTPUT,
) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    games = payload.get("games") if isinstance(payload, Mapping) else None
    loaded = [dict(game) for game in games or [] if isinstance(game, Mapping)]
    # Recompute derived annotations on load so saved snapshots never freeze an
    # obsolete public ontology or copy contract.
    return _prepare_pilot_games(loaded)


def _current_game(state: Mapping[str, Any] | None) -> Mapping[str, Any]:
    games = list((state or {}).get("games") or [])
    return games[-1] if games else {}


def _team_name(actor: Mapping[str, Any] | None) -> str | None:
    state = (actor or {}).get("state") or {}
    name = str(state.get("name") or "").strip()
    return name or None


def _team_id_from_actor(actor: Mapping[str, Any] | None) -> str:
    actor = actor or {}
    state = actor.get("state") or {}
    if actor.get("type") == "team":
        return str(actor.get("id") or state.get("id") or "")
    return str(state.get("teamId") or "")


def _grid_team_state(game: Mapping[str, Any], team_id: str) -> Mapping[str, Any]:
    for team in game.get("teams") or []:
        if str(team.get("id") or "") == team_id:
            return team
    return {}


def _objective_count(team: Mapping[str, Any], needle: str) -> int:
    count = 0
    for objective in team.get("objectives") or []:
        if needle.lower() in str(objective.get("type") or "").lower():
            count += _as_int(objective.get("completionCount"))
    return count


def _max_player_value(team: Mapping[str, Any], field: str) -> int:
    return max(
        (
            _as_int(player.get(field))
            for player in team.get("players") or []
            if isinstance(player, Mapping)
        ),
        default=0,
    )


def parse_normalized_grid(path: Path) -> list[dict[str, Any]]:
    """Parse one compact GRID normalized event archive into per-game records."""
    games: dict[str, dict[str, Any]] = {}
    picks: dict[tuple[str, str], list[str]] = defaultdict(list)
    team_names: dict[str, str] = {}
    team_sides: dict[str, str] = {}
    team_players: dict[tuple[str, str], list[str]] = {}
    team_player_ids: dict[tuple[str, str], list[str]] = {}
    winners: dict[str, str] = {}
    game_versions: dict[str, str] = {}
    latest_games: dict[str, Mapping[str, Any]] = {}
    latest_occurred_at: dict[str, str] = {}

    for envelope in _iter_jsonl(path):
        envelope_updates: dict[str, Mapping[str, Any]] = {}
        occurred_at = str(envelope.get("occurredAt") or "")
        for event in envelope.get("events") or []:
            state = event.get("seriesState") or {}
            game = _current_game(state)
            game_id = str(game.get("id") or "")
            if not game_id:
                continue
            envelope_updates[game_id] = game
            title_version = game.get("titleVersion")
            version = str(
                ((title_version or {}).get("name") or "")
                if isinstance(title_version, Mapping)
                else title_version or ""
            )
            if version:
                game_versions[game_id] = version
            event_type = str(event.get("type") or "")
            actor = event.get("actor") or {}
            team_id = _team_id_from_actor(actor)
            name = _team_name(actor) if actor.get("type") == "team" else None
            if name and team_id:
                team_names[team_id] = name
            actor_state = actor.get("state") or {}
            side = str(actor_state.get("side") or "").lower()
            if team_id and side in {"blue", "red"}:
                team_sides[team_id] = side

            if event_type == "team-picked-character":
                champion = str(((event.get("target") or {}).get("state") or {}).get("name") or "")
                if team_id and champion and champion not in picks[(game_id, team_id)]:
                    picks[(game_id, team_id)].append(champion)
                continue
            if event_type == "team-won-game" and team_id:
                winners[game_id] = team_id
                continue

            element = GRID_DRAKE_EVENTS.get(event_type)
            if not element or not team_id:
                continue
            snapshot_game = latest_games.get(game_id, game)
            teams = list(snapshot_game.get("teams") or [])
            owner = _grid_team_state(snapshot_game, team_id)
            opponent = next(
                (team for team in teams if str(team.get("id") or "") != team_id),
                {},
            )
            opponent_id = str(opponent.get("id") or "")
            for team in teams:
                found_team_id = str(team.get("id") or "")
                player_names = [
                    str(player.get("name") or "").strip()
                    for player in team.get("players") or []
                    if isinstance(player, Mapping) and str(player.get("name") or "").strip()
                ]
                if found_team_id and player_names:
                    team_players[(game_id, found_team_id)] = player_names
                player_ids = [
                    str(player.get("id") or "").strip()
                    for player in team.get("players") or []
                    if isinstance(player, Mapping) and str(player.get("id") or "").strip()
                ]
                if found_team_id and player_ids:
                    team_player_ids[(game_id, found_team_id)] = player_ids
            previous_at = latest_occurred_at.get(game_id)
            state_lag_seconds: float | None = None
            if previous_at and occurred_at:
                try:
                    current_dt = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
                    previous_dt = datetime.fromisoformat(previous_at.replace("Z", "+00:00"))
                    state_lag_seconds = max(
                        0.0,
                        (current_dt - previous_dt).total_seconds(),
                    )
                except ValueError:
                    state_lag_seconds = None
            current = games.setdefault(
                game_id,
                {
                    "id": game_id,
                    "seriesId": str(envelope.get("seriesId") or ""),
                    "patch": version,
                    "dragonEvents": [],
                    "teamIds": [],
                },
            )
            for found in (team_id, opponent_id):
                if found and found not in current["teamIds"]:
                    current["teamIds"].append(found)
            current["dragonEvents"].append(
                {
                    "element": element,
                    "occurredAt": envelope.get("occurredAt"),
                    "timeSeconds": _as_int((game.get("clock") or {}).get("currentSeconds")),
                    "ownerTeamId": team_id,
                    "ownerSide": team_sides.get(team_id),
                    "stateTiming": (
                        "previous-envelope"
                        if game_id in latest_games
                        else "same-event-fallback"
                    ),
                    "stateLagSeconds": (
                        round(state_lag_seconds, 3)
                        if state_lag_seconds is not None
                        else None
                    ),
                    "ownerNetWorth": _as_int(owner.get("netWorth")),
                    "opponentNetWorth": _as_int(opponent.get("netWorth")),
                    "goldDiff": _as_int(owner.get("netWorth")) - _as_int(opponent.get("netWorth")),
                    "ownerLoadoutValue": _as_int(owner.get("loadoutValue")),
                    "opponentLoadoutValue": _as_int(opponent.get("loadoutValue")),
                    "loadoutDiff": _as_int(owner.get("loadoutValue"))
                    - _as_int(opponent.get("loadoutValue")),
                    "ownerUnspentMoney": _as_int(owner.get("money")),
                    "opponentUnspentMoney": _as_int(opponent.get("money")),
                    "unspentMoneyDiff": _as_int(owner.get("money"))
                    - _as_int(opponent.get("money")),
                    "ownerTopPlayerNetWorth": _max_player_value(owner, "netWorth"),
                    "opponentTopPlayerNetWorth": _max_player_value(
                        opponent,
                        "netWorth",
                    ),
                    "topPlayerNetWorthDiff": _max_player_value(owner, "netWorth")
                    - _max_player_value(opponent, "netWorth"),
                    "ownerKills": _objective_count(owner, "killPlayer"),
                    "opponentKills": _objective_count(opponent, "killPlayer"),
                    "ownerTowers": _objective_count(owner, "destroyTower"),
                    "opponentTowers": _objective_count(opponent, "destroyTower"),
                }
            )
        for game_id, game in envelope_updates.items():
            latest_games[game_id] = game
            if occurred_at:
                latest_occurred_at[game_id] = occurred_at

    out = []
    for game_id, game in games.items():
        counts = Counter()
        events = sorted(game["dragonEvents"], key=lambda row: row["timeSeconds"])
        for index, event in enumerate(events, start=1):
            # Actor-side metadata is not stable across every normalized
            # envelope. The final team state is the authoritative mapping.
            event["ownerSide"] = team_sides.get(str(event["ownerTeamId"]))
            counts[event["ownerTeamId"]] += 1
            event["globalIndex"] = index
            event["ownerStack"] = counts[event["ownerTeamId"]]
        game["dragonEvents"] = events
        game["patch"] = game_versions.get(game_id, str(game.get("patch") or ""))
        game["winnerTeamId"] = winners.get(game_id)
        game["complete"] = game_id in winners
        game["teams"] = [
            {
                "id": team_id,
                "name": team_names.get(team_id, f"Team {team_id}"),
                "side": team_sides.get(team_id),
                "champions": picks.get((game_id, team_id), []),
                "players": team_players.get((game_id, team_id), []),
                "playerIds": team_player_ids.get((game_id, team_id), []),
                "won": winners.get(game_id) == team_id if game_id in winners else None,
            }
            for team_id in game.pop("teamIds")
        ]
        out.append(game)
    return out


def parse_normalized_cohort(raw_dir: Path = RAW_GRID_DIR) -> list[dict[str, Any]]:
    games: list[dict[str, Any]] = []
    compact_series: set[str] = set()
    for path in sorted(COMPACT_SERIES_DIR.glob("series_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schemaVersion") != COMPACT_SCHEMA_VERSION:
            continue
        series = payload.get("series") or {}
        series_id = str(series.get("id") or path.stem.removeprefix("series_"))
        compact_series.add(series_id)
        for game in payload.get("games") or []:
            if isinstance(game, Mapping):
                row = dict(game)
                row.setdefault("seriesId", series_id)
                row["date"] = series.get("date")
                row["tournamentId"] = series.get("tournamentId")
                row["tournament"] = series.get("tournament")
                games.append(row)
    for path in sorted(raw_dir.glob("events_*_grid.jsonl.zip")):
        match = re.match(r"events_(\d+)_grid\.jsonl\.zip$", path.name)
        if match and match.group(1) in compact_series:
            continue
        games.extend(parse_normalized_grid(path))
    unique = {f"{game['seriesId']}:{game['id']}": game for game in games}
    return list(unique.values())


def _distribution(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {element: counts.get(element, 0) for element in MECHANICS_BY_ID}


def summarize_cohort(games: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible_games = _eligible_games(games)
    events = [
        event
        for game in eligible_games
        for event in game.get("dragonEvents") or []
    ]
    first = [
        game["dragonEvents"][0]
        for game in eligible_games
        if game.get("dragonEvents")
    ]
    times = [event["timeSeconds"] for event in first if event.get("timeSeconds")]
    slot_distribution = {
        str(slot): _distribution(
            event["element"]
            for game in eligible_games
            for event in game.get("dragonEvents") or []
            if event.get("globalIndex") == slot
        )
        for slot in range(1, 5)
    }
    return {
        "games": len(games),
        "completeGames": len(eligible_games),
        "dragonEvents": len(events),
        "firstDrakes": len(first),
        "medianFirstDrakeSeconds": round(statistics.median(times)) if times else None,
        "firstDrakeDistribution": _distribution(event["element"] for event in first),
        "slotDistribution": slot_distribution,
        "outcomeModel": {
            "status": "gated",
            "reason": (
                "The current cohort is a schema-validating pilot. Outcome effects stay "
                "disabled until the sample supports patch, side, pre-drake gold, draft, "
                "and team-strength controls with held-out evaluation and uncertainty."
            ),
        },
    }


def _composition_fit(composition: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    champions = [str(row.get("champion") or "") for row in composition]
    result: dict[str, Any] = {}
    for mechanic in MECHANICS:
        wanted = set(mechanic["directTags"])
        higher_conversion = []
        for champion in champions:
            tags = champ_tags(champion)
            if tags & wanted:
                higher_conversion.append(champion)
        result[mechanic["id"]] = {
            "recipients": champions,
            "recipientCount": len(champions),
            "higherConversionCandidates": higher_conversion,
            "candidateCount": len(higher_conversion),
            "basis": (
                "all five champions receive the buff; the candidate list is a transparent "
                "archetype prior for differential conversion, not an estimated champion effect"
            ),
        }
    return result


def _prepare_pilot_games(games: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for game in games:
        for event in game.get("dragonEvents") or []:
            event["compositionFit"] = _composition_fit(event.get("composition") or [])
        out.append(game)
    return out


def _public_pilot_games(
    games: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    public = []
    for index, game in enumerate(games, start=1):
        competition = competition_metadata(game.get("tournament"))
        events = []
        for event in game.get("dragonEvents") or []:
            if not isinstance(event, Mapping):
                continue
            events.append(
                {
                    key: value
                    for key, value in event.items()
                    if key not in {"ownerTeamId"}
                }
            )
        public.append(
            {
                "id": f"snapshot-{index}",
                "date": str(game.get("date") or ""),
                "tournament": str(game.get("tournament") or ""),
                "patch": str(game.get("patch") or ""),
                "complete": bool(game.get("complete")),
                "league": competition["league"],
                "region": competition["region"],
                "regionLabel": competition["regionLabel"],
                "competitionLevel": competition["level"],
                "competitionLevelLabel": competition["levelLabel"],
                "teams": list(game.get("teams") or []),
                "observedCaptures": list(game.get("observedCaptures") or []),
                "dragonEvents": events,
            }
        )
    return public


def _public_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    public = {
        key: audit[key]
        for key in (
            "status",
            "requiredGames",
            "games",
            "events",
            "coverage",
            "modelEligibility",
            "storageBytes",
            "errors",
            "warnings",
        )
        if key in audit
    }
    reconciliation = audit.get("rawReconciliation")
    if isinstance(reconciliation, Mapping):
        public["rawReconciliation"] = {
            key: reconciliation[key]
            for key in (
                "status",
                "rawSeries",
                "comparedSeries",
                "eventsCompared",
                "elementMatches",
                "ownerMatches",
                "timeWithinTwoSeconds",
            )
            if key in reconciliation
        }
    return public


def _reconcile_pilot_names(
    pilot_games: Sequence[dict[str, Any]],
    cohort_games: Sequence[Mapping[str, Any]],
) -> None:
    """Use GRID's normalized team labels when raw Riot IDs lack known prefixes."""
    by_series: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for game in cohort_games:
        by_series[str(game.get("seriesId") or "")].append(game)
    for pilot in pilot_games:
        candidates = by_series.get(str(pilot.get("seriesId") or ""), [])
        if not candidates:
            continue
        for event in pilot.get("dragonEvents") or []:
            own = {row.get("champion") for row in event.get("composition") or []}
            enemy = {row.get("champion") for row in event.get("opponentComposition") or []}
            best_own: tuple[int, str] = (0, "")
            best_enemy: tuple[int, str] = (0, "")
            for candidate in candidates:
                for team in candidate.get("teams") or []:
                    champions = {
                        normalize_champ(str(champion))
                        for champion in team.get("champions") or []
                    }
                    name = str(team.get("name") or "")
                    own_score = len(own & champions)
                    enemy_score = len(enemy & champions)
                    if own_score > best_own[0]:
                        best_own = (own_score, name)
                    if enemy_score > best_enemy[0]:
                        best_enemy = (enemy_score, name)
            if best_own[0] >= 3:
                event["teamName"] = best_own[1]
            if best_enemy[0] >= 3:
                event["opponentName"] = best_enemy[1]
        first = (pilot.get("dragonEvents") or [None])[0]
        if not first:
            continue
        for team in pilot.get("teams") or []:
            team_id = team.get("id")
            matching = (
                first.get("teamName")
                if first.get("ownerTeamId") == team_id
                else first.get("opponentName")
            )
            if matching:
                team["name"] = matching


def _compact_path(series_id: str) -> Path:
    return COMPACT_SERIES_DIR / f"series_{series_id}.json"


def _grid_http_client() -> httpx.Client:
    client = getattr(_GRID_HTTP_CLIENTS, "client", None)
    if client is None:
        client = httpx.Client(
            follow_redirects=False,
            timeout=httpx.Timeout(120.0, connect=30.0),
            limits=httpx.Limits(
                max_connections=4,
                max_keepalive_connections=4,
                keepalive_expiry=90.0,
            ),
        )
        _GRID_HTTP_CLIENTS.client = client
    return client


def _download_normalized_archive(
    url: str,
    key: str,
    dest: Path,
) -> tuple[bool, str | None]:
    """Download through per-worker keepalive pools and GRID-only redirects."""
    global _GRID_NEXT_DOWNLOAD_AT

    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"x-api-key": key}
    status = "unknown"
    detail = ""
    for attempt in range(4):
        with _GRID_DOWNLOAD_LOCK:
            now = time.monotonic()
            delay = max(0.0, _GRID_NEXT_DOWNLOAD_AT - now)
            if delay:
                time.sleep(delay)
            _GRID_NEXT_DOWNLOAD_AT = (
                time.monotonic() + _GRID_DOWNLOAD_INTERVAL_SECONDS
            )
        try:
            client = _grid_http_client()
            current_url = url
            response: httpx.Response | None = None
            for _ in range(4):
                response = client.get(current_url, headers=headers)
                status = str(response.status_code)
                if response.status_code not in {301, 302, 303, 307, 308}:
                    break
                location = response.headers.get("location")
                if not location:
                    break
                next_url = urljoin(current_url, location)
                host = (urlparse(next_url).hostname or "").lower()
                if host != "grid.gg" and not host.endswith(".grid.gg"):
                    return False, f"blocked non-GRID redirect host={host or 'unknown'}"
                current_url = next_url
            if response is None:
                detail = "GRID returned no response"
            elif response.status_code == 200 and response.content:
                dest.write_bytes(response.content)
                return True, None
            else:
                detail = " ".join(response.text.strip().split())[:160]
        except httpx.HTTPError as exc:
            status = "transport"
            detail = str(exc)[:160]
        dest.unlink(missing_ok=True)
        if status == "404":
            break
        if status == "429":
            with _GRID_DOWNLOAD_LOCK:
                _GRID_NEXT_DOWNLOAD_AT = max(
                    _GRID_NEXT_DOWNLOAD_AT,
                    time.monotonic() + min(60.0, 10.0 * (2**attempt)),
                )
        elif attempt < 3:
            time.sleep(min(12.0, 2.0 * (2**attempt)))
    return (
        False,
        f"http={status}"
        + (f" detail={detail}" if detail else ""),
    )


def _read_compact_count(path: Path) -> int:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    if payload.get("schemaVersion") != COMPACT_SCHEMA_VERSION:
        return 0
    return sum(
        1
        for game in payload.get("games") or []
        if isinstance(game, Mapping) and game.get("dragonEvents")
    )


def _compact_is_current(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("schemaVersion") == COMPACT_SCHEMA_VERSION


def _extract_one_series(
    series: Mapping[str, Any],
    *,
    key: str,
) -> dict[str, Any]:
    """Fetch, project, and discard one normalized GRID archive."""
    series_id = str(series.get("id") or "")
    compact_path = _compact_path(series_id)
    if compact_path.exists() and _compact_is_current(compact_path):
        return {
            "seriesId": series_id,
            "status": "existing",
            "games": _read_compact_count(compact_path),
            "bytes": compact_path.stat().st_size,
        }
    try:
        existing = RAW_GRID_DIR / f"events_{series_id}_grid.jsonl.zip"
        with tempfile.TemporaryDirectory(prefix=f"grid-drakes-{series_id}-") as temp_dir:
            archive = existing if existing.exists() else Path(temp_dir) / existing.name
            url = f"{GRID_NORMALIZED_EVENTS_URL}/{series_id}"
            if not archive.exists():
                downloaded, error = _download_normalized_archive(url, key, archive)
                if not downloaded:
                    return {
                        "seriesId": series_id,
                        "status": "failed",
                        "games": 0,
                        "bytes": 0,
                        "error": error,
                    }
            games = parse_normalized_grid(archive)
            games = [game for game in games if game.get("dragonEvents")]
            payload = {
                "schemaVersion": COMPACT_SCHEMA_VERSION,
                "series": {
                    "id": series_id,
                    "date": series.get("date"),
                    "tournamentId": series.get("tournament_id"),
                    "tournament": series.get("tournament"),
                    "teams": series.get("teams") or [],
                },
                "games": games,
            }
            COMPACT_SERIES_DIR.mkdir(parents=True, exist_ok=True)
            compact_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        return {
            "seriesId": series_id,
            "status": "extracted",
            "games": len(games),
            "bytes": compact_path.stat().st_size,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "seriesId": series_id,
            "status": "failed",
            "games": 0,
            "bytes": 0,
            "error": str(exc)[:240],
        }


def consolidate_compact_parquet() -> dict[str, Any]:
    """Write the relevant per-game and per-drake columns as compressed parquet."""
    game_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    for path in sorted(COMPACT_SERIES_DIR.glob("series_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("schemaVersion") != COMPACT_SCHEMA_VERSION:
            continue
        series = payload.get("series") or {}
        competition = competition_metadata(series.get("tournament"))
        for game in payload.get("games") or []:
            if not isinstance(game, Mapping):
                continue
            teams = list(game.get("teams") or [])
            game_rows.append(
                {
                    "series_id": str(series.get("id") or ""),
                    "game_id": str(game.get("id") or ""),
                    "date": series.get("date"),
                    "tournament_id": str(series.get("tournamentId") or ""),
                    "tournament": str(series.get("tournament") or ""),
                    "competition": competition["competition"],
                    "league": competition["league"],
                    "region": competition["region"],
                    "competition_level": competition["level"],
                    "patch": str(game.get("patch") or ""),
                    "complete": bool(game.get("complete")),
                    "winner_team_id": str(game.get("winnerTeamId") or ""),
                    "team_1_id": str((teams[0] if len(teams) > 0 else {}).get("id") or ""),
                    "team_1_name": str((teams[0] if len(teams) > 0 else {}).get("name") or ""),
                    "team_1_side": str((teams[0] if len(teams) > 0 else {}).get("side") or ""),
                    "team_1_champions": json.dumps(
                        (teams[0] if len(teams) > 0 else {}).get("champions") or [],
                        separators=(",", ":"),
                    ),
                    "team_1_players": json.dumps(
                        (teams[0] if len(teams) > 0 else {}).get("players") or [],
                        separators=(",", ":"),
                    ),
                    "team_1_player_ids": json.dumps(
                        (teams[0] if len(teams) > 0 else {}).get("playerIds") or [],
                        separators=(",", ":"),
                    ),
                    "team_2_id": str((teams[1] if len(teams) > 1 else {}).get("id") or ""),
                    "team_2_name": str((teams[1] if len(teams) > 1 else {}).get("name") or ""),
                    "team_2_side": str((teams[1] if len(teams) > 1 else {}).get("side") or ""),
                    "team_2_champions": json.dumps(
                        (teams[1] if len(teams) > 1 else {}).get("champions") or [],
                        separators=(",", ":"),
                    ),
                    "team_2_players": json.dumps(
                        (teams[1] if len(teams) > 1 else {}).get("players") or [],
                        separators=(",", ":"),
                    ),
                    "team_2_player_ids": json.dumps(
                        (teams[1] if len(teams) > 1 else {}).get("playerIds") or [],
                        separators=(",", ":"),
                    ),
                }
            )
            for event in game.get("dragonEvents") or []:
                owner_team_id = str(event.get("ownerTeamId") or "")
                owner_side = next(
                    (
                        str(team.get("side") or "")
                        for team in teams
                        if str(team.get("id") or "") == owner_team_id
                    ),
                    str(event.get("ownerSide") or ""),
                )
                event_rows.append(
                    {
                        "series_id": str(series.get("id") or ""),
                        "game_id": str(game.get("id") or ""),
                        "date": series.get("date"),
                        "tournament": str(series.get("tournament") or ""),
                        "competition": competition["competition"],
                        "league": competition["league"],
                        "region": competition["region"],
                        "competition_level": competition["level"],
                        "occurred_at": event.get("occurredAt"),
                        "global_index": _as_int(event.get("globalIndex")),
                        "owner_stack": _as_int(event.get("ownerStack")),
                        "element": str(event.get("element") or ""),
                        "time_seconds": _as_int(event.get("timeSeconds")),
                        "owner_team_id": owner_team_id,
                        "owner_side": owner_side,
                        "state_timing": str(event.get("stateTiming") or ""),
                        "state_lag_seconds": _as_float(event.get("stateLagSeconds")),
                        "owner_net_worth": _as_int(event.get("ownerNetWorth")),
                        "opponent_net_worth": _as_int(event.get("opponentNetWorth")),
                        "gold_diff": _as_int(event.get("goldDiff")),
                        "owner_loadout_value": _as_int(event.get("ownerLoadoutValue")),
                        "opponent_loadout_value": _as_int(
                            event.get("opponentLoadoutValue")
                        ),
                        "loadout_diff": _as_int(event.get("loadoutDiff")),
                        "owner_unspent_money": _as_int(event.get("ownerUnspentMoney")),
                        "opponent_unspent_money": _as_int(
                            event.get("opponentUnspentMoney")
                        ),
                        "unspent_money_diff": _as_int(event.get("unspentMoneyDiff")),
                        "owner_top_player_net_worth": _as_int(
                            event.get("ownerTopPlayerNetWorth")
                        ),
                        "opponent_top_player_net_worth": _as_int(
                            event.get("opponentTopPlayerNetWorth")
                        ),
                        "top_player_net_worth_diff": _as_int(
                            event.get("topPlayerNetWorthDiff")
                        ),
                        "owner_kills": _as_int(event.get("ownerKills")),
                        "opponent_kills": _as_int(event.get("opponentKills")),
                        "owner_towers": _as_int(event.get("ownerTowers")),
                        "opponent_towers": _as_int(event.get("opponentTowers")),
                    }
                )
    COMPACT_GRID_DIR.mkdir(parents=True, exist_ok=True)
    game_df = pd.DataFrame(game_rows)
    event_df = pd.DataFrame(event_rows)
    if not game_df.empty:
        game_df = game_df.drop_duplicates(["series_id", "game_id"])
    if not event_df.empty:
        event_df = event_df.drop_duplicates(
            ["series_id", "game_id", "global_index"]
        )
    if not game_df.empty:
        game_df.to_parquet(COMPACT_GAMES_PARQUET, index=False, compression="zstd")
    if not event_df.empty:
        event_df.to_parquet(COMPACT_EVENTS_PARQUET, index=False, compression="zstd")
    return {
        "games": len(game_df),
        "events": len(event_df),
        "gameBytes": COMPACT_GAMES_PARQUET.stat().st_size if COMPACT_GAMES_PARQUET.exists() else 0,
        "eventBytes": (
            COMPACT_EVENTS_PARQUET.stat().st_size if COMPACT_EVENTS_PARQUET.exists() else 0
        ),
    }


def _series_rows_fast(
    key: str,
    start: str,
    end: str,
    limit: int,
) -> list[dict[str, Any]]:
    """Page Central Data with curl so long historical discovery is retryable."""
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    query = f"""
    query ($after: Cursor) {{
      allSeries(
        first: 50,
        after: $after,
        filter: {{
          titleId: {LOL_TITLE_ID},
          type: {ALLOWED_SERIES_TYPE},
          startTimeScheduled: {{ gte: "{start}", lte: "{end}" }}
        }},
        orderBy: StartTimeScheduled,
        orderDirection: DESC
      ) {{
        pageInfo {{ hasNextPage, endCursor }}
        edges {{
          node {{
            id
            type
            startTimeScheduled
            tournament {{ id name }}
            teams {{ baseInfo {{ name }} }}
          }}
        }}
      }}
    }}
    """
    with tempfile.TemporaryDirectory(prefix="grid-series-discovery-") as temp_dir:
        temp = Path(temp_dir)
        config_path = temp / "curl.conf"
        payload_path = temp / "payload.json"
        output_path = temp / "response.json"
        config_path.write_text(
            "\n".join(
                [
                    f'header = "x-api-key: {key}"',
                    'header = "Content-Type: application/json"',
                    'header = "Accept: application/json"',
                ]
            ),
            encoding="utf-8",
        )
        while len(rows) < limit:
            payload_path.write_text(
                json.dumps({"query": query, "variables": {"after": cursor}}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    "curl",
                    "--config",
                    str(config_path),
                    "--fail",
                    "--silent",
                    "--show-error",
                    "--retry",
                    "3",
                    "--retry-all-errors",
                    "--connect-timeout",
                    "20",
                    "--max-time",
                    "90",
                    "--request",
                    "POST",
                    "--data-binary",
                    f"@{payload_path}",
                    "--output",
                    str(output_path),
                    GRAPHQL_ENDPOINT,
                ],
                text=True,
                capture_output=True,
                timeout=120,
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"GRID series discovery failed after retries: {result.stderr[:240]}"
                )
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            if payload.get("errors"):
                raise RuntimeError(f"GRID GraphQL errors: {payload['errors']}")
            block = ((payload.get("data") or {}).get("allSeries") or {})
            edges = block.get("edges") or []
            if not edges:
                break
            for edge in edges:
                node = (edge or {}).get("node") or {}
                tournament = node.get("tournament") or {}
                tournament_name = str(tournament.get("name") or "").strip()
                teams = [
                    str(team.get("baseInfo", {}).get("name") or "").strip()
                    for team in node.get("teams") or []
                    if isinstance(team, Mapping)
                    and isinstance(team.get("baseInfo"), Mapping)
                ]
                teams = [team for team in teams if team]
                _assert_pro(
                    node.get("id"),
                    tournament_name,
                    teams,
                    context="GRID series discovery",
                )
                if (
                    str(node.get("type") or "").upper() != ALLOWED_SERIES_TYPE
                    or not tournament_name
                    or len(teams) < 2
                ):
                    continue
                rows.append(
                    {
                        "id": str(node.get("id") or ""),
                        "type": ALLOWED_SERIES_TYPE,
                        "date": node.get("startTimeScheduled"),
                        "tournament_id": str(tournament.get("id") or ""),
                        "tournament": tournament_name,
                        "teams": teams,
                    }
                )
                if len(rows) >= limit:
                    break
            if len(rows) and len(rows) % 500 == 0:
                print(
                    f"[elemental-drakes] discovered_series={len(rows)}/{limit}",
                    flush=True,
                )
            page_info = block.get("pageInfo") or {}
            if not page_info.get("hasNextPage"):
                break
            cursor = page_info.get("endCursor")
            if not cursor:
                break
    return rows[:limit]


def download_normalized_grid(
    *,
    days: int,
    series_limit: int,
    target_games: int,
    workers: int,
    env_file: Path | None,
) -> dict[str, Any]:
    """Build a resumable relevant-column store from compact GRID archives."""
    key = _api_key(env_file)
    now = datetime.now(timezone.utc)
    lower_bound = now - timedelta(days=max(days, 1))
    upper_bound = now + timedelta(hours=2)
    start = lower_bound.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = upper_bound.strftime("%Y-%m-%dT%H:%M:%SZ")
    series_rows: list[dict[str, Any]] = []
    try:
        catalog = json.loads(SERIES_CATALOG_PATH.read_text(encoding="utf-8"))
        catalog_rows = catalog.get("series") or []
        if (
            catalog.get("days") == days
            and catalog.get("seriesLimit") == series_limit
            and len(catalog_rows) >= max(series_limit, 1)
        ):
            series_rows = [
                dict(row) for row in catalog_rows if isinstance(row, Mapping)
            ][:series_limit]
            print(
                f"[elemental-drakes] loaded_series_catalog={len(series_rows)}",
                flush=True,
            )
    except (OSError, json.JSONDecodeError):
        pass
    if not series_rows:
        seen_series: set[str] = set()
        window_end = upper_bound
        while window_end > lower_bound and len(series_rows) < max(series_limit, 1):
            window_start = max(lower_bound, window_end - timedelta(days=120))
            remaining = max(series_limit, 1) - len(series_rows)
            discovered = _series_rows_fast(
                key,
                window_start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                window_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                remaining,
            )
            for series in discovered:
                series_id = str(series.get("id") or "")
                if series_id and series_id not in seen_series:
                    seen_series.add(series_id)
                    series_rows.append(series)
            print(
                f"[elemental-drakes] discovery_window={window_start.date()}.."
                f"{window_end.date()} total_series={len(series_rows)}",
                flush=True,
            )
            window_end = window_start - timedelta(seconds=1)
        COMPACT_GRID_DIR.mkdir(parents=True, exist_ok=True)
        SERIES_CATALOG_PATH.write_text(
            json.dumps(
                {
                    "generatedAt": datetime.now(timezone.utc).isoformat(),
                    "days": days,
                    "seriesLimit": series_limit,
                    "series": series_rows,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
    total_games = sum(
        _read_compact_count(path)
        for path in COMPACT_SERIES_DIR.glob("series_*.json")
    )
    status_counts: Counter[str] = Counter()
    attempted = 0
    compact_bytes = sum(
        path.stat().st_size for path in COMPACT_SERIES_DIR.glob("series_*.json")
    )
    series_iterator = iter(series_rows)
    worker_count = max(1, workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures: set[concurrent.futures.Future[dict[str, Any]]] = set()
        for _ in range(min(worker_count, len(series_rows))):
            series = next(series_iterator, None)
            if series is not None:
                futures.add(pool.submit(_extract_one_series, series, key=key))
        while futures and total_games < target_games:
            done, futures = concurrent.futures.wait(
                futures,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                result = future.result()
                attempted += 1
                status_counts[result["status"]] += 1
                if result["status"] == "extracted":
                    total_games += _as_int(result.get("games"))
                    compact_bytes += _as_int(result.get("bytes"))
                if attempted % 20 == 0 or result["status"] == "failed":
                    message = (
                        f"[elemental-drakes] series={attempted}/{len(series_rows)} "
                        f"games={total_games}/{target_games} "
                        f"status={result['status']}"
                    )
                    if result.get("error"):
                        message += f" error={result['error']}"
                    print(message, flush=True)
                if total_games < target_games:
                    series = next(series_iterator, None)
                    if series is not None:
                        futures.add(
                            pool.submit(_extract_one_series, series, key=key)
                        )
    parquet = consolidate_compact_parquet()
    return {
        "window": {"start": start, "end": end},
        "seriesSeen": len(series_rows),
        "seriesAttempted": attempted,
        "targetGames": target_games,
        "gamesExtracted": total_games,
        "statuses": dict(status_counts),
        "compactBytes": compact_bytes,
        "parquet": parquet,
    }


def build_artifact(
    *,
    raw_dir: Path = RAW_GRID_DIR,
    download_meta: Mapping[str, Any] | None = None,
    explorer_model_path: Path = DEFAULT_EXPLORER_MODEL_OUTPUT,
    audit_path: Path = DEFAULT_AUDIT_OUTPUT,
) -> dict[str, Any]:
    cohort_games = parse_normalized_cohort(raw_dir)
    pilot = _prepare_pilot_games(parse_raw_pilot(raw_dir))
    _reconcile_pilot_names(pilot, cohort_games)
    cohort = summarize_cohort(cohort_games)
    audit: dict[str, Any] = {"status": "not-run"}
    try:
        candidate = json.loads(audit_path.read_text(encoding="utf-8"))
        if isinstance(candidate, Mapping):
            audit = dict(candidate)
    except (OSError, json.JSONDecodeError):
        pass
    explorer_model: dict[str, Any] = {
        "status": "gated",
        "games": cohort["completeGames"],
        "requiredGames": 6_000,
        "reason": "The joint 5v5 explorer model has not been generated.",
    }
    explorer_model_source: dict[str, Any] | None = None
    try:
        explorer_model_bytes = explorer_model_path.read_bytes()
        candidate = json.loads(explorer_model_bytes)
        if isinstance(candidate, Mapping):
            explorer_model = dict(candidate)
            explorer_model_source = {
                "file": explorer_model_path.name,
                "bytes": len(explorer_model_bytes),
                "sha256": hashlib.sha256(explorer_model_bytes).hexdigest(),
                "schemaVersion": str(candidate.get("schemaVersion") or ""),
            }
    except (OSError, json.JSONDecodeError):
        pass
    cohort["outcomeModel"] = {
        "status": str(explorer_model.get("status") or "gated"),
        "reason": str(
            explorer_model.get("reason")
            or (
                "The joint inventory and resolved-allocation layers are available. "
                "Both remain associational because captures are selected, not randomized."
            )
        ),
    }
    patches = sorted(
        {game.get("patch") for game in pilot if game.get("patch")},
        reverse=True,
    )
    raw_events = sum(len(game.get("dragonEvents") or []) for game in pilot)
    public_pilot = load_tier_one_snapshots()
    if not public_pilot:
        public_pilot = _public_pilot_games(pilot)
    if public_pilot:
        patches = sorted(
            {
                str(game.get("patch") or "")
                for game in public_pilot
                if str(game.get("patch") or "")
            },
            reverse=True,
        )
        raw_events = sum(
            len(game.get("dragonEvents") or []) for game in public_pilot
        )
    app_pilot = [
        {
            key: value
            for key, value in game.items()
            if key != "dragonEvents"
        }
        for game in public_pilot
    ]
    public_audit = _public_audit(audit)
    return {
        "metadata": {
            "title": "What is a dragon worth to these five?",
            "generatedAt": datetime.now(timezone.utc).isoformat(),
            "patches": patches,
            "provider": "GRID Open Platform, derived pro telemetry",
            "rawFilesPublished": False,
            "pilotGames": len(app_pilot),
            "pilotDragonEvents": raw_events,
            "estimationStatus": str(explorer_model.get("status") or "gated"),
            "explorerModelSource": explorer_model_source,
            "download": dict(download_meta or {}),
        },
        "mechanics": MECHANICS,
        "cohort": cohort,
        "competitionCoverage": summarize_competition_coverage(cohort_games),
        "roleCatalog": load_role_catalog(),
        "audit": public_audit,
        "explorerModel": explorer_model,
        "pilotGames": app_pilot,
        "method": {
            "mechanics": (
                "Per-stack effects are deterministic game rules sourced from Riot patch notes. "
                "Stack displays are arithmetic illustrations, not win-probability claims."
            ),
            "composition": (
                "The common dragon effect is reported once per team. Champion lines show "
                "regularized differences above it. Supported champion-element estimates pool "
                "evidence across many pro teams and games and shrink toward the pooled and "
                "archetype terms; otherwise the archetype prior or common team effect remains. "
                "The expanded family was frozen through June and checked on July games; cells "
                "entering the final refit afterward are not individually holdout-validated. This "
                "is never an exact-five lookup, and every elemental buff still applies to all "
                "five living team members."
            ),
            "roles": (
                "Role-aware selection and randomization use observed champion-role appearances from "
                "a separate 1,194-game Leaguepedia pro-draft corpus. Role labels constrain the picker; "
                "the dragon model pools champion-element evidence across roles because the "
                "compact GRID training compositions do not carry a validated role assignment."
            ),
            "gameState": (
                "Tier 1 examples retain only role-ordered champions and the compact observed capture "
                "history projected while streaming GRID telemetry; source archives are discarded. "
                "The normalized model cohort keeps pre-capture net worth, loadout value, unspent "
                "gold, leading-player net worth, towers, side, patch, and state lag."
            ),
            "outcome": (
                "The joint two-team inventory model conditions on pre-capture state, side, both "
                "drafts, prior organization strength, and prior five-player strength. The separate "
                "resolved-capture allocation comparison asks which team receives an observed drake "
                "while holding its recorded pre-capture state fixed. Both are chronological "
                "holdout associations: neither identifies a true strategic contest-versus-leave "
                "policy effect."
                if explorer_model.get("status") == "ready"
                else
                "No observed win-rate lift is reported from the pilot. The outcome layer stays "
                "gated until at least 6,000 completed pro games support state, side, draft, and "
                "prior team-strength controls with held-out evaluation and uncertainty."
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-normalized", action="store_true")
    parser.add_argument("--days", type=int, default=1_320)
    parser.add_argument("--series-limit", type=int, default=6_000)
    parser.add_argument("--target-games", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--grid-env-file", type=Path, default=None)
    parser.add_argument("--refresh-tier-one-snapshots", action="store_true")
    parser.add_argument(
        "--snapshots-output",
        type=Path,
        default=DEFAULT_SNAPSHOTS_OUTPUT,
    )
    parser.add_argument(
        "--explorer-model",
        type=Path,
        default=DEFAULT_EXPLORER_MODEL_OUTPUT,
    )
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented JSON instead of the compact deployable artifact.",
    )
    args = parser.parse_args(argv)

    download_meta: Mapping[str, Any] | None = None
    if args.download_normalized:
        download_meta = download_normalized_grid(
            days=args.days,
            series_limit=args.series_limit,
            target_games=args.target_games,
            workers=args.workers,
            env_file=args.grid_env_file,
        )
    if args.refresh_tier_one_snapshots:
        refresh_tier_one_snapshots(
            output=args.snapshots_output,
            env_file=args.grid_env_file,
        )
    artifact = build_artifact(
        download_meta=download_meta,
        explorer_model_path=args.explorer_model,
        audit_path=args.audit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(artifact, indent=2)
        if args.pretty
        else json.dumps(artifact, separators=(",", ":"))
    )
    args.output.write_text(serialized + "\n", encoding="utf-8")
    print(
        f"[elemental-drakes] pilot_games={artifact['metadata']['pilotGames']} "
        f"cohort_games={artifact['cohort']['games']} "
        f"drake_events={artifact['cohort']['dragonEvents']} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
