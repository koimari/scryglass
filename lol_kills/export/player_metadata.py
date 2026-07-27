"""Small, source-backed player metadata artifacts for the public pack."""

from __future__ import annotations

import json
import unicodedata
import urllib.parse
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from lol_kills.etl.paths import PARQUET_DIR

LEAGUEPEDIA_COUNTRY_URL = "https://lol.fandom.com/wiki/Special:CargoExport"

# Leaguepedia's nationality/country fields are human-readable names.  Keep the
# conversion local so the frontend has no external image/font dependency.
COUNTRY_CODES = {
    "Argentina": "AR", "Australia": "AU", "Austria": "AT", "Belgium": "BE",
    "Bolivia": "BO", "Brazil": "BR", "Canada": "CA", "Chile": "CL",
    "China": "CN", "Colombia": "CO", "Costa Rica": "CR", "Croatia": "HR",
    "Czechia": "CZ", "Czech Republic": "CZ", "Denmark": "DK", "Ecuador": "EC",
    "Egypt": "EG", "Estonia": "EE", "Finland": "FI", "France": "FR",
    "Germany": "DE", "Greece": "GR", "Hong Kong": "HK", "Hungary": "HU",
    "Iceland": "IS", "India": "IN", "Indonesia": "ID", "Ireland": "IE",
    "Israel": "IL", "Italy": "IT", "Japan": "JP", "Jordan": "JO",
    "Kazakhstan": "KZ", "Lebanon": "LB", "Latvia": "LV", "Lithuania": "LT",
    "Malaysia": "MY", "Mexico": "MX", "Mongolia": "MN", "Morocco": "MA",
    "Nepal": "NP", "Netherlands": "NL", "New Zealand": "NZ", "Norway": "NO",
    "Pakistan": "PK", "Panama": "PA", "Peru": "PE", "Philippines": "PH",
    "Poland": "PL", "Portugal": "PT", "Puerto Rico": "PR", "Romania": "RO",
    "Russia": "RU", "Saudi Arabia": "SA", "Serbia": "RS", "Singapore": "SG",
    "Slovakia": "SK", "Slovenia": "SI", "South Africa": "ZA", "South Korea": "KR",
    "Spain": "ES", "Sweden": "SE", "Switzerland": "CH", "Taiwan": "TW",
    "Thailand": "TH", "Tunisia": "TN", "Turkey": "TR", "Türkiye": "TR",
    "Ukraine": "UA", "United Arab Emirates": "AE", "United Kingdom": "GB",
    "United States": "US", "Uruguay": "UY", "Uzbekistan": "UZ", "Venezuela": "VE",
    "Vietnam": "VN",
}


def _key(value: Any) -> str:
    text = "" if value is None else str(value)
    # Leaguepedia sometimes stores a disambiguator in the page title while
    # Oracle's Elixir stores only the player ID shown in match data.
    text = text.split("(", 1)[0].strip()
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _flag(code: str | None) -> str | None:
    if not code or len(code) != 2 or not code.isalpha():
        return None
    return "".join(chr(127397 + ord(letter.upper())) for letter in code)


def _country_code(value: Any) -> str | None:
    if isinstance(value, list):
        value = next((item for item in value if str(item).strip()), None)
    text = str(value or "").strip()
    if not text:
        return None
    if len(text) == 2 and text.isalpha():
        return text.upper()
    return COUNTRY_CODES.get(text)


def _provider_identity_collisions(
    player_identities: pd.DataFrame | Iterable[Any] | None,
) -> set[str]:
    """Return name keys that map to multiple stable provider IDs."""

    if player_identities is None:
        source = PARQUET_DIR / "players.parquet"
        if not source.exists():
            return set()
        try:
            frame = pd.read_parquet(
                source, columns=["playername", "playerid", "position"]
            )
        except (OSError, ValueError, KeyError):
            try:
                frame = pd.read_parquet(
                    source, columns=["playername", "playerid"]
                )
            except (OSError, ValueError, KeyError):
                return set()
    elif isinstance(player_identities, pd.DataFrame):
        frame = player_identities.copy()
    else:
        records: list[dict[str, Any]] = []
        for value in player_identities:
            if isinstance(value, dict):
                records.append(
                    {
                        "playername": value.get("playername")
                        or value.get("player"),
                        "playerid": value.get("playerid")
                        or value.get("player_id"),
                    }
                )
            elif isinstance(value, (tuple, list)) and len(value) >= 2:
                records.append(
                    {"playername": value[0], "playerid": value[1]}
                )
        frame = pd.DataFrame(records)
    if not {"playername", "playerid"}.issubset(frame.columns):
        return set()
    if "position" in frame.columns:
        frame = frame[
            frame["position"].astype(str).str.casefold().ne("team")
        ]
    frame = frame.assign(
        _name_key=frame["playername"].map(_key),
        _player_id=frame["playerid"].map(
            lambda value: (
                ""
                if value is None or pd.isna(value)
                else str(value).strip()
            )
        ),
    )
    ids_by_name = (
        frame.loc[
            frame["_name_key"].ne("") & frame["_player_id"].ne("")
        ]
        .groupby("_name_key", sort=True)["_player_id"]
        .agg(lambda values: {str(value) for value in values})
    )
    return set(ids_by_name[ids_by_name.map(len).gt(1)].index)


def _fetch_rows(cache_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset in range(0, 50_000, 5_000):
        query = urllib.parse.urlencode(
            {
                "tables": "Players",
                "fields": "Player,Country,NationalityPrimary,Team,CurrentTeams,Residency",
                "format": "json",
                "limit": "5000",
                "offset": str(offset),
            }
        )
        request = urllib.request.Request(
            f"{LEAGUEPEDIA_COUNTRY_URL}?{query}",
            headers={"User-Agent": "Scryglass public pack/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                page = json.load(response)
        except (OSError, TimeoutError, UnicodeError, json.JSONDecodeError):
            break
        if not isinstance(page, list):
            break
        rows.extend(row for row in page if isinstance(row, dict))
        if len(page) < 5_000:
            break
    if rows:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return rows


def build_player_metadata(
    player_names: Iterable[str],
    *,
    cache_path: Path | None = None,
    player_context: dict[str, str | None] | None = None,
    player_identities: pd.DataFrame | Iterable[Any] | None = None,
) -> dict[str, dict[str, str]]:
    """Return unambiguous country/flag metadata backed by source identities."""

    wanted = [str(name) for name in player_names if str(name).strip()]
    if not wanted:
        return {}
    cache = cache_path or (PARQUET_DIR / "leaguepedia_players_v2.json")
    rows: list[dict[str, Any]] = []
    if cache.exists():
        try:
            value = json.loads(cache.read_text(encoding="utf-8"))
            rows = value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            rows = []
    if not rows:
        rows = _fetch_rows(cache)
    colliding_name_keys = _provider_identity_collisions(player_identities)

    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = row.get("Player")
        if name is not None:
            by_name.setdefault(_key(name), []).append(row)

    metadata: dict[str, dict[str, str]] = {}
    for name in wanted:
        if _key(name) in colliding_name_keys:
            continue
        candidates = by_name.get(_key(name), [])
        if not candidates:
            continue
        context_team = str((player_context or {}).get(name) or "").strip()
        if context_team and len(candidates) > 1:
            context_key = _key(context_team)
            context_matches = [
                row
                for row in candidates
                if context_key
                and any(
                    context_key in _key(row.get(field))
                    for field in ("Team", "CurrentTeams")
                )
            ]
            if context_matches:
                candidates = context_matches
        resolved: dict[tuple[str, str], dict[str, Any]] = {}
        for row in candidates:
            country = str(
                row.get("NationalityPrimary") or row.get("Country") or ""
            ).strip()
            code = _country_code(country)
            if code:
                resolved[(country, code)] = row
        if len(resolved) != 1:
            continue
        country, code = next(iter(resolved))
        item: dict[str, str] = {"country": country, "country_code": code}
        flag = _flag(code)
        if flag:
            item["flag"] = flag
        metadata[name] = item
    return metadata
