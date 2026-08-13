"""Small, source-backed player metadata artifacts for the public pack."""

from __future__ import annotations

import json
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Iterable

from lol_kills.etl.paths import PARQUET_DIR
from lol_kills.net import require_https_url


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
        url = require_https_url(
            f"{LEAGUEPEDIA_COUNTRY_URL}?{query}", hosts={"lol.fandom.com"}
        )
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Scryglass public pack/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                page = json.load(response)
        except Exception:
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
) -> dict[str, dict[str, str]]:
    """Return country/flag metadata only where the source has a country."""

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

    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        name = row.get("Player")
        if name is not None:
            by_name.setdefault(_key(name), []).append(row)

    metadata: dict[str, dict[str, str]] = {}
    for name in wanted:
        candidates = by_name.get(_key(name), [])
        if not candidates:
            continue
        context_team = str((player_context or {}).get(name) or "").strip()
        if context_team and len(candidates) > 1:
            context_key = _key(context_team)
            candidates = sorted(
                candidates,
                key=lambda row: (
                    0 if context_key and context_key in _key(row.get("Team")) else 1,
                    _key(row.get("Player")),
                ),
            )
        row = candidates[0]
        country = str(row.get("NationalityPrimary") or row.get("Country") or "").strip()
        code = _country_code(country)
        if not code:
            continue
        item: dict[str, str] = {"country": country, "country_code": code}
        flag = _flag(code)
        if flag:
            item["flag"] = flag
        metadata[name] = item
    return metadata
