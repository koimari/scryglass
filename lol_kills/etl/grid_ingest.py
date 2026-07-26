"""Recent pro-series intake from GRID for the public ratings refresh.

GRID is used here as a freshness bridge for completed professional games that
have not reached the next Oracle's Elixir export yet. It is deliberately
conservative:

* only LoL ``ESPORTS`` series are queried;
* scrim/practice/private-looking series are rejected;
* a game is emitted only when a completed Riot live-stats file contains both
  ``game_info`` and ``game_end``;
* team-side resolution must be unambiguous; otherwise the game is skipped.

The resulting rows use the same small identity fields as the OE warehouse and
are merged with OE by ``gameid`` with OE taking precedence when both sources
eventually contain the same game.

Local use may read ``GRID_API_KEY`` from the environment, ``.env`` in this
repo, or the sibling lol-strength-analysis ``.env`` when it exists. CI should
provide ``GRID_API_KEY`` as a secret instead.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pandas as pd

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.paths import PARQUET_DIR, WAREHOUSE_DIR

GRAPHQL_ENDPOINT = "https://api.grid.gg/central-data/graphql"
SERIES_ENDPOINT = "https://api.grid.gg/live-data-feed/series-state/graphql"
FILE_LIST_BASE = "https://api.grid.gg/file-download/list"
RAW_GRID_DIR = WAREHOUSE_DIR / "raw_grid"

LOL_TITLE_ID = 3
ALLOWED_SERIES_TYPE = "ESPORTS"
USER_AGENT = "scryglass/grid-ingest (+pro-only; research publication)"
GRID_429_RETRIES = 1
GRID_429_MAX_DELAY_SECONDS = 30
GRID_FILE_LIST_TIMEOUT_SECONDS = 30
GRID_FILE_DOWNLOAD_TIMEOUT_SECONDS = 90

SCRIM_MARKERS = (
    "scrim",
    "scrims",
    "scrimmage",
    "practice",
    "tryout",
    "tryouts",
    "private scrim",
)

# Common pro tags used in Riot display names. This is only a resolver aid; the
# canonical team name still comes from GRID central data and aliases.py.
TEAM_TAGS: dict[str, set[str]] = {
    "Gen.G": {"gen", "geng", "gen.g"},
    "T1": {"t1", "skt"},
    "Dplus Kia": {"dk", "dplus", "dwg"},
    "Nongshim RedForce": {"ns", "nongshim"},
    "Hanwha Life Esports": {"hle", "hanwha"},
    "KT Rolster": {"kt", "ktr"},
    "BRION": {"bro", "brion"},
    "BNK FEARX": {"bfx", "fearx"},
    "G2 Esports": {"g2"},
    "Fnatic": {"fnc", "fnatic"},
    "Team Liquid": {"tl", "liquid"},
    "Cloud9": {"c9", "cloud9"},
    "FlyQuest": {"fly", "flyquest"},
    "Karmine Corp": {"kc", "kcorp", "karmine"},
    "Movistar KOI": {"m koi", "mkoi", "koi"},
    "Bilibili Gaming": {"blg", "bilibili"},
    "JD Gaming": {"jdg", "jd"},
    "Top Esports": {"tes", "top"},
    "Weibo Gaming": {"wbg", "weibo"},
    "LNG Esports": {"lng"},
    "Anyone's Legend": {"al", "anyone"},
    "Edward Gaming": {"edg", "edward"},
    "Invictus Gaming": {"ig", "invictus"},
    "FunPlus Phoenix": {"fpx", "funplus"},
    "Rare Atom": {"ra"},
    "ThunderTalk Gaming": {"tt", "thundertalk"},
    "Ultra Prime": {"up", "ultra"},
    "LGD Gaming": {"lgd"},
    "Team WE": {"we"},
    "Oh My God": {"omg"},
    "Royal Never Give Up": {"rng", "royal"},
    "Ninjas in Pyjamas": {"nip"},
    "DetonatioN FocusMe": {"dfm", "detonation"},
    "CTBC Flying Oyster": {"cfo"},
    "Fukuoka SoftBank HAWKS gaming": {"shg", "hawks"},
    "Team Secret Whales": {"tsw"},
    "Relove Deep Cross Gaming": {"rdcg", "dcg"},
    "Team Vitality": {"vit", "vitality"},
    "Pain Gaming": {"png", "pain"},
    "Arneb": {"arb"},
    "RAYN Clocks": {"rck"},
}


class GridIngestError(RuntimeError):
    """GRID request, auth, or canonicalization failure."""


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def _api_key(env_file: Path | None = None) -> str:
    if env_file:
        _load_dotenv(env_file.expanduser())
    candidates = [
        Path(os.environ["GRID_ENV_FILE"]).expanduser()
        if os.environ.get("GRID_ENV_FILE")
        else None,
        Path(__file__).resolve().parents[2] / ".env",
        Path(__file__).resolve().parents[3] / "Projects" / "lol-strength-analysis" / ".env",
    ]
    for candidate in candidates:
        if candidate:
            _load_dotenv(candidate)
    key = (os.environ.get("GRID_API_KEY") or "").strip()
    if not key:
        raise GridIngestError(
            "GRID_API_KEY is required for GRID refresh; provide it as an environment "
            "secret or use --grid-env-file."
        )
    return key


def _looks_like_scrim(*parts: Any) -> bool:
    for part in parts:
        if part is None:
            continue
        if isinstance(part, Mapping):
            if _looks_like_scrim(*part.values()):
                return True
            continue
        if isinstance(part, (list, tuple, set)):
            if _looks_like_scrim(*part):
                return True
            continue
        text = str(part).strip().lower()
        if any(marker in text for marker in SCRIM_MARKERS):
            return True
    return False


def _assert_pro(*parts: Any, context: str) -> None:
    if _looks_like_scrim(*parts):
        raise GridIngestError(
            f"{context}: blocked because the GRID key is restricted to professional series"
        )


def _headers(key: str, *, json_body: bool = False) -> dict[str, str]:
    headers = {
        "x-api-key": key,
        "Accept": "application/json,application/octet-stream,*/*",
        "User-Agent": USER_AGENT,
    }
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _request_json(
    url: str,
    key: str,
    *,
    method: str = "GET",
    body: Mapping[str, Any] | None = None,
    timeout: float = 60,
) -> dict[str, Any]:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        headers=_headers(key, json_body=body is not None),
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise GridIngestError(f"GRID HTTP {exc.code} for {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise GridIngestError(f"GRID network error for {url}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GridIngestError(f"GRID returned non-object JSON from {url}")
    return payload


def _graphql(key: str, endpoint: str, query: str, variables: Mapping[str, Any]) -> dict[str, Any]:
    payload = _request_json(
        endpoint,
        key,
        method="POST",
        body={"query": query, "variables": dict(variables)},
    )
    if payload.get("errors"):
        raise GridIngestError(f"GRID GraphQL errors: {payload['errors']}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise GridIngestError("GRID GraphQL response missing data")
    return data


def _series_rows(key: str, start: str, end: str, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor: str | None = None
    while len(rows) < limit:
        query = f"""
        query ($after: Cursor) {{
          allSeries(
            first: 50,
            after: $after,
            filter: {{
              titleId: {LOL_TITLE_ID},
              type: {ALLOWED_SERIES_TYPE},
              startTimeScheduled: {{ gte: \"{start}\", lte: \"{end}\" }}
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
        data = _graphql(key, GRAPHQL_ENDPOINT, query, {"after": cursor})
        block = data.get("allSeries") or {}
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
                if isinstance(team, Mapping) and isinstance(team.get("baseInfo"), Mapping)
            ]
            teams = [team for team in teams if team]
            _assert_pro(node.get("id"), tournament_name, teams, context="GRID series discovery")
            if str(node.get("type") or "").upper() != ALLOWED_SERIES_TYPE:
                continue
            if not tournament_name or len(teams) < 2:
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
        page_info = block.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            break
        cursor = page_info.get("endCursor")
        if not cursor:
            break
    return rows[:limit]


def _series_games(key: str, series_id: str) -> list[dict[str, Any]]:
    _assert_pro(series_id, context="GRID series metadata")
    query = """
    query ($id: ID!) {
      seriesState(id: $id) {
        games { id teams { name } }
      }
    }
    """
    data = _graphql(key, SERIES_ENDPOINT, query, {"id": str(series_id)})
    games = (data.get("seriesState") or {}).get("games") or []
    out = []
    for game in games:
        if not isinstance(game, Mapping):
            continue
        teams = [
            str(team.get("name") or "").strip()
            for team in game.get("teams") or []
            if isinstance(team, Mapping)
        ]
        out.append({"id": str(game.get("id") or ""), "teams": [x for x in teams if x]})
    return out


def _file_list(key: str, series_id: str) -> list[dict[str, Any]]:
    payload = _request_json(
        f"{FILE_LIST_BASE}/{series_id}",
        key,
        timeout=GRID_FILE_LIST_TIMEOUT_SECONDS,
    )
    files = payload.get("files")
    if not isinstance(files, list):
        raise GridIngestError(f"GRID file list for {series_id} has no files[]")
    return [dict(file) for file in files if isinstance(file, Mapping)]


def _safe_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "events.jsonl"


def _download(url: str, key: str, dest: Path) -> bool:
    req = urllib.request.Request(url, headers=_headers(key))
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(GRID_429_RETRIES + 1):
        try:
            with urllib.request.urlopen(
                req,
                timeout=GRID_FILE_DOWNLOAD_TIMEOUT_SECONDS,
            ) as response, dest.open("wb") as fh:
                while chunk := response.read(1024 * 1024):
                    fh.write(chunk)
            return True
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            dest.unlink(missing_ok=True)
            if exc.code != 429:
                raise GridIngestError(f"GRID file download HTTP {exc.code}: {detail}") from exc
            if attempt >= GRID_429_RETRIES:
                print(f"[grid] file download rate-limited after retries: {detail or 'HTTP 429'}")
                return False
            retry_after = None
            if exc.headers:
                try:
                    retry_after = float(exc.headers.get("Retry-After", ""))
                except (TypeError, ValueError):
                    retry_after = None
            delay = min(
                max(retry_after if retry_after is not None else 5 * (2**attempt), 1),
                GRID_429_MAX_DELAY_SECONDS,
            )
            print(f"[grid] file download HTTP 429; retrying in {delay:.0f}s")
            time.sleep(delay)
        except Exception:
            dest.unlink(missing_ok=True)
            raise
    return False


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [name for name in archive.namelist() if name.endswith(".jsonl")]
            if not names:
                raise GridIngestError(f"GRID archive has no JSONL: {path}")
            with archive.open(names[0]) as fh:
                for raw in fh:
                    line = raw.decode("utf-8").strip()
                    if line:
                        row = json.loads(line)
                        if isinstance(row, dict):
                            yield row
        return
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if line:
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def _parse_iso(value: Any) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _league_for(tournament: str) -> str:
    blob = tournament.upper()
    if "ESPORTS WORLD CUP" in blob or "EWC" in blob:
        return "EWC"
    if "MID-SEASON INVITATIONAL" in blob or re.search(r"\bMSI\b", blob):
        return "MSI"
    if "WORLD CHAMPIONSHIP" in blob or "WORLDS" in blob:
        return "WORLDS"
    if "FIRST STAND" in blob:
        return "FST"
    if re.search(r"\bLTA\s+N(?:ORTH)?\b", blob):
        return "LTA N"
    if re.search(r"\bLTA\s+S(?:OUTH)?\b", blob):
        return "LTA S"
    for league in ("LCK", "LPL", "LEC", "LCS", "LTA", "CBLOL", "PCS", "VCS", "LJL", "LCP", "TCL"):
        if league in blob:
            return league
    if "KESPA" in blob:
        return "LCK"
    return "INTL"


def _compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _player_prefix(player: Mapping[str, Any]) -> str:
    raw = str(
        (player.get("riotId") or {}).get("displayName")
        if isinstance(player.get("riotId"), Mapping)
        else player.get("summonerName") or ""
    ).strip()
    return raw.split()[0].lower() if raw.split() else ""


def _resolve_sides(
    participants: list[Mapping[str, Any]],
    candidates: Sequence[str],
) -> dict[int, str] | None:
    by_team: dict[int, list[str]] = {100: [], 200: []}
    for participant in participants:
        try:
            team_id = int(participant.get("teamID"))
        except (TypeError, ValueError):
            continue
        if team_id in by_team:
            prefix = _player_prefix(participant)
            if prefix:
                by_team[team_id].append(prefix)
    canonical = [normalize_team(name) for name in candidates]
    scores: dict[int, list[tuple[int, str]]] = {}
    for team_id, prefixes in by_team.items():
        scored = []
        for candidate in canonical:
            tags = {_compact(candidate).split("esports")[0]} | {
                _compact(tag) for tag in TEAM_TAGS.get(candidate, set())
            }
            score = sum(
                1
                for prefix in prefixes
                if any(_compact(prefix) == tag or _compact(prefix).startswith(tag) for tag in tags if tag)
            )
            # Candidate words are a weak but useful fallback for names such as
            # "NONGSHIM RED FORCE" paired with an "NS" player prefix.
            candidate_words = {_compact(word) for word in candidate.split() if len(_compact(word)) >= 3}
            score += sum(1 for prefix in prefixes if _compact(prefix) in candidate_words)
            scored.append((score, candidate))
        scored.sort(reverse=True)
        scores[team_id] = scored

    if any(not values or values[0][0] == 0 for values in scores.values()):
        return None
    if any(len(values) > 1 and values[0][0] == values[1][0] for values in scores.values()):
        return None
    out = {team_id: values[0][1] for team_id, values in scores.items()}
    if out.get(100) == out.get(200):
        return None
    return out


def _player_name(participant: Mapping[str, Any]) -> str:
    riot_id = participant.get("riotId")
    raw = (
        riot_id.get("displayName")
        if isinstance(riot_id, Mapping)
        else participant.get("summonerName")
    ) or participant.get("summonerName") or ""
    parts = str(raw).strip().split()
    # GRID Riot IDs commonly include the team tag ("GEN Kiin"). The final
    # token is the stable esports handle used by OE and player ratings.
    return parts[-1] if parts else "Unknown"


def _position(value: Any) -> str:
    """Map Riot live-stats role labels onto OE's compact role vocabulary."""
    raw = re.sub(r"[^a-z]", "", str(value or "").lower())
    return {
        "top": "top",
        "jungle": "jng",
        "jg": "jng",
        "middle": "mid",
        "mid": "mid",
        "bottom": "bot",
        "bot": "bot",
        "adc": "bot",
        "utility": "sup",
        "support": "sup",
        "sup": "sup",
    }.get(raw, raw or "unknown")


def _parse_events(
    path: Path,
    *,
    series: Mapping[str, Any],
    game_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]] | None:
    game_info: dict[str, Any] | None = None
    game_end: dict[str, Any] | None = None
    kills = {100: 0, 200: 0}
    for row in _iter_jsonl(path):
        schema = row.get("rfc461Schema")
        if schema == "game_info" and game_info is None:
            game_info = row
        elif schema == "game_end" and game_end is None:
            game_end = row
        elif schema == "champion_kill":
            try:
                team_id = int(row.get("killerTeamID"))
            except (TypeError, ValueError):
                team_id = 0
            if team_id in kills:
                kills[team_id] += 1
    if not game_info or not game_end or not game_info.get("participants"):
        return None

    participants = [p for p in game_info["participants"] if isinstance(p, Mapping)]
    sides = _resolve_sides(participants, series.get("teams") or [])
    if not sides:
        return None
    try:
        winner = int(game_end.get("winningTeam"))
        game_id = int(game_info.get("gameID"))
    except (TypeError, ValueError):
        return None
    platform_id = str(game_info.get("platformID") or "").strip()
    if not platform_id or winner not in (100, 200):
        return None
    date = _parse_iso(
        game_info.get("rfc460Timestamp")
        or game_info.get("repeater_timestamp")
        or game_end.get("rfc460Timestamp")
    )
    if not date:
        return None
    tournament = str(series.get("tournament") or "").strip()
    league = _league_for(tournament)
    game_uid = f"{platform_id}_{game_id}"
    base = {
        "gameid": game_uid,
        "game_uid": game_uid,
        "date": date,
        "year": int(date[:4]),
        "league": league,
        "tournament": tournament,
        "patch": str(game_info.get("gameVersion") or ""),
        "game": game_index,
        "gamelength": float(game_end.get("gameTime") or 0) / 1000.0,
        "source": "grid",
        "grid_series_id": str(series.get("id") or ""),
        "grid_game_id": str(game_info.get("gameName") or game_id),
        "grid_game_index": game_index,
    }
    team_rows = []
    for team_id, side in ((100, "Blue"), (200, "Red")):
        team = sides[team_id]
        team_rows.append(
            {
                **base,
                "position": "team",
                "side": side,
                "teamname": normalize_team(team),
                "team_id": team_id,
                "kills": kills[team_id],
                "result": 1 if winner == team_id else 0,
                "total_kills": kills[100] + kills[200],
                "length_min": base["gamelength"] / 60.0,
            }
        )

    player_rows = []
    for participant in participants:
        try:
            team_id = int(participant.get("teamID"))
        except (TypeError, ValueError):
            continue
        if team_id not in sides:
            continue
        player_rows.append(
            {
                **base,
                "position": _position(participant.get("role")),
                "side": "Blue" if team_id == 100 else "Red",
                "teamname": normalize_team(sides[team_id]),
                "team_id": team_id,
                "playername": _player_name(participant),
                "playername_raw": str(
                    ((participant.get("riotId") or {}).get("displayName"))
                    if isinstance(participant.get("riotId"), Mapping)
                    else participant.get("summonerName") or ""
                ),
                "champion": normalize_champ(str(participant.get("championName") or "Unknown")),
                "result": 1 if winner == team_id else 0,
            }
        )
    if len(team_rows) != 2 or len(player_rows) != 10:
        return None
    return {"team_rows": team_rows, "player_rows": player_rows}, team_rows


def _game_index_from_name(name: str, fallback: int) -> int:
    match = re.search(r"_(\d+)_riot", name)
    return int(match.group(1)) if match else fallback


def _parse_local_grid() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    teams: list[dict[str, Any]] = []
    players: list[dict[str, Any]] = []
    parsed_files = 0
    skipped_files = 0
    for path in sorted(RAW_GRID_DIR.glob("events_*_riot.jsonl")):
        match = re.match(r"events_(\d+)_\d+_riot\.jsonl$", path.name)
        if not match:
            skipped_files += 1
            continue
        series_id = match.group(1)
        meta_path = RAW_GRID_DIR / f"series_{series_id}.json"
        if not meta_path.exists():
            skipped_files += 1
            continue
        series = json.loads(meta_path.read_text(encoding="utf-8"))
        game_index = _game_index_from_name(path.name, 1)
        parsed = _parse_events(path, series=series, game_index=game_index)
        if parsed is None:
            skipped_files += 1
            continue
        payload, _ = parsed
        teams.extend(payload["team_rows"])
        players.extend(payload["player_rows"])
        parsed_files += 1
    team_df = pd.DataFrame(teams)
    player_df = pd.DataFrame(players)
    if not team_df.empty:
        team_df = team_df.drop_duplicates(["gameid", "side"], keep="last")
    if not player_df.empty:
        player_df = player_df.drop_duplicates(["gameid", "side", "position"], keep="last")
    meta = {
        "source": "grid",
        "parsed_files": parsed_files,
        "skipped_files": skipped_files,
        "games": int(team_df["gameid"].nunique()) if not team_df.empty else 0,
        "team_rows": int(len(team_df)),
        "player_rows": int(len(player_df)),
    }
    return team_df, player_df, meta


def _load_cached_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the last verified GRID parquet when raw event files are unavailable."""
    team_path = PARQUET_DIR / "grid_team_games.parquet"
    player_path = PARQUET_DIR / "grid_player_games.parquet"
    if not team_path.exists() or not player_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    try:
        return pd.read_parquet(team_path), pd.read_parquet(player_path)
    except Exception as exc:  # noqa: BLE001
        print(f"[grid] cached parquet could not be read: {exc}")
        return pd.DataFrame(), pd.DataFrame()


def _download_recent(
    *,
    days: int,
    limit: int,
    tournament: str | None,
    env_file: Path | None,
) -> dict[str, Any]:
    key = _api_key(env_file)
    now = datetime.now(timezone.utc)
    start = (now - timedelta(days=max(days, 1))).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (now + timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        discovery_limit = max(limit, 100) if tournament else max(limit, 1)
        series_rows = _series_rows(key, start, end, discovery_limit)
        if tournament:
            needle = tournament.strip().casefold()
            series_rows = [
                row
                for row in series_rows
                if needle in str(row.get("tournament") or "").casefold()
            ][: max(limit, 1)]
    except GridIngestError as exc:
        result = {
            "fetched_at": now.isoformat(),
            "window": {"start": start, "end": end},
            "series_seen": 0,
            "files_downloaded": 0,
            "files_existing": 0,
            "files_not_ready": 0,
            "files_failed": 1,
            "file_list_failures": 0,
            "rate_limited": "429" in str(exc),
            "provider_error": str(exc),
            "pro_only": True,
        }
        RAW_GRID_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_GRID_DIR / "grid_fetch_meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[grid] discovery unavailable; keeping cached rows: {exc}")
        return result
    downloaded = 0
    existing = 0
    not_ready = 0
    failed = 0
    file_list_failures = 0
    rate_limited = False
    for series in series_rows:
        series_id = str(series["id"])
        RAW_GRID_DIR.mkdir(parents=True, exist_ok=True)
        (RAW_GRID_DIR / f"series_{series_id}.json").write_text(
            json.dumps(series, indent=2), encoding="utf-8"
        )
        try:
            files = _file_list(key, series_id)
        except GridIngestError as exc:
            file_list_failures += 1
            rate_limited = "429" in str(exc)
            print(f"[grid] file list unavailable for {series_id}; skipping: {exc}")
            if rate_limited:
                break
            continue
        riot_files = [
            file
            for file in files
            if str(file.get("id") or "").startswith("events-riot")
        ]
        for file in riot_files:
            status = str(file.get("status") or "")
            url = str(file.get("fullURL") or "")
            name = _safe_filename(str(file.get("fileName") or file.get("id") or "events.jsonl"))
            dest = RAW_GRID_DIR / name
            if status != "ready" or not url:
                not_ready += 1
                continue
            if dest.exists() and dest.stat().st_size > 0:
                existing += 1
                continue
            if not _download(url, key, dest):
                failed += 1
                rate_limited = True
                break
            downloaded += 1
        if rate_limited:
            print("[grid] stopping this fetch window after a provider rate limit")
            break
    result = {
        "fetched_at": now.isoformat(),
        "window": {"start": start, "end": end},
        "series_seen": len(series_rows),
        "tournament_filter": tournament or "",
        "files_downloaded": downloaded,
        "files_existing": existing,
        "files_not_ready": not_ready,
        "files_failed": failed,
        "file_list_failures": file_list_failures,
        "rate_limited": rate_limited,
        "pro_only": True,
    }
    (RAW_GRID_DIR / "grid_fetch_meta.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def ingest_grid(
    *,
    download: bool = False,
    days: int = 3,
    limit: int = 40,
    tournament: str | None = None,
    env_file: Path | None = None,
    required: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download recent completed pro files optionally, then build local parquet."""
    fetch_meta: dict[str, Any] = {}
    if download:
        fetch_meta = _download_recent(
            days=days,
            limit=limit,
            tournament=tournament,
            env_file=env_file,
        )
    team_df, player_df, parse_meta = _parse_local_grid()
    cached_team, cached_player = _load_cached_grid()
    if not cached_team.empty or not cached_player.empty:
        # Freshly parsed raw files win, while a provider throttle or a new
        # worker still retains the last verified GRID rows for continuity.
        team_df = merge_source_frames(team_df, cached_team, ["gameid", "side"])
        player_df = merge_source_frames(
            player_df,
            cached_player,
            ["gameid", "side", "position"],
        )
        parse_meta["cached_games"] = int(cached_team["gameid"].nunique()) if "gameid" in cached_team else 0
    meta = {**fetch_meta, **parse_meta, "refreshed_at": datetime.now(timezone.utc).isoformat()}
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    team_df.to_parquet(PARQUET_DIR / "grid_team_games.parquet", index=False)
    player_df.to_parquet(PARQUET_DIR / "grid_player_games.parquet", index=False)
    (PARQUET_DIR / "grid_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if required and team_df.empty:
        raise GridIngestError("GRID refresh was required but produced no completed games")
    print(f"[grid] games={meta['games']} team_rows={meta['team_rows']} player_rows={meta['player_rows']}")
    return team_df, player_df


def merge_source_frames(
    primary: pd.DataFrame | None,
    supplement: pd.DataFrame | None,
    keys: Sequence[str],
) -> pd.DataFrame:
    """Merge source rows with primary-source precedence and stable ordering."""
    frames = []
    if primary is not None and not primary.empty:
        p = primary.copy()
        if "date" in p.columns:
            p["date"] = pd.to_datetime(p["date"], errors="coerce", utc=True).dt.tz_localize(None)
        if "patch" in p.columns:
            p["patch"] = p["patch"].astype("string")
        p["_source_priority"] = 2
        frames.append(p)
    if supplement is not None and not supplement.empty:
        s = supplement.copy()
        if "date" in s.columns:
            s["date"] = pd.to_datetime(s["date"], errors="coerce", utc=True).dt.tz_localize(None)
        if "patch" in s.columns:
            s["patch"] = s["patch"].astype("string")
        s["_source_priority"] = 1
        frames.append(s)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True, sort=False)
    if all(key in out.columns for key in keys):
        out = (
            out.sort_values("_source_priority", ascending=False)
            .drop_duplicates(list(keys), keep="first")
            .drop(columns=["_source_priority"], errors="ignore")
        )
    else:
        out = out.drop(columns=["_source_priority"], errors="ignore")
    return out.reset_index(drop=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--tournament", type=str, default=None)
    parser.add_argument("--grid-env-file", type=Path, default=None)
    parser.add_argument("--required", action="store_true")
    args = parser.parse_args(argv)
    ingest_grid(
        download=args.download,
        days=args.days,
        limit=args.limit,
        tournament=args.tournament,
        env_file=args.grid_env_file,
        required=args.required,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
