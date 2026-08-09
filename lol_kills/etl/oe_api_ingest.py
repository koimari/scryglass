"""Fetch recent completed Oracle's Elixir games through its public API.

The annual OE CSV is the historical source. This API bridge covers completed
games that are already visible in OE while the next annual export is pending.
It carries the same map fields used by the tier-list replay. It does not add
model authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from lol_kills.etl.paths import PARQUET_DIR, WAREHOUSE_DIR
from lol_kills.etl.source_keys import canonical_source_game_key

API_BASE = "https://oe.datalisk.io"
SCHEMA_VERSION = "scryglass:oe-api-live-bridge:v1"
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_MAX_WORKERS = 8
ROLES = ("top", "jng", "mid", "bot", "sup")
PLAYER_OUTPUT = PARQUET_DIR / "oe_api_player_games.parquet"
TEAM_OUTPUT = PARQUET_DIR / "oe_api_team_games.parquet"
META_OUTPUT = PARQUET_DIR / "oe_api_meta.json"
RAW_OUTPUT = WAREHOUSE_DIR / "raw" / "oe_api" / "tierlist-live-v1.json"


class OeApiIngestError(RuntimeError):
    """Raised when the OE API bridge cannot make a complete source receipt."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _api_key() -> str:
    key = (os.environ.get("ORACLES_ELIXIR_API_KEY") or os.environ.get("OE_API_KEY") or "").strip()
    if not key:
        raise OeApiIngestError(
            "ORACLES_ELIXIR_API_KEY is required for the OE API bridge"
        )
    return key


def _request_json(
    path: str,
    *,
    api_key: str,
    params: Mapping[str, object] | None = None,
    timeout: float = 45,
) -> Any:
    query = urllib.parse.urlencode({key: str(value) for key, value in (params or {}).items()})
    url = f"{API_BASE}{path}"
    if query:
        url = f"{url}?{query}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "scryglass/oe-api-bridge (+descriptive-tierlist)",
            "X-Api-Key": api_key,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise OeApiIngestError(f"OE API returned HTTP {exc.code} for {path}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise OeApiIngestError(f"OE API request failed for {path}") from exc


def _parse_timestamp(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _normal_key(value: object) -> str:
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _league_code(league_id: object, tournament_id: object) -> str:
    league = str(league_id or "").strip()
    tournament = str(tournament_id or "").strip()
    prefix = tournament.split("/", 1)[0]
    known = {
        "League of Legends Championship Series": "LCS",
        "LoL EMEA Championship": "LEC",
        "LoL Champions Korea": "LCK",
        "Tencent LoL Pro League": "LPL",
        "League of Legends Championship Pacific": "LCP",
        "Circuit Brazilian League of Legends": "CBLOL",
        "North American Challengers League": "NACL",
        "LCK Challengers League": "LCKC",
        "Esports World Cup": "EWC",
        "Mid-Season Invitational": "MSI",
        "First Stand": "FST",
        "EMEA Masters": "EM",
        "Asia Master": "ASI",
        "KeSPA Cup": "KESPA",
        "Pacific Championship Series": "PCS",
        "Vietnam Championship Series": "VCS",
        "LoL Japan League": "LJL",
        "Turkish Championship League": "TCL",
        "La Ligue Française": "LFL",
        "Liga Portuguesa": "LPLOL",
        "Circuito Desafiante": "CD",
        "Esports Balkan League": "EBL",
        "Liga Española de League of Legends": "LES",
        "Liga Regional Norte": "LRN",
        "Liga Regional Sur": "LRS",
        "Northern League of Legends Championship": "NLC",
        "Hellenic Legends League": "HLL",
        "Prime League Pro Division": "PRM",
        "Prime League 1st Division": "PRM",
        "Rift Legends": "RL",
        "TransIP Road Of Legends": "ROL",
        "Nexus League": "NL",
        "Arabian League": "AL",
        "LIT": "LIT",
    }
    if prefix in known:
        return known[prefix]
    if prefix in {"CBLOL", "LCP", "LCK", "LCS", "LEC", "LFL", "LIT", "LPL", "NACL", "NLC", "PCS", "TCL", "VCS"}:
        return prefix
    if league in known:
        return known[league]
    return prefix.upper().replace(" ", "_") or "UNKNOWN"


def _event_kind(league: str) -> str | None:
    return {
        "ASI": "asia_master",
        "EM": "em",
        "EWC": "ewc",
        "FST": "fst",
        "MSI": "msi",
        "WORLDS": "worlds",
        "KESPA": "kespa",
    }.get(league)


def _competition_tier(league: str, level: object, event_kind: str | None) -> str:
    if event_kind is not None:
        return "international"
    if str(level or "").casefold() == "primary":
        return "tier1"
    if league in {"LCKC", "NACL", "LFL2", "LIT2", "LPLC", "LRN", "LRS", "NLC", "NL"}:
        return "tier2"
    return "tier2" if league in {
        "AL",
        "CD",
        "EBL",
        "HLL",
        "LES",
        "LPLOL",
        "PRM",
        "RL",
        "ROL",
        "TCL",
    } else "tier1"


def _discover_tournaments(
    *,
    api_key: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int,
) -> list[dict[str, Any]]:
    rows = _request_json("/tournaments/latestByLeague", api_key=api_key)
    active = _request_json("/tournaments/latest", api_key=api_key)
    if not isinstance(rows, list) or not isinstance(active, list):
        raise OeApiIngestError("OE tournament discovery returned an invalid shape")
    active_by_id = {
        str(row.get("id")): row
        for row in active
        if isinstance(row, Mapping) and row.get("id")
    }
    floor = start - pd.Timedelta(days=lookback_days)
    discovered: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        tournament_id = str(row.get("tournamentId") or "").strip()
        started = _parse_timestamp(row.get("startDate"))
        if not tournament_id or started is None or started > end or started < floor:
            continue
        active_row = active_by_id.get(tournament_id, {})
        league_id = str(row.get("leagueId") or "")
        level = active_row.get("level") if isinstance(active_row, Mapping) else None
        code = _league_code(league_id, tournament_id)
        event_kind = _event_kind(code)
        discovered[tournament_id] = {
            "tournament_id": tournament_id,
            "tournament_name": str(active_row.get("name") or tournament_id),
            "league_id": league_id,
            "league": code,
            "level": level or "",
            "competition_tier": _competition_tier(code, level, event_kind),
            "event_kind": event_kind,
        }
    if not discovered:
        raise OeApiIngestError("OE tournament discovery returned no current tournaments")
    return [discovered[key] for key in sorted(discovered)]


def _fetch_team_ids(
    tournaments: list[dict[str, Any]],
    *,
    api_key: str,
    max_workers: int,
) -> list[str]:
    def fetch(item: dict[str, Any]) -> list[str]:
        body = _request_json(
            "/stats/teams/byTournament",
            api_key=api_key,
            params={"tournament": item["tournament_id"]},
        )
        if not isinstance(body, list):
            raise OeApiIngestError("OE team discovery returned an invalid shape")
        return [str(row["id"]) for row in body if isinstance(row, Mapping) and row.get("id")]

    teams: set[str] = set()
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch, item) for item in tournaments]
        for future in as_completed(futures):
            try:
                teams.update(future.result())
            except OeApiIngestError as exc:
                errors.append(str(exc))
    if errors:
        raise OeApiIngestError(f"OE team discovery failed for {len(errors)} tournaments")
    if not teams:
        raise OeApiIngestError("OE team discovery returned no teams")
    return sorted(teams)


def _fetch_games(
    team_ids: list[str],
    *,
    api_key: str,
    end: pd.Timestamp,
    not_before: pd.Timestamp | None,
    max_workers: int,
) -> list[dict[str, Any]]:
    boundary = _parse_timestamp(not_before) if not_before is not None else None

    def fetch(team_id: str) -> list[dict[str, Any]]:
        body = _request_json(
            f"/teams/gameDetails/{urllib.parse.quote(team_id, safe='')}",
            api_key=api_key,
            params={"beforeTime": end.isoformat().replace("+00:00", "Z")},
        )
        if not isinstance(body, list):
            raise OeApiIngestError("OE game discovery returned an invalid shape")
        selected: list[dict[str, Any]] = []
        for row in body:
            if not isinstance(row, Mapping):
                continue
            created = _parse_timestamp(row.get("gameCreation"))
            if boundary is not None and (created is None or created < boundary):
                continue
            selected.append(dict(row))
        return selected

    games: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(fetch, team_id): team_id for team_id in team_ids}
        for future in as_completed(futures):
            try:
                for game in future.result():
                    game_id = str(game.get("oeGameId") or game.get("gameId") or "")
                    if game_id:
                        games.setdefault(game_id, game)
            except OeApiIngestError as exc:
                errors.append(str(exc))
    if errors:
        raise OeApiIngestError(f"OE game discovery failed for {len(errors)} teams")
    return sorted(games.values(), key=lambda row: str(row.get("gameCreation") or ""))


def _fetch_full_games(
    games: list[dict[str, Any]],
    *,
    api_key: str,
    max_workers: int,
) -> tuple[dict[str, dict[str, Any]], int]:
    """Fetch the player names that the compact team-game endpoint omits."""

    def fetch(game: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
        game_id = str(game.get("oeGameId") or game.get("gameId") or "")
        if not game_id:
            return "", None
        try:
            body = _request_json(
                f"/games/singleGame/{urllib.parse.quote(game_id, safe='')}",
                api_key=api_key,
            )
        except OeApiIngestError:
            return game_id, None
        if not isinstance(body, list) or not body or not isinstance(body[0], Mapping):
            return game_id, None
        return game_id, dict(body[0])

    details: dict[str, dict[str, Any]] = {}
    missing = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch, game) for game in games]
        for future in as_completed(futures):
            game_id, detail = future.result()
            if not game_id or detail is None:
                missing += 1
                continue
            details[game_id] = detail
    return details, missing


def _cached_full_games(path: Path) -> dict[str, dict[str, Any]]:
    """Reuse player names from the previous completed API bridge receipt."""

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, Mapping) or payload.get("schema_version") != SCHEMA_VERSION:
        return {}
    cached: dict[str, dict[str, Any]] = {}
    for game in payload.get("games", []):
        if not isinstance(game, Mapping):
            continue
        game_id = str(game.get("oe_game_id") or "").strip()
        players = game.get("players")
        if not game_id or not isinstance(players, Mapping):
            continue
        detail: dict[str, Any] = {}
        complete = True
        for side, team_key in (("blue", "blueTeam"), ("red", "redTeam")):
            side_players = players.get(side)
            if not isinstance(side_players, Mapping):
                complete = False
                break
            detail_players: dict[str, dict[str, str]] = {}
            for role in ROLES:
                name = str(side_players.get(role) or "").strip()
                if not name:
                    complete = False
                    break
                detail_players[role] = {"name": name}
            if not complete:
                break
            detail[team_key] = {"players": detail_players}
        if complete:
            cached[game_id] = detail
    return cached


def _metadata_for_game(
    game: Mapping[str, Any],
    tournaments: list[dict[str, Any]],
) -> dict[str, Any] | None:
    by_name = {_normal_key(item["tournament_name"]): item for item in tournaments}
    tournament_name = str(game.get("tournament") or "")
    found = by_name.get(_normal_key(tournament_name))
    if found is not None:
        return found
    game_id = str(game.get("gameId") or "")
    prefix = game_id.split("/", 1)[0]
    for item in tournaments:
        if item["tournament_id"].split("/", 1)[0] == prefix:
            return item
    return None


def _rows_from_games(
    games: list[dict[str, Any]],
    *,
    tournaments: list[dict[str, Any]],
    full_games: Mapping[str, Mapping[str, Any]],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    player_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    accepted_games: list[dict[str, Any]] = []
    roles = ("top", "jng", "mid", "bot", "sup")
    for game in games:
        created = _parse_timestamp(game.get("gameCreation"))
        metadata = _metadata_for_game(game, tournaments)
        side = str(game.get("side") or "").casefold()
        own = str(game.get("ownId") or "").strip()
        opponent = str(game.get("opponentTeam") or "").strip()
        result = pd.to_numeric(game.get("result"), errors="coerce")
        if created is None or created < start or created > end or metadata is None:
            continue
        if side not in {"blue", "red"} or not own or not opponent or pd.isna(result) or int(result) not in {0, 1}:
            continue
        if any(not str(game.get(f"{color}{role}") or "").strip() for color in ("blue", "red") for role in roles):
            continue
        blue_team = own if side == "blue" else opponent
        red_team = opponent if side == "blue" else own
        blue_result = int(result) if side == "blue" else 1 - int(result)
        raw_game_uid = game.get("oeGameId") or game.get("gameId")
        game_uid = canonical_source_game_key(raw_game_uid)
        if not game_uid:
            continue
        detail = full_games.get(str(raw_game_uid).strip(), {})
        game_id = game_uid
        common = {
            "gameid": game_id,
            "game_uid": game_uid,
            "date": created,
            "league": metadata["league"],
            # Keep the OE league code as source provenance. Transport is
            # already explicit in source_transport.
            "league_source": metadata["league"],
            "competition_tier": metadata["competition_tier"],
            "event_kind": metadata["event_kind"],
            "is_international": metadata["event_kind"] is not None,
            "patch": str(game.get("patch") or ""),
            "year": int(created.year),
            "oe_year": int(created.year),
            "tournament": str(game.get("tournament") or metadata["tournament_name"]),
            "source": "oe",
            "source_transport": "oe_api",
            "datacompleteness": "complete",
            "playoffs": pd.NA,
            "result": None,
        }
        for side_name, team_name, side_result in (
            ("Blue", blue_team, blue_result),
            ("Red", red_team, 1 - blue_result),
        ):
            row = {**common, "side": side_name, "teamname": team_name, "position": "team", "result": side_result}
            team_rows.append(row)
            for role in roles:
                player_detail = {}
                team_key = "blueTeam" if side_name == "Blue" else "redTeam"
                team_detail = detail.get(team_key, {}) if isinstance(detail, Mapping) else {}
                players_detail = team_detail.get("players", {}) if isinstance(team_detail, Mapping) else {}
                if isinstance(players_detail, Mapping):
                    candidate = players_detail.get(role, {})
                    if isinstance(candidate, Mapping):
                        player_detail = candidate
                player_rows.append(
                    {
                        **common,
                        "side": side_name,
                        "teamname": team_name,
                        "position": role,
                        "champion": str(game[f"{'blue' if side_name == 'Blue' else 'red'}{role}"]).strip(),
                        "playername": str(player_detail.get("name") or "").strip() or pd.NA,
                        "player_id": str(player_detail.get("playerId") or "").strip() or pd.NA,
                        "result": side_result,
                    }
                )
        accepted_games.append(
            {
                "game_id": game_id,
                "oe_game_id": game_uid,
                "date": created.isoformat().replace("+00:00", "Z"),
                "league": metadata["league"],
                "competition_tier": metadata["competition_tier"],
                "event_kind": metadata["event_kind"],
                "patch": str(game.get("patch") or ""),
                "blue_team": blue_team,
                "red_team": red_team,
                "blue_result": blue_result,
                "roles": {
                    "blue": {role: str(game[f"blue{role}"]).strip() for role in roles},
                    "red": {role: str(game[f"red{role}"]).strip() for role in roles},
                },
                "players": {
                    "blue": {
                        role: str(
                            ((detail.get("blueTeam") or {}).get("players") or {}).get(role, {}).get("name") or ""
                        ).strip()
                        for role in roles
                    },
                    "red": {
                        role: str(
                            ((detail.get("redTeam") or {}).get("players") or {}).get(role, {}).get("name") or ""
                        ).strip()
                        for role in roles
                    },
                },
            }
        )
    if not accepted_games:
        raise OeApiIngestError("OE API returned no complete five-role games in the requested window")
    return pd.DataFrame(team_rows), pd.DataFrame(player_rows), accepted_games


def ingest_oe_api(
    root: Path | str = Path("."),
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> dict[str, Any]:
    if lookback_days < 0 or max_workers < 1:
        raise ValueError("lookback_days must be non-negative and max_workers must be positive")
    repo_root = Path(root).resolve()
    api_key = _api_key()
    start = _parse_timestamp(start)
    end = _parse_timestamp(end)
    if start is None or end is None or start > end:
        raise ValueError("start and end must be valid ordered timestamps")
    primary_latest = None
    primary_path = repo_root / PARQUET_DIR / "oe_player_games.parquet"
    if primary_path.is_file():
        try:
            primary_dates = pd.to_datetime(
                pd.read_parquet(primary_path, columns=["date"])["date"],
                errors="coerce",
                utc=True,
            ).dropna()
            if not primary_dates.empty:
                primary_latest = pd.Timestamp(primary_dates.max())
        except (OSError, KeyError, ValueError):
            primary_latest = None
    not_before = start - pd.Timedelta(days=lookback_days)
    if primary_latest is not None:
        not_before = max(not_before, primary_latest - pd.Timedelta(days=7))
    tournaments = _discover_tournaments(
        api_key=api_key,
        start=start,
        end=end,
        lookback_days=lookback_days,
    )
    team_ids = _fetch_team_ids(tournaments, api_key=api_key, max_workers=max_workers)
    games = _fetch_games(
        team_ids,
        api_key=api_key,
        end=end,
        not_before=not_before,
        max_workers=max_workers,
    )
    detail_games = [
        game
        for game in games
        if primary_latest is None
        or (
            _parse_timestamp(game.get("gameCreation")) is not None
            and _parse_timestamp(game.get("gameCreation")) > primary_latest
        )
    ]
    cached_full_games = _cached_full_games(repo_root / RAW_OUTPUT)
    detail_game_ids = {
        str(game.get("oeGameId") or game.get("gameId") or "")
        for game in detail_games
    }
    cached_detail_ids = detail_game_ids.intersection(cached_full_games)
    detail_games_to_fetch = [
        game
        for game in detail_games
        if str(game.get("oeGameId") or game.get("gameId") or "") not in cached_full_games
    ]
    fetched_full_games, fetched_missing = _fetch_full_games(
        detail_games_to_fetch,
        api_key=api_key,
        max_workers=max_workers,
    )
    full_games = {**cached_full_games, **fetched_full_games}
    full_detail_ids = detail_game_ids.intersection(full_games)
    full_games_missing = len(detail_game_ids - full_detail_ids)
    team_frame, player_frame, accepted_games = _rows_from_games(
        games,
        tournaments=tournaments,
        full_games=full_games,
        start=start,
        end=end,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "provider": "Oracle's Elixir",
            "transport": "public_datalisk_api",
            "api_base": API_BASE,
            "retrieved_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "window_start": start.isoformat().replace("+00:00", "Z"),
            "window_end": end.isoformat().replace("+00:00", "Z"),
        },
        "tournaments": tournaments,
        "team_count": len(team_ids),
        "games_discovered": len(games),
        "games_accepted": len(accepted_games),
        "full_detail_games_requested": len(detail_games),
        "full_detail_games_fetched": len(detail_games_to_fetch),
        "full_detail_games_cached": len(cached_detail_ids),
        "games_with_full_details": len(full_detail_ids),
        "games_missing_full_details": full_games_missing,
        "games": accepted_games,
        "authority": {
            "descriptive_source_freshness_evidence": True,
            "model_validation_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": "This receipt binds current OE game rows. It does not validate a model, probability, recommendation, or wager.",
    }
    raw = _canonical_json(payload) + b"\n"
    raw_path = repo_root / RAW_OUTPUT
    team_path = repo_root / TEAM_OUTPUT
    player_path = repo_root / PLAYER_OUTPUT
    meta_path = repo_root / META_OUTPUT
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    team_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(raw)
    team_frame.to_parquet(team_path, index=False)
    player_frame.to_parquet(player_path, index=False)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "retrieved_at": payload["source"]["retrieved_at"],
        "window_start": payload["source"]["window_start"],
        "window_end": payload["source"]["window_end"],
        "games": len(accepted_games),
        "source_latest": max(game["date"] for game in accepted_games),
        "full_detail_games_requested": len(detail_games),
        "full_detail_games_fetched": len(detail_games_to_fetch),
        "full_detail_games_cached": len(cached_detail_ids),
        "games_with_full_details": len(full_detail_ids),
        "games_missing_full_details": full_games_missing,
        "player_detail_complete": full_games_missing == 0,
        "player_detail_floor": primary_latest.isoformat().replace("+00:00", "Z") if primary_latest is not None else None,
        "team_rows": len(team_frame),
        "player_rows": len(player_frame),
        "player_rows_with_names": int(player_frame["playername"].notna().sum()) if "playername" in player_frame else 0,
        "raw": {"locator": str(raw_path.relative_to(repo_root)), "raw_sha256": _sha256_bytes(raw)},
        "team_output": {"locator": str(team_path.relative_to(repo_root)), "raw_sha256": _sha256_path(team_path)},
        "player_output": {"locator": str(player_path.relative_to(repo_root)), "raw_sha256": _sha256_path(player_path)},
        "authority": payload["authority"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()
    meta = ingest_oe_api(
        args.root,
        start=pd.Timestamp(args.start),
        end=pd.Timestamp(args.end),
        lookback_days=args.lookback_days,
        max_workers=args.max_workers,
    )
    print(json.dumps(meta, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
