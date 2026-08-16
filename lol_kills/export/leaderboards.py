"""Build the support-chat leaderboards artifact from the public pack payloads.

The artifact makes the support chat queriable without scanning the whole pack:
per-player aggregates (rating, role, team, league, grade-A games, win rate,
recent form), top-N lists by category, per-role rating indexes for
"who is the jungler with a rating of 1643?" style nearest lookups, and a team
rating list.

Nothing here touches ratings math; it only aggregates already-public payloads.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping

LEADERBOARDS_SCHEMA = "scryglass:leaderboards:v2"
TOP_LIMIT = 50
ROLE_ORDER = ("top", "jng", "mid", "bot", "sup")
MIN_WIN_RATE_GAMES = 20


def _number(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if _finite(parsed) else None


def _finite(value: float) -> bool:
    return math.isfinite(value)


def _grade_a_games(
    profile_records: Mapping[str, Any],
) -> tuple[dict[str, int], dict[str, int]]:
    """Count available A grades and total available grades per player."""

    grade_a: dict[str, int] = {}
    grade_total: dict[str, int] = {}
    games = profile_records.get("games")
    if not isinstance(games, Mapping):
        return grade_a, grade_total
    for game in games.values():
        if not isinstance(game, Mapping):
            continue
        players = game.get("players")
        if not isinstance(players, list):
            continue
        for entry in players:
            if not isinstance(entry, Mapping):
                continue
            player = str(entry.get("player") or "").strip()
            if not player:
                continue
            grade = entry.get("grade")
            if not isinstance(grade, Mapping) or grade.get("status") != "available":
                continue
            grade_total[player] = grade_total.get(player, 0) + 1
            if str(grade.get("grade") or "").strip().upper() == "A":
                grade_a[player] = grade_a.get(player, 0) + 1
    return grade_a, grade_total


def _recent_form(profile_records: Mapping[str, Any], player: str) -> float | None:
    """Win rate over the player's last ten profile games (chronological)."""

    games = profile_records.get("games")
    if not isinstance(games, Mapping):
        return None
    outcomes: list[bool] = []
    for game in games.values():
        if not isinstance(game, Mapping):
            continue
        players = game.get("players")
        if not isinstance(players, list):
            continue
        side_won: dict[str, bool] = {}
        for entry in players:
            if not isinstance(entry, Mapping):
                continue
            if str(entry.get("player") or "").strip() != player:
                continue
            side = str(entry.get("side") or "").strip().title()
            if side in {"Blue", "Red"}:
                side_won[side] = bool(game.get("blue_win") == (1 if side == "Blue" else 0))
        if side_won:
            outcomes.append(side_won.get("Blue", side_won.get("Red", False)))
    recent = outcomes[-10:]
    if not recent:
        return None
    return round(sum(1 for outcome in recent if outcome) / len(recent), 4)


def _rating_by_player(player_ratings: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    lookup: dict[str, float] = {}
    for row in player_ratings:
        if not isinstance(row, Mapping):
            continue
        player = str(row.get("player") or "").strip()
        rating = _number(row.get("mu_total"))
        if player and rating is not None:
            lookup[player] = rating
    return lookup


def _top(sorted_rows: list[dict[str, Any]], limit: int = TOP_LIMIT) -> list[dict[str, Any]]:
    return sorted_rows[:limit]


def _team_recent_results(
    match_index: Mapping[str, Any],
    team: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Last results for a team from the public match index (by date, newest last)."""

    games = match_index.get("games")
    if not isinstance(games, list):
        return []
    hits: list[dict[str, Any]] = []
    for entry in games:
        if not isinstance(entry, Mapping):
            continue
        blue = str(entry.get("blue_team") or "").strip()
        red = str(entry.get("red_team") or "").strip()
        if team not in (blue, red):
            continue
        blue_win = entry.get("blue_win")
        won = (blue_win == 1 and team == blue) or (blue_win == 0 and team == red)
        hits.append(
            {
                "date": str(entry.get("date") or ""),
                "league": str(entry.get("league") or ""),
                "opponent": red if team == blue else blue,
                "side": "Blue" if team == blue else "Red",
                "won": bool(won),
                "game_id": str(entry.get("game_id") or ""),
            }
        )
    hits.sort(key=lambda entry: entry["date"])
    return hits[-limit:]


def _champion_performers(
    player_champion_records: Mapping[str, Any],
    min_games: int = 5,
) -> dict[str, list[dict[str, Any]]]:
    """Best players per champion by win rate (min games), top 5 each."""

    by_champion: dict[str, list[dict[str, Any]]] = {}
    for player, records in player_champion_records.items():
        if not isinstance(records, list):
            continue
        for entry in records:
            if not isinstance(entry, Mapping):
                continue
            champion = str(entry.get("champion") or "").strip()
            games = int(entry.get("games") or 0)
            wr = _number(entry.get("wr"))
            if not champion or games < min_games or wr is None:
                continue
            by_champion.setdefault(champion, []).append(
                {
                    "player": str(player).strip(),
                    "games": games,
                    "win_rate": round(wr, 4),
                }
            )
    output: dict[str, list[dict[str, Any]]] = {}
    for champion, entries in by_champion.items():
        output[champion] = sorted(entries, key=lambda entry: float(entry["win_rate"]), reverse=True)[:5]
    return output




def _teams_draft(draft_records: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Teams ranked by mean descriptive draft edge across the archive."""
    if not isinstance(draft_records, Mapping):
        return []
    games = draft_records.get("games")
    if not isinstance(games, Mapping):
        return []
    by_team: dict[str, list[float]] = {}
    for game in games.values():
        if not isinstance(game, Mapping):
            continue
        blue = str(game.get("blue_team") or "").strip()
        red = str(game.get("red_team") or "").strip()
        edge = game.get("draft_edge")
        if not isinstance(edge, (int, float)):
            continue
        edge_value = float(edge)
        if blue:
            by_team.setdefault(blue, []).append(edge_value)
        if red:
            by_team.setdefault(red, []).append(-edge_value)
    rows = []
    for team, evidence in by_team.items():
        games_n = len(evidence)
        if games_n < 5:
            continue
        rows.append({
            "team": team,
            "games": games_n,
            "draft_edge": round(sum(evidence) / games_n, 4),
        })
    rows.sort(key=lambda row: (-row["draft_edge"], -row["games"], row["team"]))
    return rows[:TOP_LIMIT]


def _teams_draft_from_profile(profile_records: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Use the same profile-window composition rows as player profiles."""

    games = profile_records.get("games") if isinstance(profile_records, Mapping) else None
    if not isinstance(games, Mapping):
        return []
    by_team: dict[str, list[float]] = {}
    for game in games.values():
        if not isinstance(game, Mapping):
            continue
        signal = game.get("draft_contribution")
        if not isinstance(signal, Mapping):
            continue
        blue_signal = _number((signal.get("blue") or {}).get("signal")) if isinstance(signal.get("blue"), Mapping) else None
        red_signal = _number((signal.get("red") or {}).get("signal")) if isinstance(signal.get("red"), Mapping) else None
        if blue_signal is None or red_signal is None:
            continue
        edge = blue_signal - red_signal
        blue = str(game.get("blue_team") or "").strip()
        red = str(game.get("red_team") or "").strip()
        if blue:
            by_team.setdefault(blue, []).append(edge)
        if red:
            by_team.setdefault(red, []).append(-edge)
    rows = []
    for team, evidence in by_team.items():
        if len(evidence) < 5:
            continue
        rows.append({
            "team": team,
            "games": len(evidence),
            "draft_edge": round(sum(evidence) / len(evidence), 4),
        })
    rows.sort(key=lambda row: (-row["draft_edge"], -row["games"], row["team"]))
    return rows[:TOP_LIMIT]


def _players_draft(players: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Players ranked by the share of picks that were best available."""
    rows = []
    for entry in players:
        if not isinstance(entry, Mapping):
            continue
        name = str(entry.get("player") or "").strip()
        games_n = int(entry.get("games") or 0)
        score = _number(entry.get("pick_contribution"))
        best_available_rate = _number(entry.get("best_available_rate"))
        if not name or games_n < 5:
            continue
        rows.append({
            "player": name,
            "games": games_n,
            "pick_contribution": round(score, 4) if score is not None else None,
            "best_available_rate": round(best_available_rate, 4) if best_available_rate is not None else None,
            "role": str(entry.get("role") or "").strip() or None,
            "team": str(entry.get("team") or "").strip() or None,
        })
    rows.sort(key=lambda row: (
        -(row["best_available_rate"] if row["best_available_rate"] is not None else float("-inf")),
        -row["games"],
        -(row["pick_contribution"] if row["pick_contribution"] is not None else float("-inf")),
        row["player"],
    ))
    return rows[:TOP_LIMIT]

def build_leaderboards(
    player_records: Mapping[str, Any],
    profile_records: Mapping[str, Any],
    player_ratings: Sequence[Mapping[str, Any]],
    team_ratings: Sequence[Mapping[str, Any]],
    team_records: Mapping[str, Any] | None = None,
    player_champion_records: Mapping[str, Any] | None = None,
    match_index: Mapping[str, Any] | None = None,
    draft_records: Mapping[str, Any] | None = None,
    draft_players: Sequence[Mapping[str, Any]] | None = None,
    draft_profile_records: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate the public payloads into the leaderboards artifact."""

    grade_a, grade_total = _grade_a_games(profile_records)
    ratings = _rating_by_player(player_ratings)

    players: dict[str, dict[str, Any]] = {}
    for player, record in player_records.items():
        if not isinstance(record, Mapping):
            continue
        name = str(player).strip()
        if not name:
            continue
        games = int(record.get("games") or 0)
        wins = int(record.get("wins") or 0)
        wr = _number(record.get("wr"))
        players[name] = {
            "rating": ratings.get(name),
            "role": str(record.get("primary_role") or "").strip() or None,
            "team": str(record.get("current_team") or "").strip() or None,
            "league": str(record.get("current_league") or "").strip() or None,
            "grade_a_games": grade_a.get(name, 0),
            "grade_games": grade_total.get(name, 0),
            "games": games,
            "win_rate": round(wr, 4) if wr is not None else None,
            "recent_form": _recent_form(profile_records, name),
        }

    def row(name: str, key: str) -> dict[str, Any]:
        payload = players.get(name) or {}
        return {
            "player": name,
            "rating": payload.get("rating"),
            "role": payload.get("role"),
            "team": payload.get("team"),
            "league": payload.get("league"),
            "grade_a_games": payload.get("grade_a_games", 0),
            "grade_games": payload.get("grade_games", 0),
            "games": payload.get("games", 0),
            "win_rate": payload.get("win_rate"),
        }

    by_a_grades = sorted(
        (name for name, payload in players.items() if payload.get("grade_a_games", 0) > 0),
        key=lambda name: (
            players[name]["grade_a_games"],
            players[name].get("grade_games", 0),
            -players[name].get("games", 0),
        ),
        reverse=True,
    )
    by_rating = sorted(
        (name for name, payload in players.items() if payload.get("rating") is not None),
        key=lambda name: float(players[name]["rating"]),
        reverse=True,
    )
    by_win_rate = sorted(
        (
            name
            for name, payload in players.items()
            if payload.get("win_rate") is not None
            and payload.get("games", 0) >= MIN_WIN_RATE_GAMES
        ),
        key=lambda name: float(players[name]["win_rate"]),
        reverse=True,
    )
    rating_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ROLE_ORDER:
        role_players = [
            name
            for name in by_rating
            if players[name].get("role") == role
        ]
        rating_by_role[role] = _top([row(name, "rating") for name in role_players], 30)

    teams: list[dict[str, Any]] = []
    team_records_map = team_records if isinstance(team_records, Mapping) else {}
    for team_row in team_ratings:
        if not isinstance(team_row, Mapping):
            continue
        team = str(team_row.get("team") or "").strip()
        rating = _number(team_row.get("mu_total"))
        if not team or rating is None:
            continue
        record = team_records_map.get(team) if isinstance(team_records_map, Mapping) else None
        record = record if isinstance(record, Mapping) else {}
        teams.append(
            {
                "team": team,
                "team_key": record.get("team_key"),
                "rating": round(rating, 1),
                "league": str(record.get("current_league") or team_row.get("league") or "").strip() or None,
                "games": int(record.get("games") or team_row.get("n_maps") or team_row.get("n_series") or 0),
                "wins": int(record.get("wins") or 0),
                "win_rate": round(_number(record.get("wr")) or 0.0, 4),
                "recent": _team_recent_results(match_index, team) if match_index else [],
            }
        )
    teams = _top(sorted(teams, key=lambda entry: float(entry["rating"]), reverse=True))

    champions = _champion_performers(player_champion_records) if player_champion_records else {}

    indexes: dict[str, Any] = {
        "players": {name: {"role": payload.get("role"), "team": payload.get("team")} for name, payload in players.items()},
        "teams": {entry["team"]: {"team_key": entry.get("team_key")} for entry in teams},
        "champions": sorted(champions.keys()),
    }
    leagues: set[str] = set()
    for name, payload in players.items():
        if payload.get("league"):
            leagues.add(str(payload["league"]))
    for entry in teams:
        if entry.get("league"):
            leagues.add(str(entry["league"]))
    indexes["leagues"] = sorted(leagues)

    return {
        "schema_version": LEADERBOARDS_SCHEMA,
        "players": players,
        "top": {
            "a_grades": _top([row(name, "a_grades") for name in by_a_grades]),
            "rating": _top([row(name, "rating") for name in by_rating]),
            "win_rate": _top([row(name, "win_rate") for name in by_win_rate]),
            "rating_by_role": rating_by_role,
        },
        "teams": teams,
        "champions": champions,
        "indexes": indexes,
        "teams_draft": _teams_draft_from_profile(draft_profile_records) if draft_profile_records is not None else _teams_draft(draft_records),
        "players_draft": _players_draft(draft_players or []),
    }


def write_leaderboards(
    payload: dict[str, Any],
    destination: Any,
) -> None:
    """Serialize the artifact with the standard pack formatting."""
    destination.write_text(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        encoding="utf-8",
    )
