"""Public, provenance-preserving snapshots for the live Scryglass surface.

The GRID transaction stream stays private to the worker.  This module selects
the small set of fields that the public page needs, evaluates the state, and
publishes immutable snapshots plus short-lived pointers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from lol_kills.etl.aliases import fuzzy_team_match, normalize_champ, normalize_team
from lol_kills.export.upload_pack import _blob_put
from lol_kills.live_model import LiveEvaluation, evaluate_live_state

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "apps" / "lol-atlas" / "public"
DEFAULT_LOCAL_LIVE_ROOT = PUBLIC_ROOT / "live"
SCHEMA_VERSION = "live.v1"


def _text(value: Any, fallback: str | None = None) -> str | None:
    if value is None:
        return fallback
    text = str(value).strip()
    return text or fallback


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _int(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None else None


def _active_game(series_state: Mapping[str, Any]) -> tuple[int, Mapping[str, Any]] | None:
    games = [game for game in series_state.get("games") or [] if isinstance(game, Mapping)]
    if not games:
        return None
    active = [(index, game) for index, game in enumerate(games) if game.get("finished") is not True]
    return active[-1] if active else (len(games) - 1, games[-1])


def _side(team: Mapping[str, Any]) -> str | None:
    value = _text(team.get("side"), "")
    return value.lower() if value and value.lower() in {"blue", "red"} else None


def _champion(player: Mapping[str, Any]) -> str | None:
    value = player.get("champion") or player.get("character") or player.get("championName")
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("displayName")
    return normalize_champ(_text(value, "") or "") or None


def _players(team: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for player in team.get("players") or []:
        if not isinstance(player, Mapping):
            continue
        role = _text(player.get("role") or player.get("position") or player.get("lane"))
        name = _text(player.get("name") or player.get("nickname") or player.get("summonerName"))
        champion = _champion(player)
        if not name and not champion:
            continue
        output.append({"name": name or "Unknown player", "role": role, "champion": champion})
    return output[:5]


def _team_gold(team: Mapping[str, Any]) -> float | None:
    players = [player for player in team.get("players") or [] if isinstance(player, Mapping)]
    earned = [_number(player.get("totalMoneyEarned")) for player in players]
    if players and all(value is not None for value in earned):
        total = sum(value for value in earned if value is not None)
        if total > 0:
            return total
    return _number(team.get("netWorth") or team.get("money"))


def _team_kills(team: Mapping[str, Any]) -> int | None:
    for key in ("kills", "killCount"):
        if key in team and team.get(key) is not None:
            return _int(team.get(key))
    return None


def _objectives(team: Mapping[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    objectives = team.get("objectives") or []
    if isinstance(objectives, Mapping):
        objectives = [objectives]
    for objective in objectives:
        if not isinstance(objective, Mapping):
            continue
        kind = _text(objective.get("type") or objective.get("name"))
        if not kind:
            continue
        count = None
        for key in ("completionCount", "count"):
            if key in objective and objective.get(key) is not None:
                count = _int(objective.get(key))
                break
        completed = objective.get("completed")
        item: dict[str, Any] = {"name": kind}
        if count is not None:
            item["count"] = count
        if isinstance(completed, bool):
            item["completed"] = completed
        output.append(item)
    return output


def _clock_seconds(game: Mapping[str, Any]) -> float | None:
    clock = game.get("clock")
    if isinstance(clock, Mapping):
        for key in ("currentSeconds", "elapsedSeconds"):
            value = _number(clock.get(key))
            if value is not None and value >= 0:
                return value
    for key in ("currentSeconds", "elapsedSeconds"):
        value = _number(game.get(key))
        if value is not None and value >= 0:
            return value
    game_time = _number(game.get("gameTime"))
    if game_time is not None and game_time >= 0:
        return game_time / 1000.0
    return None


def _winner(game: Mapping[str, Any], team: Mapping[str, Any]) -> bool | None:
    for key in ("won", "winner"):
        value = team.get(key)
        if isinstance(value, bool):
            return value
    winner = game.get("winner")
    if isinstance(winner, Mapping):
        winner = winner.get("side") or winner.get("name") or winner.get("id")
    if winner is None:
        return None
    team_side = _side(team)
    team_id = _text(team.get("id"))
    team_name = _text(team.get("name"))
    return str(winner).lower() in {
        value.lower()
        for value in (team_side, team_id, team_name)
        if value
    }


def _teams(game: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for team in game.get("teams") or []:
        if isinstance(team, Mapping) and _side(team):
            result[_side(team) or ""] = team
    return result


def _same_team(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> bool:
    candidate_id = _text(candidate.get("id"))
    reference_id = _text(reference.get("id"))
    if candidate_id and reference_id:
        return candidate_id == reference_id
    candidate_name = _text(candidate.get("name"))
    reference_name = _text(reference.get("name"))
    return bool(
        candidate_name
        and reference_name
        and normalize_team(candidate_name).casefold()
        == normalize_team(reference_name).casefold()
    )


def _series_score(
    series_state: Mapping[str, Any],
    current_team: Mapping[str, Any],
) -> int | None:
    score = 0
    seen = False
    for game in series_state.get("games") or []:
        if not isinstance(game, Mapping):
            continue
        matching_team = next(
            (
                team
                for team in _teams(game).values()
                if _same_team(team, current_team)
            ),
            None,
        )
        winner = _winner(game, matching_team) if matching_team is not None else None
        if winner is not None:
            seen = True
            score += int(winner)
    return score if seen else None


def _public_teams(game: Mapping[str, Any], series_state: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for side in ("blue", "red"):
        team = _teams(game).get(side, {})
        result[side] = {
            "name": _text(team.get("name"), side.title()),
            "score": _series_score(series_state, team),
            "game_winner": _winner(game, team),
            "players": _players(team),
        }
    return result


def default_ratings_path() -> Path:
    latest_path = PUBLIC_ROOT / "packs" / "latest.json"
    if latest_path.is_file():
        try:
            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            pack_id = _text(latest.get("pack_id")) if isinstance(latest, Mapping) else None
            if pack_id and Path(pack_id).name == pack_id:
                return (
                    PUBLIC_ROOT
                    / "packs"
                    / pack_id
                    / "features"
                    / "ratings_snapshot.json"
                )
        except (OSError, json.JSONDecodeError):
            pass
    return (
        PUBLIC_ROOT
        / "packs"
        / "__no_current_immutable_pack__"
        / "features"
        / "ratings_snapshot.json"
    )


def resolve_team_rating_gap(
    blue_name: str | None,
    red_name: str | None,
    ratings_path: Path | None = None,
) -> tuple[float | None, dict[str, Any]]:
    """Resolve a public-pack team rating gap without making up missing teams."""
    path = Path(ratings_path or default_ratings_path())
    if not path.is_file() or not blue_name or not red_name:
        return None, {"source": None, "blue": None, "red": None, "missing": ["team_rating_gap"]}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, {"source": None, "blue": None, "red": None, "missing": ["team_rating_gap"]}
    by_name: dict[str, dict[str, Any]] = {}
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, Mapping):
            continue
        name = _text(row.get("team"))
        if name:
            by_name[normalize_team(name).lower()] = dict(row)

    def lookup(name: str) -> dict[str, Any] | None:
        direct = by_name.get(normalize_team(name).lower())
        if direct:
            return direct
        for canonical, row in by_name.items():
            if fuzzy_team_match(name, canonical):
                return row
        return None

    blue = lookup(blue_name)
    red = lookup(red_name)
    blue_mu = _number(blue.get("mu_total")) if blue else None
    red_mu = _number(red.get("mu_total")) if red else None
    gap = blue_mu - red_mu if blue_mu is not None and red_mu is not None else None
    return gap, {
        "source": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "blue": {"team": blue_name, "mu_total": blue_mu} if blue else None,
        "red": {"team": red_name, "mu_total": red_mu} if red else None,
        "missing": [] if gap is not None else ["team_rating_gap"],
    }


def _series_status(series_state: Mapping[str, Any], game: Mapping[str, Any]) -> str:
    if series_state.get("finished") is True or game.get("finished") is True:
        return "finished"
    return "live"


def build_live_snapshot(
    series_id: str,
    series_state: Mapping[str, Any],
    *,
    sequence_number: int | None,
    tournament: str | None = None,
    patch: str | None = None,
    elo_diff: float | None = None,
    rating_provenance: Mapping[str, Any] | None = None,
    rating_pack_id: str | None = None,
    emitted_utc: str | None = None,
) -> dict[str, Any]:
    active = _active_game(series_state)
    emitted = emitted_utc or datetime.now(timezone.utc).isoformat()
    if active is None:
        return {
            "schema_version": SCHEMA_VERSION,
            "series_id": str(series_id),
            "game_id": None,
            "game_number": None,
            "sequence_number": sequence_number,
            "emitted_utc": emitted,
            "status": "unavailable",
            "tournament": tournament,
            "patch": patch,
            "teams": {"blue": {"name": "Blue", "score": None, "players": []}, "red": {"name": "Red", "score": None, "players": []}},
            "game_state": {"clock_seconds": None, "gold_by_side": {}, "kills_by_side": {}, "objectives": {}},
            "evaluation": {
                "status": "unavailable",
                "p_blue": None,
                "p_red": None,
                "model": "draft-live-v2",
                "phase": "unknown",
                "features": {},
                "feature_sources": {},
                "missing": ["active_game"],
                "warnings": ["GRID has not supplied a usable game state yet."],
                "contributions": [],
            },
            "provenance": {"source": "GRID Series Events", "feed_sequence": sequence_number, "rating_pack_id": rating_pack_id, "rating": rating_provenance},
        }

    game_index, game = active
    teams = _teams(game)
    evaluation: LiveEvaluation = evaluate_live_state(series_state, elo_diff=elo_diff)
    clock_seconds = _clock_seconds(game)
    game_teams = _public_teams(game, series_state)
    game_state = {
        "clock_seconds": clock_seconds,
        "gold_by_side": {side: _team_gold(teams[side]) for side in ("blue", "red") if side in teams},
        "kills_by_side": {side: _team_kills(teams[side]) for side in ("blue", "red") if side in teams},
        "objectives": {side: _objectives(teams[side]) for side in ("blue", "red") if side in teams},
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "series_id": str(series_id),
        "game_id": _text(game.get("id")),
        "game_number": game_index + 1,
        "sequence_number": sequence_number,
        "emitted_utc": emitted,
        "status": _series_status(series_state, game),
        "tournament": tournament,
        "patch": patch,
        "teams": game_teams,
        "game_state": game_state,
        "evaluation": {**evaluation.as_dict(), "contributions": evaluation.contributions},
        "provenance": {
            "source": "GRID Series Events",
            "feed_sequence": sequence_number,
            "rating_pack_id": rating_pack_id,
            "rating": dict(rating_provenance or {}),
            "broadcast_synchronized": True,
        },
    }


@dataclass
class LivePublisher:
    """Publish to Blob in production or an explicit local live directory."""

    token: str | None = None
    local_root: Path = DEFAULT_LOCAL_LIVE_ROOT
    index_entries: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_environment(cls, local_root: Path | None = None) -> "LivePublisher":
        return cls(
            token=os.environ.get("BLOB_READ_WRITE_TOKEN") or os.environ.get("VERCEL_BLOB_READ_WRITE_TOKEN"),
            local_root=Path(local_root or os.environ.get("LIVE_LOCAL_ROOT") or DEFAULT_LOCAL_LIVE_ROOT),
        )

    def _write(self, path: str, value: Mapping[str, Any], *, immutable: bool = False) -> str:
        data = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if self.token:
            return _blob_put(
                self.token,
                path,
                data,
                "application/json",
                cache_control="public, max-age=31536000, immutable" if immutable else "public, max-age=2, must-revalidate",
                allow_overwrite=not immutable,
            )
        destination = self.local_root / path.removeprefix("live/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return "/" + path

    def publish_snapshot(self, snapshot: Mapping[str, Any]) -> dict[str, Any]:
        series_id = str(snapshot["series_id"])
        sequence = snapshot.get("sequence_number")
        suffix = str(sequence) if sequence is not None else snapshot["emitted_utc"].replace(":", "").replace("+00:00", "Z")
        snapshot_path = f"live/series/{series_id}/snapshots/{suffix}.json"
        snapshot_url = self._write(snapshot_path, snapshot, immutable=True)
        pointer = {
            "schema_version": SCHEMA_VERSION,
            "series_id": series_id,
            "sequence_number": sequence,
            "emitted_utc": snapshot.get("emitted_utc"),
            "status": snapshot.get("status"),
            "evaluation_status": (snapshot.get("evaluation") or {}).get("status"),
            "state_clock_seconds": (snapshot.get("game_state") or {}).get("clock_seconds"),
            "snapshot_path": snapshot_path,
            "snapshot_url": snapshot_url,
        }
        latest_path = f"live/series/{series_id}/latest.json"
        latest_url = self._write(latest_path, pointer)
        return {**pointer, "latest_path": latest_path, "latest_url": latest_url}

    def publish_index(self, entries: list[Mapping[str, Any]] | None = None) -> str:
        if entries is not None:
            for entry in entries:
                series_id = str(entry.get("series_id") or "")
                if series_id:
                    self.index_entries[series_id] = dict(entry)
        index = {
            "schema_version": SCHEMA_VERSION,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "series": list(self.index_entries.values()),
        }
        return self._write("live/index.json", index)

    def publish_health(self, *, status: str, message: str | None = None, active_series: int = 0) -> str:
        health = {
            "schema_version": SCHEMA_VERSION,
            "status": status,
            "message": message,
            "active_series": active_series,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
        }
        return self._write("live/health.json", health)
