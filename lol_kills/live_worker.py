"""Persistent GRID Series Events worker for verified public live snapshots.

Production usage keeps this process running in a small container or worker.
The web app only reads the resulting Blob pointers; it never holds the GRID
credential and never connects to the feed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.etl.grid_series_events import (
    GridSeriesEventsError,
    _api_key,
    assert_pro_series,
    default_config,
    discover_live_series,
    iter_series_states,
)
from lol_kills.live_snapshots import (
    LivePublisher,
    build_live_snapshot,
    default_ratings_path,
    resolve_team_rating_gap,
)

ROOT = Path(__file__).resolve().parents[1]


def _pack_id() -> str | None:
    latest_path = ROOT / "apps" / "lol-atlas" / "public" / "packs" / "latest.json"
    try:
        value = json.loads(latest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return str(value.get("pack_id")) if isinstance(value, Mapping) and value.get("pack_id") else None


def _state_teams(state: Mapping[str, Any]) -> tuple[str | None, str | None]:
    games = [game for game in state.get("games") or [] if isinstance(game, Mapping)]
    if not games:
        return None, None
    game = next((item for item in reversed(games) if item.get("finished") is not True), games[-1])
    names: dict[str, str] = {}
    for team in game.get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        side = str(team.get("side") or "").lower()
        name = str(team.get("name") or "").strip()
        if side in {"blue", "red"} and name:
            names[side] = name
    return names.get("blue"), names.get("red")


def _index_entry(snapshot: Mapping[str, Any], pointer: Mapping[str, Any], tournament: str | None) -> dict[str, Any]:
    teams = snapshot.get("teams") or {}
    return {
        **dict(pointer),
        "tournament": tournament,
        "game_number": snapshot.get("game_number"),
        "teams": {
            side: {"name": (teams.get(side) or {}).get("name")} for side in ("blue", "red")
        },
    }


async def _publish_state(
    *,
    publisher: LivePublisher,
    series_id: str,
    state: Mapping[str, Any],
    sequence_number: int | None,
    tournament: str | None,
    pack_id: str | None,
    ratings_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    blue_name, red_name = _state_teams(state)
    elo_diff, rating = resolve_team_rating_gap(blue_name, red_name, ratings_path)
    snapshot = build_live_snapshot(
        series_id,
        state,
        sequence_number=sequence_number,
        tournament=tournament,
        elo_diff=elo_diff,
        rating_provenance=rating,
        rating_pack_id=pack_id,
        emitted_utc=datetime.now(timezone.utc).isoformat(),
    )
    pointer = publisher.publish_snapshot(snapshot)
    return snapshot, _index_entry(snapshot, pointer, tournament)


async def _follow_series(
    fixture: Mapping[str, Any],
    *,
    key: str,
    publisher: LivePublisher,
    pack_id: str | None,
    ratings_path: Path,
    seconds: float,
    stop_after_one: bool = False,
) -> list[dict[str, Any]]:
    series_id = str(fixture["id"])
    stream = iter_series_states(
        series_id,
        key=key,
        config=default_config(),
        reconnect=True,
    )
    iterator = stream.__aiter__()
    entries: list[dict[str, Any]] = []
    started = asyncio.get_running_loop().time()
    try:
        while True:
            remaining = max(0.0, seconds - (asyncio.get_running_loop().time() - started)) if seconds else None
            if remaining is not None and remaining <= 0:
                break
            try:
                sequence, state = await asyncio.wait_for(iterator.__anext__(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except StopAsyncIteration:
                break
            _, entry = await _publish_state(
                publisher=publisher,
                series_id=series_id,
                state=state,
                sequence_number=sequence,
                tournament=str(fixture.get("tournament") or "") or None,
                pack_id=pack_id,
                ratings_path=ratings_path,
            )
            entries = [item for item in entries if item.get("series_id") != series_id]
            entries.append(entry)
            publisher.publish_index([entry])
            if stop_after_one:
                break
    finally:
        await stream.aclose()
    return entries


async def _state_file_mode(args: argparse.Namespace, publisher: LivePublisher) -> int:
    state_path = Path(args.state_file).expanduser()
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"state file is not valid JSON: {state_path}") from exc
    if not isinstance(state, Mapping):
        raise SystemExit("state file must contain a JSON object")
    _, entry = await _publish_state(
        publisher=publisher,
        series_id=str(args.series_id),
        state=state,
        sequence_number=args.sequence,
        tournament=args.tournament,
        pack_id=args.pack_id or _pack_id(),
        ratings_path=Path(args.ratings_file).expanduser() if args.ratings_file else default_ratings_path(),
    )
    publisher.publish_index([entry])
    publisher.publish_health(status="ok", message="Published one verified local state snapshot.", active_series=1)
    print(f"[live-worker] published series={args.series_id} sequence={args.sequence or '—'}")
    return 0


async def run_worker(args: argparse.Namespace) -> int:
    publisher = LivePublisher.from_environment(Path(args.local_root).expanduser() if args.local_root else None)
    if args.state_file:
        return await _state_file_mode(args, publisher)

    key = _api_key(Path(args.env_file).expanduser() if args.env_file else None)
    pack_id = args.pack_id or _pack_id()
    ratings_path = Path(args.ratings_file).expanduser() if args.ratings_file else default_ratings_path()
    fixed_fixture = assert_pro_series(args.series_id, key) if args.series_id else None
    entries: dict[str, dict[str, Any]] = {}
    tasks: dict[str, asyncio.Task[list[dict[str, Any]]]] = {}
    started = asyncio.get_running_loop().time()
    publisher.publish_health(status="starting", message="GRID live worker is starting.")

    while True:
        if args.seconds and asyncio.get_running_loop().time() - started >= args.seconds:
            break
        fixtures = [fixed_fixture] if fixed_fixture else discover_live_series(key, limit=args.discovery_limit)
        for fixture in fixtures:
            series_id = str(fixture["id"])
            task = tasks.get(series_id)
            if task is None or task.done():
                tasks[series_id] = asyncio.create_task(
                    _follow_series(
                        fixture,
                        key=key,
                        publisher=publisher,
                        pack_id=pack_id,
                        ratings_path=ratings_path,
                        seconds=args.series_seconds,
                        stop_after_one=args.once,
                    )
                )
        for series_id, task in list(tasks.items()):
            if not task.done():
                continue
            try:
                for entry in task.result():
                    entries[series_id] = entry
            except Exception as exc:
                publisher.publish_health(
                    status="degraded",
                    message=f"Series {series_id} stopped: {type(exc).__name__}",
                    active_series=len(tasks),
                )
            del tasks[series_id]
        if entries:
            publisher.publish_index(list(entries.values()))
        publisher.publish_health(status="ok", active_series=len(tasks))
        if args.once:
            if not tasks:
                break
            await asyncio.sleep(0.1)
        else:
            await asyncio.sleep(args.discovery_seconds)
    for task in tasks.values():
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks.values(), return_exceptions=True)
    publisher.publish_health(status="stopped", message="GRID live worker stopped.", active_series=0)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-id")
    parser.add_argument("--state-file", help="Publish one captured Series State JSON without opening GRID")
    parser.add_argument("--sequence", type=int, default=None)
    parser.add_argument("--tournament", default=None)
    parser.add_argument("--once", action="store_true", help="Wait for one state per discovered series")
    parser.add_argument("--discovery-limit", type=int, default=10)
    parser.add_argument("--discovery-seconds", type=float, default=30.0)
    parser.add_argument("--series-seconds", type=float, default=3600.0)
    parser.add_argument("--seconds", type=float, default=0.0, help="Stop the worker after this many seconds")
    parser.add_argument("--pack-id", default=None)
    parser.add_argument("--ratings-file", default=None)
    parser.add_argument("--local-root", default=None, help="Local live directory when Blob token is absent")
    parser.add_argument("--env-file", default=None)
    args = parser.parse_args(argv)
    if bool(args.state_file) != bool(args.series_id):
        parser.error("--state-file requires --series-id")
    try:
        return asyncio.run(run_worker(args))
    except GridSeriesEventsError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
