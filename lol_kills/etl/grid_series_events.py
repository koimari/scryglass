"""GRID Series Events WebSocket client for live, pro-only LoL feeds.

The Series Events feed is the low-latency companion to the existing GRID
Central Data / File Download bridge.  It is intentionally a transport layer:
it preserves GRID transactions and never turns an in-progress game into an
official ratings row.

GRID's documented connection is::

    wss://api.grid.gg/live-data-feed/series/{SERIES_ID}?key={AUTH_KEY}

The key is kept in the URL because that is the protocol GRID documents for
this feed.  Do not log the constructed URL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatus

from lol_kills.etl.grid_ingest import (
    ALLOWED_SERIES_TYPE,
    GRAPHQL_ENDPOINT,
    SERIES_ENDPOINT,
    _api_key,
    _assert_pro,
    _graphql,
    _series_rows,
)

SERIES_EVENTS_BASE = "wss://api.grid.gg/live-data-feed/series"
USER_AGENT = "scryglass/grid-series-events (+pro-only; preliminary-live-evaluation)"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


class GridSeriesEventsError(RuntimeError):
    """A safe, credential-free Series Events error."""


def series_events_url(
    series_id: str,
    key: str,
    *,
    use_config: bool = False,
    from_sequence_number: int | None = None,
    from_session_sequence_number: int | None = None,
) -> str:
    """Build the documented URL without exposing the key to callers/logs."""
    sid = str(series_id).strip()
    secret = str(key).strip()
    if not sid or not sid.isdigit():
        raise GridSeriesEventsError("series id must be a numeric GRID series id")
    if not secret:
        raise GridSeriesEventsError("GRID_API_KEY is required for Series Events")
    params: dict[str, str] = {"key": secret}
    if use_config:
        params["useConfig"] = "true"
    if from_sequence_number is not None:
        if from_sequence_number < 0:
            raise GridSeriesEventsError("from_sequence_number cannot be negative")
        params["fromSequenceNumber"] = str(from_sequence_number)
    if from_session_sequence_number is not None:
        if from_session_sequence_number < 0:
            raise GridSeriesEventsError("from_session_sequence_number cannot be negative")
        params["fromSessionSequenceNumber"] = str(from_session_sequence_number)
    return f"{SERIES_EVENTS_BASE}/{quote(sid, safe='')}?{urlencode(params)}"


def default_config() -> dict[str, Any]:
    """Request full state on the events useful to the live evaluator.

    GRID events normally carry deltas.  Full state on game starts, clock
    changes, kills, objectives, and the end event lets a reconnecting worker
    resynchronise without guessing missing values.  The raw transaction is
    still retained, so this is not a second interchange format.
    """
    full_state_targets = [
        {"actor": "grid", "action": "*", "target": "*"},
        {"actor": "series", "action": "started", "target": "game"},
        {"actor": "series", "action": "ended", "target": "game"},
        {"actor": "game", "action": "*", "target": "*"},
        {"actor": "team", "action": "killed", "target": "*"},
        {"actor": "team", "action": "completed", "target": "*"},
        {"actor": "player", "action": "killed", "target": "*"},
    ]
    return {
        "rules": [
            {
                "eventTypeMatcher": matcher,
                "exclude": False,
                "includeFullState": True,
            }
            for matcher in full_state_targets
        ]
    }


def _transaction_from_message(message: str | bytes) -> dict[str, Any]:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise GridSeriesEventsError("GRID sent a non-UTF-8 WebSocket message") from exc
    try:
        value = json.loads(message)
    except json.JSONDecodeError as exc:
        raise GridSeriesEventsError("GRID sent a non-JSON Series Events message") from exc
    if not isinstance(value, dict):
        raise GridSeriesEventsError("GRID Series Events messages must be JSON objects")
    return value


def transaction_sequence(transaction: Mapping[str, Any]) -> int | None:
    """Return the feed sequence used for idempotent replay/resume."""
    raw = transaction.get("sequenceNumber")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def transaction_state(transaction: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """Find the latest full Series State embedded in a transaction."""
    state = transaction.get("seriesState")
    if isinstance(state, Mapping):
        return state
    for event in transaction.get("events") or []:
        if isinstance(event, Mapping):
            state = event.get("seriesState")
            if isinstance(state, Mapping):
                return state
    return None


def _safe_status_error(exc: Exception, secret: str | None = None) -> str:
    def redact(value: str) -> str:
        return value.replace(secret, "***") if secret else value

    if isinstance(exc, InvalidStatus):
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", "unknown")
        body = getattr(response, "body", b"")
        text = body.decode("utf-8", "replace").strip() if isinstance(body, bytes) else str(body)
        return f"GRID Series Events rejected the connection (HTTP {status}): {redact(text[:180])}"
    if isinstance(exc, ConnectionClosed):
        return f"GRID Series Events closed the connection (code {exc.code})"
    return f"GRID Series Events connection failed: {type(exc).__name__}: {redact(str(exc)[:180])}"


def assert_pro_series(series_id: str, key: str | None = None) -> dict[str, Any]:
    """Validate the central-data pro gate before opening a live connection."""
    secret = key or _api_key()
    sid = str(series_id).strip()
    _assert_pro(sid, context="GRID Series Events pro-check")
    query = """
    query ($id: ID!) {
      series(id: $id) {
        id
        type
        startTimeScheduled
        tournament { id name }
        teams { baseInfo { id name } }
      }
    }
    """
    data = _graphql(secret, GRAPHQL_ENDPOINT, query, {"id": sid})
    series = data.get("series")
    if not isinstance(series, Mapping):
        raise GridSeriesEventsError(f"GRID central data has no series {sid}")
    series_type = str(series.get("type") or "").upper()
    tournament = series.get("tournament") or {}
    tournament_name = str(tournament.get("name") or "") if isinstance(tournament, Mapping) else ""
    teams = [
        str(team.get("baseInfo", {}).get("name") or "")
        for team in series.get("teams") or []
        if isinstance(team, Mapping) and isinstance(team.get("baseInfo"), Mapping)
    ]
    _assert_pro(series_type, tournament_name, teams, context=f"GRID series {sid}")
    if series_type != ALLOWED_SERIES_TYPE:
        raise GridSeriesEventsError(
            f"GRID series {sid} is {series_type!r}; only {ALLOWED_SERIES_TYPE} is allowed"
        )
    if not tournament_name or len([team for team in teams if team]) < 2:
        raise GridSeriesEventsError(f"GRID series {sid} has no resolvable pro fixture")
    return {
        "id": sid,
        "type": series_type,
        "scheduled_start": series.get("startTimeScheduled"),
        "tournament": tournament_name,
        "teams": [team for team in teams if team],
    }


def discover_live_series(
    key: str | None = None,
    *,
    lookback_minutes: int = 180,
    lookahead_minutes: int = 30,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Find currently started, unfinished pro series for a worker to follow.

    GRID does not expose a single "all live series" endpoint.  Discovery is a
    bounded Central Data window followed by Series State checks, so the
    WebSocket worker never guesses a series id or opens a feed for a private
    or non-pro fixture.
    """
    if limit < 1:
        return []
    if lookback_minutes < 0 or lookahead_minutes < 0:
        raise GridSeriesEventsError("live-series discovery windows cannot be negative")
    secret = key or _api_key()
    now = datetime.now(timezone.utc)
    start = (now - timedelta(minutes=lookback_minutes)).isoformat().replace("+00:00", "Z")
    end = (now + timedelta(minutes=lookahead_minutes)).isoformat().replace("+00:00", "Z")
    try:
        candidates = _series_rows(secret, start, end, max(limit * 3, limit))
    except Exception as exc:
        raise GridSeriesEventsError(
            f"GRID live-series discovery failed: {type(exc).__name__}: {str(exc)[:180]}"
        ) from exc
    query = """
    query ($id: ID!) {
      seriesState(id: $id) {
        id
        started
        finished
      }
    }
    """
    live: list[dict[str, Any]] = []
    for candidate in candidates:
        sid = str(candidate.get("id") or "")
        try:
            state_data = _graphql(secret, SERIES_ENDPOINT, query, {"id": sid})
        except Exception as exc:
            raise GridSeriesEventsError(
                f"GRID live-series discovery failed: {type(exc).__name__}: {str(exc)[:180]}"
            ) from exc
        state = state_data.get("seriesState")
        if not isinstance(state, Mapping) or state.get("started") is not True or state.get("finished") is True:
            continue
        live.append({**candidate, "series_state": dict(state)})
        if len(live) >= limit:
            break
    return live


async def iter_series_events(
    series_id: str,
    *,
    key: str | None = None,
    from_sequence_number: int | None = None,
    config: Mapping[str, Any] | None = None,
    reconnect: bool = False,
    max_reconnects: int = 2,
    backoff_seconds: float = 2.0,
) -> AsyncIterator[dict[str, Any]]:
    """Yield raw GRID transactions, resuming by sequence after a drop.

    A normal close means GRID has ended the feed and is returned to the caller.
    Abnormal closes may be retried, but the retry budget is deliberately small:
    GRID documents a maximum of five connection requests per minute per key.
    """
    secret = key or _api_key()
    sid = str(series_id).strip()
    if config is not None and not isinstance(config, Mapping):
        raise GridSeriesEventsError("Series Events config must be a JSON object")
    last_sequence = from_sequence_number
    attempts = 0

    while True:
        url = series_events_url(
            sid,
            secret,
            use_config=config is not None,
            from_sequence_number=last_sequence,
        )
        try:
            async with websockets.connect(
                url,
                additional_headers={"User-Agent": USER_AGENT},
                max_size=MAX_MESSAGE_BYTES,
                ping_interval=20,
                ping_timeout=30,
                open_timeout=15,
                close_timeout=5,
            ) as socket:
                if config is not None:
                    await socket.send(json.dumps(config, separators=(",", ":")))
                async for message in socket:
                    transaction = _transaction_from_message(message)
                    sequence = transaction_sequence(transaction)
                    if sequence is not None and last_sequence is not None and sequence <= last_sequence:
                        continue
                    if sequence is not None:
                        last_sequence = sequence
                    yield transaction
            return
        except ConnectionClosed as exc:
            if exc.code in (1000, 1001) or not reconnect or attempts >= max_reconnects:
                if exc.code in (1000, 1001):
                    return
                raise GridSeriesEventsError(_safe_status_error(exc, secret)) from exc
            attempts += 1
        except Exception as exc:
            if not reconnect or attempts >= max_reconnects:
                if isinstance(exc, GridSeriesEventsError):
                    raise
                raise GridSeriesEventsError(_safe_status_error(exc, secret)) from exc
        attempts += 1
        await asyncio.sleep(min(30.0, backoff_seconds * (2 ** (attempts - 1))))


async def iter_series_states(
    series_id: str,
    *,
    key: str | None = None,
    from_sequence_number: int | None = None,
    config: Mapping[str, Any] | None = None,
    reconnect: bool = False,
) -> AsyncIterator[tuple[int | None, Mapping[str, Any]]]:
    """Yield full Series State snapshots when GRID includes one in a transaction."""
    async for transaction in iter_series_events(
        series_id,
        key=key,
        from_sequence_number=from_sequence_number,
        config=config,
        reconnect=reconnect,
    ):
        state = transaction_state(transaction)
        if state is not None:
            yield transaction_sequence(transaction), state


async def _run_cli(args: argparse.Namespace) -> int:
    key = _api_key(Path(args.env_file).expanduser() if args.env_file else None)
    if args.discover_live:
        rows = discover_live_series(key, limit=args.discovery_limit)
        for row in rows:
            print(
                f"[grid-events] live series={row['id']} tournament={row['tournament']} "
                f"teams={', '.join(row['teams'])}"
            )
        print(f"[grid-events] live_series={len(rows)}")
        return 0
    fixture = assert_pro_series(args.series_id, key)
    print(
        f"[grid-events] connecting series={fixture['id']} tournament={fixture['tournament']} "
        f"teams={', '.join(fixture['teams'])}"
    )
    config = default_config() if args.full_state else None
    out = Path(args.out).expanduser() if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    last_sequence: int | None = None
    started = asyncio.get_running_loop().time()
    stream = iter_series_events(
        args.series_id,
        key=key,
        from_sequence_number=args.from_sequence,
        config=config,
        reconnect=args.reconnect,
    )
    iterator = stream.__aiter__()
    try:
        while True:
            remaining = max(0.0, args.seconds - (asyncio.get_running_loop().time() - started))
            if args.seconds and remaining <= 0:
                break
            try:
                transaction = await asyncio.wait_for(
                    iterator.__anext__(),
                    timeout=remaining if args.seconds else None,
                )
            except asyncio.TimeoutError:
                break
            except StopAsyncIteration:
                break
            count += 1
            last_sequence = transaction_sequence(transaction) or last_sequence
            if out:
                with out.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(transaction, separators=(",", ":")) + "\n")
            if args.limit and count >= args.limit:
                break
    finally:
        await stream.aclose()
    print(f"[grid-events] transactions={count} last_sequence={last_sequence or '—'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-id")
    parser.add_argument("--discover-live", action="store_true")
    parser.add_argument("--discovery-limit", type=int, default=10)
    parser.add_argument("--from-sequence", type=int, default=None)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--full-state", action="store_true")
    parser.add_argument("--reconnect", action="store_true")
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--env-file", type=str, default=None)
    args = parser.parse_args(argv)
    if bool(args.series_id) == args.discover_live:
        parser.error("provide exactly one of --series-id or --discover-live")
    try:
        return asyncio.run(_run_cli(args))
    except GridSeriesEventsError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
