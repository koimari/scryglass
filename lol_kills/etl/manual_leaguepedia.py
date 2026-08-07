"""Leakage-safe manual Leaguepedia forecast ledger.

This module is the small contract layer between human/source review and the
local draft engine.  It deliberately separates three immutable phases:

``freeze_pregame``
    Stores only information that was available before the map started.
``score_frozen``
    Runs the deterministic local draft engine against the frozen projection.
``reveal_outcome``
    Adds the winner/result only after the score has been sealed.

The contract is intentionally stricter than a normal historical replay.  A
current Leaguepedia match-history page may contain the result and may have
been revised after the match.  Such a page can be retained as retrospective
evidence, but it cannot silently become strict pregame evidence.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.v2.data.common import (
    ROLES,
    ContractError,
    canonical_json_bytes,
    canonicalize_role,
    parse_rfc3339,
    sha256_bytes,
    sha256_canonical_object,
    to_rfc3339,
)


SCHEMA_VERSION = "scryglass:leaguepedia-manual-run:v1"
PREGAME_PHASE = "pregame_frozen"
SCORED_PHASE = "scored"
REVEALED_PHASE = "outcome_revealed"
MODES = frozenset({"strict", "retrospective"})

# These keys are outcome/finished-state concepts, not permitted pregame
# features.  The check is recursive so a result hidden inside a nested side,
# source row, or free-form metadata object cannot pass accidentally.
FORBIDDEN_PREGAME_KEYS = frozenset(
    {
        "winner",
        "winning_team",
        "winning_side",
        "result",
        "win_loss",
        "outcome",
        "score",
        "kills",
        "gold",
        "duration",
        "gamelength",
        "game_length",
        "victory",
        "defeat",
        "finished_at",
        "ended_at",
        "vod",
    }
)

ACTIVE_STARTER_STATUSES = frozenset(
    {"active", "starter", "confirmed_starter", "temporary_starter", "confirmed_substitute"}
)
BLOCKING_STATUSES = frozenset({"leave", "inactive", "released", "suspended"})
STATUS_PRECEDENCE = {
    "active": 10,
    "starter": 50,
    "confirmed_starter": 100,
    "confirmed_substitute": 150,
    "temporary_starter": 200,
}


class ManualLeaguepediaError(ContractError):
    """Base error for the manual Leaguepedia ledger."""


class PregameLeakageError(ManualLeaguepediaError):
    """Raised when post-start or outcome information enters the pregame side."""


class RosterResolutionError(ManualLeaguepediaError):
    """Raised when a time-sliced five-player lineup is unavailable or ambiguous."""


class EngineUnavailableError(ManualLeaguepediaError):
    """Raised when the local scorer cannot produce a score."""


def _text(value: Any, field: str) -> str:
    if value is None or not str(value).strip():
        raise ManualLeaguepediaError(f"{field} is required")
    return str(value).strip()


def _parse_time(value: Any, field: str) -> datetime:
    try:
        return parse_rfc3339(_text(value, field))
    except ContractError as exc:
        raise ManualLeaguepediaError(f"{field} must be RFC-3339 UTC") from exc


def _time(value: datetime) -> str:
    return to_rfc3339(value)


def _now() -> str:
    return _time(datetime.now(timezone.utc))


def _sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes(), label=str(path))


def _slug(value: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return candidate or "page"


def leaguepedia_api_url(title: str, *, rendered: bool = False, before: str | None = None) -> str:
    """Build a deterministic Leaguepedia API URL.

    ``before`` asks MediaWiki for the latest page revision at or before the
    supplied cutoff.  It is source-time evidence, not proof that the current
    retrieval itself happened before a historical match; callers must retain
    the separate ``available_at`` timestamp.
    """

    page = _text(title, "title")
    if rendered:
        params = {
            "action": "parse",
            "page": page,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }
    else:
        params = {
            "action": "query",
            "prop": "revisions",
            "titles": page,
            "rvprop": "ids|timestamp|content",
            "rvslots": "main",
            "rvlimit": "1",
            "format": "json",
            "formatversion": "2",
        }
        if before is not None:
            params["rvstart"] = _time(_parse_time(before, "before"))
            params["rvdir"] = "older"
    return "https://lol.fandom.com/api.php?" + urllib.parse.urlencode(params)


def capture_leaguepedia_page(
    title: str,
    output_dir: Path,
    *,
    observed_at: str | None = None,
    before: str | None = None,
    user_agent: str = "Scryglass-manual-Leaguepedia/1.0",
) -> list[dict[str, Any]]:
    """Capture raw revision and rendered payloads for one Leaguepedia page."""

    observed = _parse_time(observed_at or _now(), "observed_at")
    output_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    base = _slug(title)
    for rendered in (False, True):
        url = leaguepedia_api_url(title, rendered=rendered, before=before)
        request = urllib.request.Request(url, headers={"User-Agent": user_agent})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read()
        except OSError as exc:
            raise ManualLeaguepediaError(f"Leaguepedia capture failed for {title}: {exc}") from exc
        filename = f"{base}-{'rendered' if rendered else 'revision'}-api.json"
        path = output_dir / filename
        path.write_bytes(raw)
        entry: dict[str, Any] = {
            "page": title,
            "api_url": url,
            "raw_file": filename,
            "sha256": sha256_bytes(raw, label=filename),
            "available_at": _time(observed),
            "capture_kind": "rendered" if rendered else "revision",
        }
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ManualLeaguepediaError(f"Leaguepedia returned non-JSON for {title}") from exc
        if not rendered:
            pages = parsed.get("query", {}).get("pages", [])
            if pages:
                page = pages[0]
                revisions = page.get("revisions", [])
                if revisions:
                    entry["revision_id"] = revisions[0].get("revid")
                    entry["source_updated_at"] = revisions[0].get("timestamp")
        entries.append(entry)
    return entries


def _key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def _walk_forbidden(value: Any, path: str = "$") -> list[str]:
    hits: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = _key(str(raw_key))
            child_path = f"{path}.{raw_key}"
            if key in FORBIDDEN_PREGAME_KEYS:
                hits.append(child_path)
            hits.extend(_walk_forbidden(child, child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            hits.extend(_walk_forbidden(child, f"{path}[{index}]"))
    return hits


def assert_no_outcome_fields(value: Any) -> None:
    """Reject outcome-shaped fields anywhere in a pregame projection."""

    hits = _walk_forbidden(value)
    if hits:
        raise PregameLeakageError("outcome fields are not allowed in pregame input: " + ", ".join(hits))


def _require_sha(value: Any, field: str) -> str:
    candidate = _text(value, field)
    if re.fullmatch(r"[0-9a-f]{64}", candidate) is None:
        raise ManualLeaguepediaError(f"{field} must be a lowercase SHA-256 digest")
    return candidate


def _validate_source_snapshots(
    snapshots: Any,
    *,
    as_of: datetime,
    event_start: datetime,
    mode: str,
) -> list[dict[str, Any]]:
    if not isinstance(snapshots, list) or not snapshots:
        raise ManualLeaguepediaError("source_snapshots must be a non-empty list")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(snapshots):
        if not isinstance(raw, Mapping):
            raise ManualLeaguepediaError(f"source_snapshots[{index}] must be an object")
        available_at = _parse_time(raw.get("available_at"), f"source_snapshots[{index}].available_at")
        if mode == "strict" and available_at > as_of:
            raise PregameLeakageError(
                f"source snapshot {index} became available after the forecast as_of"
            )
        if mode == "strict" and available_at >= event_start:
            raise PregameLeakageError(
                f"source snapshot {index} became available at/after event_start"
            )
        item = dict(raw)
        item["snapshot_id"] = _text(raw.get("snapshot_id"), f"source_snapshots[{index}].snapshot_id")
        item["sha256"] = _require_sha(raw.get("sha256"), f"source_snapshots[{index}].sha256")
        item["available_at"] = _time(available_at)
        if raw.get("source_updated_at") is not None:
            item["source_updated_at"] = _time(
                _parse_time(raw.get("source_updated_at"), f"source_snapshots[{index}].source_updated_at")
            )
        normalized.append(item)
    return normalized


def _validate_side(side: Any, side_name: str) -> dict[str, Any]:
    if not isinstance(side, Mapping):
        raise ManualLeaguepediaError(f"{side_name} must be an object")
    team = _text(side.get("team"), f"{side_name}.team")
    picks = side.get("picks")
    if not isinstance(picks, list) or len(picks) != len(ROLES):
        raise ManualLeaguepediaError(f"{side_name}.picks must contain exactly five champions")
    normalized_picks = [_text(pick, f"{side_name}.picks[{i}]") for i, pick in enumerate(picks)]

    raw_players = side.get("players")
    if not isinstance(raw_players, list) or len(raw_players) != len(ROLES):
        raise ManualLeaguepediaError(f"{side_name}.players must contain exactly five role rows")
    players: list[dict[str, str]] = []
    seen_roles: set[str] = set()
    seen_players: set[str] = set()
    for index, raw in enumerate(raw_players):
        if not isinstance(raw, Mapping):
            raise ManualLeaguepediaError(f"{side_name}.players[{index}] must be an object")
        try:
            role = canonicalize_role(_text(raw.get("role"), f"{side_name}.players[{index}].role"))
        except ContractError as exc:
            raise ManualLeaguepediaError(f"{side_name}.players[{index}].role is invalid") from exc
        player = _text(raw.get("player"), f"{side_name}.players[{index}].player")
        if role in seen_roles:
            raise ManualLeaguepediaError(f"{side_name} has duplicate role {role}")
        if player in seen_players:
            raise ManualLeaguepediaError(f"{side_name} has duplicate player {player}")
        seen_roles.add(role)
        seen_players.add(player)
        players.append({"role": role, "player": player})
    if seen_roles != set(ROLES):
        raise ManualLeaguepediaError(f"{side_name}.players must cover top,jungle,mid,bot,support")

    output = dict(side)
    output["team"] = team
    output["picks"] = normalized_picks
    output["players"] = sorted(players, key=lambda row: ROLES.index(row["role"]))
    return output


def _validate_roster_events(
    events: Any,
    *,
    as_of: datetime,
    event_start: datetime,
    mode: str,
) -> list[dict[str, Any]]:
    if events is None:
        return []
    if not isinstance(events, list):
        raise ManualLeaguepediaError("roster_events must be a list")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise ManualLeaguepediaError(f"roster_events[{index}] must be an object")
        try:
            role = canonicalize_role(_text(raw.get("role"), f"roster_events[{index}].role"))
        except ContractError as exc:
            raise ManualLeaguepediaError(f"roster_events[{index}].role is invalid") from exc
        effective_from = _parse_time(raw.get("effective_from"), f"roster_events[{index}].effective_from")
        effective_to = (
            _parse_time(raw.get("effective_to"), f"roster_events[{index}].effective_to")
            if raw.get("effective_to") is not None
            else None
        )
        if effective_to is not None and effective_to <= effective_from:
            raise ManualLeaguepediaError(f"roster_events[{index}] interval is not forward")
        available_at = _parse_time(raw.get("available_at"), f"roster_events[{index}].available_at")
        if mode == "strict" and available_at > as_of:
            raise PregameLeakageError(f"roster event {index} was unavailable at forecast as_of")
        if mode == "strict" and available_at >= event_start:
            raise PregameLeakageError(f"roster event {index} was not available before event_start")
        status = _key(_text(raw.get("status"), f"roster_events[{index}].status"))
        if status not in ACTIVE_STARTER_STATUSES | BLOCKING_STATUSES:
            raise ManualLeaguepediaError(f"roster_events[{index}].status is unsupported: {status}")
        item = dict(raw)
        item["team"] = _text(raw.get("team"), f"roster_events[{index}].team")
        item["role"] = role
        item["player"] = _text(raw.get("player"), f"roster_events[{index}].player")
        item["status"] = status
        item["effective_from"] = _time(effective_from)
        item["effective_to"] = _time(effective_to) if effective_to is not None else None
        item["available_at"] = _time(available_at)
        item["precedence"] = int(raw.get("precedence", STATUS_PRECEDENCE.get(status, 0)))
        if raw.get("source_snapshot_id") is not None:
            item["source_snapshot_id"] = _text(
                raw.get("source_snapshot_id"), f"roster_events[{index}].source_snapshot_id"
            )
        if raw.get("source_sha256") is not None:
            item["source_sha256"] = _require_sha(raw.get("source_sha256"), f"roster_events[{index}].source_sha256")
        normalized.append(item)
    return normalized


def resolve_time_sliced_lineup(
    events: Sequence[Mapping[str, Any]],
    team: str,
    *,
    event_start: str,
    as_of: str,
) -> dict[str, Any]:
    """Resolve one team's expected five using effective roster events.

    A leave/inactive row blocks the named player while a temporary-starter row
    can take the role.  Equal-precedence candidates are deliberately
    ambiguous rather than silently selecting one.
    """

    event_at = _parse_time(event_start, "event_start")
    forecast_at = _parse_time(as_of, "as_of")
    team_name = _text(team, "team")
    candidates: dict[str, list[Mapping[str, Any]]] = {role: [] for role in ROLES}
    blocked: dict[str, set[str]] = {role: set() for role in ROLES}
    for index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise RosterResolutionError(f"roster event {index} must be an object")
        if str(raw.get("team", "")).strip() != team_name:
            continue
        role = canonicalize_role(_text(raw.get("role"), f"roster event {index}.role"))
        effective_from = _parse_time(raw.get("effective_from"), f"roster event {index}.effective_from")
        effective_to = (
            _parse_time(raw.get("effective_to"), f"roster event {index}.effective_to")
            if raw.get("effective_to") is not None
            else None
        )
        available_at = _parse_time(raw.get("available_at"), f"roster event {index}.available_at")
        if available_at > forecast_at:
            continue
        if not (effective_from <= event_at and (effective_to is None or event_at < effective_to)):
            continue
        status = _key(_text(raw.get("status"), f"roster event {index}.status"))
        player = _text(raw.get("player"), f"roster event {index}.player")
        if status in BLOCKING_STATUSES:
            blocked[role].add(player)
        elif status in ACTIVE_STARTER_STATUSES:
            candidates[role].append(raw)

    selected: list[dict[str, Any]] = []
    errors: list[str] = []
    for role in ROLES:
        eligible = [row for row in candidates[role] if str(row.get("player")) not in blocked[role]]
        if not eligible:
            errors.append(f"missing active candidate for role={role}")
            continue
        best_precedence = max(int(row.get("precedence", STATUS_PRECEDENCE.get(_key(str(row.get("status"))), 0))) for row in eligible)
        best = [row for row in eligible if int(row.get("precedence", STATUS_PRECEDENCE.get(_key(str(row.get("status"))), 0))) == best_precedence]
        players = {str(row.get("player")) for row in best}
        if len(players) != 1:
            errors.append(f"ambiguous active candidates for role={role}: {sorted(players)}")
            continue
        row = dict(best[0])
        row["role"] = role
        row["player"] = next(iter(players))
        selected.append(row)

    player_names = [row["player"] for row in selected]
    if len(selected) == len(ROLES) and len(set(player_names)) != len(player_names):
        errors.append("same player selected for multiple roles")
    if errors:
        return {
            "status": "unavailable",
            "team": team_name,
            "players": sorted(selected, key=lambda row: ROLES.index(row["role"])),
            "errors": sorted(set(errors)),
        }
    return {
        "status": "ok",
        "team": team_name,
        "players": sorted(selected, key=lambda row: ROLES.index(row["role"])),
        "errors": [],
    }


def freeze_pregame(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and seal a pregame projection without any outcome fields."""

    if not isinstance(payload, Mapping):
        raise ManualLeaguepediaError("pregame input must be an object")
    assert_no_outcome_fields(payload)
    mode = _key(str(payload.get("mode", "strict")))
    if mode not in MODES:
        raise ManualLeaguepediaError(f"mode must be one of {sorted(MODES)}")
    event_start = _parse_time(payload.get("event_start"), "event_start")
    draft_locked_at = _parse_time(payload.get("draft_locked_at"), "draft_locked_at")
    as_of = _parse_time(payload.get("as_of"), "as_of")
    if not (draft_locked_at <= as_of < event_start):
        raise PregameLeakageError("require draft_locked_at <= as_of < event_start")
    competition = payload.get("competition")
    if not isinstance(competition, Mapping):
        raise ManualLeaguepediaError("competition must be an object")
    normalized_competition = dict(competition)
    normalized_competition["league"] = _text(competition.get("league"), "competition.league")
    normalized_competition["scope"] = _text(competition.get("scope", "regional"), "competition.scope")

    pregame = {
        "schema_version": SCHEMA_VERSION,
        "fixture_id": _text(payload.get("fixture_id"), "fixture_id"),
        "mode": mode,
        "event_start": _time(event_start),
        "draft_locked_at": _time(draft_locked_at),
        "as_of": _time(as_of),
        "competition": normalized_competition,
        "blue": _validate_side(payload.get("blue"), "blue"),
        "red": _validate_side(payload.get("red"), "red"),
        "source_snapshots": _validate_source_snapshots(
            payload.get("source_snapshots"),
            as_of=as_of,
            event_start=event_start,
            mode=mode,
        ),
        "roster_events": _validate_roster_events(
            payload.get("roster_events"),
            as_of=as_of,
            event_start=event_start,
            mode=mode,
        ),
    }
    assert_no_outcome_fields(pregame)
    return {
        "schema_version": SCHEMA_VERSION,
        "phase": PREGAME_PHASE,
        "pregame": pregame,
        "pregame_sha256": sha256_canonical_object(pregame),
        "outcome_status": "unrevealed",
    }


def _runtime_temporal_check(run: Mapping[str, Any], runtime_as_of: str) -> None:
    pregame = run["pregame"]
    if pregame["mode"] != "strict":
        return
    runtime_at = _parse_time(runtime_as_of, "runtime_as_of")
    event_at = _parse_time(pregame["event_start"], "pregame.event_start")
    if runtime_at >= event_at:
        raise PregameLeakageError(
            "strict scoring requires the model runtime as_of to precede event_start"
        )


def attach_score(
    frozen: Mapping[str, Any],
    score_output: Mapping[str, Any],
    *,
    runtime_as_of: str,
    runtime_sha256: str,
    runner_sha256: str,
    score_module_sha256: str,
    scored_at: str | None = None,
) -> dict[str, Any]:
    """Attach a score only to a valid, outcome-free frozen input."""

    verify_run(frozen, require_score=False, require_outcome=False)
    if frozen.get("phase") != PREGAME_PHASE:
        raise ManualLeaguepediaError("score attachment requires a pregame_frozen run")
    if not isinstance(score_output, Mapping):
        raise EngineUnavailableError("engine output must be an object")
    assert_no_outcome_fields(frozen["pregame"])
    _runtime_temporal_check(frozen, runtime_as_of)
    result = copy.deepcopy(dict(frozen))
    result["phase"] = SCORED_PHASE
    result["score"] = {
        "status": "complete",
        "scored_at": scored_at or _now(),
        "input_pregame_sha256": frozen["pregame_sha256"],
        "runtime_as_of": runtime_as_of,
        "runtime_sha256": _require_sha(runtime_sha256, "runtime_sha256"),
        "runner_sha256": _require_sha(runner_sha256, "runner_sha256"),
        "score_module_sha256": _require_sha(score_module_sha256, "score_module_sha256"),
        "output_sha256": sha256_canonical_object(score_output),
        "output": copy.deepcopy(dict(score_output)),
    }
    verify_run(result, require_score=True, require_outcome=False)
    return result


def reveal_outcome(
    scored: Mapping[str, Any],
    outcome: Mapping[str, Any],
    *,
    revealed_at: str | None = None,
) -> dict[str, Any]:
    """Add a result after scoring; the sealed pregame object is unchanged."""

    verify_run(scored, require_score=True, require_outcome=False)
    if scored.get("phase") != SCORED_PHASE:
        raise ManualLeaguepediaError("outcome reveal requires a scored run")
    if not isinstance(outcome, Mapping):
        raise ManualLeaguepediaError("outcome must be an object")
    winner = _text(outcome.get("winner"), "outcome.winner")
    if winner not in {scored["pregame"]["blue"]["team"], scored["pregame"]["red"]["team"]}:
        raise ManualLeaguepediaError("outcome.winner must be one of the frozen teams")
    reveal_time = _parse_time(revealed_at or outcome.get("revealed_at"), "revealed_at")
    scored_time = _parse_time(scored["score"]["scored_at"], "score.scored_at")
    event_time = _parse_time(scored["pregame"]["event_start"], "pregame.event_start")
    if reveal_time < scored_time:
        raise PregameLeakageError("outcome reveal must occur after score sealing")
    if reveal_time < event_time:
        raise PregameLeakageError("outcome reveal must not precede event_start")
    normalized = dict(outcome)
    normalized["winner"] = winner
    normalized["revealed_at"] = _time(reveal_time)
    result = copy.deepcopy(dict(scored))
    result["phase"] = REVEALED_PHASE
    result["outcome_status"] = "revealed"
    result["outcome"] = normalized
    result["outcome_sha256"] = sha256_canonical_object(normalized)
    verify_run(result, require_score=True, require_outcome=True)
    return result


def verify_run(
    run: Mapping[str, Any],
    *,
    require_score: bool = False,
    require_outcome: bool = False,
) -> None:
    """Verify phase ordering, hashes, and the absence of pregame leakage."""

    if not isinstance(run, Mapping) or run.get("schema_version") != SCHEMA_VERSION:
        raise ManualLeaguepediaError("invalid manual Leaguepedia run schema")
    pregame = run.get("pregame")
    if not isinstance(pregame, Mapping):
        raise ManualLeaguepediaError("run.pregame is required")
    assert_no_outcome_fields(pregame)
    if run.get("pregame_sha256") != sha256_canonical_object(pregame):
        raise ManualLeaguepediaError("pregame_sha256 mismatch")
    if require_score and not isinstance(run.get("score"), Mapping):
        raise ManualLeaguepediaError("score is required")
    if isinstance(run.get("score"), Mapping):
        score = run["score"]
        if score.get("input_pregame_sha256") != run["pregame_sha256"]:
            raise ManualLeaguepediaError("score input hash does not match frozen pregame hash")
        output = score.get("output")
        if not isinstance(output, Mapping):
            raise ManualLeaguepediaError("score.output is required")
        if score.get("output_sha256") != sha256_canonical_object(output):
            raise ManualLeaguepediaError("score output hash mismatch")
        _parse_time(score.get("scored_at"), "score.scored_at")
        _parse_time(score.get("runtime_as_of"), "score.runtime_as_of") if pregame["mode"] == "strict" else None
    if require_outcome and not isinstance(run.get("outcome"), Mapping):
        raise ManualLeaguepediaError("outcome is required")
    if isinstance(run.get("outcome"), Mapping):
        outcome = run["outcome"]
        if run.get("outcome_sha256") != sha256_canonical_object(outcome):
            raise ManualLeaguepediaError("outcome hash mismatch")
        reveal_time = _parse_time(outcome.get("revealed_at"), "outcome.revealed_at")
        if isinstance(run.get("score"), Mapping):
            scored_time = _parse_time(run["score"].get("scored_at"), "score.scored_at")
            if reveal_time < scored_time:
                raise PregameLeakageError("outcome was revealed before score sealing")


def score_frozen(
    run: Mapping[str, Any],
    *,
    repo: Path,
    scored_at: str | None = None,
) -> dict[str, Any]:
    """Run the checked-in local scorer against only the frozen projection."""

    verify_run(run, require_score=False, require_outcome=False)
    if run.get("phase") != PREGAME_PHASE:
        raise ManualLeaguepediaError("score_frozen requires a pregame_frozen run")
    pregame = run["pregame"]
    app = repo / "apps" / "lol-atlas"
    tsx = app / "node_modules" / ".bin" / "tsx"
    runner = Path("/Users/river/.codex/skills/who-wins-this-game/scripts/who_wins_game.ts")
    runtime_path = app / "data" / "draft" / "runtime.json"
    score_module = app / "src" / "lib" / "draftScore.ts"
    if not tsx.exists() or not runner.exists() or not runtime_path.exists() or not score_module.exists():
        raise EngineUnavailableError("local draft scorer or runtime files are missing")
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_as_of = _text(runtime.get("as_of"), "runtime.as_of")
    _runtime_temporal_check(run, runtime_as_of)
    context_path = app / "data" / "draft" / "context.json"
    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineUnavailableError("local draft context is missing or invalid") from exc
    known_players = {
        str(name).casefold()
        for name in (context.get("players") or {})
    }
    requested_players = [
        player["player"]
        for side_name in ("blue", "red")
        for player in pregame[side_name]["players"]
    ]
    player_context_available = all(
        player.casefold() in known_players for player in requested_players
    )
    # The checked-in runner automatically resolves a team's current roster if
    # no explicit player flags are supplied.  That is unsafe for a historical
    # row whose player context is absent, so use non-team sentinel names to
    # force a genuinely player-neutral calculation instead of silently
    # substituting a current roster.
    blue_name = pregame["blue"]["team"] if player_context_available else "__historical_blue_no_player_context__"
    red_name = pregame["red"]["team"] if player_context_available else "__historical_red_no_player_context__"
    command = [
        str(tsx),
        str(runner),
        "--blue",
        ",".join(pregame["blue"]["picks"]),
        "--red",
        ",".join(pregame["red"]["picks"]),
        "--league",
        pregame["competition"]["league"],
        "--blue-name",
        blue_name,
        "--red-name",
        red_name,
    ]
    role_flags = {"top": "top", "jungle": "jng", "mid": "mid", "bot": "bot", "support": "sup"}
    if player_context_available:
        for side_name in ("blue", "red"):
            for player in pregame[side_name]["players"]:
                command.extend(
                    [
                        f"--{side_name}-{role_flags[player['role']]}-player",
                        player["player"],
                    ]
                )
    completed = subprocess.run(
        command,
        cwd=app,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip() or "unknown scorer failure"
        raise EngineUnavailableError(message)
    try:
        score_output = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise EngineUnavailableError("local scorer returned non-JSON output") from exc
    score_output["teams"] = {
        "blue": pregame["blue"]["team"],
        "red": pregame["red"]["team"],
    }
    score_output["player_context_policy"] = {
        "status": "applied" if player_context_available else "unavailable",
        "reason": None
        if player_context_available
        else "one_or_more_historical_player_names_are_absent_from_the_fixed_runtime_context",
        "missing_players": sorted(
            {
                player
                for player in requested_players
                if player.casefold() not in known_players
            }
        ),
    }
    return attach_score(
        run,
        score_output,
        runtime_as_of=runtime_as_of,
        runtime_sha256=_sha256_file(runtime_path),
        runner_sha256=_sha256_file(runner),
        score_module_sha256=_sha256_file(score_module),
        scored_at=scored_at,
    )


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ManualLeaguepediaError(f"{path} must contain a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="validate and seal a pregame input")
    freeze.add_argument("--input", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)

    score = sub.add_parser("score", help="run the local scorer against a frozen pregame")
    score.add_argument("--input", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    _repo_root = Path(__file__).resolve().parents[2]
    score.add_argument("--repo", type=Path, default=_repo_root)

    reveal = sub.add_parser("reveal", help="append an outcome after score sealing")
    reveal.add_argument("--input", type=Path, required=True)
    reveal.add_argument("--outcome", type=Path, required=True)
    reveal.add_argument("--output", type=Path, required=True)

    verify = sub.add_parser("verify", help="verify a run ledger")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--require-score", action="store_true")
    verify.add_argument("--require-outcome", action="store_true")

    capture = sub.add_parser("capture-page", help="capture raw Leaguepedia revision/rendered payloads")
    capture.add_argument("--title", required=True)
    capture.add_argument("--output-dir", type=Path, required=True)
    capture.add_argument("--observed-at")
    capture.add_argument("--before", help="latest page revision at or before this RFC-3339 cutoff")

    args = parser.parse_args()
    try:
        if args.command == "freeze":
            _write_json(args.output, freeze_pregame(_read_json(args.input)))
        elif args.command == "score":
            _write_json(args.output, score_frozen(_read_json(args.input), repo=args.repo.resolve()))
        elif args.command == "reveal":
            _write_json(args.output, reveal_outcome(_read_json(args.input), _read_json(args.outcome)))
        elif args.command == "verify":
            verify_run(
                _read_json(args.input),
                require_score=args.require_score,
                require_outcome=args.require_outcome,
            )
            print("OK")
        elif args.command == "capture-page":
            entries = capture_leaguepedia_page(
                args.title,
                args.output_dir,
                observed_at=args.observed_at,
                before=args.before,
            )
            print(json.dumps(entries, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except ManualLeaguepediaError as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())


__all__ = [
    "ACTIVE_STARTER_STATUSES",
    "EngineUnavailableError",
    "FORBIDDEN_PREGAME_KEYS",
    "ManualLeaguepediaError",
    "MODES",
    "PregameLeakageError",
    "REVEALED_PHASE",
    "RosterResolutionError",
    "SCHEMA_VERSION",
    "SCORED_PHASE",
    "assert_no_outcome_fields",
    "attach_score",
    "capture_leaguepedia_page",
    "freeze_pregame",
    "leaguepedia_api_url",
    "reveal_outcome",
    "resolve_time_sliced_lineup",
    "score_frozen",
    "verify_run",
]
