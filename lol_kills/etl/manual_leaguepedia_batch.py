"""Capture and assemble a retrospective, result-blind Leaguepedia run.

This is deliberately a small batch companion to ``manual_leaguepedia``.  It
does not treat a current historical page as a strict historical forecast: the
source payloads are timestamped at retrieval, and the generated ledgers use
``retrospective`` mode.  Draft/player rows and outcome rows are fetched into
separate raw directories so the outcome payload cannot enter the frozen input
by accident.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.competition import classify_competition
from lol_kills.etl.manual_leaguepedia import (
    freeze_pregame,
    reveal_outcome,
    score_frozen,
    verify_run,
)
from lol_kills.net import require_https_url


ROOT = Path(__file__).resolve().parents[2]
UA = "Scryglass-manual-Leaguepedia-batch/1.0"
SCHEMA_VERSION = "scryglass:leaguepedia-manual-batch:v1"
ROLE_MAP = {
    "top": "top",
    "jungle": "jungle",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "support": "support",
}
ROLES = ("top", "jungle", "mid", "bot", "support")


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _rfc(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe(value: Any) -> str:
    return str(value or "").strip()


def _utc_event(value: Any) -> datetime:
    raw = _safe(value)
    parsed = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S")
    return parsed.replace(tzinfo=timezone.utc)


def _json_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"{path} is not a JSON array")
    return [dict(row) for row in value if isinstance(row, Mapping)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _cargo_url(tables: str, fields: str, where: str, *, limit: int = 500) -> str:
    params = {
        "tables": tables,
        "fields": fields,
        "where": where,
        "order_by": "ScoreboardPlayers.DateTime_UTC ASC" if tables == "ScoreboardPlayers" else "ScoreboardGames.DateTime_UTC ASC",
        "limit": str(limit),
        "format": "json",
    }
    return "https://lol.fandom.com/wiki/Special:CargoExport?" + urllib.parse.urlencode(params)


def _fetch(url: str) -> bytes:
    url = require_https_url(url, hosts={"lol.fandom.com"})
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def _game_ids(catalog_dir: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    page_for_game: dict[str, str] = {}
    for page in sorted(catalog_dir.glob("games-page-*.json")):
        page_rows = _json_rows(page)
        for row in page_rows:
            game_id = _safe(row.get("GameId"))
            if not game_id or game_id in page_for_game:
                continue
            rows.append(row)
            page_for_game[game_id] = page.name
    rows.sort(key=lambda row: (_safe(row.get("DateTime UTC")), _safe(row.get("GameId"))))
    return rows, page_for_game


def _batches(values: list[str], size: int = 35) -> Iterable[list[str]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def _where_for_ids(field: str, ids: list[str]) -> str:
    clauses = [f'{field}="{game_id.replace(chr(34), chr(92) + chr(34))}"' for game_id in ids]
    return "(" + " OR ".join(clauses) + ")"


def capture_pages(run_dir: Path, *, sleep_s: float = 0.35) -> dict[str, Any]:
    """Fetch draft rows and results into separate immutable raw payloads."""

    catalog_dir = run_dir / "raw" / "catalog"
    games, page_for_game = _game_ids(catalog_dir)
    if not games:
        raise ValueError(f"no catalog rows found in {catalog_dir}")
    observed_at = _rfc(_now())
    game_ids = [_safe(row["GameId"]) for row in games]
    draft_dir = run_dir / "raw" / "drafts"
    outcome_dir = run_dir / "raw" / "outcomes"
    draft_dir.mkdir(parents=True, exist_ok=True)
    outcome_dir.mkdir(parents=True, exist_ok=True)

    draft_fields = ",".join(
        [
            "ScoreboardPlayers.GameId",
            "ScoreboardPlayers.Team",
            "ScoreboardPlayers.Name",
            "ScoreboardPlayers.Champion",
            "ScoreboardPlayers.IngameRole",
            "ScoreboardPlayers.Side",
            "ScoreboardPlayers.DateTime_UTC",
        ]
    )
    outcome_fields = ",".join(
        [
            "ScoreboardGames.GameId",
            "ScoreboardGames.Team1",
            "ScoreboardGames.Team2",
            "ScoreboardGames.WinTeam",
            "ScoreboardGames.Team1Kills",
            "ScoreboardGames.Team2Kills",
            "ScoreboardGames.Gamelength_Number",
            "ScoreboardGames.DateTime_UTC",
        ]
    )
    draft_pages: list[dict[str, Any]] = []
    outcome_pages: list[dict[str, Any]] = []
    draft_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []

    for page_index, batch in enumerate(_batches(game_ids)):
        url = _cargo_url(
            "ScoreboardPlayers",
            draft_fields,
            _where_for_ids("ScoreboardPlayers.GameId", batch),
        )
        raw = _fetch(url)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"draft page {page_index} did not return an array")
        filename = f"players-page-{page_index:03d}.json"
        (draft_dir / filename).write_bytes(raw)
        draft_pages.append(
            {
                "page": filename,
                "game_ids": batch,
                "row_count": len(parsed),
                "sha256": _sha(raw),
                "available_at": observed_at,
                "api_url": url,
            }
        )
        draft_rows.extend(row for row in parsed if isinstance(row, Mapping))
        time.sleep(sleep_s)

    for page_index, batch in enumerate(_batches(game_ids)):
        url = _cargo_url(
            "ScoreboardGames",
            outcome_fields,
            _where_for_ids("ScoreboardGames.GameId", batch),
        )
        raw = _fetch(url)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError(f"outcome page {page_index} did not return an array")
        filename = f"games-page-{page_index:03d}.json"
        (outcome_dir / filename).write_bytes(raw)
        outcome_pages.append(
            {
                "page": filename,
                "game_ids": batch,
                "row_count": len(parsed),
                "sha256": _sha(raw),
                "available_at": observed_at,
                "api_url": url,
            }
        )
        outcome_rows.extend(row for row in parsed if isinstance(row, Mapping))
        time.sleep(sleep_s)

    # Keep normalized copies separate from the raw source payloads.  The
    # normalized draft file has no winner/kills/result columns by construction.
    _write_jsonl(run_dir / "normalized-draft-rows.jsonl", draft_rows)
    _write_jsonl(run_dir / "normalized-outcome-rows.jsonl", outcome_rows)
    catalog_pages = []
    for path in sorted(catalog_dir.glob("games-page-*.json")):
        catalog_pages.append(
            {
                "page": path.name,
                "sha256": _sha(path.read_bytes()),
                "available_at": observed_at,
                "row_count": len(_json_rows(path)),
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": "leaguepedia-manual-run-2026-07-31",
        "since": "2026-07-01T00:00:00Z",
        "until": None,
        "scope": "all teams and all catalogued maps returned by Leaguepedia",
        "observed_at": observed_at,
        "catalog": {
            "fields": ["GameId", "Team1", "Team2", "Tournament", "DateTime_UTC"],
            "pages": catalog_pages,
            "game_count": len(games),
            "unique_game_ids": len({row["GameId"] for row in games}),
        },
        "draft_capture": {
            "fields": draft_fields.split(","),
            "pages": draft_pages,
            "raw_row_count": len(draft_rows),
            "game_ids_requested": len(game_ids),
            "games_with_rows": len({_safe(row.get("GameId")) for row in draft_rows if _safe(row.get("GameId"))}),
        },
        "outcome_capture": {
            "fields": outcome_fields.split(","),
            "pages": outcome_pages,
            "raw_row_count": len(outcome_rows),
            "game_ids_requested": len(game_ids),
            "games_with_rows": len({_safe(row.get("GameId")) for row in outcome_rows if _safe(row.get("GameId"))}),
        },
        "separation": {
            "draft_rows_contain_outcome_fields": any(
                any(key in row for key in ("WinTeam", "Winner", "Team1Kills", "Team2Kills", "PlayerWin"))
                for row in draft_rows
            ),
            "outcomes_are_not_read_by_freeze": True,
        },
        "source_page_for_game": page_for_game,
    }
    _write_json(run_dir / "capture-manifest.json", manifest)
    return manifest


def _competition(tournament: str) -> dict[str, Any]:
    text = re.sub(r"\s+", " ", tournament.upper()).strip()
    direct = (
        ("ESPORTS WORLD CUP", "EWC"),
        ("MSI", "MSI"),
        ("ASIA MASTERS", "ASIA MASTER"),
        ("LCK CL", "LCKC"),
        ("NACL", "NACL"),
        ("CBLOL", "CBLOL"),
        ("LCK", "LCK"),
        ("LPL", "LPL"),
        ("LEC", "LEC"),
        ("LCS", "LCS"),
        ("LCP", "LCP"),
        ("PRM", "PRM"),
        ("LRS", "LRS"),
        ("LRN", "LRN"),
        ("HLL", "HLL"),
        ("HCC", "HCC"),
        ("UEM", "EM"),
        ("NLC", "NLC"),
        ("LFL", "LFL"),
        ("LJL", "LJL"),
        ("LIT", "LIT"),
        ("LPLOL", "LPLOL"),
        ("EBL", "EBL"),
        ("TCL", "TCL"),
        ("CD ", "CD"),
        ("ROAD OF LEGENDS", "ROL"),
        ("NEXUS LEAGUE", "NEXUS"),
        ("ARABIAN LEAGUE", "AL"),
        ("KE SPA CUP", "KESPA"),
        ("KESPA CUP", "KESPA"),
        ("EQUAL ESPORTS CUP", "ECC"),
        ("LES ", "LES"),
        ("IDL ", "IDL"),
    )
    source = next((value for token, value in direct if text.startswith(token) or token in text), text.split(" ", 1)[0] or "UNKNOWN")
    label = classify_competition(source, tournament)
    return {
        "league": label.league,
        "league_source": source,
        "scope": label.scope,
        "event_kind": label.event_kind,
        "tier": label.tier,
        "is_international": label.is_international,
        "source_tournament": tournament,
    }


def _side(value: Any) -> str | None:
    text = _safe(value).lower()
    if text in {"1", "blue"}:
        return "blue"
    if text in {"2", "red"}:
        return "red"
    return None


def _role(value: Any) -> str | None:
    return ROLE_MAP.get(_safe(value).lower())


def build_frozen_inputs(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "capture-manifest.json").read_text(encoding="utf-8"))
    games, page_for_game = _game_ids(run_dir / "raw" / "catalog")
    draft_rows = [json.loads(line) for line in (run_dir / "normalized-draft-rows.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in draft_rows:
        by_game[_safe(row.get("GameId"))].append(row)
    draft_page_by_game = {
        game_id: page["page"]
        for page in manifest["draft_capture"]["pages"]
        for game_id in page["game_ids"]
    }
    catalog_hash = {page["page"]: page["sha256"] for page in manifest["catalog"]["pages"]}
    draft_hash = {page["page"]: page["sha256"] for page in manifest["draft_capture"]["pages"]}
    observed_at = manifest["observed_at"]
    frozen: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    for game in games:
        game_id = _safe(game.get("GameId"))
        event_at = _utc_event(game.get("DateTime UTC"))
        rows = by_game.get(game_id, [])
        sides: dict[str, dict[str, dict[str, Any]]] = {"blue": {}, "red": {}}
        errors: list[str] = []
        for row in rows:
            side = _side(row.get("Side"))
            role = _role(row.get("IngameRole"))
            player = _safe(row.get("Name"))
            champion = normalize_champ(_safe(row.get("Champion")))
            if side is None or role is None or not player or not champion:
                errors.append("invalid player row")
                continue
            if role in sides[side]:
                errors.append(f"duplicate {side} role={role}")
                continue
            sides[side][role] = {
                "role": role,
                "player": player,
                "team": normalize_team(_safe(row.get("Team"))),
                "champion": champion,
            }
        for side in ("blue", "red"):
            missing = sorted(set(ROLES) - set(sides[side]))
            if missing:
                errors.append(f"{side} missing roles={','.join(missing)}")
        if errors:
            blocked.append({"fixture_id": game_id, "event_start": _rfc(event_at), "errors": sorted(set(errors))})
            continue
        blue_team = normalize_team(_safe(game.get("Team1")))
        red_team = normalize_team(_safe(game.get("Team2")))
        # ScoreboardPlayers.Side is the authority for color.  Team1/Team2 is
        # retained as the display fallback when a source row is incomplete.
        blue_row_teams = {row["team"] for row in sides["blue"].values() if row["team"]}
        red_row_teams = {row["team"] for row in sides["red"].values() if row["team"]}
        if len(blue_row_teams) == 1:
            blue_team = next(iter(blue_row_teams))
        if len(red_row_teams) == 1:
            red_team = next(iter(red_row_teams))
        competition = _competition(_safe(game.get("Tournament")))
        pregame_at = event_at - timedelta(seconds=1)
        source_catalog_page = page_for_game[game_id]
        source_draft_page = draft_page_by_game[game_id]
        source_snapshots = [
            {
                "snapshot_id": f"catalog:{source_catalog_page}",
                "raw_file": f"raw/catalog/{source_catalog_page}",
                "sha256": catalog_hash[source_catalog_page],
                "available_at": observed_at,
                "kind": "fixture_catalog_without_result",
            },
            {
                "snapshot_id": f"draft:{source_draft_page}",
                "raw_file": f"raw/drafts/{source_draft_page}",
                "sha256": draft_hash[source_draft_page],
                "available_at": observed_at,
                "kind": "draft_and_player_rows_without_result",
            },
        ]
        sides_input: dict[str, Any] = {}
        roster_events: list[dict[str, Any]] = []
        for side_name, team in (("blue", blue_team), ("red", red_team)):
            ordered = [sides[side_name][role] for role in ROLES]
            sides_input[side_name] = {
                "team": team,
                "picks": [row["champion"] for row in ordered],
                "players": [{"role": row["role"], "player": row["player"]} for row in ordered],
            }
            for row in ordered:
                roster_events.append(
                    {
                        "team": team,
                        "role": row["role"],
                        "player": row["player"],
                        "status": "confirmed_starter",
                        "effective_from": _rfc(event_at - timedelta(seconds=1)),
                        "available_at": observed_at,
                        "source_snapshot_id": f"draft:{source_draft_page}",
                        "source_sha256": draft_hash[source_draft_page],
                    }
                )
        payload = {
            "fixture_id": game_id,
            "mode": "retrospective",
            "event_start": _rfc(event_at),
            "draft_locked_at": _rfc(pregame_at),
            "as_of": _rfc(pregame_at),
            "competition": competition,
            "blue": sides_input["blue"],
            "red": sides_input["red"],
            "source_snapshots": source_snapshots,
            "roster_events": roster_events,
        }
        frozen.append(freeze_pregame(payload))
    frozen.sort(key=lambda run: (run["pregame"]["event_start"], run["pregame"]["fixture_id"]))
    _write_jsonl(run_dir / "frozen-ledger.jsonl", frozen)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "mode": "retrospective",
        "catalog_games": len(games),
        "frozen_games": len(frozen),
        "blocked_games": len(blocked),
        "blocked": blocked,
        "all_teams": sorted(
            {
                run["pregame"][side]["team"]
                for run in frozen
                for side in ("blue", "red")
            }
        ),
        "competitions": sorted(
            {
                run["pregame"]["competition"]["source_tournament"]
                for run in frozen
            }
        ),
        "notes": [
            "Frozen ledgers contain draft/player inputs but no winner, kills, gold, duration, or result field.",
            "Because Leaguepedia was retrieved after these historical maps, this is retrospective result-blind scoring, not strict pre-event forecasting.",
            "The current fixed draft runtime is recorded later by the scoring ledger and is not claimed to be historically available for each map.",
        ],
    }
    _write_json(run_dir / "freeze-summary.json", summary)
    return summary


def score_frozen_batch(run_dir: Path, *, repo: Path, workers: int = 6, scored_at: str | None = None) -> dict[str, Any]:
    """Score only the frozen JSONL; no outcome file is opened here."""

    frozen_path = run_dir / "frozen-ledger.jsonl"
    frozen = [
        json.loads(line)
        for line in frozen_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sealed_at = scored_at or _rfc(_now())

    def one(run: Mapping[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        fixture_id = run["pregame"]["fixture_id"]
        try:
            result = score_frozen(run, repo=repo.resolve(), scored_at=sealed_at)
            verify_run(result, require_score=True, require_outcome=False)
            return result, None
        except Exception as exc:  # keep one unavailable map from hiding the rest
            return None, {"fixture_id": fixture_id, "error_type": type(exc).__name__, "error": str(exc)}

    scored: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = [executor.submit(one, run) for run in frozen]
        for future in concurrent.futures.as_completed(futures):
            result, error = future.result()
            if result is not None:
                scored.append(result)
            if error is not None:
                errors.append(error)
    scored.sort(key=lambda run: (run["pregame"]["event_start"], run["pregame"]["fixture_id"]))
    errors.sort(key=lambda row: row["fixture_id"])
    _write_jsonl(run_dir / "scored-ledger.jsonl", scored)
    _write_jsonl(run_dir / "score-errors.jsonl", errors)
    player_context = {
        "applied": sum(
            1
            for run in scored
            if run["score"]["output"].get("player_context_policy", {}).get("status") == "applied"
        ),
        "unavailable": sum(
            1
            for run in scored
            if run["score"]["output"].get("player_context_policy", {}).get("status") == "unavailable"
        ),
    }
    by_league: dict[str, dict[str, int]] = defaultdict(lambda: {"scored": 0, "blocked": 0})
    for run in scored:
        by_league[run["pregame"]["competition"]["league"]]["scored"] += 1
    for error in errors:
        match = next((run for run in frozen if run["pregame"]["fixture_id"] == error["fixture_id"]), None)
        if match is not None:
            by_league[match["pregame"]["competition"]["league"]]["blocked"] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "retrospective",
        "score_sealed_at": sealed_at,
        "frozen_games": len(frozen),
        "scored_games": len(scored),
        "blocked_games": len(errors),
        "player_context": player_context,
        "by_league": dict(sorted(by_league.items())),
        "outcomes_opened": False,
        "runtime_policy": "fixed local runtime; score output is deterministic, but runtime_as_of is not asserted to precede historical event_start in retrospective mode",
    }
    _write_json(run_dir / "score-summary.json", summary)
    return summary


def reveal_outcomes_batch(run_dir: Path) -> dict[str, Any]:
    """Attach outcomes only after the scored ledger exists."""

    scored = [
        json.loads(line)
        for line in (run_dir / "scored-ledger.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    outcome_rows = [
        json.loads(line)
        for line in (run_dir / "normalized-outcome-rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    by_game = {_safe(row.get("GameId")): row for row in outcome_rows}
    manifest = json.loads((run_dir / "capture-manifest.json").read_text(encoding="utf-8"))
    outcome_page_by_game = {
        game_id: page["page"]
        for page in manifest["outcome_capture"]["pages"]
        for game_id in page["game_ids"]
    }
    outcome_hash = {page["page"]: page["sha256"] for page in manifest["outcome_capture"]["pages"]}
    revealed_at = _rfc(_now())
    revealed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for run in scored:
        fixture_id = run["pregame"]["fixture_id"]
        row = by_game.get(fixture_id)
        if row is None:
            errors.append({"fixture_id": fixture_id, "error": "outcome row missing"})
            continue
        winner = normalize_team(_safe(row.get("WinTeam")))
        source_teams = {
            normalize_team(_safe(row.get("Team1"))),
            normalize_team(_safe(row.get("Team2"))),
        }
        frozen_teams = {run["pregame"]["blue"]["team"], run["pregame"]["red"]["team"]}
        if not winner or winner not in frozen_teams:
            errors.append(
                {
                    "fixture_id": fixture_id,
                    "error": "outcome winner does not match frozen teams",
                    "winner": winner,
                    "frozen_teams": sorted(frozen_teams),
                }
            )
            continue
        if source_teams != frozen_teams:
            errors.append(
                {
                    "fixture_id": fixture_id,
                    "error": "outcome source teams do not match frozen teams",
                    "outcome_teams": sorted(source_teams),
                    "frozen_teams": sorted(frozen_teams),
                }
            )
            continue
        page = outcome_page_by_game[fixture_id]
        outcome = {
            "winner": winner,
            "winner_side": "blue" if winner == run["pregame"]["blue"]["team"] else "red",
            "source_snapshot_id": f"outcome:{page}",
            "source_sha256": outcome_hash[page],
            "source_team1": _safe(row.get("Team1")),
            "source_team2": _safe(row.get("Team2")),
            "team1_kills": row.get("Team1Kills"),
            "team2_kills": row.get("Team2Kills"),
            "duration_min": row.get("Gamelength Number"),
            "revealed_at": revealed_at,
        }
        try:
            result = reveal_outcome(run, outcome, revealed_at=revealed_at)
            verify_run(result, require_score=True, require_outcome=True)
        except Exception as exc:
            errors.append({"fixture_id": fixture_id, "error_type": type(exc).__name__, "error": str(exc)})
            continue
        draft_score = result["score"]["output"].get("draft_score", {}).get("actual_blue", {})
        blue_pct = draft_score.get("blue_pct")
        red_pct = draft_score.get("red_pct")
        if isinstance(blue_pct, (int, float)) and isinstance(red_pct, (int, float)):
            predicted_side = "blue" if blue_pct > red_pct else "red" if red_pct > blue_pct else "tie"
            actual_side = result["outcome"]["winner_side"]
            result["evaluation"] = {
                "draft_score_predicted_side": predicted_side,
                "draft_score_correct": predicted_side == actual_side,
                "blue_pct": blue_pct,
                "red_pct": red_pct,
            }
        revealed.append(result)
    revealed.sort(key=lambda run: (run["pregame"]["event_start"], run["pregame"]["fixture_id"]))
    errors.sort(key=lambda row: row["fixture_id"])
    _write_jsonl(run_dir / "revealed-ledger.jsonl", revealed)
    _write_jsonl(run_dir / "reveal-errors.jsonl", errors)
    correct = sum(1 for run in revealed if run.get("evaluation", {}).get("draft_score_correct") is True)
    evaluated = sum(1 for run in revealed if "draft_score_correct" in run.get("evaluation", {}))
    by_league: dict[str, dict[str, int]] = defaultdict(lambda: {"maps": 0, "correct": 0})
    for run in revealed:
        league = run["pregame"]["competition"]["league"]
        by_league[league]["maps"] += 1
        if run.get("evaluation", {}).get("draft_score_correct") is True:
            by_league[league]["correct"] += 1
    summary = {
        "schema_version": SCHEMA_VERSION,
        "mode": "retrospective",
        "revealed_at": revealed_at,
        "scored_games": len(scored),
        "revealed_games": len(revealed),
        "blocked_games": len(errors),
        "draft_score": {
            "evaluated_games": evaluated,
            "correct_games": correct,
            "accuracy_pct": round(100 * correct / evaluated, 2) if evaluated else None,
        },
        "by_league": dict(sorted(by_league.items())),
        "source_policy": "outcomes were opened only after the scored ledger was sealed",
    }
    _write_json(run_dir / "reveal-summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture", help="capture drafts and outcomes into separate raw directories")
    capture.add_argument("--run-dir", type=Path, required=True)
    capture.add_argument("--sleep", type=float, default=0.35)
    freeze = sub.add_parser("freeze", help="assemble and seal result-free retrospective ledgers")
    freeze.add_argument("--run-dir", type=Path, required=True)
    score = sub.add_parser("score", help="score frozen ledgers without opening outcome payloads")
    score.add_argument("--run-dir", type=Path, required=True)
    score.add_argument("--repo", type=Path, default=ROOT)
    score.add_argument("--workers", type=int, default=6)
    score.add_argument("--scored-at")
    reveal = sub.add_parser("reveal", help="attach outcomes after score sealing")
    reveal.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "capture":
        _write_json(args.run_dir / "capture-manifest.json", capture_pages(args.run_dir, sleep_s=args.sleep))
        return 0
    if args.command == "freeze":
        print(json.dumps(build_frozen_inputs(args.run_dir), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.command == "score":
        print(
            json.dumps(
                score_frozen_batch(
                    args.run_dir,
                    repo=args.repo,
                    workers=args.workers,
                    scored_at=args.scored_at,
                ),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if args.command == "reveal":
        print(json.dumps(reveal_outcomes_batch(args.run_dir), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
