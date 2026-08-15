#!/usr/bin/env python3
"""Deterministic gold/XP event ledger for one verified GRID Riot file.

This is deliberately a small, closed pipeline: the raw file hash, roster, and
team mapping are fixed inputs; only resource-bearing Riot events are retained;
each row is anchored to the first LiveStats frame after the event and carries
the observed gold/XP change since the previous event frame.  The JSON output is
the complete ledger; the Markdown output is a compact human-readable view.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

from lol_kills.research.grid_sequence_review import (
    GameData,
    _frame_after,
    _number,
    format_clock,
    load_game,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
RAW = (
    _REPO_ROOT
    / "data/lol/warehouse/private_grid/sequence_review/v1/raw/"
    / "events_2964596_1_riot_83c79b37a73a3af7c8efe89c94b1b9889349f19f224cec167bb02ac01a7b95f9.jsonl"
)
OUT = RAW.parents[1] / "reports"
TEAMS = {100: "Team Liquid", 200: "Sentinels"}

LEDGER_SCHEMA = "scryglass:grid-event-ledger:v1"
RESOURCE_SCHEMAS = {
    "epic_monster_kill",
    "turret_plate_destroyed",
    "turret_plate_gold_earned",
    "building_gold_grant",
    "building_destroyed",
    "champion_kill",
    "champion_kill_special",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def raw_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("rfc461Schema") in RESOURCE_SCHEMAS:
                rows.append(row)
    return rows


def names(game: GameData) -> dict[int, str]:
    return {
        int(row["participant_id"]): f"{row['champion']} ({row['player']})"
        for row in game.roster
    }


def event_title(row: Mapping[str, Any], roster: Mapping[int, Mapping[str, Any]]) -> str:
    schema = row["rfc461Schema"]
    if schema == "epic_monster_kill":
        monster = str(row.get("monsterType") or "monster")
        if monster.lower() == "dragon":
            monster = f"{str(row.get('dragonType') or '').title()} dragon".strip()
        return f"{monster} killed"
    if schema == "turret_plate_destroyed":
        return f"{str(row.get('lane') or '?').title()} turret plate"
    if schema == "building_destroyed":
        lane = str(row.get("lane") or "").title()
        building = str(row.get("buildingType") or "building")
        tier = str(row.get("turretTier") or "").title()
        return " ".join(part for part in (lane, tier, building, "destroyed") if part)
    if schema == "champion_kill":
        victim = roster.get(int(row.get("victim") or 0), {}).get("champion", "?")
        killer = roster.get(int(row.get("killer") or 0), {}).get("champion", "?")
        return f"{victim} killed by {killer}"
    return "special kill marker"


def frame_totals(game: GameData, frame: Any) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = defaultdict(lambda: {"gold": 0.0, "xp": 0.0, "cs": 0.0, "neutral_cs": 0.0})
    for pid, state in frame.players.items():
        team = next(int(r["team_id"]) for r in game.roster if int(r["participant_id"]) == pid)
        out[team]["gold"] += _number(state.get("total_gold"))
        out[team]["xp"] += _number(state.get("xp"))
        out[team]["cs"] += _number(state.get("stats", {}).get("MINIONS_KILLED"))
        out[team]["neutral_cs"] += _number(state.get("stats", {}).get("NEUTRAL_MINIONS_KILLED"))
    return {k: dict(v) for k, v in out.items()}


def frame_players(game: GameData, frame: Any) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for pid, state in frame.players.items():
        out[pid] = {
            "gold": _number(state.get("total_gold")),
            "xp": _number(state.get("xp")),
            "cs": _number(state.get("stats", {}).get("MINIONS_KILLED")),
            "neutral_cs": _number(state.get("stats", {}).get("NEUTRAL_MINIONS_KILLED")),
        }
    return out


def diff(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(right.get(key, 0.0)) - float(left.get(key, 0.0)), 3) for key in ("gold", "xp", "cs", "neutral_cs")}


def build(path: Path) -> dict[str, Any]:
    game = load_game(path)
    rows = raw_rows(path)
    roster = {int(row["participant_id"]): row for row in game.roster}
    # The gold-grant and plate rows are paired with their parent event by exact
    # game clock.  Keep them as annotations, not separate resource rows.
    by_time: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_time[int(row.get("gameTime") or 0)].append(row)
    event_times = sorted(by_time)
    # Several GRID events can occur between the same one-second LiveStats
    # frames (for example a camp kill immediately followed by a plate).  Map
    # first, then group by the actual post-event stats frame so the resource
    # change is counted once rather than once per timestamp.
    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for time_ms in event_times:
        frame = _frame_after(game.frames, time_ms)
        by_frame[frame.time_ms].extend(by_time[time_ms])
    frame_groups = sorted(by_frame.items())
    baseline = game.frames[0]
    previous_team = frame_totals(game, baseline)
    previous_player = frame_players(game, baseline)
    ledger: list[dict[str, Any]] = []
    for frame_ms, event_rows in frame_groups:
        # Grant-only rows are intentionally folded into the parent timestamp.
        primary = [r for r in event_rows if r["rfc461Schema"] not in {"turret_plate_gold_earned", "building_gold_grant", "champion_kill_special"}]
        if not primary:
            continue
        frame = next(frame for frame in game.frames if frame.time_ms == frame_ms)
        team_now = frame_totals(game, frame)
        player_now = frame_players(game, frame)
        team_delta = {
            str(team): diff(previous_team.get(team, {}), team_now.get(team, {}))
            for team in sorted(TEAMS)
        }
        player_delta = {
            str(pid): diff(previous_player.get(pid, {}), player_now.get(pid, {}))
            for pid in sorted(player_now)
            if any(abs(value) > 1e-6 for value in diff(previous_player.get(pid, {}), player_now.get(pid, {})).values())
        }
        direct = {
            "killer_gold": round(sum(_number(r.get("killerGold")) for r in event_rows), 3),
            "local_gold": round(sum(_number(r.get("localGold")) for r in event_rows), 3),
            "global_gold": round(sum(_number(r.get("globalGold")) for r in event_rows), 3),
            "plate_gold": round(sum(_number(r.get("bounty")) for r in event_rows if r["rfc461Schema"] == "turret_plate_gold_earned"), 3),
        }
        event_times_in_group = sorted(int(r.get("gameTime") or 0) for r in event_rows)
        time_ms = event_times_in_group[0]
        ledger.append({
            "time_ms": time_ms,
            "time": format_clock(time_ms),
            "event_times_ms": event_times_in_group,
            "events": [event_title(r, roster) for r in primary],
            "schemas": sorted({r["rfc461Schema"] for r in primary}),
            "direct_event_rewards": direct,
            "frame_ms": frame.time_ms,
            "frame_skew_ms": frame.time_ms - time_ms,
            "team_change_since_previous_event": team_delta,
            "player_change_since_previous_event": player_delta,
            "team_totals_at_frame": {
                str(team): {k: round(v, 3) for k, v in team_now.get(team, {}).items()}
                for team in sorted(TEAMS)
            },
        })
        previous_team, previous_player = team_now, player_now
    return {
        "schema": LEDGER_SCHEMA,
        "source": {
            "raw_path": str(path),
            "raw_sha256": sha256_file(path),
            "riot_game_id": game.identity.get("riot_game_id"),
            "provider_series_id": "2964596",
            "provider_game_id": "e8af9de9-0dda-4398-8054-571e4fc50872",
            "game_index": 1,
        },
        "teams": TEAMS,
        "roster": list(game.roster),
        "completeness": game.completeness,
        "event_counts": {schema: sum(1 for r in rows if r["rfc461Schema"] == schema) for schema in sorted(RESOURCE_SCHEMAS)},
        "baseline": {
            "frame_ms": baseline.time_ms,
            "time": format_clock(baseline.time_ms),
            "team_totals": {str(team): {k: round(v, 3) for k, v in values.items()} for team, values in frame_totals(game, baseline).items()},
        },
        "final_checkpoint": {
            "frame_ms": ledger[-1]["frame_ms"] if ledger else None,
            "time": ledger[-1]["time"] if ledger else None,
            "team_totals": ledger[-1]["team_totals_at_frame"] if ledger else {},
        },
        "ledger": ledger,
        "method": {
            "resource_change": "sum of LiveStats totalGold, XP, minion CS, and neutral CS changes from the previous event's first post-event stats frame",
            "event_grouping": "same gameTime is one row; plate/building grant rows are annotations",
            "scope": "all resource-bearing Riot events, including ordinary jungle camps and champion kills",
        },
    }


def markdown(report: Mapping[str, Any]) -> str:
    rows = report["ledger"]
    lines = [
        "# Team Liquid vs Sentinels — game 1 resource ledger",
        "",
        "GRID/Riot LiveStats, 2026-08-02; game 1. Gold and XP are observed changes between consecutive event checkpoints. Full event data are in the JSON artifact.",
        "",
        "| Time | Event | TL gold / XP | SEN gold / XP | Notes |",
        "|---:|---|---:|---:|---|",
    ]
    # Public-sized view: major objectives, plates, kills, and structures. The
    # ordinary camp clears remain in JSON so no evidence is discarded.
    keep = {"turret_plate_destroyed", "building_destroyed", "champion_kill"}
    for row in rows:
        event_text = "; ".join(row["events"])
        major_objective = any(
            marker in event_text.casefold()
            for marker in ("dragon killed", "voidgrub killed", "herald killed", "baron killed")
        )
        if not (set(row["schemas"]) & keep or major_objective):
            continue
        event = event_text
        tl = row["team_change_since_previous_event"]["100"]
        sen = row["team_change_since_previous_event"]["200"]
        direct = row["direct_event_rewards"]
        note = ""
        if direct["plate_gold"]:
            note = f"plate grants {direct['plate_gold']:.0f}g"
        elif direct["killer_gold"] or direct["global_gold"]:
            note = f"event fields: killer {direct['killer_gold']:.0f}g, global {direct['global_gold']:.0f}g"
        lines.append(f"| {row['time']} | {event} | {tl['gold']:+.0f}g / {tl['xp']:+.0f} XP | {sen['gold']:+.0f}g / {sen['xp']:+.0f} XP | {note} |")
    lines += [
        "",
        f"Source SHA-256: `{report['source']['raw_sha256']}`",
        f"Completeness: `{report['completeness']['status']}`; {report['completeness']['stats_frames']} stats frames; maximum gap {report['completeness']['maximum_stats_gap_ms']} ms.",
        "",
        "Interpretation: event-row gold includes ordinary ambient gold accrued since the prior checkpoint; plate gold is shown in Notes and is already included in totalGold, so it must not be added a second time.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=RAW)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    report = build(args.source)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "series_2964596_game_1_gold_xp_ledger.json"
    md_path = args.output_dir / "series_2964596_game_1_gold_xp_ledger.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "rows": len(report["ledger"]), "raw_sha256": report["source"]["raw_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
