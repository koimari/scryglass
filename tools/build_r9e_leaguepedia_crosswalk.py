"""Build a small source-backed OE 16.16 to Leaguepedia series crosswalk.

The source file exposes Oracle's Elixir game IDs.  Leaguepedia exposes a
different game ID namespace plus MatchSchedule.MatchId.  This script joins
the two namespaces only when the league, unordered team set, patch, and
timestamp identify one ScoreboardGames row.  It then requires the exact
ScoreboardGames game-ID prefix to resolve to one MatchSchedule row.

The join rejects missing or ambiguous evidence.  It does not create series
IDs from outcomes, row order, or inferred best-of lengths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


SCHEMA_VERSION = "scryglass:r9e-oe-leaguepedia-crosswalk:v1"
MAX_GAME_TIME_DELTA_SECONDS = 120
MAX_FIRST_GAME_SCHEDULE_DELTA_SECONDS = 1800


class CrosswalkError(ValueError):
    """Raised when the source evidence cannot produce a unique crosswalk."""


def _load(path: Path) -> list[dict[str, Any]] | dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, (list, dict)):
        raise CrosswalkError(f"expected an object or array: {path}")
    return value


def _time(value: Any) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise CrosswalkError("empty source timestamp")
    if text.endswith("Z"):
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00")
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc)
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise CrosswalkError(f"invalid source timestamp: {text}")


def _team_key(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    return text


def _team_set(*values: Any) -> frozenset[str]:
    return frozenset(_team_key(value) for value in values if _team_key(value))


def _game_order(game_id: str) -> tuple[str, int]:
    prefix, separator, ordinal = str(game_id).rpartition("_")
    if not separator or not ordinal.isdigit() or not prefix:
        raise CrosswalkError(f"Leaguepedia ScoreboardGames.GameId has no source game ordinal: {game_id}")
    return prefix, int(ordinal)


def _sha_object(value: Mapping[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def build_crosswalk(
    oe_rows: list[Mapping[str, Any]],
    scoreboard_rows: list[Mapping[str, Any]],
    schedule_rows: list[Mapping[str, Any]],
    *,
    source_manifest: Mapping[str, Any],
    captured_at: str,
) -> dict[str, Any]:
    issues: list[dict[str, Any]] = []
    if len(oe_rows) != 7:
        issues.append({"kind": "unexpected_oe_row_count", "count": len(oe_rows), "expected": 7})

    scoreboard = [dict(row) for row in scoreboard_rows]
    schedule = [dict(row) for row in schedule_rows]
    for row in scoreboard:
        try:
            row["_time"] = _time(row.get("DateTime UTC"))
            row["_teams"] = _team_set(row.get("Team1"), row.get("Team2"))
            row["_match_id"], row["_game_order"] = _game_order(str(row.get("GameId") or ""))
        except CrosswalkError as error:
            issues.append({"kind": "invalid_scoreboard_row", "row": row, "error": str(error)})
    for row in schedule:
        try:
            row["_time"] = _time(row.get("DateTime UTC"))
            row["_teams"] = _team_set(row.get("Team1"), row.get("Team2"))
        except CrosswalkError as error:
            issues.append({"kind": "invalid_schedule_row", "row": row, "error": str(error)})

    rows: list[dict[str, Any]] = []
    used_scoreboard_ids: set[str] = set()
    seen_oe_ids: set[str] = set()
    for oe in oe_rows:
        oe_id = str(oe.get("gameid") or "").strip()
        try:
            oe_time = _time(oe.get("date"))
        except CrosswalkError as error:
            issues.append({"kind": "invalid_oe_row", "oe_gameid": oe_id, "error": str(error)})
            continue
        oe_patch = str(oe.get("patch") or "").strip()
        oe_league = str(oe.get("league") or "").strip()
        oe_teams = _team_set(*(oe.get("teams") or []))
        if not oe_id or oe_id in seen_oe_ids:
            issues.append({"kind": "duplicate_or_missing_oe_gameid", "oe_gameid": oe_id})
            continue
        seen_oe_ids.add(oe_id)
        if oe_league != "LEC":
            issues.append({"kind": "unexpected_oe_league", "oe_gameid": oe_id, "league": oe_league})
            continue
        if oe_patch != "16.16":
            issues.append({"kind": "unexpected_oe_patch_token", "oe_gameid": oe_id, "patch": oe_patch})
            continue
        if len(oe_teams) != 2:
            issues.append({"kind": "invalid_oe_team_set", "oe_gameid": oe_id, "teams": sorted(oe_teams)})
            continue
        candidates: list[dict[str, Any]] = []
        for candidate in scoreboard:
            if str(candidate.get("Patch") or "") not in {"26.16", "26.16.0"}:
                continue
            if str(candidate.get("OverviewPage") or "") != "LEC/2026 Season/Summer Season":
                continue
            delta = abs((candidate["_time"] - oe_time).total_seconds())
            if candidate["_teams"] != oe_teams or delta > MAX_GAME_TIME_DELTA_SECONDS:
                continue
            candidates.append({**candidate, "_delta": delta})
        if len(candidates) != 1:
            issues.append({
                "kind": "scoreboard_identity_ambiguous" if len(candidates) > 1 else "scoreboard_identity_missing",
                "oe_gameid": oe_id,
                "candidate_count": len(candidates),
                "candidate_game_ids": [str(row.get("GameId") or "") for row in candidates],
            })
            continue
        selected = candidates[0]
        scoreboard_id = str(selected.get("GameId") or "")
        match_id = str(selected.get("_match_id") or "")
        schedule_candidates = [
            row for row in schedule
            if str(row.get("MatchId") or "") == match_id
        ]
        if len(schedule_candidates) != 1:
            issues.append({
                "kind": "schedule_identity_ambiguous" if len(schedule_candidates) > 1 else "schedule_identity_missing",
                "oe_gameid": oe_id,
                "scoreboard_game_id": scoreboard_id,
                "match_id": match_id,
                "candidate_count": len(schedule_candidates),
            })
            continue
        schedule_row = schedule_candidates[0]
        schedule_delta = abs((schedule_row["_time"] - oe_time).total_seconds())
        if schedule_row["_teams"] != oe_teams:
            issues.append({"kind": "schedule_team_mismatch", "oe_gameid": oe_id, "match_id": match_id})
            continue
        # MatchSchedule.DateTime_UTC is the scheduled series start.  It is
        # evidence for the first game only.  Later games can start hours
        # after that timestamp, so their game timestamp must not be compared
        # with the series start as if it were a map start time.
        if int(selected["_game_order"]) == 1 and schedule_delta > MAX_FIRST_GAME_SCHEDULE_DELTA_SECONDS:
            issues.append({
                "kind": "schedule_time_mismatch",
                "oe_gameid": oe_id,
                "match_id": match_id,
                "delta_seconds": schedule_delta,
            })
            continue
        if scoreboard_id in used_scoreboard_ids:
            issues.append({"kind": "duplicate_source_assignment", "oe_gameid": oe_id, "scoreboard_game_id": scoreboard_id})
            continue
        used_scoreboard_ids.add(scoreboard_id)
        rows.append({
            "oe_gameid": oe_id,
            "oe_date": str(oe.get("date") or ""),
            "oe_league": str(oe.get("league") or ""),
            "oe_patch_token": str(oe.get("patch") or ""),
            "oe_team_set": sorted(oe_teams),
            "scoreboard_game_id": scoreboard_id,
            "scoreboard_date": str(selected.get("DateTime UTC") or ""),
            "scoreboard_game_order": int(selected["_game_order"]),
            "series_id": match_id,
            "series_date": str(schedule_row.get("DateTime UTC") or ""),
            "series_patch": str(schedule_row.get("Patch") or ""),
            "series_team_set": sorted(schedule_row["_teams"]),
            "tournament": str(selected.get("Tournament") or ""),
            "overview_page": str(selected.get("OverviewPage") or ""),
            "game_time_delta_seconds": selected["_delta"],
            "series_time_delta_seconds": schedule_delta,
            "assignment_method": "scoreboard_game_team_timestamp_then_exact_matchschedule_prefix",
            "ambiguity_rejected": False,
        })

    rows.sort(key=lambda row: (row["oe_date"], row["oe_gameid"]))
    if len({row["scoreboard_game_id"] for row in rows}) != len(rows):
        issues.append({"kind": "duplicate_scoreboard_game_ids"})
    if len(rows) != len(oe_rows):
        issues.append({"kind": "incomplete_crosswalk", "mapped": len(rows), "expected": len(oe_rows)})

    summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        item = summary.setdefault(row["series_id"], {
            "series_id": row["series_id"],
            "tournament": row["tournament"],
            "team_set": row["series_team_set"],
            "patch": row["series_patch"],
            "game_orders": [],
            "oe_gameids": [],
        })
        item["game_orders"].append(row["scoreboard_game_order"])
        item["oe_gameids"].append(row["oe_gameid"])
    for item in summary.values():
        item["game_orders"].sort()
        item["oe_gameids"].sort()

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "captured_at": captured_at,
        "authority": "private_source_backed_crosswalk",
        "public_probability_authorized": False,
        "public_draft_authorized": False,
        "source_manifest": source_manifest,
        "join_contract": {
            "preferred_exact_riot_game_id": False,
            "fallback": "one_to_one_scoreboard_game_match_on_league_team_set_timestamp_and_source_game_order",
            "max_game_time_delta_seconds": MAX_GAME_TIME_DELTA_SECONDS,
            "max_first_game_schedule_delta_seconds": MAX_FIRST_GAME_SCHEDULE_DELTA_SECONDS,
            "ambiguity_policy": "reject",
            "outcome_inference": False,
        },
        "patch_identity": {
            "source_token": "16.16",
            "public_patch": "26.16",
            "prior_source_token_preserved": "16.15",
            "prior_public_patch": "26.15",
        },
        "counts": {
            "oe_rows": len(oe_rows),
            "scoreboard_rows": len(scoreboard_rows),
            "schedule_rows": len(schedule_rows),
            "mapped_rows": len(rows),
            "mapped_series": len(summary),
            "issues": len(issues),
        },
        "series": sorted(summary.values(), key=lambda item: item["series_id"]),
        "rows": rows,
        "issues": issues,
    }
    result["crosswalk_sha256"] = _sha_object(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--captured-at", default=datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"))
    args = parser.parse_args()
    oe_payload = _load(args.root / "oe-16-16-selected.json")
    manifest = _load(args.root / "capture-manifest.json")
    scoreboard = _load(args.root / "raw" / "scoreboardgames-lec-2026-08-14-15.json")
    schedule = _load(args.root / "raw" / "matchschedule-lec-2026-08-14-15.json")
    if not isinstance(oe_payload, dict) or not isinstance(manifest, dict):
        raise CrosswalkError("selection and capture manifest must be objects")
    if not isinstance(scoreboard, list) or not isinstance(schedule, list):
        raise CrosswalkError("raw Leaguepedia payloads must be arrays")
    result = build_crosswalk(
        oe_payload.get("rows") or [],
        scoreboard,
        schedule,
        source_manifest=manifest,
        captured_at=args.captured_at,
    )
    output = args.root / "crosswalk.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "counts": result["counts"], "series": result["series"], "sha256": result["crosswalk_sha256"]}, indent=2))
    return 0 if not result["issues"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
