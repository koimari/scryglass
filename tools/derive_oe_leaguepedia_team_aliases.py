"""Derive repeated OE to Leaguepedia team aliases from local JSON captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from lol_kills.research.oe_leaguepedia_alias_derivation import (
    AliasDerivationError,
    DEFAULT_MAX_TIMESTAMP_DELTA_SECONDS,
    DEFAULT_MIN_REPEATED_EVIDENCE,
    derive_team_alias_mapping,
)


def _load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AliasDerivationError(f"JSON input cannot be read: {path}") from error


def _rows(value: Any, *, label: str) -> list[dict[str, Any]]:
    if isinstance(value, list) and all(isinstance(row, Mapping) for row in value):
        return [dict(row) for row in value]
    if isinstance(value, Mapping):
        for key in ("rows", "games", "data", "result"):
            nested = value.get(key)
            if isinstance(nested, list) and all(isinstance(row, Mapping) for row in nested):
                return [dict(row) for row in nested]
    raise AliasDerivationError(f"{label} JSON must be an array of objects")


def _record(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        if isinstance(value.get("source_records"), Mapping) and isinstance(value["source_records"].get(label), Mapping):
            return dict(value["source_records"][label])
        if isinstance(value.get("input_bindings"), Mapping) and isinstance(value["input_bindings"].get(label), Mapping):
            return dict(value["input_bindings"][label])
        return dict(value)
    raise AliasDerivationError(f"{label} source record must be an object")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oe", type=Path, required=True, help="frozen OE game-row JSON")
    parser.add_argument("--scoreboardgames", type=Path, required=True, help="captured ScoreboardGames JSON")
    parser.add_argument("--matchschedule", type=Path, help="captured MatchSchedule JSON")
    parser.add_argument("--oe-record", type=Path, required=True, help="OE source record or manifest JSON")
    parser.add_argument("--scoreboard-record", type=Path, required=True, help="ScoreboardGames source record or manifest JSON")
    parser.add_argument("--matchschedule-record", type=Path, help="MatchSchedule source record or manifest JSON")
    parser.add_argument("--output", type=Path, required=True, help="derived alias mapping JSON")
    parser.add_argument("--max-seconds", type=int, default=DEFAULT_MAX_TIMESTAMP_DELTA_SECONDS, help="maximum timestamp delta, default 300")
    parser.add_argument("--min-evidence", type=int, default=DEFAULT_MIN_REPEATED_EVIDENCE, help="minimum distinct repeated games, at least 2")
    parser.add_argument("--captured-at", default=None, help="UTC capture timestamp")
    parser.add_argument("--allow-review-only", action="store_true", help="return success while keeping review-required status")
    args = parser.parse_args(argv)
    try:
        oe_raw = _load(args.oe)
        scoreboard_raw = _load(args.scoreboardgames)
        if (args.matchschedule is None) != (args.matchschedule_record is None):
            raise AliasDerivationError("--matchschedule and --matchschedule-record must be supplied together")
        oe_rows = _rows(oe_raw, label="OE")
        scoreboard_rows = _rows(scoreboard_raw, label="ScoreboardGames")
        schedule_rows = _rows(_load(args.matchschedule), label="MatchSchedule") if args.matchschedule else None
        oe_record = _record(_load(args.oe_record), label="oe")
        scoreboard_record = _record(_load(args.scoreboard_record), label="scoreboardgames")
        schedule_record = _record(_load(args.matchschedule_record), label="matchschedule") if args.matchschedule_record else None
        raw_source_bytes = {"oe": args.oe.read_bytes(), "scoreboardgames": args.scoreboardgames.read_bytes()}
        if args.matchschedule:
            raw_source_bytes["matchschedule"] = args.matchschedule.read_bytes()
        result = derive_team_alias_mapping(
            oe_rows,
            scoreboard_rows,
            oe_source_record=oe_record,
            scoreboard_source_record=scoreboard_record,
            schedule_rows=schedule_rows,
            schedule_source_record=schedule_record,
            raw_source_bytes=raw_source_bytes,
            max_timestamp_delta_seconds=args.max_seconds,
            minimum_repeated_evidence=args.min_evidence,
            captured_at=args.captured_at,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(args.output), "status": result["status"], "coverage": result["coverage"], "mapping_sha256": result["mapping_sha256"]}, indent=2, sort_keys=True))
        if result["status"] != "complete_research_only" and not args.allow_review_only:
            return 2
        return 0
    except (AliasDerivationError, OSError) as error:
        parser.error(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
