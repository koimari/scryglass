#!/usr/bin/env python3
"""Run the resident exact oracle against hard item/rune modifier packets.

This is the LoL-mechanics analogue of autoresearch's fixed evaluation loop:
the question corpus and acceptance metric stay fixed while router/packet
changes are kept only when they improve correctness without regressing warm
latency.  It intentionally runs in-process; the separate MCP resident-startup
cost is measured by the server tests and is not mixed into per-answer timing.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack


DEFAULT_INDEX = ROOT / "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json"
DEFAULT_CASES = Path(__file__).with_name("league_oracle_modifier_cases.jsonl")


def _engine(index: Path) -> LeagueOracleEngine:
    return LeagueOracleEngine(
        compile_fastpack(index),
        raw_champion_root=index.parent / "raw" / "champions",
    )


def _cases(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("id"), str):
            raise ValueError(f"invalid case at {path}:{line_number}")
        rows.append(row)
    return rows


def _matches(result: dict[str, Any], case: dict[str, Any]) -> bool:
    if result.get("status") != case.get("expected_status"):
        return False
    if "expected_value" in case and result.get("value") != case["expected_value"]:
        return False
    if "expected_remainder" in case and result.get("remainder") != case["expected_remainder"]:
        return False
    return True


def run(index: Path, cases_path: Path) -> dict[str, Any]:
    started = time.perf_counter()
    engine = _engine(index)
    startup_ms = (time.perf_counter() - started) * 1000.0
    cases = _cases(cases_path)
    # Warm the same code path that a resident process uses before measuring.
    if cases:
        engine.answer(str(cases[0]["question"]))
    timings: list[float] = []
    failures: list[dict[str, Any]] = []
    for case in cases:
        question = str(case["question"])
        began = time.perf_counter()
        result = engine.answer(question)
        timings.append((time.perf_counter() - began) * 1000.0)
        if not _matches(result, case):
            failures.append(
                {
                    "id": case["id"],
                    "expected": {
                        key: case[key]
                        for key in ("expected_status", "expected_value", "expected_remainder")
                        if key in case
                    },
                    "actual": {
                        key: result.get(key)
                        for key in ("status", "value", "remainder", "calculation", "reason")
                    },
                }
            )
    ordered = sorted(timings)
    p95_index = max(0, min(len(ordered) - 1, int(len(ordered) * 0.95) - 1)) if ordered else 0
    return {
        "cases": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "accuracy": (len(cases) - len(failures)) / len(cases) if cases else 1.0,
        "startup_ms": round(startup_ms, 3),
        "warm_mean_ms": round(statistics.fmean(timings), 3) if timings else 0.0,
        "warm_p95_ms": round(ordered[p95_index], 3) if ordered else 0.0,
        "warm_max_ms": round(max(timings), 3) if timings else 0.0,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    args = parser.parse_args()
    result = run(args.index, args.cases)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
