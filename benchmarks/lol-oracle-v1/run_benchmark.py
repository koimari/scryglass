#!/usr/bin/env python3
"""Run the 500-question benchmark against the resident fast mechanics path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack
from lol_kills.knowledge.lol_oracle import LeagueOracleEngine


DEFAULT_INDEX = ROOT / "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json"
DEFAULT_QUESTIONS = Path(__file__).with_name("questions.jsonl")


def load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)
    def pick(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
        return ordered[index]
    return {
        "min_ms": round(ordered[0], 4),
        "p50_ms": round(pick(0.50), 4),
        "p95_ms": round(pick(0.95), 4),
        "max_ms": round(ordered[-1], 4),
    }


def run(questions_path: Path, index_path: Path) -> dict[str, Any]:
    startup_started = time.perf_counter()
    pack = compile_fastpack(index_path)
    engine = LeagueOracleEngine(
        pack,
        raw_champion_root=index_path.parent / "raw" / "champions",
    )
    startup_ms = (time.perf_counter() - startup_started) * 1000.0
    rows = load_questions(questions_path)
    timings: list[float] = []
    by_difficulty: dict[str, list[float]] = defaultdict(list)
    status_counts: Counter[str] = Counter()
    baseline_mismatches: list[dict[str, Any]] = []
    exact_matches = 0
    exact_required = 0
    links_complete = 0
    result_rows: list[dict[str, Any]] = []

    for row in rows:
        started = time.perf_counter()
        answer = engine.answer(row["question"])
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timings.append(elapsed_ms)
        by_difficulty[row["difficulty"]].append(elapsed_ms)
        status = answer.get("status")
        status_counts[status] += 1
        expected_status = row["baseline"]["expected_status"]
        status_ok = status == expected_status
        if not status_ok:
            baseline_mismatches.append(
                {"id": row["id"], "expected": expected_status, "actual": status, "question": row["question"]}
            )
        target = row["target"]
        exact_required += int(bool(target.get("exact_required")))
        exact_ok = True
        if target.get("exact_required") and target.get("value") is not None and status == "available":
            expected = target["value"]
            actual = answer.get("value")
            tolerance = 0.01 if isinstance(expected, float) else 0
            exact_ok = actual is not None and abs(float(actual) - float(expected)) <= tolerance
            exact_matches += int(exact_ok)
        links_ok = len(row.get("sources", [])) >= 2 and all(str(item.get("url", "")).startswith("http") for item in row.get("sources", []))
        links_complete += int(links_ok)
        result_rows.append(
            {
                "id": row["id"],
                "difficulty": row["difficulty"],
                "domain": row["domain"],
                "elapsed_ms": round(elapsed_ms, 4),
                "status": status,
                "expected_baseline_status": expected_status,
                "status_ok": status_ok,
                "exact_ok": exact_ok,
                "links_ok": links_ok,
            }
        )

    total_ms = sum(timings)
    return {
        "benchmark": "lol-oracle-v1",
        "questions": len(rows),
        "questions_sha256": sha256_file(questions_path),
        "index_sha256": sha256_file(index_path),
        "patch": pack.get("patch"),
        "client_patch": pack.get("client_patch"),
        "startup_ms": round(startup_ms, 4),
        "warm_answer_latency": percentiles(timings),
        "warm_total_answer_ms": round(total_ms, 4),
        "answer_rate_per_second": round(len(rows) / (total_ms / 1000.0), 2) if total_ms else None,
        "by_difficulty": {difficulty: percentiles(values) for difficulty, values in sorted(by_difficulty.items())},
        "status_counts": dict(sorted(status_counts.items())),
        "baseline_status_accuracy": round((len(rows) - len(baseline_mismatches)) / len(rows), 6) if rows else 0.0,
        "baseline_mismatches": baseline_mismatches,
        "exact_matches": exact_matches,
        "exact_required": exact_required,
        "target_blocked": sum(1 for row in rows if row["target"]["status"] == "blocked"),
        "target_answerable": sum(1 for row in rows if row["target"]["status"] == "available"),
        "current_exact_coverage": round(exact_matches / exact_required, 6) if exact_required else 0.0,
        "link_complete": links_complete,
        "link_total": len(rows),
        "results": result_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("baseline-results.json"))
    args = parser.parse_args()
    report = run(args.questions, args.index)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = {key: value for key, value in report.items() if key not in {"results", "baseline_mismatches"}}
    print(json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
