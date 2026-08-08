#!/usr/bin/env python3
"""Benchmark semantic slot filling and closed-state execution.

The original 500-row runner measures the resident exact-answer path.  This
runner measures the next frontier separately: every formerly impossible row
must receive the right semantic disposition, and closed completion fixtures
must execute to their exact expected result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack
from lol_kills.knowledge.semantic_engine import SemanticOracleEngine


DEFAULT_INDEX = ROOT / "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json"
DEFAULT_QUESTIONS = Path(__file__).with_name("questions.jsonl")
DEFAULT_CASES = Path(__file__).with_name("semantic_cases.jsonl")


EXPECTED_IMPOSSIBLE_STATUS = {
    "missing_patch": "needs_input",
    "ambiguous_entities": "needs_input",
    "hidden_or_outcome_state": "needs_input",
    "nonexistent_or_cross_mode_rule": "invalid_scenario",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentiles(values: list[float]) -> dict[str, float]:
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(
    questions_path: Path = DEFAULT_QUESTIONS,
    cases_path: Path = DEFAULT_CASES,
    index_path: Path = DEFAULT_INDEX,
) -> dict[str, Any]:
    started = time.perf_counter()
    pack = compile_fastpack(index_path)
    base = LeagueOracleEngine(
        pack, raw_champion_root=index_path.parent / "raw" / "champions"
    )
    engine = SemanticOracleEngine(base)
    startup_ms = (time.perf_counter() - started) * 1000.0

    impossible_rows = [
        row for row in _load_jsonl(questions_path) if row.get("difficulty") == "impossible"
    ]
    triage_times: list[float] = []
    triage_mismatches: list[dict[str, Any]] = []
    triage_statuses: Counter[str] = Counter()
    contract_complete = 0
    for row in impossible_rows:
        expected = EXPECTED_IMPOSSIBLE_STATUS.get(str(row.get("domain")), "unsupported")
        started_row = time.perf_counter()
        answer = engine.answer(str(row["question"]))
        elapsed_ms = (time.perf_counter() - started_row) * 1000.0
        triage_times.append(elapsed_ms)
        actual = str(answer.get("status"))
        triage_statuses[actual] += 1
        contract_ok = actual == expected and answer.get("value") is None
        if actual == "needs_input":
            contract_ok = contract_ok and bool(answer.get("required_inputs"))
        if actual == "invalid_scenario":
            contract_ok = contract_ok and bool(answer.get("validation"))
        contract_complete += int(contract_ok)
        if not contract_ok:
            triage_mismatches.append(
                {
                    "id": row["id"],
                    "domain": row["domain"],
                    "expected": expected,
                    "actual": actual,
                    "question": row["question"],
                }
            )

    completion_rows = _load_jsonl(cases_path)
    completion_times: list[float] = []
    completion_matches = 0
    completion_mismatches: list[dict[str, Any]] = []
    for row in completion_rows:
        started_row = time.perf_counter()
        answer = engine.answer(str(row["question"]), row.get("context"))
        elapsed_ms = (time.perf_counter() - started_row) * 1000.0
        completion_times.append(elapsed_ms)
        expected_status = row["expected_status"]
        value_ok = answer.get("value") == row.get("expected_value")
        ok = answer.get("status") == expected_status and value_ok and len(answer.get("sources", [])) >= 2
        completion_matches += int(ok)
        if not ok:
            completion_mismatches.append(
                {
                    "id": row["id"],
                    "expected_status": expected_status,
                    "actual_status": answer.get("status"),
                    "expected_value": row.get("expected_value"),
                    "actual_value": answer.get("value"),
                }
            )

    return {
        "benchmark": "lol-oracle-semantic-v1",
        "patch": pack.get("patch"),
        "client_patch": pack.get("client_patch"),
        "index_sha256": _sha256(index_path),
        "questions_sha256": _sha256(questions_path),
        "semantic_cases_sha256": _sha256(cases_path),
        "startup_ms": round(startup_ms, 4),
        "impossible_questions": len(impossible_rows),
        "semantic_contract_complete": contract_complete,
        "semantic_contract_accuracy": round(contract_complete / len(impossible_rows), 6) if impossible_rows else 0.0,
        "impossible_status_counts": dict(sorted(triage_statuses.items())),
        "impossible_latency": _percentiles(triage_times),
        "triage_mismatches": triage_mismatches,
        "completion_questions": len(completion_rows),
        "completion_exact_matches": completion_matches,
        "completion_exact_coverage": round(completion_matches / len(completion_rows), 6) if completion_rows else 0.0,
        "completion_latency": _percentiles(completion_times),
        "completion_mismatches": completion_mismatches,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("semantic-results.json"))
    args = parser.parse_args()
    report = run(args.questions, args.cases, args.index)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
