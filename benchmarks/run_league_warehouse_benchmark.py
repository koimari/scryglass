#!/usr/bin/env python3
"""Benchmark the resident League state warehouse.

This is the database analogue of the local autoresearch loop: the workload is
fixed, the packet receipt is recorded, and a candidate is acceptable only when
all 500 champion-state rows are exact and the resident p95 remains below the
three-minute product ceiling.  SQL and structured-state paths are measured
separately so a fast answer cannot hide a slow materialization step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lol_kills.knowledge.lol_oracle import LeagueOracleEngine
from lol_kills.knowledge.quick_mechanics_fastpack import compile_fastpack
from lol_kills.knowledge.state_warehouse import StateWarehouse


DEFAULT_INDEX = ROOT / "data/lol/knowledge/patch-packets/cdragon/2026/26.15/mechanics-index.json"
DEFAULT_CORPUS = ROOT / "benchmarks/lol-oracle-v1/questions.jsonl"
PRODUCT_LIMIT_MS = 180_000.0


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _corpus_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _cases(warehouse: StateWarehouse, count: int) -> list[tuple[str, int, str]]:
    rows = warehouse.connection.execute(
        "SELECT champion_key, name FROM dim_champion ORDER BY champion_id, champion_key"
    ).fetchall()
    if not rows:
        return []
    cases: list[tuple[str, int, str]] = []
    for index in range(count):
        row = rows[index % len(rows)]
        level = index % 18 + 1
        cases.append((str(row["name"]), level, str(row["champion_key"])))
    return cases


def run(index_path: Path = DEFAULT_INDEX, corpus_path: Path = DEFAULT_CORPUS) -> dict[str, Any]:
    started = time.perf_counter()
    pack = compile_fastpack(index_path)
    oracle = LeagueOracleEngine(pack, raw_champion_root=index_path.parent / "raw" / "champions")
    warehouse = StateWarehouse(pack, oracle=oracle, index_path=index_path)
    startup_ms = (time.perf_counter() - started) * 1000.0
    cases = _cases(warehouse, 500)
    sql_times: list[float] = []
    state_times: list[float] = []
    failures: list[dict[str, Any]] = []
    state_digests: list[str] = []
    for champion, level, champion_key in cases:
        sql = (
            "SELECT champion_key, level, max_health, attack_damage, armor, magic_resist "
            "FROM fact_champion_stat WHERE champion_key = "
            + repr(champion_key)
            + f" AND level = {level}"
        )
        began = time.perf_counter()
        sql_result = warehouse.query_sql(sql)
        sql_times.append((time.perf_counter() - began) * 1000.0)
        began = time.perf_counter()
        state_result = warehouse.state_query({"champion": champion, "level": level, "items": [], "runes": []})
        state_times.append((time.perf_counter() - began) * 1000.0)
        state_digests.append(hashlib.sha256(json.dumps(state_result, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
        if (
            sql_result.get("status") != "available"
            or sql_result.get("row_count") != 1
            or state_result.get("status") != "available"
            or state_result.get("champion") != champion
            or state_result.get("level") != level
        ):
            failures.append(
                {
                    "champion": champion,
                    "level": level,
                    "sql_status": sql_result.get("status"),
                    "sql_row_count": sql_result.get("row_count"),
                    "state_status": state_result.get("status"),
                    "state_champion": state_result.get("champion"),
                }
            )
    digest = hashlib.sha256("".join(state_digests).encode()).hexdigest()
    result = {
        "benchmark": "league-state-warehouse-v1",
        "corpus": str(corpus_path),
        "corpus_sha256": _sha(corpus_path) if corpus_path.is_file() else None,
        "corpus_questions": _corpus_count(corpus_path) if corpus_path.is_file() else None,
        "cases": len(cases),
        "passed": len(cases) - len(failures),
        "failed": len(failures),
        "exact_state_accuracy": round((len(cases) - len(failures)) / len(cases), 6) if cases else 0.0,
        "startup_ms": round(startup_ms, 4),
        "sql_latency": _percentiles(sql_times),
        "state_latency": _percentiles(state_times),
        "under_3_minute_p95": bool((max(sql_times + state_times) if sql_times or state_times else 0.0) < PRODUCT_LIMIT_MS),
        "deterministic_state_digest": digest,
        "warehouse": warehouse.status(),
        "failures": failures,
    }
    warehouse.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.index, args.corpus)
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(encoded)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if result["failed"] == 0 and result["under_3_minute_p95"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
