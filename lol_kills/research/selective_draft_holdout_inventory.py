"""Inspect sealed Draft holdout batches without opening their outcomes."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.research.seal_selective_draft_holdout import (
    SCHEMA_VERSION as BATCH_SCHEMA_VERSION,
)
from lol_kills.research.selective_draft_probability import canonical_sha256


SCHEMA_VERSION = "scryglass:selective-draft-holdout-inventory:v1"
MINIMUM_SELECTED_ROWS = 100
MINIMUM_COVERAGE = 0.75
MINIMUM_LEAGUES_WITH_20_SELECTED_ROWS = 3


class SelectiveDraftHoldoutInventoryError(ValueError):
    """Raised when sealed batch receipts do not form one valid inventory."""


def _verified_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(value)
    expected = receipt.get("receipt_sha256")
    unsigned = {key: item for key, item in receipt.items() if key != "receipt_sha256"}
    if expected != canonical_sha256(unsigned):
        raise SelectiveDraftHoldoutInventoryError("batch receipt does not match")
    if (
        receipt.get("schema_version") != BATCH_SCHEMA_VERSION
        or receipt.get("outcome_blind") is not True
    ):
        raise SelectiveDraftHoldoutInventoryError("batch is not outcome-blind")
    game_ids = receipt.get("game_ids")
    if (
        not isinstance(game_ids, list)
        or len(game_ids) != receipt.get("rows")
        or len(set(game_ids)) != len(game_ids)
        or receipt.get("game_ids_sha256") != canonical_sha256(game_ids)
    ):
        raise SelectiveDraftHoldoutInventoryError("batch game inventory changed")
    rows = receipt.get("rows")
    selected = receipt.get("selected_rows")
    if (
        not isinstance(rows, int)
        or not isinstance(selected, int)
        or rows <= 0
        or selected < 0
        or selected > rows
        or receipt.get("coverage") != selected / rows
    ):
        raise SelectiveDraftHoldoutInventoryError("batch counts changed")
    return receipt


def summarize_holdout_inventory(
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return only count readiness from hash-bound, outcome-blind receipts."""

    if not receipts:
        raise SelectiveDraftHoldoutInventoryError("holdout inventory is empty")
    batches = [_verified_receipt(receipt) for receipt in receipts]
    candidate_receipts = {item.get("candidate_receipt_sha256") for item in batches}
    if len(candidate_receipts) != 1 or None in candidate_receipts:
        raise SelectiveDraftHoldoutInventoryError("candidate receipt changed")

    ordered = sorted(batches, key=lambda item: item["window"]["start"])
    prior_end: str | None = None
    all_ids: set[str] = set()
    league_rows: Counter[str] = Counter()
    selected_league_rows: Counter[str] = Counter()
    for item in ordered:
        window = item.get("window")
        if not isinstance(window, dict):
            raise SelectiveDraftHoldoutInventoryError("batch window is missing")
        start = window.get("start")
        end = window.get("end_exclusive")
        if not isinstance(start, str) or not isinstance(end, str) or start >= end:
            raise SelectiveDraftHoldoutInventoryError("batch window is invalid")
        if prior_end is not None and start < prior_end:
            raise SelectiveDraftHoldoutInventoryError("batch windows overlap")
        prior_end = end
        current_ids = set(item["game_ids"])
        if all_ids.intersection(current_ids):
            raise SelectiveDraftHoldoutInventoryError("batch games overlap")
        all_ids.update(current_ids)
        league_rows.update(
            {str(key): int(value) for key, value in item["league_rows"].items()}
        )
        selected_league_rows.update(
            {
                str(key): int(value)
                for key, value in item["selected_league_rows"].items()
            }
        )

    rows = sum(int(item["rows"]) for item in ordered)
    selected_rows = sum(int(item["selected_rows"]) for item in ordered)
    coverage = selected_rows / rows
    leagues_ready = sum(value >= 20 for value in selected_league_rows.values())
    gates = {
        "minimum_selected_rows": selected_rows >= MINIMUM_SELECTED_ROWS,
        "coverage": coverage >= MINIMUM_COVERAGE,
        "regional_coverage": (
            leagues_ready >= MINIMUM_LEAGUES_WITH_20_SELECTED_ROWS
        ),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "ready_for_one_outcome_join" if all(gates.values()) else "waiting",
        "outcome_blind": True,
        "candidate_receipt_sha256": next(iter(candidate_receipts)),
        "batch_receipt_sha256": [item["receipt_sha256"] for item in ordered],
        "rows": rows,
        "selected_rows": selected_rows,
        "coverage": coverage,
        "league_rows": dict(sorted(league_rows.items())),
        "selected_league_rows": dict(sorted(selected_league_rows.items())),
        "gates": {**gates, "passed": all(gates.values())},
        "outcomes_may_be_joined": all(gates.values()),
        "public_probability": False,
        "public_recommendation": False,
    }
    report["receipt_sha256"] = canonical_sha256(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, action="append", required=True)
    args = parser.parse_args()
    values = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.receipt
    ]
    print(json.dumps(summarize_holdout_inventory(values), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
