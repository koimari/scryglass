"""Build the immutable receipt for the eight-map source-census drift."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lol_kills.research.future_value_census_drift import (
    build_census_drift_audit,
    verify_census_drift_audit,
)
from lol_kills.research.future_value_series_authority import (
    canonical_json_bytes,
    file_record,
)


ROOT = Path(__file__).resolve().parents[1]
CURRENT_SOURCE = ROOT / "data/lol/v2/evaluation/future-value-source-receipt-20260820.json"
EXTERNAL_SOURCE = Path(
    "/private/tmp/scryglass-four-variant-freeze-20260820T145129/"
    "future-value-source-receipt.json"
)
BRIDGE_OE_ROWS = Path(
    "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
    "crosswalk-inputs/oe-games-v2.json"
)
CROSSWALK = Path(
    "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
    "oe-leaguepedia-series-crosswalk-v5.json"
)
CROSSWALK_RECEIPT = Path(
    "/private/tmp/scryglass-leaguepedia-series-2025-2026/"
    "oe-leaguepedia-series-crosswalk-v5.receipt.json"
)
DEFAULT_OUTPUT = ROOT / "data/lol/v2/evaluation/future-value-census-drift-audit-v1.json"


def _load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"JSON object rows required: {path}")
    return value


def _repo_record(path: Path) -> dict[str, Any]:
    return file_record(path, locator=path.resolve().relative_to(ROOT).as_posix())


def _external_record(path: Path, locator: str) -> dict[str, Any]:
    return file_record(path, locator=locator)


def build_from_paths(
    *,
    current_source_path: Path = CURRENT_SOURCE,
    external_source_path: Path = EXTERNAL_SOURCE,
    bridge_oe_rows_path: Path = BRIDGE_OE_ROWS,
    crosswalk_path: Path = CROSSWALK,
    crosswalk_receipt_path: Path = CROSSWALK_RECEIPT,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    audit = build_census_drift_audit(
        current_source=_load_object(current_source_path),
        external_source=_load_object(external_source_path),
        bridge_oe_rows=_load_rows(bridge_oe_rows_path),
        crosswalk=_load_object(crosswalk_path),
        crosswalk_receipt=_load_object(crosswalk_receipt_path),
        current_source_artifact=_repo_record(current_source_path),
        external_source_artifact=_external_record(
            external_source_path,
            "external:scryglass-four-variant-freeze-20260820T145129/"
            + external_source_path.name,
        ),
        bridge_oe_artifact=_external_record(
            bridge_oe_rows_path,
            "external:scryglass-leaguepedia-series-2025-2026/"
            "crosswalk-inputs/"
            + bridge_oe_rows_path.name,
        ),
        crosswalk_artifact=_external_record(
            crosswalk_path,
            "external:scryglass-leaguepedia-series-2025-2026/"
            + crosswalk_path.name,
        ),
        crosswalk_receipt_artifact=_external_record(
            crosswalk_receipt_path,
            "external:scryglass-leaguepedia-series-2025-2026/"
            + crosswalk_receipt_path.name,
        ),
    )
    verify_census_drift_audit(audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(audit) + b"\n")
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-source", type=Path, default=CURRENT_SOURCE)
    parser.add_argument("--external-source", type=Path, default=EXTERNAL_SOURCE)
    parser.add_argument("--bridge-oe-rows", type=Path, default=BRIDGE_OE_ROWS)
    parser.add_argument("--crosswalk", type=Path, default=CROSSWALK)
    parser.add_argument("--crosswalk-receipt", type=Path, default=CROSSWALK_RECEIPT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    audit = build_from_paths(
        current_source_path=args.current_source,
        external_source_path=args.external_source,
        bridge_oe_rows_path=args.bridge_oe_rows,
        crosswalk_path=args.crosswalk,
        crosswalk_receipt_path=args.crosswalk_receipt,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "receipt_sha256": audit["receipt_sha256"],
                "external_only_game_count": audit["census_diff"][
                    "external_only_game_count"
                ],
                "verified_series_count": audit["series_summary"][
                    "verified_series_count"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
