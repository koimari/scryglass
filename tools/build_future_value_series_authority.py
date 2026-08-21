"""Build the immutable future-value series authority audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lol_kills.research.future_value_series_authority import (
    TARGET_PROXY_MAP_COUNT,
    build_series_authority_audit,
    canonical_json_bytes,
    file_record,
    verify_series_authority_audit,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data/lol/v2/evaluation/future-value-source-receipt-20260820.json"
DEFAULT_CENSUS = ROOT / "data/lol/v2/evaluation/future-phase-accepted-census.json"
DEFAULT_PHASE = ROOT / "data/lol/v2/evaluation/future-phase-evaluation.json"
DEFAULT_PROXY = ROOT / "data/lol/v2/models/draft-interactions/series-cluster-proxy.json"
DEFAULT_OUTPUT = ROOT / "data/lol/v2/evaluation/future-value-series-authority-audit-v1.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _external_record(path: Path, locator: str) -> dict[str, Any]:
    return file_record(path, locator=locator)


def _repo_record(path: Path) -> dict[str, Any]:
    return file_record(path, locator=path.resolve().relative_to(ROOT).as_posix())


def build_from_paths(
    *,
    source_path: Path = DEFAULT_SOURCE,
    census_path: Path = DEFAULT_CENSUS,
    phase_path: Path = DEFAULT_PHASE,
    proxy_path: Path = DEFAULT_PROXY,
    leaguepedia_crosswalk_path: Path | None = None,
    leaguepedia_crosswalk_receipt_path: Path | None = None,
    variant_bundle_path: Path | None = None,
    output_path: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    crosswalk = (
        _load(leaguepedia_crosswalk_path)
        if leaguepedia_crosswalk_path is not None
        else None
    )
    crosswalk_receipt = (
        _load(leaguepedia_crosswalk_receipt_path)
        if leaguepedia_crosswalk_receipt_path is not None
        else None
    )
    crosswalk_record = (
        _external_record(
            leaguepedia_crosswalk_path,
            "external:scryglass-leaguepedia-series-2025-2026/"
            + leaguepedia_crosswalk_path.name,
        )
        if leaguepedia_crosswalk_path is not None
        else None
    )
    crosswalk_receipt_record = (
        _external_record(
            leaguepedia_crosswalk_receipt_path,
            "external:scryglass-leaguepedia-series-2025-2026/"
            + leaguepedia_crosswalk_receipt_path.name,
        )
        if leaguepedia_crosswalk_receipt_path is not None
        else None
    )
    variant_bundle = (
        _load(variant_bundle_path) if variant_bundle_path is not None else None
    )
    variant_bundle_record = (
        _external_record(
            variant_bundle_path,
            "external:scryglass-four-variant-runs/" + variant_bundle_path.name,
        )
        if variant_bundle_path is not None
        else None
    )
    audit = build_series_authority_audit(
        source_receipt=_load(source_path),
        accepted_census=_load(census_path),
        phase_evaluation=_load(phase_path),
        proxy_artifact=_load(proxy_path),
        source_receipt_artifact=_repo_record(source_path),
        accepted_census_artifact=_repo_record(census_path),
        phase_evaluation_artifact=_repo_record(phase_path),
        proxy_artifact_file=_repo_record(proxy_path),
        requested_proxy_map_count=TARGET_PROXY_MAP_COUNT,
        leaguepedia_crosswalk=crosswalk,
        leaguepedia_crosswalk_receipt=crosswalk_receipt,
        leaguepedia_crosswalk_artifact_file=crosswalk_record,
        leaguepedia_crosswalk_receipt_file=crosswalk_receipt_record,
        variant_bundle=variant_bundle,
        variant_bundle_file=variant_bundle_record,
    )
    verify_series_authority_audit(audit)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_json_bytes(audit) + b"\n")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    parser.add_argument("--proxy", type=Path, default=DEFAULT_PROXY)
    parser.add_argument("--leaguepedia-crosswalk", type=Path)
    parser.add_argument("--leaguepedia-crosswalk-receipt", type=Path)
    parser.add_argument("--variant-bundle", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    audit = build_from_paths(
        source_path=args.source,
        census_path=args.census,
        phase_path=args.phase,
        proxy_path=args.proxy,
        leaguepedia_crosswalk_path=args.leaguepedia_crosswalk,
        leaguepedia_crosswalk_receipt_path=args.leaguepedia_crosswalk_receipt,
        variant_bundle_path=args.variant_bundle,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "status": audit["status"],
                "receipt_sha256": audit["receipt_sha256"],
                "blocker_count": len(audit["blockers"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
