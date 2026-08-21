"""Build the full model-eligible V2 Tier List shadow and optional candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

from lol_kills.research.future_value_v2_tier_shadow import (
    V2TierShadowError,
    build_frozen_v2_tier_shadow,
    canonical_json_bytes,
    sha256_path,
)
from lol_kills.research.future_value_tierlist import (
    PINNED_TRUST_MANIFEST_RAW_SHA256,
    load_trust_manifest,
)
from lol_kills.v2.tierlists.pooled_candidate import build_pooled_candidate


RUN_SCHEMA_VERSION = "scryglass:future-value-v2-tier-shadow-run:v1"


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise V2TierShadowError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(dict(value)) + b"\n")


def _stage_tier_runtime(
    runtime_root: Path,
    *,
    source_root: Path,
    repository_root: Path,
    trust: Mapping[str, Any],
) -> None:
    """Stage the frozen OE player source and repository Tier assets."""

    if runtime_root.exists() and (runtime_root.is_symlink() or any(runtime_root.iterdir())):
        raise V2TierShadowError("Tier runtime root must be empty and safe")
    runtime_root.mkdir(parents=True, exist_ok=True)
    player_source = source_root / "source/oe_player_games.parquet"
    meta_source = source_root / "source/meta.json"
    for source, locator in (
        (player_source, "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"),
        (meta_source, "data/lol/warehouse/parquet/oe_live/meta.json"),
    ):
        if not source.is_file() or source.is_symlink():
            raise V2TierShadowError(f"Tier source is missing or unsafe: {source}")
        target = runtime_root / locator
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    assets = trust.get("tier_assets")
    if not isinstance(assets, Mapping) or not assets:
        raise V2TierShadowError("Tier trust manifest has no assets")
    for locator, expected_sha256 in assets.items():
        source = repository_root / locator
        if (
            not source.is_file()
            or source.is_symlink()
            or sha256_path(source) != str(expected_sha256)
        ):
            raise V2TierShadowError(f"Tier asset is missing or unsafe: {locator}")
        target = runtime_root / locator
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-receipt", type=Path, required=True)
    parser.add_argument("--source-receipt-file-sha256", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--model-receipt", type=Path, required=True)
    parser.add_argument("--model-receipt-file-sha256", required=True)
    parser.add_argument("--run-receipt", type=Path, required=True)
    parser.add_argument("--run-receipt-sha256", required=True)
    parser.add_argument("--current-ledger", type=Path, required=True)
    parser.add_argument("--current-ledger-sha256", required=True)
    parser.add_argument("--current-receipt", type=Path, required=True)
    parser.add_argument("--current-receipt-file-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--build-pooled-candidate",
        action="store_true",
        help="Run the existing pooled Tier candidate with the verified offsets.",
    )
    parser.add_argument("--tier-trust-manifest", type=Path)
    parser.add_argument("--tier-trust-manifest-sha256")
    args = parser.parse_args()

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and (output_root.is_symlink() or any(output_root.iterdir())):
        raise V2TierShadowError("output root must be empty and safe")
    output_root.mkdir(parents=True, exist_ok=True)
    result = build_frozen_v2_tier_shadow(
        source_root=args.source_root,
        source_receipt_path=args.source_receipt,
        expected_source_receipt_file_sha256=args.source_receipt_file_sha256,
        model_path=args.model,
        model_receipt_path=args.model_receipt,
        run_receipt_path=args.run_receipt,
        expected_model_sha256=args.model_sha256,
        expected_model_receipt_file_sha256=args.model_receipt_file_sha256,
        expected_run_receipt_sha256=args.run_receipt_sha256,
        current_ledger_path=args.current_ledger,
        current_receipt_path=args.current_receipt,
        expected_current_ledger_sha256=args.current_ledger_sha256,
        expected_current_receipt_file_sha256=args.current_receipt_file_sha256,
        destination=output_root / "v2-tier-offset-ledger.json",
    )
    pooled_input = {
        "schema_version": "scryglass:future-value-v2-tier-pooled-input:v1",
        "status": "research_only",
        "authority": False,
        "allowed_game_ids": list(result.game_ids),
        "pre_map_offset_override": dict(result.offsets),
        "pre_map_offset_provenance": dict(result.provenance),
        "expected_pre_map_offset_source_receipt_sha256": result.provenance[
            "source_receipt_sha256"
        ],
        "ledger_receipt_sha256": result.receipt["receipt_sha256"],
    }
    _write_json(output_root / "pooled-candidate-input.json", pooled_input)
    candidate_record = None
    if args.build_pooled_candidate:
        if (
            args.tier_trust_manifest is None
            or args.tier_trust_manifest_sha256
            != PINNED_TRUST_MANIFEST_RAW_SHA256
        ):
            raise V2TierShadowError(
                "the code-pinned Tier trust manifest is required for a candidate build"
            )
        trust = load_trust_manifest(
            args.tier_trust_manifest,
            expected_raw_sha256=args.tier_trust_manifest_sha256,
        )
        runtime = output_root / "tier-runtime"
        _stage_tier_runtime(
            runtime,
            source_root=args.source_root.resolve(),
            repository_root=args.repository_root.resolve(),
            trust=trust,
        )
        candidate = build_pooled_candidate(
            runtime,
            source_mode="oe_only",
            allowed_game_ids=list(result.game_ids),
            pre_map_offset_override=result.offsets,
            pre_map_offset_provenance=result.provenance,
            expected_pre_map_offset_source_receipt_sha256=str(
                result.provenance["source_receipt_sha256"]
            ),
        )
        candidate_path = output_root / "v2-tier-candidate.json"
        _write_json(candidate_path, candidate)
        candidate_record = {
            "path": candidate_path.name,
            "bytes": candidate_path.stat().st_size,
            "sha256": sha256_path(candidate_path),
        }
    run = {
        "schema_version": RUN_SCHEMA_VERSION,
        "status": "research_only",
        "authority": {
            "research_only": True,
            "public_tierlist": False,
            "merge": False,
            "deployment": False,
        },
        "game_count": len(result.game_ids),
        "game_identity_sha256": result.provenance["source_identity_sha256"],
        "ledger": {
            "path": result.ledger_path.name,
            "bytes": result.ledger_path.stat().st_size,
            "sha256": sha256_path(result.ledger_path),
        },
        "receipt": {
            "path": result.receipt_path.name,
            "bytes": result.receipt_path.stat().st_size,
            "sha256": sha256_path(result.receipt_path),
            "receipt_sha256": result.receipt["receipt_sha256"],
        },
        "pooled_candidate": candidate_record,
        "blockers": [
            "retrospective_full_census_model_fit_not_chronological_evaluation",
            "public_tierlist_authority_missing",
        ],
    }
    _write_json(output_root / "run-receipt.json", run)
    print(json.dumps(run, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
