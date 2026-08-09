"""Outcome-free verifier for the pending latent-capacity assay shell."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any, Sequence

from .oe_nuisance_baseline import validate_artifact as validate_nuisance_artifact
from .oe_target_authority import require_exact_human_authority
from .oe_target_evidence import validate_evidence, validate_split

from .representation_rank_assay import (
    RepresentationRankAssayError,
    canonical_bytes,
    validate_config,
    validate_report,
    verify_config_sources,
)

DEFAULT_CONFIG_PATH = Path(
    "data/lol/v2/models/draft-interactions/representation-rank-assay-config.json"
)
DEFAULT_REPORT_PATH = Path(
    "data/lol/v2/models/draft-interactions/representation-rank-assay-report.json"
)


def _load(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RepresentationRankAssayError(f"{label} is not a regular file")
    raw = path.read_bytes()
    payload = json.loads(raw)
    if raw != canonical_bytes(payload):
        raise RepresentationRankAssayError(f"{label} is not canonical JSON")
    return payload


def verify_pending_shell(
    config_path: Path = DEFAULT_CONFIG_PATH,
    report_path: Path = DEFAULT_REPORT_PATH,
    *,
    root: Path = Path.cwd(),
) -> dict[str, Any]:
    config = _load(config_path, "config")
    report = _load(report_path, "report")
    validate_config(config)
    validate_report(report)
    verify_config_sources(config, root=root)
    if report["config_artifact_sha256"] != config["artifact_sha256"]:
        raise RepresentationRankAssayError("report/config identity mismatch")
    sources = config["source_identity"]

    def source_path(name: str) -> Path:
        locator = Path(sources[name]["locator"])
        return locator if locator.is_absolute() else root / locator

    split = _load(source_path("outcome_free_split"), "outcome-free split")
    evidence = _load(source_path("target_evidence"), "target evidence")
    nuisance = _load(source_path("nuisance_artifact"), "nuisance artifact")
    validate_split(split)
    validate_evidence(evidence)
    authority_bytes = source_path("human_authority").read_bytes()
    fit_authority = require_exact_human_authority(
        authority_bytes, evidence, split, action="model_fit"
    )
    selection_authority = require_exact_human_authority(
        authority_bytes, evidence, split, action="rank_selection"
    )
    if (
        fit_authority.get("reviewer_identity") != "KOI_MARI"
        or selection_authority.get("reviewer_identity") != "KOI_MARI"
        or fit_authority.get("final_temporal_holdout_sealed") is not True
    ):
        raise RepresentationRankAssayError("KOI_MARI authority eligibility failed")
    validate_nuisance_artifact(nuisance)
    gate = nuisance.get("descriptive_diagnostics", {}).get(
        "outer_confirmation_gate", {}
    )
    if (
        gate.get("passed") is not True
        or gate.get("eligible_for_downstream_rank_assay") is not True
        or gate.get("changes_frozen_nuisance_predictions") is not False
        or nuisance.get("oof_materialization", {}).get("rows") != 5702
        or nuisance.get("oof_materialization", {}).get(
            "predicted_game_membership_sha256"
        )
        != "76f7d44585920abf4e1dd37ba478e3849079f45430910444e78dd28b1a8bfa4b"
        or nuisance.get("fold_contract", {}).get("final_temporal_holdout")
        != {
            "status": "sealed_unaccessed",
            "maps": 361,
            "targets_read": False,
            "predictions": 0,
            "fit_rows": 0,
            "score_rows": 0,
        }
        or sum(
            row.get("split") == "final_temporal_holdout"
            for row in split.get("assignments", ())
        )
        != 361
    ):
        raise RepresentationRankAssayError("nuisance downstream eligibility failed")
    runtime = {
        name: importlib.metadata.version(name)
        for name in ("numpy", "pandas", "pyarrow", "scipy")
    }
    if runtime != config["executable_identity"]["runtime_versions"]:
        raise RepresentationRankAssayError("runtime versions changed")
    return {
        "status": report["status"],
        "config_artifact_sha256": config["artifact_sha256"],
        "report_artifact_sha256": report["artifact_sha256"],
        "real_candidate_outcomes_loaded": False,
        "final_temporal_holdout_loaded": False,
        "authoritative_feature_domain_loaded": False,
        "authoritative_target_domain_loaded": False,
        "semantic_authority_reviewer": "KOI_MARI",
        "nuisance_downstream_eligible": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-pending", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify_pending:
        raise RepresentationRankAssayError("private fitting remains disabled")
    print(json.dumps(verify_pending_shell(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
