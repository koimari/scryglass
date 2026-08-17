"""Freeze one Draft Score evaluation protocol after matrix construction."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from lol_kills.research.atomized_rf_composite import (
    CATEGORICAL_CONTEXT_COLUMNS,
    FEATURE_SCHEMA_VERSION,
    MODEL_COLUMNS,
)
from lol_kills.research.public_draft_score_promotion import sha256_path


class ProtocolFreezeError(ValueError):
    """Raised when an evaluation protocol cannot be frozen safely."""


def freeze_protocol(
    *,
    base_protocol_path: Path,
    matrix_path: Path,
    manifest_path: Path,
    output_path: Path,
    iteration_id: str,
    previous_receipt: str,
    frozen_utc: str,
    single_change: str,
) -> dict[str, Any]:
    if output_path.exists():
        raise ProtocolFreezeError("frozen protocol already exists")
    if not re.fullmatch(r"v[0-9]+", iteration_id):
        raise ProtocolFreezeError("iteration ID is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", previous_receipt):
        raise ProtocolFreezeError("previous receipt SHA-256 is invalid")
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", frozen_utc):
        raise ProtocolFreezeError("frozen time is not UTC RFC3339")
    if not single_change.strip():
        raise ProtocolFreezeError("single-change statement is empty")
    if not base_protocol_path.is_file():
        raise ProtocolFreezeError("base protocol is missing")
    if base_protocol_path.parent.resolve() != output_path.parent.resolve():
        raise ProtocolFreezeError(
            "base and new protocols must share one directory"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolFreezeError("matrix manifest is invalid") from error
    if not isinstance(manifest, dict):
        raise ProtocolFreezeError("matrix manifest is invalid")
    matrix_sha256 = sha256_path(matrix_path)
    manifest_sha256 = sha256_path(manifest_path)
    if manifest.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise ProtocolFreezeError("matrix feature schema changed")
    if manifest.get("matrix_sha256") != matrix_sha256:
        raise ProtocolFreezeError("matrix manifest binds another matrix")
    if manifest.get("model_columns") != list(MODEL_COLUMNS):
        raise ProtocolFreezeError("matrix model columns changed")
    if manifest.get("categorical_columns") != list(
        CATEGORICAL_CONTEXT_COLUMNS
    ):
        raise ProtocolFreezeError("matrix categorical columns changed")
    if manifest.get("columns") != [
        *MODEL_COLUMNS,
        *CATEGORICAL_CONTEXT_COLUMNS,
    ]:
        raise ProtocolFreezeError("matrix column inventory changed")
    document = {
        "schema_version": (
            f"scryglass:public-draft-score-promotion-protocol:{iteration_id}"
        ),
        "status": "frozen_before_first_v1_model_evaluation",
        "frozen_utc": frozen_utc,
        "inherits": base_protocol_path.name,
        "iteration": {
            "id": iteration_id,
            "single_change": single_change.strip(),
            "previous_receipt": previous_receipt,
            "matrix_sha256": matrix_sha256,
            "matrix_manifest_sha256": manifest_sha256,
        },
        "feature_contract": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "current_map_state_allowed": False,
            "final_outcome_features_allowed": False,
            "rating_receipt_required": (
                "scryglass:resolved-rating-source:v1"
            ),
            "rating_context_required": (
                "scryglass:public-draft-score-rating-context:v1"
            ),
            "categorical_context": (
                "exact pre-match team, player, champion, role, patch, "
                "competition, first-pick, and ban identities"
            ),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iteration-id", required=True)
    parser.add_argument("--previous-receipt", required=True)
    parser.add_argument("--frozen-utc", required=True)
    parser.add_argument("--single-change", required=True)
    args = parser.parse_args()
    document = freeze_protocol(
        base_protocol_path=args.base_protocol,
        matrix_path=args.matrix,
        manifest_path=args.manifest,
        output_path=args.output,
        iteration_id=args.iteration_id,
        previous_receipt=args.previous_receipt,
        frozen_utc=args.frozen_utc,
        single_change=args.single_change,
    )
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
