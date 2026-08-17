"""Verify an independent public Draft Score promotion decision."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from lol_kills.research.evaluate_selective_draft_holdout import (
    SCHEMA_VERSION as EVALUATION_SCHEMA_VERSION,
)
from lol_kills.research.public_draft_score_promotion import sha256_path
from lol_kills.research.selective_draft_probability import canonical_sha256


DECISION_SCHEMA_VERSION = "scryglass:selective-draft-promotion-decision:v1"
RECEIPT_SCHEMA_VERSION = "scryglass:public-draft-score-promotion-receipt:v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
UTC_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z"
)
APPROVED_FIELDS = (
    "match_win_probability",
    "controlled_draft_score",
    "side_recommendation",
)


class SelectiveDraftPromotionVerificationError(ValueError):
    """Raised when an independent promotion decision is invalid."""


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SelectiveDraftPromotionVerificationError("input is not an object")
    return value


def _receipt_matches(value: Mapping[str, Any]) -> bool:
    expected = value.get("receipt_sha256")
    unsigned = {key: item for key, item in value.items() if key != "receipt_sha256"}
    return expected == canonical_sha256(unsigned)


def verify_promotion_decision(
    *,
    evaluation_path: Path,
    expected_evaluation_sha256: str,
    decision_path: Path,
    expected_decision_sha256: str,
    output_path: Path,
) -> dict[str, Any]:
    """Return public authority only for one independent passing decision."""

    if output_path.exists():
        raise SelectiveDraftPromotionVerificationError("receipt output already exists")
    pairs = (
        (evaluation_path, expected_evaluation_sha256, "evaluation"),
        (decision_path, expected_decision_sha256, "decision"),
    )
    for path, expected, label in pairs:
        if not SHA256_PATTERN.fullmatch(expected):
            raise SelectiveDraftPromotionVerificationError(
                f"{label} SHA-256 is invalid"
            )
        if not path.is_file() or sha256_path(path) != expected:
            raise SelectiveDraftPromotionVerificationError(f"{label} changed")
    evaluation = _json(evaluation_path)
    decision = _json(decision_path)
    if not _receipt_matches(evaluation) or not _receipt_matches(decision):
        raise SelectiveDraftPromotionVerificationError("input receipt changed")
    if (
        evaluation.get("schema_version") != EVALUATION_SCHEMA_VERSION
        or evaluation.get("status") != "independent_promotion_receipt_required"
        or evaluation.get("gates", {}).get("passed") is not True
        or evaluation.get("public_probability") is not False
        or evaluation.get("public_recommendation") is not False
    ):
        raise SelectiveDraftPromotionVerificationError("evaluation did not pass")
    paired_receipts = evaluation.get("controlled_intervention_receipt_sha256")
    if (
        not isinstance(paired_receipts, list)
        or not paired_receipts
        or any(
            not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value)
            for value in paired_receipts
        )
    ):
        raise SelectiveDraftPromotionVerificationError(
            "paired Draft intervention evidence is invalid"
        )
    if (
        decision.get("schema_version") != DECISION_SCHEMA_VERSION
        or decision.get("decision") != "promoted"
        or decision.get("independent_from_model_development") is not True
        or not str(decision.get("reviewer_identity") or "").strip()
    ):
        raise SelectiveDraftPromotionVerificationError(
            "independent decision is invalid"
        )
    issued = decision.get("issued_utc")
    if not isinstance(issued, str) or not UTC_PATTERN.fullmatch(issued):
        raise SelectiveDraftPromotionVerificationError("decision time is invalid")
    datetime.fromisoformat(issued.removesuffix("Z") + "+00:00")
    if (
        decision.get("evaluation_file_sha256") != expected_evaluation_sha256
        or decision.get("evaluation_receipt_sha256")
        != evaluation.get("receipt_sha256")
        or decision.get("candidate_receipt_sha256")
        != evaluation.get("candidate_receipt_sha256")
        or decision.get("outcomes_sha256") != evaluation.get("outcomes_sha256")
    ):
        raise SelectiveDraftPromotionVerificationError("decision binding changed")
    if tuple(decision.get("approved_public_fields") or ()) != APPROVED_FIELDS:
        raise SelectiveDraftPromotionVerificationError("public fields are invalid")
    if decision.get("betting_odds_ev_stake") is not False:
        raise SelectiveDraftPromotionVerificationError("betting fields are enabled")
    model_version = str(decision.get("model_version") or "").strip()
    if not model_version:
        raise SelectiveDraftPromotionVerificationError("model version is missing")

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "promoted",
        "authority": "promoted",
        "model_version": model_version,
        "candidate_receipt_sha256": evaluation["candidate_receipt_sha256"],
        "protocol_file_sha256": evaluation["protocol_file_sha256"],
        "evaluation_file_sha256": expected_evaluation_sha256,
        "evaluation_receipt_sha256": evaluation["receipt_sha256"],
        "decision_file_sha256": expected_decision_sha256,
        "decision_receipt_sha256": decision["receipt_sha256"],
        "outcomes_sha256": evaluation["outcomes_sha256"],
        "controlled_intervention_receipt_sha256": paired_receipts,
        "reviewer_identity": decision["reviewer_identity"],
        "issued_utc": issued,
        "approved_public_fields": list(APPROVED_FIELDS),
        "public_probability": True,
        "public_recommendation": True,
        "betting_odds_ev_stake": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--evaluation-sha256", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = verify_promotion_decision(
        evaluation_path=args.evaluation,
        expected_evaluation_sha256=args.evaluation_sha256,
        decision_path=args.decision,
        expected_decision_sha256=args.decision_sha256,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
