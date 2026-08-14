from __future__ import annotations

import json
from pathlib import Path

from lol_kills.research.auc_gate import (
    BASELINE_AUC,
    AucGateError,
    build_reference_receipt,
    receipt_is_valid,
    validate_auc,
)


HASH_A = "a" * 64
HASH_B = "b" * 64


def test_auc_gate_keeps_the_best_validated_reference() -> None:
    result = validate_auc(BASELINE_AUC)
    assert result["auc_noninferior"] is True
    assert validate_auc(BASELINE_AUC - 0.00001)["auc_noninferior"] is False


def test_auc_gate_rejects_missing_or_nonfinite_candidate() -> None:
    assert validate_auc(None)["auc_noninferior"] is False
    assert validate_auc(float("nan"))["auc_noninferior"] is False


def test_reference_receipt_is_development_only_and_hash_bound() -> None:
    receipt = build_reference_receipt(source_sha256=HASH_A, scorer_sha256=HASH_B)
    assert receipt_is_valid(receipt)
    assert receipt["authority"] == "development_only"
    assert receipt["claim_ceiling"]["public_prediction"] is False


def test_reference_receipt_rejects_bad_hashes() -> None:
    try:
        build_reference_receipt(source_sha256="bad", scorer_sha256=HASH_B)
    except AucGateError:
        pass
    else:
        raise AssertionError("invalid source hash was accepted")


def test_checked_in_reference_receipt_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt = json.loads(
        (root / "data/lol/v2/evaluation/auc-preservation-r9e-receipt.json").read_text()
    )
    assert receipt_is_valid(receipt)
    phase = json.loads((root / "data/lol/models/draft_phase_curve.json").read_text())
    assert phase["authority"] == "unavailable"
    assert phase["lcc_atomization"]["public_patch"] == "26.16"
