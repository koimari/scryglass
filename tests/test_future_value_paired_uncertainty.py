from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from benchmarks.future_value_paired_uncertainty import (
    COMPARISONS,
    PairedUncertaintyError,
    _canonical_bytes,
    _ledger_hash,
    _paired_bootstrap,
    build_report,
    write_report,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


SOURCE = {
    "source_as_of": "2026-08-20T14:51:29Z",
    "source_game_count": 17764,
    "source_identity_sha256": "a" * 64,
    "model_eligible_game_count": 12,
    "model_eligible_identity_sha256": "b" * 64,
    "source_receipt_sha256": "c" * 64,
    "source_receipt_file_sha256": "d" * 64,
}
AUTHORITY = {
    "research_only": True,
    "public_probability": False,
    "public_player_rating": False,
    "public_team_rating": False,
    "promotion": False,
    "deployment": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
}


def _rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fold in (1, 2, 3):
        for map_index in range(4):
            game_id = f"g-{fold}-{map_index}"
            rows.append(
                {
                    "fold": fold,
                    "game_id": game_id,
                    "target": float((map_index + fold) % 2),
                    "series_id": f"series-{fold}-{map_index // 2}",
                    "date": f"2026-01-{fold}{map_index + 1:02d}T00:00:00Z",
                }
            )
    return rows


def _prediction_rows(rows: list[dict[str, Any]], variant_index: int) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        target = float(row["target"])
        base = 0.35 + 0.1 * target
        prediction = min(0.95, max(0.05, base + 0.04 * variant_index))
        output.append(
            {
                "fold": row["fold"],
                "game_id": row["game_id"],
                "target": target,
                "candidate": prediction,
            }
        )
    return output


def _model_document(variant: str, rows: list[dict[str, Any]], variant_index: int) -> dict[str, Any]:
    ledger_rows = _prediction_rows(rows, variant_index)
    ledger = {
        "schema_version": "scryglass:future-value-prediction-ledger:v1",
        "columns": ["fold", "game_id", "target", "candidate"],
        "row_count": len(ledger_rows),
        "game_identity_sha256": identity_sha256(row["game_id"] for row in ledger_rows),
        "rows": ledger_rows,
    }
    ledger["sha256"] = _ledger_hash(ledger_rows)
    return {
        "schema_version": "scryglass:future-value-rating-variants:v2",
        "authority": AUTHORITY,
        "source": SOURCE,
        "variants": {
            variant: {
                "status": "development_evaluated",
                "authority": AUTHORITY,
                "source": SOURCE,
                "prediction_ledger": ledger,
            }
        },
    }


def _bundle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    folds: dict[str, dict[str, Any]] = {}
    for fold in (1, 2, 3):
        folds[str(fold)] = {
            "attrs": {},
            "rows": [row for row in rows if row["fold"] == fold],
        }
    variants = {variant: {"folds": copy.deepcopy(folds)} for variant in (
        "current_only",
        "future_player_form",
        "scaling_curve",
        "both",
    )}
    payload: dict[str, Any] = {
        "schema_version": "scryglass:future-value-four-variant-ledger-bundle:v1",
        "status": "research_only",
        "authority": AUTHORITY,
        "source": SOURCE,
        "variants": variants,
    }
    payload["bundle_sha256"] = __import__("hashlib").sha256(
        _canonical_bytes(payload)
    ).hexdigest()
    return payload


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    rows = _rows()
    evaluation_root = tmp_path / "evaluation"
    for index, variant in enumerate(
        ("current_only", "future_player_form", "scaling_curve", "both")
    ):
        variant_root = evaluation_root / variant
        variant_root.mkdir(parents=True)
        (variant_root / "model.json").write_text(
            json.dumps(_model_document(variant, rows, index), sort_keys=True),
            encoding="utf-8",
        )
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text(json.dumps(_bundle(rows), sort_keys=True), encoding="utf-8")
    return evaluation_root, bundle_path


def test_paired_bootstrap_is_deterministic_and_keeps_series_whole() -> None:
    target = np.asarray([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
    candidate = np.asarray([0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    baseline = np.asarray([0.4, 0.6, 0.4, 0.6, 0.5, 0.5])
    series = ["s1", "s1", "s2", "s2", "s3", "s3"]

    first = _paired_bootstrap(target, candidate, baseline, series, draws=1000, seed=461)
    second = _paired_bootstrap(target, candidate, baseline, series, draws=1000, seed=461)

    assert first == second
    assert first["series_count"] == 3
    assert first["draws_accepted"] == 1000
    assert set(first["metrics"]) == {"log_loss", "brier", "auc", "all_three_metrics"}


def test_build_report_binds_all_variants_and_writes_csv(tmp_path: Path) -> None:
    evaluation_root, bundle_path = _fixture(tmp_path)
    report = build_report(
        evaluation_root=evaluation_root,
        bundle_path=bundle_path,
        draws=1000,
        seed=461,
    )

    assert report["status"] == "research_only"
    assert report["authority"]["research_only"] is True
    assert all(value is False for key, value in report["authority"].items() if key != "research_only")
    assert report["coverage"]["rows"] == 12
    assert report["coverage"]["series_count"] == 6
    assert set(report["comparisons"]) == {
        f"{candidate}_vs_{baseline}" for candidate, baseline in COMPARISONS
    }
    assert all(
        0.0 <= comparison["metrics"]["log_loss"]["probability_candidate_improves"] <= 1.0
        for comparison in report["comparisons"].values()
    )
    json_path, csv_path = write_report(report, tmp_path / "out")
    assert json_path.is_file()
    assert csv_path.is_file()
    assert len(csv_path.read_text(encoding="utf-8").splitlines()) == 13


def test_tampered_prediction_ledger_fails_closed(tmp_path: Path) -> None:
    evaluation_root, bundle_path = _fixture(tmp_path)
    model_path = evaluation_root / "current_only" / "model.json"
    document = json.loads(model_path.read_text(encoding="utf-8"))
    document["variants"]["current_only"]["prediction_ledger"]["rows"][0]["candidate"] = 0.99
    model_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    with pytest.raises(PairedUncertaintyError, match="ledger hash changed"):
        build_report(
            evaluation_root=evaluation_root,
            bundle_path=bundle_path,
            draws=1000,
            seed=461,
        )
