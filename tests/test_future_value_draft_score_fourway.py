from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarks.future_value_draft_score_fourway import (
    CURRENT_FEATURES,
    VARIANTS,
    _fit_zero_intercept,
    _side_swap_evidence,
    build_report,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()


def _write_json(path: Path, value: object) -> str:
    raw = _canonical(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _source(tmp_path: Path, game_ids: list[str]) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    source_root.mkdir()
    dates = pd.date_range("2026-01-01", periods=len(game_ids), freq="D", tz="UTC")
    maps = pd.DataFrame(
        {
            "game_uid": game_ids,
            "date": dates,
            "y_blue_win": [0, 1, 1, 0, 1, 0][: len(game_ids)],
        }
    )
    maps.to_parquet(source_root / "maps.parquet")
    source = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-02-01T00:00:00Z",
        "source_game_count": len(game_ids),
        "source_identity_sha256": identity_sha256(game_ids),
        "accepted_game_ids": game_ids,
    }
    source["receipt_sha256"] = hashlib.sha256(_canonical(source)).hexdigest()
    source_path = tmp_path / "source-receipt.json"
    _write_json(source_path, source)
    return source_path, source_root


def _public_pack(tmp_path: Path, source: dict[str, object], game_ids: list[str]) -> tuple[Path, str]:
    root = tmp_path / "pack"
    model_path = tmp_path / "model.json"
    model_raw = b"verified-model-artifact"
    model_path.write_bytes(model_raw)
    model_hash = hashlib.sha256(model_raw).hexdigest()
    authority_path = tmp_path / "authority.json"
    draft_rows: dict[str, dict[str, object]] = {}
    for index, game_id in enumerate(game_ids):
        edge = {
            "base": 0.10 + index * 0.01,
            "ally_synergy": 0.02,
            "enemy_counter": -0.01,
            "same_role": 0.0,
            "archetype_interactions": -0.03,
        }
        draft_rows[game_id] = {
            "date": f"2026-01-{index + 1:02d}T00:00:00Z",
            "edge_components": {**edge, "total": sum(edge.values())},
        }
    draft = {
        "schema_version": "scryglass:draft-records:v1",
        "authority": "descriptive",
        "estimand": "composition_only",
        "model_version": "draft-recommendation-static-v2",
        "fit_through": "2025-12-31T00:00:00Z",
        "source_identity_sha256": source["source_identity_sha256"],
        "artifact_sha256": model_hash,
        "authority_receipt_sha256": "b" * 64,
        "games": draft_rows,
    }
    draft_path = root / "features" / "draft_records.json"
    draft_hash = _write_json(draft_path, draft)
    authority = {
        "schema_version": "scryglass:draft-authority:v1",
        "status": "descriptive",
        "estimand": "composition_only",
        "artifact_sha256": model_hash,
        "probability_authority": False,
        "recommendation_authority": False,
        "betting_authority": False,
    }
    authority_hash = _write_json(authority_path, authority)
    manifest = {
        "source_identity_sha256": source["source_identity_sha256"],
        "source_game_count": source["source_game_count"],
        "draft": {
            "artifact_sha256": draft["artifact_sha256"],
            "receipt_sha256": draft["authority_receipt_sha256"],
        },
        "files": [
            {
                "relative": "features/draft_records.json",
                "bytes": draft_path.stat().st_size,
                "sha256": draft_hash,
            }
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_hash = _write_json(manifest_path, manifest)
    return root, manifest_hash


def _fold_inputs(
    tmp_path: Path,
    game_ids: list[str],
    source_root: Path,
    source: dict[str, object],
) -> tuple[Path, Path]:
    folds = tmp_path / "folds"
    evaluation = tmp_path / "evaluation-v2"
    for fold in (1, 2, 3):
        root = folds / f"fold-{fold}"
        current_root = root / "current-v2"
        scaling_root = root / "scaling-v2"
        current_root.mkdir(parents=True)
        scaling_root.mkdir(parents=True)
        train = game_ids[:2]
        validation = game_ids[2:]
        _write_json(
            folds / f"fold-{fold}-spec.json",
            {
                "fit_window_end": "2026-01-02T12:00:00Z",
                "fold": fold,
                "train_game_ids": train,
                "validation_game_ids": validation,
            },
        )
        current = pd.DataFrame(
            {
                "game_id": game_ids,
                "date": pd.date_range("2026-01-01", periods=6, tz="UTC"),
                "series_id": [f"series-{i}" for i in range(6)],
                "base_team_logit": np.linspace(-0.2, 0.2, 6),
                "team_rating_diff_scaled": np.linspace(0.1, 0.3, 6),
                "base_player_logit": np.linspace(0.3, -0.2, 6),
                "player_rating_diff_scaled": np.linspace(-0.1, 0.2, 6),
            }
        )
        current.to_parquet(current_root / "current-rating-feature-ledger.parquet")
        scaling = pd.DataFrame(
            {
                "game_id": game_ids,
                "date": current["date"],
                "forecast_scaling_index": np.arange(6, dtype=float),
                "forecast_snowball_index": np.arange(6, dtype=float) * -0.5,
                "forecast_curve_available": [True] * 6,
            }
        )
        scaling.to_parquet(scaling_root / "scaling-native.parquet")
    evidence_rows = []
    for index, game_id in enumerate(game_ids):
        evidence_rows.append(
            {
                "game_id": game_id,
                "player_value_logit": float(index) / 10.0,
                "support_status": "adequate",
            }
        )
    evidence = {"rows": evidence_rows}
    evidence["row_count"] = len(evidence_rows)
    evidence["sha256"] = hashlib.sha256(_canonical(evidence_rows)).hexdigest()
    model = {
        "schema_version": "scryglass:future-value-four-variant-evaluation:v1",
        "source": {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
        },
        "variants": {
            "future_player_form": {
                "folds": [
                    {
                        "component_evidence": evidence,
                        "train_end": "2026-01-02T00:00:00Z",
                        "validation_start": "2026-01-03T00:00:00Z",
                    }
                    for _ in (1, 2, 3)
                ]
            }
        },
    }
    model_path = evaluation / "future_player_form" / "model.json"
    _write_json(model_path, model)
    return folds, evaluation


def test_zero_intercept_and_side_swap_are_exact() -> None:
    frame = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0]})
    coefficients, fit = _fit_zero_intercept(frame.to_numpy(), np.array([0.0, 0.0, 1.0, 1.0]))
    assert fit["intercept"] == 0.0
    assert np.isfinite(coefficients).all()
    evidence = _side_swap_evidence(frame, ("x",), coefficients, frame["x"].to_numpy() * coefficients[0])
    assert evidence["status"] == "passed"
    assert evidence["max_logit_error"] == 0.0


def test_fourway_evaluation_fits_all_variants_on_complete_fixture(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    public_root, manifest_hash = _public_pack(tmp_path, source, game_ids)
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    report = build_report(
        source_receipt_path=source_path,
        source_root=source_root,
        folds_root=folds,
        evaluation_root=evaluation,
        public_pack_root=public_root,
        expected_manifest_sha256=manifest_hash,
        authority_path=tmp_path / "authority.json",
        expected_authority_sha256=hashlib.sha256((tmp_path / "authority.json").read_bytes()).hexdigest(),
        model_artifact_path=tmp_path / "model.json",
    )
    assert report["status"] == "research_only"
    assert report["coverage"]["descriptive_subset_game_count"] == 6
    assert not report["blockers"]
    assert set(report["variants"]) == set(VARIANTS)
    static_digests = {
        result["static_components_sha256"] for result in report["variants"].values()
    }
    assert len(static_digests) == 1
    for variant in VARIANTS:
        result = report["variants"][variant]
        assert result["status"] == "evaluated"
        assert result["valid_fold_count"] == 3
        assert all(fold["fit"]["intercept"] == 0.0 for fold in result["folds"])
        assert all(fold["side_swap"]["status"] == "passed" for fold in result["folds"])
        assert all(fold["component_reconstruction_error_max"] == 0.0 for fold in result["folds"])


def test_fourway_fails_closed_when_atom_rows_do_not_cover_train(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    public_root, manifest_hash = _public_pack(tmp_path, source, game_ids[2:])
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    report = build_report(
        source_receipt_path=source_path,
        source_root=source_root,
        folds_root=folds,
        evaluation_root=evaluation,
        public_pack_root=public_root,
        expected_manifest_sha256=manifest_hash,
        authority_path=tmp_path / "authority.json",
        expected_authority_sha256=hashlib.sha256((tmp_path / "authority.json").read_bytes()).hexdigest(),
        model_artifact_path=tmp_path / "model.json",
    )
    assert report["coverage"]["descriptive_subset_game_count"] == 4
    assert report["variants"]["both"]["status"] == "blocked"
    assert "fold_1_static_atom_coverage_missing" in report["blockers"]
    assert "both_requires_three_valid_folds" in report["blockers"]
