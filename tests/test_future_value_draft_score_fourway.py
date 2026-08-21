from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.future_value_draft_score_fourway import (
    CURRENT_FEATURES,
    FourWayDraftScoreError,
    VARIANTS,
    _canonical_bytes,
    _fit_zero_intercept,
    _load_current,
    _load_future,
    _load_scaling,
    _map_source,
    _scaling_json_value,
    _side_swap_evidence,
    build_report,
    write_report,
)
from lol_kills.v2.tierlists.accepted_census import identity_sha256
from lol_kills.research.atomized_rf_composite import _strict_canonical_sha256
from lol_kills.research.future_value_rating_ledger import _artifact_digest as _current_artifact_digest


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


def test_variants_do_not_require_unselected_optional_producers(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    public_root, manifest_hash = _public_pack(tmp_path, source, game_ids)
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    model_path = evaluation / "future_player_form" / "model.json"
    model = json.loads(model_path.read_text())
    for fold in (1, 2, 3):
        spec = json.loads((folds / f"fold-{fold}-spec.json").read_text())
        validation = set(spec["validation_game_ids"])
        rows = [
            row
            for row in model["variants"]["future_player_form"]["folds"][fold - 1][
                "component_evidence"
            ]["rows"]
            if row["game_id"] in validation
        ]
        evidence = {"rows": rows, "row_count": len(rows)}
        evidence["sha256"] = hashlib.sha256(_canonical(rows)).hexdigest()
        model["variants"]["future_player_form"]["folds"][fold - 1][
            "component_evidence"
        ] = evidence
    _write_json(model_path, model)
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
    assert report["variants"]["current_only"]["status"] == "evaluated"
    assert report["variants"]["scaling_curve"]["status"] == "evaluated"
    assert report["variants"]["future_player_form"]["status"] == "blocked"
    assert report["variants"]["both"]["status"] == "blocked"


def test_map_source_rejects_mutated_frozen_bytes_before_parquet_read(tmp_path: Path) -> None:
    game_ids = ["g1", "g2"]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    map_path = source_root / "maps.parquet"
    raw = map_path.read_bytes()
    source["source_files"] = {
        "maps": {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    }
    map_path.write_bytes(raw + b"mutation")
    with pytest.raises(FourWayDraftScoreError, match="source maps bytes changed"):
        _map_source(source_root, set(game_ids), source=source)


def _write_current_receipt_fixture(tmp_path: Path) -> tuple[Path, list[str]]:
    root = tmp_path / "fold" / "current-v2"
    root.mkdir(parents=True)
    game_ids = ["g1", "g2"]
    frame = pd.DataFrame(
        {
            "game_id": game_ids,
            "date": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "series_id": ["s1", "s2"],
            **{feature: [0.1, -0.2] for feature in CURRENT_FEATURES},
        }
    )
    artifact_path = root / "current-rating-feature-ledger.parquet"
    frame.to_parquet(artifact_path)
    output = frame[["game_id", "date", "series_id", *CURRENT_FEATURES]].assign(
        game_id=lambda value: value["game_id"].astype(str)
    )
    receipt = {
        "schema_version": "scryglass:future-value-current-rating-ledger-receipt:v2",
        "ledger_schema_version": "scryglass:future-value-current-rating-ledger:v2",
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "merge": False,
            "deployment": False,
            "betting": False,
        },
        "fit_window_end": "2025-12-31T00:00:00Z",
        "feature_names": list(CURRENT_FEATURES),
        "output_game_ids": game_ids,
        "output_game_count": len(game_ids),
        "train_game_ids": game_ids,
        "validation_game_ids": [],
        "artifact": {
            "path": str(artifact_path),
            "bytes": artifact_path.stat().st_size,
            "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        },
        "ledger_rows_sha256": _current_artifact_digest(output, CURRENT_FEATURES),
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write_json(root / "current-rating-feature-ledger.receipt.json", receipt)
    return root, game_ids


def test_current_rating_row_mutation_is_rejected_by_receipt(tmp_path: Path) -> None:
    root, game_ids = _write_current_receipt_fixture(tmp_path)
    frame = pd.read_parquet(root / "current-rating-feature-ledger.parquet")
    frame.loc[0, "base_team_logit"] = 99.0
    frame.to_parquet(root / "current-rating-feature-ledger.parquet")
    with pytest.raises(FourWayDraftScoreError, match="current rating artifact bytes changed"):
        _load_current(
            root,
            train_ids=game_ids,
            cutoff_text="2025-12-31T00:00:00Z",
            require_receipt=True,
        )


def _write_scaling_receipt_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "fold" / "scaling-v2"
    root.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "date": pd.date_range("2026-01-01", periods=2, tz="UTC"),
            "forecast_scaling_index": [0.1, -0.2],
            "forecast_snowball_index": [0.3, -0.4],
            "forecast_curve_available": [True, True],
        }
    )
    artifact_path = root / "scaling-native.parquet"
    frame.to_parquet(artifact_path)
    ordered = frame.sort_values(["date", "game_id"], kind="mergesort")
    row_values = [
        {str(column): _scaling_json_value(value) for column, value in row.items()}
        for row in ordered[frame.columns].to_dict("records")
    ]
    receipt = {
        "schema_version": "scryglass:atomized-scaling-feature-ledger:v1",
        "status": "research_only",
        "public_authority": False,
        "fold": 1,
        "fit_window_end": "2025-12-31T00:00:00Z",
        "fold_evaluation_usable": True,
        "output_game_ids": ["g1", "g2"],
        "output_game_count": 2,
        "columns": list(frame.columns),
        "row_value_digest_sha256": _strict_canonical_sha256(row_values),
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write_json(root / "scaling-native-receipt.json", receipt)
    frame.loc[0, "forecast_scaling_index"] = 99.0
    frame.to_parquet(artifact_path)
    return root


def test_scaling_row_mutation_is_rejected_by_receipt(tmp_path: Path) -> None:
    root = _write_scaling_receipt_fixture(tmp_path)
    with pytest.raises(FourWayDraftScoreError, match="scaling native row values changed"):
        _load_scaling(
            root,
            fold=1,
            cutoff_text="2025-12-31T00:00:00Z",
            require_receipt=True,
        )


def test_future_component_value_mutation_is_rejected_by_ledger_hash(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    _, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    model_path = evaluation / "future_player_form" / "model.json"
    model = json.loads(model_path.read_text())
    rows = model["variants"]["future_player_form"]["folds"][0]["component_evidence"]["rows"]
    rows[0]["player_value_logit"] = 99.0
    _write_json(model_path, model)
    with pytest.raises(FourWayDraftScoreError, match="component ledger changed"):
        _load_future(model_path, 1)


def test_chronology_mutation_blocks_fold_evaluation(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    public_root, manifest_hash = _public_pack(tmp_path, source, game_ids)
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    spec_path = folds / "fold-1-spec.json"
    spec = json.loads(spec_path.read_text())
    spec["fit_window_end"] = "2026-01-03T12:00:01Z"
    _write_json(spec_path, spec)
    model_path = evaluation / "future_player_form" / "model.json"
    model = json.loads(model_path.read_text())
    model["variants"]["future_player_form"]["folds"][0][
        "validation_start"
    ] = "2026-01-03T12:00:01Z"
    _write_json(model_path, model)
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
    assert report["variants"]["current_only"]["folds"][0]["status"] == "blocked"
    assert "current_only_fold_1_validation_chronology_invalid" in report["blockers"]


def test_report_hash_covers_written_payload_and_descriptive_subset(tmp_path: Path) -> None:
    report = {
        "schema_version": "scryglass:future-value-draft-score-fourway:v1",
        "status": "research_only",
        "descriptive_rows": [{"game_id": "g1", "static_total_logit": None}],
    }
    output = tmp_path / "report"
    write_report(report, output)
    payload = json.loads((output / "fourway-report.json").read_text())
    descriptive_raw = (output / "descriptive-subset.json").read_bytes()
    claimed_report = payload.pop("report_sha256")
    assert claimed_report == hashlib.sha256(_canonical(payload)).hexdigest()
    assert payload["descriptive_rows_sha256"] == hashlib.sha256(descriptive_raw).hexdigest()
    (output / "descriptive-subset.json").write_bytes(descriptive_raw + b"mutation")
    assert payload["descriptive_rows_sha256"] != hashlib.sha256(
        (output / "descriptive-subset.json").read_bytes()
    ).hexdigest()
