from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from benchmarks.future_value_draft_score_fourway import (
    CURRENT_FEATURES,
    FourWayDraftScoreError,
    VARIANTS,
    _canonical_bytes,
    _fit_zero_intercept,
    _fit_zero_intercept_log_loss,
    _load_current,
    _load_future,
    _load_scaling,
    _load_source,
    _map_source,
    _scaling_json_value,
    _side_swap_evidence,
    _verify_strict_prior_form_artifact,
    build_report,
    write_report,
)
from benchmarks.freeze_future_value_draft_score_fourway import (
    DraftScoreFreezeError,
    build_freeze,
    write_freeze,
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
    players_path = source_root / "players.parquet"
    pd.DataFrame({"game_uid": game_ids, "player_id": [f"p{i}" for i in range(len(game_ids))]}).to_parquet(players_path)
    maps_raw = (source_root / "maps.parquet").read_bytes()
    players_raw = players_path.read_bytes()
    source = {
        "schema_version": "scryglass:future-value-rating-source:v1",
        "status": "accepted_source_bound_development_only",
        "source_as_of": "2026-02-01T00:00:00Z",
        "source_game_count": len(game_ids),
        "source_identity_sha256": identity_sha256(game_ids),
        "accepted_game_ids": game_ids,
        "source_files": {
            "maps": {"bytes": len(maps_raw), "sha256": hashlib.sha256(maps_raw).hexdigest()},
            "players": {"bytes": len(players_raw), "sha256": hashlib.sha256(players_raw).hexdigest()},
        },
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
        current_artifact = current_root / "current-rating-feature-ledger.parquet"
        current_output = current[["game_id", "date", "series_id", *CURRENT_FEATURES]].copy()
        current_receipt = {
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
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
            "fit_window_end": "2026-01-02T12:00:00Z",
            "feature_names": list(CURRENT_FEATURES),
            "output_game_ids": game_ids,
            "output_game_count": len(game_ids),
            "train_game_ids": train,
            "validation_game_ids": validation,
            "artifact": {
                "path": str(current_artifact),
                "bytes": current_artifact.stat().st_size,
                "sha256": hashlib.sha256(current_artifact.read_bytes()).hexdigest(),
            },
            "ledger_rows_sha256": _current_artifact_digest(current_output, CURRENT_FEATURES),
        }
        current_receipt["receipt_sha256"] = hashlib.sha256(_canonical(current_receipt)).hexdigest()
        _write_json(current_root / "current-rating-feature-ledger.receipt.json", current_receipt)
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
        scaling_columns = list(scaling.columns)
        ordered = scaling.sort_values(["date", "game_id"], kind="mergesort")
        scaling_values = [
            {str(column): _scaling_json_value(value) for column, value in row.items()}
            for row in ordered[scaling_columns].to_dict("records")
        ]
        scaling_receipt = {
            "schema_version": "scryglass:atomized-scaling-feature-ledger:v1",
            "status": "research_only",
            "public_authority": False,
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
            "accepted_game_count": source["source_game_count"],
            "fold": fold,
            "fit_window_end": "2026-01-02T12:00:00Z",
            "fold_evaluation_usable": True,
            "output_game_ids": game_ids,
            "output_game_count": len(game_ids),
            "columns": scaling_columns,
            "row_value_digest_sha256": _strict_canonical_sha256(scaling_values),
        }
        scaling_receipt["receipt_sha256"] = hashlib.sha256(_canonical(scaling_receipt)).hexdigest()
        _write_json(scaling_root / "scaling-native-receipt.json", scaling_receipt)
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


def _strict_fold_inputs(
    tmp_path: Path,
    game_ids: list[str],
    source: dict[str, object],
    folds: Path,
) -> Path:
    root = tmp_path / "strict-folds"
    prior_validation: set[str] = set()
    dates = {game_id: pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=index) for index, game_id in enumerate(game_ids)}
    for fold in (1, 2, 3):
        spec = json.loads((folds / f"fold-{fold}-spec.json").read_text())
        train = set(spec["train_game_ids"])
        validation = set(spec["validation_game_ids"])
        excluded = train & prior_validation
        effective = train - excluded
        contract = {
            "fold": fold,
            "fit_window_end": "2026-01-02T12:00:00Z",
            "train_game_count": len(train),
            "train_game_identity_sha256": identity_sha256(train),
            "effective_train_game_count": len(effective),
            "effective_train_game_identity_sha256": identity_sha256(effective),
            "validation_game_count": len(validation),
            "validation_game_identity_sha256": identity_sha256(validation),
            "excluded_previous_validation_game_count": len(excluded),
            "excluded_previous_validation_identity_sha256": identity_sha256(excluded),
            "validation_feature_state": "frozen_effective_training_before_cutoff_calendar_day",
        }
        source_binding = {
            "source_as_of": source["source_as_of"],
            "source_game_count": source["source_game_count"],
            "source_identity_sha256": source["source_identity_sha256"],
            "source_receipt_sha256": source["receipt_sha256"],
            "input_files": source["source_files"],
        }
        atom_rows = []
        form_rows = []
        for index, game_id in enumerate(sorted(train | validation)):
            date = dates[game_id]
            is_excluded = game_id in excluded
            edge = {
                "base": 0.1 + index * 0.01,
                "ally_synergy": 0.02,
                "enemy_counter": -0.01,
                "same_role": 0.0,
                "archetype_interactions": -0.03,
            }
            fit_through = "2026-01-01T00:00:00Z" if date > pd.Timestamp("2026-01-01", tz="UTC") else None
            atom_rows.append({
                "game_id": game_id,
                "date": date.isoformat().replace("+00:00", "Z"),
                "fit_through": None if is_excluded else fit_through,
                "status": "unavailable" if is_excluded else "available",
                "reason": "excluded_previous_outer_validation" if is_excluded else None,
                "edge_components": None if is_excluded else {**edge, "total": sum(edge.values())},
            })
            form_rows.append({
                "game_id": game_id,
                "date": date.isoformat().replace("+00:00", "Z"),
                "fit_through": None if is_excluded else fit_through,
                "status": "unavailable" if is_excluded else "available",
                "reason": "excluded_previous_outer_validation" if is_excluded else None,
                "future_player_form_logit": None if is_excluded else float(index) / 10.0,
            })
        common = {
            "status": "research_only",
            "authority": {"research_only": True, "public": False, "probability": False, "promotion": False, "deployment": False},
            "source": source_binding,
            "fold_contract": contract,
        }
        atom = {
            **common,
            "schema_version": "scryglass:strict-prior-composition-atoms:v1",
            "producer": {
                "training_order": "effective outer-fold training rows only; validation state frozen at cutoff",
                "producer_code_sha256": "a" * 64,
                "composition_signal_code_sha256": "b" * 64,
                "component_mapping": {},
            },
            "coverage": {"fit_through_max": "2026-01-01T00:00:00Z"},
            "rows": atom_rows,
            "rows_sha256": hashlib.sha256(_canonical(atom_rows)).hexdigest(),
        }
        atom["artifact_sha256"] = hashlib.sha256(_canonical(atom)).hexdigest()
        form = {
            **common,
            "schema_version": "scryglass:strict-prior-player-form:v1",
            "producer": {
                "training_order": "effective outer-fold training rows only; validation state frozen at cutoff",
                "producer_code_sha256": "a" * 64,
            },
            "coverage": {"fit_through_max": "2026-01-01T00:00:00Z"},
            "rows": form_rows,
            "rows_sha256": hashlib.sha256(_canonical(form_rows)).hexdigest(),
        }
        form["artifact_sha256"] = hashlib.sha256(_canonical(form)).hexdigest()
        _write_json(root / f"fold-{fold}" / "strict-prior-composition-atoms.json", atom)
        _write_json(root / f"fold-{fold}" / "strict-prior-player-form.json", form)
        prior_validation.update(validation)
    return root


def _freeze_inputs(tmp_path: Path, source_path: Path, folds: Path, strict_root: Path) -> tuple[Path, str]:
    path = tmp_path / "fourway-freeze.json"
    payload = build_freeze(
        source_receipt_path=source_path,
        folds_root=folds,
        strict_fold_root=strict_root,
        evaluation_root=tmp_path / "evaluation-v2",
    )
    digest = write_freeze(payload, path)
    return path, digest


def test_freeze_binds_shared_future_model_for_each_fold(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    payload = build_freeze(
        source_receipt_path=source_path,
        folds_root=folds,
        strict_fold_root=strict_root,
        evaluation_root=evaluation,
    )
    model_path = evaluation / "future_player_form" / "model.json"
    expected = {
        "bytes": model_path.stat().st_size,
        "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    for fold in ("1", "2", "3"):
        assert payload["folds"][fold]["future"]["artifact"] == expected


def test_future_model_mutation_is_rejected_by_freeze_binding(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    payload = build_freeze(
        source_receipt_path=source_path,
        folds_root=folds,
        strict_fold_root=strict_root,
        evaluation_root=evaluation,
    )
    model_path = evaluation / "future_player_form" / "model.json"
    model_path.write_bytes(model_path.read_bytes() + b"mutation")
    with pytest.raises(FourWayDraftScoreError, match="unsafe|bytes changed"):
        from benchmarks.future_value_draft_score_fourway import _verify_pinned_file

        _verify_pinned_file(
            model_path,
            payload["folds"]["1"]["future"]["artifact"],
            label="future player form artifact",
        )


def test_freeze_rejects_symlinked_evaluation_root(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    evaluation_link = tmp_path / "evaluation-link"
    evaluation_link.symlink_to(evaluation, target_is_directory=True)
    with pytest.raises(DraftScoreFreezeError, match="unsafe"):
        build_freeze(
            source_receipt_path=source_path,
            folds_root=folds,
            strict_fold_root=strict_root,
            evaluation_root=evaluation_link,
        )


def test_freeze_cli_requires_and_emits_future_binding(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    output = tmp_path / "cli-freeze.json"
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "benchmarks" / "freeze_future_value_draft_score_fourway.py"),
        "--source-receipt",
        str(source_path),
        "--folds-root",
        str(folds),
        "--strict-fold-root",
        str(strict_root),
        "--evaluation-root",
        str(evaluation),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["path"] == str(output.resolve())
    payload = json.loads(output.read_text())
    assert payload["folds"]["1"]["future"]["artifact"]["bytes"] == (
        evaluation / "future_player_form" / "model.json"
    ).stat().st_size


def test_zero_intercept_and_side_swap_are_exact() -> None:
    frame = pd.DataFrame({"x": [-2.0, -1.0, 1.0, 2.0]})
    coefficients, fit = _fit_zero_intercept(frame.to_numpy(), np.array([0.0, 0.0, 1.0, 1.0]))
    assert fit["intercept"] == 0.0
    assert np.isfinite(coefficients).all()
    evidence = _side_swap_evidence(frame, ("x",), coefficients, frame["x"].to_numpy() * coefficients[0])
    assert evidence["status"] == "passed"
    assert evidence["max_logit_error"] == 0.0


def test_regularized_log_loss_fit_is_deterministic_and_zero_intercept() -> None:
    frame = np.array([[-2.0, 0.5], [-1.0, 0.25], [1.0, -0.25], [2.0, -0.5]])
    target = np.array([0.0, 0.0, 1.0, 1.0])
    coefficients_a, fit_a = _fit_zero_intercept_log_loss(frame, target)
    coefficients_b, fit_b = _fit_zero_intercept_log_loss(frame, target)
    np.testing.assert_array_equal(coefficients_a, coefficients_b)
    assert fit_a == fit_b
    assert fit_a["method"] == "zero_intercept_log_loss_ridge_v1"
    assert fit_a["objective"] == "mean_log_loss_plus_l2"
    assert fit_a["intercept"] == 0.0
    assert fit_a["l2_penalty"] == 0.01
    assert fit_a["converged"] is True
    assert fit_a["gradient_inf_norm"] <= 1e-10


def test_regularized_log_loss_rejects_non_binary_targets() -> None:
    with pytest.raises(FourWayDraftScoreError, match="targets must be binary"):
        _fit_zero_intercept_log_loss(
            np.array([[-1.0], [1.0]]),
            np.array([0.0, 0.5]),
        )


def test_fourway_evaluation_fits_all_variants_on_complete_fixture(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    public_root, manifest_hash = _public_pack(tmp_path, source, game_ids)
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    freeze_path, freeze_hash = _freeze_inputs(tmp_path, source_path, folds, strict_root)
    report = build_report(
        source_receipt_path=source_path,
        expected_source_receipt_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        trust_root_path=freeze_path,
        expected_trust_root_sha256=freeze_hash,
        source_root=source_root,
        folds_root=folds,
        evaluation_root=evaluation,
        public_pack_root=public_root,
        expected_manifest_sha256=manifest_hash,
        authority_path=tmp_path / "authority.json",
        expected_authority_sha256=hashlib.sha256((tmp_path / "authority.json").read_bytes()).hexdigest(),
        model_artifact_path=tmp_path / "model.json",
        strict_fold_root=strict_root,
    )
    assert report["status"] == "research_only"
    assert report["coverage"]["descriptive_subset_game_count"] == 6
    assert not report["blockers"]
    bootstrap = report["paired_bootstrap"]
    assert bootstrap["status"] == "evaluated"
    assert bootstrap["seed"] == 20260821
    assert bootstrap["draws"] == 2000
    assert bootstrap["input"]["row_count"] == 12
    assert len(bootstrap["input"]["rows_sha256"]) == 64
    assert len(bootstrap["input"]["row_identity_sha256"]) == 64
    assert set(bootstrap["comparisons"]) == {
        "v2_vs_v1",
        "v3_vs_v1",
        "v4_vs_v1",
        "v4_vs_v2",
    }
    for comparison in bootstrap["comparisons"].values():
        for metric in ("log_loss", "brier", "auc"):
            result = comparison["metrics"][metric]
            assert result["status"] == "evaluated"
            assert 0.0 <= result["improvement_probability"] <= 1.0
            assert 0 < result["finite_draws"] <= 2000
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
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    atom_path = strict_root / "fold-1" / "strict-prior-composition-atoms.json"
    atom = json.loads(atom_path.read_text())
    validation = set(json.loads((folds / "fold-1-spec.json").read_text())["validation_game_ids"])
    for row in atom["rows"]:
        if row["game_id"] in validation:
            row["status"] = "unavailable"
            row["reason"] = "test_missing"
            row["edge_components"] = None
    atom["rows_sha256"] = hashlib.sha256(_canonical(atom["rows"])).hexdigest()
    atom.pop("artifact_sha256")
    atom["artifact_sha256"] = hashlib.sha256(_canonical(atom)).hexdigest()
    _write_json(atom_path, atom)
    freeze_path, freeze_hash = _freeze_inputs(tmp_path, source_path, folds, strict_root)
    report = build_report(
        source_receipt_path=source_path,
        expected_source_receipt_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        trust_root_path=freeze_path,
        expected_trust_root_sha256=freeze_hash,
        source_root=source_root,
        folds_root=folds,
        evaluation_root=evaluation,
        public_pack_root=public_root,
        expected_manifest_sha256=manifest_hash,
        authority_path=tmp_path / "authority.json",
        expected_authority_sha256=hashlib.sha256((tmp_path / "authority.json").read_bytes()).hexdigest(),
        model_artifact_path=tmp_path / "model.json",
        strict_fold_root=strict_root,
    )
    assert report["coverage"]["descriptive_subset_game_count"] == 6
    assert report["variants"]["both"]["status"] == "blocked"
    assert "fold_1_static_atom_partial_coverage" in report["blockers"]
    assert "both_requires_three_valid_folds" in report["blockers"]


def test_variants_do_not_require_unselected_optional_producers(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    public_root, manifest_hash = _public_pack(tmp_path, source, game_ids)
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    for fold in (1, 2, 3):
        spec = json.loads((folds / f"fold-{fold}-spec.json").read_text())
        train = set(spec["train_game_ids"])
        form_path = strict_root / f"fold-{fold}" / "strict-prior-player-form.json"
        form = json.loads(form_path.read_text())
        for row in form["rows"]:
            if row["game_id"] in train:
                row["status"] = "unavailable"
                row["reason"] = "test_missing"
                row["future_player_form_logit"] = None
        form["rows_sha256"] = hashlib.sha256(_canonical(form["rows"])).hexdigest()
        form.pop("artifact_sha256")
        form["artifact_sha256"] = hashlib.sha256(_canonical(form)).hexdigest()
        _write_json(form_path, form)
    freeze_path, freeze_hash = _freeze_inputs(tmp_path, source_path, folds, strict_root)
    report = build_report(
        source_receipt_path=source_path,
        expected_source_receipt_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        trust_root_path=freeze_path,
        expected_trust_root_sha256=freeze_hash,
        source_root=source_root,
        folds_root=folds,
        evaluation_root=evaluation,
        public_pack_root=public_root,
        expected_manifest_sha256=manifest_hash,
        authority_path=tmp_path / "authority.json",
        expected_authority_sha256=hashlib.sha256((tmp_path / "authority.json").read_bytes()).hexdigest(),
        model_artifact_path=tmp_path / "model.json",
        strict_fold_root=strict_root,
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


def _write_current_receipt_fixture(tmp_path: Path) -> tuple[Path, list[str], dict[str, object]]:
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
    receipt_path = root / "current-rating-feature-ledger.receipt.json"
    _write_json(receipt_path, receipt)
    binding = {
        "receipt": {"bytes": receipt_path.stat().st_size, "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
        "artifact": {"bytes": artifact_path.stat().st_size, "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest()},
    }
    return root, game_ids, binding


def test_current_rating_row_mutation_is_rejected_by_receipt(tmp_path: Path) -> None:
    root, game_ids, binding = _write_current_receipt_fixture(tmp_path)
    frame = pd.read_parquet(root / "current-rating-feature-ledger.parquet")
    frame.loc[0, "base_team_logit"] = 99.0
    frame.to_parquet(root / "current-rating-feature-ledger.parquet")
    with pytest.raises(FourWayDraftScoreError, match="bytes changed from trust root"):
        _load_current(
            root,
            trust_binding=binding,
            train_ids=game_ids,
            cutoff_text="2025-12-31T00:00:00Z",
            require_receipt=True,
        )


def _write_scaling_receipt_fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
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
    receipt_path = root / "scaling-native-receipt.json"
    _write_json(receipt_path, receipt)
    binding = {
        "receipt": {"bytes": receipt_path.stat().st_size, "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest()},
        "artifact": {"bytes": artifact_path.stat().st_size, "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest()},
    }
    frame.loc[0, "forecast_scaling_index"] = 99.0
    frame.to_parquet(artifact_path)
    return root, binding


def test_scaling_row_mutation_is_rejected_by_receipt(tmp_path: Path) -> None:
    root, binding = _write_scaling_receipt_fixture(tmp_path)
    with pytest.raises(FourWayDraftScoreError, match="bytes changed from trust root"):
        _load_scaling(
            root,
            trust_binding=binding,
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


def test_future_source_uses_variant_accepted_census(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    _, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    model_path = evaluation / "future_player_form" / "model.json"
    model = json.loads(model_path.read_text())
    model["source"]["normalized_source_files"] = source["source_files"]
    authority = {
        "research_only": True,
        "public_player_rating": False,
        "public_team_rating": False,
        "public_probability": False,
        "promotion": False,
        "deployment": False,
        "recommendation": False,
        "odds": False,
        "expected_value": False,
        "betting": False,
    }
    model["authority"] = authority
    payload = model["variants"]["future_player_form"]
    payload["authority"] = authority
    payload["source"] = {
        "source_as_of": source["source_as_of"],
        "source_game_count": source["source_game_count"],
        "source_identity_sha256": source["source_identity_sha256"],
        "source_receipt_sha256": source["receipt_sha256"],
        "accepted_game_ids": game_ids,
    }
    _write_json(model_path, model)

    frame, _ = _load_future(model_path, 1, source=source)
    assert set(frame["game_id"]) == set(game_ids)

    model = json.loads(model_path.read_text())
    model["variants"]["future_player_form"]["source"]["accepted_game_ids"] = game_ids[:-1]
    _write_json(model_path, model)
    with pytest.raises(FourWayDraftScoreError, match="accepted census changed"):
        _load_future(model_path, 1, source=source)


def test_source_receipt_without_source_files_is_rejected_even_when_resealed(tmp_path: Path) -> None:
    source_path, _ = _source(tmp_path, ["g1", "g2"])
    source = json.loads(source_path.read_text())
    source.pop("source_files")
    source.pop("receipt_sha256")
    source["receipt_sha256"] = hashlib.sha256(_canonical(source)).hexdigest()
    file_hash = _write_json(source_path, source)
    with pytest.raises(FourWayDraftScoreError, match="file bindings are required"):
        _load_source(source_path, expected_file_sha256=file_hash)


def test_future_component_digest_is_required(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    _, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    model_path = evaluation / "future_player_form" / "model.json"
    model = json.loads(model_path.read_text())
    model["variants"]["future_player_form"]["folds"][0]["component_evidence"].pop("sha256")
    _write_json(model_path, model)
    with pytest.raises(FourWayDraftScoreError, match="component ledger hash"):
        _load_future(model_path, 1)


def test_current_artifact_and_receipt_co_mutation_is_rejected_by_freeze(tmp_path: Path) -> None:
    root, game_ids, binding = _write_current_receipt_fixture(tmp_path)
    artifact_path = root / "current-rating-feature-ledger.parquet"
    receipt_path = root / "current-rating-feature-ledger.receipt.json"
    frame = pd.read_parquet(artifact_path)
    frame.loc[0, "base_team_logit"] = 42.0
    frame.to_parquet(artifact_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["artifact"]["bytes"] = artifact_path.stat().st_size
    receipt["artifact"]["sha256"] = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    output = frame[["game_id", "date", "series_id", *CURRENT_FEATURES]].copy()
    receipt["ledger_rows_sha256"] = _current_artifact_digest(output, CURRENT_FEATURES)
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = hashlib.sha256(_canonical(receipt)).hexdigest()
    _write_json(receipt_path, receipt)
    with pytest.raises(FourWayDraftScoreError, match="bytes changed from trust root"):
        _load_current(
            root,
            trust_binding=binding,
            train_ids=game_ids,
            cutoff_text="2025-12-31T00:00:00Z",
            require_receipt=True,
        )


def test_fold_form_feature_at_cutoff_is_rejected(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    folds, _ = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
    form_path = strict_root / "fold-1" / "strict-prior-player-form.json"
    form = json.loads(form_path.read_text())
    validation = set(json.loads((folds / "fold-1-spec.json").read_text())["validation_game_ids"])
    for row in form["rows"]:
        if row["game_id"] in validation:
            row["fit_through"] = "2026-01-02T12:00:00Z"
    form["rows_sha256"] = hashlib.sha256(_canonical(form["rows"])).hexdigest()
    form.pop("artifact_sha256")
    form["artifact_sha256"] = hashlib.sha256(_canonical(form)).hexdigest()
    file_hash = _write_json(form_path, form)
    maps = _map_source(source_root, set(game_ids), source=source)
    with pytest.raises(FourWayDraftScoreError, match="validation fit crossed cutoff"):
        _verify_strict_prior_form_artifact(
            form_path,
            source,
            maps,
            expected_sha256=file_hash,
            fold=1,
            train_ids=["g1", "g2"],
            validation_ids=["g3", "g4", "g5", "g6"],
            cutoff_text="2026-01-02T12:00:00Z",
        )


def test_chronology_mutation_blocks_fold_evaluation(tmp_path: Path) -> None:
    game_ids = [f"g{i}" for i in range(1, 7)]
    source_path, source_root = _source(tmp_path, game_ids)
    source = json.loads(source_path.read_text())
    public_root, manifest_hash = _public_pack(tmp_path, source, game_ids)
    folds, evaluation = _fold_inputs(tmp_path, game_ids, source_root, source)
    strict_root = _strict_fold_inputs(tmp_path, game_ids, source, folds)
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
    freeze_path, freeze_hash = _freeze_inputs(tmp_path, source_path, folds, strict_root)
    with pytest.raises(FourWayDraftScoreError, match="fold contract changed: fit_window_end"):
        build_report(
            source_receipt_path=source_path,
            expected_source_receipt_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            trust_root_path=freeze_path,
            expected_trust_root_sha256=freeze_hash,
            source_root=source_root,
            folds_root=folds,
            evaluation_root=evaluation,
            public_pack_root=public_root,
            expected_manifest_sha256=manifest_hash,
            authority_path=tmp_path / "authority.json",
            expected_authority_sha256=hashlib.sha256((tmp_path / "authority.json").read_bytes()).hexdigest(),
            model_artifact_path=tmp_path / "model.json",
            strict_fold_root=strict_root,
        )


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
