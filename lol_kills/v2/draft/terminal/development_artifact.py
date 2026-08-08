"""Build the bounded neutral Draft Score development artifact.

This generator is intentionally separate from promotion.  It fits only the
registered role-additive neutral candidate, uses a time-separated calibration
slice, and emits the exact coefficient shape consumed by the Python and
TypeScript replay implementations.  The resulting bytes are still
development-only until an independent L2 record authorizes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse import csr_matrix, hstack

from .development_evaluation import (
    BASELINE_CONFIG,
    CANDIDATE_ORDER,
    _design,
    _fit_calibration,
    _parse_time,
    _probabilities,
    _canonical_json,
    _sha256_bytes,
    baseline_adjusted_logits,
    composition_logits,
    fit_baseline_adjusted,
    load_snapshot,
    pre_event_team_elo_logits,
)


MODEL_VERSION = "draft-terminal-neutral-dev-v1.1.0"
MODEL_AS_OF = "2026-06-30T23:59:59Z"
CALIBRATION_START = "2026-05-01T00:00:00Z"
CANDIDATE_ID = "m0-role-additive"


def _strict_number(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise FloatingPointError(f"{field} is not finite")
    return number


def _artifact_mapping(
    train_rows: list[Any],
    calibration_rows: list[Any],
    baseline_logits: dict[str, float],
) -> tuple[dict[str, Any], dict[str, Any]]:
    fit = fit_baseline_adjusted(train_rows, CANDIDATE_ID, baseline_logits)
    vocabulary = fit.vocabulary
    beta = fit.beta
    train_design = _design(train_rows, CANDIDATE_ID, vocabulary)
    calibration_design = _design(calibration_rows, CANDIDATE_ID, vocabulary)
    train_logits = composition_logits(train_rows, fit)
    calibration_logits = composition_logits(calibration_rows, fit)
    calibration_labels = [row.label_a for row in calibration_rows]
    scale, calibration_loss = _fit_calibration(calibration_logits, calibration_labels, "symmetrized_platt")

    # Diagonal Laplace uncertainty for the fitted coefficients.  This is a
    # conservative development diagnostic, not an independent reliability
    # interval: covariance terms are deliberately not presented as validated.
    train_adjusted_logits = baseline_adjusted_logits(train_rows, fit, baseline_logits)
    train_probabilities = _probabilities(train_adjusted_logits, 1.0)
    weights = train_probabilities * (1.0 - train_probabilities)
    nuisance = np.asarray([baseline_logits[row.game_id] for row in train_rows], dtype=float)
    full_design = hstack([train_design, csr_matrix(nuisance.reshape(-1, 1))], format="csr")
    diagonal_information = np.asarray(full_design.multiply(full_design).T @ weights, dtype=float).reshape(-1) + 1.0
    coefficient_variance = 1.0 / np.maximum(diagonal_information, 1e-12)
    prediction_variance = np.asarray(
        calibration_design.multiply(calibration_design) @ coefficient_variance[: len(vocabulary)],
        dtype=float,
    )
    uncertainty_logit_sd = float(scale * math.sqrt(max(float(np.mean(prediction_variance)), 1e-12)))

    champion_role_logit: dict[str, float] = {}
    for feature, coefficient in zip(vocabulary, beta):
        if not feature.startswith("main|"):
            raise ValueError(f"unexpected feature in {CANDIDATE_ID}: {feature}")
        _, role, champion = feature.split("|", 2)
        value = _strict_number(coefficient, f"champion_role_logit.{role}|{champion}")
        if abs(value) > 1e-15:
            champion_role_logit[f"{role}|{champion}"] = value

    artifact = {
        "ally_synergy_logit": {},
        "calibration_intercept": 0.0,
        "calibration_slope": _strict_number(scale, "calibration_slope"),
        "champion_role_logit": dict(sorted(champion_role_logit.items())),
        "counter_logit": {},
        "intercept": 0.0,
        "model_as_of": MODEL_AS_OF,
        "model_version": MODEL_VERSION,
        "uncertainty_logit_sd": _strict_number(uncertainty_logit_sd, "uncertainty_logit_sd"),
    }
    metadata = {
        "candidate_id": CANDIDATE_ID,
        "candidate_order": list(CANDIDATE_ORDER),
        "model_version": MODEL_VERSION,
        "model_as_of": MODEL_AS_OF,
        "calibration_start": CALIBRATION_START,
        "train_rows": len(train_rows),
        "calibration_rows": len(calibration_rows),
        "feature_count": len(vocabulary),
        "baseline_feature_count": 1,
        "baseline_feature": "pre_event_team_elo_logit",
        "baseline_coefficient": _strict_number(fit.baseline_coefficient, "baseline_coefficient"),
        "baseline_adjustment": {
            **BASELINE_CONFIG,
            "config_sha256": _sha256_bytes(_canonical_json(BASELINE_CONFIG)),
            "served_baseline_logit": 0.0,
            "baseline_is_not_serialized_in_served_artifact": True,
        },
        "calibration_method": "symmetrized_platt",
        "calibration_target": "equalized_draft_logit",
        "calibration_parameter": _strict_number(scale, "calibration_parameter"),
        "calibration_log_loss": _strict_number(calibration_loss, "calibration_log_loss"),
        "uncertainty_method": "diagonal_laplace_development_diagnostic",
        "production_eligible": False,
        "public_probability_authorized": False,
    }
    return artifact, metadata


def build_development_artifact(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    rows, source_hashes = load_snapshot(root)
    as_of = _parse_time(MODEL_AS_OF)
    calibration_start = _parse_time(CALIBRATION_START)
    historical_rows = [row for row in rows if row.date <= as_of]
    baseline_logits = pre_event_team_elo_logits(historical_rows)
    train_rows = [row for row in historical_rows if row.date < calibration_start]
    calibration_rows = [row for row in historical_rows if calibration_start <= row.date <= as_of]
    if len(train_rows) < 100 or len(calibration_rows) < 100:
        raise ValueError("development artifact requires non-trivial time-separated train and calibration slices")
    artifact, metadata = _artifact_mapping(train_rows, calibration_rows, baseline_logits)
    metadata["source_snapshot"] = source_hashes
    metadata["source_date_range"] = {
        "train_start": min(row.date for row in train_rows).isoformat(),
        "train_end": max(row.date for row in train_rows).isoformat(),
        "calibration_start": min(row.date for row in calibration_rows).isoformat(),
        "calibration_end": max(row.date for row in calibration_rows).isoformat(),
    }
    return artifact, metadata


def write_development_artifacts(root: Path) -> tuple[Path, Path]:
    artifact, metadata = build_development_artifact(root)
    artifact_path = root / "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v1.json"
    report_path = root / "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-report.json"
    artifact_bytes = json.dumps(artifact, sort_keys=True, separators=(",", ":")).encode("utf-8")
    artifact_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    metadata["artifact_sha256"] = artifact_sha256
    artifact_path.write_bytes(artifact_bytes)
    report_path.write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    artifact_raw = artifact_bytes
    from .model import TerminalDraft, TerminalModel, score_terminal_draft

    fixture_path = root / "data/lol/v2/models/draft-terminal/terminal-neutral-development-replay-fixture.json"
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    fixture_draft = fixture["draft"]
    replay_draft = TerminalDraft.from_sides(
        fixture_draft["side_a"],
        fixture_draft["side_b"],
        event_start=fixture_draft["event_start"],
        source_available_at=fixture_draft["source_available_at"],
        source_record_id=fixture_draft["source_record_id"],
        source_payload_sha256=fixture_draft["source_payload_sha256"],
        source_rights_status=fixture_draft["source_rights_status"],
        mode=fixture_draft["mode"],
        actions=fixture_draft.get("actions"),
        final_assignments=fixture_draft.get("final_assignments"),
    )
    model = TerminalModel.from_artifact_bytes(artifact_raw, expected_artifact_sha256=artifact_sha256)
    fixture["model_artifact_sha256"] = artifact_sha256
    fixture["expected_development"] = score_terminal_draft(replay_draft, model, development=True)
    fixture_path.write_text(json.dumps(fixture, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    registry_path = root / "data/lol/v2/models/draft-terminal/draft-terminal-candidate-registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for candidate in registry.get("candidates", ()):
        if candidate.get("candidate_id") == CANDIDATE_ID:
            candidate["artifact_sha256"] = artifact_sha256
            candidate["artifact_status"] = "development_only_not_promoted"
    registry_path.write_text(json.dumps(registry, sort_keys=False, indent=2) + "\n", encoding="utf-8")

    from .development_evaluation import evaluate

    runner_report = evaluate(root)
    runner_serialized = json.dumps(runner_report, sort_keys=True, indent=2) + "\n"
    selected_tests = []
    for fold in runner_report["folds"]:
        selected = next(item for item in fold["candidates"] if item["selected_for_outer_test"])
        selected_tests.append(
            {
                "fold_id": fold["fold_id"],
                "candidate_id": selected["candidate_id"],
                "transform": fold["selection"]["calibration_transform"],
                "validation_log_loss": selected["validation"]["log_loss"],
                "validation_brier_score": selected["validation"]["brier_score"],
                "log_loss": selected["locked_outer_test"]["log_loss"],
                "brier_score": selected["locked_outer_test"]["brier_score"],
                "baseline_log_loss": selected["baseline_locked_outer_test"]["log_loss"],
                "baseline_brier_score": selected["baseline_locked_outer_test"]["brier_score"],
                "baseline_adjusted_log_loss": selected["baseline_adjusted_locked_outer_test"]["log_loss"],
                "baseline_adjusted_brier_score": selected["baseline_adjusted_locked_outer_test"]["brier_score"],
            }
        )
    summary = {
        "artifact_id": "draft-terminal-development-evaluation-summary-v1",
        "schema_version": runner_report["schema_version"],
        "status": runner_report["status"],
        "production_eligible": False,
        "public_probability_authorized": False,
        "run_command": "python3 -W error -m lol_kills.v2.draft.terminal.development_evaluation",
        "run_output_sha256": hashlib.sha256(runner_serialized.encode("utf-8")).hexdigest(),
        "source_snapshot": runner_report["source_snapshot"],
        "baseline_adjustment": runner_report["baseline_adjustment"],
        "population": runner_report["population"],
        "fold_policy": {
            "outer_folds": len(runner_report["folds"]),
            "series_grouped": runner_report["split_policy"]["series_grouped"],
            "dependence_clustered": runner_report["split_policy"]["dependence_clustered"],
            "chronological": runner_report["split_policy"]["chronological"],
            "calibration_fit_inside_fold": True,
            "outer_test_opened_for_search": runner_report["split_policy"]["candidate_search_opened_on_outer_test"],
            "candidate_selection_on_validation_only": runner_report["split_policy"]["candidate_selection_on_validation_only"],
            "outer_test_scored_only_for_selected_candidate": runner_report["split_policy"]["outer_test_scored_only_for_selected_candidate"],
            "feature_support_minimum": 10,
            "participant_cluster_status": "team identifiers available; player participant identifiers unavailable",
            "series_identity_status": runner_report["split_policy"]["series_identity_status"],
        },
        "candidate_order": runner_report["candidate_order"],
        "calibration_order": runner_report["calibration_order"],
        "fold_locked_selected_test": selected_tests,
        "selection": {
            "status": "development_only_selected",
            "winner_candidate_id": selected_tests[0]["candidate_id"],
            "winner_transform": selected_tests[0]["transform"],
            "reason": "the registered candidate was selected for the development artifact after baseline adjustment; this is not an independent L2 reliability or promotion record",
        },
        "holdouts": runner_report["holdouts"],
        "grid_promotion_gate": {
            "status": "not_passed",
            "baseline_source": "OE",
            "candidate_source": "GRID",
            "primary_source_for_cohort": "OE",
            "public_reproducibility_benchmark": "OE",
            "reason": "no authorized complete hash-verified GRID Draft Score cohort has passed the gate",
        },
        "coefficient_artifact_status": "development_fixture_refit_with_baseline_adjustment_not_promoted",
    }
    summary_path = root / "data/lol/v2/models/draft-terminal/development-evaluation-summary.json"
    summary_path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return artifact_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--metadata", action="store_true", help="print generator metadata after the artifact")
    parser.add_argument("--write", action="store_true", help="write the checked-in development artifact and report")
    args = parser.parse_args()
    if args.write:
        artifact_path, report_path = write_development_artifacts(args.root)
        print(json.dumps({"artifact_path": str(artifact_path), "report_path": str(report_path)}, sort_keys=True))
        return
    artifact, metadata = build_development_artifact(args.root)
    print(json.dumps(artifact, sort_keys=True, separators=(",", ":")))
    if args.metadata:
        print(json.dumps(metadata, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
