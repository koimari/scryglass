"""Freeze the incremental-context terminal Draft Score candidate for the future.

This is the first terminal artifact whose candidate was selected on the
identified development target: incremental prediction from adding draft terms
to pre-event context.  Its neutral output remains an equal-strength index, not
a directly outcome-calibrated win probability.  The artifact is prospective
test material only and grants no serving or betting authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .development_evaluation import (
    CALIBRATION_ORDER,
    _design,
    _fit_calibration,
    _probabilities,
    pre_event_team_elo_logits,
)
from .development_evaluation_v2 import (
    RIDGE_STRENGTH_ORDER,
    baseline_adjusted_logits,
    fit_penalized,
)
from .development_evaluation_v3 import (
    DEFAULT_SUMMARY,
    baseline_only_logits,
    evaluate,
    fit_baseline_only,
)
from .development_snapshot import load_development_snapshot
from .model import TerminalDraft, TerminalModel, score_terminal_draft


MODEL_VERSION = "draft-terminal-neutral-dev-v3.0.0"
SELECTED_CANDIDATE_ID = "m0-role-additive"
SELECTED_RIDGE_STRENGTH = 0.05
DEFAULT_ARTIFACT = Path(
    "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v3.json"
)
DEFAULT_REPORT = Path(
    "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v3.report.json"
)
DEFAULT_FIXTURE = Path(
    "data/lol/v2/models/draft-terminal/terminal-neutral-development-v3.replay-fixture.json"
)
BASE_FIXTURE = Path(
    "data/lol/v2/models/draft-terminal/terminal-neutral-development-replay-fixture.json"
)


class DevelopmentArtifactV3Error(ValueError):
    """Raised when v3 prospective candidate material is inconsistent."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevelopmentArtifactV3Error(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DevelopmentArtifactV3Error(f"{field} is not finite")
    return result


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DevelopmentArtifactV3Error(f"refusing to overwrite v3 artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise DevelopmentArtifactV3Error(
                f"refusing to overwrite v3 artifact: {path}"
            )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _bound_summary(root: Path) -> tuple[dict[str, Any], str]:
    raw = (root / DEFAULT_SUMMARY).read_bytes()
    summary = json.loads(raw)
    if not isinstance(summary, dict):
        raise DevelopmentArtifactV3Error("v3 development summary is malformed")
    report = evaluate(root)
    report_raw = (
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    if summary.get("run_output_sha256") != _sha256(report_raw):
        raise DevelopmentArtifactV3Error(
            "v3 development summary does not bind the current frozen evaluation"
        )
    candidate = summary.get("development_candidate_for_future_freeze")
    if not isinstance(candidate, Mapping):
        raise DevelopmentArtifactV3Error("v3 development selected no future candidate")
    if (
        candidate.get("candidate_id") != SELECTED_CANDIDATE_ID
        or candidate.get("ridge_strength") != SELECTED_RIDGE_STRENGTH
        or candidate.get("all_validation_folds_nonharmful") is not True
        or SELECTED_RIDGE_STRENGTH not in RIDGE_STRENGTH_ORDER
    ):
        raise DevelopmentArtifactV3Error(
            "v3 generator differs from the incremental validation selection"
        )
    return summary, _sha256(raw)


def _calibration(
    logits: np.ndarray, labels: list[int]
) -> tuple[str, float, list[dict[str, Any]]]:
    choices: list[tuple[float, int, str, float]] = []
    reports: list[dict[str, Any]] = []
    for method in CALIBRATION_ORDER:
        parameter, loss = _fit_calibration(logits, labels, method)
        choices.append((float(loss), CALIBRATION_ORDER.index(method), method, parameter))
        reports.append(
            {
                "method": method,
                "parameter": parameter,
                "calibration_log_loss": loss,
            }
        )
    _, _, selected_method, selected_parameter = min(choices)
    scale = (
        1.0 / selected_parameter
        if selected_method == "symmetric_temperature"
        else selected_parameter
    )
    for report in reports:
        report["selected"] = report["method"] == selected_method
    return selected_method, float(scale), reports


def build_development_artifact(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, summary_raw_sha256 = _bound_summary(root)
    rows, source_snapshot = load_development_snapshot(root)
    cluster_latest: dict[str, Any] = {}
    for row in rows:
        cluster_latest[row.dependence_cluster_id] = max(
            cluster_latest.get(row.dependence_cluster_id, row.date), row.date
        )
    order = [
        cluster_id
        for cluster_id, _ in sorted(
            cluster_latest.items(), key=lambda item: (item[1], item[0])
        )
    ]
    calibration_count = max(20, len(order) // 10)
    train_ids = set(order[:-calibration_count])
    calibration_ids = set(order[-calibration_count:])
    train = [row for row in rows if row.dependence_cluster_id in train_ids]
    calibration = [
        row for row in rows if row.dependence_cluster_id in calibration_ids
    ]
    if not train or not calibration:
        raise DevelopmentArtifactV3Error("v3 train or calibration slice is empty")
    if max(row.date for row in train) >= min(row.date for row in calibration):
        raise DevelopmentArtifactV3Error(
            "v3 calibration boundary is not strictly chronological"
        )
    baseline_logits = pre_event_team_elo_logits(rows)
    fit = fit_penalized(
        train,
        SELECTED_CANDIDATE_ID,
        SELECTED_RIDGE_STRENGTH,
        baseline_logits,
    )
    baseline_coefficient = fit_baseline_only(train, baseline_logits)
    labels = [row.label_a for row in calibration]
    candidate_method, candidate_scale, candidate_transforms = _calibration(
        baseline_adjusted_logits(calibration, fit, baseline_logits), labels
    )
    baseline_method, baseline_scale, baseline_transforms = _calibration(
        baseline_only_logits(calibration, baseline_coefficient, baseline_logits),
        labels,
    )

    champion_role_logit: dict[str, float] = {}
    ally_synergy_logit: dict[str, float] = {}
    counter_logit: dict[str, float] = {}
    for feature, coefficient in zip(fit.vocabulary, fit.beta):
        value = _strict_number(coefficient, f"coefficient.{feature}")
        if abs(value) <= 1e-15:
            continue
        if feature.startswith("main|"):
            _, role, champion = feature.split("|", 2)
            champion_role_logit[f"{role}|{champion}"] = value
        elif feature.startswith("ally|"):
            _, first, second = feature.split("|", 2)
            ally_synergy_logit[f"{first}|{second}"] = value
        elif feature.startswith("counter|"):
            _, role, first, second = feature.split("|", 3)
            counter_logit[f"{role}|{first}|{second}"] = value
        else:
            raise DevelopmentArtifactV3Error(f"unknown terminal feature: {feature}")

    train_design = _design(train, fit.candidate_id, fit.vocabulary).tocsr()
    calibration_design = _design(
        calibration, fit.candidate_id, fit.vocabulary
    ).tocsr()
    train_probabilities = _probabilities(
        baseline_adjusted_logits(train, fit, baseline_logits), 1.0
    )
    weights = train_probabilities * (1.0 - train_probabilities)
    diagonal_information = np.asarray(
        train_design.multiply(train_design).T @ weights, dtype=float
    ).reshape(-1)
    diagonal_information += len(train) * SELECTED_RIDGE_STRENGTH
    coefficient_variance = 1.0 / np.maximum(diagonal_information, 1e-12)
    prediction_variance = np.asarray(
        calibration_design.multiply(calibration_design) @ coefficient_variance,
        dtype=float,
    ).reshape(-1)
    uncertainty_logit_sd = float(
        candidate_scale * math.sqrt(max(float(np.mean(prediction_variance)), 1e-12))
    )
    model_as_of = max(row.date for row in rows).isoformat().replace("+00:00", "Z")
    artifact = {
        "ally_synergy_logit": dict(sorted(ally_synergy_logit.items())),
        "calibration_intercept": 0.0,
        "calibration_slope": _strict_number(candidate_scale, "calibration_slope"),
        "champion_role_logit": dict(sorted(champion_role_logit.items())),
        "counter_logit": dict(sorted(counter_logit.items())),
        "intercept": 0.0,
        "model_as_of": model_as_of,
        "model_version": MODEL_VERSION,
        "uncertainty_logit_sd": _strict_number(
            uncertainty_logit_sd, "uncertainty_logit_sd"
        ),
    }
    candidate_calibration_metrics = {
        "brier_score": float(
            np.mean(
                (
                    np.asarray(labels, dtype=float)
                    - _probabilities(
                        baseline_adjusted_logits(calibration, fit, baseline_logits),
                        candidate_scale,
                    )
                )
                ** 2
            )
        ),
        "log_loss": float(
            -np.mean(
                np.asarray(labels, dtype=float)
                * np.log(
                    _probabilities(
                        baseline_adjusted_logits(calibration, fit, baseline_logits),
                        candidate_scale,
                    )
                )
                + (1.0 - np.asarray(labels, dtype=float))
                * np.log1p(
                    -_probabilities(
                        baseline_adjusted_logits(calibration, fit, baseline_logits),
                        candidate_scale,
                    )
                )
            )
        ),
    }
    baseline_calibration_probabilities = _probabilities(
        baseline_only_logits(calibration, baseline_coefficient, baseline_logits),
        baseline_scale,
    )
    baseline_calibration_metrics = {
        "brier_score": float(
            np.mean(
                (np.asarray(labels, dtype=float) - baseline_calibration_probabilities)
                ** 2
            )
        ),
        "log_loss": float(
            -np.mean(
                np.asarray(labels, dtype=float)
                * np.log(baseline_calibration_probabilities)
                + (1.0 - np.asarray(labels, dtype=float))
                * np.log1p(-baseline_calibration_probabilities)
            )
        ),
    }
    report = {
        "schema_version": "scryglass:draft-terminal-development-artifact-report:v3",
        "status": "development_only_frozen_for_prospective_incremental_evaluation",
        "production_eligible": False,
        "public_probability_authorized": False,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "ridge_strength": SELECTED_RIDGE_STRENGTH,
        "model_version": MODEL_VERSION,
        "model_as_of": model_as_of,
        "train_rows": len(train),
        "train_dependence_clusters": len(train_ids),
        "calibration_rows": len(calibration),
        "calibration_dependence_clusters": len(calibration_ids),
        "train_start": min(row.date for row in train).isoformat(),
        "train_end": max(row.date for row in train).isoformat(),
        "calibration_start": min(row.date for row in calibration).isoformat(),
        "calibration_end": max(row.date for row in calibration).isoformat(),
        "feature_count": len(fit.vocabulary),
        "coefficient_l2": float(np.linalg.norm(fit.beta)),
        "candidate_baseline_coefficient": fit.baseline_coefficient,
        "baseline_only_coefficient": baseline_coefficient,
        "baseline_is_not_serialized_in_neutral_artifact": True,
        "candidate_calibration_target": "context_plus_draft_observed_outcome",
        "neutral_equal_strength_index_directly_outcome_calibrated": False,
        "candidate_calibration_transform": candidate_method,
        "candidate_calibration_slope": candidate_scale,
        "candidate_calibration_transforms": candidate_transforms,
        "baseline_calibration_transform": baseline_method,
        "baseline_calibration_slope": baseline_scale,
        "baseline_calibration_transforms": baseline_transforms,
        "candidate_calibration_metrics": candidate_calibration_metrics,
        "baseline_calibration_metrics": baseline_calibration_metrics,
        "calibration_incremental_delta": {
            "brier_score": candidate_calibration_metrics["brier_score"]
            - baseline_calibration_metrics["brier_score"],
            "log_loss": candidate_calibration_metrics["log_loss"]
            - baseline_calibration_metrics["log_loss"],
            "selection_or_validation_use": False,
        },
        "optimizer_iterations": fit.optimizer_iterations,
        "optimizer_gradient_max_abs": fit.optimizer_gradient_max_abs,
        "uncertainty_method": (
            "diagonal_laplace_development_diagnostic_without_covariance; "
            "not a reliability interval"
        ),
        "source_snapshot": source_snapshot,
        "development_evaluation_summary_locator": str(DEFAULT_SUMMARY),
        "development_evaluation_summary_raw_sha256": summary_raw_sha256,
        "supersedes_for_future_protocol": {
            "model_version": "draft-terminal-neutral-dev-v2.0.0",
            "reason": "v2 selected against observed outcomes without the required contextual comparator",
            "v2_authority_rehabilitated": False,
        },
        "claim_ceiling": {
            "development_diagnostic": True,
            "equal_strength_composition_index": True,
            "outcome_calibrated_neutral_probability": False,
            "independent_validation": False,
            "recommendation": False,
            "betting": False,
        },
    }
    return artifact, report


def _replay_fixture(
    root: Path, artifact_raw: bytes, artifact_sha256: str
) -> dict[str, Any]:
    base = json.loads((root / BASE_FIXTURE).read_text(encoding="utf-8"))
    draft_mapping = base.get("draft")
    if not isinstance(draft_mapping, Mapping):
        raise DevelopmentArtifactV3Error("base replay fixture draft is missing")
    draft = TerminalDraft.from_sides(
        draft_mapping["side_a"],
        draft_mapping["side_b"],
        event_start=draft_mapping["event_start"],
        source_available_at=draft_mapping["source_available_at"],
        source_record_id=draft_mapping["source_record_id"],
        source_payload_sha256=draft_mapping["source_payload_sha256"],
        source_rights_status=draft_mapping["source_rights_status"],
        mode=draft_mapping.get("mode", "neutral"),
        actions=draft_mapping.get("actions"),
        final_assignments=draft_mapping.get("final_assignments"),
    )
    model = TerminalModel.from_artifact_bytes(
        artifact_raw, expected_artifact_sha256=artifact_sha256
    )
    return {
        "schema_version": "scryglass:draft-terminal-development-replay:v3",
        "status": "development_only_equal_strength_index",
        "draft": dict(draft_mapping),
        "model_artifact_locator": str(DEFAULT_ARTIFACT),
        "model_artifact_sha256": artifact_sha256,
        "expected_development": score_terminal_draft(draft, model, development=True),
        "probability_semantics": "equal_strength_composition_index_not_directly_outcome_calibrated",
        "claim_ceiling": {
            "development_diagnostic": True,
            "prediction": False,
            "recommendation": False,
            "betting": False,
        },
    }


def write_development_artifact(root: Path) -> tuple[Path, Path, Path]:
    artifact, report = build_development_artifact(root)
    artifact_raw = json.dumps(
        artifact, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    artifact_sha256 = _sha256(artifact_raw)
    report["artifact_locator"] = str(DEFAULT_ARTIFACT)
    report["artifact_raw_sha256"] = artifact_sha256
    fixture = _replay_fixture(root, artifact_raw, artifact_sha256)
    artifact_path = root / DEFAULT_ARTIFACT
    report_path = root / DEFAULT_REPORT
    fixture_path = root / DEFAULT_FIXTURE
    _atomic_write_new(artifact_path, artifact_raw)
    _atomic_write_new(
        report_path,
        (json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n").encode(
            "ascii"
        ),
    )
    _atomic_write_new(
        fixture_path,
        (
            json.dumps(fixture, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        ).encode("ascii"),
    )
    return artifact_path, report_path, fixture_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    try:
        artifact, report, fixture = write_development_artifact(args.root)
    except (OSError, ValueError, DevelopmentArtifactV3Error) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "artifact": str(artifact),
                "artifact_raw_sha256": _sha256(artifact.read_bytes()),
                "report": str(report),
                "fixture": str(fixture),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

