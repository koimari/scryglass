"""Freeze the regularized terminal Draft Score v2 candidate for future tests.

The candidate is selected only from the source-frozen v2 development
validation slices.  This generator fits it on the earlier 90% of dependence
clusters and calibrates it on the final 10%, then emits exact model bytes and a
replay fixture.  All outputs remain development-only and authorize no
probability, recommendation, or bet.
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
    DEFAULT_SUMMARY,
    RIDGE_STRENGTH_ORDER,
    baseline_adjusted_logits,
    composition_logits,
    evaluate,
    fit_penalized,
)
from .development_snapshot import load_development_snapshot
from .model import TerminalDraft, TerminalModel, score_terminal_draft


MODEL_VERSION = "draft-terminal-neutral-dev-v2.0.0"
SELECTED_CANDIDATE_ID = "m0-role-additive"
SELECTED_RIDGE_STRENGTH = 0.02
DEFAULT_ARTIFACT = Path(
    "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v2.json"
)
DEFAULT_REPORT = Path(
    "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v2.report.json"
)
DEFAULT_FIXTURE = Path(
    "data/lol/v2/models/draft-terminal/terminal-neutral-development-v2.replay-fixture.json"
)
BASE_FIXTURE = Path(
    "data/lol/v2/models/draft-terminal/terminal-neutral-development-replay-fixture.json"
)


class DevelopmentArtifactV2Error(ValueError):
    """Raised when the v2 development artifact cannot be frozen exactly."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DevelopmentArtifactV2Error(f"{field} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DevelopmentArtifactV2Error(f"{field} is not finite")
    return result


def _atomic_write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise DevelopmentArtifactV2Error(f"refusing to overwrite artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise DevelopmentArtifactV2Error(f"refusing to overwrite artifact: {path}")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_bound_summary(root: Path) -> tuple[dict[str, Any], str]:
    path = root / DEFAULT_SUMMARY
    raw = path.read_bytes()
    summary = json.loads(raw)
    if not isinstance(summary, dict):
        raise DevelopmentArtifactV2Error("v2 development summary must be an object")
    report = evaluate(root)
    report_raw = (
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    if summary.get("run_output_sha256") != _sha256(report_raw):
        raise DevelopmentArtifactV2Error(
            "v2 development summary does not bind the current frozen evaluation"
        )
    selected = summary.get("development_candidate_for_future_freeze")
    if not isinstance(selected, Mapping):
        raise DevelopmentArtifactV2Error("v2 development candidate is missing")
    if (
        selected.get("candidate_id") != SELECTED_CANDIDATE_ID
        or selected.get("ridge_strength") != SELECTED_RIDGE_STRENGTH
        or SELECTED_RIDGE_STRENGTH not in RIDGE_STRENGTH_ORDER
    ):
        raise DevelopmentArtifactV2Error(
            "v2 selected candidate differs from the frozen artifact generator"
        )
    return summary, _sha256(raw)


def build_development_artifact(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    summary, summary_raw_sha256 = _load_bound_summary(root)
    rows, source_snapshot = load_development_snapshot(root)
    cluster_latest: dict[str, Any] = {}
    for row in rows:
        cluster_latest[row.dependence_cluster_id] = max(
            cluster_latest.get(row.dependence_cluster_id, row.date), row.date
        )
    cluster_order = [
        cluster_id
        for cluster_id, _ in sorted(
            cluster_latest.items(), key=lambda item: (item[1], item[0])
        )
    ]
    calibration_cluster_count = max(20, len(cluster_order) // 10)
    train_clusters = set(cluster_order[:-calibration_cluster_count])
    calibration_clusters = set(cluster_order[-calibration_cluster_count:])
    train = [row for row in rows if row.dependence_cluster_id in train_clusters]
    calibration = [
        row for row in rows if row.dependence_cluster_id in calibration_clusters
    ]
    if not train or not calibration:
        raise DevelopmentArtifactV2Error("v2 train or calibration slice is empty")
    if max(row.date for row in train) >= min(row.date for row in calibration):
        # A dependence cluster can span timestamps, but cluster ordering by its
        # latest map guarantees no selected train cluster ends after the first
        # selected calibration cluster begins only when this check passes.
        raise DevelopmentArtifactV2Error(
            "v2 calibration boundary is not strictly chronological"
        )
    baseline_logits = pre_event_team_elo_logits(rows)
    fit = fit_penalized(
        train,
        SELECTED_CANDIDATE_ID,
        SELECTED_RIDGE_STRENGTH,
        baseline_logits,
    )
    calibration_logits = composition_logits(calibration, fit)
    calibration_labels = [row.label_a for row in calibration]
    transform_reports: list[dict[str, Any]] = []
    choices: list[tuple[float, int, str, float]] = []
    for method in CALIBRATION_ORDER:
        parameter, loss = _fit_calibration(
            calibration_logits, calibration_labels, method
        )
        choices.append((float(loss), CALIBRATION_ORDER.index(method), method, parameter))
        transform_reports.append(
            {
                "method": method,
                "parameter": parameter,
                "calibration_log_loss": loss,
            }
        )
    _, _, selected_transform, selected_parameter = min(choices)
    calibration_slope = (
        1.0 / selected_parameter
        if selected_transform == "symmetric_temperature"
        else selected_parameter
    )
    for item in transform_reports:
        item["selected"] = item["method"] == selected_transform

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
            raise DevelopmentArtifactV2Error(
                f"unrecognized v2 terminal feature: {feature}"
            )

    train_design = _design(train, fit.candidate_id, fit.vocabulary).tocsr()
    calibration_design = _design(
        calibration, fit.candidate_id, fit.vocabulary
    ).tocsr()
    adjusted_logits = baseline_adjusted_logits(train, fit, baseline_logits)
    train_probabilities = _probabilities(adjusted_logits, 1.0)
    weights = train_probabilities * (1.0 - train_probabilities)
    diagonal_information = np.asarray(
        train_design.multiply(train_design).T @ weights, dtype=float
    ).reshape(-1)
    diagonal_information += len(train) * SELECTED_RIDGE_STRENGTH
    coefficient_variance = 1.0 / np.maximum(diagonal_information, 1e-12)
    calibration_prediction_variance = np.asarray(
        calibration_design.multiply(calibration_design) @ coefficient_variance,
        dtype=float,
    ).reshape(-1)
    uncertainty_logit_sd = float(
        calibration_slope
        * math.sqrt(max(float(np.mean(calibration_prediction_variance)), 1e-12))
    )
    model_as_of = max(row.date for row in rows).isoformat().replace("+00:00", "Z")
    artifact = {
        "ally_synergy_logit": dict(sorted(ally_synergy_logit.items())),
        "calibration_intercept": 0.0,
        "calibration_slope": _strict_number(
            calibration_slope, "calibration_slope"
        ),
        "champion_role_logit": dict(sorted(champion_role_logit.items())),
        "counter_logit": dict(sorted(counter_logit.items())),
        "intercept": 0.0,
        "model_as_of": model_as_of,
        "model_version": MODEL_VERSION,
        "uncertainty_logit_sd": _strict_number(
            uncertainty_logit_sd, "uncertainty_logit_sd"
        ),
    }
    metadata = {
        "schema_version": "scryglass:draft-terminal-development-artifact-report:v2",
        "status": "development_only_frozen_for_future_evaluation",
        "production_eligible": False,
        "public_probability_authorized": False,
        "candidate_id": SELECTED_CANDIDATE_ID,
        "ridge_strength": SELECTED_RIDGE_STRENGTH,
        "objective": summary["objective"],
        "model_version": MODEL_VERSION,
        "model_as_of": model_as_of,
        "train_rows": len(train),
        "train_dependence_clusters": len(train_clusters),
        "calibration_rows": len(calibration),
        "calibration_dependence_clusters": len(calibration_clusters),
        "train_start": min(row.date for row in train).isoformat(),
        "train_end": max(row.date for row in train).isoformat(),
        "calibration_start": min(row.date for row in calibration).isoformat(),
        "calibration_end": max(row.date for row in calibration).isoformat(),
        "feature_count": len(fit.vocabulary),
        "coefficient_l2": float(np.linalg.norm(fit.beta)),
        "baseline_coefficient": fit.baseline_coefficient,
        "baseline_is_not_serialized_in_served_artifact": True,
        "optimizer_iterations": fit.optimizer_iterations,
        "optimizer_gradient_max_abs": fit.optimizer_gradient_max_abs,
        "calibration_transforms": transform_reports,
        "selected_calibration_transform": selected_transform,
        "calibration_slope": calibration_slope,
        "uncertainty_method": (
            "diagonal_laplace_development_diagnostic_without_covariance; "
            "not a reliability interval"
        ),
        "source_snapshot": source_snapshot,
        "development_evaluation_summary_locator": str(DEFAULT_SUMMARY),
        "development_evaluation_summary_raw_sha256": summary_raw_sha256,
        "claim_ceiling": {
            "development_diagnostic": True,
            "independent_validation": False,
            "probability": False,
            "recommendation": False,
            "betting": False,
        },
    }
    return artifact, metadata


def _build_replay_fixture(
    root: Path, artifact_raw: bytes, artifact_sha256: str
) -> dict[str, Any]:
    base = json.loads((root / BASE_FIXTURE).read_text(encoding="utf-8"))
    draft_mapping = base.get("draft")
    if not isinstance(draft_mapping, Mapping):
        raise DevelopmentArtifactV2Error("base replay fixture draft is missing")
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
        "schema_version": "scryglass:draft-terminal-development-replay:v2",
        "status": "development_only",
        "draft": dict(draft_mapping),
        "model_artifact_locator": str(DEFAULT_ARTIFACT),
        "model_artifact_sha256": artifact_sha256,
        "expected_development": score_terminal_draft(draft, model, development=True),
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
    fixture = _build_replay_fixture(root, artifact_raw, artifact_sha256)
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
    except (OSError, ValueError, DevelopmentArtifactV2Error) as exc:
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

