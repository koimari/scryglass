"""Development evaluation v4: do LCC atom features add draft value?

Research extension (R-22), not a frozen candidate: adds two atom-aware
candidates to the terminal-Draft development harness and compares them, in
the same chronological folds, against the existing role-additive candidate and
against the pre-event team-strength baseline.

    m3-role-additive-atom-presence   m0 features + per-role atom family presence
    m4-atom-presence-only            per-role atom family presence only

The per-role atom features are champion-mechanic presence flags (from the
pinned LCC atom bridge) differenced blue minus red per role.  m4 is the
zero-play transfer probe: a never-seen champion still enters the design
through its atom profile, so its composition value is nonzero *as a
mechanistic prior* -- never as an empirical residual or an outcome claim.

All fits use the frozen clustered cohort and the v2 normalized ridge
objective; candidate and ridge choices remain adaptive development
diagnostics.  Nothing here is promotion, calibration, or betting authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from lol_kills.v2.champions.atoms.consume import AtomBridge
from lol_kills.v2.champions.atoms.schema import BRIDGE_SCHEMA_ID, CHAMPION_ATOM_FAMILIES
from lol_kills.v2.draft.terminal.development_evaluation import (
    CALIBRATION_ORDER,
    DraftRow,
    _brier,
    _cluster_metrics,
    _fit_calibration,
    _league_metrics,
    _log_loss,
    _probabilities,
    chronological_folds,
    pre_event_team_elo_logits,
)
from lol_kills.v2.draft.terminal.development_evaluation_v2 import (
    RIDGE_STRENGTH_ORDER,
    _baseline_initial_coefficient,
    _fold_rows,
    baseline_adjusted_logits,
    composition_logits,
)
from lol_kills.v2.draft.terminal.development_snapshot import (
    load_development_snapshot,
)

SCHEMA_VERSION = "draft-terminal-development-evaluation-v4-atoms"
SUMMARY_SCHEMA_VERSION = "scryglass:draft-terminal-development-evaluation-summary:v4-atoms"
DEFAULT_SUMMARY = Path(
    "data/lol/v2/models/draft-terminal/development-evaluation-summary-v4-atoms.json"
)
DEFAULT_CROSSWALK = Path(
    "data/lol/v2/champions/champion-id-crosswalk-v1.json"
)
ATOM_CANDIDATES = (
    "m3-role-additive-atom-presence",
    "m4-atom-presence-only",
)
OPTIMIZER_MAX_ITERATIONS = 500
OPTIMIZER_GRADIENT_TOLERANCE = 1e-8
OPTIMIZER_ABSOLUTE_PARAMETER_BOUND = 20.0
MIN_FEATURE_SUPPORT = 10


class DevelopmentEvaluationV4Error(ValueError):
    """Raised when the atom-aware development evaluation fails closed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("ascii")


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))


def _log_loss_of(logits: np.ndarray, labels: Sequence[int]) -> float:
    return _log_loss(labels, _probabilities(np.asarray(logits, dtype=float), 1.0))


def _brier_of(logits: np.ndarray, labels: Sequence[int]) -> float:
    return _brier(labels, _probabilities(np.asarray(logits, dtype=float), 1.0))


def _normalize_oe(name: str) -> str:
    # Match the crosswalk's normalized OE vocabulary (lowercase, apostrophes kept).
    return name.strip().lower()


def load_atom_resolver(root: Path) -> tuple[dict[str, dict[str, bool]], Mapping[str, Any]]:
    """Return (oe_normalized_name -> family presence) + provenance.

    Fails closed when the bridge or crosswalk artifact is missing or invalid.
    """
    bridge = AtomBridge.load(root / "data/lol/v2/champions/lcc-atom-bridge-v1.json")
    crosswalk_raw = (root / DEFAULT_CROSSWALK).read_bytes()
    try:
        crosswalk = json.loads(crosswalk_raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DevelopmentEvaluationV4Error("crosswalk artifact is not valid JSON") from exc
    if not isinstance(crosswalk, dict) or not isinstance(crosswalk.get("entries"), list):
        raise DevelopmentEvaluationV4Error("crosswalk artifact shape changed")
    resolver: dict[str, dict[str, bool]] = {}
    missing: list[str] = []
    for entry in crosswalk["entries"]:
        if not isinstance(entry, dict):
            continue
        oe = entry.get("normalized_oe_name")
        stable = entry.get("stable_champion_id")
        if not isinstance(oe, str) or not isinstance(stable, str):
            continue
        profile = bridge.profile(stable)
        if profile is None:
            missing.append(stable)
            continue
        resolver[oe] = profile["family_presence"]
    if missing:
        raise DevelopmentEvaluationV4Error(
            f"crosswalk references {len(missing)} champions absent from the atom bridge"
        )
    return resolver, {
        "bridge_schema_id": BRIDGE_SCHEMA_ID,
        "bridge_artifact_sha256": bridge.artifact_sha256,
        "bridge_lcc_commit": bridge.provenance.get("lcc_commit"),
        "bridge_generated_at": bridge.generated_at,
        "crosswalk_raw_sha256": _sha256(crosswalk_raw),
    }


def atom_feature_map(
    row: DraftRow,
    candidate_id: str,
    resolver: Mapping[str, Mapping[str, bool]],
) -> dict[str, float]:
    """Per-role atom family presence, blue minus red.

    Unknown OE champion strings fail closed instead of being silently zero.
    """
    features: dict[str, float] = {}
    for side, sign in ((row.side_a, 1.0), (row.side_b, -1.0)):
        for role, champion in side:
            key = _normalize_oe(champion)
            presence = resolver.get(key)
            if presence is None:
                raise DevelopmentEvaluationV4Error(
                    f"champion {champion!r} missing from atom resolver (row {row.game_id})"
                )
            for family in CHAMPION_ATOM_FAMILIES:
                name = f"atom|{role}|{family}"
                features[name] = features.get(name, 0.0) + sign * (1.0 if presence[family] else 0.0)
    if candidate_id == "m4-atom-presence-only":
        return features
    for role, champion in row.side_a:
        features[f"main|{role}|{champion}"] = features.get(f"main|{role}|{champion}", 0.0) + 1.0
    for role, champion in row.side_b:
        features[f"main|{role}|{champion}"] = features.get(f"main|{role}|{champion}", 0.0) - 1.0
    return features


def atom_feature_vocabulary(
    rows: Sequence[DraftRow],
    candidate_id: str,
    resolver: Mapping[str, Mapping[str, bool]],
) -> tuple[str, ...]:
    support: dict[str, int] = {}
    for row in rows:
        for name, value in atom_feature_map(row, candidate_id, resolver).items():
            if value:
                support[name] = support.get(name, 0) + 1
    return tuple(sorted(name for name, count in support.items() if count >= MIN_FEATURE_SUPPORT))


def _atom_design(
    rows: Sequence[DraftRow],
    candidate_id: str,
    vocabulary: Sequence[str],
    resolver: Mapping[str, Mapping[str, bool]],
) -> np.ndarray:
    design = np.zeros((len(rows), len(vocabulary)), dtype=float)
    index = {name: position for position, name in enumerate(vocabulary)}
    for row_index, row in enumerate(rows):
        for name, value in atom_feature_map(row, candidate_id, resolver).items():
            column = index.get(name)
            if column is not None and value:
                design[row_index, column] = value
    return design


def fit_atom_candidate(
    rows: Sequence[DraftRow],
    candidate_id: str,
    ridge_strength: float,
    baseline_logits: Mapping[str, float],
    resolver: Mapping[str, Mapping[str, bool]],
) -> tuple[tuple[str, ...], np.ndarray, float]:
    if candidate_id not in ATOM_CANDIDATES:
        raise DevelopmentEvaluationV4Error(f"unregistered atom candidate: {candidate_id}")
    if ridge_strength not in RIDGE_STRENGTH_ORDER:
        raise DevelopmentEvaluationV4Error(f"unregistered ridge strength: {ridge_strength}")
    vocabulary = atom_feature_vocabulary(rows, candidate_id, resolver)
    design = _atom_design(rows, candidate_id, vocabulary, resolver)
    labels = np.asarray([row.label_a for row in rows], dtype=float)
    nuisance = np.asarray(
        [float(baseline_logits[row.game_id]) for row in rows], dtype=float
    )
    if not np.all(np.isfinite(nuisance)):
        raise DevelopmentEvaluationV4Error("baseline nuisance contains non-finite values")
    if design.shape[1] == 0:
        return vocabulary, np.zeros(0, dtype=float), 0.0
    initial = np.zeros(design.shape[1] + 1, dtype=float)
    initial[-1] = _baseline_initial_coefficient(nuisance, labels)

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        beta = parameters[:-1]
        baseline_coefficient = float(parameters[-1])
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            logits = design @ beta + baseline_coefficient * nuisance
            probabilities = _sigmoid(logits)
            loss = float(
                np.mean(np.logaddexp(0.0, logits) - labels * logits)
                + 0.5 * ridge_strength * float(np.sum(np.square(beta)))
            )
            residual = (probabilities - labels) / len(labels)
            gradient_beta = design.T @ residual + ridge_strength * beta
            gradient_baseline = float(np.sum(nuisance * residual))
        gradient = np.concatenate(
            [gradient_beta, np.asarray([gradient_baseline], dtype=float)]
        )
        return loss, gradient

    optimized = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        bounds=[
            (-OPTIMIZER_ABSOLUTE_PARAMETER_BOUND, OPTIMIZER_ABSOLUTE_PARAMETER_BOUND)
        ]
        * len(initial),
        options={
            "maxiter": OPTIMIZER_MAX_ITERATIONS,
            "gtol": OPTIMIZER_GRADIENT_TOLERANCE,
            "ftol": 1e-12,
            "maxls": 50,
        },
    )
    parameters = np.asarray(optimized.x, dtype=float)
    if not optimized.success or not np.all(np.isfinite(parameters)):
        raise DevelopmentEvaluationV4Error(
            f"atom candidate fit failed for {candidate_id}@ridge-{ridge_strength:.2f}"
        )
    return vocabulary, parameters[:-1], float(parameters[-1])


def _candidate_logits(
    fit_rows: Sequence[DraftRow],
    score_rows: Sequence[DraftRow],
    candidate_id: str,
    ridge_strength: float,
    baseline_logits: Mapping[str, float],
    resolver: Mapping[str, Mapping[str, bool]],
) -> np.ndarray:
    """Fit on fit_rows, return baseline-adjusted logits for score_rows.

    Fitting and scoring the same slice is forbidden (that would leak).
    """
    vocabulary, beta, baseline_coefficient = fit_atom_candidate(
        fit_rows, candidate_id, ridge_strength, baseline_logits, resolver
    )
    if beta.size == 0:
        raise DevelopmentEvaluationV4Error(
            f"{candidate_id}@ridge-{ridge_strength:.2f} produced no features"
        )
    design = _atom_design(score_rows, candidate_id, vocabulary, resolver)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        logits = design @ beta + baseline_coefficient * np.asarray(
            [float(baseline_logits[row.game_id]) for row in score_rows], dtype=float
        )
    if not np.all(np.isfinite(logits)) or np.max(np.abs(logits)) > 50.0:
        raise DevelopmentEvaluationV4Error(
            f"{candidate_id}@ridge-{ridge_strength:.2f} produced degenerate logits"
        )
    return logits


def zero_play_transfer_diagnostic(
    train: Sequence[DraftRow],
    all_rows: Sequence[DraftRow],
    candidate_id: str,
    ridge_strength: float,
    baseline_logits: Mapping[str, float],
    resolver: Mapping[str, Mapping[str, bool]],
    bridge: AtomBridge,
) -> dict[str, Any]:
    """Structural zero-play transfer check (R-22 / L3 DoD).

    New champions are absent from the OE crosswalk (no pro play yet), so the
    resolver cannot name them -- but the LCC atom bridge can.  Pick a bridge
    champion with zero cohort appearances, build its atom features directly
    from the bridge profile, and verify the fitted atom design gives it a
    nonzero composition value.  The role-additive design (m0) structurally
    cannot represent it (no main-effect column exists).  This proves
    archetype transfer is *structurally possible* without an empirical
    residual; it proves nothing about outcome accuracy.
    """
    if candidate_id != "m4-atom-presence-only":
        raise DevelopmentEvaluationV4Error("transfer diagnostic requires m4")
    seen = {champion for row in all_rows for _, champion in (*row.side_a, *row.side_b)}
    # probe candidates: bridge champions with no cohort appearances
    probe_ids = [
        cid for cid in bridge.champion_ids()
        if bridge.profile(cid)["display_name"] not in seen
    ]
    if not probe_ids:
        return {"status": "no_unseen_champion_in_cohort"}
    probe_id = probe_ids[0]
    probe_profile = bridge.profile(probe_id)
    probe_presence = probe_profile["family_presence"]

    vocabulary, beta, _ = fit_atom_candidate(
        train, candidate_id, ridge_strength, baseline_logits, resolver
    )
    # construct the synthetic row's atom features: top lane becomes the probe
    # (bridge profile), every other slot keeps its real champion (resolver).
    synthetic_features: dict[str, float] = {}
    template = train[0]
    for side, sign in ((template.side_a, 1.0), (template.side_b, -1.0)):
        for role, champion in side:
            if role == "top":
                presence = probe_presence
            else:
                presence = resolver.get(_normalize_oe(champion))
                if presence is None:
                    raise DevelopmentEvaluationV4Error(
                        f"row {template.game_id}: champion {champion!r} missing from resolver"
                    )
            for family in CHAMPION_ATOM_FAMILIES:
                name = f"atom|{role}|{family}"
                synthetic_features[name] = synthetic_features.get(name, 0.0) + sign * (
                    1.0 if presence[family] else 0.0
                )
    index = {name: position for position, name in enumerate(vocabulary)}
    design_row = np.zeros(len(vocabulary), dtype=float)
    for name, value in synthetic_features.items():
        column = index.get(name)
        if column is not None and value:
            design_row[column] = value
    atom_logit = float(design_row @ beta)
    return {
        "status": "checked",
        "probe_champion_id": probe_id,
        "probe_display_name": probe_profile["display_name"],
        "probe_family_presence": probe_presence,
        "cohort_champion_count": len(seen),
        "atom_logit_for_unseen_champion": round(atom_logit, 6),
        "structurally_possible": abs(atom_logit) > 1e-12,
        "note": "bridge atom presence gives a never-seen champion a nonzero composition value by construction",
    }


def _code_bindings() -> dict[str, str]:
    directory = Path(__file__).resolve().parent
    return {
        "development_evaluation_v4_atoms": _sha256(Path(__file__).read_bytes()),
        "development_evaluation_v2_fit": _sha256((directory / "development_evaluation_v2.py").read_bytes()),
        "development_evaluation_helpers": _sha256((directory / "development_evaluation.py").read_bytes()),
        "development_snapshot": _sha256((directory / "development_snapshot.py").read_bytes()),
    }


def evaluate(root: Path) -> dict[str, Any]:
    rows, source_snapshot = load_development_snapshot(root)
    resolver, resolver_provenance = load_atom_resolver(root)

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
    folds = chronological_folds(len(cluster_order))

    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        train = _fold_rows(rows, cluster_order, fold.train)
        validation = _fold_rows(rows, cluster_order, fold.validation)
        calibration = _fold_rows(rows, cluster_order, fold.calibration)
        test = _fold_rows(rows, cluster_order, fold.test)
        test_start = min((row.date for row in test), default=None)
        if test_start is None:
            raise DevelopmentEvaluationV4Error(f"{fold.fold_id} outer test is empty")
        baseline_logits = pre_event_team_elo_logits(rows, freeze_at=test_start)

        # baseline-only model (pre-event team-strength nuisance only)
        from .development_evaluation_v3 import baseline_only_logits, fit_baseline_only
        baseline_coefficient = fit_baseline_only(train, baseline_logits)
        baseline_logits_test = baseline_only_logits(test, baseline_coefficient, baseline_logits)
        labels_test = [row.label_a for row in test]
        baseline_test_log_loss = _log_loss_of(np.asarray(baseline_logits_test, dtype=float), labels_test)
        baseline_test_brier = _brier_of(np.asarray(baseline_logits_test, dtype=float), labels_test)

        # m0 reference from the existing v2 machinery for the same fold
        m0_report = _fit_m0_reference(train, validation, calibration, test, baseline_logits)
        m0_test_log_loss = float(m0_report.get("test_log_loss", float("nan")))
        m0_test_brier = float(m0_report.get("test_brier_score", float("nan")))

        zero_play_diagnostic = zero_play_transfer_diagnostic(
            train, rows, "m4-atom-presence-only", 0.05, baseline_logits, resolver,
            AtomBridge.load(root / "data/lol/v2/champions/lcc-atom-bridge-v1.json"),
        )

        results: dict[str, Any] = {}
        for candidate_id in ATOM_CANDIDATES:
            variants: list[dict[str, Any]] = []
            for ridge in RIDGE_STRENGTH_ORDER:
                try:
                    # Fit on TRAIN only; score the validation slice.
                    validation_logits = _candidate_logits(
                        train, validation, candidate_id, ridge, baseline_logits, resolver
                    )
                except DevelopmentEvaluationV4Error:
                    continue
                labels = [row.label_a for row in validation]
                variants.append(
                    {
                        "variant_id": f"{candidate_id}@ridge-{ridge:.2f}",
                        "log_loss": _log_loss_of(validation_logits, labels),
                        "brier_score": _brier_of(validation_logits, labels),
                        "ridge_strength": ridge,
                    }
                )
            if not variants:
                continue
            selected = min(variants, key=lambda v: (v["log_loss"], v["brier_score"]))
            ridge = selected["ridge_strength"]
            test_logits = _candidate_logits(
                train, test, candidate_id, ridge, baseline_logits, resolver
            )
            results[candidate_id] = {
                "selected_variant": selected["variant_id"],
                "validation_variants": variants,
                "test_log_loss": _log_loss_of(test_logits, labels_test),
                "test_brier_score": _brier_of(test_logits, labels_test),
                "incremental_vs_baseline_only": {
                    "log_loss": _log_loss_of(test_logits, labels_test) - baseline_test_log_loss,
                    "brier_score": _brier_of(test_logits, labels_test) - baseline_test_brier,
                    "pass_rule": "both deltas must be nonpositive",
                    "passed": _log_loss_of(test_logits, labels_test) <= baseline_test_log_loss
                    and _brier_of(test_logits, labels_test) <= baseline_test_brier,
                    "negative_is_better": True,
                },
                "incremental_vs_m0": {
                    "log_loss": _log_loss_of(test_logits, labels_test) - m0_test_log_loss,
                    "brier_score": _brier_of(test_logits, labels_test) - m0_test_brier,
                    "pass_rule": "both deltas must be nonpositive",
                    "negative_is_better": True,
                },
            }
        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "results": results,
                "m0_role_additive_reference": m0_report,
                "zero_play_transfer_diagnostic": zero_play_diagnostic,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "summary_schema_version": SUMMARY_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "development_only": True,
        "claim_scope": "R-22 development diagnostic; no authority",
        "code_bindings": _code_bindings(),
        "source_snapshot": source_snapshot,
        "resolver_provenance": resolver_provenance,
        "atom_candidates": list(ATOM_CANDIDATES),
        "ridge_strengths": list(RIDGE_STRENGTH_ORDER),
        "folds": fold_reports,
    }
    return payload


def _calibration(
    logits: np.ndarray, labels: Sequence[int]
) -> tuple[str, float, list[dict[str, Any]]]:
    choices: list[tuple[float, int, str, float]] = []
    reports: list[dict[str, Any]] = []
    for method in CALIBRATION_ORDER:
        parameter, loss = _fit_calibration(logits, labels, method)
        choices.append((float(loss), CALIBRATION_ORDER.index(method), method, parameter))
        reports.append({"method": method, "parameter": parameter, "calibration_log_loss": loss})
    _, _, selected_method, selected_parameter = min(choices)
    scale = (
        1.0 / selected_parameter
        if selected_method == "symmetric_temperature"
        else selected_parameter
    )
    for report in reports:
        report["selected"] = report["method"] == selected_method
    return selected_method, float(scale), reports


def _fit_m0_reference(
    train: Sequence[DraftRow],
    validation: Sequence[DraftRow],
    calibration: Sequence[DraftRow],
    test: Sequence[DraftRow],
    baseline_logits: Mapping[str, float],
) -> dict[str, Any]:
    """Reference: the frozen m0-role-additive candidate via the v2 machinery."""
    from .development_evaluation_v2 import baseline_adjusted_logits, fit_penalized

    variants = []
    for ridge in RIDGE_STRENGTH_ORDER:
        try:
            fit = fit_penalized(train, "m0-role-additive", ridge, baseline_logits)
            logits = baseline_adjusted_logits(validation, fit, baseline_logits)
        except Exception:
            continue
        labels = [row.label_a for row in validation]
        variants.append(
            {
                "variant_id": f"m0-role-additive@ridge-{ridge:.2f}",
                "log_loss": _log_loss_of(logits, labels),
                "brier_score": _brier_of(logits, labels),
                "ridge_strength": ridge,
            }
        )
    if not variants:
        return {"status": "unavailable"}
    selected = min(variants, key=lambda v: (v["log_loss"], v["brier_score"]))
    fit = fit_penalized(train, "m0-role-additive", selected["ridge_strength"], baseline_logits)
    logits = baseline_adjusted_logits(test, fit, baseline_logits)
    labels = [row.label_a for row in test]
    return {
        "selected_variant": selected["variant_id"],
        "test_log_loss": _log_loss_of(logits, labels),
        "test_brier_score": _brier_of(logits, labels),
    }


def write_summary(payload: dict[str, Any], path: Path = DEFAULT_SUMMARY) -> str:
    digest = hashlib.sha256(_canonical_json(payload)).hexdigest()
    artifact = dict(payload)
    artifact["artifact_sha256"] = digest
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return digest


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the R-22 atom-aware draft development evaluation.")
    ap.add_argument("--root", type=Path, default=Path.cwd())
    ap.add_argument("--out", type=Path, default=DEFAULT_SUMMARY)
    args = ap.parse_args()
    payload = evaluate(args.root)
    digest = write_summary(payload, args.out)
    print(f"wrote {args.out}")
    print(f"artifact_sha256 = {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
