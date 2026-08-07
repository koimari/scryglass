"""Approval-gated private G5 orchestration and a nonpromotable synthetic harness.

The public production entrypoint accepts only a run identity.  Authority is
human-rooted and every protected loader is reached only after a durable STARTED
process-ledger entry.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from . import contract, result


ROOT = Path(__file__).resolve().parents[5]
NAMESPACE = ROOT / "data/lol/v2/models/draft-interactions/g5-exploratory"
CONTRACT_SHA = "28eaac0c407b2fd422ac6fa1936f5f7f63946142cc0a8be3542bc4d1cf71ac53"
PREFIT_CORE_SHA = "af14e15fed83ae859f57b201396d256d7ef2cbeb1dc836dd18055eabbd6357bf"
PREFIT_REVIEW_SHA = "f2ee55946410e564a1453833f113195f7ef323d3dec4f1a1dd5e1579bef46a28"
FEATURE_MANIFEST_SHA = "7e559054ac3f1bd79f1821121c17b778927736f6a3a52c85b48b5d3d0460189c"
ROLES = ("top", "jungle", "mid", "bot", "support")
EXPECTED_MAPS = 1226
EXPECTED_PICKS = 12260
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_BASE_SEED = 2026073005
RESULT_LOCATOR = "data/lol/v2/models/draft-interactions/g5-exploratory/execution-result.json"


class G5RunnerError(RuntimeError):
    """A fail-closed execution invariant was violated."""


@dataclass(frozen=True)
class SyntheticMap:
    map_key: str
    fold: str
    cluster_key: str
    b0_logit: float
    picks: tuple[tuple[str, str, str], ...]


@dataclass(frozen=True)
class SyntheticEvaluation:
    map_key: str
    label: int


@dataclass(frozen=True)
class DraftPick:
    source_side: str
    role: str
    stable_champion_id: str


@dataclass(frozen=True)
class OutcomeFreeMap:
    map_key: str
    fold: str
    source_local_event_start: str
    cluster_key: str
    b0_logit_mean: float
    b0_logit_variance: float
    b0_probability: float
    picks: tuple[DraftPick, ...]


@dataclass(frozen=True)
class EvaluationLedger:
    map_key: str
    fold: str
    label: int


@dataclass(frozen=True)
class AlignedInputs:
    maps: tuple[Any, ...]
    feature_rows: tuple[Mapping[str, Any], ...]
    clusters: Mapping[str, str]
    feature_manifest: Mapping[str, Any]
    cluster_artifact_sha256: str


@dataclass(frozen=True)
class WinnerAggregate:
    score_subject: Mapping[str, Any]
    B0_probability: float
    D1_logit_increment: float
    neutral_completed_draft_probability: float
    probability_increment_over_B0: float
    D1_conditional_interval: Mapping[str, Any]


@dataclass(frozen=True)
class AggregateEvidence:
    """Internal computation evidence; deliberately has no REAL schema method."""

    state: str
    blocker: str | None
    selected_candidate: str | None
    counts: Mapping[str, Any]
    membership_hashes: Mapping[str, Any]
    source_and_feature_review_pins: Mapping[str, Any]
    G2_core_pins: Mapping[str, Any]
    development_metric: Mapping[str, Any]
    validation_metric: Mapping[str, Any]
    bootstrap: Mapping[str, Any]
    objective_gradient_hessian_diagnostics: Mapping[str, Any]
    solver_diagnostics: Mapping[str, Any]
    uncertainty: Mapping[str, Any]
    prior_only_variance_components: Mapping[str, Any]
    coverage_and_prior_only_flags: Mapping[str, Any]
    invariance_tests: Mapping[str, Any]
    contribution_reconciliation: Mapping[str, Any]
    score_subject: Mapping[str, Any]
    context: Mapping[str, Any]
    winner: WinnerAggregate | None


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise G5RunnerError("noncanonical execution value") from error


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _raw(path: Path, *, label: str = "reviewed subject") -> str:
    metadata = os.lstat(path)
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise G5RunnerError(f"unsafe {label}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_content_addressed_json(path: Path, expected: str, field: str) -> Mapping[str, Any]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise G5RunnerError("frozen metadata is not JSON") from error
    if not isinstance(payload, Mapping) or raw != _canonical(payload) + b"\n":
        raise G5RunnerError("frozen metadata is not canonical newline JSON")
    unsigned = dict(payload)
    claimed = unsigned.pop(field, None)
    if claimed != expected or _sha(unsigned) != expected:
        raise G5RunnerError("frozen metadata canonical identity mismatch")
    return payload


def _frozen_prefit() -> None:
    for name, expected, field in (
        ("contract.json", CONTRACT_SHA, "artifact_sha256"),
        ("review-core.json", PREFIT_CORE_SHA, "artifact_sha256"),
        ("pre-fit-review.json", PREFIT_REVIEW_SHA, "artifact_sha256"),
    ):
        _verify_content_addressed_json(NAMESPACE / name, expected, field)
    try:
        contract.verify_bound_dependencies()
    except Exception as error:
        raise G5RunnerError("frozen dependency identity mismatch") from error


def _runtime_bindings() -> dict[str, Any]:
    return {
        "numpy": {"version": importlib.metadata.version("numpy"), "module_raw_sha256": _raw(Path(np.__file__), label="numpy module")},
        "scipy_optimize": {
            "version": importlib.metadata.version("scipy"),
            "module_raw_sha256": _raw(Path(inspect.getsourcefile(minimize) or ""), label="scipy optimizer module"),
        },
    }


def build_runner_review_bundle() -> dict[str, dict[str, Any]]:
    """Build metadata-only review artifacts without approval or protected rows."""

    _frozen_prefit()
    subjects = {
        "runner": "lol_kills/v2/draft/interactions/g5_exploratory/runner.py",
        "result": "lol_kills/v2/draft/interactions/g5_exploratory/result.py",
        "approval_contract": "lol_kills/v2/draft/interactions/g5_exploratory/execution_approval.py",
        "focused_test": "tests/model_v2/draft/interactions/test_g5_exploratory_runner.py",
        "approval_test": "tests/model_v2/draft/interactions/test_g5_execution_approval.py",
    }
    from . import execution_approval

    core: dict[str, Any] = {
        "schema_id": "scryglass:g5-runner-review-core:v5",
        "prefit": {
            "contract_sha256": CONTRACT_SHA,
            "core_sha256": PREFIT_CORE_SHA,
            "review_sha256": PREFIT_REVIEW_SHA,
        },
        "review_subject_bytes": {
            key: {"locator": locator, "raw_sha256": _raw(ROOT / locator)}
            for key, locator in subjects.items()
        },
        "execution_dependency_pins": {
            "G1": contract.G1,
            "G1_features": contract.G1_FEATURES,
            "G2": contract.G2,
            "clusters": contract.CLUSTERS,
            "runtime": _runtime_bindings(),
        },
        "execution_approval_contract": {
            "locator": subjects["approval_contract"],
            "raw_sha256": _raw(ROOT / subjects["approval_contract"]),
            "approval_locator": execution_approval.APPROVAL_LOCATOR,
            "ledger_locator": execution_approval.LEDGER_LOCATOR,
            "result_locator": execution_approval.RESULT_LOCATOR,
            "human_root": execution_approval.ROOT_AUTHORITY,
            "scope": execution_approval.SCOPE,
            "uniqueness_limit": "PROCESS_AND_CONTROL_ENFORCED_NOT_ADVERSARIAL_OR_CONCURRENT",
        },
        "claim_ceiling": {
            "real_execution": False,
            "synthetic_only": True,
            "prediction": False,
            "publication": False,
            "final_holdout": False,
        },
    }
    core["artifact_sha256"] = _sha(core)
    missing = ["final_independent_runner_review", "canonical_execution_approval"]
    pending: dict[str, Any] = {
        "schema_id": "scryglass:g5-runner-pending-report:v5",
        "state": "PREFIT_CONTRACT_FROZEN_EXECUTION_NOT_AUTHORIZED",
        "runner_core_sha256": core["artifact_sha256"],
        "protected_reads": 0,
        "completed": [
            "production_orchestration",
            "strict_result_schema",
            "approval_bound_process_ledger_writer",
            "scripted_no_protected_read_tests",
        ],
        "missing": missing,
        "claim_ceiling": core["claim_ceiling"],
        "approval": {"locator": execution_approval.APPROVAL_LOCATOR, "status": "MISSING_NOT_ISSUED"},
    }
    pending["artifact_sha256"] = _sha(pending)
    return {"execution-review-core.json": core, "execution-pending-report.json": pending}


def _safe_fixed_files(paths: Sequence[Path]) -> None:
    if len({path.absolute() for path in paths}) != len(paths):
        raise G5RunnerError("duplicate immutable output path")
    root = ROOT.absolute()
    for path in paths:
        if ".." in path.parts:
            raise G5RunnerError("unsafe output traversal")
        try:
            relative = path.absolute().relative_to(root)
        except ValueError as error:
            raise G5RunnerError("output outside repository") from error
        current = root
        for part in relative.parts[:-1]:
            current = current / part
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise G5RunnerError("unsafe output parent")
        try:
            metadata = os.lstat(path)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise G5RunnerError("unsafe output leaf")


def _immutable_write_many(items: Sequence[tuple[Path, bytes]]) -> None:
    """Stage all bytes, preflight all leaves, and roll back a partial commit."""

    paths = [path for path, _ in items]
    if any(path.absolute() == (ROOT / RESULT_LOCATOR).absolute() for path in paths):
        raise G5RunnerError("real result path is execute_real-only")
    _safe_fixed_files(paths)
    existence = [path.exists() for path in paths]
    if any(existence):
        if not all(existence):
            raise G5RunnerError("immutable output bundle is partial")
        if all(path.read_bytes() == data for (path, data) in items):
            return
        raise G5RunnerError("immutable output already exists with different bytes")
    staged: list[tuple[Path, str]] = []
    committed: list[Path] = []
    try:
        for path, data in items:
            descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            staged.append((path, temporary))
        _safe_fixed_files(paths)
        if any(path.exists() for path in paths):
            raise G5RunnerError("immutable output appeared during staging")
        for path, temporary in staged:
            os.link(temporary, path)
            committed.append(path)
        directory_descriptor = os.open(str(paths[0].parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        for path in reversed(committed):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
        raise
    finally:
        for _, temporary in staged:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def write_runner_review_bundle(directory: Path = NAMESPACE) -> dict[str, str]:
    if directory.absolute() != NAMESPACE.absolute():
        raise G5RunnerError("review bundle path is fixed")
    bundle = build_runner_review_bundle()
    items = [
        (NAMESPACE / name, _canonical(payload) + b"\n")
        for name, payload in sorted(bundle.items())
    ]
    _immutable_write_many(items)
    return {name: payload["artifact_sha256"] for name, payload in bundle.items()}


def _ledger_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ledger_timestamp_not_before(prior: str) -> str:
    return max(_ledger_timestamp(), prior)


def _earliest_completed(
    entries: Sequence[Mapping[str, Any]], *, approval_id: str
) -> Mapping[str, Any] | None:
    return next(
        (
            entry
            for entry in entries
            if entry.get("state") == "COMPLETED"
            and entry.get("approval_id") == approval_id
        ),
        None,
    )


def execute_real(run_id: str) -> Mapping[str, Any]:
    """Run only after exact human approval and a durable STARTED ledger entry."""

    from . import execution_approval

    _frozen_prefit()
    core = build_runner_review_bundle()["execution-review-core.json"]
    core_sha256 = core["artifact_sha256"]
    try:
        approval = execution_approval.load_approval(
            expected_runner_core_sha256=core_sha256,
            expected_run_id=run_id,
        )
        ledger = execution_approval.load_ledger()
    except Exception as error:
        raise G5RunnerError("EXECUTION_BLOCKED:APPROVAL_INVALID") from error
    try:
        ledger_state = execution_approval.validate_ledger_history(
            ledger,
            approval=approval,
            expected_runner_core_sha256=core_sha256,
            expected_run_id=run_id,
        )
    except Exception as error:
        raise G5RunnerError("EXECUTION_BLOCKED:LEDGER_INVALID") from error
    if ledger_state == "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY":
        raise G5RunnerError("EXECUTION_BLOCKED:LEDGER_INVALID")
    if ledger_state == "COMPLETED_TERMINAL":
        completed = ledger[1]
        try:
            execution_approval.validate_completed_result(
                completed,
                approval=approval,
                expected_runner_core_sha256=core_sha256,
                expected_run_id=run_id,
                started_entry_sha256=ledger[0]["entry_sha256"],
            )
        except Exception as error:
            raise G5RunnerError("EXECUTION_BLOCKED:LEDGER_INVALID") from error
        try:
            execution_approval.append_ledger_entry({
                "state": "INVALID_DUPLICATE",
                "approval_id": approval["approval_id"],
                "run_id": run_id,
                "runner_core_sha256": core_sha256,
                "approval_sha256": approval["approval_sha256"],
                "result_locator": execution_approval.RESULT_LOCATOR,
                "sequence": len(ledger) + 1,
                "authoritative_completed_entry_sha256": completed["entry_sha256"],
                "authoritative_result_artifact_sha256": completed["result_artifact_sha256"],
                "recorded_at": _ledger_timestamp_not_before(
                    ledger[-1].get("recorded_at", completed["completed_at"])
                ),
            })
        except Exception:
            pass
        raise G5RunnerError("EXECUTION_BLOCKED:INVALID_DUPLICATE")
    try:
        started = execution_approval.append_ledger_entry({
            "state": "STARTED",
            "approval_id": approval["approval_id"],
            "run_id": run_id,
            "runner_core_sha256": core_sha256,
            "approval_sha256": approval["approval_sha256"],
            "result_locator": execution_approval.RESULT_LOCATOR,
            "sequence": 1,
            "started_at": _ledger_timestamp(),
        })
    except Exception as error:
        raise G5RunnerError("EXECUTION_BLOCKED:LEDGER_INVALID") from error
    try:
        current_ledger = execution_approval.load_ledger()
        if execution_approval.validate_ledger_history(
            current_ledger,
            approval=approval,
            expected_runner_core_sha256=core_sha256,
            expected_run_id=run_id,
        ) != "STARTED_INCOMPLETE_NO_AUTOMATIC_RETRY":
            raise G5RunnerError("unexpected post-start ledger state")
    except Exception as error:
        raise G5RunnerError("EXECUTION_BLOCKED:LEDGER_INVALID") from error

    try:
        evidence = _execute_bound_pipeline()
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        blocker = (
            str(error)
            if isinstance(error, G5RunnerError)
            else "EXECUTION_BLOCKED:UNEXPECTED_PIPELINE_FAILURE"
        )
        evidence = _blocked_evidence(blocker=blocker)

    prior_payload = {
        **dict(evidence.prior_only_variance_components),
        "coordinate_exposure_witness": [
            dict(record)
            for record in evidence.prior_only_variance_components[
                "coordinate_exposure_witness"
            ]
        ],
    }
    unsigned = {
        "schema_version": result.REAL_SCHEMA,
        "state": evidence.state,
        "blocker": evidence.blocker,
        "selected_candidate": evidence.selected_candidate,
        "counts": dict(evidence.counts),
        "membership_hashes": dict(evidence.membership_hashes),
        "source_and_feature_review_pins": dict(evidence.source_and_feature_review_pins),
        "G2_core_pins": dict(evidence.G2_core_pins),
        "development_metric": dict(evidence.development_metric),
        "validation_metric": dict(evidence.validation_metric),
        "bootstrap": dict(evidence.bootstrap),
        "objective_gradient_hessian_diagnostics": dict(evidence.objective_gradient_hessian_diagnostics),
        "solver_diagnostics": dict(evidence.solver_diagnostics),
        "uncertainty": dict(evidence.uncertainty),
        "prior_only_variance_components": prior_payload,
        "coverage_and_prior_only_flags": dict(evidence.coverage_and_prior_only_flags),
        "invariance_tests": dict(evidence.invariance_tests),
        "contribution_reconciliation": dict(evidence.contribution_reconciliation),
        "score_subject": dict(evidence.score_subject),
        "context": dict(evidence.context),
        "execution_binding": {
            "run_id_sha256": _sha(run_id),
            "runner_core_sha256": core_sha256,
            "approval_sha256": approval["approval_sha256"],
            "started_entry_sha256": started["entry_sha256"],
            "result_locator": execution_approval.RESULT_LOCATOR,
            "uniqueness_enforcement": "PROCESS_AND_CONTROL_ONLY",
        },
        "claim_ceiling": result.CLAIM_CEILING,
        "execution_limitation": (
            "Run uniqueness is process/control enforced only and provides no G9, public, "
            "concurrent, or adversarial single-use authority."
        ),
    }
    if evidence.winner is not None:
        unsigned.update({
            "private_retrospective_exploratory_score_probability": evidence.winner.neutral_completed_draft_probability,
            "fit_evidence": "TRAIN_ONLY",
            "rank_selection_evidence": "DEVELOPMENT_LOCKED_VALIDATION_GATED",
            "B0_probability": evidence.winner.B0_probability,
            "D1_logit_increment": evidence.winner.D1_logit_increment,
            "neutral_completed_draft_probability": evidence.winner.neutral_completed_draft_probability,
            "probability_increment_over_B0": evidence.winner.probability_increment_over_B0,
            "D1_conditional_interval": dict(evidence.winner.D1_conditional_interval),
        })
    payload = {**unsigned, "artifact_sha256": result.sha256(unsigned)}
    result.validate_real(payload)
    if execution_approval.load_approval(
        expected_runner_core_sha256=core_sha256, expected_run_id=run_id
    ) != approval:
        raise G5RunnerError("EXECUTION_BLOCKED:APPROVAL_INVALID")
    path = execution_approval._safe_path(
        execution_approval.RESULT_LOCATOR, may_be_missing=True
    )
    data = result.canonical_bytes(payload) + b"\n"
    _safe_fixed_files((path,))
    if path.exists():
        if path.read_bytes() != data:
            raise G5RunnerError("immutable real result already exists")
    else:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=str(path.parent)
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    execution_approval.append_ledger_entry({
        "state": "COMPLETED",
        "approval_id": approval["approval_id"],
        "run_id": run_id,
        "runner_core_sha256": core_sha256,
        "approval_sha256": approval["approval_sha256"],
        "result_locator": execution_approval.RESULT_LOCATOR,
        "sequence": 2,
        "started_entry_sha256": started["entry_sha256"],
        "result_artifact_sha256": payload["artifact_sha256"],
        "result_raw_sha256": hashlib.sha256(data).hexdigest(),
        "completed_at": _ledger_timestamp_not_before(started["started_at"]),
    })
    final_ledger = execution_approval.load_ledger()
    if execution_approval.validate_ledger_history(
        final_ledger,
        approval=approval,
        expected_runner_core_sha256=core_sha256,
        expected_run_id=run_id,
    ) != "COMPLETED_TERMINAL":
        raise G5RunnerError("EXECUTION_BLOCKED:LEDGER_INVALID")
    execution_approval.validate_completed_result(
        final_ledger[1],
        approval=approval,
        expected_runner_core_sha256=core_sha256,
        expected_run_id=run_id,
        started_entry_sha256=final_ledger[0]["entry_sha256"],
    )
    return payload


def _blocked_evidence(*, blocker: str) -> AggregateEvidence:
    """Return non-serializable post-start blocked evidence."""

    if blocker not in result.BLOCKER_CODES:
        blocker = "EXECUTION_BLOCKED:UNEXPECTED_PIPELINE_FAILURE"
    empty = _sha([])
    return AggregateEvidence(
        state="EXECUTION_BLOCKED",
        blocker=blocker,
        selected_candidate=None,
        counts={"maps": 0, "picks": 0, "TRAIN": 0, "DEVELOPMENT": 0, "VALIDATION": 0},
        membership_hashes={
            "all_maps_sha256": empty,
            "TRAIN_maps_sha256": empty,
            "DEVELOPMENT_maps_sha256": empty,
            "VALIDATION_maps_sha256": empty,
            "cluster_membership_sha256": empty,
            "origin_membership_sha256": empty,
            "feature_membership_sha256": empty,
        },
        source_and_feature_review_pins={
            "G1_manifest_sha256": contract.G1["manifest_sha256"],
            "G1_rows_sha256": contract.G1["rows_sha256"],
            "selected_target_sha256": contract.G1["selected_target_sha256"],
            "split_payload_sha256": contract.G1["split_payload_sha256"],
            "feature_manifest_sha256": contract.G1_FEATURES["manifest_canonical_sha256"],
            "feature_rows_raw_sha256": contract.G1_FEATURES["rows_raw_sha256"],
            "feature_rows_canonical_sha256": contract.G1_FEATURES["rows_canonical_sha256"],
            "feature_review_sha256": contract.G1_FEATURES["independent_review_canonical_sha256"],
            "cluster_artifact_sha256": contract.CLUSTERS["artifact_canonical_sha256"],
        },
        G2_core_pins={
            "runner_raw_sha256": contract.G2["runner_raw_sha256"],
            "model_raw_sha256": contract.G2["model_raw_sha256"],
            "artifact_raw_sha256": contract.G2["artifact_raw_sha256"],
            "artifact_canonical_sha256": contract.G2["artifact_canonical_sha256"],
            "candidate": "static_baseline",
        },
        development_metric={
            "locked_candidate": None, "map_count": 0, "evaluations": 0,
            "B0_mean_log_loss": None, "D1_mean_log_loss": None,
            "mean_LL_B0_minus_LL_D1": None,
        },
        validation_metric={
            "locked_candidate": None, "map_count": 0, "evaluations": 0,
            "B0_mean_log_loss": None, "locked_candidate_mean_log_loss": None,
            "mean_LL_B0_minus_LL_locked_candidate": None,
        },
        bootstrap={"status": "BLOCKED", "replicates": 0, "base_seed": None, "quantile": None, "lower_bound": None, "map_weighted": True},
        objective_gradient_hessian_diagnostics={
            "objective": None,
            "gradient_infinity_norm": None,
            "hessian_dimension": 0,
            "hessian_symmetric_atol_1e_12": False,
            "hessian_positive_definite": False,
        },
        solver_diagnostics={
            "status": "BLOCKED",
            "method": "L-BFGS-B",
            "analytic_jacobian": True,
            "iterations": 0,
            "function_evaluations": 0,
            "message_sha256": _sha(blocker),
        },
        uncertainty={
            "B0_latent_mean_available": False,
            "B0_latent_variance_available": False,
            "D1_conditional_covariance": "UNAVAILABLE",
            "total_B0_plus_D1_interval": "PROHIBITED",
        },
        prior_only_variance_components={
            "status": "BLOCKED",
            "role_delta_count": 0,
            "variance_per_coordinate": 0.01,
            "total_variance": 0.0,
            "mean_score_aggregate_variance": 0.0,
            "conditional_mean_logit_variance": 0.0,
            "slot_membership_sha256": empty,
            "signed_exposure_sha256": empty,
            "coordinate_exposure_witness": [],
        },
        coverage_and_prior_only_flags={
            "complete_maps": False,
            "complete_picks": False,
            "champion_absent_from_TRAIN": blocker.endswith("CHAMPION_ABSENT_FROM_TRAIN"),
            "prior_only_role_delta_used": False,
            "final_holdout_reads": 0,
        },
        invariance_tests={
            "side_swap": {"status": "BLOCKED", "map_count": 0, "absolute_tolerance": 1e-12, "max_absolute_error": None},
            "record_order": {"status": "BLOCKED", "map_count": 0, "absolute_tolerance": 1e-12, "max_absolute_error": None},
            "role_relabel": {"status": "NOT_INVARIANT_BY_CONTRACT"},
        },
        contribution_reconciliation={"status": "BLOCKED", "absolute_tolerance": 1e-12, "max_absolute_error": None},
        score_subject={"status": "NOT_RUN", "kind": "VALIDATION_COHORT_AGGREGATE", "fold": "VALIDATION", "map_count": 0, "weighting": "MAP_EQUAL", "order_invariant": True},
        context={"status": "UNAVAILABLE", "blocker": "CONTEXTUAL_EXACT_FIVE_OR_PLAYER_CHAMPION_EVIDENCE_UNAVAILABLE"},
        winner=None,
    )


def _load_accepted_g1() -> Any:
    from lol_kills.v2.ratings.player.real_v1_adapter import (
        load_accepted_lpl_private_player_rating_input,
    )

    return load_accepted_lpl_private_player_rating_input()


def _load_accepted_features() -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    from lol_kills.v2.data import g1_draft_features

    manifest = g1_draft_features.verify(expected_manifest_sha256=FEATURE_MANIFEST_SHA)
    path = ROOT / contract.G1_FEATURES["rows_locator"]
    if _raw(path, label="accepted feature rows") != contract.G1_FEATURES["rows_raw_sha256"]:
        raise G5RunnerError("EXECUTION_BLOCKED:FEATURE_IDENTITY_MISMATCH")
    rows = tuple(json.loads(line) for line in path.read_bytes().splitlines())
    if _sha(list(rows)) != contract.G1_FEATURES["rows_canonical_sha256"]:
        raise G5RunnerError("EXECUTION_BLOCKED:FEATURE_IDENTITY_MISMATCH")
    return manifest, rows


def _load_accepted_clusters() -> Mapping[str, Any]:
    from lol_kills.v2.draft.interactions import series_cluster_proxy

    return series_cluster_proxy.load_and_replay_artifact(
        ROOT / contract.CLUSTERS["artifact_locator"],
        source_root=ROOT,
    )


def _assert_g1_identity(g1: Any) -> None:
    actual = (
        g1.manifest_sha256,
        g1.rows_sha256,
        g1.selected_target_sha256,
        g1.split_payload_sha256,
        g1.map_count,
        g1.player_observation_count,
    )
    expected = (
        contract.G1["manifest_sha256"],
        contract.G1["rows_sha256"],
        contract.G1["selected_target_sha256"],
        contract.G1["split_payload_sha256"],
        EXPECTED_MAPS,
        EXPECTED_PICKS,
    )
    if actual != expected:
        raise G5RunnerError("EXECUTION_BLOCKED:G1_IDENTITY_MISMATCH")


def align_inputs(
    g1: Any,
    feature_manifest: Mapping[str, Any],
    feature_rows: Sequence[Mapping[str, Any]],
    cluster_artifact: Mapping[str, Any],
) -> AlignedInputs:
    """Align exact map, fold, time, lineup, feature, and cluster identities."""

    _assert_g1_identity(g1)
    from lol_kills.v2.ratings.player import private_development_runner as g2

    try:
        g2._folds_by_id(g1)
    except Exception as error:
        raise G5RunnerError("EXECUTION_BLOCKED:G1_MEMBERSHIP_ORIGIN_DIGEST_MISMATCH") from error
    if (
        feature_manifest.get("manifest_sha256") != FEATURE_MANIFEST_SHA
        or feature_manifest.get("rows_raw_sha256") != contract.G1_FEATURES["rows_raw_sha256"]
        or feature_manifest.get("rows_canonical_sha256") != contract.G1_FEATURES["rows_canonical_sha256"]
    ):
        raise G5RunnerError("EXECUTION_BLOCKED:FEATURE_IDENTITY_MISMATCH")
    maps = tuple(item for fold in g1.folds for item in fold.map_observations)
    accepted_membership = feature_manifest.get("accepted_membership_origin")
    if (
        not isinstance(accepted_membership, Mapping)
        or accepted_membership.get("fold_map_digests")
        != {fold.fold_id: fold.ordered_map_ids_sha256 for fold in g1.folds}
        or accepted_membership.get("fold_origin_digests")
        != {fold.fold_id: fold.ordered_origin_identities_sha256 for fold in g1.folds}
    ):
        raise G5RunnerError("EXECUTION_BLOCKED:FEATURE_MEMBERSHIP_ORIGIN_DIGEST_MISMATCH")
    if (
        len(maps) != EXPECTED_MAPS
        or len({item.source_game_id for item in maps}) != EXPECTED_MAPS
        or len(feature_rows) != EXPECTED_MAPS
    ):
        raise G5RunnerError("EXECUTION_BLOCKED:MAP_MEMBERSHIP_ALIGNMENT")
    map_by_key = {item.source_game_id: item for item in maps}
    feature_by_key: dict[str, Mapping[str, Any]] = {}
    for row in feature_rows:
        key = row.get("source_game_id")
        if not isinstance(key, str) or key in feature_by_key:
            raise G5RunnerError("EXECUTION_BLOCKED:MAP_MEMBERSHIP_ALIGNMENT")
        feature_by_key[key] = row
    if set(map_by_key) != set(feature_by_key):
        raise G5RunnerError("EXECUTION_BLOCKED:MAP_MEMBERSHIP_ALIGNMENT")

    for key, observation in map_by_key.items():
        row = feature_by_key[key]
        if (
            observation.fold_id not in {"TRAIN", "DEVELOPMENT", "VALIDATION"}
            or row.get("partition") != observation.fold_id
            or row.get("source_local_event_start") != observation.source_local_event_start
            or row.get("availability") != "COMPLETED_DRAFT_AVAILABLE_AT_OR_BEFORE_EVENT_START"
            or row.get("unavailability") is not None
        ):
            raise G5RunnerError("EXECUTION_BLOCKED:FOLD_TIME_OR_FEATURE_ALIGNMENT")
        expected = {
            (
                player.game_side,
                player.role,
                player.source_player_id,
                player.source_team_id,
            )
            for player in observation.player_observations
        }
        picks = row.get("picks")
        if not isinstance(picks, list) or len(picks) != 10:
            raise G5RunnerError("EXECUTION_BLOCKED:PICK_IDENTITY_ALIGNMENT")
        actual = {
            (
                pick.get("source_side"),
                pick.get("role"),
                pick.get("source_player_id"),
                pick.get("source_team_id"),
            )
            for pick in picks
        }
        champions = [pick.get("stable_champion_id") for pick in picks]
        if (
            actual != expected
            or len(actual) != 10
            or any(not isinstance(champion, str) or not champion for champion in champions)
            or len(set(champions)) != 10
        ):
            raise G5RunnerError("EXECUTION_BLOCKED:PICK_IDENTITY_ALIGNMENT")

    assignments = cluster_artifact.get("assignments")
    if not isinstance(assignments, list):
        raise G5RunnerError("EXECUTION_BLOCKED:CLUSTER_ASSIGNMENT_UNAVAILABLE")
    cluster_by_key: dict[str, str] = {}
    for assignment in assignments:
        key = assignment.get("game_id")
        if key not in map_by_key:
            continue
        cluster_key = assignment.get("dependence_cluster_id")
        if key in cluster_by_key or not isinstance(cluster_key, str) or not cluster_key:
            raise G5RunnerError("EXECUTION_BLOCKED:CLUSTER_MEMBERSHIP_ALIGNMENT")
        cluster_by_key[key] = cluster_key
    if set(cluster_by_key) != set(map_by_key):
        raise G5RunnerError("EXECUTION_BLOCKED:CLUSTER_MEMBERSHIP_ALIGNMENT")
    artifact_sha = cluster_artifact.get("artifact_sha256")
    if artifact_sha != contract.CLUSTERS["artifact_canonical_sha256"]:
        raise G5RunnerError("EXECUTION_BLOCKED:CLUSTER_IDENTITY_MISMATCH")
    return AlignedInputs(
        maps=maps,
        feature_rows=tuple(feature_by_key[item.source_game_id] for item in maps),
        clusters=cluster_by_key,
        feature_manifest=feature_manifest,
        cluster_artifact_sha256=artifact_sha,
    )


def _latent_from_states(states: Mapping[str, Any], observation: Any) -> tuple[float, float]:
    blue = tuple(item.source_player_id for item in observation.player_observations if item.game_side == "blue")
    red = tuple(item.source_player_id for item in observation.player_observations if item.game_side == "red")
    mean = sum(states[item].mean for item in blue) / 5.0 - sum(states[item].mean for item in red) / 5.0
    variance = sum(states[item].variance for item in blue + red) / 25.0
    if not math.isfinite(mean) or not math.isfinite(variance) or variance < 0.0:
        raise G5RunnerError("EXECUTION_BLOCKED:B0_UNCERTAINTY_UNAVAILABLE")
    return float(mean), float(variance)


def build_b0_scores(aligned: AlignedInputs) -> tuple[tuple[OutcomeFreeMap, ...], tuple[EvaluationLedger, ...]]:
    """Score TRAIN pre-update, freeze end-TRAIN STATIC, then score DEV/VAL."""

    from lol_kills.v2.ratings.player import private_development_runner as g2
    from lol_kills.v2.ratings.player.model import posterior_predictive_expected_result

    candidate = g2.Candidate("static_baseline", "STATIC", None, 0.0)
    maps_by_key = {item.source_game_id: item for item in aligned.maps}
    train = [item for item in aligned.maps if item.fold_id == "TRAIN"]
    later = [item for item in aligned.maps if item.fold_id in {"DEVELOPMENT", "VALIDATION"}]
    if not train or not later:
        raise G5RunnerError("EXECUTION_BLOCKED:FOLD_COVERAGE")
    if [
        (item.source_local_event_start, item.source_game_id) for item in train
    ] != sorted((item.source_local_event_start, item.source_game_id) for item in train):
        raise G5RunnerError("EXECUTION_BLOCKED:B0_TRAIN_CHRONOLOGY")
    train_ids = [item.source_game_id for item in train]
    train_id_set = set(train_ids)
    for target in train:
        if any(identifier not in train_id_set for identifier in target.ordered_origin_map_ids):
            raise G5RunnerError("EXECUTION_BLOCKED:B0_TRAIN_ORIGIN_FOLD_LEAKAGE")
    for target in later:
        origins = tuple(target.ordered_origin_map_ids)
        if any(identifier not in origins for identifier in train_ids):
            raise G5RunnerError("EXECUTION_BLOCKED:B0_TRAIN_ORIGIN_NOT_ELIGIBLE_FOR_LATER_TARGET")
        target_time = g2._time(target.source_local_event_start)
        for identifier in train_ids:
            origin = maps_by_key[identifier]
            if (
                origin.source_series_id == target.source_series_id
                or (target_time - g2._time(origin.source_local_event_start)).total_seconds() <= 48.0 * 3600.0
            ):
                raise G5RunnerError("EXECUTION_BLOCKED:B0_TRAIN_ORIGIN_NOT_ELIGIBLE_FOR_LATER_TARGET")

    states: dict[str, Any] = {}
    raw_scores: dict[str, tuple[float, float, float]] = {}
    for observation in train:
        exact_states = g2._state_for_exact_origins(
            maps=maps_by_key,
            origin_ids=tuple(observation.ordered_origin_map_ids),
            origin_sha256=observation.ordered_origin_sha256,
            target=observation,
            candidate=candidate,
        )
        probability, _uncertainty, transitioned = g2._predict(exact_states, observation, candidate)
        mean, variance = _latent_from_states(transitioned, observation)
        accepted_probability = posterior_predictive_expected_result(mean, variance)
        complement = posterior_predictive_expected_result(-mean, variance)
        if abs(probability - accepted_probability) > 1e-15 or abs(accepted_probability + complement - 1.0) > 1e-12:
            raise G5RunnerError("EXECUTION_BLOCKED:B0_PRIMITIVE_RECONCILIATION")
        raw_scores[observation.source_game_id] = (mean, variance, accepted_probability)
        # Own outcome enters only after its prequential score.
        g2._update(states, observation, candidate)

    frozen_states = dict(states)
    for observation in later:
        probability, _uncertainty, transitioned = g2._predict(frozen_states, observation, candidate)
        mean, variance = _latent_from_states(transitioned, observation)
        accepted_probability = posterior_predictive_expected_result(mean, variance)
        complement = posterior_predictive_expected_result(-mean, variance)
        if abs(probability - accepted_probability) > 1e-15 or abs(accepted_probability + complement - 1.0) > 1e-12:
            raise G5RunnerError("EXECUTION_BLOCKED:B0_PRIMITIVE_RECONCILIATION")
        raw_scores[observation.source_game_id] = (mean, variance, accepted_probability)
    if states != frozen_states:
        raise G5RunnerError("EXECUTION_BLOCKED:B0_LATER_STATE_UPDATE")

    feature_by_key = {
        row["source_game_id"]: row
        for row in aligned.feature_rows
    }
    scores: list[OutcomeFreeMap] = []
    labels: list[EvaluationLedger] = []
    for observation in aligned.maps:
        mean, variance, probability = raw_scores[observation.source_game_id]
        row = feature_by_key[observation.source_game_id]
        picks = tuple(
            DraftPick(pick["source_side"], pick["role"], pick["stable_champion_id"])
            for pick in row["picks"]
        )
        scores.append(
            OutcomeFreeMap(
                map_key=observation.source_game_id,
                fold=observation.fold_id,
                source_local_event_start=observation.source_local_event_start,
                cluster_key=aligned.clusters[observation.source_game_id],
                b0_logit_mean=mean,
                b0_logit_variance=variance,
                b0_probability=probability,
                picks=picks,
            )
        )
        labels.append(EvaluationLedger(observation.source_game_id, observation.fold_id, observation.blue_win))
    return tuple(scores), tuple(labels)


def _map_parts(item: SyntheticMap | OutcomeFreeMap) -> tuple[str, str, float, tuple[tuple[str, str, str], ...]]:
    if isinstance(item, SyntheticMap):
        return item.map_key, item.fold, item.b0_logit, item.picks
    return (
        item.map_key,
        item.fold,
        item.b0_logit_mean,
        tuple((pick.source_side, pick.role, pick.stable_champion_id) for pick in item.picks),
    )


def _validate_map(item: SyntheticMap | OutcomeFreeMap) -> None:
    _key, fold, offset, picks = _map_parts(item)
    if fold not in {"TRAIN", "DEVELOPMENT", "VALIDATION"} or not math.isfinite(offset):
        raise G5RunnerError("map boundary")
    if len(picks) != 10 or {side for side, _, _ in picks} != {"blue", "red"}:
        raise G5RunnerError("completed draft required")
    if any({role for side0, role, _ in picks if side0 == side} != set(ROLES) for side in ("blue", "red")):
        raise G5RunnerError("role completion required")
    champions = [champion for _, _, champion in picks]
    if any(not isinstance(champion, str) or not champion for champion in champions) or len(set(champions)) != 10:
        raise G5RunnerError("global champion uniqueness required")


def _design(
    maps: Sequence[SyntheticMap | OutcomeFreeMap],
    vocabulary: Sequence[str] | None = None,
) -> tuple[np.ndarray, tuple[str, ...], tuple[tuple[str, str], ...]]:
    train = [item for item in maps if _map_parts(item)[1] == "TRAIN"]
    vocab = tuple(
        sorted(
            vocabulary if vocabulary is not None else {
                champion
                for item in train
                for _, _, champion in _map_parts(item)[3]
            },
            key=lambda value: value.encode("utf-8"),
        )
    )
    cells = tuple(
        sorted(
            {
                (role, champion)
                for item in train
                for _, role, champion in _map_parts(item)[3]
            },
            key=lambda value: (ROLES.index(value[0]), value[1].encode("utf-8")),
        )
    )
    index: dict[tuple[str, ...], int] = {
        ("mu", champion): offset for offset, champion in enumerate(vocab)
    }
    index.update({
        ("delta", role, champion): len(vocab) + offset
        for offset, (role, champion) in enumerate(cells)
    })
    matrix = np.zeros((len(maps), len(vocab) + len(cells)), dtype=float)
    for row_index, item in enumerate(maps):
        _validate_map(item)
        for side, role, champion in _map_parts(item)[3]:
            sign = 1.0 if side == "blue" else -1.0
            if champion not in vocab:
                raise G5RunnerError("EXECUTION_BLOCKED:CHAMPION_ABSENT_FROM_TRAIN")
            matrix[row_index, index[("mu", champion)]] += sign
            delta = ("delta", role, champion)
            if delta in index:
                matrix[row_index, index[delta]] += sign
    return matrix, vocab, cells


def objective_gradient_hessian(
    beta: np.ndarray,
    x: np.ndarray,
    offset: np.ndarray,
    y: np.ndarray,
    penalty: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    arrays = tuple(np.asarray(value, dtype=float) for value in (beta, x, offset, y, penalty))
    beta, x, offset, y, penalty = arrays
    if (
        x.ndim != 2
        or beta.ndim != 1
        or offset.ndim != 1
        or y.ndim != 1
        or penalty.ndim != 1
        or x.shape != (len(y), len(beta))
        or len(offset) != len(y)
        or len(penalty) != len(beta)
        or not all(np.all(np.isfinite(value)) for value in arrays)
        or not np.all((y == 0.0) | (y == 1.0))
        or not np.all(penalty > 0.0)
    ):
        raise G5RunnerError("EXECUTION_BLOCKED:SOLVER_OR_OBJECTIVE_FAILURE")
    z = offset + x @ beta
    probability = expit(z)
    objective = float(
        np.sum(np.logaddexp(0.0, z) - y * z)
        + 0.5 * beta @ (penalty * beta)
    )
    gradient = x.T @ (probability - y) + penalty * beta
    hessian = x.T @ ((probability * (1.0 - probability))[:, None] * x) + np.diag(penalty)
    if not math.isfinite(objective) or not np.all(np.isfinite(gradient)) or not np.all(np.isfinite(hessian)):
        raise G5RunnerError("EXECUTION_BLOCKED:SOLVER_OR_OBJECTIVE_FAILURE")
    return objective, gradient, hessian


def fit_d1_train(
    maps: Sequence[SyntheticMap | OutcomeFreeMap],
    labels: Mapping[str, int],
) -> dict[str, Any]:
    train = [item for item in maps if _map_parts(item)[1] == "TRAIN"]
    keys = {_map_parts(item)[0] for item in train}
    if not train or set(labels) != keys or any(type(value) is not int or value not in (0, 1) for value in labels.values()):
        raise G5RunnerError("strict TRAIN label separation")
    x, vocabulary, cells = _design(train)
    y = np.asarray([labels[_map_parts(item)[0]] for item in train], dtype=float)
    offset = np.asarray([_map_parts(item)[2] for item in train], dtype=float)
    penalty = np.asarray([12.5] * len(vocabulary) + [50.0] * len(cells), dtype=float)
    start = np.zeros(x.shape[1], dtype=float)

    def fun(beta: np.ndarray) -> tuple[float, np.ndarray]:
        objective, gradient, _hessian = objective_gradient_hessian(beta, x, offset, y, penalty)
        return objective, gradient

    solved = minimize(
        fun,
        start,
        jac=True,
        method="L-BFGS-B",
        bounds=None,
        options={"maxiter": 1000, "gtol": 1e-8, "ftol": 1e-12, "maxls": 50},
    )
    beta = np.asarray(solved.x, dtype=float)
    objective, gradient, hessian = objective_gradient_hessian(beta, x, offset, y, penalty)
    gradient_inf = float(np.linalg.norm(gradient, np.inf))
    if not solved.success or gradient_inf > 1e-6:
        raise G5RunnerError("EXECUTION_BLOCKED:SOLVER_OR_OBJECTIVE_FAILURE")
    if not np.array_equal(hessian, hessian.T) and not np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12):
        raise G5RunnerError("EXECUTION_BLOCKED:CONDITIONAL_COVARIANCE_UNAVAILABLE")
    try:
        cholesky = np.linalg.cholesky(hessian)
        identity = np.eye(hessian.shape[0], dtype=float)
        covariance = np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, identity))
    except np.linalg.LinAlgError as error:
        raise G5RunnerError("EXECUTION_BLOCKED:CONDITIONAL_COVARIANCE_UNAVAILABLE") from error
    if (
        not np.all(np.isfinite(covariance))
        or np.any(np.diag(covariance) < 0.0)
        or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
    ):
        raise G5RunnerError("EXECUTION_BLOCKED:CONDITIONAL_COVARIANCE_UNAVAILABLE")
    return {
        "beta": beta,
        "covariance": covariance,
        "vocabulary": vocabulary,
        "cells": cells,
        "objective": objective,
        "gradient_inf": gradient_inf,
        "hessian": hessian,
        "solver": {
            "iterations": int(getattr(solved, "nit", 0)),
            "function_evaluations": int(getattr(solved, "nfev", 0)),
            "message": str(solved.message),
        },
    }


def _score_d1_math(item: SyntheticMap | OutcomeFreeMap, fit: Mapping[str, Any]) -> dict[str, Any]:
    _validate_map(item)
    vocabulary = tuple(fit["vocabulary"])
    cells = tuple(tuple(value) for value in fit["cells"])
    beta = np.asarray(fit["beta"], dtype=float)
    covariance = np.asarray(fit["covariance"], dtype=float)
    vector = np.zeros(len(beta), dtype=float)
    contributions: list[float] = []
    prior_only: list[tuple[str, str, str]] = []
    for side, role, champion in sorted(
        _map_parts(item)[3],
        key=lambda value: (value[0], ROLES.index(value[1])),
    ):
        if champion not in vocabulary:
            raise G5RunnerError("EXECUTION_BLOCKED:CHAMPION_ABSENT_FROM_TRAIN")
        sign = 1.0 if side == "blue" else -1.0
        slot = np.zeros(len(beta), dtype=float)
        slot[vocabulary.index(champion)] = sign
        cell = (role, champion)
        if cell in cells:
            slot[len(vocabulary) + cells.index(cell)] = sign
        else:
            prior_only.append((side, role, champion))
        vector += slot
        contributions.append(float(slot @ beta))
    if len(prior_only) != len(set(prior_only)):
        raise G5RunnerError("EXECUTION_BLOCKED:PRIOR_ONLY_ROLE_DELTA_DUPLICATED")
    increment = float(vector @ beta)
    error = abs(math.fsum(contributions) - increment)
    if error > 1e-12:
        raise G5RunnerError("EXECUTION_BLOCKED:CONTRIBUTION_RECONCILIATION_FAILURE")
    variance = float(vector @ covariance @ vector + 0.01 * len(prior_only))
    if not math.isfinite(variance) or variance < 0.0:
        raise G5RunnerError("EXECUTION_BLOCKED:CONDITIONAL_COVARIANCE_UNAVAILABLE")
    return {
        "increment": increment,
        "variance": variance,
        "prior_only": tuple(sorted(prior_only)),
        "contributions": tuple(contributions),
        "reconciliation_error": error,
        "fitted_vector": vector,
    }


def score_d1(item: SyntheticMap | OutcomeFreeMap, fit: Mapping[str, Any]) -> dict[str, Any]:
    return _score_d1_math(item, fit)


def _logloss(label: int, logit: float) -> float:
    if type(label) is not int or label not in (0, 1) or not math.isfinite(logit):
        raise G5RunnerError("EXECUTION_BLOCKED:NONFINITE_EVALUATION")
    return float(np.logaddexp(0.0, logit) - label * logit)


def _probability_logloss(label: int, probability: float) -> float:
    if (
        type(label) is not int
        or label not in (0, 1)
        or isinstance(probability, bool)
        or not isinstance(probability, (int, float))
        or not math.isfinite(float(probability))
        or not 0.0 < float(probability) < 1.0
    ):
        raise G5RunnerError("EXECUTION_BLOCKED:NONFINITE_EVALUATION")
    return -math.log(float(probability) if label == 1 else 1.0 - float(probability))


def _bootstrap_replicate(cluster_summaries: Mapping[str, Mapping[str, object]], seed: int) -> float:
    from lol_kills.v2.draft.interactions.series_cluster_proxy import (
        map_weighted_cluster_bootstrap_replicate,
    )

    payload = map_weighted_cluster_bootstrap_replicate(cluster_summaries, seed=seed)
    value = payload.get("replicate")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise G5RunnerError("EXECUTION_BLOCKED:BOOTSTRAP_FAILURE")
    return float(value)


def validation_bootstrap(
    deltas: Sequence[tuple[str, str, float]],
) -> tuple[float, np.ndarray]:
    """Run exactly 2,000 bound map-weighted cluster bootstrap calls."""

    if not deltas or len({map_key for map_key, _cluster, _delta in deltas}) != len(deltas):
        raise G5RunnerError("EXECUTION_BLOCKED:BOOTSTRAP_MEMBERSHIP")
    grouped: dict[str, list[float]] = defaultdict(list)
    for _map_key, cluster_key, delta in deltas:
        if not isinstance(cluster_key, str) or not cluster_key or not math.isfinite(delta):
            raise G5RunnerError("EXECUTION_BLOCKED:BOOTSTRAP_MEMBERSHIP")
        grouped[cluster_key].append(float(delta))
    summaries = {
        cluster_key: {
            "cluster_delta_total": math.fsum(values),
            "cluster_map_count": len(values),
        }
        for cluster_key, values in grouped.items()
    }
    try:
        replicates = np.asarray(
            [
                _bootstrap_replicate(summaries, BOOTSTRAP_BASE_SEED + index)
                for index in range(BOOTSTRAP_REPLICATES)
            ],
            dtype=float,
        )
        if replicates.shape != (BOOTSTRAP_REPLICATES,) or not np.all(np.isfinite(replicates)):
            raise G5RunnerError("EXECUTION_BLOCKED:BOOTSTRAP_FAILURE")
        lower = float(np.quantile(replicates, 0.05, method="linear"))
    except G5RunnerError:
        raise
    except Exception as error:
        raise G5RunnerError("EXECUTION_BLOCKED:BOOTSTRAP_FAILURE") from error
    if not math.isfinite(lower):
        raise G5RunnerError("EXECUTION_BLOCKED:BOOTSTRAP_FAILURE")
    return lower, replicates


def d1_validation_wins(mean_improvement: float, lower_bound: float) -> bool:
    if not math.isfinite(mean_improvement) or not math.isfinite(lower_bound):
        raise G5RunnerError("EXECUTION_BLOCKED:NONFINITE_EVALUATION")
    return bool(mean_improvement >= 0.005 and lower_bound > 0.0)


def _membership_hashes(maps: Sequence[OutcomeFreeMap], aligned: AlignedInputs) -> dict[str, str]:
    by_fold = {
        fold: [item.map_key for item in maps if item.fold == fold]
        for fold in ("TRAIN", "DEVELOPMENT", "VALIDATION")
    }
    return {
        "all_maps_sha256": _sha([item.map_key for item in maps]),
        "TRAIN_maps_sha256": _sha(by_fold["TRAIN"]),
        "DEVELOPMENT_maps_sha256": _sha(by_fold["DEVELOPMENT"]),
        "VALIDATION_maps_sha256": _sha(by_fold["VALIDATION"]),
        "cluster_membership_sha256": _sha([
            [item.map_key, item.cluster_key] for item in maps
        ]),
        "origin_membership_sha256": _sha([
            [item.source_game_id, list(item.ordered_origin_map_ids)]
            for item in aligned.maps
        ]),
        "feature_membership_sha256": _sha([
            row["source_game_id"] for row in aligned.feature_rows
        ]),
    }


def _base_evidence(
    *,
    state: str,
    blocker: str | None,
    selected_candidate: str | None,
    maps: Sequence[OutcomeFreeMap],
    aligned: AlignedInputs,
    fit: Mapping[str, Any],
    development: Mapping[str, Any],
    validation: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    prior_summary: Mapping[str, Any],
    invariance: Mapping[str, Any],
    reconciliation_error: float | None,
    score_subject: Mapping[str, Any],
    winner: WinnerAggregate | None,
) -> AggregateEvidence:
    counts = {
        "maps": len(maps),
        "picks": len(maps) * 10,
        "TRAIN": sum(item.fold == "TRAIN" for item in maps),
        "DEVELOPMENT": sum(item.fold == "DEVELOPMENT" for item in maps),
        "VALIDATION": sum(item.fold == "VALIDATION" for item in maps),
    }
    hessian = np.asarray(fit["hessian"])
    return AggregateEvidence(
        state=state,
        blocker=blocker,
        selected_candidate=selected_candidate,
        counts=counts,
        membership_hashes=_membership_hashes(maps, aligned),
        source_and_feature_review_pins={
            "G1_manifest_sha256": contract.G1["manifest_sha256"],
            "G1_rows_sha256": contract.G1["rows_sha256"],
            "selected_target_sha256": contract.G1["selected_target_sha256"],
            "split_payload_sha256": contract.G1["split_payload_sha256"],
            "feature_manifest_sha256": contract.G1_FEATURES["manifest_canonical_sha256"],
            "feature_rows_raw_sha256": contract.G1_FEATURES["rows_raw_sha256"],
            "feature_rows_canonical_sha256": contract.G1_FEATURES["rows_canonical_sha256"],
            "feature_review_sha256": contract.G1_FEATURES["independent_review_canonical_sha256"],
            "cluster_artifact_sha256": aligned.cluster_artifact_sha256,
        },
        G2_core_pins={
            "runner_raw_sha256": contract.G2["runner_raw_sha256"],
            "model_raw_sha256": contract.G2["model_raw_sha256"],
            "artifact_raw_sha256": contract.G2["artifact_raw_sha256"],
            "artifact_canonical_sha256": contract.G2["artifact_canonical_sha256"],
            "candidate": "static_baseline",
        },
        development_metric=dict(development),
        validation_metric=dict(validation),
        bootstrap=dict(bootstrap),
        objective_gradient_hessian_diagnostics={
            "objective": float(fit["objective"]),
            "gradient_infinity_norm": float(fit["gradient_inf"]),
            "hessian_dimension": int(hessian.shape[0]),
            "hessian_symmetric_atol_1e_12": bool(np.allclose(hessian, hessian.T, rtol=0.0, atol=1e-12)),
            "hessian_positive_definite": True,
        },
        solver_diagnostics={
            "status": "CONVERGED",
            "method": "L-BFGS-B",
            "analytic_jacobian": True,
            "iterations": fit["solver"]["iterations"],
            "function_evaluations": fit["solver"]["function_evaluations"],
            "message_sha256": _sha(fit["solver"]["message"]),
        },
        uncertainty={
            "B0_latent_mean_available": True,
            "B0_latent_variance_available": True,
            "D1_conditional_covariance": "AVAILABLE",
            "total_B0_plus_D1_interval": "PROHIBITED",
        },
        prior_only_variance_components=dict(prior_summary),
        coverage_and_prior_only_flags={
            "complete_maps": len(maps) == EXPECTED_MAPS,
            "complete_picks": len(maps) * 10 == EXPECTED_PICKS,
            "champion_absent_from_TRAIN": False,
            "prior_only_role_delta_used": bool(prior_summary["role_delta_count"]),
            "final_holdout_reads": 0,
        },
        invariance_tests=dict(invariance),
        contribution_reconciliation={
            "status": "PASSED",
            "absolute_tolerance": 1e-12,
            "max_absolute_error": reconciliation_error,
        },
        score_subject=dict(score_subject),
        context={
            "status": "UNAVAILABLE",
            "blocker": "CONTEXTUAL_EXACT_FIVE_OR_PLAYER_CHAMPION_EVIDENCE_UNAVAILABLE",
        },
        winner=winner,
    )


def _measure_invariances(
    maps: Sequence[OutcomeFreeMap],
    fit: Mapping[str, Any],
) -> dict[str, Any]:
    from lol_kills.v2.ratings.player.model import posterior_predictive_expected_result

    side_max = 0.0
    order_max = 0.0
    for item in maps:
        original = _score_d1_math(item, fit)
        swapped = OutcomeFreeMap(
            map_key=item.map_key,
            fold=item.fold,
            source_local_event_start=item.source_local_event_start,
            cluster_key=item.cluster_key,
            b0_logit_mean=-item.b0_logit_mean,
            b0_logit_variance=item.b0_logit_variance,
            b0_probability=posterior_predictive_expected_result(
                -item.b0_logit_mean, item.b0_logit_variance
            ),
            picks=tuple(
                DraftPick("red" if pick.source_side == "blue" else "blue", pick.role, pick.stable_champion_id)
                for pick in item.picks
            ),
        )
        swapped_score = _score_d1_math(swapped, fit)
        original_d1 = posterior_predictive_expected_result(
            item.b0_logit_mean + original["increment"], item.b0_logit_variance
        )
        swapped_d1 = posterior_predictive_expected_result(
            swapped.b0_logit_mean + swapped_score["increment"], swapped.b0_logit_variance
        )
        side_max = max(
            side_max,
            abs(item.b0_probability + swapped.b0_probability - 1.0),
            abs(original_d1 + swapped_d1 - 1.0),
            abs(original["increment"] + swapped_score["increment"]),
            abs(original["variance"] - swapped_score["variance"]),
        )
        reordered = OutcomeFreeMap(
            map_key=item.map_key,
            fold=item.fold,
            source_local_event_start=item.source_local_event_start,
            cluster_key=item.cluster_key,
            b0_logit_mean=item.b0_logit_mean,
            b0_logit_variance=item.b0_logit_variance,
            b0_probability=item.b0_probability,
            picks=tuple(reversed(item.picks)),
        )
        reordered_score = _score_d1_math(reordered, fit)
        order_max = max(
            order_max,
            abs(original["increment"] - reordered_score["increment"]),
            abs(original["variance"] - reordered_score["variance"]),
            0.0 if original["prior_only"] == reordered_score["prior_only"] else math.inf,
        )
    if side_max > 1e-12 or order_max > 1e-12:
        raise G5RunnerError("EXECUTION_BLOCKED:INVARIANCE_FAILURE")
    return {
        "side_swap": {
            "status": "PASSED",
            "map_count": len(maps),
            "absolute_tolerance": 1e-12,
            "max_absolute_error": side_max,
        },
        "record_order": {
            "status": "PASSED",
            "map_count": len(maps),
            "absolute_tolerance": 1e-12,
            "max_absolute_error": order_max,
        },
        "role_relabel": {"status": "NOT_INVARIANT_BY_CONTRACT"},
    }


def _prior_aggregate(
    validation_scores: Sequence[tuple[OutcomeFreeMap, Mapping[str, Any]]],
    fit: Mapping[str, Any],
) -> tuple[dict[str, Any], float]:
    if not validation_scores:
        empty = _sha([])
        return {
            "status": "NOT_EVALUATED",
            "role_delta_count": 0,
            "variance_per_coordinate": 0.01,
            "total_variance": 0.0,
            "mean_score_aggregate_variance": 0.0,
            "conditional_mean_logit_variance": 0.0,
            "slot_membership_sha256": empty,
            "signed_exposure_sha256": empty,
            "coordinate_exposure_witness": [],
        }, 0.0
    coordinate_exposures: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"blue": 0, "red": 0}
    )
    fitted_vectors: list[np.ndarray] = []
    count = len(validation_scores)
    for _item, scored in validation_scores:
        fitted_vectors.append(np.asarray(scored["fitted_vector"], dtype=float))
        for side, role, champion in scored["prior_only"]:
            coordinate_exposures[(role, champion)][side] += 1
    ordered_vectors = sorted(fitted_vectors, key=lambda vector: vector.tobytes())
    mean_vector = np.sum(np.stack(ordered_vectors), axis=0) / count
    covariance = np.asarray(fit["covariance"], dtype=float)
    fitted_variance = float(mean_vector @ covariance @ mean_vector)
    coordinates = sorted(coordinate_exposures)
    witness = sorted(
        [
            {
                "coordinate_commitment_sha256": _sha({
                    "domain": "g5-prior-only-role-champion-coordinate:v1",
                    "role": role,
                    "stable_champion_id": champion,
                }),
                "blue_count": coordinate_exposures[(role, champion)]["blue"],
                "red_count": coordinate_exposures[(role, champion)]["red"],
                "net_count": coordinate_exposures[(role, champion)]["blue"]
                - coordinate_exposures[(role, champion)]["red"],
                "validation_map_count": count,
            }
            for role, champion in coordinates
        ],
        key=lambda record: record["coordinate_commitment_sha256"],
    )
    commitments = [record["coordinate_commitment_sha256"] for record in witness]
    signed_exposures = [
        [
            record["coordinate_commitment_sha256"],
            record["blue_count"],
            record["red_count"],
            record["net_count"],
            record["validation_map_count"],
        ]
        for record in witness
    ]
    prior_mean_variance = 0.01 * math.fsum(
        (record["net_count"] / record["validation_map_count"]) ** 2
        for record in witness
    )
    aggregate_variance = fitted_variance + prior_mean_variance
    if not math.isfinite(aggregate_variance) or aggregate_variance < 0.0:
        raise G5RunnerError("EXECUTION_BLOCKED:CONDITIONAL_COVARIANCE_UNAVAILABLE")
    return {
        "status": "EVALUATED",
        "role_delta_count": len(coordinates),
        "variance_per_coordinate": 0.01,
        "total_variance": 0.01 * len(coordinates),
        "mean_score_aggregate_variance": prior_mean_variance,
        "conditional_mean_logit_variance": aggregate_variance,
        "slot_membership_sha256": _sha(commitments),
        "signed_exposure_sha256": _sha(signed_exposures),
        "coordinate_exposure_witness": witness,
    }, aggregate_variance


def compute_aggregate_evidence(aligned: AlignedInputs) -> AggregateEvidence:
    """Run fixture-callable computation without any REAL schema capability."""

    from lol_kills.v2.ratings.player.model import posterior_predictive_expected_result

    maps, ledgers = build_b0_scores(aligned)
    labels_by_fold = {
        fold: {item.map_key: item.label for item in ledgers if item.fold == fold}
        for fold in ("TRAIN", "DEVELOPMENT", "VALIDATION")
    }
    fit = fit_d1_train(maps, labels_by_fold["TRAIN"])
    invariance = _measure_invariances(maps, fit)
    development_maps = [item for item in maps if item.fold == "DEVELOPMENT"]
    development_b0_losses: list[float] = []
    development_d1_losses: list[float] = []
    for item in development_maps:
        scored = score_d1(item, fit)
        label = labels_by_fold["DEVELOPMENT"][item.map_key]
        d1_probability = posterior_predictive_expected_result(
            item.b0_logit_mean + scored["increment"],
            item.b0_logit_variance,
        )
        development_b0_losses.append(_probability_logloss(label, item.b0_probability))
        development_d1_losses.append(_probability_logloss(label, d1_probability))
    if not development_maps:
        raise G5RunnerError("EXECUTION_BLOCKED:FOLD_COVERAGE")
    development_b0 = math.fsum(sorted(development_b0_losses)) / len(development_b0_losses)
    development_d1 = math.fsum(sorted(development_d1_losses)) / len(development_d1_losses)
    development_mean = development_b0 - development_d1
    locked = "D1" if development_mean > 0.0 else "B0"
    development_metric = {
        "locked_candidate": locked,
        "map_count": len(development_maps),
        "evaluations": 1,
        "B0_mean_log_loss": development_b0,
        "D1_mean_log_loss": development_d1,
        "mean_LL_B0_minus_LL_D1": development_mean,
    }
    validation_maps = [item for item in maps if item.fold == "VALIDATION"]
    if not validation_maps:
        raise G5RunnerError("EXECUTION_BLOCKED:FOLD_COVERAGE")
    if locked == "B0":
        validation_b0_losses = [
            _probability_logloss(labels_by_fold["VALIDATION"][item.map_key], item.b0_probability)
            for item in validation_maps
        ]
        validation_b0 = math.fsum(sorted(validation_b0_losses)) / len(validation_b0_losses)
        validation_metric = {
            "locked_candidate": "B0",
            "map_count": len(validation_maps),
            "evaluations": 1,
            "B0_mean_log_loss": validation_b0,
            "locked_candidate_mean_log_loss": validation_b0,
            "mean_LL_B0_minus_LL_locked_candidate": 0.0,
        }
        bootstrap = {
            "status": "NOT_RUN_B0_LOCKED",
            "replicates": 0,
            "base_seed": None,
            "quantile": None,
            "lower_bound": None,
            "map_weighted": True,
        }
        prior_summary, _aggregate_variance = _prior_aggregate((), fit)
        return _base_evidence(
            state="NO_INCREMENTAL_DRAFT_WINNER",
            blocker=None,
            selected_candidate="B0",
            maps=maps,
            aligned=aligned,
            fit=fit,
            development=development_metric,
            validation=validation_metric,
            bootstrap=bootstrap,
            prior_summary=prior_summary,
            invariance=invariance,
            reconciliation_error=0.0,
            score_subject={"status": "WITHHELD_NO_WINNER", "kind": "VALIDATION_COHORT_AGGREGATE", "fold": "VALIDATION", "map_count": len(validation_maps), "weighting": "MAP_EQUAL", "order_invariant": True},
            winner=None,
        )

    validation_deltas: list[tuple[str, str, float]] = []
    validation_clusters = {item.cluster_key for item in validation_maps}
    if any(
        item.cluster_key in validation_clusters and item.fold != "VALIDATION"
        for item in maps
    ):
        raise G5RunnerError("EXECUTION_BLOCKED:BOOTSTRAP_MEMBERSHIP")
    validation_scores: list[tuple[OutcomeFreeMap, Mapping[str, Any]]] = []
    validation_b0_losses: list[float] = []
    validation_d1_losses: list[float] = []
    validation_b0_probabilities: list[float] = []
    validation_d1_probabilities: list[float] = []
    validation_increments: list[float] = []
    max_reconciliation = 0.0
    for item in validation_maps:
        scored = score_d1(item, fit)
        validation_scores.append((item, scored))
        max_reconciliation = max(max_reconciliation, scored["reconciliation_error"])
        label = labels_by_fold["VALIDATION"][item.map_key]
        d1_probability = posterior_predictive_expected_result(
            item.b0_logit_mean + scored["increment"],
            item.b0_logit_variance,
        )
        b0_loss = _probability_logloss(label, item.b0_probability)
        d1_loss = _probability_logloss(label, d1_probability)
        delta = b0_loss - d1_loss
        validation_deltas.append((item.map_key, item.cluster_key, delta))
        validation_b0_losses.append(b0_loss)
        validation_d1_losses.append(d1_loss)
        validation_b0_probabilities.append(item.b0_probability)
        validation_d1_probabilities.append(d1_probability)
        validation_increments.append(scored["increment"])
    validation_b0 = math.fsum(sorted(validation_b0_losses)) / len(validation_b0_losses)
    validation_d1 = math.fsum(sorted(validation_d1_losses)) / len(validation_d1_losses)
    validation_mean = validation_b0 - validation_d1
    lower, _replicates = validation_bootstrap(validation_deltas)
    validation_metric = {
        "locked_candidate": "D1",
        "map_count": len(validation_maps),
        "evaluations": 1,
        "B0_mean_log_loss": validation_b0,
        "locked_candidate_mean_log_loss": validation_d1,
        "mean_LL_B0_minus_LL_locked_candidate": validation_mean,
    }
    bootstrap = {
        "status": "COMPLETED",
        "replicates": BOOTSTRAP_REPLICATES,
        "base_seed": BOOTSTRAP_BASE_SEED,
        "quantile": 0.05,
        "lower_bound": lower,
        "map_weighted": True,
    }
    winner = d1_validation_wins(validation_mean, lower)
    prior_summary, aggregate_variance = _prior_aggregate(validation_scores, fit)
    winner_aggregate: WinnerAggregate | None = None
    if winner:
        mean_b0_probability = math.fsum(sorted(validation_b0_probabilities)) / len(validation_b0_probabilities)
        mean_d1_probability = math.fsum(sorted(validation_d1_probabilities)) / len(validation_d1_probabilities)
        mean_increment = math.fsum(sorted(validation_increments)) / len(validation_increments)
        standard_error = math.sqrt(aggregate_variance)
        winner_aggregate = WinnerAggregate(
            score_subject={"status": "AVAILABLE", "kind": "VALIDATION_COHORT_AGGREGATE", "fold": "VALIDATION", "map_count": len(validation_maps), "weighting": "MAP_EQUAL", "order_invariant": True},
            B0_probability=mean_b0_probability,
            D1_logit_increment=mean_increment,
            neutral_completed_draft_probability=mean_d1_probability,
            probability_increment_over_B0=mean_d1_probability - mean_b0_probability,
            D1_conditional_interval={
                "lower": mean_increment - 1.959963984540054 * standard_error,
                "upper": mean_increment + 1.959963984540054 * standard_error,
                "level": 0.95,
                "scale": "conditional_mean_validation_logit_increment",
            },
        )
    return _base_evidence(
        state="PRIVATE_EXPLORATORY_INCREMENTAL_DRAFT_WINNER" if winner else "NO_INCREMENTAL_DRAFT_WINNER",
        blocker=None,
        selected_candidate="D1",
        maps=maps,
        aligned=aligned,
        fit=fit,
        development=development_metric,
        validation=validation_metric,
        bootstrap=bootstrap,
        prior_summary=prior_summary,
        invariance=invariance,
        reconciliation_error=max_reconciliation,
        score_subject=winner_aggregate.score_subject if winner_aggregate is not None else {"status": "WITHHELD_NO_WINNER", "kind": "VALIDATION_COHORT_AGGREGATE", "fold": "VALIDATION", "map_count": len(validation_maps), "weighting": "MAP_EQUAL", "order_invariant": True},
        winner=winner_aggregate,
    )


def _execute_bound_pipeline() -> AggregateEvidence:
    """The only protected-read path; every loader is concrete and fixed."""

    g1 = _load_accepted_g1()
    feature_manifest, feature_rows = _load_accepted_features()
    clusters = _load_accepted_clusters()
    aligned = align_inputs(g1, feature_manifest, feature_rows, clusters)
    return compute_aggregate_evidence(aligned)


def synthetic_execute(
    maps: Sequence[SyntheticMap],
    train_labels: Mapping[str, int],
    evaluations: Sequence[SyntheticEvaluation],
) -> dict[str, Any]:
    """Scripted synthetic harness; it cannot construct any real schema/state."""

    for item in maps:
        _validate_map(item)
    fit = fit_d1_train(maps, train_labels)
    by_key = {item.map_key: item for item in maps}
    if len(by_key) != len(maps):
        raise G5RunnerError("synthetic map membership")
    evaluation_by_key: dict[str, SyntheticEvaluation] = {}
    for evaluation in evaluations:
        if evaluation.map_key in evaluation_by_key or type(evaluation.label) is not int or evaluation.label not in (0, 1):
            raise G5RunnerError("synthetic evaluation membership")
        evaluation_by_key[evaluation.map_key] = evaluation
    dev = [item for item in evaluations if by_key[item.map_key].fold == "DEVELOPMENT"]
    val = [item for item in evaluations if by_key[item.map_key].fold == "VALIDATION"]
    if not dev or not val or set(evaluation_by_key) != {
        item.map_key for item in maps if item.fold in {"DEVELOPMENT", "VALIDATION"}
    }:
        raise G5RunnerError("synthetic evaluation membership")
    development_deltas = []
    for evaluation in dev:
        item = by_key[evaluation.map_key]
        increment = score_d1(item, fit)["increment"]
        development_deltas.append(
            _logloss(evaluation.label, item.b0_logit)
            - _logloss(evaluation.label, item.b0_logit + increment)
        )
    development_mean = float(np.mean(development_deltas))
    if development_mean <= 0.0:
        state = "SYNTHETIC_SELECTION_B0"
        validation = {"called_once": True, "candidate": "B0", "maps": len(val)}
    else:
        validation_deltas = []
        for evaluation in val:
            item = by_key[evaluation.map_key]
            increment = score_d1(item, fit)["increment"]
            validation_deltas.append(
                _logloss(evaluation.label, item.b0_logit)
                - _logloss(evaluation.label, item.b0_logit + increment)
            )
        state = "SYNTHETIC_SELECTION_D1"
        validation = {
            "called_once": True,
            "candidate": "D1",
            "mean_improvement": float(np.mean(validation_deltas)),
            "maps": len(val),
        }
    unsigned = {
        "schema_version": result.SYNTHETIC_SCHEMA,
        "state": state,
        "development": {
            "mean_logloss_improvement": development_mean,
            "maps": len(dev),
        },
        "validation": validation,
        "solver": {
            "objective": fit["objective"],
            "gradient_inf": fit["gradient_inf"],
        },
        "claim_ceiling": {
            "synthetic_only": True,
            "real_evidence": False,
            "publication": False,
            "prediction": False,
        },
    }
    payload = {**unsigned, "artifact_sha256": result.sha256(unsigned)}
    result.validate_synthetic(payload)
    return payload
