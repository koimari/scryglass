"""Fail-closed private runner for the audited representation-rank assay.

Readiness is deliberately outcome-free: it verifies pinned bytes, then asks
the feature parquet reader for an explicit allow-list that cannot contain the
target.  The target/M0 loader is a separate, future execution boundary and is
never called by readiness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from . import representation_rank_assay as assay
from . import representation_rank_private_result as private_result
from .oe_nuisance_baseline import (
    validate_artifact as validate_nuisance_artifact,
)
from .oe_target_authority import load_and_require_exact_human_authority
from .oe_target_evidence import validate_evidence, validate_split
from . import representation_rank_runner_review_authority as runner_review_authority


CONTRACT_SCHEMA_ID = "scryglass.representation-rank-private-run-contract.v1"
PENDING_REPORT_SCHEMA_ID = (
    "scryglass.representation-rank-private-run-pending-report.v1"
)
NONHOLDOUT_SPLITS = ("train", "development", "validation")
ROLE_COLUMNS = tuple(
    f"{side}_{role}_stable_champion_id"
    for side in ("blue", "red")
    for role in assay.CANONICAL_POSITIONS
)
SAFE_FEATURE_COLUMNS = (
    "game_id",
    "dependence_cluster_id",
    "split",
    "oe_date_naive",
    "canonical_league",
    *ROLE_COLUMNS,
)
TARGET_M0_COLUMNS = (
    "game_id",
    "dependence_cluster_id",
    "split",
    "oe_date_naive",
    "p_blue_win_nuisance_oof",
    "y_blue_win",
)
SAFE_FIT_AVAILABILITY_COLUMNS = (
    "game_id",
    "dependence_cluster_id",
    "split",
    "oe_date_naive",
    "prediction_fold_month_naive",
    "fit_maximum_date_naive",
    "fit_rows",
    "fit_dependence_clusters",
    "fit_cluster_membership_sha256",
    "selected_regularization_C",
    "selected_nuisance_method",
)
TARGET_MEMBERSHIP_COLUMNS = (
    "game_id",
    "dependence_cluster_id",
    "split",
    "oe_date_naive",
)
PINNED_OOF_ROWS = 5702
PINNED_OOF_MEMBERSHIP_SHA256 = (
    "76f7d44585920abf4e1dd37ba478e3849079f45430910444e78dd28b1a8bfa4b"
)
PINNED_FINAL_ROWS = 361
# A PASS remains impossible until an independent review artifact is separately
# accepted and its immutable byte identity is pinned in reviewed source.
FORBIDDEN_READINESS_COLUMNS = frozenset(
    {
        "y_blue_win",
        "blue_result",
        "red_result",
        "p_blue_win_nuisance_oof",
        "p_blue_win_richer_candidate_oof",
        "p_blue_win_intercept_oof",
    }
)
DEFAULT_CONTRACT_PATH = Path(
    "data/lol/v2/models/draft-interactions/"
    "representation-rank-private-run-contract.json"
)
DEFAULT_PENDING_REPORT_PATH = Path(
    "data/lol/v2/models/draft-interactions/"
    "representation-rank-private-run-pending-report.json"
)


class PrivateRunnerError(ValueError):
    """Raised when the private runner cannot prove its complete contract."""


@dataclass(frozen=True)
class FeatureEnvelope:
    domain: assay.FeatureDomain
    ordered_rows: tuple[tuple[Any, ...], ...]
    selected_columns: tuple[str, ...]
    source_raw_sha256: str
    logical_rows_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class TargetM0Envelope:
    target_domain: assay.TargetDomain
    m0_by_game_id: tuple[tuple[str, float], ...]
    ordered_rows: tuple[tuple[Any, ...], ...]
    source_raw_sha256: str
    logical_rows_sha256: str
    ordered_logical_rows_sha256: str
    artifact_sha256: str


@dataclass(frozen=True)
class FixedPointSupport:
    ordered_fit_game_ids: tuple[str, ...]
    eligible_nodes: tuple[bool, ...]
    convergence_checks: int
    changing_rounds: int
    artifact_sha256: str


@dataclass(frozen=True)
class FamilyResult:
    status: str
    selected: float | None
    rows: tuple[dict[str, Any], ...]
    fallback: str
    reason_code: str | None


class StatisticalRunInconclusive(RuntimeError):
    """A scientifically valid terminal fallback, not a source/programmer fault."""

    def __init__(self, reason_code: str = "fit_unavailable") -> None:
        if reason_code != "fit_unavailable":
            raise PrivateRunnerError("statistical failure reason code invalid")
        self.reason_code = reason_code
        super().__init__(reason_code)


@dataclass(frozen=True)
class MonthRunContext:
    split: str
    calendar_month: str
    eligibility_binding: assay.EligibilityBinding
    prediction_game_ids: tuple[str, ...]
    prediction_cluster_ids: tuple[str, ...]
    m0_probability: np.ndarray
    membership_sha256: str
    prepared_fold: assay.PreparedFold | None
    coverage_report: Mapping[str, Any]


@dataclass(frozen=True)
class FitRequest:
    sequence: int
    stage: str
    split: str
    calendar_month: str
    family: str
    fit_role: str
    width: int
    lambda_ally: float
    lambda_enemy: float
    context: MonthRunContext
    target_domain: assay.TargetDomain
    verified_nuisance_oof: Mapping[str, float]


@dataclass(frozen=True)
class FitExecution:
    prediction_by_game_id: tuple[tuple[str, float], ...]
    objective: float
    max_gradient: float
    converged_starts: int
    stability_rms: float


def _resolve(root: Path, locator: str) -> Path:
    path = Path(locator)
    return path if path.is_absolute() else root / path


def _load_canonical(path: Path, *, expected_raw: str | None = None) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise PrivateRunnerError(f"{path} is not a regular file")
    raw = path.read_bytes()
    observed = assay.raw_sha256(path)
    if expected_raw is not None and observed != expected_raw:
        raise PrivateRunnerError(f"{path} source bytes changed")
    payload = json.loads(raw)
    if raw != assay.canonical_bytes(payload):
        raise PrivateRunnerError(f"{path} is not canonical JSON")
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed is not None and claimed != assay.canonical_sha256(unsigned):
        raise PrivateRunnerError(f"{path} payload hash changed")
    return payload


def load_contract(
    path: Path = DEFAULT_CONTRACT_PATH, *, root: Path = Path.cwd()
) -> dict[str, Any]:
    resolved = path if path.is_absolute() else root / path
    payload = _load_canonical(resolved)
    validate_contract(payload)
    return payload


_REVIEW_CORE_EXCLUDED_FIELDS = frozenset(
    {
        "artifact_sha256",
        "runner_review_core_sha256",
        "runner_review_status",
        "status",
        "runner_review_permit",
    }
)


def contract_review_core_sha256(payload: Mapping[str, Any]) -> str:
    """Digest every immutable contract field while excluding activation state."""
    return assay.canonical_sha256(
        {
            key: value
            for key, value in payload.items()
            if key not in _REVIEW_CORE_EXCLUDED_FIELDS
        }
    )


def validate_contract(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != assay.canonical_sha256(unsigned):
        raise PrivateRunnerError("private-run contract hash mismatch")
    fixed = {
        "schema_id": CONTRACT_SCHEMA_ID,
        "default_cli_action": "refuse_fitting",
        "development_only": True,
        "real_target_loader_invoked": False,
        "target_rows_loaded": False,
        "outcome_columns_loaded": False,
        "final_target_loaded": False,
        "candidate_fit_started": False,
        "final_temporal_holdout": "sealed_prohibited",
        "failure_policy": "assay_inconclusive_and_fallback_M0",
    }
    if any(payload.get(key) != value for key, value in fixed.items()):
        raise PrivateRunnerError("private-run fixed boundary changed")
    review = payload.get("runner_review_status")
    expected_status = (
        "private_runner_review_pass"
        if review == "PASS"
        else "private_runner_pending_review"
    )
    if review not in {"PENDING", "PASS"} or payload.get("status") != expected_status:
        raise PrivateRunnerError("runner review state invalid")
    if (
        review == "PASS"
        and runner_review_authority.PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256 is None
    ):
        raise PrivateRunnerError("independent runner-review permit is not pinned")
    if review == "PENDING" and payload.get("runner_review_permit") is not None:
        raise PrivateRunnerError("pending contract cannot contain a review permit")
    if review == "PASS":
        permit = payload.get("runner_review_permit")
        if (
            not isinstance(permit, Mapping)
            or set(permit) != {"locator", "raw_sha256"}
            or permit["raw_sha256"]
            != runner_review_authority.PINNED_RUNNER_REVIEW_PERMIT_RAW_SHA256
        ):
            raise PrivateRunnerError("runner-review permit identity invalid")
    if payload.get("approved_actions") != {
        "reviewer_identity": "KOI_MARI",
        "model_fit": True,
        "rank_selection": True,
        "publication": False,
        "production": False,
        "reliability": False,
        "promotion": False,
        "sota_claim": False,
    }:
        raise PrivateRunnerError("private authority action boundary changed")
    if payload.get("safe_feature_projection") != list(SAFE_FEATURE_COLUMNS):
        raise PrivateRunnerError("safe feature projection changed")
    if payload.get("safe_fit_availability_projection") != list(
        SAFE_FIT_AVAILABILITY_COLUMNS
    ):
        raise PrivateRunnerError("safe fit-availability projection changed")
    if set(payload.get("audited_shell", {})) != {
        "module",
        "generator",
        "tests",
        "runner_module",
        "runner_tests",
        "result_module",
        "result_tests",
        "coordinator_tests",
        "config",
        "report",
    }:
        raise PrivateRunnerError("audited shell identity incomplete")
    if payload.get("runner_review_core_sha256") != contract_review_core_sha256(
        payload
    ):
        raise PrivateRunnerError("runner review core identity changed")
    if payload.get("runner_review_authority_root") != {
        "locator": (
            "lol_kills/v2/draft/interactions/"
            "representation_rank_runner_review_authority.py"
        ),
        "status": "pending_unpinned",
    }:
        raise PrivateRunnerError("runner review authority root boundary changed")
    for source in (
        *payload["audited_shell"].values(),
        *payload.get("source_identity", {}).values(),
    ):
        if (
            not isinstance(source, Mapping)
            or set(source) < {"locator", "raw_sha256"}
            or len(str(source["raw_sha256"])) != 64
        ):
            raise PrivateRunnerError("pinned source identity invalid")
    paths = payload.get("future_private_artifacts")
    if (
        paths
        != {
            "aggregate_only": True,
            "expected_git_ignored": True,
            "manifest_locator": (
                "data/lol/warehouse/private_v2/draft-interactions/"
                "representation-rank-private-result.json"
            ),
            "parquet_output": False,
        }
    ):
        raise PrivateRunnerError("future artifact boundary invalid")


def _verify_pinned_files(
    sources: Mapping[str, Mapping[str, Any]], *, root: Path
) -> None:
    for name, source in sources.items():
        path = _resolve(root, str(source["locator"]))
        if not path.is_file() or path.is_symlink():
            raise PrivateRunnerError(f"{name} is not a regular file")
        if assay.raw_sha256(path) != str(source["raw_sha256"]):
            raise PrivateRunnerError(f"{name} source bytes changed")


def _feature_logical_sha256(frame: pd.DataFrame) -> str:
    rows = [
        {
            column: (
                int(value)
                if isinstance(value, (np.integer,))
                else str(value)
            )
            for column, value in zip(SAFE_FEATURE_COLUMNS, row)
        }
        for row in frame.loc[:, SAFE_FEATURE_COLUMNS]
        .sort_values("game_id", kind="mergesort")
        .itertuples(index=False, name=None)
    ]
    return assay.canonical_sha256(rows)


def _node_lookup(node_domain: assay.NodeDomain) -> dict[tuple[str, str], int]:
    return {
        (champion, role): index
        for index, (champion, role) in enumerate(
            zip(node_domain.node_champion_ids, node_domain.node_roles)
        )
    }


def load_authoritative_features(
    contract: Mapping[str, Any],
    *,
    root: Path = Path.cwd(),
    read_parquet: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> FeatureEnvelope:
    """Load only the explicit outcome-free nonholdout feature projection."""
    validate_contract(contract)
    sources = contract["source_identity"]
    required = {
        "target_evidence",
        "outcome_free_split",
        "dependence_cluster_proxy",
        "champion_crosswalk",
        "private_feature_materialization",
    }
    if not required <= set(sources):
        raise PrivateRunnerError("feature source identity incomplete")
    _verify_pinned_files({name: sources[name] for name in required}, root=root)

    evidence = _load_canonical(
        _resolve(root, sources["target_evidence"]["locator"]),
        expected_raw=sources["target_evidence"]["raw_sha256"],
    )
    split = _load_canonical(
        _resolve(root, sources["outcome_free_split"]["locator"]),
        expected_raw=sources["outcome_free_split"]["raw_sha256"],
    )
    validate_evidence(evidence)
    validate_split(split)
    materialization = evidence.get("private_materialization", {})
    if (
        materialization.get("locator")
        != sources["private_feature_materialization"]["locator"]
        or materialization.get("raw_sha256")
        != sources["private_feature_materialization"]["raw_sha256"]
    ):
        raise PrivateRunnerError("feature materialization is not evidence-bound")

    node_domain = assay.load_node_domain(
        _resolve(root, sources["champion_crosswalk"]["locator"]),
        expected_raw_sha256=sources["champion_crosswalk"]["raw_sha256"],
    )
    cluster_domain = assay.load_cluster_domain(
        cluster_proxy_path=_resolve(
            root, sources["dependence_cluster_proxy"]["locator"]
        ),
        split_path=_resolve(root, sources["outcome_free_split"]["locator"]),
        expected_cluster_proxy_raw_sha256=sources[
            "dependence_cluster_proxy"
        ]["raw_sha256"],
        expected_split_raw_sha256=sources["outcome_free_split"]["raw_sha256"],
    )
    if FORBIDDEN_READINESS_COLUMNS & set(SAFE_FEATURE_COLUMNS):
        raise PrivateRunnerError("outcome entered readiness projection")
    frame = read_parquet(
        _resolve(root, sources["private_feature_materialization"]["locator"]),
        columns=list(SAFE_FEATURE_COLUMNS),
        filters=[("split", "in", list(NONHOLDOUT_SPLITS))],
    )
    if tuple(frame.columns) != SAFE_FEATURE_COLUMNS:
        raise PrivateRunnerError("feature reader did not honor exact projection")
    if frame.empty or frame["game_id"].duplicated().any():
        raise PrivateRunnerError("feature game identity invalid")
    if not set(frame["split"]) <= set(NONHOLDOUT_SPLITS):
        raise PrivateRunnerError("sealed or unknown split entered feature projection")
    if frame.loc[:, ROLE_COLUMNS].isna().any().any():
        raise PrivateRunnerError("feature champion identity missing")

    assignment = {str(row["game_id"]): row for row in split["assignments"]}
    for row in frame.itertuples(index=False):
        expected = assignment.get(str(row.game_id))
        if (
            expected is None
            or str(expected["dependence_cluster_id"])
            != str(row.dependence_cluster_id)
            or str(expected["split"]) != str(row.split)
            or str(expected["oe_date_naive"]) != str(row.oe_date_naive)
        ):
            raise PrivateRunnerError("feature/split identity mismatch")

    lookup = _node_lookup(node_domain)
    fixture_rows = []
    for row in frame.itertuples(index=False):
        nodes = []
        for column in ROLE_COLUMNS:
            role = column.split("_")[1]
            key = (str(getattr(row, column)), role)
            if key not in lookup:
                raise PrivateRunnerError("feature champion-role is outside crosswalk")
            nodes.append(lookup[key])
        fixture_rows.append(
            {
                "game_id": str(row.game_id),
                "split": str(row.split),
                "league": str(row.canonical_league),
                "nodes": nodes,
            }
        )
    logical = _feature_logical_sha256(frame)
    pinned_logical = contract.get("feature_projection_logical_rows_sha256")
    if pinned_logical is not None and logical != pinned_logical:
        raise PrivateRunnerError("feature projection logical rows changed")
    domain = assay._build_feature_domain(
        fixture_rows,
        node_domain=node_domain,
        cluster_domain=cluster_domain,
        source_raw_sha256=sources["private_feature_materialization"]["raw_sha256"],
    )
    assay.validate_feature_domain(domain)
    ordered_rows = tuple(
        tuple(row)
        for row in frame.loc[:, SAFE_FEATURE_COLUMNS]
        .sort_values("game_id", kind="mergesort")
        .itertuples(index=False, name=None)
    )
    unsigned = {
        "feature_domain_sha256": domain.artifact_sha256,
        "selected_columns": list(SAFE_FEATURE_COLUMNS),
        "source_raw_sha256": sources["private_feature_materialization"]["raw_sha256"],
        "logical_rows_sha256": logical,
        "rows": len(ordered_rows),
        "splits": {
            key: int(value)
            for key, value in frame["split"].value_counts().sort_index().items()
        },
        "authoritative_runner_envelope": True,
        "shell_authoritative_source_verified": False,
        "outcome_columns_loaded": False,
        "final_target_loaded": False,
    }
    return FeatureEnvelope(
        domain=domain,
        ordered_rows=ordered_rows,
        selected_columns=SAFE_FEATURE_COLUMNS,
        source_raw_sha256=sources["private_feature_materialization"]["raw_sha256"],
        logical_rows_sha256=logical,
        artifact_sha256=assay.canonical_sha256(unsigned),
    )


def load_authoritative_target_m0(
    contract: Mapping[str, Any],
    feature: FeatureEnvelope,
    *,
    root: Path = Path.cwd(),
    read_parquet: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> TargetM0Envelope:
    """Future reviewed boundary. Never called by ``verify_ready``."""
    validate_contract(contract)
    if contract["runner_review_status"] != "PASS":
        raise PrivateRunnerError("target/M0 loading requires runner review PASS")
    try:
        runner_review_authority.require_independent_runner_review_permit(
            contract["runner_review_permit"],
            review_core_sha256=contract["runner_review_core_sha256"],
            root=root,
        )
    except runner_review_authority.RunnerReviewAuthorityError as exc:
        raise PrivateRunnerError(str(exc)) from exc
    sources = contract["source_identity"]
    _verify_pinned_files(
        {
            name: sources[name]
            for name in (
                "target_evidence",
                "human_authority",
                "nuisance_artifact",
                "nuisance_oof_materialization",
                "outcome_free_split",
            )
        },
        root=root,
    )
    evidence = _load_canonical(
        _resolve(root, sources["target_evidence"]["locator"]),
        expected_raw=sources["target_evidence"]["raw_sha256"],
    )
    split = _load_canonical(
        _resolve(root, sources["outcome_free_split"]["locator"]),
        expected_raw=sources["outcome_free_split"]["raw_sha256"],
    )
    validate_evidence(evidence)
    validate_split(split)
    authority_path = _resolve(root, sources["human_authority"]["locator"])
    for action in ("model_fit", "rank_selection"):
        authority = load_and_require_exact_human_authority(
            evidence, split, action=action, authority_path=authority_path
        )
        if (
            authority.get("reviewer_identity") != "KOI_MARI"
            or authority.get("final_temporal_holdout_sealed") is not True
        ):
            raise PrivateRunnerError("KOI_MARI target authority invalid")
    nuisance = _load_canonical(
        _resolve(root, sources["nuisance_artifact"]["locator"]),
        expected_raw=sources["nuisance_artifact"]["raw_sha256"],
    )
    validate_nuisance_artifact(nuisance)
    oof_identity = nuisance["oof_materialization"]
    if any(
        oof_identity.get(key)
        != sources["nuisance_oof_materialization"].get(key)
        for key in (
            "locator",
            "raw_sha256",
            "logical_rows_sha256",
            "ordered_logical_rows_sha256",
        )
    ):
        raise PrivateRunnerError("OOF materialization is not nuisance-bound")
    if (
        oof_identity.get("rows") != PINNED_OOF_ROWS
        or sources["nuisance_oof_materialization"].get("rows")
        != PINNED_OOF_ROWS
        or oof_identity.get("predicted_game_membership_sha256")
        != PINNED_OOF_MEMBERSHIP_SHA256
        or sources["nuisance_oof_materialization"].get(
            "predicted_game_membership_sha256"
        )
        != PINNED_OOF_MEMBERSHIP_SHA256
    ):
        raise PrivateRunnerError("pinned nuisance OOF membership changed")
    oof_path = _resolve(root, sources["nuisance_oof_materialization"]["locator"])
    membership = read_parquet(
        oof_path, columns=list(TARGET_MEMBERSHIP_COLUMNS)
    )
    if (
        tuple(membership.columns) != TARGET_MEMBERSHIP_COLUMNS
        or len(membership) != PINNED_OOF_ROWS
        or membership["game_id"].duplicated().any()
    ):
        raise PrivateRunnerError("OOF membership projection invalid")
    membership_ids = tuple(sorted(membership["game_id"].astype(str)))
    if assay.canonical_sha256(list(membership_ids)) != PINNED_OOF_MEMBERSHIP_SHA256:
        raise PrivateRunnerError("OOF game membership changed")
    assignments = {str(row["game_id"]): row for row in split["assignments"]}
    final_ids = {
        game_id
        for game_id, row in assignments.items()
        if row["split"] == assay.FINAL_SPLIT
    }
    if len(final_ids) != PINNED_FINAL_ROWS or set(membership_ids) & final_ids:
        raise PrivateRunnerError("sealed final membership reached OOF")
    for row in membership.itertuples(index=False):
        assigned = assignments.get(str(row.game_id))
        if (
            assigned is None
            or assigned["split"] == assay.FINAL_SPLIT
            or str(assigned["dependence_cluster_id"])
            != str(row.dependence_cluster_id)
            or str(assigned["split"]) != str(row.split)
            or pd.Timestamp(assigned["oe_date_naive"])
            != pd.Timestamp(row.oe_date_naive)
        ):
            raise PrivateRunnerError("OOF membership differs from frozen split")
    fit_availability = assay._build_fit_availability_domain(
        membership_ids, source_raw_sha256=oof_identity["raw_sha256"]
    )
    assay.validate_fit_availability_domain(fit_availability)
    frame = read_parquet(oof_path, columns=list(TARGET_M0_COLUMNS))
    if tuple(frame.columns) != TARGET_M0_COLUMNS:
        raise PrivateRunnerError("target/M0 schema incomplete")
    if (
        frame.empty
        or len(frame) != PINNED_OOF_ROWS
        or frame["game_id"].duplicated().any()
        or not set(frame["split"]) <= set(NONHOLDOUT_SPLITS)
        or not bool(
            frame["y_blue_win"].map(lambda value: value in (0, 1)).all()
        )
    ):
        raise PrivateRunnerError("target/M0 domain invalid")
    feature_rows = {str(row[0]): row for row in feature.ordered_rows}
    for row in frame.itertuples(index=False):
        source = feature_rows.get(str(row.game_id))
        if (
            source is None
            or str(source[1]) != str(row.dependence_cluster_id)
            or str(source[2]) != str(row.split)
            or str(source[3]) != str(row.oe_date_naive)
        ):
            raise PrivateRunnerError("target/M0 feature identity mismatch")
    p0 = frame["p_blue_win_nuisance_oof"].to_numpy(dtype=float)
    if not np.isfinite(p0).all() or np.any(p0 <= 0) or np.any(p0 >= 1):
        raise PrivateRunnerError("M0 probabilities invalid")
    target_domain = assay._build_target_domain(
        dict(zip(frame["game_id"].astype(str), frame["y_blue_win"].astype(int))),
        source_raw_sha256=sources["nuisance_oof_materialization"]["raw_sha256"],
    )
    assay.validate_target_domain(target_domain)
    target_ids = {game_id for game_id, _ in target_domain.ordered_targets}
    m0_ids = set(frame["game_id"].astype(str))
    availability_ids = set(fit_availability.ordered_game_ids)
    if not target_ids == m0_ids == availability_ids:
        raise PrivateRunnerError(
            "TargetDomain IDs do not equal M0/FitAvailabilityDomain IDs"
        )
    ordered_rows = tuple(
        tuple(row)
        for row in frame.sort_values("game_id", kind="mergesort").itertuples(
            index=False, name=None
        )
    )
    unsigned = {
        "target_domain_sha256": target_domain.artifact_sha256,
        "m0_float64_hex": sorted(
            (str(game_id), float(value).hex())
            for game_id, value in zip(frame["game_id"], p0)
        ),
        "source_raw_sha256": oof_identity["raw_sha256"],
        "logical_rows_sha256": oof_identity["logical_rows_sha256"],
        "ordered_logical_rows_sha256": oof_identity[
            "ordered_logical_rows_sha256"
        ],
        "rows": len(frame),
        "final_target_loaded": False,
    }
    return TargetM0Envelope(
        target_domain=target_domain,
        m0_by_game_id=tuple(
            sorted((str(game_id), float(value)) for game_id, value in zip(frame["game_id"], p0))
        ),
        ordered_rows=ordered_rows,
        source_raw_sha256=oof_identity["raw_sha256"],
        logical_rows_sha256=oof_identity["logical_rows_sha256"],
        ordered_logical_rows_sha256=oof_identity[
            "ordered_logical_rows_sha256"
        ],
        artifact_sha256=assay.canonical_sha256(unsigned),
    )


def load_fit_availability_domain(
    contract: Mapping[str, Any],
    feature: FeatureEnvelope,
    *,
    root: Path = Path.cwd(),
    read_parquet: Callable[..., pd.DataFrame] = pd.read_parquet,
) -> assay.FitAvailabilityDomain:
    """Load OOF membership/provenance only; never request y or probabilities."""
    validate_contract(contract)
    source = contract["source_identity"]["nuisance_oof_materialization"]
    path = _resolve(root, source["locator"])
    if assay.raw_sha256(path) != source["raw_sha256"]:
        raise PrivateRunnerError("fit-availability source bytes changed")
    if FORBIDDEN_READINESS_COLUMNS & set(SAFE_FIT_AVAILABILITY_COLUMNS):
        raise PrivateRunnerError("outcome entered fit-availability projection")
    frame = read_parquet(path, columns=list(SAFE_FIT_AVAILABILITY_COLUMNS))
    if tuple(frame.columns) != SAFE_FIT_AVAILABILITY_COLUMNS:
        raise PrivateRunnerError("fit-availability projection changed")
    if (
        len(frame) != source["rows"]
        or frame["game_id"].duplicated().any()
        or not set(frame["split"]) <= set(NONHOLDOUT_SPLITS)
    ):
        raise PrivateRunnerError("fit-availability membership invalid")
    feature_rows = {str(row[0]): row for row in feature.ordered_rows}
    for row in frame.itertuples(index=False):
        source_row = feature_rows.get(str(row.game_id))
        if (
            source_row is None
            or str(source_row[1]) != str(row.dependence_cluster_id)
            or str(source_row[2]) != str(row.split)
            or str(source_row[3]) != str(row.oe_date_naive)
        ):
            raise PrivateRunnerError("fit-availability feature identity mismatch")
    domain = assay._build_fit_availability_domain(
        frame["game_id"].astype(str).tolist(),
        source_raw_sha256=source["raw_sha256"],
    )
    assay.validate_fit_availability_domain(domain)
    if assay.canonical_sha256(sorted(domain.ordered_game_ids)) != source[
        "predicted_game_membership_sha256"
    ]:
        raise PrivateRunnerError("fit-availability membership hash changed")
    return domain


def exact_fit_game_ids(
    feature: FeatureEnvelope,
    target: TargetM0Envelope,
    *,
    prediction_month: str,
    eligible_nodes: Sequence[bool] | None = None,
) -> tuple[str, ...]:
    targets = dict(target.target_domain.ordered_targets)
    m0_ids = {game_id for game_id, _ in target.m0_by_game_id}
    if set(targets) != m0_ids:
        raise PrivateRunnerError("TargetDomain IDs do not exactly equal M0 IDs")
    mask = (
        np.ones(len(feature.domain.node_domain.node_roles), dtype=bool)
        if eligible_nodes is None
        else np.asarray(eligible_nodes, dtype=bool)
    )
    if mask.shape != (len(feature.domain.node_domain.node_roles),):
        raise PrivateRunnerError("eligible node mask invalid")
    return tuple(
        row[0]
        for row in feature.domain.records
        if row[0] in targets
        and row[1] in NONHOLDOUT_SPLITS
        and row[3] < prediction_month
        and np.all(mask[np.asarray(row[5], dtype=int)])
    )


def likelihood_feature_domain(
    feature: FeatureEnvelope, fit_availability_ids: Sequence[object]
) -> assay.FeatureDomain:
    """Restrict feature support to the immutable outcome-free OOF membership."""
    ids = tuple(str(value) for value in fit_availability_ids)
    if len(ids) != len(set(ids)):
        raise PrivateRunnerError("fit-availability identity duplicated")
    available = set(ids)
    records = [
        {
            "game_id": row[0],
            "split": row[1],
            "league": row[4],
            "nodes": row[5],
        }
        for row in feature.domain.records
        if row[0] in available
    ]
    if {row["game_id"] for row in records} != available:
        raise PrivateRunnerError("fit-availability ID is outside FeatureDomain")
    return assay._build_feature_domain(
        records,
        node_domain=feature.domain.node_domain,
        cluster_domain=feature.domain.cluster_domain,
        source_raw_sha256=feature.source_raw_sha256,
    )


def fixed_point_fit_support(
    feature_domain: assay.FeatureDomain,
    *,
    prediction_month: str,
    minimum_clusters: int = assay.MIN_NODE_CLUSTERS,
) -> FixedPointSupport:
    """Find the unique maximal monotone node/row support core."""
    assay.validate_feature_domain(feature_domain)
    candidates = [
        row
        for row in feature_domain.records
        if row[1] in NONHOLDOUT_SPLITS and row[3] < prediction_month
    ]
    if not candidates:
        raise PrivateRunnerError("fixed-point fit population unavailable")
    nodes = np.asarray([row[5] for row in candidates], dtype=np.int64)
    clusters = np.asarray([row[2] for row in candidates], dtype=object)
    active = np.ones(len(candidates), dtype=bool)
    checks = 0
    changes = 0
    while True:
        checks += 1
        eligible = np.zeros(
            len(feature_domain.node_domain.node_roles), dtype=bool
        )
        for node in range(len(eligible)):
            eligible[node] = (
                len(set(clusters[active & np.any(nodes == node, axis=1)]))
                >= minimum_clusters
            )
        updated = active & np.all(eligible[nodes], axis=1)
        if np.array_equal(updated, active):
            break
        active = updated
        changes += 1
    ids = tuple(
        row[0] for row, keep in zip(candidates, active) if bool(keep)
    )
    unsigned = {
        "prediction_month": prediction_month,
        "minimum_distinct_clusters": minimum_clusters,
        "ordered_fit_game_ids": list(ids),
        "eligible_nodes": eligible.tolist(),
        "convergence_checks": checks,
        "changing_rounds": changes,
        "derivation": "maximal_monotone_fixed_point",
    }
    return FixedPointSupport(
        ordered_fit_game_ids=ids,
        eligible_nodes=tuple(bool(value) for value in eligible),
        convergence_checks=checks,
        changing_rounds=changes,
        artifact_sha256=assay.canonical_sha256(unsigned),
    )


def run_penalty_family(
    *,
    family: str,
    evaluate: Callable[[str, float], Mapping[str, Any]],
) -> FamilyResult:
    """Run all six months for every penalty; any missing fit fails the family."""
    rows: list[dict[str, Any]] = []
    try:
        for penalty in assay.PENALTY_GRID:
            for month in assay.INNER_MONTHS:
                rows.append(dict(evaluate(month, penalty)))
        selected = assay.select_separate_penalty(rows, family=family)
    except Exception:
        return FamilyResult(
            status="inconclusive",
            selected=None,
            rows=tuple(rows),
            fallback="M0",
            reason_code="penalty_selection_failed",
        )
    return FamilyResult(
        status="pass",
        selected=float(selected),
        rows=tuple(rows),
        fallback="none",
        reason_code=None,
    )


def choose_development_width(
    *,
    prepared_fold: assay.PreparedFold,
    game_ids: Sequence[object],
    target_domain: assay.TargetDomain,
    width_predictions: Mapping[int, Sequence[object]],
    m0: Sequence[object],
    m8_optimization_stable: bool,
) -> tuple[int | None, dict[str, Any]]:
    """Use exact widths 1/2/4/8 and make M8 an exact alias of width 8."""
    if set(width_predictions) != set(assay.WIDTHS):
        return None, {"status": "inconclusive", "fallback": "M0"}
    predictions: dict[int | str, Sequence[object]] = dict(width_predictions)
    predictions["M0"] = m0
    predictions["M8"] = width_predictions[8]
    try:
        width, diagnostics = assay.select_development_width(
            prepared_fold=prepared_fold,
            game_ids=game_ids,
            target_domain=target_domain,
            predictions=predictions,
            m8_optimization_stable=m8_optimization_stable,
        )
    except Exception:
        return None, {
            "status": "inconclusive",
            "fallback": "M0",
            "reason_code": "development_gate_failed",
        }
    return width, {"status": "pass", "fallback": "none", **diagnostics}


def validate_locked_candidate(
    *,
    prepared_fold: assay.PreparedFold,
    game_ids: Sequence[object],
    locked_width: int,
    target_domain: assay.TargetDomain,
    locked_prediction: Sequence[object],
    m0: Sequence[object],
    m8: Sequence[object],
    m8_optimization_stable: bool,
) -> dict[str, Any]:
    """Validation accepts only the frozen width, M0, and M8; no reselection."""
    predictions: dict[int | str, Sequence[object]] = {
        locked_width: locked_prediction,
        "M0": m0,
        "M8": m8,
    }
    try:
        result = assay.validate_locked_width(
            prepared_fold=prepared_fold,
            game_ids=game_ids,
            locked_width=locked_width,
            target_domain=target_domain,
            predictions=predictions,
            m8_optimization_stable=m8_optimization_stable,
        )
    except Exception:
        return {
            "status": "inconclusive",
            "fallback": "M0",
            "reason_code": "validation_gate_failed",
        }
    return {"status": "pass", "fallback": "none", **result}


def score_latent_fit(
    *,
    fit: assay.LatentFit,
    eligibility_binding: assay.EligibilityBinding,
    game_ids: Sequence[object],
    verified_nuisance_oof: Mapping[str, float],
) -> dict[str, float]:
    assay._validate_eligibility_binding(eligibility_binding)
    if fit.eligibility_binding_sha256 != eligibility_binding.artifact_sha256:
        raise PrivateRunnerError("fit eligibility provenance changed")
    records = {row[0]: row for row in eligibility_binding.feature_domain.records}
    ids = tuple(str(value) for value in game_ids)
    eligible_nodes = np.asarray(eligibility_binding.eligible_nodes, dtype=bool)
    expected_ids = tuple(
        game_id
        for game_id in eligibility_binding.ordered_source_game_ids
        if np.all(
            eligible_nodes[np.asarray(records[game_id][5], dtype=np.int64)]
        )
    )
    if ids != expected_ids:
        raise PrivateRunnerError(
            "score rows differ from bound eligible prediction population"
        )
    p0 = np.asarray([verified_nuisance_oof[game_id] for game_id in ids], dtype=float)
    if np.any(p0 <= 0) or np.any(p0 >= 1) or not np.isfinite(p0).all():
        raise PrivateRunnerError("score M0 invalid")
    nodes = np.asarray([records[game_id][5] for game_id in ids], dtype=np.int64)
    interaction = assay.interaction_logits(
        nodes[:, :5],
        nodes[:, 5:],
        fit.ally_centered,
        fit.enemy_centered,
        width=fit.width,
        node_domain=eligibility_binding.node_domain,
    )[2]
    probability = expit(logit(p0) + interaction)
    return dict(zip(ids, probability.astype(float)))


def _build_month_context(
    *,
    feature_domain: assay.FeatureDomain,
    availability: assay.FitAvailabilityDomain,
    target: TargetM0Envelope,
    split: str,
    calendar_month: str,
) -> MonthRunContext:
    records = {row[0]: row for row in feature_domain.records}
    score_ids = tuple(
        row[0]
        for row in feature_domain.records
        if row[1] == split and row[3] == calendar_month
    )
    fit_ids = tuple(
        row[0] for row in feature_domain.records if row[3] < calendar_month
    )
    if not score_ids or not fit_ids:
        raise PrivateRunnerError("frozen monthly population is unavailable")
    m0_by_id = dict(target.m0_by_game_id)
    if set(score_ids) - set(m0_by_id):
        raise PrivateRunnerError("monthly M0 membership is incomplete")
    coverage = assay.outcome_free_coverage(
        feature_domain=feature_domain,
        score_game_ids=score_ids,
        fit_game_ids=fit_ids,
        split=split,
        fit_availability_domain=availability,
    )
    binding = coverage.eligibility_binding
    eligible_nodes = np.asarray(binding.eligible_nodes, dtype=bool)
    prediction_ids = tuple(
        game_id
        for game_id in binding.ordered_source_game_ids
        if np.all(
            eligible_nodes[np.asarray(records[game_id][5], dtype=np.int64)]
        )
    )
    prediction_clusters = tuple(records[game_id][2] for game_id in prediction_ids)
    m0 = np.asarray([m0_by_id[game_id] for game_id in prediction_ids], dtype=float)
    membership_sha256 = assay.canonical_sha256(
        {
            "split": split,
            "calendar_month": calendar_month,
            "eligibility_binding_sha256": binding.artifact_sha256,
            "ordered_prediction_membership": [
                [game_id, cluster_id]
                for game_id, cluster_id in zip(
                    prediction_ids, prediction_clusters
                )
            ],
        }
    )
    prepared: assay.PreparedFold | None = None
    if split in {"development", "validation"}:
        if coverage.report["passed"]:
            prepared = assay.prepare_outer_fold(
                feature_domain=feature_domain,
                score_game_ids=score_ids,
                fit_game_ids=fit_ids,
                nuisance_probability=[
                    m0_by_id[game_id] for game_id in score_ids
                ],
                verified_nuisance_oof=m0_by_id,
                split=split,
                fit_availability_domain=availability,
            )
            if prepared.ordered_eligible_game_ids != prediction_ids:
                raise PrivateRunnerError(
                    "monthly prepared/scoring population identity differs"
                )
    return MonthRunContext(
        split=split,
        calendar_month=calendar_month,
        eligibility_binding=binding,
        prediction_game_ids=prediction_ids,
        prediction_cluster_ids=prediction_clusters,
        m0_probability=m0,
        membership_sha256=membership_sha256,
        prepared_fold=prepared,
        coverage_report=coverage.report,
    )


def _execute_real_fit(request: FitRequest) -> FitExecution:
    """Future PASS-only fit adapter. Tests replace this function with a spy."""
    binding = request.context.eligibility_binding
    records = {row[0]: row for row in binding.feature_domain.records}
    fit_ids = binding.ordered_fit_game_ids
    fit_nodes = np.asarray([records[game_id][5] for game_id in fit_ids], dtype=int)
    mode = (
        "ally_only"
        if request.family == "ally"
        else "enemy_only"
        if request.family == "enemy"
        else "joint"
    )
    try:
        fit = assay.fit_latent_candidate(
            blue_nodes=fit_nodes[:, :5],
            red_nodes=fit_nodes[:, 5:],
            target_domain=request.target_domain,
            split_identity=request.split,
            game_ids=fit_ids,
            verified_nuisance_oof=request.verified_nuisance_oof,
            eligibility_binding=binding,
            width=request.width,
            lambda_ally=request.lambda_ally,
            lambda_enemy=request.lambda_enemy,
            mode=mode,
        )
    except assay.RepresentationRankAssayError as exc:
        raise StatisticalRunInconclusive("fit_unavailable") from exc
    prediction = score_latent_fit(
        fit=fit,
        eligibility_binding=binding,
        game_ids=request.context.prediction_game_ids,
        verified_nuisance_oof=request.verified_nuisance_oof,
    )
    return FitExecution(
        prediction_by_game_id=tuple(prediction.items()),
        objective=float(fit.objective),
        max_gradient=float(fit.maximum_absolute_gradient),
        converged_starts=int(fit.converged_starts),
        stability_rms=float(fit.best_two_interaction_logit_rms),
    )


def _validated_execution(
    execution: FitExecution, *, request: FitRequest
) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(execution, FitExecution):
        raise PrivateRunnerError("fit adapter returned an invalid result type")
    expected = request.context.prediction_game_ids
    observed = tuple(game_id for game_id, _ in execution.prediction_by_game_id)
    probability = np.asarray(
        [value for _, value in execution.prediction_by_game_id], dtype=float
    )
    try:
        optimization_passed = assay.optimization_gate_decision(
            converged_starts=execution.converged_starts,
            max_gradient=execution.max_gradient,
            stability_rms=execution.stability_rms,
        )
    except assay.RepresentationRankAssayError as exc:
        raise PrivateRunnerError(
            "fit adapter result identity/schema invalid"
        ) from exc
    if (
        observed != expected
        or probability.shape != (len(expected),)
        or not np.isfinite(probability).all()
        or np.any(probability <= 0)
        or np.any(probability >= 1)
        or not np.isfinite(
            [
                execution.objective,
                execution.max_gradient,
                execution.stability_rms,
            ]
        ).all()
        or execution.objective < 0
        or not optimization_passed
    ):
        raise PrivateRunnerError("fit adapter result identity/schema invalid")
    target_by_id = dict(request.target_domain.ordered_targets)
    if set(expected) - set(target_by_id):
        raise PrivateRunnerError("fit scoring target membership incomplete")
    y = np.asarray([target_by_id[game_id] for game_id in expected], dtype=float)
    log_loss_total = float(
        np.sum(assay._loss_rows(y, probability, "log_loss"))
    )
    brier_total = float(np.sum(assay._loss_rows(y, probability, "brier")))
    summary = {
        "execution_status": "passed",
        "maps": len(expected),
        "clusters": len(set(request.context.prediction_cluster_ids)),
        "membership_sha256": request.context.membership_sha256,
        "objective": float(execution.objective),
        "max_gradient": float(execution.max_gradient),
        "converged_starts": int(execution.converged_starts),
        "stability_rms": float(execution.stability_rms),
        "log_loss_total": log_loss_total,
        "brier_total": brier_total,
    }
    return probability, summary


def _populate_plan_row(
    row: dict[str, Any],
    *,
    context: MonthRunContext,
    summary: Mapping[str, Any],
    lambda_ally: float,
    lambda_enemy: float,
    width: int,
) -> None:
    row["lambda_ally"] = float(lambda_ally)
    row["lambda_enemy"] = float(lambda_enemy)
    row["width"] = int(width)
    for key, value in summary.items():
        row[key] = value
    if row["membership_sha256"] != context.membership_sha256:
        raise PrivateRunnerError("fit-plan membership summary changed")


def _bind_plan_request_identity(
    row: dict[str, Any],
    *,
    request: FitRequest,
) -> None:
    if (
        row["sequence"] != request.sequence
        or row["stage"] != request.stage
        or row["split"] != request.split
        or row["calendar_month"] != request.calendar_month
        or row["family"] != request.family
        or row["fit_role"] != request.fit_role
    ):
        raise PrivateRunnerError("fit request differs from frozen plan slot")
    row["lambda_ally"] = float(request.lambda_ally)
    row["lambda_enemy"] = float(request.lambda_enemy)
    row["width"] = int(request.width)


def _population_identity(
    contexts: Sequence[MonthRunContext], *, split: str
) -> dict[str, Any]:
    return private_result.derive_population_identity(
        split=split,
        ordered_month_blocks=tuple(
            (
                context.membership_sha256,
                len(context.prediction_game_ids),
            )
            for context in contexts
        ),
    )


def _aggregate_coverage(
    contexts: Sequence[MonthRunContext],
) -> list[dict[str, Any]]:
    count_keys = (
        "maps",
        "eligible_maps",
        "clusters",
        "eligible_clusters",
    )
    output: list[dict[str, Any]] = []
    for context in contexts:
        report = context.coverage_report
        if (
            not isinstance(report, Mapping)
            or not isinstance(report.get("overall"), Mapping)
            or not isinstance(report.get("by_month"), Mapping)
            or not isinstance(report.get("by_league"), Mapping)
            or not isinstance(report.get("passed"), bool)
            or set(report["by_month"]) != {context.calendar_month}
            or not 1 <= len(report["by_league"]) <= 32
        ):
            raise PrivateRunnerError("coverage report aggregate schema changed")

        def project_counts(source: Mapping[str, Any]) -> dict[str, Any]:
            if not set(count_keys) <= set(source):
                raise PrivateRunnerError("coverage report counts incomplete")
            return {key: source[key] for key in count_keys}

        overall = project_counts(report["overall"])
        month_counts = project_counts(
            report["by_month"][context.calendar_month]
        )
        league_rows = [
            {
                "league": league,
                **project_counts(source),
            }
            for league, source in sorted(report["by_league"].items())
            if isinstance(league, str)
        ]
        if len(league_rows) != len(report["by_league"]):
            raise PrivateRunnerError("coverage league identity invalid")
        try:
            derived_passed = assay.coverage_gate_decision(
                overall=overall,
                month_rows=(month_counts,),
                league_rows=league_rows,
            )
        except assay.RepresentationRankAssayError as exc:
            raise PrivateRunnerError(
                "coverage report aggregate identities invalid"
            ) from exc
        if report["passed"] is not derived_passed:
            raise PrivateRunnerError(
                "coverage report pass differs from frozen gates"
            )
        output.append(
            {
            "split": context.split,
            "calendar_month": context.calendar_month,
            "passed": derived_passed,
            **{key: int(overall[key]) for key in count_keys},
            "month": {
                "calendar_month": context.calendar_month,
                **{key: int(month_counts[key]) for key in count_keys},
            },
            "leagues": [
                {
                    "league": row["league"],
                    **{key: int(row[key]) for key in count_keys},
                }
                for row in league_rows
            ],
            "membership_sha256": context.membership_sha256,
            }
        )
    return output


def _result_payload(
    *,
    contract: Mapping[str, Any],
    config: Mapping[str, Any],
    feature: FeatureEnvelope,
    target: TargetM0Envelope,
    availability: assay.FitAvailabilityDomain,
    inner_contexts: Sequence[MonthRunContext],
    development_contexts: Sequence[MonthRunContext],
    validation_contexts: Sequence[MonthRunContext],
    fit_plan: Sequence[Mapping[str, Any]],
    run_status: str,
    reason_code: str | None,
    reason_context: Mapping[str, Any] | None,
    lambda_ally: float | None,
    lambda_enemy: float | None,
    selected_width: int | None,
    stage_status: Mapping[str, Any],
    development_diagnostics: Any,
    validation_diagnostics: Any,
) -> dict[str, Any]:
    development = _population_identity(
        development_contexts, split="development"
    )
    validation = _population_identity(validation_contexts, split="validation")
    combined_digest = assay.canonical_sha256(
        {
            "development_membership_sha256": development[
                "membership_sha256"
            ],
            "validation_membership_sha256": validation["membership_sha256"],
            "maps": development["maps"] + validation["maps"],
        }
    )
    permit = contract.get("runner_review_permit")
    if not isinstance(permit, Mapping):
        raise PrivateRunnerError("PASS run lacks a pinned runner-review permit")
    unsigned = {
        "schema_id": private_result.SCHEMA_ID,
        "aggregate_only": True,
        "development_only": True,
        "run_status": run_status,
        "fallback": "none" if run_status == "accepted" else "M0",
        "reason_code": None if run_status == "accepted" else reason_code,
        "reason_context": (
            None if run_status == "accepted" else dict(reason_context or {})
        ),
        "selected_model": (
            "latent_candidate" if run_status == "accepted" else "M0"
        ),
        "selected_width": selected_width if run_status == "accepted" else None,
        "contract_artifact_sha256": contract["artifact_sha256"],
        "contract_review_core_sha256": contract[
            "runner_review_core_sha256"
        ],
        "runner_review_permit_raw_sha256": permit["raw_sha256"],
        "source_identity_sha256": assay.canonical_sha256(
            contract["source_identity"]
        ),
        "runtime_identity_sha256": assay.canonical_sha256(
            config["executable_identity"]["runtime_versions"]
        ),
        "feature_domain_sha256": feature.domain.artifact_sha256,
        "target_domain_sha256": target.target_domain.artifact_sha256,
        "fit_availability_domain_sha256": availability.artifact_sha256,
        "population": {
            "development_maps": development["maps"],
            "development_membership_sha256": development[
                "membership_sha256"
            ],
            "validation_maps": validation["maps"],
            "validation_membership_sha256": validation["membership_sha256"],
            "combined_maps": development["maps"] + validation["maps"],
            "combined_membership_sha256": combined_digest,
        },
        "penalties": {
            "lambda_ally": lambda_ally,
            "lambda_enemy": lambda_enemy,
        },
        "fit_counts": {
            "actual": sum(
                row["execution_status"] == "passed" for row in fit_plan
            ),
            "planned_slots": 56,
        },
        "stage_status": dict(stage_status),
        "coverage_diagnostics": private_result.build_coverage_diagnostics(
            _aggregate_coverage(
                [
                    *inner_contexts,
                    *development_contexts,
                    *validation_contexts,
                ]
            )
        ),
        "development_diagnostics": development_diagnostics,
        "validation_diagnostics": validation_diagnostics,
        "fit_plan": [dict(row) for row in fit_plan],
        "final_target_loaded": False,
        "publication_authority": False,
        "production_authority": False,
        "reliability_authority": False,
        "promotion_authority": False,
        "sota_claim_authority": False,
    }
    payload = private_result.with_artifact_sha256(unsigned)
    private_result.validate_private_result(payload)
    return payload


def run_private(
    contract: Mapping[str, Any],
    *,
    root: Path = Path.cwd(),
    feature_loader: Callable[..., FeatureEnvelope] | None = None,
    availability_loader: Callable[..., assay.FitAvailabilityDomain] | None = None,
    target_loader: Callable[..., TargetM0Envelope] | None = None,
    fit_executor: Callable[[FitRequest], FitExecution] | None = None,
    result_writer: Callable[..., None] | None = None,
) -> dict[str, Any]:
    """Run the frozen private assay after an independently permitted PASS."""
    validate_contract(contract)
    if contract["runner_review_status"] != "PASS":
        raise PrivateRunnerError("private run requires runner review PASS")
    try:
        runner_review_authority.require_independent_runner_review_permit(
            contract["runner_review_permit"],
            review_core_sha256=contract["runner_review_core_sha256"],
            root=root,
        )
    except runner_review_authority.RunnerReviewAuthorityError as exc:
        raise PrivateRunnerError(str(exc)) from exc
    feature_loader = feature_loader or load_authoritative_features
    availability_loader = availability_loader or load_fit_availability_domain
    target_loader = target_loader or load_authoritative_target_m0
    fit_executor = fit_executor or _execute_real_fit
    result_writer = result_writer or private_result.write_private_result

    config_path = _resolve(
        root, contract["audited_shell"]["config"]["locator"]
    )
    config = _load_canonical(
        config_path,
        expected_raw=contract["audited_shell"]["config"]["raw_sha256"],
    )
    assay.validate_config(config)

    # The ordering and cardinality of these calls is an execution invariant.
    feature = feature_loader(contract, root=root)
    availability = availability_loader(contract, feature, root=root)
    target = target_loader(contract, feature, root=root)
    target_ids = {game_id for game_id, _ in target.target_domain.ordered_targets}
    m0_by_id = dict(target.m0_by_game_id)
    availability_ids = set(availability.ordered_game_ids)
    if not target_ids == set(m0_by_id) == availability_ids:
        raise PrivateRunnerError(
            "TargetDomain IDs do not equal M0/FitAvailabilityDomain IDs"
        )
    feature_records = {row[0]: row for row in feature.domain.records}
    if (
        set(target_ids) - set(feature_records)
        or any(
            feature_records[game_id][1] == assay.FINAL_SPLIT
            for game_id in target_ids
        )
        or any(
            feature_records[game_id][1] not in NONHOLDOUT_SPLITS
            for game_id in target_ids
        )
    ):
        raise PrivateRunnerError("sealed or unknown target membership reached run")
    likelihood = likelihood_feature_domain(
        feature, availability.ordered_game_ids
    )

    inner_contexts = {
        month: _build_month_context(
            feature_domain=likelihood,
            availability=availability,
            target=target,
            split="train",
            calendar_month=month,
        )
        for month in assay.INNER_MONTHS
    }
    development_contexts = [
        _build_month_context(
            feature_domain=likelihood,
            availability=availability,
            target=target,
            split="development",
            calendar_month=month,
        )
        for month, _, _ in assay.ELIGIBLE_GATE_BLOCKS["development"]
    ]
    validation_contexts = [
        _build_month_context(
            feature_domain=likelihood,
            availability=availability,
            target=target,
            split="validation",
            calendar_month=month,
        )
        for month, _, _ in assay.ELIGIBLE_GATE_BLOCKS["validation"]
    ]
    for context in [
        *inner_contexts.values(),
        *development_contexts,
        *validation_contexts,
    ]:
        source_ids = tuple(
            context.eligibility_binding.ordered_source_game_ids
        )
        if (
            not set(context.prediction_game_ids) <= set(source_ids)
            or any(
                game_id not in feature_records
                or feature_records[game_id][1] != context.split
                or feature_records[game_id][1] == assay.FINAL_SPLIT
                for game_id in source_ids
            )
        ):
            raise PrivateRunnerError(
                "monthly binding contains sealed or wrong-split membership"
            )
    development_population = sum(
        len(context.prediction_game_ids) for context in development_contexts
    )
    validation_population = sum(
        len(context.prediction_game_ids) for context in validation_contexts
    )
    if development_population != 981 or validation_population != 1084:
        raise PrivateRunnerError("frozen eligible prediction population changed")

    plan = private_result.empty_fit_plan()
    stage_status = {
        stage: {"status": "not_run", "reason_code": None}
        for stage in ("inner", "development", "validation")
    }
    lambda_ally: float | None = None
    lambda_enemy: float | None = None
    selected_width: int | None = None
    development_diagnostics: Any = None
    validation_diagnostics: Any = None
    paths = contract["future_private_artifacts"]
    result_path = _resolve(root, paths["manifest_locator"])

    def emit_inconclusive(
        *,
        stage: str,
        reason_code: str,
        failed_index: int | None = None,
        failed_context: MonthRunContext | None = None,
    ) -> dict[str, Any]:
        if failed_index is not None:
            row = plan[failed_index]
            row["execution_status"] = "failed"
            if failed_context is not None:
                row["maps"] = len(failed_context.prediction_game_ids)
                row["clusters"] = len(
                    set(failed_context.prediction_cluster_ids)
                )
                row["membership_sha256"] = (
                    failed_context.membership_sha256
                )
        stage_status[stage] = {
            "status": "failed",
            "reason_code": reason_code,
        }
        context_value = private_result.reason_context(
            stage=stage,
            sequence=(
                failed_index + 1 if failed_index is not None else None
            ),
            calendar_month=(
                failed_context.calendar_month
                if failed_context is not None
                else None
            ),
            family=(
                plan[failed_index]["family"]
                if failed_index is not None
                else None
            ),
            width=(
                plan[failed_index]["width"]
                if failed_index is not None
                else None
            ),
        )
        development_gate_terminal = (
            reason_code == "development_gate_failed"
        )
        development_dto = private_result.build_development_diagnostics(
            (
                development_diagnostics
                if stage_status["development"]["status"] == "passed"
                or development_gate_terminal
                else None
            ),
            status=stage_status["development"]["status"],
            selected_width=(
                selected_width
                if stage_status["development"]["status"] == "passed"
                else None
            ),
        )
        validation_gate_terminal = reason_code == "validation_gate_failed"
        validation_dto = private_result.build_validation_diagnostics(
            (
                validation_diagnostics
                if stage_status["validation"]["status"] == "passed"
                or validation_gate_terminal
                else None
            ),
            status=stage_status["validation"]["status"],
            selected_width=(
                selected_width
                if stage_status["validation"]["status"] == "passed"
                or validation_gate_terminal
                else None
            ),
        )
        payload = _result_payload(
            contract=contract,
            config=config,
            feature=feature,
            target=target,
            availability=availability,
            inner_contexts=list(inner_contexts.values()),
            development_contexts=development_contexts,
            validation_contexts=validation_contexts,
            fit_plan=plan,
            run_status="inconclusive",
            reason_code=reason_code,
            reason_context=context_value,
            lambda_ally=lambda_ally,
            lambda_enemy=lambda_enemy,
            selected_width=None,
            stage_status=stage_status,
            development_diagnostics=development_dto,
            validation_diagnostics=validation_dto,
        )
        result_writer(payload, path=result_path)
        return payload

    all_contexts = [
        *inner_contexts.values(),
        *development_contexts,
        *validation_contexts,
    ]
    coverage_rows = _aggregate_coverage(all_contexts)
    inconsistent_prepared = next(
        (
            context
            for context, coverage_row in zip(
                [*development_contexts, *validation_contexts],
                coverage_rows[len(inner_contexts) :],
            )
            if context.prepared_fold is None and coverage_row["passed"]
        ),
        None,
    )
    if inconsistent_prepared is not None:
        raise PrivateRunnerError(
            "passed coverage block lacks a prepared scoring fold"
        )
    failed_coverage = next(
        (
            context
            for context, coverage_row in zip(all_contexts, coverage_rows)
            if not coverage_row["passed"]
        ),
        None,
    )
    if failed_coverage is not None:
        return emit_inconclusive(
            stage=(
                "inner"
                if failed_coverage.split == "train"
                else failed_coverage.split
            ),
            reason_code="coverage_gate_failed",
            failed_context=failed_coverage,
        )

    # Inner penalty fitting: family, penalty, month is the frozen serial order.
    penalty_rows_by_family: dict[str, list[dict[str, Any]]] = {
        "ally": [],
        "enemy": [],
    }
    plan_index = 0
    for family in ("ally", "enemy"):
        for penalty in assay.PENALTY_GRID:
            for month in assay.INNER_MONTHS:
                context = inner_contexts[month]
                request = FitRequest(
                    sequence=plan_index + 1,
                    stage="inner",
                    split="train",
                    calendar_month=month,
                    family=family,
                    fit_role=f"{family}_penalty",
                    width=8,
                    lambda_ally=float(penalty) if family == "ally" else 1.0,
                    lambda_enemy=float(penalty) if family == "enemy" else 1.0,
                    context=context,
                    target_domain=target.target_domain,
                    verified_nuisance_oof=m0_by_id,
                )
                _bind_plan_request_identity(plan[plan_index], request=request)
                try:
                    execution = fit_executor(request)
                except StatisticalRunInconclusive as exc:
                    return emit_inconclusive(
                        stage="inner",
                        reason_code=exc.reason_code,
                        failed_index=plan_index,
                        failed_context=context,
                    )
                _, summary = _validated_execution(execution, request=request)
                _populate_plan_row(
                    plan[plan_index],
                    context=context,
                    summary=summary,
                    lambda_ally=request.lambda_ally,
                    lambda_enemy=request.lambda_enemy,
                    width=8,
                )
                penalty_rows_by_family[family].append(
                    {
                        "family": family,
                        "lambda": float(penalty),
                        "width": 8,
                        "calendar_month": month,
                        "split": "train",
                        "maps": summary["maps"],
                        "clusters": summary["clusters"],
                        "membership_sha256": summary["membership_sha256"],
                        "log_loss_total": summary["log_loss_total"],
                        "brier_total": summary["brier_total"],
                        "strictly_earlier_fit": True,
                        "cluster_atomic": True,
                    }
                )
                plan_index += 1
    try:
        lambda_ally = float(
            assay.select_separate_penalty(
                penalty_rows_by_family["ally"], family="ally"
            )
        )
        lambda_enemy = float(
            assay.select_separate_penalty(
                penalty_rows_by_family["enemy"], family="enemy"
            )
        )
    except assay.RepresentationRankAssayError as exc:
        return emit_inconclusive(
            stage="inner", reason_code="penalty_selection_failed"
        )
    stage_status["inner"] = {"status": "passed", "reason_code": None}

    development_folds = [
        context.prepared_fold for context in development_contexts
    ]
    if any(fold is None for fold in development_folds):
        return emit_inconclusive(
            stage="development", reason_code="coverage_gate_failed"
        )
    try:
        development_prepared = assay.combine_prepared_folds(
            development_folds, split="development"  # type: ignore[arg-type]
        )
    except assay.RepresentationRankAssayError as exc:
        return emit_inconclusive(
            stage="development", reason_code="coverage_gate_failed"
        )
    development_by_width: dict[int, list[np.ndarray]] = {
        width: [] for width in assay.WIDTHS
    }
    development_m8_stable = True
    for context in development_contexts:
        for width in assay.WIDTHS:
            request = FitRequest(
                sequence=plan_index + 1,
                stage="development",
                split="development",
                calendar_month=context.calendar_month,
                family="joint",
                fit_role="candidate_width",
                width=width,
                lambda_ally=lambda_ally,
                lambda_enemy=lambda_enemy,
                context=context,
                target_domain=target.target_domain,
                verified_nuisance_oof=m0_by_id,
            )
            _bind_plan_request_identity(plan[plan_index], request=request)
            try:
                execution = fit_executor(request)
            except StatisticalRunInconclusive as exc:
                return emit_inconclusive(
                    stage="development",
                    reason_code=exc.reason_code,
                    failed_index=plan_index,
                    failed_context=context,
                )
            probability, summary = _validated_execution(
                execution, request=request
            )
            _populate_plan_row(
                plan[plan_index],
                context=context,
                summary=summary,
                lambda_ally=lambda_ally,
                lambda_enemy=lambda_enemy,
                width=width,
            )
            development_by_width[width].append(probability)
            if width == 8:
                development_m8_stable = development_m8_stable and (
                    assay.optimization_gate_decision(
                        converged_starts=execution.converged_starts,
                        max_gradient=execution.max_gradient,
                        stability_rms=execution.stability_rms,
                    )
                )
            plan_index += 1
    try:
        selected_width, development_diagnostics = assay.select_development_width(
            prepared_fold=development_prepared,
            game_ids=development_prepared.ordered_eligible_game_ids,
            target_domain=target.target_domain,
            predictions={
                **{
                    width: np.concatenate(development_by_width[width])
                    for width in assay.WIDTHS
                },
                "M0": development_prepared.m0_probability,
                "M8": np.concatenate(development_by_width[8]),
            },
            m8_optimization_stable=development_m8_stable,
        )
    except assay.RepresentationRankAssayError as exc:
        development_diagnostics = exc.diagnostics
        if development_diagnostics is None:
            raise PrivateRunnerError(
                "development gate failure lacks aggregate diagnostics"
            ) from exc
        return emit_inconclusive(
            stage="development", reason_code="development_gate_failed"
        )
    stage_status["development"] = {
        "status": "passed",
        "reason_code": None,
    }

    validation_folds = [context.prepared_fold for context in validation_contexts]
    if any(fold is None for fold in validation_folds):
        return emit_inconclusive(
            stage="validation", reason_code="coverage_gate_failed"
        )
    try:
        validation_prepared = assay.combine_prepared_folds(
            validation_folds, split="validation"  # type: ignore[arg-type]
        )
    except assay.RepresentationRankAssayError as exc:
        return emit_inconclusive(
            stage="validation", reason_code="coverage_gate_failed"
        )
    locked_predictions: list[np.ndarray] = []
    m8_predictions: list[np.ndarray] = []
    validation_m8_stable = True
    for context in validation_contexts:
        locked_index = plan_index
        locked_request = FitRequest(
            sequence=locked_index + 1,
            stage="validation",
            split="validation",
            calendar_month=context.calendar_month,
            family="joint",
            fit_role="locked_width",
            width=selected_width,
            lambda_ally=lambda_ally,
            lambda_enemy=lambda_enemy,
            context=context,
            target_domain=target.target_domain,
            verified_nuisance_oof=m0_by_id,
        )
        _bind_plan_request_identity(
            plan[locked_index],
            request=locked_request,
        )
        try:
            locked_execution = fit_executor(locked_request)
        except StatisticalRunInconclusive as exc:
            return emit_inconclusive(
                stage="validation",
                reason_code=exc.reason_code,
                failed_index=locked_index,
                failed_context=context,
            )
        locked_probability, locked_summary = _validated_execution(
            locked_execution, request=locked_request
        )
        _populate_plan_row(
            plan[locked_index],
            context=context,
            summary=locked_summary,
            lambda_ally=lambda_ally,
            lambda_enemy=lambda_enemy,
            width=selected_width,
        )
        locked_predictions.append(locked_probability)
        plan_index += 1
        m8_index = plan_index
        if selected_width == 8:
            plan[m8_index].update(
                {
                    "lambda_ally": lambda_ally,
                    "lambda_enemy": lambda_enemy,
                    "width": 8,
                    "execution_status": "aliased",
                    "maps": len(context.prediction_game_ids),
                    "clusters": len(set(context.prediction_cluster_ids)),
                    "membership_sha256": context.membership_sha256,
                    "objective": locked_execution.objective,
                    "max_gradient": locked_execution.max_gradient,
                    "converged_starts": locked_execution.converged_starts,
                    "stability_rms": locked_execution.stability_rms,
                    "log_loss_total": locked_summary["log_loss_total"],
                    "brier_total": locked_summary["brier_total"],
                }
            )
            m8_predictions.append(locked_probability)
            m8_execution = locked_execution
        else:
            m8_request = FitRequest(
                sequence=m8_index + 1,
                stage="validation",
                split="validation",
                calendar_month=context.calendar_month,
                family="joint",
                fit_role="M8_reference",
                width=8,
                lambda_ally=lambda_ally,
                lambda_enemy=lambda_enemy,
                context=context,
                target_domain=target.target_domain,
                verified_nuisance_oof=m0_by_id,
            )
            _bind_plan_request_identity(
                plan[m8_index],
                request=m8_request,
            )
            try:
                m8_execution = fit_executor(m8_request)
            except StatisticalRunInconclusive as exc:
                return emit_inconclusive(
                    stage="validation",
                    reason_code=exc.reason_code,
                    failed_index=m8_index,
                    failed_context=context,
                )
            m8_probability, m8_summary = _validated_execution(
                m8_execution, request=m8_request
            )
            _populate_plan_row(
                plan[m8_index],
                context=context,
                summary=m8_summary,
                lambda_ally=lambda_ally,
                lambda_enemy=lambda_enemy,
                width=8,
            )
            m8_predictions.append(m8_probability)
        validation_m8_stable = validation_m8_stable and (
            assay.optimization_gate_decision(
                converged_starts=m8_execution.converged_starts,
                max_gradient=m8_execution.max_gradient,
                stability_rms=m8_execution.stability_rms,
            )
        )
        plan_index += 1
    try:
        validation_diagnostics = assay.validate_locked_width(
            prepared_fold=validation_prepared,
            game_ids=validation_prepared.ordered_eligible_game_ids,
            locked_width=selected_width,
            target_domain=target.target_domain,
            predictions={
                selected_width: np.concatenate(locked_predictions),
                "M0": validation_prepared.m0_probability,
                "M8": np.concatenate(m8_predictions),
            },
            m8_optimization_stable=validation_m8_stable,
        )
    except assay.RepresentationRankAssayError as exc:
        validation_diagnostics = exc.diagnostics
        if validation_diagnostics is None:
            raise PrivateRunnerError(
                "validation gate failure lacks aggregate diagnostics"
            ) from exc
        return emit_inconclusive(
            stage="validation", reason_code="validation_gate_failed"
        )
    stage_status["validation"] = {
        "status": "passed",
        "reason_code": None,
    }
    development_dto = private_result.build_development_diagnostics(
        development_diagnostics,
        status="passed",
        selected_width=selected_width,
    )
    validation_dto = private_result.build_validation_diagnostics(
        validation_diagnostics,
        status="passed",
        selected_width=selected_width,
    )
    payload = _result_payload(
        contract=contract,
        config=config,
        feature=feature,
        target=target,
        availability=availability,
        inner_contexts=list(inner_contexts.values()),
        development_contexts=development_contexts,
        validation_contexts=validation_contexts,
        fit_plan=plan,
        run_status="accepted",
        reason_code=None,
        reason_context=None,
        lambda_ally=lambda_ally,
        lambda_enemy=lambda_enemy,
        selected_width=selected_width,
        stage_status=stage_status,
        development_diagnostics=development_dto,
        validation_diagnostics=validation_dto,
    )
    result_writer(payload, path=result_path)
    return payload


def logical_artifact_sha256(frame: pd.DataFrame) -> str:
    raise PrivateRunnerError(
        "row-level private result hashing is prohibited; use aggregate result"
    )


def write_private_artifact(
    frame: pd.DataFrame, *, parquet_path: Path, manifest_path: Path
) -> dict[str, Any]:
    raise PrivateRunnerError(
        "row-level/parquet private results are prohibited; use aggregate JSON"
    )


def verify_private_artifact(
    *, parquet_path: Path, manifest_path: Path
) -> dict[str, Any]:
    raise PrivateRunnerError(
        "row-level/parquet private results are prohibited; use aggregate JSON"
    )


def verify_ready(
    contract: Mapping[str, Any], *, root: Path = Path.cwd()
) -> dict[str, Any]:
    validate_contract(contract)
    _verify_pinned_files(contract["audited_shell"], root=root)
    # The audited config validates its complete source/runtime contract.
    shell_config_path = _resolve(root, contract["audited_shell"]["config"]["locator"])
    shell_config = json.loads(shell_config_path.read_bytes())
    assay.verify_config_sources(shell_config, root=root)
    evidence = _load_canonical(
        _resolve(root, contract["source_identity"]["target_evidence"]["locator"]),
        expected_raw=contract["source_identity"]["target_evidence"]["raw_sha256"],
    )
    split = _load_canonical(
        _resolve(root, contract["source_identity"]["outcome_free_split"]["locator"]),
        expected_raw=contract["source_identity"]["outcome_free_split"]["raw_sha256"],
    )
    authority_path = _resolve(
        root, contract["source_identity"]["human_authority"]["locator"]
    )
    for action in ("model_fit", "rank_selection"):
        load_and_require_exact_human_authority(
            evidence, split, action=action, authority_path=authority_path
        )
    feature = load_authoritative_features(contract, root=root)
    availability = load_fit_availability_domain(
        contract, feature, root=root
    )
    counts: dict[str, int] = {}
    for row in feature.domain.records:
        counts[row[1]] = counts.get(row[1], 0) + 1
    return {
        "schema_id": "scryglass.representation-rank-private-readiness.v1",
        "status": "READY_FOR_RUNNER_REVIEW",
        "feature_rows": len(feature.domain.records),
        "feature_rows_by_split": dict(sorted(counts.items())),
        "feature_projection_logical_rows_sha256": feature.logical_rows_sha256,
        "feature_envelope_sha256": feature.artifact_sha256,
        "fit_availability_rows": len(availability.ordered_game_ids),
        "fit_availability_domain_sha256": availability.artifact_sha256,
        "fit_availability_columns": list(SAFE_FIT_AVAILABILITY_COLUMNS),
        "safe_feature_projection": list(feature.selected_columns),
        "shell_feature_domain_authoritative_source_verified": False,
        "runner_feature_envelope_authoritative": True,
        "real_target_loader_invoked": False,
        "target_rows_loaded": False,
        "outcome_columns_loaded": False,
        "final_target_loaded": False,
        "candidate_fit_started": False,
    }


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--verify-ready", action="store_true")
    action.add_argument("--run-private", action="store_true")
    args = parser.parse_args(argv)
    contract = load_contract(args.contract)
    if args.run_private:
        if contract["runner_review_status"] != "PASS":
            raise PrivateRunnerError(
                "private run blocked: runner_review_status is not PASS"
            )
        result = run_private(contract)
        print(
            json.dumps(
                {
                    "run_status": result["run_status"],
                    "artifact_sha256": result["artifact_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0
    if not args.verify_ready:
        raise PrivateRunnerError("fitting refused by default; use --verify-ready")
    print(json.dumps(verify_ready(contract), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
