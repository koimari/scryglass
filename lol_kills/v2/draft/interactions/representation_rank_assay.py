"""Fixture engine for a private incremental latent pair-capacity assay.

M0 is an externally verified nuisance OOF probability keyed by game_id.  This
module never refits its common features.  Widths 1/2/4/8 directly fit latent
ally-similarity and enemy-symplectic capacity on strictly earlier rows.
Coordinates are noninterpretable; only induced pair scores and probabilities
may be compared.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logit


CONFIG_SCHEMA_ID = "scryglass.draft-interaction-latent-capacity-assay-config.v1"
REPORT_SCHEMA_ID = "scryglass.draft-interaction-latent-capacity-assay-report.v1"
WIDTHS = (1, 2, 4, 8)
PENALTY_GRID = (0.01, 0.1, 1.0)
INNER_MONTHS = tuple(f"2025-{month:02d}" for month in range(4, 10))
DEVELOPMENT_SEED = 2_026_072_901
VALIDATION_SEED = 2_026_072_902
PRIMARY_REPLICATES = 20_000
DEVELOPMENT_ENDPOINT = 19_875
VALIDATION_ENDPOINT = 19_667
MIN_NODE_CLUSTERS = 5
GRADIENT_TOLERANCE = 1e-5
STABILITY_RMS_TOLERANCE = 0.01
MAX_CONVERGED_STARTS = 3
BLOCK_LOG_LOSS_DELTA_LIMIT = 0.010
METRIC_GATE_RULES = {
    "M0": {
        "log_loss": (0.0, True),
        "brier": (0.001, False),
        "calibration": (0.010, False),
    },
    "M8": {
        "log_loss": (0.002, False),
        "brier": (0.001, False),
        "calibration": (0.010, False),
    },
}
ALLOWED_NONHOLDOUT_SPLITS = ("train", "development", "validation")
FINAL_SPLIT = "final_temporal_holdout"
CANONICAL_POSITIONS = ("top", "jungle", "mid", "bot", "support")

CHRONOLOGICAL_BLOCKS = {
    "development": (
        ("2025-10", 140, 56),
        ("2026-01", 342, 190),
        ("2026-02", 461, 235),
        ("2026-03", 223, 84),
    ),
    "validation": (("2026-04", 581, 297), ("2026-05", 648, 244)),
}
ELIGIBLE_GATE_BLOCKS = {
    "development": (
        ("2025-10", 128, 55),
        ("2026-01", 230, 152),
        ("2026-02", 421, 221),
        ("2026-03", 202, 83),
    ),
    "validation": (("2026-04", 515, 288), ("2026-05", 569, 243)),
}


class RepresentationRankAssayError(ValueError):
    """Raised when a frozen assay contract fails closed."""

    def __init__(
        self,
        message: str,
        *,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics


def metric_gate_decision(
    *,
    comparator: str,
    metric: str,
    upper: object,
) -> dict[str, Any]:
    """Return the frozen metric decision from its confidence endpoint."""
    if (
        comparator not in METRIC_GATE_RULES
        or metric not in METRIC_GATE_RULES[comparator]
        or isinstance(upper, bool)
        or not isinstance(upper, numbers.Real)
        or not math.isfinite(float(upper))
    ):
        raise RepresentationRankAssayError("metric gate input invalid")
    limit, strict = METRIC_GATE_RULES[comparator][metric]
    passed = float(upper) < limit if strict else float(upper) <= limit
    return {
        "limit": limit,
        "strict": strict,
        "passed": passed,
    }


def block_gate_decision(delta: object) -> bool:
    """Return the frozen inclusive chronological-block decision."""
    if (
        isinstance(delta, bool)
        or not isinstance(delta, numbers.Real)
        or not math.isfinite(float(delta))
    ):
        raise RepresentationRankAssayError("block gate input invalid")
    return float(delta) <= BLOCK_LOG_LOSS_DELTA_LIMIT


def optimization_gate_decision(
    *,
    converged_starts: object,
    max_gradient: object,
    stability_rms: object,
) -> bool:
    """Return whether a completed fit satisfies the frozen optimizer gates."""
    if (
        isinstance(converged_starts, bool)
        or not isinstance(converged_starts, numbers.Integral)
        or isinstance(max_gradient, bool)
        or not isinstance(max_gradient, numbers.Real)
        or not math.isfinite(float(max_gradient))
        or isinstance(stability_rms, bool)
        or not isinstance(stability_rms, numbers.Real)
        or not math.isfinite(float(stability_rms))
    ):
        raise RepresentationRankAssayError("optimization gate input invalid")
    return bool(
        2 <= int(converged_starts) <= MAX_CONVERGED_STARTS
        and 0.0 <= float(max_gradient) <= GRADIENT_TOLERANCE
        and 0.0 <= float(stability_rms) <= STABILITY_RMS_TOLERANCE
    )


def coverage_gate_decision(
    *,
    overall: Mapping[str, object],
    month_rows: Sequence[Mapping[str, object]],
    league_rows: Sequence[Mapping[str, object]],
) -> bool:
    """Recompute the complete outcome-free coverage decision from counts."""
    required = {
        "maps",
        "eligible_maps",
        "clusters",
        "eligible_clusters",
    }

    def counts(
        row: Mapping[str, object], *, require_positive: bool
    ) -> tuple[int, int, int, int]:
        if not isinstance(row, Mapping) or not required <= set(row):
            raise RepresentationRankAssayError("coverage count schema invalid")
        values = tuple(row[key] for key in required)
        if any(
            isinstance(value, bool)
            or not isinstance(value, numbers.Integral)
            or int(value) < 0
            for value in values
        ):
            raise RepresentationRankAssayError("coverage count invalid")
        maps = int(row["maps"])
        eligible_maps = int(row["eligible_maps"])
        clusters = int(row["clusters"])
        eligible_clusters = int(row["eligible_clusters"])
        if (
            (require_positive and (maps <= 0 or clusters <= 0))
            or clusters > maps
            or eligible_maps > maps
            or eligible_clusters > clusters
            or eligible_clusters > eligible_maps
        ):
            raise RepresentationRankAssayError(
                "coverage count identities invalid"
            )
        return maps, eligible_maps, clusters, eligible_clusters

    overall_counts = counts(overall, require_positive=True)
    months = [counts(row, require_positive=True) for row in month_rows]
    leagues = [counts(row, require_positive=True) for row in league_rows]
    if (
        len(months) != 1
        or not 1 <= len(leagues) <= 32
        or months[0] != overall_counts
        or tuple(
            sum(row[index] for row in leagues) for index in range(4)
        )
        != overall_counts
    ):
        raise RepresentationRankAssayError(
            "coverage aggregate identities inconsistent"
        )
    maps, eligible_maps, clusters, eligible_clusters = overall_counts
    month_maps, month_eligible, _, month_eligible_clusters = months[0]
    return bool(
        5 * eligible_maps >= 4 * maps
        and 5 * eligible_clusters >= 4 * clusters
        and 3 * month_eligible >= 2 * month_maps
        and month_eligible_clusters >= 15
        and all(
            4 * league_eligible >= 3 * league_maps
            for league_maps, league_eligible, league_clusters, _ in leagues
            if league_maps >= 30 and league_clusters >= 10
        )
    )


@dataclass(frozen=True)
class NodeDomain:
    node_roles: tuple[str, ...]
    node_champion_ids: tuple[str, ...]
    source_raw_sha256: str
    artifact_sha256: str


def _build_node_domain(
    crosswalk_entries: Sequence[Mapping[str, Any]], *, source_raw_sha256: str
) -> NodeDomain:
    champions = tuple(str(row.get("stable_champion_id", "")) for row in crosswalk_entries)
    if (
        len(source_raw_sha256) != 64
        or len(champions) < 10
        or any(not champion for champion in champions)
        or len(set(champions)) != len(champions)
    ):
        raise RepresentationRankAssayError("node-domain crosswalk invalid")
    roles = tuple(role for _ in champions for role in CANONICAL_POSITIONS)
    champion_ids = tuple(
        champion for champion in champions for _ in CANONICAL_POSITIONS
    )
    unsigned = {
        "source_raw_sha256": source_raw_sha256,
        "ordering": "crosswalk_entry_order_then_top_jungle_mid_bot_support",
        "node_roles": list(roles),
        "node_champion_ids": list(champion_ids),
    }
    return NodeDomain(
        node_roles=roles,
        node_champion_ids=champion_ids,
        source_raw_sha256=source_raw_sha256,
        artifact_sha256=canonical_sha256(unsigned),
    )


def load_node_domain(path: Path, *, expected_raw_sha256: str) -> NodeDomain:
    from lol_kills.v2.champions.id_crosswalk import validate_artifact

    if not path.is_file() or path.is_symlink():
        raise RepresentationRankAssayError("node-domain source is not a regular file")
    raw = path.read_bytes()
    observed = hashlib.sha256(raw).hexdigest()
    if observed != expected_raw_sha256:
        raise RepresentationRankAssayError("node-domain source bytes changed")
    payload = json.loads(raw)
    if raw != canonical_bytes(payload):
        raise RepresentationRankAssayError("node-domain source is not canonical")
    validate_artifact(payload)
    return _build_node_domain(
        payload["entries"], source_raw_sha256=observed
    )


def validate_node_domain(domain: NodeDomain) -> None:
    unsigned = {
        "source_raw_sha256": domain.source_raw_sha256,
        "ordering": "crosswalk_entry_order_then_top_jungle_mid_bot_support",
        "node_roles": list(domain.node_roles),
        "node_champion_ids": list(domain.node_champion_ids),
    }
    if (
        len(domain.source_raw_sha256) != 64
        or domain.artifact_sha256 != canonical_sha256(unsigned)
        or len(domain.node_roles) != len(domain.node_champion_ids)
        or len(domain.node_roles) < 50
        or any(
            domain.node_roles[index] != CANONICAL_POSITIONS[index % 5]
            for index in range(len(domain.node_roles))
        )
        or any(
            len(set(domain.node_champion_ids[index : index + 5])) != 1
            for index in range(0, len(domain.node_roles), 5)
        )
        or len(set(domain.node_champion_ids[::5]))
        != len(domain.node_champion_ids) // 5
    ):
        raise RepresentationRankAssayError("node-domain identity invalid")


@dataclass(frozen=True)
class ClusterDomain:
    ordered_game_clusters: tuple[tuple[str, str], ...]
    ordered_cluster_max_months: tuple[tuple[str, str], ...]
    source_raw_sha256: str
    artifact_sha256: str


def _build_cluster_domain(
    assignments: Sequence[Mapping[str, Any]], *, source_raw_sha256: str
) -> ClusterDomain:
    game_clusters: dict[str, str] = {}
    cluster_dates: dict[str, list[str]] = {}
    for row in assignments:
        game_id = str(row.get("game_id", ""))
        cluster = str(row.get("dependence_cluster_id", ""))
        date = str(row.get("oe_date_naive", ""))
        if (
            not game_id
            or not cluster
            or len(date) < 7
            or game_id in game_clusters
        ):
            raise RepresentationRankAssayError("cluster-domain assignment invalid")
        game_clusters[game_id] = cluster
        cluster_dates.setdefault(cluster, []).append(date)
    cluster_months = {
        cluster: max(dates)[:7] for cluster, dates in cluster_dates.items()
    }
    unsigned = {
        "source_raw_sha256": source_raw_sha256,
        "ordered_game_clusters": sorted(game_clusters.items()),
        "ordered_cluster_max_months": sorted(cluster_months.items()),
    }
    return ClusterDomain(
        ordered_game_clusters=tuple(sorted(game_clusters.items())),
        ordered_cluster_max_months=tuple(sorted(cluster_months.items())),
        source_raw_sha256=source_raw_sha256,
        artifact_sha256=canonical_sha256(unsigned),
    )


def load_cluster_domain(
    *,
    cluster_proxy_path: Path,
    split_path: Path,
    expected_cluster_proxy_raw_sha256: str,
    expected_split_raw_sha256: str,
) -> ClusterDomain:
    from .oe_target_evidence import validate_split
    from .series_cluster_proxy import validate_artifact as validate_cluster_proxy

    payloads = []
    for path, expected, validator, label in (
        (
            cluster_proxy_path,
            expected_cluster_proxy_raw_sha256,
            validate_cluster_proxy,
            "cluster proxy",
        ),
        (split_path, expected_split_raw_sha256, validate_split, "split"),
    ):
        if not path.is_file() or path.is_symlink():
            raise RepresentationRankAssayError(f"{label} is not a regular file")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise RepresentationRankAssayError(f"{label} source bytes changed")
        payload = json.loads(raw)
        if raw != canonical_bytes(payload):
            raise RepresentationRankAssayError(f"{label} is not canonical JSON")
        validator(payload)
        payloads.append(payload)
    proxy, split = payloads
    proxy_map = {
        str(row["game_id"]): str(row["dependence_cluster_id"])
        for row in proxy["assignments"]
    }
    assignments = list(split["assignments"])
    if any(
        proxy_map.get(str(row["game_id"]))
        != str(row["dependence_cluster_id"])
        for row in assignments
    ):
        raise RepresentationRankAssayError("cluster/split assignment mismatch")
    combined_source = canonical_sha256(
        {
            "cluster_proxy_raw_sha256": expected_cluster_proxy_raw_sha256,
            "split_raw_sha256": expected_split_raw_sha256,
        }
    )
    return _build_cluster_domain(
        assignments, source_raw_sha256=combined_source
    )


def validate_cluster_domain(domain: ClusterDomain) -> None:
    unsigned = {
        "source_raw_sha256": domain.source_raw_sha256,
        "ordered_game_clusters": [list(row) for row in domain.ordered_game_clusters],
        "ordered_cluster_max_months": [
            list(row) for row in domain.ordered_cluster_max_months
        ],
    }
    if (
        len(domain.source_raw_sha256) != 64
        or domain.artifact_sha256 != canonical_sha256(unsigned)
        or len(dict(domain.ordered_game_clusters)) != len(domain.ordered_game_clusters)
        or len(dict(domain.ordered_cluster_max_months))
        != len(domain.ordered_cluster_max_months)
    ):
        raise RepresentationRankAssayError("cluster-domain identity invalid")


@dataclass(frozen=True)
class FeatureDomain:
    records: tuple[tuple[str, str, str, str, str, tuple[int, ...]], ...]
    node_domain: NodeDomain
    cluster_domain: ClusterDomain
    source_raw_sha256: str
    authoritative_source_verified: bool
    artifact_sha256: str


@dataclass(frozen=True)
class FitAvailabilityDomain:
    """Outcome-free identity of rows having a verified nuisance OOF slot."""

    ordered_game_ids: tuple[str, ...]
    source_raw_sha256: str
    artifact_sha256: str


def _build_fit_availability_domain(
    game_ids: Sequence[object], *, source_raw_sha256: str
) -> FitAvailabilityDomain:
    ids = tuple(sorted(str(value) for value in game_ids))
    if (
        len(source_raw_sha256) != 64
        or not ids
        or len(ids) != len(set(ids))
        or any(not value for value in ids)
    ):
        raise RepresentationRankAssayError("fit-availability identity invalid")
    unsigned = {
        "ordered_game_ids": list(ids),
        "source_raw_sha256": source_raw_sha256,
        "contains_targets": False,
        "contains_probabilities": False,
    }
    return FitAvailabilityDomain(
        ordered_game_ids=ids,
        source_raw_sha256=source_raw_sha256,
        artifact_sha256=canonical_sha256(unsigned),
    )


def validate_fit_availability_domain(domain: FitAvailabilityDomain) -> None:
    if not isinstance(domain, FitAvailabilityDomain):
        raise RepresentationRankAssayError("fit-availability domain is required")
    unsigned = {
        "ordered_game_ids": list(domain.ordered_game_ids),
        "source_raw_sha256": domain.source_raw_sha256,
        "contains_targets": False,
        "contains_probabilities": False,
    }
    if (
        len(domain.source_raw_sha256) != 64
        or not domain.ordered_game_ids
        or tuple(sorted(domain.ordered_game_ids)) != domain.ordered_game_ids
        or len(set(domain.ordered_game_ids)) != len(domain.ordered_game_ids)
        or domain.artifact_sha256 != canonical_sha256(unsigned)
    ):
        raise RepresentationRankAssayError("fit-availability identity invalid")


@dataclass(frozen=True)
class TargetDomain:
    ordered_targets: tuple[tuple[str, int], ...]
    source_raw_sha256: str
    artifact_sha256: str


def _build_target_domain(
    target_by_game_id: Mapping[str, object], *, source_raw_sha256: str
) -> TargetDomain:
    if any(
        not isinstance(value, (bool, np.bool_, numbers.Integral))
        or int(value) not in (0, 1)
        for value in target_by_game_id.values()
    ):
        raise RepresentationRankAssayError("target-domain fixture invalid")
    rows = tuple(
        sorted((str(game_id), int(value)) for game_id, value in target_by_game_id.items())
    )
    if (
        len(source_raw_sha256) != 64
        or len(dict(rows)) != len(rows)
        or any(value not in (0, 1) for _, value in rows)
    ):
        raise RepresentationRankAssayError("target-domain fixture invalid")
    unsigned = {
        "source_raw_sha256": source_raw_sha256,
        "ordered_targets": [list(row) for row in rows],
        "authoritative_loader_status": "future_private_runner_required",
    }
    return TargetDomain(
        ordered_targets=rows,
        source_raw_sha256=source_raw_sha256,
        artifact_sha256=canonical_sha256(unsigned),
    )


def validate_target_domain(domain: TargetDomain) -> None:
    unsigned = {
        "source_raw_sha256": domain.source_raw_sha256,
        "ordered_targets": [list(row) for row in domain.ordered_targets],
        "authoritative_loader_status": "future_private_runner_required",
    }
    if (
        len(domain.source_raw_sha256) != 64
        or len(dict(domain.ordered_targets)) != len(domain.ordered_targets)
        or any(value not in (0, 1) for _, value in domain.ordered_targets)
        or domain.artifact_sha256 != canonical_sha256(unsigned)
    ):
        raise RepresentationRankAssayError("target-domain identity invalid")


def _build_feature_domain(
    rows: Sequence[Mapping[str, Any]],
    *,
    node_domain: NodeDomain,
    cluster_domain: ClusterDomain,
    source_raw_sha256: str,
) -> FeatureDomain:
    validate_node_domain(node_domain)
    validate_cluster_domain(cluster_domain)
    clusters = dict(cluster_domain.ordered_game_clusters)
    months = dict(cluster_domain.ordered_cluster_max_months)
    records = []
    seen: set[str] = set()
    for row in rows:
        game_id = str(row.get("game_id", ""))
        if game_id in seen or game_id not in clusters:
            raise RepresentationRankAssayError("feature-domain game identity invalid")
        seen.add(game_id)
        nodes = _validate_ten_node_rows(
            np.asarray([row.get("nodes")], dtype=object),
            node_domain=node_domain,
            label="feature domain",
        )[0]
        cluster = clusters[game_id]
        records.append(
            (
                game_id,
                str(row.get("split", "")),
                cluster,
                months[cluster],
                str(row.get("league", "")),
                tuple(int(value) for value in nodes),
            )
        )
    unsigned = {
        "source_raw_sha256": source_raw_sha256,
        "node_domain_sha256": node_domain.artifact_sha256,
        "cluster_domain_sha256": cluster_domain.artifact_sha256,
        "records": records,
        "authoritative_source_verified": False,
        "authoritative_loader_status": "future_private_runner_required",
    }
    return FeatureDomain(
        records=tuple(records),
        node_domain=node_domain,
        cluster_domain=cluster_domain,
        source_raw_sha256=source_raw_sha256,
        authoritative_source_verified=False,
        artifact_sha256=canonical_sha256(unsigned),
    )


def validate_feature_domain(domain: FeatureDomain) -> None:
    validate_node_domain(domain.node_domain)
    validate_cluster_domain(domain.cluster_domain)
    unsigned = {
        "source_raw_sha256": domain.source_raw_sha256,
        "node_domain_sha256": domain.node_domain.artifact_sha256,
        "cluster_domain_sha256": domain.cluster_domain.artifact_sha256,
        "records": domain.records,
        "authoritative_source_verified": domain.authoritative_source_verified,
        "authoritative_loader_status": "future_private_runner_required",
    }
    clusters = dict(domain.cluster_domain.ordered_game_clusters)
    months = dict(domain.cluster_domain.ordered_cluster_max_months)
    if (
        len(domain.source_raw_sha256) != 64
        or domain.authoritative_source_verified is not False
        or domain.artifact_sha256 != canonical_sha256(unsigned)
        or len({row[0] for row in domain.records}) != len(domain.records)
    ):
        raise RepresentationRankAssayError("feature-domain identity invalid")
    for game_id, split, cluster, month, league, nodes in domain.records:
        if (
            not split
            or not league
            or clusters.get(game_id) != cluster
            or months.get(cluster) != month
        ):
            raise RepresentationRankAssayError("feature-domain provenance invalid")
        _validate_ten_node_rows(
            np.asarray([nodes], dtype=object),
            node_domain=domain.node_domain,
            label="feature domain",
        )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise RepresentationRankAssayError("config artifact hash mismatch")
    exact = {
        "schema_id": CONFIG_SCHEMA_ID,
        "status": "private_pending_fixture_review",
        "development_only": True,
        "real_candidate_outcomes_loaded": False,
        "predictive_authority": False,
        "authorizes_prediction": False,
        "authorizes_publication": False,
        "authorizes_production": False,
        "authorizes_reliability": False,
        "authorizes_promotion": False,
        "authorizes_sota_claim": False,
        "authorizes_whole_composition_result": False,
        "content_addressing_confers_authority": False,
        "candidate_widths": list(WIDTHS),
        "claim_ceiling": (
            "private retrospective latent pair-capacity assay only; no intrinsic "
            "table rank, whole-composition result, prediction, publication, "
            "production, Reliability, promotion, or SOTA authority"
        ),
    }
    if any(payload.get(key) != value for key, value in exact.items()):
        raise RepresentationRankAssayError("config fixed fields changed")
    if payload.get("estimand") != {
        "kind": "incremental_directly_fitted_latent_pair_capacity",
        "baseline": "exact verified nuisance OOF probability by game_id",
        "whole_composition_H": "unavailable_not_in_assay",
        "whole_composition_K": "unavailable_not_in_assay",
        "intrinsic_pair_table_rank": False,
        "coordinates_interpretable": False,
        "reported_objects": ["predictions", "induced_pair_tables"],
    }:
        raise RepresentationRankAssayError("estimand changed")
    if payload.get("latent_parameterization") != {
        "ally": {
            "name": "PSD latent-similarity dimension",
            "shape": "eligible_nodes_by_r",
            "centering": "exact within role over fit-fold observed eligible nodes",
            "score": (
                "(sum 10 blue ally dot products - sum 10 red ally dot "
                "products) / 10"
            ),
        },
        "enemy": {
            "name": "skew/symplectic modes",
            "shape": "eligible_nodes_by_2r",
            "centering": "exact within role over fit-fold observed eligible nodes",
            "J": "blockdiag([[0,1],[-1,0]]) repeated r times",
            "score": "sum 25 E_blue^T J E_red / 25",
            "induced_matrix_rank_upper": "2r",
        },
        "eta": "logit(p_M0) + g_A + g_E",
        "side_swap": (
            "interaction residual negates exactly; full probability need not "
            "complement because M0 may contain blue/context intercept"
        ),
    }:
        raise RepresentationRankAssayError("latent parameterization changed")
    if payload.get("eligibility_and_coverage") != {
        "node": (
            "maximal monotone core: every retained champion_by_role node occurs "
            "in at least 5 distinct strictly earlier optimization clusters"
        ),
        "fit_availability": (
            "FeatureDomain intersection pinned nuisance-OOF game membership; "
            "game/provenance identity only, no target or probability"
        ),
        "all_feature_history": "exposure_diagnostic_only_not_coordinate_support",
        "fixed_point": "drop unsupported rows and recompute until unchanged",
        "identical_row_mask": "M0 and every width",
        "pair_novelty": "allowed",
        "unseen_node": "unavailable",
        "split_minimum_map_fraction": 0.8,
        "split_minimum_cluster_fraction": 0.8,
        "month_minimum_map_fraction": 2 / 3,
        "month_minimum_clusters": 15,
        "league_gate_population": "leagues with at least 30 maps and 10 clusters",
        "league_minimum_map_fraction": 0.75,
        "exclusion_ledger_required": True,
        "derivation": "outcome_free",
    }:
        raise RepresentationRankAssayError("coverage contract changed")
    if payload.get("legal_draft_domain") != {
        "position_order_each_side": list(CANONICAL_POSITIONS),
        "node_id": "integer_in_range",
        "node_metadata": (
            "immutable NodeDomain derived from canonical pinned crosswalk bytes"
        ),
        "node_order": "crosswalk entry order then top,jungle,mid,bot,support",
        "crosswalk_source_bytes_verified": True,
        "champion_identity_rule": "ten_unique_across_map",
        "validation_surfaces": [
            "source_fit_rows",
            "coverage_rows",
            "objective_and_gradient",
            "fitting",
            "scoring",
        ],
        "failure": "reject_before_math",
    }:
        raise RepresentationRankAssayError("legal draft domain changed")
    if payload.get("gate_binding") != {
        "preparation": (
            "one chronological prediction month with strictly earlier "
            "FitAvailabilityDomain optimization clusters"
        ),
        "aggregation": "ordered concatenation of frozen monthly PreparedFold components",
        "bound_fields": [
            "ordered_full_game_ids",
            "ordered_full_cluster_ids",
            "ordered_full_chronological_blocks",
            "ordered_eligible_game_ids",
            "ordered_eligible_cluster_ids",
            "ordered_chronological_blocks",
            "exact_float64_M0",
            "exact_eligible_node_mask_per_block",
            "NodeDomain_SHA256",
            "ClusterDomain_SHA256",
            "FitAvailabilityDomain_SHA256",
        ],
        "digest": "canonical component-and-row membership SHA-256",
        "gate_checks": [
            "game_identity_and_order_exact",
            "M0_bitwise_exact",
            "cluster_single_block",
            "fit_prediction_cluster_disjoint",
            "full_and_eligible_inventory_exact",
            "fitter_population_exact_from_feature_target_intersection",
            "maximal_monotone_fixed_point_rederived",
        ],
        "cluster_source_bytes_verified": True,
        "feature_domain_authoritative_loader": (
            "future_private_runner_required_before_candidate_outcomes"
        ),
        "feature_fixture_builder_authoritative": False,
    }:
        raise RepresentationRankAssayError("gate binding changed")
    if payload.get("target_binding") != {
        "interface": "TargetDomain keyed by exact game_id",
        "ordered_arrays_accepted": False,
        "digest": "canonical keyed target membership SHA-256",
        "authoritative_loader": (
            "future_private_runner must verify pinned nonholdout parquet bytes "
            "and logical rows"
        ),
        "pending_shell_authoritative_source_verified": False,
        "identity_rule": (
            "TargetDomain IDs equal M0 IDs equal FitAvailabilityDomain IDs"
        ),
    }:
        raise RepresentationRankAssayError("target binding changed")
    if payload.get("penalty_selection") != {
        "population": "fixed_train_only",
        "width": 8,
        "prediction_months": list(INNER_MONTHS),
        "grid": list(PENALTY_GRID),
        "ally": "tune lambda_A on ally-only width-8 fits",
        "enemy": "tune lambda_E on enemy-only width-8 fits",
        "joint_cartesian_search": False,
        "fold_rule": (
            "cluster-atomic expanding month OOF with strictly earlier train fit rows"
        ),
        "primary": "map-weighted mean natural log loss",
        "tie_tolerance": 1e-12,
        "tie_break_1": "lower map-weighted Brier score",
        "tie_break_2": "larger lambda",
        "eligible_months": "exactly_all_six_fixed_months",
        "minimum_inner_oof_maps": 1500,
        "insufficient_support": "assay_unavailable_no_fallback",
        "freeze_for_all_outer_folds": True,
    }:
        raise RepresentationRankAssayError("penalty protocol changed")
    if payload.get("objective") != {
        "loss": "mean binary natural log loss",
        "ally_penalty": (
            "lambda_A * ||A_centered||^2 / (2 * N_eligible_nodes)"
        ),
        "enemy_penalty": (
            "lambda_E * ||E_centered||^2 / (2 * N_eligible_nodes)"
        ),
        "penalty_denominator": (
            "N_eligible_nodes frozen by the chronological PreparedFold component; "
            "unused or ineligible vocabulary cannot change regularization"
        ),
        "ally_score_normalization": 10,
        "enemy_score_normalization": 25,
        "offset": "exact logit(p_M0)",
    }:
        raise RepresentationRankAssayError("objective changed")
    if payload.get("optimization") != {
        "starts": 3,
        "start_1": "fit_only_residual_informed_nonzero_start",
        "starts_2_3": "fixed-seed nonzero perturbations",
        "zero_padding_without_perturbation": False,
        "selection": "minimum fit-only objective",
        "finite_convergence_required": True,
        "gradient_infinity_norm_maximum": GRADIENT_TOLERANCE,
        "minimum_converged_starts": 2,
        "best_two_fit_interaction_logit_RMS_maximum": STABILITY_RMS_TOLERANCE,
        "failure": "candidate_unavailable",
    }:
        raise RepresentationRankAssayError("optimization protocol changed")
    if payload.get("decision") != {
        "M0": "exact nuisance OOF comparator",
        "M8": "width-8 latent candidate; not full",
        "M8_gate": "must pass M0 superiority and optimization/stability",
        "development": (
            "choose smallest width passing M0 superiority and M8 noninferiority"
        ),
        "validation": "test locked width only; no reselection",
        "failure": "assay_inconclusive_and_fallback_M0",
        "M0_log_loss_upper_strict": 0.0,
        "M8_log_loss_upper": 0.002,
        "brier_upper_against_each": 0.001,
        "calibration_upper_against_each": 0.010,
        "calibration_definition": "abs(mean(y-p))",
        "every_block_log_loss_upper_against_each": 0.010,
    }:
        raise RepresentationRankAssayError("decision contract changed")
    if payload.get("bootstrap") != {
        "bit_generator": "PCG64DXSM",
        "replicates": PRIMARY_REPLICATES,
        "sampling": "uniform whole clusters with replacement",
        "draws_per_replicate": "K",
        "cluster_order": "lexicographic",
        "paired_resamples": True,
        "estimator": "map-weighted ratio of cluster totals",
        "development_seed": DEVELOPMENT_SEED,
        "development_family": "4 widths x 2 hypotheses",
        "development_gamma": 0.99375,
        "development_endpoint_1_indexed": DEVELOPMENT_ENDPOINT,
        "validation_seed": VALIDATION_SEED,
        "validation_family": (
            "M8 superiority to M0; locked superiority to M0; locked "
            "noninferiority to M8"
        ),
        "validation_gamma": 0.9833333333333333,
        "validation_endpoint_1_indexed": VALIDATION_ENDPOINT,
    }:
        raise RepresentationRankAssayError("bootstrap contract changed")
    expected_blocks = {
        split: [
            {"calendar_month": month, "maps": maps, "clusters": clusters}
            for month, maps, clusters in rows
        ]
        for split, rows in CHRONOLOGICAL_BLOCKS.items()
    }
    if payload.get("chronological_blocks") != expected_blocks:
        raise RepresentationRankAssayError("chronological blocks changed")
    expected_eligible_blocks = {
        split: [
            {"calendar_month": month, "maps": maps, "clusters": clusters}
            for month, maps, clusters in rows
        ]
        for split, rows in ELIGIBLE_GATE_BLOCKS.items()
    }
    if payload.get("eligible_gate_blocks") != expected_eligible_blocks:
        raise RepresentationRankAssayError("eligible gate blocks changed")
    if payload.get("chronological_block_rule") != (
        "calendar month of dependence cluster maximum OE date; every map in a "
        "cluster inherits that month"
    ):
        raise RepresentationRankAssayError("block derivation changed")
    if payload.get("final_temporal_holdout") != {
        "maps": 361,
        "status": "sealed_prohibited",
        "targets_loaded": False,
        "fit_rows": 0,
        "predictions": 0,
        "score_rows": 0,
    }:
        raise RepresentationRankAssayError("final holdout changed")
    sources = payload.get("source_identity")
    required_sources = {
        "target_evidence",
        "outcome_free_split",
        "human_authority",
        "dependence_cluster_proxy",
        "champion_crosswalk",
        "nuisance_artifact",
    }
    if not isinstance(sources, Mapping) or set(sources) != required_sources:
        raise RepresentationRankAssayError("source identity incomplete")
    for source in sources.values():
        if (
            not isinstance(source, Mapping)
            or set(source) != {"locator", "raw_sha256"}
            or len(str(source["raw_sha256"])) != 64
        ):
            raise RepresentationRankAssayError("source identity invalid")
    executable = payload.get("executable_identity")
    if (
        not isinstance(executable, Mapping)
        or set(executable)
        != {"module", "generator", "runtime_versions", "bit_generator"}
        or executable.get("bit_generator") != "PCG64DXSM"
    ):
        raise RepresentationRankAssayError("executable identity incomplete")
    for name in ("module", "generator"):
        item = executable[name]
        if (
            not isinstance(item, Mapping)
            or set(item) != {"locator", "raw_sha256"}
            or len(str(item["raw_sha256"])) != 64
        ):
            raise RepresentationRankAssayError("executable source identity invalid")
    runtime = executable["runtime_versions"]
    if not isinstance(runtime, Mapping) or set(runtime) != {
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
    }:
        raise RepresentationRankAssayError("runtime identity incomplete")
    if payload.get("excluded_reference_only") != {
        "champion_representation_contract": {
            "locator": (
                "data/lol/v2/champions/champion-representation-contract-v2.json"
            ),
            "role": "excluded_reference_only_not_consumed",
            "semantic_features_enter_model": False,
        }
    }:
        raise RepresentationRankAssayError("excluded reference boundary changed")


def verify_config_sources(
    payload: Mapping[str, Any], *, root: Path = Path.cwd()
) -> None:
    validate_config(payload)
    for name, source in payload["source_identity"].items():
        path = Path(source["locator"])
        path = path if path.is_absolute() else root / path
        if not path.is_file() or path.is_symlink():
            raise RepresentationRankAssayError(f"{name} is not a regular file")
        if raw_sha256(path) != source["raw_sha256"]:
            raise RepresentationRankAssayError(f"{name} source bytes changed")
    for name, source in (
        ("module", payload["executable_identity"]["module"]),
        ("generator", payload["executable_identity"]["generator"]),
    ):
        path = Path(source["locator"])
        path = path if path.is_absolute() else root / path
        if raw_sha256(path) != source["raw_sha256"]:
            raise RepresentationRankAssayError(f"{name} executable bytes changed")


def require_m0_offsets(
    game_ids: Sequence[object],
    probabilities: Sequence[object],
    verified_oof: Mapping[str, float],
) -> np.ndarray:
    ids = [str(value) for value in game_ids]
    values = np.asarray(probabilities, dtype=float)
    if len(ids) != len(set(ids)) or values.shape != (len(ids),):
        raise RepresentationRankAssayError("M0 game identity invalid")
    if set(ids) - set(verified_oof):
        raise RepresentationRankAssayError("M0 verified offset missing by game_id")
    expected = np.asarray([verified_oof[game_id] for game_id in ids], dtype=float)
    if (
        not np.isfinite(values).all()
        or not np.isfinite(expected).all()
        or np.any(values <= 0)
        or np.any(values >= 1)
        or not np.array_equal(values, expected)
    ):
        raise RepresentationRankAssayError("M0 offset reproduction failed")
    return logit(values)


@dataclass(frozen=True)
class CoverageResult:
    eligible_rows: np.ndarray
    eligible_nodes: np.ndarray
    report: dict[str, Any]
    eligibility_binding: "EligibilityBinding"


@dataclass(frozen=True)
class EligibilityBinding:
    prediction_split: str
    prediction_block: str
    feature_domain: FeatureDomain
    node_domain: NodeDomain
    fit_availability_domain: FitAvailabilityDomain
    cluster_domain_sha256: str
    feature_domain_sha256: str
    fit_availability_domain_sha256: str
    fit_availability_source_raw_sha256: str
    eligible_nodes: tuple[bool, ...]
    ordered_fit_game_ids: tuple[str, ...]
    ordered_source_game_ids: tuple[str, ...]
    ordered_fit_cluster_blocks: tuple[tuple[str, str], ...]
    ordered_source_cluster_ids: tuple[str, ...]
    artifact_sha256: str


@dataclass(frozen=True)
class PreparedFold:
    split: str
    eligible_rows: np.ndarray
    eligibility_bindings_by_block: tuple[EligibilityBinding, ...]
    ordered_full_game_ids: tuple[str, ...]
    ordered_full_cluster_ids: tuple[str, ...]
    ordered_full_blocks: tuple[str, ...]
    ordered_eligible_game_ids: tuple[str, ...]
    ordered_eligible_cluster_ids: tuple[str, ...]
    ordered_eligible_blocks: tuple[str, ...]
    m0_probability: np.ndarray
    membership_sha256: str
    component_membership_sha256: tuple[str, ...]
    coverage_report: dict[str, Any]


def _integer_node_array(value: np.ndarray, *, columns: int, label: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[1] != columns:
        raise RepresentationRankAssayError(f"{label} draft shape invalid")
    if any(
        isinstance(item, (bool, np.bool_))
        or not isinstance(item, numbers.Integral)
        for item in raw.ravel()
    ):
        raise RepresentationRankAssayError(f"{label} node ids must be integers")
    return raw.astype(np.int64, copy=False)


def _validate_drafts(
    *,
    blue_nodes: np.ndarray,
    red_nodes: np.ndarray,
    node_domain: NodeDomain,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate the exact five-role, ten-distinct-champion legal draft domain."""
    blue = _integer_node_array(blue_nodes, columns=5, label=label)
    red = _integer_node_array(red_nodes, columns=5, label=label)
    validate_node_domain(node_domain)
    roles = np.asarray(node_domain.node_roles, dtype=object)
    champion_ids = node_domain.node_champion_ids
    if blue.shape != red.shape:
        raise RepresentationRankAssayError(f"{label} draft side shapes differ")
    combined = np.column_stack((blue, red))
    if (
        np.any(combined < 0)
        or np.any(combined >= len(roles))
        or any(
            np.any(roles[combined[:, position]] != expected_role)
            for position, expected_role in enumerate(CANONICAL_POSITIONS)
        )
        or any(
            np.any(roles[combined[:, 5 + position]] != expected_role)
            for position, expected_role in enumerate(CANONICAL_POSITIONS)
        )
    ):
        raise RepresentationRankAssayError(
            f"{label} draft is outside canonical role domain"
        )
    for row in combined:
        identities = [champion_ids[int(node)] for node in row]
        if len(set(identities)) != 10:
            raise RepresentationRankAssayError(
                f"{label} draft repeats a champion identity"
            )
    return blue, red


def _validate_ten_node_rows(
    rows: np.ndarray,
    *,
    node_domain: NodeDomain,
    label: str,
) -> np.ndarray:
    draft = _integer_node_array(rows, columns=10, label=label)
    blue, red = _validate_drafts(
        blue_nodes=draft[:, :5],
        red_nodes=draft[:, 5:],
        node_domain=node_domain,
        label=label,
    )
    return np.column_stack((blue, red))


def _assert_clusters_single_block(
    clusters: Sequence[object], blocks: Sequence[object]
) -> None:
    observed: dict[str, str] = {}
    for cluster, block in zip(clusters, blocks):
        cluster_id, block_id = str(cluster), str(block)
        prior = observed.setdefault(cluster_id, block_id)
        if prior != block_id:
            raise RepresentationRankAssayError(
                "dependence cluster spans chronological blocks"
            )


def _membership_sha256(
    *,
    split: str,
    full_game_ids: Sequence[str],
    full_cluster_ids: Sequence[str],
    full_blocks: Sequence[str],
    eligible_game_ids: Sequence[str],
    eligible_cluster_ids: Sequence[str],
    eligible_blocks: Sequence[str],
    m0_probability: np.ndarray,
    eligibility_binding_sha256: Sequence[str],
) -> str:
    return canonical_sha256(
        {
            "split": split,
            "ordered_full_rows": [
                {
                    "game_id": str(game_id),
                    "cluster_id": str(cluster_id),
                    "chronological_block": str(block),
                }
                for game_id, cluster_id, block in zip(
                    full_game_ids, full_cluster_ids, full_blocks
                )
            ],
            "ordered_eligible_rows": [
                {
                    "game_id": str(game_id),
                    "cluster_id": str(cluster_id),
                    "chronological_block": str(block),
                    "m0_probability_float64_hex": float(probability).hex(),
                }
                for game_id, cluster_id, block, probability in zip(
                    eligible_game_ids,
                    eligible_cluster_ids,
                    eligible_blocks,
                    m0_probability,
                )
            ],
            "ordered_eligibility_binding_sha256": list(
                eligibility_binding_sha256
            ),
        }
    )


def _eligibility_binding_sha256(binding: EligibilityBinding) -> str:
    return canonical_sha256(
        {
            "prediction_block": binding.prediction_block,
            "prediction_split": binding.prediction_split,
            "node_domain_sha256": binding.node_domain.artifact_sha256,
            "cluster_domain_sha256": binding.cluster_domain_sha256,
            "feature_domain_sha256": binding.feature_domain_sha256,
            "fit_availability_domain_sha256": (
                binding.fit_availability_domain_sha256
            ),
            "fit_availability_source_raw_sha256": (
                binding.fit_availability_source_raw_sha256
            ),
            "eligible_nodes": list(binding.eligible_nodes),
            "ordered_fit_game_ids": list(binding.ordered_fit_game_ids),
            "ordered_source_game_ids": list(binding.ordered_source_game_ids),
            "ordered_fit_cluster_blocks": [
                list(row) for row in binding.ordered_fit_cluster_blocks
            ],
            "ordered_source_cluster_ids": list(
                binding.ordered_source_cluster_ids
            ),
        }
    )


def _validate_eligibility_binding(binding: EligibilityBinding) -> None:
    validate_feature_domain(binding.feature_domain)
    validate_node_domain(binding.node_domain)
    validate_fit_availability_domain(binding.fit_availability_domain)
    if (
        binding.node_domain != binding.feature_domain.node_domain
        or binding.cluster_domain_sha256
        != binding.feature_domain.cluster_domain.artifact_sha256
        or binding.feature_domain_sha256 != binding.feature_domain.artifact_sha256
        or binding.fit_availability_domain_sha256
        != binding.fit_availability_domain.artifact_sha256
        or binding.fit_availability_source_raw_sha256
        != binding.fit_availability_domain.source_raw_sha256
    ):
        raise RepresentationRankAssayError("eligibility source domain changed")
    if (
        len(binding.eligible_nodes) != len(binding.node_domain.node_roles)
        or len(binding.cluster_domain_sha256) != 64
        or len(binding.feature_domain_sha256) != 64
        or len(binding.fit_availability_domain_sha256) != 64
        or len(binding.fit_availability_source_raw_sha256) != 64
        or not any(binding.eligible_nodes)
        or binding.artifact_sha256 != _eligibility_binding_sha256(binding)
    ):
        raise RepresentationRankAssayError("eligibility binding invalid")
    records = {row[0]: row for row in binding.feature_domain.records}
    if (
        len(set(binding.ordered_fit_game_ids)) != len(binding.ordered_fit_game_ids)
        or len(set(binding.ordered_source_game_ids))
        != len(binding.ordered_source_game_ids)
        or set(binding.ordered_fit_game_ids) - set(records)
        or set(binding.ordered_source_game_ids) - set(records)
        or set(binding.ordered_fit_game_ids)
        - set(binding.fit_availability_domain.ordered_game_ids)
        or set(binding.ordered_source_game_ids)
        - set(binding.fit_availability_domain.ordered_game_ids)
    ):
        raise RepresentationRankAssayError("eligibility evidence identity invalid")
    fit_records = [records[game_id] for game_id in binding.ordered_fit_game_ids]
    source_records = [
        records[game_id] for game_id in binding.ordered_source_game_ids
    ]
    if (
        any(row[1] != binding.prediction_split for row in source_records)
        or any(row[3] != binding.prediction_block for row in source_records)
        or tuple(row[2] for row in source_records)
        != binding.ordered_source_cluster_ids
    ):
        raise RepresentationRankAssayError("eligibility prediction evidence changed")
    fit_clusters = tuple(row[2] for row in fit_records)
    fit_blocks = tuple(row[3] for row in fit_records)
    _assert_clusters_single_block(fit_clusters, fit_blocks)
    observed_cluster_blocks = tuple(sorted(set(zip(fit_clusters, fit_blocks))))
    if observed_cluster_blocks != binding.ordered_fit_cluster_blocks:
        raise RepresentationRankAssayError("eligibility fit evidence changed")
    if (
        any(block >= binding.prediction_block for block in fit_blocks)
        or set(fit_clusters) & set(binding.ordered_source_cluster_ids)
    ):
        raise RepresentationRankAssayError(
            "eligibility fit clusters are not atomic strictly earlier support"
        )
    fit_nodes = _validate_ten_node_rows(
        np.asarray([row[5] for row in fit_records], dtype=object),
        node_domain=binding.node_domain,
        label="eligibility fit evidence",
    )
    recomputed = np.zeros(len(binding.node_domain.node_roles), dtype=bool)
    fit_cluster_values = np.asarray(fit_clusters, dtype=object)
    for node in range(len(recomputed)):
        recomputed[node] = (
            len(set(fit_cluster_values[np.any(fit_nodes == node, axis=1)]))
            >= MIN_NODE_CLUSTERS
        )
    if not np.array_equal(
        recomputed, np.asarray(binding.eligible_nodes, dtype=bool)
    ):
        raise RepresentationRankAssayError(
            "eligibility node mask does not match bound fit evidence"
        )


def _validate_prepared_fold(prepared: PreparedFold, *, split: str) -> None:
    if prepared.split != split:
        raise RepresentationRankAssayError("prepared fold split changed")
    rows = len(prepared.ordered_eligible_game_ids)
    if (
        len(set(prepared.ordered_eligible_game_ids)) != rows
        or len(prepared.ordered_eligible_cluster_ids) != rows
        or len(prepared.ordered_eligible_blocks) != rows
        or prepared.m0_probability.shape != (rows,)
        or not np.isfinite(prepared.m0_probability).all()
        or np.any(prepared.m0_probability <= 0)
        or np.any(prepared.m0_probability >= 1)
        or prepared.eligible_rows.dtype != np.bool_
        or int(prepared.eligible_rows.sum()) != rows
        or tuple(
            binding.prediction_block
            for binding in prepared.eligibility_bindings_by_block
        )
        != tuple(month for month, _, _ in ELIGIBLE_GATE_BLOCKS[split])
        or len(prepared.component_membership_sha256)
        != len(ELIGIBLE_GATE_BLOCKS[split])
    ):
        raise RepresentationRankAssayError("prepared fold membership invalid")
    if (
        len(prepared.ordered_full_game_ids) != len(prepared.eligible_rows)
        or len(set(prepared.ordered_full_game_ids))
        != len(prepared.ordered_full_game_ids)
        or len(prepared.ordered_full_cluster_ids)
        != len(prepared.ordered_full_game_ids)
        or len(prepared.ordered_full_blocks) != len(prepared.ordered_full_game_ids)
    ):
        raise RepresentationRankAssayError("prepared full membership invalid")
    for binding in prepared.eligibility_bindings_by_block:
        _validate_eligibility_binding(binding)
    _assert_clusters_single_block(
        prepared.ordered_full_cluster_ids, prepared.ordered_full_blocks
    )
    full_ids_array = np.asarray(prepared.ordered_full_game_ids, dtype=object)
    full_clusters_array = np.asarray(
        prepared.ordered_full_cluster_ids, dtype=object
    )
    full_blocks_array = np.asarray(prepared.ordered_full_blocks, dtype=object)
    if (
        tuple(full_ids_array[prepared.eligible_rows])
        != prepared.ordered_eligible_game_ids
        or tuple(full_clusters_array[prepared.eligible_rows])
        != prepared.ordered_eligible_cluster_ids
        or tuple(full_blocks_array[prepared.eligible_rows])
        != prepared.ordered_eligible_blocks
    ):
        raise RepresentationRankAssayError(
            "prepared eligible rows do not filter full membership exactly"
        )
    _validate_inventory_transition(
        full_blocks_array,
        full_clusters_array,
        prepared.eligible_rows,
        split=split,
    )
    _assert_clusters_single_block(
        prepared.ordered_eligible_cluster_ids, prepared.ordered_eligible_blocks
    )
    _validate_frozen_blocks(
        np.asarray(prepared.ordered_eligible_blocks, dtype=object),
        np.asarray(prepared.ordered_eligible_cluster_ids, dtype=object),
        split=split,
    )
    eligible_blocks_array = np.asarray(
        prepared.ordered_eligible_blocks, dtype=object
    )
    eligible_ids_array = np.asarray(
        prepared.ordered_eligible_game_ids, dtype=object
    )
    eligible_clusters_array = np.asarray(
        prepared.ordered_eligible_cluster_ids, dtype=object
    )
    for index, (block, _, _) in enumerate(CHRONOLOGICAL_BLOCKS[split]):
        full_selected = full_blocks_array == block
        eligible_selected = eligible_blocks_array == block
        binding = prepared.eligibility_bindings_by_block[index]
        if (
            binding.prediction_split != prepared.split
            or binding.prediction_block != block
            or binding.ordered_source_game_ids
            != tuple(full_ids_array[full_selected])
            or binding.ordered_source_cluster_ids
            != tuple(full_clusters_array[full_selected])
        ):
            raise RepresentationRankAssayError(
                "prepared component source evidence differs from parent"
            )
        _validate_inventory_block(
            full_blocks_array[full_selected],
            full_clusters_array[full_selected],
            prepared.eligible_rows[full_selected],
            split=split,
            block=block,
        )
        component_digest = _membership_sha256(
            split=split,
            full_game_ids=tuple(full_ids_array[full_selected]),
            full_cluster_ids=tuple(full_clusters_array[full_selected]),
            full_blocks=tuple(full_blocks_array[full_selected]),
            eligible_game_ids=tuple(eligible_ids_array[eligible_selected]),
            eligible_cluster_ids=tuple(
                eligible_clusters_array[eligible_selected]
            ),
            eligible_blocks=tuple(eligible_blocks_array[eligible_selected]),
            m0_probability=prepared.m0_probability[eligible_selected],
            eligibility_binding_sha256=[binding.artifact_sha256],
        )
        if prepared.component_membership_sha256[index] != component_digest:
            raise RepresentationRankAssayError(
                "prepared component membership digest mismatch"
            )
    ordered_digest = _membership_sha256(
        split=split,
        full_game_ids=prepared.ordered_full_game_ids,
        full_cluster_ids=prepared.ordered_full_cluster_ids,
        full_blocks=prepared.ordered_full_blocks,
        eligible_game_ids=prepared.ordered_eligible_game_ids,
        eligible_cluster_ids=prepared.ordered_eligible_cluster_ids,
        eligible_blocks=prepared.ordered_eligible_blocks,
        m0_probability=prepared.m0_probability,
        eligibility_binding_sha256=[
            binding.artifact_sha256
            for binding in prepared.eligibility_bindings_by_block
        ],
    )
    expected = canonical_sha256(
        {
            "split": split,
            "ordered_component_membership_sha256": list(
                prepared.component_membership_sha256
            ),
            "ordered_eligible_membership_sha256": ordered_digest,
        }
    )
    if prepared.membership_sha256 != expected:
        raise RepresentationRankAssayError("prepared fold membership digest mismatch")


def outcome_free_coverage(
    *,
    feature_domain: FeatureDomain,
    score_game_ids: Sequence[object],
    fit_game_ids: Sequence[object],
    split: str,
    fit_availability_domain: FitAvailabilityDomain,
) -> CoverageResult:
    """Derive one identical M0/all-width row mask without reading outcomes."""
    validate_feature_domain(feature_domain)
    records = {row[0]: row for row in feature_domain.records}
    score_ids = tuple(str(value) for value in score_game_ids)
    fit_ids = tuple(str(value) for value in fit_game_ids)
    validate_fit_availability_domain(fit_availability_domain)
    available = set(fit_availability_domain.ordered_game_ids)
    if set(score_ids) - available or set(fit_ids) - available:
        raise RepresentationRankAssayError(
            "coverage row lacks verified fit availability"
        )
    if (
        len(score_ids) != len(set(score_ids))
        or len(fit_ids) != len(set(fit_ids))
        or set(score_ids) - set(records)
        or set(fit_ids) - set(records)
    ):
        raise RepresentationRankAssayError("coverage game identity invalid")
    score_records = [records[game_id] for game_id in score_ids]
    fit_records = [records[game_id] for game_id in fit_ids]
    if any(row[1] != split for row in score_records):
        raise RepresentationRankAssayError("coverage split identity changed")
    rows = _validate_ten_node_rows(
        np.asarray([row[5] for row in score_records], dtype=object),
        node_domain=feature_domain.node_domain,
        label="score",
    )
    fit = _validate_ten_node_rows(
        np.asarray([row[5] for row in fit_records], dtype=object),
        node_domain=feature_domain.node_domain,
        label="source fit",
    )
    clusters = np.asarray([row[2] for row in score_records], dtype=object)
    months = np.asarray([row[3] for row in score_records], dtype=object)
    leagues = np.asarray([row[4] for row in score_records], dtype=object)
    fit_cluster_values = np.asarray([row[2] for row in fit_records], dtype=object)
    fit_month_values = np.asarray([row[3] for row in fit_records], dtype=object)
    _assert_clusters_single_block(clusters, months)
    _assert_clusters_single_block(fit_cluster_values, fit_month_values)
    if set(clusters) & set(fit_cluster_values):
        raise RepresentationRankAssayError(
            "prediction and eligibility-fit clusters overlap"
        )
    if len(set(months)) != 1 or any(
        month >= str(months[0]) for month in fit_month_values
    ):
        raise RepresentationRankAssayError(
            "feature eligibility fit rows are not strictly earlier"
        )
    node_count = len(feature_domain.node_domain.node_roles)
    active_fit_rows = np.ones(len(fit), dtype=bool)
    convergence_checks = 0
    changing_rounds = 0
    while True:
        convergence_checks += 1
        eligible_nodes = np.zeros(node_count, dtype=bool)
        for node in range(node_count):
            distinct = set(
                fit_cluster_values[
                    active_fit_rows & np.any(fit == node, axis=1)
                ]
            )
            eligible_nodes[node] = len(distinct) >= MIN_NODE_CLUSTERS
        updated = active_fit_rows & np.all(eligible_nodes[fit], axis=1)
        if np.array_equal(updated, active_fit_rows):
            break
        active_fit_rows = updated
        changing_rounds += 1
    eligible_rows = np.all(eligible_nodes[rows], axis=1)

    def coverage(mask: np.ndarray) -> dict[str, Any]:
        total_clusters = len(set(clusters[mask]))
        kept_clusters = len(set(clusters[mask & eligible_rows]))
        total_maps = int(mask.sum())
        kept_maps = int((mask & eligible_rows).sum())
        return {
            "maps": total_maps,
            "eligible_maps": kept_maps,
            "map_fraction": kept_maps / total_maps if total_maps else None,
            "clusters": total_clusters,
            "eligible_clusters": kept_clusters,
            "cluster_fraction": kept_clusters / total_clusters
            if total_clusters
            else None,
        }

    overall = coverage(np.ones(len(rows), dtype=bool))
    month_rows = {
        month: coverage(months == month) for month in sorted(set(months))
    }
    league_rows = {
        league: coverage(leagues == league) for league in sorted(set(leagues))
    }
    passed = coverage_gate_decision(
        overall=overall,
        month_rows=tuple(month_rows.values()),
        league_rows=tuple(league_rows.values()),
    )
    report = {
        "split": split,
        "passed": passed,
        "overall": overall,
        "by_month": month_rows,
        "by_league": league_rows,
        "excluded_maps": [
            {"row_index": int(index), "reason": "node_below_5_fit_clusters"}
            for index in np.flatnonzero(~eligible_rows)
        ],
        "fit_support": {
            "derivation": "maximal_monotone_fixed_point",
            "input_maps": len(fit_ids),
            "retained_maps": int(active_fit_rows.sum()),
            "eligible_nodes": int(eligible_nodes.sum()),
            "convergence_checks": convergence_checks,
            "changing_rounds": changing_rounds,
            "fit_availability_bound": True,
        },
    }
    retained_fit_ids = tuple(
        game_id
        for game_id, retained in zip(fit_ids, active_fit_rows)
        if bool(retained)
    )
    fit_cluster_blocks = tuple(
        sorted(
            set(
                zip(
                    fit_cluster_values[active_fit_rows].tolist(),
                    fit_month_values[active_fit_rows].tolist(),
                )
            )
        )
    )
    provisional = EligibilityBinding(
        prediction_split=split,
        prediction_block=str(months[0]),
        feature_domain=feature_domain,
        node_domain=feature_domain.node_domain,
        fit_availability_domain=fit_availability_domain,
        cluster_domain_sha256=feature_domain.cluster_domain.artifact_sha256,
        feature_domain_sha256=feature_domain.artifact_sha256,
        fit_availability_domain_sha256=fit_availability_domain.artifact_sha256,
        fit_availability_source_raw_sha256=(
            fit_availability_domain.source_raw_sha256
        ),
        eligible_nodes=tuple(bool(value) for value in eligible_nodes),
        ordered_fit_game_ids=retained_fit_ids,
        ordered_source_game_ids=score_ids,
        ordered_fit_cluster_blocks=fit_cluster_blocks,
        ordered_source_cluster_ids=tuple(str(value) for value in clusters),
        artifact_sha256="",
    )
    binding = EligibilityBinding(
        **{
            **provisional.__dict__,
            "artifact_sha256": _eligibility_binding_sha256(provisional),
        }
    )
    _validate_eligibility_binding(binding)
    return CoverageResult(eligible_rows, eligible_nodes, report, binding)


def prepare_outer_fold(
    *,
    feature_domain: FeatureDomain,
    score_game_ids: Sequence[object],
    fit_game_ids: Sequence[object],
    nuisance_probability: Sequence[object],
    verified_nuisance_oof: Mapping[str, float],
    split: str,
    fit_availability_domain: FitAvailabilityDomain,
) -> PreparedFold:
    if split not in {"development", "validation"}:
        raise RepresentationRankAssayError("outer fold split is not allowed")
    coverage = outcome_free_coverage(
        feature_domain=feature_domain,
        score_game_ids=score_game_ids,
        fit_game_ids=fit_game_ids,
        split=split,
        fit_availability_domain=fit_availability_domain,
    )
    month_coverage = next(iter(coverage.report["by_month"].values()), {})
    if not (
        len(coverage.report["by_month"]) == 1
        and month_coverage.get("map_fraction") is not None
        and month_coverage["map_fraction"] >= 2 / 3
        and month_coverage.get("eligible_clusters", 0) >= 15
    ):
        raise RepresentationRankAssayError(
            "outcome-free monthly coverage gate failed"
        )
    records = {row[0]: row for row in feature_domain.records}
    ids = tuple(str(value) for value in score_game_ids)
    months_array = np.asarray([records[game_id][3] for game_id in ids], dtype=object)
    clusters_array = np.asarray([records[game_id][2] for game_id in ids], dtype=object)
    block = coverage.eligibility_binding.prediction_block
    _validate_inventory_block(
        months_array,
        clusters_array,
        coverage.eligible_rows,
        split=split,
        block=block,
    )
    if len(ids) != len(set(ids)):
        raise RepresentationRankAssayError("outer fold game identity invalid")
    probability = np.asarray(nuisance_probability, dtype=float)
    require_m0_offsets(ids, probability, verified_nuisance_oof)
    if len(ids) != len(coverage.eligible_rows):
        raise RepresentationRankAssayError("coverage/M0 row membership mismatch")
    eligible_game_ids = tuple(
        game_id for game_id, eligible in zip(ids, coverage.eligible_rows) if eligible
    )
    eligible_clusters = tuple(
        str(cluster)
        for cluster, eligible in zip(clusters_array, coverage.eligible_rows)
        if eligible
    )
    eligible_blocks = tuple(
        str(block)
        for block, eligible in zip(months_array, coverage.eligible_rows)
        if eligible
    )
    eligible_probability = probability[coverage.eligible_rows].copy()
    component_digest = _membership_sha256(
        split=split,
        full_game_ids=ids,
        full_cluster_ids=tuple(str(value) for value in clusters_array),
        full_blocks=tuple(str(value) for value in months_array),
        eligible_game_ids=eligible_game_ids,
        eligible_cluster_ids=eligible_clusters,
        eligible_blocks=eligible_blocks,
        m0_probability=eligible_probability,
        eligibility_binding_sha256=[
            coverage.eligibility_binding.artifact_sha256
        ],
    )
    return PreparedFold(
        split=split,
        eligible_rows=coverage.eligible_rows.copy(),
        eligibility_bindings_by_block=(coverage.eligibility_binding,),
        ordered_full_game_ids=ids,
        ordered_full_cluster_ids=tuple(str(value) for value in clusters_array),
        ordered_full_blocks=tuple(str(value) for value in months_array),
        ordered_eligible_game_ids=eligible_game_ids,
        ordered_eligible_cluster_ids=eligible_clusters,
        ordered_eligible_blocks=eligible_blocks,
        m0_probability=eligible_probability,
        membership_sha256=component_digest,
        component_membership_sha256=(component_digest,),
        coverage_report=coverage.report,
    )


def combine_prepared_folds(
    folds: Sequence[PreparedFold], *, split: str
) -> PreparedFold:
    """Bind ordered per-month eligibility without leaking later fit support."""
    expected_blocks = tuple(month for month, _, _ in ELIGIBLE_GATE_BLOCKS[split])
    if (
        len(folds) != len(expected_blocks)
        or tuple(
            fold.ordered_eligible_blocks[0]
            if fold.ordered_eligible_blocks
            else ""
            for fold in folds
        )
        != expected_blocks
        or any(fold.split != split for fold in folds)
        or any(len(set(fold.ordered_eligible_blocks)) != 1 for fold in folds)
    ):
        raise RepresentationRankAssayError(
            "prepared chronological components changed"
        )
    for fold in folds:
        block = fold.ordered_eligible_blocks[0]
        expected_component = _membership_sha256(
            split=split,
            full_game_ids=fold.ordered_full_game_ids,
            full_cluster_ids=fold.ordered_full_cluster_ids,
            full_blocks=fold.ordered_full_blocks,
            eligible_game_ids=fold.ordered_eligible_game_ids,
            eligible_cluster_ids=fold.ordered_eligible_cluster_ids,
            eligible_blocks=fold.ordered_eligible_blocks,
            m0_probability=fold.m0_probability,
            eligibility_binding_sha256=[
                binding.artifact_sha256
                for binding in fold.eligibility_bindings_by_block
            ],
        )
        if (
            fold.membership_sha256 != expected_component
            or fold.component_membership_sha256 != (expected_component,)
            or fold.eligibility_bindings_by_block[0].prediction_block != block
        ):
            raise RepresentationRankAssayError(
                "prepared component membership digest mismatch"
            )
    full_game_ids = tuple(
        game_id for fold in folds for game_id in fold.ordered_full_game_ids
    )
    if len(full_game_ids) != len(set(full_game_ids)):
        raise RepresentationRankAssayError("prepared full game identity duplicated")
    full_clusters = tuple(
        cluster for fold in folds for cluster in fold.ordered_full_cluster_ids
    )
    full_blocks = tuple(
        block for fold in folds for block in fold.ordered_full_blocks
    )
    _assert_clusters_single_block(full_clusters, full_blocks)
    game_ids = tuple(
        game_id for fold in folds for game_id in fold.ordered_eligible_game_ids
    )
    if len(game_ids) != len(set(game_ids)):
        raise RepresentationRankAssayError("prepared game identity duplicated")
    clusters = tuple(
        cluster
        for fold in folds
        for cluster in fold.ordered_eligible_cluster_ids
    )
    blocks = tuple(
        block for fold in folds for block in fold.ordered_eligible_blocks
    )
    probability = np.concatenate([fold.m0_probability for fold in folds])
    component_digests = tuple(fold.membership_sha256 for fold in folds)
    ordered_digest = _membership_sha256(
        split=split,
        full_game_ids=full_game_ids,
        full_cluster_ids=full_clusters,
        full_blocks=full_blocks,
        eligible_game_ids=game_ids,
        eligible_cluster_ids=clusters,
        eligible_blocks=blocks,
        m0_probability=probability,
        eligibility_binding_sha256=[
            binding.artifact_sha256
            for fold in folds
            for binding in fold.eligibility_bindings_by_block
        ],
    )
    membership_sha256 = canonical_sha256(
        {
            "split": split,
            "ordered_component_membership_sha256": list(component_digests),
            "ordered_eligible_membership_sha256": ordered_digest,
        }
    )
    total_maps = sum(
        fold.coverage_report["overall"]["maps"] for fold in folds
    )
    eligible_maps = sum(
        fold.coverage_report["overall"]["eligible_maps"] for fold in folds
    )
    total_clusters = sum(
        fold.coverage_report["overall"]["clusters"] for fold in folds
    )
    eligible_clusters = sum(
        fold.coverage_report["overall"]["eligible_clusters"] for fold in folds
    )
    leagues = sorted(
        {
            league
            for fold in folds
            for league in fold.coverage_report["by_league"]
        }
    )
    league_coverage = {}
    for league in leagues:
        values = [
            fold.coverage_report["by_league"].get(league)
            for fold in folds
        ]
        values = [value for value in values if value is not None]
        league_maps = sum(value["maps"] for value in values)
        league_kept = sum(value["eligible_maps"] for value in values)
        league_clusters = sum(value["clusters"] for value in values)
        league_coverage[league] = {
            "maps": league_maps,
            "eligible_maps": league_kept,
            "map_fraction": league_kept / league_maps,
            "clusters": league_clusters,
        }
    aggregate_passed = bool(
        eligible_maps / total_maps >= 0.8
        and eligible_clusters / total_clusters >= 0.8
        and all(
            value["map_fraction"] >= 0.75
            for value in league_coverage.values()
            if value["maps"] >= 30 and value["clusters"] >= 10
        )
    )
    if not aggregate_passed:
        raise RepresentationRankAssayError(
            "outcome-free aggregate coverage gate failed"
        )
    prepared = PreparedFold(
        split=split,
        eligible_rows=np.concatenate([fold.eligible_rows for fold in folds]),
        eligibility_bindings_by_block=tuple(
            binding
            for fold in folds
            for binding in fold.eligibility_bindings_by_block
        ),
        ordered_full_game_ids=full_game_ids,
        ordered_full_cluster_ids=full_clusters,
        ordered_full_blocks=full_blocks,
        ordered_eligible_game_ids=game_ids,
        ordered_eligible_cluster_ids=clusters,
        ordered_eligible_blocks=blocks,
        m0_probability=probability,
        membership_sha256=membership_sha256,
        component_membership_sha256=component_digests,
        coverage_report={
            "split": split,
            "passed": aggregate_passed,
            "overall": {
                "maps": total_maps,
                "eligible_maps": eligible_maps,
                "map_fraction": eligible_maps / total_maps,
                "clusters": total_clusters,
                "eligible_clusters": eligible_clusters,
                "cluster_fraction": eligible_clusters / total_clusters,
            },
            "by_league": league_coverage,
            "chronological_components": [
                fold.coverage_report for fold in folds
            ],
        },
    )
    _validate_prepared_fold(prepared, split=split)
    return prepared


def _validate_inventory_block(
    months: np.ndarray,
    clusters: np.ndarray,
    eligible_rows: np.ndarray,
    *,
    split: str,
    block: str,
) -> None:
    source = {month: (maps, count) for month, maps, count in CHRONOLOGICAL_BLOCKS[split]}
    eligible = {
        month: (maps, count) for month, maps, count in ELIGIBLE_GATE_BLOCKS[split]
    }
    if block not in source or set(months) != {block}:
        raise RepresentationRankAssayError("prediction block is not frozen")
    _assert_clusters_single_block(clusters, months)
    for mask, inventory, label in (
        (np.ones(len(months), dtype=bool), source, "source"),
        (eligible_rows, eligible, "eligible"),
    ):
        maps, cluster_count = inventory[block]
        if int(mask.sum()) != maps or len(set(clusters[mask])) != cluster_count:
            raise RepresentationRankAssayError(
                f"{label} block inventory arithmetic changed"
            )


def _validate_inventory_transition(
    months: np.ndarray,
    clusters: np.ndarray,
    eligible_rows: np.ndarray,
    *,
    split: str,
) -> None:
    if split not in CHRONOLOGICAL_BLOCKS:
        raise RepresentationRankAssayError("inventory split is not frozen")
    _assert_clusters_single_block(clusters, months)
    for expected_rows, mask, label in (
        (CHRONOLOGICAL_BLOCKS[split], np.ones(len(months), dtype=bool), "source"),
        (ELIGIBLE_GATE_BLOCKS[split], eligible_rows, "eligible"),
    ):
        expected = {
            month: (maps, cluster_count)
            for month, maps, cluster_count in expected_rows
        }
        observed_months = set(months[mask])
        if observed_months != set(expected):
            raise RepresentationRankAssayError(f"{label} inventory months changed")
        for month, (maps, cluster_count) in expected.items():
            selected = mask & (months == month)
            if (
                int(selected.sum()) != maps
                or len(set(clusters[selected])) != cluster_count
            ):
                raise RepresentationRankAssayError(
                    f"{label} inventory arithmetic changed"
                )


def load_nonholdout_rows(
    path: Path, *, columns: Sequence[str]
) -> Any:
    """Predicate-filter nonholdout parquet rows before target materialization."""
    import pandas as pd

    if not path.is_file() or path.is_symlink():
        raise RepresentationRankAssayError("private rows are not a regular file")
    requested = tuple(str(column) for column in columns)
    if "split" not in requested:
        raise RepresentationRankAssayError("loader requires split identity")
    frame = pd.read_parquet(
        path,
        columns=list(requested),
        filters=[("split", "in", list(ALLOWED_NONHOLDOUT_SPLITS))],
    )
    if FINAL_SPLIT in set(frame["split"].astype(str)):
        raise RepresentationRankAssayError("sealed final holdout materialized")
    if not set(frame["split"].astype(str)).issubset(ALLOWED_NONHOLDOUT_SPLITS):
        raise RepresentationRankAssayError("unknown split materialized")
    return frame


def center_by_role(
    values: np.ndarray, node_roles: Sequence[object], eligible_nodes: np.ndarray
) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    roles = np.asarray([str(value) for value in node_roles], dtype=object)
    if result.ndim != 2 or roles.shape != (len(result),):
        raise RepresentationRankAssayError("centering inputs invalid")
    for role in sorted(set(roles[eligible_nodes])):
        selected = (roles == role) & eligible_nodes
        if not np.any(selected):
            raise RepresentationRankAssayError("eligible role has no nodes")
        result[selected] -= result[selected].mean(axis=0, keepdims=True)
    result[~eligible_nodes] = 0.0
    return result


def symplectic_J(width: int) -> np.ndarray:
    if width not in WIDTHS:
        raise RepresentationRankAssayError("width not preregistered")
    matrix = np.zeros((2 * width, 2 * width))
    for index in range(width):
        start = 2 * index
        matrix[start, start + 1] = 1.0
        matrix[start + 1, start] = -1.0
    return matrix


def interaction_logits(
    blue_nodes: np.ndarray,
    red_nodes: np.ndarray,
    ally: np.ndarray,
    enemy: np.ndarray,
    *,
    width: int,
    node_domain: NodeDomain,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    blue, red = _validate_drafts(
        blue_nodes=blue_nodes,
        red_nodes=red_nodes,
        node_domain=node_domain,
        label="score",
    )
    ally_values = np.asarray(ally, dtype=float)
    enemy_values = np.asarray(enemy, dtype=float)
    if (
        ally_values.shape != (len(node_domain.node_roles), width)
        or enemy_values.shape != (len(node_domain.node_roles), 2 * width)
        or not np.isfinite(ally_values).all()
        or not np.isfinite(enemy_values).all()
    ):
        raise RepresentationRankAssayError("latent score arrays invalid")
    blue_ally = ally_values[blue]
    red_ally = ally_values[red]
    blue_ally_sum = blue_ally.sum(axis=1)
    red_ally_sum = red_ally.sum(axis=1)
    blue_pairs = (
        np.sum(blue_ally_sum * blue_ally_sum, axis=1)
        - np.sum(blue_ally * blue_ally, axis=(1, 2))
    ) / 2.0
    red_pairs = (
        np.sum(red_ally_sum * red_ally_sum, axis=1)
        - np.sum(red_ally * red_ally, axis=(1, 2))
    ) / 2.0
    ally_score = (blue_pairs - red_pairs) / 10.0
    J = symplectic_J(width)
    blue_enemy_sum = enemy_values[blue].sum(axis=1)
    red_enemy_sum = enemy_values[red].sum(axis=1)
    enemy_score = np.einsum(
        "ni,ij,nj->n", blue_enemy_sum, J, red_enemy_sum
    ) / 25.0
    return ally_score, enemy_score, ally_score + enemy_score


def _gradient_parameters(
    residual: np.ndarray,
    blue: np.ndarray,
    red: np.ndarray,
    ally: np.ndarray,
    enemy: np.ndarray,
    width: int,
) -> tuple[np.ndarray, np.ndarray]:
    gradient_ally = np.zeros_like(ally)
    gradient_enemy = np.zeros_like(enemy)
    J = symplectic_J(width)
    blue_ally = ally[blue]
    red_ally = ally[red]
    blue_ally_other = blue_ally.sum(axis=1)[:, None, :] - blue_ally
    red_ally_other = red_ally.sum(axis=1)[:, None, :] - red_ally
    np.add.at(
        gradient_ally,
        blue.ravel(),
        (residual[:, None, None] * blue_ally_other / 10.0).reshape(
            -1, ally.shape[1]
        ),
    )
    np.add.at(
        gradient_ally,
        red.ravel(),
        (-residual[:, None, None] * red_ally_other / 10.0).reshape(
            -1, ally.shape[1]
        ),
    )
    blue_enemy_sum = enemy[blue].sum(axis=1)
    red_enemy_sum = enemy[red].sum(axis=1)
    blue_enemy_gradient = np.einsum("ij,nj->ni", J, red_enemy_sum)
    red_enemy_gradient = np.einsum("ij,nj->ni", J.T, blue_enemy_sum)
    np.add.at(
        gradient_enemy,
        blue.ravel(),
        np.repeat(
            residual[:, None] * blue_enemy_gradient / 25.0, 5, axis=0
        ),
    )
    np.add.at(
        gradient_enemy,
        red.ravel(),
        np.repeat(
            residual[:, None] * red_enemy_gradient / 25.0, 5, axis=0
        ),
    )
    return gradient_ally, gradient_enemy


@dataclass(frozen=True)
class LatentFit:
    width: int
    ally_centered: np.ndarray
    enemy_centered: np.ndarray
    objective: float
    maximum_absolute_gradient: float
    converged_starts: int
    best_two_interaction_logit_rms: float
    eligibility_binding_sha256: str


def latent_objective_and_gradient(
    vector: np.ndarray,
    *,
    blue: np.ndarray,
    red: np.ndarray,
    target: np.ndarray,
    p0: np.ndarray,
    node_domain: NodeDomain,
    eligible_nodes: np.ndarray,
    width: int,
    lambda_ally: float,
    lambda_enemy: float,
    mode: str,
) -> tuple[float, np.ndarray]:
    validate_node_domain(node_domain)
    node_roles = node_domain.node_roles
    node_count = len(node_roles)
    eligible = np.asarray(eligible_nodes)
    if eligible.shape != (node_count,) or eligible.dtype != np.bool_:
        raise RepresentationRankAssayError("eligible-node mask invalid")
    eligible_count = int(eligible.sum())
    if eligible_count == 0:
        raise RepresentationRankAssayError("no eligible nodes")
    validated_blue, validated_red = _validate_drafts(
        blue_nodes=blue,
        red_nodes=red,
        node_domain=node_domain,
        label="objective",
    )
    active = np.unique(np.concatenate((validated_blue.ravel(), validated_red.ravel())))
    if np.any(~eligible[active]):
        raise RepresentationRankAssayError(
            "active unseen/ineligible node reached objective"
        )
    split = node_count * width
    raw_a = vector[:split].reshape(node_count, width)
    raw_e = vector[split:].reshape(node_count, 2 * width)
    ally = center_by_role(raw_a, node_roles, eligible_nodes)
    enemy = center_by_role(raw_e, node_roles, eligible_nodes)
    if mode == "ally_only":
        enemy[:] = 0
    elif mode == "enemy_only":
        ally[:] = 0
    _, _, interaction = interaction_logits(
        validated_blue,
        validated_red,
        ally,
        enemy,
        width=width,
        node_domain=node_domain,
    )
    eta = logit(p0) + interaction
    probability = expit(eta)
    value = float(np.mean(np.logaddexp(0, eta) - target * eta))
    value += lambda_ally * float(np.sum(ally * ally)) / (2 * eligible_count)
    value += lambda_enemy * float(np.sum(enemy * enemy)) / (2 * eligible_count)
    gradient_ally, gradient_enemy = _gradient_parameters(
        (probability - target) / len(target),
        validated_blue,
        validated_red,
        ally,
        enemy,
        width,
    )
    gradient_ally += lambda_ally * ally / eligible_count
    gradient_enemy += lambda_enemy * enemy / eligible_count
    gradient_ally = center_by_role(
        gradient_ally, node_roles, eligible_nodes
    )
    gradient_enemy = center_by_role(
        gradient_enemy, node_roles, eligible_nodes
    )
    if mode == "ally_only":
        gradient_enemy[:] = 0
    elif mode == "enemy_only":
        gradient_ally[:] = 0
    return value, np.concatenate((gradient_ally.ravel(), gradient_enemy.ravel()))


def deterministic_starts(
    eligible_node_ids: Sequence[object],
    width: int,
    fit_nodes: np.ndarray,
    residual: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """One fit-only residual start and two fixed nonzero perturbation starts."""
    eligible = tuple(int(value) for value in eligible_node_ids)
    compact_index = {node: index for index, node in enumerate(eligible)}
    if len(eligible) != len(compact_index):
        raise RepresentationRankAssayError("eligible coordinate identity duplicated")
    signal = np.zeros(len(eligible))
    for row, nodes in enumerate(fit_nodes):
        for node in nodes[:5]:
            signal[compact_index[int(node)]] += residual[row]
        for node in nodes[5:]:
            signal[compact_index[int(node)]] -= residual[row]
    scale = max(float(np.linalg.norm(signal)), 1.0)
    base_a = np.outer(signal / scale, np.linspace(0.01, 0.02, width))
    base_e = np.outer(signal / scale, np.linspace(0.01, 0.02, 2 * width))
    starts = [(base_a, base_e)]
    for seed in (41_001 + width, 41_002 + width):
        rng = np.random.Generator(np.random.PCG64DXSM(seed))
        starts.append(
            (
                base_a + rng.normal(0, 0.01, size=base_a.shape),
                base_e + rng.normal(0, 0.01, size=base_e.shape),
            )
        )
    return starts


def fit_latent_candidate(
    *,
    blue_nodes: np.ndarray,
    red_nodes: np.ndarray,
    target_domain: TargetDomain,
    split_identity: str,
    game_ids: Sequence[object],
    verified_nuisance_oof: Mapping[str, float],
    eligibility_binding: EligibilityBinding,
    width: int,
    lambda_ally: float,
    lambda_enemy: float,
    mode: str = "joint",
) -> LatentFit:
    _validate_eligibility_binding(eligibility_binding)
    node_domain = eligibility_binding.node_domain
    node_roles = node_domain.node_roles
    eligible_nodes = np.asarray(
        eligibility_binding.eligible_nodes, dtype=bool
    )
    blue, red = _validate_drafts(
        blue_nodes=blue_nodes,
        red_nodes=red_nodes,
        node_domain=node_domain,
        label="fit",
    )
    if split_identity == FINAL_SPLIT:
        raise RepresentationRankAssayError("sealed final holdout reached fitter")
    if split_identity not in ALLOWED_NONHOLDOUT_SPLITS:
        raise RepresentationRankAssayError("unknown fitter split")
    ids = tuple(str(value) for value in game_ids)
    validate_target_domain(target_domain)
    verified_target_by_game_id = dict(target_domain.ordered_targets)
    feature_records = {
        row[0]: row for row in eligibility_binding.feature_domain.records
    }
    if any(
        feature_records.get(game_id, (None, None))[1] == FINAL_SPLIT
        for game_id in verified_target_by_game_id
    ):
        raise RepresentationRankAssayError(
            "sealed final holdout target reached fitter domain"
        )
    required_fit_ids = tuple(
        row[0]
        for row in eligibility_binding.feature_domain.records
        if row[0] in verified_target_by_game_id
        and row[1] in ALLOWED_NONHOLDOUT_SPLITS
        and row[3] < eligibility_binding.prediction_block
        and np.all(eligible_nodes[np.asarray(row[5], dtype=int)])
    )
    if (
        required_fit_ids != eligibility_binding.ordered_fit_game_ids
        or ids != required_fit_ids
    ):
        raise RepresentationRankAssayError(
            "fitter rows differ from bound required fit population"
        )
    if set(ids) - set(verified_target_by_game_id):
        raise RepresentationRankAssayError("verified target missing by game_id")
    target = np.asarray(
        [verified_target_by_game_id[game_id] for game_id in ids], dtype=float
    )
    expected_fit_nodes = np.asarray(
        [feature_records[game_id][5] for game_id in ids], dtype=np.int64
    )
    if not np.array_equal(
        np.column_stack((blue, red)), expected_fit_nodes
    ):
        raise RepresentationRankAssayError(
            "fitter draft rows differ from bound FeatureDomain"
        )
    p0 = np.asarray(
        [verified_nuisance_oof[game_id] for game_id in ids], dtype=float
    ) if set(ids).issubset(verified_nuisance_oof) else np.asarray([], dtype=float)
    require_m0_offsets(ids, p0, verified_nuisance_oof)
    if (
        width not in WIDTHS
        or mode not in {"joint", "ally_only", "enemy_only"}
        or target.shape != (len(blue),)
        or p0.shape != target.shape
        or not set(target).issubset({0.0, 1.0})
        or np.any(p0 <= 0)
        or np.any(p0 >= 1)
        or lambda_ally <= 0
        or lambda_enemy <= 0
    ):
        raise RepresentationRankAssayError("latent fit inputs invalid")
    node_count = len(node_roles)
    eligible = np.asarray(eligible_nodes)
    if eligible.shape != (node_count,) or eligible.dtype != np.bool_:
        raise RepresentationRankAssayError("eligible-node mask invalid")
    active = np.unique(np.concatenate((blue.ravel(), red.ravel())))
    if (
        np.any(active < 0)
        or np.any(active >= node_count)
        or np.any(~eligible_nodes[active])
    ):
        raise RepresentationRankAssayError("active unseen/ineligible node reached fitter")
    fit_nodes = np.column_stack((blue, red))
    residual = target - p0
    eligible_indices = np.flatnonzero(eligible)
    eligible_count = len(eligible_indices)
    starts = deterministic_starts(eligible_indices, width, fit_nodes, residual)
    converged: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, float]] = []
    parameter_width = (
        eligible_count * width
        if mode == "ally_only"
        else eligible_count * 2 * width
        if mode == "enemy_only"
        else eligible_count * 3 * width
    )

    def expand(vector: np.ndarray) -> np.ndarray:
        full_a = np.zeros((node_count, width), dtype=float)
        full_e = np.zeros((node_count, 2 * width), dtype=float)
        if mode != "enemy_only":
            compact_split = eligible_count * width
            full_a[eligible_indices] = vector[:compact_split].reshape(
                eligible_count, width
            )
        else:
            compact_split = 0
        if mode != "ally_only":
            full_e[eligible_indices] = vector[compact_split:].reshape(
                eligible_count, 2 * width
            )
        return np.concatenate((full_a.ravel(), full_e.ravel()))

    def unpack(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        expanded = expand(vector)
        split = node_count * width
        raw_a = expanded[:split].reshape(node_count, width)
        raw_e = expanded[split:].reshape(node_count, 2 * width)
        a = center_by_role(raw_a, node_roles, eligible_nodes)
        e = center_by_role(raw_e, node_roles, eligible_nodes)
        if mode == "ally_only":
            e[:] = 0
        elif mode == "enemy_only":
            a[:] = 0
        return a, e

    def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
        value, full_gradient = latent_objective_and_gradient(
            expand(vector),
            blue=blue,
            red=red,
            target=target,
            p0=p0,
            node_domain=node_domain,
            eligible_nodes=eligible_nodes,
            width=width,
            lambda_ally=lambda_ally,
            lambda_enemy=lambda_enemy,
            mode=mode,
        )
        split = node_count * width
        gradient_a = full_gradient[:split].reshape(node_count, width)
        gradient_e = full_gradient[split:].reshape(node_count, 2 * width)
        if mode == "ally_only":
            compact_gradient = gradient_a[eligible_indices].ravel()
        elif mode == "enemy_only":
            compact_gradient = gradient_e[eligible_indices].ravel()
        else:
            compact_gradient = np.concatenate(
                (
                    gradient_a[eligible_indices].ravel(),
                    gradient_e[eligible_indices].ravel(),
                )
            )
        return value, compact_gradient

    for start_a, start_e in starts:
        if mode == "ally_only":
            start = start_a.ravel()
        elif mode == "enemy_only":
            start = start_e.ravel()
        else:
            start = np.concatenate(
                (
                    start_a.ravel(),
                    start_e.ravel(),
                )
            )
        if start.shape != (parameter_width,) or not np.any(start):
            raise RepresentationRankAssayError("zero/invalid optimization start")
        result = minimize(
            objective,
            start,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 3000, "ftol": 1e-12, "gtol": 1e-7},
        )
        value, gradient = objective(result.x)
        maximum_gradient = float(np.max(np.abs(gradient)))
        if (
            result.success
            and math.isfinite(value)
            and np.isfinite(result.x).all()
            and maximum_gradient <= GRADIENT_TOLERANCE
        ):
            a, e = unpack(result.x)
            interaction = interaction_logits(
                blue,
                red,
                a,
                e,
                width=width,
                node_domain=node_domain,
            )[2]
            converged.append((value, a, e, interaction, maximum_gradient))
    if len(converged) < 2:
        raise RepresentationRankAssayError("fewer than two stable converged starts")
    converged.sort(key=lambda item: item[0])
    rms = float(np.sqrt(np.mean((converged[0][3] - converged[1][3]) ** 2)))
    if rms > STABILITY_RMS_TOLERANCE:
        raise RepresentationRankAssayError("best two latent solutions are unstable")
    best = converged[0]
    return LatentFit(
        width=width,
        ally_centered=best[1],
        enemy_centered=best[2],
        objective=float(best[0]),
        maximum_absolute_gradient=float(best[4]),
        converged_starts=len(converged),
        best_two_interaction_logit_rms=rms,
        eligibility_binding_sha256=eligibility_binding.artifact_sha256,
    )


def select_separate_penalty(
    rows: Sequence[Mapping[str, Any]], *, family: str
) -> float:
    if family not in {"ally", "enemy"}:
        raise RepresentationRankAssayError("unknown penalty family")
    required = {
        "family",
        "lambda",
        "width",
        "calendar_month",
        "split",
        "maps",
        "clusters",
        "membership_sha256",
        "log_loss_total",
        "brier_total",
        "strictly_earlier_fit",
        "cluster_atomic",
    }
    records = [dict(row) for row in rows]
    if not records or any(set(row) != required for row in records):
        raise RepresentationRankAssayError("penalty rows invalid")
    expected: dict[str, tuple[int, int, str]] | None = None
    scores: list[tuple[float, float, float]] = []
    for penalty in PENALTY_GRID:
        selected = [
            row
            for row in records
            if row["family"] == family and float(row["lambda"]) == penalty
        ]
        if (
            len(selected) != len(INNER_MONTHS)
            or {row["calendar_month"] for row in selected} != set(INNER_MONTHS)
            or any(row["width"] != 8 for row in selected)
            or any(row["split"] != "train" for row in selected)
            or any(row["strictly_earlier_fit"] is not True for row in selected)
            or any(row["cluster_atomic"] is not True for row in selected)
        ):
            raise RepresentationRankAssayError("penalty fold contract changed")
        current = {
            row["calendar_month"]: (
                int(row["maps"]),
                int(row["clusters"]),
                str(row["membership_sha256"]),
            )
            for row in selected
        }
        if len(current) != len(selected):
            raise RepresentationRankAssayError("duplicate penalty month")
        if expected is None:
            expected = current
        elif current != expected:
            raise RepresentationRankAssayError("penalty memberships differ")
        maps = sum(value[0] for value in current.values())
        if maps < 1500:
            raise RepresentationRankAssayError("insufficient inner OOF support")
        scores.append(
            (
                penalty,
                sum(float(row["log_loss_total"]) for row in selected) / maps,
                sum(float(row["brier_total"]) for row in selected) / maps,
            )
        )
    minimum = min(score[1] for score in scores)
    tied = [score for score in scores if score[1] <= minimum + 1e-12]
    minimum_brier = min(score[2] for score in tied)
    tied = [score for score in tied if score[2] <= minimum_brier + 1e-12]
    return max(score[0] for score in tied)


def bootstrap_multiplicity(
    cluster_ids: Sequence[object], *, replicates: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    unique = np.asarray(sorted({str(value) for value in cluster_ids}), dtype=object)
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    draws = rng.integers(0, len(unique), size=(replicates, len(unique)))
    return unique, np.stack(
        [np.bincount(draw, minlength=len(unique)) for draw in draws]
    )


def nearest_rank_endpoint(draws: Sequence[object], *, one_indexed: int) -> float:
    values = np.sort(np.asarray(draws, dtype=float))
    if one_indexed < 1 or one_indexed > len(values):
        raise RepresentationRankAssayError("endpoint invalid")
    return float(values[one_indexed - 1])


def _probability_vector(value: Sequence[object], rows: int, label: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if (
        result.shape != (rows,)
        or not np.isfinite(result).all()
        or np.any(result <= 0)
        or np.any(result >= 1)
    ):
        raise RepresentationRankAssayError(f"{label} probabilities invalid")
    return result


def _loss_rows(y: np.ndarray, probability: np.ndarray, metric: str) -> np.ndarray:
    if metric == "log_loss":
        return -(y * np.log(probability) + (1 - y) * np.log(1 - probability))
    if metric == "brier":
        return (probability - y) ** 2
    raise RepresentationRankAssayError("unknown gate metric")


def _cluster_totals(
    values: np.ndarray, clusters: np.ndarray, unique: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    totals = np.asarray(
        [values[clusters == cluster].sum() for cluster in unique], dtype=float
    )
    counts = np.asarray(
        [np.count_nonzero(clusters == cluster) for cluster in unique], dtype=float
    )
    return totals, counts


def _bootstrap_metric_delta(
    y: np.ndarray,
    candidate: np.ndarray,
    comparator: np.ndarray,
    clusters: np.ndarray,
    unique: np.ndarray,
    multiplicity: np.ndarray,
    *,
    metric: str,
) -> np.ndarray:
    weights = np.asarray(multiplicity, dtype=float)
    if metric in {"log_loss", "brier"}:
        delta = _loss_rows(y, candidate, metric) - _loss_rows(
            y, comparator, metric
        )
        totals, counts = _cluster_totals(delta, clusters, unique)
        numerator = np.sum(weights * totals[None, :], axis=1)
        denominator = np.sum(weights * counts[None, :], axis=1)
        return numerator / denominator
    if metric == "calibration":
        candidate_residual, counts = _cluster_totals(
            y - candidate, clusters, unique
        )
        comparator_residual, _ = _cluster_totals(
            y - comparator, clusters, unique
        )
        denominator = np.sum(weights * counts[None, :], axis=1)
        candidate_total = np.sum(
            weights * candidate_residual[None, :], axis=1
        )
        comparator_total = np.sum(
            weights * comparator_residual[None, :], axis=1
        )
        return (
            np.abs(candidate_total / denominator)
            - np.abs(comparator_total / denominator)
        )
    raise RepresentationRankAssayError("unknown bootstrap gate metric")


def _validate_frozen_blocks(
    blocks: np.ndarray, clusters: np.ndarray, *, split: str
) -> None:
    if split not in ELIGIBLE_GATE_BLOCKS:
        raise RepresentationRankAssayError("gate split is not frozen")
    _assert_clusters_single_block(clusters, blocks)
    expected = {
        month: (maps, cluster_count)
        for month, maps, cluster_count in ELIGIBLE_GATE_BLOCKS[split]
    }
    if set(blocks) != set(expected):
        raise RepresentationRankAssayError("frozen block membership changed")
    for month, (maps, cluster_count) in expected.items():
        selected = blocks == month
        if int(selected.sum()) != maps or len(set(clusters[selected])) != cluster_count:
            raise RepresentationRankAssayError("frozen block arithmetic changed")


def _bound_gate_inputs(
    *,
    prepared_fold: PreparedFold,
    split: str,
    game_ids: Sequence[object],
    predictions: Mapping[int | str, Sequence[object]],
) -> tuple[np.ndarray, np.ndarray]:
    _validate_prepared_fold(prepared_fold, split=split)
    ids = tuple(str(value) for value in game_ids)
    if ids != prepared_fold.ordered_eligible_game_ids:
        raise RepresentationRankAssayError(
            "gate game identity/order differs from PreparedFold"
        )
    m0 = _probability_vector(
        predictions["M0"], len(ids), "M0"
    )
    if not np.array_equal(m0, prepared_fold.m0_probability):
        raise RepresentationRankAssayError(
            "gate M0 differs bitwise from PreparedFold"
        )
    return (
        np.asarray(prepared_fold.ordered_eligible_cluster_ids, dtype=object),
        np.asarray(prepared_fold.ordered_eligible_blocks, dtype=object),
    )


def _bound_targets(
    prepared_fold: PreparedFold,
    target_domain: TargetDomain,
) -> np.ndarray:
    validate_target_domain(target_domain)
    verified_target_by_game_id = dict(target_domain.ordered_targets)
    ids = prepared_fold.ordered_eligible_game_ids
    if set(ids) - set(verified_target_by_game_id):
        raise RepresentationRankAssayError("verified gate target missing by game_id")
    target = np.asarray(
        [verified_target_by_game_id[game_id] for game_id in ids], dtype=float
    )
    if target.shape != (len(ids),) or not set(target).issubset({0.0, 1.0}):
        raise RepresentationRankAssayError("verified gate target invalid")
    return target


def evaluate_candidate_gates(
    *,
    y: Sequence[object],
    candidate_probability: Sequence[object],
    m0_probability: Sequence[object],
    m8_probability: Sequence[object],
    cluster_ids: Sequence[object],
    chronological_blocks: Sequence[object],
    multiplicity: np.ndarray,
    unique_clusters: np.ndarray,
    endpoint_1_indexed: int,
) -> dict[str, Any]:
    target = np.asarray(y, dtype=float)
    rows = len(target)
    candidate = _probability_vector(candidate_probability, rows, "candidate")
    comparators = {
        "M0": _probability_vector(m0_probability, rows, "M0"),
        "M8": _probability_vector(m8_probability, rows, "M8"),
    }
    clusters = np.asarray([str(value) for value in cluster_ids], dtype=object)
    blocks = np.asarray([str(value) for value in chronological_blocks], dtype=object)
    if (
        target.shape != (rows,)
        or not set(target).issubset({0.0, 1.0})
        or clusters.shape != (rows,)
        or blocks.shape != (rows,)
        or multiplicity.shape[1] != len(unique_clusters)
        or list(unique_clusters) != sorted(set(clusters))
    ):
        raise RepresentationRankAssayError("gate row/multiplicity inputs invalid")
    results: dict[str, Any] = {}
    passed = True
    for name, comparator in comparators.items():
        gates: dict[str, Any] = {}
        for metric in ("log_loss", "brier", "calibration"):
            draws = _bootstrap_metric_delta(
                target,
                candidate,
                comparator,
                clusters,
                unique_clusters,
                multiplicity,
                metric=metric,
            )
            upper = nearest_rank_endpoint(
                draws, one_indexed=endpoint_1_indexed
            )
            decision = metric_gate_decision(
                comparator=name,
                metric=metric,
                upper=upper,
            )
            accepted = bool(decision["passed"])
            gates[metric] = {
                "upper": upper,
                "limit": decision["limit"],
                "strict": decision["strict"],
                "passed": accepted,
            }
            passed = passed and accepted
        block_results: dict[str, Any] = {}
        for block in sorted(set(blocks)):
            selected = blocks == block
            delta = float(
                np.mean(_loss_rows(target[selected], candidate[selected], "log_loss"))
                - np.mean(
                    _loss_rows(target[selected], comparator[selected], "log_loss")
                )
            )
            accepted = block_gate_decision(delta)
            block_results[block] = {"delta": delta, "passed": accepted}
            passed = passed and accepted
        gates["blocks"] = block_results
        results[name] = gates
    return {"passed": bool(passed), "comparators": results}


def select_development_width(
    *,
    prepared_fold: PreparedFold,
    game_ids: Sequence[object],
    target_domain: TargetDomain,
    predictions: Mapping[int | str, Sequence[object]],
    m8_optimization_stable: bool,
) -> tuple[int, dict[str, Any]]:
    if set(predictions) != {*WIDTHS, "M0", "M8"}:
        raise RepresentationRankAssayError("development candidate set changed")
    if not np.array_equal(
        np.asarray(predictions[8], dtype=float),
        np.asarray(predictions["M8"], dtype=float),
    ):
        raise RepresentationRankAssayError("width 8 and M8 predictions diverged")
    target = _bound_targets(prepared_fold, target_domain)
    clusters, blocks = _bound_gate_inputs(
        prepared_fold=prepared_fold,
        split="development",
        game_ids=game_ids,
        predictions=predictions,
    )
    unique, multiplicity = bootstrap_multiplicity(
        clusters, replicates=PRIMARY_REPLICATES, seed=DEVELOPMENT_SEED
    )
    m8_prerequisite = evaluate_candidate_gates(
        y=target,
        candidate_probability=predictions["M8"],
        m0_probability=predictions["M0"],
        m8_probability=predictions["M8"],
        cluster_ids=clusters,
        chronological_blocks=blocks,
        multiplicity=multiplicity,
        unique_clusters=unique,
        endpoint_1_indexed=DEVELOPMENT_ENDPOINT,
    )
    # M8-vs-M8 gates are identities; its prerequisite is the M0 side plus
    # independently established optimization/stability.
    m8_passed = bool(
        m8_optimization_stable
        and all(
            gate["passed"]
            for key, gate in m8_prerequisite["comparators"]["M0"].items()
            if key != "blocks"
        )
        and all(
            gate["passed"]
            for gate in m8_prerequisite["comparators"]["M0"]["blocks"].values()
        )
    )
    if not m8_passed:
        raise RepresentationRankAssayError(
            "M8 prerequisite failed",
            diagnostics={
                "M8_prerequisite": m8_prerequisite,
                "widths": {},
                "locked_width": None,
            },
        )
    diagnostics: dict[str, Any] = {"M8_prerequisite": m8_prerequisite, "widths": {}}
    for width in WIDTHS:
        result = evaluate_candidate_gates(
            y=target,
            candidate_probability=predictions[width],
            m0_probability=predictions["M0"],
            m8_probability=predictions["M8"],
            cluster_ids=clusters,
            chronological_blocks=blocks,
            multiplicity=multiplicity,
            unique_clusters=unique,
            endpoint_1_indexed=DEVELOPMENT_ENDPOINT,
        )
        diagnostics["widths"][str(width)] = result
        if result["passed"]:
            diagnostics["locked_width"] = width
            return width, diagnostics
    diagnostics["locked_width"] = None
    raise RepresentationRankAssayError(
        "no latent width passed development gates",
        diagnostics=diagnostics,
    )


def validate_locked_width(
    *,
    prepared_fold: PreparedFold,
    game_ids: Sequence[object],
    locked_width: int,
    target_domain: TargetDomain,
    predictions: Mapping[int | str, Sequence[object]],
    m8_optimization_stable: bool,
) -> dict[str, Any]:
    if locked_width not in WIDTHS or set(predictions) != {
        locked_width,
        "M0",
        "M8",
    }:
        raise RepresentationRankAssayError("validation attempted reselection")
    if locked_width == 8 and not np.array_equal(
        np.asarray(predictions[8], dtype=float),
        np.asarray(predictions["M8"], dtype=float),
    ):
        raise RepresentationRankAssayError("locked width 8 and M8 predictions diverged")
    clusters, blocks = _bound_gate_inputs(
        prepared_fold=prepared_fold,
        split="validation",
        game_ids=game_ids,
        predictions=predictions,
    )
    target = _bound_targets(prepared_fold, target_domain)
    unique, multiplicity = bootstrap_multiplicity(
        clusters, replicates=PRIMARY_REPLICATES, seed=VALIDATION_SEED
    )
    m8_result = evaluate_candidate_gates(
        y=target,
        candidate_probability=predictions["M8"],
        m0_probability=predictions["M0"],
        m8_probability=predictions["M8"],
        cluster_ids=clusters,
        chronological_blocks=blocks,
        multiplicity=multiplicity,
        unique_clusters=unique,
        endpoint_1_indexed=VALIDATION_ENDPOINT,
    )
    m8_passed = bool(
        m8_optimization_stable
        and all(
            gate["passed"]
            for key, gate in m8_result["comparators"]["M0"].items()
            if key != "blocks"
        )
        and all(
            gate["passed"]
            for gate in m8_result["comparators"]["M0"]["blocks"].values()
        )
    )
    locked = evaluate_candidate_gates(
        y=target,
        candidate_probability=predictions[locked_width],
        m0_probability=predictions["M0"],
        m8_probability=predictions["M8"],
        cluster_ids=clusters,
        chronological_blocks=blocks,
        multiplicity=multiplicity,
        unique_clusters=unique,
        endpoint_1_indexed=VALIDATION_ENDPOINT,
    )
    diagnostics = {
        "passed": bool(m8_passed and locked["passed"]),
        "locked_width": locked_width,
        "M8": m8_result,
        "locked": locked,
    }
    if not diagnostics["passed"]:
        raise RepresentationRankAssayError(
            "locked width failed validation",
            diagnostics=diagnostics,
        )
    return diagnostics


def validate_report(payload: Mapping[str, Any]) -> None:
    unsigned = dict(payload)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != canonical_sha256(unsigned):
        raise RepresentationRankAssayError("report artifact hash mismatch")
    if (
        payload.get("schema_id") != REPORT_SCHEMA_ID
        or payload.get("status") != "private_pending_fixture_review"
        or payload.get("aggregate_only") is not True
        or payload.get("real_candidate_outcomes_loaded") is not False
        or payload.get("authoritative_feature_domain_loaded") is not False
        or payload.get("authoritative_target_domain_loaded") is not False
        or payload.get("width_selected") is not False
        or payload.get("results")
        != {
            "lambda_A": None,
            "lambda_E": None,
            "coverage": None,
            "development": None,
            "validation": None,
        }
    ):
        raise RepresentationRankAssayError("pending report contract changed")
    for field in (
        "predictive_authority",
        "authorizes_prediction",
        "authorizes_publication",
        "authorizes_production",
        "authorizes_reliability",
        "authorizes_promotion",
        "authorizes_sota_claim",
        "authorizes_whole_composition_result",
    ):
        if payload.get(field) is not False:
            raise RepresentationRankAssayError("report authority exceeded")
    if payload.get("claim_ceiling") != (
        "private retrospective latent pair-capacity assay only; no intrinsic "
        "table rank, whole-composition result, prediction, publication, "
        "production, Reliability, promotion, or SOTA authority"
    ):
        raise RepresentationRankAssayError("report claim ceiling changed")
