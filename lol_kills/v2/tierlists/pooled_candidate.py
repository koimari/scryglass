"""Build the atom-informed pooled tier-list candidate.

The candidate uses one Bernoulli likelihood per completed map. Champion
strength, atom matchup features, scope effects, OE patch effects, and
same-role pair residuals enter that likelihood together. This module builds
the data adapter and the descriptive tier rows. It does not grant production
authority.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit

from lol_kills.v2.tierlists.atom_matchup_features import (
    AtomMatchupFeatureError,
    AtomMatchupFeatureResolver,
    ExactAtomSnapshotMapping,
    FEATURE_ORDER,
)
from lol_kills.v2.tierlists.joint_pooled_model import (
    AtomFeatureRegistry,
    AtomFeatureVector,
    JointMapObservation,
    JointPooledFit,
    PriorScales,
    design_vector_for_observation,
    fit_joint_pooled_model,
)
from lol_kills.v2.tierlists.patch_mapping import (
    DEFAULT_MAPPING_PATH,
    MappingArtifact,
    PatchMappingError,
    load_mapping,
    normalize_oe_token,
    resolve_oe_patch,
)
from lol_kills.v2.patch_identity import CURRENT_PUBLIC_PATCH
from lol_kills.v2.tierlists.accepted_census import identity_sha256

from .champion_elo import (
    ATOM_BRIDGE_LOCATORS,
    DEFAULT_MIN_APPEARANCES,
    HISTORY_START,
    INITIAL_RATING,
    LEGAL_OPPONENT_COUNT,
    LIVE_WINDOW_START,
    MATCHUP_MAX_POSTERIOR_SD,
    MATCHUP_MIN_EFFECTIVE_MAPS,
    MATCHUP_MIN_SERIES,
    RECENCY_HALF_LIFE_DAYS,
    ROLES,
    SOURCE_LOCATOR,
    SOURCE_MODES,
    TEAM_K,
    TIER_BUCKETS,
    TIER_MEMBERSHIP_PROBABILITY,
    _assign_tier_buckets,
    _build_maps,
    _canonical_json,
    _load_crosswalk,
    _load_source,
    _logit,
    _normalize_name,
    _sha256_bytes,
    _sha256_path,
    _utc_stamp,
)


POOLED_CANDIDATE_SCHEMA = "scryglass:champion-role-elo-candidate:v2"
ATOM_DEVIATION_DIM = 2
POSTERIOR_SEED = 20260808
MAX_LOO_SERIES = 8
COUNTER_EFFECT_THRESHOLD_LOGIT = 0.05
COUNTER_POSTERIOR_THRESHOLD = 0.80
BLIND_TAIL_SHARE = 0.20
RESPONSE_INTERVAL_Z = 1.2815515655446004
STRENGTH_MAX_CONTRAST_SD = 0.90
POSTERIOR_DRAWS = 2000
REGIONAL_CONTEXT_ORDER = (
    "LCK",
    "LPL",
    "LEC",
    "LCS",
    "CBLOL",
    "LCP",
    "PCS",
    "VCS",
    "LJL",
    "TCL",
    "INTERNATIONAL",
)
REGIONAL_CONTEXT_LEAGUES: dict[str, frozenset[str]] = {
    "LCK": frozenset({"LCK"}),
    "LPL": frozenset({"LPL"}),
    "LEC": frozenset({"LEC"}),
    "LCS": frozenset({"LCS"}),
    "CBLOL": frozenset({"CBLOL"}),
    "LCP": frozenset({"LCP"}),
    "PCS": frozenset({"PCS"}),
    "VCS": frozenset({"VCS"}),
    "LJL": frozenset({"LJL"}),
    "TCL": frozenset({"TCL"}),
    "INTERNATIONAL": frozenset({"MSI", "EWC", "WORLDS", "FST"}),
}


class PooledCandidateError(ValueError):
    """Raised when a pooled candidate cannot be built safely."""


def _empty_vector(registry: AtomFeatureRegistry) -> AtomFeatureVector:
    return AtomFeatureVector.from_values(
        (None,) * len(registry.features),
        available=(False,) * len(registry.features),
    )


def _rfc3339(value: object) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


def _weighted_lower_tail(values: np.ndarray, weights: np.ndarray, share: float) -> np.ndarray:
    """Return a weighted lower-tail mean for every posterior draw."""

    if values.ndim != 2 or weights.ndim != 1 or values.shape[1] != weights.size:
        raise PooledCandidateError("weighted tail inputs have incompatible shapes")
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = weights[order]
    prior_weight = np.cumsum(sorted_weights, axis=1) - sorted_weights
    portions = np.minimum(sorted_weights, np.clip(share - prior_weight, 0.0, None))
    return np.sum(portions * sorted_values, axis=1) / share


def _blind_point_estimate(probability_matrix: np.ndarray, weights: np.ndarray) -> float:
    """Return the expected weakest common matchup without an uncertainty penalty."""

    if probability_matrix.ndim != 2 or probability_matrix.shape[0] != weights.size:
        raise PooledCandidateError("blind point-estimate inputs have incompatible shapes")
    posterior_means = probability_matrix.mean(axis=1)
    return float(_weighted_lower_tail(posterior_means[None, :], weights, BLIND_TAIL_SHARE)[0])


def _counter_count_point_estimate(theta_matrix: np.ndarray) -> int:
    """Count common opponents with a positive posterior-mean model contrast."""

    if theta_matrix.ndim != 2:
        raise PooledCandidateError("counter point-estimate input must be a matrix")
    return int(np.count_nonzero(theta_matrix.mean(axis=1) > COUNTER_EFFECT_THRESHOLD_LOGIT))


def _response_basis(*, effective_maps: float, atom_supported: bool) -> str:
    if effective_maps > 0.0:
        return "observed_pair_plus_model"
    if atom_supported:
        return "atom_and_strength_inferred"
    return "strength_only_inferred"


def _scope_atom_patch(scope_games: Sequence[Mapping[str, Any]]) -> str | None:
    patches = {
        str(game.get("atom_snapshot_patch") or "").strip()
        for game in scope_games
        if str(game.get("atom_snapshot_patch") or "").strip()
    }
    return next(iter(patches)) if len(patches) == 1 else None


def _safe_normalized_patch(value: object) -> str:
    try:
        return normalize_oe_token(value)
    except PatchMappingError:
        text = str(value).strip()
        return text if text and text.casefold() != "nan" else "unknown"


def _mapping_for_root(root: Path) -> tuple[MappingArtifact | None, dict[str, Any]]:
    path = root / DEFAULT_MAPPING_PATH
    if not path.exists():
        return None, {
            "status": "unavailable",
            "reason": "mapping_sidecar_missing",
            "locator": DEFAULT_MAPPING_PATH.as_posix(),
        }
    try:
        artifact = load_mapping(path, verify_source_hashes=True, repo_root=root)
    except PatchMappingError as exc:
        raise PooledCandidateError(f"audited OE-to-atom mapping is invalid: {exc}") from exc
    return artifact, {
        "status": "audited",
        "locator": str(path.relative_to(root)),
        "artifact_sha256": artifact.payload["artifact_sha256"],
        "schema_version": artifact.payload["schema_version"],
        "source_window": artifact.payload.get("source_window"),
        "mapping_rows": len(artifact.rows),
        "live_source": artifact.live_source,
    }


def _atom_snapshot_mapping(
    artifact: MappingArtifact,
    patch: str,
    bridge_sha256: str,
) -> ExactAtomSnapshotMapping | None:
    for snapshot in artifact.payload.get("atom_snapshots", []):
        if not isinstance(snapshot, Mapping) or snapshot.get("patch") != patch:
            continue
        return ExactAtomSnapshotMapping(
            requested_patch=patch,
            snapshot_patch=patch,
            snapshot_as_of=str(snapshot["generated_at"]),
            bridge_artifact_sha256=bridge_sha256,
            time_safe=True,
        )
    return None


def _map_atom_features(
    game: Mapping[str, Any],
    *,
    stable_roles: Mapping[str, Mapping[str, str]],
    resolvers: Mapping[str, AtomMatchupFeatureResolver],
    registry: AtomFeatureRegistry,
    mapping: MappingArtifact | None,
    pair_cache: dict[tuple[str, str, str | None], AtomFeatureVector],
) -> tuple[dict[str, AtomFeatureVector], dict[str, Any]]:
    unavailable = _empty_vector(registry)
    vectors = {role: unavailable for role in ROLES}
    raw_patch = str(game.get("patch") or "").strip()
    oe_patch = _safe_normalized_patch(raw_patch)
    resolution = None
    if mapping is not None:
        resolution = resolve_oe_patch(
            raw_patch,
            _rfc3339(game["date"]),
            mapping=mapping,
            verify_source_hashes=False,
        )
    if resolution is not None and resolution.oe_token:
        oe_patch = resolution.oe_token
    atom_patch = resolution.atom_snapshot_patch if resolution is not None and resolution.exact_atom_snapshot else None
    resolver = resolvers.get(str(atom_patch)) if atom_patch is not None else None
    snapshot_mapping = (
        _atom_snapshot_mapping(mapping, atom_patch, resolver.bridge.artifact_sha256)
        if mapping is not None and atom_patch is not None and resolver is not None
        else None
    )
    exact_pair_count = 0
    available_feature_count = 0
    for role in ROLES:
        blue_id = stable_roles[role]["blue"]
        red_id = stable_roles[role]["red"]
        cache_key = (blue_id, red_id, atom_patch)
        if cache_key in pair_cache:
            vector = pair_cache[cache_key]
        elif atom_patch is None or snapshot_mapping is None or resolver is None:
            vector = unavailable
        else:
            try:
                pair = resolver.resolve_pair(
                    blue_id,
                    red_id,
                    requested_patch=atom_patch,
                    snapshot_mapping=snapshot_mapping,
                )
            except (AtomMatchupFeatureError, KeyError):
                vector = unavailable
            else:
                vector = AtomFeatureVector.from_values(
                    tuple(pair["features"][name] for name in FEATURE_ORDER),
                    available=tuple(pair["availability"][name] for name in FEATURE_ORDER),
                )
            pair_cache[cache_key] = vector
        vectors[role] = vector
        if any(vector.available):
            exact_pair_count += 1
            available_feature_count += sum(vector.available)
    return vectors, {
        "oe_patch": oe_patch,
        "official_patch": resolution.official_patch if resolution is not None else None,
        "atom_snapshot_patch": atom_patch,
        "mapping_status": resolution.status if resolution is not None else "unavailable",
        "mapping_reason": resolution.reason if resolution is not None else "mapping_sidecar_missing",
        "exact_atom_role_pairs": exact_pair_count,
        "available_atom_features": available_feature_count,
    }


def _previous_rows(previous: Mapping[str, Any] | None) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    rows: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for cell in (previous or {}).get("cells", []):
        if not isinstance(cell, Mapping):
            continue
        scope_id = str(cell.get("scope_id") or "")
        role = str(cell.get("role") or "")
        for row in cell.get("rows", []):
            if not isinstance(row, Mapping):
                continue
            key = _normalize_name(row.get("champion") or row.get("champion_name") or "")
            if key:
                rows[(scope_id, role, key)] = row
            champion_id = row.get("champion_id")
            if isinstance(champion_id, str) and champion_id:
                rows[(scope_id, role, champion_id)] = row
    return rows


def _team_offsets(maps: Sequence[Mapping[str, Any]]) -> list[float]:
    ratings: dict[str, float] = {}
    offsets: list[float] = []
    for game in maps:
        blue = _normalize_name(game["blue_team"])
        red = _normalize_name(game["red_team"])
        blue_rating = ratings.get(blue, INITIAL_RATING)
        red_rating = ratings.get(red, INITIAL_RATING)
        p_team = 1.0 / (1.0 + 10.0 ** ((red_rating - blue_rating) / 400.0))
        offsets.append(_logit(p_team))
        residual = float(game["y_blue_win"]) - p_team
        ratings[blue] = blue_rating + TEAM_K * residual
        ratings[red] = red_rating - TEAM_K * residual
    return offsets


def _pair_stats(
    prepared_maps: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    stats: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for game in prepared_maps:
        for role in ROLES:
            blue = game["stable_roles"][role]["blue"]
            red = game["stable_roles"][role]["red"]
            left, right = sorted((blue, red))
            key = (game["scope_id"], role, left, right)
            record = stats.setdefault(
                key,
                {"effective_maps": 0.0, "series": set(), "outcomes": set()},
            )
            record["effective_maps"] += float(game["weight"])
            record["series"].add(str(game["series_id"]))
            record["outcomes"].add(int(game["y_blue_win"]))
    return stats


def _pool_pair_stats(
    scoped: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
) -> dict[tuple[str, str, str], dict[str, Any]]:
    pooled: dict[tuple[str, str, str], dict[str, Any]] = {}
    for (_scope_id, role, left, right), source in scoped.items():
        key = (role, left, right)
        target = pooled.setdefault(
            key,
            {"effective_maps": 0.0, "series": set(), "outcomes": set()},
        )
        target["effective_maps"] += float(source.get("effective_maps", 0.0))
        target["series"].update(source.get("series", ()))
        target["outcomes"].update(source.get("outcomes", ()))
    return pooled


def _regional_contexts(game: Mapping[str, Any]) -> tuple[str, ...]:
    """Return exact public league contexts for one completed map.

    These contexts are a secondary view over the patch-wide fit. They do not
    create separate league-specific tier ladders.
    """

    league = str(game.get("league") or "").strip().upper()
    event = str(game.get("event_kind") or "").strip().upper()
    contexts = [
        context
        for context in REGIONAL_CONTEXT_ORDER
        if league in REGIONAL_CONTEXT_LEAGUES[context]
        or (context == "INTERNATIONAL" and event in REGIONAL_CONTEXT_LEAGUES[context])
    ]
    return tuple(contexts)


def _build_regional_views(
    *,
    rows: Sequence[Mapping[str, Any]],
    scope_id: str,
    role: str,
    regional_counts: Mapping[tuple[str, str, str], Mapping[str, int]],
    regional_game_ids: Mapping[tuple[str, str], set[str]],
) -> list[dict[str, Any]]:
    """Build regional context rows from the fixed patch-wide model.

    A regional view answers which globally fitted champions were observed in a
    selected league. It reports the patch-wide strength value and the local
    appearance count. It does not fit a second league-specific model.
    """

    by_champion = {str(row["champion_id"]): row for row in rows}
    views: list[dict[str, Any]] = []
    for context in REGIONAL_CONTEXT_ORDER:
        counts = regional_counts.get((scope_id, context, role), {})
        if not counts:
            continue
        regional_rows: list[dict[str, Any]] = []
        for champion_id, appearances in counts.items():
            source = by_champion.get(str(champion_id))
            if source is None:
                continue
            regional_rows.append(
                {
                    "champion": source["champion"],
                    "champion_id": source["champion_id"],
                    "global_rank": source["rank"],
                    "strength_score_pp": source["tier_value_pp"],
                    "played_maps": int(appearances),
                    "sample_status": "thin" if int(appearances) < 3 else "observed",
                }
            )
        regional_rows.sort(
            key=lambda row: (
                -float(row["strength_score_pp"]),
                -int(row["played_maps"]),
                _normalize_name(row["champion"]),
            )
        )
        for rank, row in enumerate(regional_rows, start=1):
            row["regional_rank"] = rank
        regional_rows = [
            {
                "champion": row["champion"],
                "champion_id": row["champion_id"],
                "regional_rank": row["regional_rank"],
                "global_rank": row["global_rank"],
                "strength_score_pp": row["strength_score_pp"],
                "played_maps": row["played_maps"],
                "sample_status": row["sample_status"],
            }
            for row in regional_rows
        ]
        views.append(
            {
                "id": context,
                "label": context.replace("_", " "),
                "maps": len(regional_game_ids.get((scope_id, context), set())),
                "basis": "patch_wide_model_with_regional_appearance_filter",
                "rows": regional_rows,
            }
        )
    return views


def _legal_pool(
    counts: Mapping[str, int],
    display_names: Mapping[str, str],
) -> tuple[list[str], list[dict[str, Any]], str]:
    ordered = sorted(
        counts,
        key=lambda champion: (-int(counts[champion]), _normalize_name(display_names.get(champion, champion))),
    )
    pool = ordered[: LEGAL_OPPONENT_COUNT + 1]
    total = sum(int(counts[champion]) for champion in pool)
    weights = [
        (float(counts[champion]) / total if total else 1.0 / max(1, len(pool)))
        for champion in pool
    ]
    legal = [
        {
            "champion": display_names.get(champion, champion),
            "champion_id": champion,
            "weight": round(weight, 8),
        }
        for champion, weight in zip(pool, weights)
    ]
    digest = _sha256_bytes(
        _canonical_json(
            {
                "pool": legal,
                "rule": "take the five highest-pick legal opponents after excluding the focal champion",
            }
        )
    )
    return pool, legal, digest


def _hypothetical_observation(
    *,
    scope_id: str,
    patch_id: str,
    role: str,
    focal: str,
    opponent: str,
    reference_champions: Mapping[str, str],
    atom_vector: AtomFeatureVector,
    empty_vector: AtomFeatureVector,
) -> JointMapObservation:
    picks: dict[str, tuple[str, str]] = {}
    vectors: dict[str, AtomFeatureVector] = {}
    for candidate_role in ROLES:
        if candidate_role == role:
            picks[candidate_role] = (focal, opponent)
            vectors[candidate_role] = atom_vector
        else:
            reference = reference_champions[candidate_role]
            picks[candidate_role] = (reference, reference)
            vectors[candidate_role] = empty_vector
    return JointMapObservation(
        map_id=f"hypothetical:{scope_id}:{patch_id}:{role}:{focal}:{opponent}",
        outcome=0,
        scope_id=scope_id,
        oe_patch_id=patch_id,
        picks=picks,
        atom_pair_features=vectors,
        offset=0.0,
        weight=1.0,
        synthetic=False,
    )


def _posterior_pair_matrix(
    fit: JointPooledFit,
    observations: Sequence[JointMapObservation],
    *,
    posterior_draws: int,
    seed: int,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    rows = [
        design_vector_for_observation(
            fit,
            observation,
            allow_unseen_pairs=True,
            validate=False,
        )
        for observation in observations
    ]
    design = sparse.vstack(rows, format="csr")
    values = np.empty((len(observations), posterior_draws), dtype=float)
    cursor = 0
    for chunk in fit.iter_posterior(
        posterior_draws=posterior_draws,
        seed=seed,
        chunk_size=128,
    ):
        width = chunk.shape[0]
        values[:, cursor : cursor + width] = np.asarray(design @ chunk.T, dtype=float)
        cursor += width
    if cursor != posterior_draws or not np.all(np.isfinite(values)):
        raise PooledCandidateError("posterior pair matrix is incomplete or non-finite")
    return design, values


def _matchup_metrics_available(
    *,
    opponent_count: int,
    supported_opponent_count: int,
    contrast_sd: float,
) -> bool:
    """Expose OE matchup metrics when the full legal pool is supported.

    Exact atom snapshots can refine the joint fit. They are not required for
    the public OE matchup view because the support and uncertainty gates are
    evaluated from completed OE maps and series.
    """

    return (
        opponent_count == LEGAL_OPPONENT_COUNT
        and supported_opponent_count == LEGAL_OPPONENT_COUNT
        and math.isfinite(contrast_sd)
        and contrast_sd <= STRENGTH_MAX_CONTRAST_SD
    )


def _pair_support_details(
    *,
    scope_id: str,
    role: str,
    focal: str,
    opponent: str,
    posterior_sd: float,
    pair_stats: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    pooled_pair_stats: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    left, right = sorted((focal, opponent))
    stat = pair_stats.get((scope_id, role, left, right), {})
    pair_maps = float(stat.get("effective_maps", 0.0))
    pair_series = len(stat.get("series", ()))
    variation = len(stat.get("outcomes", ())) == 2
    support_source = "scope"
    if not (
        pair_maps >= MATCHUP_MIN_EFFECTIVE_MAPS
        and pair_series >= MATCHUP_MIN_SERIES
        and variation
    ):
        stat = pooled_pair_stats.get((role, left, right), {})
        pair_maps = float(stat.get("effective_maps", 0.0))
        pair_series = len(stat.get("series", ()))
        variation = len(stat.get("outcomes", ())) == 2
        support_source = "pooled_scopes"
    supported = (
        pair_maps >= MATCHUP_MIN_EFFECTIVE_MAPS
        and pair_series >= MATCHUP_MIN_SERIES
        and variation
        and math.isfinite(posterior_sd)
        and posterior_sd <= MATCHUP_MAX_POSTERIOR_SD
    )
    return {
        "supported": bool(supported),
        "effective_maps": pair_maps,
        "series_count": pair_series,
        "evidence_source": support_source,
    }


def _response_matrix(
    *,
    fit: JointPooledFit,
    scope_id: str,
    role: str,
    patch_id: str,
    champion_order: Sequence[str],
    display_names: Mapping[str, str],
    reference_champions: Mapping[str, str],
    pair_stats: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    pooled_pair_stats: Mapping[tuple[str, str, str], Mapping[str, Any]],
    resolvers: Mapping[str, AtomMatchupFeatureResolver],
    target_mapping: ExactAtomSnapshotMapping | None,
    exact_atom_patch: str | None,
    empty_vector: AtomFeatureVector,
    atom_vector_cache: dict[tuple[str, str, str | None], AtomFeatureVector],
) -> dict[str, Any]:
    pair_keys: list[tuple[str, str]] = []
    atom_supported: dict[tuple[str, str], bool] = {}
    hypotheses: list[JointMapObservation] = []
    for focal in champion_order:
        for opponent in champion_order:
            if focal == opponent:
                continue
            vector = empty_vector
            resolver = resolvers.get(str(exact_atom_patch)) if exact_atom_patch is not None else None
            if exact_atom_patch is not None and target_mapping is not None and resolver is not None:
                cache_key = (focal, opponent, exact_atom_patch)
                vector = atom_vector_cache.get(cache_key, empty_vector)
                if cache_key not in atom_vector_cache:
                    try:
                        pair = resolver.resolve_pair(
                            focal,
                            opponent,
                            requested_patch=exact_atom_patch,
                            snapshot_mapping=target_mapping,
                        )
                    except (AtomMatchupFeatureError, KeyError):
                        vector = empty_vector
                    else:
                        vector = AtomFeatureVector.from_values(
                            tuple(pair["features"][name] for name in FEATURE_ORDER),
                            available=tuple(pair["availability"][name] for name in FEATURE_ORDER),
                        )
                    atom_vector_cache[cache_key] = vector
            hypotheses.append(
                _hypothetical_observation(
                    scope_id=scope_id,
                    patch_id=patch_id,
                    role=role,
                    focal=focal,
                    opponent=opponent,
                    reference_champions=reference_champions,
                    atom_vector=vector,
                    empty_vector=empty_vector,
                )
            )
            pair_keys.append((focal, opponent))
            atom_supported[(focal, opponent)] = any(vector.available)

    if not hypotheses:
        return {"champions": [], "edge_pp": [], "interval_low_pp": [], "interval_high_pp": [], "evidence": [], "effective_maps": [], "basis": []}

    design = sparse.vstack(
        [
            design_vector_for_observation(
                fit,
                observation,
                allow_unseen_pairs=True,
                validate=False,
            )
            for observation in hypotheses
        ],
        format="csr",
    )
    mean_logit = np.asarray(design @ fit.coefficients, dtype=float).reshape(-1)
    variance = np.asarray(design.power(2) @ fit.covariance_diagonal, dtype=float).reshape(-1)
    posterior_sd = np.sqrt(np.maximum(variance, 0.0))
    if not (
        np.all(np.isfinite(mean_logit))
        and np.all(np.isfinite(posterior_sd))
    ):
        raise PooledCandidateError("response matrix contains a non-finite posterior contrast")

    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for index, (focal, opponent) in enumerate(pair_keys):
        support = _pair_support_details(
            scope_id=scope_id,
            role=role,
            focal=focal,
            opponent=opponent,
            posterior_sd=float(posterior_sd[index]),
            pair_stats=pair_stats,
            pooled_pair_stats=pooled_pair_stats,
        )
        center = float(mean_logit[index])
        spread = float(posterior_sd[index])
        lookup[(focal, opponent)] = {
            "edge": round(100.0 * (float(expit(center)) - 0.5), 2),
            "low": round(100.0 * (float(expit(center - RESPONSE_INTERVAL_Z * spread)) - 0.5), 2),
            "high": round(100.0 * (float(expit(center + RESPONSE_INTERVAL_Z * spread)) - 0.5), 2),
            "evidence": "supported" if support["supported"] else "limited",
            "effective_maps": round(float(support["effective_maps"]), 1),
            "basis": _response_basis(
                effective_maps=float(support["effective_maps"]),
                atom_supported=atom_supported[(focal, opponent)],
            ),
        }

    def matrix(field: str) -> list[list[Any]]:
        return [
            [None if focal == opponent else lookup[(focal, opponent)][field] for opponent in champion_order]
            for focal in champion_order
        ]

    return {
        "champions": [
            {"champion_id": champion, "champion": display_names.get(champion, champion)}
            for champion in champion_order
        ],
        "edge_pp": matrix("edge"),
        "interval_low_pp": matrix("low"),
        "interval_high_pp": matrix("high"),
        "evidence": matrix("evidence"),
        "effective_maps": matrix("effective_maps"),
        "basis": matrix("basis"),
        "grade_thresholds_pp": {"S": 7.5, "A": 3.0, "B": -3.0, "C": -7.5},
    }


def _build_cell_metrics(
    *,
    fit: JointPooledFit,
    scope_id: str,
    role: str,
    patch_id: str,
    champions: Sequence[str],
    counts: Mapping[str, int],
    display_names: Mapping[str, str],
    reference_champions: Mapping[str, str],
    pair_stats: Mapping[tuple[str, str, str, str], Mapping[str, Any]],
    pooled_pair_stats: Mapping[tuple[str, str, str], Mapping[str, Any]],
    resolvers: Mapping[str, AtomMatchupFeatureResolver],
    mapping: MappingArtifact | None,
    exact_atom_patch: str | None,
    posterior_draws: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    registry = AtomFeatureRegistry.from_names(FEATURE_ORDER, source="validated_atom_bridge")
    empty_vector = _empty_vector(registry)
    atom_vector_cache: dict[tuple[str, str, str | None], AtomFeatureVector] = {}
    pool, _pool_legal, pool_hash = _legal_pool(counts, display_names)
    target_mapping = (
        _atom_snapshot_mapping(
            mapping,
            exact_atom_patch,
            resolvers[str(exact_atom_patch)].bridge.artifact_sha256,
        )
        if mapping is not None
        and exact_atom_patch is not None
        and str(exact_atom_patch) in resolvers
        else None
    )
    resolver = resolvers.get(str(exact_atom_patch)) if exact_atom_patch is not None else None
    pair_keys: list[tuple[str, str]] = []
    hypotheses: list[JointMapObservation] = []
    for focal in champions:
        opponents = [opponent for opponent in pool if opponent != focal]
        if len(opponents) < LEGAL_OPPONENT_COUNT:
            opponents.extend(
                opponent
                for opponent in champions
                if opponent != focal and opponent not in opponents
            )
        opponents = opponents[:LEGAL_OPPONENT_COUNT]
        for opponent in opponents:
            vector = empty_vector
            if exact_atom_patch is not None and target_mapping is not None and resolver is not None:
                cache_key = (focal, opponent, exact_atom_patch)
                vector = atom_vector_cache.get(cache_key, empty_vector)
                if cache_key not in atom_vector_cache:
                    try:
                        pair = resolver.resolve_pair(
                            focal,
                            opponent,
                            requested_patch=exact_atom_patch,
                            snapshot_mapping=target_mapping,
                        )
                    except (AtomMatchupFeatureError, KeyError):
                        vector = empty_vector
                    else:
                        vector = AtomFeatureVector.from_values(
                            tuple(pair["features"][name] for name in FEATURE_ORDER),
                            available=tuple(pair["availability"][name] for name in FEATURE_ORDER),
                        )
                    atom_vector_cache[cache_key] = vector
            hypotheses.append(
                _hypothetical_observation(
                    scope_id=scope_id,
                    patch_id=patch_id,
                    role=role,
                    focal=focal,
                    opponent=opponent,
                    reference_champions=reference_champions,
                    atom_vector=vector,
                    empty_vector=empty_vector,
                )
            )
            pair_keys.append((focal, opponent))
    if not hypotheses:
        return [], {"pool": [], "hash": pool_hash}, {}

    design, theta = _posterior_pair_matrix(
        fit,
        hypotheses,
        posterior_draws=posterior_draws,
        seed=POSTERIOR_SEED ^ int(hashlib.sha256(f"{scope_id}|{role}".encode()).hexdigest()[:8], 16),
    )
    probabilities = expit(theta)
    by_focal: dict[str, list[int]] = defaultdict(list)
    for index, (focal, _opponent) in enumerate(pair_keys):
        by_focal[focal].append(index)
    rows: list[dict[str, Any]] = []
    for focal in champions:
        indices = by_focal.get(focal, [])
        opponents = [pair_keys[index][1] for index in indices]
        matchup_profile: list[dict[str, Any]] = []
        support_sources: list[str] = []
        if not indices:
            mean_probability = 0.5
            blind_draws = np.full(posterior_draws, 0.5)
            counter_draws = np.zeros(posterior_draws)
            counter_probabilities = np.zeros(0)
            effective_maps = 0.0
            supported = []
            contrast_sd = float("inf")
        else:
            opponent_weights = np.asarray(
                [float(counts.get(opponent, 0)) for opponent in opponents],
                dtype=float,
            )
            if not np.any(opponent_weights):
                opponent_weights = np.ones(len(opponents), dtype=float)
            opponent_weights /= opponent_weights.sum()
            focal_probability_matrix = probabilities[indices]
            blind_draws = _weighted_lower_tail(focal_probability_matrix.T, opponent_weights, BLIND_TAIL_SHARE)
            mean_probability = float(np.dot(opponent_weights, focal_probability_matrix.mean(axis=1)))
            theta_sd = np.std(theta[indices], axis=1, ddof=1) if posterior_draws > 1 else np.zeros(len(indices))
            counter_probabilities = np.mean(theta[indices] > COUNTER_EFFECT_THRESHOLD_LOGIT, axis=1)
            counter_draws = LEGAL_OPPONENT_COUNT * np.sum(
                opponent_weights[:, None] * (theta[indices] > COUNTER_EFFECT_THRESHOLD_LOGIT),
                axis=0,
            )
            supported = []
            effective_maps = 0.0
            pair_support: list[dict[str, Any]] = []
            for local_index, opponent in enumerate(opponents):
                support = _pair_support_details(
                    scope_id=scope_id,
                    role=role,
                    focal=focal,
                    opponent=opponent,
                    posterior_sd=float(theta_sd[local_index]),
                    pair_stats=pair_stats,
                    pooled_pair_stats=pooled_pair_stats,
                )
                if support["supported"]:
                    supported.append(opponent)
                    support_sources.append(str(support["evidence_source"]))
                    effective_maps += float(support["effective_maps"])
                pair_support.append(support)
            contrast_sd = float(np.max(theta_sd)) if theta_sd.size else math.inf
            row_weight = opponent_weights

            for local_index, opponent in enumerate(opponents):
                pair_probabilities = expit(theta[indices[local_index]])
                support = pair_support[local_index]
                matchup_profile.append(
                    {
                        "champion": display_names.get(opponent, opponent),
                        "champion_id": opponent,
                        "model_edge_pp": round(100.0 * (float(np.mean(pair_probabilities)) - 0.5), 4),
                        "posterior_interval_pp": {
                            "low": round(100.0 * (float(np.quantile(pair_probabilities, 0.10)) - 0.5), 4),
                            "high": round(100.0 * (float(np.quantile(pair_probabilities, 0.90)) - 0.5), 4),
                        },
                        "posterior_positive_probability": round(float(counter_probabilities[local_index]), 6),
                        "effective_maps": round(float(support["effective_maps"]), 4),
                        "series_count": int(support["series_count"]),
                        "evidence_status": (
                            "supported"
                            if support["supported"]
                            else "limited"
                        ),
                        "evidence_source": support["evidence_source"],
                    }
                )

        legal_opponents = [
            {
                "champion": display_names.get(opponent, opponent),
                "champion_id": opponent,
                "weight": round(float(weight), 8),
            }
            for opponent, weight in zip(opponents, row_weight if indices else ())
        ]
        legal_hash = _sha256_bytes(_canonical_json(legal_opponents))
        available = _matchup_metrics_available(
            opponent_count=len(opponents),
            supported_opponent_count=len(supported),
            contrast_sd=contrast_sd,
        )
        blind_score = (
            _blind_point_estimate(focal_probability_matrix, row_weight)
            if indices
            else 0.5
        )
        counter_count = (
            _counter_count_point_estimate(theta[indices])
            if indices
            else 0
        )
        expected_breadth = float(np.dot(row_weight, counter_probabilities)) if indices else 0.0
        row = {
            "champion": display_names.get(focal, focal),
            "champion_id": focal,
            "tier_value_pp": round(100.0 * (mean_probability - 0.5), 4),
            "strength_score": round(mean_probability, 6),
            "strength_sd_logit": round(float(np.std(theta[indices], ddof=1)) if indices and posterior_draws > 1 else 0.0, 6),
            "rating": round(
                INITIAL_RATING
                + float(fit.scope_role_champion_strengths[scope_id][role].get(focal, 0.0))
                * 400.0
                / math.log(10.0),
                4,
            ),
            "played_maps": int(counts.get(focal, 0)),
            "counterability_status": "available" if available else "unavailable",
            "counterability": round(100.0 * (1.0 - blind_score), 4) if available else None,
            "matchup_maps": round(effective_maps, 4),
            "matchup_opponents": len(supported),
            "blind_score_pp": round(100.0 * (blind_score - 0.5), 4) if available else None,
            "counter_score": round(LEGAL_OPPONENT_COUNT * expected_breadth, 4) if available else None,
            "expected_counter_breadth": round(LEGAL_OPPONENT_COUNT * expected_breadth, 4) if available else None,
            "countered_opponent_count": counter_count if available else None,
            "countered_opponent_share": round(counter_count / LEGAL_OPPONENT_COUNT, 4) if available else None,
            "legal_opponent_distribution_sha256": legal_hash,
            "legal_opponents": legal_opponents,
            "matchup_profile": matchup_profile,
            "legal_opponent_coverage": round(len(supported) / max(1, len(opponents)), 4),
            "counterability_evidence_scope": (
                "scope" if support_sources and all(source == "scope" for source in support_sources)
                else "pooled_scopes" if support_sources else None
            ),
            "strength_component_identified": True,
            "maximum_strength_contrast_sd": round(contrast_sd, 6) if math.isfinite(contrast_sd) else None,
            "_blind_draws": blind_draws,
            "_counter_draws": counter_draws,
        }
        rows.append(row)

    rows.sort(key=lambda row: (-float(row["strength_score"]), _normalize_name(row["champion"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    _assign_tier_buckets(rows)
    response_matrix = _response_matrix(
        fit=fit,
        scope_id=scope_id,
        role=role,
        patch_id=patch_id,
        champion_order=[str(row["champion_id"]) for row in rows],
        display_names=display_names,
        reference_champions=reference_champions,
        pair_stats=pair_stats,
        pooled_pair_stats=pooled_pair_stats,
        resolvers=resolvers,
        target_mapping=target_mapping,
        exact_atom_patch=exact_atom_patch,
        empty_vector=empty_vector,
        atom_vector_cache=atom_vector_cache,
    )
    design_summary = {
        "model_schema": fit.metadata["schema_version"],
        "fit_coordinates": "sparse reference-coded joint map likelihood",
        "parameter_count": fit.metadata["n_parameters"],
        "posterior_draws": posterior_draws,
        "design_rows": len(hypotheses),
        "atom_patch": exact_atom_patch,
    }
    pair_designs = {
        (focal, opponent): design[index]
        for index, (focal, opponent) in enumerate(pair_keys)
    }
    return rows, {
        "pool": [
            {
                "champion": display_names.get(champion, champion),
                "champion_id": champion,
                "weight": round(float(weight), 8),
            }
            for champion, weight in zip(pool, np.asarray([counts.get(champion, 0) for champion in pool], dtype=float) / max(1, sum(counts.get(champion, 0) for champion in pool)))
        ],
        "hash": pool_hash,
        "selection_rule": "top six role picks in the exact scope; exclude focal champion; take five and renormalize",
    }, {
        "design": design,
        "pair_keys": pair_keys,
        "rows": rows,
        "response_matrix": response_matrix,
        "summary": design_summary,
    }


def _loo_stability(
    fit: JointPooledFit,
    observations: Sequence[JointMapObservation],
    cell_designs: Mapping[tuple[str, str], Mapping[str, Any]],
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Run first-order Laplace leave-one-series-out tier sensitivity.

    The deletion uses the exact per-series score contribution and the
    diagonal Laplace Hessian from the joint fit. It is a one-step LOO
    approximation. The report labels that approximation so it is not treated
    as an exact refit.
    """

    if not observations:
        return {"status": "unavailable", "reason": "no_observations"}
    predictions = np.asarray([float(row["probability"]) for row in fit.map_predictions], dtype=float)
    outcomes = np.asarray([row.outcome for row in observations], dtype=float)
    weights = np.asarray([float(row.weight) for row in observations], dtype=float)
    groups: dict[str, list[int]] = defaultdict(list)
    for index, observation in enumerate(observations):
        groups[str(observation.series_id or observation.map_id)].append(index)
    ordered_series = sorted(
        groups,
        key=lambda series: max(str(observations[index].map_id) for index in groups[series]),
        reverse=True,
    )[:MAX_LOO_SERIES]
    full_labels = {
        (str(cell["scope_id"]), str(cell["role"]), str(row["champion_id"])): row["tier_bucket"]
        for cell in cells
        for row in cell.get("rows", [])
    }
    agreements: list[float] = []
    for series in ordered_series:
        gradient = np.asarray(
            fit.design_matrix[groups[series]].T
            @ (weights[groups[series]] * (predictions[groups[series]] - outcomes[groups[series]])),
            dtype=float,
        ).reshape(-1)
        beta_loo = fit.coefficients + fit.covariance_diagonal * gradient
        same = 0
        total = 0
        for cell in cells:
            scope_id = str(cell["scope_id"])
            role = str(cell["role"])
            entry = cell_designs.get((scope_id, role))
            if not entry:
                continue
            pair_rows: list[sparse.spmatrix] = []
            pair_keys: list[tuple[str, str]] = []
            scores: list[dict[str, Any]] = []
            for champion_id, pair_map in entry["by_champion"].items():
                for opponent, row in pair_map.items():
                    pair_rows.append(row)
                    pair_keys.append((champion_id, opponent))
            if not pair_rows:
                continue
            pair_scores = np.asarray(
                sparse.vstack(pair_rows, format="csr") @ beta_loo,
                dtype=float,
            ).reshape(-1)
            status_by_champion = {
                str(row["champion_id"]): row["counterability_status"]
                for row in cell["rows"]
            }
            for champion_id in entry["by_champion"]:
                indices = [
                    index
                    for index, (focal, _opponent) in enumerate(pair_keys)
                    if focal == champion_id
                ]
                probabilities = pair_scores[indices]
                scores.append(
                    {
                        "champion": champion_id,
                        "strength_score": float(np.mean(expit(probabilities))) if probabilities.size else 0.5,
                        "counterability_status": status_by_champion[champion_id],
                        "blind_score_pp": 0.0,
                        "counter_score": float(np.sum(probabilities > COUNTER_EFFECT_THRESHOLD_LOGIT)),
                        "countered_opponent_count": int(np.sum(probabilities > COUNTER_EFFECT_THRESHOLD_LOGIT)),
                        "countered_opponent_share": 0.0,
                        "_blind_draws": np.asarray([float(np.min(expit(probabilities)))]) if probabilities.size else np.asarray([0.5]),
                        "_counter_draws": np.asarray([float(np.sum(probabilities > COUNTER_EFFECT_THRESHOLD_LOGIT))]),
                        "rating": 1500.0,
                    }
                )
            _assign_tier_buckets(scores)
            for row in scores:
                key = (scope_id, role, row["champion"])
                if key in full_labels:
                    same += int(full_labels[key] == row["tier_bucket"])
                    total += 1
        if total:
            agreements.append(same / total)
    return {
        "status": "complete" if agreements else "unavailable",
        "method": "first-order Laplace leave-one-series-out deletion influence",
        "refit": False,
        "series_total": len(groups),
        "series_evaluated": len(ordered_series),
        "series_selection": "latest deterministic series by map id",
        "tier_agreement_mean": round(float(np.mean(agreements)), 6) if agreements else None,
        "tier_agreement_min": round(float(np.min(agreements)), 6) if agreements else None,
        "posterior_draws": POSTERIOR_DRAWS,
        "claim_ceiling": "diagnostic stability; exact refit remains outside this candidate builder",
    }


def build_pooled_candidate(
    root: Path,
    *,
    as_of: pd.Timestamp | None = None,
    expected_live_as_of: pd.Timestamp | None = None,
    previous: Mapping[str, Any] | None = None,
    min_appearances: int = DEFAULT_MIN_APPEARANCES,
    source_mode: str = "oe_only",
    allowed_game_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    if source_mode not in SOURCE_MODES:
        raise PooledCandidateError(f"source_mode must be one of {', '.join(SOURCE_MODES)}")
    frame, source_sha256, source_locator = _load_source(
        root,
        as_of=as_of,
        allowed_game_ids=allowed_game_ids,
    )
    raw_maps, rejected_maps = _build_maps(frame)
    if not raw_maps:
        raise PooledCandidateError("no complete five-role maps remain after identity checks")
    crosswalk, identity_sources = _load_crosswalk(root)
    from lol_kills.v2.champions.atoms.consume import AtomBridge

    resolvers: dict[str, AtomMatchupFeatureResolver] = {}
    bridge_paths: dict[str, Path] = {}
    for declared_patch, locator in ATOM_BRIDGE_LOCATORS.items():
        path = root / locator
        if not path.is_file():
            continue
        bridge = AtomBridge.load(path)
        resolver = AtomMatchupFeatureResolver(bridge)
        actual_patch = resolver.snapshot_patch
        if actual_patch != declared_patch:
            raise PooledCandidateError(
                f"atom bridge patch mismatch for {locator}: {actual_patch!r} != {declared_patch!r}"
            )
        resolvers[declared_patch] = resolver
        bridge_paths[declared_patch] = path
    if not resolvers:
        raise PooledCandidateError("no validated LCC atom bridge is available")
    default_patch = "26.15" if "26.15" in resolvers else sorted(resolvers)[0]
    default_resolver = resolvers[default_patch]
    registry = AtomFeatureRegistry.from_names(FEATURE_ORDER, source="validated_atom_bridge")
    mapping, mapping_meta = _mapping_for_root(root)

    offsets = _team_offsets(raw_maps)
    observations: list[JointMapObservation] = []
    prepared: list[dict[str, Any]] = []
    unresolved: set[str] = set()
    display_names: dict[str, str] = {}
    appearance_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    regional_counts: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
    regional_game_ids: dict[tuple[str, str], set[str]] = defaultdict(set)
    patch_meta_counts: Counter[str] = Counter()
    resolution_counts: Counter[str] = Counter()
    atom_pair_cache: dict[tuple[str, str, str | None], AtomFeatureVector] = {}
    reference_date = max(pd.Timestamp(game["date"]) for game in raw_maps)
    for game, offset in zip(raw_maps, offsets):
        stable_roles: dict[str, dict[str, str]] = {}
        valid = True
        for role in ROLES:
            stable_roles[role] = {}
            for side in ("blue", "red"):
                name = str(game["roles"][role][f"{side}_champion"]).strip()
                stable_id = crosswalk.get(_normalize_name(name))
                if stable_id is None:
                    unresolved.add(name)
                    valid = False
                else:
                    stable_roles[role][side] = stable_id
                    display_names.setdefault(stable_id, name)
        if not valid:
            continue
        age_days = max(0.0, (reference_date - pd.Timestamp(game["date"])).total_seconds() / 86400.0)
        weight = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
        atom_vectors, patch_meta = _map_atom_features(
            game,
            stable_roles=stable_roles,
            resolvers=resolvers,
            registry=registry,
            mapping=mapping,
            pair_cache=atom_pair_cache,
        )
        patch_meta_counts[patch_meta["mapping_status"]] += 1
        resolution_counts[patch_meta["mapping_reason"]] += 1
        patch_scope_id = f"patch:{patch_meta['oe_patch']}"
        prepared_game = dict(game)
        prepared_game.update(
            {
                "scope_id": patch_scope_id,
                "scope_kind": "patch",
                "scope_label": f"Patch {patch_meta['oe_patch']}",
                "stable_roles": stable_roles,
                "weight": weight,
                "team_logit": offset,
                "oe_patch_id": patch_meta["oe_patch"],
                "official_patch": patch_meta["official_patch"],
                "atom_snapshot_patch": patch_meta["atom_snapshot_patch"],
                "atom_exact_role_pairs": patch_meta["exact_atom_role_pairs"],
            }
        )
        prepared.append(prepared_game)
        for role in ROLES:
            for side in ("blue", "red"):
                appearance_counts[(patch_scope_id, role)][stable_roles[role][side]] += 1
        for context in _regional_contexts(prepared_game):
            regional_game_ids[(patch_scope_id, context)].add(str(game["game_id"]))
            for role in ROLES:
                for side in ("blue", "red"):
                    regional_counts[(patch_scope_id, context, role)][stable_roles[role][side]] += 1
        observations.append(
            JointMapObservation(
                map_id=str(game["game_id"]),
                outcome=int(game["y_blue_win"]),
                scope_id=patch_scope_id,
                oe_patch_id=str(patch_meta["oe_patch"]),
                picks={role: (stable_roles[role]["blue"], stable_roles[role]["red"]) for role in ROLES},
                atom_pair_features=atom_vectors,
                series_id=str(game["series_id"]),
                offset=float(offset),
                weight=float(weight),
                synthetic=False,
            )
        )
    if not observations:
        raise PooledCandidateError("no maps have complete champion identity coverage")

    latest_prepared = max(prepared, key=lambda game: (game["date"], game["game_id"]))
    current_patch_verified = (
        mapping is not None
        and any(game.get("atom_snapshot_patch") == CURRENT_PUBLIC_PATCH for game in prepared)
        and CURRENT_PUBLIC_PATCH in resolvers
    )

    fit = fit_joint_pooled_model(
        observations,
        feature_registry=registry,
        roles=ROLES,
        atom_deviation_dim=ATOM_DEVIATION_DIM,
        priors=PriorScales(),
        max_iter=500,
    )
    pair_stats = _pair_stats(prepared)
    pooled_pair_stats = _pool_pair_stats(pair_stats)
    previous_rows = _previous_rows(previous)
    cells: list[dict[str, Any]] = []
    cell_designs: dict[tuple[str, str], Mapping[str, Any]] = {}
    scope_observations: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for game in prepared:
        scope_observations[str(game["scope_id"])].append(game)

    for scope_id in sorted(scope_observations):
        scope_games = scope_observations[scope_id]
        latest_game = max(scope_games, key=lambda game: (game["date"], game["game_id"]))
        patch_id = str(latest_game["oe_patch_id"])
        # Each board pools every eligible competition in one patch. Exact atom
        # features remain available only when that patch has an audited mapping.
        exact_atom_patch = _scope_atom_patch(scope_games) if mapping is not None else None
        for role in ROLES:
            counts = appearance_counts[(scope_id, role)]
            champions = sorted(
                counts,
                key=lambda champion: _normalize_name(display_names.get(champion, champion)),
            )
            reference_champions = {
                candidate_role: sorted(
                    appearance_counts[(scope_id, candidate_role)],
                    key=lambda champion: _normalize_name(display_names.get(champion, champion)),
                )[0]
                for candidate_role in ROLES
            }
            rows, pool_meta, design_meta = _build_cell_metrics(
                fit=fit,
                scope_id=scope_id,
                role=role,
                patch_id=patch_id,
                champions=champions,
                counts=counts,
                display_names=display_names,
                reference_champions=reference_champions,
                pair_stats=pair_stats,
                pooled_pair_stats=pooled_pair_stats,
                resolvers=resolvers,
                mapping=mapping,
                exact_atom_patch=exact_atom_patch,
                posterior_draws=POSTERIOR_DRAWS,
            )
            profile_resolver = (
                default_resolver
                if exact_atom_patch is None
                else resolvers.get(str(exact_atom_patch))
            )
            for row in rows:
                prior = previous_rows.get((scope_id, role, row["champion"])) or previous_rows.get((scope_id, role, row["champion_id"]))
                previous_rank = prior.get("rank") if isinstance(prior, Mapping) and isinstance(prior.get("rank"), int) else None
                previous_rating = prior.get("rating") if isinstance(prior, Mapping) and isinstance(prior.get("rating"), (int, float)) else None
                rank_delta = previous_rank - row["rank"] if previous_rank is not None else None
                rating = float(fit.scope_role_champion_strengths[scope_id][role].get(row["champion_id"], 0.0))
                row["rating"] = round(INITIAL_RATING + rating * 400.0 / math.log(10.0), 4)
                rating_delta = row["rating"] - float(previous_rating) if previous_rating is not None else None
                row.update(
                    {
                        "previous_rank": previous_rank,
                        "rank_delta": rank_delta,
                        "rating_delta": round(rating_delta, 4) if rating_delta is not None else None,
                        "movement": "new" if rank_delta is None else "up" if rank_delta > 0 else "down" if rank_delta < 0 else "flat",
                        "atom_profile_status": (
                            profile_resolver.bridge.profile(row["champion_id"]) or {}
                        ).get("profile_status", "unavailable")
                        if profile_resolver is not None
                        else "unavailable",
                        "atom_patch_last_changed": (
                            profile_resolver.bridge.profile(row["champion_id"]) or {}
                        ).get("lcc_patch_last_changed")
                        if profile_resolver is not None
                        else None,
                    }
                )
            scope_kind = "patch"
            scope_label = f"Patch {patch_id}"
            league = None
            competition_tier = None
            event_kind = None
            region = None
            regional_views = _build_regional_views(
                rows=rows,
                scope_id=scope_id,
                role=role,
                regional_counts=regional_counts,
                regional_game_ids=regional_game_ids,
            )
            cell = {
                "scope_id": scope_id,
                "scope_kind": scope_kind,
                "scope_label": scope_label,
                "region": region,
                "league": league,
                "event_kind": event_kind,
                "competition_tier": competition_tier,
                "role": role,
                "patches": [patch_id],
                "oe_patches": sorted({str(game["oe_patch_id"]) for game in scope_games}),
                "official_patches": sorted({str(game["official_patch"]) for game in scope_games if game.get("official_patch")}),
                "atom_snapshot_patches": sorted({str(game["atom_snapshot_patch"]) for game in scope_games if game.get("atom_snapshot_patch")})
                or ([exact_atom_patch] if exact_atom_patch else []),
                "as_of": _utc_stamp(max(game["date"] for game in scope_games)),
                "status": "development_only",
                "identity_status": "complete" if not unresolved else "unavailable",
                "unresolved_champion_identities": sorted(unresolved),
                "row_count": len(rows),
                "legal_opponents": pool_meta["pool"],
                "legal_opponent_distribution_sha256": pool_meta["hash"],
                "legal_opponent_selection_rule": pool_meta["selection_rule"],
                "strength_design": design_meta["summary"],
                "atom_snapshot_status": "exact" if exact_atom_patch else "unavailable",
                "regional_views": regional_views,
                "response_matrix": design_meta["response_matrix"],
                "rows": rows,
            }
            cells.append(cell)
            cell_designs[(scope_id, role)] = {
                "by_champion": {
                    focal: {
                        opponent: design_meta["design"][index]
                        for index, (candidate, opponent) in enumerate(design_meta["pair_keys"])
                        if candidate == focal
                    }
                    for focal in champions
                }
            }

    stability = _loo_stability(fit, observations, cell_designs, cells)
    source_latest = max(game["date"] for game in raw_maps)
    expected = None
    if expected_live_as_of is not None:
        expected = pd.Timestamp(expected_live_as_of)
        if expected.tzinfo is None:
            expected = expected.tz_localize("UTC")
        else:
            expected = expected.tz_convert("UTC")
    source_complete = expected is None or source_latest >= expected
    exact_atom_maps = sum(bool(game.get("atom_snapshot_patch")) for game in prepared)
    atom_role_pairs = sum(int(game.get("atom_exact_role_pairs", 0)) for game in prepared)
    patch_counts = Counter(str(game["oe_patch_id"]) for game in prepared)
    selected_patch = str(latest_prepared.get("atom_snapshot_patch") or default_patch)
    if selected_patch not in resolvers:
        selected_patch = default_patch
    selected_resolver = resolvers[selected_patch]
    selected_bridge_path = bridge_paths[selected_patch]
    atom_bridges = {
        patch: {
            "locator": str(path.relative_to(root)),
            "raw_sha256": _sha256_path(path),
            "artifact_sha256": resolvers[patch].bridge.artifact_sha256,
            "generated_at": resolvers[patch].bridge.generated_at,
            "data_patch": resolvers[patch].snapshot_patch,
            "lcc_commit": resolvers[patch].bridge.provenance.get("lcc_commit"),
        }
        for patch, path in sorted(bridge_paths.items())
    }
    payload: dict[str, Any] = {
        "schema_version": POOLED_CANDIDATE_SCHEMA,
        "legacy_schema_version": "scryglass:champion-role-elo-candidate:v1",
        "artifact_kind": "tier_list_candidate",
        "artifact_sha256": "",
        "status": "development_only",
        "development_only": True,
        "publication_eligible": False,
        "production_eligible": False,
        "source_mode": source_mode,
        "history_start": _utc_stamp(HISTORY_START),
        "live_window_start": _utc_stamp(LIVE_WINDOW_START),
        "as_of": _utc_stamp(source_latest),
        "expected_live_as_of": _utc_stamp(expected) if expected is not None else None,
        "source_complete_through_expected_live_as_of": source_complete,
        "source": {
            "locator": source_locator,
            "raw_sha256": source_sha256,
            "source_files": [source_locator],
            "maps_replayed": len(raw_maps),
            "source_identity_sha256": identity_sha256(
                str(game["game_id"]) for game in raw_maps
            ),
            "maps_used_in_joint_likelihood": len(observations),
            "maps_rejected_incomplete_roles": rejected_maps,
            "maps_rejected_identity": len(raw_maps) - len(prepared),
            "maps_in_live_window": sum(pd.Timestamp(game["date"]) >= LIVE_WINDOW_START for game in raw_maps),
            "source_earliest_replayed": _utc_stamp(min(game["date"] for game in raw_maps)),
            "source_latest_replayed": _utc_stamp(source_latest),
        },
        "identity_sources": identity_sources,
        "patch_ingestion": {
            "mode": "champion_atomization",
            "canonical_data_patch": selected_resolver.snapshot_patch,
            "atom_bridge_locator": str(selected_bridge_path.relative_to(root)),
            "atom_bridge_artifact_sha256": selected_resolver.bridge.artifact_sha256,
            "atom_bridge_raw_sha256": _sha256_path(selected_bridge_path),
            "atom_bridge_generated_at": selected_resolver.bridge.generated_at,
            "lcc_commit": selected_resolver.bridge.provenance.get("lcc_commit"),
            "atom_bridges": atom_bridges,
            "oe_patch_namespace": "Oracle's Elixir source patch token",
            "official_to_oe_patch_mapping": mapping_meta,
            "oe_patch_counts": dict(sorted(patch_counts.items())),
            "mapping_status_counts": dict(sorted(patch_meta_counts.items())),
            "resolution_status_counts": dict(sorted(resolution_counts.items())),
            "exact_atom_snapshot_maps": exact_atom_maps,
            "exact_atom_role_pairs": atom_role_pairs,
            "historical_atom_snapshot_policy": "unavailable unless the audited sidecar names an exact time-safe snapshot",
            "use": "patch update agenda, champion-state provenance, and structured atom feature source",
        },
        "joint_model": {
            **fit.metadata,
            "posterior_draws_requested": POSTERIOR_DRAWS,
            "posterior_draws_verified": POSTERIOR_DRAWS,
            "posterior_seed": POSTERIOR_SEED,
            "joint_posterior_draws_gate": True,
            "atom_feature_count": len(FEATURE_ORDER),
            "atom_feature_schema": selected_resolver.feature_schema_sha256,
            "atom_exact_map_coverage": round(exact_atom_maps / max(1, len(prepared)), 6),
        },
        "stability": stability,
        "current_patch_verified": current_patch_verified,
        "options": {
            "roles": list(ROLES),
            "patches": sorted({patch for cell in cells for patch in cell["patches"]}),
            "tier_buckets": list(TIER_BUCKETS),
        },
        "rating_method": {
            "name": "patch-wide joint five-role pooled map likelihood with pre-map team Elo control",
            "initial_rating": INITIAL_RATING,
            "team_k": TEAM_K,
            "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
            "fit": "penalized maximum a posteriori with full observed-Hessian diagonal Laplace covariance",
            "fit_coordinates": "sparse reference-coded joint map rows",
            "update_order": "team control is chronological; each patch board pools every eligible completed map once across leagues and events",
            "maximum_supported_strength_contrast_sd": STRENGTH_MAX_CONTRAST_SD,
            "rating_claim": "standardized descriptive paired-comparison strength; not an outcome-calibrated probability",
        },
        "matchup_shape_method": {
            "name": "OE-supported blind tail risk and counter breadth from the same joint map likelihood",
            "blind_definition": "posterior-mean lower-tail matchup edge across five common role opponents",
            "blind_tail_share": BLIND_TAIL_SHARE,
            "counter_definition": "count of five common role opponents with a positive model contrast above the logit effect threshold",
            "counter_posterior_probability_threshold": COUNTER_POSTERIOR_THRESHOLD,
            "counter_effect_threshold_logit": COUNTER_EFFECT_THRESHOLD_LOGIT,
            "response_matrix_definition": "complete same-role MAP contrast matrix with diagonal-Laplace 80 percent intervals",
            "response_matrix_interval_z": RESPONSE_INTERVAL_Z,
            "minimum_effective_maps": MATCHUP_MIN_EFFECTIVE_MAPS,
            "minimum_effective_series": MATCHUP_MIN_SERIES,
            "outcome_variation_required": True,
            "minimum_opponents": LEGAL_OPPONENT_COUNT,
            "legal_opponent_count": LEGAL_OPPONENT_COUNT,
            "legal_opponent_selection": "top six role picks in the exact scope; exclude the focal champion; take five and renormalize",
            "maximum_pair_posterior_sd": MATCHUP_MAX_POSTERIOR_SD,
            "atom_adjustment": "used when an exact time-safe atom snapshot is available; not required for OE-supported matchup publication",
            "posterior_draws": POSTERIOR_DRAWS,
            "minimum_special_tier_membership_probability": TIER_MEMBERSHIP_PROBABILITY,
            "pick_order_claim": False,
            "rating_claim": "descriptive matchup-shape proxy; not a causal counter-pick estimate",
        },
        "claim_ceiling": {
            "production": False,
            "publication": False,
            "outcome_calibrated_probability": False,
            "recommendation": False,
            "betting": False,
            "causal_draft_effect": False,
        },
        "unresolved_champion_identities": sorted(unresolved),
        "cells": cells,
    }
    payload["artifact_sha256"] = _sha256_bytes(
        _canonical_json({key: value for key, value in payload.items() if key != "artifact_sha256"})
    )
    return payload


__all__ = ["POOLED_CANDIDATE_SCHEMA", "PooledCandidateError", "build_pooled_candidate"]
