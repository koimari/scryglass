"""Wave-2 L6 neutral composition interaction model.

This module provides a replayable, deterministic development implementation for
finished five-versus-five champion/composition interactions under the neutral,
side-invariant contract.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
from itertools import combinations
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np

from lol_kills.v2.champions import load_champion_ontology
from lol_kills.v2.evaluation.types import canonical_sha256

from .fixtures import reveal_synthetic_seed
from .types import (
    CANONICAL_ROLES,
    DraftCompositionRow,
    DraftInteractionCandidate,
    DraftInteractionError,
    DraftInteractionFit,
    DraftInteractionFitDiagnostics,
    DraftInteractionPrediction,
    DraftInteractionSelectionReport,
)

CANONICAL_ROLE_LIST = tuple(CANONICAL_ROLES)
ROLE_INDEX = {role: idx for idx, role in enumerate(CANONICAL_ROLE_LIST)}


def _sigmoid(values: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(values, dtype=float), -25.0, 25.0)))


def _safe_float(value: Any, *, field_name: str) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating, np.float32, np.float64, np.float16)):
        raise DraftInteractionError(f"{field_name} must be numeric")
    if isinstance(value, bool):
        raise DraftInteractionError(f"{field_name} must be numeric")
    parsed = float(value)
    if not isfinite(parsed):
        raise DraftInteractionError(f"{field_name} must be finite")
    return parsed


def _stable_seed(value: str) -> int:
    return int(sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def _ordered_role_champ(role: str, champion: str) -> tuple[int, str]:
    return ROLE_INDEX.get(role, 999), champion


def _ordered_pair(
    left_role: str,
    left_champion: str,
    right_role: str,
    right_champion: str,
) -> tuple[tuple[str, str], tuple[str, str], float]:
    left_key = _ordered_role_champ(left_role, left_champion)
    right_key = _ordered_role_champ(right_role, right_champion)
    if left_key <= right_key:
        return (left_role, left_champion), (right_role, right_champion), 1.0
    return (right_role, right_champion), (left_role, left_champion), -1.0


def _ordered_cross_pair(
    left_role: str,
    left_champion: str,
    left_side: str,
    right_role: str,
    right_champion: str,
    right_side: str,
) -> tuple[tuple[str, str, str], tuple[str, str, str], float]:
    left_key = _ordered_role_champ(left_role, left_champion)
    right_key = _ordered_role_champ(right_role, right_champion)
    if left_key <= right_key:
        return (left_role, left_champion, left_side), (right_role, right_champion, right_side), 1.0
    return (right_role, right_champion, right_side), (left_role, left_champion, left_side), -1.0


def _role_pair_signature(roles: Sequence[str], champions: Sequence[str]) -> str:
    return "|".join(f"{role}:{champion}" for role, champion in zip(roles, champions))


@dataclass(frozen=True)
class _TermContribution:
    term_id: str
    block: str
    value: float
    metadata: Mapping[str, Any]


def _structured_term_id(contribution: _TermContribution) -> str:
    metadata = dict(contribution.metadata)
    block = contribution.block
    scope = str(metadata.get("scope", "global"))
    roles = list(metadata.get("roles", ()))
    champions = list(metadata.get("champions", ()))
    buckets = list(metadata.get("buckets", ()))
    if roles and champions and len(roles) == len(champions):
        records = sorted(
            zip(roles, champions, buckets or [""] * len(roles)),
            key=lambda item: _ordered_role_champ(str(item[0]), str(item[1])),
        )
        roles = [str(item[0]) for item in records]
        champions = [str(item[1]) for item in records]
        if buckets:
            buckets = [str(item[2]) for item in records]
    identity: dict[str, Any] = {"block": block, "scope": scope}
    if scope == "competition_scope":
        identity["competition_scope_id"] = metadata.get(
            "competition_scope_id"
        )
    if scope in {"league", "patch"}:
        identity["league_id"] = metadata.get("league_id")
    if scope == "patch":
        identity["patch_id"] = metadata.get("patch_id")
    if block == "main_global":
        identity["champion_id"] = metadata.get("champion_id")
    elif block.startswith("main_"):
        identity.update(
            {
                "role": metadata.get("role"),
                "champion_id": metadata.get("champion_id"),
            }
        )
    elif block.startswith(("ally_sparse", "enemy_sparse")):
        identity.update({"roles": roles, "buckets": buckets})
    elif block == "fm_sparse":
        identity.update({"roles": roles, "dimension": metadata.get("dimension")})
    elif block == "whole_team":
        identity["team"] = metadata.get("team")
    elif block == "cross_team":
        identity["teams"] = sorted(metadata.get("teams", ()))
    else:
        identity.update({"roles": roles, "champions": champions})
        if metadata.get("role") is not None:
            identity["role"] = metadata.get("role")
        if metadata.get("champion_id") is not None:
            identity["champion_id"] = metadata.get("champion_id")
    return f"{block}:{canonical_sha256(identity)}"


@dataclass(frozen=True)
class DraftInteractionFamily:
    family_id: str
    include_ally_sparse: bool
    include_ally_exact: bool
    include_enemy_sparse: bool
    include_enemy_exact: bool
    include_whole_team: bool
    include_cross_team: bool
    include_factorization: bool
    use_archetype_transfer: bool
    description: str = ""


class DraftInteractionModel:
    """Replayable neutral finished-composition interaction model."""

    _BLOCK_ORDER = (
        "main_global",
        "main_role",
        "main_scope",
        "main_league",
        "main_patch",
        "ally_sparse",
        "ally_exact",
        "enemy_sparse",
        "enemy_exact",
        "whole_team",
        "cross_team",
        "fm_sparse",
    )

    FAMILY_REGISTRY: tuple[DraftInteractionFamily, ...] = (
        DraftInteractionFamily(
            family_id="main-only",
            include_ally_sparse=False,
            include_ally_exact=False,
            include_enemy_sparse=False,
            include_enemy_exact=False,
            include_whole_team=False,
            include_cross_team=False,
            include_factorization=False,
            use_archetype_transfer=True,
            description="main effects only",
        ),
        DraftInteractionFamily(
            family_id="pair-baseline",
            include_ally_sparse=True,
            include_ally_exact=False,
            include_enemy_sparse=True,
            include_enemy_exact=False,
            include_whole_team=True,
            include_cross_team=False,
            include_factorization=False,
            use_archetype_transfer=True,
            description="four-way family with sparse pair effects and whole-team residual",
        ),
        DraftInteractionFamily(
            family_id="fm-sparse",
            include_ally_sparse=False,
            include_ally_exact=False,
            include_enemy_sparse=False,
            include_enemy_exact=False,
            include_whole_team=True,
            include_cross_team=False,
            include_factorization=True,
            use_archetype_transfer=True,
            description="sparse low-rank interaction baseline",
        ),
        DraftInteractionFamily(
            family_id="no-archetype-transfer",
            include_ally_sparse=True,
            include_ally_exact=True,
            include_enemy_sparse=True,
            include_enemy_exact=True,
            include_whole_team=True,
            include_cross_team=True,
            include_factorization=False,
            use_archetype_transfer=False,
            description="exact family without archetype prior fallback",
        ),
        DraftInteractionFamily(
            family_id="residual-full",
            include_ally_sparse=True,
            include_ally_exact=True,
            include_enemy_sparse=True,
            include_enemy_exact=True,
            include_whole_team=True,
            include_cross_team=True,
            include_factorization=False,
            use_archetype_transfer=True,
            description="stronger full residual decomposition",
        ),
    )

    _PRIOR_VARIANCE = {
        "main_global": 0.20,
        "main_role": 0.16,
        "main_scope": 0.13,
        "main_league": 0.10,
        "main_patch": 0.06,
        "ally_sparse": 0.30,
        "ally_exact": 0.18,
        "enemy_sparse": 0.30,
        "enemy_exact": 0.18,
        "whole_team": 0.22,
        "cross_team": 0.36,
        "fm_sparse": 0.24,
    }

    _SELECTION_MIN_ROWS = 8
    _FM_DIMENSIONS = 4
    _GLOBAL_TRANSFORM_CACHE: dict[
        str,
        tuple[
            np.ndarray,
            tuple[int, ...],
            str,
            str,
            Mapping[str, float],
        ],
    ] = {}

    def __init__(self, *, draw_count: int = 64, draw_seed: int = 20260728) -> None:
        self.draw_count = int(max(8, draw_count))
        self.draw_seed = int(draw_seed)
        self._ontology = load_champion_ontology()
        self._family_by_id = {family.family_id: family for family in self.FAMILY_REGISTRY}
        self._bucket_cache: dict[tuple[str, str, str, bool], tuple[str, str]] = {}
        self._feature_index_cache: dict[str, set[str]] = {}
        self._transform_cache = self._GLOBAL_TRANSFORM_CACHE

    def fit(
        self,
        rows: Sequence[DraftCompositionRow],
        *,
        family_id: str = "residual-full",
        _skip_stability_check: bool = False,
    ) -> DraftInteractionFit:
        family = self._family_by_id.get(family_id)
        if family is None:
            raise DraftInteractionError(f"unknown family: {family_id}")

        canonical_rows = tuple(self._coerce_rows(rows))
        (
            design,
            term_meta,
            term_ids,
            raw_term_ids,
            transform,
            transform_sha256,
            reference_sha256,
            reference_scores,
        ) = self._build_design_bundle(canonical_rows, family)
        if design.size == 0 or design.shape[1] == 0:
            raise DraftInteractionError("no engineered features for requested family")

        y = np.array([float(row.label) for row in canonical_rows], dtype=float)
        prior_means, prior_vars = self._build_prior(canonical_rows, term_meta, family)
        coefficients = self._fit_logistic_ridge(design, y, prior_means, prior_vars)
        if coefficients.size != design.shape[1]:
            raise DraftInteractionError("internal fit coefficient mismatch")

        fit_terms = dict(zip(term_ids, map(float, coefficients)))
        covariance, covariance_factor, posterior_max_correlation = self._approximate_covariance(
            design, coefficients, prior_vars
        )
        diagnostics = self._build_diagnostics(
            canonical_rows,
            design,
            term_meta,
            term_ids,
            coefficients,
            family,
            posterior_max_correlation=posterior_max_correlation,
            _skip_stability_check=_skip_stability_check,
        )
        diagnostics = replace(
            diagnostics,
            orthogonality={
                **dict(diagnostics.orthogonality),
                **dict(reference_scores),
            },
        )
        decomposition_mode = "identified" if diagnostics.identification_status == "identified" else "total_only"
        identification_proof = (
            canonical_sha256(
                {
                    "diagnostics": diagnostics.as_payload,
                    "transform_sha256": transform_sha256,
                    "reference_sha256": reference_sha256,
                }
            )
            if decomposition_mode == "identified"
            else None
        )

        feature_terms = tuple(term_ids)
        row_signature = canonical_sha256([row.payload_sha256 for row in canonical_rows])
        fit = DraftInteractionFit(
            family_id=family_id,
            coefficients=fit_terms,
            feature_terms=feature_terms,
            raw_feature_terms=raw_term_ids,
            transform_matrix=tuple(
                tuple(float(value) for value in row) for row in transform
            ),
            transform_sha256=transform_sha256,
            reference_sha256=reference_sha256,
            identification_proof_sha256=identification_proof,
            term_metadata=tuple(
                (term, dict(meta) | {"term_id": term}) for term, meta in zip(feature_terms, term_meta)
            ),
            orthogonalization=self._orthogonalization_summary(term_meta, design, feature_terms),
            diagnostics=diagnostics,
            supports=self._compute_support(design),
            decomposition_mode=decomposition_mode,
            covariance_diag=tuple(float(v) for v in covariance),
            covariance_factor=tuple(
                tuple(float(value) for value in row)
                for row in covariance_factor
            ),
            covariance_seed=int(
                self.draw_seed + _stable_seed(family_id) + len(canonical_rows) + diagnostics.fit_rank
            ),
            draw_count=self.draw_count,
            selection_tag=f"l6:{family_id}:{row_signature}:{len(canonical_rows)}",
            raw_rows=canonical_rows,
        )

        self._feature_index_cache[family_id] = set(fit.feature_terms)
        return fit

    def fit_all_families(self, rows: Sequence[DraftCompositionRow]) -> dict[str, DraftInteractionFit]:
        rows = self._coerce_rows(rows)
        out: dict[str, DraftInteractionFit] = {}
        for family in self.FAMILY_REGISTRY:
            out[family.family_id] = self.fit(rows, family_id=family.family_id)
        return out

    def predict(self, fit: DraftInteractionFit, row: DraftCompositionRow | Mapping[str, Any]) -> DraftInteractionPrediction:
        row = row if isinstance(row, DraftCompositionRow) else DraftCompositionRow.from_payload(row)
        DraftCompositionRow.validate(row)
        if fit is None:
            raise DraftInteractionError("fit is required")

        family = self._family_by_id.get(fit.family_id)
        if family is None:
            raise DraftInteractionError(f"unknown family in fit payload: {fit.family_id}")

        term_terms, raw_term_values = self._row_features(row, family)
        coefficients = fit.coefficients
        raw_index = {
            term: idx for idx, term in enumerate(fit.raw_feature_terms)
        }
        raw_vectors = {
            key: np.zeros(len(fit.raw_feature_terms), dtype=float)
            for key in ("value", "a_value", "b_value", "neutral_value")
        }
        for term, value in zip(term_terms, raw_term_values):
            idx = raw_index.get(term)
            if idx is None:
                continue
            for key in raw_vectors:
                raw_vectors[key][idx] = _safe_float(
                    value.get(key, 0.0), field_name=f"feature {key}"
                )
        transform = np.asarray(fit.transform_matrix, dtype=float)
        transformed_vectors = {
            key: np.einsum("i,ij->j", vector, transform)
            for key, vector in raw_vectors.items()
        }
        row_vector = transformed_vectors["value"]
        metadata_by_term = {term: dict(meta) for term, meta in fit.term_metadata}
        term_values = tuple(
            {
                "term_id": term,
                **metadata_by_term[term],
                **{
                    key: float(transformed_vectors[key][index])
                    for key in transformed_vectors
                },
            }
            for index, term in enumerate(fit.feature_terms)
        )

        coeff_vector = np.array([coefficients.get(term, 0.0) for term in fit.feature_terms], dtype=float)
        raw_logit = float(np.dot(row_vector, coeff_vector))
        raw_probability = float(_sigmoid(raw_logit))

        identified = self._identification_proven(fit)
        if not identified:
            return DraftInteractionPrediction(
                row_id=row.row_id,
                raw_logit=raw_logit,
                raw_probability=raw_probability,
                lower_95=None,
                upper_95=None,
                decomposition_mode="total_only",
                ledger={
                    "status": "unavailable",
                    "reason": "identification_not_available",
                    "family_id": fit.family_id,
                    "row_id": row.row_id,
                },
            )

        draws = self._predictive_draws(
            row_vector,
            coeff_vector,
            fit.covariance_diag,
            covariance_factor=fit.covariance_factor,
            draw_count=fit.draw_count,
            covariance_seed=fit.covariance_seed,
        )
        lower, upper = self._interval_from_draws(draws)
        ledger = self._reconcile_ledger(row, fit, term_values)
        return DraftInteractionPrediction(
            row_id=row.row_id,
            raw_logit=raw_logit,
            raw_probability=raw_probability,
            lower_95=lower,
            upper_95=upper,
            decomposition_mode="identified",
            ledger=ledger,
        )

    def _identification_proven(self, fit: DraftInteractionFit) -> bool:
        if (
            fit.diagnostics.identification_status != "identified"
            or fit.identification_proof_sha256 is None
        ):
            return False
        expected = canonical_sha256(
            {
                "diagnostics": fit.diagnostics.as_payload,
                "transform_sha256": fit.transform_sha256,
                "reference_sha256": fit.reference_sha256,
            }
        )
        return expected == fit.identification_proof_sha256

    def _coerce_rows(self, rows: Sequence[DraftCompositionRow]) -> tuple[DraftCompositionRow, ...]:
        out: list[DraftCompositionRow] = []
        for row in rows:
            if isinstance(row, DraftCompositionRow):
                DraftCompositionRow.validate(row)
                out.append(row)
                continue
            if isinstance(row, Mapping):
                out.append(DraftCompositionRow.from_payload(row))
                continue
            raise DraftInteractionError("rows must be DraftCompositionRow or mapping payloads")

        if not out:
            raise DraftInteractionError("no rows provided")
        return tuple(out)

    def _build_design_matrix(
        self,
        rows: tuple[DraftCompositionRow, ...],
        family: DraftInteractionFamily,
    ) -> tuple[np.ndarray, tuple[Mapping[str, Any], ...], tuple[str, ...]]:
        design, metadata, terms, *_ = self._build_design_bundle(rows, family)
        return design, metadata, terms

    def _build_design_bundle(
        self,
        rows: tuple[DraftCompositionRow, ...],
        family: DraftInteractionFamily,
    ) -> tuple[
        np.ndarray,
        tuple[Mapping[str, Any], ...],
        tuple[str, ...],
        tuple[str, ...],
        np.ndarray,
        str,
        str,
        Mapping[str, float],
    ]:
        row_terms = [self._row_terms(row, family) for row in rows]

        term_index: dict[str, int] = {}
        term_meta: list[Mapping[str, Any]] = []
        for contributions in row_terms:
            for contribution in contributions:
                if contribution.term_id in term_index:
                    continue
                term_index[contribution.term_id] = len(term_meta)
                term_meta.append(dict(contribution.metadata) | {"term_id": contribution.term_id, "block": contribution.block})

        term_ids = tuple(term_index.keys())
        if not term_ids:
            raise DraftInteractionError("no raw legal interaction terms")

        design = np.zeros((len(rows), len(term_ids)), dtype=float)
        for row_idx, contributions in enumerate(row_terms):
            for contribution in contributions:
                col = term_index[contribution.term_id]
                design[row_idx, col] += contribution.value

        (
            transform,
            supported,
            transform_sha256,
            reference_sha256,
            reference_scores,
        ) = self._reference_transform(term_ids, tuple(term_meta), family)
        projected = np.einsum("ij,jk->ik", design, transform)
        projected_meta = tuple(term_meta[index] for index in supported)
        projected_ids = tuple(term_ids[index] for index in supported)
        return (
            projected,
            projected_meta,
            projected_ids,
            term_ids,
            transform,
            transform_sha256,
            reference_sha256,
            reference_scores,
        )

    def _project_blockwise(self, matrix: np.ndarray, term_meta: tuple[Mapping[str, Any], ...]) -> np.ndarray:
        del term_meta
        return np.asarray(matrix, dtype=float).copy()

    def _legal_reference_rows(self) -> tuple[DraftCompositionRow, ...]:
        champions = (
            "riot:champion:115",
            "riot:champion:101",
            "riot:champion:161",
            "riot:champion:266",
            "riot:champion:114",
            "riot:champion:99911",
            "riot:champion:99912",
            "riot:champion:99913",
            "riot:champion:99914",
            "riot:champion:99915",
            *(f"riot:champion:{88001 + index}" for index in range(20)),
        )
        contexts = (
            ("LEC", "26.14", "emea"),
            ("LEC", "26.15", "emea"),
            ("LEC", "26.16", "emea"),
            ("LCS", "26.14", "americas"),
            ("LCS", "26.15", "americas"),
            ("LCS", "26.16", "americas"),
        )
        rows: list[DraftCompositionRow] = []
        for base_index in range(400):
            ordered = tuple(
                sorted(
                    champions,
                    key=lambda champion: canonical_sha256(
                        {
                            "reference_index": base_index,
                            "champion_id": champion,
                        }
                    ),
                )[:10]
            )
            league, patch, scope = contexts[base_index % len(contexts)]
            sides = (ordered, ordered[5:] + ordered[:5])
            for swap_index, assignment in enumerate(sides):
                rows.append(
                    DraftCompositionRow(
                    row_id=f"legal-reference-{base_index:02d}-{swap_index}",
                    patch_id=patch,
                    league_id=league,
                    side_a=tuple(zip(CANONICAL_ROLE_LIST, assignment[:5])),
                    side_b=tuple(zip(CANONICAL_ROLE_LIST, assignment[5:])),
                    label=0,
                    source_id="frozen-legal-reference",
                    source_patch_pool="26.x",
                    competition_scope_id=scope,
                    metadata={"reference_only": True, "weight": 1.0 / 800.0},
                    )
                )
        return tuple(rows)

    def legal_reference_distribution(self) -> dict[str, Any]:
        rows = self._legal_reference_rows()
        if not rows:
            raise DraftInteractionError("legal reference distribution is empty")
        tolerance = 1e-12
        weight = 1.0 / float(len(rows))
        payload_rows: list[dict[str, Any]] = []
        for ordinal, row in enumerate(rows):
            identity = {
                "side_a": list(row.side_a),
                "side_b": list(row.side_b),
                "league_id": row.league_id,
                "patch_id": row.patch_id,
                "competition_scope_id": row.competition_scope_id,
            }
            payload_rows.append(
                {
                    "ordinal": ordinal,
                    "row_id": canonical_sha256(identity),
                    **identity,
                    "weight": weight,
                }
            )
        body = {
            "id": "l6-finite-legal-composition-reference-v2",
            "operation": "weighted_block_residualization",
            "normalization": "uniform_positive_weights_sum_to_one",
            "normalization_tolerance": tolerance,
            "learned_from_evaluation_rows": False,
            "row_count": len(payload_rows),
            "rows": payload_rows,
        }
        distribution = {**body, "sha256": canonical_sha256(body)}
        self.validate_legal_reference_distribution(distribution)
        return distribution

    @staticmethod
    def validate_legal_reference_distribution(
        distribution: Mapping[str, Any],
    ) -> str:
        required = {
            "id",
            "operation",
            "normalization",
            "normalization_tolerance",
            "learned_from_evaluation_rows",
            "row_count",
            "rows",
            "sha256",
        }
        if set(distribution) != required:
            raise DraftInteractionError("legal reference fields are not canonical")
        if distribution["id"] != "l6-finite-legal-composition-reference-v2":
            raise DraftInteractionError("legal reference id is not canonical")
        if distribution["operation"] != "weighted_block_residualization":
            raise DraftInteractionError("legal reference operation is not canonical")
        if distribution["normalization"] != "uniform_positive_weights_sum_to_one":
            raise DraftInteractionError("legal reference normalization is not canonical")
        if distribution["learned_from_evaluation_rows"] is not False:
            raise DraftInteractionError("legal reference may not use evaluation outcomes")
        tolerance = float(distribution["normalization_tolerance"])
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise DraftInteractionError("legal reference tolerance must be finite and positive")
        rows = distribution["rows"]
        if not isinstance(rows, list) or len(rows) != int(distribution["row_count"]):
            raise DraftInteractionError("legal reference row count mismatch")
        if not rows:
            raise DraftInteractionError("legal reference distribution is empty")
        expected_weight = 1.0 / float(len(rows))
        row_ids: list[str] = []
        weights: list[float] = []
        for ordinal, item in enumerate(rows):
            if not isinstance(item, Mapping) or item.get("ordinal") != ordinal:
                raise DraftInteractionError("legal reference row order is not canonical")
            identity = {
                "side_a": item.get("side_a"),
                "side_b": item.get("side_b"),
                "league_id": item.get("league_id"),
                "patch_id": item.get("patch_id"),
                "competition_scope_id": item.get("competition_scope_id"),
            }
            expected_row_id = canonical_sha256(identity)
            if item.get("row_id") != expected_row_id:
                raise DraftInteractionError("legal reference row id mismatch")
            row_ids.append(expected_row_id)
            value = float(item.get("weight", float("nan")))
            if not np.isfinite(value) or value <= 0.0:
                raise DraftInteractionError("legal reference weights must be finite and positive")
            if abs(value - expected_weight) > tolerance:
                raise DraftInteractionError("legal reference weights are not uniformly normalized")
            weights.append(value)
        if len(set(row_ids)) != len(row_ids):
            raise DraftInteractionError("legal reference row ids must be unique")
        if abs(float(sum(weights)) - 1.0) > tolerance:
            raise DraftInteractionError("legal reference weights must sum to one")
        body = {key: distribution[key] for key in required if key != "sha256"}
        digest = canonical_sha256(body)
        if distribution["sha256"] != digest:
            raise DraftInteractionError("legal reference digest mismatch")
        return digest

    def _reference_transform(
        self,
        term_ids: tuple[str, ...],
        term_meta: tuple[Mapping[str, Any], ...],
        family: DraftInteractionFamily,
    ) -> tuple[
        np.ndarray,
        tuple[int, ...],
        str,
        str,
        Mapping[str, float],
    ]:
        reference_rows = self._legal_reference_rows()
        reference_distribution = self.legal_reference_distribution()
        reference_sha256 = self.validate_legal_reference_distribution(
            reference_distribution
        )
        cache_key = canonical_sha256(
            {
                "family": family.family_id,
                "terms": term_ids,
                "reference": reference_sha256,
            }
        )
        cached = self._transform_cache.get(cache_key)
        if cached is not None:
            return cached
        raw_reference = self._design_for_terms(
            reference_rows, family, term_ids
        )
        weights = np.asarray(
            [item["weight"] for item in reference_distribution["rows"]],
            dtype=float,
        )
        weighted_mean = np.einsum("n,ni->i", weights, raw_reference)
        centered_reference = raw_reference - weighted_mean[None, :]
        transform = np.eye(len(term_ids), dtype=float)
        supported: list[int] = []
        block_to_cols: dict[str, list[int]] = {}
        for index, metadata in enumerate(term_meta):
            block_to_cols.setdefault(str(metadata["block"]), []).append(index)
        reference_scores: dict[str, float] = {}
        for block in self._BLOCK_ORDER:
            columns = block_to_cols.get(block, [])
            if not columns:
                continue
            target = centered_reference[:, columns]
            if supported:
                lower = np.einsum(
                    "ij,jk->ik",
                    centered_reference,
                    transform[:, supported],
                )
                weighted_lower = lower * np.sqrt(weights)[:, None]
                weighted_target = target * np.sqrt(weights)[:, None]
                projection = np.linalg.lstsq(
                    weighted_lower, weighted_target, rcond=1e-10
                )[0]
                transform[:, columns] = (
                    np.eye(len(term_ids))[:, columns]
                    - np.einsum(
                        "ij,jk->ik",
                        transform[:, supported],
                        projection,
                    )
                )
            residual = np.einsum(
                "ij,jk->ik", centered_reference, transform[:, columns]
            )
            norms = np.sqrt(
                np.einsum("n,ni,ni->i", weights, residual, residual)
            )
            retained = [
                column
                for column, norm in zip(columns, norms)
                if float(norm) > 1e-9
            ]
            if supported and retained:
                lower = np.einsum(
                    "ij,jk->ik",
                    centered_reference,
                    transform[:, supported],
                )
                cross = np.einsum(
                    "n,ni,nj->ij",
                    weights,
                    lower,
                    np.einsum(
                        "ij,jk->ik",
                        centered_reference,
                        transform[:, retained],
                    ),
                )
                reference_scores[f"reference_{block}_crossmax"] = float(
                    np.max(np.abs(cross))
                )
            else:
                reference_scores[f"reference_{block}_crossmax"] = 0.0
            supported.extend(retained)
        if not supported:
            raise DraftInteractionError("legal reference supports no residual terms")
        reduced_transform = transform[:, supported]
        transform_payload = {
            "reference_sha256": reference_sha256,
            "raw_terms": term_ids,
            "supported_indices": supported,
            "matrix": reduced_transform.tolist(),
            "tolerance": 1e-9,
        }
        result = (
            reduced_transform,
            tuple(supported),
            canonical_sha256(transform_payload),
            reference_sha256,
            reference_scores,
        )
        self._transform_cache[cache_key] = result
        return result

    def _row_terms(self, row: DraftCompositionRow, family: DraftInteractionFamily) -> list[_TermContribution]:
        terms: list[_TermContribution] = []
        league = row.league_id
        patch = row.patch_id
        competition_scope = row.competition_scope_id or "unscoped"

        def _make_main_terms(side_tag: str, role: str, champion: str, side_sign: float) -> None:
            role_key = str(role)
            champion_key = str(champion)
            base_metadata = {
                "league_id": league,
                "patch_id": patch,
                "role": role_key,
                "champion_id": champion_key,
                "side": side_tag,
                "competition_scope_id": competition_scope,
            }
            terms.append(
                _TermContribution(
                    term_id=f"main.scope|scope={competition_scope}|role={role_key}|champ={champion_key}",
                    block="main_scope",
                    value=side_sign,
                    metadata={
                        **base_metadata,
                        "type": "main",
                        "scope": "competition_scope",
                    },
                )
            )
            terms.append(
                _TermContribution(
                    term_id=f"main.global|champ={champion_key}",
                    block="main_global",
                    value=side_sign,
                    metadata={**base_metadata, "type": "main", "scope": "global"},
                )
            )
            terms.append(
                _TermContribution(
                    term_id=f"main.role|role={role_key}|champ={champion_key}",
                    block="main_role",
                    value=side_sign,
                    metadata={**base_metadata, "type": "main", "scope": "role"},
                )
            )
            terms.append(
                _TermContribution(
                    term_id=f"main.league|league={league}|role={role_key}|champ={champion_key}",
                    block="main_league",
                    value=side_sign,
                    metadata={**base_metadata, "type": "main", "scope": "league"},
                )
            )
            terms.append(
                _TermContribution(
                    term_id=f"main.patch|league={league}|patch={patch}|role={role_key}|champ={champion_key}",
                    block="main_patch",
                    value=side_sign,
                    metadata={**base_metadata, "type": "main", "scope": "patch"},
                )
            )

        # Side-local effects and same-side pairs.
        for side_tag, side_payload in (("A", row.side_a), ("B", row.side_b)):
            side_sign = 1.0 if side_tag == "A" else -1.0
            side_roles = tuple(side_payload)
            team_signature = _role_pair_signature(
                [role for role, _ in side_roles],
                [champion for _, champion in side_roles],
            )

            for role, champion in side_roles:
                _make_main_terms(side_tag, role, champion, side_sign)

            if family.include_ally_exact or family.include_ally_sparse or family.include_factorization:
                for (role_a, champ_a), (role_b, champ_b) in combinations(side_roles, 2):
                    (ordered_role_a, ordered_champ_a), (ordered_role_b, ordered_champ_b), _orientation = _ordered_pair(
                        role_a,
                        champ_a,
                        role_b,
                        champ_b,
                    )
                    common_meta = {
                        "roles": [ordered_role_a, ordered_role_b],
                        "champions": [ordered_champ_a, ordered_champ_b],
                        "league_id": league,
                        "patch_id": patch,
                        "side": side_tag,
                        "scope": "ally",
                        "type": "ally",
                    }
                    ally_pair_signature = _role_pair_signature(
                        [ordered_role_a, ordered_role_b],
                        [ordered_champ_a, ordered_champ_b],
                    )

                    if family.include_ally_exact:
                        terms.extend(
                            (
                                _TermContribution(
                                    term_id=f"ally.exact|scope=global|roles={ally_pair_signature}",
                                    block="ally_exact",
                                    value=side_sign,
                                    metadata={**common_meta, "patch_fallback": "global", "scope": "global"},
                                ),
                                _TermContribution(
                                    term_id=f"ally.exact|scope=league|league={league}|roles={ally_pair_signature}",
                                    block="ally_exact",
                                    value=side_sign,
                                    metadata={**common_meta, "patch_fallback": "league", "scope": "league"},
                                ),
                                _TermContribution(
                                    term_id=f"ally.exact|scope=patch|league={league}|patch={patch}|roles={ally_pair_signature}",
                                    block="ally_exact",
                                    value=side_sign,
                                    metadata={**common_meta, "patch_fallback": "exact", "scope": "patch"},
                                ),
                            )
                        )

                    if family.include_ally_sparse and family.use_archetype_transfer:
                        bucket_a, _, reason_a = self._archetype_bucket(
                            champion_id=champ_a,
                            role=role_a,
                            patch_id=patch,
                            allow_unknown=family.use_archetype_transfer,
                        )
                        bucket_b, _, reason_b = self._archetype_bucket(
                            champion_id=champ_b,
                            role=role_b,
                            patch_id=patch,
                            allow_unknown=family.use_archetype_transfer,
                        )
                        sparse_base_meta = {
                            **common_meta,
                            "buckets": [bucket_a, bucket_b],
                            "bucket_reasons": [reason_a, reason_b],
                        }
                        terms.extend(
                            (
                                _TermContribution(
                                    term_id=f"ally.sparse|scope=global|roles={ally_pair_signature}|buckets={bucket_a}|{bucket_b}",
                                    block="ally_sparse",
                                    value=side_sign,
                                    metadata={**sparse_base_meta, "patch_fallback": "global", "scope": "global"},
                                ),
                                _TermContribution(
                                    term_id=f"ally.sparse|scope=competition_scope|competition_scope={competition_scope}|roles={ally_pair_signature}|buckets={bucket_a}|{bucket_b}",
                                    block="ally_sparse",
                                    value=side_sign,
                                    metadata={
                                        **sparse_base_meta,
                                        "competition_scope_id": competition_scope,
                                        "patch_fallback": "competition_scope",
                                        "scope": "competition_scope",
                                    },
                                ),
                                _TermContribution(
                                    term_id=f"ally.sparse|scope=league|league={league}|roles={ally_pair_signature}|buckets={bucket_a}|{bucket_b}",
                                    block="ally_sparse",
                                    value=side_sign,
                                    metadata={**sparse_base_meta, "patch_fallback": "league", "scope": "league"},
                                ),
                                _TermContribution(
                                    term_id=f"ally.sparse|scope=patch|league={league}|patch={patch}|roles={ally_pair_signature}|buckets={bucket_a}|{bucket_b}",
                                    block="ally_sparse",
                                    value=side_sign,
                                    metadata={**sparse_base_meta, "patch_fallback": "exact", "scope": "patch"},
                                ),
                            )
                        )

                    if family.include_factorization:
                        profile_a = self._latent_profile(champ_a, role_a, patch)
                        profile_b = self._latent_profile(champ_b, role_b, patch)
                        for dim, (va, vb) in enumerate(zip(profile_a, profile_b)):
                            terms.append(
                                _TermContribution(
                                    term_id=f"fm_sparse|dim={dim}|league={league}|patch={patch}|roles={ally_pair_signature}",
                                    block="fm_sparse",
                                    value=side_sign * float(va) * float(vb),
                                    metadata={**common_meta, "dimension": dim, "patch_fallback": "exact", "scope": "ally"},
                                )
                            )

            if family.include_whole_team:
                terms.extend(
                    (
                        _TermContribution(
                            term_id=f"whole_team|scope=global|team={team_signature}",
                            block="whole_team",
                            value=side_sign,
                            metadata={
                                "block": "whole_team",
                                "scope": "global",
                                "type": "whole_team",
                                "side": side_tag,
                                "league_id": league,
                                "patch_id": patch,
                            },
                        ),
                        _TermContribution(
                            term_id=f"whole_team|scope=league|league={league}|team={team_signature}",
                            block="whole_team",
                            value=side_sign,
                            metadata={
                                "block": "whole_team",
                                "scope": "league",
                                "type": "whole_team",
                                "side": side_tag,
                                "league_id": league,
                                "patch_id": patch,
                            },
                        ),
                        _TermContribution(
                            term_id=f"whole_team|scope=patch|league={league}|patch={patch}|team={team_signature}",
                            block="whole_team",
                            value=side_sign,
                            metadata={
                                "block": "whole_team",
                                "scope": "patch",
                                "type": "whole_team",
                                "side": side_tag,
                                "league_id": league,
                                "patch_id": patch,
                            },
                        ),
                    )
                )

        if family.include_cross_team:
            side_a_signature = _role_pair_signature(
                [role for role, _ in row.side_a],
                [champion for _, champion in row.side_a],
            )
            side_b_signature = _role_pair_signature(
                [role for role, _ in row.side_b],
                [champion for _, champion in row.side_b],
            )
            left_team, right_team = sorted((side_a_signature, side_b_signature))
            orientation = 1.0 if side_a_signature == left_team else -1.0
            orientation_side = "A" if orientation > 0 else "B"
            cross_metadata = {
                "block": "cross_team",
                "type": "cross_team",
                "side": orientation_side,
                "league_id": league,
                "patch_id": patch,
                "teams": [left_team, right_team],
            }
            terms.extend(
                (
                    _TermContribution(
                        term_id=f"cross_team|scope=global|teams={left_team}|{right_team}",
                        block="cross_team",
                        value=orientation,
                        metadata={**cross_metadata, "scope": "global"},
                    ),
                    _TermContribution(
                        term_id=f"cross_team|scope=league|league={league}|teams={left_team}|{right_team}",
                        block="cross_team",
                        value=orientation,
                        metadata={**cross_metadata, "scope": "league"},
                    ),
                    _TermContribution(
                        term_id=f"cross_team|scope=patch|league={league}|patch={patch}|teams={left_team}|{right_team}",
                        block="cross_team",
                        value=orientation,
                        metadata={**cross_metadata, "scope": "patch"},
                    ),
                )
            )

        # Cross-team: exact 25 role-pair allocations per row.
        if family.include_enemy_exact or family.include_enemy_sparse:
            for role_a, champ_a in row.side_a:
                for role_b, champ_b in row.side_b:
                    (
                        (left_role, left_champ, left_side),
                        (right_role, right_champ, right_side),
                        _orientation,
                    ) = _ordered_cross_pair(role_a, champ_a, "A", role_b, champ_b, "B")
                    ordered_roles = [left_role, right_role]
                    ordered_champs = [left_champ, right_champ]
                    ordered_value = 1.0 if left_side == "A" else -1.0
                    pair_meta = {
                        "roles": ordered_roles,
                        "champions": ordered_champs,
                        "league_id": league,
                        "patch_id": patch,
                        "side": left_side,
                        "type": "enemy",
                    }
                    enemy_pair_signature = _role_pair_signature(ordered_roles, ordered_champs)

                    if family.include_enemy_exact:
                        terms.extend(
                            (
                                _TermContribution(
                                    term_id=f"enemy.exact|scope=global|roles={enemy_pair_signature}",
                                    block="enemy_exact",
                                    value=ordered_value,
                                    metadata={**pair_meta, "scope": "global", "patch_fallback": "global"},
                                ),
                                _TermContribution(
                                    term_id=f"enemy.exact|scope=league|league={league}|roles={enemy_pair_signature}",
                                    block="enemy_exact",
                                    value=ordered_value,
                                    metadata={**pair_meta, "scope": "league", "patch_fallback": "league"},
                                ),
                                _TermContribution(
                                    term_id=f"enemy.exact|scope=patch|league={league}|patch={patch}|roles={enemy_pair_signature}",
                                    block="enemy_exact",
                                    value=ordered_value,
                                    metadata={**pair_meta, "scope": "patch", "patch_fallback": "exact"},
                                ),
                            )
                        )

                    if family.include_enemy_sparse and family.use_archetype_transfer:
                        bucket_a, _, reason_a = self._archetype_bucket(
                            champion_id=champ_a,
                            role=role_a,
                            patch_id=patch,
                            allow_unknown=family.use_archetype_transfer,
                        )
                        bucket_b, _, reason_b = self._archetype_bucket(
                            champion_id=champ_b,
                            role=role_b,
                            patch_id=patch,
                            allow_unknown=family.use_archetype_transfer,
                        )
                        bucket_by_assignment = {
                            (role_a, champ_a): (bucket_a, reason_a),
                            (role_b, champ_b): (bucket_b, reason_b),
                        }
                        ordered_bucket_a, ordered_reason_a = bucket_by_assignment[
                            (ordered_roles[0], ordered_champs[0])
                        ]
                        ordered_bucket_b, ordered_reason_b = bucket_by_assignment[
                            (ordered_roles[1], ordered_champs[1])
                        ]
                        sparse_meta = {
                            **pair_meta,
                            "buckets": [ordered_bucket_a, ordered_bucket_b],
                            "bucket_reasons": [
                                ordered_reason_a,
                                ordered_reason_b,
                            ],
                        }
                        terms.extend(
                            (
                                _TermContribution(
                                    term_id=f"enemy.sparse|scope=global|roles={enemy_pair_signature}|buckets={bucket_a}|{bucket_b}",
                                    block="enemy_sparse",
                                    value=ordered_value,
                                    metadata={**sparse_meta, "scope": "global", "patch_fallback": "global"},
                                ),
                                _TermContribution(
                                    term_id=f"enemy.sparse|scope=competition_scope|competition_scope={competition_scope}|roles={enemy_pair_signature}|buckets={ordered_bucket_a}|{ordered_bucket_b}",
                                    block="enemy_sparse",
                                    value=ordered_value,
                                    metadata={
                                        **sparse_meta,
                                        "competition_scope_id": competition_scope,
                                        "scope": "competition_scope",
                                        "patch_fallback": "competition_scope",
                                    },
                                ),
                                _TermContribution(
                                    term_id=f"enemy.sparse|scope=league|league={league}|roles={enemy_pair_signature}|buckets={bucket_a}|{bucket_b}",
                                    block="enemy_sparse",
                                    value=ordered_value,
                                    metadata={**sparse_meta, "scope": "league", "patch_fallback": "league"},
                                ),
                                _TermContribution(
                                    term_id=f"enemy.sparse|scope=patch|league={league}|patch={patch}|roles={enemy_pair_signature}|buckets={bucket_a}|{bucket_b}",
                                    block="enemy_sparse",
                                    value=ordered_value,
                                    metadata={**sparse_meta, "scope": "patch", "patch_fallback": "exact"},
                                ),
                            )
                        )

        canonical_terms: list[_TermContribution] = []
        for term in terms:
            metadata = dict(term.metadata)
            if term.block == "whole_team" and "team" not in metadata:
                side_payload = row.side_a if metadata.get("side") == "A" else row.side_b
                metadata["team"] = [
                    {"role": role, "champion_id": champion}
                    for role, champion in side_payload
                ]
            canonical_term = _TermContribution(
                term_id="",
                block=term.block,
                value=term.value,
                metadata=metadata,
            )
            canonical_terms.append(
                _TermContribution(
                    term_id=_structured_term_id(canonical_term),
                    block=term.block,
                    value=term.value,
                    metadata=metadata,
                )
            )
        return canonical_terms

    def _archetype_bucket(
        self,
        champion_id: str,
        role: str,
        patch_id: str,
        *,
        allow_unknown: bool,
    ) -> tuple[str, float, str]:
        if not isinstance(champion_id, str) or not champion_id:
            return "archetype::wide::invalid_champion", 0.0, "invalid_champion"

        cache_key = (champion_id, role, patch_id, allow_unknown)
        cached = self._bucket_cache.get(cache_key)
        if cached is not None:
            return cached[0], 1.0, cached[1]

        try:
            prior = self._ontology.build_archetype_prior(
                champion_id=champion_id,
                role=role,
                patch_id=patch_id,
                league_id=None,
            )
            vector = np.asarray(prior.get("vector", []), dtype=float)
            fallback_level = str(prior.get("fallback_level") or "exact")
            if not allow_unknown and fallback_level != "none":
                raise DraftInteractionError(
                    f"archetype transfer disabled for fallback level={fallback_level} champion={champion_id} role={role}"
                )
            residual_info = prior.get("residual") or {}
            sigma = _safe_float(residual_info.get("sigma", 1.0), field_name="residual sigma")
            quantized = tuple(
                int(v >= 0.5) for v in np.clip(vector, 0.0, 1.0)
            )
            bucket = f"archetype|{role}|{','.join(str(item) for item in quantized[:6]) or 'wide'}"
            confidence = float(max(0.08, min(0.95, 1.0 - 0.7 * sigma)))
            reason = str(fallback_level or "exact")
            if reason and reason != "exact":
                confidence *= 0.75
            self._bucket_cache[cache_key] = (bucket, reason)
            return bucket, confidence, reason
        except Exception as exc:  # noqa: BLE001
            if not allow_unknown:
                raise DraftInteractionError("archetype fallback disabled") from exc
            fallback = self._nearest_archetype_bucket(champion_id=champion_id, role=role, patch_id=patch_id)
            self._bucket_cache[cache_key] = (fallback, "nearest_unknown")
            return fallback, 0.12, "nearest_unknown"

    def _nearest_archetype_bucket(self, champion_id: str, role: str, patch_id: str) -> str:
        champion_ids = self._ontology.champion_ids()
        if not isinstance(champion_ids, Sequence):
            champion_ids = list(champion_ids)

        query = None
        try:
            prior = self._ontology.build_archetype_prior(
                champion_id=champion_id,
                role=role,
                patch_id=patch_id,
                league_id=None,
            )
            query = np.asarray(prior.get("vector", []), dtype=float)
        except Exception:
            query = None

        if query is None or query.size == 0:
            return f"archetype|{role}|wide"

        def _distance(candidate: str) -> float:
            if candidate == champion_id:
                return float("inf")
            try:
                candidate_prior = self._ontology.build_archetype_prior(
                    champion_id=candidate,
                    role=role,
                    patch_id=patch_id,
                    league_id=None,
                )
                candidate_vector = np.asarray(candidate_prior.get("vector", []), dtype=float)
            except Exception:
                return float("inf")
            if candidate_vector.size != query.size or candidate_vector.size == 0:
                return float("inf")
            return float(np.linalg.norm(candidate_vector - query))

        best_distance = float("inf")
        best_candidate = f"archetype|{role}|wide"
        best_vector: np.ndarray | None = None
        for candidate in champion_ids:
            distance = _distance(candidate)
            if distance == float("inf"):
                continue
            if distance < best_distance:
                best_distance = distance
                try:
                    best_vector = np.asarray(
                        self._ontology.build_archetype_prior(
                            champion_id=candidate,
                            role=role,
                            patch_id=patch_id,
                            league_id=None,
                        ).get("vector", []),
                        dtype=float,
                    )
                except Exception:
                    best_vector = None
        if best_vector is not None and best_vector.size:
            quantized = tuple(int(value >= 0.5) for value in best_vector[:6])
            best_candidate = (
                f"archetype|{role}|"
                f"{','.join(str(item) for item in quantized) or 'wide'}"
            )

        return best_candidate

    def _latent_profile(self, champion_id: str, role: str, patch_id: str, dim: int | None = None) -> tuple[float, ...]:
        dim = self._FM_DIMENSIONS if dim is None else int(dim)
        bucket, _, _ = self._archetype_bucket(
            champion_id=champion_id,
            role=role,
            patch_id=patch_id,
            allow_unknown=True,
        )
        tokens = [segment for segment in bucket.split("|") if segment]
        values = np.asarray(
            [
                int(hashlib.sha256(part.encode("utf-8")).hexdigest()[:8], 16) / float(0xFFFFFFFF)
                for part in tokens
            ],
            dtype=float,
        )
        if values.size == 0:
            values = np.array([0.33, 0.27, 0.21, 0.19], dtype=float)
        values = values[:dim]
        if values.size < dim:
            values = np.resize(values, dim)
        norm = float(np.linalg.norm(values))
        if norm <= 0:
            values = np.full(dim, 0.5, dtype=float)
            norm = float(np.linalg.norm(values))
        values = values / norm
        return tuple(float(v) for v in values)

    def _build_prior(
        self,
        rows: tuple[DraftCompositionRow, ...],
        term_meta: tuple[Mapping[str, Any], ...],
        family: DraftInteractionFamily,
    ) -> tuple[np.ndarray, np.ndarray]:
        means: list[float] = []
        variances: list[float] = []
        total_rows = float(len(rows))

        for metadata in term_meta:
            block = str(metadata.get("block", "main_global"))
            base_var = float(self._PRIOR_VARIANCE.get(block, 0.24))
            support = float(self._context_support_count(rows, metadata))
            scale = max(1.0, np.sqrt(total_rows / max(1.0, support)))

            if block in {"ally_sparse", "enemy_sparse", "fm_sparse"} and not family.use_archetype_transfer:
                var = base_var * 1.75 * scale
            else:
                var = base_var * scale

            means.append(0.0)
            variances.append(max(1e-6, var))

        return np.array(means, dtype=float), np.array(variances, dtype=float)

    def _context_support_count(self, rows: tuple[DraftCompositionRow, ...], metadata: Mapping[str, Any]) -> int:
        scope = str(metadata.get("scope", "global"))
        league = str(metadata.get("league_id", ""))
        patch = str(metadata.get("patch_id", ""))
        competition_scope = str(
            metadata.get("competition_scope_id", "")
        )
        roles = list(metadata.get("roles", ()))
        champs = list(metadata.get("champions", ()))

        count = 0
        for row in rows:
            if scope == "league" and row.league_id != league:
                continue
            if (
                scope == "competition_scope"
                and row.competition_scope_id != competition_scope
            ):
                continue
            if scope == "patch" and not (row.league_id == league and row.patch_id == patch):
                continue

            assignments = list(zip(roles, champs))
            side_a = set(row.side_a)
            side_b = set(row.side_b)
            term_type = str(metadata.get("type", ""))
            if assignments:
                if term_type == "ally" and not (
                    all(item in side_a for item in assignments)
                    or all(item in side_b for item in assignments)
                ):
                    continue
                if term_type == "enemy" and len(assignments) == 2 and not (
                    (assignments[0] in side_a and assignments[1] in side_b)
                    or (assignments[0] in side_b and assignments[1] in side_a)
                ):
                    continue
                if term_type not in {"ally", "enemy"} and not all(
                    item in side_a or item in side_b for item in assignments
                ):
                    continue
            role = metadata.get("role")
            champion = metadata.get("champion_id")
            if role is not None and champion is not None and (
                (str(role), str(champion)) not in side_a
                and (str(role), str(champion)) not in side_b
            ):
                continue
            count += 1

        return max(1, count)

    def _fit_logistic_ridge(
        self,
        x: np.ndarray,
        y: np.ndarray,
        prior_means: np.ndarray,
        prior_vars: np.ndarray,
    ) -> np.ndarray:
        beta = np.array(prior_means, dtype=float)
        precision = np.where(prior_vars > 0.0, 1.0 / prior_vars, 1.0)

        if x.size == 0:
            return beta

        for _ in range(200):
            linear = np.clip(np.einsum("ij,j->i", x, beta), -30.0, 30.0)
            p = _sigmoid(linear)
            w = np.clip(p * (1.0 - p), 1e-8, 0.25)
            z = linear + (y - p) / w
            hessian = np.einsum("ni,nj,n->ij", x, x, w) + np.diag(
                precision
            )
            rhs = np.einsum("ni,n->i", x, w * z) + precision * prior_means
            try:
                next_beta = np.linalg.solve(hessian, rhs)
            except np.linalg.LinAlgError:
                next_beta = np.linalg.lstsq(hessian, rhs, rcond=None)[0]

            next_beta = np.asarray(next_beta, dtype=float)
            if not np.isfinite(next_beta).all():
                next_beta = beta
            delta = float(np.max(np.abs(next_beta - beta)))
            beta = 0.65 * beta + 0.35 * next_beta
            if delta < 1e-9:
                break

        return np.clip(beta, -8.0, 8.0)

    def _approximate_covariance(
        self,
        x: np.ndarray,
        beta: np.ndarray,
        prior_vars: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        if x.size == 0:
            return (
                np.ones(beta.shape[0], dtype=float),
                np.zeros((0, beta.shape[0]), dtype=float),
                0.0,
            )
        linear = np.clip(np.einsum("ij,j->i", x, beta), -30.0, 30.0)
        p = _sigmoid(linear)
        w = np.clip(p * (1.0 - p), 1e-8, 0.25)
        safe_prior_vars = np.where(prior_vars > 0.0, prior_vars, 1.0)
        weighted_inverse = np.diag(1.0 / w) + np.einsum(
            "ni,i,mi->nm", x, safe_prior_vars, x
        )
        try:
            cholesky = np.linalg.cholesky(weighted_inverse)
            covariance_factor = np.linalg.solve(
                cholesky, x * safe_prior_vars[None, :]
            )
            solved = np.linalg.solve(
                weighted_inverse, x * safe_prior_vars[None, :]
            )
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(weighted_inverse)
            inverse_root = eigenvectors @ np.diag(
                1.0 / np.sqrt(np.maximum(eigenvalues, 1e-12))
            ) @ eigenvectors.T
            covariance_factor = inverse_root @ (
                x * safe_prior_vars[None, :]
            )
            solved = np.linalg.lstsq(
                weighted_inverse,
                x * safe_prior_vars[None, :],
                rcond=None,
            )[0]
        correction = np.einsum(
            "i,ni,nj,j->ij",
            safe_prior_vars,
            x,
            solved,
            np.ones_like(safe_prior_vars),
        )
        inv = np.diag(safe_prior_vars) - correction
        diag = np.maximum(np.diag(inv), 1e-12)
        scale = np.sqrt(np.outer(diag, diag))
        corr = np.divide(inv, scale, out=np.zeros_like(inv), where=scale > 0.0)
        np.fill_diagonal(corr, 0.0)
        posterior_max_correlation = float(np.max(np.abs(corr))) if corr.size else 0.0
        return safe_prior_vars, covariance_factor, posterior_max_correlation

    def _max_corr(self, matrix: np.ndarray) -> float:
        if matrix.shape[1] < 2:
            return 0.0
        centered = matrix - np.mean(matrix, axis=0, keepdims=True)
        norms = np.linalg.norm(centered, axis=0)
        valid = norms > 1e-12
        if int(np.count_nonzero(valid)) < 2:
            return 0.0
        normalized = centered[:, valid] / norms[valid]
        corr = np.einsum("ni,nj->ij", normalized, normalized)
        upper = np.triu_indices_from(corr, k=1)
        if upper[0].size == 0:
            return 0.0
        return float(np.max(np.abs(corr[upper])))

    def _orthogonality_scores(self, x: np.ndarray, term_meta: tuple[Mapping[str, Any], ...]) -> dict[str, float]:
        if x.size == 0:
            return {}
        block_to_cols: dict[str, list[int]] = {}
        for idx, md in enumerate(term_meta):
            block = str(md.get("block", "unclassified"))
            block_to_cols.setdefault(block, []).append(idx)

        scores: dict[str, float] = {}
        for block in self._BLOCK_ORDER:
            cols = block_to_cols.get(block, [])
            if not cols:
                scores[f"{block}_mean_max"] = 0.0
                continue
            means = np.mean(x[:, cols], axis=0)
            scores[f"{block}_mean_max"] = float(np.max(np.abs(means)))

        for i, left in enumerate(self._BLOCK_ORDER):
            left_cols = block_to_cols.get(left, [])
            if not left_cols:
                continue
            for right in self._BLOCK_ORDER[i + 1 :]:
                right_cols = block_to_cols.get(right, [])
                if not right_cols:
                    continue
                score = float(
                    np.max(
                        np.abs(
                            np.einsum(
                                "ni,nj->ij",
                                x[:, left_cols],
                                x[:, right_cols],
                            )
                        )
                    )
                )
                scores[f"{left}_x_{right}_crossmax"] = score

        return scores

    def _build_diagnostics(
        self,
        rows: tuple[DraftCompositionRow, ...],
        x: np.ndarray,
        term_meta: tuple[Mapping[str, Any], ...],
        term_ids: tuple[str, ...],
        coeff: np.ndarray,
        family: DraftInteractionFamily,
        *,
        posterior_max_correlation: float = 0.0,
        _skip_stability_check: bool,
    ) -> DraftInteractionFitDiagnostics:
        row_count, feature_count = x.shape
        if row_count < 2:
            return DraftInteractionFitDiagnostics(
                family_id=family.family_id,
                feature_count=feature_count,
                row_count=row_count,
                fit_rank=0,
                condition_number=float("inf"),
                identification_status="unidentified",
                orthogonality={},
                fallback_term_count=0,
                fallback_counts={"insufficient_rows": feature_count},
                feature_block_counts=Counter(str(md.get("block", "unclassified")) for md in term_meta),
                draw_count=0,
                covariance_seed=0,
                warnings=("insufficient_rows",),
                collinearity_max_correlation=1.0,
                min_support=0,
            )

        rank = int(np.linalg.matrix_rank(x))
        condition = (
            float("inf")
            if rank < feature_count
            else float(np.linalg.cond(x))
            if x.size
            else float("inf")
        )
        col_corr = self._max_corr(x)
        block_counts: Counter[str] = Counter(str(md.get("block", "unclassified")) for md in term_meta)
        supports = [int(np.count_nonzero(x[:, idx])) for idx in range(feature_count)]
        min_support = int(min(supports)) if supports else 0
        column_counts = Counter(
            np.ascontiguousarray(x[:, idx]).tobytes()
            for idx in range(feature_count)
        )
        duplicate_column_pairs = sum(
            count * (count - 1) // 2 for count in column_counts.values()
        )

        warnings: list[str] = []
        fallback_counts: dict[str, int] = {
            "rank_deficiency": 0,
            "ill_conditioned": 0,
            "collinearity_detected": 0,
            "weak_support": 0,
            "cooccurrence_insufficient": 0,
            "duplicate_columns": 0,
            "posterior_dependence": 0,
            "source_removal_instability": 0,
            "patch_removal_instability": 0,
        }

        if rank < feature_count:
            warnings.append("rank_deficiency")
            fallback_counts["rank_deficiency"] = 1
        if condition > 1e6:
            warnings.append("ill_conditioned")
            fallback_counts["ill_conditioned"] = 1
        if col_corr > 0.99:
            warnings.append("collinearity_detected")
            fallback_counts["collinearity_detected"] = 1
        if min_support < 2:
            warnings.append("weak_support")
            fallback_counts["weak_support"] = 1
            warnings.append("cooccurrence_insufficient")
            fallback_counts["cooccurrence_insufficient"] = 1
        if duplicate_column_pairs:
            warnings.append("duplicate_columns")
            fallback_counts["duplicate_columns"] = int(duplicate_column_pairs)
        if posterior_max_correlation > 0.98:
            warnings.append("posterior_dependence")
            fallback_counts["posterior_dependence"] = int(
                np.ceil(posterior_max_correlation * 100)
            )

        if not _skip_stability_check and row_count >= 4:
            source_delta = self._source_removal_stability(rows, family)
            if source_delta > 0.08:
                warnings.append("source_removal_instability")
                fallback_counts["source_removal_instability"] = int(np.ceil(source_delta * 100))

            patch_delta = self._patch_removal_stability(rows, family)
            if patch_delta > 0.08:
                warnings.append("patch_removal_instability")
                fallback_counts["patch_removal_instability"] = int(np.ceil(patch_delta * 100))

        orthogonality = self._orthogonality_scores(x, term_meta)
        orthogonality.update(
            {
                "raw_design_rank": float(rank),
                "transformed_design_rank": float(rank),
                "raw_condition_number": float(condition),
                "transformed_condition_number": float(condition),
                "raw_duplicate_column_pairs": float(duplicate_column_pairs),
                "transformed_duplicate_column_pairs": float(
                    duplicate_column_pairs
                ),
            }
        )
        status = "identified" if not warnings else "unidentified"

        return DraftInteractionFitDiagnostics(
            family_id=family.family_id,
            feature_count=feature_count,
            row_count=row_count,
            fit_rank=rank,
            condition_number=condition,
            identification_status=status,
            orthogonality=orthogonality,
            fallback_term_count=sum(1 for value in fallback_counts.values() if value),
            fallback_counts={key: int(value) for key, value in fallback_counts.items() if value},
            feature_block_counts={key: int(value) for key, value in block_counts.items()},
            draw_count=0 if _skip_stability_check else self.draw_count,
            covariance_seed=int(self.draw_seed + _stable_seed(family.family_id) + row_count + len(coeff)),
            warnings=tuple(warnings),
            collinearity_max_correlation=float(col_corr),
            min_support=min_support,
        )

    def _source_removal_stability(self, rows: tuple[DraftCompositionRow, ...], family: DraftInteractionFamily) -> float:
        source_ids = sorted({row.source_id for row in rows})
        if len(source_ids) < 2 or len(rows) <= 4:
            return 0.0

        (
            base_x,
            base_meta,
            _,
            raw_term_ids,
            transform,
            *_,
        ) = self._build_design_bundle(rows, family)
        base_y = np.asarray([row.label for row in rows], dtype=float)
        prior_mean, prior_var = self._build_prior(rows, base_meta, family)
        base_fit = self._fit_logistic_ridge(
            base_x, base_y, prior_mean, prior_var
        )
        deltas: list[float] = []

        for source_id in source_ids:
            reduced_rows = tuple(row for row in rows if row.source_id != source_id)
            if len(reduced_rows) < max(3, len(rows) // 2):
                continue
            reduced_x = np.einsum(
                "ij,jk->ik",
                self._design_for_terms(reduced_rows, family, raw_term_ids),
                transform,
            )
            reduced_y = np.asarray(
                [row.label for row in reduced_rows], dtype=float
            )
            reduced_fit = self._fit_logistic_ridge(
                reduced_x, reduced_y, prior_mean, prior_var
            )
            deltas.append(float(np.max(np.abs(base_fit - reduced_fit))))

        return float(max(deltas)) if deltas else 0.0

    def _patch_removal_stability(self, rows: tuple[DraftCompositionRow, ...], family: DraftInteractionFamily) -> float:
        patch_ids = sorted({row.patch_id for row in rows})
        if len(patch_ids) < 2 or len(rows) <= 4:
            return 0.0

        (
            base_x,
            base_meta,
            _,
            raw_term_ids,
            transform,
            *_,
        ) = self._build_design_bundle(rows, family)
        base_y = np.asarray([row.label for row in rows], dtype=float)
        prior_mean, prior_var = self._build_prior(rows, base_meta, family)
        base_fit = self._fit_logistic_ridge(
            base_x, base_y, prior_mean, prior_var
        )
        deltas: list[float] = []

        for patch_id in patch_ids:
            reduced_rows = tuple(row for row in rows if row.patch_id != patch_id)
            if len(reduced_rows) < max(3, len(rows) // 2):
                continue
            reduced_x = np.einsum(
                "ij,jk->ik",
                self._design_for_terms(reduced_rows, family, raw_term_ids),
                transform,
            )
            reduced_y = np.asarray(
                [row.label for row in reduced_rows], dtype=float
            )
            reduced_fit = self._fit_logistic_ridge(
                reduced_x, reduced_y, prior_mean, prior_var
            )
            deltas.append(float(np.max(np.abs(base_fit - reduced_fit))))

        return float(max(deltas)) if deltas else 0.0

    def _design_for_terms(
        self,
        rows: tuple[DraftCompositionRow, ...],
        family: DraftInteractionFamily,
        term_ids: tuple[str, ...],
    ) -> np.ndarray:
        term_index = {term_id: index for index, term_id in enumerate(term_ids)}
        design = np.zeros((len(rows), len(term_ids)), dtype=float)
        for row_index, row in enumerate(rows):
            for term in self._row_terms(row, family):
                column = term_index.get(term.term_id)
                if column is not None:
                    design[row_index, column] += term.value
        return design

    def _orthogonalization_summary(
        self,
        term_meta: tuple[Mapping[str, Any], ...],
        x: np.ndarray,
        term_ids: tuple[str, ...],
    ) -> tuple[Mapping[str, Any], ...]:
        block_to_count: dict[str, int] = {}
        for md in term_meta:
            block_to_count[str(md.get("block", "unclassified"))] = block_to_count.get(str(md.get("block", "unclassified")), 0) + 1

        summary: list[Mapping[str, Any]] = []
        for term_id, md in zip(term_ids, term_meta):
            block = str(md.get("block", "unclassified"))
            summary.append(
                {
                    "term_id": term_id,
                    "block": block,
                    "projection": "weighted_residualization_on_frozen_legal_reference",
                    "legal_reference_id": "l6-finite-legal-composition-reference-v2",
                    "index": len(summary),
                    "block_rank_est": block_to_count.get(block, 0),
                    "projection_seed": _stable_seed(f"{block}|{term_id}"),
                }
            )
        return tuple(summary)

    def _compute_support(self, x: np.ndarray) -> Mapping[str, int]:
        return {f"col_{idx}": int(np.count_nonzero(x[:, idx])) for idx in range(x.shape[1])}

    def _predictive_draws(
        self,
        row_vector: np.ndarray,
        coeff: np.ndarray,
        cov_diag: tuple[float, ...],
        *,
        covariance_factor: tuple[tuple[float, ...], ...] = (),
        draw_count: int | None = None,
        covariance_seed: int | None = None,
    ) -> np.ndarray:
        if row_vector.size == 0:
            return np.array([], dtype=float)
        seed = (
            int(covariance_seed)
            if covariance_seed is not None
            else self.draw_seed + int(row_vector.size) * 17 + len(coeff) * 19
        )
        rng = np.random.default_rng(seed)
        variance = self._predictive_variance(
            row_vector, cov_diag, covariance_factor
        )
        sample_count = int(draw_count if draw_count is not None else self.draw_count)
        logits = rng.normal(
            float(np.dot(coeff, row_vector)),
            np.sqrt(variance),
            size=sample_count,
        )
        return _sigmoid(logits)

    def _predictive_variance(
        self,
        row_vector: np.ndarray,
        covariance_prior_diag: tuple[float, ...] | np.ndarray,
        covariance_factor: tuple[tuple[float, ...], ...] | np.ndarray,
    ) -> float:
        diagonal = np.maximum(
            np.asarray(covariance_prior_diag, dtype=float), 1e-12
        )
        variance = float(np.dot(row_vector * diagonal, row_vector))
        factor = np.asarray(covariance_factor, dtype=float)
        if factor.size:
            projected = np.einsum("ij,j->i", factor, row_vector)
            variance -= float(np.dot(projected, projected))
        return max(variance, 1e-12)

    def _interval_from_draws(self, draws: np.ndarray) -> tuple[float | None, float | None]:
        if draws.size == 0:
            return None, None
        return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))

    def _row_features(
        self,
        row: DraftCompositionRow,
        family: DraftInteractionFamily,
    ) -> tuple[tuple[str, ...], tuple[Mapping[str, Any], ...]]:
        contributions = self._row_terms(row, family)
        grouped: dict[str, Mapping[str, Any]] = {}

        for contribution in contributions:
            bucket = grouped.setdefault(
                contribution.term_id,
                {
                    "term_id": contribution.term_id,
                    "value": 0.0,
                    "a_value": 0.0,
                    "b_value": 0.0,
                    "neutral_value": 0.0,
                    **dict(contribution.metadata),
                    "block": contribution.block,
                    "metadata": dict(contribution.metadata),
                },
            )
            current = dict(bucket)
            current["value"] = float(current["value"]) + _safe_float(contribution.value, field_name="contribution value")
            side = str(contribution.metadata.get("side", "neutral")).upper()
            if side == "A":
                current["a_value"] = float(current["a_value"]) + _safe_float(contribution.value, field_name="side-a contribution value")
            elif side == "B":
                current["b_value"] = float(current["b_value"]) + _safe_float(contribution.value, field_name="side-b contribution value")
            else:
                current["neutral_value"] = float(current["neutral_value"]) + _safe_float(contribution.value, field_name="neutral contribution value")
            grouped[contribution.term_id] = current

        term_ids = tuple(sorted(grouped.keys()))
        term_values = tuple(
            {
                **grouped[term_id],
                "value": _safe_float(grouped[term_id]["value"], field_name="aggregated term value"),
                "a_value": _safe_float(grouped[term_id]["a_value"], field_name="aggregated a value"),
                "b_value": _safe_float(grouped[term_id]["b_value"], field_name="aggregated b value"),
                "neutral_value": _safe_float(grouped[term_id]["neutral_value"], field_name="aggregated neutral value"),
            }
            for term_id in term_ids
        )
        return term_ids, term_values

    def _reconcile_ledger(
        self,
        row: DraftCompositionRow,
        fit: DraftInteractionFit,
        term_values: tuple[Mapping[str, Any], ...],
    ) -> Mapping[str, Any]:
        terms = {entry["term_id"]: dict(entry) for entry in term_values}

        buckets = {
            "main": {"a": 0.0, "b": 0.0, "neutral": 0.0},
            "ally": {"a": 0.0, "b": 0.0, "neutral": 0.0},
            "enemy": {"a": 0.0, "b": 0.0, "neutral": 0.0},
            "whole_team": {"a": 0.0, "b": 0.0, "neutral": 0.0},
            "cross_team": {"a": 0.0, "b": 0.0, "neutral": 0.0},
            "fm_sparse": {"a": 0.0, "b": 0.0, "neutral": 0.0},
            "unclassified": {"a": 0.0, "b": 0.0, "neutral": 0.0},
        }

        coeff_map = fit.coefficients
        used_terms: list[Mapping[str, Any]] = []
        side_a_total = 0.0
        side_b_total = 0.0
        neutral_total = 0.0

        for term in fit.feature_terms:
            if term not in terms:
                continue
            metadata = terms[term]
            block = str(metadata.get("block", "unclassified"))
            if block == "main_global":
                bucket = "main"
            elif block == "main_role":
                bucket = "main"
            elif block == "main_scope":
                bucket = "main"
            elif block == "main_league":
                bucket = "main"
            elif block == "main_patch":
                bucket = "main"
            elif block.startswith("ally_"):
                bucket = "ally"
            elif block.startswith("enemy_"):
                bucket = "enemy"
            elif block == "whole_team":
                bucket = "whole_team"
            elif block == "cross_team":
                bucket = "cross_team"
            elif block == "fm_sparse":
                bucket = "fm_sparse"
            else:
                bucket = "unclassified"

            coeff = _safe_float(coeff_map.get(term, 0.0), field_name="coefficient")
            total_value = _safe_float(metadata.get("value", 0.0), field_name="term value")
            a_value = _safe_float(metadata.get("a_value", 0.0), field_name="term a value")
            b_value = _safe_float(metadata.get("b_value", 0.0), field_name="term b value")
            neutral_value = _safe_float(metadata.get("neutral_value", 0.0), field_name="term neutral value")

            a_contrib = coeff * a_value
            b_contrib = coeff * b_value
            neutral_contrib = coeff * neutral_value

            buckets[bucket]["a"] += a_contrib
            buckets[bucket]["b"] += b_contrib
            buckets[bucket]["neutral"] += neutral_contrib

            side_a_total += a_contrib
            side_b_total += b_contrib
            neutral_total += neutral_contrib

            used_terms.append(
                {
                    "term_id": term,
                    "block": bucket,
                    "value": total_value,
                    "weight": coeff * total_value,
                    "split": {
                        "a": a_contrib,
                        "b": b_contrib,
                        "neutral": neutral_contrib,
                    },
                    "metadata": dict(metadata),
                }
            )

        feature_terms = tuple(fit.feature_terms)
        feature_matrix = np.array(
            [
                _safe_float(
                    terms.get(t, {}).get("value", 0.0),
                    field_name="ledger feature",
                )
                for t in feature_terms
            ],
            dtype=float,
        )
        coeff_vector = np.array([coeff_map.get(t, 0.0) for t in feature_terms], dtype=float)
        prediction = float(np.dot(feature_matrix, coeff_vector))

        block_total = 0.0
        for bucket in ("main", "ally", "enemy", "whole_team", "cross_team", "fm_sparse", "unclassified"):
            value = buckets[bucket]["a"] + buckets[bucket]["b"] + buckets[bucket]["neutral"]
            block_total += value

        return {
            "status": "available",
            "family_id": fit.family_id,
            "row_id": row.row_id,
            "reconciliation_error": float(abs(block_total - prediction)),
            "reconciliation_total": {
                "prediction_logit": prediction,
                "sum_by_buckets": block_total,
                "difference": float(abs(block_total - prediction)),
            },
            "side_totals": {
                "a": side_a_total,
                "b": side_b_total,
                "a_minus_b": side_a_total + side_b_total,
                "neutral": neutral_total,
                "a_minus_b_plus_neutral": side_a_total + side_b_total + neutral_total,
            },
            "side_split": {
                "a_minus_b": "side_A_minus_side_B",
                "neutral_handled_as_side_zero": True,
            },
            "block_contributions": {name: {key: float(value) for key, value in parts.items()} for name, parts in buckets.items()},
            "used_terms": used_terms,
            "used_term_count": len(used_terms),
            "term_count_by_block": {
                name: sum(1 for item in used_terms if item["block"] == name) for name in buckets
            },
            "split_convention": "A-B neutral composition value with a/b term accumulation",
        }

    def _development_splits(
        self,
        rows: tuple[DraftCompositionRow, ...],
    ) -> tuple[
        tuple[tuple[DraftCompositionRow, ...], tuple[DraftCompositionRow, ...]],
        ...,
    ]:
        folds: list[
            tuple[
                tuple[DraftCompositionRow, ...],
                tuple[DraftCompositionRow, ...],
            ]
        ] = []
        for source_id in sorted({row.source_id for row in rows}):
            eval_rows = tuple(row for row in rows if row.source_id == source_id)
            train_rows = tuple(row for row in rows if row.source_id != source_id)
            if (
                len(train_rows) >= 4
                and {row.label for row in train_rows} == {0, 1}
                and {row.label for row in eval_rows} == {0, 1}
            ):
                folds.append((train_rows, eval_rows))
        return tuple(folds)

    def _evaluate_dev_folds(self, rows: tuple[DraftCompositionRow, ...], family_id: str) -> list[tuple[float, float, float, int]]:
        if len(rows) < 4:
            return []
        folds = self._development_splits(rows)
        output: list[tuple[float, float, float, int]] = []
        for train_rows, eval_rows in folds:
            if len(train_rows) < 4 or not eval_rows:
                continue
            fit = self.fit(train_rows, family_id=family_id, _skip_stability_check=True)
            if fit.decomposition_mode != "identified":
                continue
            pred = np.array([self.predict(fit, row).raw_probability for row in eval_rows], dtype=float)
            target = np.array([float(row.label) for row in eval_rows], dtype=float)
            pred = np.clip(pred, 1e-6, 1.0 - 1e-6)
            log_loss = float(np.mean(-(target * np.log(pred) + (1.0 - target) * np.log(1.0 - pred))))
            brier = float(np.mean((pred - target) ** 2))
            ece = float(self._expected_calibration_error(target, pred))
            output.append((log_loss, brier, ece, len(eval_rows)))
        return output

    def _expected_calibration_error(self, y: np.ndarray, p: np.ndarray, bins: int = 6) -> float:
        if y.size == 0:
            return 1.0
        edges = np.linspace(0.0, 1.0, num=bins + 1)
        ece = 0.0
        total = y.size
        for idx in range(bins):
            lower = edges[idx]
            upper = edges[idx + 1]
            if idx == bins - 1:
                mask = (p >= lower) & (p <= upper)
            else:
                mask = (p >= lower) & (p < upper)
            if not np.any(mask):
                continue
            p_bin = p[mask]
            y_bin = y[mask]
            ece += (len(p_bin) / total) * abs(float(p_bin.mean()) - float(y_bin.mean()))
        return float(ece)


# Public row-level helper.
def score_row_pair_swap(row: DraftCompositionRow) -> DraftCompositionRow:
    row = row if isinstance(row, DraftCompositionRow) else DraftCompositionRow.from_payload(row)
    return DraftCompositionRow(
        row_id=f"swap::{row.row_id}",
        patch_id=row.patch_id,
        league_id=row.league_id,
        side_a=row.side_b,
        side_b=row.side_a,
        label=1 - row.label,
        source_id=row.source_id,
        source_patch_pool=row.source_patch_pool,
        metadata={"base_row": row.row_id, "swap": True},
        competition_scope_id=row.competition_scope_id,
    )


def run_candidate_selection(
    rows: Sequence[DraftCompositionRow],
    *,
    draw_count: int = 64,
    draw_seed: int = 20260728,
    family_order: Sequence[str] | None = None,
) -> DraftInteractionSelectionReport:
    model = DraftInteractionModel(draw_count=draw_count, draw_seed=draw_seed)
    rows = model._coerce_rows(rows)
    if not rows:
        raise DraftInteractionError("selection requires at least one row")

    ordered_ids = tuple(f.family_id for f in model.FAMILY_REGISTRY) if family_order is None else tuple(family_order)

    candidates: list[DraftInteractionCandidate] = []
    for family_id in ordered_ids:
        family = model._family_by_id.get(family_id)
        if family is None:
            candidates.append(
                DraftInteractionCandidate(
                    family_id=family_id,
                    status="rejected",
                    metrics={},
                    diagnostics={"error": f"unknown family requested: {family_id}"},
                    candidate_sha256=canonical_sha256({"family_id": family_id, "error": "unknown"}),
                    selected=False,
                )
            )
            continue

        try:
            fit = model.fit(rows, family_id=family_id)
            folds = model._evaluate_dev_folds(rows, family_id)
            if folds:
                log_loss = float(np.mean([entry[0] for entry in folds]))
                brier = float(np.mean([entry[1] for entry in folds]))
                ece = float(np.mean([entry[2] for entry in folds]))
                eval_rows = int(sum(entry[3] for entry in folds))
            else:
                log_loss, brier, ece, eval_rows = 1.0, 1.0, 1.0, 0

            metrics = {
                "log_loss": log_loss,
                "brier": brier,
                "ece": ece,
                "eval_rows": eval_rows,
                "feature_count": fit.diagnostics.feature_count,
                "identification": fit.diagnostics.identification_status,
            }
            diagnostic_payload = {
                **fit.diagnostics.as_payload,
                "reference_sha256": fit.reference_sha256,
                "transform_sha256": fit.transform_sha256,
                "raw_feature_count": len(fit.raw_feature_terms),
                "supported_feature_count": len(fit.feature_terms),
                "unsupported_feature_count": len(fit.raw_feature_terms)
                - len(fit.feature_terms),
            }
            payload = {
                "family_id": family_id,
                "status": fit.diagnostics.identification_status,
                "metrics": metrics,
                "diagnostics": diagnostic_payload,
                "folds": [
                    {"log_loss": value[0], "brier": value[1], "ece": value[2], "eval_rows": value[3]}
                    for value in folds
                ],
                "fit_sha256": canonical_sha256(
                    {
                        "family_id": family_id,
                        "row_count": len(rows),
                        "terms": sorted(fit.feature_terms),
                    }
                ),
            }
            candidates.append(
                DraftInteractionCandidate(
                    family_id=family_id,
                    status=fit.diagnostics.identification_status,
                    metrics=metrics,
                    diagnostics=diagnostic_payload,
                    candidate_sha256=canonical_sha256(payload),
                    selected=False,
                )
            )
        except DraftInteractionError as exc:
            candidates.append(
                DraftInteractionCandidate(
                    family_id=family_id,
                    status="rejected",
                    metrics={},
                    diagnostics={"error": str(exc)},
                    candidate_sha256=canonical_sha256({"family_id": family_id, "error": str(exc)}),
                    selected=False,
                )
            )

    ordered_candidates = sorted(
        candidates,
        key=lambda candidate: (
            0 if candidate.status == "identified" else 1,
            candidate.metrics.get("log_loss", 1e9),
            candidate.metrics.get("brier", 1e9),
            candidate.family_id,
        ),
    )

    eligible = [candidate for candidate in ordered_candidates if candidate.status == "identified"]
    selected_family: str | None = None
    selected_sha256: str | None = None
    selected_status = "blocked_no_identified_candidate"

    if (
        eligible
        and len(rows) >= model._SELECTION_MIN_ROWS
        and all(
        int(candidate.metrics.get("eval_rows", 0)) > 0 for candidate in eligible
        )
    ):
        best = min(
            eligible,
            key=lambda candidate: (
                candidate.metrics.get("log_loss", 1e9),
                candidate.metrics.get("brier", 1e9),
                candidate.metrics.get("ece", 1e9),
            ),
        )
        selected_family = best.family_id
        selected_sha256 = best.candidate_sha256
        selected_status = "selected"

    selection_config = {
        "selection_seed": reveal_synthetic_seed(),
        "draw_seed": draw_seed,
        "draw_count": draw_count,
        "candidate_family_order": list(ordered_ids),
        "row_count": len(rows),
        "unique_sources": sorted({row.source_id for row in rows}),
        "unique_patches": sorted({row.patch_id for row in rows}),
        "min_rows_for_selection": model._SELECTION_MIN_ROWS,
        "selected_family": selected_family,
        "selection_status": selected_status,
        "eligible_count": len(eligible),
    }
    selection_sha256 = canonical_sha256({"selection": selection_config, "candidates": [candidate.as_payload for candidate in ordered_candidates]})

    return DraftInteractionSelectionReport(
        config_sha256=canonical_sha256(selection_config),
        candidate_count=len(ordered_candidates),
        candidates=tuple(
            DraftInteractionCandidate(
                family_id=candidate.family_id,
                status=candidate.status,
                metrics=candidate.metrics,
                diagnostics=candidate.diagnostics,
                candidate_sha256=candidate.candidate_sha256,
                selected=(candidate.family_id == selected_family),
            )
            for candidate in ordered_candidates
        ),
        selection_status=selected_status,
        selected_family=selected_family,
        selected_sha256=selected_sha256,
        selection_sha256=selection_sha256,
    )
