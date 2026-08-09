"""Analytic structured priors for the L6 draft-interaction model.

This module is a development kernel.  It deliberately does not switch the
production model, issue an authority claim, or infer hyperparameters from the
same observations whose coefficients it regularizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import re
from typing import Any, Mapping, Sequence

import numpy as np

from lol_kills.v2.evaluation.types import (
    canonical_sha256,
    canonical_timestamp,
    parse_utc_timestamp,
)


class StructuredPriorError(ValueError):
    """Raised when a structured-prior input or constraint is invalid."""


_BLOCK_NAMES = (
    "main",
    "ally",
    "enemy",
    "champion_residual",
    "H",
    "K",
)
_HIERARCHY_NAMES = ("global", "competition_scope", "league", "patch")
_PARENTS = {
    "main": None,
    "ally": "main/archetype",
    "enemy": "main/archetype",
    "champion_residual": "main/archetype",
    "H": "ally",
    "K": "enemy",
}
_FIXED_BLOCK_SDS = {
    "main": 0.35,
    "ally": 0.15,
    "enemy": 0.20,
    "champion_residual": 0.20,
    "H": 0.08,
    "K": 0.08,
}
_FIXED_HIERARCHY_MULTIPLIERS = {
    "global": 1.0,
    "competition_scope": 0.35,
    "league": 0.35,
    "patch": 0.25,
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _immutable_array(value: Any) -> np.ndarray:
    """Return an ndarray backed by immutable bytes.

    Unlike a normally owned read-only ndarray, callers cannot re-enable writes
    with ``setflags(write=True)`` because the underlying buffer itself is
    immutable.
    """

    array = np.ascontiguousarray(np.asarray(value, dtype=float))
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _finite_positive(value: Any, *, name: str, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise StructuredPriorError(f"{name} must be numeric")
    parsed = float(value)
    if not isfinite(parsed) or (parsed < 0.0 if allow_zero else parsed <= 0.0):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise StructuredPriorError(f"{name} must be finite and {qualifier}")
    return parsed


def _ordered_pairs(
    values: Sequence[tuple[str, float]], *, expected: Sequence[str], label: str
) -> tuple[tuple[str, float], ...]:
    parsed = {str(name): _finite_positive(value, name=f"{label}.{name}") for name, value in values}
    if set(parsed) != set(expected) or len(values) != len(expected):
        raise StructuredPriorError(f"{label} must contain exactly {tuple(expected)}")
    return tuple((name, parsed[name]) for name in expected)


@dataclass(frozen=True)
class StructuredPriorSpec:
    """Immutable, canonical specification for the single principal model."""

    version: str = "l6-structured-prior-v1"
    ontology_rank: int = 2
    block_logit_sds: tuple[tuple[str, float], ...] = (
        ("main", 0.35),
        ("ally", 0.15),
        ("enemy", 0.20),
        ("champion_residual", 0.20),
        ("H", 0.08),
        ("K", 0.08),
    )
    hierarchy_multipliers: tuple[tuple[str, float], ...] = (
        ("global", 1.0),
        ("competition_scope", 0.35),
        ("league", 0.35),
        ("patch", 0.25),
    )
    unavailable_blocks: tuple[str, ...] = ("H", "K")
    reference_conditioning_version: str = "weighted-contrast-conditioning-v1"
    minimum_effective_weight: float = 1e-8
    maximum_absolute_basis_value: float = 1e4
    maximum_effect_covariance_condition_number: float = 1e9

    def __post_init__(self) -> None:
        if self.version != "l6-structured-prior-v1":
            raise StructuredPriorError("unsupported structured-prior version")
        if self.ontology_rank != 2:
            raise StructuredPriorError("the principal ontology latent rank must be 2")
        if self.reference_conditioning_version != "weighted-contrast-conditioning-v1":
            raise StructuredPriorError("unsupported reference conditioning version")
        minimum_weight = _finite_positive(
            self.minimum_effective_weight, name="minimum_effective_weight"
        )
        maximum_basis = _finite_positive(
            self.maximum_absolute_basis_value, name="maximum_absolute_basis_value"
        )
        maximum_covariance_condition = _finite_positive(
            self.maximum_effect_covariance_condition_number,
            name="maximum_effect_covariance_condition_number",
        )
        if (
            minimum_weight != 1e-8
            or maximum_basis != 1e4
            or maximum_covariance_condition != 1e9
        ):
            raise StructuredPriorError("v1 reference conditioning bounds are fixed")
        block_sds = _ordered_pairs(
            self.block_logit_sds,
            expected=_BLOCK_NAMES,
            label="block_logit_sds",
        )
        if dict(block_sds) != _FIXED_BLOCK_SDS:
            raise StructuredPriorError("v1 block logit SDs are fixed")
        object.__setattr__(
            self,
            "block_logit_sds",
            block_sds,
        )
        hierarchy = _ordered_pairs(
            self.hierarchy_multipliers,
            expected=_HIERARCHY_NAMES,
            label="hierarchy_multipliers",
        )
        if dict(hierarchy) != _FIXED_HIERARCHY_MULTIPLIERS:
            raise StructuredPriorError("v1 hierarchy multipliers are fixed")
        object.__setattr__(
            self,
            "hierarchy_multipliers",
            hierarchy,
        )
        unavailable = tuple(sorted(set(self.unavailable_blocks)))
        if unavailable != ("H", "K"):
            raise StructuredPriorError("future H and K blocks must remain unavailable")
        object.__setattr__(self, "unavailable_blocks", unavailable)

    def to_payload(self) -> dict[str, Any]:
        block_sds = dict(self.block_logit_sds)
        return {
            "schema": self.version,
            "principal_model": {
                "ontology_latent_rank": self.ontology_rank,
                "block_logit_sds": block_sds,
                "block_logit_sd_semantics": (
                    "original_effect_space_marginal_standard_deviation_upper_bound"
                ),
                "contrast_coordinate_prior": (
                    "derived_from_effect_space_bound_not_isotropic"
                ),
                "hierarchy_multipliers": dict(self.hierarchy_multipliers),
                "reference_conditioning": {
                    "version": self.reference_conditioning_version,
                    "minimum_effective_weight": self.minimum_effective_weight,
                    "maximum_absolute_basis_value": self.maximum_absolute_basis_value,
                    "maximum_effect_covariance_condition_number": (
                        self.maximum_effect_covariance_condition_number
                    ),
                    "rationale": (
                        "permits realistic role-specific champion weights while "
                        "rejecting numerically dominant point masses"
                    ),
                },
                "block_availability": {
                    name: name not in self.unavailable_blocks for name in _BLOCK_NAMES
                },
                "strong_hierarchy": {
                    "status": (
                        "declared_principal_model_constraint_not_yet_fit_enforced"
                    ),
                    "blocks": {
                        name: {
                            "parent": _PARENTS[name],
                            "semantics": (
                                "deviation_not_replacement"
                                if name == "champion_residual"
                                else "hierarchical_component"
                            ),
                        }
                        for name in _BLOCK_NAMES
                    },
                },
            },
            "authority": {
                "development_only": True,
                "authorizes_predictive_claims": False,
                "authorizes_publication": False,
            },
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_sha256(self.to_payload())

    def block_sd(self, block: str) -> float:
        try:
            return dict(self.block_logit_sds)[block]
        except KeyError as exc:
            raise StructuredPriorError(f"unknown interaction block: {block}") from exc

    def hierarchy_multiplier(self, hierarchy: str) -> float:
        try:
            return dict(self.hierarchy_multipliers)[hierarchy]
        except KeyError as exc:
            raise StructuredPriorError(f"unknown hierarchy: {hierarchy}") from exc


PRINCIPAL_PRIOR_SPEC = StructuredPriorSpec()


@dataclass(frozen=True)
class ContrastDiagnostics:
    rank: int
    max_centering_error: float
    max_whitening_error: float
    minimum_effective_weight: float
    maximum_absolute_basis_value: float
    conditioning_version: str


@dataclass(frozen=True)
class WeightedContrast:
    """Canonical weighted-orthonormal contrast basis."""

    labels: tuple[str, ...]
    weights: np.ndarray
    basis: np.ndarray
    diagnostics: ContrastDiagnostics

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "l6-weighted-contrast-v1",
            "labels": list(self.labels),
            "weights": self.weights.tolist(),
            "basis": self.basis.tolist(),
            "diagnostics": {
                "rank": self.diagnostics.rank,
                "max_centering_error": self.diagnostics.max_centering_error,
                "max_whitening_error": self.diagnostics.max_whitening_error,
                "minimum_effective_weight": self.diagnostics.minimum_effective_weight,
                "maximum_absolute_basis_value": (
                    self.diagnostics.maximum_absolute_basis_value
                ),
                "conditioning_version": self.diagnostics.conditioning_version,
            },
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class EffectSpacePriorDiagnostics:
    coordinate_rank: int
    gram_condition_number: float
    minimum_gram_eigenvalue: float
    maximum_gram_eigenvalue: float
    minimum_coordinate_covariance_eigenvalue: float
    minimum_induced_effect_eigenvalue: float
    maximum_induced_marginal_variance: float
    marginal_variance_upper_bound: float
    max_coordinate_symmetry_error: float
    max_induced_symmetry_error: float
    validation_tolerance: float


@dataclass(frozen=True)
class EffectSpacePriorCovariance:
    """Contrast-coordinate covariance derived from an effect-space SD bound."""

    contrast_sha256: str
    labels: tuple[str, ...]
    effect_space_sd_upper_bound: float
    coordinate_covariance: np.ndarray
    diagnostics: EffectSpacePriorDiagnostics

    def to_payload(self) -> dict[str, Any]:
        diagnostics = self.diagnostics
        return {
            "schema": "l6-effect-space-prior-covariance-v1",
            "contrast_sha256": self.contrast_sha256,
            "labels": list(self.labels),
            "effect_space_sd_upper_bound": self.effect_space_sd_upper_bound,
            "coordinate_covariance": self.coordinate_covariance.tolist(),
            "diagnostics": {
                "coordinate_rank": diagnostics.coordinate_rank,
                "gram_condition_number": diagnostics.gram_condition_number,
                "minimum_gram_eigenvalue": diagnostics.minimum_gram_eigenvalue,
                "maximum_gram_eigenvalue": diagnostics.maximum_gram_eigenvalue,
                "minimum_coordinate_covariance_eigenvalue": (
                    diagnostics.minimum_coordinate_covariance_eigenvalue
                ),
                "minimum_induced_effect_eigenvalue": (
                    diagnostics.minimum_induced_effect_eigenvalue
                ),
                "maximum_induced_marginal_variance": (
                    diagnostics.maximum_induced_marginal_variance
                ),
                "marginal_variance_upper_bound": (
                    diagnostics.marginal_variance_upper_bound
                ),
                "max_coordinate_symmetry_error": (
                    diagnostics.max_coordinate_symmetry_error
                ),
                "max_induced_symmetry_error": diagnostics.max_induced_symmetry_error,
                "validation_tolerance": diagnostics.validation_tolerance,
            },
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_sha256(self.to_payload())


def effect_space_prior_covariance(
    contrast: WeightedContrast,
    effect_space_sd_upper_bound: float,
    *,
    tolerance: float = 1e-10,
    spec: StructuredPriorSpec = PRINCIPAL_PRIOR_SPEC,
) -> EffectSpacePriorCovariance:
    """Derive ``Sigma_theta = tau^2 (B.T B)^-1`` without a raw inverse.

    This parameterization makes ``tau`` an upper bound on every original
    effect-space marginal SD.  All rank, conditioning, symmetry, and PSD
    properties are revalidated rather than trusted from the input dataclass.
    """

    if not isinstance(contrast, WeightedContrast):
        raise StructuredPriorError("contrast must be a WeightedContrast")
    tau = _finite_positive(
        effect_space_sd_upper_bound, name="effect_space_sd_upper_bound"
    )
    tolerance = _finite_positive(tolerance, name="tolerance")
    basis = np.asarray(contrast.basis, dtype=float)
    weights = np.asarray(contrast.weights, dtype=float)
    level_count = len(contrast.labels)
    if (
        basis.shape != (level_count, level_count - 1)
        or weights.shape != (level_count,)
        or level_count < 2
        or not np.all(np.isfinite(basis))
        or not np.all(np.isfinite(weights))
    ):
        raise StructuredPriorError("contrast dimensions and values must be valid")
    if (
        abs(float(weights.sum()) - 1.0) > tolerance
        or float(np.min(weights)) < spec.minimum_effective_weight
        or float(np.max(np.abs(basis))) > spec.maximum_absolute_basis_value
    ):
        raise StructuredPriorError("contrast violates reference conditioning bounds")
    centering = np.einsum("i,ij->j", weights, basis)
    if not np.all(np.isfinite(centering)):
        raise StructuredPriorError("contrast centering diagnostics must be finite")
    if float(np.max(np.abs(centering))) > tolerance:
        raise StructuredPriorError("contrast is not weighted-centered")
    weighted_gram = np.einsum(
        "ji,jk->ik", basis, weights[:, None] * basis
    )
    if not np.all(np.isfinite(weighted_gram)):
        raise StructuredPriorError("contrast weighted Gram matrix must be finite")
    if float(np.max(np.abs(weighted_gram - np.eye(level_count - 1)))) > tolerance:
        raise StructuredPriorError("contrast is not weighted-whitened")

    gram = np.einsum("ji,jk->ik", basis, basis)
    gram = 0.5 * (gram + gram.T)
    if not np.all(np.isfinite(gram)):
        raise StructuredPriorError("contrast Gram matrix must be finite")
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
    except np.linalg.LinAlgError as exc:
        raise StructuredPriorError("contrast Gram eigendecomposition failed") from exc
    if not (
        np.all(np.isfinite(eigenvalues))
        and np.all(np.isfinite(eigenvectors))
    ):
        raise StructuredPriorError("contrast Gram eigensystem must be finite")
    maximum_gram_eigenvalue = float(eigenvalues[-1])
    rank_threshold = max(
        np.finfo(float).eps * (level_count - 1) * maximum_gram_eigenvalue,
        tolerance * maximum_gram_eigenvalue,
    )
    coordinate_rank = int(np.count_nonzero(eigenvalues > rank_threshold))
    if coordinate_rank != level_count - 1 or float(eigenvalues[0]) <= 0.0:
        raise StructuredPriorError("contrast Gram matrix is numerically rank-deficient")
    condition_number = maximum_gram_eigenvalue / float(eigenvalues[0])
    if (
        not np.isfinite(condition_number)
        or condition_number > spec.maximum_effect_covariance_condition_number
    ):
        raise StructuredPriorError("contrast Gram condition number exceeds the bound")

    inverse_eigenvalues = (tau * tau) / eigenvalues
    if not np.all(np.isfinite(inverse_eigenvalues)):
        raise StructuredPriorError("inverse Gram eigenvalues must be finite")
    coordinate_covariance = np.einsum(
        "ik,k,jk->ij",
        eigenvectors,
        inverse_eigenvalues,
        eigenvectors,
    )
    coordinate_covariance = 0.5 * (
        coordinate_covariance + coordinate_covariance.T
    )
    induced = np.einsum(
        "ik,kl,jl->ij",
        basis,
        coordinate_covariance,
        basis,
    )
    induced = 0.5 * (induced + induced.T)
    if not (
        np.all(np.isfinite(coordinate_covariance))
        and np.all(np.isfinite(induced))
    ):
        raise StructuredPriorError("derived effect-space covariance must be finite")
    coordinate_symmetry_error = float(
        np.max(np.abs(coordinate_covariance - coordinate_covariance.T))
    )
    induced_symmetry_error = float(np.max(np.abs(induced - induced.T)))
    try:
        coordinate_eigenvalues = np.linalg.eigvalsh(coordinate_covariance)
        induced_eigenvalues = np.linalg.eigvalsh(induced)
    except np.linalg.LinAlgError as exc:
        raise StructuredPriorError("prior covariance PSD validation failed") from exc
    if not (
        np.all(np.isfinite(coordinate_eigenvalues))
        and np.all(np.isfinite(induced_eigenvalues))
    ):
        raise StructuredPriorError("prior covariance eigenvalues must be finite")
    psd_tolerance = tolerance * max(1.0, tau * tau)
    minimum_coordinate_eigenvalue = float(coordinate_eigenvalues[0])
    minimum_induced_eigenvalue = float(induced_eigenvalues[0])
    maximum_marginal_variance = float(np.max(np.diag(induced)))
    if (
        coordinate_symmetry_error > tolerance
        or induced_symmetry_error > tolerance
        or minimum_coordinate_eigenvalue < -psd_tolerance
        or minimum_induced_eigenvalue < -psd_tolerance
        or maximum_marginal_variance > tau * tau + psd_tolerance
        or not all(
            np.isfinite(value)
            for value in (
                coordinate_symmetry_error,
                induced_symmetry_error,
                minimum_coordinate_eigenvalue,
                minimum_induced_eigenvalue,
                maximum_marginal_variance,
            )
        )
    ):
        raise StructuredPriorError("derived effect-space covariance failed validation")

    diagnostics = EffectSpacePriorDiagnostics(
        coordinate_rank=coordinate_rank,
        gram_condition_number=float(condition_number),
        minimum_gram_eigenvalue=float(eigenvalues[0]),
        maximum_gram_eigenvalue=maximum_gram_eigenvalue,
        minimum_coordinate_covariance_eigenvalue=minimum_coordinate_eigenvalue,
        minimum_induced_effect_eigenvalue=minimum_induced_eigenvalue,
        maximum_induced_marginal_variance=maximum_marginal_variance,
        marginal_variance_upper_bound=tau * tau,
        max_coordinate_symmetry_error=coordinate_symmetry_error,
        max_induced_symmetry_error=induced_symmetry_error,
        validation_tolerance=tolerance,
    )
    return EffectSpacePriorCovariance(
        contrast_sha256=contrast.payload_sha256,
        labels=contrast.labels,
        effect_space_sd_upper_bound=tau,
        coordinate_covariance=_immutable_array(coordinate_covariance),
        diagnostics=diagnostics,
    )


def _canonical_order(
    weights: Sequence[float], labels: Sequence[str] | None
) -> tuple[np.ndarray, tuple[str, ...]]:
    array = np.asarray(weights, dtype=float)
    if array.ndim != 1 or array.size < 2:
        raise StructuredPriorError("weights must be a one-dimensional vector of length >= 2")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise StructuredPriorError("weights must be finite and strictly positive")
    if abs(float(array.sum()) - 1.0) > 1e-12:
        raise StructuredPriorError("weights must sum to 1")
    if labels is None:
        canonical_labels = tuple(str(index) for index in range(array.size))
        order = np.arange(array.size)
    else:
        if len(labels) != array.size:
            raise StructuredPriorError("labels and weights must have equal length")
        canonical_labels = tuple(str(label) for label in labels)
        if any(not label for label in canonical_labels) or len(set(canonical_labels)) != len(
            canonical_labels
        ):
            raise StructuredPriorError("labels must be unique non-empty strings")
        order = np.asarray(sorted(range(array.size), key=lambda index: canonical_labels[index]))
        canonical_labels = tuple(canonical_labels[index] for index in order)
    return array[order], canonical_labels


def weighted_contrast_basis(
    weights: Sequence[float],
    *,
    labels: Sequence[str] | None = None,
    tolerance: float = 1e-10,
    spec: StructuredPriorSpec = PRINCIPAL_PRIOR_SPEC,
) -> WeightedContrast:
    """Construct B with ``w.T @ B = 0`` and ``B.T @ diag(w) @ B = I``.

    A Householder reflection maps ``sqrt(w)`` to the first coordinate.  Its
    remaining columns span the exact weighted-contrast space, so no catalogue
    of legal five-champion compositions is required.
    """

    tolerance = _finite_positive(tolerance, name="tolerance")
    ordered_weights, ordered_labels = _canonical_order(weights, labels)
    minimum_weight = float(np.min(ordered_weights))
    if minimum_weight < spec.minimum_effective_weight:
        raise StructuredPriorError(
            "weights violate the manifested minimum effective weight"
        )
    root = np.sqrt(ordered_weights)
    first = np.zeros_like(root)
    first[0] = 1.0
    direction = root - first
    norm_squared = float(np.einsum("i,i->", direction, direction))
    if norm_squared <= np.finfo(float).eps:
        raise StructuredPriorError("weights are numerically rank-deficient")
    reflection = np.eye(root.size) - 2.0 * np.outer(direction, direction) / norm_squared
    null_columns = reflection[:, 1:]
    basis = null_columns / root[:, None]

    centering_error = float(
        np.max(np.abs(np.einsum("i,ij->j", ordered_weights, basis)))
    )
    gram = np.einsum(
        "ji,jk->ik", basis, ordered_weights[:, None] * basis
    )
    whitening_error = float(np.max(np.abs(gram - np.eye(root.size - 1))))
    maximum_basis = float(np.max(np.abs(basis)))
    rank = int(np.linalg.matrix_rank(basis, tol=tolerance))
    if (
        rank != root.size - 1
        or centering_error > tolerance
        or whitening_error > tolerance
        or maximum_basis > spec.maximum_absolute_basis_value
    ):
        raise StructuredPriorError("weighted contrast construction failed its constraints")

    return WeightedContrast(
        labels=ordered_labels,
        weights=_immutable_array(ordered_weights),
        basis=_immutable_array(basis),
        diagnostics=ContrastDiagnostics(
            rank=rank,
            max_centering_error=centering_error,
            max_whitening_error=whitening_error,
            minimum_effective_weight=minimum_weight,
            maximum_absolute_basis_value=maximum_basis,
            conditioning_version=spec.reference_conditioning_version,
        ),
    )


@dataclass(frozen=True)
class FrozenOntologyMoments:
    labels: tuple[str, ...]
    weights: np.ndarray
    mean: np.ndarray
    centered_embeddings: np.ndarray
    payload_sha256: str

    def to_payload(self) -> dict[str, Any]:
        error = float(np.max(np.abs(self.weights @ self.centered_embeddings)))
        return {
            "schema": "l6-ontology-moments-v1",
            "ontology_rank": int(self.centered_embeddings.shape[1]),
            "labels": list(self.labels),
            "weights": self.weights.tolist(),
            "weighted_mean": self.mean.tolist(),
            "centered_embeddings": self.centered_embeddings.tolist(),
            "max_centering_error": error,
        }

    @property
    def payload(self) -> Mapping[str, Any]:
        return self.to_payload()


def center_ontology_embeddings(
    embeddings: Sequence[Sequence[float]],
    weights: Sequence[float],
    *,
    labels: Sequence[str],
    expected_rank: int = 2,
    spec: StructuredPriorSpec = PRINCIPAL_PRIOR_SPEC,
) -> FrozenOntologyMoments:
    """Center a role-specific ontology matrix under manifested role weights."""

    if expected_rank != 2:
        raise StructuredPriorError("the principal ontology latent rank must be 2")
    matrix = np.asarray(embeddings, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != expected_rank or matrix.shape[0] < 2:
        raise StructuredPriorError(
            f"ontology embeddings must have shape (n, {expected_rank}) with n >= 2"
        )
    if not np.all(np.isfinite(matrix)):
        raise StructuredPriorError("ontology embeddings must be finite")
    ordered_weights, ordered_labels = _canonical_order(weights, labels)
    if float(np.min(ordered_weights)) < spec.minimum_effective_weight:
        raise StructuredPriorError(
            "ontology weights violate the manifested minimum effective weight"
        )
    input_labels = tuple(str(label) for label in labels)
    order = np.asarray([input_labels.index(label) for label in ordered_labels])
    ordered_matrix = matrix[order]
    mean = ordered_weights @ ordered_matrix
    centered = ordered_matrix - mean
    error = float(np.max(np.abs(ordered_weights @ centered)))
    if error > 1e-12:
        raise StructuredPriorError("ontology centering failed")
    payload = {
        "schema": "l6-ontology-moments-v1",
        "ontology_rank": expected_rank,
        "labels": list(ordered_labels),
        "weights": ordered_weights.tolist(),
        "weighted_mean": mean.tolist(),
        "centered_embeddings": centered.tolist(),
        "max_centering_error": error,
    }
    return FrozenOntologyMoments(
        labels=ordered_labels,
        weights=_immutable_array(ordered_weights),
        mean=_immutable_array(mean),
        centered_embeddings=_immutable_array(centered),
        payload_sha256=canonical_sha256(payload),
    )


@dataclass(frozen=True)
class RoleRelations:
    """Immutable role-block relation matrices."""

    kind: str
    roles: tuple[str, ...]
    rank: int
    blocks: tuple[tuple[str, str, np.ndarray], ...]

    def block(self, left_role: str, right_role: str) -> np.ndarray:
        for left, right, matrix in self.blocks:
            if left == left_role and right == right_role:
                return matrix
        raise StructuredPriorError(f"missing {self.kind} block {left_role},{right_role}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "l6-role-relations-v1",
            "kind": self.kind,
            "roles": list(self.roles),
            "rank": self.rank,
            "blocks": [
                {"left_role": left, "right_role": right, "matrix": matrix.tolist()}
                for left, right, matrix in self.blocks
            ],
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_sha256(self.to_payload())


def _validate_roles(roles: Sequence[str], rank: int) -> tuple[str, ...]:
    canonical = tuple(str(role) for role in roles)
    if len(canonical) == 0 or any(not role for role in canonical) or len(set(canonical)) != len(
        canonical
    ):
        raise StructuredPriorError("roles must be unique non-empty strings")
    if rank != 2:
        raise StructuredPriorError("relation matrices must use ontology rank 2")
    return canonical


def _matrix(value: Any, *, rank: int, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (rank, rank) or not np.all(np.isfinite(matrix)):
        raise StructuredPriorError(f"{label} must be a finite {rank}x{rank} matrix")
    return matrix


def make_ally_relations(
    roles: Sequence[str],
    blocks: Mapping[tuple[str, str], Sequence[Sequence[float]]],
    *,
    rank: int = 2,
    tolerance: float = 1e-12,
) -> RoleRelations:
    """Validate ally blocks under ``M_sr = M_rs.T``."""

    canonical = _validate_roles(roles, rank)
    tolerance = _finite_positive(tolerance, name="tolerance")
    parsed: list[tuple[str, str, np.ndarray]] = []
    for left in canonical:
        for right in canonical:
            if (left, right) not in blocks:
                raise StructuredPriorError(f"missing ally block {left},{right}")
            matrix = _matrix(blocks[(left, right)], rank=rank, label="ally block")
            counterpart = _matrix(
                blocks.get((right, left)), rank=rank, label="ally counterpart"
            )
            if float(np.max(np.abs(matrix - counterpart.T))) > tolerance:
                raise StructuredPriorError("ally blocks must satisfy M_sr = M_rs.T")
            parsed.append((left, right, _immutable_array(matrix)))
    return RoleRelations("ally", canonical, rank, tuple(parsed))


def orient_enemy_relations(
    roles: Sequence[str],
    raw_blocks: Mapping[tuple[str, str], Sequence[Sequence[float]]],
    *,
    rank: int = 2,
) -> RoleRelations:
    """Project raw enemy blocks onto ``N_sr = -N_rs.T``.

    Same-role blocks become skew-symmetric.  Cross-role pairs use the
    antisymmetric part when both orientations are supplied, or the supplied
    orientation and its forced counterpart otherwise.
    """

    canonical = _validate_roles(roles, rank)
    unknown = set(raw_blocks) - {(left, right) for left in canonical for right in canonical}
    if unknown:
        raise StructuredPriorError(f"unknown enemy role blocks: {sorted(unknown)}")
    built: dict[tuple[str, str], np.ndarray] = {}
    for index, left in enumerate(canonical):
        diagonal = _matrix(
            raw_blocks.get((left, left), np.zeros((rank, rank))),
            rank=rank,
            label="enemy same-role block",
        )
        built[(left, left)] = 0.5 * (diagonal - diagonal.T)
        for right in canonical[index + 1 :]:
            forward_raw = raw_blocks.get((left, right))
            reverse_raw = raw_blocks.get((right, left))
            if forward_raw is None and reverse_raw is None:
                forward = np.zeros((rank, rank))
            elif forward_raw is None:
                forward = -_matrix(
                    reverse_raw, rank=rank, label="enemy reverse block"
                ).T
            elif reverse_raw is None:
                forward = _matrix(forward_raw, rank=rank, label="enemy forward block")
            else:
                forward = 0.5 * (
                    _matrix(forward_raw, rank=rank, label="enemy forward block")
                    - _matrix(reverse_raw, rank=rank, label="enemy reverse block").T
                )
            built[(left, right)] = forward
            built[(right, left)] = -forward.T
    frozen: list[tuple[str, str, np.ndarray]] = []
    for left in canonical:
        for right in canonical:
            frozen.append((left, right, _immutable_array(built[(left, right)])))
    relation = RoleRelations("enemy", canonical, rank, tuple(frozen))
    validate_enemy_relations(relation)
    return relation


def validate_enemy_relations(
    relation: RoleRelations, *, tolerance: float = 1e-12
) -> None:
    if relation.kind != "enemy":
        raise StructuredPriorError("expected enemy relations")
    tolerance = _finite_positive(tolerance, name="tolerance")
    for left in relation.roles:
        for right in relation.roles:
            violation = relation.block(left, right) + relation.block(right, left).T
            if float(np.max(np.abs(violation))) > tolerance:
                raise StructuredPriorError("enemy blocks must satisfy N_sr = -N_rs.T")


def _side_vectors(
    vectors: Mapping[str, Sequence[float]], relation: RoleRelations
) -> dict[str, np.ndarray]:
    if set(vectors) != set(relation.roles):
        raise StructuredPriorError("side vectors must contain exactly the manifested roles")
    parsed: dict[str, np.ndarray] = {}
    for role in relation.roles:
        vector = np.asarray(vectors[role], dtype=float)
        if vector.shape != (relation.rank,) or not np.all(np.isfinite(vector)):
            raise StructuredPriorError(
                f"{role} vector must be finite with shape ({relation.rank},)"
            )
        parsed[role] = vector
    return parsed


def score_ally_side(
    vectors: Mapping[str, Sequence[float]], relation: RoleRelations
) -> float:
    if relation.kind != "ally":
        raise StructuredPriorError("expected ally relations")
    parsed = _side_vectors(vectors, relation)
    total = 0.0
    for index, left in enumerate(relation.roles):
        for right in relation.roles[index + 1 :]:
            total += float(parsed[left] @ relation.block(left, right) @ parsed[right])
    return total


def score_enemy_ordered(
    side_a: Mapping[str, Sequence[float]],
    side_b: Mapping[str, Sequence[float]],
    relation: RoleRelations,
) -> float:
    validate_enemy_relations(relation)
    left_vectors = _side_vectors(side_a, relation)
    right_vectors = _side_vectors(side_b, relation)
    return float(
        sum(
            left_vectors[left] @ relation.block(left, right) @ right_vectors[right]
            for left in relation.roles
            for right in relation.roles
        )
    )


def score_side_swap_logit(
    side_a: Mapping[str, Sequence[float]],
    side_b: Mapping[str, Sequence[float]],
    *,
    ally: RoleRelations,
    enemy: RoleRelations,
    main_a: float = 0.0,
    main_b: float = 0.0,
) -> float:
    """Score A-vs-B so exchanging sides negates the complete logit."""

    if (
        isinstance(main_a, bool)
        or isinstance(main_b, bool)
        or not isinstance(main_a, (int, float, np.number))
        or not isinstance(main_b, (int, float, np.number))
    ):
        raise StructuredPriorError("main scores must be numeric")
    main_a = float(main_a)
    main_b = float(main_b)
    if not isfinite(main_a) or not isfinite(main_b):
        raise StructuredPriorError("main scores must be finite")
    return (
        main_a
        - main_b
        + score_ally_side(side_a, ally)
        - score_ally_side(side_b, ally)
        + score_enemy_ordered(side_a, side_b, enemy)
    )


@dataclass(frozen=True)
class PriorScaleDecision:
    block: str
    parent_block: str | None
    structural_effect_variance_upper_bound: float | None
    predictive_epistemic_variance: float | None
    mean: float | None
    structural_prior_identity_sha256: str | None
    evidence_identity_sha256: str
    available: bool
    evidence_exposure: str
    reason: str | None


@dataclass(frozen=True)
class OntologyEvidenceIdentity:
    """Time-safe identity for ontology reliability known before the target fit."""

    snapshot_sha256: str
    as_of: str
    reliability: float
    time_safe: bool = True
    ontology_dimensions_present: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_sha256, str) or not _SHA256_RE.fullmatch(
            self.snapshot_sha256
        ):
            raise StructuredPriorError("snapshot_sha256 must be lowercase SHA-256")
        try:
            parsed: datetime = parse_utc_timestamp(self.as_of)
        except (AttributeError, TypeError, ValueError) as exc:
            raise StructuredPriorError("ontology evidence as_of must be timezone-aware") from exc
        object.__setattr__(self, "as_of", canonical_timestamp(parsed))
        reliability = _finite_positive(
            self.reliability, name="ontology evidence reliability", allow_zero=True
        )
        if reliability > 1.0:
            raise StructuredPriorError("ontology evidence reliability must be in [0, 1]")
        object.__setattr__(self, "reliability", reliability)
        if self.time_safe is not True:
            raise StructuredPriorError("ontology evidence must be time-safe")
        if not isinstance(self.ontology_dimensions_present, bool):
            raise StructuredPriorError("ontology_dimensions_present must be boolean")

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": "l6-ontology-evidence-identity-v1",
            "snapshot_sha256": self.snapshot_sha256,
            "as_of": self.as_of,
            "reliability": self.reliability,
            "time_safe": self.time_safe,
            "ontology_dimensions_present": self.ontology_dimensions_present,
        }

    @property
    def payload_sha256(self) -> str:
        return canonical_sha256(self.to_payload())


def prior_scale(
    block: str,
    *,
    hierarchy: str = "global",
    evidence_identity: OntologyEvidenceIdentity | None,
    same_fit_support: int | None = None,
    spec: StructuredPriorSpec = PRINCIPAL_PRIOR_SPEC,
) -> PriorScaleDecision:
    """Return fixed structural regularization and separate predictive uncertainty."""

    if evidence_identity is None:
        raise StructuredPriorError("a time-safe ontology evidence identity is required")
    if not isinstance(evidence_identity, OntologyEvidenceIdentity):
        raise StructuredPriorError("invalid ontology evidence identity")
    base_sd = spec.block_sd(block)
    multiplier = spec.hierarchy_multiplier(hierarchy)
    reliability = evidence_identity.reliability
    if same_fit_support is not None and (
        isinstance(same_fit_support, bool)
        or not isinstance(same_fit_support, (int, np.integer))
        or same_fit_support < 0
    ):
        raise StructuredPriorError("same_fit_support must be a nonnegative integer")
    support = None if same_fit_support is None else int(same_fit_support)
    exposure = (
        "unknown"
        if support is None
        else ("zero" if support == 0 else ("sparse" if support < 5 else "observed"))
    )

    # Reliability is manifested before the target fit.  It may tighten, but
    # never widen, the structural coefficient prior.
    structural = (base_sd * multiplier * reliability) ** 2
    reliability_penalty = 1.0 / max(reliability, 0.05)
    evidence_penalty = (
        3.0
        if support is None
        else (2.0 if support == 0 else 1.0 + 1.0 / np.sqrt(support))
    )
    ontology_penalty = 1.0 if evidence_identity.ontology_dimensions_present else 4.0
    predictive = (
        (base_sd * multiplier) ** 2
        * reliability_penalty
        * evidence_penalty
        * ontology_penalty
    )

    reason: str | None = None
    available = True
    if block in spec.unavailable_blocks:
        available = False
        reason = "future interaction block is not estimated"
    elif not evidence_identity.ontology_dimensions_present or reliability == 0.0:
        available = False
        reason = "required ontology representation is unavailable"
    elif block == "champion_residual" and (support is None or support == 0):
        available = False
        reason = "exact champion residual evidence is absent or unknown"

    structural_identity = canonical_sha256(
        {
            "schema": "l6-structural-prior-identity-v1",
            "spec_sha256": spec.payload_sha256,
            "evidence_identity_sha256": evidence_identity.payload_sha256,
            "block": block,
            "hierarchy": hierarchy,
        }
    )
    estimand_mean: float | None = 0.0 if available else None
    estimand_structural: float | None = float(structural) if available else None
    estimand_predictive: float | None = float(predictive) if available else None

    return PriorScaleDecision(
        block=block,
        parent_block=_PARENTS[block],
        structural_effect_variance_upper_bound=estimand_structural,
        predictive_epistemic_variance=estimand_predictive,
        mean=estimand_mean,
        structural_prior_identity_sha256=structural_identity,
        evidence_identity_sha256=evidence_identity.payload_sha256,
        available=available,
        evidence_exposure=exposure,
        reason=reason,
    )
