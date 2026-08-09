"""Sparse joint pooled estimator for champion tier-list maps.

This module is an isolated model boundary.  It does not publish a tier list,
change the existing estimator, or claim production readiness.

One input map produces one Bernoulli row.  The row contains all role
contrasts, the registered antisymmetric atom-pair features, the scope and
OE-patch low-dimensional atom deviations, and canonical same-role pair
residuals.  These terms enter one logistic linear predictor.

The fit uses reference-coded champion strengths.  A global role/champion
coefficient is pooled across scopes.  A scope-role deviation has its own
proper Gaussian prior.  The atom coefficient is shared across scopes and
patches.  Scope-role and OE-patch-role latent deviations are also Gaussian
regularised.  Pair residuals are keyed by an unordered champion pair and use
the orientation sign of the supplied blue-minus-red observation.

The Laplace sampler stores only the diagonal of the observed negative
log-posterior Hessian.  It never allocates a parameter-by-parameter dense
covariance matrix.  The approximation is recorded in the returned metadata.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit


DEFAULT_ROLES: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")
MODEL_SCHEMA_ID = "scryglass.tierlists.joint-pooled-model.v1"
MODEL_SCHEMA_VERSION = "joint-pooled-model-v1"


class JointPooledModelError(ValueError):
    """Raised when a joint pooled fit cannot be constructed safely."""


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise JointPooledModelError(f"{label} must be a finite float")
    result = float(value)
    if not math.isfinite(result):
        raise JointPooledModelError(f"{label} must be a finite float")
    return result


def _positive_float(value: object, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0.0:
        raise JointPooledModelError(f"{label} must be positive")
    return result


@dataclass(frozen=True)
class RegisteredAtomFeature:
    """One feature in the fixed registered atom vocabulary."""

    name: str
    source: str = "registered_atom_source"
    registered: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise JointPooledModelError("registered atom feature names must be non-empty strings")
        if not isinstance(self.source, str) or not self.source.strip():
            raise JointPooledModelError("registered atom feature sources must be non-empty strings")
        if self.registered is not True:
            raise JointPooledModelError(
                f"feature {self.name!r} is not registered; omit it from the model registry"
            )


@dataclass(frozen=True)
class AtomFeatureRegistry:
    """Immutable feature registry used to validate every map vector."""

    features: tuple[RegisteredAtomFeature, ...]
    schema_id: str = "scryglass.tierlists.registered-atom-features.v1"

    def __post_init__(self) -> None:
        if not self.features:
            raise JointPooledModelError("the atom feature registry must contain one feature")
        names = [feature.name for feature in self.features]
        if len(names) != len(set(names)):
            raise JointPooledModelError("atom feature names must be unique")
        if not isinstance(self.schema_id, str) or not self.schema_id.strip():
            raise JointPooledModelError("atom feature registry schema_id must be non-empty")

    @classmethod
    def from_names(
        cls,
        names: Sequence[str],
        *,
        source: str = "registered_atom_source",
        schema_id: str = "scryglass.tierlists.registered-atom-features.v1",
    ) -> "AtomFeatureRegistry":
        return cls(
            features=tuple(RegisteredAtomFeature(name=name, source=source) for name in names),
            schema_id=schema_id,
        )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(feature.name for feature in self.features)

    @property
    def sources(self) -> dict[str, str]:
        return {feature.name: feature.source for feature in self.features}

    @property
    def sha256(self) -> str:
        payload = {
            "schema_id": self.schema_id,
            "features": [
                {"name": feature.name, "source": feature.source, "registered": feature.registered}
                for feature in self.features
            ],
        }
        encoded = _canonical_json(payload).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_sha256": self.sha256,
            "features": [
                {"name": feature.name, "source": feature.source, "registered": feature.registered}
                for feature in self.features
            ],
            "availability_policy": {
                "row_values": "ordered by registry feature names",
                "row_mask": "required explicit boolean per feature",
                "missing_value": "None",
                "missing_policy": "omit unavailable design entries; no zero imputation",
            },
        }


@dataclass(frozen=True)
class AtomFeatureVector:
    """An ordered blue-minus-red atom vector with an explicit mask.

    Values are already antisymmetric pair features.  Reversing the map must
    negate available values and preserve the availability mask.
    """

    values: tuple[float | None, ...]
    available: tuple[bool, ...]

    @classmethod
    def from_values(
        cls,
        values: Sequence[float | None],
        *,
        available: Sequence[bool] | None = None,
    ) -> "AtomFeatureVector":
        raw_values = tuple(values)
        if available is None:
            raw_available = tuple(value is not None for value in raw_values)
        else:
            raw_available = tuple(available)
        if len(raw_values) != len(raw_available):
            raise JointPooledModelError("atom values and availability must have equal length")
        return cls(values=raw_values, available=raw_available)

    def reversed(self) -> "AtomFeatureVector":
        values = tuple(None if value is None else -float(value) for value in self.values)
        return AtomFeatureVector(values=values, available=self.available)


@dataclass(frozen=True)
class JointMapObservation:
    """One completed map and its role-wise ordered atom pair vectors.

    Real observations must provide ``offset`` and ``weight``.  The defaults
    are available only when ``synthetic`` is explicitly true.  This keeps
    existing synthetic tests concise while preventing a production adapter
    from silently dropping team-strength control or temporal weighting.
    """

    map_id: str
    outcome: int
    scope_id: str
    oe_patch_id: str
    picks: Mapping[str, tuple[str, str]]
    atom_pair_features: Mapping[str, AtomFeatureVector]
    series_id: str | None = None
    offset: float | None = None
    weight: float | None = None
    synthetic: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.synthetic, (bool, np.bool_)):
            raise JointPooledModelError("synthetic must be boolean")
        if self.offset is None:
            if not self.synthetic:
                raise JointPooledModelError("offset is required for non-synthetic map observations")
            object.__setattr__(self, "offset", 0.0)
        if self.weight is None:
            if not self.synthetic:
                raise JointPooledModelError("weight is required for non-synthetic map observations")
            object.__setattr__(self, "weight", 1.0)
        object.__setattr__(self, "offset", _finite_float(self.offset, "offset"))
        object.__setattr__(self, "weight", _positive_float(self.weight, "weight"))

    @classmethod
    def from_mapping(
        cls,
        row: Mapping[str, Any],
        *,
        registry: AtomFeatureRegistry,
    ) -> "JointMapObservation":
        """Adapt a plain map row without silently filling missing features."""

        if not isinstance(row, Mapping):
            raise JointPooledModelError("map rows must be mappings or JointMapObservation objects")
        picks_raw = row.get("picks", row.get("roles"))
        atom_raw = row.get("atom_pair_features", row.get("atom_features"))
        if not isinstance(picks_raw, Mapping):
            raise JointPooledModelError("map row requires a picks mapping")
        if not isinstance(atom_raw, Mapping):
            raise JointPooledModelError("map row requires an atom_pair_features mapping")

        picks: dict[str, tuple[str, str]] = {}
        for role, value in picks_raw.items():
            if isinstance(value, Mapping):
                blue = value.get("blue", value.get("blue_champion"))
                red = value.get("red", value.get("red_champion"))
                if blue is None or red is None:
                    raise JointPooledModelError(f"map row role {role!r} needs blue and red champions")
                picks[str(role)] = (str(blue), str(red))
                continue
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
                raise JointPooledModelError(f"map row role {role!r} needs a two-item pick pair")
            picks[str(role)] = (str(value[0]), str(value[1]))

        atom_pair_features: dict[str, AtomFeatureVector] = {}
        for role, value in atom_raw.items():
            role_name = str(role)
            if isinstance(value, AtomFeatureVector):
                atom_pair_features[role_name] = value
                continue
            if not isinstance(value, Mapping):
                raise JointPooledModelError(f"map row atom vector for role {role!r} is invalid")
            if "values" in value or "available" in value:
                if "values" not in value or "available" not in value:
                    raise JointPooledModelError(
                        f"map row atom vector for role {role!r} needs values and available"
                    )
                atom_pair_features[role_name] = AtomFeatureVector.from_values(
                    value["values"], available=value["available"]
                )
                continue
            if set(value) != set(registry.names):
                raise JointPooledModelError(
                    f"map row atom vector for role {role!r} must use the registered feature names"
                )
            atom_pair_features[role_name] = AtomFeatureVector.from_values(
                tuple(value[name] for name in registry.names)
            )

        patch = row.get("oe_patch_id", row.get("patch_id"))
        if patch is None:
            raise JointPooledModelError("map row requires oe_patch_id or patch_id")
        synthetic = row.get("synthetic", row.get("test_row", False))
        return cls(
            map_id=str(row.get("map_id", "")),
            outcome=int(row.get("outcome")),
            scope_id=str(row.get("scope_id", row.get("league", ""))),
            oe_patch_id=str(patch),
            picks=picks,
            atom_pair_features=atom_pair_features,
            series_id=(str(row["series_id"]) if row.get("series_id") is not None else None),
            offset=row.get("offset"),
            weight=row.get("weight"),
            synthetic=synthetic,
        )


@dataclass(frozen=True)
class PriorScales:
    """Proper Gaussian prior standard deviations for every parameter family."""

    global_strength_sd: float = 1.0
    scope_strength_deviation_sd: float = 0.50
    global_atom_pair_sd: float = 0.50
    scope_atom_deviation_sd: float = 0.25
    oe_patch_atom_deviation_sd: float = 0.25
    pair_residual_sd: float = 0.50

    def as_dict(self) -> dict[str, float]:
        return {
            "global_strength": float(self.global_strength_sd),
            "scope_strength_deviation": float(self.scope_strength_deviation_sd),
            "global_atom_pair": float(self.global_atom_pair_sd),
            "scope_atom_deviation": float(self.scope_atom_deviation_sd),
            "oe_patch_atom_deviation": float(self.oe_patch_atom_deviation_sd),
            "scope_role_pair_residual": float(self.pair_residual_sd),
        }

    def validate(self) -> None:
        for family, value in self.as_dict().items():
            if not math.isfinite(value) or value <= 0.0:
                raise JointPooledModelError(f"prior standard deviation for {family} must be positive and finite")


@dataclass(frozen=True)
class JointPooledFit:
    """Fit result and stable downstream adapter surface."""

    parameter_names: tuple[str, ...]
    fitted_parameters: Mapping[str, float]
    coefficients: np.ndarray
    covariance_diagonal: np.ndarray
    design_matrix: sparse.csr_matrix
    design_metadata: Mapping[str, Any]
    pair_posteriors: Mapping[str, Mapping[str, Any]]
    map_predictions: tuple[Mapping[str, Any], ...]
    scope_role_champion_strengths: Mapping[str, Mapping[str, Mapping[str, float]]]
    atom_basis: np.ndarray
    metadata: Mapping[str, Any]

    def sample_posterior(
        self,
        *,
        posterior_draws: int = 2000,
        seed: int = 0,
        chunk_size: int = 256,
    ) -> np.ndarray:
        """Draw complete parameter vectors from the recorded diagonal Laplace fit."""

        if isinstance(posterior_draws, bool) or not isinstance(posterior_draws, int):
            raise JointPooledModelError("posterior_draws must be an integer")
        if posterior_draws < 1:
            raise JointPooledModelError("posterior_draws must be positive")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise JointPooledModelError("chunk_size must be a positive integer")
        coefficients = np.asarray(self.coefficients, dtype=float)
        variance = np.asarray(self.covariance_diagonal, dtype=float)
        if coefficients.ndim != 1 or variance.shape != coefficients.shape:
            raise JointPooledModelError("posterior location and diagonal covariance shapes do not match")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(variance)):
            raise JointPooledModelError("posterior location and diagonal covariance must be finite")
        if np.any(variance <= 0.0):
            raise JointPooledModelError("posterior diagonal covariance must be positive")

        draws = np.concatenate(
            tuple(self.iter_posterior(
                posterior_draws=posterior_draws,
                seed=seed,
                chunk_size=chunk_size,
            )),
            axis=0,
        )
        if not np.all(np.isfinite(draws)):
            raise JointPooledModelError("posterior sampler produced a non-finite draw")
        return draws

    def iter_posterior(
        self,
        *,
        posterior_draws: int = 2000,
        seed: int = 0,
        chunk_size: int = 256,
    ) -> Iterator[np.ndarray]:
        """Yield diagonal-Laplace posterior draws in bounded memory chunks."""

        if isinstance(posterior_draws, bool) or not isinstance(posterior_draws, int):
            raise JointPooledModelError("posterior_draws must be an integer")
        if posterior_draws < 1:
            raise JointPooledModelError("posterior_draws must be positive")
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
            raise JointPooledModelError("chunk_size must be a positive integer")
        coefficients = np.asarray(self.coefficients, dtype=float)
        variance = np.asarray(self.covariance_diagonal, dtype=float)
        if coefficients.ndim != 1 or variance.shape != coefficients.shape:
            raise JointPooledModelError("posterior location and diagonal covariance shapes do not match")
        if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(variance)):
            raise JointPooledModelError("posterior location and diagonal covariance must be finite")
        if np.any(variance <= 0.0):
            raise JointPooledModelError("posterior diagonal covariance must be positive")
        rng = np.random.default_rng(seed)
        standard_deviation = np.sqrt(variance)
        for start in range(0, posterior_draws, chunk_size):
            stop = min(posterior_draws, start + chunk_size)
            chunk = rng.standard_normal((stop - start, coefficients.size))
            chunk *= standard_deviation
            chunk += coefficients
            if not np.all(np.isfinite(chunk)):
                raise JointPooledModelError("posterior sampler produced a non-finite draw")
            yield chunk

    @property
    def posterior_sampler(self):
        """Expose the sampler as a callable for adapters that prefer a function."""

        return self.sample_posterior


@dataclass(frozen=True)
class _Column:
    name: str
    family: str
    prior_sd: float
    role: str | None = None
    scope_id: str | None = None
    oe_patch_id: str | None = None
    champion: str | None = None
    pair: tuple[str, str] | None = None
    feature: str | None = None
    latent_index: int | None = None

    def metadata(self, index: int) -> dict[str, Any]:
        return {
            "index": index,
            "name": self.name,
            "family": self.family,
            "prior_sd": self.prior_sd,
            "role": self.role,
            "scope_id": self.scope_id,
            "oe_patch_id": self.oe_patch_id,
            "champion": self.champion,
            "pair": list(self.pair) if self.pair is not None else None,
            "feature": self.feature,
            "latent_index": self.latent_index,
        }


def fit_joint_pooled_model(
    maps: Sequence[JointMapObservation | Mapping[str, Any]],
    *,
    feature_registry: AtomFeatureRegistry,
    roles: Sequence[str] | None = None,
    atom_deviation_dim: int = 2,
    pair_min_observations: int = 3,
    priors: PriorScales | None = None,
    max_iter: int = 1000,
) -> JointPooledFit:
    """Fit the deterministic sparse joint pooled logistic MAP model.

    ``atom_pair_features[role]`` must be an ordered blue-minus-red vector
    from the registered feature resolver.  The function validates its mask
    and does not zero-impute unavailable entries.  A plain mapping row is
    accepted through the adapter on :class:`JointMapObservation`.
    """

    if not isinstance(feature_registry, AtomFeatureRegistry):
        raise JointPooledModelError("feature_registry must be an AtomFeatureRegistry")
    if isinstance(atom_deviation_dim, bool) or not isinstance(atom_deviation_dim, int):
        raise JointPooledModelError("atom_deviation_dim must be an integer")
    if atom_deviation_dim < 1:
        raise JointPooledModelError("atom_deviation_dim must be positive")
    if isinstance(pair_min_observations, bool) or not isinstance(pair_min_observations, int):
        raise JointPooledModelError("pair_min_observations must be an integer")
    if pair_min_observations < 1:
        raise JointPooledModelError("pair_min_observations must be positive")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
        raise JointPooledModelError("max_iter must be a positive integer")
    prior_scales = priors or PriorScales()
    if not isinstance(prior_scales, PriorScales):
        raise JointPooledModelError("priors must be a PriorScales object")
    prior_scales.validate()

    normalized = [
        row
        if isinstance(row, JointMapObservation)
        else JointMapObservation.from_mapping(row, registry=feature_registry)
        for row in maps
    ]
    if not normalized:
        raise JointPooledModelError("at least one map is required")
    normalized.sort(key=lambda row: row.map_id)
    _validate_maps(normalized, feature_registry, roles)
    role_order = tuple(roles) if roles is not None else tuple(sorted(normalized[0].picks))
    if not role_order:
        raise JointPooledModelError("at least one role is required")

    basis = _atom_basis(feature_registry.names, atom_deviation_dim)
    columns, lookup = _build_columns(
        normalized,
        role_order,
        feature_registry,
        atom_deviation_dim,
        prior_scales,
        pair_min_observations,
    )
    rows, row_metadata = _build_rows(
        normalized,
        role_order,
        feature_registry,
        basis,
        atom_deviation_dim,
        lookup,
    )
    n_maps = len(normalized)
    n_parameters = len(columns)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows):
        for column_index, value in sorted(row.items()):
            if value == 0.0:
                continue
            if not math.isfinite(value):
                raise JointPooledModelError("sparse design contains a non-finite value")
            row_indices.append(row_index)
            column_indices.append(column_index)
            values.append(value)
    design = sparse.csr_matrix(
        (np.asarray(values, dtype=float), (row_indices, column_indices)),
        shape=(n_maps, n_parameters),
        dtype=float,
    )
    design.sum_duplicates()
    outcomes = np.asarray([row.outcome for row in normalized], dtype=float)
    offsets = np.asarray([_finite_float(row.offset, f"map {row.map_id} offset") for row in normalized], dtype=float)
    temporal_weights = np.asarray(
        [_positive_float(row.weight, f"map {row.map_id} weight") for row in normalized],
        dtype=float,
    )
    precision = np.asarray([1.0 / (column.prior_sd ** 2) for column in columns], dtype=float)
    if not np.all(np.isfinite(precision)) or np.any(precision <= 0.0):
        raise JointPooledModelError("prior precision is not finite and positive")

    def objective_and_gradient(beta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = offsets + np.asarray(design @ beta, dtype=float)
        if not np.all(np.isfinite(eta)):
            raise JointPooledModelError("linear predictor became non-finite during optimization")
        negative_log_likelihood = np.dot(
            temporal_weights,
            np.logaddexp(0.0, eta) - outcomes * eta,
        )
        negative_log_prior = 0.5 * np.dot(precision, beta * beta)
        gradient = np.asarray(
            design.T @ (temporal_weights * (expit(eta) - outcomes)),
            dtype=float,
        ).reshape(-1)
        gradient += precision * beta
        value = float(negative_log_likelihood + negative_log_prior)
        if not math.isfinite(value) or not np.all(np.isfinite(gradient)):
            raise JointPooledModelError("objective became non-finite during optimization")
        return value, gradient

    initial = np.zeros(n_parameters, dtype=float)
    result = minimize(
        objective_and_gradient,
        initial,
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": max_iter,
            "ftol": 1e-12,
            "gtol": 1e-8,
            "maxls": 50,
        },
    )
    coefficients = np.asarray(result.x, dtype=float)
    if not result.success:
        raise JointPooledModelError(f"L-BFGS MAP optimization failed: {result.message}")
    if coefficients.shape != (n_parameters,) or not np.all(np.isfinite(coefficients)):
        raise JointPooledModelError("MAP coefficients are not finite")

    eta = offsets + np.asarray(design @ coefficients, dtype=float)
    probabilities = expit(eta)
    if not np.all(np.isfinite(eta)) or not np.all(np.isfinite(probabilities)):
        raise JointPooledModelError("MAP predictions are not finite")
    curvature = temporal_weights * probabilities * (1.0 - probabilities)
    squared_design = design.copy()
    squared_design.data **= 2
    hessian_diagonal = precision + np.asarray(squared_design.T @ curvature, dtype=float).reshape(-1)
    if not np.all(np.isfinite(hessian_diagonal)) or np.any(hessian_diagonal <= 0.0):
        raise JointPooledModelError("diagonal Laplace curvature is not finite and positive")
    covariance_diagonal = 1.0 / hessian_diagonal
    if not np.all(np.isfinite(covariance_diagonal)) or np.any(covariance_diagonal <= 0.0):
        raise JointPooledModelError("diagonal Laplace covariance is not finite and positive")

    fitted_parameters = {
        name: float(value) for name, value in zip((column.name for column in columns), coefficients)
    }
    column_by_name = {column.name: column for column in columns}
    pair_posteriors = _pair_posteriors(columns, coefficients, covariance_diagonal, design)
    scope_strengths = _scope_role_strengths(normalized, role_order, columns, fitted_parameters)
    map_predictions = _map_predictions(
        normalized,
        eta,
        probabilities,
        design,
        columns,
        coefficients,
    )

    row_metadata_with_csr = []
    for index, metadata in enumerate(row_metadata):
        start, stop = int(design.indptr[index]), int(design.indptr[index + 1])
        metadata_copy = dict(metadata)
        metadata_copy["csr_indices"] = [int(value) for value in design.indices[start:stop]]
        metadata_copy["csr_data"] = [float(value) for value in design.data[start:stop]]
        row_metadata_with_csr.append(metadata_copy)
    design_metadata = {
        "matrix_format": "scipy.sparse.csr_matrix",
        "shape": [int(value) for value in design.shape],
        "nnz": int(design.nnz),
        "indptr": [int(value) for value in design.indptr],
        "indices": [int(value) for value in design.indices],
        "data": [float(value) for value in design.data],
        "columns": [column.metadata(index) for index, column in enumerate(columns)],
        "rows": row_metadata_with_csr,
        "map_contribution_rule": {
            "rows_per_map": 1,
            "outcomes_per_map": 1,
            "role_outcomes": 0,
            "likelihood": "one Bernoulli outcome per completed map",
        },
        # This lookup is an in-process acceleration cache. It is not part of
        # the serialized evidence contract because the ordered column metadata
        # above remains the source of truth.
        "_parameter_lookup": lookup,
        "_feature_registry": feature_registry,
        "_known_scopes": {row.scope_id for row in normalized},
        "_known_patches": {row.oe_patch_id for row in normalized},
    }
    metadata = {
        "schema_id": MODEL_SCHEMA_ID,
        "schema_version": MODEL_SCHEMA_VERSION,
        "roles": list(role_order),
        "n_maps": n_maps,
        "n_parameters": n_parameters,
        "pair_residual_min_observations": pair_min_observations,
        "map_ids": [row.map_id for row in normalized],
        "outcome_count": n_maps,
        "outcomes_used_once": True,
        "offset_in_likelihood": True,
        "temporal_weight_in_likelihood": True,
        "likelihood": {
            "linear_predictor": "eta_i = offset_i + X_i beta",
            "objective": "sum_i weight_i * Bernoulli_negative_log_likelihood(eta_i, y_i) + Gaussian_prior_penalty(beta)",
            "hessian_diagonal": "prior_precision_j + sum_i weight_i * X_ij^2 * p_i * (1 - p_i)",
        },
        "feature_registry": feature_registry.metadata(),
        "atom_basis": {
            "construction": "deterministic signed hash bucket projection",
            "dimension": atom_deviation_dim,
            "feature_order": list(feature_registry.names),
            "matrix": basis.tolist(),
        },
        "partial_pooling": {
            "scope_role_champion_strength": "global role/champion coefficient plus scope-role deviation",
            "global_atom_pair": "one role-specific coefficient shared across scopes and OE patches",
            "global_role_pair_residual": "one role-specific canonical pair coefficient pooled across scopes",
            "scope_role_atom": "scope-role latent deviation around the shared global atom coefficient",
            "oe_patch_role_atom": "OE-patch-role latent deviation around the shared global atom coefficient",
            "scope_role_pair_residual": "canonical unordered pair coefficient within scope and role",
        },
        "symmetry": {
            "input_contract": "atom vectors are blue-minus-red antisymmetric features",
            "strength_contrast": "blue champion strength minus red champion strength",
            "pair_orientation": "+1 for canonical blue-to-red order and -1 after reversal",
            "side_reversal": "linear predictor changes sign and probability becomes one minus the original probability",
            "intercept": "none",
        },
        "priors": prior_scales.as_dict(),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "gradient_norm": float(np.linalg.norm(np.asarray(result.jac, dtype=float), ord=np.inf)),
        },
        "laplace": {
            "covariance_type": "diagonal",
            "covariance_storage": "one variance per parameter",
            "covariance_is_full": False,
            "off_diagonal_terms": "omitted by approximation",
            "finite_checks": True,
            "sampler": "independent normal draws from diagonal Laplace approximation",
        },
        "pair_posterior": {
            "posterior_scale": "diagonal Laplace marginal",
            "width": "95 percent normal half-width",
        },
        "not_production_ready": True,
    }
    # Keep this lookup in the fit path so accidental duplicate column names
    # fail early if a future adapter changes the naming scheme.
    if len(column_by_name) != len(columns):
        raise JointPooledModelError("parameter names are not unique")
    return JointPooledFit(
        parameter_names=tuple(column.name for column in columns),
        fitted_parameters=fitted_parameters,
        coefficients=coefficients,
        covariance_diagonal=covariance_diagonal,
        design_matrix=design,
        design_metadata=design_metadata,
        pair_posteriors=pair_posteriors,
        map_predictions=map_predictions,
        scope_role_champion_strengths=scope_strengths,
        atom_basis=basis,
        metadata=metadata,
    )


def sample_posterior(
    fit: JointPooledFit,
    *,
    posterior_draws: int = 2000,
    seed: int = 0,
    chunk_size: int = 256,
) -> np.ndarray:
    """Module-level adapter for complete parameter posterior draws."""

    if not isinstance(fit, JointPooledFit):
        raise JointPooledModelError("fit must be a JointPooledFit")
    return fit.sample_posterior(
        posterior_draws=posterior_draws,
        seed=seed,
        chunk_size=chunk_size,
    )


def design_vector_for_observation(
    fit: JointPooledFit,
    observation: JointMapObservation | Mapping[str, Any],
    *,
    allow_unseen_pairs: bool = False,
    validate: bool = True,
) -> sparse.csr_matrix:
    """Return the exact fitted-column CSR row for a hypothetical map.

    The returned row contains coefficient terms only.  The observation
    offset is retained by :func:`predict_linear_predictor`, because it is a
    known likelihood offset and is not a fitted coefficient column.
    Unknown scope, patch, champion, role, or canonical pair keys fail closed.
    """

    registry = _registry_from_fit(fit)
    normalized = _coerce_observation(observation, registry)
    row, _ = _strict_design_row(
        fit,
        normalized,
        registry,
        allow_unseen_pairs=allow_unseen_pairs,
        validate=validate,
    )
    return row


def predict_linear_predictor(
    fit: JointPooledFit,
    observation: JointMapObservation | Mapping[str, Any],
    coefficients: Sequence[float] | Mapping[str, float] | None = None,
) -> float:
    """Predict one map logit from a fitted or explicitly supplied coefficient vector."""

    registry = _registry_from_fit(fit)
    normalized = _coerce_observation(observation, registry)
    row, _ = _strict_design_row(fit, normalized, registry)
    coefficient_vector = _coefficient_vector(fit, coefficients)
    linear_predictor = float(normalized.offset) + (row @ coefficient_vector).item()
    if not math.isfinite(linear_predictor):
        raise JointPooledModelError("hypothetical map linear predictor is not finite")
    return linear_predictor


def _registry_from_fit(fit: JointPooledFit) -> AtomFeatureRegistry:
    if not isinstance(fit, JointPooledFit):
        raise JointPooledModelError("fit must be a JointPooledFit")
    cached = fit.design_metadata.get("_feature_registry")
    if isinstance(cached, AtomFeatureRegistry):
        return cached
    payload = fit.metadata.get("feature_registry")
    if not isinstance(payload, Mapping):
        raise JointPooledModelError("fit is missing feature registry metadata")
    raw_features = payload.get("features")
    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        raise JointPooledModelError("fit feature registry metadata is invalid")
    features: list[RegisteredAtomFeature] = []
    for raw_feature in raw_features:
        if not isinstance(raw_feature, Mapping):
            raise JointPooledModelError("fit feature registry entry is invalid")
        features.append(
            RegisteredAtomFeature(
                name=str(raw_feature.get("name", "")),
                source=str(raw_feature.get("source", "")),
                registered=raw_feature.get("registered", False),
            )
        )
    registry = AtomFeatureRegistry(
        features=tuple(features),
        schema_id=str(payload.get("schema_id", "")),
    )
    if payload.get("schema_sha256") != registry.sha256:
        raise JointPooledModelError("fit feature registry metadata hash does not match its entries")
    return registry


def _coerce_observation(
    observation: JointMapObservation | Mapping[str, Any],
    registry: AtomFeatureRegistry,
) -> JointMapObservation:
    if isinstance(observation, JointMapObservation):
        return observation
    if isinstance(observation, Mapping):
        return JointMapObservation.from_mapping(observation, registry=registry)
    raise JointPooledModelError("observation must be a JointMapObservation or mapping")


def _strict_design_row(
    fit: JointPooledFit,
    observation: JointMapObservation,
    registry: AtomFeatureRegistry,
    *,
    allow_unseen_pairs: bool = False,
    validate: bool = True,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    roles = tuple(fit.metadata.get("roles", ()))
    if not roles:
        raise JointPooledModelError("fit is missing role metadata")
    if validate:
        _validate_maps([observation], registry, roles)
    columns = fit.design_metadata.get("columns")
    if not isinstance(columns, Sequence) or isinstance(columns, (str, bytes)):
        raise JointPooledModelError("fit is missing column metadata")
    cached_lookup = fit.design_metadata.get("_parameter_lookup")
    lookup = (
        cached_lookup
        if isinstance(cached_lookup, Mapping)
        else _lookup_from_column_metadata(fit, columns)
    )
    known_scopes = fit.design_metadata.get("_known_scopes")
    if not isinstance(known_scopes, set):
        known_scopes = {
            str(row.get("scope_id"))
            for row in fit.design_metadata.get("rows", ())
            if isinstance(row, Mapping) and row.get("scope_id") is not None
        }
    if observation.scope_id not in known_scopes:
        raise JointPooledModelError(f"unknown scope key: {observation.scope_id}")

    known_patches = fit.design_metadata.get("_known_patches")
    if not isinstance(known_patches, set):
        known_patches = {
            str(column.get("oe_patch_id"))
            for column in columns
            if isinstance(column, Mapping)
            and column.get("family") == "oe_patch_atom_deviation"
            and column.get("oe_patch_id") is not None
        }
    if observation.oe_patch_id not in known_patches:
        raise JointPooledModelError(f"unknown OE patch key: {observation.oe_patch_id}")

    scope_strengths = fit.scope_role_champion_strengths.get(observation.scope_id)
    if not isinstance(scope_strengths, Mapping):
        raise JointPooledModelError(f"unknown scope key: {observation.scope_id}")
    for role in roles:
        role_champions = scope_strengths.get(role)
        if not isinstance(role_champions, Mapping):
            raise JointPooledModelError(f"unknown scope-role key: {observation.scope_id}/{role}")
        for champion in observation.picks[role]:
            if champion not in role_champions:
                raise JointPooledModelError(
                    f"unknown champion key for {observation.scope_id}/{role}: {champion}"
                )
            if champion != min(role_champions):
                if ("global_strength", role, champion) not in lookup:
                    raise JointPooledModelError(f"unknown global champion key: {role}/{champion}")
                if ("scope_strength_deviation", observation.scope_id, role, champion) not in lookup:
                    raise JointPooledModelError(
                        f"unknown scope-role champion key: {observation.scope_id}/{role}/{champion}"
                    )
        blue, red = observation.picks[role]
        if blue != red:
            left, right = sorted((blue, red))
            global_pair_key = ("global_role_pair_residual", role, left, right)
            if global_pair_key not in lookup and not allow_unseen_pairs:
                raise JointPooledModelError(
                    f"unknown canonical pair key (global): {role}/{left}~{right}"
                )
            pair_key = ("scope_role_pair_residual", observation.scope_id, role, left, right)
            if pair_key not in lookup and not allow_unseen_pairs:
                raise JointPooledModelError(
                    f"unknown canonical pair key: {observation.scope_id}/{role}/{left}~{right}"
                )
        for latent_index in range(fit.atom_basis.shape[1]):
            patch_key = ("oe_patch_atom_deviation", observation.oe_patch_id, role, latent_index)
            scope_key = ("scope_atom_deviation", observation.scope_id, role, latent_index)
            if patch_key not in lookup:
                raise JointPooledModelError(
                    f"unknown OE patch-role key: {observation.oe_patch_id}/{role}"
                )
            if scope_key not in lookup:
                raise JointPooledModelError(
                    f"unknown scope-role atom key: {observation.scope_id}/{role}"
                )
        for feature in registry.names:
            if ("global_atom_pair", role, feature) not in lookup:
                raise JointPooledModelError(f"unknown global atom key: {role}/{feature}")

    rows, row_metadata = _build_rows(
        [observation],
        roles,
        registry,
        fit.atom_basis,
        fit.atom_basis.shape[1],
        lookup,
    )
    row_data = rows[0]
    row_indices = sorted(row_data)
    row_values = [float(row_data[index]) for index in row_indices]
    row = sparse.csr_matrix(
        (np.asarray(row_values, dtype=float), (np.zeros(len(row_indices), dtype=int), row_indices)),
        shape=(1, len(fit.parameter_names)),
        dtype=float,
    )
    if not np.all(np.isfinite(row.data)):
        raise JointPooledModelError("hypothetical map design row is not finite")
    return row, row_metadata[0]


def _lookup_from_column_metadata(
    fit: JointPooledFit,
    columns: Sequence[Any],
) -> dict[tuple[Any, ...], int]:
    if len(columns) != len(fit.parameter_names):
        raise JointPooledModelError("fit column metadata length does not match parameters")
    lookup: dict[tuple[Any, ...], int] = {}
    for expected_index, raw_column in enumerate(columns):
        if not isinstance(raw_column, Mapping):
            raise JointPooledModelError("fit column metadata entry is invalid")
        index = raw_column.get("index")
        name = raw_column.get("name")
        if index != expected_index or name != fit.parameter_names[expected_index]:
            raise JointPooledModelError("fit column metadata order does not match parameter order")
        family = raw_column.get("family")
        role = raw_column.get("role")
        scope_id = raw_column.get("scope_id")
        patch_id = raw_column.get("oe_patch_id")
        champion = raw_column.get("champion")
        feature = raw_column.get("feature")
        latent_index = raw_column.get("latent_index")
        if family == "global_strength":
            key = (family, role, champion)
        elif family == "scope_strength_deviation":
            key = (family, scope_id, role, champion)
        elif family == "global_atom_pair":
            key = (family, role, feature)
        elif family == "global_role_pair_residual":
            pair = raw_column.get("pair")
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise JointPooledModelError("fit global pair column metadata is invalid")
            key = (family, role, str(pair[0]), str(pair[1]))
        elif family == "scope_atom_deviation":
            key = (family, scope_id, role, latent_index)
        elif family == "oe_patch_atom_deviation":
            key = (family, patch_id, role, latent_index)
        elif family == "scope_role_pair_residual":
            pair = raw_column.get("pair")
            if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
                raise JointPooledModelError("fit pair column metadata is invalid")
            key = (family, scope_id, role, str(pair[0]), str(pair[1]))
        else:
            raise JointPooledModelError(f"unknown fit parameter family: {family}")
        if key in lookup:
            raise JointPooledModelError(f"duplicate fit parameter key: {key!r}")
        lookup[key] = int(index)
    return lookup


def _coefficient_vector(
    fit: JointPooledFit,
    coefficients: Sequence[float] | Mapping[str, float] | None,
) -> np.ndarray:
    if coefficients is None:
        result = np.asarray(fit.coefficients, dtype=float)
    elif isinstance(coefficients, Mapping):
        if set(coefficients) != set(fit.parameter_names):
            raise JointPooledModelError("coefficient mapping keys do not match fit parameters")
        result = np.asarray([coefficients[name] for name in fit.parameter_names], dtype=float)
    else:
        result = np.asarray(coefficients, dtype=float)
    if result.shape != (len(fit.parameter_names),) or not np.all(np.isfinite(result)):
        raise JointPooledModelError("coefficient vector must match the fit and be finite")
    return result


def _canonical_json(value: Any) -> str:
    return __import__("json").dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _safe_component(value: str) -> str:
    return str(value).replace("%", "%25").replace("|", "%7C").replace("=", "%3D")


def _parameter_name(family: str, **parts: str | int) -> str:
    encoded = "|".join(f"{key}={_safe_component(str(value))}" for key, value in parts.items())
    return f"{family}|{encoded}" if encoded else family


def _validate_maps(
    maps: Sequence[JointMapObservation],
    registry: AtomFeatureRegistry,
    roles: Sequence[str] | None,
) -> None:
    map_ids = [row.map_id for row in maps]
    if any(not map_id.strip() for map_id in map_ids):
        raise JointPooledModelError("map_id must be a non-empty string")
    if len(map_ids) != len(set(map_ids)):
        raise JointPooledModelError("each map must contribute exactly once; map_id is duplicated")
    if any(row.outcome not in (0, 1) for row in maps):
        raise JointPooledModelError("map outcome must be exactly 0 or 1")
    expected_roles = tuple(roles) if roles is not None else tuple(sorted(maps[0].picks))
    if len(expected_roles) != len(set(expected_roles)) or not expected_roles:
        raise JointPooledModelError("roles must be unique and non-empty")
    for row in maps:
        if not row.scope_id.strip() or not row.oe_patch_id.strip():
            raise JointPooledModelError("scope_id and oe_patch_id must be non-empty strings")
        _finite_float(row.offset, f"map {row.map_id} offset")
        _positive_float(row.weight, f"map {row.map_id} weight")
        if set(row.picks) != set(expected_roles):
            raise JointPooledModelError(f"map {row.map_id} does not contain exactly the registered roles")
        if set(row.atom_pair_features) != set(expected_roles):
            raise JointPooledModelError(
                f"map {row.map_id} does not contain one atom vector per registered role"
            )
        for role in expected_roles:
            pick_pair = row.picks[role]
            if not isinstance(pick_pair, Sequence) or len(pick_pair) != 2:
                raise JointPooledModelError(f"map {row.map_id} role {role} needs two champions")
            if any(not isinstance(champion, str) or not champion.strip() for champion in pick_pair):
                raise JointPooledModelError(f"map {row.map_id} role {role} has an invalid champion")
            vector = row.atom_pair_features[role]
            if not isinstance(vector, AtomFeatureVector):
                raise JointPooledModelError(f"map {row.map_id} role {role} atom vector is invalid")
            _validate_vector(vector, registry, f"map {row.map_id} role {role}")


def _validate_vector(vector: AtomFeatureVector, registry: AtomFeatureRegistry, label: str) -> None:
    if len(vector.values) != len(registry.features) or len(vector.available) != len(registry.features):
        raise JointPooledModelError(
            f"{label} must contain one value and one availability flag for every registered feature"
        )
    for index, (value, available) in enumerate(zip(vector.values, vector.available)):
        if not isinstance(available, (bool, np.bool_)):
            raise JointPooledModelError(f"{label} availability at index {index} must be boolean")
        if not available:
            if value is not None:
                raise JointPooledModelError(
                    f"{label} has a value for unavailable feature {registry.names[index]!r}"
                )
            continue
        if value is None or isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise JointPooledModelError(
                f"{label} available feature {registry.names[index]!r} must have a numeric value"
            )
        if not math.isfinite(float(value)):
            raise JointPooledModelError(f"{label} has a non-finite feature value")


def _atom_basis(feature_names: Sequence[str], dimension: int) -> np.ndarray:
    basis = np.zeros((len(feature_names), dimension), dtype=float)
    bucket_counts = np.zeros(dimension, dtype=int)
    for index, feature_name in enumerate(feature_names):
        digest = hashlib.sha256(f"{feature_name}|{index}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") % dimension
        sign = 1.0 if digest[8] % 2 == 0 else -1.0
        basis[index, bucket] = sign
        bucket_counts[bucket] += 1
    for bucket, count in enumerate(bucket_counts):
        if count:
            basis[:, bucket] /= math.sqrt(float(count))
    if not np.all(np.isfinite(basis)):
        raise JointPooledModelError("atom basis is not finite")
    return basis


def _latent_coordinates(
    vector: AtomFeatureVector,
    basis: np.ndarray,
) -> tuple[np.ndarray, tuple[bool, ...]]:
    values = np.asarray([0.0 if value is None else float(value) for value in vector.values], dtype=float)
    available = np.asarray(vector.available, dtype=bool)
    coordinates = np.zeros(basis.shape[1], dtype=float)
    latent_available: list[bool] = []
    for dimension in range(basis.shape[1]):
        support = np.flatnonzero(basis[:, dimension] != 0.0)
        is_available = support.size > 0 and bool(np.all(available[support]))
        latent_available.append(is_available)
        if is_available:
            coordinates[dimension] = float(np.dot(basis[support, dimension], values[support]))
    if not np.all(np.isfinite(coordinates)):
        raise JointPooledModelError("latent atom coordinates are not finite")
    return coordinates, tuple(latent_available)


def _build_columns(
    maps: Sequence[JointMapObservation],
    roles: Sequence[str],
    registry: AtomFeatureRegistry,
    atom_deviation_dim: int,
    priors: PriorScales,
    pair_min_observations: int,
) -> tuple[list[_Column], dict[tuple[Any, ...], int]]:
    columns: list[_Column] = []
    lookup: dict[tuple[Any, ...], int] = {}

    def add(key: tuple[Any, ...], column: _Column) -> None:
        if key in lookup:
            raise JointPooledModelError(f"duplicate parameter key: {key!r}")
        lookup[key] = len(columns)
        columns.append(column)

    champions_by_role = {
        role: sorted(
            {
                champion
                for row in maps
                for champion in row.picks[role]
            }
        )
        for role in roles
    }
    scope_role_champions = {
        (scope_id, role): sorted(
            {
                champion
                for row in maps
                if row.scope_id == scope_id
                for champion in row.picks[role]
            }
        )
        for scope_id in sorted({row.scope_id for row in maps})
        for role in roles
    }
    scope_roles = sorted(scope_role_champions)
    patch_roles = sorted({(row.oe_patch_id, role) for row in maps for role in roles})
    pair_counts = Counter(
        (row.scope_id, role, *sorted(row.picks[role]))
        for row in maps
        for role in roles
        if row.picks[role][0] != row.picks[role][1]
    )
    pair_keys = sorted(
        pair
        for pair, count in pair_counts.items()
        if count >= pair_min_observations
    )
    global_pair_counts = Counter(
        (role, *sorted(row.picks[role]))
        for row in maps
        for role in roles
        if row.picks[role][0] != row.picks[role][1]
    )
    global_pair_keys = sorted(
        pair
        for pair, count in global_pair_counts.items()
        if count >= pair_min_observations
    )

    # Reference coding removes the role-specific global and scope location
    # gauge.  The reference champion is still returned with strength zero.
    for role in roles:
        reference = champions_by_role[role][0]
        for champion in champions_by_role[role]:
            if champion == reference:
                continue
            add(
                ("global_strength", role, champion),
                _Column(
                    name=_parameter_name("global_strength", role=role, champion=champion),
                    family="global_strength",
                    prior_sd=priors.global_strength_sd,
                    role=role,
                    champion=champion,
                ),
            )
    for scope_id, role in scope_roles:
        reference = champions_by_role[role][0]
        for champion in scope_role_champions[(scope_id, role)]:
            if champion == reference:
                continue
            add(
                ("scope_strength_deviation", scope_id, role, champion),
                _Column(
                    name=_parameter_name(
                        "scope_strength_deviation", scope=scope_id, role=role, champion=champion
                    ),
                    family="scope_strength_deviation",
                    prior_sd=priors.scope_strength_deviation_sd,
                    role=role,
                    scope_id=scope_id,
                    champion=champion,
                ),
            )
    for role in roles:
        for feature in registry.features:
            add(
                ("global_atom_pair", role, feature.name),
                _Column(
                    name=_parameter_name("global_atom_pair", role=role, feature=feature.name),
                    family="global_atom_pair",
                    prior_sd=priors.global_atom_pair_sd,
                    role=role,
                    feature=feature.name,
                ),
            )
    for role, left, right in global_pair_keys:
        add(
            ("global_role_pair_residual", role, left, right),
            _Column(
                name=_parameter_name(
                    "global_role_pair_residual",
                    role=role,
                    pair=f"{left}~{right}",
                ),
                family="global_role_pair_residual",
                prior_sd=priors.pair_residual_sd,
                role=role,
                pair=(left, right),
            ),
        )
    for scope_id, role in scope_roles:
        for latent_index in range(atom_deviation_dim):
            add(
                ("scope_atom_deviation", scope_id, role, latent_index),
                _Column(
                    name=_parameter_name(
                        "scope_atom_deviation",
                        scope=scope_id,
                        role=role,
                        latent=latent_index,
                    ),
                    family="scope_atom_deviation",
                    prior_sd=priors.scope_atom_deviation_sd,
                    role=role,
                    scope_id=scope_id,
                    latent_index=latent_index,
                ),
            )
    for oe_patch_id, role in patch_roles:
        for latent_index in range(atom_deviation_dim):
            add(
                ("oe_patch_atom_deviation", oe_patch_id, role, latent_index),
                _Column(
                    name=_parameter_name(
                        "oe_patch_atom_deviation",
                        oe_patch=oe_patch_id,
                        role=role,
                        latent=latent_index,
                    ),
                    family="oe_patch_atom_deviation",
                    prior_sd=priors.oe_patch_atom_deviation_sd,
                    role=role,
                    oe_patch_id=oe_patch_id,
                    latent_index=latent_index,
                ),
            )
    for scope_id, role, left, right in pair_keys:
        add(
            ("scope_role_pair_residual", scope_id, role, left, right),
            _Column(
                name=_parameter_name(
                    "scope_role_pair_residual",
                    scope=scope_id,
                    role=role,
                    pair=f"{left}~{right}",
                ),
                family="scope_role_pair_residual",
                prior_sd=priors.pair_residual_sd,
                role=role,
                scope_id=scope_id,
                pair=(left, right),
            ),
        )
    if not columns:
        raise JointPooledModelError("the map set produced no model parameters")
    return columns, lookup


def _build_rows(
    maps: Sequence[JointMapObservation],
    roles: Sequence[str],
    registry: AtomFeatureRegistry,
    basis: np.ndarray,
    atom_deviation_dim: int,
    lookup: Mapping[tuple[Any, ...], int],
) -> tuple[list[dict[int, float]], list[dict[str, Any]]]:
    rows: list[dict[int, float]] = []
    metadata: list[dict[str, Any]] = []

    def add(row: dict[int, float], column_index: int | None, value: float) -> None:
        if column_index is None or value == 0.0:
            return
        if not math.isfinite(value):
            raise JointPooledModelError("a design entry is not finite")
        row[column_index] = row.get(column_index, 0.0) + float(value)

    for observation in maps:
        row: dict[int, float] = {}
        available_features: dict[str, list[str]] = {}
        available_latents: dict[str, list[int]] = {}
        for role in roles:
            blue, red = observation.picks[role]
            for champion, sign in ((blue, 1.0), (red, -1.0)):
                global_key = ("global_strength", role, champion)
                scope_key = ("scope_strength_deviation", observation.scope_id, role, champion)
                add(row, lookup.get(global_key), sign)
                add(row, lookup.get(scope_key), sign)

            vector = observation.atom_pair_features[role]
            coordinates, latent_mask = _latent_coordinates(vector, basis)
            feature_names = [
                feature_name
                for feature_name, is_available in zip(registry.names, vector.available)
                if is_available
            ]
            available_features[role] = feature_names
            available_latents[role] = [
                index for index, is_available in enumerate(latent_mask) if is_available
            ]
            for feature_index, feature_name in enumerate(registry.names):
                if vector.available[feature_index]:
                    value = vector.values[feature_index]
                    if value is None:
                        raise JointPooledModelError("available atom feature has no value")
                    add(row, lookup[("global_atom_pair", role, feature_name)], float(value))
            for latent_index in range(atom_deviation_dim):
                if latent_mask[latent_index]:
                    add(
                        row,
                        lookup[("scope_atom_deviation", observation.scope_id, role, latent_index)],
                        float(coordinates[latent_index]),
                    )
                    add(
                        row,
                        lookup[("oe_patch_atom_deviation", observation.oe_patch_id, role, latent_index)],
                        float(coordinates[latent_index]),
                    )
            if blue != red:
                left, right = sorted((blue, red))
                orientation = 1.0 if (blue, red) == (left, right) else -1.0
                add(
                    row,
                    lookup.get(("global_role_pair_residual", role, left, right)),
                    orientation,
                )
                add(
                    row,
                    lookup.get(("scope_role_pair_residual", observation.scope_id, role, left, right)),
                    orientation,
                )
        rows.append(row)
        metadata.append(
            {
                "row_index": len(rows) - 1,
                "map_id": observation.map_id,
                "outcome": observation.outcome,
                "scope_id": observation.scope_id,
                "oe_patch_id": observation.oe_patch_id,
                "series_id": observation.series_id,
                "offset": float(observation.offset),
                "weight": float(observation.weight),
                "roles": list(roles),
                "available_atom_features": available_features,
                "available_atom_latents": available_latents,
            }
        )
    return rows, metadata


def _pair_posteriors(
    columns: Sequence[_Column],
    coefficients: np.ndarray,
    covariance_diagonal: np.ndarray,
    design: sparse.csr_matrix,
) -> dict[str, Mapping[str, Any]]:
    posterior: dict[str, Mapping[str, Any]] = {}
    pair_map_counts = np.diff(design.tocsc().indptr)
    for index, column in enumerate(columns):
        if column.family not in {"global_role_pair_residual", "scope_role_pair_residual"}:
            continue
        sd = math.sqrt(float(covariance_diagonal[index]))
        width = 1.96 * sd
        count = int(pair_map_counts[index])
        posterior[column.name] = {
            "parameter": column.name,
            "scope_id": column.scope_id,
            "role": column.role,
            "pair": list(column.pair or ()),
            "mean": float(coefficients[index]),
            "sd": sd,
            "width": width,
            "lower_95": float(coefficients[index] - width),
            "upper_95": float(coefficients[index] + width),
            "maps": count,
            "covariance_type": "diagonal_laplace_marginal",
        }
    return posterior


def _scope_role_strengths(
    maps: Sequence[JointMapObservation],
    roles: Sequence[str],
    columns: Sequence[_Column],
    fitted_parameters: Mapping[str, float],
) -> dict[str, dict[str, dict[str, float]]]:
    global_values: dict[tuple[str, str], float] = {}
    scope_values: dict[tuple[str, str, str], float] = {}
    for column in columns:
        value = float(fitted_parameters[column.name])
        if column.family == "global_strength":
            global_values[(column.role or "", column.champion or "")] = value
        elif column.family == "scope_strength_deviation":
            scope_values[(column.scope_id or "", column.role or "", column.champion or "")] = value
    champions: dict[str, set[str]] = {role: set() for role in roles}
    for row in maps:
        for role in roles:
            champions[role].update(row.picks[role])
    result: dict[str, dict[str, dict[str, float]]] = {}
    for scope_id in sorted({row.scope_id for row in maps}):
        result[scope_id] = {}
        for role in roles:
            result[scope_id][role] = {}
            reference = min(champions[role])
            for champion in sorted(champions[role]):
                if champion == reference:
                    result[scope_id][role][champion] = 0.0
                    continue
                result[scope_id][role][champion] = float(
                    global_values.get((role, champion), 0.0)
                    + scope_values.get((scope_id, role, champion), 0.0)
                )
    return result


def _map_predictions(
    maps: Sequence[JointMapObservation],
    eta: np.ndarray,
    probabilities: np.ndarray,
    design: sparse.csr_matrix,
    columns: Sequence[_Column],
    coefficients: np.ndarray,
) -> tuple[Mapping[str, Any], ...]:
    predictions: list[Mapping[str, Any]] = []
    for row_index, observation in enumerate(maps):
        start, stop = int(design.indptr[row_index]), int(design.indptr[row_index + 1])
        component_totals = {
            "offset": float(observation.offset),
            "strength": 0.0,
            "global_atom_pair": 0.0,
            "global_role_pair_residual": 0.0,
            "scope_atom_deviation": 0.0,
            "oe_patch_atom_deviation": 0.0,
            "scope_role_pair_residual": 0.0,
        }
        for column_index, value in zip(
            design.indices[start:stop], design.data[start:stop]
        ):
            family = columns[int(column_index)].family
            contribution = float(value) * float(coefficients[int(column_index)])
            if family in ("global_strength", "scope_strength_deviation"):
                component_totals["strength"] += contribution
            elif family in component_totals:
                component_totals[family] += contribution
        predictions.append(
            {
                "map_id": observation.map_id,
                "outcome": observation.outcome,
                "scope_id": observation.scope_id,
                "oe_patch_id": observation.oe_patch_id,
                "offset": float(observation.offset),
                "weight": float(observation.weight),
                "linear_predictor": float(eta[row_index]),
                "probability": float(probabilities[row_index]),
                "design_row_index": row_index,
                "contributions": component_totals,
            }
        )
    return tuple(predictions)


__all__ = [
    "AtomFeatureRegistry",
    "AtomFeatureVector",
    "DEFAULT_ROLES",
    "JointMapObservation",
    "JointPooledFit",
    "JointPooledModelError",
    "MODEL_SCHEMA_ID",
    "MODEL_SCHEMA_VERSION",
    "PriorScales",
    "RegisteredAtomFeature",
    "design_vector_for_observation",
    "fit_joint_pooled_model",
    "predict_linear_predictor",
    "sample_posterior",
]
