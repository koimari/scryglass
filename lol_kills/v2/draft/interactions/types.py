"""Typed contracts for Wave-2 L6 draft interaction core.

The L6 scope is explicitly synthetic-development only: these dataclasses model
neutral five-versus-five composition structure and keep all projection and
decomposition artifacts explicit so they can be replayed and audited.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from math import isfinite
from typing import Any, Mapping

from lol_kills.v2.champions.schema import ROLES as CHAMPION_ROLES
from lol_kills.v2.evaluation.types import canonical_json
from lol_kills.v2.evaluation.types import canonical_sha256


CANONICAL_ROLES: tuple[str, ...] = tuple(CHAMPION_ROLES)
_PATCH_RE = re.compile(r"^[0-9]+\.[0-9]+$")
_CHAMPION_ID_RE = re.compile(r"^[A-Za-z0-9:_-]+$")


class DraftInteractionError(ValueError):
    """Raised when composition payloads, fit/evidence state, or outputs are invalid."""


def _ensure_payload_mapping(payload: Mapping[str, Any], *, label: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise DraftInteractionError(f"{label} must be a mapping")
    return payload


def validate_side_roles(side: Mapping[str, str]) -> None:
    if set(side.keys()) != set(CANONICAL_ROLES):
        raise DraftInteractionError(
            f"composition must contain exactly one champion for each role: {CANONICAL_ROLES}"
        )
    champion_ids = [side[role] for role in CANONICAL_ROLES]
    if any(
        not isinstance(champion_id, str)
        or not _CHAMPION_ID_RE.fullmatch(champion_id)
        for champion_id in champion_ids
    ):
        raise DraftInteractionError(
            "champion ids must use canonical collision-safe characters"
        )
    if len(set(champion_ids)) != len(champion_ids):
        raise DraftInteractionError("side composition cannot contain duplicate champions")


def normalize_side(side: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    validate_side_roles(side)
    return tuple((role, str(side[role]).strip()) for role in CANONICAL_ROLES)


def side_side_hash(side: Mapping[str, str]) -> str:
    payload = [{"role": role, "champion_id": str(side[role]).strip()} for role in CANONICAL_ROLES]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DraftCompositionRow:
    """Canonical neutral composition row with strict role-composition validation."""

    row_id: str
    patch_id: str
    league_id: str
    side_a: tuple[tuple[str, str], ...]
    side_b: tuple[tuple[str, str], ...]
    label: int
    source_id: str = "synth"
    source_patch_pool: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    competition_scope_id: str | None = None

    def __post_init__(self) -> None:
        normalized_a = normalize_side(dict(self.side_a))
        normalized_b = normalize_side(dict(self.side_b))
        overlap = {champion for _, champion in normalized_a} & {
            champion for _, champion in normalized_b
        }
        if overlap:
            raise DraftInteractionError(
                f"champions must be unique across the full composition: {sorted(overlap)}"
            )
        if not isinstance(self.patch_id, str) or not _PATCH_RE.fullmatch(
            self.patch_id
        ):
            raise DraftInteractionError("patch_id must be exactly numeric major.minor")
        object.__setattr__(self, "side_a", normalized_a)
        object.__setattr__(self, "side_b", normalized_b)

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "DraftCompositionRow":
        payload = _ensure_payload_mapping(payload, label="composition payload")
        row_id = payload.get("row_id")
        patch_id = payload.get("patch_id")
        league_id = payload.get("league_id")
        side_a = payload.get("side_a")
        side_b = payload.get("side_b")
        label = payload.get("label")
        source_id = payload.get("source_id", "synth")

        if not isinstance(row_id, str) or not row_id.strip():
            raise DraftInteractionError("row_id must be a non-empty string")
        if not isinstance(patch_id, str) or not _PATCH_RE.fullmatch(patch_id):
            raise DraftInteractionError("patch_id must be exactly numeric major.minor")
        if not isinstance(league_id, str) or not league_id.strip():
            raise DraftInteractionError("league_id must be a non-empty string")
        if not isinstance(side_a, Mapping):
            raise DraftInteractionError("side_a must be a mapping")
        if not isinstance(side_b, Mapping):
            raise DraftInteractionError("side_b must be a mapping")
        if label not in (0, 1):
            raise DraftInteractionError("label must be 0 or 1")
        if not isinstance(source_id, str) or not source_id.strip():
            raise DraftInteractionError("source_id must be a non-empty string")

        normalized_a = normalize_side(side_a)
        normalized_b = normalize_side(side_b)
        champions_a = {champion for _, champion in normalized_a}
        champions_b = {champion for _, champion in normalized_b}
        overlap = champions_a & champions_b
        if overlap:
            raise DraftInteractionError(
                f"champions must be unique across the full composition: {sorted(overlap)}"
            )

        return cls(
            row_id=row_id,
            patch_id=patch_id,
            league_id=league_id,
            side_a=normalized_a,
            side_b=normalized_b,
            label=int(label),
            source_id=source_id,
            source_patch_pool=payload.get("source_patch_pool"),
            metadata=dict(payload.get("metadata", {})),
            competition_scope_id=payload.get("competition_scope_id"),
        )

    @staticmethod
    def validate(row: "DraftCompositionRow | Mapping[str, Any]") -> None:
        if isinstance(row, DraftCompositionRow):
            DraftCompositionRow.from_payload(row.to_payload())
            return
        if not isinstance(row, Mapping):
            raise DraftInteractionError("row must be DraftCompositionRow or mapping")
        DraftCompositionRow.from_payload(row)

    def to_payload(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "patch_id": self.patch_id,
            "league_id": self.league_id,
            "side_a": {role: champion_id for role, champion_id in self.side_a},
            "side_b": {role: champion_id for role, champion_id in self.side_b},
            "label": self.label,
            "source_id": self.source_id,
            "source_patch_pool": self.source_patch_pool,
            "metadata": dict(self.metadata),
            "competition_scope_id": self.competition_scope_id,
        }

    @property
    def side_a_hash(self) -> str:
        return side_side_hash({role: champ_id for role, champ_id in self.side_a})

    @property
    def side_b_hash(self) -> str:
        return side_side_hash({role: champ_id for role, champ_id in self.side_b})

    @property
    def payload_sha256(self) -> str:
        return canonical_sha256(self.to_payload())


@dataclass(frozen=True)
class DraftInteractionFitDiagnostics:
    """Diagnostics emitted by a fitted family.

    All values are deterministic by construction and can be used for fail-closed
    routing in downstream selection/selection-report steps.
    """

    family_id: str
    feature_count: int
    row_count: int
    fit_rank: int
    condition_number: float
    identification_status: str
    orthogonality: Mapping[str, float]
    fallback_term_count: int
    fallback_counts: Mapping[str, int]
    feature_block_counts: Mapping[str, int]
    draw_count: int
    covariance_seed: int
    warnings: tuple[str, ...] = ()
    collinearity_max_correlation: float = 0.0
    min_support: int = 0

    @property
    def as_payload(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "feature_count": self.feature_count,
            "row_count": self.row_count,
            "fit_rank": self.fit_rank,
            "condition_number": (
                self.condition_number
                if isfinite(self.condition_number)
                else None
            ),
            "identification_status": self.identification_status,
            "orthogonality": {
                key: value if isfinite(float(value)) else None
                for key, value in self.orthogonality.items()
            },
            "fallback_term_count": self.fallback_term_count,
            "fallback_counts": dict(self.fallback_counts),
            "feature_block_counts": dict(self.feature_block_counts),
            "draw_count": self.draw_count,
            "covariance_seed": self.covariance_seed,
            "warnings": list(self.warnings),
            "collinearity_max_correlation": self.collinearity_max_correlation,
            "min_support": self.min_support,
        }


@dataclass(frozen=True)
class DraftInteractionPrediction:
    """Public prediction payload for a neutral composition value row."""

    row_id: str
    raw_logit: float
    raw_probability: float
    lower_95: float | None
    upper_95: float | None
    decomposition_mode: str
    ledger: Mapping[str, Any]

    @property
    def as_payload(self) -> dict[str, Any]:
        if self.decomposition_mode != "identified":
            return {
                "row_id": self.row_id,
                "internal_development_value": self.raw_logit,
                "decomposition_mode": "total_only",
                "component_payload_available": False,
                "public_serving_status": "unavailable",
                "public_probability_authorized": False,
                "public_interval_authorized": False,
                "blockers": [
                    "identification predicates are not satisfied",
                    "synthetic development mechanics only",
                ],
                "ledger": dict(self.ledger),
            }
        return {
            "row_id": self.row_id,
            "raw_logit": self.raw_logit,
            "raw_probability": self.raw_probability,
            "lower_95": self.lower_95,
            "upper_95": self.upper_95,
            "decomposition_mode": self.decomposition_mode,
            "ledger": dict(self.ledger),
        }


@dataclass(frozen=True)
class DraftInteractionCandidate:
    """Candidate record used by selection and report artifacts."""

    family_id: str
    status: str
    metrics: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    candidate_sha256: str
    selected: bool = False

    @property
    def as_payload(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "status": self.status,
            "metrics": dict(self.metrics),
            "diagnostics": dict(self.diagnostics),
            "candidate_sha256": self.candidate_sha256,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class DraftInteractionSelectionReport:
    """Development selection summary for all L6 candidate families."""

    config_sha256: str
    candidate_count: int
    candidates: tuple[DraftInteractionCandidate, ...]
    selection_status: str
    selected_family: str | None
    selected_sha256: str | None
    selection_sha256: str

    @property
    def as_payload(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "candidate_count": self.candidate_count,
            "selection_status": self.selection_status,
            "selected_family": self.selected_family,
            "selected_candidate_sha256": self.selected_sha256,
            "selection_sha256": self.selection_sha256,
            "candidates": [candidate.as_payload for candidate in self.candidates],
        }


@dataclass(frozen=True)
class DraftInteractionFit:
    """A fitted and replayable L6 family fit."""

    family_id: str
    coefficients: Mapping[str, float]
    feature_terms: tuple[str, ...]
    raw_feature_terms: tuple[str, ...]
    transform_matrix: tuple[tuple[float, ...], ...]
    transform_sha256: str
    reference_sha256: str
    identification_proof_sha256: str | None
    term_metadata: tuple[tuple[str, Mapping[str, Any]], ...]
    orthogonalization: tuple[Mapping[str, Any], ...]
    diagnostics: DraftInteractionFitDiagnostics
    supports: Mapping[str, int]
    decomposition_mode: str
    covariance_diag: tuple[float, ...]
    covariance_factor: tuple[tuple[float, ...], ...]
    covariance_seed: int
    draw_count: int
    selection_tag: str
    raw_rows: tuple[DraftCompositionRow, ...] = field(default_factory=tuple)

    @property
    def as_payload(self) -> dict[str, Any]:
        identified = (
            self.diagnostics.identification_status == "identified"
            and self.identification_proof_sha256 is not None
        )
        if not identified:
            return {
                "family_id": self.family_id,
                "decomposition_mode": "total_only",
                "diagnostics": self.diagnostics.as_payload,
                "draw_count": self.draw_count,
                "covariance_seed": self.covariance_seed,
                "selection_tag": self.selection_tag,
                "reference_sha256": self.reference_sha256,
                "transform_sha256": self.transform_sha256,
                "component_payload_available": False,
                "row_count": len(self.raw_rows),
            }
        return {
            "family_id": self.family_id,
            "coefficients": dict(self.coefficients),
            "feature_terms": list(term for term in self.feature_terms),
            "raw_feature_terms": list(self.raw_feature_terms),
            "transform_matrix": [list(row) for row in self.transform_matrix],
            "transform_sha256": self.transform_sha256,
            "reference_sha256": self.reference_sha256,
            "identification_proof_sha256": self.identification_proof_sha256,
            "term_metadata": [
                {"term_id": term, **metadata}
                for term, metadata in self.term_metadata
            ],
            "orthogonalization": [dict(item) for item in self.orthogonalization],
            "diagnostics": self.diagnostics.as_payload,
            "supports": dict(self.supports),
            "decomposition_mode": self.decomposition_mode,
            "covariance_diag": list(self.covariance_diag),
            "covariance_factor": [list(row) for row in self.covariance_factor],
            "covariance_seed": self.covariance_seed,
            "draw_count": self.draw_count,
            "selection_tag": self.selection_tag,
            "row_count": len(self.raw_rows),
        }
