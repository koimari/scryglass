"""Research-only tournament for team-strength-offset draft composition models.

This module evaluates whether role-aware champion composition adds predictive
information after a caller-supplied, strictly pre-event dynamic team-strength
logit.  It does not download data, mutate production artifacts, or make a
promotion decision.

The fitted candidate has the decomposition

    logit P(blue wins)
      = pre_event_team_strength_logit
      + league_side_nuisance_logit
      + raw_composition_logit.

The team-strength logit is a fixed offset in the likelihood.  The raw
composition term contains no organization, player, roster, league, or side
intercept and can therefore be evaluated at neutral team context.  Its signed
feature construction is exactly antisymmetric when the two five-champion sides
are swapped.

The tournament uses four UTC-date-group-safe chronological partitions:

* candidate hyperparameters are fit on train and compared on validation;
* one validation winner per feature family is refit on train + validation and
  compared on selection;
* the selected composition candidate is frozen before final labels are read,
  refit on all pre-final rows, and evaluated on final exactly once.

All result objects are immutable.  The final paired ledger retains the caller's
dependence identifier (normally a verified series identifier), so uncertainty
can be estimated with a paired series-cluster or moving-block procedure without
reconstructing event alignment.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import expit

TOURNAMENT_VERSION = "team-offset-draft-residual-v2"
ROLES = ("top", "jng", "mid", "bot", "sup")
SPLIT_NAMES = ("train", "validation", "selection", "final")
OFFSET_ONLY_BASELINE = "baseline_exact_pre_event_team_offset"
LEAGUE_SIDE_BASELINE = "baseline_team_offset_plus_league_side"
COMPOSITION_FAMILIES = frozenset({"additive", "synergy", "opposition"})
DRAFT_MODES = frozenset({"tournament_draft", "blind_pick"})
UNKNOWN_PATCH = "UNKNOWN"
_UNKNOWN_PATCH_VALUES = frozenset({"unknown", "unk", "n/a", "na", "none", "null"})
_MISSING_CHAMPION_VALUES = frozenset(
    {"unknown", "unk", "n/a", "na", "none", "null", "tbd"}
)


class DraftResidualTournamentError(ValueError):
    """Raised when the residual-tournament evidence contract is violated."""


@dataclass(frozen=True)
class PreparedDraftMap:
    """One already-prepared, complete map at the map-result grain.

    ``dependence_id`` should identify the smallest dependence cluster to
    preserve during inference, normally the containing series. Inferred
    clusters must be explicitly labeled by the caller.
    """

    event_id: str
    dependence_id: str
    event_time: pd.Timestamp
    league: str
    patch: str
    blue: tuple[tuple[str, str], ...]
    red: tuple[tuple[str, str], ...]
    y_blue_win: int
    draft_mode: str = "tournament_draft"


@dataclass(frozen=True)
class PreEventTeamLogit:
    """Immutable provenance for one externally produced team-strength offset.

    ``team_strength_only`` and ``includes_draft`` are enforceable interface
    assertions, not proof of the upstream model's contents.  A source audit is
    still required before production use.
    """

    event_id: str
    logit: float
    as_of: pd.Timestamp
    model_version: str
    provenance: str
    team_strength_only: bool = True
    includes_draft: bool = False
    includes_side: bool = False
    orientation: str = "blue_minus_red"
    scale: str = "natural_log_odds"
    dynamic: bool = True


@dataclass(frozen=True)
class CandidateSpec:
    """One predeclared sparse residual-model specification."""

    family: str
    l2: float
    min_support: int = 3
    include_patch_main: bool = False
    patch_l2_multiplier: float = 2.0

    def validate(self) -> None:
        if self.family not in COMPOSITION_FAMILIES:
            raise DraftResidualTournamentError(
                f"unknown composition family: {self.family!r}"
            )
        if not math.isfinite(self.l2) or self.l2 <= 0.0:
            raise DraftResidualTournamentError(
                "composition l2 must be finite and positive"
            )
        if self.min_support < 1:
            raise DraftResidualTournamentError("min_support must be positive")
        if (
            not math.isfinite(self.patch_l2_multiplier)
            or self.patch_l2_multiplier < 1.0
        ):
            raise DraftResidualTournamentError(
                "patch_l2_multiplier must be finite and at least one"
            )

    @property
    def components(self) -> tuple[str, ...]:
        if self.family == "additive":
            return ("main",)
        if self.family == "synergy":
            return ("main", "synergy")
        return ("main", "synergy", "opposition")

    @property
    def candidate_id(self) -> str:
        l2_text = f"{self.l2:.8g}".replace(".", "p")
        patch_l2_text = f"{self.patch_l2_multiplier:.8g}".replace(".", "p")
        patch = "patch" if self.include_patch_main else "global"
        return (
            f"offset_residual_{self.family}"
            f"__l2_{l2_text}"
            f"__support_{self.min_support}"
            f"__{patch}"
            f"__patch_l2_{patch_l2_text}"
        )


@dataclass(frozen=True)
class TournamentConfig:
    """Frozen gates and numerical settings for one tournament run."""

    primary_score: str = "log_loss"
    split_fractions: tuple[float, float, float, float] = (
        0.55,
        0.15,
        0.15,
        0.15,
    )
    split_end_dates: tuple[str, str, str] | None = None
    minimum_events_per_split: int = 20
    tie_tolerance: float = 1e-6
    ece_bins: int = 10
    global_side_l2: float = 0.0
    league_side_l2: float = 20.0
    maximum_iterations: int = 1_000
    optimizer_ftol: float = 1e-15
    optimizer_gtol: float = 1e-7
    maximum_gradient_inf_norm: float = 1e-4
    invariant_check_events: int = 16
    bootstrap_replicates: int = 5_000
    bootstrap_block_size: int = 12
    bootstrap_random_seed: int = 20_260_727
    bootstrap_alpha: float = 0.05
    calibration_methods: tuple[str, ...] = ("identity", "platt")
    calibration_primary_score: str = "log_loss"
    calibration_tie_tolerance: float = 1e-6
    calibration_l2_identity_centered: float = 1e-6
    calibration_min_slope: float = 0.05
    calibration_max_slope: float = 5.0
    calibration_max_abs_intercept: float = 8.0
    calibration_probability_floor: float = 1e-9
    candidate_specs: tuple[CandidateSpec, ...] = ()

    def validate(self) -> None:
        if self.primary_score not in {"log_loss", "brier"}:
            raise DraftResidualTournamentError(
                "primary_score must be 'log_loss' or 'brier'"
            )
        if (
            len(self.split_fractions) != 4
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in self.split_fractions
            )
            or not math.isclose(sum(self.split_fractions), 1.0, abs_tol=1e-12)
        ):
            raise DraftResidualTournamentError(
                "four positive split fractions summing to one are required"
            )
        if self.split_end_dates is not None:
            if len(self.split_end_dates) != 3:
                raise DraftResidualTournamentError(
                    "split_end_dates must contain train, validation, and "
                    "selection end dates"
                )
            parsed = tuple(
                _as_utc(
                    value,
                    field="split_end_date",
                    event_id="tournament-config",
                ).normalize()
                for value in self.split_end_dates
            )
            if not parsed[0] < parsed[1] < parsed[2]:
                raise DraftResidualTournamentError(
                    "split_end_dates must be strictly increasing"
                )
        if self.minimum_events_per_split < 1:
            raise DraftResidualTournamentError(
                "minimum_events_per_split must be positive"
            )
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise DraftResidualTournamentError(
                "tie_tolerance must be finite and non-negative"
            )
        if self.ece_bins < 2:
            raise DraftResidualTournamentError("ece_bins must be at least two")
        if (
            not math.isfinite(self.global_side_l2)
            or self.global_side_l2 < 0.0
            or not math.isfinite(self.league_side_l2)
            or self.league_side_l2 <= 0.0
        ):
            raise DraftResidualTournamentError(
                "side penalties must be finite; league_side_l2 must be positive"
            )
        if self.maximum_iterations < 1:
            raise DraftResidualTournamentError("maximum_iterations must be positive")
        if (
            not math.isfinite(self.optimizer_ftol)
            or self.optimizer_ftol <= 0.0
            or not math.isfinite(self.optimizer_gtol)
            or self.optimizer_gtol <= 0.0
            or not math.isfinite(self.maximum_gradient_inf_norm)
            or self.maximum_gradient_inf_norm <= 0.0
        ):
            raise DraftResidualTournamentError(
                "optimizer tolerances and the gradient acceptance bound "
                "must be finite and positive"
            )
        if self.invariant_check_events < 1:
            raise DraftResidualTournamentError(
                "invariant_check_events must be positive"
            )
        if self.bootstrap_replicates < 2:
            raise DraftResidualTournamentError(
                "bootstrap_replicates must be at least two"
            )
        if self.bootstrap_block_size < 1:
            raise DraftResidualTournamentError(
                "bootstrap_block_size must be positive"
            )
        if (
            not isinstance(self.bootstrap_random_seed, int)
            or isinstance(self.bootstrap_random_seed, bool)
            or self.bootstrap_random_seed < 0
        ):
            raise DraftResidualTournamentError(
                "bootstrap_random_seed must be a non-negative integer"
            )
        if (
            not math.isfinite(self.bootstrap_alpha)
            or not 0.0 < self.bootstrap_alpha < 1.0
        ):
            raise DraftResidualTournamentError(
                "bootstrap_alpha must be strictly between zero and one"
            )
        if (
            not self.calibration_methods
            or len(set(self.calibration_methods))
            != len(self.calibration_methods)
            or set(self.calibration_methods) != {"identity", "platt"}
        ):
            raise DraftResidualTournamentError(
                "calibration_methods must contain identity and platt exactly"
            )
        if self.calibration_primary_score not in {"log_loss", "brier"}:
            raise DraftResidualTournamentError(
                "calibration_primary_score must be log_loss or brier"
            )
        if (
            not math.isfinite(self.calibration_tie_tolerance)
            or self.calibration_tie_tolerance < 0.0
            or not math.isfinite(self.calibration_l2_identity_centered)
            or self.calibration_l2_identity_centered < 0.0
        ):
            raise DraftResidualTournamentError(
                "calibration tie tolerance and L2 must be finite and "
                "non-negative"
            )
        if (
            not math.isfinite(self.calibration_min_slope)
            or not math.isfinite(self.calibration_max_slope)
            or not 0.0 < self.calibration_min_slope
            < self.calibration_max_slope
        ):
            raise DraftResidualTournamentError(
                "calibration slope bounds are invalid"
            )
        if (
            not math.isfinite(self.calibration_max_abs_intercept)
            or self.calibration_max_abs_intercept <= 0.0
        ):
            raise DraftResidualTournamentError(
                "calibration_max_abs_intercept must be positive"
            )
        if (
            not math.isfinite(self.calibration_probability_floor)
            or not 0.0 < self.calibration_probability_floor < 0.5
        ):
            raise DraftResidualTournamentError(
                "calibration_probability_floor must be in (0, 0.5)"
            )


@dataclass(frozen=True)
class DateSplits:
    """Four chronological, UTC-date-group-safe map partitions."""

    train: tuple[PreparedDraftMap, ...]
    validation: tuple[PreparedDraftMap, ...]
    selection: tuple[PreparedDraftMap, ...]
    final: tuple[PreparedDraftMap, ...]

    def items(
        self,
    ) -> tuple[tuple[str, tuple[PreparedDraftMap, ...]], ...]:
        return (
            ("train", self.train),
            ("validation", self.validation),
            ("selection", self.selection),
            ("final", self.final),
        )


@dataclass(frozen=True)
class SplitSummary:
    name: str
    events: int
    dependence_clusters: int
    date_groups: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    event_ids: tuple[str, ...]


@dataclass(frozen=True)
class FitManifest:
    """Immutable fit provenance for one frozen candidate or nuisance model."""

    model_id: str
    model_version: str
    fit_stage: str
    training_event_ids: tuple[str, ...]
    training_event_digest: str
    offset_digest: str
    training_events: int
    trained_through: pd.Timestamp
    maximum_offset_as_of: pd.Timestamp


@dataclass(frozen=True)
class ResidualCompositionModel:
    """Frozen coefficients with context and raw-composition terms separated."""

    candidate_id: str
    family: str
    include_patch_main: bool
    nuisance_feature_names: tuple[str, ...]
    nuisance_coefficients: tuple[float, ...]
    composition_feature_names: tuple[str, ...]
    composition_coefficients: tuple[float, ...]
    known_champion_roles: tuple[str, ...]
    supported_champion_roles: tuple[str, ...]
    seen_patches: tuple[str, ...]
    seen_leagues: tuple[str, ...]
    l2: float
    min_support: int
    patch_l2_multiplier: float
    optimizer_iterations: int
    optimizer_gradient_inf_norm: float
    sparse_design: bool
    model_version: str


@dataclass(frozen=True)
class CompositionEstimate:
    """Raw composition estimate at neutral team and league/side context."""

    raw_composition_logit: float
    neutral_team_probability: float
    patch_status: str
    unknown_champion_roles: tuple[str, ...]
    unsupported_champion_roles: tuple[str, ...]


@dataclass(frozen=True)
class PredictionRow:
    """One immutable out-of-sample prediction and its exact decomposition."""

    prediction_id: str
    model_id: str
    model_version: str
    phase: str
    event_id: str
    dependence_id: str
    event_time: pd.Timestamp
    league: str
    patch: str
    outcome: int
    probability: float
    contextual_logit: float
    team_offset_logit: float
    team_offset_as_of: pd.Timestamp
    team_offset_model_version: str
    team_offset_provenance: str
    league_side_logit: float
    raw_composition_logit: float
    neutral_team_probability: float
    patch_status: str
    unknown_league: bool
    unknown_champion_roles: tuple[str, ...]
    unsupported_champion_roles: tuple[str, ...]
    trained_through: pd.Timestamp | None
    training_event_digest: str


@dataclass(frozen=True)
class ModelDiagnostics:
    model_id: str
    events: int
    log_loss: float
    brier: float
    ece: float
    event_rate: float
    mean_probability: float
    primary_score_name: str
    primary_score: float


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    family: str
    stage: str
    events: int
    log_loss: float
    brier: float
    ece: float
    primary_score_name: str
    primary_score: float
    complexity: int
    model_version: str


@dataclass(frozen=True)
class PairedLossRow:
    """Final-event seam for paired cluster/block uncertainty estimation."""

    event_id: str
    dependence_id: str
    event_time: pd.Timestamp
    outcome: int
    candidate_model_id: str
    candidate_probability: float
    offset_probability: float
    league_side_probability: float
    candidate_log_loss: float
    offset_log_loss: float
    league_side_log_loss: float
    candidate_brier: float
    offset_brier: float
    league_side_brier: float
    candidate_minus_offset_log_loss: float
    candidate_minus_league_side_log_loss: float
    candidate_minus_offset_brier: float
    candidate_minus_league_side_brier: float


@dataclass(frozen=True)
class PairedComparison:
    """Descriptive paired score difference; negative favors the candidate."""

    candidate_model_id: str
    baseline_model_id: str
    events: int
    dependence_clusters: int
    log_loss_delta: float
    brier_delta: float
    ece_delta: float


@dataclass(frozen=True)
class PairedBlockBootstrapResult:
    """Paired circular block-bootstrap interval for one proper-score delta."""

    candidate_model_id: str
    baseline_model_id: str
    metric: str
    events: int
    dependence_clusters: int
    dependence_status: str
    replicates: int
    requested_block_size: int
    effective_block_size: int
    block_rule: str
    random_seed: int
    alpha: float
    interval_method: str
    point_delta: float
    bootstrap_mean_delta: float
    bootstrap_standard_error: float
    interval_lower: float
    interval_upper: float
    bootstrap_distribution_sha256: str


@dataclass(frozen=True)
class CalibrationModel:
    """Frozen identity or antisymmetric Platt map selected before final."""

    method: str
    intercept: float
    slope: float
    l2_identity_centered: float
    probability_floor: float
    fit_split: str
    fit_events: int
    fit_event_digest: str
    fit_through: pd.Timestamp
    selected_on_split: str
    optimizer_iterations: int
    optimizer_gradient_inf_norm: float
    model_version: str

    def apply(
        self,
        probability: Sequence[float] | np.ndarray,
        *,
        side_indicator: Sequence[float] | np.ndarray | float = 1.0,
    ) -> np.ndarray:
        """Apply the frozen map without consulting outcomes."""

        raw = np.asarray(probability, dtype=np.float64)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise DraftResidualTournamentError(
                "calibration probabilities must be finite one-dimensional"
            )
        if np.any((raw <= 0.0) | (raw >= 1.0)):
            raise DraftResidualTournamentError(
                "calibration probabilities must lie strictly inside (0, 1)"
            )
        if self.method == "identity":
            return raw.copy()
        if self.method != "platt":
            raise DraftResidualTournamentError(
                f"unknown frozen calibration method: {self.method!r}"
            )
        side = np.asarray(side_indicator, dtype=np.float64)
        if side.ndim == 0:
            side = np.full(len(raw), float(side), dtype=np.float64)
        if side.shape != raw.shape or not np.isin(side, (-1.0, 1.0)).all():
            raise DraftResidualTournamentError(
                "calibration side_indicator must be scalar or aligned +/-1"
            )
        raw_logit = np.log(raw) - np.log1p(-raw)
        linear = self.intercept * side + self.slope * raw_logit
        return np.asarray(
            [
                _bounded_calibration_probability(
                    float(value),
                    self.probability_floor,
                )
                for value in linear
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class CalibrationCandidateEvaluation:
    """Selection-gate score for a validation-fitted calibration map."""

    method: str
    model_version: str
    fit_split: str
    score_split: str
    fit_events: int
    score_events: int
    intercept: float
    slope: float
    log_loss: float
    brier: float
    ece: float
    primary_score_name: str
    primary_score: float
    complexity: int


@dataclass(frozen=True)
class CalibrationPredictionRow:
    """Untouched-final raw and calibrated candidate probability."""

    event_id: str
    dependence_id: str
    event_time: pd.Timestamp
    outcome: int
    candidate_model_id: str
    raw_probability: float
    calibrated_probability: float
    calibration_method: str
    calibration_model_version: str
    calibration_fit_through: pd.Timestamp


@dataclass(frozen=True)
class CalibrationTransferResult:
    """Development-selected calibration and untouched-final transfer."""

    candidate_model_id: str
    selection_scores: tuple[CalibrationCandidateEvaluation, ...]
    selected_method: str
    frozen_model: CalibrationModel
    final_predictions: tuple[CalibrationPredictionRow, ...]
    raw_final_diagnostics: ModelDiagnostics
    calibrated_final_diagnostics: ModelDiagnostics
    final_log_loss_delta: float
    final_brier_delta: float
    final_ece_delta: float
    selected_before_final: bool
    final_labels_used_for_selection: bool


@dataclass(frozen=True)
class InvariantReport:
    checked_events: int
    exact_raw_side_swap_antisymmetry: bool
    raw_estimand_excludes_team_offset: bool
    offset_baseline_exact: bool
    probability_bounds: bool
    no_future_leakage: bool
    sparse_design: bool
    unknown_states_explicit: bool
    paired_dependence_blocks_preserved: bool
    calibration_frozen_before_final: bool


@dataclass(frozen=True)
class TournamentResult:
    """Research evidence bundle; intentionally contains no promotion field."""

    tournament_version: str
    config: TournamentConfig
    primary_score: str
    split_summaries: tuple[SplitSummary, ...]
    validation_scores: tuple[CandidateEvaluation, ...]
    family_winner_ids: tuple[str, ...]
    selection_scores: tuple[CandidateEvaluation, ...]
    selected_candidate_id: str
    fit_manifests: tuple[FitManifest, ...]
    prediction_rows: tuple[PredictionRow, ...]
    final_model: ResidualCompositionModel
    final_diagnostics: tuple[ModelDiagnostics, ...]
    final_paired_ledger: tuple[PairedLossRow, ...]
    final_comparisons: tuple[PairedComparison, ...]
    final_bootstrap_inference: tuple[PairedBlockBootstrapResult, ...]
    calibration_transfer: CalibrationTransferResult
    invariants: InvariantReport

    def diagnostics_for(self, model_id: str) -> ModelDiagnostics:
        for diagnostics in self.final_diagnostics:
            if diagnostics.model_id == model_id:
                return diagnostics
        raise KeyError(model_id)


@dataclass(frozen=True)
class _FittedResidual:
    model: ResidualCompositionModel
    manifest: FitManifest
    complexity: int


@dataclass(frozen=True)
class _FittedLeagueSide:
    feature_names: tuple[str, ...]
    coefficients: tuple[float, ...]
    seen_leagues: tuple[str, ...]
    manifest: FitManifest
    optimizer_iterations: int
    optimizer_gradient_inf_norm: float


def default_candidate_specs(
    *,
    l2_grid: Sequence[float] = (10.0, 40.0),
    min_support_grid: Sequence[int] = (3,),
    include_patch_grid: Sequence[bool] = (False, True),
) -> tuple[CandidateSpec, ...]:
    """Return a deterministic, bounded candidate grid."""

    specs: list[CandidateSpec] = []
    for family in ("additive", "synergy", "opposition"):
        for min_support in min_support_grid:
            for l2 in l2_grid:
                for include_patch in include_patch_grid:
                    spec = CandidateSpec(
                        family=family,
                        l2=float(l2),
                        min_support=int(min_support),
                        include_patch_main=bool(include_patch),
                    )
                    spec.validate()
                    specs.append(spec)
    return tuple(specs)


def _as_utc(
    value: Any,
    *,
    field: str,
    event_id: str,
) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(timestamp):
        raise DraftResidualTournamentError(f"event {event_id!r} has invalid {field}")
    return pd.Timestamp(timestamp)


def _event_time(row: PreparedDraftMap) -> pd.Timestamp:
    return _as_utc(
        row.event_time,
        field="event_time",
        event_id=str(row.event_id),
    )


def _event_day(row: PreparedDraftMap) -> pd.Timestamp:
    return _event_time(row).normalize()


def _clean_text(value: Any) -> str:
    return str(value).strip()


def _validate_side(
    side: Sequence[tuple[str, str]],
    *,
    side_name: str,
    event_id: str,
) -> tuple[str, ...]:
    if len(side) != 5:
        raise DraftResidualTournamentError(
            f"event {event_id!r} {side_name} must contain exactly five picks"
        )
    roles = [_clean_text(role) for role, _champion in side]
    if set(roles) != set(ROLES) or len(set(roles)) != len(ROLES):
        raise DraftResidualTournamentError(
            f"event {event_id!r} {side_name} roles must be complete and unique"
        )
    champions: list[str] = []
    for role, raw_champion in side:
        if _clean_text(role) != str(role):
            raise DraftResidualTournamentError(
                f"event {event_id!r} has a non-canonical role label"
            )
        champion = _clean_text(raw_champion)
        if (
            not champion
            or champion.casefold() in _MISSING_CHAMPION_VALUES
            or champion != str(raw_champion)
            or "|" in champion
            or ":" in champion
        ):
            raise DraftResidualTournamentError(
                f"event {event_id!r} {side_name} has a missing or "
                "non-canonical champion"
            )
        champions.append(champion)
    if len({champion.casefold() for champion in champions}) != 5:
        raise DraftResidualTournamentError(
            f"event {event_id!r} {side_name} repeats a champion"
        )
    return tuple(champions)


def _validate_rows(
    rows: Sequence[PreparedDraftMap],
) -> tuple[PreparedDraftMap, ...]:
    if not rows:
        raise DraftResidualTournamentError(
            "at least one prepared draft map is required"
        )
    seen_event_ids: set[str] = set()
    validated: list[PreparedDraftMap] = []
    for row in rows:
        event_id = _clean_text(row.event_id)
        if not event_id or event_id != str(row.event_id):
            raise DraftResidualTournamentError(
                "event_id must be non-empty and canonical"
            )
        if event_id in seen_event_ids:
            raise DraftResidualTournamentError(f"duplicate event_id: {event_id}")
        seen_event_ids.add(event_id)
        dependence_id = _clean_text(row.dependence_id)
        if not dependence_id or dependence_id != str(row.dependence_id):
            raise DraftResidualTournamentError(
                f"event {event_id!r} requires a canonical dependence_id"
            )
        league = _clean_text(row.league)
        patch = _clean_text(row.patch)
        draft_mode = _clean_text(row.draft_mode)
        if (
            not league
            or league != str(row.league)
            or not patch
            or patch != str(row.patch)
            or "|" in league
            or "|" in patch
        ):
            raise DraftResidualTournamentError(
                f"event {event_id!r} requires canonical league and patch"
            )
        if draft_mode not in DRAFT_MODES:
            raise DraftResidualTournamentError(
                f"event {event_id!r} has unsupported draft_mode "
                f"{draft_mode!r}"
            )
        if (
            isinstance(row.y_blue_win, bool)
            or not isinstance(row.y_blue_win, (int, np.integer))
            or int(row.y_blue_win) not in (0, 1)
        ):
            raise DraftResidualTournamentError(
                f"event {event_id!r} outcome must be integer zero or one"
            )
        blue = _validate_side(row.blue, side_name="blue", event_id=event_id)
        red = _validate_side(row.red, side_name="red", event_id=event_id)
        all_champions = (*blue, *red)
        if (
            draft_mode != "blind_pick"
            and len({champion.casefold() for champion in all_champions}) != 10
        ):
            raise DraftResidualTournamentError(
                f"event {event_id!r} illegally duplicates a champion "
                "across the ten-pick draft"
            )
        _event_time(row)
        validated.append(row)
    return tuple(
        sorted(
            validated,
            key=lambda row: (_event_time(row), str(row.event_id)),
        )
    )


def _validate_offsets(
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
) -> dict[str, PreEventTeamLogit]:
    row_ids = {str(row.event_id) for row in rows}
    offset_ids = {str(event_id) for event_id in offsets}
    missing = sorted(row_ids - offset_ids)
    extra = sorted(offset_ids - row_ids)
    if missing or extra:
        raise DraftResidualTournamentError(
            "team offsets must match prepared events exactly; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    validated: dict[str, PreEventTeamLogit] = {}
    row_by_id = {str(row.event_id): row for row in rows}
    for event_id in sorted(row_ids):
        offset = offsets[event_id]
        if str(offset.event_id) != event_id:
            raise DraftResidualTournamentError(
                f"team offset key/event mismatch for {event_id!r}"
            )
        if not math.isfinite(float(offset.logit)):
            raise DraftResidualTournamentError(
                f"event {event_id!r} has a non-finite team offset"
            )
        if not _clean_text(offset.model_version):
            raise DraftResidualTournamentError(
                f"event {event_id!r} team offset lacks model_version"
            )
        if not _clean_text(offset.provenance):
            raise DraftResidualTournamentError(
                f"event {event_id!r} team offset lacks provenance"
            )
        if (
            not offset.team_strength_only
            or offset.includes_draft
            or offset.includes_side
            or not offset.dynamic
            or offset.orientation != "blue_minus_red"
            or offset.scale != "natural_log_odds"
        ):
            raise DraftResidualTournamentError(
                f"event {event_id!r} offset must be a dynamic, "
                "team-strength-only blue-minus-red natural log-odds value "
                "that excludes side and draft features"
            )
        as_of = _as_utc(
            offset.as_of,
            field="team offset as_of",
            event_id=event_id,
        )
        if not as_of < _event_time(row_by_id[event_id]):
            raise DraftResidualTournamentError(
                f"event {event_id!r} team offset is not strictly pre-event"
            )
        validated[event_id] = offset
    return validated


def chronological_date_splits(
    rows: Sequence[PreparedDraftMap],
    *,
    fractions: tuple[float, float, float, float] = (
        0.55,
        0.15,
        0.15,
        0.15,
    ),
    end_dates: tuple[str, str, str] | None = None,
    minimum_events_per_split: int = 1,
) -> DateSplits:
    """Split prepared maps without dividing a UTC calendar-date group."""

    if (
        len(fractions) != 4
        or any(not math.isfinite(value) or value <= 0.0 for value in fractions)
        or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12)
    ):
        raise DraftResidualTournamentError(
            "four positive fractions summing to one are required"
        )
    if minimum_events_per_split < 1:
        raise DraftResidualTournamentError("minimum_events_per_split must be positive")
    ordered = _validate_rows(rows)
    grouped: list[tuple[pd.Timestamp, list[PreparedDraftMap]]] = []
    for row in ordered:
        day = _event_day(row)
        if not grouped or grouped[-1][0] != day:
            grouped.append((day, []))
        grouped[-1][1].append(row)
    if len(grouped) < 4:
        raise DraftResidualTournamentError(
            "four chronological splits require at least four distinct dates"
        )

    if end_dates is not None:
        if len(end_dates) != 3:
            raise DraftResidualTournamentError(
                "end_dates must contain exactly three UTC dates"
            )
        cutoffs = tuple(
            _as_utc(
                value,
                field="split_end_date",
                event_id="chronological-split",
            ).normalize()
            for value in end_dates
        )
        if not cutoffs[0] < cutoffs[1] < cutoffs[2]:
            raise DraftResidualTournamentError(
                "end_dates must be strictly increasing"
            )
        explicit_partitions = (
            tuple(row for row in ordered if _event_day(row) <= cutoffs[0]),
            tuple(
                row
                for row in ordered
                if cutoffs[0] < _event_day(row) <= cutoffs[1]
            ),
            tuple(
                row
                for row in ordered
                if cutoffs[1] < _event_day(row) <= cutoffs[2]
            ),
            tuple(row for row in ordered if _event_day(row) > cutoffs[2]),
        )
        if any(
            len(partition) < minimum_events_per_split
            for partition in explicit_partitions
        ):
            raise DraftResidualTournamentError(
                "an explicit date split is smaller than "
                "minimum_events_per_split"
            )
        dependence_to_split: dict[str, str] = {}
        for split_name, partition in zip(SPLIT_NAMES, explicit_partitions):
            for row in partition:
                dependence_id = str(row.dependence_id)
                previous = dependence_to_split.setdefault(
                    dependence_id,
                    split_name,
                )
                if previous != split_name:
                    raise DraftResidualTournamentError(
                        "a dependence cluster crosses an explicit date "
                        f"boundary: {dependence_id!r}"
                    )
        return DateSplits(*explicit_partitions)

    cumulative = np.cumsum([len(group) for _day, group in grouped])
    total = len(ordered)
    targets = np.cumsum(fractions)[:-1] * total
    dependence_ranges: dict[str, tuple[int, int]] = {}
    for group_index, (_day, date_rows) in enumerate(grouped):
        for row in date_rows:
            dependence_id = str(row.dependence_id)
            first, last = dependence_ranges.get(
                dependence_id,
                (group_index, group_index),
            )
            dependence_ranges[dependence_id] = (
                min(first, group_index),
                max(last, group_index),
            )
    safe_boundaries = {
        boundary
        for boundary in range(1, len(grouped))
        if all(
            not first < boundary <= last for first, last in dependence_ranges.values()
        )
    }
    cuts: list[int] = []
    previous_group = 0
    previous_events = 0
    for boundary_index, target in enumerate(targets):
        remaining_splits = 3 - boundary_index
        candidates: list[int] = []
        for group_index in range(
            previous_group + 1,
            len(grouped) - remaining_splits + 1,
        ):
            if group_index not in safe_boundaries:
                continue
            events_through = int(cumulative[group_index - 1])
            if events_through - previous_events < minimum_events_per_split:
                continue
            if total - events_through < remaining_splits * minimum_events_per_split:
                continue
            candidates.append(group_index)
        if not candidates:
            raise DraftResidualTournamentError(
                "cannot satisfy date-safe boundaries and minimum split sizes"
            )
        cut = min(
            candidates,
            key=lambda index: (
                abs(float(cumulative[index - 1]) - float(target)),
                index,
            ),
        )
        cuts.append(cut)
        previous_group = cut
        previous_events = int(cumulative[cut - 1])

    ranges = (
        (0, cuts[0]),
        (cuts[0], cuts[1]),
        (cuts[1], cuts[2]),
        (cuts[2], len(grouped)),
    )
    partitions: list[tuple[PreparedDraftMap, ...]] = []
    for start, end in ranges:
        partition = tuple(
            row for _day, date_rows in grouped[start:end] for row in date_rows
        )
        if len(partition) < minimum_events_per_split:
            raise DraftResidualTournamentError(
                "date-safe split is smaller than minimum_events_per_split"
            )
        partitions.append(partition)

    splits = DateSplits(*partitions)
    previous_max: pd.Timestamp | None = None
    seen_days: set[pd.Timestamp] = set()
    for name, partition in splits.items():
        days = {_event_day(row) for row in partition}
        if seen_days.intersection(days):
            raise DraftResidualTournamentError(
                f"UTC date group leaked into split {name}"
            )
        if previous_max is not None and not previous_max < min(days):
            raise DraftResidualTournamentError(
                "split chronology is not strictly ordered"
            )
        previous_max = max(days)
        seen_days.update(days)
    return splits


def _role_map(
    side: Sequence[tuple[str, str]],
) -> dict[str, str]:
    return {str(role): str(champion) for role, champion in side}


def _opposition_key(
    first: tuple[str, str],
    second: tuple[str, str],
) -> tuple[str, float]:
    if first == second:
        raise DraftResidualTournamentError(
            "an opposition feature cannot compare an identical role/champion"
        )
    low, high = (first, second) if first < second else (second, first)
    sign = 1.0 if first == low else -1.0
    return (
        f"opposition|{low[0]}|{low[1]}|{high[0]}|{high[1]}",
        sign,
    )


def _global_feature_values(
    row: PreparedDraftMap,
    components: Iterable[str],
) -> dict[str, float]:
    component_set = frozenset(components)
    blue = _role_map(row.blue)
    red = _role_map(row.red)
    values: dict[str, float] = defaultdict(float)
    if "main" in component_set:
        for role in ROLES:
            values[f"main|{role}|{blue[role]}"] += 1.0
            values[f"main|{role}|{red[role]}"] -= 1.0
    if "synergy" in component_set:
        for first_index, first_role in enumerate(ROLES):
            for second_role in ROLES[first_index + 1 :]:
                values[
                    "synergy|"
                    f"{first_role}|{blue[first_role]}|"
                    f"{second_role}|{blue[second_role]}"
                ] += 1.0
                values[
                    "synergy|"
                    f"{first_role}|{red[first_role]}|"
                    f"{second_role}|{red[second_role]}"
                ] -= 1.0
    if "opposition" in component_set:
        for blue_role in ROLES:
            for red_role in ROLES:
                if (
                    blue_role == red_role
                    and blue[blue_role] == red[red_role]
                ):
                    # In an explicitly identified blind-pick map, a mirrored
                    # same-role champion is unchanged by side swap. Its only
                    # valid antisymmetric opposition contribution is zero.
                    continue
                key, sign = _opposition_key(
                    (blue_role, blue[blue_role]),
                    (red_role, red[red_role]),
                )
                values[key] += sign
    return {key: float(value) for key, value in values.items() if value != 0.0}


def _is_explicit_unknown_patch(patch: str) -> bool:
    return str(patch).strip().casefold() in _UNKNOWN_PATCH_VALUES


def _feature_values(
    row: PreparedDraftMap,
    spec: CandidateSpec,
) -> dict[str, float]:
    values = _global_feature_values(row, spec.components)
    if spec.include_patch_main and not _is_explicit_unknown_patch(row.patch):
        blue = _role_map(row.blue)
        red = _role_map(row.red)
        for role in ROLES:
            values[f"patch_main|{row.patch}|{role}|{blue[role]}"] = (
                values.get(f"patch_main|{row.patch}|{role}|{blue[role]}", 0.0) + 1.0
            )
            values[f"patch_main|{row.patch}|{role}|{red[role]}"] = (
                values.get(f"patch_main|{row.patch}|{role}|{red[role]}", 0.0) - 1.0
            )
    return {key: float(value) for key, value in values.items() if value != 0.0}


def _canonical_sides(
    row: PreparedDraftMap,
) -> tuple[
    tuple[tuple[str, str], ...],
    tuple[tuple[str, str], ...],
    float,
]:
    blue = tuple(sorted((str(role), str(champion)) for role, champion in row.blue))
    red = tuple(sorted((str(role), str(champion)) for role, champion in row.red))
    if blue < red:
        return blue, red, 1.0
    return red, blue, -1.0


def _model_spec(model: ResidualCompositionModel) -> CandidateSpec:
    return CandidateSpec(
        family=model.family,
        l2=model.l2,
        min_support=model.min_support,
        include_patch_main=model.include_patch_main,
        patch_l2_multiplier=model.patch_l2_multiplier,
    )


def _coefficient_pairs(
    names: Sequence[str],
    coefficients: Sequence[float],
) -> Iterable[tuple[str, float]]:
    if len(names) != len(coefficients):
        raise DraftResidualTournamentError(
            "coefficient names and values must have equal length"
        )
    return zip(names, coefficients)


def _direct_raw_composition_logit(
    model: ResidualCompositionModel,
    row: PreparedDraftMap,
) -> float:
    coefficients = dict(
        _coefficient_pairs(
            model.composition_feature_names,
            model.composition_coefficients,
        )
    )
    values = _feature_values(row, _model_spec(model))
    terms = (
        float(values[name]) * float(coefficients[name])
        for name in sorted(values)
        if name in coefficients
    )
    return float(math.fsum(terms))


def raw_composition_logit(
    model: ResidualCompositionModel,
    row: PreparedDraftMap,
) -> float:
    """Return a bit-exact antisymmetric raw composition logit.

    Team offset and league/side nuisance terms are deliberately absent.
    """

    canonical_blue, canonical_red, sign = _canonical_sides(row)
    canonical = PreparedDraftMap(
        event_id="canonical",
        dependence_id="canonical",
        event_time=row.event_time,
        league=row.league,
        patch=row.patch,
        blue=canonical_blue,
        red=canonical_red,
        y_blue_win=0,
        draft_mode=row.draft_mode,
    )
    value = _direct_raw_composition_logit(model, canonical)
    return value if sign > 0.0 else -value


def _symmetric_probability(logit: float) -> float:
    if logit == 0.0:
        return 0.5
    magnitude = float(expit(abs(logit)))
    return magnitude if logit > 0.0 else 1.0 - magnitude


def score_neutral_composition(
    model: ResidualCompositionModel,
    row: PreparedDraftMap,
) -> CompositionEstimate:
    """Score only composition, with team and league/side context set to zero."""

    raw = raw_composition_logit(model, row)
    known = set(model.known_champion_roles)
    supported = set(model.supported_champion_roles)
    present = tuple(
        (
            f"{side_name}:{role}:{champion}",
            f"{role}|{champion}",
        )
        for side_name, side in (("blue", row.blue), ("red", row.red))
        for role, champion in side
    )
    unknown = tuple(label for label, identity in present if identity not in known)
    unsupported = tuple(
        label
        for label, identity in present
        if identity in known and identity not in supported
    )
    if _is_explicit_unknown_patch(row.patch):
        patch_status = "explicit_unknown_global_fallback"
    elif row.patch not in set(model.seen_patches):
        patch_status = "unseen_global_fallback"
    elif model.include_patch_main:
        patch_status = "known_patch_adjusted"
    else:
        patch_status = "known_global_only"
    return CompositionEstimate(
        raw_composition_logit=raw,
        neutral_team_probability=_symmetric_probability(raw),
        patch_status=patch_status,
        unknown_champion_roles=unknown,
        unsupported_champion_roles=unsupported,
    )


def _nuisance_names(
    rows: Sequence[PreparedDraftMap],
) -> tuple[str, ...]:
    leagues = tuple(sorted({str(row.league) for row in rows}))
    return ("side|global", *(f"side|league|{league}" for league in leagues))


def _nuisance_values(
    row: PreparedDraftMap,
    names: Sequence[str],
) -> dict[str, float]:
    values = {"side|global": 1.0}
    league_name = f"side|league|{row.league}"
    if league_name in set(names):
        values[league_name] = 1.0
    return values


def _build_sparse_matrix(
    rows: Sequence[PreparedDraftMap],
    feature_names: Sequence[str],
    value_fn: Any,
) -> sparse.csr_matrix:
    index = {name: column for column, name in enumerate(feature_names)}
    data: list[float] = []
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, row in enumerate(rows):
        values = value_fn(row)
        for name in sorted(values):
            column_index = index.get(name)
            value = float(values[name])
            if column_index is None or value == 0.0:
                continue
            row_indices.append(row_index)
            column_indices.append(column_index)
            data.append(value)
    return sparse.csr_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(rows), len(feature_names)),
        dtype=np.float64,
    )


def _penalized_offset_objective(
    coefficients: np.ndarray,
    design: sparse.csr_matrix,
    outcome: np.ndarray,
    offset: np.ndarray,
    penalty: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return penalized Bernoulli negative log-likelihood and gradient."""

    linear = offset + np.asarray(design @ coefficients).ravel()
    loss = float(
        np.logaddexp(0.0, linear).sum()
        - np.dot(outcome, linear)
        + 0.5 * np.dot(penalty, np.square(coefficients))
    )
    residual = expit(linear) - outcome
    gradient = np.asarray(design.T @ residual).ravel()
    gradient += penalty * coefficients
    return loss, gradient


def _fit_penalized_offset_logistic(
    design: sparse.csr_matrix,
    outcome: np.ndarray,
    offset: np.ndarray,
    penalty: np.ndarray,
    *,
    config: TournamentConfig,
) -> tuple[np.ndarray, int, float]:
    if not sparse.isspmatrix_csr(design):
        raise DraftResidualTournamentError("design matrix must be CSR sparse")
    if design.shape[0] != len(outcome) or len(outcome) != len(offset):
        raise DraftResidualTournamentError(
            "design, outcome, and offset rows must align"
        )
    if design.shape[1] != len(penalty):
        raise DraftResidualTournamentError(
            "one regularization value is required per coefficient"
        )
    if len(np.unique(outcome)) < 2:
        raise DraftResidualTournamentError("every fit stage requires both map outcomes")
    if (
        not np.isfinite(outcome).all()
        or not np.isin(outcome, [0.0, 1.0]).all()
        or not np.isfinite(offset).all()
        or not np.isfinite(penalty).all()
        or np.any(penalty < 0.0)
    ):
        raise DraftResidualTournamentError("fit arrays contain invalid values")

    def objective(coefficients: np.ndarray) -> tuple[float, np.ndarray]:
        return _penalized_offset_objective(
            coefficients,
            design,
            outcome,
            offset,
            penalty,
        )

    def hessian_product(
        coefficients: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        linear = offset + np.asarray(design @ coefficients).ravel()
        probability = expit(linear)
        weight = probability * (1.0 - probability)
        projected = np.asarray(design @ direction).ravel()
        result = np.asarray(design.T @ (weight * projected)).ravel()
        result += penalty * direction
        return result

    initial = np.zeros(design.shape[1], dtype=np.float64)
    fitted = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": int(config.maximum_iterations),
            "ftol": float(config.optimizer_ftol),
            "gtol": float(config.optimizer_gtol),
            "maxls": 50,
        },
    )
    coefficients = np.asarray(fitted.x, dtype=np.float64)
    _loss, gradient = objective(coefficients)
    gradient_inf_norm = float(np.max(np.abs(gradient), initial=0.0))
    iterations = int(fitted.nit)
    statuses = [
        f"lbfgs(status={fitted.status}, message={fitted.message})"
    ]
    if (
        np.isfinite(coefficients).all()
        and math.isfinite(gradient_inf_norm)
        and gradient_inf_norm > config.maximum_gradient_inf_norm
    ):
        # L-BFGS-B can stop on machine-scale relative objective reduction
        # while a summed-likelihood gradient is still just above the explicit
        # acceptance bound. Newton-CG is a deterministic convex polish from
        # that solution; it does not alter the objective, features, or data.
        polished = minimize(
            objective,
            coefficients,
            method="Newton-CG",
            jac=True,
            hessp=hessian_product,
            options={
                "maxiter": min(int(config.maximum_iterations), 250),
                "xtol": min(
                    float(config.optimizer_gtol),
                    float(config.maximum_gradient_inf_norm) / 100.0,
                ),
            },
        )
        polished_coefficients = np.asarray(polished.x, dtype=np.float64)
        _polished_loss, polished_gradient = objective(
            polished_coefficients
        )
        polished_gradient_inf_norm = float(
            np.max(np.abs(polished_gradient), initial=0.0)
        )
        statuses.append(
            f"newton-cg(status={polished.status}, "
            f"message={polished.message})"
        )
        if (
            np.isfinite(polished_coefficients).all()
            and math.isfinite(polished_gradient_inf_norm)
            and polished_gradient_inf_norm < gradient_inf_norm
        ):
            coefficients = polished_coefficients
            gradient_inf_norm = polished_gradient_inf_norm
        iterations += int(polished.nit)
    if (
        not np.isfinite(coefficients).all()
        or not math.isfinite(gradient_inf_norm)
        or gradient_inf_norm > config.maximum_gradient_inf_norm
    ):
        raise DraftResidualTournamentError(
            "sparse offset-logistic optimization failed; "
            f"attempts={'; '.join(statuses)}, "
            f"gradient_inf_norm={gradient_inf_norm:.6g}"
        )
    return coefficients, iterations, gradient_inf_norm


def _offset_array(
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
) -> np.ndarray:
    return np.asarray(
        [float(offsets[str(row.event_id)].logit) for row in rows],
        dtype=np.float64,
    )


def _outcome_array(
    rows: Sequence[PreparedDraftMap],
) -> np.ndarray:
    return np.asarray(
        [int(row.y_blue_win) for row in rows],
        dtype=np.float64,
    )


def _training_champion_roles(
    rows: Sequence[PreparedDraftMap],
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"{role}|{champion}"
                for row in rows
                for side in (row.blue, row.red)
                for role, champion in side
            }
        )
    )


def _supported_champion_roles(
    feature_names: Sequence[str],
) -> tuple[str, ...]:
    values: set[str] = set()
    for name in feature_names:
        parts = name.split("|")
        if parts[0] == "main" and len(parts) == 3:
            values.add(f"{parts[1]}|{parts[2]}")
    return tuple(sorted(values))


def _model_fingerprint(
    *,
    model_id: str,
    nuisance_names: Sequence[str],
    nuisance_coefficients: Sequence[float],
    composition_names: Sequence[str] = (),
    composition_coefficients: Sequence[float] = (),
) -> str:
    terms = [f"model_id={model_id}"]
    terms.extend(
        f"nuisance:{name}={float(value):.17g}"
        for name, value in _coefficient_pairs(
            nuisance_names,
            nuisance_coefficients,
        )
    )
    terms.extend(
        f"composition:{name}={float(value):.17g}"
        for name, value in _coefficient_pairs(
            composition_names,
            composition_coefficients,
        )
    )
    return hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()


def _fit_manifest(
    *,
    model_id: str,
    fit_stage: str,
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    fingerprint: str,
) -> FitManifest:
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (_event_time(row), str(row.event_id)),
        )
    )
    event_ids = tuple(str(row.event_id) for row in ordered)
    digest = hashlib.sha256("\n".join(event_ids).encode("utf-8")).hexdigest()
    offset_digest = hashlib.sha256(
        "\n".join(
            (
                f"{event_id}|{float(offsets[event_id].logit):.17g}|"
                f"{_as_utc(offsets[event_id].as_of, field='as_of', event_id=event_id).isoformat()}|"
                f"{offsets[event_id].model_version}"
            )
            for event_id in event_ids
        ).encode("utf-8")
    ).hexdigest()
    version_payload = (
        f"{TOURNAMENT_VERSION}\n{model_id}\n{fit_stage}\n"
        f"{digest}\n{offset_digest}\n{fingerprint}"
    )
    version = hashlib.sha256(version_payload.encode("utf-8")).hexdigest()[:20]
    return FitManifest(
        model_id=model_id,
        model_version=version,
        fit_stage=fit_stage,
        training_event_ids=event_ids,
        training_event_digest=digest,
        offset_digest=offset_digest,
        training_events=len(event_ids),
        trained_through=max(_event_time(row) for row in ordered),
        maximum_offset_as_of=max(
            _as_utc(
                offsets[event_id].as_of,
                field="as_of",
                event_id=event_id,
            )
            for event_id in event_ids
        ),
    )


def _fit_league_side(
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    *,
    fit_stage: str,
    config: TournamentConfig,
) -> _FittedLeagueSide:
    names = _nuisance_names(rows)
    design = _build_sparse_matrix(
        rows,
        names,
        lambda row: _nuisance_values(row, names),
    )
    penalty = np.asarray(
        [
            config.global_side_l2 if name == "side|global" else config.league_side_l2
            for name in names
        ],
        dtype=np.float64,
    )
    coefficients, iterations, gradient = _fit_penalized_offset_logistic(
        design,
        _outcome_array(rows),
        _offset_array(rows, offsets),
        penalty,
        config=config,
    )
    fingerprint = _model_fingerprint(
        model_id=LEAGUE_SIDE_BASELINE,
        nuisance_names=names,
        nuisance_coefficients=coefficients,
    )
    manifest = _fit_manifest(
        model_id=LEAGUE_SIDE_BASELINE,
        fit_stage=fit_stage,
        rows=rows,
        offsets=offsets,
        fingerprint=fingerprint,
    )
    return _FittedLeagueSide(
        feature_names=names,
        coefficients=tuple(float(value) for value in coefficients),
        seen_leagues=tuple(sorted({str(row.league) for row in rows})),
        manifest=manifest,
        optimizer_iterations=iterations,
        optimizer_gradient_inf_norm=gradient,
    )


def _fit_residual(
    spec: CandidateSpec,
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    *,
    fit_stage: str,
    config: TournamentConfig,
) -> _FittedResidual:
    spec.validate()
    nuisance_names = _nuisance_names(rows)
    support: Counter[str] = Counter()
    row_features: dict[str, dict[str, float]] = {}
    for row in rows:
        values = _feature_values(row, spec)
        row_features[str(row.event_id)] = values
        support.update(name for name, value in values.items() if value != 0.0)
    composition_names = tuple(
        sorted(name for name, count in support.items() if count >= spec.min_support)
    )
    nuisance_design = _build_sparse_matrix(
        rows,
        nuisance_names,
        lambda row: _nuisance_values(row, nuisance_names),
    )
    composition_design = _build_sparse_matrix(
        rows,
        composition_names,
        lambda row: row_features[str(row.event_id)],
    )
    design = sparse.hstack(
        (nuisance_design, composition_design),
        format="csr",
        dtype=np.float64,
    )
    nuisance_penalty = [
        config.global_side_l2 if name == "side|global" else config.league_side_l2
        for name in nuisance_names
    ]
    composition_penalty = [
        spec.l2 * (spec.patch_l2_multiplier if name.startswith("patch_main|") else 1.0)
        for name in composition_names
    ]
    penalty = np.asarray(
        (*nuisance_penalty, *composition_penalty),
        dtype=np.float64,
    )
    coefficients, iterations, gradient = _fit_penalized_offset_logistic(
        design,
        _outcome_array(rows),
        _offset_array(rows, offsets),
        penalty,
        config=config,
    )
    nuisance_count = len(nuisance_names)
    nuisance_coefficients = coefficients[:nuisance_count]
    composition_coefficients = coefficients[nuisance_count:]
    fingerprint = _model_fingerprint(
        model_id=spec.candidate_id,
        nuisance_names=nuisance_names,
        nuisance_coefficients=nuisance_coefficients,
        composition_names=composition_names,
        composition_coefficients=composition_coefficients,
    )
    manifest = _fit_manifest(
        model_id=spec.candidate_id,
        fit_stage=fit_stage,
        rows=rows,
        offsets=offsets,
        fingerprint=fingerprint,
    )
    model = ResidualCompositionModel(
        candidate_id=spec.candidate_id,
        family=spec.family,
        include_patch_main=spec.include_patch_main,
        nuisance_feature_names=nuisance_names,
        nuisance_coefficients=tuple(float(value) for value in nuisance_coefficients),
        composition_feature_names=composition_names,
        composition_coefficients=tuple(
            float(value) for value in composition_coefficients
        ),
        known_champion_roles=_training_champion_roles(rows),
        supported_champion_roles=_supported_champion_roles(composition_names),
        seen_patches=tuple(
            sorted(
                {
                    str(row.patch)
                    for row in rows
                    if not _is_explicit_unknown_patch(row.patch)
                }
            )
        ),
        seen_leagues=tuple(sorted({str(row.league) for row in rows})),
        l2=float(spec.l2),
        min_support=int(spec.min_support),
        patch_l2_multiplier=float(spec.patch_l2_multiplier),
        optimizer_iterations=iterations,
        optimizer_gradient_inf_norm=gradient,
        sparse_design=sparse.isspmatrix_csr(design),
        model_version=manifest.model_version,
    )
    return _FittedResidual(
        model=model,
        manifest=manifest,
        complexity=len(nuisance_names) + len(composition_names),
    )


def _linear_terms(
    names: Sequence[str],
    coefficients: Sequence[float],
    values: Mapping[str, float],
) -> float:
    coefficient_by_name = dict(_coefficient_pairs(names, coefficients))
    return float(
        math.fsum(
            float(values[name]) * float(coefficient_by_name[name])
            for name in sorted(values)
            if name in coefficient_by_name
        )
    )


def _league_side_logit(
    model: ResidualCompositionModel | _FittedLeagueSide,
    row: PreparedDraftMap,
) -> float:
    if isinstance(model, ResidualCompositionModel):
        names = model.nuisance_feature_names
        coefficients = model.nuisance_coefficients
    else:
        names = model.feature_names
        coefficients = model.coefficients
    return _linear_terms(
        names,
        coefficients,
        _nuisance_values(row, names),
    )


def _probability(logit: float) -> float:
    probability = float(expit(float(logit)))
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise DraftResidualTournamentError(f"invalid probability from logit {logit!r}")
    return probability


def _prediction_id(
    *,
    phase: str,
    model_id: str,
    model_version: str,
    event_id: str,
) -> str:
    payload = f"{phase}\n{model_id}\n{model_version}\n{event_id}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _offset_prediction_rows(
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    *,
    phase: str,
) -> tuple[PredictionRow, ...]:
    predictions: list[PredictionRow] = []
    for row in rows:
        event_id = str(row.event_id)
        offset = offsets[event_id]
        as_of = _as_utc(offset.as_of, field="as_of", event_id=event_id)
        contextual_logit = float(offset.logit)
        probability = _probability(contextual_logit)
        predictions.append(
            PredictionRow(
                prediction_id=_prediction_id(
                    phase=phase,
                    model_id=OFFSET_ONLY_BASELINE,
                    model_version=str(offset.model_version),
                    event_id=event_id,
                ),
                model_id=OFFSET_ONLY_BASELINE,
                model_version=str(offset.model_version),
                phase=phase,
                event_id=event_id,
                dependence_id=str(row.dependence_id),
                event_time=_event_time(row),
                league=str(row.league),
                patch=str(row.patch),
                outcome=int(row.y_blue_win),
                probability=probability,
                contextual_logit=contextual_logit,
                team_offset_logit=float(offset.logit),
                team_offset_as_of=as_of,
                team_offset_model_version=str(offset.model_version),
                team_offset_provenance=str(offset.provenance),
                league_side_logit=0.0,
                raw_composition_logit=0.0,
                neutral_team_probability=0.5,
                patch_status="not_applicable",
                unknown_league=False,
                unknown_champion_roles=(),
                unsupported_champion_roles=(),
                trained_through=None,
                training_event_digest="",
            )
        )
    return tuple(predictions)


def _league_side_prediction_rows(
    model: _FittedLeagueSide,
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    *,
    phase: str,
) -> tuple[PredictionRow, ...]:
    predictions: list[PredictionRow] = []
    seen_leagues = set(model.seen_leagues)
    for row in rows:
        event_id = str(row.event_id)
        offset = offsets[event_id]
        side_logit = _league_side_logit(model, row)
        contextual_logit = float(offset.logit) + side_logit
        predictions.append(
            PredictionRow(
                prediction_id=_prediction_id(
                    phase=phase,
                    model_id=LEAGUE_SIDE_BASELINE,
                    model_version=model.manifest.model_version,
                    event_id=event_id,
                ),
                model_id=LEAGUE_SIDE_BASELINE,
                model_version=model.manifest.model_version,
                phase=phase,
                event_id=event_id,
                dependence_id=str(row.dependence_id),
                event_time=_event_time(row),
                league=str(row.league),
                patch=str(row.patch),
                outcome=int(row.y_blue_win),
                probability=_probability(contextual_logit),
                contextual_logit=contextual_logit,
                team_offset_logit=float(offset.logit),
                team_offset_as_of=_as_utc(
                    offset.as_of, field="as_of", event_id=event_id
                ),
                team_offset_model_version=str(offset.model_version),
                team_offset_provenance=str(offset.provenance),
                league_side_logit=side_logit,
                raw_composition_logit=0.0,
                neutral_team_probability=0.5,
                patch_status="not_applicable",
                unknown_league=str(row.league) not in seen_leagues,
                unknown_champion_roles=(),
                unsupported_champion_roles=(),
                trained_through=model.manifest.trained_through,
                training_event_digest=(model.manifest.training_event_digest),
            )
        )
    return tuple(predictions)


def _candidate_prediction_rows(
    fitted: _FittedResidual,
    rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    *,
    phase: str,
) -> tuple[PredictionRow, ...]:
    predictions: list[PredictionRow] = []
    seen_leagues = set(fitted.model.seen_leagues)
    for row in rows:
        event_id = str(row.event_id)
        offset = offsets[event_id]
        composition = score_neutral_composition(fitted.model, row)
        side_logit = _league_side_logit(fitted.model, row)
        contextual_logit = (
            float(offset.logit) + side_logit + composition.raw_composition_logit
        )
        predictions.append(
            PredictionRow(
                prediction_id=_prediction_id(
                    phase=phase,
                    model_id=fitted.model.candidate_id,
                    model_version=fitted.manifest.model_version,
                    event_id=event_id,
                ),
                model_id=fitted.model.candidate_id,
                model_version=fitted.manifest.model_version,
                phase=phase,
                event_id=event_id,
                dependence_id=str(row.dependence_id),
                event_time=_event_time(row),
                league=str(row.league),
                patch=str(row.patch),
                outcome=int(row.y_blue_win),
                probability=_probability(contextual_logit),
                contextual_logit=contextual_logit,
                team_offset_logit=float(offset.logit),
                team_offset_as_of=_as_utc(
                    offset.as_of, field="as_of", event_id=event_id
                ),
                team_offset_model_version=str(offset.model_version),
                team_offset_provenance=str(offset.provenance),
                league_side_logit=side_logit,
                raw_composition_logit=(composition.raw_composition_logit),
                neutral_team_probability=(composition.neutral_team_probability),
                patch_status=composition.patch_status,
                unknown_league=str(row.league) not in seen_leagues,
                unknown_champion_roles=(composition.unknown_champion_roles),
                unsupported_champion_roles=(composition.unsupported_champion_roles),
                trained_through=fitted.manifest.trained_through,
                training_event_digest=(fitted.manifest.training_event_digest),
            )
        )
    return tuple(predictions)


def _proper_score(
    outcome: np.ndarray,
    probability: np.ndarray,
    score: str,
) -> np.ndarray:
    clipped = np.clip(probability, 1e-15, 1.0 - 1e-15)
    if score == "brier":
        return np.square(probability - outcome)
    if score == "log_loss":
        return -(outcome * np.log(clipped) + (1.0 - outcome) * np.log1p(-clipped))
    raise DraftResidualTournamentError(f"unsupported score: {score}")


def _ece(
    outcome: np.ndarray,
    probability: np.ndarray,
    bins: int,
) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    result = 0.0
    # Keep Python 3.9 compatibility; itertools.pairwise starts in Python 3.10.
    for index, (low, high) in enumerate(
        zip(edges[:-1], edges[1:])  # noqa: RUF007
    ):
        mask = (probability >= low) & (
            probability <= high if index == bins - 1 else probability < high
        )
        if np.any(mask):
            result += float(mask.mean()) * abs(
                float(outcome[mask].mean()) - float(probability[mask].mean())
            )
    return float(result)


def _diagnostics_from_arrays(
    model_id: str,
    outcome: np.ndarray,
    probability: np.ndarray,
    config: TournamentConfig,
) -> ModelDiagnostics:
    outcome = np.asarray(outcome, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    if (
        outcome.ndim != 1
        or probability.ndim != 1
        or len(outcome) != len(probability)
        or not len(outcome)
    ):
        raise DraftResidualTournamentError(
            f"cannot diagnose zero predictions for {model_id}"
        )
    if (
        not np.isfinite(outcome).all()
        or not np.isin(outcome, (0.0, 1.0)).all()
        or not np.isfinite(probability).all()
        or np.any((probability < 0.0) | (probability > 1.0))
    ):
        raise DraftResidualTournamentError(
            f"invalid diagnostic inputs for {model_id}"
        )
    log_loss = float(_proper_score(outcome, probability, "log_loss").mean())
    brier = float(_proper_score(outcome, probability, "brier").mean())
    primary = log_loss if config.primary_score == "log_loss" else brier
    return ModelDiagnostics(
        model_id=model_id,
        events=len(outcome),
        log_loss=log_loss,
        brier=brier,
        ece=_ece(outcome, probability, config.ece_bins),
        event_rate=float(outcome.mean()),
        mean_probability=float(probability.mean()),
        primary_score_name=config.primary_score,
        primary_score=primary,
    )


def _diagnostics(
    model_id: str,
    rows: Sequence[PredictionRow],
    config: TournamentConfig,
) -> ModelDiagnostics:
    if not rows:
        raise DraftResidualTournamentError(
            f"cannot diagnose zero predictions for {model_id}"
        )
    return _diagnostics_from_arrays(
        model_id,
        np.asarray([row.outcome for row in rows], dtype=np.float64),
        np.asarray([row.probability for row in rows], dtype=np.float64),
        config,
    )


def _evaluation(
    *,
    candidate_id: str,
    family: str,
    stage: str,
    rows: Sequence[PredictionRow],
    complexity: int,
    model_version: str,
    config: TournamentConfig,
) -> CandidateEvaluation:
    diagnostics = _diagnostics(candidate_id, rows, config)
    return CandidateEvaluation(
        candidate_id=candidate_id,
        family=family,
        stage=stage,
        events=diagnostics.events,
        log_loss=diagnostics.log_loss,
        brier=diagnostics.brier,
        ece=diagnostics.ece,
        primary_score_name=diagnostics.primary_score_name,
        primary_score=diagnostics.primary_score,
        complexity=int(complexity),
        model_version=model_version,
    )


def select_simplest_within_tolerance(
    evaluations: Sequence[CandidateEvaluation],
    *,
    tie_tolerance: float,
) -> CandidateEvaluation:
    """Select by proper score, resolving practical ties toward simplicity."""

    if not evaluations:
        raise DraftResidualTournamentError(
            "cannot select from zero candidate evaluations"
        )
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise DraftResidualTournamentError(
            "tie_tolerance must be finite and non-negative"
        )
    best = min(row.primary_score for row in evaluations)
    eligible = [row for row in evaluations if row.primary_score <= best + tie_tolerance]
    return min(
        eligible,
        key=lambda row: (
            row.complexity,
            row.primary_score,
            row.candidate_id,
        ),
    )


def _split_summary(
    name: str,
    rows: Sequence[PreparedDraftMap],
) -> SplitSummary:
    event_times = tuple(_event_time(row) for row in rows)
    return SplitSummary(
        name=name,
        events=len(rows),
        dependence_clusters=len({str(row.dependence_id) for row in rows}),
        date_groups=len({timestamp.normalize() for timestamp in event_times}),
        date_min=min(event_times),
        date_max=max(event_times),
        event_ids=tuple(str(row.event_id) for row in rows),
    )


def _baseline_evaluations(
    *,
    fit_rows: Sequence[PreparedDraftMap],
    score_rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    fit_stage: str,
    phase: str,
    config: TournamentConfig,
) -> tuple[
    tuple[PredictionRow, ...],
    tuple[PredictionRow, ...],
    CandidateEvaluation,
    CandidateEvaluation,
    _FittedLeagueSide,
]:
    offset_rows = _offset_prediction_rows(score_rows, offsets, phase=phase)
    league_model = _fit_league_side(
        fit_rows,
        offsets,
        fit_stage=fit_stage,
        config=config,
    )
    league_rows = _league_side_prediction_rows(
        league_model,
        score_rows,
        offsets,
        phase=phase,
    )
    offset_versions = hashlib.sha256(
        "\n".join(
            sorted(
                {str(offsets[str(row.event_id)].model_version) for row in score_rows}
            )
        ).encode("utf-8")
    ).hexdigest()[:20]
    offset_evaluation = _evaluation(
        candidate_id=OFFSET_ONLY_BASELINE,
        family="baseline_offset_only",
        stage=phase,
        rows=offset_rows,
        complexity=0,
        model_version=offset_versions,
        config=config,
    )
    league_evaluation = _evaluation(
        candidate_id=LEAGUE_SIDE_BASELINE,
        family="baseline_league_side",
        stage=phase,
        rows=league_rows,
        complexity=len(league_model.feature_names),
        model_version=league_model.manifest.model_version,
        config=config,
    )
    return (
        offset_rows,
        league_rows,
        offset_evaluation,
        league_evaluation,
        league_model,
    )


def _event_score(y: int, probability: float, score: str) -> float:
    return float(
        _proper_score(
            np.asarray([y], dtype=np.float64),
            np.asarray([probability], dtype=np.float64),
            score,
        )[0]
    )


def _paired_ledger(
    *,
    candidate_rows: Sequence[PredictionRow],
    offset_rows: Sequence[PredictionRow],
    league_rows: Sequence[PredictionRow],
) -> tuple[PairedLossRow, ...]:
    candidate = {row.event_id: row for row in candidate_rows}
    offset = {row.event_id: row for row in offset_rows}
    league = {row.event_id: row for row in league_rows}
    if not candidate or set(candidate) != set(offset) or set(candidate) != set(league):
        raise DraftResidualTournamentError(
            "paired models must predict identical final events"
        )
    ledger: list[PairedLossRow] = []
    for event_id in sorted(
        candidate,
        key=lambda value: (
            candidate[value].event_time,
            value,
        ),
    ):
        candidate_row = candidate[event_id]
        offset_row = offset[event_id]
        league_row = league[event_id]
        identities = {
            (
                row.outcome,
                row.dependence_id,
                row.event_time,
            )
            for row in (candidate_row, offset_row, league_row)
        }
        if len(identities) != 1:
            raise DraftResidualTournamentError(
                f"paired event provenance disagrees for {event_id!r}"
            )
        y = candidate_row.outcome
        candidate_log_loss = _event_score(y, candidate_row.probability, "log_loss")
        offset_log_loss = _event_score(y, offset_row.probability, "log_loss")
        league_log_loss = _event_score(y, league_row.probability, "log_loss")
        candidate_brier = _event_score(y, candidate_row.probability, "brier")
        offset_brier = _event_score(y, offset_row.probability, "brier")
        league_brier = _event_score(y, league_row.probability, "brier")
        ledger.append(
            PairedLossRow(
                event_id=event_id,
                dependence_id=candidate_row.dependence_id,
                event_time=candidate_row.event_time,
                outcome=y,
                candidate_model_id=candidate_row.model_id,
                candidate_probability=candidate_row.probability,
                offset_probability=offset_row.probability,
                league_side_probability=league_row.probability,
                candidate_log_loss=candidate_log_loss,
                offset_log_loss=offset_log_loss,
                league_side_log_loss=league_log_loss,
                candidate_brier=candidate_brier,
                offset_brier=offset_brier,
                league_side_brier=league_brier,
                candidate_minus_offset_log_loss=(candidate_log_loss - offset_log_loss),
                candidate_minus_league_side_log_loss=(
                    candidate_log_loss - league_log_loss
                ),
                candidate_minus_offset_brier=(candidate_brier - offset_brier),
                candidate_minus_league_side_brier=(candidate_brier - league_brier),
            )
        )
    return tuple(ledger)


def _paired_comparisons(
    *,
    ledger: Sequence[PairedLossRow],
    candidate_diagnostics: ModelDiagnostics,
    offset_diagnostics: ModelDiagnostics,
    league_diagnostics: ModelDiagnostics,
) -> tuple[PairedComparison, ...]:
    cluster_count = len({row.dependence_id for row in ledger})
    return (
        PairedComparison(
            candidate_model_id=candidate_diagnostics.model_id,
            baseline_model_id=OFFSET_ONLY_BASELINE,
            events=len(ledger),
            dependence_clusters=cluster_count,
            log_loss_delta=float(
                np.mean([row.candidate_minus_offset_log_loss for row in ledger])
            ),
            brier_delta=float(
                np.mean([row.candidate_minus_offset_brier for row in ledger])
            ),
            ece_delta=(candidate_diagnostics.ece - offset_diagnostics.ece),
        ),
        PairedComparison(
            candidate_model_id=candidate_diagnostics.model_id,
            baseline_model_id=LEAGUE_SIDE_BASELINE,
            events=len(ledger),
            dependence_clusters=cluster_count,
            log_loss_delta=float(
                np.mean([row.candidate_minus_league_side_log_loss for row in ledger])
            ),
            brier_delta=float(
                np.mean([row.candidate_minus_league_side_brier for row in ledger])
            ),
            ece_delta=(candidate_diagnostics.ece - league_diagnostics.ece),
        ),
    )


def paired_circular_block_bootstrap(
    ledger: Sequence[PairedLossRow],
    *,
    replicates: int = 5_000,
    block_size: int = 12,
    random_seed: int = 20_260_727,
    alpha: float = 0.05,
) -> tuple[PairedBlockBootstrapResult, ...]:
    """Bootstrap paired map-loss deltas in ordered dependence-cluster blocks.

    The block rule is fixed consecutive clusters in earliest-event order,
    wrapped circularly and capped only when fewer clusters are available.
    Every sampled cluster contributes all of its maps, retaining the map-level
    proper-score estimand and within-cluster dependence.
    """

    if len(ledger) < 1:
        raise DraftResidualTournamentError(
            "paired block bootstrap requires a non-empty ledger"
        )
    if replicates < 2:
        raise DraftResidualTournamentError(
            "paired block bootstrap requires at least two replicates"
        )
    if block_size < 1:
        raise DraftResidualTournamentError(
            "paired block bootstrap block_size must be positive"
        )
    if (
        not isinstance(random_seed, int)
        or isinstance(random_seed, bool)
        or random_seed < 0
    ):
        raise DraftResidualTournamentError(
            "paired block bootstrap seed must be a non-negative integer"
        )
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
        raise DraftResidualTournamentError(
            "paired block bootstrap alpha must be in (0, 1)"
        )
    event_ids = [str(row.event_id) for row in ledger]
    if len(set(event_ids)) != len(event_ids):
        raise DraftResidualTournamentError(
            "paired block bootstrap event IDs must be unique"
        )
    candidate_ids = {
        _clean_text(row.candidate_model_id) for row in ledger
    }
    if len(candidate_ids) != 1 or "" in candidate_ids:
        raise DraftResidualTournamentError(
            "paired block bootstrap requires one candidate model"
        )
    if any(not _clean_text(row.dependence_id) for row in ledger):
        raise DraftResidualTournamentError(
            "paired block bootstrap requires non-empty dependence IDs"
        )

    delta_fields = (
        (
            OFFSET_ONLY_BASELINE,
            "log_loss",
            "candidate_minus_offset_log_loss",
        ),
        (
            OFFSET_ONLY_BASELINE,
            "brier",
            "candidate_minus_offset_brier",
        ),
        (
            LEAGUE_SIDE_BASELINE,
            "log_loss",
            "candidate_minus_league_side_log_loss",
        ),
        (
            LEAGUE_SIDE_BASELINE,
            "brier",
            "candidate_minus_league_side_brier",
        ),
    )
    by_cluster: dict[str, list[PairedLossRow]] = defaultdict(list)
    for row in ledger:
        by_cluster[str(row.dependence_id)].append(row)
    cluster_ids = tuple(
        sorted(
            by_cluster,
            key=lambda cluster_id: (
                min(
                    _as_utc(
                        row.event_time,
                        field="bootstrap event_time",
                        event_id=str(row.event_id),
                    )
                    for row in by_cluster[cluster_id]
                ),
                cluster_id,
            ),
        )
    )
    n_clusters = len(cluster_ids)
    if n_clusters < 2:
        raise DraftResidualTournamentError(
            "paired block bootstrap requires at least two dependence clusters"
        )
    cluster_counts = np.asarray(
        [len(by_cluster[cluster_id]) for cluster_id in cluster_ids],
        dtype=np.int64,
    )
    cluster_sums = np.asarray(
        [
            [
                math.fsum(
                    float(getattr(row, field))
                    for row in by_cluster[cluster_id]
                )
                for _baseline, _metric, field in delta_fields
            ]
            for cluster_id in cluster_ids
        ],
        dtype=np.float64,
    )
    if not np.isfinite(cluster_sums).all():
        raise DraftResidualTournamentError(
            "paired block bootstrap loss deltas must be finite"
        )

    effective_block_size = min(int(block_size), n_clusters)
    blocks_needed = (
        n_clusters + effective_block_size - 1
    ) // effective_block_size
    block_offsets = np.arange(effective_block_size, dtype=np.int64)
    rng = np.random.default_rng(random_seed)
    bootstrap = np.empty(
        (int(replicates), len(delta_fields)),
        dtype=np.float64,
    )
    for replicate in range(int(replicates)):
        starts = rng.integers(
            0,
            n_clusters,
            size=blocks_needed,
        )
        sampled_indices = (
            (starts[:, None] + block_offsets[None, :]) % n_clusters
        ).ravel()[:n_clusters]
        sampled_events = int(cluster_counts[sampled_indices].sum())
        if sampled_events < 1:
            raise DraftResidualTournamentError(
                "paired block bootstrap sampled zero events"
            )
        bootstrap[replicate] = (
            cluster_sums[sampled_indices].sum(axis=0) / sampled_events
        )

    all_dependence_ids = tuple(str(row.dependence_id) for row in ledger)
    dependence_status = (
        "explicit_inferred_unverified"
        if all(
            value.startswith("inferred-unverified:")
            for value in all_dependence_ids
        )
        else "caller_supplied_not_verified_by_module"
    )
    point_deltas = np.asarray(
        [
            math.fsum(float(getattr(row, field)) for row in ledger)
            / len(ledger)
            for _baseline, _metric, field in delta_fields
        ],
        dtype=np.float64,
    )
    results: list[PairedBlockBootstrapResult] = []
    for column, (baseline_id, metric, _field) in enumerate(delta_fields):
        distribution = bootstrap[:, column]
        lower, upper = np.quantile(
            distribution,
            (alpha / 2.0, 1.0 - alpha / 2.0),
            method="linear",
        )
        distribution_digest = hashlib.sha256(
            np.asarray(distribution, dtype="<f8").tobytes()
        ).hexdigest()
        results.append(
            PairedBlockBootstrapResult(
                candidate_model_id=next(iter(candidate_ids)),
                baseline_model_id=baseline_id,
                metric=metric,
                events=len(ledger),
                dependence_clusters=n_clusters,
                dependence_status=dependence_status,
                replicates=int(replicates),
                requested_block_size=int(block_size),
                effective_block_size=effective_block_size,
                block_rule=(
                    "fixed_consecutive_cluster_count_in_earliest_event_order"
                    "_circular_capped_at_available"
                ),
                random_seed=int(random_seed),
                alpha=float(alpha),
                interval_method="paired_percentile_circular_block_bootstrap",
                point_delta=float(point_deltas[column]),
                bootstrap_mean_delta=float(distribution.mean()),
                bootstrap_standard_error=float(
                    distribution.std(ddof=1)
                ),
                interval_lower=float(lower),
                interval_upper=float(upper),
                bootstrap_distribution_sha256=distribution_digest,
            )
        )
    return tuple(results)


def _bounded_calibration_probability(
    logit: float,
    probability_floor: float,
) -> float:
    if not math.isfinite(logit):
        raise DraftResidualTournamentError(
            "calibration logit must be finite"
        )
    if logit == 0.0:
        return 0.5
    magnitude = abs(float(logit))
    unit = 1.0 if magnitude >= 40.0 else float(expit(magnitude))
    upper = probability_floor + (1.0 - 2.0 * probability_floor) * unit
    return upper if logit > 0.0 else 1.0 - upper


def _calibration_objective(
    coefficients: np.ndarray,
    design: np.ndarray,
    outcome: np.ndarray,
    *,
    l2_identity_centered: float,
    probability_floor: float,
) -> tuple[float, np.ndarray]:
    """Bounded Platt negative log likelihood and analytic gradient."""

    coefficients = np.asarray(coefficients, dtype=np.float64)
    design = np.asarray(design, dtype=np.float64)
    outcome = np.asarray(outcome, dtype=np.float64)
    if (
        coefficients.shape != (2,)
        or design.ndim != 2
        or design.shape != (len(outcome), 2)
    ):
        raise DraftResidualTournamentError(
            "calibration objective arrays do not align"
        )
    if (
        not np.isfinite(design).all()
        or not np.isfinite(outcome).all()
        or not np.isin(outcome, (0.0, 1.0)).all()
    ):
        raise DraftResidualTournamentError(
            "calibration objective inputs must be finite and binary"
        )
    center = np.asarray((0.0, 1.0), dtype=np.float64)
    if not np.isfinite(coefficients).all():
        direction = np.nan_to_num(
            coefficients - center,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return (
            float(np.finfo(np.float64).max / 100.0),
            np.clip(direction, -1.0, 1.0) * 1e100,
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        linear = np.asarray(design @ coefficients).ravel()
    if not np.isfinite(linear).all():
        direction = np.clip(coefficients - center, -1.0, 1.0)
        return (
            float(np.finfo(np.float64).max / 100.0),
            direction * 1e100,
        )
    logistic = expit(linear)
    probability_scale = 1.0 - 2.0 * probability_floor
    calibrated = probability_floor + probability_scale * logistic
    loss = float(
        -np.sum(
            outcome * np.log(calibrated)
            + (1.0 - outcome) * np.log1p(-calibrated)
        )
        + 0.5
        * l2_identity_centered
        * np.square(coefficients - center).sum()
    )
    derivative = (
        (calibrated - outcome)
        / (calibrated * (1.0 - calibrated))
        * probability_scale
        * logistic
        * (1.0 - logistic)
    )
    gradient = (
        design.T @ derivative
        + l2_identity_centered * (coefficients - center)
    )
    return loss, np.asarray(gradient, dtype=np.float64)


def _ordered_prediction_rows(
    rows: Sequence[PredictionRow],
    *,
    context: str,
) -> tuple[PredictionRow, ...]:
    if not rows:
        raise DraftResidualTournamentError(
            f"{context} requires non-empty prediction rows"
        )
    ordered = tuple(
        sorted(
            rows,
            key=lambda row: (
                _as_utc(
                    row.event_time,
                    field=f"{context} event_time",
                    event_id=str(row.event_id),
                ),
                str(row.event_id),
            ),
        )
    )
    event_ids = [str(row.event_id) for row in ordered]
    if len(set(event_ids)) != len(event_ids):
        raise DraftResidualTournamentError(
            f"{context} event IDs must be unique"
        )
    model_ids = {_clean_text(row.model_id) for row in ordered}
    if len(model_ids) != 1 or "" in model_ids:
        raise DraftResidualTournamentError(
            f"{context} requires one candidate model"
        )
    probabilities = np.asarray(
        [row.probability for row in ordered],
        dtype=np.float64,
    )
    outcomes = np.asarray(
        [row.outcome for row in ordered],
        dtype=np.float64,
    )
    if (
        not np.isfinite(probabilities).all()
        or np.any((probabilities <= 0.0) | (probabilities >= 1.0))
        or not np.isin(outcomes, (0.0, 1.0)).all()
    ):
        raise DraftResidualTournamentError(
            f"{context} requires interior probabilities and binary outcomes"
        )
    return ordered


def _fit_calibration_model(
    method: str,
    rows: Sequence[PredictionRow],
    *,
    fit_split: str,
    selected_on_split: str,
    config: TournamentConfig,
) -> CalibrationModel:
    ordered = _ordered_prediction_rows(
        rows,
        context=f"{method} calibration fit",
    )
    if method not in set(config.calibration_methods):
        raise DraftResidualTournamentError(
            f"undeclared calibration method: {method!r}"
        )
    raw_probability = np.asarray(
        [row.probability for row in ordered],
        dtype=np.float64,
    )
    outcome = np.asarray(
        [row.outcome for row in ordered],
        dtype=np.float64,
    )
    event_digest = hashlib.sha256(
        "\n".join(
            (
                f"{row.event_id}|{float(row.probability):.17g}|"
                f"{int(row.outcome)}|{row.model_version}"
            )
            for row in ordered
        ).encode("utf-8")
    ).hexdigest()
    intercept = 0.0
    slope = 1.0
    iterations = 0
    gradient_inf_norm = 0.0
    if method == "platt":
        raw_logit = np.log(raw_probability) - np.log1p(-raw_probability)
        design = np.column_stack(
            (np.ones(len(raw_probability), dtype=np.float64), raw_logit)
        )
        center = np.asarray((0.0, 1.0), dtype=np.float64)
        penalty = float(config.calibration_l2_identity_centered)
        floor = float(config.calibration_probability_floor)

        def objective(
            coefficients: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            return _calibration_objective(
                coefficients,
                design,
                outcome,
                l2_identity_centered=penalty,
                probability_floor=floor,
            )

        optimizer = minimize(
            objective,
            center.copy(),
            method="L-BFGS-B",
            jac=True,
            bounds=(
                (
                    -float(config.calibration_max_abs_intercept),
                    float(config.calibration_max_abs_intercept),
                ),
                (
                    float(config.calibration_min_slope),
                    float(config.calibration_max_slope),
                ),
            ),
            options={
                "maxiter": 500,
                "ftol": 1e-15,
                "gtol": 1e-9,
                "maxls": 50,
            },
        )
        coefficients = np.asarray(optimizer.x, dtype=np.float64)
        _loss, gradient = objective(coefficients)
        gradient_inf_norm = float(
            np.max(np.abs(gradient), initial=0.0)
        )
        if (
            not optimizer.success
            or not np.isfinite(coefficients).all()
            or not math.isfinite(gradient_inf_norm)
            or gradient_inf_norm > config.maximum_gradient_inf_norm
        ):
            raise DraftResidualTournamentError(
                "calibration optimization failed; "
                f"status={optimizer.status}, message={optimizer.message}, "
                f"gradient_inf_norm={gradient_inf_norm:.6g}"
            )
        intercept = float(coefficients[0])
        slope = float(coefficients[1])
        iterations = int(optimizer.nit)

    model_payload = (
        f"{TOURNAMENT_VERSION}|calibration|{method}|{fit_split}|"
        f"{selected_on_split}|{event_digest}|{intercept:.17g}|"
        f"{slope:.17g}|{config.calibration_l2_identity_centered:.17g}|"
        f"{config.calibration_probability_floor:.17g}"
    )
    return CalibrationModel(
        method=method,
        intercept=intercept,
        slope=slope,
        l2_identity_centered=float(
            config.calibration_l2_identity_centered
        ),
        probability_floor=float(config.calibration_probability_floor),
        fit_split=fit_split,
        fit_events=len(ordered),
        fit_event_digest=event_digest,
        fit_through=max(
            _as_utc(
                row.event_time,
                field="calibration fit event_time",
                event_id=str(row.event_id),
            )
            for row in ordered
        ),
        selected_on_split=selected_on_split,
        optimizer_iterations=iterations,
        optimizer_gradient_inf_norm=gradient_inf_norm,
        model_version=hashlib.sha256(
            model_payload.encode("utf-8")
        ).hexdigest()[:20],
    )


def _calibration_candidate_evaluation(
    model: CalibrationModel,
    score_rows: Sequence[PredictionRow],
    *,
    config: TournamentConfig,
) -> CalibrationCandidateEvaluation:
    ordered = _ordered_prediction_rows(
        score_rows,
        context=f"{model.method} calibration score",
    )
    outcome = np.asarray(
        [row.outcome for row in ordered],
        dtype=np.float64,
    )
    probability = model.apply(
        np.asarray(
            [row.probability for row in ordered],
            dtype=np.float64,
        )
    )
    diagnostics = _diagnostics_from_arrays(
        f"{ordered[0].model_id}::calibration::{model.method}",
        outcome,
        probability,
        config,
    )
    primary_score = (
        diagnostics.log_loss
        if config.calibration_primary_score == "log_loss"
        else diagnostics.brier
    )
    return CalibrationCandidateEvaluation(
        method=model.method,
        model_version=model.model_version,
        fit_split=model.fit_split,
        score_split="selection",
        fit_events=model.fit_events,
        score_events=len(ordered),
        intercept=model.intercept,
        slope=model.slope,
        log_loss=diagnostics.log_loss,
        brier=diagnostics.brier,
        ece=diagnostics.ece,
        primary_score_name=config.calibration_primary_score,
        primary_score=primary_score,
        complexity=0 if model.method == "identity" else 2,
    )


def _select_calibration_evaluation(
    evaluations: Sequence[CalibrationCandidateEvaluation],
    *,
    tie_tolerance: float,
) -> CalibrationCandidateEvaluation:
    if not evaluations:
        raise DraftResidualTournamentError(
            "cannot select from zero calibration evaluations"
        )
    best = min(row.primary_score for row in evaluations)
    eligible = [
        row
        for row in evaluations
        if row.primary_score <= best + tie_tolerance
    ]
    return min(
        eligible,
        key=lambda row: (
            row.complexity,
            row.primary_score,
            row.method,
        ),
    )


def _run_calibration_transfer(
    *,
    candidate_model_id: str,
    validation_rows: Sequence[PredictionRow],
    selection_rows: Sequence[PredictionRow],
    final_rows: Sequence[PredictionRow],
    raw_final_diagnostics: ModelDiagnostics,
    config: TournamentConfig,
) -> CalibrationTransferResult:
    validation = _ordered_prediction_rows(
        validation_rows,
        context="calibration validation fit",
    )
    selection = _ordered_prediction_rows(
        selection_rows,
        context="calibration selection score",
    )
    final = _ordered_prediction_rows(
        final_rows,
        context="calibration final transfer",
    )
    if any(
        row.model_id != candidate_model_id
        for row in (*validation, *selection, *final)
    ):
        raise DraftResidualTournamentError(
            "calibration rows do not match the selected candidate"
        )
    validation_ids = {row.event_id for row in validation}
    selection_ids = {row.event_id for row in selection}
    final_ids = {row.event_id for row in final}
    if (
        validation_ids & selection_ids
        or validation_ids & final_ids
        or selection_ids & final_ids
    ):
        raise DraftResidualTournamentError(
            "calibration gates must contain disjoint events"
        )

    validation_models = tuple(
        _fit_calibration_model(
            method,
            validation,
            fit_split="validation_oos",
            selected_on_split="selection",
            config=config,
        )
        for method in config.calibration_methods
    )
    selection_scores = tuple(
        sorted(
            (
                _calibration_candidate_evaluation(
                    model,
                    selection,
                    config=config,
                )
                for model in validation_models
            ),
            key=lambda row: row.method,
        )
    )
    selected = _select_calibration_evaluation(
        selection_scores,
        tie_tolerance=config.calibration_tie_tolerance,
    )
    calibration_development = (*validation, *selection)
    frozen = _fit_calibration_model(
        selected.method,
        calibration_development,
        fit_split="validation+selection_oos",
        selected_on_split="selection",
        config=config,
    )
    final_start = min(
        _as_utc(
            row.event_time,
            field="calibration final event_time",
            event_id=str(row.event_id),
        )
        for row in final
    )
    if not frozen.fit_through < final_start:
        raise DraftResidualTournamentError(
            "calibration fit is not frozen before final"
        )
    raw_probability = np.asarray(
        [row.probability for row in final],
        dtype=np.float64,
    )
    calibrated_probability = frozen.apply(raw_probability)
    final_outcome = np.asarray(
        [row.outcome for row in final],
        dtype=np.float64,
    )
    calibrated_diagnostics = _diagnostics_from_arrays(
        f"{candidate_model_id}::calibrated::{frozen.method}",
        final_outcome,
        calibrated_probability,
        config,
    )
    final_predictions = tuple(
        CalibrationPredictionRow(
            event_id=str(row.event_id),
            dependence_id=str(row.dependence_id),
            event_time=_as_utc(
                row.event_time,
                field="calibration final event_time",
                event_id=str(row.event_id),
            ),
            outcome=int(row.outcome),
            candidate_model_id=candidate_model_id,
            raw_probability=float(raw),
            calibrated_probability=float(calibrated),
            calibration_method=frozen.method,
            calibration_model_version=frozen.model_version,
            calibration_fit_through=frozen.fit_through,
        )
        for row, raw, calibrated in zip(
            final,
            raw_probability,
            calibrated_probability,
        )
    )
    return CalibrationTransferResult(
        candidate_model_id=candidate_model_id,
        selection_scores=selection_scores,
        selected_method=selected.method,
        frozen_model=frozen,
        final_predictions=final_predictions,
        raw_final_diagnostics=raw_final_diagnostics,
        calibrated_final_diagnostics=calibrated_diagnostics,
        final_log_loss_delta=(
            calibrated_diagnostics.log_loss
            - raw_final_diagnostics.log_loss
        ),
        final_brier_delta=(
            calibrated_diagnostics.brier - raw_final_diagnostics.brier
        ),
        final_ece_delta=(
            calibrated_diagnostics.ece - raw_final_diagnostics.ece
        ),
        selected_before_final=True,
        final_labels_used_for_selection=False,
    )


def _audit_no_future_leakage(
    *,
    prediction_rows: Sequence[PredictionRow],
    manifests: Sequence[FitManifest],
) -> bool:
    by_version = {manifest.model_version: manifest for manifest in manifests}
    if len(by_version) != len(manifests):
        raise DraftResidualTournamentError("fit manifest model-version collision")
    for row in prediction_rows:
        if not row.team_offset_as_of < row.event_time:
            raise DraftResidualTournamentError(
                f"team offset leaks into event {row.event_id!r}"
            )
        if row.model_id == OFFSET_ONLY_BASELINE:
            continue
        manifest = by_version.get(row.model_version)
        if manifest is None or manifest.model_id != row.model_id:
            raise DraftResidualTournamentError(
                f"prediction {row.prediction_id} lacks fit provenance"
            )
        if row.event_id in set(manifest.training_event_ids):
            raise DraftResidualTournamentError(
                f"event {row.event_id!r} trained its own prediction"
            )
        if not manifest.trained_through < row.event_time:
            raise DraftResidualTournamentError(
                f"model training reaches event {row.event_id!r}"
            )
        if row.training_event_digest != manifest.training_event_digest:
            raise DraftResidualTournamentError(
                f"prediction {row.prediction_id} has a bad training digest"
            )
    return True


def _audit_invariants(
    *,
    model: ResidualCompositionModel,
    final_rows: Sequence[PreparedDraftMap],
    offsets: Mapping[str, PreEventTeamLogit],
    prediction_rows: Sequence[PredictionRow],
    manifests: Sequence[FitManifest],
    paired_ledger: Sequence[PairedLossRow],
    bootstrap_inference: Sequence[PairedBlockBootstrapResult],
    calibration_transfer: CalibrationTransferResult,
    maximum_events: int,
) -> InvariantReport:
    checked = tuple(
        sorted(
            final_rows,
            key=lambda row: (_event_time(row), str(row.event_id)),
        )[:maximum_events]
    )
    for row in checked:
        swapped = PreparedDraftMap(
            event_id=f"{row.event_id}:swapped",
            dependence_id=row.dependence_id,
            event_time=row.event_time,
            league=row.league,
            patch=row.patch,
            blue=row.red,
            red=row.blue,
            y_blue_win=1 - int(row.y_blue_win),
            draft_mode=row.draft_mode,
        )
        score = raw_composition_logit(model, row)
        swapped_score = raw_composition_logit(model, swapped)
        if score != -swapped_score:
            raise DraftResidualTournamentError(
                f"raw side-swap antisymmetry failed for {row.event_id!r}"
            )
        if score_neutral_composition(model, row).raw_composition_logit != score:
            raise DraftResidualTournamentError(
                f"neutral score is not the raw score for {row.event_id!r}"
            )

    if any(
        not math.isfinite(row.probability) or not 0.0 <= row.probability <= 1.0
        for row in prediction_rows
    ):
        raise DraftResidualTournamentError(
            "prediction probability bounds invariant failed"
        )
    offset_predictions = [
        row for row in prediction_rows if row.model_id == OFFSET_ONLY_BASELINE
    ]
    for prediction in offset_predictions:
        expected = _probability(float(offsets[prediction.event_id].logit))
        if prediction.probability != expected:
            raise DraftResidualTournamentError(
                f"offset-only baseline changed event {prediction.event_id!r}"
            )
    no_future = _audit_no_future_leakage(
        prediction_rows=prediction_rows,
        manifests=manifests,
    )
    paired_clusters = {
        str(row.dependence_id) for row in paired_ledger
    }
    if (
        len(bootstrap_inference) != 4
        or any(
            result.events != len(paired_ledger)
            or result.dependence_clusters != len(paired_clusters)
            for result in bootstrap_inference
        )
    ):
        raise DraftResidualTournamentError(
            "paired block-bootstrap provenance does not match final ledger"
        )
    final_candidate_by_id = {
        row.event_id: row
        for row in prediction_rows
        if row.phase == "final"
        and row.model_id == calibration_transfer.candidate_model_id
    }
    if (
        calibration_transfer.final_labels_used_for_selection
        or not calibration_transfer.selected_before_final
        or set(final_candidate_by_id)
        != {
            row.event_id
            for row in calibration_transfer.final_predictions
        }
    ):
        raise DraftResidualTournamentError(
            "calibration transfer provenance is not final-safe"
        )
    for calibrated in calibration_transfer.final_predictions:
        raw = final_candidate_by_id[calibrated.event_id]
        if (
            calibrated.raw_probability != raw.probability
            or not calibrated.calibration_fit_through
            < calibrated.event_time
        ):
            raise DraftResidualTournamentError(
                f"calibration transfer leaks into {calibrated.event_id!r}"
            )
    return InvariantReport(
        checked_events=len(checked),
        exact_raw_side_swap_antisymmetry=True,
        raw_estimand_excludes_team_offset=True,
        offset_baseline_exact=True,
        probability_bounds=True,
        no_future_leakage=no_future,
        sparse_design=bool(model.sparse_design),
        unknown_states_explicit=True,
        paired_dependence_blocks_preserved=True,
        calibration_frozen_before_final=True,
    )


def run_draft_residual_tournament(
    rows: Sequence[PreparedDraftMap],
    pre_event_team_logits: Mapping[str, PreEventTeamLogit],
    *,
    config: TournamentConfig | None = None,
) -> TournamentResult:
    """Run the research-only four-split offset residual tournament in memory."""

    settings = config or TournamentConfig()
    settings.validate()
    ordered = _validate_rows(rows)
    offsets = _validate_offsets(ordered, pre_event_team_logits)
    specs = (
        settings.candidate_specs
        if settings.candidate_specs
        else default_candidate_specs()
    )
    for spec in specs:
        spec.validate()
    candidate_ids = [spec.candidate_id for spec in specs]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise DraftResidualTournamentError(
            "candidate specifications must have unique IDs"
        )
    missing_families = sorted(COMPOSITION_FAMILIES - {spec.family for spec in specs})
    if missing_families:
        raise DraftResidualTournamentError(
            f"candidate grid is missing mandatory families: {missing_families}"
        )

    splits = chronological_date_splits(
        ordered,
        fractions=settings.split_fractions,
        end_dates=settings.split_end_dates,
        minimum_events_per_split=settings.minimum_events_per_split,
    )
    predictions: list[PredictionRow] = []
    manifests: list[FitManifest] = []
    validation_scores: list[CandidateEvaluation] = []

    for spec in specs:
        fitted = _fit_residual(
            spec,
            splits.train,
            offsets,
            fit_stage="train",
            config=settings,
        )
        manifests.append(fitted.manifest)
        candidate_rows = _candidate_prediction_rows(
            fitted,
            splits.validation,
            offsets,
            phase="validation",
        )
        predictions.extend(candidate_rows)
        validation_scores.append(
            _evaluation(
                candidate_id=spec.candidate_id,
                family=spec.family,
                stage="validation",
                rows=candidate_rows,
                complexity=fitted.complexity,
                model_version=fitted.manifest.model_version,
                config=settings,
            )
        )

    (
        validation_offset_rows,
        validation_league_rows,
        validation_offset_evaluation,
        validation_league_evaluation,
        validation_league_model,
    ) = _baseline_evaluations(
        fit_rows=splits.train,
        score_rows=splits.validation,
        offsets=offsets,
        fit_stage="train",
        phase="validation",
        config=settings,
    )
    predictions.extend((*validation_offset_rows, *validation_league_rows))
    manifests.append(validation_league_model.manifest)
    validation_scores.extend(
        (validation_offset_evaluation, validation_league_evaluation)
    )

    family_winners: list[CandidateEvaluation] = []
    for family in sorted(COMPOSITION_FAMILIES):
        family_winners.append(
            select_simplest_within_tolerance(
                [score for score in validation_scores if score.family == family],
                tie_tolerance=settings.tie_tolerance,
            )
        )

    spec_by_id = {spec.candidate_id: spec for spec in specs}
    development = (*splits.train, *splits.validation)
    selection_scores: list[CandidateEvaluation] = []
    for winner in family_winners:
        spec = spec_by_id[winner.candidate_id]
        fitted = _fit_residual(
            spec,
            development,
            offsets,
            fit_stage="train+validation",
            config=settings,
        )
        manifests.append(fitted.manifest)
        candidate_rows = _candidate_prediction_rows(
            fitted,
            splits.selection,
            offsets,
            phase="selection",
        )
        predictions.extend(candidate_rows)
        selection_scores.append(
            _evaluation(
                candidate_id=spec.candidate_id,
                family=spec.family,
                stage="selection",
                rows=candidate_rows,
                complexity=fitted.complexity,
                model_version=fitted.manifest.model_version,
                config=settings,
            )
        )

    (
        selection_offset_rows,
        selection_league_rows,
        selection_offset_evaluation,
        selection_league_evaluation,
        selection_league_model,
    ) = _baseline_evaluations(
        fit_rows=development,
        score_rows=splits.selection,
        offsets=offsets,
        fit_stage="train+validation",
        phase="selection",
        config=settings,
    )
    predictions.extend((*selection_offset_rows, *selection_league_rows))
    manifests.append(selection_league_model.manifest)
    selection_scores.extend((selection_offset_evaluation, selection_league_evaluation))

    selected = select_simplest_within_tolerance(
        [score for score in selection_scores if score.family in COMPOSITION_FAMILIES],
        tie_tolerance=settings.tie_tolerance,
    )
    selected_validation_rows = tuple(
        row
        for row in predictions
        if row.phase == "validation"
        and row.model_id == selected.candidate_id
    )
    selected_selection_rows = tuple(
        row
        for row in predictions
        if row.phase == "selection"
        and row.model_id == selected.candidate_id
    )
    if (
        len(selected_validation_rows) != len(splits.validation)
        or len(selected_selection_rows) != len(splits.selection)
    ):
        raise DraftResidualTournamentError(
            "selected candidate lacks complete validation/selection "
            "prediction ledgers for calibration"
        )

    prefinal = (*development, *splits.selection)
    selected_spec = spec_by_id[selected.candidate_id]
    final_fitted = _fit_residual(
        selected_spec,
        prefinal,
        offsets,
        fit_stage="train+validation+selection",
        config=settings,
    )
    manifests.append(final_fitted.manifest)
    final_candidate_rows = _candidate_prediction_rows(
        final_fitted,
        splits.final,
        offsets,
        phase="final",
    )
    predictions.extend(final_candidate_rows)

    (
        final_offset_rows,
        final_league_rows,
        _final_offset_evaluation,
        _final_league_evaluation,
        final_league_model,
    ) = _baseline_evaluations(
        fit_rows=prefinal,
        score_rows=splits.final,
        offsets=offsets,
        fit_stage="train+validation+selection",
        phase="final",
        config=settings,
    )
    predictions.extend((*final_offset_rows, *final_league_rows))
    manifests.append(final_league_model.manifest)

    candidate_diagnostics = _diagnostics(
        selected.candidate_id,
        final_candidate_rows,
        settings,
    )
    offset_diagnostics = _diagnostics(
        OFFSET_ONLY_BASELINE,
        final_offset_rows,
        settings,
    )
    league_diagnostics = _diagnostics(
        LEAGUE_SIDE_BASELINE,
        final_league_rows,
        settings,
    )
    paired = _paired_ledger(
        candidate_rows=final_candidate_rows,
        offset_rows=final_offset_rows,
        league_rows=final_league_rows,
    )
    comparisons = _paired_comparisons(
        ledger=paired,
        candidate_diagnostics=candidate_diagnostics,
        offset_diagnostics=offset_diagnostics,
        league_diagnostics=league_diagnostics,
    )
    bootstrap_inference = paired_circular_block_bootstrap(
        paired,
        replicates=settings.bootstrap_replicates,
        block_size=settings.bootstrap_block_size,
        random_seed=settings.bootstrap_random_seed,
        alpha=settings.bootstrap_alpha,
    )
    calibration_transfer = _run_calibration_transfer(
        candidate_model_id=selected.candidate_id,
        validation_rows=selected_validation_rows,
        selection_rows=selected_selection_rows,
        final_rows=final_candidate_rows,
        raw_final_diagnostics=candidate_diagnostics,
        config=settings,
    )
    invariants = _audit_invariants(
        model=final_fitted.model,
        final_rows=splits.final,
        offsets=offsets,
        prediction_rows=predictions,
        manifests=manifests,
        paired_ledger=paired,
        bootstrap_inference=bootstrap_inference,
        calibration_transfer=calibration_transfer,
        maximum_events=settings.invariant_check_events,
    )

    return TournamentResult(
        tournament_version=TOURNAMENT_VERSION,
        config=settings,
        primary_score=settings.primary_score,
        split_summaries=tuple(
            _split_summary(name, partition) for name, partition in splits.items()
        ),
        validation_scores=tuple(
            sorted(
                validation_scores,
                key=lambda score: (score.family, score.candidate_id),
            )
        ),
        family_winner_ids=tuple(
            winner.candidate_id
            for winner in sorted(family_winners, key=lambda score: score.family)
        ),
        selection_scores=tuple(
            sorted(
                selection_scores,
                key=lambda score: (score.family, score.candidate_id),
            )
        ),
        selected_candidate_id=selected.candidate_id,
        fit_manifests=tuple(manifests),
        prediction_rows=tuple(predictions),
        final_model=final_fitted.model,
        final_diagnostics=(
            candidate_diagnostics,
            offset_diagnostics,
            league_diagnostics,
        ),
        final_paired_ledger=paired,
        final_comparisons=comparisons,
        final_bootstrap_inference=bootstrap_inference,
        calibration_transfer=calibration_transfer,
        invariants=invariants,
    )
