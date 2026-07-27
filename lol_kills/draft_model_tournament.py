"""Leakage-safe tournament for complete League of Legends draft models.

The tournament is intentionally research-only: it returns frozen prediction
rows and diagnostics and never writes or promotes a production artifact.

Selection has three pre-final stages:

1. fit every hyperparameter specification on ``train`` and choose one per
   family on ``validation``;
2. refit those family winners on ``train + validation`` and choose the
   tournament winner on ``selection``;
3. refit only the preselected winner (and the preselected best composition
   candidate) on all pre-final data, then evaluate ``final`` exactly once.

All boundaries are made between UTC calendar-date groups.  Champion features
reuse the signed feature construction and shrinkage implementation in
``lol_kills.composition_model``; team, player, and roster identity are not
accepted by the game schema or by the feature-name audit.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lol_kills.composition_model import (
    ROLES,
    CompositionGame,
    _fit_logistic,
    _fit_low_rank_residual,
    _low_rank_value,
    feature_values,
)
from lol_kills.model_tournament import (
    corp_calibration_diagnostics,
    proper_score_vector,
)


TOURNAMENT_VERSION = "complete-draft-tournament-v1"
BASELINE_OVERALL = "baseline_overall_blue_rate"
BASELINE_LEAGUE = "baseline_side_league_blue_rate"
COMPOSITION_FAMILIES = frozenset(
    {"additive", "synergy", "opposition", "low_rank"}
)
MANDATORY_COMPOSITION_FAMILIES = frozenset(
    {"additive", "synergy", "opposition"}
)
ALLOWED_FEATURE_GROUPS = frozenset(
    {"main", "league", "patch", "synergy", "opposition"}
)
SPLIT_NAMES = ("train", "validation", "selection", "final")


class DraftTournamentError(ValueError):
    """Raised when the tournament evidence contract is violated."""


@dataclass(frozen=True)
class CandidateSpec:
    """One predeclared composition-model hyperparameter specification."""

    family: str
    min_support: int = 3
    prior_n: float = 25.0
    low_rank_rank: int = 0

    def validate(self) -> None:
        if self.family not in COMPOSITION_FAMILIES:
            raise DraftTournamentError(
                f"unknown candidate family: {self.family!r}"
            )
        if self.min_support < 1:
            raise DraftTournamentError("min_support must be positive")
        if not math.isfinite(self.prior_n) or self.prior_n <= 0.0:
            raise DraftTournamentError("prior_n must be finite and positive")
        if self.low_rank_rank < 0:
            raise DraftTournamentError("low_rank_rank cannot be negative")
        if self.family == "low_rank" and self.low_rank_rank < 1:
            raise DraftTournamentError(
                "the low_rank family requires low_rank_rank >= 1"
            )
        if self.family != "low_rank" and self.low_rank_rank:
            raise DraftTournamentError(
                "low_rank_rank is only valid for the low_rank family"
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
        prior = f"{self.prior_n:.8g}".replace(".", "p")
        return (
            f"draft_{self.family}"
            f"__support_{self.min_support}"
            f"__prior_{prior}"
            f"__rank_{self.low_rank_rank}"
        )


@dataclass(frozen=True)
class TournamentConfig:
    """Frozen selection, split, and baseline settings."""

    primary_score: str = "log_loss"
    split_fractions: tuple[float, float, float, float] = (
        0.55,
        0.15,
        0.15,
        0.15,
    )
    minimum_events_per_split: int = 20
    league_prior_n: float = 20.0
    tie_tolerance: float = 1e-6
    invariant_check_games: int = 16
    candidate_specs: tuple[CandidateSpec, ...] = ()

    def validate(self) -> None:
        if self.primary_score not in {"log_loss", "brier"}:
            raise DraftTournamentError(
                "primary_score must be 'log_loss' or 'brier'"
            )
        if len(self.split_fractions) != 4:
            raise DraftTournamentError("exactly four split fractions are required")
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in self.split_fractions)
            or not math.isclose(sum(self.split_fractions), 1.0, abs_tol=1e-12)
        ):
            raise DraftTournamentError(
                "split_fractions must be positive and sum to one"
            )
        if self.minimum_events_per_split < 1:
            raise DraftTournamentError(
                "minimum_events_per_split must be positive"
            )
        if not math.isfinite(self.league_prior_n) or self.league_prior_n <= 0.0:
            raise DraftTournamentError(
                "league_prior_n must be finite and positive"
            )
        if not math.isfinite(self.tie_tolerance) or self.tie_tolerance < 0.0:
            raise DraftTournamentError(
                "tie_tolerance must be finite and non-negative"
            )
        if self.invariant_check_games < 1:
            raise DraftTournamentError(
                "invariant_check_games must be positive"
            )


@dataclass(frozen=True)
class DateSplits:
    """Four chronological, date-group-safe event partitions."""

    train: tuple[CompositionGame, ...]
    validation: tuple[CompositionGame, ...]
    selection: tuple[CompositionGame, ...]
    final: tuple[CompositionGame, ...]

    def items(self) -> tuple[tuple[str, tuple[CompositionGame, ...]], ...]:
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
    date_groups: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    event_ids: tuple[str, ...]


@dataclass(frozen=True)
class FitManifest:
    """Immutable provenance for one fitted model version."""

    model_id: str
    model_version: str
    fit_stage: str
    training_event_ids: tuple[str, ...]
    training_event_digest: str
    training_events: int
    trained_through: pd.Timestamp


@dataclass(frozen=True)
class PredictionRow:
    """One immutable, event-level out-of-sample forecast."""

    prediction_id: str
    model_id: str
    model_version: str
    phase: str
    event_id: str
    event_date: pd.Timestamp
    league: str
    patch: str
    outcome: int
    probability: float
    trained_through: pd.Timestamp
    training_events: int
    training_event_digest: str


@dataclass(frozen=True)
class CandidateEvaluation:
    candidate_id: str
    family: str
    stage: str
    events: int
    primary_score_name: str
    primary_score: float
    log_loss: float
    brier: float
    complexity: int
    model_version: str


@dataclass(frozen=True)
class ModelDiagnostics:
    model_id: str
    events: int
    primary_score_name: str
    primary_score: float
    log_loss: float
    brier: float
    event_rate: float
    mean_probability: float
    ece_10: float
    pav_recalibrated_brier: float
    calibration_miscalibration: float
    calibration_discrimination: float
    calibration_uncertainty: float
    calibration_decomposition_residual: float


@dataclass(frozen=True)
class CompositionLift:
    """Proper-score gain of composition over the side/league intercept."""

    composition_model_id: str
    baseline_model_id: str
    events: int
    log_loss_lift: float
    brier_lift: float
    primary_score_name: str
    primary_score_lift: float
    composition_better: bool


@dataclass(frozen=True)
class FuturePatchDiagnostics:
    status: str
    patches: tuple[str, ...]
    events: int
    composition: ModelDiagnostics | None
    side_league_baseline: ModelDiagnostics | None
    composition_lift: CompositionLift | None


@dataclass(frozen=True)
class InvariantReport:
    checked_games: int
    exact_side_swap_antisymmetry: bool
    role_preserving_permutation: bool
    probability_bounds: bool
    identity_free_features: bool
    outcome_blind_features: bool
    no_temporal_leakage: bool


@dataclass(frozen=True)
class TournamentResult:
    """Complete immutable evidence bundle except for fitted model internals."""

    tournament_version: str
    primary_score: str
    split_summaries: tuple[SplitSummary, ...]
    validation_scores: tuple[CandidateEvaluation, ...]
    family_winner_ids: tuple[str, ...]
    selection_scores: tuple[CandidateEvaluation, ...]
    winner_id: str
    winner_is_composition: bool
    best_composition_id: str
    fit_manifests: tuple[FitManifest, ...]
    prediction_rows: tuple[PredictionRow, ...]
    final_diagnostics: tuple[ModelDiagnostics, ...]
    composition_lift: CompositionLift
    future_patch: FuturePatchDiagnostics
    invariants: InvariantReport

    def diagnostics_for(self, model_id: str) -> ModelDiagnostics:
        for row in self.final_diagnostics:
            if row.model_id == model_id:
                return row
        raise KeyError(model_id)


@dataclass(frozen=True)
class _BaseRateModel:
    overall_probability: float
    league_probabilities: tuple[tuple[str, float], ...]

    def predict(self, league: str, *, league_specific: bool) -> float:
        if league_specific:
            probability = dict(self.league_probabilities).get(
                league, self.overall_probability
            )
        else:
            probability = self.overall_probability
        return float(probability)


@dataclass(frozen=True)
class _FittedComposition:
    spec: CandidateSpec
    model: Mapping[str, Any]
    manifest: FitManifest
    complexity: int


def default_candidate_specs(
    *,
    min_support_grid: Sequence[int] = (3, 8),
    prior_n_grid: Sequence[float] = (15.0, 40.0),
    low_rank_ranks: Sequence[int] = (),
) -> tuple[CandidateSpec, ...]:
    """Return a deterministic, computationally bounded candidate grid.

    Additive, synergy, and opposition families are mandatory.  Low-rank
    residual candidates are included only when positive ``low_rank_ranks`` are
    supplied.
    """

    specs: list[CandidateSpec] = []
    for family in ("additive", "synergy", "opposition"):
        for min_support in min_support_grid:
            for prior_n in prior_n_grid:
                specs.append(
                    CandidateSpec(
                        family=family,
                        min_support=int(min_support),
                        prior_n=float(prior_n),
                    )
                )
    for rank in low_rank_ranks:
        for min_support in min_support_grid:
            for prior_n in prior_n_grid:
                specs.append(
                    CandidateSpec(
                        family="low_rank",
                        min_support=int(min_support),
                        prior_n=float(prior_n),
                        low_rank_rank=int(rank),
                    )
                )
    for spec in specs:
        spec.validate()
    return tuple(specs)


def _event_day(game: CompositionGame) -> pd.Timestamp:
    if game.date is None:
        raise DraftTournamentError(
            f"game {game.game_id!r} has no date; chronological evaluation fails closed"
        )
    timestamp = pd.Timestamp(game.date)
    if pd.isna(timestamp):
        raise DraftTournamentError(
            f"game {game.game_id!r} has an invalid date"
        )
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.normalize()


def _validate_games(games: Sequence[CompositionGame]) -> tuple[CompositionGame, ...]:
    if not games:
        raise DraftTournamentError("at least one complete draft is required")
    seen: set[str] = set()
    validated: list[CompositionGame] = []
    expected_roles = set(ROLES)
    for game in games:
        event_id = str(game.game_id).strip()
        if not event_id:
            raise DraftTournamentError("game_id must be non-empty")
        if event_id in seen:
            raise DraftTournamentError(f"duplicate game_id: {event_id}")
        seen.add(event_id)
        if game.y not in (0, 1):
            raise DraftTournamentError(
                f"game {event_id!r} has a non-binary outcome"
            )
        if not str(game.league).strip() or not str(game.patch).strip():
            raise DraftTournamentError(
                f"game {event_id!r} requires league and patch provenance"
            )
        for side_name, side in (("blue", game.blue), ("red", game.red)):
            if len(side) != 5:
                raise DraftTournamentError(
                    f"game {event_id!r} {side_name} side does not have five picks"
                )
            roles = [str(role) for role, _champion in side]
            champions = [str(champion).strip() for _role, champion in side]
            if set(roles) != expected_roles or len(set(roles)) != 5:
                raise DraftTournamentError(
                    f"game {event_id!r} {side_name} roles are not complete and unique"
                )
            if any(not champion for champion in champions):
                raise DraftTournamentError(
                    f"game {event_id!r} {side_name} has a blank champion"
                )
        _event_day(game)
        validated.append(game)
    return tuple(
        sorted(validated, key=lambda game: (_event_day(game), str(game.game_id)))
    )


def chronological_date_splits(
    games: Sequence[CompositionGame],
    *,
    fractions: tuple[float, float, float, float] = (0.55, 0.15, 0.15, 0.15),
    minimum_events_per_split: int = 1,
) -> DateSplits:
    """Split complete drafts without ever dividing a UTC calendar-date group."""

    if (
        len(fractions) != 4
        or any(not math.isfinite(value) or value <= 0.0 for value in fractions)
        or not math.isclose(sum(fractions), 1.0, abs_tol=1e-12)
    ):
        raise DraftTournamentError(
            "four positive fractions summing to one are required"
        )
    if minimum_events_per_split < 1:
        raise DraftTournamentError(
            "minimum_events_per_split must be positive"
        )
    ordered = _validate_games(games)
    grouped: list[tuple[pd.Timestamp, list[CompositionGame]]] = []
    for game in ordered:
        day = _event_day(game)
        if not grouped or grouped[-1][0] != day:
            grouped.append((day, []))
        grouped[-1][1].append(game)
    if len(grouped) < 4:
        raise DraftTournamentError(
            "four chronological splits require at least four distinct dates"
        )

    cumulative = np.cumsum([len(group) for _day, group in grouped])
    total = len(ordered)
    target_events = np.cumsum(fractions)[:-1] * total
    cuts: list[int] = []
    previous_group = 0
    previous_events = 0
    for boundary_index, target in enumerate(target_events):
        remaining_splits = 3 - boundary_index
        candidates: list[int] = []
        for group_index in range(
            previous_group + 1,
            len(grouped) - remaining_splits + 1,
        ):
            events_through = int(cumulative[group_index - 1])
            if events_through - previous_events < minimum_events_per_split:
                continue
            if total - events_through < remaining_splits * minimum_events_per_split:
                continue
            candidates.append(group_index)
        if not candidates:
            raise DraftTournamentError(
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

    group_ranges = (
        (0, cuts[0]),
        (cuts[0], cuts[1]),
        (cuts[1], cuts[2]),
        (cuts[2], len(grouped)),
    )
    partitions: list[tuple[CompositionGame, ...]] = []
    for start, end in group_ranges:
        partition = tuple(
            game
            for _day, date_games in grouped[start:end]
            for game in date_games
        )
        if len(partition) < minimum_events_per_split:
            raise DraftTournamentError(
                "date-safe split is smaller than minimum_events_per_split"
            )
        partitions.append(partition)

    splits = DateSplits(*partitions)
    previous_max: pd.Timestamp | None = None
    all_days: set[pd.Timestamp] = set()
    for name, partition in splits.items():
        days = {_event_day(game) for game in partition}
        if all_days.intersection(days):
            raise DraftTournamentError(f"date group leaked into split {name}")
        all_days.update(days)
        date_min, date_max = min(days), max(days)
        if previous_max is not None and not previous_max < date_min:
            raise DraftTournamentError("split chronology is not strictly ordered")
        previous_max = date_max
    return splits


def validate_identity_free_feature_names(feature_names: Iterable[str]) -> None:
    """Reject any feature outside the champion/league/patch composition grammar."""

    expected_parts = {
        "main": 3,
        "league": 4,
        "patch": 4,
        "synergy": 3,
        "opposition": 3,
    }
    invalid: list[str] = []
    for name in feature_names:
        parts = str(name).split("|")
        group = parts[0] if parts else ""
        if (
            group not in ALLOWED_FEATURE_GROUPS
            or len(parts) != expected_parts[group]
            or any(not part.strip() for part in parts)
        ):
            invalid.append(str(name))
    if invalid:
        raise DraftTournamentError(
            "team/player/roster or otherwise unapproved feature names are forbidden; "
            f"examples={sorted(invalid)[:5]}"
        )


def _canonical_side(
    side: Sequence[tuple[str, str]],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(role), str(champion)) for role, champion in side))


def _direct_composition_logit(
    model: Mapping[str, Any],
    game: CompositionGame,
) -> float:
    values = feature_values(game, model.get("components") or ())
    specs = model.get("feature_specs") or {}
    feature_terms = [
        float(values[key]) * float(specs[key].get("coef", 0.0))
        for key in sorted(values)
        if key in specs
    ]
    low_rank = model.get("low_rank") or {}
    low_rank_terms: list[float] = []
    if int(low_rank.get("rank", 0)) > 0:
        for _blue_role, blue_champion in game.blue:
            for _red_role, red_champion in game.red:
                low_rank_terms.append(
                    _low_rank_value(low_rank, blue_champion, red_champion)
                )
    return float(math.fsum((*feature_terms, *low_rank_terms)))


def composition_logit(
    model: Mapping[str, Any],
    game: CompositionGame,
) -> float:
    """Return a bit-exact signed composition score under a side swap.

    The fixed blue-side intercept is deliberately excluded.  Canonicalizing the
    two role-labelled sides means a swapped draft is represented as unary
    negation of the same floating-point value, rather than as a second
    independently accumulated sum.
    """

    blue = _canonical_side(game.blue)
    red = _canonical_side(game.red)
    if blue == red:
        return 0.0
    if blue < red:
        canonical_blue, canonical_red, sign = blue, red, 1.0
    else:
        canonical_blue, canonical_red, sign = red, blue, -1.0
    canonical = CompositionGame(
        game_id="canonical",
        blue=canonical_blue,
        red=canonical_red,
        y=0,
        league=game.league,
        patch=game.patch,
        date=game.date,
    )
    value = _direct_composition_logit(model, canonical)
    return value if sign > 0.0 else -value


def _bounded_sigmoid(value: float) -> float:
    if value >= 35.0:
        return 1.0 - 1e-15
    if value <= -35.0:
        return 1e-15
    return float(1.0 / (1.0 + math.exp(-value)))


def _symmetric_composition_probability(value: float) -> float:
    if value == 0.0:
        return 0.5
    magnitude_probability = _bounded_sigmoid(abs(value))
    return (
        magnitude_probability
        if value > 0.0
        else 1.0 - magnitude_probability
    )


def _candidate_probability(
    model: Mapping[str, Any],
    game: CompositionGame,
) -> float:
    return _bounded_sigmoid(
        float(model.get("intercept", 0.0)) + composition_logit(model, game)
    )


def _fit_manifest(
    model_id: str,
    fit_stage: str,
    games: Sequence[CompositionGame],
    model_fingerprint: str,
) -> FitManifest:
    ordered = tuple(
        sorted(
            ((str(game.game_id), _event_day(game)) for game in games),
            key=lambda item: (item[1], item[0]),
        )
    )
    event_ids = tuple(event_id for event_id, _day in ordered)
    event_digest = hashlib.sha256(
        "\n".join(event_ids).encode("utf-8")
    ).hexdigest()
    version_payload = (
        f"{TOURNAMENT_VERSION}\n{model_id}\n{fit_stage}\n"
        f"{event_digest}\n{model_fingerprint}"
    )
    model_version = hashlib.sha256(
        version_payload.encode("utf-8")
    ).hexdigest()[:20]
    return FitManifest(
        model_id=model_id,
        model_version=model_version,
        fit_stage=fit_stage,
        training_event_ids=event_ids,
        training_event_digest=event_digest,
        training_events=len(event_ids),
        trained_through=max(day for _event_id, day in ordered),
    )


def _composition_fingerprint(model: Mapping[str, Any]) -> str:
    specs = model.get("feature_specs") or {}
    terms = [
        f"{key}={float(specs[key].get('coef', 0.0)):.17g}"
        for key in sorted(specs)
    ]
    low_rank = model.get("low_rank") or {}
    terms.extend(
        (
            f"intercept={float(model.get('intercept', 0.0)):.17g}",
            f"rank={int(low_rank.get('rank', 0))}",
            f"champions={','.join(low_rank.get('champions') or [])}",
            f"left={repr(low_rank.get('left') or [])}",
            f"right={repr(low_rank.get('right') or [])}",
        )
    )
    return hashlib.sha256("\n".join(terms).encode("utf-8")).hexdigest()


def _fit_composition(
    spec: CandidateSpec,
    games: Sequence[CompositionGame],
    *,
    fit_stage: str,
) -> _FittedComposition:
    spec.validate()
    if len({game.y for game in games}) < 2:
        raise DraftTournamentError(
            f"{fit_stage} training data need both outcomes"
        )
    model = _fit_logistic(
        games,
        spec.components,
        min_support=spec.min_support,
        prior_n=spec.prior_n,
    )
    model["low_rank"] = _fit_low_rank_residual(
        model,
        games,
        spec.low_rank_rank,
    )
    validate_identity_free_feature_names((model.get("feature_specs") or {}).keys())
    low_rank = model.get("low_rank") or {}
    effective_rank = int(low_rank.get("rank", 0))
    low_rank_parameters = (
        2 * len(low_rank.get("champions") or []) * effective_rank
    )
    complexity = 1 + len(model.get("feature_specs") or {}) + low_rank_parameters
    manifest = _fit_manifest(
        spec.candidate_id,
        fit_stage,
        games,
        _composition_fingerprint(model),
    )
    return _FittedComposition(
        spec=spec,
        model=model,
        manifest=manifest,
        complexity=complexity,
    )


def _fit_base_rates(
    games: Sequence[CompositionGame],
    league_prior_n: float,
) -> _BaseRateModel:
    outcomes = np.asarray([game.y for game in games], dtype=float)
    overall = float((outcomes.sum() + 0.5) / (len(outcomes) + 1.0))
    league_probabilities: list[tuple[str, float]] = []
    for league in sorted({game.league for game in games}):
        league_outcomes = np.asarray(
            [game.y for game in games if game.league == league],
            dtype=float,
        )
        probability = float(
            (league_outcomes.sum() + league_prior_n * overall)
            / (len(league_outcomes) + league_prior_n)
        )
        league_probabilities.append((league, probability))
    return _BaseRateModel(
        overall_probability=overall,
        league_probabilities=tuple(league_probabilities),
    )


def _base_rate_fingerprint(
    model: _BaseRateModel,
    *,
    league_specific: bool,
) -> str:
    payload = [
        f"league_specific={league_specific}",
        f"overall={model.overall_probability:.17g}",
        *(
            f"{league}={probability:.17g}"
            for league, probability in model.league_probabilities
        ),
    ]
    return hashlib.sha256("\n".join(payload).encode("utf-8")).hexdigest()


def _prediction_rows(
    *,
    model_id: str,
    manifest: FitManifest,
    phase: str,
    games: Sequence[CompositionGame],
    probability_fn: Any,
) -> tuple[PredictionRow, ...]:
    if phase not in {"validation", "selection", "final"}:
        raise DraftTournamentError(f"invalid prediction phase: {phase}")
    rows: list[PredictionRow] = []
    for game in sorted(games, key=lambda item: (_event_day(item), item.game_id)):
        event_date = _event_day(game)
        probability = float(probability_fn(game))
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise DraftTournamentError(
                f"{model_id} emitted an invalid probability for {game.game_id}"
            )
        identity = (
            f"{phase}\n{model_id}\n{manifest.model_version}\n{game.game_id}"
        )
        prediction_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
        rows.append(
            PredictionRow(
                prediction_id=prediction_id,
                model_id=model_id,
                model_version=manifest.model_version,
                phase=phase,
                event_id=str(game.game_id),
                event_date=event_date,
                league=str(game.league),
                patch=str(game.patch),
                outcome=int(game.y),
                probability=probability,
                trained_through=manifest.trained_through,
                training_events=manifest.training_events,
                training_event_digest=manifest.training_event_digest,
            )
        )
    return tuple(rows)


def _ece_10(outcome: np.ndarray, probability: np.ndarray) -> float:
    ece = 0.0
    edges = np.linspace(0.0, 1.0, 11)
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= low) & (
            probability <= high if index == 9 else probability < high
        )
        if np.any(mask):
            ece += float(mask.mean()) * abs(
                float(outcome[mask].mean()) - float(probability[mask].mean())
            )
    return float(ece)


def _diagnostics(
    model_id: str,
    rows: Sequence[PredictionRow],
    primary_score_name: str,
) -> ModelDiagnostics:
    if not rows:
        raise DraftTournamentError(
            f"cannot diagnose zero predictions for {model_id}"
        )
    outcome = np.asarray([row.outcome for row in rows], dtype=float)
    probability = np.asarray([row.probability for row in rows], dtype=float)
    log_loss = float(proper_score_vector(outcome, probability, "log_loss").mean())
    brier = float(proper_score_vector(outcome, probability, "brier").mean())
    calibration = corp_calibration_diagnostics(outcome, probability)
    primary_score = log_loss if primary_score_name == "log_loss" else brier
    return ModelDiagnostics(
        model_id=model_id,
        events=len(rows),
        primary_score_name=primary_score_name,
        primary_score=primary_score,
        log_loss=log_loss,
        brier=brier,
        event_rate=float(outcome.mean()),
        mean_probability=float(probability.mean()),
        ece_10=_ece_10(outcome, probability),
        pav_recalibrated_brier=float(calibration["recalibrated_brier"]),
        calibration_miscalibration=float(calibration["miscalibration"]),
        calibration_discrimination=float(calibration["discrimination"]),
        calibration_uncertainty=float(calibration["uncertainty"]),
        calibration_decomposition_residual=float(
            calibration["decomposition_residual"]
        ),
    )


def _evaluation(
    candidate_id: str,
    family: str,
    stage: str,
    rows: Sequence[PredictionRow],
    primary_score_name: str,
    complexity: int,
    model_version: str,
) -> CandidateEvaluation:
    diagnostics = _diagnostics(candidate_id, rows, primary_score_name)
    return CandidateEvaluation(
        candidate_id=candidate_id,
        family=family,
        stage=stage,
        events=diagnostics.events,
        primary_score_name=primary_score_name,
        primary_score=diagnostics.primary_score,
        log_loss=diagnostics.log_loss,
        brier=diagnostics.brier,
        complexity=int(complexity),
        model_version=model_version,
    )


def select_simplest_within_tolerance(
    evaluations: Sequence[CandidateEvaluation],
    *,
    tie_tolerance: float,
) -> CandidateEvaluation:
    """Choose the lower-complexity model whenever proper scores are tied."""

    if not evaluations:
        raise DraftTournamentError("cannot select from zero evaluations")
    if not math.isfinite(tie_tolerance) or tie_tolerance < 0.0:
        raise DraftTournamentError("tie_tolerance must be finite and non-negative")
    best_score = min(row.primary_score for row in evaluations)
    tied = [
        row
        for row in evaluations
        if row.primary_score <= best_score + tie_tolerance
    ]
    return min(
        tied,
        key=lambda row: (
            row.complexity,
            row.primary_score,
            row.candidate_id,
        ),
    )


def _lift(
    composition: ModelDiagnostics,
    baseline: ModelDiagnostics,
) -> CompositionLift:
    if composition.events != baseline.events:
        raise DraftTournamentError(
            "composition and baseline diagnostics are not event-aligned"
        )
    log_loss_lift = baseline.log_loss - composition.log_loss
    brier_lift = baseline.brier - composition.brier
    primary_lift = baseline.primary_score - composition.primary_score
    return CompositionLift(
        composition_model_id=composition.model_id,
        baseline_model_id=baseline.model_id,
        events=composition.events,
        log_loss_lift=float(log_loss_lift),
        brier_lift=float(brier_lift),
        primary_score_name=composition.primary_score_name,
        primary_score_lift=float(primary_lift),
        composition_better=bool(primary_lift > 0.0),
    )


def _summary(name: str, games: Sequence[CompositionGame]) -> SplitSummary:
    days = tuple(_event_day(game) for game in games)
    return SplitSummary(
        name=name,
        events=len(games),
        date_groups=len(set(days)),
        date_min=min(days),
        date_max=max(days),
        event_ids=tuple(str(game.game_id) for game in games),
    )


def _audit_temporal_leakage(
    rows: Sequence[PredictionRow],
    manifests: Sequence[FitManifest],
) -> bool:
    by_version = {manifest.model_version: manifest for manifest in manifests}
    if len(by_version) != len(manifests):
        raise DraftTournamentError("model_version collision in fit manifests")
    training_ids_by_version = {
        version: frozenset(manifest.training_event_ids)
        for version, manifest in by_version.items()
    }
    for row in rows:
        manifest = by_version.get(row.model_version)
        if manifest is None or manifest.model_id != row.model_id:
            raise DraftTournamentError(
                f"prediction {row.prediction_id} lacks a matching fit manifest"
            )
        if row.event_id in training_ids_by_version[row.model_version]:
            raise DraftTournamentError(
                f"event {row.event_id} was used to train its own prediction"
            )
        if not manifest.trained_through < row.event_date:
            raise DraftTournamentError(
                "temporal leakage: training must end before the prediction date"
            )
        if (
            row.training_events != manifest.training_events
            or row.training_event_digest != manifest.training_event_digest
        ):
            raise DraftTournamentError(
                f"prediction {row.prediction_id} has inconsistent fit provenance"
            )
    return True


def _audit_invariants(
    model: Mapping[str, Any],
    games: Sequence[CompositionGame],
    rows: Sequence[PredictionRow],
    manifests: Sequence[FitManifest],
    *,
    maximum_games: int,
) -> InvariantReport:
    validate_identity_free_feature_names((model.get("feature_specs") or {}).keys())
    checked = tuple(
        sorted(games, key=lambda game: (_event_day(game), game.game_id))[
            :maximum_games
        ]
    )
    for game in checked:
        swapped = CompositionGame(
            game_id=f"{game.game_id}:swapped",
            blue=game.red,
            red=game.blue,
            y=1 - game.y,
            league=game.league,
            patch=game.patch,
            date=game.date,
        )
        score = composition_logit(model, game)
        swapped_score = composition_logit(model, swapped)
        if score != -swapped_score:
            raise DraftTournamentError(
                f"side-swap antisymmetry failed for {game.game_id}"
            )
        if (
            _symmetric_composition_probability(score)
            + _symmetric_composition_probability(swapped_score)
            != 1.0
        ):
            raise DraftTournamentError(
                f"side-swap probability complement failed for {game.game_id}"
            )

        permuted = CompositionGame(
            game_id=f"{game.game_id}:permuted",
            blue=tuple((*game.blue[1:], game.blue[0])),
            red=tuple((game.red[-1], *game.red[:-1])),
            y=game.y,
            league=game.league,
            patch=game.patch,
            date=game.date,
        )
        if composition_logit(model, game) != composition_logit(model, permuted):
            raise DraftTournamentError(
                f"role-preserving permutation failed for {game.game_id}"
            )
        flipped_outcome = CompositionGame(
            game_id=f"{game.game_id}:flipped-label",
            blue=game.blue,
            red=game.red,
            y=1 - game.y,
            league=game.league,
            patch=game.patch,
            date=game.date,
        )
        if feature_values(
            game, model.get("components") or ()
        ) != feature_values(
            flipped_outcome, model.get("components") or ()
        ):
            raise DraftTournamentError(
                f"outcome leaked into features for {game.game_id}"
            )

    if any(
        not math.isfinite(row.probability)
        or not 0.0 <= row.probability <= 1.0
        for row in rows
    ):
        raise DraftTournamentError("prediction probability bounds invariant failed")
    no_leakage = _audit_temporal_leakage(rows, manifests)
    return InvariantReport(
        checked_games=len(checked),
        exact_side_swap_antisymmetry=True,
        role_preserving_permutation=True,
        probability_bounds=True,
        identity_free_features=True,
        outcome_blind_features=True,
        no_temporal_leakage=no_leakage,
    )


def _base_rate_rows(
    *,
    model: _BaseRateModel,
    fit_games: Sequence[CompositionGame],
    score_games: Sequence[CompositionGame],
    fit_stage: str,
    phase: str,
) -> tuple[
    tuple[PredictionRow, ...],
    tuple[PredictionRow, ...],
    FitManifest,
    FitManifest,
]:
    overall_manifest = _fit_manifest(
        BASELINE_OVERALL,
        fit_stage,
        fit_games,
        _base_rate_fingerprint(model, league_specific=False),
    )
    league_manifest = _fit_manifest(
        BASELINE_LEAGUE,
        fit_stage,
        fit_games,
        _base_rate_fingerprint(model, league_specific=True),
    )
    overall_rows = _prediction_rows(
        model_id=BASELINE_OVERALL,
        manifest=overall_manifest,
        phase=phase,
        games=score_games,
        probability_fn=lambda game: model.predict(
            game.league, league_specific=False
        ),
    )
    league_rows = _prediction_rows(
        model_id=BASELINE_LEAGUE,
        manifest=league_manifest,
        phase=phase,
        games=score_games,
        probability_fn=lambda game: model.predict(
            game.league, league_specific=True
        ),
    )
    return overall_rows, league_rows, overall_manifest, league_manifest


def run_draft_model_tournament(
    games: Sequence[CompositionGame],
    *,
    config: TournamentConfig | None = None,
) -> TournamentResult:
    """Run the complete four-way chronological model tournament in memory."""

    settings = config or TournamentConfig()
    settings.validate()
    specs = (
        settings.candidate_specs
        if settings.candidate_specs
        else default_candidate_specs()
    )
    for spec in specs:
        spec.validate()
    families = {spec.family for spec in specs}
    missing_families = sorted(MANDATORY_COMPOSITION_FAMILIES - families)
    if missing_families:
        raise DraftTournamentError(
            f"candidate grid is missing mandatory families: {missing_families}"
        )
    candidate_ids = [spec.candidate_id for spec in specs]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise DraftTournamentError("candidate specifications must be unique")
    if {BASELINE_OVERALL, BASELINE_LEAGUE}.intersection(candidate_ids):
        raise DraftTournamentError("candidate id collides with a baseline id")

    splits = chronological_date_splits(
        games,
        fractions=settings.split_fractions,
        minimum_events_per_split=settings.minimum_events_per_split,
    )
    rows: list[PredictionRow] = []
    manifests: list[FitManifest] = []
    validation_scores: list[CandidateEvaluation] = []

    # Hyperparameters are evaluated only on validation after fitting train.
    for spec in specs:
        fitted = _fit_composition(spec, splits.train, fit_stage="train")
        manifests.append(fitted.manifest)
        predicted = _prediction_rows(
            model_id=spec.candidate_id,
            manifest=fitted.manifest,
            phase="validation",
            games=splits.validation,
            probability_fn=lambda game, model=fitted.model: _candidate_probability(
                model, game
            ),
        )
        rows.extend(predicted)
        validation_scores.append(
            _evaluation(
                spec.candidate_id,
                spec.family,
                "validation",
                predicted,
                settings.primary_score,
                fitted.complexity,
                fitted.manifest.model_version,
            )
        )

    validation_base_model = _fit_base_rates(
        splits.train, settings.league_prior_n
    )
    (
        validation_overall_rows,
        validation_league_rows,
        validation_overall_manifest,
        validation_league_manifest,
    ) = _base_rate_rows(
        model=validation_base_model,
        fit_games=splits.train,
        score_games=splits.validation,
        fit_stage="train",
        phase="validation",
    )
    rows.extend((*validation_overall_rows, *validation_league_rows))
    manifests.extend(
        (validation_overall_manifest, validation_league_manifest)
    )
    validation_scores.extend(
        (
            _evaluation(
                BASELINE_OVERALL,
                "baseline_overall",
                "validation",
                validation_overall_rows,
                settings.primary_score,
                1,
                validation_overall_manifest.model_version,
            ),
            _evaluation(
                BASELINE_LEAGUE,
                "baseline_side_league",
                "validation",
                validation_league_rows,
                settings.primary_score,
                1 + len(validation_base_model.league_probabilities),
                validation_league_manifest.model_version,
            ),
        )
    )

    family_winners: list[CandidateEvaluation] = []
    for family in sorted(families):
        family_winners.append(
            select_simplest_within_tolerance(
                [
                    row
                    for row in validation_scores
                    if row.family == family
                ],
                tie_tolerance=settings.tie_tolerance,
            )
        )

    # Candidate family selection happens only on the later selection split.
    development = (*splits.train, *splits.validation)
    spec_by_id = {spec.candidate_id: spec for spec in specs}
    selection_scores: list[CandidateEvaluation] = []
    for family_winner in family_winners:
        spec = spec_by_id[family_winner.candidate_id]
        fitted = _fit_composition(
            spec, development, fit_stage="train+validation"
        )
        manifests.append(fitted.manifest)
        predicted = _prediction_rows(
            model_id=spec.candidate_id,
            manifest=fitted.manifest,
            phase="selection",
            games=splits.selection,
            probability_fn=lambda game, model=fitted.model: _candidate_probability(
                model, game
            ),
        )
        rows.extend(predicted)
        selection_scores.append(
            _evaluation(
                spec.candidate_id,
                spec.family,
                "selection",
                predicted,
                settings.primary_score,
                fitted.complexity,
                fitted.manifest.model_version,
            )
        )

    selection_base_model = _fit_base_rates(
        development, settings.league_prior_n
    )
    (
        selection_overall_rows,
        selection_league_rows,
        selection_overall_manifest,
        selection_league_manifest,
    ) = _base_rate_rows(
        model=selection_base_model,
        fit_games=development,
        score_games=splits.selection,
        fit_stage="train+validation",
        phase="selection",
    )
    rows.extend((*selection_overall_rows, *selection_league_rows))
    manifests.extend((selection_overall_manifest, selection_league_manifest))
    selection_scores.extend(
        (
            _evaluation(
                BASELINE_OVERALL,
                "baseline_overall",
                "selection",
                selection_overall_rows,
                settings.primary_score,
                1,
                selection_overall_manifest.model_version,
            ),
            _evaluation(
                BASELINE_LEAGUE,
                "baseline_side_league",
                "selection",
                selection_league_rows,
                settings.primary_score,
                1 + len(selection_base_model.league_probabilities),
                selection_league_manifest.model_version,
            ),
        )
    )

    best_composition = select_simplest_within_tolerance(
        [
            row
            for row in selection_scores
            if row.family in COMPOSITION_FAMILIES
        ],
        tie_tolerance=settings.tie_tolerance,
    )
    winner = select_simplest_within_tolerance(
        selection_scores,
        tie_tolerance=settings.tie_tolerance,
    )

    # The winner is frozen before this point.  Final labels are not consulted
    # above and only the winner/best-composition plus mandatory baselines are
    # scored below.
    prefinal = (*development, *splits.selection)
    best_spec = spec_by_id[best_composition.candidate_id]
    fitted_final_composition = _fit_composition(
        best_spec,
        prefinal,
        fit_stage="train+validation+selection",
    )
    manifests.append(fitted_final_composition.manifest)
    final_composition_rows = _prediction_rows(
        model_id=best_spec.candidate_id,
        manifest=fitted_final_composition.manifest,
        phase="final",
        games=splits.final,
        probability_fn=lambda game: _candidate_probability(
            fitted_final_composition.model, game
        ),
    )
    rows.extend(final_composition_rows)

    final_base_model = _fit_base_rates(prefinal, settings.league_prior_n)
    (
        final_overall_rows,
        final_league_rows,
        final_overall_manifest,
        final_league_manifest,
    ) = _base_rate_rows(
        model=final_base_model,
        fit_games=prefinal,
        score_games=splits.final,
        fit_stage="train+validation+selection",
        phase="final",
    )
    rows.extend((*final_overall_rows, *final_league_rows))
    manifests.extend((final_overall_manifest, final_league_manifest))

    final_rows_by_model = {
        best_spec.candidate_id: final_composition_rows,
        BASELINE_OVERALL: final_overall_rows,
        BASELINE_LEAGUE: final_league_rows,
    }
    diagnostics_by_model = {
        model_id: _diagnostics(
            model_id, model_rows, settings.primary_score
        )
        for model_id, model_rows in final_rows_by_model.items()
    }
    ordered_diagnostic_ids = tuple(
        dict.fromkeys(
            (
                winner.candidate_id,
                best_spec.candidate_id,
                BASELINE_OVERALL,
                BASELINE_LEAGUE,
            )
        )
    )
    final_diagnostics = tuple(
        diagnostics_by_model[model_id] for model_id in ordered_diagnostic_ids
    )
    composition_lift = _lift(
        diagnostics_by_model[best_spec.candidate_id],
        diagnostics_by_model[BASELINE_LEAGUE],
    )

    development_patches = {
        game.patch for game in prefinal
    }
    future_event_ids = {
        str(game.game_id)
        for game in splits.final
        if game.patch not in development_patches
    }
    future_patches = tuple(
        sorted(
            {
                game.patch
                for game in splits.final
                if str(game.game_id) in future_event_ids
            }
        )
    )
    if future_event_ids:
        future_composition_rows = tuple(
            row
            for row in final_composition_rows
            if row.event_id in future_event_ids
        )
        future_league_rows = tuple(
            row
            for row in final_league_rows
            if row.event_id in future_event_ids
        )
        future_composition_diagnostics = _diagnostics(
            best_spec.candidate_id,
            future_composition_rows,
            settings.primary_score,
        )
        future_baseline_diagnostics = _diagnostics(
            BASELINE_LEAGUE,
            future_league_rows,
            settings.primary_score,
        )
        future_patch = FuturePatchDiagnostics(
            status="available",
            patches=future_patches,
            events=len(future_event_ids),
            composition=future_composition_diagnostics,
            side_league_baseline=future_baseline_diagnostics,
            composition_lift=_lift(
                future_composition_diagnostics,
                future_baseline_diagnostics,
            ),
        )
    else:
        future_patch = FuturePatchDiagnostics(
            status="not_available_no_unseen_final_patch",
            patches=(),
            events=0,
            composition=None,
            side_league_baseline=None,
            composition_lift=None,
        )

    invariant_report = _audit_invariants(
        fitted_final_composition.model,
        splits.final,
        rows,
        manifests,
        maximum_games=settings.invariant_check_games,
    )
    split_summaries = tuple(
        _summary(name, partition) for name, partition in splits.items()
    )
    return TournamentResult(
        tournament_version=TOURNAMENT_VERSION,
        primary_score=settings.primary_score,
        split_summaries=split_summaries,
        validation_scores=tuple(
            sorted(
                validation_scores,
                key=lambda row: (row.family, row.candidate_id),
            )
        ),
        family_winner_ids=tuple(
            row.candidate_id
            for row in sorted(family_winners, key=lambda row: row.family)
        ),
        selection_scores=tuple(
            sorted(selection_scores, key=lambda row: row.candidate_id)
        ),
        winner_id=winner.candidate_id,
        winner_is_composition=winner.family in COMPOSITION_FAMILIES,
        best_composition_id=best_spec.candidate_id,
        fit_manifests=tuple(manifests),
        prediction_rows=tuple(rows),
        final_diagnostics=final_diagnostics,
        composition_lift=composition_lift,
        future_patch=future_patch,
        invariants=invariant_report,
    )
