"""Leakage-safe dynamic Bradley--Terry research candidate.

This module is deliberately a benchmark candidate, not a production rating
promotion.  It keeps one immutable state per supplied organization key and
processes binary map outcomes in chronological, prequential order.

The model is a diagonal-Gaussian online approximation:

* ``P(blue wins)`` has a learned blue-side intercept and a team-strength
  difference;
* team states follow a Gaussian random walk, with optional mean reversion and
  variance inflation during inactivity;
* optional league/context offsets are used only after historical cross-context
  rows make the relevant bridge component identifiable under explicit
  thresholds;
* every map is an observation.  Series length, scheduled best-of format, and
  series winner are intentionally absent from the likelihood;
* rows sharing an exact timestamp are all predicted before any outcome in that
  timestamp batch is assimilated.

Predictive probabilities integrate diagonal state uncertainty with a
logistic-normal approximation.  ``side_indicator=+1`` is the ordinary
blue-team perspective.  Swapping the two teams and their context while also
negating ``side_indicator`` gives the complementary perspective exactly.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import minimize


RESEARCH_STATUS = "research_candidate_not_for_production_promotion"


@dataclass(frozen=True)
class DynamicBTColumns:
    """Input column contract.

    Team keys must already represent immutable organization identities.  Team
    display names, league labels, and tournament names are never used to
    construct or mutate those identities.
    """

    row_id: str = "game_uid"
    timestamp: str = "date"
    outcome: str = "y_blue_win"
    blue_team_key: str = "blue_team_key"
    red_team_key: str = "red_team_key"
    blue_context: str | None = "blue_league"
    red_context: str | None = "red_league"
    competition: str | None = "competition"
    side_indicator: str | None = None


@dataclass(frozen=True)
class DynamicBTConfig:
    """Hyperparameters for the online Gaussian filter, in log-odds units."""

    team_prior_sd: float = 0.90
    context_prior_sd: float = 0.65
    blue_side_prior_logit: float = 0.08
    blue_side_prior_sd: float = 0.30
    team_variance_per_day: float = 0.003
    context_variance_per_day: float = 0.001
    side_variance_per_day: float = 0.0
    mean_reversion_half_life_days: float | None = 365.0
    min_variance: float = 1e-5
    max_team_variance: float = 4.0
    max_context_variance: float = 3.0
    max_side_variance: float = 1.0
    max_abs_mean: float = 8.0
    probability_floor: float = 1e-6
    enable_bridge_terms: bool = True
    min_bridge_maps: int = 3
    min_bridge_teams_per_context: int = 2
    min_bridge_competitions: int = 1
    unsupported_bridge_variance: float = 2.0

    def __post_init__(self) -> None:
        positive = {
            "team_prior_sd": self.team_prior_sd,
            "context_prior_sd": self.context_prior_sd,
            "blue_side_prior_sd": self.blue_side_prior_sd,
            "min_variance": self.min_variance,
            "max_team_variance": self.max_team_variance,
            "max_context_variance": self.max_context_variance,
            "max_side_variance": self.max_side_variance,
            "max_abs_mean": self.max_abs_mean,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative = {
            "team_variance_per_day": self.team_variance_per_day,
            "context_variance_per_day": self.context_variance_per_day,
            "side_variance_per_day": self.side_variance_per_day,
            "unsupported_bridge_variance": self.unsupported_bridge_variance,
        }
        for name, value in nonnegative.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.mean_reversion_half_life_days is not None:
            if (
                not math.isfinite(self.mean_reversion_half_life_days)
                or self.mean_reversion_half_life_days <= 0
            ):
                raise ValueError(
                    "mean_reversion_half_life_days must be positive or None"
                )
        if not 0 <= self.probability_floor < 0.5:
            raise ValueError("probability_floor must be in [0, 0.5)")
        if self.min_variance > min(
            self.max_team_variance,
            self.max_context_variance,
            self.max_side_variance,
        ):
            raise ValueError("min_variance cannot exceed a maximum variance")
        for name, value in {
            "min_bridge_maps": self.min_bridge_maps,
            "min_bridge_teams_per_context": self.min_bridge_teams_per_context,
            "min_bridge_competitions": self.min_bridge_competitions,
        }.items():
            if value < 1:
                raise ValueError(f"{name} must be at least one")


@dataclass
class GaussianState:
    mean: float
    variance: float
    last_timestamp: pd.Timestamp | None = None
    last_observed: pd.Timestamp | None = None
    observations: int = 0


@dataclass
class _BridgeEdge:
    left: str
    right: str
    maps: int = 0
    left_teams: set[str] = field(default_factory=set)
    right_teams: set[str] = field(default_factory=set)
    competitions: set[str] = field(default_factory=set)

    def active(self, config: DynamicBTConfig) -> bool:
        return bool(
            self.maps >= config.min_bridge_maps
            and len(self.left_teams) >= config.min_bridge_teams_per_context
            and len(self.right_teams) >= config.min_bridge_teams_per_context
            and len(self.competitions) >= config.min_bridge_competitions
        )


@dataclass(frozen=True)
class BridgeSupport:
    status: str
    component: str | None
    direct_maps: int


class _BridgeTracker:
    """Historical bridge graph; registration happens only after prediction."""

    def __init__(self, config: DynamicBTConfig) -> None:
        self.config = config
        self.edges: dict[tuple[str, str], _BridgeEdge] = {}

    @staticmethod
    def _edge_key(left: str, right: str) -> tuple[str, str]:
        return tuple(sorted((left, right)))

    def register(
        self,
        blue_context: str,
        red_context: str,
        blue_team: str,
        red_team: str,
        competition: str,
    ) -> None:
        if (
            not self.config.enable_bridge_terms
            or not blue_context
            or not red_context
            or blue_context == red_context
        ):
            return
        left, right = self._edge_key(blue_context, red_context)
        edge = self.edges.setdefault((left, right), _BridgeEdge(left, right))
        edge.maps += 1
        if blue_context == left:
            edge.left_teams.add(blue_team)
            edge.right_teams.add(red_team)
        else:
            edge.left_teams.add(red_team)
            edge.right_teams.add(blue_team)
        edge.competitions.add(competition or "UNKNOWN")

    def _active_adjacency(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = defaultdict(set)
        for (left, right), edge in self.edges.items():
            if edge.active(self.config):
                graph[left].add(right)
                graph[right].add(left)
        return graph

    def support(self, blue_context: str, red_context: str) -> BridgeSupport:
        if not self.config.enable_bridge_terms:
            return BridgeSupport("disabled", None, 0)
        if not blue_context or not red_context:
            return BridgeSupport("missing_context", None, 0)
        if blue_context == red_context:
            return BridgeSupport("within_context", blue_context, 0)

        edge = self.edges.get(self._edge_key(blue_context, red_context))
        direct_maps = edge.maps if edge is not None else 0
        graph = self._active_adjacency()
        if blue_context not in graph or red_context not in graph:
            return BridgeSupport("unsupported", None, direct_maps)

        queue: deque[str] = deque([blue_context])
        seen = {blue_context}
        while queue:
            node = queue.popleft()
            for neighbor in sorted(graph.get(node, ())):
                if neighbor in seen:
                    continue
                seen.add(neighbor)
                queue.append(neighbor)
        if red_context not in seen:
            return BridgeSupport("unsupported", None, direct_maps)
        return BridgeSupport("supported", "|".join(sorted(seen)), direct_maps)

    def audit(self) -> dict[str, Any]:
        edges: list[dict[str, Any]] = []
        for (left, right), edge in sorted(self.edges.items()):
            edges.append(
                {
                    "left": left,
                    "right": right,
                    "maps": edge.maps,
                    "left_teams": sorted(edge.left_teams),
                    "right_teams": sorted(edge.right_teams),
                    "competitions": sorted(edge.competitions),
                    "identifiable": edge.active(self.config),
                }
            )
        return {
            "enabled": self.config.enable_bridge_terms,
            "thresholds": {
                "min_bridge_maps": self.config.min_bridge_maps,
                "min_bridge_teams_per_context": (
                    self.config.min_bridge_teams_per_context
                ),
                "min_bridge_competitions": self.config.min_bridge_competitions,
            },
            "edges": edges,
            "active_edges": int(sum(bool(edge["identifiable"]) for edge in edges)),
        }


@dataclass(frozen=True)
class DynamicBTPrediction:
    probability: float
    latent_logit: float
    predictive_variance: float
    predictive_sigma: float
    blue_team_mean: float
    red_team_mean: float
    blue_team_variance: float
    red_team_variance: float
    blue_side_logit: float
    blue_side_variance: float
    blue_context_mean: float
    red_context_mean: float
    bridge_status: str
    bridge_component: str | None
    bridge_direct_maps: int


@dataclass(frozen=True)
class PreparedDynamicBTData:
    frame: pd.DataFrame
    exclusions: pd.DataFrame
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class PrequentialRun:
    predictions: pd.DataFrame
    exclusions: pd.DataFrame
    audit: Mapping[str, Any]
    model: "DynamicBradleyTerry"


@dataclass(frozen=True)
class BinaryEvaluation:
    n: int
    log_loss: float
    brier: float
    calibration_inputs: pd.DataFrame
    calibration_bins: pd.DataFrame


@dataclass(frozen=True)
class CalibrationConfig:
    """Validation-only probability-calibration selection."""

    mode: str = "auto"
    l2_identity_centered: float = 1e-6
    min_slope: float = 0.05
    max_slope: float = 5.0
    max_abs_intercept: float = 8.0
    slope_tolerance: float = 0.10
    minimum_log_loss_improvement: float = 1e-6

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "identity", "platt"}:
            raise ValueError(
                "calibration mode must be auto, identity, or platt"
            )
        if (
            not math.isfinite(self.l2_identity_centered)
            or self.l2_identity_centered < 0
        ):
            raise ValueError("calibration L2 penalty cannot be negative")
        if (
            not math.isfinite(self.min_slope)
            or not math.isfinite(self.max_slope)
            or not 0 <= self.min_slope < self.max_slope
        ):
            raise ValueError("calibration slope bounds are invalid")
        if (
            not math.isfinite(self.max_abs_intercept)
            or self.max_abs_intercept <= 0
        ):
            raise ValueError("max_abs_intercept must be positive")
        if (
            not math.isfinite(self.slope_tolerance)
            or not 0 <= self.slope_tolerance < 1
        ):
            raise ValueError("slope_tolerance must be in [0, 1)")
        if (
            not math.isfinite(self.minimum_log_loss_improvement)
            or self.minimum_log_loss_improvement < 0
        ):
            raise ValueError(
                "minimum_log_loss_improvement cannot be negative"
            )


@dataclass(frozen=True)
class ProbabilityCalibration:
    """Frozen identity or Platt map fitted exclusively on validation labels."""

    method: str
    intercept: float
    slope: float
    probability_floor: float
    fit_split: str
    fit_rows: int
    fit_row_ids_sha256: str
    validation_log_loss: float
    validation_brier: float
    diagnostics: Mapping[str, Any]

    def apply(
        self,
        probability: np.ndarray | pd.Series | Sequence[float],
        *,
        side_indicator: (
            np.ndarray | pd.Series | Sequence[float] | float
        ) = 1.0,
    ) -> np.ndarray:
        """Apply frozen calibration without consulting any outcomes.

        The intercept is a side nuisance term.  Negating it with a swapped
        ``side_indicator`` preserves complementary side-swap predictions.
        """

        raw = np.asarray(probability, dtype=float)
        if raw.ndim != 1 or not np.isfinite(raw).all():
            raise ValueError("calibration probabilities must be finite 1D")
        if np.any((raw <= 0.0) | (raw >= 1.0)):
            raise ValueError(
                "calibration probabilities must be strictly within (0, 1)"
            )
        if self.method == "identity":
            return raw.copy()
        side = np.asarray(side_indicator, dtype=float)
        if side.ndim == 0:
            side = np.full(len(raw), float(side), dtype=float)
        if side.shape != raw.shape or not np.isin(side, [-1.0, 1.0]).all():
            raise ValueError(
                "calibration side_indicator must be scalar or aligned +/-1"
            )
        raw_logit = np.log(raw) - np.log1p(-raw)
        calibrated_logit = self.intercept * side + self.slope * raw_logit
        calibrated = np.array(
            [
                _bounded_antisymmetric_probability(
                    value, 0.0, self.probability_floor
                )
                for value in calibrated_logit
            ],
            dtype=float,
        )
        return calibrated


@dataclass(frozen=True)
class HyperparameterCandidate:
    name: str
    config: DynamicBTConfig


@dataclass(frozen=True)
class HyperparameterTournamentResult:
    selected_candidate: str
    selected_config: DynamicBTConfig
    calibration: ProbabilityCalibration
    calibration_scores: pd.DataFrame
    validation_scores: pd.DataFrame
    validation_raw_evaluation: BinaryEvaluation
    validation_calibrated_evaluation: BinaryEvaluation
    test_raw_evaluation: BinaryEvaluation
    test_calibrated_evaluation: BinaryEvaluation
    validation_raw_ledger: pd.DataFrame
    validation_calibrated_ledger: pd.DataFrame
    test_raw_ledger: pd.DataFrame
    test_calibrated_ledger: pd.DataFrame
    # Backward-compatible aliases retain the original raw-score meaning.
    validation_evaluation: BinaryEvaluation
    test_evaluation: BinaryEvaluation
    validation_predictions: pd.DataFrame
    test_predictions: pd.DataFrame
    exclusions: pd.DataFrame
    audit: Mapping[str, Any]

    def promotion_evidence(
        self,
        comparison: pd.DataFrame | Sequence[float] | np.ndarray,
        *,
        canonical_series_ids: (
            Mapping[str, str] | pd.Series | Sequence[str] | None
        ) = None,
        candidate_variant: str = "calibrated",
        baseline_probability_column: str = "probability",
        baseline_row_id_column: str = "row_id",
        candidate_model_id: str = "dynamic_bt_candidate",
        baseline_model_id: str = "baseline",
        estimand_id: str = "organization_map_win",
        primary_score: str = "log_loss",
        minimum_test_events: int = 1,
        bootstrap_replicates: int = 1000,
        moving_block_size: int = 8,
        alpha: float = 0.05,
        noninferiority_margin: float = 0.0,
        random_seed: int = 0,
    ) -> dict[str, Any]:
        """Build paired series-block evidence without authorizing promotion.

        A complete canonical two-model prediction ledger is delegated to
        :mod:`lol_kills.model_tournament`.  A probability vector/frame instead
        requires explicit canonical series ids and uses the same event-weighted
        circular moving-block bootstrap locally.
        """

        return _promotion_evidence(
            self,
            comparison,
            canonical_series_ids=canonical_series_ids,
            candidate_variant=candidate_variant,
            baseline_probability_column=baseline_probability_column,
            baseline_row_id_column=baseline_row_id_column,
            candidate_model_id=candidate_model_id,
            baseline_model_id=baseline_model_id,
            estimand_id=estimand_id,
            primary_score=primary_score,
            minimum_test_events=minimum_test_events,
            bootstrap_replicates=bootstrap_replicates,
            moving_block_size=moving_block_size,
            alpha=alpha,
            noninferiority_margin=noninferiority_margin,
            random_seed=random_seed,
        )


def _clean_key(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.casefold() in {"", "nan", "none", "<na>"} else text


def _utc_timestamp(value: Any) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _iso(value: pd.Timestamp | None) -> str | None:
    return value.isoformat() if value is not None else None


def _bounded_antisymmetric_probability(
    latent_logit: float,
    predictive_variance: float,
    probability_floor: float,
) -> float:
    """Return a bounded probability with exact ``p(-x) == 1 - p(x)``."""

    variance = max(float(predictive_variance), 0.0)
    scale = math.sqrt(1.0 + math.pi * variance / 8.0)
    magnitude = abs(float(latent_logit)) / scale
    if magnitude >= 40.0:
        unit = 1.0
    else:
        unit = 1.0 / (1.0 + math.exp(-magnitude))
    upper_side = probability_floor + (1.0 - 2.0 * probability_floor) * unit
    if latent_logit > 0:
        return upper_side
    if latent_logit < 0:
        return 1.0 - upper_side
    return 0.5


def _plain_logistic(value: float) -> float:
    if value >= 0:
        tail = math.exp(-min(value, 40.0))
        return 1.0 / (1.0 + tail)
    tail = math.exp(max(value, -40.0))
    return tail / (1.0 + tail)


def prepare_dynamic_bt_data(
    maps: pd.DataFrame,
    *,
    columns: DynamicBTColumns | None = None,
    data_cutoff: Any = None,
) -> PreparedDynamicBTData:
    """Validate, audit, and deterministically sort map-grain observations."""

    columns = columns or DynamicBTColumns()
    frame = pd.DataFrame() if maps is None else maps.copy()
    required = {
        columns.row_id,
        columns.timestamp,
        columns.outcome,
        columns.blue_team_key,
        columns.red_team_key,
    }
    missing = sorted(column for column in required if column not in frame.columns)
    if len(frame) and missing:
        raise ValueError(f"missing required dynamic BT columns: {missing}")

    requested_cutoff = _utc_timestamp(data_cutoff) if data_cutoff is not None else None
    if data_cutoff is not None and requested_cutoff is None:
        raise ValueError("data_cutoff is not a valid timestamp")

    records: list[dict[str, Any]] = []
    for source_position, (source_index, row) in enumerate(frame.iterrows()):
        row_id = _clean_key(row.get(columns.row_id))
        timestamp = _utc_timestamp(row.get(columns.timestamp))
        blue_team = _clean_key(row.get(columns.blue_team_key))
        red_team = _clean_key(row.get(columns.red_team_key))
        blue_context = (
            _clean_key(row.get(columns.blue_context))
            if columns.blue_context and columns.blue_context in frame.columns
            else ""
        )
        red_context = (
            _clean_key(row.get(columns.red_context))
            if columns.red_context and columns.red_context in frame.columns
            else ""
        )
        competition = (
            _clean_key(row.get(columns.competition))
            if columns.competition and columns.competition in frame.columns
            else ""
        )
        outcome_raw = pd.to_numeric(
            pd.Series([row.get(columns.outcome)]), errors="coerce"
        ).iloc[0]
        side_raw = (
            row.get(columns.side_indicator)
            if columns.side_indicator and columns.side_indicator in frame.columns
            else 1.0
        )
        side_numeric = pd.to_numeric(
            pd.Series([side_raw]), errors="coerce"
        ).iloc[0]

        reasons: list[str] = []
        if not row_id:
            reasons.append("missing_row_id")
        if timestamp is None:
            reasons.append("invalid_timestamp")
        if pd.isna(outcome_raw) or float(outcome_raw) not in (0.0, 1.0):
            reasons.append("invalid_binary_outcome")
        if not blue_team or not red_team:
            reasons.append("missing_team_key")
        elif blue_team == red_team:
            reasons.append("self_match")
        if pd.isna(side_numeric) or float(side_numeric) not in (-1.0, 1.0):
            reasons.append("invalid_side_indicator")
        if (
            requested_cutoff is not None
            and timestamp is not None
            and timestamp > requested_cutoff
        ):
            reasons.append("after_data_cutoff")

        records.append(
            {
                "_source_position": source_position,
                "_source_index": str(source_index),
                "row_id": row_id,
                "timestamp": timestamp,
                "y_true": (
                    float(outcome_raw) if not pd.isna(outcome_raw) else math.nan
                ),
                "blue_team_key": blue_team,
                "red_team_key": red_team,
                "blue_context": blue_context,
                "red_context": red_context,
                "competition": competition or "UNKNOWN",
                "side_indicator": (
                    float(side_numeric) if not pd.isna(side_numeric) else math.nan
                ),
                "_reasons": reasons,
            }
        )

    # Keep the earliest uniquely identified occurrence of a row id.  Later
    # duplicates cannot retroactively remove an earlier prediction.  If the
    # earliest timestamp itself is duplicated, all rows at that timestamp are
    # ambiguous and excluded.
    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["row_id"]:
            by_id[record["row_id"]].append(record)
    for duplicates in by_id.values():
        if len(duplicates) <= 1:
            continue
        dated = [record for record in duplicates if record["timestamp"] is not None]
        if not dated:
            for record in duplicates:
                record["_reasons"].append("duplicate_row_id")
            continue
        earliest = min(record["timestamp"] for record in dated)
        earliest_rows = [
            record for record in dated if record["timestamp"] == earliest
        ]
        if len(earliest_rows) > 1:
            for record in duplicates:
                record["_reasons"].append("duplicate_row_id")
        else:
            keeper = earliest_rows[0]
            for record in duplicates:
                if record is not keeper:
                    record["_reasons"].append("duplicate_row_id_after_first")

    accepted_rows = [
        {key: value for key, value in record.items() if not key.startswith("_")}
        for record in records
        if not record["_reasons"]
    ]
    accepted = pd.DataFrame(
        accepted_rows,
        columns=[
            "row_id",
            "timestamp",
            "y_true",
            "blue_team_key",
            "red_team_key",
            "blue_context",
            "red_context",
            "competition",
            "side_indicator",
        ],
    )
    if not accepted.empty:
        accepted = accepted.sort_values(
            ["timestamp", "row_id"], kind="mergesort"
        ).reset_index(drop=True)

    exclusion_rows: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    for record in records:
        unique_reasons = sorted(set(record["_reasons"]))
        if not unique_reasons:
            continue
        reason_counts.update(unique_reasons)
        exclusion_rows.append(
            {
                "source_index": record["_source_index"],
                "row_id": record["row_id"] or None,
                "timestamp": _iso(record["timestamp"]),
                "reasons": unique_reasons,
            }
        )
    exclusions = pd.DataFrame(
        exclusion_rows,
        columns=["source_index", "row_id", "timestamp", "reasons"],
    )
    if not exclusions.empty:
        exclusions["_reason_sort"] = exclusions["reasons"].map(
            lambda values: "|".join(values)
        )
        exclusions = (
            exclusions.sort_values(
                ["timestamp", "row_id", "_reason_sort"],
                na_position="last",
                kind="mergesort",
            )
            .drop(columns="_reason_sort")
            .reset_index(drop=True)
        )

    date_min = accepted["timestamp"].min() if not accepted.empty else None
    date_max = accepted["timestamp"].max() if not accepted.empty else None
    effective_cutoff = requested_cutoff if requested_cutoff is not None else date_max
    audit: dict[str, Any] = {
        "status": RESEARCH_STATUS,
        "source_rows": int(len(frame)),
        "accepted_map_rows": int(len(accepted)),
        "excluded_rows": int(len(exclusions)),
        "row_exclusion_counts": dict(sorted(reason_counts.items())),
        "data_start": _iso(date_min),
        "data_end": _iso(date_max),
        "data_cutoff": _iso(effective_cutoff),
        "data_cutoff_source": (
            "supplied" if requested_cutoff is not None else "observed_max"
        ),
        "sort_keys": ["timestamp_utc", "immutable_row_id"],
        "same_timestamp_policy": (
            "predict every map in the timestamp batch before assimilating "
            "any outcome from that batch"
        ),
        "team_identity_policy": (
            "use supplied immutable team keys verbatim; never derive identity "
            "from competition or display name"
        ),
        "observation_unit": "binary map; no scheduled-series-format assumption",
        "required_columns": sorted(required),
    }
    return PreparedDynamicBTData(accepted, exclusions, audit)


class DynamicBradleyTerry:
    """Mutable online state for the dynamic research candidate."""

    def __init__(self, config: DynamicBTConfig | None = None) -> None:
        self.config = config or DynamicBTConfig()
        self.teams: dict[str, GaussianState] = {}
        self.contexts: dict[str, GaussianState] = {}
        self.blue_side = GaussianState(
            mean=self.config.blue_side_prior_logit,
            variance=self.config.blue_side_prior_sd**2,
        )
        self.bridges = _BridgeTracker(self.config)
        self.observed_maps = 0

    def _propagate(
        self,
        state: GaussianState,
        timestamp: pd.Timestamp,
        *,
        variance_per_day: float,
        max_variance: float,
        mean_revert: bool,
    ) -> None:
        if state.last_timestamp is None:
            state.last_timestamp = timestamp
            return
        if timestamp < state.last_timestamp:
            raise ValueError("dynamic state cannot be propagated backwards in time")
        elapsed_days = (
            timestamp - state.last_timestamp
        ).total_seconds() / 86400.0
        if elapsed_days <= 0:
            return
        if mean_revert and self.config.mean_reversion_half_life_days is not None:
            decay = math.exp(
                -math.log(2.0)
                * elapsed_days
                / self.config.mean_reversion_half_life_days
            )
            state.mean *= decay
        # This is the Gaussian random-walk transition.  Because propagation is
        # incremental, prediction-only calls cannot double-count inactivity.
        state.variance = min(
            max_variance,
            max(
                self.config.min_variance,
                state.variance + variance_per_day * elapsed_days,
            ),
        )
        state.last_timestamp = timestamp

    def _team(self, key: str, timestamp: pd.Timestamp) -> GaussianState:
        state = self.teams.get(key)
        if state is None:
            state = GaussianState(0.0, self.config.team_prior_sd**2)
            self.teams[key] = state
        self._propagate(
            state,
            timestamp,
            variance_per_day=self.config.team_variance_per_day,
            max_variance=self.config.max_team_variance,
            mean_revert=True,
        )
        return state

    def _context(self, key: str, timestamp: pd.Timestamp) -> GaussianState:
        state = self.contexts.get(key)
        if state is None:
            state = GaussianState(0.0, self.config.context_prior_sd**2)
            self.contexts[key] = state
        self._propagate(
            state,
            timestamp,
            variance_per_day=self.config.context_variance_per_day,
            max_variance=self.config.max_context_variance,
            mean_revert=True,
        )
        return state

    def _propagate_side(self, timestamp: pd.Timestamp) -> None:
        self._propagate(
            self.blue_side,
            timestamp,
            variance_per_day=self.config.side_variance_per_day,
            max_variance=self.config.max_side_variance,
            mean_revert=False,
        )

    def predict(
        self,
        blue_team_key: str,
        red_team_key: str,
        *,
        timestamp: Any,
        blue_context: str = "",
        red_context: str = "",
        side_indicator: float = 1.0,
    ) -> DynamicBTPrediction:
        """Predict before observing an outcome.

        ``side_indicator=+1`` means the first team is the actual blue side.
        For a complementary side-swap check, swap teams and contexts and pass
        ``side_indicator=-1``.
        """

        blue_team_key = _clean_key(blue_team_key)
        red_team_key = _clean_key(red_team_key)
        if not blue_team_key or not red_team_key:
            raise ValueError("prediction requires non-empty immutable team keys")
        if blue_team_key == red_team_key:
            raise ValueError("prediction requires two different team keys")
        if float(side_indicator) not in (-1.0, 1.0):
            raise ValueError("side_indicator must be +1 or -1")
        moment = _utc_timestamp(timestamp)
        if moment is None:
            raise ValueError("prediction timestamp is invalid")
        blue_context = _clean_key(blue_context)
        red_context = _clean_key(red_context)

        blue = self._team(blue_team_key, moment)
        red = self._team(red_team_key, moment)
        self._propagate_side(moment)
        bridge = self.bridges.support(blue_context, red_context)

        blue_context_state: GaussianState | None = None
        red_context_state: GaussianState | None = None
        context_logit = 0.0
        context_variance = 0.0
        if bridge.status == "supported":
            blue_context_state = self._context(blue_context, moment)
            red_context_state = self._context(red_context, moment)
            context_logit = blue_context_state.mean - red_context_state.mean
            context_variance = math.fsum(
                (blue_context_state.variance, red_context_state.variance)
            )
        elif bridge.status == "unsupported":
            context_variance = self.config.unsupported_bridge_variance

        team_logit = blue.mean - red.mean
        side_logit = float(side_indicator) * self.blue_side.mean
        latent_logit = math.fsum((side_logit, team_logit, context_logit))
        predictive_variance = math.fsum(
            (
                self.blue_side.variance,
                blue.variance,
                red.variance,
                context_variance,
            )
        )
        probability = _bounded_antisymmetric_probability(
            latent_logit,
            predictive_variance,
            self.config.probability_floor,
        )
        return DynamicBTPrediction(
            probability=probability,
            latent_logit=latent_logit,
            predictive_variance=predictive_variance,
            predictive_sigma=math.sqrt(predictive_variance),
            blue_team_mean=blue.mean,
            red_team_mean=red.mean,
            blue_team_variance=blue.variance,
            red_team_variance=red.variance,
            blue_side_logit=self.blue_side.mean,
            blue_side_variance=self.blue_side.variance,
            blue_context_mean=(
                blue_context_state.mean if blue_context_state is not None else 0.0
            ),
            red_context_mean=(
                red_context_state.mean if red_context_state is not None else 0.0
            ),
            bridge_status=bridge.status,
            bridge_component=bridge.component,
            bridge_direct_maps=bridge.direct_maps,
        )

    def register_bridge_evidence(
        self,
        *,
        blue_team_key: str,
        red_team_key: str,
        blue_context: str,
        red_context: str,
        competition: str,
    ) -> None:
        self.bridges.register(
            _clean_key(blue_context),
            _clean_key(red_context),
            _clean_key(blue_team_key),
            _clean_key(red_team_key),
            _clean_key(competition) or "UNKNOWN",
        )

    def observe(
        self,
        blue_team_key: str,
        red_team_key: str,
        outcome: float,
        *,
        timestamp: Any,
        blue_context: str = "",
        red_context: str = "",
        side_indicator: float = 1.0,
    ) -> None:
        """Assimilate one already-predicted binary map outcome."""

        if float(outcome) not in (0.0, 1.0):
            raise ValueError("outcome must be binary")
        moment = _utc_timestamp(timestamp)
        if moment is None:
            raise ValueError("observation timestamp is invalid")
        blue_team_key = _clean_key(blue_team_key)
        red_team_key = _clean_key(red_team_key)
        if not blue_team_key or not red_team_key:
            raise ValueError("observation requires non-empty immutable team keys")
        if blue_team_key == red_team_key:
            raise ValueError("observation requires two different team keys")
        if float(side_indicator) not in (-1.0, 1.0):
            raise ValueError("side_indicator must be +1 or -1")
        blue_context = _clean_key(blue_context)
        red_context = _clean_key(red_context)
        blue = self._team(blue_team_key, moment)
        red = self._team(red_team_key, moment)
        self._propagate_side(moment)

        features: list[tuple[GaussianState, float, float]] = [
            (blue, 1.0, self.config.max_team_variance),
            (red, -1.0, self.config.max_team_variance),
            (
                self.blue_side,
                float(side_indicator),
                self.config.max_side_variance,
            ),
        ]
        bridge = self.bridges.support(blue_context, red_context)
        if bridge.status == "supported":
            features.extend(
                [
                    (
                        self._context(blue_context, moment),
                        1.0,
                        self.config.max_context_variance,
                    ),
                    (
                        self._context(red_context, moment),
                        -1.0,
                        self.config.max_context_variance,
                    ),
                ]
            )

        latent_logit = math.fsum(
            coefficient * state.mean for state, coefficient, _ in features
        )
        latent_probability = _plain_logistic(latent_logit)
        curvature = max(
            latent_probability * (1.0 - latent_probability), 1e-9
        )
        linear_variance = math.fsum(
            coefficient * coefficient * state.variance
            for state, coefficient, _ in features
        )
        denominator = 1.0 + curvature * linear_variance
        residual = float(outcome) - latent_probability
        updates: list[tuple[GaussianState, float, float]] = []
        for state, coefficient, max_variance in features:
            old_variance = state.variance
            gain = old_variance * coefficient / denominator
            new_mean = state.mean + gain * residual
            new_variance = old_variance - (
                curvature
                * (old_variance * coefficient) ** 2
                / denominator
            )
            updates.append(
                (
                    state,
                    min(
                        self.config.max_abs_mean,
                        max(-self.config.max_abs_mean, new_mean),
                    ),
                    min(
                        max_variance,
                        max(self.config.min_variance, new_variance),
                    ),
                )
            )
        for state, new_mean, new_variance in updates:
            state.mean = new_mean
            state.variance = new_variance
            state.last_observed = moment
            state.observations += 1
        self.observed_maps += 1

    def audit_snapshot(self) -> dict[str, Any]:
        return {
            "status": RESEARCH_STATUS,
            "observed_maps": self.observed_maps,
            "n_team_states": len(self.teams),
            "n_context_states": len(self.contexts),
            "blue_side": {
                "mean_logit": self.blue_side.mean,
                "sd": math.sqrt(self.blue_side.variance),
                "observations": self.blue_side.observations,
            },
            "bridge": self.bridges.audit(),
            "config": asdict(self.config),
        }


def _prediction_record(
    row: pd.Series,
    prediction: DynamicBTPrediction,
) -> dict[str, Any]:
    return {
        "row_id": row["row_id"],
        "timestamp": row["timestamp"],
        "y_true": float(row["y_true"]),
        "blue_team_key": row["blue_team_key"],
        "red_team_key": row["red_team_key"],
        "blue_context": row["blue_context"],
        "red_context": row["red_context"],
        "competition": row["competition"],
        "side_indicator": float(row["side_indicator"]),
        "p_blue": prediction.probability,
        "latent_logit": prediction.latent_logit,
        "predictive_variance": prediction.predictive_variance,
        "predictive_sigma": prediction.predictive_sigma,
        "blue_team_mean": prediction.blue_team_mean,
        "red_team_mean": prediction.red_team_mean,
        "blue_team_variance": prediction.blue_team_variance,
        "red_team_variance": prediction.red_team_variance,
        "blue_side_logit": prediction.blue_side_logit,
        "blue_side_variance": prediction.blue_side_variance,
        "blue_context_mean": prediction.blue_context_mean,
        "red_context_mean": prediction.red_context_mean,
        "bridge_status": prediction.bridge_status,
        "bridge_component": prediction.bridge_component,
        "bridge_direct_maps": prediction.bridge_direct_maps,
        "prediction_before_outcome": True,
    }


def _run_prepared_frame(
    frame: pd.DataFrame,
    config: DynamicBTConfig,
) -> tuple[pd.DataFrame, DynamicBradleyTerry]:
    model = DynamicBradleyTerry(config)
    records: list[dict[str, Any]] = []
    if frame.empty:
        return pd.DataFrame(), model

    ordered = frame.sort_values(["timestamp", "row_id"], kind="mergesort")
    for _, batch in ordered.groupby("timestamp", sort=True):
        batch = batch.sort_values("row_id", kind="mergesort")

        # Every prediction in this exact timestamp batch is emitted from the
        # same outcome-free state.
        for _, row in batch.iterrows():
            prediction = model.predict(
                row["blue_team_key"],
                row["red_team_key"],
                timestamp=row["timestamp"],
                blue_context=row["blue_context"],
                red_context=row["red_context"],
                side_indicator=float(row["side_indicator"]),
            )
            records.append(_prediction_record(row, prediction))

        # Structural bridge evidence contains no outcome and is registered
        # only after all batch predictions.  Outcomes are then assimilated in
        # immutable row-id order for deterministic future state.
        for _, row in batch.iterrows():
            model.register_bridge_evidence(
                blue_team_key=row["blue_team_key"],
                red_team_key=row["red_team_key"],
                blue_context=row["blue_context"],
                red_context=row["red_context"],
                competition=row["competition"],
            )
        for _, row in batch.iterrows():
            model.observe(
                row["blue_team_key"],
                row["red_team_key"],
                float(row["y_true"]),
                timestamp=row["timestamp"],
                blue_context=row["blue_context"],
                red_context=row["red_context"],
                side_indicator=float(row["side_indicator"]),
            )

    predictions = pd.DataFrame(records)
    return predictions, model


def run_prequential_dynamic_bt(
    maps: pd.DataFrame,
    *,
    config: DynamicBTConfig | None = None,
    columns: DynamicBTColumns | None = None,
    data_cutoff: Any = None,
) -> PrequentialRun:
    """Run chronological map predictions, always before the corresponding outcome."""

    config = config or DynamicBTConfig()
    prepared = prepare_dynamic_bt_data(
        maps, columns=columns, data_cutoff=data_cutoff
    )
    predictions, model = _run_prepared_frame(prepared.frame, config)
    audit = {
        **dict(prepared.audit),
        "prediction_protocol": (
            "chronological online/prequential; predict before each outcome; "
            "same-timestamp maps predicted before any same-timestamp update"
        ),
        "model": model.audit_snapshot(),
    }
    return PrequentialRun(predictions, prepared.exclusions, audit, model)


def evaluate_binary_predictions(
    predictions: pd.DataFrame,
    *,
    probability_column: str = "p_blue",
    outcome_column: str = "y_true",
    calibration_bins: int = 10,
) -> BinaryEvaluation:
    """Compute proper scores and retain row-level calibration inputs."""

    if calibration_bins < 2:
        raise ValueError("calibration_bins must be at least two")
    required = {probability_column, outcome_column}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"missing prediction columns: {missing}")
    if predictions.empty:
        empty_inputs = pd.DataFrame(
            columns=["row_id", "timestamp", "y_true", "probability", "bin"]
        )
        empty_bins = pd.DataFrame(
            columns=[
                "bin",
                "lower",
                "upper",
                "n",
                "mean_probability",
                "observed_rate",
            ]
        )
        return BinaryEvaluation(0, math.nan, math.nan, empty_inputs, empty_bins)

    probability = pd.to_numeric(
        predictions[probability_column], errors="coerce"
    ).to_numpy(float)
    outcome = pd.to_numeric(
        predictions[outcome_column], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(probability).all() or not np.isfinite(outcome).all():
        raise ValueError("evaluation inputs must be finite")
    if not np.isin(outcome, [0.0, 1.0]).all():
        raise ValueError("evaluation outcomes must be binary")
    if not ((probability > 0.0) & (probability < 1.0)).all():
        raise ValueError("evaluation probabilities must be strictly bounded")

    log_loss = float(
        -np.mean(
            outcome * np.log(probability)
            + (1.0 - outcome) * np.log1p(-probability)
        )
    )
    brier = float(np.mean((probability - outcome) ** 2))
    bin_index = np.minimum(
        (probability * calibration_bins).astype(int),
        calibration_bins - 1,
    )
    calibration_inputs = pd.DataFrame(
        {
            "row_id": (
                predictions["row_id"].astype(str).to_numpy()
                if "row_id" in predictions
                else np.arange(len(predictions)).astype(str)
            ),
            "timestamp": (
                predictions["timestamp"].to_numpy()
                if "timestamp" in predictions
                else pd.Series([pd.NaT] * len(predictions)).to_numpy()
            ),
            "y_true": outcome,
            "probability": probability,
            "bin": bin_index,
        }
    )
    grouped_rows: list[dict[str, Any]] = []
    for index, group in calibration_inputs.groupby("bin", sort=True):
        grouped_rows.append(
            {
                "bin": int(index),
                "lower": float(index / calibration_bins),
                "upper": float((index + 1) / calibration_bins),
                "n": int(len(group)),
                "mean_probability": float(group["probability"].mean()),
                "observed_rate": float(group["y_true"].mean()),
            }
        )
    return BinaryEvaluation(
        n=int(len(predictions)),
        log_loss=log_loss,
        brier=brier,
        calibration_inputs=calibration_inputs,
        calibration_bins=pd.DataFrame(grouped_rows),
    )


def _calibration_row_hash(predictions: pd.DataFrame) -> str:
    row_ids = predictions["row_id"].astype(str).tolist()
    encoded = json.dumps(row_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slope_status(
    slope: float,
    *,
    tolerance: float,
    lower_bound: float,
    upper_bound: float,
) -> str:
    boundary_tolerance = 1e-7
    if slope <= lower_bound + boundary_tolerance:
        return "lower_boundary"
    if slope >= upper_bound - boundary_tolerance:
        return "upper_boundary"
    if slope < 1.0 - tolerance:
        return "below_one_overconfident"
    if slope > 1.0 + tolerance:
        return "above_one_underconfident"
    return "near_one"


def fit_validation_calibration(
    validation_predictions: pd.DataFrame,
    *,
    config: CalibrationConfig | None = None,
    probability_floor: float = 1e-6,
    calibration_bins: int = 10,
) -> tuple[ProbabilityCalibration, pd.DataFrame]:
    """Fit/select identity versus Platt using validation labels only."""

    config = config or CalibrationConfig()
    required = {"row_id", "y_true", "p_blue"}
    missing = sorted(required - set(validation_predictions.columns))
    if missing:
        raise ValueError(f"calibration input is missing columns: {missing}")
    if validation_predictions.empty:
        raise ValueError("calibration requires non-empty validation predictions")

    raw_probability = pd.to_numeric(
        validation_predictions["p_blue"], errors="coerce"
    ).to_numpy(float)
    outcome = pd.to_numeric(
        validation_predictions["y_true"], errors="coerce"
    ).to_numpy(float)
    if (
        not np.isfinite(raw_probability).all()
        or np.any((raw_probability <= 0.0) | (raw_probability >= 1.0))
        or not np.isin(outcome, [0.0, 1.0]).all()
    ):
        raise ValueError("calibration requires bounded probabilities and labels")
    side = (
        pd.to_numeric(
            validation_predictions["side_indicator"], errors="coerce"
        ).to_numpy(float)
        if "side_indicator" in validation_predictions
        else np.ones(len(validation_predictions), dtype=float)
    )
    if not np.isin(side, [-1.0, 1.0]).all():
        raise ValueError("calibration side indicators must be +/-1")

    row_hash = _calibration_row_hash(validation_predictions)
    raw_evaluation = evaluate_binary_predictions(
        validation_predictions,
        probability_column="p_blue",
        calibration_bins=calibration_bins,
    )
    identity_diagnostics: dict[str, Any] = {
        "optimizer_success": True,
        "optimizer_message": "identity map",
        "intercept_standard_error": None,
        "slope_standard_error": None,
        "slope_wald_z_vs_one": 0.0,
        "slope_ci95": [1.0, 1.0],
        "slope_status": "identity",
        "standard_error_method": "not_applicable",
        "test_labels_used": False,
    }
    identity = ProbabilityCalibration(
        method="identity",
        intercept=0.0,
        slope=1.0,
        probability_floor=probability_floor,
        fit_split="validation",
        fit_rows=int(len(validation_predictions)),
        fit_row_ids_sha256=row_hash,
        validation_log_loss=raw_evaluation.log_loss,
        validation_brier=raw_evaluation.brier,
        diagnostics=identity_diagnostics,
    )
    score_rows: list[dict[str, Any]] = [
        {
            "method": "identity",
            "intercept": 0.0,
            "slope": 1.0,
            "validation_log_loss": raw_evaluation.log_loss,
            "validation_brier": raw_evaluation.brier,
            "optimizer_success": True,
        }
    ]
    if config.mode == "identity":
        scores = pd.DataFrame(score_rows)
        scores["selected"] = True
        return identity, scores

    raw_logit = np.log(raw_probability) - np.log1p(-raw_probability)
    design = np.column_stack((side, raw_logit))
    center = np.array([0.0, 1.0], dtype=float)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        linear = design @ beta
        probability = np.array(
            [_plain_logistic(float(value)) for value in linear],
            dtype=float,
        )
        value = float(
            np.sum(np.logaddexp(0.0, linear) - outcome * linear)
            + 0.5
            * config.l2_identity_centered
            * np.square(beta - center).sum()
        )
        gradient = (
            design.T @ (probability - outcome)
            + config.l2_identity_centered * (beta - center)
        )
        return value, gradient

    optimizer = minimize(
        lambda beta: objective(beta)[0],
        center.copy(),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        bounds=[
            (-config.max_abs_intercept, config.max_abs_intercept),
            (config.min_slope, config.max_slope),
        ],
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-9},
    )
    beta = np.asarray(optimizer.x, dtype=float)
    intercept = float(beta[0])
    slope = float(beta[1])
    fitted_linear = design @ beta
    fitted_probability = np.array(
        [_plain_logistic(float(value)) for value in fitted_linear],
        dtype=float,
    )
    weights = fitted_probability * (1.0 - fitted_probability)
    hessian = (
        design.T @ (weights[:, None] * design)
        + config.l2_identity_centered * np.eye(2)
    )
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        covariance = np.linalg.pinv(hessian, rcond=1e-12)
    standard_errors = np.sqrt(
        np.maximum(np.diag(covariance), 0.0)
    )
    slope_se = float(standard_errors[1])
    slope_z = (
        float((slope - 1.0) / slope_se)
        if slope_se > 0 and math.isfinite(slope_se)
        else math.nan
    )
    slope_ci = [
        float(slope - 1.959963984540054 * slope_se),
        float(slope + 1.959963984540054 * slope_se),
    ]
    diagnostics: dict[str, Any] = {
        "optimizer_success": bool(optimizer.success),
        "optimizer_message": str(optimizer.message),
        "intercept_standard_error": float(standard_errors[0]),
        "slope_standard_error": slope_se,
        "slope_wald_z_vs_one": slope_z,
        "slope_ci95": slope_ci,
        "slope_status": _slope_status(
            slope,
            tolerance=config.slope_tolerance,
            lower_bound=config.min_slope,
            upper_bound=config.max_slope,
        ),
        "standard_error_method": (
            "local inverse penalized validation Hessian"
        ),
        "raw_logit_min": float(raw_logit.min()),
        "raw_logit_max": float(raw_logit.max()),
        "validation_positive_rate": float(outcome.mean()),
        "test_labels_used": False,
    }
    provisional = ProbabilityCalibration(
        method="platt",
        intercept=intercept,
        slope=slope,
        probability_floor=probability_floor,
        fit_split="validation",
        fit_rows=int(len(validation_predictions)),
        fit_row_ids_sha256=row_hash,
        validation_log_loss=math.nan,
        validation_brier=math.nan,
        diagnostics=diagnostics,
    )
    platt_frame = validation_predictions.copy()
    platt_frame["p_blue_platt"] = provisional.apply(
        raw_probability, side_indicator=side
    )
    platt_evaluation = evaluate_binary_predictions(
        platt_frame,
        probability_column="p_blue_platt",
        calibration_bins=calibration_bins,
    )
    platt = ProbabilityCalibration(
        method="platt",
        intercept=intercept,
        slope=slope,
        probability_floor=probability_floor,
        fit_split="validation",
        fit_rows=int(len(validation_predictions)),
        fit_row_ids_sha256=row_hash,
        validation_log_loss=platt_evaluation.log_loss,
        validation_brier=platt_evaluation.brier,
        diagnostics=diagnostics,
    )
    score_rows.append(
        {
            "method": "platt",
            "intercept": intercept,
            "slope": slope,
            "validation_log_loss": platt_evaluation.log_loss,
            "validation_brier": platt_evaluation.brier,
            "optimizer_success": bool(optimizer.success),
        }
    )

    if config.mode == "platt":
        if not optimizer.success:
            raise RuntimeError(
                f"validation Platt optimization failed: {optimizer.message}"
            )
        selected = platt
    else:
        log_loss_improvement = (
            raw_evaluation.log_loss - platt_evaluation.log_loss
        )
        brier_improved = platt_evaluation.brier < raw_evaluation.brier
        selected = (
            platt
            if optimizer.success
            and (
                log_loss_improvement
                > config.minimum_log_loss_improvement
                or (
                    math.isclose(
                        log_loss_improvement,
                        config.minimum_log_loss_improvement,
                        rel_tol=0.0,
                        abs_tol=1e-15,
                    )
                    and brier_improved
                )
            )
            else identity
        )
    scores = pd.DataFrame(score_rows)
    scores["selected"] = scores["method"].eq(selected.method)
    return selected, scores


def _add_calibrated_probabilities(
    predictions: pd.DataFrame,
    calibration: ProbabilityCalibration,
) -> pd.DataFrame:
    output = predictions.copy()
    output["p_blue_raw"] = output["p_blue"].to_numpy(float)
    output["p_blue_calibrated"] = calibration.apply(
        output["p_blue_raw"].to_numpy(float),
        side_indicator=output["side_indicator"].to_numpy(float),
    )
    output["calibration_method"] = calibration.method
    output["calibration_intercept"] = calibration.intercept
    output["calibration_slope"] = calibration.slope
    return output


def _probability_ledger(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    variant: str,
    split: str,
    calibration: ProbabilityCalibration,
) -> pd.DataFrame:
    ledger = predictions.copy()
    ledger["probability"] = ledger[probability_column].to_numpy(float)
    ledger["probability_variant"] = variant
    ledger["split"] = split
    ledger["calibration_fit_split"] = calibration.fit_split
    ledger["calibration_fit_rows"] = calibration.fit_rows
    ledger["test_labels_used_for_calibration"] = False
    return ledger


def default_hyperparameter_candidates() -> tuple[HyperparameterCandidate, ...]:
    """Small, deterministic research grid; intentionally not an exhaustive search."""

    return (
        HyperparameterCandidate(
            "conservative",
            DynamicBTConfig(
                team_prior_sd=0.75,
                team_variance_per_day=0.0015,
                mean_reversion_half_life_days=540.0,
            ),
        ),
        HyperparameterCandidate(
            "balanced",
            DynamicBTConfig(
                team_prior_sd=0.90,
                team_variance_per_day=0.003,
                mean_reversion_half_life_days=365.0,
            ),
        ),
        HyperparameterCandidate(
            "adaptive",
            DynamicBTConfig(
                team_prior_sd=1.00,
                team_variance_per_day=0.008,
                mean_reversion_half_life_days=180.0,
            ),
        ),
    )


def _candidate_fingerprint(
    candidate: HyperparameterCandidate,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
    data_cutoff: str | None,
) -> str:
    payload = {
        "candidate": candidate.name,
        "config": asdict(candidate.config),
        "validation_start": _iso(validation_start),
        "test_start": _iso(test_start),
        "data_cutoff": data_cutoff,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def _calibration_fingerprint(
    calibration: ProbabilityCalibration,
    candidate_fingerprint: str,
) -> str:
    payload = {
        "candidate_fingerprint": candidate_fingerprint,
        "method": calibration.method,
        "intercept": calibration.intercept,
        "slope": calibration.slope,
        "fit_split": calibration.fit_split,
        "fit_rows": calibration.fit_rows,
        "fit_row_ids_sha256": calibration.fit_row_ids_sha256,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def run_hyperparameter_tournament(
    maps: pd.DataFrame,
    *,
    validation_start: Any,
    test_start: Any,
    candidates: Sequence[HyperparameterCandidate] | None = None,
    columns: DynamicBTColumns | None = None,
    data_cutoff: Any = None,
    calibration_bins: int = 10,
    calibration_config: CalibrationConfig | None = None,
) -> HyperparameterTournamentResult:
    """Select hyperparameters/calibration on validation, then freeze for test.

    The selected configuration is refit prequentially from the start.  Test
    outcomes are allowed to update the online state only *after* their own
    predictions, so later test rows mimic deployment-time adaptation.  No
    unselected candidate is evaluated on test.  Identity versus Platt
    calibration is fitted and chosen using selected-model validation rows
    only, then applied to test probabilities without reading test outcomes.
    """

    validation_at = _utc_timestamp(validation_start)
    test_at = _utc_timestamp(test_start)
    if validation_at is None or test_at is None:
        raise ValueError("validation_start and test_start must be valid timestamps")
    if validation_at >= test_at:
        raise ValueError("validation_start must be earlier than test_start")

    candidate_grid = tuple(candidates or default_hyperparameter_candidates())
    if not candidate_grid:
        raise ValueError("at least one hyperparameter candidate is required")
    names = [candidate.name for candidate in candidate_grid]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("candidate names must be non-empty and unique")

    prepared = prepare_dynamic_bt_data(
        maps, columns=columns, data_cutoff=data_cutoff
    )
    train = prepared.frame[prepared.frame["timestamp"] < validation_at]
    validation = prepared.frame[
        (prepared.frame["timestamp"] >= validation_at)
        & (prepared.frame["timestamp"] < test_at)
    ]
    test = prepared.frame[prepared.frame["timestamp"] >= test_at]
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "temporal tournament requires non-empty train, validation, and test rows"
        )

    selection_frame = pd.concat([train, validation], ignore_index=True)
    score_rows: list[dict[str, Any]] = []
    for candidate in candidate_grid:
        candidate_predictions, _ = _run_prepared_frame(
            selection_frame, candidate.config
        )
        candidate_validation = candidate_predictions[
            (candidate_predictions["timestamp"] >= validation_at)
            & (candidate_predictions["timestamp"] < test_at)
        ].reset_index(drop=True)
        evaluation = evaluate_binary_predictions(
            candidate_validation, calibration_bins=calibration_bins
        )
        score_rows.append(
            {
                "candidate": candidate.name,
                "validation_n": evaluation.n,
                "validation_log_loss": evaluation.log_loss,
                "validation_brier": evaluation.brier,
            }
        )

    validation_scores = (
        pd.DataFrame(score_rows)
        .sort_values(
            [
                "validation_log_loss",
                "validation_brier",
                "candidate",
            ],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    selected_name = str(validation_scores.iloc[0]["candidate"])
    selected = next(
        candidate
        for candidate in candidate_grid
        if candidate.name == selected_name
    )

    final_predictions, final_model = _run_prepared_frame(
        prepared.frame, selected.config
    )
    selected_validation_raw = final_predictions[
        (final_predictions["timestamp"] >= validation_at)
        & (final_predictions["timestamp"] < test_at)
    ].reset_index(drop=True)
    test_raw = final_predictions[
        final_predictions["timestamp"] >= test_at
    ].reset_index(drop=True)
    calibration, calibration_scores = fit_validation_calibration(
        selected_validation_raw,
        config=calibration_config,
        probability_floor=selected.config.probability_floor,
        calibration_bins=calibration_bins,
    )
    selected_validation_predictions = _add_calibrated_probabilities(
        selected_validation_raw, calibration
    )
    test_predictions = _add_calibrated_probabilities(test_raw, calibration)
    validation_raw_evaluation = evaluate_binary_predictions(
        selected_validation_predictions,
        probability_column="p_blue_raw",
        calibration_bins=calibration_bins,
    )
    validation_calibrated_evaluation = evaluate_binary_predictions(
        selected_validation_predictions,
        probability_column="p_blue_calibrated",
        calibration_bins=calibration_bins,
    )
    test_raw_evaluation = evaluate_binary_predictions(
        test_predictions,
        probability_column="p_blue_raw",
        calibration_bins=calibration_bins,
    )
    test_calibrated_evaluation = evaluate_binary_predictions(
        test_predictions,
        probability_column="p_blue_calibrated",
        calibration_bins=calibration_bins,
    )
    validation_raw_ledger = _probability_ledger(
        selected_validation_predictions,
        probability_column="p_blue_raw",
        variant="raw",
        split="validation",
        calibration=calibration,
    )
    validation_calibrated_ledger = _probability_ledger(
        selected_validation_predictions,
        probability_column="p_blue_calibrated",
        variant=f"calibrated_{calibration.method}",
        split="validation",
        calibration=calibration,
    )
    test_raw_ledger = _probability_ledger(
        test_predictions,
        probability_column="p_blue_raw",
        variant="raw",
        split="test",
        calibration=calibration,
    )
    test_calibrated_ledger = _probability_ledger(
        test_predictions,
        probability_column="p_blue_calibrated",
        variant=f"calibrated_{calibration.method}",
        split="test",
        calibration=calibration,
    )
    selected_fingerprint = _candidate_fingerprint(
        selected,
        validation_at,
        test_at,
        str(prepared.audit.get("data_cutoff"))
        if prepared.audit.get("data_cutoff") is not None
        else None,
    )
    calibrated_fingerprint = _calibration_fingerprint(
        calibration, selected_fingerprint
    )
    audit: dict[str, Any] = {
        "status": RESEARCH_STATUS,
        "input": dict(prepared.audit),
        "split": {
            "validation_start": _iso(validation_at),
            "test_start": _iso(test_at),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
        },
        "selection": {
            "candidate_names": names,
            "criterion": (
                "minimum validation log loss; then validation Brier; "
                "then candidate name"
            ),
            "selected_candidate": selected.name,
            "selected_fingerprint": selected_fingerprint,
            "test_used_for_selection": False,
            "hyperparameters_frozen_before_test": True,
        },
        "calibration": {
            "config": asdict(calibration_config or CalibrationConfig()),
            "selected_method": calibration.method,
            "intercept": calibration.intercept,
            "slope": calibration.slope,
            "slope_diagnostics": dict(calibration.diagnostics),
            "fit_split": calibration.fit_split,
            "fit_rows": calibration.fit_rows,
            "fit_row_ids_sha256": calibration.fit_row_ids_sha256,
            "candidate_scores": calibration_scores.to_dict("records"),
            "test_labels_used": False,
            "frozen_calibrated_fingerprint": calibrated_fingerprint,
        },
        "test_protocol": (
            "single selected configuration; chronological prequential test "
            "predictions; each test outcome updates only later raw states; "
            "frozen validation calibration reads no test outcomes"
        ),
        "evaluation_aliases": {
            "validation_evaluation": "validation_raw_evaluation",
            "test_evaluation": "test_raw_evaluation",
        },
        "final_model": final_model.audit_snapshot(),
    }
    return HyperparameterTournamentResult(
        selected_candidate=selected.name,
        selected_config=selected.config,
        calibration=calibration,
        calibration_scores=calibration_scores,
        validation_scores=validation_scores,
        validation_raw_evaluation=validation_raw_evaluation,
        validation_calibrated_evaluation=(
            validation_calibrated_evaluation
        ),
        test_raw_evaluation=test_raw_evaluation,
        test_calibrated_evaluation=test_calibrated_evaluation,
        validation_raw_ledger=validation_raw_ledger,
        validation_calibrated_ledger=validation_calibrated_ledger,
        test_raw_ledger=test_raw_ledger,
        test_calibrated_ledger=test_calibrated_ledger,
        validation_evaluation=validation_raw_evaluation,
        test_evaluation=test_raw_evaluation,
        validation_predictions=selected_validation_predictions,
        test_predictions=test_predictions,
        exclusions=prepared.exclusions,
        audit=audit,
    )


def _candidate_test_for_variant(
    result: HyperparameterTournamentResult,
    candidate_variant: str,
) -> pd.DataFrame:
    if candidate_variant == "raw":
        return result.test_raw_ledger.copy()
    if candidate_variant == "calibrated":
        return result.test_calibrated_ledger.copy()
    raise ValueError("candidate_variant must be raw or calibrated")


def _series_ids_for_rows(
    row_ids: pd.Series,
    canonical_series_ids: (
        Mapping[str, str] | pd.Series | Sequence[str] | None
    ),
    baseline_frame: pd.DataFrame | None,
) -> np.ndarray:
    if canonical_series_ids is None:
        available_column = next(
            (
                column
                for column in ("canonical_series_id", "series_id")
                if baseline_frame is not None and column in baseline_frame
            ),
            None,
        )
        if baseline_frame is None or available_column is None:
            raise ValueError(
                "canonical_series_ids are required; series format is never "
                "inferred from map order"
            )
        values = baseline_frame[available_column].astype(str).to_numpy()
    elif isinstance(canonical_series_ids, Mapping):
        missing = [
            row_id
            for row_id in row_ids.astype(str)
            if row_id not in canonical_series_ids
        ]
        if missing:
            raise ValueError(
                f"canonical series ids missing row ids: {missing[:5]}"
            )
        values = np.array(
            [
                str(canonical_series_ids[row_id])
                for row_id in row_ids.astype(str)
            ],
            dtype=object,
        )
    elif isinstance(canonical_series_ids, pd.Series):
        index_as_text = canonical_series_ids.index.astype(str)
        row_id_set = set(row_ids.astype(str))
        if row_id_set.issubset(set(index_as_text)):
            lookup = {
                str(index): str(value)
                for index, value in canonical_series_ids.items()
            }
            values = np.array(
                [lookup[row_id] for row_id in row_ids.astype(str)],
                dtype=object,
            )
        else:
            values = canonical_series_ids.astype(str).to_numpy()
    else:
        values = np.asarray(list(canonical_series_ids), dtype=object)
    if len(values) != len(row_ids):
        raise ValueError("canonical series ids must align one-for-one with test rows")
    cleaned = np.array([_clean_key(value) for value in values], dtype=object)
    if any(not value for value in cleaned):
        raise ValueError("canonical series ids must be non-empty")
    return cleaned


def _promotion_evidence(
    result: HyperparameterTournamentResult,
    comparison: pd.DataFrame | Sequence[float] | np.ndarray,
    *,
    canonical_series_ids: (
        Mapping[str, str] | pd.Series | Sequence[str] | None
    ),
    candidate_variant: str,
    baseline_probability_column: str,
    baseline_row_id_column: str,
    candidate_model_id: str,
    baseline_model_id: str,
    estimand_id: str,
    primary_score: str,
    minimum_test_events: int,
    bootstrap_replicates: int,
    moving_block_size: int,
    alpha: float,
    noninferiority_margin: float,
    random_seed: int,
) -> dict[str, Any]:
    from lol_kills.model_tournament import (
        REQUIRED_PREDICTION_COLUMNS,
        TournamentSpec,
        corp_calibration_diagnostics,
        paired_moving_block_comparison,
        proper_score_vector,
    )

    spec = TournamentSpec(
        estimand_id=estimand_id,
        primary_score=primary_score,
        minimum_test_events=minimum_test_events,
        bootstrap_replicates=bootstrap_replicates,
        moving_block_size=moving_block_size,
        alpha=alpha,
        noninferiority_margin=noninferiority_margin,
        random_seed=random_seed,
    )
    spec.validate()
    candidate = _candidate_test_for_variant(result, candidate_variant)
    candidate = candidate.sort_values(
        ["timestamp", "row_id"], kind="mergesort"
    ).reset_index(drop=True)

    if isinstance(comparison, pd.DataFrame) and REQUIRED_PREDICTION_COLUMNS.issubset(
        comparison.columns
    ):
        canonical = comparison.copy()
        candidate_rows = canonical.loc[
            canonical["split"].eq("test")
            & canonical["model_id"].eq(candidate_model_id)
        ].copy()
        baseline_rows = canonical.loc[
            canonical["split"].eq("test")
            & canonical["model_id"].eq(baseline_model_id)
        ].copy()
        if candidate_rows.empty or baseline_rows.empty:
            raise ValueError(
                "canonical ledger must already contain both candidate and "
                "baseline test rows; candidate provenance is never fabricated"
            )
        expected = candidate.set_index("row_id")[
            ["probability", "y_true"]
        ].sort_index()
        observed = candidate_rows.assign(
            event_id=candidate_rows["event_id"].astype(str)
        ).set_index("event_id")[["probability", "outcome"]].sort_index()
        if not expected.index.equals(observed.index):
            raise ValueError(
                "canonical candidate events must exactly match dynamic test row ids"
            )
        if not np.allclose(
            observed["probability"].to_numpy(float),
            expected["probability"].to_numpy(float),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError(
                "canonical candidate probabilities do not match the retained "
                f"{candidate_variant} ledger"
            )
        if not np.array_equal(
            observed["outcome"].to_numpy(int),
            expected["y_true"].to_numpy(int),
        ):
            raise ValueError(
                "canonical candidate outcomes do not match frozen test labels"
            )
        report = paired_moving_block_comparison(
            canonical,
            candidate_model_id=candidate_model_id,
            baseline_model_id=baseline_model_id,
            spec=spec,
        )
        return {
            **report,
            "candidate_variant": candidate_variant,
            "evidence_path": "lol_kills.model_tournament canonical ledger",
            "research_status": RESEARCH_STATUS,
            "promotion_authorized": False,
        }

    baseline_frame: pd.DataFrame | None
    if isinstance(comparison, pd.DataFrame):
        baseline_frame = comparison.copy()
        required = {baseline_row_id_column, baseline_probability_column}
        missing = sorted(required - set(baseline_frame.columns))
        if missing:
            raise ValueError(
                f"baseline probability frame missing columns: {missing}"
            )
        if baseline_frame[baseline_row_id_column].duplicated().any():
            raise ValueError("baseline row ids must be unique")
        baseline_frame[baseline_row_id_column] = baseline_frame[
            baseline_row_id_column
        ].astype(str)
        candidate_ids = candidate["row_id"].astype(str)
        baseline_frame = (
            candidate_ids.to_frame(name=baseline_row_id_column)
            .merge(
                baseline_frame,
                on=baseline_row_id_column,
                how="left",
                validate="one_to_one",
                sort=False,
            )
        )
        if baseline_frame[baseline_probability_column].isna().any():
            raise ValueError(
                "baseline probabilities must cover every candidate test row"
            )
        if "outcome" in baseline_frame:
            if not np.array_equal(
                pd.to_numeric(
                    baseline_frame["outcome"], errors="raise"
                ).to_numpy(int),
                candidate["y_true"].to_numpy(int),
            ):
                raise ValueError(
                    "baseline outcomes do not match frozen candidate test labels"
                )
        baseline_probability = pd.to_numeric(
            baseline_frame[baseline_probability_column], errors="coerce"
        ).to_numpy(float)
    else:
        baseline_frame = None
        baseline_probability = np.asarray(list(comparison), dtype=float)
        if len(baseline_probability) != len(candidate):
            raise ValueError(
                "baseline probability vector must align with candidate test rows"
            )

    if (
        not np.isfinite(baseline_probability).all()
        or np.any((baseline_probability < 0.0) | (baseline_probability > 1.0))
    ):
        raise ValueError("baseline probabilities must be finite within [0, 1]")
    series_ids = _series_ids_for_rows(
        candidate["row_id"], canonical_series_ids, baseline_frame
    )
    candidate_probability = candidate["probability"].to_numpy(float)
    outcome = candidate["y_true"].to_numpy(float)
    candidate_score = proper_score_vector(
        outcome, candidate_probability, primary_score
    )
    baseline_score = proper_score_vector(
        outcome, baseline_probability, primary_score
    )
    paired = pd.DataFrame(
        {
            "row_id": candidate["row_id"].astype(str).to_numpy(),
            "timestamp": pd.to_datetime(
                candidate["timestamp"], errors="raise", utc=True
            ).to_numpy(),
            "series_id": series_ids,
            "score_delta": candidate_score - baseline_score,
        }
    )
    if len(paired) < minimum_test_events:
        raise ValueError(
            f"test population has {len(paired)} events; "
            f"minimum is {minimum_test_events}"
        )
    series_order = (
        paired.groupby("series_id", as_index=False)
        .agg(timestamp=("timestamp", "min"))
        .sort_values(["timestamp", "series_id"], kind="mergesort")
        ["series_id"]
        .tolist()
    )
    series_deltas = [
        paired.loc[
            paired["series_id"].eq(series_id), "score_delta"
        ].to_numpy(float)
        for series_id in series_order
    ]
    n_clusters = len(series_deltas)
    block_size = min(moving_block_size, n_clusters)
    rng = np.random.default_rng(random_seed)
    bootstrap = np.empty(bootstrap_replicates, dtype=float)
    blocks_needed = int(np.ceil(n_clusters / block_size))
    offsets = np.arange(block_size)
    for replicate in range(bootstrap_replicates):
        starts = rng.integers(0, n_clusters, size=blocks_needed)
        indices = ((starts[:, None] + offsets[None, :]) % n_clusters).ravel()
        sampled = np.concatenate(
            [series_deltas[index] for index in indices[:n_clusters]]
        )
        bootstrap[replicate] = float(sampled.mean())

    point = float(paired["score_delta"].mean())
    low, high = np.quantile(
        bootstrap, [alpha / 2.0, 1.0 - alpha / 2.0]
    )
    if high < 0.0:
        decision = "superior"
    elif high <= noninferiority_margin:
        decision = "noninferior"
    elif low > noninferiority_margin:
        decision = "inferior"
    else:
        decision = "inconclusive"
    candidate_calibration = {
        key: value
        for key, value in corp_calibration_diagnostics(
            outcome, candidate_probability
        ).items()
        if key != "fitted_probability"
    }
    baseline_calibration = {
        key: value
        for key, value in corp_calibration_diagnostics(
            outcome, baseline_probability
        ).items()
        if key != "fitted_probability"
    }
    return {
        "estimand_id": estimand_id,
        "candidate_model_id": candidate_model_id,
        "baseline_model_id": baseline_model_id,
        "candidate_variant": candidate_variant,
        "primary_score": primary_score,
        "events": int(len(paired)),
        "series_clusters": int(n_clusters),
        "candidate_score": float(candidate_score.mean()),
        "baseline_score": float(baseline_score.mean()),
        "candidate_minus_baseline": point,
        "confidence_level": float(1.0 - alpha),
        "confidence_interval": [float(low), float(high)],
        "noninferiority_margin": float(noninferiority_margin),
        "decision": decision,
        "bootstrap": {
            "method": (
                "paired circular moving-block bootstrap over ordered canonical "
                "series ids, with event-weighted proper scores"
            ),
            "replicates": int(bootstrap_replicates),
            "block_size_series": int(block_size),
            "seed": int(random_seed),
        },
        "candidate_calibration": candidate_calibration,
        "baseline_calibration": baseline_calibration,
        "evidence_path": "explicit canonical series ids",
        "research_status": RESEARCH_STATUS,
        "promotion_authorized": False,
        "spec": spec.to_dict(),
    }


__all__ = [
    "BinaryEvaluation",
    "CalibrationConfig",
    "DynamicBTColumns",
    "DynamicBTConfig",
    "DynamicBTPrediction",
    "DynamicBradleyTerry",
    "HyperparameterCandidate",
    "HyperparameterTournamentResult",
    "ProbabilityCalibration",
    "PrequentialRun",
    "PreparedDynamicBTData",
    "default_hyperparameter_candidates",
    "evaluate_binary_predictions",
    "fit_validation_calibration",
    "prepare_dynamic_bt_data",
    "run_hyperparameter_tournament",
    "run_prequential_dynamic_bt",
]
