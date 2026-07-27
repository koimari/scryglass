"""Leakage-safe series-level dynamic Bradley--Terry ratings.

The public entry points in this module deliberately consume
``hierarchical_bt._observations``.  That function is the repository's audited
one-row-per-completed-series contract.  Its canonical ``team_a``/``team_b``
orientation replaces map side, so this model has no blue-side, first-map-side,
pick-order, series-score, map-count, roster, or other post-series covariate.

The candidate is an auditable diagonal-Gaussian assumed-density filter:

* immutable organization keys own one state for their full history;
* predictions are emitted before outcomes are assimilated;
* every exact-timestamp batch is predicted from one outcome-free state and
  updated with one order-invariant diagonal Newton step;
* team and optional context uncertainty follows a Gaussian random walk during
  inactivity, with optional mean reversion;
* context offsets are used only after completed historical cross-context
  series satisfy explicit bridge-support thresholds.

Tournament selection is validation-only.  The selected dynamic configuration
and selected outcome-only rolling series Elo are frozen before the final test.
The final gate uses paired circular moving-block intervals over chronologically
ordered series, reports log loss, Brier score, and equal-width ECE, and passes
only when log-loss noninferiority or superiority is established against both
declared baselines.  Passing this gate is not a state-of-the-art claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from lol_kills.etl.series_ledger import build_canonical_series_ledger
from lol_kills.ratings.hierarchical_bt import (
    _observations as _verified_series_observations,
)


MODEL_ID = "series_dynamic_bt"
ELO_BASELINE_ID = "rolling_series_elo"
BASE_RATE_ID = "historical_symmetric_base_rate"
LOGIT_TO_RATING = 400.0 / math.log(10.0)
RATING_P05_Z = NormalDist().inv_cdf(0.95)
_OBSERVATION_HALF_LIFE_PLACEHOLDER_DAYS = 365.0
_MODEL_CODE_DEPENDENCIES = (
    Path(__file__),
    Path(__file__).with_name("hierarchical_bt.py"),
    Path(__file__).parents[1] / "etl" / "series_ledger.py",
    Path(__file__).parents[1] / "etl" / "competition.py",
    Path(__file__).parents[1] / "etl" / "aliases.py",
)


def _finite_positive(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _finite_nonnegative(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")


@dataclass(frozen=True)
class SeriesDynamicBTConfig:
    """Diagonal-Gaussian filter configuration, in natural-logit units."""

    team_prior_sigma: float = 0.90
    team_variance_per_day: float = 0.002
    mean_reversion_half_life_days: float | None = 365.0
    context_enabled: bool = False
    context_prior_sigma: float = 0.55
    context_variance_per_day: float = 0.0005
    min_bridge_series: int = 3
    min_bridge_teams_per_context: int = 2
    min_variance: float = 1e-5
    max_team_variance: float = 4.0
    max_context_variance: float = 2.0
    max_abs_mean: float = 8.0
    probability_floor: float = 1e-6
    base_rating: float = 1500.0

    def __post_init__(self) -> None:
        for name, value in {
            "team_prior_sigma": self.team_prior_sigma,
            "context_prior_sigma": self.context_prior_sigma,
            "min_variance": self.min_variance,
            "max_team_variance": self.max_team_variance,
            "max_context_variance": self.max_context_variance,
            "max_abs_mean": self.max_abs_mean,
        }.items():
            _finite_positive(name, float(value))
        for name, value in {
            "team_variance_per_day": self.team_variance_per_day,
            "context_variance_per_day": self.context_variance_per_day,
        }.items():
            _finite_nonnegative(name, float(value))
        if self.mean_reversion_half_life_days is not None:
            _finite_positive(
                "mean_reversion_half_life_days",
                float(self.mean_reversion_half_life_days),
            )
        if self.min_variance > min(
            self.max_team_variance, self.max_context_variance
        ):
            raise ValueError("min_variance cannot exceed a maximum variance")
        if self.min_bridge_series < 1:
            raise ValueError("min_bridge_series must be positive")
        if self.min_bridge_teams_per_context < 1:
            raise ValueError("min_bridge_teams_per_context must be positive")
        if not 0.0 <= self.probability_floor < 0.5:
            raise ValueError("probability_floor must be in [0, 0.5)")
        if not math.isfinite(self.base_rating):
            raise ValueError("base_rating must be finite")


@dataclass(frozen=True)
class SeriesEloConfig:
    """Outcome-only rolling Elo baseline configuration."""

    k_factor: float = 24.0
    scale: float = 400.0
    base_rating: float = 1500.0
    probability_floor: float = 1e-6

    def __post_init__(self) -> None:
        _finite_positive("k_factor", float(self.k_factor))
        _finite_positive("scale", float(self.scale))
        if not math.isfinite(self.base_rating):
            raise ValueError("base_rating must be finite")
        if not 0.0 <= self.probability_floor < 0.5:
            raise ValueError("probability_floor must be in [0, 0.5)")


@dataclass(frozen=True)
class SeriesTournamentSpec:
    """Predeclared chronological split and terminal gate."""

    validation_start: str
    test_start: str
    data_cutoff: str | None = None
    primary_score: str = "log_loss"
    ece_bins: int = 10
    bootstrap_replicates: int = 2000
    moving_block_size: int = 8
    alpha: float = 0.05
    noninferiority_margin: float = 0.005
    random_seed: int = 0
    minimum_test_series: int = 1

    def timestamps(self) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp | None]:
        validation = _timestamp(self.validation_start, "validation_start")
        test = _timestamp(self.test_start, "test_start")
        cutoff = (
            _timestamp(self.data_cutoff, "data_cutoff")
            if self.data_cutoff is not None
            else None
        )
        if validation >= test:
            raise ValueError("validation_start must precede test_start")
        if cutoff is not None and cutoff < test:
            raise ValueError("data_cutoff must be on or after test_start")
        return validation, test, cutoff

    def validate(self) -> None:
        self.timestamps()
        if self.primary_score != "log_loss":
            raise ValueError("the production gate primary score must be log_loss")
        if self.ece_bins < 2:
            raise ValueError("ece_bins must be at least two")
        if self.bootstrap_replicates < 100:
            raise ValueError("bootstrap_replicates must be at least 100")
        if self.moving_block_size < 1:
            raise ValueError("moving_block_size must be positive")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be strictly between zero and one")
        _finite_nonnegative(
            "noninferiority_margin", float(self.noninferiority_margin)
        )
        if self.minimum_test_series < 1:
            raise ValueError("minimum_test_series must be positive")


@dataclass
class GaussianState:
    mean: float
    variance: float
    last_timestamp: pd.Timestamp | None = None
    last_observed: pd.Timestamp | None = None
    observations: int = 0


@dataclass
class _BridgeEdge:
    series: int = 0
    left_teams: set[str] = field(default_factory=set)
    right_teams: set[str] = field(default_factory=set)

    def active(self, config: SeriesDynamicBTConfig) -> bool:
        return (
            self.series >= config.min_bridge_series
            and len(self.left_teams) >= config.min_bridge_teams_per_context
            and len(self.right_teams) >= config.min_bridge_teams_per_context
        )


class HistoricalBridgeTracker:
    """Completed-series support graph for optional context differences."""

    def __init__(self, config: SeriesDynamicBTConfig) -> None:
        self.config = config
        self.edges: dict[tuple[str, str], _BridgeEdge] = {}

    @staticmethod
    def _edge(
        context_a: str, context_b: str
    ) -> tuple[tuple[str, str], bool]:
        if context_a <= context_b:
            return (context_a, context_b), False
        return (context_b, context_a), True

    def register(
        self,
        context_a: str,
        context_b: str,
        team_a: str,
        team_b: str,
    ) -> None:
        if (
            not self.config.context_enabled
            or not _known_context(context_a)
            or not _known_context(context_b)
            or context_a == context_b
        ):
            return
        key, reversed_order = self._edge(context_a, context_b)
        edge = self.edges.setdefault(key, _BridgeEdge())
        edge.series += 1
        if reversed_order:
            edge.left_teams.add(team_b)
            edge.right_teams.add(team_a)
        else:
            edge.left_teams.add(team_a)
            edge.right_teams.add(team_b)

    def _active_graph(self) -> dict[str, set[str]]:
        graph: dict[str, set[str]] = defaultdict(set)
        for (left, right), edge in self.edges.items():
            if edge.active(self.config):
                graph[left].add(right)
                graph[right].add(left)
        return graph

    def supported(self, context_a: str, context_b: str) -> bool:
        if (
            not self.config.context_enabled
            or not _known_context(context_a)
            or not _known_context(context_b)
            or context_a == context_b
        ):
            return False
        graph = self._active_graph()
        queue: deque[str] = deque([context_a])
        visited = {context_a}
        while queue:
            current = queue.popleft()
            for neighbor in graph.get(current, set()):
                if neighbor == context_b:
                    return True
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def context_supported(self, context: str) -> bool:
        return bool(_known_context(context) and self._active_graph().get(context))

    def audit(self) -> dict[str, Any]:
        rows = []
        for (left, right), edge in sorted(self.edges.items()):
            rows.append(
                {
                    "context_a": left,
                    "context_b": right,
                    "series": edge.series,
                    "teams_a": sorted(edge.left_teams),
                    "teams_b": sorted(edge.right_teams),
                    "active": edge.active(self.config),
                }
            )
        return {
            "enabled": self.config.context_enabled,
            "min_bridge_series": self.config.min_bridge_series,
            "min_bridge_teams_per_context": (
                self.config.min_bridge_teams_per_context
            ),
            "edges": rows,
            "active_edges": sum(bool(row["active"]) for row in rows),
        }


@dataclass(frozen=True)
class SeriesPrediction:
    probability: float
    map_probability: float
    latent_logit: float
    predictive_variance: float
    predictive_sigma: float
    context_supported: bool
    team_a_mean: float
    team_b_mean: float
    scheduled_best_of: int


@dataclass(frozen=True)
class PrequentialSeriesRun:
    predictions: pd.DataFrame
    model: "SeriesDynamicBradleyTerry"
    audit: Mapping[str, Any]


@dataclass(frozen=True)
class SeriesTournamentResult:
    selected_config: SeriesDynamicBTConfig
    selected_elo_config: SeriesEloConfig
    validation_scores: pd.DataFrame
    elo_validation_scores: pd.DataFrame
    prediction_ledger: pd.DataFrame
    final_metrics: Mapping[str, Mapping[str, Any]]
    comparisons: Mapping[str, Mapping[str, Any]]
    gate: Mapping[str, Any]
    snapshot: pd.DataFrame
    metadata: Mapping[str, Any]


def _timestamp(value: Any, name: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        raise ValueError(f"{name} is not a valid timestamp")
    return pd.Timestamp(parsed)


def _known_context(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text and text.upper() != "UNKNOWN")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def config_sha256(config: Any) -> str:
    """Return the exact canonical SHA-256 for a dataclass configuration."""

    if not hasattr(config, "__dataclass_fields__"):
        raise TypeError("config_sha256 requires a dataclass instance")
    return hashlib.sha256(_canonical_json(asdict(config))).hexdigest()


def model_code_sha256() -> str:
    """Hash the complete rating and canonical-observation implementation.

    File names and byte lengths are framed explicitly so concatenation cannot
    create an ambiguous digest. Upstream release rows are pinned separately by
    ``observation_rows_sha256``.
    """

    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(
        (dependency.resolve() for dependency in _MODEL_CODE_DEPENDENCIES),
        key=lambda value: value.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def model_identity(config: Any, *, model_id: str = MODEL_ID) -> dict[str, str]:
    """Return full, non-truncated code/config hashes and a compact version."""

    code_hash = model_code_sha256()
    configuration_hash = config_sha256(config)
    return {
        "model_id": model_id,
        "model_code_sha256": code_hash,
        "model_config_sha256": configuration_hash,
        "model_version": (
            f"{model_id}:{code_hash[:12]}:{configuration_hash[:12]}"
        ),
    }


def _empty_observations() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "series_key",
            "prediction_time",
            "date",
            "team_a",
            "team_b",
            "team_a_name",
            "team_b_name",
            "home_a",
            "home_b",
            "y_a",
            "n_maps",
            "international",
            "league",
            "scheduled_best_of",
            "series_source",
            "source_series_id",
            "completion_source",
            "format_source",
            "series_provenance",
            "rating_eligible",
            "series_weight",
        ]
    )


def prepare_series_observations(
    maps: pd.DataFrame,
    *,
    data_cutoff: Any = None,
) -> pd.DataFrame:
    """Build the audited, order-neutral one-row-per-series input.

    The verified source applies ``data_cutoff`` at series completion, after
    canonical construction.  This is essential: truncating raw maps first
    could turn the prefix of a future-completed Bo3 into a false completed Bo1.
    """

    source = pd.DataFrame() if maps is None else maps.copy()
    cutoff = (
        _timestamp(data_cutoff, "data_cutoff")
        if data_cutoff is not None
        else None
    )
    source_rows = len(source)
    canonical = build_canonical_series_ledger(source)
    observations = _verified_series_observations(
        source,
        cutoff,
        _OBSERVATION_HALF_LIFE_PLACEHOLDER_DAYS,
    )
    source_attrs = dict(observations.attrs)
    if observations.empty:
        out = _empty_observations()
        out.attrs.update(source_attrs)
        out.attrs.update(
            {
                "source_map_rows": source_rows,
                "data_cutoff": cutoff.isoformat() if cutoff is not None else None,
                "cutoff_policy": (
                    "canonicalize full source, then admit only series whose "
                    "verified completion is at or before cutoff"
                ),
                "observation_source": (
                    "lol_kills.ratings.hierarchical_bt._observations"
                ),
            }
        )
        return out

    required = {
        "series_key",
        "prediction_time",
        "date",
        "team_a",
        "team_b",
        "team_a_name",
        "team_b_name",
        "home_a",
        "home_b",
        "y_a",
        "n_maps",
        "international",
        "league",
        "scheduled_best_of",
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise RuntimeError(
            f"verified series observations are missing columns: {missing}"
        )
    out = observations[list(sorted(required))].copy()
    out["series_key"] = out["series_key"].astype(str)
    provenance_columns = [
        "canonical_series_id",
        "source",
        "source_series_id",
        "completion_status",
        "completion_source",
        "series_format_source",
        "rating_eligible",
    ]
    missing_provenance = sorted(
        set(provenance_columns).difference(canonical.series.columns)
    )
    if missing_provenance:
        raise RuntimeError(
            "canonical series ledger lacks required provenance columns: "
            f"{missing_provenance}"
        )
    provenance = canonical.series[provenance_columns].rename(
        columns={
            "canonical_series_id": "series_key",
            "source": "series_source",
            "series_format_source": "format_source",
        }
    )
    provenance["series_key"] = provenance["series_key"].astype(str)
    out = out.merge(provenance, on="series_key", how="left", validate="one_to_one")
    out["format_source"] = out["format_source"].fillna(
        "explicit_source_series_format"
    )
    out["series_provenance"] = (
        out["series_source"].fillna("").astype(str)
        + ":"
        + out["source_series_id"]
        .fillna(out["series_key"])
        .astype(str)
    )
    out["prediction_time"] = pd.to_datetime(
        out["prediction_time"], errors="coerce", utc=True
    )
    out["date"] = pd.to_datetime(out["date"], errors="coerce", utc=True)
    out["series_key"] = out["series_key"].astype(str).str.strip()
    for column in ("team_a", "team_b"):
        out[column] = out[column].astype(str).str.strip()
    out["y_a"] = pd.to_numeric(out["y_a"], errors="coerce")
    invalid = (
        out["date"].isna()
        | out["prediction_time"].isna()
        | out["prediction_time"].gt(out["date"])
        | out["series_key"].eq("")
        | out["team_a"].eq("")
        | out["team_b"].eq("")
        | out["team_a"].eq(out["team_b"])
        | ~out["y_a"].isin([0.0, 1.0])
        | ~pd.to_numeric(
            out["scheduled_best_of"], errors="coerce"
        ).isin([1, 3, 5])
        | ~out["rating_eligible"].fillna(False).astype(bool)
        | ~out["completion_status"].eq("completed")
        | out["completion_source"].fillna("").astype(str).str.strip().eq("")
        | out["format_source"].fillna("").astype(str).str.strip().eq("")
        | out["series_source"].fillna("").astype(str).str.strip().eq("")
    )
    if invalid.any():
        examples = out.loc[
            invalid,
            [
                "series_key",
                "prediction_time",
                "date",
                "team_a",
                "team_b",
                "y_a",
                "scheduled_best_of",
                "completion_status",
                "completion_source",
                "format_source",
            ],
        ].head(5)
        raise RuntimeError(
            "verified series source returned invalid observations: "
            f"{examples.to_dict('records')}"
        )
    if out["series_key"].duplicated().any():
        duplicates = (
            out.loc[out["series_key"].duplicated(False), "series_key"]
            .head(5)
            .tolist()
        )
        raise RuntimeError(
            f"verified series keys must be unique; examples={duplicates}"
        )
    if (out["team_a"] > out["team_b"]).any():
        raise RuntimeError(
            "verified series orientation must use canonical immutable team keys"
        )
    if cutoff is not None and out["date"].gt(cutoff).any():
        raise RuntimeError("verified series source exceeded data_cutoff")

    out["team_a_name"] = out["team_a_name"].fillna(out["team_a"]).astype(str)
    out["team_b_name"] = out["team_b_name"].fillna(out["team_b"]).astype(str)
    out["home_a"] = out["home_a"].fillna("UNKNOWN").astype(str)
    out["home_b"] = out["home_b"].fillna("UNKNOWN").astype(str)
    out["y_a"] = out["y_a"].astype(float)
    out["n_maps"] = pd.to_numeric(out["n_maps"], errors="raise").astype(int)
    out["scheduled_best_of"] = pd.to_numeric(
        out["scheduled_best_of"], errors="raise"
    ).astype(int)
    if out["n_maps"].lt(1).any():
        raise RuntimeError("verified series must contain at least one map")
    out["series_weight"] = 1.0
    out = out.sort_values(
        ["prediction_time", "date", "series_key"], kind="mergesort"
    ).reset_index(drop=True)
    out.attrs.update(source_attrs)
    out.attrs.update(
        {
            "source_map_rows": source_rows,
            "data_cutoff": cutoff.isoformat() if cutoff is not None else None,
            "cutoff_policy": (
                "canonicalize full source, then admit only series whose "
                "verified completion is at or before cutoff"
            ),
            "observation_source": (
                "lol_kills.ratings.hierarchical_bt._observations"
            ),
            "observation_unit": "one equally weighted completed series",
            "estimand": (
                "pre-series win probability for the verified scheduled format, "
                "derived from latent per-map Bradley-Terry probability"
            ),
            "side_policy": (
                "canonical immutable team-key order; no map-side or order term"
            ),
            "forbidden_predictors": [
                "first_map_side",
                "map_side",
                "pick_order",
                "n_maps",
                "series_score",
                "post_series_covariates",
            ],
        }
    )
    return out


def _validate_prepared_observations(observations: pd.DataFrame) -> pd.DataFrame:
    required = {
        "series_key",
        "prediction_time",
        "date",
        "team_a",
        "team_b",
        "team_a_name",
        "team_b_name",
        "home_a",
        "home_b",
        "y_a",
        "scheduled_best_of",
        "series_provenance",
        "completion_source",
        "format_source",
        "rating_eligible",
    }
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"series observations missing columns: {missing}")
    frame = observations.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["prediction_time"] = pd.to_datetime(
        frame["prediction_time"], errors="coerce", utc=True
    )
    frame["y_a"] = pd.to_numeric(frame["y_a"], errors="coerce")
    invalid = (
        frame["date"].isna()
        | frame["prediction_time"].isna()
        | frame["prediction_time"].gt(frame["date"])
        | frame["series_key"].astype(str).str.strip().eq("")
        | frame["team_a"].astype(str).str.strip().eq("")
        | frame["team_b"].astype(str).str.strip().eq("")
        | frame["team_a"].eq(frame["team_b"])
        | ~frame["y_a"].isin([0.0, 1.0])
        | ~pd.to_numeric(
            frame["scheduled_best_of"], errors="coerce"
        ).isin([1, 3, 5])
        | ~frame["rating_eligible"].fillna(False).astype(bool)
        | frame["series_provenance"].astype(str).str.strip().eq("")
        | frame["completion_source"].astype(str).str.strip().eq("")
        | frame["format_source"].astype(str).str.strip().eq("")
    )
    if invalid.any():
        raise ValueError("series observations contain invalid rows")
    if frame["series_key"].astype(str).duplicated().any():
        raise ValueError("series_key must be unique")
    frame["series_key"] = frame["series_key"].astype(str)
    frame["team_a"] = frame["team_a"].astype(str)
    frame["team_b"] = frame["team_b"].astype(str)
    frame["home_a"] = frame["home_a"].fillna("UNKNOWN").astype(str)
    frame["home_b"] = frame["home_b"].fillna("UNKNOWN").astype(str)
    frame["team_a_name"] = frame["team_a_name"].fillna(frame["team_a"]).astype(
        str
    )
    frame["team_b_name"] = frame["team_b_name"].fillna(frame["team_b"]).astype(
        str
    )
    frame["y_a"] = frame["y_a"].astype(float)
    frame["scheduled_best_of"] = pd.to_numeric(
        frame["scheduled_best_of"], errors="raise"
    ).astype(int)
    frame["series_weight"] = 1.0
    return frame.sort_values(
        ["prediction_time", "date", "series_key"], kind="mergesort"
    ).reset_index(drop=True)


def _bounded_probability(
    latent_logit: float, predictive_variance: float, floor: float
) -> float:
    """Logistic-normal approximation with exact order complementation."""

    scale = math.sqrt(
        1.0 + math.pi * max(float(predictive_variance), 0.0) / 8.0
    )
    magnitude = abs(float(latent_logit)) / scale
    unit = 1.0 if magnitude >= 40.0 else 1.0 / (1.0 + math.exp(-magnitude))
    upper = floor + (1.0 - 2.0 * floor) * unit
    if latent_logit > 0.0:
        return upper
    if latent_logit < 0.0:
        return 1.0 - upper
    return 0.5


def _logistic(value: float) -> float:
    if value >= 0.0:
        tail = math.exp(-min(value, 40.0))
        return 1.0 / (1.0 + tail)
    tail = math.exp(max(value, -40.0))
    return tail / (1.0 + tail)


def series_win_probability(map_probability: float, scheduled_best_of: int) -> float:
    """Convert constant per-map win probability to a verified series forecast."""

    probability = float(map_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("map_probability must be within [0, 1]")
    if scheduled_best_of == 1:
        return probability
    if scheduled_best_of == 3:
        return probability**2 * (3.0 - 2.0 * probability)
    if scheduled_best_of == 5:
        return probability**3 * (
            10.0 - 15.0 * probability + 6.0 * probability**2
        )
    raise ValueError("scheduled_best_of must be one of 1, 3, or 5")


def _series_likelihood_terms(
    latent_logit: float,
    outcome: float,
    scheduled_best_of: int,
) -> tuple[float, float]:
    """Return score and expected curvature with respect to latent logit."""

    map_probability = _logistic(latent_logit)
    series_probability = series_win_probability(
        map_probability, scheduled_best_of
    )
    if scheduled_best_of == 1:
        derivative_probability = map_probability * (1.0 - map_probability)
    elif scheduled_best_of == 3:
        derivative_probability = (
            6.0 * map_probability**2 * (1.0 - map_probability) ** 2
        )
    elif scheduled_best_of == 5:
        derivative_probability = (
            30.0 * map_probability**3 * (1.0 - map_probability) ** 3
        )
    else:
        raise ValueError("scheduled_best_of must be one of 1, 3, or 5")
    denominator = max(
        series_probability * (1.0 - series_probability), 1e-12
    )
    score = (
        (float(outcome) - series_probability)
        * derivative_probability
        / denominator
    )
    information = derivative_probability**2 / denominator
    return score, max(information, 1e-12)


class SeriesDynamicBradleyTerry:
    """Mutable series-level diagonal-Gaussian Bradley--Terry filter."""

    def __init__(self, config: SeriesDynamicBTConfig | None = None) -> None:
        self.config = config or SeriesDynamicBTConfig()
        self.teams: dict[str, GaussianState] = {}
        self.contexts: dict[str, GaussianState] = {}
        self.bridges = HistoricalBridgeTracker(self.config)
        self.team_metadata: dict[str, dict[str, Any]] = {}
        self.comparison_edges: set[tuple[str, str]] = set()
        self.observed_series = 0

    def _propagate(
        self,
        state: GaussianState,
        timestamp: pd.Timestamp,
        *,
        variance_per_day: float,
        maximum_variance: float,
    ) -> None:
        if state.last_timestamp is None:
            state.last_timestamp = timestamp
            return
        if timestamp < state.last_timestamp:
            raise ValueError("rating state cannot move backwards in time")
        elapsed_days = (
            timestamp - state.last_timestamp
        ).total_seconds() / 86400.0
        if elapsed_days <= 0.0:
            return
        if self.config.mean_reversion_half_life_days is not None:
            state.mean *= math.exp(
                -math.log(2.0)
                * elapsed_days
                / self.config.mean_reversion_half_life_days
            )
        state.variance = min(
            maximum_variance,
            max(
                self.config.min_variance,
                state.variance + variance_per_day * elapsed_days,
            ),
        )
        state.last_timestamp = timestamp

    def _team(self, key: str, timestamp: pd.Timestamp) -> GaussianState:
        clean = str(key).strip()
        if not clean:
            raise ValueError("team key must be non-empty")
        state = self.teams.get(clean)
        if state is None:
            state = GaussianState(0.0, self.config.team_prior_sigma**2)
            self.teams[clean] = state
        self._propagate(
            state,
            timestamp,
            variance_per_day=self.config.team_variance_per_day,
            maximum_variance=self.config.max_team_variance,
        )
        return state

    def _context(self, key: str, timestamp: pd.Timestamp) -> GaussianState:
        state = self.contexts.get(key)
        if state is None:
            state = GaussianState(0.0, self.config.context_prior_sigma**2)
            self.contexts[key] = state
        self._propagate(
            state,
            timestamp,
            variance_per_day=self.config.context_variance_per_day,
            maximum_variance=self.config.max_context_variance,
        )
        return state

    def _features(
        self,
        team_a: str,
        team_b: str,
        timestamp: pd.Timestamp,
        home_a: str,
        home_b: str,
    ) -> tuple[list[tuple[GaussianState, float]], bool]:
        if not team_a or not team_b or team_a == team_b:
            raise ValueError("prediction requires two different immutable keys")
        features: list[tuple[GaussianState, float]] = [
            (self._team(team_a, timestamp), 1.0),
            (self._team(team_b, timestamp), -1.0),
        ]
        context_supported = self.bridges.supported(home_a, home_b)
        if context_supported:
            features.extend(
                [
                    (self._context(home_a, timestamp), 1.0),
                    (self._context(home_b, timestamp), -1.0),
                ]
            )
        return features, context_supported

    def predict(
        self,
        team_a: str,
        team_b: str,
        *,
        timestamp: Any,
        home_a: str = "UNKNOWN",
        home_b: str = "UNKNOWN",
        scheduled_best_of: int = 1,
    ) -> SeriesPrediction:
        """Predict the first immutable key winning, without a side term."""

        moment = _timestamp(timestamp, "timestamp")
        first = str(team_a).strip()
        second = str(team_b).strip()
        context_a = str(home_a or "UNKNOWN").strip()
        context_b = str(home_b or "UNKNOWN").strip()
        features, context_supported = self._features(
            first, second, moment, context_a, context_b
        )
        latent = math.fsum(state.mean * coefficient for state, coefficient in features)
        variance = math.fsum(
            state.variance * coefficient * coefficient
            for state, coefficient in features
        )
        map_probability = _bounded_probability(
            latent, variance, self.config.probability_floor
        )
        series_probability = series_win_probability(
            map_probability, scheduled_best_of
        )
        series_probability = min(
            1.0 - self.config.probability_floor,
            max(self.config.probability_floor, series_probability),
        )
        return SeriesPrediction(
            probability=series_probability,
            map_probability=map_probability,
            latent_logit=latent,
            predictive_variance=variance,
            predictive_sigma=math.sqrt(variance),
            context_supported=context_supported,
            team_a_mean=self.teams[first].mean,
            team_b_mean=self.teams[second].mean,
            scheduled_best_of=scheduled_best_of,
        )

    def observe_batch(
        self,
        batch: pd.DataFrame,
        *,
        prediction_lookup: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> None:
        """Assimilate one already-predicted exact-timestamp batch.

        With ``prediction_lookup``, likelihood terms are frozen at pre-series
        prediction time and applied only now, at verified completion. Applying
        the completed-series sums once makes the update invariant to row order.
        """

        frame = _validate_prepared_observations(batch)
        if frame.empty:
            return
        if frame["date"].nunique() != 1:
            raise ValueError("observe_batch requires one exact timestamp")
        moment = pd.Timestamp(frame["date"].iloc[0])
        gradients: dict[int, float] = defaultdict(float)
        curvatures: dict[int, float] = defaultdict(float)
        exposures: dict[int, int] = defaultdict(int)
        states: dict[int, GaussianState] = {}

        for _, row in frame.iterrows():
            features, context_supported_now = self._features(
                row["team_a"],
                row["team_b"],
                moment,
                row["home_a"],
                row["home_b"],
            )
            frozen = (
                prediction_lookup.get(str(row["series_key"]))
                if prediction_lookup is not None
                else None
            )
            if prediction_lookup is not None and frozen is None:
                raise ValueError(
                    "completed series lacks its pre-series prediction record"
                )
            if frozen is not None:
                context_supported = bool(frozen["context_supported"])
                # Context support is frozen at prediction. A bridge completed
                # during the series cannot enter its own likelihood.
                if not context_supported and context_supported_now:
                    features = features[:2]
                elif context_supported and not context_supported_now:
                    raise RuntimeError(
                        "historical bridge support cannot disappear"
                    )
                latent = float(frozen["latent_logit"])
            else:
                latent = math.fsum(
                    state.mean * coefficient for state, coefficient in features
                )
            score_term, curvature = _series_likelihood_terms(
                latent,
                float(row["y_a"]),
                int(row["scheduled_best_of"]),
            )
            for state, coefficient in features:
                identifier = id(state)
                states[identifier] = state
                gradients[identifier] += coefficient * score_term
                curvatures[identifier] += curvature * coefficient * coefficient
                exposures[identifier] += 1

        context_state_ids = {id(value) for value in self.contexts.values()}
        for identifier, state in states.items():
            posterior_variance = 1.0 / (
                1.0 / state.variance + curvatures[identifier]
            )
            state.mean = min(
                self.config.max_abs_mean,
                max(
                    -self.config.max_abs_mean,
                    state.mean + posterior_variance * gradients[identifier],
                ),
            )
            maximum = (
                self.config.max_context_variance
                if identifier in context_state_ids
                else self.config.max_team_variance
            )
            state.variance = min(
                maximum,
                max(self.config.min_variance, posterior_variance),
            )
            state.last_observed = moment
            state.observations += exposures[identifier]

        # Bridge and display metadata become historical only after the complete
        # timestamp batch has been assimilated.
        for _, row in frame.iterrows():
            self.comparison_edges.add(
                tuple(sorted((str(row["team_a"]), str(row["team_b"]))))
            )
            self.bridges.register(
                row["home_a"],
                row["home_b"],
                row["team_a"],
                row["team_b"],
            )
            self._record_metadata(
                row["team_a"],
                row["team_a_name"],
                row["home_a"],
                moment,
                row["series_key"],
            )
            self._record_metadata(
                row["team_b"],
                row["team_b_name"],
                row["home_b"],
                moment,
                row["series_key"],
            )
        self.observed_series += len(frame)

    def _record_metadata(
        self,
        team_key: str,
        display: str,
        home: str,
        timestamp: pd.Timestamp,
        series_key: str,
    ) -> None:
        previous = self.team_metadata.get(team_key)
        order = (timestamp, str(series_key))
        if previous is not None and order < previous["_order"]:
            return
        resolved_home = str(home or "UNKNOWN")
        if (
            not _known_context(resolved_home)
            and previous is not None
            and _known_context(previous["home_league"])
        ):
            resolved_home = previous["home_league"]
        self.team_metadata[team_key] = {
            "display": str(display or team_key),
            "home_league": resolved_home,
            "last_series_at": timestamp,
            "_order": order,
        }

    def snapshot(self, *, as_of: Any = None) -> pd.DataFrame:
        """Return current mean, sigma, and one-sided fifth-percentile rating."""

        if not self.teams:
            return pd.DataFrame(
                columns=[
                    "team_key",
                    "team",
                    "home_league",
                    "mean",
                    "sigma",
                    "rating_p05",
                    "sigma_kind",
                    "comparison_component_id",
                    "comparison_component_size",
                    "cross_component_rankable",
                ]
            )
        if as_of is None:
            moments = [
                state.last_timestamp
                for state in self.teams.values()
                if state.last_timestamp is not None
            ]
            moment = max(moments)
        else:
            moment = _timestamp(as_of, "as_of")

        parent = {key: key for key in self.teams}

        def find(key: str) -> str:
            root = key
            while parent[root] != root:
                root = parent[root]
            while parent[key] != key:
                next_key = parent[key]
                parent[key] = root
                key = next_key
            return root

        for left, right in sorted(self.comparison_edges):
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)
        components: dict[str, list[str]] = defaultdict(list)
        for key in sorted(self.teams):
            components[find(key)].append(key)
        component_metadata: dict[str, tuple[str, int]] = {}
        for members in components.values():
            digest = hashlib.sha256(
                "\n".join(sorted(members)).encode("utf-8")
            ).hexdigest()[:16]
            for key in members:
                component_metadata[key] = (f"component-{digest}", len(members))

        rows: list[dict[str, Any]] = []
        for key in sorted(self.teams):
            team_state = self._team(key, moment)
            metadata = self.team_metadata.get(
                key,
                {
                    "display": key,
                    "home_league": "UNKNOWN",
                    "last_series_at": None,
                },
            )
            home = str(metadata["home_league"])
            context_mean = 0.0
            context_variance = 0.0
            context_included = (
                self.config.context_enabled
                and self.bridges.context_supported(home)
                and home in self.contexts
            )
            if context_included:
                context_state = self._context(home, moment)
                context_mean = context_state.mean
                context_variance = context_state.variance
            latent_mean = team_state.mean + context_mean
            latent_sigma = math.sqrt(team_state.variance + context_variance)
            mean = self.config.base_rating + LOGIT_TO_RATING * latent_mean
            sigma = LOGIT_TO_RATING * latent_sigma
            component_id, component_size = component_metadata[key]
            rows.append(
                {
                    "team_key": key,
                    "team": metadata["display"],
                    "home_league": home,
                    "mean": mean,
                    "sigma": sigma,
                    "rating_p05": mean - RATING_P05_Z * sigma,
                    "sigma_kind": "diagonal_filter_approximation_sd",
                    "rating_p05_interpretation": (
                        "normal-approximation lower quantile; empirical coverage "
                        "has not been established"
                    ),
                    "comparison_component_id": component_id,
                    "comparison_component_size": component_size,
                    "cross_component_rankable": False,
                    "team_mean_logit": team_state.mean,
                    "team_sigma_logit": math.sqrt(team_state.variance),
                    "context_mean_logit": context_mean,
                    "context_included": context_included,
                    "series_observed": team_state.observations,
                    "last_series_at": metadata["last_series_at"],
                    "as_of": moment,
                }
            )
        return pd.DataFrame(rows).sort_values(
            ["rating_p05", "mean", "team_key"],
            ascending=[False, False, True],
            kind="mergesort",
        ).reset_index(drop=True)


def _prediction_record(
    row: pd.Series, prediction: SeriesPrediction
) -> dict[str, Any]:
    return {
        "series_key": row["series_key"],
        "prediction_time": row["prediction_time"],
        "completion_time": row["date"],
        "timestamp": row["prediction_time"],
        "team_a": row["team_a"],
        "team_b": row["team_b"],
        "home_a": row["home_a"],
        "home_b": row["home_b"],
        "y_true": float(row["y_a"]),
        "probability": prediction.probability,
        "map_probability": prediction.map_probability,
        "latent_logit": prediction.latent_logit,
        "predictive_variance": prediction.predictive_variance,
        "predictive_sigma": prediction.predictive_sigma,
        "context_supported": prediction.context_supported,
        "scheduled_best_of": int(row["scheduled_best_of"]),
        "n_maps": int(row["n_maps"]),
        "international": bool(row["international"]),
        "series_provenance": row["series_provenance"],
        "completion_source": row["completion_source"],
        "format_source": row["format_source"],
        "series_weight": 1.0,
        "prediction_before_outcome": True,
        "outcome_assimilated_at_verified_completion": True,
    }


def run_prequential_series(
    observations: pd.DataFrame,
    *,
    config: SeriesDynamicBTConfig | None = None,
) -> PrequentialSeriesRun:
    """Run chronological series predictions with exact-timestamp batching."""

    frame = _validate_prepared_observations(observations)
    model = SeriesDynamicBradleyTerry(config)
    records: list[dict[str, Any]] = []
    prediction_lookup: dict[str, dict[str, Any]] = {}
    event_times = sorted(
        set(frame["prediction_time"].tolist()) | set(frame["date"].tolist())
    )
    for event_time in event_times:
        starts = frame.loc[frame["prediction_time"].eq(event_time)].sort_values(
            "series_key", kind="mergesort"
        )
        completions = frame.loc[frame["date"].eq(event_time)].sort_values(
            "series_key", kind="mergesort"
        )
        # At a shared timestamp, every pre-series forecast is emitted before
        # any series completing at that timestamp enters the state.
        for _, row in starts.iterrows():
            prediction = model.predict(
                row["team_a"],
                row["team_b"],
                timestamp=row["prediction_time"],
                home_a=row["home_a"],
                home_b=row["home_b"],
                scheduled_best_of=int(row["scheduled_best_of"]),
            )
            record = _prediction_record(row, prediction)
            records.append(record)
            prediction_lookup[str(row["series_key"])] = record
        if not completions.empty:
            model.observe_batch(
                completions, prediction_lookup=prediction_lookup
            )
    predictions = pd.DataFrame(records).sort_values(
        ["prediction_time", "series_key"], kind="mergesort"
    ).reset_index(drop=True)
    return PrequentialSeriesRun(
        predictions=predictions,
        model=model,
        audit={
            "series": len(frame),
            "observation_unit": "one equally weighted completed series",
            "prediction_protocol": (
                "predict at verified series start; assimilate only at verified "
                "completion; exact-timestamp predictions precede completions"
            ),
            "side_term": "absent",
            "partial_series_policy": "excluded by verified rating eligibility",
            "format_probability": {
                "Bo1": "q1(p)=p",
                "Bo3": "q3(p)=p^2(3-2p)",
                "Bo5": "q5(p)=p^3(10-15p+6p^2)",
            },
            "bridge": model.bridges.audit(),
            "identity": model_identity(model.config),
        },
    )


def _elo_probability(
    rating_a: float, rating_b: float, config: SeriesEloConfig
) -> float:
    latent = math.log(10.0) * (rating_a - rating_b) / config.scale
    return _bounded_probability(latent, 0.0, config.probability_floor)


def _run_rolling_elo(
    observations: pd.DataFrame, config: SeriesEloConfig
) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = _validate_prepared_observations(observations)
    ratings: dict[str, float] = defaultdict(lambda: config.base_rating)
    records: list[dict[str, Any]] = []
    prediction_lookup: dict[str, float] = {}
    event_times = sorted(
        set(frame["prediction_time"].tolist()) | set(frame["date"].tolist())
    )
    for event_time in event_times:
        starts = frame.loc[frame["prediction_time"].eq(event_time)].sort_values(
            "series_key", kind="mergesort"
        )
        completions = frame.loc[frame["date"].eq(event_time)].sort_values(
            "series_key", kind="mergesort"
        )
        for _, row in starts.iterrows():
            map_probability = _elo_probability(
                ratings[row["team_a"]], ratings[row["team_b"]], config
            )
            probability = series_win_probability(
                map_probability, int(row["scheduled_best_of"])
            )
            probability = min(
                1.0 - config.probability_floor,
                max(config.probability_floor, probability),
            )
            prediction_lookup[str(row["series_key"])] = probability
            records.append(
                {
                    "series_key": row["series_key"],
                    "prediction_time": row["prediction_time"],
                    "completion_time": row["date"],
                    "timestamp": row["prediction_time"],
                    "y_true": float(row["y_a"]),
                    "probability": probability,
                    "map_probability": map_probability,
                    "scheduled_best_of": int(row["scheduled_best_of"]),
                    "series_weight": 1.0,
                    "prediction_before_outcome": True,
                }
            )
        deltas: dict[str, float] = defaultdict(float)
        for _, row in completions.iterrows():
            probability = prediction_lookup.get(str(row["series_key"]))
            if probability is None:
                raise RuntimeError(
                    "rolling Elo completion lacks pre-series prediction"
                )
            change = config.k_factor * (float(row["y_a"]) - probability)
            deltas[row["team_a"]] += change
            deltas[row["team_b"]] -= change
        for key, delta in deltas.items():
            ratings[key] += delta
    return (
        pd.DataFrame(records).sort_values(
            ["prediction_time", "series_key"], kind="mergesort"
        ).reset_index(drop=True),
        dict(ratings),
    )


def _run_historical_base_rate(observations: pd.DataFrame) -> pd.DataFrame:
    """Return the order-neutral empirical historical base rate.

    Every completed head-to-head series contributes one winner and one loser
    when both team perspectives are represented.  The historical base rate for
    an otherwise unidentified first team is therefore exactly 1/2.  This
    construction avoids smuggling canonical key order in as a predictor.
    """

    frame = _validate_prepared_observations(observations)
    return pd.DataFrame(
        {
            "series_key": frame["series_key"],
            "prediction_time": frame["prediction_time"],
            "completion_time": frame["date"],
            "timestamp": frame["prediction_time"],
            "y_true": frame["y_a"].astype(float),
            "probability": np.full(len(frame), 0.5, dtype=float),
            "scheduled_best_of": frame["scheduled_best_of"].astype(int),
            "series_weight": np.ones(len(frame), dtype=float),
            "prediction_before_outcome": np.ones(len(frame), dtype=bool),
        }
    )


def proper_scores(
    outcome: Sequence[float] | pd.Series | np.ndarray,
    probability: Sequence[float] | pd.Series | np.ndarray,
    *,
    ece_bins: int = 10,
    scheduled_best_of: (
        Sequence[int] | pd.Series | np.ndarray | None
    ) = None,
) -> dict[str, Any]:
    """Compute series-weighted log loss, Brier score, and equal-width ECE."""

    if ece_bins < 2:
        raise ValueError("ece_bins must be at least two")
    y = np.asarray(outcome, dtype=float)
    p = np.asarray(probability, dtype=float)
    if (
        y.ndim != 1
        or p.ndim != 1
        or len(y) != len(p)
        or len(y) == 0
    ):
        raise ValueError("outcome and probability must be aligned non-empty 1D")
    if not np.isfinite(y).all() or not np.isin(y, [0.0, 1.0]).all():
        raise ValueError("outcome must be finite and binary")
    if not np.isfinite(p).all() or np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("probability must be finite and strictly within (0, 1)")
    log_loss_vector = -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    brier_vector = np.square(p - y)
    bin_index = np.minimum((p * ece_bins).astype(int), ece_bins - 1)
    calibration: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(ece_bins):
        selected = bin_index == index
        count = int(selected.sum())
        if count == 0:
            continue
        mean_probability = float(p[selected].mean())
        observed_rate = float(y[selected].mean())
        gap = abs(mean_probability - observed_rate)
        ece += count / len(y) * gap
        calibration.append(
            {
                "bin": index,
                "lower": index / ece_bins,
                "upper": (index + 1) / ece_bins,
                "n": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
                "absolute_gap": gap,
            }
        )
    result: dict[str, Any] = {
        "n": int(len(y)),
        "log_loss": float(log_loss_vector.mean()),
        "brier": float(brier_vector.mean()),
        "ece": float(ece),
        "ece_bins": ece_bins,
        "calibration_bins": calibration,
        "series_weighting": "one row, one unit per completed series",
    }
    if scheduled_best_of is not None:
        formats = np.asarray(scheduled_best_of, dtype=int)
        if formats.shape != y.shape or not np.isin(formats, [1, 3, 5]).all():
            raise ValueError(
                "scheduled_best_of must align and contain only 1, 3, or 5"
            )
        strata: dict[str, Any] = {}
        for best_of in (1, 3, 5):
            selected = formats == best_of
            if not selected.any():
                continue
            stratum = proper_scores(
                y[selected],
                p[selected],
                ece_bins=ece_bins,
            )
            strata[f"Bo{best_of}"] = {
                "n": stratum["n"],
                "log_loss": stratum["log_loss"],
                "brier": stratum["brier"],
                "ece": stratum["ece"],
                "ece_bins": stratum["ece_bins"],
                "calibration_bins": stratum["calibration_bins"],
            }
        result["format_stratified_calibration"] = strata
    return result


def _score_vector(y: np.ndarray, p: np.ndarray, score: str) -> np.ndarray:
    if score == "log_loss":
        return -(y * np.log(p) + (1.0 - y) * np.log1p(-p))
    if score == "brier":
        return np.square(p - y)
    raise ValueError(f"unsupported proper score: {score}")


def paired_circular_moving_block_bootstrap(
    frame: pd.DataFrame,
    *,
    candidate_column: str,
    baseline_column: str,
    score: str = "log_loss",
    replicates: int = 2000,
    block_size: int = 8,
    alpha: float = 0.05,
    noninferiority_margin: float = 0.0,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Compare paired predictions over chronologically ordered series."""

    required = {
        "series_key",
        "timestamp",
        "y_true",
        candidate_column,
        baseline_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"bootstrap frame missing columns: {missing}")
    if frame.empty or frame["series_key"].duplicated().any():
        raise ValueError("bootstrap requires one non-empty row per series")
    if replicates < 100:
        raise ValueError("replicates must be at least 100")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be strictly between zero and one")
    _finite_nonnegative(
        "noninferiority_margin", float(noninferiority_margin)
    )
    ordered = frame.assign(
        _time=pd.to_datetime(frame["timestamp"], errors="raise", utc=True)
    ).sort_values(["_time", "series_key"], kind="mergesort")
    y = ordered["y_true"].to_numpy(float)
    candidate = ordered[candidate_column].to_numpy(float)
    baseline = ordered[baseline_column].to_numpy(float)
    for values, name in ((candidate, candidate_column), (baseline, baseline_column)):
        if not np.isfinite(values).all() or np.any(
            (values <= 0.0) | (values >= 1.0)
        ):
            raise ValueError(f"{name} must be finite and strictly bounded")
    delta = _score_vector(y, candidate, score) - _score_vector(y, baseline, score)
    count = len(delta)
    width = min(block_size, count)
    blocks_needed = math.ceil(count / width)
    offsets = np.arange(width)
    rng = np.random.default_rng(random_seed)
    boot = np.empty(replicates, dtype=float)
    for index in range(replicates):
        starts = rng.integers(0, count, size=blocks_needed)
        sampled = ((starts[:, None] + offsets[None, :]) % count).ravel()[:count]
        boot[index] = float(delta[sampled].mean())
    low, high = np.quantile(boot, [alpha / 2.0, 1.0 - alpha / 2.0])
    if high < 0.0:
        decision = "superior"
    elif high <= noninferiority_margin:
        decision = "noninferior"
    elif low > noninferiority_margin:
        decision = "inferior"
    else:
        decision = "inconclusive"
    return {
        "score": score,
        "series": count,
        "candidate_score": float(_score_vector(y, candidate, score).mean()),
        "baseline_score": float(_score_vector(y, baseline, score).mean()),
        "candidate_minus_baseline": float(delta.mean()),
        "confidence_level": 1.0 - alpha,
        "confidence_interval": [float(low), float(high)],
        "noninferiority_margin": float(noninferiority_margin),
        "decision": decision,
        "bootstrap": {
            "method": (
                "paired circular moving-block bootstrap by ordered completed series"
            ),
            "replicates": replicates,
            "block_size_series": width,
            "seed": random_seed,
        },
    }


def evaluate_promotion_gate(
    comparisons: Mapping[str, Mapping[str, Any]],
    candidate_metrics: Mapping[str, Any],
    *,
    required_baselines: Iterable[str] = (ELO_BASELINE_ID, BASE_RATE_ID),
) -> dict[str, Any]:
    """Apply the terminal log-loss and calibration-reporting gate."""

    accepted = {"superior", "noninferior"}
    required = tuple(required_baselines)
    missing = [name for name in required if name not in comparisons]
    decisions = {
        name: (
            str(comparisons[name].get("decision"))
            if name in comparisons
            else "missing"
        )
        for name in required
    }
    calibration_reported = (
        int(candidate_metrics.get("n", 0)) > 0
        and isinstance(candidate_metrics.get("calibration_bins"), list)
        and len(candidate_metrics.get("calibration_bins", [])) > 0
        and math.isfinite(float(candidate_metrics.get("ece", math.nan)))
    )
    score_contract_ok = all(
        comparisons[name].get("score") == "log_loss"
        for name in required
        if name in comparisons
    )
    passed = (
        not missing
        and calibration_reported
        and score_contract_ok
        and all(decision in accepted for decision in decisions.values())
    )
    reasons: list[str] = []
    if missing:
        reasons.append(f"missing baseline comparisons: {missing}")
    if not score_contract_ok:
        reasons.append("primary comparison must be log_loss")
    if not calibration_reported:
        reasons.append("candidate ECE/calibration bins were not reported")
    failed_decisions = {
        name: decision
        for name, decision in decisions.items()
        if decision not in accepted
    }
    if failed_decisions:
        reasons.append(f"log-loss gate not established: {failed_decisions}")
    return {
        "passed": passed,
        "status": "passed" if passed else "failed",
        "required_decisions": sorted(accepted),
        "decisions": decisions,
        "calibration_reported": calibration_reported,
        "ece": candidate_metrics.get("ece"),
        "reasons": reasons,
        "claim_boundary": (
            "a passing gate supports this frozen tournament and snapshot only; "
            "it is not a state-of-the-art claim"
        ),
    }


def default_dynamic_candidates() -> tuple[SeriesDynamicBTConfig, ...]:
    """Small predeclared smooth-dynamics grid for validation selection."""

    return (
        SeriesDynamicBTConfig(team_variance_per_day=0.0005),
        SeriesDynamicBTConfig(team_variance_per_day=0.002),
        SeriesDynamicBTConfig(team_variance_per_day=0.008),
    )


def default_elo_candidates() -> tuple[SeriesEloConfig, ...]:
    return tuple(SeriesEloConfig(k_factor=value) for value in (12.0, 24.0, 36.0))


def _assign_split(
    prediction_time: pd.Series,
    completion_time: pd.Series,
    validation_start: pd.Timestamp,
    test_start: pd.Timestamp,
) -> pd.Series:
    return pd.Series(
        np.where(
            prediction_time.lt(validation_start),
            "train",
            np.where(
                prediction_time.ge(test_start),
                "test",
                np.where(completion_time.lt(test_start), "validation", "embargo"),
            ),
        ),
        index=prediction_time.index,
        dtype="object",
    )


def _observations_sha256(frame: pd.DataFrame) -> str:
    columns = [
        "series_key",
        "prediction_time",
        "date",
        "team_a",
        "team_b",
        "home_a",
        "home_b",
        "y_a",
        "scheduled_best_of",
        "series_provenance",
    ]
    rows = []
    for _, row in frame[columns].iterrows():
        rows.append(
            {
                "series_key": str(row["series_key"]),
                "prediction_time": pd.Timestamp(
                    row["prediction_time"]
                ).isoformat(),
                "date": pd.Timestamp(row["date"]).isoformat(),
                "team_a": str(row["team_a"]),
                "team_b": str(row["team_b"]),
                "home_a": str(row["home_a"]),
                "home_b": str(row["home_b"]),
                "y_a": float(row["y_a"]),
                "scheduled_best_of": int(row["scheduled_best_of"]),
                "series_provenance": str(row["series_provenance"]),
            }
        )
    return hashlib.sha256(_canonical_json(rows)).hexdigest()


def _validation_score_rows(
    observations: pd.DataFrame,
    candidates: Sequence[SeriesDynamicBTConfig],
    *,
    validation_start: pd.Timestamp,
    ece_bins: int,
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        run = run_prequential_series(observations, config=config)
        selected = run.predictions["timestamp"].ge(validation_start)
        metrics = proper_scores(
            run.predictions.loc[selected, "y_true"],
            run.predictions.loc[selected, "probability"],
            ece_bins=ece_bins,
            scheduled_best_of=run.predictions.loc[
                selected, "scheduled_best_of"
            ],
        )
        rows.append(
            {
                **model_identity(config),
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "n": metrics["n"],
                "config": config,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["log_loss", "brier", "model_config_sha256"], kind="mergesort"
    ).reset_index(drop=True)


def _elo_validation_score_rows(
    observations: pd.DataFrame,
    candidates: Sequence[SeriesEloConfig],
    *,
    validation_start: pd.Timestamp,
    ece_bins: int,
) -> pd.DataFrame:
    rows = []
    for config in candidates:
        predictions, _ = _run_rolling_elo(observations, config)
        selected = predictions["timestamp"].ge(validation_start)
        metrics = proper_scores(
            predictions.loc[selected, "y_true"],
            predictions.loc[selected, "probability"],
            ece_bins=ece_bins,
            scheduled_best_of=predictions.loc[
                selected, "scheduled_best_of"
            ],
        )
        rows.append(
            {
                **model_identity(config, model_id=ELO_BASELINE_ID),
                "log_loss": metrics["log_loss"],
                "brier": metrics["brier"],
                "ece": metrics["ece"],
                "n": metrics["n"],
                "config": config,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["log_loss", "brier", "model_config_sha256"], kind="mergesort"
    ).reset_index(drop=True)


def run_series_rating_tournament(
    maps: pd.DataFrame,
    *,
    spec: SeriesTournamentSpec,
    dynamic_candidates: Sequence[SeriesDynamicBTConfig] | None = None,
    elo_candidates: Sequence[SeriesEloConfig] | None = None,
) -> SeriesTournamentResult:
    """Select on validation, score one frozen test, and build a current snapshot."""

    spec.validate()
    validation_start, test_start, cutoff = spec.timestamps()
    candidates = tuple(dynamic_candidates or default_dynamic_candidates())
    elo_grid = tuple(elo_candidates or default_elo_candidates())
    if not candidates:
        raise ValueError("dynamic_candidates cannot be empty")
    if not elo_grid:
        raise ValueError("elo_candidates cannot be empty")
    observations = prepare_series_observations(maps, data_cutoff=cutoff)
    if observations.empty:
        raise ValueError("no rating-eligible completed series")
    observation_audit = dict(observations.attrs)
    observations = _validate_prepared_observations(observations)
    observations["split"] = _assign_split(
        observations["prediction_time"],
        observations["date"],
        validation_start,
        test_start,
    )
    split_counts = observations["split"].value_counts().to_dict()
    if split_counts.get("validation", 0) < 1:
        raise ValueError("validation split contains no completed series")
    if split_counts.get("test", 0) < spec.minimum_test_series:
        raise ValueError(
            "test split contains fewer completed series than minimum_test_series"
        )

    # Selection sees only labels whose verified completion precedes the frozen
    # test boundary. A series crossing that boundary is embargoed.
    development = observations.loc[observations["date"].lt(test_start)].copy()
    validation_scores = _validation_score_rows(
        development,
        candidates,
        validation_start=validation_start,
        ece_bins=spec.ece_bins,
    )
    selected_config = validation_scores.iloc[0]["config"]
    elo_validation_scores = _elo_validation_score_rows(
        development,
        elo_grid,
        validation_start=validation_start,
        ece_bins=spec.ece_bins,
    )
    selected_elo_config = elo_validation_scores.iloc[0]["config"]

    dynamic_run = run_prequential_series(observations, config=selected_config)
    elo_predictions, _ = _run_rolling_elo(observations, selected_elo_config)
    base_predictions = _run_historical_base_rate(observations)
    ledger = dynamic_run.predictions[
        [
            "series_key",
            "prediction_time",
            "completion_time",
            "timestamp",
            "y_true",
            "probability",
            "map_probability",
            "scheduled_best_of",
            "team_a",
            "team_b",
            "n_maps",
            "international",
            "series_provenance",
            "completion_source",
            "format_source",
            "series_weight",
        ]
    ].rename(columns={"probability": MODEL_ID})
    ledger = ledger.merge(
        elo_predictions[["series_key", "probability"]].rename(
            columns={"probability": ELO_BASELINE_ID}
        ),
        on="series_key",
        validate="one_to_one",
    ).merge(
        base_predictions[["series_key", "probability"]].rename(
            columns={"probability": BASE_RATE_ID}
        ),
        on="series_key",
        validate="one_to_one",
    )
    split_lookup = observations.set_index("series_key")["split"]
    ledger["split"] = ledger["series_key"].map(split_lookup)
    test = ledger.loc[ledger["split"].eq("test")].copy()
    final_metrics = {
        model_id: proper_scores(
            test["y_true"],
            test[model_id],
            ece_bins=spec.ece_bins,
            scheduled_best_of=test["scheduled_best_of"],
        )
        for model_id in (MODEL_ID, ELO_BASELINE_ID, BASE_RATE_ID)
    }
    comparisons = {
        baseline: paired_circular_moving_block_bootstrap(
            test,
            candidate_column=MODEL_ID,
            baseline_column=baseline,
            score=spec.primary_score,
            replicates=spec.bootstrap_replicates,
            block_size=spec.moving_block_size,
            alpha=spec.alpha,
            noninferiority_margin=spec.noninferiority_margin,
            random_seed=spec.random_seed,
        )
        for baseline in (ELO_BASELINE_ID, BASE_RATE_ID)
    }
    gate = evaluate_promotion_gate(comparisons, final_metrics[MODEL_ID])
    snapshot_as_of = cutoff if cutoff is not None else observations["date"].max()
    snapshot = dynamic_run.model.snapshot(as_of=snapshot_as_of)
    identity = model_identity(selected_config)
    elo_identity = model_identity(
        selected_elo_config, model_id=ELO_BASELINE_ID
    )
    base_rate_config = {
        "probability": 0.5,
        "orientation": "both_team_perspectives",
        "observation_unit": "completed_series",
    }
    base_rate_config_hash = hashlib.sha256(
        _canonical_json(base_rate_config)
    ).hexdigest()
    code_hash = model_code_sha256()
    metadata = {
        "model": identity,
        "elo_baseline": elo_identity,
        "base_rate": {
            "model_id": BASE_RATE_ID,
            "probability": 0.5,
            "model_code_sha256": code_hash,
            "model_config_sha256": base_rate_config_hash,
            "model_version": (
                f"{BASE_RATE_ID}:{code_hash[:12]}:"
                f"{base_rate_config_hash[:12]}"
            ),
            "definition": (
                "historical completed-series base rate after including both "
                "team perspectives; independent of canonical key order"
            ),
        },
        "tournament_spec": asdict(spec),
        "tournament_spec_sha256": config_sha256(spec),
        "observation_rows_sha256": _observations_sha256(observations),
        "observation_audit": observation_audit,
        "split": {key: int(value) for key, value in split_counts.items()},
        "selection": {
            "criterion": "lowest validation log_loss; brier and config hash break ties",
            "test_labels_used_for_selection": False,
            "hyperparameters_frozen_before_test": True,
            "candidate_config_sha256": identity["model_config_sha256"],
            "elo_config_sha256": elo_identity["model_config_sha256"],
        },
        "final_test": {
            "evaluated_once_after_selection": True,
            "prequential_updates_allowed_after_each_observed_series": True,
            "prediction_time": "verified series start",
            "outcome_assimilation_time": "verified series completion",
            "cross_boundary_series": "embargoed from validation selection",
            "series_weighting": "one completed series equals one unit",
            "format_probability": {
                "Bo1": "q1(p)=p",
                "Bo3": "q3(p)=p^2(3-2p)",
                "Bo5": "q5(p)=p^3(10-15p+6p^2)",
            },
        },
        "snapshot": {
            "as_of": pd.Timestamp(snapshot_as_of).isoformat(),
            "uncertainty": {
                "sigma": "diagonal-filter approximation SD in rating points",
                "rating_p05": (
                    "normal-approximation lower quantile: mean - "
                    "1.6448536269514722 * sigma"
                ),
                "coverage_claim": False,
                "display_inflation": "none",
            },
            "display_policy": (
                "latest historically observed display and home league"
            ),
            "gate_passed": gate["passed"],
        },
        "excluded_model_family": (
            "static hierarchy is not a tournament baseline or candidate"
        ),
        "claim_boundary": (
            "report tournament scores, calibration, uncertainty, and gate status; "
            "do not claim state of the art"
        ),
    }
    return SeriesTournamentResult(
        selected_config=selected_config,
        selected_elo_config=selected_elo_config,
        validation_scores=validation_scores.drop(columns=["config"]),
        elo_validation_scores=elo_validation_scores.drop(columns=["config"]),
        prediction_ledger=ledger,
        final_metrics=final_metrics,
        comparisons=comparisons,
        gate=gate,
        snapshot=snapshot,
        metadata=metadata,
    )


def build_current_snapshot(
    maps: pd.DataFrame,
    *,
    config: SeriesDynamicBTConfig | None = None,
    data_cutoff: Any = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one declared configuration and return its leakage-safe snapshot."""

    selected = config or SeriesDynamicBTConfig()
    observations = prepare_series_observations(maps, data_cutoff=data_cutoff)
    if observations.empty:
        return SeriesDynamicBradleyTerry(selected).snapshot(), {
            **model_identity(selected),
            "n_series": 0,
            "observation_audit": dict(observations.attrs),
        }
    run = run_prequential_series(observations, config=selected)
    as_of = (
        _timestamp(data_cutoff, "data_cutoff")
        if data_cutoff is not None
        else observations["date"].max()
    )
    return run.model.snapshot(as_of=as_of), {
        **model_identity(selected),
        "n_series": len(observations),
        "as_of": pd.Timestamp(as_of).isoformat(),
        "observation_rows_sha256": _observations_sha256(observations),
        "observation_audit": dict(observations.attrs),
        "bridge": run.model.bridges.audit(),
        "series_weighting": "one completed series equals one unit",
        "side_term": "absent",
    }


# Descriptive alias for callers that prefer the model family in the function name.
run_series_dynamic_bt_tournament = run_series_rating_tournament


__all__ = [
    "BASE_RATE_ID",
    "ELO_BASELINE_ID",
    "MODEL_ID",
    "GaussianState",
    "HistoricalBridgeTracker",
    "PrequentialSeriesRun",
    "SeriesDynamicBTConfig",
    "SeriesDynamicBradleyTerry",
    "SeriesEloConfig",
    "SeriesPrediction",
    "SeriesTournamentResult",
    "SeriesTournamentSpec",
    "build_current_snapshot",
    "config_sha256",
    "default_dynamic_candidates",
    "default_elo_candidates",
    "evaluate_promotion_gate",
    "model_code_sha256",
    "model_identity",
    "paired_circular_moving_block_bootstrap",
    "prepare_series_observations",
    "proper_scores",
    "run_prequential_series",
    "run_series_dynamic_bt_tournament",
    "run_series_rating_tournament",
    "series_win_probability",
]
