"""Research-only, role-specific player performance rating candidate.

This module estimates a *descriptive* player-map performance signal.  It does
not estimate causal player skill, match-win probability, or value toward a
team win.  In particular, ``result``, kills, deaths, assists, and all
team-outcome fields are deliberately absent from the target and design.

The estimand is a transparent latent early-resource performance score:

    mean(
        robust_z(gold differential at 15),
        robust_z(experience differential at 15),
        robust_z(creep-score differential at 15),
    )

Each component is standardized using training-only, symmetric player-map
distributions within role / league / patch, with predeclared hierarchical
fallbacks.  The source statistics are recorded after a map but are measured at
15 minutes and do not use the eventual match result.  They remain descriptive:
draft, lane assignments, coaching, team coordination, and measurement quality
can all affect them.

The candidate borrows defensible principles from SIDO (hierarchical
champion/player/team context and uncertainty) and PandaSkill (separate
role-specific performance models and shrinkage), while intentionally rejecting
the use of a team-win classifier as an individual-performance label.

Only exact provider player IDs are modeled.  Player names are display metadata,
never identity keys.  Fits use sparse penalized Gaussian regressions with
signed player, opponent, champion, and team effects.  Hyperparameters are
selected on a chronological validation split, then frozen for a single
chronological test evaluation.

This file is not integrated with production and must not be promoted without
external review, population audits, and an explicit product decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.sparse.linalg import LinearOperator, cg, lsmr
from scipy.stats import spearmanr


CANONICAL_ROLES: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")
TARGET_METRICS: tuple[str, ...] = (
    "golddiffat15",
    "xpdiffat15",
    "csdiffat15",
)
REQUIRED_COLUMNS: tuple[str, ...] = (
    "gameid",
    "date",
    "patch",
    "league",
    "source",
    "datacompleteness",
    "side",
    "position",
    "playerid",
    "playername",
    "team_key",
    "champion",
    *TARGET_METRICS,
)
ESTIMAND = (
    "Descriptive role-relative 15-minute resource performance: the equal-weight "
    "mean of training-only robust standardized gold, experience, and creep-score "
    "differentials at 15 minutes."
)
NON_ESTIMANDS: tuple[str, ...] = (
    "causal player skill",
    "match-win probability",
    "win contribution",
    "roster-independent player value",
    "complete-game performance",
)
PROMOTION_STATUS = "research_candidate_not_production"
RESEARCH_ANCHORS: tuple[str, ...] = (
    "SIDO: arXiv:2403.04873",
    "PandaSkill: arXiv:2501.10049",
)


class PlayerPerformanceDataError(ValueError):
    """Raised when the OE-shaped input cannot support the declared estimand."""


@dataclass(frozen=True)
class PlayerPerformanceConfig:
    """Predeclared data, model, split, and uncertainty choices."""

    target_metrics: tuple[str, ...] = TARGET_METRICS
    canonical_roles: tuple[str, ...] = CANONICAL_ROLES
    min_context_player_maps: int = 40
    min_stable_identity_matchup_coverage: float = 0.90
    robust_z_clip: float = 5.0
    train_fraction: float = 0.60
    validation_fraction: float = 0.20
    ridge_grid: tuple[float, ...] = (2.0, 8.0, 32.0, 128.0)
    champion_penalty_multiplier: float = 2.0
    team_penalty_multiplier: float = 4.0
    context_penalty_multiplier: float = 4.0
    intercept_penalty: float = 1e-6
    min_train_matchups_per_role: int = 40
    lsmr_tolerance: float = 1e-8
    lsmr_maxiter: int = 2_000
    exact_covariance_max_features: int = 256
    uncertainty_probes: int = 12
    uncertainty_cg_tolerance: float = 1e-7
    uncertainty_cg_maxiter: int = 1_500
    conservative_z: float = 1.645
    random_seed: int = 20260727
    minimum_test_relative_rmse_lift: float = 0.0
    minimum_player_incremental_rmse_lift: float = 0.0
    metric_bootstrap_replicates: int = 5_000
    metric_ci_level: float = 0.95
    require_positive_player_lift_ci: bool = True

    def __post_init__(self) -> None:
        if self.target_metrics != TARGET_METRICS:
            raise ValueError(
                "the research estimand is predeclared to the three 15-minute "
                "resource differentials"
            )
        if self.canonical_roles != CANONICAL_ROLES:
            raise ValueError("canonical roles are fixed for this candidate")
        if self.min_context_player_maps < 4:
            raise ValueError("min_context_player_maps must be at least four")
        if not 0.0 < self.min_stable_identity_matchup_coverage <= 1.0:
            raise ValueError(
                "min_stable_identity_matchup_coverage must be in (0, 1]"
            )
        if not math.isfinite(self.robust_z_clip) or self.robust_z_clip <= 0.0:
            raise ValueError("robust_z_clip must be finite and positive")
        if not 0.0 < self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be in (0, 1)")
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train and validation fractions must leave test data")
        if not self.ridge_grid or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.ridge_grid
        ):
            raise ValueError("ridge_grid must contain finite positive values")
        for name, value in (
            ("champion_penalty_multiplier", self.champion_penalty_multiplier),
            ("team_penalty_multiplier", self.team_penalty_multiplier),
            ("context_penalty_multiplier", self.context_penalty_multiplier),
            ("intercept_penalty", self.intercept_penalty),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.min_train_matchups_per_role < 4:
            raise ValueError("min_train_matchups_per_role must be at least four")
        if not 0.0 < self.lsmr_tolerance < 1.0:
            raise ValueError("lsmr_tolerance must be in (0, 1)")
        if self.lsmr_maxiter < 1:
            raise ValueError("lsmr_maxiter must be positive")
        if self.exact_covariance_max_features < 2:
            raise ValueError("exact_covariance_max_features must be at least two")
        if self.uncertainty_probes < 2:
            raise ValueError("uncertainty_probes must be at least two")
        if not 0.0 < self.uncertainty_cg_tolerance < 1.0:
            raise ValueError("uncertainty_cg_tolerance must be in (0, 1)")
        if self.uncertainty_cg_maxiter < 1:
            raise ValueError("uncertainty_cg_maxiter must be positive")
        if not math.isfinite(self.conservative_z) or self.conservative_z < 0.0:
            raise ValueError("conservative_z must be finite and non-negative")
        if not math.isfinite(self.minimum_player_incremental_rmse_lift):
            raise ValueError(
                "minimum_player_incremental_rmse_lift must be finite"
            )
        if self.metric_bootstrap_replicates < 100:
            raise ValueError("metric_bootstrap_replicates must be at least 100")
        if not 0.0 < self.metric_ci_level < 1.0:
            raise ValueError("metric_ci_level must be in (0, 1)")


@dataclass(frozen=True)
class PlayerMapAudit:
    """Fail-closed provenance, identity, completeness, and target audit."""

    input_rows: int
    input_games: int
    source_counts: tuple[tuple[str, int], ...]
    completeness_counts: tuple[tuple[str, int], ...]
    missing_by_column: tuple[tuple[str, int], ...]
    non_oe_rows: int
    partial_rows_excluded: int
    complete_rows: int
    complete_games: int
    invalid_side_rows: int
    invalid_role_rows: int
    duplicate_game_side_role_cells: int
    malformed_complete_games: int
    duplicate_stable_player_games: int
    missing_stable_player_rows: int
    missing_team_key_rows: int
    complete_target_missing_rows: int
    nonfinite_target_rows: int
    non_antisymmetric_matchups: int
    numeric_patch_rows: int
    stable_identity_matchups: int
    complete_matchups: int
    eligible_matchups: int
    stable_identity_matchup_coverage: float
    player_ids_with_multiple_names: int
    names_with_multiple_player_ids: int
    hard_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.hard_failures and self.eligible_matchups > 0

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.__dict__)
        result["source_counts"] = dict(self.source_counts)
        result["completeness_counts"] = dict(self.completeness_counts)
        result["missing_by_column"] = dict(self.missing_by_column)
        result["ready"] = self.ready
        return result


@dataclass(frozen=True)
class PreparedPlayerMaps:
    """One Blue-perspective row per exact same-role matchup."""

    matchups: pd.DataFrame
    audit: PlayerMapAudit


@dataclass(frozen=True)
class RobustScale:
    median: float
    scale: float
    n: int


@dataclass(frozen=True)
class RobustContextStandardizer:
    """Training-only symmetric robust scales with hierarchical fallbacks."""

    tables: Mapping[str, Mapping[str, Mapping[tuple[str, ...], RobustScale]]]
    min_context_player_maps: int
    clip: float
    fitted_matchups: int
    fitted_through: pd.Timestamp

    @classmethod
    def fit(
        cls,
        matchups: pd.DataFrame,
        config: PlayerPerformanceConfig,
    ) -> "RobustContextStandardizer":
        if matchups.empty:
            raise PlayerPerformanceDataError(
                "cannot fit robust standardizer on empty matchups"
            )
        tables: dict[
            str, dict[str, dict[tuple[str, ...], RobustScale]]
        ] = {}
        levels: tuple[tuple[str, tuple[str, ...]], ...] = (
            ("role_league_patch", ("role", "league", "patch_context")),
            ("role_league", ("role", "league")),
            ("role", ("role",)),
            ("global", ()),
        )
        for metric in config.target_metrics:
            metric_tables: dict[str, dict[tuple[str, ...], RobustScale]] = {}
            values = pd.to_numeric(matchups[metric], errors="coerce")
            if values.isna().any() or not np.isfinite(values.to_numpy()).all():
                raise PlayerPerformanceDataError(
                    f"{metric} contains missing or non-finite fit values"
                )
            for level_name, columns in levels:
                level_table: dict[tuple[str, ...], RobustScale] = {}
                if columns:
                    grouper: str | list[str]
                    grouper = columns[0] if len(columns) == 1 else list(columns)
                    grouped: Iterable[tuple[Any, pd.DataFrame]] = matchups.assign(
                        _metric=values
                    ).groupby(grouper, sort=True, dropna=False)
                else:
                    grouped = [((), matchups.assign(_metric=values))]
                for key, group in grouped:
                    if not isinstance(key, tuple):
                        key = (key,)
                    canonical_key = tuple(str(value) for value in key)
                    blue_values = pd.to_numeric(
                        group["_metric"], errors="coerce"
                    ).to_numpy(dtype=float)
                    symmetric = np.concatenate((blue_values, -blue_values))
                    level_table[canonical_key] = RobustScale(
                        median=0.0,
                        scale=_robust_scale(symmetric),
                        n=int(symmetric.size),
                    )
                metric_tables[level_name] = level_table
            tables[metric] = metric_tables
        return cls(
            tables=tables,
            min_context_player_maps=config.min_context_player_maps,
            clip=config.robust_z_clip,
            fitted_matchups=len(matchups),
            fitted_through=pd.Timestamp(matchups["date"].max()),
        )

    def transform(self, matchups: pd.DataFrame) -> pd.DataFrame:
        transformed = matchups.copy()
        z_columns: list[str] = []
        scale_levels: list[list[str]] = []
        for metric, metric_tables in self.tables.items():
            z_values: list[float] = []
            metric_levels: list[str] = []
            for row in transformed.itertuples(index=False):
                keys = (
                    (
                        "role_league_patch",
                        (
                            str(row.role),
                            str(row.league),
                            str(row.patch_context),
                        ),
                    ),
                    (
                        "role_league",
                        (str(row.role), str(row.league)),
                    ),
                    ("role", (str(row.role),)),
                    ("global", ()),
                )
                selected: RobustScale | None = None
                selected_level = ""
                for level_name, key in keys:
                    candidate = metric_tables[level_name].get(key)
                    if candidate is None:
                        continue
                    if (
                        candidate.n >= self.min_context_player_maps
                        or level_name == "global"
                    ):
                        selected = candidate
                        selected_level = level_name
                        break
                if selected is None:
                    raise PlayerPerformanceDataError(
                        f"no training scale is available for {metric}"
                    )
                raw = float(getattr(row, metric))
                z_values.append(
                    float(
                        np.clip(
                            (raw - selected.median) / selected.scale,
                            -self.clip,
                            self.clip,
                        )
                    )
                )
                metric_levels.append(selected_level)
            z_column = f"_z_{metric}"
            transformed[z_column] = z_values
            z_columns.append(z_column)
            scale_levels.append(metric_levels)
        transformed["observed_performance"] = transformed[z_columns].mean(
            axis=1
        )
        transformed["target_component_sd"] = transformed[z_columns].std(
            axis=1, ddof=0
        )
        level_rank = {
            "role_league_patch": 0,
            "role_league": 1,
            "role": 2,
            "global": 3,
        }
        combined_levels: list[str] = []
        for index in range(len(transformed)):
            levels = [metric_levels[index] for metric_levels in scale_levels]
            combined_levels.append(max(levels, key=level_rank.__getitem__))
        transformed["scale_fallback"] = combined_levels
        return transformed


@dataclass(frozen=True)
class PenaltySpec:
    player_l2: float
    champion_l2: float
    team_l2: float
    context_l2: float
    intercept_l2: float

    @classmethod
    def from_base(
        cls,
        base: float,
        config: PlayerPerformanceConfig,
    ) -> "PenaltySpec":
        return cls(
            player_l2=float(base),
            champion_l2=float(
                base * config.champion_penalty_multiplier
            ),
            team_l2=float(base * config.team_penalty_multiplier),
            context_l2=float(base * config.context_penalty_multiplier),
            intercept_l2=float(config.intercept_penalty),
        )


@dataclass(frozen=True)
class RolePerformanceModel:
    role: str
    feature_names: tuple[str, ...]
    feature_blocks: tuple[str, ...]
    coefficients: np.ndarray
    coefficient_variance: np.ndarray
    penalty_diagonal: np.ndarray
    reference_league: str
    reference_patch: str
    residual_scale: float
    n_fit_matchups: int
    lsmr_iterations: int
    lsmr_condition_estimate: float
    uncertainty_method: str

    def feature_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.feature_names)}

    def predict(self, matchups: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        design, metadata = _transform_design(matchups, self)
        return np.asarray(design @ self.coefficients).reshape(-1), metadata


@dataclass(frozen=True)
class PlayerPerformanceCandidate:
    """Sparse role-specific research fit at one frozen penalty."""

    role_models: Mapping[str, RolePerformanceModel]
    standardizer: RobustContextStandardizer
    penalty: PenaltySpec
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    fit_matchups: int
    includes_player_effects: bool

    def predict(self, matchups: pd.DataFrame) -> pd.DataFrame:
        transformed = self.standardizer.transform(matchups)
        pieces: list[pd.DataFrame] = []
        for role in CANONICAL_ROLES:
            role_rows = transformed[transformed["role"].eq(role)].copy()
            if role_rows.empty:
                continue
            model = self.role_models.get(role)
            if model is None:
                raise PlayerPerformanceDataError(
                    f"no fitted role model is available for {role}"
                )
            predicted, metadata = model.predict(role_rows)
            role_rows["predicted_performance"] = predicted
            for column in metadata.columns:
                role_rows[column] = metadata[column].to_numpy()
            pieces.append(role_rows)
        if not pieces:
            return transformed.assign(
                predicted_performance=pd.Series(dtype=float)
            )
        return (
            pd.concat(pieces, ignore_index=True)
            .sort_values(
                ["date", "game_id", "_role_order"],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )

    def player_ratings(
        self,
        fit_matchups: pd.DataFrame,
        config: PlayerPerformanceConfig,
    ) -> pd.DataFrame:
        appearances = _expand_matchup_metadata(fit_matchups)
        rows: list[dict[str, Any]] = []
        for role, model in self.role_models.items():
            role_appearances = appearances[
                appearances["role"].eq(role)
            ].copy()
            counts = role_appearances.groupby("player_id", sort=True).size()
            latest = (
                role_appearances.sort_values(
                    ["date", "game_id"], kind="mergesort"
                )
                .groupby("player_id", sort=True)
                .tail(1)
                .set_index("player_id")
            )
            index = model.feature_index()
            for player_id, n_maps in counts.items():
                feature = f"player:{player_id}"
                if feature not in index:
                    continue
                column = index[feature]
                mean = float(model.coefficients[column])
                variance = float(model.coefficient_variance[column])
                sd = math.sqrt(max(variance, 0.0))
                metadata = latest.loc[player_id]
                rows.append(
                    {
                        "player_id": str(player_id),
                        "player_name": str(metadata["player_name"]),
                        "role": role,
                        "last_team_key": str(metadata["team_key"]),
                        "last_date": metadata["date"],
                        "maps": int(n_maps),
                        "performance_mean": mean,
                        "performance_sd": sd,
                        "conservative_performance": (
                            mean - config.conservative_z * sd
                        ),
                        "estimand": ESTIMAND,
                        "uncertainty_method": model.uncertainty_method,
                        "promotion_status": PROMOTION_STATUS,
                    }
                )
        return (
            pd.DataFrame(rows)
            .sort_values(
                [
                    "role",
                    "conservative_performance",
                    "performance_mean",
                    "player_id",
                ],
                ascending=[True, False, False, True],
                kind="mergesort",
            )
            .reset_index(drop=True)
        )


@dataclass(frozen=True)
class PerformanceMetrics:
    rows: int
    rmse: float
    mae: float
    r2: float
    spearman: float
    zero_baseline_rmse: float
    relative_rmse_lift: float

    @classmethod
    def unavailable(cls) -> "PerformanceMetrics":
        return cls(
            rows=0,
            rmse=float("nan"),
            mae=float("nan"),
            r2=float("nan"),
            spearman=float("nan"),
            zero_baseline_rmse=float("nan"),
            relative_rmse_lift=float("nan"),
        )


@dataclass(frozen=True)
class PairedRMSEContrast:
    """Calendar-day block-bootstrap contrast against a paired baseline."""

    rows: int
    calendar_day_blocks: int
    candidate_rmse: float
    baseline_rmse: float
    relative_rmse_lift: float
    ci_low: float
    ci_high: float
    confidence_level: float
    bootstrap_replicates: int
    resampling_unit: str = "calendar_day"


@dataclass(frozen=True)
class PlayerPerformanceTournament:
    audit: PlayerMapAudit
    selected_base_penalty: float
    selected_context_base_penalty: float
    validation_candidates: pd.DataFrame
    split_boundaries: Mapping[str, str]
    prediction_ledger: pd.DataFrame
    train_metrics: PerformanceMetrics
    validation_metrics: PerformanceMetrics
    test_metrics: PerformanceMetrics
    validation_context_baseline_metrics: PerformanceMetrics
    test_context_baseline_metrics: PerformanceMetrics
    future_patch_test_metrics: PerformanceMetrics
    roster_move_test_metrics: PerformanceMetrics
    player_ratings: pd.DataFrame
    player_incremental_test_rmse_lift: float
    player_incremental_test_contrast: PairedRMSEContrast
    test_gate_passed: bool
    estimand: str = ESTIMAND
    non_estimands: tuple[str, ...] = NON_ESTIMANDS
    promotion_status: str = PROMOTION_STATUS
    research_anchors: tuple[str, ...] = RESEARCH_ANCHORS
    limitations: tuple[str, ...] = (
        "The target covers only 15-minute resource performance.",
        "The target is observational and can reflect draft, lane, and team context.",
        "Provider numeric patch values can lose trailing-zero precision.",
        "Gaussian ridge uncertainty is an approximation, not a full Bayesian posterior.",
        "Player/team/champion effects remain prior-sensitive when exposure is collinear.",
        "GRID-only rows are excluded because the target fields are OE-derived.",
    )


def _clean_text(series: pd.Series) -> pd.Series:
    cleaned = series.astype("string").fillna("").str.strip()
    return cleaned.mask(cleaned.str.casefold().isin({"nan", "none", "<na>"}), "")


def _count_values(series: pd.Series) -> tuple[tuple[str, int], ...]:
    counts = series.astype("string").fillna("<missing>").value_counts(dropna=False)
    return tuple(sorted((str(key), int(value)) for key, value in counts.items()))


def _patch_context(value: Any) -> str:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        return f"string:{text}"
    if value is None or pd.isna(value):
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return f"other:{str(value).strip()}"
    if not math.isfinite(numeric):
        return ""
    text = f"{numeric:.8f}".rstrip("0").rstrip(".")
    return f"numeric:{text}"


def _robust_scale(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise PlayerPerformanceDataError("cannot robust-scale an empty metric")
    median = float(np.median(finite))
    mad_scale = 1.4826 * float(np.median(np.abs(finite - median)))
    if math.isfinite(mad_scale) and mad_scale > 1e-9:
        return mad_scale
    q25, q75 = np.quantile(finite, [0.25, 0.75])
    iqr_scale = float(q75 - q25) / 1.349
    if math.isfinite(iqr_scale) and iqr_scale > 1e-9:
        return iqr_scale
    std = float(np.std(finite))
    if math.isfinite(std) and std > 1e-9:
        return std
    return 1.0


def _diagnose_and_prepare(
    player_rows: pd.DataFrame,
    config: PlayerPerformanceConfig,
) -> PreparedPlayerMaps:
    if not isinstance(player_rows, pd.DataFrame):
        raise TypeError("player_rows must be a pandas DataFrame")
    frame = player_rows.copy()
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in frame.columns
    ]
    if missing_columns:
        audit = PlayerMapAudit(
            input_rows=len(frame),
            input_games=0,
            source_counts=(),
            completeness_counts=(),
            missing_by_column=tuple((column, len(frame)) for column in missing_columns),
            non_oe_rows=0,
            partial_rows_excluded=0,
            complete_rows=0,
            complete_games=0,
            invalid_side_rows=0,
            invalid_role_rows=0,
            duplicate_game_side_role_cells=0,
            malformed_complete_games=0,
            duplicate_stable_player_games=0,
            missing_stable_player_rows=0,
            missing_team_key_rows=0,
            complete_target_missing_rows=0,
            nonfinite_target_rows=0,
            non_antisymmetric_matchups=0,
            numeric_patch_rows=0,
            stable_identity_matchups=0,
            complete_matchups=0,
            eligible_matchups=0,
            stable_identity_matchup_coverage=0.0,
            player_ids_with_multiple_names=0,
            names_with_multiple_player_ids=0,
            hard_failures=(
                "missing required columns: " + ", ".join(missing_columns),
            ),
        )
        return PreparedPlayerMaps(pd.DataFrame(), audit)

    text_columns = (
        "gameid",
        "league",
        "source",
        "datacompleteness",
        "side",
        "position",
        "playerid",
        "playername",
        "team_key",
        "champion",
    )
    for column in text_columns:
        frame[column] = _clean_text(frame[column])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["_patch_context"] = frame["patch"].map(_patch_context)
    for metric in config.target_metrics:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")

    source_counts = _count_values(frame["source"])
    completeness_counts = _count_values(frame["datacompleteness"])
    missing_by_column = tuple(
        (
            column,
            int(
                frame[column].isna().sum()
                if not pd.api.types.is_string_dtype(frame[column])
                else frame[column].eq("").sum()
            ),
        )
        for column in REQUIRED_COLUMNS
    )
    non_oe = int((~frame["source"].str.casefold().eq("oe")).sum())
    invalid_sides = int((~frame["side"].isin(("Blue", "Red"))).sum())
    invalid_roles = int(
        (~frame["position"].isin(config.canonical_roles)).sum()
    )
    complete_mask = frame["datacompleteness"].str.casefold().eq("complete")
    partial_rows = int((~complete_mask).sum())
    complete = frame[complete_mask].copy()
    target_missing = complete[list(config.target_metrics)].isna().any(axis=1)
    target_matrix = complete[list(config.target_metrics)].to_numpy(dtype=float)
    target_nonfinite = (
        ~np.isfinite(target_matrix).all(axis=1)
        & ~target_missing.to_numpy()
    )

    cell_counts = (
        complete.groupby(["gameid", "side", "position"], dropna=False)
        .size()
    )
    duplicate_cells = int((cell_counts > 1).sum())
    expected_cells = {
        (side, role)
        for side in ("Blue", "Red")
        for role in config.canonical_roles
    }
    malformed_games = 0
    duplicate_player_games = 0
    for _, game in complete.groupby("gameid", sort=False):
        cells = set(zip(game["side"], game["position"]))
        if len(game) != 10 or cells != expected_cells:
            malformed_games += 1
        stable_ids = game.loc[game["playerid"].ne(""), "playerid"]
        if stable_ids.duplicated().any():
            duplicate_player_games += 1

    identity_pairs = complete.loc[
        complete["playerid"].ne(""), ["playerid", "playername"]
    ].drop_duplicates()
    ids_multiple_names = int(
        (
            identity_pairs.groupby("playerid", sort=True)["playername"].nunique()
            > 1
        ).sum()
    )
    names_multiple_ids = int(
        (
            identity_pairs.assign(
                _name=identity_pairs["playername"].str.casefold()
            )
            .groupby("_name", sort=True)["playerid"]
            .nunique()
            > 1
        ).sum()
    )

    role_order = {role: index for index, role in enumerate(CANONICAL_ROLES)}
    blue = complete[complete["side"].eq("Blue")].copy()
    red = complete[complete["side"].eq("Red")].copy()
    merge_columns = ["gameid", "position"]
    carry_columns = [
        "date",
        "league",
        "_patch_context",
        "playerid",
        "playername",
        "team_key",
        "champion",
        *config.target_metrics,
    ]
    matchups = blue[merge_columns + carry_columns].merge(
        red[merge_columns + carry_columns],
        on=merge_columns,
        how="inner",
        suffixes=("_blue", "_red"),
        validate="one_to_one" if duplicate_cells == 0 else "many_to_many",
    )
    context_mismatch = pd.Series(False, index=matchups.index)
    for column in ("date", "league", "_patch_context"):
        context_mismatch |= matchups[f"{column}_blue"].ne(
            matchups[f"{column}_red"]
        )
    non_antisymmetric = pd.Series(False, index=matchups.index)
    for metric in config.target_metrics:
        left = pd.to_numeric(matchups[f"{metric}_blue"], errors="coerce")
        right = pd.to_numeric(matchups[f"{metric}_red"], errors="coerce")
        valid = left.notna() & right.notna()
        non_antisymmetric |= valid & ((left + right).abs() > 1e-8)

    complete_matchups = len(matchups)
    stable_identity = (
        matchups["playerid_blue"].ne("")
        & matchups["playerid_red"].ne("")
        & matchups["playerid_blue"].ne(matchups["playerid_red"])
    )
    stable_identity_matchups = int(stable_identity.sum())
    target_available = pd.Series(True, index=matchups.index)
    for metric in config.target_metrics:
        left = pd.to_numeric(matchups[f"{metric}_blue"], errors="coerce")
        right = pd.to_numeric(matchups[f"{metric}_red"], errors="coerce")
        target_available &= (
            left.notna()
            & right.notna()
            & np.isfinite(left)
            & np.isfinite(right)
        )
    team_available = matchups["team_key_blue"].ne("") & matchups[
        "team_key_red"
    ].ne("")
    champion_available = matchups["champion_blue"].ne("") & matchups[
        "champion_red"
    ].ne("")
    context_available = (
        matchups["date_blue"].notna()
        & matchups["league_blue"].ne("")
        & matchups["_patch_context_blue"].ne("")
        & ~context_mismatch
    )
    eligible = (
        stable_identity
        & target_available
        & team_available
        & champion_available
        & context_available
        & ~non_antisymmetric
    )
    eligible_matchups = int(eligible.sum())
    coverage = (
        stable_identity_matchups / complete_matchups
        if complete_matchups
        else 0.0
    )

    hard_failures: list[str] = []
    warnings: list[str] = []
    if non_oe:
        hard_failures.append(
            f"{non_oe} non-OE rows violate the declared target provenance"
        )
    if invalid_sides:
        hard_failures.append(f"{invalid_sides} rows use non-canonical sides")
    if invalid_roles:
        hard_failures.append(f"{invalid_roles} rows use non-canonical roles")
    if duplicate_cells:
        hard_failures.append(
            f"{duplicate_cells} duplicate game/side/role cells"
        )
    if malformed_games:
        hard_failures.append(
            f"{malformed_games} complete games are not exact canonical 5v5s"
        )
    if duplicate_player_games:
        hard_failures.append(
            f"{duplicate_player_games} games repeat a stable player identity"
        )
    if int(target_missing.sum()):
        hard_failures.append(
            f"{int(target_missing.sum())} complete rows lack a target component"
        )
    if int(target_nonfinite.sum()):
        hard_failures.append(
            f"{int(target_nonfinite.sum())} complete rows have non-finite targets"
        )
    if int(context_mismatch.sum()):
        hard_failures.append(
            f"{int(context_mismatch.sum())} role matchups disagree on context"
        )
    if int(non_antisymmetric.sum()):
        hard_failures.append(
            f"{int(non_antisymmetric.sum())} role matchups violate exact "
            "opponent-differential antisymmetry"
        )
    if coverage < config.min_stable_identity_matchup_coverage:
        hard_failures.append(
            "stable player identity coverage "
            f"{coverage:.3%} is below the predeclared "
            f"{config.min_stable_identity_matchup_coverage:.3%} gate"
        )
    if partial_rows:
        warnings.append(
            f"{partial_rows} non-complete rows are excluded from the estimand"
        )
    missing_player_rows = int(complete["playerid"].eq("").sum())
    if missing_player_rows:
        warnings.append(
            f"{missing_player_rows} complete rows without stable player IDs "
            "are excluded; player-name fallback is forbidden"
        )
    missing_team_rows = int(complete["team_key"].eq("").sum())
    if missing_team_rows:
        warnings.append(
            f"{missing_team_rows} complete rows without stable team keys are excluded"
        )
    numeric_patch_rows = int(
        complete["patch"].map(
            lambda value: not isinstance(value, str) and pd.notna(value)
        ).sum()
    )
    if numeric_patch_rows:
        warnings.append(
            f"{numeric_patch_rows} complete rows have numeric patch provenance; "
            "trailing-zero patch precision cannot be reconstructed"
        )
    if names_multiple_ids:
        warnings.append(
            f"{names_multiple_ids} normalized display names map to multiple "
            "stable player IDs; IDs remain separate"
        )
    if ids_multiple_names:
        warnings.append(
            f"{ids_multiple_names} stable player IDs have multiple display names; "
            "the latest name is metadata only"
        )
    if eligible_matchups == 0:
        hard_failures.append("no eligible stable-identity role matchups remain")

    audit = PlayerMapAudit(
        input_rows=len(frame),
        input_games=int(frame["gameid"].nunique()),
        source_counts=source_counts,
        completeness_counts=completeness_counts,
        missing_by_column=missing_by_column,
        non_oe_rows=non_oe,
        partial_rows_excluded=partial_rows,
        complete_rows=len(complete),
        complete_games=int(complete["gameid"].nunique()),
        invalid_side_rows=invalid_sides,
        invalid_role_rows=invalid_roles,
        duplicate_game_side_role_cells=duplicate_cells,
        malformed_complete_games=malformed_games,
        duplicate_stable_player_games=duplicate_player_games,
        missing_stable_player_rows=missing_player_rows,
        missing_team_key_rows=missing_team_rows,
        complete_target_missing_rows=int(target_missing.sum()),
        nonfinite_target_rows=int(target_nonfinite.sum()),
        non_antisymmetric_matchups=int(non_antisymmetric.sum()),
        numeric_patch_rows=numeric_patch_rows,
        stable_identity_matchups=stable_identity_matchups,
        complete_matchups=complete_matchups,
        eligible_matchups=eligible_matchups,
        stable_identity_matchup_coverage=coverage,
        player_ids_with_multiple_names=ids_multiple_names,
        names_with_multiple_player_ids=names_multiple_ids,
        hard_failures=tuple(hard_failures),
        warnings=tuple(warnings),
    )

    if matchups.empty:
        return PreparedPlayerMaps(pd.DataFrame(), audit)
    selected = matchups[eligible].copy()
    prepared = pd.DataFrame(
        {
            "game_id": selected["gameid"],
            "date": selected["date_blue"],
            "event_day": selected["date_blue"].dt.floor("D"),
            "league": selected["league_blue"],
            "patch_context": selected["_patch_context_blue"],
            "role": selected["position"],
            "_role_order": selected["position"].map(role_order),
            "blue_player_id": selected["playerid_blue"],
            "red_player_id": selected["playerid_red"],
            "blue_player_name": selected["playername_blue"],
            "red_player_name": selected["playername_red"],
            "blue_team_key": selected["team_key_blue"],
            "red_team_key": selected["team_key_red"],
            "blue_champion": selected["champion_blue"],
            "red_champion": selected["champion_red"],
            **{
                metric: pd.to_numeric(
                    selected[f"{metric}_blue"], errors="coerce"
                )
                for metric in config.target_metrics
            },
        }
    )
    prepared = (
        prepared.sort_values(
            ["date", "game_id", "_role_order"], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    return PreparedPlayerMaps(prepared, audit)


def audit_player_map_input(
    player_rows: pd.DataFrame,
    config: PlayerPerformanceConfig | None = None,
) -> PlayerMapAudit:
    """Return the fail-closed input audit without fitting any model."""

    cfg = config or PlayerPerformanceConfig()
    return _diagnose_and_prepare(player_rows, cfg).audit


def prepare_player_map_matchups(
    player_rows: pd.DataFrame,
    config: PlayerPerformanceConfig | None = None,
) -> PreparedPlayerMaps:
    """Validate and prepare exact same-role OE matchups.

    Non-complete rows and complete matchups without exact stable player IDs are
    excluded and quantified.  All structural or provenance violations fail
    closed.
    """

    cfg = config or PlayerPerformanceConfig()
    prepared = _diagnose_and_prepare(player_rows, cfg)
    if not prepared.audit.ready:
        raise PlayerPerformanceDataError(
            "; ".join(prepared.audit.hard_failures)
            or "player-map input is not ready"
        )
    return prepared


def _split_matchups(
    matchups: pd.DataFrame,
    config: PlayerPerformanceConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    days = pd.Index(matchups["event_day"].drop_duplicates().sort_values())
    if len(days) < 5:
        raise PlayerPerformanceDataError(
            "at least five distinct event days are required for chronological splits"
        )
    train_cut = max(1, int(math.floor(len(days) * config.train_fraction)))
    validation_cut = max(
        train_cut + 1,
        int(
            math.floor(
                len(days)
                * (config.train_fraction + config.validation_fraction)
            )
        ),
    )
    validation_cut = min(validation_cut, len(days) - 1)
    train_days = set(days[:train_cut])
    validation_days = set(days[train_cut:validation_cut])
    test_days = set(days[validation_cut:])
    train = matchups[matchups["event_day"].isin(train_days)].copy()
    validation = matchups[
        matchups["event_day"].isin(validation_days)
    ].copy()
    test = matchups[matchups["event_day"].isin(test_days)].copy()
    if train.empty or validation.empty or test.empty:
        raise PlayerPerformanceDataError(
            "chronological split produced an empty partition"
        )
    if not train["date"].max() < validation["date"].min():
        raise PlayerPerformanceDataError("train/validation chronology overlaps")
    if not validation["date"].max() < test["date"].min():
        raise PlayerPerformanceDataError("validation/test chronology overlaps")
    boundaries = {
        "train_start": str(train["date"].min()),
        "train_end": str(train["date"].max()),
        "validation_start": str(validation["date"].min()),
        "validation_end": str(validation["date"].max()),
        "test_start": str(test["date"].min()),
        "test_end": str(test["date"].max()),
    }
    return train, validation, test, boundaries


def _design_schema(
    role_rows: pd.DataFrame,
    *,
    include_player_effects: bool,
) -> tuple[tuple[str, ...], tuple[str, ...], str, str]:
    players = (
        sorted(
            set(role_rows["blue_player_id"])
            | set(role_rows["red_player_id"])
        )
        if include_player_effects
        else []
    )
    champions = sorted(
        set(role_rows["blue_champion"])
        | set(role_rows["red_champion"])
    )
    teams = sorted(
        set(role_rows["blue_team_key"])
        | set(role_rows["red_team_key"])
    )
    leagues = sorted(set(role_rows["league"]))
    patches = sorted(set(role_rows["patch_context"]))
    reference_league = leagues[0]
    reference_patch = patches[0]
    names = (
        ("intercept",)
        + tuple(f"player:{value}" for value in players)
        + tuple(f"champion:{value}" for value in champions)
        + tuple(f"team:{value}" for value in teams)
        + tuple(f"league:{value}" for value in leagues[1:])
        + tuple(f"patch:{value}" for value in patches[1:])
    )
    blocks = (
        ("intercept",)
        + ("player",) * len(players)
        + ("champion",) * len(champions)
        + ("team",) * len(teams)
        + ("context",) * (len(leagues) - 1)
        + ("context",) * (len(patches) - 1)
    )
    return names, blocks, reference_league, reference_patch


def _build_design(
    role_rows: pd.DataFrame,
    feature_names: Sequence[str],
) -> sparse.csr_matrix:
    index = {name: column for column, name in enumerate(feature_names)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []

    def add(row: int, feature: str, value: float) -> None:
        column = index.get(feature)
        if column is None:
            return
        row_indices.append(row)
        column_indices.append(column)
        values.append(value)

    for row_index, row in enumerate(role_rows.itertuples(index=False)):
        add(row_index, "intercept", 1.0)
        add(row_index, f"player:{row.blue_player_id}", 1.0)
        add(row_index, f"player:{row.red_player_id}", -1.0)
        add(row_index, f"champion:{row.blue_champion}", 1.0)
        add(row_index, f"champion:{row.red_champion}", -1.0)
        add(row_index, f"team:{row.blue_team_key}", 1.0)
        add(row_index, f"team:{row.red_team_key}", -1.0)
        add(row_index, f"league:{row.league}", 1.0)
        add(row_index, f"patch:{row.patch_context}", 1.0)
    return sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(role_rows), len(feature_names)),
        dtype=float,
    )


def _transform_design(
    role_rows: pd.DataFrame,
    model: RolePerformanceModel,
) -> tuple[sparse.csr_matrix, pd.DataFrame]:
    feature_set = set(model.feature_names)
    metadata = pd.DataFrame(
        {
            "unseen_player": [
                (
                    f"player:{blue}" not in feature_set
                    or f"player:{red}" not in feature_set
                )
                for blue, red in zip(
                    role_rows["blue_player_id"],
                    role_rows["red_player_id"],
                )
            ],
            "unseen_champion": [
                (
                    f"champion:{blue}" not in feature_set
                    or f"champion:{red}" not in feature_set
                )
                for blue, red in zip(
                    role_rows["blue_champion"],
                    role_rows["red_champion"],
                )
            ],
            "unseen_team": [
                (
                    f"team:{blue}" not in feature_set
                    or f"team:{red}" not in feature_set
                )
                for blue, red in zip(
                    role_rows["blue_team_key"],
                    role_rows["red_team_key"],
                )
            ],
            "unseen_league": [
                (
                    league != model.reference_league
                    and f"league:{league}" not in feature_set
                )
                for league in role_rows["league"]
            ],
            "unseen_patch": [
                (
                    patch != model.reference_patch
                    and f"patch:{patch}" not in feature_set
                )
                for patch in role_rows["patch_context"]
            ],
        }
    )
    return _build_design(role_rows, model.feature_names), metadata


def _penalty_diagonal(
    blocks: Sequence[str],
    penalty: PenaltySpec,
) -> np.ndarray:
    values = {
        "intercept": penalty.intercept_l2,
        "player": penalty.player_l2,
        "champion": penalty.champion_l2,
        "team": penalty.team_l2,
        "context": penalty.context_l2,
    }
    return np.asarray([values[block] for block in blocks], dtype=float)


def _covariance_diagonal(
    design: sparse.csr_matrix,
    penalty_diagonal: np.ndarray,
    residual_variance: float,
    config: PlayerPerformanceConfig,
    *,
    seed_offset: int,
) -> tuple[np.ndarray, str]:
    feature_count = design.shape[1]
    hessian = (design.T @ design).tocsc()
    hessian = hessian + sparse.diags(penalty_diagonal, format="csc")
    if feature_count <= config.exact_covariance_max_features:
        dense = hessian.toarray()
        factor = cho_factor(dense, lower=True, check_finite=True)
        covariance = cho_solve(
            factor,
            np.eye(feature_count, dtype=float),
            check_finite=True,
        )
        return (
            np.maximum(np.diag(covariance) * residual_variance, 0.0),
            "exact_gaussian_ridge_covariance",
        )

    operator = LinearOperator(
        shape=(feature_count, feature_count),
        dtype=float,
        matvec=lambda vector: np.asarray(hessian @ vector).reshape(-1),
    )
    rng = np.random.default_rng(config.random_seed + seed_offset)
    estimate = np.zeros(feature_count, dtype=float)
    successful = 0
    for _ in range(config.uncertainty_probes):
        probe = rng.choice(np.asarray([-1.0, 1.0]), size=feature_count)
        solution, info = cg(
            operator,
            probe,
            rtol=config.uncertainty_cg_tolerance,
            atol=0.0,
            maxiter=config.uncertainty_cg_maxiter,
        )
        if info != 0 or not np.isfinite(solution).all():
            continue
        estimate += probe * solution
        successful += 1
    if successful < max(2, config.uncertainty_probes // 2):
        conservative = residual_variance / np.maximum(
            np.asarray(design.power(2).sum(axis=0)).reshape(-1)
            + penalty_diagonal,
            1e-12,
        )
        return (
            np.maximum(conservative, 0.0),
            "diagonal_information_lower_fidelity",
        )
    estimate /= successful
    return (
        np.maximum(estimate * residual_variance, 0.0),
        f"hutchinson_gaussian_ridge_covariance_{successful}_probes",
    )


def _fit_role_model(
    role_rows: pd.DataFrame,
    role: str,
    penalty: PenaltySpec,
    config: PlayerPerformanceConfig,
    *,
    compute_uncertainty: bool,
    include_player_effects: bool,
) -> RolePerformanceModel:
    if len(role_rows) < config.min_train_matchups_per_role:
        raise PlayerPerformanceDataError(
            f"{role} has {len(role_rows)} train matchups; "
            f"{config.min_train_matchups_per_role} are required"
        )
    feature_names, blocks, reference_league, reference_patch = _design_schema(
        role_rows,
        include_player_effects=include_player_effects,
    )
    design = _build_design(role_rows, feature_names)
    target = role_rows["observed_performance"].to_numpy(dtype=float)
    penalties = _penalty_diagonal(blocks, penalty)
    augmented = sparse.vstack(
        (design, sparse.diags(np.sqrt(penalties), format="csr")),
        format="csr",
    )
    augmented_target = np.concatenate(
        (target, np.zeros(len(feature_names), dtype=float))
    )
    solution = lsmr(
        augmented,
        augmented_target,
        atol=config.lsmr_tolerance,
        btol=config.lsmr_tolerance,
        maxiter=config.lsmr_maxiter,
    )
    coefficients = np.asarray(solution[0], dtype=float)
    if not np.isfinite(coefficients).all():
        raise PlayerPerformanceDataError(
            f"{role} fit produced non-finite coefficients"
        )
    residual = target - np.asarray(design @ coefficients).reshape(-1)
    residual_variance = max(float(np.mean(np.square(residual))), 1e-12)
    if compute_uncertainty:
        variances, uncertainty_method = _covariance_diagonal(
            design,
            penalties,
            residual_variance,
            config,
            seed_offset=CANONICAL_ROLES.index(role) * 10_000,
        )
    else:
        variances = residual_variance / np.maximum(
            np.asarray(design.power(2).sum(axis=0)).reshape(-1) + penalties,
            1e-12,
        )
        uncertainty_method = "diagonal_information_validation_only"
    return RolePerformanceModel(
        role=role,
        feature_names=tuple(feature_names),
        feature_blocks=tuple(blocks),
        coefficients=coefficients,
        coefficient_variance=np.asarray(variances, dtype=float),
        penalty_diagonal=penalties,
        reference_league=reference_league,
        reference_patch=reference_patch,
        residual_scale=math.sqrt(residual_variance),
        n_fit_matchups=len(role_rows),
        lsmr_iterations=int(solution[2]),
        lsmr_condition_estimate=float(solution[6]),
        uncertainty_method=uncertainty_method,
    )


def fit_player_performance_candidate(
    fit_matchups: pd.DataFrame,
    *,
    standardizer: RobustContextStandardizer | None = None,
    base_penalty: float = 32.0,
    config: PlayerPerformanceConfig | None = None,
    compute_uncertainty: bool = True,
    include_player_effects: bool = True,
) -> PlayerPerformanceCandidate:
    """Fit one frozen sparse role-specific candidate.

    ``fit_matchups`` must come from :func:`prepare_player_map_matchups`.
    """

    cfg = config or PlayerPerformanceConfig()
    if fit_matchups.empty:
        raise PlayerPerformanceDataError("fit_matchups is empty")
    scaler = standardizer or RobustContextStandardizer.fit(
        fit_matchups, cfg
    )
    transformed = scaler.transform(fit_matchups)
    penalty = PenaltySpec.from_base(base_penalty, cfg)
    models: dict[str, RolePerformanceModel] = {}
    for role in cfg.canonical_roles:
        role_rows = transformed[transformed["role"].eq(role)].copy()
        models[role] = _fit_role_model(
            role_rows,
            role,
            penalty,
            cfg,
            compute_uncertainty=compute_uncertainty,
            include_player_effects=include_player_effects,
        )
    return PlayerPerformanceCandidate(
        role_models=models,
        standardizer=scaler,
        penalty=penalty,
        fit_start=pd.Timestamp(fit_matchups["date"].min()),
        fit_end=pd.Timestamp(fit_matchups["date"].max()),
        fit_matchups=len(fit_matchups),
        includes_player_effects=include_player_effects,
    )


def _expand_matchup_metadata(matchups: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for side, opponent in (("blue", "red"), ("red", "blue")):
        sign = 1.0 if side == "blue" else -1.0
        piece = pd.DataFrame(
            {
                "game_id": matchups["game_id"],
                "date": matchups["date"],
                "event_day": matchups["event_day"],
                "league": matchups["league"],
                "patch_context": matchups["patch_context"],
                "role": matchups["role"],
                "_role_order": matchups["_role_order"],
                "side": side.title(),
                "player_id": matchups[f"{side}_player_id"],
                "player_name": matchups[f"{side}_player_name"],
                "opponent_player_id": matchups[f"{opponent}_player_id"],
                "team_key": matchups[f"{side}_team_key"],
                "opponent_team_key": matchups[f"{opponent}_team_key"],
                "champion": matchups[f"{side}_champion"],
                "opponent_champion": matchups[f"{opponent}_champion"],
                "_sign": sign,
            }
        )
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def _prediction_ledger(
    predicted_matchups: pd.DataFrame,
    *,
    split: str,
    evaluation_kind: str,
    fit_end: pd.Timestamp,
    known_patches: set[str],
    previous_team: Mapping[str, str],
) -> pd.DataFrame:
    metadata = _expand_matchup_metadata(predicted_matchups)
    repeated = pd.concat(
        (predicted_matchups, predicted_matchups), ignore_index=True
    )
    sign = metadata["_sign"].to_numpy(dtype=float)
    ledger = metadata.drop(columns=["_sign"]).copy()
    ledger["observed_performance"] = (
        repeated["observed_performance"].to_numpy(dtype=float) * sign
    )
    ledger["predicted_performance"] = (
        repeated["predicted_performance"].to_numpy(dtype=float) * sign
    )
    ledger["residual"] = (
        ledger["observed_performance"]
        - ledger["predicted_performance"]
    )
    ledger["target_component_sd"] = repeated[
        "target_component_sd"
    ].to_numpy(dtype=float)
    ledger["scale_fallback"] = repeated["scale_fallback"].to_numpy()
    for column in (
        "unseen_player",
        "unseen_champion",
        "unseen_team",
        "unseen_league",
        "unseen_patch",
    ):
        ledger[column] = repeated[column].to_numpy(dtype=bool)
    ledger["future_patch"] = ~ledger["patch_context"].isin(known_patches)
    ledger["previous_team_key"] = ledger["player_id"].map(previous_team)
    ledger["roster_move"] = (
        ledger["previous_team_key"].notna()
        & ledger["previous_team_key"].ne(ledger["team_key"])
    )
    ledger["split"] = split
    ledger["evaluation_kind"] = evaluation_kind
    ledger["model_fit_through"] = fit_end
    ledger["target_uses_match_result"] = False
    ledger["estimand"] = ESTIMAND
    return (
        ledger.sort_values(
            ["date", "game_id", "_role_order", "side"],
            kind="mergesort",
        )
        .drop(columns=["_role_order"])
        .reset_index(drop=True)
    )


def _latest_team(matchups: pd.DataFrame) -> dict[str, str]:
    appearances = _expand_matchup_metadata(matchups)
    latest = (
        appearances.sort_values(["date", "game_id"], kind="mergesort")
        .groupby("player_id", sort=True)
        .tail(1)
    )
    return dict(zip(latest["player_id"], latest["team_key"]))


def performance_metrics(frame: pd.DataFrame) -> PerformanceMetrics:
    if frame.empty:
        return PerformanceMetrics.unavailable()
    observed = frame["observed_performance"].to_numpy(dtype=float)
    predicted = frame["predicted_performance"].to_numpy(dtype=float)
    residual = observed - predicted
    rmse = math.sqrt(float(np.mean(np.square(residual))))
    mae = float(np.mean(np.abs(residual)))
    baseline_rmse = math.sqrt(float(np.mean(np.square(observed))))
    denominator = float(np.sum(np.square(observed - observed.mean())))
    r2 = (
        1.0 - float(np.sum(np.square(residual))) / denominator
        if denominator > 0.0
        else float("nan")
    )
    correlation = spearmanr(observed, predicted).statistic
    spearman = (
        float(correlation) if correlation is not None else float("nan")
    )
    relative_lift = (
        (baseline_rmse - rmse) / baseline_rmse
        if baseline_rmse > 0.0
        else float("nan")
    )
    return PerformanceMetrics(
        rows=len(frame),
        rmse=rmse,
        mae=mae,
        r2=r2,
        spearman=spearman,
        zero_baseline_rmse=baseline_rmse,
        relative_rmse_lift=relative_lift,
    )


def _paired_rmse_contrast(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    config: PlayerPerformanceConfig,
) -> PairedRMSEContrast:
    key_columns = [
        "game_id",
        "date",
        "event_day",
        "role",
        "side",
        "player_id",
    ]
    candidate_ordered = candidate.sort_values(
        key_columns, kind="mergesort"
    ).reset_index(drop=True)
    baseline_ordered = baseline.sort_values(
        key_columns, kind="mergesort"
    ).reset_index(drop=True)
    if len(candidate_ordered) != len(baseline_ordered):
        raise PlayerPerformanceDataError(
            "candidate and context baseline ledgers have different rows"
        )
    if not candidate_ordered[key_columns].equals(
        baseline_ordered[key_columns]
    ):
        raise PlayerPerformanceDataError(
            "candidate and context baseline ledgers are not paired"
        )
    blocks = pd.DataFrame(
        {
            "event_day": candidate_ordered["event_day"],
            "candidate_sq": np.square(
                candidate_ordered["residual"].to_numpy(dtype=float)
            ),
            "baseline_sq": np.square(
                baseline_ordered["residual"].to_numpy(dtype=float)
            ),
        }
    ).groupby("event_day", sort=True).agg(
        candidate_sse=("candidate_sq", "sum"),
        baseline_sse=("baseline_sq", "sum"),
        rows=("candidate_sq", "size"),
    )
    if len(blocks) < 2:
        raise PlayerPerformanceDataError(
            "paired metric uncertainty requires at least two calendar days"
        )
    candidate_sse = blocks["candidate_sse"].to_numpy(dtype=float)
    baseline_sse = blocks["baseline_sse"].to_numpy(dtype=float)
    row_counts = blocks["rows"].to_numpy(dtype=float)
    candidate_rmse = math.sqrt(candidate_sse.sum() / row_counts.sum())
    baseline_rmse = math.sqrt(baseline_sse.sum() / row_counts.sum())
    observed_lift = (
        (baseline_rmse - candidate_rmse) / baseline_rmse
        if baseline_rmse > 0.0
        else float("nan")
    )
    rng = np.random.default_rng(config.random_seed + 991_337)
    lifts = np.empty(config.metric_bootstrap_replicates, dtype=float)
    block_count = len(blocks)
    for replicate in range(config.metric_bootstrap_replicates):
        sampled = rng.integers(0, block_count, size=block_count)
        denominator = row_counts[sampled].sum()
        candidate_sample_rmse = math.sqrt(
            candidate_sse[sampled].sum() / denominator
        )
        baseline_sample_rmse = math.sqrt(
            baseline_sse[sampled].sum() / denominator
        )
        lifts[replicate] = (
            (baseline_sample_rmse - candidate_sample_rmse)
            / baseline_sample_rmse
            if baseline_sample_rmse > 0.0
            else float("nan")
        )
    alpha = (1.0 - config.metric_ci_level) / 2.0
    finite_lifts = lifts[np.isfinite(lifts)]
    if finite_lifts.size < config.metric_bootstrap_replicates * 0.95:
        raise PlayerPerformanceDataError(
            "too few finite paired bootstrap replicates"
        )
    ci_low, ci_high = np.quantile(
        finite_lifts, [alpha, 1.0 - alpha]
    )
    return PairedRMSEContrast(
        rows=len(candidate_ordered),
        calendar_day_blocks=block_count,
        candidate_rmse=candidate_rmse,
        baseline_rmse=baseline_rmse,
        relative_rmse_lift=observed_lift,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        confidence_level=config.metric_ci_level,
        bootstrap_replicates=config.metric_bootstrap_replicates,
    )


def run_player_performance_tournament(
    player_rows: pd.DataFrame,
    config: PlayerPerformanceConfig | None = None,
) -> PlayerPerformanceTournament:
    """Run the predeclared chronological train/validation/test tournament.

    Hyperparameters see training and validation only.  The test split is scored
    once after selection.  All reported predictive metrics evaluate held-out
    player-map performance, never map wins.
    """

    cfg = config or PlayerPerformanceConfig()
    prepared = prepare_player_map_matchups(player_rows, cfg)
    train, validation, test, boundaries = _split_matchups(
        prepared.matchups, cfg
    )
    train_scaler = RobustContextStandardizer.fit(train, cfg)
    validation_rows: list[dict[str, Any]] = []
    candidates: dict[float, PlayerPerformanceCandidate] = {}
    context_candidates: dict[float, PlayerPerformanceCandidate] = {}
    for base_penalty in cfg.ridge_grid:
        candidate = fit_player_performance_candidate(
            train,
            standardizer=train_scaler,
            base_penalty=base_penalty,
            config=cfg,
            compute_uncertainty=False,
        )
        candidates[float(base_penalty)] = candidate
        predicted = candidate.predict(validation)
        metrics = performance_metrics(
            _prediction_ledger(
                predicted,
                split="validation",
                evaluation_kind="validation_holdout",
                fit_end=candidate.fit_end,
                known_patches=set(train["patch_context"]),
                previous_team=_latest_team(train),
            )
        )
        validation_rows.append(
            {
                "base_penalty": float(base_penalty),
                "validation_rows": metrics.rows,
                "validation_rmse": metrics.rmse,
                "validation_mae": metrics.mae,
                "validation_r2": metrics.r2,
                "validation_spearman": metrics.spearman,
                "zero_baseline_rmse": metrics.zero_baseline_rmse,
                "relative_rmse_lift": metrics.relative_rmse_lift,
            }
        )
        context_candidate = fit_player_performance_candidate(
            train,
            standardizer=train_scaler,
            base_penalty=base_penalty,
            config=cfg,
            compute_uncertainty=False,
            include_player_effects=False,
        )
        context_candidates[float(base_penalty)] = context_candidate
        context_predicted = context_candidate.predict(validation)
        context_metrics = performance_metrics(
            _prediction_ledger(
                context_predicted,
                split="validation",
                evaluation_kind="validation_context_only_baseline",
                fit_end=context_candidate.fit_end,
                known_patches=set(train["patch_context"]),
                previous_team=_latest_team(train),
            )
        )
        validation_rows[-1].update(
            {
                "context_validation_rmse": context_metrics.rmse,
                "context_validation_mae": context_metrics.mae,
                "context_validation_r2": context_metrics.r2,
                "context_validation_spearman": context_metrics.spearman,
            }
        )
    validation_candidates = pd.DataFrame(validation_rows).sort_values(
        ["validation_rmse", "validation_mae", "base_penalty"],
        kind="mergesort",
    )
    selected_base = float(validation_candidates.iloc[0]["base_penalty"])
    selected_context_base = float(
        validation_candidates.sort_values(
            [
                "context_validation_rmse",
                "context_validation_mae",
                "base_penalty",
            ],
            kind="mergesort",
        ).iloc[0]["base_penalty"]
    )
    validation_candidate = candidates[selected_base]
    validation_context_candidate = context_candidates[selected_context_base]
    train_predicted = validation_candidate.predict(train)
    validation_predicted = validation_candidate.predict(validation)
    previous_train_team = _latest_team(train)
    train_ledger = _prediction_ledger(
        train_predicted,
        split="train",
        evaluation_kind="in_sample_fit_diagnostic",
        fit_end=validation_candidate.fit_end,
        known_patches=set(train["patch_context"]),
        previous_team=previous_train_team,
    )
    validation_ledger = _prediction_ledger(
        validation_predicted,
        split="validation",
        evaluation_kind="validation_holdout",
        fit_end=validation_candidate.fit_end,
        known_patches=set(train["patch_context"]),
        previous_team=previous_train_team,
    )
    validation_context_ledger = _prediction_ledger(
        validation_context_candidate.predict(validation),
        split="validation",
        evaluation_kind="validation_context_only_baseline",
        fit_end=validation_context_candidate.fit_end,
        known_patches=set(train["patch_context"]),
        previous_team=previous_train_team,
    )

    train_validation = (
        pd.concat((train, validation), ignore_index=True)
        .sort_values(
            ["date", "game_id", "_role_order"], kind="mergesort"
        )
        .reset_index(drop=True)
    )
    final_scaler = RobustContextStandardizer.fit(train_validation, cfg)
    final_candidate = fit_player_performance_candidate(
        train_validation,
        standardizer=final_scaler,
        base_penalty=selected_base,
        config=cfg,
        compute_uncertainty=True,
    )
    final_context_candidate = fit_player_performance_candidate(
        train_validation,
        standardizer=final_scaler,
        base_penalty=selected_context_base,
        config=cfg,
        compute_uncertainty=False,
        include_player_effects=False,
    )
    test_predicted = final_candidate.predict(test)
    previous_final_team = _latest_team(train_validation)
    test_ledger = _prediction_ledger(
        test_predicted,
        split="test",
        evaluation_kind="frozen_test_holdout",
        fit_end=final_candidate.fit_end,
        known_patches=set(train_validation["patch_context"]),
        previous_team=previous_final_team,
    )
    test_context_ledger = _prediction_ledger(
        final_context_candidate.predict(test),
        split="test",
        evaluation_kind="frozen_test_context_only_baseline",
        fit_end=final_context_candidate.fit_end,
        known_patches=set(train_validation["patch_context"]),
        previous_team=previous_final_team,
    )
    ledger = pd.concat(
        (train_ledger, validation_ledger, test_ledger),
        ignore_index=True,
    )
    train_metrics = performance_metrics(
        ledger[ledger["split"].eq("train")]
    )
    validation_metrics = performance_metrics(
        ledger[ledger["split"].eq("validation")]
    )
    test_frame = ledger[ledger["split"].eq("test")]
    test_metrics = performance_metrics(test_frame)
    validation_context_metrics = performance_metrics(
        validation_context_ledger
    )
    test_context_metrics = performance_metrics(test_context_ledger)
    future_patch_metrics = performance_metrics(
        test_frame[test_frame["future_patch"]]
    )
    roster_move_metrics = performance_metrics(
        test_frame[test_frame["roster_move"]]
    )
    player_ratings = final_candidate.player_ratings(
        train_validation, cfg
    )
    player_incremental_lift = (
        (
            test_context_metrics.rmse - test_metrics.rmse
        )
        / test_context_metrics.rmse
        if test_context_metrics.rmse > 0.0
        else float("nan")
    )
    player_incremental_contrast = _paired_rmse_contrast(
        test_ledger,
        test_context_ledger,
        cfg,
    )
    gate = bool(
        math.isfinite(test_metrics.relative_rmse_lift)
        and test_metrics.relative_rmse_lift
        >= cfg.minimum_test_relative_rmse_lift
        and math.isfinite(player_incremental_lift)
        and player_incremental_lift
        >= cfg.minimum_player_incremental_rmse_lift
        and (
            not cfg.require_positive_player_lift_ci
            or player_incremental_contrast.ci_low
            >= cfg.minimum_player_incremental_rmse_lift
        )
    )
    return PlayerPerformanceTournament(
        audit=prepared.audit,
        selected_base_penalty=selected_base,
        selected_context_base_penalty=selected_context_base,
        validation_candidates=validation_candidates.reset_index(drop=True),
        split_boundaries=boundaries,
        prediction_ledger=ledger,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        validation_context_baseline_metrics=validation_context_metrics,
        test_context_baseline_metrics=test_context_metrics,
        future_patch_test_metrics=future_patch_metrics,
        roster_move_test_metrics=roster_move_metrics,
        player_ratings=player_ratings,
        player_incremental_test_rmse_lift=player_incremental_lift,
        player_incremental_test_contrast=player_incremental_contrast,
        test_gate_passed=gate,
    )
