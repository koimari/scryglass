"""Research-candidate adjusted plus-minus for complete 5v5 map lineups.

The observed outcome is a team result, not an individual result.  This module
therefore treats player effects as a regularized decomposition of lineup
outcomes and reports the design limitations that make many individual effects
inseparable.  In particular, players with identical signed map exposure have
identical design columns; ridge shrinkage can choose a decomposition for them,
but the maps cannot.

The public entry points are:

* :func:`validate_lineups` for strict canonical 5v5 validation;
* :func:`fit_player_apm_candidate` for a fit with predeclared penalties;
* :func:`select_player_apm_candidate` for validation-only hyperparameter choice;
* :func:`chronological_player_apm_evaluation` for a train/validation/test ledger.

This is intentionally labelled a research candidate.  It is not a production
rating and should not be promoted without population, calibration, stability,
and external-holdout review.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import LinearOperator, cg, lsmr, svds
from scipy.special import expit
from scipy.stats import norm


CANONICAL_ROLES: tuple[str, ...] = ("top", "jng", "mid", "bot", "sup")
LINEUP_PLAYER_COLUMNS: tuple[str, ...] = tuple(
    f"{side}_{role}" for side in ("blue", "red") for role in CANONICAL_ROLES
)
REQUIRED_LINEUP_COLUMNS: tuple[str, ...] = (
    "game_id",
    "date",
    "blue_win",
    *LINEUP_PLAYER_COLUMNS,
)
MODEL_KIND = "research_candidate_player_apm"
PROMOTION_STATUS = "research_candidate_not_production"


class LineupValidationError(ValueError):
    """Raised when a map is not an exact canonical role-complete 5v5."""


@dataclass(frozen=True)
class PlayerAPMConfig:
    """Predeclared model and validation choices.

    ``league_levels`` fixes the nuisance feature space before fitting.  When a
    global blue-side term is present, the first league is the reference and the
    remaining league coefficients are deviations.  Without a global side term,
    each predeclared league receives its own side-intercept column.

    The player prior is zero-centred grouped ridge.  ``role_l2_multipliers``
    permits a predeclared role-specific prior scale while preserving one
    coefficient per player; a role-swapping player's multiplier is the
    exposure-weighted mean from the fit data only.
    """

    include_side_term: bool = True
    league_levels: tuple[str, ...] = ()
    player_l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    nuisance_l2_grid: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)
    role_l2_multipliers: tuple[tuple[str, float], ...] = (
        ("top", 1.0),
        ("jng", 1.0),
        ("mid", 1.0),
        ("bot", 1.0),
        ("sup", 1.0),
    )
    selection_metric: str = "log_loss"
    max_iter: int = 500
    optimizer_tolerance: float = 1e-10
    rank_rtol: float = 1e-10
    covariance_rcond: float = 1e-10
    probability_clip: float = 1e-9
    diagnostic_exact_max_cells: int = 2_000_000
    diagnostic_exact_max_columns: int = 256
    diagnostic_svd_components: int = 64
    max_uncertainty_players: int = 32
    uncertainty_cg_maxiter: int = 2_000
    identifiability_lsmr_maxiter: int = 2_000

    def __post_init__(self) -> None:
        if self.selection_metric not in {"brier", "log_loss"}:
            raise ValueError("selection_metric must be 'brier' or 'log_loss'")
        if not self.player_l2_grid or not self.nuisance_l2_grid:
            raise ValueError("ridge grids must be non-empty")
        for name, values in (
            ("player_l2_grid", self.player_l2_grid),
            ("nuisance_l2_grid", self.nuisance_l2_grid),
        ):
            if any(not math.isfinite(value) or value <= 0.0 for value in values):
                raise ValueError(f"{name} must contain finite positive values")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")
        if not 0.0 < self.optimizer_tolerance < 1.0:
            raise ValueError("optimizer_tolerance must be between zero and one")
        if not 0.0 < self.rank_rtol < 1.0:
            raise ValueError("rank_rtol must be between zero and one")
        if not 0.0 < self.covariance_rcond < 1.0:
            raise ValueError("covariance_rcond must be between zero and one")
        if not 0.0 < self.probability_clip < 0.5:
            raise ValueError("probability_clip must be between zero and 0.5")
        if self.diagnostic_exact_max_cells < 1:
            raise ValueError("diagnostic_exact_max_cells must be positive")
        if self.diagnostic_exact_max_columns < 1:
            raise ValueError("diagnostic_exact_max_columns must be positive")
        if self.diagnostic_svd_components < 2:
            raise ValueError("diagnostic_svd_components must be at least two")
        if self.max_uncertainty_players < 1:
            raise ValueError("max_uncertainty_players must be positive")
        if self.uncertainty_cg_maxiter < 1:
            raise ValueError("uncertainty_cg_maxiter must be positive")
        if self.identifiability_lsmr_maxiter < 1:
            raise ValueError("identifiability_lsmr_maxiter must be positive")

        levels = tuple(self.league_levels)
        if len(set(levels)) != len(levels):
            raise ValueError("league_levels must be unique and predeclared")
        if any(not value or value != value.strip() for value in levels):
            raise ValueError("league_levels must be non-empty canonical strings")

        multipliers = dict(self.role_l2_multipliers)
        if set(multipliers) != set(CANONICAL_ROLES):
            raise ValueError(
                "role_l2_multipliers must declare each canonical role exactly once"
            )
        if len(multipliers) != len(self.role_l2_multipliers):
            raise ValueError("role_l2_multipliers contains duplicate roles")
        if any(
            not math.isfinite(value) or value <= 0.0
            for value in multipliers.values()
        ):
            raise ValueError("role_l2_multipliers must be finite and positive")


@dataclass(frozen=True)
class ExposureCohort:
    """Players whose complete signed map-exposure vectors are exactly equal."""

    players: tuple[str, ...]
    n_maps: int
    signature_sha256: str
    separately_identified_by_outcomes: bool = False
    interpretation: str = (
        "identical outcome-design columns; within-cohort differences are "
        "determined by the prior, not by map outcomes"
    )


@dataclass(frozen=True)
class DesignDiagnostics:
    n_maps: int
    n_columns: int
    rank: int
    nullity: int
    effective_rank: float
    condition_number: float
    nonzero_condition_number: float
    n_player_columns: int
    player_rank: int
    player_nullity: int
    player_effective_rank: float
    player_condition_number: float
    player_nonzero_condition_number: float
    singular_values: tuple[float, ...]
    player_singular_values: tuple[float, ...]
    identical_exposure_cohorts: tuple[ExposureCohort, ...]
    warning: str
    diagnostic_method: str
    player_diagnostic_method: str
    diagnostics_approximate: bool
    rank_interpretation: str
    condition_interpretation: str
    effective_rank_definition: str = (
        "entropy effective rank of the reported unpenalized singular spectrum"
    )


@dataclass(frozen=True)
class DesignMatrix:
    values: sparse.csr_matrix
    outcomes: np.ndarray
    frame: pd.DataFrame
    player_order: tuple[str, ...]
    feature_names: tuple[str, ...]
    unknown_players_by_map: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PlayerCovariance:
    """Joint prior-conditioned Laplace covariance for requested players."""

    players: tuple[str, ...]
    estimates: np.ndarray
    covariance: np.ndarray
    correlation: np.ndarray
    level_data_identified: tuple[bool, ...]
    identical_exposure_cohorts: tuple[ExposureCohort, ...]
    identifiability_method: str
    uncertainty_kind: str = (
        "bounded_joint_penalized_laplace_via_sparse_hessian_solves"
    )
    warning: str = (
        "marginal player levels are regularization-anchored; use covariance-aware "
        "contrasts and inspect data_identified before interpretation"
    )


@dataclass(frozen=True)
class PlayerContrast:
    players: tuple[str, ...]
    weights: np.ndarray
    estimate: float
    standard_error: float
    interval_low: float
    interval_high: float
    confidence: float
    covariance: np.ndarray
    data_identified: bool
    identical_exposure_confounded: bool
    identifiability_method: str
    uncertainty_kind: str = (
        "bounded_joint_penalized_laplace_contrast_via_sparse_hessian_solves"
    )
    warning: str = ""


@dataclass(frozen=True)
class CandidateSelection:
    model: "PlayerAPMCandidate"
    player_l2: float
    nuisance_l2: float
    validation_metrics: Mapping[str, float]
    candidate_ledger: pd.DataFrame
    train_game_ids: tuple[str, ...]
    validation_game_ids: tuple[str, ...]
    selection_data_scope: str = "chronological_validation_only"


@dataclass(frozen=True)
class ChronologicalPlayerAPMEvaluation:
    ledger: pd.DataFrame
    metrics: Mapping[str, Mapping[str, float]]
    selection: CandidateSelection
    test_model: "PlayerAPMCandidate"
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    model_kind: str = MODEL_KIND
    promotion_status: str = PROMOTION_STATUS
    leakage_note: str = (
        "hyperparameters use chronological validation only; test outcomes are "
        "used only for final scoring"
    )


def _canonical_game_column(frame: pd.DataFrame) -> str:
    # Match the repository adapters: canonical game_uid takes precedence over
    # the source-provider gameid when both are retained for provenance.
    for candidate in ("game_id", "game_uid", "gameid"):
        if candidate in frame:
            return candidate
    raise LineupValidationError(
        "lineups require one of game_id, game_uid, or gameid"
    )


def _coerce_binary(values: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    invalid = numeric.isna() | ~numeric.isin((0.0, 1.0))
    if invalid.any():
        rows = [int(value) for value in np.flatnonzero(invalid.to_numpy())[:5]]
        raise LineupValidationError(
            f"{label} must be exactly binary 0/1; invalid rows {rows}"
        )
    return numeric.astype(float)


def _clean_player(value: Any, *, game_id: str, column: str) -> str:
    if value is None or pd.isna(value):
        raise LineupValidationError(
            f"map {game_id!r} has missing player in {column}"
        )
    player = str(value)
    if not player or player != player.strip() or player.casefold() == "nan":
        raise LineupValidationError(
            f"map {game_id!r} has non-canonical player value in {column}"
        )
    return player


def _wide_from_long(lineups: pd.DataFrame) -> pd.DataFrame:
    required = {"side", "position", "playername", "date", "result"}
    missing = sorted(required - set(lineups.columns))
    if missing:
        raise LineupValidationError(
            f"long lineup input is missing columns: {missing}"
        )
    game_column = _canonical_game_column(lineups)
    frame = lineups.copy()
    raw_game_id = frame[game_column]
    frame["_game_id"] = raw_game_id.astype(str)
    invalid_game_id = (
        raw_game_id.isna()
        | frame["_game_id"].eq("")
        | frame["_game_id"].ne(frame["_game_id"].str.strip())
        | frame["_game_id"].str.casefold().eq("nan")
    )
    if invalid_game_id.any():
        raise LineupValidationError(
            "game identifiers must be non-empty canonical strings"
        )

    # Current Oracle's Elixir-shaped inputs may include one aggregate team row.
    # It is not a player slot and is excluded before exact 5v5 validation.
    positions = frame["position"].astype(str)
    frame = frame[positions.str.casefold().ne("team")].copy()
    if frame.empty:
        raise LineupValidationError("long lineup input contains no player rows")

    invalid_roles = sorted(
        set(frame["position"].astype(str)) - set(CANONICAL_ROLES)
    )
    if invalid_roles:
        raise LineupValidationError(
            "positions must use exact canonical roles "
            f"{CANONICAL_ROLES}; found {invalid_roles}"
        )
    normalized_side = frame["side"].astype(str).str.casefold()
    invalid_sides = sorted(set(normalized_side) - {"blue", "red"})
    if invalid_sides:
        raise LineupValidationError(
            f"side must be Blue or Red; found {invalid_sides}"
        )
    frame["_side"] = normalized_side
    frame["_result"] = _coerce_binary(frame["result"], "result")
    canonical_names = frame["playername"].map(
        lambda value: _clean_player(
            value,
            game_id="<identity-resolution>",
            column="playername",
        )
    )
    name_keys = canonical_names.str.casefold()
    if "playerid" in frame:
        raw_ids = frame["playerid"].map(
            lambda value: (
                ""
                if value is None or pd.isna(value)
                else str(value).strip()
            )
        )
        ids_by_name = (
            pd.DataFrame({"name": name_keys, "playerid": raw_ids})
            .loc[lambda values: values["playerid"].ne("")]
            .groupby("name", sort=False)["playerid"]
            .agg(lambda values: tuple(sorted(set(values))))
            .to_dict()
        )
        ambiguous_missing = [
            canonical_names.iloc[index]
            for index, (name_key, player_id) in enumerate(
                zip(name_keys, raw_ids)
            )
            if not player_id and len(ids_by_name.get(name_key, ())) > 1
        ]
        if ambiguous_missing:
            raise LineupValidationError(
                "missing playerid cannot be resolved because the same handle "
                f"maps to multiple identities: {sorted(set(ambiguous_missing))[:5]}"
            )
        frame["_player_identity"] = [
            player_id
            or (
                ids_by_name[name_key][0]
                if len(ids_by_name.get(name_key, ())) == 1
                else f"name:{name_key}"
            )
            for name_key, player_id in zip(name_keys, raw_ids)
        ]
    else:
        frame["_player_identity"] = [f"name:{value}" for value in name_keys]

    rows: list[dict[str, Any]] = []
    for game_id, group in frame.groupby("_game_id", sort=False):
        if len(group) != 10:
            raise LineupValidationError(
                f"map {game_id!r} must have exactly 10 player rows; found {len(group)}"
            )
        duplicate_slots = group.duplicated(["_side", "position"], keep=False)
        if duplicate_slots.any():
            slots = sorted(
                {
                    f"{row['_side']}_{row['position']}"
                    for _, row in group[duplicate_slots].iterrows()
                }
            )
            raise LineupValidationError(
                f"map {game_id!r} has duplicate role slots: {slots}"
            )
        observed_slots = {
            (str(row["_side"]), str(row["position"]))
            for _, row in group.iterrows()
        }
        expected_slots = {
            (side, role)
            for side in ("blue", "red")
            for role in CANONICAL_ROLES
        }
        if observed_slots != expected_slots:
            missing_slots = sorted(expected_slots - observed_slots)
            extra_slots = sorted(observed_slots - expected_slots)
            raise LineupValidationError(
                f"map {game_id!r} is not role-complete; "
                f"missing={missing_slots}, extra={extra_slots}"
            )

        dates = pd.to_datetime(group["date"], errors="coerce", utc=True)
        if dates.isna().any() or dates.nunique() != 1:
            raise LineupValidationError(
                f"map {game_id!r} must have one valid shared date"
            )
        side_results: dict[str, float] = {}
        for side in ("blue", "red"):
            values = group.loc[group["_side"].eq(side), "_result"].unique()
            if len(values) != 1:
                raise LineupValidationError(
                    f"map {game_id!r} has inconsistent {side} outcomes"
                )
            side_results[side] = float(values[0])
        if side_results["blue"] + side_results["red"] != 1.0:
            raise LineupValidationError(
                f"map {game_id!r} blue/red outcomes must be complementary"
            )

        row: dict[str, Any] = {
            "game_id": str(game_id),
            "date": dates.iloc[0],
            "blue_win": side_results["blue"],
        }
        for _, player_row in group.iterrows():
            column = f"{player_row['_side']}_{player_row['position']}"
            row[column] = player_row["_player_identity"]

        if "league" in group:
            league_values = group["league"].dropna().astype(str).unique()
            if len(league_values) > 1:
                raise LineupValidationError(
                    f"map {game_id!r} must have one shared league"
                )
            row["league"] = league_values[0] if len(league_values) else None
        rows.append(row)
    return pd.DataFrame(rows)


def _wide_input(lineups: pd.DataFrame) -> pd.DataFrame:
    if {"side", "position", "playername"}.issubset(lineups.columns):
        return _wide_from_long(lineups)

    game_column = _canonical_game_column(lineups)
    outcome_candidates = [
        value for value in ("blue_win", "y_blue_win") if value in lineups
    ]
    if not outcome_candidates:
        raise LineupValidationError(
            "wide lineup input requires blue_win or y_blue_win"
        )
    if len(outcome_candidates) == 2:
        left = pd.to_numeric(lineups["blue_win"], errors="coerce")
        right = pd.to_numeric(lineups["y_blue_win"], errors="coerce")
        if not np.array_equal(
            left.to_numpy(dtype=float),
            right.to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise LineupValidationError(
                "blue_win and y_blue_win contain conflicting outcomes"
            )
    frame = lineups.copy()
    frame["game_id"] = frame[game_column]
    frame["blue_win"] = frame[outcome_candidates[0]]
    return frame


def validate_lineups(
    lineups: pd.DataFrame,
    config: PlayerAPMConfig | None = None,
) -> pd.DataFrame:
    """Return strictly validated canonical wide map lineups.

    Accepted input shapes are the repository's long player-map rows or a wide
    table with one ``blue_<role>`` and ``red_<role>`` column for every canonical
    role.  Role aliases are deliberately rejected.  Every map must contain ten
    distinct, non-empty player identifiers and one binary blue-side result.
    """

    cfg = config or PlayerAPMConfig()
    if not isinstance(lineups, pd.DataFrame) or lineups.empty:
        raise LineupValidationError("lineups must be a non-empty DataFrame")
    frame = _wide_input(lineups)
    missing = sorted(set(REQUIRED_LINEUP_COLUMNS) - set(frame.columns))
    if missing:
        raise LineupValidationError(
            f"wide lineup input is missing columns: {missing}"
        )

    canonical = frame.loc[
        :, [*REQUIRED_LINEUP_COLUMNS, *(["league"] if "league" in frame else [])]
    ].copy().reset_index(drop=True)
    raw_game_id = canonical["game_id"]
    canonical["game_id"] = raw_game_id.astype(str)
    invalid_game_id = (
        raw_game_id.isna()
        | canonical["game_id"].eq("")
        | canonical["game_id"].ne(canonical["game_id"].str.strip())
        | canonical["game_id"].str.casefold().eq("nan")
    )
    if invalid_game_id.any():
        raise LineupValidationError(
            "game_id must be a non-empty canonical string"
        )
    duplicated = canonical["game_id"].duplicated(keep=False)
    if duplicated.any():
        values = sorted(canonical.loc[duplicated, "game_id"].unique())[:5]
        raise LineupValidationError(f"duplicate game_id values: {values}")

    parsed_dates = pd.to_datetime(canonical["date"], errors="coerce", utc=True)
    if parsed_dates.isna().any():
        rows = [
            int(value)
            for value in np.flatnonzero(parsed_dates.isna().to_numpy())[:5]
        ]
        raise LineupValidationError(f"date must be valid; invalid rows {rows}")
    canonical["date"] = parsed_dates
    canonical["blue_win"] = _coerce_binary(
        canonical["blue_win"], "blue_win"
    )

    for column in LINEUP_PLAYER_COLUMNS:
        raw_players = canonical[column]
        text_players = raw_players.astype(str)
        invalid_players = (
            raw_players.isna()
            | text_players.eq("")
            | text_players.ne(text_players.str.strip())
            | text_players.str.casefold().eq("nan")
        )
        if invalid_players.any():
            row_index = int(
                np.flatnonzero(invalid_players.to_numpy())[0]
            )
            _clean_player(
                raw_players.iloc[row_index],
                game_id=str(canonical.iloc[row_index]["game_id"]),
                column=column,
            )
        canonical[column] = text_players

    player_matrix = canonical.loc[
        :, list(LINEUP_PLAYER_COLUMNS)
    ].to_numpy(dtype=str)
    sorted_players = np.sort(player_matrix, axis=1)
    duplicate_rows = np.any(
        sorted_players[:, 1:] == sorted_players[:, :-1], axis=1
    )
    if duplicate_rows.any():
        row_index = int(np.flatnonzero(duplicate_rows)[0])
        values, counts = np.unique(
            player_matrix[row_index], return_counts=True
        )
        duplicates = sorted(values[counts > 1].tolist())
        raise LineupValidationError(
            f"map {canonical.iloc[row_index]['game_id']!r} must have "
            f"10 distinct players; duplicates={duplicates}"
        )

    if cfg.league_levels:
        if "league" not in canonical:
            raise LineupValidationError(
                "league is required when league_levels are predeclared"
            )
        invalid: set[str] = set()
        for value in canonical["league"]:
            if value is None or pd.isna(value):
                invalid.add("<missing>")
                continue
            text = str(value)
            if text != text.strip() or text not in cfg.league_levels:
                invalid.add(text)
        if invalid:
            raise LineupValidationError(
                "league values must belong to predeclared league_levels; "
                f"found {sorted(invalid)}"
            )
        canonical["league"] = canonical["league"].astype(str)
    elif "league" not in canonical:
        canonical["league"] = None

    return canonical.sort_values(
        ["date", "game_id"], kind="mergesort"
    ).reset_index(drop=True)


def _sorted_players(frame: pd.DataFrame) -> tuple[str, ...]:
    values = {
        str(value)
        for column in LINEUP_PLAYER_COLUMNS
        for value in frame[column]
    }
    return tuple(sorted(values, key=lambda value: (value.casefold(), value)))


def _nuisance_feature_names(config: PlayerAPMConfig) -> tuple[str, ...]:
    names: list[str] = []
    if config.include_side_term:
        names.append("nuisance:blue_side")
    if config.league_levels:
        levels = (
            config.league_levels[1:]
            if config.include_side_term
            else config.league_levels
        )
        names.extend(f"nuisance:league:{value}" for value in levels)
    return tuple(names)


def build_design_row(
    lineup: Mapping[str, Any] | pd.Series,
    player_order: Sequence[str],
    config: PlayerAPMConfig | None = None,
) -> np.ndarray:
    """Build one canonical row: +1/5 blue, -1/5 red, then nuisances."""

    cfg = config or PlayerAPMConfig()
    missing = [column for column in LINEUP_PLAYER_COLUMNS if column not in lineup]
    if missing:
        raise LineupValidationError(
            f"design row is missing canonical lineup slots: {missing}"
        )
    game_id = str(lineup.get("game_id", "<design-row>"))
    lineup_players = [
        _clean_player(lineup[column], game_id=game_id, column=column)
        for column in LINEUP_PLAYER_COLUMNS
    ]
    if len(set(lineup_players)) != 10:
        raise LineupValidationError(
            f"map {game_id!r} must have 10 distinct players"
        )
    if cfg.league_levels:
        if "league" not in lineup:
            raise LineupValidationError(
                "design row requires league for predeclared league terms"
            )
        league = str(lineup["league"])
        if league not in cfg.league_levels:
            raise LineupValidationError(
                f"league {league!r} is not predeclared"
            )

    player_order_tuple = tuple(str(value) for value in player_order)
    if len(set(player_order_tuple)) != len(player_order_tuple):
        raise ValueError("player_order must contain unique player identifiers")
    player_index = {player: index for index, player in enumerate(player_order_tuple)}
    row = np.zeros(
        len(player_order_tuple) + len(_nuisance_feature_names(cfg)),
        dtype=float,
    )
    for role in CANONICAL_ROLES:
        blue = str(lineup[f"blue_{role}"])
        red = str(lineup[f"red_{role}"])
        if blue in player_index:
            row[player_index[blue]] += 1.0 / 5.0
        if red in player_index:
            row[player_index[red]] -= 1.0 / 5.0

    offset = len(player_order_tuple)
    if cfg.include_side_term:
        row[offset] = 1.0
        offset += 1
    if cfg.league_levels:
        league = str(lineup["league"])
        levels = (
            cfg.league_levels[1:]
            if cfg.include_side_term
            else cfg.league_levels
        )
        for level in levels:
            row[offset] = 1.0 if league == level else 0.0
            offset += 1
    return row


def build_design_matrix(
    lineups: pd.DataFrame,
    player_order: Sequence[str] | None = None,
    config: PlayerAPMConfig | None = None,
) -> DesignMatrix:
    """Build a deterministic matrix without learning identities from holdouts."""

    cfg = config or PlayerAPMConfig()
    frame = validate_lineups(lineups, cfg)
    players = (
        _sorted_players(frame)
        if player_order is None
        else tuple(str(value) for value in player_order)
    )
    if len(set(players)) != len(players):
        raise ValueError("player_order must contain unique player identifiers")
    player_index = {player: index for index, player in enumerate(players)}
    nuisance_names = _nuisance_feature_names(cfg)
    map_indices = np.arange(len(frame), dtype=np.int64)
    row_blocks: list[np.ndarray] = []
    column_blocks: list[np.ndarray] = []
    data_blocks: list[np.ndarray] = []
    unknown_sets: list[set[str]] = [set() for _ in range(len(frame))]
    for side, sign in (("blue", 1.0 / 5.0), ("red", -1.0 / 5.0)):
        for role in CANONICAL_ROLES:
            column = f"{side}_{role}"
            mapped = frame[column].map(player_index)
            known_mask = mapped.notna().to_numpy()
            row_blocks.append(map_indices[known_mask])
            column_blocks.append(
                mapped.loc[known_mask].to_numpy(dtype=np.int64)
            )
            data_blocks.append(
                np.full(int(known_mask.sum()), sign, dtype=float)
            )
            for row_index in map_indices[~known_mask]:
                unknown_sets[int(row_index)].add(
                    str(frame.iloc[int(row_index)][column])
                )

    offset = len(players)
    if cfg.include_side_term:
        row_blocks.append(map_indices)
        column_blocks.append(
            np.full(len(frame), offset, dtype=np.int64)
        )
        data_blocks.append(np.ones(len(frame), dtype=float))
        offset += 1
    if cfg.league_levels:
        levels = (
            cfg.league_levels[1:]
            if cfg.include_side_term
            else cfg.league_levels
        )
        league_values = frame["league"].to_numpy(dtype=str)
        for level in levels:
            level_mask = league_values == level
            row_blocks.append(map_indices[level_mask])
            column_blocks.append(
                np.full(int(level_mask.sum()), offset, dtype=np.int64)
            )
            data_blocks.append(
                np.ones(int(level_mask.sum()), dtype=float)
            )
            offset += 1

    row_indices = np.concatenate(row_blocks)
    column_indices = np.concatenate(column_blocks)
    data = np.concatenate(data_blocks)
    unknown_by_map = tuple(
        tuple(sorted(values, key=lambda value: (value.casefold(), value)))
        for values in unknown_sets
    )
    names = tuple(f"player:{player}" for player in players) + nuisance_names
    values = sparse.csr_matrix(
        (data, (row_indices, column_indices)),
        shape=(len(frame), len(names)),
        dtype=float,
    )
    values.sum_duplicates()
    values.sort_indices()
    return DesignMatrix(
        values=values,
        outcomes=frame["blue_win"].to_numpy(dtype=float),
        frame=frame,
        player_order=players,
        feature_names=names,
        unknown_players_by_map=unknown_by_map,
    )


def _cohorts_from_design(
    player_values: sparse.spmatrix,
    player_order: Sequence[str],
) -> tuple[ExposureCohort, ...]:
    signatures: dict[
        tuple[tuple[int, ...], tuple[int, ...]], list[str]
    ] = {}
    columns = player_values.tocsc(copy=True)
    columns.sum_duplicates()
    columns.sort_indices()
    for index, player in enumerate(player_order):
        start = int(columns.indptr[index])
        stop = int(columns.indptr[index + 1])
        exposure_rows = tuple(
            int(value) for value in columns.indices[start:stop]
        )
        exposure_signs = tuple(
            int(value)
            for value in np.rint(columns.data[start:stop] * 5.0).astype(
                np.int8
            )
        )
        signature = (exposure_rows, exposure_signs)
        signatures.setdefault(signature, []).append(str(player))
    cohorts: list[ExposureCohort] = []
    for signature, players in signatures.items():
        if len(players) < 2:
            continue
        payload = (
            np.asarray(signature[0], dtype=np.int64).tobytes()
            + np.asarray(signature[1], dtype=np.int8).tobytes()
        )
        cohorts.append(
            ExposureCohort(
                players=tuple(
                    sorted(players, key=lambda value: (value.casefold(), value))
                ),
                n_maps=int(player_values.shape[0]),
                signature_sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    return tuple(
        sorted(
            cohorts,
            key=lambda cohort: (
                -len(cohort.players),
                tuple(value.casefold() for value in cohort.players),
            ),
        )
    )


def detect_identical_exposure_cohorts(
    lineups: pd.DataFrame,
    config: PlayerAPMConfig | None = None,
) -> tuple[ExposureCohort, ...]:
    """Detect exact equal player columns over the supplied maps."""

    design = build_design_matrix(lineups, config=config)
    return _cohorts_from_design(
        design.values[:, : len(design.player_order)], design.player_order
    )


def _matrix_diagnostics(
    values: sparse.csr_matrix,
    player_count: int,
    cohorts: tuple[ExposureCohort, ...],
    config: PlayerAPMConfig,
) -> tuple[DesignDiagnostics, np.ndarray | None]:
    def summarize(
        matrix: sparse.csr_matrix,
        *,
        need_null_space: bool,
    ) -> tuple[
        int,
        int,
        float,
        float,
        float,
        tuple[float, ...],
        str,
        bool,
        np.ndarray | None,
    ]:
        exact = (
            matrix.shape[1] <= config.diagnostic_exact_max_columns
            and matrix.shape[0] * matrix.shape[1]
            <= config.diagnostic_exact_max_cells
        )
        null_space: np.ndarray | None = None
        if exact:
            dense = matrix.toarray()
            if need_null_space:
                _, singular, vh = np.linalg.svd(
                    dense, full_matrices=True
                )
            else:
                singular = np.linalg.svd(
                    dense, compute_uv=False
                )
                vh = None
            method = "exact_dense_svd_selected_model_only"
        else:
            components = min(
                config.diagnostic_svd_components,
                max(min(matrix.shape) - 1, 0),
            )
            if components < 1:
                singular = np.asarray(
                    [float(sparse.linalg.norm(matrix))], dtype=float
                )
            else:
                singular = svds(
                    matrix,
                    k=components,
                    which="LM",
                    return_singular_vectors=False,
                    solver="arpack",
                    v0=np.linspace(
                        1.0, 2.0, num=min(matrix.shape), dtype=float
                    ),
                )
                singular = np.sort(singular)[::-1]
            vh = None
            method = (
                "deterministic_arpack_truncated_svd_approximation_"
                "selected_model_only"
            )
        if singular.size == 0 or singular[0] == 0.0:
            return (
                0,
                matrix.shape[1],
                0.0,
                math.inf if exact else math.nan,
                math.inf,
                tuple(),
                method,
                not exact,
                (
                    np.eye(matrix.shape[1], dtype=float)
                    if exact and need_null_space
                    else None
                ),
            )
        tolerance = config.rank_rtol * max(matrix.shape) * float(singular[0])
        positive = singular[singular > tolerance]
        rank = int(len(positive))
        nullity = int(matrix.shape[1] - rank)
        mass = singular / singular.sum()
        effective = float(np.exp(-np.sum(mass[mass > 0] * np.log(mass[mass > 0]))))
        nonzero_condition = (
            float(positive[0] / positive[-1]) if len(positive) else math.inf
        )
        condition = (
            (
                nonzero_condition
                if rank == matrix.shape[1]
                else math.inf
            )
            if exact
            else math.nan
        )
        if exact and need_null_space and vh is not None:
            null_space = vh[rank:, :].T.copy()
        return (
            rank,
            nullity,
            effective,
            condition,
            nonzero_condition,
            tuple(float(value) for value in singular),
            method,
            not exact,
            null_space,
        )

    (
        rank,
        nullity,
        effective,
        condition,
        nonzero_condition,
        singular,
        method,
        approximate,
        null_space_basis,
    ) = summarize(values, need_null_space=True)
    (
        player_rank,
        player_nullity,
        player_effective,
        player_condition,
        player_nonzero_condition,
        player_singular,
        player_method,
        player_approximate,
        _,
    ) = summarize(
        values[:, :player_count].tocsr(), need_null_space=False
    )
    warning_parts: list[str] = []
    if player_nullity and not player_approximate:
        warning_parts.append(
            f"player design is rank deficient ({player_rank}/{player_count})"
        )
    elif player_approximate:
        warning_parts.append(
            "player-design rank deficiency is not decided by the truncated "
            "spectrum; reported rank is only a lower bound"
        )
    if cohorts:
        warning_parts.append(
            f"{len(cohorts)} exact identical-exposure cohort(s) are not "
            "individually identified"
        )
    if not warning_parts:
        warning_parts.append(
            "no exact duplicate columns detected; weak near-collinearity may remain"
        )
    diagnostics_approximate = approximate or player_approximate
    if diagnostics_approximate:
        warning_parts.append(
            "rank/effective-rank are truncated-spectrum lower bounds and "
            "full condition number is not estimated"
        )
    diagnostics = DesignDiagnostics(
        n_maps=int(values.shape[0]),
        n_columns=int(values.shape[1]),
        rank=rank,
        nullity=nullity,
        effective_rank=effective,
        condition_number=condition,
        nonzero_condition_number=nonzero_condition,
        n_player_columns=player_count,
        player_rank=player_rank,
        player_nullity=player_nullity,
        player_effective_rank=player_effective,
        player_condition_number=player_condition,
        player_nonzero_condition_number=player_nonzero_condition,
        singular_values=singular,
        player_singular_values=player_singular,
        identical_exposure_cohorts=cohorts,
        warning="; ".join(warning_parts),
        diagnostic_method=method,
        player_diagnostic_method=player_method,
        diagnostics_approximate=diagnostics_approximate,
        rank_interpretation=(
            "exact numerical rank"
            if not diagnostics_approximate
            else (
                "rank is a lower bound from retained ARPACK singular "
                "components; nullity is the corresponding upper bound"
            )
        ),
        condition_interpretation=(
            "exact; infinity denotes numerical rank deficiency"
            if not diagnostics_approximate
            else (
                "full condition number intentionally not estimated; "
                "nonzero_condition_number is a retained-spectrum lower bound"
            )
        ),
    )
    return diagnostics, null_space_basis


def _player_penalty_multipliers(
    frame: pd.DataFrame,
    player_order: Sequence[str],
    config: PlayerAPMConfig,
) -> np.ndarray:
    player_index = {
        player: index for index, player in enumerate(player_order)
    }
    counts = np.zeros(
        (len(player_order), len(CANONICAL_ROLES)), dtype=float
    )
    for role_index, role in enumerate(CANONICAL_ROLES):
        observed = pd.concat(
            [frame[f"blue_{role}"], frame[f"red_{role}"]],
            ignore_index=True,
        ).value_counts(sort=False)
        known = observed.index.to_series().map(player_index).notna()
        known_players = observed.index[known.to_numpy()]
        indices = np.asarray(
            [player_index[str(player)] for player in known_players],
            dtype=np.int64,
        )
        counts[indices, role_index] = observed.loc[
            known_players
        ].to_numpy(dtype=float)
    totals = counts.sum(axis=1)
    if np.any(totals <= 0.0):
        player = player_order[int(np.flatnonzero(totals <= 0.0)[0])]
        raise RuntimeError(f"training player {player!r} has no exposure")
    role_multipliers = np.asarray(
        [
            dict(config.role_l2_multipliers)[role]
            for role in CANONICAL_ROLES
        ],
        dtype=float,
    )
    weighted_counts = np.einsum(
        "ij,j->i", counts, role_multipliers, optimize=True
    )
    return weighted_counts / totals


def _fit_logistic(
    values: sparse.csr_matrix,
    outcomes: np.ndarray,
    penalty: np.ndarray,
    config: PlayerAPMConfig,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        safe_beta = np.clip(
            np.nan_to_num(beta, nan=0.0, posinf=20.0, neginf=-20.0),
            -20.0,
            20.0,
        )
        eta = np.asarray(values @ safe_beta, dtype=float).ravel()
        probability = expit(eta)
        objective_value = float(
            np.sum(np.logaddexp(0.0, eta) - outcomes * eta)
            + 0.5 * np.dot(penalty, safe_beta * safe_beta)
        )
        gradient = (
            np.asarray(
                values.T @ (probability - outcomes),
                dtype=float,
            ).ravel()
            + penalty * safe_beta
        )
        return objective_value, gradient

    result = minimize(
        lambda beta: objective(beta)[0],
        np.zeros(values.shape[1], dtype=float),
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        bounds=[(-20.0, 20.0)] * values.shape[1],
        options={
            "maxiter": config.max_iter,
            "ftol": config.optimizer_tolerance,
            "gtol": config.optimizer_tolerance,
            "maxls": 50,
        },
    )
    if not result.success:
        raise RuntimeError(
            "player APM candidate optimization failed: "
            f"{result.status} {result.message}"
        )
    coefficients = np.clip(
        np.nan_to_num(result.x, nan=0.0, posinf=20.0, neginf=-20.0),
        -20.0,
        20.0,
    )
    fitted_probability = expit(
        np.asarray(values @ coefficients, dtype=float).ravel()
    )
    weights = fitted_probability * (1.0 - fitted_probability)
    return (
        coefficients,
        weights,
        float(result.fun),
        int(result.nit),
    )


@dataclass(frozen=True)
class PlayerAPMCandidate:
    """A fitted, explicitly non-production player-lineup candidate."""

    config: PlayerAPMConfig
    player_order: tuple[str, ...]
    feature_names: tuple[str, ...]
    coefficients: np.ndarray
    penalty: np.ndarray
    player_l2: float
    nuisance_l2: float
    diagnostics: DesignDiagnostics | None
    fitted_through: pd.Timestamp
    training_game_ids: tuple[str, ...]
    objective_value: float
    optimizer_iterations: int
    null_space_basis: np.ndarray | None = field(repr=False)
    _design_values: sparse.csr_matrix | None = field(repr=False)
    _fitted_weights: np.ndarray | None = field(repr=False)
    postfit_computed: bool
    full_covariance_formed: bool = False
    model_kind: str = MODEL_KIND
    promotion_status: str = PROMOTION_STATUS
    estimand: str = (
        "regularized decomposition of pre-map blue win log-odds from exact "
        "five-player lineups, conditional on predeclared nuisance terms"
    )
    limitation: str = (
        "team outcomes generally do not identify individual teammate effects; "
        "ridge estimates and Laplace uncertainty are prior-conditioned"
    )

    @property
    def covariance(self) -> np.ndarray:
        """Refuse an unconditional dense inverse.

        Kept as an explicit compatibility guard: callers must state a bounded
        player set through :meth:`covariance_for_players`.
        """

        raise RuntimeError(
            "full covariance is intentionally not formed; use "
            "covariance_for_players() or contrast()"
        )

    def _player_index(self, player: str) -> int:
        try:
            return self.player_order.index(player)
        except ValueError as exc:
            raise KeyError(
                f"player {player!r} was not observed in the fit window"
            ) from exc

    def predict_logit(self, lineups: pd.DataFrame) -> np.ndarray:
        design = build_design_matrix(
            lineups, player_order=self.player_order, config=self.config
        )
        return np.asarray(
            design.values @ self.coefficients, dtype=float
        ).ravel()

    def predict_proba(self, lineups: pd.DataFrame) -> np.ndarray:
        probability = expit(self.predict_logit(lineups))
        return np.clip(
            probability,
            self.config.probability_clip,
            1.0 - self.config.probability_clip,
        )

    def player_effect(self, player: str) -> float:
        return float(self.coefficients[self._player_index(player)])

    def covariance_for_players(
        self, players: Sequence[str]
    ) -> PlayerCovariance:
        requested = tuple(str(value) for value in players)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("players must be a non-empty unique sequence")
        if len(requested) > self.config.max_uncertainty_players:
            raise ValueError(
                "requested covariance exceeds max_uncertainty_players="
                f"{self.config.max_uncertainty_players}"
            )
        if (
            not self.postfit_computed
            or self._design_values is None
            or self._fitted_weights is None
            or self.diagnostics is None
        ):
            raise RuntimeError(
                "post-fit uncertainty was skipped for this validation-grid fit"
            )
        indices = np.asarray(
            [self._player_index(player) for player in requested], dtype=int
        )
        covariance = self._requested_inverse_submatrix(indices)
        estimates = self.coefficients[indices].copy()
        standard = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        denominator = np.outer(standard, standard)
        correlation = np.divide(
            covariance,
            denominator,
            out=np.zeros_like(covariance),
            where=denominator > 0.0,
        )
        np.fill_diagonal(correlation, 1.0)
        identified_levels: list[bool] = []
        identifiability_methods: set[str] = set()
        for index in indices:
            vector = np.zeros(len(self.feature_names), dtype=float)
            vector[index] = 1.0
            identified, method = self._is_data_identified(vector)
            identified_levels.append(identified)
            identifiability_methods.add(method)
        requested_set = set(requested)
        cohorts = tuple(
            cohort
            for cohort in self.diagnostics.identical_exposure_cohorts
            if requested_set.intersection(cohort.players)
        )
        return PlayerCovariance(
            players=requested,
            estimates=estimates,
            covariance=covariance,
            correlation=correlation,
            level_data_identified=tuple(identified_levels),
            identical_exposure_cohorts=cohorts,
            identifiability_method="+".join(
                sorted(identifiability_methods)
            ),
        )

    def _requested_inverse_submatrix(
        self, indices: np.ndarray
    ) -> np.ndarray:
        if self._design_values is None or self._fitted_weights is None:
            raise RuntimeError("selected-model Hessian inputs are unavailable")
        design = self._design_values
        fitted_weights = self._fitted_weights
        dimension = design.shape[1]

        def hessian_product(vector: np.ndarray) -> np.ndarray:
            projected = np.asarray(design @ vector, dtype=float).ravel()
            return (
                np.asarray(
                    design.T @ (fitted_weights * projected),
                    dtype=float,
                ).ravel()
                + self.penalty * vector
            )

        operator = LinearOperator(
            (dimension, dimension),
            matvec=hessian_product,
            rmatvec=hessian_product,
            dtype=float,
        )
        squared = design.copy()
        squared.data = squared.data * squared.data
        hessian_diagonal = (
            np.asarray(
                squared.T @ fitted_weights, dtype=float
            ).ravel()
            + self.penalty
        )
        preconditioner = LinearOperator(
            (dimension, dimension),
            matvec=lambda vector: vector / hessian_diagonal,
            rmatvec=lambda vector: vector / hessian_diagonal,
            dtype=float,
        )
        covariance = np.empty((len(indices), len(indices)), dtype=float)
        for column, coefficient_index in enumerate(indices):
            unit = np.zeros(dimension, dtype=float)
            unit[int(coefficient_index)] = 1.0
            solution, info = cg(
                operator,
                unit,
                M=preconditioner,
                rtol=self.config.covariance_rcond,
                atol=0.0,
                maxiter=self.config.uncertainty_cg_maxiter,
            )
            if info != 0 or not np.isfinite(solution).all():
                raise RuntimeError(
                    "bounded Hessian solve failed for requested covariance "
                    f"column {column} (solver info={info})"
                )
            covariance[:, column] = solution[indices]
        return 0.5 * (covariance + covariance.T)

    def _is_data_identified(
        self, vector: np.ndarray
    ) -> tuple[bool, str]:
        threshold = self.config.rank_rtol * max(
            1.0, float(np.linalg.norm(vector))
        )
        if self.null_space_basis is not None:
            if self.null_space_basis.shape[1] == 0:
                return True, "exact_dense_null_space_selected_model_only"
            projection = self.null_space_basis.T @ vector
            return (
                bool(np.linalg.norm(projection) <= threshold),
                "exact_dense_null_space_selected_model_only",
            )
        if self._design_values is None:
            raise RuntimeError(
                "post-fit design is unavailable for identifiability"
            )
        result = lsmr(
            self._design_values.T.tocsr(),
            vector,
            atol=self.config.rank_rtol,
            btol=self.config.rank_rtol,
            maxiter=self.config.identifiability_lsmr_maxiter,
        )
        residual_norm = float(result[3])
        return (
            residual_norm <= threshold,
            "sparse_lsmr_row_space_approximation_selected_model_only",
        )

    def contrast(
        self,
        weights: Mapping[str, float],
        confidence: float = 0.95,
    ) -> PlayerContrast:
        """Return a covariance-aware player contrast and identifiability flag."""

        if not weights:
            raise ValueError("contrast weights must be non-empty")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between zero and one")
        players = tuple(str(player) for player in weights)
        numeric_weights = np.asarray(
            [float(weights[player]) for player in players], dtype=float
        )
        if not np.isfinite(numeric_weights).all() or np.allclose(
            numeric_weights, 0.0
        ):
            raise ValueError("contrast weights must be finite and non-zero")
        joint = self.covariance_for_players(players)
        full_vector = np.zeros(len(self.feature_names), dtype=float)
        indices = [self._player_index(player) for player in players]
        full_vector[indices] = numeric_weights
        variance = max(
            float(numeric_weights @ joint.covariance @ numeric_weights), 0.0
        )
        standard_error = math.sqrt(variance)
        estimate = float(numeric_weights @ joint.estimates)
        critical = float(norm.ppf(0.5 + confidence / 2.0))
        data_identified, identifiability_method = (
            self._is_data_identified(full_vector)
        )

        identical_confounded = False
        weight_by_player = dict(zip(players, numeric_weights))
        for cohort in self.diagnostics.identical_exposure_cohorts:
            cohort_weights = [
                weight_by_player[player]
                for player in cohort.players
                if player in weight_by_player
            ]
            if cohort_weights and (
                len(cohort_weights) < len(cohort.players)
                or not np.allclose(cohort_weights, cohort_weights[0])
            ):
                identical_confounded = True
                break
        if identical_confounded:
            data_identified = False
        warning = (
            ""
            if data_identified
            else (
                "contrast is not identified by the unpenalized map design; "
                "its estimate and interval depend on the ridge prior"
            )
        )
        return PlayerContrast(
            players=players,
            weights=numeric_weights,
            estimate=estimate,
            standard_error=standard_error,
            interval_low=estimate - critical * standard_error,
            interval_high=estimate + critical * standard_error,
            confidence=confidence,
            covariance=joint.covariance,
            data_identified=data_identified,
            identical_exposure_confounded=identical_confounded,
            identifiability_method=identifiability_method,
            warning=warning,
        )

    def player_table(self) -> pd.DataFrame:
        """Return candidate-labelled effects with exposure-cohort disclosure."""

        if self.diagnostics is None:
            raise RuntimeError(
                "player table is unavailable for a validation-grid-only fit"
            )
        cohort_by_player: dict[str, ExposureCohort] = {}
        for cohort in self.diagnostics.identical_exposure_cohorts:
            for player in cohort.players:
                cohort_by_player[player] = cohort
        rows: list[dict[str, Any]] = []
        for index, player in enumerate(self.player_order):
            cohort = cohort_by_player.get(player)
            rows.append(
                {
                    "player": player,
                    "candidate_effect_logit": float(self.coefficients[index]),
                    "identical_exposure_cohort": (
                        list(cohort.players) if cohort is not None else [player]
                    ),
                    "separately_identified_from_exact_cohort": cohort is None,
                    "uncertainty": (
                        "request joint covariance or a covariance-aware contrast"
                    ),
                    "model_kind": self.model_kind,
                    "promotion_status": self.promotion_status,
                }
            )
        return pd.DataFrame(rows)


def fit_player_apm_candidate(
    lineups: pd.DataFrame,
    *,
    player_l2: float,
    nuisance_l2: float,
    config: PlayerAPMConfig | None = None,
    compute_postfit: bool = True,
) -> PlayerAPMCandidate:
    """Fit a research candidate with explicitly supplied ridge strengths."""

    cfg = config or PlayerAPMConfig()
    if not math.isfinite(player_l2) or player_l2 <= 0.0:
        raise ValueError("player_l2 must be finite and positive")
    if not math.isfinite(nuisance_l2) or nuisance_l2 <= 0.0:
        raise ValueError("nuisance_l2 must be finite and positive")
    design = build_design_matrix(lineups, config=cfg)
    multipliers = _player_penalty_multipliers(
        design.frame, design.player_order, cfg
    )
    return _fit_candidate_from_design(
        design,
        multipliers,
        player_l2=float(player_l2),
        nuisance_l2=float(nuisance_l2),
        config=cfg,
        compute_postfit=compute_postfit,
    )


def _fit_candidate_from_design(
    design: DesignMatrix,
    player_multipliers: np.ndarray,
    *,
    player_l2: float,
    nuisance_l2: float,
    config: PlayerAPMConfig,
    compute_postfit: bool,
) -> PlayerAPMCandidate:
    player_count = len(design.player_order)
    penalty = np.concatenate(
        [
            player_l2 * player_multipliers,
            np.full(
                design.values.shape[1] - player_count,
                nuisance_l2,
                dtype=float,
            ),
        ]
    )
    coefficients, fitted_weights, objective_value, iterations = _fit_logistic(
        design.values, design.outcomes, penalty, config
    )
    diagnostics: DesignDiagnostics | None = None
    null_space_basis: np.ndarray | None = None
    if compute_postfit:
        cohorts = _cohorts_from_design(
            design.values[:, :player_count], design.player_order
        )
        diagnostics, null_space_basis = _matrix_diagnostics(
            design.values, player_count, cohorts, config
        )
    return PlayerAPMCandidate(
        config=config,
        player_order=design.player_order,
        feature_names=design.feature_names,
        coefficients=coefficients,
        penalty=penalty,
        player_l2=player_l2,
        nuisance_l2=nuisance_l2,
        diagnostics=diagnostics,
        fitted_through=pd.Timestamp(design.frame["date"].max()),
        training_game_ids=tuple(design.frame["game_id"].astype(str)),
        objective_value=objective_value,
        optimizer_iterations=iterations,
        null_space_basis=null_space_basis,
        _design_values=design.values if compute_postfit else None,
        _fitted_weights=fitted_weights if compute_postfit else None,
        postfit_computed=compute_postfit,
    )


def binary_prediction_metrics(
    outcomes: Sequence[float] | np.ndarray,
    probabilities: Sequence[float] | np.ndarray,
    *,
    probability_clip: float = 1e-9,
) -> dict[str, float]:
    """Compute deterministic proper scores for binary predictions."""

    y = np.asarray(outcomes, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("outcomes and probabilities must be equal non-empty vectors")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("outcomes must be binary")
    if not np.isfinite(p).all():
        raise ValueError("probabilities must be finite")
    clipped = np.clip(p, probability_clip, 1.0 - probability_clip)
    return {
        "n_maps": float(len(y)),
        "brier": float(np.mean((clipped - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(clipped) + (1.0 - y) * np.log1p(-clipped))
        ),
    }


def select_player_apm_candidate(
    train_lineups: pd.DataFrame,
    validation_lineups: pd.DataFrame,
    config: PlayerAPMConfig | None = None,
) -> CandidateSelection:
    """Select all shrinkage hyperparameters on chronological validation only."""

    cfg = config or PlayerAPMConfig()
    train = validate_lineups(train_lineups, cfg)
    validation = validate_lineups(validation_lineups, cfg)
    overlap = set(train["game_id"]).intersection(validation["game_id"])
    if overlap:
        raise ValueError(
            f"train and validation game IDs overlap: {sorted(overlap)[:5]}"
        )
    if train["date"].max() >= validation["date"].min():
        raise ValueError(
            "chronological validation must begin strictly after training"
        )

    nuisance_values = (
        cfg.nuisance_l2_grid
        if _nuisance_feature_names(cfg)
        else (cfg.nuisance_l2_grid[0],)
    )
    train_design = build_design_matrix(train, config=cfg)
    validation_design = build_design_matrix(
        validation,
        player_order=train_design.player_order,
        config=cfg,
    )
    player_multipliers = _player_penalty_multipliers(
        train_design.frame, train_design.player_order, cfg
    )
    ledger_rows: list[dict[str, Any]] = []
    for player_l2 in cfg.player_l2_grid:
        for nuisance_l2 in nuisance_values:
            model = _fit_candidate_from_design(
                train_design,
                player_multipliers,
                player_l2=float(player_l2),
                nuisance_l2=float(nuisance_l2),
                config=cfg,
                compute_postfit=False,
            )
            probability = np.clip(
                expit(
                    np.asarray(
                        validation_design.values @ model.coefficients,
                        dtype=float,
                    ).ravel()
                ),
                cfg.probability_clip,
                1.0 - cfg.probability_clip,
            )
            metrics = binary_prediction_metrics(
                validation_design.outcomes,
                probability,
                probability_clip=cfg.probability_clip,
            )
            key = (float(player_l2), float(nuisance_l2))
            ledger_rows.append(
                {
                    "player_l2": key[0],
                    "nuisance_l2": key[1],
                    "validation_brier": metrics["brier"],
                    "validation_log_loss": metrics["log_loss"],
                    "n_validation_maps": metrics["n_maps"],
                    "postfit_diagnostics_computed": False,
                    "full_covariance_formed": False,
                    "design_format": "scipy_sparse_csr",
                }
            )
    ledger = pd.DataFrame(ledger_rows)
    # Prefer stronger shrinkage when validation scores are numerically tied.
    ordered = ledger.sort_values(
        [
            f"validation_{cfg.selection_metric}",
            "validation_brier",
            "validation_log_loss",
            "player_l2",
            "nuisance_l2",
        ],
        ascending=[True, True, True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    selected_row = ordered.iloc[0]
    selected_key = (
        float(selected_row["player_l2"]),
        float(selected_row["nuisance_l2"]),
    )
    selected_metrics = {
        "n_maps": float(selected_row["n_validation_maps"]),
        "brier": float(selected_row["validation_brier"]),
        "log_loss": float(selected_row["validation_log_loss"]),
    }
    selected_model = _fit_candidate_from_design(
        train_design,
        player_multipliers,
        player_l2=selected_key[0],
        nuisance_l2=selected_key[1],
        config=cfg,
        compute_postfit=True,
    )
    return CandidateSelection(
        model=selected_model,
        player_l2=selected_key[0],
        nuisance_l2=selected_key[1],
        validation_metrics=selected_metrics,
        candidate_ledger=ordered,
        train_game_ids=tuple(train["game_id"].astype(str)),
        validation_game_ids=tuple(validation["game_id"].astype(str)),
    )


def _prediction_ledger_block(
    model: PlayerAPMCandidate,
    lineups: pd.DataFrame,
    *,
    split: str,
    prediction_kind: str,
) -> pd.DataFrame:
    design = build_design_matrix(
        lineups, player_order=model.player_order, config=model.config
    )
    probability = np.clip(
        expit(
            np.asarray(
                design.values @ model.coefficients, dtype=float
            ).ravel()
        ),
        model.config.probability_clip,
        1.0 - model.config.probability_clip,
    )
    outcome = design.outcomes
    clipped = np.clip(
        probability,
        model.config.probability_clip,
        1.0 - model.config.probability_clip,
    )
    return pd.DataFrame(
        {
            "game_id": design.frame["game_id"].astype(str),
            "date": design.frame["date"],
            "split": split,
            "blue_win": outcome,
            "predicted_blue_win": probability,
            "brier": (probability - outcome) ** 2,
            "log_loss": -(
                outcome * np.log(clipped)
                + (1.0 - outcome) * np.log1p(-clipped)
            ),
            "prediction_kind": prediction_kind,
            "fit_through": model.fitted_through,
            "player_l2": model.player_l2,
            "nuisance_l2": model.nuisance_l2,
            "unknown_players": [
                list(values) for values in design.unknown_players_by_map
            ],
            "model_kind": model.model_kind,
            "promotion_status": model.promotion_status,
        }
    )


def chronological_player_apm_evaluation(
    lineups: pd.DataFrame,
    *,
    train_end: str | pd.Timestamp,
    validation_end: str | pd.Timestamp,
    config: PlayerAPMConfig | None = None,
) -> ChronologicalPlayerAPMEvaluation:
    """Build a leakage-safe chronological prediction ledger and proper scores.

    Candidate penalties are selected from train-to-validation predictions.
    Validation outcomes are then permitted in a train+validation refit used for
    the strictly later test maps.  Test outcomes never affect features,
    hyperparameters, coefficients, or predictions.
    """

    cfg = config or PlayerAPMConfig()
    frame = validate_lineups(lineups, cfg)
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)
    if train_cutoff.tzinfo is None:
        train_cutoff = train_cutoff.tz_localize("UTC")
    else:
        train_cutoff = train_cutoff.tz_convert("UTC")
    if validation_cutoff.tzinfo is None:
        validation_cutoff = validation_cutoff.tz_localize("UTC")
    else:
        validation_cutoff = validation_cutoff.tz_convert("UTC")
    if train_cutoff >= validation_cutoff:
        raise ValueError("train_end must be strictly before validation_end")

    train = frame[frame["date"] <= train_cutoff].copy()
    validation = frame[
        (frame["date"] > train_cutoff)
        & (frame["date"] <= validation_cutoff)
    ].copy()
    test = frame[frame["date"] > validation_cutoff].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "chronological evaluation requires non-empty train, validation, and test"
        )

    selection = select_player_apm_candidate(train, validation, cfg)
    train_validation = pd.concat(
        [train, validation], ignore_index=True
    ).sort_values(["date", "game_id"], kind="mergesort")
    test_model = fit_player_apm_candidate(
        train_validation,
        player_l2=selection.player_l2,
        nuisance_l2=selection.nuisance_l2,
        config=cfg,
    )
    ledger = pd.concat(
        [
            _prediction_ledger_block(
                selection.model,
                train,
                split="train",
                prediction_kind="in_sample_fit",
            ),
            _prediction_ledger_block(
                selection.model,
                validation,
                split="validation",
                prediction_kind="hyperparameter_selection_holdout",
            ),
            _prediction_ledger_block(
                test_model,
                test,
                split="test",
                prediction_kind="untouched_chronological_holdout",
            ),
        ],
        ignore_index=True,
    )
    metrics: dict[str, Mapping[str, float]] = {}
    for split, group in ledger.groupby("split", sort=False):
        metrics[str(split)] = binary_prediction_metrics(
            group["blue_win"].to_numpy(float),
            group["predicted_blue_win"].to_numpy(float),
            probability_clip=cfg.probability_clip,
        )
    return ChronologicalPlayerAPMEvaluation(
        ledger=ledger,
        metrics=metrics,
        selection=selection,
        test_model=test_model,
        train_end=train_cutoff,
        validation_end=validation_cutoff,
    )


# Short aliases for exploratory notebooks; both retain explicit candidate names
# in their returned objects and ledgers.
fit_player_apm = fit_player_apm_candidate
evaluate_player_apm_chronologically = chronological_player_apm_evaluation
