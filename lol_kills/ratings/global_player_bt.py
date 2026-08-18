"""One connected, regularized player results rating.

The fit uses only completed map results and verified ten-player lineups. Each
player has one coefficient across every league, team, and competition tier.
Roster transfers and cross-circuit matches connect domestic pools. Competition
tier is never used as a rating bonus or penalty.

Map results alone cannot separate two players who never appear apart: their
design columns are identical, so a shrink-to-zero ridge hands them identical
coefficients. The ridge therefore shrinks toward a per-player performance
anchor instead of toward zero. The anchor is built from role-normalized
contribution metrics, is centered to exactly zero mean, and never adds a
competition-tier level bonus: metrics are z-scored *within* their own
(role, competition tier) pool, so tier labels only choose the comparison
group and cannot lift or lower a player's anchor on their own.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, hstack

from lol_kills.etl.source_keys import canonical_source_game_key


LOGIT_TO_ELO = 400.0 / math.log(10.0)
ROLE_ALIAS = {
    "top": "top",
    "jng": "jng",
    "jungle": "jng",
    "mid": "mid",
    "bot": "bot",
    "adc": "bot",
    "sup": "sup",
    "support": "sup",
    "utility": "sup",
}

# UNFITTED DEVELOPMENT DEFAULT.  These weights have never been fitted, tuned,
# or validated against any outcome; they are a deliberate equal-weight
# placeholder so the anchor stays inspectable while a weight study is pending.
# Do not describe a fit that uses them as a fitted contribution model.
PERFORMANCE_ANCHOR_METRIC_WEIGHTS: dict[str, float] = {
    "cs_per_min": 1.0,
    "gold_per_min": 1.0,
    "gold_share_pct": 1.0,
    "damage_per_min": 1.0,
    "damage_share_pct": 1.0,
    "kda_role_weighted": 1.0,
    "wards_per_min": 1.0,
    "wards_cleared_per_min": 1.0,
}
PERFORMANCE_ANCHOR_WEIGHTS_STATUS = "unfitted_development_default"

# Raw source columns `_contribution_metrics` reads. Any caller that builds the
# rating input by column projection MUST carry these, or the anchor is inert.
# `lol_kills.export.public_pack.PLAYER_CONTRIBUTION_COLUMNS` is pinned as a
# superset of this tuple by test_public_pack_projects_every_anchor_source_column.
PERFORMANCE_ANCHOR_SOURCE_COLUMNS: tuple[str, ...] = (
    "gamelength",
    "totalgold",
    "cspm",
    "dpm",
    "damageshare",
    "kills",
    "deaths",
    "assists",
    "wpm",
    "wcpm",
)
_ANCHOR_ZERO_MEAN_TOLERANCE = 1e-9

# Minimum number of STRICTLY EARLIER observations a (role, competition tier)
# baseline needs before it may normalize a metric.  Below the floor the metric
# stays unavailable (NaN) for that player-map; it is never imputed.
#
# This mirrors ``lol_kills.ratings.player_elo.ATTRIBUTION_MIN_BASELINE_OBS`` and
# must stay equal to it.  It is duplicated rather than imported because
# ``player_elo`` imports this module, so importing back would be circular;
# ``test_anchor_baseline_floor_matches_the_elo_constant`` pins the two together.
ANCHOR_MIN_BASELINE_OBS = 20

# A player needs this many of their own scored maps before the anchor speaks for
# them. Guards the case where one malformed row is a large share of a player's
# record; below the floor the player stays exactly neutral and is counted.
ANCHOR_MIN_PLAYER_MAPS = 5

# Upper bound on the magnitude of any single normalized metric before it enters
# the composite.  The upstream completeness gate at
# ``lol_kills/etl/oe_database.py:548-559`` only checks finiteness, nonnegativity
# and a handful of ratio bounds, so an implausibly large but finite statistic
# survives ingestion.  Without this clip one such value would dominate the
# composite and then the player-level standardization, moving a low-sample
# player by hundreds of Elo.
ANCHOR_METRIC_Z_CLIP = 3.0

# Consistency constants for the ROBUST baseline used by ``_prior_baseline_z``.
#
# The baseline is a median and a median absolute deviation, not a mean and a
# standard deviation.  A mean/std baseline has a breakdown point of 1/n: one
# malformed but ingestible statistic (``cspm = 1e12`` clears the completeness
# gate at ``lol_kills/etl/oe_database.py:548-559``) inflates the pool's mean and
# std enough that every LATER row in that pool reads as roughly -1/sqrt(n)
# standard deviations instead of 0.  That is the classical masking failure, and
# clipping the resulting z cannot repair it because the contamination is in the
# statistic the z is measured AGAINST, not in the z.  The median and the MAD
# both have a 50% breakdown point, so a single row cannot move either.
#
# ``_MAD_TO_SIGMA`` makes the MAD a consistent estimator of sigma under
# normality; ``_IQR_TO_SIGMA`` does the same for the interquartile range.
_MAD_TO_SIGMA = 1.4826
_IQR_TO_SIGMA = 1.349


class GlobalPlayerRatingError(RuntimeError):
    """Raised when the shared player scale cannot pass its release checks."""


@dataclass(frozen=True)
class GlobalPlayerBTConfig:
    l2: float = 2.0
    side_l2: float = 0.01
    prior_rating: float = 1500.0
    holdout_fraction: float = 0.20
    minimum_maps: int = 100
    minimum_connected_share: float = 0.95
    minimum_holdout_gain: float = 0.005
    max_iterations: int = 400
    performance_anchor_scale: float = 0.15
    performance_anchor_enabled: bool = True


class _Components:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            self.size[value] = 1
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        a = self.find(left)
        b = self.find(right)
        if a == b:
            return
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]


def _role(value: Any) -> str | None:
    text = str(value or "").strip().casefold()
    if text in ROLE_ALIAS:
        return ROLE_ALIAS[text]
    for prefix, role in ROLE_ALIAS.items():
        if text.startswith(prefix):
            return role
    return None


def _canonical_game_ids(frame: pd.DataFrame) -> pd.Series:
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        return pd.Series(
            [
                canonical_source_game_key(
                    value,
                    fallback.loc[index] if fallback is not None else None,
                )
                for index, value in frame["game_uid"].items()
            ],
            index=frame.index,
            dtype="string",
        )
    if "gameid" in frame.columns:
        return frame["gameid"].map(canonical_source_game_key).astype("string")
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def _complete_lineups(players: pd.DataFrame) -> dict[str, dict[str, list[tuple[str, str]]]]:
    required = {"side", "position", "playername"}
    if players is None or players.empty or not required.issubset(players.columns):
        return {}
    frame = players.copy()
    frame["_game_id"] = _canonical_game_ids(frame)
    frame["_side"] = frame["side"].astype(str).str.title()
    frame["_role"] = frame["position"].map(_role)
    frame["_player"] = frame["playername"].astype("string").str.strip()
    frame = frame[
        frame["_game_id"].notna()
        & frame["_game_id"].str.strip().ne("")
        & frame["_side"].isin({"Blue", "Red"})
        & frame["_role"].notna()
        & frame["_player"].notna()
        & frame["_player"].ne("")
        & frame["_player"].str.casefold().ne("nan")
    ]
    order = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    output: dict[str, dict[str, list[tuple[str, str]]]] = {}
    for (game_id, side), group in frame.groupby(["_game_id", "_side"], sort=False):
        rows = sorted(
            zip(group["_player"].astype(str), group["_role"].astype(str)),
            key=lambda value: order.get(value[1], 9),
        )
        by_role: dict[str, str] = {}
        for player, role in rows:
            by_role.setdefault(role, player)
        if set(by_role) != set(order) or len(set(by_role.values())) != 5:
            continue
        output.setdefault(str(game_id), {})[str(side)] = [
            (by_role[role], role) for role in order
        ]
    return {
        game_id: sides
        for game_id, sides in output.items()
        if len(sides.get("Blue", [])) == 5 and len(sides.get("Red", [])) == 5
    }


def _model_rows(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    through: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, list[tuple[str, str]]]]]:
    if maps is None or maps.empty:
        return pd.DataFrame(), {}
    frame = maps.copy()
    frame["game_id"] = _canonical_game_ids(frame)
    frame["date"] = pd.to_datetime(frame.get("date"), utc=True, errors="coerce").dt.tz_localize(None)
    frame["result"] = pd.to_numeric(frame.get("y_blue_win"), errors="coerce")
    frame = frame[
        frame["game_id"].notna()
        & frame["game_id"].str.strip().ne("")
        & frame["date"].notna()
        & frame["result"].isin({0, 1})
    ]
    if through is not None:
        cutoff = pd.Timestamp(through)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        frame = frame[frame["date"].le(cutoff)]
    frame = frame.sort_values(["date", "game_id"], kind="stable").drop_duplicates("game_id", keep="last")
    lineups = _complete_lineups(players)
    frame = frame[frame["game_id"].isin(lineups)].reset_index(drop=True)
    return frame, lineups


def _design(
    frame: pd.DataFrame,
    lineups: dict[str, dict[str, list[tuple[str, str]]]],
    names: list[str],
) -> csr_matrix:
    index = {name: position for position, name in enumerate(names)}
    row_index: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_number, game_id in enumerate(frame["game_id"].astype(str)):
        for side, sign in (("Blue", 1.0), ("Red", -1.0)):
            lineup = lineups[game_id][side]
            for player, _role_name in lineup:
                row_index.append(row_number)
                columns.append(index[player])
                values.append(sign / len(lineup))
    return csr_matrix((values, (row_index, columns)), shape=(len(frame), len(names)))


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    """Finite numeric view of one column; an absent column reads as all-NaN."""

    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").astype(float)
    return values.where(np.isfinite(values))


def _robust_block_baseline(
    pool: np.ndarray,
    available: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding median and robust scale for one group's blocks.

    ``pool`` holds a single group's PRESENT values in date order and
    ``available[k]`` is how many of them exist strictly before block ``k``
    begins, so ``pool[:available[k]]`` is exactly that block's baseline sample.

    The scale is the MAD rescaled by ``_MAD_TO_SIGMA``.  Its one known failure
    mode is MAD == 0, which happens whenever more than half the sample ties --
    entirely plausible for wards-per-minute pools where many rows are exactly
    0.  The fallback is the interquartile range rescaled by ``_IQR_TO_SIGMA``,
    which is still robust (25% breakdown) and survives ties the MAD cannot.

    FAIL CLOSED: a sample under the floor, or one where BOTH robust scales are
    zero or non-finite, returns NaN.  Nothing is imputed, no group mean is
    substituted, and nothing is ever divided by zero.
    """

    location = np.full(len(available), np.nan, dtype=float)
    scale = np.full(len(available), np.nan, dtype=float)
    if len(pool) == 0:
        return location, scale
    # ``expanding().median()`` is pandas' C skiplist, and it is bit-identical
    # to ``np.median(pool[:n])`` for every n (pinned by
    # ``test_expanding_median_is_bit_identical_to_numpy_median``).  Only the
    # MAD needs the per-block pass, because its deviations are taken against
    # that block's own median and so cannot be accumulated.
    running_median = pd.Series(pool).expanding().median().to_numpy(dtype=float)
    floor = max(int(min_obs), 1)
    previous = -1
    for position in range(len(available)):
        count = int(available[position])
        if count < floor:
            continue
        if count == previous:
            # A block with no present rows leaves the sample untouched.
            location[position] = location[position - 1]
            scale[position] = scale[position - 1]
            continue
        previous = count
        prefix = pool[:count]
        centre = float(running_median[count - 1])
        if not np.isfinite(centre):
            continue
        deviation = np.abs(prefix - centre)
        spread = _MAD_TO_SIGMA * float(np.median(deviation, overwrite_input=True))
        if not np.isfinite(spread) or spread <= 0.0:
            low, high = np.quantile(prefix, (0.25, 0.75))
            spread = float(high - low) / _IQR_TO_SIGMA
        if not np.isfinite(spread) or spread <= 0.0:
            continue
        location[position] = centre
        scale[position] = spread
    return location, scale


def _prior_baseline_z(
    values: pd.Series,
    group: pd.Series,
    date: pd.Series,
    min_obs: int,
) -> tuple[pd.Series, pd.Series]:
    """Robust z-score against a baseline built only from strictly earlier dates.

    Mirrors ``lol_kills.ratings.player_elo._prior_baseline_z`` (that module
    imports this one, so it cannot be imported back).  The mirror is pinned by
    ``test_prior_baseline_z_matches_the_elo_implementation``; the only
    difference is that this copy also returns the prior observation count so
    the caller can report how many cells the floor withheld.

    The baseline for a row is the expanding MEDIAN and MAD over every row in
    the same ``group`` whose date is strictly before the row's own date, so map
    ``t`` never contributes to its own baseline.  Rows sharing a timestamp form
    one block and cannot see each other.  A baseline thinner than ``min_obs``
    or with a degenerate robust spread yields an unavailable (NaN) z-score.

    Median/MAD rather than mean/std is the whole point: see ``_MAD_TO_SIGMA``.
    A single malformed row cannot move a 50%-breakdown estimator, so it cannot
    poison the baseline that every LATER row in the pool is measured against.
    """

    index = values.index
    x = values.to_numpy(dtype=float)
    present = np.isfinite(x)

    location = np.full(len(index), np.nan, dtype=float)
    scale = np.full(len(index), np.nan, dtype=float)
    prior_count = np.zeros(len(index), dtype=float)

    work = pd.DataFrame(
        {"_g": group.to_numpy(), "_d": date.to_numpy()},
        index=pd.RangeIndex(len(index)),
    )
    work["_x"] = x
    work["_p"] = present
    # A row with no group or no date cannot be placed in the prior ordering, so
    # it is left unavailable rather than scored against a baseline it might
    # belong inside.  This is what ``dropna=True`` did for the block groupby.
    placed = work["_g"].notna().to_numpy() & work["_d"].notna().to_numpy()

    for _key, sub in work[placed].groupby("_g", sort=False):
        if sub.empty:
            continue
        sub = sub.sort_values("_d", kind="mergesort")
        positions = sub.index.to_numpy()
        dates = sub["_d"].to_numpy()
        rows_present = sub["_p"].to_numpy(dtype=bool)
        pool = sub["_x"].to_numpy(dtype=float)[rows_present]

        # One block per distinct timestamp; every row in a block shares the
        # baseline taken as of the last row STRICTLY BEFORE the block starts.
        starts = np.empty(len(sub), dtype=bool)
        starts[0] = True
        starts[1:] = dates[1:] != dates[:-1]
        block_of_row = np.cumsum(starts) - 1
        available = np.concatenate(([0], np.cumsum(rows_present)))[
            np.flatnonzero(starts)
        ]

        block_location, block_scale = _robust_block_baseline(
            pool, available, min_obs
        )
        location[positions] = block_location[block_of_row]
        scale[positions] = block_scale[block_of_row]
        prior_count[positions] = available[block_of_row].astype(float)

    with np.errstate(invalid="ignore", divide="ignore"):
        usable = (
            present
            & (prior_count >= float(min_obs))
            & np.isfinite(location)
            & np.isfinite(scale)
            & (scale > 0.0)
        )
        z = np.where(usable, (x - location) / scale, np.nan)
    return (
        pd.Series(z, index=index, dtype=float),
        pd.Series(prior_count, index=index, dtype=float),
    )


def _map_dates(frame: pd.DataFrame) -> pd.Series:
    """Authoritative map date per canonical game id, taken from the map rows."""

    return pd.Series(
        frame["date"].to_numpy(),
        index=pd.Index(frame["game_id"].astype(str), name="game_id"),
    )


def _contribution_metrics(
    players: pd.DataFrame,
    map_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Per player-map contribution metrics used to build the ridge anchor.

    Every metric is fail-closed: a missing column, a missing value, or an
    impossible denominator yields NaN for that metric on that map. Nothing is
    imputed and no league or role mean is ever substituted.

    ``map_dates`` carries the authoritative map date per canonical game id and
    is what orders the shifted/expanding baselines.  A row whose map has no
    usable date cannot be placed in that order, so it is dropped rather than
    scored against a baseline it might belong inside.
    """

    required = {"side", "position", "playername"}
    if players is None or players.empty or not required.issubset(players.columns):
        return pd.DataFrame()

    frame = pd.DataFrame(index=players.index)
    frame["_game_id"] = _canonical_game_ids(players)
    if map_dates is not None:
        dates = frame["_game_id"].astype(str).map(map_dates)
    elif "date" in players.columns:
        dates = players["date"]
    else:
        dates = pd.Series(pd.NaT, index=players.index)
    frame["_date"] = pd.to_datetime(
        pd.Series(dates, index=players.index), utc=True, errors="coerce"
    ).dt.tz_localize(None)
    frame["_side"] = players["side"].astype(str).str.title()
    frame["_role"] = players["position"].map(_role)
    frame["_player"] = players["playername"].astype("string").str.strip()
    if "competition_tier" in players.columns:
        frame["_tier"] = players["competition_tier"].astype("string").str.strip().str.casefold()
    else:
        frame["_tier"] = pd.Series(pd.NA, index=players.index, dtype="string")
    for source in PERFORMANCE_ANCHOR_SOURCE_COLUMNS:
        frame[f"_raw_{source}"] = _numeric(players, source)

    frame = frame[
        frame["_game_id"].notna()
        & frame["_game_id"].astype(str).str.strip().ne("")
        & frame["_date"].notna()
        & frame["_side"].isin({"Blue", "Red"})
        & frame["_role"].notna()
        & frame["_player"].notna()
        & frame["_player"].ne("")
        & frame["_player"].str.casefold().ne("nan")
    ]
    # One row per player and map so a duplicated feed row cannot double-weight
    # a single performance, and so team totals stay a five-player sum.
    frame = frame.drop_duplicates(["_game_id", "_player"], keep="first")
    if frame.empty:
        return pd.DataFrame()

    minutes = frame["_raw_gamelength"].where(frame["_raw_gamelength"] > 0) / 60.0
    total_gold = frame["_raw_totalgold"].where(frame["_raw_totalgold"] >= 0)
    # A share needs a complete denominator. If any seat on the team is missing
    # its gold, the team total is short and every teammate's share would be
    # silently inflated, so the whole side's share is withheld instead.
    side_keys = [frame["_game_id"], frame["_side"]]
    gold_by_side = total_gold.groupby(side_keys, dropna=False)
    team_gold = gold_by_side.transform("sum", min_count=1)
    team_gold = team_gold.where(
        gold_by_side.transform("count").eq(gold_by_side.transform("size"))
    )
    deaths = frame["_raw_deaths"].where(frame["_raw_deaths"] >= 0)

    frame["cs_per_min"] = frame["_raw_cspm"]
    frame["gold_per_min"] = total_gold / minutes
    frame["gold_share_pct"] = 100.0 * total_gold / team_gold.where(team_gold > 0)
    frame["damage_per_min"] = frame["_raw_dpm"]
    frame["damage_share_pct"] = frame["_raw_damageshare"]
    frame["kda_role_weighted"] = (
        frame["_raw_kills"] + frame["_raw_assists"]
    ) / deaths.clip(lower=1.0)
    frame["wards_per_min"] = frame["_raw_wpm"]
    frame["wards_cleared_per_min"] = frame["_raw_wcpm"]

    keep = ["_game_id", "_date", "_side", "_role", "_player", "_tier", *PERFORMANCE_ANCHOR_METRIC_WEIGHTS]
    metrics = frame[keep].copy()
    for metric in PERFORMANCE_ANCHOR_METRIC_WEIGHTS:
        values = metrics[metric].astype(float)
        metrics[metric] = values.where(np.isfinite(values))
    return metrics


def _role_normalized_composite(
    metrics: pd.DataFrame,
) -> tuple[pd.Series, str, dict[str, Any]]:
    """Weighted mean of within-(role, tier) z-scores for each player-map row.

    Role normalization is what makes the anchor fair: a support's 0.87 cs/min
    is normal for a support and must not read as a bad performance.

    Each metric is normalized against a shifted/expanding baseline over
    STRICTLY EARLIER maps in the same (role, competition tier) pool, so a map
    never contributes to its own baseline and never sees a later map.  A
    baseline with fewer than ``ANCHOR_MIN_BASELINE_OBS`` prior observations
    withholds the metric entirely, and every surviving z-score is clipped to
    +/-``ANCHOR_METRIC_Z_CLIP`` before it enters the composite so no single
    malformed statistic can dominate.
    """

    group_keys = ["_role"]
    normalization = "role"
    if "_tier" in metrics.columns and metrics["_tier"].notna().any():
        group_keys = ["_role", "_tier"]
        normalization = "role+competition_tier"
    # Missing tier stays an explicit bucket instead of collapsing into another
    # pool or being silently discarded.
    group = metrics[group_keys[0]].astype(str)
    for key in group_keys[1:]:
        group = group + "\x1f" + metrics[key].astype(str).fillna("")
    date = metrics["_date"]

    diagnostics: dict[str, Any] = {
        "baseline_min_prior_observations": int(ANCHOR_MIN_BASELINE_OBS),
        "normalized_metric_clip": float(ANCHOR_METRIC_Z_CLIP),
        "metric_cells_present": 0,
        "metric_cells_observed": 0,
        "metric_cells_withheld_below_baseline_floor": 0,
        "metric_cells_withheld_degenerate_baseline": 0,
        "metric_cells_clipped": 0,
        "normalized_metric_min": None,
        "normalized_metric_max": None,
    }

    weighted_sum = pd.Series(0.0, index=metrics.index)
    weight_total = pd.Series(0.0, index=metrics.index)
    observed_low: float | None = None
    observed_high: float | None = None
    for metric, weight in PERFORMANCE_ANCHOR_METRIC_WEIGHTS.items():
        if weight <= 0.0 or metric not in metrics.columns:
            continue
        values = metrics[metric]
        raw_z, prior_obs = _prior_baseline_z(
            values, group, date, ANCHOR_MIN_BASELINE_OBS
        )
        present = values.notna() & np.isfinite(values.astype(float))
        below_floor = present & prior_obs.lt(float(ANCHOR_MIN_BASELINE_OBS))
        withheld = present & raw_z.isna()
        z = raw_z.clip(lower=-ANCHOR_METRIC_Z_CLIP, upper=ANCHOR_METRIC_Z_CLIP)
        observed = z.notna()

        diagnostics["metric_cells_present"] += int(present.sum())
        diagnostics["metric_cells_observed"] += int(observed.sum())
        diagnostics["metric_cells_withheld_below_baseline_floor"] += int(below_floor.sum())
        diagnostics["metric_cells_withheld_degenerate_baseline"] += int(
            (withheld & ~below_floor).sum()
        )
        diagnostics["metric_cells_clipped"] += int(
            (raw_z.abs() > ANCHOR_METRIC_Z_CLIP).sum()
        )
        if observed.any():
            low = float(z[observed].min())
            high = float(z[observed].max())
            observed_low = low if observed_low is None else min(observed_low, low)
            observed_high = high if observed_high is None else max(observed_high, high)

        weighted_sum = weighted_sum + z.where(observed, 0.0) * weight
        weight_total = weight_total + observed.astype(float) * weight

    diagnostics["normalized_metric_min"] = observed_low
    diagnostics["normalized_metric_max"] = observed_high
    composite = weighted_sum / weight_total.where(weight_total > 0)
    return composite.where(np.isfinite(composite)), normalization, diagnostics


def _performance_anchor(
    metrics: pd.DataFrame,
    names: list[str],
    game_ids: set[str],
    cfg: GlobalPlayerBTConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Zero-mean ridge anchor in logit units, one entry per fitted player.

    Returns the anchor, a boolean mask of players that actually received one,
    and the release evidence for the anchor itself.
    """

    anchor = np.zeros(len(names), dtype=float)
    anchored = np.zeros(len(names), dtype=bool)
    evidence: dict[str, Any] = {
        "enabled": bool(cfg.performance_anchor_enabled),
        "scale_logit": float(cfg.performance_anchor_scale),
        "elo_per_contribution_sd": float(LOGIT_TO_ELO * cfg.performance_anchor_scale),
        "metric_weights": dict(PERFORMANCE_ANCHOR_METRIC_WEIGHTS),
        "weights_status": PERFORMANCE_ANCHOR_WEIGHTS_STATUS,
        "normalization": None,
        "player_map_rows_used": 0,
        "players_anchored": 0,
        "players_without_metrics": len(names),
        "anchor_mean_logit": 0.0,
        "anchor_sd_logit": 0.0,
        "baseline_min_prior_observations": int(ANCHOR_MIN_BASELINE_OBS),
        "normalized_metric_clip": float(ANCHOR_METRIC_Z_CLIP),
        "metric_cells_present": 0,
        "metric_cells_observed": 0,
        "metric_cells_withheld_below_baseline_floor": 0,
        "metric_cells_withheld_degenerate_baseline": 0,
        "metric_cells_clipped": 0,
        "normalized_metric_min": None,
        "normalized_metric_max": None,
    }
    if not cfg.performance_anchor_enabled or metrics is None or metrics.empty or not names:
        return anchor, anchored, evidence

    wanted = set(names)
    scoped = metrics[
        metrics["_game_id"].astype(str).isin(game_ids)
        & metrics["_player"].astype(str).isin(wanted)
    ]
    if scoped.empty:
        return anchor, anchored, evidence

    composite, normalization, diagnostics = _role_normalized_composite(scoped)
    evidence["normalization"] = normalization
    evidence.update(diagnostics)
    evidence["player_map_rows_used"] = int(composite.notna().sum())
    # Median, not mean: clipping bounds a single row to +/-ANCHOR_METRIC_Z_CLIP,
    # but the mean of a low-sample player is still dominated by one extreme row,
    # so a malformed feed value could move that player the full anchor range.
    # The median makes a lone outlier unable to carry the player's composite.
    grouped = composite.groupby(scoped["_player"].astype(str))
    per_player = grouped.median()
    # A player must also have enough of their OWN maps before the anchor speaks
    # for them. Below the floor the player stays exactly neutral and is counted.
    per_player_maps = grouped.count()
    thin = per_player_maps < ANCHOR_MIN_PLAYER_MAPS
    evidence["players_withheld_below_player_map_floor"] = int(thin.sum())
    per_player = per_player.where(~thin)
    aligned = per_player.reindex(names).to_numpy(dtype=float)

    observed = np.isfinite(aligned)
    # Fewer than two anchored players leaves the spread undefined, so the whole
    # anchor stays at zero rather than inventing a scale.
    if int(observed.sum()) < 2:
        return anchor, anchored, evidence
    sample = aligned[observed]
    spread = float(np.std(sample, ddof=0))
    if not np.isfinite(spread) or spread <= 0.0:
        return anchor, anchored, evidence

    standardized = (sample - float(sample.mean())) / spread
    # Two centering passes so the residual float drift of the mean lands at
    # machine zero: the global rating scale must not move at all.
    standardized = standardized - float(standardized.mean())
    standardized = standardized - float(standardized.mean())

    anchor[observed] = cfg.performance_anchor_scale * standardized
    anchored = observed
    drift = float(anchor.sum())
    if abs(drift) > _ANCHOR_ZERO_MEAN_TOLERANCE * max(len(names), 1):
        raise GlobalPlayerRatingError(
            f"performance anchor is not zero-mean: total drift {drift:.3e}"
        )
    evidence["players_anchored"] = int(observed.sum())
    evidence["players_without_metrics"] = int(len(names) - observed.sum())
    evidence["anchor_mean_logit"] = float(anchor.mean())
    evidence["anchor_sd_logit"] = float(np.std(anchor[observed], ddof=0))
    return anchor, anchored, evidence


def _fit(
    design: csr_matrix,
    outcome: np.ndarray,
    cfg: GlobalPlayerBTConfig,
    *,
    anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    side = csr_matrix(np.ones((design.shape[0], 1), dtype=float))
    matrix = hstack([design, side], format="csr")
    penalty = np.concatenate(
        [
            np.full(design.shape[1], cfg.l2, dtype=float),
            np.asarray([cfg.side_l2], dtype=float),
        ]
    )
    # The side term is always anchored at zero; a zero player anchor reproduces
    # the plain shrink-to-zero ridge exactly.
    if anchor is None:
        player_anchor = np.zeros(design.shape[1], dtype=float)
    else:
        player_anchor = np.asarray(anchor, dtype=float).reshape(-1)
        if player_anchor.shape[0] != design.shape[1]:
            raise GlobalPlayerRatingError(
                f"anchor has {player_anchor.shape[0]} entries for {design.shape[1]} players"
            )
        if not np.isfinite(player_anchor).all():
            raise GlobalPlayerRatingError("anchor contains non-finite entries")
    anchor_vector = np.concatenate([player_anchor, np.zeros(1, dtype=float)])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = np.asarray(matrix @ parameters).reshape(-1)
        residual = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35))) - outcome
        loss = float(np.logaddexp(0.0, logits).sum() - np.dot(outcome, logits))
        delta = parameters - anchor_vector
        loss += 0.5 * float(np.dot(penalty, delta**2))
        gradient = np.asarray(matrix.T @ residual).reshape(-1) + penalty * delta
        return loss, gradient

    fitted = minimize(
        objective,
        np.zeros(matrix.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": cfg.max_iterations, "ftol": 1e-10, "gtol": 1e-6},
    )
    if not fitted.success:
        raise GlobalPlayerRatingError(f"global player fit failed: {fitted.message}")
    return fitted.x[:-1], float(fitted.x[-1])


def _log_loss(outcome: np.ndarray, logits: np.ndarray) -> float:
    return float(np.mean(np.logaddexp(0.0, logits) - outcome * logits))


def _component_summary(
    frame: pd.DataFrame,
    lineups: dict[str, dict[str, list[tuple[str, str]]]],
) -> tuple[dict[str, str], dict[str, int], str, float]:
    components = _Components()
    for game_id in frame["game_id"].astype(str):
        names = [player for side in ("Blue", "Red") for player, _ in lineups[game_id][side]]
        for player in names:
            components.find(player)
        for player in names[1:]:
            components.union(names[0], player)
    roots = {player: components.find(player) for player in components.parent}
    sizes: dict[str, int] = {}
    for root in roots.values():
        sizes[root] = sizes.get(root, 0) + 1
    largest = max(sizes, key=sizes.get)
    return roots, sizes, largest, sizes[largest] / max(len(roots), 1)


def fit_global_player_bt(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: GlobalPlayerBTConfig | None = None,
    *,
    through: pd.Timestamp | None = None,
    validate: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit one player-results scale and return its release evidence."""

    cfg = cfg or GlobalPlayerBTConfig()
    frame, lineups = _model_rows(maps, players, through=through)
    if len(frame) < cfg.minimum_maps:
        raise GlobalPlayerRatingError(
            f"global player fit has {len(frame)} complete maps; {cfg.minimum_maps} required"
        )
    names = sorted(
        {
            player
            for game_id in frame["game_id"].astype(str)
            for side in ("Blue", "Red")
            for player, _ in lineups[game_id][side]
        },
        key=str.casefold,
    )
    design = _design(frame, lineups, names)
    outcome = frame["result"].to_numpy(dtype=float)
    game_ids = frame["game_id"].astype(str)
    metrics = (
        _contribution_metrics(players, _map_dates(frame))
        if cfg.performance_anchor_enabled
        else pd.DataFrame()
    )
    anchor, anchored, anchor_evidence = _performance_anchor(metrics, names, set(game_ids), cfg)
    # FAIL CLOSED.  An anchor that reaches zero players is not a neutral
    # anchor, it is a silently inert one: the published ladder would go back to
    # handing byte-identical ratings to every player who never appears apart
    # from a teammate.  This is exactly what happened when the release
    # projection at lol_kills/export/public_pack.py:1546 dropped the
    # contribution columns, so a release-grade fit must refuse to publish.
    if validate and cfg.performance_anchor_enabled and not anchor_evidence["players_anchored"]:
        raise GlobalPlayerRatingError(
            "performance anchor is enabled but anchored 0 of "
            f"{len(names)} players: contribution statistics are absent from the "
            "rating input, so the published ladder would keep every teammate "
            "tie. Check that the caller's column projection carries "
            + ", ".join(PERFORMANCE_ANCHOR_SOURCE_COLUMNS)
        )
    roots, component_sizes, largest, connected_share = _component_summary(frame, lineups)
    if connected_share < cfg.minimum_connected_share:
        raise GlobalPlayerRatingError(
            f"largest player component covers {connected_share:.1%}; "
            f"{cfg.minimum_connected_share:.1%} required"
        )

    holdout: dict[str, float | int | None] = {
        "train_maps": None,
        "test_maps": None,
        "model_log_loss": None,
        "side_only_log_loss": None,
        "gain": None,
    }
    if validate:
        split = min(max(int(len(frame) * (1.0 - cfg.holdout_fraction)), 1), len(frame) - 1)
        train_x = design[:split]
        test_x = design[split:]
        train_y = outcome[:split]
        test_y = outcome[split:]
        # The holdout anchor sees train maps only. Contribution metrics are
        # measured on the same maps as the outcome, so a full-census anchor
        # would leak test-window performance into the gate.
        train_anchor, _train_anchored, _train_evidence = _performance_anchor(
            metrics, names, set(game_ids.iloc[:split]), cfg
        )
        train_coefficients, train_side = _fit(train_x, train_y, cfg, anchor=train_anchor)
        model_loss = _log_loss(test_y, np.asarray(test_x @ train_coefficients).reshape(-1) + train_side)
        blue_rate = min(max(float(train_y.mean()), 1e-6), 1.0 - 1e-6)
        side_only = math.log(blue_rate / (1.0 - blue_rate))
        baseline_loss = _log_loss(test_y, np.full(len(test_y), side_only))
        gain = baseline_loss - model_loss
        holdout = {
            "train_maps": int(split),
            "test_maps": int(len(frame) - split),
            "model_log_loss": model_loss,
            "side_only_log_loss": baseline_loss,
            "gain": gain,
        }
        if gain < cfg.minimum_holdout_gain:
            raise GlobalPlayerRatingError(
                f"holdout log-loss gain is {gain:.6f}; {cfg.minimum_holdout_gain:.6f} required"
            )

    coefficients, side_advantage = _fit(design, outcome, cfg, anchor=anchor)
    appearances: dict[str, int] = {name: 0 for name in names}
    for game_id in frame["game_id"].astype(str):
        for side in ("Blue", "Red"):
            for player, _ in lineups[game_id][side]:
                appearances[player] += 1
    rows = []
    for position, (name, coefficient) in enumerate(zip(names, coefficients)):
        root = roots[name]
        row = {
            "player": name,
            "global_rating": cfg.prior_rating + LOGIT_TO_ELO * float(coefficient),
            "global_logit": float(coefficient),
            "global_connected": int(root == largest),
            "global_component_id": str(root),
            "global_component_size": int(component_sizes[root]),
            "global_model_maps": int(appearances[name]),
        }
        if cfg.performance_anchor_enabled:
            row["global_performance_anchor_logit"] = float(anchor[position])
            row["global_performance_anchored"] = int(bool(anchored[position]))
        rows.append(row)
    snapshot = pd.DataFrame(rows).sort_values(
        ["global_connected", "global_rating", "player"],
        ascending=[False, False, True],
        kind="stable",
    )
    meta: dict[str, Any] = {
        "model": "regularized_global_player_bt",
        "claim": "One descriptive results scale across all accepted competition tiers.",
        "n_maps": int(len(frame)),
        "n_players": int(len(names)),
        "n_components": int(len(component_sizes)),
        "largest_component_players": int(component_sizes[largest]),
        "connected_share": float(connected_share),
        "side_advantage_logit": float(side_advantage),
        "config": asdict(cfg),
        "holdout": holdout,
        "tier_adjustments": False,
        "player_statistics_used": bool(anchor_evidence["players_anchored"] > 0),
        "performance_anchor": anchor_evidence,
    }
    return snapshot.reset_index(drop=True), meta
