#!/usr/bin/env python3
"""Player Dual-Elo team aggregate (DESCRIPTIVE BASELINE).

This track is the descriptive baseline for the public player ladder.  It is
NOT the v2 dynamic Player Rating.  A shared team outcome still drives every
player on a side; the per-player attribution below only reallocates that one
shared residual using leakage-safe, role-normalized box-score evidence, and
the multipliers average to 1 within a side.  The baseline therefore still
cannot identify individual causal contribution, posterior displacement,
precision, or source/context coverage — reallocating a team residual is a
descriptive split, not an identification result.  Roster moves travel with
the player: team strength is the mean of the five pre-match player μs
(regional + meta), not a sticky org rating.

The attribution composite weights are an UNFITTED development default (see
``ATTRIBUTION_FEATURE_WEIGHTS``).  Protocol v5 requires fitted weights and an
independent acceptance record before any promotion.

The v2 dynamic Player Rating lives in ``lol_kills/v2/ratings/player/`` and
remains development-only until its acceptance record passes; until then this
baseline carries the public label with an explicit descriptive claim ceiling.

This module measures historical results. It does not identify a player's
causal contribution and does not authorize predictions or betting decisions.

  python3 -m lol_kills.ratings.player_elo
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from lol_kills.etl.aliases import normalize_team
from lol_kills.etl.competition import (
    canonicalize_competition_frame,
    classify_competition,
    is_team_affiliation_league,
)
from lol_kills.etl.paths import FEATURES_DIR, PARQUET_DIR
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.ratings.dual_elo import _is_intl, expected_score
from lol_kills.ratings.momentum_config import (
    DEFAULT_MOMENTUM_SCALE,
    DEFAULT_MOMENTUM_WINDOW_GAMES,
    registered_momentum_bundle,
    selected_momentum_configuration,
)
from lol_kills.ratings.global_player_bt import (
    GlobalPlayerBTConfig,
    GlobalPlayerFitCache,
    GlobalPlayerFitWorkspace,
    GlobalPlayerRatingError,
    PrefixBaselineCache,
    _frame_digest as _global_frame_digest,
    _player_baseline_group as _shared_player_baseline_group,
    _kth_abs_distance as _shared_kth_abs_distance,
    _linear_quantile_sorted as _shared_linear_quantile_sorted,
    _prior_baseline_z as _shared_prior_baseline_z,
    _role_normalized_composite as _shared_role_normalized_composite,
    _robust_block_baseline as _shared_robust_block_baseline_reference,
    _robust_block_baseline_fast as _shared_robust_block_baseline_fast,
    fit_global_player_bt,
)

# Slight role weights for aggregation (still sums≈5)
ROLE_WEIGHT = {
    "top": 0.95,
    "jng": 1.05,
    "jungle": 1.05,
    "mid": 1.10,
    "bot": 1.05,
    "adc": 1.05,
    "sup": 0.90,
    "support": 0.90,
    "utility": 0.90,
}

# ---------------------------------------------------------------------------
# Per-player performance attribution (DEVELOPMENT ONLY)
#
# The baseline applies one shared team-outcome residual to all five players on
# a side, so teammates whose sigma has converged receive byte-identical rating
# updates forever.  Attribution reallocates that same team update among the
# five players using per-player, leakage-safe, role-normalized box-score
# evidence.  It is CONSERVATIVE by construction: the multipliers are
# re-centered so their mean over a side is exactly 1, therefore the side's
# aggregate update is unchanged and only the split among teammates moves.
#
# This does not identify causal contribution.  It reallocates a descriptive
# team residual using descriptive per-map evidence.
# ---------------------------------------------------------------------------

# Raw per-player columns carried out of the lineup builder.  Everything here is
# read straight from the OE player frame; nothing is imputed.
ATTRIBUTION_METRIC_COLUMNS: tuple[str, ...] = (
    "cspm",
    "dpm",
    "damageshare",
    "totalgold",
    "earnedgold",
    "kills",
    "deaths",
    "assists",
    "teamkills",
    "gamelength",
    "wpm",
    "wcpm",
)

# !!! UNFITTED DEVELOPMENT DEFAULT — NOT A FITTED RESULT !!!
# These are equal weights chosen so the composite is a plain mean of the
# available z-scores.  They have been fitted against nothing, validated
# against nothing, and carry no public, production, probability,
# recommendation, odds, EV, or promotion authority.  Protocol v5 requires
# fitted weights with an independent acceptance record before this composite
# may be promoted.  Treat any ordering produced with these weights as
# development scaffolding only.
ATTRIBUTION_FEATURE_WEIGHTS: dict[str, float] = {
    "cs_per_min": 1.0,
    "gold_per_min": 1.0,
    "gold_share_pct": 1.0,
    "damage_per_min": 1.0,
    "damage_share_pct": 1.0,
    "kda_role_weighted": 1.0,
    "wpm": 1.0,
    "wcpm": 1.0,
}
ATTRIBUTION_WEIGHTS_STATUS = "unfitted_development_default"

# A (role, competition_tier) baseline needs at least this many observations
# from strictly earlier maps before it may standardize anything.  Below the
# floor the metric is unavailable and the player falls back to neutral.
ATTRIBUTION_MIN_BASELINE_OBS = 20


def _rating_source_identity(maps: pd.DataFrame | None) -> str:
    """Hash the canonical map census used to bind the persistent cache."""

    canonical: set[str] = set()
    if maps is None or maps.empty:
        pass
    elif "game_uid" in maps.columns:
        fallback = maps["gameid"] if "gameid" in maps.columns else None
        for index, value in maps["game_uid"].items():
            game_id = canonical_source_game_key(
                value,
                fallback.loc[index] if fallback is not None else None,
            )
            if game_id:
                canonical.add(str(game_id))
    elif "gameid" in maps.columns:
        for value in maps["gameid"].tolist():
            game_id = canonical_source_game_key(value)
            if game_id:
                canonical.add(str(game_id))
    raw = ("\n".join(sorted(canonical)) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _replay_source_identity(maps: pd.DataFrame, players: pd.DataFrame) -> str:
    """Hash the canonical rows that one sequential replay can read."""

    map_frame = canonicalize_competition_frame(maps).copy()
    map_frame["date"] = pd.to_datetime(
        map_frame.get("date"), errors="coerce", utc=True
    ).dt.tz_localize(None)
    if "game_uid" in map_frame.columns:
        fallback = map_frame["gameid"] if "gameid" in map_frame.columns else None
        map_frame["game_uid"] = [
            canonical_source_game_key(
                value,
                fallback.loc[index] if fallback is not None else None,
            )
            for index, value in map_frame["game_uid"].items()
        ]
    elif "gameid" in map_frame.columns:
        map_frame["game_uid"] = map_frame["gameid"].map(canonical_source_game_key)
    else:
        map_frame["game_uid"] = ""
    map_frame = map_frame[
        map_frame["game_uid"].astype(str).str.strip().ne("")
    ].sort_values(["date", "game_uid"], kind="mergesort").reset_index(drop=True)
    map_ids = set(map_frame["game_uid"].astype(str))

    player_frame = players.copy()
    if "game_uid" in player_frame.columns:
        fallback = player_frame["gameid"] if "gameid" in player_frame.columns else None
        player_ids = pd.Series(
            [
                canonical_source_game_key(
                    value,
                    fallback.loc[index] if fallback is not None else None,
                )
                for index, value in player_frame["game_uid"].items()
            ],
            index=player_frame.index,
        )
    elif "gameid" in player_frame.columns:
        player_ids = player_frame["gameid"].map(canonical_source_game_key)
    else:
        player_ids = pd.Series("", index=player_frame.index)
    player_frame = player_frame.loc[
        player_ids.astype(str).isin(map_ids)
    ].reset_index(drop=True)
    raw = (
        _global_frame_digest(map_frame) + _global_frame_digest(player_frame)
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _rating_cache_schema(players: pd.DataFrame | None) -> str:
    """Fingerprint input columns and the exact baseline implementation."""

    columns = sorted(str(column) for column in (players.columns if players is not None else []))
    implementation = (
        _robust_block_baseline,
        _robust_block_baseline_fast,
        _shared_robust_block_baseline_reference,
        _shared_robust_block_baseline_fast,
        _shared_kth_abs_distance,
        _shared_linear_quantile_sorted,
        _shared_prior_baseline_z,
        _shared_role_normalized_composite,
        GlobalPlayerFitWorkspace,
        PrefixBaselineCache,
    )
    source_parts: list[str] = []
    for function in implementation:
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            source = repr(function)
        source_parts.append(source)
    raw = ("\n".join(columns) + "\n" + "\n".join(source_parts)).encode("utf-8")
    return "rating-input:v2:" + hashlib.sha256(raw).hexdigest()

# Consistency constants for the ROBUST baseline used by ``_prior_baseline_z``.
# Mirrors ``lol_kills.ratings.global_player_bt._MAD_TO_SIGMA`` /
# ``_IQR_TO_SIGMA``; see that module for why the baseline is a median and a
# MAD rather than a mean and a standard deviation.  In short: mean/std has a
# breakdown point of 1/n, so one malformed but ingestible statistic poisons the
# baseline every LATER row in the pool is measured against, and clipping the
# resulting z cannot undo that.  Median and MAD both break down only at 50%.
_MAD_TO_SIGMA = 1.4826
_IQR_TO_SIGMA = 1.349

# Guard for the post-tanh renormalization divisor.
_ATTRIBUTION_MEAN_FLOOR = 1e-9

# Diagnostic record from the most recent ``_run_player_elo`` call.  Read-only
# for callers; it exists so the run manifest can report attribution coverage
# and fail-closed fallbacks without changing the replay return contract.
LAST_ATTRIBUTION_STATS: dict[str, object] = {}


@dataclass
class PlayerState:
    mu_regional: float = 1500.0
    mu_meta: float = 0.0
    sigma: float = 90.0
    last_date: pd.Timestamp | None = None
    n_maps: int = 0
    last_team: str | None = None
    home_league: str | None = None
    momentum_history: list[float] = field(default_factory=list)
    momentum_residual: float = 0.0


@dataclass
class PlayerEloConfig:
    k_regional: float = 18.0
    k_meta: float = 10.0
    sigma0: float = 90.0
    sigma_min: float = 28.0
    sigma_month_inflate: float = 1.0
    team_switch_sigma_bump: float = 12.0
    mov_scale: float = 1.0
    use_role_weights: bool = True
    tier2_bridge_sigma: float = 45.0
    tier3_bridge_sigma: float = 60.0
    bridge_support_scale: float = 10.0
    # Blend toward prior when <5 known starters
    prior_mu: float = 1500.0
    momentum_window_games: int = DEFAULT_MOMENTUM_WINDOW_GAMES
    momentum_scale: float = DEFAULT_MOMENTUM_SCALE
    # Per-player attribution.  ``attribution_beta`` is the maximum fractional
    # deviation of a teammate's share of the shared team update; the tanh with
    # ``attribution_clip`` bounds the effect of extreme composite grades.
    attribution_beta: float = 0.25
    attribution_clip: float = 1.5
    attribution_enabled: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.momentum_window_games, bool)
            or not isinstance(self.momentum_window_games, int)
            or self.momentum_window_games < 0
        ):
            raise ValueError("momentum_window_games must be a non-negative integer")
        try:
            scale = float(self.momentum_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError("momentum_scale must be a finite non-negative value") from exc
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("momentum_scale must be a finite non-negative value")
        self.momentum_scale = scale
        try:
            beta = float(self.attribution_beta)
        except (TypeError, ValueError) as exc:
            raise ValueError("attribution_beta must be a finite value in [0, 1)") from exc
        if not math.isfinite(beta) or beta < 0.0 or beta >= 1.0:
            # beta >= 1 would let a multiplier reach zero or flip sign, which
            # would silently invert a teammate's update.  Fail closed.
            raise ValueError("attribution_beta must be a finite value in [0, 1)")
        self.attribution_beta = beta
        try:
            clip = float(self.attribution_clip)
        except (TypeError, ValueError) as exc:
            raise ValueError("attribution_clip must be a finite positive value") from exc
        if not math.isfinite(clip) or clip <= 0.0:
            raise ValueError("attribution_clip must be a finite positive value")
        self.attribution_clip = clip
        if not isinstance(self.attribution_enabled, bool):
            raise ValueError("attribution_enabled must be a bool")


def total_mu(st: PlayerState) -> float:
    return st.mu_regional + st.mu_meta


def _momentum_residual(st: PlayerState, cfg: PlayerEloConfig) -> float:
    if cfg.momentum_window_games <= 0:
        return 0.0
    if st.momentum_history:
        return float(np.mean(st.momentum_history[-cfg.momentum_window_games :]))
    return float(st.momentum_residual)


def _append_momentum(st: PlayerState, residual: float, cfg: PlayerEloConfig) -> None:
    if cfg.momentum_window_games <= 0:
        return
    st.momentum_history.append(float(residual))
    if len(st.momentum_history) > cfg.momentum_window_games:
        del st.momentum_history[:-cfg.momentum_window_games]
    st.momentum_residual = float(np.mean(st.momentum_history))


def _norm_role(r: str) -> str:
    r = str(r or "").lower().strip()
    if r in ROLE_WEIGHT:
        return r if r not in ("jungle", "adc", "support", "utility") else {
            "jungle": "jng",
            "adc": "bot",
            "support": "sup",
            "utility": "sup",
        }[r]
    for a, b in (
        ("jng", "jng"),
        ("jung", "jng"),
        ("mid", "mid"),
        ("top", "top"),
        ("bot", "bot"),
        ("adc", "bot"),
        ("sup", "sup"),
        ("supp", "sup"),
        ("util", "sup"),
    ):
        if r.startswith(a):
            return b
    return r[:3] if r else "unk"


def _role_w(role: str, cfg: PlayerEloConfig) -> float:
    if not cfg.use_role_weights:
        return 1.0
    return float(ROLE_WEIGHT.get(_norm_role(role), 1.0))


def _lineups_by_game(
    players: pd.DataFrame,
    *,
    with_metrics: bool = False,
):
    """
    game_uid → {Blue|Red: [(playername, role), ...]}
    Only position rows with a player name (skip team aggregates).

    With ``with_metrics=True`` this also returns the per-player metric frame for
    exactly the rows that survived role dedupe and the five-slot cap, so the
    attribution baseline sees the same population the rating update sees.  The
    ``(playername, role)`` tuple contract and ordering are unchanged, so the
    default call site and every existing consumer keep the old return value.
    """
    empty_metrics = pd.DataFrame(
        columns=["_gid", "side", "_name", "_role", "_attr_date", "_attr_tier"]
    )
    if players is None or players.empty or "playername" not in players.columns:
        return ({}, empty_metrics) if with_metrics else {}
    p = players.copy()
    if "game_uid" in p.columns:
        fallback = p["gameid"] if "gameid" in p.columns else None
        p["_gid"] = [
            canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
            for index, value in p["game_uid"].items()
        ]
    elif "gameid" in p.columns:
        p["_gid"] = p["gameid"].map(canonical_source_game_key)
    else:
        return ({}, empty_metrics) if with_metrics else {}
    p = p[p["_gid"].notna() & p["_gid"].str.strip().ne("")]
    p["side"] = p["side"].astype(str).str.title()
    p["position"] = p.get("position", pd.Series("unk", index=p.index)).astype(str)
    pos = p["position"].str.lower()
    p = p[pos != "team"].copy()
    p = p[p["playername"].notna() & (p["playername"].astype(str).str.len() > 0)]
    p["_role"] = p["position"].map(_norm_role)
    p["_name"] = p["playername"].astype(str).str.strip()
    p = p[p["_name"].str.lower() != "nan"]
    # Positional labels so the accepted-row index below is unambiguous even if
    # the caller handed us a frame with a duplicated index.
    p = p.reset_index(drop=True)

    out: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: {"Blue": [], "Red": []})
    accepted: list[int] = []
    for (gid, side), g in p.groupby(["_gid", "side"], sort=False):
        if side not in ("Blue", "Red"):
            continue
        # stable role order
        order = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
        rows = list(zip(g.index.tolist(), g["_name"].tolist(), g["_role"].tolist()))
        # Sort only on role, and stably, so the (name, role) order is identical
        # to the pre-attribution implementation.
        rows.sort(key=lambda item: order.get(item[2], 9))
        # dedupe by role keep first
        seen = set()
        cleaned = []
        kept: list[int] = []
        for label, name, role in rows:
            if role in seen:
                continue
            seen.add(role)
            cleaned.append((name, role))
            kept.append(label)
        out[str(gid)][side] = cleaned[:5]
        if with_metrics:
            accepted.extend(kept[:5])
    if not with_metrics:
        return out
    return out, _attribution_metric_frame(p, accepted)


def _attribution_metric_frame(p: pd.DataFrame, accepted: list[int]) -> pd.DataFrame:
    """Per-player metric rows for exactly the accepted lineup slots.

    Missing metric columns stay missing.  Nothing is imputed here: a column the
    source does not carry simply never becomes an available feature.
    """

    keep = ["_gid", "side", "_name", "_role"]
    metrics = pd.DataFrame(index=p.index)
    for column in keep:
        metrics[column] = p[column]
    metrics["_gid"] = metrics["_gid"].astype(str)
    metrics["side"] = metrics["side"].astype(str)
    metrics["_name"] = metrics["_name"].astype(str)
    metrics["_role"] = metrics["_role"].astype(str)

    if "date" in p.columns:
        metrics["_attr_date"] = pd.to_datetime(
            p["date"], errors="coerce", utc=True
        ).dt.tz_localize(None)
    else:
        metrics["_attr_date"] = pd.NaT
    metrics["_attr_tier"] = _attribution_tier(p)

    for column in ATTRIBUTION_METRIC_COLUMNS:
        if column in p.columns:
            metrics[column] = pd.to_numeric(p[column], errors="coerce")
    return metrics.loc[accepted].reset_index(drop=True)


def _attribution_tier(p: pd.DataFrame) -> pd.Series:
    """Competition tier used to group the role baselines.

    Prefers the canonical column already carried by the warehouse frame.  When
    it is absent the tier is derived from the unique (league, tournament)
    labels — cheap, and identical to what ``canonicalize_competition_frame``
    would produce — rather than guessed.  When neither is available the tier is
    missing and every row fails closed to a neutral multiplier.
    """

    if "competition_tier" in p.columns:
        tier = p["competition_tier"].astype("string").str.strip()
        return tier.mask(tier.isna() | tier.eq(""), pd.NA)
    if "league" not in p.columns:
        return pd.Series(pd.NA, index=p.index, dtype="string")
    league = p["league"].astype("string")
    if "tournament" in p.columns:
        tournament = p["tournament"].astype("string")
    else:
        tournament = pd.Series(pd.NA, index=p.index, dtype="string")
    pairs = pd.DataFrame({"league": league, "tournament": tournament})
    unique = pairs.drop_duplicates()
    lookup = {
        (row.league, row.tournament): classify_competition(row.league, row.tournament).tier
        for row in unique.itertuples(index=False)
    }
    resolved = [
        lookup.get((lg, tn)) for lg, tn in zip(pairs["league"], pairs["tournament"])
    ]
    tier = pd.Series(resolved, index=p.index, dtype="string").str.strip()
    return tier.mask(tier.isna() | tier.eq(""), pd.NA)


def _finite(values: pd.Series | None, index: pd.Index) -> pd.Series:
    """Float series with every non-finite entry marked unavailable (NaN)."""

    if values is None:
        return pd.Series(np.nan, index=index, dtype=float)
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    return numeric.where(np.isfinite(numeric.to_numpy(dtype=float)), np.nan)


def _attribution_features(metrics: pd.DataFrame) -> pd.DataFrame:
    """Derived per-player inputs.

    Raw gold is never a direct feature: it enters only as a per-minute rate and
    as a within-team share.  Any input the source does not carry stays NaN and
    is reported as unavailable rather than replaced by zero.
    """

    index = metrics.index
    column = lambda name: metrics[name] if name in metrics.columns else None  # noqa: E731

    cspm = _finite(column("cspm"), index)
    dpm = _finite(column("dpm"), index)
    damageshare = _finite(column("damageshare"), index)
    totalgold = _finite(column("totalgold"), index)
    gamelength = _finite(column("gamelength"), index)
    kills = _finite(column("kills"), index)
    deaths = _finite(column("deaths"), index)
    assists = _finite(column("assists"), index)
    wpm = _finite(column("wpm"), index)
    wcpm = _finite(column("wcpm"), index)

    # gamelength is seconds in the OE frame; a non-positive length is unusable.
    minutes = gamelength / 60.0
    minutes = minutes.where(minutes > 0.0, np.nan)

    if totalgold.notna().any():
        team_gold = totalgold.groupby(
            [metrics["_gid"], metrics["side"]], sort=False
        ).transform("sum", min_count=1)
    else:
        team_gold = pd.Series(np.nan, index=index, dtype=float)
    team_gold = team_gold.where(team_gold > 0.0, np.nan)

    kda = (kills + assists) / deaths.clip(lower=1.0)

    features = pd.DataFrame(index=index)
    features["cs_per_min"] = cspm
    features["gold_per_min"] = totalgold / minutes
    features["gold_share_pct"] = totalgold / team_gold
    features["damage_per_min"] = dpm
    features["damage_share_pct"] = damageshare
    features["kda_role_weighted"] = kda
    features["wpm"] = wpm
    features["wcpm"] = wcpm
    for name in features.columns:
        values = features[name].to_numpy(dtype=float)
        features[name] = features[name].where(np.isfinite(values), np.nan)
    return features


def _robust_block_baseline(
    pool: np.ndarray,
    available: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Expanding median and robust scale for one group's blocks.

    Mirrors ``lol_kills.ratings.global_player_bt._robust_block_baseline``.

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


def _robust_block_baseline_fast(
    pool: np.ndarray,
    available: np.ndarray,
    min_obs: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the shared exact sorted-prefix implementation."""

    return _shared_robust_block_baseline_fast(pool, available, min_obs)


def _prior_baseline_z(
    values: pd.Series,
    group: pd.Series,
    date: pd.Series,
    min_obs: int,
    *,
    baseline_cache: PrefixBaselineCache | None = None,
    metric_key: str | None = None,
    row_key: pd.Series | None = None,
    prepared_query: object | None = None,
) -> pd.Series:
    """Robust z-score against a baseline built only from strictly earlier dates.

    Mirrors ``lol_kills.ratings.global_player_bt._prior_baseline_z``, which
    additionally returns the prior observation count; the two are pinned
    together by ``test_prior_baseline_z_matches_the_elo_implementation``.

    The baseline for a row is the expanding MEDIAN and MAD over every row in
    the same ``group`` whose date is strictly before the row's own date, so map
    ``t`` never contributes to its own baseline.  Rows sharing a timestamp form
    one block and cannot see each other.  A baseline thinner than ``min_obs``
    or with a degenerate robust spread yields an unavailable (NaN) z-score.

    Median/MAD rather than mean/std is the whole point: see ``_MAD_TO_SIGMA``.
    A single malformed row cannot move a 50%-breakdown estimator, so it cannot
    poison the baseline that every LATER row in the pool is measured against.
    """

    if baseline_cache is not None and metric_key is not None and row_key is not None:
        cached = baseline_cache.lookup(
            values,
            group,
            date,
            min_obs,
            metric_key=metric_key,
            row_key=row_key,
            block_baseline=_robust_block_baseline_fast,
            prepared_query=prepared_query,
        )
        if cached is not None:
            return cached[0]

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

        block_location, block_scale = _robust_block_baseline_fast(
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
    output = pd.Series(z, index=index, dtype=float)
    if baseline_cache is not None and metric_key is not None and row_key is not None:
        baseline_cache.store(
            values,
            group,
            date,
            min_obs,
            metric_key=metric_key,
            row_key=row_key,
            z=output,
            prior_count=pd.Series(prior_count, index=index, dtype=float),
            prepared_query=prepared_query,
        )
    return output


def player_attribution_multipliers(
    metrics: pd.DataFrame,
    cfg: PlayerEloConfig,
    *,
    baseline_cache: PrefixBaselineCache | None = None,
) -> tuple[dict[tuple[str, str, str], float], dict[str, object]]:
    """Conservative per-player multipliers for the shared team update.

    Returns ``{(game_uid, side, playername): a_i}`` plus a diagnostic record.
    A player absent from the mapping is neutral (``a_i = 1.0``, i.e. exactly
    the pre-attribution behaviour).  Within a side the returned multipliers
    average to exactly 1, so the side's aggregate update is unchanged.
    """

    weights = dict(ATTRIBUTION_FEATURE_WEIGHTS)
    stats: dict[str, object] = {
        "enabled": bool(cfg.attribution_enabled),
        "weights": weights,
        "weights_status": ATTRIBUTION_WEIGHTS_STATUS,
        "min_baseline_obs": int(ATTRIBUTION_MIN_BASELINE_OBS),
        "beta": float(cfg.attribution_beta),
        "clip": float(cfg.attribution_clip),
        "rows_total": 0,
        "rows_graded": 0,
        "rows_attributed": 0,
        "rows_neutral_total": 0,
        "rows_neutral_no_composite": 0,
        "rows_neutral_single_graded_side": 0,
        "rows_neutral_renorm_guard": 0,
        "feature_available_rows": {name: 0 for name in weights},
        "unavailable_reason": None,
    }
    if not cfg.attribution_enabled:
        stats["unavailable_reason"] = "attribution_disabled"
        return {}, stats
    if metrics is None or metrics.empty:
        stats["unavailable_reason"] = "no_player_metric_rows"
        return {}, stats

    stats["rows_total"] = int(len(metrics))
    if "_attr_date" not in metrics.columns or metrics["_attr_date"].isna().all():
        stats["unavailable_reason"] = "no_usable_map_date"
        stats["rows_neutral_no_composite"] = int(len(metrics))
        stats["rows_neutral_total"] = int(len(metrics))
        return {}, stats
    if "_attr_tier" not in metrics.columns or metrics["_attr_tier"].isna().all():
        stats["unavailable_reason"] = "no_usable_competition_tier"
        stats["rows_neutral_no_composite"] = int(len(metrics))
        stats["rows_neutral_total"] = int(len(metrics))
        return {}, stats

    features = _attribution_features(metrics)
    # Role normalization happens here: the baseline is grouped by role so every
    # z-score is relative to same-role, same-tier prior maps.
    group = _shared_player_baseline_group(metrics["_role"], metrics["_attr_tier"])
    date = metrics["_attr_date"]

    ordered = [name for name in weights if name in features.columns]
    if not ordered:
        stats["unavailable_reason"] = "no_attribution_features_present"
        stats["rows_neutral_no_composite"] = int(len(metrics))
        stats["rows_neutral_total"] = int(len(metrics))
        return {}, stats

    z_frame = pd.DataFrame(index=metrics.index)
    row_key = pd.Series(
        list(
            zip(
                metrics["_gid"].astype(str),
                metrics["side"].astype(str),
                metrics["_name"].astype(str),
                metrics["_role"].astype(str),
            )
        ),
        index=metrics.index,
    )
    prepared_query = (
        baseline_cache.prepare_query(group, date, row_key)
        if baseline_cache is not None
        else None
    )
    for name in ordered:
        z_frame[name] = _prior_baseline_z(
            features[name],
            group,
            date,
            ATTRIBUTION_MIN_BASELINE_OBS,
            baseline_cache=baseline_cache,
            metric_key=name,
            row_key=row_key,
            prepared_query=prepared_query,
        )
        stats["feature_available_rows"][name] = int(z_frame[name].notna().sum())

    z_values = z_frame.to_numpy(dtype=float)
    weight_row = np.array([float(weights[name]) for name in ordered], dtype=float)
    available = np.isfinite(z_values)
    weight_sum = (available * weight_row).sum(axis=1)
    weighted = np.where(available, z_values * weight_row, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        composite = np.where(weight_sum > 0.0, weighted / weight_sum, np.nan)
    composite = np.where(np.isfinite(composite), composite, np.nan)

    graded_mask = np.isfinite(composite)
    stats["rows_graded"] = int(graded_mask.sum())
    stats["rows_neutral_no_composite"] = int((~graded_mask).sum())
    if not graded_mask.any():
        stats["unavailable_reason"] = "no_row_cleared_the_baseline_floor"
        stats["rows_neutral_total"] = int(len(metrics))
        return {}, stats

    graded = pd.DataFrame(
        {
            "_gid": metrics["_gid"].to_numpy()[graded_mask],
            "side": metrics["side"].to_numpy()[graded_mask],
            "_name": metrics["_name"].to_numpy()[graded_mask],
            "_g": composite[graded_mask],
        }
    )
    side_key = graded["_gid"].astype(str) + "|" + graded["side"].astype(str)
    grouped = graded["_g"].groupby(side_key, sort=False)
    side_size = grouped.transform("size").to_numpy(dtype=float)

    # Within-team centering: only the graded members of a side participate, so
    # an ungraded teammate keeps a_i exactly 1.0 and the mean over the whole
    # side is still exactly 1.
    centered = graded["_g"].to_numpy(dtype=float) - grouped.transform("mean").to_numpy(
        dtype=float
    )
    raw = 1.0 + float(cfg.attribution_beta) * np.tanh(
        centered / float(cfg.attribution_clip)
    )
    # The tanh is nonlinear, so the centered grades do not give mean(a) == 1.
    # Renormalize within the side to restore exact conservation of the team's
    # aggregate update.
    raw_series = pd.Series(raw, index=graded.index)
    raw_mean = raw_series.groupby(side_key, sort=False).transform("mean").to_numpy(
        dtype=float
    )
    guard = np.isfinite(raw_mean) & (raw_mean > _ATTRIBUTION_MEAN_FLOOR)
    multipliers = np.where(guard, raw / np.where(guard, raw_mean, 1.0), 1.0)
    multipliers = np.where(np.isfinite(multipliers), multipliers, 1.0)

    stats["rows_neutral_renorm_guard"] = int((~guard).sum())
    single = side_size < 2.0
    stats["rows_neutral_single_graded_side"] = int(single.sum())
    stats["rows_attributed"] = int((guard & ~single).sum())
    stats["rows_neutral_total"] = int(
        stats["rows_neutral_no_composite"]
        + stats["rows_neutral_single_graded_side"]
        + stats["rows_neutral_renorm_guard"]
    )

    mapping = {
        (str(gid), str(side), str(name)): float(value)
        for gid, side, name, value in zip(
            graded["_gid"].to_numpy(),
            graded["side"].to_numpy(),
            graded["_name"].to_numpy(),
            multipliers,
        )
    }
    return mapping, stats


def _aggregate(
    states: dict[str, PlayerState],
    lineup: list[tuple[str, str]],
    cfg: PlayerEloConfig,
    *,
    include_momentum: bool = True,
) -> tuple[float, float, int, list[dict]]:
    """Return (mu, sigma_mean, n_known, per-player detail)."""
    if not lineup:
        return cfg.prior_mu, cfg.sigma0, 0, []
    details = []
    w_sum = 0.0
    mu_acc = 0.0
    sig_acc = 0.0
    known = 0
    for name, role in lineup[:5]:
        st = states.get(name)
        w = _role_w(role, cfg)
        if st is None:
            mu_base = cfg.prior_mu
            momentum_residual = 0.0
            sig = cfg.sigma0
        else:
            mu_base = total_mu(st)
            momentum_residual = _momentum_residual(st, cfg)
            sig = st.sigma
            known += 1
        momentum_points = cfg.momentum_scale * momentum_residual if include_momentum else 0.0
        mu = mu_base + momentum_points
        details.append(
            {
                "player": name,
                "role": role,
                "mu": round(mu, 2),
                "mu_base": round(mu_base, 2),
                "momentum_residual": round(momentum_residual, 5),
                "momentum_points": round(momentum_points, 2),
                "sigma": round(sig, 2),
                "w": w,
            }
        )
        mu_acc += w * mu
        sig_acc += w * sig
        w_sum += w
    if w_sum <= 0:
        return cfg.prior_mu, cfg.sigma0, 0, details
    # If fewer than 5 known, shrink toward prior
    mu = mu_acc / w_sum
    if known < 5:
        shrink = known / 5.0
        mu = cfg.prior_mu + shrink * (mu - cfg.prior_mu)
    sig = sig_acc / w_sum
    return mu, sig, known, details


def _snapshot_rows(
    states: dict[str, PlayerState],
    recent_mus: dict[str, list[float]] | None = None,
    cfg: PlayerEloConfig | None = None,
) -> list[dict[str, object]]:
    cfg = cfg or PlayerEloConfig()
    recent = recent_mus or {}
    rows = []
    for name, st in states.items():
        history = recent.get(name) or []
        stability = None
        if len(history) >= 2:
            deltas = [abs(history[i] - history[i - 1]) for i in range(1, len(history))]
            stability = float(sum(deltas) / len(deltas))
        rows.append(
            {
                "player": name,
                "mu_base_total": total_mu(st),
                "mu_total": total_mu(st) + cfg.momentum_scale * _momentum_residual(st, cfg),
                "mu_effective": total_mu(st) + cfg.momentum_scale * _momentum_residual(st, cfg),
                "momentum_residual": _momentum_residual(st, cfg),
                "mu_regional": st.mu_regional,
                "mu_meta": st.mu_meta,
                "sigma": st.sigma,
                "n_maps": st.n_maps,
                "last_team": st.last_team,
                "home_league": st.home_league,
                "last_game_date": st.last_date.isoformat() if st.last_date is not None else None,
                "evidence_stability": stability,
            }
        )
    return rows


def _apply_global_scale(
    rows: list[dict[str, object]],
    global_snapshot: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
) -> list[dict[str, object]]:
    """Replace local-pool means with the connected global results scale."""

    cfg = cfg or PlayerEloConfig()
    by_player = {
        str(row["player"]): row
        for _, row in global_snapshot.iterrows()
    }
    output = []
    for source in rows:
        row = dict(source)
        global_row = by_player.get(str(row["player"]))
        connected = int(global_row.get("global_connected") or 0) if global_row is not None else 0
        row["global_connected"] = connected
        row["rating_model"] = "regularized_global_player_bt"
        if connected:
            rating = float(global_row["global_rating"])
            row["mu_base_total"] = rating
            row["mu_total"] = rating + cfg.momentum_scale * float(row.get("momentum_residual") or 0.0)
            row["mu_effective"] = row["mu_total"]
            row["mu_regional"] = rating
            row["mu_meta"] = 0.0
            row["global_component_size"] = int(global_row["global_component_size"])
            row["global_model_maps"] = int(global_row["global_model_maps"])
        output.append(row)
    return output


@dataclass
class PlayerBridgeContext:
    """Canonical player rows and reusable bridge support counts.

    The weekly ladder applies the same bridge rule at five cutoffs. Keep the
    canonical competition frame and the filtered support rows in one
    object-local context so each cutoff only performs its date filter and
    groupby. The raw source digest prevents a context from crossing a refresh
    with changed player rows.
    """

    canonical_players: pd.DataFrame
    support_rows: pd.DataFrame
    source_players_digest: str
    counts_by_cutoff: dict[str, pd.Series] = field(default_factory=dict, repr=False)
    bound_players_id: int | None = field(default=None, repr=False)

    @classmethod
    def build(cls, players: pd.DataFrame) -> "PlayerBridgeContext":
        canonical_players = canonicalize_competition_frame(players).copy()
        frame = canonical_players.copy()
        date_source = (
            frame["date"]
            if "date" in frame.columns
            else pd.Series(pd.NaT, index=frame.index)
        )
        frame["_date"] = pd.to_datetime(
            date_source, utc=True, errors="coerce"
        ).dt.tz_localize(None)
        if "game_uid" in frame.columns:
            fallback = frame["gameid"] if "gameid" in frame.columns else None
            frame["_game_id"] = [
                canonical_source_game_key(
                    value,
                    fallback.loc[index] if fallback is not None else None,
                )
                for index, value in frame["game_uid"].items()
            ]
        elif "gameid" in frame.columns:
            frame["_game_id"] = frame["gameid"].map(canonical_source_game_key)
        else:
            frame["_game_id"] = ""
        frame["_player"] = frame.get(
            "playername", pd.Series("", index=frame.index)
        ).astype("string").str.strip()
        competition_tier = frame.get(
            "competition_tier", pd.Series("", index=frame.index)
        )
        frame["_competition_tier"] = competition_tier
        frame = frame[
            frame["_player"].notna()
            & frame["_player"].ne("")
            & frame["_game_id"].astype(str).str.strip().ne("")
            & frame["_competition_tier"].isin({"tier1", "tier2", "tier3"})
        ][["_date", "_game_id", "_player", "_competition_tier"]]
        return cls(
            canonical_players=canonical_players,
            support_rows=frame,
            source_players_digest=_global_frame_digest(
                players.reset_index(drop=True)
            ),
        )

    def matches_players(self, players: pd.DataFrame) -> bool:
        """Check source identity without canonicalizing the candidate again."""

        return bool(
            self.source_players_digest
            == _global_frame_digest(players.reset_index(drop=True))
        )

    def bind_players(self, players: pd.DataFrame) -> None:
        """Record the validated source object used by one rating pass."""

        self.bound_players_id = id(players)

    @staticmethod
    def _cutoff_key(through: pd.Timestamp | None) -> tuple[str, pd.Timestamp | None]:
        if through is None:
            return "__all__", None
        cutoff = pd.Timestamp(through)
        if cutoff.tzinfo is not None:
            cutoff = cutoff.tz_convert("UTC").tz_localize(None)
        return cutoff.isoformat(), cutoff

    def counts_for(self, through: pd.Timestamp | None) -> pd.Series:
        """Return bridge support counts at an inclusive cutoff."""

        key, cutoff = self._cutoff_key(through)
        cached = self.counts_by_cutoff.get(key)
        if cached is not None:
            return cached
        support = self.support_rows
        if cutoff is not None:
            support = support[support["_date"].le(cutoff)]
        support = support.drop_duplicates(["_player", "_game_id"])
        counts = support.groupby(
            ["_player", "_competition_tier"], sort=False
        ).size()
        self.counts_by_cutoff[key] = counts
        return counts


def _current_tier_records(canonical_players: pd.DataFrame) -> dict[str, object]:
    """Return only the current-tier field used by weekly rank ordering.

    This follows ``build_player_records`` with no team record override. It
    keeps rows with valid results, removes team summary rows, and selects the
    latest team-affiliation league row per player. A stable sort keeps the
    first source row when dates tie, matching ``_latest_row``.
    """

    if (
        canonical_players is None
        or canonical_players.empty
        or "playername" not in canonical_players.columns
        or "league" not in canonical_players.columns
        or "competition_tier" not in canonical_players.columns
    ):
        return {}
    from lol_kills.export.pack_records import INVALID_COMPETITION_LABELS
    if not canonical_players.index.is_unique:
        from lol_kills.export.pack_records import build_player_records

        full_records = build_player_records(
            canonical_players,
            canonicalized=True,
        )
        return {
            str(player): record.get("current_tier")
            for player, record in full_records.items()
        }

    frame = canonical_players.copy()
    frame = frame[frame["playername"].notna()]
    if "position" in frame.columns:
        frame = frame[frame["position"].astype(str).str.lower().ne("team")]
    if frame.empty:
        return {}
    frame["_result"] = pd.to_numeric(frame.get("result"), errors="coerce")
    frame = frame[frame["_result"].notna()].copy()
    if frame.empty:
        return {}
    frame["_player"] = frame["playername"].astype(str)
    valid_league = ~frame["league"].astype(str).str.upper().isin(
        INVALID_COMPETITION_LABELS
    )
    affiliation = frame.loc[valid_league]
    affiliation = affiliation[
        affiliation["league"].map(is_team_affiliation_league)
    ].copy()
    records = {
        player: None for player in frame["_player"].astype(str).unique()
    }
    if affiliation.empty or "date" not in affiliation.columns:
        return records
    affiliation["_date"] = pd.to_datetime(
        affiliation["date"], errors="coerce", utc=True
    )
    affiliation = affiliation[affiliation["_date"].notna()]
    if affiliation.empty:
        return records
    latest = (
        affiliation.sort_values(
            ["_player", "_date"],
            ascending=[True, False],
            kind="mergesort",
        )
        .drop_duplicates("_player", keep="first")
    )
    records.update(
        {
            str(row["_player"]): str(row["competition_tier"])
            for _, row in latest.iterrows()
        }
    )
    return records


def _apply_bridge_uncertainty(
    rows: list[dict[str, object]],
    players: pd.DataFrame,
    player_records: Mapping[str, Mapping[str, object]],
    cfg: PlayerEloConfig,
    *,
    through: pd.Timestamp | None = None,
    bridge_context: PlayerBridgeContext | None = None,
) -> list[dict[str, object]]:
    """Widen weak cross-tier anchors without moving the fitted mean."""

    context = bridge_context
    if context is None:
        context = PlayerBridgeContext.build(players)
    elif context.bound_players_id != id(players):
        if not context.matches_players(players):
            context = PlayerBridgeContext.build(players)
        context.bind_players(players)
    tier_counts = context.counts_for(through)

    output = []
    for source in rows:
        row = dict(source)
        player = str(row["player"])
        tier = str((player_records.get(player) or {}).get("current_tier") or "")
        if tier == "tier2":
            stronger_maps = int(tier_counts.get((player, "tier1"), 0))
            base = cfg.tier2_bridge_sigma
        elif tier == "tier3":
            stronger_maps = int(tier_counts.get((player, "tier1"), 0)) + int(
                tier_counts.get((player, "tier2"), 0)
            )
            base = cfg.tier3_bridge_sigma
        else:
            stronger_maps = 0
            base = 0.0
        bridge_sigma = base / math.sqrt(1.0 + stronger_maps / cfg.bridge_support_scale)
        row["global_bridge_maps"] = stronger_maps
        row["global_bridge_sigma"] = bridge_sigma
        row["sigma"] = min(
            160.0,
            math.hypot(float(row.get("sigma") or cfg.sigma0), bridge_sigma),
        )
        output.append(row)
    return output


def _run_player_elo(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig,
    checkpoint_dates: list[pd.Timestamp] | None = None,
    *,
    baseline_cache: PrefixBaselineCache | None = None,
) -> tuple[pd.DataFrame, dict[str, PlayerState], dict[pd.Timestamp, list[dict[str, object]]]]:
    """Run the sequential player model and optionally capture dated states."""

    # Apply the same source-preserving competition taxonomy as team ratings so
    # player regional/meta updates cannot drift from the public team contract.
    df = canonicalize_competition_frame(maps).copy()
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    if "game_uid" in df.columns:
        fallback = df["gameid"] if "gameid" in df.columns else None
        df["game_uid"] = [
            canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
            for index, value in df["game_uid"].items()
        ]
    elif "gameid" in df.columns:
        df["game_uid"] = df["gameid"].map(canonical_source_game_key)
    else:
        raise ValueError("player Elo maps have no game identity column")
    df = df[df["game_uid"].str.strip().ne("")].copy()
    df = df.sort_values(["date", "game_uid"], kind="mergesort").reset_index(drop=True)
    if cfg.attribution_enabled:
        lineups, attribution_metrics = _lineups_by_game(players, with_metrics=True)
        attribution, attribution_stats = player_attribution_multipliers(
            attribution_metrics,
            cfg,
            baseline_cache=baseline_cache,
        )
    else:
        lineups = _lineups_by_game(players)
        attribution, attribution_stats = player_attribution_multipliers(
            pd.DataFrame(), cfg
        )
    global LAST_ATTRIBUTION_STATS
    LAST_ATTRIBUTION_STATS = attribution_stats
    states: dict[str, PlayerState] = {}
    recent_mus: dict[str, list[float]] = {}
    targets = sorted({pd.Timestamp(value).tz_localize(None) for value in (checkpoint_dates or [])})
    checkpoints: dict[pd.Timestamp, list[dict[str, object]]] = {}
    target_idx = 0

    def capture_before(date: pd.Timestamp | None) -> None:
        nonlocal target_idx
        while target_idx < len(targets) and (date is None or date > targets[target_idx]):
            target = targets[target_idx]
            checkpoints[target] = _snapshot_rows(states, cfg=cfg)
            target_idx += 1

    # Pre-extract the columns the sequential loop reads (avoids per-row pandas access).
    _gid_arr = df["game_uid"].to_numpy(dtype=object)
    _date_arr = df["date"].to_numpy(dtype="datetime64[ns]")
    _bt_col = "blue_team" if "blue_team" in df.columns else "blue_teamname"
    _rt_col = "red_team" if "red_team" in df.columns else "red_teamname"
    _bt_arr = df[_bt_col].astype(str).to_numpy(dtype=object)
    _rt_arr = df[_rt_col].astype(str).to_numpy(dtype=object)
    _y_arr = df["y_blue_win"].to_numpy(dtype=object) if "y_blue_win" in df.columns else np.full(len(df), np.nan, dtype=object)
    _league_arr = df["league"].astype(str).to_numpy(dtype=object) if "league" in df.columns else np.full(len(df), "", dtype=object)
    _tourn_arr = df["tournament"].astype(str).to_numpy(dtype=object) if "tournament" in df.columns else np.full(len(df), "", dtype=object)
    _g15_arr = df["blue_golddiffat15"].to_numpy(dtype=object) if "blue_golddiffat15" in df.columns else np.full(len(df), np.nan, dtype=object)
    _g10_arr = df["blue_golddiffat10"].to_numpy(dtype=object) if "blue_golddiffat10" in df.columns else np.full(len(df), np.nan, dtype=object)
    _len_arr = df["length_min"].to_numpy(dtype=object) if "length_min" in df.columns else np.full(len(df), np.nan, dtype=object)
    _glen_arr = df["gamelength"].to_numpy(dtype=object) if "gamelength" in df.columns else np.full(len(df), np.nan, dtype=object)

    rows = []
    for i in range(len(df)):
        gid = str(_gid_arr[i] or "")
        _dv = _date_arr[i]
        d = pd.Timestamp(_dv) if not pd.isna(_dv) else None
        capture_before(d)
        blue_lu = lineups.get(gid, {}).get("Blue") or []
        red_lu = lineups.get(gid, {}).get("Red") or []
        bt = normalize_team(str(_bt_arr[i] or ""))
        rt = normalize_team(str(_rt_arr[i] or ""))

        # inactivity + team-switch uncertainty
        for name, role in list(blue_lu[:5]) + list(red_lu[:5]):
            if name not in states:
                states[name] = PlayerState(sigma=cfg.sigma0)
            st = states[name]
            if d is not None and st.last_date is not None:
                months = max((d - st.last_date).days / 30.0, 0.0)
                st.sigma = min(160.0, st.sigma + cfg.sigma_month_inflate * months)
            team_now = bt if any(n == name for n, _ in blue_lu[:5]) else rt
            if st.last_team and team_now and st.last_team != team_now:
                st.sigma = min(160.0, st.sigma + cfg.team_switch_sigma_bump)
            states[name] = st

        base_mu_b, sig_b, known_b, _ = _aggregate(
            states, blue_lu, cfg, include_momentum=False
        )
        base_mu_r, sig_r, known_r, _ = _aggregate(
            states, red_lu, cfg, include_momentum=False
        )
        mu_b, _, _, det_b = _aggregate(states, blue_lu, cfg)
        mu_r, _, _, det_r = _aggregate(states, red_lu, cfg)
        sig = math.hypot(sig_b, sig_r)
        p_base = expected_score(base_mu_b, base_mu_r)
        p = expected_score(mu_b, mu_r)
        shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
        p_shrunk = 0.5 + (p - 0.5) * shrink
        blue_by_role = {str(item["role"]): item for item in det_b}
        red_by_role = {str(item["role"]): item for item in det_r}
        role_rating_context: dict[str, float] = {}
        for role in ("top", "jng", "mid", "bot", "sup"):
            blue_detail = blue_by_role.get(role)
            red_detail = red_by_role.get(role)
            available = float(
                blue_detail is not None and red_detail is not None
            )
            role_rating_context.update(
                {
                    f"player_role_mu_diff_{role}": (
                        float(blue_detail["mu"]) - float(red_detail["mu"])
                        if available
                        else 0.0
                    ),
                    f"player_role_sigma_pair_{role}": (
                        math.hypot(
                            float(blue_detail["sigma"]),
                            float(red_detail["sigma"]),
                        )
                        if available
                        else math.hypot(cfg.sigma0, cfg.sigma0)
                    ),
                    f"player_role_momentum_diff_{role}": (
                        float(blue_detail["momentum_points"])
                        - float(red_detail["momentum_points"])
                        if available
                        else 0.0
                    ),
                    f"player_role_rating_available_{role}": available,
                }
            )

        rows.append(
            {
                "game_uid": gid,
                "date": _dv,
                "blue_team": bt,
                "red_team": rt,
                "player_mu_blue": mu_b,
                "player_mu_red": mu_r,
                "player_mu_diff": mu_b - mu_r,
                "player_mu_base_blue": base_mu_b,
                "player_mu_base_red": base_mu_r,
                "player_momentum_blue": mu_b - base_mu_b,
                "player_momentum_red": mu_r - base_mu_r,
                "player_momentum_diff": (mu_b - base_mu_b) - (mu_r - base_mu_r),
                "player_sigma_blue": sig_b,
                "player_sigma_red": sig_r,
                "player_sigma_pair": sig,
                "player_known_blue": known_b,
                "player_known_red": known_r,
                "p_player_elo": p_shrunk,
                "p_player_elo_raw": p,
                "p_player_elo_base": 0.5 + (p_base - 0.5) * shrink,
                "p_player_elo_base_raw": p_base,
                "player_momentum_window_games": cfg.momentum_window_games,
                "player_momentum_scale": cfg.momentum_scale,
                **role_rating_context,
            }
        )

        y = _y_arr[i]
        if pd.isna(y):
            continue
        y = float(y)
        intl = _is_intl(str(_league_arr[i] or ""), _tourn_arr[i])

        g10 = _g15_arr[i]
        if pd.isna(g10):
            g10 = _g10_arr[i]
        _len = _len_arr[i]
        if pd.notna(_len):
            length = float(_len)
        elif pd.notna(_glen_arr[i]):
            length = float(_glen_arr[i]) / 60.0
        else:
            length = 30.0
        mov = 1.0
        if pd.notna(g10) and length:
            mov = 1.0 + cfg.mov_scale * math.tanh(float(g10) / (200.0 * max(float(length), 1.0)))

        exp_b = p

        for name, role in blue_lu[:5]:
            st = states.setdefault(name, PlayerState(sigma=cfg.sigma0))
            k_scale = st.sigma / cfg.sigma0
            # Conservative per-player share of the shared team update.  Neutral
            # (1.0) whenever attribution is unavailable for this player-map.
            a_i = attribution.get((gid, "Blue", name), 1.0)
            if intl:
                st.mu_meta += cfg.k_meta * k_scale * mov * (y - exp_b) * a_i
            else:
                st.mu_regional += cfg.k_regional * k_scale * mov * (y - exp_b) * a_i
            st.sigma = max(cfg.sigma_min, st.sigma * 0.985)
            st.n_maps += 1
            if d is not None:
                st.last_date = d
            st.last_team = bt
            league = str(_league_arr[i] or "")
            if is_team_affiliation_league(league):
                st.home_league = league
            _append_momentum(st, y - p_base, cfg)
            states[name] = st
        for name, role in red_lu[:5]:
            st = states.setdefault(name, PlayerState(sigma=cfg.sigma0))
            k_scale = st.sigma / cfg.sigma0
            # Conservative per-player share of the shared team update.  Neutral
            # (1.0) whenever attribution is unavailable for this player-map.
            a_i = attribution.get((gid, "Red", name), 1.0)
            if intl:
                st.mu_meta += cfg.k_meta * k_scale * mov * ((1 - y) - (1 - exp_b)) * a_i
            else:
                st.mu_regional += (
                    cfg.k_regional * k_scale * mov * ((1 - y) - (1 - exp_b)) * a_i
                )
            st.sigma = max(cfg.sigma_min, st.sigma * 0.985)
            st.n_maps += 1
            if d is not None:
                st.last_date = d
            st.last_team = rt
            league = str(_league_arr[i] or "")
            if is_team_affiliation_league(league):
                st.home_league = league
            _append_momentum(st, (1 - y) - (1 - p_base), cfg)
            states[name] = st

        # Stability history: keep the last 10 posterior totals per player so
        # the snapshot can expose mean displacement per game.
        for name, role in list(blue_lu[:5]) + list(red_lu[:5]):
            if name in states:
                recent_mus.setdefault(name, []).append(total_mu(states[name]))
                recent_mus[name] = recent_mus[name][-10:]

    while target_idx < len(targets):
        target = targets[target_idx]
        checkpoints[target] = _snapshot_rows(states, cfg=cfg)
        target_idx += 1
    return pd.DataFrame(rows), states, checkpoints, recent_mus


# ---------------------------------------------------------------------------
# Research-only sequential baseline
# ---------------------------------------------------------------------------


class SequentialPlayerEloBaselineError(ValueError):
    """The research replay cannot prove a leakage-safe baseline."""


_SEQUENTIAL_BASELINE_SCHEMA = "scryglass:sequential-player-elo-baseline:v1"
_SEQUENTIAL_BASELINE_STRUCTURAL_PLAYER_COLUMNS = frozenset(
    {
        "game_uid",
        "gameid",
        "side",
        "position",
        "playername",
        "playerid",
        "teamid",
        "date",
        "league",
        "tournament",
        "competition_tier",
        "teamname",
        "blue_team",
        "red_team",
        "champion",
    }
)
_SEQUENTIAL_BASELINE_STRUCTURAL_MAP_COLUMNS = frozenset(
    {
        "game_uid",
        "gameid",
        "date",
        "blue_team",
        "red_team",
        "blue_teamname",
        "red_teamname",
        "league",
        "tournament",
        "competition_tier",
        "patch",
        "series_id",
    }
)


def _sequential_baseline_identity(values: object) -> str:
    """Hash canonical game IDs with the accepted-census convention."""

    if isinstance(values, pd.Series):
        source = values.tolist()
    elif isinstance(values, (str, bytes)):
        source = [values]
    else:
        try:
            source = list(values)  # type: ignore[arg-type]
        except TypeError:
            source = [values]
    ids = sorted(
        {
            str(game_id)
            for value in source
            if (game_id := canonical_source_game_key(value))
        }
    )
    return hashlib.sha256(("\n".join(ids) + "\n").encode("utf-8")).hexdigest()


def _sequential_baseline_timestamp(value: object) -> tuple[pd.Timestamp, str]:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise SequentialPlayerEloBaselineError("strict cutoff is not a timestamp") from error
    if pd.isna(stamp):
        raise SequentialPlayerEloBaselineError("strict cutoff is missing")
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    utc = stamp.tz_localize(None)
    return utc, stamp.isoformat().replace("+00:00", "Z")


def _sequential_baseline_source_receipt(
    source_receipt: Mapping[str, object] | None,
) -> tuple[str, str, tuple[str, ...]]:
    if not isinstance(source_receipt, Mapping):
        raise SequentialPlayerEloBaselineError("verified source receipt is required")
    required = (
        "source_as_of",
        "source_game_count",
        "source_identity_sha256",
        "accepted_game_ids",
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
        "source_files",
        "receipt_sha256",
    )
    if any(field not in source_receipt for field in required):
        raise SequentialPlayerEloBaselineError("source receipt binding is incomplete")
    receipt_hash = str(source_receipt.get("receipt_sha256") or "")
    eligible_hash = str(source_receipt.get("model_eligible_identity_sha256") or "")
    raw_ids = source_receipt.get("model_eligible_game_ids")
    if (
        len(receipt_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in receipt_hash)
        or len(eligible_hash) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in eligible_hash)
        or not isinstance(raw_ids, (list, tuple))
    ):
        raise SequentialPlayerEloBaselineError("source receipt binding is incomplete")
    payload = dict(source_receipt)
    payload.pop("receipt_sha256", None)
    try:
        canonical_payload = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SequentialPlayerEloBaselineError(
            "source receipt contains a non-canonical value"
        ) from error
    if hashlib.sha256(canonical_payload).hexdigest() != receipt_hash.lower():
        raise SequentialPlayerEloBaselineError(
            "source receipt hash does not match its payload"
        )
    raw_accepted_ids = source_receipt.get("accepted_game_ids")
    if not isinstance(raw_accepted_ids, (list, tuple)):
        raise SequentialPlayerEloBaselineError("source receipt accepted census is invalid")
    accepted_ids = tuple(
        sorted(
            {
                str(game_id)
                for value in raw_accepted_ids
                if (game_id := canonical_source_game_key(value))
            }
        )
    )
    eligible_ids = tuple(
        sorted(
            {
                str(game_id)
                for value in raw_ids
                if (game_id := canonical_source_game_key(value))
            }
        )
    )
    try:
        source_game_count = int(source_receipt["source_game_count"])
        eligible_game_count = int(source_receipt["model_eligible_game_count"])
    except (TypeError, ValueError) as error:
        raise SequentialPlayerEloBaselineError(
            "source receipt census count is invalid"
        ) from error
    if (
        not accepted_ids
        or len(accepted_ids) != len(raw_accepted_ids)
        or list(accepted_ids) != list(raw_accepted_ids)
        or source_game_count != len(accepted_ids)
        or _sequential_baseline_identity(accepted_ids)
        != str(source_receipt["source_identity_sha256"]).lower()
        or not eligible_ids
        or len(eligible_ids) != len(raw_ids)
        or list(eligible_ids) != list(raw_ids)
        or eligible_game_count != len(eligible_ids)
        or _sequential_baseline_identity(eligible_ids) != eligible_hash.lower()
    ):
        raise SequentialPlayerEloBaselineError(
            "source receipt census identity is invalid"
        )
    try:
        source_as_of = pd.Timestamp(source_receipt["source_as_of"])
    except (TypeError, ValueError) as error:
        raise SequentialPlayerEloBaselineError(
            "source receipt source_as_of is invalid"
        ) from error
    if pd.isna(source_as_of) or source_as_of.tzinfo is None:
        raise SequentialPlayerEloBaselineError(
            "source receipt source_as_of must include a timezone"
        )
    source_files = source_receipt.get("source_files")
    if not isinstance(source_files, Mapping) or not source_files:
        raise SequentialPlayerEloBaselineError("source receipt file binding is invalid")
    for label, record in source_files.items():
        if not isinstance(record, Mapping) or not isinstance(record.get("bytes"), int):
            raise SequentialPlayerEloBaselineError(
                f"source receipt file binding is invalid: {label}"
            )
        digest = str(record.get("sha256") or "")
        if len(digest) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in digest
        ):
            raise SequentialPlayerEloBaselineError(
                f"source receipt file binding is invalid: {label}"
            )
    return receipt_hash.lower(), eligible_hash.lower(), eligible_ids


def _validate_sequential_baseline_lineups(
    maps: pd.DataFrame,
    map_ids: pd.Series,
    players: pd.DataFrame,
    player_ids: pd.Series,
    requested_ids: set[str],
) -> None:
    """Require one exact, identity-complete five-player lineup per side."""

    blue_column = "blue_team" if "blue_team" in maps.columns else "blue_teamname"
    red_column = "red_team" if "red_team" in maps.columns else "red_teamname"
    if blue_column not in maps.columns or red_column not in maps.columns:
        raise SequentialPlayerEloBaselineError("maps have incomplete team identity")
    for index, game_id in map_ids.items():
        if str(game_id) not in requested_ids:
            continue
        blue_value = maps.loc[index, blue_column]
        red_value = maps.loc[index, red_column]
        blue = "" if pd.isna(blue_value) else str(blue_value).strip()
        red = "" if pd.isna(red_value) else str(red_value).strip()
        if (
            not blue
            or blue.casefold() in {"nan", "none", "<na>"}
            or not red
            or red.casefold() in {"nan", "none", "<na>"}
            or normalize_team(blue).casefold() == normalize_team(red).casefold()
        ):
            raise SequentialPlayerEloBaselineError(
                f"maps have incomplete team identity: {game_id}"
            )

    required_columns = {"side", "position", "playername", "playerid", "teamid"}
    missing_columns = sorted(required_columns - set(players.columns))
    if missing_columns:
        raise SequentialPlayerEloBaselineError(
            "player lineup identity columns are missing: " + ", ".join(missing_columns)
        )
    work = players.copy()
    work["_game_id"] = player_ids.astype(str).to_numpy()
    work = work[work["_game_id"].isin(requested_ids)].copy()
    work["_side"] = work["side"].astype("string").str.strip().str.title()
    work["_position"] = work["position"].astype("string").str.strip()
    work = work[work["_position"].str.casefold().ne("team")].copy()
    work["_role"] = work["_position"].map(_norm_role)
    expected_roles = {"top", "jng", "mid", "bot", "sup"}
    for game_id in sorted(requested_ids):
        game = work[work["_game_id"].eq(game_id)]
        problems: list[str] = []
        if len(game) != 10:
            problems.append("player_row_count_not_10")
        names = game["playername"].astype("string").str.strip()
        player_identity = game["playerid"].astype("string").str.strip()
        team_identity = game["teamid"].astype("string").str.strip()
        if (
            names.isna().any()
            or names.eq("").any()
            or names.str.casefold().isin({"nan", "none", "<na>"}).any()
            or names.nunique() != 10
        ):
            problems.append("player_name_identity_invalid")
        if (
            player_identity.isna().any()
            or not player_identity.str.startswith("oe:player:").all()
            or player_identity.nunique() != 10
        ):
            problems.append("stable_player_identity_invalid")
        if team_identity.isna().any() or not team_identity.str.startswith("oe:team:").all():
            problems.append("stable_team_identity_invalid")
        for side in ("Blue", "Red"):
            side_rows = game[game["_side"].eq(side)]
            if (
                len(side_rows) != 5
                or set(side_rows["_role"].astype(str)) != expected_roles
                or side_rows["playername"].astype(str).nunique() != 5
                or side_rows["playerid"].astype(str).nunique() != 5
                or side_rows["teamid"].astype(str).nunique() != 1
            ):
                problems.append(f"{side.casefold()}_exact_five_closure_invalid")
        if team_identity.nunique() != 2:
            problems.append("team_identity_closure_invalid")
        if problems:
            raise SequentialPlayerEloBaselineError(
                f"player lineup closure is invalid: {game_id}: "
                + ", ".join(sorted(set(problems)))
            )


def _sequential_baseline_game_ids(frame: pd.DataFrame) -> pd.Series:
    if "game_uid" in frame.columns:
        fallback = frame["gameid"] if "gameid" in frame.columns else None
        values = [
            canonical_source_game_key(
                value,
                fallback.loc[index] if fallback is not None else None,
            )
            for index, value in frame["game_uid"].items()
        ]
    elif "gameid" in frame.columns:
        values = [canonical_source_game_key(value) for value in frame["gameid"]]
    else:
        raise SequentialPlayerEloBaselineError("maps have no canonical game identity")
    result = pd.Series(values, index=frame.index, dtype="string")
    if result.isna().any() or result.str.strip().eq("").any():
        raise SequentialPlayerEloBaselineError("maps contain an empty game identity")
    return result


def _sequential_baseline_implementation_digest() -> str:
    functions = (
        build_sequential_player_elo_baseline,
        _sequential_baseline_source_receipt,
        _validate_sequential_baseline_lineups,
        _run_player_elo,
        _lineups_by_game,
        _aggregate,
        _snapshot_rows,
        expected_score,
        SequentialPlayerEloBaselineError,
    )
    source_parts: list[str] = []
    for function in functions:
        try:
            source_parts.append(inspect.getsource(function))
        except (OSError, TypeError):
            source_parts.append(repr(function))
    return hashlib.sha256("\n".join(source_parts).encode("utf-8")).hexdigest()


def _sequential_baseline_output_digest(output: pd.DataFrame) -> str:
    frame = output.reset_index(drop=True).copy()
    return _global_frame_digest(frame)


def build_sequential_player_elo_baseline(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    train_game_ids: Iterable[object],
    validation_game_ids: Iterable[object],
    strict_cutoff: object,
    source_receipt: Mapping[str, object] | None,
    cfg: PlayerEloConfig | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Replay the player Elo baseline for one research validation fold.

    The replay owns a fresh in-memory state.  It receives only the requested
    train and validation maps.  Validation outcomes and final player metrics
    are masked before the replay starts.  The returned rows contain the
    pre-map probability for each validation map.  No cache, directory, or
    production rating artifact is used.

    ``strict_cutoff`` is the validation boundary.  Every training map must
    have a date strictly before it.  Every validation map must be on or after
    it.  This leaves equal-timestamp rows on one side of the boundary and
    prevents a train row from seeing a validation outcome.
    """

    source_hash, eligible_hash, eligible_ids = _sequential_baseline_source_receipt(
        source_receipt
    )
    cutoff, cutoff_text = _sequential_baseline_timestamp(strict_cutoff)
    train_values = (
        [train_game_ids]
        if isinstance(train_game_ids, (str, bytes))
        else list(train_game_ids)
    )
    validation_values = (
        [validation_game_ids]
        if isinstance(validation_game_ids, (str, bytes))
        else list(validation_game_ids)
    )
    train_ids = tuple(
        sorted(
            {
                str(game_id)
                for value in train_values
                if (game_id := canonical_source_game_key(value))
            }
        )
    )
    validation_ids = tuple(
        sorted(
            {
                str(game_id)
                for value in validation_values
                if (game_id := canonical_source_game_key(value))
            }
        )
    )
    if not train_ids or not validation_ids:
        raise SequentialPlayerEloBaselineError("train and validation IDs must be non-empty")
    if set(train_ids) & set(validation_ids):
        raise SequentialPlayerEloBaselineError("train and validation IDs overlap")
    requested_ids = set(train_ids) | set(validation_ids)
    if not requested_ids.issubset(set(eligible_ids)):
        raise SequentialPlayerEloBaselineError(
            "train or validation IDs are outside model-eligible census"
        )

    maps = maps.copy().reset_index(drop=True)
    players = players.copy().reset_index(drop=True)
    map_ids = _sequential_baseline_game_ids(maps)
    if map_ids.duplicated().any():
        raise SequentialPlayerEloBaselineError("maps contain duplicate game identities")
    if "date" not in maps.columns:
        raise SequentialPlayerEloBaselineError("maps have no date column")
    selected_mask = map_ids.isin(requested_ids)
    selected_maps = maps.loc[selected_mask].copy()
    selected_ids = map_ids.loc[selected_mask]
    missing_source_ids = sorted(requested_ids - set(selected_ids.astype(str)))
    if missing_source_ids:
        raise SequentialPlayerEloBaselineError(
            "requested maps are missing: " + ", ".join(missing_source_ids)
        )
    selected_dates = pd.to_datetime(
        selected_maps["date"], utc=True, errors="coerce"
    ).dt.tz_localize(None)
    if selected_dates.isna().any():
        raise SequentialPlayerEloBaselineError("maps contain missing dates")
    train_date_mask = selected_ids.isin(train_ids)
    validation_date_mask = selected_ids.isin(validation_ids)
    if not bool((selected_dates.loc[train_date_mask] < cutoff).all()):
        raise SequentialPlayerEloBaselineError(
            "training maps are not strictly before the cutoff"
        )
    if not bool((selected_dates.loc[validation_date_mask] >= cutoff).all()):
        raise SequentialPlayerEloBaselineError(
            "validation maps are before the strict cutoff"
        )
    if (
        selected_dates.loc[train_date_mask].max()
        >= selected_dates.loc[validation_date_mask].min()
    ):
        raise SequentialPlayerEloBaselineError(
            "train and validation dates do not have a strict boundary"
        )
    if "y_blue_win" not in selected_maps.columns:
        raise SequentialPlayerEloBaselineError("maps have no y_blue_win outcome")
    train_outcomes = pd.to_numeric(
        selected_maps.loc[train_date_mask, "y_blue_win"], errors="coerce"
    )
    if train_outcomes.isna().any() or not train_outcomes.isin({0.0, 1.0}).all():
        raise SequentialPlayerEloBaselineError(
            "training maps have missing or invalid outcomes"
        )

    selected_maps["_research_game_id"] = selected_ids.astype(str).to_numpy()
    map_outcome_columns = [
        str(column)
        for column in selected_maps.columns
        if str(column) not in _SEQUENTIAL_BASELINE_STRUCTURAL_MAP_COLUMNS
        and str(column) != "_research_game_id"
    ]
    validation_row_mask = selected_maps["_research_game_id"].isin(validation_ids)
    for column in map_outcome_columns:
        values = selected_maps[column].astype(object)
        values.loc[validation_row_mask] = np.nan
        selected_maps[column] = values
    selected_maps = selected_maps.drop(columns=["_research_game_id"])

    player_ids = _sequential_baseline_game_ids(players)
    _validate_sequential_baseline_lineups(
        maps,
        map_ids,
        players,
        player_ids,
        requested_ids,
    )
    selected_players = players.loc[player_ids.isin(requested_ids)].copy()
    selected_player_ids = player_ids.loc[player_ids.isin(requested_ids)]
    missing_player_ids = sorted(requested_ids - set(selected_player_ids.astype(str)))
    if missing_player_ids:
        raise SequentialPlayerEloBaselineError(
            "player rows are missing for: " + ", ".join(missing_player_ids)
        )
    selected_players["_research_game_id"] = selected_player_ids.astype(str).to_numpy()
    validation_player_mask = selected_players["_research_game_id"].isin(validation_ids)
    player_mask_columns = [
        str(column)
        for column in selected_players.columns
        if str(column) not in _SEQUENTIAL_BASELINE_STRUCTURAL_PLAYER_COLUMNS
        and str(column) != "_research_game_id"
    ]
    for column in player_mask_columns:
        values = selected_players[column].astype(object)
        values.loc[validation_player_mask] = np.nan
        selected_players[column] = values
    selected_players = selected_players.drop(columns=["_research_game_id"])

    replay_cfg = cfg or PlayerEloConfig()
    # Deliberately omit ``baseline_cache``.  This keeps the fold isolated from
    # persistent state and makes the receipt describe this exact replay.
    replay, _states, _checkpoints, _recent_mus = _run_player_elo(
        selected_maps,
        selected_players,
        replay_cfg,
        baseline_cache=None,
    )
    if replay.empty or "game_uid" not in replay.columns:
        raise SequentialPlayerEloBaselineError("replay returned no map predictions")
    validation_output = replay[
        replay["game_uid"].astype(str).isin(validation_ids)
    ].copy()
    output_ids = validation_output["game_uid"].astype(str)
    missing_ids = sorted(set(validation_ids) - set(output_ids))
    if output_ids.duplicated().any():
        raise SequentialPlayerEloBaselineError(
            "replay returned duplicate validation predictions"
        )
    probability = pd.to_numeric(validation_output.get("p_player_elo"), errors="coerce")
    missing_ids.extend(
        validation_output.loc[~np.isfinite(probability.to_numpy(dtype=float)), "game_uid"]
        .astype(str)
        .tolist()
    )
    missing_ids = sorted(set(missing_ids))
    if missing_ids or len(validation_output) != len(validation_ids):
        raise SequentialPlayerEloBaselineError(
            "validation prediction coverage is incomplete: " + ", ".join(missing_ids)
        )
    validation_output = validation_output.sort_values(
        ["date", "game_uid"], kind="mergesort"
    ).reset_index(drop=True)
    output_digest = _sequential_baseline_output_digest(validation_output)
    receipt = {
        "schema_version": _SEQUENTIAL_BASELINE_SCHEMA,
        "source_receipt_sha256": source_hash,
        "model_eligible_identity_sha256": eligible_hash,
        "model_eligible_game_count": len(eligible_ids),
        "scope_game_count": len(requested_ids),
        "scope_game_identity_sha256": _sequential_baseline_identity(requested_ids),
        "train_game_count": len(train_ids),
        "train_game_ids": list(train_ids),
        "train_game_identity_sha256": _sequential_baseline_identity(train_ids),
        "validation_game_count": len(validation_ids),
        "validation_game_ids": list(validation_ids),
        "validation_game_identity_sha256": _sequential_baseline_identity(validation_ids),
        "strict_cutoff": cutoff_text,
        "rating_config": dict(replay_cfg.__dict__),
        "implementation_digest": _sequential_baseline_implementation_digest(),
        "output_rows": int(len(validation_output)),
        "output_sha256": output_digest,
        "missing_game_ids": missing_ids,
        "masked_map_columns": map_outcome_columns,
        "masked_player_columns": player_mask_columns,
        "state": "fresh_in_memory_replay",
        "writes_production_artifacts": False,
        "authority": {
            "research_only": True,
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "promotion": False,
            "deployment": False,
            "betting": False,
        },
    }
    return validation_output, receipt


def build_player_ratings(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
    output_dir: Path | None = None,
    player_records: Mapping[str, Mapping[str, object]] | None = None,
    checkpoint_dates: list[pd.Timestamp] | None = None,
    replay_out: dict[str, object] | None = None,
) -> pd.DataFrame:
    """Sequential player Elo; player ratings travel across org changes."""

    cfg = cfg or PlayerEloConfig()
    destination = Path(output_dir or FEATURES_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    baseline_cache_path = destination / "player_prefix_baseline_cache"
    cache_source_identity = _rating_source_identity(maps)
    cache_schema_fingerprint = _rating_cache_schema(players)
    baseline_cache = PrefixBaselineCache(
        storage_path=baseline_cache_path,
        source_identity=cache_source_identity,
        schema_fingerprint=cache_schema_fingerprint,
    )
    fit_cache = GlobalPlayerFitCache(
        storage_path=destination / "player_global_fit_cache",
    )
    out, states, checkpoints, recent_mus = _run_player_elo(
        maps,
        players,
        cfg,
        checkpoint_dates=checkpoint_dates,
        baseline_cache=baseline_cache,
    )
    attribution_stats = dict(LAST_ATTRIBUTION_STATS)
    path = destination / "player_ratings.parquet"
    out.to_parquet(path, index=False)

    bridge_context = None
    if replay_out is not None or player_records is not None:
        bridge_context = PlayerBridgeContext.build(players)
        bridge_context.bind_players(players)
    global_workspace = GlobalPlayerFitWorkspace.build(
        maps,
        players,
        baseline_cache=baseline_cache,
    )
    global_snapshot, global_meta = fit_global_player_bt(
        maps,
        players,
        baseline_cache=baseline_cache,
        fit_cache=fit_cache,
        fit_cache_slot="current",
        workspace=global_workspace,
    )
    snap = _apply_global_scale(
        _snapshot_rows(states, recent_mus, cfg), global_snapshot, cfg
    )
    if player_records is not None:
        for row in snap:
            record = player_records.get(str(row["player"]))
            if record is None:
                continue
            row["last_team"] = record.get("current_team")
            row["home_league"] = record.get("current_league") or "UNKNOWN"
        snap = _apply_bridge_uncertainty(
            snap,
            players,
            player_records,
            cfg,
            bridge_context=bridge_context,
        )
    snap_df = pd.DataFrame(snap).sort_values("mu_effective", ascending=False)
    snap_df.to_parquet(destination / "player_ratings_snapshot.parquet", index=False)
    if replay_out is not None:
        replay_out.clear()
        replay_out.update(
            {
                "source_identity": _replay_source_identity(maps, players),
                "config": dict(cfg.__dict__),
                "states": states,
                "checkpoints": checkpoints,
                "recent_mus": recent_mus,
                "current_global": global_snapshot.copy(deep=True),
                "current_global_meta": dict(global_meta),
                "global_workspace": global_workspace,
                "bridge_context": bridge_context,
                "baseline_cache": baseline_cache,
            }
        )
    (destination / "player_ratings_meta.json").write_text(
        json.dumps(
            {
                "n_maps": len(out),
                "n_players": len(snap),
                "config": cfg.__dict__,
                "momentum": selected_momentum_configuration(
                    window_games=cfg.momentum_window_games,
                    scale=cfg.momentum_scale,
                ),
                "registered_momentum": registered_momentum_bundle(),
                "global_rating": global_meta,
                "player_attribution": {
                    **attribution_stats,
                    "authority": {
                        "public": False,
                        "production": False,
                        "probability": False,
                        "recommendation": False,
                        "odds": False,
                        "ev": False,
                        "promotion": False,
                    },
                    "note": (
                        "Development-only reallocation of the shared team update. "
                        "Weights are an UNFITTED equal-weight default; protocol v5 "
                        "requires fitted weights and an independent acceptance "
                        "record before promotion. Multipliers average to exactly 1 "
                        "within a side, so the team aggregate update is unchanged."
                    ),
                },
                "note": (
                    "PUBLIC RESULTS RATING: one regularized Bradley-Terry fit uses every "
                    "accepted complete lineup on one connected scale. Competition tier is "
                    "not a bonus or penalty. The rating remains lineup-linked and does not "
                    "identify individual causal contribution."
                ),
            },
            indent=2,
        )
    )
    baseline_cache.flush()
    fit_cache.flush()
    print(f"[player_elo] wrote {path} n={len(out)} players={len(snap)}")
    return out


def _sunday_utc(as_of: pd.Timestamp | None) -> pd.Timestamp:
    stamp = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if stamp.tzinfo is not None:
        stamp = stamp.tz_convert("UTC").tz_localize(None)
    return stamp.normalize() - pd.Timedelta(days=(stamp.weekday() + 1) % 7)


def _recent_baseline_anchor(
    previous_as_of: pd.Timestamp | None,
    sunday_baseline: pd.Timestamp,
    cutoff: pd.Timestamp,
) -> pd.Timestamp:
    """Previous-refresh movement anchor with safe fallbacks.

    The CLI passes ISO strings with a trailing ``Z`` (tz-aware); the cutoff
    from the refresh is naive.  Normalize every input so the comparison is
    robust regardless of caller convention.
    """
    if previous_as_of is None:
        return sunday_baseline
    anchor = pd.Timestamp(previous_as_of)
    if anchor.tzinfo is not None:
        anchor = anchor.tz_convert("UTC").tz_localize(None)
    base = pd.Timestamp(sunday_baseline)
    if base.tzinfo is not None:
        base = base.tz_convert("UTC").tz_localize(None)
    cap = pd.Timestamp(cutoff)
    if cap.tzinfo is not None:
        cap = cap.tz_convert("UTC").tz_localize(None)
    if anchor >= cap or anchor < base - pd.Timedelta(days=400):
        return base
    return anchor


def weekly_replay_checkpoint_dates(
    as_of: pd.Timestamp | None,
    previous_as_of: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """Return the exact replay checkpoints used by weekly movement ranks."""

    week_start = _sunday_utc(as_of)
    previous_start = week_start - pd.Timedelta(days=7)
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    comparison_cutoffs = [
        cutoff - pd.DateOffset(months=1),
        cutoff - pd.DateOffset(months=3),
        cutoff - pd.DateOffset(months=12),
    ]
    recent_anchor = _recent_baseline_anchor(previous_as_of, previous_start, cutoff)
    return [recent_anchor, *comparison_cutoffs]


def build_player_weekly_ranks(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    cfg: PlayerEloConfig | None = None,
    *,
    output_dir: Path | None = None,
    as_of: pd.Timestamp | None = None,
    min_games: int = 20,
    player_records: Mapping[str, Mapping[str, object]] | None = None,
    previous_as_of: pd.Timestamp | None = None,
    replay: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Return current ranks with recent and calendar-month movement.

    The player ladder is still the current sequential Elo snapshot.  The
    recent movement baseline is the previous refresh's cutoff when
    ``previous_as_of`` is provided (so movement reflects every published
    cycle - after every batch of games), falling back to the prior Sunday
    00:00 UTC snapshot otherwise.  Calendar-month comparisons (1/3/12m)
    are unchanged.
    """

    cfg = cfg or PlayerEloConfig()
    destination = Path(output_dir or FEATURES_DIR)
    destination.mkdir(parents=True, exist_ok=True)
    baseline_cache_path = destination / "player_prefix_baseline_cache"
    cache_source_identity = _rating_source_identity(maps)
    cache_schema_fingerprint = _rating_cache_schema(players)
    fit_cache = GlobalPlayerFitCache(
        storage_path=destination / "player_global_fit_cache",
    )
    week_start = _sunday_utc(as_of)
    previous_start = week_start - pd.Timedelta(days=7)
    frame = maps.copy()
    frame["date"] = pd.to_datetime(frame.get("date"), errors="coerce", utc=True).dt.tz_localize(None)
    cutoff = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.now(tz="UTC")
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    comparison_cutoffs = {
        "1m": cutoff - pd.DateOffset(months=1),
        "3m": cutoff - pd.DateOffset(months=3),
        "12m": cutoff - pd.DateOffset(months=12),
    }
    if as_of is not None:
        frame = frame[frame["date"].le(cutoff)]

    recent_anchor = _recent_baseline_anchor(previous_as_of, previous_start, cutoff)
    required_checkpoints = [recent_anchor, *comparison_cutoffs.values()]
    replay_hit = False
    saved_workspace = None
    saved_bridge_context = None
    if replay is not None:
        saved_source = replay.get("source_identity")
        saved_config = replay.get("config")
        saved_states = replay.get("states")
        saved_checkpoints = replay.get("checkpoints")
        saved_recent_mus = replay.get("recent_mus")
        saved_global = replay.get("current_global")
        saved_workspace = replay.get("global_workspace")
        saved_bridge_context = replay.get("bridge_context")
        checkpoint_keys = set(saved_checkpoints) if isinstance(saved_checkpoints, dict) else set()
        replay_hit = bool(
            saved_source == _replay_source_identity(frame, players)
            and saved_config == dict(cfg.__dict__)
            and isinstance(saved_states, dict)
            and isinstance(saved_checkpoints, dict)
            and set(required_checkpoints).issubset(checkpoint_keys)
            and isinstance(saved_recent_mus, dict)
            and isinstance(saved_global, pd.DataFrame)
        )
    saved_baseline_cache = (
        replay.get("baseline_cache") if replay is not None else None
    )
    if (
        replay_hit
        and isinstance(saved_baseline_cache, PrefixBaselineCache)
        and saved_baseline_cache.source_identity == cache_source_identity
        and saved_baseline_cache.schema_fingerprint == cache_schema_fingerprint
        and saved_baseline_cache.storage_path is not None
        and saved_baseline_cache.storage_path.resolve() == baseline_cache_path.resolve()
    ):
        baseline_cache = saved_baseline_cache
    else:
        baseline_cache = PrefixBaselineCache(
            storage_path=baseline_cache_path,
            source_identity=cache_source_identity,
            schema_fingerprint=cache_schema_fingerprint,
        )
    global_workspace = None
    if isinstance(saved_workspace, GlobalPlayerFitWorkspace):
        if saved_workspace.matches_source(frame, players):
            global_workspace = saved_workspace
    if global_workspace is None:
        global_workspace = GlobalPlayerFitWorkspace.build(
            frame,
            players,
            baseline_cache=baseline_cache,
        )
    bridge_context = None
    if isinstance(saved_bridge_context, PlayerBridgeContext):
        if saved_bridge_context.matches_players(players):
            bridge_context = saved_bridge_context
    if bridge_context is None:
        bridge_context = PlayerBridgeContext.build(players)
    bridge_context.bind_players(players)
    if replay_hit:
        states = saved_states
        checkpoints = saved_checkpoints
        current_global = saved_global.copy(deep=True)
    else:
        _, states, checkpoints, _ = _run_player_elo(
            frame,
            players,
            cfg,
            checkpoint_dates=required_checkpoints,
            baseline_cache=baseline_cache,
        )
        current_global, _current_meta = fit_global_player_bt(
            frame,
            players,
            GlobalPlayerBTConfig(minimum_maps=1),
            through=cutoff,
            validate=False,
            baseline_cache=baseline_cache,
            fit_cache=fit_cache,
            fit_cache_slot="current",
            workspace=global_workspace,
        )
    # Current affiliation is the publication filter. Historical matches in a
    # different circuit remain evidence for the rating but cannot place a
    # developmental player in the current Tier 1 board.
    current_records = (
        dict(player_records)
        if player_records is not None
        else {
            player: {"current_tier": tier}
            for player, tier in _current_tier_records(
                bridge_context.canonical_players
            ).items()
        }
    )
    current_rows = _apply_bridge_uncertainty(
        _apply_global_scale(_snapshot_rows(states, cfg=cfg), current_global, cfg),
        players,
        current_records,
        cfg,
        through=cutoff,
        bridge_context=bridge_context,
    )
    def historical_rows(anchor: pd.Timestamp, anchor_label: str) -> list[dict[str, object]]:
        snapshot = checkpoints.get(anchor, [])
        if not snapshot:
            return []
        try:
            historical_cache = fit_cache if anchor_label == "recent" else None
            historical_cache_slot = anchor_label if anchor_label == "recent" else None
            historical_global, _historical_meta = fit_global_player_bt(
                frame,
                players,
                GlobalPlayerBTConfig(minimum_maps=1),
                through=anchor,
                validate=False,
                baseline_cache=baseline_cache,
                fit_cache=historical_cache,
                fit_cache_slot=historical_cache_slot,
                workspace=global_workspace,
            )
        except GlobalPlayerRatingError:
            return []
        return _apply_bridge_uncertainty(
            _apply_global_scale(snapshot, historical_global, cfg),
            players,
            current_records,
            cfg,
            through=anchor,
            bridge_context=bridge_context,
        )

    previous_rows = historical_rows(recent_anchor, "recent")
    comparison_rows = {
        label: historical_rows(anchor, label)
        for label, anchor in comparison_cutoffs.items()
    }
    current_tiers = {
        player: record.get("current_tier")
        for player, record in current_records.items()
    }

    def order(rows: list[dict[str, object]], scope: str) -> dict[str, int]:
        eligible = []
        for row in rows:
            player = str(row["player"])
            games = int(row.get("n_maps") or 0)
            tier = current_tiers.get(player)
            if int(row.get("global_connected") or 0) != 1:
                continue
            if games < max(1, int(min_games)):
                continue
            if scope != "all" and tier != scope:
                continue
            mu = float(row.get("mu_total") or 0)
            sigma = float(row.get("sigma") or 0)
            adjusted = mu - max(0.0, sigma - 28.0)
            eligible.append((adjusted, player))
        eligible.sort(key=lambda value: (-value[0], value[1].casefold()))
        return {player: rank for rank, (_, player) in enumerate(eligible, start=1)}

    scopes = ("all", "tier1", "tier2", "tier3")
    current_rank = {scope: order(current_rows, scope) for scope in scopes}
    previous_rank = {scope: order(previous_rows, scope) for scope in scopes}
    comparison_rank = {
        label: {scope: order(rows, scope) for scope in scopes}
        for label, rows in comparison_rows.items()
    }
    by_player: dict[str, dict[str, dict[str, object]]] = {}
    for player, rank in current_rank["all"].items():
        values: dict[str, dict[str, object]] = {}
        for scope in scopes:
            current = current_rank[scope].get(player)
            if current is None:
                continue
            prior = previous_rank[scope].get(player)
            position_deltas: dict[str, dict[str, object]] = {}
            for label, anchor in comparison_cutoffs.items():
                historical = comparison_rank[label][scope].get(player)
                position_deltas[label] = {
                    "as_of": f"{anchor.isoformat()}Z",
                    "rank": historical,
                    "delta": (historical - current) if historical is not None else None,
                }
            values[scope] = {
                "rank": current,
                "delta": (prior - current) if prior is not None else None,
                "position_deltas": position_deltas,
            }
        by_player[player] = values

    baseline_cache.flush()
    fit_cache.flush()
    return {
        "as_of": f"{week_start.isoformat()}Z",
        "previous_as_of": f"{recent_anchor.isoformat()}Z",
        "current_through": f"{cutoff.isoformat()}Z",
        "position_delta_as_of": {
            label: f"{anchor.isoformat()}Z"
            for label, anchor in comparison_cutoffs.items()
        },
        "min_games": int(min_games),
        "by_player": by_player,
        "note": "Rank movement compares the adjusted global player results rating with the previous refresh (or the prior Sunday when no earlier refresh exists) and the positions one, three, and twelve calendar months earlier. Positive delta means a climb.",
    }


def build_maps_frame_from_players(players: pd.DataFrame) -> pd.DataFrame:
    """One map row per OE game_uid from player rows (full history, not warehouse-filtered)."""
    pl = players.copy()
    if "game_uid" in pl.columns:
        fallback = pl["gameid"] if "gameid" in pl.columns else None
        pl["_gid"] = [
            canonical_source_game_key(value, fallback.loc[index] if fallback is not None else None)
            for index, value in pl["game_uid"].items()
        ]
    elif "gameid" in pl.columns:
        pl["_gid"] = pl["gameid"].map(canonical_source_game_key)
    else:
        return pd.DataFrame()
    pl = pl[pl["_gid"].notna() & pl["_gid"].str.strip().ne("")]
    pl["side"] = pl["side"].astype(str).str.title()
    pl["position"] = pl.get("position", pd.Series("", index=pl.index)).astype(str).str.lower()
    pl = pl[pl["position"] != "team"]
    blue = pl[pl["side"] == "Blue"].drop_duplicates("_gid")
    red = pl[pl["side"] == "Red"].drop_duplicates("_gid")
    m = blue[["_gid", "date", "league", "result", "teamname"]].rename(
        columns={"_gid": "game_uid", "result": "y_blue_win", "teamname": "blue_team"}
    )
    m = m.merge(
        red[["_gid", "teamname"]].rename(columns={"_gid": "game_uid", "teamname": "red_team"}),
        on="game_uid",
        how="inner",
    )
    m["y_blue_win"] = pd.to_numeric(m["y_blue_win"], errors="coerce")
    m["date"] = pd.to_datetime(m["date"], errors="coerce", utc=True).dt.tz_localize(None)
    return m.dropna(subset=["date", "y_blue_win"]).sort_values("date").reset_index(drop=True)


# Module caches — board hot path (avoid re-reading parquet / remapping teamnames)
_PLAYERS_CACHE: pd.DataFrame | None = None
_ROSTER_CACHE: dict[str, list[tuple[str, str]]] = {}


def load_players_cached() -> pd.DataFrame:
    global _PLAYERS_CACHE
    if _PLAYERS_CACHE is None:
        path = PARQUET_DIR / "players.parquet"
        _PLAYERS_CACHE = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return _PLAYERS_CACHE


@lru_cache(maxsize=1)
def _snapshot_by_player() -> dict:
    path = FEATURES_DIR / "player_ratings_snapshot.parquet"
    snap = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    by: dict = {}
    if not snap.empty:
        n_maps_col = (
            snap["n_maps"].fillna(0).astype(int)
            if "n_maps" in snap.columns
            else pd.Series([0] * len(snap))
        )
        last_team_col = (
            snap["last_team"] if "last_team" in snap.columns else pd.Series([None] * len(snap))
        )
        momentum_col = (
            snap["momentum_residual"].astype(float)
            if "momentum_residual" in snap.columns
            else pd.Series([0.0] * len(snap))
        )
        for player, mu_r, mu_m, sig, n_maps, last_team, momentum_residual in zip(
            snap["player"].astype(str),
            snap["mu_regional"].astype(float),
            snap["mu_meta"].astype(float),
            snap["sigma"].astype(float),
            n_maps_col,
            last_team_col,
            momentum_col,
        ):
            by[player] = PlayerState(
                mu_regional=float(mu_r),
                mu_meta=float(mu_m),
                sigma=float(sig),
                n_maps=int(n_maps),
                last_team=last_team,
                momentum_residual=float(momentum_residual),
            )
    return by


def score_player_lineups(
    blue_players: list[str],
    red_players: list[str],
    *,
    blue_roles: list[str] | None = None,
    red_roles: list[str] | None = None,
    snapshot: pd.DataFrame | None = None,
    cfg: PlayerEloConfig | None = None,
) -> dict:
    """Score a concrete roster from the player-rating snapshot (roster moves travel)."""
    cfg = cfg or PlayerEloConfig()
    if snapshot is None:
        by = _snapshot_by_player()
    else:
        by = {}
        if not snapshot.empty:
            by = {
                str(r["player"]): PlayerState(
                    mu_regional=float(r["mu_regional"]),
                    mu_meta=float(r["mu_meta"]),
                    sigma=float(r["sigma"]),
                    n_maps=int(r.get("n_maps") or 0),
                    last_team=r.get("last_team"),
                    momentum_residual=float(r.get("momentum_residual") or 0.0),
                )
                for _, r in snapshot.iterrows()
            }
    br = blue_roles or ["top", "jng", "mid", "bot", "sup"]
    rr = red_roles or ["top", "jng", "mid", "bot", "sup"]
    blu = list(zip([str(x) for x in blue_players], br[: len(blue_players)]))
    red = list(zip([str(x) for x in red_players], rr[: len(red_players)]))
    base_mu_b, sig_b, known_b, _ = _aggregate(by, blu, cfg, include_momentum=False)
    base_mu_r, sig_r, known_r, _ = _aggregate(by, red, cfg, include_momentum=False)
    mu_b, _, _, det_b = _aggregate(by, blu, cfg)
    mu_r, _, _, det_r = _aggregate(by, red, cfg)
    sig = math.hypot(sig_b, sig_r)
    mu_diff = mu_b - mu_r
    p = expected_score(mu_b, mu_r)
    shrink = 1.0 / (1.0 + (sig / 130.0) ** 2)
    p_shrunk = 0.5 + (p - 0.5) * shrink
    # Prefer time-safe Elo→WR calibration when available (avoids hot player scale)
    try:
        from lol_kills.ratings.calibrate_elo_wr import calibrated_player_p, load_calibration

        cal = load_calibration()
        if cal.get("player"):
            p_cal = calibrated_player_p(mu_diff, cal)
            p_shrunk = 0.5 + (p_cal - 0.5) * shrink
    except Exception:
        # Calibration is optional. Keep the descriptive uncalibrated score.
        pass
    return {
        "player_mu_blue": round(mu_b, 2),
        "player_mu_red": round(mu_r, 2),
        "player_mu_diff": round(mu_diff, 2),
        "player_mu_base_blue": round(base_mu_b, 2),
        "player_mu_base_red": round(base_mu_r, 2),
        "player_momentum_blue": round(mu_b - base_mu_b, 2),
        "player_momentum_red": round(mu_r - base_mu_r, 2),
        "player_momentum_diff": round((mu_b - base_mu_b) - (mu_r - base_mu_r), 2),
        "player_momentum_window_games": cfg.momentum_window_games,
        "player_momentum_scale": cfg.momentum_scale,
        "p_player_elo": round(p_shrunk, 4),
        "player_known_blue": known_b,
        "player_known_red": known_r,
        "blue_detail": det_b,
        "red_detail": det_r,
    }


def latest_roster_for_team(players: pd.DataFrame, team: str, n: int = 5) -> list[tuple[str, str]]:
    """Most recent 5-man roster (name, role) for a team from OE player rows."""
    team_n = normalize_team(team)
    # Normalize unique teamnames once (not every row) — board hot path.
    uniq = players["teamname"].dropna().astype(str).unique()
    mapping = {t: normalize_team(t) for t in uniq}
    want_raw = {t for t, nrm in mapping.items() if nrm == team_n}
    if not want_raw:
        return []
    p = players[players["teamname"].astype(str).isin(want_raw)]
    p = p[p["position"].astype(str).str.lower() != "team"]
    p = p[p["playername"].notna()]
    if p.empty or "date" not in p.columns:
        return []
    dates = pd.to_datetime(p["date"], errors="coerce")
    last_idx = dates.idxmax()
    if pd.isna(last_idx):
        return []
    last_gid = str(p.loc[last_idx, "game_uid"])
    g = p[p["game_uid"].astype(str) == last_gid].copy()
    g["_role"] = g["position"].map(_norm_role)
    order = {"top": 0, "jng": 1, "mid": 2, "bot": 3, "sup": 4}
    rows = [(str(r["playername"]), str(r["_role"])) for _, r in g.iterrows()]
    rows.sort(key=lambda x: order.get(x[1], 9))
    seen = set()
    out = []
    for name, role in rows:
        if role in seen:
            continue
        seen.add(role)
        out.append((name, role))
    return out[:n]


def latest_roster_cached(team: str, n: int = 5) -> list[tuple[str, str]]:
    key = f"{normalize_team(team)}|{n}"
    if key not in _ROSTER_CACHE:
        _ROSTER_CACHE[key] = latest_roster_for_team(load_players_cached(), team, n=n)
    return _ROSTER_CACHE[key]


def main() -> None:
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    maps_all = build_maps_frame_from_players(players)
    print(f"[player_elo] full OE maps={len(maps_all)}")
    build_player_ratings(maps_all, players)


if __name__ == "__main__":
    main()
