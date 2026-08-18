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
_ANCHOR_ZERO_MEAN_TOLERANCE = 1e-9


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


def _contribution_metrics(players: pd.DataFrame) -> pd.DataFrame:
    """Per player-map contribution metrics used to build the ridge anchor.

    Every metric is fail-closed: a missing column, a missing value, or an
    impossible denominator yields NaN for that metric on that map. Nothing is
    imputed and no league or role mean is ever substituted.
    """

    required = {"side", "position", "playername"}
    if players is None or players.empty or not required.issubset(players.columns):
        return pd.DataFrame()

    frame = pd.DataFrame(index=players.index)
    frame["_game_id"] = _canonical_game_ids(players)
    frame["_side"] = players["side"].astype(str).str.title()
    frame["_role"] = players["position"].map(_role)
    frame["_player"] = players["playername"].astype("string").str.strip()
    if "competition_tier" in players.columns:
        frame["_tier"] = players["competition_tier"].astype("string").str.strip().str.casefold()
    else:
        frame["_tier"] = pd.Series(pd.NA, index=players.index, dtype="string")
    for source in ("gamelength", "totalgold", "cspm", "dpm", "damageshare", "kills", "deaths", "assists", "wpm", "wcpm"):
        frame[f"_raw_{source}"] = _numeric(players, source)

    frame = frame[
        frame["_game_id"].notna()
        & frame["_game_id"].astype(str).str.strip().ne("")
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

    keep = ["_game_id", "_side", "_role", "_player", "_tier", *PERFORMANCE_ANCHOR_METRIC_WEIGHTS]
    metrics = frame[keep].copy()
    for metric in PERFORMANCE_ANCHOR_METRIC_WEIGHTS:
        values = metrics[metric].astype(float)
        metrics[metric] = values.where(np.isfinite(values))
    return metrics


def _role_normalized_composite(metrics: pd.DataFrame) -> tuple[pd.Series, str]:
    """Weighted mean of within-(role, tier) z-scores for each player-map row.

    Role normalization is what makes the anchor fair: a support's 0.87 cs/min
    is normal for a support and must not read as a bad performance.
    """

    group_keys = ["_role"]
    normalization = "role"
    if "_tier" in metrics.columns and metrics["_tier"].notna().any():
        group_keys = ["_role", "_tier"]
        normalization = "role+competition_tier"
    keys = [metrics[key] for key in group_keys]

    weighted_sum = pd.Series(0.0, index=metrics.index)
    weight_total = pd.Series(0.0, index=metrics.index)
    for metric, weight in PERFORMANCE_ANCHOR_METRIC_WEIGHTS.items():
        if weight <= 0.0 or metric not in metrics.columns:
            continue
        values = metrics[metric]
        grouped = values.groupby(keys, dropna=False)
        # A group with fewer than two observed values has an undefined spread,
        # so its z-score stays NaN rather than collapsing to a guessed zero.
        spread = grouped.transform("std")
        centered = values - grouped.transform("mean")
        z = centered / spread.where(spread > 0)
        z = z.where(np.isfinite(z))
        observed = z.notna()
        weighted_sum = weighted_sum + z.where(observed, 0.0) * weight
        weight_total = weight_total + observed.astype(float) * weight
    composite = weighted_sum / weight_total.where(weight_total > 0)
    return composite.where(np.isfinite(composite)), normalization


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

    composite, normalization = _role_normalized_composite(scoped)
    evidence["normalization"] = normalization
    evidence["player_map_rows_used"] = int(composite.notna().sum())
    per_player = composite.groupby(scoped["_player"].astype(str)).mean()
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
    metrics = _contribution_metrics(players) if cfg.performance_anchor_enabled else pd.DataFrame()
    anchor, anchored, anchor_evidence = _performance_anchor(metrics, names, set(game_ids), cfg)
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
