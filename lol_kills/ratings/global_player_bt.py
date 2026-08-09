"""One connected, regularized player results rating.

The fit uses only completed map results and verified ten-player lineups. Each
player has one coefficient across every league, team, and competition tier.
Roster transfers and cross-circuit matches connect domestic pools. Competition
tier is never used as a rating bonus or penalty.
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


def _fit(
    design: csr_matrix,
    outcome: np.ndarray,
    cfg: GlobalPlayerBTConfig,
) -> tuple[np.ndarray, float]:
    side = csr_matrix(np.ones((design.shape[0], 1), dtype=float))
    matrix = hstack([design, side], format="csr")
    penalty = np.concatenate(
        [
            np.full(design.shape[1], cfg.l2, dtype=float),
            np.asarray([cfg.side_l2], dtype=float),
        ]
    )

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        logits = np.asarray(matrix @ parameters).reshape(-1)
        residual = 1.0 / (1.0 + np.exp(-np.clip(logits, -35, 35))) - outcome
        loss = float(np.logaddexp(0.0, logits).sum() - np.dot(outcome, logits))
        loss += 0.5 * float(np.dot(penalty, parameters**2))
        gradient = np.asarray(matrix.T @ residual).reshape(-1) + penalty * parameters
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
        train_coefficients, train_side = _fit(train_x, train_y, cfg)
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

    coefficients, side_advantage = _fit(design, outcome, cfg)
    appearances: dict[str, int] = {name: 0 for name in names}
    for game_id in frame["game_id"].astype(str):
        for side in ("Blue", "Red"):
            for player, _ in lineups[game_id][side]:
                appearances[player] += 1
    rows = []
    for name, coefficient in zip(names, coefficients):
        root = roots[name]
        rows.append(
            {
                "player": name,
                "global_rating": cfg.prior_rating + LOGIT_TO_ELO * float(coefficient),
                "global_logit": float(coefficient),
                "global_connected": int(root == largest),
                "global_component_id": str(root),
                "global_component_size": int(component_sizes[root]),
                "global_model_maps": int(appearances[name]),
            }
        )
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
        "player_statistics_used": False,
    }
    return snapshot.reset_index(drop=True), meta
