"""Leakage-safe, shrunk player--champion features for draft research.

These features are predictive context, not a causal measure of champion
mastery.  A player--champion result is affected by team quality, opponents,
side, patch, and pick selection.  The state therefore reports a conservative
residual with an explicit support weight instead of exposing a raw win rate.

The state is updated in event order.  Games with the same timestamp are scored
from the same state and only then applied, so a result cannot leak sideways
into another map at the same event time.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd


ROLES = ("top", "jng", "mid", "bot", "sup")
PLAYER_CHAMPION_FEATURES = (
    "player_champion_edge",
    "player_champion_experience",
    "player_role_edge",
    "ally_interaction_edge",
    "enemy_interaction_edge",
    "ally_interaction_experience",
    "enemy_interaction_experience",
)
DEFAULT_HALF_LIFE_DAYS = 365
DEFAULT_PLAYER_CHAMPION_PRIOR = 8.0
DEFAULT_PLAYER_ROLE_PRIOR = 12.0
DEFAULT_CHAMPION_ROLE_PRIOR = 18.0
DEFAULT_INTERACTION_PRIOR = 16.0


@dataclass
class _Stats:
    games: float = 0.0
    wins: float = 0.0


def _timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _clip_probability(value: float) -> float:
    return max(0.01, min(0.99, float(value)))


def _logit(probability: float) -> float:
    p = _clip_probability(probability)
    return math.log(p / (1.0 - p))


def _sigmoid(value: float) -> float:
    if value >= 35:
        return 1.0
    if value <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _posterior(stats: _Stats, prior_probability: float, prior_games: float) -> float:
    denominator = stats.games + prior_games
    if denominator <= 0:
        return _clip_probability(prior_probability)
    return _clip_probability(
        (stats.wins + prior_games * prior_probability) / denominator
    )


class PlayerChampionState:
    """Time-decayed hierarchical state evaluated strictly before each map."""

    def __init__(
        self,
        *,
        half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
        player_champion_prior: float = DEFAULT_PLAYER_CHAMPION_PRIOR,
        player_role_prior: float = DEFAULT_PLAYER_ROLE_PRIOR,
        champion_role_prior: float = DEFAULT_CHAMPION_ROLE_PRIOR,
        interaction_prior: float = DEFAULT_INTERACTION_PRIOR,
    ) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive")
        self.half_life_days = int(half_life_days)
        self.player_champion_prior = float(player_champion_prior)
        self.player_role_prior = float(player_role_prior)
        self.champion_role_prior = float(champion_role_prior)
        self.interaction_prior = float(interaction_prior)
        self._pc: defaultdict[tuple[str, str, str], _Stats] = defaultdict(_Stats)
        self._player_role: defaultdict[tuple[str, str], _Stats] = defaultdict(_Stats)
        self._champion_role: defaultdict[tuple[str, str], _Stats] = defaultdict(_Stats)
        self._role: defaultdict[str, _Stats] = defaultdict(_Stats)
        self._interaction: defaultdict[
            tuple[str, str, str, str, str], _Stats
        ] = defaultdict(_Stats)
        self._last_timestamp: pd.Timestamp | None = None

    @staticmethod
    def _player(pick: Mapping[str, Any]) -> str:
        return str(pick.get("player") or "").strip().casefold()

    @staticmethod
    def _champion(pick: Mapping[str, Any]) -> str:
        return str(pick.get("champion") or "").strip()

    def _decay_to(self, timestamp: pd.Timestamp) -> None:
        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            return
        days = max(0.0, (timestamp - self._last_timestamp).total_seconds() / 86400.0)
        if days <= 0:
            return
        factor = 0.5 ** (days / self.half_life_days)
        for mapping in (
            self._pc,
            self._player_role,
            self._champion_role,
            self._role,
        ):
            for stats in mapping.values():
                stats.games *= factor
                stats.wins *= factor
        self._last_timestamp = timestamp

    def _pick_features(
        self,
        *,
        player: str,
        role: str,
        champion: str,
        others: list[tuple[str, str, str]],
    ) -> dict[str, float | int | str]:
        role_stats = self._role[role]
        role_probability = _posterior(role_stats, 0.5, 24.0)
        player_role_stats = self._player_role[(player, role)]
        champion_role_stats = self._champion_role[(champion, role)]
        player_role_probability = _posterior(
            player_role_stats,
            role_probability,
            self.player_role_prior,
        )
        champion_role_probability = _posterior(
            champion_role_stats,
            role_probability,
            self.champion_role_prior,
        )
        baseline_logit = 0.5 * (
            _logit(player_role_probability) + _logit(champion_role_probability)
        )
        baseline_probability = _sigmoid(baseline_logit)
        pc_stats = self._pc[(player, role, champion)]
        player_champion_probability = _posterior(
            pc_stats,
            baseline_probability,
            self.player_champion_prior,
        )
        reliability = pc_stats.games / (
            pc_stats.games + self.player_champion_prior
        )
        residual_logit = (
            _logit(player_champion_probability) - baseline_logit
        ) * reliability
        residual_logit = max(-1.0, min(1.0, residual_logit))
        player_role_reliability = player_role_stats.games / (
            player_role_stats.games + self.player_role_prior
        )
        player_role_edge = (
            _logit(player_role_probability) - _logit(role_probability)
        ) * player_role_reliability
        player_role_edge = max(-1.0, min(1.0, player_role_edge))
        interaction_rows: list[dict[str, Any]] = []
        interaction_edges = {"ally": 0.0, "enemy": 0.0}
        interaction_experience = {"ally": 0.0, "enemy": 0.0}
        for relation, other_role, other_champion in others:
            stats = self._interaction[
                (player, role, champion, relation, other_champion)
            ]
            interaction_probability = _posterior(
                stats,
                player_champion_probability,
                self.interaction_prior,
            )
            interaction_reliability = stats.games / (
                stats.games + self.interaction_prior
            )
            interaction_residual = (
                _logit(interaction_probability)
                - _logit(player_champion_probability)
            ) * interaction_reliability
            interaction_residual = max(-0.75, min(0.75, interaction_residual))
            experience = math.log1p(stats.games)
            interaction_edges[relation] += interaction_residual
            interaction_experience[relation] += experience
            interaction_rows.append(
                {
                    "relation": relation,
                    "other_role": other_role,
                    "other_champion": other_champion,
                    "games": round(float(stats.games), 6),
                    "player_champion_probability": round(
                        float(player_champion_probability), 6
                    ),
                    "interaction_probability": round(
                        float(interaction_probability), 6
                    ),
                    "reliability": round(float(interaction_reliability), 6),
                    "residual_logit": round(float(interaction_residual), 6),
                }
            )
        return {
            "player": player,
            "role": role,
            "champion": champion,
            "games": round(float(pc_stats.games), 6),
            "player_role_games": round(float(player_role_stats.games), 6),
            "baseline_probability": round(float(baseline_probability), 6),
            "player_champion_probability": round(
                float(player_champion_probability), 6
            ),
            "reliability": round(float(reliability), 6),
            "residual_logit": round(float(residual_logit), 6),
            "player_role_edge": round(float(player_role_edge), 6),
            "experience": round(float(math.log1p(pc_stats.games)), 6),
            "ally_interaction_edge": interaction_edges["ally"],
            "enemy_interaction_edge": interaction_edges["enemy"],
            "ally_interaction_experience": interaction_experience["ally"],
            "enemy_interaction_experience": interaction_experience["enemy"],
            "interactions": interaction_rows,
        }

    def score(self, game: Mapping[str, Any]) -> dict[str, Any]:
        """Return aggregate and per-pick features using state before ``game``."""

        self._decay_to(_timestamp(game["date"]))
        aggregate = {name: 0.0 for name in PLAYER_CHAMPION_FEATURES}
        breakdown: list[dict[str, Any]] = []
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            for role in ROLES:
                pick = game[side][role]
                player = self._player(pick)
                champion = self._champion(pick)
                if not player or not champion:
                    continue
                others: list[tuple[str, str, str]] = []
                for other_role in ROLES:
                    if other_role != role:
                        other_champion = self._champion(game[side][other_role])
                        if other_champion:
                            others.append(("ally", other_role, other_champion))
                other_side = "red" if side == "blue" else "blue"
                for other_role in ROLES:
                    other_champion = self._champion(game[other_side][other_role])
                    if other_champion:
                        others.append(("enemy", other_role, other_champion))
                row = self._pick_features(
                    player=player,
                    role=role,
                    champion=champion,
                    others=others,
                )
                aggregate["player_champion_edge"] += sign * float(
                    row["residual_logit"]
                )
                aggregate["player_champion_experience"] += sign * float(
                    row["experience"]
                )
                aggregate["player_role_edge"] += sign * float(
                    row["player_role_edge"]
                )
                for name in (
                    "ally_interaction_edge",
                    "enemy_interaction_edge",
                    "ally_interaction_experience",
                    "enemy_interaction_experience",
                ):
                    aggregate[name] += sign * float(row[name])
                breakdown.append(
                    {
                        "side": side,
                        "role": role,
                        "player": row["player"],
                        "champion": row["champion"],
                        "games": row["games"],
                        "baseline_probability": row["baseline_probability"],
                        "player_champion_probability": row[
                            "player_champion_probability"
                        ],
                        "reliability": row["reliability"],
                        "residual_logit": row["residual_logit"],
                        "signed_residual_logit": round(
                            sign * float(row["residual_logit"]),
                            6,
                        ),
                        "interactions": [
                            {
                                **interaction,
                                "signed_residual_logit": round(
                                    sign * float(interaction["residual_logit"]),
                                    6,
                                ),
                            }
                            for interaction in row["interactions"]
                        ],
                    }
                )
        return {
            **{key: float(value) for key, value in aggregate.items()},
            "breakdown": breakdown,
        }

    def update(self, game: Mapping[str, Any]) -> None:
        """Apply the map result after all same-time maps have been scored."""

        timestamp = _timestamp(game["date"])
        self._decay_to(timestamp)
        y = float(game["y"])
        for side, side_result in (("blue", y), ("red", 1.0 - y)):
            for role in ROLES:
                pick = game[side][role]
                player = self._player(pick)
                champion = self._champion(pick)
                if not player or not champion:
                    continue
                for stats in (
                    self._pc[(player, role, champion)],
                    self._player_role[(player, role)],
                    self._champion_role[(champion, role)],
                    self._role[role],
                ):
                    stats.games += 1.0
                    stats.wins += side_result
                for other_role in ROLES:
                    if other_role != role:
                        other_champion = self._champion(game[side][other_role])
                        if other_champion:
                            stats = self._interaction[
                                (
                                    player,
                                    role,
                                    champion,
                                    "ally",
                                    other_champion,
                                )
                            ]
                            stats.games += 1.0
                            stats.wins += side_result
                other_side = "red" if side == "blue" else "blue"
                for other_role in ROLES:
                    other_champion = self._champion(game[other_side][other_role])
                    if other_champion:
                        stats = self._interaction[
                            (
                                player,
                                role,
                                champion,
                                "enemy",
                                other_champion,
                            )
                        ]
                        stats.games += 1.0
                        stats.wins += side_result


def player_champion_feature_rows(
    games: list[dict[str, Any]],
    *,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS,
) -> dict[str, dict[str, Any]]:
    """Build pre-map player--champion rows for an ordered game population."""

    ordered = sorted(games, key=lambda game: (game["date"], game["game_uid"]))
    state = PlayerChampionState(half_life_days=half_life_days)
    result: dict[str, dict[str, Any]] = {}
    index = 0
    while index < len(ordered):
        end = index + 1
        timestamp = ordered[index]["date"]
        while end < len(ordered) and ordered[end]["date"] == timestamp:
            end += 1
        for game in ordered[index:end]:
            result[str(game["game_uid"])] = state.score(game)
        for game in ordered[index:end]:
            state.update(game)
        index = end
    return result
