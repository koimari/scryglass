"""Full-covariance prequential development for multi-league Player Rating.

This runner consumes only :mod:`multileague_development` inputs.  It evaluates
fixed temporal candidates with a Bernoulli-logistic rank-one Laplace/ADF
update, a full Gaussian covariance matrix, series-frozen forecasts, and a
strict 48-hour post-series availability embargo.  Domestic league effects are
identified only through MSI/EWC bridge matches whose home-league identity was
available before the match.

The output is non-authorizing development evidence.  It does not open sealed
targets, establish pre-event rosters, or emit usable match probabilities,
odds, expected value, or betting recommendations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import heapq
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from lol_kills.v2.evaluation.bootstrap import BootstrapResult, series_cluster_bootstrap

from .model import DISPLAY_ANCHOR, DISPLAY_LOGIT_SCALE, posterior_predictive_expected_result
from .multileague_development import (
    CLAIM_CEILING as INPUT_CLAIM_CEILING,
    DOMESTIC_LEAGUES,
    LEAGUES,
    ROLE_ORDER,
    SEALED_FINAL_START,
    DevelopmentMap,
    DevelopmentSeries,
    ObservedLineup,
    PrivateMultiLeagueRatingInput,
    load_multileague_development_input,
    source_local_datetime,
)


SCHEMA_VERSION = "scryglass:multileague-player-rating-development:v1"
AVAILABILITY_EMBARGO_HOURS = 48
MINIMUM_VALIDATION_SERIES = 20
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20260801
MINIMUM_VARIANCE = 1e-10
PSD_TOLERANCE = 1e-8

PLAYER_PRIOR_VARIANCE = 1.0
LEAGUE_PRIOR_VARIANCE = 0.25
BLUE_SIDE_PRIOR_VARIANCE = 0.25
RANDOM_WALK_VARIANCE_PER_DAY = 0.0005
MEAN_REVERSION_HALF_LIFE_DAYS = 120.0


class MultiLeagueRunnerError(ValueError):
    """A development replay or artifact violated its fail-closed contract."""


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    kind: str
    process_variance_per_day: float
    half_life_days: float | None = None

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "kind": self.kind,
            "process_variance_per_day": self.process_variance_per_day,
            "half_life_days": self.half_life_days,
            "reset_policy": "NO_RESET",
            "outcome_update": "BERNOULLI_LOGISTIC_FULL_COVARIANCE_RANK_ONE_LAPLACE_ADF",
        }


CANDIDATES = (
    Candidate("static_no_reset", "STATIC", 0.0),
    Candidate("random_walk_no_reset", "RANDOM_WALK", RANDOM_WALK_VARIANCE_PER_DAY),
    Candidate(
        "mean_reversion_no_reset",
        "MEAN_REVERSION",
        RANDOM_WALK_VARIANCE_PER_DAY,
        MEAN_REVERSION_HALF_LIFE_DAYS,
    ),
)


@dataclass(frozen=True)
class _FeatureVector:
    weights: Mapping[str, float]
    blue_home_league: str | None
    red_home_league: str | None
    bridge_status: str


@dataclass(frozen=True)
class _PendingSeries:
    available_at: datetime
    series: DevelopmentSeries
    features: tuple[_FeatureVector, ...]


@dataclass
class _ReplayResult:
    candidate: Candidate
    predictions: list[dict[str, Any]]
    state: "_GaussianState"
    player_metadata: dict[str, dict[str, Any]]
    team_lineups: dict[str, dict[str, Any]]
    team_home_leagues: dict[str, str]
    bridge_diagnostics: dict[str, int]
    applied_series: int
    applied_maps: int
    ordered_applied_series: list[str]


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise MultiLeagueRunnerError("canonical artifact contains a non-finite value") from error


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256(_canonical_bytes(value))


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MultiLeagueRunnerError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise MultiLeagueRunnerError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MultiLeagueRunnerError(f"{label} must be finite")
    return result


def _sigmoid(value: float) -> float:
    if not math.isfinite(value):
        raise MultiLeagueRunnerError("conditional logit is non-finite")
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    numerator = math.exp(value)
    return numerator / (1.0 + numerator)


def _player_key(player_id: str) -> str:
    return f"player:{player_id}"


def _league_key(league: str) -> str:
    return f"league:{league}"


BLUE_SIDE_KEY = "global:blue-side"


class _GaussianState:
    """Mutable exact Gaussian state for the bounded development replay."""

    def __init__(self, candidate: Candidate):
        self.candidate = candidate
        self.keys: list[str] = []
        self.index: dict[str, int] = {}
        self.kinds: list[str] = []
        self.last_at: list[datetime | None] = []
        self.evidence_counts: dict[str, int] = {}
        self.mean = np.zeros(0, dtype=float)
        self.covariance = np.zeros((0, 0), dtype=float)

    def clone(self) -> "_GaussianState":
        value = _GaussianState(self.candidate)
        value.keys = list(self.keys)
        value.index = dict(self.index)
        value.kinds = list(self.kinds)
        value.last_at = list(self.last_at)
        value.evidence_counts = dict(self.evidence_counts)
        value.mean = self.mean.copy()
        value.covariance = self.covariance.copy()
        return value

    def ensure(
        self,
        key: str,
        *,
        kind: str,
        prior_variance: float,
        at: datetime | None = None,
    ) -> int:
        if key in self.index:
            index = self.index[key]
            if self.kinds[index] != kind:
                raise MultiLeagueRunnerError("latent key kind changed during replay")
            return index
        if not key or kind not in {"player", "league", "blue_side"}:
            raise MultiLeagueRunnerError("invalid latent key")
        prior_variance = _finite(prior_variance, "prior variance")
        if prior_variance <= 0.0:
            raise MultiLeagueRunnerError("prior variance must be positive")
        old = len(self.keys)
        expanded = np.zeros((old + 1, old + 1), dtype=float)
        if old:
            expanded[:old, :old] = self.covariance
        expanded[old, old] = prior_variance
        self.covariance = expanded
        self.mean = np.append(self.mean, 0.0)
        self.keys.append(key)
        self.index[key] = old
        self.kinds.append(kind)
        self.last_at.append(at if kind == "player" else None)
        self.evidence_counts[key] = 0
        return old

    def _transition_player(self, index: int, target: datetime) -> None:
        previous = self.last_at[index]
        if previous is None:
            self.last_at[index] = target
            return
        days = (target - previous).total_seconds() / 86400.0
        if days < -1e-12:
            raise MultiLeagueRunnerError("player state attempted to move backward in source-naive time")
        days = max(days, 0.0)
        candidate = self.candidate
        if candidate.kind == "RANDOM_WALK":
            self.covariance[index, index] += candidate.process_variance_per_day * days
        elif candidate.kind == "MEAN_REVERSION":
            if candidate.half_life_days is None or candidate.half_life_days <= 0.0:
                raise MultiLeagueRunnerError("mean-reversion candidate has no positive half-life")
            phi = math.exp(-math.log(2.0) * days / candidate.half_life_days)
            self.mean[index] *= phi
            self.covariance[index, :] *= phi
            self.covariance[:, index] *= phi
            self.covariance[index, index] += candidate.process_variance_per_day * days
        elif candidate.kind != "STATIC":
            raise MultiLeagueRunnerError("unknown temporal candidate")
        self.last_at[index] = target

    def transition_players(self, player_ids: Sequence[str], target: datetime) -> None:
        for player_id in sorted(set(player_ids)):
            index = self.ensure(
                _player_key(player_id),
                kind="player",
                prior_variance=PLAYER_PRIOR_VARIANCE,
                at=target,
            )
            self._transition_player(index, target)
        self._cheap_checks()

    def ensure_structural_keys(self, home_leagues: Sequence[str]) -> None:
        self.ensure(
            BLUE_SIDE_KEY,
            kind="blue_side",
            prior_variance=BLUE_SIDE_PRIOR_VARIANCE,
        )
        for league in sorted(set(home_leagues)):
            if league not in DOMESTIC_LEAGUES:
                raise MultiLeagueRunnerError("non-domestic league used as a home-league effect")
            self.ensure(
                _league_key(league),
                kind="league",
                prior_variance=LEAGUE_PRIOR_VARIANCE,
            )

    def vector(self, weights: Mapping[str, float]) -> np.ndarray:
        vector = np.zeros(len(self.keys), dtype=float)
        for key, raw_weight in weights.items():
            if key not in self.index:
                raise MultiLeagueRunnerError("feature references an unregistered latent key")
            weight = _finite(raw_weight, "feature weight")
            vector[self.index[key]] += weight
        return vector

    def moments(self, weights: Mapping[str, float]) -> tuple[float, float]:
        vector = self.vector(weights)
        mean = float(np.einsum("i,i->", vector, self.mean, optimize=True))
        variance = float(
            np.einsum("i,ij,j->", vector, self.covariance, vector, optimize=True)
        )
        if variance < -PSD_TOLERANCE:
            raise MultiLeagueRunnerError("negative posterior predictive variance")
        return _finite(mean, "posterior predictive mean"), max(variance, 0.0)

    def predict(self, weights: Mapping[str, float]) -> tuple[float, float, float]:
        mean, variance = self.moments(weights)
        try:
            probability = posterior_predictive_expected_result(mean, variance)
        except Exception as error:
            raise MultiLeagueRunnerError("posterior-predictive integration failed") from error
        if not 0.0 < probability < 1.0:
            raise MultiLeagueRunnerError("posterior-predictive probability is outside (0,1)")
        return probability, mean, variance

    def update(self, weights: Mapping[str, float], outcome: int) -> None:
        if outcome not in (0, 1):
            raise MultiLeagueRunnerError("outcome update must be binary")
        vector = self.vector(weights)
        eta = float(np.einsum("i,i->", vector, self.mean, optimize=True))
        probability = _sigmoid(eta)
        curvature = probability * (1.0 - probability)
        sigma_x = np.einsum("ij,j->i", self.covariance, vector, optimize=True)
        x_sigma_x = float(np.einsum("i,i->", vector, sigma_x, optimize=True))
        if x_sigma_x < -PSD_TOLERANCE:
            raise MultiLeagueRunnerError("negative update variance")
        denominator = 1.0 + curvature * max(x_sigma_x, 0.0)
        if not math.isfinite(denominator) or denominator <= 0.0:
            raise MultiLeagueRunnerError("rank-one Laplace denominator is invalid")
        residual = float(outcome) - probability
        self.mean = self.mean + sigma_x * residual / denominator
        self.covariance = self.covariance - (
            curvature / denominator
        ) * np.outer(sigma_x, sigma_x)
        # Floating-point rank-one arithmetic can introduce sub-ulp asymmetry.
        self.covariance = (self.covariance + self.covariance.T) / 2.0
        for key, weight in weights.items():
            if abs(float(weight)) > 1e-15:
                self.evidence_counts[key] = self.evidence_counts.get(key, 0) + 1
        self._cheap_checks()

    def _cheap_checks(self) -> None:
        if not np.isfinite(self.mean).all() or not np.isfinite(self.covariance).all():
            raise MultiLeagueRunnerError("Gaussian posterior contains a non-finite value")
        if not np.allclose(self.covariance, self.covariance.T, atol=1e-11, rtol=0.0):
            raise MultiLeagueRunnerError("Gaussian covariance lost symmetry")
        diagonal = np.diag(self.covariance)
        if diagonal.size and float(diagonal.min()) <= MINIMUM_VARIANCE:
            raise MultiLeagueRunnerError("Gaussian covariance has non-positive marginal variance")

    def assert_psd(self) -> dict[str, float]:
        self._cheap_checks()
        if not len(self.keys):
            return {"minimum_eigenvalue": 0.0, "maximum_asymmetry": 0.0}
        eigenvalues = np.linalg.eigvalsh(self.covariance)
        minimum = float(eigenvalues.min())
        if minimum < -PSD_TOLERANCE:
            raise MultiLeagueRunnerError("Gaussian covariance is not positive semidefinite")
        return {
            "minimum_eigenvalue": minimum,
            "maximum_asymmetry": float(np.abs(self.covariance - self.covariance.T).max()),
        }


def _lineup_player_ids(lineup: ObservedLineup) -> tuple[str, ...]:
    if tuple(slot.role for slot in lineup.players) != ROLE_ORDER:
        raise MultiLeagueRunnerError("lineup role order changed after adaptation")
    values = tuple(slot.player_id for slot in lineup.players)
    if len(set(values)) != 5 or any(slot.team_id != lineup.team_id for slot in lineup.players):
        raise MultiLeagueRunnerError("lineup identity changed after adaptation")
    return values


def _series_player_ids(series: DevelopmentSeries) -> tuple[str, ...]:
    values: list[str] = []
    for item in series.maps:
        blue = _lineup_player_ids(item.blue_lineup)
        red = _lineup_player_ids(item.red_lineup)
        if len(set(blue + red)) != 10:
            raise MultiLeagueRunnerError("map no longer has ten distinct players")
        values.extend(blue + red)
    return tuple(sorted(set(values)))


def _home_leagues(
    item: DevelopmentMap,
    known: Mapping[str, str],
) -> tuple[str | None, str | None, str]:
    if item.league in DOMESTIC_LEAGUES:
        return item.league, item.league, "DOMESTIC_NOT_A_BRIDGE"
    blue = known.get(item.blue_lineup.team_id)
    red = known.get(item.red_lineup.team_id)
    if blue is not None and blue not in DOMESTIC_LEAGUES:
        raise MultiLeagueRunnerError("invalid known blue home league")
    if red is not None and red not in DOMESTIC_LEAGUES:
        raise MultiLeagueRunnerError("invalid known red home league")
    if blue is not None and red is not None:
        status = (
            "INTERNATIONAL_BOTH_HOME_LEAGUES_KNOWN"
            if blue != red
            else "INTERNATIONAL_SAME_HOME_LEAGUE"
        )
    elif blue is not None or red is not None:
        status = "INTERNATIONAL_ONE_HOME_LEAGUE_MISSING"
    else:
        status = "INTERNATIONAL_BOTH_HOME_LEAGUES_MISSING"
    return blue, red, status


def _feature_vector(
    state: _GaussianState,
    item: DevelopmentMap,
    known_home_leagues: Mapping[str, str],
) -> _FeatureVector:
    blue_home, red_home, status = _home_leagues(item, known_home_leagues)
    state.ensure_structural_keys(
        [league for league in (blue_home, red_home) if league is not None]
    )
    weights: dict[str, float] = defaultdict(float)
    for slot in item.blue_lineup.players:
        weights[_player_key(slot.player_id)] += 0.2
    for slot in item.red_lineup.players:
        weights[_player_key(slot.player_id)] -= 0.2
    if blue_home is not None:
        weights[_league_key(blue_home)] += 1.0
    if red_home is not None:
        weights[_league_key(red_home)] -= 1.0
    weights[BLUE_SIDE_KEY] += 1.0
    compact = {key: value for key, value in weights.items() if abs(value) > 1e-15}
    return _FeatureVector(compact, blue_home, red_home, status)


def _prediction_row(
    state: _GaussianState,
    series: DevelopmentSeries,
    item: DevelopmentMap,
    feature: _FeatureVector,
) -> dict[str, Any]:
    probability, latent_mean, latent_variance = state.predict(feature.weights)
    return {
        "game_id": item.game_id,
        "series_id": series.series_id,
        "series_identity_kind": series.series_identity_kind,
        "fold_id": series.fold_id,
        "league": series.league,
        "source_local_start": item.source_local_start,
        "game_number": item.game_number,
        "probability": probability,
        "latent_mean": latent_mean,
        "latent_variance": latent_variance,
        "outcome": item.blue_win,
        "league_bridge_status": feature.bridge_status,
        "blue_home_league": feature.blue_home_league,
        "red_home_league": feature.red_home_league,
    }


def _record_available_metadata(
    pending: _PendingSeries,
    *,
    player_metadata: dict[str, dict[str, Any]],
    team_lineups: dict[str, dict[str, Any]],
    team_home_leagues: dict[str, str],
    team_home_order: dict[str, tuple[datetime, int, str]],
) -> None:
    series = pending.series
    for item in series.maps:
        at = source_local_datetime(item.source_local_start)
        order = (at, item.game_number, item.game_id)
        for lineup in (item.blue_lineup, item.red_lineup):
            previous_team = team_lineups.get(lineup.team_id)
            if previous_team is None or order > previous_team["_order"]:
                team_lineups[lineup.team_id] = {
                    "_order": order,
                    "team_id": lineup.team_id,
                    "team_key": lineup.team_key,
                    "team_name": lineup.team_name,
                    "league": series.league,
                    "source_game_id": item.game_id,
                    "source_series_id": series.series_id,
                    "source_local_start": item.source_local_start,
                    "players": [
                        {
                            "role": slot.role,
                            "player_id": slot.player_id,
                            "player_name": slot.player_name,
                        }
                        for slot in lineup.players
                    ],
                }
            for slot in lineup.players:
                previous_player = player_metadata.get(slot.player_id)
                if previous_player is None or order > previous_player["_order"]:
                    player_metadata[slot.player_id] = {
                        "_order": order,
                        "player_id": slot.player_id,
                        "player_name": slot.player_name,
                        "role": slot.role,
                        "team_id": lineup.team_id,
                        "team_key": lineup.team_key,
                        "team_name": lineup.team_name,
                        "source_game_id": item.game_id,
                        "source_series_id": series.series_id,
                        "source_local_start": item.source_local_start,
                    }
            if series.league in DOMESTIC_LEAGUES:
                previous_home = team_home_order.get(lineup.team_id)
                if previous_home is None or order > previous_home:
                    team_home_order[lineup.team_id] = order
                    team_home_leagues[lineup.team_id] = series.league


def _apply_pending_series(
    state: _GaussianState,
    pending: _PendingSeries,
    *,
    player_metadata: dict[str, dict[str, Any]],
    team_lineups: dict[str, dict[str, Any]],
    team_home_leagues: dict[str, str],
    team_home_order: dict[str, tuple[datetime, int, str]],
) -> int:
    state.transition_players(_series_player_ids(pending.series), pending.available_at)
    for item, feature in zip(pending.series.maps, pending.features):
        state.update(feature.weights, item.blue_win)
    _record_available_metadata(
        pending,
        player_metadata=player_metadata,
        team_lineups=team_lineups,
        team_home_leagues=team_home_leagues,
        team_home_order=team_home_order,
    )
    return len(pending.series.maps)


def _partition_payload(input_data: PrivateMultiLeagueRatingInput) -> list[dict[str, Any]]:
    return [
        {
            "series_id": item.series_id,
            "identity_kind": item.series_identity_kind,
            "fold_id": item.fold_id,
            "game_ids": [value.game_id for value in item.maps],
        }
        for item in input_data.development_series
    ] + [
        {
            "series_id": item.series_id,
            "identity_kind": item.series_identity_kind,
            "fold_id": "SEALED_FINAL",
            "game_ids": [value.game_id for value in item.maps],
        }
        for item in input_data.sealed_series_metadata
    ]


def _validate_input(
    input_data: PrivateMultiLeagueRatingInput,
    *,
    expected_maps_sha256: str,
    expected_players_sha256: str,
) -> None:
    expected_maps_sha256 = _require_sha256(expected_maps_sha256, "expected_maps_sha256")
    expected_players_sha256 = _require_sha256(expected_players_sha256, "expected_players_sha256")
    if input_data.maps_sha256 != expected_maps_sha256 or input_data.players_sha256 != expected_players_sha256:
        raise MultiLeagueRunnerError("adapter input does not match the caller's independent warehouse pins")
    if dict(input_data.claim_ceiling) != dict(INPUT_CLAIM_CEILING):
        raise MultiLeagueRunnerError("adapter claim ceiling changed")
    if input_data.claim_ceiling.get("sealed_final_targets_accessed") is not False:
        raise MultiLeagueRunnerError("sealed-final target isolation is not intact")
    if _canonical_sha256(_partition_payload(input_data)) != input_data.cluster_partition_sha256:
        raise MultiLeagueRunnerError("adapter cluster partition digest mismatch")

    series_ids: set[str] = set()
    game_ids: set[str] = set()
    previous_start: datetime | None = None
    for series in input_data.development_series:
        start = source_local_datetime(series.source_local_start)
        end = source_local_datetime(series.source_local_end)
        if start > end or end >= SEALED_FINAL_START.to_pydatetime():
            raise MultiLeagueRunnerError("development series crosses the sealed-final boundary")
        if previous_start is not None and start < previous_start:
            raise MultiLeagueRunnerError("development series are not chronological")
        previous_start = start
        if series.series_id in series_ids or not series.maps:
            raise MultiLeagueRunnerError("duplicate or empty development series")
        series_ids.add(series.series_id)
        for item in series.maps:
            if (
                item.series_id != series.series_id
                or item.fold_id != series.fold_id
                or item.league != series.league
                or item.game_id in game_ids
            ):
                raise MultiLeagueRunnerError("development map identity/series/fold mismatch")
            game_ids.add(item.game_id)
            _lineup_player_ids(item.blue_lineup)
            _lineup_player_ids(item.red_lineup)
    for series in input_data.sealed_series_metadata:
        start = source_local_datetime(series.source_local_start)
        if start < SEALED_FINAL_START.to_pydatetime() or series.series_id in series_ids:
            raise MultiLeagueRunnerError("sealed series identity or boundary mismatch")
        series_ids.add(series.series_id)
        for item in series.maps:
            if item.game_id in game_ids:
                raise MultiLeagueRunnerError("map identity repeats across development/sealed lanes")
            game_ids.add(item.game_id)
            if hasattr(item, "blue_win"):
                raise MultiLeagueRunnerError("sealed map unexpectedly exposes an outcome")
    quarantined_game_ids = {
        game_id for cluster in input_data.quarantined_clusters for game_id in cluster.game_ids
    }
    if game_ids.intersection(quarantined_game_ids):
        raise MultiLeagueRunnerError("accepted and quarantined map identities overlap")
    if len(game_ids) + len(quarantined_game_ids) != input_data.coverage.get("selected_maps"):
        raise MultiLeagueRunnerError("adapter coverage does not reconcile")


def _replay(
    input_data: PrivateMultiLeagueRatingInput,
    candidate: Candidate,
) -> _ReplayResult:
    state = _GaussianState(candidate)
    predictions: list[dict[str, Any]] = []
    pending: list[tuple[datetime, int, str, _PendingSeries]] = []
    sequence = 0
    player_metadata: dict[str, dict[str, Any]] = {}
    team_lineups: dict[str, dict[str, Any]] = {}
    team_home_leagues: dict[str, str] = {}
    team_home_order: dict[str, tuple[datetime, int, str]] = {}
    bridge_diagnostics: dict[str, int] = defaultdict(int)
    ordered_applied_series: list[str] = []
    applied_series = 0
    applied_maps = 0

    def flush(boundary: datetime) -> None:
        nonlocal applied_series, applied_maps
        # Strict inequality preserves the declared >48-hour eligibility rule.
        while pending and pending[0][0] < boundary:
            _available, _sequence, _identity, value = heapq.heappop(pending)
            applied_maps += _apply_pending_series(
                state,
                value,
                player_metadata=player_metadata,
                team_lineups=team_lineups,
                team_home_leagues=team_home_leagues,
                team_home_order=team_home_order,
            )
            applied_series += 1
            ordered_applied_series.append(value.series.series_id)

    for series in input_data.development_series:
        start = source_local_datetime(series.source_local_start)
        flush(start)
        state.transition_players(_series_player_ids(series), start)
        features: list[_FeatureVector] = []
        for item in series.maps:
            feature = _feature_vector(state, item, team_home_leagues)
            features.append(feature)
            bridge_diagnostics[feature.bridge_status] += 1
            predictions.append(_prediction_row(state, series, item, feature))
        available = source_local_datetime(series.source_local_end) + timedelta(
            hours=AVAILABILITY_EMBARGO_HOURS
        )
        value = _PendingSeries(available, series, tuple(features))
        heapq.heappush(pending, (available, sequence, series.series_id, value))
        sequence += 1

    cutoff = SEALED_FINAL_START.to_pydatetime()
    flush(cutoff)
    player_ids = [
        key.removeprefix("player:")
        for key in state.keys
        if key.startswith("player:")
    ]
    state.transition_players(player_ids, cutoff)
    state.assert_psd()
    return _ReplayResult(
        candidate=candidate,
        predictions=predictions,
        state=state,
        player_metadata=player_metadata,
        team_lineups=team_lineups,
        team_home_leagues=team_home_leagues,
        bridge_diagnostics=dict(sorted(bridge_diagnostics.items())),
        applied_series=applied_series,
        applied_maps=applied_maps,
        ordered_applied_series=ordered_applied_series,
    )


def _loss(probability: float, outcome: int, metric: str) -> float:
    probability = _finite(probability, "evaluation probability")
    if outcome not in (0, 1) or not 0.0 < probability < 1.0:
        raise MultiLeagueRunnerError("invalid evaluation row")
    if metric == "log_loss":
        return -(outcome * math.log(probability) + (1 - outcome) * math.log(1.0 - probability))
    if metric == "brier":
        return (probability - outcome) ** 2
    raise MultiLeagueRunnerError("unknown proper scoring rule")


def _metric_payload(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "maps": 0,
            "series": 0,
            "map_weighted": {"log_loss": None, "brier": None},
            "series_macro": {"log_loss": None, "brier": None},
            "calibration": None,
        }
    by_series: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    log_losses: list[float] = []
    briers: list[float] = []
    bins = [
        {"count": 0, "probability_sum": 0.0, "outcome_sum": 0.0}
        for _ in range(10)
    ]
    for row in rows:
        probability = _finite(row["probability"], "evaluation probability")
        outcome = int(row["outcome"])
        log_losses.append(_loss(probability, outcome, "log_loss"))
        briers.append(_loss(probability, outcome, "brier"))
        by_series[str(row["series_id"])].append(row)
        bucket = min(9, int(probability * 10.0))
        bins[bucket]["count"] += 1
        bins[bucket]["probability_sum"] += probability
        bins[bucket]["outcome_sum"] += outcome
    series_log = []
    series_brier = []
    for values in by_series.values():
        series_log.append(
            float(
                np.mean(
                    [_loss(float(row["probability"]), int(row["outcome"]), "log_loss") for row in values]
                )
            )
        )
        series_brier.append(
            float(
                np.mean(
                    [_loss(float(row["probability"]), int(row["outcome"]), "brier") for row in values]
                )
            )
        )
    calibration_bins = []
    ece = 0.0
    for index, value in enumerate(bins):
        count = value["count"]
        mean_probability = value["probability_sum"] / count if count else None
        observed_rate = value["outcome_sum"] / count if count else None
        if count:
            ece += abs(float(mean_probability) - float(observed_rate)) * count / len(rows)
        calibration_bins.append(
            {
                "bin": index,
                "lower_inclusive": index / 10.0,
                "upper_exclusive": None if index == 9 else (index + 1) / 10.0,
                "count": count,
                "mean_probability": mean_probability,
                "observed_rate": observed_rate,
            }
        )
    return {
        "maps": len(rows),
        "series": len(by_series),
        "map_weighted": {
            "log_loss": float(np.mean(log_losses)),
            "brier": float(np.mean(briers)),
        },
        "series_macro": {
            "log_loss": float(np.mean(series_log)),
            "brier": float(np.mean(series_brier)),
        },
        "calibration": {
            "kind": "ten_equal_width_map_bins_descriptive_only",
            "ece": ece,
            "bins": calibration_bins,
        },
    }


def _fold_rows(
    predictions: Sequence[Mapping[str, Any]],
    fold_id: str,
    league: str | None = None,
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in predictions
        if row["fold_id"] == fold_id and (league is None or row["league"] == league)
    ]


def _evaluation_payload(
    predictions: Sequence[Mapping[str, Any]],
    fold_id: str,
) -> dict[str, Any]:
    rows = _fold_rows(predictions, fold_id)
    return {
        "fold_id": fold_id,
        "overall": _metric_payload(rows),
        "by_league": [
            {"league": league, **_metric_payload(_fold_rows(predictions, fold_id, league))}
            for league in LEAGUES
        ],
        "prediction_rows_sha256": _canonical_sha256(list(rows)),
    }


def _bootstrap_payload(result: BootstrapResult) -> dict[str, Any]:
    sizes: dict[str, int] = defaultdict(int)
    for size in result.cluster_size_distribution.values():
        sizes[str(int(size))] += 1
    return {
        "point": result.point,
        "lower_95": result.lower_95,
        "upper_95": result.upper_95,
        "cluster_count": result.cluster_count,
        "resolved_cluster_count": result.resolved_cluster_count,
        "cluster_unit": result.cluster_unit,
        "cluster_size_distribution": dict(sorted(sizes.items(), key=lambda item: int(item[0]))),
    }


def _paired_comparison(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    *,
    fold_id: str,
    league: str | None,
) -> dict[str, Any]:
    candidate_rows = _fold_rows(candidate, fold_id, league)
    baseline_rows = _fold_rows(baseline, fold_id, league)
    candidate_by_id = {str(row["game_id"]): row for row in candidate_rows}
    baseline_by_id = {str(row["game_id"]): row for row in baseline_rows}
    if set(candidate_by_id) != set(baseline_by_id):
        raise MultiLeagueRunnerError("candidate and baseline evaluation populations differ")
    ordered_ids = sorted(candidate_by_id)
    if not ordered_ids:
        return {
            "status": "UNAVAILABLE_NO_ROWS",
            "league": league,
            "maps": 0,
            "series": 0,
            "log_loss_candidate_minus_static": None,
            "brier_candidate_minus_static": None,
        }
    cluster_ids = [str(candidate_by_id[game_id]["series_id"]) for game_id in ordered_ids]
    series_count = len(set(cluster_ids))
    values: dict[str, dict[str, Any]] = {}
    for offset, metric in enumerate(("log_loss", "brier")):
        deltas = [
            _loss(
                float(candidate_by_id[game_id]["probability"]),
                int(candidate_by_id[game_id]["outcome"]),
                metric,
            )
            - _loss(
                float(baseline_by_id[game_id]["probability"]),
                int(baseline_by_id[game_id]["outcome"]),
                metric,
            )
            for game_id in ordered_ids
        ]
        result = series_cluster_bootstrap(
            deltas,
            cluster_ids,
            [True] * len(deltas),
            row_ids=ordered_ids,
            n_boot=BOOTSTRAP_SAMPLES,
            random_seed=BOOTSTRAP_SEED + offset,
            cluster_unit="source-or-derived-series-dependence-cluster",
        )
        values[metric] = _bootstrap_payload(result)
    enough = series_count >= MINIMUM_VALIDATION_SERIES
    pass_interval = enough and all(values[metric]["upper_95"] <= 0.0 for metric in values)
    return {
        "status": (
            "PASS_NONPOSITIVE_UPPER_95"
            if pass_interval
            else "FAIL_INSUFFICIENT_SERIES"
            if not enough
            else "FAIL_UPPER_95_ABOVE_ZERO"
        ),
        "league": league,
        "maps": len(ordered_ids),
        "series": series_count,
        "minimum_required_series": MINIMUM_VALIDATION_SERIES,
        "log_loss_candidate_minus_static": values["log_loss"],
        "brier_candidate_minus_static": values["brier"],
    }


def _comparison_payload(
    candidate: Sequence[Mapping[str, Any]],
    baseline: Sequence[Mapping[str, Any]],
    fold_id: str,
) -> dict[str, Any]:
    return {
        "fold_id": fold_id,
        "overall": _paired_comparison(
            candidate,
            baseline,
            fold_id=fold_id,
            league=None,
        ),
        "domestic_leagues": [
            _paired_comparison(
                candidate,
                baseline,
                fold_id=fold_id,
                league=league,
            )
            for league in DOMESTIC_LEAGUES
        ],
    }


def _validation_gate(comparison: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if comparison["overall"]["status"] != "PASS_NONPOSITIVE_UPPER_95":
        failures.append("overall_validation_interval_does_not_dominate_static")
    for item in comparison["domestic_leagues"]:
        if item["status"] != "PASS_NONPOSITIVE_UPPER_95":
            failures.append(
                f"{str(item['league']).lower()}_validation_interval_does_not_dominate_static"
            )
    return not failures, failures


def _state_digest(state: _GaussianState) -> str:
    digest = hashlib.sha256()
    digest.update(_canonical_bytes({"keys": state.keys, "kinds": state.kinds}))
    digest.update(np.asarray(state.mean, dtype="<f8").tobytes(order="C"))
    digest.update(np.asarray(state.covariance, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _marginal_payload(state: _GaussianState, weights: Mapping[str, float]) -> dict[str, float]:
    mean, variance = state.moments(weights)
    uncertainty = math.sqrt(max(variance, 0.0))
    return {
        "latent_mean": mean,
        "latent_standard_deviation": uncertainty,
        "rating_mean": DISPLAY_ANCHOR + DISPLAY_LOGIT_SCALE * mean,
        "rating_standard_deviation": DISPLAY_LOGIT_SCALE * uncertainty,
        "rating_lower_approx_95": DISPLAY_ANCHOR
        + DISPLAY_LOGIT_SCALE * (mean - 1.959963984540054 * uncertainty),
        "rating_upper_approx_95": DISPLAY_ANCHOR
        + DISPLAY_LOGIT_SCALE * (mean + 1.959963984540054 * uncertainty),
    }


def _player_snapshot(replay: _ReplayResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for player_id, metadata in replay.player_metadata.items():
        key = _player_key(player_id)
        if key not in replay.state.index:
            raise MultiLeagueRunnerError("available player metadata has no latent state")
        value = _marginal_payload(replay.state, {key: 1.0})
        rows.append(
            {
                "player_id": player_id,
                "player_name": metadata["player_name"],
                "last_observed_role": metadata["role"],
                "last_observed_team_id": metadata["team_id"],
                "last_observed_team_key": metadata["team_key"],
                "last_observed_team_name": metadata["team_name"],
                "last_observed_source_game_id": metadata["source_game_id"],
                "last_observed_source_series_id": metadata["source_series_id"],
                "last_observed_source_local_start": metadata["source_local_start"],
                "outcome_evidence_maps": replay.state.evidence_counts.get(key, 0),
                "estimand": (
                    "equal-role-weight player contribution to team map outcome; "
                    "not individual box-score performance and not rank-comparable across roles"
                ),
                **value,
            }
        )
    rows.sort(key=lambda item: (-item["rating_mean"], item["player_id"]))
    return rows


def _team_snapshot(replay: _ReplayResult) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for team_id, raw_lineup in replay.team_lineups.items():
        lineup = {key: value for key, value in raw_lineup.items() if key != "_order"}
        player_weights = {
            _player_key(str(slot["player_id"])): 0.2 for slot in lineup["players"]
        }
        if len(player_weights) != 5 or any(key not in replay.state.index for key in player_weights):
            raise MultiLeagueRunnerError("last observed team lineup has no exact five-player posterior")
        player_component = _marginal_payload(replay.state, player_weights)
        home_league = replay.team_home_leagues.get(team_id)
        league_component: dict[str, Any]
        globally_bridged: dict[str, Any] | None = None
        if home_league is None:
            league_component = {
                "status": "UNAVAILABLE",
                "value_rating_points": None,
                "standard_deviation_rating_points": None,
                "reason": "no_prior_available_domestic_home_league_identity",
            }
        else:
            key = _league_key(home_league)
            evidence = replay.state.evidence_counts.get(key, 0)
            if key not in replay.state.index or evidence == 0:
                league_component = {
                    "status": "UNAVAILABLE",
                    "value_rating_points": None,
                    "standard_deviation_rating_points": None,
                    "reason": "home_league_effect_not_identified_by_available_international_bridges",
                }
            else:
                index = replay.state.index[key]
                mean = float(replay.state.mean[index])
                standard_deviation = math.sqrt(max(float(replay.state.covariance[index, index]), 0.0))
                league_component = {
                    "status": "ESTIMATED",
                    "home_league": home_league,
                    "value_rating_points": DISPLAY_LOGIT_SCALE * mean,
                    "standard_deviation_rating_points": DISPLAY_LOGIT_SCALE
                    * standard_deviation,
                    "outcome_evidence_maps": evidence,
                }
                global_weights = dict(player_weights)
                global_weights[key] = 1.0
                globally_bridged = _marginal_payload(replay.state, global_weights)
        rows.append(
            {
                "team_id": team_id,
                "team_key": lineup["team_key"],
                "team_name": lineup["team_name"],
                "home_league": home_league,
                "roster_semantics": "LAST_OBSERVED_HISTORICAL_LINEUP_NOT_PRE_EVENT_AUTHORITY",
                "last_observed_lineup": {
                    "source_game_id": lineup["source_game_id"],
                    "source_series_id": lineup["source_series_id"],
                    "source_local_start": lineup["source_local_start"],
                    "players": lineup["players"],
                },
                "components": {
                    "player_aggregate": {
                        "status": "ESTIMATED",
                        **player_component,
                    },
                    "league_adjustment": league_component,
                    "lineup_synergy": {
                        "status": "UNAVAILABLE",
                        "value": None,
                        "reason": "not_identified_by_the_player_plus_league_outcome_estimand",
                    },
                    "team_policy": {
                        "status": "UNAVAILABLE",
                        "value": None,
                        "reason": "no_independent_pre_event_policy_measurement_model",
                    },
                },
                "within_league_player_strength": player_component,
                "globally_bridged_player_plus_league_strength": globally_bridged,
                "interpretation": (
                    "The within-league value is the exact last-observed roster's player aggregate. "
                    "A global value exists only when the home-league adjustment has bridge evidence. "
                    "Unavailable synergy or policy is not treated as zero."
                ),
            }
        )
    rows.sort(
        key=lambda item: (
            -item["within_league_player_strength"]["rating_mean"],
            item["team_id"],
        )
    )
    return rows


def _structural_snapshot(replay: _ReplayResult) -> dict[str, Any]:
    leagues = []
    for league in DOMESTIC_LEAGUES:
        key = _league_key(league)
        if key not in replay.state.index:
            leagues.append(
                {
                    "league": league,
                    "status": "UNAVAILABLE",
                    "rating_point_adjustment": None,
                    "standard_deviation_rating_points": None,
                    "outcome_evidence_maps": 0,
                }
            )
            continue
        index = replay.state.index[key]
        evidence = replay.state.evidence_counts.get(key, 0)
        leagues.append(
            {
                "league": league,
                "status": "ESTIMATED" if evidence else "UNAVAILABLE",
                "rating_point_adjustment": (
                    DISPLAY_LOGIT_SCALE * float(replay.state.mean[index]) if evidence else None
                ),
                "standard_deviation_rating_points": (
                    DISPLAY_LOGIT_SCALE
                    * math.sqrt(max(float(replay.state.covariance[index, index]), 0.0))
                    if evidence
                    else None
                ),
                "outcome_evidence_maps": evidence,
            }
        )
    blue = replay.state.index.get(BLUE_SIDE_KEY)
    return {
        "league_adjustments": leagues,
        "blue_side": (
            None
            if blue is None
            else {
                "latent_mean": float(replay.state.mean[blue]),
                "rating_point_adjustment": DISPLAY_LOGIT_SCALE
                * float(replay.state.mean[blue]),
                "standard_deviation_rating_points": DISPLAY_LOGIT_SCALE
                * math.sqrt(max(float(replay.state.covariance[blue, blue]), 0.0)),
                "outcome_evidence_maps": replay.state.evidence_counts.get(BLUE_SIDE_KEY, 0),
            }
        ),
    }


def _config_payload() -> dict[str, Any]:
    return {
        "candidates": [candidate.payload() for candidate in CANDIDATES],
        "priors": {
            "player_variance": PLAYER_PRIOR_VARIANCE,
            "league_variance": LEAGUE_PRIOR_VARIANCE,
            "blue_side_variance": BLUE_SIDE_PRIOR_VARIANCE,
        },
        "lineup_design": {
            "role_order": list(ROLE_ORDER),
            "each_blue_player_weight": 0.2,
            "each_red_player_weight": -0.2,
            "blue_home_league_weight": 1.0,
            "red_home_league_weight": -1.0,
            "blue_side_weight": 1.0,
        },
        "availability": {
            "series_state_frozen_within_series": True,
            "outcome_available_strictly_after_hours": AVAILABILITY_EMBARGO_HOURS,
            "availability_order": "source_series_end_plus_embargo",
            "source_timestamp_semantics": "timezone-naive; 48-hour embargo exceeds plausible timezone offset disagreement",
        },
        "selection": {
            "fold": "DEVELOPMENT",
            "primary": "series_macro_log_loss",
            "secondary": "series_macro_brier",
            "tie_break": "candidate_id",
        },
        "validation": {
            "fold": "VALIDATION",
            "comparison": "paired_candidate_minus_static",
            "resampling_unit": "source-or-derived-series-dependence-cluster",
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "minimum_series_overall_and_per_domestic_league": MINIMUM_VALIDATION_SERIES,
            "gate": "overall_and_each_domestic_league_log_loss_and_brier_upper_95_must_be_nonpositive",
        },
        "sealed_final": {
            "starts_at": SEALED_FINAL_START.isoformat(),
            "targets_accessed": False,
            "candidate_opening_requires_separate_external_authority": True,
        },
    }


def build_multileague_development_artifact(
    *,
    expected_maps_sha256: str,
    expected_players_sha256: str,
    input_loader: Callable[..., PrivateMultiLeagueRatingInput] = load_multileague_development_input,
) -> dict[str, Any]:
    """Run fixed candidates; selection is development-only and final stays sealed."""

    expected_maps_sha256 = _require_sha256(expected_maps_sha256, "expected_maps_sha256")
    expected_players_sha256 = _require_sha256(expected_players_sha256, "expected_players_sha256")
    input_data = input_loader(
        expected_maps_sha256=expected_maps_sha256,
        expected_players_sha256=expected_players_sha256,
    )
    _validate_input(
        input_data,
        expected_maps_sha256=expected_maps_sha256,
        expected_players_sha256=expected_players_sha256,
    )
    config = _config_payload()
    replays = [_replay(input_data, candidate) for candidate in CANDIDATES]
    by_id = {replay.candidate.candidate_id: replay for replay in replays}
    baseline = by_id["static_no_reset"]
    candidate_results: list[dict[str, Any]] = []
    development_metrics: dict[str, dict[str, Any]] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    for replay in replays:
        development = _evaluation_payload(replay.predictions, "DEVELOPMENT")
        validation = _evaluation_payload(replay.predictions, "VALIDATION")
        development_metrics[replay.candidate.candidate_id] = development
        comparison = {
            "development": _comparison_payload(
                replay.predictions, baseline.predictions, "DEVELOPMENT"
            ),
            "validation": _comparison_payload(
                replay.predictions, baseline.predictions, "VALIDATION"
            ),
        }
        comparisons[replay.candidate.candidate_id] = comparison
        diagnostics = replay.state.assert_psd()
        candidate_results.append(
            {
                "candidate": replay.candidate.payload(),
                "development": development,
                "validation": validation,
                "paired_against_static": comparison,
                "replay": {
                    "series_predictions_are_prior_frozen": True,
                    "applied_available_series_at_sealed_boundary": replay.applied_series,
                    "applied_available_maps_at_sealed_boundary": replay.applied_maps,
                    "ordered_applied_series_sha256": _canonical_sha256(
                        replay.ordered_applied_series
                    ),
                    "league_bridge_diagnostics": replay.bridge_diagnostics,
                    "latent_dimension": len(replay.state.keys),
                    "posterior_state_sha256": _state_digest(replay.state),
                },
                "numerics": {
                    "finite": True,
                    "covariance": "FULL_GAUSSIAN_RANK_ONE_UPDATED",
                    **diagnostics,
                },
            }
        )

    winner = min(
        replays,
        key=lambda replay: (
            development_metrics[replay.candidate.candidate_id]["overall"]["series_macro"]["log_loss"],
            development_metrics[replay.candidate.candidate_id]["overall"]["series_macro"]["brier"],
            replay.candidate.candidate_id,
        ),
    )
    if winner.candidate.candidate_id == "static_no_reset":
        validation_passed = False
        gate_failures = ["development_selected_static_baseline"]
        result_state = "DEVELOPMENT_BASELINE_ONLY"
        sealed_candidate = None
    else:
        validation_passed, gate_failures = _validation_gate(
            comparisons[winner.candidate.candidate_id]["validation"]
        )
        result_state = (
            "DEVELOPMENT_CANDIDATE_VALIDATION_GATE_PASSED"
            if validation_passed
            else "DEVELOPMENT_CANDIDATE_VALIDATION_GATE_FAILED"
        )
        sealed_candidate = winner.candidate.candidate_id if validation_passed else None

    latest_development = max(
        source_local_datetime(series.source_local_end)
        for series in input_data.development_series
    ).isoformat()
    latest_sealed_metadata = max(
        source_local_datetime(series.source_local_end)
        for series in input_data.sealed_series_metadata
    ).isoformat()
    posterior = {
        "candidate_id": winner.candidate.candidate_id,
        "as_of_exclusive": SEALED_FINAL_START.isoformat(),
        "ratings_are_non_authorizing_development_outputs": True,
        "players": _player_snapshot(winner),
        "teams": _team_snapshot(winner),
        "structural_effects": _structural_snapshot(winner),
    }
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": result_state,
        "private_scope": {
            "available": ["retrospective_model_fit", "development_rank_selection"],
            "unavailable": [
                "pre_event_roster_authority",
                "sealed_final_evaluation",
                "production_rating_authority",
                "match_probability",
                "fair_odds",
                "expected_value",
                "bet_recommendation",
                "publication",
            ],
        },
        "input": {
            "maps_locator": input_data.maps_locator,
            "players_locator": input_data.players_locator,
            "maps_sha256": input_data.maps_sha256,
            "players_sha256": input_data.players_sha256,
            "development_selected_rows_sha256": input_data.development_selected_rows_sha256,
            "sealed_selected_metadata_sha256": input_data.sealed_selected_metadata_sha256,
            "player_selected_metadata_sha256": input_data.player_selected_metadata_sha256,
            "cluster_partition_sha256": input_data.cluster_partition_sha256,
            "coverage": dict(input_data.coverage),
            "quarantined_clusters": [
                {
                    "cluster_id": item.cluster_id,
                    "game_ids": list(item.game_ids),
                    "reasons": list(item.reasons),
                }
                for item in input_data.quarantined_clusters
            ],
            "claim_ceiling": dict(input_data.claim_ceiling),
        },
        "freshness": {
            "latest_development_target_source_local_end": latest_development,
            "latest_outcome_free_sealed_metadata_source_local_end": latest_sealed_metadata,
            "rating_as_of_exclusive": SEALED_FINAL_START.isoformat(),
            "current_event_freshness_authorized": False,
        },
        "config": config,
        "config_sha256": _canonical_sha256(config),
        "candidate_results": candidate_results,
        "selection": {
            "development_winner_candidate_id": winner.candidate.candidate_id,
            "validation_gate_passed": validation_passed,
            "validation_gate_failures": gate_failures,
            "candidate_eligible_for_separately_authorized_sealed_evaluation": sealed_candidate,
            "sealed_final_opened": False,
        },
        "development_posterior": posterior,
        "decision_outputs": {
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
        "generator": {
            "source_locator": "lol_kills/v2/ratings/player/multileague_runner.py",
            "source_raw_sha256": _sha256(Path(__file__).read_bytes()),
            "runtime_versions": {
                "numpy": importlib.metadata.version("numpy"),
                "pandas": importlib.metadata.version("pandas"),
                "scipy": importlib.metadata.version("scipy"),
            },
        },
    }
    artifact["artifact_sha256"] = _canonical_sha256(artifact)
    return artifact


def validate_multileague_development_artifact(
    artifact: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise MultiLeagueRunnerError("development artifact must be an object")
    unsigned = dict(artifact)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != _canonical_sha256(unsigned):
        raise MultiLeagueRunnerError("development artifact digest mismatch")
    if artifact.get("schema_version") != SCHEMA_VERSION:
        raise MultiLeagueRunnerError("development artifact schema mismatch")
    if artifact.get("result_state") not in {
        "DEVELOPMENT_BASELINE_ONLY",
        "DEVELOPMENT_CANDIDATE_VALIDATION_GATE_PASSED",
        "DEVELOPMENT_CANDIDATE_VALIDATION_GATE_FAILED",
    }:
        raise MultiLeagueRunnerError("development artifact state is invalid")
    selection = artifact.get("selection")
    if not isinstance(selection, Mapping) or selection.get("sealed_final_opened") is not False:
        raise MultiLeagueRunnerError("sealed-final isolation is not declared")
    inputs = artifact.get("input")
    if (
        not isinstance(inputs, Mapping)
        or inputs.get("claim_ceiling", {}).get("sealed_final_targets_accessed") is not False
    ):
        raise MultiLeagueRunnerError("input claim ceiling does not preserve sealed-final isolation")
    outputs = artifact.get("decision_outputs")
    if not isinstance(outputs, Mapping) or set(outputs) != {
        "match_probability",
        "fair_odds",
        "expected_value",
        "bet_recommendation",
    }:
        raise MultiLeagueRunnerError("decision-output closure is invalid")
    if any(value is not None for value in outputs.values()):
        raise MultiLeagueRunnerError("development artifact emitted an actionable decision output")
    posterior = artifact.get("development_posterior")
    if not isinstance(posterior, Mapping) or posterior.get(
        "ratings_are_non_authorizing_development_outputs"
    ) is not True:
        raise MultiLeagueRunnerError("development posterior boundary is missing")
    generator = artifact.get("generator")
    if (
        not isinstance(generator, Mapping)
        or generator.get("source_locator")
        != "lol_kills/v2/ratings/player/multileague_runner.py"
        or generator.get("source_raw_sha256") != _sha256(Path(__file__).read_bytes())
    ):
        raise MultiLeagueRunnerError("development artifact generator source binding is stale")
    for team in posterior.get("teams", []):
        components = team.get("components", {}) if isinstance(team, Mapping) else {}
        for name in ("lineup_synergy", "team_policy"):
            component = components.get(name, {})
            if component.get("status") != "UNAVAILABLE" or component.get("value") is not None:
                raise MultiLeagueRunnerError("unidentified team component was not preserved as unavailable")
    for candidate in artifact.get("candidate_results", []):
        numerics = candidate.get("numerics", {}) if isinstance(candidate, Mapping) else {}
        if (
            numerics.get("finite") is not True
            or numerics.get("covariance") != "FULL_GAUSSIAN_RANK_ONE_UPDATED"
            or _finite(numerics.get("minimum_eigenvalue"), "minimum eigenvalue")
            < -PSD_TOLERANCE
        ):
            raise MultiLeagueRunnerError("candidate numerical checks are invalid")
    return dict(artifact)


def verify_multileague_development_artifact(
    artifact: Mapping[str, Any],
    *,
    expected_artifact_sha256: str,
) -> dict[str, Any]:
    """Verify content plus a mandatory digest supplied by independent review."""

    expected = _require_sha256(expected_artifact_sha256, "expected_artifact_sha256")
    validated = validate_multileague_development_artifact(artifact)
    if validated["artifact_sha256"] != expected:
        raise MultiLeagueRunnerError("development artifact does not match the independent pin")
    return validated


def write_multileague_development_artifact_no_clobber(
    artifact: Mapping[str, Any],
    path: Path,
) -> str:
    """Write once without replacing any existing reviewed candidate bytes."""

    validated = validate_multileague_development_artifact(artifact)
    raw = _canonical_bytes(validated) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise MultiLeagueRunnerError("development artifact path already exists; refusing to clobber") from error
        metadata = os.lstat(path)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise MultiLeagueRunnerError("development artifact output is not a regular file")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a non-authorizing pinned multi-league rating development artifact."
    )
    parser.add_argument("--expected-maps-sha256", required=True)
    parser.add_argument("--expected-players-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = build_multileague_development_artifact(
        expected_maps_sha256=args.expected_maps_sha256,
        expected_players_sha256=args.expected_players_sha256,
    )
    raw_sha256 = write_multileague_development_artifact_no_clobber(
        artifact, args.output
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "raw_sha256": raw_sha256,
                "artifact_sha256": artifact["artifact_sha256"],
                "result_state": artifact["result_state"],
                "selection": artifact["selection"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by artifact generation
    raise SystemExit(main())
