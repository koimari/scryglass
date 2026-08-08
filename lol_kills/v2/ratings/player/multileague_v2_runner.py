"""Adaptive development runner for the locked multi-league v2 rating family.

All outcomes before 2026-04-01 are treated as adaptive development evidence.
The runner never reads outcomes from the protocol's sealed-final cohort.  Its
selected Player/Organization decomposition is prior-regularized and therefore
descriptive; only the joint predictive contrast is identified by map outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from . import multileague_benchmark as benchmark
from . import multileague_development as adapter
from . import multileague_runner as rating
from . import multileague_v2_protocol as protocol


SCHEMA_VERSION = "scryglass:multileague-rating-v2-adaptive-development:v1"
RESULT_SELECTED = "ADAPTIVE_CANDIDATE_SELECTED_SEALED_FINAL_UNOPENED"
RESULT_NO_ELIGIBLE = "NO_ELIGIBLE_ADAPTIVE_CANDIDATE_SEALED_FINAL_UNOPENED"
DEFAULT_PROTOCOL = protocol.DEFAULT_OUTPUT
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v2/adaptive-development-artifact-v1.json"
)
SOURCE_LOCATOR = "lol_kills/v2/ratings/player/multileague_v2_runner.py"
ORGANIZATION_ID_PREFIX = "organization:"


class MultiLeagueV2RunnerError(ValueError):
    """The locked adaptive replay or its artifact failed closed."""


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    player_weight_per_role: float
    player_prior_variance: float
    player_process_variance_per_day: float
    organization_weight: float
    organization_prior_variance: float
    organization_process_variance_per_day: float
    organization_retention_floor: float

    @classmethod
    def from_payload(cls, value: Mapping[str, Any]) -> "CandidateSpec":
        retention = value.get("organization_roster_retention") or {}
        try:
            result = cls(
                candidate_id=str(value["candidate_id"]),
                player_weight_per_role=float(value["player_weight_per_role"]),
                player_prior_variance=float(value["player_prior_variance"]),
                player_process_variance_per_day=float(
                    value["player_process_variance_per_day"]
                ),
                organization_weight=float(value["organization_weight"]),
                organization_prior_variance=float(
                    value["organization_prior_variance"]
                ),
                organization_process_variance_per_day=float(
                    value["organization_process_variance_per_day"]
                ),
                organization_retention_floor=float(retention["floor"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise MultiLeagueV2RunnerError("locked candidate payload is malformed") from error
        numeric = (
            result.player_weight_per_role,
            result.player_prior_variance,
            result.player_process_variance_per_day,
            result.organization_weight,
            result.organization_prior_variance,
            result.organization_process_variance_per_day,
            result.organization_retention_floor,
        )
        if any(not math.isfinite(item) for item in numeric):
            raise MultiLeagueV2RunnerError("locked candidate contains a non-finite value")
        if (
            result.player_weight_per_role != 0.2
            or result.player_prior_variance <= 0.0
            or result.organization_weight <= 0.0
            or result.organization_prior_variance <= 0.0
            or result.player_process_variance_per_day < 0.0
            or result.organization_process_variance_per_day < 0.0
            or not 0.0 < result.organization_retention_floor <= 1.0
        ):
            raise MultiLeagueV2RunnerError("locked candidate is outside its supported domain")
        return result


@dataclass(frozen=True)
class _PendingSeries:
    available_at: datetime
    series: adapter.DevelopmentSeries
    features: tuple[rating._FeatureVector, ...]


@dataclass
class ReplayResult:
    candidate: CandidateSpec
    predictions: list[dict[str, Any]]
    state: "HierarchicalGaussianState"
    player_metadata: dict[str, dict[str, Any]]
    team_lineups: dict[str, dict[str, Any]]
    team_home_leagues: dict[str, str]
    bridge_diagnostics: dict[str, int]
    roster_transition_diagnostics: dict[str, int]
    applied_series: int
    applied_maps: int


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
        raise MultiLeagueV2RunnerError("artifact contains a non-canonical value") from error


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _artifact_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return _sha256(_canonical_bytes(body))


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise MultiLeagueV2RunnerError(f"{label} must be a lowercase SHA-256")
    return value


def _organization_identity(team_id: str) -> str:
    return f"{ORGANIZATION_ID_PREFIX}{team_id}"


def _organization_key(team_id: str) -> str:
    return rating._player_key(_organization_identity(team_id))


def _lineup_identity(
    lineup: adapter.ObservedLineup,
) -> tuple[tuple[str, str], ...]:
    if tuple(slot.role for slot in lineup.players) != adapter.ROLE_ORDER:
        raise MultiLeagueV2RunnerError("lineup role order changed after adaptation")
    value = tuple((slot.role, slot.player_id) for slot in lineup.players)
    if len({player_id for _role, player_id in value}) != 5:
        raise MultiLeagueV2RunnerError("lineup does not contain five distinct players")
    return value


def _series_team_ids(series: adapter.DevelopmentSeries) -> tuple[str, str]:
    teams = sorted(
        {
            lineup.team_id
            for item in series.maps
            for lineup in (item.blue_lineup, item.red_lineup)
        }
    )
    if len(teams) != 2:
        raise MultiLeagueV2RunnerError("series does not contain exactly two teams")
    return teams[0], teams[1]


class HierarchicalGaussianState(rating._GaussianState):
    """Full covariance state with separately locked player/org dynamics."""

    def __init__(self, candidate: CandidateSpec):
        super().__init__(
            rating.Candidate(
                candidate.candidate_id,
                "RANDOM_WALK",
                candidate.player_process_variance_per_day,
            )
        )
        self.spec = candidate

    def ensure_player(self, player_id: str, at: datetime) -> int:
        return self.ensure(
            rating._player_key(player_id),
            kind="player",
            prior_variance=self.spec.player_prior_variance,
            at=at,
        )

    def ensure_organization(self, team_id: str, at: datetime) -> int:
        return self.ensure(
            _organization_key(team_id),
            kind="player",
            prior_variance=self.spec.organization_prior_variance,
            at=at,
        )

    def _transition_player(self, index: int, target: datetime) -> None:
        previous = self.last_at[index]
        if previous is None:
            self.last_at[index] = target
            return
        days = (target - previous).total_seconds() / 86400.0
        if days < -1e-12:
            raise MultiLeagueV2RunnerError("dynamic state attempted to move backward")
        key = self.keys[index]
        process = (
            self.spec.organization_process_variance_per_day
            if key.startswith(f"player:{ORGANIZATION_ID_PREFIX}")
            else self.spec.player_process_variance_per_day
        )
        self.covariance[index, index] += process * max(days, 0.0)
        self.last_at[index] = target

    def transition_entities(
        self,
        player_ids: Sequence[str],
        team_ids: Sequence[str],
        target: datetime,
    ) -> None:
        for player_id in sorted(set(player_ids)):
            index = self.ensure_player(player_id, target)
            self._transition_player(index, target)
        for team_id in sorted(set(team_ids)):
            index = self.ensure_organization(team_id, target)
            self._transition_player(index, target)
        self._cheap_checks()

    def apply_roster_transition(
        self,
        team_id: str,
        previous: tuple[tuple[str, str], ...] | None,
        current: tuple[tuple[str, str], ...],
    ) -> float | None:
        if previous is None or previous == current:
            return None
        key = _organization_key(team_id)
        if key not in self.index:
            raise MultiLeagueV2RunnerError("organization state is missing at roster transition")
        previous_players = {player_id for _role, player_id in previous}
        current_players = {player_id for _role, player_id in current}
        retained = len(previous_players.intersection(current_players))
        phi = self.spec.organization_retention_floor + (
            1.0 - self.spec.organization_retention_floor
        ) * retained / 5.0
        index = self.index[key]
        self.mean[index] *= phi
        self.covariance[index, :] *= phi
        self.covariance[:, index] *= phi
        self.covariance[index, index] += (
            1.0 - phi * phi
        ) * self.spec.organization_prior_variance
        self._cheap_checks()
        return phi


def _feature_vector(
    state: HierarchicalGaussianState,
    item: adapter.DevelopmentMap,
    home_leagues: Mapping[str, str],
) -> rating._FeatureVector:
    blue_home, red_home, status = rating._home_leagues(item, home_leagues)
    state.ensure_structural_keys(
        [league for league in (blue_home, red_home) if league is not None]
    )
    weights: dict[str, float] = defaultdict(float)
    for slot in item.blue_lineup.players:
        weights[rating._player_key(slot.player_id)] += state.spec.player_weight_per_role
    for slot in item.red_lineup.players:
        weights[rating._player_key(slot.player_id)] -= state.spec.player_weight_per_role
    weights[_organization_key(item.blue_lineup.team_id)] += state.spec.organization_weight
    weights[_organization_key(item.red_lineup.team_id)] -= state.spec.organization_weight
    if blue_home is not None:
        weights[rating._league_key(blue_home)] += 1.0
    if red_home is not None:
        weights[rating._league_key(red_home)] -= 1.0
    weights[rating.BLUE_SIDE_KEY] += 1.0
    return rating._FeatureVector(
        {key: value for key, value in weights.items() if abs(value) > 1e-15},
        blue_home,
        red_home,
        status,
    )


def replay_candidate(
    input_data: adapter.PrivateMultiLeagueRatingInput,
    candidate: CandidateSpec,
) -> ReplayResult:
    state = HierarchicalGaussianState(candidate)
    predictions: list[dict[str, Any]] = []
    pending: list[tuple[datetime, int, str, _PendingSeries]] = []
    sequence = 0
    applied_series = 0
    applied_maps = 0
    player_metadata: dict[str, dict[str, Any]] = {}
    team_lineups: dict[str, dict[str, Any]] = {}
    team_home_leagues: dict[str, str] = {}
    team_home_order: dict[str, tuple[datetime, int, str]] = {}
    forecast_lineups: dict[str, tuple[tuple[str, str], ...]] = {}
    bridge_diagnostics: dict[str, int] = defaultdict(int)
    roster_diagnostics: dict[str, int] = defaultdict(int)

    def apply(value: _PendingSeries) -> None:
        nonlocal applied_series, applied_maps
        for item, feature in zip(value.series.maps, value.features):
            state.update(feature.weights, item.blue_win)
        rating._record_available_metadata(
            value,
            player_metadata=player_metadata,
            team_lineups=team_lineups,
            team_home_leagues=team_home_leagues,
            team_home_order=team_home_order,
        )
        applied_series += 1
        applied_maps += len(value.series.maps)

    def flush(boundary: datetime) -> None:
        while pending and pending[0][0] < boundary:
            _available, _sequence, _identity, value = heapq.heappop(pending)
            apply(value)

    for series in input_data.development_series:
        start = adapter.source_local_datetime(series.source_local_start)
        flush(start)
        team_ids = _series_team_ids(series)
        state.transition_entities(rating._series_player_ids(series), team_ids, start)

        first_lineups = {
            lineup.team_id: _lineup_identity(lineup)
            for lineup in (
                series.maps[0].blue_lineup,
                series.maps[0].red_lineup,
            )
        }
        for team_id in team_ids:
            previous = forecast_lineups.get(team_id)
            phi = state.apply_roster_transition(
                team_id,
                previous,
                first_lineups[team_id],
            )
            if previous is None:
                roster_diagnostics["NO_PRIOR_OBSERVED_LINEUP"] += 1
            elif phi is None:
                roster_diagnostics["EXACT_LINEUP_STABLE"] += 1
            else:
                roster_diagnostics["EXACT_LINEUP_CHANGED"] += 1

        features: list[rating._FeatureVector] = []
        lineups_in_series: dict[str, set[tuple[tuple[str, str], ...]]] = defaultdict(set)
        for item in series.maps:
            for lineup in (item.blue_lineup, item.red_lineup):
                lineups_in_series[lineup.team_id].add(_lineup_identity(lineup))
            feature = _feature_vector(state, item, team_home_leagues)
            features.append(feature)
            bridge_diagnostics[feature.bridge_status] += 1
            probability, latent_mean, latent_variance = state.predict(feature.weights)
            predictions.append(
                {
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
            )
        for team_id, identities in lineups_in_series.items():
            if len(identities) > 1:
                roster_diagnostics["INTRA_SERIES_LINEUP_CHANGE_OBSERVED"] += 1
        for item in series.maps:
            for lineup in (item.blue_lineup, item.red_lineup):
                forecast_lineups[lineup.team_id] = _lineup_identity(lineup)

        available_at = adapter.source_local_datetime(
            series.source_local_end
        ) + timedelta(hours=rating.AVAILABILITY_EMBARGO_HOURS)
        pending_value = _PendingSeries(available_at, series, tuple(features))
        heapq.heappush(
            pending,
            (available_at, sequence, series.series_id, pending_value),
        )
        sequence += 1

    cutoff = adapter.SEALED_FINAL_START.to_pydatetime()
    flush(cutoff)
    remaining_players = [
        key.removeprefix("player:")
        for key in state.keys
        if key.startswith("player:")
        and not key.startswith(f"player:{ORGANIZATION_ID_PREFIX}")
    ]
    remaining_teams = [
        key.removeprefix(f"player:{ORGANIZATION_ID_PREFIX}")
        for key in state.keys
        if key.startswith(f"player:{ORGANIZATION_ID_PREFIX}")
    ]
    state.transition_entities(remaining_players, remaining_teams, cutoff)
    state.assert_psd()
    return ReplayResult(
        candidate=candidate,
        predictions=predictions,
        state=state,
        player_metadata=player_metadata,
        team_lineups=team_lineups,
        team_home_leagues=team_home_leagues,
        bridge_diagnostics=dict(sorted(bridge_diagnostics.items())),
        roster_transition_diagnostics=dict(sorted(roster_diagnostics.items())),
        applied_series=applied_series,
        applied_maps=applied_maps,
    )


def _window_rows(
    rows: Sequence[Mapping[str, Any]],
    window: Mapping[str, Any] | None = None,
) -> list[Mapping[str, Any]]:
    if window is None:
        start = datetime.fromisoformat("2025-07-01T00:00:00")
        end = datetime.fromisoformat("2026-04-01T00:00:00")
    else:
        start = datetime.fromisoformat(str(window["start_inclusive"]))
        end = datetime.fromisoformat(str(window["end_exclusive"]))
    return [
        row
        for row in rows
        if start
        <= adapter.source_local_datetime(str(row["source_local_start"]))
        < end
    ]


def _metric_bundle(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall = rating._metric_payload(rows)
    return {
        "overall": overall,
        "by_domestic_league": [
            {
                "league": league,
                **rating._metric_payload(
                    [row for row in rows if row.get("league") == league]
                ),
            }
            for league in adapter.DOMESTIC_LEAGUES
        ],
        "by_roster_change_stratum": [
            {
                "roster_change_stratum": stratum,
                **rating._metric_payload(
                    [
                        row
                        for row in rows
                        if row.get("roster_change_stratum") == stratum
                    ]
                ),
            }
            for stratum in benchmark.ROSTER_STRATA
        ],
    }


def _attach_roster_strata(
    rows: Sequence[Mapping[str, Any]],
    organization_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    strata = {
        str(row["game_id"]): {
            "blue_roster_status": row["blue_roster_status"],
            "red_roster_status": row["red_roster_status"],
            "roster_change_stratum": row["roster_change_stratum"],
        }
        for row in organization_rows
    }
    if {str(row["game_id"]) for row in rows} != set(strata):
        raise MultiLeagueV2RunnerError("candidate and baseline map identities differ")
    return [dict(row) | strata[str(row["game_id"])] for row in rows]


def _series_macro_value(bundle: Mapping[str, Any], metric: str) -> float:
    try:
        value = float(bundle["overall"]["series_macro"][metric])
    except (KeyError, TypeError, ValueError) as error:
        raise MultiLeagueV2RunnerError("adaptive metric is unavailable") from error
    if not math.isfinite(value):
        raise MultiLeagueV2RunnerError("adaptive metric is non-finite")
    return value


def _candidate_report(
    replay: ReplayResult,
    rows: Sequence[Mapping[str, Any]],
    baseline_windows: Mapping[str, Mapping[str, Any]],
    selection_rule: Mapping[str, Any],
) -> dict[str, Any]:
    windows = []
    regrets = []
    for window in protocol.DISCOVERY_WINDOWS:
        selected = _window_rows(rows, window)
        metrics = _metric_bundle(selected)
        log_loss = _series_macro_value(metrics, "log_loss")
        baseline = baseline_windows[str(window["window_id"])]
        baseline_log_loss = min(
            float(baseline["player"]["series_macro"]["log_loss"]),
            float(baseline["organization"]["series_macro"]["log_loss"]),
        )
        regret = log_loss - baseline_log_loss
        regrets.append(regret)
        windows.append(
            {
                **dict(window),
                "metrics": metrics,
                "better_baseline_log_loss": baseline_log_loss,
                "log_loss_regret": regret,
            }
        )
    pooled = _metric_bundle(_window_rows(rows))
    minimum_series = int(selection_rule["minimum_series_per_scored_window"])
    finite_psd = True
    enough = all(
        int(item["metrics"]["overall"]["series"]) >= minimum_series
        for item in windows
    )
    worst_regret = max(regrets)
    eligible = (
        finite_psd
        and enough
        and worst_regret
        <= float(selection_rule["maximum_allowed_worst_window_log_loss_regret"])
    )
    return {
        "candidate": {
            "candidate_id": replay.candidate.candidate_id,
            "player_weight_per_role": replay.candidate.player_weight_per_role,
            "player_prior_variance": replay.candidate.player_prior_variance,
            "player_process_variance_per_day": (
                replay.candidate.player_process_variance_per_day
            ),
            "organization_weight": replay.candidate.organization_weight,
            "organization_prior_variance": replay.candidate.organization_prior_variance,
            "organization_process_variance_per_day": (
                replay.candidate.organization_process_variance_per_day
            ),
            "organization_retention_floor": (
                replay.candidate.organization_retention_floor
            ),
        },
        "windows": windows,
        "pooled_adaptive_development": pooled,
        "selection_diagnostics": {
            "finite_psd_replay": finite_psd,
            "minimum_series_per_window_met": enough,
            "worst_window_log_loss_regret": worst_regret,
            "maximum_allowed_worst_window_log_loss_regret": selection_rule[
                "maximum_allowed_worst_window_log_loss_regret"
            ],
            "eligible": eligible,
        },
        "replay": {
            "applied_series": replay.applied_series,
            "applied_maps": replay.applied_maps,
            "bridge_diagnostics": replay.bridge_diagnostics,
            "roster_transition_diagnostics": replay.roster_transition_diagnostics,
            "posterior_psd": replay.state.assert_psd(),
        },
    }


def _posterior_payload(replay: ReplayResult) -> dict[str, Any]:
    state = replay.state
    players = []
    for player_id, metadata in sorted(replay.player_metadata.items()):
        key = rating._player_key(player_id)
        if key not in state.index:
            continue
        index = state.index[key]
        mean = float(state.mean[index])
        sd = math.sqrt(float(state.covariance[index, index]))
        players.append(
            {
                key_name: item
                for key_name, item in metadata.items()
                if key_name != "_order"
            }
            | {
                "component_status": "ESTIMATED",
                "posterior_mean_logit": mean,
                "posterior_sd_logit": sd,
                "display_rating_mean": rating.DISPLAY_ANCHOR
                + rating.DISPLAY_LOGIT_SCALE * mean,
                "display_rating_sd": rating.DISPLAY_LOGIT_SCALE * sd,
                "outcome_evidence_updates": state.evidence_counts.get(key, 0),
            }
        )

    teams = []
    for team_id, lineup in sorted(replay.team_lineups.items()):
        player_weights = {
            rating._player_key(str(item["player_id"])): replay.candidate.player_weight_per_role
            for item in lineup["players"]
        }
        organization_weights = {
            _organization_key(team_id): replay.candidate.organization_weight
        }
        joint_weights = dict(player_weights) | dict(organization_weights)
        player_mean, player_variance = state.moments(player_weights)
        org_mean, org_variance = state.moments(organization_weights)
        joint_mean, joint_variance = state.moments(joint_weights)
        home_league = replay.team_home_leagues.get(team_id)
        league_component: dict[str, Any]
        league_key = rating._league_key(home_league) if home_league else None
        if (
            league_key is None
            or league_key not in state.index
            or state.evidence_counts.get(league_key, 0) <= 0
        ):
            league_component = {
                "status": "UNAVAILABLE",
                "posterior_mean_logit": None,
                "posterior_sd_logit": None,
                "reason": "home_league_effect_not_identified_by_available_bridge_outcomes",
            }
        else:
            league_mean, league_variance = state.moments({league_key: 1.0})
            league_component = {
                "status": "ESTIMATED",
                "posterior_mean_logit": league_mean,
                "posterior_sd_logit": math.sqrt(league_variance),
                "reason": None,
            }
        teams.append(
            {
                key_name: item
                for key_name, item in lineup.items()
                if key_name != "_order"
            }
            | {
                "roster_semantics": (
                    "LAST_OBSERVED_HISTORICAL_LINEUP_NOT_PRE_EVENT_AUTHORITY"
                ),
                "components": {
                    "player_aggregate": {
                        "status": "ESTIMATED",
                        "posterior_mean_logit": player_mean,
                        "posterior_sd_logit": math.sqrt(player_variance),
                        "reason": None,
                    },
                    "organization_residual": {
                        "status": "ESTIMATED",
                        "posterior_mean_logit": org_mean,
                        "posterior_sd_logit": math.sqrt(org_variance),
                        "reason": None,
                    },
                    "league_adjustment": league_component,
                    "lineup_synergy": {
                        "status": "UNAVAILABLE",
                        "posterior_mean_logit": None,
                        "posterior_sd_logit": None,
                        "reason": "no_identified_lineup_specific_interaction_model",
                    },
                    "team_policy": {
                        "status": "UNAVAILABLE",
                        "posterior_mean_logit": None,
                        "posterior_sd_logit": None,
                        "reason": "no_independent_pre_event_policy_measurement_model",
                    },
                },
                "joint_player_plus_organization": {
                    "status": "ESTIMATED",
                    "posterior_mean_logit": joint_mean,
                    "posterior_sd_logit": math.sqrt(joint_variance),
                    "display_rating_mean": rating.DISPLAY_ANCHOR
                    + rating.DISPLAY_LOGIT_SCALE * joint_mean,
                    "display_rating_sd": rating.DISPLAY_LOGIT_SCALE
                    * math.sqrt(joint_variance),
                    "identifiability": (
                        "joint contrast identified; component allocation prior-regularized"
                    ),
                },
                "unavailable_components_are_not_zero": True,
            }
        )
    teams.sort(
        key=lambda item: (
            -float(item["joint_player_plus_organization"]["display_rating_mean"]),
            str(item["team_id"]),
        )
    )
    return {
        "as_of_exclusive": "2026-04-01T00:00:00",
        "candidate_id": replay.candidate.candidate_id,
        "players": players,
        "teams": teams,
        "ratings_are_non_authorizing_adaptive_development_outputs": True,
    }


def _read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise MultiLeagueV2RunnerError(f"cannot read bound artifact: {path}") from error
    if not isinstance(value, dict):
        raise MultiLeagueV2RunnerError(f"bound artifact is not an object: {path}")
    return raw, value


def _source_record(root: Path, locator: str, kind: str) -> dict[str, Any]:
    try:
        raw = (root / locator).read_bytes()
    except OSError as error:
        raise MultiLeagueV2RunnerError(f"bound source is unavailable: {locator}") from error
    return {
        "kind": kind,
        "locator": locator,
        "bytes": len(raw),
        "raw_sha256": _sha256(raw),
    }


def build_adaptive_development_artifact(
    root: Path | str = Path("."),
    *,
    built_at: str,
) -> dict[str, Any]:
    repo_root = Path(root)
    protocol_raw, protocol_payload = _read_object(repo_root / DEFAULT_PROTOCOL)
    try:
        protocol_payload = protocol.validate_protocol_lock(
            protocol_payload,
            root=repo_root,
        )
    except protocol.MultiLeagueV2ProtocolError as error:
        raise MultiLeagueV2RunnerError("protocol lock is invalid") from error
    boundary = protocol_payload["information_boundary"]
    if (
        boundary["sealed_final_targets_accessed"] is not False
        or protocol_payload["sealed_final_gate"]["opened"] is not False
    ):
        raise MultiLeagueV2RunnerError("sealed-final isolation is not intact")
    binding = protocol_payload["input_binding"]
    input_data = adapter.load_multileague_development_input(
        expected_maps_sha256=str(binding["maps_sha256"]),
        expected_players_sha256=str(binding["players_sha256"]),
    )
    if input_data.cluster_partition_sha256 != binding["cluster_partition_sha256"]:
        raise MultiLeagueV2RunnerError("adapter partition no longer matches protocol")
    if input_data.sealed_selected_metadata_sha256 != binding[
        "sealed_selected_metadata_sha256"
    ]:
        raise MultiLeagueV2RunnerError("sealed metadata cohort no longer matches protocol")

    specs = [
        CandidateSpec.from_payload(item)
        for item in protocol_payload["candidate_family"]["candidates"]
    ]
    player_replay = rating._replay(
        input_data,
        next(
            item
            for item in rating.CANDIDATES
            if item.candidate_id == "random_walk_no_reset"
        ),
    )
    organization_replay = benchmark._organization_replay(
        input_data,
        next(
            item
            for item in benchmark.ORGANIZATION_CANDIDATES
            if item.candidate_id == "organization_random_walk_no_reset"
        ),
    )
    player_rows = benchmark._attach_roster_strata(
        player_replay.predictions,
        organization_replay.predictions,
    )
    organization_rows = organization_replay.predictions
    baseline_windows: dict[str, dict[str, Any]] = {}
    for window in protocol.DISCOVERY_WINDOWS:
        baseline_windows[str(window["window_id"])] = {
            "player": rating._metric_payload(_window_rows(player_rows, window)),
            "organization": rating._metric_payload(
                _window_rows(organization_rows, window)
            ),
        }

    selection_rule = protocol_payload["adaptive_development"]["selection_rule"]
    reports = []
    replays: dict[str, ReplayResult] = {}
    for spec in specs:
        replay = replay_candidate(input_data, spec)
        rows = _attach_roster_strata(
            replay.predictions,
            organization_replay.predictions,
        )
        reports.append(
            _candidate_report(replay, rows, baseline_windows, selection_rule)
        )
        replays[spec.candidate_id] = replay
    eligible = [
        item
        for item in reports
        if item["selection_diagnostics"]["eligible"] is True
    ]
    selected = (
        min(
            eligible,
            key=lambda item: (
                item["selection_diagnostics"]["worst_window_log_loss_regret"],
                _series_macro_value(item["pooled_adaptive_development"], "log_loss"),
                _series_macro_value(item["pooled_adaptive_development"], "brier"),
                item["candidate"]["candidate_id"],
            ),
        )
        if eligible
        else None
    )
    selected_id = None if selected is None else selected["candidate"]["candidate_id"]
    posterior = None if selected_id is None else _posterior_payload(replays[selected_id])

    source_locks = [
        _source_record(repo_root, DEFAULT_PROTOCOL.as_posix(), "protocol_lock"),
        _source_record(repo_root, adapter.DEFAULT_MAPS_LOCATOR, "warehouse_maps"),
        _source_record(repo_root, adapter.DEFAULT_PLAYERS_LOCATOR, "warehouse_players"),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_development.py",
            "input_adapter_source",
        ),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_runner.py",
            "predecessor_player_source",
        ),
        _source_record(
            repo_root,
            "lol_kills/v2/ratings/player/multileague_benchmark.py",
            "predecessor_organization_source",
        ),
        _source_record(
            repo_root,
            protocol.SOURCE_LOCATOR,
            "protocol_source",
        ),
        _source_record(repo_root, SOURCE_LOCATOR, "adaptive_runner_source"),
    ]
    if source_locks[0]["raw_sha256"] != _sha256(protocol_raw):
        raise MultiLeagueV2RunnerError("protocol raw byte binding is inconsistent")

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "built_at": built_at,
        "result_state": RESULT_SELECTED if selected is not None else RESULT_NO_ELIGIBLE,
        "protocol": {
            "locator": DEFAULT_PROTOCOL.as_posix(),
            "raw_sha256": _sha256(protocol_raw),
            "artifact_sha256": protocol_payload["artifact_sha256"],
            "validation_disclosure_status": protocol_payload[
                "validation_disclosure"
            ]["status"],
        },
        "input": {
            "maps_sha256": input_data.maps_sha256,
            "players_sha256": input_data.players_sha256,
            "cluster_partition_sha256": input_data.cluster_partition_sha256,
            "development_selected_rows_sha256": (
                input_data.development_selected_rows_sha256
            ),
            "sealed_selected_metadata_sha256": (
                input_data.sealed_selected_metadata_sha256
            ),
            "sealed_metadata_series": input_data.coverage["sealed_metadata_series"],
            "sealed_metadata_maps": input_data.coverage["sealed_metadata_maps"],
            "sealed_final_targets_accessed": False,
        },
        "baselines": {
            "windows": [
                {
                    **dict(window),
                    **baseline_windows[str(window["window_id"])],
                }
                for window in protocol.DISCOVERY_WINDOWS
            ],
            "pooled_adaptive_development": {
                "player": _metric_bundle(_window_rows(player_rows)),
                "organization": _metric_bundle(_window_rows(organization_rows)),
            },
        },
        "candidate_results": reports,
        "selection": {
            "selection_is_adaptive_not_independent_validation": True,
            "eligible_candidate_ids": [
                item["candidate"]["candidate_id"] for item in eligible
            ],
            "selected_candidate_id": selected_id,
            "selection_rank": selection_rule["stage_3_rank"],
            "sealed_final_opened": False,
            "candidate_eligible_for_independently_approved_sealed_evaluation": (
                selected is not None
            ),
        },
        "adaptive_posterior": posterior,
        "sealed_final": {
            "opened": False,
            "targets_accessed": False,
            "series": input_data.coverage["sealed_metadata_series"],
            "maps": input_data.coverage["sealed_metadata_maps"],
            "opening_authority_present": False,
            "gate_passed": False,
        },
        "source_locks": source_locks,
        "claim_ceiling": protocol_payload["claim_ceiling"],
        "decision_outputs": {
            "match_probability": None,
            "fair_odds": None,
            "expected_value": None,
            "bet_recommendation": None,
        },
    }
    payload["artifact_sha256"] = _artifact_sha256(payload)
    return validate_adaptive_development_artifact(payload, root=repo_root)


def validate_adaptive_development_artifact(
    payload: Mapping[str, Any],
    *,
    root: Path | str = Path("."),
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise MultiLeagueV2RunnerError("adaptive artifact must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION:
        raise MultiLeagueV2RunnerError("adaptive artifact schema is unsupported")
    if value.get("result_state") not in {RESULT_SELECTED, RESULT_NO_ELIGIBLE}:
        raise MultiLeagueV2RunnerError("adaptive artifact result state is invalid")
    declared = _require_sha256(value.get("artifact_sha256"), "artifact_sha256")
    if declared != _artifact_sha256(value):
        raise MultiLeagueV2RunnerError("adaptive artifact digest mismatch")
    if (
        (value.get("input") or {}).get("sealed_final_targets_accessed") is not False
        or (value.get("sealed_final") or {}).get("opened") is not False
        or (value.get("sealed_final") or {}).get("targets_accessed") is not False
        or (value.get("selection") or {}).get("sealed_final_opened") is not False
    ):
        raise MultiLeagueV2RunnerError("adaptive artifact opened sealed-final targets")
    outputs = value.get("decision_outputs") or {}
    if set(outputs) != {
        "match_probability",
        "fair_odds",
        "expected_value",
        "bet_recommendation",
    } or any(item is not None for item in outputs.values()):
        raise MultiLeagueV2RunnerError("adaptive artifact contains decision outputs")

    candidate_results = value.get("candidate_results")
    if not isinstance(candidate_results, list) or len(candidate_results) != 12:
        raise MultiLeagueV2RunnerError("adaptive candidate result inventory changed")
    expected_ids = {
        item["candidate_id"] for item in protocol._candidate_payloads()
    }
    actual_ids = {
        (item.get("candidate") or {}).get("candidate_id")
        for item in candidate_results
        if isinstance(item, Mapping)
    }
    if actual_ids != expected_ids:
        raise MultiLeagueV2RunnerError("adaptive candidate identities changed")
    eligible = sorted(
        item["candidate"]["candidate_id"]
        for item in candidate_results
        if (item.get("selection_diagnostics") or {}).get("eligible") is True
    )
    selection = value.get("selection") or {}
    if sorted(selection.get("eligible_candidate_ids") or []) != eligible:
        raise MultiLeagueV2RunnerError("adaptive eligible candidate inventory changed")
    selected_id = selection.get("selected_candidate_id")
    if selected_id is not None and selected_id not in eligible:
        raise MultiLeagueV2RunnerError("adaptive selected candidate is not eligible")
    if (selected_id is None) != (value.get("adaptive_posterior") is None):
        raise MultiLeagueV2RunnerError("adaptive posterior selection binding changed")
    if (selected_id is None) != (value.get("result_state") == RESULT_NO_ELIGIBLE):
        raise MultiLeagueV2RunnerError("adaptive result state and selection disagree")
    if selected_id is not None:
        posterior = value.get("adaptive_posterior") or {}
        if posterior.get("candidate_id") != selected_id:
            raise MultiLeagueV2RunnerError("adaptive posterior candidate binding changed")
        for team in posterior.get("teams") or []:
            components = (team or {}).get("components") or {}
            for name in ("lineup_synergy", "team_policy"):
                component = components.get(name) or {}
                if (
                    component.get("status") != "UNAVAILABLE"
                    or component.get("posterior_mean_logit") is not None
                    or component.get("posterior_sd_logit") is not None
                ):
                    raise MultiLeagueV2RunnerError(
                        "unavailable team component was converted to a numeric value"
                    )

    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != 8:
        raise MultiLeagueV2RunnerError("adaptive source-lock inventory changed")
    repo_root = Path(root)
    seen: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise MultiLeagueV2RunnerError("adaptive source-lock record is malformed")
        locator = record.get("locator")
        if not isinstance(locator, str) or not locator or locator in seen:
            raise MultiLeagueV2RunnerError("adaptive source-lock locator is invalid")
        seen.add(locator)
        expected = _require_sha256(record.get("raw_sha256"), f"{locator} raw_sha256")
        try:
            raw = (repo_root / locator).read_bytes()
        except OSError as error:
            raise MultiLeagueV2RunnerError(f"bound source is unavailable: {locator}") from error
        if len(raw) != record.get("bytes") or _sha256(raw) != expected:
            raise MultiLeagueV2RunnerError(f"bound source drifted: {locator}")
    return value


def write_adaptive_artifact_no_clobber(
    path: Path,
    payload: Mapping[str, Any],
) -> str:
    encoded = (json.dumps(dict(payload), indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"refusing to replace existing adaptive artifact: {path}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return _sha256(encoded)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--built-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_adaptive_development_artifact(built_at=args.built_at)
    raw_sha256 = write_adaptive_artifact_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "eligible_candidate_ids": payload["selection"][
                    "eligible_candidate_ids"
                ],
                "selected_candidate_id": payload["selection"][
                    "selected_candidate_id"
                ],
                "sealed_final_opened": payload["sealed_final"]["opened"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CandidateSpec",
    "DEFAULT_OUTPUT",
    "HierarchicalGaussianState",
    "MultiLeagueV2RunnerError",
    "RESULT_NO_ELIGIBLE",
    "RESULT_SELECTED",
    "SCHEMA_VERSION",
    "build_adaptive_development_artifact",
    "replay_candidate",
    "validate_adaptive_development_artifact",
    "write_adaptive_artifact_no_clobber",
]
