"""Capture outcome-free, pre-event predictions for the v3 ratings holdout.

Each receipt embeds the exact roster and patch evidence, replays the frozen
candidate and both locked comparators from the immutable source, and records
their probabilities before event start. These probabilities exist only for
future evaluation. They are not rating, probability, recommendation, or
betting authority.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills import pregame_roster_capture as roster_capture

from . import multileague_benchmark as benchmark
from . import multileague_development as adapter
from . import multileague_runner as rating
from . import multileague_v2_runner as hierarchical
from .multileague_v3_future_protocol import FUTURE_SEALED_START
from .multileague_v3_preflight_v2 import _future_boundary, _state_sha256
from .multileague_v3_preflight_v3_registry import (
    REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
    REGISTERED_PREFLIGHT_LOCATOR,
    REGISTERED_PREFLIGHT_RAW_SHA256,
    validate_registered_source_preflight_v3,
)
from .multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v3,
)
from .multileague_v3_source_registry_v2 import (
    MANIFEST_CANONICAL_SHA256,
    MANIFEST_LOCATOR,
    MANIFEST_RAW_SHA256,
    PACKAGE_ID,
    validate_registered_source_snapshot_v2,
)


ROOT = Path(__file__).resolve().parents[4]
RECEIPT_SCHEMA_VERSION = "scryglass:multileague-rating-v3-pre-event-prediction:v2"
REGISTRY_SCHEMA_VERSION = "scryglass:multileague-rating-v3-prediction-ledger:v2"
RESULT_STATE = "SYSTEM_CLOCKED_PRE_EVENT_EVALUATION_PREDICTION_CAPTURED_UNREVIEWED"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_prediction_ledger.py"
)
RECEIPT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/multileague-v3/predictions"
)
DEFAULT_REGISTRY = Path(
    "data/lol/v2/evaluation/multileague-v3/prediction-ledger.json"
)
PATCH_RECEIPT_SCHEMA = "scryglass:leaguepedia-patch-revisions:v1"
PATCH_RE = re.compile(r"^26\.(?:0[1-9]|1[0-9]|2[0-9])$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MODEL_IDS = (
    "hierarchical-orgw100-orgv025-retain100",
    "predecessor-player-random-walk",
    "predecessor-organization-random-walk",
)
OUTCOME_KEYS = frozenset(
    {
        "outcome",
        "outcomes",
        "result",
        "results",
        "winner",
        "winnerteamid",
        "winteam",
        "lossteam",
        "team1score",
        "team2score",
        "won",
    }
)
AUTHORITY_KEYS = (
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    "lol_kills/pregame_roster_capture.py",
    "lol_kills/etl/leaguepedia_patch_revisions.py",
    "lol_kills/v2/ratings/player/multileague_development.py",
    "lol_kills/v2/ratings/player/multileague_runner.py",
    "lol_kills/v2/ratings/player/multileague_benchmark.py",
    "lol_kills/v2/ratings/player/multileague_v2_runner.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v2.py",
    "lol_kills/v2/ratings/player/multileague_v3_preflight_v3_registry.py",
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_source_registry_v2.py",
    REGISTERED_PREFLIGHT_LOCATOR.as_posix(),
    REGISTERED_PROTOCOL_LOCATOR.as_posix(),
    MANIFEST_LOCATOR.as_posix(),
)


class PredictionLedgerError(RuntimeError):
    """A pre-event prediction or its outcome-free ledger failed closed."""


@dataclass(frozen=True)
class _ForecastMap:
    league: str
    blue_lineup: adapter.ObservedLineup
    red_lineup: adapter.ObservedLineup


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PredictionLedgerError("prediction value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise PredictionLedgerError(f"{label} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionLedgerError(f"{label} must be a non-empty string")
    return value.strip()


def _timestamp(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise PredictionLedgerError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PredictionLedgerError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], label: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PredictionLedgerError(
            f"{label} clock must return a timezone-aware datetime"
        )
    return value.astimezone(timezone.utc)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise PredictionLedgerError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                PredictionLedgerError(f"non-finite JSON number in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PredictionLedgerError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PredictionLedgerError(f"{label} must be a JSON object")
    return value


def _assert_no_outcomes(value: Any, path: str = "input") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise PredictionLedgerError(f"event outcome field is forbidden: {path}.{key}")
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file():
        raise PredictionLedgerError(f"bound prediction source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _validate_patch_receipt(value: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "schema_version",
        "fixture_id",
        "event_start",
        "as_of",
        "patch",
        "client_patch",
        "authority_status",
        "pregame_authorized",
        "blockers",
        "evidence",
        "evidence_hash",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise PredictionLedgerError("patch receipt keys changed")
    _assert_no_outcomes(value, "patch_receipt")
    if (
        value.get("schema_version") != PATCH_RECEIPT_SCHEMA
        or value.get("authority_status") != "pre_event_revision"
        or value.get("pregame_authorized") is not True
        or value.get("blockers") != []
    ):
        raise PredictionLedgerError("patch receipt is not pre-event authorized")
    patch = _nonempty(value.get("patch"), "patch")
    client = _nonempty(value.get("client_patch"), "client_patch")
    if not PATCH_RE.fullmatch(patch) or client != f"16.{int(patch.split('.')[1])}":
        raise PredictionLedgerError("patch/client-patch identity changed")
    event_start = _timestamp(value.get("event_start"), "patch.event_start")
    as_of = _timestamp(value.get("as_of"), "patch.as_of")
    if as_of >= event_start:
        raise PredictionLedgerError("patch receipt was not captured before event")
    evidence = value.get("evidence")
    if not isinstance(evidence, Mapping):
        raise PredictionLedgerError("patch evidence is unavailable")
    revision_time = _timestamp(
        evidence.get("revision_timestamp"), "patch.evidence.revision_timestamp"
    )
    if revision_time >= event_start:
        raise PredictionLedgerError("patch revision is not strictly pre-event")
    expected_evidence_hash = _canonical_sha256(
        {"fixture_id": value["fixture_id"], "evidence": evidence}
    )
    if value.get("evidence_hash") != expected_evidence_hash:
        raise PredictionLedgerError("patch evidence hash changed")
    return dict(value)


def _lineup(team: Mapping[str, Any]) -> adapter.ObservedLineup:
    players = tuple(
        adapter.PlayerSlot(
            role=str(player["role"]),
            player_id=str(player["player_id"]),
            player_name=str(player["display_name"]),
            team_id=str(team["organization_id"]),
        )
        for player in team["players"]
    )
    return adapter.ObservedLineup(
        side=str(team["side"]),
        team_id=str(team["organization_id"]),
        team_key=str(team["organization_id"]),
        team_name=str(team["organization_name"]),
        players=players,
    )


def _lineup_identity(lineup: adapter.ObservedLineup) -> tuple[tuple[str, str], ...]:
    return tuple((slot.role, slot.player_id) for slot in lineup.players)


def _historical_lineup_identity(
    metadata: Mapping[str, Any] | None,
) -> tuple[tuple[str, str], ...] | None:
    if not metadata:
        return None
    players = metadata.get("players")
    if not isinstance(players, list) or len(players) != len(adapter.ROLE_ORDER):
        return None
    return tuple((str(item["role"]), str(item["player_id"])) for item in players)


def _roster_status(
    previous: tuple[tuple[str, str], ...] | None,
    current: tuple[tuple[str, str], ...],
) -> str:
    if previous is None:
        return "NO_PRIOR_EXACT_LINEUP"
    return "STABLE" if previous == current else "CHANGED"


def _prediction(state: rating._GaussianState, weights: Mapping[str, float]) -> dict[str, Any]:
    probability, latent_mean, latent_variance = state.predict(weights)
    return {
        "p_blue": probability,
        "p_red": 1.0 - probability,
        "latent_mean": latent_mean,
        "latent_variance": latent_variance,
    }


def _player_components(
    replay: hierarchical.ReplayResult,
    lineups: Sequence[adapter.ObservedLineup],
) -> list[dict[str, Any]]:
    rows = []
    for lineup in lineups:
        for slot in lineup.players:
            key = rating._player_key(slot.player_id)
            mean, variance = replay.state.moments({key: 1.0})
            evidence = int(replay.state.evidence_counts.get(key, 0))
            rows.append(
                {
                    "side": lineup.side,
                    "role": slot.role,
                    "player_id": slot.player_id,
                    "display_name": slot.player_name,
                    "status": "ESTIMATED" if evidence else "PRIOR_ONLY",
                    "posterior_mean_logit": mean,
                    "posterior_sd_logit": math.sqrt(variance),
                    "display_rating_mean": rating.DISPLAY_ANCHOR
                    + rating.DISPLAY_LOGIT_SCALE * mean,
                    "display_rating_sd": rating.DISPLAY_LOGIT_SCALE
                    * math.sqrt(variance),
                    "outcome_evidence_updates": evidence,
                }
            )
    return rows


def _team_components(
    replay: hierarchical.ReplayResult,
    lineups: Sequence[adapter.ObservedLineup],
    roster_statuses: Mapping[str, str],
    retention: Mapping[str, float | None],
) -> list[dict[str, Any]]:
    rows = []
    for lineup in lineups:
        player_weights = {
            rating._player_key(slot.player_id): replay.candidate.player_weight_per_role
            for slot in lineup.players
        }
        organization_weights = {
            hierarchical._organization_key(lineup.team_id): (
                replay.candidate.organization_weight
            )
        }
        joint_weights = dict(player_weights) | dict(organization_weights)
        player_mean, player_variance = replay.state.moments(player_weights)
        organization_mean, organization_variance = replay.state.moments(
            organization_weights
        )
        joint_mean, joint_variance = replay.state.moments(joint_weights)
        rows.append(
            {
                "side": lineup.side,
                "organization_id": lineup.team_id,
                "organization_name": lineup.team_name,
                "roster_status": roster_statuses[lineup.side],
                "organization_retention_phi": retention[lineup.side],
                "components": {
                    "player_aggregate": {
                        "status": "ESTIMATED_OR_PRIOR_MIX",
                        "posterior_mean_logit": player_mean,
                        "posterior_sd_logit": math.sqrt(player_variance),
                    },
                    "organization_residual": {
                        "status": "ESTIMATED_OR_PRIOR",
                        "posterior_mean_logit": organization_mean,
                        "posterior_sd_logit": math.sqrt(organization_variance),
                    },
                    "lineup_synergy": {
                        "status": "UNAVAILABLE",
                        "posterior_mean_logit": None,
                        "posterior_sd_logit": None,
                    },
                    "team_policy": {
                        "status": "UNAVAILABLE",
                        "posterior_mean_logit": None,
                        "posterior_sd_logit": None,
                    },
                },
                "joint_player_plus_organization": {
                    "status": "ESTIMATED_OR_PRIOR_MIX",
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
    return rows


def _fit_and_forecast(
    *,
    root: Path,
    league: str,
    event_start: datetime,
    blue_lineup: adapter.ObservedLineup,
    red_lineup: adapter.ObservedLineup,
) -> dict[str, Any]:
    source = validate_registered_source_snapshot_v2(root=root)
    protocol = validate_registered_future_protocol_v3(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    files = source["files"]
    with _future_boundary():
        input_data = adapter.load_multileague_development_input(
            expected_maps_sha256=files["maps"]["raw_sha256"],
            expected_players_sha256=files["players"]["raw_sha256"],
            root=root,
            maps_locator=files["maps"]["locator"],
            players_locator=files["players"]["locator"],
        )
        rating._validate_input(
            input_data,
            expected_maps_sha256=files["maps"]["raw_sha256"],
            expected_players_sha256=files["players"]["raw_sha256"],
        )
        candidate_spec = hierarchical.CandidateSpec.from_payload(
            protocol["locked_candidate"]["definition"]
        )
        candidate_replay = hierarchical.replay_candidate(input_data, candidate_spec)
        player_candidate = next(
            item
            for item in rating.CANDIDATES
            if item.candidate_id == "random_walk_no_reset"
        )
        player_replay = rating._replay(input_data, player_candidate)
        organization_candidate = next(
            item
            for item in benchmark.ORGANIZATION_CANDIDATES
            if item.candidate_id == "organization_random_walk_no_reset"
        )
        organization_replay = benchmark._organization_replay(
            input_data, organization_candidate
        )

    pre_event_state_sha256 = _state_sha256(candidate_replay)
    if (
        pre_event_state_sha256
        != preflight["numerical_preflight"]["posterior_state_sha256"]
    ):
        raise PredictionLedgerError("candidate preflight state did not replay")
    event_naive = event_start.astimezone(timezone.utc).replace(tzinfo=None)
    lineups = (blue_lineup, red_lineup)
    player_ids = [slot.player_id for lineup in lineups for slot in lineup.players]
    team_ids = [lineup.team_id for lineup in lineups]
    candidate_replay.state.transition_entities(player_ids, team_ids, event_naive)
    player_replay.state.transition_players(player_ids, event_naive)
    organization_replay.state.transition_players(
        [benchmark._organization_identity(team_id) for team_id in team_ids],
        event_naive,
    )

    statuses: dict[str, str] = {}
    retention: dict[str, float | None] = {}
    for lineup in lineups:
        previous = _historical_lineup_identity(
            candidate_replay.team_lineups.get(lineup.team_id)
        )
        current = _lineup_identity(lineup)
        statuses[lineup.side] = _roster_status(previous, current)
        retention[lineup.side] = candidate_replay.state.apply_roster_transition(
            lineup.team_id,
            previous,
            current,
        )
    roster_change_stratum = (
        "NO_PRIOR_EXACT_LINEUP"
        if "NO_PRIOR_EXACT_LINEUP" in statuses.values()
        else "ONE_OR_BOTH_ROSTERS_CHANGED"
        if "CHANGED" in statuses.values()
        else "BOTH_ROSTERS_STABLE"
    )

    forecast = _ForecastMap(league, blue_lineup, red_lineup)
    candidate_feature = hierarchical._feature_vector(
        candidate_replay.state,
        forecast,
        candidate_replay.team_home_leagues,
    )
    player_feature = rating._feature_vector(
        player_replay.state,
        forecast,
        player_replay.team_home_leagues,
    )
    organization_feature = benchmark._organization_feature(
        organization_replay.state,
        forecast,
        candidate_replay.team_home_leagues,
    )
    if league in adapter.INTERNATIONAL_LEAGUES and candidate_feature.bridge_status not in {
        "INTERNATIONAL_BOTH_HOME_LEAGUES_KNOWN",
        "INTERNATIONAL_SAME_HOME_LEAGUE",
    }:
        raise PredictionLedgerError(
            "international prediction lacks both pre-event home-league identities"
        )

    model_predictions = {
        MODEL_IDS[0]: _prediction(candidate_replay.state, candidate_feature.weights),
        MODEL_IDS[1]: _prediction(player_replay.state, player_feature.weights),
        MODEL_IDS[2]: _prediction(
            organization_replay.state, organization_feature.weights
        ),
    }
    return {
        "protocol": protocol,
        "source": source,
        "preflight": preflight,
        "input": input_data,
        "candidate_replay": candidate_replay,
        "model_predictions": model_predictions,
        "player_comparator_state_sha256": rating._state_digest(player_replay.state),
        "organization_comparator_state_sha256": rating._state_digest(
            organization_replay.state
        ),
        "event_candidate_state_sha256": _state_sha256(candidate_replay),
        "bridge_status": candidate_feature.bridge_status,
        "blue_home_league": candidate_feature.blue_home_league,
        "red_home_league": candidate_feature.red_home_league,
        "roster_statuses": statuses,
        "roster_change_stratum": roster_change_stratum,
        "retention": retention,
        "players": _player_components(candidate_replay, lineups),
        "teams": _team_components(
            candidate_replay,
            lineups,
            statuses,
            retention,
        ),
    }


def build_pre_event_prediction_receipt(
    *,
    roster_receipt_raw: bytes,
    patch_receipt_raw: bytes,
    series_id: str,
    game_number: int,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    capture_time = _clock_sample(clock, "prediction capture")
    roster_object = _read_object(roster_receipt_raw, "roster receipt")
    patch_object = _read_object(patch_receipt_raw, "patch receipt")
    _assert_no_outcomes(patch_object, "patch_receipt")
    checked_roster = roster_capture.validate_pregame_roster_receipt(roster_object)
    checked_patch = _validate_patch_receipt(patch_object)
    event_start = _timestamp(checked_roster["event_start"], "event_start")
    event_source_time = event_start.astimezone(timezone.utc).replace(tzinfo=None)
    if event_source_time < FUTURE_SEALED_START:
        raise PredictionLedgerError("event predates the future holdout boundary")
    if capture_time >= event_start:
        raise PredictionLedgerError("prediction was not captured before event start")
    if capture_time < _timestamp(checked_roster["captured_at"], "roster.captured_at"):
        raise PredictionLedgerError("prediction predates its roster evidence")
    if capture_time < _timestamp(checked_patch["as_of"], "patch.as_of"):
        raise PredictionLedgerError("prediction predates its patch evidence")
    if checked_patch["fixture_id"] != checked_roster["event_id"]:
        raise PredictionLedgerError("patch and roster fixture identities differ")
    if _timestamp(checked_patch["event_start"], "patch.event_start") != event_start:
        raise PredictionLedgerError("patch and roster event starts differ")
    protocol = validate_registered_future_protocol_v3(root=root)
    if capture_time <= _timestamp(protocol["locked_at_utc"], "protocol.locked_at"):
        raise PredictionLedgerError("prediction predates the superseding protocol")
    if checked_roster["league"] not in protocol["future_holdout"]["eligibility"][
        "leagues"
    ]:
        raise PredictionLedgerError("event league is outside the frozen protocol")
    if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
        raise PredictionLedgerError("game_number must be a positive integer")

    blue_lineup = _lineup(checked_roster["teams"][0])
    red_lineup = _lineup(checked_roster["teams"][1])
    forecast = _fit_and_forecast(
        root=root,
        league=checked_roster["league"],
        event_start=event_start,
        blue_lineup=blue_lineup,
        red_lineup=red_lineup,
    )
    latest_source = forecast["preflight"]["source_snapshot"][
        "latest_observed_source_time"
    ]
    payload = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "captured_at_utc": capture_time.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": capture_time.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "capture_time_not_after_builder_observation": True,
        },
        "protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "future_holdout_start": FUTURE_SEALED_START.isoformat(),
        },
        "source_snapshot": {
            "package_id": PACKAGE_ID,
            "manifest_locator": MANIFEST_LOCATOR.as_posix(),
            "manifest_raw_sha256": MANIFEST_RAW_SHA256,
            "manifest_canonical_sha256": MANIFEST_CANONICAL_SHA256,
            "latest_observed_source_time": latest_source,
            "maps_raw_sha256": forecast["source"]["files"]["maps"]["raw_sha256"],
            "players_raw_sha256": forecast["source"]["files"]["players"][
                "raw_sha256"
            ],
        },
        "source_preflight": {
            "locator": REGISTERED_PREFLIGHT_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PREFLIGHT_RAW_SHA256,
            "artifact_sha256": REGISTERED_PREFLIGHT_ARTIFACT_SHA256,
            "posterior_state_sha256": forecast["preflight"]["numerical_preflight"][
                "posterior_state_sha256"
            ],
        },
        "event": {
            "event_id": checked_roster["event_id"],
            "series_id": _nonempty(series_id, "series_id"),
            "game_number": game_number,
            "event_start_utc": event_start.isoformat(),
            "source_time_utc_naive": event_source_time.isoformat(),
            "league": checked_roster["league"],
            "patch": checked_patch["patch"],
            "client_patch": checked_patch["client_patch"],
            "blue_organization_id": checked_roster["teams"][0]["organization_id"],
            "blue_organization_name": checked_roster["teams"][0][
                "organization_name"
            ],
            "red_organization_id": checked_roster["teams"][1]["organization_id"],
            "red_organization_name": checked_roster["teams"][1][
                "organization_name"
            ],
            "roster_change_stratum": forecast["roster_change_stratum"],
            "blue_roster_status": forecast["roster_statuses"]["blue"],
            "red_roster_status": forecast["roster_statuses"]["red"],
            "bridge_status": forecast["bridge_status"],
            "blue_home_league": forecast["blue_home_league"],
            "red_home_league": forecast["red_home_league"],
        },
        "input_receipts": {
            "roster": {
                "raw_sha256": _sha256_bytes(roster_receipt_raw),
                "canonical_sha256": checked_roster["receipt_sha256"],
                "receipt": roster_object,
            },
            "patch": {
                "raw_sha256": _sha256_bytes(patch_receipt_raw),
                "canonical_sha256": _canonical_sha256(patch_object),
                "receipt": patch_object,
            },
        },
        "evaluation_predictions": forecast["model_predictions"],
        "event_rating_diagnostics": {
            "players": forecast["players"],
            "teams": forecast["teams"],
            "candidate_event_state_sha256": forecast[
                "event_candidate_state_sha256"
            ],
            "player_comparator_event_state_sha256": forecast[
                "player_comparator_state_sha256"
            ],
            "organization_comparator_event_state_sha256": forecast[
                "organization_comparator_state_sha256"
            ],
            "component_allocation_is_prior_regularized": True,
            "ratings_are_evaluation_diagnostics_only": True,
        },
        "qualification": {
            "event_on_or_after_future_boundary": True,
            "prediction_strictly_before_event_start": True,
            "exact_ten_player_identity_present": True,
            "pre_event_roster_receipt_valid": True,
            "pre_event_patch_receipt_valid": True,
            "candidate_and_both_comparators_frozen": True,
            "system_clock_sampled_inside_builder": True,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
            "protocol_eligible_candidate": True,
            "independently_pinned_ledger_entry": False,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": (
            "This is an outcome-free pre-event prediction for future model evaluation. "
            "It does not authorize ratings, match probabilities, odds, expected value, "
            "recommendations, or wagers."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_pre_event_prediction_receipt(payload, root=root)


def validate_pre_event_prediction_receipt(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PredictionLedgerError("prediction receipt must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "prediction_receipt")
    if (
        value.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise PredictionLedgerError("prediction receipt identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise PredictionLedgerError("prediction receipt canonical hash mismatch")
    protocol = validate_registered_future_protocol_v3(root=root)
    preflight = validate_registered_source_preflight_v3(root=root)
    source = validate_registered_source_snapshot_v2(root=root)
    protocol_record = value.get("protocol") or {}
    if (
        protocol_record.get("raw_sha256") != REGISTERED_PROTOCOL_RAW_SHA256
        or protocol_record.get("artifact_sha256")
        != protocol.get("artifact_sha256")
    ):
        raise PredictionLedgerError("prediction protocol binding changed")
    source_record = value.get("source_snapshot") or {}
    if (
        source_record.get("package_id") != source.get("package_id")
        or source_record.get("manifest_raw_sha256") != MANIFEST_RAW_SHA256
        or source_record.get("manifest_canonical_sha256") != MANIFEST_CANONICAL_SHA256
    ):
        raise PredictionLedgerError("prediction source binding changed")
    preflight_record = value.get("source_preflight") or {}
    if (
        preflight_record.get("artifact_sha256") != preflight.get("artifact_sha256")
        or preflight_record.get("posterior_state_sha256")
        != preflight["numerical_preflight"]["posterior_state_sha256"]
    ):
        raise PredictionLedgerError("prediction preflight binding changed")
    event = value.get("event") or {}
    if set(event) != {
        "event_id",
        "series_id",
        "game_number",
        "event_start_utc",
        "source_time_utc_naive",
        "league",
        "patch",
        "client_patch",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
        "roster_change_stratum",
        "blue_roster_status",
        "red_roster_status",
        "bridge_status",
        "blue_home_league",
        "red_home_league",
    }:
        raise PredictionLedgerError("prediction event structure changed")
    event_start = _timestamp(event.get("event_start_utc"), "event.event_start_utc")
    captured_at = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": captured_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "capture_time_not_after_builder_observation": True,
    }:
        raise PredictionLedgerError("prediction clock attestation changed")
    if (
        captured_at >= event_start
        or event_start.astimezone(timezone.utc).replace(tzinfo=None)
        < FUTURE_SEALED_START
    ):
        raise PredictionLedgerError("prediction temporal boundary changed")
    inputs = value.get("input_receipts") or {}
    roster_record = inputs.get("roster") or {}
    patch_record = inputs.get("patch") or {}
    roster_object = roster_record.get("receipt")
    patch_object = patch_record.get("receipt")
    if not isinstance(roster_object, Mapping) or not isinstance(patch_object, Mapping):
        raise PredictionLedgerError("embedded prediction inputs are unavailable")
    _sha(roster_record.get("raw_sha256"), "roster.raw_sha256")
    _sha(patch_record.get("raw_sha256"), "patch.raw_sha256")
    checked_roster = roster_capture.validate_pregame_roster_receipt(roster_object)
    checked_patch = _validate_patch_receipt(patch_object)
    if (
        roster_record.get("canonical_sha256") != checked_roster["receipt_sha256"]
        or patch_record.get("canonical_sha256") != _canonical_sha256(patch_object)
        or checked_roster["event_id"] != event.get("event_id")
        or checked_patch["fixture_id"] != event.get("event_id")
        or _timestamp(checked_roster["event_start"], "roster.event_start")
        != event_start
        or _timestamp(checked_patch["event_start"], "patch.event_start")
        != event_start
        or checked_patch["patch"] != event.get("patch")
        or checked_roster["league"] != event.get("league")
        or checked_roster["teams"][0]["organization_id"]
        != event.get("blue_organization_id")
        or checked_roster["teams"][0]["organization_name"]
        != event.get("blue_organization_name")
        or checked_roster["teams"][1]["organization_id"]
        != event.get("red_organization_id")
        or checked_roster["teams"][1]["organization_name"]
        != event.get("red_organization_name")
        or captured_at
        < _timestamp(checked_roster["captured_at"], "roster.captured_at")
        or captured_at < _timestamp(checked_patch["as_of"], "patch.as_of")
        or captured_at
        <= _timestamp(protocol["locked_at_utc"], "protocol.locked_at")
    ):
        raise PredictionLedgerError("embedded event input binding changed")
    _nonempty(event.get("series_id"), "event.series_id")
    if (
        isinstance(event.get("game_number"), bool)
        or not isinstance(event.get("game_number"), int)
        or event["game_number"] < 1
        or event.get("roster_change_stratum")
        not in {
            "BOTH_ROSTERS_STABLE",
            "ONE_OR_BOTH_ROSTERS_CHANGED",
            "NO_PRIOR_EXACT_LINEUP",
        }
        or event.get("blue_roster_status") not in {"STABLE", "CHANGED", "NO_PRIOR_EXACT_LINEUP"}
        or event.get("red_roster_status") not in {"STABLE", "CHANGED", "NO_PRIOR_EXACT_LINEUP"}
    ):
        raise PredictionLedgerError("prediction event metadata is invalid")
    predictions = value.get("evaluation_predictions")
    if not isinstance(predictions, Mapping) or set(predictions) != set(MODEL_IDS):
        raise PredictionLedgerError("locked prediction model inventory changed")
    for model_id, prediction in predictions.items():
        if not isinstance(prediction, Mapping) or set(prediction) != {
            "p_blue",
            "p_red",
            "latent_mean",
            "latent_variance",
        }:
            raise PredictionLedgerError(f"{model_id} prediction structure changed")
        try:
            p_blue = float(prediction["p_blue"])
            p_red = float(prediction["p_red"])
            variance = float(prediction["latent_variance"])
        except (TypeError, ValueError) as exc:
            raise PredictionLedgerError("prediction contains a non-numeric value") from exc
        if (
            not 0.0 < p_blue < 1.0
            or not 0.0 < p_red < 1.0
            or not math.isclose(p_blue + p_red, 1.0, abs_tol=1e-12)
            or not math.isfinite(float(prediction["latent_mean"]))
            or not math.isfinite(variance)
            or variance < 0.0
        ):
            raise PredictionLedgerError("prediction probability or variance is invalid")
    diagnostics = value.get("event_rating_diagnostics") or {}
    players = diagnostics.get("players")
    teams = diagnostics.get("teams")
    if (
        not isinstance(players, list)
        or len(players) != 10
        or not isinstance(teams, list)
        or len(teams) != 2
        or diagnostics.get("component_allocation_is_prior_regularized") is not True
        or diagnostics.get("ratings_are_evaluation_diagnostics_only") is not True
    ):
        raise PredictionLedgerError("event rating diagnostics are malformed")
    for field in (
        "candidate_event_state_sha256",
        "player_comparator_event_state_sha256",
        "organization_comparator_event_state_sha256",
    ):
        _sha(diagnostics.get(field), f"event_rating_diagnostics.{field}")
    expected_players = [
        (team["side"], player["role"], player["player_id"], player["display_name"])
        for team in checked_roster["teams"]
        for player in team["players"]
    ]
    observed_players = [
        (
            player.get("side"),
            player.get("role"),
            player.get("player_id"),
            player.get("display_name"),
        )
        for player in players
        if isinstance(player, Mapping)
    ]
    if observed_players != expected_players:
        raise PredictionLedgerError("event player rating identities changed")
    for index, team in enumerate(teams):
        if not isinstance(team, Mapping):
            raise PredictionLedgerError("event team rating is malformed")
        roster_team = checked_roster["teams"][index]
        if (
            team.get("side") != roster_team["side"]
            or team.get("organization_id") != roster_team["organization_id"]
            or team.get("organization_name") != roster_team["organization_name"]
            or team.get("unavailable_components_are_not_zero") is not True
        ):
            raise PredictionLedgerError("event team rating identity changed")
        components = team.get("components") or {}
        for unavailable in ("lineup_synergy", "team_policy"):
            component = components.get(unavailable) or {}
            if (
                component.get("status") != "UNAVAILABLE"
                or component.get("posterior_mean_logit") is not None
                or component.get("posterior_sd_logit") is not None
            ):
                raise PredictionLedgerError(
                    "unavailable team component was treated as an estimate"
                )
    qualification = value.get("qualification") or {}
    expected_qualification = {
        "event_on_or_after_future_boundary": True,
        "prediction_strictly_before_event_start": True,
        "exact_ten_player_identity_present": True,
        "pre_event_roster_receipt_valid": True,
        "pre_event_patch_receipt_valid": True,
        "candidate_and_both_comparators_frozen": True,
        "system_clock_sampled_inside_builder": True,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
        "protocol_eligible_candidate": True,
        "independently_pinned_ledger_entry": False,
    }
    if qualification != expected_qualification:
        raise PredictionLedgerError("prediction qualification changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise PredictionLedgerError("prediction receipt exceeds authority")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise PredictionLedgerError("prediction source inventory changed")
    if [record.get("locator") for record in records if isinstance(record, Mapping)] != list(SOURCE_LOCKS):
        raise PredictionLedgerError("prediction source order changed")
    for record in records:
        locator = str(record["locator"])
        path = root / locator
        if (
            not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or _sha256_path(path) != record.get("raw_sha256")
        ):
            raise PredictionLedgerError(f"prediction source drifted: {locator}")
    return value


def replay_pre_event_prediction_receipt(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    value = validate_pre_event_prediction_receipt(payload, root=root)
    inputs = value["input_receipts"]
    rebuilt = build_pre_event_prediction_receipt(
        roster_receipt_raw=(
            json.dumps(
                inputs["roster"]["receipt"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
        patch_receipt_raw=(
            json.dumps(
                inputs["patch"]["receipt"],
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
        series_id=value["event"]["series_id"],
        game_number=value["event"]["game_number"],
        root=root,
        clock=lambda: _timestamp(value["captured_at_utc"], "captured_at_utc"),
    )
    comparable = (
        "evaluation_predictions",
        "event_rating_diagnostics",
        "event",
        "source_snapshot",
        "source_preflight",
    )
    if any(rebuilt[key] != value[key] for key in comparable):
        raise PredictionLedgerError("pre-event prediction replay changed")
    return value


def _receipt_locator(value: Any) -> PurePosixPath:
    path = PurePosixPath(_nonempty(value, "receipt_locator"))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(RECEIPT_PREFIX.parts)]) != RECEIPT_PREFIX.parts
        or path.suffix != ".json"
    ):
        raise PredictionLedgerError("prediction receipt locator is outside its root")
    return path


def build_prediction_ledger_registry(
    *,
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    created = _clock_sample(clock, "prediction ledger")
    entries = []
    for locator, payload in receipts:
        checked = validate_pre_event_prediction_receipt(payload, root=root)
        _receipt_locator(locator)
        if created < _timestamp(checked["captured_at_utc"], "captured_at_utc"):
            raise PredictionLedgerError("ledger predates a prediction receipt")
        event = checked["event"]
        entries.append(
            {
                "event_id": event["event_id"],
                "series_id": event["series_id"],
                "game_number": event["game_number"],
                "event_start_utc": event["event_start_utc"],
                "league": event["league"],
                "patch": event["patch"],
                "roster_change_stratum": event["roster_change_stratum"],
                "captured_at_utc": checked["captured_at_utc"],
                "receipt_locator": locator,
                "receipt_artifact_sha256": checked["artifact_sha256"],
            }
        )
    entries.sort(key=lambda item: (item["event_start_utc"], item["event_id"]))
    if len({entry["event_id"] for entry in entries}) != len(entries):
        raise PredictionLedgerError("prediction ledger contains duplicate events")
    series_by_league: dict[str, set[str]] = {}
    changed_series: set[str] = set()
    all_series: set[str] = set()
    for entry in entries:
        all_series.add(entry["series_id"])
        series_by_league.setdefault(entry["league"], set()).add(entry["series_id"])
        if entry["roster_change_stratum"] == "ONE_OR_BOTH_ROSTERS_CHANGED":
            changed_series.add(entry["series_id"])
    support = {
        "overall_series": len(all_series),
        "series_by_league": {
            league: len(series_by_league.get(league, set()))
            for league in adapter.DOMESTIC_LEAGUES
        },
        "one_or_both_rosters_changed_series": len(changed_series),
    }
    protocol = validate_registered_future_protocol_v3(root=root)
    rule = protocol["future_holdout"]["support_stopping_rule"]
    support_met = (
        support["overall_series"] >= rule["overall_series_minimum"]
        and all(
            support["series_by_league"][league]
            >= rule["each_domestic_league_series_minimum"]
            for league in rule["domestic_leagues"]
        )
        and support["one_or_both_rosters_changed_series"]
        >= rule["one_or_both_rosters_changed_series_minimum"]
    )
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": (
            "SUPPORT_MET_OUTCOMES_UNOPENED"
            if support_met
            else "COLLECTING_OUTCOME_FREE_PREDICTIONS"
        ),
        "created_at_utc": created.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": created.isoformat(),
            "user_supplied_timestamp_allowed": False,
            "creation_time_not_after_builder_observation": True,
        },
        "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "outcomes_present": False,
        "outcomes_accessed": False,
        "independently_pinned": False,
        "opening_authority": False,
        "entries": entries,
        "metadata_support": support,
        "support_stopping_rule": rule,
        "authority": {name: False for name in AUTHORITY_KEYS},
    }
    registry["artifact_sha256"] = _canonical_sha256(registry)
    return validate_prediction_ledger_registry(registry, root=root)


def validate_prediction_ledger_registry(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PredictionLedgerError("prediction ledger must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "prediction_ledger")
    if set(value) != {
        "schema_version",
        "status",
        "created_at_utc",
        "clock_attestation",
        "protocol_artifact_sha256",
        "outcomes_present",
        "outcomes_accessed",
        "independently_pinned",
        "opening_authority",
        "entries",
        "metadata_support",
        "support_stopping_rule",
        "authority",
        "artifact_sha256",
    }:
        raise PredictionLedgerError("prediction ledger structure changed")
    if value.get("schema_version") != REGISTRY_SCHEMA_VERSION or value.get("status") not in {
        "COLLECTING_OUTCOME_FREE_PREDICTIONS",
        "SUPPORT_MET_OUTCOMES_UNOPENED",
    }:
        raise PredictionLedgerError("prediction ledger identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise PredictionLedgerError("prediction ledger canonical hash mismatch")
    if (
        value.get("protocol_artifact_sha256") != REGISTERED_PROTOCOL_ARTIFACT_SHA256
        or value.get("outcomes_present") is not False
        or value.get("outcomes_accessed") is not False
        or value.get("independently_pinned") is not False
        or value.get("opening_authority") is not False
    ):
        raise PredictionLedgerError("prediction ledger exceeds its claim boundary")
    created_at = _timestamp(value.get("created_at_utc"), "created_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": created_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
        "creation_time_not_after_builder_observation": True,
    }:
        raise PredictionLedgerError("prediction ledger clock attestation changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise PredictionLedgerError("prediction ledger exceeds authority")
    entries = value.get("entries")
    if not isinstance(entries, list):
        raise PredictionLedgerError("prediction ledger entries are malformed")
    if entries != sorted(entries, key=lambda item: (item["event_start_utc"], item["event_id"])):
        raise PredictionLedgerError("prediction ledger entries are not ordered")
    if len({entry.get("event_id") for entry in entries}) != len(entries):
        raise PredictionLedgerError("prediction ledger contains duplicate events")
    expected_entry_keys = {
        "event_id",
        "series_id",
        "game_number",
        "event_start_utc",
        "league",
        "patch",
        "roster_change_stratum",
        "captured_at_utc",
        "receipt_locator",
        "receipt_artifact_sha256",
    }
    series_by_league: dict[str, set[str]] = {}
    changed_series: set[str] = set()
    all_series: set[str] = set()
    for entry in entries:
        if not isinstance(entry, Mapping) or set(entry) != expected_entry_keys:
            raise PredictionLedgerError("prediction ledger entry structure changed")
        event_id = _nonempty(entry.get("event_id"), "entry.event_id")
        series_id = _nonempty(entry.get("series_id"), "entry.series_id")
        event_start = _timestamp(entry.get("event_start_utc"), "entry.event_start_utc")
        captured = _timestamp(entry.get("captured_at_utc"), "entry.captured_at_utc")
        if (
            captured >= event_start
            or created_at < captured
            or event_start.astimezone(timezone.utc).replace(tzinfo=None)
            < FUTURE_SEALED_START
        ):
            raise PredictionLedgerError("prediction ledger entry is not pre-event")
        game_number = entry.get("game_number")
        if isinstance(game_number, bool) or not isinstance(game_number, int) or game_number < 1:
            raise PredictionLedgerError("prediction ledger game number is invalid")
        league = _nonempty(entry.get("league"), "entry.league")
        if league not in adapter.LEAGUES:
            raise PredictionLedgerError("prediction ledger league is unsupported")
        if not PATCH_RE.fullmatch(_nonempty(entry.get("patch"), "entry.patch")):
            raise PredictionLedgerError("prediction ledger patch is invalid")
        stratum = entry.get("roster_change_stratum")
        if stratum not in {
            "BOTH_ROSTERS_STABLE",
            "ONE_OR_BOTH_ROSTERS_CHANGED",
            "NO_PRIOR_EXACT_LINEUP",
        }:
            raise PredictionLedgerError("prediction ledger roster stratum is invalid")
        _receipt_locator(entry.get("receipt_locator"))
        _sha(entry.get("receipt_artifact_sha256"), "entry.receipt_artifact_sha256")
        all_series.add(series_id)
        series_by_league.setdefault(league, set()).add(series_id)
        if stratum == "ONE_OR_BOTH_ROSTERS_CHANGED":
            changed_series.add(series_id)
        _nonempty(event_id, "entry.event_id")
    expected_support = {
        "overall_series": len(all_series),
        "series_by_league": {
            league: len(series_by_league.get(league, set()))
            for league in adapter.DOMESTIC_LEAGUES
        },
        "one_or_both_rosters_changed_series": len(changed_series),
    }
    if value.get("metadata_support") != expected_support:
        raise PredictionLedgerError("prediction ledger metadata support does not reconcile")
    protocol = validate_registered_future_protocol_v3(root=root)
    rule = protocol["future_holdout"]["support_stopping_rule"]
    if value.get("support_stopping_rule") != rule:
        raise PredictionLedgerError("prediction ledger stopping rule changed")
    support_met = (
        expected_support["overall_series"] >= rule["overall_series_minimum"]
        and all(
            expected_support["series_by_league"][league]
            >= rule["each_domestic_league_series_minimum"]
            for league in rule["domestic_leagues"]
        )
        and expected_support["one_or_both_rosters_changed_series"]
        >= rule["one_or_both_rosters_changed_series_minimum"]
    )
    expected_status = (
        "SUPPORT_MET_OUTCOMES_UNOPENED"
        if support_met
        else "COLLECTING_OUTCOME_FREE_PREDICTIONS"
    )
    if value.get("status") != expected_status:
        raise PredictionLedgerError("prediction ledger support status is fabricated")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = _sha256_bytes(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"refusing to replace prediction artifact: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster-receipt", type=Path, required=True)
    parser.add_argument("--patch-receipt", type=Path, required=True)
    parser.add_argument("--series-id", required=True)
    parser.add_argument("--game-number", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = build_pre_event_prediction_receipt(
        roster_receipt_raw=args.roster_receipt.read_bytes(),
        patch_receipt_raw=args.patch_receipt.read_bytes(),
        series_id=args.series_id,
        game_number=args.game_number,
    )
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "event_id": payload["event"]["event_id"],
                "model_ids": list(payload["evaluation_predictions"]),
                "probability_authority": False,
                "betting_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_KEYS",
    "DEFAULT_REGISTRY",
    "MODEL_IDS",
    "PredictionLedgerError",
    "RECEIPT_SCHEMA_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "RESULT_STATE",
    "build_pre_event_prediction_receipt",
    "build_prediction_ledger_registry",
    "replay_pre_event_prediction_receipt",
    "validate_pre_event_prediction_receipt",
    "validate_prediction_ledger_registry",
    "write_no_clobber",
]
