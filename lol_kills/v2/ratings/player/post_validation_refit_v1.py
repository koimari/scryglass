"""Fresh post-validation player/team rating refit for one pre-event roster.

The prospective phase-one ledger must stay frozen for an honest model test, but
that frozen state is not a suitable long-lived deployment state.  This module
replays the already locked model family on an immutable, newly captured source
snapshot, using only series available strictly before the target event.  It
emits rating components, never a match probability or betting decision.

The builder is intentionally unusable until the joint phase-one result has
been independently registered as a pass.  Its output remains non-authorizing
until separate semantic deployment and event registries approve the exact
bytes.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import base64
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
import threading
from typing import Any, Callable, Iterator, Mapping, Sequence

import pandas as pd

from lol_kills import pregame_roster_capture as roster_capture
from lol_kills.v2.market import phase_one_recalibration_v1 as recalibration
from lol_kills.v2.market import phase_one_evaluation_registry_v1 as phase_registry

from . import multileague_development as adapter
from . import multileague_runner as rating
from . import multileague_v2_runner as hierarchical
from . import multileague_v3_prediction_ledger as prediction_ledger
from .multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v3,
)


ROOT = Path(__file__).resolve().parents[4]
SOURCE_LOCATOR = "lol_kills/v2/ratings/player/post_validation_refit_v1.py"
SOURCE_SNAPSHOT_SCHEMA_VERSION = (
    "scryglass:post-validation-rating-source-snapshot:v1"
)
SCHEMA_VERSION = "scryglass:post-validation-event-rating-refit:v1"
RESULT_STATE = "FRESH_PRE_EVENT_RATING_REFIT_NON_AUTHORIZING"
SNAPSHOT_PREFIX = PurePosixPath(
    "data/lol/v2/snapshots/rating-deployment"
)
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/rating-deployment/refits"
)
ROLES = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
MAXIMUM_DATA_AGE_SECONDS = 14 * 24 * 60 * 60
AVAILABILITY_EMBARGO_HOURS = rating.AVAILABILITY_EMBARGO_HOURS
INTERVAL_CRITICAL_VALUE = 1.96
AUTHORITY = {
    "player_rating_authority": False,
    "team_rating_authority": False,
    "match_probability_authority": False,
    "odds_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
DECISION_OUTPUTS = {
    "match_probability": None,
    "fair_odds": None,
    "expected_value": None,
    "bet_recommendation": None,
    "stake": None,
}
CLAIM_CEILING = (
    "Fresh outcome-free pre-event player/team rating-component candidate only. "
    "Independent phase-one registration, deployment review, semantic authority, "
    "event roster/rating registries, probability calibration, uncertainty, quote, "
    "market authority, and transaction authority remain separate requirements."
)


class PostValidationRefitError(RuntimeError):
    """Fresh source, registered pass, replay, or rating output failed closed."""


_CUTOFF_LOCK = threading.RLock()


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
        raise PostValidationRefitError("rating refit is not canonical JSON") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise PostValidationRefitError(f"{field} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PostValidationRefitError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PostValidationRefitError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], field: str) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PostValidationRefitError(f"{field} clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _finite(value: Any, field: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PostValidationRefitError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise PostValidationRefitError(f"{field} is outside its numeric contract")
    return result


def _snapshot_locator(value: str, *, suffix: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(SNAPSHOT_PREFIX.parts)])
        != SNAPSHOT_PREFIX.parts
        or path.suffix != suffix
    ):
        raise PostValidationRefitError(
            "rating source locator is outside the immutable snapshot root"
        )
    return path.as_posix()


def _output_locator(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(OUTPUT_PREFIX.parts)]) != OUTPUT_PREFIX.parts
        or path.suffix != ".json"
    ):
        raise PostValidationRefitError(
            "rating refit locator is outside the immutable output root"
        )
    return path.as_posix()


def _regular_bytes(root: Path, locator: str, label: str) -> bytes:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PostValidationRefitError(f"{label} is not an unaliased regular file")
    return path.read_bytes()


def _strict_object(raw: bytes, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise PostValidationRefitError(f"duplicate JSON key in {label}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(ValueError(item)),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PostValidationRefitError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise PostValidationRefitError(f"{label} must contain an object")
    return value


def build_source_snapshot_manifest_v1(
    *,
    snapshot_id: str,
    maps_locator: str,
    players_locator: str,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Bind already persisted immutable Parquet bytes at a system-clock sample."""

    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise PostValidationRefitError("snapshot_id must be nonempty")
    maps = _snapshot_locator(maps_locator, suffix=".parquet")
    players = _snapshot_locator(players_locator, suffix=".parquet")
    if PurePosixPath(maps).parent != PurePosixPath(players).parent:
        raise PostValidationRefitError("rating source files must share one snapshot directory")
    captured = _clock_sample(clock, "source snapshot")
    payload: dict[str, Any] = {
        "schema_version": SOURCE_SNAPSHOT_SCHEMA_VERSION,
        "result_state": "IMMUTABLE_RATING_SOURCE_BYTES_CAPTURED_NON_AUTHORIZING",
        "snapshot_id": snapshot_id.strip(),
        "captured_at_utc": captured.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_after_source_files_persisted",
            "observed_wall_clock_utc": captured.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "files": {
            "maps": {
                "locator": maps,
                "raw_sha256": _sha256_bytes(_regular_bytes(root, maps, "maps source")),
            },
            "players": {
                "locator": players,
                "raw_sha256": _sha256_bytes(
                    _regular_bytes(root, players, "players source")
                ),
            },
        },
        "target_event_outcome_accessed": False,
        "authority": dict(AUTHORITY),
        "claim_ceiling": (
            "Exact source-byte identity only; model fit, rating, probability, and "
            "betting authority are absent."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_source_snapshot_manifest_v1(payload, root=root)


def validate_source_snapshot_manifest_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PostValidationRefitError("source snapshot must be an object")
    value = dict(payload)
    expected_keys = {
        "schema_version",
        "result_state",
        "snapshot_id",
        "captured_at_utc",
        "clock_attestation",
        "files",
        "target_event_outcome_accessed",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected_keys:
        raise PostValidationRefitError("source snapshot structure changed")
    declared = _sha(value.get("artifact_sha256"), "artifact_sha256")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if declared != _canonical_sha256(unsigned):
        raise PostValidationRefitError("source snapshot artifact hash changed")
    if (
        value.get("schema_version") != SOURCE_SNAPSHOT_SCHEMA_VERSION
        or value.get("result_state")
        != "IMMUTABLE_RATING_SOURCE_BYTES_CAPTURED_NON_AUTHORIZING"
        or value.get("target_event_outcome_accessed") is not False
        or value.get("authority") != AUTHORITY
    ):
        raise PostValidationRefitError("source snapshot claim boundary changed")
    captured = _timestamp(value.get("captured_at_utc"), "captured_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_after_source_files_persisted",
        "observed_wall_clock_utc": captured.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise PostValidationRefitError("source snapshot clock attestation changed")
    if not isinstance(value.get("snapshot_id"), str) or not value["snapshot_id"]:
        raise PostValidationRefitError("source snapshot identity is missing")
    files = value.get("files")
    if not isinstance(files, Mapping) or set(files) != {"maps", "players"}:
        raise PostValidationRefitError("source snapshot file inventory changed")
    parents: set[PurePosixPath] = set()
    for name in ("maps", "players"):
        record = files[name]
        if not isinstance(record, Mapping) or set(record) != {"locator", "raw_sha256"}:
            raise PostValidationRefitError("source snapshot file binding changed")
        locator = _snapshot_locator(str(record.get("locator")), suffix=".parquet")
        parents.add(PurePosixPath(locator).parent)
        raw = _regular_bytes(root, locator, f"{name} source")
        if _sha256_bytes(raw) != _sha(record.get("raw_sha256"), f"{name} raw hash"):
            raise PostValidationRefitError(f"{name} source bytes changed")
    if len(parents) != 1:
        raise PostValidationRefitError("source snapshot files are not colocated")
    return value


@contextmanager
def _dynamic_cutoff(cutoff: datetime) -> Iterator[None]:
    """Run legacy loader/replay under one exclusive, explicit target cutoff."""

    if cutoff.tzinfo is not None:
        raise PostValidationRefitError("rating replay cutoff must be timezone-naive")
    with _CUTOFF_LOCK:
        previous = adapter.SEALED_FINAL_START
        adapter.SEALED_FINAL_START = pd.Timestamp(cutoff)
        try:
            yield
        finally:
            adapter.SEALED_FINAL_START = previous


def _load_input_candidate(
    *,
    source: Mapping[str, Any],
    event_start: datetime,
    root: Path,
) -> tuple[Any, hierarchical.CandidateSpec]:
    if event_start.tzinfo is None:
        raise PostValidationRefitError("rating event start must include a timezone")
    protocol = validate_registered_future_protocol_v3(root=root)
    cutoff = event_start.astimezone(timezone.utc).replace(tzinfo=None)
    files = source["files"]
    with _dynamic_cutoff(cutoff):
        try:
            input_data = adapter.load_multileague_development_input(
                expected_maps_sha256=files["maps"]["raw_sha256"],
                expected_players_sha256=files["players"]["raw_sha256"],
                root=root,
                maps_locator=files["maps"]["locator"],
                players_locator=files["players"]["locator"],
            )
            candidate = hierarchical.CandidateSpec.from_payload(
                protocol["locked_candidate"]["definition"]
            )
        except Exception as exc:
            raise PostValidationRefitError("fresh rating source load failed") from exc
    return input_data, candidate


def _registered_pass(
    *, result_locator: str, root: Path, environment: Mapping[str, str]
) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    try:
        registered, result, raw = recalibration._registered_pass(
            result_locator=result_locator,
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise PostValidationRefitError(
            "independently registered phase-one pass is unavailable"
        ) from exc
    if result.get("phase_one_models_passed") is not True:
        raise PostValidationRefitError("phase-one models did not pass")
    return registered, raw, result


def _interval(mean: float, variance: float) -> list[float]:
    if variance < 0.0 or not math.isfinite(variance):
        raise PostValidationRefitError("rating variance is invalid")
    radius = INTERVAL_CRITICAL_VALUE * math.sqrt(variance)
    return [mean - radius, mean + radius]


def _data_age_seconds(cutoff: datetime, latest_available: datetime) -> float:
    if cutoff.tzinfo is not None or latest_available.tzinfo is not None:
        raise PostValidationRefitError("rating source times must be timezone-naive")
    age = (cutoff - latest_available).total_seconds()
    if not 0.0 <= age <= MAXIMUM_DATA_AGE_SECONDS:
        raise PostValidationRefitError("fresh rating state exceeds its data-age ceiling")
    return age


def _component(
    state: hierarchical.HierarchicalGaussianState,
    weights: Mapping[str, float],
) -> dict[str, Any]:
    mean, variance = state.moments(weights)
    return {
        "status": "ESTIMATED_OR_PRIOR_MIX",
        "posterior_mean_logit": mean,
        "posterior_sd_logit": math.sqrt(variance),
        "posterior_interval_95_logit": _interval(mean, variance),
    }


def _unavailable(reason: str) -> dict[str, Any]:
    return {
        "status": "UNAVAILABLE",
        "posterior_mean_logit": None,
        "posterior_sd_logit": None,
        "posterior_interval_95_logit": None,
        "reason": reason,
    }


def _derive_ratings(
    *,
    source: Mapping[str, Any],
    roster: Mapping[str, Any],
    patch: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    event_start = _timestamp(roster["event_start"], "roster.event_start")
    cutoff = event_start.replace(tzinfo=None)
    input_data, candidate = _load_input_candidate(
        source=source,
        event_start=event_start,
        root=root,
    )
    with _dynamic_cutoff(cutoff):
        try:
            replay = hierarchical.replay_candidate(input_data, candidate)
        except Exception as exc:
            raise PostValidationRefitError("fresh rating refit failed") from exc

    source_captured = _timestamp(source["captured_at_utc"], "source.captured_at")
    captured_naive = source_captured.replace(tzinfo=None)
    if any(
        adapter.source_local_datetime(series.source_local_end) > captured_naive
        for series in input_data.development_series
    ):
        raise PostValidationRefitError(
            "source contains a pre-event outcome timestamp after source capture"
        )
    available_series = [
        series
        for series in input_data.development_series
        if adapter.source_local_datetime(series.source_local_end)
        + timedelta(hours=AVAILABILITY_EMBARGO_HOURS)
        < cutoff
    ]
    if not available_series:
        raise PostValidationRefitError("no rating series was available before the event")
    latest_available = max(
        adapter.source_local_datetime(series.source_local_end)
        for series in available_series
    )
    data_age_seconds = _data_age_seconds(cutoff, latest_available)

    blue = prediction_ledger._lineup(roster["teams"][0])
    red = prediction_ledger._lineup(roster["teams"][1])
    lineups = (blue, red)
    player_ids = [slot.player_id for lineup in lineups for slot in lineup.players]
    team_ids = [lineup.team_id for lineup in lineups]
    replay.state.transition_entities(player_ids, team_ids, cutoff)
    roster_statuses: dict[str, str] = {}
    retention: dict[str, float | None] = {}
    for lineup in lineups:
        previous = prediction_ledger._historical_lineup_identity(
            replay.team_lineups.get(lineup.team_id)
        )
        current = prediction_ledger._lineup_identity(lineup)
        roster_statuses[lineup.side] = prediction_ledger._roster_status(
            previous, current
        )
        retention[lineup.side] = replay.state.apply_roster_transition(
            lineup.team_id, previous, current
        )

    event_league = str(roster["league"])
    probe = prediction_ledger._ForecastMap(event_league, blue, red)
    feature = hierarchical._feature_vector(
        replay.state,
        probe,
        replay.team_home_leagues,
    )
    if event_league in adapter.INTERNATIONAL_LEAGUES and feature.bridge_status not in {
        "INTERNATIONAL_BOTH_HOME_LEAGUES_KNOWN",
        "INTERNATIONAL_SAME_HOME_LEAGUE",
    }:
        raise PostValidationRefitError(
            "international team rating lacks exact home-league identities"
        )

    teams: list[dict[str, Any]] = []
    team_weights: list[dict[str, float]] = []
    for lineup, home_league in zip(
        lineups, (feature.blue_home_league, feature.red_home_league)
    ):
        player_weights = {
            rating._player_key(slot.player_id): candidate.player_weight_per_role
            for slot in lineup.players
        }
        organization_weights = {
            hierarchical._organization_key(lineup.team_id): candidate.organization_weight
        }
        league_weights: dict[str, float] = {}
        if event_league in adapter.INTERNATIONAL_LEAGUES:
            if home_league is None:
                raise PostValidationRefitError("team home league is unavailable")
            league_weights[rating._league_key(home_league)] = 1.0
        joint_weights = dict(player_weights) | dict(organization_weights) | dict(
            league_weights
        )
        team_weights.append(joint_weights)
        players: list[dict[str, Any]] = []
        roster_team = roster["teams"][SIDES.index(lineup.side)]
        for slot, roster_player in zip(lineup.players, roster_team["players"]):
            key = rating._player_key(slot.player_id)
            mean, variance = replay.state.moments({key: 1.0})
            players.append(
                {
                    "role": slot.role,
                    "player_id": slot.player_id,
                    "display_name": roster_player["display_name"],
                    "status": (
                        "ESTIMATED"
                        if replay.state.evidence_counts.get(key, 0)
                        else "PRIOR_ONLY"
                    ),
                    "posterior_mean_logit": mean,
                    "posterior_sd_logit": math.sqrt(variance),
                    "posterior_interval_95_logit": _interval(mean, variance),
                    "outcome_evidence_updates": int(
                        replay.state.evidence_counts.get(key, 0)
                    ),
                }
            )
        joint = _component(replay.state, joint_weights)
        joint_mean = float(joint["posterior_mean_logit"])
        joint_sd = float(joint["posterior_sd_logit"])
        teams.append(
            {
                "side": lineup.side,
                "organization_id": lineup.team_id,
                "organization_name": lineup.team_name,
                "roster_id": roster_team["roster_id"],
                "roster_status": roster_statuses[lineup.side],
                "organization_retention_phi": retention[lineup.side],
                "home_league": home_league,
                "players": players,
                "components": {
                    "player_aggregate": _component(replay.state, player_weights),
                    "organization_residual": _component(
                        replay.state, organization_weights
                    ),
                    "league_adjustment": (
                        _component(replay.state, league_weights)
                        if league_weights
                        else {
                            "status": "REFERENCE_ZERO_WITHIN_LEAGUE",
                            "posterior_mean_logit": 0.0,
                            "posterior_sd_logit": 0.0,
                            "posterior_interval_95_logit": [0.0, 0.0],
                        }
                    ),
                    "lineup_synergy": _unavailable(
                        "lineup synergy is not separately identified"
                    ),
                    "team_policy": _unavailable(
                        "team policy is not separately identified"
                    ),
                },
                "joint_identified_strength": {
                    **joint,
                    "display_rating_mean": rating.DISPLAY_ANCHOR
                    + rating.DISPLAY_LOGIT_SCALE * joint_mean,
                    "display_rating_sd": rating.DISPLAY_LOGIT_SCALE * joint_sd,
                    "estimand": (
                        "player aggregate plus roster-retained organization residual"
                        + (
                            " plus home-league adjustment"
                            if league_weights
                            else " within-league reference"
                        )
                    ),
                },
                "unavailable_components_are_not_zero": True,
            }
        )

    difference_weights = dict(team_weights[0])
    for key, weight in team_weights[1].items():
        difference_weights[key] = difference_weights.get(key, 0.0) - weight
    difference_weights = {
        key: weight
        for key, weight in difference_weights.items()
        if abs(weight) > 1e-15
    }
    difference = _component(replay.state, difference_weights)
    expected_difference = (
        teams[0]["joint_identified_strength"]["posterior_mean_logit"]
        - teams[1]["joint_identified_strength"]["posterior_mean_logit"]
    )
    if not math.isclose(
        float(difference["posterior_mean_logit"]),
        float(expected_difference),
        rel_tol=0.0,
        abs_tol=1e-10,
    ):
        raise PostValidationRefitError("team-strength difference does not reconcile")

    return {
        "data_cutoff_source_time_naive": latest_available.isoformat(),
        "data_age_seconds_at_event": data_age_seconds,
        "maximum_data_age_seconds": MAXIMUM_DATA_AGE_SECONDS,
        "availability_embargo_hours": AVAILABILITY_EMBARGO_HOURS,
        "applied_series": replay.applied_series,
        "applied_maps": replay.applied_maps,
        "posterior_state_sha256": prediction_ledger._state_sha256(replay),
        "bridge_status": feature.bridge_status,
        "teams": teams,
        "strength_difference_blue_minus_red": {
            **difference,
            "orientation": "blue_minus_red",
            "cross_team_covariance_retained": True,
            "blue_side_effect_included": False,
        },
        "unavailable_components_are_not_zero": True,
        "target_event_outcome_present": False,
        "target_event_outcome_accessed": False,
        "patch": patch["patch"],
    }


def build_post_validation_refit_v1(
    *,
    phase_one_result_locator: str,
    source_snapshot_locator: str,
    roster_receipt_raw: bytes,
    patch_receipt_raw: bytes,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    """Refit the locked family on fresh pre-event data after a registered pass."""

    built_at = _clock_sample(clock, "rating refit")
    source_locator = _snapshot_locator(source_snapshot_locator, suffix=".json")
    source_raw = _regular_bytes(root, source_locator, "source snapshot manifest")
    source = validate_source_snapshot_manifest_v1(
        _strict_object(source_raw, "source snapshot manifest"), root=root
    )
    try:
        roster = roster_capture.validate_pregame_roster_receipt(
            _strict_object(roster_receipt_raw, "roster receipt")
        )
        patch = prediction_ledger._validate_patch_receipt(
            _strict_object(patch_receipt_raw, "patch receipt")
        )
    except Exception as exc:
        raise PostValidationRefitError("pre-event roster or patch receipt is invalid") from exc
    event_start = _timestamp(roster["event_start"], "event_start")
    if patch["fixture_id"] != roster["event_id"]:
        raise PostValidationRefitError("patch and roster event identities differ")
    if _timestamp(patch["event_start"], "patch.event_start") != event_start:
        raise PostValidationRefitError("patch and roster event starts differ")
    if not (
        _timestamp(source["captured_at_utc"], "source.captured_at")
        <= built_at
        < event_start
    ):
        raise PostValidationRefitError("rating refit is not pre-event and post-source")
    if built_at < _timestamp(roster["captured_at"], "roster.captured_at"):
        raise PostValidationRefitError("rating refit predates roster evidence")
    if built_at < _timestamp(patch["as_of"], "patch.as_of"):
        raise PostValidationRefitError("rating refit predates patch evidence")
    phase_registry_value, phase_raw, phase_result = _registered_pass(
        result_locator=phase_one_result_locator,
        root=root,
        environment=environment,
    )
    phase_registry_receipt = phase_registry_value.get("receipt") or {}
    registered_at = _timestamp(
        phase_registry_receipt.get("registered_at_utc"),
        "phase-one registry registered_at_utc",
    )
    if built_at < registered_at:
        raise PostValidationRefitError("rating refit predates phase-one registration")
    ratings = _derive_ratings(
        source=source,
        roster=roster,
        patch=patch,
        root=root,
    )
    protocol = validate_registered_future_protocol_v3(root=root)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "built_at_utc": built_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_refit_builder",
            "observed_wall_clock_utc": built_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "phase_one_pass": {
            "result_locator": phase_one_result_locator,
            "result_raw_sha256": _sha256_bytes(phase_raw),
            "result_artifact_sha256": phase_result["artifact_sha256"],
            "registry_locator": phase_registry.REGISTRY_LOCATOR.as_posix(),
            "registry_raw_sha256": phase_registry_value["receipt_raw_sha256"],
            "registry_id": phase_registry_receipt["registry_id"],
            "registered_at_utc": registered_at.isoformat(),
            "phase_one_models_passed": True,
            "independent_registration_required_and_replayed": True,
        },
        "source_snapshot": {
            "locator": source_locator,
            "raw_sha256": _sha256_bytes(source_raw),
            "artifact_sha256": source["artifact_sha256"],
            "captured_at_utc": source["captured_at_utc"],
        },
        "event": {
            "event_id": roster["event_id"],
            "event_start_utc": event_start.isoformat(),
            "league": roster["league"],
            "patch": patch["patch"],
            "blue_organization_id": roster["teams"][0]["organization_id"],
            "red_organization_id": roster["teams"][1]["organization_id"],
        },
        "input_receipts": {
            "roster_raw_base64": base64.b64encode(roster_receipt_raw).decode("ascii"),
            "roster_raw_sha256": _sha256_bytes(roster_receipt_raw),
            "roster_canonical_sha256": roster["receipt_sha256"],
            "patch_raw_base64": base64.b64encode(patch_receipt_raw).decode("ascii"),
            "patch_raw_sha256": _sha256_bytes(patch_receipt_raw),
        },
        "model_family": {
            "protocol_locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "protocol_raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "selected_candidate_id": protocol["locked_candidate"]["candidate_id"],
            "hyperparameters_changed_after_phase_one": False,
            "single_process_exclusive_dynamic_cutoff": True,
        },
        "ratings": ratings,
        "decision_outputs": dict(DECISION_OUTPUTS),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_post_validation_refit_v1(
        payload,
        root=root,
        environment=environment,
    )


def _decode(raw_base64: Any, expected_sha256: Any, label: str) -> bytes:
    if not isinstance(raw_base64, str):
        raise PostValidationRefitError(f"{label} base64 is missing")
    try:
        raw = base64.b64decode(raw_base64, validate=True)
    except (ValueError, TypeError) as exc:
        raise PostValidationRefitError(f"{label} base64 is invalid") from exc
    if _sha256_bytes(raw) != _sha(expected_sha256, f"{label} raw hash"):
        raise PostValidationRefitError(f"{label} raw bytes changed")
    return raw


def validate_post_validation_refit_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PostValidationRefitError("rating refit must be an object")
    value = dict(payload)
    expected_keys = {
        "schema_version",
        "result_state",
        "built_at_utc",
        "clock_attestation",
        "phase_one_pass",
        "source_snapshot",
        "event",
        "input_receipts",
        "model_family",
        "ratings",
        "decision_outputs",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected_keys:
        raise PostValidationRefitError("rating refit structure changed")
    declared = _sha(value.get("artifact_sha256"), "artifact_sha256")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if declared != _canonical_sha256(unsigned):
        raise PostValidationRefitError("rating refit artifact hash changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
        or value.get("decision_outputs") != DECISION_OUTPUTS
        or value.get("authority") != AUTHORITY
        or value.get("claim_ceiling") != CLAIM_CEILING
    ):
        raise PostValidationRefitError("rating refit claim boundary changed")
    built_at = _timestamp(value.get("built_at_utc"), "built_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_refit_builder",
        "observed_wall_clock_utc": built_at.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise PostValidationRefitError("rating refit clock attestation changed")

    phase = value.get("phase_one_pass")
    if not isinstance(phase, Mapping):
        raise PostValidationRefitError("phase-one pass binding is missing")
    phase_registry_value, phase_raw, phase_result = _registered_pass(
        result_locator=str(phase.get("result_locator")),
        root=root,
        environment=environment,
    )
    phase_registry_receipt = phase_registry_value.get("receipt") or {}
    registered_at = _timestamp(
        phase_registry_receipt.get("registered_at_utc"),
        "phase-one registry registered_at_utc",
    )
    if phase != {
        "result_locator": phase.get("result_locator"),
        "result_raw_sha256": _sha256_bytes(phase_raw),
        "result_artifact_sha256": phase_result["artifact_sha256"],
        "registry_locator": phase_registry.REGISTRY_LOCATOR.as_posix(),
        "registry_raw_sha256": phase_registry_value["receipt_raw_sha256"],
        "registry_id": phase_registry_receipt["registry_id"],
        "registered_at_utc": registered_at.isoformat(),
        "phase_one_models_passed": True,
        "independent_registration_required_and_replayed": True,
    }:
        raise PostValidationRefitError("phase-one pass binding changed")
    if built_at < registered_at:
        raise PostValidationRefitError("rating refit predates phase-one registration")

    source_binding = value.get("source_snapshot")
    if not isinstance(source_binding, Mapping):
        raise PostValidationRefitError("source snapshot binding is missing")
    source_locator = _snapshot_locator(
        str(source_binding.get("locator")), suffix=".json"
    )
    source_raw = _regular_bytes(root, source_locator, "source snapshot manifest")
    if _sha256_bytes(source_raw) != _sha(
        source_binding.get("raw_sha256"), "source snapshot raw hash"
    ):
        raise PostValidationRefitError("source snapshot raw bytes changed")
    source = validate_source_snapshot_manifest_v1(
        _strict_object(source_raw, "source snapshot manifest"), root=root
    )
    if source_binding != {
        "locator": source_locator,
        "raw_sha256": _sha256_bytes(source_raw),
        "artifact_sha256": source["artifact_sha256"],
        "captured_at_utc": source["captured_at_utc"],
    }:
        raise PostValidationRefitError("source snapshot binding changed")

    receipts = value.get("input_receipts")
    if not isinstance(receipts, Mapping):
        raise PostValidationRefitError("input receipt binding is missing")
    roster_raw = _decode(
        receipts.get("roster_raw_base64"),
        receipts.get("roster_raw_sha256"),
        "roster receipt",
    )
    patch_raw = _decode(
        receipts.get("patch_raw_base64"),
        receipts.get("patch_raw_sha256"),
        "patch receipt",
    )
    try:
        roster = roster_capture.validate_pregame_roster_receipt(
            _strict_object(roster_raw, "roster receipt")
        )
        patch = prediction_ledger._validate_patch_receipt(
            _strict_object(patch_raw, "patch receipt")
        )
    except Exception as exc:
        raise PostValidationRefitError("embedded roster or patch is invalid") from exc
    if receipts.get("roster_canonical_sha256") != roster["receipt_sha256"]:
        raise PostValidationRefitError("roster canonical binding changed")
    event_start = _timestamp(roster["event_start"], "event_start")
    if not (
        _timestamp(source["captured_at_utc"], "source.captured_at")
        <= built_at
        < event_start
    ):
        raise PostValidationRefitError("rating refit timing changed")
    expected_event = {
        "event_id": roster["event_id"],
        "event_start_utc": event_start.isoformat(),
        "league": roster["league"],
        "patch": patch["patch"],
        "blue_organization_id": roster["teams"][0]["organization_id"],
        "red_organization_id": roster["teams"][1]["organization_id"],
    }
    if value.get("event") != expected_event:
        raise PostValidationRefitError("event binding changed")
    protocol = validate_registered_future_protocol_v3(root=root)
    expected_model = {
        "protocol_locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
        "protocol_raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "protocol_artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "selected_candidate_id": protocol["locked_candidate"]["candidate_id"],
        "hyperparameters_changed_after_phase_one": False,
        "single_process_exclusive_dynamic_cutoff": True,
    }
    if value.get("model_family") != expected_model:
        raise PostValidationRefitError("model-family binding changed")
    expected_ratings = _derive_ratings(
        source=source,
        roster=roster,
        patch=patch,
        root=root,
    )
    if value.get("ratings") != expected_ratings:
        raise PostValidationRefitError("fresh player/team ratings do not replay")
    return value


def load_post_validation_refit_v1(
    locator_value: str,
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> tuple[str, bytes, dict[str, Any]]:
    """Load and fully replay one immutable post-validation refit artifact."""

    locator = _output_locator(locator_value)
    raw = _regular_bytes(root, locator, "post-validation rating refit")
    value = validate_post_validation_refit_v1(
        _strict_object(raw, "post-validation rating refit"),
        root=root,
        environment=environment,
    )
    return locator, raw, value


def prepare_probability_replay_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Prepare the exact fresh source and roster used by probability draws."""

    value = validate_post_validation_refit_v1(
        payload,
        root=root,
        environment=environment,
    )
    source_binding = value["source_snapshot"]
    source_raw = _regular_bytes(
        root, source_binding["locator"], "source snapshot manifest"
    )
    source = validate_source_snapshot_manifest_v1(
        _strict_object(source_raw, "source snapshot manifest"), root=root
    )
    receipts = value["input_receipts"]
    roster_raw = _decode(
        receipts["roster_raw_base64"],
        receipts["roster_raw_sha256"],
        "roster receipt",
    )
    patch_raw = _decode(
        receipts["patch_raw_base64"],
        receipts["patch_raw_sha256"],
        "patch receipt",
    )
    try:
        roster = roster_capture.validate_pregame_roster_receipt(
            _strict_object(roster_raw, "roster receipt")
        )
        patch = prediction_ledger._validate_patch_receipt(
            _strict_object(patch_raw, "patch receipt")
        )
    except Exception as exc:
        raise PostValidationRefitError(
            "fresh probability replay receipts are invalid"
        ) from exc
    event_start = _timestamp(roster["event_start"], "roster.event_start")
    input_data, candidate = _load_input_candidate(
        source=source,
        event_start=event_start,
        root=root,
    )
    return {
        "refit": value,
        "source": source,
        "source_raw": source_raw,
        "roster": roster,
        "roster_raw": roster_raw,
        "patch": patch,
        "patch_raw": patch_raw,
        "event_start": event_start,
        "event_cutoff_naive": event_start.replace(tzinfo=None),
        "input_data": input_data,
        "candidate": candidate,
    }


def _probability_replay_v1(
    prepared: Mapping[str, Any], sampled_indices: Sequence[int]
) -> tuple[float, Any, Any]:
    input_data = prepared["input_data"]
    original = input_data.development_series
    if len(sampled_indices) != len(original) or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= len(original)
        for index in sampled_indices
    ):
        raise PostValidationRefitError("rating replay sample inventory is invalid")
    sampled = [original[index] for index in sampled_indices]
    sampled.sort(key=lambda item: (item.source_local_start, item.series_id))
    bootstrap_input = replace(input_data, development_series=tuple(sampled))
    cutoff = prepared["event_cutoff_naive"]
    with _dynamic_cutoff(cutoff):
        try:
            replay = hierarchical.replay_candidate(
                bootstrap_input, prepared["candidate"]
            )
        except Exception as exc:
            raise PostValidationRefitError("fresh rating probability replay failed") from exc

    roster = prepared["roster"]
    blue = prediction_ledger._lineup(roster["teams"][0])
    red = prediction_ledger._lineup(roster["teams"][1])
    lineups = (blue, red)
    player_ids = [slot.player_id for lineup in lineups for slot in lineup.players]
    team_ids = [lineup.team_id for lineup in lineups]
    replay.state.transition_entities(player_ids, team_ids, cutoff)
    for lineup in lineups:
        previous = prediction_ledger._historical_lineup_identity(
            replay.team_lineups.get(lineup.team_id)
        )
        current = prediction_ledger._lineup_identity(lineup)
        replay.state.apply_roster_transition(lineup.team_id, previous, current)
    forecast = prediction_ledger._ForecastMap(roster["league"], blue, red)
    feature = hierarchical._feature_vector(
        replay.state, forecast, replay.team_home_leagues
    )
    if (
        roster["league"] in adapter.INTERNATIONAL_LEAGUES
        and feature.bridge_status
        not in {
            "INTERNATIONAL_BOTH_HOME_LEAGUES_KNOWN",
            "INTERNATIONAL_SAME_HOME_LEAGUE",
        }
    ):
        raise PostValidationRefitError(
            "fresh rating probability lacks exact international bridge identity"
        )
    probability, _mean, _variance = replay.state.predict(feature.weights)
    if not 0.0 < probability < 1.0:
        raise PostValidationRefitError("fresh rating probability is invalid")
    return float(probability), replay, feature


def point_rating_probability_v1(prepared: Mapping[str, Any]) -> float:
    """Replay the unresampled fresh point estimate and its exact state identity."""

    population = len(prepared["input_data"].development_series)
    probability, replay, feature = _probability_replay_v1(
        prepared, list(range(population))
    )
    expected = prepared["refit"]["ratings"]
    if (
        replay.applied_series != expected["applied_series"]
        or replay.applied_maps != expected["applied_maps"]
        or prediction_ledger._state_sha256(replay)
        != expected["posterior_state_sha256"]
        or feature.bridge_status != expected["bridge_status"]
    ):
        raise PostValidationRefitError(
            "fresh rating probability state differs from rating components"
        )
    return probability


def sampled_rating_probability_v1(
    prepared: Mapping[str, Any], sampled_indices: Sequence[int]
) -> float:
    """Replay one cluster-bootstrap rating probability from the fresh bytes."""

    probability, _replay, _feature = _probability_replay_v1(
        prepared, sampled_indices
    )
    return probability


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PostValidationRefitError(f"refusing to replace rating artifact: {path}")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PostValidationRefitError(
                f"refusing to replace rating artifact: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256_bytes(raw)


__all__ = [
    "AVAILABILITY_EMBARGO_HOURS",
    "AUTHORITY",
    "CLAIM_CEILING",
    "DECISION_OUTPUTS",
    "MAXIMUM_DATA_AGE_SECONDS",
    "OUTPUT_PREFIX",
    "PostValidationRefitError",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SNAPSHOT_PREFIX",
    "SOURCE_SNAPSHOT_SCHEMA_VERSION",
    "build_post_validation_refit_v1",
    "build_source_snapshot_manifest_v1",
    "load_post_validation_refit_v1",
    "point_rating_probability_v1",
    "prepare_probability_replay_v1",
    "sampled_rating_probability_v1",
    "validate_post_validation_refit_v1",
    "validate_source_snapshot_manifest_v1",
    "write_no_clobber",
]
