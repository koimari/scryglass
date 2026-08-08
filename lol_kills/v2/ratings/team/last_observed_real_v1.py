"""Pinned, private last-observed exact-five LPL Team Rating table.

The source rows identify *observed map participants*.  They do not identify
an official, active, or current roster.  Each receipt is labelled ``last
observed roster as of <that receipt's source observation date>``; the frozen
timestamp is only the table boundary.  The module computes a player-only
ordered-five aggregation on the accepted L4 scale, and fail-closes every
source/identity/freshness ambiguity.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping, Sequence

from lol_kills.v2.ratings.player.private_development_runner import (
    DISPLAY_ANCHOR,
    LEGACY_DISPLAY_SCALE_V2,
    verify_legacy_private_development_artifact_v2,
)

# The checked-in table is a frozen v2 descriptive artifact.  Keep its
# historical conversion exact while the current Player path uses v3.
DISPLAY_SCALE = LEGACY_DISPLAY_SCALE_V2
from lol_kills.v2.ratings.player.real_v1_adapter import (
    ACCEPTED_G1_PINS,
    MapObservation,
    PrivatePlayerRatingInput,
    load_accepted_lpl_private_player_rating_input,
)


ROOT = Path(__file__).resolve().parents[4]
PLAYER_ARTIFACT_PATH = ROOT / "data/lol/v2/models/player/real-v1/private-development-artifact.json"
PLAYER_ARTIFACT_SHA256 = "35e8831fb4d39fd60ec7f8f59b934ff5571f788ec8dc1151c78661b67ab6d4fd"
LAST_OBSERVED_ARTIFACT_PATH = ROOT / "data/lol/v2/models/team/real-v1/last-observed-exact-five-team-table.json"
SCHEMA_VERSION = "scryglass:team-real-v1-last-observed-exact-five:v1"
ROLES = ("top", "jungle", "mid", "bot", "support")

# The manifest's sealed boundary is safer than the timestamp of its final
# accepted map: every selected observation must be strictly earlier.  The
# 21-day MVP ceiling is predeclared as observed-gap p90 (14 days) plus one
# weekly-cadence cushion.  It is a non-validated descriptive policy, never
# evidence that a roster is current, active, or official.
FROZEN_AS_OF_SOURCE_LOCAL = "2026-06-01T00:00:00"
FRESHNESS_CEILING_DAYS = 21
CLAIM_CEILING = {
    "private_last_observed_player_only_table": True,
    "current_roster": False,
    "official_roster": False,
    "active_roster": False,
    "forecast": False,
    "prediction": False,
    "production": False,
    "publication": False,
    "promotion": False,
    "sota": False,
    "final_holdout": False,
}


class LastObservedTeamError(ValueError):
    """The private receipt/table cannot be reproduced safely."""


class LastObservedTeamUnavailable(LastObservedTeamError):
    """A team has no eligible last-observed exact-five receipt."""


def _canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise LastObservedTeamError("non-canonical or non-finite private artifact") from error


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _parse_local(value: Any, *, label: str) -> datetime:
    if not isinstance(value, str) or value.endswith("Z"):
        raise LastObservedTeamError(f"{label} must be source-local naive ISO-8601")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise LastObservedTeamError(f"{label} is not ISO-8601") from error
    if parsed.tzinfo is not None:
        raise LastObservedTeamError(f"{label} must be source-local naive ISO-8601")
    return parsed


def _finite(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise LastObservedTeamError(f"{label} must be finite")
    return float(value)


def _load_pinned_player_artifact(path: Path = PLAYER_ARTIFACT_PATH) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LastObservedTeamUnavailable("PLAYER_ARTIFACT_UNAVAILABLE") from error
    # This descriptive table is byte-pinned to the frozen v2 Player artifact.
    # The current v3 development output is a separate, scale-correct artifact;
    # do not rewrite historical evidence to make the two paths look identical.
    artifact = verify_legacy_private_development_artifact_v2(raw, expected_artifact_sha256=PLAYER_ARTIFACT_SHA256)
    if artifact.get("decision") != {
        "development_winner_candidate_id": "static_baseline",
        "external_validation_gate_passed": True,
        "selected_candidate_id": "static_baseline",
    }:
        raise LastObservedTeamUnavailable("PLAYER_STATIC_BASELINE_UNAVAILABLE")
    posterior = artifact.get("development_winner_posterior_ratings")
    if not isinstance(posterior, Mapping) or posterior.get("candidate_id") != "static_baseline" or posterior.get("validation_gate_passed") is not True:
        raise LastObservedTeamUnavailable("PLAYER_STATIC_POSTERIOR_UNAVAILABLE")
    return artifact


def _load_pinned_g1_input() -> PrivatePlayerRatingInput:
    value = load_accepted_lpl_private_player_rating_input()
    actual = (value.manifest_sha256, value.rows_sha256, value.selected_target_sha256, value.split_payload_sha256)
    expected = (
        ACCEPTED_G1_PINS.manifest_sha256,
        ACCEPTED_G1_PINS.rows_sha256,
        ACCEPTED_G1_PINS.selected_target_sha256,
        ACCEPTED_G1_PINS.split_payload_sha256,
    )
    if actual != expected:
        raise LastObservedTeamUnavailable("G1_SOURCE_PIN_MISMATCH")
    if tuple(fold.fold_id for fold in value.folds) != ("TRAIN", "DEVELOPMENT", "VALIDATION"):
        raise LastObservedTeamUnavailable("G1_NONFINAL_ACCEPTED_FOLD_MISMATCH")
    return value


def _maps_by_id(input_data: PrivatePlayerRatingInput) -> dict[str, MapObservation]:
    result: dict[str, MapObservation] = {}
    for fold in input_data.folds:
        for observation in fold.map_observations:
            if observation.fold_id not in {"TRAIN", "DEVELOPMENT", "VALIDATION"} or observation.source_game_id in result:
                raise LastObservedTeamUnavailable("G1_NONFINAL_ACCEPTED_MAP_IDENTITY_MISMATCH")
            result[observation.source_game_id] = observation
    return result


def _posterior_by_player(player_artifact: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    posterior = player_artifact["development_winner_posterior_ratings"]
    result: dict[str, Mapping[str, Any]] = {}
    for item in posterior.get("ratings", []):
        if not isinstance(item, Mapping) or not isinstance(item.get("player_id"), str) or item["player_id"] in result:
            raise LastObservedTeamUnavailable("PLAYER_POSTERIOR_IDENTITY_MISMATCH")
        _finite(item.get("posterior_mean"), "player posterior mean")
        if _finite(item.get("posterior_uncertainty"), "player posterior uncertainty") < 0.0:
            raise LastObservedTeamUnavailable("PLAYER_POSTERIOR_UNCERTAINTY_INVALID")
        result[item["player_id"]] = item
    if not result:
        raise LastObservedTeamUnavailable("PLAYER_POSTERIOR_UNAVAILABLE")
    return result


def _lineups_for_map(observation: MapObservation) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    """Return only exact team fives, and a typed invalid reason per other team."""

    grouped: dict[str, list[Any]] = defaultdict(list)
    teams_by_player: dict[str, set[str]] = defaultdict(set)
    for player in observation.player_observations:
        grouped[player.source_team_id].append(player)
        teams_by_player[player.source_player_id].add(player.source_team_id)
    colliding_players = {
        player_id for player_id, team_ids in teams_by_player.items() if len(team_ids) > 1
    }
    valid: dict[str, tuple[str, ...]] = {}
    invalid: dict[str, str] = {}
    for team_id, members in grouped.items():
        if any(member.source_player_id in colliding_players for member in members):
            invalid[team_id] = "CROSS_TEAM_PLAYER_IDENTITY_COLLISION"
            continue
        if len(members) != 5:
            invalid[team_id] = "MISSING_OR_DUPLICATE_ROLES"
            continue
        roles = tuple(member.role for member in members)
        player_ids = tuple(member.source_player_id for member in members)
        if set(roles) != set(ROLES) or len(set(roles)) != 5 or len(set(player_ids)) != 5:
            invalid[team_id] = "MISSING_OR_DUPLICATE_ROLES"
            continue
        role_map = {member.role: member.source_player_id for member in members}
        valid[team_id] = tuple(role_map[role] for role in ROLES)
    return valid, invalid


def _source_material(input_data: PrivatePlayerRatingInput, player_artifact: Mapping[str, Any]) -> dict[str, Any]:
    posterior = player_artifact["development_winner_posterior_ratings"]
    return {
        "g1_manifest_sha256": input_data.manifest_sha256,
        "g1_rows_sha256": input_data.rows_sha256,
        "g1_selected_target_sha256": input_data.selected_target_sha256,
        "g1_split_payload_sha256": input_data.split_payload_sha256,
        "g2_player_artifact_sha256": PLAYER_ARTIFACT_SHA256,
        "g2_candidate_id": "static_baseline",
        "g2_as_of_source_game_id": posterior["as_of_source_game_id"],
        "g2_ordered_origin_sha256": posterior["ordered_origin_sha256"],
    }


def _freshness_diagnostic(maps: Mapping[str, MapObservation]) -> dict[str, Any]:
    """Replay the predeclared descriptive ceiling from pinned calendar dates."""

    dates_by_team: dict[str, set[str]] = defaultdict(set)
    for observation in maps.values():
        _parse_local(observation.source_local_event_start, label="observed map time")
        for player in observation.player_observations:
            dates_by_team[player.source_team_id].add(observation.source_local_event_start[:10])
    gaps: list[int] = []
    for dates in dates_by_team.values():
        ordered = sorted(dates)
        gaps.extend((datetime.fromisoformat(right) - datetime.fromisoformat(left)).days for left, right in zip(ordered, ordered[1:]))
    gaps.sort()
    if not gaps:
        raise LastObservedTeamUnavailable("FRESHNESS_DIAGNOSTIC_UNAVAILABLE")
    index = math.ceil(0.90 * len(gaps)) - 1
    diagnostic = {
        "source_time_semantics": "source_local_naive_calendar_dates",
        "population": "all pinned G1 non-final accepted team calendar-date appearances before receipt filtering",
        "interappearance_gap_n": len(gaps),
        "quantile": 0.90,
        "quantile_convention": "nearest_rank_ceiling_zero_based_index",
        "quantile_index": index,
        "p90_days": gaps[index],
        "cushion_days": 7,
        "ceiling_days": FRESHNESS_CEILING_DAYS,
        "rated_count_not_used": True,
    }
    if diagnostic["p90_days"] != 14 or diagnostic["p90_days"] + diagnostic["cushion_days"] != FRESHNESS_CEILING_DAYS:
        raise LastObservedTeamUnavailable("FRESHNESS_DECLARATION_REPLAY_MISMATCH")
    return diagnostic


def build_last_observed_exact_five_receipts() -> dict[str, Any]:
    """Derive source-stable observed fives from the hard-pinned G1/G2 inputs."""

    input_data = _load_pinned_g1_input()
    player = _load_pinned_player_artifact()
    posterior = player["development_winner_posterior_ratings"]
    maps = _maps_by_id(input_data)
    freshness_diagnostic = _freshness_diagnostic(maps)
    as_of_map = maps.get(posterior.get("as_of_source_game_id"))
    if as_of_map is None or _parse_local(as_of_map.source_local_event_start, label="player artifact as-of map") >= _parse_local(FROZEN_AS_OF_SOURCE_LOCAL, label="frozen artifact as-of"):
        raise LastObservedTeamUnavailable("FROZEN_AS_OF_SOURCE_IDENTITY_MISMATCH")
    frozen = _parse_local(FROZEN_AS_OF_SOURCE_LOCAL, label="frozen artifact as-of")
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_latest: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in maps.values():
        observed_at = _parse_local(observation.source_local_event_start, label="observed map time")
        if observed_at >= frozen:
            raise LastObservedTeamUnavailable("G1_MAP_AFTER_FROZEN_AS_OF")
        valid, invalid = _lineups_for_map(observation)
        for team_id, player_ids in valid.items():
            candidates[team_id].append({
                "team_id": team_id,
                "source_game_id": observation.source_game_id,
                "source_series_id": observation.source_series_id,
                "source_partition": observation.fold_id,
                "source_local_event_start": observation.source_local_event_start,
                "player_ids_by_role": list(player_ids),
            })
        for team_id, reason in invalid.items():
            invalid_latest[team_id].append({
                "team_id": team_id,
                "source_game_id": observation.source_game_id,
                "source_series_id": observation.source_series_id,
                "source_partition": observation.fold_id,
                "source_local_event_start": observation.source_local_event_start,
                "reason": reason,
            })
    receipts, withheld = [], []
    all_team_ids = sorted(set(candidates) | set(invalid_latest))
    for team_id in all_team_ids:
        options = candidates.get(team_id, [])
        invalid = invalid_latest.get(team_id, [])
        latest_time = max(
            [_parse_local(item["source_local_event_start"], label="observed map time") for item in options + invalid],
            default=None,
        )
        if latest_time is None:
            withheld.append({"team_id": team_id, "reason": "NO_OBSERVED_LINEUP"})
            continue
        latest_invalid = [item for item in invalid if _parse_local(item["source_local_event_start"], label="observed map time") == latest_time]
        if latest_invalid:
            withheld.append({"team_id": team_id, "reason": latest_invalid[0]["reason"], "source_game_ids": sorted(item["source_game_id"] for item in latest_invalid), "source_partitions": sorted(set(item["source_partition"] for item in latest_invalid))})
            continue
        latest = [item for item in options if _parse_local(item["source_local_event_start"], label="observed map time") == latest_time]
        lineups = {tuple(item["player_ids_by_role"]) for item in latest}
        if len(lineups) != 1:
            withheld.append({"team_id": team_id, "reason": "CONFLICTING_SAME_TIME_LAST_OBSERVATION", "source_game_ids": sorted(item["source_game_id"] for item in latest), "source_partitions": sorted(set(item["source_partition"] for item in latest))})
            continue
        # If equal-time duplicate entries agree, retain both map/series IDs in
        # the receipt rather than using an arbitrary tie-breaker.
        prototype = min(latest, key=lambda item: (item["source_series_id"], item["source_game_id"]))
        age_seconds = (frozen - latest_time).total_seconds()
        if age_seconds < 0:
            withheld.append({"team_id": team_id, "reason": "OBSERVATION_AFTER_FROZEN_AS_OF"})
            continue
        receipt = {
            "receipt_kind": "last_observed_exact_five",
            "label": f"last observed roster as of {prototype['source_local_event_start'][:10]}",
            "table_label": f"last observed roster table at frozen boundary {FROZEN_AS_OF_SOURCE_LOCAL[:10]}",
            "team_id": team_id,
            "league_id": "LPL",
            "source_local_event_start": prototype["source_local_event_start"],
            "source_game_ids": sorted(item["source_game_id"] for item in latest),
            "source_series_ids": sorted(set(item["source_series_id"] for item in latest)),
            "source_partitions": sorted(set(item["source_partition"] for item in latest)),
            "player_ids_by_role": [{"role": role, "player_id": player_id} for role, player_id in zip(ROLES, prototype["player_ids_by_role"])],
            "frozen_as_of_source_local": FROZEN_AS_OF_SOURCE_LOCAL,
            "age_seconds_at_frozen_as_of": age_seconds,
            "freshness_ceiling_days": FRESHNESS_CEILING_DAYS,
            "source_pins": _source_material(input_data, player),
        }
        receipt["receipt_sha256"] = _sha256(receipt)
        receipts.append(receipt)
    value = {
        "schema_version": "scryglass:last-observed-exact-five-receipts:v1",
        "frozen_as_of_source_local": FROZEN_AS_OF_SOURCE_LOCAL,
        "freshness_ceiling_days": FRESHNESS_CEILING_DAYS,
        "freshness_diagnostic": freshness_diagnostic,
        "source_pins": _source_material(input_data, player),
        "receipts": sorted(receipts, key=lambda item: item["team_id"]),
        "withheld": sorted(withheld, key=lambda item: item["team_id"]),
    }
    value["receipt_set_sha256"] = _sha256(value)
    return value


def load_last_observed_exact_five_receipts(value: Mapping[str, Any]) -> dict[str, Any]:
    """Verify a receipt set against independent G1/G2 pins, never its caller."""

    input_data = _load_pinned_g1_input()
    player = _load_pinned_player_artifact()
    unsigned = dict(value)
    claimed = unsigned.pop("receipt_set_sha256", None)
    if claimed != _sha256(unsigned):
        raise LastObservedTeamUnavailable("RECEIPT_SET_DIGEST_MISMATCH")
    if value.get("schema_version") != "scryglass:last-observed-exact-five-receipts:v1":
        raise LastObservedTeamUnavailable("RECEIPT_SET_SCHEMA_MISMATCH")
    if value.get("frozen_as_of_source_local") != FROZEN_AS_OF_SOURCE_LOCAL or value.get("freshness_ceiling_days") != FRESHNESS_CEILING_DAYS:
        raise LastObservedTeamUnavailable("RECEIPT_SET_FROZEN_BOUNDARY_MISMATCH")
    expected_source = _source_material(input_data, player)
    if value.get("source_pins") != expected_source:
        raise LastObservedTeamUnavailable("RECEIPT_SET_SOURCE_PIN_MISMATCH")
    # A caller's self-rehash is not authority.  Reconstruct the complete
    # receipt set from the independently pinned rows and Player artifact and
    # require byte-identical canonical content.
    expected = build_last_observed_exact_five_receipts()
    if _canonical_bytes(value) != _canonical_bytes(expected):
        raise LastObservedTeamUnavailable("RECEIPT_SET_REPLAY_MISMATCH")
    receipts = value.get("receipts")
    if not isinstance(receipts, list) or [item.get("team_id") for item in receipts] != sorted(item.get("team_id") for item in receipts):
        raise LastObservedTeamUnavailable("RECEIPT_SET_ORDER_MISMATCH")
    seen = set()
    frozen = _parse_local(FROZEN_AS_OF_SOURCE_LOCAL, label="frozen artifact as-of")
    for receipt in receipts:
        if not isinstance(receipt, Mapping):
            raise LastObservedTeamUnavailable("RECEIPT_INVALID")
        raw = dict(receipt)
        receipt_sha = raw.pop("receipt_sha256", None)
        if receipt_sha != _sha256(raw) or raw.get("source_pins") != expected_source:
            raise LastObservedTeamUnavailable("RECEIPT_DIGEST_OR_SOURCE_PIN_MISMATCH")
        team_id = raw.get("team_id")
        players = raw.get("player_ids_by_role")
        if not isinstance(team_id, str) or team_id in seen or raw.get("league_id") != "LPL":
            raise LastObservedTeamUnavailable("RECEIPT_TEAM_IDENTITY_MISMATCH")
        seen.add(team_id)
        if not isinstance(players, list) or [(item.get("role"), item.get("player_id")) for item in players] != [(role, item.get("player_id")) for role, item in zip(ROLES, players)] or len({item.get("player_id") for item in players}) != 5:
            raise LastObservedTeamUnavailable("RECEIPT_ROLE_IDENTITY_MISMATCH")
        observed = _parse_local(raw.get("source_local_event_start"), label="receipt observed time")
        expected_age = (frozen - observed).total_seconds()
        if raw.get("age_seconds_at_frozen_as_of") != expected_age or expected_age < 0:
            raise LastObservedTeamUnavailable("RECEIPT_AGE_IDENTITY_MISMATCH")
    return dict(value)


def _aggregate_receipt(receipt: Mapping[str, Any], ratings: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    players = receipt["player_ids_by_role"]
    player_ratings = []
    for item in players:
        posterior = ratings.get(item["player_id"])
        if posterior is None:
            raise LastObservedTeamUnavailable("UNKNOWN_PLAYER_IDENTITY")
        player_ratings.append((item["role"], item["player_id"], _finite(posterior["posterior_mean"], "posterior mean"), _finite(posterior["posterior_uncertainty"], "posterior uncertainty")))
    covariance = [[uncertainty ** 2 if i == j else 0.0 for j, (_, _, _, uncertainty) in enumerate(player_ratings)] for i, (_, _, _, uncertainty) in enumerate(player_ratings)]
    latent_covariance = [[entry / (DISPLAY_SCALE ** 2) for entry in row] for row in covariance]
    weights = [0.2] * 5
    latent_mean = sum(weight * ((mean - DISPLAY_ANCHOR) / DISPLAY_SCALE) for weight, (_, _, mean, _) in zip(weights, player_ratings))
    latent_variance = sum(weights[i] * latent_covariance[i][j] * weights[j] for i in range(5) for j in range(5))
    display_variance = DISPLAY_SCALE ** 2 * latent_variance
    if display_variance < 0.0 or not math.isfinite(display_variance):
        raise LastObservedTeamError("TEAM_COVARIANCE_NOT_PSD")
    return {
        "team_posterior_latent_mean": latent_mean,
        "team_posterior_latent_variance": latent_variance,
        "team_posterior_display_mean": DISPLAY_ANCHOR + DISPLAY_SCALE * latent_mean,
        "team_posterior_display_variance": display_variance,
        "team_posterior_display_uncertainty": math.sqrt(display_variance),
        "player_display_covariance": covariance,
        "player_latent_covariance": latent_covariance,
        "covariance_assumption": {
            "kind": "DIAGONAL_ASSUMED_DENSITY_REPRESENTATION",
            "joint_covariance_status": "unavailable",
            "off_diagonal": "zero_by_l4_diagonal_model_assumption_not_new_estimate",
            "full_joint_covariance": None,
        },
        "components": {
            "player_aggregation": {"status": "available", "kind": "ordered_five_x_transpose_mu_weights_0_2"},
            "lineup_synergy": {"status": "unavailable", "value": None, "blocker": "not identified"},
            "policy": {"status": "unavailable", "value": None, "blocker": "not identified"},
            "league_rating": {"status": "unavailable", "value": None, "blocker": "not identified for regional LPL table"},
        },
    }


def build_last_observed_lpl_team_table() -> dict[str, Any]:
    """Build the first nonempty fresh, dated LPL player-only table."""

    receipt_set = load_last_observed_exact_five_receipts(build_last_observed_exact_five_receipts())
    player = _load_pinned_player_artifact()
    ratings = _posterior_by_player(player)
    table, withheld = [], list(receipt_set["withheld"])
    for receipt in receipt_set["receipts"]:
        if receipt["age_seconds_at_frozen_as_of"] > FRESHNESS_CEILING_DAYS * 86400:
            withheld.append({"team_id": receipt["team_id"], "reason": "STALE_LAST_OBSERVED_RECEIPT", "last_observed_source_local_date": receipt["source_local_event_start"][:10], "source_partitions": receipt["source_partitions"], "age_seconds_at_frozen_as_of": receipt["age_seconds_at_frozen_as_of"]})
            continue
        try:
            rating = _aggregate_receipt(receipt, ratings)
        except LastObservedTeamUnavailable as error:
            withheld.append({"team_id": receipt["team_id"], "reason": str(error)})
            continue
        table.append({
            "status": "private_development_only",
            "table_label": f"last observed roster table at frozen boundary {FROZEN_AS_OF_SOURCE_LOCAL[:10]}",
            "last_observed_source_local_date": receipt["source_local_event_start"][:10],
            "last_observed_label": f"last observed roster as of {receipt['source_local_event_start'][:10]}",
            "team_id": receipt["team_id"],
            "last_observed_exact_five_receipt": receipt,
            "rating": rating,
        })
    counts = Counter(item["reason"] for item in withheld)
    rated_partition_memberships = Counter(partition for item in table for partition in item["last_observed_exact_five_receipt"]["source_partitions"])
    withheld_partition_memberships = Counter(partition for item in withheld for partition in item.get("source_partitions", []))
    artifact: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": "PRIVATE_LAST_OBSERVED_DESCRIPTIVE_TABLE",
        "label": f"last observed roster table at frozen boundary {FROZEN_AS_OF_SOURCE_LOCAL[:10]}",
        "frozen_as_of_source_local": FROZEN_AS_OF_SOURCE_LOCAL,
        "freshness_policy": {
            "ceiling_days": FRESHNESS_CEILING_DAYS,
            "rationale": "predeclared 21-day MVP ceiling: observed inter-appearance-gap p90 of 14 days plus a 7-day weekly-cadence cushion; non-validated operational freshness policy that never establishes a current, active, or official roster",
            "runtime_clock_used": False,
            "supporting_diagnostic": receipt_set["freshness_diagnostic"],
        },
        "private_scope": {"authorizes": ["private_last_observed_player_only_table"], "blocked": ["current_roster", "official_roster", "active_roster", "forecast", "prediction", "production", "publication", "promotion", "sota", "final_holdout", "team_rank"]},
        "claim_ceiling": dict(CLAIM_CEILING),
        "receipt_set": receipt_set,
        "rated_team_order": "team_id_lexicographic_not_rating_rank",
        "rated_teams": table,
        "withheld_teams": sorted(withheld, key=lambda item: (item["team_id"], item["reason"])),
        "counts": {"teams_observed": len(receipt_set["receipts"]) + len(receipt_set["withheld"]), "rated": len(table), "withheld": len(withheld), "withheld_by_reason": dict(sorted(counts.items())), "rated_receipt_partition_memberships": dict(sorted(rated_partition_memberships.items())), "withheld_receipt_partition_memberships": dict(sorted(withheld_partition_memberships.items()))},
        "predictive_comparison": {"status": "unavailable", "blocker": "last-observed descriptive aggregation is not a separately identified team predictive model"},
    }
    artifact["artifact_sha256"] = _sha256(artifact)
    return artifact


def verify_last_observed_lpl_team_table(artifact: Mapping[str, Any], *, expected_artifact_sha256: str) -> dict[str, Any]:
    raw = dict(artifact)
    claimed = raw.pop("artifact_sha256", None)
    if claimed != _sha256(raw):
        raise LastObservedTeamError("TABLE_ARTIFACT_DIGEST_MISMATCH")
    if claimed != expected_artifact_sha256:
        raise LastObservedTeamError("TABLE_ARTIFACT_EXTERNAL_PIN_MISMATCH")
    if artifact.get("schema_version") != SCHEMA_VERSION or artifact.get("result_state") != "PRIVATE_LAST_OBSERVED_DESCRIPTIVE_TABLE":
        raise LastObservedTeamError("TABLE_ARTIFACT_SCHEMA_OR_STATE_MISMATCH")
    if artifact.get("claim_ceiling") != CLAIM_CEILING or artifact.get("predictive_comparison", {}).get("status") != "unavailable":
        raise LastObservedTeamError("TABLE_ARTIFACT_CLAIM_BOUNDARY_MISMATCH")
    load_last_observed_exact_five_receipts(artifact.get("receipt_set", {}))
    return dict(artifact)


def _safe_atomic_write(path: Path, payload: bytes) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    parent_parts = absolute.parts[1:-1]
    for logical_anchor in (ROOT, Path(tempfile.gettempdir())):
        try:
            relative = absolute.relative_to(logical_anchor.absolute())
        except ValueError:
            continue
        current, parent_parts = logical_anchor.resolve(), relative.parts[:-1]
        break
    for part in parent_parts:
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise LastObservedTeamError("TABLE_OUTPUT_PARENT_MISSING") from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise LastObservedTeamError("TABLE_OUTPUT_PARENT_UNSAFE")
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        metadata = None
    if metadata is not None and (stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1):
        raise LastObservedTeamError("TABLE_OUTPUT_UNSAFE")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_last_observed_lpl_team_table(path: Path = LAST_OBSERVED_ARTIFACT_PATH) -> str:
    artifact = build_last_observed_lpl_team_table()
    _safe_atomic_write(path, _canonical_bytes(artifact) + b"\n")
    return artifact["artifact_sha256"]
