"""Private event-bound Player/Team Rating registration.

Rating outputs are candidates until a separate reviewer registry is pinned by
digest outside the repository.  Registration is deliberately event-specific:
the exact ordered pre-event roster, model/evaluation evidence, freshness, and
uncertainty contract must all replay before a rating can be consumed.

Even an approved rating registration is only a rating-component authority.  It
does not authorize a match-win probability, fair odds, EV, or a betting action.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from lol_kills.v2.ratings import semantic_rating_authority_v1 as semantic_authority
from lol_kills.v2.ratings.player import (
    multileague_v3_prediction_ledger as ratings_ledger,
)


ROOT = Path(__file__).resolve().parents[1]
RECEIPT_SCHEMA_VERSION = "scryglass.private-event-rating.v3"
REGISTRY_SCHEMA_VERSION = "scryglass.private-event-rating-registry.v3"
REGISTRY_SCOPE = "private_personal_decision_support"
RECEIPT_PREFIX = PurePosixPath(
    "data/lol/private_rating_authority/receipts"
)
ARTIFACT_PREFIXES = (
    PurePosixPath("data/lol/v2/models"),
    PurePosixPath("data/lol/private_rating_authority/evidence"),
)
ROLES = ("top", "jungle", "mid", "bot", "support")
SIDES = ("blue", "red")
ARTIFACT_KINDS = (
    "source_snapshot",
    "player_model",
    "team_model",
    "evaluation",
    "reliability",
    "uncertainty",
)
TEAM_COMPONENTS = (
    "player_aggregate",
    "organization_residual",
    "lineup_synergy",
    "team_policy",
)
REVIEW_GATES = (
    "source_provenance",
    "pre_event_temporal_replay",
    "identity_and_role_resolution",
    "out_of_sample_predictive_value",
    "posterior_interval_coverage",
    "series_cluster_dependence",
    "roster_change_holdout",
    "patch_and_tournament_holdout",
    "replay_reproducibility",
)
AUTHORIZED_USES = (
    "private_player_rating",
    "private_team_rating",
    "private_live_state_diagnostic_input",
)
AGGREGATION_METHOD = "exact_roster_joint_identified_component_posterior_v2"
UNCERTAINTY_METHOD = "joint_posterior_with_series_clustered_validation"
NORMAL_95_Z = 1.959963984540054
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MAX_RATING_AGE_SECONDS = 365 * 24 * 60 * 60


class RatingAuthorityError(ValueError):
    """A rating receipt or registry violates its frozen contract."""


class RegisteredEventRatingUnavailable(RatingAuthorityError):
    """No independently registered event rating is usable."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def canonical_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RatingAuthorityError("value is not canonical finite JSON") from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RatingAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_json_bytes(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RatingAuthorityError(
                    f"non-finite JSON number in {label}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RatingAuthorityError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise RatingAuthorityError(f"{label} must be a JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise RatingAuthorityError(
            f"{label} keys do not match the frozen contract"
        )


def _nonempty(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RatingAuthorityError(f"{label} must be a non-empty string")
    return value.strip()


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise RatingAuthorityError(f"{label} must be a lowercase SHA-256")
    return value


def _timestamp(value: Any, label: str) -> datetime:
    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RatingAuthorityError(f"{label} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise RatingAuthorityError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _finite(value: Any, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RatingAuthorityError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise RatingAuthorityError(f"{label} is outside the frozen numeric contract")
    return result


def _interval(value: Any, label: str, mean: float) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RatingAuthorityError(f"{label} must contain lower and upper bounds")
    low = _finite(value[0], f"{label}.lower")
    high = _finite(value[1], f"{label}.upper")
    if low > mean or mean > high or low >= high:
        raise RatingAuthorityError(f"{label} must contain its posterior mean")
    return [low, high]


def _allowed_artifact_locator(value: Any, label: str) -> PurePosixPath:
    path = PurePosixPath(_nonempty(value, label))
    allowed = any(
        tuple(path.parts[: len(prefix.parts)]) == prefix.parts
        for prefix in ARTIFACT_PREFIXES
    ) or path.as_posix() == ratings_ledger.SOURCE_LOCATOR
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or not allowed
    ):
        raise RatingAuthorityError(
            f"{label} is outside the allowed private rating evidence roots"
        )
    return path


def _receipt_locator(value: Any) -> PurePosixPath:
    path = PurePosixPath(_nonempty(value, "receipt_locator"))
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or tuple(path.parts[: len(RECEIPT_PREFIX.parts)]) != RECEIPT_PREFIX.parts
        or path.suffix != ".json"
    ):
        raise RatingAuthorityError(
            "receipt_locator is outside the private rating receipt root"
        )
    return path


def _prediction_receipt_reference(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise RatingAuthorityError("prediction_receipt must be a mapping")
    _exact_keys(
        value,
        {"locator", "raw_sha256", "artifact_sha256"},
        "prediction_receipt",
    )
    locator = PurePosixPath(
        _nonempty(value.get("locator"), "prediction_receipt.locator")
    )
    prefix = ratings_ledger.RECEIPT_PREFIX
    if (
        locator.is_absolute()
        or any(part in {"", ".", ".."} for part in locator.parts)
        or tuple(locator.parts[: len(prefix.parts)]) != prefix.parts
        or locator.suffix != ".json"
    ):
        raise RatingAuthorityError(
            "prediction_receipt.locator is outside the frozen v3 prediction root"
        )
    return {
        "locator": locator.as_posix(),
        "raw_sha256": _sha(
            value.get("raw_sha256"), "prediction_receipt.raw_sha256"
        ),
        "artifact_sha256": _sha(
            value.get("artifact_sha256"),
            "prediction_receipt.artifact_sha256",
        ),
    }


def _normalize_artifacts(value: Any) -> dict[str, dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != set(ARTIFACT_KINDS):
        raise RatingAuthorityError(
            "artifacts must bind every frozen rating evidence kind"
        )
    normalized: dict[str, dict[str, str]] = {}
    for kind in ARTIFACT_KINDS:
        reference = value[kind]
        if not isinstance(reference, Mapping):
            raise RatingAuthorityError(f"artifacts.{kind} must be a mapping")
        _exact_keys(reference, {"locator", "raw_sha256"}, f"artifacts.{kind}")
        locator = str(
            _allowed_artifact_locator(
                reference.get("locator"), f"artifacts.{kind}.locator"
            )
        )
        normalized[kind] = {
            "locator": locator,
            "raw_sha256": _sha(
                reference.get("raw_sha256"),
                f"artifacts.{kind}.raw_sha256",
            ),
        }
    return normalized


def _normalize_component(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RatingAuthorityError(f"{label} must be a mapping")
    _exact_keys(
        value,
        {"status", "posterior_mean", "posterior_sd", "unavailable_reason"},
        label,
    )
    status = value.get("status")
    if status == "ESTIMATED":
        if value.get("unavailable_reason") is not None:
            raise RatingAuthorityError(
                f"{label}.unavailable_reason must be null when estimated"
            )
        return {
            "status": "ESTIMATED",
            "posterior_mean": _finite(
                value.get("posterior_mean"), f"{label}.posterior_mean"
            ),
            "posterior_sd": _finite(
                value.get("posterior_sd"), f"{label}.posterior_sd", minimum=0.0
            ),
            "unavailable_reason": None,
        }
    if status == "UNAVAILABLE":
        if value.get("posterior_mean") is not None or value.get("posterior_sd") is not None:
            raise RatingAuthorityError(
                f"{label} unavailable values must remain null rather than zero"
            )
        return {
            "status": "UNAVAILABLE",
            "posterior_mean": None,
            "posterior_sd": None,
            "unavailable_reason": _nonempty(
                value.get("unavailable_reason"), f"{label}.unavailable_reason"
            ),
        }
    raise RatingAuthorityError(f"{label}.status must be ESTIMATED or UNAVAILABLE")


def _normalize_teams(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != len(SIDES):
        raise RatingAuthorityError("teams must contain exact blue and red ratings")
    normalized: list[dict[str, Any]] = []
    all_players: set[str] = set()
    for index, side in enumerate(SIDES):
        team = value[index]
        if not isinstance(team, Mapping):
            raise RatingAuthorityError(f"teams.{side} must be a mapping")
        _exact_keys(
            team,
            {
                "side",
                "organization_id",
                "organization_name",
                "roster_id",
                "players",
                "components",
                "estimand",
                "posterior_mean",
                "posterior_interval_95",
            },
            f"teams.{side}",
        )
        if team.get("side") != side:
            raise RatingAuthorityError("teams must be ordered blue then red")
        players = team.get("players")
        if not isinstance(players, list) or len(players) != len(ROLES):
            raise RatingAuthorityError(f"teams.{side} requires exactly five players")
        normalized_players: list[dict[str, Any]] = []
        player_means: list[float] = []
        for role_index, role in enumerate(ROLES):
            player = players[role_index]
            if not isinstance(player, Mapping):
                raise RatingAuthorityError(
                    f"teams.{side}.players.{role} must be a mapping"
                )
            _exact_keys(
                player,
                {
                    "role",
                    "player_id",
                    "display_name",
                    "posterior_mean",
                    "posterior_sd",
                },
                f"teams.{side}.players.{role}",
            )
            if player.get("role") != role:
                raise RatingAuthorityError(
                    f"teams.{side}.players must be ordered {ROLES}"
                )
            player_id = _nonempty(
                player.get("player_id"),
                f"teams.{side}.players.{role}.player_id",
            )
            if player_id in all_players:
                raise RatingAuthorityError("event ratings cannot repeat a player")
            all_players.add(player_id)
            posterior_mean = _finite(
                player.get("posterior_mean"),
                f"teams.{side}.players.{role}.posterior_mean",
            )
            player_means.append(posterior_mean)
            normalized_players.append(
                {
                    "role": role,
                    "player_id": player_id,
                    "display_name": _nonempty(
                        player.get("display_name"),
                        f"teams.{side}.players.{role}.display_name",
                    ),
                    "posterior_mean": posterior_mean,
                    "posterior_sd": _finite(
                        player.get("posterior_sd"),
                        f"teams.{side}.players.{role}.posterior_sd",
                        minimum=0.0,
                    ),
                }
            )
        components = team.get("components")
        if not isinstance(components, Mapping) or set(components) != set(TEAM_COMPONENTS):
            raise RatingAuthorityError(
                f"teams.{side}.components must declare every frozen component"
            )
        normalized_components = {
            component: _normalize_component(
                components[component], f"teams.{side}.components.{component}"
            )
            for component in TEAM_COMPONENTS
        }
        player_aggregate = sum(player_means) / len(player_means)
        player_component = normalized_components["player_aggregate"]
        if player_component["status"] != "ESTIMATED":
            raise RatingAuthorityError(
                f"teams.{side}.components.player_aggregate must be estimated"
            )
        if not math.isclose(
            float(player_component["posterior_mean"]),
            player_aggregate,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RatingAuthorityError(
                f"teams.{side}.components.player_aggregate does not replay the exact roster"
            )
        included_components = [
            component
            for component in TEAM_COMPONENTS
            if normalized_components[component]["status"] == "ESTIMATED"
        ]
        unavailable_components = [
            component
            for component in TEAM_COMPONENTS
            if normalized_components[component]["status"] == "UNAVAILABLE"
        ]
        expected_scope = "exact_roster_player_plus_organization_identified_components"
        estimand = team.get("estimand")
        if not isinstance(estimand, Mapping):
            raise RatingAuthorityError(f"teams.{side}.estimand must be a mapping")
        _exact_keys(
            estimand,
            {"scope", "included_components", "unavailable_components"},
            f"teams.{side}.estimand",
        )
        if (
            estimand.get("scope") != expected_scope
            or estimand.get("included_components") != included_components
            or estimand.get("unavailable_components") != unavailable_components
        ):
            raise RatingAuthorityError(
                f"teams.{side}.estimand does not match component availability"
            )
        posterior_mean = _finite(
            team.get("posterior_mean"), f"teams.{side}.posterior_mean"
        )
        identified_sum = sum(
            float(normalized_components[component]["posterior_mean"])
            for component in included_components
        )
        if not math.isclose(
            posterior_mean,
            identified_sum,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RatingAuthorityError(
                f"teams.{side}.posterior_mean does not equal its identified estimand components"
            )
        normalized.append(
            {
                "side": side,
                "organization_id": _nonempty(
                    team.get("organization_id"),
                    f"teams.{side}.organization_id",
                ),
                "organization_name": _nonempty(
                    team.get("organization_name"),
                    f"teams.{side}.organization_name",
                ),
                "roster_id": _nonempty(
                    team.get("roster_id"), f"teams.{side}.roster_id"
                ),
                "players": normalized_players,
                "components": normalized_components,
                "estimand": {
                    "scope": expected_scope,
                    "included_components": included_components,
                    "unavailable_components": unavailable_components,
                },
                "posterior_mean": posterior_mean,
                "posterior_interval_95": _interval(
                    team.get("posterior_interval_95"),
                    f"teams.{side}.posterior_interval_95",
                    posterior_mean,
                ),
            }
        )
    if len({team["organization_id"] for team in normalized}) != 2:
        raise RatingAuthorityError("event rating organizations must be distinct")
    if len({team["roster_id"] for team in normalized}) != 2:
        raise RatingAuthorityError("event rating roster ids must be distinct")
    if normalized[0]["estimand"] != normalized[1]["estimand"]:
        raise RatingAuthorityError(
            "blue and red event ratings must use the same identified-component estimand"
        )
    return normalized


def _normalize_strength_difference(
    value: Any, teams: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RatingAuthorityError("strength_difference must be a mapping")
    _exact_keys(
        value,
        {
            "orientation",
            "estimand_scope",
            "included_components",
            "posterior_mean",
            "posterior_interval_95",
        },
        "strength_difference",
    )
    if value.get("orientation") != "blue_minus_red":
        raise RatingAuthorityError(
            "strength_difference orientation must be blue_minus_red"
        )
    team_estimand = teams[0]["estimand"]
    if (
        value.get("estimand_scope") != team_estimand["scope"]
        or value.get("included_components") != team_estimand["included_components"]
    ):
        raise RatingAuthorityError(
            "strength_difference does not preserve the teams' identified-component estimand"
        )
    mean = _finite(value.get("posterior_mean"), "strength_difference.posterior_mean")
    expected = float(teams[0]["posterior_mean"]) - float(
        teams[1]["posterior_mean"]
    )
    if not math.isclose(mean, expected, rel_tol=0.0, abs_tol=1e-9):
        raise RatingAuthorityError(
            "strength_difference does not replay blue minus red"
        )
    return {
        "orientation": "blue_minus_red",
        "estimand_scope": team_estimand["scope"],
        "included_components": list(team_estimand["included_components"]),
        "posterior_mean": mean,
        "posterior_interval_95": _interval(
            value.get("posterior_interval_95"),
            "strength_difference.posterior_interval_95",
            mean,
        ),
    }


def build_event_rating_receipt(
    *,
    rating_record_id: str,
    producer_id: str,
    event_id: str,
    event_start: str,
    league: str,
    roster_receipt_sha256: str,
    roster_registry_sha256: str,
    data_cutoff_at: str,
    produced_at: str,
    valid_until: str,
    maximum_data_age_seconds: int,
    artifacts: Mapping[str, Mapping[str, str]],
    prediction_receipt: Mapping[str, str],
    teams: Sequence[Mapping[str, Any]],
    strength_difference: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a non-authorizing event-rating candidate."""
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "candidate",
        "scope": REGISTRY_SCOPE,
        "public_or_transactional_use": False,
        "rating_record_id": _nonempty(rating_record_id, "rating_record_id"),
        "producer_id": _nonempty(producer_id, "producer_id"),
        "event_id": _nonempty(event_id, "event_id"),
        "event_start": event_start,
        "league": _nonempty(league, "league"),
        "roster_receipt_sha256": _sha(
            roster_receipt_sha256, "roster_receipt_sha256"
        ),
        "roster_registry_sha256": _sha(
            roster_registry_sha256, "roster_registry_sha256"
        ),
        "data_cutoff_at": data_cutoff_at,
        "produced_at": produced_at,
        "valid_until": valid_until,
        "maximum_data_age_seconds": maximum_data_age_seconds,
        "aggregation_method": AGGREGATION_METHOD,
        "uncertainty_method": UNCERTAINTY_METHOD,
        "artifacts": dict(artifacts),
        "prediction_receipt": dict(prediction_receipt),
        "teams": list(teams),
        "strength_difference": dict(strength_difference),
    }
    validate_event_rating_receipt(receipt)
    return receipt


def validate_event_rating_receipt(
    receipt: Mapping[str, Any], *, expected_receipt_sha256: str | None = None
) -> dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise RatingAuthorityError("event rating receipt must be a mapping")
    _exact_keys(
        receipt,
        {
            "schema_version",
            "status",
            "scope",
            "public_or_transactional_use",
            "rating_record_id",
            "producer_id",
            "event_id",
            "event_start",
            "league",
            "roster_receipt_sha256",
            "roster_registry_sha256",
            "data_cutoff_at",
            "produced_at",
            "valid_until",
            "maximum_data_age_seconds",
            "aggregation_method",
            "uncertainty_method",
            "artifacts",
            "prediction_receipt",
            "teams",
            "strength_difference",
        },
        "event rating receipt",
    )
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("status") != "candidate"
        or receipt.get("scope") != REGISTRY_SCOPE
        or receipt.get("public_or_transactional_use") is not False
    ):
        raise RatingAuthorityError("event rating receipt claim boundary is invalid")
    actual_sha256 = sha256_json(receipt)
    if expected_receipt_sha256 is not None and actual_sha256 != _sha(
        expected_receipt_sha256, "expected_receipt_sha256"
    ):
        raise RatingAuthorityError("event rating receipt digest mismatch")
    for field in ("rating_record_id", "producer_id", "event_id"):
        _nonempty(receipt.get(field), field)
    league = _nonempty(receipt.get("league"), "league")
    if league != league.upper():
        raise RatingAuthorityError("league must use its canonical uppercase id")
    event_start = _timestamp(receipt.get("event_start"), "event_start")
    data_cutoff = _timestamp(receipt.get("data_cutoff_at"), "data_cutoff_at")
    produced_at = _timestamp(receipt.get("produced_at"), "produced_at")
    valid_until = _timestamp(receipt.get("valid_until"), "valid_until")
    if data_cutoff > produced_at:
        raise RatingAuthorityError("data_cutoff_at cannot be after produced_at")
    if produced_at >= event_start:
        raise RatingAuthorityError("event rating must be produced before event_start")
    if valid_until < event_start:
        raise RatingAuthorityError("event rating must remain valid through event_start")
    maximum_age = receipt.get("maximum_data_age_seconds")
    if (
        isinstance(maximum_age, bool)
        or not isinstance(maximum_age, int)
        or maximum_age <= 0
        or maximum_age > MAX_RATING_AGE_SECONDS
    ):
        raise RatingAuthorityError("maximum_data_age_seconds is outside policy")
    if (event_start - data_cutoff).total_seconds() > maximum_age:
        raise RatingAuthorityError("rating data cutoff exceeds its freshness policy")
    _sha(receipt.get("roster_receipt_sha256"), "roster_receipt_sha256")
    _sha(receipt.get("roster_registry_sha256"), "roster_registry_sha256")
    if receipt.get("aggregation_method") != AGGREGATION_METHOD:
        raise RatingAuthorityError("rating aggregation method is not recognized")
    if receipt.get("uncertainty_method") != UNCERTAINTY_METHOD:
        raise RatingAuthorityError("rating uncertainty method is not recognized")
    artifacts = _normalize_artifacts(receipt.get("artifacts"))
    prediction_receipt = _prediction_receipt_reference(
        receipt.get("prediction_receipt")
    )
    teams = _normalize_teams(receipt.get("teams"))
    strength = _normalize_strength_difference(
        receipt.get("strength_difference"), teams
    )
    return {
        **dict(receipt),
        "artifacts": artifacts,
        "prediction_receipt": prediction_receipt,
        "teams": teams,
        "strength_difference": strength,
        "receipt_sha256": actual_sha256,
    }


def build_event_rating_registry(
    *,
    receipts: Sequence[tuple[str, Mapping[str, Any]]],
    registry_id: str,
    independent_reviewer_id: str,
    issued_at: str,
) -> dict[str, Any]:
    """Build a registry candidate whose digest still needs external pinning."""
    issued = _timestamp(issued_at, "issued_at")
    reviewer = _nonempty(independent_reviewer_id, "independent_reviewer_id")
    entries: list[dict[str, Any]] = []
    for locator, candidate in receipts:
        checked = validate_event_rating_receipt(candidate)
        _receipt_locator(locator)
        if reviewer == checked["producer_id"]:
            raise RatingAuthorityError(
                "independent reviewer cannot be the rating producer"
            )
        if issued < _timestamp(checked["produced_at"], "produced_at"):
            raise RatingAuthorityError("registry cannot predate its rating receipt")
        if issued >= _timestamp(checked["event_start"], "event_start"):
            raise RatingAuthorityError("rating registry must be issued before event_start")
        entries.append(
            {
                "event_id": checked["event_id"],
                "event_start": checked["event_start"],
                "league": checked["league"],
                "blue_organization_id": checked["teams"][0]["organization_id"],
                "blue_organization_name": checked["teams"][0]["organization_name"],
                "red_organization_id": checked["teams"][1]["organization_id"],
                "red_organization_name": checked["teams"][1]["organization_name"],
                "roster_receipt_sha256": checked["roster_receipt_sha256"],
                "roster_registry_sha256": checked["roster_registry_sha256"],
                "rating_record_id": checked["rating_record_id"],
                "producer_id": checked["producer_id"],
                "receipt_locator": locator,
                "receipt_sha256": checked["receipt_sha256"],
            }
        )
    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "status": "approved",
        "scope": REGISTRY_SCOPE,
        "public_or_transactional_use": False,
        "match_probability_authorized": False,
        "betting_decision_authorized": False,
        "registry_id": _nonempty(registry_id, "registry_id"),
        "independent_reviewer_id": reviewer,
        "issued_at": issued_at,
        "authorized_uses": list(AUTHORIZED_USES),
        "review_gates": {gate: "passed" for gate in REVIEW_GATES},
        "entries": sorted(entries, key=lambda entry: entry["event_id"]),
    }
    validate_event_rating_registry(
        registry, expected_registry_sha256=sha256_json(registry)
    )
    return registry


def validate_event_rating_registry(
    registry: Mapping[str, Any], *, expected_registry_sha256: str | None
) -> dict[str, Any]:
    if expected_registry_sha256 is None:
        raise RegisteredEventRatingUnavailable("rating_registry_not_registered")
    expected_sha = _sha(expected_registry_sha256, "expected_registry_sha256")
    if not isinstance(registry, Mapping):
        raise RatingAuthorityError("event rating registry must be a mapping")
    if sha256_json(registry) != expected_sha:
        raise RegisteredEventRatingUnavailable("rating_registry_digest_mismatch")
    _exact_keys(
        registry,
        {
            "schema_version",
            "status",
            "scope",
            "public_or_transactional_use",
            "match_probability_authorized",
            "betting_decision_authorized",
            "registry_id",
            "independent_reviewer_id",
            "issued_at",
            "authorized_uses",
            "review_gates",
            "entries",
        },
        "event rating registry",
    )
    if (
        registry.get("schema_version") != REGISTRY_SCHEMA_VERSION
        or registry.get("status") != "approved"
        or registry.get("scope") != REGISTRY_SCOPE
        or registry.get("public_or_transactional_use") is not False
        or registry.get("match_probability_authorized") is not False
        or registry.get("betting_decision_authorized") is not False
    ):
        raise RatingAuthorityError("event rating registry claim boundary is invalid")
    _nonempty(registry.get("registry_id"), "registry_id")
    reviewer = _nonempty(
        registry.get("independent_reviewer_id"), "independent_reviewer_id"
    )
    _timestamp(registry.get("issued_at"), "issued_at")
    if registry.get("authorized_uses") != list(AUTHORIZED_USES):
        raise RatingAuthorityError("event rating authorized uses are invalid")
    gates = registry.get("review_gates")
    if not isinstance(gates, Mapping) or set(gates) != set(REVIEW_GATES):
        raise RatingAuthorityError("event rating review gates are incomplete")
    if any(gates[gate] != "passed" for gate in REVIEW_GATES):
        raise RatingAuthorityError("event rating review gate is not passed")
    entries = registry.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RatingAuthorityError("event rating registry entries must be non-empty")
    expected_entry_keys = {
        "event_id",
        "event_start",
        "league",
        "blue_organization_id",
        "blue_organization_name",
        "red_organization_id",
        "red_organization_name",
        "roster_receipt_sha256",
        "roster_registry_sha256",
        "rating_record_id",
        "producer_id",
        "receipt_locator",
        "receipt_sha256",
    }
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise RatingAuthorityError("event rating registry entry must be a mapping")
        _exact_keys(entry, expected_entry_keys, "event rating registry entry")
        event_id = _nonempty(entry.get("event_id"), "entry.event_id")
        if event_id in seen:
            raise RatingAuthorityError("event rating registry contains an ambiguous event")
        seen.add(event_id)
        _timestamp(entry.get("event_start"), "entry.event_start")
        for field in (
            "league",
            "blue_organization_id",
            "blue_organization_name",
            "red_organization_id",
            "red_organization_name",
            "rating_record_id",
            "producer_id",
        ):
            _nonempty(entry.get(field), f"entry.{field}")
        if entry.get("producer_id") == reviewer:
            raise RatingAuthorityError(
                "independent reviewer cannot be the rating producer"
            )
        _sha(entry.get("roster_receipt_sha256"), "entry.roster_receipt_sha256")
        _sha(entry.get("roster_registry_sha256"), "entry.roster_registry_sha256")
        _receipt_locator(entry.get("receipt_locator"))
        _sha(entry.get("receipt_sha256"), "entry.receipt_sha256")
        normalized.append(dict(entry))
    if normalized != sorted(normalized, key=lambda entry: entry["event_id"]):
        raise RatingAuthorityError("event rating registry entries are not ordered")
    return {**dict(registry), "entries": normalized}


def _safe_repo_file(root: Path, locator: str) -> Path:
    relative = PurePosixPath(locator)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise RegisteredEventRatingUnavailable("rating_artifact_path_invalid")
    root_real = root.resolve(strict=True)
    current = root_real
    for part in relative.parts:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as exc:
            raise RegisteredEventRatingUnavailable(
                "rating_artifact_missing"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise RegisteredEventRatingUnavailable(
                "rating_artifact_symlink_rejected"
            )
    metadata = os.lstat(current)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RegisteredEventRatingUnavailable(
            "rating_artifact_not_unaliased_file"
        )
    try:
        current.resolve(strict=True).relative_to(root_real)
    except ValueError as exc:
        raise RegisteredEventRatingUnavailable(
            "rating_artifact_path_escape"
        ) from exc
    return current


def _assert_roster_binding(
    receipt: Mapping[str, Any], registered_roster: Mapping[str, Any]
) -> None:
    if registered_roster.get("status") != "registered":
        raise RegisteredEventRatingUnavailable("pre_event_roster_not_registered")
    if registered_roster.get("receipt_sha256") != receipt["roster_receipt_sha256"]:
        raise RegisteredEventRatingUnavailable(
            "rating_roster_receipt_binding_mismatch"
        )
    if registered_roster.get("registry_sha256") != receipt["roster_registry_sha256"]:
        raise RegisteredEventRatingUnavailable(
            "rating_roster_registry_binding_mismatch"
        )
    roster = registered_roster.get("roster")
    if not isinstance(roster, Mapping) or not isinstance(roster.get("teams"), list):
        raise RegisteredEventRatingUnavailable("registered_roster_payload_missing")
    roster_teams = roster["teams"]
    if len(roster_teams) != 2:
        raise RegisteredEventRatingUnavailable("registered_roster_payload_invalid")
    for index, side in enumerate(SIDES):
        roster_team = roster_teams[index]
        rating_team = receipt["teams"][index]
        for field in (
            "side",
            "organization_id",
            "organization_name",
            "roster_id",
        ):
            if roster_team.get(field) != rating_team[field]:
                raise RegisteredEventRatingUnavailable(
                    f"rating_roster_{side}_{field}_binding_mismatch"
                )
        roster_players = roster_team.get("players")
        if not isinstance(roster_players, list) or len(roster_players) != len(ROLES):
            raise RegisteredEventRatingUnavailable("registered_roster_payload_invalid")
        for role_index, role in enumerate(ROLES):
            roster_player = roster_players[role_index]
            rating_player = rating_team["players"][role_index]
            for field in ("role", "player_id", "display_name"):
                if roster_player.get(field) != rating_player[field]:
                    raise RegisteredEventRatingUnavailable(
                        f"rating_roster_{side}_{role}_{field}_binding_mismatch"
                    )


def _prediction_timestamp(value: Any, label: str) -> datetime:
    """Treat the sealed source snapshot's naive timestamp as UTC."""

    text = _nonempty(value, label)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RegisteredEventRatingUnavailable(
            "rating_prediction_source_time_invalid"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _expected_rating_outputs_from_prediction(
    prediction: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    diagnostics = prediction["event_rating_diagnostics"]
    embedded_roster = prediction["input_receipts"]["roster"]["receipt"]
    roster_teams = embedded_roster["teams"]
    players = diagnostics["players"]
    diagnostic_teams = diagnostics["teams"]
    expected_teams: list[dict[str, Any]] = []
    for index, side in enumerate(SIDES):
        roster_team = roster_teams[index]
        diagnostic_team = diagnostic_teams[index]
        side_players = [player for player in players if player["side"] == side]
        runtime_components = diagnostic_team["components"]
        player_component = runtime_components["player_aggregate"]
        organization_component = runtime_components["organization_residual"]
        joint = diagnostic_team["joint_player_plus_organization"]
        player_mean = (
            ratings_ledger.rating.DISPLAY_ANCHOR
            + ratings_ledger.rating.DISPLAY_LOGIT_SCALE
            * float(player_component["posterior_mean_logit"])
        )
        organization_mean = (
            ratings_ledger.rating.DISPLAY_LOGIT_SCALE
            * float(organization_component["posterior_mean_logit"])
        )
        team_mean = float(joint["display_rating_mean"])
        team_sd = float(joint["display_rating_sd"])
        expected_teams.append(
            {
                "side": side,
                "organization_id": roster_team["organization_id"],
                "organization_name": roster_team["organization_name"],
                "roster_id": roster_team["roster_id"],
                "players": [
                    {
                        "role": player["role"],
                        "player_id": player["player_id"],
                        "display_name": player["display_name"],
                        "posterior_mean": float(player["display_rating_mean"]),
                        "posterior_sd": float(player["display_rating_sd"]),
                    }
                    for player in side_players
                ],
                "components": {
                    "player_aggregate": {
                        "status": "ESTIMATED",
                        "posterior_mean": player_mean,
                        "posterior_sd": (
                            ratings_ledger.rating.DISPLAY_LOGIT_SCALE
                            * float(player_component["posterior_sd_logit"])
                        ),
                        "unavailable_reason": None,
                    },
                    "organization_residual": {
                        "status": "ESTIMATED",
                        "posterior_mean": organization_mean,
                        "posterior_sd": (
                            ratings_ledger.rating.DISPLAY_LOGIT_SCALE
                            * float(organization_component["posterior_sd_logit"])
                        ),
                        "unavailable_reason": None,
                    },
                    "lineup_synergy": {
                        "status": "UNAVAILABLE",
                        "posterior_mean": None,
                        "posterior_sd": None,
                        "unavailable_reason": "not_identified_by_evaluated_runtime",
                    },
                    "team_policy": {
                        "status": "UNAVAILABLE",
                        "posterior_mean": None,
                        "posterior_sd": None,
                        "unavailable_reason": "not_identified_by_evaluated_runtime",
                    },
                },
                "estimand": {
                    "scope": "exact_roster_player_plus_organization_identified_components",
                    "included_components": [
                        "player_aggregate",
                        "organization_residual",
                    ],
                    "unavailable_components": ["lineup_synergy", "team_policy"],
                },
                "posterior_mean": team_mean,
                "posterior_interval_95": [
                    team_mean - NORMAL_95_Z * team_sd,
                    team_mean + NORMAL_95_Z * team_sd,
                ],
            }
        )
    candidate = prediction["evaluation_predictions"][ratings_ledger.MODEL_IDS[0]]
    difference_mean = (
        ratings_ledger.rating.DISPLAY_LOGIT_SCALE
        * float(candidate["latent_mean"])
    )
    difference_sd = (
        ratings_ledger.rating.DISPLAY_LOGIT_SCALE
        * math.sqrt(float(candidate["latent_variance"]))
    )
    expected_difference = {
        "orientation": "blue_minus_red",
        "estimand_scope": "exact_roster_player_plus_organization_identified_components",
        "included_components": ["player_aggregate", "organization_residual"],
        "posterior_mean": difference_mean,
        "posterior_interval_95": [
            difference_mean - NORMAL_95_Z * difference_sd,
            difference_mean + NORMAL_95_Z * difference_sd,
        ],
    }
    return expected_teams, expected_difference


def _assert_exact_prediction_receipt_binding(
    receipt: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    reference = receipt["prediction_receipt"]
    prediction_path = _safe_repo_file(root, reference["locator"])
    raw = prediction_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != reference["raw_sha256"]:
        raise RegisteredEventRatingUnavailable(
            "rating_prediction_receipt_digest_mismatch"
        )
    try:
        prediction = ratings_ledger.replay_pre_event_prediction_receipt(
            _read_json_bytes(raw, "rating prediction receipt"), root=root
        )
    except (ratings_ledger.PredictionLedgerError, RatingAuthorityError) as exc:
        raise RegisteredEventRatingUnavailable(
            "rating_prediction_receipt_replay_failed"
        ) from exc
    if prediction.get("artifact_sha256") != reference["artifact_sha256"]:
        raise RegisteredEventRatingUnavailable(
            "rating_prediction_receipt_artifact_mismatch"
        )
    event = prediction["event"]
    roster_binding = prediction["input_receipts"]["roster"]
    expected_event = {
        "event_id": receipt["event_id"],
        "event_start_utc": receipt["event_start"],
        "league": receipt["league"],
    }
    for field, expected in expected_event.items():
        if event.get(field) != expected:
            raise RegisteredEventRatingUnavailable(
                f"rating_prediction_{field}_binding_mismatch"
            )
    if roster_binding.get("canonical_sha256") != receipt["roster_receipt_sha256"]:
        raise RegisteredEventRatingUnavailable(
            "rating_prediction_roster_receipt_binding_mismatch"
        )
    if _timestamp(prediction["captured_at_utc"], "prediction.captured_at_utc") != _timestamp(
        receipt["produced_at"], "produced_at"
    ):
        raise RegisteredEventRatingUnavailable(
            "rating_prediction_capture_time_binding_mismatch"
        )
    if _prediction_timestamp(
        prediction["source_snapshot"]["latest_observed_source_time"],
        "prediction.source_snapshot.latest_observed_source_time",
    ) != _timestamp(receipt["data_cutoff_at"], "data_cutoff_at"):
        raise RegisteredEventRatingUnavailable(
            "rating_prediction_data_cutoff_binding_mismatch"
        )
    expected_teams, expected_difference = _expected_rating_outputs_from_prediction(
        prediction
    )
    if receipt["teams"] != expected_teams:
        raise RegisteredEventRatingUnavailable(
            "rating_outputs_do_not_replay_exact_prediction"
        )
    if receipt["strength_difference"] != expected_difference:
        raise RegisteredEventRatingUnavailable(
            "rating_strength_difference_does_not_replay_exact_prediction"
        )
    return prediction


def build_event_rating_receipt_from_prediction(
    *,
    prediction_receipt_locator: str,
    rating_record_id: str,
    producer_id: str,
    roster_registry_sha256: str,
    valid_until: str,
    maximum_data_age_seconds: int,
    artifacts: Mapping[str, Mapping[str, str]],
    root: Path = ROOT,
) -> dict[str, Any]:
    """Build event ratings only by replaying one exact v3 prediction receipt."""

    locator = _prediction_receipt_reference(
        {
            "locator": prediction_receipt_locator,
            "raw_sha256": "0" * 64,
            "artifact_sha256": "0" * 64,
        }
    )["locator"]
    path = _safe_repo_file(root, locator)
    raw = path.read_bytes()
    try:
        prediction = ratings_ledger.replay_pre_event_prediction_receipt(
            _read_json_bytes(raw, "rating prediction receipt"), root=root
        )
    except (ratings_ledger.PredictionLedgerError, RatingAuthorityError) as exc:
        raise RatingAuthorityError(
            "exact v3 prediction receipt did not replay"
        ) from exc
    teams, strength_difference = _expected_rating_outputs_from_prediction(
        prediction
    )
    event = prediction["event"]
    roster = prediction["input_receipts"]["roster"]
    return build_event_rating_receipt(
        rating_record_id=rating_record_id,
        producer_id=producer_id,
        event_id=event["event_id"],
        event_start=event["event_start_utc"],
        league=event["league"],
        roster_receipt_sha256=roster["canonical_sha256"],
        roster_registry_sha256=roster_registry_sha256,
        data_cutoff_at=_prediction_timestamp(
            prediction["source_snapshot"]["latest_observed_source_time"],
            "prediction.source_snapshot.latest_observed_source_time",
        ).isoformat(),
        produced_at=_timestamp(
            prediction["captured_at_utc"], "prediction.captured_at_utc"
        ).isoformat(),
        valid_until=valid_until,
        maximum_data_age_seconds=maximum_data_age_seconds,
        artifacts=artifacts,
        prediction_receipt={
            "locator": locator,
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "artifact_sha256": _sha(
                prediction.get("artifact_sha256"),
                "prediction.artifact_sha256",
            ),
        },
        teams=teams,
        strength_difference=strength_difference,
    )


def load_registered_event_rating(
    *,
    registry_locator: str,
    expected_registry_sha256: str | None,
    registered_roster: Mapping[str, Any],
    event_id: str,
    event_start: str,
    league: str,
    blue_organization_name: str,
    red_organization_name: str,
    as_of: datetime,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    """Replay one independently registered event-rating component bundle."""
    try:
        semantic = semantic_authority.load_active_semantic_rating_authority_v1(
            root=root,
            environment=environment,
            as_of=as_of,
        )
    except semantic_authority.SemanticRatingAuthorityError as exc:
        raise RegisteredEventRatingUnavailable(
            "semantic_rating_authority_unavailable"
        ) from exc
    if not expected_registry_sha256:
        raise RegisteredEventRatingUnavailable("rating_registry_not_registered")
    registry_path = _safe_repo_file(root, registry_locator)
    registry = validate_event_rating_registry(
        _read_json_bytes(registry_path.read_bytes(), "event rating registry"),
        expected_registry_sha256=expected_registry_sha256,
    )
    as_of_utc = as_of.astimezone(timezone.utc)
    if _timestamp(registry["issued_at"], "issued_at") > as_of_utc:
        raise RegisteredEventRatingUnavailable("rating_registry_from_future")
    semantic_receipt = semantic["receipt"]
    if _timestamp(
        semantic_receipt["issued_at_utc"], "semantic authority issued_at_utc"
    ) > _timestamp(registry["issued_at"], "issued_at"):
        raise RegisteredEventRatingUnavailable(
            "rating_registry_predates_semantic_authority"
        )
    matches = [entry for entry in registry["entries"] if entry["event_id"] == event_id]
    if len(matches) != 1:
        raise RegisteredEventRatingUnavailable("registered_event_rating_unavailable")
    entry = matches[0]
    expected_bindings = {
        "event_start": _timestamp(event_start, "event_start"),
        "league": league,
        "blue_organization_name": blue_organization_name,
        "red_organization_name": red_organization_name,
        "roster_receipt_sha256": registered_roster.get("receipt_sha256"),
        "roster_registry_sha256": registered_roster.get("registry_sha256"),
    }
    for field, expected in expected_bindings.items():
        actual: Any = entry.get(field)
        if field == "event_start":
            actual = _timestamp(actual, "entry.event_start")
        if actual != expected:
            raise RegisteredEventRatingUnavailable(
                f"rating_{field}_binding_mismatch"
            )
    receipt_path = _safe_repo_file(root, entry["receipt_locator"])
    receipt = validate_event_rating_receipt(
        _read_json_bytes(receipt_path.read_bytes(), "event rating receipt"),
        expected_receipt_sha256=entry["receipt_sha256"],
    )
    for field in (
        "event_id",
        "event_start",
        "league",
        "roster_receipt_sha256",
        "roster_registry_sha256",
        "rating_record_id",
        "producer_id",
    ):
        if receipt.get(field) != entry.get(field):
            raise RegisteredEventRatingUnavailable(
                f"rating_{field}_binding_mismatch"
            )
    for index, side in enumerate(SIDES):
        team = receipt["teams"][index]
        for suffix in ("organization_id", "organization_name"):
            if team[suffix] != entry[f"{side}_{suffix}"]:
                raise RegisteredEventRatingUnavailable(
                    f"rating_{side}_{suffix}_binding_mismatch"
                )
    if _timestamp(receipt["produced_at"], "produced_at") > as_of_utc:
        raise RegisteredEventRatingUnavailable("rating_receipt_from_future")
    if as_of_utc > _timestamp(receipt["valid_until"], "valid_until"):
        raise RegisteredEventRatingUnavailable("registered_event_rating_expired")
    if _timestamp(
        semantic_receipt["valid_until_utc"], "semantic authority valid_until_utc"
    ) < _timestamp(receipt["event_start"], "event_start"):
        raise RegisteredEventRatingUnavailable(
            "semantic_rating_authority_expires_before_event"
        )
    maximum_semantic_age = semantic_receipt["deployment_policy"][
        "maximum_data_age_seconds"
    ]
    if receipt["maximum_data_age_seconds"] > maximum_semantic_age:
        raise RegisteredEventRatingUnavailable(
            "rating_freshness_exceeds_semantic_authority"
        )
    _assert_roster_binding(receipt, registered_roster)
    if receipt["artifacts"] != semantic["deployment_artifacts"]:
        raise RegisteredEventRatingUnavailable(
            "rating_artifacts_not_semantically_authorized"
        )
    for kind, reference in receipt["artifacts"].items():
        artifact_path = _safe_repo_file(root, reference["locator"])
        if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != reference[
            "raw_sha256"
        ]:
            raise RegisteredEventRatingUnavailable(
                f"rating_{kind}_artifact_digest_mismatch"
            )
    prediction = _assert_exact_prediction_receipt_binding(receipt, root=root)
    return {
        "status": "registered",
        "player_rating_authorized": True,
        "team_rating_authorized": True,
        "match_probability_authorized": False,
        "betting_decision_authorized": False,
        "ratings": {
            "teams": receipt["teams"],
            "strength_difference": receipt["strength_difference"],
            "aggregation_method": receipt["aggregation_method"],
            "uncertainty_method": receipt["uncertainty_method"],
            "data_cutoff_at": receipt["data_cutoff_at"],
            "valid_until": receipt["valid_until"],
        },
        "receipt_sha256": receipt["receipt_sha256"],
        "rating_record_id": receipt["rating_record_id"],
        "prediction_receipt": dict(receipt["prediction_receipt"]),
        "prediction_runtime_model_ids": list(
            prediction["evaluation_predictions"]
        ),
        "semantic_authority_id": semantic_receipt["authority_id"],
        "semantic_authority_sha256": semantic["receipt_raw_sha256"],
        "registry_id": registry["registry_id"],
        "registry_sha256": expected_registry_sha256,
        "roster_receipt_sha256": receipt["roster_receipt_sha256"],
        "roster_registry_sha256": receipt["roster_registry_sha256"],
        "review_gates": dict(registry["review_gates"]),
        "blockers": [
            "rating_to_match_probability_calibration_unavailable",
            "draft_rating_combination_authority_unavailable",
        ],
    }


__all__ = [
    "AGGREGATION_METHOD",
    "ARTIFACT_KINDS",
    "AUTHORIZED_USES",
    "RatingAuthorityError",
    "RegisteredEventRatingUnavailable",
    "REVIEW_GATES",
    "TEAM_COMPONENTS",
    "UNCERTAINTY_METHOD",
    "build_event_rating_receipt",
    "build_event_rating_receipt_from_prediction",
    "build_event_rating_registry",
    "canonical_bytes",
    "load_registered_event_rating",
    "sha256_json",
    "validate_event_rating_receipt",
    "validate_event_rating_registry",
]
