"""Provisional L5 exact-roster Team Rating mechanics.

This module is deliberately development-only.  It consumes supplied L4-like
posterior states; it does not authorize, publish, or rank a Team Rating.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence


ROLES = ("top", "jungle", "mid", "bot", "support")
DISPLAY_ANCHOR = 1500.0
# Registered 1500/400 Elo display contract (mathematical-contract.md sections
# 2-3, estimands.md): 400 display points map to ln(10) logits, so the
# logit-to-point conversion is c_E = 400/log(10), the same c_E as Player
# Rating.  The provisional mechanics must not use the legacy linear 400.0
# per-logit scale: it would inflate every rating difference and interval by
# a factor of ln(10) relative to the player scale.
DISPLAY_SCALE = 400.0 / math.log(10.0)
CLAIM_CEILING = MappingProxyType(
    {
        "authorized_team_rating": False,
        "rank_eligible": False,
        "public": False,
        "production": False,
        "probability": False,
        "reliability": False,
        "real_data": False,
        "sota": False,
        "c2_pass": False,
        "c3_pass": False,
        "pass_b2": False,
    }
)


class TeamRatingError(ValueError):
    """Base error for invalid or unavailable provisional Team Ratings."""


class RosterValidationError(TeamRatingError):
    """The requested roster is not an exact active official five."""


class TeamRatingUnavailable(TeamRatingError):
    """The requested estimand is structurally unavailable."""


class ArtifactIntegrityError(TeamRatingError):
    """A development artifact does not match its content identity."""


class AuthorizationError(TeamRatingError):
    """No independent later registrar authorizes this candidate."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RosterValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise RosterValidationError(f"{field} must be finite")
    return result


def _parse_time(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise RosterValidationError(f"{field} must be a non-empty timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RosterValidationError(f"{field} is not ISO-8601") from exc
    if parsed.tzinfo is None:
        raise RosterValidationError(f"{field} must include a timezone")
    return value


@dataclass(frozen=True)
class PlayerPosterior:
    player_id: str
    display_name: str
    role: str
    league_id: str
    scope: str
    posterior_mean: float
    active: bool
    as_of: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PlayerPosterior":
        required = {
            "player_id",
            "display_name",
            "role",
            "league_id",
            "scope",
            "posterior_mean",
            "active",
            "as_of",
        }
        if set(value) != required:
            raise RosterValidationError(
                f"player fields must be exactly {sorted(required)}"
            )
        player_id = value["player_id"]
        role = value["role"]
        league_id = value["league_id"]
        scope = value["scope"]
        if not isinstance(player_id, str) or not player_id:
            raise RosterValidationError("player_id must be non-empty")
        display_name = value["display_name"]
        if not isinstance(display_name, str) or not display_name:
            raise RosterValidationError("display_name must be non-empty")
        if role not in ROLES:
            raise RosterValidationError(f"unknown role: {role!r}")
        if not isinstance(league_id, str) or not league_id:
            raise RosterValidationError("league_id must be non-empty")
        if scope not in {"regional", "global"}:
            raise RosterValidationError("scope must be regional or global")
        if value["active"] is not True:
            raise RosterValidationError(f"inactive player: {player_id}")
        return cls(
            player_id=player_id,
            display_name=display_name,
            role=role,
            league_id=league_id,
            scope=scope,
            posterior_mean=_finite_number(
                value["posterior_mean"], "posterior_mean"
            ),
            active=True,
            as_of=_parse_time(value["as_of"], "player.as_of"),
        )


@dataclass(frozen=True)
class ExactRoster:
    roster_id: str
    organization_id: str
    league_id: str
    effective_at: str
    as_of: str
    players: tuple[PlayerPosterior, ...]
    source_receipt_sha256: str

    @property
    def player_ids(self) -> tuple[str, ...]:
        return tuple(player.player_id for player in self.players)

    @property
    def player_names(self) -> tuple[str, ...]:
        return tuple(player.display_name for player in self.players)

    @property
    def player_roles(self) -> tuple[str, ...]:
        return tuple(player.role for player in self.players)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ExactRoster":
        required = {
            "organization_id",
            "league_id",
            "effective_at",
            "as_of",
            "official",
            "active",
            "ambiguous",
            "custom",
            "hypothetical",
            "substitute",
            "fresh",
            "players",
        }
        if set(value) != required:
            raise RosterValidationError(
                f"roster fields must be exactly {sorted(required)}"
            )
        for field in ("organization_id", "league_id"):
            if not isinstance(value[field], str) or not value[field]:
                raise RosterValidationError(f"{field} must be non-empty")
        required_true = ("official", "active", "fresh")
        required_false = ("ambiguous", "custom", "hypothetical", "substitute")
        if any(value[field] is not True for field in required_true):
            raise RosterValidationError("roster must be official, active, and fresh")
        if any(value[field] is not False for field in required_false):
            raise RosterValidationError(
                "ambiguous, custom, hypothetical, or substitute rosters are unavailable"
            )
        effective_at = _parse_time(value["effective_at"], "effective_at")
        as_of = _parse_time(value["as_of"], "roster.as_of")
        if datetime.fromisoformat(effective_at.replace("Z", "+00:00")) > datetime.fromisoformat(
            as_of.replace("Z", "+00:00")
        ):
            raise RosterValidationError("effective_at cannot be after as_of")
        raw_players = value["players"]
        if not isinstance(raw_players, list) or len(raw_players) != 5:
            raise RosterValidationError("Team Rating requires exactly five players")
        players = tuple(PlayerPosterior.from_mapping(item) for item in raw_players)
        if len(set(player.player_id for player in players)) != 5:
            raise RosterValidationError("duplicate player identity")
        if tuple(player.role for player in players) != ROLES:
            raise RosterValidationError(
                f"players must be ordered in exact role order {ROLES}"
            )
        league_id = value["league_id"]
        if any(player.league_id != league_id for player in players):
            raise RosterValidationError("player and roster league identity mismatch")
        if any(player.as_of != as_of for player in players):
            raise RosterValidationError("player and roster as_of identity mismatch")
        identity = {
            "effective_at": effective_at,
            "as_of": as_of,
            "roles": [
                {"role": player.role, "player_id": player.player_id}
                for player in players
            ],
        }
        # The source receipt binds the exact ordered five with roles, names,
        # source times, and organization/league identity so the published
        # roster can be replayed against the source that produced it.
        receipt = {
            "roster_id": sha256_json(identity),
            "organization_id": value["organization_id"],
            "league_id": league_id,
            "effective_at": effective_at,
            "as_of": as_of,
            "players": [
                {
                    "player_id": player.player_id,
                    "display_name": player.display_name,
                    "role": player.role,
                    "posterior_mean": player.posterior_mean,
                }
                for player in players
            ],
        }
        return cls(
            roster_id=sha256_json(identity),
            organization_id=value["organization_id"],
            league_id=league_id,
            effective_at=effective_at,
            as_of=as_of,
            players=players,
            source_receipt_sha256=sha256_json(receipt),
        )


@dataclass(frozen=True)
class LeagueRating:
    league_id: str
    posterior_mean: float
    posterior_variance: float
    centered: bool
    reference_constrained: bool
    transfer_continuity: bool
    mobility_bridge_full_rank: bool
    separately_identified: bool

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "LeagueRating":
        required = {
            "league_id",
            "posterior_mean",
            "posterior_variance",
            "centered",
            "reference_constrained",
            "transfer_continuity",
            "mobility_bridge_full_rank",
            "separately_identified",
        }
        if set(value) != required:
            raise TeamRatingUnavailable(
                f"league rating fields must be exactly {sorted(required)}"
            )
        variance = _finite_number(
            value["posterior_variance"], "league.posterior_variance"
        )
        if variance < 0:
            raise TeamRatingUnavailable("league variance cannot be negative")
        return cls(
            league_id=str(value["league_id"]),
            posterior_mean=_finite_number(
                value["posterior_mean"], "league.posterior_mean"
            ),
            posterior_variance=variance,
            centered=value["centered"] is True,
            reference_constrained=value["reference_constrained"] is True,
            transfer_continuity=value["transfer_continuity"] is True,
            mobility_bridge_full_rank=value["mobility_bridge_full_rank"] is True,
            separately_identified=value["separately_identified"] is True,
        )

    @property
    def structurally_eligible(self) -> bool:
        return all(
            (
                self.centered,
                self.reference_constrained,
                self.transfer_continuity,
                self.mobility_bridge_full_rank,
                self.separately_identified,
            )
        )


@dataclass(frozen=True)
class TeamRating:
    roster_id: str
    scope: str
    player_ids: tuple[str, ...]
    player_names: tuple[str, ...]
    player_roles: tuple[str, ...]
    effective_at: str
    as_of: str
    roster_receipt_sha256: str
    evidence_state: str
    player_posterior_means: tuple[float, ...]
    roster_latent_mean: float
    roster_latent_variance: float
    league_latent_mean: float
    league_latent_variance: float
    posterior_mean: float
    posterior_interval_95: tuple[float, float]
    roster_strength_component: float
    league_rating_component: float
    lineup_synergy_component: dict[str, Any] | None
    policy_component: dict[str, Any] | None
    rank_eligibility: bool
    missing_c2_dependency: bool
    development_only: bool

    def _component_availability(self) -> dict[str, Any]:
        if self.lineup_synergy_component is None and self.policy_component is None:
            return {
                "policy": {
                    "available": False,
                    "status": "unavailable",
                    "reason": "resource-policy selection was deferred",
                    "blocker": "policy estimand is not identified",
                },
                "lineup_synergy": {
                    "available": False,
                    "status": "unavailable",
                    "reason": "lineup-synergy identification was deferred",
                    "blocker": "synergy estimand is not identified",
                },
            }
        return {
            "policy": {
                "available": self.policy_component is not None,
                "status": "estimated_with_uncertainty" if self.policy_component else "unavailable",
            },
            "lineup_synergy": {
                "available": self.lineup_synergy_component is not None,
                "status": "estimated_with_uncertainty" if self.lineup_synergy_component else "unavailable",
            },
        }

    def _reference_convention(self) -> dict[str, Any]:
        if self.lineup_synergy_component is None and self.policy_component is None:
            return {
                "status": "non_estimated",
                "computational_offset": 0.0,
                "contributes_exactly_zero": True,
                "covers": ["policy", "lineup_synergy"],
                "consumer_rule": "must_not_be_treated_as_an_estimate",
            }
        return {
            "status": "estimated_with_uncertainty",
            "computational_offset": 0.0,
            "contributes_exactly_zero": False,
            "covers": ["policy", "lineup_synergy"],
            "consumer_rule": "policy and synergy are dev estimates; rating remains development_only",
        }

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": "development_only",
            "roster_id": self.roster_id,
            "scope": self.scope,
            "model_scope": self.scope,
            "players": list(self.player_ids),
            "player_names": list(self.player_names),
            "player_roles": list(self.player_roles),
            "roster_effective_at": self.effective_at,
            "roster_as_of": self.as_of,
            "roster_receipt_sha256": self.roster_receipt_sha256,
            "evidence_state": self.evidence_state,
            "player_posterior_means": list(self.player_posterior_means),
            "rating_display": {
                "anchor": DISPLAY_ANCHOR,
                "scale": DISPLAY_SCALE,
            },
            "estimand": {
                "c_E": DISPLAY_SCALE,
                "A_q": self.roster_latent_mean,
                "gamma_q": (
                    self.lineup_synergy_component["gamma_hat"]
                    if self.lineup_synergy_component is not None else None
                ),
                "policy_q": (
                    self.policy_component["weights"]
                    if self.policy_component is not None else None
                ),
                "lambda_L": self.league_latent_mean,
            },
            "roster_strength_component": self.roster_strength_component,
            "league_rating_component": self.league_rating_component,
            "lineup_synergy_component": self.lineup_synergy_component,
            "policy_component": self.policy_component,
            "component_availability": self._component_availability(),
            "reference_convention": self._reference_convention(),
            "posterior_mean": self.posterior_mean,
            "posterior_interval_95": list(self.posterior_interval_95),
            "posterior_variance": DISPLAY_SCALE**2
            * (self.roster_latent_variance + self.league_latent_variance),
            "rank_eligibility": self.rank_eligibility,
            "missing_c2_dependency": self.missing_c2_dependency,
            "development_only": self.development_only,
            "claim_ceiling": dict(CLAIM_CEILING),
            "schema_conformance": {
                "production_team_rating_schema": False,
                "reason": "Reliability and independent authority are unavailable",
            },
        }
        verify_development_payload(payload)
        return payload


def verify_development_payload(payload: Mapping[str, Any]) -> None:
    """Reject any representation that turns unavailable components into estimates."""
    try:
        estimand = payload["estimand"]
        availability = payload["component_availability"]
        convention = payload["reference_convention"]
        claim_ceiling = payload["claim_ceiling"]
        schema_conformance = payload["schema_conformance"]
    except (KeyError, TypeError) as exc:
        raise ArtifactIntegrityError(
            "development payload is missing non-identification metadata"
        ) from exc
    expected_components = {"policy", "lineup_synergy"}
    if set(availability) != expected_components:
        raise ArtifactIntegrityError("component availability closure mismatch")
    synergy_component = payload.get("lineup_synergy_component")
    policy_component = payload.get("policy_component")
    if policy_component is None and synergy_component is not None:
        raise ArtifactIntegrityError("unavailable lineup synergy must be null")
    if synergy_component is None and policy_component is not None:
        raise ArtifactIntegrityError("unavailable policy component must be null")
    if (
        synergy_component is None
        and policy_component is None
        and (estimand.get("gamma_q") is not None or estimand.get("policy_q") is not None)
    ):
        raise ArtifactIntegrityError("unavailable policy/synergy estimands must be null")
    estimated = (
        synergy_component is not None
        or policy_component is not None
        or estimand.get("gamma_q") is not None
        or estimand.get("policy_q") is not None
    )
    if not estimated:
        for name in expected_components:
            component = availability[name]
            if component.get("available") is not False:
                raise ArtifactIntegrityError(f"{name} availability must be false")
            if component.get("status") != "unavailable":
                raise ArtifactIntegrityError(f"{name} status must be unavailable")
            if not component.get("reason") or not component.get("blocker"):
                raise ArtifactIntegrityError(f"{name} requires reason and blocker")
            if any(
                "interval" in key or "claim" in key
                for key in component
            ):
                raise ArtifactIntegrityError(
                    f"{name} cannot expose an interval or claim"
                )
        if convention != {
            "status": "non_estimated",
            "computational_offset": 0.0,
            "contributes_exactly_zero": True,
            "covers": ["policy", "lineup_synergy"],
            "consumer_rule": "must_not_be_treated_as_an_estimate",
        }:
            raise ArtifactIntegrityError("reference convention is not exact")
        expected_mean = (
            payload["rating_display"]["anchor"]
            + payload["roster_strength_component"]
            + payload["league_rating_component"]
            + convention["computational_offset"]
        )
        if not math.isclose(payload["posterior_mean"], expected_mean, abs_tol=1e-12):
            raise ArtifactIntegrityError("available-component identity does not reconcile")
    else:
        # Audited estimated-with-uncertainty state (identification audit strong).
        if synergy_component is None or policy_component is None:
            raise ArtifactIntegrityError("estimated state requires both components")
        if (
            availability["policy"].get("available") is not True
            or availability["policy"].get("status") != "estimated_with_uncertainty"
            or availability["lineup_synergy"].get("available") is not True
            or availability["lineup_synergy"].get("status") != "estimated_with_uncertainty"
        ):
            raise ArtifactIntegrityError("estimated availability must be explicit")
        if (
            not isinstance(estimand.get("gamma_q"), (int, float))
            or not isinstance(estimand.get("policy_q"), dict)
        ):
            raise ArtifactIntegrityError("estimated estimands must be numeric gamma and weight dict")
        if convention.get("status") != "estimated_with_uncertainty":
            raise ArtifactIntegrityError("estimated reference convention required")
        expected_mean = (
            payload["rating_display"]["anchor"]
            + payload["roster_strength_component"]
            + payload["league_rating_component"]
            + DISPLAY_SCALE * float(estimand["gamma_q"])
        )
        if not math.isclose(payload["posterior_mean"], expected_mean, abs_tol=1e-9):
            raise ArtifactIntegrityError("estimated-component identity does not reconcile")
    if schema_conformance.get("production_team_rating_schema") is not False:
        raise ArtifactIntegrityError("production schema conformance must remain false")


def _validate_covariance(covariance: Sequence[Sequence[Any]]) -> tuple[tuple[float, ...], ...]:
    if len(covariance) != 5 or any(
        not isinstance(row, Sequence) or len(row) != 5 for row in covariance
    ):
        raise RosterValidationError("joint covariance must be 5x5")
    matrix = tuple(
        tuple(_finite_number(value, "covariance") for value in row)
        for row in covariance
    )
    for i in range(5):
        if matrix[i][i] < 0:
            raise RosterValidationError("covariance diagonal cannot be negative")
        for j in range(5):
            if not math.isclose(matrix[i][j], matrix[j][i], abs_tol=1e-12):
                raise RosterValidationError("covariance must be symmetric")
    aggregate_variance = sum(sum(row) for row in matrix)
    if aggregate_variance < -1e-12:
        raise RosterValidationError("aggregate covariance is not positive")
    return matrix


def aggregate_team_rating(
    roster: ExactRoster,
    covariance: Sequence[Sequence[Any]],
    *,
    scope: str,
    league_rating: LeagueRating | None = None,
    estimand_inputs: Mapping[str, Any] | None = None,
) -> TeamRating:
    """Aggregate exact-five player latent states and their joint covariance.

    When ``estimand_inputs`` is supplied (bridge, roster_champions,
    resource_share, reference_weights, player_span), the L5 policy / lineup
    synergy estimand opener runs; if its identification audit is strong the
    components are estimated-with-uncertainty, otherwise they remain the
    null-with-blocker fallback (fail closed).
    """
    if scope not in {"regional", "global"}:
        raise TeamRatingUnavailable("scope must be regional or global")
    if any(player.scope != scope for player in roster.players):
        raise TeamRatingUnavailable(f"{scope} requires {scope}-scoped player states")
    matrix = _validate_covariance(covariance)
    player_means = tuple(player.posterior_mean for player in roster.players)
    roster_mean = sum(player_means)
    roster_variance = max(0.0, sum(sum(row) for row in matrix))
    league_mean = 0.0
    league_variance = 0.0
    if scope == "regional":
        if league_rating is not None:
            raise TeamRatingUnavailable(
                "regional Team Rating cannot include League Rating"
            )
    else:
        if league_rating is None:
            raise TeamRatingUnavailable("global Team Rating requires League Rating")
        if league_rating.league_id != roster.league_id:
            raise TeamRatingUnavailable("League Rating identity mismatch")
        if not league_rating.structurally_eligible:
            raise TeamRatingUnavailable(
                "global player and League effects are not separately identified"
            )
        league_mean = league_rating.posterior_mean
        league_variance = league_rating.posterior_variance
    roster_component = DISPLAY_SCALE * roster_mean
    league_component = DISPLAY_SCALE * league_mean
    lineup_synergy_component: dict[str, Any] | None = None
    policy_component: dict[str, Any] | None = None
    gamma_mean = 0.0
    gamma_variance = 0.0
    if estimand_inputs is not None:
        try:
            from .estimands_v1 import opened_estimands
            opened = opened_estimands(
                estimand_inputs["bridge"],
                estimand_inputs["roster_champions"],
                estimand_inputs["resource_share"],
                estimand_inputs["reference_weights"],
                estimand_inputs["player_span"],
            )
        except (KeyError, TypeError):
            opened = None
        if opened is not None:
            lineup_synergy_component = opened["lineup_synergy"]
            policy_component = opened["policy"]
            gamma_mean = float(opened["lineup_synergy"]["gamma_hat"])
            gamma_variance = float(opened["lineup_synergy"]["gamma_sd"]) ** 2
    posterior_mean = DISPLAY_ANCHOR + roster_component + league_component + DISPLAY_SCALE * gamma_mean
    posterior_sd = DISPLAY_SCALE * math.sqrt(
        roster_variance + league_variance + gamma_variance
    )
    # Fail-closed evidence state (issue #47): the exact ordered five is
    # exact/official/active/fresh by construction, but the v2 Team Rating is
    # development-only, so the honest public state is never "settled".  Any
    # scope or interval problem downgrades the state explicitly.
    interval_width = 2.0 * 1.96 * math.sqrt(max(0.0, roster_variance + league_variance))
    if not all(player.active for player in roster.players):
        evidence_state = "inactive"
    elif interval_width > 200.0 * DISPLAY_SCALE / 400.0:
        evidence_state = "wide_interval"
    elif scope == "global" and league_rating is not None and not league_rating.structurally_eligible:
        evidence_state = "ood"
    else:
        evidence_state = "development_only"
    return TeamRating(
        roster_id=roster.roster_id,
        scope=scope,
        player_ids=roster.player_ids,
        player_names=roster.player_names,
        player_roles=roster.player_roles,
        effective_at=roster.effective_at,
        as_of=roster.as_of,
        roster_receipt_sha256=roster.source_receipt_sha256,
        evidence_state=evidence_state,
        player_posterior_means=player_means,
        roster_latent_mean=roster_mean,
        roster_latent_variance=roster_variance,
        league_latent_mean=league_mean,
        league_latent_variance=league_variance,
        posterior_mean=posterior_mean,
        posterior_interval_95=(
            posterior_mean - 1.96 * posterior_sd,
            posterior_mean + 1.96 * posterior_sd,
        ),
        roster_strength_component=roster_component,
        league_rating_component=league_component,
        lineup_synergy_component=lineup_synergy_component,
        policy_component=policy_component,
        rank_eligibility=False,
        missing_c2_dependency=True,
        development_only=True,
    )


def build_development_candidate(
    config: Mapping[str, Any], fixtures: Mapping[str, Any]
) -> dict[str, Any]:
    config_hash = sha256_json(config)
    fixtures_hash = sha256_json(fixtures)
    report = {
        "artifact_kind": "provisional_l5_mechanics_report",
        "development_only": True,
        "missing_c2_dependency": True,
        "claim_ceiling": dict(CLAIM_CEILING),
        "config_sha256": config_hash,
        "fixtures_sha256": fixtures_hash,
        "deferred_blockers": [
            "resource-policy selection",
            "lineup-synergy identification",
            "mobility/bridge refits",
            "hostile filesystem breadth",
            "coverage authority",
            "external authorization",
        ],
    }
    report_hash = sha256_json(report)
    identity_material = {
        "kind": "provisional_l5_team_mechanics_candidate",
        "config_sha256": config_hash,
        "fixtures_sha256": fixtures_hash,
        "report_sha256": report_hash,
        "missing_c2_dependency": True,
        "development_only": True,
        "authorizing": False,
    }
    return {
        **identity_material,
        "candidate_id": sha256_json(identity_material),
        "report": report,
    }


def verify_development_candidate(
    candidate: Mapping[str, Any],
    config: Mapping[str, Any],
    fixtures: Mapping[str, Any],
) -> None:
    expected = build_development_candidate(config, fixtures)
    if canonical_json(candidate) != canonical_json(expected):
        raise ArtifactIntegrityError("development candidate content identity mismatch")


def load_authorized_bundle(*_: Any, **__: Any) -> None:
    raise AuthorizationError(
        "no independent later registrar authorizes provisional L5 Team Rating"
    )


def load_json(path: Path) -> Any:
    with path.open("rb") as handle:
        return json.load(handle)
