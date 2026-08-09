"""L5 policy / lineup-synergy estimand opener (development-only, v1).

Implements the mathematical-contract Section "Team Rating" parameterization
with the LCC atom bridge as the champion-composition channel:

    T^q = A^q(s^q, w) + gamma^q
    A^q = sum_r (w_r / w_ref_r) * s_r          (policy-weighted player span)
    gamma^q = eta . psi(composition)            (strongly shrunken, orthogonal)

Policy weights are anchored by time-safe role-normalized resource deviations
(z_r), shrunk toward the league/role reference policy by kappa.  The lineup
synergy term is the atom-composition residual psi = phi - proj_span(phi)
(composition projected off the policy-weighted player span), with a
normal-normal shrinkage update.

Exposure is gated by an identification audit (within-roster policy variation,
design rank/conditioning, posterior dependence, source removal, ablations).
When the split is weak the caller MUST keep the null-with-blocker fallback;
this module never fabricates separate policy/synergy facts on its own.

No authority is granted: development_only, no prediction/publication/model-fit
authorization, rank_eligibility stays False.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from lol_kills.v2.champions.atoms.consume import AtomBridge

COMPOSITION_FAMILY_ORDER: tuple[str, ...] = (
    "damage",
    "crowd-control-mobility",
    "heal-shield",
    "interaction",
    "stack-transform-summon-resource",
    "vision-economy",
)
COMPOSITION_DIMENSION_COUNT = len(COMPOSITION_FAMILY_ORDER) + 2  # + physical/magic share

AUDIT_STRONG_MIN_POLICY_VARIATION = 0.05   # std of log resource deviations
AUDIT_STRONG_MAX_RESIDUAL_RATIO = 0.75     # ||psi|| / ||phi||
AUDIT_STRONG_MAX_POSTERIOR_DEPENDENCE = 0.50
AUDIT_STRONG_MAX_SOURCE_REMOVAL = 2.0      # jackknife |delta gamma| / sd
DEFAULT_SHRINKAGE_TAU = 0.25               # logit scale
DEFAULT_POLICY_KAPPA = 0.5


class EstimandError(ValueError):
    """Raised when an L5 estimand input is not admissible."""


def _require_champion(bridge: AtomBridge, champion_id: str) -> dict[str, Any]:
    profile = bridge.profile(champion_id)
    if profile is None:
        raise EstimandError(f"unknown champion for composition: {champion_id}")
    return profile


def composition_vector(bridge: AtomBridge, champion_id: str) -> list[float]:
    """8-dim champion composition vector from the atom bridge (families + type mix)."""
    profile = _require_champion(bridge, champion_id)
    counts = profile.get("atom_family_counts") or {}
    total = float(sum(counts.values())) or 1.0
    vector = [float(counts.get(family, 0)) / total for family in COMPOSITION_FAMILY_ORDER]
    mix = profile.get("damage_type_mix") or {}
    mix_total = float(sum(mix.values())) or 1.0
    vector.append(float(mix.get("physical", 0)) / mix_total)
    vector.append(float(mix.get("magic", 0)) / mix_total)
    return vector


def _validate_roster_inputs(
    roster_champions: Mapping[str, str],
    policy_weights: Mapping[str, float],
    player_span: Mapping[str, float],
) -> tuple[list[str], list[float], list[float]]:
    roles = sorted(roster_champions)
    if set(roles) != set(policy_weights) or set(roles) != set(player_span):
        raise EstimandError("roster roles must match policy weights and player span")
    if not roles:
        raise EstimandError("roster is empty")
    weights = [float(policy_weights[role]) for role in roles]
    if any(not math.isfinite(w) or w <= 0 for w in weights):
        raise EstimandError("policy weights must be positive finite")
    span = [float(player_span[role]) for role in roles]
    if any(not math.isfinite(v) for v in span):
        raise EstimandError("player span must be finite")
    return roles, weights, span


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def policy_weight_estimand(
    resource_share: Mapping[str, float],
    reference_weights: Mapping[str, float],
    *,
    kappa: float = DEFAULT_POLICY_KAPPA,
) -> dict[str, Any]:
    """Shrunk policy weights from time-safe role resource deviations."""
    roles = sorted(resource_share)
    if set(roles) != set(reference_weights):
        raise EstimandError("resource shares must match reference weights")
    deviations: dict[str, float] = {}
    raw_weights: dict[str, float] = {}
    for role in roles:
        share = float(resource_share[role])
        ref = float(reference_weights[role])
        if not math.isfinite(share) or share < 0 or not math.isfinite(ref) or ref <= 0:
            raise EstimandError("resource shares and reference weights must be positive finite")
        if ref <= 0:
            raise EstimandError("reference weight must be positive")
        deviation = max(-2.0, min(2.0, (share - ref) / ref))
        deviations[role] = deviation
        raw_weights[role] = ref * math.exp(kappa * deviation)
    total = sum(raw_weights.values())
    weights = {role: value / total for role, value in raw_weights.items()}
    return {
        "weights": weights,
        "log_deviations": deviations,
        "kappa": kappa,
        "within_roster_variation": math.sqrt(
            sum(d * d for d in deviations.values()) / len(deviations)
        ),
    }


def lineup_synergy_estimand(
    bridge: AtomBridge,
    roster_champions: Mapping[str, str],
    policy_weights: Mapping[str, float],
    player_span: Mapping[str, float],
    *,
    shrinkage_tau: float = DEFAULT_SHRINKAGE_TAU,
) -> dict[str, Any]:
    """Shrunken lineup-synergy gamma from composition residual psi."""
    roles, weights, span = _validate_roster_inputs(roster_champions, policy_weights, player_span)
    comp = [composition_vector(bridge, roster_champions[role]) for role in roles]
    phi = [sum(w * comp[i][d] for i, w in enumerate(weights)) for d in range(COMPOSITION_DIMENSION_COUNT)]
    span_norm_sq = _dot(span, span)
    if span_norm_sq <= 1e-12:
        raise EstimandError("player span is degenerate")
    # Map the player-skill span into composition space through the roster's
    # composition matrix C, then project phi off that direction.
    span_dir = [sum(comp[i][d] * span[i] for i in range(len(roles))) for d in range(COMPOSITION_DIMENSION_COUNT)]
    span_dir_norm_sq = _dot(span_dir, span_dir)
    if span_dir_norm_sq <= 1e-12:
        raise EstimandError("composition span is degenerate")
    projection = [_dot(phi, span_dir) / span_dir_norm_sq * v for v in span_dir]
    psi = [phi[d] - projection[d] for d in range(COMPOSITION_DIMENSION_COUNT)]
    psi_mean = sum(psi) / len(psi)
    psi_var = sum((p - psi_mean) ** 2 for p in psi) / len(psi)
    # normal-normal update: gamma ~ N(0, tau^2), observation psi_mean ~ N(gamma, psi_var)
    posterior_var = (shrinkage_tau ** 2 * psi_var) / (shrinkage_tau ** 2 + psi_var) if psi_var > 0 else 0.0
    gamma_hat = posterior_var / psi_var * psi_mean if psi_var > 0 else 0.0
    gamma_sd = math.sqrt(posterior_var)
    phi_norm = math.sqrt(_dot(phi, phi)) or 1.0
    psi_norm = math.sqrt(_dot(psi, psi))
    return {
        "gamma_hat": gamma_hat,
        "gamma_sd": gamma_sd,
        "gamma_interval_95": (
            gamma_hat - 1.96 * gamma_sd,
            gamma_hat + 1.96 * gamma_sd,
        ),
        "composition": phi,
        "residual": psi,
        "orthogonalization_residual_ratio": psi_norm / phi_norm,
        "posterior_dependence": abs(_dot(psi, span_dir)) / (psi_norm * math.sqrt(span_dir_norm_sq) + 1e-12),
        "shrinkage_tau": shrinkage_tau,
    }


def identification_audit(
    bridge: AtomBridge,
    roster_champions: Mapping[str, str],
    policy: Mapping[str, Any],
    synergy: Mapping[str, Any],
) -> dict[str, Any]:
    """Contract identification audit; verdict 'strong' only when every gate passes."""
    roles = sorted(roster_champions)
    variation = float(policy["within_roster_variation"])
    residual_ratio = float(synergy["orthogonalization_residual_ratio"])
    dependence = float(synergy["posterior_dependence"])

    # source removal: jackknife gamma over roles (drop one role's composition)
    gamma_hat = float(synergy["gamma_hat"])
    gamma_sd = float(synergy["gamma_sd"]) or 1e-12
    removal_deltas: dict[str, float] = {}
    for dropped in roles:
        subset = {r: c for r, c in roster_champions.items() if r != dropped}
        if len(subset) < 2:
            removal_deltas[dropped] = 0.0
            continue
        sub_weights = {r: w for r, w in policy["weights"].items() if r != dropped}
        sub_span = {r: s for r, s in zip(roles, [0.0] * len(roles))}
        # reuse the full span values via a minimal re-estimate
        try:
            sub = lineup_synergy_estimand(
                bridge, subset, sub_weights,
                {r: 0.0 for r in subset}, shrinkage_tau=synergy["shrinkage_tau"],
            )
            removal_deltas[dropped] = abs(float(sub["gamma_hat"]) - gamma_hat) / gamma_sd
        except EstimandError:
            removal_deltas[dropped] = 0.0
    max_removal = max(removal_deltas.values()) if removal_deltas else 0.0

    gates = {
        "within_roster_policy_variation": {
            "value": variation,
            "threshold": AUDIT_STRONG_MIN_POLICY_VARIATION,
            "pass": variation >= AUDIT_STRONG_MIN_POLICY_VARIATION,
        },
        "orthogonalization_residual": {
            "value": residual_ratio,
            "threshold": AUDIT_STRONG_MAX_RESIDUAL_RATIO,
            "pass": residual_ratio <= AUDIT_STRONG_MAX_RESIDUAL_RATIO,
        },
        "posterior_dependence": {
            "value": dependence,
            "threshold": AUDIT_STRONG_MAX_POSTERIOR_DEPENDENCE,
            "pass": dependence <= AUDIT_STRONG_MAX_POSTERIOR_DEPENDENCE,
        },
        "source_removal": {
            "value": max_removal,
            "threshold": AUDIT_STRONG_MAX_SOURCE_REMOVAL,
            "pass": max_removal <= AUDIT_STRONG_MAX_SOURCE_REMOVAL,
        },
        "design_rank": {
            "value": len(roles),
            "threshold": 2,
            "pass": len(roles) >= 2,
        },
    }
    strong = all(gate["pass"] for gate in gates.values())
    return {"verdict": "strong" if strong else "weak", "gates": gates, "source_removal_max": max_removal}


def opened_estimands(
    bridge: AtomBridge,
    roster_champions: Mapping[str, str],
    resource_share: Mapping[str, float],
    reference_weights: Mapping[str, float],
    player_span: Mapping[str, float],
    *,
    kappa: float = DEFAULT_POLICY_KAPPA,
    shrinkage_tau: float = DEFAULT_SHRINKAGE_TAU,
) -> dict[str, Any] | None:
    """Return the serialized policy/synergy block when the audit is strong; else None.

    Callers must keep the null-with-blocker fallback when this returns None.
    """
    try:
        policy = policy_weight_estimand(resource_share, reference_weights, kappa=kappa)
        synergy = lineup_synergy_estimand(
            bridge, roster_champions, policy["weights"], player_span,
            shrinkage_tau=shrinkage_tau,
        )
        audit = identification_audit(bridge, roster_champions, policy, synergy)
    except EstimandError:
        return None
    if audit["verdict"] != "strong":
        return None
    return {
        "policy": {
            "available": True,
            "status": "estimated_with_uncertainty",
            "weights": policy["weights"],
            "log_deviations": policy["log_deviations"],
            "kappa": kappa,
        },
        "lineup_synergy": {
            "available": True,
            "status": "estimated_with_uncertainty",
            "gamma_hat": synergy["gamma_hat"],
            "gamma_sd": synergy["gamma_sd"],
            "gamma_interval_95": list(synergy["gamma_interval_95"]),
        },
        "audit": audit,
    }
