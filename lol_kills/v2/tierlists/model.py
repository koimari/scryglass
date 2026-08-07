"""L9 tier-list core: incremental Tier Value and response-regret counterability.

Development-only foundation for role x league x current-patch tier lists.
This module implements the estimand math from docs/model-v2/estimands.md
("Tier Value and counterability") and the L9 acceptance gates:

- Tier Value is a model-standardized incremental probability-point difference
  (IV = E[100 * (p_D(c) - p_D(c_ref))]), never a raw win rate and never a raw
  logit index.
- Counterability is a nonnegative response-specific lower-tail regret over
  the terminal model's counter-logit distribution per role.  The frozen
  development artifact (terminal-model-neutral-development-v3.json) exposes
  an empty counter-logit distribution, so counterability serializes as
  unavailable (null) and its weight lambda_C is exactly zero; a zero would be
  a fabricated estimate, so we never emit one.
- rank eligibility remains false until independent L2 promotion.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from lol_kills.v2.data.common import ROLES, canonical_json_bytes, parse_rfc3339, sha256_bytes, to_rfc3339

# ---------------------------------------------------------------------------
# Frozen development inputs consumed by this foundation (development-only).
# The terminal candidate is the L7 development artifact selected in the v3
# candidate registry; its claim ceiling explicitly forbids outcome-calibrated
# probability, recommendation, and betting wording.
# ---------------------------------------------------------------------------

TERMINAL_MODEL_ARTIFACT = {
    "locator": "data/lol/v2/models/draft-terminal/terminal-model-neutral-development-v3.json",
    "raw_sha256": "d6ddfbd0238ff5b9d1b259e393b1724717bcb6dd9d7e407d92fc44395da6d6c3",
    "model_version": "draft-terminal-neutral-dev-v3.0.0",
    "model_as_of": "2026-07-18T16:33:48Z",
    "candidate_id": "m0-role-additive",
    "variant_id": "m0-role-additive@ridge-0.05",
    "served_neutral_semantics": "equal_strength_composition_index_not_directly_outcome_calibrated",
}

CROSSWALK_ARTIFACT = {
    "locator": "data/lol/v2/champions/champion-id-crosswalk-v1.json",
    "schema_id": "scryglass.champion-id-crosswalk.v1",
    "artifact_sha256": "59fdf214b570487f64e08f060ba51c82b24e87f3bbe4d6e308fd1bdd42ef14f7",
}

APPEARANCE_SOURCE = {
    "locator": "data/lol/warehouse/parquet/oe_player_games.parquet",
    "source_kind": "oracles_elixir_player_games",
    "availability_approximation": "oe_date_column_used_as_event_end_proxy",
}

SCHEMA_VERSION = "scryglass.tierlist-artifact.v1"
ARTIFACT_KIND = "tier_list_development"

HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PATCH_RE = re.compile(r"^\d{1,2}\.\d{1,2}$")

# Regional scope follows AGENTS.md: Europe = LEC only, Americas = LCS only.
REGIONS: dict[str, tuple[str, ...]] = {
    "europe": ("LEC",),
    "americas": ("LCS",),
    "asia": ("LCK", "LPL", "PCS", "VCS", "LJL"),
    "international": ("MSI", "EWC", "WORLDS"),
}
INTERNATIONAL_SCOPES = frozenset(("MSI", "EWC", "WORLDS"))
COMPETITION_TIERS = ("tier1", "tier2", "tier3")

COUNTERABILITY_TAIL_ALPHA = 0.25  # dev convention; frozen by L2 under R-10 before any weight
COUNTERABILITY_WEIGHT_LAMBDA_C = 0.0
COUNTERABILITY_WEIGHT_SELECTION = "unavailable_no_l2_validation"
REFERENCE_MIXTURE_RULE = "equal_weight_played_membership"

CLAIM_CEILING = {
    "rank_eligibility": False,
    "outcome_calibrated_probability": False,
    "recommendation": False,
    "betting": False,
    "causal_draft_effect": False,
    "publication": False,
    "production": False,
    "reliability": False,
    "independent_validation": False,
}

# Reproducibility allowlist for the source-tree digest (contract: files that
# can influence the build; tests and generated artifacts are excluded).
SOURCE_TREE_ALLOWLIST = (
    "lol_kills/v2/tierlists/__init__.py",
    "lol_kills/v2/tierlists/schema.py",
    "lol_kills/v2/tierlists/model.py",
    "lol_kills/v2/tierlists/appearances.py",
    "lol_kills/v2/tierlists/artifact.py",
    "lol_kills/v2/tierlists/generate_artifacts.py",
)

_FORBIDDEN_RAW_WIN_RATE_KEYS = frozenset(("win_rate", "raw_win_rate", "wr", "wins", "losses"))


class TierListError(ValueError):
    """Raised when tier-list inputs or payloads are not admissible."""


class TierListIntegrityError(TierListError):
    """Raised when a persisted artifact no longer matches its content identity."""


class TierListUnavailable(TierListError):
    """Raised when the requested cell is structurally unavailable."""


# ---------------------------------------------------------------------------
# Estimand math
# ---------------------------------------------------------------------------


def _sigmoid(logit: float) -> float:
    if logit >= 40.0:
        return 1.0
    if logit <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-logit))


def calibrated_probability(
    logit: float,
    *,
    calibration_slope: float,
    calibration_intercept: float,
) -> float:
    """Registered terminal calibration transform g_T for the frozen artifact.

    The v3 development artifact uses calibration_intercept = 0 and
    calibration_slope = 0.8 (symmetric temperature 1.25).  The transform is
    monotone and complement-symmetric; here it maps a composition logit to
    the equal-strength composition probability used inside IV.
    """
    return _sigmoid(calibration_intercept + calibration_slope * logit)


def standardized_replacement_probability_points(
    champion_logit: float,
    reference_logit: float,
    *,
    calibration_slope: float,
    calibration_intercept: float,
) -> float:
    """Delta_c for the degenerate dev context: 100 * (g_T(beta_c) - g_T(beta_ref)).

    This is the model-standardized incremental value in probability points
    (pp).  It is an equal-strength composition index, not a raw win rate and
    not an outcome-calibrated probability.
    """
    return 100.0 * (
        calibrated_probability(champion_logit, calibration_slope=calibration_slope, calibration_intercept=calibration_intercept)
        - calibrated_probability(reference_logit, calibration_slope=calibration_slope, calibration_intercept=calibration_intercept)
    )


def reference_mixture_logit(member_logits: Sequence[float]) -> float:
    """Equal-weight role-patch reference mixture over played membership."""
    if not member_logits:
        raise TierListError("reference mixture requires at least one member")
    return float(sum(member_logits) / len(member_logits))


def _pair(first: str, second: str) -> str:
    return "|".join(sorted((first, second)))


def counter_effect(role: str, first: str, second: str, counter_logit: Mapping[str, float]) -> float:
    """Signed response contribution, mirroring the terminal estimator."""
    value = counter_logit.get(f"{role}|{_pair(first, second)}", 0.0)
    return value if first <= second else -value


def context_ally_contribution(champion: str, allies: Sequence[str], ally_synergy_logit: Mapping[str, float]) -> float:
    """Ally-term contribution of one champion inside a frozen allied context z."""
    return float(sum(ally_synergy_logit.get(_pair(champion, ally), 0.0) for ally in allies))


def _type7_quantile(sorted_values: Sequence[float], alpha: float) -> float:
    """Numpy-compatible type-7 linear-interpolation quantile (deterministic)."""
    if not sorted_values:
        raise TierListError("quantile requires a non-empty sample")
    values = sorted(sorted_values)
    if len(values) == 1:
        return values[0]
    h = (len(values) - 1) * alpha
    lower = int(math.floor(h))
    upper = int(math.ceil(h))
    if lower == upper:
        return values[lower]
    return values[lower] + (h - lower) * (values[upper] - values[lower])


def response_regret(
    *,
    role: str,
    champion: str,
    champion_logit: float,
    reference_logit: float,
    member_logits: Mapping[str, float],
    counter_logit: Mapping[str, float],
    allies: Sequence[str] = (),
    ally_synergy_logit: Mapping[str, float] | None = None,
    tail_alpha: float = COUNTERABILITY_TAIL_ALPHA,
    calibration_slope: float,
    calibration_intercept: float,
) -> dict[str, Any] | None:
    """Nonnegative response-specific lower-tail regret over plausible responses.

    R_plaus and R_ref (dev conventions, frozen under R-10 by L2): uniform over
    the role's plausible-response support derived from the terminal model's
    counter-logit keys.  Returns None (unavailable) when the distribution is
    empty: an empty distribution carries no response-specific information, so
    a zero regret would be fabricated.
    """
    if not 0.0 < tail_alpha < 1.0:
        raise TierListError("tail alpha must be in (0, 1)")
    synergy = ally_synergy_logit or {}
    role_support = sorted(
        {
            name
            for key in counter_logit
            if isinstance(key, str) and key.startswith(f"{role}|")
            for name in key.split("|")[1:]
            if name != champion
        }
    )
    if not role_support:
        return None
    # The reference mixture is the frozen cell-level mixture over ALL played
    # members (common across champions), including the evaluated champion.
    members = [name for name in member_logits]
    deltas: list[float] = []
    for opponent in role_support:
        champion_side = champion_logit + context_ally_contribution(champion, allies, synergy) + counter_effect(role, champion, opponent, counter_logit)
        reference_side = reference_logit + float(
            sum(context_ally_contribution(member, allies, synergy) for member in members) / len(members)
        ) + float(
            sum(counter_effect(role, member, opponent, counter_logit) for member in members) / len(members)
        )
        delta = 100.0 * (
            calibrated_probability(champion_side, calibration_slope=calibration_slope, calibration_intercept=calibration_intercept)
            - calibrated_probability(reference_side, calibration_slope=calibration_slope, calibration_intercept=calibration_intercept)
        )
        deltas.append(delta)
    mean_delta = float(sum(deltas) / len(deltas))
    lower_tail = _type7_quantile(deltas, tail_alpha)
    regret = max(0.0, mean_delta - lower_tail)
    return {
        "regret": regret,
        "nonnegative": True,
        "mean_delta": mean_delta,
        "lower_tail_quantile": lower_tail,
        "tail_alpha": tail_alpha,
        "support_size": len(role_support),
        "support": role_support,
        "response_policy": "uniform_over_role_counter_support_dev_convention",
    }


# ---------------------------------------------------------------------------
# Crosswalk consumption (vocabulary only; L1 owns full replay)
# ---------------------------------------------------------------------------


def load_crosswalk_vocabulary(root: Path) -> dict[str, str]:
    """Load champion identity vocabulary from the frozen crosswalk artifact.

    The artifact carries its own embedded digest; this loader verifies the
    canonical payload and the embedded sha256.  The strict L1 source-replay
    (maps/players parquet byte pins) is currently unavailable because the
    warehouse has been refreshed since the crosswalk was built; that stronger
    replay is L1-owned and recorded as such in lineage.
    """
    path = root / CROSSWALK_ARTIFACT["locator"]
    if not path.is_file() or path.is_symlink():
        raise TierListError("champion crosswalk artifact is missing or not a regular file")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TierListError("champion crosswalk artifact is not strict UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise TierListError("champion crosswalk artifact must be a JSON object")
    if payload.get("schema_id") != CROSSWALK_ARTIFACT["schema_id"]:
        raise TierListError("champion crosswalk schema_id mismatch")
    submitted = payload.get("artifact_sha256")
    if not isinstance(submitted, str) or not HASH_RE.fullmatch(submitted):
        raise TierListError("champion crosswalk artifact_sha256 is invalid")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    if hashlib.sha256(canonical_json_bytes(unsigned)).hexdigest() != submitted:
        raise TierListError("champion crosswalk artifact_sha256 does not match canonical payload")
    if submitted != CROSSWALK_ARTIFACT["artifact_sha256"]:
        raise TierListError("champion crosswalk artifact is not the frozen vocabulary artifact")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise TierListError("champion crosswalk entries are missing")
    table: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise TierListError("champion crosswalk entry is malformed")
        normalized = entry.get("normalized_oe_name")
        stable_id = entry.get("stable_champion_id")
        if not isinstance(normalized, str) or not normalized or not isinstance(stable_id, str) or not stable_id:
            raise TierListError("champion crosswalk entry identity is malformed")
        if stable_id in table.values() and table.get(normalized) != stable_id:
            raise TierListError("champion crosswalk stable-id collision")
        prior = table.get(normalized)
        if prior is not None and prior != stable_id:
            raise TierListError("champion crosswalk normalized-name collision")
        table[normalized] = stable_id
    return table
