"""Pure, descriptive Draft Score from the frozen interaction artifact.

The serving recipe reads champion, role, ally, cross-side, and same-role
terms. It excludes team controls, player controls, ratings, pace, phase, and
all probability mappings. The result is a model-unit composition edge.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Mapping

from lol_kills.etl.aliases import normalize_champ


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / "data" / "lol" / "models" / "draft_recommendation.json"
MODEL_VERSION = "draft-recommendation-static-v2"
SCHEMA_VERSION = "scryglass:draft-descriptive-signal:v1"
ROLES = ("top", "jng", "mid", "bot", "sup")
CHAMPION_PRIOR_N = 25.0
ROLE_PRIOR_N = 18.0
MIN_SUPPORT_GAMES = 40
ARCHETYPE_INTERACTION_SOURCE = {
    "id": "legacy-manual-archetype-tags-v1",
    "status": "bound_in_artifact",
    "lcc_atoms": "excluded",
    "reason": "LCC-derived atoms remain research-only and are not part of this descriptive release.",
}

INCLUDED_TERMS = (
    "champion_effects",
    "role_conditioned_champion_effects",
    "ally_synergy",
    "archetype_interactions",
    "enemy_counter",
    "same_role",
    "archetype_synergy",
    "archetype_counters",
)
EXCLUDED_TERMS = (
    "team_controls",
    "player_controls",
    "team_rating",
    "player_rating",
    "elo",
    "pace",
    "kill_beta",
    "phase_curve",
    "live_state",
    "outcome_probability",
    "r9e",
    "development_composite",
    "match_probability",
    "match_win_expectation",
    "odds",
    "ev",
)


class DescriptiveDraftScoreError(ValueError):
    """The static descriptive artifact or input is invalid."""


def _pair(first: str, second: str) -> str:
    return "|".join(sorted((first, second), key=str.casefold))


def _number(value: Any, default: float = 0.0) -> float:
    if isinstance(value, bool) or value is None:
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _round_signal(value: float | None) -> float | None:
    if value is None:
        return None
    return float(
        Decimal(str(value)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_EVEN)
    )


def _logit_value(table: Mapping[str, Any], key: str) -> float:
    row = table.get(key)
    return _number(row.get("logit")) if isinstance(row, Mapping) else 0.0


def _antisymmetric_pair_value(table: Mapping[str, Any], first: str, second: str) -> float:
    """Return the oriented pair contribution relative to a side swap."""

    return 0.5 * (
        _logit_value(table, f"{first}|{second}")
        - _logit_value(table, f"{second}|{first}")
    )


def _weighted_effect(
    champion: str,
    role: str,
    model: Mapping[str, Any],
) -> tuple[float, str]:
    counts = model.get("champ_game_counts")
    role_counts = model.get("champion_role_counts")
    if not isinstance(counts, Mapping) or champion not in counts:
        return 0.0, "unavailable"
    champion_weight = _number(counts.get(champion)) / (
        _number(counts.get(champion)) + CHAMPION_PRIOR_N
    )
    champion_effect = _number((model.get("win_delta") or {}).get(champion))
    role_count = 0.0
    if isinstance(role_counts, Mapping) and isinstance(role_counts.get(champion), Mapping):
        role_count = _number(role_counts[champion].get(role))
    role_weight = role_count / (role_count + ROLE_PRIOR_N) if role_count else 0.0
    role_effect = _logit_value(model.get("role_effects") or {}, f"{role}|{champion}")
    evidence_status = "available" if role_count >= MIN_SUPPORT_GAMES else "role_estimate"
    return champion_weight * champion_effect + role_weight * role_effect, evidence_status


def load_model(path: Path = ARTIFACT_PATH) -> tuple[dict[str, Any], str]:
    """Load the frozen artifact and return its raw SHA-256 digest."""

    try:
        raw = path.read_bytes()
        model = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DescriptiveDraftScoreError("static Draft Score artifact is unavailable") from error
    if not isinstance(model, dict):
        raise DescriptiveDraftScoreError("static Draft Score artifact is malformed")
    required = {
        "champ_game_counts",
        "champion_role_counts",
        "win_delta",
        "role_effects",
        "ally_synergy",
        "counter_pairs",
        "role_pairs",
        "archetype_synergy",
        "archetype_counters",
        "champion_archetypes",
    }
    if not required.issubset(model):
        raise DescriptiveDraftScoreError("static Draft Score artifact is incomplete")
    return model, hashlib.sha256(raw).hexdigest()


def _side_picks(game: Mapping[str, Any], side: str) -> dict[str, str]:
    source = game.get(side.casefold())
    if not isinstance(source, Mapping):
        raise DescriptiveDraftScoreError(f"{side} composition is missing")
    result: dict[str, str] = {}
    for role in ROLES:
        pick = source.get(role)
        champion = normalize_champ(str(pick.get("champion") or "").strip()) if isinstance(pick, Mapping) else ""
        if not champion:
            raise DescriptiveDraftScoreError(f"{side} {role} champion is missing")
        result[role] = champion
    if len(set(result.values())) != len(ROLES):
        raise DescriptiveDraftScoreError(f"{side} composition has duplicate champions")
    return result


def score_game(
    game: Mapping[str, Any],
    *,
    model: Mapping[str, Any] | None = None,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Score one complete draft with pure static model terms."""

    if model is None:
        model, artifact_sha256 = load_model()
    blue = _side_picks(game, "Blue")
    red = _side_picks(game, "Red")
    all_champions = [*blue.values(), *red.values()]
    if len(set(all_champions)) != 10:
        raise DescriptiveDraftScoreError("composition has duplicate champions")

    base_by_side: dict[str, dict[str, float]] = {"Blue": {}, "Red": {}}
    evidence_by_side: dict[str, dict[str, str]] = {"Blue": {}, "Red": {}}
    supported = True
    for side, picks in (("Blue", blue), ("Red", red)):
        for role, champion in picks.items():
            value, evidence_status = _weighted_effect(champion, role, model)
            base_by_side[side][role] = value
            evidence_by_side[side][role] = evidence_status
            supported = supported and evidence_status != "unavailable"
    ally_by_side = {"Blue": 0.0, "Red": 0.0}
    archetype_ally_by_side = {"Blue": 0.0, "Red": 0.0}
    ally = model.get("ally_synergy") or {}
    archetype_ally = model.get("archetype_synergy") or {}
    for side, picks in (("Blue", blue), ("Red", red)):
        for first, second in itertools.combinations(picks.values(), 2):
            ally_by_side[side] += _logit_value(ally, _pair(first, second))
            for first_tag in (model.get("champion_archetypes") or {}).get(first, []):
                for second_tag in (model.get("champion_archetypes") or {}).get(second, []):
                    archetype_ally_by_side[side] += _logit_value(
                        archetype_ally,
                        _pair(str(first_tag), str(second_tag)),
                    )

    counter_terms: list[tuple[str, float]] = []
    archetype_counter_terms: list[tuple[str, float]] = []
    counters = model.get("counter_pairs") or {}
    archetype_counters = model.get("archetype_counters") or {}
    champion_archetypes = model.get("champion_archetypes") or {}
    for blue_champion in blue.values():
        for red_champion in red.values():
            counter_terms.append(
                (
                    _pair(blue_champion, red_champion),
                    _antisymmetric_pair_value(counters, blue_champion, red_champion),
                )
            )
            for blue_tag in champion_archetypes.get(blue_champion, []):
                for red_tag in champion_archetypes.get(red_champion, []):
                    first_tag = str(blue_tag)
                    second_tag = str(red_tag)
                    archetype_counter_terms.append(
                        (
                            _pair(first_tag, second_tag),
                            _antisymmetric_pair_value(
                                archetype_counters,
                                first_tag,
                                second_tag,
                            ),
                        )
                    )
    counter_edge = sum(value for _, value in sorted(counter_terms))
    archetype_counter_edge = sum(value for _, value in sorted(archetype_counter_terms))

    role_pairs = model.get("role_pairs") or {}
    same_role_edge = sum(
        _logit_value(role_pairs, f"{role}|{blue[role]}|{red[role]}")
        - _logit_value(role_pairs, f"{role}|{red[role]}|{blue[role]}")
        for role in ROLES
    ) / 2.0

    base_blue = sum(base_by_side["Blue"].values())
    base_red = sum(base_by_side["Red"].values())
    ally_blue = ally_by_side["Blue"]
    ally_red = ally_by_side["Red"]
    edge_components = {
        "base": base_blue - base_red,
        "ally_synergy": ally_blue - ally_red,
        "archetype_interactions": (
            archetype_ally_by_side["Blue"]
            - archetype_ally_by_side["Red"]
            + archetype_counter_edge
        ),
        "enemy_counter": counter_edge,
        "same_role": same_role_edge,
    }
    edge_components["total"] = sum(edge_components.values())
    archetype_cross_blue = archetype_counter_edge / 2.0
    side_components = {
        "blue": {
            "base": base_blue,
            "ally_synergy": ally_blue,
            "archetype_interactions": archetype_ally_by_side["Blue"] + archetype_cross_blue,
            "enemy_counter": counter_edge / 2.0,
            "same_role": same_role_edge / 2.0,
        },
        "red": {
            "base": base_red,
            "ally_synergy": ally_red,
            "archetype_interactions": archetype_ally_by_side["Red"] - archetype_cross_blue,
            "enemy_counter": -counter_edge / 2.0,
            "same_role": -same_role_edge / 2.0,
        },
    }
    picks = []
    for side, side_picks in (("Blue", blue), ("Red", red)):
        for role, champion in side_picks.items():
            contribution = base_by_side[side][role]
            picks.append(
                {
                    "side": side,
                    "role": role,
                    "champion": champion,
                    "contribution": round(contribution, 6)
                    if evidence_by_side[side][role] != "unavailable"
                    else None,
                    "prior_role_games": int(
                        _number((model.get("champion_role_counts") or {}).get(champion, {}).get(role))
                        if isinstance((model.get("champion_role_counts") or {}).get(champion), Mapping)
                        else 0
                    ),
                    "evidence_status": evidence_by_side[side][role],
                }
            )
    status = "available" if supported else "limited"
    if status == "available":
        side_components = {
            side: {
                key: _round_signal(value)
                for key, value in components.items()
            }
            for side, components in side_components.items()
        }
        blue_signal = _round_signal(sum(side_components["blue"].values()))
        red_signal = _round_signal(sum(side_components["red"].values()))
        edge_components = {
            key: _round_signal(
                side_components["blue"][key] - side_components["red"][key]
            )
            for key in side_components["blue"]
        }
        edge_components["total"] = _round_signal(
            sum(edge_components[key] for key in edge_components)
        )
    else:
        blue_signal = None
        red_signal = None
        edge_components = {key: None for key in (*edge_components.keys(),)}
        side_components = {
            "blue": {key: None for key in side_components["blue"]},
            "red": {key: None for key in side_components["red"]},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "authority": "descriptive",
        "estimand": "composition_only",
        "model_version": MODEL_VERSION,
        "artifact_sha256": artifact_sha256,
        "fit_through": None,
        "artifact_as_of": str(model.get("as_of") or "") or None,
        "archetype_interaction_source": dict(ARCHETYPE_INTERACTION_SOURCE),
        "blue": {
            "signal": blue_signal,
            "prior_role_games": sum(
                int(_number((model.get("champion_role_counts") or {}).get(champion, {}).get(role)))
                for role, champion in blue.items()
                if isinstance((model.get("champion_role_counts") or {}).get(champion), Mapping)
            ),
            "components": {
                key: value
                for key, value in side_components["blue"].items()
            },
        },
        "red": {
            "signal": red_signal,
            "prior_role_games": sum(
                int(_number((model.get("champion_role_counts") or {}).get(champion, {}).get(role)))
                for role, champion in red.items()
                if isinstance((model.get("champion_role_counts") or {}).get(champion), Mapping)
            ),
            "components": {
                key: value
                for key, value in side_components["red"].items()
            },
        },
        "edge_components": {
            key: value
            for key, value in edge_components.items()
        },
        "picks": picks,
        "player_comfort": {
            "status": "unavailable",
            "contribution": None,
            "source": None,
            "sha256": None,
            "reason": "No release-bound player familiarity source is available.",
        },
        "note": (
            "Descriptive Draft Score from the frozen champion, role, ally, "
            "enemy-counter, and same-role artifact. Values are model units. "
            "The recipe has no team or player strength controls and publishes no probability."
        ),
    }
