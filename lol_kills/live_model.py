"""Preliminary live-game evaluation from a GRID Series State snapshot.

This is deliberately separate from Dual Elo updates.  It consumes the
published, time-safe live coefficient artifact and emits a provisional
probability with field provenance.  It does not mutate ratings or claim that a
partial feed is a completed match.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from lol_kills.draft_score import draft_score

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_COEFS = ROOT / "data" / "lol" / "models" / "draft_live_coefs.json"
LIVE_MINUTE_START = 8.0
LIVE_MINUTE_END = 20.0


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _team_side(team: Mapping[str, Any]) -> str | None:
    side = str(team.get("side") or "").strip().lower()
    if side in {"blue", "red"}:
        return side
    return None


def _active_game(series_state: Mapping[str, Any]) -> Mapping[str, Any] | None:
    games = [game for game in series_state.get("games") or [] if isinstance(game, Mapping)]
    if not games:
        return None
    active = [game for game in games if game.get("finished") is not True]
    return active[-1] if active else games[-1]


def _teams(game: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for team in game.get("teams") or []:
        if not isinstance(team, Mapping):
            continue
        side = _team_side(team)
        if side:
            result[side] = team
    return result


def _clock_minutes(game: Mapping[str, Any]) -> float | None:
    clock = game.get("clock")
    if isinstance(clock, Mapping):
        for key in ("currentSeconds", "elapsedSeconds"):
            value = _number(clock.get(key))
            if value is not None and value >= 0:
                return value / 60.0
    for key in ("currentSeconds", "elapsedSeconds", "gameTime"):
        value = _number(game.get(key))
        if value is not None and value >= 0:
            # GRID gameTime is milliseconds.  Its unit is a field contract,
            # never something inferred from the observed magnitude.
            return value / (60_000.0 if key == "gameTime" else 60.0)
    return None


def _team_gold(team: Mapping[str, Any]) -> tuple[float | None, str | None]:
    players = [player for player in team.get("players") or [] if isinstance(player, Mapping)]
    earned = [_number(player.get("totalMoneyEarned")) for player in players]
    earned = [value for value in earned if value is not None]
    if len(earned) == len(players) and earned and sum(earned) > 0:
        return sum(earned), "players.totalMoneyEarned"
    net_worth = _number(team.get("netWorth"))
    if net_worth is not None and net_worth > 0:
        return net_worth, "team.netWorth"
    money = _number(team.get("money"))
    if money is not None and money > 0:
        return money, "team.money"
    return None, None


def _objective_count(team: Mapping[str, Any], needle: str) -> float:
    total = 0.0
    objectives = team.get("objectives") or []
    if isinstance(objectives, Mapping):
        objectives = [objectives]
    for objective in objectives:
        if not isinstance(objective, Mapping):
            continue
        kind = str(objective.get("type") or objective.get("name") or "").lower()
        if needle not in kind:
            continue
        count = _number(objective.get("completionCount"))
        total += count if count is not None else 1.0
    return total


def _objective_diff(teams: Mapping[str, Mapping[str, Any]], needle: str) -> float:
    return _objective_count(teams.get("blue", {}), needle) - _objective_count(teams.get("red", {}), needle)


_ROLE_ORDER = ("top", "jng", "mid", "bot", "sup")
_ROLE_ALIASES = {
    "top": "top",
    "jungle": "jng",
    "jng": "jng",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "support": "sup",
    "sup": "sup",
    "utility": "sup",
}


def _player_champion(player: Mapping[str, Any]) -> str:
    value = (
        player.get("champion")
        or player.get("character")
        or player.get("championName")
    )
    if isinstance(value, Mapping):
        value = value.get("name") or value.get("displayName")
    return str(value or "").strip()


def _role_assigned_picks(
    teams: Mapping[str, Mapping[str, Any]],
) -> tuple[list[str], list[str]]:
    """Return Top/Jungle/Mid/Bot/Support picks only from explicit role data."""

    output: dict[str, list[str]] = {"blue": [], "red": []}
    for side in ("blue", "red"):
        by_role: dict[str, str] = {}
        for player in teams.get(side, {}).get("players") or []:
            if not isinstance(player, Mapping):
                continue
            raw_role = str(
                player.get("role")
                or player.get("position")
                or player.get("lane")
                or ""
            ).strip().lower()
            role = _ROLE_ALIASES.get(raw_role)
            champion = _player_champion(player)
            if not role or not champion or role in by_role:
                continue
            by_role[role] = champion
        picks = [by_role.get(role, "") for role in _ROLE_ORDER]
        if all(picks) and len({pick.casefold() for pick in picks}) == 5:
            output[side] = picks
    return output["blue"], output["red"]


def _load_coefficients(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("live coefficient artifact unavailable") from exc
    if not isinstance(value, dict) or not isinstance(value.get("phase_coefs"), Mapping):
        raise RuntimeError("live coefficient artifact is not a supported version")
    gate = value.get("publication_gate")
    antisymmetry = value.get("antisymmetry_validation")
    holdout = value.get("chronological_holdout")
    if (
        not isinstance(gate, Mapping)
        or gate.get("status") != "passed"
        or not isinstance(antisymmetry, Mapping)
        or antisymmetry.get("passed") is not True
        or _number(antisymmetry.get("max_probability_complement_error")) is None
        or float(antisymmetry["max_probability_complement_error"]) > 1e-10
        or not isinstance(holdout, Mapping)
        or holdout.get("fit_before_holdout") is not True
        or int(_number(holdout.get("maps")) or 0) <= 0
    ):
        raise RuntimeError(
            "live coefficient artifact lacks a passing chronological and "
            "side-swap publication gate"
        )
    return value


@dataclass(frozen=True)
class LiveEvaluation:
    """A transparent, preliminary live-game estimate."""

    status: str
    model: str
    phase: str
    minute: float | None
    p_blue: float | None
    p_red: float | None
    blue_team: str | None
    red_team: str | None
    draft_status: str
    strength_status: str
    features: dict[str, float | None]
    feature_sources: dict[str, str | None]
    missing: tuple[str, ...]
    warnings: tuple[str, ...]
    contributions: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "model": self.model,
            "phase": self.phase,
            "minute": self.minute,
            "p_blue": self.p_blue,
            "p_red": self.p_red,
            "blue_team": self.blue_team,
            "red_team": self.red_team,
            "draft_status": self.draft_status,
            "strength_status": self.strength_status,
            "features": self.features,
            "feature_sources": self.feature_sources,
            "missing": list(self.missing),
            "warnings": list(self.warnings),
            "contributions": self.contributions,
        }


def evaluate_live_state(
    series_state: Mapping[str, Any],
    *,
    elo_diff: float | None = None,
    coefficients_path: Path = DEFAULT_COEFS,
) -> LiveEvaluation:
    """Evaluate the latest GRID state, failing closed when no game is usable."""
    game = _active_game(series_state)
    if game is None:
        return LiveEvaluation(
            status="unavailable",
            model="draft-live-v2",
            phase="unknown",
            minute=None,
            p_blue=None,
            p_red=None,
            blue_team=None,
            red_team=None,
            draft_status="unavailable",
            strength_status="unavailable",
            features={},
            feature_sources={},
            missing=("active_game",),
            warnings=("GRID has not supplied a usable game state yet.",),
            contributions=[],
        )

    teams = _teams(game)
    blue = teams.get("blue")
    red = teams.get("red")
    if blue is None or red is None:
        return LiveEvaluation(
            status="unavailable",
            model="draft-live-v2",
            phase="unknown",
            minute=_clock_minutes(game),
            p_blue=None,
            p_red=None,
            blue_team=str(blue.get("name")) if blue else None,
            red_team=str(red.get("name")) if red else None,
            draft_status="unavailable",
            strength_status="unavailable",
            features={},
            feature_sources={},
            missing=("blue_and_red_teams",),
            warnings=("GRID state did not include both sides; no probability was emitted.",),
            contributions=[],
        )

    minute = _clock_minutes(game)
    phase = "early" if minute is None or minute < 12 else "mid" if minute < 20 else "late"
    try:
        coefs = _load_coefficients(coefficients_path)
    except RuntimeError as exc:
        return LiveEvaluation(
            status="unavailable-unvalidated-model",
            model="withheld",
            phase=phase,
            minute=minute,
            p_blue=None,
            p_red=None,
            blue_team=str(blue.get("name") or "Blue"),
            red_team=str(red.get("name") or "Red"),
            draft_status="unavailable",
            strength_status="unavailable",
            features={},
            feature_sources={},
            missing=("validated_live_model",),
            warnings=(
                "Live game state is available, but probability is withheld.",
                str(exc),
            ),
            contributions=[],
        )
    phase_coef = coefs.get("phase_coefs", {}).get(phase) or {}
    blue_gold, blue_gold_source = _team_gold(blue)
    red_gold, red_gold_source = _team_gold(red)
    gold_k = (blue_gold - red_gold) / 1000.0 if blue_gold is not None and red_gold is not None else None
    dragon_diff = _objective_diff(teams, "dragon")
    herald_diff = _objective_diff(teams, "herald")
    tower_diff = _objective_diff(teams, "tower") + _objective_diff(teams, "structure")
    blue_picks, red_picks = _role_assigned_picks(teams)

    draft_edge: float | None = None
    draft_status = "incomplete"
    if len(blue_picks) == 5 and len(red_picks) == 5:
        draft = draft_score(blue_picks, red_picks, league=None, elo_diff=None)
        draft_edge = float((draft.get("components") or {}).get("win_edge"))
        draft_status = "complete"

    # The concentration/scaling features are not inferred from absent GRID
    # fields. They remain neutral until a complete player-level state supports
    # them, and the missing list makes that limitation public to the caller.
    conc_diff = 0.0
    scaling = 0.0
    carry = 0.0
    features: dict[str, float | None] = {
        "elo_z": elo_diff / 400.0 if elo_diff is not None else None,
        "draft_edge": draft_edge,
        "gold_k": gold_k,
        "first_dragon": 1.0 if dragon_diff > 0 else -1.0 if dragon_diff < 0 else 0.0,
        "first_herald": 1.0 if herald_diff > 0 else -1.0 if herald_diff < 0 else 0.0,
        "first_tower": 1.0 if tower_diff > 0 else -1.0 if tower_diff < 0 else 0.0,
        "draft_x_gold": draft_edge * gold_k if draft_edge is not None and gold_k is not None else None,
        "conc_x_gold": conc_diff * gold_k if gold_k is not None else None,
        "scaling_x_gold": scaling * gold_k if gold_k is not None else None,
        "blue_carry_x_gold": carry * gold_k if gold_k is not None else None,
    }
    sources = {
        "elo_z": "caller.team_rating_gap" if elo_diff is not None else None,
        "draft_edge": "GRID.game.draftActions + Draft Score v3" if draft_edge is not None else None,
        "gold_k": f"GRID.{blue_gold_source} − GRID.{red_gold_source} / 1000" if gold_k is not None else None,
        "first_dragon": "GRID.game.teams[].objectives",
        "first_herald": "GRID.game.teams[].objectives",
        "first_tower": "GRID.game.teams[].objectives",
        "draft_x_gold": "derived",
        "conc_x_gold": None,
        "scaling_x_gold": None,
        "blue_carry_x_gold": None,
    }
    missing = tuple(
        name
        for name, value in features.items()
        if value is None or name in {"conc_x_gold", "scaling_x_gold", "blue_carry_x_gold"}
    )
    warnings: list[str] = [
        "Preliminary live estimate; it is not a finalized rating update.",
        "GRID live state is broadcast-synchronized and may differ from public scoreboard totals.",
        "The coefficient artifact is fit to approximately 10:00 and 15:00 game checkpoints.",
    ]
    if draft_status != "complete":
        warnings.append("Draft is incomplete; draft contribution is withheld rather than guessed.")
    if elo_diff is None:
        warnings.append("Pre-match team-strength gap was not supplied; strength contribution is withheld.")
    if blue_gold_source == "team.netWorth" or red_gold_source == "team.netWorth":
        warnings.append("Gold proxy uses GRID netWorth because totalMoneyEarned was unavailable.")

    contributions: list[dict[str, Any]] = []
    feature_labels = {
        "elo_z": "Pre-match team strength",
        "draft_edge": "Draft composition",
        "gold_k": "Current gold",
        "first_dragon": "Dragon control",
        "first_herald": "Herald control",
        "first_tower": "Tower control",
        "draft_x_gold": "Draft × current gold",
        "conc_x_gold": "Strength concentration × gold",
        "scaling_x_gold": "Scaling × gold",
        "blue_carry_x_gold": "Carry concentration × gold",
    }

    if minute is None:
        p_blue = None
        p_red = None
        status = "preliminary-incomplete"
        missing = tuple(dict.fromkeys((*missing, "clock")))
        warnings.append("Game clock is missing; live probability is withheld.")
    elif minute < LIVE_MINUTE_START or minute >= LIVE_MINUTE_END:
        p_blue = None
        p_red = None
        status = "preliminary-out-of-calibration"
        missing = tuple(dict.fromkeys((*missing, "calibration_window")))
        warnings.append(
            f"Live probability is withheld outside the calibrated window ({LIVE_MINUTE_START:.0f}:00–{LIVE_MINUTE_END:.0f}:00)."
        )
    elif any(features[name] is None for name in ("gold_k", "elo_z")):
        p_blue = None
        p_red = None
        status = "preliminary-incomplete"
    else:
        logit = float(phase_coef.get("intercept") or 0.0)
        for name, value in features.items():
            if value is None:
                value = 0.0
            logit += float(phase_coef.get(name) or 0.0) * float(value)
        p_blue = round(max(0.005, min(0.995, _sigmoid(logit))), 4)
        p_red = round(1.0 - p_blue, 4)
        status = "preliminary"
        for name, value in features.items():
            coefficient = float(phase_coef.get(name) or 0.0)
            if value is None or coefficient == 0.0:
                continue
            without_feature = _sigmoid(logit - coefficient * float(value))
            delta_pp = round((p_blue - without_feature) * 100.0, 2)
            contributions.append(
                {
                    "key": name,
                    "label": feature_labels.get(name, name),
                    "delta_pp": delta_pp,
                    "value": round(float(value), 4),
                    "source": sources.get(name),
                }
            )
        contributions.sort(key=lambda row: abs(float(row["delta_pp"])), reverse=True)

    return LiveEvaluation(
        status=status,
        model=f"draft-live-v{coefs.get('version', 'unknown')}",
        phase=phase,
        minute=minute,
        p_blue=p_blue,
        p_red=p_red,
        blue_team=str(blue.get("name") or "Blue"),
        red_team=str(red.get("name") or "Red"),
        draft_status=draft_status,
        strength_status="complete" if elo_diff is not None else "incomplete",
        features=features,
        feature_sources=sources,
        missing=missing,
        warnings=tuple(warnings),
        contributions=contributions,
    )
