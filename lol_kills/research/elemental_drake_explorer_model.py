#!/usr/bin/env python3
"""Fit the joint elemental-drake state and resolved-allocation explorer models.

The two public estimands are deliberately narrower than a strategic objective
policy:

* ``jointState`` estimates the adjusted association between the complete
  post-capture inventories (including soul) and map win.
* ``captureAllocation`` estimates the adjusted association between which side
  received an already-resolved elemental capture and map win.

Neither estimand identifies the causal value of starting, contesting, taking,
or leaving an objective.  Every usable capture contributes two mirrored team
perspectives, and every game receives equal total fitting/evaluation weight so
long games with many captures do not dominate the models.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.linalg import null_space
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.draft_archetypes import ARCHETYPE_NAMES, champ_tags
from lol_kills.draft_recommendation import CURRENT_CHAMPIONS
from lol_kills.etl.aliases import normalize_champ
from lol_kills.research.elemental_drake_model import (
    ELEMENTS,
    HOLDOUT_FRACTION,
    MIN_PUBLIC_GAMES,
    PUBLIC_HOLDOUT_END,
    PUBLIC_HOLDOUT_START,
    RATING_BASE,
    STANDARDIZED_CLIP,
    FittedLogit,
    _fit,
    _json_list,
    pregame_strengths,
)
from lol_kills.research.elemental_drakes import (
    COMPACT_EVENTS_PARQUET,
    COMPACT_GAMES_PARQUET,
)

ROOT = Path(__file__).resolve().parents[2]
CHAMPION_TAXONOMY_PATH = ROOT / "lol_kills" / "draft_archetypes.py"
DEFAULT_OUTPUT = (
    ROOT / "data" / "lol" / "models" / "elemental_drake_explorer_model.json"
)
SCHEMA_VERSION = "elemental-drake-explorer-v4"
FEATURE_BUILDER_VERSION = "joint-drake-signed-v2"
ALPHA_GRID = (0.03, 0.1, 0.3, 1.0)
MAX_STATE_LAG_SECONDS = 60.0
MAX_STACKS = 4
MAX_GLOBAL_STAGE = 7
DIRECT_FAMILY = "champion-direct-inventory-residual"
DIRECT_MIN_GAME_GRID = (50, 75, 100)
DIRECT_MIN_SERIES = 25
DIRECT_MIN_OWNERSHIP_GAMES = 20
DIRECT_MIN_NONOWNERSHIP_GAMES = 20
DIRECT_MIN_ORGS = 3
DIRECT_LAMBDA_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)
DIRECT_GATE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
DIRECT_MAX_BRIER_REGRESSION = 0.0025
DIRECT_MAX_LOG_LOSS_REGRESSION = 0.01
DIRECT_MAX_ECE_REGRESSION = 0.02
DIRECT_MAX_CONSTRAINT_VIOLATION = 1e-8
PUBLICATION_AUDIT_START = pd.Timestamp("2026-07-01T00:00:00Z")
PUBLICATION_AUDIT_END = pd.Timestamp("2026-08-01T00:00:00Z")
PUBLICATION_AUDIT_MIN_GAMES = 500
PUBLICATION_AUDIT_MIN_SERIES = 200

STATE_NUMERIC = (
    "gold_diff_k",
    "loadout_diff_k",
    "unspent_money_diff_k",
    "top_player_net_worth_diff_k",
    "tower_diff",
    "org_elo_diff",
    "player_elo_diff",
)

PUBLIC_WORDING = {
    "champions": (
        "A supported champion-element cell may use an exploratory champion "
        "estimate only when the model family clears both its "
        "March-April evaluation and a whole-series July publication-expansion "
        "audit. Eligibility uses exposure counts only; outcomes are retained as "
        "diagnostics and never decide whether a cell enters the vocabulary. The "
        "champion estimate is partially pooled toward the archetype prior; it is "
        "not a second archetype bonus. Unsupported tagged cells use the archetype "
        "prior only, while untagged unsupported cells have no modeled champion "
        "differential; the common dragon effect remains team-level. This is not "
        "an exact-composition lookup."
    ),
    "lines": (
        "Individual champion lines are reconciled allocations of the modeled "
        "team effect. A partially pooled champion estimate can change that "
        "allocation when supported, but the lines are not champion win rates, "
        "personal-stat conversions, or causal champion-by-dragon responses."
    ),
    "allocation": (
        "Capture allocation is selected and not randomized. It describes which "
        "side received an already-resolved capture; it is not a strategic leave "
        "policy and must not be read as a causal take-versus-leave estimate."
    ),
    "draftContext": (
        "Generic Draft Score champion, ally-synergy, and enemy-counter "
        "coefficients are not applied to the dragon estimate. Those coefficients "
        "measure a different pre-match composition estimand and do not establish "
        "champion-by-dragon response."
    ),
    "directResidual": (
        "The champion-estimate layer is exploratory because its March-April "
        "2026 evaluation was added after the base holdout protocol was specified. "
        "A second July 2026 audit tests the expanded pre-July vocabulary before "
        "the final full-cohort refit. These are family-level checks, not claims "
        "that each champion-element coefficient was independently validated. "
        "The family remains hidden when either audit is materially worse than "
        "its frozen base model."
    ),
}


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _file_provenance(path: Path) -> dict[str, Any]:
    """Content-address an exact model input without exposing its local path."""
    digest = hashlib.sha256()
    size = 0
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return {
        "fileName": Path(path).name,
        "sha256": digest.hexdigest(),
        "bytes": int(size),
    }


def _identifier_set_provenance(values: Sequence[Any]) -> dict[str, Any]:
    """Hash a sorted identifier set without serializing provider identifiers."""
    identifiers = sorted({str(value) for value in values})
    payload = "\n".join(identifiers).encode("utf-8")
    return {
        "count": int(len(identifiers)),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "encoding": "sorted unique UTF-8 identifiers joined by newline",
    }


def _winner_is_valid(game: Mapping[str, Any]) -> bool:
    team_ids = {
        str(game.get("team_1_id") or ""),
        str(game.get("team_2_id") or ""),
    }
    return (
        bool(game.get("complete"))
        and "" not in team_ids
        and len(team_ids) == 2
        and str(game.get("winner_team_id") or "") in team_ids
    )


def _side_assignment_is_valid(game: Mapping[str, Any]) -> bool:
    return {
        str(game.get("team_1_side") or "").casefold(),
        str(game.get("team_2_side") or "").casefold(),
    } == {"blue", "red"}


def _composition_counts(champions: Sequence[str]) -> dict[str, float]:
    counts = {tag: 0.0 for tag in ARCHETYPE_NAMES}
    for champion in champions:
        for tag in champ_tags(normalize_champ(str(champion))):
            counts[tag] += 1.0
    return counts


def _inventory_total(
    inventory: Mapping[str, Mapping[str, int]],
    team_id: str,
) -> int:
    return int(sum(int(inventory[team_id][element]) for element in ELEMENTS))


def _inventory_is_legal(
    inventory: Mapping[str, Mapping[str, int]],
    team_ids: Sequence[str],
) -> bool:
    totals = []
    for team_id in team_ids:
        counts = [int(inventory[team_id][element]) for element in ELEMENTS]
        if any(count < 0 or count > MAX_STACKS for count in counts):
            return False
        totals.append(sum(counts))
    return all(0 <= total <= MAX_STACKS for total in totals) and sum(totals) <= (
        MAX_GLOBAL_STAGE
    )


def _capture_path_is_legal(
    events: pd.DataFrame,
    team_ids: Sequence[str],
) -> bool:
    """Validate the ordered elemental path before admitting any game rows."""
    inventory = {
        team_id: {element: 0 for element in ELEMENTS} for team_id in team_ids
    }
    opening: list[str] = []
    rift: str | None = None
    for expected_stage, event in enumerate(
        events.to_dict(orient="records"),
        start=1,
    ):
        owner = str(event.get("owner_team_id") or "")
        element = str(event.get("element") or "")
        if (
            expected_stage > MAX_GLOBAL_STAGE
            or owner not in team_ids
            or element not in ELEMENTS
            or any(
                _inventory_total(inventory, team_id) == MAX_STACKS
                for team_id in team_ids
            )
        ):
            return False
        observed_stage = int(_finite(event.get("global_index"), expected_stage))
        if observed_stage != expected_stage:
            return False
        if expected_stage <= 2:
            if element in opening:
                return False
            opening.append(element)
        elif expected_stage == 3:
            if element in opening:
                return False
            rift = element
        elif element != rift:
            return False
        inventory[owner][element] += 1
        if not _inventory_is_legal(inventory, team_ids):
            return False
        observed_owner_stack = int(
            _finite(
                event.get("owner_stack"),
                _inventory_total(inventory, owner),
            )
        )
        if observed_owner_stack != _inventory_total(inventory, owner):
            return False
    return bool(opening)


def _valid_capture_state(event: Mapping[str, Any]) -> bool:
    return (
        str(event.get("state_timing") or "") == "previous-envelope"
        and 0 <= _finite(event.get("state_lag_seconds")) <= MAX_STATE_LAG_SECONDS
        and _finite(event.get("owner_net_worth")) > 0
        and _finite(event.get("opponent_net_worth")) > 0
        and 4.5 <= _finite(event.get("time_seconds")) / 60.0 <= 50.0
        and abs(_finite(event.get("gold_diff"))) <= 15_000
    )


def _side_prefix(game: Mapping[str, Any], team_id: str) -> str:
    if team_id == str(game.get("team_1_id") or ""):
        return "team_1"
    if team_id == str(game.get("team_2_id") or ""):
        return "team_2"
    raise ValueError(f"Team {team_id!r} is not in the game.")


def _signed_state(event: Mapping[str, Any], took_current: bool) -> dict[str, float]:
    sign = 1.0 if took_current else -1.0
    return {
        "gold_diff_k": sign * _finite(event.get("gold_diff")) / 1_000.0,
        "loadout_diff_k": sign
        * _finite(event.get("loadout_diff"))
        / 1_000.0,
        "unspent_money_diff_k": sign
        * _finite(event.get("unspent_money_diff"))
        / 1_000.0,
        "top_player_net_worth_diff_k": sign
        * _finite(event.get("top_player_net_worth_diff"))
        / 1_000.0,
        "kill_diff": sign
        * (
            _finite(event.get("owner_kills"))
            - _finite(event.get("opponent_kills"))
        ),
        "tower_diff": sign
        * (
            _finite(event.get("owner_towers"))
            - _finite(event.get("opponent_towers"))
        ),
    }


def _row_for_perspective(
    *,
    game: Mapping[str, Any],
    event: Mapping[str, Any],
    strength: Mapping[str, Any],
    team_ids: tuple[str, str],
    perspective_team_id: str,
    pre_inventory: Mapping[str, Mapping[str, int]],
    post_inventory: Mapping[str, Mapping[str, int]],
    pre_souls: Mapping[str, str | None],
    post_souls: Mapping[str, str | None],
    stage: int,
) -> dict[str, Any]:
    owner_id = str(event.get("owner_team_id") or "")
    took_current = perspective_team_id == owner_id
    opponent_id = team_ids[1] if perspective_team_id == team_ids[0] else team_ids[0]
    own_prefix = _side_prefix(game, perspective_team_id)
    opp_prefix = _side_prefix(game, opponent_id)
    own_champions = _json_list(game.get(f"{own_prefix}_champions"))
    opp_champions = _json_list(game.get(f"{opp_prefix}_champions"))
    own_arch = _composition_counts(own_champions)
    opp_arch = _composition_counts(opp_champions)
    date = pd.to_datetime(game.get("date"), errors="coerce", utc=True)
    own_side = str(game.get(f"{own_prefix}_side") or "").casefold()
    own_org = _finite(strength.get(f"{own_prefix}_org_elo"), RATING_BASE)
    opp_org = _finite(strength.get(f"{opp_prefix}_org_elo"), RATING_BASE)
    own_players = _finite(
        strength.get(f"{own_prefix}_player_elo"),
        RATING_BASE,
    )
    opp_players = _finite(
        strength.get(f"{opp_prefix}_player_elo"),
        RATING_BASE,
    )
    won = int(perspective_team_id == str(game.get("winner_team_id") or ""))
    row: dict[str, Any] = {
        "series_id": str(game.get("series_id") or ""),
        "game_id": str(game.get("game_id") or ""),
        "date": date,
        "year": str(date.year) if pd.notna(date) else "unknown",
        "patch": str(game.get("patch") or "unknown"),
        "competition": str(game.get("competition") or "unknown"),
        "league": str(game.get("league") or "unknown"),
        "region": str(game.get("region") or "unknown"),
        "competition_level": str(
            game.get("competition_level") or "unknown"
        ),
        "perspective": own_prefix,
        "perspective_sign": 1.0 if own_prefix == "team_1" else -1.0,
        "own_team_id": perspective_team_id,
        "opp_team_id": opponent_id,
        "blue_sign": 1.0 if own_side == "blue" else -1.0,
        "perspective_won": won,
        # Compatibility with the established chronological diagnostic target.
        "owner_won": won,
        "current_element": str(event.get("element") or ""),
        "stage": int(stage),
        "minute": _finite(event.get("time_seconds")) / 60.0,
        "state_lag_seconds": _finite(event.get("state_lag_seconds")),
        "took_current": int(took_current),
        "allocation_sign": 1.0 if took_current else -1.0,
        "own_champions": own_champions,
        "opp_champions": opp_champions,
        "own_soul_element_pre": pre_souls[perspective_team_id],
        "opp_soul_element_pre": pre_souls[opponent_id],
        "own_soul_element_after": post_souls[perspective_team_id],
        "opp_soul_element_after": post_souls[opponent_id],
        "org_elo_diff": (own_org - opp_org) / 400.0,
        "player_elo_diff": (own_players - opp_players) / 400.0,
        "roster_coverage": min(
            _finite(strength.get(f"{own_prefix}_roster_coverage")),
            _finite(strength.get(f"{opp_prefix}_roster_coverage")),
        ),
    }
    row.update(_signed_state(event, took_current))
    for tag in ARCHETYPE_NAMES:
        row[f"own_{tag}"] = own_arch[tag]
        row[f"opp_{tag}"] = opp_arch[tag]
        row[f"diff_{tag}"] = own_arch[tag] - opp_arch[tag]
    for element in ELEMENTS:
        row[f"pre_own_count_{element}"] = int(
            pre_inventory[perspective_team_id][element]
        )
        row[f"pre_opp_count_{element}"] = int(
            pre_inventory[opponent_id][element]
        )
        row[f"post_own_count_{element}"] = int(
            post_inventory[perspective_team_id][element]
        )
        row[f"post_opp_count_{element}"] = int(
            post_inventory[opponent_id][element]
        )
    row["pre_own_total"] = _inventory_total(pre_inventory, perspective_team_id)
    row["pre_opp_total"] = _inventory_total(pre_inventory, opponent_id)
    row["post_own_total"] = _inventory_total(post_inventory, perspective_team_id)
    row["post_opp_total"] = _inventory_total(post_inventory, opponent_id)
    return row


def prepare_joint_rows(
    games: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return two legal mirrored team perspectives per usable capture."""
    event_frame = events.copy()
    if "occurred_at" not in event_frame.columns and "occurredAt" in event_frame:
        event_frame = event_frame.rename(columns={"occurredAt": "occurred_at"})
    if "occurred_at" not in event_frame:
        event_frame["occurred_at"] = event_frame.get("date")
    strengths = pregame_strengths(games, event_frame)
    game_lookup = {
        (str(row.series_id), str(row.game_id)): row._asdict()
        for row in games.itertuples(index=False)
    }
    strength_lookup = {
        (str(row.series_id), str(row.game_id)): row._asdict()
        for row in strengths.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    for (series_id, game_id), raw_group in event_frame.groupby(
        ["series_id", "game_id"],
        sort=False,
    ):
        key = (str(series_id), str(game_id))
        game = game_lookup.get(key)
        strength = strength_lookup.get(key)
        if (
            not game
            or not strength
            or not _winner_is_valid(game)
            or not _side_assignment_is_valid(game)
        ):
            continue
        team_ids = (
            str(game.get("team_1_id") or ""),
            str(game.get("team_2_id") or ""),
        )
        if (
            len(_json_list(game.get("team_1_champions"))) != 5
            or len(_json_list(game.get("team_2_champions"))) != 5
        ):
            continue
        inventory: dict[str, dict[str, int]] = {
            team_id: {element: 0 for element in ELEMENTS}
            for team_id in team_ids
        }
        souls: dict[str, str | None] = {team_id: None for team_id in team_ids}
        sort_columns = [
            column
            for column in ("global_index", "time_seconds", "occurred_at")
            if column in raw_group.columns
        ]
        group = raw_group.sort_values(sort_columns)
        if not _capture_path_is_legal(group, team_ids):
            continue
        for raw_event in group.itertuples(index=False):
            event = raw_event._asdict()
            owner = str(event.get("owner_team_id") or "")
            element = str(event.get("element") or "")
            if owner not in team_ids or element not in ELEMENTS:
                continue
            # No elemental capture is legal after either team has claimed soul.
            if any(souls.values()):
                break
            pre_inventory = {
                team_id: dict(inventory[team_id]) for team_id in team_ids
            }
            pre_souls = dict(souls)
            inventory[owner][element] += 1
            stage = sum(
                _inventory_total(inventory, team_id) for team_id in team_ids
            )
            if not _inventory_is_legal(inventory, team_ids):
                break
            if _inventory_total(inventory, owner) == MAX_STACKS:
                souls[owner] = element
            post_inventory = {
                team_id: dict(inventory[team_id]) for team_id in team_ids
            }
            post_souls = dict(souls)
            if not _valid_capture_state(event):
                continue
            capture_rows = [
                _row_for_perspective(
                    game=game,
                    event=event,
                    strength=strength,
                    team_ids=team_ids,
                    perspective_team_id=perspective_team_id,
                    pre_inventory=pre_inventory,
                    post_inventory=post_inventory,
                    pre_souls=pre_souls,
                    post_souls=post_souls,
                    stage=stage,
                )
                for perspective_team_id in team_ids
            ]
            if (
                capture_rows[0]["perspective_won"]
                + capture_rows[1]["perspective_won"]
                != 1
                or capture_rows[0]["took_current"]
                + capture_rows[1]["took_current"]
                != 1
            ):
                raise ValueError("Mirrored capture rows are not complementary.")
            rows.extend(capture_rows)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    if not (
        frame["post_own_total"].between(0, MAX_STACKS).all()
        and frame["post_opp_total"].between(0, MAX_STACKS).all()
        and frame["stage"].between(1, MAX_GLOBAL_STAGE).all()
    ):
        raise ValueError("Prepared drake rows contain an illegal inventory.")
    return frame.reset_index(drop=True)


def _number(rows: pd.DataFrame, column: str) -> pd.Series:
    if column not in rows:
        return pd.Series(0.0, index=rows.index, dtype=float)
    return pd.to_numeric(rows[column], errors="coerce").fillna(0.0).astype(float)


def _base_signed_features(rows: pd.DataFrame) -> dict[str, pd.Series]:
    columns = {column: _number(rows, column) for column in STATE_NUMERIC}
    columns["blue_sign"] = _number(rows, "blue_sign")
    columns["stage_x_blue"] = _number(rows, "stage") * columns["blue_sign"]
    columns["minute_x_blue"] = _number(rows, "minute") * columns["blue_sign"]
    columns["roster_gap_from_five_x_blue"] = (
        5.0 - _number(rows, "roster_coverage")
    ) * columns["blue_sign"]
    for tag in ARCHETYPE_NAMES:
        difference = _number(rows, f"own_{tag}") - _number(rows, f"opp_{tag}")
        columns[f"trait_diff_{tag}"] = difference
        columns[f"trait_diff_{tag}_x_minute"] = (
            difference * _number(rows, "minute")
        )
    for category in (
        "competition",
        "league",
        "region",
        "competition_level",
        "patch",
        "year",
    ):
        if category not in rows:
            continue
        for value in sorted(rows[category].fillna("unknown").astype(str).unique()):
            columns[f"{category}={value}_x_blue"] = (
                (rows[category].fillna("unknown").astype(str) == value).astype(float)
                * columns["blue_sign"]
            )
    return columns


def _add_inventory_features(
    columns: dict[str, pd.Series],
    rows: pd.DataFrame,
    *,
    prefix: str,
) -> None:
    minute = _number(rows, "minute")
    for element in ELEMENTS:
        own = _number(rows, f"{prefix}_own_count_{element}")
        opp = _number(rows, f"{prefix}_opp_count_{element}")
        difference = own - opp
        columns[f"{prefix}_inventory_diff_{element}"] = difference
        columns[f"{prefix}_inventory_diff_{element}_x_minute"] = (
            difference * minute
        )
        for tag in ARCHETYPE_NAMES:
            own_tag = _number(rows, f"own_{tag}")
            opp_tag = _number(rows, f"opp_{tag}")
            columns[f"{prefix}_{element}_own_trait_{tag}"] = (
                own * own_tag - opp * opp_tag
            )
            columns[f"{prefix}_{element}_enemy_trait_{tag}"] = (
                own * opp_tag - opp * own_tag
            )


def _add_soul_features(
    columns: dict[str, pd.Series],
    rows: pd.DataFrame,
    *,
    suffix: str,
) -> None:
    own_soul = rows.get(
        f"own_soul_element_{suffix}",
        pd.Series(None, index=rows.index),
    )
    opp_soul = rows.get(
        f"opp_soul_element_{suffix}",
        pd.Series(None, index=rows.index),
    )
    minute = _number(rows, "minute")
    for element in ELEMENTS:
        own = (own_soul == element).astype(float)
        opp = (opp_soul == element).astype(float)
        difference = own - opp
        columns[f"soul_{suffix}_{element}"] = difference
        columns[f"soul_{suffix}_{element}_x_minute"] = difference * minute
        for tag in ARCHETYPE_NAMES:
            own_tag = _number(rows, f"own_{tag}")
            opp_tag = _number(rows, f"opp_{tag}")
            columns[f"soul_{suffix}_{element}_own_trait_{tag}"] = (
                own * own_tag - opp * opp_tag
            )
            columns[f"soul_{suffix}_{element}_enemy_trait_{tag}"] = (
                own * opp_tag - opp * own_tag
            )


def _current_element_features(
    columns: dict[str, pd.Series],
    rows: pd.DataFrame,
    *,
    include_allocation: bool,
) -> None:
    current = rows["current_element"].astype(str)
    minute = _number(rows, "minute")
    allocation = _number(rows, "allocation_sign")
    for element in ELEMENTS:
        indicator = (current == element).astype(float)
        for state in STATE_NUMERIC:
            columns[f"current_{element}_x_{state}"] = (
                indicator * _number(rows, state)
            )
        if include_allocation:
            direction = indicator * allocation
            columns[f"allocation_{element}"] = direction
            columns[f"allocation_{element}_x_minute"] = direction * minute
            for state in STATE_NUMERIC:
                # Absolute state is invariant to the mirrored perspective, so
                # multiplying by allocation direction remains antisymmetric.
                columns[f"allocation_{element}_x_abs_{state}"] = (
                    direction * _number(rows, state).abs()
                )
            for tag in ARCHETYPE_NAMES:
                own_tag = _number(rows, f"own_{tag}")
                opp_tag = _number(rows, f"opp_{tag}")
                took = _number(rows, "took_current")
                columns[f"allocation_{element}_own_trait_{tag}"] = indicator * (
                    took * own_tag - (1.0 - took) * opp_tag
                )
                columns[f"allocation_{element}_enemy_trait_{tag}"] = indicator * (
                    took * opp_tag - (1.0 - took) * own_tag
                )


def _design_state(rows: pd.DataFrame) -> pd.DataFrame:
    """Signed feature design for joint post-capture inventory association."""
    columns = _base_signed_features(rows)
    _add_inventory_features(columns, rows, prefix="post")
    _add_soul_features(columns, rows, suffix="after")
    return pd.DataFrame(columns, index=rows.index).astype(float)


def _design_allocation(rows: pd.DataFrame) -> pd.DataFrame:
    """Signed feature design for resolved capture-allocation association."""
    columns = _base_signed_features(rows)
    _add_inventory_features(columns, rows, prefix="pre")
    # The hypothetical post inventory and soul are deterministic functions of
    # the pre state, current element, and treatment assignment.
    _add_inventory_features(columns, rows, prefix="post")
    _add_soul_features(columns, rows, suffix="after")
    _current_element_features(columns, rows, include_allocation=True)
    return pd.DataFrame(columns, index=rows.index).astype(float)


def allocation_counterfactual_rows(
    rows: pd.DataFrame,
    took_current: int | Sequence[int] | np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Change only allocation treatment and its deterministic post-state."""
    changed = rows.copy(deep=True)
    if np.isscalar(took_current):
        treatment = pd.Series(int(took_current), index=changed.index)
    else:
        treatment = pd.Series(took_current, index=changed.index).astype(int)
    if not treatment.isin([0, 1]).all():
        raise ValueError("Allocation treatment must be zero or one.")
    changed["took_current"] = treatment
    changed["allocation_sign"] = treatment * 2.0 - 1.0
    current = changed["current_element"].astype(str)
    for element in ELEMENTS:
        own_pre = _number(changed, f"pre_own_count_{element}")
        opp_pre = _number(changed, f"pre_opp_count_{element}")
        is_current = (current == element).astype(int)
        changed[f"post_own_count_{element}"] = (
            own_pre + treatment * is_current
        ).astype(int)
        changed[f"post_opp_count_{element}"] = (
            opp_pre + (1 - treatment) * is_current
        ).astype(int)
    changed["post_own_total"] = sum(
        _number(changed, f"post_own_count_{element}") for element in ELEMENTS
    ).astype(int)
    changed["post_opp_total"] = sum(
        _number(changed, f"post_opp_count_{element}") for element in ELEMENTS
    ).astype(int)
    if (
        (changed["post_own_total"] > MAX_STACKS).any()
        or (changed["post_opp_total"] > MAX_STACKS).any()
    ):
        raise ValueError("Counterfactual allocation would exceed four stacks.")
    own_soul = np.where(
        changed["post_own_total"].eq(MAX_STACKS),
        current,
        None,
    )
    opp_soul = np.where(
        changed["post_opp_total"].eq(MAX_STACKS),
        current,
        None,
    )
    changed["own_soul_element_after"] = own_soul
    changed["opp_soul_element_after"] = opp_soul
    return changed


def _game_weights(rows: pd.DataFrame) -> np.ndarray:
    """Give every game equal total weight despite differing capture counts."""
    keys = rows["series_id"].astype(str) + "\x1f" + rows["game_id"].astype(str)
    counts = keys.groupby(keys).transform("size").to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights * (len(weights) / weights.sum())


@dataclass
class TemporalPartitions:
    ordered: pd.DataFrame
    train: pd.DataFrame
    holdout: pd.DataFrame
    inner_train: pd.DataFrame
    inner_validation: pd.DataFrame


@dataclass
class PublicationAuditPartitions:
    ordered: pd.DataFrame
    train: pd.DataFrame
    holdout: pd.DataFrame
    cutoff: pd.Timestamp


@dataclass
class ChampionResidualElementSpec:
    element: str
    champions: tuple[str, ...]
    feature_names: tuple[str, ...]
    support: tuple[dict[str, Any], ...]
    basis: np.ndarray
    constraint: np.ndarray


@dataclass
class ChampionResidualSpec:
    selected_min_games: int
    elements: dict[str, ChampionResidualElementSpec]

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(
            feature_name
            for element in ELEMENTS
            for feature_name in (
                self.elements[element].feature_names
                if element in self.elements
                else ()
            )
        )

    @property
    def degrees_of_freedom(self) -> int:
        return int(sum(spec.basis.shape[1] for spec in self.elements.values()))


@dataclass
class ChampionResidualFit:
    selected_lambda: float
    raw_coefficients: dict[str, float]
    constraint_max_abs: dict[str, float]

    def linear_score(
        self,
        rows: pd.DataFrame,
        spec: ChampionResidualSpec,
    ) -> np.ndarray:
        raw = _direct_raw_design(rows, spec)
        coefficients = np.array(
            [self.raw_coefficients.get(name, 0.0) for name in raw.columns],
            dtype=float,
        )
        return _checked_matmul(
            raw.to_numpy(dtype=float),
            coefficients,
            context="champion residual score",
        )


def _checked_matmul(
    left: np.ndarray,
    right: np.ndarray,
    *,
    context: str,
) -> np.ndarray:
    """Multiply finite arrays and fail closed if the result is not finite.

    Apple's accelerated BLAS can leave floating-point status flags set after
    SciPy's null-space decomposition, which makes later finite matrix products
    emit spurious overflow warnings. Suppressing those flags is safe only
    because the operands and result are checked explicitly here.
    """
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if not np.isfinite(left_array).all() or not np.isfinite(right_array).all():
        raise ValueError(f"{context} received a non-finite operand.")
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        result = np.matmul(left_array, right_array)
    if not np.isfinite(result).all():
        raise ValueError(f"{context} produced a non-finite result.")
    return np.asarray(result, dtype=float)


def _temporal_partitions(rows: pd.DataFrame) -> TemporalPartitions:
    """Return the frozen chronological partitions used by both model layers."""
    ordered = rows.sort_values(
        ["date", "series_id", "game_id", "stage", "perspective"]
    ).reset_index(drop=True)
    train = ordered.loc[ordered["date"] < PUBLIC_HOLDOUT_START]
    holdout = ordered.loc[
        ordered["date"].between(
            PUBLIC_HOLDOUT_START,
            PUBLIC_HOLDOUT_END,
            inclusive="left",
        )
    ]
    if train.empty or holdout.empty:
        raise ValueError(
            "The prespecified March-April 2026 holdout requires both earlier "
            "training rows and in-window evaluation rows."
        )
    if set(train["series_id"].astype(str)) & set(
        holdout["series_id"].astype(str)
    ):
        raise ValueError("A GRID series crosses the public holdout boundary.")
    train_series_order = (
        train.groupby("series_id", as_index=False)["date"]
        .min()
        .sort_values(["date", "series_id"])
    )
    inner_split = max(
        1,
        min(
            len(train_series_order) - 1,
            int(len(train_series_order) * (1.0 - HOLDOUT_FRACTION)),
        ),
    )
    inner_train_series = set(
        train_series_order.iloc[:inner_split]["series_id"].astype(str)
    )
    inner_train = train.loc[
        train["series_id"].astype(str).isin(inner_train_series)
    ]
    inner_validation = train.loc[
        ~train["series_id"].astype(str).isin(inner_train_series)
    ]
    if inner_train.empty or inner_validation.empty:
        raise ValueError(
            "Regularization selection requires at least two chronological "
            "training-series partitions."
        )
    return TemporalPartitions(
        ordered=ordered,
        train=train,
        holdout=holdout,
        inner_train=inner_train,
        inner_validation=inner_validation,
    )


def _publication_audit_partitions(
    rows: pd.DataFrame,
    *,
    cutoff: pd.Timestamp = PUBLICATION_AUDIT_START,
    end: pd.Timestamp = PUBLICATION_AUDIT_END,
    minimum_games: int = PUBLICATION_AUDIT_MIN_GAMES,
    minimum_series: int = PUBLICATION_AUDIT_MIN_SERIES,
) -> PublicationAuditPartitions:
    """Freeze a pre-cutoff vocabulary and evaluate it on whole July series."""
    ordered = rows.sort_values(
        ["date", "series_id", "game_id", "stage", "perspective"]
    ).reset_index(drop=True)
    train = ordered.loc[ordered["date"] < cutoff]
    holdout = ordered.loc[
        ordered["date"].between(cutoff, end, inclusive="left")
    ]
    if train.empty or holdout.empty:
        raise ValueError(
            "The publication expansion audit requires both pre-July training "
            "rows and July evaluation rows."
        )
    train_series = set(train["series_id"].astype(str))
    holdout_series = set(holdout["series_id"].astype(str))
    outside_series = set(
        ordered.loc[
            ~ordered["date"].between(cutoff, end, inclusive="left"),
            "series_id",
        ].astype(str)
    )
    if train_series & holdout_series or holdout_series & outside_series:
        raise ValueError(
            "A GRID series crosses a publication-audit boundary."
        )
    holdout_games = int(
        holdout[["series_id", "game_id"]].drop_duplicates().shape[0]
    )
    holdout_series_count = int(holdout["series_id"].nunique())
    if (
        holdout_games < int(minimum_games)
        or holdout_series_count < int(minimum_series)
    ):
        raise ValueError(
            "The publication expansion audit has insufficient whole-series "
            f"July support: {holdout_games} games/{holdout_series_count} "
            f"series; requires {int(minimum_games)} games/"
            f"{int(minimum_series)} series."
        )
    return PublicationAuditPartitions(
        ordered=ordered,
        train=train,
        holdout=holdout,
        cutoff=cutoff,
    )


def _canonical_champions(value: Any) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            normalize_champ(champion)
            for champion in _json_list(value)
            if normalize_champ(champion)
        )
    )


def _direct_feature_name(champion: str, element: str) -> str:
    return f"champion_direct_inventory::{element}::{champion}"


def _champion_element_support(rows: pd.DataFrame) -> list[dict[str, Any]]:
    """Count independent team-games; mirrored/stage rows never inflate support."""
    records: list[dict[str, Any]] = []
    group_columns = ["series_id", "game_id", "perspective"]
    for (_, _, _), group in rows.groupby(group_columns, sort=False):
        first = group.iloc[0]
        champions = _canonical_champions(first.get("own_champions"))
        if not champions:
            continue
        series_id = str(first["series_id"])
        game_id = str(first["game_id"])
        own_team_id = str(
            first.get("own_team_id")
            or f"{first.get('perspective', 'unknown')}"
        )
        team_game = "\x1f".join((series_id, game_id, own_team_id))
        won = int(first["perspective_won"])
        for element in ELEMENTS:
            own_max = float(
                pd.to_numeric(
                    group[f"post_own_count_{element}"],
                    errors="coerce",
                )
                .fillna(0.0)
                .max()
            )
            opp_max = float(
                pd.to_numeric(
                    group[f"post_opp_count_{element}"],
                    errors="coerce",
                )
                .fillna(0.0)
                .max()
            )
            if max(own_max, opp_max) <= 0:
                # A champion is not a comparator for an element that never
                # appeared in this game.
                continue
            owns = int(own_max > 0)
            for champion in champions:
                records.append(
                    {
                        "champion": champion,
                        "element": element,
                        "teamGame": team_game,
                        "series": series_id,
                        "org": own_team_id,
                        "owns": owns,
                        "won": won,
                    }
                )
    if not records:
        return []
    frame = pd.DataFrame(records).drop_duplicates(
        ["champion", "element", "teamGame"]
    )
    support: list[dict[str, Any]] = []
    for (champion, element), group in frame.groupby(
        ["champion", "element"],
        sort=True,
    ):
        owns = group["owns"].eq(1)
        wins = group["won"].eq(1)
        ownership_games = int(owns.sum())
        nonownership_games = int((~owns).sum())
        effective_games = (
            4.0
            * ownership_games
            * nonownership_games
            / (ownership_games + nonownership_games)
            if ownership_games and nonownership_games
            else 0.0
        )
        support.append(
            {
                "champion": str(champion),
                "element": str(element),
                "featureName": _direct_feature_name(
                    str(champion),
                    str(element),
                ),
                "trainingGames": int(group["teamGame"].nunique()),
                "trainingSeries": int(group["series"].nunique()),
                "orgRosters": int(group["org"].nunique()),
                "organizations": int(group["org"].nunique()),
                "ownershipGames": ownership_games,
                "nonOwnershipGames": nonownership_games,
                "wins": int(wins.sum()),
                "losses": int((~wins).sum()),
                "ownershipWins": int((owns & wins).sum()),
                "ownershipLosses": int((owns & ~wins).sum()),
                "nonOwnershipWins": int((~owns & wins).sum()),
                "nonOwnershipLosses": int((~owns & ~wins).sum()),
                "effectiveGames": round(float(effective_games), 6),
                "supportWeight": max(float(effective_games), 1.0),
                "tags": sorted(champ_tags(str(champion))),
            }
        )
    element_order = {element: index for index, element in enumerate(ELEMENTS)}
    return sorted(
        support,
        key=lambda entry: (
            element_order.get(str(entry["element"]), len(ELEMENTS)),
            str(entry["champion"]).casefold(),
        ),
    )


def _cell_is_direct_eligible(
    support: Mapping[str, Any],
    *,
    min_games: int,
) -> bool:
    if int(min_games) < 50:
        return False
    return not _direct_cell_failed_exposure_rules(
        support,
        min_games=min_games,
    )


def _direct_cell_failed_exposure_rules(
    support: Mapping[str, Any],
    *,
    min_games: int,
) -> list[str]:
    """Return exposure-only release failures; outcomes are diagnostics only."""
    failures = []
    if int(support.get("trainingGames") or 0) < int(min_games):
        failures.append("minimum-games")
    if int(support.get("trainingSeries") or 0) < DIRECT_MIN_SERIES:
        failures.append("minimum-series")
    if (
        int(support.get("ownershipGames") or 0)
        < DIRECT_MIN_OWNERSHIP_GAMES
    ):
        failures.append("minimum-ownership-games")
    if (
        int(support.get("nonOwnershipGames") or 0)
        < DIRECT_MIN_NONOWNERSHIP_GAMES
    ):
        failures.append("minimum-nonownership-games")
    if int(support.get("orgRosters") or 0) < DIRECT_MIN_ORGS:
        failures.append("minimum-organizations")
    return failures


def _freeze_champion_residual_spec(
    support: Sequence[Mapping[str, Any]],
    *,
    min_games: int,
) -> ChampionResidualSpec:
    """Freeze eligible cells and a pooled+archetype null-space per element."""
    if int(min_games) < 50:
        raise ValueError("Direct champion support must require at least 50 games.")
    elements: dict[str, ChampionResidualElementSpec] = {}
    for element in ELEMENTS:
        eligible = [
            dict(entry)
            for entry in support
            if str(entry.get("element")) == element
            and _cell_is_direct_eligible(entry, min_games=min_games)
        ]
        eligible.sort(key=lambda entry: str(entry["champion"]).casefold())
        if not eligible:
            continue
        champions = tuple(str(entry["champion"]) for entry in eligible)
        tag_matrix = np.array(
            [
                [
                    float(tag in set(entry.get("tags", ())))
                    for tag in ARCHETYPE_NAMES
                ]
                for entry in eligible
            ],
            dtype=float,
        )
        basis_matrix = np.column_stack(
            [np.ones(len(eligible), dtype=float), tag_matrix]
        )
        support_weight = np.array(
            [float(entry["supportWeight"]) for entry in eligible],
            dtype=float,
        )
        constraint = basis_matrix.T * support_weight[None, :]
        residual_basis = null_space(constraint, rcond=1e-10)
        if residual_basis.shape[1] == 0:
            continue
        elements[element] = ChampionResidualElementSpec(
            element=element,
            champions=champions,
            feature_names=tuple(
                str(entry["featureName"]) for entry in eligible
            ),
            support=tuple(eligible),
            basis=residual_basis,
            constraint=constraint,
        )
    return ChampionResidualSpec(
        selected_min_games=int(min_games),
        elements=elements,
    )


def _champion_residual_cell_keys(
    spec: ChampionResidualSpec,
) -> set[tuple[str, str]]:
    return {
        (str(cell["champion"]), str(cell["element"]))
        for element_spec in spec.elements.values()
        for cell in element_spec.support
    }


def _champion_residual_vocabulary_summary(
    spec: ChampionResidualSpec,
) -> dict[str, Any]:
    keys = _champion_residual_cell_keys(spec)
    return {
        "cells": int(len(keys)),
        "champions": int(len({champion for champion, _ in keys})),
        "degreesOfFreedom": int(spec.degrees_of_freedom),
        "byElement": {
            element: int(len(spec.elements[element].champions))
            if element in spec.elements
            else 0
            for element in ELEMENTS
        },
    }


def _serialize_observed_champion_cells(
    support: Sequence[Mapping[str, Any]],
    *,
    min_games: int,
) -> list[dict[str, Any]]:
    observed = []
    for raw_cell in support:
        cell = dict(raw_cell)
        failures = _direct_cell_failed_exposure_rules(
            cell,
            min_games=min_games,
        )
        observed.append(
            {
                **{
                    key: value
                    for key, value in cell.items()
                    if key != "supportWeight"
                },
                "supportWeight": round(
                    float(cell.get("supportWeight") or 0.0),
                    6,
                ),
                "championEligible": not failures,
                "failedExposureRules": failures,
                "outcomeCountsUsedForEligibility": False,
            }
        )
    return observed


def _direct_cell_design(
    rows: pd.DataFrame,
    cells: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    """Generate raw own-use inventory features for serialized cells."""
    own_sets = [
        set(_canonical_champions(value))
        for value in rows["own_champions"]
    ]
    opp_sets = [
        set(_canonical_champions(value))
        for value in rows["opp_champions"]
    ]
    columns: dict[str, np.ndarray] = {}
    for cell in cells:
        champion = str(cell["champion"])
        element = str(cell["element"])
        feature_name = str(cell["featureName"])
        own_count = _number(
            rows,
            f"post_own_count_{element}",
        ).to_numpy(dtype=float)
        opp_count = _number(
            rows,
            f"post_opp_count_{element}",
        ).to_numpy(dtype=float)
        own_has = np.fromiter(
            (champion in champions for champions in own_sets),
            dtype=float,
            count=len(rows),
        )
        opp_has = np.fromiter(
            (champion in champions for champions in opp_sets),
            dtype=float,
            count=len(rows),
        )
        columns[feature_name] = own_count * own_has - opp_count * opp_has
    return pd.DataFrame(columns, index=rows.index, dtype=float)


def _direct_raw_design(
    rows: pd.DataFrame,
    spec: ChampionResidualSpec,
) -> pd.DataFrame:
    """Generate antisymmetric own-use inventory features for eligible cells."""
    cells = [
        cell
        for element in ELEMENTS
        for cell in (
            spec.elements[element].support
            if element in spec.elements
            else ()
        )
    ]
    return _direct_cell_design(rows, cells)


def _direct_basis_design(
    rows: pd.DataFrame,
    spec: ChampionResidualSpec,
) -> tuple[pd.DataFrame, dict[str, slice]]:
    raw = _direct_raw_design(rows, spec)
    blocks: list[np.ndarray] = []
    columns: list[str] = []
    slices: dict[str, slice] = {}
    offset = 0
    for element in ELEMENTS:
        element_spec = spec.elements.get(element)
        if element_spec is None:
            continue
        values = raw.loc[:, element_spec.feature_names].to_numpy(dtype=float)
        block = _checked_matmul(
            values,
            element_spec.basis,
            context=f"{element} champion residual basis design",
        )
        width = block.shape[1]
        blocks.append(block)
        columns.extend(
            f"__champion_direct_basis::{element}::{index}"
            for index in range(width)
        )
        slices[element] = slice(offset, offset + width)
        offset += width
    if not blocks:
        return pd.DataFrame(index=rows.index), {}
    return (
        pd.DataFrame(
            np.column_stack(blocks),
            index=rows.index,
            columns=columns,
        ),
        slices,
    )


def _fitted_logit_score(
    fit: FittedLogit,
    design: pd.DataFrame,
) -> np.ndarray:
    aligned = design.reindex(columns=fit.columns, fill_value=0.0)
    scaled = fit.scaler.transform(aligned.to_numpy(dtype=float))
    scaled = np.clip(scaled, -STANDARDIZED_CLIP, STANDARDIZED_CLIP)
    return (
        _checked_matmul(
            scaled,
            np.asarray(fit.model.coef_[0], dtype=float),
            context="frozen base logit score",
        )
        + float(fit.model.intercept_[0])
    )


def _fit_offset_ridge(
    rows: pd.DataFrame,
    spec: ChampionResidualSpec,
    *,
    base_score: np.ndarray,
    outcome: np.ndarray,
    sample_weight: np.ndarray,
    selected_lambda: float,
) -> ChampionResidualFit:
    """Fit a no-intercept logistic residual around a frozen base offset."""
    design, slices = _direct_basis_design(rows, spec)
    if design.shape[1] == 0:
        raise ValueError("Champion residual design has no eligible degrees of freedom.")
    values = design.to_numpy(dtype=float)
    base_score = np.asarray(base_score, dtype=float)
    outcome = np.asarray(outcome, dtype=float)
    sample_weight = np.asarray(sample_weight, dtype=float)
    if not (
        len(values)
        == len(base_score)
        == len(outcome)
        == len(sample_weight)
    ):
        raise ValueError("Offset inputs must have identical row counts.")
    if not (
        np.isfinite(values).all()
        and np.isfinite(base_score).all()
        and np.isfinite(sample_weight).all()
        and float(sample_weight.sum()) > 0
    ):
        raise ValueError("Offset inputs must be finite with positive weight.")
    weight = sample_weight / float(sample_weight.sum())
    penalty = float(selected_lambda)

    def objective(coefficient: np.ndarray) -> tuple[float, np.ndarray]:
        score = base_score + _checked_matmul(
            values,
            coefficient,
            context="champion residual objective score",
        )
        probability = expit(score)
        loss = float(
            np.sum(weight * (np.logaddexp(0.0, score) - outcome * score))
            + 0.5 * penalty * np.dot(coefficient, coefficient)
        )
        gradient = _checked_matmul(
            values.T,
            weight * (probability - outcome),
            context="champion residual objective gradient",
        )
        gradient += penalty * coefficient
        return loss, np.asarray(gradient, dtype=float)

    result = minimize(
        objective,
        np.zeros(values.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if (
        not result.success
        or not np.isfinite(result.x).all()
        or not math.isfinite(float(result.fun))
    ):
        raise ValueError(f"Champion residual optimization failed: {result.message}")
    raw_coefficients: dict[str, float] = {}
    constraint_max_abs: dict[str, float] = {}
    for element in ELEMENTS:
        element_spec = spec.elements.get(element)
        if element_spec is None:
            continue
        gamma = np.asarray(result.x[slices[element]], dtype=float)
        delta = _checked_matmul(
            element_spec.basis,
            gamma,
            context=f"{element} champion residual coefficient recovery",
        )
        violation = _checked_matmul(
            element_spec.constraint,
            delta,
            context=f"{element} champion residual constraint check",
        )
        constraint_max_abs[element] = float(
            np.max(np.abs(violation)) if len(violation) else 0.0
        )
        raw_coefficients.update(
            {
                feature_name: float(coefficient)
                for feature_name, coefficient in zip(
                    element_spec.feature_names,
                    delta,
                )
            }
        )
    worst_constraint_violation = max(
        constraint_max_abs.values(),
        default=0.0,
    )
    if worst_constraint_violation > DIRECT_MAX_CONSTRAINT_VIOLATION:
        raise ValueError(
            "Champion residual effect-coding constraint violation "
            f"{worst_constraint_violation:.3e} exceeds "
            f"{DIRECT_MAX_CONSTRAINT_VIOLATION:.3e}."
        )
    return ChampionResidualFit(
        selected_lambda=penalty,
        raw_coefficients=raw_coefficients,
        constraint_max_abs=constraint_max_abs,
    )


def _prediction_metrics(
    outcome: np.ndarray,
    probability: np.ndarray,
    sample_weight: np.ndarray,
) -> dict[str, float]:
    return {
        "brier": float(
            brier_score_loss(
                outcome,
                probability,
                sample_weight=sample_weight,
            )
        ),
        "logLoss": float(
            log_loss(
                outcome,
                probability,
                labels=[0, 1],
                sample_weight=sample_weight,
            )
        ),
        "ece10": float(
            _weighted_ece(
                np.asarray(outcome, dtype=int),
                np.asarray(probability, dtype=float),
                np.asarray(sample_weight, dtype=float),
            )
        ),
    }


def _series_block_delta_brier_interval(
    rows: pd.DataFrame,
    *,
    outcome: np.ndarray,
    base_probability: np.ndarray,
    augmented_probability: np.ndarray,
    sample_weight: np.ndarray,
    iterations: int = 2_000,
    seed: int = 461,
) -> dict[str, float | int]:
    """Paired no-refit uncertainty interval from resampled locked series."""
    if not (
        len(rows)
        == len(outcome)
        == len(base_probability)
        == len(augmented_probability)
        == len(sample_weight)
    ):
        raise ValueError("Bootstrap inputs must have identical row counts.")
    frame = pd.DataFrame(
        {
            "series": rows["series_id"].astype(str).to_numpy(),
            "weight": np.asarray(sample_weight, dtype=float),
            "baseError": (
                np.asarray(base_probability, dtype=float)
                - np.asarray(outcome, dtype=float)
            )
            ** 2,
            "augmentedError": (
                np.asarray(augmented_probability, dtype=float)
                - np.asarray(outcome, dtype=float)
            )
            ** 2,
        }
    )
    frame["deltaNumerator"] = frame["weight"] * (
        frame["augmentedError"] - frame["baseError"]
    )
    grouped = frame.groupby("series", sort=True).agg(
        weight=("weight", "sum"),
        deltaNumerator=("deltaNumerator", "sum"),
    )
    if grouped.empty:
        raise ValueError("Bootstrap requires at least one holdout series.")
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(grouped),
        size=(int(iterations), len(grouped)),
    )
    series_weight = grouped["weight"].to_numpy(dtype=float)
    series_delta = grouped["deltaNumerator"].to_numpy(dtype=float)
    denominator = series_weight[draws].sum(axis=1)
    delta = series_delta[draws].sum(axis=1) / denominator
    return {
        "lower": float(np.quantile(delta, 0.025)),
        "upper": float(np.quantile(delta, 0.975)),
        "iterations": int(iterations),
        "series": int(len(grouped)),
        "seed": int(seed),
    }


def _direct_holdout_is_materially_worse(
    *,
    base: Mapping[str, float],
    augmented: Mapping[str, float],
    delta_brier_interval: Mapping[str, float] | None = None,
) -> bool:
    return bool(
        float(augmented["brier"])
        > float(base["brier"]) + DIRECT_MAX_BRIER_REGRESSION
        or float(augmented["logLoss"])
        > float(base["logLoss"]) + DIRECT_MAX_LOG_LOSS_REGRESSION
        or float(augmented["ece10"])
        > min(
            0.10,
            float(base["ece10"]) + DIRECT_MAX_ECE_REGRESSION,
        )
        or (
            delta_brier_interval is not None
            and float(delta_brier_interval["upper"])
            > DIRECT_MAX_BRIER_REGRESSION
        )
    )


def _direct_family_gate(
    *,
    selected_gate: float,
    inner_base: Mapping[str, float],
    inner_augmented: Mapping[str, float],
    holdout_base: Mapping[str, float],
    holdout_augmented: Mapping[str, float],
    delta_brier_interval: Mapping[str, float] | None = None,
) -> tuple[str, str]:
    if float(selected_gate) <= 0:
        return (
            "withheld",
            "Development tuning selected a zero family gate.",
        )
    if float(inner_augmented["brier"]) >= float(inner_base["brier"]) - 1e-9:
        return (
            "withheld",
            "The direct family did not improve development Brier score.",
        )
    materially_worse = _direct_holdout_is_materially_worse(
        base=holdout_base,
        augmented=holdout_augmented,
        delta_brier_interval=delta_brier_interval,
    )
    if materially_worse:
        return (
            "withheld",
            "The exploratory direct family was materially worse than the frozen "
            "base on at least one locked-holdout gate.",
        )
    return (
        "ready",
        "The exploratory family improved development Brier score and was not "
        "materially worse than the frozen base on the locked holdout.",
    )


def _publication_expansion_audit_gate(
    *,
    selected_gate: float,
    base: Mapping[str, float],
    evaluation_vocabulary: Mapping[str, float],
    expanded: Mapping[str, float],
    versus_base_interval: Mapping[str, float],
    versus_evaluation_interval: Mapping[str, float],
) -> tuple[str, str]:
    if float(selected_gate) <= 0:
        return (
            "withheld",
            "Development tuning selected a zero family gate.",
        )
    if _direct_holdout_is_materially_worse(
        base=base,
        augmented=expanded,
        delta_brier_interval=versus_base_interval,
    ):
        return (
            "withheld",
            "The pre-July expanded champion vocabulary was materially worse "
            "than its frozen base on the whole-series July audit.",
        )
    if _direct_holdout_is_materially_worse(
        base=evaluation_vocabulary,
        augmented=expanded,
        delta_brier_interval=versus_evaluation_interval,
    ):
        return (
            "withheld",
            "The pre-July expanded champion vocabulary was materially worse "
            "than the original evaluation vocabulary refitted pre-July on the "
            "whole-series July audit.",
        )
    return (
        "ready",
        "The pre-July expanded champion vocabulary was not materially worse "
        "than either its frozen base or the original evaluation vocabulary "
        "refitted pre-July on the whole-series July audit.",
    )


def _weighted_ece(
    outcome: np.ndarray,
    probability: np.ndarray,
    weight: np.ndarray,
    bins: int = 10,
) -> float:
    total = float(weight.sum())
    if total <= 0:
        return math.nan
    error = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (
            probability < high if high < 1 else probability <= high
        )
        if not mask.any():
            continue
        bin_weight = weight[mask]
        share = float(bin_weight.sum() / total)
        observed = float(np.average(outcome[mask], weights=bin_weight))
        predicted = float(np.average(probability[mask], weights=bin_weight))
        error += share * abs(observed - predicted)
    return float(error)


def _diagnostics(
    rows: pd.DataFrame,
    design_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[dict[str, Any], FittedLogit]:
    """Locked temporal diagnostics with equal-total-per-game weighting."""
    partitions = _temporal_partitions(rows)
    ordered = partitions.ordered
    train = partitions.train
    test = partitions.holdout
    inner_train = partitions.inner_train
    inner_validation = partitions.inner_validation
    inner_train_design = design_fn(inner_train)
    inner_validation_design = design_fn(inner_validation)
    train_design = design_fn(train)
    test_design = design_fn(test)
    full_design = design_fn(ordered)
    inner_train_outcome = inner_train["perspective_won"].to_numpy(dtype=int)
    inner_validation_outcome = inner_validation[
        "perspective_won"
    ].to_numpy(dtype=int)
    train_outcome = train["perspective_won"].to_numpy(dtype=int)
    test_y = test["perspective_won"].to_numpy(dtype=int)
    full_outcome = ordered["perspective_won"].to_numpy(dtype=int)
    inner_train_weight = _game_weights(inner_train)
    inner_validation_weight = _game_weights(inner_validation)
    train_weight = _game_weights(train)
    test_weight = _game_weights(test)
    full_weights = _game_weights(ordered)

    alpha_scores: dict[float, float] = {}
    for alpha in ALPHA_GRID:
        candidate = _fit(
            inner_train_design,
            inner_train_outcome,
            alpha=alpha,
            sample_weight=inner_train_weight,
        )
        probability = candidate.predict(
            inner_validation_design,
        )
        alpha_scores[alpha] = float(
            brier_score_loss(
                inner_validation_outcome,
                probability,
                sample_weight=inner_validation_weight,
            )
        )
    selected_alpha = min(
        ALPHA_GRID,
        key=lambda alpha: (alpha_scores[alpha], -alpha),
    )
    holdout_fit = _fit(
        train_design,
        train_outcome,
        alpha=selected_alpha,
        sample_weight=train_weight,
    )
    probability = holdout_fit.predict(test_design)
    train_mean = float(
        np.average(train_outcome, weights=train_weight)
    )
    null_probability = np.repeat(train_mean, len(test))
    diagnostics = {
        "trainRows": int(len(train)),
        "holdoutRows": int(len(test)),
        "trainGames": int(train[["series_id", "game_id"]].drop_duplicates().shape[0]),
        "holdoutGames": int(test[["series_id", "game_id"]].drop_duplicates().shape[0]),
        "trainSeries": int(train["series_id"].nunique()),
        "holdoutSeries": int(test["series_id"].nunique()),
        "seriesSets": {
            "innerTrain": _identifier_set_provenance(
                inner_train["series_id"].astype(str).tolist()
            ),
            "innerValidation": _identifier_set_provenance(
                inner_validation["series_id"].astype(str).tolist()
            ),
            "train": _identifier_set_provenance(
                train["series_id"].astype(str).tolist()
            ),
            "holdout": _identifier_set_provenance(
                test["series_id"].astype(str).tolist()
            ),
        },
        "holdoutStart": (
            test["date"].min().isoformat()
            if len(test) and pd.notna(test["date"].min())
            else None
        ),
        "holdoutEnd": PUBLIC_HOLDOUT_END.isoformat(),
        "holdoutActualEnd": (
            test["date"].max().isoformat()
            if len(test) and pd.notna(test["date"].max())
            else None
        ),
        "postHoldoutRows": int((ordered["date"] >= PUBLIC_HOLDOUT_END).sum()),
        "auc": round(
            float(roc_auc_score(test_y, probability, sample_weight=test_weight)),
            4,
        )
        if len(np.unique(test_y)) == 2
        else None,
        "brier": round(
            float(
                brier_score_loss(
                    test_y,
                    probability,
                    sample_weight=test_weight,
                )
            ),
            4,
        ),
        "nullBrier": round(
            float(
                brier_score_loss(
                    test_y,
                    null_probability,
                    sample_weight=test_weight,
                )
            ),
            4,
        ),
        "logLoss": round(
            float(
                log_loss(
                    test_y,
                    probability,
                    labels=[0, 1],
                    sample_weight=test_weight,
                )
            ),
            4,
        ),
        "ece10": round(
            _weighted_ece(test_y, probability, test_weight),
            4,
        ),
        "selectedAlpha": selected_alpha,
        "innerValidationBrier": round(alpha_scores[selected_alpha], 4),
        "weighting": (
            "equal total weight per game, normalized independently inside "
            "each fitting and evaluation partition"
        ),
        "designFreeze": (
            "each evaluation design and scaler is fitted on its training "
            "partition; held-out rows are aligned to that frozen schema"
        ),
    }
    final_fit = _fit(
        full_design,
        full_outcome,
        alpha=selected_alpha,
        sample_weight=full_weights,
    )
    return diagnostics, final_fit


def _rounded_metrics(metrics: Mapping[str, float]) -> dict[str, float]:
    return {
        name: round(float(value), 6)
        for name, value in metrics.items()
    }


def _champion_residual_support_policy() -> dict[str, Any]:
    return {
        "supportUnit": (
            "unique team-game perspective where the element appeared; capture "
            "stages and mirrored rows do not add support"
        ),
        "eligibilityRule": (
            "exposure-only: games, series, ownership/nonownership comparison "
            "games, and distinct organizations/team contexts"
        ),
        "outcomeCounts": (
            "wins and losses are serialized as diagnostics but never determine "
            "vocabulary eligibility"
        ),
        "evaluationVocabularyFrozenFrom": (
            "inner chronological training series before March 2026; used only "
            "to select support threshold, ridge penalty, and family gate"
        ),
        "publicationAuditVocabularyFrozenFrom": (
            "all whole series before 2026-07-01; coefficients are fitted "
            "pre-cutoff and evaluated on available whole July 2026 series "
            "through the reported actual holdout end"
        ),
        "publicationVocabularyFrozenFrom": (
            "the full cohort only after both family-level evaluation gates pass; "
            "the final coefficients are then refitted on the full cohort"
        ),
        "minGameCandidates": list(DIRECT_MIN_GAME_GRID),
        "minimumReleaseGames": min(DIRECT_MIN_GAME_GRID),
        "minimumSeries": DIRECT_MIN_SERIES,
        "minimumOwnershipGames": DIRECT_MIN_OWNERSHIP_GAMES,
        "minimumNonOwnershipGames": DIRECT_MIN_NONOWNERSHIP_GAMES,
        "minimumOrganizations": DIRECT_MIN_ORGS,
        "publicationAudit": {
            "start": PUBLICATION_AUDIT_START.isoformat(),
            "end": PUBLICATION_AUDIT_END.isoformat(),
            "minimumGames": PUBLICATION_AUDIT_MIN_GAMES,
            "minimumSeries": PUBLICATION_AUDIT_MIN_SERIES,
            "unit": "whole GRID series",
        },
        "validationScope": (
            "both temporal checks validate the champion-residual family and "
            "vocabulary-expansion procedure; they do not independently validate "
            "each champion-element coefficient"
        ),
        "supportWeight": (
            "four times ownershipGames times nonOwnershipGames divided by "
            "their sum; this equals total games when exposure is balanced"
        ),
    }


def _disabled_champion_families() -> dict[str, dict[str, str]]:
    return {
        "championSoul": {
            "status": "disabled",
            "reason": (
                "Soul support is too small for a separate champion-identity "
                "family; soul remains pooled plus archetype."
            ),
        },
        "stageSpecificChampion": {
            "status": "disabled",
            "reason": (
                "The champion estimate is one per inventory stack and is not "
                "split by capture stage."
            ),
        },
        "allyPair": {
            "status": "disabled",
            "reason": (
                "Exact ally-pair-by-element interactions have not passed an "
                "independent support and holdout gate."
            ),
        },
        "enemyIdentity": {
            "status": "disabled",
            "reason": (
                "Enemy champion and directional counter interactions have not "
                "passed an independent support and holdout gate."
            ),
        },
        "genericDraftScore": {
            "status": "disabled",
            "reason": (
                "Generic Draft Score coefficients estimate a different "
                "pre-match composition estimand and are not imported."
            ),
        },
    }


def _empty_champion_residual_result(
    *,
    reason: str,
    diagnostics: Mapping[str, Any] | None = None,
    observed_cells: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "exploratory": True,
        "family": DIRECT_FAMILY,
        "familyGate": 0.0,
        "selectedLambda": None,
        "selectedMinGames": None,
        "supportPolicy": _champion_residual_support_policy(),
        "effectCoding": {
            "rawFeature": (
                "post own element stacks * own champion indicator minus post "
                "opponent stacks * opponent champion indicator"
            ),
            "constraint": "B^T Q delta = 0 separately for each element",
            "basis": (
                "weighted null space of pooled intercept plus all archetype tags"
            ),
            "anchor": (
                "delta is a ridge-shrunk deviation from the frozen pooled and "
                "archetype base, not a second archetype bonus"
            ),
        },
        "diagnostics": {
            "reason": reason,
            **dict(diagnostics or {}),
        },
        "vocabularies": {
            "evaluation": None,
            "publicationAudit": None,
            "publication": None,
            "addedPostAudit": {
                "cells": 0,
                "champions": 0,
                "cellKeys": [],
            },
        },
        "eligibleCells": [],
        "observedCells": [dict(cell) for cell in observed_cells],
        "disabledFamilies": _disabled_champion_families(),
    }


def _run_publication_expansion_audit(
    rows: pd.DataFrame,
    *,
    evaluation_spec: ChampionResidualSpec,
    selected_alpha: float,
    selected_min_games: int,
    selected_lambda: float,
    selected_gate: float,
) -> tuple[str, str, ChampionResidualSpec | None, dict[str, Any]]:
    """Evaluate a pre-July expanded vocabulary on untouched July series."""
    try:
        partitions = _publication_audit_partitions(rows)
    except ValueError as error:
        return (
            "withheld",
            str(error),
            None,
            {
                "status": "withheld",
                "reason": str(error),
                "cutoff": PUBLICATION_AUDIT_START.isoformat(),
                "end": PUBLICATION_AUDIT_END.isoformat(),
            },
        )

    train = partitions.train
    holdout = partitions.holdout
    train_design = _design_state(train)
    holdout_design = _design_state(holdout)
    train_outcome = train["perspective_won"].to_numpy(dtype=int)
    holdout_outcome = holdout["perspective_won"].to_numpy(dtype=int)
    train_weight = _game_weights(train)
    holdout_weight = _game_weights(holdout)
    support = _champion_element_support(train)
    spec = _freeze_champion_residual_spec(
        support,
        min_games=selected_min_games,
    )
    vocabulary = _champion_residual_vocabulary_summary(spec)
    if spec.degrees_of_freedom == 0:
        reason = (
            "The pre-July publication-audit vocabulary has no estimable "
            "champion-estimate degrees of freedom."
        )
        return (
            "withheld",
            reason,
            spec,
            {
                "status": "withheld",
                "reason": reason,
                "cutoff": PUBLICATION_AUDIT_START.isoformat(),
                "end": PUBLICATION_AUDIT_END.isoformat(),
                "vocabulary": vocabulary,
            },
        )

    base_fit = _fit(
        train_design,
        train_outcome,
        alpha=float(selected_alpha),
        sample_weight=train_weight,
    )
    train_base_score = _fitted_logit_score(
        base_fit,
        train_design,
    )
    holdout_base_score = _fitted_logit_score(
        base_fit,
        holdout_design,
    )
    try:
        expanded_fit = _fit_offset_ridge(
            train,
            spec,
            base_score=train_base_score,
            outcome=train_outcome,
            sample_weight=train_weight,
            selected_lambda=selected_lambda,
        )
        evaluation_fit = _fit_offset_ridge(
            train,
            evaluation_spec,
            base_score=train_base_score,
            outcome=train_outcome,
            sample_weight=train_weight,
            selected_lambda=selected_lambda,
        )
    except ValueError as error:
        reason = (
            "The pre-July publication-audit residual fit failed closed: "
            f"{error}"
        )
        return (
            "withheld",
            reason,
            spec,
            {
                "status": "withheld",
                "reason": reason,
                "cutoff": PUBLICATION_AUDIT_START.isoformat(),
                "end": PUBLICATION_AUDIT_END.isoformat(),
                "vocabulary": vocabulary,
            },
        )
    expanded_residual = expanded_fit.linear_score(holdout, spec)
    evaluation_residual = evaluation_fit.linear_score(
        holdout,
        evaluation_spec,
    )
    base_probability = expit(holdout_base_score)
    expanded_probability = expit(
        holdout_base_score + float(selected_gate) * expanded_residual
    )
    evaluation_probability = expit(
        holdout_base_score + float(selected_gate) * evaluation_residual
    )
    base_metrics = _prediction_metrics(
        holdout_outcome,
        base_probability,
        holdout_weight,
    )
    expanded_metrics = _prediction_metrics(
        holdout_outcome,
        expanded_probability,
        holdout_weight,
    )
    evaluation_metrics = _prediction_metrics(
        holdout_outcome,
        evaluation_probability,
        holdout_weight,
    )
    versus_base_interval = _series_block_delta_brier_interval(
        holdout,
        outcome=holdout_outcome,
        base_probability=base_probability,
        augmented_probability=expanded_probability,
        sample_weight=holdout_weight,
        seed=7626,
    )
    versus_evaluation_interval = _series_block_delta_brier_interval(
        holdout,
        outcome=holdout_outcome,
        base_probability=evaluation_probability,
        augmented_probability=expanded_probability,
        sample_weight=holdout_weight,
        seed=7627,
    )
    status, reason = _publication_expansion_audit_gate(
        selected_gate=selected_gate,
        base=base_metrics,
        evaluation_vocabulary=evaluation_metrics,
        expanded=expanded_metrics,
        versus_base_interval=versus_base_interval,
        versus_evaluation_interval=versus_evaluation_interval,
    )
    evaluation_keys = _champion_residual_cell_keys(evaluation_spec)
    expanded_keys = _champion_residual_cell_keys(spec)
    added_keys = expanded_keys - evaluation_keys
    added_cells = [
        dict(cell)
        for element_spec in spec.elements.values()
        for cell in element_spec.support
        if (str(cell["champion"]), str(cell["element"])) in added_keys
    ]
    added_design = _direct_cell_design(holdout, added_cells)
    if added_design.shape[1]:
        exposed_columns = [
            column
            for column in added_design.columns
            if bool((added_design[column].abs() > 0).any())
        ]
        affected_mask = (
            added_design.loc[:, exposed_columns].abs().gt(0).any(axis=1)
            if exposed_columns
            else pd.Series(False, index=holdout.index)
        )
    else:
        exposed_columns = []
        affected_mask = pd.Series(False, index=holdout.index)
    affected = holdout.loc[affected_mask]
    expansion_exposure = {
        "addedCells": int(len(added_keys)),
        "addedCellsWithNonzeroJulyExposure": int(len(exposed_columns)),
        "affectedRows": int(len(affected)),
        "affectedGames": int(
            affected[["series_id", "game_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "affectedSeries": int(affected["series_id"].nunique()),
        "cellKeysWithNonzeroJulyExposure": sorted(
            (
                f"{cell['champion']}::{cell['element']}"
                for cell in added_cells
                if str(cell["featureName"]) in set(exposed_columns)
            ),
            key=str.casefold,
        ),
        "interpretation": (
            "Nonzero raw July feature exposure for cells added beyond the "
            "original evaluation vocabulary; this is an audit-coverage "
            "diagnostic, not individual cell validation."
        ),
    }
    diagnostics = {
        "status": status,
        "reason": reason,
        "cutoff": PUBLICATION_AUDIT_START.isoformat(),
        "end": PUBLICATION_AUDIT_END.isoformat(),
        "plannedStart": PUBLICATION_AUDIT_START.isoformat(),
        "plannedEnd": PUBLICATION_AUDIT_END.isoformat(),
        "trainRows": int(len(train)),
        "holdoutRows": int(len(holdout)),
        "trainGames": int(
            train[["series_id", "game_id"]].drop_duplicates().shape[0]
        ),
        "holdoutGames": int(
            holdout[["series_id", "game_id"]].drop_duplicates().shape[0]
        ),
        "trainSeries": int(train["series_id"].nunique()),
        "holdoutSeries": int(holdout["series_id"].nunique()),
        "seriesSets": {
            "preJulyTrain": _identifier_set_provenance(
                train["series_id"].astype(str).tolist()
            ),
            "julyHoldout": _identifier_set_provenance(
                holdout["series_id"].astype(str).tolist()
            ),
        },
        "holdoutStart": (
            holdout["date"].min().isoformat()
            if pd.notna(holdout["date"].min())
            else None
        ),
        "holdoutEnd": (
            holdout["date"].max().isoformat()
            if pd.notna(holdout["date"].max())
            else None
        ),
        "wholeSeries": True,
        "designFreeze": (
            "base feature columns and scaler fitted from pre-July rows only; "
            "July rows are reindexed to that frozen training schema"
        ),
        "weighting": (
            "equal total weight per game, normalized independently within "
            "pre-July fitting and July evaluation partitions"
        ),
        "vocabulary": vocabulary,
        "evaluationVocabulary": (
            _champion_residual_vocabulary_summary(evaluation_spec)
        ),
        "expandedCellExposure": expansion_exposure,
        "base": _rounded_metrics(base_metrics),
        "evaluationVocabularyRefit": _rounded_metrics(evaluation_metrics),
        "expanded": _rounded_metrics(expanded_metrics),
        "augmented": _rounded_metrics(expanded_metrics),
        "deltaBrier": round(
            float(expanded_metrics["brier"] - base_metrics["brier"]),
            6,
        ),
        "deltaLogLoss": round(
            float(
                expanded_metrics["logLoss"] - base_metrics["logLoss"]
            ),
            6,
        ),
        "deltaEce10": round(
            float(expanded_metrics["ece10"] - base_metrics["ece10"]),
            6,
        ),
        "deltaBrierSeriesBootstrap95": {
            "lower": round(float(versus_base_interval["lower"]), 6),
            "upper": round(float(versus_base_interval["upper"]), 6),
            "iterations": int(versus_base_interval["iterations"]),
            "series": int(versus_base_interval["series"]),
            "seed": int(versus_base_interval["seed"]),
            "gate": (
                "upper bound must be at most the allowed Brier regression"
            ),
        },
        "comparisons": {
            "expandedVersusBase": {
                "deltaBrier": round(
                    float(
                        expanded_metrics["brier"]
                        - base_metrics["brier"]
                    ),
                    6,
                ),
                "deltaLogLoss": round(
                    float(
                        expanded_metrics["logLoss"]
                        - base_metrics["logLoss"]
                    ),
                    6,
                ),
                "deltaEce10": round(
                    float(
                        expanded_metrics["ece10"]
                        - base_metrics["ece10"]
                    ),
                    6,
                ),
                "deltaBrierSeriesBootstrap95": {
                    "lower": round(
                        float(versus_base_interval["lower"]),
                        6,
                    ),
                    "upper": round(
                        float(versus_base_interval["upper"]),
                        6,
                    ),
                    "iterations": int(
                        versus_base_interval["iterations"]
                    ),
                    "series": int(versus_base_interval["series"]),
                    "seed": int(versus_base_interval["seed"]),
                },
            },
            "expandedVersusEvaluationVocabulary": {
                "deltaBrier": round(
                    float(
                        expanded_metrics["brier"]
                        - evaluation_metrics["brier"]
                    ),
                    6,
                ),
                "deltaLogLoss": round(
                    float(
                        expanded_metrics["logLoss"]
                        - evaluation_metrics["logLoss"]
                    ),
                    6,
                ),
                "deltaEce10": round(
                    float(
                        expanded_metrics["ece10"]
                        - evaluation_metrics["ece10"]
                    ),
                    6,
                ),
                "deltaBrierSeriesBootstrap95": {
                    "lower": round(
                        float(versus_evaluation_interval["lower"]),
                        6,
                    ),
                    "upper": round(
                        float(versus_evaluation_interval["upper"]),
                        6,
                    ),
                    "iterations": int(
                        versus_evaluation_interval["iterations"]
                    ),
                    "series": int(
                        versus_evaluation_interval["series"]
                    ),
                    "seed": int(versus_evaluation_interval["seed"]),
                },
            },
        },
        "materialRegressionLimits": {
            "brier": DIRECT_MAX_BRIER_REGRESSION,
            "logLoss": DIRECT_MAX_LOG_LOSS_REGRESSION,
            "ece10": DIRECT_MAX_ECE_REGRESSION,
        },
        "validationScope": (
            "family-level publication-vocabulary expansion audit; no "
            "individual champion-element coefficient is separately validated"
        ),
    }
    return status, reason, spec, diagnostics


def _fit_champion_residual_family(
    rows: pd.DataFrame,
    *,
    selected_alpha: float,
    final_base_fit: FittedLogit,
) -> dict[str, Any]:
    """Tune and evaluate the exploratory champion inventory residual family."""
    partitions = _temporal_partitions(rows)
    ordered = partitions.ordered
    inner_train = partitions.inner_train
    inner_validation = partitions.inner_validation
    train = partitions.train
    holdout = partitions.holdout
    full_design = _design_state(ordered)
    inner_train_design = _design_state(inner_train)
    inner_validation_design = _design_state(inner_validation)
    train_design = _design_state(train)
    holdout_design = _design_state(holdout)
    full_outcome = ordered["perspective_won"].to_numpy(dtype=int)
    inner_train_outcome = inner_train["perspective_won"].to_numpy(dtype=int)
    inner_validation_outcome = inner_validation[
        "perspective_won"
    ].to_numpy(dtype=int)
    train_outcome = train["perspective_won"].to_numpy(dtype=int)
    holdout_outcome = holdout["perspective_won"].to_numpy(dtype=int)
    full_weights = _game_weights(ordered)
    inner_train_weight = _game_weights(inner_train)
    inner_validation_weight = _game_weights(inner_validation)
    train_weight = _game_weights(train)
    holdout_weight = _game_weights(holdout)
    full_support = _champion_element_support(ordered)

    base_inner_fit = _fit(
        inner_train_design,
        inner_train_outcome,
        alpha=float(selected_alpha),
        sample_weight=inner_train_weight,
    )
    base_inner_train_score = _fitted_logit_score(
        base_inner_fit,
        inner_train_design,
    )
    base_inner_validation_score = _fitted_logit_score(
        base_inner_fit,
        inner_validation_design,
    )
    inner_base_metrics = _prediction_metrics(
        inner_validation_outcome,
        expit(base_inner_validation_score),
        inner_validation_weight,
    )

    support = _champion_element_support(inner_train)
    candidates: list[dict[str, Any]] = []
    candidate_cell_counts: dict[str, int] = {}
    for min_games in DIRECT_MIN_GAME_GRID:
        spec = _freeze_champion_residual_spec(
            support,
            min_games=min_games,
        )
        candidate_cell_counts[str(min_games)] = len(spec.feature_names)
        if spec.degrees_of_freedom == 0:
            continue
        for selected_lambda in DIRECT_LAMBDA_GRID:
            fit = _fit_offset_ridge(
                inner_train,
                spec,
                base_score=base_inner_train_score,
                outcome=inner_train_outcome,
                sample_weight=inner_train_weight,
                selected_lambda=selected_lambda,
            )
            validation_residual = fit.linear_score(
                inner_validation,
                spec,
            )
            for family_gate in DIRECT_GATE_GRID:
                probability = expit(
                    base_inner_validation_score
                    + float(family_gate) * validation_residual
                )
                metrics = _prediction_metrics(
                    inner_validation_outcome,
                    probability,
                    inner_validation_weight,
                )
                candidates.append(
                    {
                        "spec": spec,
                        "fit": fit,
                        "selectedLambda": float(selected_lambda),
                        "familyGate": float(family_gate),
                        "metrics": metrics,
                    }
                )
    if not candidates:
        return _empty_champion_residual_result(
            reason=(
                "No champion-element inventory cells met the inner-training "
                "support and estimability rules."
            ),
            diagnostics={
                "candidateCellCounts": candidate_cell_counts,
                "innerTrainingRows": int(len(inner_train)),
                "innerTrainingGames": int(
                    inner_train[
                        ["series_id", "game_id"]
                    ].drop_duplicates().shape[0]
                ),
            },
            observed_cells=_serialize_observed_champion_cells(
                full_support,
                min_games=min(DIRECT_MIN_GAME_GRID),
            ),
        )

    selected = min(
        candidates,
        key=lambda candidate: (
            float(candidate["metrics"]["brier"]),
            float(candidate["metrics"]["logLoss"]),
            float(candidate["metrics"]["ece10"]),
            float(candidate["familyGate"]),
            -int(candidate["spec"].selected_min_games),
            -float(candidate["selectedLambda"]),
        ),
    )
    selected_spec: ChampionResidualSpec = selected["spec"]
    selected_lambda = float(selected["selectedLambda"])
    selected_development_gate = float(selected["familyGate"])
    inner_augmented_metrics = dict(selected["metrics"])

    base_train_fit = _fit(
        train_design,
        train_outcome,
        alpha=float(selected_alpha),
        sample_weight=train_weight,
    )
    base_train_score = _fitted_logit_score(
        base_train_fit,
        train_design,
    )
    holdout_base_score = _fitted_logit_score(
        base_train_fit,
        holdout_design,
    )
    train_residual_fit = _fit_offset_ridge(
        train,
        selected_spec,
        base_score=base_train_score,
        outcome=train_outcome,
        sample_weight=train_weight,
        selected_lambda=selected_lambda,
    )
    holdout_residual = train_residual_fit.linear_score(
        holdout,
        selected_spec,
    )
    holdout_base_probability = expit(holdout_base_score)
    holdout_augmented_probability = expit(
        holdout_base_score
        + selected_development_gate * holdout_residual
    )
    holdout_base_metrics = _prediction_metrics(
        holdout_outcome,
        holdout_base_probability,
        holdout_weight,
    )
    holdout_augmented_metrics = _prediction_metrics(
        holdout_outcome,
        holdout_augmented_probability,
        holdout_weight,
    )
    delta_brier_interval = _series_block_delta_brier_interval(
        holdout,
        outcome=holdout_outcome,
        base_probability=holdout_base_probability,
        augmented_probability=holdout_augmented_probability,
        sample_weight=holdout_weight,
    )
    status, reason = _direct_family_gate(
        selected_gate=selected_development_gate,
        inner_base=inner_base_metrics,
        inner_augmented=inner_augmented_metrics,
        holdout_base=holdout_base_metrics,
        holdout_augmented=holdout_augmented_metrics,
        delta_brier_interval=delta_brier_interval,
    )

    audit_spec: ChampionResidualSpec | None = None
    publication_spec: ChampionResidualSpec | None = None
    publication_audit_diagnostics: dict[str, Any] = {
        "status": "not-run",
        "reason": (
            "The March-April family evaluation did not clear publication gates."
        ),
        "cutoff": PUBLICATION_AUDIT_START.isoformat(),
        "end": PUBLICATION_AUDIT_END.isoformat(),
    }
    if status == "ready":
        (
            audit_status,
            audit_reason,
            audit_spec,
            publication_audit_diagnostics,
        ) = _run_publication_expansion_audit(
            ordered,
            evaluation_spec=selected_spec,
            selected_alpha=float(selected_alpha),
            selected_min_games=selected_spec.selected_min_games,
            selected_lambda=selected_lambda,
            selected_gate=selected_development_gate,
        )
        if audit_status != "ready":
            status = "withheld"
            reason = audit_reason

    final_fit: ChampionResidualFit | None = None
    if status == "ready":
        publication_spec = _freeze_champion_residual_spec(
            full_support,
            min_games=selected_spec.selected_min_games,
        )
        if publication_spec.degrees_of_freedom == 0:
            status = "withheld"
            reason = (
                "Both temporal gates passed, but the full-cohort publication "
                "vocabulary has no estimable residual degrees of freedom."
            )
            publication_spec = None
        else:
            final_base_score = _fitted_logit_score(
                final_base_fit,
                full_design,
            )
            try:
                final_fit = _fit_offset_ridge(
                    ordered,
                    publication_spec,
                    base_score=final_base_score,
                    outcome=full_outcome,
                    sample_weight=full_weights,
                    selected_lambda=selected_lambda,
                )
            except ValueError as error:
                status = "withheld"
                reason = (
                    "Both temporal gates passed but the full-cohort publication "
                    f"refit failed closed: {error}"
                )
                publication_spec = None

    if status == "ready" and (
        publication_audit_diagnostics.get("status") != "ready"
        or publication_spec is None
        or final_fit is None
    ):
        status = "withheld"
        reason = (
            "Publication invariants failed: a ready family requires a passing "
            "July audit and a fitted full-cohort publication vocabulary."
        )
        publication_spec = None
        final_fit = None

    applied_gate = selected_development_gate if status == "ready" else 0.0
    audit_keys = (
        _champion_residual_cell_keys(audit_spec)
        if audit_spec is not None
        else set()
    )
    publication_keys = (
        _champion_residual_cell_keys(publication_spec)
        if publication_spec is not None
        else set()
    )
    eligible_cells: list[dict[str, Any]] = []
    if publication_spec is not None and final_fit is not None:
        for element in ELEMENTS:
            element_spec = publication_spec.elements.get(element)
            if element_spec is None:
                continue
            for cell in element_spec.support:
                feature_name = str(cell["featureName"])
                coefficient = float(
                    final_fit.raw_coefficients[feature_name]
                )
                key = (str(cell["champion"]), str(cell["element"]))
                provenance = (
                    "publication-audit-vocabulary"
                    if key in audit_keys
                    else "post-audit-full-refit"
                )
                eligible_cells.append(
                    {
                        **{
                            field: value
                            for field, value in cell.items()
                            if field != "supportWeight"
                        },
                        "supportWeight": round(
                            float(cell["supportWeight"]),
                            6,
                        ),
                        "championEligible": True,
                        "failedExposureRules": [],
                        "vocabularyProvenance": provenance,
                        "individualCellValidated": False,
                        "coefficient": coefficient,
                        "gatedCoefficient": float(
                            applied_gate * coefficient
                        ),
                    }
                )

    if status == "ready":
        serialized_keys = [
            (str(cell["champion"]), str(cell["element"]))
            for cell in eligible_cells
        ]
        coefficients_valid = all(
            math.isfinite(float(cell["coefficient"]))
            and math.isfinite(float(cell["gatedCoefficient"]))
            and math.isclose(
                float(cell["gatedCoefficient"]),
                float(applied_gate) * float(cell["coefficient"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for cell in eligible_cells
        )
        provenance_valid = all(
            cell.get("vocabularyProvenance")
            in {
                "publication-audit-vocabulary",
                "post-audit-full-refit",
            }
            and cell.get("individualCellValidated") is False
            for cell in eligible_cells
        )
        if (
            len(serialized_keys) != len(publication_keys)
            or len(set(serialized_keys)) != len(serialized_keys)
            or set(serialized_keys) != publication_keys
            or not coefficients_valid
            or not provenance_valid
        ):
            status = "withheld"
            reason = (
                "Publication invariants failed: serialized champion cells did "
                "not reconcile exactly with the fitted publication vocabulary, "
                "finite gated coefficients, and allowed provenance."
            )
            applied_gate = 0.0
            eligible_cells = []
            publication_spec = None
            final_fit = None
            publication_keys = set()

    effect_diagnostics = []
    if publication_spec is not None:
        for element in ELEMENTS:
            element_spec = publication_spec.elements.get(element)
            if element_spec is None:
                continue
            effect_diagnostics.append(
                {
                    "element": element,
                    "eligibleCells": len(element_spec.champions),
                    "constraintRows": int(element_spec.constraint.shape[0]),
                    "degreesOfFreedom": int(element_spec.basis.shape[1]),
                    "maxAbsConstraintViolation": (
                        round(
                            float(final_fit.constraint_max_abs[element]),
                            12,
                        )
                        if final_fit is not None
                        else None
                    ),
                }
            )

    added_post_audit_keys = publication_keys - audit_keys
    added_post_audit_champions = {
        champion for champion, _ in added_post_audit_keys
    }
    observed_cells = _serialize_observed_champion_cells(
        full_support,
        min_games=selected_spec.selected_min_games,
    )

    return {
        "status": status,
        "exploratory": True,
        "family": DIRECT_FAMILY,
        "familyGate": float(applied_gate),
        "selectedLambda": selected_lambda,
        "selectedMinGames": selected_spec.selected_min_games,
        "supportPolicy": _champion_residual_support_policy(),
        "effectCoding": {
            "rawFeature": (
                "post own element stacks * own champion indicator minus post "
                "opponent stacks * opponent champion indicator"
            ),
            "constraint": "B^T Q delta = 0 separately for each element",
            "basis": (
                "weighted null space of pooled intercept plus all archetype tags"
            ),
            "anchor": (
                "delta is a ridge-shrunk deviation from the frozen pooled and "
                "archetype base, not a second archetype bonus"
            ),
            "perElement": effect_diagnostics,
        },
        "diagnostics": {
            "reason": reason,
            "holdoutPrespecifiedForFamily": False,
            "selectedDevelopmentGate": selected_development_gate,
            "candidateCellCounts": candidate_cell_counts,
            "innerValidation": {
                "base": _rounded_metrics(inner_base_metrics),
                "augmented": _rounded_metrics(inner_augmented_metrics),
                "deltaBrier": round(
                    float(
                        inner_augmented_metrics["brier"]
                        - inner_base_metrics["brier"]
                    ),
                    6,
                ),
            },
            "lockedHoldout": {
                "base": _rounded_metrics(holdout_base_metrics),
                "augmented": _rounded_metrics(holdout_augmented_metrics),
                "deltaBrier": round(
                    float(
                        holdout_augmented_metrics["brier"]
                        - holdout_base_metrics["brier"]
                    ),
                    6,
                ),
                "deltaBrierSeriesBootstrap95": {
                    "lower": round(
                        float(delta_brier_interval["lower"]),
                        6,
                    ),
                    "upper": round(
                        float(delta_brier_interval["upper"]),
                        6,
                    ),
                    "iterations": int(delta_brier_interval["iterations"]),
                    "series": int(delta_brier_interval["series"]),
                    "seed": int(delta_brier_interval["seed"]),
                    "gate": (
                        "upper bound must be at most the allowed Brier regression"
                    ),
                },
                "materialRegressionLimits": {
                    "brier": DIRECT_MAX_BRIER_REGRESSION,
                    "logLoss": DIRECT_MAX_LOG_LOSS_REGRESSION,
                    "ece10": DIRECT_MAX_ECE_REGRESSION,
                },
            },
            "publicationExpansionAudit": publication_audit_diagnostics,
            "weighting": (
                "equal total weight per game, normalized independently inside "
                "each fitting and evaluation partition"
            ),
            "designFreeze": (
                "each temporal base fit uses only its training-partition feature "
                "schema and scaler; held-out rows align to that frozen schema"
            ),
            "validationScope": (
                "The temporal checks validate the residual family and publication "
                "expansion procedure, not each champion-element coefficient."
            ),
        },
        "vocabularies": {
            "evaluation": {
                **_champion_residual_vocabulary_summary(selected_spec),
                "frozenFrom": "inner-chronological-training",
            },
            "publicationAudit": (
                {
                    **_champion_residual_vocabulary_summary(audit_spec),
                    "frozenFrom": "whole-series-pre-2026-07-01",
                }
                if audit_spec is not None
                else None
            ),
            "publication": (
                {
                    **_champion_residual_vocabulary_summary(publication_spec),
                    "frozenFrom": "full-cohort-after-family-audits",
                }
                if publication_spec is not None
                else None
            ),
            "addedPostAudit": {
                "cells": int(len(added_post_audit_keys)),
                "champions": int(len(added_post_audit_champions)),
                "cellKeys": [
                    f"{champion}::{element}"
                    for champion, element in sorted(
                        added_post_audit_keys,
                        key=lambda key: (
                            ELEMENTS.index(key[1]),
                            key[0].casefold(),
                        ),
                    )
                ],
                "interpretation": (
                    "These cells first cleared the same frozen exposure rule in "
                    "the full cohort. Their coefficients are included only in "
                    "the final full-cohort refit and carry no individual "
                    "cell-validation claim."
                ),
            },
        },
        "eligibleCells": eligible_cells,
        "observedCells": observed_cells,
        "disabledFamilies": _disabled_champion_families(),
    }


def _effective_runtime(
    fit: FittedLogit,
    support_design: pd.DataFrame,
) -> dict[str, Any]:
    """Fold scaler parameters into raw weights while preserving clip behavior."""
    coefficient = np.asarray(fit.model.coef_[0], dtype=float)
    mean = np.asarray(fit.scaler.mean_, dtype=float)
    scale = np.asarray(fit.scaler.scale_, dtype=float)
    raw_weight = coefficient / scale
    intercept = float(fit.model.intercept_[0] - np.sum(raw_weight * mean))
    aligned = support_design.reindex(columns=fit.columns, fill_value=0.0)
    features = []
    for index, name in enumerate(fit.columns):
        observed = aligned[name].to_numpy(dtype=float)
        features.append(
            {
                "name": name,
                "weight": float(raw_weight[index]),
                "clipLow": float(mean[index] - STANDARDIZED_CLIP * scale[index]),
                "clipHigh": float(mean[index] + STANDARDIZED_CLIP * scale[index]),
                "observedMin": float(np.min(observed)),
                "observedMax": float(np.max(observed)),
            }
        )
    return {
        "format": "effective-raw-logit-v1",
        "intercept": intercept,
        "features": features,
        "standardizationFolded": True,
        "clipProtocol": (
            "Clamp each raw feature to clipLow/clipHigh, multiply by weight, "
            "sum with intercept, then apply logistic."
        ),
        "reconciliation": (
            "For public team/champion lines, score both mirrored perspectives "
            "and use half their logit difference before applying logistic. "
            "This enforces complementary team probabilities."
        ),
    }


def _augment_joint_runtime(
    base_runtime: Mapping[str, Any],
    champion_residual: Mapping[str, Any],
    rows: pd.DataFrame,
) -> dict[str, Any]:
    """Append validated raw residual features; withheld families change nothing."""
    runtime = copy.deepcopy(dict(base_runtime))
    if (
        str(champion_residual.get("status")) != "ready"
        or float(champion_residual.get("familyGate") or 0.0) <= 0
    ):
        return runtime
    cells = [
        dict(cell)
        for cell in champion_residual.get("eligibleCells", [])
    ]
    direct_design = _direct_cell_design(rows, cells)
    features = list(runtime.get("features", []))
    for cell in cells:
        feature_name = str(cell["featureName"])
        coefficient = cell.get("coefficient")
        gated_coefficient = cell.get("gatedCoefficient")
        if coefficient is None or gated_coefficient is None:
            raise ValueError(
                "A ready champion residual cell must include both coefficients."
            )
        observed = _number(direct_design, feature_name).to_numpy(dtype=float)
        features.append(
            {
                "name": feature_name,
                "family": DIRECT_FAMILY,
                "champion": str(cell["champion"]),
                "element": str(cell["element"]),
                "weight": float(gated_coefficient),
                "coefficientUngated": float(coefficient),
                "familyGate": float(champion_residual["familyGate"]),
                "clipLow": float(-MAX_STACKS),
                "clipHigh": float(MAX_STACKS),
                "observedMin": float(np.min(observed)),
                "observedMax": float(np.max(observed)),
            }
        )
    runtime["features"] = features
    runtime["featureFamilies"] = {
        **dict(runtime.get("featureFamilies", {})),
        DIRECT_FAMILY: {
            "status": "ready",
            "familyGate": float(champion_residual["familyGate"]),
            "featureCount": len(cells),
            "interpretation": (
                "Per-stack champion estimates, partially pooled toward the "
                "archetype prior, for supported champion-element cells."
            ),
        },
    }
    return runtime


def runtime_score(
    runtime: Mapping[str, Any],
    design: pd.DataFrame,
) -> np.ndarray:
    """Return raw logits from an exported effective runtime."""
    features = list(runtime["features"])
    if not features:
        return np.repeat(float(runtime["intercept"]), len(design))
    names = [str(feature["name"]) for feature in features]
    matrix = design.reindex(columns=names, fill_value=0.0).to_numpy(
        dtype=float,
        copy=True,
    )
    lower = np.asarray(
        [float(feature["clipLow"]) for feature in features],
        dtype=float,
    )
    upper = np.asarray(
        [float(feature["clipHigh"]) for feature in features],
        dtype=float,
    )
    np.clip(matrix, lower, upper, out=matrix)
    weights = np.asarray(
        [float(feature["weight"]) for feature in features],
        dtype=float,
    )
    return float(runtime["intercept"]) + _checked_matmul(
        matrix,
        weights,
        context="exported runtime score",
    )


def runtime_predict(
    runtime: Mapping[str, Any],
    design: pd.DataFrame,
) -> np.ndarray:
    """Evaluate an exported effective runtime (used by tests and browser parity)."""
    return expit(runtime_score(runtime, design))


def _coverage(
    games: pd.DataFrame,
    rows: pd.DataFrame,
    *,
    min_games: int,
) -> dict[str, Any]:
    valid = games.loc[
        games.apply(lambda row: _winner_is_valid(row.to_dict()), axis=1)
    ].copy()
    game_key = valid["series_id"].astype(str) + "\x1f" + valid["game_id"].astype(str)
    valid = valid.loc[~game_key.duplicated()].copy()

    def counts(column: str) -> list[dict[str, Any]]:
        if column not in valid:
            return []
        return [
            {
                column: str(value),
                "games": int(count),
                "meets6000GameSubgroupThreshold": int(count) >= min_games,
            }
            for value, count in valid[column]
            .fillna("unknown")
            .astype(str)
            .value_counts()
            .items()
        ]

    champion_counts: Counter[str] = Counter()
    composition_counts: Counter[str] = Counter()
    for game in valid.itertuples(index=False):
        for prefix in ("team_1", "team_2"):
            champions = tuple(
                sorted(
                    _json_list(getattr(game, f"{prefix}_champions")),
                    key=str.casefold,
                )
            )
            if len(champions) == 5:
                composition_counts["|".join(champions)] += 1
                champion_counts.update(champions)
    repeated = list(composition_counts.values())
    return {
        "regions": counts("region"),
        "competitionLevels": counts("competition_level"),
        "leagues": counts("league"),
        "competitions": counts("competition"),
        "championGameAppearances": {
            champion: int(champion_counts.get(champion, 0))
            for champion in CURRENT_CHAMPIONS
        },
        "exactCompositionSupport": {
            "uniqueFiveChampionSides": len(composition_counts),
            "medianGamesPerExactSide": round(float(np.median(repeated)), 2)
            if repeated
            else 0.0,
            "maximumGamesForOneExactSide": max(repeated) if repeated else 0,
            "interpretation": (
                "Exact five-champion combinations are sparse. Runtime effects "
                "are factorized through common, archetype-prior, and gated "
                "partially pooled champion-element terms, not looked up as exact-"
                "composition win rates."
            ),
        },
        "soulPerspectiveRows": int(
            (
                rows["own_soul_element_after"].notna()
                | rows["opp_soul_element_after"].notna()
            ).sum()
        ),
    }


def _champion_catalog(
    coverage: Mapping[str, Any],
    champion_residual: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    appearances = coverage.get("championGameAppearances", {})
    residual = dict(champion_residual or {})
    residual_status = str(residual.get("status") or "unavailable")
    direct_ready = (
        residual_status == "ready"
        and float(residual.get("familyGate") or 0.0) > 0
    )
    cell_lookup = {
        (str(cell["champion"]), str(cell["element"])): dict(cell)
        for cell in residual.get("eligibleCells", [])
    }
    observed_lookup = {
        (str(cell["champion"]), str(cell["element"])): dict(cell)
        for cell in residual.get("observedCells", [])
    }

    def element_evidence(champion: str) -> dict[str, dict[str, Any]]:
        tagged = bool(champ_tags(champion))
        fallback_source = (
            "archetype-fallback" if tagged else "team-common-only"
        )
        evidence: dict[str, dict[str, Any]] = {}
        for element in ELEMENTS:
            cell = cell_lookup.get((champion, element))
            observed = observed_lookup.get((champion, element))
            support = cell or observed
            champion_eligible = cell is not None
            exposure_eligible = bool(
                observed is not None
                and observed.get("championEligible")
            )
            failed_rules = list(
                (support or {}).get("failedExposureRules") or []
            )
            if champion_eligible and direct_ready:
                source = "direct-residual"
                status = "ready"
                interpretation = (
                    "Exploratory champion estimate, partially pooled toward "
                    "the common effect and archetype prior. The temporal checks "
                    "validate the family and expansion procedure, not this "
                    "champion-element coefficient by itself."
                )
            elif champion_eligible:
                source = fallback_source
                status = "withheld"
                interpretation = (
                    "The cell met exposure support, but the champion-estimate "
                    "family did not clear both publication gates; the estimate "
                    "remains archetype prior only."
                )
            elif observed is not None and failed_rules:
                source = fallback_source
                status = "below-threshold"
                interpretation = (
                    "Observed champion-element evidence is below at least one "
                    "exposure-only release threshold; the estimate remains "
                    "archetype prior only. Win/loss counts do not determine "
                    "eligibility."
                )
            elif observed is not None and exposure_eligible:
                source = fallback_source
                status = "withheld"
                interpretation = (
                    "The observed cell clears exposure thresholds, but no "
                    "published champion coefficient is available because the "
                    "family or estimable vocabulary failed closed."
                )
            else:
                source = fallback_source
                status = "unobserved"
                interpretation = (
                    "No observed pro team-game comparator for this "
                    "champion-element cell; use the archetype prior only."
                    if tagged
                    else (
                        "No observed pro team-game comparator or archetype tag; "
                        "no champion differential is assigned and the common "
                        "effect remains team-level."
                    )
                )
            evidence[element] = {
                "source": source,
                "championEligible": champion_eligible,
                "directEligible": champion_eligible,
                "exposureEligible": exposure_eligible,
                "featureName": (
                    str(support["featureName"])
                    if support is not None
                    else None
                ),
                "trainingGames": (
                    int(support["trainingGames"]) if support is not None else 0
                ),
                "trainingSeries": (
                    int(support["trainingSeries"]) if support is not None else 0
                ),
                "ownershipGames": (
                    int(support["ownershipGames"])
                    if support is not None
                    else 0
                ),
                "nonOwnershipGames": (
                    int(support["nonOwnershipGames"])
                    if support is not None
                    else 0
                ),
                "orgRosters": (
                    int(support["orgRosters"]) if support is not None else 0
                ),
                "organizations": (
                    int(
                        support.get(
                            "organizations",
                            support.get("orgRosters", 0),
                        )
                    )
                    if support is not None
                    else 0
                ),
                "wins": int(support["wins"]) if support is not None else 0,
                "losses": (
                    int(support["losses"]) if support is not None else 0
                ),
                "failedExposureRules": failed_rules,
                "vocabularyProvenance": (
                    str(cell["vocabularyProvenance"])
                    if cell is not None
                    and cell.get("vocabularyProvenance")
                    else None
                ),
                "individualCellValidated": False,
                "outcomeCountsUsedForEligibility": False,
                "status": status,
                "interpretation": interpretation,
            }
        return evidence

    return [
        {
            "name": champion,
            "tags": sorted(champ_tags(champion)),
            "proGameAppearances": int(appearances.get(champion, 0)),
            "allocationKind": "reconciled-allocation",
            "allocationSource": (
                "element-specific-direct-or-fallback"
                if any(
                    (champion, element) in cell_lookup
                    for element in ELEMENTS
                )
                and direct_ready
                else (
                    "archetype-fallback"
                    if champ_tags(champion)
                    else "team-common-only"
                )
            ),
            "fallback": "team-common-only"
            if not champ_tags(champion)
            else None,
            "elementEvidence": element_evidence(champion),
        }
        for champion in CURRENT_CHAMPIONS
    ]


def _stage_reference(rows: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "stage": int(stage),
            "medianSeconds": int(round(float(group["minute"].median() * 60))),
            "perspectiveRows": int(len(group)),
        }
        for stage, group in rows.groupby("stage", sort=True)
    ]


def _standardization_draft_rows(
    rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return one row per actual draft perspective and its mirrored row."""
    keys = ["series_id", "game_id", "perspective"]
    drafts = (
        rows.sort_values(
            ["series_id", "game_id", "perspective", "stage"],
            kind="stable",
        )
        .drop_duplicates(keys, keep="first")
        .reset_index(drop=True)
    )
    pair_keys = ["series_id", "game_id"]
    pair_sizes = drafts.groupby(pair_keys, sort=False).size()
    if pair_sizes.empty or not pair_sizes.eq(2).all():
        raise ValueError(
            "Overall dragon standardization requires two draft perspectives "
            "for every modeled game."
        )
    mirror_positions = np.empty(len(drafts), dtype=int)
    for positions in drafts.groupby(pair_keys, sort=False).indices.values():
        left, right = tuple(int(position) for position in positions)
        mirror_positions[left] = right
        mirror_positions[right] = left
    mirrors = drafts.iloc[mirror_positions].reset_index(drop=True)
    if not (
        drafts["series_id"].astype(str).to_numpy()
        == mirrors["series_id"].astype(str).to_numpy()
    ).all() or not (
        drafts["game_id"].astype(str).to_numpy()
        == mirrors["game_id"].astype(str).to_numpy()
    ).all():
        raise ValueError("Draft-perspective mirror alignment failed.")
    return drafts, mirrors


def _standardization_state(
    rows: pd.DataFrame,
    *,
    stage: int,
    minute: float,
    own_inventory: Mapping[str, int],
    opp_inventory: Mapping[str, int],
    own_soul: str | None = None,
    opp_soul: str | None = None,
) -> pd.DataFrame:
    """Apply a neutral state while preserving each row's actual two drafts."""
    changed = rows.copy(deep=True)
    for column in STATE_NUMERIC:
        changed[column] = 0.0
    changed["blue_sign"] = 0.0
    changed["roster_coverage"] = 5.0
    changed["stage"] = int(stage)
    changed["minute"] = float(minute)
    for element in ELEMENTS:
        changed[f"post_own_count_{element}"] = int(
            own_inventory.get(element, 0)
        )
        changed[f"post_opp_count_{element}"] = int(
            opp_inventory.get(element, 0)
        )
    changed["post_own_total"] = int(sum(own_inventory.values()))
    changed["post_opp_total"] = int(sum(opp_inventory.values()))
    changed["own_soul_element_after"] = own_soul
    changed["opp_soul_element_after"] = opp_soul
    return changed


def _runtime_state_design(
    runtime: Mapping[str, Any],
    rows: pd.DataFrame,
) -> pd.DataFrame:
    design = _design_state(rows)
    direct_cells = [
        {
            "champion": feature["champion"],
            "element": feature["element"],
            "featureName": feature["name"],
        }
        for feature in runtime.get("features", [])
        if feature.get("family") == DIRECT_FAMILY
    ]
    if direct_cells:
        design = pd.concat(
            [design, _direct_cell_design(rows, direct_cells)],
            axis=1,
        )
    return design


def _reconciled_runtime_probability(
    runtime: Mapping[str, Any],
    focal_rows: pd.DataFrame,
    mirror_rows: pd.DataFrame,
) -> np.ndarray:
    """Score both perspectives and enforce complementary probabilities."""
    focal_score = runtime_score(
        runtime,
        _runtime_state_design(runtime, focal_rows),
    )
    mirror_score = runtime_score(
        runtime,
        _runtime_state_design(runtime, mirror_rows),
    )
    return expit(0.5 * (focal_score - mirror_score))


class _StandardizedRuntimeScorer:
    """Fast neutral-state scorer for many inventory counterfactuals.

    The exported runtime is linear before the logistic transform.  Draft-only
    terms are therefore cached once per reference minute, while inventory,
    soul, and direct champion residual terms are cached by their small state
    blocks.  The final probability still scores the aligned mirrored draft and
    applies the exact public half-logit reconciliation.
    """

    def __init__(
        self,
        runtime: Mapping[str, Any],
        drafts: pd.DataFrame,
        mirrors: pd.DataFrame,
    ) -> None:
        self.runtime = runtime
        self.features = {
            str(feature["name"]): dict(feature)
            for feature in runtime.get("features", [])
        }
        self.direct_by_element: dict[str, list[dict[str, Any]]] = {
            element: [] for element in ELEMENTS
        }
        for feature in runtime.get("features", []):
            if feature.get("family") != DIRECT_FAMILY:
                continue
            element = str(feature.get("element") or "")
            if element in self.direct_by_element:
                self.direct_by_element[element].append(dict(feature))
        self.contexts = (
            self._context(drafts),
            self._context(mirrors),
        )
        self.row_count = len(drafts)
        if self.row_count != len(mirrors):
            raise ValueError(
                "Standardized runtime scorer requires aligned mirrored rows."
            )
        zero_score = float(runtime.get("intercept", 0.0))
        for feature in self.features.values():
            zero_score += self._clipped_zero(feature) * float(feature["weight"])
        self.zero_score = zero_score
        self.base_cache: dict[tuple[int, float], np.ndarray] = {}
        self.inventory_cache: dict[
            tuple[int, float, str, int, int],
            np.ndarray,
        ] = {}
        self.soul_cache: dict[
            tuple[int, float, str | None, str | None],
            np.ndarray,
        ] = {}
        self.probability_cache: dict[
            tuple[
                float,
                tuple[int, ...],
                tuple[int, ...],
                str | None,
                str | None,
            ],
            np.ndarray,
        ] = {}

    def _context(self, rows: pd.DataFrame) -> dict[str, Any]:
        own_sets = [
            set(_canonical_champions(value))
            for value in rows["own_champions"]
        ]
        opp_sets = [
            set(_canonical_champions(value))
            for value in rows["opp_champions"]
        ]
        direct_champions = {
            str(feature["champion"])
            for features in self.direct_by_element.values()
            for feature in features
        }
        return {
            "ownTags": {
                tag: _number(rows, f"own_{tag}").to_numpy(dtype=float)
                for tag in ARCHETYPE_NAMES
            },
            "oppTags": {
                tag: _number(rows, f"opp_{tag}").to_numpy(dtype=float)
                for tag in ARCHETYPE_NAMES
            },
            "ownChampion": {
                champion: np.fromiter(
                    (
                        champion in champions
                        for champions in own_sets
                    ),
                    dtype=float,
                    count=len(rows),
                )
                for champion in direct_champions
            },
            "oppChampion": {
                champion: np.fromiter(
                    (
                        champion in champions
                        for champions in opp_sets
                    ),
                    dtype=float,
                    count=len(rows),
                )
                for champion in direct_champions
            },
        }

    @staticmethod
    def _clipped_zero(feature: Mapping[str, Any]) -> float:
        return float(
            np.clip(
                0.0,
                float(feature["clipLow"]),
                float(feature["clipHigh"]),
            )
        )

    def _add_feature_delta(
        self,
        total: np.ndarray,
        name: str,
        values: float | np.ndarray,
    ) -> None:
        feature = self.features.get(name)
        if feature is None:
            return
        clipped = np.clip(
            values,
            float(feature["clipLow"]),
            float(feature["clipHigh"]),
        )
        total += (
            clipped - self._clipped_zero(feature)
        ) * float(feature["weight"])

    def _base_score(self, context_index: int, minute: float) -> np.ndarray:
        key = (context_index, float(minute))
        cached = self.base_cache.get(key)
        if cached is not None:
            return cached
        context = self.contexts[context_index]
        score = np.full(self.row_count, self.zero_score, dtype=float)
        for tag in ARCHETYPE_NAMES:
            difference = context["ownTags"][tag] - context["oppTags"][tag]
            self._add_feature_delta(score, f"trait_diff_{tag}", difference)
            self._add_feature_delta(
                score,
                f"trait_diff_{tag}_x_minute",
                difference * minute,
            )
        self.base_cache[key] = score
        return score

    def _inventory_contribution(
        self,
        context_index: int,
        minute: float,
        element: str,
        own_count: int,
        opp_count: int,
    ) -> np.ndarray:
        key = (
            context_index,
            float(minute),
            element,
            int(own_count),
            int(opp_count),
        )
        cached = self.inventory_cache.get(key)
        if cached is not None:
            return cached
        context = self.contexts[context_index]
        contribution = np.zeros(self.row_count, dtype=float)
        difference = float(own_count - opp_count)
        self._add_feature_delta(
            contribution,
            f"post_inventory_diff_{element}",
            difference,
        )
        self._add_feature_delta(
            contribution,
            f"post_inventory_diff_{element}_x_minute",
            difference * minute,
        )
        for tag in ARCHETYPE_NAMES:
            own_tag = context["ownTags"][tag]
            opp_tag = context["oppTags"][tag]
            self._add_feature_delta(
                contribution,
                f"post_{element}_own_trait_{tag}",
                own_count * own_tag - opp_count * opp_tag,
            )
            self._add_feature_delta(
                contribution,
                f"post_{element}_enemy_trait_{tag}",
                own_count * opp_tag - opp_count * own_tag,
            )
        for feature in self.direct_by_element[element]:
            champion = str(feature["champion"])
            self._add_feature_delta(
                contribution,
                str(feature["name"]),
                (
                    own_count * context["ownChampion"][champion]
                    - opp_count * context["oppChampion"][champion]
                ),
            )
        self.inventory_cache[key] = contribution
        return contribution

    def _soul_contribution(
        self,
        context_index: int,
        minute: float,
        own_soul: str | None,
        opp_soul: str | None,
    ) -> np.ndarray:
        key = (
            context_index,
            float(minute),
            own_soul,
            opp_soul,
        )
        cached = self.soul_cache.get(key)
        if cached is not None:
            return cached
        context = self.contexts[context_index]
        contribution = np.zeros(self.row_count, dtype=float)
        for element in ELEMENTS:
            own = float(own_soul == element)
            opp = float(opp_soul == element)
            if not own and not opp:
                continue
            difference = own - opp
            self._add_feature_delta(
                contribution,
                f"soul_after_{element}",
                difference,
            )
            self._add_feature_delta(
                contribution,
                f"soul_after_{element}_x_minute",
                difference * minute,
            )
            for tag in ARCHETYPE_NAMES:
                own_tag = context["ownTags"][tag]
                opp_tag = context["oppTags"][tag]
                self._add_feature_delta(
                    contribution,
                    f"soul_after_{element}_own_trait_{tag}",
                    own * own_tag - opp * opp_tag,
                )
                self._add_feature_delta(
                    contribution,
                    f"soul_after_{element}_enemy_trait_{tag}",
                    own * opp_tag - opp * own_tag,
                )
        self.soul_cache[key] = contribution
        return contribution

    @staticmethod
    def _inventory_tuple(inventory: Mapping[str, int]) -> tuple[int, ...]:
        counts = tuple(int(inventory.get(element, 0)) for element in ELEMENTS)
        if (
            any(count < 0 or count > MAX_STACKS for count in counts)
            or sum(counts) > MAX_STACKS
        ):
            raise ValueError("Standardized inventory is outside legal team bounds.")
        return counts

    def _state_score(
        self,
        context_index: int,
        minute: float,
        own_inventory: tuple[int, ...],
        opp_inventory: tuple[int, ...],
        own_soul: str | None,
        opp_soul: str | None,
    ) -> np.ndarray:
        score = self._base_score(context_index, minute).copy()
        for index, element in enumerate(ELEMENTS):
            own_count = own_inventory[index]
            opp_count = opp_inventory[index]
            if own_count or opp_count:
                score += self._inventory_contribution(
                    context_index,
                    minute,
                    element,
                    own_count,
                    opp_count,
                )
        if own_soul or opp_soul:
            score += self._soul_contribution(
                context_index,
                minute,
                own_soul,
                opp_soul,
            )
        return score

    def probability(
        self,
        *,
        minute: float,
        own_inventory: Mapping[str, int],
        opp_inventory: Mapping[str, int],
        own_soul: str | None = None,
        opp_soul: str | None = None,
    ) -> np.ndarray:
        own = self._inventory_tuple(own_inventory)
        opp = self._inventory_tuple(opp_inventory)
        key = (
            float(minute),
            own,
            opp,
            own_soul,
            opp_soul,
        )
        cached = self.probability_cache.get(key)
        if cached is not None:
            return cached
        focal_score = self._state_score(
            0,
            minute,
            own,
            opp,
            own_soul,
            opp_soul,
        )
        mirror_score = self._state_score(
            1,
            minute,
            opp,
            own,
            opp_soul,
            own_soul,
        )
        probability = expit(0.5 * (focal_score - mirror_score))
        self.probability_cache[key] = probability
        return probability


def _standardized_element_rankings(
    rows: pd.DataFrame,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Return comparable focal-capture effects over the observed draft mix."""
    drafts, mirrors = _standardization_draft_rows(rows)
    references = {
        int(reference["stage"]): reference
        for reference in _stage_reference(rows)
    }
    required_stages = set(range(1, 7))
    missing_stages = sorted(required_stages - set(references))
    if missing_stages:
        raise ValueError(
            "Element standardization requires reference support for stages "
            f"one through six; missing {missing_stages}."
        )
    zero = {element: 0 for element in ELEMENTS}
    scorer = _StandardizedRuntimeScorer(runtime, drafts, mirrors)
    first_minute = float(references[1]["medianSeconds"]) / 60.0
    second_minute = float(references[2]["medianSeconds"]) / 60.0
    first_pre = scorer.probability(
        minute=first_minute,
        own_inventory=zero,
        opp_inventory=zero,
    )
    legal_prior_elements = len(ELEMENTS) - 1
    first_owner_assignments = 2
    legal_second_contexts = legal_prior_elements * first_owner_assignments
    legal_opening_pairs = math.comb(len(ELEMENTS) - 1, 2)
    opening_owner_assignments = 4
    legal_map_paths = legal_opening_pairs * opening_owner_assignments
    map_path_capture_counts: list[int] = []
    rankings: list[dict[str, Any]] = []
    for element in ELEMENTS:
        first_inventory = dict(zero)
        first_inventory[element] = 1
        first_post = scorer.probability(
            minute=first_minute,
            own_inventory=first_inventory,
            opp_inventory=zero,
        )
        first_delta = first_post - first_pre

        second_deltas: list[np.ndarray] = []
        for prior_element in (
            candidate for candidate in ELEMENTS if candidate != element
        ):
            for focal_owned_first in (False, True):
                own_pre = dict(zero)
                opp_pre = dict(zero)
                (
                    own_pre
                    if focal_owned_first
                    else opp_pre
                )[prior_element] = 1
                own_post = dict(own_pre)
                own_post[element] += 1
                pre_probability = scorer.probability(
                    minute=second_minute,
                    own_inventory=own_pre,
                    opp_inventory=opp_pre,
                )
                post_probability = scorer.probability(
                    minute=second_minute,
                    own_inventory=own_post,
                    opp_inventory=opp_pre,
                )
                second_deltas.append(post_probability - pre_probability)
        if len(second_deltas) != legal_second_contexts:
            raise ValueError(
                "Unexpected number of legal second-capture contexts."
            )

        map_path_deltas: list[np.ndarray] = []
        element_path_capture_counts: list[int] = []
        openings = [
            candidate for candidate in ELEMENTS if candidate != element
        ]
        for left, right in combinations(openings, 2):
            for left_focal in (False, True):
                for right_focal in (False, True):
                    own_inventory = dict(zero)
                    opp_inventory = dict(zero)
                    (
                        own_inventory
                        if left_focal
                        else opp_inventory
                    )[left] = 1
                    (
                        own_inventory
                        if right_focal
                        else opp_inventory
                    )[right] = 1
                    path_increments: list[np.ndarray] = []
                    map_capture_index = 0
                    while sum(own_inventory.values()) < MAX_STACKS:
                        map_capture_index += 1
                        stage = 2 + map_capture_index
                        if stage > 6:
                            raise ValueError(
                                "A legal focal map route must end by stage six."
                            )
                        if (
                            sum(own_inventory.values())
                            + sum(opp_inventory.values())
                            != stage - 1
                        ):
                            raise ValueError(
                                "Map-route pre-state is not a legal capture prefix."
                            )
                        minute = (
                            float(references[stage]["medianSeconds"]) / 60.0
                        )
                        pre_probability = scorer.probability(
                            minute=minute,
                            own_inventory=own_inventory,
                            opp_inventory=opp_inventory,
                        )
                        post_inventory = dict(own_inventory)
                        post_inventory[element] += 1
                        has_soul = (
                            sum(post_inventory.values()) == MAX_STACKS
                        )
                        if (
                            sum(post_inventory.values())
                            + sum(opp_inventory.values())
                            != stage
                        ):
                            raise ValueError(
                                "Map-route post-state is not a legal capture state."
                            )
                        post_probability = scorer.probability(
                            minute=minute,
                            own_inventory=post_inventory,
                            opp_inventory=opp_inventory,
                            own_soul=element if has_soul else None,
                        )
                        path_increments.append(
                            post_probability - pre_probability
                        )
                        own_inventory = post_inventory
                    if not path_increments:
                        raise ValueError(
                            "A legal map route must include at least one capture."
                        )
                    element_path_capture_counts.append(len(path_increments))
                    map_path_deltas.append(
                        np.mean(np.vstack(path_increments), axis=0)
                    )
        if len(map_path_deltas) != legal_map_paths:
            raise ValueError("Unexpected number of legal map-phase paths.")
        if not map_path_capture_counts:
            map_path_capture_counts = element_path_capture_counts
        elif map_path_capture_counts != element_path_capture_counts:
            raise ValueError(
                "Map-phase path lengths changed across target elements."
            )
        rankings.append(
            {
                "element": element,
                "firstCapturePp": round(
                    float(np.mean(first_delta) * 100),
                    6,
                ),
                "secondCapturePp": round(
                    float(np.mean(np.vstack(second_deltas)) * 100),
                    6,
                ),
                "mapPhaseCapturePp": round(
                    float(np.mean(np.vstack(map_path_deltas)) * 100),
                    6,
                ),
            }
        )
    rankings.sort(
        key=lambda ranking: (
            -float(ranking["mapPhaseCapturePp"]),
            -float(ranking["secondCapturePp"]),
            -float(ranking["firstCapturePp"]),
        )
    )
    direct_feature_count = sum(
        feature.get("family") == DIRECT_FAMILY
        for feature in runtime.get("features", [])
    )
    captured_rows = rows.loc[rows["took_current"] == 1].copy()
    soul_rows = captured_rows.loc[
        captured_rows["own_soul_element_after"].notna()
    ].copy()

    def element_counts(frame: pd.DataFrame, column: str) -> dict[str, int]:
        counts = frame[column].astype(str).value_counts()
        return {
            element: int(counts.get(element, 0))
            for element in ELEMENTS
        }

    first_support = captured_rows.loc[captured_rows["stage"] == 1]
    second_support = captured_rows.loc[captured_rows["stage"] == 2]
    map_support = captured_rows.loc[captured_rows["stage"] >= 3]
    path_length_counts = Counter(map_path_capture_counts)
    return {
        "estimand": (
            "Adjusted associational focal-capture map-win changes standardized "
            "over both actual draft perspectives from every modeled game."
        ),
        "estimands": {
            "firstCapturePp": (
                "Focal side receives the first global capture of the target "
                "element versus the identical zero-inventory pre-state at the "
                "stage-one median time."
            ),
            "secondCapturePp": (
                "Focal side receives the second global capture of the target "
                "element versus the identical legal one-capture pre-state at "
                "the stage-two median time, averaged equally over five legal "
                "first elements and both first-capture owner assignments."
            ),
            "mapPhaseCapturePp": (
                "Mean single-capture change while the focal side takes each "
                "repeated target-element map dragon from global capture three "
                "until its fourth total stack and soul. Each post-state is "
                "compared with its identical pre-state at that capture-stage "
                "median; the soul term enters only on the final increment. "
                "Increments are averaged within each path, then equally over "
                "legal opening states and actual draft perspectives."
            ),
        },
        "unit": "percentage points of modeled map-win probability",
        "rankings": rankings,
        "support": {
            "modeledGames": int(
                drafts[["series_id", "game_id"]].drop_duplicates().shape[0]
            ),
            "actualDraftPerspectives": int(len(drafts)),
            "resolvedCaptures": int(len(captured_rows)),
            "soulCaptures": int(len(soul_rows)),
            "stageReferencePerspectiveRows": {
                str(stage): int(references[stage]["perspectiveRows"])
                for stage in sorted(required_stages)
            },
            "observedFirstCapturesByElement": element_counts(
                first_support,
                "current_element",
            ),
            "observedSecondCapturesByElement": element_counts(
                second_support,
                "current_element",
            ),
            "observedMapPhaseCapturesByElement": element_counts(
                map_support,
                "current_element",
            ),
            "observedSoulCapturesByElement": element_counts(
                soul_rows,
                "own_soul_element_after",
            ),
            "legalFirstContextsPerElement": 1,
            "legalSecondContextsPerElement": legal_second_contexts,
            "legalOpeningPairsPerMapElement": legal_opening_pairs,
            "openingOwnerAssignmentsPerPair": opening_owner_assignments,
            "legalMapPathsPerElement": legal_map_paths,
            "mapCaptureIncrementsPerElement": int(
                sum(map_path_capture_counts)
            ),
            "mapPathCountsByCaptureLength": {
                str(length): int(count)
                for length, count in sorted(path_length_counts.items())
            },
            "championResidualApplied": bool(direct_feature_count),
            "championResidualFeatureCount": int(direct_feature_count),
        },
        "weighting": {
            "draftPerspectives": "equal",
            "modeledGames": "two actual team perspectives per modeled game",
            "firstCaptureContexts": "one legal zero-inventory pre-state",
            "secondCaptureContexts": (
                f"equal across {legal_second_contexts} legal first-element "
                "and first-owner contexts"
            ),
            "mapOpeningPaths": (
                f"equal across {legal_map_paths} legal unordered opening-pair "
                "and opening-owner paths for each map element"
            ),
            "mapCapturesWithinPath": (
                "equal within each path before paths receive equal weight"
            ),
        },
        "reference": {
            "state": (
                "neutral signed game state and side with each modeled game's "
                "actual two-team drafts"
            ),
            "comparison": (
                "post-capture probability minus the identical pre-capture "
                "inventory probability at the same stage median"
            ),
            "firstCapture": {
                "stage": 1,
                "medianSeconds": int(references[1]["medianSeconds"]),
                "preInventory": "0/0",
                "postInventory": "focal side adds one target-element stack",
            },
            "secondCapture": {
                "stage": 2,
                "medianSeconds": int(references[2]["medianSeconds"]),
                "legalPreStates": (
                    "one non-target opening element owned by either side"
                ),
                "postInventory": "focal side adds one target-element stack",
            },
            "mapPhaseCapture": {
                "startsAtGlobalCapture": 3,
                "possibleEndStages": [4, 5, 6],
                "stageMedianSeconds": {
                    str(stage): int(references[stage]["medianSeconds"])
                    for stage in range(3, 7)
                },
                "openingElements": (
                    "two distinct non-target elements; order is irrelevant to "
                    "the joint inventory model"
                ),
                "openingOwners": "all four focal/opponent assignments",
                "mapRoute": (
                    "focal side receives each repeated target element until "
                    "its fourth total stack; the final capture adds target soul"
                ),
            },
        },
        "reconciliation": (
            "Score both mirrored perspectives and apply logistic to half their "
            "logit difference before standardization."
        ),
        "ordering": (
            "Descending mapPhaseCapturePp, then secondCapturePp, then "
            "firstCapturePp."
        ),
        "pointEstimateCaveat": (
            "These ranks order fitted point estimates only. The artifact does "
            "not provide effect-level confidence intervals or rank "
            "probabilities, so small gaps do not establish that one element is "
            "statistically better."
        ),
    }


def _feature_schema(
    rows: pd.DataFrame,
    champion_residual: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    residual = dict(champion_residual or {})
    direct_status = str(residual.get("status") or "unavailable")
    direct_gate = float(residual.get("familyGate") or 0.0)
    direct_ready = direct_status == "ready" and direct_gate > 0
    categories = {}
    for column in (
        "competition",
        "league",
        "region",
        "competition_level",
        "patch",
        "year",
    ):
        categories[column] = sorted(rows[column].astype(str).unique().tolist())
    return {
        "version": FEATURE_BUILDER_VERSION,
        "elements": list(ELEMENTS),
        "archetypes": list(ARCHETYPE_NAMES),
        "stateInputs": list(STATE_NUMERIC),
        "stateInputUnits": {
            "gold_diff_k": "perspective minus opponent, thousands of gold",
            "loadout_diff_k": (
                "perspective minus opponent loadout value, thousands of gold"
            ),
            "unspent_money_diff_k": (
                "perspective minus opponent unspent money, thousands of gold"
            ),
            "top_player_net_worth_diff_k": (
                "perspective minus opponent leading-player net worth, "
                "thousands of gold"
            ),
            "tower_diff": "perspective minus opponent tower count",
            "org_elo_diff": "perspective minus opponent organization Elo, divided by 400",
            "player_elo_diff": (
                "perspective minus opponent five-player aggregate Elo, divided by 400"
            ),
        },
        "captureInputs": {
            "currentElement": "one elemental id for modeled stages one through seven",
            "stage": (
                "integer one through seven; stage zero is an unscored display baseline"
            ),
            "minute": "capture/reference time in minutes",
            "tookCurrent": (
                "one when the perspective team receives the current capture, "
                "zero when the opposing team receives it"
            ),
        },
        "inventoryInputs": {
            "pre": [
                f"pre_{side}_count_{element}"
                for side in ("own", "opp")
                for element in ELEMENTS
            ],
            "post": [
                f"post_{side}_count_{element}"
                for side in ("own", "opp")
                for element in ELEMENTS
            ],
            "maximumPerTeam": MAX_STACKS,
            "maximumGlobalStage": MAX_GLOBAL_STAGE,
        },
        "legalStateRules": [
            "All element counts are non-negative integers.",
            "Each team total is at most four and the two totals sum to stage.",
            "At most one team has soul; soul is present exactly when that team total reaches four.",
            "An ordered path must follow distinct first and second elements, then one repeated rift element for later elemental spawns.",
            "No elemental capture or modeled stage follows soul.",
        ],
        "soulInputs": [
            "own_soul_element_after",
            "opp_soul_element_after",
        ],
        "categories": categories,
        "categoryEncoding": (
            "Selected category indicators are multiplied by blue_sign. A "
            "neutral side baseline uses blue_sign=0, so category terms are zero."
        ),
        "categoryFeatureName": "{field}={value}_x_blue",
        "compositionEncoding": (
            "Tagged champions contribute hand-authored archetype indicators as "
            "the prior basis. A support-eligible champion-element inventory "
            "cell can use a partially pooled, archetype-anchored champion "
            "estimate from a temporally audited family. "
            "Untagged unsupported champions receive no champion differential; "
            "the common effect remains team-level. No exact-composition lookup "
            "is used."
        ),
        "championAllocation": {
            "kind": "reconciled-allocation",
            "taggedSource": "archetype-fallback",
            "untaggedSource": "team-common-only",
            "championSpecificEmpiricalEvidence": direct_ready,
            "separatelyFittedChampionEffects": False,
            "jointlyFittedChampionResiduals": direct_ready,
            "directResidualFamily": {
                "family": DIRECT_FAMILY,
                "status": direct_status,
                "familyGate": direct_gate,
                "sourceWhenReady": "direct-residual",
                "fallbackWhenWithheld": (
                    "archetype-prior-only for tagged cells; team-common-only otherwise"
                ),
                "interpretation": (
                    "A champion estimate is a regularized, partially pooled "
                    "element-specific deviation from the common effect and "
                    "archetype prior, not an additional archetype bonus."
                ),
            },
        },
        "disabledChampionFamilies": _disabled_champion_families(),
        "genericDraftScoreContext": {
            "appliedToDragonEstimate": False,
            "championEffectsApplied": False,
            "allySynergyApplied": False,
            "enemyCounterApplied": False,
            "reason": (
                "Generic Draft Score coefficients estimate a different "
                "pre-match composition association and are not evidence of an "
                "element-specific champion-by-dragon response."
            ),
        },
        "soulEncoding": (
            "Soul has element-specific signed main effects plus own-team and "
            "enemy-team archetype interactions; there are no exact "
            "champion-by-soul coefficients."
        ),
        "featureFamilies": {
            "signedState": "perspective value minus opposing value",
            "traitDifference": "own archetype count minus opposing archetype count",
            "inventoryDifference": "own element count minus opposing element count",
            "inventoryOwnTrait": (
                "own element count * own trait count - opposing element count "
                "* opposing trait count"
            ),
            "inventoryEnemyTrait": (
                "own element count * opposing trait count - opposing element "
                "count * own trait count"
            ),
            "championDirectInventoryResidual": (
                "For eligible cells only: own post-inventory stacks times own "
                "champion identity minus opposing stacks times opposing champion "
                "identity, effect-coded against pooled and archetype bases."
            ),
            "soulMain": (
                "1[own soul is element] - 1[opposing soul is element]"
            ),
            "soulOwnTrait": (
                "1[own soul is element] * own trait count - 1[opposing soul is "
                "element] * opposing trait count"
            ),
            "soulEnemyTrait": (
                "1[own soul is element] * opposing trait count - 1[opposing "
                "soul is element] * own trait count"
            ),
            "allocationDirection": "2 * took_current - 1",
            "allocationOwnTrait": (
                "1[current element] * (took_current * own trait count - "
                "(1 - took_current) * opposing trait count)"
            ),
            "allocationEnemyTrait": (
                "1[current element] * (took_current * opposing trait count - "
                "(1 - took_current) * own trait count)"
            ),
            "time": (
                "minute enters through signed side, inventory, soul, trait, "
                "and allocation interactions"
            ),
        },
        "signedUtilityConvention": (
            "Swapping the two teams negates every generated feature. Runtime "
            "consumers score both perspectives and reconcile their logits, so "
            "the two team probabilities sum to one."
        ),
        "neutralReference": {
            "signedState": {name: 0.0 for name in STATE_NUMERIC},
            "organizationStrengthDifference": 0.0,
            "playerStrengthDifference": 0.0,
            "blueSign": 0.0,
            "categoryTerms": 0.0,
            "archetypes": "derived from the two selected five-champion teams",
            "time": "use the selected stage medianSeconds",
        },
    }


def _quality_passes(diagnostics: Mapping[str, Any]) -> bool:
    return (
        float(diagnostics["brier"]) < float(diagnostics["nullBrier"])
        and float(diagnostics["ece10"]) <= 0.10
    )


def build_explorer_artifact(
    *,
    games_path: Path = COMPACT_GAMES_PARQUET,
    events_path: Path = COMPACT_EVENTS_PARQUET,
    min_games: int = MIN_PUBLIC_GAMES,
) -> dict[str, Any]:
    source_provenance = {
        "schemaVersion": SCHEMA_VERSION,
        "featureBuilderVersion": FEATURE_BUILDER_VERSION,
        "inputs": {
            "games": _file_provenance(games_path),
            "events": _file_provenance(events_path),
            "championTaxonomy": _file_provenance(
                CHAMPION_TAXONOMY_PATH
            ),
        },
    }
    games = pd.read_parquet(games_path)
    valid_games = games.loc[
        games.apply(lambda row: _winner_is_valid(row.to_dict()), axis=1),
        ["series_id", "game_id"],
    ].drop_duplicates()
    completed_games = int(len(valid_games))
    if completed_games < min_games:
        return {
            "schemaVersion": SCHEMA_VERSION,
            "status": "gated",
            "games": completed_games,
            "requiredGames": min_games,
            "reason": "The prespecified professional-game threshold has not been reached.",
            "provenance": source_provenance,
            "publicWording": PUBLIC_WORDING,
        }
    events = pd.read_parquet(events_path)
    rows = prepare_joint_rows(games, events)
    modeled_games = int(rows[["series_id", "game_id"]].drop_duplicates().shape[0])
    if modeled_games < int(min_games * 0.8):
        raise RuntimeError(
            f"Only {modeled_games} games retained usable mirrored capture rows "
            f"from {completed_games} completed games."
        )

    state_diagnostics, state_fit = _diagnostics(rows, _design_state)
    allocation_diagnostics, allocation_fit = _diagnostics(
        rows,
        _design_allocation,
    )
    validation = {
        "jointStateBrierBeatsNull": (
            state_diagnostics["brier"] < state_diagnostics["nullBrier"]
        ),
        "jointStateEceAtMostTenPp": state_diagnostics["ece10"] <= 0.10,
        "captureAllocationBrierBeatsNull": (
            allocation_diagnostics["brier"]
            < allocation_diagnostics["nullBrier"]
        ),
        "captureAllocationEceAtMostTenPp": (
            allocation_diagnostics["ece10"] <= 0.10
        ),
    }
    if not all(validation.values()):
        return {
            "schemaVersion": SCHEMA_VERSION,
            "status": "gated",
            "games": completed_games,
            "requiredGames": min_games,
            "reason": (
                "The sample threshold passed, but at least one prespecified "
                "chronological holdout gate failed. Explorer estimates remain hidden."
            ),
            "validation": validation,
            "diagnostics": {
                "jointState": state_diagnostics,
                "captureAllocation": allocation_diagnostics,
            },
            "provenance": {
                **source_provenance,
                "cohort": {
                    "dateMin": rows["date"].min().isoformat(),
                    "dateMax": rows["date"].max().isoformat(),
                },
            },
            "publicWording": PUBLIC_WORDING,
        }

    champion_residual = _fit_champion_residual_family(
        rows,
        selected_alpha=float(state_diagnostics["selectedAlpha"]),
        final_base_fit=state_fit,
    )
    validation["championDirectResidualStatus"] = champion_residual["status"]
    validation["championDirectResidualPublished"] = (
        champion_residual["status"] == "ready"
        and float(champion_residual["familyGate"]) > 0
    )
    state_design = _design_state(rows)
    allocation_design = _design_allocation(rows)
    joint_runtime = _augment_joint_runtime(
        _effective_runtime(state_fit, state_design),
        champion_residual,
        rows,
    )
    overall_element_rankings = _standardized_element_rankings(
        rows,
        joint_runtime,
    )
    coverage = _coverage(games, rows, min_games=min_games)
    captured_rows = rows.loc[rows["took_current"] == 1].copy()
    allocation_selection = {
        "resolvedCaptures": int(len(captured_rows)),
        "rawCapturerWinRate": round(
            float(captured_rows["perspective_won"].mean()),
            4,
        ),
        "stateLagSeconds": {
            "median": round(float(captured_rows["state_lag_seconds"].median()), 3),
            "p95": round(
                float(captured_rows["state_lag_seconds"].quantile(0.95)),
                3,
            ),
            "maximumAllowed": 60,
        },
        "byStage": [
            {
                "stage": int(stage),
                "captures": int(len(group)),
                "rawCapturerWinRate": round(
                    float(group["perspective_won"].mean()),
                    4,
                ),
            }
            for stage, group in captured_rows.groupby("stage", sort=True)
        ],
        "interpretation": (
            "The team that ultimately secures a dragon is already selected by game "
            "state, setup, contest outcome, and strategic concession. The raw rate "
            "is descriptive evidence of that selection, not dragon value."
        ),
    }
    publication_audit = champion_residual.get("diagnostics", {}).get(
        "publicationExpansionAudit",
        {},
    )
    artifact = {
        "schemaVersion": SCHEMA_VERSION,
        "status": "ready",
        "provenance": {
            **source_provenance,
            "cohort": {
                "dateMin": rows["date"].min().isoformat(),
                "dateMax": rows["date"].max().isoformat(),
            },
            "temporalSplits": {
                "marchAprilEvaluation": {
                    "plannedStart": PUBLIC_HOLDOUT_START.isoformat(),
                    "plannedEnd": PUBLIC_HOLDOUT_END.isoformat(),
                    "actualStart": state_diagnostics.get(
                        "holdoutStart"
                    ),
                    "actualEnd": state_diagnostics.get(
                        "holdoutActualEnd"
                    ),
                    "seriesSets": state_diagnostics.get(
                        "seriesSets",
                        {},
                    ),
                },
                "publicationExpansionAudit": {
                    "plannedStart": PUBLICATION_AUDIT_START.isoformat(),
                    "plannedEnd": PUBLICATION_AUDIT_END.isoformat(),
                    "actualStart": publication_audit.get(
                        "holdoutStart"
                    ),
                    "actualEnd": publication_audit.get("holdoutEnd"),
                    "seriesSets": publication_audit.get(
                        "seriesSets",
                        {},
                    ),
                },
            },
        },
        "cohort": {
            "completedGames": completed_games,
            "modeledGames": modeled_games,
            "series": int(rows["series_id"].nunique()),
            "captures": int(len(rows) // 2),
            "mirroredPerspectiveRows": int(len(rows)),
            "dateMin": rows["date"].min().isoformat(),
            "dateMax": rows["date"].max().isoformat(),
            "tierAndRegionCoverage": {
                "regions": coverage["regions"],
                "competitionLevels": coverage["competitionLevels"],
                "leagues": coverage["leagues"],
                "competitions": coverage["competitions"],
            },
        },
        "featureSchema": _feature_schema(rows, champion_residual),
        "championCatalog": _champion_catalog(
            coverage,
            champion_residual,
        ),
        "stageReference": _stage_reference(rows),
        "support": {
            "exactComposition": coverage["exactCompositionSupport"],
            "soulPerspectiveRows": coverage["soulPerspectiveRows"],
            "championElement": {
                "status": champion_residual["status"],
                "eligibleCells": len(champion_residual["eligibleCells"]),
                "observedCells": len(
                    champion_residual.get("observedCells", [])
                ),
                "selectedMinGames": champion_residual["selectedMinGames"],
                "supportPolicy": champion_residual["supportPolicy"],
                "vocabularies": champion_residual.get("vocabularies", {}),
            },
            "featureRanges": {
                "minute": [
                    round(float(rows["minute"].min()), 3),
                    round(float(rows["minute"].max()), 3),
                ],
                "goldDifferenceThousands": [
                    round(float(rows["gold_diff_k"].min()), 3),
                    round(float(rows["gold_diff_k"].max()), 3),
                ],
                "stage": [
                    int(rows["stage"].min()),
                    int(rows["stage"].max()),
                ],
            },
            "outOfDistributionRule": (
                "Flag inputs outside observed runtime feature ranges or category "
                "support. Predictions still use the documented eight-standard-"
                "deviation clamp, but should not be described as observed evidence."
            ),
        },
        "models": {
            "jointState": {
                "estimand": (
                    "Adjusted map-win association from the complete joint "
                    "post-capture elemental inventory and soul state."
                ),
                "diagnostics": state_diagnostics,
                "runtime": joint_runtime,
                "overallElementRankings": overall_element_rankings,
                "championResidual": champion_residual,
                "referenceProtocol": (
                    "The public delta compares a legal inventory with an all-zero "
                    "inventory at the same median stage time. The later-stage zero "
                    "inventory is impossible in play and is therefore an explicit "
                    "synthetic extrapolation, not an observed control state."
                ),
            },
            "captureAllocation": {
                "estimand": (
                    "Adjusted map-win association comparing which side received "
                    "an already-resolved elemental capture at the same measured "
                    "pre-capture state and inventory."
                ),
                "diagnostics": allocation_diagnostics,
                "runtime": _effective_runtime(
                    allocation_fit,
                    allocation_design,
                ),
                "counterfactualProtocol": (
                    "Hold pre-capture state, both compositions, time, patch, "
                    "competition, and pre-inventories fixed; switch took_current "
                    "and recompute only the deterministic post-inventory and soul."
                ),
                "selectionDiagnostics": allocation_selection,
            },
        },
        "validation": validation,
        "publicWording": PUBLIC_WORDING,
        "controls": [
            "pre-capture net worth, loadout value, unspent money, leading-player net worth, towers, and side",
            "both teams' five-champion archetype counts",
            "both teams' elemental inventories and soul",
            "prior-game organization and five-player aggregate strength",
            "capture time and global stage",
            "exact patch, canonical competition, league, region, level, and calendar year",
            "regularization selected inside pre-March 2026 training data",
            "champion exposure threshold, ridge penalty, family gate, and evaluation vocabulary selected inside the pre-March inner training and validation partitions",
            "locked March-April 2026 family evaluation",
            "expanded pre-July exposure-only vocabulary fitted before the cutoff and audited on available whole July 2026 series through the reported actual holdout end against both the base and original evaluation-vocabulary models",
            "full-cohort publication vocabulary and coefficients refitted only after both temporal family gates pass",
            "equal total fitting and evaluation weight per game",
        ],
        "limitations": [
            PUBLIC_WORDING["allocation"],
            PUBLIC_WORDING["champions"],
            PUBLIC_WORDING["lines"],
            PUBLIC_WORDING["draftContext"],
            PUBLIC_WORDING["directResidual"],
            "GRID state is the latest observed event state before capture and may lag by up to 60 seconds.",
            "May-June 2026 games extend the frozen pre-July publication-audit fit; available whole July 2026 series through the reported actual holdout end evaluate that expanded vocabulary before the final full-cohort refit.",
            "Region and competition-level coverage is reported separately; no subgroup with fewer than 6,000 games is presented as independently normalized.",
        ],
    }
    return artifact


def build_model_artifact(
    *,
    games_path: Path = COMPACT_GAMES_PARQUET,
    events_path: Path = COMPACT_EVENTS_PARQUET,
    min_games: int = MIN_PUBLIC_GAMES,
) -> dict[str, Any]:
    """Compatibility entrypoint matching the earlier elemental model."""
    return build_explorer_artifact(
        games_path=games_path,
        events_path=events_path,
        min_games=min_games,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, default=COMPACT_GAMES_PARQUET)
    parser.add_argument("--events", type=Path, default=COMPACT_EVENTS_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-games", type=int, default=MIN_PUBLIC_GAMES)
    args = parser.parse_args(argv)
    artifact = build_explorer_artifact(
        games_path=args.games,
        events_path=args.events,
        min_games=args.min_games,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        "[elemental-drake-explorer] "
        f"status={artifact['status']} "
        f"games={artifact.get('cohort', {}).get('completedGames', artifact.get('games', 0))} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
