#!/usr/bin/env python3
"""Estimate a transparent pre-fight win probability from cached ranked timelines.

The target is deliberately narrow: among event-centred Void Grub engagements
with a non-zero local champion-kill differential, estimate the probability that
a focal team finishes the engagement with more local kills than its opponent.
Only information recorded before the first grub kill is used as a predictor.

Each engagement contributes two mirrored team perspectives.  Mirrored rows are
kept in the same cross-validation fold, preventing the complementary outcome of
one engagement from leaking into its held-out prediction.  The primary model is
the one-variable symmetric logit

    logit(P(focal team wins local kill exchange)) = beta * gold_lead / 1000.

This is a ranked-timeline pilot, not a professional-data estimate and not a
causal effect of choosing to contest.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

from lol_kills.etl.paths import MODELS_DIR
from lol_kills.etl.riot_timelines import TIMELINE_DIR
from lol_kills.research.grubs_intrinsic_value import contest_ev_terminal_states

OUT_JSON = MODELS_DIR / "grubs_fight_probability.json"

HORDE = "HORDE"
PRIMARY_RADIUS = 2200.0
RADIUS_SENSITIVITY = (1500.0, 2200.0, 3000.0)
WINDOW_BEFORE_MS = 20_000
WINDOW_AFTER_MS = 35_000
PREDICTION_LEAD_MS = 30_000
MAX_HORDE_GAP_MS = 45_000
MAX_EPISODE_AFTER_FIRST_MS = 90_000
FIRST_GRUB_MIN_MS = 7 * 60_000
FIRST_GRUB_MAX_MS = 11 * 60_000


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _team_for_participant(participant_id: int) -> int | None:
    if 1 <= participant_id <= 5:
        return 100
    if 6 <= participant_id <= 10:
        return 200
    return None


def _event_position(event: dict[str, Any]) -> tuple[float, float] | None:
    pos = event.get("position") or {}
    if pos.get("x") is None or pos.get("y") is None:
        return None
    return float(pos["x"]), float(pos["y"])


def _frame_team_state(frame: dict[str, Any], pit: tuple[float, float]) -> dict[int, dict[str, float]]:
    state = {
        100: {"gold": 0.0, "near": 0.0},
        200: {"gold": 0.0, "near": 0.0},
    }
    for raw_pid, pdata in (frame.get("participantFrames") or {}).items():
        try:
            pid = int(raw_pid)
        except (TypeError, ValueError):
            continue
        team = _team_for_participant(pid)
        if team is None:
            continue
        pdata = pdata or {}
        state[team]["gold"] += float(pdata.get("totalGold") or 0.0)
        pos = pdata.get("position") or {}
        if pos.get("x") is not None and pos.get("y") is not None:
            if _distance((float(pos["x"]), float(pos["y"])), pit) <= PRIMARY_RADIUS:
                state[team]["near"] += 1.0
    return state


def _extract_episode(path: Path) -> dict[str, Any] | None:
    try:
        timeline = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    frames = (timeline.get("info") or timeline).get("frames") or []
    horde: list[dict[str, Any]] = []
    champion_kills: list[dict[str, Any]] = []
    for frame in frames:
        for event in frame.get("events") or []:
            event_type = event.get("type")
            if event_type == "ELITE_MONSTER_KILL" and str(event.get("monsterType", "")).upper() == HORDE:
                horde.append(event)
            elif event_type == "CHAMPION_KILL":
                champion_kills.append(event)
    if not horde or len(horde) > 3:
        return None
    horde.sort(key=lambda event: int(event.get("timestamp") or 0))
    first_ts = int(horde[0].get("timestamp") or 0)
    if not (FIRST_GRUB_MIN_MS <= first_ts <= FIRST_GRUB_MAX_MS):
        return None
    cluster = [horde[0]]
    for event in horde[1:]:
        event_ts = int(event.get("timestamp") or 0)
        prior_ts = int(cluster[-1].get("timestamp") or first_ts)
        if event_ts - prior_ts > MAX_HORDE_GAP_MS:
            break
        cluster.append(event)
    last_ts = int(cluster[-1].get("timestamp") or first_ts)
    episode_end = min(last_ts + WINDOW_AFTER_MS, first_ts + MAX_EPISODE_AFTER_FIRST_MS)
    pit_positions = [position for event in cluster if (position := _event_position(event)) is not None]
    if not pit_positions:
        return None
    pit = (
        float(np.median([position[0] for position in pit_positions])),
        float(np.median([position[1] for position in pit_positions])),
    )
    first_killer = int(horde[0].get("killerId") or 0)
    first_secure_team = _team_for_participant(first_killer)
    if first_secure_team is None:
        return None

    target_ts = first_ts - PREDICTION_LEAD_MS
    eligible_frames = [frame for frame in frames if int(frame.get("timestamp") or 0) <= target_ts]
    if not eligible_frames:
        return None
    pre_frame = max(eligible_frames, key=lambda frame: int(frame.get("timestamp") or 0))
    pre_state = _frame_team_state(pre_frame, pit)
    if pre_state[100]["gold"] <= 0 or pre_state[200]["gold"] <= 0:
        return None

    kill_counts_by_radius = {
        int(radius): {100: 0, 200: 0}
        for radius in RADIUS_SENSITIVITY
    }
    for event in champion_kills:
        timestamp = int(event.get("timestamp") or 0)
        if timestamp < first_ts - WINDOW_BEFORE_MS or timestamp > episode_end:
            continue
        position = _event_position(event)
        if position is None:
            continue
        killer_team = _team_for_participant(int(event.get("killerId") or 0))
        if killer_team is None:
            continue
        distance = _distance(position, pit)
        for radius in RADIUS_SENSITIVITY:
            if distance <= radius:
                kill_counts_by_radius[int(radius)][killer_team] += 1
    primary_counts = kill_counts_by_radius[int(PRIMARY_RADIUS)]
    winner = None
    if primary_counts[100] != primary_counts[200]:
        winner = 100 if primary_counts[100] > primary_counts[200] else 200
    match_id = str((timeline.get("metadata") or {}).get("matchId") or path.name.removesuffix(".json"))
    return {
        "match_id": match_id,
        "first_grub_ms": first_ts,
        "last_grub_ms": last_ts,
        "pre_frame_ms": int(pre_frame.get("timestamp") or 0),
        "first_secure_team": first_secure_team,
        "winner_team": winner,
        "kill_counts_by_radius": kill_counts_by_radius,
        "blue_gold": pre_state[100]["gold"],
        "red_gold": pre_state[200]["gold"],
        "blue_near": int(pre_state[100]["near"]),
        "red_near": int(pre_state[200]["near"]),
    }


def _oriented_rows(engagements: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for engagement in engagements:
        for focal, opponent in ((100, 200), (200, 100)):
            focal_prefix = "blue" if focal == 100 else "red"
            opponent_prefix = "red" if opponent == 200 else "blue"
            rows.append({
                "match_id": engagement["match_id"],
                "focal_team": focal,
                "won": int(engagement["winner_team"] == focal),
                "gold_lead": float(engagement[f"{focal_prefix}_gold"] - engagement[f"{opponent_prefix}_gold"]),
                "presence_lead": int(engagement[f"{focal_prefix}_near"] - engagement[f"{opponent_prefix}_near"]),
            })
    return rows


def _fit_symmetric_logit(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1e4, fit_intercept=False, max_iter=4000, solver="lbfgs")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\..*")
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=int))
    return model


def _grouped_cv(X: np.ndarray, y: np.ndarray, groups: np.ndarray, folds: int = 10) -> dict[str, float]:
    splitter = GroupKFold(n_splits=folds)
    pred = np.empty(len(y), dtype=float)
    for train, test in splitter.split(X, y, groups):
        model = _fit_symmetric_logit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    return {
        "folds": folds,
        "n_oriented_rows": int(len(y)),
        "n_engagements": int(len(np.unique(groups))),
        "auc": float(roc_auc_score(y, pred)),
        "brier": float(brier_score_loss(y, pred)),
        "null_brier": float(brier_score_loss(y, np.full(len(y), 0.5))),
        "log_loss": float(log_loss(y, pred)),
        "kind": "10-fold grouped out-of-fold validation; mirrored team rows remain in the same fold",
    }


def _wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return center - half, center + half


def _empirical_bins(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bins = [
        ("below -1000g", -math.inf, -1000.0),
        ("-1000g to -500g", -1000.0, -500.0),
        ("-500g to parity", -500.0, 0.0),
        ("parity to +500g", 0.0, 500.0),
        ("+500g to +1000g", 500.0, 1000.0),
        ("above +1000g", 1000.0, math.inf),
    ]
    output: list[dict[str, Any]] = []
    for label, lower, upper in bins:
        subset = [row for row in rows if lower <= row["gold_lead"] < upper]
        successes = sum(row["won"] for row in subset)
        lo, hi = _wilson(successes, len(subset))
        output.append({
            "label": label,
            "lower_gold": lower if math.isfinite(lower) else None,
            "upper_gold": upper if math.isfinite(upper) else None,
            "n_oriented_rows": len(subset),
            "wins": successes,
            "observed_win_probability": successes / len(subset) if subset else None,
            "wilson_95_low": lo if subset else None,
            "wilson_95_high": hi if subset else None,
        })
    return output


def build_report() -> dict[str, Any]:
    paths = sorted(
        path for path in TIMELINE_DIR.glob("*.json")
        if not path.name.endswith(".match.json")
    )
    episodes: list[dict[str, Any]] = []
    seen_match_ids: set[str] = set()
    counts = {
        "timeline_json_files": len(paths),
        "valid_first_grub_episodes": 0,
        "kill_resolved_engagements": 0,
    }
    for path in paths:
        episode = _extract_episode(path)
        if episode is None or episode["match_id"] in seen_match_ids:
            continue
        seen_match_ids.add(episode["match_id"])
        episodes.append(episode)
    counts["valid_first_grub_episodes"] = len(episodes)
    engagements = [episode for episode in episodes if episode["winner_team"] is not None]
    counts["kill_resolved_engagements"] = len(engagements)
    rows = _oriented_rows(engagements)
    if len(engagements) < 30:
        raise RuntimeError("Too few resolved event-centred engagements for the pilot model")

    gold = np.asarray([row["gold_lead"] / 1000.0 for row in rows], dtype=float)
    presence = np.asarray([row["presence_lead"] for row in rows], dtype=float)
    y = np.asarray([row["won"] for row in rows], dtype=int)
    groups = np.asarray([row["match_id"] for row in rows])
    variants = {
        "gold_only": gold.reshape(-1, 1),
        "gold_plus_presence": np.column_stack([gold, presence]),
    }
    validation = {
        key: _grouped_cv(X, y, groups)
        for key, X in variants.items()
    }
    primary = _fit_symmetric_logit(variants["gold_only"], y)
    beta = float(primary.coef_[0][0])

    intrinsic = json.loads((MODELS_DIR / "grubs_intrinsic_value.json").read_text())
    pro = intrinsic["logits"]["gold10"]
    objective_gold = float(intrinsic["mechanical_package"]["gold"]) + float(
        intrinsic["burn"]["wiki_scenarios"]["pre_26_11_brief_8s"]["gold_equivalent"]
    )
    leave_gold = float(
        intrinsic["leave_farm"]["scenarios"]["two_laners_one_wave"]["gold"]
    )
    comparison_grid = []
    for gold_lead in range(-2000, 2001, 100):
        p_hat = 1.0 / (1.0 + math.exp(-beta * gold_lead / 1000.0))
        _, p_star, _ = contest_ev_terminal_states(
            float(pro["intercept"]),
            float(pro["coef"]),
            baseline_gold=float(gold_lead),
            objective_gold=objective_gold,
            leave_farm_gold=leave_gold,
            win_kill_gold=600.0,
            loss_kill_gold=-600.0,
            p_secure_if_win=1.0,
            p_secure_if_lose=0.0,
        )
        comparison_grid.append({
            "gold_lead": gold_lead,
            "p_hat_fight_win": p_hat,
            "p_star_reference": p_star,
            "decision_margin": None if p_star is None else p_hat - p_star,
        })

    first_secure_match = sum(
        engagement["winner_team"] == engagement["first_secure_team"]
        for engagement in engagements
    )
    radius_sensitivity = []
    for radius in RADIUS_SENSITIVITY:
        local = decisive = tied = 0
        for episode in episodes:
            counts_at_radius = episode["kill_counts_by_radius"][int(radius)]
            total = counts_at_radius[100] + counts_at_radius[200]
            if total > 0:
                local += 1
            if counts_at_radius[100] != counts_at_radius[200]:
                decisive += 1
            elif total > 0:
                tied += 1
        radius_sensitivity.append({
            "radius": int(radius),
            "episodes_with_local_kill": local,
            "decisive_local_exchanges": decisive,
            "tied_local_exchanges": tied,
        })
    report = {
        "version": 1,
        "population": "cached high-elo ranked solo-queue timelines; exact rank/platform manifest unavailable",
        "target": "p_dec: probability a focal team wins, conditional on a decisive local champion-kill exchange around the first Void Grub camp",
        "sample": counts,
        "event_definition": {
            "era_filter": "one to three HORDE kills; first HORDE between 7:00 and 11:00",
            "fight_window": "20 seconds before first HORDE through 35 seconds after the last HORDE in the first contiguous cluster, capped at 90 seconds after first HORDE",
            "cluster_rule": "stop the first HORDE cluster when the inter-event gap exceeds 45 seconds",
            "spatial_rule": "primary radius 2200 units around the event-centred median HORDE position; 1500 and 3000 reported as sensitivity",
            "pre_fight_frame": "latest participant frame at least 30 seconds before first HORDE; outcome window starts 20 seconds before first HORDE",
            "resolved_rule": "model only non-tied local-kill exchanges; report non-decisive episodes separately",
            "orientation": "two mirrored team perspectives per engagement; no future objective outcome selects the focal side",
        },
        "descriptive": {
            "decisive_exchange_rate": len(engagements) / len(episodes),
            "radius_sensitivity": radius_sensitivity,
            "first_secure_team_won_local_exchange_n": int(first_secure_match),
            "first_secure_team_won_local_exchange_rate": first_secure_match / len(engagements),
            "empirical_gold_bins": _empirical_bins(rows),
        },
        "primary_model": {
            "formula": "logit(p_hat_dec) = beta * focal_pre_fight_gold_lead / 1000",
            "intercept_fixed_by_symmetry": 0.0,
            "beta_per_1000_gold": beta,
            "validation": validation["gold_only"],
        },
        "candidate_models": validation,
        "reference_decision_comparison": {
            "reward": "90g cash plus brief Touch ceiling",
            "outside_option": "two average early waves preserved by conceding without fighting",
            "fight_swing": "+/-600g",
            "capture_rule": "secure if fight won; opponent secures if fight lost",
            "grid": comparison_grid,
            "comparability_warning": "p_hat_dec is conditional on a decisive kill exchange and is not directly commensurate with the unconditional binary p in the structural threshold model.",
        },
        "limitations": [
            "The exact Diamond versus Master+ and platform labels cannot be reconstructed because the collection manifest was not preserved with the expanded cache.",
            "The model predicts which side wins conditional on a decisive local kill exchange, not whether an episode becomes decisive, objective secure, map win, or the causal effect of choosing to contest.",
            "Champion identity, health, items, cooldowns, summoners, vision, wave priority, and smite availability are unavailable in the present predictor set.",
            "The kill-resolved filter excludes disengages and zero-kill contests, so the target is conditional on an engagement producing a non-tied local kill result.",
            "Repeated players and collection anchors cannot be clustered without the missing manifest.",
            "Ranked estimates are not pooled with or presented as professional estimates.",
        ],
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def main() -> None:
    report = build_report()
    print(json.dumps({
        "sample": report["sample"],
        "primary_model": report["primary_model"],
        "candidate_models": report["candidate_models"],
    }, indent=2))


if __name__ == "__main__":
    main()
