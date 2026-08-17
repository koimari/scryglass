"""Fit the public draft-recommendation model and current team/player context.

The model is deliberately additive and inspectable:

* champion main effects;
* allied champion-pair synergy;
* cross-team counters;
* same-role counters;
* player fixed effects and team fixed effects as training-only controls.

Interaction terms are L2-regularized, tuned on a chronological validation
slice, and globally gated by held-out Brier score.  The serving artifact never
uses team/player fixed effects as champion effects; current strength enters
through the separately calibrated Dual Elo and player Elo models.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import FEATURES_DIR, MODELS_DIR, PARQUET_DIR
from lol_kills.export.pack_records import build_player_records, build_team_records
from lol_kills.draft_archetypes import ARCHETYPE_NAMES, champ_tags

ROOT = Path(__file__).resolve().parents[1]
SCRYGLASS_DRAFT_DIR = ROOT / "apps" / "scryglass" / "data" / "draft"
PUBLIC_PACK = ROOT / "apps" / "scryglass" / "public" / "packs" / "v2026.07.26"
MODEL_OUT = MODELS_DIR / "draft_recommendation.json"
CONTEXT_OUT = FEATURES_DIR / "draft_context.json"

ROLES = ("top", "jng", "mid", "bot", "sup")
ROLE_ALIASES = {
    "top": "top",
    "jng": "jng",
    "jungle": "jng",
    "mid": "mid",
    "middle": "mid",
    "bot": "bot",
    "bottom": "bot",
    "adc": "bot",
    "sup": "sup",
    "support": "sup",
    "utility": "sup",
}

# CommunityDragon champion-summary, checked 2026-07-26.  The warehouse has
# professional evidence for 172 of these 173 champions; Locke remains a
# selectable neutral-prior champion until pro evidence exists.
CURRENT_CHAMPIONS = (
    "Aatrox", "Ahri", "Akali", "Akshan", "Alistar", "Ambessa", "Amumu", "Anivia",
    "Annie", "Aphelios", "Ashe", "Aurelion Sol", "Aurora", "Azir", "Bard",
    "Bel'Veth", "Blitzcrank", "Brand", "Braum", "Briar", "Caitlyn", "Camille",
    "Cassiopeia", "Cho'Gath", "Corki", "Darius", "Diana", "Dr. Mundo", "Draven",
    "Ekko", "Elise", "Evelynn", "Ezreal", "Fiddlesticks", "Fiora", "Fizz", "Galio",
    "Gangplank", "Garen", "Gnar", "Gragas", "Graves", "Gwen", "Hecarim",
    "Heimerdinger", "Hwei", "Illaoi", "Irelia", "Ivern", "Janna", "Jarvan IV",
    "Jax", "Jayce", "Jhin", "Jinx", "K'Sante", "Kai'Sa", "Kalista", "Karma",
    "Karthus", "Kassadin", "Katarina", "Kayle", "Kayn", "Kennen", "Kha'Zix",
    "Kindred", "Kled", "Kog'Maw", "LeBlanc", "Lee Sin", "Leona", "Lillia",
    "Lissandra", "Locke", "Lucian", "Lulu", "Lux", "Malphite", "Malzahar",
    "Maokai", "Master Yi", "Mel", "Milio", "Miss Fortune", "Mordekaiser", "Morgana",
    "Naafiri", "Nami", "Nasus", "Nautilus", "Neeko", "Nidalee", "Nilah",
    "Nocturne", "Nunu & Willump", "Olaf", "Orianna", "Ornn", "Pantheon", "Poppy",
    "Pyke", "Qiyana", "Quinn", "Rakan", "Rammus", "Rek'Sai", "Rell",
    "Renata Glasc", "Renekton", "Rengar", "Riven", "Rumble", "Ryze", "Samira",
    "Sejuani", "Senna", "Seraphine", "Sett", "Shaco", "Shen", "Shyvana", "Singed",
    "Sion", "Sivir", "Skarner", "Smolder", "Sona", "Soraka", "Swain", "Sylas",
    "Syndra", "Tahm Kench", "Taliyah", "Talon", "Taric", "Teemo", "Thresh",
    "Tristana", "Trundle", "Tryndamere", "Twisted Fate", "Twitch", "Udyr", "Urgot",
    "Varus", "Vayne", "Veigar", "Vel'Koz", "Vex", "Vi", "Viego", "Viktor",
    "Vladimir", "Volibear", "Warwick", "Wukong", "Xayah", "Xerath", "Xin Zhao",
    "Yasuo", "Yone", "Yorick", "Yunara", "Yuumi", "Zaahen", "Zac", "Zed", "Zeri",
    "Ziggs", "Zilean", "Zoe", "Zyra",
)


def _role(value: object) -> str:
    raw = str(value or "").strip().lower()
    return ROLE_ALIASES.get(raw, raw[:3])


def _pair(a: str, b: str) -> str:
    return "|".join(sorted((a, b), key=str.casefold))


def _logit(probability: float) -> float:
    p = min(max(float(probability), 1e-5), 1 - 1e-5)
    return math.log(p / (1 - p))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def build_games(
    players: pd.DataFrame, *, require_result: bool = True
) -> list[dict[str, Any]]:
    """Return complete professional drafts in chronological order.

    Research fitting keeps the historical-result requirement. Pre-match
    inference can construct the same draft features without a result column.
    """
    frame = players.copy()
    if require_result and "result" not in frame:
        raise ValueError("historical draft rows require results")
    frame["_gid"] = frame["game_uid"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["side"] = frame["side"].astype(str).str.title()
    frame["role"] = frame["position"].map(_role)
    frame["champion"] = frame["champion"].map(lambda value: normalize_champ(str(value)))
    required = ["date", "champion"]
    if require_result:
        required.append("result")
    frame = frame.dropna(subset=required)
    frame = frame[frame["side"].isin(("Blue", "Red")) & frame["role"].isin(ROLES)]

    grouped: dict[str, dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        gid = str(row["_gid"])
        side = str(row["side"])
        role = str(row["role"])
        game = grouped.setdefault(
            gid,
            {
                "game_uid": gid,
                "blue": {},
                "red": {},
                "blue_metadata": None,
                "red_team": None,
            },
        )
        side_key = side.casefold()
        if role not in game[side_key]:
            game[side_key][role] = {
                "champion": str(row["champion"]),
                "player": str(row.get("playername") or ""),
            }
        if side == "Blue" and game["blue_metadata"] is None:
            metadata = {
                "date": pd.Timestamp(row["date"]),
                "league": str(row.get("league") or "UNKNOWN").upper(),
                "blue_team": str(row.get("teamname") or ""),
            }
            result = row.get("result")
            if result is not None and not pd.isna(result):
                metadata["y"] = float(result)
            game["blue_metadata"] = metadata
        if side == "Red" and game["red_team"] is None:
            game["red_team"] = str(row.get("teamname") or "")

    games: list[dict[str, Any]] = []
    for game in grouped.values():
        if game["blue_metadata"] is None or game["red_team"] is None:
            continue
        if any(role not in game["blue"] or role not in game["red"] for role in ROLES):
            continue
        games.append(
            {
                "game_uid": game["game_uid"],
                **game["blue_metadata"],
                "red_team": game["red_team"],
                "blue": game["blue"],
                "red": game["red"],
            }
        )
    return sorted(games, key=lambda game: (game["date"], game["game_uid"]))


def _interaction_counts(games: Iterable[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for game in games:
        for side in ("blue", "red"):
            champions = [game[side][role]["champion"] for role in ROLES]
            for role in ROLES:
                counts[f"MR|{role}|{game[side][role]['champion']}"] += 1
            counts.update(f"S|{_pair(a, b)}" for a, b in combinations(champions, 2))
            for first, second in combinations(champions, 2):
                for first_tag in champ_tags(first):
                    for second_tag in champ_tags(second):
                        counts[f"AS|{_pair(first_tag, second_tag)}"] += 1
        for blue_role in ROLES:
            blue_champion = game["blue"][blue_role]["champion"]
            for red_role in ROLES:
                red_champion = game["red"][red_role]["champion"]
                counts[f"C|{blue_champion}|{red_champion}"] += 1
                for blue_tag in champ_tags(blue_champion):
                    for red_tag in champ_tags(red_champion):
                        counts[f"AC|{blue_tag}|{red_tag}"] += 1
            red_champion = game["red"][blue_role]["champion"]
            counts[f"R|{blue_role}|{blue_champion}|{red_champion}"] += 1
    return counts


def _vocabulary(games: list[dict[str, Any]]) -> tuple[dict[str, int], set[str]]:
    interactions = _interaction_counts(games)
    keys: set[str] = set()
    for champion in CURRENT_CHAMPIONS:
        keys.add(f"M|{champion}")
        for role in ROLES:
            keys.add(f"MR|{role}|{champion}")
    for game in games:
        keys.add(f"L|{game['league']}")
        keys.add(f"T|{game['blue_team']}")
        keys.add(f"T|{game['red_team']}")
        for side in ("blue", "red"):
            for role in ROLES:
                player = game[side][role]["player"]
                if player:
                    keys.add(f"P|{player}")
    selected_interactions = {
        key
        for key, count in interactions.items()
        if (
            (key.startswith("S|") and count >= 18)
            or (key.startswith("C|") and count >= 24)
            or (key.startswith("R|") and count >= 12)
        )
    }
    selected_interactions.update(
        f"AS|{_pair(first, second)}"
        for index, first in enumerate(ARCHETYPE_NAMES)
        for second in ARCHETYPE_NAMES[index:]
    )
    selected_interactions.update(
        f"AC|{blue_tag}|{red_tag}"
        for blue_tag in ARCHETYPE_NAMES
        for red_tag in ARCHETYPE_NAMES
    )
    ordered = sorted(keys) + sorted(selected_interactions)
    return {key: index for index, key in enumerate(ordered)}, selected_interactions


def _feature_rows(
    games: list[dict[str, Any]],
    vocabulary: dict[str, int],
) -> sparse.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []

    def add(row: int, key: str, value: float, scale: float = 1.0) -> None:
        column = vocabulary.get(key)
        if column is None:
            return
        rows.append(row)
        cols.append(column)
        values.append(value * scale)

    for row_index, game in enumerate(games):
        add(row_index, f"L|{game['league']}", 1)
        add(row_index, f"T|{game['blue_team']}", 1)
        add(row_index, f"T|{game['red_team']}", -1)
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            champions = []
            for role in ROLES:
                pick = game[side][role]
                champion = pick["champion"]
                champions.append(champion)
                add(row_index, f"M|{champion}", sign)
                add(row_index, f"MR|{role}|{champion}", sign, 0.7)
                if pick["player"]:
                    add(row_index, f"P|{pick['player']}", sign)
            for first, second in combinations(champions, 2):
                add(row_index, f"S|{_pair(first, second)}", sign, 0.55)
                for first_tag in champ_tags(first):
                    for second_tag in champ_tags(second):
                        add(
                            row_index,
                            f"AS|{_pair(first_tag, second_tag)}",
                            sign,
                            0.25,
                        )
        for blue_role in ROLES:
            blue_champion = game["blue"][blue_role]["champion"]
            for red_role in ROLES:
                red_champion = game["red"][red_role]["champion"]
                add(row_index, f"C|{blue_champion}|{red_champion}", 1, 0.45)
                for blue_tag in champ_tags(blue_champion):
                    for red_tag in champ_tags(red_champion):
                        add(
                            row_index,
                            f"AC|{blue_tag}|{red_tag}",
                            1,
                            0.2,
                        )
            red_champion = game["red"][blue_role]["champion"]
            add(
                row_index,
                f"R|{blue_role}|{blue_champion}|{red_champion}",
                1,
                0.65,
            )
    return sparse.csr_matrix(
        (values, (rows, cols)),
        shape=(len(games), len(vocabulary)),
        dtype=np.float64,
    )


def _fit(
    matrix: sparse.csr_matrix,
    outcomes: np.ndarray,
    weights: np.ndarray,
    alpha: float,
) -> SGDClassifier:
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=3000,
        tol=1e-6,
        random_state=461,
        average=True,
    )
    model.fit(matrix, outcomes, sample_weight=weights)
    return model


def _metrics(outcomes: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    clipped = np.clip(probabilities, 1e-5, 1 - 1e-5)
    return {
        "n": int(len(outcomes)),
        "brier": round(float(brier_score_loss(outcomes, clipped)), 6),
        "log_loss": round(float(log_loss(outcomes, clipped)), 6),
        "auc": round(float(roc_auc_score(outcomes, clipped)), 6),
    }


def _recency_weights(
    games: list[dict[str, Any]],
    reference: pd.Timestamp,
    half_life_days: int,
) -> np.ndarray:
    return np.array(
        [
            0.5 ** (max(0, (reference - game["date"]).days) / half_life_days)
            for game in games
        ],
        dtype=float,
    )


def _interaction_columns(
    vocabulary: dict[str, int],
    interaction_keys: set[str],
) -> dict[str, np.ndarray]:
    families = {
        "role_main": ("MR|",),
        "synergy": ("S|", "AS|"),
        "direct_counter": ("C|",),
        "composition_counter": ("AC|",),
        "lane": ("R|",),
    }
    return {
        family: np.array(
            [
                vocabulary[key]
                for key in vocabulary
                if key.startswith(prefixes)
            ],
            dtype=int,
        )
        for family, prefixes in families.items()
    }


def _decision_groups(
    matrix: sparse.csr_matrix,
    coefficient: np.ndarray,
    columns_by_family: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {
        family: (
            np.asarray(matrix[:, columns] @ coefficient[columns]).reshape(-1)
            if columns.size
            else np.zeros(matrix.shape[0])
        )
        for family, columns in columns_by_family.items()
    }


def fit_recommendation_model(players: pd.DataFrame) -> dict[str, Any]:
    games = build_games(players)
    if len(games) < 500:
        raise RuntimeError("not enough complete professional drafts")
    n = len(games)
    validation_end = int(n * 0.85)
    holdout_games = games[validation_end:]

    development_games = games[:validation_end]
    vocabulary, interaction_keys = _vocabulary(development_games)
    development_matrix = _feature_rows(development_games, vocabulary)
    holdout_matrix = _feature_rows(holdout_games, vocabulary)
    development_y = np.array([game["y"] for game in development_games], dtype=int)
    holdout_y = np.array([game["y"] for game in holdout_games], dtype=int)
    latest = games[-1]["date"]

    best_alpha = 0.0003
    best_half_life = 180
    best_gates = {
        "role_main": 0.0,
        "synergy": 0.0,
        "direct_counter": 0.0,
        "composition_counter": 0.0,
        "lane": 0.0,
    }
    best_brier = float("inf")
    interaction_columns = _interaction_columns(vocabulary, interaction_keys)
    gate_candidates = (0.0, 0.25, 0.5, 0.75, 1.0)
    # Three rolling-origin validation folds keep model selection from depending
    # on one patch window. The final 15% remains outside all tuning.
    fold_boundaries = ((0.55, 0.65), (0.65, 0.75), (0.75, 0.85))
    selection_rows: list[dict[str, Any]] = []
    for half_life in (90, 180, 365):
        for alpha in (0.0003, 0.001, 0.003, 0.01):
            folds = []
            for train_fraction, validation_fraction in fold_boundaries:
                fold_train_end = int(n * train_fraction)
                fold_validation_end = int(n * validation_fraction)
                fold_train_games = games[:fold_train_end]
                fold_validation_games = games[fold_train_end:fold_validation_end]
                fold_train_matrix = development_matrix[:fold_train_end]
                fold_validation_matrix = development_matrix[
                    fold_train_end:fold_validation_end
                ]
                fold_train_y = development_y[:fold_train_end]
                fold_validation_y = development_y[
                    fold_train_end:fold_validation_end
                ]
                fold_reference = fold_train_games[-1]["date"]
                candidate = _fit(
                    fold_train_matrix,
                    fold_train_y,
                    _recency_weights(fold_train_games, fold_reference, half_life),
                    alpha,
                )
                groups = _decision_groups(
                    fold_validation_matrix,
                    candidate.coef_[0],
                    interaction_columns,
                )
                folds.append(
                    {
                        "y": fold_validation_y,
                        "base": (
                            candidate.decision_function(fold_validation_matrix)
                            - sum(groups.values())
                        ),
                        "groups": groups,
                    }
                )
            for (
                role_main_gate,
                synergy_gate,
                direct_counter_gate,
                composition_counter_gate,
                lane_gate,
            ) in product(
                gate_candidates, repeat=5
            ):
                gates = {
                    "role_main": role_main_gate,
                    "synergy": synergy_gate,
                    "direct_counter": direct_counter_gate,
                    "composition_counter": composition_counter_gate,
                    "lane": lane_gate,
                }
                fold_briers = [
                    brier_score_loss(
                        fold["y"],
                        _sigmoid(
                            fold["base"]
                            + sum(
                                gates[family] * values
                                for family, values in fold["groups"].items()
                            )
                        ),
                    )
                    for fold in folds
                ]
                mean_brier = float(np.mean(fold_briers))
                if mean_brier < best_brier:
                    best_brier = mean_brier
                    best_alpha = alpha
                    best_half_life = half_life
                    best_gates = gates
                    selection_rows = [
                        {
                            "train_end": str(
                                games[int(n * train_fraction) - 1]["date"]
                            ),
                            "validation_end": str(
                                games[int(n * validation_fraction) - 1]["date"]
                            ),
                            "brier": round(float(fold_brier), 6),
                        }
                        for (train_fraction, validation_fraction), fold_brier in zip(
                            fold_boundaries, fold_briers
                        )
                    ]

    fit_weights = _recency_weights(
        development_games,
        development_games[-1]["date"],
        best_half_life,
    )
    holdout_model = _fit(
        development_matrix,
        development_y,
        fit_weights,
        best_alpha,
    )
    coefficient = holdout_model.coef_[0]
    holdout_groups = _decision_groups(
        holdout_matrix,
        coefficient,
        interaction_columns,
    )
    base_logit = holdout_model.decision_function(holdout_matrix) - sum(
        holdout_groups.values()
    )
    holdout_probability = _sigmoid(
        base_logit
        + sum(
            best_gates[family] * values
            for family, values in holdout_groups.items()
        )
    )
    baseline_probability = _sigmoid(base_logit)

    full_vocabulary, full_interactions = _vocabulary(games)
    full_matrix = _feature_rows(games, full_vocabulary)
    full_y = np.array([game["y"] for game in games], dtype=int)
    full_weights = _recency_weights(games, latest, best_half_life)
    final_model = _fit(full_matrix, full_y, full_weights, best_alpha)
    final_coef = final_model.coef_[0]
    counts = _interaction_counts(games)

    def coefficients(prefix: str, serving_scale: float) -> dict[str, dict[str, float | int]]:
        output: dict[str, dict[str, float | int]] = {}
        for key, column in full_vocabulary.items():
            if not key.startswith(prefix):
                continue
            public_key = key[len(prefix):]
            logit = float(final_coef[column]) * serving_scale
            family = (
                "role_main"
                if prefix == "MR|"
                else "synergy"
                if prefix in {"S|", "AS|"}
                else "direct_counter"
                if prefix == "C|"
                else "composition_counter"
                if prefix == "AC|"
                else "lane"
                if prefix == "R|"
                else None
            )
            if family:
                logit *= best_gates[family]
            output[public_key] = {
                "logit": round(logit, 6),
                "n": int(counts.get(key, 0)),
            }
        return output

    champion_counts = Counter(
        pick["champion"]
        for game in games
        for side in ("blue", "red")
        for pick in game[side].values()
    )
    role_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for game in games:
        for side in ("blue", "red"):
            for role in ROLES:
                role_counts[game[side][role]["champion"]][role] += 1
    champion_roles = {}
    for champion in CURRENT_CHAMPIONS:
        total = sum(role_counts[champion].values())
        threshold = max(2, math.ceil(total * 0.05))
        champion_roles[champion] = [
            role for role in ROLES if role_counts[champion][role] >= threshold
        ]

    draft_calibration = {}
    draft_calibration_path = MODELS_DIR / "draft_wr_calibration.json"
    if draft_calibration_path.exists():
        draft_calibration = json.loads(draft_calibration_path.read_text())
    elo_calibration = {}
    elo_calibration_path = MODELS_DIR / "elo_wr_calibration.json"
    if elo_calibration_path.exists():
        elo_calibration = json.loads(elo_calibration_path.read_text())

    return {
        "version": 2,
        "as_of": str(latest),
        "population": "Oracle's Elixir professional maps, 2025-2026",
        "n_maps": n,
        "champion_roster_source": "CommunityDragon champion-summary checked 2026-07-26",
        "champion_roster": list(CURRENT_CHAMPIONS),
        "champ_game_counts": {
            champion: int(champion_counts.get(champion, 0))
            for champion in CURRENT_CHAMPIONS
        },
        "champion_roles": champion_roles,
        "champion_archetypes": {
            champion: sorted(champ_tags(champion))
            for champion in CURRENT_CHAMPIONS
        },
        "win_delta": {
            key: row["logit"]
            for key, row in coefficients("M|", 1.0).items()
        },
        "champion_effects": coefficients("M|", 1.0),
        "role_effects": coefficients("MR|", 0.7),
        "champion_role_counts": {
            champion: {
                role: int(role_counts[champion].get(role, 0))
                for role in ROLES
            }
            for champion in CURRENT_CHAMPIONS
        },
        "ally_synergy": coefficients("S|", 0.55),
        "counter_pairs": coefficients("C|", 0.45),
        "role_pairs": coefficients("R|", 0.65),
        "archetype_synergy": coefficients("AS|", 0.25),
        "archetype_counters": coefficients("AC|", 0.2),
        "kill_beta": {},
        "calibration": draft_calibration,
        "elo_calibration": elo_calibration,
        "evaluation": {
            "split": (
                "Three rolling-origin development folds through the first 85%; "
                "final 15% chronological evaluation"
            ),
            "regularization": "L2 logistic SGD with validation-selected recency half-life",
            "selected_alpha": best_alpha,
            "selected_half_life_days": best_half_life,
            "interaction_gates": {
                "role_residual": best_gates["role_main"],
                "ally_synergy": best_gates["synergy"],
                "cross_counter": best_gates["direct_counter"],
                "composition_counter": best_gates["composition_counter"],
                "same_role": best_gates["lane"],
            },
            "rolling_validation": selection_rows,
            "rolling_validation_mean_brier": round(best_brier, 6),
            "baseline_holdout": _metrics(holdout_y, baseline_probability),
            "interaction_holdout": _metrics(holdout_y, holdout_probability),
            "interpretation": (
                "The champion-by-role residual, synergy, exact cross-counter, composition response, "
                "and same-role evidence are gated separately. Synergy combines sparse champion-pair "
                "and lower-dimensional archetype evidence. "
                "Family gates are selected across "
                "three rolling-origin validation folds from 0, .25, .5, .75, 1, then frozen "
                "for the final chronological evaluation. Zero means that family did not earn "
                "serving weight."
            ),
        },
    }


def _adjusted(mu: float, sigma: float, floor: float) -> float:
    return float(mu) - max(0.0, float(sigma) - floor)


def build_draft_context(
    players: pd.DataFrame,
    team_ratings: list[dict[str, Any]],
    player_ratings: list[dict[str, Any]],
    team_records: dict[str, dict[str, Any]],
    player_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    frame = players.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["role"] = frame["position"].map(_role)
    frame["champion"] = frame["champion"].map(lambda value: normalize_champ(str(value)))
    frame["result"] = pd.to_numeric(frame["result"], errors="coerce")
    frame = frame.dropna(subset=["date", "playername", "champion", "result"])
    latest = frame["date"].max()
    frame["_weight"] = 0.5 ** ((latest - frame["date"]).dt.days / 365.0)
    global_effective = float(frame["_weight"].sum())
    global_wins = float((frame["_weight"] * frame["result"]).sum())
    global_probability = (global_wins + 20.0 * 0.5) / (global_effective + 20.0)
    champion_probability: dict[str, float] = {}
    for champion, champion_rows in frame.groupby("champion"):
        effective = float(champion_rows["_weight"].sum())
        wins = float((champion_rows["_weight"] * champion_rows["result"]).sum())
        champion_probability[str(champion)] = (
            wins + 30.0 * global_probability
        ) / (effective + 30.0)

    player_rating_by = {str(row["player"]): row for row in player_ratings}
    current_players: dict[str, list[str]] = defaultdict(list)
    for player, record in player_records.items():
        team = str(record.get("current_team") or "")
        if not team or player not in player_rating_by:
            continue
        team_date = pd.to_datetime(team_records.get(team, {}).get("current_date"), errors="coerce")
        player_date = pd.to_datetime(record.get("current_date"), errors="coerce")
        if pd.notna(team_date) and pd.notna(player_date) and (team_date - player_date).days > 90:
            continue
        current_players[team].append(player)

    player_payload: dict[str, dict[str, Any]] = {}
    for team, names in current_players.items():
        for player in names:
            history = frame[frame["playername"].astype(str) == player]
            if history.empty:
                continue
            recent = history.sort_values("date").tail(30)
            role = (
                recent[recent["role"].isin(ROLES)]["role"].value_counts().index[0]
                if recent["role"].isin(ROLES).any()
                else None
            )
            weighted_games = float(history["_weight"].sum())
            weighted_wins = float((history["_weight"] * history["result"]).sum())
            overall = (weighted_wins + 6.0 * 0.5) / (weighted_games + 6.0)
            mastery: dict[str, dict[str, float | int]] = {}
            for champion, champion_rows in history.groupby("champion"):
                games = int(len(champion_rows))
                if games < 2:
                    continue
                effective = float(champion_rows["_weight"].sum())
                wins = float((champion_rows["_weight"] * champion_rows["result"]).sum())
                champion_delta = _logit(
                    champion_probability.get(str(champion), global_probability)
                ) - _logit(global_probability)
                expected = 1.0 / (
                    1.0 + math.exp(-(_logit(overall) + champion_delta))
                )
                probability = (wins + 8.0 * expected) / (effective + 8.0)
                reliability = effective / (effective + 12.0)
                residual = max(
                    -0.18,
                    min(
                        0.18,
                        (_logit(probability) - _logit(expected)) * reliability,
                    ),
                )
                mastery[str(champion)] = {
                    "logit": round(residual, 5),
                    "n": games,
                    "effective_n": round(effective, 2),
                }
            rating = player_rating_by[player]
            player_payload[player] = {
                "player": player,
                "team": team,
                "role": role,
                "rating": round(_adjusted(rating["mu_total"], rating["sigma"], 28.0), 2),
                "raw_rating": round(float(rating["mu_total"]), 2),
                "sigma": round(float(rating["sigma"]), 2),
                "n_maps": int(rating.get("n_maps") or 0),
                "mastery": mastery,
            }

    teams = []
    for rating in team_ratings:
        name = str(rating["team"])
        record = team_records.get(name, {})
        current_date = pd.to_datetime(record.get("current_date"), errors="coerce")
        if pd.isna(current_date) or (latest - current_date).days > 120:
            continue
        roster = [
            player_payload[player]
            for player in current_players.get(name, [])
            if player in player_payload
        ]
        roster.sort(key=lambda row: (ROLES.index(row["role"]) if row["role"] in ROLES else 99, row["player"]))
        tier = record.get("current_tier")
        if tier not in {"tier1", "tier2", "tier3"} and len(roster) < 3:
            continue
        teams.append(
            {
                "team": name,
                "league": record.get("current_league") or rating.get("home_league"),
                "tier": tier,
                "rating": round(
                    float(rating.get("rating_p10"))
                    if rating.get("rating_p10") is not None
                    else _adjusted(rating["mu_total"], rating["sigma"], 25.0),
                    2,
                ),
                "raw_rating": round(float(rating["mu_total"]), 2),
                "sigma": round(float(rating["sigma"]), 2),
                "roster": roster,
            }
        )
    teams.sort(key=lambda row: ((row["tier"] or "tier9"), -(row["rating"]), row["team"]))
    active_player_names = {
        str(player["player"])
        for team in teams
        for player in team["roster"]
    }
    active_players = {
        player: payload
        for player, payload in player_payload.items()
        if player in active_player_names
    }
    return {
        "version": 1,
        "as_of": str(latest),
        "teams": teams,
        "players": active_players,
        "note": (
            "Team and player strength use adjusted public Elo. Player-champion comfort "
            "is a recency-weighted, Bayesian-shrunk residual after both that player's "
            "baseline and the champion's global strength are removed."
        ),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_local_context(players: pd.DataFrame | None = None) -> dict[str, Any]:
    frame = players if players is not None else pd.read_parquet(
        PARQUET_DIR / "players.parquet"
    )
    features = PUBLIC_PACK / "features"
    team_ratings = _read_json(features / "ratings_snapshot.json")
    player_ratings = _read_json(features / "player_ratings_snapshot.json")
    team_records = _read_json(features / "team_records.json")
    player_records = _read_json(features / "player_records.json")
    context = build_draft_context(
        frame,
        team_ratings,
        player_ratings,
        team_records,
        player_records,
    )
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    SCRYGLASS_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    CONTEXT_OUT.write_text(json.dumps(context, separators=(",", ":")))
    (SCRYGLASS_DRAFT_DIR / "context.json").write_text(json.dumps(context, separators=(",", ":")))
    return context


def build_local_artifacts() -> tuple[dict[str, Any], dict[str, Any]]:
    players = pd.read_parquet(PARQUET_DIR / "players.parquet")
    model = write_recommendation_model(players)
    context = write_local_context(players)
    SCRYGLASS_DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    (SCRYGLASS_DRAFT_DIR / "runtime.json").write_text(json.dumps(model, separators=(",", ":")))
    return model, context


def write_recommendation_model(players: pd.DataFrame | None = None) -> dict[str, Any]:
    frame = players if players is not None else pd.read_parquet(PARQUET_DIR / "players.parquet")
    model = fit_recommendation_model(frame)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.write_text(json.dumps(model, separators=(",", ":")))
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="fit and write local serving artifacts")
    parser.add_argument(
        "--context-only",
        action="store_true",
        help="refresh current team, lineup, and player-comfort context without refitting",
    )
    args = parser.parse_args()
    if args.context_only:
        context = write_local_context()
        print(
            json.dumps(
                {
                    "context": CONTEXT_OUT.as_posix(),
                    "teams": len(context["teams"]),
                    "players": len(context["players"]),
                },
                indent=2,
            )
        )
    elif args.build:
        model, context = build_local_artifacts()
        print(
            json.dumps(
                {
                    "model": MODEL_OUT.as_posix(),
                    "maps": model["n_maps"],
                    "champions": len(model["champion_roster"]),
                    "interaction_gates": model["evaluation"]["interaction_gates"],
                    "holdout": model["evaluation"]["interaction_holdout"],
                    "teams": len(context["teams"]),
                    "players": len(context["players"]),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
