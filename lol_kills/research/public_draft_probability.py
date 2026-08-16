"""Bounded research search for a composition-only draft probability.

The experiment keeps the pre-match estimand separate from team ratings,
player strength, observed state, and the R9E composite.  It fits each fold
from the picks available before that fold.  The final chronological block is
held back until candidate selection is complete.

This module is research-only.  It does not write public model authority and it
does not change the application scorer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.etl.aliases import normalize_champ
from lol_kills.research.composition_signal import (
    ROLES,
    build_composition_games,
)


SCHEMA_VERSION = "scryglass:public-draft-probability-research:v1"
MODEL_VERSION = "public-draft-probability-composition-v1"
PUBLIC_PATCH_MAP = {"16.16": "26.16"}
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_BOOTSTRAP_REPS = 500
DEFAULT_SEED = 461

# This map is a calibration grouping.  It does not encode team strength.
LEAGUE_REGION = {
    "LCK": "KR",
    "LCKC": "KR",
    "LPL": "CN",
    "LPL2": "CN",
    "LEC": "EMEA",
    "LFL": "EMEA",
    "LFL2": "EMEA",
    "LVP SL": "EMEA",
    "NLC": "EMEA",
    "LIT": "EMEA",
    "LJL": "PACIFIC",
    "PCS": "PACIFIC",
    "VCS": "PACIFIC",
    "LCP": "PACIFIC",
    "CBLOL": "AMERICAS",
    "LCS": "AMERICAS",
    "LLA": "AMERICAS",
    "AMERICAS": "AMERICAS",
    "TCL": "EMEA",
    "EM": "EMEA",
}

ATOM_FAMILIES = (
    "crowd-control-mobility",
    "damage",
    "heal-shield",
    "interaction",
    "stack-transform-summon-resource",
    "vision-economy",
)
ATOM_ATTRIBUTES = (
    "abilityReliance",
    "control",
    "damage",
    "difficulty",
    "mobility",
    "toughness",
    "utility",
)
ATOM_DIMENSIONS = (
    "crowd_control",
    "damage_profile",
    "durability_frontline",
    "engage",
    "mobility",
    "scaling",
    "sustain",
    "target_access",
    "wave_control",
)
ATOM_SLUG_ALIASES = {
    "wukong": "monkeyking",
    "nunu & willump": "nunu",
    "renata glasc": "renata",
}

# Explicitly kept beside the feature builder so a future edit cannot add a
# strength or live-state field without changing the audit contract.
FORBIDDEN_FEATURE_TERMS = (
    "elo",
    "mu_diff",
    "sigma",
    "rating",
    "momentum",
    "gold",
    "objective",
    "tower",
    "dragon",
    "baron",
    "inhibitor",
    "outcome",
    "r9e",
    "history",
    "form",
)


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    champion_role: bool = True
    ally: bool = True
    counter: bool = True
    same_role: bool = True
    atoms: bool = True
    atom_interactions: bool = True
    calibration: bool = True
    region_calibration: bool = True
    patch_calibration: bool = True
    tournament_calibration: bool = True
    comfort: bool = False
    support: int = 20
    c: float = 0.10


@dataclass(frozen=True)
class PreparedGame:
    game: Mapping[str, Any]
    region: str
    scope: str
    event_kind: str
    tournament: str
    comfort_blue: tuple[float, ...]
    comfort_red: tuple[float, ...]
    roster_change: bool


_WORKER_ITEMS: Sequence[PreparedGame] | None = None
_WORKER_ATOM_VECTORS: Mapping[str, np.ndarray] | None = None
_WORKER_ATOM_NAMES: Sequence[str] | None = None


def _worker_init(
    items: Sequence[PreparedGame],
    atom_vectors: Mapping[str, np.ndarray],
    atom_names: Sequence[str],
) -> None:
    global _WORKER_ITEMS, _WORKER_ATOM_VECTORS, _WORKER_ATOM_NAMES
    _WORKER_ITEMS = items
    _WORKER_ATOM_VECTORS = atom_vectors
    _WORKER_ATOM_NAMES = atom_names


def _dev_candidate_worker(config: CandidateConfig) -> dict[str, Any]:
    if _WORKER_ITEMS is None or _WORKER_ATOM_VECTORS is None or _WORKER_ATOM_NAMES is None:
        raise RuntimeError("research worker state is not initialized")
    _, _, folds = fixed_chronological_folds(_WORKER_ITEMS)
    return _dev_candidate(config, _WORKER_ITEMS, folds, _WORKER_ATOM_VECTORS, _WORKER_ATOM_NAMES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _patch_token(value: Any) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", text)
    if not match:
        return text or "UNKNOWN"
    return f"{int(match.group(1))}.{int(match.group(2)):02d}"


def _slug(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return "".join(ch for ch in text if ch.isalnum())


def _atom_champion_key(value: Any) -> str:
    text = str(value or "").strip().casefold()
    return ATOM_SLUG_ALIASES.get(text, _slug(text))


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _region(row: Mapping[str, Any]) -> str:
    scope = str(row.get("competition_scope") or "").strip().casefold()
    if scope == "international" or bool(row.get("is_international")):
        return "INTERNATIONAL"
    if scope == "interregional" or bool(row.get("is_interregional")):
        return "INTERREGIONAL"
    league = str(row.get("league") or "").strip().upper()
    return LEAGUE_REGION.get(league, "OTHER")


def _metadata_by_game(players: pd.DataFrame) -> dict[str, dict[str, Any]]:
    columns = [
        "game_uid",
        "league",
        "competition_scope",
        "event_kind",
        "competition_tier",
        "is_international",
        "is_interregional",
        "tournament",
        "patch",
        "oe_patch_token",
        "date",
    ]
    available = [column for column in columns if column in players.columns]
    frame = players[available].copy()
    frame["game_uid"] = frame["game_uid"].astype(str)
    metadata: dict[str, dict[str, Any]] = {}
    for game_uid, group in frame.groupby("game_uid", sort=False):
        first = group.iloc[0].to_dict()
        metadata[str(game_uid)] = first
    return metadata


def _atom_vectors(path: Path) -> tuple[dict[str, np.ndarray], list[str], str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    vector_names: list[str] = []
    for family in ATOM_FAMILIES:
        vector_names.append(f"family:{family}")
    for attribute in ATOM_ATTRIBUTES:
        vector_names.append(f"attribute:{attribute}")
    for dimension in ATOM_DIMENSIONS:
        labels: set[str] = set()
        for champion in payload.get("champions", []):
            prior = (champion.get("ontology_prior") or {}).get(dimension) or {}
            labels.update((prior.get("labels") or {}).keys())
        for label in sorted(labels):
            vector_names.append(f"ontology:{dimension}:{label}")
    vectors: dict[str, np.ndarray] = {}
    for champion in payload.get("champions", []):
        values: list[float] = []
        families = champion.get("atom_family_counts") or {}
        # Family counts are atom counts, not probabilities.  Keep the fixed
        # LCC scale bounded before adding ally and cross-team products.
        values.extend(float(families.get(family, 0.0)) / 25.0 for family in ATOM_FAMILIES)
        attributes = champion.get("lcc_attribute_ratings") or {}
        values.extend(float(attributes.get(attribute, 0.0)) / 20.0 for attribute in ATOM_ATTRIBUTES)
        for dimension in ATOM_DIMENSIONS:
            labels = (champion.get("ontology_prior") or {}).get(dimension) or {}
            distributions = labels.get("labels") or {}
            for name in vector_names:
                prefix = f"ontology:{dimension}:"
                if name.startswith(prefix):
                    values.append(float(distributions.get(name[len(prefix) :], 0.0)))
        vectors[_atom_champion_key(champion.get("display_name"))] = np.asarray(values, dtype=float)
    return vectors, vector_names, str(payload.get("artifact_sha256") or "")


def _safe_vector(vectors: Mapping[str, np.ndarray], champion: Any, width: int) -> np.ndarray:
    return vectors.get(_atom_champion_key(champion), np.zeros(width, dtype=float))


def _side_champions(game: Mapping[str, Any], side: str) -> list[tuple[str, str]]:
    return [(role, normalize_champ(str(game[side][role].get("champion") or ""))) for role in ROLES]


def _canonical_pair(left: tuple[str, str], right: tuple[str, str]) -> tuple[tuple[str, str], tuple[str, str], float]:
    if left <= right:
        return left, right, 1.0
    return right, left, -1.0


def _comfort_history(games: Sequence[Mapping[str, Any]]) -> dict[str, tuple[tuple[float, ...], tuple[float, ...]]]:
    counts: dict[tuple[str, str], int] = {}
    result: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
    for game in games:
        blue_values: list[float] = []
        red_values: list[float] = []
        for side, target in (("blue", blue_values), ("red", red_values)):
            for role in ROLES:
                pick = game[side][role]
                key = (str(pick.get("player") or "").casefold(), normalize_champ(pick.get("champion")))
                target.append(math.log1p(counts.get(key, 0)))
        result[str(game.get("game_uid"))] = (tuple(blue_values), tuple(red_values))
        for side in ("blue", "red"):
            for role in ROLES:
                pick = game[side][role]
                key = (str(pick.get("player") or "").casefold(), normalize_champ(pick.get("champion")))
                counts[key] = counts.get(key, 0) + 1
    return result


def _roster_change_flags(games: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    last: dict[str, frozenset[str]] = {}
    result: dict[str, bool] = {}
    for game in games:
        changed = False
        for side in ("blue", "red"):
            team = str(game.get(f"{side}_team") or "")
            roster = frozenset(
                str(game[side][role].get("player") or "").casefold() for role in ROLES
            )
            previous = last.get(team)
            if previous is not None and len(roster & previous) < 5:
                changed = True
            last[team] = roster
        result[str(game.get("game_uid"))] = changed
    return result


def prepare_games(players: pd.DataFrame, atom_path: Path) -> tuple[list[PreparedGame], dict[str, Any]]:
    if "game_uid" not in players.columns:
        raise ValueError("the source snapshot needs game_uid")
    frame = players.copy()
    if "oe_patch_token" in frame.columns:
        frame["patch"] = frame["oe_patch_token"].where(frame["oe_patch_token"].notna(), frame.get("patch"))
    frame["patch"] = frame["patch"].map(_patch_token)
    games = build_composition_games(frame)
    games = sorted(games, key=lambda item: (pd.Timestamp(item["date"]), str(item["game_uid"])))
    metadata = _metadata_by_game(frame)
    comfort = _comfort_history(games)
    roster_changes = _roster_change_flags(games)
    prepared: list[PreparedGame] = []
    for game in games:
        row = metadata.get(str(game["game_uid"]), {})
        patch = _patch_token(game.get("patch"))
        game = dict(game)
        game["patch"] = patch
        scope = str(row.get("competition_scope") or row.get("competition_tier") or "UNKNOWN").strip().upper()
        event_kind = str(row.get("event_kind") or row.get("competition_tier") or "UNKNOWN").strip().upper()
        tournament = str(row.get("tournament") or "").strip() or event_kind
        region = _region(row)
        blue_comfort, red_comfort = comfort.get(str(game["game_uid"]), ((), ()))
        prepared.append(
            PreparedGame(
                game=game,
                region=region,
                scope=scope,
                event_kind=event_kind,
                tournament=tournament,
                comfort_blue=blue_comfort,
                comfort_red=red_comfort,
                roster_change=roster_changes.get(str(game["game_uid"]), False),
            )
        )
    source_meta = {
        "source_rows": int(len(frame)),
        "source_games": int(frame["game_uid"].nunique()),
        "prepared_games": int(len(prepared)),
        "date_min": str(frame["date"].min()),
        "date_max": str(frame["date"].max()),
        "patch_counts": {
            str(k): int(v)
            for k, v in frame.drop_duplicates("game_uid")["patch"].value_counts().sort_index().items()
        },
        "source_patch_public_map": PUBLIC_PATCH_MAP,
        "atom_artifact_sha256": _atom_vectors(atom_path)[2],
    }
    return prepared, source_meta


def _atom_arrays(path: Path) -> tuple[dict[str, np.ndarray], list[str], str]:
    return _atom_vectors(path)


def _tokens(
    item: PreparedGame,
    config: CandidateConfig,
    atom_vectors: Mapping[str, np.ndarray],
    atom_names: Sequence[str],
) -> list[tuple[str, float]]:
    game = item.game
    tokens: list[tuple[str, float]] = []
    blue = _side_champions(game, "blue")
    red = _side_champions(game, "red")
    if config.champion_role:
        for sign, side in ((1.0, blue), (-1.0, red)):
            for role, champion in side:
                tokens.append((f"CH|{role}|{champion}", sign))
    if config.ally:
        for sign, side in ((1.0, blue), (-1.0, red)):
            for left, right in combinations(side, 2):
                first, second = sorted((left, right))
                tokens.append((f"ALLY|{first[0]}:{second[0]}|{first[1]}|{second[1]}", sign))
                tokens.append((f"ALLYCH|{first[1]}|{second[1]}", sign))
    if config.counter or config.same_role:
        for blue_pick, red_pick in product_pairs(blue, red):
            first, second, sign = _canonical_pair(blue_pick, red_pick)
            if config.counter:
                tokens.append((f"CTR|{first[0]}:{second[0]}|{first[1]}|{second[1]}", sign))
            if config.same_role and blue_pick[0] == red_pick[0]:
                tokens.append((f"SAME|{blue_pick[0]}|{first[1]}|{second[1]}", sign))
    if config.atoms:
        width = len(atom_names)
        blue_vectors = np.asarray([_safe_vector(atom_vectors, champion, width) for _, champion in blue])
        red_vectors = np.asarray([_safe_vector(atom_vectors, champion, width) for _, champion in red])
        blue_sum = blue_vectors.sum(axis=0)
        red_sum = red_vectors.sum(axis=0)
        for name, value in zip(atom_names, (blue_sum - red_sum) / 5.0):
            if value:
                tokens.append((f"ATOM|{name}", float(value)))
        if config.atom_interactions:
            blue_family = blue_sum[: len(ATOM_FAMILIES)]
            red_family = red_sum[: len(ATOM_FAMILIES)]
            for i in range(len(ATOM_FAMILIES)):
                for j in range(i, len(ATOM_FAMILIES)):
                    ally_value = float(blue_family[i] * blue_family[j] - red_family[i] * red_family[j]) / 25.0
                    cross_value = float(blue_family[i] * red_family[j] - red_family[i] * blue_family[j]) / 25.0
                    if ally_value:
                        tokens.append((f"ATOMALLY|{ATOM_FAMILIES[i]}|{ATOM_FAMILIES[j]}", ally_value))
                    if cross_value:
                        tokens.append((f"ATOMCTR|{ATOM_FAMILIES[i]}|{ATOM_FAMILIES[j]}", cross_value))
    if config.calibration:
        if config.region_calibration:
            tokens.extend(
                [
                    (f"CAL|region|{item.region}", 1.0),
                    (f"CAL|scope|{item.scope}", 1.0),
                ]
            )
        if config.tournament_calibration:
            tokens.extend(
                [
                    (f"CAL|event|{item.event_kind}", 1.0),
                    (f"CAL|tournament|{item.tournament}", 1.0),
                ]
            )
        if config.patch_calibration:
            tokens.extend(
                [
                    (f"CAL|patch-major|{game.get('patch', 'UNKNOWN').split('.')[0]}", 1.0),
                    (f"CAL|patch|{game.get('patch', 'UNKNOWN')}", 1.0),
                ]
            )
    if config.comfort:
        for role, blue_value, red_value in zip(ROLES, item.comfort_blue, item.comfort_red):
            tokens.append((f"COMFORT|{role}", float(blue_value - red_value)))
        tokens.append(
            (
                "COMFORT|all",
                float(np.mean(item.comfort_blue) - np.mean(item.comfort_red))
                if item.comfort_blue and item.comfort_red
                else 0.0,
            )
        )
    return tokens


def product_pairs(left: Sequence[tuple[str, str]], right: Sequence[tuple[str, str]]) -> Iterable[tuple[tuple[str, str], tuple[str, str]]]:
    for first in left:
        for second in right:
            yield first, second


def _feature_tokens(
    items: Sequence[PreparedGame],
    config: CandidateConfig,
    atom_vectors: Mapping[str, np.ndarray],
    atom_names: Sequence[str],
) -> list[list[tuple[str, float]]]:
    rows = [_tokens(item, config, atom_vectors, atom_names) for item in items]
    for row in rows:
        for key, _ in row:
            segments = {segment.casefold() for segment in key.split("|")}
            if segments.intersection(FORBIDDEN_FEATURE_TERMS):
                raise AssertionError(f"forbidden feature term reached matrix: {key}")
    return rows


def _vocabulary(rows: Sequence[Sequence[tuple[str, float]]], support: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for key in {key for key, _ in row}:
            counts[key] = counts.get(key, 0) + 1
    keys = sorted(key for key, count in counts.items() if count >= support or key.startswith(("ATOM|", "ATOMALLY|", "ATOMCTR|", "COMFORT|")))
    return {key: index for index, key in enumerate(keys)}


def _matrix(rows: Sequence[Sequence[tuple[str, float]]], vocabulary: Mapping[str, int]) -> sparse.csr_matrix:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows):
        for key, value in row:
            column = vocabulary.get(key)
            if column is not None and value:
                row_indices.append(row_index)
                column_indices.append(column)
                values.append(float(value))
    return sparse.csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(rows), len(vocabulary)),
        dtype=np.float64,
    )


def _ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float | None:
    if len(y) == 0:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for index in range(bins):
        mask = (p >= edges[index]) & (p < edges[index + 1] if index < bins - 1 else p <= edges[index + 1])
        if not np.any(mask):
            continue
        total += float(mask.mean()) * abs(float(y[mask].mean()) - float(p[mask].mean()))
    return total


def _calibration(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    if len(y) < 30 or len(np.unique(y)) < 2:
        return {"intercept": None, "slope": None}
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    # Bound extreme validation logits.  A two-parameter bounded likelihood
    # avoids unstable matrix products on perfectly separated folds.
    logits = np.clip(logits, -20.0, 20.0)

    def objective(theta: np.ndarray) -> float:
        linear = theta[0] + theta[1] * logits
        return float(np.sum(np.logaddexp(0.0, linear) - y * linear))

    result = minimize(
        objective,
        np.asarray([0.0, 1.0]),
        method="L-BFGS-B",
        bounds=((-12.0, 12.0), (0.0, 12.0)),
        options={"maxiter": 200},
    )
    if not result.success or not np.isfinite(result.x).all():
        return {"intercept": None, "slope": None}
    return {"intercept": float(result.x[0]), "slope": float(result.x[1])}


def metrics(y: Sequence[int], probabilities: Sequence[float]) -> dict[str, Any]:
    target = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    auc = None
    if len(np.unique(target)) > 1:
        auc = float(roc_auc_score(target, p))
    return {
        "n": int(len(target)),
        "positive_rate": float(target.mean()) if len(target) else None,
        "auc": auc,
        "brier": float(brier_score_loss(target, p)) if len(target) else None,
        "log_loss": float(log_loss(target, p, labels=[0.0, 1.0])) if len(target) else None,
        "ece": _ece(target, p),
        "calibration": _calibration(target, p),
    }


def _bootstrap(y: np.ndarray, p: np.ndarray, reps: int, seed: int) -> dict[str, Any]:
    if len(y) < 30 or len(np.unique(y)) < 2:
        return {"reps": 0, "auc": None, "brier": None, "log_loss": None}
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    brier_values: list[float] = []
    loss_values: list[float] = []
    for _ in range(reps):
        indices = rng.integers(0, len(y), size=len(y))
        yy = y[indices]
        pp = p[indices]
        if len(np.unique(yy)) < 2:
            continue
        auc_values.append(float(roc_auc_score(yy, pp)))
        brier_values.append(float(np.mean((yy - pp) ** 2)))
        loss_values.append(float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log1p(-pp))))

    def interval(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"mean": None, "lower": None, "upper": None}
        q = np.percentile(values, [2.5, 50.0, 97.5])
        return {"mean": float(q[1]), "lower": float(q[0]), "upper": float(q[2])}

    return {
        "reps": int(len(auc_values)),
        "auc": interval(auc_values),
        "brier": interval(brier_values),
        "log_loss": interval(loss_values),
    }


def _fit_predict(
    train: Sequence[PreparedGame],
    validation: Sequence[PreparedGame],
    config: CandidateConfig,
    atom_vectors: Mapping[str, np.ndarray],
    atom_names: Sequence[str],
    row_lookup: Mapping[str, Sequence[tuple[str, float]]] | None = None,
) -> tuple[np.ndarray, int]:
    if row_lookup is None:
        train_rows = _feature_tokens(train, config, atom_vectors, atom_names)
        validation_rows = _feature_tokens(validation, config, atom_vectors, atom_names)
    else:
        train_rows = [row_lookup[str(item.game["game_uid"])] for item in train]
        validation_rows = [row_lookup[str(item.game["game_uid"])] for item in validation]
    vocabulary = _vocabulary(train_rows, config.support)
    x_train = _matrix(train_rows, vocabulary)
    x_validation = _matrix(validation_rows, vocabulary)
    model = LogisticRegression(
        C=float(config.c),
        solver="liblinear",
        max_iter=1200,
        random_state=DEFAULT_SEED,
    )
    model.fit(x_train, [int(item.game["y"]) for item in train])
    return (
        model.predict_proba(x_validation)[:, 1],
        len(vocabulary),
    )


def _fold_boundaries(items: Sequence[PreparedGame], parts: int) -> list[int]:
    dates = [pd.Timestamp(item.game["date"]).normalize() for item in items]
    clusters: list[int] = []
    last: pd.Timestamp | None = None
    for index, date in enumerate(dates):
        if last is None or date != last:
            clusters.append(index)
            last = date
    clusters.append(len(items))
    if len(clusters) < parts + 1:
        return [int(value) for value in np.linspace(0, len(items), parts + 1)]
    positions = np.linspace(0, len(clusters) - 1, parts + 1).astype(int)
    return [clusters[int(position)] for position in positions]


def fixed_chronological_folds(items: Sequence[PreparedGame], holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION) -> tuple[list[PreparedGame], list[PreparedGame], list[tuple[list[PreparedGame], list[PreparedGame]]]]:
    if not 0.05 <= holdout_fraction <= 0.40:
        raise ValueError("holdout_fraction must be between 0.05 and 0.40")
    boundary = _fold_boundaries(items, 5)[-2]
    development = list(items[:boundary])
    final_holdout = list(items[boundary:])
    dev_boundaries = _fold_boundaries(development, 4)
    folds: list[tuple[list[PreparedGame], list[PreparedGame]]] = []
    for index in range(1, len(dev_boundaries) - 1):
        train = development[: dev_boundaries[index]]
        validation = development[dev_boundaries[index] : dev_boundaries[index + 1]]
        if train and validation:
            folds.append((train, validation))
    return development, final_holdout, folds


def _dev_candidate(config: CandidateConfig, items: Sequence[PreparedGame], folds: Sequence[tuple[list[PreparedGame], list[PreparedGame]]], atom_vectors: Mapping[str, np.ndarray], atom_names: Sequence[str]) -> dict[str, Any]:
    rows = _feature_tokens(items, config, atom_vectors, atom_names)
    row_lookup = {str(item.game["game_uid"]): row for item, row in zip(items, rows)}
    fold_metrics: list[dict[str, Any]] = []
    vocabulary_sizes: list[int] = []
    for index, (train, validation) in enumerate(folds, 1):
        probabilities, vocabulary_size = _fit_predict(train, validation, config, atom_vectors, atom_names, row_lookup)
        fold_metrics.append({"fold": index, **metrics([int(item.game["y"]) for item in validation], probabilities)})
        vocabulary_sizes.append(vocabulary_size)
    aucs = [row["auc"] for row in fold_metrics if row["auc"] is not None]
    losses = [row["log_loss"] for row in fold_metrics if row["log_loss"] is not None]
    return {
        "config": config.__dict__,
        "folds": fold_metrics,
        "mean_auc": float(np.mean(aucs)) if aucs else None,
        "mean_log_loss": float(np.mean(losses)) if losses else None,
        "vocabulary_sizes": vocabulary_sizes,
    }


def _group_metrics(items: Sequence[PreparedGame], probabilities: np.ndarray, key: str) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    values: list[str] = []
    for item in items:
        if key == "region":
            values.append(item.region)
        elif key == "patch":
            values.append(str(item.game.get("patch") or "UNKNOWN"))
        elif key == "event_kind":
            values.append(item.event_kind)
        elif key == "roster_change":
            values.append("changed" if item.roster_change else "stable_or_first")
        else:
            raise ValueError(key)
    for value in sorted(set(values)):
        mask = np.asarray([entry == value for entry in values], dtype=bool)
        grouped[value] = metrics(
            [int(item.game["y"]) for index, item in enumerate(items) if mask[index]],
            probabilities[mask],
        )
    return grouped


def _sparse_bucket(items: Sequence[PreparedGame], train: Sequence[PreparedGame], config: CandidateConfig, atom_vectors: Mapping[str, np.ndarray], atom_names: Sequence[str]) -> np.ndarray:
    rows = _feature_tokens(list(train) + list(items), config, atom_vectors, atom_names)
    train_rows = rows[: len(train)]
    item_rows = rows[len(train) :]
    vocabulary = _vocabulary(train_rows, config.support)
    values: list[float] = []
    for row in item_rows:
        static = [key for key, _ in row if key.startswith(("CH|", "ALLY|", "ALLYCH|", "CTR|", "SAME|"))]
        values.append(float(sum(key not in vocabulary for key in static)))
    return np.asarray(values)


def _run_ablations(
    development: Sequence[PreparedGame],
    final_holdout: Sequence[PreparedGame],
    selected: CandidateConfig,
    atom_vectors: Mapping[str, np.ndarray],
    atom_names: Sequence[str],
    items: Sequence[PreparedGame],
) -> dict[str, Any]:
    variants = {
        "selected": selected,
        "without_atoms": replace(selected, name="without_atoms", atoms=False, atom_interactions=False),
        "without_ally": replace(selected, name="without_ally", ally=False),
        "without_counters": replace(selected, name="without_counters", counter=False, same_role=False),
        "without_calibration": replace(selected, name="without_calibration", calibration=False),
        "without_region_calibration": replace(selected, name="without_region_calibration", region_calibration=False),
        "without_patch_calibration": replace(selected, name="without_patch_calibration", patch_calibration=False),
        "without_tournament_calibration": replace(selected, name="without_tournament_calibration", tournament_calibration=False),
        "without_comfort": replace(selected, name="without_comfort", comfort=False),
    }
    out: dict[str, Any] = {}
    for name, config in variants.items():
        rows = _feature_tokens(items, config, atom_vectors, atom_names)
        row_lookup = {str(item.game["game_uid"]): row for item, row in zip(items, rows)}
        probabilities, vocabulary_size = _fit_predict(development, final_holdout, config, atom_vectors, atom_names, row_lookup)
        out[name] = {
            "config": config.__dict__,
            "vocabulary_size": vocabulary_size,
            "metrics": metrics([int(item.game["y"]) for item in final_holdout], probabilities),
        }
    return out


def run_experiment(
    players_path: Path,
    atom_path: Path,
    *,
    output_dir: Path,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    max_workers: int | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    players = pd.read_parquet(players_path)
    items, source_meta = prepare_games(players, atom_path)
    atom_vectors, atom_names, atom_sha = _atom_arrays(atom_path)
    development, final_holdout, folds = fixed_chronological_folds(items)
    configs = [
        CandidateConfig("role_champion", ally=False, counter=False, same_role=False, atoms=False, atom_interactions=False, calibration=False, c=0.10),
        CandidateConfig("role_champion_calibrated", ally=False, counter=False, same_role=False, atoms=False, atom_interactions=False, c=0.10),
        CandidateConfig("role_ally_counter", atoms=False, atom_interactions=False, calibration=False, c=0.03),
        CandidateConfig("role_ally_counter_calibrated", atoms=False, atom_interactions=False, c=0.03),
        CandidateConfig("composition_atoms", ally=False, counter=False, same_role=False, c=0.03),
        CandidateConfig("composition_atoms_calibrated", ally=False, counter=False, same_role=False, c=0.03),
        CandidateConfig("full_composition", c=0.03),
        CandidateConfig("full_composition_regularized", support=40, c=0.10),
        CandidateConfig("full_composition_comfort", comfort=True, c=0.03),
        CandidateConfig("full_composition_strict", support=40, c=0.03),
    ]
    dev_results: list[dict[str, Any]] = []
    workers = max_workers or min(10, os.cpu_count() or 1)
    # Token generation is Python-heavy.  Use processes so candidate searches
    # use independent cores instead of contending on the interpreter lock.
    process_context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=process_context,
        initializer=_worker_init,
        initargs=(items, atom_vectors, atom_names),
    ) as executor:
        futures = {
            executor.submit(_dev_candidate_worker, config): config.name
            for config in configs
        }
        for future in as_completed(futures):
            dev_results.append(future.result())
    dev_results.sort(key=lambda row: (-(row["mean_auc"] or 0.0), row["mean_log_loss"] or 9.0, row["config"]["name"]))
    selected = CandidateConfig(**dev_results[0]["config"])
    selected_rows = _feature_tokens(items, selected, atom_vectors, atom_names)
    selected_lookup = {str(item.game["game_uid"]): row for item, row in zip(items, selected_rows)}
    final_probabilities, final_vocabulary_size = _fit_predict(development, final_holdout, selected, atom_vectors, atom_names, selected_lookup)
    final_y = np.asarray([int(item.game["y"]) for item in final_holdout], dtype=float)
    final_metrics = metrics(final_y, final_probabilities)
    final_metrics["bootstrap_95"] = _bootstrap(final_y, final_probabilities, bootstrap_reps, seed)
    final_metrics["by_region"] = _group_metrics(final_holdout, final_probabilities, "region")
    final_metrics["by_patch"] = _group_metrics(final_holdout, final_probabilities, "patch")
    final_metrics["by_event_kind"] = _group_metrics(final_holdout, final_probabilities, "event_kind")
    final_metrics["by_roster_change"] = _group_metrics(final_holdout, final_probabilities, "roster_change")
    sparse_count = _sparse_bucket(final_holdout, development, selected, atom_vectors, atom_names)
    final_metrics["sparse_evidence"] = {
        "definition": "at least one role-champion or interaction term was unseen in development",
        "sparse_n": int((sparse_count > 0).sum()),
        "dense_n": int((sparse_count == 0).sum()),
        "sparse": metrics(final_y[sparse_count > 0], final_probabilities[sparse_count > 0]),
        "dense": metrics(final_y[sparse_count == 0], final_probabilities[sparse_count == 0]),
    }
    ablations = _run_ablations(development, final_holdout, selected, atom_vectors, atom_names, items)
    output = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "authority": "unavailable",
        "public_probability": False,
        "estimand": "pre_match_composition_probability",
        "allowed_inputs": [
            "ten champion picks and roles",
            "role-conditioned champion effects",
            "exact ally synergy",
            "exact enemy counters",
            "same-role terms",
            "LCC-derived atomized archetype interactions",
            "patch, region, and tournament calibration",
            "declared pre-event champion comfort as pick-count familiarity only",
        ],
        "excluded_inputs": sorted(FORBIDDEN_FEATURE_TERMS),
        "source": {
            **source_meta,
            "players_path": str(players_path),
            "players_sha256": _sha256(players_path),
            "atom_path": str(atom_path),
            "atom_sha256": atom_sha,
        },
        "split": {
            "protocol": "chronological_expanding_development_and_sealed_final_holdout",
            "development_n": len(development),
            "final_holdout_n": len(final_holdout),
            "development_date_min": str(development[0].game["date"]),
            "development_date_max": str(development[-1].game["date"]),
            "final_date_min": str(final_holdout[0].game["date"]),
            "final_date_max": str(final_holdout[-1].game["date"]),
            "development_folds": [
                {"train_n": len(train), "validation_n": len(validation), "train_through": str(train[-1].game["date"]), "validation_through": str(validation[-1].game["date"])}
                for train, validation in folds
            ],
            "final_holdout_sealed_during_selection": True,
        },
        "window_bug_fix": {
            "status": "fixed",
            "source": "lol_kills/research/composition_signal.py",
            "rule": "attach each history candidate to the local window payload",
        },
        "development_selection": {
            "candidate_count": len(configs),
            "workers": workers,
            "results": dev_results,
            "selected": selected.__dict__,
        },
        "final": {
            "config": selected.__dict__,
            "vocabulary_size": final_vocabulary_size,
            "metrics": final_metrics,
            "ablations": ablations,
            "auc_floor_0_70_passes": bool(final_metrics.get("auc") is not None and final_metrics["auc"] > 0.70),
            "promotion_status": "candidate_auc_floor_passed" if bool(final_metrics.get("auc") is not None and final_metrics["auc"] > 0.70) else "blocked_auc_floor",
            "selection_holdout_consumed": True,
            "comfort_authority": "historical_pick_count_familiarity_only; no declared_comfort_receipt",
            "patch_transfer": {
                "source_patch": "16.16",
                "public_patch": "26.16",
                "metrics": final_metrics.get("by_patch", {}).get("16.16"),
            },
        },
    }
    output["reproducibility"] = {
        "command": "python3 -m lol_kills.research.public_draft_probability --players <path> --atom-path <path> --output-dir <path>",
        "code_sha256": _sha256(Path(__file__)),
        "source_file_sha256": _sha256(players_path),
        "atom_file_sha256": _sha256(atom_path),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(_json_safe(output), indent=2, sort_keys=True), encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--atom-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_experiment(
        args.players,
        args.atom_path,
        output_dir=args.output_dir,
        bootstrap_reps=args.bootstrap_reps,
        max_workers=args.max_workers,
        seed=args.seed,
    )
    final = report["final"]["metrics"]
    print(json.dumps({"selected": report["development_selection"]["selected"], "final": final}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
