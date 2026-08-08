"""Build a role-specific champion interaction topology from prior maps.

This is an exploratory identity representation.  It separates ordinary
role/champion strength from residual ally/enemy interaction profiles, applies
empirical-Bayes shrinkage, and uses deterministic PCA for the coordinates.
Coordinates are diagnostics until they pass a chronological forecast test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from lol_kills.draft_recommendation import build_games
from lol_kills.research.temporal_draft_runtime import (
    _recency_weights,
    _source_frame,
)


SCHEMA_VERSION = "scryglass:champion-topology:v1"
ROLES = ("top", "jng", "mid", "bot", "sup")
DEFAULT_PRIOR_GAMES = 24.0
DEFAULT_HALF_LIFE_DAYS = 365


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def _role_champion_rows(
    games: list[dict[str, Any]],
) -> tuple[sparse.csr_matrix, dict[str, int]]:
    keys = sorted(
        {
            f"RC|{role}|{game[side][role]['champion']}"
            for game in games
            for side in ("blue", "red")
            for role in ROLES
        }
    )
    vocabulary = {key: index for index, key in enumerate(keys)}
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row_index, game in enumerate(games):
        for side, sign in (("blue", 1.0), ("red", -1.0)):
            for role in ROLES:
                key = f"RC|{role}|{game[side][role]['champion']}"
                rows.append(row_index)
                columns.append(vocabulary[key])
                values.append(sign)
    return sparse.csr_matrix(
        (values, (rows, columns)),
        shape=(len(games), len(vocabulary)),
        dtype=np.float64,
    ), vocabulary


def _fit_baseline(games: list[dict[str, Any]]) -> tuple[SGDClassifier, dict[str, int]]:
    matrix, vocabulary = _role_champion_rows(games)
    outcomes = np.array([int(game["y"]) for game in games], dtype=int)
    weights = _recency_weights(games, games[-1]["date"], DEFAULT_HALF_LIFE_DAYS)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=0.003,
        max_iter=3000,
        tol=1e-6,
        random_state=461,
        average=True,
    )
    model.fit(matrix, outcomes, sample_weight=weights)
    return model, vocabulary


def _posterior_residual(
    residual_sum: float,
    count: float,
    prior_games: float = DEFAULT_PRIOR_GAMES,
) -> float:
    reliability = count / (count + prior_games)
    return float((residual_sum / (count + prior_games)) * reliability)


def _profiles(
    games: list[dict[str, Any]],
    baseline: SGDClassifier,
    *,
    prior_games: float = DEFAULT_PRIOR_GAMES,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, int]]:
    matrix, _ = _role_champion_rows(games)
    p_blue = 1.0 / (
        1.0 + np.exp(-np.clip(baseline.decision_function(matrix), -35, 35))
    )
    stats: dict[tuple[str, str, str, str, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0]
    )
    anchors: set[tuple[str, str]] = set()
    for game_index, game in enumerate(games):
        y = float(game["y"])
        for side, side_probability, side_outcome in (
            ("blue", float(p_blue[game_index]), y),
            ("red", float(1.0 - p_blue[game_index]), 1.0 - y),
        ):
            opponents = "red" if side == "blue" else "blue"
            residual = side_outcome - side_probability
            for role in ROLES:
                own = game[side][role]
                anchor = (role, str(own["champion"]))
                anchors.add(anchor)
                for other_role in ROLES:
                    if other_role == role:
                        continue
                    ally = str(game[side][other_role]["champion"])
                    key = (role, anchor[1], "ally", other_role, ally)
                    stats[key][0] += residual
                    stats[key][1] += 1.0
                for other_role in ROLES:
                    enemy = str(game[opponents][other_role]["champion"])
                    key = (role, anchor[1], "enemy", other_role, enemy)
                    stats[key][0] += residual
                    stats[key][1] += 1.0

    partner_keys = sorted(
        {
            (relation, other_role, other_champion)
            for (_, _, relation, other_role, other_champion) in stats
        }
    )
    feature_keys = [
        f"{relation}|{other_role}|{other_champion}"
        for relation, other_role, other_champion in partner_keys
    ]
    feature_vocabulary = {
        key: index for index, key in enumerate(feature_keys)
    }
    profiles = np.zeros((len(anchors), len(feature_vocabulary)), dtype=float)
    ordered_anchors = sorted(anchors)
    anchor_rows = []
    for row_index, (role, champion) in enumerate(ordered_anchors):
        for relation, other_role, other_champion in partner_keys:
            residual_sum, count = stats.get(
                (role, champion, relation, other_role, other_champion),
                [0.0, 0.0],
            )
            if count:
                reliability = math.sqrt(count / (count + prior_games))
                profiles[
                    row_index,
                    feature_vocabulary[
                        f"{relation}|{other_role}|{other_champion}"
                    ],
                ] = _posterior_residual(
                    residual_sum,
                    count,
                    prior_games,
                ) * reliability
        anchor_rows.append(
            {
                "role": role,
                "champion": champion,
                "anchor_id": f"{role}|{champion}",
            }
        )
    return anchor_rows, profiles, feature_vocabulary


def fit_topology(
    games: list[dict[str, Any]],
    *,
    n_components: int = 3,
    prior_games: float = DEFAULT_PRIOR_GAMES,
) -> dict[str, Any]:
    if len(games) < 100:
        raise ValueError("not enough maps to build champion topology")
    games = sorted(games, key=lambda game: (game["date"], game["game_uid"]))
    baseline, baseline_vocabulary = _fit_baseline(games)
    anchors, profiles, profile_vocabulary = _profiles(
        games,
        baseline,
        prior_games=prior_games,
    )
    if profiles.shape[1] < n_components:
        raise ValueError("profile matrix has fewer columns than components")
    centered = profiles - profiles.mean(axis=1, keepdims=True)
    scaler = StandardScaler(with_mean=False)
    scaled = scaler.fit_transform(centered)
    pca = PCA(n_components=n_components, svd_solver="full", random_state=461)
    coordinates = pca.fit_transform(scaled)
    for index, row in enumerate(anchors):
        row["coordinates"] = [
            round(float(value), 8) for value in coordinates[index]
        ]
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": "champion-topology-v1.0.0",
        "training_games": len(games),
        "as_of": str(games[-1]["date"]),
        "fit": {
            "prior_games": prior_games,
            "half_life_days": DEFAULT_HALF_LIFE_DAYS,
            "components": n_components,
            "baseline": "role/champion additive logistic model",
            "profile": "empirical-Bayes residual ally/enemy interactions",
            "projection": "full-solver PCA on row-centered support-weighted profiles",
        },
        "explained_variance_ratio": [
            round(float(value), 10) for value in pca.explained_variance_ratio_
        ],
        "anchors": anchors,
        "profile_feature_count": len(profile_vocabulary),
        "baseline_feature_count": len(baseline_vocabulary),
        "interpretation": (
            "Coordinates represent similarity in corrected role-specific "
            "interaction profiles. They are not champion strength, causal "
            "identity, or a serving probability by themselves."
        ),
    }


def run(run_dir: Path, output_path: Path, cutoff: pd.Timestamp) -> dict[str, Any]:
    frame, _, _ = _source_frame(run_dir)
    games = [
        game for game in build_games(frame) if game["date"] < cutoff
    ]
    result = fit_topology(games)
    result["source_hashes"] = {
        "runner": _sha_file(Path(__file__)),
        "prior_drafts": _sha_file(
            run_dir
            / "autoresearch/raw/prior-drafts/normalized-prior-draft-rows.jsonl"
        ),
    }
    _write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cutoff", default="2026-07-01T00:00:00Z")
    args = parser.parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    if cutoff.tzinfo is not None:
        cutoff = cutoff.tz_convert("UTC").tz_localize(None)
    result = run(args.run_dir, args.output, cutoff)
    print(
        json.dumps(
            {
                "schema_version": result["schema_version"],
                "training_games": result["training_games"],
                "anchors": len(result["anchors"]),
                "profile_feature_count": result["profile_feature_count"],
                "explained_variance_ratio": result[
                    "explained_variance_ratio"
                ],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
