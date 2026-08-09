#!/usr/bin/env python3
"""Fit state-controlled elemental-drake comparisons on compact GRID pro data.

This model is intentionally associational. It compares drake elements after
conditioning on information available immediately before capture:

* gold, kill, and tower difference;
* loadout value, unspent gold, and leading-player net worth;
* game clock and side;
* champion composition archetypes;
* organization and five-player ratings built from prior GRID games only; and
* exact patch, tournament scope, and calendar year.

The public artifact is gated at 6,000 completed professional games. Model
quality is evaluated on a chronological holdout. Uncertainty is clustered by
series in the bootstrap.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

from lol_kills.draft_archetypes import ARCHETYPE_NAMES, champ_tags
from lol_kills.etl.aliases import normalize_champ, normalize_team
from lol_kills.etl.grid_ingest import _league_for
from lol_kills.research.elemental_drakes import (
    COMPACT_EVENTS_PARQUET,
    COMPACT_GAMES_PARQUET,
    MECHANICS,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "lol" / "models" / "elemental_drake_model.json"
ELEMENTS = [row["id"] for row in MECHANICS]
MIN_PUBLIC_GAMES = 6_000
HOLDOUT_FRACTION = 0.2
PUBLIC_HOLDOUT_START = pd.Timestamp("2026-03-01T00:00:00Z")
PUBLIC_HOLDOUT_END = pd.Timestamp("2026-05-01T00:00:00Z")
RATING_BASE = 1_500.0
ORG_K = 20.0
PLAYER_K = 10.0
STANDARDIZED_CLIP = 8.0
ALPHA_GRID = (0.01, 0.03, 0.1, 0.3)


@dataclass
class FittedLogit:
    scaler: StandardScaler
    model: SGDClassifier
    columns: list[str]

    def predict(self, design: pd.DataFrame) -> np.ndarray:
        aligned = design.reindex(columns=self.columns, fill_value=0.0)
        values = aligned.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Prediction design contains non-finite values.")
        scaled = self.scaler.transform(values)
        if not np.isfinite(scaled).all():
            raise ValueError("Scaled prediction design contains non-finite values.")
        scaled = np.clip(scaled, -STANDARDIZED_CLIP, STANDARDIZED_CLIP)
        score = np.sum(scaled * self.model.coef_[0], axis=1)
        score += float(self.model.intercept_[0])
        probability = expit(score)
        if not np.isfinite(probability).all():
            raise ValueError("Model prediction produced non-finite probabilities.")
        return probability


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return [str(item) for item in parsed if str(item)] if isinstance(parsed, list) else []


def _clean_player(value: str) -> str:
    return " ".join(str(value).strip().casefold().split())


def _expected(left: float, right: float) -> float:
    return 1.0 / (1.0 + 10 ** ((right - left) / 400.0))


def _player_average(players: Sequence[str], ratings: Mapping[str, float]) -> float:
    keys = [_clean_player(player) for player in players if _clean_player(player)]
    return float(np.mean([ratings.get(key, RATING_BASE) for key in keys])) if keys else RATING_BASE


def pregame_strengths(
    games: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Build org and player ratings using only outcomes earlier in time."""
    first_times = (
        events.groupby(["series_id", "game_id"], as_index=False)["occurred_at"]
        .min()
        .rename(columns={"occurred_at": "first_drake_at"})
    )
    ordered = games.merge(first_times, on=["series_id", "game_id"], how="left")
    ordered["order_time"] = pd.to_datetime(
        ordered["first_drake_at"].fillna(ordered["date"]),
        errors="coerce",
        utc=True,
    )
    ordered = ordered.sort_values(["order_time", "series_id", "game_id"]).reset_index(drop=True)

    org_ratings: dict[str, float] = defaultdict(lambda: RATING_BASE)
    player_ratings: dict[str, float] = defaultdict(lambda: RATING_BASE)
    rows: list[dict[str, Any]] = []
    for game in ordered.itertuples(index=False):
        team_1 = normalize_team(str(game.team_1_name))
        team_2 = normalize_team(str(game.team_2_name))
        players_1 = _json_list(getattr(game, "team_1_player_ids", None))
        players_2 = _json_list(getattr(game, "team_2_player_ids", None))
        if not players_1:
            players_1 = _json_list(game.team_1_players)
        if not players_2:
            players_2 = _json_list(game.team_2_players)
        org_1 = org_ratings[team_1]
        org_2 = org_ratings[team_2]
        player_1 = _player_average(players_1, player_ratings)
        player_2 = _player_average(players_2, player_ratings)
        rows.append(
            {
                "series_id": game.series_id,
                "game_id": game.game_id,
                "team_1_org_elo": org_1,
                "team_2_org_elo": org_2,
                "team_1_player_elo": player_1,
                "team_2_player_elo": player_2,
                "team_1_roster_coverage": len(players_1),
                "team_2_roster_coverage": len(players_2),
            }
        )
        winner = str(game.winner_team_id or "")
        if not bool(game.complete) or winner not in {str(game.team_1_id), str(game.team_2_id)}:
            continue
        result_1 = 1.0 if winner == str(game.team_1_id) else 0.0
        expected_org_1 = _expected(org_1, org_2)
        org_delta = ORG_K * (result_1 - expected_org_1)
        org_ratings[team_1] += org_delta
        org_ratings[team_2] -= org_delta

        expected_player_1 = _expected(player_1, player_2)
        player_delta = PLAYER_K * (result_1 - expected_player_1)
        for player in players_1:
            player_ratings[_clean_player(player)] += player_delta
        for player in players_2:
            player_ratings[_clean_player(player)] -= player_delta
    return pd.DataFrame(rows)


def _archetype_counts(champions: Sequence[str]) -> dict[str, float]:
    counts = {name: 0.0 for name in ARCHETYPE_NAMES}
    for champion in champions:
        for tag in champ_tags(normalize_champ(champion)):
            counts[tag] += 1.0
    return counts


def _perspective_row(
    game: Mapping[str, Any],
    event: Mapping[str, Any],
    strength: Mapping[str, Any],
) -> dict[str, Any]:
    owner_id = str(event.get("owner_team_id") or "")
    owner_is_1 = owner_id == str(game.get("team_1_id") or "")
    owner_prefix = "team_1" if owner_is_1 else "team_2"
    opponent_prefix = "team_2" if owner_is_1 else "team_1"
    own_champions = _json_list(game.get(f"{owner_prefix}_champions"))
    opponent_champions = _json_list(game.get(f"{opponent_prefix}_champions"))
    own_arch = _archetype_counts(own_champions)
    opponent_arch = _archetype_counts(opponent_champions)
    winner = str(game.get("winner_team_id") or "")
    date = pd.to_datetime(game.get("date"), errors="coerce", utc=True)
    row: dict[str, Any] = {
        "series_id": str(game.get("series_id") or ""),
        "game_id": str(game.get("game_id") or ""),
        "date": date,
        "year": str(date.year) if pd.notna(date) else "unknown",
        "patch": str(game.get("patch") or "unknown"),
        "league": _league_for(str(game.get("tournament") or "")),
        "element": str(event.get("element") or ""),
        "current_element": str(event.get("element") or ""),
        "owner_won": int(owner_id == winner),
        "gold_diff_k": float(event.get("gold_diff") or 0.0) / 1_000.0,
        "loadout_diff_k": float(event.get("loadout_diff") or 0.0) / 1_000.0,
        "unspent_money_diff_k": float(
            event.get("unspent_money_diff") or 0.0
        )
        / 1_000.0,
        "top_player_net_worth_diff_k": float(
            event.get("top_player_net_worth_diff") or 0.0
        )
        / 1_000.0,
        "minute": float(event.get("time_seconds") or 0.0) / 60.0,
        "kill_diff": float(event.get("owner_kills") or 0.0)
        - float(event.get("opponent_kills") or 0.0),
        "tower_diff": float(event.get("owner_towers") or 0.0)
        - float(event.get("opponent_towers") or 0.0),
        "blue": int(
            str(game.get(f"{owner_prefix}_side") or "").lower() == "blue"
        ),
        "state_lag_seconds": float(event.get("state_lag_seconds") or 0.0),
        "org_elo_diff": (
            float(strength.get(f"{owner_prefix}_org_elo") or RATING_BASE)
            - float(strength.get(f"{opponent_prefix}_org_elo") or RATING_BASE)
        )
        / 400.0,
        "player_elo_diff": (
            float(strength.get(f"{owner_prefix}_player_elo") or RATING_BASE)
            - float(strength.get(f"{opponent_prefix}_player_elo") or RATING_BASE)
        )
        / 400.0,
        "roster_coverage": float(
            strength.get(f"{owner_prefix}_roster_coverage") or 0.0
        ),
    }
    for tag in ARCHETYPE_NAMES:
        row[f"own_{tag}"] = own_arch[tag]
        row[f"opp_{tag}"] = opponent_arch[tag]
        row[f"diff_{tag}"] = own_arch[tag] - opponent_arch[tag]
    return row


def prepare_rows(
    games: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    strengths = pregame_strengths(games, events)
    game_lookup = {
        (str(row.series_id), str(row.game_id)): row._asdict()
        for row in games.itertuples(index=False)
    }
    strength_lookup = {
        (str(row.series_id), str(row.game_id)): row._asdict()
        for row in strengths.itertuples(index=False)
    }
    first_rows: list[dict[str, Any]] = []
    stack_rows: list[dict[str, Any]] = []
    for (series_id, game_id), group in events.groupby(["series_id", "game_id"], sort=False):
        key = (str(series_id), str(game_id))
        game = game_lookup.get(key)
        strength = strength_lookup.get(key)
        if not game or not strength or not bool(game.get("complete")):
            continue
        team_ids = {str(game.get("team_1_id") or ""), str(game.get("team_2_id") or "")}
        if str(game.get("winner_team_id") or "") not in team_ids:
            continue
        if (
            len(_json_list(game.get("team_1_champions"))) != 5
            or len(_json_list(game.get("team_2_champions"))) != 5
        ):
            continue
        group = group.sort_values("global_index")
        owner_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: {element: 0 for element in ELEMENTS}
        )
        owner_totals: dict[str, int] = defaultdict(int)
        for event_row in group.itertuples(index=False):
            event = event_row._asdict()
            owner = str(event.get("owner_team_id") or "")
            if (
                owner not in team_ids
                or str(event.get("state_timing") or "") != "previous-envelope"
            ):
                continue
            if float(event.get("state_lag_seconds") or 0.0) > 60:
                continue
            if (
                float(event.get("owner_net_worth") or 0.0) <= 0
                or float(event.get("opponent_net_worth") or 0.0) <= 0
            ):
                continue
            element = str(event.get("element") or "")
            if element not in ELEMENTS:
                continue
            owner_counts[owner][element] += 1
            owner_totals[owner] += 1
            row = _perspective_row(game, event, strength)
            row["global_index"] = int(event.get("global_index") or 0)
            row["owner_stack"] = owner_totals[owner]
            opponent = next(
                (
                    str(team_id)
                    for team_id in (game.get("team_1_id"), game.get("team_2_id"))
                    if str(team_id) != owner
                ),
                "",
            )
            row["opponent_stack"] = owner_totals[opponent]
            for drake in ELEMENTS:
                row[f"own_count_{drake}"] = owner_counts[owner][drake]
                row[f"opp_count_{drake}"] = owner_counts[opponent][drake]
            stack_rows.append(row)
            if int(event.get("global_index") or 0) == 1:
                first_rows.append(row.copy())
    first = pd.DataFrame(first_rows)
    stack = pd.DataFrame(stack_rows)
    if not first.empty:
        first = first[
            first["minute"].between(4.5, 20)
            & first["gold_diff_k"].abs().le(8)
            & first["element"].isin(ELEMENTS)
        ].reset_index(drop=True)
    if not stack.empty:
        stack = stack[
            stack["owner_stack"].between(1, 4)
            & stack["minute"].between(4.5, 50)
            & stack["gold_diff_k"].abs().le(15)
        ].reset_index(drop=True)
    return first, stack


def _design_first(rows: pd.DataFrame) -> pd.DataFrame:
    columns: dict[str, pd.Series] = {}
    base_numeric = [
        "gold_diff_k",
        "loadout_diff_k",
        "unspent_money_diff_k",
        "top_player_net_worth_diff_k",
        "minute",
        "kill_diff",
        "tower_diff",
        "blue",
        "org_elo_diff",
        "player_elo_diff",
        "roster_coverage",
    ]
    for column in base_numeric:
        columns[column] = pd.to_numeric(rows[column], errors="coerce").fillna(0.0)
    for tag in ARCHETYPE_NAMES:
        columns[f"diff_{tag}"] = pd.to_numeric(
            rows[f"diff_{tag}"], errors="coerce"
        ).fillna(0.0)
        columns[f"diff_{tag}_x_minute"] = (
            columns[f"diff_{tag}"] * columns["minute"]
        )
    for element in ELEMENTS:
        indicator = (rows["element"] == element).astype(float)
        columns[f"element_{element}"] = indicator
        columns[f"{element}_x_gold"] = indicator * columns["gold_diff_k"]
        columns[f"{element}_x_loadout"] = indicator * columns["loadout_diff_k"]
        columns[f"{element}_x_top_player"] = (
            indicator * columns["top_player_net_worth_diff_k"]
        )
        columns[f"{element}_x_minute"] = indicator * columns["minute"]
        for tag in ARCHETYPE_NAMES:
            columns[f"{element}_x_{tag}"] = indicator * columns[f"diff_{tag}"]
            columns[f"{element}_x_{tag}_x_minute"] = (
                indicator
                * columns[f"diff_{tag}"]
                * columns["minute"]
            )
    frame = pd.DataFrame(columns, index=rows.index)
    categories = pd.get_dummies(
        rows[["league", "patch", "year"]].astype(str),
        prefix=["league", "patch", "year"],
        dtype=float,
    )
    return pd.concat([frame, categories], axis=1).astype(float)


def _design_stack(rows: pd.DataFrame) -> pd.DataFrame:
    first = _design_first(rows.assign(element=rows["current_element"]))
    columns: dict[str, pd.Series] = {
        "owner_stack": pd.to_numeric(
            rows["owner_stack"], errors="coerce"
        ).fillna(0.0),
        "opponent_stack": pd.to_numeric(
            rows["opponent_stack"], errors="coerce"
        ).fillna(0.0),
    }
    for element in ELEMENTS:
        count = pd.to_numeric(rows[f"own_count_{element}"], errors="coerce").fillna(0.0)
        columns[f"own_count_{element}"] = count
        columns[f"opp_count_{element}"] = pd.to_numeric(
            rows[f"opp_count_{element}"], errors="coerce"
        ).fillna(0.0)
        for tag in ARCHETYPE_NAMES:
            columns[f"stack_{element}_x_{tag}"] = count * pd.to_numeric(
                rows[f"own_{tag}"], errors="coerce"
            ).fillna(0.0)
    stack = pd.DataFrame(columns, index=rows.index)
    return pd.concat([first, stack], axis=1).astype(float)


def _fit(
    design: pd.DataFrame,
    outcome: np.ndarray,
    *,
    alpha: float = 0.1,
    sample_weight: np.ndarray | None = None,
) -> FittedLogit:
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=float)
        if len(sample_weight) != len(design):
            raise ValueError("Sample weights must align with the model design.")
        positive = sample_weight > 0
        design = design.loc[positive]
        outcome = outcome[positive]
        sample_weight = sample_weight[positive]
    variance = design.var(axis=0, ddof=0)
    kept_columns = variance.index[variance.gt(1e-12)].tolist()
    if not kept_columns:
        raise ValueError("Model design has no varying columns.")
    design = design.loc[:, kept_columns]
    values = design.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Model design contains non-finite values.")
    if len(np.unique(outcome)) < 2:
        raise ValueError("Model outcome must contain both map-win classes.")
    scaler = StandardScaler()
    if sample_weight is None:
        scaler.fit(values)
    else:
        weight_sum = float(sample_weight.sum())
        if not math.isfinite(weight_sum) or weight_sum <= 0:
            raise ValueError("Sample weights must have a positive finite sum.")
        mean = np.sum(values * sample_weight[:, None], axis=0) / weight_sum
        centered = values - mean
        variance = (
            np.sum(centered * centered * sample_weight[:, None], axis=0)
            / weight_sum
        )
        scale = np.sqrt(variance)
        scale[~np.isfinite(scale) | (scale <= 1e-12)] = 1.0
        scaler.mean_ = mean
        scaler.var_ = variance
        scaler.scale_ = scale
        scaler.n_features_in_ = values.shape[1]
        scaler.n_samples_seen_ = weight_sum
    scaled = scaler.transform(values)
    if not np.isfinite(scaled).all():
        raise ValueError("Scaled model design contains non-finite values.")
    scaled = np.clip(scaled, -STANDARDIZED_CLIP, STANDARDIZED_CLIP)
    model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=5_000,
        tol=1e-6,
        random_state=461,
        average=True,
    ).fit(scaled, outcome, sample_weight=sample_weight)
    if not np.isfinite(model.coef_).all() or not np.isfinite(model.intercept_).all():
        raise ValueError("Model fit produced non-finite coefficients.")
    if float(np.abs(model.coef_).max()) > 1_000:
        raise ValueError(
            "Model fit produced an unstable coefficient magnitude: "
            f"{float(np.abs(model.coef_).max()):.3g}"
        )
    return FittedLogit(scaler=scaler, model=model, columns=list(design.columns))


def _ece(outcome: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    total = len(outcome)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (probability >= low) & (
            probability < high if high < 1 else probability <= high
        )
        if not mask.any():
            continue
        error += mask.mean() * abs(outcome[mask].mean() - probability[mask].mean())
    return float(error if total else math.nan)


def _diagnostics(
    rows: pd.DataFrame,
    design_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> tuple[dict[str, Any], FittedLogit]:
    ordered = rows.sort_values(["date", "series_id", "game_id"]).reset_index(drop=True)
    train = ordered.loc[ordered["date"] < PUBLIC_HOLDOUT_START]
    test = ordered.loc[
        ordered["date"].between(
            PUBLIC_HOLDOUT_START,
            PUBLIC_HOLDOUT_END,
            inclusive="left",
        )
    ]
    if train.empty or test.empty:
        raise ValueError(
            "The prespecified March-April 2026 holdout requires both earlier "
            "training rows and in-window evaluation rows."
        )
    if set(train["series_id"].astype(str)) & set(test["series_id"].astype(str)):
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
            int(len(train_series_order) * (1 - HOLDOUT_FRACTION)),
        ),
    )
    full_design = design_fn(ordered)
    inner_train_series = set(
        train_series_order.iloc[:inner_split]["series_id"].astype(str)
    )
    inner_train_mask = train["series_id"].astype(str).isin(inner_train_series)
    inner_train = train.loc[inner_train_mask]
    inner_validation = train.loc[~inner_train_mask]
    if inner_train.empty or inner_validation.empty:
        raise ValueError(
            "Regularization selection requires at least two chronological "
            "training-series partitions."
        )
    inner_y = inner_train["owner_won"].to_numpy(dtype=int)
    inner_validation_y = inner_validation["owner_won"].to_numpy(dtype=int)
    alpha_scores: dict[float, float] = {}
    for alpha in ALPHA_GRID:
        candidate = _fit(
            full_design.loc[inner_train.index],
            inner_y,
            alpha=alpha,
        )
        candidate_probability = candidate.predict(
            full_design.loc[inner_validation.index]
        )
        alpha_scores[alpha] = float(
            brier_score_loss(inner_validation_y, candidate_probability)
        )
    selected_alpha = min(
        ALPHA_GRID,
        key=lambda alpha: (alpha_scores[alpha], alpha),
    )
    train_design = full_design.loc[train.index]
    test_design = full_design.loc[test.index]
    train_y = train["owner_won"].to_numpy(dtype=int)
    test_y = test["owner_won"].to_numpy(dtype=int)
    holdout_fit = _fit(train_design, train_y, alpha=selected_alpha)
    probability = holdout_fit.predict(test_design)
    diagnostics = {
        "trainRows": len(train),
        "holdoutRows": len(test),
        "holdoutStart": (
            test["date"].min().isoformat() if len(test) and pd.notna(test["date"].min()) else None
        ),
        "holdoutEnd": PUBLIC_HOLDOUT_END.isoformat(),
        "trainSeries": int(train["series_id"].nunique()),
        "holdoutSeries": int(test["series_id"].nunique()),
        "postHoldoutRows": int((ordered["date"] >= PUBLIC_HOLDOUT_END).sum()),
        "auc": round(float(roc_auc_score(test_y, probability)), 4)
        if len(np.unique(test_y)) == 2
        else None,
        "brier": round(float(brier_score_loss(test_y, probability)), 4),
        "nullBrier": round(
            float(brier_score_loss(test_y, np.repeat(train_y.mean(), len(test_y)))),
            4,
        ),
        "logLoss": round(float(log_loss(test_y, probability, labels=[0, 1])), 4),
        "ece10": round(_ece(test_y, probability), 4),
        "selectedAlpha": selected_alpha,
        "innerValidationBrier": round(alpha_scores[selected_alpha], 4),
    }
    final_fit = _fit(
        full_design,
        ordered["owner_won"].to_numpy(dtype=int),
        alpha=selected_alpha,
    )
    return diagnostics, final_fit


def _counterfactual_first(
    fit: FittedLogit,
    reference: pd.DataFrame,
) -> dict[str, float]:
    values = {}
    for element in ELEMENTS:
        changed = reference.copy()
        changed["element"] = element
        values[element] = float(fit.predict(_design_first(changed)).mean())
    return values


def _effect_rows(values: Mapping[str, float]) -> list[dict[str, Any]]:
    center = float(np.mean(list(values.values())))
    return [
        {
            "element": element,
            "adjustedWinProbability": round(values[element], 4),
            "relativePp": round((values[element] - center) * 100, 2),
        }
        for element in sorted(values, key=values.get, reverse=True)
    ]


def _bootstrap_first(
    rows: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    replicates: int,
    seed: int,
    alpha: float,
) -> dict[str, dict[str, float]]:
    if replicates <= 0:
        return {}
    rng = np.random.default_rng(seed)
    series_ids = rows["series_id"].drop_duplicates().to_numpy()
    series_values = rows["series_id"].astype(str)
    draws: dict[str, list[float]] = {element: [] for element in ELEMENTS}
    for _ in range(replicates):
        sampled = rng.choice(series_ids, size=len(series_ids), replace=True)
        multiplicity = Counter(str(series) for series in sampled)
        mask = series_values.isin(multiplicity)
        boot = rows.loc[mask]
        if boot["owner_won"].nunique() < 2:
            continue
        weights = series_values.loc[mask].map(multiplicity).to_numpy(dtype=float)
        fit = _fit(
            _design_first(boot),
            boot["owner_won"].to_numpy(dtype=int),
            alpha=alpha,
            sample_weight=weights,
        )
        values = _counterfactual_first(fit, reference)
        center = float(np.mean(list(values.values())))
        for element in ELEMENTS:
            draws[element].append((values[element] - center) * 100)
    return {
        element: {
            "lowPp": round(float(np.quantile(values, 0.025)), 2),
            "highPp": round(float(np.quantile(values, 0.975)), 2),
        }
        for element, values in draws.items()
        if values
    }


def build_model_artifact(
    *,
    games_path: Path = COMPACT_GAMES_PARQUET,
    events_path: Path = COMPACT_EVENTS_PARQUET,
    min_games: int = MIN_PUBLIC_GAMES,
    bootstrap: int = 200,
    seed: int = 461,
) -> dict[str, Any]:
    games = pd.read_parquet(games_path)
    events = pd.read_parquet(events_path).rename(columns={"occurredAt": "occurred_at"})
    if "occurred_at" not in events.columns:
        events["occurred_at"] = events.get("date")
    valid_winner = (
        games["winner_team_id"].astype(str).eq(games["team_1_id"].astype(str))
        | games["winner_team_id"].astype(str).eq(games["team_2_id"].astype(str))
    )
    distinct_teams = (
        games["team_1_id"].astype(str) != games["team_2_id"].astype(str)
    )
    completed_games = int(
        games.loc[
            games["complete"].astype(bool) & valid_winner & distinct_teams,
            ["series_id", "game_id"],
        ]
        .drop_duplicates()
        .shape[0]
    )
    if completed_games < min_games:
        return {
            "status": "gated",
            "games": completed_games,
            "requiredGames": min_games,
            "reason": "The prespecified professional-game threshold has not been reached.",
        }
    first, stack = prepare_rows(games, events)
    if len(first) < min_games * 0.8:
        raise RuntimeError(
            f"Only {len(first)} first-drake rows survived validation from "
            f"{completed_games} completed games."
        )

    cohort = {
        "completedGames": completed_games,
        "firstDrakeRows": len(first),
        "stackRows": len(stack),
        "series": int(games["series_id"].nunique()),
        "dateMin": str(pd.to_datetime(games["date"], utc=True).min().date()),
        "dateMax": str(pd.to_datetime(games["date"], utc=True).max().date()),
    }
    first_diagnostics, first_fit = _diagnostics(first, _design_first)
    first_reference = first.sample(min(len(first), 2_500), random_state=seed)
    first_values = _counterfactual_first(first_fit, first_reference)
    stack_diagnostics, stack_fit = _diagnostics(stack, _design_stack)
    validation = {
        "firstBrierBeatsNull": (
            first_diagnostics["brier"] < first_diagnostics["nullBrier"]
        ),
        "firstEceAtMostEightPp": first_diagnostics["ece10"] <= 0.08,
        "stackBrierBeatsNull": (
            stack_diagnostics["brier"] < stack_diagnostics["nullBrier"]
        ),
        "stackEceAtMostTenPp": stack_diagnostics["ece10"] <= 0.10,
    }
    if not all(validation.values()):
        return {
            "status": "gated",
            "games": completed_games,
            "requiredGames": min_games,
            "reason": (
                "The sample threshold passed, but the prespecified chronological "
                "holdout diagnostics did not. Adjusted element effects remain hidden."
            ),
            "cohort": cohort,
            "validation": validation,
            "diagnostics": {
                "firstDragon": first_diagnostics,
                "compounding": stack_diagnostics,
            },
        }

    intervals = _bootstrap_first(
        first,
        first_reference,
        replicates=bootstrap,
        seed=seed,
        alpha=float(first_diagnostics["selectedAlpha"]),
    )
    first_effects = _effect_rows(first_values)
    for row in first_effects:
        row["interval95"] = intervals.get(row["element"])

    legal_reference = stack[stack["owner_stack"] == 4]
    legal_paths: list[dict[str, Any]] = []
    if not legal_reference.empty:
        legal_reference = legal_reference.sample(
            min(len(legal_reference), 2_000),
            random_state=seed,
        )
        for control_pattern in ("perfect-control", "second-traded"):
            pattern_rows: list[dict[str, Any]] = []
            for first_element in ELEMENTS:
                for second_element in ELEMENTS:
                    if second_element == first_element:
                        continue
                    for rift_element in ELEMENTS:
                        if rift_element in {first_element, second_element}:
                            continue
                        changed = legal_reference.copy()
                        changed["current_element"] = rift_element
                        changed["owner_stack"] = 4
                        changed["opponent_stack"] = (
                            0 if control_pattern == "perfect-control" else 1
                        )
                        for element in ELEMENTS:
                            if control_pattern == "perfect-control":
                                own_count = (
                                    int(element == first_element)
                                    + int(element == second_element)
                                    + 2 * int(element == rift_element)
                                )
                            else:
                                own_count = (
                                    int(element == first_element)
                                    + 3 * int(element == rift_element)
                                )
                            changed[f"own_count_{element}"] = own_count
                            changed[f"opp_count_{element}"] = int(
                                control_pattern == "second-traded"
                                and element == second_element
                            )
                        probability = float(
                            stack_fit.predict(_design_stack(changed)).mean()
                        )
                        pattern_rows.append(
                            {
                                "controlPattern": control_pattern,
                                "spawnPath": [
                                    first_element,
                                    second_element,
                                    rift_element,
                                    rift_element,
                                ],
                                "teamPath": (
                                    [
                                        first_element,
                                        second_element,
                                        rift_element,
                                        rift_element,
                                    ]
                                    if control_pattern == "perfect-control"
                                    else [
                                        first_element,
                                        rift_element,
                                        rift_element,
                                        rift_element,
                                    ]
                                ),
                                "adjustedWinProbability": probability,
                            }
                        )
            center = float(
                np.mean(
                    [row["adjustedWinProbability"] for row in pattern_rows]
                )
            )
            for row in pattern_rows:
                row["adjustedWinProbability"] = round(
                    row["adjustedWinProbability"],
                    4,
                )
                row["relativePp"] = round(
                    (row["adjustedWinProbability"] - center) * 100,
                    2,
                )
            legal_paths.extend(pattern_rows)
        legal_paths.sort(key=lambda row: row["adjustedWinProbability"], reverse=True)

    return {
        "status": "ready",
        "cohort": cohort,
        "validation": validation,
        "estimand": (
            "Adjusted association between elemental identity and the capturing "
            "team's map-win probability among observed professional captures."
        ),
        "firstDragon": {
            "diagnostics": first_diagnostics,
            "elements": first_effects,
            "bootstrap": {
                "replicates": bootstrap,
                "cluster": "GRID series",
                "seed": seed,
            },
        },
        "compounding": {
            "diagnostics": stack_diagnostics,
            "legalPaths": legal_paths,
            "legalPathRule": (
                "Global spawns are A, B, C, C. The first owner reaches four "
                "stacks as A, B, C, C under perfect control or A, C, C, C "
                "after trading the second spawn."
            ),
        },
        "controls": [
            "pre-capture gold, kills, towers, clock, and side",
            "pre-capture loadout value, unspent gold, and leading-player net worth",
            "own and opposing champion archetype counts",
            "prior-game organization Elo",
            "prior-game five-player aggregate Elo",
            "tournament scope and calendar year",
            "exact GRID title version",
            "regularization selected inside pre-March 2026 training data",
            "locked March-April 2026 temporal evaluation window",
            "continuous model features standardized and capped at eight standard deviations",
        ],
        "limitations": [
            "Capture is selected, not randomized; estimates are not causal objective values.",
            "GRID state is the latest event state before capture and may lag by up to 60 seconds.",
            "Champion fit uses static multi-label archetypes rather than item-level kit simulation.",
            "May-July 2026 games were excluded from evaluation and used only in the post-validation full refit.",
            "Legal-path results are exploratory point estimates without path-level bootstrap intervals.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=Path, default=COMPACT_GAMES_PARQUET)
    parser.add_argument("--events", type=Path, default=COMPACT_EVENTS_PARQUET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-games", type=int, default=MIN_PUBLIC_GAMES)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=461)
    args = parser.parse_args(argv)
    artifact = build_model_artifact(
        games_path=args.games,
        events_path=args.events,
        min_games=args.min_games,
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(
        f"[elemental-drake-model] status={artifact['status']} "
        f"games={artifact.get('cohort', {}).get('completedGames', artifact.get('games', 0))} "
        f"output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
