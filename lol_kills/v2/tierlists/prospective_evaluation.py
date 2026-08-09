"""Run a time-safe development evaluation for L9 tier values.

The evaluator rebuilds the tier value before each eligible map from the
played membership available before that map. It then evaluates future map
outcomes with a nested adapter that removes the evaluated champion's ordinary
role contribution and inserts the pre-event Tier Value row.

This module is diagnostic. The OE warehouse has no source-observed series ID
for most scopes, so the current run uses the separately documented dependence
cluster proxy. The proxy supports diagnostics and fold blocking. It grants no
independent authority and cannot promote a tier-list artifact.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from .artifact import load_frozen_terminal_model
from .model import TERMINAL_MODEL_ARTIFACT


SCHEMA_VERSION = "scryglass:tierlist-prospective-evaluation:v1"
SOURCE_LOCATOR = Path("data/lol/warehouse/parquet/oe_player_games.parquet")
CLUSTER_PROXY_LOCATOR = Path(
    "data/lol/v2/models/draft-interactions/series-cluster-proxy.json"
)
TARGET_LEAGUES = ("LEC", "LCS", "LCK", "LPL")
TARGET_EVENTS = ("MSI", "EWC")
ROLES = ("top", "jungle", "mid", "bot", "support")
SOURCE_ROLE_ALIASES = {
    "top": "top",
    "jng": "jungle",
    "jungle": "jungle",
    "mid": "mid",
    "bot": "bot",
    "sup": "support",
    "support": "support",
}
SOURCE_COLUMNS = (
    "gameid",
    "date",
    "patch",
    "league",
    "position",
    "champion",
    "result",
    "side",
    "team_key",
    "event_kind",
    "competition_tier",
)
SCOPE_COLUMNS = TARGET_LEAGUES + TARGET_EVENTS
TEAM_UPDATE_RATE = 0.25
BOOTSTRAP_SEED = 20260808


class TierListEvaluationError(ValueError):
    """Raised when the evaluation cannot establish a time-safe input state."""


@dataclass(frozen=True)
class MapRecord:
    map_id: str
    event_start: datetime
    scope: str
    patch: str
    teams: Mapping[str, str]
    results: Mapping[str, int]
    picks: Mapping[str, Mapping[str, str]]
    dependence_cluster_id: str | None


@dataclass(frozen=True)
class EvaluationRow:
    row_id: str
    map_id: str
    event_start: datetime
    scope: str
    patch: str
    role: str
    champion: str
    side: str
    label: int
    strength_logit: float
    draft_other_logit: float
    tier_feature: float
    tier_value_pp: float
    played_champion_count: int
    dependence_cluster_id: str | None


@dataclass(frozen=True)
class _FittedLogistic:
    coefficients: np.ndarray
    intercept: float

    @property
    def coef_(self) -> np.ndarray:
        return self.coefficients.reshape(1, -1)

    @property
    def intercept_(self) -> np.ndarray:
        return np.asarray([self.intercept], dtype=float)

    def predict_proba(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=float)
        probability = np.asarray(
            _sigmoid(self.intercept + _linear_predictor(matrix, self.coefficients)),
            dtype=float,
        )
        return np.column_stack((1.0 - probability, probability))


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _sigmoid(value: float | np.ndarray) -> float | np.ndarray:
    if isinstance(value, np.ndarray):
        return 1.0 / (1.0 + np.exp(-np.clip(value, -40.0, 40.0)))
    if value >= 40.0:
        return 1.0
    if value <= -40.0:
        return 0.0
    return 1.0 / (1.0 + math.exp(-value))


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return np.log(clipped / (1.0 - clipped))


def _linear_predictor(features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    vector = np.asarray(coefficients, dtype=float)
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
        raise TierListEvaluationError("logistic predictor arrays have incompatible shapes")
    if not np.isfinite(matrix).all() or not np.isfinite(vector).all():
        raise TierListEvaluationError("logistic predictor arrays contain non-finite values")
    if np.max(np.abs(matrix), initial=0.0) > 1e6 or np.max(np.abs(vector), initial=0.0) > 1e6:
        raise TierListEvaluationError("logistic predictor arrays exceed the safe numeric bound")
    result = np.asarray(np.einsum("ij,j->i", matrix, vector, optimize=True), dtype=float)
    if not np.isfinite(result).all():
        raise TierListEvaluationError("logistic predictor produced non-finite values")
    return result


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise TierListEvaluationError("evaluation cutoff must include a timezone")
    return parsed.astimezone(timezone.utc)


def _naive_utc(value: Any) -> datetime:
    parsed = pd.Timestamp(value)
    if pd.isna(parsed):
        raise TierListEvaluationError("map timestamp is missing")
    if parsed.tzinfo is not None:
        parsed = parsed.tz_convert("UTC").tz_localize(None)
    return parsed.to_pydatetime().replace(tzinfo=timezone.utc)


def _patch_token(value: Any) -> str:
    if value is None or pd.isna(value):
        raise TierListEvaluationError("patch is missing")
    text = str(value).strip()
    if not re.fullmatch(r"\d{1,2}\.\d{1,2}", text):
        raise TierListEvaluationError(f"patch is invalid: {text}")
    major, minor = text.split(".")
    return f"{int(major)}.{int(minor)}"


def _scope_for_frame(frame: pd.DataFrame) -> str | None:
    events = {str(value).strip().lower() for value in frame["event_kind"].dropna()}
    leagues = {str(value).strip().upper() for value in frame["league"].dropna()}
    if len(events) > 1 or len(leagues) != 1:
        return None
    event = next(iter(events), "")
    league = next(iter(leagues))
    if event in {event.lower() for event in TARGET_EVENTS}:
        return event.upper()
    if league in TARGET_LEAGUES and set(frame["competition_tier"].dropna().astype(str)) == {"tier1"}:
        return league
    return None


def _load_cluster_proxy(root: Path) -> tuple[dict[str, str], str, bool]:
    path = root / CLUSTER_PROXY_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise TierListEvaluationError("dependence cluster proxy is missing")
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TierListEvaluationError("dependence cluster proxy is invalid JSON") from exc
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise TierListEvaluationError("dependence cluster proxy assignments are missing")
    result: dict[str, str] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise TierListEvaluationError("dependence cluster proxy assignment is malformed")
        game_id = assignment.get("game_id")
        cluster_id = assignment.get("dependence_cluster_id")
        if not isinstance(game_id, str) or not game_id or not isinstance(cluster_id, str) or not cluster_id:
            raise TierListEvaluationError("dependence cluster proxy identity is malformed")
        if game_id in result:
            raise TierListEvaluationError(f"dependence cluster proxy repeats map: {game_id}")
        result[game_id] = cluster_id
    return result, _sha256_bytes(raw), payload.get("authoritative_series_identity") is True


def _load_maps(root: Path) -> tuple[list[MapRecord], dict[str, Any]]:
    source = root / SOURCE_LOCATOR
    if not source.is_file() or source.is_symlink():
        raise TierListEvaluationError("OE player-games source is missing")
    source_raw = source.read_bytes()
    clusters, cluster_raw_sha256, series_identity_authoritative = _load_cluster_proxy(root)
    try:
        frame = pd.read_parquet(source, columns=list(SOURCE_COLUMNS))
    except (OSError, KeyError, ValueError) as exc:
        raise TierListEvaluationError("OE player-games source cannot be read") from exc
    frame["role"] = frame["position"].map(SOURCE_ROLE_ALIASES)
    frame["side_norm"] = (
        frame["side"].astype(str).str.strip().str.lower().map({"blue": "blue", "red": "red", "1": "blue", "2": "red"})
    )
    frame["event_norm"] = frame["event_kind"].fillna("").astype(str).str.strip().str.lower()
    frame = frame[frame["role"].notna() & frame["side_norm"].notna()].copy()

    maps: list[MapRecord] = []
    excluded = defaultdict(int)
    for map_id, group in frame.groupby("gameid", sort=False):
        if len(group) != 10:
            excluded["map_row_count"] += 1
            continue
        scope = _scope_for_frame(group)
        if scope is None:
            excluded["scope_or_tier"] += 1
            continue
        if group["date"].nunique(dropna=False) != 1 or group["patch"].nunique(dropna=False) != 1:
            excluded["mixed_map_metadata"] += 1
            continue
        try:
            event_start = _naive_utc(group["date"].iloc[0])
            patch = _patch_token(group["patch"].iloc[0])
        except TierListEvaluationError:
            excluded["missing_time_or_patch"] += 1
            continue
        if set(group["side_norm"]) != {"blue", "red"}:
            excluded["side_identity"] += 1
            continue
        teams: dict[str, str] = {}
        results: dict[str, int] = {}
        picks: dict[str, dict[str, str]] = {"blue": {}, "red": {}}
        valid = True
        for side, side_group in group.groupby("side_norm"):
            if len(side_group) != 5 or side_group["team_key"].nunique(dropna=False) != 1 or side_group["result"].nunique(dropna=False) != 1:
                valid = False
                break
            team_key = str(side_group["team_key"].iloc[0]).strip()
            if not team_key:
                valid = False
                break
            teams[side] = team_key
            results[side] = int(side_group["result"].iloc[0])
            for role, role_group in side_group.groupby("role"):
                if len(role_group) != 1 or pd.isna(role_group["champion"].iloc[0]):
                    valid = False
                    break
                picks[side][str(role)] = str(role_group["champion"].iloc[0]).strip()
            if not valid:
                break
        if not valid or set(picks["blue"]) != set(ROLES) or set(picks["red"]) != set(ROLES):
            excluded["role_or_team_identity"] += 1
            continue
        if teams["blue"] == teams["red"] or set(results.values()) != {0, 1}:
            excluded["result_or_team_identity"] += 1
            continue
        maps.append(
            MapRecord(
                map_id=str(map_id),
                event_start=event_start,
                scope=scope,
                patch=patch,
                teams=teams,
                results=results,
                picks=picks,
                dependence_cluster_id=clusters.get(str(map_id)),
            )
        )
    maps.sort(key=lambda item: (item.event_start, item.map_id))
    return maps, {
        "source_locator": SOURCE_LOCATOR.as_posix(),
        "source_raw_sha256": _sha256_bytes(source_raw),
        "cluster_proxy_locator": CLUSTER_PROXY_LOCATOR.as_posix(),
        "cluster_proxy_raw_sha256": cluster_raw_sha256,
        "series_identity_authoritative": series_identity_authoritative,
        "maps_loaded": len(maps),
        "maps_without_cluster_proxy": sum(item.dependence_cluster_id is None for item in maps),
        "excluded_maps": dict(sorted(excluded.items())),
    }


def _tier_value(champion_logit: float, reference_logit: float, calibration_slope: float) -> float:
    return 100.0 * (
        float(_sigmoid(calibration_slope * champion_logit))
        - float(_sigmoid(calibration_slope * reference_logit))
    )


def build_evaluation_rows(root: Path | str = Path(".")) -> tuple[list[EvaluationRow], dict[str, Any]]:
    """Build rows using only state available before each map."""

    repo_root = Path(root)
    model = load_frozen_terminal_model(repo_root)
    maps, source = _load_maps(repo_root)
    coefficients = {str(key): float(value) for key, value in model.champion_role_logit.items()}
    members: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    team_strength: defaultdict[str, float] = defaultdict(float)
    rows: list[EvaluationRow] = []
    excluded = defaultdict(int)

    for game in maps:
        side_logits: dict[str, float] = {}
        for side in ("blue", "red"):
            values = [coefficients.get(f"{role}|{game.picks[side][role]}") for role in ROLES]
            if any(value is None or not math.isfinite(float(value)) for value in values):
                excluded["missing_terminal_draft_coverage"] += 1
                side_logits = {}
                break
            side_logits[side] = float(sum(float(value) for value in values))

        strength_logit = team_strength[game.teams["blue"]] - team_strength[game.teams["red"]]
        if side_logits:
            for side in ("blue", "red"):
                other_side = "red" if side == "blue" else "blue"
                sign = 1.0 if side == "blue" else -1.0
                for role in ROLES:
                    champion = game.picks[side][role]
                    key = f"{role}|{champion}"
                    prior = members[(game.scope, game.patch, role)]
                    known = sorted(champ for champ in prior if f"{role}|{champ}" in coefficients)
                    if key not in coefficients:
                        excluded["evaluated_champion_missing_coverage"] += 1
                        continue
                    if not known:
                        excluded["no_played_reference_membership"] += 1
                        continue
                    reference_logit = sum(coefficients[f"{role}|{champ}"] for champ in known) / len(known)
                    tier_value = _tier_value(float(coefficients[key]), reference_logit, float(model.calibration_slope))
                    draft_other = (side_logits[side] - coefficients[key]) - side_logits[other_side]
                    rows.append(
                        EvaluationRow(
                            row_id=f"{game.map_id}:{side}:{role}",
                            map_id=game.map_id,
                            event_start=game.event_start,
                            scope=game.scope,
                            patch=game.patch,
                            role=role,
                            champion=champion,
                            side=side,
                            label=int(game.results[side]),
                            strength_logit=float(strength_logit),
                            draft_other_logit=float(sign * draft_other),
                            tier_feature=float(sign * tier_value / 100.0),
                            tier_value_pp=float(tier_value),
                            played_champion_count=len(known),
                            dependence_cluster_id=game.dependence_cluster_id,
                        )
                    )

        expected_blue = float(_sigmoid(strength_logit))
        blue_result = float(game.results["blue"])
        team_strength[game.teams["blue"]] += TEAM_UPDATE_RATE * (blue_result - expected_blue)
        team_strength[game.teams["red"]] += TEAM_UPDATE_RATE * ((1.0 - blue_result) - (1.0 - expected_blue))
        for side in ("blue", "red"):
            for role in ROLES:
                members[(game.scope, game.patch, role)].add(game.picks[side][role])

    return rows, {
        "model_locator": TERMINAL_MODEL_ARTIFACT["locator"],
        "model_raw_sha256": TERMINAL_MODEL_ARTIFACT["raw_sha256"],
        "model_version": model.model_version,
        "source": source,
        "rows_built": len(rows),
        "maps_with_rows": len({row.map_id for row in rows}),
        "rows_without_authoritative_series": sum(row.dependence_cluster_id is None for row in rows),
        "excluded_rows": dict(sorted(excluded.items())),
    }


def _feature_frame(rows: Sequence[EvaluationRow], *, include_tier: bool) -> pd.DataFrame:
    values: list[dict[str, float]] = []
    columns = [
        "strength_logit",
        "draft_other_logit",
        *[f"scope_{scope}" for scope in SCOPE_COLUMNS],
    ]
    if include_tier:
        columns.append("tier_feature")
    for row in rows:
        item: dict[str, float] = {
            "strength_logit": row.strength_logit,
            "draft_other_logit": row.draft_other_logit,
        }
        for scope in SCOPE_COLUMNS:
            item[f"scope_{scope}"] = float(row.scope == scope)
        if include_tier:
            item["tier_feature"] = row.tier_feature
        values.append(item)
    return pd.DataFrame(values, columns=columns)


def _sample_weights(rows: Sequence[EvaluationRow]) -> np.ndarray:
    counts = pd.Series([row.map_id for row in rows]).value_counts()
    return np.asarray([1.0 / float(counts[row.map_id]) for row in rows], dtype=float)


def _fit_model(rows: Sequence[EvaluationRow], *, include_tier: bool) -> _FittedLogistic:
    if not rows or len({row.label for row in rows}) < 2:
        raise TierListEvaluationError("adapter fit needs both outcome classes")
    return _fit_logistic(
        _feature_frame(rows, include_tier=include_tier).to_numpy(dtype=float),
        np.asarray([row.label for row in rows], dtype=float),
        _sample_weights(rows),
    )


def _fit_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    regularization_c: float = 1.0,
) -> _FittedLogistic:
    features = np.asarray(features, dtype=float)
    labels = np.asarray(labels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if features.ndim != 2 or labels.ndim != 1 or len(features) != len(labels) or len(labels) != len(weights):
        raise TierListEvaluationError("logistic adapter arrays have incompatible shapes")
    if not np.isfinite(features).all() or not np.isfinite(labels).all() or not np.isfinite(weights).all():
        raise TierListEvaluationError("logistic adapter arrays contain non-finite values")
    if not np.isin(labels, (0.0, 1.0)).all() or (weights < 0.0).any():
        raise TierListEvaluationError("logistic adapter labels or weights are invalid")
    total_weight = float(weights.sum())
    if total_weight <= 0.0:
        raise TierListEvaluationError("logistic adapter weights have no mass")
    if regularization_c <= 0.0 or not math.isfinite(regularization_c):
        raise TierListEvaluationError("logistic adapter regularization is invalid")

    parameters = np.zeros(features.shape[1] + 1, dtype=float)
    augmented = np.column_stack((np.ones(len(features), dtype=float), features))
    if not np.isfinite(augmented).all():
        raise TierListEvaluationError("logistic adapter design is non-finite")
    lipschitz = 0.25 * float(np.max(np.einsum("ij,ij->i", augmented, augmented), initial=0.0))
    lipschitz += 1.0 / regularization_c
    if not math.isfinite(lipschitz) or lipschitz <= 0.0:
        raise TierListEvaluationError("logistic adapter smoothness bound is invalid")
    learning_rate = min(1.0, 0.5 / lipschitz)

    def objective(candidate: np.ndarray) -> float:
        logits = np.clip(
            candidate[0] + _linear_predictor(features, candidate[1:]),
            -40.0,
            40.0,
        )
        value = float(
            (
                weights * (np.logaddexp(0.0, logits) - labels * logits)
            ).sum()
            / total_weight
        )
        value += 0.5 * float(np.dot(candidate[1:], candidate[1:])) / regularization_c
        if not math.isfinite(value):
            raise TierListEvaluationError("logistic adapter objective is non-finite")
        return value

    current_objective = objective(parameters)
    converged = False
    for _ in range(5000):
        intercept = float(parameters[0])
        coefficients = parameters[1:]
        logits = np.clip(intercept + _linear_predictor(features, coefficients), -40.0, 40.0)
        probability = np.asarray(_sigmoid(logits), dtype=float)
        residual = weights * (probability - labels) / total_weight
        gradient = np.concatenate(
            (
                [float(residual.sum())],
                np.asarray(np.einsum("ij,i->j", features, residual, optimize=True), dtype=float),
            )
        )
        gradient[1:] += coefficients / regularization_c
        if not np.isfinite(gradient).all():
            raise TierListEvaluationError("logistic adapter derivative is non-finite")
        gradient_norm = float(np.max(np.abs(gradient), initial=0.0))
        if gradient_norm < 1e-9:
            converged = True
            break
        updated = np.clip(parameters - learning_rate * gradient, -20.0, 20.0)
        if not np.isfinite(updated).all():
            raise TierListEvaluationError("logistic adapter update is non-finite")
        updated_objective = objective(updated)
        if updated_objective > current_objective + 1e-12:
            raise TierListEvaluationError("logistic adapter objective increased")
        if abs(current_objective - updated_objective) < 1e-12 and gradient_norm < 1e-7:
            parameters = updated
            converged = True
            break
        parameters = updated
        current_objective = updated_objective
    if not converged:
        raise TierListEvaluationError("logistic adapter did not converge")
    if not np.isfinite(parameters).all():
        raise TierListEvaluationError("logistic adapter fit produced non-finite parameters")
    return _FittedLogistic(coefficients=parameters[1:].copy(), intercept=float(parameters[0]))


def _fit_calibrator(rows: Sequence[EvaluationRow], raw_probability: np.ndarray) -> tuple[float, float]:
    if not rows or len({row.label for row in rows}) < 2:
        raise TierListEvaluationError("calibration fit needs both outcome classes")
    calibrator = _fit_logistic(
        _logit(raw_probability).reshape(-1, 1),
        np.asarray([row.label for row in rows], dtype=float),
        _sample_weights(rows),
    )
    return calibrator.intercept, float(calibrator.coefficients[0])


def _calibrated_prediction(model: _FittedLogistic, rows: Sequence[EvaluationRow], *, include_tier: bool, calibration: tuple[float, float]) -> np.ndarray:
    raw = model.predict_proba(_feature_frame(rows, include_tier=include_tier))[:, 1]
    intercept, slope = calibration
    return np.asarray(_sigmoid(intercept + slope * _logit(raw)), dtype=float)


def _score(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    clipped = np.clip(probability, 1e-9, 1.0 - 1e-9)
    return {
        "log_loss": float(log_loss(y, clipped, labels=[0, 1])),
        "brier": float(brier_score_loss(y, clipped)),
    }


def _bootstrap_deltas(
    rows: Sequence[EvaluationRow],
    baseline_probability: np.ndarray,
    candidate_probability: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    clusters = sorted({row.dependence_cluster_id for row in rows if row.dependence_cluster_id})
    if len(clusters) < 2:
        raise TierListEvaluationError("bootstrap needs at least two dependence clusters")
    by_cluster: dict[str, np.ndarray] = {}
    for cluster in clusters:
        by_cluster[cluster] = np.asarray([index for index, row in enumerate(rows) if row.dependence_cluster_id == cluster], dtype=int)
    rng = np.random.default_rng(seed)
    deltas = {"log_loss": [], "brier": []}
    y = np.asarray([row.label for row in rows], dtype=int)
    for _ in range(replicates):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        indexes = np.concatenate([by_cluster[cluster] for cluster in sampled])
        base = _score(y[indexes], baseline_probability[indexes])
        candidate = _score(y[indexes], candidate_probability[indexes])
        for metric in deltas:
            deltas[metric].append(candidate[metric] - base[metric])
    return {
        metric: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for metric, values in deltas.items()
    }


def evaluate(
    root: Path | str = Path("."),
    *,
    cutoff: str | None = None,
    bootstrap_replicates: int = 2000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a development evaluation report with production claims closed."""

    rows, inventory = build_evaluation_rows(root)
    if not rows:
        raise TierListEvaluationError("no time-safe tier-list evaluation rows were built")
    model_cutoff = cutoff or TERMINAL_MODEL_ARTIFACT["model_as_of"].replace("+00:00", "Z")
    cutoff_time = _parse_timestamp(model_cutoff)
    pre = [row for row in rows if row.event_start < cutoff_time]
    future = [row for row in rows if row.event_start >= cutoff_time]
    pre = [row for row in pre if row.dependence_cluster_id is not None]
    future = [row for row in future if row.dependence_cluster_id is not None]
    if len(pre) < 100 or len(future) < 20:
        raise TierListEvaluationError("time-safe development evaluation has insufficient pre/future rows")
    clusters = sorted({row.dependence_cluster_id for row in pre if row.dependence_cluster_id})
    if len(clusters) < 10:
        raise TierListEvaluationError("time-safe development evaluation has insufficient clusters")
    calibration_count = max(2, int(len(clusters) * 0.20))
    calibration_clusters = set(clusters[-calibration_count:])
    fit_rows = [row for row in pre if row.dependence_cluster_id not in calibration_clusters]
    calibration_rows = [row for row in pre if row.dependence_cluster_id in calibration_clusters]
    baseline_model = _fit_model(fit_rows, include_tier=False)
    candidate_model = _fit_model(fit_rows, include_tier=True)
    baseline_calibration = _fit_calibrator(
        calibration_rows,
        baseline_model.predict_proba(_feature_frame(calibration_rows, include_tier=False))[:, 1],
    )
    candidate_calibration = _fit_calibrator(
        calibration_rows,
        candidate_model.predict_proba(_feature_frame(calibration_rows, include_tier=True))[:, 1],
    )
    baseline_probability = _calibrated_prediction(
        baseline_model, future, include_tier=False, calibration=baseline_calibration
    )
    candidate_probability = _calibrated_prediction(
        candidate_model, future, include_tier=True, calibration=candidate_calibration
    )
    labels = np.asarray([row.label for row in future], dtype=int)
    baseline_scores = _score(labels, baseline_probability)
    candidate_scores = _score(labels, candidate_probability)
    delta = {metric: candidate_scores[metric] - baseline_scores[metric] for metric in baseline_scores}
    intervals = _bootstrap_deltas(
        future,
        baseline_probability,
        candidate_probability,
        replicates=bootstrap_replicates,
        seed=seed,
    )
    proper_score_passed = bool(
        delta["log_loss"] <= 0.0
        and delta["brier"] <= 0.0
        and (
            intervals["log_loss"]["upper_95"] <= 0.0
            or intervals["brier"]["upper_95"] <= 0.0
        )
    )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only_evaluation_complete",
        "decision": "development_only",
        "production_eligible": False,
        "prospective": True,
        "synthetic_only": False,
        "future_observed_outcomes": True,
        "future_prediction_capture_present": False,
        "nested_adapter": "time_safe_tv_substitution",
        "proper_score_passed": proper_score_passed,
        "calibration_passed": False,
        "roster_strength_time_safe": True,
        "current_patch_verified": False,
        "counterability_policy_validated": False,
        "counterability_weight_manifested": True,
        "counterability_weight": 0.0,
        "series_identity_authoritative": bool(inventory["source"]["series_identity_authoritative"]),
        "dependence_cluster_proxy_used": True,
        "cutoff_utc": cutoff_time.isoformat().replace("+00:00", "Z"),
        "model": {
            "locator": TERMINAL_MODEL_ARTIFACT["locator"],
            "raw_sha256": TERMINAL_MODEL_ARTIFACT["raw_sha256"],
            "model_version": TERMINAL_MODEL_ARTIFACT["model_version"],
            "candidate_id": TERMINAL_MODEL_ARTIFACT["candidate_id"],
        },
        "inventory": inventory,
        "sample": {
            "fit_rows": len(fit_rows),
            "calibration_rows": len(calibration_rows),
            "future_rows": len(future),
            "fit_clusters": len({row.dependence_cluster_id for row in fit_rows}),
            "calibration_clusters": len(calibration_clusters),
            "future_clusters": len({row.dependence_cluster_id for row in future}),
        },
        "adapter": {
            "baseline_features": ["time_safe_team_strength", "other_terminal_draft_logit"],
            "candidate_feature": "pre_event_tier_value_pp_signed_over_100",
            "team_update_rate": TEAM_UPDATE_RATE,
            "fit_regularization_c": 1.0,
            "calibration_source": "pre_cutoff_calibration_clusters",
            "baseline_coefficients": baseline_model.coef_[0].tolist(),
            "candidate_coefficients": candidate_model.coef_[0].tolist(),
            "baseline_intercept": float(baseline_model.intercept_[0]),
            "candidate_intercept": float(candidate_model.intercept_[0]),
            "baseline_calibration": {
                "intercept": baseline_calibration[0],
                "slope": baseline_calibration[1],
            },
            "candidate_calibration": {
                "intercept": candidate_calibration[0],
                "slope": candidate_calibration[1],
            },
        },
        "proper_scores": {
            "baseline": baseline_scores,
            "candidate": candidate_scores,
            "candidate_minus_baseline": delta,
            "series_cluster_bootstrap_95": intervals,
            "bootstrap_replicates": bootstrap_replicates,
            "bootstrap_seed": seed,
        },
        "counterability": {
            "status": "descriptive_only_no_l2_validation",
            "lambda_c": 0.0,
            "out_of_sample_weight_decision": "pending_independent_l2",
        },
        "claim_ceiling": {
            "descriptive_pre_map_association": False,
            "model_standardized_development_diagnostic": True,
            "rank_eligibility": False,
            "publication": False,
            "outcome_calibrated_probability": False,
            "causal_draft_effect": False,
            "recommendation": False,
            "betting": False,
        },
    }
    report["artifact_sha256"] = _canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cutoff", default=None, help="RFC-3339 future-outcome cutoff; defaults to the frozen model as_of")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true", help="return 1 because the diagnostic is not production-authorized")
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        raise SystemExit("--bootstrap-replicates must be at least 100")
    report = evaluate(args.root, cutoff=args.cutoff, bootstrap_replicates=args.bootstrap_replicates)
    raw = (_canonical_json(report) + "\n").encode("utf-8")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(raw)
        print(f"wrote {args.output}")
    else:
        print(raw.decode("utf-8"), end="")
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
