"""Bounded chronological development evaluation for the L7 terminal estimator.

This harness is deliberately not a promotion runner. It uses the checked-in
Oracle's Elixir player-game parquet for development diagnostics only, fits
every candidate inside each expanding series fold, calibrates only on that
fold's calibration slice, and records missing source/holdout authority instead
of manufacturing it. Player identity is read only to identify dependence
clusters; it is never a feature of the neutral estimator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

from lol_kills.v2.data.common import ROLES, ROLE_ALIASES


SCHEMA_VERSION = "draft-terminal-development-evaluation-v1"
CANDIDATE_ORDER = (
    "m0-role-additive",
    "m1-role-additive-allied-synergy",
    "m2-role-additive-allied-and-counter",
)
CALIBRATION_ORDER = ("identity", "symmetric_temperature", "symmetrized_platt")
RIDGE_LAMBDA = 1.0
# Sparse champion terms are excluded until they have ten independent map
# appearances in the fitting slice; the sparse/new-champion holdout remains a
# separate diagnostic and is never silently turned into a neutral zero claim.
MIN_FEATURE_SUPPORT = 10
EPSILON = 1e-12
BASELINE_K_FACTOR = 20.0
BASELINE_INITIAL_RATING = 1500.0
BASELINE_RATING_SCALE = 400.0
BASELINE_CONFIG = {
    "kind": "pre_event_team_elo_logit",
    "team_identifier": "teamid",
    "initial_rating": BASELINE_INITIAL_RATING,
    "k_factor": BASELINE_K_FACTOR,
    "expected_score": "elo_base10_scale_400",
    "update_unit": "dependence_cluster",
    "current_cluster_outcomes_update_after_cluster": True,
    "served_baseline_logit": 0.0,
    "development_only": True,
}


@dataclass(frozen=True)
class DraftRow:
    game_id: str
    dependence_cluster_id: str
    date: datetime
    patch: str
    league: str
    team_a: str
    team_b: str
    side_a: tuple[tuple[str, str], ...]
    side_b: tuple[tuple[str, str], ...]
    label_a: int


@dataclass(frozen=True)
class Fold:
    fold_id: str
    train: tuple[int, int]
    validation: tuple[int, int]
    calibration: tuple[int, int]
    test: tuple[int, int]


@dataclass(frozen=True)
class BaselineAdjustedFit:
    candidate_id: str
    vocabulary: tuple[str, ...]
    beta: np.ndarray
    baseline_coefficient: float


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _parse_time(value: Any) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _role(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return ROLE_ALIASES.get(value.strip().lower())


def _identifier(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def _normalize_patch(value: Any) -> str:
    token = _identifier(value)
    if not token:
        return ""
    try:
        number = float(token)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number) or number <= 0:
        return ""
    centesimal = round(number * 100)
    if abs(number * 100 - centesimal) > 1e-8:
        return ""
    return f"{centesimal / 100:.2f}"


def _load_dependence_clusters(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Load the outcome-free development cluster proxy without treating it as authority."""

    path = root / "data/lol/v2/models/draft-interactions/series-cluster-proxy.json"
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("dependence cluster proxy must be an object")
    if payload.get("schema_id") != "scryglass.draft-interaction-dependence-cluster-proxy.v1":
        raise ValueError("unexpected dependence cluster proxy schema")
    if payload.get("source_mode") != "pinned_development_source":
        raise ValueError("dependence cluster proxy source mode is not pinned development data")
    for field in ("development_only", "outcome_free"):
        if payload.get(field) is not True:
            raise ValueError(f"dependence cluster proxy must declare {field}=true")
    for field in ("predictive_authority", "authoritative_series_identity", "authorizes_model_selection", "authorizes_publication"):
        if payload.get(field) is not False:
            raise ValueError(f"dependence cluster proxy must declare {field}=false")
    claimed_artifact_hash = payload.get("artifact_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    calculated_artifact_hash = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if claimed_artifact_hash != calculated_artifact_hash:
        raise ValueError("dependence cluster proxy artifact hash does not match its payload")
    assignments = payload.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("dependence cluster proxy assignments must be a list")
    mapping: dict[str, str] = {}
    for assignment in assignments:
        if not isinstance(assignment, Mapping):
            raise ValueError("dependence cluster assignment must be an object")
        game_id = _identifier(assignment.get("game_id"))
        cluster_id = _identifier(assignment.get("dependence_cluster_id"))
        if not game_id or not cluster_id or game_id in mapping:
            raise ValueError("dependence cluster proxy has an invalid or duplicate assignment")
        mapping[game_id] = cluster_id
    return mapping, {
        "dependence_cluster_proxy_raw_sha256": _sha256_bytes(raw),
        "dependence_cluster_proxy_artifact_sha256": str(claimed_artifact_hash),
    }


def load_snapshot(root: Path) -> tuple[list[DraftRow], dict[str, str]]:
    """Load complete neutral rows from the checked-in OE player-game snapshot."""

    parquet_path = root / "data/lol/warehouse/parquet/players.parquet"
    metadata_path = root / "data/lol/warehouse/parquet/oe_meta.json"
    dependence_clusters, dependence_hashes = _load_dependence_clusters(root)
    raw = parquet_path.read_bytes()
    metadata_raw = metadata_path.read_bytes() if metadata_path.exists() else b""
    frame = pd.read_parquet(
        parquet_path,
        columns=[
            "game_uid",
            "date",
            "patch",
            "league",
            "side",
            "position",
            "champion",
            "result",
            "teamid",
            "teamname",
        ],
    )

    rows: list[DraftRow] = []
    for raw_game_id, group in frame.groupby("game_uid", sort=False):
        game_id = _identifier(raw_game_id)
        if not game_id or len(group) != 10:
            continue
        patch_values = {_normalize_patch(value) for value in group["patch"].tolist() if _normalize_patch(value)}
        league_values = {_identifier(value).upper() for value in group["league"].tolist() if _identifier(value)}
        dates = {_parse_time(value) for value in group["date"].tolist() if _identifier(value)}
        if len(patch_values) != 1 or len(league_values) != 1 or len(dates) != 1:
            continue
        by_side: dict[str, dict[str, tuple[str, int]]] = {"Blue": {}, "Red": {}}
        team_keys: dict[str, str] = {}
        malformed = False
        for record in group.to_dict("records"):
            side = _identifier(record.get("side"))
            role = _role(record.get("position"))
            champion = _identifier(record.get("champion"))
            team_key = _identifier(record.get("teamid")) or _identifier(record.get("teamname"))
            result = record.get("result")
            if side not in by_side or role not in ROLES or not champion or not team_key:
                malformed = True
                break
            try:
                label = int(result)
            except (TypeError, ValueError):
                malformed = True
                break
            if label not in {0, 1} or role in by_side[side]:
                malformed = True
                break
            by_side[side][role] = (champion, label)
            team_keys[side] = team_key
        if malformed or any(set(by_side[side]) != set(ROLES) for side in ("Blue", "Red")):
            continue
        if team_keys.get("Blue") == team_keys.get("Red"):
            continue
        blue_champions = [by_side["Blue"][role][0] for role in ROLES]
        red_champions = [by_side["Red"][role][0] for role in ROLES]
        if len(set((*blue_champions, *red_champions))) != 10:
            continue
        canonical = sorted(((team_keys["Blue"], "Blue"), (team_keys["Red"], "Red")))
        team_a, side_a_name = canonical[0]
        team_b, side_b_name = canonical[1]
        composition_a = tuple((role, by_side[side_a_name][role][0]) for role in ROLES)
        composition_b = tuple((role, by_side[side_b_name][role][0]) for role in ROLES)
        rows.append(
            DraftRow(
                game_id=game_id,
                dependence_cluster_id=dependence_clusters.get(game_id, f"unclustered-game:{game_id}"),
                date=next(iter(dates)),
                patch=next(iter(patch_values)),
                league=next(iter(league_values)),
                team_a=team_a,
                team_b=team_b,
                side_a=composition_a,
                side_b=composition_b,
                label_a=by_side[side_a_name][ROLES[0]][1],
            )
        )
    rows.sort(key=lambda row: (row.date, row.dependence_cluster_id, row.game_id))
    return rows, {
        "oe_players_snapshot_sha256": _sha256_bytes(raw),
        "oe_metadata_snapshot_sha256": _sha256_bytes(metadata_raw) if metadata_raw else "",
        **dependence_hashes,
    }


def chronological_folds(series_count: int) -> tuple[Fold, ...]:
    if series_count < 120:
        raise ValueError("at least 120 complete series are required for the bounded development folds")
    # These boundaries are frozen as a percentage-independent series policy:
    # expanding train, then validation, calibration, and untouched test blocks.
    block = max(20, series_count // 10)
    return (
        Fold("outer-00", (0, 4 * block), (4 * block, 5 * block), (5 * block, 6 * block), (6 * block, 7 * block)),
        Fold("outer-01", (0, 5 * block), (5 * block, 6 * block), (6 * block, 7 * block), (7 * block, 8 * block)),
        Fold("outer-02", (0, 6 * block), (6 * block, 7 * block), (7 * block, 8 * block), (8 * block, series_count)),
    )


def _pairs(side: Sequence[tuple[str, str]]) -> Iterable[tuple[str, str]]:
    for index, (_, first) in enumerate(side):
        for _, second in side[index + 1 :]:
            yield tuple(sorted((first, second)))


def feature_map(row: DraftRow, candidate_id: str) -> dict[str, float]:
    values: dict[str, float] = defaultdict(float)
    for role, champion in row.side_a:
        values[f"main|{role}|{champion}"] += 1.0
    for role, champion in row.side_b:
        values[f"main|{role}|{champion}"] -= 1.0
    if candidate_id in {"m1-role-additive-allied-synergy", "m2-role-additive-allied-and-counter"}:
        for first, second in _pairs(row.side_a):
            values[f"ally|{first}|{second}"] += 1.0
        for first, second in _pairs(row.side_b):
            values[f"ally|{first}|{second}"] -= 1.0
    if candidate_id == "m2-role-additive-allied-and-counter":
        for (role_a, first), (role_b, second) in zip(row.side_a, row.side_b):
            if role_a != role_b:
                raise ValueError("row roles are not aligned")
            first_key, second_key = sorted((first, second))
            values[f"counter|{role_a}|{first_key}|{second_key}"] += 1.0 if first <= second else -1.0
    return dict(values)


def _design(rows: Sequence[DraftRow], candidate_id: str, vocabulary: Sequence[str]) -> csr_matrix:
    index = {name: position for position, name in enumerate(vocabulary)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for row_index, row in enumerate(rows):
        for name, value in feature_map(row, candidate_id).items():
            column_index = index.get(name)
            if column_index is not None and value:
                row_indices.append(row_index)
                column_indices.append(column_index)
                values.append(float(value))
    return csr_matrix((values, (row_indices, column_indices)), shape=(len(rows), len(vocabulary)), dtype=float)


def _fit_sparse_logistic(X: csr_matrix, labels: Sequence[int], label: str) -> np.ndarray:
    if X.shape[1] == 0:
        return np.zeros(0, dtype=float)
    model = LogisticRegression(
        penalty="l2",
        C=float(len(labels)) / RIDGE_LAMBDA,
        fit_intercept=False,
        solver="liblinear",
        max_iter=1000,
        tol=1e-8,
        random_state=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            model.fit(X, np.asarray(labels, dtype=float))
        except ConvergenceWarning as exc:
            raise FloatingPointError(f"development fit did not converge for {label}") from exc
    beta = np.asarray(model.coef_[0], dtype=float)
    if not np.all(np.isfinite(beta)):
        raise FloatingPointError(f"non-finite development fit state for {label}")
    return beta


def feature_vocabulary(rows: Sequence[DraftRow], candidate_id: str) -> tuple[str, ...]:
    maps = [feature_map(row, candidate_id) for row in rows]
    support: dict[str, int] = defaultdict(int)
    for features in maps:
        for name, value in features.items():
            if value:
                support[name] += 1
    return tuple(sorted(name for name, count in support.items() if count >= MIN_FEATURE_SUPPORT))


def fit_logistic(rows: Sequence[DraftRow], candidate_id: str) -> tuple[tuple[str, ...], np.ndarray]:
    vocabulary = feature_vocabulary(rows, candidate_id)
    X = _design(rows, candidate_id, vocabulary)
    if not vocabulary:
        return vocabulary, np.zeros(0, dtype=float)
    return vocabulary, _fit_sparse_logistic(X, [row.label_a for row in rows], candidate_id)


def pre_event_team_elo_logits(
    rows: Sequence[DraftRow],
    *,
    freeze_at: datetime | None = None,
) -> dict[str, float]:
    """Return deterministic pre-event team-strength logits for each map.

    Ratings are updated only after every map in a dependence cluster. The
    current map and the other maps in its cluster therefore cannot influence
    their own baseline. When ``freeze_at`` is supplied, any cluster touching
    or after that instant is scored against the state at the test boundary and
    never updates it; this keeps a chronological outer-test evaluation from
    learning from outer-test results, including a cluster that crosses the
    boundary. This is a development nuisance
    baseline only; it is never serialized into the neutral served score.
    """

    grouped: dict[str, list[DraftRow]] = defaultdict(list)
    for row in rows:
        grouped[row.dependence_cluster_id].append(row)
    ordered_clusters = sorted(
        grouped.values(),
        key=lambda cluster: (
            min(row.date for row in cluster),
            min(row.dependence_cluster_id for row in cluster),
        ),
    )
    ratings: dict[str, float] = defaultdict(lambda: BASELINE_INITIAL_RATING)
    logits_by_game: dict[str, float] = {}
    for cluster in ordered_clusters:
        updates: dict[str, float] = defaultdict(float)
        cluster_is_entirely_pre_event = freeze_at is None or max(item.date for item in cluster) < freeze_at
        for row in sorted(cluster, key=lambda item: item.game_id):
            difference = ratings[row.team_a] - ratings[row.team_b]
            pre_event_logit = difference * math.log(10.0) / BASELINE_RATING_SCALE
            logits_by_game[row.game_id] = float(pre_event_logit)
            if cluster_is_entirely_pre_event:
                expected = 1.0 / (1.0 + 10.0 ** (-difference / BASELINE_RATING_SCALE))
                delta = BASELINE_K_FACTOR * (float(row.label_a) - expected)
                updates[row.team_a] += delta
                updates[row.team_b] -= delta
        if cluster_is_entirely_pre_event:
            for team, delta in updates.items():
                ratings[team] += delta
    if set(logits_by_game) != {row.game_id for row in rows}:
        raise ValueError("pre-event baseline did not produce one value per development map")
    return logits_by_game


def fit_baseline_adjusted(
    rows: Sequence[DraftRow],
    candidate_id: str,
    baseline_logits: Mapping[str, float],
) -> BaselineAdjustedFit:
    """Fit composition terms while explicitly adjusting for pre-event strength."""

    vocabulary = feature_vocabulary(rows, candidate_id)
    composition = _design(rows, candidate_id, vocabulary)
    nuisance = np.asarray([float(baseline_logits[row.game_id]) for row in rows], dtype=float)
    if not np.all(np.isfinite(nuisance)):
        raise FloatingPointError("non-finite pre-event baseline logits")
    full = hstack([composition, csr_matrix(nuisance.reshape(-1, 1))], format="csr")
    coefficients = _fit_sparse_logistic(
        full,
        [row.label_a for row in rows],
        f"{candidate_id}:baseline-adjusted",
    )
    return BaselineAdjustedFit(
        candidate_id=candidate_id,
        vocabulary=vocabulary,
        beta=coefficients[:-1],
        baseline_coefficient=float(coefficients[-1]),
    )


def composition_logits(rows: Sequence[DraftRow], fit: BaselineAdjustedFit) -> np.ndarray:
    result = np.asarray(_design(rows, fit.candidate_id, fit.vocabulary) @ fit.beta, dtype=float)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite equalized composition logits")
    return result


def baseline_adjusted_logits(
    rows: Sequence[DraftRow],
    fit: BaselineAdjustedFit,
    baseline_logits: Mapping[str, float],
) -> np.ndarray:
    nuisance = np.asarray([float(baseline_logits[row.game_id]) for row in rows], dtype=float)
    result = composition_logits(rows, fit) + fit.baseline_coefficient * nuisance
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite baseline-adjusted logits")
    return result


def fit_baseline_only(rows: Sequence[DraftRow], baseline_logits: Mapping[str, float]) -> float:
    nuisance = np.asarray([float(baseline_logits[row.game_id]) for row in rows], dtype=float)
    coefficient = _fit_sparse_logistic(
        csr_matrix(nuisance.reshape(-1, 1)),
        [row.label_a for row in rows],
        "pre-event-team-elo-baseline",
    )
    return float(coefficient[0])


def logits(rows: Sequence[DraftRow], candidate_id: str, vocabulary: Sequence[str], beta: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = _design(rows, candidate_id, vocabulary) @ beta
    if not np.all(np.isfinite(result)):
        raise FloatingPointError(f"non-finite development prediction state for {candidate_id}")
    return result


def _probabilities(raw_logits: np.ndarray, scale: float) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = np.clip(1.0 / (1.0 + np.exp(-np.clip(raw_logits * scale, -40, 40))), EPSILON, 1.0 - EPSILON)
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("non-finite calibrated development probabilities")
    return result


def _log_loss(labels: Sequence[int], probabilities: np.ndarray) -> float:
    y = np.asarray(labels, dtype=float)
    return float(np.mean(-(y * np.log(probabilities) + (1.0 - y) * np.log1p(-probabilities))))


def _brier(labels: Sequence[int], probabilities: np.ndarray) -> float:
    return float(np.mean((np.asarray(labels, dtype=float) - probabilities) ** 2))


def _fit_calibration(raw_logits: np.ndarray, labels: Sequence[int], method: str) -> tuple[float, float]:
    if method == "identity":
        return 1.0, _log_loss(labels, _probabilities(raw_logits, 1.0))
    if method == "symmetric_temperature":
        candidates = np.linspace(0.5, 2.0, 31)
        scored = [(float(temp), _log_loss(labels, _probabilities(raw_logits, 1.0 / float(temp)))) for temp in candidates]
        return min(scored, key=lambda item: (item[1], item[0]))
    if method == "symmetrized_platt":
        candidates = np.linspace(0.25, 4.0, 76)
        scored = [(float(scale), _log_loss(labels, _probabilities(raw_logits, float(scale)))) for scale in candidates]
        return min(scored, key=lambda item: (item[1], item[0]))
    raise ValueError(f"unknown calibration method: {method}")


def _cluster_metrics(rows: Sequence[DraftRow], probabilities: np.ndarray) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row, probability in zip(rows, probabilities):
        grouped[row.dependence_cluster_id].append((row.label_a, float(probability)))
    losses: list[float] = []
    briers: list[float] = []
    for values in grouped.values():
        labels = [label for label, _ in values]
        probs = np.asarray([probability for _, probability in values])
        losses.append(_log_loss(labels, probs))
        briers.append(_brier(labels, probs))
    return {
        "row_count": len(rows),
        "dependence_cluster_count": len(grouped),
        "log_loss": float(np.mean(losses)) if losses else None,
        "brier_score": float(np.mean(briers)) if briers else None,
    }


def _league_metrics(rows: Sequence[DraftRow], probabilities: np.ndarray) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for league in sorted({row.league for row in rows}):
        indices = [index for index, row in enumerate(rows) if row.league == league]
        subset = [rows[index] for index in indices]
        output[league] = _cluster_metrics(subset, probabilities[indices])
    return output


def _fold_rows(rows: Sequence[DraftRow], series_order: Sequence[str], span: tuple[int, int]) -> list[DraftRow]:
    selected = set(series_order[span[0] : span[1]])
    return [row for row in rows if row.dependence_cluster_id in selected]


def _patch_sort_key(patch: str) -> tuple[int, int, str]:
    match = re.match(r"^(\d+)\.(\d+)$", patch)
    if not match:
        return (-1, -1, patch)
    return (int(match.group(1)), int(match.group(2)), patch)


def evaluate(root: Path) -> dict[str, Any]:
    rows, source_hashes = load_snapshot(root)
    series_latest: dict[str, datetime] = {}
    for row in rows:
        series_latest[row.dependence_cluster_id] = max(series_latest.get(row.dependence_cluster_id, row.date), row.date)
    series_order = [series_id for series_id, _ in sorted(series_latest.items(), key=lambda item: (item[1], item[0]))]
    patches = sorted({row.patch for row in rows}, key=_patch_sort_key)
    latest_patch = patches[-1] if patches else None
    international_leagues = {"MSI", "EWC", "WORLDS", "WORLD CHAMPIONSHIP"}
    international_rows = [row for row in rows if row.league in international_leagues]
    unclustered_rows = [row for row in rows if row.dependence_cluster_id.startswith("unclustered-game:")]
    proxy_clustered_rows = len(rows) - len(unclustered_rows)
    baseline_config_sha256 = _sha256_bytes(_canonical_json(BASELINE_CONFIG))
    folds = chronological_folds(len(series_order))
    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        train = _fold_rows(rows, series_order, fold.train)
        validation = _fold_rows(rows, series_order, fold.validation)
        calibration = _fold_rows(rows, series_order, fold.calibration)
        test = _fold_rows(rows, series_order, fold.test)
        test_start = min((row.date for row in test), default=None)
        fold_baseline_logits = pre_event_team_elo_logits(rows, freeze_at=test_start)
        candidate_reports: list[dict[str, Any]] = []
        fitted: dict[str, BaselineAdjustedFit] = {}
        baseline_only_coefficient = fit_baseline_only(train, fold_baseline_logits)
        validation_baseline_logits = np.asarray(
            [fold_baseline_logits[row.game_id] for row in validation], dtype=float
        )
        baseline_validation_probabilities = _probabilities(
            baseline_only_coefficient * validation_baseline_logits,
            1.0,
        )
        for candidate_id in CANDIDATE_ORDER:
            fit = fit_baseline_adjusted(train, candidate_id, fold_baseline_logits)
            fitted[candidate_id] = fit
            validation_adjusted = baseline_adjusted_logits(validation, fit, fold_baseline_logits)
            validation_equalized = composition_logits(validation, fit)
            candidate_reports.append(
                {
                    "candidate_id": candidate_id,
                    "feature_count": len(fit.vocabulary),
                    "baseline_feature_count": 1,
                    "baseline_feature": "pre_event_team_elo_logit",
                    "baseline_coefficient": fit.baseline_coefficient,
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "calibration_rows": len(calibration),
                    "test_rows": len(test),
                    "validation": _cluster_metrics(
                        validation,
                        _probabilities(validation_equalized, 1.0),
                    ),
                    "validation_equalized_draft": _cluster_metrics(
                        validation,
                        _probabilities(validation_equalized, 1.0),
                    ),
                    "validation_baseline_adjusted": _cluster_metrics(
                        validation,
                        _probabilities(validation_adjusted, 1.0),
                    ),
                    "validation_baseline_only": _cluster_metrics(
                        validation,
                        baseline_validation_probabilities,
                    ),
                }
            )
        selected_candidate = min(
            candidate_reports,
            key=lambda report: (
                float(report["validation_equalized_draft"]["log_loss"]),
                float(report["validation_equalized_draft"]["brier_score"]),
                CANDIDATE_ORDER.index(report["candidate_id"]),
            ),
        )
        selected_candidate_id = str(selected_candidate["candidate_id"])
        selected_fit = fitted[selected_candidate_id]
        selected_calibration_logits = composition_logits(calibration, selected_fit)
        transform_choices = []
        transform_reports: list[dict[str, Any]] = []
        for method in CALIBRATION_ORDER:
            scale, calibration_loss = _fit_calibration(
                selected_calibration_logits,
                [row.label_a for row in calibration],
                method,
            )
            choice = {
                "method": method,
                "parameter": scale,
                "calibration_log_loss": calibration_loss,
            }
            transform_choices.append((float(calibration_loss), CALIBRATION_ORDER.index(method), method, scale))
            transform_reports.append(choice)
        _, _, selected_transform, selected_scale = min(transform_choices)
        for report in transform_reports:
            report["selected"] = report["method"] == selected_transform
        calibration_scale = (
            1.0 / selected_scale if selected_transform == "symmetric_temperature" else selected_scale
        )
        selected_test_equalized_logits = composition_logits(test, selected_fit)
        selected_test_adjusted_logits = baseline_adjusted_logits(test, selected_fit, fold_baseline_logits)
        test_baseline_logits = np.asarray([fold_baseline_logits[row.game_id] for row in test], dtype=float)
        selected_test_probabilities = _probabilities(selected_test_equalized_logits, calibration_scale)
        selected_test_adjusted_probabilities = _probabilities(selected_test_adjusted_logits, calibration_scale)
        baseline_test_probabilities = _probabilities(
            baseline_only_coefficient * test_baseline_logits,
            1.0,
        )
        selected_candidate["selected_for_outer_test"] = True
        selected_candidate["calibration_transforms"] = transform_reports
        selected_candidate["locked_outer_test"] = _cluster_metrics(test, selected_test_probabilities)
        selected_candidate["locked_outer_test_by_league"] = _league_metrics(test, selected_test_probabilities)
        selected_candidate["baseline_locked_outer_test"] = _cluster_metrics(test, baseline_test_probabilities)
        selected_candidate["baseline_adjusted_locked_outer_test"] = _cluster_metrics(
            test,
            selected_test_adjusted_probabilities,
        )
        selected_candidate["equalized_draft_locked_outer_test"] = selected_candidate["locked_outer_test"]
        selected_candidate["baseline_strength_separation"] = {
            "baseline_config_sha256": baseline_config_sha256,
            "baseline_coefficient": selected_fit.baseline_coefficient,
            "baseline_logit_sd_outer_test": float(np.std(test_baseline_logits)),
            "baseline_nonzero_outer_test_rows": int(np.count_nonzero(test_baseline_logits)),
            "baseline_only": selected_candidate["baseline_locked_outer_test"],
            "baseline_adjusted": selected_candidate["baseline_adjusted_locked_outer_test"],
            "equalized_draft": selected_candidate["locked_outer_test"],
            "served_baseline_logit": BASELINE_CONFIG["served_baseline_logit"],
            "baseline_is_not_serialized_in_served_artifact": True,
        }
        for report in candidate_reports:
            report.setdefault("selected_for_outer_test", False)
        fold_reports.append(
            {
                "fold_id": fold.fold_id,
                "series_spans": {"train": fold.train, "validation": fold.validation, "calibration": fold.calibration, "test": fold.test},
                "dependence_cluster_spans": {"train": fold.train, "validation": fold.validation, "calibration": fold.calibration, "test": fold.test},
                "date_ranges": {
                    name: {
                        "start": min((_parse_time(row.date) for row in subset), default=None).isoformat() if subset else None,
                        "end": max((_parse_time(row.date) for row in subset), default=None).isoformat() if subset else None,
                    }
                    for name, subset in (("train", train), ("validation", validation), ("calibration", calibration), ("test", test))
                },
                "cluster_units": ["development_dependence_cluster_id", "team_id_or_team_name"],
                "selection": {
                    "candidate_id": selected_candidate_id,
                    "criterion": "equalized_draft_validation_log_loss_then_brier_then_preregistered_order",
                    "validation": selected_candidate["validation"],
                    "calibration_transform": selected_transform,
                    "outer_test_locked": True,
                },
                "baseline_state_policy": {
                    "outer_test_frozen_at": test_start.isoformat() if test_start else None,
                    "outer_test_baseline_frozen_before_test": True,
                    "outer_test_outcomes_update_baseline": False,
                },
                "candidates": candidate_reports,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "development_only",
        "production_eligible": False,
        "public_probability_authorized": False,
        "claim_ceiling": {
            "descriptive_premap_association": False,
            "causal_draft_effect": False,
            "recommendation": False,
            "betting": False,
            "reliability": False,
        },
        "source_snapshot": {
            **source_hashes,
            "source_ids": [
                "scryglass:source:oracle-elixir-player-games",
                "scryglass:development:outcome-free-dependence-cluster-proxy",
            ],
            "availability_status": "snapshot_only_field_time_structurally_preevent_retrieval_unverified",
            "rights_status": "not_revalidated_for_public_serving",
        },
        "baseline_adjustment": {
            **BASELINE_CONFIG,
            "config_sha256": baseline_config_sha256,
            "status": "development_nuisance_only",
            "fit_scope": "pre_event_team_strength_is_fit_inside_each_temporal_fold",
            "served_scope": "baseline_logit_is_fixed_to_zero_for_neutral_score",
            "current_map_outcome_used_for_current_baseline": False,
            "future_outcomes_used_for_current_baseline": False,
            "team_identity_in_served_artifact": False,
        },
        "population": {
            "complete_rows": len(rows),
            "series_clusters": len(series_order),
            "dependence_clusters": len(series_order),
            "proxy_clustered_rows": proxy_clustered_rows,
            "unclustered_single_game_rows": len(unclustered_rows),
            "leagues": sorted({row.league for row in rows}),
            "patches": patches,
            "international_event_rows": len(international_rows),
        },
        "split_policy": {
            "folds": len(folds),
            "chronological": True,
            "series_grouped": False,
            "dependence_clustered": True,
            "series_identity_status": "authoritative_series_ids_unavailable; outcome_free_proxy_used_for_development_blocking",
            "participant_dependence_status": "outcome_free team-pair/time proxy used; player ids are not neutral features",
            "candidate_selection_on_validation_only": True,
            "outer_test_scored_only_for_selected_candidate": True,
            "candidate_search_opened_on_outer_test": False,
            "calibration_fit_on_outer_test": False,
            "baseline_fit_uses_only_pre_event_results": True,
            "baseline_updates_after_dependence_cluster": True,
            "outer_test_baseline_frozen_before_test": True,
            "outer_test_outcomes_update_baseline": False,
            "served_neutral_baseline_equalized": True,
        },
        "holdouts": {
            "future_patch": {
                "status": "development_diagnostic_only" if latest_patch else "unavailable",
                "patch_id": latest_patch,
                "rows": sum(row.patch == latest_patch for row in rows) if latest_patch else 0,
                "series_clusters": len({row.dependence_cluster_id for row in rows if row.patch == latest_patch}) if latest_patch else 0,
                "promotion": False,
            },
            "league": {"status": "available", "scored_within_each_test_fold": True},
            "international_event_or_meta": {
                "status": "development_diagnostic_only" if international_rows else "unavailable",
                "rows": len(international_rows),
                "leagues": sorted({row.league for row in international_rows}),
                "promotion": False,
            },
            "roster_change": {"status": "not_applicable", "reason": "neutral estimator contains no player or exact-roster identity terms"},
            "sparse_or_new_champion": {"status": "development_diagnostic_only", "promotion": False},
        },
        "candidate_order": list(CANDIDATE_ORDER),
        "calibration_order": list(CALIBRATION_ORDER),
        "selection": {"status": "not_selected", "winner_candidate_id": None, "winner_transform": None},
        "folds": fold_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    args = parser.parse_args()
    print(json.dumps(evaluate(args.root), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
