#!/usr/bin/env python3
"""Full-composition League of Legends draft model.

The model is deliberately a signed, pre-match composition estimator.  A row
contains one five-role composition per side and the only label is the observed
blue-side result.  Team/player strength is kept out of the pure draft edge,
then combined as a separate contextualized score when pre-match ratings are
available.

The first production slice uses a hierarchical ridge-logistic model:

* role-aware champion main effects, with league and patch deviations;
* unordered within-team synergy pairs;
* all observed blue-vs-red champion opposition pairs;
* an optional antisymmetric low-rank residual for sparse opposition cells.

Feature-specific penalties implement partial pooling: context and interaction
terms with little support are shrunk more strongly toward zero.  Prediction
also returns a diagonal-Laplace uncertainty approximation and an additive
ledger that allocates every pair contribution across its participating
champions.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from lol_kills.etl.aliases import normalize_champ
from lol_kills.etl.paths import MODELS_DIR, PARQUET_DIR

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = MODELS_DIR / "draft_composition.json"
RUNTIME_PATH = ROOT / "apps" / "lol-atlas" / "data" / "draft" / "composition_runtime.json"

ROLES = ("top", "jng", "mid", "bot", "sup")
DEFAULT_PRIOR_N = 25.0
DEFAULT_LOW_RANK = 4
PATCH_RE = re.compile(r"^(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class CompositionGame:
    game_id: str
    blue: tuple[tuple[str, str], ...]
    red: tuple[tuple[str, str], ...]
    y: int
    league: str
    patch: str
    date: pd.Timestamp | None


def _norm_role(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s.startswith("jng") or s.startswith("jung"):
        return "jng"
    if s.startswith("bot") or s.startswith("adc") or s.startswith("bottom"):
        return "bot"
    if s.startswith("sup") or s.startswith("util") or s.startswith("support"):
        return "sup"
    if s.startswith("mid"):
        return "mid"
    if s.startswith("top"):
        return "top"
    return s[:3]


def normalize_patch(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    match = PATCH_RE.match(s)
    if not match:
        return s
    major, minor = match.groups()
    # OE sometimes stores patch 16.10 as numeric 16.1. A one-digit suffix is
    # therefore right-padded (16.1 -> 16.10), while 16.01 remains 16.01.
    minor_text = (minor or "0").ljust(2, "0")
    return f"{int(major)}.{minor_text}"


def _patch_number(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _opposition_key(a: str, b: str) -> tuple[str, int]:
    """Canonical pair key plus orientation toward the first argument."""
    p = _pair(a, b)
    return f"opposition|{p[0]}|{p[1]}", 1 if (a, b) == p else -1


def _as_text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _strength_calibration() -> dict[str, Any]:
    """Return the existing time-safe team/player strength calibration."""
    path = MODELS_DIR / "elo_wr_calibration.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        team = data.get("team") or {}
        player = data.get("player") or {}
        blend = data.get("strength_blend") or {}
        return {
            "team_intercept": float(team.get("intercept", 0.14729)),
            "team_coef": float(team.get("coef", 2.37625)),
            "player_intercept": float(player.get("intercept", 0.13166)),
            "player_coef": float(player.get("coef", 3.64257)),
            "blend_intercept": float(blend.get("intercept", -2.47489)),
            "blend_coef_team": float(blend.get("coef_team", 2.84763)),
            "blend_coef_player": float(blend.get("coef_player", 2.07485)),
            "source": "time-safe team/player Dual Elo calibration",
        }
    except (OSError, ValueError, TypeError):
        return {
            "team_intercept": 0.14729,
            "team_coef": 2.37625,
            "player_intercept": 0.13166,
            "player_coef": 3.64257,
            "blend_intercept": -2.47489,
            "blend_coef_team": 2.84763,
            "blend_coef_player": 2.07485,
            "source": "documented calibration defaults",
        }


def default_training_paths() -> tuple[Path, Path]:
    """Find warehouse data first, then the checked-in public pack."""
    warehouse_maps = PARQUET_DIR / "maps.parquet"
    warehouse_players = PARQUET_DIR / "players.parquet"
    if warehouse_maps.exists() and warehouse_players.exists():
        return warehouse_maps, warehouse_players
    pack_root = ROOT / "apps" / "lol-atlas" / "public" / "packs"
    maps = sorted(pack_root.glob("v*/maps/year=*/part.parquet"))
    players = sorted(pack_root.glob("v*/player_games/year=*/part.parquet"))
    if maps and players:
        return maps[-1].parent.parent.parent, players[-1].parent.parent.parent
    raise FileNotFoundError("No maps/player parquet pair found for composition model")


def load_training_frames(
    maps_path: Path | None = None,
    players_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one warehouse pair or all year partitions in a public pack."""
    if maps_path is None or players_path is None:
        maps_path, players_path = default_training_paths()

    def read_partitioned(path: Path, leaf: str) -> pd.DataFrame:
        if path.is_file():
            return _read_table(path)
        parts = sorted(path.glob(f"{leaf}/year=*/part.parquet"))
        if not parts:
            parts = sorted(path.glob("year=*/part.parquet"))
        if not parts:
            raise FileNotFoundError(f"No parquet partitions under {path}")
        return pd.concat([_read_table(p) for p in parts], ignore_index=True)

    maps = read_partitioned(Path(maps_path), "maps")
    players = read_partitioned(Path(players_path), "player_games")
    return maps, players


def build_games(maps: pd.DataFrame, players: pd.DataFrame) -> list[CompositionGame]:
    """Join map labels to complete five-role drafts without outcome features."""
    required = {"y_blue_win"}
    missing = required - set(maps.columns)
    if missing:
        raise ValueError(f"maps is missing required columns: {sorted(missing)}")
    if "gameid" not in players.columns or "champion" not in players.columns:
        raise ValueError("players must contain gameid and champion")

    player_by_game: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in players.to_dict("records"):
        player_by_game[_as_text(row.get("gameid"))].append(row)

    games: list[CompositionGame] = []
    for raw in maps.to_dict("records"):
        y_raw = raw.get("y_blue_win")
        try:
            y = int(float(y_raw))
        except (TypeError, ValueError):
            continue
        if y not in (0, 1):
            continue
        gid = _as_text(raw.get("game_uid") or raw.get("oe_gameid"))
        plist = player_by_game.get(gid, [])
        by_side: dict[str, dict[str, str]] = {"Blue": {}, "Red": {}}
        for p in plist:
            side = _as_text(p.get("side")).strip().title()
            role = _norm_role(p.get("position"))
            champ = normalize_champ(_as_text(p.get("champion")))
            if side not in by_side or role not in ROLES or not champ:
                continue
            # A duplicate role is an ambiguous draft; fail closed for training.
            if role in by_side[side] and by_side[side][role] != champ:
                by_side[side][role] = ""
            else:
                by_side[side][role] = champ
        if any(len(by_side[s]) != 5 or any(not by_side[s].get(r) for r in ROLES) for s in by_side):
            continue
        date = pd.to_datetime(raw.get("date"), errors="coerce")
        date_value = None if pd.isna(date) else pd.Timestamp(date)
        games.append(
            CompositionGame(
                game_id=gid,
                blue=tuple((role, by_side["Blue"][role]) for role in ROLES),
                red=tuple((role, by_side["Red"][role]) for role in ROLES),
                y=y,
                league=_as_text(raw.get("league")).upper().strip() or "UNKNOWN",
                patch=normalize_patch(raw.get("patch")),
                date=date_value,
            )
        )
    games.sort(key=lambda g: (g.date is None, g.date or pd.Timestamp.min, g.game_id))
    return games


def _main_features(game: CompositionGame) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for sign, side in ((1.0, game.blue), (-1.0, game.red)):
        for role, champ in side:
            out[f"main|{role}|{champ}"] += sign
            out[f"league|{game.league}|{role}|{champ}"] += sign
            out[f"patch|{game.patch}|{role}|{champ}"] += sign
    return out


def feature_values(
    game: CompositionGame,
    components: Iterable[str] = ("main", "synergy", "opposition"),
) -> dict[str, float]:
    """Return the signed sparse feature vector for one complete draft."""
    enabled = set(components)
    out: dict[str, float] = defaultdict(float)
    if "main" in enabled:
        for key, value in _main_features(game).items():
            out[key] += value
    if "synergy" in enabled:
        for sign, side in ((1.0, game.blue), (-1.0, game.red)):
            champs = [champ for _role, champ in side]
            for i in range(len(champs)):
                for j in range(i + 1, len(champs)):
                    a, b = _pair(champs[i], champs[j])
                    out[f"synergy|{a}|{b}"] += sign
    if "opposition" in enabled:
        for _role_b, blue in game.blue:
            for _role_r, red in game.red:
                key, orientation = _opposition_key(blue, red)
                out[key] += float(orientation)
    return dict(out)


def _feature_group(key: str) -> str:
    return key.split("|", 1)[0]


def _penalty(group: str, n: int, prior_n: float = DEFAULT_PRIOR_N) -> float:
    """Feature-specific ridge penalty; sparse terms pool harder to zero."""
    support = max(float(n), 1.0)
    if group == "main":
        base, extra = 1.0, 4.0
    elif group in {"league", "patch"}:
        base, extra = 3.0, 12.0
    elif group == "synergy":
        base, extra = 5.0, 35.0
    else:
        base, extra = 6.0, 55.0
    return base + extra * prior_n / support


def _recency_weights(games: Sequence[CompositionGame], half_life_days: float = 365.0) -> np.ndarray:
    dates = [g.date for g in games if g.date is not None]
    if not dates:
        return np.ones(len(games), dtype=float)
    latest = max(dates)
    weights = []
    for game in games:
        if game.date is None:
            weights.append(0.5)
            continue
        age = max((latest - game.date).total_seconds() / 86400.0, 0.0)
        weights.append(0.5 ** (age / half_life_days))
    return np.asarray(weights, dtype=float)


def _matrix_for_games(
    games: Sequence[CompositionGame],
    components: Iterable[str],
    min_support: int,
) -> tuple[sparse.csr_matrix, list[str], dict[str, int]]:
    rows = [feature_values(game, components) for game in games]
    counts: Counter[str] = Counter()
    for values in rows:
        counts.update(key for key, value in values.items() if value != 0)
    feature_names = sorted(key for key, n in counts.items() if n >= min_support)
    index = {key: i for i, key in enumerate(feature_names)}
    data: list[float] = []
    row_idx: list[int] = []
    col_idx: list[int] = []
    for i, values in enumerate(rows):
        for key, value in values.items():
            j = index.get(key)
            if j is not None and value:
                row_idx.append(i)
                col_idx.append(j)
                data.append(value)
    matrix = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(len(games), len(feature_names)))
    return matrix, feature_names, dict(counts)


def _fit_logistic(
    games: Sequence[CompositionGame],
    components: Iterable[str],
    min_support: int = 3,
    prior_n: float = DEFAULT_PRIOR_N,
) -> dict[str, Any]:
    if not games:
        raise ValueError("cannot fit composition model on zero games")
    X, names, counts = _matrix_for_games(games, components, min_support)
    groups = [_feature_group(name) for name in names]
    lambdas = np.asarray([_penalty(group, counts[name], prior_n) for name, group in zip(names, groups)], dtype=float)
    if names:
        X_scaled = X.multiply(1.0 / np.sqrt(lambdas))
    else:
        X_scaled = sparse.csr_matrix((len(games), 0))
    y = np.asarray([g.y for g in games], dtype=int)
    weights = _recency_weights(games)
    clf = LogisticRegression(
        C=1.0,
        fit_intercept=True,
        penalty="l2",
        solver="saga",
        max_iter=1200,
        tol=1e-4,
        random_state=0,
    )
    clf.fit(X_scaled, y, sample_weight=weights)
    scaled_coef = clf.coef_[0] if names else np.zeros(0, dtype=float)
    coef = scaled_coef / np.sqrt(lambdas) if names else np.zeros(0, dtype=float)
    p = np.clip(clf.predict_proba(X_scaled)[:, 1], 1e-6, 1 - 1e-6)
    feature_specs: dict[str, dict[str, Any]] = {}
    for j, name in enumerate(names):
        col = X.getcol(j).toarray().ravel()
        info = float(np.sum(weights * p * (1.0 - p) * col * col) + lambdas[j])
        feature_specs[name] = {
            "coef": float(coef[j]),
            "n": int(counts[name]),
            "prior_n": prior_n,
            "shrinkage": float(counts[name] / (counts[name] + prior_n)),
            "se": float(1.0 / math.sqrt(max(info, 1e-12))),
            "penalty": float(lambdas[j]),
            "group": groups[j],
        }
    role_champion_counts: Counter[str] = Counter()
    champion_counts: Counter[str] = Counter()
    for game in games:
        for _role, champ in game.blue + game.red:
            champion_counts[champ] += 1
        for role, champ in game.blue + game.red:
            role_champion_counts[f"{role}|{champ}"] += 1
    return {
        "intercept": float(clf.intercept_[0]),
        "feature_specs": feature_specs,
        "champion_counts": dict(champion_counts),
        "role_champion_counts": dict(role_champion_counts),
        "n_games": len(games),
        "components": sorted(set(components)),
        "min_support": min_support,
        "prior_n": prior_n,
        "recency_half_life_days": 365.0,
    }


def _raw_edge(model: Mapping[str, Any], game: CompositionGame) -> float:
    values = feature_values(game, model.get("components") or ())
    specs = model.get("feature_specs") or {}
    return float(model.get("intercept", 0.0) + sum(values.get(key, 0.0) * float(row.get("coef", 0.0)) for key, row in specs.items()))


def _fit_low_rank_residual(
    model: dict[str, Any],
    games: Sequence[CompositionGame],
    rank: int,
    min_pair_support: int = 5,
) -> dict[str, Any]:
    if rank <= 0:
        return {"rank": 0, "champions": [], "left": [], "right": []}
    champs = sorted({champ for game in games for _role, champ in game.blue + game.red})
    idx = {champ: i for i, champ in enumerate(champs)}
    mat = np.zeros((len(champs), len(champs)), dtype=float)
    den = np.zeros_like(mat)
    pair_n = np.zeros_like(mat)
    for game in games:
        base_p = 1.0 / (1.0 + math.exp(-_raw_edge(model, game)))
        residual = float(game.y) - base_p
        for _role_b, blue in game.blue:
            for _role_r, red in game.red:
                i, j = idx[blue], idx[red]
                sign = 1.0 if blue <= red else -1.0
                weight = 1.0
                mat[i, j] += weight * sign * residual
                den[i, j] += weight * base_p * (1.0 - base_p) + 4.0
                pair_n[i, j] += 1.0
    raw = np.divide(mat, den, out=np.zeros_like(mat), where=den > 0)
    raw = np.where(pair_n >= min_pair_support, raw, 0.0)
    anti = 0.5 * (raw - raw.T)
    if not np.any(anti):
        return {"rank": 0, "champions": champs, "left": [], "right": []}
    u, singular, vh = np.linalg.svd(anti, full_matrices=False)
    k = min(rank, len(singular))
    left = u[:, :k] * np.sqrt(singular[:k])
    right = vh[:k, :].T * np.sqrt(singular[:k])
    return {
        "rank": int(k),
        "champions": champs,
        "left": left.tolist(),
        "right": right.tolist(),
        "pair_support_floor": min_pair_support,
    }


def _low_rank_value(low_rank: Mapping[str, Any], blue: str, red: str) -> float:
    champs = low_rank.get("champions") or []
    try:
        i, j = champs.index(blue), champs.index(red)
    except ValueError:
        return 0.0
    left = low_rank.get("left") or []
    right = low_rank.get("right") or []
    if not left or not right:
        return 0.0
    value = float(np.dot(np.asarray(left[i], dtype=float), np.asarray(right[j], dtype=float)))
    reverse = float(np.dot(np.asarray(left[j], dtype=float), np.asarray(right[i], dtype=float)))
    return 0.5 * (value - reverse)


def _calibration(model: Mapping[str, Any], games: Sequence[CompositionGame]) -> dict[str, float]:
    if not games:
        return {"intercept": 0.0, "slope": 1.0}
    x = np.asarray([_raw_edge(model, game) for game in games], dtype=float)
    y = np.asarray([game.y for game in games], dtype=int)
    return _fit_calibration_curve(x, y)


def _fit_calibration_curve(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Stable two-parameter calibration fit without high-dimensional solver state."""
    if len(np.unique(y)) < 2:
        return {"intercept": 0.0, "slope": 1.0}

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(theta[0] + theta[1] * x, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        loss = -float(np.sum(y * np.log(np.clip(p, 1e-12, 1.0)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1.0))))
        loss += 1e-6 * float(np.dot(theta, theta))
        grad = np.asarray([np.sum(p - y), np.sum((p - y) * x)], dtype=float) + 2e-6 * theta
        return loss, grad

    result = minimize(lambda theta: objective(theta)[0], np.asarray([0.0, 1.0]), jac=lambda theta: objective(theta)[1], method="BFGS")
    theta = result.x if np.all(np.isfinite(result.x)) else np.asarray([0.0, 1.0])
    return {"intercept": float(theta[0]), "slope": float(theta[1])}


def _sigmoid(x: float) -> float:
    if x >= 30:
        return 1.0
    if x <= -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _evidence_label(n: int) -> str:
    if n >= 100:
        return "well supported"
    if n >= 30:
        return "supported"
    if n >= 10:
        return "thin"
    return "very thin"


def predict_composition(
    model: Mapping[str, Any],
    blue: Sequence[str],
    red: Sequence[str],
    *,
    blue_roles: Sequence[str] | None = None,
    red_roles: Sequence[str] | None = None,
    league: str | None = None,
    patch: str | None = None,
    elo_diff: float | None = None,
    team_elo_diff: float | None = None,
    player_elo_diff: float | None = None,
    strength_source: str | None = None,
) -> dict[str, Any]:
    """Score a five-v-five draft and return an exactly reconciling ledger."""
    if len(blue) != 5 or len(red) != 5:
        raise ValueError("need five picks per side")
    roles_b = tuple(_norm_role(r) for r in (blue_roles or ROLES))
    roles_r = tuple(_norm_role(r) for r in (red_roles or ROLES))
    if len(roles_b) != 5 or len(roles_r) != 5:
        raise ValueError("need five roles per side")
    b = tuple((roles_b[i], normalize_champ(str(blue[i]))) for i in range(5))
    r = tuple((roles_r[i], normalize_champ(str(red[i]))) for i in range(5))
    game = CompositionGame("query", b, r, 0, (league or "UNKNOWN").upper().strip() or "UNKNOWN", normalize_patch(patch), None)
    specs = model.get("feature_specs") or {}
    components = set(model.get("components") or ())
    champion_parts: list[dict[str, Any]] = []
    for side_name, sign, picks in (("blue", 1.0, b), ("red", -1.0, r)):
        for role, champ in picks:
            direct = 0.0
            direct_var = 0.0
            for key in (
                f"main|{role}|{champ}",
                f"league|{game.league}|{role}|{champ}",
                f"patch|{game.patch}|{role}|{champ}",
            ):
                if key in specs:
                    direct += sign * float(specs[key].get("coef", 0.0))
                    direct_var += float(specs[key].get("se", 0.0)) ** 2
            champion_parts.append(
                {
                    "champion": champ,
                    "side": side_name,
                    "role": role,
                    "direct_effect": direct,
                    "team_synergy": 0.0,
                    "enemy_interaction": 0.0,
                    "edge_contribution": direct,
                    "uncertainty_logit": math.sqrt(direct_var),
                }
            )

    by_side_index = {(row["side"], row["champion"], row["role"]): row for row in champion_parts}
    main_logit = 0.0
    synergy_logit = 0.0
    opposition_logit = 0.0
    low_rank_logit = 0.0
    edge_var = 0.0
    # Main term contribution is already allocated above.
    for row in champion_parts:
        main_logit += float(row["direct_effect"])
        edge_var += float(row["uncertainty_logit"]) ** 2

    if "synergy" in components:
        for side_name, sign, picks in (("blue", 1.0, b), ("red", -1.0, r)):
            for i in range(5):
                for j in range(i + 1, 5):
                    a, c = _pair(picks[i][1], picks[j][1])
                    spec = specs.get(f"synergy|{a}|{c}")
                    if not spec:
                        continue
                    value = sign * float(spec.get("coef", 0.0))
                    synergy_logit += value
                    share = value / 2.0
                    for role, champ in (picks[i], picks[j]):
                        row = by_side_index[(side_name, champ, role)]
                        row["team_synergy"] += share
                        row["edge_contribution"] += share
                        row["uncertainty_logit"] = math.sqrt(
                            float(row["uncertainty_logit"]) ** 2 + (float(spec.get("se", 0.0)) / 2.0) ** 2
                        )
                    edge_var += float(spec.get("se", 0.0)) ** 2

    low_rank = model.get("low_rank") or {}
    for _role_b, blue_champ in b:
        for _role_r, red_champ in r:
            if "opposition" in components:
                key, orientation = _opposition_key(blue_champ, red_champ)
                spec = specs.get(key)
                if spec:
                    value = float(orientation) * float(spec.get("coef", 0.0))
                    opposition_logit += value
                    edge_var += float(spec.get("se", 0.0)) ** 2
                    blue_row = next(row for row in champion_parts if row["side"] == "blue" and row["champion"] == blue_champ and row["role"] == next(role for role, champ in b if champ == blue_champ))
                    red_row = next(row for row in champion_parts if row["side"] == "red" and row["champion"] == red_champ and row["role"] == next(role for role, champ in r if champ == red_champ))
                    blue_row["enemy_interaction"] += value / 2.0
                    red_row["enemy_interaction"] += value / 2.0
                    blue_row["edge_contribution"] += value / 2.0
                    red_row["edge_contribution"] += value / 2.0
                    blue_row["uncertainty_logit"] = math.sqrt(float(blue_row["uncertainty_logit"]) ** 2 + (float(spec.get("se", 0.0)) / 2.0) ** 2)
                    red_row["uncertainty_logit"] = math.sqrt(float(red_row["uncertainty_logit"]) ** 2 + (float(spec.get("se", 0.0)) / 2.0) ** 2)
            if int(low_rank.get("rank", 0)) > 0:
                value = _low_rank_value(low_rank, blue_champ, red_champ)
                low_rank_logit += value
                blue_row = next(row for row in champion_parts if row["side"] == "blue" and row["champion"] == blue_champ and row["role"] == next(role for role, champ in b if champ == blue_champ))
                red_row = next(row for row in champion_parts if row["side"] == "red" and row["champion"] == red_champ and row["role"] == next(role for role, champ in r if champ == red_champ))
                blue_row["enemy_interaction"] += value / 2.0
                red_row["enemy_interaction"] += value / 2.0
                blue_row["edge_contribution"] += value / 2.0
                red_row["edge_contribution"] += value / 2.0

    composition_edge = main_logit + synergy_logit + opposition_logit + low_rank_logit
    side_advantage = float(model.get("intercept", 0.0))
    model_edge = side_advantage + composition_edge
    edge_se = math.sqrt(max(edge_var, 1e-12))
    cal = model.get("calibration") or {"intercept": 0.0, "slope": 1.0}
    cal_intercept = float(cal.get("intercept", 0.0))
    cal_slope = float(cal.get("slope", 1.0))
    calibrated_logit = cal_intercept + cal_slope * model_edge
    p_blue = _sigmoid(calibrated_logit)
    p_lo = _sigmoid(cal_intercept + cal_slope * (model_edge - 1.96 * edge_se))
    p_hi = _sigmoid(cal_intercept + cal_slope * (model_edge + 1.96 * edge_se))
    strength = model.get("strength_calibration") or {}
    team_diff = team_elo_diff if team_elo_diff is not None else elo_diff
    player_diff = player_elo_diff
    team_p = (
        _sigmoid(float(strength.get("team_intercept", 0.14729)) + float(strength.get("team_coef", 2.37625)) * float(team_diff) / 400.0)
        if team_diff is not None
        else None
    )
    player_p = (
        _sigmoid(float(strength.get("player_intercept", 0.13166)) + float(strength.get("player_coef", 3.64257)) * float(player_diff) / 400.0)
        if player_diff is not None
        else None
    )
    strength_logit = None
    if team_p is not None and player_p is not None:
        strength_logit = (
            float(strength.get("blend_intercept", -2.47489))
            + float(strength.get("blend_coef_team", 2.84763)) * float(team_p)
            + float(strength.get("blend_coef_player", 2.07485)) * float(player_p)
        )
    elif team_p is not None:
        strength_logit = math.log(max(team_p, 1e-12) / max(1.0 - team_p, 1e-12))
    elif player_p is not None:
        strength_logit = math.log(max(player_p, 1e-12) / max(1.0 - player_p, 1e-12))
    p_strength = _sigmoid(strength_logit + cal_slope * composition_edge) if strength_logit is not None else None
    role_counts = model.get("role_champion_counts") or {}
    for row in champion_parts:
        n = int(role_counts.get(f"{row['role']}|{row['champion']}", 0))
        row["evidence"] = {
            "games": n,
            "shrinkage": n / (n + DEFAULT_PRIOR_N),
            "label": _evidence_label(n),
            "uncertainty_logit": round(float(row["uncertainty_logit"]), 4),
        }
        for key in ("direct_effect", "team_synergy", "enemy_interaction", "edge_contribution"):
            row[key] = round(float(row[key]), 6)
    contribution_sum = sum(float(row["edge_contribution"]) for row in champion_parts)
    explanation = {
        "edge": round(contribution_sum + side_advantage, 6),
        "composition_edge": round(contribution_sum, 6),
        "side_advantage": round(side_advantage, 6),
        "champions": champion_parts,
        "reconciles": abs(contribution_sum - composition_edge) < 1e-5,
        "attribution": "symmetric pair allocation: each synergy/opposition pair is split equally across its two champions",
    }
    return {
        "draft_score_blue": round(100.0 * p_blue, 2),
        "draft_score_red": round(100.0 * (1.0 - p_blue), 2),
        "draft_edge": round(100.0 * (2.0 * p_blue - 1.0), 2),
        "confidence": round(float(np.clip(1.0 / (1.0 + 2.0 * edge_se), 0.05, 0.98)), 3),
        "p_blue_draft": round(p_blue, 4),
        "raw": {
            "p_blue": round(p_blue, 4),
            "score_blue": round(100.0 * p_blue, 2),
            "score_red": round(100.0 * (1.0 - p_blue), 2),
            "edge": round(100.0 * (2.0 * p_blue - 1.0), 2),
            "source": "composition only; no roster/player strength",
        },
        "contextualized": (
            {
                "p_blue": round(p_strength, 4),
                "score_blue": round(100.0 * p_strength, 2),
                "score_red": round(100.0 * (1.0 - p_strength), 2),
                "edge": round(100.0 * (2.0 * p_strength - 1.0), 2),
            "source": strength_source or "pre-match strength input",
            }
            if p_strength is not None
            else None
        ),
        "strength": {
            "team_elo_diff": round(float(team_diff), 2) if team_diff is not None else None,
            "player_elo_diff": round(float(player_diff), 2) if player_diff is not None else None,
            "source": strength_source or ("explicit pre-match strength" if team_diff is not None else "unavailable"),
        },
        "wr_bump_pp": round(100.0 * (p_blue - 0.5), 2),
        "posterior_width": round(edge_se, 4),
        "uncertainty": {
            "edge_se_logit": round(edge_se, 4),
            "p_blue_95": [round(p_lo, 4), round(p_hi, 4)],
            "method": "diagonal Laplace approximation; coefficient correlations are not represented",
        },
        "calibration": {
            "league": league,
            "patch": patch,
            "source": model.get("calibration_source", "time-heldout"),
            "intercept": round(cal_intercept, 4),
            "slope": round(cal_slope, 4),
            "p_blue_with_strength": round(p_strength, 4) if p_strength is not None else None,
        },
        "components": {
            "main_logit": round(main_logit, 6),
            "synergy_logit": round(synergy_logit, 6),
            "opposition_logit": round(opposition_logit, 6),
            "low_rank_logit": round(low_rank_logit, 6),
            "composition_edge": round(composition_edge, 6),
            "model_edge": round(model_edge, 6),
            "side_advantage_logit": round(side_advantage, 6),
            # Compatibility names for existing board consumers.
            "win_logit_blue": round(main_logit, 6),
            "win_logit_red": 0.0,
            "pair_logit": round(synergy_logit + opposition_logit + low_rank_logit, 6),
            "win_edge": round(composition_edge, 6),
            "known_frac_blue": round(sum(1 for row in champion_parts if row["side"] == "blue" and row["evidence"]["games"] > 0) / 5.0, 3),
            "known_frac_red": round(sum(1 for row in champion_parts if row["side"] == "red" and row["evidence"]["games"] > 0) / 5.0, 3),
        },
        "explanation": explanation,
        "blue": [champ for _role, champ in b],
        "red": [champ for _role, champ in r],
        "note": "Full-composition draft model: role-aware direct effects, within-team synergy, all 25 enemy interactions, and sparse low-rank residual. Strength is reported separately when supplied.",
    }


def _metrics(y: Sequence[int], p: Sequence[float]) -> dict[str, float]:
    y_arr = np.asarray(y, dtype=int)
    p_arr = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(p_arr / (1.0 - p_arr))
    if len(np.unique(y_arr)) > 1:
        fitted = _fit_calibration_curve(logits, y_arr)
        slope = fitted["slope"]
        intercept = fitted["intercept"]
    else:
        slope, intercept = 1.0, 0.0
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
        mask = (p_arr >= lo) & ((p_arr < hi) if hi < 1 else (p_arr <= hi))
        if np.any(mask):
            ece += float(np.sum(mask)) / len(p_arr) * abs(float(np.mean(p_arr[mask])) - float(np.mean(y_arr[mask])))
    return {
        "n": int(len(y_arr)),
        "log_loss": float(log_loss(y_arr, p_arr, labels=[0, 1])),
        "brier": float(brier_score_loss(y_arr, p_arr)),
        "calibration_slope": slope,
        "calibration_intercept": intercept,
        "ece_10": ece,
    }


def _split_time(games: Sequence[CompositionGame], train_frac: float = 0.8) -> tuple[list[CompositionGame], list[CompositionGame]]:
    cut = max(1, min(len(games) - 1, int(len(games) * train_frac)))
    ordered = sorted(games, key=lambda g: (g.date is None, g.date or pd.Timestamp.min, g.game_id))
    return ordered[:cut], ordered[cut:]


def _evaluate(model: Mapping[str, Any], games: Sequence[CompositionGame]) -> dict[str, float]:
    preds = [predict_composition(model, [c for _r, c in g.blue], [c for _r, c in g.red], blue_roles=[r for r, _c in g.blue], red_roles=[r for r, _c in g.red], league=g.league, patch=g.patch)["p_blue_draft"] for g in games]
    return _metrics([g.y for g in games], preds)


def _fit_holdout(
    train: Sequence[CompositionGame],
    test: Sequence[CompositionGame],
    components: Iterable[str],
    low_rank_rank: int = 0,
) -> dict[str, float]:
    model = _fit_logistic(train, components)
    model["low_rank"] = _fit_low_rank_residual(model, train, low_rank_rank)
    cal_train, cal = _split_time(train, 0.8) if len(train) > 20 else (list(train), [])
    if cal:
        cal_model = _fit_logistic(cal_train, components)
        cal_model["low_rank"] = _fit_low_rank_residual(cal_model, cal_train, low_rank_rank)
        model["calibration"] = _calibration(cal_model, cal)
    return _evaluate(model, test)


def fit_composition_artifact(
    games: Sequence[CompositionGame],
    *,
    low_rank_rank: int = DEFAULT_LOW_RANK,
    min_support: int = 3,
    validate: bool = True,
) -> dict[str, Any]:
    """Fit the checked-in artifact with time/future-patch/league diagnostics."""
    if len(games) < 50:
        raise ValueError(f"need at least 50 complete drafts, got {len(games)}")
    train, test = _split_time(games, 0.8)
    cal_train, calibration_games = _split_time(train, 0.8)
    model = _fit_logistic(cal_train, ("main", "synergy", "opposition"), min_support=min_support)
    model["low_rank"] = _fit_low_rank_residual(model, cal_train, low_rank_rank)
    model["calibration"] = _calibration(model, calibration_games)
    model["calibration_source"] = "time-heldout calibration slice"
    # Preserve the separately fit strength channel; it is not part of the raw
    # draft edge and is combined only for the contextualized score.
    model["strength_calibration"] = _strength_calibration()

    validation: dict[str, Any] = {}
    if validate:
        validation["time_holdout"] = _evaluate(model, test)
        patch_values = sorted({g.patch for g in games}, key=_patch_number)
        future = set(patch_values[-2:]) if len(patch_values) >= 3 else set(patch_values[-1:])
        patch_train = [g for g in games if g.patch not in future]
        patch_test = [g for g in games if g.patch in future]
        if patch_train and patch_test:
            validation["future_patch_holdout"] = _fit_holdout(patch_train, patch_test, ("main", "synergy", "opposition"), low_rank_rank)
            validation["future_patch"] = sorted(future)
        leagues = sorted({g.league for g in games})
        league = max(leagues, key=lambda x: sum(g.league == x for g in games))
        league_train = [g for g in games if g.league != league]
        league_test = [g for g in games if g.league == league]
        if league_train and league_test:
            validation["league_holdout"] = _fit_holdout(league_train, league_test, ("main", "synergy", "opposition"), low_rank_rank)
            validation["league"] = league
        ablations = {
            "additive_only": (("main",), 0),
            "plus_synergy": (("main", "synergy"), 0),
            "plus_opposition": (("main", "synergy", "opposition"), 0),
            "plus_low_rank": (("main", "synergy", "opposition"), low_rank_rank),
        }
        validation["ablations_time_holdout"] = {
            name: _fit_holdout(cal_train, test, components, rank) for name, (components, rank) in ablations.items()
        }
    model.update(
        {
            "version": 1,
            "estimand": "pre-match blue-side map-win probability conditional on champion composition, role, league, and patch; no roster/player/team strength in pure draft edge",
            "n_games_total": len(games),
            "n_games_fit": len(cal_train),
            "date_min": min((g.date for g in games if g.date is not None), default=None),
            "date_max": max((g.date for g in games if g.date is not None), default=None),
            "validation": validation,
            "limitations": [
                "observational draft data cannot identify causal champion effects",
                "role/league/patch deviations are ridge-pooled and should be treated as estimates, not matchup truths",
                "uncertainty is a diagonal Laplace approximation and omits coefficient covariance",
                "low-rank residual is a post-fit sparse correction, not a replacement for a larger embedding model",
            ],
        }
    )
    return model


def export_runtime(model: Mapping[str, Any], path: Path = RUNTIME_PATH) -> dict[str, Any]:
    """Write the browser-sized artifact, omitting validation text."""
    runtime = {
        "version": model.get("version", 1),
        "estimand": model.get("estimand"),
        "intercept": model.get("intercept", 0.0),
        "feature_specs": model.get("feature_specs", {}),
        "role_champion_counts": model.get("role_champion_counts", {}),
        "components": model.get("components", []),
        "prior_n": model.get("prior_n", DEFAULT_PRIOR_N),
        "low_rank": model.get("low_rank", {"rank": 0, "champions": [], "left": [], "right": []}),
        "calibration": model.get("calibration", {"intercept": 0.0, "slope": 1.0}),
        "calibration_source": model.get("calibration_source", "time-heldout calibration slice"),
        "strength_calibration": model.get("strength_calibration", {}),
        "n_games_fit": model.get("n_games_fit"),
        "date_min": str(model.get("date_min")) if model.get("date_min") is not None else None,
        "date_max": str(model.get("date_max")) if model.get("date_max") is not None else None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return runtime


def fit_from_paths(
    maps_path: Path | None = None,
    players_path: Path | None = None,
    *,
    output: Path = MODEL_PATH,
    runtime: Path = RUNTIME_PATH,
    low_rank_rank: int = DEFAULT_LOW_RANK,
    validate: bool = True,
) -> dict[str, Any]:
    maps, players = load_training_frames(maps_path, players_path)
    games = build_games(maps, players)
    model = fit_composition_artifact(games, low_rank_rank=low_rank_rank, validate=validate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
    export_runtime(model, runtime)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=None)
    parser.add_argument("--players", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--runtime", type=Path, default=RUNTIME_PATH)
    parser.add_argument("--low-rank", type=int, default=DEFAULT_LOW_RANK)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    model = fit_from_paths(
        args.maps,
        args.players,
        output=args.output,
        runtime=args.runtime,
        low_rank_rank=args.low_rank,
        validate=not args.no_validate,
    )
    print(json.dumps({"n_games": model["n_games_total"], "fit": model["n_games_fit"], "validation": model.get("validation", {})}, indent=2, default=str))


if __name__ == "__main__":
    main()
