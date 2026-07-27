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
* no low-rank residual in bounded/public scores until its uncertainty is
  estimated and propagated.

Feature-specific penalties implement partial pooling: context and interaction
terms with little support are shrunk more strongly toward zero.  Prediction
also returns a diagonal-Laplace uncertainty approximation and an additive
ledger that allocates every pair contribution across its participating
champions.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import importlib.metadata
import json
import math
import platform
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
PACKED_RUNTIME_PATH = (
    ROOT
    / "apps"
    / "lol-atlas"
    / "data"
    / "draft"
    / "composition_runtime.json.gz.b64"
)

ROLES = ("top", "jng", "mid", "bot", "sup")
DEFAULT_PRIOR_N = 25.0
DEFAULT_LOW_RANK = 0
RUNTIME_VERSION = 2
UNCERTAINTY_SCHEMA_VERSION = "1.0.0"
STRENGTH_CALIBRATION_SCHEMA_VERSION = "1.0.0"
PATCH_RE = re.compile(r"^(\d+)(?:\.(\d+))?")
MODEL_CODE_PATHS = (
    Path(__file__).resolve(),
    (ROOT / "lol_kills" / "etl" / "aliases.py").resolve(),
    (ROOT / "requirements-model-lock.txt").resolve(),
)


class CompositionArtifactError(RuntimeError):
    """Raised when a bounded score cannot honor its artifact contract."""


def model_code_sha256(paths: Sequence[Path] = MODEL_CODE_PATHS) -> str:
    """Hash the exact source bundle that defines fitted draft coefficients."""
    digest = hashlib.sha256()
    for source_path in sorted((Path(path).resolve() for path in paths), key=str):
        try:
            relative = source_path.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            relative = source_path.as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def numerical_environment() -> dict[str, Any]:
    """Record the numerical runtime needed to reproduce the fit exactly."""
    packages = {}
    for distribution in ("numpy", "pandas", "scipy", "scikit-learn"):
        packages[distribution] = importlib.metadata.version(distribution)
    return {
        "python": platform.python_version(),
        "packages": packages,
    }


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


def normalize_patch(
    value: Any,
    *,
    allow_source_numeric_minor: bool = False,
) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"
    s = str(value).strip()
    if not s:
        return "unknown"
    match = PATCH_RE.match(s)
    if not match:
        return s
    major, minor = match.groups()
    minor_text = minor or "0"
    if len(minor_text) == 1:
        if not allow_source_numeric_minor:
            raise CompositionArtifactError(
                f"ambiguous patch {s!r}; use an explicit two-digit minor"
            )
        # Oracle's Elixir historically serialises patch numbers through a
        # numeric field, so 16.10 reaches this warehouse boundary as "16.1".
        # Public/query strings never receive this source-only coercion.
        minor_text = minor_text.ljust(2, "0")
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
    if a == b:
        # A mirrored champion matchup carries no directional opposition
        # information.  Keeping the key is useful for deterministic feature
        # lookup, but its signed feature value must be exactly zero.
        return f"opposition|{p[0]}|{p[1]}", 0
    return f"opposition|{p[0]}|{p[1]}", 1 if (a, b) == p else -1


def _as_text(value: Any) -> str:
    return "" if value is None or (isinstance(value, float) and math.isnan(value)) else str(value)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _disabled_low_rank() -> dict[str, Any]:
    return {
        "status": "disabled",
        "rank": 0,
        "champions": [],
        "left": [],
        "right": [],
        "reason": (
            "bounded/public scoring disables low-rank residuals until their "
            "fit and prediction uncertainty are estimated"
        ),
    }


def _finite_number(value: Any, label: str, *, non_negative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise CompositionArtifactError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise CompositionArtifactError(f"{label} must be finite")
    if non_negative and number < 0:
        raise CompositionArtifactError(f"{label} must be non-negative")
    return number


def _nonempty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CompositionArtifactError(f"{label} must be a non-empty string")
    return value.strip()


def _relative_artifact_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _unavailable_strength_calibration(
    path: Path,
    reason: str,
    *,
    artifact_sha256: str | None,
    artifact_version: int | None,
) -> dict[str, Any]:
    return {
        "schema_version": STRENGTH_CALIBRATION_SCHEMA_VERSION,
        "status": "unavailable",
        "reason": reason,
        "source": {
            "artifact": _relative_artifact_path(path),
            "artifact_sha256": artifact_sha256,
            "artifact_version": artifact_version,
        },
    }


def _strength_calibration(path: Path | None = None) -> dict[str, Any]:
    """Load complete chronological metadata or explicitly disable context.

    Legacy and partial artifacts are never supplemented with application
    constants. Raw composition scoring remains independent, while any request
    for contextual strength fails closed.
    """

    source_path = path or (MODELS_DIR / "elo_wr_calibration.json")
    try:
        raw = source_path.read_bytes()
    except OSError as exc:
        return _unavailable_strength_calibration(
            source_path,
            f"strength calibration source is unreadable: {exc.__class__.__name__}",
            artifact_sha256=None,
            artifact_version=None,
        )
    digest = hashlib.sha256(raw).hexdigest()
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return _unavailable_strength_calibration(
            source_path,
            f"strength calibration source is invalid JSON: {exc.__class__.__name__}",
            artifact_sha256=digest,
            artifact_version=None,
        )
    version_raw = data.get("version") if isinstance(data, Mapping) else None
    version = (
        int(version_raw)
        if isinstance(version_raw, int) and not isinstance(version_raw, bool)
        else None
    )
    try:
        if not isinstance(data, Mapping):
            raise CompositionArtifactError("strength calibration must be an object")
        if version is None or version < 2:
            raise CompositionArtifactError(
                "strength calibration requires version 2 or newer"
            )
        if data.get("status") != "validated_time_holdout":
            raise CompositionArtifactError(
                "strength calibration is not validated_time_holdout"
            )
        split = data.get("time_split")
        if not isinstance(split, Mapping):
            raise CompositionArtifactError("strength calibration lacks time_split")
        fit_cutoff = _nonempty_string(
            split.get("train_end"), "strength calibration time_split.train_end"
        )
        holdout_start = _nonempty_string(
            split.get("holdout_start"),
            "strength calibration time_split.holdout_start",
        )
        if split.get("strictly_future_holdout") is not True:
            raise CompositionArtifactError(
                "strength calibration holdout is not marked strictly future"
            )
        fit_ts = pd.Timestamp(fit_cutoff)
        holdout_ts = pd.Timestamp(holdout_start)
        if pd.isna(fit_ts) or pd.isna(holdout_ts) or fit_ts >= holdout_ts:
            raise CompositionArtifactError(
                "strength calibration fit cutoff must precede holdout start"
            )

        team = data.get("team")
        player = data.get("player")
        blend = data.get("strength_blend")
        if not isinstance(team, Mapping):
            raise CompositionArtifactError("strength calibration lacks team block")
        if not isinstance(player, Mapping):
            raise CompositionArtifactError("strength calibration lacks player block")
        if not isinstance(blend, Mapping):
            raise CompositionArtifactError("strength calibration lacks blend block")
        for label, block in (("team", team), ("player", player)):
            if block.get("fit_split") != "train":
                raise CompositionArtifactError(
                    f"strength calibration {label}.fit_split must equal train"
                )
            _finite_number(block.get("intercept"), f"{label}.intercept")
            _finite_number(block.get("coef"), f"{label}.coef")
            if int(block.get("n_train") or 0) <= 0:
                raise CompositionArtifactError(f"{label}.n_train must be positive")
            holdout = block.get("holdout")
            if not isinstance(holdout, Mapping) or int(holdout.get("n") or 0) <= 0:
                raise CompositionArtifactError(
                    f"{label}.holdout.n must be positive"
                )
        if blend.get("fit_split") != "train":
            raise CompositionArtifactError(
                "strength calibration blend.fit_split must equal train"
            )
        for key in ("intercept", "coef_team", "coef_player"):
            _finite_number(blend.get(key), f"blend.{key}")
        if int(blend.get("n_train") or 0) <= 0:
            raise CompositionArtifactError("blend.n_train must be positive")
        if int(blend.get("n_holdout") or 0) <= 0:
            raise CompositionArtifactError("blend.n_holdout must be positive")
    except (
        CompositionArtifactError,
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        return _unavailable_strength_calibration(
            source_path,
            str(exc),
            artifact_sha256=digest,
            artifact_version=version,
        )

    calibration_id = f"strength-calibration-v{version}-{digest[:16]}"
    return {
        "schema_version": STRENGTH_CALIBRATION_SCHEMA_VERSION,
        "status": "available",
        "calibration_id": calibration_id,
        "fit_cutoff": fit_cutoff,
        "holdout_start": holdout_start,
        "source": {
            "artifact": _relative_artifact_path(source_path),
            "artifact_sha256": digest,
            "artifact_version": version,
        },
        "team": {
            "model_id": f"{calibration_id}-team",
            "intercept": _finite_number(team.get("intercept"), "team.intercept"),
            "coef": _finite_number(team.get("coef"), "team.coef"),
        },
        "player": {
            "model_id": f"{calibration_id}-player",
            "intercept": _finite_number(
                player.get("intercept"), "player.intercept"
            ),
            "coef": _finite_number(player.get("coef"), "player.coef"),
        },
        "blend": {
            "model_id": f"{calibration_id}-blend",
            "intercept": _finite_number(blend.get("intercept"), "blend.intercept"),
            "coef_team": _finite_number(blend.get("coef_team"), "blend.coef_team"),
            "coef_player": _finite_number(
                blend.get("coef_player"), "blend.coef_player"
            ),
        },
    }


def _require_strength_calibration(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompositionArtifactError("strength_calibration is required")
    if value.get("schema_version") != STRENGTH_CALIBRATION_SCHEMA_VERSION:
        raise CompositionArtifactError(
            "strength_calibration.schema_version is unsupported"
        )
    status = value.get("status")
    if status != "available":
        reason = value.get("reason")
        detail = f": {reason}" if isinstance(reason, str) and reason else ""
        raise CompositionArtifactError(
            f"strength calibration is unavailable{detail}"
        )
    _nonempty_string(value.get("calibration_id"), "strength calibration ID")
    fit_cutoff = _nonempty_string(
        value.get("fit_cutoff"), "strength calibration fit_cutoff"
    )
    holdout_start = _nonempty_string(
        value.get("holdout_start"), "strength calibration holdout_start"
    )
    try:
        fit_ts = pd.Timestamp(fit_cutoff)
        holdout_ts = pd.Timestamp(holdout_start)
    except (TypeError, ValueError) as exc:
        raise CompositionArtifactError(
            "strength calibration timestamps are invalid"
        ) from exc
    if pd.isna(fit_ts) or pd.isna(holdout_ts) or fit_ts >= holdout_ts:
        raise CompositionArtifactError(
            "strength calibration fit_cutoff must precede holdout_start"
        )
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise CompositionArtifactError("strength calibration source is required")
    _nonempty_string(source.get("artifact"), "strength calibration source artifact")
    digest = _nonempty_string(
        source.get("artifact_sha256"), "strength calibration source SHA"
    )
    if not re.fullmatch(r"[a-f0-9]{64}", digest, flags=re.IGNORECASE):
        raise CompositionArtifactError(
            "strength calibration source SHA must be 64 hexadecimal characters"
        )
    version = source.get("artifact_version")
    if (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < 2
    ):
        raise CompositionArtifactError(
            "strength calibration source version must be at least 2"
        )
    for label, coefficient_names in (
        ("team", ("intercept", "coef")),
        ("player", ("intercept", "coef")),
        ("blend", ("intercept", "coef_team", "coef_player")),
    ):
        block = value.get(label)
        if not isinstance(block, Mapping):
            raise CompositionArtifactError(
                f"strength calibration {label} block is required"
            )
        _nonempty_string(
            block.get("model_id"), f"strength calibration {label} model_id"
        )
        for coefficient_name in coefficient_names:
            _finite_number(
                block.get(coefficient_name),
                f"strength calibration {label}.{coefficient_name}",
            )
    return value


def _validate_strength_calibration_envelope(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompositionArtifactError("strength_calibration is required")
    if value.get("schema_version") != STRENGTH_CALIBRATION_SCHEMA_VERSION:
        raise CompositionArtifactError(
            "strength_calibration.schema_version is unsupported"
        )
    if value.get("status") == "available":
        return _require_strength_calibration(value)
    if value.get("status") != "unavailable":
        raise CompositionArtifactError(
            "strength_calibration.status must be available or unavailable"
        )
    _nonempty_string(value.get("reason"), "strength calibration unavailable reason")
    source = value.get("source")
    if not isinstance(source, Mapping):
        raise CompositionArtifactError("strength calibration source is required")
    _nonempty_string(source.get("artifact"), "strength calibration source artifact")
    digest = source.get("artifact_sha256")
    if digest is not None and (
        not isinstance(digest, str)
        or re.fullmatch(r"[a-f0-9]{64}", digest, flags=re.IGNORECASE) is None
    ):
        raise CompositionArtifactError(
            "unavailable strength calibration source SHA is invalid"
        )
    version = source.get("artifact_version")
    if version is not None and (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < 1
    ):
        raise CompositionArtifactError(
            "unavailable strength calibration source version is invalid"
        )
    for key in ("team", "player", "blend"):
        if key in value:
            raise CompositionArtifactError(
                f"unavailable strength calibration cannot contain {key} coefficients"
            )
    return value


def _require_disabled_low_rank(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CompositionArtifactError("low_rank contract is required")
    if value.get("status") != "disabled" or value.get("rank") != 0:
        raise CompositionArtifactError(
            "bounded/public scores require low_rank.status=disabled and rank=0"
        )
    for key in ("champions", "left", "right"):
        if value.get(key) != []:
            raise CompositionArtifactError(
                f"disabled low_rank.{key} must be empty"
            )
    _nonempty_string(value.get("reason"), "low_rank.reason")
    return value


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
                patch=normalize_patch(
                    raw.get("patch"),
                    allow_source_numeric_minor=True,
                ),
                date=date_value,
            )
        )
    games.sort(key=lambda g: (g.date is None, g.date or pd.Timestamp.min, g.game_id))
    return games


def composition_games_sha256(games: Sequence[CompositionGame]) -> str:
    """Hash the complete ordered estimand population, not source file layout."""
    digest = hashlib.sha256()
    ordered = sorted(
        games,
        key=lambda game: (
            game.date is None,
            game.date or pd.Timestamp.min,
            game.game_id,
        ),
    )
    for game in ordered:
        payload = {
            "game_id": game.game_id,
            "blue": game.blue,
            "red": game.red,
            "y": game.y,
            "league": game.league,
            "patch": game.patch,
            "date": game.date.isoformat() if game.date is not None else None,
        }
        digest.update(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


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
        "intercept_se": float(
            1.0
            / math.sqrt(
                max(float(np.sum(weights * p * (1.0 - p))), 1e-12)
            )
        ),
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
        return _disabled_low_rank()
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


def _calibration(model: Mapping[str, Any], games: Sequence[CompositionGame]) -> dict[str, Any]:
    if not games:
        raise CompositionArtifactError(
            "composition calibration requires a non-empty chronological slice"
        )
    x = np.asarray([_raw_edge(model, game) for game in games], dtype=float)
    y = np.asarray([game.y for game in games], dtype=int)
    calibration = _fit_calibration_curve(x, y)
    dates = [game.date for game in games if game.date is not None]
    calibration.update(
        {
            "schema_version": UNCERTAINTY_SCHEMA_VERSION,
            "fit_n": len(games),
            "fit_start": str(min(dates)) if dates else None,
            "fit_cutoff": str(max(dates)) if dates else None,
        }
    )
    return calibration


def _fit_calibration_curve(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Stable two-parameter calibration fit without high-dimensional solver state."""
    if len(np.unique(y)) < 2:
        raise CompositionArtifactError(
            "composition calibration requires both outcome classes"
        )

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(theta[0] + theta[1] * x, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-eta))
        loss = -float(np.sum(y * np.log(np.clip(p, 1e-12, 1.0)) + (1 - y) * np.log(np.clip(1 - p, 1e-12, 1.0))))
        loss += 1e-6 * float(np.dot(theta, theta))
        grad = np.asarray([np.sum(p - y), np.sum((p - y) * x)], dtype=float) + 2e-6 * theta
        return loss, grad

    result = minimize(lambda theta: objective(theta)[0], np.asarray([0.0, 1.0]), jac=lambda theta: objective(theta)[1], method="BFGS")
    if not result.success or not np.all(np.isfinite(result.x)):
        raise CompositionArtifactError(
            f"composition calibration fit failed: {result.message}"
        )
    theta = result.x
    eta = np.clip(theta[0] + theta[1] * x, -35.0, 35.0)
    p = 1.0 / (1.0 + np.exp(-eta))
    design = np.column_stack([np.ones(len(x), dtype=float), x])
    hessian = design.T @ ((p * (1.0 - p))[:, None] * design)
    hessian += 2e-6 * np.eye(2, dtype=float)
    covariance = np.linalg.pinv(hessian)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise CompositionArtifactError(
            "composition calibration covariance is unavailable"
        )
    return {
        "intercept": float(theta[0]),
        "slope": float(theta[1]),
        "covariance": covariance.tolist(),
    }


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


def _require_composition_calibration(
    value: Any,
) -> tuple[float, float, np.ndarray]:
    if not isinstance(value, Mapping):
        raise CompositionArtifactError("composition calibration is required")
    intercept = _finite_number(
        value.get("intercept"), "composition calibration intercept"
    )
    slope = _finite_number(value.get("slope"), "composition calibration slope")
    covariance = np.asarray(value.get("covariance"), dtype=float)
    if covariance.shape != (2, 2) or not np.all(np.isfinite(covariance)):
        raise CompositionArtifactError(
            "composition calibration covariance must be a finite 2x2 matrix"
        )
    if np.any(np.diag(covariance) < 0):
        raise CompositionArtifactError(
            "composition calibration covariance diagonal must be non-negative"
        )
    return intercept, slope, covariance


def _feature_coefficient(spec: Any, key: str) -> tuple[float, float]:
    if not isinstance(spec, Mapping):
        raise CompositionArtifactError(f"feature {key} must be an object")
    coefficient = _finite_number(spec.get("coef"), f"feature {key}.coef")
    standard_error = _finite_number(
        spec.get("se"), f"feature {key}.se", non_negative=True
    )
    return coefficient, standard_error


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
    _require_disabled_low_rank(model.get("low_rank"))
    _finite_number(
        model.get("intercept_se"), "composition intercept_se", non_negative=True
    )
    cal_intercept, cal_slope, cal_covariance = (
        _require_composition_calibration(model.get("calibration"))
    )
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
                    coefficient, standard_error = _feature_coefficient(
                        specs[key], key
                    )
                    direct += sign * coefficient
                    direct_var += standard_error**2
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
                    key = f"synergy|{a}|{c}"
                    spec = specs.get(key)
                    if not spec:
                        continue
                    coefficient, standard_error = _feature_coefficient(spec, key)
                    value = sign * coefficient
                    synergy_logit += value
                    share = value / 2.0
                    for role, champ in (picks[i], picks[j]):
                        row = by_side_index[(side_name, champ, role)]
                        row["team_synergy"] += share
                        row["edge_contribution"] += share
                        row["uncertainty_logit"] = math.sqrt(
                            float(row["uncertainty_logit"]) ** 2
                            + (standard_error / 2.0) ** 2
                        )
                    edge_var += standard_error**2

    for _role_b, blue_champ in b:
        for _role_r, red_champ in r:
            if "opposition" in components:
                key, orientation = _opposition_key(blue_champ, red_champ)
                spec = specs.get(key)
                if spec and orientation:
                    coefficient, standard_error = _feature_coefficient(spec, key)
                    value = float(orientation) * coefficient
                    opposition_logit += value
                    edge_var += standard_error**2
                    blue_row = next(row for row in champion_parts if row["side"] == "blue" and row["champion"] == blue_champ and row["role"] == next(role for role, champ in b if champ == blue_champ))
                    red_row = next(row for row in champion_parts if row["side"] == "red" and row["champion"] == red_champ and row["role"] == next(role for role, champ in r if champ == red_champ))
                    blue_row["enemy_interaction"] += value / 2.0
                    red_row["enemy_interaction"] += value / 2.0
                    blue_row["edge_contribution"] += value / 2.0
                    red_row["edge_contribution"] += value / 2.0
                    blue_row["uncertainty_logit"] = math.sqrt(
                        float(blue_row["uncertainty_logit"]) ** 2
                        + (standard_error / 2.0) ** 2
                    )
                    red_row["uncertainty_logit"] = math.sqrt(
                        float(red_row["uncertainty_logit"]) ** 2
                        + (standard_error / 2.0) ** 2
                    )

    composition_edge = main_logit + synergy_logit + opposition_logit + low_rank_logit
    side_advantage = _finite_number(
        model.get("intercept"), "composition intercept"
    )
    model_edge = side_advantage + composition_edge
    # The calibrated estimand is the probability of a blue-side map win.  The
    # calibration fit was trained on the model's complete linear predictor, so
    # both the fitted side intercept and calibration intercept belong here.
    # Composition antisymmetry is preserved in `composition_edge`; a blue-side
    # probability is not generally complementary under a composition-only swap
    # because the side baseline remains attached to blue.
    calibrated_logit = cal_intercept + cal_slope * model_edge
    intercept_se = _finite_number(
        model.get("intercept_se"), "composition intercept_se", non_negative=True
    )
    model_edge_var = edge_var + intercept_se**2
    calibration_parameter_var = (
        float(cal_covariance[0, 0])
        + 2.0 * model_edge * float(cal_covariance[0, 1])
        + model_edge**2 * float(cal_covariance[1, 1])
    )
    calibrated_var = (
        cal_slope**2 * model_edge_var
        + calibration_parameter_var
    )
    edge_se = math.sqrt(max(calibrated_var, 1e-12))
    p_blue = _sigmoid(calibrated_logit)
    p_lo = _sigmoid(calibrated_logit - 1.96 * edge_se)
    p_hi = _sigmoid(calibrated_logit + 1.96 * edge_se)
    neutral_blue = _sigmoid(cal_intercept + cal_slope * side_advantage)
    team_diff = team_elo_diff if team_elo_diff is not None else elo_diff
    player_diff = player_elo_diff
    strength = (
        _require_strength_calibration(model.get("strength_calibration"))
        if team_diff is not None or player_diff is not None
        else None
    )
    team_block = strength.get("team") if strength is not None else None
    player_block = strength.get("player") if strength is not None else None
    blend_block = strength.get("blend") if strength is not None else None
    team_p = None
    if team_diff is not None:
        assert isinstance(team_block, Mapping)
        team_p = _sigmoid(
            _finite_number(team_block.get("intercept"), "team.intercept")
            + _finite_number(team_block.get("coef"), "team.coef")
            * float(team_diff)
            / 400.0
        )
    player_p = None
    if player_diff is not None:
        assert isinstance(player_block, Mapping)
        player_p = _sigmoid(
            _finite_number(player_block.get("intercept"), "player.intercept")
            + _finite_number(player_block.get("coef"), "player.coef")
            * float(player_diff)
            / 400.0
        )
    strength_logit = None
    if team_p is not None and player_p is not None:
        assert isinstance(blend_block, Mapping)
        strength_logit = (
            _finite_number(blend_block.get("intercept"), "blend.intercept")
            + _finite_number(blend_block.get("coef_team"), "blend.coef_team")
            * float(team_p)
            + _finite_number(
                blend_block.get("coef_player"), "blend.coef_player"
            )
            * float(player_p)
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
        "wr_bump_pp": round(100.0 * (p_blue - neutral_blue), 2),
        "posterior_width": round(edge_se, 4),
        "uncertainty": {
            "edge_se_logit": round(edge_se, 4),
            "p_blue_95": [round(p_lo, 4), round(p_hi, 4)],
            "method": (
                "delta-method interval from explicit composition-term diagonal "
                "Laplace variance, model-intercept variance, and the full "
                "chronological calibration covariance; low-rank is disabled"
            ),
        },
        "calibration": {
            "league": league,
            "patch": patch,
            "source": _nonempty_string(
                model.get("calibration_source"), "calibration_source"
            ),
            "intercept": round(cal_intercept, 4),
            "slope": round(cal_slope, 4),
            "neutral_blue_baseline": round(neutral_blue, 4),
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
        "note": (
            "Full-composition draft model: role-aware direct effects, "
            "within-team synergy, and all 25 explicit enemy interactions. "
            "Low-rank residuals are disabled until their uncertainty is "
            "estimable; strength is reported separately when supplied."
        ),
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
) -> dict[str, float]:
    cal_train, cal = _split_time(train, 0.8) if len(train) > 20 else (list(train), [])
    if cal:
        # Fit, calibrate, and evaluate one frozen model.  Calibrating a model
        # fitted on `cal_train` and then evaluating different coefficients
        # fitted on all of `train` invalidates the calibration contract.
        model = _fit_logistic(cal_train, components)
        model["low_rank"] = _disabled_low_rank()
        model["calibration"] = _calibration(model, cal)
        model["calibration_source"] = "chronological calibration slice"
    else:
        raise CompositionArtifactError(
            "holdout evaluation requires a chronological calibration slice"
        )
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
    if low_rank_rank != 0:
        raise CompositionArtifactError(
            "bounded/public composition artifacts require low_rank_rank=0 "
            "until low-rank uncertainty is estimated"
        )
    train, test = _split_time(games, 0.8)
    cal_train, calibration_games = _split_time(train, 0.8)
    model = _fit_logistic(cal_train, ("main", "synergy", "opposition"), min_support=min_support)
    model["low_rank"] = _disabled_low_rank()
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
            validation["future_patch_holdout"] = _fit_holdout(
                patch_train,
                patch_test,
                ("main", "synergy", "opposition"),
            )
            validation["future_patch"] = sorted(future)
        leagues = sorted({g.league for g in games})
        league = max(leagues, key=lambda x: sum(g.league == x for g in games))
        league_train = [g for g in games if g.league != league]
        league_test = [g for g in games if g.league == league]
        if league_train and league_test:
            validation["league_holdout"] = _fit_holdout(
                league_train,
                league_test,
                ("main", "synergy", "opposition"),
            )
            validation["league"] = league
        ablations = {
            "additive_only": ("main",),
            "plus_synergy": ("main", "synergy"),
            "plus_opposition": ("main", "synergy", "opposition"),
        }
        validation["ablations_time_holdout"] = {
            name: _fit_holdout(cal_train, test, components)
            for name, components in ablations.items()
        }
        validation["low_rank"] = {
            "status": "disabled",
            "reason": (
                "no bounded/public evaluation until low-rank fit and "
                "prediction uncertainty are estimated"
            ),
        }
    model.update(
        {
            "version": RUNTIME_VERSION,
            "model_code_sha256": model_code_sha256(),
            "training_population_sha256": composition_games_sha256(games),
            "numerical_environment": numerical_environment(),
            "estimand": "pre-match blue-side map-win probability conditional on champion composition, role, league, and patch; no roster/player/team strength in pure draft edge",
            "n_games_total": len(games),
            "n_games_fit": len(cal_train),
            "date_min": min((g.date for g in games if g.date is not None), default=None),
            "date_max": max((g.date for g in games if g.date is not None), default=None),
            "validation": validation,
            "uncertainty": {
                "schema_version": UNCERTAINTY_SCHEMA_VERSION,
                "method": (
                    "delta method from explicit composition-term diagonal "
                    "Laplace variance, model-intercept variance, and the full "
                    "chronological calibration covariance"
                ),
                "active_terms": [
                    "main",
                    "league",
                    "patch",
                    "synergy",
                    "opposition",
                    "model_intercept",
                    "calibration_intercept",
                    "calibration_slope",
                    "calibration_intercept_slope_covariance",
                ],
                "low_rank_status": "disabled",
            },
            "limitations": [
                "observational draft data cannot identify causal champion effects",
                "role/league/patch deviations are ridge-pooled and should be treated as estimates, not matchup truths",
                "explicit feature uncertainty is a diagonal Laplace approximation and omits feature-coefficient covariance",
                "low-rank residuals are disabled because their fit and prediction uncertainty are not estimated",
            ],
        }
    )
    return model


def export_runtime(
    model: Mapping[str, Any],
    path: Path = RUNTIME_PATH,
    *,
    artifact_sha256: str | None = None,
) -> dict[str, Any]:
    """Write the browser-sized artifact with its public evidence metadata."""
    if model.get("version") != RUNTIME_VERSION:
        raise CompositionArtifactError(
            f"composition artifact version must equal {RUNTIME_VERSION}"
        )
    _finite_number(model.get("intercept"), "composition intercept")
    _finite_number(
        model.get("intercept_se"), "composition intercept_se", non_negative=True
    )
    _require_disabled_low_rank(model.get("low_rank"))
    _require_composition_calibration(model.get("calibration"))
    _validate_strength_calibration_envelope(model.get("strength_calibration"))
    for key in ("model_code_sha256", "training_population_sha256"):
        digest = model.get(key)
        if not isinstance(digest, str) or re.fullmatch(
            r"[a-f0-9]{64}", digest, flags=re.IGNORECASE
        ) is None:
            raise CompositionArtifactError(
                f"{key} must be a 64-character hexadecimal digest"
            )
    environment = model.get("numerical_environment")
    packages = environment.get("packages") if isinstance(environment, Mapping) else None
    if (
        not isinstance(environment, Mapping)
        or not isinstance(environment.get("python"), str)
        or not environment["python"].strip()
        or not isinstance(packages, Mapping)
        or any(
            not isinstance(packages.get(name), str) or not packages[name].strip()
            for name in ("numpy", "pandas", "scipy", "scikit-learn")
        )
    ):
        raise CompositionArtifactError(
            "numerical_environment must pin Python and numerical-library versions"
        )
    if not isinstance(artifact_sha256, str) or re.fullmatch(
        r"[a-f0-9]{64}", artifact_sha256, flags=re.IGNORECASE
    ) is None:
        raise CompositionArtifactError(
            "artifact_sha256 must be a 64-character hexadecimal digest"
        )
    validation = model.get("validation")
    if not isinstance(validation, Mapping) or not isinstance(
        validation.get("time_holdout"), Mapping
    ):
        raise CompositionArtifactError(
            "validated runtime requires validation.time_holdout"
        )
    uncertainty = model.get("uncertainty")
    if (
        not isinstance(uncertainty, Mapping)
        or uncertainty.get("schema_version") != UNCERTAINTY_SCHEMA_VERSION
        or uncertainty.get("low_rank_status") != "disabled"
    ):
        raise CompositionArtifactError(
            "versioned uncertainty metadata with disabled low rank is required"
        )
    runtime = {
        "version": model["version"],
        "model_code_sha256": model["model_code_sha256"],
        "training_population_sha256": model["training_population_sha256"],
        "numerical_environment": model["numerical_environment"],
        "estimand": _nonempty_string(model.get("estimand"), "estimand"),
        "intercept": model["intercept"],
        "intercept_se": model["intercept_se"],
        "feature_specs": model["feature_specs"],
        "role_champion_counts": model["role_champion_counts"],
        "components": model["components"],
        "prior_n": model["prior_n"],
        "low_rank": model["low_rank"],
        "calibration": model["calibration"],
        "calibration_source": _nonempty_string(
            model.get("calibration_source"), "calibration_source"
        ),
        "strength_calibration": model["strength_calibration"],
        "n_games_fit": model["n_games_fit"],
        "n_games_total": model["n_games_total"],
        "date_min": str(model.get("date_min")) if model.get("date_min") is not None else None,
        "date_max": str(model.get("date_max")) if model.get("date_max") is not None else None,
        "min_support": model["min_support"],
        "recency_half_life_days": model["recency_half_life_days"],
        "validation": validation,
        "uncertainty": uncertainty,
        "limitations": model["limitations"],
        "artifact_sha256": artifact_sha256,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return runtime


def export_packed_runtime(
    runtime_path: Path,
    packed_path: Path = PACKED_RUNTIME_PATH,
) -> None:
    """Write the deterministic browser bundle consumed by the Next runtime."""

    encoded = base64.b64encode(
        gzip.compress(runtime_path.read_bytes(), mtime=0)
    ).decode("ascii")
    packed_path.parent.mkdir(parents=True, exist_ok=True)
    packed_path.write_text(encoded + "\n", encoding="ascii")


def fit_from_paths(
    maps_path: Path | None = None,
    players_path: Path | None = None,
    *,
    output: Path = MODEL_PATH,
    runtime: Path = RUNTIME_PATH,
    packed_runtime: Path | None = None,
    low_rank_rank: int = DEFAULT_LOW_RANK,
    validate: bool = True,
) -> dict[str, Any]:
    maps, players = load_training_frames(maps_path, players_path)
    games = build_games(maps, players)
    model = fit_composition_artifact(games, low_rank_rank=low_rank_rank, validate=validate)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(model, indent=2, default=str, sort_keys=True) + "\n", encoding="utf-8")
    artifact_sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
    export_runtime(model, runtime, artifact_sha256=artifact_sha256)
    if packed_runtime is not None:
        export_packed_runtime(runtime, packed_runtime)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=None)
    parser.add_argument("--players", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=MODEL_PATH)
    parser.add_argument("--runtime", type=Path, default=RUNTIME_PATH)
    parser.add_argument(
        "--packed-runtime",
        type=Path,
        default=PACKED_RUNTIME_PATH,
    )
    parser.add_argument("--low-rank", type=int, default=DEFAULT_LOW_RANK)
    parser.add_argument("--no-validate", action="store_true")
    args = parser.parse_args()
    model = fit_from_paths(
        args.maps,
        args.players,
        output=args.output,
        runtime=args.runtime,
        packed_runtime=args.packed_runtime,
        low_rank_rank=args.low_rank,
        validate=not args.no_validate,
    )
    print(json.dumps({"n_games": model["n_games_total"], "fit": model["n_games_fit"], "validation": model.get("validation", {})}, indent=2, default=str))


if __name__ == "__main__":
    main()
