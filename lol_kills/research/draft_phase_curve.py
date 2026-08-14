"""Leakage-safe pre-match phase forecasts for the private Draft Score.

The static Draft Score and the phase curve answer different questions.  The
static score describes the composition before a game starts.  This module
forecasts later gold states from information that exists before the game.  It
does not read observed gold while it builds those forecasts.

The live model is a separate optional boundary.  It may use an observed
timeline with gold, objectives, and the current clock.  A phase-curve artifact
is unavailable until the accepted 16.16 source rows, the frozen evaluation,
and a receipt-bound promotion are present.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from lol_kills.v2.patch_identity import PatchIdentityError, client_patch, public_patch


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "data" / "lol" / "models"
PHASES = ("10", "15", "20", "25")
SCHEMA_VERSION = "scryglass:draft-phase-curve:v1"
MODEL_VERSION = "draft-phase-curve-v1"
BASELINE_AUC_FLOOR = 0.70681
REQUIRED_CLIENT_PATCH = "16.16"
REQUIRED_PUBLIC_PATCH = "26.16"
RAW_OE_SOURCE_TOKEN = "16.15"
PHASE_CURVE_PATH = MODELS_DIR / "draft_phase_curve.json"
ATOM_BRIDGE_PATH = ROOT / "data" / "lol" / "v2" / "champions" / "lcc-atom-bridge-26.16.json"
ATOM_RECEIPT_PATH = ROOT / "data" / "lol" / "v2" / "champions" / "lcc-atom-refresh-26.16-receipt.json"

_SOURCE_PATCH_RE = re.compile(r"^(?:15|16)\.\d{1,2}(?:\.\d+)?$")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
PROMOTION_SCHEMA_VERSION = "scryglass:draft-phase-curve-promotion:v1"
PROMOTION_ROOT = MODELS_DIR / "promotions"
REQUIRED_PROMOTION_GATES = (
    "chronological",
    "regional",
    "patch_transfer",
    "roster_change",
    "sparse_data",
    "missingness",
    "early_snowball",
    "late_scaling",
    "comeback_behind_10",
    "comeback_behind_15",
)
_NUMERIC_FEATURES = (
    "draft_win_logit_blue",
    "elo_diff",
    "player_elo_diff",
    "team_form_diff",
    "roster_change",
)
_CATEGORICAL_FEATURES = ("league", "region", "oe_patch_token", "tier")
_FORBIDDEN_EXACT = frozenset(
    {
        "gold",
        "goldat10",
        "goldat15",
        "goldat20",
        "goldat25",
        "goldat30",
        "gold_diff_10",
        "gold_diff_15",
        "gold_diff_20",
        "gold_diff_25",
        "golddiffat10",
        "golddiffat15",
        "golddiffat20",
        "golddiffat25",
        "totalgold",
        "earnedgold",
        "xpa10",
        "xpa15",
        "xpa20",
        "xpa25",
        "csa10",
        "csa15",
        "csa20",
        "csa25",
        "killsat10",
        "killsat15",
        "killsat20",
        "killsat25",
        "assistsat10",
        "assistsat15",
        "assistsat20",
        "assistsat25",
        "deathsat10",
        "deathsat15",
        "deathsat20",
        "deathsat25",
        "y_blue_win",
        "result",
        "final_result",
    }
)
_FORBIDDEN_FRAGMENTS = (
    "gold_diff",
    "golddiff",
    "goldat",
    "xpat",
    "xpdiff",
    "csat",
    "csdiff",
    "killsat",
    "assistsat",
    "deathsat",
    "totalgold",
    "earnedgold",
    "objective",
    "first_dragon",
    "first_herald",
    "first_tower",
    "baron",
    "match_result",
    "game_result",
)
_BRIDGE_CACHE: dict[str, Mapping[str, Any]] | None = None


class PhaseCurveUnavailable(RuntimeError):
    """Raised when a candidate cannot be fit under the phase contract."""


def _empty_values() -> dict[str, None]:
    return {phase: None for phase in PHASES}


def lcc_atomization_metadata() -> dict[str, Any]:
    """Return the non-authorizing LCC bridge metadata used by this module."""

    bridge_sha256 = None
    receipt_sha256 = None
    try:
        bridge = json.loads(ATOM_BRIDGE_PATH.read_text(encoding="utf-8"))
        bridge_sha256 = bridge.get("artifact_sha256")
    except (OSError, json.JSONDecodeError, TypeError):
        bridge_sha256 = None
    try:
        receipt = json.loads(ATOM_RECEIPT_PATH.read_text(encoding="utf-8"))
        receipt_sha256 = receipt.get("atom_bridge_artifact_sha256")
    except (OSError, json.JSONDecodeError, TypeError):
        receipt_sha256 = None
    return {
        "status": "staged",
        "authority": "development_only",
        "public_patch": REQUIRED_PUBLIC_PATCH,
        "client_patch": REQUIRED_CLIENT_PATCH,
        "bridge": "data/lol/v2/champions/lcc-atom-bridge-26.16.json",
        "depths": [2, 3, 4],
        "auc_reference": BASELINE_AUC_FLOOR,
        "bridge_sha256": bridge_sha256,
        "receipt_bridge_sha256": receipt_sha256,
    }


def unavailable_phase_curve(
    *,
    source: str = "oe_only",
    blockers: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the public-safe unavailable contract.

    The function intentionally emits no numeric phase result.  Extra metadata
    helps research and diagnostics explain why the result is withheld.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "unavailable",
        "source": source,
        "window": list(PHASES),
        "expected_gold_diff": _empty_values(),
        "phase_draft_edge": _empty_values(),
        "scaling_index": None,
        "snowball_index": None,
        "comeback_resilience": None,
        "sample_window": None,
        "model_version": None,
        "lcc_atomization": lcc_atomization_metadata(),
        "blockers": list(dict.fromkeys(str(item) for item in blockers)),
    }


def _normalise_name(value: Any) -> str:
    return str(value or "").strip().casefold().replace("-", "_")


def _is_forbidden_feature(name: Any) -> bool:
    value = _normalise_name(name)
    if value in _FORBIDDEN_EXACT:
        return True
    return any(fragment in value for fragment in _FORBIDDEN_FRAGMENTS)


def assert_no_gold_leakage(feature_names: Sequence[str]) -> None:
    """Reject post-draft mediator or terminal fields in pre-match features."""

    forbidden = [str(name) for name in feature_names if _is_forbidden_feature(name)]
    if forbidden:
        raise ValueError(
            "pre-match phase features contain post-game or observed-state fields: "
            + ", ".join(sorted(forbidden))
        )


def _read_frame(value: Any) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    path = Path(value)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".json", ".jsonl"}:
        return pd.read_json(path, lines=path.suffix.lower() == ".jsonl")
    raise TypeError(f"unsupported phase source: {value!r}")


def _first_value(row: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in row and pd.notna(row[name]):
            return row[name]
    return None


def _game_column(frame: pd.DataFrame) -> str:
    for name in ("game_uid", "gameid", "game_id", "gameId"):
        if name in frame.columns:
            return name
    raise ValueError("phase source has no game identifier")


def _source_patch(row: Mapping[str, Any]) -> str | None:
    # ``patch`` is the derived, realm-aware token. ``oe_patch_token`` is the
    # raw export token and is the fallback for older frames.
    for name in ("patch", "derived_patch", "client_patch", "game_patch", "oe_patch_token", "source_patch"):
        value = row.get(name)
        if value is None or pd.isna(value):
            continue
        token = str(value).strip()
        if _SOURCE_PATCH_RE.fullmatch(token):
            return token
    return None


def _target_value(
    row: Mapping[str, Any],
    phase: str,
    *,
    opponent: Mapping[str, Any] | None = None,
    map_row: Mapping[str, Any] | None = None,
) -> float | None:
    direct_names = (
        f"golddiffat{phase}",
        f"gold_diff_{phase}",
        f"gold_diffat{phase}",
        f"goldDiffAt{phase}",
    )
    value = _first_value(row, direct_names)
    if value is not None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = math.nan
        if math.isfinite(number):
            return number

    gold_names = (f"goldat{phase}", f"gold_at_{phase}", f"goldAt{phase}")
    blue_gold = _first_value(row, (f"blue_{name}" for name in gold_names))
    red_gold = _first_value(row, (f"red_{name}" for name in gold_names))
    if blue_gold is not None and red_gold is not None:
        try:
            return float(blue_gold) - float(red_gold)
        except (TypeError, ValueError):
            return None

    own_gold = _first_value(row, gold_names)
    other_gold = _first_value(opponent or {}, gold_names)
    if own_gold is not None and other_gold is not None:
        try:
            return float(own_gold) - float(other_gold)
        except (TypeError, ValueError):
            return None

    map_row = map_row or {}
    blue_gold = _first_value(map_row, (f"blue_{name}" for name in gold_names))
    red_gold = _first_value(map_row, (f"red_{name}" for name in gold_names))
    if blue_gold is not None and red_gold is not None:
        try:
            return float(blue_gold) - float(red_gold)
        except (TypeError, ValueError):
            return None
    return None


def _row_map(frame: pd.DataFrame, key: str) -> dict[str, Mapping[str, Any]]:
    if key not in frame.columns:
        return {}
    return {
        str(row[key]): row.to_dict()
        for _, row in frame.iterrows()
        if pd.notna(row[key])
    }


def prepare_phase_frame(
    team_games: Any,
    maps: Any,
    *,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a blue-side target frame from the OE team and map tables.

    The target columns are observed gold differences.  They are never copied
    into the pre-match design matrix.  Missing @25 rows stay missing because
    they are censored observations.
    """

    teams = _read_frame(team_games)
    map_frame = _read_frame(maps)
    team_key = _game_column(teams)
    map_key = _game_column(map_frame)
    teams = teams.copy()
    map_frame = map_frame.copy()
    teams["_game_key"] = teams[team_key].astype(str)
    map_frame["_game_key"] = map_frame[map_key].astype(str)

    side_col = next(
        (name for name in ("side", "teamcolor", "team_color") if name in teams.columns),
        None,
    )
    if side_col is None:
        blue = teams
        red = pd.DataFrame(columns=teams.columns)
    else:
        sides = teams[side_col].astype(str).str.casefold()
        blue = teams[sides.isin({"blue", "b"})]
        red = teams[sides.isin({"red", "r"})]
    if blue.empty:
        raise ValueError("phase source has no blue-side team rows")

    red_by_game: dict[str, Mapping[str, Any]] = {}
    if not red.empty:
        red_by_game = {
            str(row["_game_key"]): row.to_dict()
            for _, row in red.iterrows()
        }
    maps_by_game = _row_map(map_frame, "_game_key")
    feature_by_game: dict[str, Mapping[str, Any]] = {}
    if features is not None:
        feature_frame = _read_frame(features)
        feature_key = _game_column(feature_frame)
        feature_by_game = {
            str(row[feature_key]): row.to_dict()
            for _, row in feature_frame.iterrows()
            if pd.notna(row[feature_key])
        }

    rows: list[dict[str, Any]] = []
    for _, source_row in blue.iterrows():
        row = source_row.to_dict()
        game_key = str(row["_game_key"])
        map_row = maps_by_game.get(game_key, {})
        extra = feature_by_game.get(game_key, {})
        merged: dict[str, Any] = {
            "game_uid": game_key,
            "date": _first_value(
                map_row,
                ("date", "game_date", "played_at", "start_time"),
            )
            or _first_value(row, ("date", "game_date", "played_at", "start_time")),
            "league": _first_value(map_row, ("league", "league_name"))
            or _first_value(row, ("league", "league_name")),
            "region": _first_value(
                map_row,
                ("region", "competition_scope", "league_source", "region_name"),
            )
            or _first_value(
                row,
                ("region", "competition_scope", "league_source", "region_name"),
            ),
            # Keep the source token in provenance. The model token below is
            # derived from the realm-aware patch when it is present.
            "oe_source_token": _source_patch(
                {"oe_patch_token": row.get("oe_patch_token")}
            )
            or _source_patch({"oe_patch_token": map_row.get("oe_patch_token")}),
            "y_blue_win": _first_value(map_row, ("y_blue_win", "blue_win", "result"))
            if map_row
            else _first_value(row, ("y_blue_win", "blue_win", "result")),
        }
        derived_patch = _source_patch(row) or _source_patch(map_row)
        if derived_patch is None:
            derived_patch = merged["oe_source_token"]
        merged["oe_patch_token"] = derived_patch
        merged["patch"] = derived_patch
        if merged["oe_patch_token"] is not None:
            try:
                merged["public_patch"] = public_patch(merged["oe_patch_token"])
            except PatchIdentityError:
                merged["public_patch"] = None
        for name in _NUMERIC_FEATURES + ("tier",):
            if name in extra:
                merged[name] = extra[name]
            elif name in row:
                merged[name] = row[name]
        for name, value in {**row, **extra}.items():
            if str(name).startswith(("lcc_atom_", "atom_")):
                merged[str(name)] = value
        opponent = red_by_game.get(game_key)
        for phase in PHASES:
            merged[f"gold_diff_{phase}"] = _target_value(
                row,
                phase,
                opponent=opponent,
                map_row=map_row,
            )
        rows.append(merged)
    return pd.DataFrame(rows)


def _fit_patch_token(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    token = str(value).strip()
    if not _SOURCE_PATCH_RE.fullmatch(token):
        raise PhaseCurveUnavailable(
            f"phase fit requires client/OE patch tokens such as 15.16 or 16.16, got {token!r}"
        )
    try:
        canonical = client_patch(token)
    except PatchIdentityError as exc:
        raise PhaseCurveUnavailable(f"invalid OE patch token: {token!r}") from exc
    if not canonical.startswith(("15.", "16.")):
        raise PhaseCurveUnavailable(f"source patch was rewritten or is not a client token: {token!r}")
    return token


def _normalise_fit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    value = frame.copy()
    if "patch" in value.columns:
        derived = value["patch"]
        if "oe_patch_token" in value.columns:
            derived = derived.where(derived.notna(), value["oe_patch_token"])
        value["oe_patch_token"] = derived
    elif "oe_patch_token" not in value.columns:
        for name in ("source_patch", "client_patch", "patch"):
            if name in value.columns:
                value["oe_patch_token"] = value[name]
                break
    if "game_uid" not in value.columns:
        for name in ("gameid", "game_id"):
            if name in value.columns:
                value["game_uid"] = value[name].astype(str)
                break
    if "date" not in value.columns:
        for name in ("played_at", "game_date", "start_time"):
            if name in value.columns:
                value["date"] = value[name]
                break
    if "game_uid" not in value.columns:
        value["game_uid"] = [str(index) for index in range(len(value))]
    return value


def build_pre_match_design(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Create a deterministic design matrix from pre-match fields only."""

    value = _normalise_fit_frame(frame)
    names: list[str] = []
    columns: list[np.ndarray] = []
    candidate_names = [
        name for name in (*_NUMERIC_FEATURES, *_CATEGORICAL_FEATURES)
        if name in value.columns
    ]
    candidate_names.extend(
        str(name)
        for name in value.columns
        if str(name).startswith(("lcc_atom_", "atom_"))
    )
    assert_no_gold_leakage(candidate_names)
    for name in _NUMERIC_FEATURES:
        if name not in value.columns:
            continue
        numeric = pd.to_numeric(value[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        names.append(f"num:{name}")
        columns.append(numeric)
    atom_names = sorted(
        str(name)
        for name in value.columns
        if str(name).startswith(("lcc_atom_", "atom_"))
    )
    assert_no_gold_leakage(atom_names)
    for name in atom_names:
        numeric = pd.to_numeric(value[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        names.append(f"num:{name}")
        columns.append(numeric)
    for name in _CATEGORICAL_FEATURES:
        if name not in value.columns:
            continue
        categories = sorted(
            {
                str(item).strip()
                for item in value[name].tolist()
                if pd.notna(item) and str(item).strip()
            }
        )
        for category in categories:
            names.append(f"cat:{name}={category}")
            columns.append(
                np.asarray(
                    [1.0 if str(item).strip() == category else 0.0 for item in value[name]],
                    dtype=float,
                )
            )
    if not columns:
        return np.zeros((len(value), 0), dtype=float), []
    return np.column_stack(columns), names


def _pick_names(side: Sequence[Any] | Mapping[str, Any]) -> list[str]:
    if isinstance(side, Mapping):
        roles = ("top", "jng", "mid", "bot", "sup", "jungle", "adc", "support")
        values = [side.get(role) for role in roles if role in side]
    else:
        values = list(side)
    result: list[str] = []
    for value in values:
        if isinstance(value, Mapping):
            value = value.get("champion") or value.get("name")
        if value:
            result.append(str(value))
    return result


def _bridge_slug(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").casefold())


def _atom_bridge() -> dict[str, Mapping[str, Any]]:
    """Load the patch-26.16 LCC bridge by display-name slug."""

    global _BRIDGE_CACHE
    if _BRIDGE_CACHE is not None:
        return _BRIDGE_CACHE
    try:
        payload = json.loads(ATOM_BRIDGE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        _BRIDGE_CACHE = {}
        return _BRIDGE_CACHE
    rows = payload.get("champions") if isinstance(payload, Mapping) else None
    result: dict[str, Mapping[str, Any]] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            for key in (row.get("display_name"), row.get("lcc_key")):
                if key:
                    result[_bridge_slug(key)] = row
    _BRIDGE_CACHE = result
    return result


def _bridge_champion_features(champion: str) -> dict[str, float]:
    row = _atom_bridge().get(_bridge_slug(champion))
    if row is None:
        return {}
    values: dict[str, float] = {}
    family_counts = row.get("atom_family_counts")
    if isinstance(family_counts, Mapping):
        for key, value in family_counts.items():
            try:
                values[f"lcc_atom_bridge_family_{key}"] = float(value) / 25.0
            except (TypeError, ValueError):
                continue
    attributes = row.get("lcc_attribute_ratings")
    if isinstance(attributes, Mapping):
        for key, value in attributes.items():
            try:
                values[f"lcc_atom_bridge_attribute_{key}"] = float(value) / 100.0
            except (TypeError, ValueError):
                continue
    ontology = row.get("ontology_prior")
    if isinstance(ontology, Mapping):
        for dimension, entry in ontology.items():
            labels = entry.get("labels") if isinstance(entry, Mapping) else None
            if not isinstance(labels, Mapping):
                continue
            for label, value in labels.items():
                try:
                    values[f"lcc_atom_bridge_{dimension}_{label}"] = float(value)
                except (TypeError, ValueError):
                    continue
    return values


def lcc_atom_profile_for_champion(champion: str) -> dict[str, float]:
    """Expose the bridge profile used to place players and teams in atom space."""

    return _bridge_champion_features(champion)


def atomized_draft_features(
    blue: Sequence[Any] | Mapping[str, Any],
    red: Sequence[Any] | Mapping[str, Any],
) -> dict[str, float]:
    """Return blue-minus-red LCC atom descriptor features for a draft."""

    try:
        from lol_kills.research.composition_signal import _atom_desc_value, _atom_term_keys

        keys = tuple(_atom_term_keys())
    except Exception:
        return {}
    blue_names = _pick_names(blue)
    red_names = _pick_names(red)
    if not blue_names or not red_names:
        return {}
    result: dict[str, float] = {}
    for key in keys:
        blue_value = float(np.mean([_atom_desc_value(name, key) for name in blue_names]))
        red_value = float(np.mean([_atom_desc_value(name, key) for name in red_names]))
        value = blue_value - red_value
        if math.isfinite(value) and abs(value) > 1e-12:
            result[f"lcc_atom_{key}"] = value
    bridge_keys = sorted(
        {
            key
            for name in (*blue_names, *red_names)
            for key in _bridge_champion_features(name)
        }
    )
    for key in bridge_keys:
        blue_value = float(
            np.mean([_bridge_champion_features(name).get(key, 0.0) for name in blue_names])
        )
        red_value = float(
            np.mean([_bridge_champion_features(name).get(key, 0.0) for name in red_names])
        )
        value = blue_value - red_value
        if math.isfinite(value) and abs(value) > 1e-12:
            result[key] = value
    return result


def pre_match_features_for_draft(
    blue: Sequence[Any] | Mapping[str, Any],
    red: Sequence[Any] | Mapping[str, Any],
    draft_logit: float,
    *,
    elo_diff: float | None = None,
    player_elo_diff: float | None = None,
    team_form_diff: float | None = None,
    league: str | None = None,
    region: str | None = None,
    patch: str | None = None,
    roster_change: float | None = None,
    extra_features: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build safe pre-match inputs and attach LCC descriptors."""

    result: dict[str, Any] = {
        "draft_win_logit_blue": float(draft_logit),
        "elo_diff": elo_diff,
        "player_elo_diff": player_elo_diff,
        "team_form_diff": team_form_diff,
        "roster_change": roster_change,
        "league": league,
        "region": region,
        "oe_patch_token": patch,
    }
    result.update(atomized_draft_features(blue, red))
    if extra_features:
        assert_no_gold_leakage(list(extra_features.keys()))
        explicit_patch = extra_features.get("patch")
        if explicit_patch is not None:
            result["oe_patch_token"] = explicit_patch
        for name, value in extra_features.items():
            if name == "oe_patch_token" and explicit_patch is not None:
                continue
            if name in {"league", "region", "oe_patch_token", "tier"}:
                result[name] = value
            elif name == "patch":
                result["oe_patch_token"] = value
            elif name in _NUMERIC_FEATURES or str(name).startswith(("lcc_atom_", "atom_")):
                result[name] = value
    assert_no_gold_leakage(list(result.keys()))
    return result


def _model_source_snapshot(frame: pd.DataFrame) -> dict[str, Any]:
    date_values = frame["date"] if "date" in frame else pd.Series(dtype=object)
    dates = pd.to_datetime(date_values, errors="coerce", utc=True).dropna()
    patch_values = frame["oe_patch_token"] if "oe_patch_token" in frame else pd.Series(dtype=object)
    region_values = frame["region"] if "region" in frame else pd.Series(dtype=object)
    patches = sorted({str(value) for value in patch_values if pd.notna(value)})
    regions = sorted({str(value) for value in region_values if pd.notna(value)})
    return {
        "oe_source_token": RAW_OE_SOURCE_TOKEN,
        "derived_client_patches": sorted(
            {str(value) for value in frame.get("oe_patch_token", ()) if pd.notna(value)}
        ),
        "required_client_patch": REQUIRED_CLIENT_PATCH,
        "source_patches": patches,
        "regions": regions,
        "date_min": dates.min().isoformat() if not dates.empty else None,
        "date_max": dates.max().isoformat() if not dates.empty else None,
        "gold_30_available": False,
    }


def _candidate_evaluation(
    frame: pd.DataFrame,
    matrix: np.ndarray,
    *,
    auc_floor: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "auc": None,
        "brier": None,
        "log_loss": None,
        "auc_floor": auc_floor,
        "auc_noninferior": False,
        "split": {"train": 0, "test": 0},
        "gates": {
            "chronological": False,
            "regional": False,
            "patch_transfer": False,
            "roster_change": False,
            "sparse_data": False,
            "missingness": False,
            "early_snowball": False,
            "late_scaling": False,
            "comeback_behind_10": False,
            "comeback_behind_15": False,
        },
    }
    if "y_blue_win" not in frame.columns or len(frame) < 8:
        return result
    labels = pd.to_numeric(frame["y_blue_win"], errors="coerce")
    dates = pd.to_datetime(
        frame["date"] if "date" in frame.columns else pd.Series(pd.NaT, index=frame.index),
        errors="coerce",
        utc=True,
    )
    valid = labels.isin([0, 1]) & dates.notna()
    indices = np.flatnonzero(valid.to_numpy())
    if len(indices) < 8:
        return result
    ordered = indices[np.argsort(dates.iloc[indices].astype("int64").to_numpy())]
    split_at = max(1, min(len(ordered) - 1, int(len(ordered) * 0.8)))
    train_indices = ordered[:split_at]
    test_indices = ordered[split_at:]
    result["split"] = {"train": int(len(train_indices)), "test": int(len(test_indices))}
    y_train = labels.iloc[train_indices].astype(int).to_numpy()
    y_test = labels.iloc[test_indices].astype(int).to_numpy()
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return result
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    model.fit(matrix[train_indices], y_train)
    predictions = model.predict_proba(matrix[test_indices])[:, 1]
    auc = float(roc_auc_score(y_test, predictions))
    result.update(
        {
            "auc": auc,
            "brier": float(brier_score_loss(y_test, predictions)),
            "log_loss": float(log_loss(y_test, predictions, labels=[0, 1])),
            "auc_noninferior": bool(auc >= auc_floor),
        }
    )
    result["gates"]["chronological"] = True
    result["gates"]["regional"] = bool(
        "region" in frame.columns
        and frame["region"].dropna().astype(str).nunique() >= 2
    )
    result["gates"]["patch_transfer"] = bool(
        "oe_patch_token" in frame.columns
        and frame["oe_patch_token"].dropna().astype(str).nunique() >= 2
    )
    result["gates"]["roster_change"] = bool(
        "roster_change" in frame.columns and frame["roster_change"].notna().any()
    )
    result["gates"]["sparse_data"] = len(test_indices) >= 20
    result["gates"]["missingness"] = any(
        frame[f"gold_diff_{phase}"].isna().any() for phase in PHASES if f"gold_diff_{phase}" in frame
    )
    result["gates"]["early_snowball"] = bool(
        "gold_diff_10" in frame.columns and frame["gold_diff_10"].notna().sum() >= 20
    )
    result["gates"]["late_scaling"] = bool(
        "gold_diff_25" in frame.columns and frame["gold_diff_25"].notna().sum() >= 20
    )
    result["gates"]["comeback_behind_10"] = bool(
        "gold_diff_10" in frame.columns
        and ((pd.to_numeric(frame["gold_diff_10"], errors="coerce") < 0) & labels.notna()).sum() >= 20
    )
    result["gates"]["comeback_behind_15"] = bool(
        "gold_diff_15" in frame.columns
        and ((pd.to_numeric(frame["gold_diff_15"], errors="coerce") < 0) & labels.notna()).sum() >= 20
    )
    return result


def fit_phase_curve(
    frame: pd.DataFrame,
    *,
    model_version: str = MODEL_VERSION,
    require_patch: str = REQUIRED_CLIENT_PATCH,
    min_patch_rows: int = 100,
    baseline_auc_floor: float = BASELINE_AUC_FLOOR,
    strict: bool = False,
) -> dict[str, Any]:
    """Fit a candidate phase curve and keep authority unavailable.

    ``strict=True`` raises when the required accepted patch is absent.  The
    default returns a structured unavailable artifact so refresh jobs can keep
    publishing the prior verified model.
    """

    if not math.isfinite(float(baseline_auc_floor)) or baseline_auc_floor < 0.5:
        raise ValueError("baseline AUC floor must be a finite value at least 0.5")
    value = _normalise_fit_frame(frame)
    if "oe_patch_token" not in value.columns:
        reason = "source_patch_column_missing"
        if strict:
            raise PhaseCurveUnavailable(reason)
        return unavailable_phase_curve(blockers=[reason])
    patch_tokens: list[str] = []
    for item in value["oe_patch_token"].tolist():
        if pd.isna(item):
            continue
        token = _fit_patch_token(item)
        if token:
            patch_tokens.append(token)
    required_client = client_patch(require_patch)
    required_rows = [
        token
        for token in patch_tokens
        if client_patch(token) == required_client
    ]
    if len(required_rows) < int(min_patch_rows):
        reason = f"accepted_oe_patch_{required_client}_rows_missing"
        if strict:
            raise PhaseCurveUnavailable(reason)
        artifact = unavailable_phase_curve(blockers=[reason, "frozen_phase_curve_promotion_receipt_missing"])
        artifact["source_snapshot"] = _model_source_snapshot(value)
        artifact["source_snapshot"]["required_patch_rows"] = len(required_rows)
        artifact["source_snapshot"]["required_patch_minimum"] = int(min_patch_rows)
        artifact["evaluation"] = {"baseline_auc_floor": baseline_auc_floor, "authority": "unavailable"}
        return artifact

    matrix, feature_names = build_pre_match_design(value)
    assert_no_gold_leakage(feature_names)
    if matrix.shape[1] == 0:
        reason = "pre_match_feature_schema_empty"
        if strict:
            raise PhaseCurveUnavailable(reason)
        return unavailable_phase_curve(
            blockers=[reason, "frozen_phase_curve_promotion_receipt_missing"]
        )
    phase_models: dict[str, dict[str, Any]] = {}
    training_counts: dict[str, int] = {}
    coverage: dict[str, float | None] = {}
    draft_coefficients: dict[str, float | None] = {}
    for phase in PHASES:
        target_name = f"gold_diff_{phase}"
        if target_name not in value.columns:
            training_counts[phase] = 0
            coverage[phase] = None
            continue
        target = pd.to_numeric(value[target_name], errors="coerce")
        valid = target.notna().to_numpy()
        training_counts[phase] = int(valid.sum())
        coverage[phase] = float(valid.mean()) if len(valid) else None
        if valid.sum() < 3:
            continue
        model = Ridge(alpha=10.0)
        model.fit(matrix[valid], target.to_numpy(dtype=float)[valid])
        coefficients = [float(item) for item in model.coef_]
        phase_models[phase] = {
            "coefficients": coefficients,
            "intercept": float(model.intercept_),
        }
        try:
            draft_coefficients[phase] = coefficients[feature_names.index("num:draft_win_logit_blue")]
        except ValueError:
            draft_coefficients[phase] = None
    evaluation = _candidate_evaluation(value, matrix, auc_floor=float(baseline_auc_floor))
    blockers = ["frozen_phase_curve_promotion_receipt_missing"]
    if not evaluation.get("auc_noninferior"):
        blockers.append("auc_floor_not_met")
    for gate, passed in (evaluation.get("gates") or {}).items():
        if passed is not True:
            blockers.append(f"evaluation_gate_{gate}_not_met")
    sample_dates = pd.to_datetime(
        value["date"] if "date" in value.columns else pd.Series(pd.NaT, index=value.index),
        errors="coerce",
        utc=True,
    ).dropna()
    sample_window = (
        [sample_dates.min().date().isoformat(), sample_dates.max().date().isoformat()]
        if not sample_dates.empty
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "unavailable",
        "source": "oe_only",
        "window": list(PHASES),
        "expected_gold_diff": _empty_values(),
        "phase_draft_edge": _empty_values(),
        "scaling_index": None,
        "snowball_index": None,
        "comeback_resilience": None,
        "sample_window": sample_window,
        "model_version": model_version,
        "source_snapshot": _model_source_snapshot(value),
        "lcc_atomization": lcc_atomization_metadata(),
        "design_features": feature_names,
        "phase_models": phase_models,
        "phase_draft_coefficients": draft_coefficients,
        "training_counts": training_counts,
        "coverage": coverage,
        "evaluation": evaluation,
        "leakage_contract": {
            "pre_match_features": True,
            "gold_targets": list(PHASES),
            "gold_30_available": False,
            "missing_25_censored": True,
            "observed_gold_allowed": False,
        },
        "blockers": blockers,
    }


def _feature_value(features: Mapping[str, Any], feature_name: str) -> float:
    if feature_name.startswith("num:"):
        value = features.get(feature_name[4:])
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        return number if math.isfinite(number) else 0.0
    if feature_name.startswith("cat:") and "=" in feature_name:
        name, expected = feature_name[4:].split("=", 1)
        return 1.0 if str(features.get(name) or "").strip() == expected else 0.0
    return 0.0


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _phase_model_payload(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_version": artifact.get("model_version"),
        "design_features": artifact.get("design_features") or [],
        "phase_models": artifact.get("phase_models") or {},
        "phase_draft_coefficients": artifact.get("phase_draft_coefficients") or {},
    }


def _artifact_payload_without_promotion(artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in artifact.items() if key != "promotion"}


def _safe_promotion_path(artifact_path: Path, receipt_path: object) -> Path | None:
    if not isinstance(receipt_path, str) or not receipt_path or Path(receipt_path).is_absolute():
        return None
    candidate = (ROOT / receipt_path).resolve()
    try:
        candidate.relative_to(PROMOTION_ROOT.resolve())
    except ValueError:
        return None
    if candidate.suffix.lower() != ".json":
        return None
    try:
        artifact_path.resolve().relative_to(MODELS_DIR.resolve())
    except ValueError:
        return None
    return candidate


def _load_promotion_receipt(receipt_path: Path) -> tuple[dict[str, Any], str] | None:
    try:
        raw = receipt_path.read_bytes()
    except OSError:
        return None
    if not raw or len(raw) > 256 * 1024:
        return None
    file_sha256 = hashlib.sha256(raw).hexdigest()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(value, Mapping):
        return None
    return dict(value), file_sha256


def _promoted_receipt_is_valid(
    artifact: Mapping[str, Any],
    *,
    artifact_path: Path | None,
) -> bool:
    promotion = artifact.get("promotion")
    if not isinstance(promotion, Mapping) or artifact_path is None:
        return False
    receipt_path = _safe_promotion_path(artifact_path, promotion.get("receipt_path"))
    if receipt_path is None:
        return False
    loaded = _load_promotion_receipt(receipt_path)
    if loaded is None:
        return False
    receipt, file_sha256 = loaded
    claimed_file_sha256 = str(promotion.get("receipt_sha256") or "").lower()
    if not _HASH_RE.fullmatch(claimed_file_sha256) or claimed_file_sha256 != file_sha256:
        return False
    if receipt.get("schema_version") != PROMOTION_SCHEMA_VERSION:
        return False
    if receipt.get("status") != "approved" or receipt.get("authority") != "independent":
        return False
    receipt_model_hash = str(receipt.get("model_sha256") or "").lower()
    promotion_model_hash = str(promotion.get("model_sha256") or "").lower()
    calculated_model_hash = hashlib.sha256(_canonical_json_bytes(_phase_model_payload(artifact))).hexdigest()
    if (
        not _HASH_RE.fullmatch(receipt_model_hash)
        or receipt_model_hash != promotion_model_hash
        or receipt_model_hash != calculated_model_hash
    ):
        return False
    artifact_hash = hashlib.sha256(
        _canonical_json_bytes(_artifact_payload_without_promotion(artifact))
    ).hexdigest()
    if str(receipt.get("artifact_sha256") or "").lower() != artifact_hash:
        return False
    if receipt.get("model_version") != artifact.get("model_version"):
        return False
    evaluation = artifact.get("evaluation")
    gates = evaluation.get("gates") if isinstance(evaluation, Mapping) else None
    receipt_gates = receipt.get("gates")
    if not isinstance(gates, Mapping) or not isinstance(receipt_gates, Mapping):
        return False
    if evaluation.get("auc_noninferior") is not True:
        return False
    if any(gates.get(name) is not True or receipt_gates.get(name) is not True for name in REQUIRED_PROMOTION_GATES):
        return False
    return True


def score_phase_curve(
    artifact: Mapping[str, Any],
    features: Mapping[str, Any],
    *,
    draft_logit: float,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    """Score only a receipt-bound promoted artifact."""

    if artifact.get("authority") != "promoted":
        blockers = artifact.get("blockers") or ["phase_curve_authority_unavailable"]
        return unavailable_phase_curve(source=str(artifact.get("source") or "oe_only"), blockers=blockers)
    if not _promoted_receipt_is_valid(artifact, artifact_path=artifact_path):
        return unavailable_phase_curve(blockers=["phase_curve_promotion_receipt_invalid"])
    evaluation = artifact.get("evaluation")
    try:
        auc = float(evaluation.get("auc")) if isinstance(evaluation, Mapping) else math.nan
        auc_floor = (
            float(evaluation.get("auc_floor", BASELINE_AUC_FLOOR))
            if isinstance(evaluation, Mapping)
            else BASELINE_AUC_FLOOR
        )
    except (TypeError, ValueError):
        auc = math.nan
        auc_floor = BASELINE_AUC_FLOOR
    if (
        not isinstance(evaluation, Mapping)
        or evaluation.get("auc_noninferior") is not True
        or not math.isfinite(auc)
        or auc < auc_floor
    ):
        return unavailable_phase_curve(blockers=["auc_floor_not_met"])
    feature_names = [str(item) for item in artifact.get("design_features") or []]
    assert_no_gold_leakage(feature_names)
    vector = np.asarray([_feature_value(features, name) for name in feature_names], dtype=float)
    expected: dict[str, float | None] = {}
    edges: dict[str, float | None] = {}
    models = artifact.get("phase_models") or {}
    coefficients = artifact.get("phase_draft_coefficients") or {}
    for phase in PHASES:
        model = models.get(phase) or {}
        coef = np.asarray(model.get("coefficients") or [], dtype=float)
        if len(coef) != len(vector):
            expected[phase] = None
        else:
            expected[phase] = round(float(model.get("intercept") or 0.0) + float(coef @ vector), 4)
        draft_coef = coefficients.get(phase)
        edges[phase] = round(float(draft_coef) * float(draft_logit), 4) if draft_coef is not None else None
    scaling = (
        round(edges["25"] - edges["10"], 4)
        if edges["10"] is not None and edges["25"] is not None
        else None
    )
    snowball = (
        round(edges["10"] - edges["25"], 4)
        if edges["10"] is not None and edges["25"] is not None
        else None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "promoted",
        "source": str(artifact.get("source") or "oe_only"),
        "window": list(PHASES),
        "expected_gold_diff": expected,
        "phase_draft_edge": edges,
        "scaling_index": scaling,
        "snowball_index": snowball,
        "comeback_resilience": artifact.get("comeback_resilience"),
        "sample_window": artifact.get("sample_window"),
        "model_version": artifact.get("model_version"),
        "lcc_atomization": artifact.get("lcc_atomization") or lcc_atomization_metadata(),
    }


def phase_curve_for_draft(
    blue: Sequence[Any] | Mapping[str, Any],
    red: Sequence[Any] | Mapping[str, Any],
    *,
    draft_logit: float,
    league: str | None = None,
    region: str | None = None,
    patch: str | None = None,
    elo_diff: float | None = None,
    player_elo_diff: float | None = None,
    roster_change: float | None = None,
    extra_features: Mapping[str, Any] | None = None,
    artifact_path: Path = PHASE_CURVE_PATH,
) -> dict[str, Any]:
    """Load and score the phase artifact, failing closed by default."""

    try:
        artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return unavailable_phase_curve(blockers=["phase_curve_artifact_missing"])
    if not isinstance(artifact, Mapping):
        return unavailable_phase_curve(blockers=["phase_curve_artifact_invalid"])
    resolved_artifact_path = Path(artifact_path).expanduser().resolve()
    if artifact.get("authority") != "promoted":
        return unavailable_phase_curve(
            source=str(artifact.get("source") or "oe_only"),
            blockers=artifact.get("blockers") or ["phase_curve_authority_unavailable"],
        )
    if not _promoted_receipt_is_valid(artifact, artifact_path=resolved_artifact_path):
        return unavailable_phase_curve(blockers=["phase_curve_promotion_receipt_invalid"])
    features = pre_match_features_for_draft(
        blue,
        red,
        draft_logit,
        elo_diff=elo_diff,
        player_elo_diff=player_elo_diff,
        league=league,
        region=region,
        patch=patch,
        roster_change=roster_change,
        extra_features=extra_features,
    )
    return score_phase_curve(
        artifact,
        features,
        draft_logit=draft_logit,
        artifact_path=resolved_artifact_path,
    )


def live_state_features(observed: Mapping[str, Any]) -> dict[str, Any]:
    """Mark observed state as live-only input.

    This helper exists to make the boundary explicit in tests and research
    notebooks.  The pre-match builder rejects these same fields.
    """

    result = {
        str(name): value
        for name, value in observed.items()
        if str(name).lower()
        in {
            "gold_diff",
            "goldat10",
            "goldat15",
            "goldat20",
            "goldat25",
            "clock",
            "objectives",
        }
    }
    result["source"] = "timeline_live_state"
    return result


__all__ = [
    "BASELINE_AUC_FLOOR",
    "PHASES",
    "PHASE_CURVE_PATH",
    "PhaseCurveUnavailable",
    "SCHEMA_VERSION",
    "assert_no_gold_leakage",
    "atomized_draft_features",
    "build_pre_match_design",
    "fit_phase_curve",
    "lcc_atomization_metadata",
    "lcc_atom_profile_for_champion",
    "live_state_features",
    "phase_curve_for_draft",
    "pre_match_features_for_draft",
    "prepare_phase_frame",
    "score_phase_curve",
    "unavailable_phase_curve",
]
