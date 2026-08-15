"""Build the descriptive champion-role Elo candidate for the tier API.

The ladder starts with the complete 2025 source window and replays maps in
time order. Each champion is rated against the opposing champion in the same
role. The pre-map team Elo probability controls for team strength. A map
updates the ladder only after its result is known.

This module writes a development candidate. It does not create a production
manifest or grant model authority. A separate worker can replay the same
source, compare against the previous published artifact, and publish an
immutable production pointer after the required reviews pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, ndtr

from lol_kills.etl.oe_live_source import _game_keys, _identity_complete_player_game_ids
from lol_kills.v2.champions.atoms.consume import AtomBridge

SCHEMA_VERSION = "scryglass:champion-role-elo-candidate:v1"
ARTIFACT_KIND = "tier_list_candidate"
SOURCE_MODES = ("oe_only", "oe_plus_grid")
DEFAULT_SOURCE_MODE = "oe_only"
HISTORY_START = pd.Timestamp("2025-01-01T00:00:00Z")
LIVE_WINDOW_START = pd.Timestamp("2026-07-18T00:00:00Z")
ROLES = ("top", "jungle", "mid", "bot", "support")
INTERNATIONAL_EVENTS = {"msi", "ewc", "worlds", "fst", "first stand"}
TIER_BUCKETS = ("Z Blind", "Z Counter", "S Blind", "S Counter", "A", "B", "C", "D")
SOURCE_LOCATOR = "data/lol/warehouse/parquet/oe_live/oe_player_games.parquet"
ANNUAL_SOURCE_LOCATOR = "data/lol/warehouse/parquet/oe_player_games.parquet"
IDENTITY_CROSSWALK_LOCATOR = "data/lol/v2/champions/champion-id-crosswalk-v1.json"
IDENTITY_METADATA_LOCATOR = "data/lol/v2/champions/sources/riot-champion-metadata-16.14.1.json"
ATOM_BRIDGE_LOCATOR = "data/lol/v2/champions/lcc-atom-bridge-v1.json"
# Keep the historical bridge as the default for older fixtures and rows. New
# patch rows must use the bridge for their exact atom snapshot.
ATOM_BRIDGE_LOCATORS = {
    "26.15": ATOM_BRIDGE_LOCATOR,
    "26.16": "data/lol/v2/champions/lcc-atom-bridge-26.16.json",
}
DEFAULT_OUTPUT = Path("data/lol/v2/tierlists/champion-elo-candidate-v1.json")
DEFAULT_MIN_APPEARANCES = 1
INITIAL_RATING = 1500.0
CHAMPION_K = 12.0
TEAM_K = 20.0
RATING_TO_PP = 25.0 * math.log(10.0) / 400.0
STRENGTH_PRIOR_SD = 0.75
MATCHUP_PRIOR_SD = 0.35
RECENCY_HALF_LIFE_DAYS = 120.0
MATCHUP_MIN_EFFECTIVE_MAPS = 3.0
MATCHUP_MIN_OPPONENTS = 5
MATCHUP_MIN_SERIES = 3
COUNTER_POSTERIOR_THRESHOLD = 0.80
COUNTER_EFFECT_THRESHOLD_LOGIT = 0.05
MATCHUP_MAX_POSTERIOR_SD = 0.90
STRENGTH_MAX_CONTRAST_SD = 0.50
LEGAL_OPPONENT_COUNT = 5
BLIND_TAIL_SHARE = 0.20
BLIND_CREDIBLE_Z = 1.2815515655446004
POSTERIOR_DRAWS = 384
TIER_MEMBERSHIP_PROBABILITY = 0.65


class ChampionEloError(ValueError):
    """Raised when the source cannot support a deterministic candidate."""


@dataclass
class ChampState:
    rating: float = INITIAL_RATING
    appearances: int = 0


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_stamp(value: pd.Timestamp) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.isoformat().replace("+00:00", "Z")


from functools import lru_cache as _lru_cache

@_lru_cache(maxsize=65536)
def _normalize_name(value: object) -> str:
    return "".join(ch for ch in str(value).strip().casefold() if ch.isalnum())


def _canonical_role(value: object) -> str | None:
    token = str(value).strip().casefold()
    return {
        "top": "top",
        "jng": "jungle",
        "jungle": "jungle",
        "mid": "mid",
        "middle": "mid",
        "bot": "bot",
        "bottom": "bot",
        "adc": "bot",
        "sup": "support",
        "support": "support",
        "utility": "support",
    }.get(token)


def _side(value: object) -> str | None:
    token = str(value).strip().casefold()
    if token in {"blue", "1"}:
        return "blue"
    if token in {"red", "2"}:
        return "red"
    return None


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def _logit(value: float) -> float:
    clipped = max(1e-6, min(1.0 - 1e-6, value))
    return math.log(clipped / (1.0 - clipped))


def _scope_for(league: str, competition_tier: str | None, event_kind: str | None) -> tuple[str, str, str | None]:
    event = (event_kind or "").strip().casefold()
    league_token = league.strip().upper()
    if event in INTERNATIONAL_EVENTS or competition_tier == "international":
        event_name = event.replace(" ", "_").upper() or league_token
        return f"event:{event_name.lower()}", event_name, "international"
    tier = competition_tier or "unknown"
    return f"league:{league_token.lower()}:{tier}", league_token, tier


def _load_crosswalk(root: Path) -> tuple[dict[str, str], dict[str, Any]]:
    path = root / IDENTITY_CROSSWALK_LOCATOR
    try:
        crosswalk = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ChampionEloError(f"champion crosswalk cannot be read: {path}") from exc
    out: dict[str, str] = {}
    for entry in crosswalk.get("entries", []):
        if not isinstance(entry, Mapping):
            continue
        stable = entry.get("stable_champion_id")
        if not isinstance(stable, str):
            continue
        for name in (entry.get("oe_name"), entry.get("riot_display_name"), entry.get("normalized_oe_name")):
            if isinstance(name, str) and name.strip():
                out[_normalize_name(name)] = stable
    identity_sources: dict[str, Any] = {
        "crosswalk": {
            "locator": IDENTITY_CROSSWALK_LOCATOR,
            "raw_sha256": _sha256_path(path),
            "coverage": crosswalk.get("coverage"),
        }
    }

    # The frozen crosswalk covers the names in its original preflight. The
    # refreshed warehouse can contain later observed names. Resolve those
    # exact display names from the pinned Riot metadata vocabulary. This keeps
    # the identity repair explicit and hash-bound without changing the older
    # crosswalk artifact in place.
    metadata_path = root / IDENTITY_METADATA_LOCATOR
    supplemental: list[str] = []
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        metadata = None
        identity_sources["metadata_error"] = f"{type(exc).__name__}: {exc}"
    if isinstance(metadata, Mapping) and isinstance(metadata.get("data"), Mapping):
        for entry in metadata["data"].values():
            if not isinstance(entry, Mapping):
                continue
            name = entry.get("name")
            numeric_id = entry.get("key")
            if not isinstance(name, str) or not isinstance(numeric_id, str) or not numeric_id.isdecimal():
                continue
            key = _normalize_name(name)
            stable = f"riot:champion:{int(numeric_id)}"
            if key not in out:
                out[key] = stable
                supplemental.append(name)
        identity_sources["metadata"] = {
            "locator": IDENTITY_METADATA_LOCATOR,
            "raw_sha256": _sha256_path(metadata_path),
            "version": metadata.get("version"),
            "supplemental_names": sorted(supplemental, key=_normalize_name),
        }
    else:
        identity_sources["metadata"] = {
            "locator": IDENTITY_METADATA_LOCATOR,
            "present": False,
            "supplemental_names": [],
        }
    return out, identity_sources


def _load_source(
    root: Path,
    *,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, str, str]:
    source_paths = [root / SOURCE_LOCATOR, root / ANNUAL_SOURCE_LOCATOR]
    primary_path = next(
        (path for path in source_paths if path.is_file() and not path.is_symlink()),
        None,
    )
    if primary_path is None:
        raise ChampionEloError(
            "source is missing or not a regular file: "
            f"{root / SOURCE_LOCATOR}"
        )
    needed = [
        "gameid",
        "date",
        "league",
        "competition_tier",
        "event_kind",
        "patch",
        "position",
        "champion",
        "side",
        "teamname",
        "playername",
        "result",
    ]
    source_paths = [primary_path]
    frames: list[pd.DataFrame] = []
    source_bindings: list[dict[str, str]] = []
    for path in source_paths:
        try:
            frames.append(pd.read_parquet(path, columns=needed))
        except (OSError, KeyError, ValueError) as exc:
            raise ChampionEloError(f"source columns cannot be read: {exc}") from exc
        source_bindings.append(
            {
                "locator": str(path.relative_to(root)),
                "raw_sha256": _sha256_path(path),
            }
        )
    frame = pd.concat(frames, ignore_index=True, sort=False)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame["role"] = frame["position"].map(_canonical_role)
    frame["side_norm"] = frame["side"].map(_side)
    frame["result_num"] = pd.to_numeric(frame["result"], errors="coerce")
    frame["league_norm"] = frame["league"].astype(str).str.strip().str.upper()
    frame["competition_tier_norm"] = frame["competition_tier"].where(frame["competition_tier"].notna(), None)
    frame["event_kind_norm"] = frame["event_kind"].astype(str).str.strip().str.casefold()
    frame = frame[
        frame["date"].notna()
        & frame["date"].ge(HISTORY_START)
        & frame["role"].isin(ROLES)
        & frame["side_norm"].isin(("blue", "red"))
        & frame["result_num"].isin((0, 1))
        & frame["league_norm"].ne("")
    ].copy()
    if as_of is not None:
        cutoff = pd.Timestamp(as_of)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        else:
            cutoff = cutoff.tz_convert("UTC")
        frame = frame[frame["date"].le(cutoff)].copy()
    if frame.empty:
        raise ChampionEloError("source has no completed maps in the requested window")
    frame["game_id"] = _game_keys(frame)
    complete_game_ids = _identity_complete_player_game_ids(
        frame.assign(game_uid=frame["game_id"])
    )
    frame = frame[frame["game_id"].isin(complete_game_ids)].copy()
    if frame.empty:
        raise ChampionEloError("source has no identity-complete five-role maps")
    return (
        frame,
        _sha256_bytes(_canonical_json(source_bindings)),
        str(primary_path.relative_to(root)),
    )


def _build_maps(frame: pd.DataFrame) -> tuple[list[dict[str, Any]], int]:
    maps: list[dict[str, Any]] = []
    rejected = 0
    seen_signatures: set[tuple[Any, ...]] = set()
    for game_id, group in frame.groupby("game_id", sort=False):
        if group["date"].nunique() != 1:
            rejected += 1
            continue
        blue = group[group["side_norm"] == "blue"]
        red = group[group["side_norm"] == "red"]
        roles: dict[str, dict[str, Any]] = {}
        valid = True
        for role in ROLES:
            b = blue[blue["role"] == role]
            r = red[red["role"] == role]
            if len(b) != 1 or len(r) != 1:
                valid = False
                break
            roles[role] = {
                "blue_champion": str(b.iloc[0]["champion"]).strip(),
                "red_champion": str(r.iloc[0]["champion"]).strip(),
            }
        if not valid:
            rejected += 1
            continue
        first = group.iloc[0]
        blue_result = blue[blue["role"] == "top"].iloc[0]["result_num"]
        scope_id, scope_label, scope_tier = _scope_for(
            str(first["league_norm"]),
            None if pd.isna(first["competition_tier_norm"]) else str(first["competition_tier_norm"]),
            None if pd.isna(first["event_kind_norm"]) else str(first["event_kind_norm"]),
        )
        map_record = {
            "game_id": str(game_id),
            "date": pd.Timestamp(first["date"]),
            "league": str(first["league_norm"]),
            "competition_tier": scope_tier,
            "event_kind": None if scope_id.startswith("league:") else scope_label.casefold(),
            "patch": str(first["patch"]).strip(),
            "scope_id": scope_id,
            "scope_label": scope_label,
            "blue_team": str(blue.iloc[0]["teamname"]),
            "red_team": str(red.iloc[0]["teamname"]),
            "series_id": "|".join(
                (
                    str(first["league_norm"]),
                    pd.Timestamp(first["date"]).strftime("%Y-%m-%d"),
                    *sorted(
                        (
                            _normalize_name(str(blue.iloc[0]["teamname"])),
                            _normalize_name(str(red.iloc[0]["teamname"])),
                        )
                    ),
                )
            ),
            "y_blue_win": int(float(blue_result)),
            "roles": roles,
        }
        signature = (
            _utc_stamp(map_record["date"]),
            map_record["league"],
            _normalize_name(map_record["blue_team"]),
            _normalize_name(map_record["red_team"]),
            map_record["y_blue_win"],
            tuple(
                _normalize_name(map_record["roles"][role][side])
                for role in ROLES
                for side in ("blue_champion", "red_champion")
            ),
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        maps.append(map_record)
    maps.sort(key=lambda row: (row["date"], row["game_id"]))
    return maps, rejected


def _lower_tail_mean(values: list[float], share: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.5
    count = max(1, math.ceil(len(ordered) * share))
    return sum(ordered[:count]) / count


def _fit_hierarchical_cell(
    states: Mapping[str, ChampState],
    observations: list[dict[str, Any]],
    *,
    min_appearances: int,
    reference_date: pd.Timestamp,
) -> list[dict[str, Any]]:
    raise ChampionEloError(
        "retired single-role tier fit cannot be used; call _fit_joint_scope"
    )


def _assign_tier_buckets(rows: list[dict[str, Any]]) -> None:
    available = [row for row in rows if row["counterability_status"] == "available"]
    assigned: dict[str, str] = {}

    def take(label: str, ordered: list[dict[str, Any]], quota: int) -> None:
        for row in ordered:
            key = _normalize_name(row["champion"])
            if key in assigned:
                continue
            assigned[key] = label
            if sum(candidate == label for candidate in assigned.values()) >= quota:
                return

    if len(available) >= 4:
        z_quota = 1
        s_quota = max(1, min(math.ceil(len(available) * 0.20), (len(available) - 2) // 2))
        blind_order = sorted(
            available,
            key=lambda row: (-float(row["blind_score_pp"]), -row["rating"], _normalize_name(row["champion"])),
        )
        counter_order = sorted(
            [row for row in available if float(row["counter_score"]) > 0.0],
            key=lambda row: (
                -float(row["counter_score"]),
                -int(row["countered_opponent_count"]),
                -float(row["countered_opponent_share"]),
                -row["rating"],
                _normalize_name(row["champion"]),
            ),
        )
        take("Z Counter", counter_order, z_quota)
        take("Z Blind", blind_order, z_quota)
        take("S Counter", counter_order, s_quota)
        take("S Blind", blind_order, s_quota)

    draw_count = min(
        (
            len(row.get("_blind_draws", ()))
            for row in available
            if row.get("_blind_draws") is not None
        ),
        default=0,
    )
    membership_counts = {key: 0 for key in assigned}
    if assigned and draw_count:
        for draw_index in range(draw_count):
            draw_assignment: dict[str, str] = {}

            def draw_take(label: str, ordered: list[dict[str, Any]], quota: int) -> None:
                for row in ordered:
                    key = _normalize_name(row["champion"])
                    if key in draw_assignment:
                        continue
                    draw_assignment[key] = label
                    if sum(candidate == label for candidate in draw_assignment.values()) >= quota:
                        return

            draw_blind = sorted(
                available,
                key=lambda row: (
                    -float(row["_blind_draws"][draw_index]),
                    -row["rating"],
                    _normalize_name(row["champion"]),
                ),
            )
            draw_counter = sorted(
                available,
                key=lambda row: (
                    -float(row["_counter_draws"][draw_index]),
                    -row["rating"],
                    _normalize_name(row["champion"]),
                ),
            )
            draw_take("Z Counter", draw_counter, z_quota)
            draw_take("Z Blind", draw_blind, z_quota)
            draw_take("S Counter", draw_counter, s_quota)
            draw_take("S Blind", draw_blind, s_quota)
            for key, label in assigned.items():
                if draw_assignment.get(key) == label:
                    membership_counts[key] += 1

    stable_assigned: set[str] = set()
    for row in rows:
        key = _normalize_name(row["champion"])
        label = assigned.get(key)
        probability = membership_counts.get(key, draw_count) / draw_count if label and draw_count else (1.0 if label else None)
        row["tier_membership_probability"] = None if probability is None else round(probability, 4)
        if label and probability is not None and probability >= TIER_MEMBERSHIP_PROBABILITY:
            row["tier_bucket"] = label
            stable_assigned.add(key)
        row.pop("_blind_draws", None)
        row.pop("_counter_draws", None)

    remaining = [row for row in rows if _normalize_name(row["champion"]) not in stable_assigned]
    base_buckets = ("A", "B", "C", "D")
    for index, row in enumerate(remaining):
        row["tier_bucket"] = base_buckets[
            min(len(base_buckets) - 1, index * len(base_buckets) // max(1, len(remaining)))
        ]


def _weighted_lower_tail(values: np.ndarray, weights: np.ndarray, share: float) -> float:
    order = np.argsort(values)
    remaining = share
    total = 0.0
    for index in order:
        portion = min(remaining, float(weights[index]))
        total += portion * float(values[index])
        remaining -= portion
        if remaining <= 1e-12:
            break
    return total / share


def _weighted_lower_tail_rows(values: np.ndarray, weights: np.ndarray, share: float) -> np.ndarray:
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    sorted_weights = weights[order]
    prior_weight = np.cumsum(sorted_weights, axis=1) - sorted_weights
    portions = np.minimum(sorted_weights, np.clip(share - prior_weight, 0.0, None))
    return np.sum(portions * sorted_values, axis=1) / share


def _fit_joint_scope(
    states_by_role: Mapping[str, Mapping[str, ChampState]],
    observations: list[dict[str, Any]],
    *,
    min_appearances: int,
    reference_date: pd.Timestamp,
    scope_id: str,
) -> dict[str, dict[str, Any]]:
    champions_by_role = {
        role: sorted(
            (champion for champion, state in states_by_role.get(role, {}).items() if state.appearances >= min_appearances),
            key=_normalize_name,
        )
        for role in ROLES
    }
    parameter_keys = [
        (role, champion)
        for role in ROLES
        for champion in champions_by_role[role]
    ]
    parameter_index = {key: index for index, key in enumerate(parameter_keys)}
    usable = [
        row for row in observations
        if all(
            (role, row["roles"][role][side]) in parameter_index
            for role in ROLES
            for side in ("blue_champion", "red_champion")
        )
    ]
    if not usable:
        return {role: {"rows": [], "legal_opponents": [], "design": {}} for role in ROLES}
    n_maps = len(usable)
    n_strength = len(parameter_keys)
    side_index = n_strength
    design = np.zeros((n_maps, n_strength + 1), dtype=float)
    for map_index, row in enumerate(usable):
        for role in ROLES:
            design[map_index, parameter_index[(role, row["roles"][role]["blue_champion"])]] += 1.0
            design[map_index, parameter_index[(role, row["roles"][role]["red_champion"])]] -= 1.0
        design[map_index, side_index] = 1.0
    outcome = np.asarray([float(row["outcome"]) for row in usable])
    offset = np.asarray([float(row["team_logit"]) for row in usable])
    age_days = np.asarray([
        max(0.0, (reference_date - pd.Timestamp(row["date"])).total_seconds() / 86400.0)
        for row in usable
    ])
    weight = np.power(0.5, age_days / RECENCY_HALF_LIFE_DAYS)
    precision = np.full(n_strength + 1, 1.0 / (STRENGTH_PRIOR_SD ** 2))
    precision[side_index] = 1.0 / (0.30 ** 2)

    component_by_role_champion: dict[tuple[str, str], int] = {}
    component_counts: dict[str, int] = {}
    contrast_columns: list[np.ndarray] = []
    for role in ROLES:
        champions = champions_by_role[role]
        adjacency = {champion: set() for champion in champions}
        for row in usable:
            blue_champion = row["roles"][role]["blue_champion"]
            red_champion = row["roles"][role]["red_champion"]
            adjacency[blue_champion].add(red_champion)
            adjacency[red_champion].add(blue_champion)
        components: list[list[str]] = []
        remaining = set(champions)
        while remaining:
            seed = min(remaining, key=_normalize_name)
            stack = [seed]
            component: list[str] = []
            remaining.remove(seed)
            while stack:
                champion = stack.pop()
                component.append(champion)
                for neighbor in sorted(adjacency[champion], key=_normalize_name, reverse=True):
                    if neighbor in remaining:
                        remaining.remove(neighbor)
                        stack.append(neighbor)
            components.append(sorted(component, key=_normalize_name))
        component_counts[role] = len(components)
        for component_id, component in enumerate(components):
            indices = [parameter_index[(role, champion)] for champion in component]
            for champion in component:
                component_by_role_champion[(role, champion)] = component_id
            if len(indices) < 2:
                continue
            raw_basis = np.zeros((len(indices), len(indices) - 1), dtype=float)
            raw_basis[:-1, :] = np.eye(len(indices) - 1)
            raw_basis[-1, :] = -1.0
            orthonormal_basis, _ = np.linalg.qr(raw_basis, mode="reduced")
            for local_column in range(orthonormal_basis.shape[1]):
                column = np.zeros(n_strength + 1, dtype=float)
                column[indices] = orthonormal_basis[:, local_column]
                contrast_columns.append(column)
    side_column = np.zeros(n_strength + 1, dtype=float)
    side_column[side_index] = 1.0
    contrast_columns.append(side_column)
    contrast = np.column_stack(contrast_columns)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reduced_design = design @ contrast
        reduced_precision = contrast.T @ (precision[:, None] * contrast)
    if not np.all(np.isfinite(reduced_design)) or not np.all(np.isfinite(reduced_precision)):
        raise ChampionEloError(f"identified contrast system is non-finite for {scope_id}")
    reduced_parameter_count = contrast.shape[1]

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            eta = offset + reduced_design @ parameters
            probability = expit(eta)
            loss = float(np.sum(weight * (np.logaddexp(0.0, eta) - outcome * eta)))
            loss += 0.5 * float(parameters @ reduced_precision @ parameters)
            gradient = reduced_design.T @ (weight * (probability - outcome)) + reduced_precision @ parameters
        if not math.isfinite(loss) or not np.all(np.isfinite(gradient)):
            return float("inf"), np.zeros_like(parameters)
        return loss, gradient

    fit = minimize(
        objective,
        np.zeros(reduced_parameter_count, dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-5.0, 5.0)] * (reduced_parameter_count - 1) + [(-2.0, 2.0)],
        options={"maxiter": 500, "ftol": 1e-11, "gtol": 1e-7},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ChampionEloError(f"joint hierarchical strength fit failed for {scope_id}: {fit.message}")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        eta = offset + reduced_design @ fit.x
        information = weight * expit(eta) * (1.0 - expit(eta))
        hessian = reduced_design.T @ (information[:, None] * reduced_design) + reduced_precision
    if not np.all(np.isfinite(hessian)):
        raise ChampionEloError(f"joint hierarchical Hessian is non-finite for {scope_id}")
    reduced_covariance = np.linalg.inv(hessian)
    reduced_covariance = 0.5 * (reduced_covariance + reduced_covariance.T)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        full_mean = contrast @ fit.x
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        covariance = contrast @ reduced_covariance @ contrast.T
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(full_mean)) or not np.all(np.isfinite(covariance)):
        raise ChampionEloError(f"identified joint fit is non-finite for {scope_id}")
    weighted_design = reduced_design * np.sqrt(weight)[:, None]
    design_rank = int(np.linalg.matrix_rank(weighted_design))
    design_columns = reduced_parameter_count
    design_rank_full = design_rank == design_columns
    design_condition = float(np.linalg.cond(weighted_design)) if design_rank_full else float("inf")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        eta = offset + reduced_design @ fit.x
    if not np.all(np.isfinite(eta)):
        raise ChampionEloError(f"identified joint linear predictor is non-finite for {scope_id}")

    pair_observations: dict[
        tuple[str, str, str],
        list[tuple[float, float, float, float, int, str]],
    ] = {}
    for map_index, row in enumerate(usable):
        for role in ROLES:
            blue_champion = row["roles"][role]["blue_champion"]
            red_champion = row["roles"][role]["red_champion"]
            pair = tuple(sorted((blue_champion, red_champion), key=_normalize_name))
            sign = 1.0 if blue_champion == pair[0] else -1.0
            pair_observations.setdefault((role, pair[0], pair[1]), []).append(
                (
                    outcome[map_index],
                    eta[map_index],
                    weight[map_index],
                    sign,
                    map_index,
                    str(row["series_id"]),
                )
            )

    pair_posteriors: dict[tuple[str, str, str], dict[str, Any]] = {}
    matchup_precision = 1.0 / (MATCHUP_PRIOR_SD ** 2)
    for key, pair_rows in pair_observations.items():
        gamma = 0.0
        for _ in range(50):
            gradient = matchup_precision * gamma
            info = matchup_precision
            for y, base_eta, row_weight, sign, _, _ in pair_rows:
                probability = float(expit(base_eta + sign * gamma))
                gradient += row_weight * sign * (probability - y)
                info += row_weight * probability * (1.0 - probability)
            step = gradient / info
            gamma -= step
            if abs(step) < 1e-9:
                break
        cross_information = np.zeros(reduced_parameter_count, dtype=float)
        for y, base_eta, row_weight, sign, map_index, _ in pair_rows:
            probability = float(expit(base_eta + sign * gamma))
            cross_information += (
                row_weight * probability * (1.0 - probability) * sign * reduced_design[map_index]
            )
        weights = np.asarray([row[2] for row in pair_rows])
        effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
        series_weights: dict[str, float] = {}
        canonical_outcomes: set[int] = set()
        for y, _, row_weight, sign, _, series_id in pair_rows:
            series_weights[series_id] = max(series_weights.get(series_id, 0.0), float(row_weight))
            canonical_outcomes.add(int(y if sign > 0 else 1.0 - y))
        series_weight_values = np.asarray(list(series_weights.values()), dtype=float)
        effective_series = float(
            series_weight_values.sum() ** 2 / np.square(series_weight_values).sum()
        )
        conditional_sd = 1.0 / math.sqrt(info)
        sensitivity = cross_information / info
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            marginal_variance = conditional_sd ** 2 + float(
                sensitivity @ reduced_covariance @ sensitivity
            )
        if not math.isfinite(marginal_variance):
            raise ChampionEloError(f"matchup marginal variance is non-finite for {scope_id} {key}")
        pair_posteriors[key] = {
            "mean": gamma,
            "conditional_sd": conditional_sd,
            "sd": math.sqrt(max(0.0, marginal_variance)),
            "effective_n": effective_n,
            "effective_series": effective_series,
            "outcome_variation": len(canonical_outcomes) == 2,
            "sensitivity": sensitivity,
        }

    output: dict[str, dict[str, Any]] = {}
    for role in ROLES:
        champions = champions_by_role[role]
        role_parameter_indices = np.asarray([parameter_index[(role, champion)] for champion in champions])
        role_mean = full_mean[role_parameter_indices]
        role_covariance = covariance[np.ix_(role_parameter_indices, role_parameter_indices)]
        pick_counts = np.asarray([states_by_role[role][champion].appearances for champion in champions], dtype=float)
        legal_pool_indices = sorted(
            range(len(champions)),
            key=lambda index: (-pick_counts[index], _normalize_name(champions[index])),
        )[:LEGAL_OPPONENT_COUNT + 1]
        legal_pool_weights = pick_counts[legal_pool_indices]
        legal_pool_weights = legal_pool_weights / legal_pool_weights.sum()
        legal_opponents = [
            {"champion": champions[index], "weight": round(float(weight), 8)}
            for index, weight in zip(legal_pool_indices, legal_pool_weights)
        ]
        legal_hash = _sha256_bytes(_canonical_json({
            "pool": legal_opponents,
            "rule": "take the five highest-pick legal opponents after excluding the focal champion",
        }))
        seed = int(hashlib.sha256(f"{scope_id}|{role}".encode("utf-8")).hexdigest()[:16], 16)
        rng = np.random.default_rng(seed)
        role_covariance = 0.5 * (role_covariance + role_covariance.T)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            reduced_parameter_draws = rng.multivariate_normal(
                fit.x,
                reduced_covariance,
                size=POSTERIOR_DRAWS,
                check_valid="raise",
            )
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            parameter_draws = reduced_parameter_draws @ contrast.T
        if not np.all(np.isfinite(parameter_draws)):
            raise ChampionEloError(f"identified posterior draws are non-finite for {scope_id} {role}")
        strength_draws = parameter_draws[:, role_parameter_indices]
        if not np.all(np.isfinite(strength_draws)):
            raise ChampionEloError(f"joint hierarchical posterior draws are non-finite for {scope_id} {role}")
        rows: list[dict[str, Any]] = []
        for champion_i, champion in enumerate(champions):
            opponent_indices = [
                index for index in legal_pool_indices if index != champion_i
            ][:LEGAL_OPPONENT_COUNT]
            opponent_weights = pick_counts[opponent_indices]
            if opponent_weights.size:
                opponent_weights = opponent_weights / opponent_weights.sum()
            row_legal_opponents = [
                {"champion": champions[index], "weight": round(float(weight), 8)}
                for index, weight in zip(opponent_indices, opponent_weights)
            ]
            row_legal_hash = _sha256_bytes(_canonical_json(row_legal_opponents))
            if not opponent_indices:
                standardized_strength = 0.5
                blind_draws = np.full(POSTERIOR_DRAWS, 0.5)
                supported = []
                countered = 0
                expected_counter_breadth = 0.0
                countered_weight = 0.0
                counter_breadth_draws = np.zeros(POSTERIOR_DRAWS)
                effective_maps = 0.0
            else:
                map_probabilities = []
                matchup_draw_columns = []
                supported = []
                countered = 0
                expected_counter_breadth = 0.0
                countered_weight = 0.0
                counter_breadth_draws = np.zeros(POSTERIOR_DRAWS)
                effective_maps = 0.0
                for opponent_i, opponent_weight in zip(opponent_indices, opponent_weights):
                    opponent = champions[opponent_i]
                    pair = tuple(sorted((champion, opponent), key=_normalize_name))
                    posterior = pair_posteriors.get((role, pair[0], pair[1]))
                    if posterior is None:
                        interaction_draw = rng.normal(0.0, MATCHUP_PRIOR_SD, size=POSTERIOR_DRAWS)
                    else:
                        sign = 1.0 if champion == pair[0] else -1.0
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            canonical_draw = (
                                posterior["mean"]
                                - (reduced_parameter_draws - fit.x) @ posterior["sensitivity"]
                                + rng.normal(0.0, posterior["conditional_sd"], size=POSTERIOR_DRAWS)
                            )
                        if not np.all(np.isfinite(canonical_draw)):
                            raise ChampionEloError(
                                f"matchup posterior draws are non-finite for {scope_id} {role} {pair}"
                            )
                        interaction_draw = sign * canonical_draw
                        if (
                            posterior["effective_n"] >= MATCHUP_MIN_EFFECTIVE_MAPS
                            and posterior["effective_series"] >= MATCHUP_MIN_SERIES
                            and posterior["outcome_variation"]
                            and posterior["sd"] <= MATCHUP_MAX_POSTERIOR_SD
                        ):
                            supported.append(opponent)
                            effective_maps += posterior["effective_n"]
                            probability_positive = float(
                                np.mean(interaction_draw > COUNTER_EFFECT_THRESHOLD_LOGIT)
                            )
                            expected_counter_breadth += float(opponent_weight) * probability_positive
                            counter_breadth_draws += float(opponent_weight) * (
                                interaction_draw > COUNTER_EFFECT_THRESHOLD_LOGIT
                            )
                            if probability_positive >= COUNTER_POSTERIOR_THRESHOLD:
                                countered += 1
                                countered_weight += float(opponent_weight)
                    matchup_draw_columns.append(
                        expit(
                            strength_draws[:, champion_i]
                            - strength_draws[:, opponent_i]
                            + interaction_draw
                        )
                    )
                    map_probabilities.append(float(expit(role_mean[champion_i] - role_mean[opponent_i])))
                matchup_draw_matrix = np.column_stack(matchup_draw_columns)
                blind_draws = _weighted_lower_tail_rows(
                    matchup_draw_matrix,
                    opponent_weights,
                    BLIND_TAIL_SHARE,
                )
                standardized_strength = float(np.dot(opponent_weights, np.asarray(map_probabilities)))
            required_supported = len(opponent_indices)
            legal_components = {
                component_by_role_champion[(role, champions[index])]
                for index in opponent_indices
            }
            champion_component = component_by_role_champion[(role, champion)]
            component_identified = legal_components == {champion_component}
            contrast_sds = [
                math.sqrt(max(
                    0.0,
                    float(
                        role_covariance[champion_i, champion_i]
                        + role_covariance[opponent_i, opponent_i]
                        - 2.0 * role_covariance[champion_i, opponent_i]
                    ),
                ))
                for opponent_i in opponent_indices
            ]
            maximum_strength_contrast_sd = max(contrast_sds, default=float("inf"))
            available = (
                len(opponent_indices) == LEGAL_OPPONENT_COUNT
                and len(supported) == required_supported
                and component_identified
                and maximum_strength_contrast_sd <= STRENGTH_MAX_CONTRAST_SD
            )
            blind_score = float(np.quantile(blind_draws, 0.10))
            share = countered_weight if supported else 0.0
            alpha_sd = math.sqrt(float(role_covariance[champion_i, champion_i]))
            rows.append(
                {
                    "champion": champion,
                    "rating": round(INITIAL_RATING + role_mean[champion_i] * 400.0 / math.log(10.0), 4),
                    "tier_value_pp": round(100.0 * (standardized_strength - 0.5), 4),
                    "strength_score": round(standardized_strength, 6),
                    "strength_sd_logit": round(alpha_sd, 6),
                    "played_maps": states_by_role[role][champion].appearances,
                    "counterability_status": "available" if available else "unavailable",
                    "counterability": round(100.0 * (1.0 - blind_score), 4) if available else None,
                    "matchup_maps": round(effective_maps, 4),
                    "matchup_opponents": len(supported),
                    "blind_score_pp": round(100.0 * (blind_score - 0.5), 4) if available else None,
                    "counter_score": round(LEGAL_OPPONENT_COUNT * expected_counter_breadth, 4) if available else None,
                    "expected_counter_breadth": round(LEGAL_OPPONENT_COUNT * expected_counter_breadth, 4) if available else None,
                    "countered_opponent_count": countered if available else None,
                    "countered_opponent_share": round(share, 4) if available else None,
                    "legal_opponent_distribution_sha256": row_legal_hash,
                    "legal_opponents": row_legal_opponents,
                    "legal_opponent_coverage": 1.0 if available else round(len(supported) / max(1, required_supported), 4),
                    "strength_design_rank_full": design_rank_full,
                    "strength_design_condition_number": None if not math.isfinite(design_condition) else round(design_condition, 4),
                    "strength_component_identified": component_identified,
                    "maximum_strength_contrast_sd": (
                        round(maximum_strength_contrast_sd, 6)
                        if math.isfinite(maximum_strength_contrast_sd)
                        else None
                    ),
                    "_blind_draws": blind_draws,
                    "_counter_draws": LEGAL_OPPONENT_COUNT * counter_breadth_draws,
                }
            )
        rows.sort(key=lambda row: (-row["strength_score"], _normalize_name(row["champion"])))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        _assign_tier_buckets(rows)
        output[role] = {
            "rows": rows,
            "legal_opponents": legal_opponents,
            "legal_opponent_distribution_sha256": legal_hash,
            "design": {
                "rank": design_rank,
                "columns": design_columns,
                "rank_full": design_rank_full,
                "condition_number": None if not math.isfinite(design_condition) else round(design_condition, 4),
                "role_location_gauge": "sum_to_zero_per_connected_component",
                "connected_components_by_role": component_counts,
                "fit_coordinates": "orthonormal reduced contrasts",
            },
        }
    return output


def build_candidate(
    root: Path,
    *,
    as_of: pd.Timestamp | None = None,
    expected_live_as_of: pd.Timestamp | None = None,
    previous: Mapping[str, Any] | None = None,
    min_appearances: int = DEFAULT_MIN_APPEARANCES,
    source_mode: str = DEFAULT_SOURCE_MODE,
) -> dict[str, Any]:
    if min_appearances < 1:
        raise ChampionEloError("min_appearances must be at least 1")
    if source_mode not in SOURCE_MODES:
        raise ChampionEloError(
            f"source_mode must be one of {', '.join(SOURCE_MODES)}"
        )
    from .pooled_candidate import build_pooled_candidate

    return build_pooled_candidate(
        root,
        as_of=as_of,
        expected_live_as_of=expected_live_as_of,
        previous=previous,
        min_appearances=min_appearances,
        source_mode=source_mode,
    )


def write_candidate(path: Path, payload: Mapping[str, Any]) -> str:
    raw = _canonical_json(dict(payload)) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return _sha256_bytes(raw)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--as-of", default=None)
    parser.add_argument("--expected-live-as-of", default=None)
    parser.add_argument("--previous", type=Path, default=None)
    parser.add_argument("--min-appearances", type=int, default=DEFAULT_MIN_APPEARANCES)
    parser.add_argument(
        "--source-mode",
        choices=SOURCE_MODES,
        default=DEFAULT_SOURCE_MODE,
        help="source provenance mode for this replay",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    as_of = pd.Timestamp(args.as_of) if args.as_of else None
    expected = pd.Timestamp(args.expected_live_as_of) if args.expected_live_as_of else None
    previous = json.loads(args.previous.read_text()) if args.previous else None
    payload = build_candidate(
        args.root,
        as_of=as_of,
        expected_live_as_of=expected,
        previous=previous,
        min_appearances=args.min_appearances,
        source_mode=args.source_mode,
    )
    raw_sha = write_candidate(args.out, payload)
    print(json.dumps({
        "out": str(args.out),
        "raw_sha256": raw_sha,
        "artifact_sha256": payload["artifact_sha256"],
        "as_of": payload["as_of"],
        "maps_replayed": payload["source"]["maps_replayed"],
        "maps_in_live_window": payload["source"]["maps_in_live_window"],
        "source_mode": payload["source_mode"],
        "cells": len(payload["cells"]),
        "source_complete_through_expected_live_as_of": payload["source_complete_through_expected_live_as_of"],
        "unresolved_champion_identities": payload["unresolved_champion_identities"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
