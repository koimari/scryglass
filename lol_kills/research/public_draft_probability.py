"""Research-only composition probability replay.

The draft component contains antisymmetric pick and interaction terms.  It has
zero intercept.  Pre-event side, patch, region, and tournament context uses a
separate feature block.  Team strength, player strength, observed state, and
outcome-derived controls are outside the contract.

The 2026-04-23 through 2026-08-15 holdout was consumed by the first research
run.  This module can replay one development-selected configuration on that
block for reproducibility diagnostics.  Such a replay cannot promote a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
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
from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.research.composition_signal import (
    ROLES,
    build_composition_games,
)


SCHEMA_VERSION = "scryglass:public-draft-probability-research:v2"
MODEL_VERSION = "public-draft-probability-composition-v2"
PUBLIC_PATCH_MAP = {"16.16": "26.16"}
DEFAULT_HOLDOUT_FRACTION = 0.20
DEFAULT_BOOTSTRAP_REPS = 500
DEFAULT_SEED = 461
CONSUMED_HOLDOUT_START = "2026-04-23T05:08:13Z"

# This map defines pre-event regional context.  It does not encode team strength.
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

COMPOSITION_PREFIXES = (
    "CH|",
    "ALLY|",
    "ALLYCH|",
    "CTR|",
    "SAME|",
    "ATOM|",
    "ATOMALLY|",
    "ATOMCTR|",
)
CONTEXT_PREFIXES = (
    "CTX|SIDE|",
    "CTX|REGION|",
    "CTX|SCOPE|",
    "CTX|EVENT|",
    "CTX|TIER|",
    "CTX|TOURNAMENT|",
    "CTX|PATCH-MAJOR|",
    "CTX|PATCH|",
)
FEATURE_KEY_RE = re.compile(r"^[A-Z0-9_.:-]+(?:\|[A-Z0-9_.:-]+)+$")


@dataclass(frozen=True)
class CandidateConfig:
    name: str
    champion_role: bool = True
    ally: bool = True
    counter: bool = True
    same_role: bool = True
    atoms: bool = True
    atom_interactions: bool = True
    context: bool = True
    region_context: bool = True
    patch_context: bool = True
    tournament_context: bool = True
    support: int = 20
    c: float = 0.10


@dataclass(frozen=True)
class PreparedGame:
    game: Mapping[str, Any]
    region: str
    scope: str
    event_kind: str
    competition_tier: str
    tournament: str
    roster_change: bool
    series_cluster: str
    atom_status: str
    atom_snapshot_patch: str | None


@dataclass(frozen=True)
class AtomRouter:
    vectors_by_source_patch: Mapping[str, Mapping[str, np.ndarray]]
    vector_names: tuple[str, ...]
    status_by_source_patch: Mapping[str, str]
    snapshot_by_source_patch: Mapping[str, str]
    snapshot_meta: Mapping[str, Mapping[str, Any]]


_WORKER_ITEMS: Sequence[PreparedGame] | None = None
_WORKER_FOLDS: Sequence[tuple[list[PreparedGame], list[PreparedGame]]] | None = None
_WORKER_ATOM_ROUTER: AtomRouter | None = None


def _worker_init(
    items: Sequence[PreparedGame],
    folds: Sequence[tuple[list[PreparedGame], list[PreparedGame]]],
    atom_router: AtomRouter,
) -> None:
    global _WORKER_ITEMS, _WORKER_FOLDS, _WORKER_ATOM_ROUTER
    _WORKER_ITEMS = items
    _WORKER_FOLDS = folds
    _WORKER_ATOM_ROUTER = atom_router


def _dev_candidate_worker(config: CandidateConfig) -> dict[str, Any]:
    if _WORKER_ITEMS is None or _WORKER_FOLDS is None or _WORKER_ATOM_ROUTER is None:
        raise RuntimeError("research worker state is not initialized")
    return _dev_candidate(config, _WORKER_ITEMS, _WORKER_FOLDS, _WORKER_ATOM_ROUTER)


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


def _key_component(value: Any, fallback: str = "UNKNOWN") -> str:
    text = str(value or "").strip().upper()
    text = re.sub(r"[^A-Z0-9_.:-]+", "_", text).strip("_")
    return text[:160] or fallback


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


def _atom_feature_maps(payload: Mapping[str, Any]) -> tuple[dict[str, dict[str, float]], list[str], str]:
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
    vectors: dict[str, dict[str, float]] = {}
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
        vectors[_atom_champion_key(champion.get("display_name"))] = {
            name: value for name, value in zip(vector_names, values)
        }
    return vectors, vector_names, str(payload.get("artifact_sha256") or "")


def _resolve_atom_locator(manifest_path: Path, locator: Any) -> Path:
    path = Path(str(locator or ""))
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    for parent in manifest_path.parents:
        candidate = parent / path
        if candidate.exists():
            return candidate
    return cwd_path


def load_atom_router(manifest_path: Path) -> AtomRouter:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    snapshots = {
        str(row.get("patch") or ""): row
        for row in manifest.get("atom_snapshots", [])
        if row.get("patch")
    }
    raw_by_source: dict[str, dict[str, dict[str, float]]] = {}
    status_by_source: dict[str, str] = {}
    snapshot_by_source: dict[str, str] = {}
    snapshot_meta: dict[str, dict[str, Any]] = {}
    all_names: set[str] = set()
    loaded: dict[str, tuple[dict[str, dict[str, float]], list[str], str, str, Path]] = {}
    for mapping in manifest.get("mappings", []):
        source_patch = _patch_token(mapping.get("oe_token"))
        snapshot_patch = str(mapping.get("atom_snapshot_patch") or "")
        if mapping.get("ambiguity_status") != "none" or not snapshot_patch:
            status_by_source[source_patch] = str(mapping.get("ambiguity_status") or "snapshot_unavailable")
            continue
        snapshot = snapshots.get(snapshot_patch)
        if not snapshot:
            status_by_source[source_patch] = "snapshot_registration_missing"
            continue
        if snapshot_patch not in loaded:
            source = snapshot.get("source") or {}
            path = _resolve_atom_locator(manifest_path, source.get("locator"))
            if not path.is_file():
                status_by_source[source_patch] = "snapshot_file_missing"
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            maps, names, artifact_sha = _atom_feature_maps(payload)
            expected_sha = str(snapshot.get("bridge_artifact_sha256") or "")
            loaded[snapshot_patch] = (maps, names, artifact_sha, expected_sha, path)
        maps, names, artifact_sha, expected_sha, path = loaded[snapshot_patch]
        if not expected_sha or artifact_sha != expected_sha:
            status_by_source[source_patch] = "snapshot_artifact_hash_mismatch"
            continue
        raw_by_source[source_patch] = maps
        all_names.update(names)
        status_by_source[source_patch] = "available"
        snapshot_by_source[source_patch] = snapshot_patch
        snapshot_meta[snapshot_patch] = {
            "artifact_sha256": artifact_sha,
            "file_sha256": _sha256(path),
            "locator": str(path),
        }
    vector_names = tuple(sorted(all_names))
    vectors_by_source: dict[str, dict[str, np.ndarray]] = {}
    for source_patch, champion_maps in raw_by_source.items():
        vectors_by_source[source_patch] = {
            champion: np.asarray([values.get(name, 0.0) for name in vector_names], dtype=float)
            for champion, values in champion_maps.items()
        }
    return AtomRouter(
        vectors_by_source_patch=vectors_by_source,
        vector_names=vector_names,
        status_by_source_patch=status_by_source,
        snapshot_by_source_patch=snapshot_by_source,
        snapshot_meta=snapshot_meta,
    )


def _safe_vector(router: AtomRouter, source_patch: str, champion: Any) -> np.ndarray:
    vectors = router.vectors_by_source_patch.get(source_patch)
    if vectors is None:
        return np.zeros(len(router.vector_names), dtype=float)
    return vectors.get(_atom_champion_key(champion), np.zeros(len(router.vector_names), dtype=float))


def _side_champions(game: Mapping[str, Any], side: str) -> list[tuple[str, str]]:
    return [(role, normalize_champ(str(game[side][role].get("champion") or ""))) for role in ROLES]


def _canonical_pair(left: tuple[str, str], right: tuple[str, str]) -> tuple[tuple[str, str], tuple[str, str], float]:
    if left <= right:
        return left, right, 1.0
    return right, left, -1.0


def _roster_change_flags(games: Sequence[Mapping[str, Any]]) -> dict[str, bool]:
    last: dict[str, frozenset[str]] = {}
    result: dict[str, bool] = {}
    ordered = sorted(games, key=lambda game: (pd.Timestamp(game["date"]), str(game["game_uid"])))
    index = 0
    while index < len(ordered):
        timestamp = pd.Timestamp(ordered[index]["date"])
        stop = index
        while stop < len(ordered) and pd.Timestamp(ordered[stop]["date"]) == timestamp:
            stop += 1
        pending: dict[str, set[frozenset[str]]] = {}
        for game in ordered[index:stop]:
            changed = False
            for side in ("blue", "red"):
                team = str(game.get(f"{side}_team") or "")
                roster = frozenset(str(game[side][role].get("player") or "") for role in ROLES)
                previous = last.get(team)
                if previous is not None and roster != previous:
                    changed = True
                pending.setdefault(team, set()).add(roster)
            result[str(game.get("game_uid"))] = changed
        for team, rosters in pending.items():
            if len(rosters) == 1:
                last[team] = next(iter(rosters))
            else:
                last.pop(team, None)
        index = stop
    return result


def _first_nonempty(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype=object)
    for column in columns:
        if column not in frame.columns:
            continue
        candidate = frame[column].fillna("").astype(str).str.strip()
        mask = result.astype(str).str.strip().eq("") & candidate.ne("") & candidate.ne("<NA>")
        result.loc[mask] = candidate.loc[mask]
    return result


def canonicalize_player_maps(players: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int | str]]:
    if "game_uid" not in players.columns:
        raise ValueError("the source snapshot needs game_uid")
    frame = players.copy()
    frame["_raw_game_uid"] = frame["game_uid"].fillna("").astype(str)
    fallback = frame["gameid"] if "gameid" in frame.columns else pd.Series("", index=frame.index)
    frame["_canonical_game_uid"] = [
        canonical_source_game_key(value, fallback.loc[index])
        for index, value in frame["game_uid"].items()
    ]
    frame["_stable_player_id"] = _first_nonempty(frame, ("playerid", "player_id"))
    frame["_stable_team_id"] = _first_nonempty(frame, ("teamid", "team_id", "teamname"))
    frame["_side_key"] = frame["side"].fillna("").astype(str).str.title()
    role_aliases = {"jungle": "jng", "support": "sup", "utility": "sup", "bottom": "bot", "adc": "bot"}
    frame["_role_key"] = frame["position"].fillna("").astype(str).str.casefold().replace(role_aliases)
    frame["_slot_key"] = frame["_side_key"] + "|" + frame["_role_key"]
    metadata_columns = [column for column in ("date", "league", "patch", "oe_patch_token") if column in frame.columns]
    signature_columns = [
        "_slot_key",
        "_stable_player_id",
        "_stable_team_id",
        "champion",
        "result",
        *metadata_columns,
    ]
    deduped = frame.drop_duplicates(signature_columns, keep="first").copy()
    duplicate_rows_removed = len(frame) - len(deduped)
    deduped["_champion_key"] = deduped["champion"].map(normalize_champ)
    valid_slots = {f"{side}|{role}" for side in ("Blue", "Red") for role in ROLES}
    deduped["_valid_slot"] = deduped["_slot_key"].isin(valid_slots)
    grouped = deduped.groupby("_canonical_game_uid", sort=False, dropna=False)
    stats = grouped.agg(
        rows=("_slot_key", "size"),
        slots=("_slot_key", "nunique"),
        valid_slots=("_valid_slot", "all"),
        players=("_stable_player_id", "nunique"),
        champions=("_champion_key", "nunique"),
    )
    missing_by_map = frame["_stable_player_id"].astype(str).str.strip().eq("").groupby(frame["_canonical_game_uid"]).any()
    missing_ids = set(missing_by_map[missing_by_map].index.astype(str))
    accepted_ids = set(
        stats[
            (stats.index.astype(str) != "")
            & stats["rows"].eq(10)
            & stats["slots"].eq(10)
            & stats["valid_slots"]
            & stats["players"].eq(10)
            & stats["champions"].eq(10)
        ].index.astype(str)
    ) - missing_ids
    output = deduped[deduped["_canonical_game_uid"].astype(str).isin(accepted_ids)].copy()
    output = output.sort_values(["_canonical_game_uid", "_slot_key"]).reset_index(drop=True)
    maps_seen = int(frame["_canonical_game_uid"].nunique())
    missing_identity = len(missing_ids)
    ambiguous = max(0, maps_seen - len(accepted_ids) - missing_identity)
    output["game_uid"] = output["_canonical_game_uid"]
    output["playername"] = output["_stable_player_id"]
    output["teamname"] = output["_stable_team_id"]
    return output, {
        "stable_player_id_columns": "playerid,player_id",
        "canonical_maps_seen": maps_seen,
        "canonical_maps_accepted": int(output["game_uid"].nunique()) if not output.empty else 0,
        "ambiguous_maps_excluded": int(ambiguous),
        "missing_stable_player_id_maps_excluded": int(missing_identity),
        "duplicate_rows_removed": int(duplicate_rows_removed),
    }


def _series_cluster(game: Mapping[str, Any]) -> str:
    series_id = str(game.get("series_id") or "").strip()
    if series_id:
        return f"series:{series_id}"
    teams = sorted((str(game.get("blue_team") or ""), str(game.get("red_team") or "")))
    date = pd.Timestamp(game["date"]).strftime("%Y-%m-%d")
    return f"derived:{date}:{game.get('league', 'UNKNOWN')}:{teams[0]}:{teams[1]}"


def prepare_games(players: pd.DataFrame, atom_manifest_path: Path) -> tuple[list[PreparedGame], dict[str, Any], AtomRouter]:
    if "game_uid" not in players.columns:
        raise ValueError("the source snapshot needs game_uid")
    source_rows = len(players)
    frame, dedup_meta = canonicalize_player_maps(players)
    if "oe_patch_token" in frame.columns:
        frame["patch"] = frame["oe_patch_token"].where(frame["oe_patch_token"].notna(), frame.get("patch"))
    frame["patch"] = frame["patch"].map(_patch_token)
    atom_router = load_atom_router(atom_manifest_path)
    games = build_composition_games(frame)
    games = sorted(games, key=lambda item: (pd.Timestamp(item["date"]), str(item["game_uid"])))
    metadata = _metadata_by_game(frame)
    roster_changes = _roster_change_flags(games)
    prepared: list[PreparedGame] = []
    for game in games:
        row = metadata.get(str(game["game_uid"]), {})
        patch = _patch_token(game.get("patch"))
        game = dict(game)
        game["patch"] = patch
        scope = str(row.get("competition_scope") or "UNKNOWN").strip().upper()
        event_kind = str(row.get("event_kind") or "UNKNOWN").strip().upper()
        competition_tier = str(row.get("competition_tier") or "UNKNOWN").strip().upper()
        tournament = str(row.get("tournament") or "UNKNOWN").strip()
        region = _region(row)
        atom_status = atom_router.status_by_source_patch.get(patch, "source_patch_unregistered")
        prepared.append(
            PreparedGame(
                game=game,
                region=region,
                scope=scope,
                event_kind=event_kind,
                competition_tier=competition_tier,
                tournament=tournament,
                roster_change=roster_changes.get(str(game["game_uid"]), False),
                series_cluster=_series_cluster(game),
                atom_status=atom_status,
                atom_snapshot_patch=atom_router.snapshot_by_source_patch.get(patch),
            )
        )
    source_meta = {
        "source_rows": int(source_rows),
        "source_games": int(frame["game_uid"].nunique()),
        "prepared_games": int(len(prepared)),
        "date_min": str(frame["date"].min()) if not frame.empty else None,
        "date_max": str(frame["date"].max()) if not frame.empty else None,
        "patch_counts": {
            str(k): int(v)
            for k, v in frame.drop_duplicates("game_uid")["patch"].value_counts().sort_index().items()
        },
        "source_patch_public_map": PUBLIC_PATCH_MAP,
        "atom_routing": {
            "manifest": str(atom_manifest_path),
            "status_by_source_patch": dict(atom_router.status_by_source_patch),
            "snapshot_by_source_patch": dict(atom_router.snapshot_by_source_patch),
            "snapshots": dict(atom_router.snapshot_meta),
        },
        **dedup_meta,
    }
    return prepared, source_meta, atom_router


def _composition_tokens(
    item: PreparedGame,
    config: CandidateConfig,
    atom_router: AtomRouter,
) -> list[tuple[str, float]]:
    game = item.game
    tokens: list[tuple[str, float]] = []
    blue = _side_champions(game, "blue")
    red = _side_champions(game, "red")
    if config.champion_role:
        for sign, side in ((1.0, blue), (-1.0, red)):
            for role, champion in side:
                tokens.append((f"CH|{_key_component(role)}|{_key_component(champion)}", sign))
    if config.ally:
        for sign, side in ((1.0, blue), (-1.0, red)):
            for left, right in combinations(side, 2):
                first, second = sorted((left, right))
                tokens.append((f"ALLY|{_key_component(first[0])}:{_key_component(second[0])}|{_key_component(first[1])}|{_key_component(second[1])}", sign))
                tokens.append((f"ALLYCH|{_key_component(first[1])}|{_key_component(second[1])}", sign))
    if config.counter or config.same_role:
        for blue_pick, red_pick in product_pairs(blue, red):
            first, second, sign = _canonical_pair(blue_pick, red_pick)
            if config.counter:
                tokens.append((f"CTR|{_key_component(first[0])}:{_key_component(second[0])}|{_key_component(first[1])}|{_key_component(second[1])}", sign))
            if config.same_role and blue_pick[0] == red_pick[0]:
                tokens.append((f"SAME|{_key_component(blue_pick[0])}|{_key_component(first[1])}|{_key_component(second[1])}", sign))
    source_patch = _patch_token(game.get("patch"))
    if config.atoms and item.atom_status == "available":
        atom_names = atom_router.vector_names
        blue_vectors = np.asarray([_safe_vector(atom_router, source_patch, champion) for _, champion in blue])
        red_vectors = np.asarray([_safe_vector(atom_router, source_patch, champion) for _, champion in red])
        blue_sum = blue_vectors.sum(axis=0)
        red_sum = red_vectors.sum(axis=0)
        for name, value in zip(atom_names, (blue_sum - red_sum) / 5.0):
            if value:
                tokens.append((f"ATOM|{_key_component(name)}", float(value)))
        if config.atom_interactions:
            name_to_index = {name: index for index, name in enumerate(atom_names)}
            family_indices = [name_to_index[f"family:{family}"] for family in ATOM_FAMILIES]
            blue_family = blue_sum[family_indices]
            red_family = red_sum[family_indices]
            for i in range(len(ATOM_FAMILIES)):
                for j in range(i, len(ATOM_FAMILIES)):
                    ally_value = float(blue_family[i] * blue_family[j] - red_family[i] * red_family[j]) / 25.0
                    cross_value = float(blue_family[i] * red_family[j] - red_family[i] * blue_family[j]) / 25.0
                    if ally_value:
                        tokens.append((f"ATOMALLY|{_key_component(ATOM_FAMILIES[i])}|{_key_component(ATOM_FAMILIES[j])}", ally_value))
                    if cross_value:
                        tokens.append((f"ATOMCTR|{_key_component(ATOM_FAMILIES[i])}|{_key_component(ATOM_FAMILIES[j])}", cross_value))
    return tokens


def _context_tokens(item: PreparedGame, config: CandidateConfig) -> list[tuple[str, float]]:
    if not config.context:
        return []
    tokens = [("CTX|SIDE|BLUE", 1.0)]
    if config.region_context:
        tokens.extend(
            [
                (f"CTX|REGION|{_key_component(item.region)}", 1.0),
                (f"CTX|SCOPE|{_key_component(item.scope)}", 1.0),
            ]
        )
    if config.tournament_context:
        tokens.extend(
            [
                (f"CTX|EVENT|{_key_component(item.event_kind)}", 1.0),
                (f"CTX|TIER|{_key_component(item.competition_tier)}", 1.0),
                (f"CTX|TOURNAMENT|{_key_component(item.tournament)}", 1.0),
            ]
        )
    if config.patch_context:
        patch = _patch_token(item.game.get("patch"))
        tokens.extend(
            [
                (f"CTX|PATCH-MAJOR|{_key_component(patch.split('.')[0])}", 1.0),
                (f"CTX|PATCH|{_key_component(patch)}", 1.0),
            ]
        )
    return tokens


def product_pairs(left: Sequence[tuple[str, str]], right: Sequence[tuple[str, str]]) -> Iterable[tuple[tuple[str, str], tuple[str, str]]]:
    for first in left:
        for second in right:
            yield first, second


def _validate_feature_rows(
    rows: Sequence[Sequence[tuple[str, float]]],
    allowed_prefixes: Sequence[str],
) -> None:
    for row in rows:
        for key, value in row:
            if not any(key.startswith(prefix) for prefix in allowed_prefixes):
                raise AssertionError(f"feature key is outside the allowlist: {key}")
            if not FEATURE_KEY_RE.fullmatch(key):
                raise AssertionError(f"feature key is malformed: {key}")
            if not np.isfinite(float(value)):
                raise AssertionError(f"feature value is not finite: {key}")
            segments = {segment.casefold() for segment in key.split("|")}
            if segments.intersection(FORBIDDEN_FEATURE_TERMS):
                raise AssertionError(f"forbidden feature term reached matrix: {key}")


def _feature_blocks(
    items: Sequence[PreparedGame],
    config: CandidateConfig,
    atom_router: AtomRouter,
) -> tuple[list[list[tuple[str, float]]], list[list[tuple[str, float]]]]:
    composition = [_composition_tokens(item, config, atom_router) for item in items]
    context = [_context_tokens(item, config) for item in items]
    _validate_feature_rows(composition, COMPOSITION_PREFIXES)
    _validate_feature_rows(context, CONTEXT_PREFIXES)
    return composition, context


def _vocabulary(rows: Sequence[Sequence[tuple[str, float]]], support: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        for key in {key for key, _ in row}:
            counts[key] = counts.get(key, 0) + 1
    keys = sorted(key for key, count in counts.items() if count >= support or key.startswith(("ATOM|", "ATOMALLY|", "ATOMCTR|", "CTX|SIDE|")))
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


def _series_cluster_bootstrap(
    items: Sequence[PreparedGame],
    p: np.ndarray,
    reps: int,
    seed: int,
) -> dict[str, Any]:
    y = np.asarray([int(item.game["y"]) for item in items], dtype=float)
    if len(y) < 30 or len(np.unique(y)) < 2 or reps <= 0:
        return {"unit": "series", "clusters": 0, "reps": 0, "auc": None, "brier": None, "log_loss": None}
    cluster_rows: dict[str, list[int]] = {}
    for index, item in enumerate(items):
        cluster_rows.setdefault(item.series_cluster, []).append(index)
    clusters = sorted(cluster_rows)
    rng = np.random.default_rng(seed)
    auc_values: list[float] = []
    brier_values: list[float] = []
    loss_values: list[float] = []
    for _ in range(reps):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        indices = np.asarray(
            [index for cluster_index in sampled for index in cluster_rows[clusters[int(cluster_index)]]],
            dtype=int,
        )
        yy = y[indices]
        pp = p[indices]
        if len(np.unique(yy)) < 2:
            continue
        auc_values.append(float(roc_auc_score(yy, pp)))
        brier_values.append(float(np.mean((yy - pp) ** 2)))
        loss_values.append(float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log1p(-pp))))

    def interval(values: Sequence[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "lower": None, "upper": None}
        q = np.percentile(values, [2.5, 50.0, 97.5])
        return {"median": float(q[1]), "lower": float(q[0]), "upper": float(q[2])}

    return {
        "unit": "series",
        "clusters": len(clusters),
        "reps": int(len(auc_values)),
        "auc": interval(auc_values),
        "brier": interval(brier_values),
        "log_loss": interval(loss_values),
    }


def _fit_predict(
    train: Sequence[PreparedGame],
    validation: Sequence[PreparedGame],
    config: CandidateConfig,
    atom_router: AtomRouter,
    row_lookup: Mapping[str, tuple[Sequence[tuple[str, float]], Sequence[tuple[str, float]]]] | None = None,
) -> dict[str, Any]:
    if row_lookup is None:
        train_draft, train_context = _feature_blocks(train, config, atom_router)
        validation_draft, validation_context = _feature_blocks(validation, config, atom_router)
    else:
        train_pairs = [row_lookup[str(item.game["game_uid"])] for item in train]
        validation_pairs = [row_lookup[str(item.game["game_uid"])] for item in validation]
        train_draft = [pair[0] for pair in train_pairs]
        train_context = [pair[1] for pair in train_pairs]
        validation_draft = [pair[0] for pair in validation_pairs]
        validation_context = [pair[1] for pair in validation_pairs]
    draft_vocabulary = _vocabulary(train_draft, config.support)
    context_vocabulary = _vocabulary(train_context, min(config.support, 5))
    x_train_draft = _matrix(train_draft, draft_vocabulary)
    x_validation_draft = _matrix(validation_draft, draft_vocabulary)
    x_train_context = _matrix(train_context, context_vocabulary)
    x_validation_context = _matrix(validation_context, context_vocabulary)
    x_train = sparse.hstack((x_train_draft, x_train_context), format="csr")
    x_validation = sparse.hstack((x_validation_draft, x_validation_context), format="csr")
    model = LogisticRegression(
        C=float(config.c),
        solver="liblinear",
        max_iter=1200,
        random_state=DEFAULT_SEED,
        fit_intercept=False,
    )
    model.fit(x_train, [int(item.game["y"]) for item in train])
    coefficients = model.coef_[0]
    draft_width = len(draft_vocabulary)
    draft_logits = np.asarray(x_validation_draft @ coefficients[:draft_width]).reshape(-1)
    context_logits = np.asarray(x_validation_context @ coefficients[draft_width:]).reshape(-1)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(draft_logits + context_logits, -35.0, 35.0)))
    return {
        "probabilities": probabilities,
        "draft_logits": draft_logits,
        "context_logits": context_logits,
        "draft_vocabulary_size": len(draft_vocabulary),
        "context_vocabulary_size": len(context_vocabulary),
        "draft_intercept": 0.0,
    }


def _timestamp_cluster_starts(items: Sequence[PreparedGame]) -> list[int]:
    starts: list[int] = []
    last: pd.Timestamp | None = None
    for index, item in enumerate(items):
        timestamp = pd.Timestamp(item.game["date"])
        if last is None or timestamp != last:
            starts.append(index)
            last = timestamp
    starts.append(len(items))
    return starts


def _boundary_near(items: Sequence[PreparedGame], proportion: float) -> int:
    starts = _timestamp_cluster_starts(items)
    if len(starts) < 3:
        raise ValueError("at least two timestamp batches are required")
    target = float(len(items)) * proportion
    return min(starts[1:-1], key=lambda value: (abs(value - target), value))


def fixed_chronological_folds(
    items: Sequence[PreparedGame],
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    holdout_start: str | pd.Timestamp | None = None,
) -> tuple[list[PreparedGame], list[PreparedGame], list[tuple[list[PreparedGame], list[PreparedGame]]]]:
    if not 0.05 <= holdout_fraction <= 0.40:
        raise ValueError("holdout_fraction must be between 0.05 and 0.40")
    if holdout_start is None:
        boundary = _boundary_near(items, 1.0 - holdout_fraction)
    else:
        cutoff = pd.Timestamp(holdout_start)
        if cutoff.tzinfo is None:
            cutoff = cutoff.tz_localize("UTC")
        boundary = next(
            (index for index, item in enumerate(items) if pd.Timestamp(item.game["date"]) >= cutoff),
            len(items),
        )
        if boundary <= 0 or boundary >= len(items):
            raise ValueError("the fixed holdout start does not divide the prepared games")
    development = list(items[:boundary])
    final_holdout = list(items[boundary:])
    dev_boundaries = [0]
    dev_boundaries.extend(_boundary_near(development, fraction) for fraction in (0.25, 0.50, 0.75))
    dev_boundaries.append(len(development))
    folds: list[tuple[list[PreparedGame], list[PreparedGame]]] = []
    for index in range(1, len(dev_boundaries) - 1):
        train = development[: dev_boundaries[index]]
        validation = development[dev_boundaries[index] : dev_boundaries[index + 1]]
        if train and validation:
            folds.append((train, validation))
    return development, final_holdout, folds


def _dev_candidate(config: CandidateConfig, items: Sequence[PreparedGame], folds: Sequence[tuple[list[PreparedGame], list[PreparedGame]]], atom_router: AtomRouter) -> dict[str, Any]:
    draft_rows, context_rows = _feature_blocks(items, config, atom_router)
    row_lookup = {
        str(item.game["game_uid"]): (draft_row, context_row)
        for item, draft_row, context_row in zip(items, draft_rows, context_rows)
    }
    fold_metrics: list[dict[str, Any]] = []
    vocabulary_sizes: list[dict[str, int]] = []
    for index, (train, validation) in enumerate(folds, 1):
        prediction = _fit_predict(train, validation, config, atom_router, row_lookup)
        fold_metrics.append({"fold": index, **metrics([int(item.game["y"]) for item in validation], prediction["probabilities"])})
        vocabulary_sizes.append({
            "draft": prediction["draft_vocabulary_size"],
            "context": prediction["context_vocabulary_size"],
        })
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
        elif key == "competition_tier":
            values.append(item.competition_tier)
        elif key == "scope":
            values.append(item.scope)
        elif key == "atom_status":
            values.append(item.atom_status)
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


def _sparse_bucket(items: Sequence[PreparedGame], train: Sequence[PreparedGame], config: CandidateConfig, atom_router: AtomRouter) -> np.ndarray:
    rows, _ = _feature_blocks(list(train) + list(items), config, atom_router)
    train_rows = rows[: len(train)]
    item_rows = rows[len(train) :]
    vocabulary = _vocabulary(train_rows, config.support)
    values: list[float] = []
    for row in item_rows:
        static = [key for key, _ in row if key.startswith(("CH|", "ALLY|", "ALLYCH|", "CTR|", "SAME|"))]
        values.append(float(sum(key not in vocabulary for key in static)))
    return np.asarray(values)


def candidate_configs(*, include_atoms: bool = False) -> list[CandidateConfig]:
    configs = [
        CandidateConfig("role_champion_context_free", ally=False, counter=False, same_role=False, atoms=False, atom_interactions=False, context=False, c=0.10),
        CandidateConfig("role_champion_context", ally=False, counter=False, same_role=False, atoms=False, atom_interactions=False, context=True, c=0.10),
        CandidateConfig("role_ally", ally=True, counter=False, same_role=False, atoms=False, atom_interactions=False, context=True, c=0.03),
        CandidateConfig("role_counter_same", ally=False, counter=True, same_role=True, atoms=False, atom_interactions=False, context=True, c=0.03),
        CandidateConfig("full_exact", atoms=False, atom_interactions=False, context=True, c=0.03),
        CandidateConfig("full_exact_strict", atoms=False, atom_interactions=False, context=True, support=40, c=0.03),
        CandidateConfig("full_exact_regularized", atoms=False, atom_interactions=False, context=True, support=20, c=0.01),
    ]
    if include_atoms:
        configs.extend(
            [
                CandidateConfig("role_champion_atoms", ally=False, counter=False, same_role=False, atoms=True, atom_interactions=True, context=True, c=0.03),
                CandidateConfig("full_atoms", atoms=True, atom_interactions=True, context=True, c=0.03),
            ]
        )
    signatures: set[tuple[Any, ...]] = set()
    for config in configs:
        signature = tuple(value for key, value in config.__dict__.items() if key != "name")
        if signature in signatures:
            raise AssertionError(f"duplicate candidate configuration: {config.name}")
        signatures.add(signature)
    return configs


def run_experiment(
    players_path: Path,
    atom_manifest_path: Path,
    *,
    output_dir: Path,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    max_workers: int | None = None,
    seed: int = DEFAULT_SEED,
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION,
    replay_consumed_holdout: bool = False,
    holdout_start: str | None = None,
) -> dict[str, Any]:
    players = pd.read_parquet(players_path)
    items, source_meta, atom_router = prepare_games(players, atom_manifest_path)
    effective_holdout_start = holdout_start
    if replay_consumed_holdout and effective_holdout_start is None:
        effective_holdout_start = CONSUMED_HOLDOUT_START
    development, final_holdout, folds = fixed_chronological_folds(
        items,
        holdout_fraction,
        effective_holdout_start,
    )
    atom_development_n = sum(item.atom_status == "available" for item in development)
    atom_candidate_minimum = 100
    configs = candidate_configs(include_atoms=atom_development_n >= atom_candidate_minimum)
    dev_results: list[dict[str, Any]] = []
    workers = max_workers or min(8, os.cpu_count() or 1)
    # Token generation is Python-heavy.  Use processes so candidate searches
    # use independent cores instead of contending on the interpreter lock.
    process_context = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=process_context,
        initializer=_worker_init,
        initargs=(items, folds, atom_router),
    ) as executor:
        futures = {
            executor.submit(_dev_candidate_worker, config): config.name
            for config in configs
        }
        for future in as_completed(futures):
            dev_results.append(future.result())
    dev_results.sort(key=lambda row: (-(row["mean_auc"] or 0.0), row["mean_log_loss"] or 9.0, row["config"]["name"]))
    selected = CandidateConfig(**dev_results[0]["config"])
    diagnostic: dict[str, Any] = {
        "status": "not_run",
        "promotion_eligible": False,
        "reason": "the holdout is consumed; pass --replay-consumed-holdout for one reproducibility diagnostic",
    }
    if replay_consumed_holdout:
        prediction = _fit_predict(development, final_holdout, selected, atom_router)
        final_probabilities = prediction["probabilities"]
        final_y = np.asarray([int(item.game["y"]) for item in final_holdout], dtype=float)
        final_metrics = metrics(final_y, final_probabilities)
        final_metrics["series_cluster_bootstrap_95"] = _series_cluster_bootstrap(
            final_holdout,
            final_probabilities,
            bootstrap_reps,
            seed,
        )
        for key in ("region", "patch", "scope", "event_kind", "competition_tier", "atom_status", "roster_change"):
            final_metrics[f"by_{key}"] = _group_metrics(final_holdout, final_probabilities, key)
        sparse_count = _sparse_bucket(final_holdout, development, selected, atom_router)
        final_metrics["sparse_evidence"] = {
            "definition": "at least one exact composition term was unseen in development",
            "sparse_n": int((sparse_count > 0).sum()),
            "dense_n": int((sparse_count == 0).sum()),
            "sparse": metrics(final_y[sparse_count > 0], final_probabilities[sparse_count > 0]),
            "dense": metrics(final_y[sparse_count == 0], final_probabilities[sparse_count == 0]),
        }
        draft_probabilities = 1.0 / (1.0 + np.exp(-np.clip(prediction["draft_logits"], -35.0, 35.0)))
        context_probabilities = 1.0 / (1.0 + np.exp(-np.clip(prediction["context_logits"], -35.0, 35.0)))
        diagnostic = {
            "status": "consumed_holdout_reproducibility_only",
            "promotion_eligible": False,
            "selection_or_tuning_permitted": False,
            "config": selected.__dict__,
            "vocabulary_size": {
                "draft": prediction["draft_vocabulary_size"],
                "context": prediction["context_vocabulary_size"],
            },
            "draft_intercept": prediction["draft_intercept"],
            "metrics": final_metrics,
            "draft_component_metrics": metrics(final_y, draft_probabilities),
            "context_component_metrics": metrics(final_y, context_probabilities),
            "patch_transfer": {
                "source_patch": "16.16",
                "public_patch": "26.16",
                "metrics": final_metrics.get("by_patch", {}).get("16.16"),
                "atom_status": atom_router.status_by_source_patch.get("16.16", "source_patch_unregistered"),
            },
        }
    output = {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "authority": "unavailable",
        "public_probability": False,
        "estimand": "pre_match_composition_probability",
        "promotion_status": "unavailable_consumed_holdout",
        "allowed_inputs": [
            "ten champion picks and roles",
            "role-conditioned champion effects",
            "exact ally synergy",
            "exact enemy counters",
            "same-role terms",
            "source-patch-routed LCC atom interactions where a hash-bound snapshot is available",
            "separate pre-event side, patch, region, scope, event, tier, and tournament context",
        ],
        "excluded_inputs": sorted((*FORBIDDEN_FEATURE_TERMS, "player champion comfort without an identity receipt")),
        "component_contract": {
            "draft": "antisymmetric under blue-red swap with zero intercept",
            "context": "separate pre-event metadata block",
            "combined_probability": "sigmoid(draft_logit + context_logit)",
        },
        "source": {
            **source_meta,
            "players_path": str(players_path),
            "players_sha256": _sha256(players_path),
            "atom_manifest_path": str(atom_manifest_path),
            "atom_manifest_sha256": _sha256(atom_manifest_path),
        },
        "split": {
            "protocol": "equal-timestamp-batched chronological development and consumed final holdout",
            "requested_holdout_fraction": holdout_fraction,
            "fixed_holdout_start": effective_holdout_start,
            "observed_holdout_fraction": len(final_holdout) / len(items),
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
            "equal_timestamps_batched": True,
            "final_holdout_status": "consumed_before_this_revision",
        },
        "development_selection": {
            "candidate_count": len(configs),
            "workers": workers,
            "atom_candidates": {
                "status": "available" if atom_development_n >= atom_candidate_minimum else "unavailable",
                "reason": None if atom_development_n >= atom_candidate_minimum else "no supported atom snapshot coverage in development",
                "available_development_maps": atom_development_n,
                "minimum_development_maps": atom_candidate_minimum,
            },
            "results": dev_results,
            "selected": selected.__dict__,
        },
        "consumed_holdout_reproducibility": diagnostic,
    }
    output["reproducibility"] = {
        "command": "python3 -m lol_kills.research.public_draft_probability --players <path> --atom-manifest <path> --output-dir <path> --replay-consumed-holdout",
        "code_sha256": _sha256(Path(__file__)),
        "source_file_sha256": _sha256(players_path),
        "atom_manifest_sha256": _sha256(atom_manifest_path),
        "bootstrap_unit": "series",
        "comfort_status": "excluded_until_stable_identity_receipt_exists",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(json.dumps(_json_safe(output), indent=2, sort_keys=True), encoding="utf-8")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--players", type=Path, required=True)
    parser.add_argument("--atom-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=DEFAULT_BOOTSTRAP_REPS)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout-fraction", type=float, default=DEFAULT_HOLDOUT_FRACTION)
    parser.add_argument("--holdout-start", type=str, default=None)
    parser.add_argument("--replay-consumed-holdout", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = run_experiment(
        args.players,
        args.atom_manifest,
        output_dir=args.output_dir,
        bootstrap_reps=args.bootstrap_reps,
        max_workers=args.max_workers,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
        replay_consumed_holdout=args.replay_consumed_holdout,
        holdout_start=args.holdout_start,
    )
    print(
        json.dumps(
            {
                "selected": report["development_selection"]["selected"],
                "consumed_holdout_reproducibility": report["consumed_holdout_reproducibility"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
