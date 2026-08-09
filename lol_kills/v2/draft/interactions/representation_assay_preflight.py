"""Outcome-free empirical preflight for role-conditioned draft interactions.

This module deliberately stops before effect estimation or representation-rank
selection.  It asks whether the observed draft design has enough support,
temporal reuse, graph connectivity, and algebraic identifiability to justify a
later assay.  The unsigned incidence design is only a necessary support
diagnostic; it does not establish identifiability for a later signed,
antisymmetric counter model.  Only identity, draft, and source-completeness
columns are read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import structural_rank


SCHEMA_ID = "scryglass.draft-interaction-representation-assay-preflight.v1"
GENERATOR_VERSION = "representation-assay-preflight-generator.v2"
ROLE_ORDER = ("top", "jungle", "mid", "bot", "support")
ROLE_ALIASES = {"top": "top", "jng": "jungle", "mid": "mid", "bot": "bot", "sup": "support"}
SIDE_ALIASES = {"blue": "blue", "red": "red"}
MAP_COLUMNS = (
    "oe_gameid",
    "datacompleteness",
    "league",
    "year",
    "date",
    "patch",
    "competition_scope",
    "event_kind",
    "is_international",
)
PLAYER_COLUMNS = (
    "gameid",
    "datacompleteness",
    "league",
    "year",
    "date",
    "patch",
    "participantid",
    "side",
    "position",
    "champion",
)
MAP_ELIGIBILITY_FIELDS = (
    "oe_gameid",
    "league",
    "year",
    "date",
    "patch",
    "competition_scope",
    "event_kind",
    "is_international",
)
PLAYER_ELIGIBILITY_FIELDS = (
    "gameid",
    "league",
    "year",
    "date",
    "patch",
    "participantid",
    "side",
    "position",
    "champion",
)
DEFAULT_MAPS_PATH = Path("data/lol/warehouse/parquet/maps.parquet")
DEFAULT_PLAYERS_PATH = Path("data/lol/warehouse/parquet/oe_player_games.parquet")
DEFAULT_ARTIFACT_PATH = Path(
    "data/lol/v2/models/draft-interactions/representation-assay-preflight.json"
)
_TOP_LEVEL_FIELDS = {
    "schema_id",
    "development_only",
    "outcome_free",
    "predictive_authority",
    "representation_rank_selected",
    "authorizes_model_selection",
    "authorizes_publication",
    "content_addressing_confers_authority",
    "claim_ceiling",
    "generator",
    "source",
    "eligibility",
    "datacompleteness",
    "observation_contract",
    "support",
    "temporal_overlap",
    "graph_connectivity",
    "design_diagnostics",
    "artifact_sha256",
}


class RepresentationAssayPreflightError(ValueError):
    """Raised when inputs or an artifact violate the fail-closed contract."""


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def raw_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generator_identity() -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    return {
        "version": GENERATOR_VERSION,
        "executable_dependency_boundary": [
            {
                "locator": "lol_kills/v2/draft/interactions/representation_assay_preflight.py",
                "raw_sha256": raw_sha256(module_path),
            }
        ],
        "runtime_versions": {
            "numpy": importlib.metadata.version("numpy"),
            "pandas": importlib.metadata.version("pandas"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "scipy": importlib.metadata.version("scipy"),
        },
        "identity_scope": (
            "the generator module bytes plus numerical/dataframe runtime versions; "
            "source data bytes are pinned separately"
        ),
    }


def _canonical_scalar(value: object) -> object:
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def selected_input_sha256(frame: pd.DataFrame) -> str:
    """Hash selected input values independent of physical row order."""
    records = [
        [_canonical_scalar(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    records.sort(key=canonical_bytes)
    return canonical_sha256({"columns": list(frame.columns), "rows": records})


def _not_missing(value: object) -> bool:
    return not pd.isna(value) and str(value).strip() != ""


def _text(value: object) -> str:
    return str(value).strip()


def _patch_id(value: object) -> str:
    if isinstance(value, bool) or not _not_missing(value):
        raise RepresentationAssayPreflightError("patch is missing")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RepresentationAssayPreflightError("patch is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise RepresentationAssayPreflightError("patch is invalid")
    centesimal = round(number * 100)
    if abs(number * 100 - centesimal) > 1e-8:
        raise RepresentationAssayPreflightError(
            "patch must be an exact centesimal OE numeric token"
        )
    return f"{centesimal / 100:.2f}"


def _year_id(value: object) -> int:
    if isinstance(value, bool) or not _not_missing(value):
        raise RepresentationAssayPreflightError("year is missing")
    number = float(value)
    if not math.isfinite(number) or not number.is_integer():
        raise RepresentationAssayPreflightError("year is not integral")
    return int(number)


def _date_id(value: object) -> tuple[str, str]:
    if not _not_missing(value):
        raise RepresentationAssayPreflightError("date is missing")
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise RepresentationAssayPreflightError("date is not parseable") from exc
    if pd.isna(timestamp):
        raise RepresentationAssayPreflightError("date is missing")
    awareness = "aware" if timestamp.tzinfo is not None else "naive"
    return timestamp.isoformat(), awareness


def _node(champion: object, position: object) -> tuple[str, str]:
    if not _not_missing(champion):
        raise RepresentationAssayPreflightError("champion is missing")
    role = ROLE_ALIASES.get(_text(position).lower())
    if role is None:
        raise RepresentationAssayPreflightError("position is not a canonical role")
    return _text(champion), role


def _node_id(node: tuple[str, str]) -> str:
    return f"{node[0]}::{node[1]}"


def _canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def _role_pair(left: str, right: str) -> str:
    order = {role: index for index, role in enumerate(ROLE_ORDER)}
    return "|".join((left, right) if order[left] <= order[right] else (right, left))


def _required_completeness(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    rows = int(len(frame))
    fields = []
    for column in columns:
        present = int(frame[column].map(_not_missing).sum())
        fields.append(
            {
                "field": column,
                "present_rows": present,
                "missing_rows": rows - present,
                "present_fraction": (present / rows) if rows else None,
            }
        )
    return {"rows": rows, "fields": fields}


def _global_datacompleteness(frame: pd.DataFrame) -> list[dict[str, Any]]:
    counts = Counter(
        "<missing>" if not _not_missing(value) else _text(value)
        for value in frame["datacompleteness"].tolist()
    )
    return [{"label": label, "rows": counts[label]} for label in sorted(counts)]


@dataclass(frozen=True)
class _ValidMap:
    game_id: str
    year: int
    patch: str
    league: str
    competition_scope: str
    event_kind: str
    is_international: bool
    date: str
    date_timezone_awareness: str
    source_datacompleteness: str
    nodes_by_side: Mapping[str, tuple[tuple[str, str], ...]]


def _validate_map(
    game_id: str,
    map_rows: pd.DataFrame,
    player_rows: pd.DataFrame,
) -> tuple[_ValidMap | None, str | None]:
    if len(map_rows) != 1:
        return None, "map_registry_row_count_not_one"
    if len(player_rows) != 10:
        return None, "player_row_count_not_ten"
    map_row = next(map_rows.itertuples(index=False))
    try:
        year = _year_id(map_row.year)
        patch = _patch_id(map_row.patch)
        date, date_awareness = _date_id(map_row.date)
        league = _text(map_row.league) if _not_missing(map_row.league) else ""
    except RepresentationAssayPreflightError:
        return None, "invalid_map_metadata"
    if not league:
        return None, "invalid_map_metadata"
    if not _not_missing(map_row.oe_gameid) or _text(map_row.oe_gameid) != game_id:
        return None, "invalid_map_game_id"
    competition_scope = (
        _text(map_row.competition_scope)
        if _not_missing(map_row.competition_scope)
        else "unknown"
    )
    event_kind = _text(map_row.event_kind) if _not_missing(map_row.event_kind) else "unknown"
    if not isinstance(map_row.is_international, (bool, np.bool_)):
        return None, "invalid_competition_scope"
    is_international = bool(map_row.is_international)
    source_datacompleteness = (
        _text(map_row.datacompleteness)
        if _not_missing(map_row.datacompleteness)
        else "<missing>"
    )
    player_completeness = {
        _text(value) if _not_missing(value) else "<missing>"
        for value in player_rows["datacompleteness"]
    }
    if player_completeness != {source_datacompleteness}:
        return None, "map_player_datacompleteness_mismatch"

    for column in ("league", "year", "patch"):
        if any(not _not_missing(value) for value in player_rows[column]):
            return None, "missing_player_metadata"
    try:
        player_years = {_year_id(value) for value in player_rows["year"]}
        player_patches = {_patch_id(value) for value in player_rows["patch"]}
        player_dates = {_date_id(value) for value in player_rows["date"]}
    except RepresentationAssayPreflightError:
        return None, "invalid_player_metadata"
    player_leagues = {_text(value) for value in player_rows["league"]}
    if len(player_leagues) != 1:
        return None, "inconsistent_player_league_metadata"
    if (
        player_years != {year}
        or player_patches != {patch}
        or player_dates != {(date, date_awareness)}
    ):
        return None, "map_player_metadata_mismatch"
    if (
        not player_rows["participantid"].map(_not_missing).all()
        or player_rows["participantid"].duplicated().any()
    ):
        return None, "invalid_participant_ids"

    sides: dict[str, list[tuple[str, str]]] = {"blue": [], "red": []}
    try:
        for row in player_rows.itertuples(index=False):
            side = SIDE_ALIASES.get(_text(row.side).lower())
            if side is None:
                return None, "invalid_side"
            sides[side].append(_node(row.champion, row.position))
    except RepresentationAssayPreflightError:
        return None, "invalid_champion_or_role"
    for side in ("blue", "red"):
        if len(sides[side]) != 5 or {role for _, role in sides[side]} != set(ROLE_ORDER):
            return None, "side_missing_exact_role_roster"
        if len({_node_id(node) for node in sides[side]}) != 5:
            return None, "duplicate_role_conditioned_node"
    if len({champion for nodes in sides.values() for champion, _ in nodes}) != 10:
        return None, "duplicate_champion_across_map"
    return (
        _ValidMap(
            game_id=game_id,
            year=year,
            patch=patch,
            league=league,
            competition_scope=competition_scope,
            event_kind=event_kind,
            is_international=is_international,
            date=date,
            date_timezone_awareness=date_awareness,
            source_datacompleteness=source_datacompleteness,
            nodes_by_side={
                side: tuple(sorted(nodes, key=lambda item: ROLE_ORDER.index(item[1])))
                for side, nodes in sides.items()
            },
        ),
        None,
    )


def _quantiles(counter: Counter[Any]) -> dict[str, int | None]:
    values = sorted(counter.values())
    if not values:
        return {"minimum": None, "median": None, "maximum": None}
    return {
        "minimum": int(values[0]),
        "median": int(np.median(np.asarray(values))),
        "maximum": int(values[-1]),
    }


def _global_support(counter: Counter[Any]) -> dict[str, Any]:
    values = np.asarray(sorted(counter.values()), dtype=int)
    return {
        "unique_cells": int(len(values)),
        "observations": int(values.sum()) if len(values) else 0,
        "support": {
            "minimum": int(values[0]) if len(values) else None,
            "median": int(np.median(values)) if len(values) else None,
            "p75": int(np.percentile(values, 75, method="lower")) if len(values) else None,
            "maximum": int(values[-1]) if len(values) else None,
        },
        "cells_with_support_at_least": {
            "2": int((values >= 2).sum()),
            "5": int((values >= 5).sum()),
            "10": int((values >= 10).sum()),
        },
    }


def _support_records(
    counters: Mapping[tuple[Any, ...], Counter[Any]],
    labels: Sequence[str],
    item_label: str,
) -> list[dict[str, Any]]:
    records = []
    for key in sorted(counters, key=lambda value: tuple(map(str, value))):
        counter = counters[key]
        record = {label: value for label, value in zip(labels, key)}
        record.update(
            {
                "observations": int(sum(counter.values())),
                f"unique_{item_label}": int(len(counter)),
                "support_per_unique": _quantiles(counter),
            }
        )
        records.append(record)
    return records


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        while parent != self.parent[parent]:
            parent = self.parent[parent]
        while value != parent:
            next_value = self.parent[value]
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)

    def summary(self) -> dict[str, Any]:
        groups = Counter(self.find(value) for value in self.parent)
        sizes = sorted(groups.values(), reverse=True)
        return {
            "nodes": len(self.parent),
            "components": len(sizes),
            "largest_component_nodes": sizes[0] if sizes else 0,
            "isolated_nodes": sum(size == 1 for size in sizes),
            "component_sizes_descending": sizes,
        }


def _graph_triplet(
    nodes: set[str],
    ally_edges: set[tuple[str, str]],
    enemy_edges: set[tuple[str, str]],
) -> dict[str, Any]:
    ally_graph = _UnionFind(nodes)
    enemy_graph = _UnionFind(nodes)
    combined_graph = _UnionFind(nodes)
    for left, right in ally_edges:
        ally_graph.union(left, right)
        combined_graph.union(left, right)
    for left, right in enemy_edges:
        enemy_graph.union(left, right)
        combined_graph.union(left, right)
    return {
        "ally": ally_graph.summary(),
        "enemy": enemy_graph.summary(),
        "combined": combined_graph.summary(),
    }


def _scoped_connectivity_records(
    groups: Mapping[
        tuple[Any, ...],
        Mapping[str, set[Any]],
    ],
    labels: Sequence[str],
) -> list[dict[str, Any]]:
    records = []
    for key in sorted(groups, key=lambda value: tuple(map(str, value))):
        group = groups[key]
        record = {label: value for label, value in zip(labels, key)}
        record.update(
            _graph_triplet(
                group["nodes"],
                group["ally_edges"],
                group["enemy_edges"],
            )
        )
        records.append(record)
    return records


def _overlap(left: set[Any], right: set[Any]) -> dict[str, Any]:
    intersection = len(left & right)
    union = len(left | right)
    return {
        "left_unique": len(left),
        "right_unique": len(right),
        "intersection": intersection,
        "union": union,
        "jaccard": (intersection / union) if union else None,
    }


def _scope_label(year: int, patch: str) -> str:
    return f"{year}:{patch}"


def _build_design(
    observations: Sequence[tuple[set[str], set[tuple[str, str]], set[tuple[str, str]]]]
) -> tuple[sparse.csr_matrix, dict[str, list[str]]]:
    nodes = sorted({item for row in observations for item in row[0]})
    allies = sorted({item for row in observations for item in row[1]})
    enemies = sorted({item for row in observations for item in row[2]})
    labels = {
        "intercept": ["intercept"],
        "node": nodes,
        "ally": [f"{a}~~{b}" for a, b in allies],
        "enemy": [f"{a}~~{b}" for a, b in enemies],
    }
    node_index = {value: 1 + index for index, value in enumerate(nodes)}
    ally_offset = 1 + len(nodes)
    ally_index = {value: ally_offset + index for index, value in enumerate(allies)}
    enemy_offset = ally_offset + len(allies)
    enemy_index = {value: enemy_offset + index for index, value in enumerate(enemies)}
    row_indices: list[int] = []
    column_indices: list[int] = []
    for row_index, (row_nodes, row_allies, row_enemies) in enumerate(observations):
        columns = [0]
        columns.extend(node_index[value] for value in sorted(row_nodes))
        columns.extend(ally_index[value] for value in sorted(row_allies))
        columns.extend(enemy_index[value] for value in sorted(row_enemies))
        row_indices.extend([row_index] * len(columns))
        column_indices.extend(columns)
    matrix = sparse.csr_matrix(
        (
            np.ones(len(row_indices), dtype=np.int8),
            (np.asarray(row_indices), np.asarray(column_indices)),
        ),
        shape=(len(observations), enemy_offset + len(enemies)),
        dtype=np.int8,
    )
    return matrix, labels


def _numeric_spectrum(
    matrix: sparse.csr_matrix, ordered_game_ids: Sequence[str]
) -> dict[str, Any]:
    """Return a bounded deterministic sampled-row-space diagnostic."""
    if not matrix.shape[0]:
        return {
            "method": "not_available_no_valid_maps",
            "numeric_rank_tolerance": None,
            "largest_singular_value": None,
            "smallest_nonzero_singular_value": None,
            "condition_number_nonzero_subspace": None,
            "sampled_maps": 0,
            "approximation": True,
        }
    if len(ordered_game_ids) != matrix.shape[0]:
        raise RepresentationAssayPreflightError("design game-id count mismatch")
    selected = sorted(
        range(len(ordered_game_ids)),
        key=lambda index: (
            hashlib.sha256(ordered_game_ids[index].encode("utf-8")).digest(),
            ordered_game_ids[index],
        ),
    )[:256]
    selected_game_ids = [ordered_game_ids[index] for index in selected]
    sample = matrix[selected].astype(np.float64)
    row_gram = (sample @ sample.T).toarray()
    eigenvalues = np.linalg.eigvalsh(row_gram)
    singular_values = np.sqrt(np.clip(eigenvalues, 0.0, None))[::-1]
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    rank_tolerance = max(sample.shape) * np.finfo(float).eps * largest
    positive = singular_values[singular_values > rank_tolerance]
    smallest = float(positive[-1]) if len(positive) else None
    sample_rank = int(len(positive))
    return {
        "method": (
            "dense eigvalsh of A_sample A_sample^T for at most 256 rows of the "
            "sparse map-by-canonical-feature incidence design"
        ),
        "sampling_rule": (
            "take the 256 lexicographically smallest sha256(UTF-8 game_id), "
            "tie-broken by game_id"
        ),
        "sampled_maps": len(selected),
        "sample_game_ids_sha256": canonical_sha256(sorted(selected_game_ids)),
        "numeric_rank_tolerance": float(rank_tolerance),
        "largest_singular_value": largest,
        "smallest_nonzero_singular_value": smallest,
        "condition_number_nonzero_subspace": (largest / smallest) if smallest else None,
        "sampled_row_space_rank": sample_rank,
        "sampled_row_space_nullity": len(selected) - sample_rank,
        "approximation": True,
        "rank_scope": (
            "Rank, nullity, and singular extrema describe only the manifested "
            "sampled row space. They are not full-design numeric rank, parameter "
            "identifiability, or a representation-rank selection."
        ),
    }


def analyze_frames(
    maps: pd.DataFrame,
    players: pd.DataFrame,
    *,
    maps_raw_sha256: str,
    players_raw_sha256: str,
    maps_locator: str,
    players_locator: str,
) -> dict[str, Any]:
    """Analyze already-loaded allowlisted columns without reading outcomes."""
    if tuple(maps.columns) != MAP_COLUMNS:
        raise RepresentationAssayPreflightError("maps columns do not match the outcome-free allowlist")
    if tuple(players.columns) != PLAYER_COLUMNS:
        raise RepresentationAssayPreflightError(
            "player columns do not match the outcome-free allowlist"
        )
    for digest in (maps_raw_sha256, players_raw_sha256):
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise RepresentationAssayPreflightError("source digest is not a lowercase sha256")

    map_groups = {str(key): group for key, group in maps.groupby("oe_gameid", dropna=False, sort=True)}
    relevant_players = players[players["gameid"].astype(str).isin(map_groups)]
    player_groups = {
        str(key): group
        for key, group in relevant_players.groupby("gameid", dropna=False, sort=True)
    }
    player_league_disagreements = 0
    for game_id, map_group in map_groups.items():
        player_group = player_groups.get(game_id)
        if len(map_group) == 1 and player_group is not None and len(player_group):
            map_league = map_group.iloc[0]["league"]
            player_leagues = {
                _text(value) for value in player_group["league"] if _not_missing(value)
            }
            if _not_missing(map_league) and player_leagues != {_text(map_league)}:
                player_league_disagreements += 1
    valid: list[_ValidMap] = []
    exclusions: Counter[str] = Counter()
    rejection_ledger: list[dict[str, str]] = []
    for game_id in sorted(map_groups):
        result, reason = _validate_map(
            game_id,
            map_groups[game_id],
            player_groups.get(game_id, players.iloc[0:0]),
        )
        if result is None:
            exclusions[reason or "unknown"] += 1
            rejection_ledger.append({"game_id": game_id, "reason": reason or "unknown"})
        else:
            valid.append(result)

    node_support: defaultdict[tuple[Any, ...], Counter[str]] = defaultdict(Counter)
    ally_support: defaultdict[tuple[Any, ...], Counter[tuple[str, str]]] = defaultdict(Counter)
    enemy_support: defaultdict[tuple[Any, ...], Counter[tuple[str, str]]] = defaultdict(Counter)
    temporal_sets: defaultdict[tuple[str, int, str], set[Any]] = defaultdict(set)
    all_nodes: set[str] = set()
    all_ally_edges: set[tuple[str, str]] = set()
    all_enemy_edges: set[tuple[str, str]] = set()
    global_node_support: Counter[str] = Counter()
    global_ally_support: Counter[tuple[str, str]] = Counter()
    global_enemy_support: Counter[tuple[str, str]] = Counter()
    cohort_support: defaultdict[str, dict[str, Counter[Any]]] = defaultdict(
        lambda: {"nodes": Counter(), "ally_edges": Counter(), "enemy_edges": Counter()}
    )
    cohort_maps: Counter[str] = Counter()
    observations: list[tuple[set[str], set[tuple[str, str]], set[tuple[str, str]]]] = []
    connectivity_groups: dict[
        str, defaultdict[tuple[Any, ...], dict[str, set[Any]]]
    ] = {
        dimension: defaultdict(
            lambda: {"nodes": set(), "ally_edges": set(), "enemy_edges": set()}
        )
        for dimension in ("year_patch", "league", "year_patch_league")
    }

    for item in sorted(valid, key=lambda value: value.game_id):
        cohort_maps[item.source_datacompleteness] += 1
        scope = (
            item.year,
            item.patch,
            item.league,
            item.competition_scope,
            item.event_kind,
            item.is_international,
        )
        row_nodes: set[str] = set()
        row_allies: set[tuple[str, str]] = set()
        row_enemies: set[tuple[str, str]] = set()
        for side in ("blue", "red"):
            nodes = item.nodes_by_side[side]
            for node in nodes:
                node_id = _node_id(node)
                row_nodes.add(node_id)
                node_support[scope + (node[1],)][node_id] += 1
                global_node_support[node_id] += 1
                cohort_support[item.source_datacompleteness]["nodes"][node_id] += 1
            for left, right in combinations(nodes, 2):
                edge = _canonical_pair(_node_id(left), _node_id(right))
                row_allies.add(edge)
                ally_support[scope + (_role_pair(left[1], right[1]),)][edge] += 1
                global_ally_support[edge] += 1
                cohort_support[item.source_datacompleteness]["ally_edges"][edge] += 1
        for blue in item.nodes_by_side["blue"]:
            for red in item.nodes_by_side["red"]:
                edge = _canonical_pair(_node_id(blue), _node_id(red))
                row_enemies.add(edge)
                enemy_support[scope + (_role_pair(blue[1], red[1]),)][edge] += 1
                global_enemy_support[edge] += 1
                cohort_support[item.source_datacompleteness]["enemy_edges"][edge] += 1
        if len(row_nodes) != 10 or len(row_allies) != 20 or len(row_enemies) != 25:
            raise RepresentationAssayPreflightError(
                f"valid map {item.game_id} violated the 10/20/25 observation contract"
            )
        observations.append((row_nodes, row_allies, row_enemies))
        all_nodes.update(row_nodes)
        all_ally_edges.update(row_allies)
        all_enemy_edges.update(row_enemies)
        for dimension, key in (
            ("year_patch", (item.year, item.patch)),
            ("league", (item.league,)),
            ("year_patch_league", (item.year, item.patch, item.league)),
        ):
            group = connectivity_groups[dimension][key]
            group["nodes"].update(row_nodes)
            group["ally_edges"].update(row_allies)
            group["enemy_edges"].update(row_enemies)
        for category, values in (
            ("node", row_nodes),
            ("ally", row_allies),
            ("enemy", row_enemies),
        ):
            temporal_sets[(category, item.year, item.patch)].update(values)

    global_graphs = _graph_triplet(all_nodes, all_ally_edges, all_enemy_edges)

    yearly: defaultdict[tuple[str, int], set[Any]] = defaultdict(set)
    for (category, year, _patch), values in temporal_sets.items():
        yearly[(category, year)].update(values)
    years = sorted({year for _, year in yearly})
    year_transitions = []
    for left_year, right_year in zip(years, years[1:]):
        for category in ("node", "ally", "enemy"):
            year_transitions.append(
                {
                    "category": category,
                    "left_year": left_year,
                    "right_year": right_year,
                    **_overlap(
                        yearly.get((category, left_year), set()),
                        yearly.get((category, right_year), set()),
                    ),
                }
            )
    patches = sorted(
        {(year, patch) for _, year, patch in temporal_sets},
        key=lambda item: (item[0], tuple(map(int, item[1].split(".")))),
    )
    patch_transitions = []
    for left_scope, right_scope in zip(patches, patches[1:]):
        for category in ("node", "ally", "enemy"):
            patch_transitions.append(
                {
                    "category": category,
                    "left_scope": _scope_label(*left_scope),
                    "right_scope": _scope_label(*right_scope),
                    **_overlap(
                        temporal_sets.get((category, *left_scope), set()),
                        temporal_sets.get((category, *right_scope), set()),
                    ),
                }
            )

    matrix, design_labels = _build_design(observations)
    generic_rank = int(structural_rank(matrix)) if matrix.shape[0] else 0
    spectrum = _numeric_spectrum(
        matrix,
        [item.game_id for item in sorted(valid, key=lambda value: value.game_id)],
    )
    column_counts = np.asarray(matrix.sum(axis=0)).ravel().astype(int)
    nonzero_counts = column_counts[column_counts > 0]
    identity_count = 2 * len(design_labels["node"]) + 3
    payload: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "development_only": True,
        "outcome_free": True,
        "predictive_authority": False,
        "representation_rank_selected": False,
        "authorizes_model_selection": False,
        "authorizes_publication": False,
        "content_addressing_confers_authority": False,
        "claim_ceiling": {
            "support_and_design_diagnostics_only": True,
            "no_effect_estimates": True,
            "no_predictive_claim": True,
            "no_representation_rank_claim": True,
        },
        "generator": _generator_identity(),
        "source": {
            "maps": {
                "locator": maps_locator,
                "raw_sha256": maps_raw_sha256,
                "selected_input_sha256": selected_input_sha256(maps),
                "columns_read": list(MAP_COLUMNS),
                "rows": int(len(maps)),
            },
            "player_games": {
                "locator": players_locator,
                "raw_sha256": players_raw_sha256,
                "selected_input_sha256": selected_input_sha256(players),
                "columns_read": list(PLAYER_COLUMNS),
                "rows": int(len(players)),
                "rows_joined_to_map_registry": int(len(relevant_players)),
            },
        },
        "eligibility": {
            "registry_maps": len(map_groups),
            "valid_maps": len(valid),
            "excluded_maps": len(map_groups) - len(valid),
            "exclusion_reasons": [
                {"reason": reason, "maps": exclusions[reason]} for reason in sorted(exclusions)
            ],
            "rejection_ledger": sorted(
                rejection_ledger, key=lambda item: (item["game_id"], item["reason"])
            ),
        },
        "datacompleteness": {
            "maps_global_labels": _global_datacompleteness(maps),
            "player_games_full_source_global_labels": _global_datacompleteness(players),
            "player_games_joined_global_labels": _global_datacompleteness(relevant_players),
            "maps_required_fields": _required_completeness(maps, MAP_ELIGIBILITY_FIELDS),
            "player_games_full_source_required_fields": _required_completeness(
                players, PLAYER_ELIGIBILITY_FIELDS
            ),
            "player_games_joined_required_fields": _required_completeness(
                relevant_players, PLAYER_ELIGIBILITY_FIELDS
            ),
            "joined_map_player_league_label_disagreements": player_league_disagreements,
            "note": (
                "Global datacompleteness labels are source annotations and are not "
                "used as a substitute for field-level eligibility. The map registry "
                "contains the canonical league label; disagreement with the raw "
                "player-game league label is recorded, not silently relabeled."
            ),
        },
        "observation_contract": {
            "role_order": list(ROLE_ORDER),
            "role_aliases": dict(sorted(ROLE_ALIASES.items())),
            "nodes_per_valid_map": 10,
            "canonical_ally_observations_per_valid_map": 20,
            "canonical_cross_team_observations_per_valid_map": 25,
            "ally_definition": "unordered role-conditioned node pair within one side",
            "cross_team_definition": "unordered canonical role-conditioned node pair across sides",
            "patch_semantics": (
                "patch is an oe_source_patch_token normalized from the OE numeric "
                "field to two decimal places; it is not a verified official Riot "
                "patch identifier and no 16.x to 26.x mapping is applied"
            ),
            "date_semantics": (
                "source date must be parseable and exactly equal between map and "
                "all joined player rows; timezone-naive source timestamps remain "
                "naive and no timezone is imputed"
            ),
        },
        "support": {
            "global": {
                "maps": len(valid),
                "unique_event_timestamps": len({item.date for item in valid}),
                "unique_calendar_dates": len(
                    {pd.Timestamp(item.date).date().isoformat() for item in valid}
                ),
                "date_timezone_awareness": sorted(
                    {item.date_timezone_awareness for item in valid}
                ),
                "nodes": _global_support(global_node_support),
                "ally_edges": _global_support(global_ally_support),
                "enemy_edges": _global_support(global_enemy_support),
            },
            "source_datacompleteness_cohorts": [
                {
                    "cohort": cohort,
                    "maps": maps_count,
                    **{
                        category: _global_support(
                            {
                                "nodes": global_node_support,
                                "ally_edges": global_ally_support,
                                "enemy_edges": global_enemy_support,
                            }[category]
                            if cohort == "field_eligible_all"
                            else cohort_support[cohort][category]
                        )
                        for category in ("nodes", "ally_edges", "enemy_edges")
                    },
                }
                for cohort, maps_count in (
                    ("field_eligible_all", len(valid)),
                    ("complete", cohort_maps.get("complete", 0)),
                    ("partial", cohort_maps.get("partial", 0)),
                )
            ],
            "node_by_year_patch_league_role": _support_records(
                node_support,
                (
                    "year",
                    "patch",
                    "league",
                    "competition_scope",
                    "event_kind",
                    "is_international",
                    "role",
                ),
                "nodes",
            ),
            "ally_by_year_patch_league_role_pair": _support_records(
                ally_support,
                (
                    "year",
                    "patch",
                    "league",
                    "competition_scope",
                    "event_kind",
                    "is_international",
                    "role_pair",
                ),
                "edges",
            ),
            "enemy_by_year_patch_league_role_pair": _support_records(
                enemy_support,
                (
                    "year",
                    "patch",
                    "league",
                    "competition_scope",
                    "event_kind",
                    "is_international",
                    "role_pair",
                ),
                "edges",
            ),
        },
        "temporal_overlap": {
            "metric": "set Jaccard; descriptive support reuse, not interaction invariance",
            "consecutive_years": year_transitions,
            "consecutive_observed_patches": patch_transitions,
        },
        "graph_connectivity": {
            "node_definition": "champion::role",
            "scope_statement": (
                "unsigned support-graph connectivity only; connectivity does not "
                "establish effect stability, signed counter geometry, or "
                "identifiability"
            ),
            **global_graphs,
            "by_year_patch": _scoped_connectivity_records(
                connectivity_groups["year_patch"], ("year", "patch")
            ),
            "by_league": _scoped_connectivity_records(
                connectivity_groups["league"], ("league",)
            ),
            "by_year_patch_league": _scoped_connectivity_records(
                connectivity_groups["year_patch_league"],
                ("year", "patch", "league"),
            ),
        },
        "design_diagnostics": {
            "design": (
                "one row per valid map; columns are intercept, 10 node incidences, "
                "20 canonical ally incidences, and 25 canonical unsigned enemy "
                "co-occurrence incidences"
            ),
            "rows": int(matrix.shape[0]),
            "columns": int(matrix.shape[1]),
            "nonzero_entries": int(matrix.nnz),
            "columns_by_block": {key: len(value) for key, value in design_labels.items()},
            "structural_rank_upper_bound": generic_rank,
            "structural_column_nullity_lower_bound": int(matrix.shape[1] - generic_rank),
            "structural_rank_method": (
                "exact maximum bipartite matching on the sparse zero pattern; it is "
                "an upper bound on numeric rank and ignores coefficient identities"
            ),
            "known_exact_linear_identity_vectors": identity_count,
            "identity_description": (
                "for every node, ally incidence degree equals 4 times node incidence "
                "and enemy incidence degree equals 5 times node incidence; three "
                "additional block-sum identities tie node, ally, and enemy totals to "
                "the intercept. The reported count is not claimed independent."
            ),
            "full_parameter_condition_number": "infinite",
            "condition_reason": (
                "the exact nonzero null vectors make the unrestrained column "
                "parameterization singular; no finite condition estimate or effect "
                "identifiability is claimed. This necessary unsigned support "
                "diagnostic does not diagnose rank or identifiability of a later "
                "signed antisymmetric counter design"
            ),
            "column_support": {
                "minimum": int(nonzero_counts.min()) if len(nonzero_counts) else None,
                "median": int(np.median(nonzero_counts)) if len(nonzero_counts) else None,
                "maximum": int(nonzero_counts.max()) if len(nonzero_counts) else None,
            },
            "numeric_spectrum": spectrum,
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    validate_artifact(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> None:
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise RepresentationAssayPreflightError("artifact top-level fields are not exact")
    if payload.get("schema_id") != SCHEMA_ID:
        raise RepresentationAssayPreflightError("artifact schema_id mismatch")
    if (
        payload.get("development_only") is not True
        or payload.get("outcome_free") is not True
        or payload.get("predictive_authority") is not False
        or payload.get("representation_rank_selected") is not False
        or payload.get("authorizes_model_selection") is not False
        or payload.get("authorizes_publication") is not False
        or payload.get("content_addressing_confers_authority") is not False
    ):
        raise RepresentationAssayPreflightError("artifact authority flags exceed preflight scope")
    ceiling = payload.get("claim_ceiling")
    if not isinstance(ceiling, Mapping) or ceiling != {
        "support_and_design_diagnostics_only": True,
        "no_effect_estimates": True,
        "no_predictive_claim": True,
        "no_representation_rank_claim": True,
    }:
        raise RepresentationAssayPreflightError("artifact claim ceiling mismatch")
    generator = payload.get("generator")
    if not isinstance(generator, Mapping) or set(generator) != {
        "version",
        "executable_dependency_boundary",
        "runtime_versions",
        "identity_scope",
    }:
        raise RepresentationAssayPreflightError("generator identity fields mismatch")
    if generator.get("version") != GENERATOR_VERSION:
        raise RepresentationAssayPreflightError("generator version mismatch")
    boundary = generator.get("executable_dependency_boundary")
    if (
        not isinstance(boundary, list)
        or len(boundary) != 1
        or not isinstance(boundary[0], Mapping)
        or set(boundary[0]) != {"locator", "raw_sha256"}
        or boundary[0].get("locator")
        != "lol_kills/v2/draft/interactions/representation_assay_preflight.py"
    ):
        raise RepresentationAssayPreflightError("generator dependency boundary mismatch")
    generator_digest = boundary[0].get("raw_sha256")
    if (
        not isinstance(generator_digest, str)
        or len(generator_digest) != 64
        or any(char not in "0123456789abcdef" for char in generator_digest)
    ):
        raise RepresentationAssayPreflightError("generator code sha256 invalid")
    submitted = payload.get("artifact_sha256")
    if (
        not isinstance(submitted, str)
        or len(submitted) != 64
        or any(char not in "0123456789abcdef" for char in submitted)
    ):
        raise RepresentationAssayPreflightError("artifact_sha256 is missing")
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256")
    if submitted != canonical_sha256(unsigned):
        raise RepresentationAssayPreflightError("artifact_sha256 does not match canonical payload")
    contract = payload.get("observation_contract", {})
    if (
        contract.get("nodes_per_valid_map") != 10
        or contract.get("canonical_ally_observations_per_valid_map") != 20
        or contract.get("canonical_cross_team_observations_per_valid_map") != 25
    ):
        raise RepresentationAssayPreflightError("observation contract mismatch")
    source = payload.get("source", {})
    expected_columns = {"maps": list(MAP_COLUMNS), "player_games": list(PLAYER_COLUMNS)}
    for key, columns in expected_columns.items():
        entry = source.get(key, {})
        if entry.get("columns_read") != columns:
            raise RepresentationAssayPreflightError(f"{key} outcome-free columns mismatch")
        digest = entry.get("raw_sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise RepresentationAssayPreflightError(f"{key} raw_sha256 invalid")
        selected_digest = entry.get("selected_input_sha256")
        if (
            not isinstance(selected_digest, str)
            or len(selected_digest) != 64
            or any(char not in "0123456789abcdef" for char in selected_digest)
        ):
            raise RepresentationAssayPreflightError(f"{key} selected_input_sha256 invalid")
    eligibility = payload.get("eligibility", {})
    registry_maps = eligibility.get("registry_maps")
    valid_maps = eligibility.get("valid_maps")
    excluded_maps = eligibility.get("excluded_maps")
    if (
        isinstance(registry_maps, bool)
        or not isinstance(registry_maps, int)
        or isinstance(valid_maps, bool)
        or not isinstance(valid_maps, int)
        or isinstance(excluded_maps, bool)
        or not isinstance(excluded_maps, int)
        or min(registry_maps, valid_maps, excluded_maps) < 0
        or valid_maps + excluded_maps != registry_maps
    ):
        raise RepresentationAssayPreflightError("eligibility arithmetic mismatch")
    ledger = eligibility.get("rejection_ledger")
    reasons = eligibility.get("exclusion_reasons")
    if not isinstance(ledger, list) or len(ledger) != excluded_maps:
        raise RepresentationAssayPreflightError("rejection ledger arithmetic mismatch")
    ledger_counts = Counter(
        item.get("reason") for item in ledger if isinstance(item, Mapping)
    )
    expected_counts = {
        item.get("reason"): item.get("maps")
        for item in reasons
        if isinstance(item, Mapping)
    }
    if dict(ledger_counts) != expected_counts:
        raise RepresentationAssayPreflightError("exclusion reason arithmetic mismatch")
    diagnostics = payload.get("design_diagnostics", {})
    if diagnostics.get("full_parameter_condition_number") != "infinite":
        raise RepresentationAssayPreflightError("singular-design diagnostic was removed")
    rows = diagnostics.get("rows")
    columns = diagnostics.get("columns")
    blocks = diagnostics.get("columns_by_block")
    generic_rank = diagnostics.get("structural_rank_upper_bound")
    generic_nullity = diagnostics.get("structural_column_nullity_lower_bound")
    if (
        rows != valid_maps
        or diagnostics.get("nonzero_entries") != 56 * valid_maps
        or not isinstance(blocks, Mapping)
        or columns != sum(blocks.values())
        or not isinstance(generic_rank, int)
        or generic_rank < 0
        or generic_rank > min(rows, columns)
        or generic_nullity != columns - generic_rank
    ):
        raise RepresentationAssayPreflightError("design arithmetic mismatch")
    global_support = payload.get("support", {}).get("global", {})
    if global_support.get("maps") != valid_maps:
        raise RepresentationAssayPreflightError("global support map count mismatch")
    for key, multiplier in (("nodes", 10), ("ally_edges", 20), ("enemy_edges", 25)):
        if global_support.get(key, {}).get("observations") != multiplier * valid_maps:
            raise RepresentationAssayPreflightError(f"global {key} support arithmetic mismatch")
    spectrum = diagnostics.get("numeric_spectrum", {})
    sampled_maps = spectrum.get("sampled_maps")
    if (
        spectrum.get("approximation") is not True
        or not isinstance(sampled_maps, int)
        or sampled_maps != min(256, valid_maps)
    ):
        raise RepresentationAssayPreflightError("bounded spectrum manifest mismatch")
    encoded = canonical_bytes(payload).decode("ascii").lower()
    forbidden_claims = ('"effect_estimate"', '"selected_rank"', '"win_probability"')
    if any(token in encoded for token in forbidden_claims):
        raise RepresentationAssayPreflightError("artifact contains a forbidden model claim")


def build_from_parquet(
    maps_path: Path = DEFAULT_MAPS_PATH,
    players_path: Path = DEFAULT_PLAYERS_PATH,
    *,
    maps_locator: str | None = None,
    players_locator: str | None = None,
) -> dict[str, Any]:
    for path in (maps_path, players_path):
        if not path.is_file() or path.is_symlink():
            raise RepresentationAssayPreflightError(f"source is not a regular file: {path}")
    maps = pd.read_parquet(maps_path, columns=list(MAP_COLUMNS))
    players = pd.read_parquet(players_path, columns=list(PLAYER_COLUMNS))
    maps = maps.loc[:, list(MAP_COLUMNS)]
    players = players.loc[:, list(PLAYER_COLUMNS)]
    return analyze_frames(
        maps,
        players,
        maps_raw_sha256=raw_sha256(maps_path),
        players_raw_sha256=raw_sha256(players_path),
        maps_locator=maps_locator if maps_locator is not None else maps_path.as_posix(),
        players_locator=(
            players_locator if players_locator is not None else players_path.as_posix()
        ),
    )


def load_and_replay_artifact(
    artifact_path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    source_root: Path | None = None,
) -> dict[str, Any]:
    """Replay pinned sources and reject caller-rehashed artifact mutations."""
    try:
        persisted_bytes = artifact_path.read_bytes()
        payload = json.loads(persisted_bytes)
    except (OSError, json.JSONDecodeError) as exc:
        raise RepresentationAssayPreflightError(
            f"cannot load persisted preflight artifact: {artifact_path}"
        ) from exc
    if not isinstance(payload, dict):
        raise RepresentationAssayPreflightError("persisted artifact must be an object")
    validate_artifact(payload)
    if persisted_bytes != canonical_bytes(payload):
        raise RepresentationAssayPreflightError(
            "persisted artifact bytes are not canonical generator bytes"
        )
    if payload["generator"] != _generator_identity():
        raise RepresentationAssayPreflightError(
            "persisted artifact generator identity does not match executable generator"
        )
    root = source_root if source_root is not None else Path.cwd()

    def resolve_source(locator: object, label: str) -> Path:
        if not isinstance(locator, str) or not locator:
            raise RepresentationAssayPreflightError(f"{label} source locator is invalid")
        path = Path(locator)
        resolved = path if path.is_absolute() else root / path
        if not resolved.is_file() or resolved.is_symlink():
            raise RepresentationAssayPreflightError(
                f"{label} pinned source is not a regular non-symlink file"
            )
        return resolved

    maps_path = resolve_source(payload["source"]["maps"].get("locator"), "maps")
    players_path = resolve_source(
        payload["source"]["player_games"].get("locator"), "player_games"
    )
    if raw_sha256(maps_path) != payload["source"]["maps"]["raw_sha256"]:
        raise RepresentationAssayPreflightError("maps pinned source bytes changed")
    if raw_sha256(players_path) != payload["source"]["player_games"]["raw_sha256"]:
        raise RepresentationAssayPreflightError("player_games pinned source bytes changed")
    replayed = build_from_parquet(
        maps_path,
        players_path,
        maps_locator=payload["source"]["maps"]["locator"],
        players_locator=payload["source"]["player_games"]["locator"],
    )
    if canonical_bytes(replayed) != persisted_bytes:
        raise RepresentationAssayPreflightError(
            "source-backed replay does not match persisted canonical payload"
        )
    return payload


def write_artifact(
    output_path: Path = DEFAULT_ARTIFACT_PATH,
    *,
    maps_path: Path = DEFAULT_MAPS_PATH,
    players_path: Path = DEFAULT_PLAYERS_PATH,
) -> dict[str, Any]:
    payload = build_from_parquet(maps_path, players_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(canonical_bytes(payload))
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", type=Path, default=DEFAULT_MAPS_PATH)
    parser.add_argument("--players", type=Path, default=DEFAULT_PLAYERS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH)
    parser.add_argument(
        "--verify-existing",
        action="store_true",
        help="replay and compare an existing --output artifact instead of writing",
    )
    args = parser.parse_args(argv)
    if args.verify_existing:
        payload = load_and_replay_artifact(args.output)
        print(
            json.dumps(
                {
                    "artifact": args.output.as_posix(),
                    "artifact_sha256": payload["artifact_sha256"],
                    "replay_verified": True,
                },
                sort_keys=True,
            )
        )
        return 0
    payload = write_artifact(args.output, maps_path=args.maps, players_path=args.players)
    print(
        json.dumps(
            {
                "artifact": args.output.as_posix(),
                "artifact_sha256": payload["artifact_sha256"],
                "valid_maps": payload["eligibility"]["valid_maps"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
