"""Quantify participant identity coverage and dependence for terminal Draft.

This is a retrospective development diagnostic, not a reliability or promotion
artifact.  It binds the frozen Draft cohort to the exact player warehouse bytes
and tests whether game components connected by shared participants can be used
as atomic evaluation clusters.  A degenerate component graph remains a blocker.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import pyarrow.parquet as pq

from lol_kills.v2.data.common import ROLES, parse_rfc3339, sha256_canonical_object

from . import development_snapshot


ROOT = Path(__file__).resolve().parents[4]
SOURCE_LOCATOR = (
    "lol_kills/v2/draft/terminal/participant_dependence_diagnostic_v1.py"
)
DEVELOPMENT_SNAPSHOT_SOURCE_LOCATOR = (
    "lol_kills/v2/draft/terminal/development_snapshot.py"
)
SCHEMA_VERSION = "scryglass:draft-terminal-participant-dependence-diagnostic:v1"
RESULT_STATE = "PARTICIPANT_IDENTITIES_AVAILABLE_ATOMIC_COMPONENT_SPLIT_DEGENERATE"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/draft-terminal/participant-dependence-diagnostic-v1.json"
)
PLAYERS_LOCATOR = Path("data/lol/warehouse/parquet/players.parquet")
_SIDES = {"Blue", "Red"}
_POSITION_TO_ROLE = {
    "top": "top",
    "jng": "jungle",
    "mid": "mid",
    "bot": "bot",
    "sup": "support",
}


class ParticipantDependenceDiagnosticError(ValueError):
    """The participant diagnostic or one of its bound inputs is invalid."""


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise ParticipantDependenceDiagnosticError(
            f"diagnostic source is unavailable: {locator}"
        )
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _clock(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ParticipantDependenceDiagnosticError(
            "diagnostic clock must be timezone-aware"
        )
    return value.astimezone(timezone.utc)


def _participant_rows(root: Path, game_ids: set[str]) -> dict[str, list[tuple[str | None, str | None, str | None]]]:
    path = root / PLAYERS_LOCATOR
    if path.is_symlink() or not path.is_file():
        raise ParticipantDependenceDiagnosticError(
            "players parquet is unavailable or aliased"
        )
    table = pq.read_table(
        path,
        columns=["game_uid", "playerid", "position", "side"],
    )
    values = table.to_pydict()
    grouped: dict[str, list[tuple[str | None, str | None, str | None]]] = defaultdict(list)
    for game_id, player_id, position, side in zip(
        values["game_uid"],
        values["playerid"],
        values["position"],
        values["side"],
    ):
        if game_id in game_ids:
            grouped[game_id].append((player_id, position, side))
    return grouped


def _valid_assignments(
    grouped: Mapping[str, list[tuple[str | None, str | None, str | None]]]
) -> dict[str, tuple[tuple[str, str, str], ...]]:
    expected_slots = {(side, role) for side in _SIDES for role in ROLES}
    valid: dict[str, tuple[tuple[str, str, str], ...]] = {}
    for game_id, rows in grouped.items():
        if len(rows) != 10:
            continue
        if any(
            not isinstance(player_id, str)
            or not player_id
            or position not in _POSITION_TO_ROLE
            or side not in _SIDES
            for player_id, position, side in rows
        ):
            continue
        typed = [
            (str(player_id), _POSITION_TO_ROLE[str(position)], str(side))
            for player_id, position, side in rows
        ]
        if len({player_id for player_id, _, _ in typed}) != 10:
            continue
        if {(side, position) for _, position, side in typed} != expected_slots:
            continue
        valid[game_id] = tuple(sorted(typed, key=lambda item: (item[2], item[1], item[0])))
    return valid


def _component_summary(
    assignments: Mapping[str, tuple[tuple[str, str, str], ...]]
) -> dict[str, Any]:
    games = sorted(assignments)
    parent = {game_id: game_id for game_id in games}
    size = {game_id: 1 for game_id in games}

    def find(game_id: str) -> str:
        while parent[game_id] != game_id:
            parent[game_id] = parent[parent[game_id]]
            game_id = parent[game_id]
        return game_id

    def union(first: str, second: str) -> None:
        left, right = find(first), find(second)
        if left == right:
            return
        if size[left] < size[right]:
            left, right = right, left
        parent[right] = left
        size[left] += size[right]

    first_game_by_participant: dict[str, str] = {}
    for game_id in games:
        for player_id, _, _ in assignments[game_id]:
            previous = first_game_by_participant.setdefault(player_id, game_id)
            union(game_id, previous)
    components = Counter(find(game_id) for game_id in games)
    largest = max(components.values(), default=0)
    return {
        "graph_definition": "games_are_connected_when_they_share_at_least_one_exact_player_id",
        "transitive_component_count": len(components),
        "largest_component_maps": largest,
        "largest_component_fraction": largest / len(games) if games else 0.0,
        "all_valid_maps_in_one_component": bool(games) and largest == len(games),
        "atomic_component_temporal_split_available": len(components) > 1 and largest < len(games),
    }


def _chronological_overlap(
    rows: list[Any],
    assignments: Mapping[str, tuple[tuple[str, str, str], ...]],
) -> dict[str, Any]:
    ordered = sorted(
        (row.date, row.game_id) for row in rows if row.game_id in assignments
    )
    thirds: list[set[str]] = []
    maps_per_third: list[int] = []
    for index in range(3):
        start = index * len(ordered) // 3
        end = (index + 1) * len(ordered) // 3
        games = [game_id for _, game_id in ordered[start:end]]
        maps_per_third.append(len(games))
        thirds.append(
            {
                player_id
                for game_id in games
                for player_id, _, _ in assignments[game_id]
            }
        )
    return {
        "partition": "three_equal_map_count_chronological_blocks",
        "maps_per_block": maps_per_third,
        "participants_block_0_and_1": len(thirds[0] & thirds[1]),
        "participants_block_1_and_2": len(thirds[1] & thirds[2]),
        "participants_block_0_and_2": len(thirds[0] & thirds[2]),
        "participants_all_three_blocks": len(thirds[0] & thirds[1] & thirds[2]),
    }


def _diagnostic_inputs(root: Path) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    rows, snapshot = development_snapshot.load_development_snapshot(root)
    game_ids = {row.game_id for row in rows}
    grouped = _participant_rows(root, game_ids)
    assignments = _valid_assignments(grouped)
    assignment_digest = sha256_canonical_object(
        [
            {
                "game_id": game_id,
                "participants": [
                    {"player_id": player_id, "position": position, "side": side}
                    for player_id, position, side in assignments[game_id]
                ],
            }
            for game_id in sorted(assignments)
        ]
    )
    statistics = {
        "snapshot_maps": len(rows),
        "maps_joined_to_player_rows": len(grouped),
        "joined_player_rows": sum(len(items) for items in grouped.values()),
        "maps_with_exact_ten_unique_players_and_roles": len(assignments),
        "maps_without_exact_ten_unique_players_and_roles": len(rows) - len(assignments),
        "exact_assignment_coverage_fraction": len(assignments) / len(rows),
        "unique_participants": len(
            {
                player_id
                for items in assignments.values()
                for player_id, _, _ in items
            }
        ),
        "participant_assignment_sha256": assignment_digest,
        "component_graph": _component_summary(assignments),
        "chronological_overlap": _chronological_overlap(rows, assignments),
    }
    return rows, snapshot, statistics


def build_participant_dependence_diagnostic_v1(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    created = _clock(clock)
    rows, snapshot, statistics = _diagnostic_inputs(root)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "created_at_utc": created.isoformat(),
        "inputs": {
            "development_snapshot_manifest_locator": development_snapshot.DEFAULT_MANIFEST.as_posix(),
            "development_snapshot_manifest_raw_sha256": snapshot[
                "development_snapshot_manifest_raw_sha256"
            ],
            "development_snapshot_manifest_artifact_sha256": snapshot[
                "development_snapshot_manifest_artifact_sha256"
            ],
            "development_snapshot_payload_raw_sha256": snapshot[
                "development_snapshot_payload_raw_sha256"
            ],
            "players_parquet_locator": PLAYERS_LOCATOR.as_posix(),
            "players_parquet_raw_sha256": _sha256_path(root / PLAYERS_LOCATOR),
        },
        "population": {
            "development_rows": len(rows),
            **statistics,
        },
        "decision": {
            "participant_identity_available_for_development_diagnostic": True,
            "participant_identity_complete_for_all_snapshot_maps": False,
            "participant_atomic_component_split_available": False,
            "participant_dependence_support_verified": False,
            "reliability_gate_passed": False,
            "promotion_eligible": False,
            "required_next_evidence": (
                "A predeclared and independently reviewed participant-dependence "
                "method that does not treat the single connected component as "
                "independent observations, followed by untouched future evaluation."
            ),
        },
        "source_locks": [
            _source_record(root, SOURCE_LOCATOR),
            _source_record(root, DEVELOPMENT_SNAPSHOT_SOURCE_LOCATOR),
        ],
        "authority": {
            "model_validation_authority": False,
            "reliability_authority": False,
            "probability_authority": False,
            "odds_authority": False,
            "expected_value_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "Retrospective participant-dependence diagnostic only. It does not "
            "validate Draft Score, reliability, probability, odds, expected value, "
            "recommendations, or betting."
        ),
    }
    payload["artifact_sha256"] = sha256_canonical_object(payload)
    return validate_participant_dependence_diagnostic_v1(payload, root=root)


def validate_participant_dependence_diagnostic_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ParticipantDependenceDiagnosticError("diagnostic must be an object")
    value = dict(payload)
    expected_keys = {
        "schema_version",
        "result_state",
        "created_at_utc",
        "inputs",
        "population",
        "decision",
        "source_locks",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected_keys:
        raise ParticipantDependenceDiagnosticError("diagnostic structure changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise ParticipantDependenceDiagnosticError("diagnostic identity changed")
    try:
        parse_rfc3339(value.get("created_at_utc"))
    except (TypeError, ValueError) as exc:
        raise ParticipantDependenceDiagnosticError("diagnostic timestamp is invalid") from exc
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != sha256_canonical_object(unsigned):
        raise ParticipantDependenceDiagnosticError("diagnostic artifact hash changed")
    _, snapshot, statistics = _diagnostic_inputs(root)
    expected_inputs = {
        "development_snapshot_manifest_locator": development_snapshot.DEFAULT_MANIFEST.as_posix(),
        "development_snapshot_manifest_raw_sha256": snapshot[
            "development_snapshot_manifest_raw_sha256"
        ],
        "development_snapshot_manifest_artifact_sha256": snapshot[
            "development_snapshot_manifest_artifact_sha256"
        ],
        "development_snapshot_payload_raw_sha256": snapshot[
            "development_snapshot_payload_raw_sha256"
        ],
        "players_parquet_locator": PLAYERS_LOCATOR.as_posix(),
        "players_parquet_raw_sha256": _sha256_path(root / PLAYERS_LOCATOR),
    }
    if value.get("inputs") != expected_inputs:
        raise ParticipantDependenceDiagnosticError("diagnostic inputs drifted")
    population = value.get("population")
    if not isinstance(population, Mapping) or dict(population) != {
        "development_rows": statistics["snapshot_maps"],
        **statistics,
    }:
        raise ParticipantDependenceDiagnosticError("diagnostic population drifted")
    fraction = float(population["exact_assignment_coverage_fraction"])
    if not math.isclose(
        fraction,
        population["maps_with_exact_ten_unique_players_and_roles"]
        / population["snapshot_maps"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ParticipantDependenceDiagnosticError("diagnostic coverage does not reconcile")
    graph = population["component_graph"]
    if (
        graph["all_valid_maps_in_one_component"] is not True
        or graph["atomic_component_temporal_split_available"] is not False
    ):
        raise ParticipantDependenceDiagnosticError("participant component conclusion changed")
    expected_decision = {
        "participant_identity_available_for_development_diagnostic": True,
        "participant_identity_complete_for_all_snapshot_maps": False,
        "participant_atomic_component_split_available": False,
        "participant_dependence_support_verified": False,
        "reliability_gate_passed": False,
        "promotion_eligible": False,
        "required_next_evidence": (
            "A predeclared and independently reviewed participant-dependence "
            "method that does not treat the single connected component as "
            "independent observations, followed by untouched future evaluation."
        ),
    }
    if value.get("decision") != expected_decision:
        raise ParticipantDependenceDiagnosticError("diagnostic decision changed")
    expected_sources = [
        _source_record(root, SOURCE_LOCATOR),
        _source_record(root, DEVELOPMENT_SNAPSHOT_SOURCE_LOCATOR),
    ]
    if value.get("source_locks") != expected_sources:
        raise ParticipantDependenceDiagnosticError("diagnostic source locks changed")
    authority = value.get("authority")
    if not isinstance(authority, Mapping) or any(authority.values()):
        raise ParticipantDependenceDiagnosticError("diagnostic exceeds authority")
    if value.get("claim_ceiling") != (
        "Retrospective participant-dependence diagnostic only. It does not "
        "validate Draft Score, reliability, probability, odds, expected value, "
        "recommendations, or betting."
    ):
        raise ParticipantDependenceDiagnosticError("diagnostic claim ceiling changed")
    return value


__all__ = [
    "DEFAULT_OUTPUT",
    "ParticipantDependenceDiagnosticError",
    "build_participant_dependence_diagnostic_v1",
    "validate_participant_dependence_diagnostic_v1",
]
