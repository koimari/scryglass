"""Strict private bridge from the accepted G1 LPL snapshot to Player Rating.

This adapter intentionally does *not* call the synthetic Player Rating replay.
The G1 rows contain observed map participants and source-local timestamps, not
the canonical identity, pre-event roster, RFC3339 availability, or player
performance receipts required by that model.  It instead exposes a frozen,
typed development input for the next private runner to enrich under its own
authority boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
import hashlib
import json
import os
import stat

from lol_kills.v2.data.common import ROLES, canonical_json_bytes, sha256_bytes


ROOT = Path(__file__).resolve().parents[4]
MANIFEST_LOCATOR = "data/lol/v2/snapshots/real-v1/lpl-private-development-manifest.json"
ROWS_LOCATOR = "data/lol/v2/snapshots/real-v1/lpl-private-development-rows.jsonl"
ALLOWED_FOLDS = ("TRAIN", "DEVELOPMENT", "VALIDATION")
EMBARGO_HOURS = 48


class PrivatePlayerRatingAdapterError(ValueError):
    """Raised when the accepted private G1 input cannot be safely bridged."""


@dataclass(frozen=True)
class AcceptedG1Pins:
    """Independent identities for the only G1 snapshot this adapter accepts."""

    manifest_sha256: str
    rows_sha256: str
    selected_target_sha256: str
    split_payload_sha256: str


ACCEPTED_G1_PINS = AcceptedG1Pins(
    manifest_sha256="3af87fffb2b32fd95aeb920409abe0254fa158b3dc7f079650b3472731d4ff72",
    rows_sha256="4ed79abb0b2471a666ab5643b91edf33c2fdde19e361c456aa589d2e9a4df846",
    selected_target_sha256="4c332fa4e6cb155341bcffd83bd0ee1be2e04f3b5950b8a7745931253dd8bd2d",
    split_payload_sha256="469c8d2c568a6a4480db277bf41f7eacf72964e33997f0a4e1f53f60285cd3e4",
)


@dataclass(frozen=True)
class PlayerLineupObservation:
    """One observed player-role row, grouped by a frozen map observation."""

    observation_id: str
    source_game_id: str
    fold_id: str
    game_side: str
    role: str
    source_player_id: str
    source_team_id: str
    blue_win: int
    ordered_origin_map_ids: tuple[str, ...]
    ordered_origin_sha256: str


@dataclass(frozen=True)
class MapObservation:
    """A target map and its two exact observed role-ordered lineups."""

    source_game_id: str
    source_series_id: str
    fold_id: str
    source_local_event_start: str
    source_blue_result_id: str
    source_red_result_id: str
    blue_win: int
    player_observations: tuple[PlayerLineupObservation, ...]
    ordered_origin_map_ids: tuple[str, ...]
    ordered_origin_sha256: str


@dataclass(frozen=True)
class PrivatePlayerRatingFold:
    fold_id: str
    map_observations: tuple[MapObservation, ...]
    ordered_map_ids_sha256: str
    ordered_origin_identities_sha256: str


@dataclass(frozen=True)
class PrivatePlayerRatingInput:
    """G2 handoff object; explicitly not a fit-ready public Player Rating input."""

    schema_version: str
    manifest_sha256: str
    rows_sha256: str
    selected_target_sha256: str
    split_payload_sha256: str
    folds: tuple[PrivatePlayerRatingFold, ...]
    map_count: int
    player_observation_count: int
    claim_ceiling: Mapping[str, bool]


CLAIM_CEILING: Mapping[str, bool] = {
    "private_development_model_fit": True,
    "private_rank_selection": True,
    "historical_live_ingest": False,
    "pre_event_roster_authority": False,
    "prediction": False,
    "production": False,
    "publication": False,
    "promotion": False,
    "sota": False,
    "final_holdout": False,
}


def _sha(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise PrivatePlayerRatingAdapterError(f"{label} must be a lowercase sha256 digest")
    return value


def _safe_repo_file(root: Path, locator: str) -> Path:
    path = PurePosixPath(locator)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise PrivatePlayerRatingAdapterError("G1 locator must be a contained relative POSIX path")
    current = root.resolve()
    for index, part in enumerate(path.parts):
        current = current / part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise PrivatePlayerRatingAdapterError(f"G1 artifact is missing: {locator}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise PrivatePlayerRatingAdapterError(f"G1 artifact must not contain a symlink: {locator}")
        if index < len(path.parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise PrivatePlayerRatingAdapterError(f"G1 artifact parent must be a directory: {locator}")
    metadata = os.lstat(current)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise PrivatePlayerRatingAdapterError(f"G1 artifact must be an unaliased regular file: {locator}")
    return current


def _read_rows(raw: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line:
            raise PrivatePlayerRatingAdapterError(f"rows contain a blank line at {number}")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise PrivatePlayerRatingAdapterError(f"row {number} is not JSON") from error
        if not isinstance(row, dict):
            raise PrivatePlayerRatingAdapterError(f"row {number} must be an object")
        rows.append(row)
    if not rows:
        raise PrivatePlayerRatingAdapterError("rows must be non-empty")
    return rows


def _validate_manifest(manifest: Mapping[str, Any], pins: AcceptedG1Pins) -> None:
    unsigned = dict(manifest)
    self_hash = unsigned.pop("manifest_sha256", None)
    if self_hash != _sha(unsigned):
        raise PrivatePlayerRatingAdapterError("G1 manifest self hash mismatch")
    if self_hash != pins.manifest_sha256:
        raise PrivatePlayerRatingAdapterError("G1 manifest does not match the accepted pin")
    if manifest.get("schema_version") != "scryglass:real-v1-lpl-private-g2-input:v1":
        raise PrivatePlayerRatingAdapterError("unexpected G1 manifest schema")
    if manifest.get("rows_locator") != ROWS_LOCATOR:
        raise PrivatePlayerRatingAdapterError("G1 manifest rows locator mismatch")
    if manifest.get("rows_sha256") != pins.rows_sha256:
        raise PrivatePlayerRatingAdapterError("G1 manifest rows hash mismatch")
    if manifest.get("canonical_selected_target_rows_sha256") != pins.selected_target_sha256:
        raise PrivatePlayerRatingAdapterError("G1 selected target digest mismatch")
    authority = manifest.get("target_authority")
    if not isinstance(authority, Mapping) or authority.get("split_payload_sha256") != pins.split_payload_sha256:
        raise PrivatePlayerRatingAdapterError("G1 split payload binding mismatch")
    if manifest.get("final_holdout", {}).get("accessed") is not False:
        raise PrivatePlayerRatingAdapterError("G1 final holdout must remain unread")
    scope = manifest.get("claim_scope")
    if not isinstance(scope, Mapping) or scope.get("state") != "PRIVATE_RETROSPECTIVE_MODEL_FIT_AND_RANK_SELECTION_AVAILABLE":
        raise PrivatePlayerRatingAdapterError("G1 private claim scope is not accepted")
    if not {"private_model_fit", "private_rank_selection"} <= set(scope.get("available_claims", ())):
        raise PrivatePlayerRatingAdapterError("G1 does not authorize the narrow private handoff")
    policy = manifest.get("distribution_policy")
    if not isinstance(policy, Mapping) or policy.get("public_pack_eligible") is not False or policy.get("vercel_deploy_eligible") is not False:
        raise PrivatePlayerRatingAdapterError("G1 private artifact must remain excluded from public distribution")


def _target_receipt(row: Mapping[str, Any]) -> dict[str, Any]:
    target = row.get("target")
    if not isinstance(target, Mapping):
        raise PrivatePlayerRatingAdapterError("row target is missing")
    game_id = row.get("source_game_id")
    fold = row.get("partition")
    timestamp = row.get("source_local_event_start")
    cluster = row.get("dependence_cluster_diagnostic")
    if not all(isinstance(value, str) and value for value in (game_id, fold, timestamp, cluster)):
        raise PrivatePlayerRatingAdapterError("row target identity is incomplete")
    outcome = target.get("y_blue_win")
    if type(outcome) is not int or outcome not in (0, 1):
        raise PrivatePlayerRatingAdapterError("row target must be binary")
    blue_id = target.get("source_blue_result_id")
    red_id = target.get("source_red_result_id")
    if blue_id != f"oe-team-row:{game_id}:100" or red_id != f"oe-team-row:{game_id}:200":
        raise PrivatePlayerRatingAdapterError("row target/source result identity mismatch")
    return {
        "game_id": game_id,
        "split": fold,
        "oe_date_naive": timestamp,
        "y_blue_win": outcome,
        "source_blue_result_id": blue_id,
        "source_red_result_id": red_id,
        "dependence_cluster_id": cluster,
    }


def _validate_lineups(row: Mapping[str, Any]) -> tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...]:
    lineups = row.get("observed_lineups")
    if not isinstance(lineups, list) or len(lineups) != 2:
        raise PrivatePlayerRatingAdapterError("row must contain exactly two observed lineups")
    canonical: list[tuple[str, str, tuple[tuple[str, str], ...]]] = []
    all_players: set[str] = set()
    teams: set[str] = set()
    expected_sides = ("blue", "red")
    for expected_side, lineup in zip(expected_sides, lineups):
        if not isinstance(lineup, Mapping) or lineup.get("observed_game_side") != expected_side:
            raise PrivatePlayerRatingAdapterError("observed lineups must be ordered blue then red")
        team_id = lineup.get("team_id")
        role_map = lineup.get("player_ids_by_role")
        if not isinstance(team_id, str) or not team_id or not isinstance(role_map, Mapping) or set(role_map) != set(ROLES):
            raise PrivatePlayerRatingAdapterError("observed lineup identity or role closure is invalid")
        ordered = tuple((role, role_map[role]) for role in ROLES)
        if any(not isinstance(player_id, str) or not player_id for _role, player_id in ordered):
            raise PrivatePlayerRatingAdapterError("stable player identity is missing")
        player_ids = {player_id for _role, player_id in ordered}
        if len(player_ids) != len(ROLES) or all_players.intersection(player_ids):
            raise PrivatePlayerRatingAdapterError("repeated or ambiguous stable player identity")
        if team_id in teams:
            raise PrivatePlayerRatingAdapterError("repeated or ambiguous stable team identity")
        all_players.update(player_ids)
        teams.add(team_id)
        canonical.append((expected_side, team_id, ordered))
    return tuple(canonical)


def _source_start(row: Mapping[str, Any]) -> str:
    value = row.get("source_local_event_start")
    if not isinstance(value, str) or not value or value.endswith("Z") or "+" in value[10:]:
        raise PrivatePlayerRatingAdapterError("source-local event time must remain timezone-naive")
    return value


def _expected_origins(rows: Sequence[Mapping[str, Any]], index: int) -> tuple[str, ...]:
    current = rows[index]
    current_start = _source_start(current)
    current_series = current.get("source_series_id")
    if not isinstance(current_series, str) or not current_series:
        raise PrivatePlayerRatingAdapterError("source series identity is missing")
    expected: list[str] = []
    for prior in rows[:index]:
        prior_start = _source_start(prior)
        prior_series = prior.get("source_series_id")
        prior_id = prior.get("source_game_id")
        if not isinstance(prior_series, str) or not isinstance(prior_id, str):
            raise PrivatePlayerRatingAdapterError("origin identity is missing")
        # Both strings are fixed-width, timezone-naive ISO timestamps.  The
        # adapter keeps that source-local contract intact rather than inventing
        # RFC3339 timestamps for Player Rating.
        from datetime import datetime, timedelta
        if prior_series != current_series and datetime.fromisoformat(prior_start) + timedelta(hours=EMBARGO_HOURS) < datetime.fromisoformat(current_start):
            expected.append(prior_id)
    return tuple(expected)


def build_private_player_rating_input(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    pins: AcceptedG1Pins,
    rows_sha256: str,
) -> PrivatePlayerRatingInput:
    """Validate frozen G1 payloads and return a deterministic typed handoff."""

    _validate_manifest(manifest, pins)
    if _require_sha256(rows_sha256, "rows_sha256") != pins.rows_sha256:
        raise PrivatePlayerRatingAdapterError("rows do not match the accepted pin")
    if not rows:
        raise PrivatePlayerRatingAdapterError("G1 rows are empty")
    source_order = [(_source_start(row), row.get("source_game_id")) for row in rows]
    if any(not isinstance(game_id, str) or not game_id for _time, game_id in source_order):
        raise PrivatePlayerRatingAdapterError("source game identity is missing")
    if source_order != sorted(source_order):
        raise PrivatePlayerRatingAdapterError("G1 rows are not in frozen source order")
    source_ids = [game_id for _time, game_id in source_order]
    if len(set(source_ids)) != len(source_ids):
        raise PrivatePlayerRatingAdapterError("repeated stable source game identity")

    target_rows: list[dict[str, Any]] = []
    by_fold: dict[str, list[MapObservation]] = {fold: [] for fold in ALLOWED_FOLDS}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise PrivatePlayerRatingAdapterError("G1 row must be an object")
        fold = row.get("partition")
        if fold not in ALLOWED_FOLDS:
            raise PrivatePlayerRatingAdapterError("final or unknown split row is forbidden")
        if row.get("participant_lineup_kind") != "OBSERVED_MAP_PARTICIPANTS_NOT_PRE_EVENT_ROSTER_AUTHORITY":
            raise PrivatePlayerRatingAdapterError("unexpected roster-authority elevation")
        receipt = _target_receipt(row)
        target_rows.append(receipt)
        lineups = _validate_lineups(row)
        actual_origins = row.get("eligible_prior_origin_map_ids")
        if not isinstance(actual_origins, list) or any(not isinstance(origin, str) or not origin for origin in actual_origins):
            raise PrivatePlayerRatingAdapterError("ordered origin identities are invalid")
        expected_origins = _expected_origins(rows, index)
        if tuple(actual_origins) != expected_origins:
            raise PrivatePlayerRatingAdapterError("ordered origins are missing, extra, reordered, or otherwise invalid")
        if row.get("eligible_prior_origin_count") != len(expected_origins):
            raise PrivatePlayerRatingAdapterError("ordered origin count mismatch")
        origin_sha256 = _sha(list(expected_origins))
        game_id = receipt["game_id"]
        observations: list[PlayerLineupObservation] = []
        for side, team_id, players in lineups:
            for role, player_id in players:
                observations.append(PlayerLineupObservation(
                    observation_id=f"{game_id}:{side}:{role}",
                    source_game_id=game_id,
                    fold_id=fold,
                    game_side=side,
                    role=role,
                    source_player_id=player_id,
                    source_team_id=team_id,
                    blue_win=receipt["y_blue_win"],
                    ordered_origin_map_ids=expected_origins,
                    ordered_origin_sha256=origin_sha256,
                ))
        source_series_id = row.get("source_series_id")
        if not isinstance(source_series_id, str) or not source_series_id:
            raise PrivatePlayerRatingAdapterError("source series identity is missing")
        by_fold[fold].append(MapObservation(
            source_game_id=game_id,
            source_series_id=source_series_id,
            fold_id=fold,
            source_local_event_start=_source_start(row),
            source_blue_result_id=receipt["source_blue_result_id"],
            source_red_result_id=receipt["source_red_result_id"],
            blue_win=receipt["y_blue_win"],
            player_observations=tuple(observations),
            ordered_origin_map_ids=expected_origins,
            ordered_origin_sha256=origin_sha256,
        ))
    if _sha(sorted(target_rows, key=lambda item: item["game_id"])) != pins.selected_target_sha256:
        raise PrivatePlayerRatingAdapterError("selected target receipt digest does not match accepted pin")
    expected_partition_counts = manifest.get("coverage", {}).get("partition_counts")
    actual_partition_counts = dict(sorted((fold, len(items)) for fold, items in by_fold.items()))
    if expected_partition_counts != actual_partition_counts:
        raise PrivatePlayerRatingAdapterError("fold coverage differs from the frozen G1 manifest")

    folds: list[PrivatePlayerRatingFold] = []
    for fold in ALLOWED_FOLDS:
        maps = tuple(by_fold[fold])
        if not maps:
            raise PrivatePlayerRatingAdapterError(f"frozen fold has no observations: {fold}")
        ordered_map_ids = [item.source_game_id for item in maps]
        origin_identities = [
            {"source_game_id": item.source_game_id, "ordered_origin_map_ids": list(item.ordered_origin_map_ids)}
            for item in maps
        ]
        folds.append(PrivatePlayerRatingFold(
            fold_id=fold,
            map_observations=maps,
            ordered_map_ids_sha256=_sha(ordered_map_ids),
            ordered_origin_identities_sha256=_sha(origin_identities),
        ))
    return PrivatePlayerRatingInput(
        schema_version="scryglass:player-rating-private-g2-observed-lineups:v1",
        manifest_sha256=pins.manifest_sha256,
        rows_sha256=pins.rows_sha256,
        selected_target_sha256=pins.selected_target_sha256,
        split_payload_sha256=pins.split_payload_sha256,
        folds=tuple(folds),
        map_count=len(rows),
        player_observation_count=len(rows) * 10,
        claim_ceiling=CLAIM_CEILING,
    )


def load_accepted_lpl_private_player_rating_input(root: Path = ROOT) -> PrivatePlayerRatingInput:
    """Load exactly the independently pinned accepted G1 private snapshot."""

    manifest_path = _safe_repo_file(root, MANIFEST_LOCATOR)
    rows_path = _safe_repo_file(root, ROWS_LOCATOR)
    manifest_raw = manifest_path.read_bytes()
    rows_raw = rows_path.read_bytes()
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PrivatePlayerRatingAdapterError("G1 manifest must be JSON") from error
    if not isinstance(manifest, Mapping):
        raise PrivatePlayerRatingAdapterError("G1 manifest must be an object")
    return build_private_player_rating_input(
        manifest,
        _read_rows(rows_raw),
        pins=ACCEPTED_G1_PINS,
        rows_sha256=sha256_bytes(rows_raw),
    )


__all__ = [
    "ACCEPTED_G1_PINS",
    "CLAIM_CEILING",
    "AcceptedG1Pins",
    "MapObservation",
    "PlayerLineupObservation",
    "PrivatePlayerRatingAdapterError",
    "PrivatePlayerRatingFold",
    "PrivatePlayerRatingInput",
    "build_private_player_rating_input",
    "load_accepted_lpl_private_player_rating_input",
]
