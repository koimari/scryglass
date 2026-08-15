"""Fail-closed real-v1 pre-event input freeze.

This module deliberately stops before fitting, scoring, or reading any sealed
holdout outcome.  Its generic readiness packet contains no labels.  The narrow
LPL adapter is separately authorized to attach non-final private retrospective
targets for model fitting and rank selection only.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import hashlib
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
from typing import Any, Iterable, Mapping

from .common import ROLES, canonical_json_bytes, parse_rfc3339, sha256_bytes
from .identity import IdentityCrosswalkRow, IdentityRegistry
from .rosters import RosterRegistry, RosterRow


SCHEMA_VERSION = "scryglass:real-v1-pre-event-input:v1"
PACKET_KIND = "scryglass:real-v1-pre-event-readiness:v1"
EMBARGO_HOURS = 48
ALLOWED_PARTITIONS = ("TRAIN", "DEVELOPMENT", "VALIDATION")
PRIVATE_RIGHTS_STATUS = "PRIVATE_REVIEWED"
PRIVATE_TARGET_STATUS = "PRIVATE_VERIFIED"
SEALED_FINAL_STATUS = "SEALED_UNREAD"
KOI_MARI_AUTHORITY_RAW_SHA256 = "b1d0a6e37abb9a74dee8689dc19ab54d30fd15516bd4ee454906a075d8f20788"
KOI_MARI_AUTHORITY_LOCATOR = "data/lol/v2/models/draft-interactions/oe-private-target-authority-2026-07-29.json"
LEGACY_KOI_MARI_AUTHORITY_LOCATOR = "data/lol/v2/models/draft-interactions/oe-private-target-authority.json"
KOI_MARI_EVIDENCE_PAYLOAD_SHA256 = "6697ed142324f86e9b233c4a2b36dd501584e7e64449bb6cd9404f6a367d74f9"
KOI_MARI_SPLIT_PAYLOAD_SHA256 = "469c8d2c568a6a4480db277bf41f7eacf72964e33997f0a4e1f53f60285cd3e4"
EXPECTED_LPL_PRIVATE_PARTITION_COUNTS = {"TRAIN": 805, "DEVELOPMENT": 214, "VALIDATION": 207}
EXPECTED_LPL_PRIVATE_SOURCE_FAMILIES = 471
LPL_FAMILY_RE = re.compile(r"^(?P<bmid>[0-9]+)-(?P=bmid)_game_(?P<ordinal>[1-5])$")
LPL_URL_BMID_RE = re.compile(r"(?:[?&])bmid=(?P<bmid>[0-9]+)(?:[&#]|$)")
REPO_ROOT = Path(__file__).resolve().parents[3]
G0_BENCHMARK_CONTRACT_PATH = REPO_ROOT / "data/lol/v2/evaluation/real-v1/benchmark-contract.json"
G0_BENCHMARK_CONTRACT_RAW_SHA256 = "b77fe451105d6e216b71928ad2381117c1ba5d0a5bce30b0b658414ab8559128"
G0_BASELINE_REGISTRY_PATH = REPO_ROOT / "data/lol/v2/evaluation/real-v1/baseline-registry.json"
G0_BASELINE_REGISTRY_RAW_SHA256 = "73805104e92242ee49952df7e167f34a14566c79b43bbf1209371cdd3248e298"


class RealSpineError(ValueError):
    """Raised when a claimed real-v1 input receipt is not defensible."""


@dataclass(frozen=True)
class Blocker:
    code: str
    scope: str
    claim_effect: str

    def to_payload(self) -> dict[str, str]:
        return asdict(self)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RealSpineError(f"{name} keys mismatch; missing={missing}; extra={extra}")


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RealSpineError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    value = _require_text(value, name)
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise RealSpineError(f"{name} must be a sha256 digest")
    return value.lower()


def _require_time(value: Any, name: str) -> datetime:
    try:
        return parse_rfc3339(_require_text(value, name))
    except (TypeError, ValueError) as error:
        raise RealSpineError(f"{name} must be RFC3339 UTC") from error


def _safe_relative_path(value: Any, name: str) -> str:
    raw = _require_text(value, name)
    if "\\" in raw:
        raise RealSpineError(f"{name} must use a relative POSIX path")
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise RealSpineError(f"{name} must be a contained relative path")
    return path.as_posix()


def _safe_receipt_file(root: Path, locator: str) -> Path:
    """Resolve a receipt file without following symlinks or hardlink aliases."""

    base = root.resolve()
    current = base
    parts = PurePosixPath(locator).parts
    for index, component in enumerate(parts):
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise RealSpineError(f"receipt file is missing: {locator}") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RealSpineError(f"receipt path must not contain a symlink: {locator}")
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise RealSpineError(f"receipt path parent must be a directory: {locator}")
    metadata = os.lstat(current)
    if not stat.S_ISREG(metadata.st_mode):
        raise RealSpineError(f"receipt path must be a regular file: {locator}")
    if metadata.st_nlink != 1:
        raise RealSpineError(f"receipt path must not be hard-linked: {locator}")
    resolved = current.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as error:  # defensive after lstat checks
        raise RealSpineError(f"receipt file escapes root: {locator}") from error
    return resolved


def _safe_output_target(path: Path) -> Path:
    """Preflight every existing component before an atomic local replacement."""

    absolute = path.absolute()
    anchor = Path(absolute.anchor)
    current = anchor
    for part in absolute.parts[1:-1]:
        current = current / part
        if current.exists():
            metadata = os.lstat(current)
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise RealSpineError(f"output parent must be a real directory: {path}")
        else:
            current.mkdir()
    if absolute.exists():
        metadata = os.lstat(absolute)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise RealSpineError(f"output path must be a regular file when it exists: {path}")
        if metadata.st_nlink != 1:
            raise RealSpineError(f"output path must not be hard-linked: {path}")
    return absolute


def _atomic_safe_write(path: Path, data: bytes) -> None:
    target = _safe_output_target(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path = Path(temporary)
        metadata = os.lstat(temporary_path)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RealSpineError("temporary output lost regular-file identity")
        # Recheck the destination immediately before replacement.
        _safe_output_target(target)
        os.replace(temporary_path, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _stage_safe_output(target: Path, data: bytes) -> Path:
    """Stage one regular-file replacement after its destination was preflighted."""

    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        staged = Path(temporary)
        metadata = os.lstat(staged)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise RealSpineError("staged output lost regular-file identity")
        return staged
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _atomic_safe_write_many(outputs: Iterable[tuple[Path, bytes]]) -> None:
    """Commit related outputs together, with no first-file write on late failure.

    Every destination is preflighted before staging begins.  Replacement still
    cannot be made perfectly crash-atomic across two files, so existing files
    are moved aside and restored if a later replacement fails.
    """

    pairs = list(outputs)
    if not pairs:
        raise RealSpineError("transaction requires at least one output")
    targets = [_safe_output_target(path) for path, _data in pairs]
    if len(set(targets)) != len(targets):
        raise RealSpineError("transaction output destinations must be distinct")
    staged: list[Path] = []
    backups: list[Path | None] = [None] * len(targets)
    committed: list[bool] = [False] * len(targets)
    try:
        for target, (_path, data) in zip(targets, pairs):
            staged.append(_stage_safe_output(target, data))
        # Detect an alias or directory swap occurring while output bytes were
        # being staged, before touching either final destination.
        for target in targets:
            _safe_output_target(target)
        for index, target in enumerate(targets):
            if target.exists():
                descriptor, backup_name = tempfile.mkstemp(prefix=f".{target.name}.backup.", dir=str(target.parent))
                os.close(descriptor)
                backup = Path(backup_name)
                os.unlink(backup)
                os.replace(target, backup)
                backups[index] = backup
            os.replace(staged[index], target)
            committed[index] = True
    except BaseException:
        for index in reversed(range(len(targets))):
            target = targets[index]
            backup = backups[index]
            if committed[index] and target.exists():
                os.unlink(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    else:
        # Backups are no longer part of correctness once every replacement has
        # succeeded.  Best-effort cleanup must not trigger a rollback.
        for backup in backups:
            if backup is not None and backup.exists():
                try:
                    os.unlink(backup)
                except OSError:
                    pass
    finally:
        for staged_path in staged:
            if staged_path.exists():
                os.unlink(staged_path)
        for backup in backups:
            if backup is not None and backup.exists():
                os.unlink(backup)


def _validate_source_receipts(
    values: Any,
    *,
    evidence_root: Path | None,
) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise RealSpineError("source_receipts must be a non-empty list")
    expected = {
        "receipt_id", "source_id", "source_snapshot_id", "source_snapshot_row_id",
        "source_record_id", "source_content_sha256", "source_observed_at",
        "rights_status", "evidence_locator", "evidence_sha256",
    }
    receipts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise RealSpineError(f"source_receipts[{index}] must be an object")
        _require_exact_keys(raw, expected, f"source_receipts[{index}]")
        receipt = dict(raw)
        identifier = _require_text(receipt["receipt_id"], "source receipt_id")
        if identifier in receipts:
            raise RealSpineError(f"duplicate source receipt_id: {identifier}")
        for field in ("source_id", "source_snapshot_id", "source_snapshot_row_id", "source_record_id"):
            _require_text(receipt[field], f"source receipt {field}")
        receipt["source_content_sha256"] = _require_sha256(receipt["source_content_sha256"], "source_content_sha256")
        receipt["evidence_sha256"] = _require_sha256(receipt["evidence_sha256"], "evidence_sha256")
        _require_time(receipt["source_observed_at"], "source_observed_at")
        if receipt["rights_status"] != PRIVATE_RIGHTS_STATUS:
            raise RealSpineError("source receipt rights_status must be PRIVATE_REVIEWED")
        locator = _safe_relative_path(receipt["evidence_locator"], "evidence_locator")
        receipt["evidence_locator"] = locator
        if evidence_root is not None:
            path = _safe_receipt_file(evidence_root, locator)
            if sha256_bytes(path.read_bytes()) != receipt["evidence_sha256"]:
                raise RealSpineError(f"receipt evidence hash drift: {locator}")
        receipts[identifier] = receipt
    return receipts


def _identity_registry(values: Any, source_receipts: Mapping[str, Mapping[str, Any]]) -> IdentityRegistry:
    if not isinstance(values, list) or not values:
        raise RealSpineError("identity_rows must be a non-empty list")
    rows: list[IdentityCrosswalkRow] = []
    expected = {
        "row_id", "receipt_id", "entity_type", "canonical_id", "canonical_name", "alias",
        "effective_from", "effective_to", "precedence", "observed_at", "source_updated_at", "available_at",
    }
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise RealSpineError(f"identity_rows[{index}] must be an object")
        _require_exact_keys(raw, expected, f"identity_rows[{index}]")
        receipt = source_receipts.get(_require_text(raw["receipt_id"], "identity receipt_id"))
        if receipt is None:
            raise RealSpineError("identity row references unknown source receipt")
        rows.append(IdentityCrosswalkRow(
            row_id=_require_text(raw["row_id"], "identity row_id"),
            entity_type=_require_text(raw["entity_type"], "identity entity_type"),
            canonical_id=_require_text(raw["canonical_id"], "identity canonical_id"),
            canonical_name=_require_text(raw["canonical_name"], "identity canonical_name"),
            source_name=receipt["source_id"], source_id=receipt["source_id"],
            source_snapshot_id=receipt["source_snapshot_id"],
            source_snapshot_row_id=receipt["source_snapshot_row_id"],
            source_snapshot_content_sha256=receipt["source_content_sha256"],
            source_record_id=receipt["source_record_id"], alias=_require_text(raw["alias"], "identity alias"),
            effective_from=_require_text(raw["effective_from"], "identity effective_from"),
            effective_to=raw["effective_to"], precedence=raw["precedence"],
            observed_at=_require_text(raw["observed_at"], "identity observed_at"),
            source_updated_at=_require_text(raw["source_updated_at"], "identity source_updated_at"),
            available_at=_require_text(raw["available_at"], "identity available_at"),
        ))
    registry = IdentityRegistry.from_rows(rows)
    as_of = max(parse_rfc3339(row.available_at) for row in rows)
    collisions = registry.audit_collisions(as_of)
    if collisions:
        aliases = ",".join(sorted(collision.alias for collision in collisions))
        raise RealSpineError(f"identity collision blocks real input: {aliases}")
    return registry


def _roster_registry(values: Any, source_receipts: Mapping[str, Mapping[str, Any]]) -> RosterRegistry:
    if not isinstance(values, list) or not values:
        raise RealSpineError("roster_rows must be a non-empty list")
    expected = {
        "row_id", "receipt_id", "roster_id", "organization_id", "organization_name", "role",
        "player_id", "player_name", "effective_from", "effective_to", "precedence",
        "source_updated_at", "observed_at", "available_at", "is_substitute", "is_provisional",
    }
    registry = RosterRegistry.empty()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise RealSpineError(f"roster_rows[{index}] must be an object")
        _require_exact_keys(raw, expected, f"roster_rows[{index}]")
        receipt = source_receipts.get(_require_text(raw["receipt_id"], "roster receipt_id"))
        if receipt is None:
            raise RealSpineError("roster row references unknown source receipt")
        registry = registry.append(RosterRow(
            row_id=_require_text(raw["row_id"], "roster row_id"),
            roster_id=_require_text(raw["roster_id"], "roster_id"),
            organization_id=_require_text(raw["organization_id"], "organization_id"),
            organization_name=_require_text(raw["organization_name"], "organization_name"),
            role=_require_text(raw["role"], "roster role"),
            player_id=_require_text(raw["player_id"], "roster player_id"),
            player_name=_require_text(raw["player_name"], "roster player_name"),
            source_id=receipt["source_id"], source_name=receipt["source_id"],
            source_record_id=receipt["source_record_id"],
            source_snapshot_id=receipt["source_snapshot_id"],
            source_snapshot_row_id=receipt["source_snapshot_row_id"],
            source_snapshot_content_sha256=receipt["source_content_sha256"],
            effective_from=_require_text(raw["effective_from"], "roster effective_from"),
            effective_to=raw["effective_to"], precedence=raw["precedence"],
            source_updated_at=_require_text(raw["source_updated_at"], "roster source_updated_at"),
            observed_at=_require_text(raw["observed_at"], "roster observed_at"),
            available_at=_require_text(raw["available_at"], "roster available_at"),
            is_substitute=raw["is_substitute"], is_provisional=raw["is_provisional"],
        ))
    return registry


def _validate_target_receipts(values: Any, source_receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list) or not values:
        raise RealSpineError("target_authority_receipts must be a non-empty list")
    expected = {
        "receipt_id", "source_receipt_id", "target_record_id", "target_payload_sha256",
        "target_available_at", "correction_status", "authority_status", "authority_locator",
        "authority_raw_sha256", "evidence_payload_sha256", "split_payload_sha256",
    }
    receipts: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise RealSpineError(f"target_authority_receipts[{index}] must be an object")
        _require_exact_keys(raw, expected, f"target_authority_receipts[{index}]")
        receipt_id = _require_text(raw["receipt_id"], "target receipt_id")
        if receipt_id in receipts:
            raise RealSpineError(f"duplicate target receipt_id: {receipt_id}")
        if _require_text(raw["source_receipt_id"], "target source_receipt_id") not in source_receipts:
            raise RealSpineError("target receipt references unknown source receipt")
        if raw["authority_status"] != PRIVATE_TARGET_STATUS:
            raise RealSpineError("target authority must be PRIVATE_VERIFIED")
        if raw["correction_status"] not in {"ORIGINAL", "CORRECTED_SUPERSEDES"}:
            raise RealSpineError("target correction_status is invalid")
        _require_text(raw["target_record_id"], "target_record_id")
        _require_sha256(raw["target_payload_sha256"], "target_payload_sha256")
        _require_time(raw["target_available_at"], "target_available_at")
        authority_locator = _safe_relative_path(raw["authority_locator"], "authority_locator")
        authority_raw_sha256 = _require_sha256(
            raw["authority_raw_sha256"], "authority_raw_sha256"
        )
        authority = validate_koi_mari_authority(
            _resolve_koi_mari_authority_path(
                authority_locator,
                expected_raw_sha256=authority_raw_sha256,
            ),
            expected_raw_sha256=authority_raw_sha256,
        )
        if authority.get("evidence_payload_sha256") != _require_sha256(raw["evidence_payload_sha256"], "evidence_payload_sha256"):
            raise RealSpineError("target authority evidence payload binding mismatch")
        if authority.get("split_payload_sha256") != _require_sha256(raw["split_payload_sha256"], "split_payload_sha256"):
            raise RealSpineError("target authority split payload binding mismatch")
        normalized = dict(raw)
        normalized["authority_locator"] = authority_locator
        receipts[receipt_id] = normalized
    return receipts


def validate_koi_mari_authority(authority_path: Path, *, expected_raw_sha256: str | None = None) -> dict[str, Any]:
    """Validate the independently reviewed private retrospective authority bytes."""

    raw = authority_path.read_bytes()
    actual = sha256_bytes(raw)
    if expected_raw_sha256 is not None and actual != expected_raw_sha256:
        raise RealSpineError("target authority raw bytes do not match pinned hash")
    try:
        authority = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealSpineError("target authority receipt must be JSON") from error
    if not isinstance(authority, Mapping):
        raise RealSpineError("target authority receipt must be an object")
    if authority.get("reviewer_identity") != "KOI_MARI" or authority.get("approval_scope") != "private_retrospective_oe_target_v1":
        raise RealSpineError("target authority is outside the KOI_MARI private retrospective scope")
    approved_actions = authority.get("approved_actions", ())
    if authority.get("decision") != "approve" or not {"model_fit", "rank_selection"} <= set(approved_actions):
        raise RealSpineError("target authority does not approve model_fit and rank_selection")
    if authority.get("final_temporal_holdout_sealed") is not True:
        raise RealSpineError("target authority must preserve the sealed final holdout")
    if not all(authority.get(key) is True for key in ("source_rights_reviewed", "target_semantics_reviewed", "temporal_leakage_reviewed", "fixed_boundaries_reviewed", "independent_from_generator")) or authority.get("generator_authored") is not False:
        raise RealSpineError("target authority independent-review conditions are incomplete")
    return dict(authority)


def _resolve_koi_mari_authority_path(
    locator: str,
    *,
    expected_raw_sha256: str,
) -> Path:
    """Resolve the archived July receipt for its frozen legacy locator."""

    if (
        locator == LEGACY_KOI_MARI_AUTHORITY_LOCATOR
        and expected_raw_sha256 == KOI_MARI_AUTHORITY_RAW_SHA256
    ):
        locator = KOI_MARI_AUTHORITY_LOCATOR
    return _safe_receipt_file(REPO_ROOT, locator)


def _validate_record(
    raw: Mapping[str, Any], *, source_receipts: Mapping[str, Mapping[str, Any]],
    target_receipts: Mapping[str, Mapping[str, Any]], roster_registry: RosterRegistry,
    canonical_ids: set[str],
) -> tuple[dict[str, Any], int, int, int]:
    expected = {
        "map_id", "canonical_series_id", "source_series_id", "source_game_id", "source_game_url",
        "source_game_number", "league_id", "tournament_id", "season_id", "event_start", "event_end",
        "pre_event_as_of", "partition", "team_ids", "player_ids", "roster_ids", "side_mapping",
        "source_receipt_ids", "target_authority_receipt_id", "prior_map_receipts",
    }
    _require_exact_keys(raw, expected, "record")
    record = dict(raw)
    for field in ("map_id", "canonical_series_id", "source_series_id", "source_game_id", "source_game_url", "league_id", "tournament_id", "season_id"):
        _require_text(record[field], field)
    normalized_series_id = re.sub(r"[^a-z0-9]+", "_", record["canonical_series_id"].casefold())
    if "dependence_cluster" in normalized_series_id or "proxy" in normalized_series_id:
        raise RealSpineError("canonical_series_id must not be a dependence cluster or proxy")
    source_game_id = record["source_game_id"]
    family = record["source_series_id"]
    if family not in source_game_id or family not in record["source_game_url"]:
        raise RealSpineError("source game URL/id must bind the source-stable series family")
    suffix = f"_game_{record['source_game_number']}"
    if not isinstance(record["source_game_number"], int) or isinstance(record["source_game_number"], bool) or record["source_game_number"] < 1 or not source_game_id.endswith(suffix):
        raise RealSpineError("source_game_number must equal the source game-id ordinal")
    if record["partition"] not in ALLOWED_PARTITIONS:
        raise RealSpineError("record partition must be a development partition; sealed final rows are forbidden")
    event_start = _require_time(record["event_start"], "event_start")
    event_end = _require_time(record["event_end"], "event_end")
    as_of = _require_time(record["pre_event_as_of"], "pre_event_as_of")
    if not event_start <= event_end or as_of >= event_start:
        raise RealSpineError("pre_event_as_of must be before event_start and event_end must not precede start")
    teams = record["team_ids"]
    players = record["player_ids"]
    roster_ids = record["roster_ids"]
    if not isinstance(teams, list) or len(teams) != 2 or len(set(teams)) != 2:
        raise RealSpineError("record requires exactly two distinct team_ids")
    if not isinstance(players, list) or len(players) != 10 or len(set(players)) != 10:
        raise RealSpineError("record requires exactly ten distinct player_ids")
    if not isinstance(roster_ids, Mapping) or set(roster_ids) != set(teams):
        raise RealSpineError("roster_ids must bind exactly both teams")
    unknown = (set(teams) | set(players)) - canonical_ids
    if unknown:
        raise RealSpineError(f"record references unknown canonical identities: {sorted(unknown)}")
    for team in teams:
        resolution = roster_registry.resolve_exact_roster(team, as_of=as_of, fail_closed=False)
        if not resolution.is_ok() or resolution.roster_id != roster_ids[team]:
            raise RealSpineError(f"exact active roster unavailable for team={team}")
        if resolution.player_ids is None or not set(resolution.player_ids) <= set(players):
            raise RealSpineError(f"resolved roster does not match record player ids for team={team}")
    mapping = record["side_mapping"]
    if not isinstance(mapping, Mapping) or set(mapping) != {"A_game_side", "B_game_side", "A_draft_order", "B_draft_order"}:
        raise RealSpineError("side_mapping must separately bind canonical sides, game side, and draft order")
    if {mapping["A_game_side"], mapping["B_game_side"]} != {"blue", "red"} or {mapping["A_draft_order"], mapping["B_draft_order"]} != {"first", "second"}:
        raise RealSpineError("side_mapping must be a complete bijection")
    receipt_ids = record["source_receipt_ids"]
    if not isinstance(receipt_ids, list) or not receipt_ids or len(set(receipt_ids)) != len(receipt_ids):
        raise RealSpineError("source_receipt_ids must be non-empty and unique")
    for receipt_id in receipt_ids:
        if receipt_id not in source_receipts:
            raise RealSpineError("record references unknown source receipt")
    target = target_receipts.get(_require_text(record["target_authority_receipt_id"], "target_authority_receipt_id"))
    if target is None:
        raise RealSpineError("record references unknown target authority receipt")
    if _require_time(target["target_available_at"], "target_available_at") < event_end:
        raise RealSpineError("target cannot be available before the map ends")
    prior = record["prior_map_receipts"]
    if not isinstance(prior, list):
        raise RealSpineError("prior_map_receipts must be a list")
    eligible = 0
    same_series_excluded = 0
    eligible_prior: list[dict[str, Any]] = []
    for prior_index, item in enumerate(prior):
        if not isinstance(item, Mapping):
            raise RealSpineError(f"prior_map_receipts[{prior_index}] must be an object")
        _require_exact_keys(item, {"origin_map_id", "origin_event_end", "origin_source_series_id", "source_receipt_id", "value_sha256"}, f"prior_map_receipts[{prior_index}]")
        origin_end = _require_time(item["origin_event_end"], "origin_event_end")
        origin_series = _require_text(item["origin_source_series_id"], "origin_source_series_id")
        if origin_series == record["source_series_id"]:
            same_series_excluded += 1
            continue
        usable_at = origin_end + timedelta(hours=EMBARGO_HOURS)
        if usable_at < event_start:
            if item["source_receipt_id"] not in source_receipts:
                raise RealSpineError("eligible prior map references unknown source receipt")
            _require_sha256(item["value_sha256"], "eligible prior value_sha256")
            eligible += 1
            eligible_prior.append(dict(item))
    # Ineligible receipts quantify conservative exclusions but are deliberately
    # outside the prediction binding: mutating future-only values cannot change
    # an earlier canonical row or its digest.
    record["prior_map_receipts"] = sorted(
        eligible_prior,
        key=lambda item: (item["origin_event_end"], item["origin_source_series_id"], item["origin_map_id"]),
    )
    return record, len(prior), eligible, same_series_excluded


def build_real_v1_packet(payload: Mapping[str, Any], *, evidence_root: Path | None = None) -> dict[str, Any]:
    """Return a canonical, no-label readiness packet for the executable G1 subset.

    `evidence_root` is optional so a caller can validate a mounted private
    receipt store.  Target labels are never loaded from it.
    """

    expected = {
        "schema_version", "snapshot_id", "source_receipts", "identity_rows", "roster_rows",
        "target_authority_receipts", "records", "split_assignments", "final_holdout", "availability_policy",
    }
    _require_exact_keys(payload, expected, "real-v1 input")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise RealSpineError("unexpected real-v1 input schema_version")
    snapshot_id = _require_text(payload["snapshot_id"], "snapshot_id")
    policy = payload["availability_policy"]
    if not isinstance(policy, Mapping) or dict(policy) != {"kind": "RETROSPECTIVE_FIXED_EMBARGO", "embargo_hours": EMBARGO_HOURS, "development_only": True}:
        raise RealSpineError("availability_policy must freeze the 48-hour development-only embargo")
    final_holdout = payload["final_holdout"]
    if not isinstance(final_holdout, Mapping) or set(final_holdout) != {"status", "cutoff", "receipt_sha256"}:
        raise RealSpineError("final_holdout must contain only status, cutoff, and receipt_sha256")
    if final_holdout["status"] != SEALED_FINAL_STATUS:
        raise RealSpineError("final holdout must remain SEALED_UNREAD")
    _require_time(final_holdout["cutoff"], "final_holdout cutoff")
    _require_sha256(final_holdout["receipt_sha256"], "final_holdout receipt_sha256")

    source_receipts = _validate_source_receipts(payload["source_receipts"], evidence_root=evidence_root)
    identity_registry = _identity_registry(payload["identity_rows"], source_receipts)
    canonical_ids = {row.canonical_id for row in identity_registry.rows}
    roster_registry = _roster_registry(payload["roster_rows"], source_receipts)
    target_receipts = _validate_target_receipts(payload["target_authority_receipts"], source_receipts)
    records_raw = payload["records"]
    if not isinstance(records_raw, list) or not records_raw:
        raise RealSpineError("records must be a non-empty list")
    records: list[dict[str, Any]] = []
    raw_prior_count = 0
    eligible_prior_count = 0
    same_series_prior_count = 0
    records_with_eligible_prior = 0
    seen_map_ids: set[str] = set()
    by_source_series: dict[str, str] = {}
    for raw in records_raw:
        if not isinstance(raw, Mapping):
            raise RealSpineError("record must be an object")
        record, prior_count, eligible_count, same_series_excluded = _validate_record(raw, source_receipts=source_receipts, target_receipts=target_receipts, roster_registry=roster_registry, canonical_ids=canonical_ids)
        if record["map_id"] in seen_map_ids:
            raise RealSpineError(f"duplicate map_id: {record['map_id']}")
        seen_map_ids.add(record["map_id"])
        old = by_source_series.setdefault(record["source_series_id"], record["canonical_series_id"])
        if old != record["canonical_series_id"]:
            raise RealSpineError("one source-stable family cannot map to multiple canonical series")
        raw_prior_count += prior_count
        eligible_prior_count += eligible_count
        same_series_prior_count += same_series_excluded
        records_with_eligible_prior += int(eligible_count > 0)
        records.append(record)

    assignments = payload["split_assignments"]
    if not isinstance(assignments, list) or not assignments:
        raise RealSpineError("split_assignments must be a non-empty list")
    expected_assign = {"source_series_id", "partition"}
    split_by_series: dict[str, str] = {}
    for index, raw in enumerate(assignments):
        if not isinstance(raw, Mapping):
            raise RealSpineError(f"split_assignments[{index}] must be an object")
        _require_exact_keys(raw, expected_assign, f"split_assignments[{index}]")
        family = _require_text(raw["source_series_id"], "split source_series_id")
        partition = raw["partition"]
        if partition not in ALLOWED_PARTITIONS:
            raise RealSpineError("split assignment cannot name a final/holdout partition")
        previous = split_by_series.setdefault(family, partition)
        if previous != partition:
            raise RealSpineError("source-stable series family crosses split partitions")
    record_families = {record["source_series_id"] for record in records}
    if set(split_by_series) != record_families:
        raise RealSpineError("split_assignments must cover exactly the observed source-series families")
    for record in records:
        if split_by_series[record["source_series_id"]] != record["partition"]:
            raise RealSpineError("record partition disagrees with frozen source-series assignment")

    records.sort(key=lambda row: (row["event_start"], row["source_series_id"], row["source_game_number"], row["map_id"]))
    coverage = {
        "map_count": len(records),
        "source_series_family_count": len(record_families),
        "partition_counts": dict(sorted(Counter(record["partition"] for record in records).items())),
        "calendar_year_counts": dict(sorted(Counter(parse_rfc3339(record["event_start"]).year for record in records).items())),
        "league_counts": dict(sorted(Counter(record["league_id"] for record in records).items())),
        "prior_map_receipt_count": raw_prior_count,
        "prior_map_eligible_after_embargo_count": eligible_prior_count,
        "prior_map_excluded_by_embargo_count": raw_prior_count - eligible_prior_count,
        "prior_map_excluded_same_source_series_count": same_series_prior_count,
        "maps_with_at_least_one_eligible_prior": records_with_eligible_prior,
    }
    input_binding = {
        "source_receipt_sha256": sha256_bytes(canonical_json_bytes(list(source_receipts.values()))),
        "target_authority_receipt_sha256": sha256_bytes(canonical_json_bytes(list(target_receipts.values()))),
        "identity_rows_sha256": sha256_bytes(canonical_json_bytes(payload["identity_rows"])),
        "roster_rows_sha256": sha256_bytes(canonical_json_bytes(payload["roster_rows"])),
        "records_sha256": sha256_bytes(canonical_json_bytes(records)),
        "split_assignments_sha256": sha256_bytes(canonical_json_bytes(sorted(split_by_series.items()))),
    }
    packet = {
        "schema_version": PACKET_KIND,
        "snapshot_id": snapshot_id,
        "availability_policy": dict(policy),
        "final_holdout": {"status": SEALED_FINAL_STATUS, "cutoff": final_holdout["cutoff"], "receipt_sha256": final_holdout["receipt_sha256"], "accessed": False},
        "input_binding": input_binding,
        "coverage": coverage,
        "claim_scope": {
            "state": "PRIVATE_DEVELOPMENT_INPUT_ONLY",
            "blocked_claims": ["performance", "scoring", "publication", "promotion", "final_holdout_result"],
            "reason": "48-hour embargo is conservative retrospective availability, not historical ingest authority",
        },
        "typed_blockers": [],
    }
    packet["packet_sha256"] = sha256_bytes(canonical_json_bytes(packet))
    return packet


def validate_real_v1_packet(payload: Mapping[str, Any], *, evidence_root: Path | None = None) -> dict[str, Any]:
    """Alias used by later layers: validation is construction and fails closed."""

    return build_real_v1_packet(payload, evidence_root=evidence_root)


def canonical_packet_bytes(packet: Mapping[str, Any]) -> bytes:
    """Return the exact newline-terminated bytes used for packet persistence."""

    expected = _require_sha256(packet.get("packet_sha256"), "packet_sha256")
    body = dict(packet)
    body.pop("packet_sha256")
    actual = sha256_bytes(canonical_json_bytes(body))
    if actual != expected:
        raise RealSpineError("packet_sha256 does not match canonical packet payload")
    return canonical_json_bytes(packet) + b"\n"


def write_real_v1_packet(packet: Mapping[str, Any], path: Path) -> str:
    """Write only a validated no-label packet and return its raw-byte digest."""

    data = canonical_packet_bytes(packet)
    _atomic_safe_write(path, data)
    return sha256_bytes(data)


def _raw_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_naive_iso(value: Any) -> str:
    """Serialize OE's naive local timestamp without falsely adding a timezone."""

    text = value.isoformat()
    if text.endswith("+00:00") or text.endswith("Z"):
        raise RealSpineError("warehouse adapter must not treat OE local timestamp as UTC")
    return text


def _repo_relative_locator(path: Path) -> str:
    """Return a stable repository-relative locator, never a machine path."""

    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError as error:
        raise RealSpineError(f"artifact must be inside the repository: {path}") from error


def _safe_repo_input_file(path: Path, name: str) -> Path:
    """Resolve a G2 input only if every repository path component is safe."""

    absolute = path.absolute()
    try:
        relative = absolute.relative_to(REPO_ROOT)
    except ValueError as error:
        raise RealSpineError(f"{name} must be inside the repository") from error
    return _safe_receipt_file(REPO_ROOT, _safe_relative_path(relative.as_posix(), name))


def _safe_unaliased_file(path: Path, name: str) -> Path:
    """Read a verification artifact without accepting symlink/hardlink aliases."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for index, component in enumerate(absolute.parts[1:]):
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError as error:
            raise RealSpineError(f"{name} is missing") from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RealSpineError(f"{name} must not contain a symlink")
        if index < len(absolute.parts) - 2 and not stat.S_ISDIR(metadata.st_mode):
            raise RealSpineError(f"{name} parent must be a directory")
    metadata = os.lstat(absolute)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise RealSpineError(f"{name} must be an unaliased regular file")
    return absolute


def _validate_target_evidence(
    evidence_path: Path,
    *,
    expected_payload_sha256: str,
    target_rows_path: Path,
) -> dict[str, Any]:
    """Bind the actual private target materialization to KOI's evidence hash.

    The evidence artifact is only a byte/provenance binding here.  Its older
    self-described claim ceiling is not promoted; the independent KOI envelope
    supplies the narrow private-fit authorization.
    """

    raw = evidence_path.read_bytes()
    try:
        evidence = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealSpineError("target evidence must be JSON") from error
    if not isinstance(evidence, Mapping):
        raise RealSpineError("target evidence must be an object")
    unsigned = dict(evidence)
    claimed = unsigned.pop("artifact_sha256", None)
    if claimed != expected_payload_sha256 or sha256_bytes(canonical_json_bytes(unsigned)) != expected_payload_sha256:
        raise RealSpineError("target evidence payload does not match KOI authority binding")
    private = evidence.get("private_materialization")
    if not isinstance(private, Mapping):
        raise RealSpineError("target evidence lacks private materialization binding")
    if private.get("locator") != _repo_relative_locator(target_rows_path):
        raise RealSpineError("target evidence binds a different target materialization")
    actual_raw_sha256 = _raw_file_sha256(target_rows_path)
    if private.get("raw_sha256") != actual_raw_sha256:
        raise RealSpineError("target materialization raw bytes differ from reviewed evidence")
    return dict(evidence)


def _canonical_target_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the exact selected target receipt surface in a fixed order."""

    canonical: list[dict[str, Any]] = []
    expected = {
        "game_id", "split", "oe_date_naive", "y_blue_win",
        "source_blue_result_id", "source_red_result_id", "dependence_cluster_id",
    }
    for raw in rows:
        if set(raw) != expected:
            raise RealSpineError("target projection has an unexpected schema")
        game_id = _require_text(raw["game_id"], "target game_id")
        split = _require_text(raw["split"], "target split").upper()
        if split not in ALLOWED_PARTITIONS:
            raise RealSpineError("sealed final or unknown target split was materialized")
        timestamp = _require_text(raw["oe_date_naive"], "target oe_date_naive")
        try:
            parsed = datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise RealSpineError("target oe_date_naive is invalid") from error
        if parsed.tzinfo is not None or _local_naive_iso(parsed) != timestamp:
            raise RealSpineError("target oe_date_naive must be canonical source-local naive time")
        outcome = raw["y_blue_win"]
        if isinstance(outcome, bool) or not isinstance(outcome, int) or outcome not in (0, 1):
            raise RealSpineError("target y_blue_win must be binary")
        canonical.append({
            "game_id": game_id,
            "split": split,
            "oe_date_naive": timestamp,
            "y_blue_win": outcome,
            "source_blue_result_id": _require_text(raw["source_blue_result_id"], "source_blue_result_id"),
            "source_red_result_id": _require_text(raw["source_red_result_id"], "source_red_result_id"),
            # This is retained as a diagnostic receipt only.  It is never a
            # source-series identifier or a partitioning proxy in this adapter.
            "dependence_cluster_id": _require_text(raw["dependence_cluster_id"], "dependence_cluster_id"),
        })
    canonical.sort(key=lambda row: row["game_id"])
    if len({row["game_id"] for row in canonical}) != len(canonical):
        raise RealSpineError("target membership must be one-to-one by game_id")
    return canonical


def _g0_prebinding_contracts() -> list[dict[str, str]]:
    """Bind the frozen G0 contract bytes without claiming their execution."""

    expected = (
        (G0_BENCHMARK_CONTRACT_PATH, G0_BENCHMARK_CONTRACT_RAW_SHA256, "benchmark_contract", "contract_id", "scryglass:real-benchmark:v1"),
        (G0_BASELINE_REGISTRY_PATH, G0_BASELINE_REGISTRY_RAW_SHA256, "baseline_registry", "registry_id", "scryglass:real-baselines:v1"),
    )
    bindings: list[dict[str, str]] = []
    for path, expected_sha256, kind, identity_key, expected_identity in expected:
        safe_path = _safe_repo_input_file(path, kind)
        raw = safe_path.read_bytes()
        if sha256_bytes(raw) != expected_sha256:
            raise RealSpineError(f"frozen G0 {kind} bytes drifted")
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RealSpineError(f"frozen G0 {kind} must be JSON") from error
        if not isinstance(payload, Mapping) or payload.get(identity_key) != expected_identity:
            raise RealSpineError(f"frozen G0 {kind} identity is invalid")
        bindings.append({
            "kind": kind,
            "locator": _repo_relative_locator(safe_path),
            "raw_sha256": expected_sha256,
            "artifact_kind": _require_text(payload.get("artifact_kind"), f"G0 {kind} artifact_kind"),
        })
    return bindings


def extract_lpl_private_development_snapshot(
    *,
    maps_path: Path,
    player_games_path: Path,
    output_rows_path: Path,
    output_manifest_path: Path,
    target_rows_path: Path = REPO_ROOT / "data/lol/warehouse/private_v2/draft-interactions/oe-target-rows.parquet",
    authority_path: Path = REPO_ROOT / KOI_MARI_AUTHORITY_LOCATOR,
    target_evidence_path: Path = REPO_ROOT / "data/lol/v2/models/draft-interactions/oe-private-target-evidence.json",
    cutoff_local_naive: str = "2026-06-01T00:00:00",
) -> dict[str, Any]:
    """Emit the LPL-only, non-final private G2 input surface.

    This is deliberately not an historical live-ingest claim: source-local
    timestamps are comparable only within LPL and the 48-hour delay is an
    accepted retrospective-fit boundary.  The independently reviewed KOI
    envelope permits *only* private ``model_fit`` and ``rank_selection``.
    """

    try:
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - environment boundary
        raise RealSpineError("pyarrow is required for the private warehouse adapter") from error

    # Both final destinations must be valid before any data read, materialized
    # result, or first-file write.  This makes an invalid manifest destination
    # fail before it can leave a new or changed rows artifact behind.
    output_rows_path = _safe_output_target(output_rows_path)
    output_manifest_path = _safe_output_target(output_manifest_path)
    if output_rows_path == output_manifest_path:
        raise RealSpineError("rows and manifest destinations must be distinct")

    maps_path = _safe_repo_input_file(maps_path, "maps_path")
    player_games_path = _safe_repo_input_file(player_games_path, "player_games_path")
    target_rows_path = _safe_repo_input_file(target_rows_path, "target_rows_path")
    authority_path = _safe_repo_input_file(authority_path, "authority_path")
    target_evidence_path = _safe_repo_input_file(target_evidence_path, "target_evidence_path")
    cutoff = datetime.fromisoformat(cutoff_local_naive)
    if cutoff.tzinfo is not None:
        raise RealSpineError("cutoff_local_naive must not assert a timezone")
    map_columns = (
        "oe_gameid", "game_uid", "url", "league", "date", "game", "year", "split", "playoffs", "patch",
        "datacompleteness", "blue_firstPick",
    )
    player_columns = (
        "gameid", "league", "date", "game", "position", "playerid", "teamid", "teamname", "side",
        "datacompleteness",
    )
    source_filter = [
        ("league", "=", "LPL"),
        ("date", ">=", datetime(2025, 1, 1)),
        ("date", "<", cutoff),
    ]
    target_filter = [
        ("canonical_league", "=", "LPL"),
        ("split", "in", ["train", "development", "validation"]),
    ]
    # Predicate pushdown is required: sealed post-cutoff rows must not be
    # materialized merely to be discarded by a later dataframe filter.
    maps = pq.read_table(maps_path, columns=list(map_columns), filters=source_filter).to_pandas()
    players = pq.read_table(player_games_path, columns=list(player_columns), filters=source_filter).to_pandas()
    target_columns = (
        "game_id", "split", "oe_date_naive", "y_blue_win", "source_blue_result_id",
        "source_red_result_id", "dependence_cluster_id",
    )
    # The final temporal holdout is excluded by the dataset predicate, before
    # target fields are selected or materialized.
    selected_targets = pq.read_table(
        target_rows_path, columns=list(target_columns), filters=target_filter,
    ).to_pylist()
    authority_raw_sha256 = _raw_file_sha256(authority_path)
    authority = validate_koi_mari_authority(
        authority_path, expected_raw_sha256=KOI_MARI_AUTHORITY_RAW_SHA256,
    )
    evidence_payload_sha256 = _require_sha256(authority.get("evidence_payload_sha256"), "authority evidence_payload_sha256")
    split_payload_sha256 = _require_sha256(authority.get("split_payload_sha256"), "authority split_payload_sha256")
    if evidence_payload_sha256 != KOI_MARI_EVIDENCE_PAYLOAD_SHA256 or split_payload_sha256 != KOI_MARI_SPLIT_PAYLOAD_SHA256:
        raise RealSpineError("KOI authority does not bind the approved evidence and split payloads")
    evidence = _validate_target_evidence(
        target_evidence_path,
        expected_payload_sha256=evidence_payload_sha256,
        target_rows_path=target_rows_path,
    )
    targets = _canonical_target_rows(selected_targets)
    target_by_game = {row["game_id"]: row for row in targets}
    # The comparison occurs on the supplied source-local values.  No Z suffix,
    # availability claim, or final outcome field is created here.
    maps = maps[(maps["league"] == "LPL") & (maps["date"] >= datetime(2025, 1, 1)) & (maps["date"] < cutoff)].copy()
    players = players[(players["league"] == "LPL") & (players["date"] >= datetime(2025, 1, 1)) & (players["date"] < cutoff)].copy()
    if maps.empty or players.empty or not targets:
        raise RealSpineError("no LPL rows survive the frozen development cutoff")
    allowed_completeness = {"partial", "complete"}
    if maps["datacompleteness"].isna().any() or players["datacompleteness"].isna().any() or not set(maps["datacompleteness"]).issubset(allowed_completeness) or not set(players["datacompleteness"]).issubset(allowed_completeness):
        raise RealSpineError("source datacompleteness must be an explicit supported value")
    if maps["oe_gameid"].duplicated().any() or not (maps["oe_gameid"] == maps["game_uid"]).all():
        raise RealSpineError("maps must have one OE game id and matching source game uid")

    map_ids = {str(value) for value in maps["oe_gameid"]}
    if map_ids != set(target_by_game):
        raise RealSpineError("canonical LPL map membership must exactly equal non-final target membership")
    by_game = {game_id: group for game_id, group in players.groupby("gameid", sort=False)}
    rows: list[dict[str, Any]] = []
    seen_family_ordinal: set[tuple[str, int]] = set()
    for map_row in maps.sort_values(["date", "oe_gameid"]).to_dict("records"):
        game_id = _require_text(map_row["oe_gameid"], "oe_gameid")
        target = target_by_game.get(game_id)
        if target is None:
            raise RealSpineError("map has no canonical private target")
        match = LPL_FAMILY_RE.fullmatch(game_id)
        if match is None:
            raise RealSpineError(f"LPL game id is not source-stable bmid grammar: {game_id}")
        url_match = LPL_URL_BMID_RE.search(_require_text(map_row["url"], "url"))
        if url_match is None or url_match.group("bmid") != match.group("bmid"):
            raise RealSpineError(f"LPL URL bmid does not bind game family: {game_id}")
        ordinal = int(match.group("ordinal"))
        if map_row["game"] != ordinal:
            raise RealSpineError(f"OE game ordinal conflicts with source game id: {game_id}")
        family_key = match.group("bmid")
        if (family_key, ordinal) in seen_family_ordinal:
            raise RealSpineError(f"duplicate bmid/ordinal source game: {game_id}")
        seen_family_ordinal.add((family_key, ordinal))
        game_players = by_game.get(game_id)
        if game_players is None or len(game_players) != 10:
            raise RealSpineError(f"observed participants must have exactly ten player rows: {game_id}")
        if game_players["playerid"].isna().any() or game_players["teamid"].isna().any():
            raise RealSpineError(f"observed participants require non-null player/team ids: {game_id}")
        if set(game_players["datacompleteness"]) - allowed_completeness:
            raise RealSpineError(f"observed participant datacompleteness is unsupported: {game_id}")
        if game_players["playerid"].nunique() != 10 or game_players["teamid"].nunique() != 2:
            raise RealSpineError(f"observed participants must have 10 players and 2 teams: {game_id}")
        normalized_role = game_players["position"].replace({"jng": "jungle", "bot": "bot", "sup": "support"})
        if set(normalized_role) != set(ROLES) or any((normalized_role == role).sum() != 2 for role in ROLES):
            raise RealSpineError(f"observed participants must have two complete five-role lineups: {game_id}")
        if set(game_players["side"]) != {"Blue", "Red"} or any((game_players["side"] == side).sum() != 5 for side in ("Blue", "Red")):
            raise RealSpineError(f"observed participants require five Blue and five Red rows: {game_id}")
        lineups = []
        for side in ("Blue", "Red"):
            side_rows = game_players[game_players["side"] == side].copy()
            side_rows["normalized_role"] = side_rows["position"].replace({"jng": "jungle", "sup": "support"})
            if set(side_rows["normalized_role"]) != set(ROLES) or side_rows["normalized_role"].duplicated().any():
                raise RealSpineError(f"observed side lineup must cover each role exactly once: {game_id}")
            role_map = dict(zip(side_rows["normalized_role"], side_rows["playerid"]))
            lineups.append({
                "observed_game_side": side.lower(),
                "team_id": str(side_rows["teamid"].iloc[0]),
                "player_ids_by_role": {role: str(role_map[role]) for role in ROLES},
            })
        local_start = map_row["date"].to_pydatetime()
        if local_start.tzinfo is not None:
            raise RealSpineError("OE date must be preserved as source-local naive time")
        if target["oe_date_naive"] != _local_naive_iso(local_start):
            raise RealSpineError(f"target source-local timestamp disagrees with map source: {game_id}")
        raw_patch = map_row.get("patch")
        patch = _require_text(str(raw_patch), "map patch") if raw_patch is not None else "UNAVAILABLE"
        family_id = f"oe:lpl:bmid:{family_key}"
        if map_row["blue_firstPick"] in {0, 1, 0.0, 1.0}:
            draft_order = {"blue": "first" if int(map_row["blue_firstPick"]) == 1 else "second", "red": "second" if int(map_row["blue_firstPick"]) == 1 else "first"}
        else:
            draft_order = None
        rows.append({
            "canonical_series_id": f"scryglass:series:{family_id}",
            "source_series_id": family_id,
            "source_game_id": game_id,
            "source_game_url": str(map_row["url"]),
            "source_game_number": ordinal,
            "league_id": "LPL",
            "season_id": f"LPL:{int(map_row['year'])}:{map_row['split']}:{int(map_row['playoffs'])}",
            "calendar_year": local_start.year,
            "patch": patch,
            "technical_tournament_group": f"LPL:{int(map_row['year'])}:{map_row['split']}:{int(map_row['playoffs'])}",
            "source_local_event_start": _local_naive_iso(local_start),
            "retrospective_embargo_after_local_naive": _local_naive_iso(local_start + timedelta(hours=EMBARGO_HOURS)),
            "partition": target["split"],
            "participant_lineup_kind": "OBSERVED_MAP_PARTICIPANTS_NOT_PRE_EVENT_ROSTER_AUTHORITY",
            "source_datacompleteness": str(map_row["datacompleteness"]),
            "observed_lineups": lineups,
            "game_side": {"blue_team_id": lineups[0]["team_id"], "red_team_id": lineups[1]["team_id"]},
            "draft_order": draft_order,
            "target": {
                "y_blue_win": target["y_blue_win"],
                "source_blue_result_id": target["source_blue_result_id"],
                "source_red_result_id": target["source_red_result_id"],
                "authority": "KOI_MARI_PRIVATE_RETROSPECTIVE_MODEL_FIT_AND_RANK_SELECTION",
            },
            "dependence_cluster_diagnostic": target["dependence_cluster_id"],
        })
    rows.sort(key=lambda row: (row["source_local_event_start"], row["source_game_id"]))
    family_partitions: dict[str, str] = {}
    for row in rows:
        previous = family_partitions.setdefault(row["source_series_id"], row["partition"])
        if previous != row["partition"]:
            raise RealSpineError("source-stable bmid family crosses the fixed target split")
    for index, row in enumerate(rows):
        event_start = datetime.fromisoformat(row["source_local_event_start"])
        eligible_origins = [
            origin["source_game_id"]
            for origin in rows[:index]
            if origin["source_series_id"] != row["source_series_id"]
            and datetime.fromisoformat(origin["source_local_event_start"]) + timedelta(hours=EMBARGO_HOURS) < event_start
        ]
        row["eligible_prior_origin_map_ids"] = eligible_origins
        row["eligible_prior_origin_count"] = len(eligible_origins)
    row_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    rows_sha256 = sha256_bytes(row_bytes)
    coverage = {
        "map_count": len(rows),
        "source_series_family_count": len({row["source_series_id"] for row in rows}),
        "partition_counts": dict(sorted(Counter(row["partition"] for row in rows).items())),
        "calendar_year_counts": dict(sorted(Counter(row["source_local_event_start"][:4] for row in rows).items())),
        "season_id_counts": dict(sorted(Counter(row["season_id"] for row in rows).items())),
        "patch_counts": dict(sorted(Counter(row["patch"] for row in rows).items())),
        "source_datacompleteness_counts": {
            "maps": dict(sorted(Counter(str(value) for value in maps["datacompleteness"]).items())),
            "players": dict(sorted(Counter(str(value) for value in players["datacompleteness"]).items())),
        },
        "draft_order_observed_count": sum(row["draft_order"] is not None for row in rows),
        "target_partition_counts": dict(sorted(Counter(row["partition"] for row in rows).items())),
        "families_with_atomic_fixed_partition": len(family_partitions),
        "eligible_prior_origin_count": sum(row["eligible_prior_origin_count"] for row in rows),
    }
    if coverage["partition_counts"] != EXPECTED_LPL_PRIVATE_PARTITION_COUNTS or coverage["source_series_family_count"] != EXPECTED_LPL_PRIVATE_SOURCE_FAMILIES:
        raise RealSpineError("frozen LPL private target membership or split coverage drifted")
    canonical_target_bytes = canonical_json_bytes(targets)
    manifest = {
        "schema_version": "scryglass:real-v1-lpl-private-g2-input:v1",
        "source_scope": "LPL_2025_2026_PRE_2026_06_01",
        "source_files": [
            {"locator": _repo_relative_locator(maps_path), "sha256": _raw_file_sha256(maps_path), "columns_read": list(map_columns), "rights_status": PRIVATE_RIGHTS_STATUS},
            {"locator": _repo_relative_locator(player_games_path), "sha256": _raw_file_sha256(player_games_path), "columns_read": list(player_columns), "rights_status": PRIVATE_RIGHTS_STATUS},
            {"locator": _repo_relative_locator(target_rows_path), "sha256": _raw_file_sha256(target_rows_path), "columns_read": list(target_columns), "predicate": "canonical_league=LPL AND split IN {train,development,validation}", "rights_status": PRIVATE_RIGHTS_STATUS},
        ],
        "rows_locator": _repo_relative_locator(output_rows_path),
        "rows_sha256": rows_sha256,
        "canonical_selected_target_rows_sha256": sha256_bytes(canonical_target_bytes),
        "canonical_selected_target_row_count": len(targets),
        "target_authority": {
            "authority_locator": _repo_relative_locator(authority_path),
            "authority_raw_sha256": authority_raw_sha256,
            "evidence_locator": _repo_relative_locator(target_evidence_path),
            "evidence_payload_sha256": evidence_payload_sha256,
            "split_payload_sha256": split_payload_sha256,
            "reviewed_target_materialization_raw_sha256": evidence["private_materialization"]["raw_sha256"],
        },
        "g0_prebinding_contracts": _g0_prebinding_contracts(),
        "distribution_policy": {
            "private_artifact": True,
            "public_pack_eligible": False,
            "vercel_deploy_eligible": False,
            "reason": "private retrospective target rows are authorized only for model_fit and rank_selection",
        },
        "coverage": coverage,
        "availability_policy": {
            "kind": "PRIVATE_RETROSPECTIVE_FIXED_EMBARGO",
            "embargo_hours": EMBARGO_HOURS,
            "scope": "LPL_SOURCE_LOCAL_TIME_ONLY",
            "authorizes": ["model_fit", "rank_selection"],
            "does_not_authorize": ["historical_live_ingest", "forecast", "prediction", "production", "publication", "sota"],
        },
        "final_holdout": {"status": SEALED_FINAL_STATUS, "cutoff_local_naive": cutoff_local_naive, "accessed": False},
        "claim_scope": {
            "state": "PRIVATE_RETROSPECTIVE_MODEL_FIT_AND_RANK_SELECTION_AVAILABLE",
            "available_claims": ["private_model_fit", "private_rank_selection"],
            "blocked_claims": ["pre_event_roster_authority", "historical_ingest_availability", "forecast", "prediction", "production", "publication", "promotion", "sota", "final_holdout_result"],
        },
        "typed_blockers": [
            {"code": "OE_SOURCE_LOCAL_TIME_NOT_HISTORICAL_INGEST_AUTHORITY", "scope": "all_rows", "claim_effect": "LIVE_AND_FORECAST_CLAIMS_BLOCKED"},
            {"code": "OBSERVED_MAP_PARTICIPANTS_NOT_PRE_EVENT_ROSTER_AUTHORITY", "scope": "all_rows", "claim_effect": "PRE_EVENT_ROSTER_CLAIMS_BLOCKED"},
            {"code": "SOURCE_OBSERVED_AT_NOT_BOUND", "scope": "all_rows", "claim_effect": "FORECAST_SIMULATION_BLOCKED"},
            {"code": "G1_018_BASELINES_TYPED_UNAVAILABLE", "scope": "real-v1", "claim_effect": "REQUIRED_COMPARISON_BLOCKED_UNTIL_RESOLVED"},
        ],
        "g1_018_baseline_binding": {
            "status": "TYPED_UNAVAILABLE",
            "reason_code": "REQUIRED_BASELINES_NOT_EXECUTED_ON_FROZEN_REAL_ROWS",
            "claim_effect": "REQUIRED_COMPARISON_BLOCKED_UNTIL_RESOLVED",
        },
    }
    manifest["manifest_sha256"] = sha256_bytes(canonical_json_bytes(manifest))
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    # The manifest is fully assembled and hashed before either final artifact
    # changes.  The transaction stages both files, then commits or rolls back.
    _atomic_safe_write_many(((output_rows_path, row_bytes), (output_manifest_path, manifest_bytes)))
    return manifest


def verify_lpl_private_development_snapshot(
    *, rows_path: Path, manifest_path: Path, expected_manifest_sha256: str,
) -> dict[str, Any]:
    """Verify persisted G2 input bytes against an independently pinned digest.

    A manifest's self-hash detects accidental drift only.  The caller-provided
    digest is mandatory so changing rows plus both self-hashes cannot forge a
    verified snapshot.
    """

    rows_path = _safe_unaliased_file(rows_path, "rows_path")
    manifest_path = _safe_unaliased_file(manifest_path, "manifest_path")
    raw_manifest = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RealSpineError("private G2 manifest must be JSON") from error
    if not isinstance(manifest, Mapping):
        raise RealSpineError("private G2 manifest must be an object")
    unsigned = dict(manifest)
    claimed = unsigned.pop("manifest_sha256", None)
    if claimed != sha256_bytes(canonical_json_bytes(unsigned)):
        raise RealSpineError("private G2 manifest hash drift")
    if claimed != _require_sha256(expected_manifest_sha256, "expected_manifest_sha256"):
        raise RealSpineError("private G2 manifest does not match the independently pinned digest")
    if manifest.get("rows_sha256") != sha256_bytes(rows_path.read_bytes()):
        raise RealSpineError("private G2 row bytes drift")
    if manifest.get("final_holdout", {}).get("accessed") is not False:
        raise RealSpineError("sealed final holdout access must remain false")
    if manifest.get("claim_scope", {}).get("state") != "PRIVATE_RETROSPECTIVE_MODEL_FIT_AND_RANK_SELECTION_AVAILABLE":
        raise RealSpineError("private G2 claim scope is not the approved narrow scope")
    if manifest.get("distribution_policy", {}).get("public_pack_eligible") is not False or manifest.get("distribution_policy", {}).get("vercel_deploy_eligible") is not False:
        raise RealSpineError("private G2 snapshot must be excluded from public packs and deployment")
    return dict(manifest)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the real-v1 pre-event readiness packet")
    commands = parser.add_subparsers(dest="command", required=True)
    packet_parser = commands.add_parser("packet", help="build the generic no-label readiness packet")
    packet_parser.add_argument("input", type=Path, help="private input receipt JSON")
    packet_parser.add_argument("output", type=Path, help="public-safe readiness packet JSON")
    packet_parser.add_argument("--evidence-root", type=Path, default=None, help="optional controlled private receipt root")
    lpl_build = commands.add_parser("lpl-build", help="build the private non-final LPL G2 input snapshot")
    lpl_build.add_argument("--maps", type=Path, default=REPO_ROOT / "data/lol/warehouse/parquet/maps.parquet")
    lpl_build.add_argument("--player-games", type=Path, default=REPO_ROOT / "data/lol/warehouse/parquet/oe_player_games.parquet")
    lpl_build.add_argument("--target-rows", type=Path, default=REPO_ROOT / "data/lol/warehouse/private_v2/draft-interactions/oe-target-rows.parquet")
    lpl_build.add_argument(
        "--authority",
        type=Path,
        default=REPO_ROOT / KOI_MARI_AUTHORITY_LOCATOR,
    )
    lpl_build.add_argument("--target-evidence", type=Path, default=REPO_ROOT / "data/lol/v2/models/draft-interactions/oe-private-target-evidence.json")
    lpl_build.add_argument("--rows-output", type=Path, required=True)
    lpl_build.add_argument("--manifest-output", type=Path, required=True)
    lpl_build.add_argument("--cutoff-local-naive", default="2026-06-01T00:00:00")
    lpl_verify = commands.add_parser("lpl-verify", help="verify private LPL snapshot against an externally pinned manifest digest")
    lpl_verify.add_argument("--rows", type=Path, required=True)
    lpl_verify.add_argument("--manifest", type=Path, required=True)
    lpl_verify.add_argument("--expected-manifest-sha256", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "packet":
            payload = json.loads(args.input.read_bytes().decode("utf-8"))
            if not isinstance(payload, Mapping):
                raise RealSpineError("input JSON must be an object")
            packet = build_real_v1_packet(payload, evidence_root=args.evidence_root)
            digest = write_real_v1_packet(packet, args.output)
            result = {"packet_sha256": packet["packet_sha256"], "file_sha256": digest}
        elif args.command == "lpl-build":
            manifest = extract_lpl_private_development_snapshot(
                maps_path=args.maps,
                player_games_path=args.player_games,
                target_rows_path=args.target_rows,
                authority_path=args.authority,
                target_evidence_path=args.target_evidence,
                output_rows_path=args.rows_output,
                output_manifest_path=args.manifest_output,
                cutoff_local_naive=args.cutoff_local_naive,
            )
            result = {"manifest_sha256": manifest["manifest_sha256"], "rows_sha256": manifest["rows_sha256"], "claim_scope": manifest["claim_scope"]["state"]}
        elif args.command == "lpl-verify":
            manifest = verify_lpl_private_development_snapshot(
                rows_path=args.rows,
                manifest_path=args.manifest,
                expected_manifest_sha256=args.expected_manifest_sha256,
            )
            result = {"manifest_sha256": manifest["manifest_sha256"], "rows_sha256": manifest["rows_sha256"], "verified": True}
        else:
            raise RealSpineError(f"unsupported command: {args.command}")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RealSpineError) as error:
        parser.error(str(error))
        # argparse.error normally raises SystemExit(2). Keep the failure path
        # explicit so a test double or alternate parser cannot reach the result print.
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
