"""Research-only downstream impact checks for future player and team values.

The future-value model can change several existing descriptive artifacts.  This
module gives a later training run one bounded comparison surface.  It reads the
old and candidate JSON artifacts, checks their source binding, measures row and
numeric changes, and returns a report with public authority unavailable.

The evaluator does not edit a public pack.  It does not publish ratings.  It
does not turn an evaluation receipt into model authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import hashlib
import json
import math
import re

from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:future-value-downstream:v1"
CONTRACT_SCHEMA_VERSION = "scryglass:future-value-downstream-contract:v1"
EVALUATION_SCHEMA_VERSION = "scryglass:future-value-evaluation:v1"
SOURCE_RECEIPT_SCHEMA_VERSION = "scryglass:future-value-rating-source:v1"
SOURCE_RECEIPT_STATUS = "accepted_source_bound_development_only"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SOURCE_FIELDS = ("source_as_of", "source_game_count", "source_identity_sha256")
SOURCE_RECEIPT_FIELDS = (
    "schema_version",
    "status",
    "source_as_of",
    "source_game_count",
    "source_identity_sha256",
    "accepted_game_ids",
    "source_files",
    "authority",
    "receipt_sha256",
)
DEFAULT_EVALUATION_FILE = "future-value-evaluation-receipt.json"
DEFAULT_SOURCE_FILES = (
    "future-value-source-receipt.json",
    "source-receipt.json",
    "manifest.json",
)
REQUIRED_EVALUATION_GATES = (
    "fitted_metric_weights",
    "fold_internal_rank_3_atoms",
    "chronological_whole_series",
    "paired_current_ratings",
    "calibration_and_proper_scores",
    "regional_transfer",
    "patch_transfer",
    "roster_change_and_tournament_boundary",
    "missingness_censoring_sparse_support",
    "side_swap_invariance",
)


class FutureValueDownstreamError(ValueError):
    """The downstream comparison cannot satisfy its source or artifact contract."""


@dataclass(frozen=True)
class DownstreamArtifactSpec:
    """One consumer-facing artifact required by the impact review."""

    name: str
    path: str
    consumers: tuple[str, ...]
    identity_groups: tuple[tuple[str, ...], ...]
    value_fields: tuple[str, ...]
    glob: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "glob": self.glob,
            "consumers": list(self.consumers),
            "identity_groups": [list(group) for group in self.identity_groups],
            "value_fields": list(self.value_fields),
        }


_ARTIFACT_SPECS: tuple[DownstreamArtifactSpec, ...] = (
    DownstreamArtifactSpec(
        name="player_ratings",
        path="features/player_ratings_snapshot.json",
        consumers=("player_profiles", "player_leaderboards", "match_context"),
        identity_groups=(("player_id", "player", "id", "name"),),
        value_fields=(
            "mu_total",
            "mu_effective",
            "mu_regional",
            "mu_meta",
            "sigma",
            "rating_p10",
        ),
    ),
    DownstreamArtifactSpec(
        name="team_ratings",
        path="features/ratings_snapshot.json",
        consumers=("team_profiles", "team_leaderboards", "match_context"),
        identity_groups=(("team_key", "team_id", "team", "id", "name"),),
        value_fields=(
            "mu_total",
            "mu_effective",
            "mu_regional",
            "mu_meta",
            "sigma",
            "rating_p10",
        ),
    ),
    DownstreamArtifactSpec(
        name="tierlists",
        path="rankings/tierlists.json",
        consumers=("tier_list_explorer", "draft_context"),
        identity_groups=(
            ("champion_id", "champion", "name", "id"),
            ("role", "position"),
            ("league", "scope", "competition_tier"),
            ("patch",),
        ),
        value_fields=("rank", "score", "rating", "win_rate", "delta", "games"),
    ),
    DownstreamArtifactSpec(
        name="draft_score",
        path="features/draft_records.json",
        consumers=("draft_score", "match_explorer", "support_chat"),
        identity_groups=(("game_uid", "game_id", "match_id", "id"),),
        value_fields=(
            "draft_edge",
            "base",
            "synergy",
            "counter",
            "same_role",
            "score",
        ),
    ),
    DownstreamArtifactSpec(
        name="profiles",
        path="features/profile_records.json",
        consumers=("player_profiles", "team_profiles", "match_explorer"),
        identity_groups=(("game_uid", "game_id", "match_id", "player_id", "player", "team_key", "team", "id", "name"),),
        value_fields=(
            "mu_total",
            "mu_effective",
            "future_value",
            "team_value",
            "sigma",
            "games",
            "n_maps",
        ),
    ),
    DownstreamArtifactSpec(
        name="matches",
        path="features/match_records_*.json",
        consumers=("match_explorer", "match_profile_api", "support_chat"),
        identity_groups=(("game_uid", "game_id", "match_id", "id"),),
        value_fields=(
            "future_team_value",
            "future_player_value",
            "draft_edge",
            "blue_result",
            "red_result",
        ),
        glob=True,
    ),
    DownstreamArtifactSpec(
        name="public_manifest",
        path="manifest.json",
        consumers=("public_pack", "runtime_release", "web_manifest_gate"),
        identity_groups=(("pack_id", "release_id", "id"),),
        value_fields=(
            "source_game_count",
            "team_rating_rows",
            "player_rating_rows",
            "total_files",
            "total_bytes",
        ),
    ),
)


def required_artifact_specs() -> tuple[DownstreamArtifactSpec, ...]:
    """Return the immutable set of downstream artifacts in the review."""

    return _ARTIFACT_SPECS


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise FutureValueDownstreamError("value is not canonical JSON") from error


def _utc_timestamp(value: Any, field: str) -> datetime:
    """Parse one timezone-aware timestamp and return it in UTC."""

    if not isinstance(value, str) or not value.strip():
        raise FutureValueDownstreamError(f"{field} is missing or invalid")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise FutureValueDownstreamError(f"{field} is not a timestamp") from error
    if parsed.tzinfo is None:
        raise FutureValueDownstreamError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _canonical_utc_text(value: Any, field: str) -> str:
    stamp = _utc_timestamp(value, field)
    return stamp.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_summary(receipt: Mapping[str, Any]) -> dict[str, Any]:
    """Return the source identity fields that the downstream report exposes."""

    summary = {
        "schema_version": receipt["schema_version"],
        "status": receipt["status"],
        "source_as_of": receipt["source_as_of"],
        "source_game_count": receipt["source_game_count"],
        "source_identity_sha256": receipt["source_identity_sha256"],
        "accepted_game_ids": list(receipt["accepted_game_ids"]),
        "receipt_sha256": receipt["receipt_sha256"],
        "source_receipt_sha256": receipt["receipt_sha256"],
    }
    for field in (
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
    ):
        if field in receipt:
            value = receipt[field]
            summary[field] = list(value) if field.endswith("_ids") else value
    return summary


def _validate_source_file_records(
    source_files: Any,
    *,
    receipt_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate source file metadata and check files when a safe path is present."""

    if not isinstance(source_files, Mapping) or not source_files:
        raise FutureValueDownstreamError("verified source receipt has no source file records")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_label, raw_record in source_files.items():
        label = str(raw_label).strip()
        if not label or not isinstance(raw_record, Mapping):
            raise FutureValueDownstreamError("verified source file record is invalid")
        byte_count = raw_record.get("bytes")
        digest = raw_record.get("sha256")
        locator = raw_record.get("locator", raw_record.get("path"))
        if (
            isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or not isinstance(locator, str)
            or not locator.strip()
            or not isinstance(digest, str)
            or SHA256_RE.fullmatch(digest) is None
        ):
            raise FutureValueDownstreamError(
                f"verified source file record is invalid: {label}"
            )
        locator = locator.strip()
        locator_path = Path(locator)
        if ".." in locator_path.parts:
            raise FutureValueDownstreamError(
                f"verified source file locator escapes its root: {label}"
            )

        record = dict(raw_record)
        record["bytes"] = byte_count
        record["sha256"] = digest
        record["locator"] = locator
        # Source receipts from the accepted-source bridge use a locator because
        # the source files can live outside the artifact checkout.  A receipt
        # with an explicit path is checked when that path is available.
        if "path" in raw_record:
            path = Path(str(raw_record["path"]))
            if not path.is_absolute() and receipt_path is not None:
                path = receipt_path.parent / path
            if not path.is_file() or path.is_symlink():
                raise FutureValueDownstreamError(
                    f"verified source file is missing or unsafe: {label}"
                )
            if path.stat().st_size != byte_count or _sha256(path) != digest:
                raise FutureValueDownstreamError(
                    f"verified source file changed: {label}"
                )
        normalized[label] = record
    return normalized


def _validate_source_receipt(
    value: Mapping[str, Any],
    *,
    receipt_path: Path | None = None,
) -> dict[str, Any]:
    """Validate one complete, hash-bound future-value source receipt."""

    if not isinstance(value, Mapping):
        raise FutureValueDownstreamError("canonical verified source receipt is required")
    receipt = dict(value)
    if receipt.get("schema_version") != SOURCE_RECEIPT_SCHEMA_VERSION:
        raise FutureValueDownstreamError("verified source receipt schema is invalid")
    if receipt.get("status") != SOURCE_RECEIPT_STATUS:
        raise FutureValueDownstreamError("verified source receipt status is invalid")
    if any(field not in receipt for field in SOURCE_RECEIPT_FIELDS):
        raise FutureValueDownstreamError("verified source receipt is incomplete")

    source_as_of = receipt.get("source_as_of")
    if _canonical_utc_text(source_as_of, "source_as_of") != source_as_of:
        raise FutureValueDownstreamError("verified source receipt date is not canonical")

    raw_ids = receipt.get("accepted_game_ids")
    if not isinstance(raw_ids, list) or not raw_ids or not all(
        isinstance(value, str) and value.strip() for value in raw_ids
    ):
        raise FutureValueDownstreamError("verified source receipt accepted IDs are invalid")
    accepted_ids = canonical_game_ids(raw_ids)
    if list(accepted_ids) != raw_ids:
        raise FutureValueDownstreamError(
            "verified source receipt accepted IDs are not canonical and unique"
        )
    source_count = receipt.get("source_game_count")
    source_identity = receipt.get("source_identity_sha256")
    if (
        isinstance(source_count, bool)
        or not isinstance(source_count, int)
        or source_count != len(accepted_ids)
        or not isinstance(source_identity, str)
        or SHA256_RE.fullmatch(source_identity) is None
        or source_identity != identity_sha256(accepted_ids)
    ):
        raise FutureValueDownstreamError("verified source receipt census identity is invalid")

    optional_eligible = (
        "model_eligible_game_count",
        "model_eligible_identity_sha256",
        "model_eligible_game_ids",
    )
    if any(field in receipt for field in optional_eligible):
        if not all(field in receipt for field in optional_eligible):
            raise FutureValueDownstreamError(
                "verified source receipt eligible census is incomplete"
            )
        eligible_ids = receipt["model_eligible_game_ids"]
        if not isinstance(eligible_ids, list) or not eligible_ids or not all(
            isinstance(value, str) and value.strip() for value in eligible_ids
        ):
            raise FutureValueDownstreamError(
                "verified source receipt eligible IDs are invalid"
            )
        canonical_eligible_ids = canonical_game_ids(eligible_ids)
        if list(canonical_eligible_ids) != eligible_ids:
            raise FutureValueDownstreamError(
                "verified source receipt eligible IDs are not canonical and unique"
            )
        eligible_count = receipt["model_eligible_game_count"]
        eligible_identity = receipt["model_eligible_identity_sha256"]
        if (
            isinstance(eligible_count, bool)
            or not isinstance(eligible_count, int)
            or eligible_count != len(canonical_eligible_ids)
            or not isinstance(eligible_identity, str)
            or SHA256_RE.fullmatch(eligible_identity) is None
            or eligible_identity != identity_sha256(canonical_eligible_ids)
            or not set(canonical_eligible_ids).issubset(set(accepted_ids))
        ):
            raise FutureValueDownstreamError(
                "verified source receipt eligible census identity is invalid"
            )

    _validate_source_file_records(receipt.get("source_files"), receipt_path=receipt_path)
    authority = receipt.get("authority")
    if not isinstance(authority, Mapping) or authority.get("research_only") is not True:
        raise FutureValueDownstreamError("verified source receipt authority is invalid")
    if any(bool(flag) for name, flag in authority.items() if name != "research_only"):
        raise FutureValueDownstreamError("verified source receipt grants authority")

    receipt_hash = receipt.get("receipt_sha256")
    if not isinstance(receipt_hash, str) or SHA256_RE.fullmatch(receipt_hash) is None:
        raise FutureValueDownstreamError("verified source receipt hash is invalid")
    payload = dict(receipt)
    payload.pop("receipt_sha256", None)
    expected_hash = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    if expected_hash != receipt_hash:
        raise FutureValueDownstreamError("verified source receipt hash does not match payload")
    return receipt


def _load_source_receipt(value: Mapping[str, Any] | Path) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return _validate_source_receipt(value)
    path = Path(value)
    if not path.is_file() or path.is_symlink():
        raise FutureValueDownstreamError("canonical source receipt is missing or unsafe")
    raw = _read_json(path)
    if not isinstance(raw, Mapping):
        raise FutureValueDownstreamError("canonical source receipt is not an object")
    return _validate_source_receipt(raw, receipt_path=path)


def _safe_child(root: Path, relative: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise FutureValueDownstreamError(f"artifact path escapes root: {relative}") from error
    if path.is_symlink():
        raise FutureValueDownstreamError(f"artifact path is a symlink: {relative}")
    return path


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FutureValueDownstreamError(f"artifact JSON cannot be read: {path}") from error


def _scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _identity_part(row: Mapping[str, Any], fields: Sequence[str]) -> str | None:
    for field in fields:
        value = row.get(field)
        if value is None or isinstance(value, (dict, list, tuple)):
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _row_identity(row: Mapping[str, Any], spec: DownstreamArtifactSpec) -> str | None:
    parts: list[str] = []
    for group in spec.identity_groups:
        value = _identity_part(row, group)
        if value is None:
            container_key = row.get("_container_key")
            if len(spec.identity_groups) == 1 and container_key is not None:
                value = str(container_key).strip() or None
            if value is None:
                return None
        parts.append(value)
    return "|".join(parts)


def _numeric(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _candidate_rows(
    value: Any,
    spec: DownstreamArtifactSpec,
    *,
    container_key: str | None = None,
    keyed_mapping: bool = False,
) -> list[dict[str, Any]]:
    """Collect records from the common public-pack JSON shapes."""

    rows: list[dict[str, Any]] = []
    if isinstance(value, list):
        for item in value:
            rows.extend(
                _candidate_rows(
                    item,
                    spec,
                    container_key=container_key,
                    keyed_mapping=keyed_mapping,
                )
            )
        return rows
    if not isinstance(value, Mapping):
        return rows

    row = dict(value)
    if container_key is not None and "_container_key" not in row:
        row["_container_key"] = container_key
    identity = _row_identity(row, spec)
    has_value = any(field in row for field in spec.value_fields)
    if identity is not None and (
        has_value or spec.name in {"public_manifest", "profiles", "matches"}
    ):
        rows.append(row)

    # The manifest itself is one scalar artifact.  Nested release and ratings
    # records repeat the pack identity and must not become duplicate rows.
    if spec.name == "public_manifest":
        return rows

    if keyed_mapping:
        for key, child in value.items():
            if isinstance(child, (Mapping, list)):
                rows.extend(
                    _candidate_rows(
                        child,
                        spec,
                        container_key=str(key),
                        keyed_mapping=False,
                    )
                )
        return rows

    keyed_containers = {
        "games",
        "records",
        "rows",
        "players",
        "teams",
        "matches",
        "by_game",
        "by_player",
        "by_team",
    }
    for key, child in value.items():
        if isinstance(child, (Mapping, list)):
            rows.extend(
                _candidate_rows(
                    child,
                    spec,
                    keyed_mapping=isinstance(child, Mapping) and key in keyed_containers,
                )
            )
    return rows


def _binding_from_mapping(value: Mapping[str, Any]) -> dict[str, Any] | None:
    if not all(field in value for field in SOURCE_FIELDS):
        return None
    source_as_of = value.get("source_as_of")
    source_count = value.get("source_game_count")
    source_identity = value.get("source_identity_sha256")
    if not isinstance(source_as_of, str) or not source_as_of.strip():
        return None
    if isinstance(source_count, bool) or not isinstance(source_count, int) or source_count < 0:
        return None
    if not isinstance(source_identity, str) or not SHA256_RE.fullmatch(source_identity):
        return None
    return {
        "source_as_of": source_as_of,
        "source_game_count": source_count,
        "source_identity_sha256": source_identity,
    }


def _find_bindings(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    if depth > 5:
        return []
    if isinstance(value, Mapping):
        direct = _binding_from_mapping(value)
        found = [direct] if direct is not None else []
        for key in (
            "source",
            "source_binding",
            "source_receipt",
            "ratings",
            "metadata",
            "release",
            "receipt",
            "audit",
        ):
            child = value.get(key)
            if isinstance(child, (Mapping, list)):
                found.extend(_find_bindings(child, depth=depth + 1))
        return found
    if isinstance(value, list):
        list_found: list[dict[str, Any]] = []
        for child in value[:50]:
            list_found.extend(_find_bindings(child, depth=depth + 1))
        return list_found
    return []


def _find_source_receipts(value: Any, *, depth: int = 0) -> list[dict[str, Any]]:
    """Find embedded full source receipts without trusting shallow bindings."""

    if depth > 6:
        return []
    if isinstance(value, Mapping):
        found: list[dict[str, Any]] = []
        if value.get("schema_version") == SOURCE_RECEIPT_SCHEMA_VERSION:
            try:
                found.append(_validate_source_receipt(value))
            except FutureValueDownstreamError:
                # Keep an invalid marker so the caller can block the report.
                found.append({"__invalid_source_receipt__": True})
        for child in value.values():
            if isinstance(child, (Mapping, list)):
                found.extend(_find_source_receipts(child, depth=depth + 1))
        return found
    if isinstance(value, list):
        found: list[dict[str, Any]] = []
        for child in value[:100]:
            found.extend(_find_source_receipts(child, depth=depth + 1))
        return found
    return []


def _find_embedded_census_bindings(
    value: Any,
    *,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Find source payloads that carry the accepted IDs without a full receipt."""

    if depth > 6:
        return []
    if isinstance(value, Mapping):
        found: list[dict[str, Any]] = []
        if all(field in value for field in SOURCE_FIELDS) and "accepted_game_ids" in value:
            try:
                raw_ids = value.get("accepted_game_ids")
                if not isinstance(raw_ids, list) or not all(
                    isinstance(item, str) and item.strip() for item in raw_ids
                ):
                    raise FutureValueDownstreamError("embedded accepted IDs are invalid")
                ids = canonical_game_ids(raw_ids)
                if list(ids) != raw_ids:
                    raise FutureValueDownstreamError(
                        "embedded accepted IDs are not canonical and unique"
                    )
                if (
                    value.get("source_game_count") != len(ids)
                    or value.get("source_identity_sha256") != identity_sha256(ids)
                ):
                    raise FutureValueDownstreamError(
                        "embedded census identity is invalid"
                    )
                source_as_of = value.get("source_as_of")
                if _canonical_utc_text(source_as_of, "source_as_of") != source_as_of:
                    raise FutureValueDownstreamError("embedded source date is invalid")
                found.append(
                    {
                        "source_as_of": source_as_of,
                        "source_game_count": len(ids),
                        "source_identity_sha256": identity_sha256(ids),
                        "accepted_game_ids": list(ids),
                        "source_receipt_sha256": value.get("source_receipt_sha256"),
                    }
                )
            except FutureValueDownstreamError:
                found.append({"__invalid_embedded_census__": True})
        for child in value.values():
            if isinstance(child, (Mapping, list)):
                found.extend(_find_embedded_census_bindings(child, depth=depth + 1))
        return found
    if isinstance(value, list):
        found: list[dict[str, Any]] = []
        for child in value[:100]:
            found.extend(_find_embedded_census_bindings(child, depth=depth + 1))
        return found
    return []


def _unique_bindings(bindings: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        key = hashlib.sha256(_canonical_json_bytes(dict(binding))).hexdigest()
        unique[key] = dict(binding)
    return list(unique.values())


def _unique_receipts(receipts: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        key = hashlib.sha256(_canonical_json_bytes(dict(receipt))).hexdigest()
        unique[key] = dict(receipt)
    return list(unique.values())


def _same_binding(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return all(left.get(field) == right.get(field) for field in SOURCE_FIELDS)


def _validate_expected_binding(
    source_binding: Mapping[str, Any] | Path,
) -> dict[str, Any]:
    """Require the caller to provide the complete verified source receipt."""

    return _load_source_receipt(source_binding)


def _gate_passed(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, Mapping):
        return value.get("status") in {"passed", "complete", "available"} or value.get("passed") is True
    return isinstance(value, str) and value.casefold() in {"passed", "complete", "available"}


def _load_evaluation_receipt(
    candidate_root: Path,
    evaluation_receipt: Mapping[str, Any] | Path | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    blockers: list[str] = []
    if evaluation_receipt is None:
        path = _safe_child(candidate_root, DEFAULT_EVALUATION_FILE)
        if not path.is_file():
            return None, ["candidate_evaluation_receipt_missing"]
        raw = _read_json(path)
    elif isinstance(evaluation_receipt, Mapping):
        raw = dict(evaluation_receipt)
    else:
        path = Path(evaluation_receipt)
        if not path.is_file() or path.is_symlink():
            return None, ["candidate_evaluation_receipt_missing"]
        raw = _read_json(path)
    if not isinstance(raw, dict) or raw.get("schema_version") != EVALUATION_SCHEMA_VERSION:
        blockers.append("candidate_evaluation_receipt_schema_invalid")
        return None, blockers
    if raw.get("status") != "complete":
        blockers.append("candidate_evaluation_incomplete")
    gates = raw.get("gates")
    if not isinstance(gates, Mapping):
        blockers.append("required_evaluation_gates_missing")
    else:
        for gate in REQUIRED_EVALUATION_GATES:
            if not _gate_passed(gates.get(gate)):
                blockers.append(f"evaluation_gate_{gate}_missing_or_failed")
    authority = raw.get("authority")
    if isinstance(authority, Mapping) and any(bool(value) for value in authority.values()):
        blockers.append("evaluation_receipt_grants_authority")
    return raw, blockers


def _stats(deltas: Sequence[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in deltas if math.isfinite(float(value)))
    if not finite:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95_abs": None,
            "max_abs": None,
            "changed_count": 0,
        }
    absolute = sorted(abs(value) for value in finite)
    index = min(len(absolute) - 1, max(0, math.ceil(0.95 * len(absolute)) - 1))
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "median": finite[len(finite) // 2] if len(finite) % 2 else (finite[len(finite) // 2 - 1] + finite[len(finite) // 2]) / 2,
        "p95_abs": absolute[index],
        "max_abs": absolute[-1],
        "changed_count": sum(abs(value) > 1e-12 for value in finite),
    }


def _artifact_paths(root: Path, spec: DownstreamArtifactSpec) -> list[Path]:
    root = root.resolve()
    if spec.glob:
        paths = sorted(root.glob(spec.path))
    else:
        paths = [_safe_child(root, spec.path)]
    safe: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as error:
            raise FutureValueDownstreamError(f"artifact path escapes root: {path}") from error
        if path.is_symlink():
            raise FutureValueDownstreamError(f"artifact path is a symlink: {path}")
        safe.append(path)
    return safe


def _payload_rows(payload: Any, spec: DownstreamArtifactSpec) -> list[dict[str, Any]]:
    """Select the primary record collection for game-oriented artifacts."""

    if spec.name in {"draft_score", "profiles", "matches"} and isinstance(payload, Mapping):
        games = payload.get("games")
        if isinstance(games, Mapping):
            rows: list[dict[str, Any]] = []
            for key, value in games.items():
                if isinstance(value, Mapping):
                    row = dict(value)
                    row["_container_key"] = str(key)
                    rows.append(row)
            return rows
        if isinstance(games, list):
            return [dict(value) for value in games if isinstance(value, Mapping)]
    return _candidate_rows(payload, spec)


def _load_artifact(root: Path, spec: DownstreamArtifactSpec) -> dict[str, Any]:
    paths = _artifact_paths(root, spec)
    if not paths or any(not path.is_file() for path in paths):
        raise FutureValueDownstreamError(f"required artifact missing: {spec.name}")
    payloads = [_read_json(path) for path in paths]
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        rows.extend(_payload_rows(payload, spec))
    by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in rows:
        identity = _row_identity(row, spec)
        if identity is None:
            continue
        if identity in by_id:
            duplicate_ids.append(identity)
        else:
            by_id[identity] = row
    return {
        "paths": [str(path) for path in paths],
        "bytes": sum(path.stat().st_size for path in paths),
        "sha256": hashlib.sha256(b"".join(_sha256(path).encode() for path in paths)).hexdigest(),
        "payloads": payloads,
        "rows": by_id,
        "duplicate_ids": sorted(set(duplicate_ids)),
    }


def _compare_artifact(
    spec: DownstreamArtifactSpec,
    old: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    old_rows = dict(old["rows"])
    candidate_rows = dict(candidate["rows"])
    old_ids = set(old_rows)
    candidate_ids = set(candidate_rows)
    common_ids = sorted(old_ids & candidate_ids)
    all_fields = set(spec.value_fields)
    for row in (*old_rows.values(), *candidate_rows.values()):
        all_fields.update(
            key for key, value in row.items() if key not in {"_container_key"} and _numeric(value) is not None
        )
    deltas: dict[str, dict[str, Any]] = {}
    for field in sorted(all_fields):
        values: list[float] = []
        missing_old = 0
        missing_candidate = 0
        for identity in common_ids:
            left = _numeric(old_rows[identity].get(field))
            right = _numeric(candidate_rows[identity].get(field))
            if left is None:
                missing_old += 1
            if right is None:
                missing_candidate += 1
            if left is not None and right is not None:
                values.append(right - left)
        deltas[field] = {
            **_stats(values),
            "old_missing_count": missing_old,
            "candidate_missing_count": missing_candidate,
        }
    return {
        "old_rows": len(old_rows),
        "candidate_rows": len(candidate_rows),
        "matched_rows": len(common_ids),
        "added_rows": len(candidate_ids - old_ids),
        "removed_rows": len(old_ids - candidate_ids),
        "candidate_row_coverage": len(common_ids) / max(len(old_ids), 1),
        "old_duplicate_ids": list(old["duplicate_ids"]),
        "candidate_duplicate_ids": list(candidate["duplicate_ids"]),
        "deltas": deltas,
    }


def downstream_impact_contract() -> dict[str, Any]:
    """Return the fail-closed integration contract for downstream review."""

    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "status": "development_only",
        "purpose": "Compare current and future-value candidate downstream artifacts before any integration decision.",
        "required_source_fields": list(SOURCE_FIELDS),
        "required_source_receipt_fields": list(SOURCE_RECEIPT_FIELDS),
        "source_receipt_schema_version": SOURCE_RECEIPT_SCHEMA_VERSION,
        "source_receipt_status": SOURCE_RECEIPT_STATUS,
        "canonical_source_receipt_file": DEFAULT_SOURCE_FILES[0],
        "required_artifacts": [spec.as_dict() for spec in _ARTIFACT_SPECS],
        "required_evaluation_gates": list(REQUIRED_EVALUATION_GATES),
        "evaluation_schema_version": EVALUATION_SCHEMA_VERSION,
        "report_fields": [
            "source",
            "artifacts",
            "coverage",
            "deltas",
            "blockers",
            "authority",
        ],
        "authority": {
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
            "deployment": False,
        },
        "claim_ceiling": "source-bound research comparison only; no public rating or prediction authority",
    }


def evaluate_downstream_impact(
    old_root: Path,
    candidate_root: Path,
    *,
    source_binding: Mapping[str, Any] | Path | None = None,
    source_receipt: Mapping[str, Any] | Path | None = None,
    evaluation_receipt: Mapping[str, Any] | Path | None = None,
) -> dict[str, Any]:
    """Compare old and candidate public-facing artifacts.

    Every required artifact must exist in both roots.  A source binding must be
    found in each root, and the candidate evaluation receipt must pass every
    required gate.  A report can contain useful deltas while its authority stays
    unavailable.
    """

    old_root = Path(old_root)
    candidate_root = Path(candidate_root)
    if source_binding is not None and source_receipt is not None:
        raise FutureValueDownstreamError(
            "source_binding and source_receipt cannot both be supplied"
        )
    supplied_source = source_receipt if source_receipt is not None else source_binding
    if supplied_source is None:
        raise FutureValueDownstreamError("canonical verified source receipt is required")
    expected_receipt = _validate_expected_binding(supplied_source)
    expected = _source_summary(expected_receipt)
    blockers: list[str] = []
    loaded: dict[str, dict[str, dict[str, Any]]] = {"old": {}, "candidate": {}}
    bindings: dict[str, list[dict[str, Any]]] = {"old": [], "candidate": []}
    embedded_receipts: dict[str, list[dict[str, Any]]] = {
        "old": [],
        "candidate": [],
    }
    embedded_census: dict[str, list[dict[str, Any]]] = {
        "old": [],
        "candidate": [],
    }
    artifact_reports: dict[str, Any] = {}

    for label, root in (("old", old_root), ("candidate", candidate_root)):
        if not root.is_dir() or root.is_symlink():
            blockers.append(f"{label}_artifact_root_missing_or_unsafe")
            continue
        # A durable source receipt is allowed beside the artifact set.  It is
        # optional for older packs, but if present it must be valid and must
        # agree with the caller-supplied canonical receipt.
        try:
            receipt_path = _safe_child(root, DEFAULT_SOURCE_FILES[0])
        except FutureValueDownstreamError:
            embedded_receipts[label].append({"__invalid_source_receipt__": True})
        else:
            if receipt_path.is_file():
                try:
                    root_receipt = _validate_source_receipt(
                        _read_json(receipt_path), receipt_path=receipt_path
                    )
                except FutureValueDownstreamError:
                    embedded_receipts[label].append({"__invalid_source_receipt__": True})
                else:
                    embedded_receipts[label].append(root_receipt)
                    bindings[label].extend(_find_bindings(root_receipt))
                    embedded_census[label].extend(_find_embedded_census_bindings(root_receipt))
        for spec in _ARTIFACT_SPECS:
            try:
                item = _load_artifact(root, spec)
            except FutureValueDownstreamError:
                blockers.append(f"{label}_{spec.name}_missing_or_invalid")
                continue
            loaded[label][spec.name] = item
            for payload in item["payloads"]:
                bindings[label].extend(_find_bindings(payload))
                embedded_receipts[label].extend(_find_source_receipts(payload))
                embedded_census[label].extend(_find_embedded_census_bindings(payload))
            if item["duplicate_ids"]:
                blockers.append(f"{label}_{spec.name}_duplicate_identity_rows")
            if not item["rows"]:
                blockers.append(f"{label}_{spec.name}_rows_missing_identity")

    for label in ("old", "candidate"):
        unique = _unique_bindings(bindings[label])
        if not unique:
            blockers.append(f"{label}_source_binding_missing")
        elif len(unique) != 1:
            blockers.append(f"{label}_source_binding_conflicting")
        else:
            actual = unique[0]
            if not _same_binding(actual, expected_receipt):
                blockers.append(f"{label}_source_binding_mismatch")

        receipts = embedded_receipts[label]
        if any(item.get("__invalid_source_receipt__") for item in receipts):
            blockers.append(f"{label}_source_receipt_invalid")
        valid_receipts = [
            item for item in receipts if not item.get("__invalid_source_receipt__")
        ]
        if valid_receipts:
            unique_receipts = _unique_receipts(valid_receipts)
            if len(unique_receipts) != 1 or unique_receipts[0] != expected_receipt:
                blockers.append(f"{label}_source_receipt_mismatch")

        census_bindings = embedded_census[label]
        if any(item.get("__invalid_embedded_census__") for item in census_bindings):
            blockers.append(f"{label}_source_census_invalid")
        for item in census_bindings:
            if item.get("__invalid_embedded_census__"):
                continue
            if (
                item["source_as_of"] != expected_receipt["source_as_of"]
                or item["source_game_count"] != expected_receipt["source_game_count"]
                or item["source_identity_sha256"]
                != expected_receipt["source_identity_sha256"]
                or item["accepted_game_ids"] != expected_receipt["accepted_game_ids"]
            ):
                blockers.append(f"{label}_source_census_mismatch")
            source_receipt_hash = item.get("source_receipt_sha256")
            if source_receipt_hash is not None and source_receipt_hash != expected_receipt[
                "receipt_sha256"
            ]:
                blockers.append(f"{label}_source_receipt_hash_mismatch")

    evaluation, evaluation_blockers = _load_evaluation_receipt(candidate_root, evaluation_receipt)
    blockers.extend(evaluation_blockers)
    if evaluation is not None:
        eval_binding = _unique_bindings(_find_bindings(evaluation))
        if not eval_binding:
            blockers.append("evaluation_source_binding_missing")
        elif len(eval_binding) != 1 or not _same_binding(eval_binding[0], expected_receipt):
            blockers.append("evaluation_source_binding_mismatch")
        eval_receipts = _find_source_receipts(evaluation)
        if any(item.get("__invalid_source_receipt__") for item in eval_receipts):
            blockers.append("evaluation_source_receipt_invalid")
        valid_eval_receipts = [
            item for item in eval_receipts if not item.get("__invalid_source_receipt__")
        ]
        if valid_eval_receipts:
            unique_eval_receipts = _unique_receipts(valid_eval_receipts)
            if len(unique_eval_receipts) != 1 or unique_eval_receipts[0] != expected_receipt:
                blockers.append("evaluation_source_receipt_mismatch")
        eval_census = _find_embedded_census_bindings(evaluation)
        if any(item.get("__invalid_embedded_census__") for item in eval_census):
            blockers.append("evaluation_source_census_invalid")
        for item in eval_census:
            if item.get("__invalid_embedded_census__"):
                continue
            if (
                item["source_as_of"] != expected_receipt["source_as_of"]
                or item["source_game_count"] != expected_receipt["source_game_count"]
                or item["source_identity_sha256"]
                != expected_receipt["source_identity_sha256"]
                or item["accepted_game_ids"] != expected_receipt["accepted_game_ids"]
            ):
                blockers.append("evaluation_source_census_mismatch")
            source_receipt_hash = item.get("source_receipt_sha256")
            if source_receipt_hash is not None and source_receipt_hash != expected_receipt[
                "receipt_sha256"
            ]:
                blockers.append("evaluation_source_receipt_hash_mismatch")

    for spec in _ARTIFACT_SPECS:
        old = loaded["old"].get(spec.name)
        candidate = loaded["candidate"].get(spec.name)
        if old is None or candidate is None:
            continue
        artifact_reports[spec.name] = {
            "consumers": list(spec.consumers),
            "paths": {
                "old": old["paths"],
                "candidate": candidate["paths"],
            },
            "bytes": {"old": old["bytes"], "candidate": candidate["bytes"]},
            "sha256": {"old": old["sha256"], "candidate": candidate["sha256"]},
            "comparison": _compare_artifact(spec, old, candidate),
        }

    status = "ready_research_only" if not blockers else "blocked"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "generated_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": expected,
        "old_root": str(old_root),
        "candidate_root": str(candidate_root),
        "artifacts": artifact_reports,
        "coverage": {
            name: report["comparison"]
            for name, report in artifact_reports.items()
        },
        "deltas": {
            name: report["comparison"]["deltas"]
            for name, report in artifact_reports.items()
        },
        "evaluation": {
            "schema_version": evaluation.get("schema_version") if evaluation else None,
            "status": evaluation.get("status") if evaluation else "missing",
            "gates": evaluation.get("gates") if evaluation else {},
        },
        "blockers": sorted(set(blockers)),
        "authority": {
            "public_player_rating": False,
            "public_team_rating": False,
            "public_probability": False,
            "odds": False,
            "expected_value": False,
            "recommendation": False,
            "betting": False,
            "promotion": False,
            "deployment": False,
        },
        "claim_ceiling": "source-bound research comparison only; no public rating or prediction authority",
    }
    return report


def write_downstream_report(path: Path, report: Mapping[str, Any]) -> None:
    """Write a deterministic JSON report without changing any source artifact."""

    if report.get("schema_version") != SCHEMA_VERSION:
        raise FutureValueDownstreamError("downstream report schema is invalid")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(report), ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "CONTRACT_SCHEMA_VERSION",
    "DEFAULT_EVALUATION_FILE",
    "DownstreamArtifactSpec",
    "FutureValueDownstreamError",
    "REQUIRED_EVALUATION_GATES",
    "SCHEMA_VERSION",
    "SOURCE_FIELDS",
    "SOURCE_RECEIPT_FIELDS",
    "SOURCE_RECEIPT_SCHEMA_VERSION",
    "SOURCE_RECEIPT_STATUS",
    "downstream_impact_contract",
    "evaluate_downstream_impact",
    "required_artifact_specs",
    "write_downstream_report",
]
