"""Resolve audited Oracle's Elixir patch tokens.

The OE token and the Riot patch label use different namespaces. This module
keeps the namespaces separate and resolves only rows in the signed sidecar.
An exact atom snapshot is returned only when the row has a pinned snapshot and
the requested event time is inside the audited source interval.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


SCHEMA_VERSION = "scryglass:oe-atom-patch-map:v1"
DEFAULT_MAPPING_PATH = Path(
    "data/lol/v2/champions/oe-atom-patch-map-v1.json"
)
OE_TOKEN_RE = re.compile(r"^(?P<major>\d{2})\.(?P<minor>\d{1,2})$")
PATCH_RE = re.compile(r"^\d{2}\.\d{2}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PatchMappingError(ValueError):
    """Raised when the mapping sidecar is not safe to use."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_oe_token(value: object) -> str:
    """Return the two-decimal form of an OE token without using floats.

    OE exports can turn ``15.10`` into the string ``15.1``. A one-digit minor
    therefore means ten times that digit. It does not mean patch ``15.01``.
    """

    if isinstance(value, (bool, int, float)) or value is None:
        raise PatchMappingError("OE patch token must be a string")
    text = str(value).strip()
    match = OE_TOKEN_RE.fullmatch(text)
    if match is None:
        raise PatchMappingError(f"malformed OE patch token: {value!r}")
    minor = match.group("minor")
    canonical_minor = f"{minor}0" if len(minor) == 1 else minor
    return f"{match.group('major')}.{canonical_minor}"


def _live_oe_token_candidates(value: object) -> tuple[str, ...]:
    """Return safe aliases for a one-digit OE token from a float-like export.

    OE has used both ``15.1`` for patch ``15.10`` and ``16.2`` for patch
    ``16.02``.  The raw token has lost the leading zero, so the normalizer
    keeps its historical trailing-zero behavior for public callers.  Live
    binding may also try the zero-padded spelling, but only when that exact
    spelling exists in the audited sidecar.
    """

    normalized = normalize_oe_token(value)
    candidates = [normalized]
    text = str(value).strip()
    match = OE_TOKEN_RE.fullmatch(text)
    if match is not None and len(match.group("minor")) == 1:
        padded = f"{match.group('major')}.{match.group('minor').zfill(2)}"
        if padded not in candidates:
            candidates.append(padded)
    return tuple(candidates)


def _parse_utc(value: object, *, field: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise PatchMappingError(f"{field} is required")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as exc:
        raise PatchMappingError(f"{field} is not RFC 3339: {value!r}") from exc
    if parsed.tzinfo is None:
        raise PatchMappingError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _rfc3339(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _require_sha(value: object, *, field: str) -> str:
    text = str(value or "")
    if SHA256_RE.fullmatch(text) is None:
        raise PatchMappingError(f"{field} must be a lowercase SHA-256 hash")
    return text


def _repo_root_for(path: Path) -> Path | None:
    for candidate in (path.parent, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _unsigned_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    unsigned = dict(payload)
    unsigned.pop("artifact_sha256", None)
    return unsigned


def _artifact_sha256(payload: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(_unsigned_payload(payload)))


@dataclass(frozen=True)
class MappingArtifact:
    """Validated mapping sidecar."""

    payload: dict[str, Any]
    rows: dict[str, dict[str, Any]]
    path: Path
    repo_root: Path | None
    live_source: dict[str, Any] | None = None


@dataclass(frozen=True)
class PatchResolution:
    """Result of a time-safe patch lookup."""

    oe_token: str | None
    as_of: str | None
    status: str
    reason: str
    official_patch: str | None = None
    atom_snapshot_patch: str | None = None
    confidence: str | None = None
    ambiguity_status: str | None = None
    source_interval: dict[str, str] | None = None
    evidence: tuple[dict[str, Any], ...] = ()

    @property
    def exact_official_patch(self) -> bool:
        return self.status == "resolved" and self.official_patch is not None

    @property
    def exact_atom_snapshot(self) -> bool:
        return self.status == "resolved" and self.atom_snapshot_patch is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "oe_token": self.oe_token,
            "as_of": self.as_of,
            "status": self.status,
            "reason": self.reason,
            "official_patch": self.official_patch,
            "atom_snapshot_patch": self.atom_snapshot_patch,
            "confidence": self.confidence,
            "ambiguity_status": self.ambiguity_status,
            "source_interval": self.source_interval,
            "evidence": list(self.evidence),
        }


def _validate_source_ref(source: Mapping[str, Any], *, field: str) -> None:
    if not isinstance(source, Mapping):
        raise PatchMappingError(f"{field} must be an object")
    locator = str(source.get("locator") or "").strip()
    if not locator:
        raise PatchMappingError(f"{field}.locator is required")
    _require_sha(source.get("sha256"), field=f"{field}.sha256")


def _source_by_kind(payload: Mapping[str, Any], kind: str) -> Mapping[str, Any] | None:
    for source in payload.get("sources", []):
        if isinstance(source, Mapping) and source.get("kind") == kind:
            return source
    return None


def _verify_mutable_atom_source(path: Path, source: Mapping[str, Any]) -> None:
    try:
        bridge = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchMappingError("LCC atom bridge is not valid JSON") from exc
    if not isinstance(bridge, Mapping):
        raise PatchMappingError("LCC atom bridge must be an object")
    if bridge.get("schema_id") != "scryglass.lcc-atom-bridge.v1":
        raise PatchMappingError("LCC atom bridge schema is not supported")
    if bridge.get("version") != "lcc-atom-bridge-v1":
        raise PatchMappingError("LCC atom bridge version is not supported")
    artifact_hash = _require_sha(bridge.get("artifact_sha256"), field="LCC atom bridge artifact_sha256")
    unsigned = dict(bridge)
    unsigned.pop("artifact_sha256", None)
    if _sha256_bytes(_canonical_json(unsigned)) != artifact_hash:
        raise PatchMappingError("LCC atom bridge canonical hash is invalid")
    semantic_hash = _require_sha(
        source.get("semantic_sha256"),
        field="mutable LCC atom bridge semantic_sha256",
    )
    semantic = dict(bridge)
    semantic.pop("artifact_sha256", None)
    semantic.pop("generated_at", None)
    provenance = semantic.get("provenance")
    if isinstance(provenance, Mapping):
        stable_provenance = dict(provenance)
        stable_provenance.pop("lcc_commit", None)
        stable_provenance.pop("lcc_repo", None)
        semantic["provenance"] = stable_provenance
    if _sha256_bytes(_canonical_json(semantic)) != semantic_hash:
        raise PatchMappingError("LCC atom bridge semantic hash does not match the audited source")


def _verify_local_sources(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> None:
    if repo_root is None:
        raise PatchMappingError("cannot verify local mapping sources without a repository root")
    for index, source in enumerate(payload.get("sources", [])):
        if not isinstance(source, Mapping):
            raise PatchMappingError(f"sources[{index}] must be an object")
        locator = str(source.get("locator") or "")
        if locator.startswith(("https://", "http://")):
            continue
        path = repo_root / locator
        if not path.is_file():
            raise PatchMappingError(f"mapping source is missing: {locator}")
        if source.get("mutable_live_source") is True or source.get("mutable_source") is True:
            if source.get("kind") in {"oe_live_player_games", "oe_live_meta"}:
                continue
            if source.get("kind") == "lcc_atom_bridge":
                _verify_mutable_atom_source(path, source)
                continue
            if source.get("kind") not in {"oe_live_player_games", "oe_live_meta", "lcc_atom_bridge"}:
                raise PatchMappingError(
                    f"mutable mapping source kind is not allowed: {source.get('kind')}"
                )
        expected = str(source.get("sha256") or "")
        actual = _sha256_path(path)
        if actual != expected:
            raise PatchMappingError(
                f"mapping source hash mismatch for {locator}: {actual} != {expected}"
            )


def _live_source_binding(
    payload: Mapping[str, Any],
    *,
    repo_root: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]] | None:
    """Bind current OE intervals while retaining the static audit sidecar.

    The OE live parquet and its receipt are expected to change after a refresh.
    Their current hashes belong in the refresh receipt. The sidecar keeps the
    audited token-to-official and token-to-atom evidence. This function binds
    the current source watermark and intervals to those audited rows.
    """

    if repo_root is None:
        return None
    player_source = _source_by_kind(payload, "oe_live_player_games")
    meta_source = _source_by_kind(payload, "oe_live_meta")
    if player_source is None and meta_source is None:
        return None
    if player_source is None or meta_source is None:
        raise PatchMappingError("OE live mapping sources must include parquet and metadata")
    if player_source.get("mutable_live_source") is not True or meta_source.get("mutable_live_source") is not True:
        raise PatchMappingError("OE live mapping sources must declare mutable_live_source")

    player_locator = str(player_source.get("locator") or "")
    meta_locator = str(meta_source.get("locator") or "")
    player_path = repo_root / player_locator
    meta_path = repo_root / meta_locator
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PatchMappingError("OE live metadata is not valid JSON") from exc
    if not isinstance(meta, Mapping):
        raise PatchMappingError("OE live metadata must be an object")
    if meta.get("schema_version") != "scryglass:oe-live-source:v1":
        raise PatchMappingError("OE live metadata schema is not supported")
    if meta.get("source_mode") != "oe_only":
        raise PatchMappingError("OE live mapping requires the OE-only source mode")
    source_window = payload.get("source_window")
    if not isinstance(source_window, Mapping):
        raise PatchMappingError("source_window is required before binding the OE live source")
    source_start = _parse_utc(source_window.get("start"), field="source_window.start")
    try:
        source_latest = _parse_utc(meta.get("source_latest"), field="oe_live.source_latest")
    except PatchMappingError:
        raise

    try:
        try:
            frame = pd.read_parquet(
                player_path,
                columns=["gameid", "game_uid", "date", "patch"],
            )
        except (ValueError, KeyError):
            frame = pd.read_parquet(player_path, columns=["gameid", "date", "patch"])
    except (OSError, ValueError, ImportError) as exc:
        raise PatchMappingError("OE live player parquet cannot be read") from exc
    required = {"gameid", "date", "patch"}
    if not required.issubset(frame.columns):
        raise PatchMappingError("OE live player parquet lacks patch interval columns")
    frame = frame.copy()
    frame["gameid"] = frame["gameid"].astype("string").str.strip()
    if "game_uid" in frame.columns:
        game_uid = frame["game_uid"].astype("string").str.strip()
        frame["source_game_key"] = game_uid.where(
            game_uid.notna() & game_uid.ne(""),
            frame["gameid"],
        )
    else:
        frame["source_game_key"] = frame["gameid"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce", utc=True)
    frame = frame[
        frame["source_game_key"].notna() & frame["source_game_key"].ne("")
    ]
    frame = frame[frame["date"].notna()]
    frame = frame[frame["date"] >= pd.Timestamp(source_start)]
    if frame.empty:
        raise PatchMappingError("OE live player parquet has no rows in the audited source window")

    unique_games = frame[["source_game_key", "date", "patch"]].drop_duplicates()
    conflicts = unique_games.groupby("source_game_key", sort=False).agg(
        date_count=("date", "nunique"),
        patch_count=("patch", "nunique"),
    )
    if (conflicts["date_count"] > 1).any() or (conflicts["patch_count"] > 1).any():
        raise PatchMappingError("OE live source has a game with conflicting date or patch tokens")
    games = unique_games.drop_duplicates("source_game_key").copy()
    known_tokens = {
        str(row.get("oe_token"))
        for row in payload.get("mappings", [])
        if isinstance(row, Mapping)
    }
    normalized_tokens: list[str] = []
    for value in games["patch"].tolist():
        try:
            candidates = _live_oe_token_candidates(value)
        except PatchMappingError as exc:
            raise PatchMappingError(f"OE live source has a malformed patch token: {value!r}") from exc
        matched = next((candidate for candidate in candidates if candidate in known_tokens), None)
        normalized_tokens.append(matched or candidates[0])
    games["oe_token"] = normalized_tokens
    unknown_tokens = sorted(set(games["oe_token"]) - known_tokens)
    if unknown_tokens:
        raise PatchMappingError(
            "OE live source has patch tokens without audited mapping rows: "
            + ", ".join(unknown_tokens)
        )

    intervals: dict[str, dict[str, Any]] = {}
    for token, group in games.groupby("oe_token", sort=True):
        intervals[token] = {
            "observed_game_count": int(group["source_game_key"].nunique()),
            "oe_observed_interval": {
                "start": _rfc3339(pd.Timestamp(group["date"].min()).to_pydatetime()),
                "end": _rfc3339(pd.Timestamp(group["date"].max()).to_pydatetime()),
            },
        }
    observed_latest = pd.Timestamp(games["date"].max()).to_pydatetime().replace(tzinfo=timezone.utc)
    observed_latest_utc = observed_latest.astimezone(timezone.utc)
    if observed_latest_utc != source_latest:
        raise PatchMappingError("OE live metadata watermark does not match the player parquet")
    expected_maps = meta.get("maps")
    if isinstance(expected_maps, int) and expected_maps != int(games["source_game_key"].nunique()):
        raise PatchMappingError("OE live metadata map count does not match the player parquet")
    return intervals, {
        "status": "bound",
        "source_mode": "oe_only",
        "source_latest": _rfc3339(source_latest),
        "source_game_count": int(games["source_game_key"].nunique()),
        "source_token_count": len(intervals),
        "player_locator": player_locator,
        "player_raw_sha256": _sha256_path(player_path),
        "meta_locator": meta_locator,
        "meta_raw_sha256": _sha256_path(meta_path),
        "patch_intervals": intervals,
    }


def _bind_live_rows(
    rows: Mapping[str, dict[str, Any]],
    live_intervals: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    bound: dict[str, dict[str, Any]] = {}
    for token, row in rows.items():
        current = live_intervals.get(token)
        if current is None:
            bound[token] = dict(row)
            continue
        refreshed = dict(row)
        refreshed["observed_game_count"] = current["observed_game_count"]
        refreshed["oe_observed_interval"] = dict(current["oe_observed_interval"])
        bound[token] = refreshed
    return bound


def _validate_row(
    row: Mapping[str, Any],
    *,
    atom_snapshots: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(row, Mapping):
        raise PatchMappingError("mapping row must be an object")
    token = normalize_oe_token(row.get("oe_token"))
    if token != str(row.get("oe_token")):
        raise PatchMappingError(f"mapping row OE token is not canonical: {token}")
    source_tokens = row.get("source_tokens")
    if not isinstance(source_tokens, list) or not source_tokens:
        raise PatchMappingError(f"{token}: source_tokens must be a non-empty list")
    if any(normalize_oe_token(item) != token for item in source_tokens):
        raise PatchMappingError(f"{token}: source_tokens do not normalize to the row token")

    official_patch = row.get("official_patch")
    if official_patch is not None and PATCH_RE.fullmatch(str(official_patch)) is None:
        raise PatchMappingError(f"{token}: malformed official patch")
    atom_patch = row.get("atom_snapshot_patch")
    if atom_patch is not None and PATCH_RE.fullmatch(str(atom_patch)) is None:
        raise PatchMappingError(f"{token}: malformed atom snapshot patch")

    interval = row.get("oe_observed_interval")
    if not isinstance(interval, Mapping):
        raise PatchMappingError(f"{token}: OE interval is required")
    observed_game_count = row.get("observed_game_count")
    if (
        isinstance(observed_game_count, bool)
        or not isinstance(observed_game_count, int)
        or observed_game_count < 0
    ):
        raise PatchMappingError(f"{token}: observed_game_count must be a non-negative integer")
    start = _parse_utc(interval.get("start"), field=f"{token}.interval.start")
    end = _parse_utc(interval.get("end"), field=f"{token}.interval.end")
    if start > end:
        raise PatchMappingError(f"{token}: OE interval is reversed")

    release = _parse_utc(
        row.get("official_release_at"),
        field=f"{token}.official_release_at",
    )
    if start < release:
        raise PatchMappingError(
            f"{token}: first OE observation predates the official patch release"
        )

    audit_status = str(row.get("audit_status") or "")
    ambiguity_status = str(row.get("ambiguity_status") or "")
    confidence = str(row.get("confidence") or "")
    if audit_status not in {"audited", "unavailable"}:
        raise PatchMappingError(f"{token}: invalid audit_status")
    if ambiguity_status not in {"none", "atom_snapshot_unavailable"}:
        raise PatchMappingError(f"{token}: invalid ambiguity_status")
    if confidence not in {"high", "unavailable"}:
        raise PatchMappingError(f"{token}: invalid confidence")

    evidence = row.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 2:
        raise PatchMappingError(f"{token}: at least OE and Riot evidence are required")
    for index, source in enumerate(evidence):
        if not isinstance(source, Mapping):
            raise PatchMappingError(f"{token}.evidence[{index}] must be an object")
        _validate_source_ref(source, field=f"{token}.evidence[{index}]")

    if audit_status == "audited" and official_patch is None:
        raise PatchMappingError(f"{token}: audited row must have an official patch")
    if atom_patch is not None:
        atom_record = atom_snapshots.get(str(atom_patch))
        if atom_record is None:
            raise PatchMappingError(f"{token}: atom snapshot is not in the snapshot registry")
        if str(atom_record.get("official_patch")) != str(official_patch):
            raise PatchMappingError(f"{token}: atom snapshot and official patch disagree")
    if atom_patch is None and ambiguity_status != "atom_snapshot_unavailable":
        raise PatchMappingError(f"{token}: missing atom snapshot must be explicit")

    return dict(row)


def load_mapping(
    path: Path | str = DEFAULT_MAPPING_PATH,
    *,
    verify_source_hashes: bool = True,
    repo_root: Path | str | None = None,
) -> MappingArtifact:
    """Load and validate the sidecar, including its canonical hash."""

    mapping_path = Path(path)
    try:
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise PatchMappingError(f"cannot read patch mapping: {mapping_path}") from exc
    except json.JSONDecodeError as exc:
        raise PatchMappingError(f"patch mapping is invalid JSON: {mapping_path}") from exc
    if not isinstance(payload, dict):
        raise PatchMappingError("patch mapping root must be an object")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise PatchMappingError("patch mapping schema version is not supported")
    expected_artifact_hash = _require_sha(
        payload.get("artifact_sha256"),
        field="artifact_sha256",
    )
    actual_artifact_hash = _artifact_sha256(payload)
    if actual_artifact_hash != expected_artifact_hash:
        raise PatchMappingError(
            f"patch mapping hash mismatch: {actual_artifact_hash} != {expected_artifact_hash}"
        )

    scope = payload.get("source_window")
    if not isinstance(scope, Mapping):
        raise PatchMappingError("source_window is required")
    if _parse_utc(scope.get("start"), field="source_window.start") != datetime(
        2025, 1, 1, tzinfo=timezone.utc
    ):
        raise PatchMappingError("mapping scope must start at 2025-01-01T00:00:00Z")
    _parse_utc(scope.get("end"), field="source_window.end")

    sources = payload.get("sources")
    if not isinstance(sources, list) or not sources:
        raise PatchMappingError("sources are required")
    for index, source in enumerate(sources):
        if not isinstance(source, Mapping):
            raise PatchMappingError(f"sources[{index}] must be an object")
        _validate_source_ref(source, field=f"sources[{index}]")

    snapshot_rows = payload.get("atom_snapshots")
    if not isinstance(snapshot_rows, list):
        raise PatchMappingError("atom_snapshots must be a list")
    atom_snapshots: dict[str, Mapping[str, Any]] = {}
    for index, snapshot in enumerate(snapshot_rows):
        if not isinstance(snapshot, Mapping):
            raise PatchMappingError(f"atom_snapshots[{index}] must be an object")
        patch = str(snapshot.get("patch") or "")
        if PATCH_RE.fullmatch(patch) is None:
            raise PatchMappingError(f"atom_snapshots[{index}].patch is malformed")
        if patch in atom_snapshots:
            raise PatchMappingError(f"duplicate atom snapshot: {patch}")
        _parse_utc(snapshot.get("effective_from"), field=f"{patch}.effective_from")
        _validate_source_ref(snapshot.get("source"), field=f"atom_snapshots[{index}].source")
        atom_snapshots[patch] = snapshot

    rows = payload.get("mappings")
    if not isinstance(rows, list) or not rows:
        raise PatchMappingError("mappings must be a non-empty list")
    validated: dict[str, dict[str, Any]] = {}
    for row in rows:
        validated_row = _validate_row(row, atom_snapshots=atom_snapshots)
        token = str(validated_row["oe_token"])
        if token in validated:
            raise PatchMappingError(f"duplicate OE patch token: {token}")
        validated[token] = validated_row

    resolved_root = Path(repo_root) if repo_root is not None else _repo_root_for(mapping_path)
    live_source: dict[str, Any] | None = None
    if verify_source_hashes:
        _verify_local_sources(payload, repo_root=resolved_root)
    live_binding = _live_source_binding(payload, repo_root=resolved_root)
    if live_binding is not None:
        live_intervals, live_source = live_binding
        validated = _bind_live_rows(validated, live_intervals)

    return MappingArtifact(
        payload=dict(payload),
        rows=validated,
        path=mapping_path,
        repo_root=resolved_root,
        live_source=live_source,
    )


def _unavailable(
    *,
    token: str | None,
    as_of: str | None,
    reason: str,
    row: Mapping[str, Any] | None = None,
) -> PatchResolution:
    return PatchResolution(
        oe_token=token,
        as_of=as_of,
        status="unavailable",
        reason=reason,
        confidence=str(row.get("confidence")) if row else None,
        ambiguity_status=str(row.get("ambiguity_status")) if row else None,
        source_interval=dict(row.get("oe_observed_interval")) if row else None,
        evidence=tuple(row.get("evidence", ())) if row else (),
    )


def resolve_oe_patch(
    oe_token: object,
    as_of: object | None,
    *,
    mapping: MappingArtifact | None = None,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    verify_source_hashes: bool = True,
) -> PatchResolution:
    """Resolve one OE token at one source timestamp.

    The returned official patch is exact only for an audited row. The atom
    snapshot is exact only when that row names a registered time-safe snapshot.
    """

    try:
        token = normalize_oe_token(oe_token)
    except PatchMappingError:
        return _unavailable(token=None, as_of=None, reason="malformed_oe_token")
    artifact = mapping or load_mapping(
        mapping_path,
        verify_source_hashes=verify_source_hashes,
    )
    row = artifact.rows.get(token)
    if row is None:
        return _unavailable(token=token, as_of=None, reason="unknown_oe_token")
    if as_of is None:
        return _unavailable(token=token, as_of=None, reason="as_of_required", row=row)
    try:
        instant = _parse_utc(as_of, field="as_of")
    except PatchMappingError:
        return _unavailable(token=token, as_of=None, reason="invalid_as_of", row=row)
    as_of_text = _rfc3339(instant)
    interval = row["oe_observed_interval"]
    start = _parse_utc(interval["start"], field=f"{token}.interval.start")
    end = _parse_utc(interval["end"], field=f"{token}.interval.end")
    release = _parse_utc(row["official_release_at"], field=f"{token}.official_release_at")
    if instant < start or instant > end:
        return _unavailable(
            token=token,
            as_of=as_of_text,
            reason="as_of_outside_oe_source_interval",
            row=row,
        )
    if instant < release:
        return _unavailable(
            token=token,
            as_of=as_of_text,
            reason="as_of_predates_official_release",
            row=row,
        )
    if int(row.get("observed_game_count", 0)) < 1:
        return _unavailable(
            token=token,
            as_of=as_of_text,
            reason="no_accepted_oe_rows",
            row=row,
        )
    if row.get("audit_status") != "audited" or row.get("ambiguity_status") not in {
        "none",
        "atom_snapshot_unavailable",
    }:
        return _unavailable(
            token=token,
            as_of=as_of_text,
            reason="mapping_not_audited",
            row=row,
        )

    atom_patch = row.get("atom_snapshot_patch")
    if atom_patch is not None:
        snapshot = next(
            snapshot
            for snapshot in artifact.payload["atom_snapshots"]
            if snapshot["patch"] == atom_patch
        )
        effective_from = _parse_utc(
            snapshot["effective_from"],
            field=f"{atom_patch}.effective_from",
        )
        effective_to = snapshot.get("effective_to")
        if instant < effective_from or (
            effective_to is not None
            and instant >= _parse_utc(effective_to, field=f"{atom_patch}.effective_to")
        ):
            atom_patch = None

    return PatchResolution(
        oe_token=token,
        as_of=as_of_text,
        status="resolved",
        reason="audited_time_safe_mapping",
        official_patch=str(row["official_patch"]),
        atom_snapshot_patch=str(atom_patch) if atom_patch is not None else None,
        confidence=str(row["confidence"]),
        ambiguity_status=str(row["ambiguity_status"]),
        source_interval=dict(row["oe_observed_interval"]),
        evidence=tuple(row["evidence"]),
    )


def resolve_official_patch(
    oe_token: object,
    as_of: object | None,
    *,
    mapping: MappingArtifact | None = None,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    verify_source_hashes: bool = True,
) -> str | None:
    """Return the exact official patch or ``None`` when unavailable."""

    result = resolve_oe_patch(
        oe_token,
        as_of,
        mapping=mapping,
        mapping_path=mapping_path,
        verify_source_hashes=verify_source_hashes,
    )
    return result.official_patch if result.exact_official_patch else None


def resolve_atom_snapshot_patch(
    oe_token: object,
    as_of: object | None,
    *,
    mapping: MappingArtifact | None = None,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    verify_source_hashes: bool = True,
) -> str | None:
    """Return the exact atom snapshot patch or ``None`` on any uncertainty."""

    result = resolve_oe_patch(
        oe_token,
        as_of,
        mapping=mapping,
        mapping_path=mapping_path,
        verify_source_hashes=verify_source_hashes,
    )
    return result.atom_snapshot_patch if result.exact_atom_snapshot else None


__all__ = [
    "DEFAULT_MAPPING_PATH",
    "MappingArtifact",
    "PatchMappingError",
    "PatchResolution",
    "load_mapping",
    "normalize_oe_token",
    "resolve_atom_snapshot_patch",
    "resolve_official_patch",
    "resolve_oe_patch",
]
