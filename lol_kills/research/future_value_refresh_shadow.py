"""Write a research-only future-value shadow receipt during public refresh.

The unified refresh owns the accepted source census and the current rating
step.  This adapter records that boundary for a later future-value build.  It
does not fit a model, copy an artifact, alter a public pack, or grant model
authority.

The adapter is deliberately small.  A refresh can keep producing the current
public release while future-value inputs are missing or blocked.  An explicit
promotion request is a different operation and fails before the refresh can
stage or activate a public release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Any

from lol_kills.etl.source_keys import canonical_source_game_key
from lol_kills.v2.tierlists.accepted_census import canonical_game_ids, identity_sha256


SCHEMA_VERSION = "scryglass:future-value-refresh-shadow:v1"
RECEIPT_DIRECTORY = Path("data/lol/runtime/future-value-shadow")
ARTIFACT_NAMES = ("model", "snapshot", "tier", "draft")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

AUTHORITY: dict[str, bool] = {
    "research_only": True,
    "public_authority": False,
    "public_player_rating": False,
    "public_team_rating": False,
    "public_tierlist": False,
    "public_draft_score": False,
    "public_probability": False,
    "odds": False,
    "expected_value": False,
    "recommendation": False,
    "betting": False,
    "promotion": False,
    "deployment": False,
    "merge": False,
}


class FutureValueShadowError(RuntimeError):
    """The research shadow adapter could not complete its local receipt."""


class FutureValueShadowPromotionError(FutureValueShadowError):
    """An unauthorized request attempted to promote a shadow variant."""


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
        raise FutureValueShadowError("shadow receipt is not canonical JSON") from error


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_path(path: Path) -> str:
    """Hash one regular, non-symlink file."""

    if path.is_symlink() or not path.is_file():
        raise FutureValueShadowError(f"artifact is missing or unsafe: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash(value: object, label: str) -> str | None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value.lower()) is None:
        return None
    return value.lower()


def _utc_text(value: object) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        stamp = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamp.tzinfo is None:
        return None
    return stamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _safe_ids(values: object) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        return ()
    raw = [canonical_source_game_key(value) for value in values]
    return tuple(canonical_game_ids(value for value in raw if value))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(_canonical_json_bytes(value) + b"\n")
    os.replace(temporary, path)


def _artifact_record(name: str, value: object, blockers: list[str]) -> dict[str, Any]:
    """Validate one supplied artifact binding without copying the artifact."""

    record: dict[str, Any] = {"name": name, "status": "missing"}
    path: Path | None = None
    claimed: str | None = None
    if isinstance(value, Mapping):
        raw_path = value.get("path") or value.get("locator")
        if isinstance(raw_path, str) and raw_path.strip():
            path = Path(raw_path).expanduser().resolve()
        claimed = _hash(
            value.get("sha256") or value.get("artifact_sha256") or value.get("raw_sha256"),
            f"{name} artifact",
        )
    elif isinstance(value, str):
        if SHA256_RE.fullmatch(value.lower()):
            claimed = value.lower()
        elif value.strip():
            path = Path(value).expanduser().resolve()

    if path is not None:
        record["path"] = str(path)
        try:
            actual = sha256_path(path)
        except (OSError, FutureValueShadowError):
            blockers.append(f"{name}_artifact_unavailable")
            return record
        record["bytes"] = int(path.stat().st_size)
        record["sha256"] = actual
        if claimed is not None and claimed != actual:
            blockers.append(f"{name}_artifact_hash_mismatch")
            record["status"] = "blocked"
            record["claimed_sha256"] = claimed
            return record
        record["status"] = "available"
        return record

    if claimed is not None:
        record["sha256"] = claimed
        record["status"] = "hash_supplied"
        return record

    blockers.append(f"{name}_artifact_missing")
    return record


def _artifact_records(
    supplied: Mapping[str, Any] | None,
    blockers: list[str],
) -> dict[str, dict[str, Any]]:
    values = supplied if isinstance(supplied, Mapping) else {}
    result: dict[str, dict[str, Any]] = {}
    for name in ARTIFACT_NAMES:
        value: object | None = None
        for key in (name, f"{name}_artifact", f"future_value_{name}"):
            if key in values:
                value = values[key]
                break
        result[name] = _artifact_record(name, value, blockers)
    return result


def reject_unauthorized_promotion(requested_variant: object) -> None:
    """Reject every explicit promotion request until an independent gate exists."""

    if requested_variant is None:
        return
    if isinstance(requested_variant, str) and not requested_variant.strip():
        return
    raise FutureValueShadowPromotionError(
        "future-value shadow promotion is unavailable; independent authorization is required"
    )


def _source_binding(
    *,
    source_as_of: object,
    source_game_ids: object,
    source_game_count: object,
    source_identity_sha256: object,
    source_receipt_sha256: object,
    accepted_source_receipt_path: Path | None,
    blockers: list[str],
) -> dict[str, Any]:
    ids = _safe_ids(source_game_ids)
    claimed_count = source_game_count
    if isinstance(claimed_count, bool):
        claimed_count = None
    try:
        count = int(claimed_count) if claimed_count is not None else len(ids)
    except (TypeError, ValueError):
        count = -1
    computed_identity = identity_sha256(ids) if ids else ""
    claimed_identity = _hash(source_identity_sha256, "source identity")
    if not ids:
        blockers.append("accepted_source_census_missing")
    if count != len(ids):
        blockers.append("accepted_source_game_count_mismatch")
    if claimed_identity is None:
        blockers.append("accepted_source_identity_missing")
    elif claimed_identity != computed_identity:
        blockers.append("accepted_source_identity_mismatch")
    as_of = _utc_text(source_as_of)
    if as_of is None:
        blockers.append("accepted_source_as_of_missing_or_invalid")
    source_receipt = _hash(source_receipt_sha256, "source receipt")
    source_file: dict[str, Any] | None = None
    if accepted_source_receipt_path is not None:
        source_file = {"path": str(accepted_source_receipt_path)}
        try:
            source_file["bytes"] = int(accepted_source_receipt_path.stat().st_size)
            source_file["sha256"] = sha256_path(accepted_source_receipt_path)
        except (OSError, FutureValueShadowError):
            blockers.append("accepted_source_receipt_unavailable")
        else:
            try:
                payload = json.loads(accepted_source_receipt_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, Mapping) and "receipt_canonical_sha256" in payload:
                claimed_canonical = _hash(
                    payload.get("receipt_canonical_sha256"),
                    "accepted source receipt canonical hash",
                )
                unsigned = dict(payload)
                unsigned.pop("receipt_canonical_sha256", None)
                actual_canonical = _sha256_bytes(
                    json.dumps(
                        unsigned,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                )
                if claimed_canonical is None or claimed_canonical != actual_canonical:
                    blockers.append("accepted_source_receipt_invalid")
                elif source_receipt not in {None, claimed_canonical}:
                    blockers.append("accepted_source_receipt_hash_mismatch")
    if source_receipt is None:
        blockers.append("accepted_source_receipt_hash_missing")
    return {
        "source_as_of": as_of,
        "source_game_count": count if count >= 0 else None,
        "source_identity_sha256": claimed_identity,
        "computed_identity_sha256": computed_identity or None,
        "accepted_game_ids": list(ids),
        "source_receipt_sha256": source_receipt,
        "source_receipt_file": source_file,
    }


def _receipt_path(runtime_root: Path, run_id: str) -> Path:
    root = runtime_root.resolve()
    return root / RECEIPT_DIRECTORY / run_id / "future-value-shadow-receipt.json"


def run_future_value_refresh_shadow(
    *,
    runtime_root: Path,
    source_as_of: object,
    source_game_ids: object,
    source_game_count: object = None,
    source_identity_sha256: object = None,
    source_receipt_sha256: object = None,
    accepted_source_receipt_path: Path | None = None,
    current_ratings: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
    checked_at: datetime | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Write one durable, non-authoritative shadow receipt.

    The function records unavailable inputs as blockers and returns normally.
    This property lets the current public refresh finish when future-value
    research is incomplete.
    """

    started = time.perf_counter()
    blockers: list[str] = []
    source = _source_binding(
        source_as_of=source_as_of,
        source_game_ids=source_game_ids,
        source_game_count=source_game_count,
        source_identity_sha256=source_identity_sha256,
        source_receipt_sha256=source_receipt_sha256,
        accepted_source_receipt_path=accepted_source_receipt_path,
        blockers=blockers,
    )
    ratings = dict(current_ratings or {})
    rating_status = str(ratings.get("status") or "")
    ratings_ready = rating_status in {"published", "no_change"}
    if not ratings_ready:
        blockers.append("current_ratings_unavailable")
    rating_identity = _hash(
        ratings.get("source_identity_sha256"), "current ratings identity"
    )
    if rating_identity is not None and source["source_identity_sha256"] not in {
        None,
        rating_identity,
    }:
        blockers.append("current_ratings_source_identity_mismatch")
    artifact_bindings = _artifact_records(artifacts, blockers)
    timestamp = checked_at or datetime.now(timezone.utc)
    checked_text = timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    identifier = run_id or f"shadow-{timestamp.astimezone(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
    path = _receipt_path(runtime_root, identifier)
    # Keep ordering stable while allowing a caller to report the same blocker
    # from source and ratings checks only once.
    blockers = sorted(set(blockers))
    receipt: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "research_only_blocked" if blockers else "research_only_available",
        "checked_at": checked_text,
        "source": source,
        "current_ratings": {
            "status": rating_status or None,
            "ready": ratings_ready,
            "pack_id": ratings.get("pack_id"),
            "source_identity_sha256": rating_identity,
        },
        "artifacts": artifact_bindings,
        "coverage": {
            "accepted_game_count": source["source_game_count"],
            "accepted_game_id_count": len(source["accepted_game_ids"]),
            "artifact_count": sum(
                record["status"] in {"available", "hash_supplied"}
                for record in artifact_bindings.values()
            ),
            "artifact_required_count": len(ARTIFACT_NAMES),
        },
        "blockers": blockers,
        "timing": {},
        "authority": dict(AUTHORITY),
        "writes_public_artifacts": False,
        "stage_or_activation": False,
        "receipt_path": str(path),
        "write_error": None,
    }
    receipt["timing"] = {
        "wall_seconds": max(0.0, time.perf_counter() - started),
    }
    receipt["receipt_sha256"] = _sha256_bytes(_canonical_json_bytes(receipt))
    write_error: str | None = None
    try:
        _atomic_json(path, receipt)
    except (OSError, FutureValueShadowError) as error:
        write_error = f"{type(error).__name__}: {str(error)[:300]}"
        receipt["blockers"] = sorted(set([*blockers, "shadow_receipt_write_failed"]))
        receipt["status"] = "research_only_blocked"
        receipt["write_error"] = write_error
        receipt["receipt_sha256"] = _sha256_bytes(
            _canonical_json_bytes({key: value for key, value in receipt.items() if key != "receipt_sha256"})
        )
    receipt["write_error"] = write_error
    return receipt


# Short alias for callers that use the generic research-shadow terminology.
run_shadow = run_future_value_refresh_shadow


__all__ = [
    "ARTIFACT_NAMES",
    "AUTHORITY",
    "FutureValueShadowError",
    "FutureValueShadowPromotionError",
    "RECEIPT_DIRECTORY",
    "SCHEMA_VERSION",
    "reject_unauthorized_promotion",
    "run_future_value_refresh_shadow",
    "run_shadow",
    "sha256_path",
]
