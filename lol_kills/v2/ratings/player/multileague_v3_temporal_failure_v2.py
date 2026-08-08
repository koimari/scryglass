"""Preserve the v2/v1 future-dated receipt failure before supersession."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-temporal-failure:v1"
RESULT_STATE = "FUTURE_DATED_RECEIPTS_REJECTED_AND_SUPERSESSION_REQUIRED"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_temporal_failure_v2.py"
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/temporal-failure-v2.json"
)
TARGETS = (
    {
        "kind": "corrected_source_preflight_v2",
        "locator": "data/lol/v2/models/player/multileague-v3/source-preflight-v2.json",
        "raw_sha256": "e86f9c660278bef68d6abe873756050180603da7dfaed07edb5b605ce67f493b",
        "artifact_sha256": "bb2abaa30f657d15345796a076dfb19a7e116ed235030d29b82b625896b9c6ec",
        "declared_field": "built_at_utc",
    },
    {
        "kind": "future_protocol_v2",
        "locator": "data/lol/v2/models/player/multileague-v3/future-protocol-lock-v2.json",
        "raw_sha256": "106a36b0eafa069bee3d48c8e44c70a2e17efa087f4ea41653b812c707510127",
        "artifact_sha256": "49c91c8d1af525514b6125a453185bc41f16717ff60b011712fd7d8ce40425e2",
        "declared_field": "locked_at_utc",
    },
    {
        "kind": "capture_readiness_v1",
        "locator": "data/lol/v2/models/player/multileague-v3/capture-readiness-v1.json",
        "raw_sha256": "75a84655d952f868dcec7eb698e95aab76d105ba03e490c9c7a2186d46cbfd80",
        "artifact_sha256": "593f8abb0e4dae2774462ebe4195485990fe07db72ffc31e9622592c57113114",
        "declared_field": "locked_at_utc",
    },
)
AUTHORITY_KEYS = (
    "model_validation_authority",
    "rating_authority",
    "probability_authority",
    "recommendation_authority",
    "betting_authority",
)


class TemporalFailureError(RuntimeError):
    """The rejected receipt lineage or clock evidence changed."""


def _canonical_sha256(value: object) -> str:
    try:
        raw = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TemporalFailureError("temporal failure value is not canonical") from exc
    return hashlib.sha256(raw).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _time(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise TemporalFailureError(f"{label} must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise TemporalFailureError(f"{label} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _read_target(root: Path, spec: Mapping[str, str]) -> tuple[Path, dict[str, Any]]:
    path = root / spec["locator"]
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TemporalFailureError(f"cannot read rejected artifact: {path}") from exc
    if not isinstance(payload, dict):
        raise TemporalFailureError("rejected artifact is not an object")
    if hashlib.sha256(raw).hexdigest() != spec["raw_sha256"]:
        raise TemporalFailureError("rejected artifact raw bytes changed")
    if payload.get("artifact_sha256") != spec["artifact_sha256"]:
        raise TemporalFailureError("rejected artifact identity changed")
    return path, payload


def build_temporal_failure_receipt(
    *,
    observed_at: str,
    root: Path = ROOT,
) -> dict[str, Any]:
    observed = _time(observed_at, "observed_at")
    if observed > datetime.now(timezone.utc):
        raise TemporalFailureError("observed_at cannot be in the future")
    failures = []
    for spec in TARGETS:
        path, target = _read_target(root, spec)
        declared = _time(str(target.get(spec["declared_field"])), "declared time")
        try:
            created_epoch = path.stat().st_birthtime
        except AttributeError:
            created_epoch = path.stat().st_mtime
        created = datetime.fromtimestamp(created_epoch, timezone.utc)
        if not created < declared or not observed < declared:
            raise TemporalFailureError(
                "target does not reproduce the future-dated receipt failure"
            )
        failures.append(
            {
                "kind": spec["kind"],
                "locator": spec["locator"],
                "raw_sha256": spec["raw_sha256"],
                "artifact_sha256": spec["artifact_sha256"],
                "declared_time_field": spec["declared_field"],
                "declared_time_utc": declared.isoformat(),
                "filesystem_created_at_utc": created.isoformat(),
                "observed_existing_at_utc": observed.isoformat(),
                "failure": "artifact_existed_before_its_declared_lock_or_build_time",
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "observed_at_utc": observed.isoformat(),
        "failures": failures,
        "policy": {
            "artifacts_qualify_as_future_evidence": False,
            "numerical_diagnostics_may_be_rehearsed_again": True,
            "new_clock_checked_preflight_required": True,
            "new_hash_distinct_protocol_required": True,
            "capture_implementation_must_be_relocked": True,
            "future_outcomes_used": False,
        },
        "outcome_access": {
            "future_holdout_maps_present": 0,
            "future_holdout_targets_accessed": False,
        },
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": (
            "This failure receipt rejects future-dated timing claims. It preserves "
            "lineage only and grants no rating, probability, recommendation, or betting authority."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_temporal_failure_receipt(payload, root=root)


def validate_temporal_failure_receipt(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TemporalFailureError("temporal failure receipt must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise TemporalFailureError("temporal failure identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise TemporalFailureError("temporal failure canonical hash mismatch")
    observed = _time(str(value.get("observed_at_utc")), "observed_at_utc")
    failures = value.get("failures")
    if not isinstance(failures, list) or len(failures) != len(TARGETS):
        raise TemporalFailureError("temporal failure inventory changed")
    for failure, spec in zip(failures, TARGETS):
        if not isinstance(failure, Mapping):
            raise TemporalFailureError("temporal failure record is malformed")
        _path, target = _read_target(root, spec)
        target_declared = _time(
            str(target.get(spec["declared_field"])), "target declared time"
        )
        created = _time(
            str(failure.get("filesystem_created_at_utc")),
            "filesystem_created_at_utc",
        )
        if (
            failure.get("kind") != spec["kind"]
            or failure.get("raw_sha256") != spec["raw_sha256"]
            or failure.get("artifact_sha256") != spec["artifact_sha256"]
            or _time(str(failure.get("declared_time_utc")), "declared_time_utc")
            != target_declared
            or _time(
                str(failure.get("observed_existing_at_utc")),
                "observed_existing_at_utc",
            )
            != observed
            or not created < target_declared
            or not observed < target_declared
        ):
            raise TemporalFailureError("temporal failure no longer reproduces")
    policy = value.get("policy") or {}
    if (
        policy.get("artifacts_qualify_as_future_evidence") is not False
        or policy.get("future_outcomes_used") is not False
        or any(
            policy.get(name) is not True
            for name in (
                "numerical_diagnostics_may_be_rehearsed_again",
                "new_clock_checked_preflight_required",
                "new_hash_distinct_protocol_required",
                "capture_implementation_must_be_relocked",
            )
        )
    ):
        raise TemporalFailureError("temporal failure policy changed")
    if value.get("outcome_access") != {
        "future_holdout_maps_present": 0,
        "future_holdout_targets_accessed": False,
    }:
        raise TemporalFailureError("temporal failure claims future outcome access")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise TemporalFailureError("temporal failure receipt exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    digest = hashlib.sha256(raw).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"refusing to replace temporal failure receipt: {path}"
            ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_temporal_failure_receipt(observed_at=args.observed_at)
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "rejected_artifacts": len(payload["failures"]),
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "TemporalFailureError",
    "build_temporal_failure_receipt",
    "validate_temporal_failure_receipt",
    "write_no_clobber",
]
