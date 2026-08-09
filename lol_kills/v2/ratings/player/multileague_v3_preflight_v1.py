"""Reproducible failure receipt for the first future-protocol source package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from . import multileague_development as adapter
from .multileague_source_snapshot import validate_current_source_snapshot
from .multileague_v3_registry import validate_registered_future_protocol


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:multileague-rating-v3-source-preflight:v1"
RESULT_STATE = "SOURCE_SCHEMA_PREFLIGHT_FAILED"
EXPECTED_ADAPTER_ERROR = "selected multi-league map population is empty"
AUTHORITY_KEYS = (
    "model_validation_authority",
    "rating_authority",
    "probability_authority",
    "recommendation_authority",
    "betting_authority",
)
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/source-preflight-v1.json"
)


class SourcePreflightError(RuntimeError):
    """The expected v1 source-schema failure no longer reproduces exactly."""


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def build_source_preflight_failure(*, root: Path = ROOT) -> dict[str, Any]:
    source = validate_current_source_snapshot(root=root)
    protocol = validate_registered_future_protocol(root=root)
    files = source["files"]
    maps_path = root / files["maps"]["locator"]
    projected = pd.read_parquet(maps_path, columns=["playoffs"])
    dtype = str(projected["playoffs"].dtype)
    old_boundary = adapter.SEALED_FINAL_START
    adapter.SEALED_FINAL_START = pd.Timestamp(
        protocol["future_holdout"]["start_inclusive_source_time"]
    )
    observed_error: str | None = None
    try:
        adapter.load_multileague_development_input(
            expected_maps_sha256=files["maps"]["raw_sha256"],
            expected_players_sha256=files["players"]["raw_sha256"],
            root=root,
            maps_locator=files["maps"]["locator"],
            players_locator=files["players"]["locator"],
        )
    except adapter.MultiLeagueDevelopmentError as exc:
        observed_error = str(exc)
    finally:
        adapter.SEALED_FINAL_START = old_boundary
    if dtype != "float64" or observed_error != EXPECTED_ADAPTER_ERROR:
        raise SourcePreflightError("expected v1 schema failure did not reproduce")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "protocol_artifact_sha256": protocol["artifact_sha256"],
        "source_snapshot_manifest_sha256": source["manifest_canonical_sha256"],
        "diagnostic": {
            "field": "playoffs",
            "observed_dtype": dtype,
            "required_semantics": "boolean_or_nullable_boolean",
            "adapter_error": observed_error,
            "cause": "OE numeric 0/1 was not normalized before Parquet publication",
        },
        "outcome_access": {
            "future_holdout_maps_present": 0,
            "future_holdout_targets_accessed": False,
        },
        "remediation": {
            "policy": "supersede_before_future_boundary_without_reusing_any_future_outcome",
            "required": [
                "normalize playoffs 0/1 to nullable boolean",
                "rebuild full raw warehouse",
                "freeze a new joint maps-plus-players snapshot",
                "rerun adapter and posterior preflight",
                "issue a hash-distinct superseding protocol",
            ],
        },
        "authority": {
            "model_validation_authority": False,
            "rating_authority": False,
            "probability_authority": False,
            "recommendation_authority": False,
            "betting_authority": False,
        },
        "claim_ceiling": (
            "This is a source-schema failure receipt only. It grants no rating, "
            "probability, recommendation, or betting authority."
        ),
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_source_preflight_failure(payload, root=root)


def validate_source_preflight_failure(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SourcePreflightError("preflight receipt must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise SourcePreflightError("preflight receipt identity changed")
    declared = value.get("artifact_sha256")
    unsigned = dict(value)
    unsigned.pop("artifact_sha256", None)
    if not isinstance(declared, str) or declared != _canonical_sha256(unsigned):
        raise SourcePreflightError("preflight receipt canonical hash mismatch")
    diagnostic = value.get("diagnostic") or {}
    if diagnostic != {
        "field": "playoffs",
        "observed_dtype": "float64",
        "required_semantics": "boolean_or_nullable_boolean",
        "adapter_error": EXPECTED_ADAPTER_ERROR,
        "cause": "OE numeric 0/1 was not normalized before Parquet publication",
    }:
        raise SourcePreflightError("preflight diagnostic changed")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise SourcePreflightError("preflight receipt exceeds authority")
    if value.get("outcome_access") != {
        "future_holdout_maps_present": 0,
        "future_holdout_targets_accessed": False,
    }:
        raise SourcePreflightError("preflight receipt claims future outcome access")
    remediation = value.get("remediation") or {}
    if (
        remediation.get("policy")
        != "supersede_before_future_boundary_without_reusing_any_future_outcome"
        or remediation.get("required")
        != [
            "normalize playoffs 0/1 to nullable boolean",
            "rebuild full raw warehouse",
            "freeze a new joint maps-plus-players snapshot",
            "rerun adapter and posterior preflight",
            "issue a hash-distinct superseding protocol",
        ]
    ):
        raise SourcePreflightError("preflight remediation changed")
    source = validate_current_source_snapshot(root=root)
    protocol = validate_registered_future_protocol(root=root)
    if (
        value.get("protocol_artifact_sha256") != protocol.get("artifact_sha256")
        or value.get("source_snapshot_manifest_sha256")
        != source.get("manifest_canonical_sha256")
    ):
        raise SourcePreflightError("preflight source binding changed")
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
            raise FileExistsError(f"refusing to replace preflight receipt: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    payload = build_source_preflight_failure()
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "rating_authority": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_KEYS",
    "DEFAULT_OUTPUT",
    "EXPECTED_ADAPTER_ERROR",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SourcePreflightError",
    "build_source_preflight_failure",
    "validate_source_preflight_failure",
]
