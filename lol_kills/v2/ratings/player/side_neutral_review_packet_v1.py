"""Build a non-authorizing packet for independent side-neutral review."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.v2.draft.terminal.side_neutral_prediction_v1 import PREDICTION_PREFIX
from lol_kills.v2.market.side_neutral_capture_bundle_v1 import BUNDLE_PREFIX

from .multileague_v3_prediction_ledger import DEFAULT_REGISTRY, RECEIPT_PREFIX
from .multileague_v3_side_neutral_protocol_registry_v2 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_side_neutral_protocol_v2,
)
from .pre_side_rating_binding_v1 import BINDING_PREFIX
from .pre_side_rating_envelope_v1 import ENVELOPE_PREFIX
from .side_neutral_collection_implementation_registry_v1 import (
    validate_registered_side_neutral_collection_implementation,
)
from .side_neutral_protocol_review_v1 import REVIEW_LOCATOR, SCHEMA_VERSION as REVIEW_SCHEMA


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = "scryglass:side-neutral-independent-review-packet:v1"
RESULT_STATE = "AWAITING_INDEPENDENT_HUMAN_REVIEW_NO_COLLECTION_AUTHORITY"
SOURCE_LOCATOR = "lol_kills/v2/ratings/player/side_neutral_review_packet_v1.py"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/review/multileague-v3/side-neutral-review-packet-v1.json"
)
AUTHORITY_KEYS = (
    "prospective_collection_authority",
    "outcome_opening_authority",
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "draft_validation_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "betting_authority",
)
CLAIM_CEILING = (
    "Independent-review request only. The packet inventories frozen bytes and "
    "review questions but is not a review, approval, collection authority, "
    "outcome-opening authority, rating, probability, odds, EV, recommendation, "
    "or betting authority."
)


class SideNeutralReviewPacketError(ValueError):
    """The review packet is stale, contaminated, or malformed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SideNeutralReviewPacketError("packet value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise SideNeutralReviewPacketError(f"{field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralReviewPacketError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SideNeutralReviewPacketError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SideNeutralReviewPacketError("packet clock must be timezone-aware")
    observed = value.astimezone(timezone.utc)
    if observed <= _timestamp(REGISTERED_PROTOCOL_LOCKED_AT_UTC, "protocol lock"):
        raise SideNeutralReviewPacketError("packet must follow frozen protocol")
    return observed


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise SideNeutralReviewPacketError("packet implementation is unavailable")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _count(root: Path, prefix: object) -> int:
    directory = root / Path(str(prefix))
    if not directory.exists():
        return 0
    if not directory.is_dir() or directory.is_symlink():
        raise SideNeutralReviewPacketError(f"invalid capture root: {prefix}")
    return sum(
        path.is_file() and not path.is_symlink()
        for path in directory.rglob("*.json")
    )


def _empty_inventory(root: Path) -> dict[str, Any]:
    value = {
        "legacy_prediction_receipts": _count(root, RECEIPT_PREFIX),
        "pre_side_envelopes": _count(root, ENVELOPE_PREFIX),
        "side_bindings": _count(root, BINDING_PREFIX),
        "side_neutral_terminal_drafts": _count(root, PREDICTION_PREFIX),
        "complete_bundles": _count(root, BUNDLE_PREFIX),
        "legacy_prediction_registry_present": (root / DEFAULT_REGISTRY).exists(),
    }
    if value["legacy_prediction_registry_present"] or any(
        count for key, count in value.items() if key != "legacy_prediction_registry_present"
    ):
        raise SideNeutralReviewPacketError(
            "review packet must be created before any prospective capture"
        )
    return {
        **value,
        "outcomes_present": False,
        "outcomes_accessed": False,
    }


def _review_questions() -> list[str]:
    return [
        "Verify every protocol and implementation byte hash against the supplied locators.",
        "Verify exact ten-player roster and patch bytes are shared by both pre-side orientations.",
        "Verify schedule, page, bookmaker, and UI order cannot become blue/red authority.",
        "Verify public side evidence can only select an existing rating receipt and cannot refit.",
        "Verify terminal Draft binds the selected rating bytes and complete legal pick/ban sequence.",
        "Verify pre-side, side, Draft, and actual-start timing is strictly prospective.",
        "Verify duplicate or ambiguous side bindings invalidate a map and retrospective admission is impossible.",
        "Verify recursively outcome-bearing inputs, overwrite attempts, and forged authority fail closed.",
        "Verify the model, source snapshot, future boundary, stopping rule, comparators, evaluation, and uncertainty are unchanged.",
        "Confirm no future outcome or conditional prediction was accessed while reviewing this revision.",
    ]


def build_side_neutral_review_packet(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    created = _clock_sample(clock)
    protocol = validate_registered_side_neutral_protocol_v2(root=root)
    admission = validate_registered_side_neutral_collection_implementation(root=root)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "created_at_utc": created.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": created.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "protocol": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "locked_at_utc": REGISTERED_PROTOCOL_LOCKED_AT_UTC,
            "source_locks": protocol["source_locks"],
        },
        "admission_implementation": admission["records"],
        "pre_review_inventory": _empty_inventory(root),
        "requested_review_record": {
            "schema_version": REVIEW_SCHEMA,
            "required_locator": REVIEW_LOCATOR.as_posix(),
            "reviewer_must_be_independent_human": True,
            "external_raw_sha256_required": True,
            "review_must_bind_protocol_and_all_listed_implementation_hashes": True,
            "review_effective_time_must_precede_every_eligible_pre_side_capture": True,
            "retrospective_capture_eligibility_permitted": False,
        },
        "review_questions": _review_questions(),
        "requested_authorization": {
            "prospective_outcome_free_collection_only": True,
            "outcome_opening": False,
            "rating_or_draft_deployment": False,
            "probability_odds_ev_or_recommendation": False,
            "betting": False,
        },
        "implementation": _source_record(root),
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_side_neutral_review_packet(payload, root=root)


def validate_side_neutral_review_packet(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SideNeutralReviewPacketError("packet must be an object")
    value = dict(payload)
    expected = {
        "schema_version",
        "result_state",
        "created_at_utc",
        "clock_attestation",
        "protocol",
        "admission_implementation",
        "pre_review_inventory",
        "requested_review_record",
        "review_questions",
        "requested_authorization",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected:
        raise SideNeutralReviewPacketError("packet structure changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "result_state"
    ) != RESULT_STATE:
        raise SideNeutralReviewPacketError("packet identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise SideNeutralReviewPacketError("packet hash changed")
    created = _timestamp(value.get("created_at_utc"), "created_at_utc")
    if created <= _timestamp(REGISTERED_PROTOCOL_LOCKED_AT_UTC, "protocol lock"):
        raise SideNeutralReviewPacketError("packet predates frozen protocol")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": created.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise SideNeutralReviewPacketError("packet clock changed")
    protocol = validate_registered_side_neutral_protocol_v2(root=root)
    if value.get("protocol") != {
        "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "locked_at_utc": REGISTERED_PROTOCOL_LOCKED_AT_UTC,
        "source_locks": protocol["source_locks"],
    }:
        raise SideNeutralReviewPacketError("packet protocol binding changed")
    admission = validate_registered_side_neutral_collection_implementation(root=root)
    if value.get("admission_implementation") != admission["records"]:
        raise SideNeutralReviewPacketError("packet admission hashes changed")
    expected_empty = {
        "legacy_prediction_receipts": 0,
        "pre_side_envelopes": 0,
        "side_bindings": 0,
        "side_neutral_terminal_drafts": 0,
        "complete_bundles": 0,
        "legacy_prediction_registry_present": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
    }
    if value.get("pre_review_inventory") != expected_empty:
        raise SideNeutralReviewPacketError("packet pre-review inventory changed")
    if value.get("requested_review_record") != {
        "schema_version": REVIEW_SCHEMA,
        "required_locator": REVIEW_LOCATOR.as_posix(),
        "reviewer_must_be_independent_human": True,
        "external_raw_sha256_required": True,
        "review_must_bind_protocol_and_all_listed_implementation_hashes": True,
        "review_effective_time_must_precede_every_eligible_pre_side_capture": True,
        "retrospective_capture_eligibility_permitted": False,
    }:
        raise SideNeutralReviewPacketError("packet review contract changed")
    if value.get("review_questions") != _review_questions():
        raise SideNeutralReviewPacketError("packet review questions changed")
    if value.get("requested_authorization") != {
        "prospective_outcome_free_collection_only": True,
        "outcome_opening": False,
        "rating_or_draft_deployment": False,
        "probability_odds_ev_or_recommendation": False,
        "betting": False,
    }:
        raise SideNeutralReviewPacketError("packet requested authority changed")
    if value.get("implementation") != _source_record(root):
        raise SideNeutralReviewPacketError("packet implementation changed")
    if value.get("authority") != {name: False for name in AUTHORITY_KEYS}:
        raise SideNeutralReviewPacketError("packet fabricated authority")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise SideNeutralReviewPacketError("packet claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SideNeutralReviewPacketError(f"refusing to overwrite packet: {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise SideNeutralReviewPacketError(
                f"refusing to overwrite packet: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return hashlib.sha256(raw).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        payload = build_side_neutral_review_packet(root=root)
        expected = root / DEFAULT_OUTPUT
        if args.out.resolve(strict=False) != expected.resolve(strict=False):
            raise SideNeutralReviewPacketError(
                f"output must be the review packet locator: {DEFAULT_OUTPUT}"
            )
        raw_sha256 = write_no_clobber(args.out, payload)
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc))
        return 2
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "result_state": payload["result_state"],
                "independent_review_present": False,
                "collection_authority": False,
                "betting_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SideNeutralReviewPacketError",
    "build_side_neutral_review_packet",
    "validate_side_neutral_review_packet",
    "write_no_clobber",
]
