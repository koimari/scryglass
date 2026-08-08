"""Complete the side-neutral prospective timing chain without admitting it.

The bundle joins a pre-side envelope, public side selection, terminal Draft,
and authoritative actual map start.  It verifies their exact identities and
the required temporal order.  Because the side-neutral protocol still lacks
independent registration, even a complete bundle remains ineligible evidence
and grants no rating, probability, odds, EV, recommendation, or betting
authority.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
from typing import Any, Mapping, Sequence

from lol_kills.v2.draft.terminal import future_prediction_ledger as draft_ledger
from lol_kills.v2.draft.terminal import side_neutral_prediction_v1 as neutral_draft
from lol_kills.v2.ratings.player.multileague_v3_side_neutral_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_side_neutral_protocol,
)


ROOT = Path(__file__).resolve().parents[3]
SCHEMA_VERSION = "scryglass:side-neutral-prospective-capture-bundle:v1"
RESULT_STATE = "COMPLETE_SIDE_NEUTRAL_CAPTURE_CHAIN_AWAITING_INDEPENDENT_REVIEW"
SOURCE_LOCATOR = "lol_kills/v2/market/side_neutral_capture_bundle_v1.py"
BUNDLE_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-one/side-neutral-bundles"
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
OUTCOME_KEYS = neutral_draft.OUTCOME_KEYS
AUTHORITY_KEYS = (
    "capture_protocol_authority",
    "model_validation_authority",
    "player_rating_authority",
    "team_rating_authority",
    "draft_validation_authority",
    "outcome_opening_authority",
    "calibration_authority",
    "probability_authority",
    "odds_authority",
    "expected_value_authority",
    "recommendation_authority",
    "stake_authority",
    "transaction_authority",
    "betting_authority",
)
CLAIM_CEILING = (
    "The outcome-free four-stage capture sequence is complete and internally "
    "validated, but the side-neutral protocol has not been independently "
    "registered. This bundle counts as zero eligible maps and grants no rating, "
    "Draft, probability, odds, EV, recommendation, stake, transaction, or "
    "betting authority."
)


class SideNeutralCaptureBundleError(ValueError):
    """The side-neutral capture chain failed closed."""


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
        raise SideNeutralCaptureBundleError("bundle value is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise SideNeutralCaptureBundleError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SideNeutralCaptureBundleError(f"{field} must be non-empty")
    return value.strip()


def _timestamp(value: Any, field: str) -> datetime:
    text = _nonempty(value, field)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralCaptureBundleError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SideNeutralCaptureBundleError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise SideNeutralCaptureBundleError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _strict_object(raw: bytes, field: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                SideNeutralCaptureBundleError(
                    f"non-finite JSON number in {field}: {token}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SideNeutralCaptureBundleError(f"{field} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise SideNeutralCaptureBundleError(f"{field} must be an object")
    return value


def _assert_no_outcomes(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]+", "", str(key).casefold())
            if normalized in OUTCOME_KEYS:
                raise SideNeutralCaptureBundleError(
                    f"event outcome field is forbidden: {path}.{key}"
                )
            _assert_no_outcomes(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_outcomes(item, f"{path}[{index}]")


def _source_record(root: Path) -> dict[str, Any]:
    path = root / SOURCE_LOCATOR
    if not path.is_file() or path.is_symlink():
        raise SideNeutralCaptureBundleError("bundle implementation is unavailable")
    return {
        "locator": SOURCE_LOCATOR,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _embedded(raw: bytes, value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "raw_sha256": _sha256_bytes(raw),
        "raw_base64": base64.b64encode(raw).decode("ascii"),
        "value": dict(value),
    }


def _decode_embedded(value: Any, field: str) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(value, Mapping) or set(value) != {
        "raw_sha256",
        "raw_base64",
        "value",
    }:
        raise SideNeutralCaptureBundleError(f"{field} embedding changed")
    try:
        raw = base64.b64decode(
            _nonempty(value.get("raw_base64"), f"{field}.raw_base64"),
            validate=True,
        )
    except (ValueError, binascii.Error) as exc:
        raise SideNeutralCaptureBundleError(f"{field} base64 is invalid") from exc
    if _sha256_bytes(raw) != _sha(value.get("raw_sha256"), f"{field}.raw_sha256"):
        raise SideNeutralCaptureBundleError(f"{field} raw hash changed")
    parsed = _strict_object(raw, field)
    if parsed != value.get("value"):
        raise SideNeutralCaptureBundleError(f"{field} value changed")
    return raw, parsed


def _authority_false() -> dict[str, bool]:
    return {name: False for name in AUTHORITY_KEYS}


def _event_identity(draft: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: draft["event"][field]
        for field in (
            "event_id",
            "series_id",
            "game_number",
            "league",
            "patch",
            "blue_organization_id",
            "blue_organization_name",
            "red_organization_id",
            "red_organization_name",
        )
    }


def _timing(
    draft: Mapping[str, Any], map_start: Mapping[str, Any]
) -> dict[str, Any]:
    binding = draft["side_binding"]["value"]
    envelope_value = binding["pre_side_envelope"]["value"]
    pre_side_capture = _timestamp(
        envelope_value["captured_at_utc"], "pre-side capture"
    )
    side_capture = _timestamp(binding["captured_at_utc"], "side capture")
    draft_capture = _timestamp(draft["captured_at_utc"], "draft capture")
    actual_start = _timestamp(
        map_start["event"]["actual_map_start_utc"], "actual map start"
    )
    if not pre_side_capture < side_capture < draft_capture < actual_start:
        raise SideNeutralCaptureBundleError(
            "capture order must be pre-side < side < Draft < actual map start"
        )
    side_available = _timestamp(
        binding["public_side_source"]["available_at_utc"],
        "side source available",
    )
    child = draft["terminal_draft_prediction"]["value"]
    draft_available = _timestamp(
        child["input_receipts"]["draft_metadata"]["value"]["source"][
            "available_at_utc"
        ],
        "draft source available",
    )
    if side_available > side_capture or draft_available >= draft_capture:
        raise SideNeutralCaptureBundleError("capture predates required source evidence")
    return {
        "pre_side_captured_at_utc": pre_side_capture.isoformat(),
        "side_binding_captured_at_utc": side_capture.isoformat(),
        "terminal_draft_captured_at_utc": draft_capture.isoformat(),
        "actual_map_start_utc": actual_start.isoformat(),
        "pre_side_before_side_binding": True,
        "side_binding_before_terminal_draft": True,
        "terminal_draft_before_actual_map_start": True,
        "source_availability_precedes_each_capture": True,
    }


def build_side_neutral_capture_bundle(
    *,
    side_neutral_draft_raw: bytes,
    map_start_receipt_raw: bytes,
    root: Path = ROOT,
) -> dict[str, Any]:
    draft_object = _strict_object(side_neutral_draft_raw, "side-neutral Draft")
    draft = neutral_draft.validate_side_neutral_draft_prediction(
        draft_object, root=root
    )
    start_object = _strict_object(map_start_receipt_raw, "map-start receipt")
    map_start = draft_ledger.validate_map_start_receipt(start_object, root=root)
    event = _event_identity(draft)
    for field in ("event_id", "series_id", "game_number", "league", "patch"):
        if map_start["event"].get(field) != event.get(field):
            raise SideNeutralCaptureBundleError(
                f"Draft and map start differ: {field}"
            )
    protocol = validate_registered_side_neutral_protocol(root=root)
    timing = _timing(draft, map_start)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "event": event,
        "protocol_candidate": {
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "locked_at_utc": protocol["locked_at_utc"],
            "repository_code_pin_valid": True,
            "independent_review_present": False,
        },
        "input_receipts": {
            "side_neutral_draft": _embedded(side_neutral_draft_raw, draft),
            "actual_map_start": _embedded(map_start_receipt_raw, map_start),
        },
        "timing": timing,
        "qualification": {
            "four_stage_capture_chain_complete": True,
            "exact_event_side_and_rating_bytes_bound": True,
            "actual_map_start_authority_present": True,
            "all_capture_timing_checks_passed": True,
            "side_binding_ambiguity_checked_across_registry": False,
            "side_neutral_protocol_independently_registered": False,
            "eligible_evaluation_map": False,
            "eligible_map_count_contribution": 0,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
        },
        "implementation": _source_record(root),
        "authority": _authority_false(),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_side_neutral_capture_bundle(payload, root=root)


def validate_side_neutral_capture_bundle(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SideNeutralCaptureBundleError("bundle must be an object")
    value = dict(payload)
    _assert_no_outcomes(value, "side_neutral_bundle")
    expected = {
        "schema_version",
        "result_state",
        "event",
        "protocol_candidate",
        "input_receipts",
        "timing",
        "qualification",
        "implementation",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }
    if set(value) != expected:
        raise SideNeutralCaptureBundleError("bundle structure changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "result_state"
    ) != RESULT_STATE:
        raise SideNeutralCaptureBundleError("bundle identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise SideNeutralCaptureBundleError("bundle hash changed")
    protocol = validate_registered_side_neutral_protocol(root=root)
    if value.get("protocol_candidate") != {
        "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "locked_at_utc": protocol["locked_at_utc"],
        "repository_code_pin_valid": True,
        "independent_review_present": False,
    }:
        raise SideNeutralCaptureBundleError("protocol candidate binding changed")
    inputs = value.get("input_receipts")
    if not isinstance(inputs, Mapping) or set(inputs) != {
        "side_neutral_draft",
        "actual_map_start",
    }:
        raise SideNeutralCaptureBundleError("bundle input structure changed")
    _draft_raw, draft_object = _decode_embedded(
        inputs["side_neutral_draft"], "side_neutral_draft"
    )
    draft = neutral_draft.validate_side_neutral_draft_prediction(
        draft_object, root=root
    )
    _start_raw, start_object = _decode_embedded(
        inputs["actual_map_start"], "actual_map_start"
    )
    map_start = draft_ledger.validate_map_start_receipt(start_object, root=root)
    expected_event = _event_identity(draft)
    if value.get("event") != expected_event:
        raise SideNeutralCaptureBundleError("bundle event changed")
    for field in ("event_id", "series_id", "game_number", "league", "patch"):
        if map_start["event"].get(field) != expected_event.get(field):
            raise SideNeutralCaptureBundleError(
                f"bundle map-start identity changed: {field}"
            )
    if value.get("timing") != _timing(draft, map_start):
        raise SideNeutralCaptureBundleError("bundle timing changed")
    expected_qualification = {
        "four_stage_capture_chain_complete": True,
        "exact_event_side_and_rating_bytes_bound": True,
        "actual_map_start_authority_present": True,
        "all_capture_timing_checks_passed": True,
        "side_binding_ambiguity_checked_across_registry": False,
        "side_neutral_protocol_independently_registered": False,
        "eligible_evaluation_map": False,
        "eligible_map_count_contribution": 0,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
    }
    if value.get("qualification") != expected_qualification:
        raise SideNeutralCaptureBundleError("bundle qualification changed")
    if value.get("implementation") != _source_record(root):
        raise SideNeutralCaptureBundleError("bundle implementation changed")
    if value.get("authority") != _authority_false() or value.get(
        "claim_ceiling"
    ) != CLAIM_CEILING:
        raise SideNeutralCaptureBundleError("bundle authority boundary changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SideNeutralCaptureBundleError(f"refusing to overwrite bundle: {path}")
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
            raise SideNeutralCaptureBundleError(
                f"refusing to overwrite bundle: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return _sha256_bytes(raw)


def _slug(value: str) -> str:
    result = SAFE_SLUG_RE.sub("-", value.casefold()).strip("-.")
    if not result:
        raise SideNeutralCaptureBundleError("event id cannot form a safe locator")
    return result[:160]


def bundle_locator(payload: Mapping[str, Any]) -> str:
    event = payload.get("event") or {}
    actual_start = _timestamp(
        (payload.get("timing") or {}).get("actual_map_start_utc"),
        "actual_map_start_utc",
    )
    return (
        BUNDLE_PREFIX
        / actual_start.date().isoformat()
        / f"{_slug(_nonempty(event.get('event_id'), 'event_id'))}-g{event.get('game_number')}.json"
    ).as_posix()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--side-neutral-draft", type=Path, required=True)
    parser.add_argument("--map-start", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        payload = build_side_neutral_capture_bundle(
            side_neutral_draft_raw=args.side_neutral_draft.read_bytes(),
            map_start_receipt_raw=args.map_start.read_bytes(),
            root=root,
        )
        expected = root / bundle_locator(payload)
        if args.out.resolve(strict=False) != expected.resolve(strict=False):
            raise SideNeutralCaptureBundleError(
                f"output must match bound locator: {expected.relative_to(root)}"
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
                "eligible_evaluation_map": False,
                "eligible_map_count_contribution": 0,
                "betting_authority": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BUNDLE_PREFIX",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SideNeutralCaptureBundleError",
    "build_side_neutral_capture_bundle",
    "bundle_locator",
    "validate_side_neutral_capture_bundle",
    "write_no_clobber",
]
