"""Freeze the complete side-neutral capture and admission contract.

Version 1 was locked before the future boundary after the pre-side envelope
and side selector existed.  This successor adds the exact terminal-Draft
adapter, four-stage bundle, and independent-review admission requirements.  It
does not change the rating model, source snapshot, future boundary, stopping
rule, evaluation, comparators, uncertainty, or opening rules.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from lol_kills.v2.draft.terminal.side_neutral_prediction_v1 import (
    PREDICTION_PREFIX as SIDE_NEUTRAL_DRAFT_PREFIX,
    SCHEMA_VERSION as SIDE_NEUTRAL_DRAFT_SCHEMA,
    SOURCE_LOCATOR as SIDE_NEUTRAL_DRAFT_SOURCE,
)
from lol_kills.v2.market.side_neutral_capture_bundle_v1 import (
    BUNDLE_PREFIX,
    SCHEMA_VERSION as BUNDLE_SCHEMA,
    SOURCE_LOCATOR as BUNDLE_SOURCE,
)

from .multileague_v3_future_protocol import FUTURE_SEALED_START
from .multileague_v3_prediction_ledger import DEFAULT_REGISTRY, RECEIPT_PREFIX
from .multileague_v3_side_neutral_protocol_registry_v1 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256 as V1_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR as V1_LOCATOR,
    REGISTERED_PROTOCOL_LOCKED_AT_UTC as V1_LOCKED_AT_UTC,
    REGISTERED_PROTOCOL_RAW_SHA256 as V1_RAW_SHA256,
    validate_registered_side_neutral_protocol as validate_v1,
)
from .pre_side_rating_binding_v1 import (
    BINDING_PREFIX,
    BINDING_SCHEMA_VERSION,
    SOURCE_LOCATOR as BINDING_SOURCE,
)
from .pre_side_rating_envelope_v1 import (
    ENVELOPE_PREFIX,
    ENVELOPE_SCHEMA_VERSION,
    SOURCE_LOCATOR as ENVELOPE_SOURCE,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = (
    "scryglass:multileague-rating-v3-side-neutral-protocol-supersession:v2"
)
RESULT_STATE = "COMPLETE_SIDE_NEUTRAL_CAPTURE_PROTOCOL_LOCKED_EMPTY_AWAITING_REVIEW"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_side_neutral_protocol_v2.py"
)
DESIGN_LOCATOR = "docs/model-v2/side-neutral-prospective-capture-v1.md"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/"
    "side-neutral-protocol-supersession-v2.json"
)
INDEPENDENT_REVIEW_ENV = "SCRYGLASS_PRIVATE_SIDE_NEUTRAL_PROTOCOL_REVIEW_SHA256"
AUTHORITY_KEYS = (
    "capture_protocol_authority",
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
SOURCE_LOCKS = (
    SOURCE_LOCATOR,
    DESIGN_LOCATOR,
    ENVELOPE_SOURCE,
    BINDING_SOURCE,
    SIDE_NEUTRAL_DRAFT_SOURCE,
    BUNDLE_SOURCE,
    "lol_kills/v2/ratings/player/multileague_v3_side_neutral_protocol_v1.py",
    "lol_kills/v2/ratings/player/multileague_v3_side_neutral_protocol_registry_v1.py",
    V1_LOCATOR.as_posix(),
)
CLAIM_CEILING = (
    "Complete side-neutral capture-protocol candidate only. Repository hashes "
    "are frozen before the future boundary, but an external independent review "
    "record is absent. Collection admission, outcome opening, ratings, Draft "
    "authority, probability, odds, EV, recommendations, and betting remain "
    "unauthorized."
)


class SideNeutralProtocolV2Error(RuntimeError):
    """The complete side-neutral protocol candidate failed closed."""


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
        raise SideNeutralProtocolV2Error("protocol value is not canonical") from exc


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
        raise SideNeutralProtocolV2Error(f"{field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralProtocolV2Error(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SideNeutralProtocolV2Error(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], root: Path) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SideNeutralProtocolV2Error("protocol clock must be timezone-aware")
    observed = value.astimezone(timezone.utc)
    predecessor = validate_v1(root=root)
    if observed <= _timestamp(predecessor["locked_at_utc"], "v1 lock"):
        raise SideNeutralProtocolV2Error("v2 must follow v1 lock")
    if observed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise SideNeutralProtocolV2Error("v2 must precede future boundary")
    return observed


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file() or path.is_symlink():
        raise SideNeutralProtocolV2Error(f"bound source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _count(root: Path, prefix: object) -> int:
    directory = root / Path(str(prefix))
    if not directory.exists():
        return 0
    if not directory.is_dir() or directory.is_symlink():
        raise SideNeutralProtocolV2Error(f"invalid capture root: {prefix}")
    return sum(
        path.is_file() and not path.is_symlink()
        for path in directory.rglob("*.json")
    )


def _locked_empty_state(root: Path) -> dict[str, Any]:
    value = {
        "legacy_prediction_receipts": _count(root, RECEIPT_PREFIX),
        "pre_side_envelopes": _count(root, ENVELOPE_PREFIX),
        "side_bindings": _count(root, BINDING_PREFIX),
        "side_neutral_terminal_drafts": _count(root, SIDE_NEUTRAL_DRAFT_PREFIX),
        "complete_bundles": _count(root, BUNDLE_PREFIX),
        "legacy_prediction_registry_present": (root / DEFAULT_REGISTRY).exists(),
        "outcomes_present": False,
        "outcomes_accessed": False,
    }
    if value["legacy_prediction_registry_present"] or any(
        value[field]
        for field in (
            "legacy_prediction_receipts",
            "pre_side_envelopes",
            "side_bindings",
            "side_neutral_terminal_drafts",
            "complete_bundles",
        )
    ):
        raise SideNeutralProtocolV2Error(
            "capture artifacts already exist; v2 is not pre-collection"
        )
    return value


def _capture_contract() -> dict[str, Any]:
    return {
        "order": [
            "pre_side_envelope",
            "public_side_binding",
            "terminal_draft",
            "authoritative_actual_map_start",
            "complete_joint_bundle",
            "independent_ledger_admission",
        ],
        "pre_side_envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
        "side_binding_schema_version": BINDING_SCHEMA_VERSION,
        "side_neutral_terminal_draft_schema_version": SIDE_NEUTRAL_DRAFT_SCHEMA,
        "complete_bundle_schema_version": BUNDLE_SCHEMA,
        "both_orientations_same_model_roster_patch_and_clock_required": True,
        "schedule_order_page_order_and_bookmaker_order_are_not_side_authority": True,
        "public_side_source_must_select_one_existing_conditional": True,
        "rating_refit_after_side_observation_permitted": False,
        "terminal_draft_must_bind_selected_rating_bytes": True,
        "strict_timing_order_required": True,
        "ambiguous_or_duplicate_side_bindings_invalidate_map": True,
        "incomplete_bundle_counts_as_eligible": False,
        "retrospective_backfill_permitted": False,
        "outcome_fields_permitted": False,
        "all_persistent_writes_no_clobber": True,
    }


def _review_contract() -> dict[str, Any]:
    return {
        "external_digest_environment_variable": INDEPENDENT_REVIEW_ENV,
        "review_record_must_bind_protocol_raw_and_artifact_sha256": True,
        "review_record_must_bind_all_capture_implementation_sha256": True,
        "reviewer_must_be_independent_from_implementation": True,
        "review_must_precede_first_eligible_pre_side_capture": True,
        "review_must_confirm_future_outcomes_and_predictions_unopened": True,
        "review_may_authorize_prospective_collection_only": True,
        "review_may_authorize_outcome_opening": False,
        "review_may_authorize_ratings_probabilities_or_betting": False,
        "self_review_permitted": False,
    }


def build_side_neutral_protocol_v2(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock, root)
    predecessor = validate_v1(root=root)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": observed.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": observed.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "predecessor": {
            "locator": V1_LOCATOR.as_posix(),
            "raw_sha256": V1_RAW_SHA256,
            "artifact_sha256": V1_ARTIFACT_SHA256,
            "locked_at_utc": V1_LOCKED_AT_UTC,
            "result_state": predecessor["result_state"],
        },
        "supersession": {
            "terminal_draft_and_bundle_implementation_added": True,
            "capture_semantics_from_v1_changed": False,
            "candidate_changed": False,
            "source_snapshot_changed": False,
            "future_boundary_changed": False,
            "support_stopping_rule_changed": False,
            "evaluation_rule_changed": False,
            "comparators_changed": False,
            "uncertainty_rule_changed": False,
            "opening_rule_changed": False,
            "future_outcomes_used": False,
            "future_predictions_used": False,
        },
        "locked_empty_state": _locked_empty_state(root),
        "capture_contract": _capture_contract(),
        "independent_review_contract": _review_contract(),
        "unchanged_protocol": predecessor["unchanged_protocol"],
        "registration": {
            "repository_code_pin_present": False,
            "independent_review_present": False,
            "prospective_collection_authorized": False,
            "self_authorizing": False,
        },
        "source_locks": [_source_record(root, locator) for locator in SOURCE_LOCKS],
        "decision_outputs": {
            "sealed_evaluation": None,
            "player_rating_authority": None,
            "team_rating_authority": None,
            "draft_score_authority": None,
            "probability": None,
            "odds": None,
            "expected_value": None,
            "recommendation": None,
        },
        "authority": {name: False for name in AUTHORITY_KEYS},
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_side_neutral_protocol_v2(payload, root=root)


def validate_side_neutral_protocol_v2(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SideNeutralProtocolV2Error("v2 protocol must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "result_state"
    ) != RESULT_STATE:
        raise SideNeutralProtocolV2Error("v2 identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise SideNeutralProtocolV2Error("v2 hash changed")
    predecessor = validate_v1(root=root)
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    if (
        locked <= _timestamp(predecessor["locked_at_utc"], "v1 lock")
        or locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    ):
        raise SideNeutralProtocolV2Error("v2 timing changed")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise SideNeutralProtocolV2Error("v2 clock changed")
    if value.get("predecessor") != {
        "locator": V1_LOCATOR.as_posix(),
        "raw_sha256": V1_RAW_SHA256,
        "artifact_sha256": V1_ARTIFACT_SHA256,
        "locked_at_utc": V1_LOCKED_AT_UTC,
        "result_state": predecessor["result_state"],
    }:
        raise SideNeutralProtocolV2Error("v1 binding changed")
    supersession = value.get("supersession") or {}
    if supersession.get("terminal_draft_and_bundle_implementation_added") is not True:
        raise SideNeutralProtocolV2Error("v2 implementation scope changed")
    for field in (
        "capture_semantics_from_v1_changed",
        "candidate_changed",
        "source_snapshot_changed",
        "future_boundary_changed",
        "support_stopping_rule_changed",
        "evaluation_rule_changed",
        "comparators_changed",
        "uncertainty_rule_changed",
        "opening_rule_changed",
        "future_outcomes_used",
        "future_predictions_used",
    ):
        if supersession.get(field) is not False:
            raise SideNeutralProtocolV2Error("v2 supersession scope changed")
    if value.get("locked_empty_state") != {
        "legacy_prediction_receipts": 0,
        "pre_side_envelopes": 0,
        "side_bindings": 0,
        "side_neutral_terminal_drafts": 0,
        "complete_bundles": 0,
        "legacy_prediction_registry_present": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
    }:
        raise SideNeutralProtocolV2Error("v2 empty state changed")
    if value.get("capture_contract") != _capture_contract():
        raise SideNeutralProtocolV2Error("capture contract changed")
    if value.get("independent_review_contract") != _review_contract():
        raise SideNeutralProtocolV2Error("independent review contract changed")
    if value.get("unchanged_protocol") != predecessor["unchanged_protocol"]:
        raise SideNeutralProtocolV2Error("unchanged rating protocol drifted")
    if value.get("registration") != {
        "repository_code_pin_present": False,
        "independent_review_present": False,
        "prospective_collection_authorized": False,
        "self_authorizing": False,
    }:
        raise SideNeutralProtocolV2Error("registration was fabricated")
    records = value.get("source_locks")
    if not isinstance(records, list) or len(records) != len(SOURCE_LOCKS):
        raise SideNeutralProtocolV2Error("source inventory changed")
    if [record.get("locator") for record in records] != list(SOURCE_LOCKS):
        raise SideNeutralProtocolV2Error("source order changed")
    for record in records:
        if record != _source_record(root, str(record["locator"])):
            raise SideNeutralProtocolV2Error(
                f"bound source drifted: {record.get('locator')}"
            )
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(authority.values()):
        raise SideNeutralProtocolV2Error("v2 exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise SideNeutralProtocolV2Error("v2 contains decision outputs")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise SideNeutralProtocolV2Error("v2 claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SideNeutralProtocolV2Error(f"refusing to overwrite v2: {path}")
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
            raise SideNeutralProtocolV2Error(
                f"refusing to overwrite v2: {path}"
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
    payload = build_side_neutral_protocol_v2(root=root)
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "result_state": payload["result_state"],
                "independent_review_present": False,
                "collection_authorized": False,
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
    "INDEPENDENT_REVIEW_ENV",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SideNeutralProtocolV2Error",
    "build_side_neutral_protocol_v2",
    "validate_side_neutral_protocol_v2",
    "write_no_clobber",
]
