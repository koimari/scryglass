"""Freeze the side-neutral capture-only supersession before future outcomes.

This changes only how an unknown pre-match blue/red assignment is captured:
both orientations are sealed before the scheduled series start and a later
public side observation may select one without refitting.  The candidate,
source snapshot, future boundary, support thresholds, evaluation metrics,
comparators, uncertainty rules, and opening rules remain unchanged.

The artifact is a non-authorizing candidate for independent review.  It does
not itself permit collection, outcome opening, ratings, probabilities, odds,
EV, recommendations, or betting.
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

from .multileague_v3_future_protocol import FUTURE_SEALED_START
from .multileague_v3_prediction_ledger import DEFAULT_REGISTRY, RECEIPT_PREFIX
from .multileague_v3_registry_v3 import (
    REGISTERED_PROTOCOL_ARTIFACT_SHA256,
    REGISTERED_PROTOCOL_LOCATOR,
    REGISTERED_PROTOCOL_RAW_SHA256,
    validate_registered_future_protocol_v3,
)
from .pre_side_rating_binding_v1 import (
    BINDING_PREFIX,
    BINDING_SCHEMA_VERSION,
    SOURCE_LOCATOR as BINDING_SOURCE_LOCATOR,
)
from .pre_side_rating_envelope_v1 import (
    ENVELOPE_PREFIX,
    ENVELOPE_SCHEMA_VERSION,
    SOURCE_LOCATOR as ENVELOPE_SOURCE_LOCATOR,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMA_VERSION = (
    "scryglass:multileague-rating-v3-side-neutral-protocol-supersession:v1"
)
RESULT_STATE = "SIDE_NEUTRAL_CAPTURE_PROTOCOL_LOCKED_EMPTY_AWAITING_REVIEW"
SOURCE_LOCATOR = (
    "lol_kills/v2/ratings/player/multileague_v3_side_neutral_protocol_v1.py"
)
DESIGN_LOCATOR = "docs/model-v2/side-neutral-prospective-capture-v1.md"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/models/player/multileague-v3/"
    "side-neutral-protocol-supersession-v1.json"
)
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
    ENVELOPE_SOURCE_LOCATOR,
    BINDING_SOURCE_LOCATOR,
    "lol_kills/v2/ratings/player/multileague_v3_future_protocol_v3.py",
    "lol_kills/v2/ratings/player/multileague_v3_registry_v3.py",
    REGISTERED_PROTOCOL_LOCATOR.as_posix(),
)
CLAIM_CEILING = (
    "Non-authorizing protocol-supersession candidate only. It freezes a "
    "side-neutral prospective capture sequence before the future boundary, "
    "without changing the model or evaluation. Independent hash review is "
    "still absent, collection under this revision is not authorized, outcomes "
    "remain sealed, and no rating, probability, odds, EV, recommendation, or "
    "betting authority is granted."
)


class SideNeutralProtocolError(RuntimeError):
    """The side-neutral protocol candidate is malformed or contaminated."""


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
        raise SideNeutralProtocolError("protocol value is not canonical") from exc


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
        raise SideNeutralProtocolError(f"{field} must be a timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SideNeutralProtocolError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise SideNeutralProtocolError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock_sample(clock: Callable[[], datetime], root: Path) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise SideNeutralProtocolError(
            "protocol clock must return a timezone-aware datetime"
        )
    observed = value.astimezone(timezone.utc)
    predecessor = validate_registered_future_protocol_v3(root=root)
    if observed <= _timestamp(predecessor["locked_at_utc"], "predecessor lock"):
        raise SideNeutralProtocolError("supersession must follow predecessor lock")
    if observed >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc):
        raise SideNeutralProtocolError("supersession must precede future boundary")
    return observed


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if not path.is_file() or path.is_symlink():
        raise SideNeutralProtocolError(f"bound source unavailable: {locator}")
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _count_json(root: Path, prefix: object) -> int:
    directory = root / Path(str(prefix))
    if not directory.exists():
        return 0
    if not directory.is_dir() or directory.is_symlink():
        raise SideNeutralProtocolError(f"collection prefix is not a directory: {prefix}")
    return sum(
        path.is_file() and not path.is_symlink()
        for path in directory.rglob("*.json")
    )


def _empty_state(root: Path) -> dict[str, Any]:
    registry = root / DEFAULT_REGISTRY
    counts = {
        "legacy_prediction_receipts": _count_json(root, RECEIPT_PREFIX),
        "pre_side_envelopes": _count_json(root, ENVELOPE_PREFIX),
        "side_bindings": _count_json(root, BINDING_PREFIX),
    }
    if registry.exists() or any(counts.values()):
        raise SideNeutralProtocolError(
            "capture artifacts already exist; supersession is no longer pre-collection"
        )
    return {
        **counts,
        "legacy_prediction_registry_present": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
    }


def build_side_neutral_protocol_lock(
    *,
    root: Path = ROOT,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    observed = _clock_sample(clock, root)
    predecessor = validate_registered_future_protocol_v3(root=root)
    empty_state = _empty_state(root)
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
            "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
            "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
            "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
            "locked_at_utc": predecessor["locked_at_utc"],
            "result_state": predecessor["result_state"],
        },
        "supersession": {
            "reason": "public_map_side_may_become_observable_only_during_champion_select",
            "capture_semantics_changed": True,
            "candidate_changed": False,
            "source_snapshot_changed": False,
            "future_boundary_changed": False,
            "support_stopping_rule_changed": False,
            "evaluation_rule_changed": False,
            "comparators_changed": False,
            "uncertainty_rule_changed": False,
            "opening_rule_changed": False,
            "future_outcomes_used_to_design_revision": False,
            "future_conditional_predictions_used_to_design_revision": False,
        },
        "locked_empty_state": empty_state,
        "capture_sequence": {
            "order": [
                "pre_side_envelope",
                "public_side_binding",
                "terminal_draft",
                "authoritative_actual_map_start",
                "complete_joint_bundle",
            ],
            "pre_side_envelope_schema_version": ENVELOPE_SCHEMA_VERSION,
            "side_binding_schema_version": BINDING_SCHEMA_VERSION,
            "scheduled_series_start_is_pre_side_cutoff": True,
            "both_side_conditionals_must_share_one_system_clock_sample": True,
            "team1_team2_schedule_order_is_not_side_authority": True,
            "public_side_binding_may_only_select_an_existing_conditional": True,
            "rating_refit_after_side_observation_permitted": False,
            "exact_ten_player_rosters_must_match_both_conditionals": True,
            "terminal_draft_must_bind_selected_rating_bytes": True,
            "actual_map_start_must_be_captured_separately": True,
            "pre_side_capture_before_side_binding": True,
            "side_binding_not_after_terminal_draft": True,
            "terminal_draft_strictly_before_actual_map_start": True,
            "incomplete_or_ambiguous_map_counts_as_eligible": False,
            "retrospective_backfill_permitted": False,
        },
        "unchanged_protocol": {
            "locked_candidate": predecessor["locked_candidate"],
            "source_snapshot": predecessor["source_snapshot"],
            "clock_corrected_source_preflight": predecessor[
                "clock_corrected_source_preflight"
            ],
            "future_holdout": predecessor["future_holdout"],
            "evaluation": predecessor["evaluation"],
        },
        "registration": {
            "candidate_code_pin_required": True,
            "candidate_code_pin_present": False,
            "independent_reviewer_digest_required": True,
            "independent_reviewer_digest_present": False,
            "collection_authorized": False,
            "self_authorizing": False,
        },
        "opening_authority": {
            "independent_protocol_review_present": False,
            "independent_opening_approval_present": False,
            "outcomes_must_remain_sealed": True,
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
    return validate_side_neutral_protocol_lock(payload, root=root)


def validate_side_neutral_protocol_lock(
    payload: Mapping[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise SideNeutralProtocolError("side-neutral protocol must be an object")
    value = dict(payload)
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "result_state"
    ) != RESULT_STATE:
        raise SideNeutralProtocolError("side-neutral protocol identity changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise SideNeutralProtocolError("side-neutral protocol hash changed")
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    predecessor = validate_registered_future_protocol_v3(root=root)
    if (
        locked <= _timestamp(predecessor["locked_at_utc"], "predecessor lock")
        or locked >= FUTURE_SEALED_START.replace(tzinfo=timezone.utc)
    ):
        raise SideNeutralProtocolError("side-neutral protocol timing changed")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise SideNeutralProtocolError("side-neutral clock attestation changed")
    if value.get("predecessor") != {
        "locator": REGISTERED_PROTOCOL_LOCATOR.as_posix(),
        "raw_sha256": REGISTERED_PROTOCOL_RAW_SHA256,
        "artifact_sha256": REGISTERED_PROTOCOL_ARTIFACT_SHA256,
        "locked_at_utc": predecessor["locked_at_utc"],
        "result_state": predecessor["result_state"],
    }:
        raise SideNeutralProtocolError("predecessor binding changed")
    supersession = value.get("supersession") or {}
    expected_false = (
        "candidate_changed",
        "source_snapshot_changed",
        "future_boundary_changed",
        "support_stopping_rule_changed",
        "evaluation_rule_changed",
        "comparators_changed",
        "uncertainty_rule_changed",
        "opening_rule_changed",
        "future_outcomes_used_to_design_revision",
        "future_conditional_predictions_used_to_design_revision",
    )
    if supersession.get("capture_semantics_changed") is not True or any(
        supersession.get(field) is not False for field in expected_false
    ):
        raise SideNeutralProtocolError("supersession scope changed")
    empty = value.get("locked_empty_state") or {}
    if empty != {
        "legacy_prediction_receipts": 0,
        "pre_side_envelopes": 0,
        "side_bindings": 0,
        "legacy_prediction_registry_present": False,
        "outcomes_present": False,
        "outcomes_accessed": False,
    }:
        raise SideNeutralProtocolError("locked empty state changed")
    sequence = value.get("capture_sequence") or {}
    if (
        sequence.get("order")
        != [
            "pre_side_envelope",
            "public_side_binding",
            "terminal_draft",
            "authoritative_actual_map_start",
            "complete_joint_bundle",
        ]
        or sequence.get("pre_side_envelope_schema_version")
        != ENVELOPE_SCHEMA_VERSION
        or sequence.get("side_binding_schema_version") != BINDING_SCHEMA_VERSION
        or sequence.get("team1_team2_schedule_order_is_not_side_authority")
        is not True
        or sequence.get(
            "public_side_binding_may_only_select_an_existing_conditional"
        )
        is not True
        or sequence.get("rating_refit_after_side_observation_permitted") is not False
        or sequence.get("incomplete_or_ambiguous_map_counts_as_eligible") is not False
        or sequence.get("retrospective_backfill_permitted") is not False
    ):
        raise SideNeutralProtocolError("capture sequence changed")
    unchanged = value.get("unchanged_protocol") or {}
    if unchanged != {
        "locked_candidate": predecessor["locked_candidate"],
        "source_snapshot": predecessor["source_snapshot"],
        "clock_corrected_source_preflight": predecessor[
            "clock_corrected_source_preflight"
        ],
        "future_holdout": predecessor["future_holdout"],
        "evaluation": predecessor["evaluation"],
    }:
        raise SideNeutralProtocolError("unchanged protocol content drifted")
    registration = value.get("registration") or {}
    if registration != {
        "candidate_code_pin_required": True,
        "candidate_code_pin_present": False,
        "independent_reviewer_digest_required": True,
        "independent_reviewer_digest_present": False,
        "collection_authorized": False,
        "self_authorizing": False,
    }:
        raise SideNeutralProtocolError("registration status was fabricated")
    opening = value.get("opening_authority") or {}
    if opening != {
        "independent_protocol_review_present": False,
        "independent_opening_approval_present": False,
        "outcomes_must_remain_sealed": True,
        "self_authorizing": False,
    }:
        raise SideNeutralProtocolError("opening authority was fabricated")
    source_locks = value.get("source_locks")
    if not isinstance(source_locks, list) or len(source_locks) != len(SOURCE_LOCKS):
        raise SideNeutralProtocolError("source-lock inventory changed")
    if [record.get("locator") for record in source_locks] != list(SOURCE_LOCKS):
        raise SideNeutralProtocolError("source-lock order changed")
    for record in source_locks:
        locator = str(record["locator"])
        if record != _source_record(root, locator):
            raise SideNeutralProtocolError(f"bound source drifted: {locator}")
    authority = value.get("authority") or {}
    if set(authority) != set(AUTHORITY_KEYS) or any(
        authority.get(name) is not False for name in AUTHORITY_KEYS
    ):
        raise SideNeutralProtocolError("side-neutral protocol exceeds authority")
    if any(item is not None for item in (value.get("decision_outputs") or {}).values()):
        raise SideNeutralProtocolError("protocol candidate contains decision outputs")
    if value.get("claim_ceiling") != CLAIM_CEILING:
        raise SideNeutralProtocolError("protocol claim ceiling changed")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise SideNeutralProtocolError(f"refusing to overwrite protocol: {path}")
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
            raise SideNeutralProtocolError(
                f"refusing to overwrite protocol: {path}"
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
    payload = build_side_neutral_protocol_lock(root=root)
    raw_sha256 = write_no_clobber(args.out, payload)
    print(
        json.dumps(
            {
                "output": str(args.out),
                "raw_sha256": raw_sha256,
                "artifact_sha256": payload["artifact_sha256"],
                "locked_at_utc": payload["locked_at_utc"],
                "result_state": payload["result_state"],
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
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SideNeutralProtocolError",
    "build_side_neutral_protocol_lock",
    "validate_side_neutral_protocol_lock",
    "write_no_clobber",
]
