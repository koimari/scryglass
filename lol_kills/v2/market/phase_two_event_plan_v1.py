"""Persist the phase-two quote-attempt denominator before contacting Betano."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, Callable, Mapping

from . import event_probability_v2 as probability
from . import fast_event_uncertainty_v1 as fast_uncertainty
from . import phase_one_evaluation_v1 as evaluation


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_event_plan_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-event-quote-attempt-plan:v1"
RESULT_STATE = "OUTCOME_FREE_QUOTE_ATTEMPT_PLANNED_BEFORE_REQUEST"
OUTPUT_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/event-plans-v1"
)
QUOTE_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/betano-quotes-v2"
)
QUALIFICATION_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/qualified-betano-quotes-v1"
)
FAILURE_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/quote-failures-v1"
)
COMPLETION_PREFIX = PurePosixPath(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/attempt-completions-v1"
)
AUTHORITY = {
    "phase_two_denominator_identity_authority": False,
    "quote_identity_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Prospective immutable intent to attempt one quote for an otherwise "
    "eligible phase-two probability. It prevents success-only denominator "
    "selection but does not prove a request, response, quote, outcome, model "
    "accuracy, expected value, recommendation, transaction, or betting authority."
)


class PhaseTwoEventPlanError(RuntimeError):
    """The probability, chronology, locator, or immutable plan failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseTwoEventPlanError("event plan is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoEventPlanError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoEventPlanError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock(clock: Callable[[], datetime]) -> datetime:
    value = clock()
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise PhaseTwoEventPlanError("event-plan clock must be timezone-aware")
    return value.astimezone(timezone.utc)


def _output_locator(value: str, prefix: PurePosixPath, field: str) -> str:
    return evaluation._locator(value, prefix, field)


def _probability(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, probability.RECEIPT_PREFIX, "event_probability_locator"
    )
    raw = evaluation._read_regular(root, locator, "event probability")
    try:
        checked = probability.validate_event_probability_v2(
            evaluation._strict_object(raw, "event probability"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise PhaseTwoEventPlanError("event probability is invalid") from exc
    return locator, raw, checked


def _event_start(
    root: Path,
    receipt: Mapping[str, Any],
    environment: Mapping[str, str],
) -> datetime:
    fast_locator = receipt["input_binding"]["fast_uncertainty_locator"]
    _locator, _raw, fast = probability._uncertainty(
        root, fast_locator, environment
    )
    _rating_locator, _rating_raw, rating = fast_uncertainty._rating_artifact(
        root,
        fast["decomposition"]["rating_bootstrap_locator"],
        environment,
    )
    return _timestamp(rating["event"]["event_start_utc"], "event.start")


def _source_locks(root: Path) -> list[dict[str, Any]]:
    return [
        evaluation._source_record(root, locator)
        for locator in (SOURCE_LOCATOR, probability.SOURCE_LOCATOR)
    ]


def build_phase_two_event_plan_v1(
    *, event_probability_locator: str, quote_output_locator: str,
    qualification_output_locator: str, failure_output_locator: str,
    completion_output_locator: str,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    probability_locator, raw, receipt = _probability(
        root=root, locator_value=event_probability_locator, environment=environment
    )
    quote_locator = _output_locator(
        quote_output_locator, QUOTE_PREFIX, "quote_output_locator"
    )
    qualification_locator = _output_locator(
        qualification_output_locator,
        QUALIFICATION_PREFIX,
        "qualification_output_locator",
    )
    failure_locator = _output_locator(
        failure_output_locator, FAILURE_PREFIX, "failure_output_locator"
    )
    completion_locator = _output_locator(
        completion_output_locator, COMPLETION_PREFIX, "completion_output_locator"
    )
    planned = _clock(clock)
    if planned < _timestamp(receipt["captured_at_utc"], "probability.captured_at") or planned >= _event_start(root, receipt, environment):
        raise PhaseTwoEventPlanError("event plan was not created after probability and before event")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "planned_at_utc": planned.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_plan_builder",
            "observed_wall_clock_utc": planned.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": dict(receipt["event"]),
        "probability_binding": {
            "locator": probability_locator,
            "raw_sha256": _sha256(raw),
            "artifact_sha256": receipt["artifact_sha256"],
            "receipt_sha256": receipt["receipt_sha256"],
        },
        "reserved_outputs": {
            "quote_locator": quote_locator,
            "qualification_locator": qualification_locator,
            "failure_locator": failure_locator,
            "completion_locator": completion_locator,
            "replacement_permitted": False,
        },
        "denominator_contract": {
            "otherwise_eligible_map_recorded_before_quote_request": True,
            "plan_persists_if_request_or_quote_fails": True,
            "success_only_plan_creation_permitted": False,
            "retrospective_backfill_qualifies": False,
            "event_outcome_present": False,
            "event_outcome_accessed": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_two_event_plan_v1(
        payload, root=root, environment=environment
    )


def validate_phase_two_event_plan_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoEventPlanError("event plan must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "planned_at_utc", "clock_attestation",
        "event", "probability_binding", "reserved_outputs",
        "denominator_contract", "source_locks", "authority", "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseTwoEventPlanError("event plan structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoEventPlanError("event plan hash changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise PhaseTwoEventPlanError("event plan identity changed")
    planned = _timestamp(value.get("planned_at_utc"), "planned_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_plan_builder",
        "observed_wall_clock_utc": planned.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise PhaseTwoEventPlanError("event plan clock changed")
    binding = value.get("probability_binding") or {}
    locator, raw, receipt = _probability(
        root=root, locator_value=binding.get("locator"), environment=environment
    )
    if binding != {
        "locator": locator, "raw_sha256": _sha256(raw),
        "artifact_sha256": receipt["artifact_sha256"],
        "receipt_sha256": receipt["receipt_sha256"],
    } or value.get("event") != receipt["event"]:
        raise PhaseTwoEventPlanError("event-plan probability binding changed")
    outputs = value.get("reserved_outputs") or {}
    expected_outputs = {
        "quote_locator": _output_locator(outputs.get("quote_locator"), QUOTE_PREFIX, "quote_output_locator"),
        "qualification_locator": _output_locator(outputs.get("qualification_locator"), QUALIFICATION_PREFIX, "qualification_output_locator"),
        "failure_locator": _output_locator(
            outputs.get("failure_locator"), FAILURE_PREFIX, "failure_output_locator"
        ),
        "completion_locator": _output_locator(
            outputs.get("completion_locator"),
            COMPLETION_PREFIX,
            "completion_output_locator",
        ),
        "replacement_permitted": False,
    }
    if outputs != expected_outputs:
        raise PhaseTwoEventPlanError("reserved outputs changed")
    if planned < _timestamp(receipt["captured_at_utc"], "probability.captured_at") or planned >= _event_start(root, receipt, environment):
        raise PhaseTwoEventPlanError("event-plan chronology changed")
    if value.get("denominator_contract") != {
        "otherwise_eligible_map_recorded_before_quote_request": True,
        "plan_persists_if_request_or_quote_fails": True,
        "success_only_plan_creation_permitted": False,
        "retrospective_backfill_qualifies": False,
        "event_outcome_present": False,
        "event_outcome_accessed": False,
    }:
        raise PhaseTwoEventPlanError("denominator contract changed")
    if value.get("source_locks") != _source_locks(root):
        raise PhaseTwoEventPlanError("event-plan source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoEventPlanError("event plan exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoEventPlanError(f"refusing to replace event plan: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseTwoEventPlanError(f"refusing to replace event plan: {path}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


__all__ = [
    "OUTPUT_PREFIX", "SCHEMA_VERSION", "SOURCE_LOCATOR", "PhaseTwoEventPlanError",
    "build_phase_two_event_plan_v1", "validate_phase_two_event_plan_v1",
    "write_no_clobber",
]
