"""Finalize one planned phase-two quote attempt after authoritative map start."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from . import betano_br_quote_adapter_v2 as quote_v2
from . import betano_br_quote_qualification_v1 as qualification
from . import phase_one_evaluation_v1 as evaluation
from . import phase_two_event_plan_v1 as event_plan
from . import phase_two_quote_attempt_v1 as attempt


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_attempt_completion_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-quote-attempt-completion:v1"
RESULT_STATE = "OUTCOME_FREE_PLANNED_QUOTE_ATTEMPT_COMPLETED_AFTER_MAP_START"
STATUSES = frozenset(
    {"QUALIFIED_QUOTE", "QUOTE_ATTEMPT_FAILED", "QUOTE_RESPONSE_TOO_LATE"}
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
    "Outcome-free completion identity for one prospectively planned quote "
    "attempt after authoritative map start. It establishes denominator status "
    "only and grants no quote registry, model accuracy, expected value, "
    "recommendation, transaction, or betting authority."
)


class PhaseTwoAttemptCompletionError(RuntimeError):
    """The plan, attempt output, map start, qualification, or completion failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseTwoAttemptCompletionError("completion is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoAttemptCompletionError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoAttemptCompletionError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PhaseTwoAttemptCompletionError("completion clock must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    return [
        evaluation._source_record(root, locator)
        for locator in (
            SOURCE_LOCATOR, event_plan.SOURCE_LOCATOR, attempt.SOURCE_LOCATOR,
            quote_v2.SOURCE_LOCATOR, qualification.SOURCE_LOCATOR,
        )
    ]


def _regular_json(root: Path, locator: str, label: str) -> tuple[bytes, dict[str, Any]]:
    raw = evaluation._read_regular(root, locator, label)
    return raw, evaluation._strict_object(raw, label)


def _plan(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    return attempt._plan(root=root, locator_value=locator_value, environment=environment)


def _binding(locator: str, raw: bytes, artifact: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "locator": locator,
        "raw_sha256": _sha256(raw),
        "artifact_sha256": artifact["artifact_sha256"],
    }


def build_phase_two_attempt_completion_v1(
    *, event_plan_locator: str, map_start_locator: str, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    plan_locator, plan_raw, plan = _plan(
        root=root, locator_value=event_plan_locator, environment=environment
    )
    start_locator, start_raw, map_start = qualification._map_start(
        root=root, locator_value=map_start_locator
    )
    for field in ("event_id", "series_id", "game_number", "league", "patch"):
        if plan["event"].get(field) != map_start["event"].get(field):
            raise PhaseTwoAttemptCompletionError(f"plan and map start differ: {field}")
    outputs = plan["reserved_outputs"]
    quote_path = root / outputs["quote_locator"]
    failure_path = root / outputs["failure_locator"]
    quote_exists = quote_path.is_file() and not quote_path.is_symlink()
    failure_exists = failure_path.is_file() and not failure_path.is_symlink()
    if quote_exists == failure_exists:
        raise PhaseTwoAttemptCompletionError(
            "plan requires exactly one persisted quote or failure"
        )
    quote_binding: dict[str, Any] | None = None
    failure_binding: dict[str, Any] | None = None
    qualification_binding: dict[str, Any] | None = None
    failure_code: str | None = None
    response_to_start_seconds: float | None = None
    if failure_exists:
        failure_raw, failure_object = _regular_json(
            root, outputs["failure_locator"], "quote-attempt failure"
        )
        failure = attempt.validate_quote_attempt_failure_v1(
            failure_object, root=root, environment=environment
        )
        if failure["event_plan_binding"]["artifact_sha256"] != plan["artifact_sha256"]:
            raise PhaseTwoAttemptCompletionError("failure and plan differ")
        failure_binding = _binding(outputs["failure_locator"], failure_raw, failure)
        failure_code = failure["failure"]["code"]
        status = "QUOTE_ATTEMPT_FAILED"
    else:
        quote_raw, quote_object = _regular_json(
            root, outputs["quote_locator"], "Betano v2 quote"
        )
        quote = quote_v2.validate_betano_map_winner_quote_v2(
            quote_object, root=root, environment=environment
        )
        if quote["event_plan_binding"]["artifact_sha256"] != plan["artifact_sha256"]:
            raise PhaseTwoAttemptCompletionError("quote and plan differ")
        quote_binding = _binding(outputs["quote_locator"], quote_raw, quote)
        response = _timestamp(
            quote["frozen_v1_transport_quote"]["transport"][
                "response_received_at_utc"
            ],
            "quote.response_received_at",
        )
        actual_start = _timestamp(
            map_start["event"]["actual_map_start_utc"], "actual_map_start"
        )
        response_to_start_seconds = (actual_start - response).total_seconds()
        if response_to_start_seconds < qualification.MINIMUM_RESPONSE_TO_START_SECONDS:
            status = "QUOTE_RESPONSE_TOO_LATE"
        else:
            qualification_path = root / outputs["qualification_locator"]
            if qualification_path.exists() or qualification_path.is_symlink():
                raise PhaseTwoAttemptCompletionError(
                    "qualification predates attempt completion or was replaced"
                )
            qualified = qualification.build_betano_quote_qualification_v1(
                quote_locator=outputs["quote_locator"],
                map_start_locator=start_locator,
                qualification_output_locator=outputs["qualification_locator"],
                root=root,
                environment=environment,
                clock=clock,
            )
            qualification.write_no_clobber(qualification_path, qualified)
            qualified_raw = qualification_path.read_bytes()
            qualification_binding = _binding(
                outputs["qualification_locator"], qualified_raw, qualified
            )
            status = "QUALIFIED_QUOTE"
    completed = _clock(clock)
    if completed < _timestamp(map_start["captured_at_utc"], "map_start.captured_at"):
        raise PhaseTwoAttemptCompletionError("completion predates map-start receipt")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "completed_at_utc": completed.isoformat(),
        "event": dict(plan["event"]),
        "event_plan_binding": _binding(plan_locator, plan_raw, plan),
        "map_start_binding": _binding(start_locator, start_raw, map_start),
        "status": status,
        "quote_binding": quote_binding,
        "failure_binding": failure_binding,
        "qualification_binding": qualification_binding,
        "failure_code": failure_code,
        "response_to_actual_start_seconds": response_to_start_seconds,
        "coverage": {
            "counts_in_otherwise_eligible_denominator": True,
            "counts_as_qualified_quote": status == "QUALIFIED_QUOTE",
            "quote_after_or_within_five_seconds_of_start": status
            == "QUOTE_RESPONSE_TOO_LATE",
            "event_outcome_present": False,
            "event_outcome_accessed": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_two_attempt_completion_v1(
        payload, root=root, environment=environment
    )


def validate_phase_two_attempt_completion_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoAttemptCompletionError("completion must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "completed_at_utc", "event",
        "event_plan_binding", "map_start_binding", "status", "quote_binding",
        "failure_binding", "qualification_binding", "failure_code",
        "response_to_actual_start_seconds", "coverage", "source_locks",
        "authority", "claim_ceiling", "artifact_sha256",
    }:
        raise PhaseTwoAttemptCompletionError("completion structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoAttemptCompletionError("completion hash changed")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("result_state") != RESULT_STATE:
        raise PhaseTwoAttemptCompletionError("completion identity changed")
    status = value.get("status")
    if status not in STATUSES:
        raise PhaseTwoAttemptCompletionError("completion status changed")
    completed = _timestamp(value.get("completed_at_utc"), "completed_at_utc")
    plan_binding = value.get("event_plan_binding") or {}
    plan_locator, plan_raw, plan = _plan(
        root=root, locator_value=plan_binding.get("locator"), environment=environment
    )
    if plan_binding != _binding(plan_locator, plan_raw, plan) or value.get("event") != plan["event"]:
        raise PhaseTwoAttemptCompletionError("completion plan binding changed")
    start_binding = value.get("map_start_binding") or {}
    start_locator, start_raw, map_start = qualification._map_start(
        root=root, locator_value=start_binding.get("locator")
    )
    if start_binding != _binding(start_locator, start_raw, map_start):
        raise PhaseTwoAttemptCompletionError("completion map-start binding changed")
    if completed < _timestamp(map_start["captured_at_utc"], "map_start.captured_at"):
        raise PhaseTwoAttemptCompletionError("completion predates map-start receipt")
    outputs = plan["reserved_outputs"]
    quote_binding = value.get("quote_binding")
    failure_binding = value.get("failure_binding")
    qualification_binding = value.get("qualification_binding")
    if status == "QUOTE_ATTEMPT_FAILED":
        failure_raw, failure_object = _regular_json(root, outputs["failure_locator"], "quote-attempt failure")
        failure = attempt.validate_quote_attempt_failure_v1(failure_object, root=root, environment=environment)
        if (
            quote_binding is not None
            or qualification_binding is not None
            or failure_binding != _binding(outputs["failure_locator"], failure_raw, failure)
            or value.get("failure_code") != failure["failure"]["code"]
            or value.get("response_to_actual_start_seconds") is not None
        ):
            raise PhaseTwoAttemptCompletionError("failed completion binding changed")
    else:
        quote_raw, quote_object = _regular_json(root, outputs["quote_locator"], "Betano v2 quote")
        quote = quote_v2.validate_betano_map_winner_quote_v2(quote_object, root=root, environment=environment)
        response = _timestamp(quote["frozen_v1_transport_quote"]["transport"]["response_received_at_utc"], "quote.response")
        actual_start = _timestamp(map_start["event"]["actual_map_start_utc"], "actual_start")
        lead = (actual_start - response).total_seconds()
        if (
            quote_binding != _binding(outputs["quote_locator"], quote_raw, quote)
            or failure_binding is not None
            or value.get("failure_code") is not None
            or value.get("response_to_actual_start_seconds") != lead
        ):
            raise PhaseTwoAttemptCompletionError("quote completion binding changed")
        if status == "QUALIFIED_QUOTE":
            qualified_raw, qualified_object = _regular_json(root, outputs["qualification_locator"], "quote qualification")
            try:
                qualified = qualification.validate_betano_quote_qualification_v1(
                    qualified_object, root=root, environment=environment
                )
            except Exception as exc:
                raise PhaseTwoAttemptCompletionError(
                    "qualified completion receipt is invalid"
                ) from exc
            if qualification_binding != _binding(outputs["qualification_locator"], qualified_raw, qualified) or lead < qualification.MINIMUM_RESPONSE_TO_START_SECONDS:
                raise PhaseTwoAttemptCompletionError("qualified completion changed")
        elif qualification_binding is not None or lead >= qualification.MINIMUM_RESPONSE_TO_START_SECONDS:
            raise PhaseTwoAttemptCompletionError("late quote completion changed")
    expected_coverage = {
        "counts_in_otherwise_eligible_denominator": True,
        "counts_as_qualified_quote": status == "QUALIFIED_QUOTE",
        "quote_after_or_within_five_seconds_of_start": status == "QUOTE_RESPONSE_TOO_LATE",
        "event_outcome_present": False,
        "event_outcome_accessed": False,
    }
    if value.get("coverage") != expected_coverage:
        raise PhaseTwoAttemptCompletionError("completion coverage changed")
    if value.get("source_locks") != _source_locks(root):
        raise PhaseTwoAttemptCompletionError("completion source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoAttemptCompletionError("completion exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoAttemptCompletionError(f"refusing to replace completion: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseTwoAttemptCompletionError(f"refusing to replace completion: {path}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


__all__ = [
    "SCHEMA_VERSION", "SOURCE_LOCATOR", "STATUSES",
    "PhaseTwoAttemptCompletionError", "build_phase_two_attempt_completion_v1",
    "validate_phase_two_attempt_completion_v1", "write_no_clobber",
]
