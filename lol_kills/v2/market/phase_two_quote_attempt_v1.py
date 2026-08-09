"""Consume one phase-two event plan into a quote or typed failure receipt."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping, Sequence

from . import betano_br_quote_adapter_v2 as quote_v2
from . import phase_one_evaluation_v1 as evaluation
from . import phase_two_event_plan_v1 as event_plan


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_quote_attempt_v1.py"
FAILURE_SCHEMA_VERSION = "scryglass:phase-two-quote-attempt-failure:v1"
FAILURE_RESULT_STATE = "PLANNED_QUOTE_ATTEMPT_FAILED_NON_AUTHORIZING"
FAILURE_CODES = frozenset(
    {
        "TRANSPORT_OR_SOURCE_UNAVAILABLE",
        "MARKET_NOT_OPEN_OR_MISSING",
        "SOURCE_IDENTITY_OR_BINDING_MISMATCH",
        "EXTRACTION_OR_REPLAY_FAILURE",
        "PRE_EVENT_TIMING_FAILURE",
        "INTERNAL_VALIDATION_FAILURE",
    }
)
AUTHORITY = {
    "quote_identity_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Typed system-clocked failure of one prospectively planned quote attempt. "
    "The receipt contains no credentials, cookies, request headers, free-form "
    "exception text, outcome, probability judgment, recommendation, or betting authority."
)


class PhaseTwoQuoteAttemptError(RuntimeError):
    """The plan, attempt consumption, quote output, or failure receipt failed."""


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PhaseTwoQuoteAttemptError("attempt receipt is not canonical") from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoQuoteAttemptError(f"{field} must be RFC-3339") from exc
    if parsed.tzinfo is None:
        raise PhaseTwoQuoteAttemptError(f"{field} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _clock(clock: Callable[[], datetime]) -> datetime:
    observed = clock()
    if not isinstance(observed, datetime) or observed.tzinfo is None:
        raise PhaseTwoQuoteAttemptError("attempt clock must be timezone-aware")
    return observed.astimezone(timezone.utc)


def _source_locks(root: Path) -> list[dict[str, Any]]:
    return [
        evaluation._source_record(root, locator)
        for locator in (SOURCE_LOCATOR, event_plan.SOURCE_LOCATOR, quote_v2.SOURCE_LOCATOR)
    ]


def _plan(
    *, root: Path, locator_value: str, environment: Mapping[str, str]
) -> tuple[str, bytes, dict[str, Any]]:
    locator = evaluation._locator(
        locator_value, event_plan.OUTPUT_PREFIX, "event_plan_locator"
    )
    raw = evaluation._read_regular(root, locator, "phase-two event plan")
    try:
        checked = event_plan.validate_phase_two_event_plan_v1(
            evaluation._strict_object(raw, "phase-two event plan"),
            root=root,
            environment=environment,
        )
    except Exception as exc:
        raise PhaseTwoQuoteAttemptError("phase-two event plan is invalid") from exc
    return locator, raw, checked


def _failure_code(exc: BaseException) -> str:
    messages: list[str] = []
    current: BaseException | None = exc
    for _ in range(5):
        if current is None:
            break
        messages.append(str(current).lower())
        current = current.__cause__
    joined = " | ".join(messages)
    if any(token in joined for token in ("market is missing", "market is not open", "market is missing or ambiguous", "exact map-winner market is missing")):
        return "MARKET_NOT_OPEN_OR_MISSING"
    if any(token in joined for token in ("identity", "participant", "binding mismatch", "event plan and probability differ")):
        return "SOURCE_IDENTITY_OR_BINDING_MISMATCH"
    if any(token in joined for token in ("extract", "replay", "initial-state", "response html")):
        return "EXTRACTION_OR_REPLAY_FAILURE"
    if any(token in joined for token in ("timing", "predates", "window exceeded", "too near", "clock")):
        return "PRE_EVENT_TIMING_FAILURE"
    if any(token in joined for token in ("transport", "http 200", "redirected", "content type", "response body")):
        return "TRANSPORT_OR_SOURCE_UNAVAILABLE"
    return "INTERNAL_VALIDATION_FAILURE"


def _failure_payload(
    *, plan_locator: str, plan_raw: bytes, plan: Mapping[str, Any],
    failure_code: str, failed_at: datetime, root: Path
) -> dict[str, Any]:
    if failure_code not in FAILURE_CODES:
        raise PhaseTwoQuoteAttemptError("failure code is not frozen")
    payload: dict[str, Any] = {
        "schema_version": FAILURE_SCHEMA_VERSION,
        "result_state": FAILURE_RESULT_STATE,
        "failed_at_utc": failed_at.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_after_failed_attempt",
            "observed_wall_clock_utc": failed_at.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "event": dict(plan["event"]),
        "event_plan_binding": {
            "locator": plan_locator,
            "raw_sha256": _sha256(plan_raw),
            "artifact_sha256": plan["artifact_sha256"],
            "planned_at_utc": plan["planned_at_utc"],
        },
        "failure": {
            "code": failure_code,
            "free_form_exception_text_persisted": False,
            "request_headers_cookies_or_credentials_persisted": False,
            "counts_in_quote_coverage_denominator": True,
            "counts_as_qualified_quote": False,
            "retrospective_retry_replaces_failure": False,
        },
        "source_locks": _source_locks(root),
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return payload


def validate_quote_attempt_failure_v1(
    payload: Mapping[str, Any], *, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoQuoteAttemptError("failure receipt must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version", "result_state", "failed_at_utc", "clock_attestation",
        "event", "event_plan_binding", "failure", "source_locks", "authority",
        "claim_ceiling", "artifact_sha256",
    }:
        raise PhaseTwoQuoteAttemptError("failure receipt structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoQuoteAttemptError("failure receipt hash changed")
    if value.get("schema_version") != FAILURE_SCHEMA_VERSION or value.get("result_state") != FAILURE_RESULT_STATE:
        raise PhaseTwoQuoteAttemptError("failure receipt identity changed")
    failed = _timestamp(value.get("failed_at_utc"), "failed_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_after_failed_attempt",
        "observed_wall_clock_utc": failed.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise PhaseTwoQuoteAttemptError("failure clock changed")
    binding = value.get("event_plan_binding") or {}
    locator, raw, plan = _plan(
        root=root, locator_value=binding.get("locator"), environment=environment
    )
    if binding != {
        "locator": locator, "raw_sha256": _sha256(raw),
        "artifact_sha256": plan["artifact_sha256"],
        "planned_at_utc": plan["planned_at_utc"],
    } or value.get("event") != plan["event"]:
        raise PhaseTwoQuoteAttemptError("failure event-plan binding changed")
    if failed < _timestamp(plan["planned_at_utc"], "plan.planned_at"):
        raise PhaseTwoQuoteAttemptError("failure predates the attempt plan")
    failure = value.get("failure") or {}
    if failure != {
        "code": failure.get("code"),
        "free_form_exception_text_persisted": False,
        "request_headers_cookies_or_credentials_persisted": False,
        "counts_in_quote_coverage_denominator": True,
        "counts_as_qualified_quote": False,
        "retrospective_retry_replaces_failure": False,
    } or failure.get("code") not in FAILURE_CODES:
        raise PhaseTwoQuoteAttemptError("failure classification changed")
    if value.get("source_locks") != _source_locks(root):
        raise PhaseTwoQuoteAttemptError("failure source lock changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoQuoteAttemptError("failure receipt exceeds authority")
    return value


def _write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoQuoteAttemptError(f"refusing to replace attempt output: {path}")
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseTwoQuoteAttemptError(f"refusing to replace attempt output: {path}") from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return _sha256(raw)


def run_planned_quote_attempt_v1(
    *, event_plan_locator: str, request_url: str, betano_event_id: str,
    map_number: int, participant_bindings: Sequence[Mapping[str, str]],
    fetcher: quote_v2.transport.PublicDocumentFetcher, root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    monotonic_ns: Callable[[], int] = quote_v2.transport.time.monotonic_ns,
) -> dict[str, Any]:
    plan_locator, plan_raw, plan = _plan(
        root=root, locator_value=event_plan_locator, environment=environment
    )
    outputs = plan["reserved_outputs"]
    quote_path = root / outputs["quote_locator"]
    failure_path = root / outputs["failure_locator"]
    if any(path.exists() or path.is_symlink() for path in (quote_path, failure_path)):
        raise PhaseTwoQuoteAttemptError("planned quote attempt was already consumed")
    try:
        quote = quote_v2.capture_betano_map_winner_quote_v2(
            event_plan_locator=plan_locator,
            event_probability_locator=plan["probability_binding"]["locator"],
            request_url=request_url,
            betano_event_id=betano_event_id,
            map_number=map_number,
            participant_bindings=participant_bindings,
            fetcher=fetcher,
            root=root,
            environment=environment,
            clock=clock,
            monotonic_ns=monotonic_ns,
        )
    except Exception as exc:
        failed = _clock(clock)
        failure = _failure_payload(
            plan_locator=plan_locator, plan_raw=plan_raw, plan=plan,
            failure_code=_failure_code(exc), failed_at=failed, root=root
        )
        validate_quote_attempt_failure_v1(
            failure, root=root, environment=environment
        )
        digest = _write_no_clobber(failure_path, failure)
        return {
            "status": "failed_persisted",
            "failure_locator": outputs["failure_locator"],
            "failure_raw_sha256": digest,
            "failure_artifact_sha256": failure["artifact_sha256"],
            "failure_code": failure["failure"]["code"],
            "quote_identity_authority": False,
            "betting_authority": False,
        }
    digest = quote_v2.write_no_clobber(quote_path, quote)
    return {
        "status": "quote_persisted",
        "quote_locator": outputs["quote_locator"],
        "quote_raw_sha256": digest,
        "quote_artifact_sha256": quote["artifact_sha256"],
        "quote_identity_authority": False,
        "betting_authority": False,
    }


__all__ = [
    "FAILURE_CODES", "FAILURE_SCHEMA_VERSION", "SOURCE_LOCATOR",
    "PhaseTwoQuoteAttemptError", "run_planned_quote_attempt_v1",
    "validate_quote_attempt_failure_v1",
]
