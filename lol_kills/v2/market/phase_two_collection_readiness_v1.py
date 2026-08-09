"""Freeze the complete outcome-free phase-two collection path before opening."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

from . import betano_br_quote_adapter_v2 as quote_v2
from . import betano_br_quote_qualification_v1 as quote_qualification
from . import betano_br_quote_registry_v2 as quote_registry
from . import event_probability_registry_v2 as probability_registry
from . import event_probability_v2 as probability_v2
from . import event_rating_bootstrap_v1 as rating_bootstrap
from . import fast_event_uncertainty_v1 as fast_uncertainty
from . import phase_one_evaluation_v1 as evaluation
from . import phase_two_opening_v1 as opening
from . import phase_two_event_plan_v1 as event_plan
from . import phase_two_quote_attempt_v1 as quote_attempt
from . import phase_two_attempt_completion_v1 as attempt_completion
from . import phase_two_stopping_snapshot_v1 as stopping_snapshot
from . import phase_two_stopping_snapshot_registry_v1 as stopping_registry
from .betano_br_quote_adapter_registry_v1 import (
    load_registered_betano_quote_adapter_v1,
)
from .betano_terms_authority_v1 import (
    REGISTRY_LOCATOR as TERMS_REGISTRY_LOCATOR,
    load_pinned_betano_terms_authority_v1,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE_LOCATOR = "lol_kills/v2/market/phase_two_collection_readiness_v1.py"
SCHEMA_VERSION = "scryglass:phase-two-collection-readiness:v1"
RESULT_STATE = "OUTCOME_FREE_PHASE_TWO_COLLECTION_IMPLEMENTATION_FROZEN_BEFORE_OPENING"
DEFAULT_OUTPUT = Path(
    "data/lol/v2/evaluation/match-winner-market-v1/phase-two/"
    "collection-readiness-v1.json"
)
AUTHORITY = {
    "phase_two_opening_authority": False,
    "event_probability_identity_authority": False,
    "quote_identity_authority": False,
    "probability_authority": False,
    "expected_value_authority": False,
    "recommendation_authority": False,
    "transaction_authority": False,
    "betting_authority": False,
}
CLAIM_CEILING = (
    "Implementation and empty-state freeze before phase-two opening only. It "
    "does not open collection, register an event probability or quote, access "
    "outcomes, authorize expected value, recommend a transaction, or permit betting."
)


class PhaseTwoCollectionReadinessError(RuntimeError):
    """A dependency, source lock, signature, or empty state drifted."""


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
        raise PhaseTwoCollectionReadinessError(
            "phase-two readiness is not canonical"
        ) from exc


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp(value: Any, field: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise PhaseTwoCollectionReadinessError(
            f"{field} must be RFC-3339"
        ) from exc
    if parsed.tzinfo is None:
        raise PhaseTwoCollectionReadinessError(
            f"{field} must include a timezone"
        )
    return parsed.astimezone(timezone.utc)


def _source_record(root: Path, locator: str) -> dict[str, Any]:
    path = root / locator
    if path.is_symlink() or not path.is_file():
        raise PhaseTwoCollectionReadinessError(
            f"phase-two source is unavailable: {locator}"
        )
    return {
        "locator": locator,
        "bytes": path.stat().st_size,
        "raw_sha256": _sha256_path(path),
    }


def _source_locks(root: Path) -> list[dict[str, Any]]:
    locators = [
        SOURCE_LOCATOR,
        rating_bootstrap.SOURCE_LOCATOR,
        fast_uncertainty.SOURCE_LOCATOR,
        probability_v2.SOURCE_LOCATOR,
        probability_registry.SOURCE_LOCATOR,
        event_plan.SOURCE_LOCATOR,
        quote_attempt.SOURCE_LOCATOR,
        attempt_completion.SOURCE_LOCATOR,
        stopping_snapshot.SOURCE_LOCATOR,
        stopping_registry.SOURCE_LOCATOR,
        "lol_kills/v2/market/phase_two_collection_readiness_registry_v1.py",
        quote_v2.SOURCE_LOCATOR,
        quote_qualification.SOURCE_LOCATOR,
        quote_registry.SOURCE_LOCATOR,
        opening.SOURCE_LOCATOR,
        "lol_kills/v2/draft/terminal/future_prediction_ledger.py",
        "lol_kills/v2/market/betano_br_quote_adapter_v1.py",
        "lol_kills/v2/market/betano_br_quote_adapter_registry_v1.py",
        "lol_kills/v2/market/betano_terms_authority_v1.py",
        "lol_kills/v2/market/calibration_uncertainty_registry_v1.py",
        "lol_kills/bookmaker_quote_capture.py",
    ]
    return [_source_record(root, locator) for locator in locators]


def _dependencies(
    root: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    evaluation_binding = opening._load_evaluation(root, environment)
    calibration_binding = opening._load_calibration(root, environment)
    terms_digest = environment.get(
        "SCRYGLASS_PRIVATE_MATCH_WINNER_BOOKMAKER_TERMS_SHA256"
    )
    adapter_digest = environment.get(
        "SCRYGLASS_PRIVATE_MATCH_WINNER_QUOTE_ADAPTER_SHA256"
    )
    if not terms_digest or not adapter_digest:
        raise PhaseTwoCollectionReadinessError(
            "terms or quote-adapter external pin is missing"
        )
    terms = load_pinned_betano_terms_authority_v1(
        path=root / TERMS_REGISTRY_LOCATOR,
        external_sha256=terms_digest,
        root=root,
    )
    adapter = load_registered_betano_quote_adapter_v1(
        expected_registry_sha256=adapter_digest,
        root=root,
    )
    return {
        "phase_one_evaluation": evaluation_binding,
        "calibration_uncertainty_and_fast_parity": calibration_binding,
        "bookmaker_terms": {
            "raw_sha256": terms["receipt_raw_sha256"],
            "registry_id": terms["receipt"]["registry_id"],
            "settlement_contract_resolved": True,
        },
        "source_specific_quote_adapter": {
            "registry_sha256": adapter["registry_sha256"],
            "registry_id": adapter["registry_id"],
            "source_adapter_identity_authority": True,
        },
    }


def _contract() -> dict[str, Any]:
    signatures = {
        "build_event_rating_bootstrap": list(
            inspect.signature(
                rating_bootstrap.build_event_rating_bootstrap_v1
            ).parameters
        ),
        "build_fast_event_uncertainty": list(
            inspect.signature(
                fast_uncertainty.build_fast_event_uncertainty_v1
            ).parameters
        ),
        "build_event_probability": list(
            inspect.signature(probability_v2.build_event_probability_v2).parameters
        ),
        "capture_betano_quote": list(
            inspect.signature(
                quote_v2.capture_betano_map_winner_quote_v2
            ).parameters
        ),
        "build_phase_two_event_plan": list(
            inspect.signature(event_plan.build_phase_two_event_plan_v1).parameters
        ),
        "run_planned_quote_attempt": list(
            inspect.signature(quote_attempt.run_planned_quote_attempt_v1).parameters
        ),
        "build_attempt_completion": list(
            inspect.signature(
                attempt_completion.build_phase_two_attempt_completion_v1
            ).parameters
        ),
        "build_stopping_snapshot": list(
            inspect.signature(
                stopping_snapshot.build_phase_two_stopping_snapshot_v1
            ).parameters
        ),
        "qualify_betano_quote": list(
            inspect.signature(
                quote_qualification.build_betano_quote_qualification_v1
            ).parameters
        ),
        "expected_quote_registry_entries": list(
            inspect.signature(quote_registry.expected_entries).parameters
        ),
        "consume_phase_two_opening": list(
            inspect.signature(opening.consume_phase_two_opening).parameters
        ),
    }
    expected = {
        "build_event_rating_bootstrap": [
            "phase_one_result_locator",
            "rating_refit_locator",
            "workers",
            "root",
            "environment",
            "clock",
        ],
        "build_fast_event_uncertainty": [
            "phase_one_result_locator",
            "recalibration_artifact_locator",
            "target_prediction_locator",
            "rating_bootstrap_locator",
            "workers",
            "root",
            "environment",
            "clock",
        ],
        "build_event_probability": [
            "fast_uncertainty_locator",
            "root",
            "environment",
            "clock",
        ],
        "capture_betano_quote": [
            "event_plan_locator",
            "event_probability_locator",
            "request_url",
            "betano_event_id",
            "map_number",
            "participant_bindings",
            "fetcher",
            "root",
            "environment",
            "clock",
            "monotonic_ns",
        ],
        "build_phase_two_event_plan": [
            "event_probability_locator",
            "quote_output_locator",
            "qualification_output_locator",
            "failure_output_locator",
            "completion_output_locator",
            "root",
            "environment",
            "clock",
        ],
        "run_planned_quote_attempt": [
            "event_plan_locator",
            "request_url",
            "betano_event_id",
            "map_number",
            "participant_bindings",
            "fetcher",
            "root",
            "environment",
            "clock",
            "monotonic_ns",
        ],
        "build_attempt_completion": [
            "event_plan_locator",
            "map_start_locator",
            "root",
            "environment",
            "clock",
        ],
        "build_stopping_snapshot": [
            "completion_locators",
            "root",
            "environment",
            "clock",
        ],
        "qualify_betano_quote": [
            "quote_locator",
            "map_start_locator",
            "qualification_output_locator",
            "root",
            "environment",
            "clock",
        ],
        "expected_quote_registry_entries": [
            "qualification_locators",
            "root",
            "environment",
        ],
        "consume_phase_two_opening": ["root", "environment", "clock"],
    }
    if signatures != expected:
        raise PhaseTwoCollectionReadinessError(
            "phase-two builder signatures changed"
        )
    return {
        "schemas": {
            "event_rating_bootstrap": rating_bootstrap.SCHEMA_VERSION,
            "fast_event_uncertainty": fast_uncertainty.SCHEMA_VERSION,
            "event_probability": probability_v2.RECEIPT_SCHEMA_VERSION,
            "event_probability_registry": probability_registry.SCHEMA_VERSION,
            "phase_two_event_plan": event_plan.SCHEMA_VERSION,
            "phase_two_quote_attempt_failure": quote_attempt.FAILURE_SCHEMA_VERSION,
            "phase_two_attempt_completion": attempt_completion.SCHEMA_VERSION,
            "phase_two_stopping_snapshot": stopping_snapshot.SCHEMA_VERSION,
            "phase_two_stopping_snapshot_registry": stopping_registry.SCHEMA_VERSION,
            "betano_quote_transport": quote_v2.SCHEMA_VERSION,
            "betano_quote_qualification": quote_qualification.SCHEMA_VERSION,
            "betano_quote_registry": quote_registry.SCHEMA_VERSION,
            "phase_two_opening": opening.SCHEMA_VERSION,
        },
        "builder_parameters": signatures,
        "exact_2000_draw_slow_fast_parity_required": True,
        "fresh_post_validation_rating_refit_required": True,
        "fresh_source_roster_patch_and_point_exactly_bound": True,
        "expensive_rating_refits_completed_pre_draft": True,
        "terminal_draft_and_recalibration_draws_complete_before_probability": True,
        "percentile_interval_need_not_contain_plugin_point": True,
        "legacy_quote_bridge_is_transport_only": True,
        "probability_precedes_quote_request": True,
        "immutable_event_plan_precedes_quote_request": True,
        "failed_quote_attempts_remain_in_coverage_denominator": True,
        "planned_attempt_consumed_by_exactly_one_quote_or_typed_failure": True,
        "failure_receipts_persist_no_free_form_exception_or_credentials": True,
        "map_start_completion_finalizes_every_denominator_entry": True,
        "stopping_snapshot_uses_completion_denominator_not_quote_successes": True,
        "quote_response_precedes_actual_map_start": True,
        "quote_response_to_actual_map_start_seconds_minimum": (
            quote_qualification.MINIMUM_RESPONSE_TO_START_SECONDS
        ),
        "post_start_qualification_uses_outcome_free_map_start_authority": True,
        "qualification_candidate_grants_no_quote_identity_authority": True,
        "independent_external_quote_registry_required": True,
        "outcome_fields_forbidden_from_pre_event_receipts": True,
        "retrospective_backfill_qualifies": False,
        "collection_implementation_itself_authorizes_betting": False,
    }


def _count_json(root: Path, prefix: Path) -> int:
    path = root / prefix
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_dir():
        raise PhaseTwoCollectionReadinessError(
            "phase-two artifact directory is aliased"
        )
    return sum(1 for item in path.rglob("*.json") if item.is_file())


def _empty_state(root: Path) -> dict[str, Any]:
    state = {
        "opening_marker_present": (root / opening.MARKER_LOCATOR).is_file(),
        "event_probability_receipts": _count_json(
            root, Path(probability_v2.RECEIPT_PREFIX)
        ),
        "phase_two_event_plans": _count_json(
            root, Path(event_plan.OUTPUT_PREFIX)
        ),
        "phase_two_quote_failures": _count_json(
            root, Path(event_plan.FAILURE_PREFIX)
        ),
        "phase_two_attempt_completions": _count_json(
            root, Path(event_plan.COMPLETION_PREFIX)
        ),
        "phase_two_stopping_snapshots": _count_json(
            root, Path(stopping_snapshot.OUTPUT_PREFIX)
        ),
        "phase_two_stopping_registry_present": (
            root / stopping_registry.REGISTRY_LOCATOR
        ).is_file(),
        "betano_quote_bundles": _count_json(root, Path(quote_v2.OUTPUT_PREFIX)),
        "qualified_betano_quotes": _count_json(
            root, Path(quote_qualification.OUTPUT_PREFIX)
        ),
        "betano_quote_registry_present": (
            root / quote_registry.REGISTRY_LOCATOR
        ).is_file(),
        "phase_two_outcomes_present": False,
        "phase_two_outcomes_accessed": False,
        "phase_two_started": False,
    }
    if (
        state["opening_marker_present"]
        or state["event_probability_receipts"] != 0
        or state["phase_two_event_plans"] != 0
        or state["phase_two_quote_failures"] != 0
        or state["phase_two_attempt_completions"] != 0
        or state["phase_two_stopping_snapshots"] != 0
        or state["phase_two_stopping_registry_present"]
        or state["betano_quote_bundles"] != 0
        or state["qualified_betano_quotes"] != 0
        or state["betano_quote_registry_present"]
    ):
        raise PhaseTwoCollectionReadinessError(
            "phase-two outputs exist before implementation freeze"
        )
    return state


def build_phase_two_collection_readiness_v1(
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> dict[str, Any]:
    locked = clock()
    if not isinstance(locked, datetime) or locked.tzinfo is None:
        raise PhaseTwoCollectionReadinessError(
            "phase-two readiness clock must be timezone-aware"
        )
    locked = locked.astimezone(timezone.utc)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "result_state": RESULT_STATE,
        "locked_at_utc": locked.isoformat(),
        "clock_attestation": {
            "source": "system_utc_clock_sampled_inside_builder",
            "observed_wall_clock_utc": locked.isoformat(),
            "user_supplied_timestamp_allowed": False,
        },
        "dependencies": _dependencies(root, environment),
        "collection_contract": _contract(),
        "locked_empty_state": _empty_state(root),
        "source_locks": _source_locks(root),
        "decision_outputs": {
            "phase_two_opening_authority": None,
            "event_probability": None,
            "quote": None,
            "fair_odds": None,
            "expected_value": None,
            "recommendation": None,
            "stake": None,
        },
        "authority": dict(AUTHORITY),
        "claim_ceiling": CLAIM_CEILING,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload)
    return validate_phase_two_collection_readiness_v1(
        payload, root=root, environment=environment
    )


def validate_phase_two_collection_readiness_v1(
    payload: Mapping[str, Any],
    *,
    root: Path = ROOT,
    environment: Mapping[str, str] = os.environ,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise PhaseTwoCollectionReadinessError("readiness must be an object")
    value = dict(payload)
    if set(value) != {
        "schema_version",
        "result_state",
        "locked_at_utc",
        "clock_attestation",
        "dependencies",
        "collection_contract",
        "locked_empty_state",
        "source_locks",
        "decision_outputs",
        "authority",
        "claim_ceiling",
        "artifact_sha256",
    }:
        raise PhaseTwoCollectionReadinessError("readiness structure changed")
    unsigned = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if value.get("artifact_sha256") != _canonical_sha256(unsigned):
        raise PhaseTwoCollectionReadinessError("readiness hash changed")
    if (
        value.get("schema_version") != SCHEMA_VERSION
        or value.get("result_state") != RESULT_STATE
    ):
        raise PhaseTwoCollectionReadinessError("readiness identity changed")
    locked = _timestamp(value.get("locked_at_utc"), "locked_at_utc")
    if value.get("clock_attestation") != {
        "source": "system_utc_clock_sampled_inside_builder",
        "observed_wall_clock_utc": locked.isoformat(),
        "user_supplied_timestamp_allowed": False,
    }:
        raise PhaseTwoCollectionReadinessError("readiness clock changed")
    if value.get("dependencies") != _dependencies(root, environment):
        raise PhaseTwoCollectionReadinessError("readiness dependencies changed")
    if value.get("collection_contract") != _contract():
        raise PhaseTwoCollectionReadinessError("collection contract changed")
    if value.get("locked_empty_state") != {
        "opening_marker_present": False,
        "event_probability_receipts": 0,
        "phase_two_event_plans": 0,
        "phase_two_quote_failures": 0,
        "phase_two_attempt_completions": 0,
        "phase_two_stopping_snapshots": 0,
        "phase_two_stopping_registry_present": False,
        "betano_quote_bundles": 0,
        "qualified_betano_quotes": 0,
        "betano_quote_registry_present": False,
        "phase_two_outcomes_present": False,
        "phase_two_outcomes_accessed": False,
        "phase_two_started": False,
    }:
        raise PhaseTwoCollectionReadinessError("locked empty state changed")
    if value.get("source_locks") != _source_locks(root):
        raise PhaseTwoCollectionReadinessError("phase-two source lock changed")
    if value.get("decision_outputs") != {
        "phase_two_opening_authority": None,
        "event_probability": None,
        "quote": None,
        "fair_odds": None,
        "expected_value": None,
        "recommendation": None,
        "stake": None,
    }:
        raise PhaseTwoCollectionReadinessError("decision outputs changed")
    if value.get("authority") != AUTHORITY or value.get("claim_ceiling") != CLAIM_CEILING:
        raise PhaseTwoCollectionReadinessError("readiness exceeds authority")
    return value


def write_no_clobber(path: Path, payload: Mapping[str, Any]) -> str:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhaseTwoCollectionReadinessError(
            f"refusing to replace readiness: {path}"
        )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise PhaseTwoCollectionReadinessError(
                f"refusing to replace readiness: {path}"
            ) from exc
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(raw).hexdigest()


__all__ = [
    "DEFAULT_OUTPUT",
    "RESULT_STATE",
    "SCHEMA_VERSION",
    "SOURCE_LOCATOR",
    "PhaseTwoCollectionReadinessError",
    "build_phase_two_collection_readiness_v1",
    "validate_phase_two_collection_readiness_v1",
    "write_no_clobber",
]
